//! Store-only equivalence-preserving rewriter for g57 circuits.
//!
//! The point of this tool is what it does NOT use: no `Mixer`, no crossings,
//! splits, merges or floats. Every edit is a splice of one contiguous window
//! for a stored friend of the same canonical polynomial form, drawn from the
//! frozen replacement store alone. A C/C' pair built this way differs from C
//! only through the DB's identity relation, so it is independent of whichever
//! obfuscator is under test -- which is exactly what a control pair needs.
//!
//! It is a DIVERSIFIER, not a compressor: among the stored friends of a window
//! it picks one UNIFORMLY AT RANDOM (excluding the window itself), where
//! `find_any_replacement_db` prefers the shortest. Gate count therefore drifts
//! toward the mean friend length rather than down.
//!
//! `--target-gates N` adds a TERMINAL constraint on top of that, because the
//! consumer (`gen_sandwich_gadget` via GSS_SOURCE_C) asserts |C| == m_C and
//! sizes the whole pipeline from it: C and C' must come out at the same gate
//! count or the size difference is itself a signal. The walk stays uniform
//! over all friends until the circuit first reaches N; from then on the draw
//! is restricted to friends that move the count strictly closer to N,
//! preferring an exact landing, and the run stops the moment |C| == N. The
//! restriction LATCHES rather than tracking |C| < N: convergence from above
//! can undershoot by a gate or two, and reverting to the unrestricted rule
//! there would let a single long friend jump the count far above N again.
//!
//! The lookup below replicates `pairs.rs::frozen_lookup` rather than calling
//! it: that function is private and its helpers (`cached_db_get`,
//! `LOOKUP_NS_*`, `min_dir_lookup_mode`) are `pub(crate)`, so a separate bin
//! crate cannot reach them and nothing in `src/` had to be widened for this
//! experiment. The replica probes curated-forward, then regular forward and
//! reverse -- the MIN_DIR_LOOKUP=Legacy order, which replace.rs documents as
//! exactly equivalent to the min-direction default (the shard DBs are keyed by
//! min(canon_fwd, canon_rev), so probing both directions is a superset). It
//! also skips the process-wide lookup cache, so probe cost is the raw store's.
//!
//! Example:
//!   FROZEN_DB_DIR=... FROZEN_CURATED_DIR=... id_rewrite \
//!     --input c.g57 --n 64 --output cprime.g57 --passes 10000 --seed 1
use clap::Parser;
use local_mixing::circuit::{CircuitSeq, Permutation};
use local_mixing::db_mixing::frozen::FrozenDb;
use rand::prelude::SliceRandom;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

/// Sampled inputs per equivalence check. The spec floor for this tool; a
/// non-equivalent splice shows up on the first differing input, so this is
/// about catching rare wire-space bugs, not about a tight error bound.
const VERIFY_INPUTS: usize = 65536;

#[derive(Parser, Debug)]
#[command(name = "id_rewrite")]
struct Args {
    /// Source circuit, g57 (base-83, `;`-separated 3-wire gates)
    #[arg(long)]
    input: String,
    /// Wire count of the circuit
    #[arg(long)]
    n: usize,
    /// Output circuit, g57
    #[arg(long)]
    output: String,
    /// Number of splice ATTEMPTS (not splices: most windows miss the store).
    /// With --target-gates this is a budget, not a schedule: the run ends as
    /// soon as the target is hit.
    #[arg(long, default_value_t = 1000)]
    passes: usize,
    /// Land on EXACTLY this many gates and stop (0 = off, free-running). Two
    /// runs from the same source at the same target are then size-matched,
    /// which is what a C/C' pair needs. Missing the target is a hard error:
    /// an off-target circuit is never written.
    #[arg(long, default_value_t = 0)]
    target_gates: usize,
    /// RNG seed. The whole run -- window draws, friend draws, fresh-wire
    /// choices -- is reproducible from it.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Smallest window (gates) spliced at once
    #[arg(long, default_value_t = 1)]
    min_window: usize,
    /// Largest window (gates) spliced at once
    #[arg(long, default_value_t = 4)]
    max_window: usize,
    /// Probe the curated store (FROZEN_CURATED_DIR). Disable with --no-curated.
    #[arg(long, default_value_t = true)]
    curated: bool,
    /// Turn --curated off.
    #[arg(long, default_value_t = false)]
    no_curated: bool,
    /// Probe the regular store (FROZEN_DB_DIR). Disable with --no-regular.
    #[arg(long, default_value_t = true)]
    regular: bool,
    /// Turn --regular off.
    #[arg(long, default_value_t = false)]
    no_regular: bool,
    /// Re-verify against the original every N attempts (0 = only at the end).
    /// Each check costs a full 65536-input evaluation of both circuits.
    #[arg(long, default_value_t = 0)]
    verify_every: usize,
    /// Optional JSON summary path
    #[arg(long)]
    report: Option<String>,
}

/// One store answer: the raw value blob plus everything needed to carry a
/// friend back into the circuit's own wire space.
struct Hit {
    value: Vec<u8>,
    order: Permutation,
    is_reversed: bool,
    used: Vec<u16>,
}

fn frozen_lookup(
    sub: &CircuitSeq,
    db: &FrozenDb,
    use_curated: bool,
    use_regular: bool,
) -> Option<Hit> {
    // `None` here is the canonicalization skip (oversized window, monomial cap,
    // Rule-L budget) -- indistinguishable from a store miss for our purposes.
    let (fwd_key, fwd_order, used) = sub.canonicalize_polys_single_hashed(false);
    let fwd_key = fwd_key?;

    if use_curated && db.has_curated() {
        if let Some(value) = db.get_curated(&fwd_key) {
            return Some(Hit {
                value,
                order: fwd_order,
                is_reversed: false,
                used,
            });
        }
    }
    if use_regular {
        if let Some(value) = db.get_regular(&fwd_key) {
            return Some(Hit {
                value,
                order: fwd_order,
                is_reversed: false,
                used,
            });
        }
        let (rev_key, rev_order, _) = sub.canonicalize_polys_single_hashed(true);
        let rev_key = rev_key?;
        if let Some(value) = db.get_regular(&rev_key) {
            return Some(Hit {
                value,
                order: rev_order,
                is_reversed: true,
                used,
            });
        }
    }
    None
}

enum Rewrite {
    /// No store answer at all.
    Miss,
    /// The store answered but held nothing usable: no friend other than the
    /// window itself, or (in Toward mode) none of the right length.
    NoAlternative,
    Replaced(Vec<[u16; 3]>),
}

/// Which stored friends a draw is allowed to consider.
#[derive(Clone, Copy)]
enum Select {
    /// The default: any friend but the window itself. This is what makes the
    /// tool a diversifier.
    Any,
    /// Terminal constraint: only friends whose length moves the circuit
    /// STRICTLY closer to `target`, so the walk cannot stall in place or
    /// oscillate. An exact landing, when the store offers one, wins outright.
    Toward { cur: usize, target: usize },
}

/// Carry one decoded friend from canonical wire space back to the circuit's,
/// mirroring `find_any_replacement_db`'s rewire/unrewire dance. `None` when the
/// friend needs more wires than the circuit has spare.
fn map_back(mut repl: CircuitSeq, hit: &Hit, n: usize, rng: &mut StdRng) -> Option<Vec<[u16; 3]>> {
    if hit.is_reversed {
        repl.gates.reverse();
    }

    // A friend may touch canonical wires the window itself never did; extend
    // the recorded order with identity entries so it stays a permutation.
    let repl_n = repl.max_wire() + 1;
    let mut order_data = hit.order.data.clone();
    while order_data.len() < repl_n {
        let i = order_data.len();
        order_data.push(i);
    }
    repl.rewire(
        &Permutation { data: order_data },
        std::cmp::max(repl_n, hit.order.data.len()),
    );

    // Same story one level out: the friend's extra canonical wires need real
    // circuit wires, drawn at random from those the window does not touch.
    let repl_n_b = repl.max_wire() + 1;
    let mut used_ext = hit.used.clone();
    if used_ext.len() < repl_n_b {
        let mut used_mask = vec![false; n];
        for &w in used_ext.iter() {
            if (w as usize) < n {
                used_mask[w as usize] = true;
            }
        }
        let mut available: Vec<u16> = (0..n as u16).filter(|&w| !used_mask[w as usize]).collect();
        available.shuffle(rng);
        let mut avail = available.into_iter();
        while used_ext.len() < repl_n_b {
            avail.next().map(|w| used_ext.push(w))?;
        }
    }

    Some(CircuitSeq::unrewire_subcircuit(&repl, &used_ext).gates)
}

/// Uniform-over-friends replacement for `gates`. Under `Select::Any` this is
/// the one deliberate deviation from `find_any_replacement_db`, which filters
/// to the shortest friends first: length preference would make this a
/// compressor and collapse the diversity the control pair is for.
fn random_equivalent(
    gates: &[[u16; 3]],
    n: usize,
    db: &FrozenDb,
    use_curated: bool,
    use_regular: bool,
    select: Select,
    rng: &mut StdRng,
) -> Rewrite {
    let sub = CircuitSeq {
        gates: gates.to_vec(),
    };
    let Some(hit) = frozen_lookup(&sub, db, use_curated, use_regular) else {
        return Rewrite::Miss;
    };

    // Record friend spans without decoding them: a curated value can hold
    // hundreds of thousands of friends and a uniform draw needs exactly one.
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut pos = 0;
    while pos < hit.value.len() {
        let len = hit.value[pos] as usize;
        pos += 1;
        if pos + len > hit.value.len() {
            break;
        }
        spans.push((pos, len));
        pos += len;
    }

    // Length filtering happens on the span table, before any decode: a span is
    // 3 bytes per gate, and `map_back` only rewires, so the spliced gate count
    // is known exactly from `len / 3`.
    if let Select::Toward { cur, target } = select {
        let dist = cur.abs_diff(target);
        let closer = |span_len: usize| {
            let after = cur - gates.len() + span_len / 3;
            after.abs_diff(target) < dist
        };
        // An exact landing ends the run, so take it over any merely-closer
        // friend rather than leaving the last gate or two to another pass.
        if spans
            .iter()
            .any(|&(_, l)| cur - gates.len() + l / 3 == target)
        {
            spans.retain(|&(_, l)| cur - gates.len() + l / 3 == target);
        } else {
            spans.retain(|&(_, l)| closer(l));
        }
    }

    // Draw uniformly, discard-and-redraw on the window itself (at most one
    // friend can equal it) or on a friend that will not fit. Removing a
    // rejected index keeps the surviving draw uniform over what is left.
    while !spans.is_empty() {
        let idx = rng.random_range(0..spans.len());
        let (p, l) = spans.swap_remove(idx);
        let repl = CircuitSeq::from_blob(&hit.value[p..p + l]);
        if let Some(mapped) = map_back(repl, &hit, n, rng) {
            if mapped != gates {
                return Rewrite::Replaced(mapped);
            }
        }
    }
    Rewrite::NoAlternative
}

fn verify(orig: &CircuitSeq, cur: &CircuitSeq, n: usize, label: &str) {
    if let Err(e) = cur.probably_equal(orig, n, VERIFY_INPUTS) {
        eprintln!("[id_rewrite] EQUIVALENCE BROKEN at {label}: {e}");
        std::process::exit(1);
    }
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn main() {
    let mut args = Args::parse();
    if args.no_curated {
        args.curated = false;
    }
    if args.no_regular {
        args.regular = false;
    }
    if !args.curated && !args.regular {
        eprintln!("[id_rewrite] both stores disabled; nothing could ever be spliced");
        std::process::exit(2);
    }
    if args.min_window == 0 || args.min_window > args.max_window {
        eprintln!(
            "[id_rewrite] bad window range {}..={}",
            args.min_window, args.max_window
        );
        std::process::exit(2);
    }

    let raw = std::fs::read(&args.input).expect("read input circuit");
    let orig = CircuitSeq::from_bytes(&raw);
    if orig.gates.is_empty() {
        eprintln!("[id_rewrite] empty input circuit");
        std::process::exit(2);
    }
    if orig.max_wire() >= args.n {
        eprintln!(
            "[id_rewrite] --n {} but input touches wire {}",
            args.n,
            orig.max_wire()
        );
        std::process::exit(2);
    }
    let gates_before = orig.gates.len();

    let db = FrozenDb::from_env();
    if args.curated && !db.has_curated() {
        eprintln!("[id_rewrite] WARNING: --curated but FROZEN_CURATED_DIR is unset; regular only");
    }

    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut cur = orig.clone();
    let (mut attempts, mut hits, mut splices, mut rejected) = (0u64, 0u64, 0u64, 0u64);
    // Latched, not recomputed per pass -- see the module header.
    let mut converging = false;
    let t0 = Instant::now();

    for attempt in 0..args.passes {
        if args.target_gates > 0 {
            if cur.gates.len() == args.target_gates {
                break;
            }
            converging |= cur.gates.len() >= args.target_gates;
        }
        let hi = args.max_window.min(cur.gates.len());
        if hi < args.min_window {
            break;
        }
        attempts += 1;
        let select = if converging {
            Select::Toward {
                cur: cur.gates.len(),
                target: args.target_gates,
            }
        } else {
            Select::Any
        };
        let w = rng.random_range(args.min_window..=hi);
        let at = rng.random_range(0..=cur.gates.len() - w);
        match random_equivalent(
            &cur.gates[at..at + w],
            args.n,
            &db,
            args.curated,
            args.regular,
            select,
            &mut rng,
        ) {
            Rewrite::Miss => {}
            Rewrite::NoAlternative => {
                hits += 1;
                rejected += 1;
            }
            Rewrite::Replaced(repl) => {
                hits += 1;
                splices += 1;
                cur.gates.splice(at..at + w, repl);
            }
        }
        if args.verify_every > 0 && (attempt + 1) % args.verify_every == 0 {
            verify(&orig, &cur, args.n, &format!("attempt {}", attempt + 1));
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let gates_after = cur.gates.len();

    // Verify before the target check either way: a size miss and a broken
    // splice are different bugs, and the equivalence verdict is the tool's
    // contract regardless of whether the circuit is usable downstream.
    verify(&orig, &cur, args.n, "final");
    if args.target_gates > 0 && gates_after != args.target_gates {
        eprintln!(
            "[id_rewrite] TARGET MISSED: wanted exactly {} gates, ended at {gates_after} \
             (off by {}) after {attempts} of {} passes. No output written -- a C/C' pair \
             must be size-matched. Retry with more --passes or a different --seed.",
            args.target_gates,
            gates_after.abs_diff(args.target_gates),
            args.passes
        );
        std::process::exit(1);
    }
    std::fs::write(&args.output, cur.repr()).expect("write output circuit");

    let target_note = if args.target_gates > 0 {
        format!(" (target {} hit at pass {attempts})", args.target_gates)
    } else {
        String::new()
    };
    println!(
        "[id_rewrite] {gates_before} -> {gates_after} gates{target_note}; {attempts} attempts, \
         {hits} hits, {splices} splices; equivalence VERIFIED ({VERIFY_INPUTS} inputs)"
    );
    println!(
        "[id_rewrite] rejected {rejected} (hit, no alternative); {elapsed:.1}s \
         ({:.0} attempts/s)",
        attempts as f64 / elapsed.max(1e-9)
    );

    if let Some(path) = &args.report {
        let json = format!(
            "{{\n  \"input\": \"{}\",\n  \"output\": \"{}\",\n  \"n\": {},\n  \"seed\": {},\n  \
             \"passes\": {},\n  \"target_gates\": {},\n  \"min_window\": {},\n  \
             \"max_window\": {},\n  \
             \"curated\": {},\n  \"regular\": {},\n  \"attempts\": {},\n  \"hits\": {},\n  \
             \"splices\": {},\n  \"rejected\": {},\n  \"gates_before\": {},\n  \
             \"gates_after\": {},\n  \"elapsed_secs\": {:.3},\n  \"verify_inputs\": {},\n  \
             \"verified\": true\n}}\n",
            json_escape(&args.input),
            json_escape(&args.output),
            args.n,
            args.seed,
            args.passes,
            args.target_gates,
            args.min_window,
            args.max_window,
            args.curated,
            args.regular,
            attempts,
            hits,
            splices,
            rejected,
            gates_before,
            gates_after,
            elapsed,
            VERIFY_INPUTS,
        );
        std::fs::write(path, json).expect("write report");
    }
}
