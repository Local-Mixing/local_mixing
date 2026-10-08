//! RC's manual sequential mixing walk (2026-10-04): one left-to-right pass
//! that re-spells the WHOLE circuit out of curated-store vocabulary.
//!
//! From the current position, find the LONGEST contiguous window with a
//! curated match (longest window first, shrinking), replace it with a random
//! candidate of the LARGEST gate count, then continue from the replacement's
//! second-to-last gate (the 2-gate overlap re-mixes every seam together with
//! the following original gates). The point is the categorical/syntactic
//! guarantee: no original gate is ever emitted verbatim -- every original
//! gate is consumed by some replaced window.
//!
//! Termination guard (deviation from the literal spec, noted in output):
//! a window wholly inside previously-replaced material (the overlap tail)
//! consumes no original gate and can churn forever, so candidate windows
//! must reach at least one not-yet-consumed original gate. When only the
//! tail is left unmatched, its front gate (new material) is finalized as-is.
//!
//! A single original gate with no curated match at any window length would
//! survive verbatim; `--fallback-regular` sends exactly those through the
//! regular store instead (always stocked for single gates). Default off.
//!
//! ```text
//! FROZEN_DB_DIR=... FROZEN_CURATED_DIR=<b' dir> FROZEN_CURATED_KEY=circuit \
//!   greedy_walk IN.mpmct1 OUT.mpmct1 [--max-window 11] [--seed S] \
//!     [--fallback-regular] [--trace N]
//! ```

use local_mixing::canonicalization::xgate::XPolyBudget;
use local_mixing::circuit::formats::{read_mpmct, write_mpmct};
use local_mixing::circuit::xgate::{XGate, eval_lanes};
use local_mixing::database::frozen::FrozenDb;
use local_mixing::stages::db_mixing::replacement::{
    QC_MAX_BUCKET_BYTES, QcCandidateLimits, qc_candidates_with,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BTreeMap;
use std::error::Error;

type AnyResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

fn main() {
    if let Err(error) = run() {
        eprintln!("greedy_walk: {error}");
        std::process::exit(1);
    }
}

fn gate_sig(g: &XGate) -> (u16, bool, Vec<(u16, bool)>) {
    (g.target, g.comp, g.ctrls.iter().copied().collect())
}

fn simulate_equal(a: &[XGate], b: &[XGate], n: usize, rng: &mut StdRng, batches: usize) -> bool {
    for _ in 0..batches {
        let s0: Vec<u64> = (0..n).map(|_| rng.random()).collect();
        let mut s1 = s0.clone();
        let mut s2 = s0;
        eval_lanes(a.iter(), &mut s1);
        eval_lanes(b.iter(), &mut s2);
        if s1 != s2 {
            return false;
        }
    }
    true
}

fn run() -> AnyResult<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: greedy_walk IN.mpmct1 OUT.mpmct1 [--max-window N] [--seed S] \
             [--fallback-regular] [--trace N]"
        );
        std::process::exit(2);
    }
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse::<u64>().ok())
    };
    let max_window = flag("--max-window").unwrap_or(11) as usize;
    let seed = flag("--seed").unwrap_or_else(|| rand::rng().random());
    let fallback_regular = args.iter().any(|a| a == "--fallback-regular");
    // Reject candidates that reuse ANY gate of the original circuit verbatim:
    // b' serves bracket-dressed spellings (window + conjugation pairs) that the
    // longest-pick rule otherwise prefers, passing original gates through
    // untouched -- the opposite of the walk's point.
    let no_verbatim = args.iter().any(|a| a == "--no-verbatim");
    let trace = flag("--trace").unwrap_or(0) as usize;
    let mut rng = StdRng::seed_from_u64(seed);

    match std::env::var("FROZEN_CURATED_KEY").as_deref() {
        Ok("circuit") => {}
        other => {
            return Err(format!(
                "FROZEN_CURATED_KEY must be 'circuit' for the v3 true-friend stores (got {other:?})"
            )
            .into());
        }
    }

    let (gates, num_wires) = read_mpmct(&args[1]).map_err(|e| format!("read {}: {e}", args[1]))?;
    let regular = std::env::var("FROZEN_DB_DIR").map_err(|_| "FROZEN_DB_DIR unset")?;
    let curated = std::env::var("FROZEN_CURATED_DIR").map_err(|_| "FROZEN_CURATED_DIR unset")?;
    let db = FrozenDb::open(&regular, Some(&curated));

    let budget = XPolyBudget {
        max_mul_terms: 1 << 16,
        max_poly_terms: 4096,
        max_total_terms: 16384,
    };
    let limits = QcCandidateLimits {
        max_candidates: 4096,
        max_gates: 32,
        max_support: 24,
    };
    let cap = limits.max_candidates.saturating_add(1);

    let orig_set: std::collections::BTreeSet<(u16, bool, Vec<(u16, bool)>)> =
        gates.iter().map(gate_sig).collect();
    let clean = |c: &[XGate]| !no_verbatim || c.iter().all(|g| !orig_set.contains(&gate_sig(g)));

    let mut done: Vec<XGate> = Vec::with_capacity(gates.len() * 4);
    let mut tail: Vec<XGate> = Vec::new(); // replacement overlap, pending re-process
    let mut oi = 0usize; // next unconsumed original gate
    let mut survivors = 0usize;
    let mut fallback_used = 0usize;
    let mut tail_finalized = 0usize;
    let mut steps = 0usize;
    let mut window_hist: BTreeMap<usize, usize> = BTreeMap::new();
    let mut repl_hist: BTreeMap<usize, usize> = BTreeMap::new();
    let mut short_repl = 0usize; // replacements with k < 3 (no 2-gate overlap possible)
    let mut pool_sizes = 0usize;
    let mut lookup_count = 0usize;
    let mut dirty_pools = 0usize; // non-empty pools emptied by the no-verbatim filter

    while oi < gates.len() {
        steps += 1;
        let avail = gates.len() - oi;
        let lo = tail.len() + 1; // must reach >=1 original gate
        let hi = (tail.len() + avail).min(max_window);
        let mut matched = false;
        if lo <= hi {
            let mut window: Vec<XGate> = tail.clone();
            window.extend_from_slice(&gates[oi..oi + (hi - tail.len())]);
            // Pass 1: curated only. Pass 2 (--fallback-regular): regular store,
            // same longest-window descent, for material the curated store can
            // only dress with its own gates (the no-verbatim common case).
            'pass: for use_curated in [true, false] {
                if !use_curated && !fallback_regular {
                    break;
                }
                for l in (lo..=hi).rev() {
                    lookup_count += 1;
                    let res = qc_candidates_with(
                        &window[..l],
                        num_wires,
                        budget,
                        limits,
                        &mut rng,
                        |key, is_curated| {
                            if is_curated != use_curated {
                                None
                            } else if is_curated {
                                db.get_curated_qc(key, cap, QC_MAX_BUCKET_BYTES)
                                    .ok()
                                    .flatten()
                            } else {
                                db.get_regular_qc(key, cap, QC_MAX_BUCKET_BYTES)
                                    .ok()
                                    .flatten()
                            }
                        },
                    );
                    let cands: Vec<_> = res.candidates.iter().filter(|c| clean(&c.gates)).collect();
                    if cands.is_empty() {
                        if !res.candidates.is_empty() {
                            dirty_pools += 1;
                        }
                        continue;
                    }
                    let maxlen = cands.iter().map(|c| c.gates.len()).max().unwrap();
                    let pool: Vec<_> = cands.iter().filter(|c| c.gates.len() == maxlen).collect();
                    let pick = pool[rng.random_range(0..pool.len())].gates.clone();
                    if steps <= trace {
                        eprintln!(
                            "[step {steps}] oi={oi} tail={} L={l} -> k={} (pool {} of {}, {})",
                            tail.len(),
                            pick.len(),
                            pool.len(),
                            cands.len(),
                            if use_curated { "curated" } else { "regular" }
                        );
                    }
                    pool_sizes += cands.len();
                    oi += l - tail.len();
                    let f = pick.len().saturating_sub(2);
                    if pick.len() < 3 {
                        short_repl += 1;
                    }
                    done.extend_from_slice(&pick[..f]);
                    tail = pick[f..].to_vec();
                    *window_hist.entry(l).or_default() += 1;
                    *repl_hist.entry(pick.len()).or_default() += 1;
                    if !use_curated {
                        fallback_used += 1;
                    }
                    matched = true;
                    break 'pass;
                }
            }
        }
        if matched {
            continue;
        }
        if !tail.is_empty() {
            // No window reaching original material matches: slide one NEW gate out.
            done.push(tail.remove(0));
            tail_finalized += 1;
            continue;
        }
        done.push(gates[oi].clone());
        oi += 1;
        survivors += 1;
    }
    done.extend(tail.drain(..));

    // Syntactic overlap: final gates equal (as literal gates) to some original.
    let mut orig_count: BTreeMap<(u16, bool, Vec<(u16, bool)>), usize> = BTreeMap::new();
    for g in &gates {
        *orig_count.entry(gate_sig(g)).or_default() += 1;
    }
    let mut verbatim = 0usize;
    for g in &done {
        if let Some(c) = orig_count.get_mut(&gate_sig(g)) {
            if *c > 0 {
                *c -= 1;
                verbatim += 1;
            }
        }
    }

    let equivalent = simulate_equal(&gates, &done, num_wires, &mut rng, 64);

    write_mpmct(&args[2], &done, num_wires).map_err(|e| format!("write {}: {e}", args[2]))?;
    let matches: usize = window_hist.values().sum();
    let info = format!(
        "greedy_walk in={} curated={} seed={} max_window={} fallback_regular={} no_verbatim={}\n\
         gates {}->{} steps={} matches={} lookups={} survivors={} fallback_used={} \
         tail_finalized={} short_repl={} dirty_pools={}\n\
         window_hist={:?}\nrepl_hist={:?}\nmean_pool={:.2}\n\
         verbatim_gate_overlap={} ({:.1}% of original)\nequivalent_4096_samples={}\n",
        args[1],
        curated,
        seed,
        max_window,
        fallback_regular,
        no_verbatim,
        gates.len(),
        done.len(),
        steps,
        matches,
        lookup_count,
        survivors,
        fallback_used,
        tail_finalized,
        short_repl,
        dirty_pools,
        window_hist,
        repl_hist,
        pool_sizes as f64 / matches.max(1) as f64,
        verbatim,
        100.0 * verbatim as f64 / gates.len() as f64,
        equivalent
    );
    std::fs::write(
        format!("{}.info", args[2].trim_end_matches(".mpmct1")),
        &info,
    )?;
    print!("{info}");
    if !equivalent {
        return Err("output NOT equivalent to input (simulation mismatch)".into());
    }
    Ok(())
}
