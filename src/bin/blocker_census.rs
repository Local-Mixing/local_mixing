//! Which gates can the frozen store never reach?
//!
//! The match RATE answers "of the windows fmix happened to sample, how many hit
//! the store" — a property of the sampler as much as of the circuit. The
//! question that actually matters for mixing is per-GATE: is there any window
//! containing this gate whose permutation the store holds? If there is, the
//! gate can be re-encoded and it is not a blocker. If there is none, no amount
//! of sampling will ever move it.
//!
//! That framing also disposes of a question that is NOT the right one: whether
//! a gate's own function is in the store's "vocabulary". The store is queried
//! by WINDOW, never by single gate (fmix's windows are [2,5]), so a gate whose
//! own permutation is absent is perfectly reachable as soon as it and a
//! neighbour jointly land on something stored. An earlier rationale in this
//! tree argued that a comp=0 width-2 conjunction is unreachable because it sits
//! outside the X-free g57 span; that argument is retracted (g57+CNOT generates
//! all of S8 on three wires), and this tool measures the thing it was trying to
//! reason about.
//!
//! METHOD. Sweep every contiguous window of size `min..=max`, canonicalise and
//! look it up through the SAME `db_replace` path fmix uses — not a
//! reimplementation, so the key construction and degree guard are identical.
//! Every gate of every window that matches is marked reachable. Cost is one
//! lookup per window, about (max-min+1)*N lookups for N gates, and each gate
//! inherits the verdict of the best window it belongs to.
//!
//! TWO KINDS OF BLOCKER, reported separately:
//!   store  — no window containing the gate has its permutation in the store.
//!   policy — `--db-ctrl-cap L` makes fmix EVADE any gate with more than L
//!            controls while building a window, so such gates are excluded
//!            from re-encoding by fmix's own sampling rule whatever the store
//!            holds. Pass `--ctrl-cap 0` to measure the store alone.
//!
//! Usage: blocker_census --g <circuit> [--g-format mpmct1] [--min-window 2]
//!        [--max-window 5] [--ctrl-cap 2] [--db-max-degree 9]
//!        [--wire-terms 1024] [--total-terms 2048] [--seed 1]
//! Requires FROZEN_DB_DIR, exactly as fmix does.

use clap::Parser;
use local_mixing::postmix::db_replace::{db_replace_with, DbMode, DegreeGuard};
use local_mixing::postmix::format::{read_g57_file, read_mpmct};
use local_mixing::postmix::xgate::{max_wire, XGate};
use local_mixing::postmix::xpoly::XPolyBudget;
use local_mixing::replace::frozen::FrozenDb;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

#[derive(Parser, Debug)]
#[command(name = "blocker_census")]
struct Args {
    #[arg(long)]
    g: String,
    #[arg(long, default_value = "mpmct1")]
    g_format: String,
    #[arg(long, default_value_t = 2)]
    min_window: usize,
    #[arg(long, default_value_t = 5)]
    max_window: usize,
    /// Gates with more than this many controls are excluded from windows, as
    /// fmix's --db-ctrl-cap does. 0 = no policy filter (store-only census).
    #[arg(long, default_value_t = 2)]
    ctrl_cap: usize,
    #[arg(long, default_value_t = 9)]
    db_max_degree: usize,
    #[arg(long, default_value_t = 4)]
    db_degree_probes: usize,
    #[arg(long, default_value_t = 1024)]
    wire_terms: usize,
    #[arg(long, default_value_t = 2048)]
    total_terms: usize,
    #[arg(long, default_value_t = 1)]
    seed: u64,
}

/// The shape buckets the directive cares about, so the blocker count can be
/// attributed rather than just totalled.
fn shape(g: &XGate) -> &'static str {
    match g.ctrls.len() {
        0 => "bare X",
        1 => {
            if g.comp {
                "1-ctrl comp"
            } else {
                "CNOT"
            }
        }
        2 => {
            if g.comp && g.ctrls[0].1 != g.ctrls[1].1 {
                "g57"
            } else if g.comp {
                "2-ctrl comp"
            } else {
                "2-ctrl plain"
            }
        }
        _ => "wide (>2)",
    }
}

fn main() {
    let args = Args::parse();
    let gates: Vec<XGate> = match args.g_format.as_str() {
        "g57" => read_g57_file(&args.g).expect("read g (g57)"),
        "mpmct1" => read_mpmct(&args.g).expect("read g (mpmct1)").0,
        o => panic!("unknown --g-format {o}"),
    };
    let n = gates.len();
    let num_wires = max_wire(&gates) as usize + 1;
    let db = FrozenDb::from_env();
    let budget = XPolyBudget {
        max_mul_terms: args.wire_terms * 4,
        max_poly_terms: args.wire_terms,
        max_total_terms: args.total_terms,
    };
    let guard = DegreeGuard {
        max_degree: args.db_max_degree,
        probes: args.db_degree_probes,
    };

    println!(
        "[blocker] {n} gates, {num_wires} wires, windows {}..={}, ctrl_cap {} ({})",
        args.min_window,
        args.max_window,
        args.ctrl_cap,
        if args.ctrl_cap == 0 {
            "store-only census"
        } else {
            "fmix's sampling policy applied"
        }
    );

    let mut reachable = vec![false; n];
    // A gate the policy excludes is never placed in any window, so it is a
    // blocker regardless of the store; record it separately rather than
    // letting it silently inflate the store-blocker count.
    let policy_out: Vec<bool> = gates
        .iter()
        .map(|g| args.ctrl_cap > 0 && g.ctrls.len() > args.ctrl_cap)
        .collect();

    let mut windows_tried = 0usize;
    let mut windows_hit = 0usize;
    let mut degree_skips = 0usize;
    // Canonicalization dominates -- it runs BEFORE the store lookup, so making
    // the lookup cheap did not help. It is also per-window independent, so
    // sweep each size in parallel and merge afterwards. The sequential version
    // could skip a window whose gates were all already reachable; that
    // optimization is dropped within a size (it would need shared mutable
    // state) and kept BETWEEN sizes, which is where most of it was anyway.
    for w in args.min_window..=args.max_window.min(n) {
        let starts: Vec<usize> = (0..=(n - w))
            .filter(|&i| !(i..i + w).any(|j| policy_out[j]))
            .filter(|&i| !(i..i + w).all(|j| reachable[j]))
            .collect();
        let out: Vec<(bool, bool)> = starts
            .par_iter()
            .map(|&i| {
                // Per-window RNG: the degree guard probes randomly, and a
                // shared stream would make the result depend on scheduling.
                let mut rng = StdRng::seed_from_u64(args.seed ^ ((w as u64) << 40) ^ i as u64);
                let mut hit = false;
                let r = db_replace_with(
                    &gates[i..i + w],
                    num_wires,
                    budget,
                    DbMode::SizeAgnostic,
                    guard,
                    &mut rng,
                    |key, _curated| {
                        if db.get_regular(key).is_some() {
                            hit = true;
                        }
                        None
                    },
                );
                (hit, r.degree_skipped)
            })
            .collect();
        windows_tried += starts.len();
        for (idx, &(hit, skipped)) in out.iter().enumerate() {
            if skipped {
                degree_skips += 1;
            }
            if hit {
                windows_hit += 1;
                let i = starts[idx];
                for j in i..i + w {
                    reachable[j] = true;
                }
            }
        }
        let done = reachable.iter().filter(|&&b| b).count();
        println!(
            "  after windows of size {w}: {done}/{n} gates reachable ({:.1}%), {windows_tried} lookups, {windows_hit} hits",
            100.0 * done as f64 / n as f64
        );
    }

    let reach = reachable.iter().filter(|&&b| b).count();
    let pol = policy_out.iter().filter(|&&b| b).count();
    let store_blocked = n - reach - pol;
    println!("\n[blocker] RESULT over {n} gates");
    println!(
        "  reachable (some window hits the store) {:>9}  {:>6.2}%",
        reach,
        100.0 * reach as f64 / n as f64
    );
    println!(
        "  policy-blocked (>{} controls, evaded)  {:>9}  {:>6.2}%",
        args.ctrl_cap,
        pol,
        100.0 * pol as f64 / n as f64
    );
    println!(
        "  store-blocked (no window ever hits)    {:>9}  {:>6.2}%",
        store_blocked,
        100.0 * store_blocked as f64 / n as f64
    );
    println!("  ({windows_tried} lookups, {windows_hit} hits, {degree_skips} degree-guard skips)");

    // Attribution by gate shape: which shapes are the blockers?
    use std::collections::BTreeMap;
    let mut tot: BTreeMap<&str, (usize, usize, usize)> = BTreeMap::new();
    for (i, g) in gates.iter().enumerate() {
        let e = tot.entry(shape(g)).or_insert((0, 0, 0));
        e.0 += 1;
        if reachable[i] {
            e.1 += 1;
        } else if policy_out[i] {
            e.2 += 1;
        }
    }
    println!("\n  shape            count   reachable   policy-blocked   store-blocked");
    for (k, (c, r, p)) in tot {
        println!(
            "  {:<14} {:>7} {:>10} ({:>5.1}%) {:>10} {:>13}",
            k,
            c,
            r,
            100.0 * r as f64 / c as f64,
            p,
            c - r - p
        );
    }
}
