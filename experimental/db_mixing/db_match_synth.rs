//! Synthetic match-probability grid against a frozen store.
//!
//! For each cell (m total gates, k non-g57 gates, c controls per non-g57
//! gate, W-wire pool) sample N random windows, key each by its function and
//! look it up in the store, tally hit rate and mean equivalent count. This
//! measures the STRUCTURAL storability of a window shape independent of any
//! fmix walk; a real circuit's match rate is then (shape census) x (this
//! grid). Wire-sharing density matters, so the pool size W is an explicit
//! axis: dense = m+2 wires, sparse = 3m wires.
//!
//! Non-g57 gates are random conjunction gates (random polarities and comp
//! bit) on distinct wires; for c=2 the exact g57 pattern (comp with one
//! positive and one negative control) is resampled away. k=0 rows are the
//! pure-g57 sanity anchor.
//!
//! Usage:
//!   db_match_synth <store_dir> [--samples N] [--seed S] [--dense-only|--sparse-only]
//! Env: FROZEN_FILTER=1 recommended; CANON_RULE_L_BRANCH_CAP guards canon.

use local_mixing::db_mixing::db_replace::{DbMode, DegreeGuard, db_replace};
use local_mixing::circuit::xgate::XGate;
use local_mixing::engine::xpoly::XPolyBudget;
use local_mixing::db_mixing::frozen::FrozenDb;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn distinct_wires(rng: &mut StdRng, pool: u16, n: usize) -> Vec<u16> {
    let mut ws: Vec<u16> = Vec::with_capacity(n);
    while ws.len() < n {
        let w = rng.random_range(0..pool);
        if !ws.contains(&w) {
            ws.push(w);
        }
    }
    ws
}

fn random_g57(rng: &mut StdRng, pool: u16) -> XGate {
    let ws = distinct_wires(rng, pool, 3);
    XGate::from_g57([ws[0], ws[1], ws[2]])
}

// A random c-control conjunction gate that is NOT a g57 (for c=2 the g57
// polarity/comp pattern is resampled away).
fn random_non_g57(rng: &mut StdRng, pool: u16, c: usize) -> XGate {
    let ws = distinct_wires(rng, pool, c + 1);
    loop {
        let comp = rng.random_bool(0.5);
        let pols: Vec<bool> = (0..c).map(|_| rng.random_bool(0.5)).collect();
        // g57 = comp true + exactly one positive of two controls.
        if c == 2 && comp && (pols[0] != pols[1]) {
            continue;
        }
        let mut ctrls: local_mixing::circuit::xgate::Lits =
            ws[1..].iter().zip(&pols).map(|(&w, &p)| (w, p)).collect();
        ctrls.sort_unstable();
        return XGate { target: ws[0], comp, ctrls };
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("usage: db_match_synth <store_dir> [--samples N] [--seed S]");
    let mut samples = 5000u64;
    let mut seed = 1u64;
    let mut pools: &[&str] = &["dense", "sparse"];
    while let Some(a) = args.next() {
        match a.as_str() {
            "--samples" => samples = args.next().unwrap().parse().expect("bad --samples"),
            "--seed" => seed = args.next().unwrap().parse().expect("bad --seed"),
            "--dense-only" => pools = &["dense"],
            "--sparse-only" => pools = &["sparse"],
            _ => panic!("unknown arg {a}"),
        }
    }

    let db = FrozenDb::open(&dir, None);
    let budget = XPolyBudget::default();
    let guard = DegreeGuard { max_degree: 9, probes: 6 };
    let mut rng = StdRng::seed_from_u64(seed);

    println!(
        "{:>4} {:>3} {:>7} {:>7} {:>6} {:>8} {:>7} {:>12} {:>6}",
        "m", "k", "ctrls", "pool", "wires", "samples", "hit%", "mean_matches", "dsk"
    );
    for &pool_kind in pools {
        for m in 3..=6usize {
            let wires: u16 = match pool_kind {
                "dense" => (m + 2) as u16,
                _ => (3 * m) as u16,
            };
            for k in 0..=3usize.min(m) {
                let cs: &[usize] = if k == 0 { &[0] } else { &[1, 2, 3] };
                for &c in cs {
                    let (mut hits, mut dsk) = (0u64, 0u64);
                    let mut match_sum = 0u64;
                    for _ in 0..samples {
                        // k non-g57 positions among m slots.
                        let mut window: Vec<XGate> =
                            (0..m).map(|_| random_g57(&mut rng, wires)).collect();
                        let mut slots: Vec<usize> = (0..m).collect();
                        for i in 0..k {
                            let j = rng.random_range(i..m);
                            slots.swap(i, j);
                        }
                        for &s in &slots[..k] {
                            window[s] = random_non_g57(&mut rng, wires, c);
                        }
                        let res = db_replace(
                            &window,
                            wires as usize,
                            &db,
                            budget,
                            DbMode::SizeAgnostic,
                            guard,
                            false,
                            false,
                            true,
                            false,
                            false,
                            &mut rng,
                        );
                        if res.degree_skipped {
                            dsk += 1;
                        }
                        if res.match_count > 0 {
                            hits += 1;
                        }
                        match_sum += res.match_count as u64;
                    }
                    println!(
                        "{:>4} {:>3} {:>7} {:>7} {:>6} {:>8} {:>7.1} {:>12.2} {:>6}",
                        m,
                        k,
                        c,
                        pool_kind,
                        wires,
                        samples,
                        100.0 * hits as f64 / samples as f64,
                        match_sum as f64 / samples as f64,
                        dsk
                    );
                }
            }
        }
    }
}
