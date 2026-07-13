// Benchmark canonicalize_polys_single on window sizes drawn like the real
// compress/expand hot paths (contiguous 6-14 gate and 2-5 gate slices of a
// large random circuit), plus adversarial symmetric circuits that stress
// Rule L. Emits a golden digest over all canonical polynomial forms so
// optimizations can be verified byte-identical, and self-checks that each
// returned final_order actually reproduces the canonical polys.
//
// Env knobs:
//   BENCH_WINDOWS_COMPRESS (default 2000)  compress-style window count
//   BENCH_WINDOWS_EXPAND   (default 2000)  expand-style window count
//   BENCH_WINDOWS_REPEAT   (default 3)     timed passes over the corpus
//   BENCH_WINDOWS_RELABEL  (default 10)    relabel-invariance check every Nth window

use local_mixing::circuit::circuit::{
    CANON4_CORE_TIME, CANON4_RULE_L_BRANCHES, CANON4_RULE_L_CALLS, CANON4_RULE_L_TIME,
    CANON_CACHE_HITS, CANON_CACHE_QUERIES, CircuitSeq, Permutation, Polynomial,
    polynomial_from_terms, polys_repr_blob, trim_canonicalized,
};
use local_mixing::random::random_data::random_circuit;
use std::sync::atomic::Ordering;
use std::time::Instant;
use xxhash_rust::xxh3::Xxh3;

const PARENT_N: usize = 64;
const PARENT_M: usize = 1000;
const PARENT_SEED: u64 = 0x77696e_646f7773; // "win dows"

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn dense_remap(sub: &CircuitSeq) -> CircuitSeq {
    let used = sub.used_wires();
    let len = used.last().map_or(0, |&w| w as usize + 1);
    let mut map = vec![u16::MAX; len];
    for (i, &w) in used.iter().enumerate() {
        map[w as usize] = i as u16;
    }
    CircuitSeq {
        gates: sub
            .gates
            .iter()
            .map(|&[t, a, b]| [map[t as usize], map[a as usize], map[b as usize]])
            .collect(),
    }
}

// Re-derive the dense polys for one direction and verify that remapping them
// by final_order reproduces the canonical polys returned by
// canonicalize_polys_single. This is the invariant compress/expand rely on to
// map DB candidates back into circuit space.
fn order_reproduces_canon(
    sub: &CircuitSeq,
    reversed: bool,
    canon_polys: &[Polynomial],
    order: &Permutation,
) -> bool {
    let mut c = dense_remap(sub);
    if reversed {
        c.gates.reverse();
    }
    c.canonicalize();
    let n = c.max_wire() + 1;
    let polys = c.to_polynomial(n, 0, c.gates.len());
    if order.data.len() != n {
        return false;
    }
    let mut wire_to_pos = vec![0usize; n];
    for (pos, &w) in order.data.iter().enumerate() {
        wire_to_pos[w] = pos;
    }
    let remapped: Vec<Polynomial> = order
        .data
        .iter()
        .map(|&w| {
            polynomial_from_terms(polys[w].iter().map(|&m| {
                let mut r = 0u64;
                let mut mm = m;
                while mm != 0 {
                    let v = mm.trailing_zeros() as usize;
                    r |= 1u64 << wire_to_pos[v];
                    mm &= mm - 1;
                }
                r
            }))
        })
        .collect();
    trim_canonicalized(remapped) == canon_polys
}

// Relabel the window's wires with a random permutation of the parent wire
// space; canonical polys must be invariant.
fn relabel_invariant(sub: &CircuitSeq, rng: &mut fastrand::Rng) -> bool {
    let space = sub.max_wire() + 1;
    let mut perm: Vec<usize> = (0..space).collect();
    rng.shuffle(&mut perm);
    let mut relabeled = sub.clone();
    relabeled.rewire(&Permutation { data: perm }, space);

    let (fwd_a, _, _) = sub.canonicalize_polys_single(false);
    let (fwd_b, _, _) = relabeled.canonicalize_polys_single(false);
    let (rev_a, _, _) = sub.canonicalize_polys_single(true);
    let (rev_b, _, _) = relabeled.canonicalize_polys_single(true);
    fwd_a == fwd_b && rev_a == rev_b
}

fn parallel_gates(k: usize) -> CircuitSeq {
    CircuitSeq {
        gates: (0..k as u16).map(|i| [3 * i, 3 * i + 1, 3 * i + 2]).collect(),
    }
}

fn parallel_motifs(k: usize) -> CircuitSeq {
    CircuitSeq {
        gates: (0..k as u16)
            .flat_map(|i| [[3 * i, 3 * i + 1, 3 * i + 2], [3 * i + 1, 3 * i + 2, 3 * i]])
            .collect(),
    }
}

// Two identical random 5-gate blocks on disjoint wire sets: block-swap symmetry.
fn replicated_block(seed: u64) -> CircuitSeq {
    fastrand::seed(seed);
    let block = random_circuit(8, 5);
    let mut gates = block.gates.clone();
    gates.extend(block.gates.iter().map(|&[t, a, b]| [t + 8, a + 8, b + 8]));
    CircuitSeq { gates }
}

fn percentile(sorted: &[u128], p: f64) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx]
}

fn main() {
    let n_compress = env_usize("BENCH_WINDOWS_COMPRESS", 2000);
    let n_expand = env_usize("BENCH_WINDOWS_EXPAND", 2000);
    let repeats = env_usize("BENCH_WINDOWS_REPEAT", 3).max(1);
    let relabel_every = env_usize("BENCH_WINDOWS_RELABEL", 10).max(1);

    fastrand::seed(PARENT_SEED);
    let parent = random_circuit(PARENT_N, PARENT_M);
    let len = parent.gates.len();
    let mut rng = fastrand::Rng::with_seed(PARENT_SEED ^ 0x5eed);

    // (window, section) corpus, deterministic.
    let mut corpus: Vec<(CircuitSeq, &'static str)> = Vec::new();
    for _ in 0..n_compress {
        let size = rng.usize(6..=14);
        let start = rng.usize(0..len - size);
        corpus.push((
            CircuitSeq {
                gates: parent.gates[start..start + size].to_vec(),
            },
            "compress",
        ));
    }
    for _ in 0..n_expand {
        let size = [2usize, 4, 5][rng.usize(0..3)];
        let start = rng.usize(0..len - size);
        corpus.push((
            CircuitSeq {
                gates: parent.gates[start..start + size].to_vec(),
            },
            "expand",
        ));
    }
    for k in 3..=7 {
        corpus.push((parallel_gates(k), "rule_l"));
        corpus.push((parallel_motifs(k), "rule_l"));
    }
    for s in 0..3u64 {
        corpus.push((replicated_block(0xb10c_0000 + s), "rule_l"));
    }

    // Timed passes.
    println!("section,pass,windows,total_ms,p50_ns,p90_ns,p99_ns,max_ns");
    for section in ["compress", "expand", "rule_l"] {
        for pass in 0..repeats {
            let mut nanos: Vec<u128> = Vec::new();
            for (sub, sec) in &corpus {
                if *sec != section {
                    continue;
                }
                let t = Instant::now();
                let (fwd, _, _) = sub.canonicalize_polys_single(false);
                let (rev, _, _) = sub.canonicalize_polys_single(true);
                std::hint::black_box((&fwd, &rev));
                nanos.push(t.elapsed().as_nanos());
            }
            let total_ms = nanos.iter().sum::<u128>() as f64 / 1e6;
            nanos.sort_unstable();
            println!(
                "{section},{pass},{},{:.2},{},{},{},{}",
                nanos.len(),
                total_ms,
                percentile(&nanos, 0.5),
                percentile(&nanos, 0.9),
                percentile(&nanos, 0.99),
                nanos.last().copied().unwrap_or(0)
            );
        }
    }

    // Untimed verification + golden digest pass.
    let mut hasher = Xxh3::new();
    let mut order_failures = 0usize;
    let mut relabel_failures = 0usize;
    for (i, (sub, _)) in corpus.iter().enumerate() {
        let (fwd_polys, fwd_order, _) = sub.canonicalize_polys_single(false);
        let (rev_polys, rev_order, _) = sub.canonicalize_polys_single(true);
        hasher.update(&polys_repr_blob(&fwd_polys));
        hasher.update(&polys_repr_blob(&rev_polys));
        if !order_reproduces_canon(sub, false, &fwd_polys, &fwd_order)
            || !order_reproduces_canon(sub, true, &rev_polys, &rev_order)
        {
            order_failures += 1;
        }
        if i % relabel_every == 0 && !relabel_invariant(sub, &mut rng) {
            relabel_failures += 1;
        }
    }
    println!("digest: {:032x}", hasher.digest128());
    println!("order_check_failures: {order_failures}");
    println!("relabel_check_failures: {relabel_failures}");
    println!(
        "rule_l: calls={} branches={} time_ms={}",
        CANON4_RULE_L_CALLS.load(Ordering::Relaxed),
        CANON4_RULE_L_BRANCHES.load(Ordering::Relaxed),
        CANON4_RULE_L_TIME.load(Ordering::Relaxed) / 1_000_000
    );
    // Top-level canonicalize_polys_4 time inside canonicalize_polys_single;
    // the remainder of wall time is to_polynomial + remap + gate canonicalize.
    println!(
        "canon4_core_ms: {}",
        CANON4_CORE_TIME.load(Ordering::Relaxed) / 1_000_000
    );
    println!(
        "canon_cache: hits={} queries={}",
        CANON_CACHE_HITS.load(Ordering::Relaxed),
        CANON_CACHE_QUERIES.load(Ordering::Relaxed)
    );
}
