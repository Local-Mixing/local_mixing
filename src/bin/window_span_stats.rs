// Throwaway analysis: sample compression windows exactly like the trial loop
// (random_subcircuit), in two geometries — "global" (whole circuit, mode-1-like) and
// "local" (inside a random 50-gate slice, mode-2-like) — and cross-tab wires-spanned
// against litter composition. Optionally probe the main DB (forward + reverse keys,
// same canonicalize/key path as compression) to see which spans/classes actually hit.
// Usage: window_span_stats <circuit.txt> [span_samples] [db_probes]
// Run from ~/local_mixing_sd so ./db resolves; needs <circuit>.tags alongside the circuit.
use lmdb::{Environment, EnvironmentFlags, Transaction};
use local_mixing::circuit::CircuitSeq;
use local_mixing::circuit::circuit::polys_repr_blob;
use local_mixing::replace::main_mix::open_shard_dbs;
use local_mixing::replace::replace::{random_subcircuit, read_tags_sidecar, window_litter_stats, Tag};
use std::collections::BTreeMap;
use std::path::Path;
use xxhash_rust::xxh3::xxh3_128;

const CLASSES: [&str; 4] = ["full_litter", "single_partial", "two_litters", "three_plus"];

fn class_of(distinct: usize, full: bool) -> usize {
    if full {
        0
    } else if distinct == 1 {
        1
    } else if distinct == 2 {
        2
    } else {
        3
    }
}

fn sample<'a>(
    c: &CircuitSeq,
    tags: &'a [Tag],
    local: bool,
    rng: &mut impl rand::Rng,
) -> Option<(CircuitSeq, usize, usize)> {
    if local {
        let n = c.gates.len();
        if n < 60 {
            return None;
        }
        let off = rng.random_range(0..n - 50);
        let slice = CircuitSeq { gates: c.gates[off..off + 50].to_vec() };
        let (sub, st, en) = random_subcircuit(&slice);
        if sub.gates.is_empty() {
            return None;
        }
        let _ = &tags[off + st..off + en];
        Some((sub, off + st, off + en))
    } else {
        let (sub, st, en) = random_subcircuit(c);
        if sub.gates.is_empty() {
            return None;
        }
        Some((sub, st, en))
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: window_span_stats <circuit.txt> [span_samples] [db_probes]");
    let span_samples: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(200_000);
    let db_probes: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(10_000);

    let s = std::fs::read_to_string(&path).expect("read circuit");
    let c = CircuitSeq::from_string(&s);
    let tags = read_tags_sidecar(&path).expect("no .tags sidecar next to circuit");
    assert_eq!(tags.len(), c.gates.len(), "tags/gates length mismatch");
    eprintln!("loaded {} gates + tags from {}", c.gates.len(), path);

    let mut rng = rand::rng();

    // Pass 1: span x litter-class counts, no DB.
    for &local in &[false, true] {
        let mode = if local { "local" } else { "global" };
        let mut counts: BTreeMap<(usize, usize), u64> = BTreeMap::new();
        let mut wires_sum = [0u64; 4];
        let mut gates_sum = [0u64; 4];
        let mut n_class = [0u64; 4];
        for _ in 0..span_samples {
            let Some((sub, st, en)) = sample(&c, &tags, local, &mut rng) else { continue };
            let (distinct, full) = window_litter_stats(&tags[st..en]);
            let cls = class_of(distinct, full);
            let w = sub.used_wires().len();
            *counts.entry((cls, w)).or_insert(0) += 1;
            wires_sum[cls] += w as u64;
            gates_sum[cls] += sub.gates.len() as u64;
            n_class[cls] += 1;
        }
        for cls in 0..4 {
            if n_class[cls] > 0 {
                println!(
                    "CLASS,{},{},n={},mean_wires={:.2},mean_gates={:.2}",
                    mode,
                    CLASSES[cls],
                    n_class[cls],
                    wires_sum[cls] as f64 / n_class[cls] as f64,
                    gates_sum[cls] as f64 / n_class[cls] as f64
                );
            }
        }
        for ((cls, w), n) in &counts {
            println!("SPAN,{},{},{},{}", mode, CLASSES[*cls], w, n);
        }
    }

    // Pass 2: DB probes — hit rate by span and class (forward then reverse key, like the trial).
    let env = Environment::new()
        .set_flags(EnvironmentFlags::READ_ONLY | EnvironmentFlags::NO_LOCK)
        .set_max_readers(10000)
        .set_max_dbs(556)
        .set_map_size(800 * 1024 * 1024 * 1024)
        .open(Path::new("./db"))
        .expect("open ./db");
    let shard_dbs = open_shard_dbs(&env);

    for &local in &[false, true] {
        let mode = if local { "local" } else { "global" };
        let mut probes: BTreeMap<(usize, usize), (u64, u64)> = BTreeMap::new();
        let mut canon_bail = 0u64;
        let mut done = 0usize;
        while done < db_probes {
            let Some((sub, st, en)) = sample(&c, &tags, local, &mut rng) else { continue };
            let (distinct, full) = window_litter_stats(&tags[st..en]);
            let cls = class_of(distinct, full);
            let w = sub.used_wires().len();
            done += 1;
            let (fwd_polys, _, _) = sub.canonicalize_polys_single(false);
            if fwd_polys.is_empty() {
                canon_bail += 1;
                continue;
            }
            let txn = env.begin_ro_txn().expect("ro txn");
            let fwd_key = xxh3_128(&polys_repr_blob(&fwd_polys)).to_le_bytes().to_vec();
            let mut hit = txn.get(shard_dbs[fwd_key[0] as usize], &fwd_key).is_ok();
            if !hit {
                let (rev_polys, _, _) = sub.canonicalize_polys_single(true);
                if !rev_polys.is_empty() {
                    let rev_key = xxh3_128(&polys_repr_blob(&rev_polys)).to_le_bytes().to_vec();
                    hit = txn.get(shard_dbs[rev_key[0] as usize], &rev_key).is_ok();
                }
            }
            drop(txn);
            let e = probes.entry((cls, w)).or_insert((0, 0));
            e.0 += 1;
            if hit {
                e.1 += 1;
            }
        }
        println!("DBBAIL,{},{}", mode, canon_bail);
        for ((cls, w), (p, h)) in &probes {
            println!("DBSTAT,{},{},{},{},{}", mode, CLASSES[*cls], w, p, h);
        }
    }
}
