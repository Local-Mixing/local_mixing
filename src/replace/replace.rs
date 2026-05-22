// Replacement code used in the mixing methods

use crate::{
    circuit::circuit::{CircuitSeq, Permutation}, random::random_data::{
        contiguous_convex,
        find_convex_subcircuit_max_gates,
        find_convex_subcircuit_max_wires,
        shoot_random_gate,
        simple_find_convex_subcircuit,
        targeted_find_convex_subcircuit_deep,
    }
};
use crate::replace::identities::random_canonical_id;
use crate::replace::identities::random_id;
use crate::replace::pairs::{replace_single_pair, replace_single_pair_with_completion};
use crate::replace::mixing::split_into_random_chunk_ranges;
use rand::Rng;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::fs::File;
use std::io::Write;
use lmdb::Transaction;

extern crate lmdb_sys;

use std::{
    cmp::{max, min},
    collections::{HashMap},
    time::Instant,
};
use std::sync::atomic::{AtomicU64, Ordering};
use dashmap::DashMap;
use once_cell::sync::Lazy;
// use rand::prelude::IndexedRandom;

// Global histogram: (before_gates, after_gates) -> count, accumulated across all rounds
pub static COMPRESSION_HISTOGRAM: Lazy<DashMap<(u8, u8), u64>> = Lazy::new(DashMap::new);

pub fn write_compression_histogram(path: &str) {
    let mut entries: Vec<((u8, u8), u64)> = COMPRESSION_HISTOGRAM
        .iter()
        .map(|e| (*e.key(), *e.value()))
        .collect();
    entries.sort_by_key(|&((before, after), _)| (before, after));
    let mut f = File::create(path).expect("Failed to create histogram CSV");
    writeln!(f, "before,after,count").expect("write");
    for ((before, after), count) in &entries {
        writeln!(f, "{},{},{}", before, after, count).expect("write");
    }
    println!("Compression histogram written to {}", path);

    let log_path = "compression_log.txt";
    let mut log = File::create(log_path).expect("Failed to create compression log");
    let before_vals: Vec<u8> = {
        let mut v: Vec<u8> = entries.iter().map(|&((b, _), _)| b).collect();
        v.dedup();
        v
    };
    for before in before_vals {
        let group: Vec<_> = entries.iter().filter(|&((b, _), _)| *b == before).collect();
        let total: u64 = group.iter().map(|(_, c)| c).sum();
        writeln!(log, "{} gates before (total: {}):", before, total).expect("write");
        for ((_, after), count) in &group {
            writeln!(log, "  -> {} gates: {}", after, count).expect("write");
        }
    }
    println!("Compression log written to {}", log_path);
    println!("Written to log");
}

// Return a random contiguous subcircuit, its starting index (gate), and ending index
pub fn random_subcircuit(circuit: &CircuitSeq) -> (CircuitSeq, usize, usize) {
    let len = circuit.gates.len();
    
    if circuit.gates.len() == 0 {
        return (CircuitSeq{gates: Vec::new()}, 0, 0)
    }

    let mut rng = rand::rng();
    //get size with more bias to lower length subcircuits
    let a = rng.random_range(0..len);

    // pick one of 1, 2, 4, 8
    let shift = rng.random_range(0..4);
    let upper = 1 << shift;

    let mut b = (a + (6 + rng.random_range(0..upper))) as usize;

    if b > len {
        b = len;
    }

    if a == b {
        if b < len - 1 {
            b += 1;
        } else {
            b -= 1;
        }
    }

    let start = min(a,b);
    let end = max(a,b);

    let subcircuit = circuit.gates[start..end].to_vec();

    (CircuitSeq{ gates: subcircuit }, start, end)
}

pub fn random_subcircuit_max(circuit: &CircuitSeq, max_len: usize) -> (CircuitSeq, usize, usize) {
    let len = circuit.gates.len();
    if len == 0 {
        return (CircuitSeq { gates: Vec::new() }, 0, 0);
    }

    let mut rng = rand::rng();

    let start = rng.random_range(0..len);

    let remaining = len - start;
    let allowed_len = remaining.min(max_len);

    let shift = rng.random_range(0..4); // 0..3
    let mut sub_len = 1 << shift;        // 1,2,4,8
    if sub_len > allowed_len {
        sub_len = allowed_len;
    }

    sub_len = sub_len.max(1);

    let end = start + sub_len;
    let subcircuit = circuit.gates[start..end].to_vec();

    (CircuitSeq { gates: subcircuit }, start, end)
}

// Timing variables for benchmarking
pub static PERMUTATION_TIME: AtomicU64 = AtomicU64::new(0);
pub static DUCKDB_TIME: AtomicU64 = AtomicU64::new(0);
pub static CANON_TIME: AtomicU64 = AtomicU64::new(0);
pub static CONVEX_FIND_TIME: AtomicU64 = AtomicU64::new(0);
pub static CONVEX_MAX_WIRES_TIME: AtomicU64 = AtomicU64::new(0);
pub static CONVEX_MAX_GATES_TIME: AtomicU64 = AtomicU64::new(0);
pub static CONVEX_SIMPLE_TIME: AtomicU64 = AtomicU64::new(0);
pub static CONTIGUOUS_TIME: AtomicU64 = AtomicU64::new(0);
pub static REWIRE_TIME: AtomicU64 = AtomicU64::new(0);
pub static COMPRESS_TIME: AtomicU64 = AtomicU64::new(0);
pub static UNREWIRE_TIME: AtomicU64 = AtomicU64::new(0);
pub static REPLACE_TIME: AtomicU64 = AtomicU64::new(0);
pub static DEDUP_TIME: AtomicU64 = AtomicU64::new(0);
pub static PICK_SUBCIRCUIT_TIME: AtomicU64 = AtomicU64::new(0);
pub static CANONICALIZE_TIME: AtomicU64 = AtomicU64::new(0);
pub static CANONICALIZE_TIME_MAX_WIRES: AtomicU64 = AtomicU64::new(0);
pub static CANONICALIZE_TIME_SIMPLE: AtomicU64 = AtomicU64::new(0);
pub static CANONICALIZE_TIME_MAX_GATES: AtomicU64 = AtomicU64::new(0);
pub static ROW_FETCH_TIME: AtomicU64 = AtomicU64::new(0);
pub static SROW_FETCH_TIME: AtomicU64 = AtomicU64::new(0);
pub static SIXROW_FETCH_TIME: AtomicU64 = AtomicU64::new(0);
pub static LROW_FETCH_TIME: AtomicU64 = AtomicU64::new(0);
pub static DB_OPEN_TIME: AtomicU64 = AtomicU64::new(0);
pub static TXN_TIME: AtomicU64 = AtomicU64::new(0);
pub static LMDB_LOOKUP_TIME: AtomicU64 = AtomicU64::new(0);
pub static FROM_BLOB_TIME: AtomicU64 = AtomicU64::new(0);
pub static SPLICE_TIME: AtomicU64 = AtomicU64::new(0);
pub static TRIAL_TIME: AtomicU64 = AtomicU64::new(0);
pub static IDENTITY_TIME: AtomicU64 = AtomicU64::new(0);

pub fn compress_loop(
    circuit: &CircuitSeq,
    n: usize,
    env: &lmdb::Environment,
    shard_dbs: &[lmdb::Database],
    stable_max: usize,
    curr_round: usize,
    last_round: usize,
    output_path: &str,
) -> CircuitSeq {
    let mut acc = circuit.clone();
    let mut rng = rand::rng();
    let mut stable_count = 0;
    let mut mode = 0usize;

    while stable_count < stable_max {
        let before = acc.gates.len();

        let max_chunks = 4 * rayon::current_num_threads().max(1);
        let k = if before <= 1500 {
            1
        } else {
            ((before + 1499) / 1500).min(max_chunks)
        };

        let current_mode = [0, 1, 2][mode];
        mode = (mode + 1) % 3;

        let ranges = split_into_random_chunk_ranges(acc.gates.len(), k, &mut rng);
        let compressed_chunks: Vec<Vec<[u8; 3]>> = ranges
            .into_par_iter()
            .map(|(start, end)| {
                let sub = CircuitSeq {
                    gates: acc.gates[start..end].to_vec(),
                };
                compress_big_ancillas(&sub, 100, n, env, shard_dbs, current_mode).gates
            })
            .collect();

        let total_len: usize = compressed_chunks.iter().map(|chunk| chunk.len()).sum();
        let mut new_gates = Vec::with_capacity(total_len);
        for chunk in compressed_chunks {
            new_gates.extend(chunk);
        }

        acc.gates = new_gates;
        let after = acc.gates.len();

        if after == before {
            stable_count += 1;
            println!("  {}/{}: Stable {}/{}: {} gates", curr_round, last_round, stable_count, stable_max, after);
        } else {
            stable_count = 0;
            println!("  {}/{}: Reduced: {} gates", curr_round, last_round, after);
        }

        // Check if user created write_now
        if std::path::Path::new("write_now").exists() {
            std::fs::remove_file("write_now").ok();
            let mut f = File::create(output_path).expect("create");
            writeln!(f, "{}", acc.repr()).expect("write");
            eprintln!("Wrote {}", output_path);
        }
    }
    acc
}

pub fn expand_loop<'a>(
    circuit: &CircuitSeq,
    n: usize,
    env: &lmdb::Environment,
    shard_dbs: &[lmdb::Database],
    target_multiplier: usize,
    pair_mode: &ExpandPairMode<'a>,
) -> CircuitSeq {
    let target = circuit.gates.len() * target_multiplier;
    let mut acc = circuit.clone();
    let mut rng = rand::rng();
    let mut mode = 0usize;

    while acc.gates.len() < target {
        let before = acc.gates.len();

        let max_chunks = 4 * rayon::current_num_threads().max(1);
        let k = if before <= 1500 {
            1
        } else {
            ((before + 1499) / 1500).min(max_chunks)
        };

        let current_mode = [0, 1, 2][mode];
        mode = (mode + 1) % 3;

        let ranges = split_into_random_chunk_ranges(acc.gates.len(), k, &mut rng);
        let expanded_chunks: Vec<Vec<[u8; 3]>> = ranges
            .into_par_iter()
            .map(|(start, end)| {
                let sub = CircuitSeq {
                    gates: acc.gates[start..end].to_vec(),
                };
                expand_big_ancillas(&sub, 100, n, env, shard_dbs, current_mode, pair_mode).gates
            })
            .collect();

        let total_len: usize = expanded_chunks.iter().map(|chunk| chunk.len()).sum();
        let mut new_gates = Vec::with_capacity(total_len);
        for chunk in expanded_chunks {
            new_gates.extend(chunk);
        }

        acc.gates = new_gates;
        println!("  Expand: {} gates (target {})", acc.gates.len(), target);
    }

    acc
}

/// Single pass of expansion: one round of chunked `expand_big_ancillas` with no loop.
pub fn expand_once<'a>(
    circuit: &CircuitSeq,
    n: usize,
    env: &lmdb::Environment,
    shard_dbs: &[lmdb::Database],
    pair_mode: &ExpandPairMode<'a>,
) -> CircuitSeq {
    let mut rng = rand::rng();
    let before = circuit.gates.len();
    let max_chunks = 4 * rayon::current_num_threads().max(1);
    let k = if before <= 1500 { 1 } else { ((before + 1499) / 1500).min(max_chunks) };
    let ranges = split_into_random_chunk_ranges(before, k, &mut rng);
    let expanded_chunks: Vec<Vec<[u8; 3]>> = ranges
        .into_par_iter()
        .map(|(start, end)| {
            let sub = CircuitSeq { gates: circuit.gates[start..end].to_vec() };
            expand_big_ancillas(&sub, 100, n, env, shard_dbs, 0, pair_mode).gates
        })
        .collect();
    let mut new_gates = Vec::with_capacity(expanded_chunks.iter().map(|c| c.len()).sum());
    for chunk in expanded_chunks {
        new_gates.extend(chunk);
    }
    println!("  Expand: {} gates", new_gates.len());
    CircuitSeq { gates: new_gates }
}

/// Like `compress_loop` but exits as soon as the circuit drops below `stop` gates.
/// On the final round use the plain `compress_loop` instead.
pub fn compress_loop_early(
    circuit: &CircuitSeq,
    n: usize,
    env: &lmdb::Environment,
    shard_dbs: &[lmdb::Database],
    stable_max: usize,
    curr_round: usize,
    last_round: usize,
    output_path: &str,
    stop: usize,
) -> CircuitSeq {
    let mut acc = circuit.clone();
    let mut rng = rand::rng();
    let mut stable_count = 0;
    let mut mode = 0usize;

    while stable_count < stable_max {
        let before = acc.gates.len();

        let max_chunks = 4 * rayon::current_num_threads().max(1);
        let k = if before <= 1500 {
            1
        } else {
            ((before + 1499) / 1500).min(max_chunks)
        };

        let current_mode = [0, 1, 2][mode];
        mode = (mode + 1) % 3;

        let ranges = split_into_random_chunk_ranges(acc.gates.len(), k, &mut rng);
        let compressed_chunks: Vec<Vec<[u8; 3]>> = ranges
            .into_par_iter()
            .map(|(start, end)| {
                let sub = CircuitSeq {
                    gates: acc.gates[start..end].to_vec(),
                };
                compress_big_ancillas(&sub, 100, n, env, shard_dbs, current_mode).gates
            })
            .collect();

        let total_len: usize = compressed_chunks.iter().map(|c| c.len()).sum();
        let mut new_gates = Vec::with_capacity(total_len);
        for chunk in compressed_chunks {
            new_gates.extend(chunk);
        }

        acc.gates = new_gates;
        let after = acc.gates.len();

        if after < stop {
            println!("  {}/{}: Early stop at {} gates (below {})", curr_round, last_round, after, stop);
            break;
        }

        if after == before {
            stable_count += 1;
            println!("  {}/{}: Stable {}/{}: {} gates", curr_round, last_round, stable_count, stable_max, after);
        } else {
            stable_count = 0;
            println!("  {}/{}: Reduced: {} gates", curr_round, last_round, after);
        }

        if std::path::Path::new("write_now").exists() {
            std::fs::remove_file("write_now").ok();
            let mut f = File::create(output_path).expect("create");
            writeln!(f, "{}", acc.repr()).expect("write");
            eprintln!("Wrote {}", output_path);
        }
    }
    acc
}

// Expand with ancilla wires or gates
/// Selects which method to use when a 2-gate subcircuit is sampled in the expand functions.
/// For subcircuits of 3–5 gates the shard DB is always used regardless of this setting.
pub enum ExpandPairMode<'a> {
    /// Use the completion_m2 DB to find a longer equivalent pair.
    Curated,
    /// Use replace_single_pair (random identity from id_g{} DBs) to get a longer equivalent.
    Canonical {
        bit_shuf_list: &'a Vec<Vec<Vec<usize>>>,
        dbs: &'a HashMap<String, lmdb::Database>,
        id_len: usize,
        tower: bool,
    },
    /// Force the shard DB lookup even for 2-gate subcircuits (same path as 3-5 gates).
    Db,
}

pub fn expand_lmdb<'a>(
    c: &CircuitSeq,
    trials: usize,
    n: usize,
    env: &lmdb::Environment,
    shard_dbs: &[lmdb::Database],
    pair_mode: &ExpandPairMode<'a>,
) -> CircuitSeq {
    use xxhash_rust::xxh3::xxh3_128;
    use crate::circuit::circuit::polys_repr_blob;

    let mut expanded = c.clone();

    if expanded.gates.is_empty() {
        return CircuitSeq { gates: Vec::new() };
    }

    // Open completion DB once if we'll need it for 2-gate curated pairs.
    let comp_db: Option<lmdb::Database> = match pair_mode {
        ExpandPairMode::Curated => Some(
            env.open_db(Some("completion_m2"))
                .expect("completion_m2 DB not found — run build_completion_m2 first"),
        ),
        _ => None,
    };

    let mut rng = rand::rng();

    for _ in 0..trials {
        let t_trial = Instant::now();

        let (sub, start, end) = random_subcircuit_max(&expanded, 5);

        // Require at least 2 gates; 1-gate subcircuits cannot be expanded meaningfully.
        if sub.gates.len() < 2 {
            continue;
        }

        // --- 2-gate path: bypass the shard DB and use pair functions ---
        if sub.gates.len() == 2 {
            match pair_mode {
                ExpandPairMode::Curated => {
                    if let Some(db) = comp_db {
                        if let Some(repl) = replace_single_pair_with_completion(
                            &sub.gates[0], &sub.gates[1], n, env, db,
                        ) {
                            if repl.len() > 2 {
                                expanded.gates.splice(start..end, repl);
                            }
                        }
                    }
                    TRIAL_TIME.fetch_add(t_trial.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    continue;
                }
                ExpandPairMode::Canonical { bit_shuf_list, dbs, id_len, tower } => {
                    let (repl, _) = replace_single_pair(
                        &sub.gates[0], &sub.gates[1], n, env,
                        bit_shuf_list, dbs, *tower, *id_len, &[], &[],
                    );
                    if repl.len() > 2 {
                        expanded.gates.splice(start..end, repl);
                    }
                    TRIAL_TIME.fetch_add(t_trial.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    continue;
                }
                ExpandPairMode::Db => { /* fall through to shard DB path */ }
            }
        }

        let t_canon = Instant::now();
        let (fwd_polys, fwd_order, used) = sub.canonicalize_polys_single(false);
        CANONICALIZE_TIME.fetch_add(t_canon.elapsed().as_nanos() as u64, Ordering::Relaxed);

        if fwd_polys.is_empty() {
            continue;
        }

        let fwd_key = xxh3_128(&polys_repr_blob(&fwd_polys)).to_le_bytes().to_vec();
        let fwd_shard = fwd_key[0] as usize;

        let t_txn = Instant::now();
        let txn = match env.begin_ro_txn() {
            Ok(t) => t,
            Err(_) => continue,
        };
        TXN_TIME.fetch_add(t_txn.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t_lookup = Instant::now();
        let fwd_result = txn.get(shard_dbs[fwd_shard], &fwd_key).map(|v: &[u8]| v.to_vec());
        LMDB_LOOKUP_TIME.fetch_add(t_lookup.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let (value, final_order, is_reversed) = if let Ok(v) = fwd_result {
            (v, fwd_order, false)
        } else {
            let t_canon2 = Instant::now();
            let (rev_polys, rev_order, _) = sub.canonicalize_polys_single(true);
            CANONICALIZE_TIME.fetch_add(t_canon2.elapsed().as_nanos() as u64, Ordering::Relaxed);

            if rev_polys.is_empty() {
                continue;
            }

            let rev_key = xxh3_128(&polys_repr_blob(&rev_polys)).to_le_bytes().to_vec();
            let rev_shard = rev_key[0] as usize;

            let t_lookup2 = Instant::now();
            let rev_result = txn.get(shard_dbs[rev_shard], &rev_key).map(|v: &[u8]| v.to_vec());
            LMDB_LOOKUP_TIME.fetch_add(t_lookup2.elapsed().as_nanos() as u64, Ordering::Relaxed);

            match rev_result {
                Ok(v) => (v, rev_order, true),
                Err(_) => continue,
            }
        };

        let t_blob = Instant::now();
        let mut candidates: Vec<CircuitSeq> = Vec::new();
        let mut pos = 0;
        while pos < value.len() {
            if pos + 1 > value.len() { break; }
            let len = value[pos] as usize;
            pos += 1;
            if pos + len > value.len() { break; }
            let candidate = CircuitSeq::from_blob(&value[pos..pos + len]);
            pos += len;
            if candidate.gates.len() > sub.gates.len() {
                candidates.push(candidate);
            }
        }
        FROM_BLOB_TIME.fetch_add(t_blob.elapsed().as_nanos() as u64, Ordering::Relaxed);

        if candidates.is_empty() {
            continue;
        }

        let max_gates = candidates.iter().map(|c| c.gates.len()).max().unwrap();
        let mut best: Vec<CircuitSeq> = candidates.into_iter().filter(|c| c.gates.len() == max_gates).collect();
        let idx = rng.random_range(0..best.len());
        let mut repl = best.swap_remove(idx);

        if is_reversed {
            repl.gates.reverse();
        }

        let t_rewire = Instant::now();
        let repl_n = repl.max_wire() + 1;
        let mut order_data = final_order.data.clone();
        while order_data.len() < repl_n {
            let i = order_data.len();
            order_data.push(i);
        }

        repl.rewire(&Permutation { data: order_data }, std::cmp::max(repl_n, final_order.data.len()));

        let repl_n_b = repl.max_wire() + 1;
        let mut used_ext = used.clone();
        if used_ext.len() < repl_n_b {
            let mut available: Vec<u8> = (0..n.min(256))
                .map(|w| w as u8)
                .filter(|w| !used_ext.contains(w))
                .collect();
            rand::seq::SliceRandom::shuffle(available.as_mut_slice(), &mut rng);
            let mut avail = available.into_iter();
            while used_ext.len() < repl_n_b {
                match avail.next() {
                    Some(w) => used_ext.push(w),
                    None => break,
                }
            }
        }
        let repl = CircuitSeq::unrewire_subcircuit(&repl, &used_ext);
        REWIRE_TIME.fetch_add(t_rewire.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t_splice = Instant::now();
        expanded.gates.splice(start..end, repl.gates);
        SPLICE_TIME.fetch_add(t_splice.elapsed().as_nanos() as u64, Ordering::Relaxed);

        TRIAL_TIME.fetch_add(t_trial.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    expanded
}

pub fn compress_big(
    c: &CircuitSeq,
    trials: usize,
    num_wires: usize,
    env: &lmdb::Environment,
    shard_dbs: &[lmdb::Database],
) -> CircuitSeq {
    let mut circuit = c.clone();
    let mut rng = rand::rng();

    let mut i = 0;
    while i < circuit.gates.len().saturating_sub(1) {
        if circuit.gates[i] == circuit.gates[i + 1] {
            circuit.gates.drain(i..=i + 1);
            i = i.saturating_sub(2);
        } else {
            i += 1;
        }
    }

    for _ in 0..trials {
        shoot_random_gate(&mut circuit, 100_000);
        let t0 = Instant::now();
        let mut subcircuit_gates = vec![];
        let random_max_wires = rng.random_range(5..=7);
        let size = if random_max_wires == 7 {
            6
        } else if random_max_wires == 6 {
            4
        } else {
            3
        };
        for set_size in (3..=size).rev() {
            let (gates, _) = simple_find_convex_subcircuit(set_size, random_max_wires, num_wires, &circuit, &mut rng);
            if !gates.is_empty() {
                subcircuit_gates = gates;
                break;
            }
        }
        CONVEX_FIND_TIME.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

        if subcircuit_gates.is_empty() {
            continue;
        }

        let gates: Vec<[u8; 3]> = subcircuit_gates.iter().map(|&g| circuit.gates[g]).collect();
        subcircuit_gates.sort();

        let t1 = Instant::now();
        let (start, end) = contiguous_convex(&mut circuit, &mut subcircuit_gates, num_wires).unwrap();
        CONTIGUOUS_TIME.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let mut subcircuit = CircuitSeq { gates };

        let expected_slice: Vec<_> = subcircuit_gates.iter().map(|&i| circuit.gates[i]).collect();
        let actual_slice = &circuit.gates[start..=end];
        if actual_slice != &expected_slice[..] {
            continue;
        }

        let t2 = Instant::now();
        let used_wires = subcircuit.used_wires();
        subcircuit = CircuitSeq::rewire_subcircuit(&mut circuit, &mut subcircuit_gates, &used_wires);
        REWIRE_TIME.fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t3 = Instant::now();
        let sub_num_wires = used_wires.len();
        PERMUTATION_TIME.fetch_add(t3.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t4 = Instant::now();
        let subcircuit_temp = compress_lmdb(&subcircuit, 10, sub_num_wires, env, shard_dbs, 1);
        COMPRESS_TIME.fetch_add(t4.elapsed().as_nanos() as u64, Ordering::Relaxed);

        subcircuit = subcircuit_temp;

        let t5 = Instant::now();
        subcircuit = CircuitSeq::unrewire_subcircuit(&subcircuit, &used_wires);
        UNREWIRE_TIME.fetch_add(t5.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t6 = Instant::now();
        let repl_len = subcircuit.gates.len();
        let old_len = end - start + 1;

        if repl_len == old_len {
            for i in 0..repl_len {
                circuit.gates[start + i] = subcircuit.gates[i];
            }
        } else if repl_len < old_len {
            for i in 0..repl_len {
                circuit.gates[start + i] = subcircuit.gates[i];
            }
            for i in (end + 1)..circuit.gates.len() {
                circuit.gates[i - (old_len - repl_len)] = circuit.gates[i];
            }
            circuit.gates.truncate(circuit.gates.len() - (old_len - repl_len));
        } else {
            panic!("Replacement grew, which is not allowed");
        }
        REPLACE_TIME.fetch_add(t6.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    let t7 = Instant::now();
    let mut i = 0;
    while i < circuit.gates.len().saturating_sub(1) {
        if circuit.gates[i] == circuit.gates[i + 1] {
            circuit.gates.drain(i..=i + 1);
            i = i.saturating_sub(2);
        } else {
            i += 1;
        }
    }
    DEDUP_TIME.fetch_add(t7.elapsed().as_nanos() as u64, Ordering::Relaxed);

    circuit
}

// Sequential compression method
pub fn sequential_compress_big(
    c: &CircuitSeq,
    num_wires: usize,
    env: &lmdb::Environment,
    shard_dbs: &[lmdb::Database],
) -> CircuitSeq {
    let mut circuit = c.clone();
    let mut rng = rand::rng();

    let mut i = 0;
    while i < circuit.gates.len().saturating_sub(1) {
        if circuit.gates[i] == circuit.gates[i + 1] {
            circuit.gates.drain(i..=i + 1);
            i = i.saturating_sub(2);
        } else {
            i += 1;
        }
    }

    let mut len = circuit.gates.len();
    let mut i = 0;
    while i < len {
        let t0 = Instant::now();
        let mut subcircuit_gates = vec![];
        let random_max_wires = rng.random_range(5..=7);
        let size = if random_max_wires == 7 {
            6
        } else if random_max_wires == 6 {
            4
        } else {
            3
        };
        for set_size in (3..=size).rev() {
            let (gates, _) = targeted_find_convex_subcircuit_deep(set_size, random_max_wires, num_wires, &circuit, &mut rng, i);
            if !gates.is_empty() {
                subcircuit_gates = gates;
                break;
            }
            if set_size == 3 {
                let (gates, _) = targeted_find_convex_subcircuit_deep(set_size, 7, num_wires, &circuit, &mut rng, i);
                subcircuit_gates = gates;
            }
        }
        CONVEX_FIND_TIME.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

        if subcircuit_gates.is_empty() {
            i+=1;
            continue;
        }

        let gates: Vec<[u8; 3]> = subcircuit_gates.iter().map(|&g| circuit.gates[g]).collect();
        subcircuit_gates.sort();

        let t1 = Instant::now();
        let (start, end) = contiguous_convex(&mut circuit, &mut subcircuit_gates, num_wires).unwrap();
        CONTIGUOUS_TIME.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let mut subcircuit = CircuitSeq { gates };

        let expected_slice: Vec<_> = subcircuit_gates.iter().map(|&i| circuit.gates[i]).collect();
        let actual_slice = &circuit.gates[start..=end];
        if actual_slice != &expected_slice[..] {
            i+=1;
            continue;
        }

        let t2 = Instant::now();
        let used_wires = subcircuit.used_wires();
        subcircuit = CircuitSeq::rewire_subcircuit(&mut circuit, &mut subcircuit_gates, &used_wires);
        REWIRE_TIME.fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t3 = Instant::now();
        let sub_num_wires = used_wires.len();
        PERMUTATION_TIME.fetch_add(t3.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t4 = Instant::now();
        let subcircuit_temp = compress_lmdb(&subcircuit, 10, sub_num_wires, env, shard_dbs, 1);
        COMPRESS_TIME.fetch_add(t4.elapsed().as_nanos() as u64, Ordering::Relaxed);

        subcircuit = subcircuit_temp;

        let t5 = Instant::now();
        subcircuit = CircuitSeq::unrewire_subcircuit(&subcircuit, &used_wires);
        UNREWIRE_TIME.fetch_add(t5.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t6 = Instant::now();
        let repl_len = subcircuit.gates.len();
        let old_len = end - start + 1;

        if repl_len == old_len {
            for i in 0..repl_len {
                circuit.gates[start + i] = subcircuit.gates[i];
            }
        } else if repl_len < old_len {
            for i in 0..repl_len {
                circuit.gates[start + i] = subcircuit.gates[i];
            }
            for i in (end + 1)..circuit.gates.len() {
                circuit.gates[i - (old_len - repl_len)] = circuit.gates[i];
            }
            circuit.gates.truncate(circuit.gates.len() - (old_len - repl_len));
        } else {
            panic!("Replacement grew, which is not allowed");
        }
        REPLACE_TIME.fetch_add(t6.elapsed().as_nanos() as u64, Ordering::Relaxed);
        i += 1;
        len = circuit.gates.len();
    }

    let t7 = Instant::now();
    let mut i = 0;
    while i < circuit.gates.len().saturating_sub(1) {
        if circuit.gates[i] == circuit.gates[i + 1] {
            circuit.gates.drain(i..=i + 1);
            i = i.saturating_sub(2);
        } else {
            i += 1;
        }
    }
    DEDUP_TIME.fetch_add(t7.elapsed().as_nanos() as u64, Ordering::Relaxed);

    circuit
}

// Allow ancillas in compression
pub fn sequential_compress_big_ancillas(
    c: &CircuitSeq,
    num_wires: usize,
    env: &lmdb::Environment,
    shard_dbs: &[lmdb::Database],
) -> CircuitSeq {
    let mut circuit = c.clone();
    let mut rng = rand::rng();

    let mut i = 0;
    while i < circuit.gates.len().saturating_sub(1) {
        if circuit.gates[i] == circuit.gates[i + 1] {
            circuit.gates.drain(i..=i + 1);
            i = i.saturating_sub(2);
        } else {
            i += 1;
        }
    }

    let mut len = circuit.gates.len();
    let mut i = 0;
    while i < len {
        let t0 = Instant::now();
        let mut subcircuit_gates = vec![];
        let random_max_wires = rng.random_range(5..=7);
        let size = if random_max_wires == 7 {
            6
        } else if random_max_wires == 6 {
            4
        } else {
            3
        };
        for set_size in (3..=size).rev() {
            let (gates, _) = targeted_find_convex_subcircuit_deep(set_size, random_max_wires, num_wires, &circuit, &mut rng, i);
            if !gates.is_empty() {
                subcircuit_gates = gates;
                break;
            }
            if set_size == 3 {
                let (gates, _) = targeted_find_convex_subcircuit_deep(set_size, 7, num_wires, &circuit, &mut rng, i);
                subcircuit_gates = gates;
            }
        }
        CONVEX_FIND_TIME.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

        if subcircuit_gates.is_empty() {
            i+=1;
            continue;
        }

        let gates: Vec<[u8; 3]> = subcircuit_gates.iter().map(|&g| circuit.gates[g]).collect();
        subcircuit_gates.sort();

        let t1 = Instant::now();
        let (start, end) = contiguous_convex(&mut circuit, &mut subcircuit_gates, num_wires).unwrap();
        CONTIGUOUS_TIME.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let mut subcircuit = CircuitSeq { gates };

        let expected_slice: Vec<_> = subcircuit_gates.iter().map(|&i| circuit.gates[i]).collect();
        let actual_slice = &circuit.gates[start..=end];
        if actual_slice != &expected_slice[..] {
            i+=1;
            continue;
        }

        let t2 = Instant::now();
        let mut used_wires = subcircuit.used_wires();
        let n_wires = used_wires.len();
        let max = 7;
        let new_wires = rng.random_range(n_wires..=max);
        if new_wires > n_wires {
            let mut count = n_wires;
            while count < new_wires {
                let random = rng.random_range(0..num_wires);
                if used_wires.contains(&(random as u8)) {
                    continue
                }
                used_wires.push(random as u8);
                count += 1;
            }
        }
        subcircuit = CircuitSeq::rewire_subcircuit(&mut circuit, &mut subcircuit_gates, &used_wires);
        REWIRE_TIME.fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t3 = Instant::now();
        let sub_num_wires = used_wires.len();
        PERMUTATION_TIME.fetch_add(t3.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t4 = Instant::now();
        let subcircuit_temp = compress_lmdb(&subcircuit, 10, sub_num_wires, env, shard_dbs, 1);
        COMPRESS_TIME.fetch_add(t4.elapsed().as_nanos() as u64, Ordering::Relaxed);

        subcircuit = subcircuit_temp;

        let t5 = Instant::now();
        subcircuit = CircuitSeq::unrewire_subcircuit(&subcircuit, &used_wires);
        UNREWIRE_TIME.fetch_add(t5.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t6 = Instant::now();
        let repl_len = subcircuit.gates.len();
        let old_len = end - start + 1;

        if repl_len == old_len {
            for i in 0..repl_len {
                circuit.gates[start + i] = subcircuit.gates[i];
            }
        } else if repl_len < old_len {
            for i in 0..repl_len {
                circuit.gates[start + i] = subcircuit.gates[i];
            }
            for i in (end + 1)..circuit.gates.len() {
                circuit.gates[i - (old_len - repl_len)] = circuit.gates[i];
            }
            circuit.gates.truncate(circuit.gates.len() - (old_len - repl_len));
        } else {
            panic!("Replacement grew, which is not allowed");
        }
        REPLACE_TIME.fetch_add(t6.elapsed().as_nanos() as u64, Ordering::Relaxed);
        i += 1;
        len = circuit.gates.len();
    }

    let t7 = Instant::now();
    let mut i = 0;
    while i < circuit.gates.len().saturating_sub(1) {
        if circuit.gates[i] == circuit.gates[i + 1] {
            circuit.gates.drain(i..=i + 1);
            i = i.saturating_sub(2);
        } else {
            i += 1;
        }
    }
    DEDUP_TIME.fetch_add(t7.elapsed().as_nanos() as u64, Ordering::Relaxed);

    circuit
}


pub fn compress_lmdb(
    c: &CircuitSeq,
    trials: usize,
    n: usize,
    env: &lmdb::Environment,
    shard_dbs: &[lmdb::Database],
    mode: usize,
) -> CircuitSeq {
    use xxhash_rust::xxh3::xxh3_128;
    use crate::circuit::circuit::polys_repr_blob;

    let mut compressed = c.clone();

    let mut i = 0;
    while i < compressed.gates.len().saturating_sub(1) {
        if compressed.gates[i] == compressed.gates[i + 1] {
            compressed.gates.drain(i..=i + 1);
            i = i.saturating_sub(2);
        } else {
            i += 1;
        }
    }

    if compressed.gates.is_empty() {
        return CircuitSeq { gates: Vec::new() };
    }

    let mut rng = rand::rng();

    for _ in 0..trials {
        let t_trial = Instant::now();

        let (sub, start, end) = random_subcircuit(&compressed);

        if sub.gates.is_empty() {
            continue;
        }

        let t_canon = Instant::now();
        let (fwd_polys, fwd_order, used) = sub.canonicalize_polys_single(false);
        let canon_elapsed = t_canon.elapsed().as_nanos() as u64;
        CANONICALIZE_TIME.fetch_add(canon_elapsed, Ordering::Relaxed);
        match mode {
            0 => CANONICALIZE_TIME_MAX_WIRES.fetch_add(canon_elapsed, Ordering::Relaxed),
            2 => CANONICALIZE_TIME_MAX_GATES.fetch_add(canon_elapsed, Ordering::Relaxed),
            _ => CANONICALIZE_TIME_SIMPLE.fetch_add(canon_elapsed, Ordering::Relaxed),
        };

        if fwd_polys.is_empty() {
            continue;
        }

        let fwd_key = xxh3_128(&polys_repr_blob(&fwd_polys)).to_le_bytes().to_vec();
        let fwd_shard = fwd_key[0] as usize;

        let t_txn = Instant::now();
        let txn = match env.begin_ro_txn() {
            Ok(t) => t,
            Err(_) => continue,
        };
        TXN_TIME.fetch_add(t_txn.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t_lookup = Instant::now();
        let fwd_result = txn.get(shard_dbs[fwd_shard], &fwd_key).map(|v: &[u8]| v.to_vec());
        LMDB_LOOKUP_TIME.fetch_add(t_lookup.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let (value, final_order, is_reversed) = if let Ok(v) = fwd_result {
            (v, fwd_order, false)
        } else {
            let t_canon2 = Instant::now();
            let (rev_polys, rev_order, _) = sub.canonicalize_polys_single(true);
            let canon2_elapsed = t_canon2.elapsed().as_nanos() as u64;
            CANONICALIZE_TIME.fetch_add(canon2_elapsed, Ordering::Relaxed);
            match mode {
                0 => CANONICALIZE_TIME_MAX_WIRES.fetch_add(canon2_elapsed, Ordering::Relaxed),
                2 => CANONICALIZE_TIME_MAX_GATES.fetch_add(canon2_elapsed, Ordering::Relaxed),
                _ => CANONICALIZE_TIME_SIMPLE.fetch_add(canon2_elapsed, Ordering::Relaxed),
            };

            if rev_polys.is_empty() {
                continue;
            }

            let rev_key = xxh3_128(&polys_repr_blob(&rev_polys)).to_le_bytes().to_vec();
            let rev_shard = rev_key[0] as usize;

            let t_lookup2 = Instant::now();
            let rev_result = txn.get(shard_dbs[rev_shard], &rev_key).map(|v: &[u8]| v.to_vec());
            LMDB_LOOKUP_TIME.fetch_add(t_lookup2.elapsed().as_nanos() as u64, Ordering::Relaxed);

            match rev_result {
                Ok(v) => (v, rev_order, true),
                Err(_) => continue,
            }
        };

        let t_blob = Instant::now();
        let mut candidates: Vec<CircuitSeq> = Vec::new();
        let mut pos = 0;
        while pos < value.len() {
            if pos + 1 > value.len() { break; }
            let len = value[pos] as usize;
            pos += 1;
            if pos + len > value.len() { break; }
            let candidate = CircuitSeq::from_blob(&value[pos..pos + len]);
            pos += len;
            if candidate.gates.len() < sub.gates.len() {
                candidates.push(candidate);
            }
        }
        FROM_BLOB_TIME.fetch_add(t_blob.elapsed().as_nanos() as u64, Ordering::Relaxed);

        if candidates.is_empty() {
            continue;
        }

        let min_gates = candidates.iter().map(|c| c.gates.len()).min().unwrap();
        *COMPRESSION_HISTOGRAM.entry((sub.gates.len() as u8, min_gates as u8)).or_insert(0) += 1;
        let mut best: Vec<CircuitSeq> = candidates.into_iter().filter(|c| c.gates.len() == min_gates).collect();
        let idx = rng.random_range(0..best.len());
        let mut repl = best.swap_remove(idx);

        if is_reversed {
            repl.gates.reverse();
        }

        let t_rewire = Instant::now();
        let repl_n = repl.max_wire() + 1;
        let mut order_data = final_order.data.clone();
        while order_data.len() < repl_n {
            let i = order_data.len();
            order_data.push(i);
        }
        repl.rewire(&Permutation { data: order_data }, std::cmp::max(repl_n, final_order.data.len()));

        let repl_n_b = repl.max_wire() + 1;
        let mut used_ext = used.clone();
        if used_ext.len() < repl_n_b {
            let mut available: Vec<u8> = (0..n.min(256))
                .map(|w| w as u8)
                .filter(|w| !used_ext.contains(w))
                .collect();
            rand::seq::SliceRandom::shuffle(available.as_mut_slice(), &mut rng);
            let mut avail = available.into_iter();
            while used_ext.len() < repl_n_b {
                match avail.next() {
                    Some(w) => used_ext.push(w),
                    None => break,
                }
            }
        }
        let repl = CircuitSeq::unrewire_subcircuit(&repl, &used_ext);
        REWIRE_TIME.fetch_add(t_rewire.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t_splice = Instant::now();
        if repl.gates.len() == end - start {
            compressed.gates[start..end].copy_from_slice(&repl.gates);
        } else {
            compressed.gates.splice(start..end, repl.gates);
        }
        SPLICE_TIME.fetch_add(t_splice.elapsed().as_nanos() as u64, Ordering::Relaxed);

        TRIAL_TIME.fetch_add(t_trial.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    let mut j = 0;
    while j < compressed.gates.len().saturating_sub(1) {
        if compressed.gates[j] == compressed.gates[j + 1] {
            compressed.gates.drain(j..=j + 1);
            j = j.saturating_sub(2);
        } else {
            j += 1;
        }
    }

    compressed
}

pub fn expand_big(
    c: &CircuitSeq,
    trials: usize,
    num_wires: usize,
    _env: &lmdb::Environment,
    bit_shuf_list: &Vec<Vec<Vec<usize>>>,
    _dbs: &HashMap<String, lmdb::Database>,
) -> CircuitSeq {
    let mut circuit = c.clone();
    let mut rng = rand::rng();

    for _i in 0..trials {
        // if i % 20 == 0 {
        //     println!("{} trials so far, {} more to go", i, trials - i);
        // }
        let mut subcircuit_gates = vec![];
        let random_max_wires = rng.random_range(3..=7);
        for set_size in (3..=7).rev() {
            let (gates, _) = simple_find_convex_subcircuit(set_size, random_max_wires, num_wires, &circuit, &mut rng);
            if !gates.is_empty() {
                subcircuit_gates = gates;
                break;
            }
        }

        if subcircuit_gates.is_empty() {
            return circuit
        }
        
        let mut gates: Vec<[u8;3]> = vec![[0,0,0]; subcircuit_gates.len()];
        for (i, g) in subcircuit_gates.iter().enumerate() {
            gates[i] = circuit.gates[*g];
        }

        subcircuit_gates.sort();
        let (start, end) = contiguous_convex(&mut circuit, &mut subcircuit_gates, num_wires).unwrap();
        let mut subcircuit = CircuitSeq { gates };
        // let sub_ref = subcircuit.clone();
        let expected_slice: Vec<_> = subcircuit_gates.iter().map(|&i| circuit.gates[i]).collect();
        let actual_slice = &circuit.gates[start..=end];

        if actual_slice != &expected_slice[..] {
            break;
        }

        let mut used_wires = subcircuit.used_wires();
        let n_wires = used_wires.len();
        let max = 7;
        let new_wires = rng.random_range(n_wires..=max);

        if new_wires > n_wires {
            let mut count = n_wires;
            while count < new_wires {
                let random = rng.random_range(0..num_wires);
                if used_wires.contains(&(random as u8)) {
                    continue
                }
                used_wires.push(random as u8);
                count += 1;
            }
        }
        used_wires.sort();
        subcircuit = CircuitSeq::rewire_subcircuit(&mut circuit, &mut subcircuit_gates, &used_wires);

        
        let _bit_shuf = &bit_shuf_list[new_wires - 3];

        subcircuit = CircuitSeq::unrewire_subcircuit(&subcircuit, &used_wires);
        if subcircuit.gates.len() == end+1 - start {
            circuit.gates[start..end+1].copy_from_slice(&subcircuit.gates);
        } else {    
            circuit.gates.splice(start..end+1, subcircuit.gates);
        }
        // if c.permutation(num_wires).data != circuit.permutation(num_wires).data {
        //     panic!("splice changed something");
        // }
    }
    let mut i = 0;
    while i < circuit.gates.len().saturating_sub(1) {
        if circuit.gates[i] == circuit.gates[i + 1] {
            circuit.gates.drain(i..=i + 1);
            i = i.saturating_sub(2);
        } else {
            i += 1;
        }
    }
    circuit
}

// Old legacy code to obfuscate/inflate
pub fn obfuscate(c: &CircuitSeq, num_wires: usize) -> (CircuitSeq, Vec<usize>) {
    if c.gates.len() == 0 {
        return (CircuitSeq { gates: Vec::new() }, Vec::new() )
    }
    let mut obfuscated = CircuitSeq { gates: Vec::new() };
    let mut inverse_starts = Vec::new();

    let mut rng = rand::rng();

    // for butterfly
    let (r, r_inv) = random_id(num_wires, rng.random_range(3..=25));

    for gate in &c.gates {
        // Generate a random identity r ⋅ r⁻¹
        // let (r, r_inv) = random_id(num_wires as u8, rng.random_range(3..=25), seed);

        // Add r
        obfuscated.gates.extend(&r.gates);

        // Record where r⁻¹ starts
        inverse_starts.push(obfuscated.gates.len());

        // Add r⁻¹
        obfuscated.gates.extend(&r_inv.gates);

        // Now add the original gate
        obfuscated.gates.push(*gate);
    }

    // Add a final padding random identity
    //let (r0, r0_inv) = random_id(num_wires as u8, rng.random_range(3..=5), seed);
    //obfuscated.gates.extend(&r0.gates);
    obfuscated.gates.extend(&r.gates);
    inverse_starts.push(obfuscated.gates.len());
    //obfuscated.gates.extend(&r0_inv.gates);
    obfuscated.gates.extend(&r_inv.gates);

    (obfuscated, inverse_starts)
}

pub fn compress_big_ancillas(
    c: &CircuitSeq,
    trials: usize,
    num_wires: usize,
    env: &lmdb::Environment,
    shard_dbs: &[lmdb::Database],
    mode: usize,
) -> CircuitSeq {
    let mut circuit = c.clone();
    let mut rng = rand::rng();

    let mut i = 0;
    while i < circuit.gates.len().saturating_sub(1) {
        if circuit.gates[i] == circuit.gates[i + 1] {
            circuit.gates.drain(i..=i + 1);
            i = i.saturating_sub(2);
        } else {
            i += 1;
        }
    }

    for _ in 0..trials {
        let t0 = Instant::now();
        let (mut subcircuit_gates, _) = match mode {
            0 => find_convex_subcircuit_max_wires(0, 30, num_wires, &circuit, &mut rng),
            2 => find_convex_subcircuit_max_gates(0, 21, num_wires, &circuit, &mut rng),
            _ => simple_find_convex_subcircuit(0, 30, num_wires, &circuit, &mut rng),
        };
        let elapsed = t0.elapsed().as_nanos() as u64;
        CONVEX_FIND_TIME.fetch_add(elapsed, Ordering::Relaxed);
        match mode {
            0 => CONVEX_MAX_WIRES_TIME.fetch_add(elapsed, Ordering::Relaxed),
            2 => CONVEX_MAX_GATES_TIME.fetch_add(elapsed, Ordering::Relaxed),
            _ => CONVEX_SIMPLE_TIME.fetch_add(elapsed, Ordering::Relaxed),
        };

        if subcircuit_gates.is_empty() {
            continue;
        }

        let gates: Vec<[u8; 3]> = subcircuit_gates.iter().map(|&g| circuit.gates[g]).collect();
        subcircuit_gates.sort();

        let t1 = Instant::now();
        let (start, end) = contiguous_convex(&mut circuit, &mut subcircuit_gates, num_wires).unwrap();
        CONTIGUOUS_TIME.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let mut subcircuit = CircuitSeq { gates };

        let expected_slice: Vec<_> = subcircuit_gates.iter().map(|&i| circuit.gates[i]).collect();
        let actual_slice = &circuit.gates[start..=end];
        if actual_slice != &expected_slice[..] {
            continue;
        }

        let mut used_wires = subcircuit.used_wires();
        let n_wires = used_wires.len();
        let max = num_wires;
        let new_wires = rng.random_range(n_wires..=max);
        if new_wires > n_wires {
            let mut count = n_wires;
            while count < new_wires {
                let random = rng.random_range(0..num_wires);
                if used_wires.contains(&(random as u8)) {
                    continue
                }
                used_wires.push(random as u8);
                count += 1;
            }
        }

        let sub_num_wires = used_wires.len();

        let t4 = Instant::now();
        let subcircuit_temp = compress_lmdb(&subcircuit, 10, sub_num_wires, env, shard_dbs, mode);
        COMPRESS_TIME.fetch_add(t4.elapsed().as_nanos() as u64, Ordering::Relaxed);

        subcircuit = subcircuit_temp;

        let t6 = Instant::now();
        let repl_len = subcircuit.gates.len();
        let old_len = end - start + 1;

        if repl_len == old_len {
            for i in 0..repl_len {
                circuit.gates[start + i] = subcircuit.gates[i];
            }
        } else if repl_len < old_len {
            for i in 0..repl_len {
                circuit.gates[start + i] = subcircuit.gates[i];
            }
            for i in (end + 1)..circuit.gates.len() {
                circuit.gates[i - (old_len - repl_len)] = circuit.gates[i];
            }
            circuit.gates.truncate(circuit.gates.len() - (old_len - repl_len));
        } else {
            panic!("Replacement grew, which is not allowed");
        }
        REPLACE_TIME.fetch_add(t6.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    let t7 = Instant::now();
    let mut i = 0;
    while i < circuit.gates.len().saturating_sub(1) {
        if circuit.gates[i] == circuit.gates[i + 1] {
            circuit.gates.drain(i..=i + 1);
            i = i.saturating_sub(2);
        } else {
            i += 1;
        }
    }
    DEDUP_TIME.fetch_add(t7.elapsed().as_nanos() as u64, Ordering::Relaxed);

    circuit
}

pub fn expand_big_ancillas<'a>(
    c: &CircuitSeq,
    trials: usize,
    num_wires: usize,
    env: &lmdb::Environment,
    shard_dbs: &[lmdb::Database],
    mode: usize,
    pair_mode: &ExpandPairMode<'a>,
) -> CircuitSeq {
    let mut circuit = c.clone();
    let mut rng = rand::rng();

    // Open completion DB once if we'll need it for 2-gate curated pairs.
    let comp_db: Option<lmdb::Database> = match pair_mode {
        ExpandPairMode::Curated => Some(
            env.open_db(Some("completion_m2"))
                .expect("completion_m2 DB not found — run build_completion_m2 first"),
        ),
        _ => None,
    };

    for _ in 0..trials {
        let t0 = Instant::now();
        let (mut subcircuit_gates, _) = match mode {
            0 => find_convex_subcircuit_max_wires(0, 30, num_wires, &circuit, &mut rng),
            2 => find_convex_subcircuit_max_gates(0, 21, num_wires, &circuit, &mut rng),
            _ => simple_find_convex_subcircuit(0, 30, num_wires, &circuit, &mut rng),
        };
        let elapsed = t0.elapsed().as_nanos() as u64;
        CONVEX_FIND_TIME.fetch_add(elapsed, Ordering::Relaxed);
        match mode {
            0 => CONVEX_MAX_WIRES_TIME.fetch_add(elapsed, Ordering::Relaxed),
            2 => CONVEX_MAX_GATES_TIME.fetch_add(elapsed, Ordering::Relaxed),
            _ => CONVEX_SIMPLE_TIME.fetch_add(elapsed, Ordering::Relaxed),
        };

        if subcircuit_gates.is_empty() {
            continue;
        }

        let gates: Vec<[u8; 3]> = subcircuit_gates.iter().map(|&g| circuit.gates[g]).collect();

        if gates.len() < 2 {
            continue;
        }

        subcircuit_gates.sort();

        let t1 = Instant::now();
        let (start, end) = contiguous_convex(&mut circuit, &mut subcircuit_gates, num_wires).unwrap();
        CONTIGUOUS_TIME.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let subcircuit = CircuitSeq { gates };

        let expected_slice: Vec<_> = subcircuit_gates.iter().map(|&i| circuit.gates[i]).collect();
        let actual_slice = &circuit.gates[start..=end];
        if actual_slice != &expected_slice[..] {
            continue;
        }

        // --- 2-gate path: use pair functions directly on circuit wire values ---
        if subcircuit.gates.len() == 2 {
            let t6 = Instant::now();
            let repl_opt: Option<Vec<[u8; 3]>> = match pair_mode {
                ExpandPairMode::Curated => {
                    comp_db.and_then(|db| {
                        replace_single_pair_with_completion(
                            &circuit.gates[start], &circuit.gates[end], num_wires, env, db,
                        )
                    })
                }
                ExpandPairMode::Canonical { bit_shuf_list, dbs, id_len, tower } => {
                    let (repl, _) = replace_single_pair(
                        &circuit.gates[start], &circuit.gates[end], num_wires, env,
                        bit_shuf_list, dbs, *tower, *id_len, &[], &[],
                    );
                    Some(repl)
                }
                ExpandPairMode::Db => None, // handled by expand_lmdb below
            };
            if let Some(repl) = repl_opt {
                if repl.len() > 2 {
                    circuit.gates.splice(start..=end, repl);
                }
                REPLACE_TIME.fetch_add(t6.elapsed().as_nanos() as u64, Ordering::Relaxed);
                continue;
            }
            // Db mode falls through to expand_lmdb
            REPLACE_TIME.fetch_add(t6.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }

        // --- 3-5 gate path (and 2-gate Db mode): use shard DB via expand_lmdb ---
        // Pass num_wires (full circuit wire count) so extra wires are assigned correctly.
        let t4 = Instant::now();
        let subcircuit_temp = expand_lmdb(&subcircuit, 10, num_wires, env, shard_dbs, &ExpandPairMode::Db);
        COMPRESS_TIME.fetch_add(t4.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t6 = Instant::now();
        let repl_len = subcircuit_temp.gates.len();
        let old_len = end - start + 1;

        if repl_len > old_len {
            circuit.gates.splice(start..=end, subcircuit_temp.gates);
        }
        REPLACE_TIME.fetch_add(t6.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    circuit
}

// Replace a single gate
pub fn random_gate_replacements(c: &mut CircuitSeq, x: usize, n: usize, env: &lmdb::Environment) {
    let mut rng = rand::rng();
    for _ in 0..x {
        if c.gates.is_empty() {
            break;
        }

        let i = rng.random_range(0..c.gates.len());
        let g = &c.gates[i];

        let num = rng.random_range(3..=7);
        if let Ok(mut id) = random_canonical_id(env, num) {
            let mut used_wires = vec![g[0], g[1], g[2]];
            let mut count = 3;
            while count < num {
                let random = rng.random_range(0..n);
                if used_wires.contains(&(random as u8)) {
                    continue
                }
                used_wires.push(random as u8);
                count += 1;
            }
            used_wires.sort();
            let rewired_g = CircuitSeq::rewire_subcircuit(&c, &vec![i], &used_wires);
            // println!("rewired_g {:?} vs len: {}", rewired_g, num);
            id.rewire_first_gate(rewired_g.gates[0], num);
            id = CircuitSeq::unrewire_subcircuit(&id, &used_wires);
            id.gates.remove(0);
            c.gates.splice(i..i+1, id.gates);
        } 
    }
}

// For timing and benchmarking purposes
pub fn print_compress_timers() {
    let canon = CANON_TIME.load(Ordering::Relaxed);
    let compress = COMPRESS_TIME.load(Ordering::Relaxed);
    let rewire = REWIRE_TIME.load(Ordering::Relaxed);
    let convex_find = CONVEX_FIND_TIME.load(Ordering::Relaxed);
    let convex_max_wires = CONVEX_MAX_WIRES_TIME.load(Ordering::Relaxed);
    let convex_max_gates = CONVEX_MAX_GATES_TIME.load(Ordering::Relaxed);
    let convex_simple = CONVEX_SIMPLE_TIME.load(Ordering::Relaxed);
    let contiguous = CONTIGUOUS_TIME.load(Ordering::Relaxed);
    let replace = REPLACE_TIME.load(Ordering::Relaxed);
    let dedup = DEDUP_TIME.load(Ordering::Relaxed);
    let canonicalize = CANONICALIZE_TIME.load(Ordering::Relaxed);
    let canon_max_wires = CANONICALIZE_TIME_MAX_WIRES.load(Ordering::Relaxed);
    let canon_simple = CANONICALIZE_TIME_SIMPLE.load(Ordering::Relaxed);
    let canon_max_gates = CANONICALIZE_TIME_MAX_GATES.load(Ordering::Relaxed);
    let txn = TXN_TIME.load(Ordering::Relaxed);
    let lmdb_lookup = LMDB_LOOKUP_TIME.load(Ordering::Relaxed);
    let from_blob = FROM_BLOB_TIME.load(Ordering::Relaxed);
    let trial = TRIAL_TIME.load(Ordering::Relaxed);
    let id = IDENTITY_TIME.load(Ordering::Relaxed);

    let ns = 60_000_000_000.0f64;
    println!("--- Compression Timing Totals (minutes) ---");
    println!("Canonicalization time: {:.2} min", canon as f64 / ns);
    println!("Compress LMDB time: {:.2} min", compress as f64 / ns);
    println!("Rewire subcircuit time: {:.2} min", rewire as f64 / ns);
    println!("Convex subcircuit find time: {:.2} min", convex_find as f64 / ns);
    println!("  max_wires: {:.2} min", convex_max_wires as f64 / ns);
    println!("  max_gates: {:.2} min", convex_max_gates as f64 / ns);
    println!("  simple:    {:.2} min", convex_simple as f64 / ns);
    println!("Contiguous convex subcircuit time: {:.2} min", contiguous as f64 / ns);
    println!("Replacement time: {:.2} min", replace as f64 / ns);
    println!("Deduplication time: {:.2} min", dedup as f64 / ns);
    println!("Subcircuit canonicalize time: {:.2} min", canonicalize as f64 / ns);
    println!("  max_wires: {:.2} min", canon_max_wires as f64 / ns);
    println!("  max_gates: {:.2} min", canon_max_gates as f64 / ns);
    println!("  simple:    {:.2} min", canon_simple as f64 / ns);
    println!("LMDB transaction begin time: {:.2} min", txn as f64 / ns);
    println!("LMDB lookup time: {:.2} min", lmdb_lookup as f64 / ns);
    println!("CircuitSeq from_blob time: {:.2} min", from_blob as f64 / ns);
    println!("Trial loop time: {:.2} min", trial as f64 / ns);
    println!("Identity Sampling Time: {:.2} min", id as f64 / ns);
}
