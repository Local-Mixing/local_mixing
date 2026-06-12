// Replacement code used in the mixing methods

use crate::replace::mixing::split_into_random_chunk_ranges;
use crate::{
    circuit::circuit::{CircuitSeq, Permutation},
    random::random_data::{
        contiguous_convex, find_convex_subcircuit_max_gates, find_convex_subcircuit_max_wires,
        simple_find_convex_subcircuit,
    },
};
use lmdb::Transaction;
use rand::Rng;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::fs::File;
use std::io::Write;

extern crate lmdb_sys;

use dashmap::DashMap;
use once_cell::sync::Lazy;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::{
    cmp::{max, min},
    time::Instant,
};
// use rand::prelude::IndexedRandom;

// Global histogram: (before_gates, after_gates) -> count, accumulated across all rounds
pub static COMPRESSION_HISTOGRAM: Lazy<DashMap<(u8, u8), u64>> = Lazy::new(DashMap::new);

// Global histograms for EXPANSIONS made in the shuffle-shoot-shuffle game, accumulated
// across all rounds. One is keyed by gate counts, the other by distinct-wire counts.
pub static EXPANSION_HISTOGRAM: Lazy<DashMap<(u8, u8), u64>> = Lazy::new(DashMap::new);
pub static EXPANSION_WIRE_HISTOGRAM: Lazy<DashMap<(u8, u8), u64>> = Lazy::new(DashMap::new);

// Record one expansion: `before`/`after` gate counts and `before_wires`/`after_wires`
// distinct-wire counts.
pub fn record_expansion(before: usize, after: usize, before_wires: usize, after_wires: usize) {
    *EXPANSION_HISTOGRAM
        .entry((before as u8, after as u8))
        .or_insert(0) += 1;
    *EXPANSION_WIRE_HISTOGRAM
        .entry((before_wires as u8, after_wires as u8))
        .or_insert(0) += 1;
}

// Write a (before, after) -> count histogram to `csv_path` and a human-readable log to
// `log_path`. `unit` labels the rows in the log (e.g. "gates" or "wires").
fn write_before_after_histogram(
    hist: &DashMap<(u8, u8), u64>,
    csv_path: &str,
    log_path: &str,
    unit: &str,
) {
    let mut entries: Vec<((u8, u8), u64)> = hist.iter().map(|e| (*e.key(), *e.value())).collect();
    entries.sort_by_key(|&((before, after), _)| (before, after));
    let mut f = File::create(csv_path).expect("Failed to create histogram CSV");
    writeln!(f, "before,after,count").expect("write");
    for ((before, after), count) in &entries {
        writeln!(f, "{},{},{}", before, after, count).expect("write");
    }
    println!("Histogram written to {}", csv_path);

    let mut log = File::create(log_path).expect("Failed to create histogram log");
    let before_vals: Vec<u8> = {
        let mut v: Vec<u8> = entries.iter().map(|&((b, _), _)| b).collect();
        v.dedup();
        v
    };
    for before in before_vals {
        let group: Vec<_> = entries.iter().filter(|&((b, _), _)| *b == before).collect();
        let total: u64 = group.iter().map(|(_, c)| c).sum();
        writeln!(log, "{} {} before (total: {}):", before, unit, total).expect("write");
        for ((_, after), count) in &group {
            writeln!(log, "  -> {} {}: {}", after, unit, count).expect("write");
        }
    }
    println!("Log written to {}", log_path);
}

pub fn write_expansion_histogram(path: &str) {
    write_before_after_histogram(&EXPANSION_HISTOGRAM, path, "expansion_log.txt", "gates");
}

pub fn write_expansion_wire_histogram(path: &str) {
    write_before_after_histogram(
        &EXPANSION_WIRE_HISTOGRAM,
        path,
        "expansion_wire_log.txt",
        "wires",
    );
}

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
        return (CircuitSeq { gates: Vec::new() }, 0, 0);
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

    let start = min(a, b);
    let end = max(a, b);

    let subcircuit = circuit.gates[start..end].to_vec();

    (CircuitSeq { gates: subcircuit }, start, end)
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
    let mut sub_len = 1 << shift; // 1,2,4,8
    if sub_len > allowed_len {
        sub_len = allowed_len;
    }

    sub_len = sub_len.max(1);

    let end = start + sub_len;
    let subcircuit = circuit.gates[start..end].to_vec();

    (CircuitSeq { gates: subcircuit }, start, end)
}

pub static CANON_TIME: AtomicU64 = AtomicU64::new(0);
pub static CONVEX_FIND_TIME: AtomicU64 = AtomicU64::new(0);
pub static CONVEX_MAX_WIRES_TIME: AtomicU64 = AtomicU64::new(0);
pub static CONVEX_MAX_GATES_TIME: AtomicU64 = AtomicU64::new(0);
pub static CONVEX_SIMPLE_TIME: AtomicU64 = AtomicU64::new(0);
pub static CONTIGUOUS_TIME: AtomicU64 = AtomicU64::new(0);
pub static REWIRE_TIME: AtomicU64 = AtomicU64::new(0);
pub static COMPRESS_TIME: AtomicU64 = AtomicU64::new(0);
pub static REPLACE_TIME: AtomicU64 = AtomicU64::new(0);
pub static DEDUP_TIME: AtomicU64 = AtomicU64::new(0);
pub static CANONICALIZE_TIME: AtomicU64 = AtomicU64::new(0);
pub static CANONICALIZE_TIME_MAX_WIRES: AtomicU64 = AtomicU64::new(0);
pub static CANONICALIZE_TIME_SIMPLE: AtomicU64 = AtomicU64::new(0);
pub static CANONICALIZE_TIME_MAX_GATES: AtomicU64 = AtomicU64::new(0);
pub static TXN_TIME: AtomicU64 = AtomicU64::new(0);
pub static LMDB_LOOKUP_TIME: AtomicU64 = AtomicU64::new(0);
pub static FROM_BLOB_TIME: AtomicU64 = AtomicU64::new(0);
pub static SPLICE_TIME: AtomicU64 = AtomicU64::new(0);
pub static TRIAL_TIME: AtomicU64 = AtomicU64::new(0);

fn compression_trace_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("COMPRESSION_TRACE").is_ok())
}

fn compression_trace_threshold_ms() -> u128 {
    static THRESHOLD: OnceLock<u128> = OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        std::env::var("COMPRESSION_TRACE_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1_000)
    })
}

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
    let mut mode = 0usize;
    // Ring buffer of the last stable_max+1 gate counts. Stop when total reduction
    // over the last stable_max iterations is less than 100 gates.
    let mut recent: std::collections::VecDeque<usize> =
        std::collections::VecDeque::with_capacity(stable_max + 1);
    recent.push_back(acc.gates.len());

    loop {
        let before = acc.gates.len();

        let max_chunks = 4 * rayon::current_num_threads().max(1);
        let k = if before <= 1500 {
            1
        } else {
            ((before + 1499) / 1500).min(max_chunks)
        };

        let current_mode = [1, 2][mode];
        mode = (mode + 1) % 2;

        let trace = compression_trace_enabled();
        let trace_threshold_ms = compression_trace_threshold_ms();
        let ranges = split_into_random_chunk_ranges(acc.gates.len(), k, &mut rng);
        let compressed_chunks: Vec<(usize, usize, usize, Vec<[u16; 3]>, u128)> = ranges
            .into_iter()
            .enumerate()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(chunk_idx, (start, end))| {
                let sub = CircuitSeq {
                    gates: acc.gates[start..end].to_vec(),
                };
                let chunk_start = Instant::now();
                let gates = compress_big_ancillas(&sub, 100, n, env, shard_dbs, current_mode).gates;
                let elapsed_ms = chunk_start.elapsed().as_millis();
                if trace && elapsed_ms >= trace_threshold_ms {
                    eprintln!(
                        "[compress-trace] slow chunk mode={} idx={}/{} in_gates={} out_gates={} elapsed_ms={}",
                        current_mode,
                        chunk_idx + 1,
                        k,
                        end - start,
                        gates.len(),
                        elapsed_ms
                    );
                }
                (chunk_idx, end - start, gates.len(), gates, elapsed_ms)
            })
            .collect();

        let mut compressed_chunks = compressed_chunks;
        compressed_chunks.sort_by_key(|(chunk_idx, _, _, _, _)| *chunk_idx);

        if trace {
            let total_chunk_ms: u128 = compressed_chunks.iter().map(|(_, _, _, _, ms)| *ms).sum();
            if let Some((slow_idx, slow_in, slow_out, _, slow_ms)) = compressed_chunks
                .iter()
                .max_by_key(|(_, _, _, _, elapsed_ms)| *elapsed_ms)
            {
                eprintln!(
                    "[compress-trace] iteration mode={} chunks={} before={} slowest_idx={} slowest_in={} slowest_out={} slowest_ms={} sum_chunk_ms={}",
                    current_mode,
                    k,
                    before,
                    slow_idx + 1,
                    slow_in,
                    slow_out,
                    slow_ms,
                    total_chunk_ms
                );
            }
        }

        let total_len: usize = compressed_chunks
            .iter()
            .map(|(_, _, out_len, _, _)| *out_len)
            .sum();
        let mut new_gates = Vec::with_capacity(total_len);
        for (_, _, _, chunk, _) in compressed_chunks {
            new_gates.extend(chunk);
        }

        acc.gates = new_gates;
        let after = acc.gates.len();

        recent.push_back(after);
        if recent.len() > stable_max + 1 {
            recent.pop_front();
        }

        // Stop if the total reduction over the last stable_max iterations is < 100.
        if recent.len() == stable_max + 1 {
            let window_reduction = recent.front().unwrap().saturating_sub(after);
            if window_reduction < 50 {
                println!(
                    "  {}/{}: Early stop — only {} gates reduced over last {} iterations ({} gates)",
                    curr_round, last_round, window_reduction, stable_max, after
                );
                break;
            }
        }

        if after == before {
            println!("  {}/{}: Stable ({} gates)", curr_round, last_round, after);
        } else {
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
    let k = if before <= 1500 {
        1
    } else {
        ((before + 1499) / 1500).min(max_chunks)
    };
    let ranges = split_into_random_chunk_ranges(before, k, &mut rng);
    let expanded_chunks: Vec<Vec<[u16; 3]>> = ranges
        .into_par_iter()
        .map(|(start, end)| {
            let sub = CircuitSeq {
                gates: circuit.gates[start..end].to_vec(),
            };
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

// Expand with ancilla wires or gates
/// Selects which method to use when a 2-gate subcircuit is sampled in the expand functions.
/// For subcircuits of 3–5 gates the shard DB is always used regardless of this setting.
pub enum ExpandPairMode<'a> {
    /// Use the curated shard DBs to find a longer equivalent pair.
    Curated {
        curated_shard_dbs: &'a [lmdb::Database],
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
    use crate::circuit::circuit::polys_repr_blob;
    use xxhash_rust::xxh3::xxh3_128;

    let mut expanded = c.clone();

    if expanded.gates.is_empty() {
        return CircuitSeq { gates: Vec::new() };
    }

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
                ExpandPairMode::Curated { curated_shard_dbs } => {
                    use crate::replace::pairs::expand_curated_lmdb;
                    if let Some(repl) =
                        expand_curated_lmdb(&sub.gates, n, env, curated_shard_dbs, shard_dbs)
                    {
                        if repl.len() > 2 {
                            expanded.gates.splice(start..end, repl);
                        }
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

        let fwd_key = xxh3_128(&polys_repr_blob(&fwd_polys))
            .to_le_bytes()
            .to_vec();
        let fwd_shard = fwd_key[0] as usize;

        let t_txn = Instant::now();
        let txn = match env.begin_ro_txn() {
            Ok(t) => t,
            Err(_) => continue,
        };
        TXN_TIME.fetch_add(t_txn.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t_lookup = Instant::now();
        let fwd_result = txn
            .get(shard_dbs[fwd_shard], &fwd_key)
            .map(|v: &[u8]| v.to_vec());
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

            let rev_key = xxh3_128(&polys_repr_blob(&rev_polys))
                .to_le_bytes()
                .to_vec();
            let rev_shard = rev_key[0] as usize;

            let t_lookup2 = Instant::now();
            let rev_result = txn
                .get(shard_dbs[rev_shard], &rev_key)
                .map(|v: &[u8]| v.to_vec());
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
            if pos + 1 > value.len() {
                break;
            }
            let len = value[pos] as usize;
            pos += 1;
            if pos + len > value.len() {
                break;
            }
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
        let mut best: Vec<CircuitSeq> = candidates
            .into_iter()
            .filter(|c| c.gates.len() == max_gates)
            .collect();
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

        repl.rewire(
            &Permutation { data: order_data },
            std::cmp::max(repl_n, final_order.data.len()),
        );

        let repl_n_b = repl.max_wire() + 1;
        let mut used_ext = used.clone();
        if used_ext.len() < repl_n_b {
            let mut available: Vec<u16> = (0..n as u16).filter(|w| !used_ext.contains(w)).collect();
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

pub fn compress_lmdb(
    c: &CircuitSeq,
    trials: usize,
    n: usize,
    env: &lmdb::Environment,
    shard_dbs: &[lmdb::Database],
    mode: usize,
) -> CircuitSeq {
    use crate::circuit::circuit::polys_repr_blob;
    use xxhash_rust::xxh3::xxh3_128;

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
        if compression_trace_enabled()
            && (canon_elapsed as u128 / 1_000_000) >= compression_trace_threshold_ms()
        {
            eprintln!(
                "[compress-trace] slow inner-canon mode={} direction=forward parent_gates={} inner_start={} inner_end={} inner_gates={} inner_wires={} elapsed_ms={}",
                mode,
                compressed.gates.len(),
                start,
                end,
                sub.gates.len(),
                sub.used_wires().len(),
                canon_elapsed as u128 / 1_000_000
            );
        }

        if fwd_polys.is_empty() {
            continue;
        }

        let fwd_key = xxh3_128(&polys_repr_blob(&fwd_polys))
            .to_le_bytes()
            .to_vec();
        let fwd_shard = fwd_key[0] as usize;

        let t_txn = Instant::now();
        let txn = match env.begin_ro_txn() {
            Ok(t) => t,
            Err(_) => continue,
        };
        TXN_TIME.fetch_add(t_txn.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t_lookup = Instant::now();
        let fwd_result = txn
            .get(shard_dbs[fwd_shard], &fwd_key)
            .map(|v: &[u8]| v.to_vec());
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
            if compression_trace_enabled()
                && (canon2_elapsed as u128 / 1_000_000) >= compression_trace_threshold_ms()
            {
                eprintln!(
                    "[compress-trace] slow inner-canon mode={} direction=reverse parent_gates={} inner_start={} inner_end={} inner_gates={} inner_wires={} elapsed_ms={}",
                    mode,
                    compressed.gates.len(),
                    start,
                    end,
                    sub.gates.len(),
                    sub.used_wires().len(),
                    canon2_elapsed as u128 / 1_000_000
                );
            }

            if rev_polys.is_empty() {
                continue;
            }

            let rev_key = xxh3_128(&polys_repr_blob(&rev_polys))
                .to_le_bytes()
                .to_vec();
            let rev_shard = rev_key[0] as usize;

            let t_lookup2 = Instant::now();
            let rev_result = txn
                .get(shard_dbs[rev_shard], &rev_key)
                .map(|v: &[u8]| v.to_vec());
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
            if pos + 1 > value.len() {
                break;
            }
            let len = value[pos] as usize;
            pos += 1;
            if pos + len > value.len() {
                break;
            }
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
        *COMPRESSION_HISTOGRAM
            .entry((sub.gates.len() as u8, min_gates as u8))
            .or_insert(0) += 1;
        let mut best: Vec<CircuitSeq> = candidates
            .into_iter()
            .filter(|c| c.gates.len() == min_gates)
            .collect();
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
        repl.rewire(
            &Permutation { data: order_data },
            std::cmp::max(repl_n, final_order.data.len()),
        );

        let repl_n_b = repl.max_wire() + 1;
        let mut used_ext = used.clone();
        if used_ext.len() < repl_n_b {
            let mut available: Vec<u16> = (0..n as u16).filter(|w| !used_ext.contains(w)).collect();
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
            0 => find_convex_subcircuit_max_wires(30, num_wires, &circuit, &mut rng),
            2 => find_convex_subcircuit_max_gates(21, num_wires, &circuit, &mut rng),
            _ => simple_find_convex_subcircuit(num_wires, &circuit, &mut rng),
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

        let gates: Vec<[u16; 3]> = subcircuit_gates.iter().map(|&g| circuit.gates[g]).collect();
        subcircuit_gates.sort();

        let t1 = Instant::now();
        let cc = contiguous_convex(&mut circuit, &mut subcircuit_gates, num_wires);
        CONTIGUOUS_TIME.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let (start, end) = match cc {
            Some(se) => se,
            None => continue,
        };

        let mut subcircuit = CircuitSeq { gates };

        let mut used_wires = subcircuit.used_wires();
        let n_wires = used_wires.len();
        let max = num_wires;
        let new_wires = rng.random_range(n_wires..=max);
        if new_wires > n_wires {
            let mut count = n_wires;
            while count < new_wires {
                let random = rng.random_range(0..num_wires);
                if used_wires.contains(&(random as u16)) {
                    continue;
                }
                used_wires.push(random as u16);
                count += 1;
            }
        }

        let sub_num_wires = used_wires.len();

        let t4 = Instant::now();
        let sub_gates = subcircuit.gates.len();
        let subcircuit_temp = compress_lmdb(&subcircuit, 10, sub_num_wires, env, shard_dbs, mode);
        let compress_elapsed = t4.elapsed();
        COMPRESS_TIME.fetch_add(compress_elapsed.as_nanos() as u64, Ordering::Relaxed);
        if compression_trace_enabled()
            && compress_elapsed.as_millis() >= compression_trace_threshold_ms()
        {
            eprintln!(
                "[compress-trace] slow compress_lmdb mode={} outer_gates={} outer_wires={} outer_span={} out_gates={} elapsed_ms={}",
                mode,
                sub_gates,
                sub_num_wires,
                end - start + 1,
                subcircuit_temp.gates.len(),
                compress_elapsed.as_millis()
            );
        }

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
            circuit
                .gates
                .truncate(circuit.gates.len() - (old_len - repl_len));
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

    for _ in 0..trials {
        let t0 = Instant::now();
        let (mut subcircuit_gates, _) = match mode {
            0 => find_convex_subcircuit_max_wires(30, num_wires, &circuit, &mut rng),
            2 => find_convex_subcircuit_max_gates(21, num_wires, &circuit, &mut rng),
            _ => simple_find_convex_subcircuit(num_wires, &circuit, &mut rng),
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

        let gates: Vec<[u16; 3]> = subcircuit_gates.iter().map(|&g| circuit.gates[g]).collect();

        if gates.len() < 2 {
            continue;
        }

        subcircuit_gates.sort();

        let t1 = Instant::now();
        let cc = contiguous_convex(&mut circuit, &mut subcircuit_gates, num_wires);
        CONTIGUOUS_TIME.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let (start, end) = match cc {
            Some(se) => se,
            None => continue,
        };

        let subcircuit = CircuitSeq { gates };

        // --- 2-gate path: use pair functions directly on circuit wire values ---
        if subcircuit.gates.len() == 2 {
            let t6 = Instant::now();
            let repl_opt: Option<Vec<[u16; 3]>> = match pair_mode {
                ExpandPairMode::Curated { curated_shard_dbs } => {
                    use crate::replace::pairs::expand_curated_lmdb;
                    expand_curated_lmdb(
                        &[circuit.gates[start], circuit.gates[end]],
                        num_wires,
                        env,
                        curated_shard_dbs,
                        shard_dbs,
                    )
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
        let subcircuit_temp = expand_lmdb(
            &subcircuit,
            10,
            num_wires,
            env,
            shard_dbs,
            &ExpandPairMode::Db,
        );
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

// For timing and benchmarking purposes
pub fn print_compress_timers() {
    use crate::circuit::circuit::{
        CANON4_RULE_L_BRANCHES, CANON4_RULE_L_CALLS, CANON4_RULE_L_TIME,
    };
    use crate::replace::transpositions::{SAMF_COMPRESSIONS_FAILED, SAMF_COMPRESSIONS_MADE};

    let canon = CANON_TIME.load(Ordering::Relaxed);
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
    let rule_l_time = CANON4_RULE_L_TIME.load(Ordering::Relaxed);
    let rule_l_calls = CANON4_RULE_L_CALLS.load(Ordering::Relaxed);
    let rule_l_branches = CANON4_RULE_L_BRANCHES.load(Ordering::Relaxed);

    let samf_made = SAMF_COMPRESSIONS_MADE.load(Ordering::Relaxed);
    let samf_failed = SAMF_COMPRESSIONS_FAILED.load(Ordering::Relaxed);

    let ns = 60_000_000_000.0f64;
    let threshold_ns = 15.0 * ns; // 15 minutes in nanoseconds

    println!("--- Compression Timing Totals (minutes) ---");
    if canon as f64 >= threshold_ns {
        println!("Canonicalization time: {:.2} min", canon as f64 / ns);
    }
    if rewire as f64 >= threshold_ns {
        println!("Rewire subcircuit time: {:.2} min", rewire as f64 / ns);
    }
    if convex_find as f64 >= threshold_ns {
        println!(
            "Convex subcircuit find time: {:.2} min",
            convex_find as f64 / ns
        );
        if convex_max_wires as f64 >= threshold_ns {
            println!("  max_wires: {:.2} min", convex_max_wires as f64 / ns);
        }
        if convex_max_gates as f64 >= threshold_ns {
            println!("  max_gates: {:.2} min", convex_max_gates as f64 / ns);
        }
        if convex_simple as f64 >= threshold_ns {
            println!("  simple:    {:.2} min", convex_simple as f64 / ns);
        }
    }
    if contiguous as f64 >= threshold_ns {
        println!(
            "Contiguous convex subcircuit time: {:.2} min",
            contiguous as f64 / ns
        );
    }
    if replace as f64 >= threshold_ns {
        println!("Replacement time: {:.2} min", replace as f64 / ns);
    }
    if dedup as f64 >= threshold_ns {
        println!("Deduplication time: {:.2} min", dedup as f64 / ns);
    }
    if canonicalize as f64 >= threshold_ns {
        println!(
            "Subcircuit canonicalize time: {:.2} min",
            canonicalize as f64 / ns
        );
        if canon_max_wires as f64 >= threshold_ns {
            println!("  max_wires: {:.2} min", canon_max_wires as f64 / ns);
        }
        if canon_max_gates as f64 >= threshold_ns {
            println!("  max_gates: {:.2} min", canon_max_gates as f64 / ns);
        }
        if canon_simple as f64 >= threshold_ns {
            println!("  simple:    {:.2} min", canon_simple as f64 / ns);
        }
    }
    if txn as f64 >= threshold_ns {
        println!("LMDB transaction begin time: {:.2} min", txn as f64 / ns);
    }
    if lmdb_lookup as f64 >= threshold_ns {
        println!("LMDB lookup time: {:.2} min", lmdb_lookup as f64 / ns);
    }
    if from_blob as f64 >= threshold_ns {
        println!(
            "CircuitSeq from_blob time: {:.2} min",
            from_blob as f64 / ns
        );
    }
    if trial as f64 >= threshold_ns {
        println!("Trial loop time: {:.2} min", trial as f64 / ns);
    }
    if rule_l_time as f64 >= threshold_ns || std::env::var("COMPRESSION_TRACE").is_ok() {
        let seconds = rule_l_time as f64 / 1e9;
        let avg_ms = if rule_l_calls == 0 {
            0.0
        } else {
            rule_l_time as f64 / 1e6 / rule_l_calls as f64
        };
        let avg_branches = if rule_l_calls == 0 {
            0.0
        } else {
            rule_l_branches as f64 / rule_l_calls as f64
        };
        println!(
            "Rule L time: {:.2} s  calls: {}  avg_ms: {:.2}  avg_branches: {:.2}",
            seconds, rule_l_calls, avg_ms, avg_branches
        );
    }
    println!(
        "SAMF compressions made: {}  failed: {}",
        samf_made, samf_failed
    );

    if std::env::var("BENCH_CANON").is_ok() {
        use crate::circuit::circuit::{CANON_BENCH_CALLS, CANON4_CORE_TIME, POLYCANON_CORE_TIME};

        let c4 = CANON4_CORE_TIME.load(Ordering::Relaxed) as f64;
        let pc = POLYCANON_CORE_TIME.load(Ordering::Relaxed) as f64;
        let calls = CANON_BENCH_CALLS.load(Ordering::Relaxed).max(1) as f64;
        println!("--- Canonicalization benchmark (BENCH_CANON) ---");
        println!("matched calls:    {}", calls as u64);
        println!(
            "canon4 total:     {:.3} s   ({:.1} us/call)",
            c4 / 1e9,
            c4 / 1e3 / calls
        );
        println!(
            "polycanon total:  {:.3} s   ({:.1} us/call)",
            pc / 1e9,
            pc / 1e3 / calls
        );
        println!("ratio poly/canon4:{:.2}x", pc / c4.max(1.0));
    }
}
