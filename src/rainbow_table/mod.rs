//! Rainbow-table (replacement database) generation pipeline.
//!
//! Ported 2026-07-06 from the dbgen server worktree (`/mnt/dbgen/local_mixing`,
//! `src/random/random_data.rs`), including the server-side optimizations made
//! that day: fresh-wire dedup in `abstract_gates_for_circuit_filtered`
//! (rocksdb_1), capped mapping enumeration in `build_from_2rocks` (rocksdb_2,
//! `for_each_mapping_capped`), PID-qualified temp SST names, and the
//! stop-flag-before-println Ctrl+C handler fix.
//!
//! Pipeline stages (see also the `rocksdb_1` / `rocksdb_2` / `combine_rocks` /
//! `rocks_to_lmdb` CLI subcommands and the `build_curated` / `curated_to_lmdb`
//! bins):
//!   1. `build_m1` — base case: all canonical 1-gate circuits.
//!   2. `build_from_rocks` (rocksdb_1) — extend an m-1 DB by one gate.
//!   3. `build_from_2rocks` (rocksdb_2) — combine an m1 DB and an m2 DB into
//!      an (m1+m2) DB over all wire overlaps.
//!   4. `combine_rocks_dbs` — merge rocks_db_m1..=m9 into one keyed DB.
//!   5. `rocks_to_lmdb` — convert to the sharded LMDB the mixing code reads.
//!
//! Deliberately NOT ported: the legacy LMDB-direct generator (`main_random`)
//! and the SQL/duckdb paths — the corrected pipeline does not use them.

#[cfg(test)]
mod validation_tests;

use crate::circuit::circuit::{
    CircuitSeq, base_gates, canonicalize_polys, canonicalize_polys_4, polys_repr_blob,
    print_rule_times,
};
use crossbeam_channel::bounded;
use itertools::Itertools;
use once_cell::sync::Lazy;
use rayon::prelude::*;
use rocksdb::{
    BlockBasedOptions, Cache, DB, DBCompressionType, IngestExternalFileOptions, MergeOperands,
    Options, SstFileWriter,
};
use smallvec::SmallVec;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use xxhash_rust::xxh3::xxh3_128;

fn write_error(msg: &str) {
    eprintln!("{}", msg);
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open("error.txt") {
        let _ = writeln!(f, "{}", msg);
    }
}


fn append_merge(
    _key: &[u8],
    existing: Option<&[u8]>,
    operands: &MergeOperands,
) -> Option<Vec<u8>> {
    let mut result: Vec<u8> = existing.unwrap_or(&[]).to_vec();

    for operand in operands {
        let mut pos = 0;
        while pos + 1 <= operand.len() {
            let len = operand[pos] as usize;
            pos += 1;
            if pos + len > operand.len() {
                break;
            }
            let new_blob = &operand[pos..pos + len];
            pos += len;

            // Check for duplicate in result
            let mut rpos = 0;
            let mut found = false;
            while rpos + 1 <= result.len() {
                let rlen = result[rpos] as usize;
                rpos += 1;
                if rpos + rlen > result.len() {
                    break;
                }
                if &result[rpos..rpos + rlen] == new_blob {
                    found = true;
                    break;
                }
                rpos += rlen;
            }

            if !found {
                result.push(new_blob.len() as u8);
                result.extend_from_slice(new_blob);
            }
        }
    }

    Some(result)
}

pub fn open_db_for_write(m: usize) -> DB {
    let path = format!("test_rocks_db_m{}", m);
    let mut opts = Options::default();
    opts.create_if_missing(true);

    opts.set_merge_operator_associative("append_merge", append_merge);

    // Disable WAL for faster bulk ingestion — no recovery needed
    opts.set_manual_wal_flush(true);

    opts.increase_parallelism(num_cpus::get() as i32);
    opts.set_max_background_jobs(64);
    opts.set_max_open_files(-1);

    opts.set_write_buffer_size(256 * 1024 * 1024);
    opts.set_max_write_buffer_number(4);
    opts.set_min_write_buffer_number_to_merge(2);

    opts.set_level_zero_file_num_compaction_trigger(10);
    opts.set_max_bytes_for_level_base(512 * 1024 * 1024);
    opts.set_max_bytes_for_level_multiplier(10.0);
    opts.set_num_levels(7);

    opts.set_compression_type(DBCompressionType::Zstd);
    opts.set_bottommost_compression_type(DBCompressionType::Zstd);

    // 16 byte prefix for xxHash128
    opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(16));

    let mut block_opts = BlockBasedOptions::default();
    block_opts.set_bloom_filter(10.0, false);
    block_opts.set_block_size(16 * 1024);
    block_opts.set_cache_index_and_filter_blocks(true);
    opts.set_block_based_table_factory(&block_opts);

    DB::open(&opts, path).expect("Failed to open RocksDB for write")
}

pub fn open_db_for_read(m: usize) -> DB {
    let path = format!("rocks_db_m{}", m);
    let mut opts = Options::default();
    opts.create_if_missing(false);

    // Must register merge operator even for reads
    opts.set_merge_operator_associative("append_merge", append_merge);

    opts.increase_parallelism(num_cpus::get() as i32);

    opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(16));

    let cache = Cache::new_lru_cache(4 * 1024 * 1024 * 1024);
    let mut block_opts = BlockBasedOptions::default();
    block_opts.set_block_cache(&cache);
    block_opts.set_block_size(16 * 1024);
    block_opts.set_bloom_filter(10.0, false);
    block_opts.set_cache_index_and_filter_blocks(true);
    block_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);
    opts.set_block_based_table_factory(&block_opts);

    opts.set_disable_auto_compactions(true);

    DB::open_for_read_only(&opts, path, false).expect("Failed to open RocksDB for read")
}

/// Encode a single circuit blob as a length-prefixed entry
fn encode_circuit(circuit_blob: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(1 + circuit_blob.len());
    v.push(circuit_blob.len() as u8);
    v.extend_from_slice(circuit_blob);
    v
}

/// Merge duplicate keys in a sorted list, deduplicating circuit blobs
fn merge_sorted_entries(entries: Vec<(Vec<u8>, Vec<u8>)>) -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut merged: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();

    for (key, value) in entries {
        if let Some(last) = merged.last_mut() {
            if last.0 == key {
                // value is [u8 len | blob], extract the blob
                if value.is_empty() {
                    continue;
                }
                let new_len = value[0] as usize;
                if 1 + new_len > value.len() {
                    continue;
                }
                let new_blob = &value[1..1 + new_len];

                // Scan existing blobs for duplicate
                let mut rpos = 0;
                let mut found = false;
                while rpos + 1 <= last.1.len() {
                    let rlen = last.1[rpos] as usize;
                    rpos += 1;
                    if rpos + rlen > last.1.len() {
                        break;
                    }
                    if &last.1[rpos..rpos + rlen] == new_blob {
                        found = true;
                        break;
                    }
                    rpos += rlen;
                }

                if !found {
                    last.1.push(new_len as u8);
                    last.1.extend_from_slice(new_blob);
                }
                continue;
            }
        }
        merged.push((key, value));
    }

    merged
}

fn flush_to_sst(db: &Arc<DB>, pending: &mut Vec<(Vec<u8>, Vec<u8>)>, sst_index: &mut usize) -> Result<(), Box<dyn std::error::Error>> {
    if pending.is_empty() {
        return Ok(());
    }

    pending.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
    let merged = merge_sorted_entries(std::mem::take(pending));

    // Use /dev/shm (tmpfs, separate from DB disk) to avoid filling /dev/sda3.
    // PID-qualified so concurrent builds (e.g. rocksdb_1 and rocksdb_2 running
    // side by side) can never collide on temp SST names.
    let sst_path = format!("/dev/shm/sst_{}_{}.sst", std::process::id(), sst_index);
    *sst_index += 1;

    let mut opts = Options::default();
    opts.set_merge_operator_associative("append_merge", append_merge);
    opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(16));
    opts.set_compression_type(DBCompressionType::Zstd);

    let mut writer = SstFileWriter::create(&opts);
    writer.open(&sst_path)?;

    for (key, value) in &merged {
        if let Err(e) = writer.put(key, value) {
            let _ = std::fs::remove_file(&sst_path);
            return Err(e.into());
        }
    }
    if let Err(e) = writer.finish() {
        let _ = std::fs::remove_file(&sst_path);
        return Err(e.into());
    }

    let mut ingest_opts = IngestExternalFileOptions::default();
    ingest_opts.set_move_files(false);
    if let Err(e) = db.ingest_external_file_opts(&ingest_opts, vec![sst_path.clone()]) {
        let _ = std::fs::remove_file(&sst_path);
        return Err(e.into());
    }

    let _ = std::fs::remove_file(&sst_path);
    println!("Ingested SST file #{}", *sst_index - 1);
    Ok(())
}

/// Returns the set of wires actually touched by the circuit (appearing in any gate).
fn touched_wires(circuit: &CircuitSeq) -> Vec<u16> {
    let mut touched: Vec<u16> = Vec::new();
    for gate in &circuit.gates {
        for &w in gate.iter() {
            if !touched.contains(&w) {
                touched.push(w);
            }
        }
    }
    touched.sort();
    touched
}

/// Expand an abstract gate (possibly containing UNUSED sentinel) into concrete gates
/// by substituting actual unused wires into the UNUSED slots.
/// UNUSED slots are filled with ordered distinct selections from `untouched`.
fn expand_abstract_gate(gate: [u16; 3], untouched: &[u16]) -> Vec<[u16; 3]> {
    const UNUSED: u16 = 512;
    let slots: Vec<usize> = gate
        .iter()
        .enumerate()
        .filter(|(_, w)| **w == UNUSED)
        .map(|(i, _)| i)
        .collect();

    match slots.len() {
        0 => vec![gate],
        1 => untouched
            .iter()
            .map(|&u0| {
                let mut g = gate;
                g[slots[0]] = u0;
                g
            })
            .collect(),
        2 => {
            let mut result = Vec::new();
            for &u0 in untouched {
                for &u1 in untouched {
                    if u1 == u0 {
                        continue;
                    }
                    let mut g = gate;
                    g[slots[0]] = u0;
                    g[slots[1]] = u1;
                    result.push(g);
                }
            }
            result
        }
        3 => {
            let mut result = Vec::new();
            for &u0 in untouched {
                for &u1 in untouched {
                    if u1 == u0 {
                        continue;
                    }
                    for &u2 in untouched {
                        if u2 == u0 || u2 == u1 {
                            continue;
                        }
                        let mut g = gate;
                        g[slots[0]] = u0;
                        g[slots[1]] = u1;
                        g[slots[2]] = u2;
                        result.push(g);
                    }
                }
            }
            result
        }
        _ => unreachable!(),
    }
}

/// For a given circuit, enumerate all concrete gates worth trying when
/// appending or prepending a gate. Exploits the symmetry that all untouched
/// wires are equivalent, collapsing them into one representative (UNUSED sentinel)
/// for enumeration then expanding back to concrete gates.
///
/// For a circuit touching k wires out of n total (with n-k untouched), the
/// number of abstract options is:
///   k*(k-1)*(k-2)          -- all three wires are touched
///   + k*(k-1)              -- two touched, one untouched
///   + k                    -- one touched, two untouched
///   + 1                    -- all three untouched (if n-k >= 3)
/// Each abstract option expands to 1, (n-k), (n-k)*(n-k-1), or (n-k)*(n-k-1)*(n-k-2)
/// concrete gates respectively.
pub fn abstract_gates_for_circuit(circuit: &CircuitSeq, n: usize) -> Vec<[u16; 3]> {
    const UNUSED: u16 = 512;

    let touched = touched_wires(circuit);
    let untouched: Vec<u16> = (0..n as u16)
        .filter(|w| !touched.contains(w))
        .collect();

    let mut result = Vec::new();

    // 0 UNUSED slots: all three wires are touched
    for &a in &touched {
        for &b in &touched {
            if b == a { continue; }
            for &c in &touched {
                if c == a || c == b { continue; }
                result.push([a, b, c]);
            }
        }
    }

    if !untouched.is_empty() {
        // 1 UNUSED slot: exactly one wire is untouched, two are touched
        // UNUSED in position a
        for &b in &touched {
            for &c in &touched {
                if c == b { continue; }
                result.extend(expand_abstract_gate([UNUSED, b, c], &untouched));
            }
        }
        // UNUSED in position b
        for &a in &touched {
            for &c in &touched {
                if c == a { continue; }
                result.extend(expand_abstract_gate([a, UNUSED, c], &untouched));
            }
        }
        // UNUSED in position c
        for &a in &touched {
            for &b in &touched {
                if b == a { continue; }
                result.extend(expand_abstract_gate([a, b, UNUSED], &untouched));
            }
        }
    }

    if untouched.len() >= 2 {
        // 2 UNUSED slots: two wires are untouched, one is touched
        // UNUSED in positions b and c
        for &a in &touched {
            result.extend(expand_abstract_gate([a, UNUSED, UNUSED], &untouched));
        }
        // UNUSED in positions a and c
        for &b in &touched {
            result.extend(expand_abstract_gate([UNUSED, b, UNUSED], &untouched));
        }
        // UNUSED in positions a and b
        for &c in &touched {
            result.extend(expand_abstract_gate([UNUSED, UNUSED, c], &untouched));
        }
    }

    if untouched.len() >= 3 {
        // 3 UNUSED slots: all three wires are untouched
        result.extend(expand_abstract_gate([UNUSED, UNUSED, UNUSED], &untouched));
    }

    result
}


fn ordered2(n: usize) -> usize {
    if n >= 2 { n * (n - 1) } else { 0 }
}

fn ordered3(n: usize) -> usize {
    if n >= 3 { n * (n - 1) * (n - 2) } else { 0 }
}

/// Like abstract_gates_for_circuit, but skips whole gate classes that cannot
/// satisfy the final wire-count bounds after adding this gate. The returned
/// skip count is the number of single-gate candidates omitted; callers that
/// try both append and prepend should count it twice.
///
/// Fresh-wire dedup: wires not touched by the circuit are interchangeable —
/// any two concrete assignments of fresh wires to the same abstract gate slots
/// yield circuits that are relabelings of each other, so canonicalize_polys
/// maps them to identical (key, value) pairs. Only one representative per
/// abstract class is emitted; the remaining assignments are counted in the
/// skip total so caller-side progress accounting still sums to base_gates(n).
pub fn abstract_gates_for_circuit_filtered(
    circuit: &CircuitSeq,
    n: usize,
    min_n: usize,
    max_n: usize,
) -> (Vec<[u16; 3]>, usize) {
    let touched = touched_wires(circuit);
    let untouched: Vec<u16> = (0..n as u16)
        .filter(|w| !touched.contains(w))
        .collect();

    let old_used = touched.len();
    let fresh = untouched.len();
    let mut result = Vec::new();
    let mut skipped = 0usize;

    let allowed = |new_wires: usize| {
        let used = old_used + new_wires;
        used >= min_n && (max_n == 0 || used <= max_n)
    };

    // 0 fresh wires: every concrete gate is a distinct class.
    if allowed(0) {
        for &a in &touched {
            for &b in &touched {
                if b == a { continue; }
                for &c in &touched {
                    if c == a || c == b { continue; }
                    result.push([a, b, c]);
                }
            }
        }
    } else {
        skipped += ordered3(old_used);
    }

    // 1 fresh wire: 3 * ordered2(old_used) classes, `fresh` assignments each.
    if fresh >= 1 {
        let count = 3 * ordered2(old_used) * fresh;
        if allowed(1) {
            let u0 = untouched[0];
            for &b in &touched {
                for &c in &touched {
                    if c == b { continue; }
                    result.push([u0, b, c]);
                }
            }
            for &a in &touched {
                for &c in &touched {
                    if c == a { continue; }
                    result.push([a, u0, c]);
                }
            }
            for &a in &touched {
                for &b in &touched {
                    if b == a { continue; }
                    result.push([a, b, u0]);
                }
            }
            skipped += count - 3 * ordered2(old_used);
        } else {
            skipped += count;
        }
    }

    // 2 fresh wires: 3 * old_used classes, ordered2(fresh) assignments each.
    if fresh >= 2 {
        let count = 3 * old_used * ordered2(fresh);
        if allowed(2) {
            let (u0, u1) = (untouched[0], untouched[1]);
            for &a in &touched {
                result.push([a, u0, u1]);
            }
            for &b in &touched {
                result.push([u0, b, u1]);
            }
            for &c in &touched {
                result.push([u0, u1, c]);
            }
            skipped += count - 3 * old_used;
        } else {
            skipped += count;
        }
    }

    // 3 fresh wires: a single class with ordered3(fresh) assignments.
    if fresh >= 3 {
        let count = ordered3(fresh);
        if allowed(3) {
            result.push([untouched[0], untouched[1], untouched[2]]);
            skipped += count - 1;
        } else {
            skipped += count;
        }
    }

    (result, skipped)
}

pub fn build_from_rocks(
    old_db: &Arc<DB>,
    new_db: &Arc<DB>,
    m: usize,
    min_n: usize,
    max_n: usize,
    no_rule_l: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Running build (max CPU)");

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build_global()
        .unwrap();

    let total_rows = old_db
        .property_int_value("rocksdb.estimate-num-keys")
        .unwrap()
        .unwrap_or(0);
    println!("Estimated rows: {}", total_rows);

    let chunk_size = 500_000;
    let batch_size = 10_000;

    let upper_bound_gates = base_gates(3 * m).len();
    let total_gates_tried = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let skipped_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let no_rule_l_skipped = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let stop_flag = Arc::new(AtomicBool::new(false));
    {
        let stop_flag = stop_flag.clone();
        ctrlc::set_handler(move || {
            println!("CTRL+C detected! Finishing current batch...");
            stop_flag.store(true, Ordering::SeqCst);
        })
        .expect("Error setting CTRL+C handler");
    }

    let (tx, rx) = bounded::<Vec<(Vec<u8>, Vec<u8>)>>(1_000);
    let stop_flag_clone = stop_flag.clone();
    let new_db_writer = Arc::clone(new_db);
    let skipped_count_insert = Arc::clone(&skipped_count);

    let insert_handle = std::thread::spawn(move || {
        let start_time = std::time::Instant::now();
        let mut attempted_inserts = 0usize;
        let mut sst_index = 0usize;
        let mut pending: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        // Batches arrive far more often than progress is useful; throttle to ~1 line/s.
        let mut last_progress = std::time::Instant::now();
        let mut first_progress = true;

        while let Ok(batch) = rx.recv() {
            if stop_flag_clone.load(Ordering::SeqCst) {
                println!("Insertion thread stopping early...");
                break;
            }

            let batch_len = batch.len();
            for (key, value) in batch {
                pending.push((key, value));
            }

            attempted_inserts += batch_len;
            if first_progress || last_progress.elapsed().as_secs_f64() >= 1.0 {
                first_progress = false;
                last_progress = std::time::Instant::now();
                let skipped = skipped_count_insert.load(Ordering::Relaxed);
                let done = attempted_inserts + skipped;
                let elapsed = start_time.elapsed().as_secs_f64();
                // Estimate input rows processed: each row yields up to upper_bound_gates*2 outputs.
                let rows_done = done / upper_bound_gates.max(1) / 2 + 1;
                let rate_rows = if elapsed > 0.0 { rows_done as f64 / elapsed } else { 0.0 };
                let pct = if total_rows > 0 { rows_done as f64 / total_rows as f64 * 100.0 } else { 0.0 };
                let remaining = if rate_rows > 0.0 {
                    (total_rows.saturating_sub(rows_done as u64)) as f64 / rate_rows
                } else {
                    f64::INFINITY
                };
                let remaining_secs = remaining as u64;
                let remaining_h = remaining_secs / 3600;
                let remaining_m = (remaining_secs % 3600) / 60;
                let remaining_s = remaining_secs % 60;
                println!(
                    "Inserted: {} | skipped: {} | input rows ~{}/{} ({:.2}%) | elapsed: {:.0}s | rate: {:.0} rows/s | eta: {:02}:{:02}:{:02}",
                    attempted_inserts,
                    skipped,
                    rows_done,
                    total_rows,
                    pct,
                    elapsed,
                    rate_rows,
                    remaining_h,
                    remaining_m,
                    remaining_s,
                );
            }

            if pending.len() >= 200_000 {
                if let Err(e) = flush_to_sst(&new_db_writer, &mut pending, &mut sst_index) {
                    write_error(&format!("Writer thread: flush failed: {}", e));
                    return;
                }
            }
        }

        if !pending.is_empty() {
            if let Err(e) = flush_to_sst(&new_db_writer, &mut pending, &mut sst_index) {
                write_error(&format!("Writer thread: final flush failed: {}", e));
            }
        }

        let elapsed = start_time.elapsed().as_secs_f64();
        println!(
            "Insertion thread finished. Total inserted: {} | elapsed: {:.0}s",
            attempted_inserts,
            elapsed,
        );
    });

    let iter = old_db.iterator(rocksdb::IteratorMode::Start);

    for chunk in &iter.chunks(chunk_size) {
        if stop_flag.load(Ordering::SeqCst) {
            break;
        }

        let entries: Vec<(Vec<u8>, Vec<u8>)> = chunk
            .map(|item| {
                let (k, v) = item.expect("RocksDB iter error");
                (k.to_vec(), v.to_vec())
            })
            .collect();

        let stop_flag_par = Arc::clone(&stop_flag);
        let tx_par = tx.clone();
        let total_gates_tried_par = Arc::clone(&total_gates_tried);
        let skipped_par = Arc::clone(&skipped_count);
        let no_rule_l_skipped_par = Arc::clone(&no_rule_l_skipped);

        entries.par_chunks(20).for_each(|entry_chunk| {
            if stop_flag_par.load(Ordering::SeqCst) {
                return;
            }

            let mut local_results = Vec::new();
            let mut local_tried = 0usize;
            let mut local_skipped = 0usize;
            let mut local_no_rule_l_skipped = 0usize;

            for (_key, value) in entry_chunk {
                if value.is_empty() {
                    continue;
                }

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
                    let circuit_blob = &value[pos..pos + len];
                    pos += len;

                    let old_circuit = CircuitSeq::from_blob(circuit_blob);

                    local_tried += upper_bound_gates * 2;

                    let mut prefix: SmallVec<[[u16; 3]; 64]> = SmallVec::with_capacity(m);
                    prefix.extend_from_slice(&old_circuit.gates);

                    let (gates, filtered_gates) =
                        abstract_gates_for_circuit_filtered(&old_circuit, 3 * m, min_n, max_n);
                    local_skipped += filtered_gates * 2;

                    for g in gates.iter() {
                        let mut q1 = prefix.clone();
                        q1.push(*g);
                        let mut c1 = CircuitSeq { gates: q1.to_vec() };
                        c1.canonicalize();
                        if !c1.adjacent_id() {
                            match c1.canonicalize_polys(3 * m, !no_rule_l) {
                                None => { local_no_rule_l_skipped += 1; }
                                Some(canon1) => {
                                    let c1_hash: u128 = xxh3_128(&polys_repr_blob(&canon1.0));
                                    let c1_value = encode_circuit(&canon1.1.repr_blob());
                                    local_results.push((c1_hash.to_le_bytes().to_vec(), c1_value));
                                }
                            }
                        }

                        let mut q2: SmallVec<[[u16; 3]; 64]> = SmallVec::with_capacity(m + 1);
                        q2.push(*g);
                        q2.extend_from_slice(&prefix);
                        let mut c2 = CircuitSeq { gates: q2.to_vec() };
                        c2.canonicalize();
                        if !c2.adjacent_id() {
                            match c2.canonicalize_polys(3 * m, !no_rule_l) {
                                None => { local_no_rule_l_skipped += 1; }
                                Some(canon2) => {
                                    let c2_hash: u128 = xxh3_128(&polys_repr_blob(&canon2.0));
                                    let c2_value = encode_circuit(&canon2.1.repr_blob());
                                    local_results.push((c2_hash.to_le_bytes().to_vec(), c2_value));
                                }
                            }
                        }
                    }

                    while local_results.len() >= batch_size {
                        let drain_start = local_results.len() - batch_size;
                        let batch = local_results.split_off(drain_start);
                        if let Err(e) = tx_par.send(batch) {
                            eprintln!("Failed to send batch: {:?}", e);
                            return;
                        }
                    }

                    if stop_flag_par.load(Ordering::SeqCst) {
                        return;
                    }
                }
            }

            if !local_results.is_empty() {
                if let Err(e) = tx_par.send(local_results) {
                    eprintln!("Failed to send remaining batch: {:?}", e);
                }
            }
            total_gates_tried_par.fetch_add(local_tried, Ordering::Relaxed);
            skipped_par.fetch_add(local_skipped, Ordering::Relaxed);
            no_rule_l_skipped_par.fetch_add(local_no_rule_l_skipped, Ordering::Relaxed);
        });
    }

    drop(tx);
    insert_handle.join().expect("Insertion thread panicked");
    if no_rule_l {
        println!("Skipped (rule L required): {}", no_rule_l_skipped.load(Ordering::Relaxed));
    }

    if !stop_flag.load(Ordering::SeqCst) {
        println!("Compacting new_db for optimal read performance...");
        new_db.compact_range::<&[u8], &[u8]>(None, None);
        println!("Compaction done.");
    } else {
        println!("Stopped early, skipping compaction.");
    }

    println!("Build finished (or stopped early).");
    print_rule_times();
    Ok(())
}

// pub fn build_from_rocks(
//     old_db: &Arc<DB>,
//     new_db: &Arc<DB>,
//     m: usize,
// ) -> Result<(), Box<dyn std::error::Error>> {
//     println!("Running build (max CPU)");

//     rayon::ThreadPoolBuilder::new()
//         .num_threads(num_cpus::get())
//         .build_global()
//         .unwrap();

//     let total_rows = old_db
//         .property_int_value("rocksdb.estimate-num-keys")
//         .unwrap()
//         .unwrap_or(0);
//     println!("Estimated rows: {}", total_rows);

//     let chunk_size = 500_000;
//     let batch_size = 10_000;

//     let gates = Arc::new(base_gates(3 * m));
//     let total_gates_tried = Arc::new(std::sync::atomic::AtomicUsize::new(0));

//     let stop_flag = Arc::new(AtomicBool::new(false));
//     {
//         let stop_flag = stop_flag.clone();
//         ctrlc::set_handler(move || {
//             println!("CTRL+C detected! Finishing current batch...");
//             stop_flag.store(true, Ordering::SeqCst);
//         })
//         .expect("Error setting CTRL+C handler");
//     }

//     let (tx, rx) = bounded::<Vec<(CircuitSeq, Vec<Polynomial>, Vec<u8>)>>(1_000);
//     let stop_flag_clone = stop_flag.clone();
//     let new_db_writer = Arc::clone(new_db);
//     let total_gates_tried_insert = Arc::clone(&total_gates_tried);
//     let total_circuits = total_rows as usize * gates.len() * 2;

//     let insert_handle = std::thread::spawn(move || {
//         let start_time = std::time::Instant::now();
//         let mut attempted_inserts = 0;
//         let mut sst_index = 0usize;
//         let mut pending: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();

//         while let Ok(batch) = rx.recv() {
//             if stop_flag_clone.load(Ordering::SeqCst) {
//                 println!("Insertion thread stopping early...");
//                 break;
//             }

//             for (circuit, _canon, key) in &batch {
//                 let circuit_blob = circuit.repr_blob();
//                 let value = encode_circuit(&circuit_blob);
//                 pending.push((key.clone(), value));
//             }

//             attempted_inserts += batch.len();
//             let tried = total_gates_tried_insert.load(Ordering::Relaxed);
//             let elapsed = start_time.elapsed().as_secs_f64();
//             let rate = if elapsed > 0.0 { tried as f64 / elapsed } else { 0.0 };
//             let remaining = if rate > 0.0 {
//                 (total_circuits as f64 - tried as f64) / rate
//             } else {
//                 f64::INFINITY
//             };
//             let remaining_secs = remaining as u64;
//             let remaining_h = remaining_secs / 3600;
//             let remaining_m = (remaining_secs % 3600) / 60;
//             let remaining_s = remaining_secs % 60;
//             println!(
//                 "Attempted inserts: {} / {} ({:.2}%) | elapsed: {:.0}s | rate: {:.0}/s | eta: {:02}:{:02}:{:02}",
//                 attempted_inserts,
//                 total_circuits,
//                 if tried > 0 { (tried as f64 / total_circuits as f64) * 100.0 } else { 0.0 },
//                 elapsed,
//                 rate,
//                 remaining_h,
//                 remaining_m,
//                 remaining_s,
//             );

//             if pending.len() >= 200_000 {
//                 flush_to_sst(&new_db_writer, &mut pending, &mut sst_index);
//             }
//         }

//         if !pending.is_empty() {
//             flush_to_sst(&new_db_writer, &mut pending, &mut sst_index);
//         }

//         println!(
//             "Insertion thread finished. Total attempted: {} / {} | elapsed: {:.0}s",
//             attempted_inserts,
//             total_circuits,
//             start_time.elapsed().as_secs_f64(),
//         );
//     });

//     let iter = old_db.iterator(rocksdb::IteratorMode::Start);

//     for chunk in &iter.chunks(chunk_size) {
//         if stop_flag.load(Ordering::SeqCst) {
//             break;
//         }

//         let entries: Vec<(Vec<u8>, Vec<u8>)> = chunk
//             .map(|item| {
//                 let (k, v) = item.expect("RocksDB iter error");
//                 (k.to_vec(), v.to_vec())
//             })
//             .collect();

//         let stop_flag_par = Arc::clone(&stop_flag);
//         let tx_par = tx.clone();
//         let total_gates_tried_par = Arc::clone(&total_gates_tried);
//         let gates_par = Arc::clone(&gates);

//         entries.par_chunks(20).for_each(|entry_chunk| {
//             if stop_flag_par.load(Ordering::SeqCst) {
//                 return;
//             }

//             let mut local_results = Vec::new();

//             for (_key, value) in entry_chunk {
//                 if value.is_empty() {
//                     continue;
//                 }

//                 let mut pos = 0;
//                 while pos < value.len() {
//                     if pos + 1 > value.len() { break; }
//                     let len = value[pos] as usize;
//                     pos += 1;
//                     if pos + len > value.len() { break; }
//                     let circuit_blob = &value[pos..pos + len];
//                     pos += len;

//                     let old_circuit = CircuitSeq::from_blob(circuit_blob);

//                     total_gates_tried_par.fetch_add(gates_par.len() * 2, Ordering::Relaxed);

//                     let prefix: SmallVec<[[u8; 3]; 64]> = SmallVec::from_slice(&old_circuit.gates);

//                     for &g in gates_par.iter() {
//                         // append
//                         let mut q1 = prefix.clone();
//                         q1.push(g);
//                         let mut c1 = CircuitSeq { gates: q1.to_vec() };
//                         c1.canonicalize();
//                         if !c1.adjacent_id() {
//                             let canon1 = c1.canonicalize_polys(3 * m, true).unwrap();
//                             let hash: u128 = xxh3_128(&polys_repr_blob(&canon1.0));
//                             local_results.push((canon1.1, canon1.0, hash.to_le_bytes().to_vec()));
//                         }

//                         // prepend
//                         let mut q2: SmallVec<[[u8; 3]; 64]> = SmallVec::with_capacity(prefix.len() + 1);
//                         q2.push(g);
//                         q2.extend_from_slice(&prefix);
//                         let mut c2 = CircuitSeq { gates: q2.to_vec() };
//                         c2.canonicalize();
//                         if !c2.adjacent_id() {
//                             let canon2 = c2.canonicalize_polys(3 * m, true).unwrap();
//                             let hash: u128 = xxh3_128(&polys_repr_blob(&canon2.0));
//                             local_results.push((canon2.1, canon2.0, hash.to_le_bytes().to_vec()));
//                         }
//                     }

//                     while local_results.len() >= batch_size {
//                         let drain_start = local_results.len() - batch_size;
//                         let batch = local_results.split_off(drain_start);
//                         if let Err(e) = tx_par.send(batch) {
//                             eprintln!("Failed to send batch: {:?}", e);
//                             return;
//                         }
//                     }

//                     if stop_flag_par.load(Ordering::SeqCst) {
//                         return;
//                     }
//                 }
//             }

//             if !local_results.is_empty() {
//                 if let Err(e) = tx_par.send(local_results) {
//                     eprintln!("Failed to send remaining batch: {:?}", e);
//                 }
//             }
//         });
//     }

//     drop(tx);
//     insert_handle.join().expect("Insertion thread panicked");

//     if !stop_flag.load(Ordering::SeqCst) {
//         println!("Compacting new_db for optimal read performance...");
//         new_db.compact_range::<&[u8], &[u8]>(None, None);
//         println!("Compaction done.");
//     } else {
//         println!("Stopped early, skipping compaction.");
//     }

//     println!("Build finished (or stopped early).");
//     print_rule_times();
//     Ok(())
// }

pub fn build_m1(new_db: &Arc<DB>) -> Result<(), Box<dyn std::error::Error>> {
    println!("Building m1 base case");

    let gates = base_gates(3);
    let mut pending: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    let mut sst_index = 0usize;

    for g in gates.iter() {
        let c = CircuitSeq { gates: vec![*g] };
        let canon = canonicalize_polys(c.to_polynomial(3, 0, 1), true, false);
        let mut c = c;
        c.rewire(&canon.1.invert(), 3);

        if c.adjacent_id() {
            continue;
        }

        let canon_blob = polys_repr_blob(&canon.0);
        let hash: u128 = xxh3_128(&canon_blob);
        let key = hash.to_le_bytes().to_vec();

        let circuit_blob = c.repr_blob();
        let value = encode_circuit(&circuit_blob);

        pending.push((key, value));
    }

    if let Err(e) = flush_to_sst(new_db, &mut pending, &mut sst_index) {
        write_error(&format!("combine_rocks_dbs: flush failed: {}", e));
    }

    println!("Compacting m1 db...");
    new_db.compact_range::<&[u8], &[u8]>(None, None);
    println!("Done.");

    Ok(())
}


/// Apply a wire mapping to a circuit — remap C2's internal wires
/// to their positions in the combined circuit.
pub fn apply_wire_mapping(circuit: &CircuitSeq, mapping: &[u16]) -> CircuitSeq {
    CircuitSeq {
        gates: circuit
            .gates
            .iter()
            .map(|&[a, b, c]| [mapping[a as usize], mapping[b as usize], mapping[c as usize]])
            .collect(),
    }
}

// Cache: (n1, n2) -> Arc<(flat_mappings, stride=n2)>
// flat_mappings is all mappings concatenated contiguously.
// mapping i is at flat[i*n2..(i+1)*n2].
// Entries with more mappings than this are streamed on-the-fly instead of cached.
const LARGE_MAPPING_THRESHOLD: usize = 200_000;
// Max number of entries kept in the LRU cache.
const MAPPING_CACHE_CAP: usize = 256;

static MAPPING_CACHE: Lazy<std::sync::Mutex<lru::LruCache<(usize, usize), Arc<(Vec<u16>, usize)>>>> =
    Lazy::new(|| std::sync::Mutex::new(lru::LruCache::new(
        std::num::NonZeroUsize::new(MAPPING_CACHE_CAP).unwrap()
    )));

/// Call `f` once per mapping for the (n1, n2) pair.
/// Small pairs are cached in an LRU; large pairs are enumerated on-the-fly.
pub fn for_each_mapping<F: FnMut(&[u16])>(n1: usize, n2: usize, mut f: F) {
    let total = count_mappings(n1, n2);
    if total <= LARGE_MAPPING_THRESHOLD {
        // Try cache first
        let cached = {
            let mut cache = MAPPING_CACHE.lock().unwrap();
            cache.get(&(n1, n2)).cloned()
        };
        let entry = cached.unwrap_or_else(|| {
            let arc = Arc::new(compute_mappings(n1, n2));
            let mut cache = MAPPING_CACHE.lock().unwrap();
            cache.put((n1, n2), Arc::clone(&arc));
            arc
        });
        let (flat, stride) = &*entry;
        if *stride > 0 {
            for chunk in flat.chunks(*stride) {
                f(chunk);
            }
        }
    } else {
        // Large: enumerate directly without caching
        let mut c2_to_wire = vec![0u16; n2];
        let mut used = vec![false; n2];
        enumerate_direct_callback(0, n1, n2, &mut c2_to_wire, &mut used, &mut f);
    }
}

// Keep the old cached accessor for the warmup step.
pub fn enumerate_c2_wire_mappings_cached(n1: usize, n2: usize) -> Arc<(Vec<u16>, usize)> {
    let total = count_mappings(n1, n2);
    if total > LARGE_MAPPING_THRESHOLD {
        return Arc::new((vec![], 0)); // sentinel: will be streamed on-the-fly
    }
    let cached = {
        let mut cache = MAPPING_CACHE.lock().unwrap();
        cache.get(&(n1, n2)).cloned()
    };
    cached.unwrap_or_else(|| {
        let arc = Arc::new(compute_mappings(n1, n2));
        let mut cache = MAPPING_CACHE.lock().unwrap();
        cache.put((n1, n2), Arc::clone(&arc));
        arc
    })
}

fn count_mappings(n1: usize, n2: usize) -> usize {
    let k_max = n1.min(n2);
    let mut total = 0usize;
    let mut cnk = 1usize;
    let mut pnk = 1usize;
    for k in 0..=k_max {
        if k > 0 {
            cnk = cnk * (n1 - k + 1) / k;
            pnk *= n2 - k + 1;
        }
        total += cnk * pnk;
    }
    total
}

fn compute_mappings(n1: usize, n2: usize) -> (Vec<u16>, usize) {
    let total = count_mappings(n1, n2);
    let mut flat = vec![0u16; total * n2.max(1)];
    let mut idx = 0usize;
    let mut c2_to_wire = vec![0u16; n2];
    let mut used = vec![false; n2];

    enumerate_direct(
        0, n1, n2,
        &mut c2_to_wire,
        &mut used,
        &mut flat,
        &mut idx,
    );

    debug_assert_eq!(idx, total);
    (flat, n2)
}

fn enumerate_direct(
    pos: usize,
    n1: usize,
    n2: usize,
    c2_to_wire: &mut Vec<u16>,
    used: &mut Vec<bool>,
    flat: &mut Vec<u16>,
    idx: &mut usize,
) {
    if pos == n1 {
        // Assign fresh wires to all unassigned c2 wires in order
        let mut fresh = n1;
        for j in 0..n2 {
            if !used[j] {
                c2_to_wire[j] = fresh as u16;
                fresh += 1;
            }
        }

        // Write mapping into flat buffer
        let offset = *idx * n2;
        flat[offset..offset + n2].copy_from_slice(c2_to_wire);
        *idx += 1;

        // Undo fresh assignments
        for j in 0..n2 {
            if !used[j] {
                c2_to_wire[j] = 0;
            }
        }
        return;
    }

    // Option 1: c1 wire `pos` is not shared with any c2 wire
    enumerate_direct(pos + 1, n1, n2, c2_to_wire, used, flat, idx);

    // Option 2: share c1 wire `pos` with c2 wire j, for each unused j
    for j in 0..n2 {
        if !used[j] {
            used[j] = true;
            c2_to_wire[j] = pos as u16;
            enumerate_direct(pos + 1, n1, n2, c2_to_wire, used, flat, idx);
            used[j] = false;
            c2_to_wire[j] = 0;
        }
    }
}

fn enumerate_direct_callback<F: FnMut(&[u16])>(
    pos: usize,
    n1: usize,
    n2: usize,
    c2_to_wire: &mut Vec<u16>,
    used: &mut Vec<bool>,
    f: &mut F,
) {
    if pos == n1 {
        let mut fresh = n1;
        for j in 0..n2 {
            if !used[j] {
                c2_to_wire[j] = fresh as u16;
                fresh += 1;
            }
        }
        f(c2_to_wire);
        for j in 0..n2 {
            if !used[j] {
                c2_to_wire[j] = 0;
            }
        }
        return;
    }
    enumerate_direct_callback(pos + 1, n1, n2, c2_to_wire, used, f);
    for j in 0..n2 {
        if !used[j] {
            used[j] = true;
            c2_to_wire[j] = pos as u16;
            enumerate_direct_callback(pos + 1, n1, n2, c2_to_wire, used, f);
            used[j] = false;
            c2_to_wire[j] = 0;
        }
    }
}

// ── Capped mapping enumeration ────────────────────────────────────────────────
//
// Both constituent circuits are stored in dense canonical wire labelings, so a
// mapping that shares exactly k wires produces a combined circuit using exactly
// n1 + n2 - k wires. The min_n filter therefore only depends on k: mappings
// with k > k_max (= n1 + n2 - min_n) are guaranteed to be skipped, and the
// enumeration can prune those recursion branches instead of materializing the
// mapping, building the combined circuit, gate-canonicalizing it, and counting
// its wires just to throw it away. High-k mappings dominate the total count
// (C(n1,k)·P(n2,k) grows steeply in k), so at aggressive min_n this removes
// most of the per-pair work.
//
// Counting is exact so the caller can keep skip accounting identical to the
// unpruned enumeration (one skip per pruned mapping). u128 with saturation
// keeps the arithmetic safe for large wire counts (overflow-checks is on).

/// Number of mappings with exactly k in 0..=k_cap shared wires.
fn count_mappings_capped_u128(n1: usize, n2: usize, k_cap: usize) -> u128 {
    let k_max = n1.min(n2).min(k_cap);
    let mut total: u128 = 0;
    let mut cnk: u128 = 1;
    let mut pnk: u128 = 1;
    for k in 0..=k_max {
        if k > 0 {
            cnk = cnk.saturating_mul((n1 - k + 1) as u128) / k as u128;
            pnk = pnk.saturating_mul((n2 - k + 1) as u128);
        }
        total = total.saturating_add(cnk.saturating_mul(pnk));
    }
    total
}

/// Mappings pruned by capping shared wires at `k_cap` (i.e. those with k > k_cap).
pub fn count_mappings_pruned(n1: usize, n2: usize, k_cap: usize) -> usize {
    let all = count_mappings_capped_u128(n1, n2, n1.min(n2));
    let kept = count_mappings_capped_u128(n1, n2, k_cap);
    usize::try_from(all - kept).unwrap_or(usize::MAX)
}

/// Like `for_each_mapping`, but only visits mappings sharing at most `k_cap`
/// wires. `f` also receives the shared-wire count k of each mapping. When
/// k_cap >= min(n1, n2) this visits exactly the same mappings in exactly the
/// same order as `for_each_mapping`.
pub fn for_each_mapping_capped<F: FnMut(&[u16], usize)>(
    n1: usize,
    n2: usize,
    k_cap: usize,
    mut f: F,
) {
    let mut c2_to_wire = vec![0u16; n2];
    let mut used = vec![false; n2];
    enumerate_direct_capped(0, n1, n2, 0, k_cap, &mut c2_to_wire, &mut used, &mut f);
}

fn enumerate_direct_capped<F: FnMut(&[u16], usize)>(
    pos: usize,
    n1: usize,
    n2: usize,
    shared: usize,
    k_cap: usize,
    c2_to_wire: &mut Vec<u16>,
    used: &mut Vec<bool>,
    f: &mut F,
) {
    if pos == n1 {
        let mut fresh = n1;
        for j in 0..n2 {
            if !used[j] {
                c2_to_wire[j] = fresh as u16;
                fresh += 1;
            }
        }
        f(c2_to_wire, shared);
        for j in 0..n2 {
            if !used[j] {
                c2_to_wire[j] = 0;
            }
        }
        return;
    }
    enumerate_direct_capped(pos + 1, n1, n2, shared, k_cap, c2_to_wire, used, f);
    if shared < k_cap {
        for j in 0..n2 {
            if !used[j] {
                used[j] = true;
                c2_to_wire[j] = pos as u16;
                enumerate_direct_capped(pos + 1, n1, n2, shared + 1, k_cap, c2_to_wire, used, f);
                used[j] = false;
                c2_to_wire[j] = 0;
            }
        }
    }
}

pub fn build_from_2rocks(
    db1: &Arc<DB>,
    db2: &Arc<DB>,
    new_db: &Arc<DB>,
    m1: usize,
    m2: usize,
    min_n: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let m = m1 + m2;
    let n = 3 * m;
    let same_db = Arc::ptr_eq(db1, db2);
    println!(
        "Running build_from_2rocks: m1={} m2={} -> m={} same_db={}",
        m1, m2, m, same_db
    );

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build_global()
        .ok();

    // ── Load db2 into memory once ─────────────────────────────────────────────
    println!("Loading db2 into memory...");
    let db2_circuits: Arc<Vec<CircuitSeq>> = Arc::new({
        let iter = db2.iterator(rocksdb::IteratorMode::Start);
        let mut circuits = Vec::new();
        for item in iter {
            let (_key, value) = item.expect("RocksDB iter error");
            let mut pos = 0;
            while pos < value.len() {
                if pos + 1 > value.len() { break; }
                let len = value[pos] as usize;
                pos += 1;
                if pos + len > value.len() { break; }
                circuits.push(CircuitSeq::from_blob(&value[pos..pos + len]));
                pos += len;
            }
        }
        println!("Loaded {} circuits from db2", circuits.len());
        circuits
    });

    // Precompute c2_rev for every c2 once
    let total_c2 = db2_circuits.len();
    println!("Precomputing c2_rev (0/{})...", total_c2);
    let c2_rev_done = std::sync::atomic::AtomicUsize::new(0);
    let db2_rev: Arc<Vec<CircuitSeq>> = Arc::new(
        db2_circuits.par_iter().map(|c2| {
            let mut r = CircuitSeq { gates: c2.gates.iter().rev().cloned().collect() };
            r.canonicalize();
            // Remap to minimal wires
            let used = r.used_wires();
            let wire_map: HashMap<u16, u16> = used.iter().enumerate()
                .map(|(i, &w)| (w, i as u16))
                .collect();
            r = CircuitSeq {
                gates: r.gates.iter().map(|&[t, c1, c2]| [
                    wire_map[&t], wire_map[&c1], wire_map[&c2],
                ]).collect(),
            };
            r.canonicalize();
            let n2 = r.max_wire() as usize + 1;
            let canon = canonicalize_polys_4(r.to_polynomial(n2, 0, r.gates.len()), true).unwrap();
            r.rewire(&canon.1.invert(), n2);
            r.canonicalize();
            let done = c2_rev_done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if done % 50 == 0 || done == total_c2 {
                println!("Precomputing c2_rev ({}/{})...", done, total_c2);
            }
            r
        }).collect()
    );

    // Precompute touched wire counts for every c2 and c2_rev
    let db2_n2: Arc<Vec<usize>> = Arc::new(
        db2_circuits.par_iter().map(|c| touched_wires(c).len()).collect()
    );
    let db2_rev_n2: Arc<Vec<usize>> = Arc::new(
        db2_rev.par_iter().map(|c| touched_wires(c).len()).collect()
    );

    // Mapping enumeration is streamed per pair with min_n-based pruning
    // (see for_each_mapping_capped); no flat-mapping cache warmup needed.
    println!(
        "Mapping enumeration: capped at k <= n1 + n2 - {} shared wires (pruned analytically)",
        min_n
    );

    let total_rows = db1
        .property_int_value("rocksdb.estimate-num-keys")
        .unwrap()
        .unwrap_or(0);

    let chunk_size = 500_000;
    let batch_size = 50_000;
    let nc2 = db2_circuits.len();

    let total_pairs_est = total_rows as usize * nc2;
    println!("db1 estimated keys: {}", total_rows);
    println!("db2 circuits loaded: {}", nc2);
    println!("Estimated total pairs: {} ({:.2}B)", total_pairs_est, total_pairs_est as f64 / 1e9);
    println!("chunk_size={} batch_size={} channel_cap=1000 pending_threshold=1M", chunk_size, batch_size);

    let total_gates_tried = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let total_results_generated = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let skipped_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let build_start = std::time::Instant::now();

    let stop_flag = Arc::new(AtomicBool::new(false));
    {
        let sf = stop_flag.clone();
        let _ = ctrlc::set_handler(move || {
            println!("CTRL+C detected! Finishing current batch...");
            sf.store(true, Ordering::SeqCst);
        });
    }

    // ── Writer thread ─────────────────────────────────────────────────────────
    let (tx, rx) = bounded::<Vec<(Vec<u8>, Vec<u8>)>>(1_000);

    let stop_flag_clone = stop_flag.clone();
    let new_db_writer = Arc::clone(new_db);
    let total_gates_tried_insert = Arc::clone(&total_gates_tried);
    let total_results_insert = Arc::clone(&total_results_generated);
    let skipped_count_insert = Arc::clone(&skipped_count);

    let insert_handle = std::thread::spawn(move || {
        let start_time = std::time::Instant::now();
        let mut attempted_inserts = 0usize;
        let mut sst_count = 0usize;
        let mut sst_index = 0usize;
        let mut pending: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();

        while let Ok(batch) = rx.recv() {
            if stop_flag_clone.load(Ordering::SeqCst) {
                println!("Insertion thread stopping early...");
                break;
            }
            attempted_inserts += batch.len();
            for (key, value) in batch {
                pending.push((key, value));
            }

            let pairs_done = total_gates_tried_insert.load(Ordering::Relaxed);
            let skipped = skipped_count_insert.load(Ordering::Relaxed);
            let results_so_far = total_results_insert.load(Ordering::Relaxed);
            let elapsed = start_time.elapsed().as_secs_f64();
            let pairs_rate = if elapsed > 0.0 { pairs_done as f64 / elapsed } else { 0.0 };
            let pairs_remaining = if pairs_done < total_pairs_est { total_pairs_est - pairs_done } else { 0 };
            let eta_secs = if pairs_rate > 0.0 { (pairs_remaining as f64 / pairs_rate) as u64 } else { 0 };
            let dedup_ratio = if results_so_far > 0 { attempted_inserts as f64 / results_so_far as f64 } else { 0.0 };
            println!(
                "[writer] pairs={}/{} ({:.1}%) | pairs/s={:.0} | eta={:02}:{:02}:{:02} | inserts={} | results={} | dedup={:.1}x | pending={} | ssts={}",
                pairs_done, total_pairs_est,
                (pairs_done as f64 / total_pairs_est as f64) * 100.0,
                pairs_rate,
                eta_secs / 3600, (eta_secs % 3600) / 60, eta_secs % 60,
                attempted_inserts + skipped,
                results_so_far,
                dedup_ratio,
                pending.len(),
                sst_count,
            );

            if pending.len() >= 1_000_000 {
                if let Err(e) = flush_to_sst(&new_db_writer, &mut pending, &mut sst_index) {
                    write_error(&format!("Writer thread: flush failed: {}", e));
                    return;
                }
                sst_count += 1;
            }
        }

        println!("[writer] producers done, flushing {} remaining entries across {} final SSTs...",
            pending.len() + attempted_inserts - attempted_inserts, // pending count
            (pending.len() + 999_999) / 1_000_000
        );
        while !pending.is_empty() {
            if let Err(e) = flush_to_sst(&new_db_writer, &mut pending, &mut sst_index) {
                write_error(&format!("Writer thread: final flush failed: {}", e));
                break;
            }
            sst_count += 1;
            println!("[writer] final flush: {} remaining | sst #{}", pending.len(), sst_count);
        }
        let elapsed = start_time.elapsed().as_secs_f64();
        println!(
            "[writer] finished. total_inserts={} | total_ssts={} | elapsed={:.0}s ({:.1}h)",
            attempted_inserts, sst_count, elapsed, elapsed / 3600.0,
        );
    });

    // ── Main loop: stream db1 in chunks ──────────────────────────────────────
    let iter = db1.iterator(rocksdb::IteratorMode::Start);
    let mut chunk_idx = 0usize;
    let mut total_c1_processed = 0usize;

    for chunk in &iter.chunks(chunk_size) {
        if stop_flag.load(Ordering::SeqCst) { break; }
        chunk_idx += 1;
        let elapsed_outer = build_start.elapsed().as_secs_f64();
        let pairs_done = total_gates_tried.load(Ordering::Relaxed);
        let outer_rate = if elapsed_outer > 0.0 { pairs_done as f64 / elapsed_outer } else { 0.0 };
        let pairs_remaining = if pairs_done < total_pairs_est { total_pairs_est - pairs_done } else { 0 };
        let eta_outer = if outer_rate > 0.0 { (pairs_remaining as f64 / outer_rate) as u64 } else { 0 };
        println!(
            "[chunk {}] c1_processed={} | pairs={}/{} ({:.1}%) | {:.0} pairs/s | eta {:02}:{:02}:{:02} | elapsed={:.1}h",
            chunk_idx, total_c1_processed,
            pairs_done, total_pairs_est,
            (pairs_done as f64 / total_pairs_est as f64) * 100.0,
            outer_rate,
            eta_outer / 3600, (eta_outer % 3600) / 60, eta_outer % 60,
            elapsed_outer / 3600.0,
        );

        let entries: Vec<(Vec<u8>, Vec<u8>)> = chunk
            .map(|item| {
                let (k, v) = item.expect("RocksDB iter error");
                (k.to_vec(), v.to_vec())
            })
            .collect();

        // Decode all c1 circuits from this chunk
        let mut c1_circuits: Vec<CircuitSeq> = Vec::new();
        for (_key, value) in &entries {
            let mut pos = 0;
            while pos < value.len() {
                if pos + 1 > value.len() { break; }
                let len = value[pos] as usize;
                pos += 1;
                if pos + len > value.len() { break; }
                c1_circuits.push(CircuitSeq::from_blob(&value[pos..pos + len]));
                pos += len;
            }
        }

        // Precompute per-c1 data in parallel
        struct C1Data {
            c1: CircuitSeq,
            n1: usize,
            c1_rev: CircuitSeq,
            n1_rev: usize,
        }

        let c1_data: Vec<C1Data> = c1_circuits
            .into_par_iter()
            .map(|c1| {
                let n1 = touched_wires(&c1).len();
                let c1_rev = {
                    let mut r = CircuitSeq { gates: c1.gates.iter().rev().cloned().collect() };
                    r.canonicalize();
                    let used = r.used_wires();
                    let wire_map: HashMap<u16, u16> = used.iter().enumerate()
                        .map(|(i, &w)| (w, i as u16))
                        .collect();
                    r = CircuitSeq {
                        gates: r.gates.iter().map(|&[t, c1, c2]| [
                            wire_map[&t], wire_map[&c1], wire_map[&c2],
                        ]).collect(),
                    };
                    r.canonicalize();
                    let n1r = r.max_wire() as usize + 1;
                    let canon = canonicalize_polys_4(r.to_polynomial(n1r, 0, r.gates.len()), true).unwrap();
                    r.rewire(&canon.1.invert(), n1r);
                    r.canonicalize();
                    r
                };
                let n1_rev = touched_wires(&c1_rev).len();
                C1Data { c1, n1, c1_rev, n1_rev }
            })
            .collect();
        println!("c1_data precomputed: {} circuits", c1_data.len());

        // Build flat work list in parallel over c1, then process each item in parallel.
        // We avoid collecting into WorkItem structs and instead process directly
        // using a two-level par_iter: outer over c1, inner over (c2, mapping, case).
        // This gives maximum parallelism while keeping allocations minimal.
        let db2_ref   = &*db2_circuits;
        let db2_rev_ref  = &*db2_rev;
        let db2_n2_ref   = &*db2_n2;
        let db2_rev_n2_ref = &*db2_rev_n2;
        let stop_flag_par  = Arc::clone(&stop_flag);
        let tx_par = tx.clone();
        let total_gates_tried_par = Arc::clone(&total_gates_tried);
        let total_results_par = Arc::clone(&total_results_generated);
        let skipped_par = Arc::clone(&skipped_count);

        let total_c1 = c1_data.len();
        let c1_done = std::sync::atomic::AtomicUsize::new(0);
        let chunk_total = std::sync::atomic::AtomicUsize::new(0);
        println!("[chunk {}] processing {} c1 circuits × {} c2 = {} pairs this chunk",
            chunk_idx, total_c1, nc2, total_c1 * nc2);

        c1_data.par_iter().for_each(|d| {
                if stop_flag_par.load(Ordering::SeqCst) {
                    return;
                }

                let mut local: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
                let mut local_tried = 0usize;
                let mut local_skipped = 0usize;

                for (j, c2) in db2_ref.iter().enumerate() {
                    let n2     = db2_n2_ref[j];
                    let c2_rev = &db2_rev_ref[j];
                    let n2_rev = db2_rev_n2_ref[j];

                    local_tried += 1;

                    // A mapping sharing k wires yields a combined circuit on at
                    // most n_first + n_second - k wires, so k > n_first +
                    // n_second - min_n can never pass the min_n check below.
                    // Those mappings are pruned inside the capped enumeration
                    // and counted analytically here, exactly matching the old
                    // per-mapping skip counts.
                    let k_cap_12 = (d.n1 + n2).saturating_sub(min_n);
                    let k_cap_r2 = (d.n1_rev + n2).saturating_sub(min_n);
                    let k_cap_1r = (d.n1 + n2_rev).saturating_sub(min_n);
                    local_skipped += count_mappings_pruned(d.n1, n2, k_cap_12);
                    local_skipped += count_mappings_pruned(n2, d.n1, k_cap_12);
                    local_skipped += count_mappings_pruned(d.n1_rev, n2, k_cap_r2);
                    local_skipped += count_mappings_pruned(n2_rev, d.n1, k_cap_1r);

                    // Helper closure: concatenate, canonicalize, push if non-trivial.
                    // The min_n check stays as a guard, but mappings that cannot
                    // reach min_n wires never get here.
                    let mut try_push = |first_gates: &[[u16; 3]], second_gates: &[[u16; 3]]| {
                        let mut gates = Vec::with_capacity(first_gates.len() + second_gates.len());
                        gates.extend_from_slice(first_gates);
                        gates.extend_from_slice(second_gates);
                        let mut combined = CircuitSeq { gates };
                        combined.canonicalize();
                        if combined.used_wires().len() < min_n {
                            local_skipped += 1;
                            return;
                        }
                        if combined.adjacent_id() { return; }
                        let (canon_polys, canon_circuit, _, _, _) = combined.canonicalize_polys(n, true).unwrap();
                        let key = xxh3_128(&polys_repr_blob(&canon_polys)).to_le_bytes().to_vec();
                        let value = encode_circuit(&canon_circuit.repr_blob());
                        local.push((key, value));
                    };

                    // Case 1: c1 || mapped_c2
                    for_each_mapping_capped(d.n1, n2, k_cap_12, |mapping, _k| {
                        let c2_mapped = apply_wire_mapping(c2, mapping);
                        try_push(&d.c1.gates, &c2_mapped.gates);
                    });

                    // Case 2: c2 || mapped_c1
                    for_each_mapping_capped(n2, d.n1, k_cap_12, |mapping, _k| {
                        let c1_mapped = apply_wire_mapping(&d.c1, mapping);
                        try_push(&c2.gates, &c1_mapped.gates);
                    });

                    // Case 3: c1_rev || mapped_c2
                    for_each_mapping_capped(d.n1_rev, n2, k_cap_r2, |mapping, _k| {
                        let c2_mapped = apply_wire_mapping(c2, mapping);
                        try_push(&d.c1_rev.gates, &c2_mapped.gates);
                    });

                    // Case 4: mapped_c1 || c2_rev
                    for_each_mapping_capped(n2_rev, d.n1, k_cap_1r, |mapping, _k| {
                        let c1_mapped = apply_wire_mapping(&d.c1, mapping);
                        try_push(&c1_mapped.gates, &c2_rev.gates);
                    });

                    // // Case 5: c2_rev || mapped_c1
                    // for i in 0..n_2_rev1 {
                    //     let mapping = &flat_2_rev1[i * stride_2_rev1..(i + 1) * stride_2_rev1];
                    //     let c1_mapped = apply_wire_mapping(&d.c1, mapping);
                    //     try_push(&c2_rev.gates, &c1_mapped.gates);
                    // }

                    // // Case 6: mapped_c1 || c2
                    // for i in 0..n_2_1 {
                    //     let mapping = &flat_2_1[i * stride_2_1..(i + 1) * stride_2_1];
                    //     let c1_mapped = apply_wire_mapping(&d.c1, mapping);
                    //     try_push(&c1_mapped.gates, &c2.gates);
                    // }

                    // // Case 7: mapped_c2 || c1
                    // for i in 0..n_1_2 {
                    //     let mapping = &flat_1_2[i * stride_1_2..(i + 1) * stride_1_2];
                    //     let c2_mapped = apply_wire_mapping(c2, mapping);
                    //     try_push(&c2_mapped.gates, &d.c1.gates);
                    // }

                    // // Case 8: mapped_c2 || c1_rev
                    // for i in 0..n_rev1_2 {
                    //     let mapping = &flat_rev1_2[i * stride_rev1_2..(i + 1) * stride_rev1_2];
                    //     let c2_mapped = apply_wire_mapping(c2, mapping);
                    //     try_push(&c2_mapped.gates, &d.c1_rev.gates);
                    // }

                    if local.len() >= batch_size && !stop_flag_par.load(Ordering::SeqCst) {
                        let n = local.len();
                        let batch = std::mem::take(&mut local);
                        total_results_par.fetch_add(n, Ordering::Relaxed);
                        if let Err(e) = tx_par.send(batch) {
                            eprintln!("Failed to send batch: {:?}", e);
                        }
                    }
                }

                let n_local = local.len();
                // Send any remaining results
                if !local.is_empty() && !stop_flag_par.load(Ordering::SeqCst) {
                    total_results_par.fetch_add(n_local, Ordering::Relaxed);
                    if let Err(e) = tx_par.send(local) {
                        eprintln!("Failed to send batch: {:?}", e);
                    }
                }
                chunk_total.fetch_add(n_local, Ordering::Relaxed);
                total_gates_tried_par.fetch_add(local_tried, Ordering::Relaxed);
                skipped_par.fetch_add(local_skipped, Ordering::Relaxed);

                let done = c1_done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if done % 50 == 0 || done == total_c1 {
                    let pairs_done = total_gates_tried_par.load(Ordering::Relaxed);
                    let elapsed = build_start.elapsed().as_secs_f64();
                    let rate = if elapsed > 0.0 { pairs_done as f64 / elapsed } else { 0.0 };
                    let remaining = if pairs_done < total_pairs_est { total_pairs_est - pairs_done } else { 0 };
                    let eta = if rate > 0.0 { (remaining as f64 / rate) as u64 } else { 0 };
                    println!(
                        "[chunk {}] c1 {}/{} | pairs={}/{} ({:.1}%) | {:.0}/s | eta {:02}:{:02}:{:02} | results={}",
                        chunk_idx, done, total_c1,
                        pairs_done, total_pairs_est,
                        (pairs_done as f64 / total_pairs_est as f64) * 100.0,
                        rate,
                        eta / 3600, (eta % 3600) / 60, eta % 60,
                        chunk_total.load(Ordering::Relaxed),
                    );
                }
        });

        let chunk_results = chunk_total.load(Ordering::Relaxed);
        total_c1_processed += c1_data.len();
        let elapsed = build_start.elapsed().as_secs_f64();
        println!(
            "[chunk {}] done | c1_total_processed={} | chunk_results={} | total_results={} | elapsed={:.1}h",
            chunk_idx, total_c1_processed, chunk_results,
            total_results_generated.load(Ordering::Relaxed),
            elapsed / 3600.0,
        );

        if stop_flag.load(Ordering::SeqCst) { break; }
    }

    drop(tx);
    insert_handle.join().expect("Insertion thread panicked");

    if !stop_flag.load(Ordering::SeqCst) {
        println!("Compacting new_db...");
        new_db.compact_range::<&[u8], &[u8]>(None, None);
        println!("Compaction done.");
    }

    println!("Build finished.");
    print_rule_times();
    Ok(())
}

pub fn combine_rocks_dbs(output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    use std::collections::BinaryHeap;
    use std::cmp::Reverse;

    // Open all existing source DBs read-only
    let mut dbs: Vec<DB> = Vec::new();
    for m in 1..=9 {
        let path = format!("rocks_db_m{}", m);
        let mut read_opts = Options::default();
        read_opts.set_merge_operator_associative("append_merge", append_merge);
        read_opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(16));
        match DB::open_for_read_only(&read_opts, &path, false) {
            Ok(db) => {
                let est = db.property_value("rocksdb.estimate-num-keys")
                    .ok().flatten()
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(0);
                println!("Opened {} (~{} keys)", path, est);
                dbs.push(db);
            }
            Err(e) => println!("Skipping {} ({})", path, e),
        }
    }
    if dbs.is_empty() {
        println!("No source databases found.");
        return Ok(());
    }

    // One iterator per DB — they all iterate in sorted key order
    let mut iters: Vec<rocksdb::DBIterator> = dbs
        .iter()
        .map(|db| db.iterator(rocksdb::IteratorMode::Start))
        .collect();

    // Seed the min-heap with the first entry from each iterator
    // Reverse makes BinaryHeap a min-heap ordered by (key, value, db_idx)
    let mut heap: BinaryHeap<Reverse<(Vec<u8>, Vec<u8>, usize)>> = BinaryHeap::new();
    for (i, iter) in iters.iter_mut().enumerate() {
        if let Some(Ok((k, v))) = iter.next() {
            heap.push(Reverse((k.to_vec(), v.to_vec(), i)));
        }
    }

    // Open output DB with same options as open_db_for_write
    let output = {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.set_merge_operator_associative("append_merge", append_merge);
        opts.set_manual_wal_flush(true);
        opts.increase_parallelism(num_cpus::get() as i32);
        opts.set_max_background_jobs(64);
        opts.set_max_open_files(-1);
        opts.set_write_buffer_size(256 * 1024 * 1024);
        opts.set_max_write_buffer_number(4);
        opts.set_min_write_buffer_number_to_merge(2);
        opts.set_level_zero_file_num_compaction_trigger(10);
        opts.set_max_bytes_for_level_base(512 * 1024 * 1024);
        opts.set_max_bytes_for_level_multiplier(10.0);
        opts.set_num_levels(7);
        opts.set_compression_type(DBCompressionType::Zstd);
        opts.set_bottommost_compression_type(DBCompressionType::Zstd);
        opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(16));
        let mut block_opts = BlockBasedOptions::default();
        block_opts.set_bloom_filter(10.0, false);
        block_opts.set_block_size(16 * 1024);
        block_opts.set_cache_index_and_filter_blocks(true);
        opts.set_block_based_table_factory(&block_opts);
        Arc::new(DB::open(&opts, output_path)?)
    };

    let keys_per_sst = 500_000usize;
    let mut pending: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(keys_per_sst + 1);
    let mut sst_index = 0usize;
    let mut unique_written = 0usize;
    let start = std::time::Instant::now();

    // k-way merge: always pop the globally smallest key
    while !heap.is_empty() {
        let Reverse((min_key, first_val, db_idx)) = heap.pop().unwrap();
        if let Some(Ok((k, v))) = iters[db_idx].next() {
            heap.push(Reverse((k.to_vec(), v.to_vec(), db_idx)));
        }

        // Collect values for the same key from other DBs currently at the heap front
        let mut merged_val = first_val;
        loop {
            match heap.peek() {
                Some(Reverse((k, _, _))) if *k == min_key => {
                    let Reverse((_, v, idx)) = heap.pop().unwrap();
                    merged_val.extend_from_slice(&v);
                    if let Some(Ok((k2, v2))) = iters[idx].next() {
                        heap.push(Reverse((k2.to_vec(), v2.to_vec(), idx)));
                    }
                }
                _ => break,
            }
        }

        pending.push((min_key, merged_val));
        unique_written += 1;

        if pending.len() >= keys_per_sst {
            // Write pending (already sorted) to SST and ingest
            let sst_path = format!("/dev/shm/combine_sst_{}.sst", sst_index);
            sst_index += 1;
            {
                let mut sst_opts = Options::default();
                sst_opts.set_merge_operator_associative("append_merge", append_merge);
                sst_opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(16));
                sst_opts.set_compression_type(DBCompressionType::Zstd);
                let mut writer = SstFileWriter::create(&sst_opts);
                writer.open(&sst_path)?;
                for (k, v) in &pending {
                    writer.put(k, v)?;
                }
                writer.finish()?;
            }
            let mut ingest_opts = IngestExternalFileOptions::default();
            ingest_opts.set_move_files(false);
            output.ingest_external_file_opts(&ingest_opts, vec![sst_path.clone()])?;
            let _ = std::fs::remove_file(&sst_path);
            pending.clear();

            let elapsed = start.elapsed().as_secs_f64();
            let rate = unique_written as f64 / elapsed;
            println!(
                "[combine] sst={} | keys={} | {:.0} keys/s | elapsed={:.0}s ({:.1}h)",
                sst_index, unique_written, rate, elapsed, elapsed / 3600.0
            );
        }
    }

    // Final partial SST
    if !pending.is_empty() {
        let sst_path = format!("/dev/shm/combine_sst_{}.sst", sst_index);
        sst_index += 1;
        {
            let mut sst_opts = Options::default();
            sst_opts.set_merge_operator_associative("append_merge", append_merge);
            sst_opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(16));
            sst_opts.set_compression_type(DBCompressionType::Zstd);
            let mut writer = SstFileWriter::create(&sst_opts);
            writer.open(&sst_path)?;
            for (k, v) in &pending {
                writer.put(k, v)?;
            }
            writer.finish()?;
        }
        let mut ingest_opts = IngestExternalFileOptions::default();
        ingest_opts.set_move_files(false);
        output.ingest_external_file_opts(&ingest_opts, vec![sst_path.clone()])?;
        let _ = std::fs::remove_file(&sst_path);
    }

    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "Done. {} unique keys written to {} in {:.0}s ({:.1}h) via {} SST files",
        unique_written, output_path, elapsed, elapsed / 3600.0, sst_index
    );
    Ok(())
}

pub fn rocks_to_lmdb(
    rocks_path: &str,
    lmdb_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use lmdb::{Environment, EnvironmentFlags, DatabaseFlags, WriteFlags, Transaction};

    std::fs::create_dir_all(lmdb_path)?;

    // Bulk-load environment: NO_SYNC + WRITE_MAP + MAP_ASYNC with a single
    // explicit sync at the end (the output is rebuildable staging data until
    // validation passes). 6 TiB virtual map; max_dbs 600 leaves room for the
    // curated_XX shards added to the same environment later.
    let env = Environment::new()
        .set_flags(
            EnvironmentFlags::WRITE_MAP
                | EnvironmentFlags::MAP_ASYNC
                | EnvironmentFlags::NO_SYNC,
        )
        .set_map_size(6 * 1024 * 1024 * 1024 * 1024)
        .set_max_dbs(600)
        .open(std::path::Path::new(lmdb_path))?;

    let dbs: Vec<lmdb::Database> = (0u16..=255)
        .map(|s| env.create_db(Some(format!("{:02x}", s).as_str()), DatabaseFlags::empty()))
        .collect::<Result<_, _>>()?;

    // Reader thread decompresses RocksDB blocks while the single LMDB writer
    // appends. The RocksDB iterator yields keys in ascending order, so within
    // each shard (keyed by first byte) every put is a rightmost insert:
    // WriteFlags::APPEND skips the B-tree descent and packs pages ~full.
    // An out-of-order key would make LMDB return an error rather than
    // corrupt anything, and validation re-checks everything afterwards.
    let (tx, rx) = bounded::<Vec<(Box<[u8]>, Box<[u8]>)>>(32);
    let rocks_path_owned = rocks_path.to_string();
    let reader = std::thread::spawn(move || -> Result<(), String> {
        let mut ropts = Options::default();
        ropts.set_merge_operator_associative("append_merge", append_merge);
        let rocks = DB::open_for_read_only(&ropts, &rocks_path_owned, false)
            .map_err(|e| e.to_string())?;
        let mut read_opts = rocksdb::ReadOptions::default();
        read_opts.set_readahead_size(16 * 1024 * 1024);
        read_opts.set_verify_checksums(false);
        let mut batch: Vec<(Box<[u8]>, Box<[u8]>)> = Vec::with_capacity(65_536);
        for item in rocks.iterator_opt(rocksdb::IteratorMode::Start, read_opts) {
            let (key, value) = item.map_err(|e| e.to_string())?;
            batch.push((key, value));
            if batch.len() == 65_536 {
                if tx
                    .send(std::mem::replace(&mut batch, Vec::with_capacity(65_536)))
                    .is_err()
                {
                    return Err("lmdb writer hung up".to_string());
                }
            }
        }
        if !batch.is_empty() {
            let _ = tx.send(batch);
        }
        Ok(())
    });

    let start = std::time::Instant::now();
    let mut count = 0u64;
    let mut since_commit = 0u64;
    let mut txn = env.begin_rw_txn()?;
    for batch in rx {
        for (key, value) in &batch {
            let shard = key[0] as usize;
            txn.put(dbs[shard], &key.as_ref(), &value.as_ref(), WriteFlags::APPEND)?;
        }
        count += batch.len() as u64;
        since_commit += batch.len() as u64;
        if since_commit >= 4_000_000 {
            txn.commit()?;
            txn = env.begin_rw_txn()?;
            since_commit = 0;
            let el = start.elapsed().as_secs_f64();
            println!(
                "Inserted {} entries... ({:.0}/s, {:.1}h)",
                count,
                count as f64 / el.max(1e-9),
                el / 3600.0
            );
        }
    }
    txn.commit()?;
    reader
        .join()
        .map_err(|_| "rocks reader thread panicked")??;
    env.sync(true)?;
    let el = start.elapsed().as_secs_f64();
    println!(
        "Done. {} entries written to {} in {:.0}s ({:.1}h)",
        count, lmdb_path, el, el / 3600.0
    );
    Ok(())
}
