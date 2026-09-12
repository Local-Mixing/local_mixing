//! Rainbow-table (replacement database) generation pipeline.
//!
//! Ported 2026-07-06 from the dbgen server worktree (`/mnt/dbgen/local_mixing`,
//! `src/random/random_data.rs`), including the server-side optimizations made
//! that day: fresh-wire dedup in `abstract_gates_for_circuit_filtered`
//! (rocksdb_1), capped mapping enumeration in `build_from_2rocks` (rocksdb_2,
//! `for_each_mapping_capped`), PID-qualified temp SST names, and the
//! stop-flag-before-println Ctrl+C handler fix.
//!
//! Pipeline stages (see also the `rocksdb_1` / `rocksdb_2` /
//! `rocks_to_lmdb` CLI subcommands and the `merge_rocks_parallel` binary):
//!   1. `build_m1` — base case: all canonical 1-gate circuits.
//!   2. `build_from_rocks` (rocksdb_1) — extend an m-1 DB by one gate.
//!   3. `build_from_2rocks` (rocksdb_2) — combine an m1 DB and an m2 DB into
//!      an (m1+m2) DB over all wire overlaps.
//!   4. `merge_rocks_parallel` — merge explicit source DBs into one keyed DB.
//!   5. `rocks_to_lmdb` — convert to the sharded LMDB the mixing code reads.
//!
//! Deliberately NOT ported: the legacy LMDB-direct generator (`main_random`)
//! and the SQL/duckdb paths — the corrected pipeline does not use them.

#[cfg(test)]
#[path = "../tests/db_gen/regular_validation_tests.rs"]
mod validation_tests;

use crate::circuit::{CircuitSeq, Permutation, Polynomial, canonicalize_polys_4, polys_repr_blob};
use crossbeam_channel::bounded;
use itertools::Itertools;
#[cfg(test)]
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

type CanonicalCircuit = (Vec<Polynomial>, CircuitSeq, bool, Permutation, Vec<u16>);

/// A regular m-gate circuit can touch at most 3m wires. Keeping m <= 21
/// guarantees every generated circuit fits the u64 monomial representation.
pub const MAX_REGULAR_GATES: usize = 21;

/// Enumerate every ordered three-wire gate on `n` wires. This used to be a
/// general circuit helper, but it is only needed by the offline regular-DB
/// enumerator, so keep the compatibility surface local to this module.
fn base_gates(n: usize) -> Vec<[u16; 3]> {
    let mut gates = Vec::with_capacity(n.saturating_mul(n).saturating_mul(n));
    for target in 0..n as u16 {
        for control_a in 0..n as u16 {
            if control_a == target {
                continue;
            }
            for control_b in 0..n as u16 {
                if control_b != target && control_b != control_a {
                    gates.push([target, control_a, control_b]);
                }
            }
        }
    }
    gates
}

/// Serialize the legacy DB circuit payload after checking the two format
/// limits that the historical `repr_blob` helper silently truncated.
fn circuit_blob(circuit: &CircuitSeq) -> Vec<u8> {
    let bytes = circuit
        .gates
        .len()
        .checked_mul(3)
        .expect("regular DB circuit byte length overflow");
    assert!(
        bytes <= u8::MAX as usize,
        "regular DB circuit needs {bytes} bytes; legacy values allow at most 255"
    );
    let mut blob = Vec::with_capacity(bytes);
    for gate in &circuit.gates {
        for &wire in gate {
            blob.push(
                u8::try_from(wire)
                    .unwrap_or_else(|_| panic!("wire {wire} exceeds the legacy u8 DB format")),
            );
        }
    }
    blob
}

/// Compatibility adapter for the shuffletests generator's removed
/// bidirectional canonicalization API. It deliberately delegates the actual
/// canonical form to the current `canonicalize_polys_4`, because those bytes
/// define the keys consumed by the current runtime.
fn canonicalize_bidirectional(
    circuit: &CircuitSeq,
    allow_rule_l: bool,
) -> Option<CanonicalCircuit> {
    fn poly_vec_key(polys: &[Polynomial]) -> Vec<Vec<u64>> {
        polys
            .iter()
            .map(|poly| {
                let mut monomials = poly.clone();
                monomials.sort_unstable();
                monomials
            })
            .collect()
    }

    let used = circuit.used_wires();
    if used.is_empty() || used.len() > 64 {
        return None;
    }
    let mut dense = vec![0u16; used.last().copied().unwrap() as usize + 1];
    for (index, &wire) in used.iter().enumerate() {
        dense[wire as usize] = index as u16;
    }
    let remapped = CircuitSeq {
        gates: circuit
            .gates
            .iter()
            .map(|&[target, control_a, control_b]| {
                [
                    dense[target as usize],
                    dense[control_a as usize],
                    dense[control_b as usize],
                ]
            })
            .collect(),
    };

    let canonicalize_direction = |reversed: bool| {
        let mut canonical = remapped.clone();
        if reversed {
            canonical.gates.reverse();
        }
        canonical.canonicalize();
        let wires = canonical.max_wire() + 1;
        let (polys, permutation) = canonicalize_polys_4(
            canonical.to_polynomial(wires, 0, canonical.gates.len()),
            allow_rule_l,
        )
        .ok()?;
        canonical.rewire(&permutation.invert(), wires);
        canonical.canonicalize();
        Some((polys, canonical, permutation))
    };

    let (forward_polys, forward, forward_permutation) = canonicalize_direction(false)?;
    let (reverse_polys, reverse, reverse_permutation) = canonicalize_direction(true)?;
    let forward_key = poly_vec_key(&forward_polys);
    let reverse_key = poly_vec_key(&reverse_polys);

    Some(match forward_key.cmp(&reverse_key) {
        std::cmp::Ordering::Less => (forward_polys, forward, false, forward_permutation, used),
        std::cmp::Ordering::Greater => (reverse_polys, reverse, true, reverse_permutation, used),
        std::cmp::Ordering::Equal if forward.gates <= reverse.gates => {
            (forward_polys, forward, false, forward_permutation, used)
        }
        std::cmp::Ordering::Equal => (reverse_polys, reverse, true, reverse_permutation, used),
    })
}

fn require_uncapped_canonicalization() -> Result<(), Box<dyn std::error::Error>> {
    for variable in ["CANON_MONOMIAL_CAP", "CANON_RULE_L_BRANCH_CAP"] {
        if std::env::var_os(variable).is_some() {
            return Err(format!(
                "{variable} is set; unset canonicalization caps before generating a regular DB"
            )
            .into());
        }
    }
    Ok(())
}

fn require_supported_gate_count(m: usize) -> Result<(), Box<dyn std::error::Error>> {
    if !(1..=MAX_REGULAR_GATES).contains(&m) {
        return Err(format!(
            "regular DB gate count must be in 1..={MAX_REGULAR_GATES} so its 3m wires fit u64 monomials"
        )
        .into());
    }
    Ok(())
}

fn write_error(msg: &str) {
    eprintln!("{}", msg);
    if let Ok(mut f) = OpenOptions::new()
        .create(true)
        .append(true)
        .open("error.txt")
    {
        let _ = writeln!(f, "{}", msg);
    }
}

fn append_merge(_key: &[u8], existing: Option<&[u8]>, operands: &MergeOperands) -> Option<Vec<u8>> {
    if let Some(value) = existing {
        validate_value_chain(value).ok()?;
    }
    for operand in operands {
        validate_value_chain(operand).ok()?;
    }

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

/// Merge operator for the WIDE (MPX1) sidecar store: the same chunk-dedup
/// concatenation as `append_merge`, but walking MPX1 chunks — the legacy
/// `% 3` validation would reject every wide value by design.
pub fn append_merge_wide(
    _key: &[u8],
    existing: Option<&[u8]>,
    operands: &MergeOperands,
) -> Option<Vec<u8>> {
    let mut result: Vec<u8> = existing.unwrap_or(&[]).to_vec();
    for operand in operands {
        let mut pos = 0usize;
        while pos < operand.len() {
            let len = operand[pos] as usize;
            pos += 1;
            if len == 0 || pos + len > operand.len() {
                break;
            }
            let new_blob = &operand[pos..pos + len];
            pos += len;
            let mut rpos = 0usize;
            let mut found = false;
            while rpos < result.len() {
                let rlen = result[rpos] as usize;
                rpos += 1;
                if rlen == 0 || rpos + rlen > result.len() {
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

/// Open the wide sidecar output DB (`test_wide_db_m{m}`): same tuning as the
/// regular writer but with the MPX1-aware merge operator.
pub fn open_wide_db_for_write(m: usize) -> Result<DB, Box<dyn std::error::Error>> {
    open_wide_db_for_write_at(format!("test_wide_db_m{m}"))
}

/// Open the wide² sidecar output DB (`test_wide2_db_m{m}`): circuits carrying
/// TWO 3-control conjunction gates, derived from the wide store.
pub fn open_wide2_db_for_write(m: usize) -> Result<DB, Box<dyn std::error::Error>> {
    open_wide_db_for_write_at(format!("test_wide2_db_m{m}"))
}

fn open_wide_db_for_write_at(path: String) -> Result<DB, Box<dyn std::error::Error>> {
    require_uncapped_canonicalization()?;
    if std::path::Path::new(&path).exists() {
        return Err(format!("refusing existing wide DB output: {path}").into());
    }
    let mut opts = Options::default();
    opts.create_if_missing(true);
    opts.set_merge_operator_associative("append_merge_wide", append_merge_wide);
    opts.set_manual_wal_flush(true);
    opts.increase_parallelism(num_cpus::get() as i32);
    opts.set_max_background_jobs(64);
    opts.set_write_buffer_size(256 * 1024 * 1024);
    opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(16));
    opts.set_compression_type(rocksdb::DBCompressionType::Zstd);
    Ok(DB::open(&opts, path)?)
}

/// Read-only open of a wide sidecar store (MPX1 values; the wide merge
/// operator must be registered even for reads).
pub fn open_wide_db_for_read(path: &str) -> Result<DB, Box<dyn std::error::Error>> {
    let mut opts = Options::default();
    opts.create_if_missing(false);
    opts.set_max_open_files(-1);
    opts.set_merge_operator_associative("append_merge_wide", append_merge_wide);
    opts.set_disable_auto_compactions(true);
    Ok(DB::open_for_read_only(&opts, path, false)?)
}

/// S7 of the wide-gate design: extend band `m` by ONE 3-control conjunction
/// gate per candidate (append and prepend), writing MPX1 values keyed on the
/// FORWARD canonical function (the curated store's precedent; lookups probe
/// both frames). Reads `rocks_db_m{m}`, writes `test_wide_db_m{m}`.
///
/// v1 writer: direct batched merges — sized for the exhaustive small-band
/// tiers; the demand-driven tier gets the SST/router scale-out later.
pub fn build_wide_from_rocks(
    old_db: &Arc<DB>,
    new_db: &Arc<DB>,
    min_n: usize,
    max_n: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::circuit::xcanon::xgate_adjacent_id;
    use crate::circuit::xgate::XGate;
    use crate::db_generation::wide_gates::wide_gates_for_circuit_filtered;
    use crate::engine::mpx1;
    use crate::engine::xpoly::{XPolyBudget, canonicalize_xgates_single};

    require_uncapped_canonicalization()?;
    let total_rows = old_db
        .property_int_value("rocksdb.estimate-num-keys")
        .unwrap()
        .unwrap_or(0);
    println!("wide pass: estimated source rows {total_rows}");
    let budget = XPolyBudget::default();

    let start = std::time::Instant::now();
    let inserted = std::sync::atomic::AtomicU64::new(0);
    let skipped_classes = std::sync::atomic::AtomicU64::new(0);
    let rows = std::sync::atomic::AtomicU64::new(0);

    let iter = old_db.iterator(rocksdb::IteratorMode::Start);
    let entries: Result<Vec<(Box<[u8]>, Box<[u8]>)>, _> = iter.collect();
    let entries = entries.map_err(|e| format!("iterate source: {e}"))?;

    entries.par_chunks(1024).try_for_each(
        |chunk| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            let mut batch = rocksdb::WriteBatch::default();
            for (key, value) in chunk {
                let circuits =
                    decode_rocks_entry(key, value).map_err(std::io::Error::other)?;
                for old in &circuits {
                    let lifted: Vec<XGate> =
                        old.gates.iter().copied().map(XGate::from_g57).collect();
                    // Wire universe: 3m for the source band plus up to 4 fresh.
                    let touched = touched_wires(old);
                    let n = (touched.last().map_or(0, |&w| w as usize + 1))
                        .max(touched.len())
                        + 4;
                    let (gates, skip) =
                        wide_gates_for_circuit_filtered(&touched, n, min_n, max_n);
                    skipped_classes
                        .fetch_add(skip as u64, std::sync::atomic::Ordering::Relaxed);
                    for g in gates {
                        for prepend in [false, true] {
                            let mut circ = Vec::with_capacity(lifted.len() + 1);
                            if prepend {
                                circ.push(g.clone());
                                circ.extend(lifted.iter().cloned());
                            } else {
                                circ.extend(lifted.iter().cloned());
                                circ.push(g.clone());
                            }
                            if xgate_adjacent_id(&circ) {
                                continue;
                            }
                            // Min-dir keying, mirroring the regular store's
                            // convention: canonicalize BOTH directions and key
                            // under the lexicographically smaller serialized
                            // polys; store the winning direction's
                            // representative. One key per {F, F^-1} pair.
                            let canon_f =
                                match canonicalize_xgates_single(&circ, false, budget) {
                                    Ok(c) => c,
                                    Err(_) => continue, // budget/degree: skip candidate
                                };
                            let canon_r =
                                match canonicalize_xgates_single(&circ, true, budget) {
                                    Ok(c) => c,
                                    Err(_) => continue,
                                };
                            let blob_f = polys_repr_blob(&canon_f.polys);
                            let blob_r = polys_repr_blob(&canon_r.polys);
                            let (canon, blob, store_reversed) = if blob_r < blob_f {
                                (canon_r, blob_r, true)
                            } else {
                                (canon_f, blob_f, false)
                            };
                            let db_key = xxh3_128(&blob).to_le_bytes();
                            // Store in CANONICAL wire space: global -> dense
                            // (position in used_wires) -> canonical (inverse
                            // of order.data, which maps canonical -> dense).
                            let mut inv = vec![0u16; canon.order.data.len()];
                            for (c, &d) in canon.order.data.iter().enumerate() {
                                if (d as usize) < inv.len() {
                                    inv[d as usize] = c as u16;
                                }
                            }
                            let to_canonical = |w: u16| -> u16 {
                                let dense = canon
                                    .used_wires
                                    .binary_search(&w)
                                    .expect("gate wire is a used wire")
                                    as u16;
                                inv.get(dense as usize).copied().unwrap_or(dense)
                            };
                            let stored_src: Vec<XGate> = if store_reversed {
                                circ.iter().rev().cloned().collect()
                            } else {
                                circ.clone()
                            };
                            let mut mapped: Vec<XGate> = stored_src
                                .iter()
                                .map(|g| {
                                    XGate {
                                        target: to_canonical(g.target),
                                        comp: g.comp,
                                        ctrls: g
                                            .ctrls
                                            .iter()
                                            .map(|&(w, p)| (to_canonical(w), p))
                                            .collect(),
                                    }
                                })
                                .collect();
                            // Gate-order canonical form for the representative
                            // (legacy re-canonicalizes after its rewire too).
                            crate::circuit::xcanon::xgate_canonicalize(&mut mapped);
                            let chunk_bytes = mpx1::encode_circuit(&mapped)
                                .map_err(std::io::Error::other)?;
                            batch.merge(db_key, chunk_bytes);
                            inserted.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                }
                let r =
                    rows.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if r % 100_000 == 0 {
                    let el = start.elapsed().as_secs_f64();
                    println!(
                        "wide: rows {r}/{total_rows} | inserted {} | skipped-classes {} | {:.0} rows/s",
                        inserted.load(std::sync::atomic::Ordering::Relaxed),
                        skipped_classes.load(std::sync::atomic::Ordering::Relaxed),
                        r as f64 / el,
                    );
                }
            }
            new_db
                .write(batch)
                .map_err(|e| std::io::Error::other(format!("wide batch write: {e}")))?;
            Ok(())
        },
    )
    .map_err(|e| e as Box<dyn std::error::Error>)?;

    println!(
        "wide pass done: rows {} | inserted {} | elapsed {:.0}s",
        rows.load(std::sync::atomic::Ordering::Relaxed),
        inserted.load(std::sync::atomic::Ordering::Relaxed),
        start.elapsed().as_secs_f64()
    );
    println!("Compacting wide_db...");
    new_db.compact_range::<&[u8], &[u8]>(None, None);
    println!("Build finished.");
    Ok(())
}

/// Wires touched by an XGate circuit (targets and control wires), sorted.
fn touched_wires_x(gates: &[crate::circuit::xgate::XGate]) -> Vec<u16> {
    let mut ws: Vec<u16> = gates
        .iter()
        .flat_map(|g| std::iter::once(g.target).chain(g.ctrls.iter().map(|&(w, _)| w)))
        .collect();
    ws.sort_unstable();
    ws.dedup();
    ws
}

/// Wide² tier: extend a WIDE store's circuits (k g57 gates + one conjunction)
/// with a SECOND 3-control conjunction at either end. Same min-dir keying and
/// canonical-space representative as `build_wide_from_rocks`; the source is
/// MPX1-decoded instead of g57-lifted. Reads `wide_db_m{m}`, writes
/// `test_wide2_db_m{m}`. Exhaustive tiers only (m1-m3): each placement
/// multiplies candidates by ~8·n·C(n-1,3), so m4+ is demand-driven.
pub fn build_wide2_from_wide(
    old_db: &Arc<DB>,
    new_db: &Arc<DB>,
    min_n: usize,
    max_n: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::circuit::xcanon::xgate_adjacent_id;
    use crate::circuit::xgate::XGate;
    use crate::db_generation::wide_gates::wide_gates_for_circuit_filtered;
    use crate::engine::mpx1;
    use crate::engine::xpoly::{XPolyBudget, canonicalize_xgates_single};

    require_uncapped_canonicalization()?;
    let total_rows = old_db
        .property_int_value("rocksdb.estimate-num-keys")
        .unwrap()
        .unwrap_or(0);
    println!("wide2 pass: estimated source rows {total_rows}");
    let budget = XPolyBudget::default();

    let start = std::time::Instant::now();
    let inserted = std::sync::atomic::AtomicU64::new(0);
    let skipped_classes = std::sync::atomic::AtomicU64::new(0);
    let rows = std::sync::atomic::AtomicU64::new(0);

    let iter = old_db.iterator(rocksdb::IteratorMode::Start);
    let entries: Result<Vec<(Box<[u8]>, Box<[u8]>)>, _> = iter.collect();
    let entries = entries.map_err(|e| format!("iterate wide source: {e}"))?;

    entries.par_chunks(256).try_for_each(
        |chunk| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            let mut batch = rocksdb::WriteBatch::default();
            for (key, value) in chunk {
                if key.len() != 16 {
                    return Err(std::io::Error::other(format!(
                        "invalid wide source key length {}",
                        key.len()
                    ))
                    .into());
                }
                let circuits = mpx1::decode_value(value).map_err(std::io::Error::other)?;
                for lifted in &circuits {
                    let touched = touched_wires_x(lifted);
                    // Wire universe: the stored canonical span plus up to 4 fresh.
                    let n = (touched.last().map_or(0, |&w| w as usize + 1))
                        .max(touched.len())
                        + 4;
                    let (gates, skip) =
                        wide_gates_for_circuit_filtered(&touched, n, min_n, max_n);
                    skipped_classes
                        .fetch_add(skip as u64, std::sync::atomic::Ordering::Relaxed);
                    for g in gates {
                        for prepend in [false, true] {
                            let mut circ = Vec::with_capacity(lifted.len() + 1);
                            if prepend {
                                circ.push(g.clone());
                                circ.extend(lifted.iter().cloned());
                            } else {
                                circ.extend(lifted.iter().cloned());
                                circ.push(g.clone());
                            }
                            if xgate_adjacent_id(&circ) {
                                continue;
                            }
                            let canon_f =
                                match canonicalize_xgates_single(&circ, false, budget) {
                                    Ok(c) => c,
                                    Err(_) => continue,
                                };
                            let canon_r =
                                match canonicalize_xgates_single(&circ, true, budget) {
                                    Ok(c) => c,
                                    Err(_) => continue,
                                };
                            let blob_f = polys_repr_blob(&canon_f.polys);
                            let blob_r = polys_repr_blob(&canon_r.polys);
                            let (canon, blob, store_reversed) = if blob_r < blob_f {
                                (canon_r, blob_r, true)
                            } else {
                                (canon_f, blob_f, false)
                            };
                            let db_key = xxh3_128(&blob).to_le_bytes();
                            let mut inv = vec![0u16; canon.order.data.len()];
                            for (c, &d) in canon.order.data.iter().enumerate() {
                                if (d as usize) < inv.len() {
                                    inv[d as usize] = c as u16;
                                }
                            }
                            let to_canonical = |w: u16| -> u16 {
                                let dense = canon
                                    .used_wires
                                    .binary_search(&w)
                                    .expect("gate wire is a used wire")
                                    as u16;
                                inv.get(dense as usize).copied().unwrap_or(dense)
                            };
                            let stored_src: Vec<XGate> = if store_reversed {
                                circ.iter().rev().cloned().collect()
                            } else {
                                circ.clone()
                            };
                            let mut mapped: Vec<XGate> = stored_src
                                .iter()
                                .map(|g| {
                                    XGate {
                                        target: to_canonical(g.target),
                                        comp: g.comp,
                                        ctrls: g
                                            .ctrls
                                            .iter()
                                            .map(|&(w, p)| (to_canonical(w), p))
                                            .collect(),
                                    }
                                })
                                .collect();
                            crate::circuit::xcanon::xgate_canonicalize(&mut mapped);
                            let chunk_bytes = mpx1::encode_circuit(&mapped)
                                .map_err(std::io::Error::other)?;
                            batch.merge(db_key, chunk_bytes);
                            inserted.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                }
                let r = rows.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if r % 100_000 == 0 {
                    let el = start.elapsed().as_secs_f64();
                    println!(
                        "wide2: rows {r}/{total_rows} | inserted {} | skipped-classes {} | {:.0} rows/s",
                        inserted.load(std::sync::atomic::Ordering::Relaxed),
                        skipped_classes.load(std::sync::atomic::Ordering::Relaxed),
                        r as f64 / el,
                    );
                }
            }
            new_db
                .write(batch)
                .map_err(|e| std::io::Error::other(format!("wide2 batch write: {e}")))?;
            Ok(())
        },
    )
    .map_err(|e| e as Box<dyn std::error::Error>)?;

    println!(
        "wide2 pass done: rows {} | inserted {} | elapsed {:.0}s",
        rows.load(std::sync::atomic::Ordering::Relaxed),
        inserted.load(std::sync::atomic::Ordering::Relaxed),
        start.elapsed().as_secs_f64()
    );
    println!("Compacting wide2_db...");
    new_db.compact_range::<&[u8], &[u8]>(None, None);
    println!("Build finished.");
    Ok(())
}

pub fn open_db_for_write(m: usize) -> Result<DB, Box<dyn std::error::Error>> {
    require_uncapped_canonicalization()?;
    require_supported_gate_count(m)?;
    let path = format!("test_rocks_db_m{}", m);
    if std::path::Path::new(&path).exists() {
        return Err(format!("refusing existing regular DB output: {path}").into());
    }
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

    Ok(DB::open(&opts, path)?)
}

pub fn open_db_for_read(m: usize) -> DB {
    let name = format!("rocks_db_m{}", m);
    let path = std::env::var_os("REGULAR_DB_DIR")
        .map(std::path::PathBuf::from)
        .map(|directory| directory.join(&name))
        .unwrap_or_else(|| name.into());
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

    DB::open_for_read_only(&opts, &path, false).unwrap_or_else(|error| {
        panic!("failed to open regular RocksDB {}: {error}", path.display())
    })
}

/// Encode a single circuit blob as a length-prefixed entry
fn encode_circuit(circuit_blob: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(1 + circuit_blob.len());
    v.push(
        u8::try_from(circuit_blob.len())
            .expect("regular DB circuit exceeds the legacy 255-byte value format"),
    );
    v.extend_from_slice(circuit_blob);
    v
}

fn validate_value_chain(value: &[u8]) -> Result<(), String> {
    if value.is_empty() {
        return Err("empty replacement value".to_string());
    }
    let mut position = 0usize;
    while position < value.len() {
        let length = value[position] as usize;
        position += 1;
        if length == 0 || length % 3 != 0 {
            return Err(format!(
                "invalid circuit length {length} at value offset {}",
                position - 1
            ));
        }
        if position + length > value.len() {
            return Err(format!(
                "truncated circuit at value offset {}: need {length} bytes, have {}",
                position - 1,
                value.len() - position
            ));
        }
        position += length;
    }
    Ok(())
}

fn validate_rocks_entry(key: &[u8], value: &[u8]) -> Result<(), String> {
    if key.len() != 16 {
        return Err(format!(
            "invalid regular DB key length {} (expected 16 bytes): {key:02x?}",
            key.len()
        ));
    }
    validate_value_chain(value)
        .map_err(|error| format!("malformed RocksDB value for key {key:02x?}: {error}"))
}

pub fn decode_rocks_entry(key: &[u8], value: &[u8]) -> Result<Vec<CircuitSeq>, String> {
    validate_rocks_entry(key, value)?;
    let mut circuits = Vec::new();
    let mut position = 0usize;
    while position < value.len() {
        let length = value[position] as usize;
        position += 1;
        let circuit = CircuitSeq::from_blob(&value[position..position + length]);
        if circuit.gates.iter().any(|[target, control_a, control_b]| {
            target == control_a || target == control_b || control_a == control_b
        }) {
            return Err(format!(
                "malformed RocksDB value for key {key:02x?}: gate wires must be distinct"
            ));
        }
        if circuit.used_wires().len() > 64 {
            return Err(format!(
                "malformed RocksDB value for key {key:02x?}: circuit exceeds the 64-wire monomial ABI"
            ));
        }
        circuits.push(circuit);
        position += length;
    }
    Ok(circuits)
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

fn flush_to_sst(
    db: &Arc<DB>,
    pending: &mut Vec<(Vec<u8>, Vec<u8>)>,
    sst_index: &mut usize,
    part: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if pending.is_empty() {
        return Ok(());
    }

    pending.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
    let merged = merge_sorted_entries(std::mem::take(pending));

    // Use /dev/shm (tmpfs, separate from DB disk) to avoid filling /dev/sda3.
    // PID- and partition-qualified so concurrent builds and concurrent writer
    // threads can never collide on temp SST names.
    let sst_path = format!(
        "/dev/shm/sst_{}_{}_{}.sst",
        std::process::id(),
        part,
        sst_index
    );
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
    let untouched: Vec<u16> = (0..n as u16).filter(|w| !touched.contains(w)).collect();

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
                if b == a {
                    continue;
                }
                for &c in &touched {
                    if c == a || c == b {
                        continue;
                    }
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
                    if c == b {
                        continue;
                    }
                    result.push([u0, b, c]);
                }
            }
            for &a in &touched {
                for &c in &touched {
                    if c == a {
                        continue;
                    }
                    result.push([a, u0, c]);
                }
            }
            for &a in &touched {
                for &b in &touched {
                    if b == a {
                        continue;
                    }
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
    require_uncapped_canonicalization()?;
    require_supported_gate_count(m)?;
    if m == 1 {
        return Err("build_from_rocks requires m >= 2; use build_m1 for the base case".into());
    }
    println!("Running build (max CPU)");

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build_global()
        .ok();

    let total_rows = old_db
        .property_int_value("rocksdb.estimate-num-keys")
        .unwrap()
        .unwrap_or(0);
    println!("Estimated rows: {}", total_rows);

    let shard_mod: usize = std::env::var("LM_SOURCE_SHARDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&k| k >= 1)
        .unwrap_or(1);
    let shard_idx: usize = std::env::var("LM_SOURCE_SHARD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if shard_mod > 1 {
        if shard_idx >= shard_mod {
            return Err(format!(
                "LM_SOURCE_SHARD ({shard_idx}) must be < LM_SOURCE_SHARDS ({shard_mod})"
            )
            .into());
        }
        println!(
            "Source sharding: slice {shard_idx}/{shard_mod} (rows where key[0] % {shard_mod} == {shard_idx})"
        );
    }

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

    // Output-side range partitioning: LM_WRITERS parallel writer threads, each
    // owning a contiguous key[0] range. Disjoint ranges keep ingested SSTs from
    // overlapping across writers, so L0 compaction debt stays per-partition
    // (the single global writer measured ~1M pairs/s and unbounded L0 debt).
    // Default 1 preserves the historical single-writer pipeline byte-for-byte.
    let n_writers: usize = std::env::var("LM_WRITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n: &usize| (1..=32).contains(&n) && n.is_power_of_two())
        .unwrap_or(1);
    let part_shift = 8 - n_writers.trailing_zeros() as usize;
    if n_writers > 1 {
        println!("Partitioned writers: {n_writers} range partitions (key[0] >> {part_shift})");
    }

    let (tx, rx) = bounded::<Vec<(Vec<u8>, Vec<u8>)>>(1_000);
    let mut writer_txs: Vec<crossbeam_channel::Sender<Vec<(Vec<u8>, Vec<u8>)>>> =
        Vec::with_capacity(n_writers);
    let mut writer_handles: Vec<std::thread::JoinHandle<Result<(), String>>> =
        Vec::with_capacity(n_writers + 1);
    for p in 0..n_writers {
        let (wtx, wrx) = bounded::<Vec<(Vec<u8>, Vec<u8>)>>(256);
        writer_txs.push(wtx);
        let db = Arc::clone(new_db);
        let stop = stop_flag.clone();
        writer_handles.push(std::thread::spawn(move || -> Result<(), String> {
            let mut pending: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
            let mut sst_index = 0usize;
            while let Ok(part) = wrx.recv() {
                if stop.load(Ordering::SeqCst) {
                    break;
                }
                pending.extend(part);
                if pending.len() >= 200_000 {
                    if let Err(e) = flush_to_sst(&db, &mut pending, &mut sst_index, p) {
                        write_error(&format!("Writer {p}: flush failed: {}", e));
                        return Err(format!("regular DB writer {p} flush failed: {e}"));
                    }
                }
            }
            if !pending.is_empty() {
                if let Err(e) = flush_to_sst(&db, &mut pending, &mut sst_index, p) {
                    write_error(&format!("Writer {p}: final flush failed: {}", e));
                    return Err(format!("regular DB writer {p} final flush failed: {e}"));
                }
            }
            Ok(())
        }));
    }

    let stop_flag_clone = stop_flag.clone();
    let skipped_count_insert = Arc::clone(&skipped_count);

    let insert_handle = std::thread::spawn(move || -> Result<(), String> {
        let start_time = std::time::Instant::now();
        let mut attempted_inserts = 0usize;
        // Batches arrive far more often than progress is useful; throttle to ~1 line/s.
        let mut last_progress = std::time::Instant::now();
        let mut first_progress = true;

        while let Ok(batch) = rx.recv() {
            if stop_flag_clone.load(Ordering::SeqCst) {
                println!("Insertion thread stopping early...");
                break;
            }

            let batch_len = batch.len();
            if n_writers == 1 {
                if writer_txs[0].send(batch).is_err() {
                    return Err("regular DB writer channel closed early".to_string());
                }
            } else {
                let mut parts: Vec<Vec<(Vec<u8>, Vec<u8>)>> =
                    (0..n_writers).map(|_| Vec::new()).collect();
                for (key, value) in batch {
                    let p = (key[0] as usize) >> part_shift;
                    parts[p].push((key, value));
                }
                for (p, part) in parts.into_iter().enumerate() {
                    if !part.is_empty() && writer_txs[p].send(part).is_err() {
                        return Err(format!("regular DB writer {p} channel closed early"));
                    }
                }
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
                let rate_rows = if elapsed > 0.0 {
                    rows_done as f64 / elapsed
                } else {
                    0.0
                };
                let pct = if total_rows > 0 {
                    rows_done as f64 / total_rows as f64 * 100.0
                } else {
                    0.0
                };
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
        }

        let elapsed = start_time.elapsed().as_secs_f64();
        println!(
            "Insertion thread finished. Total inserted: {} | elapsed: {:.0}s",
            attempted_inserts, elapsed,
        );
        Ok(())
    });
    writer_handles.push(insert_handle);

    let iter = old_db.iterator(rocksdb::IteratorMode::Start);
    let mut source_error: Option<String> = None;

    for chunk in &iter.chunks(chunk_size) {
        if stop_flag.load(Ordering::SeqCst) {
            break;
        }

        // Collect raw pairs serially (cheap memcpy), decode in the worker pool:
        // the serial per-chunk decode was ~78% of wall clock on wide sources.
        let raw_pairs = chunk
            .map(|item| item.map_err(|error| format!("RocksDB iterator error: {error}")))
            .filter(|item| match item {
                Ok((key, _)) => shard_mod <= 1 || (key[0] as usize) % shard_mod == shard_idx,
                Err(_) => true,
            })
            .collect::<Result<Vec<_>, String>>();
        let entries = raw_pairs.and_then(|pairs| {
            pairs
                .par_iter()
                .map(|(key, value)| {
                    let circuits = decode_rocks_entry(key, value)?;
                    if let Some(circuit) =
                        circuits.iter().find(|circuit| circuit.gates.len() != m - 1)
                    {
                        return Err(format!(
                            "rocksdb_1 source key {key:02x?} contains a {}-gate circuit; expected m-1 = {}",
                            circuit.gates.len(),
                            m - 1
                        ));
                    }
                    Ok(circuits)
                })
                .collect::<Result<Vec<Vec<CircuitSeq>>, String>>()
        });
        let entries = match entries {
            Ok(entries) => entries,
            Err(error) => {
                source_error = Some(error);
                stop_flag.store(true, Ordering::SeqCst);
                break;
            }
        };

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

            for circuits in entry_chunk {
                for old_circuit in circuits {
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
                            match canonicalize_bidirectional(&c1, !no_rule_l) {
                                None => {
                                    local_no_rule_l_skipped += 1;
                                }
                                Some(canon1) => {
                                    let c1_hash: u128 = xxh3_128(&polys_repr_blob(&canon1.0));
                                    let c1_value = encode_circuit(&circuit_blob(&canon1.1));
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
                            match canonicalize_bidirectional(&c2, !no_rule_l) {
                                None => {
                                    local_no_rule_l_skipped += 1;
                                }
                                Some(canon2) => {
                                    let c2_hash: u128 = xxh3_128(&polys_repr_blob(&canon2.0));
                                    let c2_value = encode_circuit(&circuit_blob(&canon2.1));
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
                            stop_flag_par.store(true, Ordering::SeqCst);
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
                    stop_flag_par.store(true, Ordering::SeqCst);
                }
            }
            total_gates_tried_par.fetch_add(local_tried, Ordering::Relaxed);
            skipped_par.fetch_add(local_skipped, Ordering::Relaxed);
            no_rule_l_skipped_par.fetch_add(local_no_rule_l_skipped, Ordering::Relaxed);
        });
    }

    drop(tx);
    for handle in writer_handles {
        handle
            .join()
            .map_err(|_| "regular DB writer thread panicked")?
            .map_err(|error| -> Box<dyn std::error::Error> { error.into() })?;
    }
    if let Some(error) = source_error {
        return Err(error.into());
    }
    if no_rule_l {
        println!(
            "Skipped (rule L required): {}",
            no_rule_l_skipped.load(Ordering::Relaxed)
        );
    }

    if stop_flag.load(Ordering::SeqCst) {
        return Err(
            "regular DB build interrupted; output is partial and must not be promoted".into(),
        );
    }

    // The full-range compact can take longer than enumeration itself on
    // deep bands (measured 11h vs 8h on m7 [11,11] slices); an external
    // 256-way self-merge with merge_rocks_parallel produces the same fully
    // compacted content ~10x faster. Compaction is content-preserving, so
    // skipping it here never changes the logical DB.
    if std::env::var_os("LM_SKIP_FINAL_COMPACT").is_some() {
        println!("Skipping final compaction (LM_SKIP_FINAL_COMPACT set; finalize externally).");
    } else {
        println!("Compacting new_db for optimal read performance...");
        new_db.compact_range::<&[u8], &[u8]>(None, None);
        println!("Compaction done.");
    }
    println!("Build finished.");
    Ok(())
}

pub fn build_m1(new_db: &Arc<DB>) -> Result<(), Box<dyn std::error::Error>> {
    require_uncapped_canonicalization()?;
    println!("Building m1 base case");

    let gates = base_gates(3);
    let mut pending: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    let mut sst_index = 0usize;

    for g in gates.iter() {
        let c = CircuitSeq { gates: vec![*g] };
        let Some(canon) = canonicalize_bidirectional(&c, true) else {
            return Err("m1 canonicalization unexpectedly skipped".into());
        };

        if canon.1.adjacent_id() {
            continue;
        }

        let canon_blob = polys_repr_blob(&canon.0);
        let hash: u128 = xxh3_128(&canon_blob);
        let key = hash.to_le_bytes().to_vec();

        let circuit_blob = circuit_blob(&canon.1);
        let value = encode_circuit(&circuit_blob);

        pending.push((key, value));
    }

    flush_to_sst(new_db, &mut pending, &mut sst_index, 0)?;

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
            .map(|&[a, b, c]| {
                [
                    mapping[a as usize],
                    mapping[b as usize],
                    mapping[c as usize],
                ]
            })
            .collect(),
    }
}

// Cache: (n1, n2) -> Arc<(flat_mappings, stride=n2)>
// flat_mappings is all mappings concatenated contiguously.
// mapping i is at flat[i*n2..(i+1)*n2].
// Entries with more mappings than this are streamed on-the-fly instead of cached.
#[cfg(test)]
const LARGE_MAPPING_THRESHOLD: usize = 200_000;
// Max number of entries kept in the LRU cache.
#[cfg(test)]
const MAPPING_CACHE_CAP: usize = 256;

#[cfg(test)]
static MAPPING_CACHE: Lazy<
    std::sync::Mutex<lru::LruCache<(usize, usize), Arc<(Vec<u16>, usize)>>>,
> = Lazy::new(|| {
    std::sync::Mutex::new(lru::LruCache::new(
        std::num::NonZeroUsize::new(MAPPING_CACHE_CAP).unwrap(),
    ))
});

/// Call `f` once per mapping for the (n1, n2) pair.
/// Small pairs are cached in an LRU; large pairs are enumerated on-the-fly.
#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
fn compute_mappings(n1: usize, n2: usize) -> (Vec<u16>, usize) {
    let total = count_mappings(n1, n2);
    let mut flat = vec![0u16; total * n2.max(1)];
    let mut idx = 0usize;
    let mut c2_to_wire = vec![0u16; n2];
    let mut used = vec![false; n2];

    enumerate_direct(0, n1, n2, &mut c2_to_wire, &mut used, &mut flat, &mut idx);

    debug_assert_eq!(idx, total);
    (flat, n2)
}

#[cfg(test)]
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

#[cfg(test)]
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
    require_uncapped_canonicalization()?;
    if m1 == 0 || m2 == 0 {
        return Err("regular DB source gate counts must both be nonzero".into());
    }
    let m = m1
        .checked_add(m2)
        .ok_or("regular DB source gate-count sum overflow")?;
    require_supported_gate_count(m)?;
    println!("Running build_from_2rocks: m1={m1} m2={m2} -> m={m}");

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
            let (key, value) = item?;
            let decoded = decode_rocks_entry(&key, &value)
                .map_err(|error| -> Box<dyn std::error::Error> { error.into() })?;
            if let Some(circuit) = decoded.iter().find(|circuit| circuit.gates.len() != m2) {
                return Err(format!(
                    "rocksdb_2 second source key {key:02x?} contains a {}-gate circuit; expected m2 = {m2}",
                    circuit.gates.len()
                )
                .into());
            }
            circuits.extend(decoded);
        }
        println!("Loaded {} circuits from db2", circuits.len());
        circuits
    });

    // Precompute c2_rev for every c2 once
    let total_c2 = db2_circuits.len();
    println!("Precomputing c2_rev (0/{})...", total_c2);
    let c2_rev_done = std::sync::atomic::AtomicUsize::new(0);
    let db2_rev: Arc<Vec<CircuitSeq>> = Arc::new(
        db2_circuits
            .par_iter()
            .map(|c2| {
                let mut r = CircuitSeq {
                    gates: c2.gates.iter().rev().cloned().collect(),
                };
                r.canonicalize();
                // Remap to minimal wires
                let used = r.used_wires();
                let wire_map: HashMap<u16, u16> = used
                    .iter()
                    .enumerate()
                    .map(|(i, &w)| (w, i as u16))
                    .collect();
                r = CircuitSeq {
                    gates: r
                        .gates
                        .iter()
                        .map(|&[t, c1, c2]| [wire_map[&t], wire_map[&c1], wire_map[&c2]])
                        .collect(),
                };
                r.canonicalize();
                let n2 = r.max_wire() as usize + 1;
                let canon =
                    canonicalize_polys_4(r.to_polynomial(n2, 0, r.gates.len()), true).unwrap();
                r.rewire(&canon.1.invert(), n2);
                r.canonicalize();
                let done = c2_rev_done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if done % 50 == 0 || done == total_c2 {
                    println!("Precomputing c2_rev ({}/{})...", done, total_c2);
                }
                r
            })
            .collect(),
    );

    // Precompute touched wire counts for every c2 and c2_rev
    let db2_n2: Arc<Vec<usize>> = Arc::new(
        db2_circuits
            .par_iter()
            .map(|c| touched_wires(c).len())
            .collect(),
    );
    let db2_rev_n2: Arc<Vec<usize>> =
        Arc::new(db2_rev.par_iter().map(|c| touched_wires(c).len()).collect());

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
    println!(
        "Estimated total pairs: {} ({:.2}B)",
        total_pairs_est,
        total_pairs_est as f64 / 1e9
    );
    println!(
        "chunk_size={} batch_size={} channel_cap=1000 pending_threshold=1M",
        chunk_size, batch_size
    );

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

    let insert_handle = std::thread::spawn(move || -> Result<(), String> {
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
            let pairs_rate = if elapsed > 0.0 {
                pairs_done as f64 / elapsed
            } else {
                0.0
            };
            let pairs_remaining = if pairs_done < total_pairs_est {
                total_pairs_est - pairs_done
            } else {
                0
            };
            let eta_secs = if pairs_rate > 0.0 {
                (pairs_remaining as f64 / pairs_rate) as u64
            } else {
                0
            };
            let dedup_ratio = if results_so_far > 0 {
                attempted_inserts as f64 / results_so_far as f64
            } else {
                0.0
            };
            println!(
                "[writer] pairs={}/{} ({:.1}%) | pairs/s={:.0} | eta={:02}:{:02}:{:02} | inserts={} | results={} | dedup={:.1}x | pending={} | ssts={}",
                pairs_done,
                total_pairs_est,
                (pairs_done as f64 / total_pairs_est as f64) * 100.0,
                pairs_rate,
                eta_secs / 3600,
                (eta_secs % 3600) / 60,
                eta_secs % 60,
                attempted_inserts + skipped,
                results_so_far,
                dedup_ratio,
                pending.len(),
                sst_count,
            );

            if pending.len() >= 1_000_000 {
                if let Err(e) = flush_to_sst(&new_db_writer, &mut pending, &mut sst_index, 0) {
                    write_error(&format!("Writer thread: flush failed: {}", e));
                    return Err(format!("regular DB writer flush failed: {e}"));
                }
                sst_count += 1;
            }
        }

        println!(
            "[writer] producers done, flushing {} remaining entries across {} final SSTs...",
            pending.len() + attempted_inserts - attempted_inserts, // pending count
            (pending.len() + 999_999) / 1_000_000
        );
        while !pending.is_empty() {
            if let Err(e) = flush_to_sst(&new_db_writer, &mut pending, &mut sst_index, 0) {
                write_error(&format!("Writer thread: final flush failed: {}", e));
                return Err(format!("regular DB writer final flush failed: {e}"));
            }
            sst_count += 1;
            println!(
                "[writer] final flush: {} remaining | sst #{}",
                pending.len(),
                sst_count
            );
        }
        let elapsed = start_time.elapsed().as_secs_f64();
        println!(
            "[writer] finished. total_inserts={} | total_ssts={} | elapsed={:.0}s ({:.1}h)",
            attempted_inserts,
            sst_count,
            elapsed,
            elapsed / 3600.0,
        );
        Ok(())
    });

    // ── Main loop: stream db1 in chunks ──────────────────────────────────────
    let iter = db1.iterator(rocksdb::IteratorMode::Start);
    let mut chunk_idx = 0usize;
    let mut total_c1_processed = 0usize;
    let mut source_error: Option<String> = None;

    for chunk in &iter.chunks(chunk_size) {
        if stop_flag.load(Ordering::SeqCst) {
            break;
        }
        chunk_idx += 1;
        let elapsed_outer = build_start.elapsed().as_secs_f64();
        let pairs_done = total_gates_tried.load(Ordering::Relaxed);
        let outer_rate = if elapsed_outer > 0.0 {
            pairs_done as f64 / elapsed_outer
        } else {
            0.0
        };
        let pairs_remaining = if pairs_done < total_pairs_est {
            total_pairs_est - pairs_done
        } else {
            0
        };
        let eta_outer = if outer_rate > 0.0 {
            (pairs_remaining as f64 / outer_rate) as u64
        } else {
            0
        };
        println!(
            "[chunk {}] c1_processed={} | pairs={}/{} ({:.1}%) | {:.0} pairs/s | eta {:02}:{:02}:{:02} | elapsed={:.1}h",
            chunk_idx,
            total_c1_processed,
            pairs_done,
            total_pairs_est,
            (pairs_done as f64 / total_pairs_est as f64) * 100.0,
            outer_rate,
            eta_outer / 3600,
            (eta_outer % 3600) / 60,
            eta_outer % 60,
            elapsed_outer / 3600.0,
        );

        let entries = chunk
            .map(|item| {
                let (key, value) =
                    item.map_err(|error| format!("RocksDB iterator error: {error}"))?;
                let circuits = decode_rocks_entry(&key, &value)?;
                if let Some(circuit) = circuits.iter().find(|circuit| circuit.gates.len() != m1) {
                    return Err(format!(
                        "rocksdb_2 first source key {key:02x?} contains a {}-gate circuit; expected m1 = {m1}",
                        circuit.gates.len()
                    ));
                }
                Ok(circuits)
            })
            .collect::<Result<Vec<Vec<CircuitSeq>>, String>>();

        let c1_circuits: Vec<CircuitSeq> = match entries {
            Ok(entries) => entries.into_iter().flatten().collect(),
            Err(error) => {
                source_error = Some(error);
                stop_flag.store(true, Ordering::SeqCst);
                break;
            }
        };

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
                    let mut r = CircuitSeq {
                        gates: c1.gates.iter().rev().cloned().collect(),
                    };
                    r.canonicalize();
                    let used = r.used_wires();
                    let wire_map: HashMap<u16, u16> = used
                        .iter()
                        .enumerate()
                        .map(|(i, &w)| (w, i as u16))
                        .collect();
                    r = CircuitSeq {
                        gates: r
                            .gates
                            .iter()
                            .map(|&[t, c1, c2]| [wire_map[&t], wire_map[&c1], wire_map[&c2]])
                            .collect(),
                    };
                    r.canonicalize();
                    let n1r = r.max_wire() as usize + 1;
                    let canon =
                        canonicalize_polys_4(r.to_polynomial(n1r, 0, r.gates.len()), true).unwrap();
                    r.rewire(&canon.1.invert(), n1r);
                    r.canonicalize();
                    r
                };
                let n1_rev = touched_wires(&c1_rev).len();
                C1Data {
                    c1,
                    n1,
                    c1_rev,
                    n1_rev,
                }
            })
            .collect();
        println!("c1_data precomputed: {} circuits", c1_data.len());

        // Build flat work list in parallel over c1, then process each item in parallel.
        // We avoid collecting into WorkItem structs and instead process directly
        // using a two-level par_iter: outer over c1, inner over (c2, mapping, case).
        // This gives maximum parallelism while keeping allocations minimal.
        let db2_ref = &*db2_circuits;
        let db2_rev_ref = &*db2_rev;
        let db2_n2_ref = &*db2_n2;
        let db2_rev_n2_ref = &*db2_rev_n2;
        let stop_flag_par = Arc::clone(&stop_flag);
        let tx_par = tx.clone();
        let total_gates_tried_par = Arc::clone(&total_gates_tried);
        let total_results_par = Arc::clone(&total_results_generated);
        let skipped_par = Arc::clone(&skipped_count);

        let total_c1 = c1_data.len();
        let c1_done = std::sync::atomic::AtomicUsize::new(0);
        let chunk_total = std::sync::atomic::AtomicUsize::new(0);
        println!(
            "[chunk {}] processing {} c1 circuits × {} c2 = {} pairs this chunk",
            chunk_idx,
            total_c1,
            nc2,
            total_c1 * nc2
        );

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
                        let (canon_polys, canon_circuit, _, _, _) =
                            canonicalize_bidirectional(&combined, true)
                                .expect("uncapped regular DB canonicalization skipped");
                        let key = xxh3_128(&polys_repr_blob(&canon_polys)).to_le_bytes().to_vec();
                        let value = encode_circuit(&circuit_blob(&canon_circuit));
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

                    if local.len() >= batch_size && !stop_flag_par.load(Ordering::SeqCst) {
                        let n = local.len();
                        let batch = std::mem::take(&mut local);
                        total_results_par.fetch_add(n, Ordering::Relaxed);
                        if let Err(e) = tx_par.send(batch) {
                            eprintln!("Failed to send batch: {:?}", e);
                            stop_flag_par.store(true, Ordering::SeqCst);
                            return;
                        }
                    }
                }

                let n_local = local.len();
                // Send any remaining results
                if !local.is_empty() && !stop_flag_par.load(Ordering::SeqCst) {
                    total_results_par.fetch_add(n_local, Ordering::Relaxed);
                    if let Err(e) = tx_par.send(local) {
                        eprintln!("Failed to send batch: {:?}", e);
                        stop_flag_par.store(true, Ordering::SeqCst);
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
            chunk_idx,
            total_c1_processed,
            chunk_results,
            total_results_generated.load(Ordering::Relaxed),
            elapsed / 3600.0,
        );

        if stop_flag.load(Ordering::SeqCst) {
            break;
        }
    }

    drop(tx);
    insert_handle
        .join()
        .map_err(|_| "regular DB writer thread panicked")?
        .map_err(|error| -> Box<dyn std::error::Error> { error.into() })?;
    if let Some(error) = source_error {
        return Err(error.into());
    }

    if stop_flag.load(Ordering::SeqCst) {
        return Err(
            "regular DB build interrupted; output is partial and must not be promoted".into(),
        );
    }

    println!("Compacting new_db...");
    new_db.compact_range::<&[u8], &[u8]>(None, None);
    println!("Compaction done.");
    println!("Build finished.");
    Ok(())
}

pub fn rocks_to_lmdb(rocks_path: &str, lmdb_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    use lmdb::{DatabaseFlags, Environment, EnvironmentFlags, Transaction, WriteFlags};

    if std::path::Path::new(lmdb_path).exists() {
        return Err(format!("refusing existing LMDB output: {lmdb_path}").into());
    }
    std::fs::create_dir_all(lmdb_path)?;

    // Bulk-load environment: NO_SYNC + WRITE_MAP + MAP_ASYNC with a single
    // explicit sync at the end (the output is rebuildable staging data until
    // validation passes). 6 TiB virtual map; max_dbs 600 leaves room for the
    // curated_XX shards added to the same environment later.
    let env = Environment::new()
        .set_flags(
            EnvironmentFlags::WRITE_MAP | EnvironmentFlags::MAP_ASYNC | EnvironmentFlags::NO_SYNC,
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
        let rocks =
            DB::open_for_read_only(&ropts, &rocks_path_owned, false).map_err(|e| e.to_string())?;
        let mut read_opts = rocksdb::ReadOptions::default();
        read_opts.set_readahead_size(16 * 1024 * 1024);
        let mut batch: Vec<(Box<[u8]>, Box<[u8]>)> = Vec::with_capacity(65_536);
        for item in rocks.iterator_opt(rocksdb::IteratorMode::Start, read_opts) {
            let (key, value) = item.map_err(|e| e.to_string())?;
            validate_rocks_entry(&key, &value)?;
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
            tx.send(batch)
                .map_err(|_| "lmdb writer hung up".to_string())?;
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
            txn.put(
                dbs[shard],
                &key.as_ref(),
                &value.as_ref(),
                WriteFlags::APPEND,
            )?;
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
        count,
        lmdb_path,
        el,
        el / 3600.0
    );
    Ok(())
}
