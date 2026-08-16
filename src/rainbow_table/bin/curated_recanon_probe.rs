//! Measure how much a *second* polynomial canonicalization would dedup the
//! uncapped curated store.
//!
//! Every stored candidate is already validated to re-canonicalize to the key it
//! sits under ([`validate_and_emit`]), so grouping candidates by their canonical
//! *key* is a no-op. What is not a no-op is grouping by canonical *circuit
//! form*: `derive_identity_candidates` rewires the prefix into the prefix's own
//! canonical wire frame, but the reversed suffix is rewired through `map_wire`,
//! which assigns fresh indices to wires outside that frame. Tails are therefore
//! not stored in their own canonical frame, and two distinct stored tails can
//! share one canonical form.
//!
//! For each candidate this computes
//!   1. the polynomial canonicalization (`canonicalize_polys_single_hashed`),
//!   2. the rewiring into that circuit's own canonical wire frame,
//!   3. `CircuitSeq::canonicalize` gate ordering,
//! and counts distinct results per key. Nothing is written; this only reports
//! what such a filter would remove.
//!
//! ```text
//! curated_recanon_probe COMPOSITE_ROCKS [--stride N] [--max-per-key M] [--top K]
//! ```
//!
//! `--max-per-key` bounds the per-key hash set (0 = unlimited). A key with 512 M
//! candidates needs roughly 12 GiB unlimited, so cap it on small machines.

use local_mixing::circuit::CircuitSeq;
use local_mixing::rainbow_table::curated_full::{
    COMPOSITE_COMPLETE_MARKER, COMPOSITE_FORMAT_MARKER, FUNCTION_KEY_BYTES, split_composite_key,
};
use rayon::prelude::*;
use rocksdb::{DB, Direction, IteratorMode, Options};
use std::collections::HashSet;
use std::error::Error;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

type AnyResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

/// Rewire `circuit` into its own canonical wire frame and gate-canonicalize it.
/// `None` means canonicalization declined (oversized window, monomial cap, or
/// Rule-L budget) -- the same skip outcome `canonical_key` turns into an error.
fn canonical_form(circuit: &CircuitSeq) -> Option<Vec<u8>> {
    let (key, permutation, used) = circuit.canonicalize_polys_single_hashed(false);
    key?;
    let inverse = permutation.invert();
    let width = *used.last()? as usize + 1;
    let mut map = vec![u16::MAX; width];
    for (dense, &original) in used.iter().enumerate() {
        map[original as usize] = inverse.data[dense] as u16;
    }
    let mut canonical = CircuitSeq {
        gates: circuit
            .gates
            .iter()
            .map(|&[target, a, b]| {
                [
                    map[target as usize],
                    map[a as usize],
                    map[b as usize],
                ]
            })
            .collect(),
    };
    canonical.canonicalize();
    let mut bytes = Vec::with_capacity(canonical.gates.len() * 3);
    for gate in &canonical.gates {
        for &wire in gate {
            bytes.push(wire as u8);
        }
    }
    Some(bytes)
}

#[derive(Default, Clone)]
struct KeyStat {
    key: [u8; FUNCTION_KEY_BYTES],
    candidates: u64,
    distinct: u64,
    reframed: u64,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn flag<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn run() -> AnyResult<()> {
    let args: Vec<String> = std::env::args().collect();
    let Some(path) = args.get(1) else {
        eprintln!("usage: curated_recanon_probe COMPOSITE_ROCKS [--stride N] [--max-per-key M] [--top K]");
        std::process::exit(2);
    };
    let stride: u64 = flag(&args, "--stride", 1).max(1);
    let max_per_key: u64 = flag(&args, "--max-per-key", 0);
    let top: usize = flag(&args, "--top", 20);

    let mut options = Options::default();
    options.create_if_missing(false);
    options.set_compression_type(rocksdb::DBCompressionType::Zstd);
    options.set_bottommost_compression_type(rocksdb::DBCompressionType::Zstd);
    let database = DB::open_for_read_only(&options, path, false)?;
    if database.get(COMPOSITE_FORMAT_MARKER)?.as_deref() != Some(b"1") {
        return Err(format!("{path} is not a curated-full composite-v1 store").into());
    }
    if database.get(COMPOSITE_COMPLETE_MARKER)?.is_none() {
        return Err(format!("{path} has no completion manifest; refusing a partial store").into());
    }

    let total_candidates = AtomicU64::new(0);
    let total_distinct = AtomicU64::new(0);
    let total_reframed = AtomicU64::new(0);
    let total_keys = AtomicU64::new(0);
    let total_skipped = AtomicU64::new(0);
    let capped_keys = AtomicU64::new(0);
    let biggest: Mutex<Vec<KeyStat>> = Mutex::new(Vec::new());

    (0..256usize).into_par_iter().try_for_each(|shard| -> AnyResult<()> {
        let start = [shard as u8];
        let mut key_index = 0u64;
        let mut current: Option<[u8; FUNCTION_KEY_BYTES]> = None;
        let mut seen: HashSet<u128> = HashSet::new();
        let mut stat = KeyStat::default();
        let mut local_top: Vec<KeyStat> = Vec::new();

        let mut finish = |stat: &mut KeyStat, seen: &mut HashSet<u128>, local_top: &mut Vec<KeyStat>| {
            if stat.candidates == 0 {
                return;
            }
            stat.distinct = seen.len() as u64;
            total_keys.fetch_add(1, Ordering::Relaxed);
            total_candidates.fetch_add(stat.candidates, Ordering::Relaxed);
            total_distinct.fetch_add(stat.distinct, Ordering::Relaxed);
            total_reframed.fetch_add(stat.reframed, Ordering::Relaxed);
            local_top.push(stat.clone());
            if local_top.len() > 512 {
                local_top.sort_unstable_by(|a, b| b.candidates.cmp(&a.candidates));
                local_top.truncate(top.max(32));
            }
            seen.clear();
            *stat = KeyStat::default();
        };

        for item in database.iterator(IteratorMode::From(&start, Direction::Forward)) {
            let (record, _) = item?;
            if record.is_empty() || record[0] as usize != shard {
                break;
            }
            if &*record == COMPOSITE_FORMAT_MARKER || &*record == COMPOSITE_COMPLETE_MARKER {
                continue;
            }
            let (key, blob) = split_composite_key(&record)?;
            if current != Some(key) {
                finish(&mut stat, &mut seen, &mut local_top);
                current = Some(key);
                key_index += 1;
                stat.key = key;
            }
            // Key-level sampling: skip whole groups, never partial ones.
            if stride > 1 && key_index % stride != 0 {
                continue;
            }
            if max_per_key > 0 && stat.candidates >= max_per_key {
                if stat.candidates == max_per_key {
                    capped_keys.fetch_add(1, Ordering::Relaxed);
                    stat.candidates += 1; // count once, then stop growing the set
                }
                continue;
            }
            let circuit = CircuitSeq::from_blob(blob);
            match canonical_form(&circuit) {
                Some(form) => {
                    if form != blob {
                        stat.reframed += 1;
                    }
                    seen.insert(xxhash_rust::xxh3::xxh3_128(&form));
                }
                None => {
                    total_skipped.fetch_add(1, Ordering::Relaxed);
                }
            }
            stat.candidates += 1;
        }
        finish(&mut stat, &mut seen, &mut local_top);

        local_top.sort_unstable_by(|a, b| b.candidates.cmp(&a.candidates));
        local_top.truncate(top);
        biggest.lock().unwrap().extend(local_top);
        Ok(())
    })?;

    let candidates = total_candidates.load(Ordering::Relaxed);
    let distinct = total_distinct.load(Ordering::Relaxed);
    let reframed = total_reframed.load(Ordering::Relaxed);
    let keys = total_keys.load(Ordering::Relaxed);
    let skipped = total_skipped.load(Ordering::Relaxed);
    let capped = capped_keys.load(Ordering::Relaxed);

    println!("stride={stride} max-per-key={max_per_key}");
    println!("keys-sampled={keys}");
    println!("candidates-examined={candidates}");
    println!("distinct-canonical-forms={distinct}");
    println!(
        "removed-by-second-canon={} ({:.4}%)",
        candidates.saturating_sub(distinct),
        if candidates > 0 {
            (candidates.saturating_sub(distinct)) as f64 * 100.0 / candidates as f64
        } else {
            0.0
        }
    );
    println!(
        "candidates-not-already-in-canonical-frame={reframed} ({:.4}%)",
        if candidates > 0 {
            reframed as f64 * 100.0 / candidates as f64
        } else {
            0.0
        }
    );
    println!("canonicalization-skips={skipped}");
    if capped > 0 {
        println!("KEYS TRUNCATED BY --max-per-key={capped} (their collapse is a lower bound)");
    }

    let mut top_keys = biggest.into_inner().unwrap();
    top_keys.sort_unstable_by(|a, b| b.candidates.cmp(&a.candidates));
    top_keys.truncate(top);
    println!("\nlargest {} keys examined:", top_keys.len());
    for stat in &top_keys {
        let removed = stat.candidates.saturating_sub(stat.distinct);
        println!(
            "  {} candidates={:>12} distinct={:>12} removed={:>12} ({:.2}%)",
            stat.key.iter().map(|b| format!("{b:02x}")).collect::<String>(),
            stat.candidates,
            stat.distinct,
            removed,
            if stat.candidates > 0 {
                removed as f64 * 100.0 / stat.candidates as f64
            } else {
                0.0
            }
        );
    }
    Ok(())
}
