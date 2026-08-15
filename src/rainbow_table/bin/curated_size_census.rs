//! Size census over a curated-full composite store.
//!
//! Answers the two questions that decide how the uncapped curated database can
//! be materialized:
//!
//!   1. How many keys exceed LMDB's hard `MAXDATASIZE` (4 GiB - 1) per-value
//!      ceiling, i.e. how many keys make the legacy LMDB stage impossible.
//!   2. For each oversized key, how many other keys share its frozen bucket.
//!      Frozen concatenates a bucket's values with no per-value index, so a
//!      same-bucket neighbour sorting after a huge value must `skip_value`
//!      over the whole thing on every lookup.
//!
//! Read-only. Never writes to the store.
//!
//! ```text
//! curated_size_census COMPOSITE_ROCKS [--top N]
//! ```

use local_mixing::rainbow_table::curated_full::{
    COMPOSITE_COMPLETE_MARKER, COMPOSITE_FORMAT_MARKER, FUNCTION_KEY_BYTES, split_composite_key,
};
use rocksdb::{DB, IteratorMode, Options};
use std::collections::HashMap;
use std::error::Error;

type AnyResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

/// LMDB's non-DUPSORT per-value ceiling: `MAXDATASIZE` in mdb.c, enforced by
/// `mdb_cursor_put` (returns MDB_BAD_VALSIZE). A 32-bit node size field, so
/// this is structural rather than tunable.
const LMDB_MAX_VALUE_BYTES: u64 = 0xffff_ffff;

/// Frozen bucket index width, mirroring `replace::frozen_build`.
const BUCKET_BITS: u32 = 20;

fn split_frozen_key(key: &[u8; FUNCTION_KEY_BYTES]) -> (usize, u32, u64) {
    let hi = u64::from_be_bytes(key[0..8].try_into().unwrap());
    let lo = u64::from_be_bytes(key[8..16].try_into().unwrap());
    let shard = key[0] as usize;
    let bucket = ((hi >> 36) & ((1u64 << BUCKET_BITS) - 1)) as u32;
    let tail = ((hi & 0xF_FFFF_FFFF) << 12) | (lo >> 52);
    (shard, bucket, tail)
}

fn hex(key: &[u8; FUNCTION_KEY_BYTES]) -> String {
    key.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Histogram edges in bytes; the final open-ended class is everything above
/// the LMDB ceiling.
const EDGES: [(u64, &str); 8] = [
    (1 << 10, "<1 KiB"),
    (1 << 16, "<64 KiB"),
    (1 << 20, "<1 MiB"),
    (1 << 24, "<16 MiB"),
    (1 << 28, "<256 MiB"),
    (1 << 30, "<1 GiB"),
    (LMDB_MAX_VALUE_BYTES, "<4 GiB (LMDB limit)"),
    (u64::MAX, ">=4 GiB (LMDB IMPOSSIBLE)"),
];

/// Candidate-count classes. The first edge is the historical `shuffletests`
/// per-key friend cap, so the table shows directly how many keys the old build
/// would have truncated.
const CAND_EDGES: [(u64, &str); 10] = [
    (21, "<=20 (old shuffletests cap)"),
    (101, "21..100"),
    (1_001, "101..1k"),
    (10_001, "1k..10k"),
    (65_537, "10k..64k"),
    (1_048_577, "64k..1M"),
    (10_000_001, "1M..10M"),
    (100_000_001, "10M..100M"),
    (1_000_000_001, "100M..1G"),
    (u64::MAX, ">=1G"),
];

struct KeyRecord {
    key: [u8; FUNCTION_KEY_BYTES],
    value_bytes: u64,
    candidates: u64,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> AnyResult<()> {
    let args: Vec<String> = std::env::args().collect();
    let Some(path) = args.get(1) else {
        eprintln!("usage: curated_size_census COMPOSITE_ROCKS [--top N]");
        std::process::exit(2);
    };
    let top: usize = args
        .iter()
        .position(|a| a == "--top")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);

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

    let mut histogram = [0u64; EDGES.len()];
    let mut cand_histogram = [0u64; CAND_EDGES.len()];
    // Keys bucketed by the gate count of their SHORTEST candidate. A key whose
    // minimal circuit is a single gate describes a one-gate function, which is
    // not a useful replacement target.
    let mut min_gate_histogram = [0u64; 24];
    let mut short_keys: Vec<(KeyRecord, usize)> = Vec::new();
    // Totals restricted to keys whose minimal circuit has >= 2 gates.
    let mut kept_keys = 0u64;
    let mut kept_candidates = 0u64;
    let mut kept_value_bytes = 0u64;
    let mut kept_max_value_bytes = 0u64;
    let mut kept_max_candidates = 0u64;
    let mut total_value_bytes = 0u64;
    let mut keys = 0u64;
    let mut candidates = 0u64;
    // Every key that cannot go into LMDB, plus the largest keys overall.
    let mut oversized: Vec<KeyRecord> = Vec::new();
    let mut largest: Vec<KeyRecord> = Vec::new();
    // Frozen bucket occupancy, so we can tell whether an oversized value has
    // neighbours whose lookups would have to skip over it.
    let mut bucket_counts: HashMap<(usize, u32), u32> = HashMap::new();

    let mut current: Option<[u8; FUNCTION_KEY_BYTES]> = None;
    let mut current_bytes = 0u64;
    let mut current_candidates = 0u64;
    let mut current_min_gates = usize::MAX;

    #[allow(clippy::too_many_arguments)]
    let mut finish = |key: [u8; FUNCTION_KEY_BYTES],
                      value_bytes: u64,
                      candidate_count: u64,
                      min_gates: usize,
                      histogram: &mut [u64; EDGES.len()],
                      cand_histogram: &mut [u64; CAND_EDGES.len()],
                      min_gate_histogram: &mut [u64; 24],
                      short_keys: &mut Vec<(KeyRecord, usize)>,
                      kept_keys: &mut u64,
                      kept_candidates: &mut u64,
                      kept_value_bytes: &mut u64,
                      kept_max_value_bytes: &mut u64,
                      kept_max_candidates: &mut u64,
                      oversized: &mut Vec<KeyRecord>,
                      largest: &mut Vec<KeyRecord>| {
        min_gate_histogram[min_gates.min(23)] += 1;
        if min_gates < 2 {
            short_keys.push((
                KeyRecord {
                    key,
                    value_bytes,
                    candidates: candidate_count,
                },
                min_gates,
            ));
        } else {
            *kept_keys += 1;
            *kept_candidates += candidate_count;
            *kept_value_bytes += value_bytes;
            *kept_max_value_bytes = (*kept_max_value_bytes).max(value_bytes);
            *kept_max_candidates = (*kept_max_candidates).max(candidate_count);
        }
        let class = EDGES
            .iter()
            .position(|&(edge, _)| value_bytes < edge)
            .unwrap_or(EDGES.len() - 1);
        histogram[class] += 1;
        let cand_class = CAND_EDGES
            .iter()
            .position(|&(edge, _)| candidate_count < edge)
            .unwrap_or(CAND_EDGES.len() - 1);
        cand_histogram[cand_class] += 1;
        if value_bytes > LMDB_MAX_VALUE_BYTES {
            oversized.push(KeyRecord {
                key,
                value_bytes,
                candidates: candidate_count,
            });
        }
        largest.push(KeyRecord {
            key,
            value_bytes,
            candidates: candidate_count,
        });
        if largest.len() > 4096 {
            largest.sort_unstable_by(|a, b| b.value_bytes.cmp(&a.value_bytes));
            largest.truncate(top.max(64));
        }
    };

    for item in database.iterator(IteratorMode::Start) {
        let (record, _) = item?;
        if &*record == COMPOSITE_FORMAT_MARKER || &*record == COMPOSITE_COMPLETE_MARKER {
            continue;
        }
        let (key, blob) = split_composite_key(&record)?;
        if current != Some(key) {
            if let Some(previous) = current {
                finish(
                    previous,
                    current_bytes,
                    current_candidates,
                    current_min_gates,
                    &mut histogram,
                    &mut cand_histogram,
                    &mut min_gate_histogram,
                    &mut short_keys,
                    &mut kept_keys,
                    &mut kept_candidates,
                    &mut kept_value_bytes,
                    &mut kept_max_value_bytes,
                    &mut kept_max_candidates,
                    &mut oversized,
                    &mut largest,
                );
            }
            let (shard, bucket, _) = split_frozen_key(&key);
            *bucket_counts.entry((shard, bucket)).or_insert(0) += 1;
            keys += 1;
            current = Some(key);
            current_bytes = 0;
            current_candidates = 0;
            current_min_gates = usize::MAX;
            if keys % 2_000_000 == 0 {
                eprintln!("[census] keys={keys} candidates={candidates}");
            }
        }
        // Legacy record framing is one length byte plus the blob.
        current_bytes += blob.len() as u64 + 1;
        total_value_bytes += blob.len() as u64 + 1;
        current_candidates += 1;
        current_min_gates = current_min_gates.min(blob.len() / 3);
        candidates += 1;
    }
    if let Some(previous) = current {
        finish(
            previous,
            current_bytes,
            current_candidates,
            current_min_gates,
            &mut histogram,
            &mut cand_histogram,
            &mut min_gate_histogram,
            &mut short_keys,
            &mut kept_keys,
            &mut kept_candidates,
            &mut kept_value_bytes,
            &mut kept_max_value_bytes,
            &mut kept_max_candidates,
            &mut oversized,
            &mut largest,
        );
    }

    largest.sort_unstable_by(|a, b| b.value_bytes.cmp(&a.value_bytes));
    largest.truncate(top);
    oversized.sort_unstable_by(|a, b| b.value_bytes.cmp(&a.value_bytes));

    println!("keys={keys} candidates={candidates} total-value-bytes={total_value_bytes}");
    println!(
        "total-value-gib={:.2}  mean-bytes/key={:.1}",
        total_value_bytes as f64 / (1u64 << 30) as f64,
        total_value_bytes as f64 / keys.max(1) as f64
    );
    println!("\nper-key value size distribution:");
    for (index, (_, label)) in EDGES.iter().enumerate() {
        if histogram[index] > 0 {
            println!("  {label:<26} {:>12}", histogram[index]);
        }
    }

    println!("\nkeys by gate count of their SHORTEST candidate:");
    for (gates, count) in min_gate_histogram.iter().enumerate() {
        if *count > 0 {
            println!("  min-gates={gates:<3} {count:>12}");
        }
    }

    println!(
        "\n=== effect of dropping keys whose minimal circuit has < 2 gates ==="
    );
    println!("dropped-keys={}", short_keys.len());
    let dropped_candidates: u64 = short_keys.iter().map(|(r, _)| r.candidates).sum();
    let dropped_bytes: u64 = short_keys.iter().map(|(r, _)| r.value_bytes).sum();
    println!(
        "dropped-candidates={dropped_candidates} ({:.4}% of {candidates})",
        dropped_candidates as f64 * 100.0 / candidates.max(1) as f64
    );
    println!(
        "dropped-value-bytes={dropped_bytes} ({:.2} GiB, {:.4}% of total)",
        dropped_bytes as f64 / (1u64 << 30) as f64,
        dropped_bytes as f64 * 100.0 / total_value_bytes.max(1) as f64
    );
    for (record, min_gates) in &short_keys {
        let (shard, bucket, tail) = split_frozen_key(&record.key);
        println!(
            "  DROP key={} min-gates={min_gates} candidates={} bytes={} ({:.2} GiB) frozen[shard={:02x} bucket={:05x} tail={:012x}]",
            hex(&record.key),
            record.candidates,
            record.value_bytes,
            record.value_bytes as f64 / (1u64 << 30) as f64,
            shard,
            bucket,
            tail
        );
    }
    println!("--- surviving store (min-gates >= 2) ---");
    println!("keys={kept_keys} candidates={kept_candidates}");
    println!(
        "value-bytes={kept_value_bytes} ({:.2} GiB)",
        kept_value_bytes as f64 / (1u64 << 30) as f64
    );
    println!(
        "max-value-bytes={kept_max_value_bytes} ({:.3} GiB)  max-candidates/key={kept_max_candidates}",
        kept_max_value_bytes as f64 / (1u64 << 30) as f64
    );
    println!(
        "still above LMDB 4 GiB limit? {}",
        if kept_max_value_bytes > LMDB_MAX_VALUE_BYTES {
            "YES"
        } else {
            "no"
        }
    );

    println!("\nper-key candidate-count distribution:");
    for (index, (_, label)) in CAND_EDGES.iter().enumerate() {
        if cand_histogram[index] > 0 {
            println!("  {label:<28} {:>12}", cand_histogram[index]);
        }
    }

    // A cap decision needs "how many keys would this cap touch", i.e. the
    // number of keys at or above each edge, not the per-class counts.
    println!("\nkeys with MORE THAN N candidates (what a cap of N would truncate):");
    let mut at_or_above = 0u64;
    for index in (0..CAND_EDGES.len()).rev() {
        at_or_above += cand_histogram[index];
        if index == 0 {
            continue;
        }
        let cap = CAND_EDGES[index - 1].0 - 1;
        println!(
            "  cap={cap:<12} would truncate {at_or_above:>10} keys ({:.6}% of {keys})",
            at_or_above as f64 * 100.0 / keys.max(1) as f64
        );
    }

    println!(
        "\nkeys above LMDB MAXDATASIZE (4 GiB - 1): {}",
        oversized.len()
    );
    for record in &oversized {
        let (shard, bucket, tail) = split_frozen_key(&record.key);
        let neighbours = bucket_counts
            .get(&(shard, bucket))
            .copied()
            .unwrap_or(1)
            .saturating_sub(1);
        println!(
            "  key={} bytes={} ({:.2} GiB) candidates={} frozen[shard={:02x} bucket={:05x} tail={:012x}] bucket-neighbours={}",
            hex(&record.key),
            record.value_bytes,
            record.value_bytes as f64 / (1u64 << 30) as f64,
            record.candidates,
            shard,
            bucket,
            tail,
            neighbours
        );
    }

    println!("\nlargest {} keys by value bytes:", largest.len());
    for record in &largest {
        println!(
            "  {} bytes={:>14} candidates={:>12}",
            hex(&record.key),
            record.value_bytes,
            record.candidates
        );
    }

    let crowded = bucket_counts.values().filter(|&&n| n > 1).count();
    println!(
        "\nfrozen buckets occupied={} of {} ; buckets with >1 key={}",
        bucket_counts.len(),
        256usize << BUCKET_BITS,
        crowded
    );
    Ok(())
}
