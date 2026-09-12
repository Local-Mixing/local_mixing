//! Regular→curated key-coverage census, per minimal-gate class.
//!
//! For every key of the frozen REGULAR store, classify it by the gate count
//! of its shortest stored spelling (its m-class; the store enumerates every
//! function to 11 gates, so the shortest spelling is the true minimum), and
//! test whether the same function key exists in the curated store at all.
//! Membership is matched on the 76 key bits both frozen stores address by
//! (shard, bucket, tail), via `mix76` against an exact set built from the
//! curated composite's full 16-byte keys — the same mix the runtime's miss
//! filter uses, but as an exact hash set rather than a fuse filter.
//!
//! Answers "how much of m4 is missing from curated": total m4 keys, how many
//! appear in curated, how many do not.
//!
//! ```text
//! curated_coverage_census FROZEN_REGULAR_DIR CURATED_COMPOSITE_ROCKS [--shards N]
//! ```

use local_mixing::db_generation::curated_full::split_composite_key;
use local_mixing::db_mixing::frozen::{mix76, scan_shard_entries, split_key};
use rayon::prelude::*;
use rocksdb::{DB, IteratorMode, Options};
use rustc_hash::FxHashSet;
use std::error::Error;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

type AnyResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

const MAX_M: usize = 16;

fn min_member_gates(value: &[u8]) -> Option<usize> {
    let mut best: Option<usize> = None;
    let mut position = 0usize;
    while position < value.len() {
        let len = value[position] as usize;
        position += 1;
        if len == 0 || len % 3 != 0 || position + len > value.len() {
            break;
        }
        let gates = len / 3;
        if best.is_none_or(|b| gates < b) {
            best = Some(gates);
        }
        position += len;
    }
    best
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> AnyResult<()> {
    let args: Vec<String> = std::env::args().collect();
    let (Some(regular), Some(composite)) = (args.get(1), args.get(2)) else {
        eprintln!(
            "usage: curated_coverage_census FROZEN_REGULAR_DIR CURATED_COMPOSITE_ROCKS [--shards N]"
        );
        std::process::exit(2);
    };
    let shards: usize = args
        .iter()
        .position(|a| a == "--shards")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(256)
        .clamp(1, 256);

    // Exact curated key set as mix76 values (64-bit; with ~32M keys the
    // collision probability is ~1e-12 -- census-grade exact).
    let start = Instant::now();
    let mut options = Options::default();
    options.create_if_missing(false);
    options.set_compression_type(rocksdb::DBCompressionType::Zstd);
    options.set_bottommost_compression_type(rocksdb::DBCompressionType::Zstd);
    let db = DB::open_for_read_only(&options, composite.as_str(), false)?;
    let mut curated: FxHashSet<u64> = FxHashSet::default();
    let mut current: Option<[u8; 16]> = None;
    for item in db.iterator(IteratorMode::Start) {
        let (record, _) = item?;
        let Ok((key, _)) = split_composite_key(&record) else {
            continue;
        };
        if current == Some(key) {
            continue;
        }
        current = Some(key);
        let (shard, bucket, tail) = split_key(&key);
        curated.insert(mix76(shard, bucket, tail));
    }
    eprintln!(
        "[coverage] curated key set: {} keys in {:.0}s",
        curated.len(),
        start.elapsed().as_secs_f64()
    );
    let curated = &curated;

    let totals: Vec<AtomicU64> = (0..=MAX_M).map(|_| AtomicU64::new(0)).collect();
    let present: Vec<AtomicU64> = (0..=MAX_M).map(|_| AtomicU64::new(0)).collect();
    let done = AtomicU64::new(0);
    (0..shards).into_par_iter().for_each(|shard| {
        let mut local_totals = [0u64; MAX_M + 1];
        let mut local_present = [0u64; MAX_M + 1];
        scan_shard_entries(regular, shard, &mut |bucket, tail, value| {
            let Some(m) = min_member_gates(value) else {
                return;
            };
            let m = m.min(MAX_M);
            local_totals[m] += 1;
            if curated.contains(&mix76(shard, bucket, tail)) {
                local_present[m] += 1;
            }
        });
        for m in 0..=MAX_M {
            totals[m].fetch_add(local_totals[m], Ordering::Relaxed);
            present[m].fetch_add(local_present[m], Ordering::Relaxed);
        }
        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
        if n % 16 == 0 || n as usize == shards {
            eprintln!(
                "[coverage] {n}/{shards} shards, {:.0}s",
                start.elapsed().as_secs_f64()
            );
        }
    });

    println!(
        "shards={shards} (of 256)  elapsed={:.0}s",
        start.elapsed().as_secs_f64()
    );
    println!(
        "{:>4} {:>16} {:>16} {:>16} {:>10}",
        "m", "regular-keys", "in-curated", "missing", "coverage"
    );
    let scale = 256.0 / shards as f64;
    let mut grand_total = 0u64;
    let mut grand_present = 0u64;
    for m in 0..=MAX_M {
        let t = totals[m].load(Ordering::Relaxed);
        if t == 0 {
            continue;
        }
        let p = present[m].load(Ordering::Relaxed);
        grand_total += t;
        grand_present += p;
        println!(
            "{m:>4} {t:>16} {p:>16} {:>16} {:>9.4}%",
            t - p,
            p as f64 * 100.0 / t as f64
        );
    }
    println!(
        "{:>4} {grand_total:>16} {grand_present:>16} {:>16} {:>9.4}%",
        "all",
        grand_total - grand_present,
        grand_present as f64 * 100.0 / grand_total.max(1) as f64
    );
    if shards < 256 {
        println!("(x{scale:.0} projection applies to absolute counts; percentages are unbiased)");
    }
    Ok(())
}
