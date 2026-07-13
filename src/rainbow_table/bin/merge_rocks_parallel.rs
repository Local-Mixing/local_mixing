// Parallel k-way merge of sorted RocksDBs, range-partitioned by the first key
// byte. Keys are uniform 16-byte xxh3 hashes and every occurrence of a key
// (across all sources) shares the same first byte, so each of the 256 byte
// partitions is an independent, disjoint merge — the per-key value dedup logic
// (dedup_concat) is byte-identical to the serial merge_rocks, just run 256-way.
//
// Env:
//   MERGE_PARTITION_THREADS  rayon threads (default = num_cpus)
//   MERGE_KEYS_PER_SST       keys buffered per SST flush (default 1_000_000)
//   MERGE_TEST_PREFIX        2 hex bytes (e.g. "00a1"): restrict the whole run
//                            to keys with that 2-byte prefix, for A/B validation.

use rocksdb::{
    BlockBasedOptions, DBCompressionType, Direction, IngestExternalFileOptions, IteratorMode,
    MergeOperands, Options, ReadOptions, SstFileWriter, DB,
};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::env;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use rayon::prelude::*;

fn append_merge(_key: &[u8], existing: Option<&[u8]>, operands: &MergeOperands) -> Option<Vec<u8>> {
    let mut result: Vec<u8> = existing.unwrap_or(&[]).to_vec();
    for operand in operands {
        let mut pos = 0usize;
        while pos + 1 <= operand.len() {
            let len = operand[pos] as usize;
            pos += 1;
            if pos + len > operand.len() {
                break;
            }
            let new_blob = &operand[pos..pos + len];
            pos += len;
            let mut rpos = 0usize;
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
                result.push(len as u8);
                result.extend_from_slice(new_blob);
            }
        }
    }
    Some(result)
}

fn read_opts() -> Options {
    let mut opts = Options::default();
    opts.create_if_missing(false);
    opts.set_merge_operator_associative("append_merge", append_merge);
    opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(16));
    opts.set_disable_auto_compactions(true);
    opts.set_max_open_files(-1);
    opts
}

fn write_opts() -> Options {
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
    opts
}

fn sst_opts() -> Options {
    let mut opts = Options::default();
    opts.set_merge_operator_associative("append_merge", append_merge);
    opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(16));
    opts.set_compression_type(DBCompressionType::Zstd);
    opts
}

fn dedup_concat(values: Vec<Vec<u8>>) -> Vec<u8> {
    let mut out = Vec::new();
    let mut seen: HashSet<Vec<u8>> = HashSet::new();
    for value in values {
        let mut pos = 0usize;
        while pos + 1 <= value.len() {
            let len = value[pos] as usize;
            pos += 1;
            if pos + len > value.len() {
                break;
            }
            let blob = &value[pos..pos + len];
            pos += len;
            if seen.insert(blob.to_vec()) {
                out.push(len as u8);
                out.extend_from_slice(blob);
            }
        }
    }
    out
}

fn flush_sst(
    output: &DB,
    pending: &mut Vec<(Vec<u8>, Vec<u8>)>,
    sst_counter: &AtomicUsize,
) -> Result<(), Box<dyn std::error::Error>> {
    if pending.is_empty() {
        return Ok(());
    }
    let idx = sst_counter.fetch_add(1, Ordering::Relaxed);
    let sst_path = format!("/dev/shm/merge_par_{}_{}.sst", std::process::id(), idx);
    let opts = sst_opts();
    let mut writer = SstFileWriter::create(&opts);
    writer.open(&sst_path)?;
    for (key, value) in pending.iter() {
        writer.put(key, value)?;
    }
    writer.finish()?;
    let mut ingest_opts = IngestExternalFileOptions::default();
    ingest_opts.set_move_files(false);
    output.ingest_external_file_opts(&ingest_opts, vec![sst_path.clone()])?;
    let _ = std::fs::remove_file(&sst_path);
    pending.clear();
    Ok(())
}

// Merge all sources over the key range [lower, upper) into `output`.
fn merge_range(
    sources: &[DB],
    output: &DB,
    lower: Vec<u8>,
    upper: Option<Vec<u8>>,
    keys_per_sst: usize,
    sst_counter: &AtomicUsize,
) -> Result<usize, Box<dyn std::error::Error>> {
    let mut iters: Vec<rocksdb::DBIterator> = sources
        .iter()
        .map(|db| {
            let mut ro = ReadOptions::default();
            ro.set_total_order_seek(true);
            if let Some(u) = &upper {
                ro.set_iterate_upper_bound(u.clone());
            }
            db.iterator_opt(IteratorMode::From(&lower, Direction::Forward), ro)
        })
        .collect();

    let mut heap: BinaryHeap<Reverse<(Vec<u8>, Vec<u8>, usize)>> = BinaryHeap::new();
    for (idx, iter) in iters.iter_mut().enumerate() {
        if let Some(Ok((key, value))) = iter.next() {
            heap.push(Reverse((key.to_vec(), value.to_vec(), idx)));
        }
    }

    let mut pending: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(keys_per_sst);
    let mut unique_keys = 0usize;

    while let Some(Reverse((key, value, idx))) = heap.pop() {
        if let Some(Ok((next_key, next_value))) = iters[idx].next() {
            heap.push(Reverse((next_key.to_vec(), next_value.to_vec(), idx)));
        }
        let mut values = vec![value];
        while matches!(heap.peek(), Some(Reverse((next_key, _, _))) if *next_key == key) {
            let Reverse((_, same_value, same_idx)) = heap.pop().unwrap();
            values.push(same_value);
            if let Some(Ok((next_key, next_value))) = iters[same_idx].next() {
                heap.push(Reverse((next_key.to_vec(), next_value.to_vec(), same_idx)));
            }
        }
        pending.push((key, dedup_concat(values)));
        unique_keys += 1;
        if pending.len() >= keys_per_sst {
            flush_sst(output, &mut pending, sst_counter)?;
        }
    }
    flush_sst(output, &mut pending, sst_counter)?;
    Ok(unique_keys)
}

fn parse_hex2(s: &str) -> Option<[u8; 2]> {
    if s.len() != 4 {
        return None;
    }
    let b0 = u8::from_str_radix(&s[0..2], 16).ok()?;
    let b1 = u8::from_str_radix(&s[2..4], 16).ok()?;
    Some([b0, b1])
}

fn key16(bytes: &[u8]) -> Vec<u8> {
    let mut k = vec![0u8; 16];
    for (i, &b) in bytes.iter().enumerate().take(16) {
        k[i] = b;
    }
    k
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: merge_rocks_parallel <output> <source1> <source2> [source3 ...]");
        std::process::exit(2);
    }
    let output_path = &args[1];
    let source_paths = &args[2..];

    let threads: usize = env::var("MERGE_PARTITION_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(num_cpus::get);
    let keys_per_sst: usize = env::var("MERGE_KEYS_PER_SST")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1_000_000);
    let test_prefix = env::var("MERGE_TEST_PREFIX").ok().and_then(|s| parse_hex2(&s));

    println!(
        "merge_rocks_parallel output={output_path} sources={source_paths:?} threads={threads} keys_per_sst={keys_per_sst} test_prefix={test_prefix:?}"
    );

    let ro = read_opts();
    let mut dbs = Vec::new();
    for path in source_paths {
        let db = DB::open_for_read_only(&ro, path, false)?;
        let est = db
            .property_int_value("rocksdb.estimate-num-keys")
            .ok()
            .flatten()
            .unwrap_or(0);
        println!("opened {path} estimate_keys={est}");
        dbs.push(db);
    }
    if std::path::Path::new(output_path).exists() {
        return Err(format!("output already exists: {output_path}").into());
    }
    let output = Arc::new(DB::open(&write_opts(), output_path)?);
    let sources = Arc::new(dbs);
    let sst_counter = Arc::new(AtomicUsize::new(0));
    let done = Arc::new(AtomicUsize::new(0));
    let total_keys = Arc::new(AtomicUsize::new(0));
    let start = std::time::Instant::now();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();

    // Partition list: full byte-0 sweep, or a single tight range for validation.
    let partitions: Vec<(Vec<u8>, Option<Vec<u8>>)> = if let Some([b0, b1]) = test_prefix {
        let lower = key16(&[b0, b1]);
        let upper = if b1 == 0xff {
            if b0 == 0xff {
                None
            } else {
                Some(key16(&[b0 + 1]))
            }
        } else {
            Some(key16(&[b0, b1 + 1]))
        };
        vec![(lower, upper)]
    } else {
        (0u16..256)
            .map(|p| {
                let b0 = p as u8;
                let lower = key16(&[b0]);
                let upper = if b0 == 0xff { None } else { Some(key16(&[b0 + 1])) };
                (lower, upper)
            })
            .collect()
    };
    let nparts = partitions.len();

    let failed = AtomicUsize::new(0);
    pool.install(|| {
        partitions
            .into_par_iter()
            .for_each(|(lower, upper)| {
                match merge_range(&sources, &output, lower, upper, keys_per_sst, &sst_counter) {
                    Ok(k) => {
                        total_keys.fetch_add(k, Ordering::Relaxed);
                    }
                    Err(e) => {
                        eprintln!("partition merge failed: {e}");
                        failed.fetch_add(1, Ordering::Relaxed);
                    }
                }
                let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                let elapsed = start.elapsed().as_secs_f64();
                let keys = total_keys.load(Ordering::Relaxed);
                let rate = keys as f64 / elapsed.max(1e-9);
                println!(
                    "[merge_par] partitions {}/{} keys={} rate={:.0}/s elapsed={:.0}s ({:.2}h) ssts={}",
                    d, nparts, keys, rate, elapsed, elapsed / 3600.0,
                    sst_counter.load(Ordering::Relaxed)
                );
            });
    });

    let nfailed = failed.load(Ordering::Relaxed);
    if nfailed > 0 {
        return Err(format!("{nfailed} partition merge(s) failed").into());
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "merge_rocks_parallel done output={} keys={} ssts={} elapsed={:.0}s ({:.2}h)",
        output_path,
        total_keys.load(Ordering::Relaxed),
        sst_counter.load(Ordering::Relaxed),
        elapsed,
        elapsed / 3600.0
    );
    Ok(())
}
