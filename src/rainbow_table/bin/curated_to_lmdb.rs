use lmdb::{DatabaseFlags, Environment, EnvironmentFlags, Transaction, WriteFlags};
use rocksdb::{DB, MergeOperands, Options};
use std::path::Path;

fn append_merge(
    _key: &[u8],
    existing: Option<&[u8]>,
    operands: &MergeOperands,
) -> Option<Vec<u8>> {
    let mut result = existing.map_or_else(Vec::new, |v| v.to_vec());
    for op in operands {
        result.extend_from_slice(op);
    }
    Some(result)
}

fn main() {
    let env = Environment::new()
        .set_flags(EnvironmentFlags::WRITE_MAP | EnvironmentFlags::MAP_ASYNC | EnvironmentFlags::NO_SYNC)
        .set_max_dbs(600)
        .set_max_readers(10000)
        .set_map_size(6 * 1024 * 1024 * 1024 * 1024)
        .open(Path::new("./db"))
        .expect("Failed to open ./db");

    println!("Creating curated_{{}} shard databases...");
    let dbs: Vec<lmdb::Database> = (0u16..=255)
        .map(|s| {
            let name = format!("curated_{:02x}", s);
            env.create_db(Some(name.as_str()), DatabaseFlags::empty())
                .unwrap_or_else(|e| panic!("Failed to create {}: {:?}", name, e))
        })
        .collect();

    // Wipe any existing curated shard data first: a put only overwrites matching keys,
    // so stale entries from a previous (e.g. buggy) build would otherwise linger.
    {
        let mut txn = env.begin_rw_txn().expect("rw txn");
        for &db in &dbs {
            txn.clear_db(db).expect("clear curated shard");
        }
        txn.commit().expect("commit clear");
    }
    println!("Cleared existing curated_{{}} shards.");

    let mut opts = Options::default();
    opts.set_merge_operator_associative("append_merge", append_merge);
    let rocks = DB::open_for_read_only(&opts, "rocks_curated_db", false)
        .expect("Failed to open rocks_curated_db");

    // Display-only estimate; exact counts are cross-checked by
    // validate_lmdb_build afterwards. (The old full counting pre-pass
    // decompressed the entire DB once for nothing.)
    let total: u64 = rocks
        .property_int_value("rocksdb.estimate-num-keys")
        .ok()
        .flatten()
        .unwrap_or(0);
    println!("RocksDB estimated entries: {}", total);

    let mut count = 0u64;
    let mut txn = env.begin_rw_txn().expect("rw txn");

    for item in rocks.iterator(rocksdb::IteratorMode::Start) {
        let (key, value) = item.expect("iterator error");
        let shard = key[0] as usize;
        txn.put(dbs[shard], &key, &value, WriteFlags::APPEND)
            .expect("lmdb put");
        count += 1;
        if count % 1_000_000 == 0 {
            if let Err(e) = txn.commit() {
                eprintln!("COMMIT FAILED at count={}: {:?}", count, e);
                std::process::exit(1);
            }
            txn = env.begin_rw_txn().expect("rw txn");
            if count % 500_000 == 0 {
                println!("  {}/{} entries...", count, total);
            }
        }
    }

    if let Err(e) = txn.commit() {
        eprintln!("FINAL COMMIT FAILED at count={}: {:?}", count, e);
        std::process::exit(1);
    }
    // The environment runs NO_SYNC for speed; make the result durable now.
    if let Err(e) = env.sync(true) {
        eprintln!("FINAL ENV SYNC FAILED: {:?}", e);
        std::process::exit(1);
    }
    println!("Done. {}/{} entries written to ./db curated_{{}} shards.", count, total);
}
