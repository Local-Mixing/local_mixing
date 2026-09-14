//! End-to-end proof that the LMDB -> frozen-store conversion preserves every
//! lookup: build a synthetic sharded LMDB shaped exactly like the production
//! replacement store (256 named dbs "XX", 16-byte xxh3 keys sharded by byte 0,
//! length-prefixed 3-byte-gate blob values), convert it with frozen_build,
//! then (a) the builder's own byte-exact validate stage must PASS and (b) a
//! FrozenDb must return byte-identical values for every stored key and None
//! for random absent keys.

use lmdb::Transaction;

// Compile the dependency-light fixture builder only into this integration test.
#[path = "../db_gen/frozen_build.rs"]
#[allow(dead_code)] // This fixture uses the writer/validator subset of the offline builder.
mod frozen_build;
use frozen_build::{LmdbShards, stage_tables, stage_validate, stage_write};
use local_mixing::database::frozen::FrozenDb;
#[cfg(feature = "db-tools")]
use local_mixing::db_generation;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn synth_value(rng: &mut StdRng) -> Vec<u8> {
    // 1-3 friends per value; each friend = [len][len bytes of 3-byte gates]
    // with wires in the canonical <=8-wire space, like real stored friends.
    let mut value = Vec::new();
    for _ in 0..rng.random_range(1..=3usize) {
        let gates = rng.random_range(1..=6usize);
        value.push((gates * 3) as u8);
        for _ in 0..gates {
            let a = rng.random_range(0..8u8);
            let mut b = rng.random_range(0..8u8);
            while b == a {
                b = rng.random_range(0..8u8);
            }
            let mut c = rng.random_range(0..8u8);
            while c == a || c == b {
                c = rng.random_range(0..8u8);
            }
            value.extend([a, b, c]);
        }
    }
    value
}

#[cfg(feature = "db-tools")]
#[test]
fn rocksdb_bands_merge_keys_and_deduplicate_candidates_within_the_shard() {
    use frozen_build::{MultiBandShards, ShardReader};

    let base = std::env::temp_dir().join(format!("frozen_multiband_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    let key = |shard, tail| {
        let mut key = [0u8; 16];
        key[0] = shard;
        key[15] = tail;
        key
    };
    let first = vec![3, 0, 1, 2];
    let second = vec![3, 2, 1, 0];
    let repeated: Vec<u8> = first.iter().chain(&first).copied().collect();
    let combined: Vec<u8> = first.iter().chain(&second).copied().collect();
    let bands = [
        vec![
            (key(0x20, 1), first.clone()),
            (key(0x21, 1), repeated),
            (key(0x21, 3), second.clone()),
        ],
        vec![
            (key(0x21, 1), combined.clone()),
            (key(0x21, 2), first.clone()),
            (key(0x22, 1), second.clone()),
        ],
        Vec::new(),
    ];
    let mut paths = Vec::new();
    for (index, records) in bands.iter().enumerate() {
        let path = base.join(index.to_string());
        let db = rocksdb::DB::open_default(&path).unwrap();
        for (key, value) in records {
            db.put(key, value).unwrap();
        }
        db.flush().unwrap();
        drop(db);
        paths.push(path.to_str().unwrap().to_owned());
    }
    let source = MultiBandShards::open(&paths);
    let collect = |shard, limit| {
        let mut entries = Vec::new();
        source.for_each_entry(shard, limit, &mut |key, value| {
            entries.push((key.to_vec(), value.to_vec()));
        });
        entries
    };
    let expected = vec![
        (key(0x21, 1).to_vec(), combined),
        (key(0x21, 2).to_vec(), first),
        (key(0x21, 3).to_vec(), second),
    ];
    assert_eq!(collect(0x21, None), expected);
    assert_eq!(collect(0x21, Some(1)), expected[..1]);
    assert!(collect(0x23, None).is_empty());
    drop(source);
    std::fs::remove_dir_all(&base).unwrap();
}

#[test]
fn lmdb_to_frozen_roundtrip_preserves_all_lookups() {
    let base = std::env::temp_dir().join(format!("frozen_rt_{}", std::process::id()));
    let lmdb_dir = base.join("db");
    let out_dir = base.join("frz");
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&lmdb_dir).unwrap();
    std::fs::create_dir_all(&out_dir).unwrap();

    // --- build the synthetic sharded LMDB ---
    let mut rng = StdRng::seed_from_u64(20260717);
    let env = lmdb::Environment::new()
        .set_max_dbs(300)
        .set_map_size(1 << 30)
        .open(&lmdb_dir)
        .unwrap();
    let dbs: Vec<lmdb::Database> = (0u16..=255)
        .map(|s| {
            env.create_db(
                Some(format!("{s:02x}").as_str()),
                lmdb::DatabaseFlags::empty(),
            )
            .unwrap()
        })
        .collect();
    let mut entries: Vec<([u8; 16], Vec<u8>)> = Vec::new();
    for _ in 0..5000 {
        let mut key = [0u8; 16];
        rng.fill(&mut key);
        entries.push((key, synth_value(&mut rng)));
    }
    {
        let mut txn = env.begin_rw_txn().unwrap();
        for (key, value) in &entries {
            txn.put(dbs[key[0] as usize], key, value, lmdb::WriteFlags::empty())
                .unwrap();
        }
        txn.commit().unwrap();
    }
    drop(env);

    // --- convert and byte-exact validate against the source ---
    let ld = lmdb_dir.to_str().unwrap();
    let od = out_dir.to_str().unwrap();
    let source = LmdbShards::open(ld, "");
    stage_tables(&source, od, 400_000);
    stage_write(&source, od);
    stage_validate(&source, od); // exits nonzero on any mismatch

    // --- every stored key returns the identical bytes; absent keys miss ---
    let db = FrozenDb::open(od, None);
    for (key, value) in &entries {
        assert_eq!(
            db.get_regular(key).as_deref(),
            Some(value.as_slice()),
            "frozen value differs for key {key:02x?}"
        );
    }
    let stored: std::collections::HashSet<[u8; 16]> = entries.iter().map(|(k, _)| *k).collect();
    for _ in 0..2000 {
        let mut key = [0u8; 16];
        rng.fill(&mut key);
        if !stored.contains(&key) {
            assert!(db.get_regular(&key).is_none(), "phantom hit for absent key");
        }
    }

    // --- scan_shard's sequential walk sees exactly the stored values ---
    let mut walked: Vec<Vec<u8>> = Vec::new();
    for s in 0..256 {
        local_mixing::database::frozen::scan_shard(od, s, &mut |v: &[u8]| walked.push(v.to_vec()));
    }
    let mut want: Vec<Vec<u8>> = entries.iter().map(|(_, v)| v.clone()).collect();
    walked.sort();
    want.sort();
    assert_eq!(walked, want, "scan_shard walk differs from stored values");

    let _ = std::fs::remove_dir_all(&base);
}
