use lmdb::{Cursor, Environment, Transaction};
use local_mixing::circuit::circuit::{canonicalize_polys, polys_repr_blob, CircuitSeq};
use rayon::prelude::*;
use std::collections::HashSet;
use std::path::Path;
use std::sync::Mutex;
use xxhash_rust::xxh3::xxh3_128;

fn decode_circuits(value: &[u8]) -> Vec<CircuitSeq> {
    let mut circuits = Vec::new();
    let mut pos = 0;
    while pos < value.len() {
        if pos + 1 > value.len() { break; }
        let len = value[pos] as usize;
        pos += 1;
        if pos + len > value.len() { break; }
        circuits.push(CircuitSeq::from_blob(&value[pos..pos + len]));
        pos += len;
    }
    circuits
}

// Apply the exact same pipeline as generate_identities_parallel.
// Returns None if the circuit is trivial or non-minimal.
fn pipeline(
    gates: Vec<[u8; 3]>,
    rtxn: &lmdb::RoTransaction,
    shard_dbs: &[lmdb::Database],
) -> Option<Vec<u8>> {
    let (mut identity, _) = CircuitSeq { gates }.rewire_min();

    loop {
        let len_before = identity.gates.len();
        identity.canonicalize();
        identity.remove_adjacent_id();
        if identity.gates.is_empty() { break; }
        if identity.gates.len() == len_before { break; }
    }
    if identity.gates.is_empty() { return None; }

    identity.canonicalize();
    let (identity, _) = identity.rewire_min();

    let len = identity.gates.len();
    let half_len = len / 2;
    if half_len == 0 { return None; }

    let wire_count = identity.max_wire() + 1;
    for start in 0..=(len - half_len) {
        let end = start + half_len;
        let polys = identity.to_polynomial(wire_count, start, end);
        let (canonical, _) = canonicalize_polys(polys, true, false);
        let key = xxh3_128(&polys_repr_blob(&canonical)).to_le_bytes();
        let shard = key[0] as usize;
        if rtxn.get(shard_dbs[shard], &key).is_ok() {
            return None;
        }
    }

    Some(identity.repr_blob())
}

fn main() {
    let env = Environment::new()
        .set_max_dbs(300)
        .set_map_size(800 * 1024 * 1024 * 1024)
        .open(Path::new("./db"))
        .expect("Failed to open ./db");

    // Step 1: load S1 from id_g{} (a||rev(b) blobs from --gen-ids).
    println!("Loading id_g{{}} into S1...");
    let mut s1: HashSet<Vec<u8>> = HashSet::new();
    for i in 0..34usize {
        let name = format!("id_g{}", i);
        let db = match env.open_db(Some(name.as_str())) {
            Ok(db) => db,
            Err(_) => continue,
        };
        let txn = env.begin_ro_txn().expect("ro txn");
        let mut cursor = txn.open_ro_cursor(db).expect("cursor");
        for (_, v) in cursor.iter() {
            s1.insert(v.to_vec());
        }
        drop(cursor);
        drop(txn);
    }
    println!("S1 size: {}", s1.len());

    let shard_dbs: Vec<lmdb::Database> = (0u8..=255)
        .map(|s| {
            let name = format!("{:02x}", s);
            env.open_db(Some(name.as_str()))
                .unwrap_or_else(|e| panic!("shard {:02x}: {:?}", s, e))
        })
        .collect();

    // Step 2: scan shards using same pair selection as generate_identities_parallel,
    // compute rev(a)||b, check overlap with S1.
    println!("Scanning shards for rev(a)||b...");
    let s1_ref = &s1;
    let total_s2 = Mutex::new(HashSet::<Vec<u8>>::new());
    let overlap = Mutex::new(HashSet::<Vec<u8>>::new());

    (0..256usize).into_par_iter().for_each(|shard_idx| {
        let db = shard_dbs[shard_idx];
        let rtxn = env.begin_ro_txn().expect("ro txn");
        let mut cursor = rtxn.open_ro_cursor(db).expect("cursor");

        let mut local_s2: HashSet<Vec<u8>> = HashSet::new();
        let mut local_overlap: HashSet<Vec<u8>> = HashSet::new();

        for (_, value) in cursor.iter_start() {
            let circuits = decode_circuits(value);
            if circuits.len() < 2 { continue; }

            let minimal_len = circuits.iter().map(|c| c.gates.len()).min().unwrap();

            for i in 0..circuits.len() {
                let a = &circuits[i];
                if a.gates.len() > minimal_len { continue; }

                for j in (i + 1)..circuits.len() {
                    let b = &circuits[j];
                    if b.gates.len() > minimal_len + 1 { continue; }

                    // rev(a) || b
                    let mut gates: Vec<[u8; 3]> = a.gates.iter().rev().cloned().collect();
                    gates.extend_from_slice(&b.gates);

                    if let Some(blob) = pipeline(gates, &rtxn, &shard_dbs) {
                        if s1_ref.contains(&blob) {
                            local_overlap.insert(blob.clone());
                        }
                        local_s2.insert(blob);
                    }
                }
            }
        }
        drop(cursor);
        drop(rtxn);

        if !local_s2.is_empty() {
            total_s2.lock().unwrap().extend(local_s2);
            overlap.lock().unwrap().extend(local_overlap);
        }
        println!("  shard {:03} done", shard_idx);
    });

    let s2_size = total_s2.lock().unwrap().len();
    let overlap_size = overlap.lock().unwrap().len();
    println!("\nS1 (a||rev(b)):  {}", s1.len());
    println!("S2 (rev(a)||b):  {}", s2_size);
    println!("S2 ∩ S1:         {}", overlap_size);
    println!("S2 not in S1:    {}", s2_size.saturating_sub(overlap_size));
}
