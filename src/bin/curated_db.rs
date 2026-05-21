use lmdb::{Cursor, Environment, Transaction};
use local_mixing::circuit::circuit::{canonicalize_polys_4, polys_repr_blob, CircuitSeq};
use num_cpus;
use rayon::prelude::*;
use rocksdb::{DB, MergeOperands, Options};
use std::path::Path;
use std::sync::Arc;
use xxhash_rust::xxh3::xxh3_128;

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

fn encode_circuit(blob: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(1 + blob.len());
    v.push(blob.len() as u8);
    v.extend_from_slice(blob);
    v
}

fn main() {
    let env = Environment::new()
        .set_max_dbs(300)
        .set_map_size(800 * 1024 * 1024 * 1024)
        .open(Path::new("./db"))
        .expect("Failed to open ./db");

    println!("Loading id_g circuits...");
    let mut all_blobs: Vec<Vec<u8>> = Vec::new();
    for i in 0..34usize {
        let name = format!("id_g{}", i);
        let db = match env.open_db(Some(name.as_str())) {
            Ok(db) => db,
            Err(_) => continue,
        };
        let txn = env.begin_ro_txn().expect("ro txn");
        let mut cursor = txn.open_ro_cursor(db).expect("cursor");
        for (_, v) in cursor.iter() {
            all_blobs.push(v.to_vec());
        }
        drop(cursor);
        drop(txn);
    }
    println!("Loaded {} circuits", all_blobs.len());

    let mut opts = Options::default();
    opts.create_if_missing(true);
    opts.set_merge_operator_associative("append_merge", append_merge);
    opts.increase_parallelism(num_cpus::get() as i32);
    let rdb = Arc::new(DB::open(&opts, "rocks_curated_db").expect("open rocks_curated_db"));

    let total = all_blobs.len();
    println!("Processing {} circuits...", total);

    all_blobs.par_iter().enumerate().for_each(|(idx, blob)| {
        let circuit = CircuitSeq::from_blob(blob);
        let n = circuit.gates.len();
        if n < 2 {
            return;
        }
        let wire_count = circuit.max_wire() + 1;

        let forward: Vec<[u8; 3]> = circuit.gates.clone();
        let reversed: Vec<[u8; 3]> = circuit.gates.iter().rev().cloned().collect();

        for direction in [&forward, &reversed] {
            for rot in 0..n {
                let rotation: Vec<[u8; 3]> = direction[rot..]
                    .iter()
                    .chain(direction[..rot].iter())
                    .cloned()
                    .collect();

                for k in 1..n {
                    // Canonicalize the removed prefix (first k gates)
                    let prefix_circuit = CircuitSeq { gates: rotation[..k].to_vec() };
                    let prefix_polys = prefix_circuit.to_polynomial(wire_count, 0, k);
                    if prefix_polys.is_empty() {
                        continue;
                    }
                    let (canonical, perm) = canonicalize_polys_4(prefix_polys);
                    if canonical.is_empty() {
                        continue;
                    }
                    let key = xxh3_128(&polys_repr_blob(&canonical)).to_le_bytes();

                    // Apply inv(perm) to the reversed tail → replacement in canonical wire coords
                    let perm_inv = perm.invert();
                    let tail_gates: Vec<[u8; 3]> = rotation[k..].iter().rev()
                        .map(|&[t, c1, c2]| [
                            perm_inv.data[t as usize] as u8,
                            perm_inv.data[c1 as usize] as u8,
                            perm_inv.data[c2 as usize] as u8,
                        ])
                        .collect();
                    let tail_blob = CircuitSeq { gates: tail_gates }.repr_blob();
                    rdb.merge(&key, &encode_circuit(&tail_blob)).expect("rocksdb merge");
                }
            }
        }

        if idx % 50_000 == 0 {
            println!("  {}/{}", idx, total);
        }
    });

    println!("Done. Compacting...");
    rdb.compact_range::<&[u8], &[u8]>(None, None);
    println!("Complete.");
}
