use lmdb::{Cursor, DatabaseFlags, Environment, EnvironmentFlags, Transaction, WriteFlags};
use local_mixing::circuit::circuit::{canonicalize_polys_4, polys_repr_blob, CircuitSeq, Polynomial};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Mutex;
use xxhash_rust::xxh3::xxh3_128;

const MAX_CIRCUITS_PER_ENTRY: usize = 20;

fn decode_circuits(value: &[u8]) -> Vec<CircuitSeq> {
    let mut circuits = Vec::new();
    let mut pos = 0;
    while pos < value.len() {
        let len = value[pos] as usize;
        pos += 1;
        if pos + len > value.len() {
            break;
        }
        circuits.push(CircuitSeq::from_blob(&value[pos..pos + len]));
        pos += len;
    }
    circuits
}

fn remove_adjacent_equal(gates: &mut Vec<[u16; 3]>) {
    let mut i = 0;
    while i + 1 < gates.len() {
        if gates[i] == gates[i + 1] {
            gates.drain(i..=i + 1);
            if i > 0 {
                i -= 1;
            }
        } else {
            i += 1;
        }
    }
}

fn encode_circuit(blob: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(1 + blob.len());
    v.push(blob.len() as u8);
    v.extend_from_slice(blob);
    v
}

fn init_polys(n: usize) -> Vec<Polynomial> {
    (0..n).map(|i| HashSet::from([1u64 << i])).collect()
}

fn apply_gate(polys: &mut Vec<Polynomial>, gate: [u16; 3]) {
    let [a, b, c] = gate;
    let (a, b, c) = (a as usize, b as usize, c as usize);
    // NOT c
    let not_c: Polynomial = {
        let mut p = polys[c].clone();
        if !p.remove(&0u64) { p.insert(0u64); }
        p
    };
    // b AND NOT(c)
    let term: Polynomial = polys[b].iter()
        .flat_map(|&m1| not_c.iter().map(move |&m2| m1 | m2))
        .fold(HashSet::new(), |mut acc, m| { acc.insert(m); acc });
    // XOR into a
    let mut new_a = polys[a].clone();
    for m in term {
        if !new_a.remove(&m) { new_a.insert(m); }
    }
    if !new_a.remove(&0u64) { new_a.insert(0u64); }
    polys[a] = new_a;
}

fn main() {
    let env = Environment::new()
        .set_flags(EnvironmentFlags::WRITE_MAP | EnvironmentFlags::MAP_ASYNC | EnvironmentFlags::NO_SYNC)
        .set_max_dbs(600)
        .set_max_readers(10000)
        .set_map_size(800 * 1024 * 1024 * 1024)
        .open(Path::new("./db"))
        .expect("Failed to open ./db");

    println!("Opening source shard databases (00..ff)...");
    let src_dbs: Vec<lmdb::Database> = (0u16..256)
        .map(|s| {
            let name = format!("{:02x}", s);
            env.open_db(Some(name.as_str()))
                .unwrap_or_else(|e| panic!("Failed to open shard {}: {:?}", name, e))
        })
        .collect();

    println!("Creating curated_{{}} shard databases...");
    let curated_dbs: Vec<lmdb::Database> = (0u16..256)
        .map(|s| {
            let name = format!("curated_{:02x}", s);
            env.create_db(Some(name.as_str()), DatabaseFlags::empty())
                .unwrap_or_else(|e| panic!("Failed to create {}: {:?}", name, e))
        })
        .collect();

    let accumulators: Vec<Mutex<HashMap<[u8; 16], Vec<u8>>>> =
        (0..256).map(|_| Mutex::new(HashMap::new())).collect();

    let done = std::sync::atomic::AtomicUsize::new(0);

    (0..256usize).into_par_iter().for_each(|shard_idx| {
        let entries: Vec<Vec<u8>> = {
            let txn = env.begin_ro_txn().expect("ro txn");
            let mut cursor = txn.open_ro_cursor(src_dbs[shard_idx]).expect("cursor");
            let result: Vec<Vec<u8>> = cursor
                .iter()
                .filter_map(|(_, v)| {
                    let circuits = decode_circuits(v);
                    if circuits.len() >= 2 { Some(v.to_vec()) } else { None }
                })
                .collect();
            drop(cursor);
            drop(txn);
            result
        };

        for value in &entries {
            let circuits = decode_circuits(value);
            let circuits = if circuits.len() > MAX_CIRCUITS_PER_ENTRY {
                &circuits[..MAX_CIRCUITS_PER_ENTRY]
            } else {
                &circuits[..]
            };

            for i in 0..circuits.len() {
                for j in 0..circuits.len() {
                    if i == j { continue; }
                    let a = &circuits[i];
                    let b = &circuits[j];

                    for combo in 0..2usize {
                        let gates = if combo == 0 {
                            // a + rev(b)
                            let mut g = a.gates.clone();
                            let mut b_rev = b.gates.clone();
                            b_rev.reverse();
                            g.extend(b_rev);
                            g
                        } else {
                            // rev(a) + b
                            let mut a_rev = a.gates.clone();
                            a_rev.reverse();
                            let mut g = a_rev;
                            g.extend(b.gates.clone());
                            g
                        };

                        let mut identity = CircuitSeq { gates };
                        identity.canonicalize();
                        remove_adjacent_equal(&mut identity.gates);

                        let n = identity.gates.len();
                        if n < 2 { continue; }

                        let wire_count = identity.max_wire() + 1;

                        for direction in [false, true] {
                            let directed: Vec<[u16; 3]> = if direction {
                                identity.gates.iter().rev().cloned().collect()
                            } else {
                                identity.gates.clone()
                            };

                            for rot in 0..n {
                                let rotation: Vec<[u16; 3]> = directed[rot..]
                                    .iter()
                                    .chain(directed[..rot].iter())
                                    .cloned()
                                    .collect();

                                // Incremental: apply one gate per step instead of
                                // recomputing from scratch for each k (was O(n²))
                                let mut polys = init_polys(wire_count);

                                for k in 1..n {
                                    apply_gate(&mut polys, rotation[k - 1]);

                                    let (canonical, perm) =
                                        match canonicalize_polys_4(polys.clone(), true) {
                                            Ok(r) => r,
                                            Err(_) => continue,
                                        };
                                    if canonical.is_empty() { continue; }

                                    let key: [u8; 16] =
                                        xxh3_128(&polys_repr_blob(&canonical)).to_le_bytes();

                                    let perm_inv = perm.invert();
                                    let tail_gates: Vec<[u16; 3]> = rotation[k..].iter().rev()
                                        .map(|&[t, c1, c2]| [
                                            perm_inv.data[t as usize] as u16,
                                            perm_inv.data[c1 as usize] as u16,
                                            perm_inv.data[c2 as usize] as u16,
                                        ])
                                        .collect();
                                    let tail_blob = CircuitSeq { gates: tail_gates }.repr_blob();
                                    let encoded = encode_circuit(&tail_blob);

                                    let out_shard = key[0] as usize;
                                    accumulators[out_shard]
                                        .lock()
                                        .unwrap()
                                        .entry(key)
                                        .or_default()
                                        .extend_from_slice(&encoded);
                                }
                            }
                        }
                    }
                }
            }
        }

        let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if n % 16 == 0 || n == 256 {
            println!("  {}/256 shards processed", n);
        }
    });

    println!("Writing to LMDB curated_{{}} shards...");
    let mut total = 0u64;
    for out_shard in 0..256usize {
        let map = accumulators[out_shard].lock().unwrap();
        if map.is_empty() { continue; }
        let mut txn = env.begin_rw_txn().expect("rw txn");
        for (key, value) in map.iter() {
            txn.put(curated_dbs[out_shard], key, value, WriteFlags::empty())
                .expect("lmdb put");
            total += 1;
        }
        txn.commit().expect("commit");
    }
    println!("Done. {} curated entries written.", total);
}
