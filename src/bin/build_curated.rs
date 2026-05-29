use lmdb::{Cursor, DatabaseFlags, Environment, EnvironmentFlags, Transaction, WriteFlags};
use local_mixing::circuit::circuit::{polys_repr_blob, CircuitSeq};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
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

// Map actual wire w to DB wire space.
// Prefix wires (in used_map) get their canonical position.
// Extra tail wires (not in prefix) get fresh consecutive indices.
fn map_wire(
    w: u16,
    used_map: &HashMap<u16, u16>,
    extra_map: &mut HashMap<u16, u16>,
    next_extra: &mut u16,
) -> u16 {
    if let Some(&db) = used_map.get(&w) {
        db
    } else {
        let next = *next_extra;
        let db = *extra_map.entry(w).or_insert_with(|| {
            *next_extra += 1;
            next
        });
        db
    }
}

fn process_shard(
    shard_idx: usize,
    src_db: lmdb::Database,
    env: &Environment,
) -> HashMap<[u8; 16], Vec<u8>> {
    let mut local_acc: HashMap<[u8; 16], Vec<u8>> = HashMap::new();

    eprintln!("  shard {:02x}: scanning...", shard_idx);
    let entries: Vec<Vec<u8>> = {
        let txn = env.begin_ro_txn().expect("ro txn");
        let mut cursor = txn.open_ro_cursor(src_db).expect("cursor");
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
    eprintln!("  shard {:02x}: {} qualifying entries, processing...", shard_idx, entries.len());

    for (entry_idx, value) in entries.iter().enumerate() {
        if entry_idx > 0 && entry_idx % 5000 == 0 {
            eprintln!("  shard {:02x}: {}/{} entries, {} keys so far",
                shard_idx, entry_idx, entries.len(), local_acc.len());
        }
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
                        let mut g = a.gates.clone();
                        let mut b_rev = b.gates.clone();
                        b_rev.reverse();
                        g.extend(b_rev);
                        g
                    } else {
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
                    if n < 3 { continue; }

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

                            // Store completions for all prefix lengths k=1..n-1.
                            // expand_curated_lmdb looks up arbitrary k-gate prefixes,
                            // so we need entries for every k.
                            for k in 1..n {
                                let prefix = CircuitSeq { gates: rotation[..k].to_vec() };
                                let (canon_polys, perm4, used) =
                                    prefix.canonicalize_polys_single(false);
                                if canon_polys.is_empty() { continue; }

                                let key: [u8; 16] =
                                    xxh3_128(&polys_repr_blob(&canon_polys)).to_le_bytes();

                                // Encode tail in compact DB wire space.
                                // perm4.data[canonical_pos] = compact_index (0..|used|)
                                // perm4_inv.data[compact_index] = canonical_pos = DB wire
                                // So prefix wire used[i] → DB wire perm4_inv[i].
                                // Extra tail wires (not in prefix) get fresh DB indices >= |used|.
                                let perm4_inv = perm4.invert();
                                let used_map: HashMap<u16, u16> = used.iter().enumerate()
                                    .map(|(i, &w)| (w, perm4_inv.data[i] as u16))
                                    .collect();
                                let mut extra_map: HashMap<u16, u16> = HashMap::new();
                                let mut next_extra = used.len() as u16;

                                let mut tail_gates: Vec<[u16; 3]> = Vec::new();
                                for &[t, c1, c2] in rotation[k..].iter().rev() {
                                    tail_gates.push([
                                        map_wire(t, &used_map, &mut extra_map, &mut next_extra),
                                        map_wire(c1, &used_map, &mut extra_map, &mut next_extra),
                                        map_wire(c2, &used_map, &mut extra_map, &mut next_extra),
                                    ]);
                                }

                                let tail_blob = CircuitSeq { gates: tail_gates }.repr_blob();
                                let encoded = encode_circuit(&tail_blob);

                                local_acc
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

    if !entries.is_empty() {
        eprintln!("  shard {:02x}: {} qualifying entries, {} output keys",
            shard_idx, entries.len(), local_acc.len());
    }

    local_acc
}

fn write_accumulator(
    acc: &HashMap<[u8; 16], Vec<u8>>,
    curated_dbs: &[lmdb::Database],
    env: &Environment,
) -> u64 {
    if acc.is_empty() { return 0; }

    let mut txn = env.begin_rw_txn().expect("rw txn");
    let mut count = 0u64;

    for (key, new_data) in acc {
        let out_shard = key[0] as usize;
        let existing = txn.get(curated_dbs[out_shard], key).ok().map(|v| v.to_vec());
        let value = if let Some(mut ev) = existing {
            ev.extend_from_slice(new_data);
            ev
        } else {
            new_data.clone()
        };
        txn.put(curated_dbs[out_shard], key, &value, WriteFlags::empty())
            .expect("lmdb put");
        count += 1;
    }

    txn.commit().expect("commit");
    count
}

fn main() {
    let env = Arc::new(Environment::new()
        .set_flags(EnvironmentFlags::WRITE_MAP | EnvironmentFlags::MAP_ASYNC | EnvironmentFlags::NO_SYNC)
        .set_max_dbs(600)
        .set_max_readers(10000)
        .set_map_size(800 * 1024 * 1024 * 1024)
        .open(Path::new("./db"))
        .expect("Failed to open ./db"));

    println!("Opening source shard databases (00..ff)...");
    let src_dbs: Vec<lmdb::Database> = (0u16..256)
        .map(|s| {
            let name = format!("{:02x}", s);
            env.open_db(Some(name.as_str()))
                .unwrap_or_else(|e| panic!("Failed to open shard {}: {:?}", name, e))
        })
        .collect();

    println!("Creating/opening curated_{{}} shard databases...");
    let curated_dbs: Vec<lmdb::Database> = (0u16..256)
        .map(|s| {
            let name = format!("curated_{:02x}", s);
            env.create_db(Some(name.as_str()), DatabaseFlags::empty())
                .unwrap_or_else(|e| panic!("Failed to create {}: {:?}", name, e))
        })
        .collect();

    println!("Clearing existing curated_{{}} data...");
    {
        let mut txn = env.begin_rw_txn().expect("rw txn for clear");
        for &db in &curated_dbs {
            txn.clear_db(db).expect("clear curated db");
        }
        txn.commit().expect("commit clear");
    }
    println!("Cleared.");

    let curated_dbs = Arc::new(curated_dbs);

    let done = Arc::new(AtomicUsize::new(0));
    let total_written = Arc::new(AtomicUsize::new(0));
    let write_lock = Arc::new(Mutex::new(()));

    println!("Processing 256 shards (parallel)...");

    (0..256usize).into_par_iter().for_each(|shard_idx| {
        let env = Arc::clone(&env);
        let curated_dbs = Arc::clone(&curated_dbs);
        let done = Arc::clone(&done);
        let total_written = Arc::clone(&total_written);
        let write_lock = Arc::clone(&write_lock);

        let local_acc = process_shard(shard_idx, src_dbs[shard_idx], &env);

        if !local_acc.is_empty() {
            let _guard = write_lock.lock().unwrap();
            let written = write_accumulator(&local_acc, &curated_dbs, &env);
            total_written.fetch_add(written as usize, Ordering::Relaxed);
        }

        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
        // Print every shard for first 32, then every 8
        if n <= 32 || n % 8 == 0 || n == 256 {
            println!("  {}/256 shards done, {} keys written so far",
                n, total_written.load(Ordering::Relaxed));
        }
    });

    println!("Done. {} total curated keys written.", total_written.load(Ordering::Relaxed));
}
