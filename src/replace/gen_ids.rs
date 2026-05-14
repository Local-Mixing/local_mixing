use crate::circuit::circuit::CircuitSeq;
use crate::replace::pairs::{gate_pair_taxonomy, GatePair};
use lmdb::{Cursor, Transaction, WriteFlags};

#[cfg(test)]
mod tests {
    use super::decode_circuits;
    use lmdb::{Cursor, Environment, Transaction};
    use std::path::Path;
    use std::time::Instant;

    #[test]
    fn shard_gate_count_histogram() {
        let env = Environment::new()
            .set_max_dbs(300)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new("./db"))
            .expect("Failed to open ./db lmdb environment");

        let mut counts = [0u64; 7]; // index = gate count; 0 unused
        let print_interval_secs = 10.0f64;
        let mut last_print = Instant::now();

        for shard in 0u16..256 {
            let name = format!("{:02x}", shard);
            let db = match env.open_db(Some(name.as_str())) {
                Ok(db) => db,
                Err(_) => continue,
            };
            let txn = env.begin_ro_txn().expect("ro txn");
            let mut cursor = txn.open_ro_cursor(db).expect("cursor");
            for (_, value) in cursor.iter() {
                for circuit in decode_circuits(value) {
                    let g = circuit.gates.len().min(6);
                    if g >= 1 {
                        counts[g] += 1;
                    }
                }
            }
            drop(cursor);
            drop(txn);

            if last_print.elapsed().as_secs_f64() >= print_interval_secs {
                print_histogram(&counts, shard + 1);
                last_print = Instant::now();
            }
        }

        print_histogram(&counts, 256);
    }

    fn print_histogram(counts: &[u64; 7], shards_done: u16) {
        println!("\n--- shard {}/256 ---", shards_done);
        let total: u64 = counts[1..].iter().sum();
        for g in 1..=6 {
            println!("  {:1} gate(s): {:>12}  ({:.1}%)",
                g, counts[g],
                if total > 0 { counts[g] as f64 / total as f64 * 100.0 } else { 0.0 });
        }
        println!("  total:    {:>12}", total);
    }
}

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

fn remove_adjacent_equal(gates: &mut Vec<[u8; 3]>) {
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


pub fn open_id_dbs(env: &lmdb::Environment) -> Vec<lmdb::Database> {
    let mut txn = env.begin_rw_txn().expect("Failed to begin rw txn for id_db setup");
    let dbs: Vec<lmdb::Database> = (0..34)
        .map(|i| {
            let name = format!("id_g{}", i);
            let db = unsafe { txn.open_db(Some(&name)) }
                .unwrap_or_else(|_| {
                    unsafe { txn.create_db(Some(&name), lmdb::DatabaseFlags::empty()) }
                        .unwrap_or_else(|e| panic!("Failed to create id_g{}: {:?}", i, e))
                });
            txn.clear_db(db).unwrap_or_else(|e| panic!("Failed to clear id_g{}: {:?}", i, e));
            db
        })
        .collect();
    txn.commit().expect("Failed to commit id_db setup txn");
    dbs
}

pub fn generate_identity_db(
    env: &lmdb::Environment,
    shard_dbs: &[lmdb::Database],
    id_dbs: &[lmdb::Database],
) {
    let mut total = 0u64;
    // Per-DB counters for sequential keys; open_id_dbs clears DBs so we start at 0.
    let mut counters = vec![0u64; 34];
    // Deduplicate within this run using a per-type seen set.
    let mut seen: Vec<std::collections::HashSet<Vec<u8>>> =
        (0..34).map(|_| std::collections::HashSet::new()).collect();

    for shard_idx in 0..256usize {
        let txn = env.begin_ro_txn().expect("ro txn");
        let db = shard_dbs[shard_idx];
        let mut cursor = txn.open_ro_cursor(db).expect("cursor");

        // Collect entries with multiple circuits
        let mut multi: Vec<Vec<u8>> = Vec::new();
        for (_, value) in cursor.iter() {
            let circuits = decode_circuits(value);
            if circuits.len() >= 2 {
                multi.push(value.to_vec());
            }
        }
        drop(cursor);
        drop(txn);

        if multi.is_empty() {
            continue;
        }

        let mut wtxn = env.begin_rw_txn().expect("rw txn");

        for value in &multi {
            let circuits = decode_circuits(value);

            // All ordered pairs (a, b) with a != b
            for i in 0..circuits.len() {
                for j in 0..circuits.len() {
                    if i == j {
                        continue;
                    }

                    let a = &circuits[i];
                    let b = &circuits[j];

                    let mut gates = a.gates.clone();
                    let mut b_rev = b.gates.clone();
                    b_rev.reverse();
                    gates.extend(b_rev);

                    let mut identity = CircuitSeq { gates };
                    identity.canonicalize();
                    remove_adjacent_equal(&mut identity.gates);

                    if identity.gates.len() < 2 {
                        continue;
                    }

                    // All rotations
                    let len = identity.gates.len();
                    for rot in 0..len {
                        let mut rotated = Vec::with_capacity(len);
                        rotated.extend_from_slice(&identity.gates[rot..]);
                        rotated.extend_from_slice(&identity.gates[..rot]);

                        let g1 = rotated[0];
                        let g2 = rotated[1];
                        let ctype = GatePair::to_int(&gate_pair_taxonomy(&g1, &g2));

                        let rotated_circuit = CircuitSeq { gates: rotated };
                        let blob = rotated_circuit.repr_blob();

                        if seen[ctype].insert(blob.clone()) {
                            let idx = counters[ctype];
                            counters[ctype] += 1;
                            wtxn.put(id_dbs[ctype], &idx.to_be_bytes(), &blob, WriteFlags::empty())
                                .expect("put identity");
                            total += 1;
                        }
                    }
                }
            }
        }

        wtxn.commit().expect("commit");
        println!("Shard {:3}/256 done", shard_idx + 1);
    }

    println!("Total identity entries written: {}", total);
}
