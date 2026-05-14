use crate::circuit::circuit::CircuitSeq;
use crate::replace::pairs::{gate_pair_taxonomy, GatePair};
use lmdb::{Cursor, Transaction, WriteFlags};

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
                        let key = rotated_circuit.repr_blob();
                        let _ = wtxn.put(id_dbs[ctype], &key, &[], WriteFlags::NO_OVERWRITE);
                        total += 1;
                    }
                }
            }
        }

        wtxn.commit().expect("commit");
        println!("Shard {:3}/256 done", shard_idx + 1);
    }

    println!("Total identity entries written: {}", total);
}
