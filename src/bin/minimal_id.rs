use std::path::Path;

use lmdb::{Cursor, Environment, Transaction};
use local_mixing::{
    circuit::{CircuitSeq, circuit::poly_degree},
    open_shard_dbs,
};
use rand::seq::SliceRandom;

const LMDB_PATH: &str = "./db";

fn decode_circuits(value: &[u8]) -> Vec<CircuitSeq> {
    let mut pos = 0;
    let mut circuits = Vec::new();

    while pos < value.len() {
        if pos + 1 > value.len() {
            break;
        }

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

fn main() {
    println!("[ Minimal Identities ]");

    let env = Environment::new()
        .set_max_readers(10000)
        .set_max_dbs(266)
        .set_map_size(800 * 1024 * 1024 * 1024)
        .open(Path::new(LMDB_PATH))
        .expect("Failed to open database.");

    let shard_dbs = open_shard_dbs(&env);

    loop {
        let mut shard_indices: Vec<usize> = (0..shard_dbs.len()).collect();
        let mut rng = rand::thread_rng();
        shard_indices.shuffle(&mut rng);

        for shard_idx in shard_indices {
            let db = shard_dbs[shard_idx];
            let txn = env
                .begin_ro_txn()
                .expect("Failed to begin read-only transaction");
            let mut cursor = txn
                .open_ro_cursor(db)
                .expect("Failed to open read-only cursor");

            for (_, value) in cursor.iter_start() {
                let circuits = decode_circuits(value);
                let num_circuits = circuits.len();

                let circuit = &circuits[0];
                let n = circuit.max_wire() + 1;
                let polys = circuit.to_polynomial(n, 0, circuit.gates.len());

                let max_degree = polys.iter().map(|p| poly_degree(p)).max().unwrap_or(0);
                let max_terms = polys.iter().map(|p| p.len()).max().unwrap_or(0);

                println!(
                    "ckt={} n={} m={} deg={} terms={}",
                    num_circuits,
                    n,
                    circuit.gates.len(),
                    max_degree,
                    max_terms
                );

                if num_circuits > 1 {
                    println!("STOP: Found {} circuits in a single key", num_circuits);
                    for (idx, circuit) in circuits.iter().enumerate() {
                        let n = circuit.max_wire() + 1;
                        println!("Circuit {}:", idx);
                        println!("{}", circuit.to_string(n));
                    }
                    return;
                }
            }
        }
    }
}
