use std::path::Path;

use rand::{seq::IteratorRandom, Rng};
use lmdb::{Cursor, Environment, Transaction};
use local_mixing::{
    circuit::{circuit::poly_to_str, CircuitSeq},
    open_shard_dbs,
};


const LMDB_PATH: &str = "./db";

fn main() {
    println!("[ Minimal Identities ]");

    let env = Environment::new()
        .set_max_readers(10000)
        .set_max_dbs(266)
        .set_map_size(800 * 1024 * 1024 * 1024)
        .open(Path::new(LMDB_PATH))
        .expect("Failed to open database.");

    let shard_dbs = open_shard_dbs(&env);

    let mut rng = rand::rng();
    let mut picked_value: Option<Vec<u8>> = None;

    while picked_value.is_none() {
        let shard_idx = rng.random_range(0..shard_dbs.len());
        let txn = env
            .begin_ro_txn()
            .expect("Failed to begin read-only transaction");
        let mut cursor = txn
            .open_ro_cursor(shard_dbs[shard_idx])
            .expect("Failed to open read-only cursor");

        picked_value = cursor
            .iter_start()
            .choose(&mut rng)
            .map(|(_key, value)| value.to_vec());
    }

    let value = picked_value.expect("Failed to sample a value from LMDB");
    let mut pos = 0;
    let mut circuits: Vec<CircuitSeq> = Vec::new();

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

    if circuits.is_empty() {
        eprintln!("No circuits found in the sampled value.");
        return;
    }

    for (idx, circuit) in circuits.iter().enumerate() {
        let n = circuit.max_wire() + 1;
        let polys = circuit.to_polynomial(n, 0, circuit.gates.len());

        println!("\nCircuit #{idx} ({} wires, {} gates)", n, circuit.gates.len());
        println!("{}", circuit.to_string(n));

        println!("Polynomials:");
        for (wire, poly) in polys.iter().enumerate() {
            println!("  x{} = {}", wire, poly_to_str(poly, n));
        }
    }

}
