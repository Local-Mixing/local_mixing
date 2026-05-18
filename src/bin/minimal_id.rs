use std::{collections::HashMap, path::Path, sync::Mutex};

use lmdb::{Cursor, Environment, Transaction};
use local_mixing::{
    circuit::{CircuitSeq, circuit::poly_degree},
    open_shard_dbs,
};
use rand::seq::SliceRandom;
use rayon::prelude::*;

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

type WireCount = usize;
type GateCount = usize;

type CktShape = Option<(WireCount, GateCount)>;

fn make_shape(a: CktShape, b: CktShape) -> (CktShape, CktShape) {
    if a <= b { (a, b) } else { (b, a) }
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

    let shape_counter: Mutex<HashMap<(CktShape, CktShape), usize>> = Mutex::new(HashMap::new());

    let mut shard_indices: Vec<usize> = (0..shard_dbs.len()).collect();
    let mut rng = rand::thread_rng();
    // shard_indices.shuffle(&mut rng);

    // this should be threaded??
    shard_indices.par_iter().for_each(|shard_idx| {
        println!("Shard {}", shard_idx);
        let db = shard_dbs[*shard_idx];
        let txn = env
            .begin_ro_txn()
            .expect("Failed to begin read-only transaction");
        let mut cursor = txn
            .open_ro_cursor(db)
            .expect("Failed to open read-only cursor");

        let mut local_shape_counter: HashMap<(CktShape, CktShape), usize> = HashMap::new();

        for (_, value) in cursor.iter_start() {
            let circuits = decode_circuits(value);

            let shapes: Vec<(WireCount, GateCount)> = circuits
                .iter()
                .map(|circuit| (circuit.max_wire() + 1, circuit.gates.len()))
                .collect();

            // Circuit counter (0,0 second)
            for sh in shapes.iter() {
                *local_shape_counter.entry((Some(*sh), Some((0, 0)))).or_insert(0) += 1;
            }

            // If there is only one shape present, also record an entry pairing it with None
            if shapes.len() == 1 {
                *local_shape_counter.entry((Some(shapes[0]), None)).or_insert(0) += 1;
            } else {
                for (i, shape_i) in shapes.iter().enumerate() {
                    for shape_j in shapes.iter().skip(i + 1) {
                        let key = make_shape(Some(*shape_i), Some(*shape_j));
                        *local_shape_counter.entry(key).or_insert(0) += 1;
                    }
                }
            }
        }

        let mut shape_counter = shape_counter.lock().unwrap();
        for (key, count) in local_shape_counter {
            *shape_counter.entry(key).or_insert(0) += count;
        }
    });
    println!("{:?}", shape_counter.lock().unwrap());
}
