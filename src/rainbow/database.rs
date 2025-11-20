use crate::circuit::{Permutation};

use bincode;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, BufWriter},
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PersistPermStore {
    pub perm: Permutation,
    pub circuits: Vec<Vec<u8>>,
    pub count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Persist {
    pub wires: usize,
    pub gates: usize,
    pub store: HashMap<Vec<u8>, Vec<Vec<u8>>>,
}

impl Persist {
    pub fn save(n: usize, m: usize, store: &HashMap<Vec<u8>, Vec<Vec<u8>>>) {
        let file = File::create(format!("./db/n{}m{}.bin", n, m))
            .expect("Failed to create file");
        let writer = BufWriter::new(file);
        let persist = Persist {
            wires: n,
            gates: m,
            store: store.clone(),
        };
        bincode::serialize_into(writer, &persist)
            .expect("Failed to serialize Persist");
    }

    //correctly loads the file for m-1 db
    pub fn load(n: usize, m: usize) -> HashMap<Vec<u8>, Vec<Vec<u8>>> {
        let filename = format!("./db/n{}m{}.bin", n, m - 1);

        let file = File::open(&filename)
            .unwrap_or_else(|_| panic!("Failed to open file: {}", filename));
        let reader = BufReader::new(file);

        let persist: Persist = bincode::deserialize_from(reader)
            .expect("Failed to deserialize Persist");

        if persist.gates != m - 1 || persist.wires != n {
            panic!(
                "Database size does not match: load has n={}, m={}; requested n={}, m={}",
                persist.wires, persist.gates, n, m
            );
        }

        persist.store
    }
}

pub fn make_persist(perm: Permutation, circuits: HashMap<Vec<u8>, bool>, count: usize) -> PersistPermStore {
    let keys: Vec<Vec<u8>> = circuits.into_keys().collect();
    PersistPermStore{ perm, circuits: keys, count, }
}