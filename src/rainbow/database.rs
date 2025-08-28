use crate::{
    circuit::{Circuit, Gate, Permutation},
    circuit::control_functions::Gate_Control_Func,
    rainbow::{Canonicalization, CandSet, constants::{self, CONTROL_FUNC_TABLE}},
};

use bincode;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PersistPermStore {
    pub perm: Permutation,
    pub circuits: Vec<String>,
    pub count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Persist {
    pub wires: usize,
    pub gates: usize,
    pub store: HashMap<String, PersistPermStore>,
}

impl PersistPermStore {
    pub fn to_string(&self, w: usize) -> String {
        let mut s = format!("Perm: {:?}\n", self.perm.to_cycle());
        if self.count == 1 {
            s.push_str( "1 circuit\n");
        } else {
            s.push_str(&format!(" {} circuits, {} unique\n", self.count, self.circuits.len()));
        }

        for c in &self.circuits {
            let circuit = Circuit::from_string_compressed(w, c);
            s.push_str(&format!("{}\n", circuit.to_string()));
        }
        s
    }
}

impl Persist {
    pub fn save(n: usize, m: usize, store: &HashMap<String, PersistPermStore>) {
        let file = File::create(format!("db/n{}m{}.bin", n, m))
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

    pub fn load(n: usize, m: usize, path: &str) -> HashMap<String, PersistPermStore> {
        let file = File::open(path).expect("Failed to open file");
        let reader = BufReader::new(file);
        let persist: Persist = bincode::deserialize_from(reader)
            .expect("Failed to deserialize Persist");
        if persist.gates != m || persist.wires != n {
            panic!(
                "Database size does not match: load has n={}, m={}; requested n={}, m={}",
                persist.wires, persist.gates, n, m
            );
        }
        persist.store
    }
}

pub fn make_persist(perm: Permutation, circuits: HashMap<String, bool>, count: usize) -> PersistPermStore {
    let keys: Vec<String> = circuits.into_keys().collect();
    PersistPermStore{ perm, circuits: keys, count, }
}