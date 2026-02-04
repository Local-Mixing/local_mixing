use crate::{
    circuit::circuit::CircuitSeq,
    random::random_data::shoot_random_gate,
    replace::replace::{
        compress, 
        compress_big, 
        compress_big_ancillas, 
        expand_big, 
        obfuscate, 
        outward_compress, 
        random_id, 
        replace_pair_distances_linear, 
        replace_pairs, 
        replace_sequential_pairs,
        interleave,
        replace_single_pair
    },
};
// use crate::random::random_data::random_walk_no_skeleton;
use rand::prelude::SliceRandom;
use itertools::Itertools;
// use lmdb::RoTransaction;
use rand::Rng;
use rayon::prelude::*;
use std::io;
use std::io::Read;
use rusqlite::{Connection, OpenFlags};
use lmdb::Transaction;
use lmdb::Cursor;
use once_cell::sync::Lazy;
use lmdb::Database;
use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io::Write,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc,
        Mutex,
    },
    time::Instant,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Transpositions {
    transpositions: Vec<(u8, u8)>
}

impl Transpositions {
    pub fn gen_random(n: usize, m: usize) -> Self {
        assert!(n >= 2, "n must be at least 2");

        let mut rng = rand::rng();
        let mut transpositions = Vec::with_capacity(m);

        for _ in 0..m {
            let a = rng.random_range(0..n) as u8;
            let mut b = rng.random_range(0..n) as u8;

            while a == b {
                b = rng.random_range(0..n) as u8;
            }

            let (x, y) = if a < b { (a, b) } else { (b, a) };
            transpositions.push((x, y));
        }

        Self { transpositions }
    }

    pub fn collides(s1: &(u8, u8), s2: &(u8, u8)) -> bool {
        let (a1, b1) = s1;
        let (a2, b2) = s2;
        a1 == a2 ||
        a1 == b2 ||
        b1 == a2 ||
        b1 == b2
    }

    pub fn shoot_random_transpositions(transpositions: &mut Transpositions, rounds: usize) {
        let mut rng = rand::rng();
        let len = transpositions.transpositions.len();

        if len == 0 {
            return
        }

        for _ in 0..rounds {
            let gate_idx = rng.random_range(0..len);
            let go_left: bool = rng.random_bool(0.5);

            if go_left {
                // Shoot left
                let mut target = gate_idx;
                while target > 0 {
                    if Transpositions::collides(&transpositions.transpositions[target - 1], &transpositions.transpositions[gate_idx]) {
                        break;
                    }
                    target -= 1;
                }
                target = rng.random_range(target..=gate_idx);
                if target != gate_idx {
                    let gate = transpositions.transpositions.remove(gate_idx);
                    transpositions.transpositions.insert(target, gate);
                }
            } else {
                // Shoot right
                let mut target = gate_idx;
                while target + 1 < len {
                    if Transpositions::collides(&transpositions.transpositions[target + 1], &transpositions.transpositions[gate_idx]) {
                        break;
                    }
                    target += 1;
                }
                target = rng.random_range(gate_idx..=target);
                if target != gate_idx {
                    let gate = transpositions.transpositions.remove(gate_idx);
                    transpositions.transpositions.insert(target, gate);
                }
            }
        }
    }

    //b is greater
    pub fn ordered(s1: &(u8, u8), s2: &(u8, u8)) -> bool {
        let (a_1, b_1) = s1;
        let (a_2, b_2) = s2;
        if a_1 > a_2 {
            return false
        } else if a_1 == a_2{
            if b_1 > b_2 {
                return false
            }
        }
        true
    }

    pub fn canonicalize(&mut self) {
        for i in 1..self.transpositions.len() {
            let ti = self.transpositions[i];
            let mut to_swap: Option<usize> = None;

            let mut j = i;
            while j > 0 {
                j -= 1;
                let tj = self.transpositions[j];

                if Transpositions::collides(&ti, &tj) {
                    break;
                } else if !Transpositions::ordered(&tj, &ti) {
                    to_swap = Some(j);
                }
            }
            if let Some(pos) = to_swap {
                let g = self.transpositions[i];
                self.transpositions.remove(i);
                self.transpositions.insert(pos, g);
            }
        }
    }

    pub fn gen_gates_swap(
        n: usize, 
        swap: (u8, u8), 
        env: &lmdb::Environment, 
        dbs: &HashMap<String, Database>,
    ) -> Vec<[u8;3]> {
        let (a, b) = swap;
        let db_name = "swaps";

        let db = dbs.get(db_name).unwrap_or_else(|| {
            panic!("Failed to get DB with name: {}", db_name);
        });

        let max_entries: usize = 51;

        let mut rng = rand::rng();
        let random_index = rng.random_range(0..max_entries);

        let txn = env.begin_ro_txn().expect("Failed to start txn");
        let mut cursor = txn.open_ro_cursor(*db).expect("Failed to open ro cursor");

        let value_bytes = if n != 128 {
            cursor.iter_start()
            .nth(random_index)
            .map(|(k, _v)| k)
            .expect("Failed to get random key")
        } else {
            cursor.iter_start()
            .nth(random_index)
            .map(|(_k, v)| v)
            .expect("Failed to get random val")
        };
        let out = CircuitSeq::from_blob(value_bytes);

        let mut c;
        loop {
            c = rng.random_range(0..n as u8);
            if c != a && c != b {
                break;
            }
        }
        let used_wires = vec![c, a, b];
        let idxs: Vec<usize> = (0..out.gates.len()).collect();
        CircuitSeq::rewire_subcircuit(&out, &idxs, &used_wires).gates
    }

    pub fn to_circuit(
        &self,
        n: usize,
        env: &lmdb::Environment,
        dbs: &HashMap<String, Database>,
    ) -> CircuitSeq {
        let mut gates: Vec<[u8; 3]> = Vec::new();

        for &swap in &self.transpositions {
            gates.extend_from_slice(&Self::gen_gates_swap(n, swap, env, dbs));
        }

        CircuitSeq { gates }
    }

    pub fn filter_repeats(&mut self) {
        let mut i = 0;
        while i < self.transpositions.len().saturating_sub(1) {
            if self.transpositions[i] == self.transpositions[i + 1] {
                self.transpositions.drain(i..=i + 1);
                i = i.saturating_sub(2);
            } else {
                i += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use lmdb::Environment;
    use std::{
        fs::File,
        io::{BufRead, BufReader},
        path::Path,
    };
    use rand::prelude::IndexedRandom;
    use crate::CircuitSeq;
    use crate::replace::transpositions::Transpositions;
    #[test]
    fn test_wire_shifting() {
        use crate::replace::mixing::open_all_dbs;
        let file = File::open("initial.txt").expect("failed to open initial.txt");
        let reader = BufReader::new(file);

        let circuits: Vec<String> = reader
            .lines()
            .map(|l| l.unwrap())
            .filter(|l| !l.trim().is_empty())
            .collect();

        let mut rng = rand::rng();
        let circuit_str = circuits
            .choose(&mut rng)
            .expect("no circuits found");

        let base = CircuitSeq::from_string(circuit_str);

        let env = Environment::new()
            .set_max_dbs(258)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new("./db"))
            .expect("failed to open lmdb");

        let dbs = open_all_dbs(&env);

        let mut gates: Vec<[u8; 3]> = Vec::new();

        let t = Transpositions::gen_random(128, 100);
        gates.extend(t.to_circuit(128, &env, &dbs).gates);
        let tc = t.to_circuit(128, &env, &dbs);
        gates.extend(tc.gates.into_iter().rev());
        for &gate in &base.gates {
            gates.push(gate);

            let t = Transpositions::gen_random(128, 100);
            println!("t: {}", t.transpositions.len());
            gates.extend(t.to_circuit(128, &env, &dbs).gates);
            let tc = t.to_circuit(128, &env, &dbs);
            println!("c: {}", tc.gates.len());
            gates.extend(tc.gates.into_iter().rev());
        }

        let new_circuit = CircuitSeq { gates };
        if base.probably_equal(&new_circuit, 128, 1_000).is_err() {
            panic!("Failed to retain functionality");
        }
        std::fs::write("test.txt", new_circuit.repr())
            .expect("failed to write test.txt");
    }
}
