use crate::{
    circuit::circuit::CircuitSeq,
};
use rand::Rng;
use lmdb::{Environment, Transaction};
use lmdb::Cursor;
use lmdb::Database;
use std::{
    collections::HashMap,
};
use rand::prelude::IndexedRandom;
use crate::circuit::Permutation;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Transpositions {
    transpositions: Vec<(u8, u8, u8)>
}

impl Transpositions {
    pub fn gen_random(n: usize, _m: usize, negation_mask: &mut Vec<u8>) -> Self {
        assert!(n >= 2, "n must be at least 2");
        let mut rng = rand::rng();
        let mut transpositions = Vec::with_capacity(n);
        let n = n as u8;
        for i in (1..n).rev() {
            let negation_type = rng.random_range(0..=3);
            let j = rng.random_range(0..=i);
            if i == j {
                continue;
            }
            transpositions.push((j, i, negation_type));
            let temp = negation_mask[j as usize];
            negation_mask[j as usize] = negation_mask[i as usize];
            negation_mask[i as usize] = temp;
            if negation_type == 1 || negation_type == 3{
                negation_mask[j as usize] ^= 1;
            }
            if negation_type == 2 || negation_type == 3 {
                negation_mask[i as usize] ^= 1;
            }
        }

        Self { transpositions }
    }

    pub fn to_perm(&self, n: usize) -> Permutation {
        let mut perm = Permutation { data: Vec::with_capacity(n) };
        for i in 0..n {
            perm.data.push(self.evaluate(i as u8) as usize);
        } 
        perm
    }

    pub fn from_perm(perm: &Permutation) -> Self {
        let n = perm.data.len();
        let mut p = perm.data.clone();

        let mut inv = vec![0usize; n];
        for i in 0..n {
            inv[p[i]] = i;
        }

        let mut swaps = Vec::new();

        for i in (0..n).rev() {
            if p[i] != i {
                let j = inv[i];
                p.swap(i, j);
                inv[p[j]] = j;
                inv[p[i]] = i;

                swaps.push((i as u8, j as u8, 0));
            }
        }

        Transpositions { transpositions: swaps }
    }

    pub fn collides(s1: &(u8, u8, u8), s2: &(u8, u8, u8)) -> bool {
        let (a1, b1, _) = s1;
        let (a2, b2, _) = s2;
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
    pub fn ordered(s1: &(u8, u8, u8), s2: &(u8, u8, u8)) -> bool {
        let (a_1, b_1, _) = s1;
        let (a_2, b_2, _) = s2;
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
        swap: (u8, u8, u8), 
        env: &lmdb::Environment, 
        dbs: &HashMap<String, Database>,
    ) -> Vec<[u8;3]> {
        let (a, b, negation_type) = swap;
        let (db_name, max_entries) = if negation_type == 0 {
            ("swaps", 51)
        } else if negation_type == 1 {
            ("swapsnot1", 25)
        } else if negation_type == 2 {
            ("swapsnot2", 38)
        } else if negation_type == 3 {
            ("swapsnot12", 36)
        } else {
            panic!("Invalid negation type")
        };

        let db = dbs.get(db_name).unwrap_or_else(|| {
            panic!("Failed to get DB with name: {}", db_name);
        });

        let mut rng = rand::rng();
        let random_index = rng.random_range(0..max_entries);

        let txn = env.begin_ro_txn().expect("Failed to start txn");
        let mut cursor = txn.open_ro_cursor(*db).expect("Failed to open ro cursor");

        let value_bytes = 
            cursor.iter_start()
            .nth(random_index)
            .map(|(k, _v)| k)
            .expect("Failed to get random key");
        
        let out = CircuitSeq::from_blob(value_bytes);

        let mut c;
        loop {
            c = rng.random_range(0..n as u8);
            if c != a && c != b {
                break;
            }
        }
        let used_wires = vec![c, a, b];
        CircuitSeq::unrewire_subcircuit(&out, &used_wires).gates
    }

    pub fn gen_gates_not(
        n: usize, 
        wire: u8,
        env: &lmdb::Environment, 
        dbs: &HashMap<String, Database>,
    ) -> Vec<[u8;3]> {
        let db_name= "not";

        let db = dbs.get(db_name).unwrap_or_else(|| {
            panic!("Failed to get DB with name: {}", db_name);
        });

        let max_entries: usize = 30;

        let mut rng = rand::rng();
        let random_index = rng.random_range(0..max_entries);

        let txn = env.begin_ro_txn().expect("Failed to start txn");
        let mut cursor = txn.open_ro_cursor(*db).expect("Failed to open ro cursor");

        let value_bytes = 
            cursor.iter_start()
            .nth(random_index)
            .map(|(k, _v)| k)
            .expect("Failed to get random key");
        
        let out = CircuitSeq::from_blob(value_bytes);

        let mut a;
        loop {
            a = rng.random_range(0..n as u8);
            if a != wire {
                break;
            }
        }
        let mut b;
        loop {
            b = rng.random_range(0..n as u8);
            if b != wire && b != a{
                break;
            }
        }
        let used_wires = vec![a, wire, b];
        CircuitSeq::unrewire_subcircuit(&out, &used_wires).gates
    }

    pub fn gen_gates_cnot(
        n: usize, 
        con: u8,
        not: u8,
        env: &lmdb::Environment, 
        dbs: &HashMap<String, Database>,
    ) -> Vec<[u8;3]> {
        let db_name = "cnot";
        let max_entries = 50;

        let db = dbs.get(db_name).unwrap_or_else(|| {
            panic!("Failed to get DB with name: {}", db_name);
        });

        let mut rng = rand::rng();
        let random_index = rng.random_range(0..max_entries);

        let txn = env.begin_ro_txn().expect("Failed to start txn");
        let mut cursor = txn.open_ro_cursor(*db).expect("Failed to open ro cursor");

        let value_bytes = 
            cursor.iter_start()
            .nth(random_index)
            .map(|(k, _v)| k)
            .expect("Failed to get random key");
        
        let out = CircuitSeq::from_blob(value_bytes);

        let mut c;
        loop {
            c = rng.random_range(0..n as u8);
            if c != con && c != not {
                break;
            }
        }
        let used_wires = vec![c, con, not];
        CircuitSeq::unrewire_subcircuit(&out, &used_wires).gates
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

    pub fn evaluate(&self, input: u8) -> u8 {
        let mut val = input;
        for (a, b, _) in self.transpositions.clone() {
            if val == a {
                val = b;
            } else if val == b {
                val = a;
            }
        }

        val
    }

    pub fn concat(&self, other: &Transpositions) -> Transpositions {
        let mut new = self.clone();
        new.transpositions.extend_from_slice(&other.transpositions);
        new
    }
}

pub fn insert_wire_shuffles(
    circuit: &mut CircuitSeq, 
    n: usize,
    env: &Environment,
    dbs: &HashMap<String, Database>,
) {
    println!("Inserting wire shuffles");
    println!("Starting len: {} gates", circuit.gates.len());
    let mut t_list: Transpositions = Transpositions { transpositions: Vec::new() };
    let mut gates: Vec<[u8;3]> = Vec::new();
    let mut rng = rand::rng();
    let mut negation_mask = vec![0u8; n];
    for &gate in &circuit.gates {
        let t = Transpositions::gen_random(n, 150, &mut negation_mask);
        gates.extend_from_slice(&t.to_circuit(n, env, dbs).gates);
        t_list.transpositions.extend_from_slice(&t.transpositions);
        let a = t_list.evaluate(gate[0]);
        let b = t_list.evaluate(gate[1]);
        let c = t_list.evaluate(gate[2]);
        let gate = [a, b, c];
        if negation_mask[a as usize] == 1 {
            gates.extend_from_slice(&Transpositions::gen_gates_not(n, a, env, dbs));
            negation_mask[a as usize] = 0;
        }
        if negation_mask[b as usize] == 1 {
            gates.extend_from_slice(&Transpositions::gen_gates_not(n, b, env, dbs));
            negation_mask[b as usize] = 0;
        }
        if negation_mask[c as usize] == 1 {
            gates.extend_from_slice(&Transpositions::gen_gates_not(n, c, env, dbs));
            negation_mask[c as usize] = 0;
        }
        gates.push(gate);
    }
    let p = t_list.to_perm(n);
    let mut t = Transpositions::from_perm(&p);
    let mut wire_transpositions: HashMap<u8, (usize, usize)> = HashMap::new();

    for (i, (a, b, _)) in t.transpositions.iter().enumerate() {
        wire_transpositions.insert(*a, (i, 0));
        wire_transpositions.insert(*b, (i, 1));
    }

    const TRANSITION: [[u8; 4]; 2] = [
        // pos = 0
        [1, 0, 3, 2],
        // pos = 1
        [2, 3, 0, 1],
    ];

    for (i, val) in negation_mask.into_iter().enumerate() {
        if val == 1 {
            if let Some(swaps) = wire_transpositions.get(&(i as u8)) {
                let &(swap_idx, pos) = swaps;
                let curr_neg_type = t.transpositions[swap_idx].2;
                if pos > 1 || curr_neg_type > 3 {
                    panic!("Invalid pos or curr_neg_type");
                }
                t.transpositions[swap_idx].2 = TRANSITION[pos][curr_neg_type as usize];
                
            }
        }
    }

    let mut c = t.to_circuit(n, env, dbs).gates;
    c.reverse();
    gates.extend_from_slice(&c);
    circuit.gates = gates;
    println!("Complete. Ending len: {} gates", circuit.gates.len());
}

pub fn generate_reversible(
    c: &CircuitSeq,
    n: usize,
    env: &Environment,
    dbs: &HashMap<String, Database>, 
) -> CircuitSeq {
    let mut rev = c.clone();
    rev.gates.reverse();
    let mut gates = Vec::new();
    gates.extend_from_slice(&c.gates.clone());
    for i in 0..n {
        gates.extend_from_slice(&Transpositions::gen_gates_cnot(2 * n, i as u8, (i + n) as u8, env, dbs));
    }
    gates.extend_from_slice(&rev.gates);
    CircuitSeq { gates }
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
    use crate::{CircuitSeq, replace::transpositions::insert_wire_shuffles};
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
            .set_max_dbs(262)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new("./db"))
            .expect("failed to open lmdb");

        let dbs = open_all_dbs(&env);

        let mut gates: Vec<[u8; 3]> = Vec::new();
        let mut last = Transpositions { transpositions: Vec::new() };
        for &gate in &base.gates {

            let t = Transpositions::gen_random(64, 100, &mut Vec::new());
            // println!("t: {}", t.transpositions.len());
            if last.transpositions.is_empty() {
                gates.extend(t.to_circuit(64, &env, &dbs).gates);
            } else {
                let mut combined = last.concat(&t);
                combined.canonicalize();
                combined.filter_repeats();
                Transpositions::shoot_random_transpositions(&mut combined, 100_000);
                gates.extend(combined.to_circuit(64, &env, &dbs).gates);
            }
            let a = t.evaluate(gate[0]);
            let b = t.evaluate(gate[1]);
            let c = t.evaluate(gate[2]);
            gates.push([a, b, c]);
            last = t;
            last.transpositions.reverse();
        }
        gates.extend(last.to_circuit(64, &env, &dbs).gates);
        let new_circuit = CircuitSeq { gates };
        if base.probably_equal(&new_circuit, 64, 1_000).is_err() {
            panic!("Failed to retain functionality");
        }
        std::fs::write("test.txt", new_circuit.repr())
            .expect("failed to write test.txt");
    }

    #[test]
    fn test_transpose_shooting() {
        use crate::replace::mixing::open_all_dbs;
        let file = File::open("initial.txt").expect("failed to open initial.txt");
        let reader = BufReader::new(file);

        let circuits: Vec<String> = reader
            .lines()
            .map(|l| l.unwrap())
            .filter(|l| !l.trim().is_empty())
            .collect();

        let mut rng = rand::rng();
        let _circuit_str = circuits
            .choose(&mut rng)
            .expect("no circuits found");

        // let base = CircuitSeq::from_string(circuit_str);

        let env = Environment::new()
            .set_max_dbs(262)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new("./db"))
            .expect("failed to open lmdb");

        let dbs = open_all_dbs(&env);

        let mut t = Transpositions::gen_random(128, 500, &mut Vec::new());
        let base = t.to_circuit(128, &env, &dbs);
        Transpositions::shoot_random_transpositions(&mut t, 100_000);
        let new_circuit = t.to_circuit(128, &env, &dbs);
        if base.probably_equal(&new_circuit, 128, 1_000).is_err() {
            panic!("Failed to retain functionality after shooting");
        }
        t.canonicalize();
        t.filter_repeats();
        let new_circuit = t.to_circuit(128, &env, &dbs);
        if base.probably_equal(&new_circuit, 128, 1_000).is_err() {
            panic!("Failed to retain functionality after filtering");
        }
        println!("They are equal");
    }

    #[test]
    fn test_insert_shuffles() {
        use crate::replace::mixing::open_all_dbs;
        use std::io::Write;
        let file = File::open("initial.txt").expect("failed to open initial.txt");
        let mut reader = BufReader::new(file);

        let mut circuit_str = String::new();
        reader
            .read_line(&mut circuit_str)
            .expect("failed to read circuit");

        let circuit_str = circuit_str.trim();
        assert!(!circuit_str.is_empty(), "initial.txt is empty");

        let base = CircuitSeq::from_string(circuit_str);

        let env = Environment::new()
            .set_max_dbs(262)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new("./db"))
            .expect("failed to open lmdb");

        let dbs = open_all_dbs(&env);

        let mut new_circuit = base.clone();
        insert_wire_shuffles(&mut new_circuit, 64, &env, &dbs);

        if base.probably_equal(&new_circuit, 64, 1_000).is_err() {
            panic!("Failed to retain functionality");
        }

        let mut out = File::create("shuffled.txt")
            .expect("failed to create shuffled.txt");
        writeln!(out, "{}", new_circuit.repr()).unwrap();

        println!("They are equal and written to shuffled.txt");
    }

    #[test]
    fn test_transposition_rev() {
        use crate::replace::mixing::open_all_dbs;
        use crate::replace::transpositions::HashMap;
        let env = Environment::new()
            .set_max_dbs(262)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new("./db"))
            .expect("failed to open lmdb");

        let dbs = open_all_dbs(&env);
        let n = 64;
        let mut gates: Vec<[u8;3]> = Vec::new();
        let mut rng = rand::rng();
        let mut negation_mask = vec![0u8; n];
        let t = Transpositions::gen_random(n, 150, &mut negation_mask);
        gates.extend_from_slice(&t.to_circuit(n, &env, &dbs).gates);
        let p = t.to_perm(n);
        let mut t = Transpositions::from_perm(&p);
        let mut wire_transpositions: HashMap<u8, Vec<(usize, usize)>> = HashMap::new();
        for (i, (a, b, _)) in t.transpositions.clone().into_iter().enumerate() {
            wire_transpositions
            .entry(a)
            .or_default()
            .push((i, 0));

            wire_transpositions
                .entry(b)
                .or_default()
                .push((i, 1));
        }

        const TRANSITION: [[u8; 4]; 2] = [
            // pos = 0
            [1, 0, 3, 2],
            // pos = 1
            [2, 3, 0, 1],
        ];

        for (i, val) in negation_mask.into_iter().enumerate() {
            if val == 1 {
                gates.extend_from_slice(&Transpositions::gen_gates_not(n, i as u8, &env, &dbs));
            }
        }
        let mut tr: Vec<(u8, u8, u8)> = Vec::new();
        for (a,b,_) in t.transpositions{
            tr.push((a,b,0));
        }
        let t = Transpositions { transpositions: tr }; 
        let mut c = t.to_circuit(n, &env, &dbs).gates;
        c.reverse();
        gates.extend_from_slice(&c);

        let c = CircuitSeq { gates };
        if c.probably_equal(&CircuitSeq{ gates: Vec::new() }, 64, 1_000).is_err() {
            panic!("Lost functionality");
        }
    }
    #[test]
    fn test_wire_shifting2() {
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
            .set_max_dbs(262)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new("./db"))
            .expect("failed to open lmdb");

        let dbs = open_all_dbs(&env);
        let t = Transpositions::gen_random(64, 100, &mut Vec::new());
        let mut gates: Vec<[u8; 3]> = Vec::new();
        gates.extend(t.to_circuit(64, &env, &dbs).gates);
        for &gate in &base.gates {
            let a = t.evaluate(gate[0]);
            let b = t.evaluate(gate[1]);
            let c = t.evaluate(gate[2]);
            gates.push([a, b, c]);
        }
        let mut tc = t.to_circuit(64, &env, &dbs).gates;
        tc.reverse();
        gates.extend(&tc);
        let new_circuit = CircuitSeq { gates };
        if base.probably_equal(&new_circuit, 64, 1_000).is_err() {
            panic!("Failed to retain functionality");
        }
        std::fs::write("test.txt", new_circuit.repr())
            .expect("failed to write test.txt");
    }
}
