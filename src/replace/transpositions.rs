// For adding wire shuffles and bit flips
use std::collections::HashMap;
use heed::Env;
use rand::Rng;
use rand::seq::SliceRandom;
use lmdb::{Cursor, Database, Environment, Transaction};
use crate::circuit::{circuit::CircuitSeq, Permutation};
use crate::circuit::circuit::rewire_gate_ver;
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Transpositions {
    transpositions: Vec<(u8, u8, u8)>
}

impl Transpositions {
    // Use Knuth Shuffle to get a random wire shuffle and then choose a random negation
    pub fn gen_random_knuth(n: usize, _m: usize, negation_mask: &mut Vec<u8>) -> Self {
        assert!(n >= 2, "n must be at least 2");
        let mut rng = rand::rng();
        let mut transpositions = Vec::with_capacity(n);
        let n = (n - 1) as u8;
        for i in (1..=n).rev() {
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

    // Simple random wire shuffle with negation
    pub fn gen_random_simple(n: usize, m: usize, negation_mask: &mut Vec<u8>) -> Self {
        assert!(n >= 2, "n must be at least 2");
        let mut rng = rand::rng();
        let mut transpositions = Vec::with_capacity(m);
        for _ in 0..m {
            let negation_type = rng.random_range(0..=3);
            let mut i: usize = rng.random_range(0..n);
            let mut j: usize;
            loop {
                j = rng.random_range(0..n);
                if i != j {
                    break;
                }
            }
            // Maintain correct ordering
            if i < j {
                let temp = i;
                i = j;
                j = temp;
            }
            transpositions.push((j as u8, i as u8, negation_type));
            // Adjust negation mask appropriately
            let temp = negation_mask[j];
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

    // Simple randomization
    // Unused
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

    // Unused
    // Smallest lexicographical ordering
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

    // Generate from the LMDB
    // LMDB swaps wire 1 and wire 2
    // This relabels wire 1 to swap.0 and and wire 2 to swap.1
    pub fn gen_gates_swap(
        n: usize, 
        swap: (u8, u8, u8), 
        env: &lmdb::Environment, 
        dbs: &HashMap<String, Database>,
    ) -> Vec<[u8;3]> {
        let (a, b, negation_type) = swap;
        let (db_name, max_entries) = if negation_type == 0 {
            ("swap", 36)
        } else if negation_type == 1 {
            ("swapnot1", 21)
        } else if negation_type == 2 {
            ("swapnot2", 16)
        } else if negation_type == 3 {
            ("swapnot12", 25)
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
            c = rng.random_range(0..=(n-1) as u8);
            if c != a && c != b {
                break;
            }
        }
        let used_wires = vec![c, a, b];
        CircuitSeq::unrewire_subcircuit(&out, &used_wires).gates
    }

    // Generate from the LMDB
    // LMDB swaps wire 1 and wire 2
    // This relabels wire 1 to swap.0 and and wire 2 to swap.1
    pub fn gen_gates_swap_restricted(
        n: usize, 
        swap: (u8, u8, u8), 
        env: &lmdb::Environment, 
        dbs: &HashMap<String, Database>,
        restricted: &Vec<u8>,
    ) -> Vec<[u8;3]> {
        let (a, b, negation_type) = swap;
        let (db_name, max_entries) = if negation_type == 0 {
            ("swap", 36)
        } else if negation_type == 1 {
            ("swapnot1", 21)
        } else if negation_type == 2 {
            ("swapnot2", 16)
        } else if negation_type == 3 {
            ("swapnot12", 25)
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
            c = rng.random_range(0..=(n-1) as u8);
            if c != a && c != b && !restricted.contains(&c){
                break;
            }
        }
        let used_wires = vec![c, a, b];
        CircuitSeq::unrewire_subcircuit(&out, &used_wires).gates
    }

    // LMDB wire 1 gets flipped
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

        let max_entries: usize = 18;

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
            a = rng.random_range(0..=(n-1) as u8);
            if a != wire {
                break;
            }
        }
        let mut b;
        loop {
            b = rng.random_range(0..=(n-1) as u8);
            if b != wire && b != a{
                break;
            }
        }
        let used_wires = vec![a, wire, b];
        CircuitSeq::unrewire_subcircuit(&out, &used_wires).gates
    }

    // LMDB wire 2 gets flipped if wire 1 is true
    pub fn gen_gates_cnot(
        n: usize, 
        con: u8,
        not: u8,
        env: &lmdb::Environment, 
        dbs: &HashMap<String, Database>,
    ) -> Vec<[u8;3]> {
        let db_name = "cnot";
        let max_entries = 19;

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
            c = rng.random_range(0..=(n-1) as u8);
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

    pub fn restricted_to_circuit(
        &self,
        n: usize,
        env: &lmdb::Environment,
        dbs: &HashMap<String, Database>,
        restricted: &Vec<u8>
    ) -> CircuitSeq {
        let mut gates: Vec<[u8; 3]> = Vec::new();

        for &swap in &self.transpositions {
            gates.extend_from_slice(&Self::gen_gates_swap_restricted(n, swap, env, dbs, restricted));
        }

        CircuitSeq { gates }
    }

    pub fn restricted_to_circuit_RI(
        first: Transpositions,
        middle: Transpositions,
        second: Transpositions,
        n: usize,
        env: &lmdb::Environment,
        dbs: &HashMap<String, Database>,
        first_bounds: (usize, usize),
        second_bounds: (usize, usize),
        
    ) -> CircuitSeq {
        let mut gates: Vec<[u8; 3]> = Vec::new();

        for i in 0..first.transpositions.len() {
            let mut swap = first.transpositions[i];
            let n = first_bounds.1 - first_bounds.0 + 1;
            let offset = first_bounds.0 as u8;
            swap.0 -= offset;
            swap.1 -= offset;
            let first_circuit = Self::gen_gates_swap(n, swap, env, dbs);
            let first_circuit: Vec<[u8; 3]> = first_circuit
                .into_iter()
                .map(|[a, b, c]| [a + offset, b + offset, c + offset])
                .collect();

            gates.extend_from_slice(&first_circuit);
        }

        for i in (0..middle.transpositions.len()).rev() {
            let swap = middle.transpositions[i];
            let mut middle_circuit = Self::gen_gates_swap(n, swap, env, dbs);
            middle_circuit.reverse();
            gates.extend_from_slice(&middle_circuit);
        }

        for i in 0..second.transpositions.len() {
            let mut swap = second.transpositions[i];
            let n = second_bounds.1 - second_bounds.0 + 1;
            let offset = second_bounds.0 as u8;
            swap.0 -= offset;
            swap.1 -= offset;
            let second_circuit = Self::gen_gates_swap(n, swap, env, dbs);
            let second_circuit: Vec<[u8; 3]> = second_circuit
                .into_iter()
                .map(|[a, b, c]| [a + offset, b + offset, c + offset])
                .collect();
            
            gates.extend_from_slice(&second_circuit);
        }
        CircuitSeq { gates }
    }

    pub fn restricted_to_circuit_rewired_and_insert(
        t_rewired: Vec<(Transpositions, Permutation)>,
        c: CircuitSeq,
        n: usize,
        env: &lmdb::Environment,
        dbs: &HashMap<String, Database>,
        first_bounds: (usize, usize),
        second_bounds: (usize, usize), 
    ) -> CircuitSeq {
        let mut gates: Vec<[u8; 3]> = Vec::new();
        let first = &t_rewired[0].0;
        let mut first_gates: Vec<[u8;3]> = Vec::new();
        for i in 0..first.transpositions.len() {
            let mut swap = first.transpositions[i];
            if swap.0 == swap.1 {
                continue;
            }
            let n = first_bounds.1 - first_bounds.0 + 1;
            let offset = first_bounds.0 as u8;
            swap.0 -= offset;
            swap.1 -= offset;
            let first_circuit = Self::gen_gates_swap(n, swap, env, dbs);
            let first_circuit: Vec<[u8; 3]> = first_circuit
                .into_iter()
                .map(|[a, b, c]| [a + offset, b + offset, c + offset])
                .collect();

            first_gates.extend_from_slice(&first_circuit);
        }
        rewire_gate_ver(&mut first_gates, &t_rewired[0].1, n);
        let curr_len = first_gates.len();
        first_gates.insert(curr_len/2, c.gates[0]);
        gates.extend_from_slice(&first_gates);

        let t_len = t_rewired.len();
        let mut j = 1;
        while j < t_len-1 {
            // m's
            let middle = &t_rewired[j].0;
            let mut middle_gates: Vec<[u8;3]> = Vec::new();
            for i in (0..middle.transpositions.len()).rev() {
                let swap = middle.transpositions[i];
                if swap.0 == swap.1 {
                    continue;
                }
                let mut middle_circuit = Self::gen_gates_swap(n, swap, env, dbs);
                middle_circuit.reverse();
                middle_gates.extend_from_slice(&middle_circuit);
            }
            rewire_gate_ver(&mut middle_gates, &t_rewired[j].1, n);
            gates.extend_from_slice(&middle_gates);
            j += 1;
            if j == t_len-1 {
                break;
            }
            // B's
            let middle = &t_rewired[j].0;
            let mut middle_gates: Vec<[u8;3]> = Vec::new();
            for i in 0..middle.transpositions.len() {
                let swap = middle.transpositions[i];
                if swap.0 == swap.1 {
                    continue;
                }
                let middle_circuit = Self::gen_gates_swap_restricted(n, swap, env, dbs, &vec![13,14,15,29,30,31]);
                middle_gates.extend_from_slice(&middle_circuit);
            }
            rewire_gate_ver(&mut middle_gates, &t_rewired[j].1, n);
            let curr_len = middle_gates.len();
            middle_gates.insert(curr_len/2, c.gates[j/2]);
            gates.extend_from_slice(&middle_gates);
            j += 1;
        }
        
        let second = &t_rewired[t_rewired.len()-1];
        let mut second_gates: Vec<[u8;3]> = Vec::new();
        for i in 0..second.0.transpositions.len() {
            let mut swap = second.0.transpositions[i];
            if swap.0 == swap.1 {
                continue;
            }
            let n = second_bounds.1 - second_bounds.0 + 1;
            let offset = second_bounds.0 as u8;
            swap.0 -= offset;
            swap.1 -= offset;
            let second_circuit = Self::gen_gates_swap(n, swap, env, dbs);
            let second_circuit: Vec<[u8; 3]> = second_circuit
                .into_iter()
                .map(|[a, b, c]| [a + offset, b + offset, c + offset])
                .collect();
            
            second_gates.extend_from_slice(&second_circuit);
        }
        rewire_gate_ver(&mut second_gates, &t_rewired[t_rewired.len()-1].1, n);
        let curr_len = second_gates.len();
        let len = c.gates.len();
        second_gates.insert(curr_len/2, c.gates[len-1]);
        gates.extend_from_slice(&second_gates);
        CircuitSeq { gates }
    }

    pub fn restricted_to_circuit_rewired_and_insert_no_seams(
        t_rewired: Vec<(Transpositions, Permutation)>,
        c: CircuitSeq,
        n: usize,
        env: &lmdb::Environment,
        dbs: &HashMap<String, Database>,
        first_bounds: (usize, usize),
        second_bounds: (usize, usize), 
    ) -> CircuitSeq {
        let mut gates: Vec<[u8; 3]> = Vec::new();
        for i in 0..c.gates.len()-1 {
            gates.push(c.gates[i]);
            let i = 3*i;
            let first = &t_rewired[i].0;
            let mut first_gates: Vec<[u8;3]> = Vec::new();
            for i in 0..first.transpositions.len() {
                let mut swap = first.transpositions[i];
                if swap.0 == swap.1 {
                    continue;
                }
                let n = first_bounds.1 - first_bounds.0 + 1;
                let offset = first_bounds.0 as u8;
                swap.0 -= offset;
                swap.1 -= offset;
                let first_circuit = Self::gen_gates_swap(n, swap, env, dbs);
                let first_circuit: Vec<[u8; 3]> = first_circuit
                    .into_iter()
                    .map(|[a, b, c]| [a + offset, b + offset, c + offset])
                    .collect();

                first_gates.extend_from_slice(&first_circuit);
            }
            rewire_gate_ver(&mut first_gates, &t_rewired[i].1, n);
            gates.extend_from_slice(&first_gates);

            let middle = &t_rewired[i+1].0;
            let mut middle_gates: Vec<[u8;3]> = Vec::new();
            for i in (0..middle.transpositions.len()).rev() {
                let swap = middle.transpositions[i];
                if swap.0 == swap.1 {
                    continue;
                }

                let mut middle_circuit = Self::gen_gates_swap(n, swap, env, dbs);
                middle_circuit.reverse();
                middle_gates.extend_from_slice(&middle_circuit);
            }
            rewire_gate_ver(&mut middle_gates, &t_rewired[i+1].1, n);
            gates.extend_from_slice(&middle_gates);

            let second = &t_rewired[i+2].0;
            let mut second_gates: Vec<[u8;3]> = Vec::new();
            for i in 0..second.transpositions.len() {
                let mut swap = second.transpositions[i];
                if swap.0 == swap.1 {
                    continue;
                }
                let n = second_bounds.1 - second_bounds.0 + 1;
                let offset = second_bounds.0 as u8;
                swap.0 -= offset;
                swap.1 -= offset;
                let second_circuit = Self::gen_gates_swap(n, swap, env, dbs);
                let second_circuit: Vec<[u8; 3]> = second_circuit
                    .into_iter()
                    .map(|[a, b, c]| [a + offset, b + offset, c + offset])
                    .collect();

                second_gates.extend_from_slice(&second_circuit);
            }
            rewire_gate_ver(&mut second_gates, &t_rewired[i+2].1, n);
            gates.extend_from_slice(&second_gates);
        }
        gates.push(c.gates[c.gates.len()-1]);
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

pub fn insert_wire_shuffles_knuth(
    circuit: &mut CircuitSeq, 
    n: usize,
    env: &Environment,
    dbs: &HashMap<String, Database>,
) {
    println!("Inserting wire shuffles (knuth)");
    println!("Starting len: {} gates", circuit.gates.len());
    let mut t_list: Transpositions = Transpositions { transpositions: Vec::new() };
    let mut gates: Vec<[u8;3]> = Vec::new();
    let mut negation_mask = vec![0u8; n];

    for &gate in &circuit.gates {
        let t = Transpositions::gen_random_knuth(n, 150, &mut negation_mask);
        gates.extend_from_slice(&t.to_circuit(n, env, dbs).gates);
        t_list.transpositions.extend_from_slice(&t.transpositions);
        let a = t_list.evaluate(gate[0]);
        let b = t_list.evaluate(gate[1]);
        let c = t_list.evaluate(gate[2]);
        let gate = [a, b, c];
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

pub fn insert_wire_shuffles_simple(
    circuit: &mut CircuitSeq, 
    n: usize,
    env: &Environment,
    dbs: &HashMap<String, Database>,
) {
    println!("Inserting wire shuffles (simple)");
    println!("Starting len: {} gates", circuit.gates.len());
    let mut t_list: Transpositions = Transpositions { transpositions: Vec::new() };
    let mut gates: Vec<[u8;3]> = Vec::new();
    let mut negation_mask = vec![0u8; n];

    // Generate random points. m needed in k = m * n
    // Choose them spaced approximately evenly but with `sufficient` randomness
    let m = circuit.gates.len();
    let mut points = Vec::with_capacity(m);
    let mut rng = rand::rng();
    for i in 0..m {
        let center = i * n + n / 2;

        // allow significant variance but keep spacing structure
        let jitter = rng.random_range(-(n as i64)/2 ..= (n as i64)/2);

        let mut p = center as i64 + jitter;

        // avoid very beginning
        if p < n as i64 / 4 {
            p = n as i64 / 4;
        }

        points.push(p as usize);
    }
    let mut last = 0;
    for (i, gate) in circuit.gates.iter().enumerate() {
        let t = Transpositions::gen_random_simple(n, points[i] - last, &mut negation_mask);
        last = points[i];
        gates.extend_from_slice(&t.to_circuit(n, env, dbs).gates);
        t_list.transpositions.extend_from_slice(&t.transpositions);
        let a = t_list.evaluate(gate[0]);
        let b = t_list.evaluate(gate[1]);
        let c = t_list.evaluate(gate[2]);
        let gate = [a, b, c];
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

// Insert 2 shuffles are the beginning and end, and then an additional x number of shuffles
pub fn insert_wire_shuffles_x(
    circuit: &mut CircuitSeq, 
    n: usize,
    env: &Environment,
    dbs: &HashMap<String, Database>,
    x: usize,
) {
    println!("Inserting wire shuffles");
    println!("Starting len: {} gates", circuit.gates.len());
    let mut t_list: Transpositions = Transpositions { transpositions: Vec::new() };
    let mut gates: Vec<[u8;3]> = Vec::new();
    let mut negation_mask = vec![0u8; n];

    let start = 1;
    let end = circuit.gates.len() - 1;
    let range_size = end - start + 1;
    let mut rng = rand::rng();
    let sample = rand::seq::index::sample(&mut rng, range_size, x);

    let mut nums: Vec<usize> = sample.iter().map(|i| start + i).collect();

    nums.push(0);

    for (i, gate) in circuit.gates.iter().enumerate() {
        if nums.contains(&i) {
            let t = Transpositions::gen_random_knuth(n, 150, &mut negation_mask);
            gates.extend_from_slice(&t.to_circuit(n, env, dbs).gates);
            t_list.transpositions.extend_from_slice(&t.transpositions);
        }   
        let a = t_list.evaluate(gate[0]);
        let b = t_list.evaluate(gate[1]);
        let c = t_list.evaluate(gate[2]);
        let gate = [a, b, c];
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

// Generate a circuit R R*, but then insert a series of CNOTS in between so that the first n wires are reversible, but the last n wires can be used to compute R
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

pub fn replace_disjoint_pair((a,b, t1): (u8, u8, u8), (c,d, t2): (u8, u8, u8)) -> Vec<(u8, u8, u8)> {
    let possibilities = [
        vec![(a,c,0),(b,d,0),(a,d,0),(b,c,0)],
        vec![(b,c,0),(a,d,0),(b,d,0),(a,c,0)],
    ];

    let mut rng = rand::rng();
    let idx = rng.random_range(0..possibilities.len());
    
    let mut t = possibilities[idx].clone();

    let mut wire_transpositions: HashMap<u8, (usize, usize)> = HashMap::new();

    for (i, (a, b, _)) in t.iter().enumerate() {
        wire_transpositions.insert(*a, (i, 0));
        wire_transpositions.insert(*b, (i, 1));
    }

    const TRANSITION: [[u8; 4]; 2] = [
        // pos = 0
        [1, 0, 3, 2],
        // pos = 1
        [2, 3, 0, 1],
    ];

    let mut negation_mask: Vec<u8> = Vec::new();
    if t1 == 1 || t1 == 3 {
        negation_mask.push(a);
    }
    if t1 == 2 || t1 == 3 {
        negation_mask.push(b);
    }
    if t2 == 1 || t2 == 3 {
        negation_mask.push(c);
    }
    if t2 == 2 || t2 == 3 {
        negation_mask.push(d);
    }
    for val in negation_mask {
        if let Some(swaps) = wire_transpositions.get(&(val as u8)) {
            let &(swap_idx, pos) = swaps;
            let curr_neg_type = t[swap_idx].2;
            if pos > 1 || curr_neg_type > 3 {
                panic!("Invalid pos or curr_neg_type");
            }
            t[swap_idx].2 = TRANSITION[pos][curr_neg_type as usize];
            
        }
    }

    t
}

// Creates an identity with the first part limited to 16..=28 wires (exclude wires 29, 30, 31), the middle part spanning all 0..=31, and the last part spanning 0..=12 wires (exclude wires 13, 14, 15) 
// returns the identity, the number of transpositions of the first part, and the number of transpositions of the second part

pub fn create_ri_identities_32() -> (Transpositions, Transpositions, Transpositions, usize, usize) {
    let mut transpositions: Transpositions = Transpositions{ transpositions: Vec::new() };
    let mut first_negation_mask: Vec<u8> = vec![0u8; 32]; 
    let mut first = Transpositions::gen_random_simple(13, 50, &mut first_negation_mask);
    let mut second_negation_mask: Vec<u8> = vec![0u8; 32]; 
    let second = Transpositions::gen_random_simple(13, 50, &mut second_negation_mask);
    for i in 0..16 {
        let temp = first_negation_mask[i];
        first_negation_mask[i] = first_negation_mask[i + 16];
        first_negation_mask[i + 16] = temp;
    }

    for i in 0..50 {
        first.transpositions[i].0 += 16;
        first.transpositions[i].1 += 16;
        transpositions.transpositions.push(first.transpositions[i]);
        transpositions.transpositions.push(second.transpositions[i]);
    }

    for i in (0..50).rev() {
        let idx = 2 * i;

        let a = transpositions.transpositions[idx];
        let b = transpositions.transpositions[idx + 1];

        transpositions.transpositions.splice(idx..idx+2, replace_disjoint_pair(a, b));
    }

    (first, transpositions, second, 50, 50)
}

// Only supports 32 wires for now
pub fn insert_ri_identities(c: &mut CircuitSeq, env: &Environment, dbs: &HashMap<String, Database>) {
    let mut t_rewired: Vec<(Transpositions, Permutation)> = Vec::new();
    let mut rng = rand::rng();
    let len = c.gates.len();
    let mut used_wires:[u8;3] = [c.gates[0][0], c.gates[0][1], c.gates[0][2]];
    // Create and rewire all the RI identities
    for i in 1..len {
        let (first, middle, second, _, _) = create_ri_identities_32();
        let mut wire_shuffle1 = Permutation { data: (0..32).collect() };
        for idx in 16..32 {
            wire_shuffle1.data[idx] = 33;
        }
        let mut excluded = vec![29,30,31];
        excluded.shuffle(&mut rng);

        let mut used_targets = Vec::new();
        let mut count = 0;

        for &val in &used_wires {
            if val >= 16 {
                wire_shuffle1.data[val as usize] = excluded[count];
                used_targets.push(excluded[count]);
                count += 1;
            }
        }

        // remaining wires only in upper half
        let mut remaining: Vec<usize> = (16..32)
            .filter(|w| !used_targets.contains(w))
            .collect();

        remaining.shuffle(&mut rng);

        let mut idx = 0;
        for i in 16..32 {
            if wire_shuffle1.data[i] == 33 {
                wire_shuffle1.data[i] = remaining[idx];
                idx += 1;
            }
        }
        t_rewired.push((first.clone(), wire_shuffle1.clone()));

        used_wires = [c.gates[i][0], c.gates[i][1], c.gates[i][2]];
        let mut wire_shuffle2 = Permutation { data: (0..32).collect() };
        for idx in 0..16 {
            wire_shuffle2.data[idx] = 33;
        }
        let mut excluded = vec![13,14,15];
        excluded.shuffle(&mut rng);

        let mut used_targets = Vec::new();
        let mut count = 0;

        for &val in &used_wires {
            if val < 16 {
                wire_shuffle2.data[val as usize] = excluded[count];
                used_targets.push(excluded[count]);
                count += 1;
            }
        }

        // remaining wires only in lower half
        let mut remaining: Vec<usize> = (0..16)
            .filter(|w| !used_targets.contains(w))
            .collect();

        remaining.shuffle(&mut rng);

        let mut idx = 0;
        for i in 0..16 {
            if wire_shuffle2.data[i] == 33 {
                wire_shuffle2.data[i] = remaining[idx];
                idx += 1;
            }
        }
        let mut wire_shufflem = Permutation { data: Vec::with_capacity(32)};
        wire_shufflem.data.extend_from_slice(&wire_shuffle2.data[..16]);
        wire_shufflem.data.extend_from_slice(&wire_shuffle1.data[16..]);
        t_rewired.push((middle, wire_shufflem));
        t_rewired.push((second, wire_shuffle2));
    }
    
    // Combine the RI identities and seam them together
    let num_identities = t_rewired.len()/3;
    for i in (0..num_identities-1).rev() {
        let idx = 2 + 3*i;
        let (t1, p1) = t_rewired[idx].clone();
        let (t2, p2) = t_rewired[idx + 1].clone();

        let mut combined = Transpositions { transpositions: Vec::new() };
        for j in (0..50).rev() {
            let a = t1.transpositions[j];
            let b = t2.transpositions[j];
            let mut r = replace_disjoint_pair(a, b);
            r.extend(combined.transpositions);
            combined.transpositions = r;
        }

        let mut new_perm = Vec::with_capacity(32);
        new_perm.extend_from_slice(&p1.data[..16]);
        new_perm.extend_from_slice(&p2.data[16..]);
        let new_perm = Permutation{ data: new_perm };
        let mut sanity = combined.restricted_to_circuit(32, &env, &dbs, &vec![13,14,15,29,30,31]);
        sanity.rewire(&new_perm, 32);
        for j in 0..3 {
            if sanity.gates.iter().any(|g| g.contains(&c.gates[i+1][j])) {
                println!("{:?}", sanity);
                println!("{}: {:?}", i+1, c.gates[i+1]);
                panic!("Not a snug fit");
            }
        }
        let combined = (t1, new_perm);
        t_rewired.splice(idx..idx+2, [combined]);
    }

    *c = Transpositions::restricted_to_circuit_rewired_and_insert(t_rewired, c.clone(), 32, &env, &dbs, (16, 28), (0, 12));
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
    use crate::{CircuitSeq, replace::transpositions::insert_ri_identities};
    use crate::replace::transpositions::Transpositions;
    #[test]
    fn test_wire_shifting() {
        use crate::replace::main_mix::open_all_dbs;
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

            let t = Transpositions::gen_random_knuth(64, 100, &mut Vec::new());
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
        use crate::replace::main_mix::open_all_dbs;
        let file = File::open("initial.txt").expect("failed to open initial.txt");
        let reader = BufReader::new(file);

        // let circuits: Vec<String> = reader
        //     .lines()
        //     .map(|l| l.unwrap())
        //     .filter(|l| !l.trim().is_empty())
        //     .collect();

        // let mut rng = rand::rng();
        // let _circuit_str = circuits
        //     .choose(&mut rng)
        //     .expect("no circuits found");

        // let base = CircuitSeq::from_string(circuit_str);

        let env = Environment::new()
            .set_max_dbs(262)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new("./db"))
            .expect("failed to open lmdb");

        let dbs = open_all_dbs(&env);

        let mut t = Transpositions::gen_random_knuth(128, 500, &mut vec![0u8; 128]);
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
        use crate::replace::main_mix::open_all_dbs;
        use std::io::Write;
        use crate::replace::transpositions::insert_wire_shuffles_x;
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
        insert_wire_shuffles_x(&mut new_circuit, 64, &env, &dbs, 50);

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
        use crate::replace::main_mix::open_all_dbs;
        use crate::replace::transpositions::HashMap;
        let env = Environment::new()
            .set_max_dbs(262)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new("./db"))
            .expect("failed to open lmdb");

        let dbs = open_all_dbs(&env);
        let n = 64;
        let mut gates: Vec<[u8;3]> = Vec::new();
        let mut negation_mask = vec![0u8; n];
        let t = Transpositions::gen_random_knuth(n, 150, &mut negation_mask);
        gates.extend_from_slice(&t.to_circuit(n, &env, &dbs).gates);
        let p = t.to_perm(n);
        let t = Transpositions::from_perm(&p);
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
        use crate::replace::main_mix::open_all_dbs;
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
        let t = Transpositions::gen_random_knuth(64, 100, &mut Vec::new());
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

    #[test]
    fn test_reversible() {
        use std::fs;
        use rand::Rng;
        let circuit_str = fs::read_to_string("initial.txt")
            .expect("failed to read initial.txt");
        let circuit = CircuitSeq::from_string(&circuit_str);

        let mut c100 = circuit.clone();
        c100.gates.truncate(100);

        let mut rng = rand::rng();
        let rand64: u128 = rng.random::<u64>() as u128;
        let input: u128 = rand64; 

        let out_full = circuit.evaluate_128(input);
        let out_100 = c100.evaluate_128(input);

        let low_mask: u128 = (1u128 << 64) - 1;
        let high_mask: u128 = !low_mask;

        assert_eq!(
            out_full & low_mask,
            input & low_mask,
            "first 64 bits changed"
        );

        assert_eq!(
            out_full & high_mask,
            (out_100 & low_mask) << 64,
            "last 64 bits differ from c100 result"
        );

        let mut rev = circuit.clone();
        rev.gates.reverse();
        let out_full_rev = rev.evaluate_128(input);
        assert_eq!(
            out_full,
            out_full_rev,
            "the circuit isn't reversible"
        );
    }

    #[test]
    fn test_ri_32() {
        use crate::replace::transpositions::create_ri_identities_32;
        use crate::replace::main_mix::open_all_dbs;
        use std::io::Write;
        use rand::seq::SliceRandom;
        use crate::replace::transpositions::Permutation;
        let (first, middle, second, _, _) = create_ri_identities_32();
        let env = Environment::new()
            .set_max_dbs(262)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new("./db"))
            .expect("failed to open lmdb");

        let dbs = open_all_dbs(&env);

        let mut file = File::create("test_id.txt").expect("Failed to create file");

        let f = first.to_circuit(32, &env, &dbs);
        let mut m = middle.to_circuit(32, &env, &dbs);
        m.gates.reverse();
        let s = second.to_circuit(32, &env, &dbs);

        let mut c = CircuitSeq {gates: Vec::new() };
        c.gates.extend_from_slice(&f.gates);
        c.gates.extend_from_slice(&m.gates);
        c.gates.extend_from_slice(&s.gates);
        let id = CircuitSeq {gates: Vec::new() };
        if c.probably_equal(&id, 32, 1000).is_err() {
            panic!("Not an id");
        }
        let mut id = first.restricted_to_circuit(32, &env, &dbs, &vec![]);
        let stupid_id = CircuitSeq { gates: Vec::new() };
        if id.probably_equal(&stupid_id, 32, 1000).is_err() {
            panic!("Not an id identity");
        }
        let repr = id.repr();
        let mut shuffle: Permutation = Permutation { data: (0..32).collect() };
        shuffle.data.shuffle(&mut rand::rng());
        id.rewire(&shuffle, 32);
        if id.probably_equal(&stupid_id, 32, 1000).is_err() {
            panic!("Shuffling destroyed identity");
        }
        let repr = id.repr();
        writeln!(file, "{}", repr)
            .expect("Failed to write to file");

        println!("Wrote test circuit to file");
    }

    #[test]
    fn test_insert_ri_identities() {
        use crate::replace::main_mix::open_all_dbs;
        use std::io::Write;
        let env = Environment::new()
            .set_max_dbs(262)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new("./db"))
            .expect("failed to open lmdb");
        let dbs = open_all_dbs(&env);

        let mut file = File::create("test_id.txt").expect("Failed to create file");

        let mut c_old = CircuitSeq::from_string("vnt;otv;k3c;g8d;hkm;fn8;3p0;v92;0id;l4a;pq0;sn3;06k;roh;cld;pef;s3j;dh7;jum;l41;gio;1pf;rge;ont;3qa;731;3rg;2eg;2sl;ebg;ovf;opk;tel;hts;cql;06h;u9i;gov;lbc;04i;0as;kp9;iro;e38;bc8;0ue;hst;p9i;gom;908;0do;l5s;t9g;abd;7rs;0hk;fq9;o49;14l;7j0;vu6;clf;4mn;9g6;4vc;lkp;p73;4mi;h9k;7rg;d4a;674;73f;ojr;fpj;gct;94k;nab;3is;q2h;dvp;huv;bsp;lb7;vr2;nd7;ud3;9bv;ljg;q1e;av9;8du;3hl;cd1;mir;ris;uoc;btq;ibc;bds;");
        let mut c = CircuitSeq::from_string("vnt;otv;k3c;g8d;hkm;fn8;3p0;v92;0id;l4a;pq0;sn3;06k;roh;cld;pef;s3j;dh7;jum;l41;gio;1pf;rge;ont;3qa;731;3rg;2eg;2sl;ebg;ovf;opk;tel;hts;cql;06h;u9i;gov;lbc;04i;0as;kp9;iro;e38;bc8;0ue;hst;p9i;gom;908;0do;l5s;t9g;abd;7rs;0hk;fq9;o49;14l;7j0;vu6;clf;4mn;9g6;4vc;lkp;p73;4mi;h9k;7rg;d4a;674;73f;ojr;fpj;gct;94k;nab;3is;q2h;dvp;huv;bsp;lb7;vr2;nd7;ud3;9bv;ljg;q1e;av9;8du;3hl;cd1;mir;ris;uoc;btq;ibc;bds;");
        insert_ri_identities(&mut c, &env, &dbs);

        writeln!(file, "{}", c.repr())
            .expect("Failed to write to file");

        if c.probably_equal(&c_old, 32, 1000).is_err() {
            panic!("Changed functionality somewhere");
        }
    }
}
