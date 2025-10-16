//Basic implementation for circuit, gate, and permutations
use crate::circuit::control_functions::GateControlFunc;
use crate::rainbow::database::PersistPermStore;

// use crossbeam::channel::{unbounded, Receiver};
use rand::{seq::SliceRandom, Rng};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::{
    cmp::max as std_max,
    collections::{HashSet, HashMap},
    sync::Arc,
};
use rayon::prelude::*;
use smallvec::SmallVec;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Gate{
    pub pins: [usize;3], //one active wire (0) and two control wires (1,2)
    pub control_function: u8,
    pub id: usize
}

#[derive(Clone, Debug, Default)]
pub struct Circuit{
    pub num_wires: usize,
    pub gates: Vec<Gate>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CircuitSeq {
    pub gates: Vec<[u8;3]>, //TODO: Change to Vec<[u8;3]>
}

//Permutations are all the possible outputs of a circuit
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Permutation {
    pub data: Vec<usize>,
}

impl Gate {
    pub fn new(active: usize, first_control: usize, second_control: usize, control_function: u8, id:usize) -> Self {
        Self {
            pins: [active, first_control, second_control],
            control_function,
            id,
        }
    }
    
    //currenlty doesn't use n
    pub fn repr(&self, _n: usize) -> String {
        // Convert the gate id to a char if valid, otherwise fallback
        std::char::from_u32(self.id as u32)
            .map(|c| c.to_string())
            .unwrap_or_else(|| format!("{}", self.id))
    }

    //two gates collide if an active and a control pins are on the same wire
    pub fn collides(&self, other_gate: &Self) -> bool {
        self.pins[0] == other_gate.pins[1] 
            || self.pins[0] == other_gate.pins[2]
            || self.pins[1] == other_gate.pins[0] 
            || self.pins[2] == other_gate.pins[0]
    }

    //to evaluate a gate, we use the control function table we built in constants.rs
    // #[inline]
    // pub fn evaluate_gate(&self, state: &mut usize) {
    //     let index = ((self.control_function as usize) << 2)
    //                 | (((*state >> self.pins[1]) & 1) << 1)
    //                 | (((*state >> self.pins[2])) & 1);
    //     *state ^= (CONTROL_FUNC_TABLE[index] as usize) << self.pins[0];
    //}

    //only consider r57
    #[inline]
    pub fn evaluate_gate(&self, state: &mut usize) -> usize {
        let c1 = (*state >> self.pins[1]) & 1;
        let c2 = (*state >> self.pins[2]) & 1;
        *state ^= (c1 | ((!c2) & 1)) << self.pins[0];
        *state
    }

    pub fn equal(&self, other: &Self) -> bool {
        self.pins == other.pins && self.control_function == other.control_function
    }

    pub fn evaluate_gate_list(gate_list: &Vec<Gate>, input_wires: usize) -> usize {
    let mut current_wires = input_wires;

    for gate in gate_list {
        gate.evaluate_gate(&mut current_wires);
    }

    current_wires
}

    //give ordering to gates for later canonicalization
    pub fn ordered(&self, other: &Self) -> bool {
        if self.pins[0] > other.pins[0] {
            return false
        }
        else if self.pins[0] == other.pins[0]{
            if self.pins[1] > other.pins[1] {
                return false
            }
            else if self.pins[1] == other.pins[1] {
                return self.pins[2] < other.pins[2]
            }
        }
        true
    }

    pub fn bottom(&self) -> usize {
        // println!("bottom is {}", std_max((std_max(self.pins[0], self.pins[1])), self.pins[2]));
        std_max(std_max(self.pins[0], self.pins[1]), self.pins[2])
    }

    pub fn collides_index(gate: &[u8;3], other: &[u8;3]) -> bool {
        gate[0] == other[1] 
            || gate[0] == other[2]
            || gate[1] == other[0] 
            || gate[2] == other[0]
    }
    //b is "larger"
    pub fn ordered_index(gate: &[u8;3], other: &[u8;3]) -> bool {
        if gate[0] > other[0] {
            return false
        }
        else if gate[0] == other[0]{
            if gate[1] > other[1] {
                return false
            }
            else if gate[1] == other[1] {
                return gate[2] < other[2]
            }
        }
        true
    }

    #[inline(always)]
    pub fn evaluate_index(state: usize, gate: [u8;3]) -> usize {
        let c1 = (state >> gate[1]) & 1;
        let c2 = (state >> gate[2]) & 1;
        state ^ (c1 | ((!c2) & 1)) << gate[0]
    }

    #[inline(always)]
    pub fn evaluate_index_list(state: usize, gates: &Vec<[u8;3]>) -> usize {
        let mut current_wires = state;
        for g in gates {
            current_wires = Self::evaluate_index(current_wires, *g);
        }
        current_wires
    }
}

impl Circuit{
    pub fn new(num_wires:usize, gates: Vec<Gate>) -> Self {
        Self{num_wires, gates}
    }

    pub fn random_circuit(num_wires:usize, num_gates: usize) -> Circuit {
        let base_gates = base_gates(num_wires); //Vec<[usize;3]>
        let mut gates = Vec::with_capacity(num_gates);
        let mut last = usize::MAX;

        let mut rng = rand::rng();
        for _ in 0..num_gates {
            let mut j = last;
            while j == last{
                j = rng.random_range(0..base_gates.len());
            }
            let g = base_gates[j];
            gates.push(Gate { pins: g, control_function: 2, id: j});
            last = j;
        }
        Self::new(num_wires, gates)
    } 
    // pub fn random_circuit<R: Rng>(
    //     num_wires: usize,
    //     num_gates: usize,
    //     rng: &mut R
    // ) -> Self {
    //     let mut gates = vec![];
    //     for _ in 0..num_gates {
    //         loop{
    //             let active = rng.random_range(0..num_wires);
    //             let first_control = rng.random_range(0..num_wires);
    //             let second_control = rng.random_range(0..num_wires);

    //             if active != first_control && active != second_control && first_control != second_control {
    //                 gates.push(Gate {
    //                     pins: [active, first_control, second_control],
    //                     //control_function: (rng.random_range(0..16) as u8), any control
    //                     control_function: 2, //r57
    //                 });
    //                 break;
    //             }
    //         }
    //     }
    //     Self{num_wires, gates}
    // }
    pub fn to_string(&self) -> String{
        let mut result = String::new();
        for wire in 0..self.num_wires {
            result += &(wire.to_string() + "  --");
            for gate in &self.gates {
                if gate.pins[0] == wire {
                    result+="( )";
                } else if gate.pins[1] == wire { //a
                    result+="-●-";
                } else if gate.pins[2] == wire { //b
                    result+="-○-";
                } else {
                    result+="-|-";
                }
                result.push_str("---");
            }
            result.push_str("\n");
            }
           
        let control_fn_strings: Vec<String> = self.gates
            .iter()
            .map(|gate| GateControlFunc::from_u8(gate.control_function).to_string())
            .collect();
        result.push_str("\ncfs: ");
        result.push_str(&control_fn_strings.join(", "));
        result
    }


    pub fn probably_equal(&self, other_circuit: &Self, num_inputs: usize) -> Result<(), String> {
        if self.num_wires != other_circuit.num_wires {
            return Err("The circuits do not have the same number of wires".to_string());
        }

        let mut rng = rand::rng();
        let mask = (1 << self.num_wires) - 1;

        for _ in 0..num_inputs {
            // generate u64, then mask to get the lower num_wires bits
            let random_input = (rng.random::<u64>() as usize) & mask;

            let self_output = Gate::evaluate_gate_list(&self.gates, random_input);
            let other_output = Gate::evaluate_gate_list(&other_circuit.gates,random_input);

            if self_output != other_output {
                return Err("Circuits are not equal".to_string());
            }
        }

        Ok(())
    }

    // CAN TWO CIRCUITS WITH DIFFERENT NUMBER OF WIRES BE FUNCTIONALLY EQUIVALENT?????
    // pub fn functionally_equal()(&self, other_circuit: &Self, num_inputs: usize) -> Result<(), String> {
    //     let least_num_wires = min(self.num_wires, other_circuit.num_wires);
    //     if num_inputs > 
    // }

    pub fn evaluate(&self, input_wires: usize) -> usize {
        Gate::evaluate_gate_list(&self.gates, input_wires)
    }

    pub fn permutation(&self) -> Permutation {
        let size = 1 << self.num_wires;
        let mut output = vec![0; size]; // initialize with zeros

        for i in 0..size {
            output[i] = self.evaluate(i);
        }

        Permutation { data: output }
    }

    pub fn from_gates(gates: &Vec<Gate>) -> Circuit {
        let mut w = 0;
        for g in gates {
            w = std_max(w, g.bottom());
        }
        Circuit { num_wires: w+1, gates: gates.to_vec(), }
    }

    pub fn adjacent_id(&self) -> bool {
        for i in 0..(self.gates.len()-1) {
            if self.gates[i] == self.gates[i+1] {
                return true
            }
        }
        false
    }

    pub fn repr(&self) -> String {
        let mut sb = String::new();
        for g in &self.gates {
            sb.push_str(&g.repr(self.num_wires));
        }
        sb
    }

    //converts from strings of the form " a b c; a' b' c'; ...; a'' b'' c'' "
    pub fn from_string(str: String) -> Circuit {
        let mut gates = Vec::<Gate>::new();
        for gate_slice in str.split(';') {
            if gate_slice.trim().is_empty() {
                continue;
            }
            
            let mut pins = [0usize;3]; 
            
            for (i, wire) in gate_slice.split_whitespace().enumerate() {
                // println!("i: {}, wire: {}", i, wire);
                if i > 2 {
                    panic!("Expected 3 pins per gate");
                }
                pins[i] = wire.parse()
                    .expect(&format!("Failed to parse pin {} in gate '{}'", i, gate_slice));
            }
            // println!("pins: {:?}", pins);
            let gate = Gate {pins, control_function: 2, id: 0};

            gates.push(gate);
        }
        Self::from_gates(&gates)
    }

    pub fn len(&self) -> usize {
        self.gates.len()
    }

    pub fn used_wires(&self) -> Vec<usize> {
        let mut used: HashSet<usize> = HashSet::new();
        for gates in &self.gates {
            used.insert(gates.pins[0]);
            used.insert(gates.pins[1]);
            used.insert(gates.pins[2]);
        }
        used.into_iter().collect()
    }

    pub fn count_used_wires(&self) -> usize {
        self.used_wires().len()
    }

    pub fn minimize_wires(&self) -> Circuit {
        // Collect and sort used wire indices
        let mut used = self.used_wires();
        used.sort_unstable();
        // Build mapping: old wire index -> new wire index
        let mut wire_map = HashMap::new();
        for (new_index, old_index) in used.iter().enumerate() {
            wire_map.insert(*old_index, new_index);
        }
        // Remap all gates
        let new_gates: Vec<Gate> = self.gates
            .iter()
            .map(|g| {
                let pins = [
                    *wire_map.get(&g.pins[0]).unwrap(),
                    *wire_map.get(&g.pins[1]).unwrap(),
                    *wire_map.get(&g.pins[2]).unwrap(),
                ];
                Gate {
                    pins,
                    control_function: g.control_function, // keep original. For now, this will always be r57
                    id: g.id,
                }
            })
            .collect();
        Self::from_gates(&new_gates)
    }

    //converts from a compressed string (gates are represented by chars)
    pub fn from_bytes_compressed(n: usize, bytes: &Vec<u8>) -> Circuit {
        let base_gates = base_gates(n);
        let mut gates = Vec::with_capacity(bytes.len());

        for &gi in bytes {
            let pins = &base_gates[gi as usize];
            gates.push(Gate {
                pins: *pins,
                control_function: 2,
                id: gi as usize,
            });
        }

        Circuit { num_wires: n, gates }
    }
}

impl Permutation {
    pub fn new(data: Vec<usize>) -> Permutation {
        Permutation {
            data,
        }
    }
    pub fn is_perm(&self) -> bool {
        let mut temp_perm = self.clone();
        temp_perm.data.sort_unstable();
        temp_perm == Permutation::id_perm(self.data.len())
    }

    pub fn id_perm(n:usize) -> Permutation {
        let temp_data = (0..n).collect();
        Permutation { 
            data: temp_data, 
        }
    }

    pub fn rand_perm(n:usize) -> Permutation {
        let mut p = Permutation::id_perm(n);
        let mut rng = rand::rng();
        p.data.shuffle(&mut rng);
        p
    }

    pub fn invert(&self) -> Permutation {
        let mut inv = vec![0; self.data.len()];
        self.data.iter().enumerate().for_each(|(i, &val)| inv[val] = i);
        Permutation { 
            data: inv, 
        }
    }

    pub fn compose(&self, other: &Permutation) -> Permutation {
        if self.data.len() != other.data.len() {
            panic!("Permutation length mismatch in compose");
        }

        let data = self.data
            .iter()
            .enumerate()
            .map(|(i, &_x)| self.data[other.data[i]])
            .collect();

        Permutation { data }
    }

    pub fn repr(&self) -> String {
        self.data.iter()
            .map(|&x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")
    }

    pub fn repr_blob(&self) -> Vec<u8> {
        self.data.iter().map(|&x| x as u8).collect()
    }

    pub fn from_blob(blob: &[u8]) -> Self {
        let data = blob.iter().map(|&b| b as usize).collect();
        Permutation { data }
    }

    pub fn bits(&self) -> usize {
        let n = self.data.len();
        ((n - 1) as usize).ilog2() as usize + 1
    }

    pub fn to_string(&self) -> String {
        const MAX_LEN: isize = -1;

        // Format the inner vector as a string
        let s = format!("{:?}", self.data);

        //In case we deal with very long permutations
        if MAX_LEN > 0 && (s.len() as isize) > MAX_LEN {
            // Truncate and append "...]"
            let end = (MAX_LEN - 5) as usize;
            let mut truncated = s[..end].to_string();
            truncated.push_str("...]");
            truncated
        } else {
            s
        }
    }

    pub fn to_cycle(&self) -> Vec<Vec<usize>> {
        let n = self.data.len();
        let mut visited = vec![false; n];
        let mut cycles = Vec::new();

        for i in 0..n {
            if visited[i] {
                continue;
            }
            let mut j = self.data[i];
            visited[i] = true;

            // Skip fixed points
            if i == j {
                continue;
            }

            let mut c = vec![i];
            loop {
                visited[j] = true;
                c.push(j);
                j = self.data[j];
                if j == c[0] {
                    break;
                }
            }
            cycles.push(c);
        }

        cycles
    }

    pub fn bit_shuffle(&self, shuf: &Vec<usize>) -> Permutation {
        let n = self.data.len();
        let mut q_raw = vec![0; n];
        let mut idx = vec![0; n];

        for (s, &d) in shuf.iter().enumerate() {
            for i in 0..n {
                q_raw[i] |= ((self.data[i] >> s) & 1) << d;
                idx[i] |= ((i >> s) & 1) << d;
            }
        }

        let mut q = vec![0; n];
        for i in 0..n {
            q[idx[i]] = q_raw[i];
        }

        Permutation { data: q }
    }
}

impl CircuitSeq {
    pub fn adjacent_id(&self) -> bool {
        for i in 0..(self.gates.len()-1) {
            if self.gates[i] == self.gates[i+1] {
                return true
            }
        }
        false
    }

    pub fn evaluate(&self, input: usize) -> usize {
        Gate::evaluate_index_list(input, &self.gates)
    }

    pub fn permutation(&self, num_wires: usize) -> Permutation {
        let size = 1 << num_wires;
        //TODO: the size could be over 64, so is small vec a bad choice?
        let mut output = vec![0; size];

        for input in 0..size {
            output[input] = self.evaluate(input);
        }

        Permutation { data: output }
    }

    pub fn repr_blob(&self) -> Vec<u8> {
        let mut blob = Vec::with_capacity(self.gates.len() * 3);
        for &gate in &self.gates {
            blob.push(gate[0] as u8);
            blob.push(gate[1] as u8);
            blob.push(gate[2] as u8);
        }
        blob
    }

    pub fn to_circuit(&self) -> Circuit {
        let mut gates = Vec::new();
        let mut max: usize = 0;
        for g in &self.gates {
            gates.push(Gate{ pins: [g[0] as usize, g[1] as usize, g[2] as usize], control_function: 2, id: 0 });
            for &p in g {
                if p as usize > max {
                    max = p as usize;
                }
            }
        }
        Circuit{ num_wires: max, gates, }
    }

    // pub fn num_wires(&self) -> usize {
    //     let mut max = 0;
    //     for g in &self.gates {
    //         for &p in g {
    //             if p as usize > max {
    //                 max = p as usize;
    //             }
    //         }
    //     }
    //     max + 1
    // }

    /// Reconstruct CircuitSeq from a BLOB
    pub fn from_blob(blob: &[u8]) -> Self {
        assert!(blob.len() % 3 == 0, "Invalid blob length");
        let gates: Vec<[u8; 3]> = blob
            .chunks(3)
            .map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect();
        CircuitSeq { gates }
    }

    // wire i -> perm[i]
    pub fn rewire(&mut self, perm: &Permutation, n: usize) {
        if perm.data.is_empty() {
            return;
        }

        if perm.data.len() != n {
            panic!(
                "wrong size perm! got {}, have {} wires",
                perm.data.len(),
                n
            );
        }

        if !perm.is_perm() {
            panic!("{:?} is not a permutation!", perm);
        }

        for gate in &mut self.gates {
            *gate = [
                perm.data[gate[0] as usize] as u8,
                perm.data[gate[1] as usize] as u8,
                perm.data[gate[2] as usize] as u8,
            ];
        }
    }

    // Rewires the first gate to match `gate`, and adjusts remaining wires to a valid permutation
    pub fn rewire_first_gate(&mut self, target_gate: [u8; 3], num_wires: usize) {
        if self.gates.is_empty() {
            return
        }

        let first_gate = self.gates[0];

        // use usize::MAX to mark unused slots
        let mut perm: Vec<usize> = vec![usize::MAX; num_wires];

        // Map first gate wires -> target gate wires
        perm[first_gate[0] as usize] = target_gate[0] as usize;
        perm[first_gate[1] as usize] = target_gate[1] as usize;
        perm[first_gate[2] as usize] = target_gate[2] as usize;

        // Fill in remaining wires sequentially
        let mut next_free = 0;
        for slot in perm.iter_mut() {
            if *slot != usize::MAX {
                continue;
            }
            while next_free == target_gate[0] as usize
                || next_free == target_gate[1] as usize
                || next_free == target_gate[2] as usize
            {
                next_free += 1;
            }
            *slot = next_free;
            next_free += 1;
        }

        self.rewire(&Permutation { data: perm }, num_wires);
    }

    pub fn repr(&self) -> String {
        fn wire_to_char(w: u8) -> char {
            match w {
                0..=9 => (b'0' + w) as char,
                10..=35 => (b'a' + (w - 10)) as char,
                36..=61 => (b'A' + (w - 36)) as char,
                _ => panic!("Invalid wire index: {}", w),
            }
        }

        let mut s = String::new();
        for gate in &self.gates {
            for &wire in gate {
                s.push(wire_to_char(wire));
            }
            s.push(';');
        }
        s
    }

    pub fn from_string(s: &str) -> Self {
        fn char_to_wire(c: char) -> u8 {
            match c {
                '0'..='9' => c as u8 - b'0',
                'a'..='z' => c as u8 - b'a' + 10,
                'A'..='Z' => c as u8 - b'A' + 36,
                _ => panic!("Invalid wire char: {}", c),
            }
        }

        let gates: Vec<[u8; 3]> = s
            .trim()
            .split(';')
            .filter(|part| !part.is_empty())
            .map(|gate_str| {
                let chars = gate_str.chars().map(char_to_wire).collect::<Vec<_>>();
                if chars.len() != 3 {
                    panic!("Each gate must have exactly 3 wires: {:?}", gate_str);
                }
                [chars[0], chars[1], chars[2]]
            })
            .collect();

        CircuitSeq { gates }
    }

    pub fn to_string(&self, num_wires: usize) -> String {
        let mut result = String::new();

        // Local character map (0-9, a-z, A-Z)
        let wire_map_chars: Vec<char> = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            .chars()
            .collect();

        // --- Pretty circuit diagram ---
        for wire in 0..num_wires {
            result += &format!("{:<2} --", wire);
            for gate in &self.gates {
                if gate[0] == wire as u8 {
                    result += "( )";
                } else if gate[1] == wire as u8{
                    result += "-●-";
                } else if gate[2] == wire as u8 {
                    result += "-○-";
                } else {
                    result += "-|-";
                }
                result.push_str("---");
            }
            result.push('\n');
        }

        // Compact circuit string (like "123;124;213;")
        let compact: String = self
            .gates
            .iter()
            .map(|g| {
                g.iter()
                    .map(|&x| {
                        wire_map_chars
                            .get(x as usize)
                            .unwrap_or(&'?')
                            .to_string()
                    })
                    .collect::<String>()
                    + ";"
            })
            .collect();

        result.push_str("\n");
        result.push_str(&compact);

        result
    }

    pub fn concat(&self, other: &CircuitSeq) -> CircuitSeq {
        let mut gates = self.gates.clone();
        gates.extend_from_slice(&other.gates);
        CircuitSeq { gates }
    }

    pub fn used_wires(&self) -> Vec<u8> {
        let mut used: HashSet<u8> = HashSet::new();
        for gates in &self.gates {
            used.insert(gates[0]);
            used.insert(gates[1]);
            used.insert(gates[2]);
        }
        let mut wires: Vec<u8> = used.into_iter().collect();
        wires.sort();
        wires
    }

    pub fn count_used_wires(&self) -> usize {
        Self::used_wires(&self).len()
    }

    // Take subcircuit on X wires and rewire to x wires
    pub fn rewire_subcircuit(
        circuit: &CircuitSeq,
        subcircuit_gates: &[usize],
        used_wires: &[u8],
    ) -> CircuitSeq {
        // Build a mapping from old wire -> new wire (0..num_wires-1)
        let wire_map: HashMap<u8, u8> = used_wires
            .iter()
            .enumerate()
            .map(|(new_idx, &old_wire)| (old_wire, new_idx as u8))
            .collect();

        // Build new gates with remapped wires
        let new_gates: Vec<[u8; 3]> = subcircuit_gates
            .iter()
            .map(|&idx| {
                let [t, c1, c2] = circuit.gates[idx];
                [
                    *wire_map.get(&t).unwrap(),
                    *wire_map.get(&c1).unwrap(),
                    *wire_map.get(&c2).unwrap(),
                ]
            })
            .collect();

        CircuitSeq { gates: new_gates }
    }

    // Undo rewiring. Note: Recall that the number of wires in CircuitSeq is not stored
    pub fn unrewire_subcircuit(subcircuit: &CircuitSeq, used_wires: &[u8]) -> CircuitSeq {
        // Build a mapping from new wire -> original wire
        let wire_map: HashMap<u8, u8> = used_wires
            .iter()
            .enumerate()
            .map(|(new_idx, &orig_wire)| (new_idx as u8, orig_wire))
            .collect();

        // Replace wires in each gate with original wires
        let new_gates: Vec<[u8; 3]> = subcircuit
            .gates
            .iter()
            .map(|&[t, c1, c2]| [
                *wire_map.get(&t).unwrap(),
                *wire_map.get(&c1).unwrap(),
                *wire_map.get(&c2).unwrap(),
            ])
            .collect();

        CircuitSeq { gates: new_gates }
    }

    pub fn evaluate_evolution(&self, input: usize) -> Vec<usize> {
        let mut state = input;
        let mut evolution = vec![state];

        for gate in &self.gates {
            state = Gate::evaluate_index(state, *gate);
            evolution.push(state);
        }

        evolution
    }

}

pub fn base_gates(n: usize) -> Vec<[usize; 3]> {
    let mut gates = Vec::new();
    for a in 0..n {
        for b in 0..n {
            if b == a { continue; }
            for c in 0..n {
                if c == a || c == b { continue; }
                gates.push([a, b, c]);
            }
        }
    }
    gates
}

//threads work independently, so use Rayon instead of Receiver
//This returns an iterator for Vec<usize>, which represents a circuit.
//[0,2,1] would correspond to a circuit with 3 gates, index 0, 2, and 1, in base_gates(3)
pub fn par_all_circuits(wires: usize, gates: usize) -> impl ParallelIterator<Item = Vec<usize>>{
    //keep as usize for now
    let base_gates = base_gates(wires);
    let z = base_gates.len();
    let total = z.pow(gates as u32);

    //TODO: (J: chunk total. don't spawn millions of threads)
    (0..total).into_par_iter().filter_map(move |mut i| {
        // use a stack array to avoid heap allocations
        // more efficient for filter_map
        let mut stack_buf = [0usize; 64]; // max 64 gates
        assert!(gates <= stack_buf.len(), "Too many gates for stack buffer");

        let mut skip = false;

        for j in 0..gates {
            //TODO: These are very expensive
            stack_buf[j] = i % z;
            i /= z;

            if j > 0 && stack_buf[j] == stack_buf[j - 1] {
                skip = true;
                break;
            }
        }

        if skip {
            None
        } else {
            // Only copy the first #gates elements
            Some(stack_buf[..gates].to_vec())
        }
    })
}

//ordering of the vector will differ from the channel version
//this shouldn't matter
pub fn build_from(
    num_wires: usize,
    num_gates: usize,
    store: &Arc<HashMap<String, PersistPermStore>>,
) -> impl ParallelIterator<Item = Vec<usize>>{
    let n_base = base_gates(num_wires).len();

    store
        .par_iter() // parallelize over each stored permutation since threads are independent
        .flat_map_iter(move |(_key, perm)| {
            // iterator over the circuits for a given perm
            perm.circuits.iter().flat_map(move |circuit| { // use iter to avoid using more threads than the max
                // allocate prefix on stack (max 64 gates, adjust if needed)
                // use smallvec for stack efficiency
                let mut prefix: SmallVec<[usize; 64]> = SmallVec::with_capacity(num_gates);
                for &b in circuit.iter() { 
                    prefix.push(b as usize);
                }

                // iterator over base_gates, yielding q1 and q2
                (0..n_base).flat_map(move |idx| {
                        // skip consecutive duplicate
                        // before, we had s-1, but s was already num_gates - 1
                        if num_gates >= 2 && idx == prefix[num_gates - 2] {
                            return None;
                        }

                        // q1 = prefix + g
                        let mut q1 = SmallVec::<[usize; 64]>::with_capacity(num_gates);
                        q1.extend_from_slice(&prefix[..num_gates - 1]);
                        q1.push(idx);

                        // q2 = g + prefix
                        let mut q2 = SmallVec::<[usize; 64]>::with_capacity(num_gates);
                        q2.push(idx);
                        q2.extend_from_slice(&prefix[..num_gates - 1]);

                        // yield both q1 and q2 as Vec<usize>
                        Some([q1.into_vec(), q2.into_vec()])
                    })
                    .flatten() // flatten the array of Vecs into iterator
            })
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rainbow::init;
    // #[test]
    // fn test_to_circuit_and_canon() {
    //     init(3);
    //     let mut circuit1: Vec<usize> = Vec::new();
    //     circuit1.push(3);
    //     circuit1.push(5);
    //     circuit1.push(0);
    //     let circuit1 = CircuitSeq { gates: circuit1 };
    //     let base_gates = base_gates(3);
    //     let circuit1 = circuit1.to_circuit();

    //     let mut circuit2: Vec<usize> = Vec::new();
    //     circuit2.push(4);
    //     circuit2.push(0);
    //     circuit2.push(3);
    //     let circuit2 = CircuitSeq { gates: circuit2 };
    //     let circuit2 = circuit2.to_circuit();

    //     println!("{:?}", circuit1.permutation().canonical());
    //     println!("{:?}", circuit2.permutation().canonical());
    // } 

    #[test]
    fn test_from_string() {
        let s = "032;123;234;";
        println!("{}", CircuitSeq::from_string(s).to_string(5),);
    }

    // #[test]
    // fn test_circuit_string() {
    //     let s = "012;123;234";
    //     println!("{}", Circuit::from_string(s.to_string()).to_string());
    // }
}
