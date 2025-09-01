//Basic implementation for circuit, gate, and permutations

use crate::rainbow::constants::CONTROL_FUNC_TABLE;
use crate::circuit::control_functions::Gate_Control_Func;
use crate::rainbow::database::PersistPermStore;

use crossbeam::channel::{unbounded, Receiver};
use itertools::max;
use rand::{seq::SliceRandom, Rng, thread_rng};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::{
    cmp::max as std_max,
    collections::{HashSet, HashMap},
    fmt::{self, Write},
    path::Path,
    sync::Arc,
    thread,
};



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
    pub fn repr(&self, n: usize) -> String {
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
    // pub fn evaluate_gate(&self, wires: &mut Vec<bool>) {
    //     //use the fact that the index to the control function table is built from the control function, a, and b
    //     let index = ((self.control_function as usize) << 2) 
    //                     | ((wires[self.pins[1]] as usize) << 1)
    //                     | ((wires[self.pins[2]] as usize));
    //     //println!("{}{}",index, CONTROL_FUNC_TABLE[index]);
    //     wires[self.pins[0]] ^= CONTROL_FUNC_TABLE[index];
    // }


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
        std_max((std_max(self.pins[0], self.pins[1])), self.pins[2])
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

        let mut rng = rand::thread_rng();
        for _ in 0..num_gates {
            let mut j = last;
            while j == last{
                j = rng.gen_range(0..base_gates.len());
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
            .map(|gate| Gate_Control_Func::from_u8(gate.control_function).to_string())
            .collect();
        result.push_str("\ncfs: ");
        result.push_str(&control_fn_strings.join(", "));
        result
    }


    pub fn probably_equal(&self, other_circuit: &Self, num_inputs: usize) -> Result<(), String> {
        if self.num_wires != other_circuit.num_wires {
            return Err("The circuits do not have the same number of wires".to_string());
        }

        let mut rng = rand::thread_rng();
        let mask = (1 << self.num_wires) - 1;

        for _ in 0..num_inputs {
            // generate u64, then mask to get the lower num_wires bits
            let random_input = (rng.r#gen::<u64>() as usize) & mask;

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

    pub fn from_gates(gates: Vec<Gate>) -> Circuit {
        let mut w = 0;
        for g in &gates {
            w = std_max(w, g.bottom());
        }
        Circuit { num_wires: w+1, gates, }
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
        Self::from_gates(gates)
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
        Self::from_gates(new_gates)
    }

    //converts from a compressed string
    pub fn from_string_compressed(n: usize, s: &str) -> Circuit {
        let base_gates = base_gates(n);
        let mut gates = Vec::<Gate>::new();
        for ch in s.chars() {
            let gi = ch as usize;
            let pins = &base_gates[gi];
            gates.push(Gate { pins: *pins, control_function: 2,id: gi});
        }
        Circuit {num_wires: n, gates,}
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
        let mut rng = thread_rng();
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

    //come back to this. should be used for cache later when we create a rainbow
    pub fn repr(&self) -> String {
        let bytes: Vec<u8> = if self.data.len() > 256 {
            // Two-byte encoding (little-endian)
            let mut b = vec![0u8; 2 * self.data.len()];
            for (i, &val) in self.data.iter().enumerate() {
                b[2 * i..2 * i + 2].copy_from_slice(&(val as u16).to_le_bytes());
            }
            b
        } else {
            // Single-byte encoding
            self.data.iter().map(|&x| x as u8).collect()
        };

        // Convert bytes to a string safely
        String::from_utf8_lossy(&bytes).into_owned()
    }


    pub fn bits(&self) -> u32 {
        let n = self.data.len();
        ((n - 1) as u32).ilog2() + 1
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
// Old to_string based on gate inputs
// pub fn to_string(circuit_gates: &Vec<Gate>) -> String {
//     let mut wires: HashSet<usize> = HashSet::new();
//     for gate in circuit_gates {
//         wires.extend(gate.pins.iter());
//     }
//     let mut wire_list: Vec<usize> = wires.into_iter().collect();
//     wire_list.sort();

//     let mut result = String::new();
//     for (i, wire) in wire_list.iter().enumerate() {
//         result.push_str(&format!("{:<2} ", wire));
//         for gate in circuit_gates {
//             if gate.pins[0] == *wire {
//                 result+="( )";
//             } else if gate.pins[1] == *wire {
//                 result+="-●-";
//             } else if gate.pins[2] == *wire {
//                 result+="-○-";
//             } else {
//                 result+="-|-";
//             }
//             result.push_str("---");
//         }
//         if i != wire_list.len() - 1 {
//             result.push_str("\n");
//         }
//     }

//     let control_fn_strings: Vec<String> = circuit_gates
//         .iter()
//         .map(|gate| Gate_Control_Func::from_u8(gate.control_function).to_string())
//         .collect();
//     result.push_str("\ncfs: ");
//     result.push_str(&control_fn_strings.join(", "));
//     result
// }

// pub fn to_string(circuit: Circuit) {
//     let num_wires = circuit.num_wires;
//     let gates_list = circuit.gates;
//     let result = String::new();
//     for wire in range (0..num_wires) {
//         for gate in gates_list {
//             if gate.pins[0] == *wire {
//                 result+="( )";
//             } else if gate.pins[1] == *wire {
//                 result+="-●-";
//             } else if gate.pins[2] == *wire {
//                 result+="-○-";
//             } else {
//                 result+="-|-";
//             }
//             result.push_str("---");
//         }
//         result.push_str("---");
//         }
//         if i != wire_list.len() - 1 {
//             result.push_str("\n");
//         }
//     let control_fn_strings: Vec<String> = circuit_gates
//         .iter()
//         .map(|gate| Gate_Control_Func::from_u8(gate.control_function).to_string())
//         .collect();
//     result.push_str("\ncfs: ");
//     result.push_str(&control_fn_strings.join(", "));
//     result
// }

pub fn par_all_circuits(wires: usize, gates: usize) -> Receiver<Vec<usize>> {
    let (tx,rx) = unbounded();
    let base_gates = base_gates(wires);
    let z = base_gates.len() as i64;
    let total = (0..gates).fold(1i64, |acc, _| acc * z);

    thread::spawn(move || {
        const WORKERS: usize = 1;
        let (work_tx, work_rx) = unbounded::<i64>();

        //shadow for new threads
        let tx = Arc::new(tx);
        let mut handles = vec![];

        for _ in 0..WORKERS {
            let work_rx = work_rx.clone();
            let tx = Arc::clone(&tx);
            let handle = thread::spawn(move || {
                for mut i in work_rx.iter() {
                    let mut send = true;
                    let mut s = vec![0;gates];

                    for j in 0..gates {
                        s[j] = (i%z) as usize;
                        i /= z;
                        if j > 0 && s[j] == s[j-1] {
                            send = false;
                            break;
                        }
                    }
                    if send {
                        tx.send(s).unwrap();
                    }
                }
            });
            handles.push(handle);
        }
        //send
        for i in 0..total {
            work_tx.send(i).unwrap();
        }

        //close
        drop(work_tx);

        for handle in handles {
            handle.join().unwrap();
        }
    });
    rx
}

pub fn build_from(num_wires: usize, num_gates: usize, 
    store: &Arc<HashMap<String, PersistPermStore>>
) -> Receiver<Vec<usize>> {
    let (tx, rx) = unbounded::<Vec<usize>>();
    let base_gates = base_gates(num_wires);
    let store = Arc::clone(store);
    thread::spawn(move || {
        for perm in store.values() {
            //prefix circuit length gates-1
            let mut s = vec![0usize;num_gates-1];

            for circuits in &perm.circuits {
                let mut i = 0;
                for g in circuits.bytes() {
                    s[i] = g as usize;
                    i += 1;
                }

                for g in 0..base_gates.len() {
                    //no duplicates 
                    if g == s[s.len() - 1] {
                        continue;
                    }

                    let mut q1 = s.clone();
                    q1.push(g);
                    tx.send(q1).unwrap();

                    let mut q2 = vec![g];
                    q2.extend_from_slice(&s);
                    tx.send(q2).unwrap();

                    //TODO: also send reverse circuit?
                }
            }
        }
        drop(tx);
    });

    rx
}
