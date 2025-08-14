//Basic implementation for circuit, gate, and permutations

use crate::rainbow::constants::CONTROL_FUNC_TABLE;
use rand::seq::IndexedRandom;
use rand::Rng;
use rand::rng;
use rand::seq::SliceRandom; // for shuffle
use rand::thread_rng;  
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::fmt::{self, Write};
use std::{collections::HashSet, path::Path};
use crate::circuit::control_functions::Gate_Control_Func;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Gate{
    pub pins: [usize;3], //one active wire and two control wires
    pub control_function: u8,
}

#[derive(Clone, Debug, Default)]
pub struct Circuit{
    pub num_wires: usize,
    pub gates: Vec<Gate>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Permutation {
    pub data: Vec<usize>,
}




impl Gate {
    pub fn new(active: usize, first_control: usize, second_control: usize, control_function: u8) -> Self {
        Self {
            pins: [active, first_control, second_control],
            control_function,
        }
    }
    
    //two gates collide if an active and a control pins are on the same wire
    pub fn collides(&self, other_gate: &Self) -> bool {
        self.pins[0] == other_gate.pins[1] 
            || self.pins[0] == other_gate.pins[2]
            || self.pins[1] == other_gate.pins[0] 
            || self.pins[2] == other_gate.pins[0]
    }

    //to evaluate a gate, we use the control function table we built in constants.rs
    #[inline]
    pub fn evaluate_gate(&self, wires: &mut Vec<bool>) {
        //use the fact that the index to the control function table is built from the control function, a, and b
        let index = ((self.control_function as usize) << 2) 
                        | ((wires[self.pins[1]] as usize) << 1)
                        | ((wires[self.pins[2]] as usize));
        println!("{}{}",index, CONTROL_FUNC_TABLE[index]);
        wires[self.pins[0]] ^= CONTROL_FUNC_TABLE[index];
    }

    pub fn equal(&self, other: &Self) -> bool {
        self.pins == other.pins && self.control_function == other.control_function
    }

    pub fn evaluate_gate_list(gate_list: &Vec<Gate>, input_wires: &Vec<bool>) -> Vec<bool> {
        //first clone the input_wires and then iterate through all the gates using the original evaluate_gate function
        let mut current_wires = input_wires.to_vec();
        gate_list.iter()
                 .for_each(|gate| gate.evaluate_gate(&mut current_wires));
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
}













impl Circuit{
    pub fn new(num_wires:usize, gates: Vec<Gate>) -> Self {
        Self{num_wires, gates}
    }

    pub fn random_circuit<R: Rng>(
        num_wires: usize,
        num_gates: usize,
        rng: &mut R
    ) -> Self {
        let mut gates = vec![];
        for _ in 0..num_gates {
            loop{
                let active = rng.random_range(0..num_wires);
                let first_control = rng.random_range(0..num_wires);
                let second_control = rng.random_range(0..num_wires);

                if active != first_control && active != second_control && first_control != second_control {
                    gates.push(Gate {
                        pins: [active, first_control, second_control],
                        //control_function: (rng.random_range(0..16) as u8), any control
                        control_function: 2, //r57
                    });
                    break;
                }
            }
        }
        Self{num_wires, gates}
    }
    pub fn to_string(&self) -> String{
        let mut result = String::new();
        for wire in (0..self.num_wires) {
            result += &(wire.to_string() + "  --");
            for gate in &self.gates {
                if gate.pins[0] == wire {
                    result+="( )";
                } else if gate.pins[1] == wire {
                    result+="-●-";
                } else if gate.pins[2] == wire {
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
        let random_inputs: Vec<Vec<bool>> = (0..num_inputs)
            .map(|_| (0..self.num_wires)
                .map(|_| rand::rng().random_bool(0.5)).collect())
            .collect();
        random_inputs.iter().try_for_each(|random_input| {
            if Gate::evaluate_gate_list(&self.gates, random_input) != Gate::evaluate_gate_list(&other_circuit.gates, random_input) {
                return Err("Circuits are not equal".to_string());
            }
        Ok(())
        })
    }

    // CAN TWO CIRCUITS WITH DIFFERENT NUMBER OF WIRES BE FUNCTIONALLY EQUIVALENT?????
    // pub fn functionally_equal()(&self, other_circuit: &Self, num_inputs: usize) -> Result<(), String> {
    //     let least_num_wires = min(self.num_wires, other_circuit.num_wires);
    //     if num_inputs > 
    // }

    pub fn len(circuit: Circuit) -> usize {
        circuit.gates.len()
    }

    pub fn evaluate(&self, input_wires: &Vec<bool>) -> Vec<bool> {
        Gate::evaluate_gate_list(&self.gates, &input_wires)
    }

    // pub fn permuation(&self) -> Permutation {

    // }
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
    pub fn repr(&self) -> Vec<u8> {
        let n = self.data.len();

        if n > 256 {
            // Two-byte encoding (little-endian)
            let mut bytes = vec![0u8; 2 * n];
            for (i, &val) in self.data.iter().enumerate() {
                bytes[2 * i..2 * i + 2].copy_from_slice(&(val as u16).to_le_bytes());
            }
            bytes
        } else {
            // Single-byte encoding (0..=255)
            self.data.iter().map(|&x| x as u8).collect()
        }
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

    pub fn bit_shuffle(&self, shuf: &[usize]) -> Permutation {
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












// Alex's to_string based on gate inputs
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

