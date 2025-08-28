//This will include the compression idea used in the attack 
use crate::{
            rainbow::constants::{self, CONTROL_FUNC_TABLE},
            circuit::{self,Circuit, Gate, Permutation},
            };

impl Circuit {
    pub fn from_string_compressed(n: usize, s: &str) -> Circuit {
        let base_gates = circuit::base_gates(n);
        let mut gates = Vec::<Gate>::new();
        for ch in s.chars() {
            let gi = ch as usize;
            let pins = &base_gates[gi];
            gates.push(Gate { pins: *pins, control_function: 2,id: gi});
        }
        Circuit {num_wires: n, gates,}
    }
}
