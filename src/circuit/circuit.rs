use crate::temp::constants::CONTROL_FUNC_TABLE;
use rand::seq::IndexedRandom;
use rand::Rng;
use rand::rng;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::fmt::Write;
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
}

impl Circuit{
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
                        control_function: (rng.random_range(0..16) as u8),
                    });
                    break;
                }
            }
        }
        Self{num_wires, gates}
    }
    pub fn to_string(&self) -> String{
        to_string(&self.gates)
    }

    pub fn probably_equal(&self, other_circuit: &Self, num_inputs: usize) -> Result<(), String> {
        if self.num_wires != other_circuit.num_wires {
            return Err("The circuits do not have the same number of wires".to_string());
        }
        let random_inputs: Vec<Vec<bool>> = (0..num_inputs)
            .map(|_| (0..num_inputs)
                .map(|_| rand::rng().random_bool(0.5)).collect())
            .collect();
        random_inputs.iter().try_for_each(|random_input| {
            if Gate::evaluate_gate_list(&self.gates, random_input) != Gate::evaluate_gate_list(&other_circuit.gates, random_input) {
                return Err("Circuits are not equal".to_string());
            }
        Ok(())
        })
    }

    // pub fn functionally_equal()(&self, other_circuit: &Self, num_inputs: usize) -> Result<(), String> {

    // }
}

pub fn to_string(circuit_gates: &Vec<Gate>) -> String {
    let mut wires: HashSet<usize> = HashSet::new();
    for gate in circuit_gates {
        wires.extend(gate.pins.iter());
    }
    let mut wire_list: Vec<usize> = wires.into_iter().collect();
    wire_list.sort();

    let mut result = String::new();
    for (i, wire) in wire_list.iter().enumerate() {
        result.push_str(&format!("{:<2} ", wire));
        for gate in circuit_gates {
            if gate.pins[0] == *wire {
                result+="( )";
            } else if gate.pins[1] == *wire {
                result+=" ● ";
            } else if gate.pins[2] == *wire {
                result+=" ○ ";
            } else {
                result+=" - ";
            }
            result.push_str(" - ");
        }
        if i != wire_list.len() - 1 {
            result.push_str("\n");
        }
    }

    let control_fn_strings: Vec<String> = circuit_gates
        .iter()
        .map(|gate| Gate_Control_Func::from_u8(gate.control_function).to_string())
        .collect();
    result.push_str("\ncfs: ");
    result.push_str(&control_fn_strings.join(", "));
    result
}
