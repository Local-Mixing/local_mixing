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
use std::sync::atomic::Ordering;

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

#[derive(Clone, Debug, Default)]
pub struct CircuitSeq {
    pub gates: Vec<usize>,
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

    pub fn collides_index(a: usize, b:usize, base_gates: &Vec<[usize;3]>) -> bool {
        base_gates[a][0] == base_gates[b][1]
            || base_gates[a][0] == base_gates[b][2] 
            || base_gates[a][1] == base_gates[b][0] 
            || base_gates[a][2] == base_gates[b][0]
    }
    //b is "larger"
    pub fn ordered_index(a: usize, b:usize, base_gates: &Vec<[usize;3]>) -> bool {
        if base_gates[a][0] > base_gates[b][0] {
            return false
        }
        else if base_gates[a][0] == base_gates[b][0]{
            if base_gates[a][1] > base_gates[b][1] {
                return false
            }
            else if base_gates[a][1] == base_gates[b][1] {
                return base_gates[a][2] < base_gates[b][2]
            }
        }
        true
    }

    #[inline(always)]
    pub fn evaluate_index(state: usize, gate: usize, base_gates: &Vec<[usize;3]>) -> usize {
        let pins = base_gates[gate];
        let c1 = (state >> pins[1]) & 1;
        let c2 = (state >> pins[2]) & 1;
        state ^ (c1 | ((!c2) & 1)) << pins[0]
    }

    #[inline(always)]
    pub fn evaluate_index_list(state: usize, gates: &Vec<usize>, base_gates: &Vec<[usize;3]>) -> usize {
        let mut current_wires = state;
        for g in gates {
            current_wires = Self::evaluate_index(current_wires, *g, base_gates);
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

    pub fn repr(&self) -> String {
    self.data.iter()
        .map(|&x| x.to_string())
        .collect::<Vec<_>>()
        .join(",")
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

    pub fn evaluate(&self, input: usize, base_gates: &Vec<[usize;3]>) -> usize {
        Gate::evaluate_index_list(input, &self.gates, base_gates)
    }

    pub fn permutation(&self, num_wires: usize, base_gates: &Vec<[usize;3]>) -> Permutation {
        let size = 1 << num_wires;
        //TODO: the size could be over 64, so is small vec a bad choice?
        let mut output = vec![0; size];

        for input in 0..size {
            output[input] = self.evaluate(input, base_gates);
        }

        Permutation { data: output }
    }

    pub fn repr(&self) -> String {
        // Join all gates as decimal numbers separated by commas
        self.gates.iter()
            .map(|&id| id.to_string())
            .collect::<Vec<_>>()
            .join(",")
    }
    
    pub fn repr_bytes(&self) -> Vec<u8> {
        // just store each usize as a byte if it's <= 255
        // or u16 if larger
        if self.gates.iter().all(|&x| x <= 255) {
            self.gates.iter().map(|&x| x as u8).collect()
        } else {
            // fallback to 2-byte encoding
            let mut buf = Vec::with_capacity(self.gates.len() * 2);
            for &x in &self.gates {
                buf.extend_from_slice(&(x as u16).to_le_bytes());
            }
            buf
        }
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

