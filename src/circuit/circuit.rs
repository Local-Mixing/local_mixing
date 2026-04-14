// Basic implementation for circuit, gate, and permutations
use primitive_types::U256 as u256;
use rand::{seq::SliceRandom, RngCore,};
use serde::{Deserialize, Serialize};
use std::{
    cmp::max as std_max,
    collections::{HashSet, HashMap},
};

// pins are [active, control1, control2] for Toffoli gates
// We are only concerned with gate r57
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Gate{
    pub pins: [usize;3], //one active wire (0) and two control wires (1,2)
}

// Circuits stored as a sequence of gates [u8;3]
// Gate type is legacy
#[derive(Clone, Debug, Default, Serialize, Deserialize, Eq, Hash, PartialEq)]
pub struct CircuitSeq {
    pub gates: Vec<[u8;3]>, 
}

// Polynomial representation of circuit
pub type Monomial = u64;
pub type Polynomial = HashSet<Monomial>;

// Permutations are all the possible outputs of a circuit
// On n wires permutation length is 1 << n
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Permutation {
    pub data: Vec<usize>,
}

fn count_ones_u256(x: u256) -> u32 {
    x.0.iter().map(|w| w.count_ones()).sum()
}

// Functions on Gate struct and [u8;3]
impl Gate {
    // Returns the largest wire used
    pub fn bottom(&self) -> usize {
        // println!("bottom is {}", std_max((std_max(self.pins[0], self.pins[1])), self.pins[2]));
        std_max(std_max(self.pins[0], self.pins[1]), self.pins[2])
    }

    // Gates collide iff either active pin shares a wire with any other pin
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

    // Evaluate a bit string after a single gate under gate r57
    #[inline(always)]
    pub fn evaluate_index(state: usize, gate: [u8;3]) -> usize {
        let c1 = (state >> gate[1]) & 1;
        let c2 = (state >> gate[2]) & 1;
        state ^ (c1 | ((!c2) & 1)) << gate[0]
    }

    // Evaluate up to 128 bits
    #[inline(always)]
    pub fn evaluate_index_128(state: u128, gate: [u8;3]) -> u128 {
        let c1 = (state >> gate[1]) & 1;
        let c2 = (state >> gate[2]) & 1;
        state ^ (c1 | ((!c2) & 1)) << gate[0]
    }

    // Evaluate up to 256 bits
    #[inline(always)]
    pub fn evaluate_index_256(state: u256, gate: [u8;3]) -> u256 {
        let one = u256::one();
        let c1 = (state >> gate[1]) & one;
        let c2 = (state >> gate[2]) & one;
        state ^ ((c1 | (one ^ c2)) << gate[0])
    }

    // Evaluate a list of gates
    #[inline(always)]
    pub fn evaluate_index_list(state: usize, gates: &Vec<[u8;3]>) -> usize {
        let mut current_wires = state;
        for g in gates {
            current_wires = Self::evaluate_index(current_wires, *g);
        }
        current_wires
    }

    #[inline(always)]
    pub fn evaluate_index_list_128(state: u128, gates: &Vec<[u8;3]>) -> u128 {
        let mut current_wires = state;
        for g in gates {
            current_wires = Self::evaluate_index_128(current_wires, *g);
        }
        current_wires
    }

    #[inline(always)]
    pub fn evaluate_index_list_256(state: u256, gates: &Vec<[u8;3]>) -> u256 {
        let mut current_wires = state;
        for g in gates {
            current_wires = Self::evaluate_index_256(current_wires, *g);
        }
        current_wires
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

    // n is the length of the permutation. For a random permutation on n bits, do 1 << n
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

    // string representation is just the elements of the permutation separated by a ,
    pub fn repr(&self) -> String {
        self.data.iter()
            .map(|&x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")
    }

    // u8 representation for db
    pub fn repr_blob(&self) -> Vec<u8> {
        self.data.iter().map(|&x| x as u8).collect()
    }

    pub fn from_blob(blob: &[u8]) -> Self {
        let data = blob.iter().map(|&b| b as usize).collect();
        Permutation { data }
    }

    // Returns the number of bits needed to represent the permutation
    pub fn bits(&self) -> usize {
        let n = self.data.len();
        ((n - 1) as usize).ilog2() as usize + 1
    }

    // Cycle representation of a permutation
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

    // On permutation of len 1 << n with n bits, take a bit shuffle on n bits and apply 
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
    // Checks for the presence of two identical gates
    pub fn adjacent_id(&self) -> bool {
        for i in 0..(self.gates.len()-1) {
            if self.gates[i] == self.gates[i+1] {
                return true
            }
        }
        false
    }

    // Evaluate the entire circuit with a starting input
    pub fn evaluate(&self, input: usize) -> usize {
        Gate::evaluate_index_list(input, &self.gates)
    }

    pub fn evaluate_128(&self, input: u128) -> u128 {
        Gate::evaluate_index_list_128(input, &self.gates)
    }

    pub fn evaluate_256(&self, input: u256) -> u256 {
        Gate::evaluate_index_list_256(input, &self.gates)
    }

    // Find the permutation computed by the circuit. Permutation is on 2^n
    pub fn permutation(&self, num_wires: usize) -> Permutation {
        let size = 1 << num_wires;
        
        let mut output = vec![0; size];

        for input in 0..size {
            output[input] = self.evaluate(input);
        }

        Permutation { data: output }
    }

    // Store as sequence of u8 for dbs
    pub fn repr_blob(&self) -> Vec<u8> {
        let mut blob = Vec::with_capacity(self.gates.len() * 3);
        for &gate in &self.gates {
            blob.push(gate[0] as u8);
            blob.push(gate[1] as u8);
            blob.push(gate[2] as u8);
        }
        blob
    }

    /// Reconstruct CircuitSeq from a BLOB
    pub fn from_blob(blob: &[u8]) -> Self {
        assert!(blob.len() % 3 == 0, "Invalid blob length");
        let gates: Vec<[u8; 3]> = blob
            .chunks(3)
            .map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect();
        CircuitSeq { gates }
    }

    // Rewire wire i -> perm[i]
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

    // Representing circuit as a string
    pub fn repr(&self) -> String {
        fn wire_to_char(w: u8) -> char {
            match w {
                0..=9 => (b'0' + w) as char,          // 0-9
                10..=35 => (b'a' + (w - 10)) as char, // a-z
                36..=61 => (b'A' + (w - 36)) as char, // A-Z
                // Special characters 62..=71
                62 => '!',
                63 => '@',
                64 => '#',
                65 => '$',
                66 => '%',
                67 => '^',
                68 => '&',
                69 => '*',
                70 => '(',
                71 => ')',
                // Special characters 72..=82
                72 => '-',
                73 => '_',
                74 => '=',
                75 => '+',
                76 => '[',
                77 => ']',
                78 => '{',
                79 => '}',
                80 => '<',
                81 => '>',
                82 => '?',
                _ => panic!("Invalid wire index: {}", w),
            }
        }

        const BASE: u8 = 83; // 0..82 is base

        fn encode_wire(mut w: u32) -> String {
            let mut s = String::new();
            let mut tildes = 0;

            while w >= BASE as u32 {
                tildes += 1;
                w -= BASE as u32;
            }

            for _ in 0..tildes {
                s.push('~');
            }
            s.push(wire_to_char(w as u8));
            s
        }

        let mut s = String::new();
        for gate in &self.gates {
            for &wire in gate {
                s.push_str(&encode_wire(wire as u32));
            }
            s.push(';'); // gate separator
        }
        s
    }

    pub fn from_string(s: &str) -> Self {
        fn char_to_wire(c: char) -> u8 {
            match c {
                '0'..='9' => c as u8 - b'0',          // 0-9
                'a'..='z' => c as u8 - b'a' + 10,     // 10-35
                'A'..='Z' => c as u8 - b'A' + 36,     // 36-61
                '!' => 62,
                '@' => 63,
                '#' => 64,
                '$' => 65,
                '%' => 66,
                '^' => 67,
                '&' => 68,
                '*' => 69,
                '(' => 70,
                ')' => 71,
                '-' => 72,
                '_' => 73,
                '=' => 74,
                '+' => 75,
                '[' => 76,
                ']' => 77,
                '{' => 78,
                '}' => 79,
                '<' => 80,
                '>' => 81,
                '?' => 82,
                _ => panic!("Invalid wire char: {}", c),
            }
        }

        const BASE: u32 = 83;

        let gates: Vec<[u8; 3]> = s
            .trim()
            .split(';')
            .filter(|part| !part.is_empty())
            .map(|gate_str| {
                let mut chars = gate_str.chars().peekable();
                let mut wires = Vec::new();

                while chars.peek().is_some() {
                    // Count tildes for overflow
                    let mut overflow = 0;
                    while chars.peek() == Some(&'~') {
                        overflow += 1;
                        chars.next();
                    }

                    // Next character is the base wire
                    let c = chars.next().expect("Expected wire character after ~");
                    let wire = char_to_wire(c) as u32 + overflow * BASE;
                    wires.push(wire as u8);
                }

                if wires.len() != 3 {
                    panic!("Each gate must have exactly 3 wires: {:?}", gate_str);
                }

                [wires[0], wires[1], wires[2]]
            })
            .collect();

        CircuitSeq { gates }
    }

    // Gives a "pretty" circuit representation. Does not support over 83 wires
    pub fn to_string(&self, num_wires: usize) -> String {
        let mut result = String::new();

        // Local character map (0-9, a-z, A-Z)
        let wire_map_chars: Vec<char> = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_=+[]{}<>?"
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

    // Combine two circuits
    pub fn concat(&self, other: &CircuitSeq) -> CircuitSeq {
        let mut gates = self.gates.clone();
        gates.extend_from_slice(&other.gates);
        CircuitSeq { gates }
    }

    // Returns the wires touched by a circuit
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

    // "Bottom" function for gates
    pub fn max_wire(&self) -> usize {
        self.gates.iter().flatten().copied().max().unwrap_or(0) as usize
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

    // Evaluates how a state changes throughout the entirety of a circuit
    pub fn evaluate_evolution(&self, input: usize) -> Vec<usize> {
        let mut state = input;
        let mut evolution = vec![state];

        for gate in &self.gates {
            state = Gate::evaluate_index(state, *gate);
            evolution.push(state);
        }

        evolution
    }

    pub fn evaluate_evolution_128(&self, input: u128) -> Vec<u128> {
        let mut state = input;
        let mut evolution = vec![state];

        for gate in &self.gates {
            state = Gate::evaluate_index_128(state, *gate);
            evolution.push(state);
        }

        evolution
    }

    pub fn evaluate_evolution_256(&self, input: u256) -> Vec<u256> {
        let mut state = input;
        let mut evolution = vec![state];

        for gate in &self.gates {
            state = Gate::evaluate_index_256(state, *gate);
            evolution.push(state);
        }

        evolution
    }

    // Probablistic check on circuit equality
    // pub fn probably_equal(&self, other_circuit: &Self, num_wires: usize, num_inputs: usize) -> Result<(), String> {
    //     let mut rng = rand::rng();
    //     let mask: u128 = if num_wires < u128::BITS as usize {
    //         (1 << num_wires) - 1
    //     } else {
    //         u128::MAX
    //     };
    //     for _ in 0..num_inputs {
    //         // generate u64, then mask to get the lower num_wires bits
    //         let random_input = (rng.random::<u64>() as u128) & mask;

    //         let self_output = Gate::evaluate_index_list_128( random_input, &self.gates);
    //         let other_output = Gate::evaluate_index_list_128(random_input, &other_circuit.gates);

    //         if (self_output & mask) != (other_output & mask) {
    //             return Err("Circuits are not equal".to_string());
    //         }
    //     }

    //     Ok(())
    // }

    // Probabilistic check on circuit equality
    pub fn probably_equal(
        &self,
        other_circuit: &Self,
        num_wires: usize,
        num_inputs: usize
    ) -> Result<(), String> {

        let mut rng = rand::rng();

        // build mask with lowest num_wires bits set
        let mask = if num_wires < 256 {
            (u256::one() << num_wires) - u256::one()
        } else {
            u256::MAX
        };

        for _ in 0..num_inputs {

            // generate random 256-bit input
            let mut bytes = [0u8; 32];
            rng.fill_bytes(&mut bytes);
            let random_input = u256::from_little_endian(&bytes) & mask;

            let self_output =
                Gate::evaluate_index_list_256(random_input, &self.gates);

            let other_output =
                Gate::evaluate_index_list_256(random_input, &other_circuit.gates);

            if (self_output & mask) != (other_output & mask) {
                return Err("Circuits are not equal".to_string());
            }
        }

        Ok(())
    }

    /// Relabel wires in encounter order (first wire seen → 0, second → 1, etc.)
    fn first_wire_form(&self) -> Vec<[u8; 3]> {
        let mut map: HashMap<u8, u8> = HashMap::new();
        let mut counter: u8 = 0;

        self.gates.iter().map(|gate| {
            gate.map(|wire| {
                *map.entry(wire).or_insert_with(|| {
                    let id = counter;
                    counter += 1;
                    id
                })
            })
        }).collect()
    }

    pub fn is_relabeling_of(&self, other: &CircuitSeq) -> bool {
        self.first_wire_form() == other.first_wire_form()
    }

    pub fn to_polynomial(self, n: usize, start: usize, end: usize) -> Vec<Polynomial> {
        let gates = &self.gates[start..end];
        // Wire i starts as degree 1 monomial
        let mut polys: Vec<Polynomial> = (0..n)
        .map(|i| HashSet::from([1u64 << i]))
        .collect();
    
        for &[a, b, c] in gates {
            // a' = a XOR (not b AND c) XOR 1
            let not_b = poly_not(polys[b as usize].clone());
            let term = poly_and(&not_b, &polys[c as usize]);
            let mut new_a = poly_xor(polys[a as usize].clone(), term);
            if !new_a.remove(&0u64) {
                new_a.insert(0u64);
            }
            polys[a as usize] = new_a;
        }
    
        // XOR each wire with its initial value x_i so unchanged wires become 0
        // for i in 0..n {
        //     let xi = HashSet::from([1u64 << i]);
        //     polys[i] = poly_xor(polys[i].clone(), xi);
        // }

        polys
    }
    // Computes the upper bound of each wire's algebraic degree
    pub fn to_degree_upper(self, n: usize, start: usize, end: usize) -> Vec<u8> {
        let mut deg: Vec<u8> = vec![0u8; n];

        for &[active, ctrl1, ctrl2] in &self.gates[start..end] {
            // active ^= ctrl1 & !ctrl2
            // new degree = max(deg[active], deg[ctrl1] + deg[ctrl2])
            let new_deg = deg[active as usize]
                .max(deg[ctrl1 as usize].saturating_add(deg[ctrl2 as usize])).min(n as u8);
            
            if new_deg == 0 {
                deg[active as usize] = 1;
            } else {
                deg[active as usize] = new_deg;
            }
        }
        deg
    }
}

fn poly_xor(mut poly_1: Polynomial, poly_2: Polynomial) -> Polynomial {
    for m in poly_2 {
        if !poly_1.remove(&m) {
            poly_1.insert(m);
        }
    }
    poly_1
}

fn poly_and(poly_1: &Polynomial, poly_2: &Polynomial) -> Polynomial {
    let mut result = Polynomial::new();
    for &m1 in poly_1 {
        for &m2 in poly_2 {
            let m = m1 | m2;
            if !result.remove(&m) {
                result.insert(m);
            }
        }
    }
    result
}
 
fn poly_not(p: Polynomial) -> Polynomial {
    // NOT f = 1 + f; constant 1 is the empty monomial
    let one = HashSet::from([0u64]);
    poly_xor(one, p)
}

// Display polynomials
 
pub fn monomial_degree(m: u64) -> u32 {
    m.count_ones()
}
 
fn monomial_to_str(m: u64, n: usize) -> String {
    if m == 0 {
        return "1".to_string();
    }
    (0..n)
        .filter(|&i| (m >> i) & 1 == 1)
        .map(|i| format!("x{}", i))
        .collect::<Vec<_>>()
        .join("*")
}
 
pub fn poly_to_str(poly: &Polynomial, n: usize) -> String {
    if poly.is_empty() {
        return "1".to_string();
    }
    let mut terms: Vec<u64> = poly.iter().copied().collect();
    // Sort by degree, then by value
    terms.sort_by_key(|&m| (monomial_degree(m), m));
    terms
        .iter()
        .map(|&m| monomial_to_str(m, n))
        .collect::<Vec<_>>()
        .join(" + ")
}
 
pub fn poly_degree(poly: &Polynomial) -> u32 {
    poly.iter().map(|&m| monomial_degree(m)).max().unwrap_or(0)
}

// Rewire wire i -> perm[i]
pub fn rewire_gate_ver(gates: &mut Vec<[u8;3]>, perm: &Permutation, n: usize) {
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

    for gate in gates {
        *gate = [
            perm.data[gate[0] as usize] as u8,
            perm.data[gate[1] as usize] as u8,
            perm.data[gate[2] as usize] as u8,
        ];
    }
}

// Possible gates on n wires
pub fn base_gates(n: usize) -> Vec<[u8; 3]> {
    let n = n as u8;
    let mut gates: Vec<[u8;3]> = Vec::new();
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

/// Initial ranking method
/// Degree counts of a polynomial: [count_of_max_possible_deg, ..., count_of_deg_0]
/// Padded to max_possible_degree+1 entries so Vec comparison is always over equal-length
/// vectors and correctly ranks e.g. one degree-2 monomial above two degree-1 monomials.
fn degree_counts(poly: &Polynomial, max_possible_degree: usize) -> Vec<usize> {
    let mut counts = vec![0usize; max_possible_degree + 1];
    for m in poly {
        let deg = m.count_ones() as usize;
        counts[deg] += 1;
    }
    counts.reverse();
    counts
}

/// Used in tie-breaking
/// For a given polynomial, return a degree-bucketed count (high to low) of
/// how many monomials of each degree contain variable `wire_idx`.
fn wire_counts_in_poly(poly: &Polynomial, max_possible_degree: usize, wire_idx: usize) -> Vec<usize> {
    if poly.is_empty() {
        return vec![];
    }
    let bit = 1u64 << wire_idx;
    let mut counts = vec![0usize; max_possible_degree + 1];
    for m in poly {
        if m & bit != 0 {
            counts[m.count_ones() as usize] += 1;
        }
    }
    counts.reverse();
    counts
}

/// Used in tie-breaking. Check a single polynomial
/// Split a single tied group by their appearance counts in one polynomial.
/// Returns ordered sub-groups (highest count first), each internally still tied.
fn split_by_poly(group: &[usize], poly: &Polynomial, max_possible_degree: usize) -> Vec<Vec<usize>> {
    let mut scored: Vec<(usize, Vec<usize>)> = group
        .iter()
        .map(|&w| (w, wire_counts_in_poly(poly, max_possible_degree, w)))
        .collect();

    // Sort descending by count
    scored.sort_by(|a, b| b.1.cmp(&a.1));

    // Partition into tied sub-groups
    let mut result: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = vec![scored[0].0];
    for i in 1..scored.len() {
        if scored[i].1 == scored[i - 1].1 {
            current.push(scored[i].0);
        } else {
            result.push(current.clone());
            current = vec![scored[i].0];
        }
    }
    result.push(current);
    result
}

/// Used in tie-breaking. Check multiple polynomials in ranked order
/// Try to split a group of tied wires using the ranked polynomials (highest rank first).
/// Restarts from the top-ranked polynomial whenever any split occurs.
/// Returns a list of sub-groups in rank order.
fn split_group(group: &[usize], ranked_polys: &[&Polynomial], max_possible_degree: usize) -> Vec<Vec<usize>> {
    if group.len() <= 1 {
        return vec![group.to_vec()];
    }

    let mut subgroups: Vec<Vec<usize>> = vec![group.to_vec()];

    'outer: loop {
        for poly in ranked_polys.iter() {
            let mut new_subgroups: Vec<Vec<usize>> = Vec::new();
            let mut any_split = false;

            for sg in &subgroups {
                if sg.len() <= 1 {
                    new_subgroups.push(sg.clone());
                    continue;
                }
                let split = split_by_poly(sg, poly, max_possible_degree);
                if split.len() > 1 {
                    any_split = true;
                }
                new_subgroups.extend(split);
            }

            if any_split {
                subgroups = new_subgroups;
                continue 'outer; // restart from highest ranked poly
            }
        }
        // Full pass with no splits — we're done
        break;
    }

    subgroups
}

pub fn canonicalize_polys(polynomials: Vec<Polynomial>) -> (Vec<Polynomial>, Permutation) {
    let n = polynomials.len();
    if n == 0 {
        return (vec![], Permutation { data: vec![] });
    }
    let max_degree = n;

    // Step 1: Initial grouping by degree counts
    let mut profiles: Vec<(usize, Vec<usize>)> = (0..n)
        .map(|i| (i, degree_counts(&polynomials[i], max_degree)))
        .collect();

    // Sort descending: higher degree entries first, then lexicographically by counts
    profiles.sort_by(|a, b| b.1.cmp(&a.1));

    // Partition into initially-tied groups
    let mut pending_groups: Vec<Vec<usize>> = Vec::new();
    {
        let mut current: Vec<usize> = vec![profiles[0].0];
        for i in 1..profiles.len() {
            if profiles[i].1 == profiles[i - 1].1 {
                current.push(profiles[i].0);
            } else {
                pending_groups.push(current.clone());
                current = vec![profiles[i].0];
            }
        }
        pending_groups.push(current);
    }

    // final_order[pos] = Some(wire) once a wire is locked into that position, None while
    // still unresolved. Positions are assigned to groups upfront based on degree ranking,
    // so e.g. a singleton group at the end is immediately locked at its correct position
    // even while earlier groups are still in tiebreak. Only locked positions are visible
    // to ranked_polys_from and thus usable for comparisons.
    let mut final_order: Vec<Option<usize>> = vec![None; n];

    // Each entry is (start_pos, sub_groups):
    //   start_pos — base index into final_order for the first still-unresolved wire in this group
    //   sub_groups — current partition of unresolved wires; starts as one vec, splits as
    //                tiebreaks are resolved. Singleton sub_groups get locked into final_order
    //                and are removed; the entry is dropped when all sub_groups are singletons.
    let mut pending: Vec<(usize, Vec<Vec<usize>>)> = {
        let mut pos = 0;
        pending_groups
            .into_iter()
            .map(|g| {
                let start = pos;
                pos += g.len();
                (start, vec![g])
            })
            .collect()
    };

    // Build ranked_polys from locked-in positions only (skip None).
    // Returns polynomials in rank order (position 0 first).
    let ranked_polys_from = |order: &Vec<Option<usize>>| -> Vec<&Polynomial> {
        order
            .iter()
            .filter_map(|slot| slot.map(|w| &polynomials[w]))
            .collect()
    };

    
    loop {
        let mut any_progress = false;

        for (start_pos, sub_groups) in pending.iter_mut() {
            let mut local_progress = true;

            // Rules 2-5
            // Keep re-splitting until no more progress within this group
            let mut current = sub_groups.clone();
            while local_progress {
                local_progress = false;
                let ranked = ranked_polys_from(&final_order);
                let mut next: Vec<Vec<usize>> = Vec::new();
                for sg in &current {
                    if sg.len() <= 1 {
                        next.push(sg.clone());
                        continue;
                    }
                    let split = split_group(sg, &ranked, max_degree);
                    if split.len() > 1 {
                        local_progress = true;
                        any_progress = true;
                    }
                    next.extend(split);
                }
                current = next;
            }

            // Lock in any singletons at their positions, advancing start_pos past each one
            let mut pos = *start_pos;
            for sg in &current {
                if sg.len() == 1 && final_order[pos].is_none() {
                    final_order[pos] = Some(sg[0]);
                    any_progress = true;
                }
                pos += sg.len();
            }
            // Advance start_pos past all leading singletons so rule 6 always
            // writes the next winner into the correct slot
            *start_pos += current.iter().take_while(|sg| sg.len() == 1).count();
            *sub_groups = current.into_iter().skip_while(|sg| sg.len() == 1).collect();
        }

        // Drop fully resolved pending entries
        pending.retain(|(_, sgs)| sgs.iter().any(|sg| sg.len() > 1));

        if pending.is_empty() {
            break;
        }

        if !any_progress {
            // Rule 6: fully stuck — pick lowest wire index from first unresolved sub_group
            // of first pending entry as a single arbitrary tiebreak, then restart
            let (start_pos, sub_groups) = &mut pending[0];
            sub_groups[0].sort();
            let winner = sub_groups[0].remove(0);
            final_order[*start_pos] = Some(winner);
            *start_pos += 1;
        }
    }

    // Unwrap — all positions must be filled by now
    let final_order: Vec<usize> = final_order
        .into_iter()
        .map(|slot| slot.expect("all positions should be filled"))
        .collect();

    // final_order[pos] = wire
    // Remap: variable x_wire -> x_pos  (bit wire -> bit pos)
    // TODO: this is likely the inverse. may need to change for consistency
    let remap_monomial = |m: Monomial| -> Monomial {
        let mut result = 0u64;
        for (pos, &wire) in final_order.iter().enumerate() {
            if m & (1u64 << wire) != 0 {
                result |= 1u64 << pos;
            }
        }
        result
    };

    // Remap all of our polynomials and select them in the order
    // Both based on final_order
    let canonical: Vec<Polynomial> = final_order
    .iter()
    .map(|&wire| {
        polynomials[wire]
            .iter()
            .map(|&m| remap_monomial(m))
            .collect()
    })
    .collect();

    let data = final_order;

    (canonical, Permutation { data })
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use super::*;

    fn mono(vars: &[usize]) -> Monomial {
        vars.iter().fold(0u64, |acc, &v| acc | (1u64 << v))
    }

    fn poly(monomials: &[&[usize]]) -> Polynomial {
        let mut p = Polynomial::new();
        for &vars in monomials {
            let m = mono(vars);
            // GF(2): duplicate insertion cancels
            if !p.remove(&m) {
                p.insert(m);
            }
        }
        p
    }

    #[test]
    fn test_example_from_spec() {
        // 0-indexed wires:
        // P0 = x_0
        // P1 = x_1 + x_0 + x_0*x_2
        // P2 = x_2
        // P3 = x_3
        // P4 = x_4
        // P5 = x_5 + x_3 + x_3*x_4
        let polys = vec![
            poly(&[&[0]]),
            poly(&[&[1], &[0], &[0, 2]]),
            poly(&[&[2]]),
            poly(&[&[3]]),
            poly(&[&[4]]),
            poly(&[&[5], &[3], &[3, 4]]),
        ];

        let (canonical, perm) = canonicalize_polys(polys);

        // Expected final order: P1, P5, P0, P2, P3, P4
        // data = [1, 5, 0, 2, 3, 4]
        assert_eq!(perm.data, vec![1, 5, 0, 2, 3, 4]);

        // Remap: wire1->x0, wire5->x1, wire0->x2, wire2->x3, wire3->x4, wire4->x5
        // canonical[0] = P1 remapped: x_1 + x_0 + x_0*x_2  ->  x_0 + x_2 + x_2*x_3
        assert_eq!(canonical[0], poly(&[&[0], &[2], &[2, 3]]));

        // canonical[1] = P5 remapped: x_5 + x_3 + x_3*x_4  ->  x_1 + x_4 + x_4*x_5
        assert_eq!(canonical[1], poly(&[&[1], &[4], &[4, 5]]));

        // canonical[2] = P0 remapped: x_0 -> x_2
        assert_eq!(canonical[2], poly(&[&[2]]));

        // canonical[3] = P2 remapped: x_2 -> x_3
        assert_eq!(canonical[3], poly(&[&[3]]));

        // canonical[4] = P3 remapped: x_3 -> x_4
        assert_eq!(canonical[4], poly(&[&[4]]));

        // canonical[5] = P4 remapped: x_4 -> x_5
        assert_eq!(canonical[5], poly(&[&[5]]));
    }

    #[test]
    fn test_single_poly() {
        let polys = vec![poly(&[&[0, 1]])];
        let (canonical, perm) = canonicalize_polys(polys);
        assert_eq!(perm.data, vec![0]);
        assert_eq!(canonical[0], poly(&[&[0, 1]]));
    }

    #[test]
    fn test_already_canonical() {
        let polys = vec![
            poly(&[&[0, 1]]),
            poly(&[&[1]]),
        ];
        let (canonical, perm) = canonicalize_polys(polys);
        assert_eq!(perm.data, vec![0, 1]);
        assert_eq!(canonical[0], poly(&[&[0, 1]]));
        assert_eq!(canonical[1], poly(&[&[1]]));
    }

    #[test]
    fn test_reverse_order() {
        // P0 = x_0 (degree 1), P1 = x_0*x_1 (degree 2) -> P1 should come first
        let polys = vec![
            poly(&[&[0]]),
            poly(&[&[0, 1]]),
        ];
        let (canonical, perm) = canonicalize_polys(polys);
        // data[0]=1, data[1]=0: position 0 pulls wire 1, position 1 pulls wire 0
        assert_eq!(perm.data, vec![1, 0]);
        // P1 remapped: wire1->x0, wire0->x1 => x_0*x_1 unchanged
        assert_eq!(canonical[0], poly(&[&[0, 1]]));
        // P0 remapped: x_0 -> x_1
        assert_eq!(canonical[1], poly(&[&[1]]));
    }

    #[test]
    fn test_gf2_cancellation_via_remap() {
        // Construct a case where after remap two monomials in the same polynomial
        // become identical and must cancel in GF(2).
        // This cannot happen with a bijective wire remap (which we always have),
        // but we verify the HashSet XOR logic is correct by testing a direct
        // polynomial with a duplicate monomial supplied at construction time.
        // poly(&[&[0], &[0]]) should be the zero polynomial (empty set).
        let zero = poly(&[&[0], &[0]]);
        assert!(zero.is_empty());
    }

    #[test]
    fn test_degree2_beats_two_degree1() {
        // x_0*x_1 should rank higher than x_0 + x_1 even though both have
        // one monomial at their respective max degrees. This catches the bug
        // where profile [1] (one deg-1) was incorrectly equal to [1,0] (one deg-2).
        let polys = vec![
            poly(&[&[0], &[1]]),  // P0 = x_0 + x_1  (max degree 1)
            poly(&[&[0, 1]]),     // P1 = x_0*x_1    (max degree 2)
        ];
        let (_canonical, perm) = canonicalize_polys(polys);
        assert_eq!(perm.data, vec![1, 0]); // P1 comes first
    }

    #[test]
    fn test_singleton_group_locked_early() {
        // Groups by degree: (P1 P2) at degree 2, (P0) at degree 1, (P3) at degree 0.
        // P3 should be locked into position 3 immediately even while P1/P2 are in tiebreak,
        // and P0 locked into position 2. Neither should be usable for comparisons until locked.
        // P1 = x_0*x_1, P2 = x_2*x_3 — symmetric, tiebreak via rule 6 -> P1 first
        // P0 = x_0 (degree 1), P3 = 1 (degree 0, the constant monomial)
        let polys = vec![
            poly(&[&[0]]),        // P0: x_0         degree 1
            poly(&[&[0, 1]]),     // P1: x_0*x_1     degree 2
            poly(&[&[2, 3]]),     // P2: x_2*x_3     degree 2
            poly(&[&[]]),         // P3: 1 (constant) degree 0
        ];
        let (_canonical, perm) = canonicalize_polys(polys);
        // P1 and P2 tie at degree 2 -> rule 6 picks P1 (index 1 < 2)
        // P0 at degree 1 -> position 2
        // P3 at degree 0 -> position 3
        assert_eq!(perm.data[2], 1); // position 2 = P1... wait, data[pos]=wire
        // data = [1, 2, 0, 3]: pos0=P1, pos1=P2, pos2=P0, pos3=P3
        assert_eq!(perm.data, vec![1, 2, 0, 3]);
    }

    #[test]
    fn test_disjoint_gates_twice() {
        // Gates are disjoint on wires 0-5. Test verified by hand
        // Test on two different ones. Should canonicalize to the same thing both times, with the same permutation.
        let mut rng = rand::rng();
        let mut pins: [u8; 6] = [0, 1, 2, 3, 4, 5];
        pins.shuffle(&mut rng);
        let circuit = CircuitSeq { gates: vec![
            [pins[0], pins[1], pins[2]], 
            [pins[3], pins[4], pins[5]]] 
        };
        let polys = circuit.to_polynomial(6, 0, 2);
        let (canonical, _) = canonicalize_polys(polys);
        println!("Canonical polys:");
        for (i, poly) in canonical.iter().enumerate() {
            println!("  P{}: {}", i, poly_to_str(poly, 6));
        }
    }

    #[test]
    fn test_random_circuit_canonicalization() {
        use crate::random::random_data::random_circuit;
        let timer = std::time::Instant::now();
        for _ in 0..100_000 {
            let circuit = random_circuit(30, 20);
            let polys = circuit.to_polynomial(30, 0, 20);
            // for (i, poly) in polys.iter().enumerate() {
            //     println!("  P{}: {}", i, poly_to_str(poly, 30));
            // }
            let (canonical, _) = canonicalize_polys(polys);
            // println!("Canonical polys:");
            // for (i, poly) in canonical.iter().enumerate() {
            //     println!("  P{}: {}", i, poly_to_str(poly, 30));
            // }
        }
        println!("Total time for 100,000 random circuits: {:.2?}", timer.elapsed());
    }

    #[test]
    fn test_shuffled_canonicalization() {
        use crate::random::random_data::random_circuit;
        let mut rng = rand::rng();
        for n in 8..30 {
            println!("Testing n={} wires", n);
            for _ in 0..10_000 {
                let circuit = random_circuit(n, 20);
                let old_circuit = circuit.clone();
                let polys = old_circuit.to_polynomial(n, 0, 20);
                let (canonical, _) = canonicalize_polys(polys.clone());
                let canon_string = canonical.iter().enumerate()
                    .map(|(i, poly)| format!("P{}: {}", i, poly_to_str(poly, n)))
                    .collect::<Vec<_>>()
                    .join("\n");
                for _ in 0..100 {
                    let mut pins: Vec<usize> = (0..n).collect();
                    pins.shuffle(&mut rng);
                    let mut shuffled_circuit = circuit.clone();
                    shuffled_circuit.rewire(&Permutation { data: pins }, n);
                    let shuffled_polys = shuffled_circuit.to_polynomial(n, 0, 20);
                    let (shuffled_canonical, _) = canonicalize_polys(shuffled_polys);
                    let shuffled_string = shuffled_canonical.iter().enumerate()
                        .map(|(i, poly)| format!("P{}: {}", i, poly_to_str(poly, n)))
                        .collect::<Vec<_>>()
                        .join("\n");
                    assert_eq!(canon_string, shuffled_string,
                        "\nOriginal polys:\n{}\n",
                        polys.iter().enumerate()
                            .map(|(i, poly)| format!("P{}: {}", i, poly_to_str(poly, n)))
                            .collect::<Vec<_>>()
                            .join("\n")
                    );
                }
            }
        }
    }

    use std::fs;
    use std::fs::File;
    use std::io::Write;
    #[test]
    pub fn test_canonicalization() {
        let contents = fs::read_to_string("before_canon.txt")
            .expect("Failed to read");
        let mut circuit_a = CircuitSeq::from_string(&contents);

        // Proceed as before
        circuit_a.canonicalize();

        let c_str = circuit_a.repr();
        File::create("after_canon.txt")
            .and_then(|mut f| f.write_all(c_str.as_bytes()))
            .expect("Failed to write test_compression.txt");
    }

    #[test]
    pub fn test_probably_shuffle() {
        use rand::Rng;
        use crate::random::random_data::random_circuit;

        let mut rng = rand::rng();
        let shuffle_circuit = random_circuit(9, 15);
        let mut base_circuit = shuffle_circuit.clone();
        let bit_shuf_list: Vec<Vec<Vec<usize>>> = (3..=7)
        .map(|n| {
            (0..n)
                .permutations(n)
                .filter(|p| !p.iter().enumerate().all(|(i, &x)| i == x))
                .collect::<Vec<Vec<usize>>>()
        })
        .collect();
        let three_wire = &bit_shuf_list[0]; 
        let shuf = &three_wire[rng.random_range(0..three_wire.len())];
        base_circuit.rewire(&Permutation { data: shuf.clone() }, 3);
        assert!(base_circuit.is_relabeling_of(&shuffle_circuit) == true);
    }
}
