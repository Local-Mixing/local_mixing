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
        state ^ (c1 & ((!c2) & 1)) << gate[0]
    }

    // Evaluate up to 128 bits
    #[inline(always)]
    pub fn evaluate_index_128(state: u128, gate: [u8;3]) -> u128 {
        let c1 = (state >> gate[1]) & 1;
        let c2 = (state >> gate[2]) & 1;
        state ^ (c1 & ((!c2) & 1)) << gate[0]
    }

    // Evaluate up to 256 bits
    #[inline(always)]
    pub fn evaluate_index_256(state: u256, gate: [u8;3]) -> u256 {
        let one = u256::one();
        let c1 = (state >> gate[1]) & one;
        let c2 = (state >> gate[2]) & one;
        state ^ ((c1 & (one ^ c2)) << gate[0])
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

    pub fn to_polynomial(self, n: usize, start: usize, end: usize) -> Vec<Polynomial> {
        let gates = &self.gates[start..end];
        // Wire i starts as degree 1 monomial
        let mut polys: Vec<Polynomial> = (0..n)
        .map(|i| HashSet::from([1u64 << i]))
        .collect();
    
        for &[a, b, c] in gates {
            // a' = a XOR (b AND NOT c)
            let not_c = poly_not(polys[c as usize].clone());
            let term = poly_and(&polys[b as usize], &not_c);
            let new_a = poly_xor(polys[a as usize].clone(), term);
            polys[a as usize] = new_a;
        }
    
        // XOR each wire with its initial value x_i so unchanged wires become 0
        for i in 0..n {
            let xi = HashSet::from([1u64 << i]);
            polys[i] = poly_xor(polys[i].clone(), xi);
        }

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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_from_string() {
        let s = "lej;egu;gdt;pt0;vib;cp8;tes;8pt;p0h;h59;cl7;d7e;agc;laq;u39;pog;u39;laq;d7e;pog;agc;cl7;h59;p0h;8pt;cp8;tes;vib;pt0;032;324;024;213;pt0;vib;cp8;tes;8pt;p0h;h59;d7e;cl7;u39;pog;p8h;sd6;783;agc;la9;laq;nlc;9j2;nlc;783;sd6;9j2;u39;341;213;132;432;la9;p8h;pog;laq;agc;d7e;cl7;h59;p0h;8pt;cl7;d7e;agc;laq;8pt;p0h;h59;p8h;pog;p8h;pog;u39;laq;agc;d7e;cl7;h59;p0h;8pt;tes;cp8;tes;cp8;8pt;p0h;h59;d7e;pog;p8h;cl7;agc;la9;laq;la9;p8h;pog;laq;agc;d7e;cl7;h59;p0h;8pt;tes;cp8;vib;pt0;vib;pt0;cp8;tes;8pt;p0h;cl7;d7e;pog;h59;7hu;p8h;sd6;783;agc;la9;laq;504;tsi;7hu;tsi;504;783;u39;314;sd6;la9;p8h;pog;laq;agc;d7e;cl7;h59;p0h;8pt;tes;cp8;vib;pt0;gdt;132;gdt;pt0;vib;cp8;tes;8pt;vib;8pt;tes;cp8;pt0;gdt;egu;lej;031;lej;egu;pt0;vib;cp8;gdt;tes;d7e;8pt;p0h;h59;cl7;d7e;cl7;h59;p0h;8pt;tes;cp8;vib;pt0;gdt;egu;lej;132;lej;egu;gdt;pt0;vib;cp8;tes;8pt;p0h;h59;cl7;d7e;u39;pog;p8h;sd6;783;agc;la9;laq;nlc;9j2;nlc;504;tsi;7hu;tsi;504;9j2;783;sd6;la9;p8h;pog;laq;agc;7hu;u39;h59;d7e;cl7;h59;cl7;d7e;agc;laq;314;u39;pog;la9;sd6;783;nlc;504;783;sd6;pog;nlc;9j2;504;9j2;u39;la9;laq;agc;d7e;cl7;h59;p0h;vib;8pt;tes;cp8;vib;cp8;tes;cl7;d7e;agc;laq;u39;8pt;p0h;h59;pog;la9;sd6;783;p8h;7hu;783;sd6;la9;pog;laq;agc;7hu;u39;d7e;cl7;p8h;h59;p0h;8pt;tes;cp8;tes;cl7;d7e;cp8;agc;8pt;laq;pog;la9;sd6;783;p0h;h59;p8h;u39;432;132;213;504;nlc;504;nlc;783;sd6;la9;p8h;pog;u39;341;laq;agc;d7e;cl7;h59;p0h;8pt;tes;cp8;vib;pt0;gdt;pt0;vib;cp8;gdt;d7e;tes;sd6;8pt;p0h;h59;cl7;agc;laq;u39;pog;p8h;la9;783;9j2;nlc;504;tsi;504;nlc;9j2;783;tsi;sd6;p8h;pog;u39;d7e;la9;laq;agc;cl7;h59;p0h;8pt;cp8;tes;vib;pt0;vib;032;213;024;pt0;cp8;tes;8pt;p0h;h59;cl7;d7e;agc;laq;u39;pog;p8h;la9;pog;u39;324;laq;d7e;la9;agc;cl7;p8h;h59;cl7;d7e;agc;laq;u39;pog;sd6;783;h59;p8h;504;la9;9j2;nlc;504;nlc;9j2;783;sd6;la9;p8h;pog;u39;laq;agc;d7e;cl7;h59;p0h;8pt;tes;cp8;vib;pt0;gdt;egu;lej;032;lej;egu;gdt;pt0;vib;tes;cp8;8pt;cl7;d7e;783;agc;laq;u39;pog;la9;sd6;p0h;h59;p8h;504;9j2;nlc;504;nlc;9j2;783;sd6;la9;p8h;pog;u39;laq;agc;d7e;cl7;h59;p0h;8pt;tes;cp8;vib;pt0;gdt;egu;lej;";
        println!("{:?}", CircuitSeq::from_string(s).gates);
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
}
