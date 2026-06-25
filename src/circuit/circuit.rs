// Basic implementation for circuit, gate, and permutations
use primitive_types::U256 as u256;
use primitive_types::U512 as u512;
use rand::{RngCore, seq::SliceRandom};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering as CmpOrdering;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Instant;

use std::collections::BTreeMap;

pub static CANON4_CORE_TIME: AtomicU64 = AtomicU64::new(0);
pub static POLYCANON_CORE_TIME: AtomicU64 = AtomicU64::new(0);
pub static CANON_BENCH_CALLS: AtomicU64 = AtomicU64::new(0);
pub static CANON4_RULE_L_TIME: AtomicU64 = AtomicU64::new(0);
pub static CANON4_RULE_L_CALLS: AtomicU64 = AtomicU64::new(0);
pub static CANON4_RULE_L_BRANCHES: AtomicU64 = AtomicU64::new(0);

fn bench_canon_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("BENCH_CANON").is_ok())
}

fn compression_trace_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("COMPRESSION_TRACE").is_ok())
}

fn compression_trace_threshold_ms() -> u128 {
    static THRESHOLD: OnceLock<u128> = OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        std::env::var("COMPRESSION_TRACE_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1_000)
    })
}

// Gate [a, pos_ctrl, neg_ctrl]: flip a UNLESS neg_ctrl=1 AND NOT pos_ctrl
// (flips when pos_ctrl=1 OR neg_ctrl=0)
// We are only concerned with gate g57
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Gate {
    pub pins: [usize; 3], //one active wire (0) and two control wires (1,2)
}

// Circuits stored as a sequence of gates [u16;3]
// Gate type is legacy
#[derive(Clone, Debug, Default, Serialize, Deserialize, Eq, Hash, PartialEq)]
pub struct CircuitSeq {
    pub gates: Vec<[u16; 3]>,
}

// Polynomial representation of circuit
pub type Monomial = u64;
pub type Polynomial = Vec<Monomial>;

// Permutations are all the possible outputs of a circuit
// On n wires permutation length is 1 << n
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Permutation {
    pub data: Vec<usize>,
}

// Functions on Gate struct and [u8;3]
impl Gate {
    // Gates collide iff either active pin shares a wire with any other pin
    pub fn collides_index(gate: &[u16; 3], other: &[u16; 3]) -> bool {
        gate[0] == other[1] || gate[0] == other[2] || gate[1] == other[0] || gate[2] == other[0]
    }

    //b is "larger"
    pub fn ordered_index(gate: &[u16; 3], other: &[u16; 3]) -> bool {
        if gate[0] > other[0] {
            return false;
        } else if gate[0] == other[0] {
            if gate[1] > other[1] {
                return false;
            } else if gate[1] == other[1] {
                return gate[2] < other[2];
            }
        }
        true
    }

    // Evaluate a bit string after a single gate under gate r57
    #[inline(always)]
    pub fn evaluate_index(state: usize, gate: [u16; 3]) -> usize {
        let c1 = (state >> gate[1]) & 1;
        let c2 = (state >> gate[2]) & 1;
        state ^ (c1 | ((!c2) & 1)) << gate[0]
    }

    // Evaluate up to 256 bits
    #[inline(always)]
    pub fn evaluate_index_256(state: u256, gate: [u16; 3]) -> u256 {
        let one = u256::one();
        let c1 = (state >> gate[1]) & one;
        let c2 = (state >> gate[2]) & one;
        state ^ ((c1 | (one ^ c2)) << gate[0])
    }

    pub fn evaluate_index_512(state: u512, gate: [u16; 3]) -> u512 {
        let one = u512::one();
        let c1 = (state >> gate[1]) & one;
        let c2 = (state >> gate[2]) & one;
        state ^ ((c1 | (one ^ c2)) << gate[0])
    }

    // Evaluate a list of gates
    #[inline(always)]
    pub fn evaluate_index_list(state: usize, gates: &Vec<[u16; 3]>) -> usize {
        let mut current_wires = state;
        for g in gates {
            current_wires = Self::evaluate_index(current_wires, *g);
        }
        current_wires
    }

    #[inline(always)]
    pub fn evaluate_index_list_256(state: u256, gates: &Vec<[u16; 3]>) -> u256 {
        let mut current_wires = state;
        for g in gates {
            current_wires = Self::evaluate_index_256(current_wires, *g);
        }
        current_wires
    }

    #[inline(always)]
    pub fn evaluate_index_list_512(state: u512, gates: &Vec<[u16; 3]>) -> u512 {
        let mut current_wires = state;
        for g in gates {
            current_wires = Self::evaluate_index_512(current_wires, *g);
        }
        current_wires
    }
}

impl Permutation {
    pub fn new(data: Vec<usize>) -> Permutation {
        Permutation { data }
    }

    // Compose two permutations: (self ∘ other)[i] = self[other[i]].
    pub fn compose(&self, other: &Permutation) -> Permutation {
        if self.data.len() != other.data.len() {
            panic!("Permutation length mismatch in compose");
        }
        let data = (0..self.data.len())
            .map(|i| self.data[other.data[i]])
            .collect();
        Permutation { data }
    }
    pub fn is_perm(&self) -> bool {
        let mut temp_perm = self.clone();
        temp_perm.data.sort_unstable();
        temp_perm == Permutation::id_perm(self.data.len())
    }

    pub fn id_perm(n: usize) -> Permutation {
        let temp_data = (0..n).collect();
        Permutation { data: temp_data }
    }

    // n is the length of the permutation. For a random permutation on n bits, do 1 << n
    pub fn rand_perm(n: usize) -> Permutation {
        let mut p = Permutation::id_perm(n);
        let mut rng = rand::rng();
        p.data.shuffle(&mut rng);
        p
    }

    pub fn invert(&self) -> Permutation {
        let mut inv = vec![0; self.data.len()];
        self.data
            .iter()
            .enumerate()
            .for_each(|(i, &val)| inv[val] = i);
        Permutation { data: inv }
    }

    // string representation is just the elements of the permutation separated by a ,
    pub fn repr(&self) -> String {
        self.data
            .iter()
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
    // Evaluate the entire circuit with a starting input
    pub fn evaluate(&self, input: usize) -> usize {
        Gate::evaluate_index_list(input, &self.gates)
    }

    // Evaluate the circuit on a 256-bit input state (one bit per wire).
    pub fn evaluate_256(&self, input: u256) -> u256 {
        Gate::evaluate_index_list_256(input, &self.gates)
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
        let gates: Vec<[u16; 3]> = blob
            .chunks(3)
            .map(|chunk| [chunk[0] as u16, chunk[1] as u16, chunk[2] as u16])
            .collect();
        CircuitSeq { gates }
    }

    // Rewire wire i -> perm[i]
    pub fn rewire(&mut self, perm: &Permutation, n: usize) {
        if perm.data.is_empty() {
            return;
        }

        if perm.data.len() != n {
            panic!("wrong size perm! got {}, have {} wires", perm.data.len(), n);
        }

        if !perm.is_perm() {
            panic!("{:?} is not a permutation!", perm);
        }

        for gate in &mut self.gates {
            *gate = [
                perm.data[gate[0] as usize] as u16,
                perm.data[gate[1] as usize] as u16,
                perm.data[gate[2] as usize] as u16,
            ];
        }
    }

    /// Remap to minimal wires: e.g. [3,7,11] -> [0,1,2].
    /// Returns (remapped_circuit, used_wires) where `used` allows unrewiring back to original wires.
    pub fn rewire_min(&self) -> (CircuitSeq, Vec<u16>) {
        let used = self.used_wires();
        let wire_map: HashMap<u16, u16> = used
            .iter()
            .enumerate()
            .map(|(i, &w)| (w, i as u16))
            .collect();
        let remapped = CircuitSeq {
            gates: self
                .gates
                .iter()
                .map(|&[t, c1, c2]| [wire_map[&t], wire_map[&c1], wire_map[&c2]])
                .collect(),
        };
        (remapped, used)
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
                '0'..='9' => c as u8 - b'0',      // 0-9
                'a'..='z' => c as u8 - b'a' + 10, // 10-35
                'A'..='Z' => c as u8 - b'A' + 36, // 36-61
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

        let gates: Vec<[u16; 3]> = s
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
                    wires.push(wire as u16);
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
        let wire_map_chars: Vec<char> =
            "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_=+[]{}<>?"
                .chars()
                .collect();

        // --- Pretty circuit diagram ---
        for wire in 0..num_wires {
            result += &format!("{:<2} --", wire);
            for gate in &self.gates {
                if gate[0] == wire as u16 {
                    result += "( )";
                } else if gate[1] == wire as u16 {
                    result += "-●-";
                } else if gate[2] == wire as u16 {
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
                    .map(|&x| wire_map_chars.get(x as usize).unwrap_or(&'?').to_string())
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
    pub fn used_wires(&self) -> Vec<u16> {
        let mut used: HashSet<u16> = HashSet::new();
        for gates in &self.gates {
            used.insert(gates[0]);
            used.insert(gates[1]);
            used.insert(gates[2]);
        }
        let mut wires: Vec<u16> = used.into_iter().collect();
        wires.sort();
        wires
    }

    // "Bottom" function for gates
    pub fn max_wire(&self) -> usize {
        self.gates.iter().flatten().copied().max().unwrap_or(0) as usize
    }

    // Undo rewiring. Note: Recall that the number of wires in CircuitSeq is not stored
    pub fn unrewire_subcircuit(subcircuit: &CircuitSeq, used_wires: &[u16]) -> CircuitSeq {
        // Build a mapping from new wire -> original wire
        let wire_map: HashMap<u16, u16> = used_wires
            .iter()
            .enumerate()
            .map(|(new_idx, &orig_wire)| (new_idx as u16, orig_wire))
            .collect();

        // Replace wires in each gate with original wires
        let new_gates: Vec<[u16; 3]> = subcircuit
            .gates
            .iter()
            .map(|&[t, c1, c2]| {
                [
                    *wire_map.get(&t).unwrap(),
                    *wire_map.get(&c1).unwrap(),
                    *wire_map.get(&c2).unwrap(),
                ]
            })
            .collect();

        CircuitSeq { gates: new_gates }
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
        num_inputs: usize,
    ) -> Result<(), String> {
        use rayon::prelude::*;

        if num_wires > 256 {
            let mask = if num_wires < 512 {
                (u512::one() << num_wires) - u512::one()
            } else {
                u512::MAX
            };
            return (0..num_inputs).into_par_iter().try_for_each(|_| {
                let mut bytes = [0u8; 64];
                rand::rng().fill_bytes(&mut bytes);
                let random_input = u512::from_little_endian(&bytes) & mask;
                let self_output = Gate::evaluate_index_list_512(random_input, &self.gates);
                let other_output =
                    Gate::evaluate_index_list_512(random_input, &other_circuit.gates);
                if (self_output & mask) != (other_output & mask) {
                    Err("Circuits are not equal".to_string())
                } else {
                    Ok(())
                }
            });
        }

        let mask = if num_wires < 256 {
            (u256::one() << num_wires) - u256::one()
        } else {
            u256::MAX
        };

        (0..num_inputs).into_par_iter().try_for_each(|_| {
            let mut bytes = [0u8; 32];
            rand::rng().fill_bytes(&mut bytes);
            let random_input = u256::from_little_endian(&bytes) & mask;

            let self_output = Gate::evaluate_index_list_256(random_input, &self.gates);
            let other_output = Gate::evaluate_index_list_256(random_input, &other_circuit.gates);

            if (self_output & mask) != (other_output & mask) {
                Err("Circuits are not equal".to_string())
            } else {
                Ok(())
            }
        })
    }

    pub fn to_polynomial(&self, n: usize, start: usize, end: usize) -> Vec<Polynomial> {
        let gates = &self.gates[start..end];
        // Wire i starts as degree 1 monomial
        let mut polys: Vec<Polynomial> = (0..n).map(|i| vec![1u64 << i]).collect();

        for &[a, b, c] in gates {
            // a' = a + bc + b + 1 = a + b(c+1) = a + b*NOT(c) + 1
            let term = poly_and_not(&polys[b as usize], &polys[c as usize]);
            poly_xor_assign(&mut polys[a as usize], term);
            toggle_monomial(&mut polys[a as usize], 0u64);
        }

        // XOR each wire with its initial value x_i so unchanged wires become 0
        // for i in 0..n {
        //     let xi = vec![1u64 << i];
        //     polys[i] = poly_xor(polys[i].clone(), xi);
        // }

        polys
    }

    // Returns (canonical_polys, canonical_circuit, reversed)
    // where reversed=true means the reversed circuit produced the canonical form.
    pub fn canonicalize_polys(
        &self,
        _n: usize,
        allow_rule_l: bool,
    ) -> Option<(Vec<Polynomial>, CircuitSeq, bool, Permutation, Vec<u16>)> {
        fn poly_vec_key(polys: &Vec<Polynomial>) -> Vec<Vec<u64>> {
            polys
                .iter()
                .map(|p| {
                    let mut v: Vec<u64> = p.iter().copied().collect();
                    v.sort();
                    v
                })
                .collect()
        }
        // Remap to minimal wires: e.g. [3,7,11] -> [0,1,2].
        // `used` is returned so callers can unrewire canonical circuits back to original wires.
        let used = self.used_wires();
        let wire_map: HashMap<u16, u16> = used
            .iter()
            .enumerate()
            .map(|(i, &w)| (w, i as u16))
            .collect();
        let remapped = CircuitSeq {
            gates: self
                .gates
                .iter()
                .map(|&[t, c1, c2]| [wire_map[&t], wire_map[&c1], wire_map[&c2]])
                .collect(),
        };
        let mut c1 = remapped.clone();
        c1.canonicalize();
        let mut c2 = remapped.clone();
        c2.gates.reverse();
        c2.canonicalize();
        let n1 = c1.max_wire() as usize + 1;
        let n2 = c2.max_wire() as usize + 1;
        let polys_fwd = c1.to_polynomial(n1, 0, c1.gates.len());
        let polys_rev = c2.to_polynomial(n2, 0, c2.gates.len());
        let canon1 = canonicalize_polys_4(polys_fwd, allow_rule_l).ok()?;
        let canon2 = canonicalize_polys_4(polys_rev, allow_rule_l).ok()?;
        c1.rewire(&canon1.1.invert(), n1);
        c1.canonicalize();
        c2.rewire(&canon2.1.invert(), n2);
        c2.canonicalize();
        // final_order.data[canonical_pos] = wire in the dense remapped space (0..k-1).
        // To unrewire a canonical circuit back to original wires, apply final_order first
        // (canonical → dense), then apply `used` (dense → original).
        Some(if poly_vec_key(&canon1.0) < poly_vec_key(&canon2.0) {
            (canon1.0, c1, false, canon1.1, used)
        } else if poly_vec_key(&canon1.0) > poly_vec_key(&canon2.0) {
            (canon2.0, c2, true, canon2.1, used)
        } else if c1.gates <= c2.gates {
            (canon1.0, c1, false, canon1.1, used)
        } else {
            (canon2.0, c2, true, canon2.1, used)
        })
    }

    /// Compute canonical polynomials for one direction only (forward or reversed).
    /// Returns (canonical_polys, final_order, used_wires).
    /// Used by compress_lmdb to try forward first, then reverse on miss.
    pub fn canonicalize_polys_single(
        &self,
        reversed: bool,
    ) -> (Vec<Polynomial>, Permutation, Vec<u16>) {
        let used = self.used_wires();
        let wire_map: HashMap<u16, u16> = used
            .iter()
            .enumerate()
            .map(|(i, &w)| (w, i as u16))
            .collect();
        let mut c = CircuitSeq {
            gates: self
                .gates
                .iter()
                .map(|&[t, c1, c2]| [wire_map[&t], wire_map[&c1], wire_map[&c2]])
                .collect(),
        };
        if reversed {
            c.gates.reverse();
        }
        c.canonicalize();
        let n = c.max_wire() as usize + 1;
        let polys = c.to_polynomial(n, 0, c.gates.len());

        let bench_polys = if bench_canon_enabled() {
            Some(polys.clone())
        } else {
            None
        };

        let t4 = Instant::now();
        let canon = canonicalize_polys_4(polys, true).unwrap();
        let canon_elapsed = t4.elapsed();
        CANON4_CORE_TIME.fetch_add(canon_elapsed.as_nanos() as u64, Ordering::Relaxed);
        if compression_trace_enabled()
            && canon_elapsed.as_millis() >= compression_trace_threshold_ms()
        {
            eprintln!(
                "[compress-trace] slow canonicalize direction={} gates={} used_wires={} elapsed_ms={}",
                if reversed { "reverse" } else { "forward" },
                self.gates.len(),
                used.len(),
                canon_elapsed.as_millis()
            );
        }

        if let Some(polys) = bench_polys {
            let tp = Instant::now();
            let perm = crate::circuit::poly_canon_graph::canonicalize_graph(&polys, n);
            let _form = crate::circuit::poly_canon_graph::canonical_form(&polys, &perm);
            POLYCANON_CORE_TIME.fetch_add(tp.elapsed().as_nanos() as u64, Ordering::Relaxed);
            CANON_BENCH_CALLS.fetch_add(1, Ordering::Relaxed);
        }

        (canon.0, canon.1, used)
    }
}

pub fn polynomial_from_terms<I>(terms: I) -> Polynomial
where
    I: IntoIterator<Item = Monomial>,
{
    let mut terms: Vec<Monomial> = terms.into_iter().collect();
    normalize_polynomial(&mut terms);
    terms
}

pub fn normalize_polynomial(poly: &mut Polynomial) {
    poly.sort_unstable();

    let mut write = 0usize;
    let mut read = 0usize;
    while read < poly.len() {
        let m = poly[read];
        let mut count = 1usize;
        read += 1;
        while read < poly.len() && poly[read] == m {
            count += 1;
            read += 1;
        }
        if count % 2 == 1 {
            poly[write] = m;
            write += 1;
        }
    }
    poly.truncate(write);
}

fn toggle_monomial(poly: &mut Polynomial, m: Monomial) {
    match poly.binary_search(&m) {
        Ok(pos) => {
            poly.remove(pos);
        }
        Err(pos) => {
            poly.insert(pos, m);
        }
    }
}

fn poly_xor_assign(poly: &mut Polynomial, terms: Polynomial) {
    let old = std::mem::take(poly);
    let mut merged = Vec::with_capacity(old.len().max(terms.len()));
    let mut i = 0usize;
    let mut j = 0usize;

    while i < old.len() && j < terms.len() {
        match old[i].cmp(&terms[j]) {
            CmpOrdering::Less => {
                merged.push(old[i]);
                i += 1;
            }
            CmpOrdering::Greater => {
                merged.push(terms[j]);
                j += 1;
            }
            CmpOrdering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    merged.extend_from_slice(&old[i..]);
    merged.extend_from_slice(&terms[j..]);
    *poly = merged;
}

fn poly_and_not(poly_1: &Polynomial, poly_2: &Polynomial) -> Polynomial {
    let mut terms = Vec::with_capacity(poly_1.len() * (poly_2.len() + 1));
    for &m1 in poly_1 {
        terms.push(m1);
        for &m2 in poly_2 {
            terms.push(m1 | m2);
        }
    }
    polynomial_from_terms(terms)
}

// Display polynomials

pub fn polys_repr_blob(polys: &Vec<Polynomial>) -> Vec<u8> {
    let mut bytes = Vec::new();
    for poly in polys {
        let mut monomials: Vec<u64> = poly.iter().copied().collect();
        monomials.sort_unstable();
        for m in monomials {
            bytes.extend_from_slice(&m.to_le_bytes());
        }
        bytes.extend_from_slice(&u64::MAX.to_le_bytes()); // separator
    }
    bytes
}

// Possible gates on n wires
pub fn base_gates(n: usize) -> Vec<[u16; 3]> {
    let n = n as u16;
    let mut gates: Vec<[u16; 3]> = Vec::new();
    for a in 0..n {
        for b in 0..n {
            if b == a {
                continue;
            }
            for c in 0..n {
                if c == a || c == b {
                    continue;
                }
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

/// Used in tie-breaking.
/// For a given polynomial, return a degree-bucketed count (high to low) of
/// how many monomials of each degree contain variable `wire_idx`.
fn wire_counts_in_poly(
    poly: &Polynomial,
    max_possible_degree: usize,
    wire_idx: usize,
) -> Vec<usize> {
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

/// Filter a polynomial to only monomials containing variable `filter_var`.
/// Used in Rules 2.3 and 2.5 to restrict scoring to monomials of a given variable.
fn filter_poly_by_var(poly: &Polynomial, filter_var: usize) -> Polynomial {
    let bit = 1u64 << filter_var;
    poly.iter().copied().filter(|m| m & bit != 0).collect()
}

/// Canonicalize a polynomial vector for comparison purposes.
/// Remaps wire indices according to a given order and sorts each polynomial's monomials.
/// Used in Rule L (backtracking) to compare trial canonical forms.
fn make_canonical_form(polynomials: &[Polynomial], final_order: &[usize]) -> Vec<Vec<Monomial>> {
    let remap_monomial = |m: Monomial| -> Monomial {
        let mut result = 0u64;
        for (pos, &wire) in final_order.iter().enumerate() {
            if m & (1u64 << wire) != 0 {
                result |= 1u64 << pos;
            }
        }
        result
    };
    final_order
        .iter()
        .map(|&wire| {
            let mut remapped: Vec<Monomial> = polynomials[wire]
                .iter()
                .map(|&m| remap_monomial(m))
                .collect();
            remapped.sort();
            remapped
        })
        .collect()
}

/// Partition a list of (index, score) pairs into sub-groups by descending score.
/// Indices with equal scores remain in the same sub-group (still tied).
fn split_by_scores(mut scored: Vec<(usize, Vec<usize>)>) -> Vec<Vec<usize>> {
    scored.sort_by(|a, b| b.1.cmp(&a.1));
    let mut result: Vec<Vec<usize>> = Vec::new();
    let mut current = vec![scored[0].0];
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

/// Pack a monomial's sort key into a single u64 for fast comparison.
/// Format: [4 bits degree | 4 bits rank_var1 | 4 bits rank_var2 | ...]
/// Degree is stored inverted (15 - degree) so higher degree -> higher u64.
/// Variable ranks are stored inverted (15 - rank) so lower rank value -> higher u64.
/// This makes descending u64 order = descending monomial rank.
fn monomial_sort_key_u64(m: Monomial, var_rank: &[usize], n: usize) -> u64 {
    let degree = m.count_ones() as usize;
    // Collect ranks of present variables, sorted ascending (best rank first)
    let mut var_ranks: Vec<usize> = (0..n)
        .filter(|&i| m & (1u64 << i) != 0)
        .map(|i| var_rank[i])
        .collect();
    var_ranks.sort();

    // Pack into u64: high bits = inverted degree, then inverted ranks
    // Using 4 bits per field, max 15 variables/degree
    let mut key = ((15 - degree.min(15)) as u64) << 60;
    for (slot, &r) in var_ranks.iter().enumerate() {
        let shift = 56 - slot * 4;
        if shift < 64 {
            key |= ((15 - r.min(15)) as u64) << shift;
        }
    }
    key
}

// Static accumulators for time spent in each rule (in nanoseconds)
static TIME_RULE_2_1: AtomicU64 = AtomicU64::new(0);
static TIME_RULE_2_2: AtomicU64 = AtomicU64::new(0);
static TIME_RULE_2_4: AtomicU64 = AtomicU64::new(0);
static TIME_RULE_2_5: AtomicU64 = AtomicU64::new(0);
static TIME_RULE_L: AtomicU64 = AtomicU64::new(0);

/// Holds the current partial ordering of polynomial/variable indices as a list of groups.
/// Each group is a Vec<usize> of indices that are currently tied with each other.
/// Singletons (len == 1) are fully ranked. The position of a group in `groups`
/// reflects its rank relative to other groups (index 0 = highest rank).
struct RankingState {
    /// groups[i] = the i-th ranked group (singleton or tied).
    /// Singletons are fully resolved; tied groups are still being processed.
    groups: Vec<Vec<usize>>,
    /// Total number of polynomials/variables.
    n: usize,
}

impl RankingState {
    fn new(groups: Vec<Vec<usize>>, n: usize) -> Self {
        RankingState { groups, n }
    }

    /// Is every group a singleton? If so, ranking is complete.
    fn is_fully_ranked(&self) -> bool {
        self.groups.iter().all(|g| g.len() == 1)
    }

    /// Replace group at index `gi` with the given split sub-groups.
    fn apply_split(&mut self, gi: usize, split: Vec<Vec<usize>>) {
        self.groups.splice(gi..=gi, split);
    }

    /// Build var_rank array from current groups.
    /// All variables in the same tied group receive the same rank value.
    /// Rank 0 = highest priority (first group). Each group advances the rank
    /// counter by its size, leaving room for future splits within the group.
    fn current_var_rank(&self) -> Vec<usize> {
        let mut var_rank = vec![0usize; self.n];
        let mut rank = 0usize;
        for group in &self.groups {
            for &w in group {
                var_rank[w] = rank;
            }
            // All members of this group share `rank`. Advance by group.len()
            // so the next group gets a strictly higher rank value, leaving
            // room for future splits within this group.
            rank += group.len();
        }
        var_rank
    }

    /// Rule 2.1: for each tied group G_j, iterate all groups G_i in rank order
    /// (singletons and non-singletons alike). Score each polynomial P in G_j by
    /// summing wire_counts_in_poly over all variables in G_i:
    ///   score(P) = sum over var in G_i of wire_counts_in_poly(P, max_degree, var)
    /// When G_i is a singleton {x_k} this reduces to wire_counts_in_poly(P, max_degree, k).
    /// When G_i is a tied group this sums over all variables in the group.
    /// If any split occurs, record it and return true to restart from 2.1.
    fn try_rule_2_1(&mut self, polynomials: &[Polynomial], max_degree: usize) -> bool {
        for gi in 0..self.groups.len() {
            if self.groups[gi].len() <= 1 {
                continue;
            }
            let group = self.groups[gi].clone();

            // Iterate all groups in rank order as scoring sources
            for rgi in 0..self.groups.len() {
                let scoring_group = self.groups[rgi].clone();

                // Score each polynomial P in G_j by summing wire counts over all
                // variables in the scoring group
                let scored: Vec<(usize, Vec<usize>)> = group
                    .iter()
                    .map(|&p| {
                        let score = scoring_group.iter().fold(
                            vec![0usize; max_degree + 1],
                            |mut acc, &var| {
                                let counts = wire_counts_in_poly(&polynomials[p], max_degree, var);
                                for (a, c) in acc.iter_mut().zip(counts.iter()) {
                                    *a += c;
                                }
                                acc
                            },
                        );
                        (p, score)
                    })
                    .collect();

                let split = split_by_scores(scored);
                if split.len() > 1 {
                    self.apply_split(gi, split);
                    return true;
                }
            }
        }
        false
    }

    /// Rule 2.2: for each tied group G_j, iterate all groups G_i in rank order.
    /// Score each variable x_j in the tied group by summing wire_counts_in_poly
    /// over all P in G_i:
    ///   score(x_j) = sum over P in G_i of wire_counts_in_poly(P, max_degree, j)
    /// Variable ranking directly becomes polynomial ranking (poly index == var index).
    /// If any split occurs, record it and return true to restart from 2.1.
    fn try_rule_2_2(&mut self, polynomials: &[Polynomial], max_degree: usize) -> bool {
        for gi in 0..self.groups.len() {
            if self.groups[gi].len() <= 1 {
                continue;
            }
            let group = self.groups[gi].clone();

            // Iterate all groups in rank order as scorers
            for rgi in 0..self.groups.len() {
                let scoring_group = self.groups[rgi].clone();

                // Score each variable x_j in the tied group using scoring_group
                let scored: Vec<(usize, Vec<usize>)> = group
                    .iter()
                    .map(|&w| {
                        let score = scoring_group.iter().fold(
                            vec![0usize; max_degree + 1],
                            |mut acc, &p| {
                                let counts = wire_counts_in_poly(&polynomials[p], max_degree, w);
                                for (a, c) in acc.iter_mut().zip(counts.iter()) {
                                    *a += c;
                                }
                                acc
                            },
                        );
                        (w, score)
                    })
                    .collect();

                let split = split_by_scores(scored);
                if split.len() > 1 {
                    self.apply_split(gi, split);
                    return true;
                }
            }
        }
        false
    }

    // /// Rule 2.3: for each tied group G_j, iterate all groups G_k in rank order as
    // /// filter sources. For each G_k, iterate all groups G_i in rank order as scoring
    // /// sources. Score each variable x_j in the tied group by:
    // ///   score(x_j) = sum over P in G_i of
    // ///                sum over f in G_k of
    // ///                wire_counts_in_poly(filter(P, f), max_degree, j)
    // /// When G_k is a singleton {x_k} this reduces to the old 2.3 behaviour.
    // /// When G_k is a tied group the filter sums over all variables in the group.
    // /// If any split occurs for any (G_k, G_i) pair, record it and restart from 2.1.
    // /// If all G_i fail for a given G_k, move to the next G_k and restart G_i iteration.
    // /// Variable ranking directly becomes polynomial ranking.
    // fn try_rule_2_3(&mut self, polynomials: &[Polynomial], max_degree: usize) -> bool {
    //     for gi in 0..self.groups.len() {
    //         if self.groups[gi].len() <= 1 {
    //             continue;
    //         }
    //         let group = self.groups[gi].clone();

    //         // Outer loop: iterate all groups G_k in rank order as filter sources
    //         for fgi in 0..self.groups.len() {
    //             let filter_group = self.groups[fgi].clone();

    //             // Inner loop: iterate all groups G_i in rank order as scoring sources
    //             for rgi in 0..self.groups.len() {
    //                 let scoring_group = self.groups[rgi].clone();

    //                 // Score each variable x_j in tied group:
    //                 // sum over P in G_i of sum over f in G_k of
    //                 // wire_counts_in_poly(filter(P, f), max_degree, j)
    //                 let scored: Vec<(usize, Vec<usize>)> = group
    //                     .iter()
    //                     .map(|&w| {
    //                         let score = scoring_group.iter().fold(
    //                             vec![0usize; max_degree + 1],
    //                             |mut acc, &p| {
    //                                 // Sum over all filter variables in G_k
    //                                 for &f in &filter_group {
    //                                     let filtered =
    //                                         filter_poly_by_var(&polynomials[p], f);
    //                                     let counts =
    //                                         wire_counts_in_poly(&filtered, max_degree, w);
    //                                     for (a, c) in acc.iter_mut().zip(counts.iter()) {
    //                                         *a += c;
    //                                     }
    //                                 }
    //                                 acc
    //                             },
    //                         );
    //                         (w, score)
    //                     })
    //                     .collect();

    //                 let split = split_by_scores(scored);
    //                 if split.len() > 1 {
    //                     self.apply_split(gi, split);
    //                     return true;
    //                 }
    //             }
    //             // All G_i failed for this filter group — move to next G_k
    //         }
    //     }
    //     false
    // }

    /// Rule 2.4: for each tied group G_j, rank each polynomial's full monomial set
    /// using the current partial variable ordering, then compare polynomials
    /// lexicographically by their sort key lists all the way through.
    ///
    /// Monomial ordering:
    ///   1. Higher degree always ranks above lower degree.
    ///   2. Within the same degree, compare by variable ranks (highest ranked var first)
    ///      lexicographically. Tied variables get the same rank value, so monomials
    ///      differing only in tied variables compare as equal.
    ///
    /// Polynomial ordering: sort monomials by their rank keys highest-first, then compare
    /// the full key lists lexicographically — compare all the way through.
    /// Polynomials that compare as equal under the full ordering remain tied.
    ///
    /// If any split occurs, record it and return true to restart from 2.1.
    fn try_rule_2_4(&mut self, polynomials: &[Polynomial]) -> bool {
        let var_rank = self.current_var_rank();
        let n = self.n;

        for gi in 0..self.groups.len() {
            if self.groups[gi].len() <= 1 {
                continue;
            }
            let group = self.groups[gi].clone();

            // For each polynomial, compute sorted list of packed u64 sort keys
            let mut scored: Vec<(usize, Vec<u64>)> = group
                .iter()
                .map(|&p| {
                    let mut keys: Vec<u64> = polynomials[p]
                        .iter()
                        .map(|&m| monomial_sort_key_u64(m, &var_rank, n))
                        .collect();
                    keys.sort_unstable_by(|a, b| b.cmp(a)); // descending = best first
                    (p, keys)
                })
                .collect();

            scored.sort_by(|a, b| b.1.cmp(&a.1));

            let mut split: Vec<Vec<usize>> = Vec::new();
            let mut current = vec![scored[0].0];
            for i in 1..scored.len() {
                if scored[i].1 == scored[i - 1].1 {
                    current.push(scored[i].0);
                } else {
                    split.push(current.clone());
                    current = vec![scored[i].0];
                }
            }
            split.push(current);

            if split.len() > 1 {
                self.apply_split(gi, split);
                return true;
            }
        }
        false
    }

    /// Rule 2.5: combines the filtering of 2.3 with the monomial ranking of 2.4.
    /// For each tied group G_j, iterate all groups G_i in rank order. For each
    /// variable x_j in the tied group, build a ranked monomial key list by:
    ///   1. For each P in G_i, filter to monomials containing x_j
    ///   2. Concatenate all filtered monomial sets across all P in G_i (duplicates ok)
    ///   3. Compute sort keys for the concatenated list and sort ascending
    /// Compare the sort key lists for each x_j lexicographically to split G_j.
    /// Variable ranking directly becomes polynomial ranking.
    /// If any split occurs, record it and return true to restart from 2.1.
    fn try_rule_2_5(&mut self, polynomials: &[Polynomial]) -> bool {
        let var_rank = self.current_var_rank();
        let n = self.n;

        for gi in 0..self.groups.len() {
            if self.groups[gi].len() <= 1 {
                continue;
            }
            let group = self.groups[gi].clone();

            for rgi in 0..self.groups.len() {
                let scoring_group = self.groups[rgi].clone();

                let mut scored: Vec<(usize, Vec<u64>)> = group
                    .iter()
                    .map(|&w| {
                        let mut keys: Vec<u64> = scoring_group
                            .iter()
                            .flat_map(|&p| filter_poly_by_var(&polynomials[p], w))
                            .map(|m| monomial_sort_key_u64(m, &var_rank, n))
                            .collect();
                        keys.sort_unstable_by(|a, b| b.cmp(a)); // descending = best first
                        (w, keys)
                    })
                    .collect();

                scored.sort_by(|a, b| b.1.cmp(&a.1));

                let mut split: Vec<Vec<usize>> = Vec::new();
                let mut current = vec![scored[0].0];
                for i in 1..scored.len() {
                    if scored[i].1 == scored[i - 1].1 {
                        current.push(scored[i].0);
                    } else {
                        split.push(current.clone());
                        current = vec![scored[i].0];
                    }
                }
                split.push(current);

                if split.len() > 1 {
                    self.apply_split(gi, split);
                    return true;
                }
            }
        }
        false
    }
}

// Is `b` reachable from `a` within `candidates` under any known automorphism?
fn is_same_orbit(
    a: usize,
    b: usize,
    candidates: &[usize],
    auts1: &[Vec<usize>],
    auts2: &[Vec<usize>],
) -> bool {
    let cset: HashSet<usize> = candidates.iter().copied().collect();
    let mut visited: HashSet<usize> = HashSet::new();
    let mut frontier = vec![a];
    visited.insert(a);
    while let Some(x) = frontier.pop() {
        for aut in auts1.iter().chain(auts2.iter()) {
            let img = aut[x];
            if img == b {
                return true;
            }
            if cset.contains(&img) && visited.insert(img) {
                frontier.push(img);
            }
        }
    }
    false
}

// Given two orderings that produce the same canonical form,
// build the automorphism: sigma[order_a[pos]] = order_b[pos]
fn automorphism_from_orders(order_a: &[usize], order_b: &[usize], n: usize) -> Vec<usize> {
    let mut sigma = vec![0usize; n];
    for pos in 0..order_a.len() {
        sigma[order_a[pos]] = order_b[pos];
    }
    sigma
}

/// Inner recursive canonicalization. Runs Rules 2.1-2.5 deterministically until stuck,
/// then applies Rule L (backtracking or lowest-index) to break remaining ties.
/// `use_backtracking` toggles between canonical backtracking (correct) and lowest-index
/// tiebreak (fast but may not be canonical for non-symmetric stuck groups).
fn canonicalize_inner(
    polynomials: &[Polynomial],
    initial_groups: Vec<Vec<usize>>,
    max_degree: usize,
    use_backtracking: bool,
    mut trace: Option<&mut Vec<String>>,
    known_auts: &mut Vec<Vec<usize>>,
) -> Vec<usize> {
    let n = polynomials.len();
    let mut state = RankingState::new(initial_groups, n);

    loop {
        if state.is_fully_ranked() {
            break;
        }

        // Rule 2.1
        {
            let t = Instant::now();
            let fired = state.try_rule_2_1(polynomials, max_degree);
            TIME_RULE_2_1.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            if fired {
                if let Some(ref mut t) = trace {
                    t.push("2.1".to_string());
                }
                continue;
            }
        }

        // Rule 2.2
        {
            let t = Instant::now();
            let fired = state.try_rule_2_2(polynomials, max_degree);
            TIME_RULE_2_2.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            if fired {
                if let Some(ref mut t) = trace {
                    t.push("2.2".to_string());
                }
                continue;
            }
        }

        // Rule 2.3
        // {
        //     let t = Instant::now();
        //     let fired = state.try_rule_2_3(polynomials);
        //     TIME_RULE_2_3.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        //     if fired {
        //         if let Some(ref mut t) = trace { t.push("2.3".to_string()); }
        //         continue;
        //     }
        // }

        // Rule 2.4
        {
            let t = Instant::now();
            let fired = state.try_rule_2_4(polynomials);
            TIME_RULE_2_4.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            if fired {
                if let Some(ref mut t) = trace {
                    t.push("2.4".to_string());
                }
                continue;
            }
        }

        // Rule 2.5
        {
            let t = Instant::now();
            let fired = state.try_rule_2_5(polynomials);
            TIME_RULE_2_5.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            if fired {
                if let Some(ref mut t) = trace {
                    t.push("2.5".to_string());
                }
                continue;
            }
        }

        // Rule L
        {
            let t = Instant::now();
            if use_backtracking {
                let gi = state.groups.iter().position(|g| g.len() > 1).unwrap();
                let candidates = state.groups[gi].clone();

                let mut best_canon: Option<Vec<Vec<Monomial>>> = None;
                let mut best_order: Option<Vec<usize>> = None;
                let mut best_trace: Vec<String> = Vec::new();
                let mut best_w: Option<usize> = None;
                let mut tried: Vec<usize> = Vec::new();
                let mut local_auts: Vec<Vec<usize>> = Vec::new();

                for &w in &candidates {
                    // Pruning: skip w if it is in the orbit of any already-tried
                    // candidate under known_auts + local_auts
                    let pruned = tried
                        .iter()
                        .any(|&t| is_same_orbit(t, w, &candidates, known_auts, &local_auts));
                    if pruned {
                        if let Some(ref mut tr) = trace {
                            tr.push(format!("L(pruned {} via automorphism)", w));
                        }
                        continue;
                    }

                    // Individualize w
                    let rest: Vec<usize> = candidates.iter().copied().filter(|&x| x != w).collect();
                    let mut trial_groups = state.groups.clone();
                    let mut replacement = vec![vec![w]];
                    if !rest.is_empty() {
                        replacement.push(rest.clone());
                    }
                    trial_groups.splice(gi..=gi, replacement);

                    // Pass inherited + local auts into the child
                    let mut child_auts: Vec<Vec<usize>> = known_auts
                        .iter()
                        .chain(local_auts.iter())
                        .cloned()
                        .collect();

                    let mut trial_trace: Vec<String> = Vec::new();
                    let trial_order = canonicalize_inner(
                        polynomials,
                        trial_groups,
                        max_degree,
                        use_backtracking,
                        Some(&mut trial_trace),
                        &mut child_auts,
                    );
                    let trial_canon = make_canonical_form(polynomials, &trial_order);

                    match best_canon {
                        None => {
                            best_canon = Some(trial_canon);
                            best_order = Some(trial_order);
                            best_trace = trial_trace;
                            best_w = Some(w);
                        }
                        Some(ref bc) => {
                            if trial_canon == *bc {
                                // Same canonical form -> found an automorphism
                                let aut = automorphism_from_orders(
                                    best_order.as_ref().unwrap(),
                                    &trial_order,
                                    n,
                                );
                                local_auts.push(aut);
                                if let Some(ref mut tr) = trace {
                                    tr.push(format!("L(aut found {} ~ {})", best_w.unwrap(), w));
                                }
                            } else if trial_canon < *bc {
                                best_canon = Some(trial_canon);
                                best_order = Some(trial_order);
                                best_trace = trial_trace;
                                best_w = Some(w);
                            }
                        }
                    }

                    tried.push(w);
                }

                // Propagate local automorphisms up to parent
                known_auts.extend(local_auts);

                let best = best_w.unwrap();
                if let Some(ref mut tr) = trace {
                    tr.push(format!("L(picked {})", best));
                    tr.extend(best_trace);
                }

                let rest: Vec<usize> = candidates.iter().copied().filter(|&x| x != best).collect();
                let mut replacement = vec![vec![best]];
                if !rest.is_empty() {
                    replacement.push(rest);
                }
                state.groups.splice(gi..=gi, replacement);
            } else {
                // Non-backtracking: lowest index wins, unchanged
                let gi = state.groups.iter().position(|g| g.len() > 1).unwrap();
                let mut group = state.groups[gi].clone();
                group.sort();
                let winner = group.remove(0);
                if let Some(ref mut t) = trace {
                    t.push(format!("L(lowest {})", winner));
                }
                let mut replacement = vec![vec![winner]];
                if !group.is_empty() {
                    replacement.push(group);
                }
                state.groups.splice(gi..=gi, replacement);
            }
            TIME_RULE_L.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }

    state.groups.iter().map(|g| g[0]).collect()
}

pub fn canonicalize_polys(
    mut polynomials: Vec<Polynomial>,
    use_backtracking: bool,
    print: bool,
) -> (Vec<Polynomial>, Permutation) {
    let n = polynomials.len();
    if n == 0 {
        return (vec![], Permutation { data: vec![] });
    }
    for poly in &mut polynomials {
        normalize_polynomial(poly);
    }
    let max_degree = n;

    // Rule 1: Order polynomials by degree profile.
    // Degree counts of a polynomial: [count_of_max_possible_deg, ..., count_of_deg_0]
    // Padded to max_possible_degree+1 entries so Vec comparison is always over equal-length
    // vectors and correctly ranks e.g. one degree-2 monomial above two degree-1 monomials.
    let mut profiles: Vec<(usize, Vec<usize>)> = (0..n)
        .map(|i| (i, degree_counts(&polynomials[i], max_degree)))
        .collect();
    profiles.sort_by(|a, b| b.1.cmp(&a.1));

    // Partition into initially-tied groups. Each group occupies contiguous positions
    // in the final order based on their degree ranking.
    let mut initial_groups: Vec<Vec<usize>> = Vec::new();
    {
        let mut current = vec![profiles[0].0];
        for i in 1..profiles.len() {
            if profiles[i].1 == profiles[i - 1].1 {
                current.push(profiles[i].0);
            } else {
                initial_groups.push(current.clone());
                current = vec![profiles[i].0];
            }
        }
        initial_groups.push(current);
    }

    // Run Rules 2.1-2.5 and Rule L to fully resolve all ties
    let mut trace: Vec<String> = Vec::new();
    let mut known_auts: Vec<Vec<usize>> = Vec::new();
    let final_order = canonicalize_inner(
        &polynomials,
        initial_groups,
        max_degree,
        use_backtracking,
        if print { Some(&mut trace) } else { None },
        &mut known_auts,
    );

    if print {
        println!("Rule trace: {}", trace.join(" -> "));
    }

    // final_order[pos] = wire
    // Remap: variable x_wire -> x_pos (bit wire -> bit pos)
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

    // Remap all polynomials and select them in final_order order
    let canonical: Vec<Polynomial> = final_order
        .iter()
        .map(|&wire| polynomial_from_terms(polynomials[wire].iter().map(|&m| remap_monomial(m))))
        .collect();

    let canonical = trim_canonicalized(canonical);

    (canonical, Permutation { data: final_order })
}

/// After canonicalization, trim trailing polynomials that are uninformative.
/// Starting from the last polynomial, remove P_i if both conditions hold:
///   1. P_i is trivial — its only monomial is the single variable x_i (bitmask 1 << i)
///   2. x_i does not appear in any other polynomial in the full list
/// Stop as soon as we reach a P_i that is non-trivial OR whose variable x_i
/// appears in some other polynomial. Keep everything from that point forward.
///
/// Returns the trimmed polynomial list. The permutation is left unchanged.
pub fn trim_canonicalized(polynomials: Vec<Polynomial>) -> Vec<Polynomial> {
    let n = polynomials.len();
    let mut keep_up_to = n; // exclusive upper bound — trim everything at or after this

    for i in (0..n).rev() {
        let bit = 1u64 << i;

        // Check if P_i is trivial: exactly one monomial which is just x_i
        let is_trivial =
            polynomials[i].len() == 1 && polynomials[i].iter().next().copied().unwrap() == bit;

        if !is_trivial {
            // Non-trivial polynomial — stop trimming here
            break;
        }

        // Check if x_i appears in any other polynomial (including higher degree monomials)
        let used_elsewhere = polynomials
            .iter()
            .enumerate()
            .any(|(j, poly)| j != i && poly.iter().any(|&m| m & bit != 0));

        if used_elsewhere {
            // x_i is referenced by another polynomial — stop trimming here
            break;
        }

        // P_i is trivial and x_i is unused elsewhere — trim it
        keep_up_to = i;
    }

    polynomials[..keep_up_to].to_vec()
}

// pub fn canonicalize_polys_3(
//     polynomials: Vec<Polynomial>,
// ) -> (Vec<Polynomial>, Permutation) {
//     let n = polynomials.len();
//     if n == 0 {
//         return (vec![], Permutation { data: vec![] });
//     }
//     let max_degree = n;

//     // ── Step 1: same as canonicalize_polys ───────────────────────────────────
//     let mut profiles: Vec<(usize, Vec<usize>)> = (0..n)
//         .map(|i| (i, degree_counts(&polynomials[i], max_degree)))
//         .collect();
//     profiles.sort_by(|a, b| b.1.cmp(&a.1));

//     let mut groups: Vec<Vec<usize>> = Vec::new();
//     {
//         let mut current = vec![profiles[0].0];
//         for i in 1..profiles.len() {
//             if profiles[i].1 == profiles[i - 1].1 {
//                 current.push(profiles[i].0);
//             } else {
//                 groups.push(current.clone());
//                 current = vec![profiles[i].0];
//             }
//         }
//         groups.push(current);
//     }

//     // ── Step 2: sum polynomials within each group over N ─────────────────────
//     // Coefficients are natural numbers (count how many polys in the group
//     // contain each monomial), NOT GF(2).
//     let class_polys: Vec<HashMap<Monomial, usize>> = groups.iter().map(|group| {
//         let mut sum: HashMap<Monomial, usize> = HashMap::new();
//         for &wire in group {
//             for &m in &polynomials[wire] {
//                 *sum.entry(m).or_insert(0) += 1;
//             }
//         }
//         sum
//     }).collect();

//     // ── Step 3: find w minimizing P_{C_k} lexicographically ──────────────────
//     //
//     // w[i] = rank assigned to variable x_i, a permutation of {0..n}.
//     // Substitution: z_i = (w[i] + 2)^n.
//     // Objective: minimize P_{C_1}(z), break ties with P_{C_2}(z), etc.
//     //
//     // Greedy algorithm:
//     // Assign ranks 0, 1, 2, ... one at a time to unassigned variables.
//     // At each step, among still-unassigned variables, the one that should
//     // get the current (smallest remaining) rank is the one whose assignment
//     // most reduces the objective — i.e. the variable that, when given the
//     // smallest z value, contributes least.
//     //
//     // Because (pos+2)^n provides degree-hierarchical separation, the variable
//     // that should get rank `next_rank` is the one that appears in the
//     // *fewest / lowest-degree / lowest-coeff* monomials — so that assigning
//     // it a small z value saves the most.
//     //
//     // Equivalently: sort variables so that the one appearing in the most /
//     // highest-degree / highest-coeff monomials gets the LARGEST rank
//     // (largest z value). That way the highest-degree terms get the largest
//     // inputs, which... wait, we want to MINIMIZE, so we want small z on
//     // high-contribution variables.
//     //
//     // Let's be precise. The evaluation is:
//     //   sum_{m} coeff(m) * prod_{i in supp(m)} (w(i)+2)^n
//     //
//     // To minimize: assign w(i)=0 (smallest z) to the variable i that appears
//     // in the most/highest monomials, because giving a small multiplier to a
//     // high-weight variable reduces the sum most.
//     //
//     // So: rank 0 (z = 2^n, smallest) -> variable with highest contribution
//     //     rank n-1 (z = (n+1)^n, largest) -> variable with lowest contribution
//     //
//     // We determine "highest contribution" iteratively, fixing one variable
//     // per step and updating which monomials are still "live" for comparison.
//     //
//     // For correctness with interactions: at each step, score each remaining
//     // variable by the sorted list of (degree_of_monomial, coeff) for all
//     // class-poly monomials containing it, evaluated under the CURRENT partial
//     // assignment (substituting already-fixed variables with their z values).
//     // Then pick the variable with the highest score (give it the next rank).
//     //
//     // This is equivalent to: reduce each monomial by substituting fixed
//     // variables, then score remaining variables on the reduced polynomial.

//     // w_assignment[wire] = rank assigned to wire (-1 = unassigned)
//     let mut w: Vec<Option<usize>> = vec![None; n];
//     let mut rank_to_wire: Vec<usize> = Vec::with_capacity(n); // rank_to_wire[rank] = wire

//     // We assign rank 0 first (this wire gets z = 2^n, the smallest value).
//     // At each step we pick the wire that has the highest contribution to the
//     // class polynomials (considering interactions via already-fixed wires),
//     // and assign it the next rank.
//     //
//     // "Contribution score" of an unassigned wire `a` to class poly `ci`,
//     // given current partial assignment:
//     //   For each monomial m containing `a`:
//     //     - if m contains any other UNASSIGNED wire besides `a`: skip for now
//     //       (its contribution depends on future assignments)
//     //     - contribution = coeff(m) * prod_{j in supp(m), j != a} z_j
//     //       where z_j = (w[j]+2)^n for already-assigned j
//     //   We represent this as a BigUint for exact comparison.
//     //
//     // Actually, for the greedy to be correct we score by the full monomial
//     // profile including unresolved interactions — using the same
//     // (degree, coeff) key as before but now accounting for partial subs.
//     // The simplest correct approach: score by the reduced polynomial's
//     // contribution, where "reduced" means substitute fixed variables.

//     // Helper: given current partial w assignment, compute the "contribution
//     // score" of unassigned wire `a` to class_poly[ci] as a BigUint.
//     // For each monomial containing `a`, substitute already-assigned variables
//     // and accumulate the product * coeff. Monomials where another unassigned
//     // variable also appears contribute (rank_of_that_var + 2)^n which we
//     // don't know yet — so we use a symbolic key instead: sorted tuple of
//     // (degree_remaining, coeff * product_of_fixed_z_values).
//     // For simplicity and correctness we use BigUint throughout.
//     let score_wire = |wire: usize, w: &[Option<usize>], ci: usize| -> Vec<(usize, BigUint)> {
//         // Returns sorted-descending list of (remaining_degree, partial_coeff)
//         // for each monomial in class_poly[ci] containing `wire`.
//         // remaining_degree = number of unassigned variables in monomial (including wire).
//         // partial_coeff = coeff * product of (w[j]+2)^n for assigned j in monomial.
//         let mut entries: Vec<(usize, BigUint)> = class_polys[ci]
//             .iter()
//             .filter(|(m, _)| *m & (1u64 << wire) != 0)
//             .map(|(&m, &coeff)| {
//                 let mut partial = BigUint::from(coeff);
//                 let mut remaining_deg = 0usize;
//                 for j in 0..n {
//                     if m & (1u64 << j) != 0 {
//                         match w[j] {
//                             Some(rank) if j != wire => {
//                                 // already assigned: substitute z_j = (rank+2)^n
//                                 partial *= BigUint::from(rank + 2).pow(n as u32);
//                             }
//                             _ => {
//                                 // unassigned (or is `wire` itself)
//                                 remaining_deg += 1;
//                             }
//                         }
//                     }
//                 }
//                 (remaining_deg, partial)
//             })
//             .collect();
//         // Sort descending: higher remaining_deg first (more impactful),
//         // then higher partial_coeff first
//         entries.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)));
//         entries
//     };

//     // Assign ranks 0, 1, ..., n-1 one at a time
//     let mut unassigned: Vec<usize> = (0..n).collect();

//     for next_rank in 0..n {
//         // Among unassigned wires, find the one with the highest contribution
//         // (it gets the smallest z value = next_rank, to minimize the objective)
//         // Break ties lexicographically across class polynomials.
//         let best_wire = unassigned.iter().copied().max_by(|&a, &b| {
//             for ci in 0..class_polys.len() {
//                 let sa = score_wire(a, &w, ci);
//                 let sb = score_wire(b, &w, ci);
//                 // Compare lex: higher score = more contribution = gets smaller rank
//                 let cmp = sa.cmp(&sb);
//                 if cmp != std::cmp::Ordering::Equal {
//                     return cmp;
//                 }
//             }
//             std::cmp::Ordering::Equal
//         }).unwrap();

//         w[best_wire] = Some(next_rank);
//         rank_to_wire.push(best_wire);
//         unassigned.retain(|&x| x != best_wire);
//     }

//     // rank_to_wire[rank] = wire, so rank 0 -> wire with most contribution.
//     // final_order[pos] = wire means canonical position pos gets this wire.
//     // We want the wire with rank 0 (highest contribution, assigned smallest z)
//     // to get canonical position 0.
//     let final_order = rank_to_wire;

//     // ── Step 4: remap polynomials ─────────────────────────────────────────────
//     let mut wire_to_pos = vec![0usize; n];
//     for (pos, &wire) in final_order.iter().enumerate() {
//         wire_to_pos[wire] = pos;
//     }

//     let remap_monomial = |m: Monomial| -> Monomial {
//         let mut result = 0u64;
//         for wire in 0..n {
//             if m & (1u64 << wire) != 0 {
//                 result |= 1u64 << wire_to_pos[wire];
//             }
//         }
//         result
//     };

//     let canonical: Vec<Polynomial> = final_order
//         .iter()
//         .map(|&wire| polynomials[wire].iter().map(|&m| remap_monomial(m)).collect())
//         .collect();

//     let canonical = trim_canonicalized(canonical);

//     (canonical, Permutation { data: final_order })
// }

// ── Helpers for canonicalize_polys_4 ─────────────────────────────────────────

const MONOMIAL_RANK_KEY_LEN_4: usize = 65;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MonomialRankKey4 {
    degree: u8,
    encoded_ranks: [u8; MONOMIAL_RANK_KEY_LEN_4],
}

impl Ord for MonomialRankKey4 {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        self.encoded_ranks.cmp(&other.encoded_ranks)
    }
}

impl PartialOrd for MonomialRankKey4 {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MonomialLevelKey4 {
    rank_key: MonomialRankKey4,
    coeff: usize,
}

type LevelEntry4 = (Monomial, usize, MonomialLevelKey4);

fn monomial_rank_key_4(m: Monomial, vr: &[usize], n: usize) -> MonomialRankKey4 {
    let mut encoded_ranks = [0u8; MONOMIAL_RANK_KEY_LEN_4];
    let mut degree = 0usize;
    for v in 0..n {
        if m & (1u64 << v) != 0 {
            debug_assert!(vr[v] < u8::MAX as usize);
            encoded_ranks[degree] = (vr[v] + 1) as u8;
            degree += 1;
        }
    }
    encoded_ranks[..degree].sort_unstable();
    MonomialRankKey4 {
        degree: degree as u8,
        encoded_ranks,
    }
}

fn monomial_level_key_4(m: Monomial, coeff: usize, vr: &[usize], n: usize) -> MonomialLevelKey4 {
    MonomialLevelKey4 {
        rank_key: monomial_rank_key_4(m, vr, n),
        coeff,
    }
}

fn cmp_level_key_4(a: &MonomialLevelKey4, b: &MonomialLevelKey4) -> CmpOrdering {
    b.rank_key
        .degree
        .cmp(&a.rank_key.degree)
        .then(a.rank_key.cmp(&b.rank_key))
        .then(b.coeff.cmp(&a.coeff))
}

fn sorted_level_entries_4(
    cp: &BTreeMap<Monomial, usize>,
    vr: &[usize],
    n: usize,
    entries: &mut Vec<LevelEntry4>,
) {
    entries.clear();
    entries.extend(
        cp.iter()
            .map(|(&m, &c)| (m, c, monomial_level_key_4(m, c, vr, n))),
    );
    entries.sort_by(|a, b| cmp_level_key_4(&a.2, &b.2));
}

// Count how many monomials in a level each wire appears in.
fn wire_freq_4(level: &[LevelEntry4], n: usize, freq: &mut Vec<usize>) {
    freq.resize(n, 0);
    freq.fill(0);
    for &(m, _, _) in level {
        for v in 0..n {
            if m & (1u64 << v) != 0 {
                freq[v] += 1;
            }
        }
    }
}

// Split the FIRST (highest-priority) tied wire group whose members have different
// frequencies. Higher frequency → higher priority (lower rank number). Returns true if any split.
fn split_by_freq_4(
    vr: &mut Vec<usize>,
    n: usize,
    freq: &[usize],
    tied: &mut Vec<usize>,
    sorted: &mut Vec<usize>,
    sub_ranks: &mut Vec<usize>,
) -> bool {
    let max_rank = *vr.iter().max().unwrap_or(&0);
    for cur_rank in 0..=max_rank {
        tied.clear();
        tied.extend((0..n).filter(|&v| vr[v] == cur_rank));
        if tied.len() <= 1 {
            continue;
        }
        let first_freq = freq[tied[0]];
        if tied.iter().all(|&v| freq[v] == first_freq) {
            continue;
        }

        sorted.clear();
        sorted.extend_from_slice(tied);
        sorted.sort_by(|&a, &b| freq[b].cmp(&freq[a]));

        let mut sub_rank = 0usize;
        sub_ranks.clear();
        sub_ranks.resize(sorted.len(), 0);
        for i in 1..sorted.len() {
            if freq[sorted[i]] != freq[sorted[i - 1]] {
                sub_rank += 1;
            }
            sub_ranks[i] = sub_rank;
        }
        for v in 0..n {
            if vr[v] > cur_rank {
                vr[v] += sub_rank;
            }
        }
        for (i, &v) in sorted.iter().enumerate() {
            vr[v] = cur_rank + sub_ranks[i];
        }
        return true;
    }
    false
}

// Remapped polynomial key for tiebreak #1: replace each variable with its var_rank,
// sort ranks within each monomial, then sort monomials (highest priority first).
fn poly_key_4(poly: &Polynomial, vr: &[usize], n: usize) -> Vec<MonomialRankKey4> {
    let mut terms: Vec<MonomialRankKey4> = poly
        .iter()
        .map(|&m| monomial_rank_key_4(m, vr, n))
        .collect();
    terms.sort_by(|a, b| b.degree.cmp(&a.degree).then(a.cmp(b)));
    terms
}

fn push_flat_canonical_form_4(
    polynomials: &[Polynomial],
    final_order: &[usize],
    wire_to_pos: &mut Vec<usize>,
    monomials: &mut Vec<Monomial>,
    out: &mut Vec<Option<Monomial>>,
) {
    let n = polynomials.len();
    wire_to_pos.resize(n, 0);
    for (pos, &wire) in final_order.iter().enumerate() {
        wire_to_pos[wire] = pos;
    }

    out.clear();
    for &wire in final_order {
        monomials.clear();
        monomials.extend(polynomials[wire].iter().map(|&m| {
            let mut r = 0u64;
            for v in 0..n {
                if m & (1u64 << v) != 0 {
                    r |= 1u64 << wire_to_pos[v];
                }
            }
            r
        }));
        monomials.sort_unstable();
        out.extend(monomials.iter().copied().map(Some));
        out.push(None);
    }
}

fn scan_class_poly_levels_4(
    cp: &BTreeMap<Monomial, usize>,
    vr: &mut Vec<usize>,
    n: usize,
    level_entries: &mut Vec<LevelEntry4>,
    freq: &mut Vec<usize>,
    tied: &mut Vec<usize>,
    sorted: &mut Vec<usize>,
    sub_ranks: &mut Vec<usize>,
) -> bool {
    sorted_level_entries_4(cp, vr, n, level_entries);
    let mut start = 0usize;
    while start < level_entries.len() {
        let mut end = start + 1;
        while end < level_entries.len() && level_entries[end].2 == level_entries[start].2 {
            end += 1;
        }
        wire_freq_4(&level_entries[start..end], n, freq);
        if split_by_freq_4(vr, n, freq, tied, sorted, sub_ranks) {
            return true;
        }
        start = end;
    }
    false
}

fn has_ties_4(vr: &[usize]) -> bool {
    let n = vr.len();
    (0..n).any(|v| (0..n).any(|u| u != v && vr[u] == vr[v]))
}

// Core loop: refine var_rank until fully resolved, then return final_order.
fn canon4_run(
    polynomials: &[Polynomial],
    class_polys: &[BTreeMap<Monomial, usize>],
    mut vr: Vec<usize>,
    allow_rule_l: bool,
) -> Result<Vec<usize>, ()> {
    let n = polynomials.len();
    let mut level_entries: Vec<LevelEntry4> = Vec::new();
    let mut freq_scratch: Vec<usize> = Vec::with_capacity(n);
    let mut tied_scratch: Vec<usize> = Vec::with_capacity(n);
    let mut sorted_scratch: Vec<usize> = Vec::with_capacity(n);
    let mut sub_ranks_scratch: Vec<usize> = Vec::with_capacity(n);
    let mut d_class_poly: BTreeMap<Monomial, usize> = BTreeMap::new();

    'master: loop {
        if !has_ties_4(&vr) {
            break;
        }

        // Phase 1: scan P_{C_i} monomial levels; split by wire frequency.
        // Any split of the first splittable group → restart.
        for cp in class_polys {
            if scan_class_poly_levels_4(
                cp,
                &mut vr,
                n,
                &mut level_entries,
                &mut freq_scratch,
                &mut tied_scratch,
                &mut sorted_scratch,
                &mut sub_ranks_scratch,
            ) {
                continue 'master;
            }
        }

        if !has_ties_4(&vr) {
            break;
        }

        // Tiebreak #1: for each tied group, compare remapped polynomial keys.
        // First group where keys differ → split and restart.
        let max_rank = *vr.iter().max().unwrap_or(&0);
        for cur_rank in 0..=max_rank {
            let tied: Vec<usize> = (0..n).filter(|&v| vr[v] == cur_rank).collect();
            if tied.len() <= 1 {
                continue;
            }

            let mut sorted: Vec<(usize, Vec<MonomialRankKey4>)> = tied
                .iter()
                .map(|&v| (v, poly_key_4(&polynomials[v], &vr, n)))
                .collect();
            sorted.sort_by(|a, b| a.1.cmp(&b.1));

            let mut sub_rank = 0usize;
            let mut sub_ranks = vec![0usize; sorted.len()];
            for i in 1..sorted.len() {
                if sorted[i - 1].1 != sorted[i].1 {
                    sub_rank += 1;
                }
                sub_ranks[i] = sub_rank;
            }
            if sub_rank > 0 {
                for v in 0..n {
                    if vr[v] > cur_rank {
                        vr[v] += sub_rank;
                    }
                }
                for (i, &(v, _)) in sorted.iter().enumerate() {
                    vr[v] = cur_rank + sub_ranks[i];
                }
                continue 'master;
            }
        }

        if !has_ties_4(&vr) {
            break;
        }

        // Tiebreak #2: dynamic class polys P_{D_i} from current rank groups.
        // Apply same monomial-level scanning as Phase 1.
        let max_rank_val = *vr.iter().max().unwrap_or(&0);
        for rk in 0..=max_rank_val {
            d_class_poly.clear();
            for w in 0..n {
                if vr[w] == rk {
                    for &m in &polynomials[w] {
                        *d_class_poly.entry(m).or_insert(0) += 1;
                    }
                }
            }
            if d_class_poly.is_empty() {
                continue;
            }
            if scan_class_poly_levels_4(
                &d_class_poly,
                &mut vr,
                n,
                &mut level_entries,
                &mut freq_scratch,
                &mut tied_scratch,
                &mut sorted_scratch,
                &mut sub_ranks_scratch,
            ) {
                continue 'master;
            }
        }

        // Rule L: try each wire in the first tied group as the sole winner.
        // Take the candidate that produces the lexicographically smallest canonical form.
        let tied_rank = (0..n)
            .filter(|&v| (0..n).filter(|&u| vr[u] == vr[v]).count() > 1)
            .map(|v| vr[v])
            .min();

        if let Some(tr) = tied_rank {
            if !allow_rule_l {
                return Err(());
            }
            let candidates: Vec<usize> = (0..n).filter(|&v| vr[v] == tr).collect();
            let rule_l_start = Instant::now();
            CANON4_RULE_L_CALLS.fetch_add(1, Ordering::Relaxed);
            CANON4_RULE_L_BRANCHES.fetch_add(candidates.len() as u64, Ordering::Relaxed);
            let mut best_canonical: Vec<Option<Monomial>> = Vec::new();
            let mut trial_canonical: Vec<Option<Monomial>> = Vec::new();
            let mut canonical_monomials: Vec<Monomial> = Vec::new();
            let mut wire_to_pos: Vec<usize> = Vec::with_capacity(n);
            let mut have_best = false;
            let mut best_order: Vec<usize> = Vec::new();

            for &w in &candidates {
                let mut trial_vr = vr.clone();
                for v in 0..n {
                    if trial_vr[v] > tr {
                        trial_vr[v] += 1;
                    }
                }
                for &other in &candidates {
                    if other != w {
                        trial_vr[other] = tr + 1;
                    }
                }

                let trial_order = canon4_run(polynomials, class_polys, trial_vr, true)?;
                push_flat_canonical_form_4(
                    polynomials,
                    &trial_order,
                    &mut wire_to_pos,
                    &mut canonical_monomials,
                    &mut trial_canonical,
                );

                if !have_best || trial_canonical < best_canonical {
                    best_canonical.clear();
                    best_canonical.extend_from_slice(&trial_canonical);
                    best_order = trial_order;
                    have_best = true;
                }
            }
            let rule_l_elapsed = rule_l_start.elapsed();
            CANON4_RULE_L_TIME.fetch_add(rule_l_elapsed.as_nanos() as u64, Ordering::Relaxed);
            if compression_trace_enabled()
                && rule_l_elapsed.as_millis() >= compression_trace_threshold_ms()
            {
                eprintln!(
                    "[compress-trace] slow rule_l n={} tied_rank={} branches={} elapsed_ms={}",
                    n,
                    tr,
                    candidates.len(),
                    rule_l_elapsed.as_millis()
                );
            }
            return Ok(best_order);
        }

        break;
    }

    let mut final_order: Vec<usize> = (0..n).collect();
    final_order.sort_by_key(|&w| (vr[w], w));
    Ok(final_order)
}

pub fn canonicalize_polys_4(
    mut polynomials: Vec<Polynomial>,
    allow_rule_l: bool,
) -> Result<(Vec<Polynomial>, Permutation), ()> {
    let n = polynomials.len();
    if n == 0 {
        return Ok((vec![], Permutation { data: vec![] }));
    }
    for poly in &mut polynomials {
        normalize_polynomial(poly);
    }
    let max_degree = n;

    // Group wires by degree profile; highest-profile group = P_{C_1}.
    let mut profiles: Vec<(usize, Vec<usize>)> = (0..n)
        .map(|i| (i, degree_counts(&polynomials[i], max_degree)))
        .collect();
    profiles.sort_by(|a, b| b.1.cmp(&a.1));

    let mut class_groups: Vec<Vec<usize>> = Vec::new();
    {
        let mut current = vec![profiles[0].0];
        for i in 1..profiles.len() {
            if profiles[i].1 == profiles[i - 1].1 {
                current.push(profiles[i].0);
            } else {
                class_groups.push(current.clone());
                current = vec![profiles[i].0];
            }
        }
        class_groups.push(current);
    }

    // Build P_{C_i}: sum of polynomials in each class group (natural-number coefficients).
    let class_polys: Vec<BTreeMap<Monomial, usize>> = class_groups
        .iter()
        .map(|group| {
            let mut sum: BTreeMap<Monomial, usize> = BTreeMap::new();
            for &wire in group {
                for &m in &polynomials[wire] {
                    *sum.entry(m).or_insert(0) += 1;
                }
            }
            sum
        })
        .collect();

    // All wires start tied; canon4_run refines iteratively.
    let final_order = canon4_run(&polynomials, &class_polys, vec![0usize; n], allow_rule_l)?;

    let mut wire_to_pos = vec![0usize; n];
    for (pos, &wire) in final_order.iter().enumerate() {
        wire_to_pos[wire] = pos;
    }
    let remap_monomial = |m: Monomial| -> Monomial {
        let mut result = 0u64;
        for wire in 0..n {
            if m & (1u64 << wire) != 0 {
                result |= 1u64 << wire_to_pos[wire];
            }
        }
        result
    };
    let canonical: Vec<Polynomial> = final_order
        .iter()
        .map(|&wire| polynomial_from_terms(polynomials[wire].iter().map(|&m| remap_monomial(m))))
        .collect();
    let canonical = trim_canonicalized(canonical);
    Ok((canonical, Permutation { data: final_order }))
}

// pub fn canonicalize_polys_4(
//     polynomials: Vec<Polynomial>,
// ) -> (Vec<Polynomial>, Permutation) {
//     let n = polynomials.len();
//     if n == 0 {
//         return (vec![], Permutation { data: vec![] });
//     }
//     let max_degree = n;

//     // ── Step 1: partition into equivalence classes ───────────────────────────
//     let mut profiles: Vec<(usize, Vec<usize>)> = (0..n)
//         .map(|i| (i, degree_counts(&polynomials[i], max_degree)))
//         .collect();
//     profiles.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

//     let mut groups: Vec<Vec<usize>> = Vec::new();
//     {
//         let mut current = vec![profiles[0].0];
//         for i in 1..profiles.len() {
//             if profiles[i].1 == profiles[i - 1].1 {
//                 current.push(profiles[i].0);
//             } else {
//                 groups.push(current.clone());
//                 current = vec![profiles[i].0];
//             }
//         }
//         groups.push(current);
//     }

//     // ── Step 2: build class polynomials ──────────────────────────────────────
//     let class_polys: Vec<BTreeMap<Monomial, usize>> = groups.iter().map(|group| {
//         let mut sum: BTreeMap<Monomial, usize> = BTreeMap::new();
//         for &wire in group {
//             for &m in &polynomials[wire] {
//                 *sum.entry(m).or_insert(0) += 1;
//             }
//         }
//         sum
//     }).collect();

//     // ── Step 3: iterative refinement ─────────────────────────────────────────
//     let mut var_rank: Vec<usize> = vec![0usize; n];

//     let cmp_monomials = |m: Monomial, coeff_m: usize,
//                          mp: Monomial, coeff_mp: usize,
//                          vr: &[usize]| -> Option<std::cmp::Ordering> {
//         let deg_m  = m.count_ones() as usize;
//         let deg_mp = mp.count_ones() as usize;
//         if deg_m != deg_mp { return Some(deg_m.cmp(&deg_mp)); }

//         let mut ranks_m:  Vec<usize> = (0..n).filter(|&j| m  & (1u64<<j)!=0).map(|j| vr[j]).collect();
//         let mut ranks_mp: Vec<usize> = (0..n).filter(|&j| mp & (1u64<<j)!=0).map(|j| vr[j]).collect();
//         ranks_m.sort_unstable();
//         ranks_mp.sort_unstable();

//         if ranks_m == ranks_mp {
//             return Some(coeff_m.cmp(&coeff_mp));
//         }

//         Some(ranks_mp.cmp(&ranks_m))
//     };

//     let ranked_monomials_of = |x: usize, ci: usize, vr: &[usize]|
//         -> Vec<Vec<(Monomial, usize)>> {
//         let mut remaining: Vec<(Monomial, usize)> = class_polys[ci].iter()
//             .filter(|(m, _)| *m & (1u64 << x) != 0)
//             .map(|(&m, &coeff)| (m, coeff))
//             .collect();
//         remaining.sort_unstable();
//         if remaining.is_empty() { return vec![]; }
//         let mut levels = Vec::new();
//         while !remaining.is_empty() {
//             let mut top: Vec<(Monomial, usize)> = remaining.iter().copied()
//                 .filter(|&(m, cm)| {
//                     !remaining.iter().any(|&(mp, cmp)| {
//                         if (m,cm)==(mp,cmp) { return false; }
//                         matches!(cmp_monomials(mp,cmp,m,cm,vr), Some(std::cmp::Ordering::Greater))
//                     })
//                 })
//                 .collect();
//             top.sort_unstable();
//             let top_set: BTreeSet<(Monomial, usize)> = top.iter().copied().collect();
//             remaining.retain(|x| !top_set.contains(x));
//             levels.push(top);
//         }
//         levels
//     };

//     let ranked_monomials_of_poly = |poly: &Polynomial, vr: &[usize]|
//         -> Vec<(Monomial, usize)> {
//         let mut ms: Vec<(Monomial, usize)> = poly.iter().map(|&m| (m, 1usize)).collect();
//         ms.sort_by(|&(a, ca), &(b, cb)| {
//             cmp_monomials(a, ca, b, cb, vr)
//                 .unwrap_or(std::cmp::Ordering::Equal)
//                 .reverse()
//                 .then(a.cmp(&b))
//         });
//         ms
//     };

//     let compare_vars_in_class = |x: usize, xp: usize, ci: usize, vr: &[usize]|
//         -> Option<std::cmp::Ordering> {
//         let levels_x  = ranked_monomials_of(x,  ci, vr);
//         let levels_xp = ranked_monomials_of(xp, ci, vr);
//         let depth = levels_x.len().max(levels_xp.len());
//         for k in 0..depth {
//             match (levels_x.get(k), levels_xp.get(k)) {
//                 (None, None)    => break,
//                 (Some(_), None) => return Some(std::cmp::Ordering::Greater),
//                 (None, Some(_)) => return Some(std::cmp::Ordering::Less),
//                 (Some(lx), Some(lxp)) => {
//                     let mut saw_greater = false;
//                     let mut saw_less    = false;
//                     for &(m, cm) in lx {
//                         for &(mp, cmp) in lxp {
//                             match cmp_monomials(m, cm, mp, cmp, vr) {
//                                 Some(std::cmp::Ordering::Greater) => saw_greater = true,
//                                 Some(std::cmp::Ordering::Less)    => saw_less    = true,
//                                 _                                  => {}
//                             }
//                         }
//                     }
//                     match (saw_greater, saw_less) {
//                         (true,  false) => return Some(std::cmp::Ordering::Greater),
//                         (false, true)  => return Some(std::cmp::Ordering::Less),
//                         _ => {}
//                     }
//                     if lx.len() != lxp.len() {
//                         return Some(lx.len().cmp(&lxp.len()));
//                     }
//                 }
//             }
//         }
//         None
//     };

//     let poly_key_by_ranks = |x: usize, vr: &[usize]| -> Vec<Vec<usize>> {
//         let mut terms: Vec<Vec<usize>> = polynomials[x]
//             .iter()
//             .map(|&m| {
//                 let mut ranks: Vec<usize> = (0..n)
//                     .filter(|&v| m & (1u64<<v) != 0)
//                     .map(|v| vr[v])
//                     .collect();
//                 ranks.sort_unstable();
//                 ranks
//             })
//             .collect();
//         terms.sort_by(|a, b| b.len().cmp(&a.len()).then(a.cmp(&b)));
//         terms
//     };

//     let has_ties = |vr: &[usize]| -> bool {
//         (0..n).any(|v| (0..n).any(|u| u != v && vr[u] == vr[v]))
//     };

//     // ── Master loop ───────────────────────────────────────────────────────────
//     'master: loop {
//         if !has_ties(&var_rank) { break; }

//         // Stage 1: class polynomial refinement
//         'outer: loop {
//             for ci in 0..class_polys.len() {
//                 if !has_ties(&var_rank) { break 'outer; }
//                 let max_rank = *var_rank.iter().max().unwrap_or(&0);
//                 for cur_rank in 0..=max_rank {
//                     let tied: Vec<usize> = (0..n)
//                         .filter(|&v| var_rank[v] == cur_rank)
//                         .collect();
//                     if tied.len() <= 1 { continue; }

//                     let mut sorted_tied = tied.clone();
//                     sorted_tied.sort_by(|&a, &b| {
//                         match compare_vars_in_class(a, b, &class_polys[ci], &var_rank) {
//                             Some(std::cmp::Ordering::Greater) => std::cmp::Ordering::Less,
//                             Some(std::cmp::Ordering::Less)    => std::cmp::Ordering::Greater,
//                             _                                  => a.cmp(&b),
//                         }
//                     });

//                     let mut sub_rank = 0usize;
//                     let mut new_sub_ranks = vec![0usize; sorted_tied.len()];
//                     for i in 1..sorted_tied.len() {
//                         if compare_vars_in_class(sorted_tied[i-1], sorted_tied[i], &class_polys[ci], &var_rank).is_some() {
//                             sub_rank += 1;
//                         }
//                         new_sub_ranks[i] = sub_rank;
//                     }

//                     if sub_rank > 0 {
//                         for v in 0..n {
//                             if var_rank[v] > cur_rank { var_rank[v] += sub_rank; }
//                         }
//                         for (i, &v) in sorted_tied.iter().enumerate() {
//                             var_rank[v] = cur_rank + new_sub_ranks[i];
//                         }
//                         continue 'outer;
//                     }
//                 }
//             }
//             break;
//         }

//         if !has_ties(&var_rank) { break 'master; }

//         // Stage 2: individual poly key tiebreaker
//         let mut tb1_fired = false;
//         let max_rank = *var_rank.iter().max().unwrap_or(&0);
//         for cur_rank in 0..=max_rank {
//             let tied: Vec<usize> = (0..n)
//                 .filter(|&v| var_rank[v] == cur_rank)
//                 .collect();
//             if tied.len() <= 1 { continue; }

//             let mut tb_sorted = tied.clone();
//             tb_sorted.sort_by(|&a, &b| {
//                 poly_key_by_ranks(a, &var_rank)
//                     .cmp(&poly_key_by_ranks(b, &var_rank))
//                     .then(a.cmp(&b))
//             });

//             let mut tb_sub_rank = 0usize;
//             let mut tb_new_sub_ranks = vec![0usize; tb_sorted.len()];
//             for i in 1..tb_sorted.len() {
//                 if poly_key_by_ranks(tb_sorted[i-1], &var_rank)
//                     != poly_key_by_ranks(tb_sorted[i], &var_rank)
//                 {
//                     tb_sub_rank += 1;
//                 }
//                 tb_new_sub_ranks[i] = tb_sub_rank;
//             }

//             if tb_sub_rank > 0 {
//                 for v in 0..n {
//                     if var_rank[v] > cur_rank { var_rank[v] += tb_sub_rank; }
//                 }
//                 for (i, &v) in tb_sorted.iter().enumerate() {
//                     var_rank[v] = cur_rank + tb_new_sub_ranks[i];
//                 }
//                 tb1_fired = true;
//                 break;
//             }
//         }

//         if tb1_fired { continue 'master; }

//         // Stage 3: scan singleton P_i in rank order, use monomials to split
//         let mut tb2_fired = false;
//         let mut poly_order: Vec<usize> = (0..n).collect();
//         poly_order.sort_by_key(|&i| (var_rank[i], i));

//         'tb2: for &poly_idx in &poly_order {
//             let wire_rank = var_rank[poly_idx];
//             let is_singleton = (0..n).filter(|&v| var_rank[v] == wire_rank).count() == 1;
//             if !is_singleton { continue; }

//             let ranked = ranked_monomials_of_poly(&polynomials[poly_idx], &var_rank);

//             for &(mono, _coeff) in &ranked {
//                 let max_rank = *var_rank.iter().max().unwrap_or(&0);
//                 for cur_rank in 0..=max_rank {
//                     let tied: Vec<usize> = (0..n)
//                         .filter(|&v| var_rank[v] == cur_rank)
//                         .collect();
//                     if tied.len() <= 1 { continue; }

//                     let in_mono:  Vec<usize> = tied.iter().copied()
//                         .filter(|&v| mono & (1u64 << v) != 0)
//                         .collect();
//                     let out_mono: Vec<usize> = tied.iter().copied()
//                         .filter(|&v| mono & (1u64 << v) == 0)
//                         .collect();

//                     if in_mono.is_empty() || out_mono.is_empty() { continue; }

//                     for v in 0..n {
//                         if var_rank[v] > cur_rank { var_rank[v] += 1; }
//                     }
//                     for &v in &out_mono {
//                         var_rank[v] = cur_rank + 1;
//                     }
//                     tb2_fired = true;
//                     break 'tb2;
//                 }
//             }
//         }

//         if tb2_fired { continue 'master; }

//         break 'master;
//     }

//     // ── Step 4: build final_order ─────────────────────────────────────────────
//     let mut final_order: Vec<usize> = (0..n).collect();
//     final_order.sort_by_key(|&w| (var_rank[w], w));

//     // ── Step 5: remap polynomials ─────────────────────────────────────────────
//     let mut wire_to_pos = vec![0usize; n];
//     for (pos, &wire) in final_order.iter().enumerate() {
//         wire_to_pos[wire] = pos;
//     }

//     let remap_monomial = |m: Monomial| -> Monomial {
//         let mut result = 0u64;
//         for wire in 0..n {
//             if m & (1u64 << wire) != 0 {
//                 result |= 1u64 << wire_to_pos[wire];
//             }
//         }
//         result
//     };

//     let canonical: Vec<Polynomial> = final_order
//         .iter()
//         .map(|&wire| polynomials[wire].iter().map(|&m| remap_monomial(m)).collect())
//         .collect();

//     let canonical = trim_canonicalized(canonical);
//     (canonical, Permutation { data: final_order })
// }

// ---- Poly display helpers (restored from eb/dev for the challenge binaries) ----
pub fn monomial_degree(m: u64) -> u32 {
    m.count_ones()
}

fn mono_compressed_str(m: u64, n: usize) -> String {
    if m == 0 {
        return "I".into();
    }
    (0..n)
        .filter(|&i| (m >> i) & 1 == 1)
        .map(|i| format!("{}", i))
        .collect::<Vec<_>>()
        .join("•")
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
        return "I".to_string();
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

pub fn poly_to_compressed_str(poly: &Polynomial, n: usize) -> String {
    if poly.is_empty() {
        return "i".into();
    }
    let mut terms: Vec<u64> = poly.iter().copied().collect();
    terms.sort_by_key(|&m| (monomial_degree(m), m));
    terms
        .iter()
        .map(|&m| mono_compressed_str(m, n))
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::random_data::random_circuit;
    use std::collections::BTreeSet;

    #[test]
    fn polynomial_from_terms_sorts_and_cancels_pairs() {
        assert_eq!(polynomial_from_terms([5, 3, 5, 1, 3, 5, 7, 7]), vec![1, 5]);
    }

    #[test]
    fn polynomial_xor_assign_cancels_shared_terms() {
        let mut left = vec![1, 3, 8];
        poly_xor_assign(&mut left, vec![3, 5, 8]);
        assert_eq!(left, vec![1, 5]);
    }

    #[test]
    fn to_polynomial_keeps_terms_sorted_and_cancelled() {
        let circuit = CircuitSeq {
            gates: vec![[0, 1, 2]],
        };
        let polys = circuit.to_polynomial(3, 0, 1);
        assert_eq!(polys[0], vec![0, 1, 2, 6]);
        assert_eq!(polys[1], vec![2]);
        assert_eq!(polys[2], vec![4]);
    }

    fn old_toggle(poly: &mut BTreeSet<Monomial>, m: Monomial) {
        if !poly.remove(&m) {
            poly.insert(m);
        }
    }

    fn old_xor(mut left: BTreeSet<Monomial>, right: BTreeSet<Monomial>) -> BTreeSet<Monomial> {
        for m in right {
            old_toggle(&mut left, m);
        }
        left
    }

    fn old_and(left: &BTreeSet<Monomial>, right: &BTreeSet<Monomial>) -> BTreeSet<Monomial> {
        let mut result = BTreeSet::new();
        for &m1 in left {
            for &m2 in right {
                old_toggle(&mut result, m1 | m2);
            }
        }
        result
    }

    fn old_not(poly: BTreeSet<Monomial>) -> BTreeSet<Monomial> {
        old_xor(BTreeSet::from([0u64]), poly)
    }

    fn old_hashset_style_to_polynomial(circuit: &CircuitSeq, n: usize) -> Vec<Polynomial> {
        let mut polys: Vec<BTreeSet<Monomial>> =
            (0..n).map(|i| BTreeSet::from([1u64 << i])).collect();

        for &[a, b, c] in &circuit.gates {
            let not_c = old_not(polys[c as usize].clone());
            let term = old_and(&polys[b as usize], &not_c);
            let mut new_a = old_xor(polys[a as usize].clone(), term);
            old_toggle(&mut new_a, 0u64);
            polys[a as usize] = new_a;
        }

        polys
            .into_iter()
            .map(|poly| poly.into_iter().collect())
            .collect()
    }

    #[test]
    fn to_polynomial_matches_old_hashset_style_implementation() {
        let mut rng = fastrand::Rng::with_seed(0x706f_6c79_7665_6375);
        for _ in 0..200 {
            let n = rng.usize(3..=12);
            let m = rng.usize(0..=(3 * n));
            fastrand::seed(rng.u64(..));
            let circuit = random_circuit(n, m);

            assert_eq!(
                circuit.to_polynomial(n, 0, circuit.gates.len()),
                old_hashset_style_to_polynomial(&circuit, n)
            );
        }
    }
}
