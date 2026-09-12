//! G57 gate triples, circuit sequence edits and wire coordinates.
use super::Permutation;
use serde::{Deserialize, Serialize};
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
}

impl CircuitSeq {
    /// True if any two adjacent gates are identical. Two identical adjacent
    /// self-inverse gates cancel, so a canonicalized circuit containing an
    /// adjacent duplicate is not minimal — `db_generation::regular` uses this
    /// to reject such candidates while rebuilding the replacement DB.
    pub fn adjacent_id(&self) -> bool {
        if self.gates.is_empty() {
            return false;
        }
        for i in 0..self.gates.len() - 1 {
            if self.gates[i] == self.gates[i + 1] {
                return true;
            }
        }
        false
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

    // Combine two circuits
    pub fn concat(&self, other: &CircuitSeq) -> CircuitSeq {
        let mut gates = self.gates.clone();
        gates.extend_from_slice(&other.gates);
        CircuitSeq { gates }
    }

    // Returns the wires touched by a circuit
    pub fn used_wires(&self) -> Vec<u16> {
        // Stack-bitset fast path: mark wires in a [u64; 16] (covers wires
        // 0..1023, the overwhelmingly common case) in a single pass, then
        // emit the identical sorted list from the set bits. Falls back to the
        // heap-marking implementation the moment any wire is out of range.
        let mut words = [0u64; 16];
        for &[t, a, b] in &self.gates {
            if t >= 1024 || a >= 1024 || b >= 1024 {
                return self.used_wires_heap();
            }
            words[(t >> 6) as usize] |= 1u64 << (t & 63);
            words[(a >> 6) as usize] |= 1u64 << (a & 63);
            words[(b >> 6) as usize] |= 1u64 << (b & 63);
        }
        let count: usize = words.iter().map(|w| w.count_ones() as usize).sum();
        let mut out = Vec::with_capacity(count);
        for (wi, &word) in words.iter().enumerate() {
            let base = (wi as u16) << 6;
            let mut word = word;
            while word != 0 {
                out.push(base + word.trailing_zeros() as u16);
                word &= word - 1;
            }
        }
        out
    }

    // Heap fallback for circuits touching wires >= 1024 (u16 wires cap at
    // 65535). Identical to the historical implementation.
    fn used_wires_heap(&self) -> Vec<u16> {
        let Some(max_wire) = self.gates.iter().flatten().copied().max() else {
            return Vec::new();
        };
        let mut used = vec![false; max_wire as usize + 1];
        for &[target, control_a, control_b] in &self.gates {
            used[target as usize] = true;
            used[control_a as usize] = true;
            used[control_b as usize] = true;
        }
        used.into_iter()
            .enumerate()
            .filter_map(|(wire, is_used)| is_used.then_some(wire as u16))
            .collect()
    }

    /// Number of distinct wires touched by the circuit, without materializing
    /// the sorted wire list. Equals `self.used_wires().len()`.
    pub fn used_wires_len(&self) -> usize {
        let mut words = [0u64; 16];
        for &[t, a, b] in &self.gates {
            if t >= 1024 || a >= 1024 || b >= 1024 {
                return self.used_wires_heap().len();
            }
            words[(t >> 6) as usize] |= 1u64 << (t & 63);
            words[(a >> 6) as usize] |= 1u64 << (a & 63);
            words[(b >> 6) as usize] |= 1u64 << (b & 63);
        }
        words.iter().map(|w| w.count_ones() as usize).sum()
    }

    // "Bottom" function for gates
    pub fn max_wire(&self) -> usize {
        self.gates.iter().flatten().copied().max().unwrap_or(0) as usize
    }

    // Undo rewiring. Note: Recall that the number of wires in CircuitSeq is not stored
    pub fn unrewire_subcircuit(subcircuit: &CircuitSeq, used_wires: &[u16]) -> CircuitSeq {
        // Replace wires in each gate with original wires
        let new_gates: Vec<[u16; 3]> = subcircuit
            .gates
            .iter()
            .map(|&[t, c1, c2]| {
                [
                    used_wires[t as usize],
                    used_wires[c1 as usize],
                    used_wires[c2 as usize],
                ]
            })
            .collect();

        CircuitSeq { gates: new_gates }
    }
}
// Choose the smallest lexigraphical ordering
impl CircuitSeq {
    pub fn canonicalize(&mut self) {
        for i in 1..self.gates.len() {
            //index in base_gates of current gate
            let gi_index = self.gates[i];
            let mut to_swap: Option<usize> = None;

            let mut j = i;
            while j > 0 {
                j -= 1;
                let gj_index = self.gates[j];

                if Gate::collides_index(&gi_index, &gj_index) {
                    break;
                } else if !Gate::ordered_index(&gj_index, &gi_index) {
                    to_swap = Some(j);
                }
            }
            if let Some(pos) = to_swap {
                let g = self.gates[i];
                self.gates.remove(i);
                self.gates.insert(pos, g);
            }
        }
    }
}

/// Cancel adjacent duplicate gates to a fixed point, keeping `tags` (when
/// present) in lockstep with the gate vector. Stack-style single pass:
/// produces the identical final sequence to the historical
/// drain-with-backtrack loop (`aa -> empty` rewriting is confluent, and the
/// backtrack re-examined exactly the stack top) without the O(n) tail
/// memmove per removal.
pub fn cancel_adjacent_duplicates<T: Copy>(
    gates: &mut Vec<[u16; 3]>,
    mut tags: Option<&mut Vec<T>>,
) {
    let mut write = 0usize;
    for read in 0..gates.len() {
        if write > 0 && gates[write - 1] == gates[read] {
            write -= 1;
        } else {
            gates[write] = gates[read];
            if let Some(tags) = tags.as_deref_mut() {
                tags[write] = tags[read];
            }
            write += 1;
        }
    }
    gates.truncate(write);
    if let Some(tags) = tags {
        tags.truncate(write);
    }
}
