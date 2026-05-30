// For adding wire shuffles and bit flips
use std::collections::HashMap;
use rand::Rng;
use rand::seq::IndexedRandom;
use lmdb::{Database, Environment};
use crate::circuit::{circuit::CircuitSeq, Permutation};
use crate::circuit::circuit::rewire_gate_ver;

// Hardcoded circuits: min-2 depths per function, 3-wire and 4-wire.
// Wire convention: wire 1↔2 swapped (swap), wire 1 flipped (not), wire 1 controls wire 2 (cnot).
// Wire 0 (and wire 3 in 4-wire) are ancilla.
static SWAP_3W: &[&[[u16;3]]] = &[
    &[[1,0,2],[2,0,1],[2,1,0],[1,0,2],[1,2,0],[2,1,0]],
    &[[0,1,2],[2,0,1],[2,1,0],[1,0,2],[1,2,0],[2,0,1],[2,1,0],[0,2,1]],
    &[[1,2,0],[2,1,0],[1,0,2],[1,2,0],[2,0,1],[2,1,0],[1,0,2],[2,1,0]],
    &[[0,2,1],[1,2,0],[2,0,1],[2,1,0],[1,0,2],[1,2,0],[2,0,1],[0,1,2]],
    &[[2,1,0],[1,0,2],[1,2,0],[2,0,1],[2,1,0],[1,0,2]],
    &[[0,1,2],[1,0,2],[1,2,0],[2,0,1],[2,1,0],[1,0,2],[1,2,0],[0,2,1]],
    &[[0,2,1],[2,0,1],[2,1,0],[1,0,2],[1,2,0],[2,0,1],[2,1,0],[0,1,2]],
    &[[2,0,1],[2,1,0],[1,0,2],[1,2,0],[2,0,1],[2,1,0]],
    &[[2,0,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0],[1,2,0]],
    &[[1,0,2],[2,1,0],[1,0,2],[1,2,0],[2,0,1],[2,1,0],[1,0,2],[2,0,1]],
    &[[0,1,2],[1,2,0],[2,0,1],[2,1,0],[1,0,2],[1,2,0],[2,0,1],[0,2,1]],
    &[[0,2,1],[2,0,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0],[1,2,0],[0,1,2]],
    &[[2,1,0],[1,0,2],[2,0,1],[2,1,0],[1,0,2],[1,2,0],[2,1,0],[1,2,0]],
    &[[2,1,0],[1,2,0],[2,0,1],[2,1,0],[1,0,2],[1,2,0],[2,0,1],[1,2,0]],
    &[[0,1,2],[2,0,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0],[1,2,0],[0,2,1]],
    &[[1,2,0],[2,0,1],[2,1,0],[1,0,2],[1,2,0],[2,0,1]],
    &[[1,0,2],[2,0,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0],[1,2,0],[2,0,1]],
    &[[1,2,0],[2,0,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0],[1,2,0],[2,1,0]],
    &[[2,0,1],[1,0,2],[2,0,1],[2,1,0],[1,0,2],[1,2,0],[2,1,0],[1,0,2]],
    &[[2,0,1],[1,2,0],[2,0,1],[2,1,0],[1,0,2],[1,2,0],[2,0,1],[1,0,2]],
    &[[0,1,2],[2,1,0],[1,0,2],[1,2,0],[2,0,1],[2,1,0],[1,0,2],[0,2,1]],
    &[[1,0,2],[1,2,0],[2,0,1],[2,1,0],[1,0,2],[1,2,0]],
];
static SWAP_4W: &[&[[u16;3]]] = &[
    &[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[1,2,3],[1,3,2]],
    &[[1,2,3],[2,1,3],[2,3,1],[1,2,3],[1,3,2],[2,3,1]],
    &[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[1,0,2],[1,0,3],[1,2,0],[1,3,0]],
    &[[2,0,1],[2,1,0],[1,0,2],[1,2,0],[2,0,1],[2,1,0]],
    &[[2,0,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0],[1,2,0]],
    &[[1,3,2],[2,1,3],[2,3,1],[1,2,3],[1,3,2],[2,1,3]],
    &[[1,2,0],[2,0,1],[2,1,0],[1,0,2],[1,2,0],[2,0,1]],
];
static SWAP_N1_3W: &[&[[u16;3]]] = &[
    &[[1,2,0],[2,1,0],[1,2,0],[2,0,1],[1,0,2],[2,0,1]],
    &[[1,0,2],[2,0,1],[1,0,2],[2,1,0],[1,2,0],[2,1,0]],
    &[[1,0,2],[1,0,2],[1,2,0],[2,1,0],[1,2,0],[2,0,1],[1,0,2],[2,0,1]],
    &[[0,1,2],[1,0,2],[2,1,0],[1,2,0],[0,2,1],[2,1,0],[1,2,0],[2,0,1]],
    &[[2,0,1],[1,2,0],[2,1,0],[1,2,0],[2,0,1],[1,0,2],[2,0,1],[1,0,2]],
    &[[1,2,0],[0,1,2],[1,2,0],[2,1,0],[1,2,0],[2,0,1],[1,0,2],[0,1,2]],
    &[[2,1,0],[1,2,0],[2,1,0],[1,2,0],[2,0,1],[1,0,2],[2,0,1],[1,2,0]],
];
static SWAP_N1_4W: &[&[[u16;3]]] = &[
    &[[1,0,3],[1,2,3],[2,1,3],[1,2,3],[2,3,1],[1,3,2],[2,0,3],[2,3,1]],
    &[[1,3,2],[2,0,1],[1,0,2],[2,1,0],[1,2,0],[2,1,3]],
];
static SWAP_N2_3W: &[&[[u16;3]]] = &[
    &[[2,0,1],[1,2,0],[2,1,0],[0,2,1],[1,2,0],[2,1,0],[1,0,2],[0,1,2]],
    &[[0,2,1],[2,0,1],[1,2,0],[2,1,0],[0,1,2],[1,2,0],[2,1,0],[1,0,2]],
    &[[2,0,1],[1,0,2],[2,0,1],[1,2,0],[2,1,0],[1,2,0]],
    &[[0,1,2],[1,0,2],[2,0,1],[1,2,0],[2,1,0],[1,2,0],[0,1,2],[1,2,0]],
    &[[0,2,1],[1,2,0],[2,1,0],[1,0,2],[2,0,1],[1,0,2],[0,2,1],[1,0,2]],
    &[[2,1,0],[1,0,2],[2,0,1],[0,1,2],[1,0,2],[2,0,1],[1,2,0],[0,2,1]],
    &[[0,2,1],[1,0,2],[2,0,1],[0,1,2],[1,0,2],[2,0,1],[1,0,2],[1,2,0]],
    &[[2,0,1],[0,2,1],[1,0,2],[2,0,1],[1,2,0],[2,1,0],[0,2,1],[1,2,0]],
    &[[2,1,0],[1,2,0],[2,1,0],[1,0,2],[2,0,1],[1,0,2]],
];
static SWAP_N2_4W: &[&[[u16;3]]] = &[
    &[[2,1,0],[1,2,3],[2,1,3],[1,3,2],[2,3,1],[1,0,2]],
    &[[2,1,3],[1,2,0],[2,1,0],[1,0,2],[2,0,1],[1,3,2]],
];
static SWAP_N12_3W: &[&[[u16;3]]] = &[
    &[[2,0,1],[2,1,0],[0,2,1],[1,0,2],[2,0,1],[0,1,2],[1,0,2],[2,0,1]],
    &[[0,1,2],[1,0,2],[2,1,0],[1,2,0],[2,0,1],[1,0,2],[2,1,0],[0,1,2]],
    &[[0,1,2],[1,0,2],[2,0,1],[1,2,0],[0,2,1],[1,2,0],[2,0,1],[1,0,2]],
    &[[1,0,2],[1,2,0],[0,2,1],[2,1,0],[1,2,0],[0,1,2],[2,1,0],[1,2,0]],
    &[[2,1,0],[1,2,0],[0,2,1],[2,1,0],[1,2,0],[0,1,2],[2,0,1],[2,1,0]],
    &[[1,2,0],[2,0,1],[1,0,2],[2,1,0],[1,2,0],[2,0,1]],
    &[[0,2,1],[1,2,0],[2,1,0],[1,0,2],[0,1,2],[1,0,2],[2,1,0],[1,2,0]],
    &[[1,0,2],[2,1,0],[1,2,0],[2,0,1],[1,0,2],[2,1,0]],
    &[[1,0,2],[2,0,1],[1,2,0],[0,2,1],[1,2,0],[2,0,1],[1,0,2],[0,1,2]],
    &[[1,0,2],[0,2,1],[2,1,0],[1,2,0],[2,0,1],[1,0,2],[0,2,1],[2,1,0]],
    &[[0,1,2],[2,1,0],[1,0,2],[2,0,1],[1,2,0],[2,1,0],[1,0,2],[0,1,2]],
    &[[2,0,1],[1,2,0],[2,1,0],[1,0,2],[2,0,1],[1,2,0]],
    &[[1,2,0],[2,1,0],[0,1,2],[1,2,0],[2,1,0],[0,2,1],[1,0,2],[1,2,0]],
    &[[1,2,0],[2,1,0],[1,0,2],[0,1,2],[1,0,2],[2,1,0],[1,2,0],[0,2,1]],
    &[[2,1,0],[1,0,2],[2,0,1],[1,2,0],[2,1,0],[1,0,2]],
    &[[0,1,2],[1,2,0],[2,0,1],[1,0,2],[2,1,0],[1,2,0],[2,0,1],[0,1,2]],
];
static SWAP_N12_4W: &[&[[u16;3]]] = &[
    &[[1,3,2],[2,1,3],[1,2,3],[2,3,1],[1,3,2],[2,1,3]],
    &[[2,1,3],[1,3,2],[2,3,1],[1,2,3],[2,1,3],[1,3,2]],
    &[[1,2,0],[2,3,1],[1,3,2],[2,1,3],[1,2,3],[2,0,1]],
];
// NOT 3-wire at min-2 depths (6,7) is empty — those depths only exist for 4-wire.
static NOT_3W: &[&[[u16;3]]] = &[];
static NOT_4W: &[&[[u16;3]]] = &[
    &[[0,2,3],[1,0,2],[1,0,3],[0,2,3],[1,2,0],[1,2,3],[1,3,0]],
    &[[1,0,2],[1,0,3],[1,3,2],[2,3,0],[1,0,2],[1,3,2],[2,3,0]],
    &[[1,0,2],[1,0,3],[3,0,2],[0,2,3],[1,3,0],[0,2,3],[3,0,2]],
    &[[2,0,3],[1,2,0],[1,3,2],[2,0,3],[1,0,2],[1,2,3]],
    &[[0,2,3],[1,0,2],[1,3,0],[0,2,3],[1,0,3],[1,2,0]],
    &[[1,0,2],[2,3,0],[3,0,2],[1,0,3],[3,0,2],[1,0,3],[2,3,0]],
    &[[1,0,2],[1,3,0],[2,0,3],[1,0,2],[1,3,2],[2,0,3],[1,3,2]],
    &[[1,0,3],[2,0,3],[1,0,2],[2,0,3],[2,3,0],[1,0,2],[2,3,0]],
    &[[0,1,3],[0,3,2],[1,3,0],[0,3,2],[1,3,2],[0,1,3],[1,3,0]],
    &[[1,2,0],[1,2,3],[3,2,0],[1,3,0],[1,3,2],[3,2,0],[1,0,3]],
    &[[1,0,2],[1,0,3],[1,3,2],[2,0,3],[1,2,0],[1,2,3],[2,0,3]],
    &[[1,2,0],[1,3,2],[2,0,3],[1,0,2],[1,2,3],[2,0,3]],
    &[[1,3,0],[2,0,3],[2,3,0],[1,2,0],[2,0,3],[2,3,0],[1,2,0]],
    &[[1,0,2],[1,2,3],[2,0,3],[1,2,0],[1,3,2],[2,0,3]],
    &[[1,0,2],[3,0,2],[3,2,0],[1,3,2],[3,0,2],[3,2,0],[1,3,2]],
    &[[1,0,2],[1,3,0],[0,2,3],[1,0,3],[1,2,0],[0,2,3]],
    &[[3,0,2],[1,2,3],[3,0,2],[1,3,0],[0,3,2],[1,3,0],[0,3,2]],
    &[[0,2,1],[0,3,2],[1,0,2],[0,3,2],[1,3,2],[0,2,1],[1,0,2]],
    &[[0,3,2],[1,0,2],[1,3,0],[0,3,2],[1,0,3],[1,2,0]],
    &[[1,0,2],[1,3,2],[3,2,0],[0,3,2],[1,0,2],[0,3,2],[3,2,0]],
    &[[0,2,1],[1,3,2],[3,0,2],[1,3,2],[3,0,2],[0,2,1],[1,0,2]],
    &[[0,3,2],[1,2,0],[1,2,3],[1,3,0],[0,3,2],[1,2,0],[1,3,0]],
    &[[3,2,0],[1,0,3],[1,3,2],[3,2,0],[1,2,3],[1,3,0]],
    &[[1,0,2],[3,2,0],[1,0,3],[1,2,3],[3,2,0],[1,0,3],[1,2,3]],
    &[[1,0,3],[1,2,0],[0,3,2],[1,0,2],[1,3,0],[0,3,2]],
    &[[2,0,3],[1,0,2],[1,2,3],[2,0,3],[1,2,0],[1,3,2]],
    &[[1,0,2],[0,3,2],[1,0,3],[1,2,0],[0,3,2],[1,3,0]],
    &[[1,0,2],[1,2,3],[2,3,0],[1,2,0],[1,3,2],[2,3,0]],
    &[[3,0,2],[1,3,2],[3,0,2],[2,0,3],[1,2,0],[2,0,3],[1,3,2]],
    &[[1,0,3],[1,3,2],[2,0,3],[1,2,0],[1,2,3],[2,0,3],[1,0,2]],
    &[[2,3,0],[1,2,0],[1,3,2],[2,3,0],[1,0,2],[1,2,3]],
    &[[0,1,3],[1,0,2],[2,0,3],[1,0,2],[2,0,3],[0,1,3],[1,3,0]],
    &[[1,0,2],[3,0,2],[1,0,3],[1,2,3],[3,0,2],[1,3,0],[1,3,2]],
    &[[1,3,0],[2,0,3],[1,2,0],[2,0,3],[2,3,0],[1,2,0],[2,3,0]],
];
// CNOT 3-wire at min-2 depths (4,5) is empty — those depths only exist for 4-wire.
static CNOT_3W: &[&[[u16;3]]] = &[];
static CNOT_4W: &[&[[u16;3]]] = &[
    &[[1,3,0],[2,3,0],[2,3,1],[1,3,0],[2,1,3]],
    &[[2,0,1],[1,0,3],[2,1,0],[1,0,3]],
    &[[2,3,1],[3,1,0],[2,3,1],[3,1,0]],
    &[[2,1,3],[2,3,0],[0,3,1],[2,3,0],[0,3,1]],
    &[[2,1,0],[3,1,0],[2,1,3],[3,1,0],[2,1,3]],
    &[[1,3,0],[2,1,3],[1,3,0],[2,3,1]],
    &[[0,3,1],[2,1,3],[2,3,0],[0,3,1],[2,3,0]],
    &[[2,0,1],[0,1,3],[2,0,1],[0,1,3]],
    &[[0,1,3],[2,0,1],[0,1,3],[2,0,1]],
    &[[3,1,0],[2,3,1],[3,1,0],[2,3,1]],
    &[[2,0,3],[2,1,0],[3,0,1],[2,0,3],[3,0,1]],
    &[[2,1,0],[0,1,3],[2,1,0],[0,1,3],[2,1,3]],
    &[[0,1,3],[2,1,0],[0,1,3],[2,1,0],[2,1,3]],
    &[[2,0,3],[2,1,0],[1,0,3],[2,0,1],[1,0,3]],
    &[[2,1,0],[3,0,1],[2,0,3],[3,0,1],[2,0,3]],
    &[[1,0,3],[2,0,1],[1,0,3],[2,0,3],[2,1,0]],
    &[[1,0,3],[2,1,0],[1,0,3],[2,0,1]],
    &[[2,1,0],[2,1,3],[3,1,0],[2,1,3],[3,1,0]],
    &[[2,1,3],[1,3,0],[2,3,0],[2,3,1],[1,3,0]],
    &[[2,3,1],[1,3,0],[2,1,3],[1,3,0]],
];
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Transpositions {
    pub transpositions: Vec<(u16, u16, u16)>
}

impl Transpositions {
    // Use Knuth Shuffle to get a random wire shuffle and then choose a random negation
    pub fn gen_random_knuth(n: usize, _m: usize, negation_mask: &mut Vec<u8>) -> Self {
        assert!(n >= 2, "n must be at least 2");
        let mut rng = rand::rng();
        let mut transpositions = Vec::with_capacity(n);
        let n = (n - 1) as u16;
        for i in (1..=n).rev() {
            let negation_type = rng.random_range(0u16..=3);
            let j = rng.random_range(0..=i);
            if i == j {
                continue;
            }
            transpositions.push((j, i, negation_type));
            let temp = negation_mask[j as usize];
            negation_mask[j as usize] = negation_mask[i as usize];
            negation_mask[i as usize] = temp;
            if negation_type == 1 || negation_type == 3{
                negation_mask[j as usize] ^= 1;
            }
            if negation_type == 2 || negation_type == 3 {
                negation_mask[i as usize] ^= 1;
            }
        }

        Self { transpositions }
    }

    // Simple random wire shuffle with negation
    pub fn gen_random_simple(n: usize, m: usize, negation_mask: &mut Vec<u8>) -> Self {
        assert!(n >= 2, "n must be at least 2");
        let mut rng = rand::rng();
        let mut transpositions = Vec::with_capacity(m);
        for _ in 0..m {
            let negation_type = rng.random_range(0..=3);
            let mut i: usize = rng.random_range(0..n);
            let mut j: usize;
            loop {
                j = rng.random_range(0..n);
                if i != j {
                    break;
                }
            }
            // Maintain correct ordering
            if i < j {
                let temp = i;
                i = j;
                j = temp;
            }
            transpositions.push((j as u16, i as u16, negation_type as u16));
            // Adjust negation mask appropriately
            let temp = negation_mask[j];
            negation_mask[j as usize] = negation_mask[i as usize];
            negation_mask[i as usize] = temp;
            if negation_type == 1 || negation_type == 3{
                negation_mask[j as usize] ^= 1;
            }
            if negation_type == 2 || negation_type == 3 {
                negation_mask[i as usize] ^= 1;
            }
        }

        Self { transpositions }
    }

    // Simple random wire shuffle with negation
    // Restricted wires are never chosen
    pub fn gen_random_simple_restricted(n: usize, m: usize, negation_mask: &mut Vec<u8>, restricted: &Vec<usize>) -> Self {
        assert!(n >= 2, "n must be at least 2");
        let mut rng = rand::rng();
        let mut transpositions = Vec::with_capacity(m);
        for _ in 0..m {
            let negation_type = rng.random_range(0..=3);
            let mut i: usize;
            loop {
                i = rng.random_range(0..n);
                if !restricted.contains(&i) {
                    break;
                }
            }
            let mut j: usize;
            loop {
                j = rng.random_range(0..n);
                if i != j && !restricted.contains(&j) {
                    break;
                }
            }
            // Maintain correct ordering
            if i < j {
                let temp = i;
                i = j;
                j = temp;
            }
            transpositions.push((j as u16, i as u16, negation_type as u16));
            // Adjust negation mask appropriately
            let temp = negation_mask[j];
            negation_mask[j as usize] = negation_mask[i as usize];
            negation_mask[i as usize] = temp;
            if negation_type == 1 || negation_type == 3{
                negation_mask[j as usize] ^= 1;
            }
            if negation_type == 2 || negation_type == 3 {
                negation_mask[i as usize] ^= 1;
            }
        }

        Self { transpositions }
    }

    pub fn to_perm(&self, n: usize) -> Permutation {
        let mut perm = Permutation { data: Vec::with_capacity(n) };
        for i in 0..n {
            perm.data.push(self.evaluate(i as u16) as usize);
        } 
        perm
    }

    pub fn from_perm(perm: &Permutation) -> Self {
        let n = perm.data.len();
        let mut p = perm.data.clone();

        let mut inv = vec![0usize; n];
        for i in 0..n {
            inv[p[i]] = i;
        }

        let mut swaps = Vec::new();

        for i in (0..n).rev() {
            if p[i] != i {
                let j = inv[i];
                p.swap(i, j);
                inv[p[j]] = j;
                inv[p[i]] = i;

                swaps.push((i as u16, j as u16, 0u16));
            }
        }

        Transpositions { transpositions: swaps }
    }

    pub fn collides(s1: &(u16, u16, u16), s2: &(u16, u16, u16)) -> bool {
        let (a1, b1, _) = s1;
        let (a2, b2, _) = s2;
        a1 == a2 ||
        a1 == b2 ||
        b1 == a2 ||
        b1 == b2
    }

    // Simple randomization
    // Unused
    pub fn shoot_random_transpositions(transpositions: &mut Transpositions, rounds: usize) {
        let mut rng = rand::rng();
        let len = transpositions.transpositions.len();

        if len == 0 {
            return
        }

        for _ in 0..rounds {
            let gate_idx = rng.random_range(0..len);
            let go_left: bool = rng.random_bool(0.5);

            if go_left {
                // Shoot left
                let mut target = gate_idx;
                while target > 0 {
                    if Transpositions::collides(&transpositions.transpositions[target - 1], &transpositions.transpositions[gate_idx]) {
                        break;
                    }
                    target -= 1;
                }
                target = rng.random_range(target..=gate_idx);
                if target != gate_idx {
                    let gate = transpositions.transpositions.remove(gate_idx);
                    transpositions.transpositions.insert(target, gate);
                }
            } else {
                // Shoot right
                let mut target = gate_idx;
                while target + 1 < len {
                    if Transpositions::collides(&transpositions.transpositions[target + 1], &transpositions.transpositions[gate_idx]) {
                        break;
                    }
                    target += 1;
                }
                target = rng.random_range(gate_idx..=target);
                if target != gate_idx {
                    let gate = transpositions.transpositions.remove(gate_idx);
                    transpositions.transpositions.insert(target, gate);
                }
            }
        }
    }

    //b is greater
    pub fn ordered(s1: &(u16, u16, u16), s2: &(u16, u16, u16)) -> bool {
        let (a_1, b_1, _) = s1;
        let (a_2, b_2, _) = s2;
        if a_1 > a_2 {
            return false
        } else if a_1 == a_2{
            if b_1 > b_2 {
                return false
            }
        }
        true
    }

    // Unused
    // Smallest lexicographical ordering
    pub fn canonicalize(&mut self) {
        for i in 1..self.transpositions.len() {
            let ti = self.transpositions[i];
            let mut to_swap: Option<usize> = None;

            let mut j = i;
            while j > 0 {
                j -= 1;
                let tj = self.transpositions[j];

                if Transpositions::collides(&ti, &tj) {
                    break;
                } else if !Transpositions::ordered(&tj, &ti) {
                    to_swap = Some(j);
                }
            }
            if let Some(pos) = to_swap {
                let g = self.transpositions[i];
                self.transpositions.remove(i);
                self.transpositions.insert(pos, g);
            }
        }
    }

    // Swaps wire 1 and wire 2 in the template; relabels to swap.0 and swap.1.
    // Randomly selects from hardcoded 3-wire or 4-wire circuits at min depths.
    pub fn gen_gates_swap(
        n: usize,
        swap: (u16, u16, u16),
        _env: &lmdb::Environment,
        _dbs: &HashMap<String, Database>,
    ) -> Vec<[u16;3]> {
        let (a, b, negation_type) = swap;
        let (pool_3w, pool_4w): (&[&[[u16;3]]], &[&[[u16;3]]]) = match negation_type {
            0 => (SWAP_3W, SWAP_4W),
            1 => (SWAP_N1_3W, SWAP_N1_4W),
            2 => (SWAP_N2_3W, SWAP_N2_4W),
            3 => (SWAP_N12_3W, SWAP_N12_4W),
            _ => panic!("Invalid negation type"),
        };
        let mut rng = rand::rng();
        let use_4w = n >= 4 && !pool_4w.is_empty();
        let total = pool_3w.len() + if use_4w { pool_4w.len() } else { 0 };
        let idx = rng.random_range(0..total);
        if idx < pool_3w.len() {
            let circuit = pool_3w[idx];
            let mut c;
            loop { c = rng.random_range(0..n) as u16; if c != a && c != b { break; } }
            let out = CircuitSeq { gates: circuit.to_vec() };
            CircuitSeq::unrewire_subcircuit(&out, &[c, a, b]).gates
        } else {
            let circuit = pool_4w[idx - pool_3w.len()];
            let mut c1;
            loop { c1 = rng.random_range(0..n) as u16; if c1 != a && c1 != b { break; } }
            let mut c2;
            loop { c2 = rng.random_range(0..n) as u16; if c2 != a && c2 != b && c2 != c1 { break; } }
            let out = CircuitSeq { gates: circuit.to_vec() };
            CircuitSeq::unrewire_subcircuit(&out, &[c1, a, b, c2]).gates
        }
    }

    pub fn gen_gates_swap_restricted(
        n: usize,
        swap: (u16, u16, u16),
        _env: &lmdb::Environment,
        _dbs: &HashMap<String, Database>,
        restricted: &Vec<usize>,
    ) -> Vec<[u16;3]> {
        let (a, b, negation_type) = swap;
        let (pool_3w, pool_4w): (&[&[[u16;3]]], &[&[[u16;3]]]) = match negation_type {
            0 => (SWAP_3W, SWAP_4W),
            1 => (SWAP_N1_3W, SWAP_N1_4W),
            2 => (SWAP_N2_3W, SWAP_N2_4W),
            3 => (SWAP_N12_3W, SWAP_N12_4W),
            _ => panic!("Invalid negation type"),
        };
        let mut rng = rand::rng();
        let use_4w = n >= 4 && !pool_4w.is_empty();
        let total = pool_3w.len() + if use_4w { pool_4w.len() } else { 0 };
        let idx = rng.random_range(0..total);
        if idx < pool_3w.len() {
            let circuit = pool_3w[idx];
            let mut c;
            loop {
                c = rng.random_range(0..n) as u16;
                if c != a && c != b && !restricted.contains(&(c as usize)) { break; }
            }
            let out = CircuitSeq { gates: circuit.to_vec() };
            CircuitSeq::unrewire_subcircuit(&out, &[c, a, b]).gates
        } else {
            let circuit = pool_4w[idx - pool_3w.len()];
            let mut c1;
            loop {
                c1 = rng.random_range(0..n) as u16;
                if c1 != a && c1 != b && !restricted.contains(&(c1 as usize)) { break; }
            }
            let mut c2;
            loop {
                c2 = rng.random_range(0..n) as u16;
                if c2 != a && c2 != b && c2 != c1 && !restricted.contains(&(c2 as usize)) { break; }
            }
            let out = CircuitSeq { gates: circuit.to_vec() };
            CircuitSeq::unrewire_subcircuit(&out, &[c1, a, b, c2]).gates
        }
    }

    // Wire 1 gets flipped in the template; relabels wire 1 to `wire`.
    // Uses 4-wire circuits (min depth 6-7) with 3 ancilla wires.
    pub fn gen_gates_not(
        n: usize,
        wire: u16,
        _env: &lmdb::Environment,
        _dbs: &HashMap<String, Database>,
    ) -> Vec<[u16;3]> {
        let mut rng = rand::rng();
        let pool = if !NOT_4W.is_empty() { NOT_4W } else { NOT_3W };
        let circuit = pool.choose(&mut rng).expect("NOT pool is empty");
        if pool.as_ptr() == NOT_4W.as_ptr() {
            let mut a; loop { a = rng.random_range(0..n) as u16; if a != wire { break; } }
            let mut b; loop { b = rng.random_range(0..n) as u16; if b != wire && b != a { break; } }
            let mut c; loop { c = rng.random_range(0..n) as u16; if c != wire && c != a && c != b { break; } }
            let out = CircuitSeq { gates: circuit.to_vec() };
            CircuitSeq::unrewire_subcircuit(&out, &[a, wire, b, c]).gates
        } else {
            let mut a; loop { a = rng.random_range(0..n) as u16; if a != wire { break; } }
            let mut b; loop { b = rng.random_range(0..n) as u16; if b != wire && b != a { break; } }
            let out = CircuitSeq { gates: circuit.to_vec() };
            CircuitSeq::unrewire_subcircuit(&out, &[a, wire, b]).gates
        }
    }

    // Wire 2 gets flipped if wire 1 is true in the template; relabels to con/not.
    // Uses 4-wire circuits (min depth 4-5) with 2 ancilla wires.
    pub fn gen_gates_cnot(
        n: usize,
        con: u16,
        not: u16,
        _env: &lmdb::Environment,
        _dbs: &HashMap<String, Database>,
    ) -> Vec<[u16;3]> {
        let mut rng = rand::rng();
        let pool = if !CNOT_4W.is_empty() { CNOT_4W } else { CNOT_3W };
        let circuit = pool.choose(&mut rng).expect("CNOT pool is empty");
        if pool.as_ptr() == CNOT_4W.as_ptr() {
            let mut c1; loop { c1 = rng.random_range(0..n) as u16; if c1 != con && c1 != not { break; } }
            let mut c2; loop { c2 = rng.random_range(0..n) as u16; if c2 != con && c2 != not && c2 != c1 { break; } }
            let out = CircuitSeq { gates: circuit.to_vec() };
            CircuitSeq::unrewire_subcircuit(&out, &[c1, con, not, c2]).gates
        } else {
            let mut c; loop { c = rng.random_range(0..n) as u16; if c != con && c != not { break; } }
            let out = CircuitSeq { gates: circuit.to_vec() };
            CircuitSeq::unrewire_subcircuit(&out, &[c, con, not]).gates
        }
    }

    pub fn to_circuit(
        &self,
        n: usize,
        env: &lmdb::Environment,
        dbs: &HashMap<String, Database>,
    ) -> CircuitSeq {
        let mut gates: Vec<[u16; 3]> = Vec::new();

        for &swap in &self.transpositions {
            gates.extend_from_slice(&Self::gen_gates_swap(n, swap, env, dbs));
        }

        CircuitSeq { gates }
    }

    pub fn restricted_to_circuit(
        &self,
        n: usize,
        env: &lmdb::Environment,
        dbs: &HashMap<String, Database>,
        restricted: &Vec<usize>
    ) -> CircuitSeq {
        let mut gates: Vec<[u16; 3]> = Vec::new();

        for &swap in &self.transpositions {
            gates.extend_from_slice(&Self::gen_gates_swap_restricted(n, swap, env, dbs, restricted));
        }

        CircuitSeq { gates }
    }

    pub fn restricted_to_circuit_ri(
        first: Transpositions,
        middle: Transpositions,
        second: Transpositions,
        n: usize,
        env: &lmdb::Environment,
        dbs: &HashMap<String, Database>,
        first_bounds: (usize, usize),
        second_bounds: (usize, usize),
        
    ) -> CircuitSeq {
        let mut gates: Vec<[u16; 3]> = Vec::new();

        for i in 0..first.transpositions.len() {
            let mut swap = first.transpositions[i];
            let n = first_bounds.1 - first_bounds.0 + 1;
            let offset = first_bounds.0 as u16;
            swap.0 -= offset;
            swap.1 -= offset;
            let first_circuit = Self::gen_gates_swap(n, swap, env, dbs);
            let first_circuit: Vec<[u16; 3]> = first_circuit
                .into_iter()
                .map(|[a, b, c]| [a + offset, b + offset, c + offset])
                .collect();

            gates.extend_from_slice(&first_circuit);
        }

        for i in (0..middle.transpositions.len()).rev() {
            let swap = middle.transpositions[i];
            let mut middle_circuit = Self::gen_gates_swap(n, swap, env, dbs);
            middle_circuit.reverse();
            gates.extend_from_slice(&middle_circuit);
        }

        for i in 0..second.transpositions.len() {
            let mut swap = second.transpositions[i];
            let n = second_bounds.1 - second_bounds.0 + 1;
            let offset = second_bounds.0 as u16;
            swap.0 -= offset;
            swap.1 -= offset;
            let second_circuit = Self::gen_gates_swap(n, swap, env, dbs);
            let second_circuit: Vec<[u16; 3]> = second_circuit
                .into_iter()
                .map(|[a, b, c]| [a + offset, b + offset, c + offset])
                .collect();
            
            gates.extend_from_slice(&second_circuit);
        }
        CircuitSeq { gates }
    }

    pub fn restricted_to_circuit_rewired_and_insert(
        t_rewired: Vec<(Transpositions, Permutation)>,
        c: CircuitSeq,
        n: usize,
        env: &lmdb::Environment,
        dbs: &HashMap<String, Database>,
        first_bounds: (usize, usize),
        second_bounds: (usize, usize), 
    ) -> CircuitSeq {
        let mut gates: Vec<[u16; 3]> = Vec::new();
        let first = &t_rewired[0].0;
        let mut first_gates: Vec<[u16;3]> = Vec::new();
        for i in 0..first.transpositions.len() {
            let mut swap = first.transpositions[i];
            if swap.0 == swap.1 {
                continue;
            }
            let n = first_bounds.1 - first_bounds.0 + 1;
            let offset = first_bounds.0 as u16;
            swap.0 -= offset;
            swap.1 -= offset;
            let first_circuit = Self::gen_gates_swap(n, swap, env, dbs);
            let first_circuit: Vec<[u16; 3]> = first_circuit
                .into_iter()
                .map(|[a, b, c]| [a + offset, b + offset, c + offset])
                .collect();

            first_gates.extend_from_slice(&first_circuit);
        }
        rewire_gate_ver(&mut first_gates, &t_rewired[0].1, n);
        let curr_len = first_gates.len();
        first_gates.insert(curr_len/2, c.gates[0]);
        gates.extend_from_slice(&first_gates);

        let t_len = t_rewired.len();
        let mut j = 1;
        while j < t_len-1 {
            // m's
            let middle = &t_rewired[j].0;
            let mut middle_gates: Vec<[u16;3]> = Vec::new();
            for i in (0..middle.transpositions.len()).rev() {
                let swap = middle.transpositions[i];
                if swap.0 == swap.1 {
                    continue;
                }
                let mut middle_circuit = Self::gen_gates_swap(n, swap, env, dbs);
                middle_circuit.reverse();
                middle_gates.extend_from_slice(&middle_circuit);
            }
            rewire_gate_ver(&mut middle_gates, &t_rewired[j].1, n);
            gates.extend_from_slice(&middle_gates);
            j += 1;
            if j == t_len-1 {
                break;
            }
            // B's
            let middle = &t_rewired[j].0;
            let mut middle_gates: Vec<[u16;3]> = Vec::new();
            for i in 0..middle.transpositions.len() {
                let swap = middle.transpositions[i];
                if swap.0 == swap.1 {
                    continue;
                }
                let middle_circuit = Self::gen_gates_swap_restricted(n, swap, env, dbs, &vec![13,14,15,29,30,31]);
                middle_gates.extend_from_slice(&middle_circuit);
            }
            rewire_gate_ver(&mut middle_gates, &t_rewired[j].1, n);
            let curr_len = middle_gates.len();
            middle_gates.insert(curr_len/2, c.gates[j/2]);
            gates.extend_from_slice(&middle_gates);
            j += 1;
        }
        
        let second = &t_rewired[t_rewired.len()-1];
        let mut second_gates: Vec<[u16;3]> = Vec::new();
        for i in 0..second.0.transpositions.len() {
            let mut swap = second.0.transpositions[i];
            if swap.0 == swap.1 {
                continue;
            }
            let n = second_bounds.1 - second_bounds.0 + 1;
            let offset = second_bounds.0 as u16;
            swap.0 -= offset;
            swap.1 -= offset;
            let second_circuit = Self::gen_gates_swap(n, swap, env, dbs);
            let second_circuit: Vec<[u16; 3]> = second_circuit
                .into_iter()
                .map(|[a, b, c]| [a + offset, b + offset, c + offset])
                .collect();
            
            second_gates.extend_from_slice(&second_circuit);
        }
        rewire_gate_ver(&mut second_gates, &t_rewired[t_rewired.len()-1].1, n);
        let curr_len = second_gates.len();
        let len = c.gates.len();
        second_gates.insert(curr_len/2, c.gates[len-1]);
        gates.extend_from_slice(&second_gates);
        CircuitSeq { gates }
    }

    pub fn restricted_to_circuit_rewired_and_insert_no_seams(
        t_rewired: Vec<(Transpositions, Permutation)>,
        c: CircuitSeq,
        n: usize,
        env: &lmdb::Environment,
        dbs: &HashMap<String, Database>,
        first_bounds: (usize, usize),
        second_bounds: (usize, usize), 
    ) -> CircuitSeq {
        let mut gates: Vec<[u16; 3]> = Vec::new();
        for i in 0..c.gates.len()-1 {
            // gates.push(c.gates[i]);
            let i = 3*i;
            let first = &t_rewired[i].0;
            let mut first_gates: Vec<[u16;3]> = Vec::new();
            for i in 0..first.transpositions.len() {
                let mut swap = first.transpositions[i];
                if swap.0 == swap.1 {
                    continue;
                }
                let n = first_bounds.1 - first_bounds.0 + 1;
                let offset = first_bounds.0 as u16;
                swap.0 -= offset;
                swap.1 -= offset;
                let first_circuit = Self::gen_gates_swap(n, swap, env, dbs);
                let first_circuit: Vec<[u16; 3]> = first_circuit
                    .into_iter()
                    .map(|[a, b, c]| [a + offset, b + offset, c + offset])
                    .collect();

                first_gates.extend_from_slice(&first_circuit);
            }
            rewire_gate_ver(&mut first_gates, &t_rewired[i].1, n);
            gates.extend_from_slice(&first_gates);

            let middle = &t_rewired[i+1].0;
            let mut middle_gates: Vec<[u16;3]> = Vec::new();
            for i in (0..middle.transpositions.len()).rev() {
                let swap = middle.transpositions[i];
                if swap.0 == swap.1 {
                    continue;
                }

                let mut middle_circuit = Self::gen_gates_swap(n, swap, env, dbs);
                middle_circuit.reverse();
                middle_gates.extend_from_slice(&middle_circuit);
            }
            rewire_gate_ver(&mut middle_gates, &t_rewired[i+1].1, n);
            gates.extend_from_slice(&middle_gates);

            let second = &t_rewired[i+2].0;
            let mut second_gates: Vec<[u16;3]> = Vec::new();
            for i in 0..second.transpositions.len() {
                let mut swap = second.transpositions[i];
                if swap.0 == swap.1 {
                    continue;
                }
                let n = second_bounds.1 - second_bounds.0 + 1;
                let offset = second_bounds.0 as u16;
                swap.0 -= offset;
                swap.1 -= offset;
                let second_circuit = Self::gen_gates_swap(n, swap, env, dbs);
                let second_circuit: Vec<[u16; 3]> = second_circuit
                    .into_iter()
                    .map(|[a, b, c]| [a + offset, b + offset, c + offset])
                    .collect();

                second_gates.extend_from_slice(&second_circuit);
            }
            rewire_gate_ver(&mut second_gates, &t_rewired[i+2].1, n);
            gates.extend_from_slice(&second_gates);
        }
        // gates.push(c.gates[c.gates.len()-1]);
        CircuitSeq { gates }
    }

    pub fn filter_repeats(&mut self) {
        let mut i = 0;
        while i < self.transpositions.len().saturating_sub(1) {
            if self.transpositions[i] == self.transpositions[i + 1] {
                self.transpositions.drain(i..=i + 1);
                i = i.saturating_sub(2);
            } else {
                i += 1;
            }
        }
    }

    pub fn evaluate(&self, input: u16) -> u16 {
        let mut val = input;
        for (a, b, _) in self.transpositions.clone() {
            if val == a {
                val = b;
            } else if val == b {
                val = a;
            }
        }

        val
    }

    pub fn concat(&self, other: &Transpositions) -> Transpositions {
        let mut new = self.clone();
        new.transpositions.extend_from_slice(&other.transpositions);
        new
    }
}

pub fn insert_wire_shuffles_knuth(
    circuit: &mut CircuitSeq, 
    n: usize,
    env: &Environment,
    dbs: &HashMap<String, Database>,
) {
    println!("Inserting wire shuffles (knuth)");
    println!("Starting len: {} gates", circuit.gates.len());
    let mut t_list: Transpositions = Transpositions { transpositions: Vec::new() };
    let mut gates: Vec<[u16;3]> = Vec::new();
    let mut negation_mask = vec![0u8; n];

    for &gate in &circuit.gates {
        let t = Transpositions::gen_random_knuth(n, 150, &mut negation_mask);
        gates.extend_from_slice(&t.to_circuit(n, env, dbs).gates);
        t_list.transpositions.extend_from_slice(&t.transpositions);
        let a = t_list.evaluate(gate[0]);
        let b = t_list.evaluate(gate[1]);
        let c = t_list.evaluate(gate[2]);
        let gate = [a, b, c];
        if negation_mask[b as usize] == 1 {
            gates.extend_from_slice(&Transpositions::gen_gates_not(n, b, env, dbs));
            negation_mask[b as usize] = 0;
        }
        if negation_mask[c as usize] == 1 {
            gates.extend_from_slice(&Transpositions::gen_gates_not(n, c, env, dbs));
            negation_mask[c as usize] = 0;
        }
        gates.push(gate);
    }
    let p = t_list.to_perm(n);
    let mut t = Transpositions::from_perm(&p);
    let mut wire_transpositions: HashMap<u16, (usize, usize)> = HashMap::new();

    for (i, (a, b, _)) in t.transpositions.iter().enumerate() {
        wire_transpositions.insert(*a, (i, 0));
        wire_transpositions.insert(*b, (i, 1));
    }

    const TRANSITION: [[u8; 4]; 2] = [
        // pos = 0
        [1, 0, 3, 2],
        // pos = 1
        [2, 3, 0, 1],
    ];

    for (i, val) in negation_mask.into_iter().enumerate() {
        if val == 1 {
            if let Some(swaps) = wire_transpositions.get(&(i as u16)) {
                let &(swap_idx, pos) = swaps;
                let curr_neg_type = t.transpositions[swap_idx].2;
                if pos > 1 || curr_neg_type > 3 {
                    panic!("Invalid pos or curr_neg_type");
                }
                t.transpositions[swap_idx].2 = TRANSITION[pos][curr_neg_type as usize] as u16;
                
            }
        }
    }

    let mut c = t.to_circuit(n, env, dbs).gates;
    c.reverse();
    gates.extend_from_slice(&c);
    circuit.gates = gates;
    println!("Complete. Ending len: {} gates", circuit.gates.len());
}

pub fn insert_wire_shuffles_simple(
    circuit: &mut CircuitSeq, 
    n: usize,
    env: &Environment,
    dbs: &HashMap<String, Database>,
) {
    println!("Inserting wire shuffles (simple)");
    println!("Starting len: {} gates", circuit.gates.len());
    let mut t_list: Transpositions = Transpositions { transpositions: Vec::new() };
    let mut gates: Vec<[u16;3]> = Vec::new();
    let mut negation_mask = vec![0u8; n];

    // Generate random points. m needed in k = m * n
    // Choose them spaced approximately evenly but with `sufficient` randomness
    let m = circuit.gates.len();
    let mut points = Vec::with_capacity(m);
    let mut rng = rand::rng();
    for i in 0..m {
        let center = i * n + n / 2;

        // allow significant variance but keep spacing structure
        let jitter = rng.random_range(-(n as i64)/2 ..= (n as i64)/2);

        let mut p = center as i64 + jitter;

        // avoid very beginning
        if p < n as i64 / 4 {
            p = n as i64 / 4;
        }

        points.push(p as usize);
    }
    let mut last = 0;
    for (i, gate) in circuit.gates.iter().enumerate() {
        let t = Transpositions::gen_random_simple(n, points[i] - last, &mut negation_mask);
        last = points[i];
        gates.extend_from_slice(&t.to_circuit(n, env, dbs).gates);
        t_list.transpositions.extend_from_slice(&t.transpositions);
        let a = t_list.evaluate(gate[0]);
        let b = t_list.evaluate(gate[1]);
        let c = t_list.evaluate(gate[2]);
        let gate = [a, b, c];
        if negation_mask[b as usize] == 1 {
            gates.extend_from_slice(&Transpositions::gen_gates_not(n, b, env, dbs));
            negation_mask[b as usize] = 0;
        }
        if negation_mask[c as usize] == 1 {
            gates.extend_from_slice(&Transpositions::gen_gates_not(n, c, env, dbs));
            negation_mask[c as usize] = 0;
        }
        gates.push(gate);
    }
    let p = t_list.to_perm(n);
    let mut t = Transpositions::from_perm(&p);
    let mut wire_transpositions: HashMap<u16, (usize, usize)> = HashMap::new();

    for (i, (a, b, _)) in t.transpositions.iter().enumerate() {
        wire_transpositions.insert(*a, (i, 0));
        wire_transpositions.insert(*b, (i, 1));
    }

    const TRANSITION: [[u8; 4]; 2] = [
        // pos = 0
        [1, 0, 3, 2],
        // pos = 1
        [2, 3, 0, 1],
    ];

    for (i, val) in negation_mask.into_iter().enumerate() {
        if val == 1 {
            if let Some(swaps) = wire_transpositions.get(&(i as u16)) {
                let &(swap_idx, pos) = swaps;
                let curr_neg_type = t.transpositions[swap_idx].2;
                if pos > 1 || curr_neg_type > 3 {
                    panic!("Invalid pos or curr_neg_type");
                }
                t.transpositions[swap_idx].2 = TRANSITION[pos][curr_neg_type as usize] as u16;
                
            }
        }
    }

    let mut c = t.to_circuit(n, env, dbs).gates;
    c.reverse();
    gates.extend_from_slice(&c);
    circuit.gates = gates;
    println!("Complete. Ending len: {} gates", circuit.gates.len());
}

// Insert 2 shuffles are the beginning and end, and then an additional x number of shuffles
pub fn insert_wire_shuffles_x(
    circuit: &mut CircuitSeq, 
    n: usize,
    env: &Environment,
    dbs: &HashMap<String, Database>,
    x: usize,
) {
    println!("Inserting wire shuffles");
    println!("Starting len: {} gates", circuit.gates.len());
    let mut t_list: Transpositions = Transpositions { transpositions: Vec::new() };
    let mut gates: Vec<[u16;3]> = Vec::new();
    let mut negation_mask = vec![0u8; n];

    let start = 1;
    let end = circuit.gates.len() - 1;
    let range_size = end - start + 1;
    let mut rng = rand::rng();
    let sample = rand::seq::index::sample(&mut rng, range_size, x);

    let mut nums: Vec<usize> = sample.iter().map(|i| start + i).collect();

    nums.push(0);

    for (i, gate) in circuit.gates.iter().enumerate() {
        if nums.contains(&i) {
            let t = Transpositions::gen_random_knuth(n, 150, &mut negation_mask);
            gates.extend_from_slice(&t.to_circuit(n, env, dbs).gates);
            t_list.transpositions.extend_from_slice(&t.transpositions);
        }   
        let a = t_list.evaluate(gate[0]);
        let b = t_list.evaluate(gate[1]);
        let c = t_list.evaluate(gate[2]);
        let gate = [a, b, c];
        if negation_mask[b as usize] == 1 {
            gates.extend_from_slice(&Transpositions::gen_gates_not(n, b, env, dbs));
            negation_mask[b as usize] = 0;
        }
        if negation_mask[c as usize] == 1 {
            gates.extend_from_slice(&Transpositions::gen_gates_not(n, c, env, dbs));
            negation_mask[c as usize] = 0;
        }
        gates.push(gate);
    }
    let p = t_list.to_perm(n);
    let mut t = Transpositions::from_perm(&p);
    let mut wire_transpositions: HashMap<u16, (usize, usize)> = HashMap::new();

    for (i, (a, b, _)) in t.transpositions.iter().enumerate() {
        wire_transpositions.insert(*a, (i, 0));
        wire_transpositions.insert(*b, (i, 1));
    }

    const TRANSITION: [[u8; 4]; 2] = [
        // pos = 0
        [1, 0, 3, 2],
        // pos = 1
        [2, 3, 0, 1],
    ];

    for (i, val) in negation_mask.into_iter().enumerate() {
        if val == 1 {
            if let Some(swaps) = wire_transpositions.get(&(i as u16)) {
                let &(swap_idx, pos) = swaps;
                let curr_neg_type = t.transpositions[swap_idx].2;
                if pos > 1 || curr_neg_type > 3 {
                    panic!("Invalid pos or curr_neg_type");
                }
                t.transpositions[swap_idx].2 = TRANSITION[pos][curr_neg_type as usize] as u16;
                
            }
        }
    }

    let mut c = t.to_circuit(n, env, dbs).gates;
    c.reverse();
    gates.extend_from_slice(&c);
    circuit.gates = gates;
    println!("Complete. Ending len: {} gates", circuit.gates.len());
}

// Insert m samf between each gate
pub fn insert_wire_m_samfs_every_x(
    circuit: &mut CircuitSeq,
    n: usize,
    m: usize,
    x: usize,
    env: &Environment,
    dbs: &HashMap<String, Database>,
) {
    let n = n;
    println!("Inserting {} samfs between each gate", m);
    println!("Starting len: {} gates", circuit.gates.len());
    let mut t_list: Transpositions = Transpositions { transpositions: Vec::new() };
    let mut gates: Vec<[u16;3]> = Vec::new();
    let mut negation_mask = vec![0u8; n];

    for (i, gate) in circuit.gates.iter().enumerate() {
        if i % x == 0 {
            let t = Transpositions::gen_random_simple(n, m, &mut negation_mask);
            gates.extend_from_slice(&t.to_circuit(n, env, dbs).gates);
            t_list.transpositions.extend_from_slice(&t.transpositions);
        }
        let a = t_list.evaluate(gate[0]);
        let b = t_list.evaluate(gate[1]);
        let c = t_list.evaluate(gate[2]);
        let gate = [a, b, c];
        if negation_mask[b as usize] == 1 {
            gates.extend_from_slice(&Transpositions::gen_gates_not(n, b, env, dbs));
            negation_mask[b as usize] = 0;
        }
        if negation_mask[c as usize] == 1 {
            gates.extend_from_slice(&Transpositions::gen_gates_not(n, c, env, dbs));
            negation_mask[c as usize] = 0;
        }
        gates.push(gate);
    }
    let p = t_list.to_perm(n);
    let mut t = Transpositions::from_perm(&p);
    let mut wire_transpositions: HashMap<u16, (usize, usize)> = HashMap::new();

    for (i, (a, b, _)) in t.transpositions.iter().enumerate() {
        wire_transpositions.insert(*a, (i, 0));
        wire_transpositions.insert(*b, (i, 1));
    }

    const TRANSITION: [[u8; 4]; 2] = [
        // pos = 0
        [1, 0, 3, 2],
        // pos = 1
        [2, 3, 0, 1],
    ];

    for (i, val) in negation_mask.into_iter().enumerate() {
        if val == 1 {
            if let Some(swaps) = wire_transpositions.get(&(i as u16)) {
                let &(swap_idx, pos) = swaps;
                let curr_neg_type = t.transpositions[swap_idx].2;
                if pos > 1 || curr_neg_type > 3 {
                    panic!("Invalid pos or curr_neg_type");
                }
                t.transpositions[swap_idx].2 = TRANSITION[pos][curr_neg_type as usize] as u16;
                
            }
        }
    }

    let mut c = t.to_circuit(n, env, dbs).gates;
    c.reverse();
    gates.extend_from_slice(&c);
    circuit.gates = gates;
    println!("Complete. Ending len: {} gates", circuit.gates.len());
}

// Generate a circuit R R*, but then insert a series of CNOTS in between so that the first n wires are reversible, but the last n wires can be used to compute R
pub fn generate_reversible(
    c: &CircuitSeq,
    n: usize,
    env: &Environment,
    dbs: &HashMap<String, Database>, 
) -> CircuitSeq {
    let mut rev = c.clone();
    rev.gates.reverse();
    let mut gates = Vec::new();
    gates.extend_from_slice(&c.gates.clone());
    for i in 0..n {
        gates.extend_from_slice(&Transpositions::gen_gates_cnot(2 * n, i as u16, (i + n) as u16, env, dbs));
    }
    gates.extend_from_slice(&rev.gates);
    CircuitSeq { gates }
}

pub fn replace_disjoint_pair((a,b, t1): (u16, u16, u16), (c,d, t2): (u16, u16, u16)) -> Vec<(u16, u16, u16)> {
    let possibilities = [
        vec![(a,c,0),(b,d,0),(a,d,0),(b,c,0)],
        vec![(b,c,0),(a,d,0),(b,d,0),(a,c,0)],
    ];

    let mut rng = rand::rng();
    let idx = rng.random_range(0..possibilities.len());
    
    let mut t = possibilities[idx].clone();

    let mut wire_transpositions: HashMap<u16, (usize, usize)> = HashMap::new();

    for (i, (a, b, _)) in t.iter().enumerate() {
        wire_transpositions.insert(*a, (i, 0));
        wire_transpositions.insert(*b, (i, 1));
    }

    const TRANSITION: [[u8; 4]; 2] = [
        // pos = 0
        [1, 0, 3, 2],
        // pos = 1
        [2, 3, 0, 1],
    ];

    let mut negation_mask: Vec<u16> = Vec::new();
    if t1 == 1 || t1 == 3 {
        negation_mask.push(a);
    }
    if t1 == 2 || t1 == 3 {
        negation_mask.push(b);
    }
    if t2 == 1 || t2 == 3 {
        negation_mask.push(c);
    }
    if t2 == 2 || t2 == 3 {
        negation_mask.push(d);
    }
    for val in negation_mask {
        if let Some(swaps) = wire_transpositions.get(&val) {
            let &(swap_idx, pos) = swaps;
            let curr_neg_type = t[swap_idx].2;
            if pos > 1 || curr_neg_type > 3 {
                panic!("Invalid pos or curr_neg_type");
            }
            t[swap_idx].2 = TRANSITION[pos][curr_neg_type as usize] as u16;
            
        }
    }
    
    t
}

fn gates_collide(g1: [u16; 3], g2: [u16; 3]) -> bool {
    g1[0] == g2[1] || g1[0] == g2[2] || g2[0] == g1[1] || g2[0] == g1[2]
}

// For each collision (adjacent gates that can't commute), try inserting the first 3 gates of a
// randomly chosen SAMF (swap-and-maybe-flip circuit) after the collision window and look up an
// equal-or-shorter replacement in the curated DB.
//
// When gates_ahead > 2, first tries a wider window of gates_ahead gates + samf[0..3]. Falls back
// to the 2-gate collision pair + samf[0..3] if the wider lookup misses.
//
// If a replacement is found, output it followed by the remaining SAMF gates. Future gates are
// relabeled by the SAMF's wire permutation, and the accumulated permutation is undone at the end.
pub fn shuffled_shooting_game(
    circuit: &mut CircuitSeq,
    n: usize,
    env: &Environment,
    dbs: &HashMap<String, Database>,
    curated_shard_dbs: &[Database],
    shard_dbs: &[Database],
    gates_ahead: usize,
) -> usize {
    use crate::replace::pairs::compress_curated_lmdb;

    let mut rng = rand::rng();
    let mut output: Vec<[u16; 3]> = Vec::new();
    let mut t_list = Transpositions { transpositions: Vec::new() };
    let mut negation_mask = vec![0u8; n];
    let mut compressions: usize = 0;

    let input = circuit.gates.clone();
    let mut i = 0;

    while i < input.len() {
        let gate = input[i];
        let a = t_list.evaluate(gate[0]);
        let b = t_list.evaluate(gate[1]);
        let c = t_list.evaluate(gate[2]);

        // Attempt SAMF-assisted replacement when this gate and the next collide.
        let replaced = 'try_replace: {
            if i + 1 >= input.len() { break 'try_replace false; }

            let next = input[i + 1];
            let na = t_list.evaluate(next[0]);
            let nb = t_list.evaluate(next[1]);
            let nc = t_list.evaluate(next[2]);

            if !gates_collide([a, b, c], [na, nb, nc]) { break 'try_replace false; }

            // Collision pair controls must be clean (no pending negation corrections).
            if negation_mask[b as usize] != 0 || negation_mask[c as usize] != 0
                || negation_mask[nb as usize] != 0 || negation_mask[nc as usize] != 0
            {
                break 'try_replace false;
            }

            // Generate SAMF.
            let swap_lo: u16 = rng.random_range(0..n as u16);
            let swap_hi: u16 = loop {
                let w: u16 = rng.random_range(0..n as u16);
                if w != swap_lo { break w; }
            };
            let (swap_lo, swap_hi) = if swap_lo < swap_hi { (swap_lo, swap_hi) } else { (swap_hi, swap_lo) };
            let neg_type: u16 = rng.random_range(0..4);
            let samf_swap = (swap_lo, swap_hi, neg_type);
            let samf = Transpositions::gen_gates_swap(n, samf_swap, env, dbs);
            if samf.len() < 3 { break 'try_replace false; }

            // Build all available context gates (up to gates_ahead) with clean flags.
            // Positions 0 and 1 (collision pair controls) are already verified clean.
            let ga = gates_ahead.min(input.len() - i);
            let mut ctx: Vec<([u16; 3], bool)> = Vec::with_capacity(ga);
            ctx.push(([a, b, c], true));
            if ga >= 2 { ctx.push(([na, nb, nc], true)); }
            for k in 2..ga {
                let g = input[i + k];
                let gw0 = t_list.evaluate(g[0]);
                let gw1 = t_list.evaluate(g[1]);
                let gw2 = t_list.evaluate(g[2]);
                let clean = negation_mask[gw1 as usize] == 0 && negation_mask[gw2 as usize] == 0;
                ctx.push(([gw0, gw1, gw2], clean));
            }

            // Try all contiguous sub-windows of [context(0..ga), samf(0..3)] with:
            //   - length ≥ 4
            //   - at least 1 SAMF gate  (end > ga)
            //   - all context gates in the window have clean controls
            // Order: longest first, within same length more-SAMF-first (start descending).
            let mut found: Option<(usize, usize, Vec<[u16; 3]>)> = None; // (start, samf_used, repl)
            'outer: for len in (4..=ga + 3).rev() {
                for start in (0..ga).rev() {
                    let end = start + len;
                    if end > ga + 3 { continue; } // beyond full window
                    if end <= ga   { continue; }   // no SAMF gate
                    // All context gates in [start..end.min(ga)] must be clean.
                    if !(start..end.min(ga)).all(|k| ctx[k].1) { continue; }
                    // Build sub-window: context[start..ga] ++ samf[0..end-ga]
                    let samf_count = end - ga;
                    let mut window: Vec<[u16; 3]> = (start..ga).map(|k| ctx[k].0).collect();
                    window.extend_from_slice(&samf[..samf_count]);
                    if let Some(repl) = compress_curated_lmdb(&window, n, env, curated_shard_dbs, shard_dbs) {
                        found = Some((start, samf_count, repl));
                        break 'outer;
                    }
                }
            }

            match found {
                None => false,
                Some((start, samf_used, repl)) => {
                    // Output any context gates before the window start normally.
                    // These are clean (verified above), so no NOT corrections needed.
                    for k in 0..start {
                        output.push(ctx[k].0);
                    }
                    output.extend_from_slice(&repl);
                    output.extend_from_slice(&samf[samf_used..]);
                    compressions += 1;

                    t_list.transpositions.push(samf_swap);
                    let tmp = negation_mask[swap_lo as usize];
                    negation_mask[swap_lo as usize] = negation_mask[swap_hi as usize];
                    negation_mask[swap_hi as usize] = tmp;
                    if neg_type == 1 || neg_type == 3 { negation_mask[swap_lo as usize] ^= 1; }
                    if neg_type == 2 || neg_type == 3 { negation_mask[swap_hi as usize] ^= 1; }

                    i += ga; // advance past all context gates
                    true
                }
            }
        };

        if !replaced {
            // Normal path: flush any pending negations on control wires, then emit the gate.
            if negation_mask[b as usize] == 1 {
                output.extend_from_slice(&Transpositions::gen_gates_not(n, b, env, dbs));
                negation_mask[b as usize] = 0;
            }
            if negation_mask[c as usize] == 1 {
                output.extend_from_slice(&Transpositions::gen_gates_not(n, c, env, dbs));
                negation_mask[c as usize] = 0;
            }
            output.push([a, b, c]);
            i += 1;
        }
    }

    // Undo accumulated wire permutation (and absorbed negations) — same pattern as
    // insert_wire_shuffles_knuth.
    let p = t_list.to_perm(n);
    let mut t = Transpositions::from_perm(&p);
    let mut wire_positions: HashMap<u16, (usize, usize)> = HashMap::new();
    for (idx, (wa, wb, _)) in t.transpositions.iter().enumerate() {
        wire_positions.insert(*wa, (idx, 0));
        wire_positions.insert(*wb, (idx, 1));
    }
    const TRANSITION: [[u8; 4]; 2] = [[1, 0, 3, 2], [2, 3, 0, 1]];
    for (wire, &val) in negation_mask.iter().enumerate() {
        if val == 1 {
            if let Some(&(swap_idx, pos)) = wire_positions.get(&(wire as u16)) {
                let curr = t.transpositions[swap_idx].2;
                t.transpositions[swap_idx].2 = TRANSITION[pos][curr as usize] as u16;
            }
        }
    }
    let mut undo = t.to_circuit(n, env, dbs).gates;
    undo.reverse();
    output.extend_from_slice(&undo);

    println!("shuffled_shooting_game: {} compressions made ({} -> {} gates)",
        compressions, input.len(), output.len());

    circuit.gates = output;
    compressions
}

#[cfg(test)]
mod tests {
    use lmdb::Environment;
    use std::{
        fs::File,
        io::{BufRead, BufReader},
        path::Path,
    };
    use rand::prelude::IndexedRandom;
    use crate::{CircuitSeq, replace::identities::insert_ri_identities};
    use crate::replace::transpositions::Transpositions;
    #[test]
    fn test_wire_shifting() {
        use crate::replace::main_mix::open_all_dbs;
        let file = File::open("initial.txt").expect("failed to open initial.txt");
        let reader = BufReader::new(file);

        let circuits: Vec<String> = reader
            .lines()
            .map(|l| l.unwrap())
            .filter(|l| !l.trim().is_empty())
            .collect();

        let mut rng = rand::rng();
        let circuit_str = circuits
            .choose(&mut rng)
            .expect("no circuits found");

        let base = CircuitSeq::from_string(circuit_str);

        let env = Environment::new()
            .set_max_dbs(262)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new("./db"))
            .expect("failed to open lmdb");

        let (shard_dbs, curated_shard_dbs) = open_all_dbs(&env); let dbs = std::collections::HashMap::<String, lmdb::Database>::new();

        let mut gates: Vec<[u16; 3]> = Vec::new();
        let mut last = Transpositions { transpositions: Vec::new() };
        for &gate in &base.gates {

            let t = Transpositions::gen_random_knuth(64, 100, &mut Vec::new());
            // println!("t: {}", t.transpositions.len());
            if last.transpositions.is_empty() {
                gates.extend(t.to_circuit(64, &env, &dbs).gates);
            } else {
                let mut combined = last.concat(&t);
                combined.canonicalize();
                combined.filter_repeats();
                Transpositions::shoot_random_transpositions(&mut combined, 100_000);
                gates.extend(combined.to_circuit(64, &env, &dbs).gates);
            }
            let a = t.evaluate(gate[0]);
            let b = t.evaluate(gate[1]);
            let c = t.evaluate(gate[2]);
            gates.push([a, b, c]);
            last = t;
            last.transpositions.reverse();
        }
        gates.extend(last.to_circuit(64, &env, &dbs).gates);
        let new_circuit = CircuitSeq { gates };
        if base.probably_equal(&new_circuit, 64, 1_000).is_err() {
            panic!("Failed to retain functionality");
        }
        std::fs::write("test.txt", new_circuit.repr())
            .expect("failed to write test.txt");
    }

    #[test]
    fn test_transpose_shooting() {
        use crate::replace::main_mix::open_all_dbs;
        // let file = File::open("initial.txt").expect("failed to open initial.txt");
        // let reader = BufReader::new(file);

        // let circuits: Vec<String> = reader
        //     .lines()
        //     .map(|l| l.unwrap())
        //     .filter(|l| !l.trim().is_empty())
        //     .collect();

        // let mut rng = rand::rng();
        // let _circuit_str = circuits
        //     .choose(&mut rng)
        //     .expect("no circuits found");

        // let base = CircuitSeq::from_string(circuit_str);

        let env = Environment::new()
            .set_max_dbs(262)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new("./db"))
            .expect("failed to open lmdb");

        let (shard_dbs, curated_shard_dbs) = open_all_dbs(&env); let dbs = std::collections::HashMap::<String, lmdb::Database>::new();

        let mut t = Transpositions::gen_random_knuth(128, 500, &mut vec![0u8; 128]);
        let base = t.to_circuit(128, &env, &dbs);
        Transpositions::shoot_random_transpositions(&mut t, 100_000);
        let new_circuit = t.to_circuit(128, &env, &dbs);
        if base.probably_equal(&new_circuit, 128, 1_000).is_err() {
            panic!("Failed to retain functionality after shooting");
        }
        t.canonicalize();
        t.filter_repeats();
        let new_circuit = t.to_circuit(128, &env, &dbs);
        if base.probably_equal(&new_circuit, 128, 1_000).is_err() {
            panic!("Failed to retain functionality after filtering");
        }
        println!("They are equal");
    }

    #[test]
    fn test_insert_shuffles() {
        use crate::replace::main_mix::open_all_dbs;
        use std::io::Write;
        use crate::replace::transpositions::insert_wire_shuffles_x;
        let file = File::open("initial.txt").expect("failed to open initial.txt");
        let mut reader = BufReader::new(file);

        let mut circuit_str = String::new();
        reader
            .read_line(&mut circuit_str)
            .expect("failed to read circuit");

        let circuit_str = circuit_str.trim();
        assert!(!circuit_str.is_empty(), "initial.txt is empty");

        let base = CircuitSeq::from_string(circuit_str);

        let env = Environment::new()
            .set_max_dbs(262)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new("./db"))
            .expect("failed to open lmdb");

        let (shard_dbs, curated_shard_dbs) = open_all_dbs(&env); let dbs = std::collections::HashMap::<String, lmdb::Database>::new();

        let mut new_circuit = base.clone();
        insert_wire_shuffles_x(&mut new_circuit, 64, &env, &dbs, 50);

        if base.probably_equal(&new_circuit, 64, 1_000).is_err() {
            panic!("Failed to retain functionality");
        }

        let mut out = File::create("shuffled.txt")
            .expect("failed to create shuffled.txt");
        writeln!(out, "{}", new_circuit.repr()).unwrap();

        println!("They are equal and written to shuffled.txt");
    }

    #[test]
    fn test_transposition_rev() {
        use crate::replace::main_mix::open_all_dbs;
        use crate::replace::transpositions::HashMap;
        let env = Environment::new()
            .set_max_dbs(262)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new("./db"))
            .expect("failed to open lmdb");

        let (shard_dbs, curated_shard_dbs) = open_all_dbs(&env); let dbs = std::collections::HashMap::<String, lmdb::Database>::new();
        let n = 64;
        let mut gates: Vec<[u16;3]> = Vec::new();
        let mut negation_mask = vec![0u8; n];
        let t = Transpositions::gen_random_knuth(n, 150, &mut negation_mask);
        gates.extend_from_slice(&t.to_circuit(n, &env, &dbs).gates);
        let p = t.to_perm(n);
        let t = Transpositions::from_perm(&p);
        let mut wire_transpositions: HashMap<u16, Vec<(usize, usize)>> = HashMap::new();
        for (i, (a, b, _)) in t.transpositions.clone().into_iter().enumerate() {
            wire_transpositions
            .entry(a)
            .or_default()
            .push((i, 0));

            wire_transpositions
                .entry(b)
                .or_default()
                .push((i, 1));
        }

        for (i, val) in negation_mask.into_iter().enumerate() {
            if val == 1 {
                gates.extend_from_slice(&Transpositions::gen_gates_not(n, i as u16, &env, &dbs));
            }
        }
        let mut tr: Vec<(u16, u16, u16)> = Vec::new();
        for (a,b,_) in t.transpositions{
            tr.push((a,b,0u16));
        }
        let t = Transpositions { transpositions: tr }; 
        let mut c = t.to_circuit(n, &env, &dbs).gates;
        c.reverse();
        gates.extend_from_slice(&c);

        let c = CircuitSeq { gates };
        if c.probably_equal(&CircuitSeq{ gates: Vec::new() }, 64, 1_000).is_err() {
            panic!("Lost functionality");
        }
    }
    #[test]
    fn test_wire_shifting2() {
        use crate::replace::main_mix::open_all_dbs;
        let file = File::open("initial.txt").expect("failed to open initial.txt");
        let reader = BufReader::new(file);

        let circuits: Vec<String> = reader
            .lines()
            .map(|l| l.unwrap())
            .filter(|l| !l.trim().is_empty())
            .collect();

        let mut rng = rand::rng();
        let circuit_str = circuits
            .choose(&mut rng)
            .expect("no circuits found");

        let base = CircuitSeq::from_string(circuit_str);

        let env = Environment::new()
            .set_max_dbs(262)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new("./db"))
            .expect("failed to open lmdb");

        let (shard_dbs, curated_shard_dbs) = open_all_dbs(&env); let dbs = std::collections::HashMap::<String, lmdb::Database>::new();
        let t = Transpositions::gen_random_knuth(64, 100, &mut Vec::new());
        let mut gates: Vec<[u16; 3]> = Vec::new();
        gates.extend(t.to_circuit(64, &env, &dbs).gates);
        for &gate in &base.gates {
            let a = t.evaluate(gate[0]);
            let b = t.evaluate(gate[1]);
            let c = t.evaluate(gate[2]);
            gates.push([a, b, c]);
        }
        let mut tc = t.to_circuit(64, &env, &dbs).gates;
        tc.reverse();
        gates.extend(&tc);
        let new_circuit = CircuitSeq { gates };
        if base.probably_equal(&new_circuit, 64, 1_000).is_err() {
            panic!("Failed to retain functionality");
        }
        std::fs::write("test.txt", new_circuit.repr())
            .expect("failed to write test.txt");
    }

    #[test]
    fn test_reversible() {
        use std::fs;
        use rand::Rng;
        let circuit_str = fs::read_to_string("initial.txt")
            .expect("failed to read initial.txt");
        let circuit = CircuitSeq::from_string(&circuit_str);

        let mut c100 = circuit.clone();
        c100.gates.truncate(100);

        let mut rng = rand::rng();
        let rand64: u128 = rng.random::<u64>() as u128;
        let input: u128 = rand64; 

        let out_full = circuit.evaluate_128(input);
        let out_100 = c100.evaluate_128(input);

        let low_mask: u128 = (1u128 << 64) - 1;
        let high_mask: u128 = !low_mask;

        assert_eq!(
            out_full & low_mask,
            input & low_mask,
            "first 64 bits changed"
        );

        assert_eq!(
            out_full & high_mask,
            (out_100 & low_mask) << 64,
            "last 64 bits differ from c100 result"
        );

        let mut rev = circuit.clone();
        rev.gates.reverse();
        let out_full_rev = rev.evaluate_128(input);
        assert_eq!(
            out_full,
            out_full_rev,
            "the circuit isn't reversible"
        );
    }

    #[test]
    fn test_ri_32() {
        use crate::replace::identities::create_ri_identities_32;
        use crate::replace::main_mix::open_all_dbs;
        use std::io::Write;
        use rand::seq::SliceRandom;
        use crate::replace::transpositions::Permutation;
        let (first, middle, second, _, _) = create_ri_identities_32();
        let env = Environment::new()
            .set_max_dbs(262)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new("./db"))
            .expect("failed to open lmdb");

        let (shard_dbs, curated_shard_dbs) = open_all_dbs(&env); let dbs = std::collections::HashMap::<String, lmdb::Database>::new();

        let mut file = File::create("test_id.txt").expect("Failed to create file");

        let f = first.to_circuit(32, &env, &dbs);
        let mut m = middle.to_circuit(32, &env, &dbs);
        m.gates.reverse();
        let s = second.to_circuit(32, &env, &dbs);

        let mut c = CircuitSeq {gates: Vec::new() };
        c.gates.extend_from_slice(&f.gates);
        c.gates.extend_from_slice(&m.gates);
        c.gates.extend_from_slice(&s.gates);
        let id = CircuitSeq {gates: Vec::new() };
        if c.probably_equal(&id, 32, 1000).is_err() {
            panic!("Not an id");
        }
        let mut id = first.restricted_to_circuit(32, &env, &dbs, &vec![]);
        let stupid_id = CircuitSeq { gates: Vec::new() };
        if id.probably_equal(&stupid_id, 32, 1000).is_err() {
            panic!("Not an id identity");
        }
        // let repr = id.repr();
        let mut shuffle: Permutation = Permutation { data: (0..32).collect() };
        shuffle.data.shuffle(&mut rand::rng());
        id.rewire(&shuffle, 32);
        if id.probably_equal(&stupid_id, 32, 1000).is_err() {
            panic!("Shuffling destroyed identity");
        }
        let repr = id.repr();
        writeln!(file, "{}", repr)
            .expect("Failed to write to file");

        println!("Wrote test circuit to file");
    }

    #[test]
    fn test_insert_ri_identities() {
        use crate::replace::main_mix::open_all_dbs;
        use std::io::Write;
        let env = Environment::new()
            .set_max_dbs(262)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new("./db"))
            .expect("failed to open lmdb");
        let (shard_dbs, curated_shard_dbs) = open_all_dbs(&env); let dbs = std::collections::HashMap::<String, lmdb::Database>::new();

        let mut file = File::create("test_id.txt").expect("Failed to create file");

        let c_old = CircuitSeq::from_string("vnt;otv;k3c;g8d;hkm;fn8;3p0;v92;0id;l4a;pq0;sn3;06k;roh;cld;pef;s3j;dh7;jum;l41;gio;1pf;rge;ont;3qa;731;3rg;2eg;2sl;ebg;ovf;opk;tel;hts;cql;06h;u9i;gov;lbc;04i;0as;kp9;iro;e38;bc8;0ue;hst;p9i;gom;908;0do;l5s;t9g;abd;7rs;0hk;fq9;o49;14l;7j0;vu6;clf;4mn;9g6;4vc;lkp;p73;4mi;h9k;7rg;d4a;674;73f;ojr;fpj;gct;94k;nab;3is;q2h;dvp;huv;bsp;lb7;vr2;nd7;ud3;9bv;ljg;q1e;av9;8du;3hl;cd1;mir;ris;uoc;btq;ibc;bds;");
        let mut c = CircuitSeq::from_string("vnt;otv;k3c;g8d;hkm;fn8;3p0;v92;0id;l4a;pq0;sn3;06k;roh;cld;pef;s3j;dh7;jum;l41;gio;1pf;rge;ont;3qa;731;3rg;2eg;2sl;ebg;ovf;opk;tel;hts;cql;06h;u9i;gov;lbc;04i;0as;kp9;iro;e38;bc8;0ue;hst;p9i;gom;908;0do;l5s;t9g;abd;7rs;0hk;fq9;o49;14l;7j0;vu6;clf;4mn;9g6;4vc;lkp;p73;4mi;h9k;7rg;d4a;674;73f;ojr;fpj;gct;94k;nab;3is;q2h;dvp;huv;bsp;lb7;vr2;nd7;ud3;9bv;ljg;q1e;av9;8du;3hl;cd1;mir;ris;uoc;btq;ibc;bds;");
        // let c_old = CircuitSeq::from_string("k3c;k3c;k3c");
        // let mut c = CircuitSeq::from_string("k3c;k3c;k3c");
        insert_ri_identities(&mut c, &env, &dbs);

        writeln!(file, "{}", c.repr())
            .expect("Failed to write to file");
        let _id = CircuitSeq { gates: Vec::new() };
        if c.probably_equal(&c_old, 32, 1000).is_err() {
            panic!("Changed functionality somewhere");
        }
    }

    #[test]
    fn test_transpose_reverse_id() {
        use crate::replace::main_mix::open_all_dbs;
        let env = Environment::new()
            .set_max_dbs(262)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new("./db"))
            .expect("failed to open lmdb");
        let (shard_dbs, curated_shard_dbs) = open_all_dbs(&env); let dbs = std::collections::HashMap::<String, lmdb::Database>::new();

        let mut negation_mask: Vec<u8> = vec![0u8;3];
        let t = Transpositions::gen_random_simple(32, 50, &mut negation_mask);
        let mut t2 = t.clone();
        t2.transpositions.reverse();
        let c = t.to_circuit(32, &env, &dbs).concat(&t2.to_circuit(32, &env, &dbs));
        let id = CircuitSeq { gates: Vec::new() };
        if c.probably_equal(&id, 32, 1000).is_err() {
            panic!("Stupid identities via tranpose");
        }
    }

    #[test]
    fn test_shuffled_shooting_game() {
        use crate::replace::main_mix::open_all_dbs;
        use crate::replace::transpositions::shuffled_shooting_game;
        use std::path::Path;

        let env = Environment::new()
            .set_max_dbs(600)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new("./db"))
            .expect("failed to open lmdb");

        let (shard_dbs, curated_shard_dbs) = open_all_dbs(&env);
        let dbs = std::collections::HashMap::<String, lmdb::Database>::new();
        let n = 32;

        use crate::random::random_data::random_circuit;
        let base = random_circuit(n, 100);

        // Keep trying until at least one SAMF compression actually fires,
        // which confirms the SAMF path is exercised and functionality is maintained.
        let mut total_compressions = 0;
        for attempt in 0..200 {
            let mut circuit = base.clone();
            let compressions = shuffled_shooting_game(
                &mut circuit, n, &env, &dbs, &curated_shard_dbs, &shard_dbs, 5,
            );
            total_compressions += compressions;
            if base.probably_equal(&circuit, n, 500).is_err() {
                panic!("attempt {}: functionality broken on all {} wires", attempt, n);
            }
            if total_compressions > 0 {
                println!("SAMF compression confirmed after {} attempts ({} total compressions)",
                    attempt + 1, total_compressions);
                return;
            }
        }
        panic!("no SAMF compressions fired in 200 attempts — curated DB may be missing entries");
    }

    #[test]
    fn test_gadgetize_maintains_original_wires() {
        use crate::replace::gadgets::gadgetize;
        use crate::circuit::circuit::Gate;
        use rand::Rng;

        use crate::random::random_data::random_circuit;
        let n = 32;
        let base = random_circuit(n, 100);

        let mut rng = rand::rng();
        let gadgetized = gadgetize(&base, n, 1, &mut rng);

        // For each random 2n-bit input, the low-n bits of the gadgetized output
        // must match what the original circuit produces on the low-n bits of that input.
        let n_mask: u128 = (1u128 << n) - 1;
        let two_n_mask: u128 = (1u128 << (2 * n)) - 1;

        let mut failures = 0;
        for _ in 0..1000 {
            // Random input with both original and aux wires set.
            let full_input = rand::rng().random_range(0u128..=u128::MAX) & two_n_mask;

            let gadget_out = Gate::evaluate_index_list_128(full_input, &gadgetized.gates);
            let orig_out   = Gate::evaluate_index_list_128(full_input & n_mask, &base.gates);

            if gadget_out & n_mask != orig_out & n_mask {
                failures += 1;
            }
        }
        assert_eq!(failures, 0,
            "gadgetize broke functionality on original {} wires ({} / 1000 inputs failed)",
            n, failures);
    }
}