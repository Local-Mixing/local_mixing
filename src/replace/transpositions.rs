// For adding wire shuffles and bit flips
use crate::circuit::{Permutation, circuit::CircuitSeq};
use lmdb::{Database, Environment};
use rand::Rng;
use rand::seq::IndexedRandom;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

pub static SAMF_INSERTIONS_MADE: AtomicUsize = AtomicUsize::new(0);
pub static SAMF_COMPRESSIONS_MADE: AtomicUsize = AtomicUsize::new(0);
pub static SAMF_COMPRESSIONS_FAILED: AtomicUsize = AtomicUsize::new(0);
// Curated expansions performed at collisions in the shuffled shooting game.
pub static CURATED_REPLACEMENTS_MADE: AtomicUsize = AtomicUsize::new(0);
// Compressions made while integrating undo SAMFs/NOTs at the end of a shuffle.
pub static END_SAMF_COMPRESSIONS_MADE: AtomicUsize = AtomicUsize::new(0);

// Hardcoded circuits: min-2 depths per function, 3-wire and 4-wire.
// Wire convention: wire 1↔2 swapped (swap), wire 1 flipped (not), wire 1 controls wire 2 (cnot).
// Wire 0 (and wire 3 in 4-wire) are ancilla.
static SWAP_3W: &[&[[u16; 3]]] = &[
    &[
        [1, 0, 2],
        [2, 0, 1],
        [2, 1, 0],
        [1, 0, 2],
        [1, 2, 0],
        [2, 1, 0],
    ],
    &[
        [0, 1, 2],
        [2, 0, 1],
        [2, 1, 0],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
        [0, 2, 1],
    ],
    &[
        [1, 2, 0],
        [2, 1, 0],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
        [1, 0, 2],
        [2, 1, 0],
    ],
    &[
        [0, 2, 1],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [0, 1, 2],
    ],
    &[
        [2, 1, 0],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
        [1, 0, 2],
    ],
    &[
        [0, 1, 2],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
        [1, 0, 2],
        [1, 2, 0],
        [0, 2, 1],
    ],
    &[
        [0, 2, 1],
        [2, 0, 1],
        [2, 1, 0],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
        [0, 1, 2],
    ],
    &[
        [2, 0, 1],
        [2, 1, 0],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ],
    &[
        [2, 0, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
        [1, 2, 0],
    ],
    &[
        [1, 0, 2],
        [2, 1, 0],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
        [1, 0, 2],
        [2, 0, 1],
    ],
    &[
        [0, 1, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [0, 2, 1],
    ],
    &[
        [0, 2, 1],
        [2, 0, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
        [1, 2, 0],
        [0, 1, 2],
    ],
    &[
        [2, 1, 0],
        [1, 0, 2],
        [2, 0, 1],
        [2, 1, 0],
        [1, 0, 2],
        [1, 2, 0],
        [2, 1, 0],
        [1, 2, 0],
    ],
    &[
        [2, 1, 0],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [1, 2, 0],
    ],
    &[
        [0, 1, 2],
        [2, 0, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
        [1, 2, 0],
        [0, 2, 1],
    ],
    &[
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
    ],
    &[
        [1, 0, 2],
        [2, 0, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
        [1, 2, 0],
        [2, 0, 1],
    ],
    &[
        [1, 2, 0],
        [2, 0, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
        [1, 2, 0],
        [2, 1, 0],
    ],
    &[
        [2, 0, 1],
        [1, 0, 2],
        [2, 0, 1],
        [2, 1, 0],
        [1, 0, 2],
        [1, 2, 0],
        [2, 1, 0],
        [1, 0, 2],
    ],
    &[
        [2, 0, 1],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [1, 0, 2],
    ],
    &[
        [0, 1, 2],
        [2, 1, 0],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
        [1, 0, 2],
        [0, 2, 1],
    ],
    &[
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
        [1, 0, 2],
        [1, 2, 0],
    ],
];
static SWAP_4W: &[&[[u16; 3]]] = &[
    &[
        [1, 2, 3],
        [1, 3, 2],
        [2, 1, 3],
        [2, 3, 1],
        [1, 2, 3],
        [1, 3, 2],
    ],
    &[
        [1, 2, 3],
        [2, 1, 3],
        [2, 3, 1],
        [1, 2, 3],
        [1, 3, 2],
        [2, 3, 1],
    ],
    &[
        [1, 2, 3],
        [1, 3, 2],
        [2, 1, 3],
        [2, 3, 1],
        [1, 0, 2],
        [1, 0, 3],
        [1, 2, 0],
        [1, 3, 0],
    ],
    &[
        [2, 0, 1],
        [2, 1, 0],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ],
    &[
        [2, 0, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
        [1, 2, 0],
    ],
    &[
        [1, 3, 2],
        [2, 1, 3],
        [2, 3, 1],
        [1, 2, 3],
        [1, 3, 2],
        [2, 1, 3],
    ],
    &[
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
    ],
];
static SWAP_N1_3W: &[&[[u16; 3]]] = &[
    &[
        [1, 2, 0],
        [2, 1, 0],
        [1, 2, 0],
        [2, 0, 1],
        [1, 0, 2],
        [2, 0, 1],
    ],
    &[
        [1, 0, 2],
        [2, 0, 1],
        [1, 0, 2],
        [2, 1, 0],
        [1, 2, 0],
        [2, 1, 0],
    ],
    &[
        [1, 0, 2],
        [1, 0, 2],
        [1, 2, 0],
        [2, 1, 0],
        [1, 2, 0],
        [2, 0, 1],
        [1, 0, 2],
        [2, 0, 1],
    ],
    &[
        [0, 1, 2],
        [1, 0, 2],
        [2, 1, 0],
        [1, 2, 0],
        [0, 2, 1],
        [2, 1, 0],
        [1, 2, 0],
        [2, 0, 1],
    ],
    &[
        [2, 0, 1],
        [1, 2, 0],
        [2, 1, 0],
        [1, 2, 0],
        [2, 0, 1],
        [1, 0, 2],
        [2, 0, 1],
        [1, 0, 2],
    ],
    &[
        [1, 2, 0],
        [0, 1, 2],
        [1, 2, 0],
        [2, 1, 0],
        [1, 2, 0],
        [2, 0, 1],
        [1, 0, 2],
        [0, 1, 2],
    ],
    &[
        [2, 1, 0],
        [1, 2, 0],
        [2, 1, 0],
        [1, 2, 0],
        [2, 0, 1],
        [1, 0, 2],
        [2, 0, 1],
        [1, 2, 0],
    ],
    // Negate-then-swap (reverse of N2) implementations of the N1 permutation — structurally
    // distinct from the swap-then-negate forms above, added for gate-sequence diversity.
    &[
        [1, 0, 2],
        [2, 1, 0],
        [1, 2, 0],
        [0, 1, 2],
        [2, 1, 0],
        [1, 2, 0],
        [2, 0, 1],
        [0, 2, 1],
    ],
    &[
        [1, 0, 2],
        [0, 2, 1],
        [1, 0, 2],
        [2, 0, 1],
        [1, 0, 2],
        [2, 1, 0],
        [1, 2, 0],
        [0, 2, 1],
    ],
    &[
        [0, 2, 1],
        [1, 2, 0],
        [2, 0, 1],
        [1, 0, 2],
        [0, 1, 2],
        [2, 0, 1],
        [1, 0, 2],
        [2, 1, 0],
    ],
    &[
        [1, 2, 0],
        [1, 0, 2],
        [2, 0, 1],
        [1, 0, 2],
        [0, 1, 2],
        [2, 0, 1],
        [1, 0, 2],
        [0, 2, 1],
    ],
    &[
        [1, 2, 0],
        [0, 2, 1],
        [2, 1, 0],
        [1, 2, 0],
        [2, 0, 1],
        [1, 0, 2],
        [0, 2, 1],
        [2, 0, 1],
    ],
];
static SWAP_N1_4W: &[&[[u16; 3]]] = &[
    &[
        [1, 0, 3],
        [1, 2, 3],
        [2, 1, 3],
        [1, 2, 3],
        [2, 3, 1],
        [1, 3, 2],
        [2, 0, 3],
        [2, 3, 1],
    ],
    &[
        [1, 3, 2],
        [2, 0, 1],
        [1, 0, 2],
        [2, 1, 0],
        [1, 2, 0],
        [2, 1, 3],
    ],
];
static SWAP_N2_3W: &[&[[u16; 3]]] = &[
    &[
        [2, 0, 1],
        [1, 2, 0],
        [2, 1, 0],
        [0, 2, 1],
        [1, 2, 0],
        [2, 1, 0],
        [1, 0, 2],
        [0, 1, 2],
    ],
    &[
        [0, 2, 1],
        [2, 0, 1],
        [1, 2, 0],
        [2, 1, 0],
        [0, 1, 2],
        [1, 2, 0],
        [2, 1, 0],
        [1, 0, 2],
    ],
    &[
        [2, 0, 1],
        [1, 0, 2],
        [2, 0, 1],
        [1, 2, 0],
        [2, 1, 0],
        [1, 2, 0],
    ],
    &[
        [0, 1, 2],
        [1, 0, 2],
        [2, 0, 1],
        [1, 2, 0],
        [2, 1, 0],
        [1, 2, 0],
        [0, 1, 2],
        [1, 2, 0],
    ],
    &[
        [0, 2, 1],
        [1, 2, 0],
        [2, 1, 0],
        [1, 0, 2],
        [2, 0, 1],
        [1, 0, 2],
        [0, 2, 1],
        [1, 0, 2],
    ],
    &[
        [2, 1, 0],
        [1, 0, 2],
        [2, 0, 1],
        [0, 1, 2],
        [1, 0, 2],
        [2, 0, 1],
        [1, 2, 0],
        [0, 2, 1],
    ],
    &[
        [0, 2, 1],
        [1, 0, 2],
        [2, 0, 1],
        [0, 1, 2],
        [1, 0, 2],
        [2, 0, 1],
        [1, 0, 2],
        [1, 2, 0],
    ],
    &[
        [2, 0, 1],
        [0, 2, 1],
        [1, 0, 2],
        [2, 0, 1],
        [1, 2, 0],
        [2, 1, 0],
        [0, 2, 1],
        [1, 2, 0],
    ],
    &[
        [2, 1, 0],
        [1, 2, 0],
        [2, 1, 0],
        [1, 0, 2],
        [2, 0, 1],
        [1, 0, 2],
    ],
    // Negate-then-swap (reverse of N1) implementations of the N2 permutation.
    &[
        [2, 0, 1],
        [1, 0, 2],
        [2, 0, 1],
        [1, 2, 0],
        [2, 1, 0],
        [1, 2, 0],
        [1, 0, 2],
        [1, 0, 2],
    ],
    &[
        [1, 0, 2],
        [2, 0, 1],
        [1, 0, 2],
        [2, 0, 1],
        [1, 2, 0],
        [2, 1, 0],
        [1, 2, 0],
        [2, 0, 1],
    ],
    &[
        [1, 2, 0],
        [2, 0, 1],
        [1, 0, 2],
        [2, 0, 1],
        [1, 2, 0],
        [2, 1, 0],
        [1, 2, 0],
        [2, 1, 0],
    ],
];
static SWAP_N2_4W: &[&[[u16; 3]]] = &[
    &[
        [2, 1, 0],
        [1, 2, 3],
        [2, 1, 3],
        [1, 3, 2],
        [2, 3, 1],
        [1, 0, 2],
    ],
    &[
        [2, 1, 3],
        [1, 2, 0],
        [2, 1, 0],
        [1, 0, 2],
        [2, 0, 1],
        [1, 3, 2],
    ],
    // Negate-then-swap (reverse of N1) implementation of the N2 permutation (4-wire).
    &[
        [2, 3, 1],
        [2, 0, 3],
        [1, 3, 2],
        [2, 3, 1],
        [1, 2, 3],
        [2, 1, 3],
        [1, 2, 3],
        [1, 0, 3],
    ],
];
static SWAP_N12_3W: &[&[[u16; 3]]] = &[
    &[
        [2, 0, 1],
        [2, 1, 0],
        [0, 2, 1],
        [1, 0, 2],
        [2, 0, 1],
        [0, 1, 2],
        [1, 0, 2],
        [2, 0, 1],
    ],
    &[
        [0, 1, 2],
        [1, 0, 2],
        [2, 1, 0],
        [1, 2, 0],
        [2, 0, 1],
        [1, 0, 2],
        [2, 1, 0],
        [0, 1, 2],
    ],
    &[
        [0, 1, 2],
        [1, 0, 2],
        [2, 0, 1],
        [1, 2, 0],
        [0, 2, 1],
        [1, 2, 0],
        [2, 0, 1],
        [1, 0, 2],
    ],
    &[
        [1, 0, 2],
        [1, 2, 0],
        [0, 2, 1],
        [2, 1, 0],
        [1, 2, 0],
        [0, 1, 2],
        [2, 1, 0],
        [1, 2, 0],
    ],
    &[
        [2, 1, 0],
        [1, 2, 0],
        [0, 2, 1],
        [2, 1, 0],
        [1, 2, 0],
        [0, 1, 2],
        [2, 0, 1],
        [2, 1, 0],
    ],
    &[
        [1, 2, 0],
        [2, 0, 1],
        [1, 0, 2],
        [2, 1, 0],
        [1, 2, 0],
        [2, 0, 1],
    ],
    &[
        [0, 2, 1],
        [1, 2, 0],
        [2, 1, 0],
        [1, 0, 2],
        [0, 1, 2],
        [1, 0, 2],
        [2, 1, 0],
        [1, 2, 0],
    ],
    &[
        [1, 0, 2],
        [2, 1, 0],
        [1, 2, 0],
        [2, 0, 1],
        [1, 0, 2],
        [2, 1, 0],
    ],
    &[
        [1, 0, 2],
        [2, 0, 1],
        [1, 2, 0],
        [0, 2, 1],
        [1, 2, 0],
        [2, 0, 1],
        [1, 0, 2],
        [0, 1, 2],
    ],
    &[
        [1, 0, 2],
        [0, 2, 1],
        [2, 1, 0],
        [1, 2, 0],
        [2, 0, 1],
        [1, 0, 2],
        [0, 2, 1],
        [2, 1, 0],
    ],
    &[
        [0, 1, 2],
        [2, 1, 0],
        [1, 0, 2],
        [2, 0, 1],
        [1, 2, 0],
        [2, 1, 0],
        [1, 0, 2],
        [0, 1, 2],
    ],
    &[
        [2, 0, 1],
        [1, 2, 0],
        [2, 1, 0],
        [1, 0, 2],
        [2, 0, 1],
        [1, 2, 0],
    ],
    &[
        [1, 2, 0],
        [2, 1, 0],
        [0, 1, 2],
        [1, 2, 0],
        [2, 1, 0],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
    ],
    &[
        [1, 2, 0],
        [2, 1, 0],
        [1, 0, 2],
        [0, 1, 2],
        [1, 0, 2],
        [2, 1, 0],
        [1, 2, 0],
        [0, 2, 1],
    ],
    &[
        [2, 1, 0],
        [1, 0, 2],
        [2, 0, 1],
        [1, 2, 0],
        [2, 1, 0],
        [1, 0, 2],
    ],
    &[
        [0, 1, 2],
        [1, 2, 0],
        [2, 0, 1],
        [1, 0, 2],
        [2, 1, 0],
        [1, 2, 0],
        [2, 0, 1],
        [0, 1, 2],
    ],
];
static SWAP_N12_4W: &[&[[u16; 3]]] = &[
    &[
        [1, 3, 2],
        [2, 1, 3],
        [1, 2, 3],
        [2, 3, 1],
        [1, 3, 2],
        [2, 1, 3],
    ],
    &[
        [2, 1, 3],
        [1, 3, 2],
        [2, 3, 1],
        [1, 2, 3],
        [2, 1, 3],
        [1, 3, 2],
    ],
    &[
        [1, 2, 0],
        [2, 3, 1],
        [1, 3, 2],
        [2, 1, 3],
        [1, 2, 3],
        [2, 0, 1],
    ],
];
// NOT 3-wire at min-2 depths (6,7) is empty — those depths only exist for 4-wire.
static NOT_3W: &[&[[u16; 3]]] = &[];
static NOT_4W: &[&[[u16; 3]]] = &[
    &[
        [0, 2, 3],
        [1, 0, 2],
        [1, 0, 3],
        [0, 2, 3],
        [1, 2, 0],
        [1, 2, 3],
        [1, 3, 0],
    ],
    &[
        [1, 0, 2],
        [1, 0, 3],
        [1, 3, 2],
        [2, 3, 0],
        [1, 0, 2],
        [1, 3, 2],
        [2, 3, 0],
    ],
    &[
        [1, 0, 2],
        [1, 0, 3],
        [3, 0, 2],
        [0, 2, 3],
        [1, 3, 0],
        [0, 2, 3],
        [3, 0, 2],
    ],
    &[
        [2, 0, 3],
        [1, 2, 0],
        [1, 3, 2],
        [2, 0, 3],
        [1, 0, 2],
        [1, 2, 3],
    ],
    &[
        [0, 2, 3],
        [1, 0, 2],
        [1, 3, 0],
        [0, 2, 3],
        [1, 0, 3],
        [1, 2, 0],
    ],
    &[
        [1, 0, 2],
        [2, 3, 0],
        [3, 0, 2],
        [1, 0, 3],
        [3, 0, 2],
        [1, 0, 3],
        [2, 3, 0],
    ],
    &[
        [1, 0, 2],
        [1, 3, 0],
        [2, 0, 3],
        [1, 0, 2],
        [1, 3, 2],
        [2, 0, 3],
        [1, 3, 2],
    ],
    &[
        [1, 0, 3],
        [2, 0, 3],
        [1, 0, 2],
        [2, 0, 3],
        [2, 3, 0],
        [1, 0, 2],
        [2, 3, 0],
    ],
    &[
        [0, 1, 3],
        [0, 3, 2],
        [1, 3, 0],
        [0, 3, 2],
        [1, 3, 2],
        [0, 1, 3],
        [1, 3, 0],
    ],
    &[
        [1, 2, 0],
        [1, 2, 3],
        [3, 2, 0],
        [1, 3, 0],
        [1, 3, 2],
        [3, 2, 0],
        [1, 0, 3],
    ],
    &[
        [1, 0, 2],
        [1, 0, 3],
        [1, 3, 2],
        [2, 0, 3],
        [1, 2, 0],
        [1, 2, 3],
        [2, 0, 3],
    ],
    &[
        [1, 2, 0],
        [1, 3, 2],
        [2, 0, 3],
        [1, 0, 2],
        [1, 2, 3],
        [2, 0, 3],
    ],
    &[
        [1, 3, 0],
        [2, 0, 3],
        [2, 3, 0],
        [1, 2, 0],
        [2, 0, 3],
        [2, 3, 0],
        [1, 2, 0],
    ],
    &[
        [1, 0, 2],
        [1, 2, 3],
        [2, 0, 3],
        [1, 2, 0],
        [1, 3, 2],
        [2, 0, 3],
    ],
    &[
        [1, 0, 2],
        [3, 0, 2],
        [3, 2, 0],
        [1, 3, 2],
        [3, 0, 2],
        [3, 2, 0],
        [1, 3, 2],
    ],
    &[
        [1, 0, 2],
        [1, 3, 0],
        [0, 2, 3],
        [1, 0, 3],
        [1, 2, 0],
        [0, 2, 3],
    ],
    &[
        [3, 0, 2],
        [1, 2, 3],
        [3, 0, 2],
        [1, 3, 0],
        [0, 3, 2],
        [1, 3, 0],
        [0, 3, 2],
    ],
    &[
        [0, 2, 1],
        [0, 3, 2],
        [1, 0, 2],
        [0, 3, 2],
        [1, 3, 2],
        [0, 2, 1],
        [1, 0, 2],
    ],
    &[
        [0, 3, 2],
        [1, 0, 2],
        [1, 3, 0],
        [0, 3, 2],
        [1, 0, 3],
        [1, 2, 0],
    ],
    &[
        [1, 0, 2],
        [1, 3, 2],
        [3, 2, 0],
        [0, 3, 2],
        [1, 0, 2],
        [0, 3, 2],
        [3, 2, 0],
    ],
    &[
        [0, 2, 1],
        [1, 3, 2],
        [3, 0, 2],
        [1, 3, 2],
        [3, 0, 2],
        [0, 2, 1],
        [1, 0, 2],
    ],
    &[
        [0, 3, 2],
        [1, 2, 0],
        [1, 2, 3],
        [1, 3, 0],
        [0, 3, 2],
        [1, 2, 0],
        [1, 3, 0],
    ],
    &[
        [3, 2, 0],
        [1, 0, 3],
        [1, 3, 2],
        [3, 2, 0],
        [1, 2, 3],
        [1, 3, 0],
    ],
    &[
        [1, 0, 2],
        [3, 2, 0],
        [1, 0, 3],
        [1, 2, 3],
        [3, 2, 0],
        [1, 0, 3],
        [1, 2, 3],
    ],
    &[
        [1, 0, 3],
        [1, 2, 0],
        [0, 3, 2],
        [1, 0, 2],
        [1, 3, 0],
        [0, 3, 2],
    ],
    &[
        [2, 0, 3],
        [1, 0, 2],
        [1, 2, 3],
        [2, 0, 3],
        [1, 2, 0],
        [1, 3, 2],
    ],
    &[
        [1, 0, 2],
        [0, 3, 2],
        [1, 0, 3],
        [1, 2, 0],
        [0, 3, 2],
        [1, 3, 0],
    ],
    &[
        [1, 0, 2],
        [1, 2, 3],
        [2, 3, 0],
        [1, 2, 0],
        [1, 3, 2],
        [2, 3, 0],
    ],
    &[
        [3, 0, 2],
        [1, 3, 2],
        [3, 0, 2],
        [2, 0, 3],
        [1, 2, 0],
        [2, 0, 3],
        [1, 3, 2],
    ],
    &[
        [1, 0, 3],
        [1, 3, 2],
        [2, 0, 3],
        [1, 2, 0],
        [1, 2, 3],
        [2, 0, 3],
        [1, 0, 2],
    ],
    &[
        [2, 3, 0],
        [1, 2, 0],
        [1, 3, 2],
        [2, 3, 0],
        [1, 0, 2],
        [1, 2, 3],
    ],
    &[
        [0, 1, 3],
        [1, 0, 2],
        [2, 0, 3],
        [1, 0, 2],
        [2, 0, 3],
        [0, 1, 3],
        [1, 3, 0],
    ],
    &[
        [1, 0, 2],
        [3, 0, 2],
        [1, 0, 3],
        [1, 2, 3],
        [3, 0, 2],
        [1, 3, 0],
        [1, 3, 2],
    ],
    &[
        [1, 3, 0],
        [2, 0, 3],
        [1, 2, 0],
        [2, 0, 3],
        [2, 3, 0],
        [1, 2, 0],
        [2, 3, 0],
    ],
];
// CNOT 3-wire at min-2 depths (4,5) is empty — those depths only exist for 4-wire.

// Map a SAMF negation type to which of the two swapped wires (lo, hi) ends up
// negated, in the *current* (post-swap) wire space — i.e. the residual that
// `negation_mask` must record. This is the single source of truth for negation
// propagation; every mask update routes through it.
//
//   0 plain swap                 -> (false, false)
//   1 N1   (negate lo)           -> (true,  false)
//   2 N2   (negate hi)           -> (false, true)
//   3 N12  (negate both)         -> (true,  true)
//
// Each type's pool contains both swap-then-negate and negate-then-swap gate
// sequences (the latter are reversals of the opposite type, which compute the same
// permutation — see SWAP_N1/SWAP_N2). Only the net permutation matters here, so all
// implementations of a type share the same mask effect.
fn neg_flips(neg_type: u16) -> (bool, bool) {
    match neg_type {
        0 => (false, false),
        1 => (true, false),
        2 => (false, true),
        3 => (true, true),
        _ => panic!("Invalid negation type: {}", neg_type),
    }
}

// Apply a transposition's negation to `mask`: swap the two wires' pending
// negations, then toggle per `neg_flips`. Indices are in the current wire space.
fn apply_neg_to_mask(mask: &mut [u8], lo: usize, hi: usize, neg_type: u16) {
    mask.swap(lo, hi);
    let (flip_lo, flip_hi) = neg_flips(neg_type);
    if flip_lo {
        mask[lo] ^= 1;
    }
    if flip_hi {
        mask[hi] ^= 1;
    }
}

// Draw a random SAMF negation type (0..=3).
fn random_neg_type<R: Rng + ?Sized>(rng: &mut R) -> u16 {
    rng.random_range(0u16..=3)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Transpositions {
    pub transpositions: Vec<(u16, u16, u16)>,
}

impl Transpositions {
    // Use Knuth Shuffle to get a random wire shuffle and then choose a random negation
    pub fn gen_random_knuth(n: usize, _m: usize, negation_mask: &mut Vec<u8>) -> Self {
        assert!(n >= 2, "n must be at least 2");
        let mut rng = rand::rng();
        let mut transpositions = Vec::with_capacity(n);
        let n = (n - 1) as u16;
        for i in (1..=n).rev() {
            let negation_type = random_neg_type(&mut rng);
            let j = rng.random_range(0..=i);
            if i == j {
                continue;
            }
            transpositions.push((j, i, negation_type));
            apply_neg_to_mask(negation_mask, j as usize, i as usize, negation_type);
        }

        Self { transpositions }
    }

    // Simple random wire shuffle with negation
    pub fn gen_random_simple(n: usize, m: usize, negation_mask: &mut Vec<u8>) -> Self {
        assert!(n >= 2, "n must be at least 2");
        let mut rng = rand::rng();
        let mut transpositions = Vec::with_capacity(m);
        for _ in 0..m {
            let negation_type = random_neg_type(&mut rng);
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
            transpositions.push((j as u16, i as u16, negation_type));
            // Adjust negation mask appropriately
            apply_neg_to_mask(negation_mask, j, i, negation_type);
        }

        Self { transpositions }
    }

    pub fn to_perm(&self, n: usize) -> Permutation {
        let mut perm = Permutation {
            data: Vec::with_capacity(n),
        };
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

        Transpositions {
            transpositions: swaps,
        }
    }

    pub fn collides(s1: &(u16, u16, u16), s2: &(u16, u16, u16)) -> bool {
        let (a1, b1, _) = s1;
        let (a2, b2, _) = s2;
        a1 == a2 || a1 == b2 || b1 == a2 || b1 == b2
    }

    //b is greater
    pub fn ordered(s1: &(u16, u16, u16), s2: &(u16, u16, u16)) -> bool {
        let (a_1, b_1, _) = s1;
        let (a_2, b_2, _) = s2;
        if a_1 > a_2 {
            return false;
        } else if a_1 == a_2 {
            if b_1 > b_2 {
                return false;
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
    pub fn gen_gates_swap(n: usize, swap: (u16, u16, u16)) -> Vec<[u16; 3]> {
        let (a, b, negation_type) = swap;
        let (pool_3w, pool_4w): (&[&[[u16; 3]]], &[&[[u16; 3]]]) = match negation_type {
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
                if c != a && c != b {
                    break;
                }
            }
            let out = CircuitSeq {
                gates: circuit.to_vec(),
            };
            CircuitSeq::unrewire_subcircuit(&out, &[c, a, b]).gates
        } else {
            let circuit = pool_4w[idx - pool_3w.len()];
            let mut c1;
            loop {
                c1 = rng.random_range(0..n) as u16;
                if c1 != a && c1 != b {
                    break;
                }
            }
            let mut c2;
            loop {
                c2 = rng.random_range(0..n) as u16;
                if c2 != a && c2 != b && c2 != c1 {
                    break;
                }
            }
            let out = CircuitSeq {
                gates: circuit.to_vec(),
            };
            CircuitSeq::unrewire_subcircuit(&out, &[c1, a, b, c2]).gates
        }
    }

    // Wire 1 gets flipped in the template; relabels wire 1 to `wire`.
    // Uses 4-wire circuits (min depth 6-7) with 3 ancilla wires.
    pub fn gen_gates_not(n: usize, wire: u16) -> Vec<[u16; 3]> {
        let mut rng = rand::rng();
        let pool = if !NOT_4W.is_empty() { NOT_4W } else { NOT_3W };
        let circuit = pool.choose(&mut rng).expect("NOT pool is empty");
        if pool.as_ptr() == NOT_4W.as_ptr() {
            let mut a;
            loop {
                a = rng.random_range(0..n) as u16;
                if a != wire {
                    break;
                }
            }
            let mut b;
            loop {
                b = rng.random_range(0..n) as u16;
                if b != wire && b != a {
                    break;
                }
            }
            let mut c;
            loop {
                c = rng.random_range(0..n) as u16;
                if c != wire && c != a && c != b {
                    break;
                }
            }
            let out = CircuitSeq {
                gates: circuit.to_vec(),
            };
            CircuitSeq::unrewire_subcircuit(&out, &[a, wire, b, c]).gates
        } else {
            let mut a;
            loop {
                a = rng.random_range(0..n) as u16;
                if a != wire {
                    break;
                }
            }
            let mut b;
            loop {
                b = rng.random_range(0..n) as u16;
                if b != wire && b != a {
                    break;
                }
            }
            let out = CircuitSeq {
                gates: circuit.to_vec(),
            };
            CircuitSeq::unrewire_subcircuit(&out, &[a, wire, b]).gates
        }
    }

    pub fn to_circuit(&self, n: usize) -> CircuitSeq {
        let mut gates: Vec<[u16; 3]> = Vec::new();

        for &swap in &self.transpositions {
            gates.extend_from_slice(&Self::gen_gates_swap(n, swap));
        }

        CircuitSeq { gates }
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

// Append a SAMF/NOT gadget (`samf`) to `gates`, first trying to fuse it with the last
// up-to-3 already-emitted gates through a single curated-DB lookup. On a hit, that window
// (the trailing gates + the gadget) is replaced by the equal-or-shorter equivalent the DB
// returns; on a miss the gadget is appended verbatim. `compress_curated_lmdb` only ever
// returns a circuit equivalent to the window it was given (and returns None when the DBs
// are empty), so this is equivalence-preserving.
fn integrate_samf_compressed(
    gates: &mut Vec<[u16; 3]>,
    samf: &[[u16; 3]],
    n: usize,
    env: &Environment,
    curated_shard_dbs: &[Database],
    shard_dbs: &[Database],
) {
    use crate::replace::pairs::{compress_curated_lmdb, find_any_replacement_lmdb};
    // No DBs to consult -> append verbatim (matches the old behaviour and avoids
    // opening a read txn on a possibly-empty env).
    if curated_shard_dbs.is_empty() && shard_dbs.is_empty() {
        gates.extend_from_slice(samf);
        return;
    }
    let take = gates.len().min(3);
    let ctx_start = gates.len() - take;
    let mut window: Vec<[u16; 3]> = gates[ctx_start..].to_vec();
    window.extend_from_slice(samf);
    // Tiered replacement, best-quality first; all tiers are equivalence-preserving:
    //   1. curated compression (strictly-useful shorter replacement),
    //   2. any curated equivalent (even if not a compression) — still hides the SAMF,
    //   3. any sharded equivalent — fall back to the full sharded DB.
    // On a total miss, emit the undo SAMF gadget verbatim (correct, just unhidden).
    let repl = compress_curated_lmdb(&window, n, env, curated_shard_dbs, &[])
        .or_else(|| find_any_replacement_lmdb(&window, n, env, curated_shard_dbs, &[]))
        .or_else(|| find_any_replacement_lmdb(&window, n, env, &[], shard_dbs));
    if let Some(repl) = repl {
        gates.truncate(ctx_start);
        gates.extend_from_slice(&repl);
        END_SAMF_COMPRESSIONS_MADE.fetch_add(1, Ordering::Relaxed);
    } else {
        gates.extend_from_slice(samf);
    }
}

// Undo the accumulated wire permutation + pending negations described by `t_list` and
// `negation_mask`, appending the inverse SAMFs (and any leftover NOTs for permutation fixed
// points) to `output`. Each gadget is fused with the trailing emitted gates via the curated
// DB (integrate_samf_compressed). This is the shared final "unsamf" step for every shuffle
// function. `negation_mask` must be indexed in the same (current) wire space that `t_list`'s
// net permutation maps the original wires into.
pub fn apply_unsamf(
    output: &mut Vec<[u16; 3]>,
    t_list: &Transpositions,
    negation_mask: &[u8],
    n: usize,
    env: &Environment,
    curated_shard_dbs: &[Database],
    shard_dbs: &[Database],
) {
    let p = t_list.to_perm(n);
    let mut t = Transpositions::from_perm(&p);
    let mut wire_positions: HashMap<u16, (usize, usize)> = HashMap::new();
    for (idx, (wa, wb, _)) in t.transpositions.iter().enumerate() {
        wire_positions.insert(*wa, (idx, 0));
        wire_positions.insert(*wb, (idx, 1));
    }
    const TRANSITION: [[u8; 4]; 2] = [[1, 0, 3, 2], [2, 3, 0, 1]];
    let mut leftover_nots: Vec<u16> = Vec::new();
    for (wire, &val) in negation_mask.iter().enumerate() {
        if val == 1 {
            if let Some(&(swap_idx, pos)) = wire_positions.get(&(wire as u16)) {
                let curr = t.transpositions[swap_idx].2;
                if pos > 1 || curr > 3 {
                    panic!("Invalid pos or curr_neg_type");
                }
                t.transpositions[swap_idx].2 = TRANSITION[pos][curr as usize] as u16;
            } else {
                // Fixed point of the permutation: no transposition to fold the residual
                // negation into, so undo it with an explicit NOT after the permutation.
                leftover_nots.push(wire as u16);
            }
        }
    }
    // Emit transpositions in reverse with each gadget reversed — reproduces the permutation
    // of the old `to_circuit().reverse()` — then the leftover NOTs. The inverse is
    // reconstructed purely from (permutation, negation_mask): `from_perm` yields neg 0 and
    // the `TRANSITION` fold above only ever produces neg 0..=3.
    for &swap in t.transpositions.iter().rev() {
        let mut samf = Transpositions::gen_gates_swap(n, swap);
        samf.reverse();
        integrate_samf_compressed(output, &samf, n, env, curated_shard_dbs, shard_dbs);
    }
    for w in leftover_nots {
        let not_gates = Transpositions::gen_gates_not(n, w);
        integrate_samf_compressed(output, &not_gates, n, env, curated_shard_dbs, shard_dbs);
    }
}

pub fn insert_wire_shuffles_knuth(
    circuit: &mut CircuitSeq,
    n: usize,
    env: &Environment,
    curated_shard_dbs: &[Database],
    shard_dbs: &[Database],
) {
    println!("Inserting wire shuffles (knuth)");
    println!("Starting len: {} gates", circuit.gates.len());
    let mut t_list: Transpositions = Transpositions {
        transpositions: Vec::new(),
    };
    let mut gates: Vec<[u16; 3]> = Vec::new();
    let mut negation_mask = vec![0u8; n];

    for &gate in &circuit.gates {
        let t = Transpositions::gen_random_knuth(n, 150, &mut negation_mask);
        gates.extend_from_slice(&t.to_circuit(n).gates);
        t_list.transpositions.extend_from_slice(&t.transpositions);
        SAMF_INSERTIONS_MADE.fetch_add(t.transpositions.len(), Ordering::Relaxed);
        let a = t_list.evaluate(gate[0]);
        let b = t_list.evaluate(gate[1]);
        let c = t_list.evaluate(gate[2]);
        let gate = [a, b, c];
        if negation_mask[b as usize] == 1 {
            gates.extend_from_slice(&Transpositions::gen_gates_not(n, b));
            negation_mask[b as usize] = 0;
        }
        if negation_mask[c as usize] == 1 {
            gates.extend_from_slice(&Transpositions::gen_gates_not(n, c));
            negation_mask[c as usize] = 0;
        }
        gates.push(gate);
    }
    apply_unsamf(
        &mut gates,
        &t_list,
        &negation_mask,
        n,
        env,
        curated_shard_dbs,
        shard_dbs,
    );
    circuit.gates = gates;
    println!("Complete. Ending len: {} gates", circuit.gates.len());
}

pub fn insert_wire_shuffles_simple(
    circuit: &mut CircuitSeq,
    n: usize,
    env: &Environment,
    curated_shard_dbs: &[Database],
    shard_dbs: &[Database],
) {
    println!("Inserting wire shuffles (simple)");
    println!("Starting len: {} gates", circuit.gates.len());
    let mut t_list: Transpositions = Transpositions {
        transpositions: Vec::new(),
    };
    let mut gates: Vec<[u16; 3]> = Vec::new();
    let mut negation_mask = vec![0u8; n];

    // Generate random points. m needed in k = m * n
    // Choose them spaced approximately evenly but with `sufficient` randomness
    let m = circuit.gates.len();
    let mut points = Vec::with_capacity(m);
    let mut rng = rand::rng();
    for i in 0..m {
        let center = i * n + n / 2;

        // allow significant variance but keep spacing structure
        let jitter = rng.random_range(-(n as i64) / 2..=(n as i64) / 2);

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
        gates.extend_from_slice(&t.to_circuit(n).gates);
        t_list.transpositions.extend_from_slice(&t.transpositions);
        SAMF_INSERTIONS_MADE.fetch_add(t.transpositions.len(), Ordering::Relaxed);
        let a = t_list.evaluate(gate[0]);
        let b = t_list.evaluate(gate[1]);
        let c = t_list.evaluate(gate[2]);
        let gate = [a, b, c];
        if negation_mask[b as usize] == 1 {
            gates.extend_from_slice(&Transpositions::gen_gates_not(n, b));
            negation_mask[b as usize] = 0;
        }
        if negation_mask[c as usize] == 1 {
            gates.extend_from_slice(&Transpositions::gen_gates_not(n, c));
            negation_mask[c as usize] = 0;
        }
        gates.push(gate);
    }
    apply_unsamf(
        &mut gates,
        &t_list,
        &negation_mask,
        n,
        env,
        curated_shard_dbs,
        shard_dbs,
    );
    circuit.gates = gates;
    println!("Complete. Ending len: {} gates", circuit.gates.len());
}

// Insert 2 shuffles are the beginning and end, and then an additional x number of shuffles
pub fn insert_wire_shuffles_x(
    circuit: &mut CircuitSeq,
    n: usize,
    x: usize,
    env: &Environment,
    curated_shard_dbs: &[Database],
    shard_dbs: &[Database],
) {
    println!("Inserting wire shuffles");
    println!("Starting len: {} gates", circuit.gates.len());
    let mut t_list: Transpositions = Transpositions {
        transpositions: Vec::new(),
    };
    let mut gates: Vec<[u16; 3]> = Vec::new();
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
            gates.extend_from_slice(&t.to_circuit(n).gates);
            t_list.transpositions.extend_from_slice(&t.transpositions);
            SAMF_INSERTIONS_MADE.fetch_add(t.transpositions.len(), Ordering::Relaxed);
        }
        let a = t_list.evaluate(gate[0]);
        let b = t_list.evaluate(gate[1]);
        let c = t_list.evaluate(gate[2]);
        let gate = [a, b, c];
        if negation_mask[b as usize] == 1 {
            gates.extend_from_slice(&Transpositions::gen_gates_not(n, b));
            negation_mask[b as usize] = 0;
        }
        if negation_mask[c as usize] == 1 {
            gates.extend_from_slice(&Transpositions::gen_gates_not(n, c));
            negation_mask[c as usize] = 0;
        }
        gates.push(gate);
    }
    apply_unsamf(
        &mut gates,
        &t_list,
        &negation_mask,
        n,
        env,
        curated_shard_dbs,
        shard_dbs,
    );
    circuit.gates = gates;
    println!("Complete. Ending len: {} gates", circuit.gates.len());
}

// Insert m samf between each gate
// Core of insert_wire_m_samfs_every_x: insert `m` SAMFs every `x` gates over `input`,
// returning the rebuilt gates plus the accumulated (t_list, negation_mask) WITHOUT undoing.
// `negation_mask` is the starting pending-negation state (all-zero for a standalone call,
// or carried from a previous pass), indexed in `input`'s wire space.
fn insert_m_samfs_core(
    input: &[[u16; 3]],
    n: usize,
    m: usize,
    x: usize,
) -> (Vec<[u16; 3]>, Transpositions, Vec<u8>) {
    let mut t_list = Transpositions {
        transpositions: Vec::new(),
    };
    let mut gates: Vec<[u16; 3]> = Vec::new();
    let mut negation_mask = vec![0u8; n];
    for (i, gate) in input.iter().enumerate() {
        if i % x == 0 {
            let t = Transpositions::gen_random_simple(n, m, &mut negation_mask);
            gates.extend_from_slice(&t.to_circuit(n).gates);
            t_list.transpositions.extend_from_slice(&t.transpositions);
            SAMF_INSERTIONS_MADE.fetch_add(t.transpositions.len(), Ordering::Relaxed);
        }
        let a = t_list.evaluate(gate[0]);
        let b = t_list.evaluate(gate[1]);
        let c = t_list.evaluate(gate[2]);
        let g = [a, b, c];
        if negation_mask[b as usize] == 1 {
            gates.extend_from_slice(&Transpositions::gen_gates_not(n, b));
            negation_mask[b as usize] = 0;
        }
        if negation_mask[c as usize] == 1 {
            gates.extend_from_slice(&Transpositions::gen_gates_not(n, c));
            negation_mask[c as usize] = 0;
        }
        gates.push(g);
    }
    (gates, t_list, negation_mask)
}

// Insert m samf between each gate
pub fn insert_wire_m_samfs_every_x(
    circuit: &mut CircuitSeq,
    n: usize,
    m: usize,
    x: usize,
    env: &Environment,
    curated_shard_dbs: &[Database],
    shard_dbs: &[Database],
) {
    println!("Inserting {} samfs between each gate", m);
    println!("Starting len: {} gates", circuit.gates.len());
    let (mut gates, t_list, negation_mask) = insert_m_samfs_core(&circuit.gates, n, m, x);
    apply_unsamf(
        &mut gates,
        &t_list,
        &negation_mask,
        n,
        env,
        curated_shard_dbs,
        shard_dbs,
    );
    circuit.gates = gates;
    println!("Complete. Ending len: {} gates", circuit.gates.len());
}

fn gates_collide(g1: [u16; 3], g2: [u16; 3]) -> bool {
    g1[0] == g2[1] || g1[0] == g2[2] || g2[0] == g1[1] || g2[0] == g1[2]
}

// For each collision (adjacent gates that can't commute), make a curated EXPANSION of a window of
// up to `gates_ahead_expand` gates anchored at the colliding pair, then try to hide a single SAMF
// in the expansion's tail and look up an equal-or-shorter replacement in the curated DB.
//
// The expansion window is anchored at the colliding pair and shrinks by one gate on each
// curated-DB miss (gates_ahead_expand .. 2, down to the 2-gate pair); a window is eligible only
// if every control wire in it is clean (no pending negation correction).
//
// The SAMF-hiding window is the last `gates_ahead_samf` gates ending at the expansion's tail —
// reaching back into the already-emitted output when the expansion is shorter than
// `gates_ahead_samf` — followed by the first 3 gates of the SAMF. On a successful hide the
// remaining SAMF gates are emitted after the replacement.
//
// `type_attempts` controls how many DISTINCT SAMF gate (negation) types are tried per collision
// before giving up: each attempt samples a not-yet-tried type (without replacement) and one
// random hardcoded SAMF of that type. The first type that yields a hiding window wins;
// `type_attempts == 1` is the original single-try behaviour.
//
// If a replacement is found, output it followed by the remaining SAMF gates. Future gates are
// relabeled by the SAMF's wire permutation, and the accumulated permutation is undone at the end.
fn shuffled_shooting_game_core(
    input: &[[u16; 3]],
    n: usize,
    env: &Environment,
    curated_shard_dbs: &[Database],
    shard_dbs: &[Database],
    gates_ahead_expand: usize,
    gates_ahead_samf: usize,
    type_attempts: usize,
) -> (Vec<[u16; 3]>, Transpositions, Vec<u8>, usize) {
    use crate::replace::pairs::{compress_curated_lmdb, expand_curated_lmdb};

    let mut rng = rand::rng();
    let mut output: Vec<[u16; 3]> = Vec::new();
    let mut t_list = Transpositions {
        transpositions: Vec::new(),
    };
    let mut negation_mask = vec![0u8; n];
    let mut compressions: usize = 0;

    let mut i = 0;

    while i < input.len() {
        let gate = input[i];
        let a = t_list.evaluate(gate[0]);
        let b = t_list.evaluate(gate[1]);
        let c = t_list.evaluate(gate[2]);

        // On a collision, expand a window of up to `gates_ahead_expand` gates anchored at the
        // colliding pair, then try to hide a single SAMF in the expansion's tail. The expansion is
        // computed once; if no SAMF can be hidden we keep it verbatim (no recompute).
        let replaced = 'try_replace: {
            if i + 1 >= input.len() {
                break 'try_replace false;
            }

            let next = input[i + 1];
            let na = t_list.evaluate(next[0]);
            let nb = t_list.evaluate(next[1]);
            let nc = t_list.evaluate(next[2]);

            if !gates_collide([a, b, c], [na, nb, nc]) {
                break 'try_replace false;
            }

            // 1) Curated expansion of the largest clean window anchored at the colliding pair: an
            //    equivalent, longer circuit. The window shrinks from its far end
            //    (gates_ahead_expand .. 2) until a curated expansion is found; a window is eligible
            //    only if every control wire in it is clean (no pending negation correction), so the
            //    expansion stays equivalence-safe. `consumed` input gates map to `expansion`.
            let max_window = gates_ahead_expand.max(2).min(input.len() - i);
            let mut expanded: Option<(usize, Vec<[u16; 3]>)> = None;
            for k in (2..=max_window).rev() {
                let mut window: Vec<[u16; 3]> = Vec::with_capacity(k);
                let mut clean = true;
                for g in &input[i..i + k] {
                    let rg = [
                        t_list.evaluate(g[0]),
                        t_list.evaluate(g[1]),
                        t_list.evaluate(g[2]),
                    ];
                    // Controls (positions 1 and 2) must carry no pending negation.
                    if negation_mask[rg[1] as usize] != 0 || negation_mask[rg[2] as usize] != 0 {
                        clean = false;
                        break;
                    }
                    window.push(rg);
                }
                if !clean {
                    continue;
                }
                if let Some(e) = expand_curated_lmdb(&window, n, env, curated_shard_dbs, shard_dbs)
                {
                    if e.len() >= 3 {
                        expanded = Some((k, e));
                        break;
                    }
                }
            }
            let (consumed, expansion) = match expanded {
                Some(x) => x,
                None => break 'try_replace false, // no curated expansion -> normal path
            };
            CURATED_REPLACEMENTS_MADE.fetch_add(1, Ordering::Relaxed);
            // Distinct-wire counts of the consumed input window (evaluated) vs the expansion.
            let distinct_wires = |gates: &[[u16; 3]]| {
                let mut seen = std::collections::HashSet::new();
                for g in gates {
                    seen.insert(g[0]);
                    seen.insert(g[1]);
                    seen.insert(g[2]);
                }
                seen.len()
            };
            let before_wires = {
                let evaluated: Vec<[u16; 3]> = input[i..i + consumed]
                    .iter()
                    .map(|g| {
                        [
                            t_list.evaluate(g[0]),
                            t_list.evaluate(g[1]),
                            t_list.evaluate(g[2]),
                        ]
                    })
                    .collect();
                distinct_wires(&evaluated)
            };
            crate::replace::replace::record_expansion(
                consumed,
                expansion.len(),
                before_wires,
                distinct_wires(&expansion),
            );

            // 2) Try to hide ONE SAMF ending at the expansion's tail. The context for the hide is
            //    the last `gates_ahead_samf` gates of (output ++ expansion) — reaching back into
            //    the already-emitted output when the expansion is shorter — followed by samf[0..3].
            //    Pick the swap wires once and vary the negation type across attempts.
            let swap_lo: u16 = rng.random_range(0..n as u16);
            let swap_hi: u16 = loop {
                let w: u16 = rng.random_range(0..n as u16);
                if w != swap_lo {
                    break w;
                }
            };
            let (swap_lo, swap_hi) = if swap_lo < swap_hi {
                (swap_lo, swap_hi)
            } else {
                (swap_hi, swap_lo)
            };

            // Split the gates_ahead_samf-gate context between the expansion tail and the gates
            // immediately before it in `output`. When the expansion is shorter than the context,
            // the whole expansion is used (exp_tail_start == 0) and the remainder comes from
            // output; otherwise the context is wholly inside the expansion (from_output == 0).
            let ctx = gates_ahead_samf.max(1);
            let exp_take = expansion.len().min(ctx);
            let exp_tail_start = expansion.len() - exp_take;
            let from_output = (ctx - exp_take).min(output.len());
            let out_keep = output.len() - from_output;

            let candidate_types: Vec<u16> = (0u16..=3).collect();
            // (neg_type, samf, repl): the context window ++ samf[0..3] becomes `repl`, and
            // samf[3..] is emitted after it.
            let mut tuck: Option<(u16, Vec<[u16; 3]>, Vec<[u16; 3]>)> = None;
            'types: for &neg_type in candidate_types.choose_multiple(&mut rng, type_attempts.max(1))
            {
                let samf = Transpositions::gen_gates_swap(n, (swap_lo, swap_hi, neg_type));
                if samf.len() < 3 {
                    continue;
                }
                // Window: [output tail] ++ [expansion tail] ++ samf[0..3]. Always contains SAMF
                // gates, so every lookup is a genuine hide attempt.
                let mut window: Vec<[u16; 3]> = Vec::with_capacity(ctx + 3);
                window.extend_from_slice(&output[out_keep..]);
                window.extend_from_slice(&expansion[exp_tail_start..]);
                window.extend_from_slice(&samf[..3]);
                if let Some(repl) =
                    compress_curated_lmdb(&window, n, env, curated_shard_dbs, shard_dbs)
                {
                    // Accept only if the SAMF gates are genuinely absorbed (not surviving verbatim).
                    let samf_slice = &samf[..3];
                    let samf_hidden = repl.len() < 3 || !repl.windows(3).any(|w| w == samf_slice);
                    if samf_hidden {
                        tuck = Some((neg_type, samf, repl));
                        break 'types;
                    }
                }
            }

            match tuck {
                Some((neg_type, samf, repl)) => {
                    // The context window (output tail + expansion tail + samf[0..3]) becomes
                    // `repl`; everything before it is unchanged and samf[3..] follows.
                    output.truncate(out_keep);
                    output.extend_from_slice(&expansion[..exp_tail_start]);
                    output.extend_from_slice(&repl);
                    output.extend_from_slice(&samf[3..]);
                    t_list.transpositions.push((swap_lo, swap_hi, neg_type));
                    apply_neg_to_mask(
                        &mut negation_mask,
                        swap_lo as usize,
                        swap_hi as usize,
                        neg_type,
                    );
                    compressions += 1;
                    SAMF_COMPRESSIONS_MADE.fetch_add(1, Ordering::Relaxed);
                    SAMF_INSERTIONS_MADE.fetch_add(1, Ordering::Relaxed);
                }
                None => {
                    // Keep the curated expansion verbatim; no SAMF hidden this time.
                    output.extend_from_slice(&expansion);
                    SAMF_COMPRESSIONS_FAILED.fetch_add(1, Ordering::Relaxed);
                }
            }

            i += consumed; // consumed the expansion's input window
            true
        };
        if !replaced {
            // Normal path: flush any pending negations on control wires, then emit the gate.
            if negation_mask[b as usize] == 1 {
                output.extend_from_slice(&Transpositions::gen_gates_not(n, b));
                negation_mask[b as usize] = 0;
            }
            if negation_mask[c as usize] == 1 {
                output.extend_from_slice(&Transpositions::gen_gates_not(n, c));
                negation_mask[c as usize] = 0;
            }
            output.push([a, b, c]);
            i += 1;
        }
    }

    (output, t_list, negation_mask, compressions)
}

fn shuffled_shooting_game_repeated_core(
    input: &[[u16; 3]],
    n: usize,
    env: &Environment,
    curated_shard_dbs: &[Database],
    shard_dbs: &[Database],
    gates_ahead_expand: usize,
    gates_ahead_samf: usize,
    type_attempts: usize,
    shooting_times: usize,
) -> (Vec<[u16; 3]>, Transpositions, Vec<u8>, usize) {
    let passes = shooting_times.max(1);
    let mut input_gates = input.to_vec();
    let mut total_t = Transpositions {
        transpositions: Vec::new(),
    };
    let mut total_neg = vec![0u8; n];
    let mut total_compressions = 0usize;

    for _ in 0..passes {
        let (out, t_pass, neg_pass, compressions) = shuffled_shooting_game_core(
            &input_gates,
            n,
            env,
            curated_shard_dbs,
            shard_dbs,
            gates_ahead_expand,
            gates_ahead_samf,
            type_attempts,
        );
        let mut new_total_neg = neg_pass;
        for w in 0..n {
            if total_neg[w] == 1 {
                let cw = t_pass.evaluate(w as u16) as usize;
                new_total_neg[cw] ^= 1;
            }
        }
        total_neg = new_total_neg;
        total_t = total_t.concat(&t_pass);
        total_compressions += compressions;
        input_gates = out;
    }

    (input_gates, total_t, total_neg, total_compressions)
}

// Standalone shuffled shooting game: run the core, then undo its accumulated SAMFs.
pub fn shuffled_shooting_game(
    circuit: &mut CircuitSeq,
    n: usize,
    env: &Environment,
    curated_shard_dbs: &[Database],
    shard_dbs: &[Database],
    gates_ahead_expand: usize,
    gates_ahead_samf: usize,
    type_attempts: usize,
    shooting_times: usize,
) -> usize {
    let (mut output, t_list, negation_mask, compressions) = shuffled_shooting_game_repeated_core(
        &circuit.gates,
        n,
        env,
        curated_shard_dbs,
        shard_dbs,
        gates_ahead_expand,
        gates_ahead_samf,
        type_attempts,
        shooting_times,
    );
    apply_unsamf(
        &mut output,
        &t_list,
        &negation_mask,
        n,
        env,
        curated_shard_dbs,
        shard_dbs,
    );
    circuit.gates = output;
    compressions
}

// Core of shuffled_shoot_then_samf: run the shooting game then per-gate SAMF insertion,
// returning the rebuilt gates plus this pass's combined (t_list, negation_mask) and the
// compression count, WITHOUT undoing. `--single-end` uses this to accumulate SAMF state
// across rounds and undo only once at the very end. Insertion reprocesses the shooting
// game's output from a CLEAN negation state (its gates are self-contained; neg_a is a
// final-state adjustment, not a pre-condition), and neg_a is transported through the
// insertion permutation t_b into the returned negation_mask.
pub fn shuffled_shoot_then_samf_core(
    input: &[[u16; 3]],
    n: usize,
    m: usize,
    x: usize,
    gates_ahead_expand: usize,
    gates_ahead_samf: usize,
    type_attempts: usize,
    shooting_times: usize,
    env: &Environment,
    curated_shard_dbs: &[Database],
    shard_dbs: &[Database],
) -> (Vec<[u16; 3]>, Transpositions, Vec<u8>, usize) {
    let (out_a, t_a, neg_a, compressions) = shuffled_shooting_game_repeated_core(
        input,
        n,
        env,
        curated_shard_dbs,
        shard_dbs,
        gates_ahead_expand,
        gates_ahead_samf,
        type_attempts,
        shooting_times,
    );
    let (out_b, t_b, neg_b) = insert_m_samfs_core(&out_a, n, m, x);
    // Combined permutation: shooting game first (t_a), then insertion (t_b).
    let t_round = t_a.concat(&t_b);
    // Combined final negation: insertion's own (neg_b) plus the shooting game's (neg_a)
    // transported through the insertion permutation t_b.
    let mut neg_round = neg_b;
    for w in 0..n {
        if neg_a[w] == 1 {
            let cw = t_b.evaluate(w as u16) as usize;
            neg_round[cw] ^= 1;
        }
    }
    (out_b, t_round, neg_round, compressions)
}

// Run the shuffled shooting game then per-gate SAMF insertion as ONE shuffle with a SINGLE
// unsamf at the very end. Equivalent to the two functions run back-to-back, but with one
// undo instead of two. Returns the shooting game's compression count.
pub fn shuffled_shoot_then_samf(
    circuit: &mut CircuitSeq,
    n: usize,
    m: usize,
    x: usize,
    gates_ahead_expand: usize,
    gates_ahead_samf: usize,
    type_attempts: usize,
    shooting_times: usize,
    env: &Environment,
    curated_shard_dbs: &[Database],
    shard_dbs: &[Database],
) -> usize {
    let (mut out, t_round, neg_round, compressions) = shuffled_shoot_then_samf_core(
        &circuit.gates,
        n,
        m,
        x,
        gates_ahead_expand,
        gates_ahead_samf,
        type_attempts,
        shooting_times,
        env,
        curated_shard_dbs,
        shard_dbs,
    );
    apply_unsamf(
        &mut out,
        &t_round,
        &neg_round,
        n,
        env,
        curated_shard_dbs,
        shard_dbs,
    );
    circuit.gates = out;
    compressions
}

#[cfg(test)]
mod reversed_samf_tests {
    use super::{
        SWAP_N1_3W, SWAP_N1_4W, SWAP_N2_3W, SWAP_N2_4W, Transpositions, neg_flips, random_neg_type,
    };
    use crate::circuit::circuit::CircuitSeq;

    // Structural canonical form of a gadget UP TO (a) wire relabeling and (b) reordering of
    // commuting gates. We densify the used wires, then over every permutation of those wires
    // run CircuitSeq::canonicalize() (which canonicalizes commuting-gate order) and keep the
    // lexicographically-smallest gate sequence. Two gadgets share a key iff they are the same
    // circuit up to relabeling wires and swapping adjacent commuting gates.
    fn structural_key(gates: &[[u16; 3]]) -> Vec<[u16; 3]> {
        use itertools::Itertools;
        let c = CircuitSeq {
            gates: gates.to_vec(),
        };
        let used = c.used_wires(); // sorted, unique
        let k = used.len();
        let dense: std::collections::HashMap<u16, u16> = used
            .iter()
            .enumerate()
            .map(|(i, &w)| (w, i as u16))
            .collect();
        let base: Vec<[u16; 3]> = gates
            .iter()
            .map(|g| [dense[&g[0]], dense[&g[1]], dense[&g[2]]])
            .collect();
        let mut best: Option<Vec<[u16; 3]>> = None;
        for perm in (0..k as u16).permutations(k) {
            let relabeled: Vec<[u16; 3]> = base
                .iter()
                .map(|g| {
                    [
                        perm[g[0] as usize],
                        perm[g[1] as usize],
                        perm[g[2] as usize],
                    ]
                })
                .collect();
            let mut cc = CircuitSeq { gates: relabeled };
            cc.canonicalize();
            if best.as_ref().is_none_or(|b| &cc.gates < b) {
                best = Some(cc.gates);
            }
        }
        best.unwrap()
    }

    // The N1/N2 pools now also contain the (unique) negate-then-swap reversals of the opposite
    // type. Since reverse(N2) computes the N1 permutation (and vice-versa), every reversed
    // circuit of the opposite pool must already appear in the destination pool's structural
    // forms (up to wire relabeling + commuting-gate order) — i.e. the pools are closed under
    // reversal-of-the-opposite-type. This guards that the hardcoded reversals are complete.
    #[test]
    fn pools_closed_under_reversal() {
        use std::collections::HashSet;
        for (dst_pool, src_pool) in [
            (SWAP_N1_3W, SWAP_N2_3W),
            (SWAP_N1_4W, SWAP_N2_4W),
            (SWAP_N2_3W, SWAP_N1_3W),
            (SWAP_N2_4W, SWAP_N1_4W),
        ] {
            let dst: HashSet<Vec<[u16; 3]>> = dst_pool.iter().map(|c| structural_key(c)).collect();
            for c in src_pool.iter() {
                let mut g = c.to_vec();
                g.reverse();
                assert!(
                    dst.contains(&structural_key(&g)),
                    "a reversal of the opposite pool is missing from the destination pool"
                );
            }
        }
    }

    #[test]
    fn random_neg_type_in_range() {
        let mut rng = rand::rng();
        let mut seen = [false; 4];
        for _ in 0..5000 {
            let t = random_neg_type(&mut rng);
            assert!(t <= 3, "neg type out of range: {}", t);
            seen[t as usize] = true;
        }
        assert!(seen.iter().all(|&s| s), "not all of 0..=3 were drawn");
    }

    // Logical 2-bit op of a swap gadget on wires (a, b), all other wires held at 0.
    // Asserts every non-(a,b) wire is restored to 0 (ancilla clean).
    fn logical_op(
        gates: &[[u16; 3]],
        n: usize,
        a: u16,
        b: u16,
        xa: usize,
        xb: usize,
    ) -> (usize, usize) {
        let input = (xa << a) | (xb << b);
        let c = CircuitSeq {
            gates: gates.to_vec(),
        };
        let out = c.evaluate(input);
        for w in 0..n {
            if w as u16 != a && w as u16 != b {
                assert_eq!((out >> w) & 1, 0, "ancilla wire {} not restored to 0", w);
            }
        }
        ((out >> a) & 1, (out >> b) & 1)
    }

    // The net op of any neg_type is "swap, then negate per neg_flips":
    //   out_a = x_b ^ flip_lo,  out_b = x_a ^ flip_hi.
    fn expected(neg: u16, xa: usize, xb: usize) -> (usize, usize) {
        let (flip_lo, flip_hi) = neg_flips(neg);
        (xb ^ flip_lo as usize, xa ^ flip_hi as usize)
    }

    #[test]
    fn neg_flips_parity() {
        assert_eq!(neg_flips(0), (false, false));
        assert_eq!(neg_flips(1), (true, false));
        assert_eq!(neg_flips(2), (false, true));
        assert_eq!(neg_flips(3), (true, true));
    }

    #[test]
    fn gen_gates_swap_logical_op_all_types() {
        // n=3 exercises only 3-wire pools; n=4 also exercises 4-wire pools. Many iterations
        // cover the random pool + ancilla choices, so every pool entry (including the
        // hardcoded negate-then-swap reversals) is checked to compute its type's permutation.
        for &(n, a, b) in &[(3usize, 1u16, 2u16), (4, 1, 3), (4, 0, 2)] {
            for neg in 0u16..=3 {
                for _ in 0..300 {
                    let gates = Transpositions::gen_gates_swap(n, (a, b, neg));
                    assert!(!gates.is_empty());
                    for xa in 0..2 {
                        for xb in 0..2 {
                            assert_eq!(
                                logical_op(&gates, n, a, b, xa, xb),
                                expected(neg, xa, xb),
                                "neg={} n={} (a,b)=({},{}) input=({},{})",
                                neg,
                                n,
                                a,
                                b,
                                xa,
                                xb
                            );
                        }
                    }
                }
            }
        }
    }
}
