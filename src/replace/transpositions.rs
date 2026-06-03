// For adding wire shuffles and bit flips
use crate::circuit::{Permutation, circuit::CircuitSeq};
use lmdb::{Database, Environment};
use rand::Rng;
use rand::seq::IndexedRandom;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

pub static SAMF_COMPRESSIONS_MADE: AtomicUsize = AtomicUsize::new(0);
pub static SAMF_COMPRESSIONS_FAILED: AtomicUsize = AtomicUsize::new(0);
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

// When true, the random shufflers may also emit reversed single-negation SAMFs
// (neg_type 4 = rev-N1, 5 = rev-N2). Set to false to reproduce the pre-reversal
// behaviour (neg_type 0..=3 only). See `neg_flips` / `gen_gates_swap`.
const REVERSED_SAMF: bool = true;

// Map a SAMF negation type to which of the two swapped wires (lo, hi) ends up
// negated, in the *current* (post-swap) wire space — i.e. the residual that
// `negation_mask` must record. This is the single source of truth for negation
// propagation; every mask update routes through it.
//
//   0 plain swap                -> (false, false)
//   1 N1   (swap-then-negate lo)-> (true,  false)
//   2 N2   (swap-then-negate hi)-> (false, true)
//   3 N12  (negate both)        -> (true,  true)
//   4 rev-N1 = reverse(N1)      -> (false, true)   // same permutation as N2
//   5 rev-N2 = reverse(N2)      -> (true,  false)  // same permutation as N1
//
// Reversing a gadget reverses the whole gate list (the inverse permutation), so a
// reversed N1 has the same net swap+negation as a forward N2, and vice-versa.
// Only the emitted gate *sequence* differs (negate-then-swap vs swap-then-negate).
fn neg_flips(neg_type: u16) -> (bool, bool) {
    match neg_type {
        0 => (false, false),
        1 => (true, false),
        2 => (false, true),
        3 => (true, true),
        4 => (false, true),
        5 => (true, false),
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

// Draw a random SAMF negation type. Includes the reversed variants (4, 5) when
// `REVERSED_SAMF` is enabled.
fn random_neg_type<R: Rng + ?Sized>(rng: &mut R) -> u16 {
    if REVERSED_SAMF {
        rng.random_range(0u16..=5)
    } else {
        rng.random_range(0u16..=3)
    }
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
    //
    // neg_type 4 (rev-N1) / 5 (rev-N2) draw from the N1 / N2 pools and reverse the
    // resulting gadget — yielding a negate-then-swap gate sequence whose net effect
    // equals forward N2 / N1 respectively (see `neg_flips`). Reversal commutes with
    // the per-gate unrewiring, so reversing the final unrewired vec is correct and
    // still restores the ancilla wires to 0.
    pub fn gen_gates_swap(n: usize, swap: (u16, u16, u16)) -> Vec<[u16; 3]> {
        let (a, b, negation_type) = swap;
        let (pool_3w, pool_4w, reverse): (&[&[[u16; 3]]], &[&[[u16; 3]]], bool) =
            match negation_type {
                0 => (SWAP_3W, SWAP_4W, false),
                1 => (SWAP_N1_3W, SWAP_N1_4W, false),
                2 => (SWAP_N2_3W, SWAP_N2_4W, false),
                3 => (SWAP_N12_3W, SWAP_N12_4W, false),
                4 => (SWAP_N1_3W, SWAP_N1_4W, true),
                5 => (SWAP_N2_3W, SWAP_N2_4W, true),
                _ => panic!("Invalid negation type"),
            };
        let mut rng = rand::rng();
        let use_4w = n >= 4 && !pool_4w.is_empty();
        let total = pool_3w.len() + if use_4w { pool_4w.len() } else { 0 };
        let idx = rng.random_range(0..total);
        let mut gates = if idx < pool_3w.len() {
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
        };
        if reverse {
            gates.reverse();
        }
        gates
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
    // of the old `to_circuit().reverse()` — then the leftover NOTs.
    //
    // The inverse is reconstructed purely from (permutation, negation_mask): `from_perm`
    // yields neg 0 and the `TRANSITION` fold above only ever produces neg 0..=3, so this
    // never emits reversed (4/5) gadgets and needs no knowledge of them. Reversed SAMFs
    // inserted on the forward pass are already accounted for via `negation_mask`.
    for &swap in t.transpositions.iter().rev() {
        debug_assert!(swap.2 <= 3, "apply_unsamf expects forward neg types only");
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

// For each collision (adjacent gates that can't commute), try inserting the first 3 gates of a
// randomly chosen SAMF (swap-and-maybe-flip circuit) after the collision window and look up an
// equal-or-shorter replacement in the curated DB.
//
// `type_attempts` controls how many DISTINCT SAMF gate (negation) types are tried per collision
// before giving up: each attempt samples a not-yet-tried type (without replacement) and one
// random hardcoded SAMF of that type. The first type that yields a compressing window wins;
// `type_attempts == 1` is the original single-try behaviour.
//
// When gates_ahead > 2, first tries a wider window of gates_ahead gates + samf[0..3]. Falls back
// to the 2-gate collision pair + samf[0..3] if the wider lookup misses.
//
// If a replacement is found, output it followed by the remaining SAMF gates. Future gates are
// relabeled by the SAMF's wire permutation, and the accumulated permutation is undone at the end.
fn shuffled_shooting_game_core(
    input: &[[u16; 3]],
    n: usize,
    env: &Environment,
    curated_shard_dbs: &[Database],
    shard_dbs: &[Database],
    gates_ahead: usize,
    type_attempts: usize,
) -> (Vec<[u16; 3]>, Transpositions, Vec<u8>, usize) {
    use crate::replace::pairs::compress_curated_lmdb;

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

        // Attempt SAMF-assisted replacement when this gate and the next collide.
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

            // Collision pair controls must be clean (no pending negation corrections).
            if negation_mask[b as usize] != 0
                || negation_mask[c as usize] != 0
                || negation_mask[nb as usize] != 0
                || negation_mask[nc as usize] != 0
            {
                break 'try_replace false;
            }

            // Pick the swap wires once; we vary the gate (negation) type across attempts.
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

            // Build all available context gates (up to gates_ahead) with clean flags.
            // Positions 0 and 1 (collision pair controls) are already verified clean.
            // The context is independent of the SAMF, so it is built once and reused
            // across every gate-type attempt.
            let ga = gates_ahead.min(input.len() - i);
            let mut ctx: Vec<([u16; 3], bool)> = Vec::with_capacity(ga);
            ctx.push(([a, b, c], true));
            if ga >= 2 {
                ctx.push(([na, nb, nc], true));
            }
            for k in 2..ga {
                let g = input[i + k];
                let gw0 = t_list.evaluate(g[0]);
                let gw1 = t_list.evaluate(g[1]);
                let gw2 = t_list.evaluate(g[2]);
                let clean = negation_mask[gw1 as usize] == 0 && negation_mask[gw2 as usize] == 0;
                ctx.push(([gw0, gw1, gw2], clean));
            }

            // Try up to `type_attempts` DISTINCT gate (negation) types, sampled without
            // replacement from the available set. For each type, draw one random hardcoded
            // SAMF of that type and search for a compressing window; stop at the first type
            // that yields one. type_attempts == 1 reproduces the old single-try behaviour.
            let candidate_types: Vec<u16> = if REVERSED_SAMF {
                (0u16..=5).collect()
            } else {
                (0u16..=3).collect()
            };
            // (neg_type, samf, start, samf_used, repl)
            let mut winner: Option<(u16, Vec<[u16; 3]>, usize, usize, Vec<[u16; 3]>)> = None;
            for &neg_type in candidate_types.choose_multiple(&mut rng, type_attempts.max(1)) {
                let samf = Transpositions::gen_gates_swap(n, (swap_lo, swap_hi, neg_type));
                if samf.len() < 3 {
                    continue;
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
                        if end > ga + 3 {
                            continue;
                        } // beyond full window
                        if end <= ga {
                            continue;
                        } // no SAMF gate
                        // All context gates in [start..end.min(ga)] must be clean.
                        if !(start..end.min(ga)).all(|k| ctx[k].1) {
                            continue;
                        }
                        // Build sub-window: context[start..ga] ++ samf[0..end-ga]
                        let samf_count = end - ga;
                        let mut window: Vec<[u16; 3]> = (start..ga).map(|k| ctx[k].0).collect();
                        window.extend_from_slice(&samf[..samf_count]);
                        if let Some(repl) =
                            compress_curated_lmdb(&window, n, env, curated_shard_dbs, shard_dbs)
                        {
                            // Reject if the SAMF gates appear verbatim in the replacement —
                            // that means the compressor only touched the context and left the
                            // SAMF unhidden.
                            let samf_slice = &samf[..samf_count];
                            let samf_hidden = repl.len() < samf_count
                                || !repl.windows(samf_count).any(|w| w == samf_slice);
                            if samf_hidden {
                                found = Some((start, samf_count, repl));
                                break 'outer;
                            }
                        }
                    }
                }

                if let Some((start, samf_used, repl)) = found {
                    winner = Some((neg_type, samf, start, samf_used, repl));
                    break;
                }
            }

            match winner {
                None => {
                    SAMF_COMPRESSIONS_FAILED.fetch_add(1, Ordering::Relaxed);
                    false
                }
                Some((neg_type, samf, start, samf_used, repl)) => {
                    let samf_swap = (swap_lo, swap_hi, neg_type);
                    // Emit context gates before the window start. Only the collision pair
                    // (ctx[0], ctx[1]) was verified clean; ctx[2..start] may carry pending
                    // control negations (the window-cleanliness check only covers gates
                    // inside [start..end]). Flush those negations first, exactly like the
                    // normal path below, or the gate would read un-negated control values.
                    for k in 0..start {
                        let [_, gc1, gc2] = ctx[k].0;
                        if negation_mask[gc1 as usize] == 1 {
                            output.extend_from_slice(&Transpositions::gen_gates_not(n, gc1));
                            negation_mask[gc1 as usize] = 0;
                        }
                        if negation_mask[gc2 as usize] == 1 {
                            output.extend_from_slice(&Transpositions::gen_gates_not(n, gc2));
                            negation_mask[gc2 as usize] = 0;
                        }
                        output.push(ctx[k].0);
                    }
                    output.extend_from_slice(&repl);
                    output.extend_from_slice(&samf[samf_used..]);
                    compressions += 1;
                    SAMF_COMPRESSIONS_MADE.fetch_add(1, Ordering::Relaxed);

                    t_list.transpositions.push(samf_swap);
                    apply_neg_to_mask(
                        &mut negation_mask,
                        swap_lo as usize,
                        swap_hi as usize,
                        neg_type,
                    );

                    i += ga; // advance past all context gates
                    true
                }
            }
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

// Standalone shuffled shooting game: run the core, then undo its accumulated SAMFs.
pub fn shuffled_shooting_game(
    circuit: &mut CircuitSeq,
    n: usize,
    env: &Environment,
    curated_shard_dbs: &[Database],
    shard_dbs: &[Database],
    gates_ahead: usize,
    type_attempts: usize,
) -> usize {
    let (mut output, t_list, negation_mask, compressions) = shuffled_shooting_game_core(
        &circuit.gates,
        n,
        env,
        curated_shard_dbs,
        shard_dbs,
        gates_ahead,
        type_attempts,
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
    gates_ahead: usize,
    type_attempts: usize,
    env: &Environment,
    curated_shard_dbs: &[Database],
    shard_dbs: &[Database],
) -> (Vec<[u16; 3]>, Transpositions, Vec<u8>, usize) {
    let (out_a, t_a, neg_a, compressions) = shuffled_shooting_game_core(
        input,
        n,
        env,
        curated_shard_dbs,
        shard_dbs,
        gates_ahead,
        type_attempts,
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
    gates_ahead: usize,
    type_attempts: usize,
    env: &Environment,
    curated_shard_dbs: &[Database],
    shard_dbs: &[Database],
) -> usize {
    let (mut out, t_round, neg_round, compressions) = shuffled_shoot_then_samf_core(
        &circuit.gates,
        n,
        m,
        x,
        gates_ahead,
        type_attempts,
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
    use super::{REVERSED_SAMF, Transpositions, neg_flips, random_neg_type};
    use crate::circuit::circuit::CircuitSeq;

    #[test]
    fn random_neg_type_emits_reversed_when_enabled() {
        let mut rng = rand::rng();
        let mut seen_reversed = false;
        let mut max = 0u16;
        for _ in 0..5000 {
            let t = random_neg_type(&mut rng);
            assert!(t <= 5, "neg type out of range: {}", t);
            max = max.max(t);
            if t >= 4 {
                seen_reversed = true;
            }
        }
        // Guard against the feature silently regressing to disabled.
        assert_eq!(
            seen_reversed, REVERSED_SAMF,
            "REVERSED_SAMF wiring mismatch"
        );
        if !REVERSED_SAMF {
            assert!(max <= 3);
        }
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
        // Reversed variants share the opposite forward type's net effect.
        assert_eq!(neg_flips(4), neg_flips(2), "rev-N1 must match N2");
        assert_eq!(neg_flips(5), neg_flips(1), "rev-N2 must match N1");
    }

    #[test]
    fn gen_gates_swap_logical_op_all_types() {
        // n=3 exercises only 3-wire pools; n=4 also exercises 4-wire pools. Many
        // iterations cover the random pool + ancilla choices.
        for &(n, a, b) in &[(3usize, 1u16, 2u16), (4, 1, 3), (4, 0, 2)] {
            for neg in 0u16..=5 {
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

    #[test]
    fn reversed_gadget_is_inverse_of_forward() {
        // gen(4) = reverse(forward N1), so reversing it again recovers a forward-N1
        // net op; symmetrically gen(5) reversed recovers forward N2.
        for &(rev_type, fwd_type) in &[(4u16, 1u16), (5u16, 2u16)] {
            for _ in 0..300 {
                let mut gates = Transpositions::gen_gates_swap(4, (1, 3, rev_type));
                gates.reverse();
                for xa in 0..2 {
                    for xb in 0..2 {
                        assert_eq!(
                            logical_op(&gates, 4, 1, 3, xa, xb),
                            expected(fwd_type, xa, xb),
                            "reverse(gen({})) should equal forward {}",
                            rev_type,
                            fwd_type
                        );
                    }
                }
            }
        }
    }
}
