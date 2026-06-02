// For adding wire shuffles and bit flips
use crate::circuit::{Permutation, circuit::CircuitSeq};
use lmdb::{Database, Environment};
use rand::Rng;
use rand::seq::IndexedRandom;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

pub static SAMF_COMPRESSIONS_MADE: AtomicUsize = AtomicUsize::new(0);
pub static SAMF_COMPRESSIONS_FAILED: AtomicUsize = AtomicUsize::new(0);

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
            let negation_type = rng.random_range(0u16..=3);
            let j = rng.random_range(0..=i);
            if i == j {
                continue;
            }
            transpositions.push((j, i, negation_type));
            let temp = negation_mask[j as usize];
            negation_mask[j as usize] = negation_mask[i as usize];
            negation_mask[i as usize] = temp;
            if negation_type == 1 || negation_type == 3 {
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
            if negation_type == 1 || negation_type == 3 {
                negation_mask[j as usize] ^= 1;
            }
            if negation_type == 2 || negation_type == 3 {
                negation_mask[i as usize] ^= 1;
            }
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

pub fn insert_wire_shuffles_knuth(circuit: &mut CircuitSeq, n: usize) {
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

    let mut leftover_nots: Vec<u16> = Vec::new();
    for (i, val) in negation_mask.into_iter().enumerate() {
        if val == 1 {
            if let Some(swaps) = wire_transpositions.get(&(i as u16)) {
                let &(swap_idx, pos) = swaps;
                let curr_neg_type = t.transpositions[swap_idx].2;
                if pos > 1 || curr_neg_type > 3 {
                    panic!("Invalid pos or curr_neg_type");
                }
                t.transpositions[swap_idx].2 = TRANSITION[pos][curr_neg_type as usize] as u16;
            } else {
                // This wire is a fixed point of the permutation, so it has no
                // transposition to fold the residual negation into. Undo it with
                // an explicit NOT, emitted after the permutation is restored.
                leftover_nots.push(i as u16);
            }
        }
    }

    let mut c = t.to_circuit(n).gates;
    c.reverse();
    gates.extend_from_slice(&c);
    for wire in leftover_nots {
        gates.extend_from_slice(&Transpositions::gen_gates_not(n, wire));
    }
    circuit.gates = gates;
    println!("Complete. Ending len: {} gates", circuit.gates.len());
}

pub fn insert_wire_shuffles_simple(circuit: &mut CircuitSeq, n: usize) {
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

    let mut leftover_nots: Vec<u16> = Vec::new();
    for (i, val) in negation_mask.into_iter().enumerate() {
        if val == 1 {
            if let Some(swaps) = wire_transpositions.get(&(i as u16)) {
                let &(swap_idx, pos) = swaps;
                let curr_neg_type = t.transpositions[swap_idx].2;
                if pos > 1 || curr_neg_type > 3 {
                    panic!("Invalid pos or curr_neg_type");
                }
                t.transpositions[swap_idx].2 = TRANSITION[pos][curr_neg_type as usize] as u16;
            } else {
                // This wire is a fixed point of the permutation, so it has no
                // transposition to fold the residual negation into. Undo it with
                // an explicit NOT, emitted after the permutation is restored.
                leftover_nots.push(i as u16);
            }
        }
    }

    let mut c = t.to_circuit(n).gates;
    c.reverse();
    gates.extend_from_slice(&c);
    for wire in leftover_nots {
        gates.extend_from_slice(&Transpositions::gen_gates_not(n, wire));
    }
    circuit.gates = gates;
    println!("Complete. Ending len: {} gates", circuit.gates.len());
}

// Insert 2 shuffles are the beginning and end, and then an additional x number of shuffles
pub fn insert_wire_shuffles_x(circuit: &mut CircuitSeq, n: usize, x: usize) {
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

    let mut leftover_nots: Vec<u16> = Vec::new();
    for (i, val) in negation_mask.into_iter().enumerate() {
        if val == 1 {
            if let Some(swaps) = wire_transpositions.get(&(i as u16)) {
                let &(swap_idx, pos) = swaps;
                let curr_neg_type = t.transpositions[swap_idx].2;
                if pos > 1 || curr_neg_type > 3 {
                    panic!("Invalid pos or curr_neg_type");
                }
                t.transpositions[swap_idx].2 = TRANSITION[pos][curr_neg_type as usize] as u16;
            } else {
                // This wire is a fixed point of the permutation, so it has no
                // transposition to fold the residual negation into. Undo it with
                // an explicit NOT, emitted after the permutation is restored.
                leftover_nots.push(i as u16);
            }
        }
    }

    let mut c = t.to_circuit(n).gates;
    c.reverse();
    gates.extend_from_slice(&c);
    for wire in leftover_nots {
        gates.extend_from_slice(&Transpositions::gen_gates_not(n, wire));
    }
    circuit.gates = gates;
    println!("Complete. Ending len: {} gates", circuit.gates.len());
}

// Insert m samf between each gate
pub fn insert_wire_m_samfs_every_x(circuit: &mut CircuitSeq, n: usize, m: usize, x: usize) {
    let n = n;
    println!("Inserting {} samfs between each gate", m);
    println!("Starting len: {} gates", circuit.gates.len());
    let mut t_list: Transpositions = Transpositions {
        transpositions: Vec::new(),
    };
    let mut gates: Vec<[u16; 3]> = Vec::new();
    let mut negation_mask = vec![0u8; n];

    for (i, gate) in circuit.gates.iter().enumerate() {
        if i % x == 0 {
            let t = Transpositions::gen_random_simple(n, m, &mut negation_mask);
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

    let mut leftover_nots: Vec<u16> = Vec::new();
    for (i, val) in negation_mask.into_iter().enumerate() {
        if val == 1 {
            if let Some(swaps) = wire_transpositions.get(&(i as u16)) {
                let &(swap_idx, pos) = swaps;
                let curr_neg_type = t.transpositions[swap_idx].2;
                if pos > 1 || curr_neg_type > 3 {
                    panic!("Invalid pos or curr_neg_type");
                }
                t.transpositions[swap_idx].2 = TRANSITION[pos][curr_neg_type as usize] as u16;
            } else {
                // This wire is a fixed point of the permutation, so it has no
                // transposition to fold the residual negation into. Undo it with
                // an explicit NOT, emitted after the permutation is restored.
                leftover_nots.push(i as u16);
            }
        }
    }

    let mut c = t.to_circuit(n).gates;
    c.reverse();
    gates.extend_from_slice(&c);
    for wire in leftover_nots {
        gates.extend_from_slice(&Transpositions::gen_gates_not(n, wire));
    }
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
// When gates_ahead > 2, first tries a wider window of gates_ahead gates + samf[0..3]. Falls back
// to the 2-gate collision pair + samf[0..3] if the wider lookup misses.
//
// If a replacement is found, output it followed by the remaining SAMF gates. Future gates are
// relabeled by the SAMF's wire permutation, and the accumulated permutation is undone at the end.
pub fn shuffled_shooting_game(
    circuit: &mut CircuitSeq,
    n: usize,
    env: &Environment,
    curated_shard_dbs: &[Database],
    shard_dbs: &[Database],
    gates_ahead: usize,
) -> usize {
    use crate::replace::pairs::compress_curated_lmdb;

    let mut rng = rand::rng();
    let mut output: Vec<[u16; 3]> = Vec::new();
    let mut t_list = Transpositions {
        transpositions: Vec::new(),
    };
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

            // Generate SAMF.
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
            let neg_type: u16 = rng.random_range(0..4);
            let samf_swap = (swap_lo, swap_hi, neg_type);
            let samf = Transpositions::gen_gates_swap(n, samf_swap);
            if samf.len() < 3 {
                break 'try_replace false;
            }

            // Build all available context gates (up to gates_ahead) with clean flags.
            // Positions 0 and 1 (collision pair controls) are already verified clean.
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

            match found {
                None => {
                    SAMF_COMPRESSIONS_FAILED.fetch_add(1, Ordering::Relaxed);
                    false
                }
                Some((start, samf_used, repl)) => {
                    // Output any context gates before the window start normally.
                    // These are clean (verified above), so no NOT corrections needed.
                    for k in 0..start {
                        output.push(ctx[k].0);
                    }
                    output.extend_from_slice(&repl);
                    output.extend_from_slice(&samf[samf_used..]);
                    compressions += 1;
                    SAMF_COMPRESSIONS_MADE.fetch_add(1, Ordering::Relaxed);

                    t_list.transpositions.push(samf_swap);
                    let tmp = negation_mask[swap_lo as usize];
                    negation_mask[swap_lo as usize] = negation_mask[swap_hi as usize];
                    negation_mask[swap_hi as usize] = tmp;
                    if neg_type == 1 || neg_type == 3 {
                        negation_mask[swap_lo as usize] ^= 1;
                    }
                    if neg_type == 2 || neg_type == 3 {
                        negation_mask[swap_hi as usize] ^= 1;
                    }

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
    let mut leftover_nots: Vec<u16> = Vec::new();
    for (wire, &val) in negation_mask.iter().enumerate() {
        if val == 1 {
            if let Some(&(swap_idx, pos)) = wire_positions.get(&(wire as u16)) {
                let curr = t.transpositions[swap_idx].2;
                t.transpositions[swap_idx].2 = TRANSITION[pos][curr as usize] as u16;
            } else {
                // Wire is a fixed point of the permutation, so it has no
                // transposition to fold the residual negation into. Undo it with
                // an explicit NOT after the permutation is restored.
                leftover_nots.push(wire as u16);
            }
        }
    }
    let mut undo = t.to_circuit(n).gates;
    undo.reverse();
    output.extend_from_slice(&undo);
    for w in leftover_nots {
        output.extend_from_slice(&Transpositions::gen_gates_not(n, w));
    }

    circuit.gates = output;
    compressions
}
