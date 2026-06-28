// For adding wire shuffles and bit flips
use crate::circuit::{Permutation, circuit::CircuitSeq};
use lmdb::{Database, Environment};
use rand::Rng;
use rand::seq::IndexedRandom;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};

pub static SAMF_INSERTIONS_MADE: AtomicUsize = AtomicUsize::new(0);
pub static SAMF_COMPRESSIONS_MADE: AtomicUsize = AtomicUsize::new(0);
pub static SAMF_COMPRESSIONS_FAILED: AtomicUsize = AtomicUsize::new(0);
// Curated expansions performed at collisions in the shuffled shooting game.
pub static CURATED_REPLACEMENTS_MADE: AtomicUsize = AtomicUsize::new(0);
// #10/Stage F: expansions of an "unclean" window (>=1 control carried a pending NOT, absorbed
// into the curated lookup instead of emitting a NOT gadget).
pub static UNCLEAN_EXPANSIONS: AtomicUsize = AtomicUsize::new(0);
// Diagnostic counters for why a curated expansion did not hide a SAMF.
pub static SAMF_HIDE_ELIGIBLE_EXPANSIONS: AtomicUsize = AtomicUsize::new(0);
pub static SAMF_HIDE_SKIPPED_MATERIALIZED: AtomicUsize = AtomicUsize::new(0);
pub static SAMF_HIDE_ATTEMPTS: AtomicUsize = AtomicUsize::new(0);
pub static SAMF_HIDE_LOOKUP_MISSES: AtomicUsize = AtomicUsize::new(0);
pub static SAMF_HIDE_REJECTED_EXPOSED: AtomicUsize = AtomicUsize::new(0);
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
    tags: &mut Vec<u32>,
) {
    use crate::replace::pairs::{compress_curated_lmdb, find_any_replacement_lmdb};
    let track = !tags.is_empty();
    // No DBs to consult -> append verbatim (matches the old behaviour and avoids
    // opening a read txn on a possibly-empty env).
    if curated_shard_dbs.is_empty() && shard_dbs.is_empty() {
        gates.extend_from_slice(samf);
        if track {
            // appended unsamf gates are new: generation = local-context median + 1 (never gen 0)
            let k = tags.len().min(3);
            let nt = crate::replace::replace::new_gate_tag(&tags[tags.len() - k..]);
            tags.resize(gates.len(), nt);
        }
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
    if track {
        // the window [ctx_start..] is replaced (curated) or has the unsamf appended after it;
        // new gates get generation = floor(median(replaced window)) + 1.
        let nt = crate::replace::replace::new_gate_tag(&tags[ctx_start..]);
        tags.truncate(ctx_start);
        tags.resize(gates.len(), nt);
    }
}

// Undo the accumulated wire permutation + pending negations described by `t_list` and
// `negation_mask`, appending the inverse SAMFs (and any leftover NOTs for permutation fixed
// points) to `output`. Each gadget is fused with the trailing emitted gates via the curated
// DB (integrate_samf_compressed). This is the shared final "unsamf" step for every shuffle
// function. `negation_mask` must be indexed in the same (current) wire space that `t_list`'s
// net permutation maps the original wires into.
/// DEBUG helper: does `out` equal `wg` up to the permutation `far_perm` (any natural direction)?
/// Deterministic sample inputs + functional permutation compare — uses NO rand::rng, so calling it
/// does not perturb the shooting trajectory (the Stage C bug is RNG-trajectory-sensitive). Accepts
/// any of the candidate perm conventions; a genuine break matches none of them.
fn stagec_pass_ok(out: &[[u16; 3]], wg: &[[u16; 3]], far_perm: &[u16], n: usize) -> bool {
    use crate::circuit::circuit::{Gate, U1024};
    let mut inv = vec![0u16; n];
    for w in 0..n {
        inv[far_perm[w] as usize] = w as u16;
    }
    let mask = (U1024::one() << n) - U1024::one();
    let permute = |s: U1024, p: &[u16]| -> U1024 {
        let mut r = U1024::zero();
        for i in 0..n {
            if (s >> p[i] as usize) & U1024::one() == U1024::one() {
                r = r | (U1024::one() << i);
            }
        }
        r
    };
    let out_v = out.to_vec();
    let wg_v = wg.to_vec();
    let mut cand = [true; 5];
    let mut sm: u64 = 0x243F6A8885A308D3;
    for _ in 0..128 {
        let mut bytes = [0u8; 128];
        for chunk in bytes.chunks_mut(8) {
            sm = sm.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            chunk.copy_from_slice(&z.to_le_bytes());
        }
        let x = U1024::from_little_endian(&bytes) & mask;
        let o = Gate::evaluate_index_list_1024(x, &out_v) & mask;
        let w = Gate::evaluate_index_list_1024(x, &wg_v) & mask;
        if cand[0] && o != w {
            cand[0] = false;
        }
        if cand[1] && permute(o, far_perm) != w {
            cand[1] = false;
        }
        if cand[2] && permute(o, &inv) != w {
            cand[2] = false;
        }
        if cand[3] && o != permute(w, far_perm) {
            cand[3] = false;
        }
        if cand[4] && o != permute(w, &inv) {
            cand[4] = false;
        }
        if !cand.iter().any(|&c| c) {
            return false;
        }
    }
    cand.iter().any(|&c| c)
}

pub fn apply_unsamf(
    output: &mut Vec<[u16; 3]>,
    t_list: &Transpositions,
    negation_mask: &[u8],
    n: usize,
    env: &Environment,
    curated_shard_dbs: &[Database],
    shard_dbs: &[Database],
    tags: &mut Vec<u32>,
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
        integrate_samf_compressed(output, &samf, n, env, curated_shard_dbs, shard_dbs, tags);
    }
    for w in leftover_nots {
        let not_gates = Transpositions::gen_gates_not(n, w);
        integrate_samf_compressed(output, &not_gates, n, env, curated_shard_dbs, shard_dbs, tags);
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
        &mut Vec::new(),
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
        &mut Vec::new(),
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
        &mut Vec::new(),
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
    input_tags: &[u32],
) -> (Vec<[u16; 3]>, Transpositions, Vec<u8>, Vec<u32>) {
    let track = !input_tags.is_empty();
    let mut t_list = Transpositions {
        transpositions: Vec::new(),
    };
    let mut gates: Vec<[u16; 3]> = Vec::new();
    let mut out_tags: Vec<u32> = Vec::new();
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
        if track {
            // Everything added this iteration before `g` (SAMF gadget + NOT corrections) is new:
            // generation = gate i's generation + 1 (one mixing layer deeper, never gen 0).
            // `g` is the relabeled original input gate i (keeps its tag/generation).
            out_tags.resize(
                gates.len() - 1,
                crate::replace::replace::new_gate_tag(std::slice::from_ref(&input_tags[i])),
            );
            out_tags.push(input_tags[i]);
        }
    }
    (gates, t_list, negation_mask, out_tags)
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
    let (mut gates, t_list, negation_mask, _itags) = insert_m_samfs_core(&circuit.gates, n, m, x, &[]);
    apply_unsamf(
        &mut gates,
        &t_list,
        &negation_mask,
        n,
        env,
        curated_shard_dbs,
        shard_dbs,
        &mut Vec::new(),
    );
    circuit.gates = gates;
    println!("Complete. Ending len: {} gates", circuit.gates.len());
}

fn gates_collide(g1: [u16; 3], g2: [u16; 3]) -> bool {
    g1[0] == g2[1] || g1[0] == g2[2] || g2[0] == g1[1] || g2[0] == g1[2]
}

fn relabel_gate(gate: [u16; 3], t_list: &Transpositions) -> [u16; 3] {
    [
        t_list.evaluate(gate[0]),
        t_list.evaluate(gate[1]),
        t_list.evaluate(gate[2]),
    ]
}

fn flush_relabelled_gate_controls(
    gate: [u16; 3],
    output: &mut Vec<[u16; 3]>,
    t_list: &Transpositions,
    negation_mask: &mut [u8],
    n: usize,
) {
    let [_, b, c] = relabel_gate(gate, t_list);
    if negation_mask[b as usize] == 1 {
        output.extend_from_slice(&Transpositions::gen_gates_not(n, b));
        negation_mask[b as usize] = 0;
    }
    if negation_mask[c as usize] == 1 {
        output.extend_from_slice(&Transpositions::gen_gates_not(n, c));
        negation_mask[c as usize] = 0;
    }
}

fn emit_relabelled_gate(
    gate: [u16; 3],
    output: &mut Vec<[u16; 3]>,
    t_list: &Transpositions,
    negation_mask: &mut [u8],
    n: usize,
) {
    flush_relabelled_gate_controls(gate, output, t_list, negation_mask, n);
    output.push(relabel_gate(gate, t_list));
}

// Remove the first remaining gate and shoot it right across every gate it commutes with.
// The passed gates are removed and returned in their original order. If a collider is found,
// it remains at the front of `remaining`; otherwise the shot gate reached the end.
fn shoot_gate_to_first_collision(
    remaining: &mut VecDeque<[u16; 3]>,
    t_list: &Transpositions,
    negation_mask: &[u8],
) -> Option<([u16; 3], Vec<[u16; 3]>, bool)> {
    let shot = remaining.pop_front()?;
    let relabelled = relabel_gate(shot, t_list);
    let [_, b, c] = relabelled;
    if negation_mask[b as usize] != 0 || negation_mask[c as usize] != 0 {
        // The control-correction NOT belongs immediately before this gate. It may collide with
        // gates that the shot itself commutes with, so a dirty-control shot must stay in place.
        return Some((shot, Vec::new(), false));
    }
    let (passed, collided) =
        shoot_materialized_gate_to_first_collision(relabelled, remaining, t_list, negation_mask);
    Some((shot, passed, collided))
}

// Shoot a gate that is already in the current physical wire space. Gates in `remaining` still
// belong to the original logical wire space and are relabeled only for the collision test.
fn shoot_materialized_gate_to_first_collision(
    shot: [u16; 3],
    remaining: &mut VecDeque<[u16; 3]>,
    t_list: &Transpositions,
    negation_mask: &[u8],
) -> (Vec<[u16; 3]>, bool) {
    let mut passed = Vec::new();

    while let Some(&next) = remaining.front() {
        let relabelled = relabel_gate(next, t_list);
        let [_, b, c] = relabelled;
        if negation_mask[b as usize] != 0 || negation_mask[c as usize] != 0 {
            return (passed, false);
        }
        if gates_collide(shot, relabelled) {
            return (passed, true);
        }
        passed.push(remaining.pop_front().unwrap());
    }

    (passed, false)
}

// Shoot the first remaining gate right until its first collision. At that collision, make a
// curated EXPANSION of a window of up to `gates_ahead_expand` gates anchored at the shot gate
// and collider, then try to hide a single SAMF in the expansion's tail.
//
// Gates passed by the shot gate are emitted before the collision window. The expansion window
// shrinks by one gate on each curated-DB miss (gates_ahead_expand .. 2, down to the shot/collider
// pair); a window is eligible only if every control wire in it is clean.
//
// The SAMF-hiding window is the last `gates_ahead_samf` gates ending at the expansion's tail —
// reaching back into the already-emitted output when the expansion is shorter than
// `gates_ahead_samf` — followed by the first 3 gates of the SAMF. On a successful hide the
// remaining SAMF gates are emitted after the replacement except for the final gate, which becomes
// the next shot.
//
// `type_attempts` controls how many DISTINCT SAMF gate (negation) types are tried per collision
// before giving up: each attempt samples a not-yet-tried type (without replacement) and one
// random hardcoded SAMF of that type. The first type that yields a hiding window wins;
// `type_attempts == 1` is the original single-try behaviour.
//
// If a replacement is found, its final gate becomes the next shot. When a SAMF is successfully
// hidden, the final gate of the inserted SAMF becomes the next shot instead. Future untouched
// gates are relabeled by the SAMF's wire permutation, and the accumulated permutation is undone
// at the end.
fn shuffled_shooting_game_core(
    input: &[[u16; 3]],
    n: usize,
    env: &Environment,
    curated_shard_dbs: &[Database],
    shard_dbs: &[Database],
    gates_ahead_expand: usize,
    gates_ahead_samf: usize,
    type_attempts: usize,
    input_tags: &[u32],
    // Stage B bounded-pass controls (defaults preserve the full-sweep behavior):
    //   start_at: where to begin (None = random index, as before);
    //   max_replacements: stop after this many successful replacements (0 = unlimited);
    //   stop_on_unreplaceable: end the pass at the first collision with no curated replacement.
    start_at: Option<usize>,
    max_replacements: usize,
    stop_on_unreplaceable: bool,
) -> (Vec<[u16; 3]>, Transpositions, Vec<u8>, usize, Vec<u32>) {
    use crate::replace::pairs::{compress_curated_lmdb, expand_curated_lmdb_neg};
    use crate::replace::replace::TAG_NEW;

    // Survivor tracking: `input_tags` (non-empty when enabled) carries one origin id per input
    // gate. We mirror every push/pop of `output`/`remaining` on `output_tags`/`remaining_tags`.
    let track = !input_tags.is_empty();

    let mut rng = rand::rng();
    let mut output: Vec<[u16; 3]> = Vec::new();
    let mut output_tags: Vec<u32> = Vec::new();
    let mut t_list = Transpositions {
        transpositions: Vec::new(),
    };
    let mut negation_mask = vec![0u8; n];
    let mut compressions: usize = 0;

    let start = match start_at {
        Some(s) => s.min(input.len()),
        None if input.is_empty() => 0,
        None => rng.random_range(0..input.len()),
    };
    let mut rep_count = 0usize;
    output.extend_from_slice(&input[..start]);
    let mut remaining: VecDeque<[u16; 3]> = input[start..].iter().copied().collect();
    let mut remaining_tags: VecDeque<u32> = if track {
        output_tags.extend_from_slice(&input_tags[..start]);
        input_tags[start..].iter().copied().collect()
    } else {
        VecDeque::new()
    };
    // Replacement gates are already expressed in the current physical wire space. Keep the
    // replacement tail separate from `remaining`, whose gates still require SAMF relabeling.
    let mut materialized_shot: Option<[u16; 3]> = None;
    // Generation/tag of the carried materialized shot (gen mode); irrelevant in survivor mode.
    let mut materialized_shot_tag: u32 = TAG_NEW;

    while materialized_shot.is_some() || !remaining.is_empty() {
        let (shot, shot_is_materialized, mut passed, mut has_collision, shot_tag, mut passed_tags) =
            if let Some(shot) = materialized_shot.take() {
                let (passed, collided) = shoot_materialized_gate_to_first_collision(
                    shot,
                    &mut remaining,
                    &t_list,
                    &negation_mask,
                );
                // Materialized shot is a gate born during mixing (TAG_NEW). `passed` came off
                // the front of `remaining`, so pop their tags in order.
                let passed_tags: Vec<u32> = if track {
                    (0..passed.len())
                        .map(|_| remaining_tags.pop_front().unwrap())
                        .collect()
                } else {
                    Vec::new()
                };
                (shot, true, passed, collided, materialized_shot_tag, passed_tags)
            } else {
                let (shot, passed, collided) =
                    shoot_gate_to_first_collision(&mut remaining, &t_list, &negation_mask).unwrap();
                // The shot was popped first, then the `passed` gates.
                let (shot_tag, passed_tags) = if track {
                    let st = remaining_tags.pop_front().unwrap();
                    let pt: Vec<u32> = (0..passed.len())
                        .map(|_| remaining_tags.pop_front().unwrap())
                        .collect();
                    (st, pt)
                } else {
                    (TAG_NEW, Vec::new())
                };
                (shot, false, passed, collided, shot_tag, passed_tags)
            };

        // A control correction logically precedes an untouched input gate. Emit it before moving
        // the shot past commuting gates so the correction is not reordered across a dependent
        // gate. A materialized replacement tail has already been emitted in the current wire
        // space and must not be corrected or relabeled again.
        if !shot_is_materialized {
            flush_relabelled_gate_controls(shot, &mut output, &t_list, &mut negation_mask, n);
            if track {
                // emitted NOT-corrections are new gates: generation = shot's generation + 1
                output_tags.resize(
                    output.len(),
                    crate::replace::replace::new_gate_tag(std::slice::from_ref(&shot_tag)),
                );
            }
        }
        // Change 2: forced pseudo-collision (gen mode). If the shot found no real collision but
        // commuted past at least one gate (those gates are necessarily clean — the walk stops at a
        // dirty control), do NOT give up. Push the commuted gates back and treat the FIRST one as a
        // forced collider so the window logic tries to replace [shot, first-gate] (and reachable
        // extensions) from the DB. The shot genuinely commutes with all of them, so: on a DB hit
        // the replacement sits at the original front (passed[1..] follow, unmoved); on a miss the
        // shot is emitted ahead of them via the no-replacement path. Both are equivalence-safe.
        let forced_collision =
            !has_collision && crate::replace::replace::gen_mode() && !passed.is_empty();
        if forced_collision {
            for g in passed.drain(..).rev() {
                remaining.push_front(g);
            }
            if track {
                for t in passed_tags.drain(..).rev() {
                    remaining_tags.push_front(t);
                }
            }
            has_collision = true;
            crate::replace::replace::FORCED_COLLISIONS
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        for (i, gate) in passed.into_iter().enumerate() {
            emit_relabelled_gate(gate, &mut output, &t_list, &mut negation_mask, n);
            if track {
                // emit_relabelled_gate appends [NOT-corrections..., the gate]; the gate is last.
                // The NOT-corrections are new gates: generation = this gate's generation + 1.
                output_tags.resize(
                    output.len() - 1,
                    crate::replace::replace::new_gate_tag(std::slice::from_ref(&passed_tags[i])),
                );
                output_tags.push(passed_tags[i]);
            }
        }

        // At the first collision, expand a window anchored at [shot, collider]. The expansion is
        // computed once; if no SAMF can be hidden we keep it verbatim.
        let replaced = 'try_replace: {
            if !has_collision {
                break 'try_replace false;
            }

            // 1) Curated expansion of the largest clean window anchored at the colliding pair: an
            //    equivalent, longer circuit. The first gate is the shot gate; the collider and
            //    any additional context remain at the front of `remaining`. The window shrinks
            //    from its far end
            //    (gates_ahead_expand .. 2) until a curated expansion is found; a window is eligible
            //    only if every control wire in it is clean (no pending negation correction), so the
            //    expansion stays equivalence-safe. `consumed` includes the shot gate.
            // Outgoing-window selection. `consumed_indices` are positions in `remaining`
            // (collider = 0) that the window consumes; the window gates are
            // [shot] ++ remaining[consumed_indices] (relabeled into the current wire space).
            const OUTGOING_LOOKAHEAD: usize = 8; // how far past the collider to look for #11 extras
            const MAX_EXPAND_ATTEMPTS: usize = 16; // cap curated lookups per collision (#11)
            const FEATURE_CTX: usize = 64; // bounded context for global fanout/leeway features
            let (consumed_indices, window, expansion, picked_negated): (
                Vec<usize>,
                Vec<[u16; 3]>,
                Vec<[u16; 3]>,
                Vec<u16>,
            ) = {
                use crate::circuit::circuit::Gate;
                let shot_rg = if shot_is_materialized {
                    shot
                } else {
                    relabel_gate(shot, &t_list)
                };
                // A gate is "clean" if neither control carries a pending negation correction.
                let clean = |g: &[u16; 3]| {
                    negation_mask[g[1] as usize] == 0 && negation_mask[g[2] as usize] == 0
                };
                // Relabel enough of the front of `remaining` to cover both the #11 lookahead and
                // the legacy contiguous window (gates_ahead_expand).
                let look = OUTGOING_LOOKAHEAD
                    .max(gates_ahead_expand)
                    .min(remaining.len());
                let rg: Vec<[u16; 3]> = remaining
                    .iter()
                    .take(look)
                    .map(|&g| relabel_gate(g, &t_list))
                    .collect();

                // Candidate windows as (consumed_indices, window_gates).
                let mut candidates: Vec<(Vec<usize>, Vec<[u16; 3]>)> = Vec::new();
                if crate::replace::replace::gen_mode() {
                    // #11: size 2 = [shot, collider]; size 3/4 add gates after the collider that
                    // reach the front by commutation (rg[j] commutes past every gate before it).
                    // #10/Stage F: "unclean" windows (a control carries a pending NOT) are NOT
                    // skipped here — the pending NOT is absorbed into the curated lookup below, so
                    // the clean() gate is gone. Reachability is purely about commutation.
                    if !rg.is_empty() {
                        candidates.push((vec![0], vec![shot_rg, rg[0]]));
                        let reach1: Vec<usize> = (1..rg.len())
                            .filter(|&j| (1..j).all(|m| !Gate::collides_index(&rg[j], &rg[m])))
                            .collect();
                        for &j in &reach1 {
                            candidates.push((vec![0, j], vec![shot_rg, rg[0], rg[j]]));
                        }
                        for &j in &reach1 {
                            for k in (j + 1)..rg.len() {
                                if (1..k)
                                    .filter(|&m| m != j)
                                    .all(|m| !Gate::collides_index(&rg[k], &rg[m]))
                                {
                                    candidates
                                        .push((vec![0, j, k], vec![shot_rg, rg[0], rg[j], rg[k]]));
                                    // size 5 (lookahead 3): a 4th post-collider gate that also
                                    // reaches the front by commutation (past everything except j, k).
                                    for l in (k + 1)..rg.len() {
                                        if (1..l)
                                            .filter(|&m| m != j && m != k)
                                            .all(|m| !Gate::collides_index(&rg[l], &rg[m]))
                                        {
                                            candidates.push((
                                                vec![0, j, k, l],
                                                vec![shot_rg, rg[0], rg[j], rg[k], rg[l]],
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    // Legacy contiguous shrink: windows [shot, rg[0..k-1]] for k = 2..=max_window.
                    let max_window = gates_ahead_expand.max(2).min(remaining.len() + 1);
                    for k in 2..=max_window {
                        if !clean(&shot_rg) {
                            break;
                        }
                        if (0..k - 1).any(|m| m >= rg.len() || !clean(&rg[m])) {
                            continue;
                        }
                        let mut w = Vec::with_capacity(k);
                        w.push(shot_rg);
                        w.extend(rg[..k - 1].iter().copied());
                        candidates.push(((0..k - 1).collect(), w));
                    }
                }
                if candidates.is_empty() {
                    break 'try_replace false;
                }

                // Ranked order to try: gen mode ranks by global features (#11); legacy tries the
                // largest contiguous window first (candidates built ascending k -> reverse).
                let order: Vec<usize> = if crate::replace::replace::gen_mode() {
                    let left = &output[output.len().saturating_sub(FEATURE_CTX)..];
                    let feats: Vec<crate::replace::ranking::CandFeatures> = candidates
                        .iter()
                        .map(|(idxs, win)| {
                            let skip: std::collections::HashSet<usize> =
                                idxs.iter().copied().collect();
                            let right: Vec<[u16; 3]> = (0..rg.len())
                                .filter(|m| !skip.contains(m))
                                .take(FEATURE_CTX)
                                .map(|m| rg[m])
                                .collect();
                            crate::replace::replace::cand_features(win, left, &right)
                        })
                        .collect();
                    // Try the LARGEST window first (lookahead 3 -> size 5), shrinking to 2 on each
                    // curated-DB miss; the rank script breaks ties within a size (stable sort).
                    let mut ord = crate::replace::ranking::outgoing().order(&feats);
                    ord.sort_by(|&a, &b| candidates[b].1.len().cmp(&candidates[a].1.len()));
                    ord
                } else {
                    (0..candidates.len()).rev().collect()
                };

                // #10/Stage F: dedup ALL wires of a window that carry a pending NOT — targets as
                // well as controls. The flush mechanism only NOTs controls (a target's negation
                // commutes through its own XOR write), but the curated EXPANSION is a different
                // circuit that may read that wire as a control, so its entering negation must be
                // absorbed too. substitute_input_negation (x_w -> x_w+1) correctly flips any input
                // variable, target or control, accounting for the wire's flipped entering value.
                let negated_controls = |win: &[[u16; 3]]| -> Vec<u16> {
                    let mut s: Vec<u16> = Vec::new();
                    for g in win {
                        for &c in &[g[0], g[1], g[2]] {
                            if negation_mask[c as usize] == 1 && !s.contains(&c) {
                                s.push(c);
                            }
                        }
                    }
                    s
                };
                // Try a curated expansion of each candidate in ranked order; first success wins.
                let gen_m = crate::replace::replace::gen_mode();
                // #10/Stage F: never absorb a negation on the SHOT (window[0]). The shot may be a
                // MATERIALIZED gate already emitted in physical space, which by design is not
                // negation-corrected (the pending negation applies to FUTURE relabeled gates, not to
                // it) and may also have shot past gates that read the wire. Absorbing its negation
                // would both mis-time the correction and cross commuting gates. So if any of the
                // shot's wires is negated, fall back (no unclean expansion of this window). The
                // unmoved collider/extras (window[1..], relabeled future gates) DO see the negation
                // and are safe to absorb.
                let shot_neg = gen_m
                    && {
                        let g = &shot_rg;
                        negation_mask[g[0] as usize] == 1
                            || negation_mask[g[1] as usize] == 1
                            || negation_mask[g[2] as usize] == 1
                    };
                // #10/Stage F is DISABLED by default: it has a latent correctness bug on the
                // clean-shot / unclean-collider absorption path that escaped verification (the n=6
                // exhaustive stress had 0 firings after the dirty-shot guard; sampled checks at
                // scale gave false passes, then run8000mg1 failed equality with #10 firing 9x).
                // Default behavior restores the verified pre-#10 rule: skip any window whose gates
                // carry a pending NOT on a CONTROL (a clean expansion E==window still correctly
                // preserves a deferred TARGET negation, which commutes through the window's XOR).
                // Set env ABSORB_NOTS=1 to re-enable #10 (for debugging only).
                let absorb_nots = gen_m && std::env::var("ABSORB_NOTS").is_ok();
                let mut picked: Option<(Vec<usize>, Vec<[u16; 3]>, Vec<[u16; 3]>, Vec<u16>)> = None;
                for &ci in order.iter().take(MAX_EXPAND_ATTEMPTS) {
                    let (idxs, win) = &candidates[ci];
                    if !absorb_nots {
                        // Pre-#10: window eligible only if every control wire is clean.
                        let dirty_control = win.iter().any(|g| {
                            negation_mask[g[1] as usize] == 1 || negation_mask[g[2] as usize] == 1
                        });
                        if dirty_control {
                            continue;
                        }
                    }
                    // gen-mode #10 (when enabled) absorbs all pending NOTs in the window.
                    let neg = if absorb_nots { negated_controls(win) } else { Vec::new() };
                    // A negated shot wire can't be safely absorbed (see shot_neg): skip if so.
                    if shot_neg && !neg.is_empty() {
                        continue;
                    }
                    if let Some(e) =
                        expand_curated_lmdb_neg(win, n, env, curated_shard_dbs, shard_dbs, &neg)
                    {
                        if e.len() >= 3 {
                            picked = Some((idxs.clone(), win.clone(), e, neg));
                            break;
                        }
                    }
                }
                match picked {
                    Some(x) => x,
                    None => break 'try_replace false, // no curated expansion -> normal path
                }
            };
            // VERIFY_DB_HITS (debug, env-gated, off by default): deterministically catch a
            // non-equivalent curated replacement the instant it is spliced. Only meaningful when no
            // NOT was absorbed (#10 off): then the expansion E must be functionally equal to the
            // window it replaces. Aborts at the exact site with the offending circuits.
            if picked_negated.is_empty() && crate::replace::replace::verify_db_hits() {
                let win_c = crate::circuit::circuit::CircuitSeq { gates: window.clone() };
                let exp_c = crate::circuit::circuit::CircuitSeq { gates: expansion.clone() };
                if win_c.probably_equal(&exp_c, n, 256).is_err() {
                    let mut used: std::collections::HashSet<u16> = std::collections::HashSet::new();
                    for g in &window {
                        used.insert(g[0]);
                        used.insert(g[1]);
                        used.insert(g[2]);
                    }
                    eprintln!(
                        "[VERIFY_DB_HITS] NON-EQUIVALENT curated replacement spliced! window={} gates, expansion={} gates, distinct_wires={}",
                        window.len(),
                        expansion.len(),
                        used.len()
                    );
                    eprintln!("  window:    {:?}", window);
                    eprintln!("  expansion: {:?}", expansion);
                    std::process::exit(7);
                }
            }
            // Window size including the shot gate (legacy `consumed` semantics).
            let consumed = window.len();
            // #10/Stage F: the expansion fully absorbed the pending NOTs on `picked_negated`. The
            // deferred NOT(w) sits right before this control-read, so the true local circuit is
            // [NOT(w)][window]; the curated lookup found E with E(state) = window(flip_w(state)),
            // i.e. E == NOT(w) then window. The substituted polynomial sets wire w's output to !x_w
            // EVEN when the window never writes w (identity poly x_w -> x_w+1), so after E every
            // negated control physically holds its true (NOT-applied) value. The pending NOT is
            // therefore consumed unconditionally -> clear the mask (mirrors the skip-path, which
            // flushes NOT(w) and clears it). Clearing only on "window writes w" would leave a
            // spurious pending NOT on pure-read controls and double-negate them downstream.
            if !picked_negated.is_empty() {
                UNCLEAN_EXPANSIONS.fetch_add(1, Ordering::Relaxed);
            }
            for &w in &picked_negated {
                negation_mask[w as usize] = 0;
            }
            // Generation/tag for the gates this collision adds: median of the consumed window
            // (shot gate + the consumed remaining gates by index) + 1.
            let event_tag = if track {
                let mut win: Vec<u32> = Vec::with_capacity(consumed);
                win.push(shot_tag);
                win.extend(consumed_indices.iter().map(|&idx| remaining_tags[idx]));
                crate::replace::replace::new_gate_tag(&win)
            } else {
                TAG_NEW
            };
            CURATED_REPLACEMENTS_MADE.fetch_add(1, Ordering::Relaxed);
            SAMF_HIDE_ELIGIBLE_EXPANSIONS.fetch_add(1, Ordering::Relaxed);
            // Distinct-wire counts of the consumed input window vs the expansion. `window` is the
            // chosen outgoing subcircuit, already relabeled into the current wire space.
            let distinct_wires = |gates: &[[u16; 3]]| {
                let mut seen = std::collections::HashSet::new();
                for g in gates {
                    seen.insert(g[0]);
                    seen.insert(g[1]);
                    seen.insert(g[2]);
                }
                seen.len()
            };
            let before_wires = distinct_wires(&window);
            crate::replace::replace::record_expansion(
                consumed,
                expansion.len(),
                before_wires,
                distinct_wires(&expansion),
            );

            // 2) Try to hide ONE SAMF ending at the expansion's tail. The context for the hide is
            //    the last `gates_ahead_samf` gates of (output ++ expansion) — reaching back into
            //    the already-emitted output when the expansion is shorter — followed by samf[0..3].
            //    Cone-aware mode samples multiple swap pairs and keeps the strongest hidden
            //    context; default mode keeps the historical first-success behavior.

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
            // (score, swap_lo, swap_hi, neg_type, samf, repl): the context window ++ samf[0..3] becomes `repl`, and
            // all but the final gate of samf[3..] is emitted after it.
            let mut tuck: Option<(f64, u16, u16, u16, Vec<[u16; 3]>, Vec<[u16; 3]>)> = None;
            let cone_aware = crate::replace::sat_score::sat_cone_aware_enabled();
            let pair_attempts = if cone_aware {
                crate::replace::sat_score::sat_hidden_samf_candidates()
            } else {
                1
            };
            'pairs: for pair_idx in 0..pair_attempts {
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

                for &neg_type in candidate_types.choose_multiple(&mut rng, type_attempts.max(1)) {
                    let samf = Transpositions::gen_gates_swap(n, (swap_lo, swap_hi, neg_type));
                    if samf.len() < 3 {
                        continue;
                    }
                    SAMF_HIDE_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
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
                        let samf_hidden =
                            repl.len() < 3 || !repl.windows(3).any(|w| w == samf_slice);
                        if samf_hidden {
                            let score = if cone_aware {
                                let sat_score = crate::replace::sat_score::score_subcircuit(
                                    &repl,
                                    n,
                                    crate::replace::sat_score::sat_score_seed()
                                        ^ ((pair_idx as u64) << 8)
                                        ^ neg_type as u64,
                                );
                                crate::replace::sat_score::compression_selection_score(&sat_score)
                            } else {
                                0.0
                            };
                            let replace_best = tuck
                                .as_ref()
                                .map(|(best_score, ..)| score > *best_score)
                                .unwrap_or(true);
                            if replace_best {
                                tuck = Some((score, swap_lo, swap_hi, neg_type, samf, repl));
                            }
                            if !cone_aware {
                                break 'pairs;
                            }
                        } else {
                            SAMF_HIDE_REJECTED_EXPOSED.fetch_add(1, Ordering::Relaxed);
                        }
                    } else {
                        SAMF_HIDE_LOOKUP_MISSES.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }

            let rec_pre_len = output.len();
            let mut rec_start = rec_pre_len;
            match tuck {
                Some((_score, swap_lo, swap_hi, neg_type, samf, repl)) => {
                    // The context window (output tail + expansion tail + samf[0..3]) becomes
                    // `repl`; everything before it is unchanged. Emit the SAMF suffix up to its
                    // final gate, then keep that final gate as the next materialized shot so the
                    // collision game continues from the inserted SAMF.
                    rec_start = out_keep;
                    output.truncate(out_keep);
                    output.extend_from_slice(&expansion[..exp_tail_start]);
                    let samf_suffix = &samf[3..];
                    output.extend_from_slice(&repl);
                    if let Some((last, prefix)) = samf_suffix.split_last() {
                        output.extend_from_slice(prefix);
                        materialized_shot = Some(*last);
                        materialized_shot_tag = event_tag;
                    }
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
                    // Keep the curated expansion, but leave its final gate as the next shot.
                    let (last, prefix) = expansion.split_last().unwrap();
                    output.extend_from_slice(prefix);
                    materialized_shot = Some(*last);
                    materialized_shot_tag = event_tag;
                    SAMF_COMPRESSIONS_FAILED.fetch_add(1, Ordering::Relaxed);
                }
            }

            if track {
                // The shot gate and the consumed remaining gates are replaced; everything emitted
                // into output[rec_start..] is freshly created. New gates get `event_tag` (TAG_NEW
                // in survivor mode, floor(median(window))+1 in gen mode). The consumed window tags
                // are dropped (by index, descending so earlier indices stay valid).
                output_tags.truncate(rec_start);
                output_tags.resize(output.len(), event_tag);
                for &idx in consumed_indices.iter().rev() {
                    remaining_tags.remove(idx);
                }
            }

            if crate::replace::replace::record_enabled() {
                let rec_end = output.len();
                let rs = rec_start.min(rec_end);
                let ws: Vec<u16> = output[rs..rec_end].iter().flatten().copied().collect();
                crate::replace::replace::record_replacement(
                    "expand",
                    crate::replace::replace::REC_PASS.load(std::sync::atomic::Ordering::Relaxed),
                    rs,
                    rec_end,
                    consumed,
                    &ws,
                );
            }

            // Remove the consumed gates from `remaining` by index (descending, so earlier indices
            // stay valid). For #11 these may be non-contiguous; the gates between them commuted
            // past the consumed extras and remain in order at the front.
            for &idx in consumed_indices.iter().rev() {
                remaining.remove(idx);
            }
            true
        };
        if !replaced {
            // No collision or no expansion: the shot gate stays immediately before its first
            // collider (or at the end if it passed every remaining gate).
            if shot_is_materialized {
                output.push(shot);
                if track {
                    // carried materialized gate keeps its own tag/generation
                    output_tags.push(shot_tag);
                }
            } else {
                emit_relabelled_gate(shot, &mut output, &t_list, &mut negation_mask, n);
                if track {
                    // NOT-corrections preceding the shot are new gates: generation = shot's + 1.
                    output_tags.resize(
                        output.len() - 1,
                        crate::replace::replace::new_gate_tag(std::slice::from_ref(&shot_tag)),
                    );
                    output_tags.push(shot_tag);
                }
            }
        }
        // Stage B bounded-pass termination: stop after `max_replacements` successful replacements,
        // or at the first collision that could not be replaced.
        if replaced {
            rep_count += 1;
        }
        let stop_now = (max_replacements > 0 && rep_count >= max_replacements)
            || (stop_on_unreplaceable && has_collision && !replaced);
        if stop_now {
            break;
        }
    }

    // A bounded pass may stop before draining `remaining`. Flush the carried materialized shot and
    // the rest of `remaining` (relabeled, with tags) so `output` is a complete circuit. This is a
    // no-op on normal (full-sweep) termination where both are already empty.
    if let Some(shot) = materialized_shot.take() {
        output.push(shot);
        if track {
            output_tags.push(materialized_shot_tag);
        }
    }
    while let Some(g) = remaining.pop_front() {
        emit_relabelled_gate(g, &mut output, &t_list, &mut negation_mask, n);
        if track {
            let gt = remaining_tags.pop_front().unwrap();
            output_tags.resize(
                output.len() - 1,
                crate::replace::replace::new_gate_tag(std::slice::from_ref(&gt)),
            );
            output_tags.push(gt);
        }
    }

    debug_assert!(!track || output_tags.len() == output.len());
    (output, t_list, negation_mask, compressions, output_tags)
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

    for pass in 0..passes {
        crate::replace::replace::REC_PASS.store(pass + 1, std::sync::atomic::Ordering::Relaxed);
        let (out, t_pass, neg_pass, compressions, _tags) = shuffled_shooting_game_core(
            &input_gates,
            n,
            env,
            curated_shard_dbs,
            shard_dbs,
            gates_ahead_expand,
            gates_ahead_samf,
            type_attempts,
            &[],
            None,
            0,
            false,
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
        &mut Vec::new(),
    );
    circuit.gates = output;
    compressions
}

// Core of shuffled_shoot_then_samf: each shooting pass runs one collision game followed by
// one per-gate SAMF insertion, then all accumulated SAMF state is returned WITHOUT undoing.
// `--single-end` uses this to accumulate SAMF state across outer rounds and undo only once
// at the very end. Each insertion reprocesses that pass's shooting output from a CLEAN
// negation state (its gates are self-contained; the shooting negation is a final-state
// adjustment, not a pre-condition), and the pass negation is transported through the
// insertion permutation before being folded into the total.
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
    input_tags: &[u32],
) -> (Vec<[u16; 3]>, Transpositions, Vec<u8>, usize, Vec<u32>) {
    let track = !input_tags.is_empty();
    let passes = shooting_times.max(1);
    let mut input_gates = input.to_vec();
    let mut tags: Vec<u32> = input_tags.to_vec();
    let mut total_t = Transpositions {
        transpositions: Vec::new(),
    };
    let mut total_neg = vec![0u8; n];
    let mut total_compressions = 0usize;

    // ---- Stage B (gen mode): min-generation-anchored bidirectional bounded passes ----
    // Each pass starts at a random minimum-generation gate, shoots in a random direction (left via
    // the reverse trick), runs until `PASS_LENGTH` successful replacements (0 = unbounded) or the
    // first unreplaceable collision, then resolves its own SAMF state immediately (per-pass
    // unsamf). Repeat until every gate's generation >= MIN_GEN, capped at MAX_PASSES. Plain-SAMF
    // insertion (if m>0) runs once over the whole circuit at the end.
    if crate::replace::replace::gen_mode() && track {
        use crate::replace::segcircuit::{SamfLedger, SamfTail, SegCircuit};
        use std::sync::atomic::Ordering::Relaxed;
        const WINDOW_CAP: usize = 8192; // max gates a single short pass spans
        let mut rng = rand::rng();
        let min_gen_target = crate::replace::replace::MIN_GEN.load(Relaxed) as u32;
        let pass_length = crate::replace::replace::PASS_LENGTH.load(Relaxed);
        let max_passes = crate::replace::replace::MAX_PASSES.load(Relaxed);
        let min_gen_permille = crate::replace::replace::MIN_GEN_PERMILLE.load(Relaxed);
        // Stage D: pause shooting (return to the driver to compress) once the working circuit
        // reaches this many gates. 0 = no cap.
        let shoot_size_cap = crate::replace::replace::SHOOT_SIZE_CAP.load(Relaxed);
        // Stage C step 3b: segmented circuit + a single RIGHTWARD ledger of unresolved SAMF tails
        // (global propagation, undone once before compression). A pass anchors at a random
        // min-generation gate and shoots a window bounded so it does NOT cross any ledger entry,
        // then records its net SAMF (perm + negation) as one rightward ledger entry at the window's
        // far edge. Entries are resolved ONLY by full-circuit `apply_ledger` sweeps (periodic +
        // final) -- consistent sweeps that stay frame-correct -- and the accumulated net relabeling
        // is undone with a single right-end translation table at the very end. (v1: rightward-only.)
        // Periodic flush keeps the ledger sparse (larger windows): resolve all entries into the
        // body in one consistent sweep once the ledger grows past this. (apply_ledger composes
        // entries in the correct conjugated/prepend order, so flush+continue is correct.)
        const LEDGER_FLUSH_AT: usize = 16;
        // CENTRALIZE=p: prioritize starting passes at gates in the central p% of the circuit (index
        // band [(50-p/2)%, (50+p/2)%]) whose generation is (min_gen + 1); such gates are chosen
        // first and shot BIDIRECTIONALLY (left pass from the gate, then right pass from the right
        // edge of its replacement). 0 = off.
        let centralize_pct: usize = std::env::var("CENTRALIZE")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(0)
            .min(100);
        if centralize_pct > 0 {
            println!("[gen] CENTRALIZE active: central {}% band, gen=min+1, bidirectional", centralize_pct);
        }
        let mut centralized_passes = 0usize;
        let mut sc = SegCircuit::from_flat(&input_gates, &tags);
        let mut ledger = SamfLedger::new();
        let mut acc_net = SamfTail::identity(n); // accumulated relabeling already baked in (undo at end)

        // DEBUG (env STAGEC_CHECK): verify the GLOBAL Stage C invariant — materialize the current
        // state (resolve the ledger into the body, then undo acc_net) and check it still equals the
        // input circuit (sampled, U1024). Returns true if OK; on break, logs the pass index and the
        // accumulated state so we can localize which pass/flush first corrupts the global function.
        let stagec_check = std::env::var("STAGEC_CHECK").is_ok();
        let global_check = |sc: &SegCircuit, ledger: &SamfLedger, acc_net: &SamfTail, where_: &str| -> bool {
            if !stagec_check {
                return true;
            }
            let (fg, _ft) = sc.to_flat();
            let all: Vec<(usize, SamfTail)> =
                ledger.entries().iter().map(|(p, t)| (*p, t.clone())).collect();
            let (mut mg, _mt, net) = apply_ledger(&fg, &[], &all, n);
            let acc = acc_net.then(&net);
            let net_perm = crate::circuit::circuit::Permutation {
                data: acc.perm.iter().map(|&w| w as usize).collect(),
            };
            let net_t = Transpositions::from_perm(&net_perm);
            // Empty DBs -> apply_unsamf appends RAW gadgets (no DB compression): same function,
            // far faster, so this check is feasible even on million-gate circuits.
            apply_unsamf(&mut mg, &net_t, &acc.neg, n, env, &[], &[], &mut Vec::new());
            let materialized = CircuitSeq { gates: mg };
            let inp = CircuitSeq { gates: input.to_vec() };
            if materialized.probably_equal(&inp, n, 512).is_err() {
                let acc_nonid = (0..n).filter(|&w| acc_net.perm[w] as usize != w).count();
                let acc_negs = acc_net.neg.iter().filter(|&&v| v == 1).count();
                eprintln!(
                    "[STAGEC GLOBAL BREAK] {where_} sc_len={} ledger={} acc_perm_nonid={} acc_negs={}",
                    sc.len(), ledger.len(), acc_nonid, acc_negs
                );
                return false;
            }
            true
        };

        // A RIGHTWARD pass anchored at `anchor`: window capped + bounded so it never crosses a
        // ledger entry; shoot; resolve negation locally (NOT gadgets); record only the PERMUTATION
        // as one rightward ledger entry at the far edge. Returns the compression count.
        let do_right_pass = |sc: &mut SegCircuit, ledger: &mut SamfLedger, anchor: usize| -> usize {
            if anchor >= sc.len() {
                return 0;
            }
            let bound = ledger.next_after(anchor).saturating_sub(anchor);
            let wlen = WINDOW_CAP.min(sc.len() - anchor).min(bound.max(1));
            let (wg, wt) = sc.read_range(anchor, wlen);
            let (mut out, t_p, neg_p, comp, mut out_t) = shuffled_shooting_game_core(
                &wg, n, env, curated_shard_dbs, shard_dbs, gates_ahead_expand, gates_ahead_samf,
                type_attempts, &wt, Some(0), pass_length, true,
            );
            apply_unsamf(
                &mut out,
                &Transpositions { transpositions: Vec::new() },
                &neg_p, n, env, curated_shard_dbs, shard_dbs, &mut out_t,
            );
            let far = SamfTail::from_transpositions(&t_p, &vec![0u8; n], n);
            // DEBUG (non-perturbing): per-pass invariant — out and wg differ only by the recorded
            // permutation `far` (far.neg==0 here). Uses DETERMINISTIC sample inputs and a functional
            // permutation compare (NO rand::rng / apply_unsamf / probably_equal) so the check does
            // not shift the shooting trajectory (the bug is RNG-sensitive). Accepts any of the
            // natural perm directions to avoid a convention mismatch; a real break matches none.
            if stagec_check && !stagec_pass_ok(&out, &wg, &far.perm, n) {
                eprintln!(
                    "[RIGHT-PASS BREAK] anchor={anchor} wlen={wlen} out_len={} far_nonid={}",
                    out.len(),
                    (0..n).filter(|&w| far.perm[w] as usize != w).count()
                );
                std::process::exit(8);
            }
            let new_len = out.len();
            sc.splice(anchor, wlen, &out, &out_t);
            ledger.shift_from(anchor + wlen, new_len as isize - wlen as isize);
            if !far.is_identity() {
                ledger.insert(anchor + new_len, far);
            }
            comp
        };

        // A LEFT pass starting at gate `gate_idx` (shoot leftward via the reverse trick). It is
        // SELF-CONTAINED: its SAMF (perm + negation) is resolved locally, so it adds nothing to the
        // rightward ledger. Window is bounded left so it never crosses a ledger entry. Returns
        // (compression count, right edge of the spliced output = the replacement's far edge).
        let do_left_pass =
            |sc: &mut SegCircuit, ledger: &mut SamfLedger, gate_idx: usize| -> (usize, usize) {
                let lo = gate_idx
                    .saturating_sub(WINDOW_CAP - 1)
                    .max(ledger.prev_before(gate_idx));
                let wlen = gate_idx - lo + 1;
                let (mut wg, mut wt) = sc.read_range(lo, wlen);
                let wg_orig = if stagec_check { wg.clone() } else { Vec::new() };
                wg.reverse();
                wt.reverse();
                let (mut out, t_p, neg_p, comp, mut out_t) = shuffled_shooting_game_core(
                    &wg, n, env, curated_shard_dbs, shard_dbs, gates_ahead_expand, gates_ahead_samf,
                    type_attempts, &wt, Some(0), pass_length, true,
                );
                // Local full unsamf (perm + negation) in the reversed frame -> self-contained.
                apply_unsamf(&mut out, &t_p, &neg_p, n, env, curated_shard_dbs, shard_dbs, &mut out_t);
                out.reverse();
                out_t.reverse();
                // DEBUG (non-perturbing): the self-contained left pass output must equal the
                // original window (records NO ledger entry). Identity permutation -> direct equality.
                if stagec_check {
                    let identity: Vec<u16> = (0..n as u16).collect();
                    if !stagec_pass_ok(&out, &wg_orig, &identity, n) {
                        eprintln!("[LEFT-PASS BREAK] lo={lo} wlen={wlen} out_len={}", out.len());
                        std::process::exit(8);
                    }
                }
                let new_len = out.len();
                sc.splice(lo, wlen, &out, &out_t);
                ledger.shift_from(lo + wlen, new_len as isize - wlen as isize);
                (comp, lo + new_len)
            };

        let mut pass_idx = 0usize;
        loop {
            if sc.is_empty() {
                break;
            }
            // Stop once >= min_gen_permille/1000 of gates have reached MIN_GEN.
            let below = sc.count_below(min_gen_target);
            if below * 1000 <= sc.len() * (1000 - min_gen_permille) {
                break;
            }
            // Stage D: pause shooting once the working circuit grows past the size cap, so the
            // driver can compress it back down before the next stage. min-gen takes priority above.
            if shoot_size_cap > 0 && sc.len() >= shoot_size_cap {
                println!(
                    "[gen] stage-D: size cap {} reached ({} gates, min_gen {}); pausing to compress",
                    shoot_size_cap,
                    sc.len(),
                    sc.min_gen()
                );
                break;
            }
            if pass_idx >= max_passes {
                println!(
                    "[gen] stage-C stop: max_passes {} reached (min_gen {}, gates {}, ledger {})",
                    max_passes,
                    sc.min_gen(),
                    sc.len(),
                    ledger.len()
                );
                break;
            }
            // Periodic full-circuit flush keeps the ledger sparse so windows stay large. Resolving
            // in one consistent sweep is frame-correct; fold its net into acc_net.
            if ledger.len() >= LEDGER_FLUSH_AT {
                // DEBUG: verify the global invariant holds with the FULL ledger (before flush) and
                // again after the flush — distinguishes "a pass corrupted state" from "the flush
                // (apply_ledger) corrupted it".
                if stagec_check
                    && !global_check(&sc, &ledger, &acc_net, &format!("PRE-flush pass {pass_idx}"))
                {
                    std::process::exit(9);
                }
                let (fg, ft) = sc.to_flat();
                let all: Vec<(usize, SamfTail)> =
                    ledger.entries().iter().map(|(p, t)| (*p, t.clone())).collect();
                let (ng, nt, net) = apply_ledger(&fg, &ft, &all, n);
                acc_net = acc_net.then(&net);
                sc = SegCircuit::from_flat(&ng, &nt);
                ledger = SamfLedger::new();
                if stagec_check
                    && !global_check(&sc, &ledger, &acc_net, &format!("POST-flush pass {pass_idx}"))
                {
                    std::process::exit(9);
                }
            }
            crate::replace::replace::REC_PASS.store(pass_idx + 1, Relaxed);

            // CENTRALIZE: try a prioritized central gate at generation (min_gen + 1) first.
            let prioritized = if centralize_pct > 0 {
                let len = sc.len();
                let lo = len * (50 - centralize_pct / 2) / 100;
                let hi = (len * (50 + centralize_pct / 2) / 100).max(lo + 1).min(len);
                // Fractional floor + 1: skip the stuck bottom (1-frac) so the central band targets
                // the lowest *raisable* generation, not a permanently-stuck absolute minimum.
                let skip = ((1000 - min_gen_permille) * sc.len()) / 1000;
                let target_gen = sc.frac_min_gen(skip).saturating_add(1);
                sc.random_index_with_gen_in_range(target_gen, lo, hi, &mut rng)
            } else {
                None
            };
            if let Some(g) = prioritized {
                // Bidirectional: left pass from the gate, then right pass from the replacement edge.
                let (comp_l, right_edge) = do_left_pass(&mut sc, &mut ledger, g);
                let anchor_r = right_edge.min(sc.len().saturating_sub(1));
                let comp_r = do_right_pass(&mut sc, &mut ledger, anchor_r);
                total_compressions += comp_l + comp_r;
                centralized_passes += 1;
                pass_idx += 2;
            } else {
                // Anchor at the fractional floor: skip the bottom (1-frac) of gates (the stuck,
                // rarely-colliding ones) so the anchor keeps raising the rest instead of looping on
                // a permanently-stuck absolute minimum. skip=0 (frac=1.0) -> absolute minimum.
                let skip = ((1000 - min_gen_permille) * sc.len()) / 1000;
                let anchor = match sc.random_frac_min_gen_index(skip, &mut rng) {
                    Some(a) => a,
                    None => break,
                };
                total_compressions += do_right_pass(&mut sc, &mut ledger, anchor);
                pass_idx += 1;
            }
            if pass_idx % 50 == 0 {
                println!(
                    "[gen] stage-C pass {} min_gen {} gen0 {} circuit_gates {} ledger {}",
                    pass_idx,
                    sc.min_gen(),
                    sc.count_zero(),
                    sc.len(),
                    ledger.len()
                );
            }
        }
        if centralize_pct > 0 {
            println!(
                "[gen] stage-C done: {} passes ({} central/bidirectional)",
                pass_idx, centralized_passes
            );
        }
        // Final flush: resolve the remaining ledger entries into the body in one consistent sweep,
        // then undo the WHOLE accumulated net relabeling with a single right-end translation table.
        let (fg, ft) = sc.to_flat();
        let all: Vec<(usize, SamfTail)> =
            ledger.entries().iter().map(|(p, t)| (*p, t.clone())).collect();
        let (mut mg, mut mt, net) = apply_ledger(&fg, &ft, &all, n);
        acc_net = acc_net.then(&net);
        let net_perm = crate::circuit::circuit::Permutation {
            data: acc_net.perm.iter().map(|&w| w as usize).collect(),
        };
        let net_t = Transpositions::from_perm(&net_perm);
        apply_unsamf(
            &mut mg,
            &net_t,
            &acc_net.neg,
            n,
            env,
            curated_shard_dbs,
            shard_dbs,
            &mut mt,
        );
        input_gates = mg;
        tags = mt;
        // DEBUG: verify the Stage C OUTPUT (after final flush, before compression) equals the input.
        // Splits "bug in Stage C" from "bug in the later compression phase".
        if stagec_check {
            let fin = CircuitSeq { gates: input_gates.clone() };
            let inp = CircuitSeq { gates: input.to_vec() };
            if fin.probably_equal(&inp, n, 3000).is_err() {
                eprintln!("[STAGEC OUTPUT BREAK] Stage C output != input (bug is in Stage C), pass_idx={pass_idx}");
            } else {
                eprintln!("[STAGEC OUTPUT OK] Stage C output == input; any later failure is in COMPRESSION, pass_idx={pass_idx}");
            }
        }
        // One plain-SAMF insertion over the whole circuit (per-pass model resolves it immediately).
        if m > 0 {
            let (mut out_b, t_b, neg_b, mut tags_b) =
                insert_m_samfs_core(&input_gates, n, m, x, &tags);
            apply_unsamf(
                &mut out_b,
                &t_b,
                &neg_b,
                n,
                env,
                curated_shard_dbs,
                shard_dbs,
                &mut tags_b,
            );
            input_gates = out_b;
            tags = tags_b;
        }
        let mg = tags.iter().copied().min().unwrap_or(0);
        println!(
            "[gen] stage-B done: {} passes, min_gen {}, gates {}",
            pass_idx,
            mg,
            tags.len()
        );
        // Gen mode resolves all SAMF state internally; return identity so the caller's final
        // unsamf is a no-op.
        return (
            input_gates,
            Transpositions {
                transpositions: Vec::new(),
            },
            vec![0u8; n],
            total_compressions,
            tags,
        );
    }

    for pass in 0..passes {
        crate::replace::replace::REC_PASS.store(pass + 1, std::sync::atomic::Ordering::Relaxed);
        let (out_a, t_a, neg_a, compressions, tags_a) = shuffled_shooting_game_core(
            &input_gates,
            n,
            env,
            curated_shard_dbs,
            shard_dbs,
            gates_ahead_expand,
            gates_ahead_samf,
            type_attempts,
            if track { &tags } else { &[] },
            None,
            0,
            false,
        );
        let (out_b, t_b, neg_b, tags_b) =
            insert_m_samfs_core(&out_a, n, m, x, if track { &tags_a } else { &[] });
        if track {
            tags = tags_b;
        }

        // Combined permutation for this shooting round: collision game first, then insertion.
        let t_pass = t_a.concat(&t_b);
        // Combined final negation for this shooting round: insertion's own negation plus the
        // collision game's negation transported through the insertion permutation.
        let mut neg_pass = neg_b;
        for w in 0..n {
            if neg_a[w] == 1 {
                let cw = t_b.evaluate(w as u16) as usize;
                neg_pass[cw] ^= 1;
            }
        }

        // Fold this shooting round into the total pending SAMF/NOT state.
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
        input_gates = out_b;

        // --track-survivors: log the live survivor count every SURVIVOR_LOG_EVERY shooting passes
        // (and on the final pass), so the survivor decay can be tracked across the mixing process.
        if track {
            let every = std::env::var("SURVIVOR_LOG_EVERY")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(10)
                .max(1);
            if (pass + 1) % every == 0 || pass + 1 == passes {
                if crate::replace::replace::gen_mode() {
                    let min_gen = tags.iter().copied().min().unwrap_or(0);
                    let gen0 = tags.iter().filter(|&&t| t == 0).count();
                    println!(
                        "[gen] pass {} min_gen {} gen0 {} circuit_gates {}",
                        pass + 1,
                        min_gen,
                        gen0,
                        tags.len()
                    );
                } else {
                    let alive = tags
                        .iter()
                        .filter(|&&t| t != crate::replace::replace::TAG_NEW)
                        .count();
                    println!(
                        "[survivors] pass {} alive {} circuit_gates {}",
                        pass + 1,
                        alive,
                        tags.len()
                    );
                }
            }
        }
    }

    (input_gates, total_t, total_neg, total_compressions, tags)
}

// Run each shooting round as one collision game followed by one per-gate SAMF insertion,
// then perform a SINGLE unsamf at the very end. Returns the shooting game's compression
// count across all shooting rounds.
// Apply a set of unresolved ledger SAMF tails to a circuit in ONE consistent left-to-right sweep
// (Stage C). This is the same relabel-and-flush the shoot performs over a full circuit -- the key
// to staying frame-correct: each entry (position, tail) is folded into the running t_list +
// negation_mask when the sweep reaches its position, every gate is relabeled by t_list, and a
// gate's control negations are flushed as NOT gates before it reads them (emit_relabelled_gate).
// Returns the rewritten gates+tags and the NET tail (permutation + leftover negation) that
// relabels the circuit's output -- folded into the accumulated net and undone once at the very end.
// (Used only for full-circuit flushes; passes never partially materialize, which is what caused
// the earlier frame mismatch.)
fn apply_ledger(
    gates: &[[u16; 3]],
    tags: &[u32],
    entries: &[(usize, crate::replace::segcircuit::SamfTail)],
    n: usize,
) -> (Vec<[u16; 3]>, Vec<u32>, crate::replace::segcircuit::SamfTail) {
    use crate::replace::segcircuit::SamfTail;
    let track = !tags.is_empty();
    let mut t_list = Transpositions {
        transpositions: Vec::new(),
    };
    let mut mask = vec![0u8; n];
    let mut out: Vec<[u16; 3]> = Vec::with_capacity(gates.len());
    let mut out_t: Vec<u32> = Vec::with_capacity(if track { gates.len() } else { 0 });
    let mut ei = 0usize;
    // Fold one ledger tail into the running (t_list, mask). Each pass recorded its tail in the
    // frame that IGNORES the entries to its left (uniform relabel); at the flush those left entries
    // are applied first and CONJUGATE this tail. Equivalently, compose with this (later-position)
    // tail INNERMOST -- i.e. PREPEND its swaps so the running permutation is
    // sigma_1 . sigma_2 . ... . sigma_k (earliest-position outermost), not appended (which gave the
    // reversed sigma_k . ... . sigma_1 and only matched when the tails happened to commute).
    let mut fold = |t_list: &mut Transpositions, mask: &mut Vec<u8>, e: &SamfTail| {
        let mut nm = e.neg.clone();
        for w in 0..n {
            if mask[w] == 1 {
                nm[e.perm[w] as usize] ^= 1;
            }
        }
        *mask = nm;
        let mut swaps = e.perm_to_swaps();
        swaps.append(&mut t_list.transpositions);
        t_list.transpositions = swaps; // prepend e's swaps (e applied first in evaluate)
    };
    for (i, &g) in gates.iter().enumerate() {
        while ei < entries.len() && entries[ei].0 <= i {
            fold(&mut t_list, &mut mask, &entries[ei].1);
            ei += 1;
        }
        let before = out.len();
        emit_relabelled_gate(g, &mut out, &t_list, &mut mask, n);
        if track {
            let added = out.len() - before; // [NOT-corrections..., the gate]
            for _ in 0..added.saturating_sub(1) {
                out_t.push(crate::replace::replace::new_gate_tag(&[tags[i]]));
            }
            out_t.push(tags[i]);
        }
    }
    while ei < entries.len() {
        fold(&mut t_list, &mut mask, &entries[ei].1);
        ei += 1;
    }
    let perm: Vec<u16> = (0..n as u16).map(|w| t_list.evaluate(w)).collect();
    (out, out_t, SamfTail { perm, neg: mask })
}

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
    tags: &mut Vec<u32>,
) -> usize {
    let track = !tags.is_empty();
    let (mut out, t_round, neg_round, compressions, mut out_tags) = shuffled_shoot_then_samf_core(
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
        if track { tags } else { &[] },
    );
    apply_unsamf(
        &mut out,
        &t_round,
        &neg_round,
        n,
        env,
        curated_shard_dbs,
        shard_dbs,
        &mut out_tags,
    );
    circuit.gates = out;
    if track {
        *tags = out_tags;
    }
    compressions
}

#[cfg(test)]
mod reversed_samf_tests {
    use super::{
        SWAP_N1_3W, SWAP_N1_4W, SWAP_N2_3W, SWAP_N2_4W, Transpositions, neg_flips, random_neg_type,
        shoot_gate_to_first_collision, shoot_materialized_gate_to_first_collision,
    };
    use crate::circuit::circuit::CircuitSeq;
    use std::collections::VecDeque;

    #[test]
    fn shot_gate_moves_to_its_first_collision() {
        let shot = [0, 1, 2];
        let pass_a = [3, 4, 5];
        let pass_b = [6, 7, 8];
        let collider = [9, 0, 10];
        let suffix = [11, 12, 13];
        let mut remaining = VecDeque::from([shot, pass_a, pass_b, collider, suffix]);
        let t = Transpositions {
            transpositions: Vec::new(),
        };

        let (actual_shot, passed, collided) =
            shoot_gate_to_first_collision(&mut remaining, &t, &[0; 17]).unwrap();

        assert_eq!(actual_shot, shot);
        assert_eq!(passed, vec![pass_a, pass_b]);
        assert!(collided);
        assert_eq!(remaining, VecDeque::from([collider, suffix]));
    }

    #[test]
    fn next_shot_continues_from_suffix_after_collision() {
        let collider = [9, 0, 10];
        let suffix_a = [11, 12, 13];
        let suffix_b = [14, 15, 16];
        let mut remaining = VecDeque::from([collider, suffix_a, suffix_b]);
        let t = Transpositions {
            transpositions: Vec::new(),
        };

        let (actual_shot, passed, collided) =
            shoot_gate_to_first_collision(&mut remaining, &t, &[0; 17]).unwrap();

        assert_eq!(actual_shot, collider);
        assert_eq!(passed, vec![suffix_a, suffix_b]);
        assert!(!collided);
        assert!(remaining.is_empty());
    }

    #[test]
    fn materialized_replacement_tail_is_not_relabelled_again() {
        let shot = [0, 1, 2];
        let commuting = [3, 4, 5];
        let collider_after_relabel = [6, 7, 8];
        let mut remaining = VecDeque::from([commuting, collider_after_relabel]);
        let t = Transpositions {
            transpositions: vec![(0, 7, 0)],
        };

        let (passed, collided) =
            shoot_materialized_gate_to_first_collision(shot, &mut remaining, &t, &[0; 17]);

        assert_eq!(passed, vec![commuting]);
        assert!(collided);
        assert_eq!(remaining, VecDeque::from([collider_after_relabel]));
    }

    #[test]
    fn materialized_shot_stops_before_dirty_control_correction() {
        let shot = [0, 1, 2];
        let dirty_control_gate = [3, 4, 5];
        let suffix = [6, 7, 8];
        let mut remaining = VecDeque::from([dirty_control_gate, suffix]);
        let t = Transpositions {
            transpositions: Vec::new(),
        };
        let mut negation_mask = [0; 9];
        negation_mask[4] = 1;

        let (passed, collided) =
            shoot_materialized_gate_to_first_collision(shot, &mut remaining, &t, &negation_mask);

        assert!(passed.is_empty());
        assert!(!collided);
        assert_eq!(remaining, VecDeque::from([dirty_control_gate, suffix]));
    }

    #[test]
    fn dirty_control_shot_stays_before_commuting_suffix() {
        let shot = [0, 1, 2];
        let commuting = [3, 4, 5];
        let mut remaining = VecDeque::from([shot, commuting]);
        let t = Transpositions {
            transpositions: Vec::new(),
        };
        let mut negation_mask = [0; 6];
        negation_mask[1] = 1;

        let (actual_shot, passed, collided) =
            shoot_gate_to_first_collision(&mut remaining, &t, &negation_mask).unwrap();

        assert_eq!(actual_shot, shot);
        assert!(passed.is_empty());
        assert!(!collided);
        assert_eq!(remaining, VecDeque::from([commuting]));
    }

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

#[cfg(test)]
mod unsamf_scale_tests {
    use super::{apply_neg_to_mask, apply_unsamf, Transpositions};
    use crate::circuit::circuit::{Gate, U1024};
    use lmdb::Environment;
    use rand::{Rng, RngCore};

    // The full SAMF-insert -> unsamf cycle must be the identity, at LARGE n (n=384 = feistelized
    // n=128) where the production failures appear. Builds a random sequence of swap+negation
    // gadgets (physical) while tracking (t_list perm, mask neg); apply_unsamf appends the inverse;
    // circ ++ unsamf must compute identity on every wire. Empty DBs -> integrate appends raw, so the
    // real apply_unsamf perm/neg logic is exercised without an LMDB.
    fn run_cycle(n: usize, num_samfs: usize, trials: usize) {
        let dir = std::env::temp_dir().join(format!("unsamf_cycle_{}_{}", n, std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let env = Environment::new()
            .set_max_dbs(4)
            .set_map_size(64 * 1024 * 1024)
            .open(&dir)
            .expect("open temp env");
        let mut rng = rand::rng();
        let wmask = (U1024::one() << n) - U1024::one();
        for trial in 0..trials {
            let mut circ: Vec<[u16; 3]> = Vec::new();
            let mut t_list = Transpositions { transpositions: Vec::new() };
            let mut mask = vec![0u8; n];
            for _ in 0..num_samfs {
                let lo = rng.random_range(0..n as u16);
                let hi = loop {
                    let w = rng.random_range(0..n as u16);
                    if w != lo {
                        break w;
                    }
                };
                let neg_type = rng.random_range(0u16..=3);
                circ.extend_from_slice(&Transpositions::gen_gates_swap(n, (lo, hi, neg_type)));
                t_list.transpositions.push((lo, hi, neg_type));
                apply_neg_to_mask(&mut mask, lo as usize, hi as usize, neg_type);
            }
            let mut g = circ.clone();
            apply_unsamf(&mut g, &t_list, &mask, n, &env, &[], &[], &mut Vec::new());
            for _ in 0..100 {
                let mut bytes = [0u8; 128];
                rng.fill_bytes(&mut bytes);
                let x = U1024::from_little_endian(&bytes) & wmask;
                let out = Gate::evaluate_index_list_1024(x, &g);
                assert_eq!(out, x, "n={n} trial={trial}: circ ∘ unsamf != identity");
            }
        }
    }

    #[test]
    fn unsamf_cycle_identity_n64() {
        run_cycle(64, 200, 20);
    }

    #[test]
    fn unsamf_cycle_identity_n192() {
        run_cycle(192, 200, 20);
    }

    #[test]
    fn unsamf_cycle_identity_n384() {
        run_cycle(384, 300, 30);
    }
}
