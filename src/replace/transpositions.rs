// For adding wire shuffles and bit flips
use crate::replace::frozen::FrozenDb;
use crate::{
    circuit::{Permutation, circuit::CircuitSeq},
    replace::replace::{ExpandPairMode, expand_once_scored},
};
use rand::Rng;
use rand::seq::IndexedRandom;
use rustc_hash::FxHashMap as HashMap;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};

pub static SAMF_INSERTIONS_MADE: AtomicUsize = AtomicUsize::new(0);
pub static SAMF_COMPRESSIONS_MADE: AtomicUsize = AtomicUsize::new(0);
pub static SAMF_COMPRESSIONS_FAILED: AtomicUsize = AtomicUsize::new(0);
// Curated expansions performed at collisions in the collision game.
pub static CURATED_REPLACEMENTS_MADE: AtomicUsize = AtomicUsize::new(0);
// Stage F / #10: curated expansions where pending NOTs were absorbed into the lookup.
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
// returns; on a miss the gadget is appended verbatim. Frozen replacements are
// functionally equivalent to the window, so this is equivalence-preserving.
fn integrate_samf_compressed(
    gates: &mut Vec<[u16; 3]>,
    samf: &[[u16; 3]],
    n: usize,
    db: &FrozenDb,
    mut tags: Option<&mut Vec<u32>>,
) {
    use crate::replace::pairs::{
        LookupScope, compress_curated_frozen, find_any_replacement_frozen,
    };
    let take = gates.len().min(3);
    let ctx_start = gates.len() - take;
    let event_tag = tags
        .as_ref()
        .map(|tags| crate::replace::replace::new_gate_tag(&tags[ctx_start..]))
        .unwrap_or(crate::replace::replace::TAG_NEW);
    let mut window: Vec<[u16; 3]> = gates[ctx_start..].to_vec();
    window.extend_from_slice(samf);
    // Tiered replacement, best-quality first; all tiers are equivalence-preserving:
    //   1. curated compression (strictly-useful shorter replacement),
    //   2. any curated equivalent (even if not a compression) — still hides the SAMF,
    //   3. any sharded equivalent — fall back to the full sharded DB.
    // On a total miss, emit the undo SAMF gadget verbatim (correct, just unhidden).
    let repl = compress_curated_frozen(&window, n, db, LookupScope::CuratedOnly)
        .or_else(|| find_any_replacement_frozen(&window, n, db, LookupScope::CuratedOnly))
        .or_else(|| find_any_replacement_frozen(&window, n, db, LookupScope::RegularOnly));
    if let Some(repl) = repl {
        gates.truncate(ctx_start);
        gates.extend_from_slice(&repl);
        if let Some(tags) = tags.as_deref_mut() {
            tags.truncate(ctx_start);
            tags.extend(std::iter::repeat(event_tag).take(repl.len()));
        }
        END_SAMF_COMPRESSIONS_MADE.fetch_add(1, Ordering::Relaxed);
    } else {
        gates.extend_from_slice(samf);
        if let Some(tags) = tags.as_deref_mut() {
            tags.extend(std::iter::repeat(event_tag).take(samf.len()));
        }
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
    db: &FrozenDb,
) {
    apply_unsamf_inner(output, t_list, negation_mask, n, db, None);
}

pub fn apply_unsamf_tagged(
    output: &mut Vec<[u16; 3]>,
    t_list: &Transpositions,
    negation_mask: &[u8],
    n: usize,
    db: &FrozenDb,
    tags: &mut Vec<u32>,
) {
    apply_unsamf_inner(output, t_list, negation_mask, n, db, Some(tags));
}

fn apply_unsamf_inner(
    output: &mut Vec<[u16; 3]>,
    t_list: &Transpositions,
    negation_mask: &[u8],
    n: usize,
    db: &FrozenDb,
    mut tags: Option<&mut Vec<u32>>,
) {
    let p = t_list.to_perm(n);
    let mut t = Transpositions::from_perm(&p);
    let mut wire_positions: HashMap<u16, (usize, usize)> = HashMap::default();
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
        integrate_samf_compressed(output, &samf, n, db, tags.as_deref_mut());
    }
    for w in leftover_nots {
        let not_gates = Transpositions::gen_gates_not(n, w);
        integrate_samf_compressed(output, &not_gates, n, db, tags.as_deref_mut());
    }
}

pub fn insert_wire_shuffles_knuth(circuit: &mut CircuitSeq, n: usize, db: &FrozenDb) {
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
    apply_unsamf(&mut gates, &t_list, &negation_mask, n, db);
    circuit.gates = gates;
    println!("Complete. Ending len: {} gates", circuit.gates.len());
}

pub fn insert_wire_shuffles_simple(circuit: &mut CircuitSeq, n: usize, db: &FrozenDb) {
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
    apply_unsamf(&mut gates, &t_list, &negation_mask, n, db);
    circuit.gates = gates;
    println!("Complete. Ending len: {} gates", circuit.gates.len());
}

// Insert 2 shuffles are the beginning and end, and then an additional x number of shuffles
pub fn insert_wire_shuffles_x(circuit: &mut CircuitSeq, n: usize, x: usize, db: &FrozenDb) {
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
    apply_unsamf(&mut gates, &t_list, &negation_mask, n, db);
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
    let mut tags: Vec<u32> = Vec::new();
    let mut negation_mask = vec![0u8; n];
    for (i, gate) in input.iter().enumerate() {
        let gate_tag = input_tags
            .get(i)
            .copied()
            .unwrap_or(crate::replace::replace::TAG_NEW);
        if i % x == 0 {
            let t = Transpositions::gen_random_simple(n, m, &mut negation_mask);
            let samf_gates = t.to_circuit(n).gates;
            gates.extend_from_slice(&samf_gates);
            if track {
                tags.extend(
                    std::iter::repeat(crate::replace::replace::new_gate_tag(std::slice::from_ref(
                        &gate_tag,
                    )))
                    .take(samf_gates.len()),
                );
            }
            t_list.transpositions.extend_from_slice(&t.transpositions);
            SAMF_INSERTIONS_MADE.fetch_add(t.transpositions.len(), Ordering::Relaxed);
        }
        let a = t_list.evaluate(gate[0]);
        let b = t_list.evaluate(gate[1]);
        let c = t_list.evaluate(gate[2]);
        let g = [a, b, c];
        if negation_mask[b as usize] == 1 {
            let not_gates = Transpositions::gen_gates_not(n, b);
            gates.extend_from_slice(&not_gates);
            if track {
                tags.extend(
                    std::iter::repeat(crate::replace::replace::new_gate_tag(std::slice::from_ref(
                        &gate_tag,
                    )))
                    .take(not_gates.len()),
                );
            }
            negation_mask[b as usize] = 0;
        }
        if negation_mask[c as usize] == 1 {
            let not_gates = Transpositions::gen_gates_not(n, c);
            gates.extend_from_slice(&not_gates);
            if track {
                tags.extend(
                    std::iter::repeat(crate::replace::replace::new_gate_tag(std::slice::from_ref(
                        &gate_tag,
                    )))
                    .take(not_gates.len()),
                );
            }
            negation_mask[c as usize] = 0;
        }
        gates.push(g);
        if track {
            tags.push(gate_tag);
        }
    }
    (gates, t_list, negation_mask, tags)
}

// Insert m samf between each gate
pub fn insert_wire_m_samfs_every_x(
    circuit: &mut CircuitSeq,
    n: usize,
    m: usize,
    x: usize,
    db: &FrozenDb,
) {
    println!("Inserting {} samfs between each gate", m);
    println!("Starting len: {} gates", circuit.gates.len());
    let (mut gates, t_list, negation_mask, mut tags) =
        insert_m_samfs_core(&circuit.gates, n, m, x, &[]);
    apply_unsamf(&mut gates, &t_list, &negation_mask, n, db);
    let _ = &mut tags;
    circuit.gates = gates;
    println!("Complete. Ending len: {} gates", circuit.gates.len());
}

pub fn insert_wire_m_samfs_every_x_tagged(
    circuit: &mut CircuitSeq,
    n: usize,
    m: usize,
    x: usize,
    db: &FrozenDb,
    tags: &mut Vec<u32>,
) {
    println!("Inserting {} samfs between each gate", m);
    println!("Starting len: {} gates", circuit.gates.len());
    let (mut gates, t_list, negation_mask, mut out_tags) =
        insert_m_samfs_core(&circuit.gates, n, m, x, tags);
    apply_unsamf_tagged(&mut gates, &t_list, &negation_mask, n, db, &mut out_tags);
    circuit.gates = gates;
    *tags = out_tags;
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

#[derive(Clone, Copy, Debug, Default)]
struct CollisionPassOptions {
    anchor: Option<usize>,
    replacement_budget: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StageBPassResult {
    pub hidden_samfs: usize,
    pub replacements: usize,
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
fn collision_game_core(
    input: &[[u16; 3]],
    n: usize,
    db: &FrozenDb,
    gates_ahead_expand: usize,
    gates_ahead_samf: usize,
    type_attempts: usize,
    input_tags: &[u32],
    options: CollisionPassOptions,
) -> (
    Vec<[u16; 3]>,
    Transpositions,
    Vec<u8>,
    usize,
    Vec<u32>,
    usize,
) {
    use crate::replace::pairs::{LookupScope, compress_curated_frozen, expand_curated_frozen_neg};
    use crate::replace::replace::TAG_NEW;

    let track = !input_tags.is_empty();
    let mut rng = rand::rng();
    let mut output: Vec<[u16; 3]> = Vec::new();
    let mut output_tags: Vec<u32> = Vec::new();
    let mut t_list = Transpositions {
        transpositions: Vec::new(),
    };
    let mut negation_mask = vec![0u8; n];
    let mut compressions: usize = 0;

    let start = if input.is_empty() {
        0
    } else if let Some(anchor) = options.anchor {
        anchor.min(input.len() - 1)
    } else {
        rng.random_range(0..input.len())
    };
    output.extend_from_slice(&input[..start]);
    if track {
        output_tags.extend_from_slice(&input_tags[..start]);
    }
    let mut remaining: VecDeque<[u16; 3]> = input[start..].iter().copied().collect();
    let mut remaining_tags: VecDeque<u32> = if track {
        input_tags[start..].iter().copied().collect()
    } else {
        VecDeque::new()
    };
    // Replacement gates are already expressed in the current physical wire space. Keep the
    // replacement tail separate from `remaining`, whose gates still require SAMF relabeling.
    let mut materialized_shot: Option<[u16; 3]> = None;
    let mut materialized_shot_tag: u32 = TAG_NEW;
    let mut replacements: usize = 0;

    let flush_rest = |output: &mut Vec<[u16; 3]>,
                      output_tags: &mut Vec<u32>,
                      remaining: &mut VecDeque<[u16; 3]>,
                      remaining_tags: &mut VecDeque<u32>,
                      materialized_shot: &mut Option<[u16; 3]>,
                      materialized_shot_tag: u32,
                      t_list: &Transpositions,
                      negation_mask: &mut [u8]| {
        if let Some(shot) = materialized_shot.take() {
            output.push(shot);
            if track {
                output_tags.push(materialized_shot_tag);
            }
        }
        while let Some(gate) = remaining.pop_front() {
            let tag = if track {
                remaining_tags.pop_front().unwrap()
            } else {
                TAG_NEW
            };
            emit_relabelled_gate(gate, output, t_list, negation_mask, n);
            if track {
                output_tags.resize(
                    output.len() - 1,
                    crate::replace::replace::new_gate_tag(std::slice::from_ref(&tag)),
                );
                output_tags.push(tag);
            }
        }
    };

    while materialized_shot.is_some() || !remaining.is_empty() {
        let (shot, shot_is_materialized, mut passed, mut has_collision, shot_tag, mut passed_tags) =
            if let Some(shot) = materialized_shot.take() {
                let (passed, collided) = shoot_materialized_gate_to_first_collision(
                    shot,
                    &mut remaining,
                    &t_list,
                    &negation_mask,
                );
                let passed_tags = if track {
                    (0..passed.len())
                        .map(|_| remaining_tags.pop_front().unwrap())
                        .collect()
                } else {
                    Vec::new()
                };
                (
                    shot,
                    true,
                    passed,
                    collided,
                    materialized_shot_tag,
                    passed_tags,
                )
            } else {
                let (shot, passed, collided) =
                    shoot_gate_to_first_collision(&mut remaining, &t_list, &negation_mask).unwrap();
                let (shot_tag, passed_tags) = if track {
                    let shot_tag = remaining_tags.pop_front().unwrap();
                    let passed_tags = (0..passed.len())
                        .map(|_| remaining_tags.pop_front().unwrap())
                        .collect();
                    (shot_tag, passed_tags)
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
                output_tags.resize(
                    output.len(),
                    crate::replace::replace::new_gate_tag(std::slice::from_ref(&shot_tag)),
                );
            }
        }
        let forced_collision =
            !has_collision && crate::replace::replace::outgoing_gen_mode() && !passed.is_empty();
        if forced_collision {
            for gate in passed.drain(..).rev() {
                remaining.push_front(gate);
            }
            if track {
                for tag in passed_tags.drain(..).rev() {
                    remaining_tags.push_front(tag);
                }
            }
            has_collision = true;
            crate::replace::replace::FORCED_COLLISIONS.fetch_add(1, Ordering::Relaxed);
        }
        for (i, gate) in passed.into_iter().enumerate() {
            emit_relabelled_gate(gate, &mut output, &t_list, &mut negation_mask, n);
            if track {
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

            // 1) Curated expansion of a window anchored at the colliding pair. Legacy mode uses
            // contiguous windows. Gen outgoing mode uses the SSG #11 behavior: build candidate
            // windows up to size 5 by adding later gates that can commute left to the collider,
            // then try largest windows first with the outgoing ranker breaking ties within a size.
            const OUTGOING_LOOKAHEAD: usize = 8;
            const MAX_EXPAND_ATTEMPTS: usize = 16;
            const FEATURE_CTX: usize = 64;

            let shot_rg = if shot_is_materialized {
                shot
            } else {
                relabel_gate(shot, &t_list)
            };
            let clean_controls = |g: &[u16; 3]| {
                negation_mask[g[1] as usize] == 0 && negation_mask[g[2] as usize] == 0
            };
            let look = if crate::replace::replace::outgoing_gen_mode() {
                OUTGOING_LOOKAHEAD
            } else {
                gates_ahead_expand.max(2)
            }
            .min(remaining.len());
            let rg: Vec<[u16; 3]> = remaining
                .iter()
                .take(look)
                .map(|&g| relabel_gate(g, &t_list))
                .collect();

            let mut candidates: Vec<(Vec<usize>, Vec<[u16; 3]>)> = Vec::new();
            if crate::replace::replace::outgoing_gen_mode() {
                use crate::circuit::circuit::Gate;
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
                let max_window = gates_ahead_expand.max(2).min(remaining.len() + 1);
                for k in 2..=max_window {
                    if !clean_controls(&shot_rg) {
                        break;
                    }
                    if (0..k - 1).any(|m| m >= rg.len() || !clean_controls(&rg[m])) {
                        continue;
                    }
                    let mut window = Vec::with_capacity(k);
                    window.push(shot_rg);
                    window.extend(rg[..k - 1].iter().copied());
                    candidates.push(((0..k - 1).collect(), window));
                }
            }
            if candidates.is_empty() {
                break 'try_replace false;
            }

            let order: Vec<usize> = if crate::replace::replace::outgoing_gen_mode() {
                let left = &output[output.len().saturating_sub(FEATURE_CTX)..];
                let feats: Vec<crate::replace::ranking::CandFeatures> = candidates
                    .iter()
                    .map(|(idxs, win)| {
                        let skip: std::collections::HashSet<usize> = idxs.iter().copied().collect();
                        let right: Vec<[u16; 3]> = (0..rg.len())
                            .filter(|m| !skip.contains(m))
                            .take(FEATURE_CTX)
                            .map(|m| rg[m])
                            .collect();
                        crate::replace::replace::cand_features(win, left, &right)
                    })
                    .collect();
                let mut ord = crate::replace::ranking::outgoing().order(&feats);
                ord.sort_by(|&a, &b| candidates[b].1.len().cmp(&candidates[a].1.len()));
                ord
            } else {
                (0..candidates.len()).rev().collect()
            };

            let negated_wires = |win: &[[u16; 3]]| -> Vec<u16> {
                let mut wires = Vec::new();
                for gate in win {
                    for &wire in &[gate[0], gate[1], gate[2]] {
                        if negation_mask[wire as usize] == 1 && !wires.contains(&wire) {
                            wires.push(wire);
                        }
                    }
                }
                wires
            };
            let shot_negated = crate::replace::replace::outgoing_gen_mode()
                && (negation_mask[shot_rg[0] as usize] == 1
                    || negation_mask[shot_rg[1] as usize] == 1
                    || negation_mask[shot_rg[2] as usize] == 1);
            let absorb_nots = crate::replace::replace::outgoing_gen_mode()
                && std::env::var("ABSORB_NOTS").is_ok();

            let mut expanded: Option<(Vec<usize>, Vec<[u16; 3]>, Vec<[u16; 3]>, Vec<u16>)> = None;
            for &candidate_idx in order.iter().take(MAX_EXPAND_ATTEMPTS) {
                let (idxs, window) = &candidates[candidate_idx];
                if !absorb_nots {
                    let dirty_control = window.iter().any(|gate| {
                        negation_mask[gate[1] as usize] == 1 || negation_mask[gate[2] as usize] == 1
                    });
                    if dirty_control {
                        continue;
                    }
                }
                let neg = if absorb_nots {
                    negated_wires(window)
                } else {
                    Vec::new()
                };
                if shot_negated && !neg.is_empty() {
                    continue;
                }
                if let Some(expansion) =
                    expand_curated_frozen_neg(window, n, db, &neg, LookupScope::CuratedThenRegular)
                {
                    if expansion.len() >= 3 {
                        expanded = Some((idxs.clone(), window.clone(), expansion, neg));
                        break;
                    }
                }
            }

            let (consumed_indices, window, expansion, picked_negated) = match expanded {
                Some(x) => x,
                None => break 'try_replace false,
            };
            let consumed = window.len();
            if !picked_negated.is_empty() {
                UNCLEAN_EXPANSIONS.fetch_add(1, Ordering::Relaxed);
            }
            for &wire in &picked_negated {
                negation_mask[wire as usize] = 0;
            }
            CURATED_REPLACEMENTS_MADE.fetch_add(1, Ordering::Relaxed);
            SAMF_HIDE_ELIGIBLE_EXPANSIONS.fetch_add(1, Ordering::Relaxed);
            // Distinct-wire counts of the chosen outgoing window vs the expansion.
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
            let event_tag = if track {
                let mut win_tags = Vec::with_capacity(consumed);
                win_tags.push(shot_tag);
                win_tags.extend(consumed_indices.iter().map(|&idx| remaining_tags[idx]));
                crate::replace::replace::new_gate_tag(&win_tags)
            } else {
                TAG_NEW
            };

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
                        compress_curated_frozen(&window, n, db, LookupScope::CuratedThenRegular)
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

            let rec_start = output.len().saturating_sub(from_output);
            match tuck {
                Some((_score, swap_lo, swap_hi, neg_type, samf, repl)) => {
                    // The context window (output tail + expansion tail + samf[0..3]) becomes
                    // `repl`; everything before it is unchanged. Emit the SAMF suffix up to its
                    // final gate, then keep that final gate as the next materialized shot so the
                    // collision game continues from the inserted SAMF.
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
                    crate::replace::replace::REC_PASS.load(Ordering::Relaxed),
                    rs,
                    rec_end,
                    consumed,
                    &ws,
                );
            }
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
                    output_tags.push(shot_tag);
                }
            } else {
                emit_relabelled_gate(shot, &mut output, &t_list, &mut negation_mask, n);
                if track {
                    output_tags.resize(
                        output.len() - 1,
                        crate::replace::replace::new_gate_tag(std::slice::from_ref(&shot_tag)),
                    );
                    output_tags.push(shot_tag);
                }
            }
        } else {
            replacements += 1;
            if options.replacement_budget > 0 && replacements >= options.replacement_budget {
                flush_rest(
                    &mut output,
                    &mut output_tags,
                    &mut remaining,
                    &mut remaining_tags,
                    &mut materialized_shot,
                    materialized_shot_tag,
                    &t_list,
                    &mut negation_mask,
                );
                break;
            }
        }
    }

    debug_assert!(!track || output.len() == output_tags.len());
    (
        output,
        t_list,
        negation_mask,
        compressions,
        output_tags,
        replacements,
    )
}

fn collision_game_repeated_core(
    input: &[[u16; 3]],
    n: usize,
    db: &FrozenDb,
    gates_ahead_expand: usize,
    gates_ahead_samf: usize,
    type_attempts: usize,
    shooting_times: usize,
    input_tags: &[u32],
    options: CollisionPassOptions,
) -> (
    Vec<[u16; 3]>,
    Transpositions,
    Vec<u8>,
    usize,
    Vec<u32>,
    usize,
) {
    let passes = shooting_times.max(1);
    let mut input_gates = input.to_vec();
    let mut tags = input_tags.to_vec();
    let mut total_t = Transpositions {
        transpositions: Vec::new(),
    };
    let mut total_neg = vec![0u8; n];
    let mut total_compressions = 0usize;
    let mut total_replacements = 0usize;
    let mut remaining_budget = options.replacement_budget;

    for pass_idx in 0..passes {
        let pass_options = CollisionPassOptions {
            anchor: if pass_idx == 0 { options.anchor } else { None },
            replacement_budget: remaining_budget,
        };
        let (out, t_pass, neg_pass, compressions, out_tags, replacements) = collision_game_core(
            &input_gates,
            n,
            db,
            gates_ahead_expand,
            gates_ahead_samf,
            type_attempts,
            &tags,
            pass_options,
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
        total_replacements += replacements;
        input_gates = out;
        tags = out_tags;
        if options.replacement_budget > 0 {
            remaining_budget = remaining_budget.saturating_sub(replacements);
            if remaining_budget == 0 {
                break;
            }
        }
    }

    (
        input_gates,
        total_t,
        total_neg,
        total_compressions,
        tags,
        total_replacements,
    )
}

// Standalone collision game: run the core, then undo its accumulated SAMFs.
pub fn collision_game(
    circuit: &mut CircuitSeq,
    n: usize,
    db: &FrozenDb,
    gates_ahead_expand: usize,
    gates_ahead_samf: usize,
    type_attempts: usize,
    shooting_times: usize,
) -> usize {
    let (mut output, t_list, negation_mask, compressions, _, _) = collision_game_repeated_core(
        &circuit.gates,
        n,
        db,
        gates_ahead_expand,
        gates_ahead_samf,
        type_attempts,
        shooting_times,
        &[],
        CollisionPassOptions::default(),
    );
    apply_unsamf(&mut output, &t_list, &negation_mask, n, db);
    circuit.gates = output;
    compressions
}

// Core of shuffled_shoot_then_samf: each collision-game pass optionally runs one curated expansion
// loop pass, then `collision_rounds` collision games followed by one per-gate SAMF insertion. All accumulated
// SAMF state is returned WITHOUT undoing.
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
    collision_rounds: usize,
    expansion_game: bool,
    db: &FrozenDb,
    input_tags: &[u32],
) -> (
    Vec<[u16; 3]>,
    Transpositions,
    Vec<u8>,
    usize,
    Vec<u32>,
    usize,
) {
    let passes = shooting_times.max(1);
    let collision_passes = collision_rounds.max(1);
    let mut input_gates = input.to_vec();
    let mut tags = input_tags.to_vec();
    let mut total_t = Transpositions {
        transpositions: Vec::new(),
    };
    let mut total_neg = vec![0u8; n];
    let mut total_compressions = 0usize;
    let mut total_replacements = 0usize;

    for _ in 0..passes {
        if expansion_game {
            let pair_mode = ExpandPairMode::Curated;
            let current = CircuitSeq { gates: input_gates };
            let before = current.gates.len();
            let event_tag = if tags.is_empty() {
                crate::replace::replace::TAG_NEW
            } else {
                crate::replace::replace::new_gate_tag(&tags)
            };
            input_gates = expand_once_scored(&current, n, db, &pair_mode).gates;
            if !tags.is_empty() {
                tags = std::iter::repeat(event_tag)
                    .take(input_gates.len())
                    .collect();
            }
            println!(
                "  Expansion game: {} -> {} gates (scored one expand loop pass)",
                before,
                input_gates.len()
            );
        }

        let (out_a, t_a, neg_a, compressions, tags_a, replacements) = collision_game_repeated_core(
            &input_gates,
            n,
            db,
            gates_ahead_expand,
            gates_ahead_samf,
            type_attempts,
            collision_passes,
            &tags,
            CollisionPassOptions::default(),
        );
        let (out_b, t_b, neg_b, tags_b) = insert_m_samfs_core(&out_a, n, m, x, &tags_a);

        // Combined permutation for this collision-game pass: collision games first, then insertion.
        let t_pass = t_a.concat(&t_b);
        // Combined final negation for this collision-game pass: insertion's own negation plus the
        // collision games' negation transported through the insertion permutation.
        let mut neg_pass = neg_b;
        for w in 0..n {
            if neg_a[w] == 1 {
                let cw = t_b.evaluate(w as u16) as usize;
                neg_pass[cw] ^= 1;
            }
        }

        // Fold this collision-game pass into the total pending SAMF/NOT state.
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
        total_replacements += replacements;
        input_gates = out_b;
        tags = tags_b;
    }

    (
        input_gates,
        total_t,
        total_neg,
        total_compressions,
        tags,
        total_replacements,
    )
}

// Run exactly one low-generation shooting pass and immediately resolve its SAMF state. This is
// the Stage-B-safe variant: no ledger is carried into the next low-gen pass.
pub fn shuffled_shoot_then_samf_stage_b_pass(
    circuit: &mut CircuitSeq,
    n: usize,
    m: usize,
    x: usize,
    gates_ahead_expand: usize,
    gates_ahead_samf: usize,
    type_attempts: usize,
    collision_rounds: usize,
    expansion_game: bool,
    db: &FrozenDb,
    tags: &mut Vec<u32>,
    anchor: usize,
    replacement_budget: usize,
) -> StageBPassResult {
    let track = !tags.is_empty();
    let mut input_gates = circuit.gates.clone();
    let mut input_tags = tags.clone();

    if expansion_game {
        let pair_mode = ExpandPairMode::Curated;
        let current = CircuitSeq { gates: input_gates };
        let before = current.gates.len();
        let event_tag = if input_tags.is_empty() {
            crate::replace::replace::TAG_NEW
        } else {
            crate::replace::replace::new_gate_tag(&input_tags)
        };
        input_gates = expand_once_scored(&current, n, db, &pair_mode).gates;
        if !input_tags.is_empty() {
            input_tags = std::iter::repeat(event_tag)
                .take(input_gates.len())
                .collect();
        }
        println!(
            "  Expansion game: {} -> {} gates (scored one expand loop pass)",
            before,
            input_gates.len()
        );
    }

    let (out_a, t_a, neg_a, hidden_samfs, tags_a, replacements) = collision_game_repeated_core(
        &input_gates,
        n,
        db,
        gates_ahead_expand,
        gates_ahead_samf,
        type_attempts,
        collision_rounds.max(1),
        &input_tags,
        CollisionPassOptions {
            anchor: Some(anchor),
            replacement_budget,
        },
    );
    let (mut out_b, t_b, neg_b, mut tags_b) = insert_m_samfs_core(&out_a, n, m, x, &tags_a);

    let t_pass = t_a.concat(&t_b);
    let mut neg_pass = neg_b;
    for w in 0..n {
        if neg_a[w] == 1 {
            let cw = t_b.evaluate(w as u16) as usize;
            neg_pass[cw] ^= 1;
        }
    }

    if track {
        apply_unsamf_tagged(&mut out_b, &t_pass, &neg_pass, n, db, &mut tags_b);
        *tags = tags_b;
    } else {
        apply_unsamf(&mut out_b, &t_pass, &neg_pass, n, db);
    }
    circuit.gates = out_b;
    StageBPassResult {
        hidden_samfs,
        replacements,
    }
}

// Run each collision-game pass with an optional single curated expansion loop pass, then `collision_rounds`
// collision games followed by one per-gate SAMF insertion. Perform a SINGLE unsamf at the very
// end. Returns the collision game's compression count across all passes.
pub fn shuffled_shoot_then_samf(
    circuit: &mut CircuitSeq,
    n: usize,
    m: usize,
    x: usize,
    gates_ahead_expand: usize,
    gates_ahead_samf: usize,
    type_attempts: usize,
    shooting_times: usize,
    collision_rounds: usize,
    expansion_game: bool,
    db: &FrozenDb,
    tags: &mut Vec<u32>,
) -> usize {
    let track = !tags.is_empty();
    let (mut out, t_round, neg_round, compressions, mut out_tags, _) =
        shuffled_shoot_then_samf_core(
            &circuit.gates,
            n,
            m,
            x,
            gates_ahead_expand,
            gates_ahead_samf,
            type_attempts,
            shooting_times,
            collision_rounds,
            expansion_game,
            db,
            tags,
        );
    if track {
        apply_unsamf_tagged(&mut out, &t_round, &neg_round, n, db, &mut out_tags);
        *tags = out_tags;
    } else {
        apply_unsamf(&mut out, &t_round, &neg_round, n, db);
    }
    circuit.gates = out;
    compressions
}

#[cfg(test)]
mod reversed_samf_tests {
    use super::{
        SWAP_3W, SWAP_4W, SWAP_N1_3W, SWAP_N1_4W, SWAP_N2_3W, SWAP_N2_4W, SWAP_N12_3W, SWAP_N12_4W,
        Transpositions, neg_flips, random_neg_type, shoot_gate_to_first_collision,
        shoot_materialized_gate_to_first_collision,
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

    fn g57_evolutions(
        gates: &[[u16; 3]],
        secret_count: usize,
        randomness_count: usize,
        encode: impl Fn(usize, usize) -> usize,
    ) -> Vec<Vec<Vec<usize>>> {
        (0..secret_count)
            .map(|secret| {
                (0..randomness_count)
                    .map(|randomness| {
                        let mut state = encode(secret, randomness);
                        let mut evolution = vec![state];
                        for &gate in gates {
                            state = crate::circuit::circuit::Gate::evaluate_index(state, gate);
                            evolution.push(state);
                        }
                        evolution
                    })
                    .collect()
            })
            .collect()
    }

    fn single_probe_leaks(evolutions: &[Vec<Vec<usize>>], wires: usize) -> usize {
        let points = evolutions[0][0].len() * wires;
        (0..points)
            .filter(|&point| {
                let observed =
                    |evolution: &[usize]| (evolution[point / wires] >> (point % wires)) & 1;
                let base: usize = evolutions[0]
                    .iter()
                    .map(|evolution| observed(evolution))
                    .sum();
                evolutions[1..].iter().any(|samples| {
                    samples
                        .iter()
                        .map(|evolution| observed(evolution))
                        .sum::<usize>()
                        != base
                })
            })
            .count()
    }

    fn same_prefix_pair_leaks(evolutions: &[Vec<Vec<usize>>], wires: usize) -> usize {
        let prefixes = evolutions[0][0].len();
        let mut leaks = 0usize;
        for prefix in 0..prefixes {
            for left in 0..wires {
                for right in left + 1..wires {
                    let histogram = |samples: &[Vec<usize>]| {
                        let mut counts = [0usize; 4];
                        for evolution in samples {
                            let value = ((evolution[prefix] >> left) & 1)
                                | (((evolution[prefix] >> right) & 1) << 1);
                            counts[value] += 1;
                        }
                        counts
                    };
                    let base = histogram(&evolutions[0]);
                    if evolutions[1..]
                        .iter()
                        .any(|samples| histogram(samples) != base)
                    {
                        leaks += 1;
                    }
                }
            }
        }
        leaks
    }

    #[test]
    fn legacy_samf_pool_masking_survey() {
        let pools_3w = [SWAP_3W, SWAP_N1_3W, SWAP_N2_3W, SWAP_N12_3W];
        let pools_4w = [SWAP_4W, SWAP_N1_4W, SWAP_N2_4W, SWAP_N12_4W];
        let mut total = 0usize;
        let mut first_order_leaky = 0usize;
        let mut second_order_leaky = 0usize;

        for pool in pools_3w {
            for &gates in pool {
                total += 1;
                // Wires 1 and 2 are complementary two-share carriers; wire 0
                // is random auxiliary input.
                let two_share = g57_evolutions(gates, 2, 4, |secret, randomness| {
                    let mask = randomness & 1;
                    let helper = (randomness >> 1) & 1;
                    helper | (mask << 1) | ((mask ^ secret) << 2)
                });
                first_order_leaky += (single_probe_leaks(&two_share, 3) > 0) as usize;

                // Worst-case three-share placement: the template's auxiliary
                // wire 0 is itself the third carrier.
                let three_share = g57_evolutions(gates, 2, 4, |secret, randomness| {
                    let share1 = randomness & 1;
                    let share2 = (randomness >> 1) & 1;
                    (secret ^ share1 ^ share2) | (share1 << 1) | (share2 << 2)
                });
                second_order_leaky += (same_prefix_pair_leaks(&three_share, 3) > 0) as usize;
            }
        }
        for pool in pools_4w {
            for &gates in pool {
                total += 1;
                let two_share = g57_evolutions(gates, 2, 8, |secret, randomness| {
                    let mask = randomness & 1;
                    let helper0 = (randomness >> 1) & 1;
                    let helper1 = (randomness >> 2) & 1;
                    helper0 | (mask << 1) | ((mask ^ secret) << 2) | (helper1 << 3)
                });
                first_order_leaky += (single_probe_leaks(&two_share, 4) > 0) as usize;

                // Test both possible helper positions as the third carrier.
                let leaks_for_placement = |third_on_wire3: bool| {
                    let three_share = g57_evolutions(gates, 2, 8, |secret, randomness| {
                        let share1 = randomness & 1;
                        let share2 = (randomness >> 1) & 1;
                        let helper = (randomness >> 2) & 1;
                        let third = secret ^ share1 ^ share2;
                        if third_on_wire3 {
                            helper | (share1 << 1) | (share2 << 2) | (third << 3)
                        } else {
                            third | (share1 << 1) | (share2 << 2) | (helper << 3)
                        }
                    });
                    same_prefix_pair_leaks(&three_share, 4) > 0
                };
                second_order_leaky +=
                    (leaks_for_placement(false) || leaks_for_placement(true)) as usize;
            }
        }

        println!(
            "legacy SAMF masking survey: templates={total} first-order-leaky={first_order_leaky} second-order-leaky={second_order_leaky}"
        );
        // Keep the exact survey result as a regression baseline. Every legacy
        // template has a leaky prefix in these legal worst-case placements.
        // The native masked SAMF uses a dedicated independent helper and has
        // zero leaks in the corresponding tests.
        assert_eq!(total, 77);
        assert_eq!(first_order_leaky, total);
        assert_eq!(second_order_leaky, total);
    }
}
