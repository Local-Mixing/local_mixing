//! `contiguous_convex` must, for any convex gate set, rearrange the circuit so
//! the selected gates form a contiguous block — using only legal commuting
//! swaps (so the circuit's permutation is preserved). It must never return a
//! range that still straddles a non-member gate.
//!
//! Regression: the old two-pass (leftmost-left / rightmost-right) bubbling
//! stalled on convex sets whose interior non-members needed to exit in the
//! opposite direction, silently returning a non-contiguous range.

use local_mixing::circuit::CircuitSeq;
use local_mixing::random::random_data::{
    contiguous_convex, find_convex_subcircuit_max_gates, find_convex_subcircuit_max_wires,
    is_convex, random_circuit, simple_find_convex_subcircuit,
};

// Run contiguous_convex on a clone and assert the result is a genuine,
// semantics-preserving contiguous arrangement of exactly the selected gates.
fn assert_contiguizes(gates: Vec<[u16; 3]>, ids: &[usize], num_wires: usize) {
    let original = CircuitSeq { gates };
    assert!(
        is_convex(num_wires, &original, ids),
        "test setup: {:?} should be convex for ids {:?}",
        original.gates,
        ids
    );

    // The selected gates as a multiset (sorted) — must be unchanged.
    let mut want: Vec<[u16; 3]> = ids.iter().map(|&i| original.gates[i]).collect();
    want.sort();

    let mut circuit = original.clone();
    let mut sel = ids.to_vec();
    sel.sort();

    let (start, end) = contiguous_convex(&mut circuit, &mut sel, num_wires)
        .expect("convex set must be contiguizable");

    // Block is exactly the selected gates, no straddling non-members.
    assert_eq!(
        end - start + 1,
        ids.len(),
        "block [{},{}] must contain exactly {} gates",
        start,
        end,
        ids.len()
    );
    let mut got: Vec<[u16; 3]> = circuit.gates[start..=end].to_vec();
    got.sort();
    assert_eq!(got, want, "contiguous block must be the selected gates");

    // Rearranging commuting gates must preserve the whole circuit's permutation.
    assert!(
        circuit.probably_equal(&original, num_wires, 400).is_ok(),
        "contiguous_convex must preserve semantics"
    );
}

#[test]
fn counterexample_opposite_directions_a() {
    // Interior non-members must exit in opposite directions; old two-pass stalled.
    assert_contiguizes(
        vec![[1, 2, 3], [3, 2, 0], [4, 2, 0], [3, 0, 4], [4, 0, 1]],
        &[0, 3],
        5,
    );
}

#[test]
fn counterexample_opposite_directions_b() {
    assert_contiguizes(vec![[2, 3, 4], [3, 4, 2], [1, 0, 4], [3, 5, 1]], &[0, 3], 6);
}

#[test]
fn counterexample_interior_block() {
    assert_contiguizes(
        vec![
            [3, 5, 2],
            [4, 2, 1],
            [1, 3, 5],
            [4, 5, 2],
            [1, 2, 3],
            [0, 2, 4],
            [1, 3, 4],
        ],
        &[0, 1, 6],
        6,
    );
}

// For every set the searches actually return, contiguous_convex must succeed
// (return Some), produce a genuine contiguous block, and preserve semantics.
// A None here would mean a convex set the fixpoint failed to drain.
#[test]
fn searches_always_contiguize() {
    let n = 16;
    let mut rng = rand::rng();

    for _ in 0..3000 {
        let circuit = random_circuit(n, 200);

        let runs: [(Vec<usize>, &str); 3] = [
            (
                simple_find_convex_subcircuit(n, &circuit, &mut rng).0,
                "simple",
            ),
            (
                find_convex_subcircuit_max_wires(21, n, &circuit, &mut rng).0,
                "max_wires",
            ),
            (
                find_convex_subcircuit_max_gates(21, n, &circuit, &mut rng).0,
                "max_gates",
            ),
        ];

        for (ids, label) in runs {
            if ids.len() < 2 {
                continue; // search gave up; nothing to contiguize
            }
            assert!(
                is_convex(n, &circuit, &ids),
                "[{label}] search returned a non-convex set {ids:?}"
            );

            let original = circuit.clone();
            let mut work = circuit.clone();
            let mut sel = ids.clone();
            sel.sort();

            let res = contiguous_convex(&mut work, &mut sel, n);
            let (start, end) = res.unwrap_or_else(|| {
                panic!("[{label}] convex set {ids:?} failed to contiguize (None)")
            });

            assert_eq!(
                end - start + 1,
                ids.len(),
                "[{label}] block [{start},{end}] straddles non-members (ids {ids:?})"
            );
            assert!(
                work.probably_equal(&original, n, 200).is_ok(),
                "[{label}] contiguize changed circuit semantics"
            );
        }
    }
}
