//! `shoot` and `shuffle` transforms (DB-free) must preserve circuit semantics.
//!
//! `compress`/`sss` are intentionally NOT tested here: they read the curated +
//! sharded LMDB `./db`, which isn't available in CI.

use local_mixing::circuit::CircuitSeq;
use local_mixing::random::random_data::{random_circuit, shoot_random_gate};
use local_mixing::replace::transpositions::{
    insert_wire_shuffles_knuth, insert_wire_shuffles_simple, insert_wire_shuffles_x,
};

#[test]
fn shoot_preserves_equivalence() {
    // shoot_random_gate only slides gates past non-colliding (commuting)
    // neighbours, so it is equivalence-preserving on every call.
    let n = 32;
    let original = random_circuit(n, 100);
    let mut shot = original.clone();
    shoot_random_gate(&mut shot, 200);
    assert!(
        shot.probably_equal(&original, n, 300).is_ok(),
        "shoot_random_gate must preserve semantics"
    );
}

// A wire shuffle inserts SAMF gadgets that must net out to identity. This was
// previously only ~88% reliable (residual negations on permutation fixed points
// were dropped); it is now deterministic. Run many single-call trials so the
// regression would resurface as a hard failure rather than flakiness.
fn assert_shuffle_preserves(shuffle: impl Fn(&mut CircuitSeq, usize)) {
    let n = 16;
    for _ in 0..100 {
        let original = random_circuit(n, 40);
        let mut shuffled = original.clone();
        shuffle(&mut shuffled, n);
        assert!(
            shuffled.probably_equal(&original, n, 200).is_ok(),
            "wire shuffle changed circuit semantics (regression: dropped fixed-point negation)"
        );
        assert!(
            shuffled.gates.len() >= original.gates.len(),
            "a shuffle should insert gates, not remove them"
        );
    }
}

#[test]
fn shuffle_simple_preserves_equivalence() {
    assert_shuffle_preserves(|c, n| insert_wire_shuffles_simple(c, n));
}

#[test]
fn shuffle_knuth_preserves_equivalence() {
    assert_shuffle_preserves(|c, n| insert_wire_shuffles_knuth(c, n));
}

#[test]
fn shuffle_x_preserves_equivalence() {
    assert_shuffle_preserves(|c, n| insert_wire_shuffles_x(c, n, 2));
}
