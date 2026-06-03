//! `shoot` and `shuffle` transforms (DB-free) must preserve circuit semantics.
//!
//! `compress`/`sss` are intentionally NOT tested here: they read the curated +
//! sharded LMDB `./db`, which isn't available in CI.

use lmdb::Environment;
use local_mixing::circuit::CircuitSeq;
use local_mixing::random::random_data::{random_circuit, shoot_random_gate};
use local_mixing::replace::transpositions::{
    insert_wire_m_samfs_every_x, insert_wire_shuffles_knuth, insert_wire_shuffles_simple,
    insert_wire_shuffles_x, shuffled_shoot_then_samf, shuffled_shooting_game,
};
use std::sync::atomic::{AtomicUsize, Ordering};

// A throwaway, empty LMDB environment. The shuffle functions take env + DB slices for
// end-of-circuit SAMF compression, but with empty DB slices they never touch the env and
// fall back to appending SAMFs verbatim — keeping these tests DB-free while still
// exercising the undo restructure.
fn empty_env() -> Environment {
    static CTR: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "lm_test_env_{}_{}",
        std::process::id(),
        CTR.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("create temp env dir");
    Environment::new().open(&dir).expect("open temp lmdb env")
}

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
fn assert_shuffle_preserves(n: usize, trials: usize, shuffle: impl Fn(&mut CircuitSeq, usize)) {
    for _ in 0..trials {
        let original = random_circuit(n, 40);
        let mut shuffled = original.clone();
        shuffle(&mut shuffled, n);
        assert!(
            shuffled.probably_equal(&original, n, 200).is_ok(),
            "wire shuffle changed circuit semantics (n={n}) \
             (regression: dropped fixed-point negation / bad undo restructure)"
        );
        assert!(
            shuffled.gates.len() >= original.gates.len(),
            "a shuffle should insert gates, not remove them (n={n})"
        );
    }
}

// Exercise every shuffle across a range of wire counts, all DB-free (empty DB slices ->
// SAMFs appended verbatim, still exercising the reversed-gadget undo restructure).
const WIRE_COUNTS: [usize; 4] = [4, 8, 16, 24];

#[test]
fn shuffle_simple_preserves_equivalence() {
    let env = empty_env();
    for n in WIRE_COUNTS {
        assert_shuffle_preserves(n, 40, |c, n| {
            insert_wire_shuffles_simple(c, n, &env, &[], &[])
        });
    }
}

#[test]
fn shuffle_knuth_preserves_equivalence() {
    let env = empty_env();
    for n in WIRE_COUNTS {
        assert_shuffle_preserves(n, 40, |c, n| {
            insert_wire_shuffles_knuth(c, n, &env, &[], &[])
        });
    }
}

#[test]
fn shuffle_x_preserves_equivalence() {
    let env = empty_env();
    for n in WIRE_COUNTS {
        assert_shuffle_preserves(n, 40, |c, n| {
            insert_wire_shuffles_x(c, n, 2, &env, &[], &[])
        });
    }
}

#[test]
fn shuffle_m_samfs_every_x_preserves_equivalence() {
    let env = empty_env();
    for n in WIRE_COUNTS {
        assert_shuffle_preserves(n, 40, |c, n| {
            insert_wire_m_samfs_every_x(c, n, 4, 3, &env, &[], &[])
        });
    }
}

#[test]
fn shuffled_shooting_game_preserves_equivalence() {
    let env = empty_env();
    for n in WIRE_COUNTS {
        assert_shuffle_preserves(n, 40, |c, n| {
            shuffled_shooting_game(c, n, &env, &[], &[], 4);
        });
    }
}

#[test]
fn shuffled_shoot_then_samf_preserves_equivalence() {
    // Merged shooting-game + SAMF-insertion with a single combined unsamf.
    let env = empty_env();
    for n in WIRE_COUNTS {
        assert_shuffle_preserves(n, 40, |c, n| {
            shuffled_shoot_then_samf(c, n, 4, 3, 4, &env, &[], &[]);
        });
    }
}
