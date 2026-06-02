//! Core circuit primitives: serialization round-trip and the equivalence checker.

use local_mixing::circuit::CircuitSeq;
use local_mixing::random::random_data::random_circuit;

#[test]
fn repr_from_string_roundtrip() {
    for (n, m) in [(8usize, 20usize), (32, 100), (64, 50)] {
        let c = random_circuit(n, m);
        let back = CircuitSeq::from_string(&c.repr());
        assert_eq!(
            c, back,
            "repr/from_string round-trip changed the circuit (n={n}, m={m})"
        );
    }
}

#[test]
fn probably_equal_is_reflexive() {
    let c = random_circuit(32, 100);
    assert!(
        c.probably_equal(&c, 32, 200).is_ok(),
        "a circuit should be equivalent to itself"
    );
}

#[test]
fn probably_equal_detects_a_real_difference() {
    // One controlled gate vs. the empty (identity) circuit: they disagree on the
    // ~1/4 of inputs where both controls are set, so 500 samples detect it ~certainly.
    let one_gate = CircuitSeq {
        gates: vec![[0, 1, 2]],
    };
    let identity = CircuitSeq { gates: vec![] };
    assert!(
        one_gate.probably_equal(&identity, 8, 500).is_err(),
        "a non-trivial gate must not look equivalent to the identity"
    );
}
