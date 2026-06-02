//! `to_polynomial` returns one ANF polynomial per wire. Wire `i` starts as the
//! lone monomial `x_i` (bit `1<<i`); the constant `1` is the empty monomial
//! (`0u64`). A gate `[a,b,c]` folds `P_b * NOT(P_c) + 1` into `P_a`.
//!
//! Two related facts these tests pin down:
//!   1. A wire's polynomial can lose its *standalone* `x_i` monomial.
//!   2. Stronger: `x_i` can vanish from *every* monomial of `P_i` — output `i`
//!      becomes fully independent of input `i`. (A single write to wire `i` can
//!      never achieve this; it takes >= 2 writes, hence >= 4 gates on 3 wires.)

use local_mixing::circuit::{CircuitSeq, base_gates};
use local_mixing::random::random_data::random_circuit;

type Polys = Vec<std::collections::HashSet<u64>>;

// Does polys[i] contain the standalone monomial x_i?
fn has_own_var(polys: &Polys, i: usize) -> bool {
    polys[i].contains(&(1u64 << i))
}

// Does x_i appear in ANY monomial of polys[i]? (does output i depend on input i)
fn var_appears(polys: &Polys, i: usize) -> bool {
    polys[i].iter().any(|m| m & (1u64 << i) != 0)
}

// ---- Fact 1: standalone x_i can be dropped ---------------------------------

// Smallest hand-derived witness: P_0 ends up without its standalone x_0 term.
#[test]
fn known_circuit_drops_own_var() {
    let c = CircuitSeq {
        gates: vec![[1, 0, 2], [0, 1, 2]],
    };
    let n = 3;
    let polys = c.to_polynomial(n, 0, c.gates.len());

    assert!(
        !has_own_var(&polys, 0),
        "expected P_0 to be missing standalone x_0, got {:?}",
        polys[0]
    );
    // But x_0 still appears (in the x_0*x_2 term) — not fully independent.
    assert!(
        var_appears(&polys, 0),
        "x_0 should still appear in some term"
    );
    assert!(has_own_var(&polys, 2), "P_2 should still contain x_2");
}

#[test]
fn random_circuit_can_drop_own_var() {
    let n = 8;
    let found = (0..2000).find_map(|_| {
        let c = random_circuit(n, 25);
        let polys = c.to_polynomial(n, 0, c.gates.len());
        (0..n).find(|&i| !has_own_var(&polys, i)).map(|i| (c, i))
    });
    let (c, i) = found.expect("expected a circuit whose P_i drops standalone x_i");
    println!("P_{i} drops standalone x_{i}: {}", c.repr());
    assert!(!has_own_var(&c.to_polynomial(n, 0, c.gates.len()), i));
}

// ---- Fact 2: x_i can vanish from every term of P_i -------------------------

// Minimal witness on 3 wires (4 gates): P_0 collapses to exactly {x_1}, so x_0
// appears in no monomial at all.
#[test]
fn minimal_circuit_var_fully_absent() {
    let c = CircuitSeq {
        gates: vec![[1, 2, 0], [1, 0, 2], [0, 2, 1], [0, 1, 2]],
    };
    let n = 3;
    let polys = c.to_polynomial(n, 0, c.gates.len());

    assert!(
        !var_appears(&polys, 0),
        "expected x_0 absent, got {:?}",
        polys[0]
    );
    assert_eq!(
        polys[0],
        std::collections::HashSet::from([1u64 << 1]),
        "P_0 should be exactly x_1"
    );
}

// No circuit shorter than 4 gates on 3 wires can make any x_i fully absent
// (empirical confirmation of the >=2-writes argument).
#[test]
fn no_var_fully_absent_below_four_gates() {
    let n = 3;
    let gates = base_gates(n);
    for depth in 1..=3usize {
        let total = gates.len().pow(depth as u32);
        for code in 0..total {
            let mut k = code;
            let seq: Vec<[u16; 3]> = (0..depth)
                .map(|_| {
                    let g = gates[k % gates.len()];
                    k /= gates.len();
                    g
                })
                .collect();
            let c = CircuitSeq { gates: seq };
            let polys = c.to_polynomial(n, 0, c.gates.len());
            assert!(
                (0..n).all(|i| var_appears(&polys, i)),
                "found x_i fully absent in only {depth} gates: {}",
                c.repr()
            );
        }
    }
}

#[test]
fn random_circuit_can_drop_var_entirely() {
    // ~1 in 5 random 3-wire circuits has some P_i fully independent of x_i.
    let n = 3;
    let found = (0..2000).find_map(|_| {
        let c = random_circuit(n, 12);
        let polys = c.to_polynomial(n, 0, c.gates.len());
        (0..n).find(|&i| !var_appears(&polys, i)).map(|i| (c, i))
    });
    let (c, i) = found.expect("expected a circuit whose P_i is fully independent of x_i");
    println!("x_{i} fully absent from P_{i}: {}", c.repr());
    assert!(!var_appears(&c.to_polynomial(n, 0, c.gates.len()), i));
}
