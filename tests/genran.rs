//! `genran` core: `random_circuit` should produce well-formed circuits.

use local_mixing::random::random_data::random_circuit;

#[test]
fn gate_count_matches_request() {
    for (n, m) in [(8usize, 30usize), (32, 100), (64, 250)] {
        let c = random_circuit(n, m);
        assert_eq!(c.gates.len(), m, "expected {m} gates for n={n}");
    }
}

#[test]
fn wires_are_in_bounds_and_distinct() {
    let n = 32;
    let c = random_circuit(n, 200);
    for g in &c.gates {
        assert!(
            g.iter().all(|&w| (w as usize) < n),
            "gate {g:?} uses a wire >= n ({n})"
        );
        assert!(
            g[0] != g[1] && g[1] != g[2] && g[0] != g[2],
            "gate {g:?} has non-distinct wires"
        );
    }
}

#[test]
fn zero_gates_is_empty() {
    assert!(random_circuit(16, 0).gates.is_empty());
}
