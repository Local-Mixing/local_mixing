use super::*;
use crate::circuit::xgate::eval_lanes;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn random_gate(rng: &mut StdRng, wires: u16) -> XGate {
    let target = rng.random_range(0..wires);
    let width = rng.random_range(0..=3usize.min(wires as usize - 1));
    let mut lits = Vec::new();
    while lits.len() < width {
        let w = rng.random_range(0..wires);
        if w != target && !lits.iter().any(|&(x, _)| x == w) {
            lits.push((w, rng.random_bool(0.5)));
        }
    }
    if width == 2 && rng.random_bool(0.5) {
        // A lifted g57 (complemented, mixed polarity).
        let (x, y) = (lits[0].0, lits[1].0);
        return XGate::from_g57([target, x, y]);
    }
    XGate::conj(target, lits).expect("distinct wires never contradict")
}

fn same_function(a: &[XGate], b: &[XGate], n: usize) -> bool {
    (0..(1u64 << n)).all(|input| {
        let mut sa: Vec<u64> = (0..n).map(|w| (input >> w) & 1).collect();
        let mut sb = sa.clone();
        eval_lanes(a.iter(), &mut sa);
        eval_lanes(b.iter(), &mut sb);
        sa.iter().zip(&sb).all(|(x, y)| (x ^ y) & 1 == 0)
    })
}

// Every commuting re-spelling of a circuit must reach the same canonical
// sequence: that is what lets the identity guard recognize a candidate that
// is the window up to reordering. Covers wide conjunctions, CNOTs, X gates
// and lifted g57s, whose commutation includes the opposite-literal exemption.
#[test]
fn canonical_order_is_a_normal_form_over_commuting_respellings() {
    let mut rng = StdRng::seed_from_u64(0x5eed);
    for _ in 0..400 {
        let wires = rng.random_range(3..8u16);
        let len = rng.random_range(1..10usize);
        let gates: Vec<XGate> = (0..len).map(|_| random_gate(&mut rng, wires)).collect();
        let canon = xgate_canonical(&gates);
        assert_eq!(xgate_canonical(&canon), canon, "idempotent");
        assert!(
            same_function(&gates, &canon, wires as usize),
            "function-preserving"
        );
        for _ in 0..8 {
            let mut v = gates.clone();
            for _ in 0..(6 * len * len + 1) {
                if len < 2 {
                    break;
                }
                let i = rng.random_range(0..len - 1);
                if !XGate::collides(&v[i], &v[i + 1]) {
                    v.swap(i, i + 1);
                }
            }
            assert_eq!(xgate_canonical(&v), canon, "respelling {v:?} of {gates:?}");
        }
    }
}

/// The pre-2026-10-11 loop (`Vec::remove` + `Vec::insert`, O(n) per move),
/// kept as the reference the rotation-based walk must reproduce exactly.
fn xgate_canonicalize_reference(gates: &mut Vec<XGate>) {
    for i in 1..gates.len() {
        let mut to_swap: Option<usize> = None;
        let mut j = i;
        while j > 0 {
            j -= 1;
            if XGate::collides(&gates[i], &gates[j]) {
                break;
            } else if xgate_order(&gates[j], &gates[i]) == Ordering::Greater {
                to_swap = Some(j);
            }
        }
        if let Some(pos) = to_swap {
            let g = gates.remove(i);
            gates.insert(pos, g);
        }
    }
}

#[test]
fn rotation_walk_equals_the_remove_insert_reference() {
    let mut rng = StdRng::seed_from_u64(0xca70);
    for round in 0..300 {
        let wires = rng.random_range(2..12u16);
        let len = rng.random_range(0..60usize);
        let gates: Vec<XGate> = (0..len).map(|_| random_gate(&mut rng, wires)).collect();
        let mut fast = gates.clone();
        xgate_canonicalize(&mut fast);
        let mut slow = gates.clone();
        xgate_canonicalize_reference(&mut slow);
        assert_eq!(fast, slow, "round {round}: {gates:?}");
    }
    // a long circuit on few wires: many moves, long commuting runs
    let gates: Vec<XGate> = (0..3000).map(|_| random_gate(&mut rng, 6)).collect();
    let mut fast = gates.clone();
    xgate_canonicalize(&mut fast);
    let mut slow = gates;
    xgate_canonicalize_reference(&mut slow);
    assert_eq!(fast, slow);
}
