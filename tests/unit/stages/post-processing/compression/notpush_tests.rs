use super::*;
use crate::circuit::xgate::{XGate, eval_lanes};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn equal_on(a: &[XGate], b: &[XGate], wires: usize) -> bool {
    let mut rng = StdRng::seed_from_u64(0x1D07);
    for _ in 0..128 {
        let mut sa: Vec<u64> = (0..wires).map(|_| rng.random::<u64>()).collect();
        let mut sb = sa.clone();
        eval_lanes(a.iter(), &mut sa);
        eval_lanes(b.iter(), &mut sb);
        if sa != sb {
            return false;
        }
    }
    true
}

#[test]
fn absorbs_a_not_into_a_downstream_control() {
    // NOT on wire 1, then CNOT(0 ^= 1): the NOT should vanish and the CNOT's
    // control flip to negative, with one trailing NOT restoring wire 1.
    let gates = vec![XGate::x_gate(1), XGate::cnot(0, 1)];
    let (out, rep) = push_nots(&gates, 4);
    assert_eq!(rep.nots_absorbed, 1);
    assert_eq!(rep.lits_flipped, 1);
    assert_eq!(rep.trailing_nots, 1);
    // CNOT with negative control, then the trailing NOT on wire 1.
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].target, 0);
    assert_eq!(out[0].ctrls.as_slice(), &[(1, false)]);
    assert_eq!(out[1], XGate::x_gate(1));
    assert!(equal_on(&gates, &out, 4));
}

#[test]
fn drops_never_firing_gates() {
    // comp = true with empty controls never fires.
    let mut noop = XGate::x_gate(2);
    noop.comp = true;
    let gates = vec![XGate::cnot(0, 1), noop, XGate::cnot(2, 0)];
    let (out, rep) = push_nots(&gates, 4);
    assert_eq!(rep.noops_dropped, 1);
    assert_eq!(rep.nots_absorbed, 0);
    assert_eq!(out.len(), 2);
    assert!(equal_on(&gates, &out, 4));
}

#[test]
fn two_nots_on_one_wire_cancel() {
    let gates = vec![XGate::x_gate(0), XGate::cnot(1, 0), XGate::x_gate(0)];
    let (out, rep) = push_nots(&gates, 4);
    assert_eq!(rep.nots_absorbed, 2);
    assert_eq!(rep.trailing_nots, 0); // even parity, no residual NOT
    // The CNOT sits between the two NOTs, so it is reached with parity 1 and
    // its control flips to negative (1 ^= !x0); both NOTs are gone.
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].ctrls.as_slice(), &[(0, false)]);
    assert!(equal_on(&gates, &out, 4));
}

#[test]
fn random_circuits_preserve_function_and_shrink() {
    let mut rng = StdRng::seed_from_u64(0x_D07_u64);
    let mut total_in = 0usize;
    let mut total_out = 0usize;
    for _ in 0..400 {
        let width = rng.random_range(3..=10usize);
        let n = rng.random_range(1..=40usize);
        let mut gates: Vec<XGate> = Vec::new();
        for _ in 0..n {
            let t = rng.random_range(0..width) as u16;
            let roll = rng.random_range(0u32..10);
            if roll < 3 {
                // a bare NOT (always fires)
                gates.push(XGate::x_gate(t));
            } else if roll == 3 {
                // a never-firing no-op
                let mut g = XGate::x_gate(t);
                g.comp = true;
                gates.push(g);
            } else {
                // a conjunction gate on 1-2 distinct controls != t
                let k = rng.random_range(1..=2usize);
                let mut lits: Vec<(u16, bool)> = Vec::new();
                while lits.len() < k {
                    let w = rng.random_range(0..width) as u16;
                    if w != t && !lits.iter().any(|&(x, _)| x == w) {
                        lits.push((w, rng.random_bool(0.5)));
                    }
                }
                if let Some(mut g) = XGate::conj(t, lits) {
                    g.comp = rng.random_bool(0.5);
                    gates.push(g);
                }
            }
        }
        let (out, _rep) = push_nots(&gates, width);
        assert!(
            equal_on(&gates, &out, width),
            "push_nots changed the function"
        );
        // idempotent: a second pass does nothing but re-emit the tail NOTs
        let (out2, rep2) = push_nots(&out, width);
        assert_eq!(rep2.nots_absorbed, rep2.trailing_nots as u64);
        assert_eq!(rep2.noops_dropped, 0);
        assert!(equal_on(&out, &out2, width));
        total_in += gates.len();
        total_out += out.len();
    }
    assert!(total_out <= total_in, "NOT-pushing never grows the circuit");
}
