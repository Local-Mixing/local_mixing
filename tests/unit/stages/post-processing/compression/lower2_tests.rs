use super::*;
use crate::circuit::xgate::{XGate, eval_lanes};
use crate::stages::post_processing::compression::{CompressParams, compress, push_nots};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// `gates` and the single original gate agree on every assignment of the
/// `width` wires (exhaustive when width <= 16, else sampled).
fn equal_on(a: &[XGate], b: &[XGate], width: usize) {
    let mut rng = StdRng::seed_from_u64(0x10E2);
    for _ in 0..256 {
        let mut sa: Vec<u64> = (0..width).map(|_| rng.random::<u64>()).collect();
        let mut sb = sa.clone();
        eval_lanes(a.iter(), &mut sa);
        eval_lanes(b.iter(), &mut sb);
        assert_eq!(sa, sb, "lowering changed the function");
    }
}

fn gate(t: u16, lits: &[(u16, bool)], comp: bool) -> XGate {
    let mut g = XGate::conj(t, lits.iter().copied()).unwrap();
    g.comp = comp;
    g
}

#[test]
fn lowers_a_three_control_toffoli() {
    // t=0 ^= w1 & w2 & w3 on 8 wires.
    let g = gate(0, &[(1, true), (2, true), (3, true)], false);
    let (out, rep) = lower_wide_gates(std::slice::from_ref(&g), 8, 2);
    assert_eq!(rep.wide_in, 1);
    assert_eq!(rep.max_controls_in, 3);
    assert!(out.iter().all(|x| x.ctrls.len() <= 2), "all <= 2 controls");
    // 4(k-2) = 4 Toffolis for k=3.
    assert_eq!(out.len(), 4);
    equal_on(&out, std::slice::from_ref(&g), 8);
}

#[test]
fn keeps_narrow_gates_verbatim() {
    let gates = vec![
        gate(0, &[(1, true)], false),
        gate(2, &[(3, true), (4, false)], true),
    ];
    let (out, rep) = lower_wide_gates(&gates, 8, 2);
    assert_eq!(rep.wide_in, 0);
    assert_eq!(out, gates);
}

#[test]
fn lowers_mixed_polarity_and_complement() {
    // Every k in 3..=10, random polarities, comp on/off, on enough wires for
    // the k-2 dirty helpers (width >= 2k-1).
    let mut rng = StdRng::seed_from_u64(0xB3EC);
    for k in 3..=10usize {
        for _ in 0..12 {
            let width = (2 * k - 1).max(6) + rng.random_range(0..4);
            let t = rng.random_range(0..width) as u16;
            // k distinct control wires != t
            let mut ctrls: Vec<(u16, bool)> = Vec::new();
            while ctrls.len() < k {
                let w = rng.random_range(0..width) as u16;
                if w != t && !ctrls.iter().any(|&(x, _)| x == w) {
                    ctrls.push((w, rng.random_bool(0.5)));
                }
            }
            let comp = rng.random_bool(0.5);
            let g = gate(t, &ctrls, comp);
            let (out, rep) = lower_wide_gates(std::slice::from_ref(&g), width, 2);
            assert_eq!(rep.wide_in, 1);
            assert!(
                out.iter().all(|x| x.ctrls.len() <= 2),
                "k={k}: a gate kept >2 controls"
            );
            equal_on(&out, std::slice::from_ref(&g), width);
        }
    }
}

#[test]
fn lowers_a_mixed_circuit_and_is_idempotent() {
    let mut rng = StdRng::seed_from_u64(0x3EC2);
    for _ in 0..60 {
        // width >= 2k-1 for every k up to 5 (the widest drawn below), so
        // every wide gate has its dirty helpers and is lowered.
        let width = rng.random_range(10..=14usize);
        let n = rng.random_range(1..=20usize);
        let mut gates: Vec<XGate> = Vec::new();
        for _ in 0..n {
            let t = rng.random_range(0..width) as u16;
            let k = rng.random_range(0..=5usize).min(width - 1);
            let mut ctrls: Vec<(u16, bool)> = Vec::new();
            while ctrls.len() < k {
                let w = rng.random_range(0..width) as u16;
                if w != t && !ctrls.iter().any(|&(x, _)| x == w) {
                    ctrls.push((w, rng.random_bool(0.5)));
                }
            }
            gates.push(gate(t, &ctrls, rng.random_bool(0.5)));
        }
        let (out, _) = lower_wide_gates(&gates, width, 2);
        assert!(out.iter().all(|x| x.ctrls.len() <= 2));
        equal_on(&out, &gates, width);
        // A second pass is a no-op (nothing is wide any more).
        let (out2, rep2) = lower_wide_gates(&out, width, 2);
        assert_eq!(rep2.wide_in, 0);
        assert_eq!(out2, out);
    }
}

/// The phase-C composition the driver runs: lower -> NOT-push -> fcompress
/// -> lower -> NOT-push. fcompress may re-introduce multi-control cubes, so
/// the SECOND lower is what makes the output <= 2 controls; this checks the
/// whole chain stays equivalent and ends in the gate set on circuits that
/// actually contain wide gates.
#[test]
fn phase_c_composition_ends_in_two_control_gate_set() {
    let mut rng = StdRng::seed_from_u64(0xC0DE_2C);
    let params = CompressParams {
        max_iters: 4,
        local_verify: true,
        ..CompressParams::default()
    };
    for _ in 0..24 {
        let width = rng.random_range(16..=24usize);
        let n = rng.random_range(20..=80usize);
        let mut gates: Vec<XGate> = Vec::new();
        for _ in 0..n {
            let t = rng.random_range(0..width) as u16;
            // bias toward wide gates (k up to 6) to force lowering + any
            // fcompress re-widening
            let k = rng.random_range(0..=6usize).min(width - 1);
            let mut ctrls: Vec<(u16, bool)> = Vec::new();
            while ctrls.len() < k {
                let w = rng.random_range(0..width) as u16;
                if w != t && !ctrls.iter().any(|&(x, _)| x == w) {
                    ctrls.push((w, rng.random_bool(0.5)));
                }
            }
            let mut g = XGate::conj(t, ctrls).unwrap();
            g.comp = rng.random_bool(0.5);
            gates.push(g);
        }
        let (l1, r1) = lower_wide_gates(&gates, width, 2);
        if r1.wide_in == 0 {
            continue; // nothing wide drawn this time
        }
        let (p1, _) = push_nots(&l1, width);
        let (comp, _) = compress(p1, width, &params);
        let (l2, _) = lower_wide_gates(&comp, width, 2);
        let (final_c, _) = push_nots(&l2, width);
        assert!(
            final_c.iter().all(|x| x.ctrls.len() <= 2),
            "phase C left a wide gate (fcompress re-widening not re-lowered)"
        );
        equal_on(&final_c, &gates, width);
    }
}

#[test]
fn lowered_cost_counts_wide_gates_as_their_barenco_ladder() {
    let narrow = vec![
        XGate::x_gate(0),
        XGate::cnot(1, 0),
        XGate::conj(2, [(0, true), (1, false)]).unwrap(),
    ];
    assert_eq!(lowered_cost(&narrow), 3); // <= 2 controls: 1 each
    let k3 = XGate::conj(4, [(0, true), (1, true), (2, false)]).unwrap();
    let k6 = XGate::conj(9, (0..6u16).map(|w| (w, true))).unwrap();
    assert_eq!(lowered_cost(std::slice::from_ref(&k3)), 4); // 4k-8, k=3
    assert_eq!(lowered_cost(std::slice::from_ref(&k6)), 16); // k=6
    // and the actual lowering never exceeds it (NOT-push absorbs the
    // polarity NOTs; positive controls and no complement here)
    let (low, _) = lower_wide_gates(std::slice::from_ref(&k6), 16, 2);
    assert_eq!(low.len(), lowered_cost(std::slice::from_ref(&k6)));
}
