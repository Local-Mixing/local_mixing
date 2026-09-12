use super::*;
use crate::circuit::CircuitSeq;
use crate::circuit::xgate::{Lits, eval_lanes};
use smallvec::SmallVec;

fn eval_polys(polys: &[Polynomial], input: u64) -> u64 {
    let mut out = 0u64;
    for (wire, poly) in polys.iter().enumerate() {
        let bit = poly.iter().fold(false, |acc, &m| acc ^ ((input & m) == m));
        if bit {
            out |= 1u64 << wire;
        }
    }
    out
}

fn apply_scalar(gates: &[XGate], input: u64, n: usize) -> u64 {
    let mut lanes = (0..n)
        .map(|wire| if input >> wire & 1 == 1 { 1 } else { 0 })
        .collect::<Vec<u64>>();
    eval_lanes(gates.iter(), &mut lanes);
    lanes
        .iter()
        .enumerate()
        .fold(0, |acc, (wire, &v)| acc | ((v & 1) << wire))
}

// xgate_used_wires_len replaces `xgate_used_wires(..).len()` on the DB span
// guards, where a disagreement would silently move db_span_skips and change
// which windows reach the store. Pin the two against each other, including
// the >=1024 fallback and the empty-window case.
#[test]
fn opt_equiv_used_wires_len_matches_the_materialized_list() {
    let mut state = 0x243f_6a88_85a3_08d3u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    assert_eq!(xgate_used_wires_len(&[]), xgate_used_wires(&[]).len());
    for case in 0..600 {
        // Most cases stay under 1024 wires (the bitset path); every 7th
        // case reaches past it to exercise the sorted-list fallback.
        let hi = if case % 7 == 0 { 4096 } else { 1024 };
        let gates: Vec<XGate> = (0..(next() % 12) + 1)
            .map(|_| {
                let target = (next() % hi) as u16;
                let mut ctrls: Lits = SmallVec::new();
                for _ in 0..(next() % 5) {
                    let wire = (next() % hi) as u16;
                    if wire != target && !ctrls.iter().any(|&(w, _)| w == wire) {
                        ctrls.push((wire, next() % 2 == 0));
                    }
                }
                ctrls.sort_unstable();
                XGate {
                    target,
                    comp: next() % 2 == 0,
                    ctrls,
                }
            })
            .collect();
        assert_eq!(
            xgate_used_wires_len(&gates),
            xgate_used_wires(&gates).len(),
            "case {case}: {gates:?}"
        );
    }
}

#[test]
fn arbitrary_xgate_polynomials_match_evaluation() {
    let gates = vec![
        XGate::x_gate(0),
        XGate::conj(1, [(0, true), (2, false)]).unwrap(),
        XGate {
            target: 3,
            comp: true,
            ctrls: [(0, false), (1, true), (2, false)].into_iter().collect(),
        },
        // comp XOR empty product = 1 XOR 1 = 0: an exact no-op.
        XGate {
            target: 2,
            comp: true,
            ctrls: Default::default(),
        },
    ];
    let polys = xgates_to_polynomial(&gates, 4, XPolyBudget::default()).unwrap();
    for input in 0..16u64 {
        assert_eq!(eval_polys(&polys, input), apply_scalar(&gates, input, 4));
    }
}

#[test]
fn g57_canonical_keys_match_legacy_in_both_directions() {
    let legacy = CircuitSeq {
        gates: vec![[7, 2, 11], [2, 7, 5], [11, 5, 2], [5, 11, 7]],
    };
    let gates: Vec<XGate> = legacy.gates.iter().copied().map(XGate::from_g57).collect();
    for reversed in [false, true] {
        let old = legacy.canonicalize_polys_single(reversed);
        let new = canonicalize_xgates_single(&gates, reversed, XPolyBudget::default()).unwrap();
        assert_eq!(new.polys, old.0);
        assert_eq!(new.order, old.1);
        assert_eq!(new.used_wires, old.2);
    }
}

#[test]
fn g57_canonical_keys_match_legacy_randomized() {
    // Randomized extension of the hand-written case above: the XGate
    // canonicalization path must agree with the legacy g57 path on polys,
    // order and used_wires for arbitrary g57 circuits, both directions —
    // including the x==y degenerate lift (from_g57 -> X gate), which the
    // hand-written case never exercises. Deterministic LCG: reproducible.
    let mut state = 0x9e3779b97f4a7c15u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state >> 33
    };
    const WIRES: u64 = 12;
    const CASES: usize = 50_000;
    for case in 0..CASES {
        let len = 3 + (next() % 6) as usize;
        let mut gates_g57 = Vec::with_capacity(len);
        for gi in 0..len {
            let a = (next() % WIRES) as u16;
            let mut x = (next() % WIRES) as u16;
            let mut y = (next() % WIRES) as u16;
            while x == a {
                x = (next() % WIRES) as u16;
            }
            // Storable alphabet only (decode_rocks_entry rejects x==y):
            // the degenerate X-gate lift legitimately DIVERGES on
            // used_wires (the phantom wire disappears) — pinned by
            // `degenerate_x_lift_drops_the_phantom_wire` below.
            let _ = gi;
            while y == a || y == x {
                y = (next() % WIRES) as u16;
            }
            gates_g57.push([a, x, y]);
        }
        let legacy = CircuitSeq {
            gates: gates_g57.clone(),
        };
        let lifted: Vec<XGate> = gates_g57.iter().copied().map(XGate::from_g57).collect();
        for reversed in [false, true] {
            let old = legacy.canonicalize_polys_single(reversed);
            let new = canonicalize_xgates_single(&lifted, reversed, XPolyBudget::default())
                .unwrap_or_else(|e| {
                    panic!("case {case} rev={reversed}: xgate canon failed: {e:?}")
                });
            assert_eq!(new.polys, old.0, "case {case} rev={reversed}: polys");
            assert_eq!(new.order, old.1, "case {case} rev={reversed}: order");
            assert_eq!(
                new.used_wires, old.2,
                "case {case} rev={reversed}: used_wires"
            );
        }
    }
}

#[test]
fn degenerate_x_lift_drops_the_phantom_wire() {
    // [3,4,4] fires always: from_g57 lifts it to an X gate with EMPTY
    // controls, so wire 4 vanishes from the xgate path's used_wires while
    // the legacy path still counts it. This divergence is confined to
    // degenerate triples, which decode_rocks_entry rejects (ctrl_a ==
    // ctrl_b), so no stored circuit can ever hit it — but it means
    // from_g57 lifting is key-compatible ONLY for storable circuits.
    let legacy = CircuitSeq {
        gates: vec![[0, 1, 2], [3, 4, 4]],
    };
    let lifted: Vec<XGate> = legacy.gates.iter().copied().map(XGate::from_g57).collect();
    let old = legacy.canonicalize_polys_single(false);
    let new = canonicalize_xgates_single(&lifted, false, XPolyBudget::default()).unwrap();
    assert!(old.2.contains(&4), "legacy counts the phantom wire");
    assert!(
        !new.used_wires.contains(&4),
        "lift drops the phantom wire (X gate has no controls)"
    );
}

#[test]
fn budget_exhaustion_is_a_clean_error() {
    let gates = vec![XGate::from_g57([0, 1, 2])];
    let budget = XPolyBudget {
        max_mul_terms: 1,
        max_poly_terms: 32,
        max_total_terms: 64,
    };
    assert!(matches!(
        xgates_to_polynomial(&gates, 3, budget),
        Err(XPolyError::BudgetExceeded {
            stage: "polynomial multiply",
            ..
        })
    ));
}

#[test]
fn total_term_budget_keeps_the_exact_error_contract() {
    let gates = vec![XGate::from_g57([0, 1, 2])];
    let budget = XPolyBudget {
        max_mul_terms: 32,
        max_poly_terms: 32,
        max_total_terms: 3,
    };
    assert!(matches!(
        xgates_to_polynomial(&gates, 3, budget),
        Err(XPolyError::BudgetExceeded {
            stage: "total terms",
            attempted: 4,
            limit: 3,
        })
    ));
}

#[test]
fn exact_degree_cap_rejects_high_degree_and_keeps_low_degree() {
    let high = vec![XGate::conj(0, [(1, true), (2, true), (3, true), (4, true)]).unwrap()];
    assert!(matches!(
        canonicalize_xgates_single_capped(&high, false, XPolyBudget::default(), 3),
        Err(XPolyError::DegreeExceeded {
            degree: 4,
            limit: 3
        })
    ));

    let low = vec![XGate::from_g57([0, 1, 2])];
    assert!(canonicalize_xgates_single_capped(&low, false, XPolyBudget::default(), 2).is_ok());
}

#[test]
fn target_in_controls_is_rejected_before_reverse_lookup() {
    let gates = vec![XGate {
        target: 0,
        comp: false,
        ctrls: [(0, true)].into_iter().collect(),
    }];
    assert_eq!(
        canonicalize_xgates_single(&gates, false, XPolyBudget::default()).unwrap_err(),
        XPolyError::TargetInControls { target: 0 }
    );
}
