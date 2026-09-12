use super::*;
use crate::engine::xpoly::{XPolyBudget, canonicalize_xgates_single};

#[test]
fn accounting_sums_to_the_concrete_universe() {
    // emitted-as-concrete + skipped == 8 * n * C(n-1, 3), where emitted
    // classes expand back to their concrete counts. Since `skipped`
    // already holds (concrete - emitted) per class, this reduces to
    // out.len() + skipped == universe.
    for u in 0..12usize {
        for f in 0..10usize {
            let n = u + f;
            if n < 4 {
                continue;
            }
            let touched: Vec<u16> = (0..u as u16).collect();
            let (out, skipped) = wide_gates_for_circuit_filtered(&touched, n, 0, 0);
            let universe = 8 * n * c3(n - 1);
            assert_eq!(
                out.len() + skipped,
                universe,
                "u={u} f={f}: {} emitted, {skipped} skipped",
                out.len()
            );
        }
    }
}

#[test]
fn class_counts_match_formulas_at_j0() {
    let touched: Vec<u16> = (0..8u16).collect();
    let (out, _) = wide_gates_for_circuit_filtered(&touched, 8, 0, 0);
    // f = 0: only the j=0 class exists.
    assert_eq!(out.len(), 8 * 8 * c3(7));
}

#[test]
fn band_bounds_gate_whole_classes() {
    let touched: Vec<u16> = (0..6u16).collect();
    // min_n = 8 requires >= 2 fresh wires: j=0 and j=1 classes must skip.
    let (out, _) = wide_gates_for_circuit_filtered(&touched, 12, 8, 0);
    assert!(
        out.iter().all(|g| {
            let fresh = std::iter::once(g.target)
                .chain(g.ctrls.iter().map(|&(w, _)| w))
                .filter(|w| *w >= 6)
                .count();
            fresh >= 2
        }),
        "a class below min_n leaked through"
    );
}

#[test]
fn representatives_cover_the_full_key_set() {
    // Brute force on a tiny universe: append every CONCRETE wide gate to
    // a fixed base circuit and canonicalize; the representative set must
    // reach exactly the same canonical keys. u = 3 touched wires, f = 2.
    let base = [XGate::from_g57([0, 1, 2])];
    let touched: Vec<u16> = vec![0, 1, 2];
    let n = 5usize;
    let budget = XPolyBudget::default();

    let mut brute = std::collections::HashSet::new();
    for t in 0..n as u16 {
        for b in 0..n as u16 {
            for c in 0..n as u16 {
                for d in 0..n as u16 {
                    if t == b || t == c || t == d || b == c || b == d || c == d {
                        continue;
                    }
                    if !(b < c && c < d) {
                        continue; // unordered control set
                    }
                    for pols in 0..8u8 {
                        let g = XGate::conj(
                            t,
                            [(b, pols & 1 != 0), (c, pols & 2 != 0), (d, pols & 4 != 0)],
                        )
                        .unwrap();
                        let mut circ = base.to_vec();
                        circ.push(g);
                        let canon = canonicalize_xgates_single(&circ, false, budget).unwrap();
                        brute.insert(format!("{:?}", canon.polys));
                    }
                }
            }
        }
    }

    let (reps, _) = wide_gates_for_circuit_filtered(&touched, n, 0, 0);
    let mut via_reps = std::collections::HashSet::new();
    for g in reps {
        let mut circ = base.to_vec();
        circ.push(g);
        let canon = canonicalize_xgates_single(&circ, false, budget).unwrap();
        via_reps.insert(format!("{:?}", canon.polys));
    }
    assert_eq!(via_reps, brute, "representative classes must cover exactly");
}
