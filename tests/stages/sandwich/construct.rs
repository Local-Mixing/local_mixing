use super::*;
use crate::circuit::xgate::eval_u64;
use rand::{SeedableRng, rngs::StdRng};

#[test]
fn sandwich_slice_gates_are_dead_on_the_zero_slice() {
    let n = 4;
    let mask = (1u64 << n) - 1;
    for seed in 0..8u64 {
        let mut rng = StdRng::seed_from_u64(0x5a2d_0000 + seed);
        let block = sandwich_slice_gates(n, 5 * n, false, &mut rng);
        for g in &block {
            assert!(!g.comp);
            assert!((g.target as usize) < n, "targets in the first half");
            assert!(g.ctrls.iter().all(|&(_, p)| p), "positive controls");
            assert!(
                g.ctrls.iter().any(|&(w, _)| (w as usize) >= n),
                "every gate reads a second-half wire"
            );
        }
        // Second half zero => identity on any first-half value.
        for x in 0..=mask {
            assert_eq!(eval_u64(&block, x), x, "dead on the zero slice");
        }
    }
}

#[test]
fn mirrored_sandwich_slice_gates_are_dead_on_the_zero_first_half() {
    // The balanced variant's S2: the same block reflected through the
    // halves — targets high, reads low, dead when the FIRST half is zero.
    let n = 4;
    let mask = (1u64 << n) - 1;
    for seed in 0..8u64 {
        let mut rng = StdRng::seed_from_u64(0x5a3d_0000 + seed);
        let block = sandwich_slice_gates(n, 5 * n, true, &mut rng);
        for g in &block {
            assert!(!g.comp);
            assert!((g.target as usize) >= n, "targets in the second half");
            assert!((g.target as usize) < 2 * n, "targets stay on 2n wires");
            assert!(g.ctrls.iter().all(|&(_, p)| p), "positive controls");
            assert!(
                g.ctrls.iter().any(|&(w, _)| (w as usize) < n),
                "every gate reads a first-half wire"
            );
        }
        // First half zero => identity on any second-half value.
        for y in 0..=mask {
            let state = y << n;
            assert_eq!(eval_u64(&block, state), state, "dead on the zero slice");
        }
    }
}

#[test]
fn sliced_sandwich_computes_c_on_the_second_half_on_the_zero_slice() {
    let n = 3;
    let mask = (1u64 << n) - 1;
    let full = (1u64 << (2 * n)) - 1;
    let main = CircuitSeq {
        gates: vec![[0, 1, 2], [2, 0, 1], [1, 2, 0], [0, 2, 1]],
    };
    for seed in 0..8u64 {
        let mut rng = StdRng::seed_from_u64(0x5a4d_0000 + seed);
        let a = sliced_sandwich_cnot(&main, n, 12, 4 * n, SandwichVariant::Classic, &mut rng);
        assert_eq!(a.num_wires, 2 * n);

        // Zero slice: the second half carries C(x).
        for x in 0..=mask {
            let expected = main.evaluate(x as usize) as u64 & mask;
            assert_eq!((eval_u64(&a.gates, x) >> n) & mask, expected, "A(x,0)");
        }
        // A is a permutation of the whole 2n-bit space.
        let mut seen = std::collections::HashSet::new();
        for input in 0..=full {
            assert!(seen.insert(eval_u64(&a.gates, input)), "A not injective");
        }
        // Off-slice, the second-half output is y-masked and differs from
        // C(x) for at least some inputs on some nonzero slice.
        let differs = (1..=mask).any(|y| {
            (0..=mask).any(|x| {
                let input = x | (y << n);
                let expected = main.evaluate(x as usize) as u64 & mask;
                (eval_u64(&a.gates, input) >> n) & mask != expected
            })
        });
        assert!(differs, "seed={seed:#x}: no off-slice disturbance");
    }
}

#[test]
fn sliced_sandwich_floats_the_middle_column_into_a_band() {
    // After the float stage the N CNOTs must no longer sit as one
    // contiguous column: their positions should straddle other material
    // on both sides for at least some gates, under every seed and in
    // either variant.
    let n = 6;
    let main = CircuitSeq {
        gates: (0..24)
            .map(|k| [(k % n) as u16, ((k + 1) % n) as u16, ((k + 2) % n) as u16])
            .collect(),
    };
    for variant in [SandwichVariant::Classic, SandwichVariant::Balanced] {
        for seed in 0..8u64 {
            let mut rng = StdRng::seed_from_u64(0x5a6d_0000 + seed);
            let d_gates = random_g57_xgates(n, 20, &mut rng);
            let (a, positions) =
                sliced_sandwich_build(&main, &d_gates, n, 4 * n, variant, &mut rng);
            assert_eq!(positions.len(), n, "exactly the n column CNOTs");
            if variant == SandwichVariant::Classic {
                // In the classic layout the column is exactly the set of
                // gates targeting the second half — the identification the
                // float stage used before the variants existed.
                let by_target: Vec<usize> = (0..a.gates.len())
                    .filter(|&i| (a.gates[i].target as usize) >= n)
                    .collect();
                assert_eq!(positions, by_target, "tracked column == high-target gates");
            }
            for &p in &positions {
                let g = &a.gates[p];
                assert_eq!(g.ctrls.len(), 1, "the column is CNOTs");
                let (control, polarity) = g.ctrls[0];
                assert!(polarity);
                let (target, control) = (g.target as usize, control as usize);
                if variant.is_balanced() {
                    assert_eq!(control, target + n, "balanced N is x_i ^= y_i");
                } else {
                    assert_eq!(target, control + n, "classic N is y_i ^= x_i");
                }
            }
            let span = positions.last().unwrap() - positions.first().unwrap();
            assert!(
                span > n,
                "variant={:?} seed={seed:#x}: column still contiguous (span {span})",
                variant
            );
        }
    }
}

#[test]
fn sliced_sandwich_inverse_is_dead_slice_and_reveals_d_inverse() {
    let n = 3;
    let mask = (1u64 << n) - 1;
    let main = CircuitSeq {
        gates: vec![[0, 1, 2], [2, 0, 1]],
    };
    for seed in 0..8u64 {
        let mut rng = StdRng::seed_from_u64(0x5a5d_0000 + seed);
        let a = sliced_sandwich_cnot(&main, n, 10, 4 * n, SandwichVariant::Classic, &mut rng);
        let inverse: Vec<XGate> = a.gates.iter().rev().cloned().collect();
        // The inverse computes some permutation on the second half on the
        // zero slice (D^-1 up to the dead S2); check it is a bijection x
        // -> second-half output, i.e. the slice really carries a function.
        let mut outs = std::collections::HashSet::new();
        for p in 0..=mask {
            outs.insert((eval_u64(&inverse, p) >> n) & mask);
        }
        assert_eq!(
            outs.len() as u64,
            mask + 1,
            "inverse second-half map is a bijection on the zero slice"
        );
    }
}

#[test]
fn balanced_sliced_sandwich_computes_c_on_the_first_half_on_the_zero_slice() {
    let n = 3;
    let mask = (1u64 << n) - 1;
    let full = (1u64 << (2 * n)) - 1;
    let main = CircuitSeq {
        gates: vec![[0, 1, 2], [2, 0, 1], [1, 2, 0], [0, 2, 1]],
    };
    for seed in 0..8u64 {
        let mut rng = StdRng::seed_from_u64(0x5b4d_0000 + seed);
        let a = sliced_sandwich_cnot(&main, n, 12, 4 * n, SandwichVariant::Balanced, &mut rng);
        assert_eq!(a.num_wires, 2 * n);

        // Zero slice: the FIRST half carries C(x); D and S2 write only
        // the second half, so nothing after block 1 can touch it.
        for x in 0..=mask {
            let expected = main.evaluate(x as usize) as u64 & mask;
            assert_eq!(eval_u64(&a.gates, x) & mask, expected, "A(x,0)");
        }
        // A is a permutation of the whole 2n-bit space.
        let mut seen = std::collections::HashSet::new();
        for input in 0..=full {
            assert!(seen.insert(eval_u64(&a.gates, input)), "A not injective");
        }
        // Off-slice, the first-half output is y-masked by the flipped N
        // column and differs from C(x) on some nonzero slice.
        let differs = (1..=mask).any(|y| {
            (0..=mask).any(|x| {
                let input = x | (y << n);
                let expected = main.evaluate(x as usize) as u64 & mask;
                eval_u64(&a.gates, input) & mask != expected
            })
        });
        assert!(differs, "seed={seed:#x}: no off-slice disturbance");
    }
}

#[test]
fn balanced_sliced_sandwich_inverse_reveals_d_inverse_on_the_mirrored_slice() {
    // The balanced inverse is sliced at x = 0 (not y = 0), and there the
    // second half carries D^-1 exactly: S2 is dead with the first half
    // zero, so D^-1 runs clean, and the reversed block 1 junks only the
    // first half.
    let n = 3;
    let mask = (1u64 << n) - 1;
    let main = CircuitSeq {
        gates: vec![[0, 1, 2], [2, 0, 1]],
    };
    for seed in 0..8u64 {
        let mut rng = StdRng::seed_from_u64(0x5b5d_0000 + seed);
        let d_gates = random_g57_xgates(n, 10, &mut rng);
        let a = sliced_sandwich_with_d(
            &main,
            &d_gates,
            n,
            4 * n,
            SandwichVariant::Balanced,
            &mut rng,
        );
        let inverse: Vec<XGate> = a.gates.iter().rev().cloned().collect();
        for q in 0..=mask {
            let out = eval_u64(&inverse, q << n);
            let recovered = (out >> n) & mask;
            // D(D^-1(q)) == q, with D read on its own low-half copy.
            assert_eq!(
                eval_u64(&d_gates, recovered) & mask,
                q,
                "seed={seed:#x}: A^-1(0,q) second half is not D^-1(q)"
            );
        }
    }
}
