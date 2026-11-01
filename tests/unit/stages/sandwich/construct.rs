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

/// A small random g57 source on wires 0..n.
fn small_source(n: usize, m: usize, seed: u64) -> CircuitSeq {
    let mut rng = StdRng::seed_from_u64(seed);
    CircuitSeq {
        gates: (0..m)
            .map(|_| {
                let a = rng.random_range(0..n);
                let x = random_wire_except(n, &[a], &mut rng);
                let y = random_wire_except(n, &[a, x], &mut rng);
                [a as u16, x as u16, y as u16]
            })
            .collect(),
    }
}

#[test]
fn group_copy_is_nonlinear_and_the_register_inverts_it() {
    // groups of 3, 4 and 5 low wires (the remainder group) on n = 5 wires
    for size in 3..=5usize {
        let n = 5;
        for seed in 0..16u64 {
            for zero_preserving in [false, true] {
                let mut rng = StdRng::seed_from_u64(0x6c00_0000 + seed * 8 + size as u64);
                let mut group: Vec<u16> = (0..n as u16).collect();
                group.shuffle(&mut rng);
                group.truncate(size);
                let (copy, recovery) = group_copy(&group, n, zero_preserving, &mut rng);
                assert_eq!(copy.len(), 2 * size, "two conjunctions per register wire");
                for g in &copy {
                    assert!((g.target as usize) >= n, "copy writes the register");
                    assert_eq!(g.ctrls.len(), 2, "two-control conjunctions, no CNOT");
                    assert!(!g.comp);
                    assert!(g.ctrls.iter().all(|&(w, _)| group.contains(&w)));
                }
                for g in &recovery {
                    assert!((g.target as usize) >= n);
                    assert!(g.ctrls.iter().all(|&(w, _)| (w as usize) >= n));
                }
                let reg_mask: u64 = group.iter().map(|&w| 1u64 << (n + w as usize)).sum();
                let low_mask: u64 = group.iter().map(|&w| 1u64 << w).sum();
                let mut images = std::collections::HashSet::new();
                for x in 0..(1u64 << n) {
                    let copied = eval_u64(&copy, x);
                    assert_eq!(
                        copied & !reg_mask,
                        x,
                        "copy touches only the group's register wires"
                    );
                    images.insert((copied & reg_mask, x & low_mask));
                    let back = eval_u64(&recovery, copied);
                    // on the zero slice the register ends holding the group's values
                    let expected: u64 = group
                        .iter()
                        .map(|&w| ((x >> w) & 1) << (n + w as usize))
                        .sum();
                    assert_eq!(back & reg_mask, expected, "size {size} seed {seed}");
                    if zero_preserving && x & low_mask == 0 {
                        assert_eq!(copied & reg_mask, 0, "zero-preserving copy");
                    }
                }
                // no register wire holds an affine function of a single low wire
                for &h in &group {
                    let r = n + h as usize;
                    for &v in &group {
                        for flip in [0u64, 1] {
                            let affine = (0..(1u64 << n))
                                .all(|x| (eval_u64(&copy, x) >> r) & 1 == ((x >> v) & 1) ^ flip);
                            assert!(!affine, "register wire {r} is a copy of wire {v}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn v_gadget_sandwich_computes_c_on_the_register_and_is_a_permutation() {
    let n = 7; // groups 3 + 4
    let mask = (1u64 << n) - 1;
    let main = small_source(n, 30, 0x6d01);
    for (slicing, layer) in [(false, 1), (false, 2), (true, 1), (true, 2)] {
        for seed in 0..4u64 {
            let mut rng = StdRng::seed_from_u64(0x6d00_0000 + seed);
            let shape = SandwichShape {
                slicing,
                layer,
                v_gates: 25,
            };
            let (a, layout) = shaped_sandwich(
                &main,
                n,
                20,
                3 * n,
                SandwichVariant::Classic,
                shape,
                &mut rng,
            );
            assert_eq!(layout.copy_groups, 2);
            assert_eq!(layout.copy_gates, 2 * n);
            assert_eq!(
                a.gates.len(),
                layout.c_gates
                    + layout.s1_gates
                    + layout.copy_gates
                    + 2 * layout.v_gates
                    + layout.recovery_gates
                    + layout.d_gates
                    + layout.s2_gates
            );
            assert!(a.gates.iter().all(|g| g.ctrls.len() <= 2));
            for x in 0..=mask {
                let expected = main.evaluate(x as usize) as u64 & mask;
                assert_eq!(
                    (eval_u64(&a.gates, x) >> n) & mask,
                    expected,
                    "slicing {slicing} layer {layer} seed {seed}: A(x,0)"
                );
            }
            let mut seen = std::collections::HashSet::new();
            for input in 0..(1u64 << (2 * n)) {
                assert!(seen.insert(eval_u64(&a.gates, input)), "A not injective");
            }
            // the reverse payload is a bijection of the low input
            let inverse: Vec<XGate> = a.gates.iter().rev().cloned().collect();
            let outs: std::collections::HashSet<u64> = (0..=mask)
                .map(|p| (eval_u64(&inverse, p) >> n) & mask)
                .collect();
            assert_eq!(
                outs.len() as u64,
                mask + 1,
                "reverse payload is a bijection"
            );
        }
    }
}

#[test]
fn sliced_v_gadget_keeps_s2_dead_under_reverse_evaluation() {
    // with slice blocks the register must stay zero through V, V* and the
    // recovery in reverse, so the reverse payload is exactly the copy's
    // bijection of D^-1(p): checking it against D^-1 and the copy alone
    let n = 6;
    let mask = (1u64 << n) - 1;
    let main = small_source(n, 24, 0x6e01);
    for seed in 0..4u64 {
        let mut rng = StdRng::seed_from_u64(0x6e00_0000 + seed);
        let d_gates = random_g57_xgates(n, 18, &mut rng);
        let shape = SandwichShape {
            slicing: true,
            layer: 1,
            v_gates: 20,
        };
        let (a, _) = v_gadget_sandwich(&main, &d_gates, n, 3 * n, shape, &mut rng);
        let inverse: Vec<XGate> = a.gates.iter().rev().cloned().collect();
        // the copy gates are the ones writing the register from the low half
        let copy: Vec<XGate> = a
            .gates
            .iter()
            .filter(|g| {
                (g.target as usize) >= n
                    && g.ctrls.iter().all(|&(w, _)| (w as usize) < n)
                    && g.ctrls.len() == 2
                    && !g.comp
            })
            .cloned()
            .collect();
        let d_inverse: Vec<XGate> = d_gates.iter().rev().cloned().collect();
        for p in 0..=mask {
            let low = eval_u64(&d_inverse, p) & mask;
            let expected = (eval_u64(&copy, low) >> n) & mask;
            assert_eq!(
                (eval_u64(&inverse, p) >> n) & mask,
                expected,
                "seed {seed}: reverse register is not copy(D^-1(p))"
            );
        }
    }
}

#[test]
fn layered_triples_are_wire_disjoint_within_a_layer() {
    let (n, layer) = (13, 4);
    let mut rng = StdRng::seed_from_u64(0x6f01);
    let triples = layered_triples(n, 103, layer, |k| rng.random_range(0..k));
    assert_eq!(triples.len(), 103);
    for chunk in triples.chunks(layer) {
        let mut wires: Vec<u16> = chunk.iter().flatten().copied().collect();
        let total = wires.len();
        wires.sort_unstable();
        wires.dedup();
        assert_eq!(wires.len(), total, "a layer's gates share no wire");
        assert!(wires.iter().all(|&w| (w as usize) < n));
    }
    for pair in triples.chunks(layer).collect::<Vec<_>>().windows(2) {
        assert!(
            !pair[1].iter().any(|g| pair[0].contains(g)),
            "no gate repeats the previous layer"
        );
    }
    // the layered C on fastrand is reproducible from the seed
    fastrand::seed(77);
    let first = layered_random_circuit(12, 40, 4);
    fastrand::seed(77);
    assert_eq!(first.gates, layered_random_circuit(12, 40, 4).gates);
}

#[test]
fn legacy_shape_reproduces_the_sliced_sandwich() {
    let n = 6;
    let main = small_source(n, 24, 0x7001);
    for variant in [SandwichVariant::Classic, SandwichVariant::Balanced] {
        for seed in 0..4u64 {
            let mut r1 = StdRng::seed_from_u64(0x7000_0000 + seed);
            let mut r2 = StdRng::seed_from_u64(0x7000_0000 + seed);
            let old = sliced_sandwich_cnot(&main, n, 20, 4 * n, variant, &mut r1);
            let (new, layout) =
                shaped_sandwich(&main, n, 20, 4 * n, variant, SandwichShape::LEGACY, &mut r2);
            assert_eq!(old.gates, new.gates);
            assert_eq!(layout.copy_positions.len(), n);
        }
    }
}

#[test]
fn unsliced_column_sandwich_is_c_n_d() {
    let n = 5;
    let mask = (1u64 << n) - 1;
    let main = small_source(n, 20, 0x7101);
    let shape = SandwichShape {
        slicing: false,
        layer: 1,
        v_gates: 0,
    };
    let mut rng = StdRng::seed_from_u64(0x7102);
    let (a, layout) = shaped_sandwich(
        &main,
        n,
        15,
        4 * n,
        SandwichVariant::Classic,
        shape,
        &mut rng,
    );
    assert_eq!(a.gates.len(), 20 + n + 15);
    assert_eq!((layout.s1_gates, layout.s2_gates), (0, 0));
    for x in 0..=mask {
        let expected = main.evaluate(x as usize) as u64 & mask;
        assert_eq!((eval_u64(&a.gates, x) >> n) & mask, expected);
    }
}

#[test]
fn dependency_depth_matches_the_interleaver() {
    use crate::stages::preprocessing::three_mask::interleave;
    for seed in 0..6u64 {
        let mut rng = StdRng::seed_from_u64(0x7201 + seed);
        let gates = sampled_xgates(9, 200, 1 + (seed as usize % 3), 0, seed % 2 == 0, &mut rng);
        let (_, _, stats) = interleave(&gates, 1, &mut rng);
        assert_eq!(dependency_depth(&gates), stats.depth, "seed {seed}");
    }
}

/// `shaped_sandwich_gate_range` brackets the shaped sandwich's gate count
/// (exactly without the V gadget), for every group remainder and both
/// copy polarities; driver preflights size nonlinear291's layout by it.
#[test]
fn shaped_sandwich_gate_range_brackets_the_construction() {
    for n in [3usize, 4, 5, 7, 8, 16] {
        let main = small_source(n, 3 * n, 0x7301 + n as u64);
        let (m_d, s) = (2 * n + 1, 3 * n);
        for (slicing, v_gates) in [(true, 0), (false, 0), (true, 11), (false, 11)] {
            let shape = SandwichShape {
                slicing,
                layer: 1,
                v_gates,
            };
            let (lo, hi) = shaped_sandwich_gate_range(n, main.gates.len(), m_d, s, shape).unwrap();
            if v_gates == 0 {
                assert_eq!(lo, hi, "exact without the V gadget");
            }
            let mut seen = std::collections::BTreeSet::new();
            for seed in 0..24u64 {
                let (a, layout) = construct_shaped_sandwich(
                    &main,
                    n,
                    m_d,
                    s,
                    SandwichVariant::Classic,
                    0x7300_0000 + seed,
                    shape,
                )
                .unwrap();
                let count = a.gates.len();
                assert!(
                    (lo..=hi).contains(&count),
                    "n {n} slicing {slicing} v {v_gates}: {count} outside {lo}..={hi}"
                );
                assert_eq!(
                    count,
                    layout.c_gates
                        + layout.s1_gates
                        + layout.copy_gates
                        + 2 * layout.v_gates
                        + layout.recovery_gates
                        + layout.d_gates
                        + layout.s2_gates
                );
                seen.insert(count);
            }
            if v_gates > 0 {
                assert!(seen.len() > 1, "the corrections are drawn (n {n})");
            }
        }
    }
    assert!(shaped_sandwich_gate_range(8, usize::MAX, 1, 0, SandwichShape::LEGACY).is_none());
}
