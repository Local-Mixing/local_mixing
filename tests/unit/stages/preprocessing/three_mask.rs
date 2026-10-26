//! The `stage2_compute_pilot` generator's test-suite (`tests/correctness.rs`,
//! `cubic.rs`, `feedback.rs`, `single_mask.rs`, `temporal_masks.rs`), ported
//! with the compute, plus the stage-2 assembly checks of the three-mask mode.

use super::*;
use crate::circuit::{XGate, eval_lanes};
use crate::stages::preprocessing::embedded_masking::{
    EmbeddedMaskingExecution, EmbeddedMaskingParams,
};
use crate::stages::preprocessing::types::{PreprocessingMode, PreprocessingParams};
use crate::stages::preprocessing::verify::verify_payload;
use crate::stages::preprocessing::{construct::preprocess_sandwich, three_mask::fire::Mask};
use crate::stages::sandwich::{
    SandwichVariant, SourceConstruction, construct_seeded_sandwich, prepare_source,
};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng, rngs::StdRng};

/// Lane `l` of wire `w` holds bit `w` of the integer `base + l`: 64
/// consecutive assignments per word, for exhaustive checks.
fn exhaustive_lanes(width: usize, base: usize) -> Vec<u64> {
    (0..width)
        .map(|w| (0..64).fold(0u64, |bits, l| bits | ((((base + l) >> w) & 1) as u64) << l))
        .collect()
}

/// Forward and reverse: the compute realizes `source` on the data roles
/// `0..np` for arbitrary band values, and the reversed stream is its inverse.
fn check_payload_both_directions(
    gates: &[XGate],
    source: &[XGate],
    np: usize,
    initial: &[u64],
    context: &str,
) {
    for reverse in [false, true] {
        let mut got = initial.to_vec();
        let mut want = initial.to_vec();
        if reverse {
            eval_lanes(gates.iter().rev(), &mut got);
            eval_lanes(source.iter().rev(), &mut want);
        } else {
            eval_lanes(gates, &mut got);
            eval_lanes(source, &mut want);
        }
        assert_eq!(got[..np], want[..np], "{context} reverse {reverse}");
        if reverse {
            eval_lanes(gates, &mut got);
        } else {
            eval_lanes(gates.iter().rev(), &mut got);
        }
        assert_eq!(got, initial, "{context} round trip, reverse {reverse}");
    }
}

#[test]
fn exhaustive_factored_fire_dirty_helpers_all_polarities() {
    for q in 1..=2 {
        let width = 5 + 6 * q;
        let h = (width - 2) as u16;
        let k = h + 1;
        let mut masks = vec![vec![]; 3];
        for (w, role) in masks.iter_mut().enumerate().take(2) {
            for i in 0..q {
                let x = (3 + 3 * q * w + 3 * i) as u16;
                role.push([x, x + 1, x + 2]);
            }
        }
        for pa in [false, true] {
            for pb in [false, true] {
                for comp in [false, true] {
                    for restore in [false, true] {
                        for factored in [false, true] {
                            let mut g = XGate::conj(2, [(0, pa), (1, pb)]).unwrap();
                            g.comp = comp;
                            let units = fire::compile(
                                &g,
                                &masks,
                                [h, k],
                                factored,
                                !restore,
                                &mut Default::default(),
                            );
                            for base in (0..1usize << width).step_by(64) {
                                let initial = exhaustive_lanes(width, base);
                                let mut got = initial.clone();
                                eval_lanes(units.iter().flatten(), &mut got);
                                let logical = |w: usize| {
                                    masks[w]
                                        .iter()
                                        .fold(initial[w], |v, &m| v ^ fire::mask_value(m, &initial))
                                };
                                let a = logical(0);
                                let b = logical(1);
                                let mut want = initial.clone();
                                want[2] ^= (if pa { a } else { !a }) & (if pb { b } else { !b });
                                if comp {
                                    want[2] = !want[2];
                                }
                                assert_eq!(got[..width - 2], want[..width - 2]);
                                if restore {
                                    assert_eq!(got, want);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn overlapping_masks_repeated_and_contradictory_literals() {
    let mut rng = StdRng::seed_from_u64(52);
    for trial in 0..200 {
        let mut masks: Vec<Vec<Mask>> = vec![vec![]; 3];
        for role in masks.iter_mut().take(2) {
            for _ in 0..2 {
                let mut wires: Vec<_> = (3..11).collect();
                wires.shuffle(&mut rng);
                role.push([wires[0], wires[1], wires[2]]);
            }
        }
        let pa = rng.random();
        let pb = rng.random();
        let comp = rng.random();
        let mut g = XGate::conj(2, [(0, pa), (1, pb)]).unwrap();
        g.comp = comp;
        let initial: Vec<u64> = (0..13).map(|_| rng.random()).collect();
        let mut got = initial.clone();
        let units = fire::compile(&g, &masks, [11, 12], true, true, &mut Default::default());
        eval_lanes(units.iter().flatten(), &mut got);
        let logical = |w: usize| {
            masks[w]
                .iter()
                .fold(initial[w], |v, &m| v ^ fire::mask_value(m, &initial))
        };
        let a = logical(0);
        let b = logical(1);
        let want = initial[2]
            ^ ((if pa { a } else { !a }) & (if pb { b } else { !b }))
            ^ (if comp { !0 } else { 0 });
        assert_eq!(got[2], want, "trial {trial}");
        for w in 0..11 {
            if w != 2 {
                assert_eq!(got[w], initial[w]);
            }
        }
    }
}

#[test]
fn compute_rotation_migration_and_arbitrary_band_both_directions() {
    for seed in 0..8 {
        let np = 6;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut source = vec![];
        for i in 0..128 {
            let t = rng.random_range(0..3); // Other roles idle: retirement must still drain them.
            let mut wires: Vec<u16> = (0..np as u16).filter(|&w| w != t).collect();
            wires.shuffle(&mut rng);
            let mut g = XGate::conj(t, wires[..i % 3].iter().map(|&w| (w, rng.random()))).unwrap();
            g.comp = rng.random();
            source.push(g);
        }
        let cfg = ThreeMaskParams {
            seed,
            keep_helper_garbage: seed % 2 == 0,
            migration_segments: if seed % 3 == 0 { 0 } else { 8 },
            ..Default::default()
        };
        let out = compute(&source, np, 64, &cfg).unwrap();
        assert!(out.stats.retired_banks > 0);
        assert_eq!(out.stats.fire.pair_guard_fallbacks, 0);
        for _ in 0..32 {
            let mut initial: Vec<u64> = (0..np + 64).map(|_| rng.random()).collect();
            for (w, lanes) in initial.iter_mut().enumerate().take(np) {
                *lanes = (0..64).fold(0, |bits, l| bits | (((l >> w) & 1) as u64) << l);
            }
            check_payload_both_directions(
                &out.gates,
                &source,
                np,
                &initial,
                &format!("seed {seed}"),
            );
        }
    }
}

#[test]
fn deterministic_and_invalid_configs() {
    let g = vec![XGate::from_g57([0, 1, 2]); 8];
    let cfg = ThreeMaskParams::default();
    assert_eq!(
        compute(&g, 3, 64, &cfg).unwrap().gates,
        compute(&g, 3, 64, &cfg).unwrap().gates
    );
    assert!(compute(&g, 3, 32, &cfg).is_err());
    assert!(compute(&g, 3, 65, &cfg).is_err());
    assert!(
        compute(
            &[XGate::conj(0, [(1, true), (2, true), (3, true)]).unwrap()],
            4,
            64,
            &cfg
        )
        .is_err()
    );
    assert!(compute(&g, 2, 64, &cfg).is_err());
}

#[test]
fn extra_band_ingress_preserves_data_for_zero_and_arbitrary_band() {
    let np = 6;
    let r = 128;
    let source: Vec<_> = (0..80)
        .map(|i| XGate::from_g57([i % 6, (i + 1) % 6, (i + 2) % 6]))
        .collect();
    let mut rng = StdRng::seed_from_u64(8016);
    for migration_segments in [0, 8] {
        let cfg = ThreeMaskParams {
            banks: 16,
            retirement_budget: 8,
            max_retiring: 0,
            local_masks: true,
            prefer_fresh: true,
            rotation_interval: 1,
            maturity_sweeps: 2,
            growth_sweeps_per_visit: 2,
            seeded_band_wires: 64,
            migration_segments,
            ..Default::default()
        };
        let out = compute(&source, np, r, &cfg).unwrap();
        assert_eq!(out.stats.band_ingress_gates, 5 * 64);
        for zeros in [false, true] {
            for _ in 0..8 {
                let mut initial: Vec<u64> = (0..np + r).map(|_| rng.random()).collect();
                if zeros {
                    initial[np + 64..].fill(0);
                }
                check_payload_both_directions(&out.gates, &source, np, &initial, "ingress");
            }
        }
    }
}

#[test]
fn compact_masks_preserve_every_logical_fire_boundary() {
    let np = 6;
    let r = 128;
    let source: Vec<_> = (0..128)
        .map(|i| XGate::from_g57([i % 6, (i + 1) % 6, (i + 2) % 6]))
        .collect();
    let cfg = ThreeMaskParams {
        banks: 16,
        retirement_budget: 8,
        max_retiring: 0,
        local_masks: true,
        prefer_fresh: true,
        rotation_interval: 1,
        maturity_sweeps: 2,
        growth_sweeps_per_visit: 2,
        seeded_band_wires: 64,
        migration_segments: 8,
        ..Default::default()
    };
    let ordinary = compute(&source, np, r, &cfg).unwrap();
    let compact = compute(
        &source,
        np,
        r,
        &ThreeMaskParams {
            compact_masks: true,
            ..cfg
        },
    )
    .unwrap();
    assert_eq!(compact.stats.mask_gates * 3, ordinary.stats.mask_gates * 2);
    assert!(compact.gates.len() < ordinary.gates.len());
    let mut rng = StdRng::seed_from_u64(20261016);
    for _ in 0..8 {
        let initial: Vec<u64> = (0..np + r).map(|_| rng.random()).collect();
        let mut a = initial.clone();
        let mut b = initial.clone();
        let (mut pa, mut pb) = (0, 0);
        for (&ca, &cb) in ordinary
            .fire_boundaries
            .iter()
            .zip(&compact.fire_boundaries)
        {
            eval_lanes(&ordinary.logical_gates[pa..ca], &mut a);
            eval_lanes(&compact.logical_gates[pb..cb], &mut b);
            assert_eq!(a, b, "entire logical state at fire boundary");
            pa = ca;
            pb = cb;
        }
        for inverse in [false, true] {
            let mut got = initial.clone();
            let mut expected = initial.clone();
            if inverse {
                eval_lanes(compact.gates.iter().rev(), &mut got);
                eval_lanes(source.iter().rev(), &mut expected);
            } else {
                eval_lanes(&compact.gates, &mut got);
                eval_lanes(&source, &mut expected);
            }
            assert_eq!(got[..np], expected[..np]);
        }
    }
}

#[test]
fn two_mask_positive_and_gate_budget() {
    let masks = vec![
        vec![[3, 4, 5], [6, 7, 8]],
        vec![[9, 10, 11], [12, 13, 14]],
        vec![],
    ];
    let g = XGate::conj(2, [(0, true), (1, true)]).unwrap();
    for (factored, keep_garbage, expected) in
        [(false, false, 137), (true, false, 89), (true, true, 73)]
    {
        let units = fire::compile(
            &g,
            &masks,
            [15, 16],
            factored,
            keep_garbage,
            &mut Default::default(),
        );
        assert_eq!(units.iter().map(Vec::len).sum::<usize>(), expected);
    }
}

#[test]
fn mask_cycle_reveals_plaintext_and_forest_rejects_it() {
    use mask_rank::{Forest, deficit};
    let a = [3, 4, 5];
    let b = [6, 7, 8];
    let c = [9, 10, 11];
    let masks = vec![vec![a, b], vec![b, c], vec![a, c]];
    assert_eq!(deficit(&masks), 1);
    let mut f = Forest::default();
    assert!(f.add(a, b));
    assert!(f.add(b, c));
    assert!(!f.add(a, c));
    let mut rng = StdRng::seed_from_u64(1901);
    for _ in 0..16 {
        let s: Vec<u64> = (0..12).map(|_| rng.random()).collect();
        let plain: u64 = rng.random();
        let encoded: Vec<_> = masks
            .iter()
            .enumerate()
            .map(|(w, ms)| {
                ms.iter().fold(if w == 2 { plain } else { 0 }, |v, &m| {
                    v ^ fire::mask_value(m, &s)
                })
            })
            .collect();
        // Two zero data roles suffice to expose the third through the cycle.
        assert_eq!(encoded[0] ^ encoded[1] ^ encoded[2], plain);
    }
}

#[test]
fn forest_allocator_rotation_and_arbitrary_band() {
    let mut rejects = 0;
    for (seed, np, r) in [(1, 6, 128), (2, 128, 256), (3, 128, 256), (4, 256, 512)] {
        let mut rng = StdRng::seed_from_u64(seed);
        let source: Vec<_> = (0..192)
            .map(|i| XGate::from_g57([i % 3, (i % 3 + 1) % 3, (i % 3 + 2) % 3]))
            .collect();
        let cfg = ThreeMaskParams {
            seed,
            banks: r / 8,
            retirement_budget: 8,
            max_retiring: 0,
            local_masks: true,
            prefer_fresh: true,
            compact_masks: true,
            forest_masks: true,
            rotation_interval: 1,
            maturity_sweeps: 2,
            growth_sweeps_per_visit: 2,
            seeded_band_wires: r / 2,
            ..Default::default()
        };
        let out = compute(&source, np, r, &cfg).unwrap();
        assert_eq!(out.stats.max_mask_rank_deficit, 0);
        assert_eq!(out.stats.fires_with_mask_rank_deficit, 0);
        rejects += out.stats.mask_rank_rejections;
        for _ in 0..8 {
            let initial: Vec<u64> = (0..np + r).map(|_| rng.random()).collect();
            check_payload_both_directions(&out.gates, &source, np, &initial, "forest");
        }
    }
    assert!(rejects > 0, "exercise the resampling branch");
}

#[test]
fn fine_banks_concurrent_retirement_and_idle_roles() {
    for seed in 0..12 {
        let np = if seed < 3 { 6 } else { 64 };
        let r = if seed % 2 == 0 { 64 } else { 128 };
        let mut rng = StdRng::seed_from_u64(200 + seed);
        let source: Vec<_> = (0..192)
            .map(|i| {
                // Most data roles are idle, so natural target replacements alone
                // cannot retire the banks.
                let t = (i % 3) as u16;
                let mut ws: Vec<_> = (0..np as u16).filter(|&w| w != t).collect();
                ws.shuffle(&mut rng);
                let mut g = XGate::conj(t, ws[..i % 3].iter().map(|&w| (w, rng.random()))).unwrap();
                g.comp = rng.random();
                g
            })
            .collect();
        let cfg = ThreeMaskParams {
            seed,
            banks: r / 8,
            retirement_budget: if seed % 2 == 0 { 4 } else { 8 },
            max_retiring: 0,
            local_masks: seed < 6,
            cross_bank_pairs: seed >= 6,
            compact_masks: seed >= 6,
            prefer_fresh: seed % 3 != 0,
            rotation_interval: 1,
            maturity_sweeps: 2,
            growth_sweeps_per_visit: 2,
            migration_segments: if seed % 3 == 0 { 0 } else { 8 },
            ..Default::default()
        };
        let out = compute(&source, np, r, &cfg).unwrap();
        assert!(out.stats.retired_banks > 6);
        assert_eq!(out.stats.fire.pair_guard_fallbacks, 0);
        assert_eq!(out.fire_boundaries.len(), source.len() + 1);
        for _ in 0..8 {
            let initial: Vec<u64> = (0..np + r).map(|_| rng.random()).collect();
            check_payload_both_directions(
                &out.gates,
                &source,
                np,
                &initial,
                &format!("fine-bank seed {seed}"),
            );
        }
    }
}

#[test]
fn rejects_invalid_bank_geometry() {
    let g = [XGate::from_g57([0, 1, 2])];
    for cfg in [
        ThreeMaskParams {
            banks: 3,
            ..Default::default()
        },
        ThreeMaskParams {
            banks: 16,
            ..Default::default()
        },
        ThreeMaskParams {
            max_retiring: 2,
            ..Default::default()
        },
        ThreeMaskParams {
            growth_sweeps_per_visit: 0,
            ..Default::default()
        },
        ThreeMaskParams {
            prefer_fresh: true,
            ..Default::default()
        },
    ] {
        assert!(compute(&g, 3, 64, &cfg).is_err());
    }
}

// cubic.rs

#[test]
fn borrowed_mcx_exhaustive_dirty_helpers() {
    for k in 0..=6 {
        for negative in [false, true] {
            let controls: Vec<_> = (0..k)
                .map(|i| (i as u16, !negative || i % 2 == 0))
                .collect();
            let out = fire::borrowed_mcx(6, &controls, &[7, 8, 9, 10]);
            let want = XGate::conj(6, controls).unwrap();
            for base in (0..1usize << 11).step_by(64) {
                let initial = exhaustive_lanes(11, base);
                let mut a = initial.clone();
                let mut b = initial;
                eval_lanes(&out, &mut a);
                eval_lanes([&want], &mut b);
                assert_eq!(a, b, "controls={k}");
            }
        }
    }
}

#[test]
fn dirty_chain_exhaustive_and_large_controls() {
    let mut rng = StdRng::seed_from_u64(20261016);
    for k in 3..=12 {
        for pattern in 0..3 {
            let controls: Vec<_> = (0..k)
                .map(|i| (i as u16, pattern == 0 || (pattern == 2 && i % 2 == 0)))
                .collect();
            let helpers: Vec<_> = (k + 1..2 * k - 1).map(|i| i as u16).collect();
            let out = fire::borrowed_mcx_chain(k as u16, &controls, &helpers);
            assert_eq!(out.len(), 4 * k - 8);
            let want = XGate::conj(k as u16, controls).unwrap();
            let width = 2 * k - 1;
            for base in (0..if k <= 8 { 1usize << width } else { 65536 }).step_by(64) {
                let initial: Vec<u64> = if k <= 8 {
                    exhaustive_lanes(width, base)
                } else {
                    (0..width).map(|_| rng.random()).collect()
                };
                let mut a = initial.clone();
                let mut b = initial;
                eval_lanes(&out, &mut a);
                eval_lanes([&want], &mut b);
                assert_eq!(a, b, "controls={k}, pattern={pattern}");
            }
        }
    }
}

#[test]
fn cubic_fire_all_polarities_and_overlapping_masks() {
    let mut rng = StdRng::seed_from_u64(0x3124);
    for trial in 0..1024 {
        let mut masks = vec![
            vec![[3, 4, 5]],
            vec![if trial % 2 == 0 { [6, 7, 8] } else { [4, 5, 8] }],
            vec![],
        ];
        if trial >= 512 {
            masks[0].push([3, 7, 8]);
            masks[1].push([3, 4, 7]);
        }
        let mut g = XGate::conj(2, [(0, trial % 2 == 0), (1, trial % 3 == 0)]).unwrap();
        g.comp = trial % 5 == 0;
        let mut s: Vec<u64> = (0..13).map(|_| rng.random()).collect();
        let initial = s.clone();
        let units = fire::compile_cubic(&g, &masks, &[9, 10, 11, 12], &mut Default::default());
        eval_lanes(units.iter().flatten(), &mut s);
        let mut a = masks[0]
            .iter()
            .fold(initial[0], |v, &m| v ^ fire::cubic_mask_value(m, &initial));
        if trial % 2 != 0 {
            a = !a;
        }
        let mut b = masks[1]
            .iter()
            .fold(initial[1], |v, &m| v ^ fire::cubic_mask_value(m, &initial));
        if trial % 3 != 0 {
            b = !b;
        }
        let mut want = initial;
        want[2] ^= (a & b) ^ if g.comp { !0 } else { 0 };
        assert_eq!(s, want);
    }
}

#[test]
fn cubic_compute_arbitrary_band_forward_inverse() {
    for seed in 0..8 {
        let mut rng = StdRng::seed_from_u64(seed);
        let input: Vec<_> = (0..96)
            .map(|i| {
                let mut g = XGate::conj(
                    (i % 3) as u16,
                    [
                        ((i % 3 + 1) as u16, rng.random()),
                        ((i % 3 + 2) as u16, rng.random()),
                    ],
                )
                .unwrap();
                g.comp = rng.random();
                g
            })
            .collect();
        let cfg = ThreeMaskParams {
            seed,
            single_mask: seed < 4,
            cubic_mask: true,
            cubic_chain: seed % 2 == 1,
            forest_masks: true,
            banks: 8,
            retirement_budget: 4,
            max_retiring: 0,
            rotation_interval: 1,
            maturity_sweeps: 2,
            local_masks: true,
            prefer_fresh: true,
            ..Default::default()
        };
        let out = compute(&input, 6, 64, &cfg).unwrap();
        assert_eq!(out.stats.max_mask_rank_deficit, 0);
        for _ in 0..16 {
            let initial: Vec<u64> = (0..70).map(|_| rng.random()).collect();
            for reverse in [false, true] {
                let mut got = initial.clone();
                let mut want = initial.clone();
                if reverse {
                    eval_lanes(out.gates.iter().rev(), &mut got);
                    eval_lanes(input.iter().rev(), &mut want);
                } else {
                    eval_lanes(&out.gates, &mut got);
                    eval_lanes(&input, &mut want);
                }
                assert_eq!(got[..6], want[..6]);
            }
        }
    }
}

// feedback.rs

#[test]
fn feedback_keeps_payload_for_every_band_input() {
    for seed in 0..4 {
        for cubic in [false, true] {
            let mut rng = StdRng::seed_from_u64(seed);
            let input: Vec<_> = (0..96)
                .map(|i| {
                    let mut g = XGate::conj(
                        (i % 6) as u16,
                        [
                            (((i + 1) % 6) as u16, rng.random()),
                            (((i + 2) % 6) as u16, rng.random()),
                        ],
                    )
                    .unwrap();
                    g.comp = rng.random();
                    g
                })
                .collect();
            let cfg = ThreeMaskParams {
                seed,
                feedback_per_fire: 16,
                midfire_feedback: seed % 2 == 1,
                single_mask: cubic,
                cubic_mask: cubic,
                forest_masks: true,
                banks: 8,
                retirement_budget: 4,
                max_retiring: 0,
                rotation_interval: 1,
                maturity_sweeps: 2,
                local_masks: true,
                prefer_fresh: true,
                ..Default::default()
            };
            let out = compute(&input, 6, 64, &cfg).unwrap();
            assert_eq!(out.stats.feedback_gates, 16 * input.len());
            for _ in 0..16 {
                let initial: Vec<u64> = (0..70).map(|_| rng.random()).collect();
                for reverse in [false, true] {
                    let mut got = initial.clone();
                    let mut want = initial.clone();
                    if reverse {
                        eval_lanes(out.gates.iter().rev(), &mut got);
                        eval_lanes(input.iter().rev(), &mut want);
                    } else {
                        eval_lanes(&out.gates, &mut got);
                        eval_lanes(&input, &mut want);
                    }
                    assert_eq!(got[..6], want[..6]);
                }
            }
        }
    }
}

// single_mask.rs

#[test]
fn single_balanced_masks_arbitrary_band_and_inverse() {
    for (np, r, banks) in [(6, 64, 8), (128, 256, 32), (256, 512, 64)] {
        for seed in 1..=3 {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut source = Vec::new();
            for i in 0..400 {
                let t = rng.random_range(0..np / 2) as u16; // Leave idle roles.
                let mut ws: Vec<_> = (0..np as u16).filter(|&w| w != t).collect();
                ws.shuffle(&mut rng);
                let mut g = XGate::conj(t, ws[..i % 3].iter().map(|&w| (w, rng.random()))).unwrap();
                g.comp = rng.random();
                source.push(g);
            }
            let cfg = ThreeMaskParams {
                seed,
                single_mask: true,
                forest_masks: true,
                compact_masks: true,
                banks,
                retirement_budget: 4,
                max_retiring: 0,
                local_masks: true,
                prefer_fresh: true,
                rotation_interval: 1,
                maturity_sweeps: 2,
                growth_sweeps_per_visit: 2,
                seeded_band_wires: r / 2,
                ..Default::default()
            };
            let output = compute(&source, np, r, &cfg).unwrap();
            assert_eq!(output.stats.max_mask_rank_deficit, 0);
            assert!(output.stats.rotations.len() > 1);
            for _ in 0..8 {
                let initial: Vec<u64> = (0..np + r).map(|_| rng.random()).collect();
                check_payload_both_directions(&output.gates, &source, np, &initial, "single");
            }
        }
    }
}

// temporal_masks.rs

#[test]
fn differential_masks_exhaustive_dirty_helper() {
    for keep in [false, true] {
        let gs = fire::emit_cubic_delta(0, [1, 2, 3], [4, 5, 6], &[7], keep);
        assert_eq!(gs.len(), if keep { 15 } else { 18 });
        for base in (0..256).step_by(64) {
            let initial = exhaustive_lanes(8, base);
            for rev in [false, true] {
                let mut got = initial.clone();
                if rev {
                    eval_lanes(gs.iter().rev(), &mut got);
                } else {
                    eval_lanes(&gs, &mut got);
                }
                assert_eq!(
                    got[0],
                    initial[0]
                        ^ (initial[1] & initial[2] & initial[3])
                        ^ (initial[4] & initial[5] & initial[6])
                );
                assert_eq!(got[1..7], initial[1..7]);
                if !keep {
                    assert_eq!(got[7], initial[7]);
                }
            }
        }
    }
}

#[test]
fn factored_cubic_fire_all_polarities() {
    let masks = vec![
        vec![[3, 4, 5], [6, 7, 8]],
        vec![[9, 10, 11], [12, 13, 14]],
        vec![],
    ];
    let mut rng = StdRng::seed_from_u64(992);
    for pa in [false, true] {
        for pb in [false, true] {
            for comp in [false, true] {
                let mut g = XGate::conj(2, [(0, pa), (1, pb)]).unwrap();
                g.comp = comp;
                let units = fire::compile_cubic_options(
                    &g,
                    &masks,
                    &[15, 16, 17, 18],
                    &mut Default::default(),
                    true,
                    true,
                );
                for _ in 0..32 {
                    let s: Vec<u64> = (0..19).map(|_| rng.random()).collect();
                    let a = masks[0]
                        .iter()
                        .fold(s[0], |a, &m| a ^ fire::cubic_mask_value(m, &s));
                    let b = masks[1]
                        .iter()
                        .fold(s[1], |a, &m| a ^ fire::cubic_mask_value(m, &s));
                    let mut want = s.clone();
                    let p = (if pa { a } else { !a }) & (if pb { b } else { !b });
                    want[2] ^= if comp { !p } else { p };
                    let mut got = s;
                    eval_lanes(units.iter().flatten(), &mut got);
                    assert_eq!(got, want);
                }
            }
        }
    }
}

#[test]
fn temporal_compute_preserves_arbitrary_band_payload_both_directions() {
    for seed in 0..16 {
        let (np, r) = (12, 128);
        let mut rng = StdRng::seed_from_u64(seed);
        let gs: Vec<_> = (0..100)
            .map(|_| {
                let t = rng.random_range(0..np) as u16;
                let a = (t + 1) % np as u16;
                let b = (t + 3) % np as u16;
                let mut g = XGate::conj(t, [(a, rng.random()), (b, rng.random())]).unwrap();
                g.comp = rng.random();
                g
            })
            .collect();
        let cfg = ThreeMaskParams {
            seed,
            banks: 8,
            retirement_budget: 4,
            rotation_interval: 1,
            maturity_sweeps: 2,
            local_masks: true,
            prefer_fresh: true,
            forest_masks: true,
            cubic_mask: true,
            cubic_chain: true,
            temporal_masks: true,
            cross_bank_growth: seed % 3 != 0,
            separate_mask_banks: seed % 3 != 1 && seed < 12,
            refresh_both_masks: (6..12).contains(&seed),
            discard_fire_helpers: seed >= 12,
            mask_count: if seed >= 12 { 3 } else { 2 },
            ..Default::default()
        };
        let p = compute(&gs, np, r, &cfg).unwrap();
        for _ in 0..16 {
            let s: Vec<u64> = (0..np + r).map(|_| rng.random()).collect();
            for rev in [false, true] {
                let (mut a, mut b) = (s.clone(), s.clone());
                if rev {
                    eval_lanes(p.gates.iter().rev(), &mut a);
                    eval_lanes(gs.iter().rev(), &mut b);
                } else {
                    eval_lanes(&p.gates, &mut a);
                    eval_lanes(&gs, &mut b);
                }
                assert_eq!(a[..np], b[..np]);
            }
        }
    }
}

#[test]
fn nonrestoring_chain_changes_only_target_and_helpers() {
    for k in 3..=6 {
        for signs in 0..(1usize << k) {
            let controls: Vec<_> = (1..=k)
                .map(|w| (w as u16, signs >> (w - 1) & 1 != 0))
                .collect();
            let helpers: Vec<_> = ((k + 1)..(2 * k - 1)).map(|w| w as u16).collect();
            let gates = fire::borrowed_mcx_chain_dirty(0, &controls, &helpers);
            assert_eq!(gates.len(), 2 * k - 3);
            for start in (0..1usize << (2 * k - 1)).step_by(64) {
                let initial = exhaustive_lanes(2 * k - 1, start);
                let p = controls.iter().fold(!0u64, |a, &(w, pos)| {
                    a & if pos {
                        initial[w as usize]
                    } else {
                        !initial[w as usize]
                    }
                });
                for rev in [false, true] {
                    let mut got = initial.clone();
                    if rev {
                        eval_lanes(gates.iter().rev(), &mut got);
                    } else {
                        eval_lanes(&gates, &mut got);
                    }
                    assert_eq!(got[0], initial[0] ^ p);
                    assert_eq!(got[1..=k], initial[1..=k]);
                }
            }
        }
    }
}

// The stage-2 assembly.

#[test]
fn v2_is_the_campaign_configuration_and_auto_band_rounds_up() {
    let p = ThreeMaskParams::v2(7, 128);
    assert_eq!((p.seed, p.band_wires), (7, 128));
    assert_eq!(
        (
            p.banks,
            p.retirement_budget,
            p.max_retiring,
            p.rotation_interval,
            p.maturity_sweeps,
            p.growth_sweeps_per_visit,
            p.growth_per_fire,
            p.warmup_sweeps,
            p.migration_segments,
            p.mask_count
        ),
        (0, 2, 0, 1, 2, 2, 8, 4, 6, 3)
    );
    assert_eq!(resolve_banks(128, 0), 8);
    assert_eq!(resolve_banks(256, 0), 16);
    assert_eq!(resolve_banks(256, 8), 8);
    assert!(
        p.local_masks
            && p.prefer_fresh
            && p.compact_masks
            && p.forest_masks
            && p.cubic_mask
            && p.cubic_chain
            && p.temporal_masks
            && p.cross_bank_growth
            && p.discard_fire_helpers
            && p.random_helpers
            && p.unique_cover
    );
    assert!(
        p.keep_helper_garbage
            && p.factored
            && p.schedule
            && p.feedback_per_fire == 0
            && !p.midfire_feedback
            && !p.fold_cubic_polarity
            && !p.separate_mask_banks
            && !p.refresh_both_masks
            && !p.single_mask
            && !p.cross_bank_pairs
            && !p.bank_disjoint_masks
            && p.seeded_band_wires == 0
    );
    // The larger of 2n and 128, rounded up to a multiple of 16.
    assert_eq!(auto_band_wires(16), 128);
    assert_eq!(auto_band_wires(64), 128);
    assert_eq!(auto_band_wires(128), 128);
    assert_eq!(auto_band_wires(200), 208);
    assert_eq!(auto_band_wires(256), 256);
    assert_eq!(resolve_band_wires(64, 0), 128);
    assert_eq!(resolve_band_wires(64, 192), 192);
    assert_eq!(
        PreprocessingMode::parse("three-mask"),
        Some(PreprocessingMode::ThreeMask)
    );
    assert_eq!(PreprocessingMode::ThreeMask.canonical_name(), "three-mask");
}

/// n = 8: the three-mask stage verifies in both directions, its outer blocks
/// (opening guard, band seed, band reseed, closing guard) are exactly the
/// embedded-masking L8 outer blocks for the same gadget seed and guard RNG,
/// and the compute between them is the generator's compute of the sandwich.
#[test]
fn three_mask_mode_verifies_and_shares_the_l8_outer_blocks() {
    let n = 8;
    let (source_seed, gadget_seed, sandwich_seed) = (41, 43, 47);
    let source = prepare_source(n, 24, source_seed, None, SourceConstruction::Random).unwrap();
    let sandwich =
        construct_seeded_sandwich(&source, n, 24, 24, SandwichVariant::Classic, sandwich_seed);
    let outer = EmbeddedMaskingParams::preset_l8(gadget_seed, n);
    let three = {
        let mut rng = StdRng::seed_from_u64(gadget_seed ^ 0x6AD6_E75E);
        preprocess_sandwich(
            &sandwich,
            n,
            160,
            SandwichVariant::Classic,
            &PreprocessingParams::ThreeMask {
                params: ThreeMaskParams::v2(gadget_seed, 0),
                outer,
                balanced_seed: true,
                seed_fill_nl: 4,
            },
            &mut rng,
        )
        .unwrap()
    };
    let l8 = {
        let mut rng = StdRng::seed_from_u64(gadget_seed ^ 0x6AD6_E75E);
        preprocess_sandwich(
            &sandwich,
            n,
            160,
            SandwichVariant::Classic,
            &PreprocessingParams::EmbeddedMasking {
                params: outer,
                execution: EmbeddedMaskingExecution::default(),
                balanced_seed: true,
                seed_fill_nl: 4,
            },
            &mut rng,
        )
        .unwrap()
    };
    assert!(three.guarded);
    let report = three.three_mask_report.as_ref().unwrap();
    let baseline = l8.embedded_masking_report.as_ref().unwrap();
    // Auto band: max(2n, 128) = 128 > the 2n = 16 wires the outer blocks seed.
    assert_eq!(report.band_wires, 128);
    assert_eq!(report.seeded_band_wires, 2 * n);
    assert_eq!(three.circuit.num_wires, 2 * n + 128);
    assert_eq!(report.band_ingress_gates, 5 * (128 - 2 * n));
    assert_eq!(
        (
            report.guard_gates,
            report.band_seed_gates,
            report.band_reseed_gates
        ),
        (
            baseline.guard_gates,
            baseline.band_seed_gates,
            baseline.band_reseed_gates
        )
    );
    let start = report.guard_gates + report.band_seed_gates;
    let end = start + report.compute_gates;
    let baseline_end = start + baseline.compute_gates;
    assert_eq!(
        three.circuit.gates.len(),
        end + report.band_reseed_gates + report.guard_gates
    );
    assert_eq!(three.circuit.gates[..start], l8.circuit.gates[..start]);
    assert_eq!(three.circuit.gates[end..], l8.circuit.gates[baseline_end..]);
    let direct = compute(
        &sandwich.gates,
        sandwich.num_wires,
        128,
        &ThreeMaskParams {
            seeded_band_wires: 2 * n,
            banks: resolve_banks(128, 0),
            ..ThreeMaskParams::v2(gadget_seed, 0)
        },
    )
    .unwrap();
    assert_eq!(three.circuit.gates[start..end], direct.gates[..]);
    assert_eq!(
        verify_payload(
            &sandwich,
            &three.circuit,
            n,
            three.guarded,
            SandwichVariant::Classic,
            source_seed
        ),
        (n, 2 * n, true)
    );
    // The band the outer blocks seed must fit inside the compute band.
    let mut rng = StdRng::seed_from_u64(gadget_seed ^ 0x6AD6_E75E);
    assert!(
        preprocess_sandwich(
            &sandwich,
            n,
            160,
            SandwichVariant::Classic,
            &PreprocessingParams::ThreeMask {
                params: ThreeMaskParams::v2(gadget_seed, 64),
                outer: EmbeddedMaskingParams { r: 96, ..outer },
                balanced_seed: true,
                seed_fill_nl: 4,
            },
            &mut rng,
        )
        .is_err()
    );
}
