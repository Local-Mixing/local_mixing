use super::*;
use crate::stages::preprocessing::embedded_masking::{
    EmbeddedMaskingExecution, EmbeddedMaskingParams,
};
use crate::stages::preprocessing::verify::verify_payload;
use crate::stages::sandwich::{construct_seeded_sandwich, prepare_source};
use rand::{SeedableRng, rngs::StdRng};

// Stable structural digest of seeded CLI mpmct1 artifacts. Every
// width, gate and literal participates, so altered RNG consumption changes it.
fn structural_digest(circuit: &Circuit) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    let mut push = |word: u64| {
        for byte in word.to_le_bytes() {
            hash = (hash ^ byte as u64).wrapping_mul(0x100000001b3);
        }
    };
    push(circuit.num_wires as u64);
    push(circuit.gates.len() as u64);
    for gate in &circuit.gates {
        push(gate.target as u64);
        push(gate.comp as u64);
        push(gate.ctrls.len() as u64);
        for &(wire, polarity) in &gate.ctrls {
            push(wire as u64);
            push(polarity as u64);
        }
    }
    hash
}

#[test]
fn managed_and_raw_presets_preserve_seeded_generator_artifacts() {
    // Pinned fixture: n=6, |C|=|D|=3, s=6,
    // slice_gates=120, source_seed=41, preprocessing_seed=43, sandwich_seed=47.
    for (variant, band_only, expected) in [
        (SandwichVariant::Classic, false, 0x1492faa91c3ccdf6),
        (SandwichVariant::Classic, true, 0xfa07015544e3de02),
        (SandwichVariant::Balanced, false, 0xcc27675ed50a51d6),
    ] {
        let source = prepare_source(6, 3, 41, None).unwrap();
        let sandwich = construct_seeded_sandwich(&source, 6, 3, 6, variant, 47);
        let options = PreprocessingParams::EmbeddedMasking {
            params: if band_only {
                EmbeddedMaskingParams {
                    burst_band_only: true,
                    ..EmbeddedMaskingParams::managed_tdp(43, 6)
                }
            } else if variant == SandwichVariant::Balanced {
                EmbeddedMaskingParams {
                    active_wires: 6,
                    ..EmbeddedMaskingParams::production(43)
                }
            } else {
                EmbeddedMaskingParams::managed_tdp(43, 6)
            },
            execution: EmbeddedMaskingExecution::default(),
            balanced_seed: true,
            seed_fill_nl: 4,
        };
        let mut rng = StdRng::seed_from_u64(43 ^ 0x6AD6_E75E);
        let output = preprocess_sandwich(&sandwich, 6, 120, variant, &options, &mut rng).unwrap();
        // Hashes re-pinned 2026-10-05 (rich band seed, punchlist #5),
        // 2026-10-06 (feistel slice guard default, punchlist #7), and
        // 2026-10-06b (8-bracket ancilla-order randomisation draws).
        // Hashes re-pinned through the 2026-10-05/07 preprocessing work;
        // latest: burst hot-targeting (punchlist #8, hot leg).
        assert_eq!(structural_digest(&output.circuit), expected, "{variant:?}");
        verify_payload(&sandwich, &output.circuit, 6, output.guarded, variant, 41);
    }
}

#[test]
fn nonlinear_sandwich_preserves_all_logical_outputs_without_a_reverse_port_contract() {
    let source = prepare_source(6, 3, 41, None).unwrap();
    let sandwich = construct_seeded_sandwich(&source, 6, 3, 6, SandwichVariant::Classic, 47);
    let mut rng = StdRng::seed_from_u64(43 ^ 0x6AD6_E75E);
    let output = preprocess_sandwich(
        &sandwich,
        6,
        120,
        SandwichVariant::Classic,
        &PreprocessingParams::Nonlinear291,
        &mut rng,
    )
    .unwrap();
    // The native encoding has an ingress guard; it never installs the
    // product/embedded masking output junk guard that promises reverse ports.
    assert!(!output.guarded);
    assert_eq!(
        verify_payload(
            &sandwich,
            &output.circuit,
            6,
            output.guarded,
            SandwichVariant::Classic,
            41
        ),
        (0, 12, false)
    );
    assert!(
        output
            .circuit
            .gates
            .iter()
            .all(|gate| gate.ctrls.len() <= 2)
    );
}

#[test]
fn managed_preprocessing_shuffling_preserves_classic_and_balanced_ports() {
    use crate::circuit::eval_lanes4;

    let n = 6;
    for seed in [43, 53] {
        for variant in [SandwichVariant::Classic, SandwichVariant::Balanced] {
            let source = prepare_source(n, 3, 41, None).unwrap();
            let sandwich = construct_seeded_sandwich(&source, n, 3, 6, variant, 47);
            let make_output = |segments, return_home| {
                let options = PreprocessingParams::EmbeddedMasking {
                    params: EmbeddedMaskingParams {
                        // Enough band roles to exercise actual transfers even
                        // for this small exhaustive fixture. Band input is zero
                        // and the stage itself derives its values from data.
                        r: 32,
                        shuffling_segments: segments,
                        shuffling_return_home: return_home,
                        ..EmbeddedMaskingParams::managed_tdp(seed, n)
                    },
                    execution: EmbeddedMaskingExecution {
                        record_hot_intervals: true,
                        ..Default::default()
                    },
                    balanced_seed: true,
                    seed_fill_nl: 4,
                };
                let mut rng = StdRng::seed_from_u64(seed ^ 0x6AD6_E75E);
                preprocess_sandwich(&sandwich, n, 120, variant, &options, &mut rng).unwrap()
            };
            let baseline = make_output(0, true);
            // Per-box return home and persistent migration (tail-restored
            // ports) must both preserve every port contract below.
            for return_home in [true, false] {
                let shuffled = make_output(8, return_home);
                let baseline_report = baseline.embedded_masking_report.as_ref().unwrap();
                let shuffled_report = shuffled.embedded_masking_report.as_ref().unwrap();
                assert!(!baseline_report.hot_intervals.is_empty());
                assert_eq!(shuffled_report.hot_intervals, baseline_report.hot_intervals);
                assert_eq!(
                    shuffled_report.original_compute_gates,
                    baseline_report.compute_gates
                );
                assert!(shuffled_report.compute_gates > shuffled_report.original_compute_gates);
                assert!(
                    shuffled_report
                        .hot_intervals
                        .iter()
                        .any(|&(_, _, end)| end == shuffled_report.original_compute_gates)
                );
                assert!(shuffled.guarded);
                assert!(shuffled.circuit.gates.len() > baseline.circuit.gates.len());
                let (from, to, reverse_checked) = verify_payload(
                    &sandwich,
                    &shuffled.circuit,
                    n,
                    shuffled.guarded,
                    variant,
                    seed,
                );
                assert_eq!(reverse_checked, variant == SandwichVariant::Classic);

                // Exhaust all 2^(2n) sandwich inputs, including the nonzero upper
                // data half, with the auxiliary band on its required zero slice.
                for first_input in (0..1usize << sandwich.num_wires).step_by(256) {
                    let initial: Vec<[u64; 4]> = (0..shuffled.circuit.num_wires)
                        .map(|wire| {
                            std::array::from_fn(|batch| {
                                (0..64).fold(0, |lanes, lane| {
                                    let input = first_input + batch * 64 + lane;
                                    lanes | (((input >> wire) & 1) as u64) << lane
                                })
                            })
                        })
                        .collect();
                    let mut expected = initial.clone();
                    let mut before = initial.clone();
                    let mut actual = initial.clone();
                    eval_lanes4(&sandwich.gates, &mut expected);
                    eval_lanes4(&baseline.circuit.gates, &mut before);
                    eval_lanes4(&shuffled.circuit.gates, &mut actual);
                    assert_eq!(
                        actual[from..to],
                        expected[from..to],
                        "source payload seed={seed} {variant:?}"
                    );
                    assert_eq!(
                        actual[from..to],
                        before[from..to],
                        "baseline payload seed={seed} {variant:?}"
                    );
                    eval_lanes4(shuffled.circuit.gates.iter().rev(), &mut actual);
                    assert_eq!(actual, initial, "full inverse seed={seed} {variant:?}");
                }
            }
        }
    }
}

#[test]
fn managed_preprocessing_accepts_persistent_migration() {
    let source = prepare_source(6, 3, 41, None).unwrap();
    let sandwich = construct_seeded_sandwich(&source, 6, 3, 6, SandwichVariant::Classic, 47);
    let options = PreprocessingParams::EmbeddedMasking {
        params: EmbeddedMaskingParams {
            shuffling_segments: 8,
            shuffling_return_home: false,
            ..EmbeddedMaskingParams::managed_tdp(43, 6)
        },
        execution: EmbeddedMaskingExecution::default(),
        balanced_seed: true,
        seed_fill_nl: 4,
    };
    let mut rng = StdRng::seed_from_u64(43 ^ 0x6AD6_E75E);
    let result = preprocess_sandwich(
        &sandwich,
        6,
        120,
        SandwichVariant::Classic,
        &options,
        &mut rng,
    );
    // Persistent migration (RC 2026-10-05) is now accepted: the stream tail
    // swaps the data roles back to their physical ports, so the managed
    // stage's fixed-port promise holds without per-box return home.
    assert!(
        result.is_ok(),
        "persistent carrier migration must be accepted"
    );
}
