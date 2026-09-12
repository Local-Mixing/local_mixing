use super::*;
use crate::stages::preprocessing::quadratic_masking::{
    QuadraticMaskingExecution, QuadraticMaskingParams,
};
use crate::stages::preprocessing::verify::verify_payload;
use crate::stages::sandwich::{construct_seeded_sandwich, prepare_source};
use rand::{SeedableRng, rngs::StdRng};

// Stable structural digest of the pre-refactor CLI's mpmct1 artifacts. Every
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
    // Captured before this ownership extraction: n=6, |C|=|D|=3, s=6,
    // slice_gates=120, source_seed=41, preprocessing_seed=43, sandwich_seed=47.
    for (variant, band_only, expected) in [
        (SandwichVariant::Classic, false, 0x71ab64ed195601f7),
        (SandwichVariant::Classic, true, 0x404fa955da991cca),
        (SandwichVariant::Balanced, false, 0xe28764ca880470dc),
    ] {
        let source = prepare_source(6, 3, 41, None).unwrap();
        let sandwich = construct_seeded_sandwich(&source, 6, 3, 6, variant, 47);
        let options = PreprocessingParams::QuadraticMasking {
            params: if band_only {
                QuadraticMaskingParams {
                    burst_band_only: true,
                    ..QuadraticMaskingParams::managed_gss(43, 6)
                }
            } else if variant == SandwichVariant::Balanced {
                QuadraticMaskingParams {
                    active_wires: 6,
                    ..QuadraticMaskingParams::production(43)
                }
            } else {
                QuadraticMaskingParams::managed_gss(43, 6)
            },
            execution: QuadraticMaskingExecution::default(),
            balanced_seed: true,
            record_hot_intervals: false,
        };
        let mut rng = StdRng::seed_from_u64(43 ^ 0x6AD6_E75E);
        let output = preprocess_sandwich(&sandwich, 6, 120, variant, &options, &mut rng).unwrap();
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
    // product/quadratic masking output junk guard that promises reverse ports.
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
