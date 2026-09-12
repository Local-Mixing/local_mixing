use super::*;
use crate::circuit::xgate::eval_lanes;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn gate(kind: TemplateOp) -> XGate {
    match kind {
        TemplateOp::R57 => XGate {
            target: 2,
            comp: true,
            ctrls: [(0, false), (1, true)].into_iter().collect(),
        },
        TemplateOp::Nab => XGate::conj(2, [(0, false), (1, true)]).unwrap(),
        TemplateOp::And => XGate::conj(2, [(0, true), (1, true)]).unwrap(),
        TemplateOp::Copy => XGate::cnot(2, 0),
    }
}

fn zero_slice_columns(vary_n: usize, total: usize) -> Vec<u64> {
    assert!(vary_n <= 6);
    let samples = 1usize << vary_n;
    let mut state = vec![0u64; total];
    for (wire, column) in state.iter_mut().take(vary_n).enumerate() {
        for sample in 0..samples {
            *column |= (((sample >> wire) & 1) as u64) << sample;
        }
    }
    state
}

fn assert_zero_slice_matches(source: &[XGate], mode: NonlinearGssMode, n: usize, data_n: usize) {
    let mut rng = StdRng::seed_from_u64(9173);
    let gadget = gadgetize_xgates_nonlinear_gss(source, n, data_n, 0, mode, &mut rng).unwrap();
    // Exhaust all assignments to the stable data prefix; remaining raw
    // logical wires and every adapter auxiliary start at zero.
    let mut got = zero_slice_columns(data_n, gadget.num_wires);
    let mut want = got[..n].to_vec();
    eval_lanes(source, &mut want);
    eval_lanes(&gadget.gates, &mut got);
    assert_eq!(&got[..n], want.as_slice());

    let layout = build_layout(n, source.len(), mode).unwrap();
    assert_eq!(got[layout.shared.scratch as usize], 0);
    assert_eq!(got[layout.shared.scratch2 as usize], 0);
    if let Some(decomp) = layout.shared.decomp {
        for wire in decomp {
            assert_eq!(got[wire as usize], 0);
        }
    }
    if let Some(persistent) = layout.shared.persistent {
        for wire in persistent {
            assert_eq!(got[wire as usize], 0);
        }
    }
    if let Some(temporary) = layout.shared.temporary {
        assert_eq!(got[temporary as usize], 0);
    }
}

#[test]
fn canonical_templates_have_pinned_shapes() {
    for mode in [
        #[cfg(feature = "legacy-tools")]
        NonlinearGssMode::Nonlinear193,
        NonlinearGssMode::Nonlinear291,
    ] {
        for op in [
            TemplateOp::R57,
            TemplateOp::Nab,
            TemplateOp::And,
            TemplateOp::Copy,
        ] {
            let template = canonical_template(mode, op).unwrap();
            let (wires, gates, max_fanin) = expected_template_shape(mode, op);
            assert_eq!(template.num_wires, wires);
            assert_eq!(template.gates.len(), gates);
            assert_eq!(
                template
                    .gates
                    .iter()
                    .map(|gate| gate.ctrls.len())
                    .max()
                    .unwrap(),
                max_fanin
            );
        }
    }
}

#[test]
fn canonical_template_parser_bounds_headers_before_allocation() {
    let huge = usize::MAX;
    let error = parse_template(
        &format!("mpmct1 71 {huge}"),
        NonlinearGssMode::Nonlinear291,
        TemplateOp::R57,
    )
    .unwrap_err();
    assert!(error.contains("canonical header mismatch"), "{error}");

    let error = parse_template(
        &format!("mpmct1 71 291 0 0 {huge}"),
        NonlinearGssMode::Nonlinear291,
        TemplateOp::R57,
    )
    .unwrap_err();
    assert!(error.contains("fan-in"), "{error}");
}

#[test]
fn every_supported_operation_matches_on_the_zero_slice() {
    for mode in [
        #[cfg(feature = "legacy-tools")]
        NonlinearGssMode::Nonlinear193,
        NonlinearGssMode::Nonlinear291,
    ] {
        for op in [
            TemplateOp::R57,
            TemplateOp::Nab,
            TemplateOp::And,
            TemplateOp::Copy,
        ] {
            assert_zero_slice_matches(&[gate(op)], mode, 12, 3);
        }
    }
}

#[test]
fn heterogeneous_chain_tracks_relabelled_targets() {
    let source = vec![
        XGate {
            target: 2,
            comp: true,
            ctrls: [(0, false), (1, true)].into_iter().collect(),
        },
        XGate::conj(3, [(2, true), (0, true)]).unwrap(),
        XGate::conj(1, [(3, false), (2, true)]).unwrap(),
        XGate::cnot(0, 1),
    ];
    for mode in [
        #[cfg(feature = "legacy-tools")]
        NonlinearGssMode::Nonlinear193,
        NonlinearGssMode::Nonlinear291,
    ] {
        assert_zero_slice_matches(&source, mode, 12, 3);
    }
}

#[test]
fn gss_second_half_zero_slice_is_correct_and_ingress_is_masked() {
    const DATA_N: usize = 6;
    const LOGICAL_N: usize = 2 * DATA_N;
    let source = vec![
        XGate {
            target: 0,
            comp: true,
            ctrls: [(1, false), (2, true)].into_iter().collect(),
        },
        XGate::cnot(DATA_N as u16, 0),
        XGate::conj(1, [(2, true), (DATA_N as u16, true)]).unwrap(),
    ];
    for mode in [
        #[cfg(feature = "legacy-tools")]
        NonlinearGssMode::Nonlinear193,
        NonlinearGssMode::Nonlinear291,
    ] {
        let mut rng = StdRng::seed_from_u64(0x51ced);
        let gadget =
            gadgetize_xgates_nonlinear_gss(&source, LOGICAL_N, DATA_N, 0, mode, &mut rng).unwrap();
        // Only the source-data half varies. The sandwich/slice half and
        // every adapter auxiliary start at zero.
        let mut got = zero_slice_columns(DATA_N, gadget.num_wires);
        let mut want = got[..LOGICAL_N].to_vec();
        eval_lanes(&source, &mut want);
        eval_lanes(&gadget.gates, &mut got);
        assert_eq!(&got[..LOGICAL_N], want.as_slice());

        // A free carrier and strict per-gate borrow are balanced data
        // functions, not the all-zero columns of the minimal ingress.
        let layout = build_layout(LOGICAL_N, source.len(), mode).unwrap();
        assert_eq!(got[layout.values[0].share1[1] as usize].count_ones(), 32);
        assert_eq!(got[layout.gate_wires[0].chaff[0] as usize].count_ones(), 32);
    }
}

#[test]
fn public_capacity_planner_matches_layout_and_rejects_overflow() {
    for mode in [
        #[cfg(feature = "legacy-tools")]
        NonlinearGssMode::Nonlinear193,
        NonlinearGssMode::Nonlinear291,
    ] {
        let planned = nonlinear_gss_wire_count(6, 17, mode).unwrap();
        assert_eq!(build_layout(6, 17, mode).unwrap().total, planned);
    }
    let error = nonlinear_gss_wire_count(usize::MAX, usize::MAX, NonlinearGssMode::Nonlinear291)
        .unwrap_err();
    assert!(error.contains("overflow"), "{error}");
}

#[test]
fn nonlinear291_complete_adapter_is_fan_in_two() {
    let mut rng = StdRng::seed_from_u64(12);
    let gadget = gadgetize_xgates_nonlinear_gss(
        &[gate(TemplateOp::R57)],
        12,
        3,
        0,
        NonlinearGssMode::Nonlinear291,
        &mut rng,
    )
    .unwrap();
    assert!(gadget.gates.iter().all(|gate| gate.ctrls.len() <= 2));
}

#[test]
fn nonlinear_slice_gate_override_is_resource_bounded() {
    let mut rng = StdRng::seed_from_u64(1);
    let error = gadgetize_xgates_nonlinear_gss(
        &[],
        3,
        3,
        usize::MAX,
        NonlinearGssMode::Nonlinear291,
        &mut rng,
    )
    .unwrap_err();
    assert!(error.contains("bounded limit"), "{error}");
}

#[test]
fn seeded_build_is_deterministic() {
    let source = [gate(TemplateOp::R57), gate(TemplateOp::Copy)];
    let mut first_rng = StdRng::seed_from_u64(44);
    let first = gadgetize_xgates_nonlinear_gss(
        &source,
        12,
        3,
        7,
        NonlinearGssMode::Nonlinear291,
        &mut first_rng,
    )
    .unwrap();
    let mut second_rng = StdRng::seed_from_u64(44);
    let second = gadgetize_xgates_nonlinear_gss(
        &source,
        12,
        3,
        7,
        NonlinearGssMode::Nonlinear291,
        &mut second_rng,
    )
    .unwrap();
    assert_eq!(first.num_wires, second.num_wires);
    assert_eq!(first.gates, second.gates);
}

#[test]
fn unsupported_source_shape_reports_gate_index() {
    let unsupported = XGate::conj(2, [(0, false)]).unwrap();
    let mut rng = StdRng::seed_from_u64(1);
    let error = gadgetize_xgates_nonlinear_gss(
        &[unsupported],
        3,
        3,
        0,
        NonlinearGssMode::Nonlinear291,
        &mut rng,
    )
    .unwrap_err();
    assert!(error.contains("source gate 0"), "{error}");
    assert!(error.contains("unsupported"), "{error}");
}

#[test]
fn capacity_failure_precedes_any_narrowing_cast() {
    let mut rng = StdRng::seed_from_u64(1);
    let error =
        gadgetize_xgates_nonlinear_gss(&[], 6000, 3, 0, NonlinearGssMode::Nonlinear291, &mut rng)
            .unwrap_err();
    assert!(error.contains("needs 72029 wires"), "{error}");
    assert!(error.contains("65535"), "{error}");
}
