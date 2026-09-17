use super::*;

fn tiny_config() -> DetectorConfig {
    DetectorConfig {
        train_batches: 1,
        heldout_batches: 1,
        max_original_segments: 8,
        max_original_firings: 16,
        min_minority_count: 1,
        ..DetectorConfig::default()
    }
}

#[test]
fn controls_do_not_split_constant_value_segments() {
    let gates = vec![XGate::x_gate(0), XGate::cnot(1, 0), XGate::x_gate(0)];
    let detector = Detector::new(&gates, 2, tiny_config()).unwrap();
    let report = detector.scan(&gates).unwrap();
    assert_eq!(report.internal_segments, 1);
    assert_eq!(report.boundary_segments, 4);
    assert_eq!(report.unmappable_firing_gates, 2);
    assert_eq!(report.hot_segment_count, 1);
    let hot = &report.hot_segments[0];
    assert_eq!((hot.wire, hot.start_gate, hot.end_gate), (0, 0, 2));
    assert!(
        hot.evidence
            .iter()
            .any(|e| matches!(e, Evidence::Affine { .. }))
    );
}

#[test]
fn affine_xor_relation_reports_the_reference_terms_and_offset() {
    let config = tiny_config();
    let a = vec![0xaaaa_aaaa_aaaa_aaaa, 0xcccc_cccc_cccc_cccc];
    let b = vec![0xf0f0_f0f0_f0f0_f0f0, 0xff00_ff00_ff00_ff00];
    let basis = AffineBasis::new(&[(7, a.clone()), (19, b.clone())], 1, 2);
    let target = vec![!(a[0] ^ b[0]), !(a[1] ^ b[1])];
    match basis.predict(&target, &config).unwrap() {
        Evidence::Affine {
            original_features,
            complement,
            heldout_accuracy,
        } => {
            assert_eq!(original_features, vec![7, 19]);
            assert!(complement);
            assert_eq!(heldout_accuracy, 1.0);
        }
        _ => panic!("expected an affine relation"),
    }
}

#[test]
fn affine_training_fit_does_not_bypass_heldout_validation() {
    let config = tiny_config();
    let feature = vec![0xaaaa_aaaa_aaaa_aaaa, 0xcccc_cccc_cccc_cccc];
    let basis = AffineBasis::new(&[(0, feature.clone())], 1, 2);
    // Perfect training fit; completely wrong on a separate sample bank.
    assert!(basis.predict(&[feature[0], !feature[1]], &config).is_none());
    assert!(basis.predict(&[0, 0], &config).is_none());
}

#[test]
fn sparse_predicates_use_phi_rather_than_raw_agreement() {
    // 62/64 raw agreement, but these singleton events are not correlated.
    assert!(phi(&[1], &[2], 1).unwrap().abs() < 0.02);
    assert_eq!(phi(&[0], &[0], 1), None);
    assert_eq!(phi(&[0xaaaa], &[!0xaaaa], 1), Some(-1.0));
}

#[test]
fn firing_correlation_must_validate_with_the_same_sign() {
    let gates = vec![XGate::cnot(0, 1)];
    let mut detector = Detector::new(&gates, 2, tiny_config()).unwrap();
    let train = 0xaaaa_aaaa_aaaa_aaaa;
    let heldout = 0xcccc_cccc_cccc_cccc;
    detector.original_firings = vec![(0, vec![train, heldout])];
    assert!(detector.firing_evidence(7, &[train, !heldout]).is_none());
    match detector.firing_evidence(7, &[!train, !heldout]).unwrap() {
        Evidence::FiringCorrelation {
            mixed_gate,
            train_correlation,
            heldout_correlation,
            ..
        } => {
            assert_eq!(mixed_gate, 7);
            assert_eq!((train_correlation, heldout_correlation), (-1.0, -1.0));
        }
        _ => panic!("expected a validated negative correlation"),
    }
}

#[test]
fn firing_correlation_targets_its_producers_outgoing_segment() {
    let gates = vec![XGate::cnot(0, 1), XGate::cnot(2, 0), XGate::x_gate(0)];
    let config = DetectorConfig {
        max_original_segments: 0,
        ..tiny_config()
    };
    let report = Detector::new(&gates, 3, config)
        .unwrap()
        .scan(&gates)
        .unwrap();
    assert_eq!(report.hot_segment_count, 1);
    assert_eq!(
        (
            report.hot_segments[0].start_gate,
            report.hot_segments[0].end_gate
        ),
        (0, 2)
    );
    match &report.hot_segments[0].evidence[0] {
        Evidence::FiringCorrelation {
            mixed_gate,
            original_gate,
            train_correlation,
            heldout_correlation,
        } => {
            assert_eq!((*mixed_gate, *original_gate), (0, 0));
            assert_eq!((*train_correlation, *heldout_correlation), (1.0, 1.0));
        }
        _ => panic!("expected firing evidence"),
    }
}

#[test]
fn region_scan_includes_boundary_crossings_and_reports_coverage() {
    let gates = vec![XGate::x_gate(0); 8];
    let config = DetectorConfig {
        max_mixed_segments: 2,
        ..tiny_config()
    };
    let detector = Detector::new(&gates, 1, config).unwrap();
    let global = detector.scan(&gates).unwrap();
    assert!(global.truncated);
    assert_eq!((global.internal_segments, global.scanned_segments), (7, 2));
    let region = detector.scan_region(&gates, 3..4).unwrap();
    assert!(!region.truncated);
    assert_eq!((region.internal_segments, region.scanned_segments), (2, 2));
    assert_eq!(
        region
            .hot_segments
            .iter()
            .map(|h| (h.start_gate, h.end_gate))
            .collect::<Vec<_>>(),
        vec![(2, 3), (3, 4)]
    );
    assert!(detector.scan_region(&gates, 0..9).is_err());
}

#[test]
fn hot_record_limit_does_not_hide_total_hits() {
    let gates = vec![XGate::x_gate(0); 5];
    let config = DetectorConfig {
        max_hot_segments: 1,
        ..tiny_config()
    };
    let report = Detector::new(&gates, 1, config)
        .unwrap()
        .scan(&gates)
        .unwrap();
    assert_eq!(report.hot_segment_count, 4);
    assert_eq!(report.hot_segments.len(), 1);
    assert!(report.hot_segments_truncated);
    assert!(!report.truncated);
}

#[test]
fn moving_a_hot_firing_to_the_output_fringe_is_still_reported() {
    let original = vec![XGate::cnot(0, 1), XGate::cnot(0, 1)];
    let current = vec![XGate::cnot(0, 1)];
    let config = DetectorConfig {
        max_original_segments: 0,
        ..tiny_config()
    };
    let detector = Detector::new(&original, 2, config).unwrap();
    let report = detector.scan_region(&current, 0..1).unwrap();
    assert_eq!(report.hot_segment_count, 0);
    assert_eq!(report.hot_boundary_firings, 1);
    assert_eq!(report.scanned_boundary_firings, 1);
    assert!(!report.truncated);
    assert!(matches!(
        report.boundary_firing_evidence[0],
        Evidence::FiringCorrelation { mixed_gate: 0, .. }
    ));
}

#[test]
fn boundary_firing_budget_cannot_report_complete_coverage() {
    let gates = vec![XGate::cnot(0, 2), XGate::cnot(1, 2)];
    let config = DetectorConfig {
        max_original_segments: 0,
        max_mixed_segments: 1,
        ..tiny_config()
    };
    let report = Detector::new(&gates, 3, config)
        .unwrap()
        .scan(&gates)
        .unwrap();
    assert_eq!(report.boundary_firings, 2);
    assert_eq!(report.scanned_boundary_firings, 1);
    assert!(report.truncated);
}

#[test]
fn samples_are_deterministic_and_extra_input_wires_are_independent() {
    let original = vec![XGate::x_gate(0); 2];
    let current = vec![XGate::x_gate(1); 2];
    let config = DetectorConfig {
        max_original_firings: 0,
        ..tiny_config()
    };
    let detector = Detector::new(&original, 2, config.clone()).unwrap();
    let other = Detector::new(&original, 2, config).unwrap();
    assert_eq!(detector.inputs, other.inputs);
    assert_ne!(detector.inputs[0][0], detector.inputs[0][1]);
    assert_eq!(detector.scan(&current).unwrap().hot_segment_count, 0);
    assert!(detector.scan(&[XGate::x_gate(2)]).is_err());
}

#[test]
fn fresh_audit_changes_samples_without_changing_reference_selection() {
    let original = vec![XGate::cnot(0, 1); 30];
    let config = tiny_config();
    let original_seed = config.seed;
    let used = Detector::new(&original, 2, config.clone()).unwrap();
    let audit =
        Detector::new_with_sample_seed(&original, 2, config, original_seed ^ 0x6672_6573_685f_7163)
            .unwrap();
    assert_ne!(used.inputs, audit.inputs);
    assert_eq!(used.affine.feature_gates, audit.affine.feature_gates);
    assert_eq!(
        used.original_firings
            .iter()
            .map(|x| x.0)
            .collect::<Vec<_>>(),
        audit
            .original_firings
            .iter()
            .map(|x| x.0)
            .collect::<Vec<_>>()
    );
}

#[test]
fn storage_and_sample_limits_fail_before_allocating_traces() {
    let gates = vec![XGate::x_gate(0); 2];
    let config = DetectorConfig {
        max_trace_words: 1,
        ..tiny_config()
    };
    assert!(Detector::new(&gates, 2, config).is_err());
    let config = DetectorConfig {
        max_original_segments: 64,
        ..tiny_config()
    };
    assert!(Detector::new(&gates, 2, config).is_err());
    let config = DetectorConfig {
        min_abs_correlation: f64::NAN,
        ..tiny_config()
    };
    assert!(Detector::new(&gates, 2, config).is_err());
}
