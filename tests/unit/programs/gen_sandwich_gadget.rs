use super::*;

#[test]
fn shuffling_controls_validate_before_construction() {
    let mode = PreprocessingMode::EmbeddedMasking;
    assert_eq!(
        parse_shuffling_controls(mode, None, None, None).unwrap(),
        (0, true, 0)
    );
    assert_eq!(
        parse_shuffling_controls(mode, Some("8"), Some("1"), None).unwrap(),
        (8, true, 0)
    );
    assert_eq!(
        parse_shuffling_controls(mode, Some("0"), Some("false"), None).unwrap(),
        (0, false, 0)
    );
    for value in ["-1", "eight", "18446744073709551616"] {
        assert!(parse_shuffling_controls(mode, Some(value), None, None).is_err());
    }
    // RC 2026-10-06: small positive doses are valid (1 = one transfer per
    // original-circuit write, the punchlist-#4 granularity).
    assert_eq!(
        parse_shuffling_controls(mode, Some("1"), Some("0"), None).unwrap(),
        (1, false, 0)
    );
    // Persistent carrier migration (RC 2026-10-05): return_home=false with
    // active segments is now valid; the stage-2 tail restores the ports.
    assert_eq!(
        parse_shuffling_controls(mode, Some("8"), Some("false"), None).unwrap(),
        (8, false, 0)
    );
    assert!(parse_shuffling_controls(mode, None, Some("yes"), None).is_err());
    assert!(
        parse_shuffling_controls(PreprocessingMode::Nonlinear291, Some("0"), None, None).is_err()
    );
    // Punchlist #8: just-in-time drains need active persistent migration.
    assert_eq!(
        parse_shuffling_controls(mode, Some("8"), Some("0"), Some("24")).unwrap(),
        (8, false, 24)
    );
    assert_eq!(
        parse_shuffling_controls(mode, Some("8"), Some("0"), Some("off")).unwrap(),
        (8, false, 0)
    );
    assert!(parse_shuffling_controls(mode, Some("8"), Some("0"), Some("soon")).is_err());
    assert!(parse_shuffling_controls(mode, None, Some("0"), Some("24")).is_err());
    assert!(parse_shuffling_controls(mode, Some("8"), Some("1"), Some("24")).is_err());
}

#[test]
fn embedded_masking_has_one_canonical_name() {
    assert_eq!(
        PreprocessingMode::default(),
        PreprocessingMode::EmbeddedMasking
    );
    let mode = PreprocessingMode::parse("embedded-masking").unwrap();
    assert_eq!(mode, PreprocessingMode::EmbeddedMasking);
    assert_eq!(mode.canonical_name(), "embedded-masking");
}

#[test]
fn nonlinear291_is_supported() {
    assert_eq!(
        PreprocessingMode::parse("nonlinear291"),
        Some(PreprocessingMode::Nonlinear291)
    );
}

#[test]
fn unknown_modes_are_rejected() {
    for name in ["unknown-mode", "embedded_masking", "EMBEDDED-MASKING"] {
        assert_eq!(PreprocessingMode::parse(name), None);
    }
}

#[test]
fn managed_preset_is_explicit_and_raw_defaults_stay_unchanged() {
    let raw = EmbeddedMaskingParams::production(19);
    let managed = EmbeddedMaskingParams::managed_tdp(19, 6);
    assert!(!raw.burst_band_only);
    assert_eq!(raw.active_wires, 0);
    assert!(!managed.burst_band_only);
    assert!(EmbeddedMaskingExecution::default().ancilla_band_only);
    assert!(managed.balanced && managed.quad_fire);
    assert!(!managed.encoded_io);
    assert_eq!((managed.k, managed.max_open, managed.min_open), (2, 3, 2));
    assert_eq!((managed.seed, managed.active_wires), (19, 6));
}

#[test]
fn gen_rejects_unparsable_positionals() {
    // D17 (docs/STAGE4_RUST_DESIGN.md §2.2): absent = default, present and
    // unparsable (including empty) = an error naming the argument.
    assert_eq!(parse_positional::<u64>("seed", None).unwrap(), None);
    assert_eq!(
        parse_positional::<u64>("seed", Some("42".into())).unwrap(),
        Some(42)
    );
    for bad in ["", "x", "-1", "1.5", "18446744073709551616", " 7"] {
        let e = parse_positional::<u64>("seed", Some(bad.into())).unwrap_err();
        assert!(e.contains("seed"), "{e}");
    }
    assert!(parse_positional::<usize>("s", Some("abc".into())).is_err());
}

#[test]
fn drift_window_parses_like_the_historical_inline_reader() {
    assert_eq!(parse_drift_window("5:40"), (5, 40));
    // anything unparsable disables the drift everywhere (empty window)
    for bad in ["x", "5", ":40", "5:", "a:b", "5:x"] {
        let (lo, hi) = parse_drift_window(bad);
        assert!((0..100).all(|gi| !(gi >= lo && gi < hi)), "{bad}");
    }
}

#[test]
fn source_construction_env_defaults_to_random_and_rejects_unknown_values() {
    // TDP_SOURCE_CONSTRUCTION: unset or `random` -> Random, `balanced` ->
    // Balanced, anything else (including empty) is an error naming the
    // variable.
    assert_eq!(
        parse_source_construction(None).unwrap(),
        SourceConstruction::Random
    );
    assert_eq!(
        parse_source_construction(Some("random")).unwrap(),
        SourceConstruction::Random
    );
    assert_eq!(
        parse_source_construction(Some("balanced")).unwrap(),
        SourceConstruction::Balanced
    );
    for bad in ["", "Balanced", "RANDOM", "uniform", " balanced"] {
        let e = parse_source_construction(Some(bad)).unwrap_err();
        assert!(e.contains("TDP_SOURCE_CONSTRUCTION"), "{bad:?}: {e}");
    }
}
