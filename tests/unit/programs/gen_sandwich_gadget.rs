use super::*;

#[test]
fn shuffling_controls_validate_before_construction() {
    let mode = PreprocessingMode::EmbeddedMasking;
    assert_eq!(
        parse_shuffling_controls(mode, None, None).unwrap(),
        (0, true)
    );
    assert_eq!(
        parse_shuffling_controls(mode, Some("8"), Some("1")).unwrap(),
        (8, true)
    );
    assert_eq!(
        parse_shuffling_controls(mode, Some("0"), Some("false")).unwrap(),
        (0, false)
    );
    for value in ["1", "7", "-1", "eight", "18446744073709551616"] {
        assert!(parse_shuffling_controls(mode, Some(value), None).is_err());
    }
    assert!(parse_shuffling_controls(mode, Some("8"), Some("false")).is_err());
    assert!(parse_shuffling_controls(mode, None, Some("yes")).is_err());
    assert!(parse_shuffling_controls(PreprocessingMode::Nonlinear291, Some("0"), None).is_err());
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
