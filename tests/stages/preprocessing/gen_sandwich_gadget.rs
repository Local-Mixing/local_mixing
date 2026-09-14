use super::*;

#[test]
fn quadratic_masking_default_and_aliases_resolve_to_one_mode() {
    assert_eq!(
        PreprocessingMode::default(),
        PreprocessingMode::QuadraticMasking
    );
    for name in [
        "quadratic-masking",
        "ran-balanced",
        "blinded-v5",
        "blinded_v5",
    ] {
        let mode = PreprocessingMode::parse(name).unwrap();
        assert_eq!(mode, PreprocessingMode::QuadraticMasking);
        assert_eq!(mode.canonical_name(), "quadratic-masking");
    }
    assert_eq!(PreprocessingMode::parse("nonlinear"), None);
}

#[test]
fn nonlinear291_is_supported() {
    assert_eq!(
        PreprocessingMode::parse("nonlinear291"),
        Some(PreprocessingMode::Nonlinear291)
    );
}

#[test]
fn retired_modes_are_rejected() {
    for name in ["product-2223", "2223", "nonlinear193"] {
        assert_eq!(PreprocessingMode::parse(name), None);
    }
}

#[test]
fn managed_preset_is_explicit_and_raw_defaults_stay_unchanged() {
    let raw = QuadraticMaskingParams::production(19);
    let managed = QuadraticMaskingParams::managed_gss(19, 6);
    assert!(!raw.burst_band_only);
    assert_eq!(raw.active_wires, 0);
    assert!(!managed.burst_band_only);
    assert!(QuadraticMaskingExecution::default().ancilla_band_only);
    assert!(managed.balanced && managed.quad_fire);
    assert!(!managed.encoded_io);
    assert_eq!((managed.k, managed.max_open, managed.min_open), (2, 3, 2));
    assert_eq!((managed.seed, managed.active_wires), (19, 6));
}
