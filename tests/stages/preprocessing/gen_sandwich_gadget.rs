use super::*;

#[test]
fn quadratic_masking_default_and_aliases_resolve_to_one_mode() {
    assert_eq!(
        GadgetizationMode::default(),
        GadgetizationMode::QuadraticMasking
    );
    for name in [
        "quadratic-masking",
        "ran-balanced",
        "blinded-v5",
        "blinded_v5",
    ] {
        let mode = GadgetizationMode::parse(name).unwrap();
        assert_eq!(mode, GadgetizationMode::QuadraticMasking);
        assert_eq!(mode.canonical_name(), "quadratic-masking");
        assert_eq!(
            PreprocessingMode::parse(name),
            Some(PreprocessingMode::QuadraticMasking)
        );
    }
    assert_eq!(GadgetizationMode::parse("nonlinear"), None);
}

#[test]
fn nonlinear291_is_supported_without_legacy_dependencies() {
    assert_eq!(
        GadgetizationMode::parse("nonlinear291"),
        Some(GadgetizationMode::Nonlinear291)
    );
    assert_eq!(
        PreprocessingMode::parse("nonlinear291"),
        Some(PreprocessingMode::Nonlinear291)
    );
    for name in ["product-2223", "2223", "nonlinear193"] {
        assert_eq!(PreprocessingMode::parse(name), None);
    }
}

#[cfg(not(feature = "legacy-tools"))]
#[test]
fn comparison_modes_are_absent_from_the_default_build() {
    for name in ["product-2223", "2223", "nonlinear193"] {
        assert_eq!(GadgetizationMode::parse(name), None);
    }
}

#[cfg(feature = "legacy-tools")]
#[test]
fn historical_modes_remain_available_with_legacy_tools() {
    for (name, canonical) in [
        ("2223", "product-2223"),
        ("product-2223", "product-2223"),
        ("nonlinear193", "nonlinear193"),
    ] {
        let mode = GadgetizationMode::parse(name).unwrap();
        assert!(matches!(mode, GadgetizationMode::Legacy(_)));
        assert_eq!(mode.canonical_name(), canonical);
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
