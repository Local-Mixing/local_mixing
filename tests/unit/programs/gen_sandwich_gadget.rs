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
fn three_mask_is_supported() {
    let mode = PreprocessingMode::parse("three-mask").unwrap();
    assert_eq!(mode, PreprocessingMode::ThreeMask);
    assert_eq!(mode.canonical_name(), "three-mask");
    assert_eq!(PreprocessingMode::parse("three_mask"), None);
}

#[test]
fn three_mask_knobs_default_to_v2_and_reject_bad_values() {
    let none = |_: &str| None;
    let defaults = three_mask_params_from(19, none, &[]).unwrap();
    let v2 = ThreeMaskParams::v2(19, 0);
    assert_eq!(format!("{defaults:?}"), format!("{v2:?}"));
    let env = |pairs: &'static [(&'static str, &'static str)]| {
        move |key: &str| {
            pairs
                .iter()
                .find(|(k, _)| *k == key)
                .map(|(_, v)| v.to_string())
        }
    };
    let names = |pairs: &[(&str, &str)]| -> Vec<String> {
        pairs.iter().map(|(k, _)| k.to_string()).collect()
    };
    let set: &[(&str, &str)] = &[
        ("THREE_MASK_BAND_WIRES", "192"),
        ("THREE_MASK_BANKS", "16"),
        ("THREE_MASK_RETIRE_BUDGET", "4"),
        ("THREE_MASK_MAX_RETIRING", "3"),
        ("THREE_MASK_ROTATION", "2"),
        ("THREE_MASK_MATURITY", "3"),
        ("THREE_MASK_VISIT_SWEEPS", "1"),
        ("THREE_MASK_GROWTH", "16"),
        ("THREE_MASK_WARMUP", "2"),
        ("THREE_MASK_SEGMENTS", "8"),
        ("THREE_MASK_MASK_COUNT", "2"),
        ("THREE_MASK_RANDOM_HELPERS", "0"),
        ("THREE_MASK_UNIQUE_COVER", "false"),
    ];
    let p = three_mask_params_from(19, env(set), &names(set)).unwrap();
    assert_eq!(
        (
            p.band_wires,
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
        (192, 16, 4, 3, 2, 3, 1, 16, 2, 8, 2)
    );
    assert!(!p.random_helpers && !p.unique_cover);
    // Everything else stays at v2.
    assert!(p.temporal_masks && p.cubic_chain && p.discard_fire_helpers && p.seed == 19);
    for bad in [
        &[("THREE_MASK_BANKS", "eight")][..],
        &[("THREE_MASK_GROWTH", "-1")],
        &[("THREE_MASK_BAND_WIRES", "")],
        &[("THREE_MASK_RANDOM_HELPERS", "yes")],
        &[("THREE_MASK_UNIQUE_COVER", "2")],
    ] {
        let bad: &'static [(&'static str, &'static str)] =
            Box::leak(bad.to_vec().into_boxed_slice());
        let error = three_mask_params_from(19, env(bad), &names(bad)).unwrap_err();
        assert!(error.contains(bad[0].0), "{error}");
    }
    let typo = names(&[("THREE_MASK_BAND_WIRE", "128")]);
    let error = three_mask_params_from(19, none, &typo).unwrap_err();
    assert!(
        error.contains("unknown three-mask control(s): THREE_MASK_BAND_WIRE"),
        "{error}"
    );
}

#[test]
fn modes_refuse_each_others_controls() {
    let names = |keys: &[&str]| -> Vec<String> { keys.iter().map(|k| k.to_string()).collect() };
    assert!(
        foreign_mode_controls(
            PreprocessingMode::ThreeMask,
            &names(&["PATH", "THREE_MASK_BANKS"])
        )
        .is_ok()
    );
    let error = foreign_mode_controls(
        PreprocessingMode::ThreeMask,
        &names(&[
            "SLICE_GUARD_LEGACY",
            "EMBEDDED_MASKING_SHUFFLING",
            "EMBEDDED_MASKING_CHAIN",
        ]),
    )
    .unwrap_err();
    assert_eq!(
        error,
        "three-mask fixes the outer blocks to preset L8; unset EMBEDDED_MASKING_CHAIN, EMBEDDED_MASKING_SHUFFLING, SLICE_GUARD_LEGACY"
    );
    for mode in [
        PreprocessingMode::EmbeddedMasking,
        PreprocessingMode::Nonlinear291,
    ] {
        assert!(foreign_mode_controls(mode, &names(&["EMBEDDED_MASKING_PREMASK"])).is_ok());
        let error = foreign_mode_controls(mode, &names(&["THREE_MASK_BANKS"])).unwrap_err();
        assert_eq!(
            error,
            "THREE_MASK_* controls require the three-mask mode; unset THREE_MASK_BANKS"
        );
    }
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
