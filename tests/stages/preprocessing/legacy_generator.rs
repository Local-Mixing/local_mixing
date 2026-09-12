use super::*;

#[test]
fn five_carrier_is_an_explicit_standalone_preset() {
    let (_, default_mode) = production_preset(None);
    let (five, five_mode) = production_preset(Some("five-carrier"));
    let (strong_five, strong_five_mode) = production_preset(Some("strong-five-carrier"));
    assert_eq!(default_mode, CarrierMode::Single);
    assert_eq!(five_mode, CarrierMode::Five);
    assert_eq!(strong_five_mode, CarrierMode::StrongFive);
    assert!(five.enabled());
    assert_eq!(strong_five, five);
    assert_eq!(five.k_total(), 4, "five-carrier production mask plan");
}

#[test]
fn six_carrier_is_an_explicit_standalone_preset() {
    let (six, six_mode) = production_preset(Some("six-carrier"));
    let (strong_six, strong_six_mode) = production_preset(Some("strong-six-carrier"));
    assert_eq!(six_mode, CarrierMode::Six);
    assert_eq!(strong_six_mode, CarrierMode::StrongSix);
    assert!(six.enabled());
    assert_eq!(strong_six, six);
    assert_eq!(six.k_total(), 4, "six-carrier production mask plan");
}

#[test]
fn seven_carrier_is_an_explicit_standalone_preset() {
    let (seven, seven_mode) = production_preset(Some("seven-carrier"));
    assert_eq!(seven_mode, CarrierMode::Seven);
    assert!(seven.enabled());
    assert_eq!(seven.k_total(), 4, "seven-carrier production mask plan");
}

#[test]
fn fold_and_post_fragment_study_presets_are_explicit() {
    let (micro, micro_mode) = production_preset(Some("micro-gray"));
    let (sentinel, sentinel_mode) = production_preset(Some("sentinel-gray"));
    let (native, native_mode) = production_preset(Some("no-gray-post-native"));
    assert_eq!(micro_mode, CarrierMode::Single);
    assert_eq!(sentinel_mode, CarrierMode::Single);
    assert_eq!(native_mode, CarrierMode::Single);
    assert_eq!(micro.gray_fold, 2);
    assert_eq!(sentinel.gray_fold, 3);
    assert_eq!(native.gray_fold, 0);
    assert_eq!(preset_post_fragment(Some("micro-gray")), None);
    assert_eq!(
        preset_post_fragment(Some("no-gray-post-native")),
        Some(FragmentStyle::NativeDeep)
    );
    assert_eq!(parse_post_fragment("off"), Some(None));
    assert_eq!(
        parse_post_fragment("exact"),
        Some(Some(FragmentStyle::Exact))
    );
    assert_eq!(parse_post_fragment("bogus"), None);
}

#[test]
#[should_panic(expected = "unknown PROD_PRESET")]
fn unknown_standalone_preset_is_rejected() {
    let _ = production_preset(Some("not-a-preset"));
}

#[test]
fn gadgetization_modes_have_stable_canonical_names() {
    assert_eq!(
        GadgetizationMode::parse("2223"),
        Some(GadgetizationMode::Product2223)
    );
    assert_eq!(
        GadgetizationMode::parse("product-2223")
            .unwrap()
            .canonical_name(),
        "product-2223"
    );
    assert_eq!(
        GadgetizationMode::parse("nonlinear193"),
        Some(GadgetizationMode::Nonlinear193)
    );
    assert_eq!(
        GadgetizationMode::parse("nonlinear291"),
        Some(GadgetizationMode::Nonlinear291)
    );
    assert_eq!(GadgetizationMode::parse("nonlinear"), None);
}
