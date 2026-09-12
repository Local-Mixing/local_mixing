//! Optional product/nonlinear generator controls for historical security workflows.

use crate::circuit::wide_fragment::{FragmentStyle, fragment_wide_post_shuffle};
use crate::preprocessing::gadgets::{
    CnotCircuit, MaskConfig, ProdConfig, gadgetize_xgates_with_slice_zero_ccnot,
    gadgetize_xgates_with_slice_zero_ccnot_five_carrier,
    gadgetize_xgates_with_slice_zero_ccnot_seven_carrier,
    gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed,
    gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed_partitioned_floor1024_live_prefix,
    gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed_partitioned_floor1024_live_prefix_unshuffled,
    gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed_partitioned_live_prefix,
    gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed_partitioned_live_prefix_unshuffled,
    gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed_unshuffled,
    gadgetize_xgates_with_slice_zero_ccnot_single,
    gadgetize_xgates_with_slice_zero_ccnot_six_carrier,
    gadgetize_xgates_with_slice_zero_ccnot_strong_five_carrier,
    gadgetize_xgates_with_slice_zero_ccnot_strong_six_carrier,
};
use crate::preprocessing::nonlinear_gss::{
    NonlinearGssMode, gadgetize_xgates_nonlinear_gss, nonlinear_gss_resource_plan,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CarrierMode {
    Single,
    Five,
    StrongFive,
    Six,
    StrongSix,
    Seven,
    SevenDistributed,
    SevenDistributedUnshuffled,
    SevenDistributedPartitioned,
    SevenDistributedPartitionedUnshuffled,
    SevenDistributedPartitionedFloor1024,
    SevenDistributedPartitionedFloor1024Unshuffled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GadgetizationMode {
    Product2223,
    Nonlinear193,
    Nonlinear291,
}

impl GadgetizationMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "product-2223" | "2223" => Some(Self::Product2223),
            "nonlinear193" => Some(Self::Nonlinear193),
            "nonlinear291" => Some(Self::Nonlinear291),
            _ => None,
        }
    }

    pub fn canonical_name(self) -> &'static str {
        match self {
            Self::Product2223 => "product-2223",
            Self::Nonlinear193 => "nonlinear193",
            Self::Nonlinear291 => "nonlinear291",
        }
    }

    fn nonlinear(self) -> Option<NonlinearGssMode> {
        match self {
            Self::Product2223 => None,
            Self::Nonlinear193 => Some(NonlinearGssMode::Nonlinear193),
            Self::Nonlinear291 => Some(NonlinearGssMode::Nonlinear291),
        }
    }
}

fn production_preset(name: Option<&str>) -> (ProdConfig, CarrierMode) {
    let with_gray_mode = |mode| {
        let mut config = ProdConfig::production_single();
        config.gray_fold = mode;
        config
    };
    match name {
        Some("five-carrier") => (ProdConfig::production_five_carrier(), CarrierMode::Five),
        Some("strong-five-carrier") => (
            ProdConfig::production_five_carrier(),
            CarrierMode::StrongFive,
        ),
        Some("six-carrier") => (ProdConfig::production_six_carrier(), CarrierMode::Six),
        Some("strong-six-carrier") => {
            (ProdConfig::production_six_carrier(), CarrierMode::StrongSix)
        }
        Some("seven-carrier") => (ProdConfig::production_seven_carrier(), CarrierMode::Seven),
        Some("seven-carrier-shear") => (
            ProdConfig::production_seven_carrier(),
            CarrierMode::SevenDistributed,
        ),
        Some("seven-carrier-shear-unshuffled") => (
            ProdConfig::production_seven_carrier(),
            CarrierMode::SevenDistributedUnshuffled,
        ),
        Some("seven-carrier-partitioned") => {
            let mut config = ProdConfig::production_seven_carrier();
            config.gray_fold = 0;
            (config, CarrierMode::SevenDistributedPartitioned)
        }
        Some("seven-carrier-partitioned-unshuffled") => {
            let mut config = ProdConfig::production_seven_carrier();
            config.gray_fold = 0;
            (config, CarrierMode::SevenDistributedPartitionedUnshuffled)
        }
        Some("seven-carrier-partitioned-floor1024") => {
            let mut config = ProdConfig::production_seven_carrier();
            config.gray_fold = 0;
            (config, CarrierMode::SevenDistributedPartitionedFloor1024)
        }
        Some("seven-carrier-partitioned-floor1024-unshuffled") => {
            let mut config = ProdConfig::production_seven_carrier();
            config.gray_fold = 0;
            (
                config,
                CarrierMode::SevenDistributedPartitionedFloor1024Unshuffled,
            )
        }
        Some("no-gray-phase-a") => (
            ProdConfig::production_single_no_gray_phase_a(),
            CarrierMode::Single,
        ),
        Some("micro-gray") => (with_gray_mode(2), CarrierMode::Single),
        Some("sentinel-gray") => (with_gray_mode(3), CarrierMode::Single),
        Some("no-gray-post-exact") | Some("no-gray-post-native") => (
            ProdConfig::production_single_no_gray_phase_a(),
            CarrierMode::Single,
        ),
        Some("production") | None => (ProdConfig::production_single(), CarrierMode::Single),
        Some(other) => panic!(
            "unknown PROD_PRESET={other:?}; expected production, no-gray-phase-a, micro-gray, sentinel-gray, no-gray-post-exact, no-gray-post-native, five-carrier, strong-five-carrier, six-carrier, strong-six-carrier, seven-carrier, seven-carrier-shear, seven-carrier-shear-unshuffled, seven-carrier-partitioned[-unshuffled], or seven-carrier-partitioned-floor1024[-unshuffled]"
        ),
    }
}

fn preset_post_fragment(name: Option<&str>) -> Option<FragmentStyle> {
    match name {
        Some("no-gray-post-exact") => Some(FragmentStyle::Exact),
        Some("no-gray-post-native") => Some(FragmentStyle::NativeDeep),
        _ => None,
    }
}

fn parse_post_fragment(value: &str) -> Option<Option<FragmentStyle>> {
    match value {
        "" | "0" | "off" | "none" => Some(None),
        other => FragmentStyle::parse(other).map(Some),
    }
}

impl GadgetizationMode {
    pub fn preflight(self, n: usize, m_c: usize, m_d: usize, s: usize, slice_gates: usize) {
        let gadgetization_mode = self;
        let sandwich_n = 2 * n;
        if let Some(mode) = gadgetization_mode.nonlinear() {
            let sandwich_gate_count = m_c
                .checked_add(m_d)
                .and_then(|count| count.checked_add(s.checked_mul(2)?))
                .and_then(|count| count.checked_add(n))
                .expect("sandwich gate-count overflow");
            nonlinear_gss_resource_plan(sandwich_n, sandwich_gate_count, slice_gates, mode)
                .unwrap_or_else(|error| {
                    panic!(
                        "{} capacity check failed: {error}",
                        gadgetization_mode.canonical_name()
                    )
                });
            let mut overrides: Vec<String> = std::env::vars_os()
                .filter_map(|(key, _)| key.into_string().ok())
                .filter(|key| key.starts_with("PROD_"))
                .collect();
            overrides.sort();
            assert!(
                overrides.is_empty(),
                "{} does not accept product-share overrides; unset {}",
                gadgetization_mode.canonical_name(),
                overrides.join(", ")
            );
        }
    }
}

pub fn post_fragment(
    mut gadget: CnotCircuit,
    preset_name: Option<&str>,
    mut rng: &mut impl rand::Rng,
) -> CnotCircuit {
    let post_fragment = match std::env::var("PROD_POST_FRAGMENT") {
        Ok(value) => parse_post_fragment(&value).unwrap_or_else(|| {
            panic!("unknown PROD_POST_FRAGMENT={value:?}; expected off, exact, or native-deep")
        }),
        Err(_) => preset_post_fragment(preset_name),
    };
    if let Some(style) = post_fragment {
        let stats =
            fragment_wide_post_shuffle(&mut gadget.gates, gadget.num_wires, style, &mut rng)
                .unwrap_or_else(|error| panic!("post-layout fragmentation failed: {error}"));
        println!(
            "[gen] post-layout fragmentation style={style:?}: {} -> {} gates ({} wide macros), max controls {} -> {}, native emissions={}, exact-rung emissions={}",
            stats.input_gates,
            stats.output_gates,
            stats.fragmented_gates,
            stats.max_controls_before,
            stats.max_controls_after,
            stats.native_emissions,
            stats.exact_rung_emissions,
        );
    }
    gadget
}

pub fn gadgetize(
    sandwich: &CnotCircuit,
    n: usize,
    rg_freq: usize,
    slice_gates: usize,
    gadgetization_mode: GadgetizationMode,
    mut rng: &mut impl rand::Rng,
) -> (CnotCircuit, bool) {
    let sandwich_n = sandwich.num_wires;
    // Product-share encoding via env vars (PROD_K base deg-PROD_DEG terms +
    // PROD_K_HI tower deg-PROD_DEG_HI terms). The selected preset establishes
    // coherent representation defaults; individual PROD_* values tune it.
    let env = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    // Same rule as the sss path: a named preset establishes coherent defaults,
    // then individual environment variables may override fields.  The
    // no-gray-phase-a preset is the measured candidate for deployments that
    // reject Gray's aggregate space-time mask witness.
    let preset_name = if gadgetization_mode == GadgetizationMode::Product2223 {
        std::env::var("PROD_PRESET").ok()
    } else {
        None
    };
    let (preset, carrier_mode) = production_preset(preset_name.as_deref());
    let nonlinear_carrier = carrier_mode != CarrierMode::Single;
    assert!(
        !nonlinear_carrier || std::env::var_os("PROD_SINGLE").is_none(),
        "a nonlinear carrier PROD_PRESET conflicts with the single-carrier PROD_SINGLE override"
    );
    let prod = ProdConfig {
        k: env("PROD_K", preset.k),
        deg: env("PROD_DEG", preset.deg),
        k_hi: env("PROD_K_HI", preset.k_hi),
        deg_hi: env("PROD_DEG_HI", preset.deg_hi),
        band: env("PROD_BAND", preset.band),
        rsrc: env("PROD_RSRC", preset.rsrc),
        max_width: env("PROD_MAX_WIDTH", preset.max_width),
        fill_nl: env("PROD_FILL_NL", preset.fill_nl),
        roll: env("PROD_ROLL", preset.roll),
        src_dist: env("PROD_SRC_DIST", preset.src_dist),
        src_horizon: env("PROD_SRC_HORIZON", preset.src_horizon),
        src_lo: env("PROD_SRC_LO", preset.src_lo),
        src_hi: env("PROD_SRC_HI", preset.src_hi),
        fill_pivots: env("PROD_FILL_PIVOTS", preset.fill_pivots),
        g57_narrow: env("PROD_G57_NARROW", preset.g57_narrow),
        ladder_cap: env("PROD_LADDER_CAP", preset.ladder_cap),
        cg_jitter: env("PROD_CG_JITTER", preset.cg_jitter),
        rung_menu: env("PROD_RUNG_MENU", preset.rung_menu),
        epoch: env("PROD_EPOCH", preset.epoch),
        refill_data: env("PROD_REFILL_DATA", preset.refill_data),
        single: env("PROD_SINGLE", preset.single),
        gray_fold: env("PROD_GRAY_FOLD", preset.gray_fold),
        swap_refresh: env("PROD_SWAP", preset.swap_refresh),
        close_slice: env("PROD_CLOSE_SLICE", preset.close_slice),
    };
    assert!(
        prod.gray_fold <= 3,
        "PROD_GRAY_FOLD must be 0 (expanded), 1 (aggregate), 2 (micro), or 3 (sentinel)"
    );
    assert!(
        !nonlinear_carrier || prod.enabled(),
        "a nonlinear carrier PROD_PRESET requires a nonempty product-mask plan"
    );
    assert!(
        !nonlinear_carrier || !prod.dist(),
        "a nonlinear carrier PROD_PRESET does not support distributed product-mask sourcing"
    );
    if gadgetization_mode == GadgetizationMode::Product2223 && prod.enabled() {
        println!(
            "[gen] product-share encoding ON: representation={} k={} deg={} k_hi={} deg_hi={} cg_jitter={} band(auto)={} max_width={} ladder_cap={} gray_fold={} swap_refresh={} close_slice={} fill_nl={} roll={}",
            match carrier_mode {
                CarrierMode::Single => "single-carrier",
                CarrierMode::Five => "five-carrier",
                CarrierMode::StrongFive => "strong-five-carrier",
                CarrierMode::Six => "six-carrier",
                CarrierMode::StrongSix => "strong-six-carrier",
                CarrierMode::Seven => "seven-carrier",
                CarrierMode::SevenDistributed => "seven-carrier-shear",
                CarrierMode::SevenDistributedUnshuffled => {
                    "seven-carrier-shear-unshuffled"
                }
                CarrierMode::SevenDistributedPartitioned => "seven-carrier-partitioned",
                CarrierMode::SevenDistributedPartitionedUnshuffled => {
                    "seven-carrier-partitioned-unshuffled"
                }
                CarrierMode::SevenDistributedPartitionedFloor1024 => {
                    "seven-carrier-partitioned-floor1024"
                }
                CarrierMode::SevenDistributedPartitionedFloor1024Unshuffled => {
                    "seven-carrier-partitioned-floor1024-unshuffled"
                }
            },
            prod.k,
            prod.deg,
            prod.k_hi,
            prod.deg_hi,
            prod.cg_jitter,
            prod.band_size(sandwich_n),
            prod.max_width,
            prod.ladder_cap,
            prod.gray_fold,
            prod.swap_refresh,
            prod.close_slice,
            prod.fill_nl,
            prod.roll
        );
    }
    let gadget = if let Some(mode) = gadgetization_mode.nonlinear() {
        gadgetize_xgates_nonlinear_gss(&sandwich.gates, sandwich_n, n, slice_gates, mode, &mut rng)
            .unwrap_or_else(|error| {
                panic!(
                    "{} gadgetization failed: {error}",
                    gadgetization_mode.canonical_name()
                )
            })
    } else if carrier_mode == CarrierMode::SevenDistributedPartitionedFloor1024 {
        gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed_partitioned_floor1024_live_prefix(
            &sandwich.gates,
            sandwich_n,
            n,
            rg_freq,
            slice_gates,
            &prod,
            &mut rng,
        )
    } else if carrier_mode == CarrierMode::SevenDistributedPartitionedFloor1024Unshuffled {
        gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed_partitioned_floor1024_live_prefix_unshuffled(
            &sandwich.gates,
            sandwich_n,
            n,
            rg_freq,
            slice_gates,
            &prod,
            &mut rng,
        )
    } else if carrier_mode == CarrierMode::SevenDistributedPartitioned {
        gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed_partitioned_live_prefix(
            &sandwich.gates,
            sandwich_n,
            n,
            rg_freq,
            slice_gates,
            &prod,
            &mut rng,
        )
    } else if carrier_mode == CarrierMode::SevenDistributedPartitionedUnshuffled {
        gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed_partitioned_live_prefix_unshuffled(
            &sandwich.gates,
            sandwich_n,
            n,
            rg_freq,
            slice_gates,
            &prod,
            &mut rng,
        )
    } else if carrier_mode == CarrierMode::SevenDistributed {
        gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed(
            &sandwich.gates,
            sandwich_n,
            rg_freq,
            slice_gates,
            &prod,
            &mut rng,
        )
    } else if carrier_mode == CarrierMode::SevenDistributedUnshuffled {
        gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed_unshuffled(
            &sandwich.gates,
            sandwich_n,
            rg_freq,
            slice_gates,
            &prod,
            &mut rng,
        )
    } else if carrier_mode == CarrierMode::Seven {
        gadgetize_xgates_with_slice_zero_ccnot_seven_carrier(
            &sandwich.gates,
            sandwich_n,
            rg_freq,
            slice_gates,
            &prod,
            &mut rng,
        )
    } else if carrier_mode == CarrierMode::StrongSix {
        gadgetize_xgates_with_slice_zero_ccnot_strong_six_carrier(
            &sandwich.gates,
            sandwich_n,
            rg_freq,
            slice_gates,
            &prod,
            &mut rng,
        )
    } else if carrier_mode == CarrierMode::Six {
        gadgetize_xgates_with_slice_zero_ccnot_six_carrier(
            &sandwich.gates,
            sandwich_n,
            rg_freq,
            slice_gates,
            &prod,
            &mut rng,
        )
    } else if carrier_mode == CarrierMode::StrongFive {
        gadgetize_xgates_with_slice_zero_ccnot_strong_five_carrier(
            &sandwich.gates,
            sandwich_n,
            rg_freq,
            slice_gates,
            &prod,
            &mut rng,
        )
    } else if carrier_mode == CarrierMode::Five {
        gadgetize_xgates_with_slice_zero_ccnot_five_carrier(
            &sandwich.gates,
            sandwich_n,
            rg_freq,
            slice_gates,
            &prod,
            &mut rng,
        )
    } else if prod.single_carrier() {
        gadgetize_xgates_with_slice_zero_ccnot_single(
            &sandwich.gates,
            sandwich_n,
            rg_freq,
            slice_gates,
            &prod,
            &mut rng,
        )
    } else {
        gadgetize_xgates_with_slice_zero_ccnot(
            &sandwich.gates,
            sandwich_n,
            rg_freq,
            slice_gates,
            &MaskConfig::off(),
            &prod,
            &mut rng,
        )
    };
    let guarded =
        carrier_mode == CarrierMode::Single && prod.single_carrier() && prod.close_slice > 0;
    (
        post_fragment(gadget, preset_name.as_deref(), &mut rng),
        guarded,
    )
}

#[cfg(test)]
#[path = "../../../tests/stages/preprocessing/legacy_generator.rs"]
mod tests;
