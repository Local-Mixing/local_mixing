//! Sample a sliced sandwich and gadgetize it with the current diversified
//! scheme, writing the gadgetized circuit (mpmct1) ready for fmix.
//!
//! Pipeline: fresh random g57 C on n wires -> sliced_sandwich_cnot (samples a
//! random D, interleaves the two slice blocks, floats the N column) -> a 2n-wire
//! sandwich A with A(x,0)=(junk, C(x)) on the zero slice ->
//! the selected zero-slice gadgetizer (single-carrier by default, or the
//! supplied/strong five/six or seven-carrier nonlinear representation with the matching
//! `PROD_PRESET`) -> a
//! gadget whose low 2n output equals A on the gadget's zero slice.
//!
//! Usage: gen_sandwich_gadget <out> [n=128] [m_C=3000] [m_D=3000]
//!                            [s=n*log2 n] [rg_freq=1] [slice_gates=10*2n]
//!                            [seed=1] [gadget_seed=seed] [sandwich_seed=seed]
//!
//! `seed` fixes C only (fastrand); `sandwich_seed` drives D + slicing +
//! N-float (default = seed); `gadget_seed` drives the gadgetization only.
//! `PROD_PRESET` selects `production` (default), a fold/fragmentation study
//! arm, or a supplied/strong carrier preset; individual
//! `PROD_*` variables override common mask settings. `PROD_POST_FRAGMENT`
//! optionally applies `exact` or `native-deep` post-layout fragmentation.
//! So: vary gadget_seed alone = re-gadgetize the SAME sandwich A; vary
//! sandwich_seed (+ gadget_seed) with seed fixed = a FRESH sandwich around
//! the SAME C. The sandwich is dumped to `<out>.sandwich.mpmct1` so same-A /
//! fresh-A across runs is checkable byte-for-byte.

use local_mixing::circuit::circuit::U1024;
use local_mixing::postmix::format::write_mpmct;
use local_mixing::postmix::fragment::{FragmentStyle, fragment_wide_post_shuffle};
use local_mixing::random::random_data::random_circuit;
use local_mixing::replace::gadgets::{
    MaskConfig, ProdConfig, gadgetize_xgates_with_slice_zero_ccnot,
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
    gadgetize_xgates_with_slice_zero_ccnot_strong_six_carrier, sandwich_default_s,
    sliced_sandwich_cnot,
};
use rand::SeedableRng;
use rand::rngs::StdRng;

fn mask_bits(bits: usize) -> U1024 {
    if bits >= 1024 {
        U1024::MAX
    } else {
        (U1024::one() << bits) - U1024::one()
    }
}

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

fn main() {
    let mut a = std::env::args().skip(1);
    let out = a
        .next()
        .expect("usage: gen_sandwich_gadget <out> [n m_C m_D s rg_freq slice_gates seed]");
    let n: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(128);
    let m_c: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(3000);
    let m_d: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(3000);
    let s: usize = a
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| sandwich_default_s(n));
    let rg_freq: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let sandwich_n = 2 * n;
    let slice_gates: usize = a
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10 * sandwich_n);
    let seed: u64 = a.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let gadget_seed: u64 = a.next().and_then(|s| s.parse().ok()).unwrap_or(seed);
    let sandwich_seed: u64 = a.next().and_then(|s| s.parse().ok()).unwrap_or(seed);

    println!(
        "[gen] n={n} |C|={m_c} |D|={m_d} s={s} rg_freq={rg_freq} slice_gates={slice_gates} seed={seed} gadget_seed={gadget_seed} sandwich_seed={sandwich_seed}"
    );

    // Fresh random g57 source C (fastrand-seeded, matching sandwich_compare).
    fastrand::seed(seed);
    let c = random_circuit(n, m_c);
    // Also dump C in g57 format so hmap_affine can reconstruct against the
    // ORIGINAL computation. Regenerating with the same seed reproduces C (and
    // the whole gadget) bit-for-bit, so this recovers a past run's source.
    let c_path = format!("{out}.source_c.g57");
    std::fs::write(&c_path, c.repr()).expect("write source C");
    println!(
        "[gen] wrote source C ({} g57 gates) to {c_path}",
        c.gates.len()
    );
    let mut rng = StdRng::seed_from_u64(sandwich_seed ^ 0x5150_1CED);

    let sandwich = sliced_sandwich_cnot(&c, n, m_d, s, &mut rng);
    println!(
        "[gen] sliced sandwich: {} gates, {} wires",
        sandwich.gates.len(),
        sandwich.num_wires
    );
    let a_path = format!("{out}.sandwich.mpmct1");
    write_mpmct(&a_path, &sandwich.gates, sandwich.num_wires).expect("write sandwich");
    println!("[gen] wrote sandwich A to {a_path}");

    // Gadgetization runs on its own stream: same `seed` + different
    // `gadget_seed` = fresh gadgetization of the identical sandwich.
    let mut rng = StdRng::seed_from_u64(gadget_seed ^ 0x6AD6_E75E);

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
    let preset_name = std::env::var("PROD_PRESET").ok();
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
    if prod.enabled() {
        println!(
            "[gen] product-share encoding ON: representation={} k={} deg={} k_hi={} deg_hi={} band(auto)={} max_width={} ladder_cap={} gray_fold={} swap_refresh={} close_slice={} fill_nl={} roll={}",
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
    let mut gadget = if carrier_mode == CarrierMode::SevenDistributedPartitionedFloor1024 {
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
    println!(
        "[gen] diversified gadget: {} gates, {} wires",
        gadget.gates.len(),
        gadget.num_wires
    );

    let post_fragment = match std::env::var("PROD_POST_FRAGMENT") {
        Ok(value) => parse_post_fragment(&value).unwrap_or_else(|| {
            panic!("unknown PROD_POST_FRAGMENT={value:?}; expected off, exact, or native-deep")
        }),
        Err(_) => preset_post_fragment(preset_name.as_deref()),
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

    // Sample-verify the gadget's low-2n output equals the sandwich, on the
    // gadget zero slice (upper 2n wires pinned to 0). Bit-sliced: each u64
    // lane-word carries 64 independent samples, so 4 passes = 256 samples at
    // per-gate u64 cost instead of 256 full U1024 bignum evaluations. The
    // verify rng is dedicated (seed ^ 0xA11CE) and the artifact bytes never
    // depend on it.
    {
        // state[w] = lane word for wire w; bit L = wire w's value in sample L.
        fn eval_lanes(gates: &[local_mixing::postmix::xgate::XGate], state: &mut [u64]) {
            for g in gates {
                let mut acc = !0u64;
                for &(w, pol) in &g.ctrls {
                    let v = state[w as usize];
                    acc &= if pol { v } else { !v };
                }
                state[g.target as usize] ^= if g.comp { !acc } else { acc };
            }
        }
        let mut vrng = StdRng::seed_from_u64(seed ^ 0xA11CE);
        let total_wires = gadget.num_wires.max(sandwich.num_wires);
        // With the closing zero-slice block, the composite preserves only the
        // UPPER half of the sandwich state on the honest slice: the closing
        // guard fires against the mirror fill's band junk and perturbs the
        // low (forward-junk) half by design. The payload contract is
        // unchanged — C(x) lives on the upper half (see verify_zero_slice).
        let verify_lo = if carrier_mode == CarrierMode::Single
            && prod.single_carrier()
            && prod.close_slice > 0
        {
            sandwich_n / 2
        } else {
            0
        };
        for round in 0..4 {
            use rand::RngCore;
            let mut ga = vec![0u64; total_wires];
            for w in 0..sandwich_n {
                ga[w] = vrng.next_u64(); // upper wires stay pinned to 0
            }
            let mut sa = ga.clone();
            eval_lanes(&gadget.gates, &mut ga);
            eval_lanes(&sandwich.gates, &mut sa);
            for w in verify_lo..sandwich_n {
                assert_eq!(
                    ga[w], sa[w],
                    "gadget-low != sandwich on wire {w}, round {round}"
                );
            }
        }
        println!(
            "[gen] verify PASSED (256 bit-sliced samples, wires {verify_lo}..{sandwich_n})"
        );
    }

    write_mpmct(&out, &gadget.gates, gadget.num_wires).expect("write mpmct1");
    println!("[gen] wrote {out}");
}

#[cfg(test)]
mod tests {
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
}
