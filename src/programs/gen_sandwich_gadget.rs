//! CLI/environment adapter for sandwich construction and supported preprocessing.
//!
//! The default is the classic sandwich A(x,0)=(junk,C(x)), with balanced
//! blinded-V5 masks (K=2, max_open=3, min_open=2, quadratic fire, band helpers,
//! extra_lgis=0). The five modules are the opening guard, input-derived band
//! seed, V5 compute, independently seeded band reseed, and closing guard.
//! V5 runs on all 2n sandwich wires; its auto band is also 2n wires, so the
//! full gadget has 4n wires and allocates no clean scratch. Seed/rerand input
//! activity is restricted to the source's low n wires; encoded_io is false.
//!
//! Usage: gen_sandwich_gadget <out> [n=128] [m_C=3000] [m_D=3000]
//!                            [s=n*log2 n] [rg_freq=1] [slice_gates=10*2n]
//!                            [seed=1] [gadget_seed=seed] [sandwich_seed=seed]
//!                            [preprocessing_mode=quadratic-masking]
//!                            [sandwich_variant=classic]
//!
//! `ran-balanced`, `blinded-v5` and `blinded_v5` alias `quadratic-masking`. The separate
//! mirrored sandwich remains selectable via the final positional argument or
//! SANDWICH_VARIANT; it does not provide the classic reverse-port guarantee.
//! BV5_* comparison overrides retain their existing interpretation. Legacy
//! product/nonlinear193 modes and PROD_* controls require `legacy-tools`.
//!
//! `seed` fixes C (fastrand), `sandwich_seed` drives D, slicing and N-float,
//! and `gadget_seed` drives gadgetization. GSS_SOURCE_C optionally supplies C
//! as a g57 circuit with matching dimensions. Source and sandwich artifacts
//! are written beside the output as .source_c.g57 and .sandwich.mpmct1.

use crate::circuit::formats::write_mpmct;
use crate::stages::preprocessing::nonlinear291::{NonlinearGssMode, nonlinear_gss_resource_plan};
use crate::stages::preprocessing::quadratic_masking::{
    QuadraticMaskingExecution, QuadraticMaskingParams,
};
use crate::stages::preprocessing::verify::verify_payload;
use crate::stages::preprocessing::{PreprocessingMode, PreprocessingParams, preprocess_sandwich};
use crate::stages::sandwich::{
    SandwichVariant, construct_seeded_sandwich, prepare_source, sandwich_default_s,
};
use rand::SeedableRng;
use rand::rngs::StdRng;

// Comparison implementations are compiled only when explicitly requested.
#[cfg(feature = "legacy-tools")]
#[path = "../../security_tests/support/preprocessing/generator.rs"]
mod legacy;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum GadgetizationMode {
    #[default]
    QuadraticMasking,
    Nonlinear291,
    #[cfg(feature = "legacy-tools")]
    Legacy(legacy::GadgetizationMode),
}

impl GadgetizationMode {
    fn parse(value: &str) -> Option<Self> {
        if let Some(mode) = PreprocessingMode::parse(value) {
            return Some(match mode {
                PreprocessingMode::QuadraticMasking => Self::QuadraticMasking,
                PreprocessingMode::Nonlinear291 => Self::Nonlinear291,
            });
        }
        #[cfg(feature = "legacy-tools")]
        {
            legacy::GadgetizationMode::parse(value).map(Self::Legacy)
        }
        #[cfg(not(feature = "legacy-tools"))]
        {
            None
        }
    }

    fn canonical_name(self) -> &'static str {
        match self {
            Self::QuadraticMasking => PreprocessingMode::QuadraticMasking.canonical_name(),
            Self::Nonlinear291 => PreprocessingMode::Nonlinear291.canonical_name(),
            #[cfg(feature = "legacy-tools")]
            Self::Legacy(mode) => mode.canonical_name(),
        }
    }
}

pub fn run() {
    let mut a = std::env::args().skip(1);
    let out = a
        .next()
        .expect("usage: gen_sandwich_gadget <out> [n m_C m_D s rg_freq slice_gates seed gadget_seed sandwich_seed gadgetization_mode]");
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
    let gadgetization_mode = a
        .next()
        .as_deref()
        .map(GadgetizationMode::parse)
        .unwrap_or(Some(GadgetizationMode::default()))
        .unwrap_or_else(|| {
            panic!(
                "unknown preprocessing mode; expected quadratic-masking (alias ran-balanced/blinded-v5) or nonlinear291; historical comparisons require legacy-tools"
            )
        });
    // Chosen per run, never per build: the final positional argument, or
    // SANDWICH_VARIANT for callers that would otherwise have to spell out all
    // ten preceding positionals to reach it. Absent both, classic.
    let variant = a
        .next()
        .or_else(|| std::env::var("SANDWICH_VARIANT").ok())
        .as_deref()
        .map(|value| {
            SandwichVariant::parse(value).unwrap_or_else(|| {
                panic!("unknown sandwich variant {value:?}; expected classic or balanced")
            })
        })
        .unwrap_or(SandwichVariant::Classic);
    assert!(
        a.next().is_none(),
        "too many arguments; the sandwich variant is the final optional argument"
    );
    #[cfg(feature = "legacy-tools")]
    if let GadgetizationMode::Legacy(mode) = gadgetization_mode {
        mode.preflight(n, m_c, m_d, s, slice_gates);
    }
    if gadgetization_mode == GadgetizationMode::Nonlinear291 {
        let sandwich_gate_count = m_c
            .checked_add(m_d)
            .and_then(|count| count.checked_add(s.checked_mul(2)?))
            .and_then(|count| count.checked_add(n))
            .expect("sandwich gate-count overflow");
        nonlinear_gss_resource_plan(
            sandwich_n,
            sandwich_gate_count,
            slice_gates,
            NonlinearGssMode::Nonlinear291,
        )
        .unwrap_or_else(|error| panic!("nonlinear291 capacity check failed: {error}"));
    }
    if !cfg!(feature = "legacy-tools") || gadgetization_mode == GadgetizationMode::Nonlinear291 {
        let mut overrides: Vec<String> = std::env::vars_os()
            .filter_map(|(key, _)| key.into_string().ok())
            .filter(|key| key.starts_with("PROD_"))
            .collect();
        overrides.sort();
        assert!(
            overrides.is_empty(),
            "PROD_* controls require a historical product comparison; unset {}",
            overrides.join(", ")
        );
    }

    println!(
        "[gen] n={n} |C|={m_c} |D|={m_d} s={s} rg_freq={rg_freq} slice_gates={slice_gates} seed={seed} gadget_seed={gadget_seed} sandwich_seed={sandwich_seed} gadgetization_mode={} sandwich_variant={}",
        gadgetization_mode.canonical_name(),
        variant.name()
    );

    let source_path = std::env::var("GSS_SOURCE_C").ok();
    let supplied = source_path.as_ref().map(|path| {
        let raw = std::fs::read(path)
            .unwrap_or_else(|error| panic!("GSS_SOURCE_C: cannot read {path}: {error}"));
        crate::circuit::CircuitSeq::from_bytes(&raw)
    });
    let c = prepare_source(n, m_c, seed, supplied).unwrap_or_else(|error| {
        panic!(
            "GSS_SOURCE_C: {} {error}",
            source_path.as_deref().unwrap_or("source")
        )
    });
    if let Some(path) = source_path {
        println!(
            "[gen] source C LOADED from {path} ({} gates, {} wires)",
            c.gates.len(),
            c.max_wire() + 1
        );
    }
    // Also dump C in g57 format so hmap_affine can reconstruct against the
    // ORIGINAL computation. Regenerating with the same seed reproduces C (and
    // the whole gadget) bit-for-bit, so this recovers a past run's source.
    let c_path = format!("{out}.source_c.g57");
    std::fs::write(&c_path, c.repr()).expect("write source C");
    println!(
        "[gen] wrote source C ({} g57 gates) to {c_path}",
        c.gates.len()
    );
    let sandwich = construct_seeded_sandwich(&c, n, m_d, s, variant, sandwich_seed);
    println!(
        "[gen] {} sliced sandwich: {} gates, {} wires (payload on wires {} on the zero slice)",
        variant.name(),
        sandwich.gates.len(),
        sandwich.num_wires,
        if variant.is_balanced() {
            format!("0..{n}")
        } else {
            format!("{n}..{}", 2 * n)
        }
    );
    let a_path = format!("{out}.sandwich.mpmct1");
    write_mpmct(&a_path, &sandwich.gates, sandwich.num_wires).expect("write sandwich");
    println!("[gen] wrote sandwich A to {a_path}");

    // Gadgetization runs on its own stream: same `seed` + different
    // `gadget_seed` = fresh gadgetization of the identical sandwich.
    let mut rng = StdRng::seed_from_u64(gadget_seed ^ 0x6AD6_E75E);

    let (gadget, guarded) = match gadgetization_mode {
        GadgetizationMode::QuadraticMasking => {
            let envu = |k: &str, d: usize| {
                std::env::var(k)
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(d)
            };
            // Managed and raw defaults share data-and-band burst refresh;
            // historical direct callers can still request band-only bursts.
            let base = QuadraticMaskingParams::managed_gss(gadget_seed, n);
            let params = QuadraticMaskingParams {
                active_wires: n,
                k: envu("BV5_K", base.k),
                rerand_level: envu("BV5_RERAND", base.rerand_level),
                rerand_repair: envu("BV5_REPAIR", base.rerand_repair),
                rerand_burst: envu("BV5_BURST", base.rerand_burst),
                max_open: envu("BV5_MAX_OPEN", base.max_open),
                min_mask: envu("BV5_MIN_MASK", base.min_mask),
                extra_lgis: envu("BV5_EXTRA_LGIS", base.extra_lgis),
                quad_fire: std::env::var("BV5_QUAD_FIRE").map_or(base.quad_fire, |v| v != "0"),
                balanced: std::env::var("BV5_BALANCED").map_or(base.balanced, |v| v != "0"),
                burst_band_only: std::env::var("BV5_BURST_BANDONLY").is_ok_and(|v| v != "0"),
                encoded_io: false,
                min_open: std::env::var("BV5_MIN_OPEN")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(2),
                ..base
            };
            let options = PreprocessingParams::QuadraticMasking {
                balanced_seed: std::env::var("BV5_BAL_SEED").map_or(params.balanced, |v| v != "0"),
                execution: QuadraticMaskingExecution {
                    ancilla_band_only: std::env::var("BV5_ANC_POOL").map_or(true, |v| v != "all"),
                    diagnostics: std::env::var("BV5_DIAG").is_ok(),
                },
                record_hot_intervals: std::env::var("BV5_HOT_MANIFEST").is_ok(),
                params,
            };
            let output =
                preprocess_sandwich(&sandwich, n, slice_gates, variant, &options, &mut rng)
                    .unwrap_or_else(|error| {
                        panic!("quadratic-masking preprocessing failed: {error}")
                    });
            if let Some(report) = output.quadratic_masking_report {
                println!(
                    "[gen] quadratic-masking: {} atoms, band={} wires, {} band-seed + {} band-reseed + {} slice-guard gates each side",
                    report.atoms,
                    report.band_wires,
                    report.band_seed_gates,
                    report.band_reseed_gates,
                    report.guard_gates
                );
                if let Ok(path) = std::env::var("BV5_HOT_MANIFEST") {
                    let mut body = String::from("# wire\tstart_gate\tend_gate\tkind\n");
                    for (wire, start, end) in report.hot_intervals {
                        let kind = if start == 0 {
                            "input-fringe"
                        } else if end == report.compute_gates {
                            "output-fringe"
                        } else {
                            "INTERIOR (defect)"
                        };
                        body.push_str(&format!("{wire}\t{start}\t{end}\t{kind}\n"));
                    }
                    let _ = std::fs::write(path, body);
                }
            }
            let gadget = output.circuit;
            #[cfg(feature = "legacy-tools")]
            let gadget = legacy::post_fragment(gadget, None, &mut rng);
            (gadget, output.guarded)
        }
        GadgetizationMode::Nonlinear291 => {
            let output = preprocess_sandwich(
                &sandwich,
                n,
                slice_gates,
                variant,
                &PreprocessingParams::Nonlinear291,
                &mut rng,
            )
            .unwrap_or_else(|error| panic!("nonlinear291 preprocessing failed: {error}"));
            (output.circuit, output.guarded)
        }
        #[cfg(feature = "legacy-tools")]
        GadgetizationMode::Legacy(mode) => {
            legacy::gadgetize(&sandwich, n, rg_freq, slice_gates, mode, &mut rng)
        }
    };
    println!(
        "[gen] diversified gadget: {} gates, {} wires",
        gadget.gates.len(),
        gadget.num_wires
    );

    let (verify_from, verify_to, reverse_verified) =
        verify_payload(&sandwich, &gadget, n, guarded, variant, seed);
    println!(
        "[gen] verify PASSED (256 bit-sliced samples, payload wires {verify_from}..{verify_to})"
    );
    if reverse_verified {
        println!(
            "[gen] reverse verify PASSED (256 bit-sliced samples, wires {n}..{sandwich_n}: D^-1 emerges under backward evaluation)"
        );
    } else if guarded && variant.is_balanced() {
        println!(
            "[gen] reverse verify SKIPPED for balanced (forward payload verified on the low half; per-port reverse mirror is TODO)"
        );
    }

    write_mpmct(&out, &gadget.gates, gadget.num_wires).expect("write mpmct1");
    println!("[gen] wrote {out}");
}

#[cfg(test)]
#[path = "../../tests/stages/preprocessing/gen_sandwich_gadget.rs"]
mod tests;
