//! CLI/environment adapter for sandwich construction and supported preprocessing.
//!
//! The default is the classic sandwich A(x,0)=(junk,C(x)), with balanced
//! embedded masks (K=2, max_open=3, min_open=2, quadratic fire, band helpers,
//! extra_lgis=0). The five modules are the opening guard, input-derived band
//! seed, embedded computation, independently seeded band reseed, and closing guard.
//! Embedded masking runs on all 2n sandwich wires; its auto band is also 2n wires, so the
//! full gadget has 4n wires and allocates no clean scratch. Seed/rerand input
//! activity is restricted to the source's low n wires; encoded_io is false.
//!
//! Usage: gen_sandwich_gadget <out> [n=128] [m_C=3000] [m_D=3000]
//!                            [s=n*log2 n] [rg_freq=1] [slice_gates=10*2n]
//!                            [seed=1] [gadget_seed=seed] [sandwich_seed=seed]
//!                            [preprocessing_mode=embedded-masking]
//!                            [sandwich_variant=classic]
//!
//! The mirrored sandwich remains selectable via the final positional argument or
//! SANDWICH_VARIANT; it does not provide the classic reverse-port guarantee.
//! EMBEDDED_MASKING_* variables configure direct construction experiments.
//! EMBEDDED_MASKING_SHUFFLING=8 enables preprocessing shuffling; its
//! EMBEDDED_MASKING_SHUFFLING_RETURN_HOME flag defaults to true and is required
//! when enabled here to preserve the sandwich's physical output ports.
//!
//! `seed` fixes C (fastrand), `sandwich_seed` drives D, slicing and N-float,
//! and `gadget_seed` drives gadgetization. TDP_SOURCE_C optionally supplies C
//! as a g57 circuit with matching dimensions. Source and sandwich artifacts
//! are written beside the output as .source_c.g57 and .sandwich.mpmct1.

use crate::circuit::formats::write_mpmct;
use crate::stages::preprocessing::embedded_masking::{
    EmbeddedMaskingExecution, EmbeddedMaskingParams,
};
use crate::stages::preprocessing::nonlinear291::nonlinear_tdp_resource_plan;
use crate::stages::preprocessing::verify::verify_payload;
use crate::stages::preprocessing::{PreprocessingMode, PreprocessingParams, preprocess_sandwich};
use crate::stages::sandwich::{
    SandwichVariant, construct_seeded_sandwich, prepare_source, sandwich_default_s,
};
use rand::SeedableRng;
use rand::rngs::StdRng;

fn parse_shuffling_controls(
    mode: PreprocessingMode,
    segments: Option<&str>,
    return_home: Option<&str>,
) -> Result<(usize, bool), String> {
    if mode != PreprocessingMode::EmbeddedMasking && (segments.is_some() || return_home.is_some()) {
        return Err("shuffling controls require embedded-masking".into());
    }
    let segments = segments.unwrap_or("0").parse::<usize>().map_err(|_| {
        "EMBEDDED_MASKING_SHUFFLING must be 0 (off) or an integer at least 8".to_string()
    })?;
    if segments != 0 && segments < 8 {
        return Err("EMBEDDED_MASKING_SHUFFLING must be 0 (off) or at least 8".into());
    }
    let return_home = match return_home.unwrap_or("true") {
        "0" | "false" => false,
        "1" | "true" => true,
        _ => {
            return Err(
                "EMBEDDED_MASKING_SHUFFLING_RETURN_HOME must be 0, 1, false, or true".into(),
            );
        }
    };
    if segments > 0 && !return_home {
        return Err("EMBEDDED_MASKING_SHUFFLING_RETURN_HOME must be true to preserve the sandwich's physical output ports".into());
    }
    Ok((segments, return_home))
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
        .map(PreprocessingMode::parse)
        .unwrap_or(Some(PreprocessingMode::default()))
        .unwrap_or_else(|| {
            panic!("unknown preprocessing mode; expected embedded-masking or nonlinear291")
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
    if gadgetization_mode == PreprocessingMode::Nonlinear291 {
        let sandwich_gate_count = m_c
            .checked_add(m_d)
            .and_then(|count| count.checked_add(s.checked_mul(2)?))
            .and_then(|count| count.checked_add(n))
            .expect("sandwich gate-count overflow");
        nonlinear_tdp_resource_plan(sandwich_n, sandwich_gate_count, slice_gates)
            .unwrap_or_else(|error| panic!("nonlinear291 capacity check failed: {error}"));
    }
    {
        let mut obsolete_mask_controls: Vec<String> = std::env::vars_os()
            .filter_map(|(key, _)| key.into_string().ok())
            .filter(|key| key.starts_with("BV5_"))
            .collect();
        obsolete_mask_controls.sort();
        assert!(
            obsolete_mask_controls.is_empty(),
            "obsolete masking controls: {}; use EMBEDDED_MASKING_* instead",
            obsolete_mask_controls.join(", ")
        );
        let mut overrides: Vec<String> = std::env::vars_os()
            .filter_map(|(key, _)| key.into_string().ok())
            .filter(|key| key.starts_with("PROD_"))
            .collect();
        overrides.sort();
        assert!(
            overrides.is_empty(),
            "PROD_* controls have been retired; unset {}",
            overrides.join(", ")
        );
    }

    // Validate before writing the source or sandwich artifacts.
    let (shuffling_segments, shuffling_return_home) = parse_shuffling_controls(
        gadgetization_mode,
        std::env::var("EMBEDDED_MASKING_SHUFFLING").ok().as_deref(),
        std::env::var("EMBEDDED_MASKING_SHUFFLING_RETURN_HOME")
            .ok()
            .as_deref(),
    )
    .unwrap_or_else(|error| panic!("invalid preprocessing settings: {error}"));

    println!(
        "[gen] n={n} |C|={m_c} |D|={m_d} s={s} rg_freq={rg_freq} slice_gates={slice_gates} seed={seed} gadget_seed={gadget_seed} sandwich_seed={sandwich_seed} gadgetization_mode={} sandwich_variant={}",
        gadgetization_mode.canonical_name(),
        variant.name()
    );

    let source_path = std::env::var("TDP_SOURCE_C").ok();
    let supplied = source_path.as_ref().map(|path| {
        let raw = std::fs::read(path)
            .unwrap_or_else(|error| panic!("TDP_SOURCE_C: cannot read {path}: {error}"));
        crate::circuit::CircuitSeq::from_bytes(&raw)
    });
    let c = prepare_source(n, m_c, seed, supplied).unwrap_or_else(|error| {
        panic!(
            "TDP_SOURCE_C: {} {error}",
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
        PreprocessingMode::EmbeddedMasking => {
            let envu = |k: &str, d: usize| {
                std::env::var(k)
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(d)
            };
            // Managed and raw defaults share data-and-band burst refresh;
            // direct callers can still request band-only bursts.
            let base = EmbeddedMaskingParams::managed_tdp(gadget_seed, n);
            let params = EmbeddedMaskingParams {
                active_wires: n,
                shuffling_segments,
                shuffling_return_home,
                k: envu("EMBEDDED_MASKING_K", base.k),
                rerand_level: envu("EMBEDDED_MASKING_RERAND", base.rerand_level),
                rerand_repair: envu("EMBEDDED_MASKING_REPAIR", base.rerand_repair),
                rerand_burst: envu("EMBEDDED_MASKING_BURST", base.rerand_burst),
                max_open: envu("EMBEDDED_MASKING_MAX_OPEN", base.max_open),
                min_mask: envu("EMBEDDED_MASKING_MIN_MASK", base.min_mask),
                extra_lgis: envu("EMBEDDED_MASKING_EXTRA_LGIS", base.extra_lgis),
                quad_fire: std::env::var("EMBEDDED_MASKING_QUAD_FIRE")
                    .map_or(base.quad_fire, |v| v != "0"),
                balanced: std::env::var("EMBEDDED_MASKING_BALANCED")
                    .map_or(base.balanced, |v| v != "0"),
                burst_band_only: std::env::var("EMBEDDED_MASKING_BURST_BANDONLY")
                    .is_ok_and(|v| v != "0"),
                encoded_io: false,
                min_open: std::env::var("EMBEDDED_MASKING_MIN_OPEN")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(2),
                ..base
            };
            let options = PreprocessingParams::EmbeddedMasking {
                balanced_seed: std::env::var("EMBEDDED_MASKING_BAL_SEED")
                    .map_or(params.balanced, |v| v != "0"),
                execution: EmbeddedMaskingExecution {
                    ancilla_band_only: std::env::var("EMBEDDED_MASKING_ANC_POOL")
                        .map_or(true, |v| v != "all"),
                    diagnostics: std::env::var("EMBEDDED_MASKING_DIAG").is_ok(),
                    record_hot_intervals: std::env::var("EMBEDDED_MASKING_HOT_MANIFEST").is_ok(),
                },
                params,
            };
            let output =
                preprocess_sandwich(&sandwich, n, slice_gates, variant, &options, &mut rng)
                    .unwrap_or_else(|error| {
                        panic!("embedded-masking preprocessing failed: {error}")
                    });
            if let Some(report) = output.embedded_masking_report {
                println!(
                    "[gen] embedded-masking: {} atoms, band={} wires, {} band-seed + {} band-reseed + {} slice-guard gates each side",
                    report.atoms,
                    report.band_wires,
                    report.band_seed_gates,
                    report.band_reseed_gates,
                    report.guard_gates
                );
                if let Ok(path) = std::env::var("EMBEDDED_MASKING_HOT_MANIFEST") {
                    let mut body = String::from(
                        "# Coordinates: logical roles and original compute gates, before shuffling.\n# wire\tstart_gate\tend_gate\tkind\n",
                    );
                    for (wire, start, end) in report.hot_intervals {
                        let kind = if start == 0 {
                            "input-fringe"
                        } else if end == report.original_compute_gates {
                            "output-fringe"
                        } else {
                            "INTERIOR (defect)"
                        };
                        body.push_str(&format!("{wire}\t{start}\t{end}\t{kind}\n"));
                    }
                    let _ = std::fs::write(path, body);
                }
            }
            (output.circuit, output.guarded)
        }
        PreprocessingMode::Nonlinear291 => {
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
#[path = "../../../tests/unit/programs/gen_sandwich_gadget.rs"]
mod tests;
