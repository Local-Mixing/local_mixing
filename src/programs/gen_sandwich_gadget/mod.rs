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
//! and `gadget_seed` drives gadgetization. Any seed may be given as `@PATH`
//! (one decimal seed read from PATH, e.g. `@/dev/fd/3`) so its value stays off
//! argv; the banner then prints `(file)` in its place (and for a seed that
//! defaults to it). TDP_SOURCE_C optionally supplies C
//! as a g57 circuit with matching dimensions. TDP_SOURCE_CONSTRUCTION picks
//! how a generated C is drawn: unset or `random` (uniform g57 gates), or
//! `balanced` (RC's balanced C, every completed internal value read exactly
//! twice; exclusive with TDP_SOURCE_C). Source and sandwich artifacts
//! are written beside the output as .source_c.g57 and .sandwich.mpmct1.

use crate::circuit::formats::write_mpmct;
use crate::stages::preprocessing::embedded_masking::{
    EmbeddedMaskingExecution, EmbeddedMaskingParams,
};
use crate::stages::preprocessing::nonlinear291::nonlinear_tdp_resource_plan;
use crate::stages::preprocessing::verify::verify_payload;
use crate::stages::preprocessing::{PreprocessingMode, PreprocessingParams, preprocess_sandwich};
use crate::stages::sandwich::{
    SandwichVariant, SourceConstruction, audit_balanced, construct_seeded_sandwich, prepare_source,
    sandwich_default_s,
};
use rand::SeedableRng;
use rand::rngs::StdRng;

/// `TDP_SOURCE_CONSTRUCTION`: unset or `random` selects the uniformly random
/// C, `balanced` RC's balanced C; any other value (including empty) is an
/// error naming the variable.
pub(crate) fn parse_source_construction(value: Option<&str>) -> Result<SourceConstruction, String> {
    match value {
        None => Ok(SourceConstruction::Random),
        Some(v) => SourceConstruction::parse(v)
            .ok_or_else(|| format!("TDP_SOURCE_CONSTRUCTION={v:?}: expected random or balanced")),
    }
}

fn parse_shuffling_controls(
    mode: PreprocessingMode,
    segments: Option<&str>,
    return_home: Option<&str>,
    jit_gap: Option<&str>,
) -> Result<(usize, bool, usize), String> {
    if mode != PreprocessingMode::EmbeddedMasking
        && (segments.is_some() || return_home.is_some() || jit_gap.is_some())
    {
        return Err("shuffling controls require embedded-masking".into());
    }
    let segments = segments.unwrap_or("0").parse::<usize>().map_err(|_| {
        "EMBEDDED_MASKING_SHUFFLING must be 0 (off) or a positive integer".to_string()
    })?;
    // RC 2026-10-06: segments=1 is the punchlist-#4 intent -- one carrier
    // transfer per ORIGINAL-circuit write (3 gates per fire box); larger
    // doses additionally protect intra-box partial sums and supply
    // read-pressure refreshes.

    let return_home = match return_home.unwrap_or("true") {
        "0" | "false" => false,
        "1" | "true" => true,
        _ => {
            return Err(
                "EMBEDDED_MASKING_SHUFFLING_RETURN_HOME must be 0, 1, false, or true".into(),
            );
        }
    };
    // return_home=false selects PERSISTENT carrier migration: data roles hop
    // across wires for the whole stream and are swapped back to their
    // physical ports once, at the stage-2 tail, so the sandwich's output
    // ports are preserved in both modes.
    let jit_gap = match jit_gap.unwrap_or("0") {
        "0" | "off" => 0,
        value => value.parse::<usize>().map_err(|_| {
            "EMBEDDED_MASKING_JIT must be 0/off or a positive re-mask gap in source gates"
                .to_string()
        })?,
    };
    // Punchlist #8: just-in-time drains keep migration masks standing past
    // the fire-box end and bridge idle data-role stretches, so bare role
    // values stop being reborn at every box end (the celebrity class).
    if jit_gap > 0 && segments == 0 {
        return Err("EMBEDDED_MASKING_JIT requires EMBEDDED_MASKING_SHUFFLING > 0".into());
    }
    if jit_gap > 0 && return_home {
        return Err(
            "EMBEDDED_MASKING_JIT requires EMBEDDED_MASKING_SHUFFLING_RETURN_HOME=0".into(),
        );
    }
    Ok((segments, return_home, jit_gap))
}

/// The stage-2 construction knobs that the library takes as typed fields
/// (docs/STAGE4_RUST_DESIGN.md §11.4), from their historical environment
/// names. This adapter (and the security tools that reuse it) is the only
/// place those names are read; the library never reads the environment.
/// Unset variables keep `p`'s values.
pub fn stage2_knobs_from_env(p: EmbeddedMaskingParams) -> EmbeddedMaskingParams {
    use crate::stages::preprocessing::embedded_masking::RetireParams;
    let var = |k: &str| std::env::var(k);
    let num = |k: &str, d: usize| -> usize {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let flag = |k: &str, d: bool| -> bool { var(k).map_or(d, |v| v != "0") };
    EmbeddedMaskingParams {
        premask: flag("EMBEDDED_MASKING_PREMASK", p.premask),
        seed_avoid_c1: flag("EMBEDDED_MASKING_SEED_AVOID_C1", p.seed_avoid_c1),
        chain_k: num("EMBEDDED_MASKING_CHAIN", p.chain_k),
        chain_len: num("EMBEDDED_MASKING_CHAIN_LEN", p.chain_len),
        chain_rotate: flag("EMBEDDED_MASKING_CHAIN_ROTATE", p.chain_rotate),
        retire: RetireParams {
            k: num("EMBEDDED_MASKING_RETIRE", p.retire.k),
            warmup_pct: num("EMBEDDED_MASKING_RETIRE_WARMUP", p.retire.warmup_pct),
            safe: flag("EMBEDDED_MASKING_RETIRE_SAFE", p.retire.safe),
            burst: num("EMBEDDED_MASKING_RETIRE_BURST", p.retire.burst).max(1),
            repair: var("EMBEDDED_MASKING_RETIRE_MODE").map_or(p.retire.repair, |v| v == "repair"),
            pin_need: num("EMBEDDED_MASKING_RETIRE_PIN", p.retire.pin_need),
            norep: flag("EMBEDDED_MASKING_RETIRE_NOREPLACE", p.retire.norep),
            preopen_pct: num("EMBEDDED_MASKING_RETIRE_PREOPEN", p.retire.preopen_pct),
        },
        drift: flag("EMBEDDED_MASKING_DRIFT", p.drift),
        drift_window: match var("EMBEDDED_MASKING_DRIFT_ONLY") {
            Err(_) => p.drift_window,
            Ok(w) => Some(parse_drift_window(&w)),
        },
        anc_reserve: num("EMBEDDED_MASKING_ANC_RESERVE", p.anc_reserve),
        legacy_slice_guard: flag("SLICE_GUARD_LEGACY", p.legacy_slice_guard),
        ..p
    }
}

/// `lo:hi`; anything unparsable disables the drift everywhere (the empty
/// window `(1, 0)`), as the historical inline parse did.
pub(crate) fn parse_drift_window(w: &str) -> (usize, usize) {
    w.split_once(':')
        .and_then(|(lo, hi)| Some((lo.parse().ok()?, hi.parse().ok()?)))
        .unwrap_or((1, 0))
}

/// Diagnostics of the construction (traces, dumps) from the environment.
pub fn execution_from_env(e: EmbeddedMaskingExecution) -> EmbeddedMaskingExecution {
    EmbeddedMaskingExecution {
        lgi_debug: std::env::var("EMBEDDED_MASKING_LGI_DEBUG").is_ok(),
        straddle_census: std::env::var("EMBEDDED_MASKING_STRADDLE_CENSUS").is_ok(),
        drift_debug: std::env::var("EMBEDDED_MASKING_DRIFT_DEBUG").is_ok(),
        dump_gates: std::env::var("EMBEDDED_MASKING_DUMP_GATES")
            .ok()
            .map(std::path::PathBuf::from),
        ..e
    }
}

/// A positional argument: absent (`None`) means the default; present but
/// unparsable, including empty, is an error naming the argument (D17: a parse
/// slip must never silently become a default such as seed 1).
pub(crate) fn parse_positional<T: std::str::FromStr>(
    name: &str,
    value: Option<String>,
) -> Result<Option<T>, String> {
    match value {
        None => Ok(None),
        Some(v) => v
            .parse::<T>()
            .map(Some)
            .map_err(|_| format!("argument {name}: {v:?} is not a valid value")),
    }
}

fn positional_or_exit<T: std::str::FromStr>(name: &str, value: Option<String>) -> Option<T> {
    parse_positional(name, value).unwrap_or_else(|error| {
        eprintln!("gen_sandwich_gadget: {error}");
        std::process::exit(2)
    })
}

pub fn run() {
    let mut a = std::env::args().skip(1);
    let out = a
        .next()
        .expect("usage: gen_sandwich_gadget <out> [n m_C m_D s rg_freq slice_gates seed gadget_seed sandwich_seed gadgetization_mode] (a seed may be @FILE)");
    let n: usize = positional_or_exit("n", a.next()).unwrap_or(128);
    let m_c: usize = positional_or_exit("m_C", a.next()).unwrap_or(3000);
    let m_d: usize = positional_or_exit("m_D", a.next()).unwrap_or(3000);
    let s: usize = positional_or_exit("s", a.next()).unwrap_or_else(|| sandwich_default_s(n));
    let rg_freq: usize = positional_or_exit("rg_freq", a.next()).unwrap_or(1);
    let sandwich_n = 2 * n;
    let slice_gates: usize = positional_or_exit("slice_gates", a.next()).unwrap_or(10 * sandwich_n);
    // (value, read from a file): a file-borne seed is never printed.
    let seed_or_exit = |name: &str, value: Option<String>| -> Option<(u64, bool)> {
        match value {
            Some(v) if v.starts_with('@') => Some((
                crate::programs::read_seed_file(&v[1..]).unwrap_or_else(|error| {
                    eprintln!("gen_sandwich_gadget: argument {name}: {error}");
                    std::process::exit(2)
                }),
                true,
            )),
            other => positional_or_exit(name, other).map(|v| (v, false)),
        }
    };
    let (seed, seed_private) = seed_or_exit("seed", a.next()).unwrap_or((1, false));
    let (gadget_seed, gadget_seed_private) =
        seed_or_exit("gadget_seed", a.next()).unwrap_or((seed, seed_private));
    let (sandwich_seed, sandwich_seed_private) =
        seed_or_exit("sandwich_seed", a.next()).unwrap_or((seed, seed_private));
    let shown = |value: u64, private: bool| {
        if private {
            "(file)".to_owned()
        } else {
            value.to_string()
        }
    };
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
    let (shuffling_segments, shuffling_return_home, shuffling_jit_gap) = parse_shuffling_controls(
        gadgetization_mode,
        std::env::var("EMBEDDED_MASKING_SHUFFLING").ok().as_deref(),
        std::env::var("EMBEDDED_MASKING_SHUFFLING_RETURN_HOME")
            .ok()
            .as_deref(),
        std::env::var("EMBEDDED_MASKING_JIT").ok().as_deref(),
    )
    .unwrap_or_else(|error| panic!("invalid preprocessing settings: {error}"));
    // How a generated C is drawn; only this adapter reads the variable.
    let construction =
        parse_source_construction(std::env::var("TDP_SOURCE_CONSTRUCTION").ok().as_deref())
            .unwrap_or_else(|error| {
                eprintln!("gen_sandwich_gadget: {error}");
                std::process::exit(2)
            });

    // The driver pattern-matches the middle of this line: new fields go at
    // the end only.
    println!(
        "[gen] n={n} |C|={m_c} |D|={m_d} s={s} rg_freq={rg_freq} slice_gates={slice_gates} seed={} gadget_seed={} sandwich_seed={} gadgetization_mode={} sandwich_variant={} source_construction={}",
        shown(seed, seed_private),
        shown(gadget_seed, gadget_seed_private),
        shown(sandwich_seed, sandwich_seed_private),
        gadgetization_mode.canonical_name(),
        variant.name(),
        construction.name()
    );

    let source_path = std::env::var("TDP_SOURCE_C").ok();
    let supplied = source_path.as_ref().map(|path| {
        let raw = std::fs::read(path)
            .unwrap_or_else(|error| panic!("TDP_SOURCE_C: cannot read {path}: {error}"));
        crate::circuit::CircuitSeq::from_bytes(&raw)
    });
    let c = prepare_source(n, m_c, seed, supplied, construction).unwrap_or_else(|error| {
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
    if construction.is_balanced() {
        // The library audited C before returning it; the census is repeated
        // here only to report its numbers.
        let audit = audit_balanced(n, &c)
            .unwrap_or_else(|error| panic!("balanced source C failed its audit: {error}"));
        println!(
            "[gen] source C: balanced construction ({} gates on {} wires, {} internal births each read exactly twice, max control fan-out {}, write spread {}, final write counts {:?})",
            audit.gates,
            audit.wires,
            audit.internal_births,
            audit.maximum_control_fanout,
            audit.maximum_write_spread,
            audit.final_writes
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
                shuffling_jit_gap,
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
                // Punchlist #5: rich band seed/reseed (pivot + linear sources
                // + nonlinear products). 0 = legacy thin 2-CNOT seed.
                seed_fill_nl: std::env::var("EMBEDDED_MASKING_SEED_NL")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(4),
                execution: execution_from_env(EmbeddedMaskingExecution {
                    ancilla_band_only: std::env::var("EMBEDDED_MASKING_ANC_POOL")
                        .map_or(true, |v| v != "all"),
                    diagnostics: std::env::var("EMBEDDED_MASKING_DIAG").is_ok(),
                    record_hot_intervals: std::env::var("EMBEDDED_MASKING_HOT_MANIFEST").is_ok(),
                    ..Default::default()
                }),
                params: stage2_knobs_from_env(params),
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
                println!(
                    "[gen] compute: {} fire gates + {} shuffling gates ({} transfers, {} skipped cuts, {} ports restored, {} jit remasks, {} drain/other) ; {} rerand bursts",
                    report.original_compute_gates,
                    report.shuffling_added_gates,
                    report.transfer_count,
                    report.skipped_cuts,
                    report.restored_ports,
                    report.jit_remasks,
                    report
                        .shuffling_added_gates
                        .saturating_sub(2 * report.transfer_count + 3 * report.restored_ports),
                    report.rerand_done
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
