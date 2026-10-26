//! Resolve typed values and validate resource/path constraints before run creation.
use super::*;
use crate::stages::packing::types::{
    MAX_MONOMIAL_BUDGET, MAX_SPAN_CAP, MAX_STATE_WIRES as STAGE4_MAX_STATE_WIRES,
};
use crate::tdp::manifest::CURRENT_RECIPE_VERSION;

pub(crate) fn resolve_config<F>(
    raw: &RawConfig,
    repo_root: &Path,
    getenv: F,
    run_tag: &str,
) -> Result<ResolvedConfig, String>
where
    F: Fn(&str) -> Option<OsString>,
{
    let n = parse_usize(raw, "n", 128, 3, 4095)?;
    let run_dir_value = raw.value("run_dir");
    let run_dir = match run_dir_value {
        Some(value) => resolve_path(repo_root, Path::new(value)),
        None => fresh_default_run_dir(repo_root, n, run_tag),
    };
    validate_run_destination(repo_root, &run_dir)?;

    let build_release = parse_bool(raw, "build_release", true)?;
    let build_target_dir = match raw.value("build_target_dir") {
        Some(value) => resolve_path(repo_root, Path::new(value)),
        None => getenv("CARGO_TARGET_DIR")
            .filter(|value| !value.is_empty())
            .map(|value| resolve_path(repo_root, Path::new(&value)))
            .unwrap_or_else(|| repo_root.join("target")),
    };
    validate_build_target(repo_root, &build_target_dir)?;
    let adopt_existing_run = parse_bool(raw, "adopt_existing_run", false)?;
    let frozen_db = sourced_path(
        raw.value("frozen_db_dir"),
        "FROZEN_DB_DIR",
        repo_root,
        &getenv,
    )?;
    validate_path_separation(&run_dir, &build_target_dir, frozen_db.path.as_deref())?;

    let preprocessing_mode = parse_preprocessing_mode(raw)?;
    let source_construction = match raw.value("source_construction").unwrap_or("random") {
        value => SourceConstruction::parse(value).ok_or_else(|| {
            config_error(
                raw,
                "source_construction",
                &format!("expected \"random\" or \"balanced\"; got {value:?}"),
            )
        })?,
    };
    let calibration_only = parse_bool(raw, "calibration_only", false)?;
    let calibration_seed_file = raw
        .value("calibration_seed_file")
        .map(|value| resolve_path(repo_root, Path::new(value)));
    if calibration_seed_file.is_some() && !calibration_only {
        return Err(config_error(
            raw,
            "calibration_seed_file",
            "requires `calibration.enabled = true`; explicit seeds must never be used for deliverables",
        ));
    }
    let seeds = match &calibration_seed_file {
        Some(path) => {
            let s = read_seed_file(path)?;
            SeedPlan::Explicit {
                gen_seed: s.gen_seed,
                mix_seed: s.mix_seed,
                split_seed: s.split_seed,
                cross_seed: s.cross_seed,
                unpack_seed: s.unpack_seed,
            }
        }
        None => SeedPlan::Random,
    };

    let mut mcd = parse_optional_u64(raw, "mcd", 1, 1_000_000_000)?;
    let source_path = raw
        .value("source_path")
        .map(|value| resolve_path(repo_root, Path::new(value)));
    if source_path.is_some() && source_construction != SourceConstruction::Random {
        return Err(config_error(
            raw,
            "source_construction",
            "a supplied source.path is exclusive with the balanced construction (remove one)",
        ));
    }
    if let Some(path) = &source_path {
        let bytes =
            fs::read(path).map_err(|e| format!("cannot read source {}: {e}", path.display()))?;
        if bytes.starts_with(b"mpmct1") || bytes.starts_with(b"esop1") || bytes.starts_with(b"anf1")
        {
            return Err("TDP source.path currently requires a G57 circuit; packed/general formats are supported by circuit evaluate/compare".into());
        }
        let circuit =
            std::panic::catch_unwind(|| crate::circuit::CircuitSeq::from_bytes(&bytes))
                .map_err(|_| format!("source {} is not a valid G57 circuit", path.display()))?;
        if circuit.gates.is_empty() || circuit.max_wire() >= n {
            return Err("source must be nonempty and fit source.wires".into());
        }
        let count = circuit.gates.len().to_string();
        if mcd.as_ref().is_some_and(|value| value != &count) {
            return Err("source.gates must match the supplied circuit".into());
        }
        mcd = Some(count);
    }
    // Stage-2 preset (§11.2): authoritative; L8/A forbid every knob field.
    const KNOB_KEYS: [&str; 10] = [
        "mask_pair_wires",
        "max_open_masks",
        "min_open_masks",
        "balanced_masks",
        "shuffling_segments",
        "shuffling_return_home",
        "premask",
        "seed_avoid_c1",
        "chain",
        "chain_len",
    ];
    let present: Vec<&str> = KNOB_KEYS
        .iter()
        .copied()
        .filter(|key| raw.value(key).is_some())
        .collect();
    let (preset, knobs) = if preprocessing_mode != PreprocessingMode::EmbeddedMasking {
        let mode = preprocessing_mode.canonical_name();
        if raw.value("preset").is_some() {
            return Err(config_error(
                raw,
                "preset",
                &format!("applies only to embedded-masking; remove it for {mode}"),
            ));
        }
        if let Some(key) = present.first() {
            return Err(config_error(
                raw,
                key,
                &format!(
                    "preprocessing mask controls apply only to embedded-masking; omit them for {mode}"
                ),
            ));
        }
        (Preset::None, Stage2Knobs::BASE)
    } else {
        match raw.value("preset").unwrap_or("L8") {
            name @ ("L8" | "A") => {
                if let Some(key) = present.first() {
                    return Err(config_error(
                        raw,
                        key,
                        &format!(
                            "field {} is set by preset {name}; remove it or use preset = \"custom\"",
                            canonical_config_key(key)
                        ),
                    ));
                }
                if name == "L8" {
                    (Preset::L8, Stage2Knobs::L8)
                } else {
                    (Preset::A, Stage2Knobs::A)
                }
            }
            "custom" => {
                if let Some(key) = KNOB_KEYS.iter().find(|key| raw.value(key).is_none()) {
                    return Err(config_error(
                        raw,
                        "preset",
                        &format!(
                            "preset \"custom\" requires every stage-2 field; {} is missing",
                            canonical_config_key(key)
                        ),
                    ));
                }
                let knobs = Stage2Knobs {
                    mask_pair_wires: parse_usize(raw, "mask_pair_wires", 2, 2, 64)?,
                    max_open_masks: parse_usize(raw, "max_open_masks", 3, 2, 64)?,
                    min_open_masks: parse_usize(raw, "min_open_masks", 2, 1, 63)?,
                    balanced_masks: parse_bool(raw, "balanced_masks", true)?,
                    shuffling_segments: parse_usize(raw, "shuffling_segments", 0, 0, usize::MAX)?,
                    shuffling_return_home: parse_bool(raw, "shuffling_return_home", true)?,
                    premask: parse_bool(raw, "premask", false)?,
                    seed_avoid_c1: parse_bool(raw, "seed_avoid_c1", false)?,
                    chain: parse_usize(raw, "chain", 0, 0, 64)?,
                    chain_len: parse_usize(raw, "chain_len", 0, 0, 1_000_000)?,
                };
                if knobs.min_open_masks >= knobs.max_open_masks {
                    return Err("preprocessing.min_open_masks must be less than preprocessing.max_open_masks".into());
                }
                if (1..8).contains(&knobs.shuffling_segments) {
                    return Err(config_error(
                        raw,
                        "shuffling_segments",
                        "must be 0 (off) or at least 8",
                    ));
                }
                (Preset::Custom, knobs)
            }
            other => {
                return Err(config_error(
                    raw,
                    "preset",
                    &format!("expected \"L8\", \"A\" or \"custom\"; got {other:?}"),
                ));
            }
        }
    };
    // [three_mask]: only under preprocessing.mode = "three-mask"
    const THREE_MASK_KEYS: [&str; 13] = [
        "three_mask_band_wires",
        "three_mask_banks",
        "three_mask_retire_budget",
        "three_mask_max_retiring",
        "three_mask_rotation",
        "three_mask_maturity",
        "three_mask_visit_sweeps",
        "three_mask_growth",
        "three_mask_warmup",
        "three_mask_segments",
        "three_mask_mask_count",
        "three_mask_random_helpers",
        "three_mask_unique_cover",
    ];
    if preprocessing_mode != PreprocessingMode::ThreeMask
        && let Some(key) = THREE_MASK_KEYS
            .iter()
            .copied()
            .find(|key| raw.value(key).is_some())
    {
        return Err(config_error(
            raw,
            key,
            "applies only to preprocessing.mode = \"three-mask\"",
        ));
    }
    let tm = ThreeMaskConfig::default();
    let tm_u64 = |key: &str, default: u64, min: u64, max: u64| -> Result<u64, String> {
        Ok(parse_optional_u64(raw, key, min, max)?
            .map(|v| v.parse::<u64>().expect("validated"))
            .unwrap_or(default))
    };
    let three_mask = ThreeMaskConfig {
        band_wires: tm_u64("three_mask_band_wires", tm.band_wires, 0, 65_536)?,
        banks: tm_u64("three_mask_banks", tm.banks, 0, 4096)?,
        retire_budget: tm_u64("three_mask_retire_budget", tm.retire_budget, 1, 1000)?,
        max_retiring: tm_u64("three_mask_max_retiring", tm.max_retiring, 0, 4096)?,
        rotation: tm_u64("three_mask_rotation", tm.rotation, 0, 1_000_000)?,
        maturity: tm_u64("three_mask_maturity", tm.maturity, 0, 1_000_000)?,
        visit_sweeps: tm_u64("three_mask_visit_sweeps", tm.visit_sweeps, 1, 1_000_000)?,
        growth: tm_u64("three_mask_growth", tm.growth, 0, 1_000_000)?,
        warmup: tm_u64("three_mask_warmup", tm.warmup, 0, 1_000_000)?,
        segments: tm_u64("three_mask_segments", tm.segments, 0, 1_000_000)?,
        mask_count: tm_u64("three_mask_mask_count", tm.mask_count, 1, 3)?,
        random_helpers: parse_bool(raw, "three_mask_random_helpers", tm.random_helpers)?,
        unique_cover: parse_bool(raw, "three_mask_unique_cover", tm.unique_cover)?,
    };
    if preprocessing_mode == PreprocessingMode::ThreeMask {
        let r = three_mask.band_wires;
        if r != 0 && (r < 64 || r % 16 != 0) {
            return Err(config_error(
                raw,
                "three_mask_band_wires",
                "must be 0 (auto) or at least 64 and a multiple of 16",
            ));
        }
        if three_mask.banks != 0 && (three_mask.banks < 4 || three_mask.banks % 2 != 0) {
            return Err(config_error(
                raw,
                "three_mask_banks",
                "must be 0 (auto: banks of 16 wires) or even and at least 4",
            ));
        }
        if three_mask.max_retiring != 0
            && three_mask.banks != 0
            && three_mask.max_retiring >= three_mask.banks / 2
        {
            return Err(config_error(
                raw,
                "three_mask_max_retiring",
                "must be 0 (auto) or less than half the bank count",
            ));
        }
    }
    let qc_enabled = parse_bool(raw, "qc_enabled", false)?;
    let qc_seed = parse_optional_u64(raw, "qc_seed", 0, u64::MAX)?
        .unwrap_or_else(|| "20803".into())
        .parse::<u64>()
        .expect("validated QC seed");
    let qc_reference = raw
        .value("qc_reference")
        .map(|value| resolve_path(repo_root, Path::new(value)));
    if !qc_enabled && (qc_reference.is_some() || raw.value("qc_seed").is_some()) {
        return Err("leakage_repair.seed/reference require leakage_repair.enabled = true".into());
    }
    if let Some(path) = &qc_reference {
        fs::metadata(path).map_err(|e| {
            format!(
                "cannot read leakage-repair reference {}: {e}",
                path.display()
            )
        })?;
    }
    // Source size (§2.4): the validated table, an explicit size, or the
    // opt-in formula.
    let validated = VALIDATED_SOURCE_GATES
        .iter()
        .find(|(wires, _)| *wires == n)
        .map(|(_, gates)| *gates);
    let (mcd, mcd_rule) = match mcd {
        Some(value) => {
            if raw.value("source_gates_rule").is_some() {
                return Err(config_error(
                    raw,
                    "source_gates_rule",
                    "is exclusive with source.gates (and source.path)",
                ));
            }
            let value: usize = value.parse().expect("validated mcd");
            let rule = if Some(value) == validated {
                SourceGatesRule::Validated
            } else {
                SourceGatesRule::Explicit
            };
            (value, rule)
        }
        None => match raw.value("source_gates_rule").unwrap_or("validated") {
            "validated" => match validated {
                Some(gates) => (gates, SourceGatesRule::Validated),
                None => {
                    return Err(format!(
                        "no validated source size for n={n}; set source.gates or source.gates_rule = \"formula\""
                    ));
                }
            },
            "formula" => {
                let l = (n as f64).log2();
                (
                    (n as f64 * l * l).round_ties_even() as usize,
                    SourceGatesRule::Formula,
                )
            }
            other => {
                return Err(config_error(
                    raw,
                    "source_gates_rule",
                    &format!("expected \"validated\" or \"formula\"; got {other:?}"),
                ));
            }
        },
    };
    // nonlinear291's state width (embedded masking's is 4n <= 1024 for n <= 256)
    let mut nonlinear_state_wires = None;
    if preprocessing_mode == PreprocessingMode::Nonlinear291 {
        let m = mcd;
        let s = sandwich_default_s(n);
        let gate_count = m
            .checked_mul(2)
            .and_then(|count| count.checked_add(s.checked_mul(2)?))
            .and_then(|count| count.checked_add(n))
            .ok_or_else(|| {
                config_error(raw, "preprocessing_mode", "sandwich gate-count overflow")
            })?;
        let logical_n = n.checked_mul(2).ok_or_else(|| {
            config_error(raw, "preprocessing_mode", "logical wire-count overflow")
        })?;
        let slice_gates = logical_n
            .checked_mul(10)
            .ok_or_else(|| config_error(raw, "preprocessing_mode", "slice gate-count overflow"))?;
        let (state_wires, _) = nonlinear_tdp_resource_plan(logical_n, gate_count, slice_gates)
            .map_err(|error| {
                config_error(
                    raw,
                    "preprocessing_mode",
                    &format!(
                        "requested nonlinear layout exceeds the current wire capacity/resource budget: {error}; reduce n or set a smaller source.gates"
                    ),
                )
            })?;
        nonlinear_state_wires = Some(state_wires);
    }
    // Stage-3 schedule (§11.2): authoritative.
    let schedule = match raw.value("schedule").unwrap_or("validated") {
        "validated" => {
            for key in ["growth", "hold", "expand"] {
                if raw.value(key).is_some() {
                    return Err(config_error(
                        raw,
                        key,
                        &format!(
                            "field {} is set by schedule validated; remove it or use db_mixing.schedule = \"custom\"",
                            canonical_config_key(key)
                        ),
                    ));
                }
            }
            Schedule::Validated
        }
        "custom" => {
            for key in ["growth", "hold", "expand"] {
                if raw.value(key).is_none() {
                    return Err(config_error(
                        raw,
                        "schedule",
                        &format!("schedule \"custom\" requires {}", canonical_config_key(key)),
                    ));
                }
            }
            let growth = parse_optional_f64(
                raw,
                "growth",
                |value| value > 0.0 && value <= 10_000.0,
                "must be greater than 0 and at most 10000",
            )?
            .expect("checked");
            let hold = parse_optional_f64(
                raw,
                "hold",
                |value| (0.0..=10_000.0).contains(&value),
                "must be in 0..=10000",
            )?
            .expect("checked");
            let expand = parse_optional_f64(
                raw,
                "expand",
                |value| value > 1.0 && value <= 16.0,
                "must be greater than 1 and at most 16",
            )?
            .expect("checked");
            Schedule::Custom {
                growth,
                hold,
                expand,
            }
        }
        other => {
            return Err(config_error(
                raw,
                "schedule",
                &format!("expected \"validated\" or \"custom\"; got {other:?}"),
            ));
        }
    };
    let stop_after = parse_optional_stage(raw, "stop_after")?;
    let force_from = parse_optional_stage(raw, "force_from")?;
    // Optional stages (every stage after the first): run.skip_stages.
    let stop_after_stage: u64 = stop_after
        .as_deref()
        .unwrap_or("6")
        .parse()
        .expect("validated");
    let mut skip_stages: Vec<u64> = Vec::new();
    if let Some(list) = raw.value("skip_stages") {
        for item in list.split(',') {
            let stage: u64 = item
                .parse()
                .ok()
                .filter(|k| (2..=LAST_STAGE).contains(k))
                .ok_or_else(|| {
                    config_error(
                        raw,
                        "skip_stages",
                        "must list stages in 2..6 (stage 1 always runs; stages 6 and 7 of config_version 3 are one delivery stage 6)",
                    )
                })?;
            if !skip_stages.contains(&stage) {
                skip_stages.push(stage);
            }
        }
        skip_stages.sort_unstable();
    }
    let stage_on = |k: u64| k <= stop_after_stage && !skip_stages.contains(&k);
    if stage_on(6) && n > 256 {
        return Err(config_error(
            raw,
            "n",
            "stage 6 (delivery, adaptive or windowed) supports source.wires up to 256; stop before stage 6 or skip it for larger n",
        ));
    }
    // Refuse up front what stage 6 would refuse only after stages 1-5 ran.
    if let Some(state_wires) = nonlinear_state_wires
        && stage_on(6)
        && state_wires > STAGE4_MAX_STATE_WIRES
    {
        return Err(config_error(
            raw,
            "preprocessing_mode",
            &format!(
                "nonlinear291 makes a {state_wires}-wire stage-2 circuit; stage 6 (delivery) supports at most {STAGE4_MAX_STATE_WIRES} state wires; stop before stage 6, skip it, or use embedded-masking"
            ),
        ));
    }
    // [crossing] (stage 5), the archive-era driver's ranges
    let crossing = CrossingConfig {
        xr: parse_optional_f64(
            raw,
            "xr",
            |value| (1.0..=16.0).contains(&value),
            "must be in 1..=16",
        )?,
        xb: parse_optional_f64(
            raw,
            "xb",
            |value| (1.0..=1_000_000.0).contains(&value),
            "must be in 1..=1000000",
        )?,
        xc: parse_optional_u64(raw, "xc", 0, 1_000_000_000)?,
        xtdiv: parse_optional_u64(raw, "xtdiv", 1, 1_000_000_000)?,
        xmf: parse_optional_f64(
            raw,
            "xmf",
            |value| value > 0.0 && value <= 1000.0,
            "must be in (0, 1000]",
        )?,
        xmoves: parse_optional_u64(raw, "xmoves", 1, 1_000_000_000_000)?,
    };
    if crossing.xmf.is_some() && crossing.xmoves.is_some() {
        return Err(config_error(
            raw,
            "xmoves",
            "an absolute move budget is exclusive with `crossing.move_budget_factor`",
        ));
    }
    // [delivery]: the mode, then the windowed mode's unpacking sub-step
    let dd = DeliveryConfig::default();
    let delivery_mode = match raw.value("delivery_mode").unwrap_or("adaptive") {
        value => DeliveryMode::parse(value).ok_or_else(|| {
            config_error(
                raw,
                "delivery_mode",
                &format!("expected \"adaptive\" or \"windowed\"; got {value:?}"),
            )
        })?,
    };
    const WINDOWED_KEYS: [&str; 15] = [
        "delivery_anf_only",
        "delivery_time_limit",
        "delivery_restarts",
        "delivery_beam",
        "delivery_max_rounds",
        "packing_window_gates",
        "packing_interval_cap",
        "packing_monomial_budget",
        "packing_span_cap",
        "packing_split_budget",
        "packing_fix_iterations",
        "packing_fallback_iterations",
        "packing_anf_cap_semantics",
        "packing_allow_floor_miss",
        "packing_allow_over_budget",
    ];
    const ADAPTIVE_KEYS: [&str; 15] = [
        "adaptive_max_rounds",
        "adaptive_stop_fraction",
        "adaptive_seeds_per_block",
        "adaptive_time_limit",
        "adaptive_restarts",
        "adaptive_beam",
        "adaptive_term_cap",
        "adaptive_max_block_gates",
        "adaptive_segments",
        "adaptive_avoid_radius",
        "adaptive_refinement_levels",
        "adaptive_min_refine_gates",
        "adaptive_census_batches",
        "adaptive_accept",
        "adaptive_canonical_order",
    ];
    if delivery_mode == DeliveryMode::Adaptive
        && let Some(key) = WINDOWED_KEYS
            .iter()
            .copied()
            .find(|key| raw.value(key).is_some())
    {
        return Err(config_error(
            raw,
            key,
            "applies only to delivery.mode = \"windowed\" (the adaptive mode's parameters are [adaptive])",
        ));
    }
    if delivery_mode == DeliveryMode::Windowed
        && let Some(key) = ADAPTIVE_KEYS
            .iter()
            .copied()
            .find(|key| raw.value(key).is_some())
    {
        return Err(config_error(
            raw,
            key,
            "applies only to delivery.mode = \"adaptive\"",
        ));
    }
    let ad = AdaptiveConfig::default();
    let ad_u64 = |key: &str, default: u64, min: u64, max: u64| -> Result<u64, String> {
        Ok(parse_optional_u64(raw, key, min, max)?
            .map(|v| v.parse::<u64>().expect("validated"))
            .unwrap_or(default))
    };
    let adaptive = AdaptiveConfig {
        max_rounds: ad_u64("adaptive_max_rounds", ad.max_rounds, 0, 1000)?,
        stop_fraction: parse_optional_f64(
            raw,
            "adaptive_stop_fraction",
            |value| (0.0..1.0).contains(&value),
            "must be in [0, 1)",
        )?
        .unwrap_or(ad.stop_fraction),
        seeds_per_block: ad_u64("adaptive_seeds_per_block", ad.seeds_per_block, 1, 16)?,
        time_limit: parse_optional_f64(
            raw,
            "adaptive_time_limit",
            |value| value > 0.0 && value <= 86_400.0,
            "must be in (0, 86400]",
        )?
        .unwrap_or(ad.time_limit),
        restarts: ad_u64("adaptive_restarts", ad.restarts, 1, 1000)?,
        beam: ad_u64("adaptive_beam", ad.beam, 1, 1000)?,
        term_cap: ad_u64("adaptive_term_cap", ad.term_cap, 64, 16_777_216)?,
        max_block_gates: ad_u64(
            "adaptive_max_block_gates",
            ad.max_block_gates,
            2,
            100_000_000,
        )?,
        segments: ad_u64("adaptive_segments", ad.segments, 1, 1024)?,
        avoid_radius: ad_u64("adaptive_avoid_radius", ad.avoid_radius, 0, 1_000_000)?,
        refinement_levels: ad_u64("adaptive_refinement_levels", ad.refinement_levels, 0, 16)?,
        min_refine_gates: ad_u64(
            "adaptive_min_refine_gates",
            ad.min_refine_gates,
            1,
            100_000_000,
        )?,
        census_batches: ad_u64("adaptive_census_batches", ad.census_batches, 1, 4096)?,
        accept: match raw
            .value("adaptive_accept")
            .unwrap_or("smaller-not-leakier")
        {
            v @ ("smaller-not-leakier" | "smaller") => v.to_string(),
            other => {
                return Err(config_error(
                    raw,
                    "adaptive_accept",
                    &format!("expected \"smaller-not-leakier\" or \"smaller\"; got {other:?}"),
                ));
            }
        },
        canonical_order: parse_bool(raw, "adaptive_canonical_order", ad.canonical_order)?,
    };
    let delivery = DeliveryConfig {
        mode: delivery_mode,
        anf_only: parse_bool(raw, "delivery_anf_only", dd.anf_only)?,
        time_limit: parse_optional_f64(
            raw,
            "delivery_time_limit",
            |value| value > 0.0 && value <= 86_400.0,
            "must be in (0, 86400]",
        )?
        .unwrap_or(dd.time_limit),
        restarts: parse_optional_u64(raw, "delivery_restarts", 1, 1000)?
            .map(|v| v.parse().expect("validated"))
            .unwrap_or(dd.restarts),
        beam: parse_optional_u64(raw, "delivery_beam", 1, 1000)?
            .map(|v| v.parse().expect("validated"))
            .unwrap_or(dd.beam),
        max_rounds: parse_optional_u64(raw, "delivery_max_rounds", 1, 1000)?
            .map(|v| v.parse().expect("validated"))
            .unwrap_or(dd.max_rounds),
    };
    // [packing]
    let d = PackingConfig::default();
    let packing_u64 = |key: &str, default: u64, max: u64| -> Result<u64, String> {
        Ok(parse_optional_u64(raw, key, 1, max)?
            .map(|v| v.parse::<u64>().expect("validated"))
            .unwrap_or(default))
    };
    let packing = PackingConfig {
        window_gates: packing_u64("packing_window_gates", d.window_gates, 1_000_000)?,
        interval_cap: packing_u64("packing_interval_cap", d.interval_cap, 1_000_000)?,
        monomial_budget: packing_u64(
            "packing_monomial_budget",
            d.monomial_budget,
            MAX_MONOMIAL_BUDGET as u64,
        )?,
        span_cap: packing_u64("packing_span_cap", d.span_cap, MAX_SPAN_CAP as u64)?,
        split_budget: packing_u64(
            "packing_split_budget",
            d.split_budget,
            MAX_MONOMIAL_BUDGET as u64,
        )?,
        fix_iterations: parse_optional_u64(raw, "packing_fix_iterations", 0, 1000)?
            .map(|v| v.parse().expect("validated"))
            .unwrap_or(d.fix_iterations),
        fallback_iterations: parse_optional_u64(raw, "packing_fallback_iterations", 0, 1000)?
            .map(|v| v.parse().expect("validated"))
            .unwrap_or(d.fallback_iterations),
        anf_cap_semantics: match raw
            .value("packing_anf_cap_semantics")
            .unwrap_or("cpython39")
        {
            v @ ("cpython39" | "canonical") => v.to_string(),
            other => {
                return Err(config_error(
                    raw,
                    "packing_anf_cap_semantics",
                    &format!("expected \"cpython39\" or \"canonical\"; got {other:?}"),
                ));
            }
        },
        allow_floor_miss: parse_bool(raw, "packing_allow_floor_miss", false)?,
        allow_over_budget: parse_bool(raw, "packing_allow_over_budget", false)?,
    };
    let threads = parse_optional_u64(raw, "threads", 1, 1024)?;
    let pieces = parse_optional_u64(raw, "pieces", 1, 64)?;
    let min_block_size = parse_optional_u64(raw, "min_block_size", 2, 1_000_000_000)?;
    if pieces.is_some() && min_block_size.is_some() {
        return Err(config_error(
            raw,
            "min_block_size",
            "is mutually exclusive with `parallel.pieces` (including parallel.pieces = 1)",
        ));
    }
    let piece_threads = parse_optional_u64(raw, "piece_threads", 1, 1024)?;
    if piece_threads.is_some()
        && min_block_size.is_none()
        && pieces.as_deref().is_none_or(|value| value == "1")
    {
        return Err(config_error(
            raw,
            "piece_threads",
            "applies only when `parallel.pieces` is greater than 1 or `parallel.target_piece_gates` is set",
        ));
    }
    let allow_empty_store = parse_bool(raw, "allow_empty_store", false)?;
    let frozen_filter = match raw.value("frozen_filter").unwrap_or("auto") {
        "auto" => FrozenFilter::Auto,
        "on" => FrozenFilter::On,
        "off" => FrozenFilter::Off,
        other => {
            return Err(config_error(
                raw,
                "frozen_filter",
                &format!("expected auto, on, or off; got {other:?}"),
            ));
        }
    };

    Ok(ResolvedConfig {
        recipe_version: CURRENT_RECIPE_VERSION,
        source_path,
        preset,
        knobs,
        qc_enabled,
        qc_seed,
        qc_reference,
        n,
        run_dir,
        generated_run_dir: run_dir_value.is_none(),
        build_release,
        build_target_dir,
        adopt_existing_run,
        frozen_db,
        preprocessing_mode,
        three_mask,
        source_construction,
        calibration_only,
        seeds,
        mcd,
        mcd_rule,
        schedule,
        packing,
        crossing,
        delivery,
        adaptive,
        threads,
        stop_after,
        force_from,
        skip_stages,
        allow_empty_store,
        frozen_filter,
        pieces,
        min_block_size,
        piece_threads,
    })
}

pub(crate) fn parse_usize(
    raw: &RawConfig,
    key: &str,
    default: usize,
    min: usize,
    max: usize,
) -> Result<usize, String> {
    let Some(value) = raw.value(key) else {
        return Ok(default);
    };
    let parsed = value
        .parse::<usize>()
        .map_err(|_| config_error(raw, key, &format!("expected an integer in {min}..={max}")))?;
    if !(min..=max).contains(&parsed) {
        return Err(config_error(
            raw,
            key,
            &format!("must be in {min}..={max}, got {parsed}"),
        ));
    }
    Ok(parsed)
}

/// A pipeline stage number 2..6 (stage 1 is never a stop or rerun point on
/// its own: stages 1 and 2 run in one process).
fn parse_optional_stage(raw: &RawConfig, key: &str) -> Result<Option<String>, String> {
    parse_optional_u64(raw, key, 2, LAST_STAGE)
}

pub(crate) fn parse_optional_u64(
    raw: &RawConfig,
    key: &str,
    min: u64,
    max: u64,
) -> Result<Option<String>, String> {
    let Some(value) = raw.value(key) else {
        return Ok(None);
    };
    let parsed = value
        .parse::<u64>()
        .map_err(|_| config_error(raw, key, &format!("expected an integer in {min}..={max}")))?;
    if !(min..=max).contains(&parsed) {
        return Err(config_error(
            raw,
            key,
            &format!("must be in {min}..={max}, got {parsed}"),
        ));
    }
    Ok(Some(parsed.to_string()))
}

pub(crate) fn parse_optional_f64(
    raw: &RawConfig,
    key: &str,
    valid: impl Fn(f64) -> bool,
    expectation: &str,
) -> Result<Option<String>, String> {
    let Some(value) = raw.value(key) else {
        return Ok(None);
    };
    let parsed = value
        .parse::<f64>()
        .map_err(|_| config_error(raw, key, "expected a finite number"))?;
    if !parsed.is_finite() || !valid(parsed) {
        return Err(config_error(
            raw,
            key,
            &format!("{expectation}; got {value:?}"),
        ));
    }
    Ok(Some(parsed.to_string()))
}

pub(crate) fn parse_bool(raw: &RawConfig, key: &str, default: bool) -> Result<bool, String> {
    match raw.value(key) {
        None => Ok(default),
        Some("true") => Ok(true),
        Some("false") => Ok(false),
        Some(other) => Err(config_error(
            raw,
            key,
            &format!("expected true or false; got {other:?}"),
        )),
    }
}

pub(crate) fn config_error(raw: &RawConfig, key: &str, message: &str) -> String {
    let display_key = canonical_config_key(key);
    match raw.line(key) {
        Some(line) => format!("line {line} ({display_key}): {message}"),
        None => format!("{display_key}: {message}"),
    }
}

pub(crate) fn sourced_path<F>(
    document_value: Option<&str>,
    env_key: &str,
    repo_root: &Path,
    getenv: &F,
) -> Result<SourcedPath, String>
where
    F: Fn(&str) -> Option<OsString>,
{
    if let Some(value) = document_value {
        let path = resolve_path(repo_root, Path::new(value));
        return Ok(SourcedPath {
            path: Some(canonicalize_if_present(path)?),
            source: ValueSource::Document,
        });
    }
    match getenv(env_key) {
        Some(value) if !value.is_empty() => {
            let path = resolve_path(repo_root, Path::new(&value));
            Ok(SourcedPath {
                path: Some(canonicalize_if_present(path)?),
                source: ValueSource::Environment,
            })
        }
        _ => Ok(SourcedPath {
            path: None,
            source: ValueSource::Unset,
        }),
    }
}

pub(crate) fn canonicalize_if_present(path: PathBuf) -> Result<PathBuf, String> {
    if !path.exists() {
        return Ok(path);
    }
    path.canonicalize()
        .map_err(|error| format!("cannot canonicalize {}: {error}", path.display()))
}

/// `[calibration] seed_file`, format 2 (§8.2): a private file (mode 0600,
/// at most 256 bytes) with exactly the lines `gen_seed=<u63>` and
/// `mix_seed=<u63>`. A one-integer file (format 1) is rejected with the
/// migration message. Values are never echoed in errors.
/// The seeds an explicit seed file supplies: gen and mix are required; the
/// stage-4/5/6 seeds (`split_seed`, `cross_seed`, `unpack_seed` = the
/// delivery seed) are optional (the driver draws them when absent).
#[derive(Clone, Copy)]
pub(crate) struct ExplicitSeeds {
    pub(crate) gen_seed: u64,
    pub(crate) mix_seed: u64,
    pub(crate) split_seed: Option<u64>,
    pub(crate) cross_seed: Option<u64>,
    pub(crate) unpack_seed: Option<u64>,
}

pub(crate) fn read_seed_file(path: &Path) -> Result<ExplicitSeeds, String> {
    let metadata = fs::metadata(path)
        .map_err(|error| format!("seed_file {} cannot be read: {error}", path.display()))?;
    if !metadata.is_file() {
        return Err(format!(
            "seed_file {} is not a regular file",
            path.display()
        ));
    }
    if metadata.len() > MAX_SEED_FILE_BYTES {
        return Err(format!(
            "seed_file {} is too large (maximum {MAX_SEED_FILE_BYTES} bytes)",
            path.display()
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o077 != 0 {
            return Err(format!(
                "seed_file {} must not be accessible by group/other (use chmod 600)",
                path.display()
            ));
        }
    }
    let contents = fs::read_to_string(path)
        .map_err(|error| format!("seed_file {} cannot be read: {error}", path.display()))?;
    parse_seed_text(&contents).map_err(|e| format!("seed_file {}: {e}", path.display()))
}

pub(crate) fn parse_seed_text(contents: &str) -> Result<ExplicitSeeds, String> {
    let lines: Vec<&str> = contents.lines().filter(|l| !l.is_empty()).collect();
    if lines.len() == 1 && lines[0].trim().chars().all(|c| c.is_ascii_digit()) {
        return Err("seed file format 1 holds one seed; recipe v13 needs gen_seed= and mix_seed= lines (the old seed was the gen seed; the old mix seed was gen+2)".into());
    }
    const LINES: &str = "only gen_seed=, mix_seed=, split_seed=, cross_seed=, unpack_seed=";
    let mut gen_seed = None;
    let mut mix_seed = None;
    let mut split_seed = None;
    let mut cross_seed = None;
    let mut unpack_seed = None;
    for line in lines {
        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| format!("unexpected line ({LINES})"))?;
        let slot = match key {
            "gen_seed" => &mut gen_seed,
            "mix_seed" => &mut mix_seed,
            "split_seed" => &mut split_seed,
            "cross_seed" => &mut cross_seed,
            "unpack_seed" => &mut unpack_seed,
            _ => return Err(format!("unexpected line ({LINES})")),
        };
        if slot.is_some() {
            return Err(format!("duplicate {key}"));
        }
        if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
            return Err(format!("{key} must be an unsigned decimal integer"));
        }
        let parsed: u64 = value
            .parse()
            .ok()
            .filter(|v| *v <= MAX_SEED)
            .ok_or_else(|| format!("{key} out of range 0..={MAX_SEED}"))?;
        *slot = Some(parsed);
    }
    match (gen_seed, mix_seed) {
        (Some(gen_seed), Some(mix_seed)) => Ok(ExplicitSeeds {
            gen_seed,
            mix_seed,
            split_seed,
            cross_seed,
            unpack_seed,
        }),
        _ => Err("needs both gen_seed= and mix_seed=".into()),
    }
}
