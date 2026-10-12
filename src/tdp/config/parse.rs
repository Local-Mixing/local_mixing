//! Read the current TOML recipe, normalizing supported aliases before validation.
use super::*;

pub(crate) fn parse_config(document: &str) -> Result<RawConfig, String> {
    if document.contains("<!-- TDP_GEN_CONFIG_BEGIN -->")
        || document.contains("<!-- TDP_GEN_CONFIG_END -->")
    {
        return Err(
            "Markdown TDP recipes have been retired; use a TOML recipe based on configs/tdp.toml"
                .into(),
        );
    }
    parse_toml(document)
}

pub(crate) fn parse_preprocessing_mode(raw: &RawConfig) -> Result<PreprocessingMode, String> {
    let value = raw
        .value("preprocessing_mode")
        .unwrap_or("embedded-masking");
    if let Some(mode) = PreprocessingMode::parse(value) {
        return Ok(mode);
    }
    let message = match value {
        "product-2223" | "2223" | "nonlinear193" => format!(
            "{value} has been retired from TDP; preprocessing.mode must be embedded-masking or nonlinear291"
        ),
        other => format!("expected embedded-masking or nonlinear291; got {other:?}"),
    };
    Err(config_error(raw, "preprocessing_mode", &message))
}

// Canonical TOML name, compatibility spelling, internal recipe key, value type.
// Public names are resolved here before value validation.
pub(crate) const TOML_FIELDS: &[(&str, &str, &str, &str)] = &[
    ("run.directory", "run.directory", "run_dir", "string"),
    (
        "run.build_binaries",
        "run.build_release",
        "build_release",
        "bool",
    ),
    (
        "run.build_directory",
        "run.build_target_dir",
        "build_target_dir",
        "string",
    ),
    (
        "run.stop_after_stage",
        "run.stop_after",
        "stop_after",
        "integer",
    ),
    (
        "run.rerun_from_stage",
        "run.force_from",
        "force_from",
        "integer",
    ),
    (
        "run.skip_stages",
        "run.skip_stages",
        "skip_stages",
        "integer_list",
    ),
    (
        "run.adopt_unverified_run",
        "run.adopt_existing_run",
        "adopt_existing_run",
        "bool",
    ),
    ("source.wires", "source.wires", "n", "integer"),
    ("source.gates", "source.gates", "mcd", "integer"),
    ("source.path", "source.path", "source_path", "string"),
    (
        "source.gates_rule",
        "source.gates_rule",
        "source_gates_rule",
        "string",
    ),
    (
        "database.regular_dir",
        "database.regular_dir",
        "frozen_db_dir",
        "string",
    ),
    (
        "database.lookup_miss_filter",
        "database.filter",
        "frozen_filter",
        "string",
    ),
    (
        "database.allow_no_database_for_tests",
        "database.allow_empty_store",
        "allow_empty_store",
        "bool",
    ),
    (
        "preprocessing.mode",
        "preprocessing.mode",
        "preprocessing_mode",
        "string",
    ),
    (
        "preprocessing.preset",
        "preprocessing.preset",
        "preset",
        "string",
    ),
    (
        "preprocessing.premask",
        "preprocessing.premask",
        "premask",
        "bool",
    ),
    (
        "preprocessing.seed_avoid_c1",
        "preprocessing.seed_avoid_c1",
        "seed_avoid_c1",
        "bool",
    ),
    (
        "preprocessing.chain",
        "preprocessing.chain",
        "chain",
        "integer",
    ),
    (
        "preprocessing.chain_len",
        "preprocessing.chain_len",
        "chain_len",
        "integer",
    ),
    (
        "preprocessing.mask_pair_wires",
        "preprocessing.mask_pair_wires",
        "mask_pair_wires",
        "integer",
    ),
    (
        "preprocessing.max_open_masks",
        "preprocessing.max_open_masks",
        "max_open_masks",
        "integer",
    ),
    (
        "preprocessing.min_open_masks",
        "preprocessing.min_open_masks",
        "min_open_masks",
        "integer",
    ),
    (
        "preprocessing.balanced_masks",
        "preprocessing.balanced_masks",
        "balanced_masks",
        "bool",
    ),
    (
        "preprocessing.shuffling_segments",
        "preprocessing.shuffling_segments",
        "shuffling_segments",
        "integer",
    ),
    (
        "preprocessing.shuffling_return_home",
        "preprocessing.shuffling_return_home",
        "shuffling_return_home",
        "bool",
    ),
    (
        "db_mixing.schedule",
        "db_mixing.schedule",
        "schedule",
        "string",
    ),
    (
        "db_mixing.growth_work_units",
        "db_mixing.growth_work_units",
        "growth",
        "number",
    ),
    (
        "db_mixing.target_size_factor",
        "phase_a.expand",
        "expand",
        "number",
    ),
    (
        "db_mixing.hold_work_units",
        "phase_a.hold",
        "hold",
        "number",
    ),
    ("run.threads", "run.threads", "threads", "integer"),
    (
        "packing.window_gates",
        "packing.window_gates",
        "packing_window_gates",
        "integer",
    ),
    (
        "packing.interval_cap",
        "packing.interval_cap",
        "packing_interval_cap",
        "integer",
    ),
    (
        "packing.monomial_budget",
        "packing.monomial_budget",
        "packing_monomial_budget",
        "integer",
    ),
    (
        "packing.span_cap",
        "packing.span_cap",
        "packing_span_cap",
        "integer",
    ),
    (
        "packing.split_budget",
        "packing.split_budget",
        "packing_split_budget",
        "integer",
    ),
    (
        "packing.fix_iterations",
        "packing.fix_iterations",
        "packing_fix_iterations",
        "integer",
    ),
    (
        "packing.fallback_iterations",
        "packing.fallback_iterations",
        "packing_fallback_iterations",
        "integer",
    ),
    (
        "packing.anf_cap_semantics",
        "packing.anf_cap_semantics",
        "packing_anf_cap_semantics",
        "string",
    ),
    (
        "packing.allow_floor_miss",
        "packing.allow_floor_miss",
        "packing_allow_floor_miss",
        "bool",
    ),
    (
        "packing.allow_over_budget",
        "packing.allow_over_budget",
        "packing_allow_over_budget",
        "bool",
    ),
    ("parallel.pieces", "parallel.pieces", "pieces", "integer"),
    (
        "parallel.target_piece_gates",
        "parallel.min_block_size",
        "min_block_size",
        "integer",
    ),
    (
        "parallel.threads",
        "parallel.threads",
        "piece_threads",
        "integer",
    ),
    (
        "leakage_repair.enabled",
        "quality_control.enabled",
        "qc_enabled",
        "bool",
    ),
    (
        "leakage_repair.seed",
        "quality_control.seed",
        "qc_seed",
        "integer",
    ),
    (
        "leakage_repair.reference",
        "quality_control.reference",
        "qc_reference",
        "string",
    ),
    (
        "crossing.target_size_factor",
        "crossing.target_factor",
        "xr",
        "number",
    ),
    (
        "crossing.width_penalty_base",
        "crossing.width_base",
        "xb",
        "number",
    ),
    (
        "crossing.width_penalty_threshold",
        "crossing.width_threshold",
        "xc",
        "integer",
    ),
    (
        "crossing.size_tolerance_divisor",
        "crossing.temperature_divisor",
        "xtdiv",
        "integer",
    ),
    (
        "crossing.move_attempts",
        "crossing.moves",
        "xmoves",
        "integer",
    ),
    (
        "unpacking.time_limit",
        "unpacking.time_limit",
        "unpacking_time_limit",
        "number",
    ),
    (
        "unpacking.restarts",
        "unpacking.restarts",
        "unpacking_restarts",
        "integer",
    ),
    (
        "unpacking.beam",
        "unpacking.beam",
        "unpacking_beam",
        "integer",
    ),
    (
        "unpacking.allow_fallback",
        "unpacking.allow_fallback",
        "unpacking_allow_fallback",
        "bool",
    ),
    (
        "calibration.enabled",
        "calibration.enabled",
        "calibration_only",
        "bool",
    ),
    (
        "calibration.seed_file",
        "calibration.seed_file",
        "calibration_seed_file",
        "string",
    ),
];

/// The migration message for a config_version 1 file (§11.2).
pub(crate) const CONFIG_V1_MIGRATION: &str = "config_version 1 is not supported by recipe v11: defaults now reproduce the chosen L8 runs. Remove shuffling_segments, shuffling_return_home, hold_work_units and stop_after_stage = 3 copied from the old template; use preprocessing.preset / db_mixing.schedule; seed files need gen_seed=/mix_seed= lines. See docs/STAGE4_RUST_DESIGN.md §11.2.";
/// The migration message for a config_version 2 file (the four-stage driver).
pub(crate) const CONFIG_V2_MIGRATION: &str = "config_version 2 is the four-stage recipe (v10); the pipeline now has seven stages (4 splitting, 5 crossing, 6 packing, 7 unpacking; all on by default, run.skip_stages turns stages off). Set config_version = 3; the four-stage chain is run.skip_stages = [4, 5, 7]. See docs/tdp_pipeline.md.";

pub(crate) fn canonical_config_key(key: &str) -> &str {
    TOML_FIELDS
        .iter()
        .find(|field| field.2 == key)
        .map_or(key, |field| field.0)
}

pub(crate) fn parse_toml(document: &str) -> Result<RawConfig, String> {
    let doc = toml_edit::Document::parse(document).map_err(|e| e.to_string())?;
    match doc
        .get("config_version")
        .and_then(toml_edit::Item::as_integer)
    {
        Some(3) => {}
        Some(2) => return Err(CONFIG_V2_MIGRATION.into()),
        Some(1) => return Err(CONFIG_V1_MIGRATION.into()),
        _ => return Err("config_version must be the integer 3".into()),
    }
    let mut raw = RawConfig::default();
    for (section, item) in doc.iter() {
        if section == "config_version" {
            continue;
        }
        let table = item
            .as_table()
            .ok_or_else(|| format!("{section} must be a TOML section"))?;
        for (key, item) in table.iter() {
            let dotted = format!("{section}.{key}");
            let &(canonical, _, flat, kind) = TOML_FIELDS
                .iter()
                .find(|field| dotted == field.0 || dotted == field.1)
                .ok_or_else(|| format!("unknown TDP config key {dotted}"))?;
            if raw.entries.contains_key(flat) {
                return Err(format!(
                    "duplicate TDP setting {canonical}; {dotted} and its compatibility alias cannot both be specified"
                ));
            }
            let value = match kind {
                "string" => item.as_str().map(str::to_owned),
                "bool" => item.as_bool().map(|v| v.to_string()),
                "integer" => item.as_integer().map(|v| v.to_string()),
                "number" => item
                    .as_float()
                    .map(|v| v.to_string())
                    .or_else(|| item.as_integer().map(|v| v.to_string())),
                // A TOML array of integers, carried as "a,b,c" (empty = unset).
                "integer_list" => item.as_array().and_then(|array| {
                    array
                        .iter()
                        .map(|v| v.as_integer().map(|v| v.to_string()))
                        .collect::<Option<Vec<_>>>()
                        .map(|v| v.join(","))
                }),
                _ => unreachable!(),
            }
            .ok_or_else(|| format!("{dotted} must be a TOML {kind}"))?;
            if value.chars().any(char::is_control) {
                return Err(format!("{dotted} must not contain control characters"));
            }
            let line = item
                .span()
                .map(|span| document[..span.start].lines().count())
                .unwrap_or(1)
                .max(1);
            raw.entries.insert(
                flat.into(),
                ConfigEntry {
                    value: (!value.is_empty()).then_some(value),
                    line,
                },
            );
        }
        if table.is_empty()
            && ![
                "run",
                "source",
                "database",
                "preprocessing",
                "phase_a",
                "db_mixing",
                "parallel",
                "quality_control",
                "leakage_repair",
                "calibration",
                "packing",
                "crossing",
                "unpacking",
            ]
            .contains(&section)
        {
            return Err(format!("unknown TDP section {section}"));
        }
    }
    Ok(raw)
}
