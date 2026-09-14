//! Read the current TOML recipe, normalizing supported aliases before validation.
use super::*;

pub(crate) fn parse_config(document: &str) -> Result<RawConfig, String> {
    if document.contains("<!-- GSS_MIX_CONFIG_BEGIN -->")
        || document.contains("<!-- GSS_MIX_CONFIG_END -->")
    {
        return Err("Markdown GSS recipes have been retired; use a TOML recipe based on configs/gss.example.toml".into());
    }
    parse_toml(document)
}

pub(crate) fn parse_preprocessing_mode(raw: &RawConfig) -> Result<PreprocessingMode, String> {
    let value = raw
        .value("gadgetization_mode")
        .unwrap_or("quadratic-masking");
    if let Some(mode) = PreprocessingMode::parse(value) {
        return Ok(mode);
    }
    let message = match value {
        "product-2223" | "2223" | "nonlinear193" => format!(
            "{value} has been retired from GSS; preprocessing.mode must be quadratic-masking or nonlinear291"
        ),
        other => format!("expected quadratic-masking or nonlinear291; got {other:?}"),
    };
    Err(config_error(raw, "gadgetization_mode", &message))
}

// Canonical TOML name, compatibility spelling, internal recipe key, value type.
// Internal keys remain stable to preserve saved-run manifest identity.
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
        "run.adopt_unverified_run",
        "run.adopt_existing_run",
        "adopt_existing_run",
        "bool",
    ),
    ("source.wires", "source.wires", "n", "integer"),
    ("source.gates", "source.gates", "mcd", "integer"),
    ("source.path", "source.path", "source_path", "string"),
    (
        "database.regular_dir",
        "database.regular_dir",
        "frozen_db_dir",
        "string",
    ),
    (
        "database.curated_dir",
        "database.curated_dir",
        "frozen_curated_dir",
        "string",
    ),
    (
        "database.curated_control_order",
        "database.curated_value_convention",
        "curated_value_convention",
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
        "gadget.mode",
        "gadgetization_mode",
        "string",
    ),
    (
        "preprocessing.mask_pair_wires",
        "gadget.k",
        "bv5_k",
        "integer",
    ),
    (
        "preprocessing.max_open_masks",
        "gadget.max_open",
        "bv5_max_open",
        "integer",
    ),
    (
        "preprocessing.min_open_masks",
        "gadget.min_open",
        "bv5_min_open",
        "integer",
    ),
    (
        "preprocessing.mask_pair_wires",
        "gadget.mask_pair_wires",
        "bv5_k",
        "integer",
    ),
    (
        "preprocessing.max_open_masks",
        "gadget.max_open_masks",
        "bv5_max_open",
        "integer",
    ),
    (
        "preprocessing.min_open_masks",
        "gadget.min_open_masks",
        "bv5_min_open",
        "integer",
    ),
    (
        "preprocessing.balanced_masks",
        "gadget.balanced_masks",
        "bv5_balanced",
        "bool",
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

pub(crate) fn canonical_config_key(key: &str) -> &str {
    TOML_FIELDS
        .iter()
        .find(|field| field.2 == key)
        .map_or(key, |field| field.0)
}

pub(crate) fn parse_toml(document: &str) -> Result<RawConfig, String> {
    let doc = toml_edit::Document::parse(document).map_err(|e| e.to_string())?;
    if doc
        .get("config_version")
        .and_then(toml_edit::Item::as_integer)
        != Some(1)
    {
        return Err("config_version must be the integer 1".into());
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
                .ok_or_else(|| format!("unknown GSS config key {dotted}"))?;
            if raw.entries.contains_key(flat) {
                return Err(format!(
                    "duplicate GSS setting {canonical}; {dotted} and its compatibility alias cannot both be specified"
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
                "gadget",
                "preprocessing",
                "phase_a",
                "db_mixing",
                "parallel",
                "quality_control",
                "leakage_repair",
                "crossing",
                "calibration",
            ]
            .contains(&section)
        {
            return Err(format!("unknown GSS section {section}"));
        }
    }
    Ok(raw)
}
