//! Read TOML and historical Markdown, normalizing aliases before validation.
use super::*;

pub(crate) fn parse_markdown(document: &str) -> Result<RawConfig, String> {
    let lines: Vec<&str> = document.lines().collect();
    let begins: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter_map(|(index, line)| (line.trim() == CONFIG_BEGIN).then_some(index))
        .collect();
    let ends: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter_map(|(index, line)| (line.trim() == CONFIG_END).then_some(index))
        .collect();

    if begins.len() != 1 || ends.len() != 1 {
        return Err(format!(
            "expected exactly one {CONFIG_BEGIN} / {CONFIG_END} pair; found {} begin marker(s) and {} end marker(s)",
            begins.len(),
            ends.len()
        ));
    }
    let (begin, end) = (begins[0], ends[0]);
    if begin >= end {
        return Err(format!("{CONFIG_BEGIN} must occur before {CONFIG_END}"));
    }

    let region = &lines[begin + 1..end];
    let nonblank: Vec<usize> = region
        .iter()
        .enumerate()
        .filter_map(|(index, line)| (!line.trim().is_empty()).then_some(index))
        .collect();
    let Some((&fence_start, rest)) = nonblank.split_first() else {
        return Err(format!("line {}: config block is empty", begin + 2));
    };
    let Some(&fence_end) = rest.last() else {
        return Err(format!(
            "line {}: config block needs an opening ```ini fence and a closing ``` fence",
            begin + fence_start + 2
        ));
    };
    if region[fence_start].trim() != "```ini" {
        return Err(format!(
            "line {}: expected opening ```ini fence",
            begin + fence_start + 2
        ));
    }
    if region[fence_end].trim() != "```" {
        return Err(format!(
            "line {}: expected closing ``` fence",
            begin + fence_end + 2
        ));
    }
    for (index, line) in region.iter().enumerate() {
        if index < fence_start || index > fence_end {
            if !line.trim().is_empty() {
                return Err(format!(
                    "line {}: only the fenced config may appear between the markers",
                    begin + index + 2
                ));
            }
        }
    }

    let mut config = RawConfig {
        legacy_markdown: true,
        ..RawConfig::default()
    };
    for (index, line) in region
        .iter()
        .enumerate()
        .take(fence_end)
        .skip(fence_start + 1)
    {
        let line_number = begin + index + 2;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if trimmed.starts_with("```") {
            return Err(format!(
                "line {line_number}: nested Markdown fences are not allowed in the config"
            ));
        }
        let Some((raw_key, raw_value)) = trimmed.split_once('=') else {
            return Err(format!(
                "line {line_number}: expected `key = value`, got {trimmed:?}"
            ));
        };
        let key = raw_key.trim();
        let value = raw_value.trim();
        if !KNOWN_KEYS.contains(&key) {
            return Err(format!(
                "line {line_number}: unknown GSS config key {key:?}"
            ));
        }
        if config.entries.contains_key(key) {
            let first = config.entries[key].line;
            return Err(format!(
                "line {line_number}: duplicate GSS config key {key:?} (first set on line {first})"
            ));
        }
        if value.chars().any(char::is_control) {
            return Err(format!(
                "line {line_number}: control characters are not allowed in config values"
            ));
        }
        if value.contains(['\'', '"']) {
            return Err(format!(
                "line {line_number}: values are literal and must not be quoted"
            ));
        }
        config.entries.insert(
            key.to_owned(),
            ConfigEntry {
                value: (!value.is_empty()).then(|| value.to_owned()),
                line: line_number,
            },
        );
    }
    Ok(config)
}

/// Read the current TOML recipe, or an explicitly supplied older Markdown recipe.
pub(crate) fn parse_config(document: &str) -> Result<RawConfig, String> {
    if document.contains(CONFIG_BEGIN) || document.contains(CONFIG_END) {
        return parse_markdown(document);
    }
    parse_toml(document)
}

// Canonical TOML name, compatibility spelling, internal recipe key, value type.
// Internal keys remain stable so historical Markdown recipes and manifests keep
// their meaning. Public names are resolved here before value validation.
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
