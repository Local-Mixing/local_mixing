//! Resolve typed values and validate resource/path constraints before run creation.
use super::*;

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
    let frozen_curated = sourced_path(
        raw.value("frozen_curated_dir"),
        "FROZEN_CURATED_DIR",
        repo_root,
        &getenv,
    )?;
    validate_path_separation(
        &run_dir,
        &build_target_dir,
        frozen_db.path.as_deref(),
        frozen_curated.path.as_deref(),
    )?;

    let curated_value_convention = parse_enum(
        raw,
        "curated_value_convention",
        "native",
        &["native", "legacy-swapped-controls"],
    )?;
    let preprocessing_mode = parse_gadgetization_mode(raw)?;
    let production_preset = parse_enum(raw, "production_preset", "production", PRODUCTION_PRESETS)?;
    let post_fragment =
        parse_optional_enum(raw, "post_fragment", &["off", "exact", "native-deep"])?;
    if preprocessing_mode != RecipePreprocessingMode::Product2223 {
        if raw.value("production_preset").is_some() {
            return Err(config_error(
                raw,
                "production_preset",
                "production_preset applies only to `gadgetization_mode = product-2223`; leave it blank for nonlinear modes",
            ));
        }
        if raw.value("post_fragment").is_some() {
            return Err(config_error(
                raw,
                "post_fragment",
                "post_fragment applies only to `gadgetization_mode = product-2223`; leave it blank for nonlinear modes",
            ));
        }
    }
    let calibration_only = parse_bool(raw, "calibration_only", false)?;
    let calibration_seed_file = raw
        .value("calibration_seed_file")
        .map(|value| resolve_path(repo_root, Path::new(value)));
    if calibration_seed_file.is_some() && !calibration_only {
        return Err(config_error(
            raw,
            "calibration_seed_file",
            if raw.legacy_markdown {
                "requires `calibration_only = true`; explicit seeds must never be used for deliverables"
            } else {
                "requires `calibration.enabled = true`; explicit seeds must never be used for deliverables"
            },
        ));
    }
    let calibration_seed = calibration_seed_file
        .as_ref()
        .map(|path| read_calibration_seed(path))
        .transpose()?;

    let mut mcd = parse_optional_u64(raw, "mcd", 1, 1_000_000_000)?;
    let source_path = raw
        .value("source_path")
        .map(|value| resolve_path(repo_root, Path::new(value)));
    let mut source_hash = None;
    if let Some(path) = &source_path {
        let bytes =
            fs::read(path).map_err(|e| format!("cannot read source {}: {e}", path.display()))?;
        if bytes.starts_with(b"mpmct1") || bytes.starts_with(b"esop1") || bytes.starts_with(b"anf1")
        {
            return Err("GSS source.path currently requires a G57 circuit; packed/general formats are supported by circuit evaluate/compare".into());
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
        source_hash = Some(xxhash_rust::xxh3::xxh3_128(&bytes));
    }
    // Old recipes may record K=1 (the generator clamped it to 2). Keep reading
    // them; current-run validation rejects that misleading setting.
    let bv5_k = parse_usize(raw, "bv5_k", 2, 1, 64)?;
    let bv5_max_open = parse_usize(raw, "bv5_max_open", 3, 2, 64)?;
    let bv5_min_open = parse_usize(raw, "bv5_min_open", 2, 1, 63)?;
    if bv5_min_open >= bv5_max_open {
        return Err(
            "preprocessing.min_open_masks must be less than preprocessing.max_open_masks".into(),
        );
    }
    let bv5_balanced = parse_bool(raw, "bv5_balanced", true)?;
    let explicit_mask_controls = ["bv5_k", "bv5_max_open", "bv5_min_open", "bv5_balanced"]
        .iter()
        .any(|key| raw.value(key).is_some());
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
    let qc_reference_hash = qc_reference
        .as_ref()
        .map(|path| {
            fs::read(path)
                .map(|bytes| xxhash_rust::xxh3::xxh3_128(&bytes))
                .map_err(|e| {
                    format!(
                        "cannot read leakage-repair reference {}: {e}",
                        path.display()
                    )
                })
        })
        .transpose()?;
    if let Some(mode) = preprocessing_mode.nonlinear() {
        let m = mcd
            .as_deref()
            .map(|value| value.parse::<usize>().expect("validated mcd"))
            .unwrap_or_else(|| sandwich_default_m(n));
        let s = sandwich_default_s(n);
        let gate_count = m
            .checked_mul(2)
            .and_then(|count| count.checked_add(s.checked_mul(2)?))
            .and_then(|count| count.checked_add(n))
            .ok_or_else(|| {
                config_error(raw, "gadgetization_mode", "sandwich gate-count overflow")
            })?;
        let logical_n = n.checked_mul(2).ok_or_else(|| {
            config_error(raw, "gadgetization_mode", "logical wire-count overflow")
        })?;
        let slice_gates = logical_n
            .checked_mul(10)
            .ok_or_else(|| config_error(raw, "gadgetization_mode", "slice gate-count overflow"))?;
        nonlinear_gss_resource_plan(logical_n, gate_count, slice_gates, mode).map_err(|error| {
            config_error(
                raw,
                "gadgetization_mode",
                &format!(
                    "requested nonlinear layout exceeds the current wire capacity/resource budget: {error}; reduce n or set a smaller mcd"
                ),
            )
        })?;
    }
    let expand = parse_optional_f64(
        raw,
        "expand",
        |value| value > 1.0 && value <= 16.0,
        "must be greater than 1 and at most 16",
    )?;
    let hold = parse_optional_f64(
        raw,
        "hold",
        |value| (0.0..=10_000.0).contains(&value),
        "must be in 0..=10000",
    )?;
    // The shell's hold is 27 (profile 3,30,30). Preserve v3's historical
    // omitted representation only when reading an old Markdown recipe.
    let hold = if raw.legacy_markdown {
        hold
    } else {
        Some(hold.unwrap_or_else(|| "27".into()))
    };
    let xr = parse_optional_f64(
        raw,
        "xr",
        |value| (1.0..=16.0).contains(&value),
        "must be in 1..=16",
    )?;
    let xb = parse_optional_f64(
        raw,
        "xb",
        |value| (1.0..=1_000_000.0).contains(&value),
        "must be in 1..=1000000",
    )?;
    let xc = parse_optional_u64(raw, "xc", 0, 1_000_000_000)?;
    let xtdiv = parse_optional_u64(raw, "xtdiv", 1, 1_000_000_000)?;
    let xmoves = parse_optional_u64(raw, "xmoves", 1, 1_000_000_000_000)?;
    let stop_after = parse_optional_u64(raw, "stop_after", 2, 6)?;
    let force_from = parse_optional_u64(raw, "force_from", 2, 6)?;
    let pieces = parse_optional_u64(raw, "pieces", 1, 64)?;
    let min_block_size = parse_optional_u64(raw, "min_block_size", 2, 1_000_000_000)?;
    if pieces.is_some() && min_block_size.is_some() {
        return Err(config_error(
            raw,
            "min_block_size",
            if raw.legacy_markdown {
                "is mutually exclusive with `pieces` (including pieces = 1)"
            } else {
                "is mutually exclusive with `parallel.pieces` (including parallel.pieces = 1)"
            },
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
            if raw.legacy_markdown {
                "applies only when `pieces` is greater than 1 or `min_block_size` is set"
            } else {
                "applies only when `parallel.pieces` is greater than 1 or `parallel.target_piece_gates` is set"
            },
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
        recipe_version: if raw.legacy_markdown { 3 } else { 7 },
        source_path,
        source_hash,
        bv5_k,
        bv5_max_open,
        bv5_min_open,
        bv5_balanced,
        explicit_mask_controls,
        qc_enabled,
        qc_seed,
        qc_reference,
        qc_reference_hash,
        n,
        run_dir,
        generated_run_dir: run_dir_value.is_none(),
        build_release,
        build_target_dir,
        adopt_existing_run,
        frozen_db,
        frozen_curated,
        curated_value_convention,
        preprocessing_mode,
        production_preset,
        post_fragment,
        calibration_only,
        calibration_seed,
        mcd,
        expand,
        hold,
        xr,
        xb,
        xc,
        xtdiv,
        xmoves,
        stop_after,
        force_from,
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

pub(crate) fn parse_enum(
    raw: &RawConfig,
    key: &str,
    default: &str,
    allowed: &[&str],
) -> Result<String, String> {
    let value = raw.value(key).unwrap_or(default);
    if !allowed.contains(&value) {
        return Err(config_error(
            raw,
            key,
            &format!("expected one of {}; got {value:?}", allowed.join(", ")),
        ));
    }
    Ok(value.to_owned())
}

pub(crate) fn parse_optional_enum(
    raw: &RawConfig,
    key: &str,
    allowed: &[&str],
) -> Result<Option<String>, String> {
    let Some(value) = raw.value(key) else {
        return Ok(None);
    };
    if !allowed.contains(&value) {
        return Err(config_error(
            raw,
            key,
            &format!("expected one of {}; got {value:?}", allowed.join(", ")),
        ));
    }
    Ok(Some(value.to_owned()))
}

pub(crate) fn config_error(raw: &RawConfig, key: &str, message: &str) -> String {
    let display_key = if raw.legacy_markdown {
        key
    } else {
        canonical_config_key(key)
    };
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

pub(crate) fn read_calibration_seed(path: &Path) -> Result<String, String> {
    let metadata = fs::metadata(path).map_err(|error| {
        format!(
            "calibration_seed_file {} cannot be read: {error}",
            path.display()
        )
    })?;
    if !metadata.is_file() {
        return Err(format!(
            "calibration_seed_file {} is not a regular file",
            path.display()
        ));
    }
    if metadata.len() > MAX_SEED_FILE_BYTES {
        return Err(format!(
            "calibration_seed_file {} is too large (maximum {MAX_SEED_FILE_BYTES} bytes)",
            path.display()
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o077 != 0 {
            return Err(format!(
                "calibration_seed_file {} must not be accessible by group/other (use chmod 600)",
                path.display()
            ));
        }
    }
    let contents = fs::read_to_string(path).map_err(|error| {
        format!(
            "calibration_seed_file {} cannot be read: {error}",
            path.display()
        )
    })?;
    let seed = contents.trim();
    if seed.is_empty() || seed.chars().any(|ch| !ch.is_ascii_digit()) {
        return Err(format!(
            "calibration_seed_file {} must contain one unsigned decimal integer",
            path.display()
        ));
    }
    let parsed = seed.parse::<u64>().map_err(|_| {
        format!(
            "calibration_seed_file {} contains an out-of-range integer",
            path.display()
        )
    })?;
    let max = i64::MAX as u64 - 15;
    if parsed > max {
        return Err(format!(
            "calibration seed is too large for Bash stage-seed arithmetic (maximum {max})"
        ));
    }
    Ok(parsed.to_string())
}
