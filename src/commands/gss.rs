//! `local_mixing_bin gss`: safe front end for the supported GSS_MIX driver.
//!
//! Pipeline ordering and stage calculations remain in `scripts/gss_mix.sh`.
//! This module owns only the operator-facing Markdown configuration, strict
//! validation, release-binary preparation, and child-process execution.

use clap::{Arg, ArgAction, ArgMatches, Command};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{OsStr, OsString};
use std::fs;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{Command as ProcessCommand, ExitStatus};
use std::time::{SystemTime, UNIX_EPOCH};
use xxhash_rust::xxh3::Xxh3;

const CONFIG_BEGIN: &str = "<!-- GSS_MIX_CONFIG_BEGIN -->";
const CONFIG_END: &str = "<!-- GSS_MIX_CONFIG_END -->";
const MAX_SEED_FILE_BYTES: u64 = 64;
const MAX_MANIFEST_BYTES: u64 = 64 * 1024;

const KNOWN_KEYS: &[&str] = &[
    "n",
    "run_dir",
    "build_release",
    "build_target_dir",
    "adopt_existing_run",
    "frozen_db_dir",
    "frozen_curated_dir",
    "curated_value_convention",
    "production_preset",
    "post_fragment",
    "calibration_only",
    "calibration_seed_file",
    "mcd",
    "expand",
    "hold",
    "xr",
    "xb",
    "xc",
    "xtdiv",
    "xmoves",
    "stop_after",
    "force_from",
    "allow_empty_store",
    "frozen_filter",
];

const PRODUCTION_PRESETS: &[&str] = &[
    "production",
    "no-gray-phase-a",
    "micro-gray",
    "sentinel-gray",
    "no-gray-post-exact",
    "no-gray-post-native",
    "five-carrier",
    "strong-five-carrier",
    "six-carrier",
    "strong-six-carrier",
    "seven-carrier",
];

/// Low-level research/debug switches are intentionally outside the supported
/// GSS recipe. The direct Bash interface may still use them; the `gss`
/// command removes them so the Markdown block is its complete control surface.
const SCRUBBED_ENV: &[&str] = &[
    "PROD_K",
    "PROD_DEG",
    "PROD_K_HI",
    "PROD_DEG_HI",
    "PROD_BAND",
    "PROD_RSRC",
    "PROD_MAX_WIDTH",
    "PROD_FILL_NL",
    "PROD_ROLL",
    "PROD_SRC_DIST",
    "PROD_SRC_HORIZON",
    "PROD_SRC_LO",
    "PROD_SRC_HI",
    "PROD_FILL_PIVOTS",
    "PROD_G57_NARROW",
    "PROD_LADDER_CAP",
    "PROD_CG_JITTER",
    "PROD_RUNG_MENU",
    "PROD_EPOCH",
    "PROD_REFILL_DATA",
    "PROD_SINGLE",
    "PROD_GRAY_FOLD",
    "PROD_DISJOINT",
    "PROD_BARE_CENSUS",
    "PROD_DUMP_PAIRS",
    "ABSORB_NOTS",
    "BENCH_CANON",
    "CENTRALIZE",
    "COMPRESSION_TRACE",
    "COMPRESSION_TRACE_MS",
    "COMPRESS_CHUNK_BUDGET_MS",
    "COMPRESS_STALL_FRAC",
    "COMPRESS_STALL_WINDOW",
    "DEGREE_FILTER",
    "DEGREE_FILTER_PROBES",
    "FLOAT_SWEEP",
    "FMIX_DUMP_OUT",
    "LITTER_RULES",
    "LITTER_WINDOW_SAMPLES",
    "MIN_DIR_LOOKUP",
    "SAMF_HIDE_PAIRS",
    "SAT_BCP_MIN_RESISTANCE",
    "SAT_BCP",
    "SAT_BCP_ASSIGN_PROB",
    "SAT_BCP_OUTPUT_BITS",
    "SAT_BCP_TRIALS",
    "SAT_COMPRESS_PROTECT",
    "SAT_COMPRESS_PRESERVE_DELTA",
    "SAT_CONE_AWARE",
    "SAT_CONE_MIN_FRACTION",
    "SAT_EXPAND_MIN_DELTA",
    "SAT_HARDEN",
    "SAT_HIDDEN_SAMF_CANDIDATES",
    "SAT_PROBE",
    "SAT_PROBE_FREQUENCY",
    "SAT_PROBE_SOLVER",
    "SAT_PROBE_TARGET_BITS",
    "SAT_PROBE_TIMEOUT_MS",
    "SAT_PROBE_WINDOW_GATES",
    "SAT_SCORE",
    "SAT_SCORE_SEED",
    "SAT_SCORE_SLACK",
    "SHOOT_PARALLEL",
    "SHOOT_PROFILE",
    "SLOW_COMPRESS",
    "SLOW_COMPRESS_MOVE_STALL",
    "STABLE_MAX",
    "STAGEC_CHECK",
    "SURVIVOR_LOG_EVERY",
    "TWIST_G57_NO_RETRY",
    "TWIST_G57_NO_SLIDE",
    "VERIFY_DB_HITS",
    "SymmetricCD",
    "SymmetricG",
    "BASH_ENV",
    "ENV",
    "SHELLOPTS",
    "BASHOPTS",
    "BASH_XTRACEFD",
    "PS4",
    "CDPATH",
    "GLOBIGNORE",
    "PYTHONPATH",
    "PYTHONHOME",
];

#[derive(Clone, Debug)]
struct ConfigEntry {
    value: Option<String>,
    line: usize,
}

#[derive(Clone, Debug, Default)]
struct RawConfig {
    entries: BTreeMap<String, ConfigEntry>,
}

impl RawConfig {
    fn value(&self, key: &str) -> Option<&str> {
        self.entries
            .get(key)
            .and_then(|entry| entry.value.as_deref())
    }

    fn line(&self, key: &str) -> Option<usize> {
        self.entries.get(key).map(|entry| entry.line)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ValueSource {
    Document,
    Environment,
    Unset,
}

impl ValueSource {
    fn label(self) -> &'static str {
        match self {
            Self::Document => "GSS_MIX.md",
            Self::Environment => "environment",
            Self::Unset => "unset",
        }
    }
}

#[derive(Clone, Debug)]
struct SourcedPath {
    path: Option<PathBuf>,
    source: ValueSource,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FrozenFilter {
    Auto,
    On,
    Off,
}

/// Deliberately does not derive `Debug`: `calibration_seed` is secret.
#[derive(Clone)]
struct ResolvedConfig {
    n: usize,
    run_dir: PathBuf,
    generated_run_dir: bool,
    build_release: bool,
    build_target_dir: PathBuf,
    adopt_existing_run: bool,
    frozen_db: SourcedPath,
    frozen_curated: SourcedPath,
    curated_value_convention: String,
    production_preset: String,
    post_fragment: Option<String>,
    calibration_only: bool,
    calibration_seed: Option<String>,
    mcd: Option<String>,
    expand: Option<String>,
    hold: Option<String>,
    xr: Option<String>,
    xb: Option<String>,
    xc: Option<String>,
    xtdiv: Option<String>,
    xmoves: Option<String>,
    stop_after: Option<String>,
    force_from: Option<String>,
    allow_empty_store: bool,
    frozen_filter: FrozenFilter,
}

#[derive(Debug)]
struct GssError {
    message: String,
    exit_code: i32,
}

impl GssError {
    fn config(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            exit_code: 2,
        }
    }

    fn io(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            exit_code: 1,
        }
    }

    fn child(label: &str, status: ExitStatus) -> Self {
        let exit_code = status.code().unwrap_or(1);
        Self {
            message: format!("{label} exited with status {status}"),
            exit_code,
        }
    }
}

pub fn command() -> Command {
    Command::new("gss")
        .visible_alias("gss-mix")
        .about("Run the supported GSS_MIX pipeline from the config block in docs/GSS_MIX.md")
        .arg(
            Arg::new("config")
                .long("config")
                .value_name("PATH")
                .value_parser(clap::value_parser!(PathBuf))
                .help("Markdown file containing the marked GSS_MIX config block (default: docs/GSS_MIX.md)"),
        )
        .arg(
            Arg::new("dry_run")
                .long("dry-run")
                .action(ArgAction::SetTrue)
                .help("Resolve and validate the config without building or running the pipeline"),
        )
}

pub fn run(sub: &ArgMatches) {
    if let Err(error) = run_inner(sub) {
        eprintln!("[gss] FATAL: {}", error.message);
        std::process::exit(error.exit_code);
    }
}

fn run_inner(sub: &ArgMatches) -> Result<(), GssError> {
    let repo_root = find_repo_root()?;
    let config_path = sub
        .get_one::<PathBuf>("config")
        .map(|path| resolve_path(&repo_root, path))
        .unwrap_or_else(|| repo_root.join("docs/GSS_MIX.md"));
    let document = fs::read_to_string(&config_path).map_err(|error| {
        GssError::io(format!(
            "cannot read configuration {}: {error}",
            config_path.display()
        ))
    })?;
    let raw = parse_config(&document).map_err(GssError::config)?;
    let run_tag = default_run_tag();
    let config = resolve_config(&raw, &repo_root, |key| std::env::var_os(key), &run_tag)
        .map_err(GssError::config)?;
    let host_target = rustc_host_triple()?;
    let binary_dir = config.build_target_dir.join(&host_target).join("release");

    print_resolved(&config_path, &config, &binary_dir);
    validate_external_paths(&config).map_err(GssError::config)?;

    let manifest_path = config.run_dir.join("gss_command.conf");
    let managed_resume = manifest_path.is_file();
    let legacy_resume = fs::metadata(config.run_dir.join("SEED"))
        .is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0)
        && !managed_resume;
    if legacy_resume && !config.adopt_existing_run {
        return Err(GssError::config(format!(
            "{} is a pre-wrapper run with no gss_command.conf; set `adopt_existing_run = true` once to acknowledge that its earlier recipe cannot be verified, or use the direct Bash driver",
            config.run_dir.display()
        )));
    }
    if legacy_resume && !config.calibration_only {
        return Err(GssError::config(format!(
            "{} has unverifiable pre-wrapper seed provenance; adopting it requires both `adopt_existing_run = true` and `calibration_only = true`",
            config.run_dir.display()
        )));
    }
    if managed_resume {
        let fingerprints = production_binary_fingerprints(&binary_dir)?;
        let script_hash = hash_file(&repo_root.join("scripts/gss_mix.sh")).map_err(|error| {
            GssError::io(format!("cannot fingerprint scripts/gss_mix.sh: {error}"))
        })?;
        let desired = recipe_manifest(&config, &fingerprints, script_hash);
        compare_recipe_manifest(&manifest_path, &desired)?;
    } else if !config.build_release {
        production_binary_fingerprints(&binary_dir)?;
    }

    let script = repo_root.join("scripts/gss_mix.sh");
    let script_args = script_args(&config);
    let should_build = config.build_release && !managed_resume;
    if sub.get_flag("dry_run") {
        print_dry_run(
            &script,
            &script_args,
            &config,
            &host_target,
            should_build,
            managed_resume,
        );
        return Ok(());
    }

    if config.build_release && managed_resume {
        println!(
            "[gss] release build: skipped for resume; using the exact binaries fingerprinted by the run manifest"
        );
    }
    if should_build {
        println!("[gss] building the three GSS_MIX release executables (incremental)");
        let cargo = std::env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"));
        let status = ProcessCommand::new(cargo)
            .current_dir(&repo_root)
            .arg("build")
            .arg("--release")
            .arg("--locked")
            .arg("--target-dir")
            .arg(&config.build_target_dir)
            .arg("--target")
            .arg(&host_target)
            .arg("--bin")
            .arg("gen_sandwich_gadget")
            .arg("--bin")
            .arg("fmix")
            .arg("--bin")
            .arg("fcompress")
            .env_remove("CARGO_BUILD_TARGET")
            .status()
            .map_err(|error| GssError::io(format!("cannot start cargo build: {error}")))?;
        if !status.success() {
            return Err(GssError::child("cargo build", status));
        }
    }

    if !managed_resume {
        prepare_recipe_manifest(&config, &binary_dir, &script)?;
    }

    println!("[gss] starting scripts/gss_mix.sh");
    let mut child = ProcessCommand::new("bash");
    child
        .current_dir(&repo_root)
        .arg(&script)
        .args(&script_args);
    configure_environment(&mut child, &config, &binary_dir);
    let status = child
        .status()
        .map_err(|error| GssError::io(format!("cannot start bash: {error}")))?;
    if !status.success() {
        return Err(GssError::child("GSS_MIX pipeline", status));
    }
    Ok(())
}

fn parse_config(document: &str) -> Result<RawConfig, String> {
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

    let mut config = RawConfig::default();
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

fn resolve_config<F>(
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
    let production_preset = parse_enum(raw, "production_preset", "production", PRODUCTION_PRESETS)?;
    let post_fragment =
        parse_optional_enum(raw, "post_fragment", &["off", "exact", "native-deep"])?;
    let calibration_only = parse_bool(raw, "calibration_only", false)?;
    let calibration_seed_file = raw
        .value("calibration_seed_file")
        .map(|value| resolve_path(repo_root, Path::new(value)));
    if calibration_seed_file.is_some() && !calibration_only {
        return Err(config_error(
            raw,
            "calibration_seed_file",
            "requires `calibration_only = true`; explicit seeds must never be used for deliverables",
        ));
    }
    let calibration_seed = calibration_seed_file
        .as_ref()
        .map(|path| read_calibration_seed(path))
        .transpose()?;

    let mcd = parse_optional_u64(raw, "mcd", 1, 1_000_000_000)?;
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
        n,
        run_dir,
        generated_run_dir: run_dir_value.is_none(),
        build_release,
        build_target_dir,
        adopt_existing_run,
        frozen_db,
        frozen_curated,
        curated_value_convention,
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
    })
}

fn parse_usize(
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

fn parse_optional_u64(
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

fn parse_optional_f64(
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

fn parse_bool(raw: &RawConfig, key: &str, default: bool) -> Result<bool, String> {
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

fn parse_enum(
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

fn parse_optional_enum(
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

fn config_error(raw: &RawConfig, key: &str, message: &str) -> String {
    match raw.line(key) {
        Some(line) => format!("line {line} ({key}): {message}"),
        None => format!("{key}: {message}"),
    }
}

fn sourced_path<F>(
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

fn canonicalize_if_present(path: PathBuf) -> Result<PathBuf, String> {
    if !path.exists() {
        return Ok(path);
    }
    path.canonicalize()
        .map_err(|error| format!("cannot canonicalize {}: {error}", path.display()))
}

fn read_calibration_seed(path: &Path) -> Result<String, String> {
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

fn validate_external_paths(config: &ResolvedConfig) -> Result<(), String> {
    if config.run_dir.exists() && !config.run_dir.is_dir() {
        return Err(format!(
            "run_dir {} exists but is not a directory",
            config.run_dir.display()
        ));
    }
    let stop_after = config
        .stop_after
        .as_deref()
        .unwrap_or("6")
        .parse::<u64>()
        .expect("validated stop_after");
    let force_from = config
        .force_from
        .as_deref()
        .unwrap_or("99")
        .parse::<u64>()
        .expect("validated force_from");
    let existing_phase_a = fs::metadata(config.run_dir.join("phaseA.mpmct1"))
        .is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0);
    let stage_three_will_run = stop_after >= 3 && (force_from <= 3 || !existing_phase_a);
    if stage_three_will_run {
        if let Some(path) = &config.frozen_db.path {
            validate_frozen_store(path, "frozen_db_dir")?;
            validate_requested_filter(path, "frozen_db_dir", config.frozen_filter)?;
        }
        if let Some(path) = &config.frozen_curated.path {
            validate_frozen_store(path, "frozen_curated_dir")?;
            validate_requested_filter(path, "frozen_curated_dir", config.frozen_filter)?;
        }
    }
    if stage_three_will_run && config.allow_empty_store && config.frozen_db.path.is_some() {
        return Err(
            "allow_empty_store=true conflicts with a configured frozen_db_dir; clear one so the Phase-A mode is unambiguous"
                .to_owned(),
        );
    }
    if stage_three_will_run
        && config.frozen_db.path.is_none()
        && config.frozen_curated.path.is_some()
    {
        return Err(
            "frozen_curated_dir cannot be used without frozen_db_dir; the curated store is a cascade ahead of the regular store"
                .to_owned(),
        );
    }
    if stage_three_will_run && config.frozen_db.path.is_none() && !config.allow_empty_store {
        return Err(
            "stage 3 needs frozen_db_dir (or FROZEN_DB_DIR in the environment); `allow_empty_store = true` is only for plumbing tests"
                .to_owned(),
        );
    }
    Ok(())
}

fn validate_requested_filter(
    path: &Path,
    label: &str,
    selection: FrozenFilter,
) -> Result<(), String> {
    if selection != FrozenFilter::On {
        return Ok(());
    }
    let filter = path.join("filters.bin");
    if !fs::metadata(&filter).is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0) {
        return Err(format!(
            "frozen_filter=on requires a nonempty {}/filters.bin for {label}",
            path.display()
        ));
    }
    Ok(())
}

fn validate_frozen_store(path: &Path, label: &str) -> Result<(), String> {
    if !path.is_dir() {
        return Err(format!(
            "{label} {} is not an existing directory",
            path.display()
        ));
    }
    let tables = path.join("tables.bin");
    if !fs::metadata(&tables).is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0) {
        return Err(format!(
            "{label} {} is not a frozen store: tables.bin is missing or empty",
            path.display()
        ));
    }
    for shard in 0u16..=255 {
        let shard_path = path.join(format!("shard_{shard:02x}.frz"));
        if !fs::metadata(&shard_path).is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0)
        {
            return Err(format!(
                "{label} {} is incomplete: {} is missing or empty",
                path.display(),
                shard_path.file_name().unwrap_or_default().to_string_lossy()
            ));
        }
    }
    Ok(())
}

fn prepare_recipe_manifest(
    config: &ResolvedConfig,
    binary_dir: &Path,
    script: &Path,
) -> Result<(), GssError> {
    let fingerprints = production_binary_fingerprints(binary_dir)?;
    let script_hash = hash_file(script).map_err(|error| {
        GssError::io(format!(
            "cannot fingerprint orchestrator {}: {error}",
            script.display()
        ))
    })?;
    let mut desired = recipe_manifest(config, &fingerprints, script_hash);
    fs::create_dir_all(&config.run_dir).map_err(|error| {
        GssError::io(format!(
            "cannot create run_dir {}: {error}",
            config.run_dir.display()
        ))
    })?;
    let path = config.run_dir.join("gss_command.conf");
    if path.exists() {
        return compare_recipe_manifest(&path, &desired);
    }

    if fs::metadata(config.run_dir.join("SEED"))
        .is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0)
    {
        eprintln!(
            "[gss] WARNING: adopting an existing pre-wrapper run; its earlier recipe and binary identities cannot be independently verified"
        );
        desired.push_str("adopted_unverified=true\n");
    }
    match OpenOptions::new().write(true).create_new(true).open(&path) {
        Ok(mut file) => {
            file.write_all(desired.as_bytes()).map_err(|error| {
                GssError::io(format!(
                    "cannot write recipe manifest {}: {error}",
                    path.display()
                ))
            })?;
            file.sync_all().map_err(|error| {
                GssError::io(format!(
                    "cannot sync recipe manifest {}: {error}",
                    path.display()
                ))
            })?;
            println!("[gss] recipe manifest: {}", path.display());
            Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            compare_recipe_manifest(&path, &desired)
        }
        Err(error) => Err(GssError::io(format!(
            "cannot create recipe manifest {}: {error}",
            path.display()
        ))),
    }
}

fn production_binary_fingerprints(
    binary_dir: &Path,
) -> Result<Vec<(&'static str, u128)>, GssError> {
    ["gen_sandwich_gadget", "fmix", "fcompress"]
        .into_iter()
        .map(|name| {
            let path = binary_dir.join(name);
            hash_file(&path)
                .map(|hash| (name, hash))
                .map_err(|error| {
                    GssError::io(format!(
                        "cannot fingerprint production binary {}: {error}; set build_release=true or select the target directory containing all three binaries",
                        path.display()
                    ))
                })
        })
        .collect()
}

fn hash_file(path: &Path) -> std::io::Result<u128> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Xxh3::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            return Ok(hasher.digest128());
        }
        hasher.update(&buffer[..read]);
    }
}

fn compare_recipe_manifest(path: &Path, desired: &str) -> Result<(), GssError> {
    let metadata = fs::metadata(path).map_err(|error| {
        GssError::io(format!(
            "cannot inspect recipe manifest {}: {error}",
            path.display()
        ))
    })?;
    if !metadata.is_file() || metadata.len() > MAX_MANIFEST_BYTES {
        return Err(GssError::config(format!(
            "recipe manifest {} must be a regular file no larger than {MAX_MANIFEST_BYTES} bytes",
            path.display()
        )));
    }
    let existing = fs::read_to_string(path).map_err(|error| {
        GssError::io(format!(
            "cannot read recipe manifest {}: {error}",
            path.display()
        ))
    })?;
    let adopted_unverified = existing
        .lines()
        .any(|line| line == "adopted_unverified=true");
    let normalized_existing: String = existing
        .lines()
        .filter(|line| *line != "adopted_unverified=true")
        .map(|line| format!("{line}\n"))
        .collect();
    let normalized_desired: String = desired
        .lines()
        .filter(|line| *line != "adopted_unverified=true")
        .map(|line| format!("{line}\n"))
        .collect();
    if normalized_existing == normalized_desired {
        println!("[gss] recipe manifest matches: {}", path.display());
        if adopted_unverified {
            eprintln!(
                "[gss] WARNING: this run was adopted without a verifiable pre-wrapper recipe"
            );
        }
        return Ok(());
    }
    let old: BTreeMap<_, _> = normalized_existing
        .lines()
        .filter_map(|line| line.split_once('='))
        .collect();
    let new: BTreeMap<_, _> = normalized_desired
        .lines()
        .filter_map(|line| line.split_once('='))
        .collect();
    let keys: BTreeSet<_> = old.keys().chain(new.keys()).copied().collect();
    let differences: Vec<String> = keys
        .into_iter()
        .filter(|key| old.get(key) != new.get(key))
        .map(|key| format!("{key}: {:?} -> {:?}", old.get(key), new.get(key)))
        .take(4)
        .collect();
    Err(GssError::config(format!(
        "resolved recipe or production binaries differ from {}; refusing a mixed-provenance resume{}",
        path.display(),
        if differences.is_empty() {
            String::new()
        } else {
            format!(": {}", differences.join(", "))
        }
    )))
}

fn recipe_manifest(
    config: &ResolvedConfig,
    binaries: &[(&str, u128)],
    script_hash: u128,
) -> String {
    let path_or_unset = |path: &Option<PathBuf>| {
        path.as_deref()
            .map(|path| path.to_string_lossy().into_owned())
            .unwrap_or_else(|| "unset".to_owned())
    };
    let mut manifest = format!(
        concat!(
            "gss_command_recipe=1\n",
            "n={}\n",
            "frozen_db_dir={}\n",
            "frozen_curated_dir={}\n",
            "curated_value_convention={}\n",
            "production_preset={}\n",
            "post_fragment={}\n",
            "mcd={}\n",
            "expand={}\n",
            "hold={}\n",
            "xr={}\n",
            "xb={}\n",
            "xc={}\n",
            "xtdiv={}\n",
            "xmoves={}\n",
            "allow_empty_store={}\n",
            "calibration_only={}\n",
            "pinned_recipe=canon512/200000-cache256/1024/2048-native-regular\n",
        ),
        config.n,
        path_or_unset(&config.frozen_db.path),
        path_or_unset(&config.frozen_curated.path),
        config.curated_value_convention,
        config.production_preset,
        config.post_fragment.as_deref().unwrap_or("preset-default"),
        config.mcd.as_deref().unwrap_or("derived"),
        config.expand.as_deref().unwrap_or("2"),
        config.hold.as_deref().unwrap_or("30"),
        config.xr.as_deref().unwrap_or("2"),
        config.xb.as_deref().unwrap_or("3"),
        config.xc.as_deref().unwrap_or("1"),
        config.xtdiv.as_deref().unwrap_or("25"),
        config.xmoves.as_deref().unwrap_or("6*target"),
        config.allow_empty_store,
        config.calibration_only,
    );
    for (name, fingerprint) in binaries {
        use std::fmt::Write as _;
        let _ = writeln!(manifest, "binary_{name}_xxh3={fingerprint:032x}");
    }
    use std::fmt::Write as _;
    let _ = writeln!(manifest, "script_gss_mix_xxh3={script_hash:032x}");
    manifest
}

fn script_args(config: &ResolvedConfig) -> Vec<OsString> {
    let mut args = vec![
        OsString::from("-n"),
        OsString::from(config.n.to_string()),
        OsString::from("-o"),
        config.run_dir.as_os_str().to_owned(),
    ];
    push_optional_arg(&mut args, "--mcd", config.mcd.as_deref());
    push_optional_arg(&mut args, "--expand", config.expand.as_deref());
    push_optional_arg(&mut args, "--hold", config.hold.as_deref());
    push_optional_arg(&mut args, "--xr", config.xr.as_deref());
    push_optional_arg(&mut args, "--xb", config.xb.as_deref());
    push_optional_arg(&mut args, "--xc", config.xc.as_deref());
    push_optional_arg(&mut args, "--xtdiv", config.xtdiv.as_deref());
    push_optional_arg(&mut args, "--xmoves", config.xmoves.as_deref());
    push_optional_arg(&mut args, "--stop-after", config.stop_after.as_deref());
    push_optional_arg(&mut args, "--force-from", config.force_from.as_deref());
    if let Some(seed) = &config.calibration_seed {
        args.push(OsString::from("-s"));
        args.push(OsString::from(seed));
    }
    args
}

fn push_optional_arg(args: &mut Vec<OsString>, flag: &str, value: Option<&str>) {
    if let Some(value) = value {
        args.push(OsString::from(flag));
        args.push(OsString::from(value));
    }
}

fn configure_environment(command: &mut ProcessCommand, config: &ResolvedConfig, binary_dir: &Path) {
    // Future generator experiments should not silently become supported merely
    // by adding another PROD_* read. Clear the whole namespace first, then set
    // only the two coherent controls represented in the Markdown schema.
    for (key, _) in std::env::vars_os() {
        let key_text = key.to_string_lossy();
        if key_text.starts_with("PROD_") || key_text.starts_with("SAT_") {
            command.env_remove(key);
        }
    }
    for key in SCRUBBED_ENV {
        command.env_remove(key);
    }
    command.env("GSS_BIN_DIR", binary_dir);
    set_or_remove_env(command, "FROZEN_DB_DIR", config.frozen_db.path.as_deref());
    set_or_remove_env(
        command,
        "FROZEN_CURATED_DIR",
        config.frozen_curated.path.as_deref(),
    );
    command.env(
        "FROZEN_CURATED_VALUE_CONVENTION",
        &config.curated_value_convention,
    );
    command.env("FROZEN_REGULAR_VALUE_CONVENTION", "native");
    command.env("CANON_RULE_L_BRANCH_CAP", "512");
    command.env("CANON_MONOMIAL_CAP", "200000");
    command.env("CANON_CACHE_MB", "256");
    command.env("XPOLY_CANON_CACHE_MB", "1024");
    command.env("LOOKUP_CACHE_MB", "2048");
    command.env("PROD_PRESET", &config.production_preset);
    match &config.post_fragment {
        Some(value) => {
            command.env("PROD_POST_FRAGMENT", value);
        }
        None => {
            command.env_remove("PROD_POST_FRAGMENT");
        }
    }
    if config.allow_empty_store {
        command.env("GSS_MIX_ALLOW_EMPTY_STORE", "1");
    } else {
        command.env_remove("GSS_MIX_ALLOW_EMPTY_STORE");
    }
    match config.frozen_filter {
        FrozenFilter::Auto => {
            command.env_remove("FROZEN_FILTER");
        }
        FrozenFilter::On => {
            command.env("FROZEN_FILTER", "1");
        }
        FrozenFilter::Off => {
            command.env("FROZEN_FILTER", "0");
        }
    }
}

fn set_or_remove_env(command: &mut ProcessCommand, key: &str, value: Option<&Path>) {
    match value {
        Some(value) => {
            command.env(key, value);
        }
        None => {
            command.env_remove(key);
        }
    }
}

fn print_resolved(config_path: &Path, config: &ResolvedConfig, binary_dir: &Path) {
    println!("[gss] configuration: {}", config_path.display());
    println!("[gss] n: {}", config.n);
    println!(
        "[gss] run directory: {}{}",
        config.run_dir.display(),
        if config.generated_run_dir {
            " (fresh default)"
        } else {
            " (configured/resume)"
        }
    );
    print_store("regular frozen store", &config.frozen_db);
    print_store("curated frozen store", &config.frozen_curated);
    println!(
        "[gss] store convention/filter: curated={} filter={}",
        config.curated_value_convention,
        match config.frozen_filter {
            FrozenFilter::Auto => "auto",
            FrozenFilter::On => "on",
            FrozenFilter::Off => "off",
        }
    );
    println!("[gss] production preset: {}", config.production_preset);
    println!(
        "[gss] post fragmentation: {}",
        config.post_fragment.as_deref().unwrap_or("preset default")
    );
    println!(
        "[gss] seed: {}",
        if config.calibration_seed.is_some() {
            "EXPLICIT FROM PRIVATE FILE (redacted; CALIBRATION ONLY)"
        } else {
            "CSPRNG for a fresh run / protected SEED file for resume"
        }
    );
    println!(
        "[gss] stage 1-2: mcd={}",
        config.mcd.as_deref().unwrap_or("derived")
    );
    println!(
        "[gss] stage 3: expand={} hold={}",
        config.expand.as_deref().unwrap_or("2"),
        config.hold.as_deref().unwrap_or("30")
    );
    println!(
        "[gss] stage 5: xr={} xb={} xc={} xtdiv={} xmoves={}",
        config.xr.as_deref().unwrap_or("2"),
        config.xb.as_deref().unwrap_or("3"),
        config.xc.as_deref().unwrap_or("1"),
        config.xtdiv.as_deref().unwrap_or("25"),
        config.xmoves.as_deref().unwrap_or("6*target")
    );
    println!(
        "[gss] lifecycle: stop_after={} force_from={} build_release={}",
        config.stop_after.as_deref().unwrap_or("6"),
        config.force_from.as_deref().unwrap_or("none"),
        config.build_release
    );
    println!("[gss] release binaries: {}", binary_dir.display());
    if config.calibration_only {
        eprintln!("[gss] WARNING: calibration_only=true; this run is not a deliverable");
    }
    if config.allow_empty_store {
        eprintln!(
            "[gss] WARNING: allow_empty_store=true; empty-store Phase A is armed for plumbing tests only"
        );
    }
    let stop_after = config
        .stop_after
        .as_deref()
        .unwrap_or("6")
        .parse::<u64>()
        .expect("validated stop_after");
    if stop_after >= 5 {
        eprintln!(
            "[gss] WARNING: stage 5 is enabled; its current numerical defaults remain calibration material until deliverable-promotion policy is recorded"
        );
    }
}

fn print_store(label: &str, store: &SourcedPath) {
    match &store.path {
        Some(path) => println!(
            "[gss] {label}: {} ({})",
            path.display(),
            store.source.label()
        ),
        None => println!("[gss] {label}: unset"),
    }
}

fn print_dry_run(
    script: &Path,
    args: &[OsString],
    config: &ResolvedConfig,
    host_target: &str,
    should_build: bool,
    managed_resume: bool,
) {
    println!("[gss] dry run: no directories, builds, or pipeline processes will be started");
    if should_build {
        let build = vec![
            OsString::from("cargo"),
            OsString::from("build"),
            OsString::from("--release"),
            OsString::from("--locked"),
            OsString::from("--target-dir"),
            config.build_target_dir.clone().into_os_string(),
            OsString::from("--target"),
            OsString::from(host_target),
            OsString::from("--bin"),
            OsString::from("gen_sandwich_gadget"),
            OsString::from("--bin"),
            OsString::from("fmix"),
            OsString::from("--bin"),
            OsString::from("fcompress"),
        ];
        println!("[gss] would build: {}", render_command(&build, None));
    } else if managed_resume && config.build_release {
        println!(
            "[gss] release build: would be skipped for resume; the fingerprinted binaries would be reused"
        );
    } else {
        println!("[gss] release build: skipped by configuration");
    }
    let mut invocation = vec![OsString::from("bash"), script.as_os_str().to_owned()];
    invocation.extend_from_slice(args);
    println!(
        "[gss] would run: {}",
        render_command(&invocation, Some("-s"))
    );
}

fn render_command(args: &[OsString], redact_value_after: Option<&str>) -> String {
    let mut rendered = Vec::with_capacity(args.len());
    let mut redact_next = false;
    for arg in args {
        if redact_next {
            rendered.push("<redacted>".to_owned());
            redact_next = false;
            continue;
        }
        let text = arg.to_string_lossy();
        rendered.push(shell_quote_for_display(&text));
        if redact_value_after == Some(text.as_ref()) {
            redact_next = true;
        }
    }
    rendered.join(" ")
}

fn shell_quote_for_display(value: &str) -> String {
    if !value.is_empty()
        && value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || "_+-./:=,".contains(ch))
    {
        return value.to_owned();
    }
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn validate_run_destination(repo_root: &Path, run_dir: &Path) -> Result<(), String> {
    let physical_run_dir = resolve_existing_prefix(run_dir)
        .map_err(|error| format!("cannot resolve run_dir {}: {error}", run_dir.display()))?;
    let forbidden = [
        repo_root.to_path_buf(),
        repo_root.join(".git"),
        repo_root.join("src"),
        repo_root.join("scripts"),
        repo_root.join("target"),
    ];
    let repository_runs = repo_root.join("runs");
    if physical_run_dir.parent().is_none()
        || physical_run_dir == repo_root
        || physical_run_dir.starts_with(repo_root)
            && (physical_run_dir == repository_runs
                || !physical_run_dir.starts_with(&repository_runs))
        || forbidden
            .iter()
            .skip(1)
            .any(|path| physical_run_dir.starts_with(path))
    {
        return Err(format!(
            "refusing dangerous run_dir {}; choose a dedicated run directory",
            run_dir.display()
        ));
    }
    if run_dir.exists() {
        if !run_dir.is_dir() {
            return Err(format!(
                "run_dir {} exists but is not a directory",
                run_dir.display()
            ));
        }
        let entries: Vec<_> = fs::read_dir(run_dir)
            .map_err(|error| format!("cannot inspect run_dir {}: {error}", run_dir.display()))?
            .collect::<Result<_, _>>()
            .map_err(|error| format!("cannot inspect run_dir {}: {error}", run_dir.display()))?;
        let has_only_wrapper_manifest =
            entries.len() == 1 && entries[0].file_name() == OsStr::new("gss_command.conf");
        let seed_path = run_dir.join("SEED");
        let has_seed =
            fs::metadata(&seed_path).is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0);
        if has_seed {
            read_calibration_seed(&seed_path).map_err(|error| {
                format!("existing run has an invalid or unprotected SEED file: {error}")
            })?;
        }
        if !entries.is_empty() && !has_only_wrapper_manifest && !has_seed {
            return Err(format!(
                "existing run_dir {} is nonempty but has no protected SEED file; refusing to mix pipeline artifacts into an unrelated or incomplete directory",
                run_dir.display()
            ));
        }
    }
    Ok(())
}

fn validate_build_target(repo_root: &Path, target_dir: &Path) -> Result<(), String> {
    let physical_target = resolve_existing_prefix(target_dir).map_err(|error| {
        format!(
            "cannot resolve build_target_dir {}: {error}",
            target_dir.display()
        )
    })?;
    let forbidden = [
        repo_root.join(".git"),
        repo_root.join("src"),
        repo_root.join("scripts"),
    ];
    let repository_target = repo_root.join("target");
    if physical_target.parent().is_none()
        || physical_target == repo_root
        || physical_target.starts_with(repo_root)
            && !physical_target.starts_with(&repository_target)
        || forbidden
            .iter()
            .any(|path| physical_target.starts_with(path))
    {
        return Err(format!(
            "refusing dangerous build_target_dir {}; choose a dedicated Cargo target directory",
            target_dir.display()
        ));
    }
    Ok(())
}

fn validate_path_separation(
    run_dir: &Path,
    target_dir: &Path,
    frozen_db: Option<&Path>,
    frozen_curated: Option<&Path>,
) -> Result<(), String> {
    let mut paths = vec![
        ("run_dir", resolve_existing_prefix(run_dir)),
        ("build_target_dir", resolve_existing_prefix(target_dir)),
    ];
    if let Some(path) = frozen_db {
        paths.push(("frozen_db_dir", resolve_existing_prefix(path)));
    }
    if let Some(path) = frozen_curated {
        paths.push(("frozen_curated_dir", resolve_existing_prefix(path)));
    }
    let paths: Vec<(&str, PathBuf)> = paths
        .into_iter()
        .map(|(label, path)| {
            path.map(|path| (label, path)).map_err(|error| {
                format!("cannot resolve {label} while checking path separation: {error}")
            })
        })
        .collect::<Result<_, _>>()?;
    for left in 0..paths.len() {
        for right in left + 1..paths.len() {
            let (left_label, left_path) = &paths[left];
            let (right_label, right_path) = &paths[right];
            if left_path.starts_with(right_path) || right_path.starts_with(left_path) {
                return Err(format!(
                    "{left_label} ({}) and {right_label} ({}) overlap; run artifacts, Cargo outputs, and frozen stores must use separate directory trees",
                    left_path.display(),
                    right_path.display()
                ));
            }
        }
    }
    Ok(())
}

/// Resolve symlinks in the longest existing prefix while retaining a
/// not-yet-created suffix. This keeps `safe-link/new-run` from bypassing the
/// protected repository-directory checks when `safe-link` points at `src/`.
fn resolve_existing_prefix(path: &Path) -> std::io::Result<PathBuf> {
    let mut existing = path;
    let mut suffix = Vec::new();
    while !existing.exists() {
        let Some(name) = existing.file_name() else {
            break;
        };
        suffix.push(name.to_owned());
        let Some(parent) = existing.parent() else {
            break;
        };
        existing = parent;
    }
    let mut resolved = existing.canonicalize()?;
    for component in suffix.into_iter().rev() {
        resolved.push(component);
    }
    Ok(lexical_normalize(&resolved))
}

fn fresh_default_run_dir(repo_root: &Path, n: usize, tag: &str) -> PathBuf {
    let base = repo_root.join("runs").join(format!("gssmix_n{n}_{tag}"));
    if !base.exists() {
        return base;
    }
    for suffix in 1u32.. {
        let candidate = repo_root
            .join("runs")
            .join(format!("gssmix_n{n}_{tag}_{suffix:02}"));
        if !candidate.exists() {
            return candidate;
        }
    }
    unreachable!()
}

fn default_run_tag() -> String {
    let epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{epoch}-{}", std::process::id())
}

fn resolve_path(repo_root: &Path, path: &Path) -> PathBuf {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        repo_root.join(path)
    };
    lexical_normalize(&absolute)
}

fn lexical_normalize(path: &Path) -> PathBuf {
    let mut result = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                result.pop();
            }
            other => result.push(other.as_os_str()),
        }
    }
    result
}

fn rustc_host_triple() -> Result<String, GssError> {
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| OsString::from("rustc"));
    let output = ProcessCommand::new(rustc)
        .arg("-vV")
        .output()
        .map_err(|error| GssError::io(format!("cannot query the Rust host target: {error}")))?;
    if !output.status.success() {
        return Err(GssError::child("rustc -vV", output.status));
    }
    let stdout = String::from_utf8(output.stdout)
        .map_err(|_| GssError::io("rustc -vV returned non-UTF-8 output"))?;
    let host = stdout
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .ok_or_else(|| GssError::io("rustc -vV did not report a host target"))?;
    if host.is_empty()
        || !host
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_')
    {
        return Err(GssError::io(format!(
            "rustc reported an invalid host target {host:?}"
        )));
    }
    Ok(host.to_owned())
}

fn find_repo_root() -> Result<PathBuf, GssError> {
    // Never select a checkout merely because it is the caller's current
    // directory: that would let an unrelated directory substitute its own
    // scripts/gss_mix.sh. Prefer the checkout this binary was compiled from,
    // then support a moved checkout by walking upward from the executable.
    let mut starts = vec![PathBuf::from(env!("CARGO_MANIFEST_DIR"))];
    if let Ok(executable) = std::env::current_exe()
        && let Some(parent) = executable.parent()
    {
        starts.push(parent.to_path_buf());
    }

    for start in starts {
        for candidate in start.ancestors() {
            if is_repo_root(candidate) {
                return candidate.canonicalize().map_err(|error| {
                    GssError::io(format!(
                        "cannot resolve repository root {}: {error}",
                        candidate.display()
                    ))
                });
            }
        }
    }
    Err(GssError::io(
        "cannot locate the repository (need Cargo.toml, docs/GSS_MIX.md, and scripts/gss_mix.sh)",
    ))
}

fn is_repo_root(path: &Path) -> bool {
    path.join("Cargo.toml").is_file()
        && path.join("docs/GSS_MIX.md").is_file()
        && path.join("scripts/gss_mix.sh").is_file()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    fn temp_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "local_mixing_gss_{label}_{}_{}",
            std::process::id(),
            TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn document(body: &str) -> String {
        format!(
            "prose with ignored = examples\n{CONFIG_BEGIN}\n```ini\n{body}\n```\n{CONFIG_END}\nmore prose"
        )
    }

    fn parse(body: &str) -> RawConfig {
        parse_config(&document(body)).unwrap()
    }

    fn resolve_for_test(raw: &RawConfig) -> Result<ResolvedConfig, String> {
        resolve_config(
            raw,
            Path::new(env!("CARGO_MANIFEST_DIR")),
            |_| None,
            "test-run",
        )
    }

    #[test]
    fn command_surface_parses() {
        let matches = command()
            .try_get_matches_from(["gss", "--dry-run", "--config", "custom.md"])
            .unwrap();
        assert!(matches.get_flag("dry_run"));
        assert_eq!(
            matches.get_one::<PathBuf>("config"),
            Some(&PathBuf::from("custom.md"))
        );
    }

    #[test]
    fn parser_ignores_prose_and_accepts_crlf() {
        let doc =
            document("n = 64\nrun_dir = runs/a = $(literal); still-literal").replace('\n', "\r\n");
        let parsed = parse_config(&doc).unwrap();
        assert_eq!(parsed.value("n"), Some("64"));
        assert_eq!(
            parsed.value("run_dir"),
            Some("runs/a = $(literal); still-literal")
        );
    }

    #[test]
    fn parser_rejects_unknown_duplicate_malformed_and_quoted_values() {
        assert!(
            parse_config(&document("typo = 1"))
                .unwrap_err()
                .contains("unknown")
        );
        assert!(
            parse_config(&document("n = 32\nn = 64"))
                .unwrap_err()
                .contains("duplicate")
        );
        assert!(
            parse_config(&document("n 64"))
                .unwrap_err()
                .contains("key = value")
        );
        assert!(
            parse_config(&document("run_dir = \"runs/a\""))
                .unwrap_err()
                .contains("must not be quoted")
        );
    }

    #[test]
    fn parser_requires_one_ordered_fenced_block() {
        assert!(parse_config("no block").is_err());
        assert!(parse_config(&format!("{CONFIG_END}\n{CONFIG_BEGIN}")).is_err());
        let doubled = format!("{}\n{}", document("n ="), document("n ="));
        assert!(parse_config(&doubled).is_err());
        assert!(
            parse_config(&format!("{CONFIG_BEGIN}\nn =\n{CONFIG_END}"))
                .unwrap_err()
                .contains("```ini")
        );
    }

    #[test]
    fn blank_values_resolve_to_safe_defaults() {
        let raw =
            parse("n =\nrun_dir =\nbuild_release =\nproduction_preset =\nallow_empty_store = true");
        let resolved = resolve_for_test(&raw).unwrap();
        assert_eq!(resolved.n, 128);
        assert!(resolved.generated_run_dir);
        assert!(resolved.run_dir.ends_with("runs/gssmix_n128_test-run"));
        assert!(resolved.build_release);
        assert_eq!(resolved.production_preset, "production");
        assert!(resolved.calibration_seed.is_none());
        assert!(
            !script_args(&resolved)
                .iter()
                .any(|arg| arg == OsStr::new("-s"))
        );
    }

    #[test]
    fn every_public_tuning_value_maps_to_a_separate_script_argument() {
        let raw = parse(
            "n = 64\nrun_dir = runs/with spaces;$(literal)\nmcd = 900\nexpand = 2.5\nhold = 4.5\nxr = 2.25\nxb = 4\nxc = 2\nxtdiv = 20\nxmoves = 1234\nstop_after = 5\nforce_from = 4\nallow_empty_store = true",
        );
        let resolved = resolve_for_test(&raw).unwrap();
        let args = script_args(&resolved);
        let strings: Vec<String> = args
            .iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        for expected in [
            "-n",
            "64",
            "-o",
            "--mcd",
            "900",
            "--expand",
            "2.5",
            "--hold",
            "4.5",
            "--xr",
            "2.25",
            "--xb",
            "4",
            "--xc",
            "2",
            "--xtdiv",
            "20",
            "--xmoves",
            "1234",
            "--stop-after",
            "5",
            "--force-from",
            "4",
        ] {
            assert!(
                strings.iter().any(|actual| actual == expected),
                "missing {expected}"
            );
        }
        assert!(resolved.run_dir.ends_with("runs/with spaces;$(literal)"));
    }

    #[test]
    fn environment_store_paths_are_resolved_and_document_paths_win() {
        let raw = parse("run_dir = runs/test\nfrozen_db_dir = stores/from-doc\nstop_after = 2");
        let resolved = resolve_config(
            &raw,
            Path::new(env!("CARGO_MANIFEST_DIR")),
            |key| (key == "FROZEN_DB_DIR").then(|| OsString::from("stores/from-env")),
            "test",
        )
        .unwrap();
        assert_eq!(resolved.frozen_db.source, ValueSource::Document);
        assert!(
            resolved
                .frozen_db
                .path
                .unwrap()
                .ends_with("stores/from-doc")
        );
    }

    #[test]
    fn resume_after_phase_a_does_not_require_the_stage_three_store() {
        let run_dir = std::env::temp_dir().join(format!(
            "local_mixing_gss_resume_{}_{}",
            std::process::id(),
            default_run_tag()
        ));
        fs::create_dir_all(&run_dir).unwrap();
        fs::write(run_dir.join("phaseA.mpmct1"), "nonempty").unwrap();
        fs::write(run_dir.join("SEED"), "123456\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(run_dir.join("SEED"), fs::Permissions::from_mode(0o600)).unwrap();
        }

        let raw = parse(&format!("run_dir = {}", run_dir.display()));
        let resolved = resolve_for_test(&raw).unwrap();
        validate_external_paths(&resolved).unwrap();

        let forced = parse(&format!("run_dir = {}\nforce_from = 3", run_dir.display()));
        let forced = resolve_for_test(&forced).unwrap();
        assert!(validate_external_paths(&forced).is_err());
        fs::remove_dir_all(run_dir).unwrap();
    }

    #[test]
    fn invalid_numbers_enums_and_booleans_fail_before_launch() {
        for body in [
            "n = 2",
            "expand = NaN",
            "expand = 1",
            "hold = -1",
            "xtdiv = 0",
            "stop_after = 7",
            "production_preset = invented",
            "build_release = yes",
            "frozen_filter = maybe",
        ] {
            let raw = parse(&format!("{body}\nallow_empty_store = true"));
            assert!(
                resolve_for_test(&raw).is_err(),
                "unexpectedly accepted {body}"
            );
        }
    }

    #[test]
    fn explicit_seed_requires_calibration_gate_and_is_redacted() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let seed_path = std::env::temp_dir().join(format!(
            "local_mixing_gss_seed_{}_{}",
            std::process::id(),
            default_run_tag()
        ));
        fs::write(&seed_path, "123456\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&seed_path, fs::Permissions::from_mode(0o600)).unwrap();
        }
        let without_gate = parse(&format!(
            "run_dir = runs/test\ncalibration_seed_file = {}\nallow_empty_store = true",
            seed_path.display()
        ));
        assert!(resolve_for_test(&without_gate).is_err());

        let with_gate = parse(&format!(
            "run_dir = runs/test\ncalibration_only = true\ncalibration_seed_file = {}\nallow_empty_store = true",
            seed_path.display()
        ));
        let resolved = resolve_config(&with_gate, root, |_| None, "test").unwrap();
        let mut invocation = vec![OsString::from("bash"), OsString::from("script")];
        invocation.extend(script_args(&resolved));
        let rendered = render_command(&invocation, Some("-s"));
        assert!(rendered.contains("<redacted>"));
        assert!(!rendered.contains("123456"));
        fs::remove_file(seed_path).unwrap();
    }

    #[test]
    fn actual_manual_contains_a_valid_config_block() {
        let parsed = parse_config(include_str!("../../docs/GSS_MIX.md")).unwrap();
        assert!(
            KNOWN_KEYS
                .iter()
                .all(|key| parsed.entries.contains_key(*key))
        );
    }

    #[test]
    fn dangerous_run_destinations_are_rejected() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        for value in [
            ".",
            "src",
            "src/generated-run",
            "scripts",
            "target",
            "runs/../.git",
            "runs/../.git/gss",
        ] {
            let raw = parse(&format!("run_dir = {value}\nallow_empty_store = true"));
            assert!(resolve_config(&raw, root, |_| None, "test").is_err());
        }
    }

    #[test]
    fn recipe_manifest_is_seed_free_normalized_and_lifecycle_independent() {
        let base = resolve_for_test(&parse(
            "run_dir = runs/manifest-test\nstop_after = 2\nallow_empty_store = true",
        ))
        .unwrap();
        let fingerprints = [
            ("gen_sandwich_gadget", 1u128),
            ("fmix", 2u128),
            ("fcompress", 3u128),
        ];
        let expected = recipe_manifest(&base, &fingerprints, 4);
        assert!(expected.contains("binary_fmix_xxh3=00000000000000000000000000000002"));
        assert!(expected.contains("script_gss_mix_xxh3=00000000000000000000000000000004"));

        let explicit_defaults = resolve_for_test(&parse(
            "run_dir = runs/manifest-test\nexpand = 2.0\nhold = 3e1\nxr = 2e0\nxb = 3.0\nxc = 1\nxtdiv = 25\nstop_after = 2\nallow_empty_store = true",
        ))
        .unwrap();
        assert_eq!(
            expected,
            recipe_manifest(&explicit_defaults, &fingerprints, 4)
        );

        let mut lifecycle = base.clone();
        lifecycle.calibration_seed = Some("987654321".to_owned());
        lifecycle.build_release = false;
        lifecycle.build_target_dir = PathBuf::from("/different/target");
        lifecycle.stop_after = Some("6".to_owned());
        lifecycle.force_from = Some("4".to_owned());
        lifecycle.frozen_filter = FrozenFilter::Off;
        assert_eq!(expected, recipe_manifest(&lifecycle, &fingerprints, 4));
        assert!(!expected.contains("987654321"));

        lifecycle.expand = Some("3".to_owned());
        assert_ne!(expected, recipe_manifest(&lifecycle, &fingerprints, 4));
        lifecycle.expand = base.expand.clone();
        lifecycle.calibration_only = true;
        assert_ne!(expected, recipe_manifest(&lifecycle, &fingerprints, 4));
    }

    #[test]
    fn manifest_comparison_accepts_exact_and_rejects_changed_or_oversized() {
        let path = temp_path("manifest");
        let desired = "gss_command_recipe=1\nn=128\n";
        fs::write(&path, desired).unwrap();
        compare_recipe_manifest(&path, desired).unwrap();

        let mismatch = compare_recipe_manifest(&path, "gss_command_recipe=1\nn=129\n").unwrap_err();
        assert_eq!(mismatch.exit_code, 2);
        assert!(mismatch.message.contains("mixed-provenance"));

        fs::write(&path, vec![b'x'; MAX_MANIFEST_BYTES as usize + 1]).unwrap();
        let oversized = compare_recipe_manifest(&path, desired).unwrap_err();
        assert_eq!(oversized.exit_code, 2);
        assert!(oversized.message.contains("no larger"));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn configured_child_environment_scrubs_hidden_overrides_and_pins_recipe() {
        let config = resolve_for_test(&parse(
            "run_dir = runs/env-test\nstop_after = 2\nallow_empty_store = true",
        ))
        .unwrap();
        let mut command = ProcessCommand::new("true");
        configure_environment(&mut command, &config, Path::new("/tmp/gss-bin"));
        let environment: BTreeMap<String, Option<String>> = command
            .get_envs()
            .map(|(key, value)| {
                (
                    key.to_string_lossy().into_owned(),
                    value.map(|value| value.to_string_lossy().into_owned()),
                )
            })
            .collect();
        for key in ["BASH_ENV", "PROD_K", "SAT_HARDEN", "FMIX_DUMP_OUT"] {
            assert_eq!(environment.get(key), Some(&None), "{key} was not scrubbed");
        }
        assert_eq!(
            environment.get("CANON_RULE_L_BRANCH_CAP"),
            Some(&Some("512".to_owned()))
        );
        assert_eq!(
            environment.get("FROZEN_REGULAR_VALUE_CONVENTION"),
            Some(&Some("native".to_owned()))
        );
        assert_eq!(
            environment.get("GSS_BIN_DIR"),
            Some(&Some("/tmp/gss-bin".to_owned()))
        );
    }

    #[test]
    fn run_target_and_store_trees_must_not_overlap() {
        let base = temp_path("separation");
        fs::create_dir_all(&base).unwrap();
        let run = base.join("run");
        let target = base.join("target");
        let store = base.join("store");
        assert!(validate_path_separation(&run, &run.join("target"), None, None).is_err());
        assert!(
            validate_path_separation(&run, &target, Some(&target.join("store")), None).is_err()
        );
        validate_path_separation(&run, &target, Some(&store), None).unwrap();
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn explicitly_enabled_filter_requires_a_filter_file() {
        let store = temp_path("filter");
        fs::create_dir_all(&store).unwrap();
        assert!(validate_requested_filter(&store, "test store", FrozenFilter::On).is_err());
        validate_requested_filter(&store, "test store", FrozenFilter::Auto).unwrap();
        fs::write(store.join("filters.bin"), "filter").unwrap();
        validate_requested_filter(&store, "test store", FrozenFilter::On).unwrap();
        fs::remove_dir_all(store).unwrap();
    }

    #[test]
    fn dry_run_creates_no_run_or_target_directory() {
        let base = temp_path("dry_run");
        fs::create_dir_all(&base).unwrap();
        let config_path = base.join("manual.md");
        let run_dir = base.join("run");
        let target_dir = base.join("target");
        let store_dir = base.join("unused-store");
        fs::write(
            &config_path,
            document(&format!(
                "run_dir = {}\nbuild_target_dir = {}\nfrozen_db_dir = {}\nstop_after = 2",
                run_dir.display(),
                target_dir.display(),
                store_dir.display()
            )),
        )
        .unwrap();
        let matches = command()
            .try_get_matches_from(vec![
                OsString::from("gss"),
                OsString::from("--dry-run"),
                OsString::from("--config"),
                config_path.clone().into_os_string(),
            ])
            .unwrap();
        run_inner(&matches).unwrap();
        assert!(!run_dir.exists());
        assert!(!target_dir.exists());
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn legacy_adoption_is_permanently_calibration_only() {
        let base = temp_path("legacy");
        let run_dir = base.join("run");
        fs::create_dir_all(&run_dir).unwrap();
        fs::write(run_dir.join("SEED"), "123456\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(run_dir.join("SEED"), fs::Permissions::from_mode(0o600)).unwrap();
        }
        let config_path = base.join("manual.md");
        let target_dir = base.join("target");
        let store_dir = base.join("unused-store");
        let write_config = |calibration_only: bool| {
            fs::write(
                &config_path,
                document(&format!(
                    "run_dir = {}\nbuild_target_dir = {}\nfrozen_db_dir = {}\nstop_after = 2\nadopt_existing_run = true\ncalibration_only = {calibration_only}",
                    run_dir.display(),
                    target_dir.display(),
                    store_dir.display()
                )),
            )
            .unwrap();
        };
        let matches = || {
            command()
                .try_get_matches_from(vec![
                    OsString::from("gss"),
                    OsString::from("--dry-run"),
                    OsString::from("--config"),
                    config_path.clone().into_os_string(),
                ])
                .unwrap()
        };

        write_config(false);
        let error = run_inner(&matches()).unwrap_err();
        assert!(error.message.contains("calibration_only = true"));
        write_config(true);
        run_inner(&matches()).unwrap();
        assert!(!run_dir.join("gss_command.conf").exists());
        fs::remove_dir_all(base).unwrap();
    }
}
