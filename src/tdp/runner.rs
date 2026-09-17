//! Build preparation and stage-driver invocation.
use super::*;

/// Low-level research/debug switches are intentionally outside the supported
/// TDP recipe. The direct Bash interface may still use them; the `tdp_gen`
/// command removes them so the selected recipe defines its control surface.
pub(crate) const SCRUBBED_ENV: &[&str] = &[
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
    "CENTRALIZE",
    "COMPRESSION_TRACE",
    "COMPRESSION_TRACE_MS",
    "COMPRESS_CHUNK_BUDGET_MS",
    "COMPRESS_STALL_FRAC",
    "COMPRESS_STALL_WINDOW",
    "DEGREE_FILTER",
    "DEGREE_FILTER_PROBES",
    "FLOAT_SWEEP",
    "MIXER_DUMP_OUT",
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

pub(crate) fn run_inner(sub: &ArgMatches) -> Result<(), TdpError> {
    reject_legacy_mask_controls(std::env::vars_os().map(|(key, _)| key))?;
    let repo_root = find_repo_root()?;
    let config_path = sub
        .get_one::<PathBuf>("config")
        .map(|path| resolve_path(&repo_root, path))
        .unwrap_or_else(|| repo_root.join("configs/tdp.toml"));
    let document = fs::read_to_string(&config_path).map_err(|error| {
        TdpError::io(format!(
            "cannot read configuration {}: {error}",
            config_path.display()
        ))
    })?;
    let raw = parse_config(&document).map_err(TdpError::config)?;
    let run_tag = default_run_tag();
    let mut config = resolve_config(&raw, &repo_root, |key| std::env::var_os(key), &run_tag)
        .map_err(TdpError::config)?;
    let manifest_path = config.run_dir.join("tdp_command.conf");
    let managed_resume = manifest_path.is_file();
    let legacy_resume = fs::metadata(config.run_dir.join("SEED"))
        .is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0)
        && !managed_resume;
    if legacy_resume || config.adopt_existing_run {
        return Err(TdpError::config(format!(
            "pre-wrapper run adoption has been retired; use a fresh run.directory or the checkout that created {}; the original run is unchanged",
            config.run_dir.display()
        )));
    }
    // Reject retired recipes before building or inspecting stage executables.
    config.recipe_version = if managed_resume {
        read_recipe_version(&manifest_path)?
    } else {
        CURRENT_RECIPE_VERSION
    };
    validate_current_recipe(&config)?;
    if config.hold.is_none() {
        config.hold = Some("27".into());
    }
    let host_target = rustc_host_triple()?;
    let binary_dir = config.build_target_dir.join(&host_target).join("release");
    print_resolved(&config_path, &config, &binary_dir);
    validate_external_paths(&config).map_err(TdpError::config)?;
    let script = repo_root.join("scripts/tdp_gen.sh");
    if managed_resume {
        let fingerprints = production_binary_fingerprints(&binary_dir)?;
        let script_hash = hash_file(&script).map_err(|error| {
            TdpError::io(format!("cannot fingerprint scripts/tdp_gen.sh: {error}"))
        })?;
        let desired = recipe_manifest(&config, &fingerprints, script_hash);
        compare_recipe_manifest(&manifest_path, &desired)?;
    } else if !config.build_release {
        production_binary_fingerprints(&binary_dir)?;
    }

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
            "[tdp_gen] release build: skipped for resume; using the exact binaries fingerprinted by the run manifest"
        );
    }
    if should_build {
        println!("[tdp_gen] building the three TDP_GEN release executables (incremental)");
        let cargo = std::env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"));
        let mut build = ProcessCommand::new(cargo);
        build
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
            .arg("circuit_mixer")
            .arg("--bin")
            .arg("fcompress")
            .env_remove("CARGO_BUILD_TARGET");
        let status = build
            .status()
            .map_err(|error| TdpError::io(format!("cannot start cargo build: {error}")))?;
        if !status.success() {
            return Err(TdpError::child("cargo build", status));
        }
    }

    if !managed_resume {
        prepare_recipe_manifest(&config, &binary_dir, &script)?;
    }

    println!("[tdp_gen] starting {}", script.display());
    let mut child = ProcessCommand::new("bash");
    child
        .current_dir(&repo_root)
        .arg(&script)
        .args(&script_args);
    configure_environment(&mut child, &config, &binary_dir);
    let status = child
        .status()
        .map_err(|error| TdpError::io(format!("cannot start bash: {error}")))?;
    if !status.success() {
        return Err(TdpError::child("TDP_GEN pipeline", status));
    }
    Ok(())
}

pub(crate) fn validate_current_recipe(config: &ResolvedConfig) -> Result<(), TdpError> {
    if config.preprocessing_mode == PreprocessingMode::Nonlinear291 && config.explicit_mask_controls
    {
        return Err(TdpError::config(
            "preprocessing mask controls apply only to embedded-masking; omit them for nonlinear291",
        ));
    }
    match config.preprocessing_mode {
        PreprocessingMode::EmbeddedMasking => {
            if config.mask_pair_wires < 2 {
                return Err(TdpError::config(
                    "preprocessing.mask_pair_wires must be in 2..64 for new runs",
                ));
            }
            let cycle_width = config.mask_pair_wires - config.mask_pair_wires % 2
                + usize::from(config.balanced_masks);
            if 2 * config.n <= cycle_width {
                return Err(TdpError::config(
                    "Embedded masking requires 2 * source.wires > preprocessing.mask_pair_wires rounded down to an even number + balanced_masks; reduce preprocessing.mask_pair_wires or increase source.wires",
                ));
            }
        }
        PreprocessingMode::Nonlinear291 => {}
    }
    Ok(())
}

pub(crate) fn reject_legacy_mask_controls(
    keys: impl IntoIterator<Item = OsString>,
) -> Result<(), TdpError> {
    if keys
        .into_iter()
        .any(|key| key.to_string_lossy().starts_with("BV5_"))
    {
        return Err(TdpError::config(
            "BV5_* controls have been removed; unset them and use the preprocessing section in configs/tdp.toml",
        ));
    }
    Ok(())
}

pub(crate) fn canonical_script_flag(flag: &str) -> &str {
    match flag {
        "-n" => "--source-wires",
        "-o" => "--run-directory",
        "-s" => "--calibration-seed",
        "--mcd" => "--source-gates",
        "--expand" => "--db-mixing-target-size-factor",
        "--hold" => "--db-mixing-hold-work-units",
        "--xr" => "--crossing-target-size-factor",
        "--xb" => "--crossing-width-penalty-base",
        "--xc" => "--crossing-width-penalty-threshold",
        "--xtdiv" => "--crossing-size-tolerance-divisor",
        "--xmoves" => "--crossing-move-attempts",
        "--stop-after" => "--run-stop-after-stage",
        "--force-from" => "--run-rerun-from-stage",
        "--pieces" => "--parallel-pieces",
        "--min-block-size" => "--parallel-target-piece-gates",
        "--piece-threads" => "--parallel-threads",
        _ => flag,
    }
}

pub(crate) fn script_args(config: &ResolvedConfig) -> Vec<OsString> {
    let mut args = vec![
        OsString::from("-n"),
        OsString::from(config.n.to_string()),
        OsString::from("-o"),
        config.run_dir.as_os_str().to_owned(),
        OsString::from("--preprocessing-mode"),
        OsString::from(config.preprocessing_mode.canonical_name()),
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
    push_optional_arg(&mut args, "--pieces", config.pieces.as_deref());
    push_optional_arg(
        &mut args,
        "--min-block-size",
        config.min_block_size.as_deref(),
    );
    push_optional_arg(
        &mut args,
        "--piece-threads",
        config.piece_threads.as_deref(),
    );
    if let Some(seed) = &config.calibration_seed {
        args.push(OsString::from("-s"));
        args.push(OsString::from(seed));
    }
    // Only option positions are translated; path/value strings may equal an alias.
    for option in args.iter_mut().step_by(2) {
        if let Some(name) = option.to_str() {
            *option = OsString::from(canonical_script_flag(name));
        }
    }
    args
}

pub(crate) fn push_optional_arg(args: &mut Vec<OsString>, flag: &str, value: Option<&str>) {
    if let Some(value) = value {
        args.push(OsString::from(flag));
        args.push(OsString::from(value));
    }
}

pub(crate) fn configure_environment(
    command: &mut ProcessCommand,
    config: &ResolvedConfig,
    binary_dir: &Path,
) {
    // Clear inherited stage controls before applying the resolved recipe.
    // Retired namespaces are scrubbed too so stale shells cannot affect a run.
    for (key, _) in std::env::vars_os() {
        let key_text = key.to_string_lossy();
        if key_text.starts_with("PROD_")
            || key_text.starts_with("SAT_")
            || key_text.starts_with("EMBEDDED_MASKING_")
            || key_text.starts_with("DB_QC")
        {
            command.env_remove(key);
        }
    }
    for key in SCRUBBED_ENV {
        command.env_remove(key);
    }
    command.env("TDP_BIN_DIR", binary_dir);
    command.env("SANDWICH_VARIANT", "classic");
    set_or_remove_env(command, "TDP_SOURCE_C", config.source_path.as_deref());
    command.env("DB_QC", if config.qc_enabled { "1" } else { "0" });
    command.env("DB_QC_SEED", config.qc_seed.to_string());
    set_or_remove_env(command, "DB_QC_REFERENCE", config.qc_reference.as_deref());
    if config.preprocessing_mode == PreprocessingMode::EmbeddedMasking {
        command.env("EMBEDDED_MASKING_K", config.mask_pair_wires.to_string());
        command.env(
            "EMBEDDED_MASKING_MAX_OPEN",
            config.max_open_masks.to_string(),
        );
        command.env(
            "EMBEDDED_MASKING_MIN_OPEN",
            config.min_open_masks.to_string(),
        );
        command.env(
            "EMBEDDED_MASKING_BALANCED",
            if config.balanced_masks { "1" } else { "0" },
        );
        command.env("EMBEDDED_MASKING_QUAD_FIRE", "1");
        command.env(
            "EMBEDDED_MASKING_SHUFFLING",
            config.shuffling_segments.to_string(),
        );
        command.env(
            "EMBEDDED_MASKING_SHUFFLING_RETURN_HOME",
            if config.shuffling_return_home {
                "1"
            } else {
                "0"
            },
        );
    }
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
    command.env_remove("PROD_PRESET");
    command.env_remove("PROD_POST_FRAGMENT");
    if config.allow_empty_store {
        command.env("TDP_GEN_ALLOW_EMPTY_STORE", "1");
    } else {
        command.env_remove("TDP_GEN_ALLOW_EMPTY_STORE");
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

pub(crate) fn set_or_remove_env(command: &mut ProcessCommand, key: &str, value: Option<&Path>) {
    match value {
        Some(value) => {
            command.env(key, value);
        }
        None => {
            command.env_remove(key);
        }
    }
}

pub(crate) fn print_resolved(config_path: &Path, config: &ResolvedConfig, binary_dir: &Path) {
    println!("[tdp_gen] configuration: {}", config_path.display());
    println!("[tdp_gen] n: {}", config.n);
    println!(
        "[tdp_gen] run directory: {}{}",
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
        "[tdp_gen] store convention/filter: curated={} filter={}",
        config.curated_value_convention,
        match config.frozen_filter {
            FrozenFilter::Auto => "auto",
            FrozenFilter::On => "on",
            FrozenFilter::Off => "off",
        }
    );
    println!(
        "[tdp_gen] preprocessing mode: {}",
        config.preprocessing_mode.canonical_name()
    );
    println!(
        "[tdp_gen] seed: {}",
        if config.calibration_seed.is_some() {
            "EXPLICIT FROM PRIVATE FILE (redacted; CALIBRATION ONLY)"
        } else {
            "CSPRNG for a fresh run / protected SEED file for resume"
        }
    );
    println!(
        "[tdp_gen] stage 1-2: source.gates={}",
        config.mcd.as_deref().unwrap_or("derived")
    );
    println!(
        "[tdp_gen] db_mixing (stage 3): target_size_factor={} hold_work_units={}",
        config.expand.as_deref().unwrap_or("2"),
        config.hold.as_deref().unwrap_or("27")
    );
    if let Some(size) = &config.min_block_size {
        println!(
            "[tdp_gen] stages 3-4: automatic pieces target_piece_gates={} threads={}",
            size,
            config.piece_threads.as_deref().unwrap_or("available CPUs")
        );
    } else {
        println!(
            "[tdp_gen] stages 3-4: pieces={} threads={}",
            config.pieces.as_deref().unwrap_or("1"),
            config.piece_threads.as_deref().unwrap_or("pieces+1")
        );
    }
    println!(
        "[tdp_gen] crossing (stage 5): target_size_factor={} width_penalty_base={} width_penalty_threshold={} size_tolerance_divisor={} move_attempts={}",
        config.xr.as_deref().unwrap_or("2"),
        config.xb.as_deref().unwrap_or("3"),
        config.xc.as_deref().unwrap_or("1"),
        config.xtdiv.as_deref().unwrap_or("25"),
        config.xmoves.as_deref().unwrap_or("6*target")
    );
    println!(
        "[tdp_gen] lifecycle: stop_after_stage={} rerun_from_stage={} build_binaries={}",
        config.stop_after.as_deref().unwrap_or("6"),
        config.force_from.as_deref().unwrap_or("none"),
        config.build_release
    );
    println!("[tdp_gen] release binaries: {}", binary_dir.display());
    if config.calibration_only {
        eprintln!("[tdp_gen] WARNING: calibration_only=true; this run is not a deliverable");
    }
    if config.allow_empty_store {
        eprintln!(
            "[tdp_gen] WARNING: database.allow_no_database_for_tests=true; no-database db_mixing is armed for plumbing tests only"
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
            "[tdp_gen] WARNING: stage 5 is enabled; its current numerical defaults remain calibration material until deliverable-promotion policy is recorded"
        );
    }
}

pub(crate) fn print_store(label: &str, store: &SourcedPath) {
    match &store.path {
        Some(path) => println!(
            "[tdp_gen] {label}: {} ({})",
            path.display(),
            store.source.label()
        ),
        None => println!("[tdp_gen] {label}: unset"),
    }
}

pub(crate) fn print_dry_run(
    script: &Path,
    args: &[OsString],
    config: &ResolvedConfig,
    host_target: &str,
    should_build: bool,
    managed_resume: bool,
) {
    println!("[tdp_gen] dry run: no directories, builds, or pipeline processes will be started");
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
            OsString::from("circuit_mixer"),
            OsString::from("--bin"),
            OsString::from("fcompress"),
        ];
        println!("[tdp_gen] would build: {}", render_command(&build, None));
    } else if managed_resume && config.build_release {
        println!(
            "[tdp_gen] release build: would be skipped for resume; the fingerprinted binaries would be reused"
        );
    } else {
        println!("[tdp_gen] release build: skipped by configuration");
    }
    let mut invocation = vec![OsString::from("bash"), script.as_os_str().to_owned()];
    invocation.extend_from_slice(args);
    println!(
        "[tdp_gen] would run: {}",
        render_command(&invocation, Some("-s"))
    );
}

pub(crate) fn render_command(args: &[OsString], redact_value_after: Option<&str>) -> String {
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
        if redact_value_after
            .is_some_and(|flag| canonical_script_flag(flag) == canonical_script_flag(text.as_ref()))
        {
            redact_next = true;
        }
    }
    rendered.join(" ")
}

pub(crate) fn shell_quote_for_display(value: &str) -> String {
    if !value.is_empty()
        && value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || "_+-./:=,".contains(ch))
    {
        return value.to_owned();
    }
    format!("'{}'", value.replace('\'', "'\\''"))
}
