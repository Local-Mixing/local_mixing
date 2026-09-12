//! Build preparation and stage-driver invocation.
use super::*;

/// Low-level research/debug switches are intentionally outside the supported
/// GSS recipe. The direct Bash interface may still use them; the `gss`
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

pub(crate) fn run_inner(sub: &ArgMatches) -> Result<(), GssError> {
    let repo_root = find_repo_root()?;
    let config_path = sub
        .get_one::<PathBuf>("config")
        .map(|path| resolve_path(&repo_root, path))
        .unwrap_or_else(|| repo_root.join("configs/gss.toml"));
    let document = fs::read_to_string(&config_path).map_err(|error| {
        GssError::io(format!(
            "cannot read configuration {}: {error}",
            config_path.display()
        ))
    })?;
    let raw = parse_config(&document).map_err(GssError::config)?;
    let run_tag = default_run_tag();
    let mut config = resolve_config(&raw, &repo_root, |key| std::env::var_os(key), &run_tag)
        .map_err(GssError::config)?;
    let host_target = rustc_host_triple()?;
    let binary_dir = config.build_target_dir.join(&host_target).join("release");

    let manifest_path = config.run_dir.join("gss_command.conf");
    let managed_resume = manifest_path.is_file();
    let legacy_resume = fs::metadata(config.run_dir.join("SEED"))
        .is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0)
        && !managed_resume;
    if legacy_resume && !config.adopt_existing_run {
        return Err(GssError::config(format!(
            "{} is a pre-wrapper run with no gss_command.conf; set `run.adopt_unverified_run = true` once to acknowledge that its earlier recipe cannot be verified, or use the direct Bash driver",
            config.run_dir.display()
        )));
    }
    if legacy_resume && !config.calibration_only {
        return Err(GssError::config(format!(
            "{} has unverifiable pre-wrapper seed provenance; adopting it requires both `run.adopt_unverified_run = true` and `calibration.enabled = true`",
            config.run_dir.display()
        )));
    }
    let legacy_gadget = config.run_dir.join("gss.mpmct1");
    let legacy_has_gadget =
        fs::metadata(&legacy_gadget).is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0);
    if legacy_resume && legacy_has_gadget && config.force_from.as_deref() != Some("2") {
        let recipe_path = config.run_dir.join("stage12.recipe");
        let stored_recipe = fs::read_to_string(&recipe_path).map_err(|error| {
            GssError::config(format!(
                "cannot adopt existing stage-2 artifact without a readable {} marker: {error}; set `run.rerun_from_stage = 2` to rebuild it",
                recipe_path.display()
            ))
        })?;
        let stored_mode = stored_recipe
            .lines()
            .find_map(|line| line.strip_prefix("gadgetization_mode="))
            .ok_or_else(|| {
                GssError::config(format!(
                    "existing stage-2 recipe {} has no gadgetization_mode; set `run.rerun_from_stage = 2` to rebuild it",
                    recipe_path.display()
                ))
            })?;
        if stored_mode != config.preprocessing_mode.for_recipe(4) {
            return Err(GssError::config(format!(
                "existing stage-2 artifact is marked {}, but the requested gadgetization_mode is {}; use a fresh run directory or set `run.rerun_from_stage = 2`",
                stored_mode,
                config.preprocessing_mode.as_str()
            )));
        }
    }
    // Saved runs keep their original driver bytes and manifest spelling.
    // Pre-wrapper runs use the previous driver when explicitly adopted.
    config.recipe_version = if managed_resume {
        read_recipe_version(&manifest_path)?
    } else if legacy_resume {
        4
    } else {
        7
    };
    if config.recipe_version >= 5 || !config.bv5_balanced {
        validate_current_recipe(&config)?;
    }
    if config.recipe_version >= 4 && config.hold.is_none() {
        config.hold = Some("27".into());
    }
    if config.recipe_version == 3 && (config.source_path.is_some() || config.qc_enabled) {
        return Err(GssError::config(
            "v3 manifests cannot verify new source/leakage-repair settings; resume with the original recipe",
        ));
    }
    print_resolved(&config_path, &config, &binary_dir);
    validate_external_paths(&config).map_err(GssError::config)?;
    let script = script_for_recipe(&repo_root, config.recipe_version);
    if managed_resume {
        let fingerprints = production_binary_fingerprints(&binary_dir)?;
        let script_hash = hash_file(&script).map_err(|error| {
            GssError::io(format!("cannot fingerprint scripts/gss_mix.sh: {error}"))
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
            "[gss] release build: skipped for resume; using the exact binaries fingerprinted by the run manifest"
        );
    }
    if should_build {
        println!("[gss] building the three GSS_MIX release executables (incremental)");
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
            .arg("fmix")
            .arg("--bin")
            .arg("fcompress")
            .env_remove("CARGO_BUILD_TARGET");
        if config.preprocessing_mode.supported().is_none() {
            build.arg("--features").arg("legacy-tools");
        }
        let status = build
            .status()
            .map_err(|error| GssError::io(format!("cannot start cargo build: {error}")))?;
        if !status.success() {
            return Err(GssError::child("cargo build", status));
        }
    }

    if !managed_resume {
        prepare_recipe_manifest(&config, &binary_dir, &script)?;
    }

    println!("[gss] starting {}", script.display());
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

pub(crate) fn validate_current_recipe(config: &ResolvedConfig) -> Result<(), GssError> {
    if config.recipe_version >= 7 {
        if config.preprocessing_mode.supported().is_none() {
            return Err(GssError::config(format!(
                "{} is supported only for existing runs; fresh preprocessing.mode must be quadratic-masking or nonlinear291",
                config.preprocessing_mode.as_str()
            )));
        }
        if config.preprocessing_mode == RecipePreprocessingMode::Nonlinear291
            && config.explicit_mask_controls
        {
            return Err(GssError::config(
                "preprocessing mask controls apply only to quadratic-masking; omit them for nonlinear291",
            ));
        }
    } else if !config.bv5_balanced {
        return Err(GssError::config(
            "saved v3-v6 recipes pin balanced masks; resume with preprocessing.balanced_masks = true",
        ));
    }
    match config.preprocessing_mode {
        RecipePreprocessingMode::Product2223 => {
            return Err(GssError::config(
                "product-2223 (including alias 2223) is supported only for existing runs; use preprocessing.mode = \"quadratic-masking\" for a new GSS run",
            ));
        }
        RecipePreprocessingMode::QuadraticMasking => {
            if config.bv5_k < 2 {
                return Err(GssError::config(
                    "preprocessing.mask_pair_wires must be in 2..64 for new runs",
                ));
            }
            let cycle_width = config.bv5_k - config.bv5_k % 2 + usize::from(config.bv5_balanced);
            if 2 * config.n <= cycle_width {
                return Err(GssError::config(
                    "Quadratic masking requires 2 * source.wires > preprocessing.mask_pair_wires rounded down to an even number + balanced_masks; reduce preprocessing.mask_pair_wires or increase source.wires",
                ));
            }
        }
        RecipePreprocessingMode::Nonlinear193 | RecipePreprocessingMode::Nonlinear291 => {}
    }
    Ok(())
}

pub(crate) fn canonical_script_flag(flag: &str) -> &str {
    match flag {
        "-n" => "--source-wires",
        "-o" => "--run-directory",
        "-s" => "--calibration-seed",
        "--mcd" => "--source-gates",
        "--gadgetization-mode" => "--gadget-mode",
        "--bv5-k" => "--gadget-mask-pair-wires",
        "--bv5-max-open" => "--gadget-max-open-masks",
        "--bv5-min-open" => "--gadget-min-open-masks",
        "--bv5-balanced" => "--gadget-balanced-masks",
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

/// Public v7 spellings are isolated from the immutable v3-v6 driver adapters.
pub(crate) fn preprocessing_script_flag(flag: &str) -> &str {
    match flag {
        "--gadget-mode" => "--preprocessing-mode",
        "--gadget-mask-pair-wires" => "--preprocessing-mask-pair-wires",
        "--gadget-max-open-masks" => "--preprocessing-max-open-masks",
        "--gadget-min-open-masks" => "--preprocessing-min-open-masks",
        "--gadget-balanced-masks" => "--preprocessing-balanced-masks",
        _ => flag,
    }
}

pub(crate) fn script_args(config: &ResolvedConfig) -> Vec<OsString> {
    let mut args = vec![
        OsString::from("-n"),
        OsString::from(config.n.to_string()),
        OsString::from("-o"),
        config.run_dir.as_os_str().to_owned(),
        OsString::from("--gadgetization-mode"),
        OsString::from(config.preprocessing_mode.for_recipe(config.recipe_version)),
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
    if config.recipe_version >= 6 {
        // Only option positions are translated; path/value strings may equal an alias.
        for option in args.iter_mut().step_by(2) {
            if let Some(name) = option.to_str() {
                let canonical = canonical_script_flag(name);
                *option = OsString::from(if config.recipe_version >= 7 {
                    preprocessing_script_flag(canonical)
                } else {
                    canonical
                });
            }
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
    // Future generator experiments should not silently become supported merely
    // by adding another PROD_* read. Clear the whole namespace first, then set
    // the product-family controls only when that family was selected.
    for (key, _) in std::env::vars_os() {
        let key_text = key.to_string_lossy();
        if key_text.starts_with("PROD_")
            || key_text.starts_with("SAT_")
            || (config.recipe_version >= 4
                && (key_text.starts_with("BV5_") || key_text.starts_with("DB_QC")))
        {
            command.env_remove(key);
        }
    }
    for key in SCRUBBED_ENV {
        command.env_remove(key);
    }
    command.env("GSS_BIN_DIR", binary_dir);
    // Preserve the original environment contract when continuing a v3 run.
    if config.recipe_version >= 4 {
        command.env("SANDWICH_VARIANT", "classic");
        set_or_remove_env(command, "GSS_SOURCE_C", config.source_path.as_deref());
        command.env("DB_QC", if config.qc_enabled { "1" } else { "0" });
        command.env("DB_QC_SEED", config.qc_seed.to_string());
        set_or_remove_env(command, "DB_QC_REFERENCE", config.qc_reference.as_deref());
        if config.preprocessing_mode == RecipePreprocessingMode::QuadraticMasking {
            command.env("BV5_K", config.bv5_k.to_string());
            command.env("BV5_MAX_OPEN", config.bv5_max_open.to_string());
            command.env("BV5_MIN_OPEN", config.bv5_min_open.to_string());
            command.env("BV5_BALANCED", if config.bv5_balanced { "1" } else { "0" });
            command.env("BV5_QUAD_FIRE", "1");
        }
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
    if config.preprocessing_mode == RecipePreprocessingMode::Product2223 {
        command.env("PROD_PRESET", &config.production_preset);
        match &config.post_fragment {
            Some(value) => {
                command.env("PROD_POST_FRAGMENT", value);
            }
            None => {
                command.env_remove("PROD_POST_FRAGMENT");
            }
        }
    } else {
        command.env_remove("PROD_PRESET");
        command.env_remove("PROD_POST_FRAGMENT");
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
    println!(
        "[gss] preprocessing mode: {}",
        config.preprocessing_mode.as_str()
    );
    if config.preprocessing_mode == RecipePreprocessingMode::Product2223 {
        println!("[gss] production preset: {}", config.production_preset);
        println!(
            "[gss] post fragmentation: {}",
            config.post_fragment.as_deref().unwrap_or("preset default")
        );
    }
    println!(
        "[gss] seed: {}",
        if config.calibration_seed.is_some() {
            "EXPLICIT FROM PRIVATE FILE (redacted; CALIBRATION ONLY)"
        } else {
            "CSPRNG for a fresh run / protected SEED file for resume"
        }
    );
    println!(
        "[gss] stage 1-2: source.gates={}",
        config.mcd.as_deref().unwrap_or("derived")
    );
    println!(
        "[gss] db_mixing (stage 3): target_size_factor={} hold_work_units={}",
        config.expand.as_deref().unwrap_or("2"),
        config.hold.as_deref().unwrap_or("27")
    );
    if let Some(size) = &config.min_block_size {
        println!(
            "[gss] stages 3-4: automatic pieces target_piece_gates={} threads={}",
            size,
            config.piece_threads.as_deref().unwrap_or("available CPUs")
        );
    } else {
        println!(
            "[gss] stages 3-4: pieces={} threads={}",
            config.pieces.as_deref().unwrap_or("1"),
            config.piece_threads.as_deref().unwrap_or("pieces+1")
        );
    }
    println!(
        "[gss] crossing (stage 5): target_size_factor={} width_penalty_base={} width_penalty_threshold={} size_tolerance_divisor={} move_attempts={}",
        config.xr.as_deref().unwrap_or("2"),
        config.xb.as_deref().unwrap_or("3"),
        config.xc.as_deref().unwrap_or("1"),
        config.xtdiv.as_deref().unwrap_or("25"),
        config.xmoves.as_deref().unwrap_or("6*target")
    );
    println!(
        "[gss] lifecycle: stop_after_stage={} rerun_from_stage={} build_binaries={}",
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
            "[gss] WARNING: database.allow_no_database_for_tests=true; no-database db_mixing is armed for plumbing tests only"
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

pub(crate) fn print_store(label: &str, store: &SourcedPath) {
    match &store.path {
        Some(path) => println!(
            "[gss] {label}: {} ({})",
            path.display(),
            store.source.label()
        ),
        None => println!("[gss] {label}: unset"),
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
