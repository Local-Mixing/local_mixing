//! Build preparation and stage-driver invocation.
use super::*;

pub(crate) fn run_inner(sub: &ArgMatches) -> Result<(), TdpError> {
    reject_inherited_environment(std::env::vars_os().map(|(key, _)| key))?;
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
    let legacy_resume = config.run_dir.join("SEED").exists();
    if legacy_resume {
        return Err(TdpError::config(format!(
            "{} was created by recipe v9 (a one-line SEED file); resume it with the checkout that created it; the original run is unchanged",
            config.run_dir.display()
        )));
    }
    if config.adopt_existing_run {
        return Err(TdpError::config(format!(
            "pre-wrapper run adoption has been retired; use a fresh run.directory or the checkout that created {}; the original run is unchanged",
            config.run_dir.display()
        )));
    }
    if !managed_resume && config.run_dir.join("SEEDS").exists() {
        return Err(TdpError::config(format!(
            "{} has a SEEDS file but no managed manifest (a standalone scripts/tdp_gen.sh run, or an interrupted start); use a fresh run.directory (remove SEEDS only if nothing else was written)",
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
    // Before any build, SEEDS or manifest: scripts/tdp_gen.sh needs a current
    // bash, and macOS ships 3.2 as /bin/bash (and often first on PATH).
    let bash = resolve_bash()?;
    let host_target = rustc_host_triple()?;
    let binary_dir = config.build_target_dir.join(&host_target).join("release");
    print_resolved(&config_path, &config, &binary_dir);
    validate_external_paths(&config).map_err(TdpError::config)?;
    let script = repo_root.join("scripts/tdp_gen.sh");
    if managed_resume {
        let fingerprints = run_binary_fingerprints(&binary_dir)?;
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
            &bash,
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
        println!("[tdp_gen] building the TDP_GEN release executables (incremental)");
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
            .env_remove("CARGO_BUILD_TARGET");
        for name in PRODUCTION_BINARIES {
            build.arg("--bin").arg(name);
        }
        let status = build
            .status()
            .map_err(|error| TdpError::io(format!("cannot start cargo build: {error}")))?;
        if !status.success() {
            return Err(TdpError::child("cargo build", status));
        }
        // The stage-2 span report (val_chain.sh runs it on every chain) needs
        // hmap_affine, a security-tools binary: a separate build, so the
        // feature never reaches the production binaries.
        println!("[tdp_gen] building the stage-2 span report tool ({REPORT_BINARY})");
        let cargo = std::env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"));
        let status = ProcessCommand::new(cargo)
            .current_dir(&repo_root)
            .args(report_build_args(&config.build_target_dir, &host_target))
            .env_remove("CARGO_BUILD_TARGET")
            .status()
            .map_err(|error| TdpError::io(format!("cannot start cargo build: {error}")))?;
        if !status.success() {
            return Err(TdpError::child("cargo build (hmap_affine)", status));
        }
    }

    if !managed_resume {
        // §8.2: the private SEEDS file exists before the manifest.
        create_seeds_file(&config)?;
        prepare_recipe_manifest(&config, &binary_dir, &script)?;
    } else {
        check_seeds_file(&config)?;
    }

    println!(
        "[tdp_gen] starting {} (bash {}.{}: {})",
        script.display(),
        bash.version.0,
        bash.version.1,
        bash.path.display()
    );
    // A failed stage 6 leaves a marker naming the sub-step (scripts/tdp_gen.sh),
    // so a binary's exit-code meaning is attached only to that failure.
    let delivery_failed = config.run_dir.join("delivery.failed");
    match fs::remove_file(&delivery_failed) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(TdpError::io(format!(
                "cannot remove {}: {error}",
                delivery_failed.display()
            )));
        }
    }
    let mut child = ProcessCommand::new(&bash.path);
    child
        .current_dir(&repo_root)
        .arg(&script)
        .args(&script_args);
    configure_environment(&mut child, &config, &binary_dir);
    let status = child
        .status()
        .map_err(|error| TdpError::io(format!("cannot start bash: {error}")))?;
    if !status.success() {
        let code = status.code().unwrap_or(1);
        let mut error = TdpError::child("TDP_GEN pipeline", status);
        match stage_failure(&delivery_failed) {
            Some((failed, DeliveryStep::Packing)) if failed == code => {
                let meaning = crate::stages::packing::types::Stage4Error::meaning(code);
                error.message = format!(
                    "{} (stage 6 delivery, tdp_packing: {meaning})",
                    error.message
                );
            }
            Some((failed, DeliveryStep::Unpacking)) if failed == code => {
                let meaning = unpacking_exit_meaning(code);
                error.message = format!(
                    "{} (stage 6 delivery, tdp_unpacking --refine: {meaning})",
                    error.message
                );
            }
            _ => {}
        }
        return Err(error);
    }
    let stages = config.stages_text();
    // The stage's final output and the NOT_RECOMMENDED marker next to it:
    // the packer's deliverable under anf_only, else the refined deliverable
    // and the resynthesized circuit in unpacking/.
    let last_dir = config.run_dir.join(if config.unpacking_on() {
        "unpacking"
    } else {
        "packing"
    });
    let not_recommended = config.stage_on(6) && last_dir.join("NOT_RECOMMENDED").exists();
    let last = if config.unpacking_on() {
        Some(last_dir.join("unpacked.mpmct1"))
    } else if config.stage_on(6) {
        Some(last_dir.join("deliverable.tdp4"))
    } else {
        None
    };
    if let Some(last) = last {
        if not_recommended {
            eprintln!(
                "[tdp_gen] stages {stages} finished with an ALLOWED FAILURE: {} is NOT a recommended deliverable (see packing/packing.report and unpacking/unpacking.report)",
                last.display()
            );
        } else {
            println!("[tdp_gen] stages {stages} complete: {}", last.display());
        }
    }
    Ok(())
}

/// `tdp_unpacking` exit codes (the unpacking sub-step of stage 6;
/// docs/STAGE7_UNPACKING.md), as scripts/tdp_gen.sh reports them: the
/// binary's own table.
pub(crate) fn unpacking_exit_meaning(code: i32) -> &'static str {
    crate::stages::unpacking::types::Stage7Error::meaning(code)
}

/// Which sub-step of the delivery stage failed (`step=` in `delivery.failed`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DeliveryStep {
    Packing,
    Unpacking,
}

/// The oldest bash scripts/tdp_gen.sh runs under. Associative arrays and
/// `mapfile` need 4.0; 5.1 is the first release that feeds a here-string
/// smaller than the pipe buffer through a pipe instead of a temporary file
/// under TMPDIR, and the seed hand-over (`3<<< "$GEN_SEED"`) relies on that
/// to keep the seeds off disk. tdp_gen.sh checks the same bound.
pub(crate) const MIN_BASH: (u32, u32) = (5, 1);

pub(crate) struct Bash {
    pub(crate) path: PathBuf,
    pub(crate) version: (u32, u32),
}

/// Where a bash is looked for: every absolute PATH entry in order, then the
/// usual install locations (Homebrew on Apple silicon and Intel, the system).
pub(crate) fn bash_candidates(path: Option<&OsStr>) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = path
        .map(|p| {
            std::env::split_paths(p)
                .filter(|dir| dir.is_absolute())
                .map(|dir| dir.join("bash"))
                .collect()
        })
        .unwrap_or_default();
    for fixed in [
        "/opt/homebrew/bin/bash",
        "/usr/local/bin/bash",
        "/bin/bash",
        "/usr/bin/bash",
    ] {
        let fixed = PathBuf::from(fixed);
        if !out.contains(&fixed) {
            out.push(fixed);
        }
    }
    out
}

/// `MAJOR.MINOR` as printed by `echo "${BASH_VERSINFO[0]}.${BASH_VERSINFO[1]}"`.
pub(crate) fn parse_bash_version(text: &str) -> Option<(u32, u32)> {
    let (major, minor) = text.trim().split_once('.')?;
    Some((major.parse().ok()?, minor.parse().ok()?))
}

/// The first candidate bash at least [`MIN_BASH`]; an error naming every
/// bash found (and its version) when none is.
pub(crate) fn resolve_bash() -> Result<Bash, TdpError> {
    let mut seen = Vec::new();
    for candidate in bash_candidates(std::env::var_os("PATH").as_deref()) {
        if !candidate.is_file() {
            continue;
        }
        let output = ProcessCommand::new(&candidate)
            .env_clear()
            .arg("-c")
            .arg("echo \"${BASH_VERSINFO[0]}.${BASH_VERSINFO[1]}\"")
            .output();
        let version = output
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| parse_bash_version(&String::from_utf8_lossy(&o.stdout)));
        match version {
            Some(version) if version >= MIN_BASH => {
                return Ok(Bash {
                    path: candidate,
                    version,
                });
            }
            Some((major, minor)) => seen.push(format!("{} ({major}.{minor})", candidate.display())),
            None => seen.push(format!("{} (version unknown)", candidate.display())),
        }
    }
    Err(TdpError::config(format!(
        "scripts/tdp_gen.sh needs bash >= {}.{} (associative arrays; seed here-strings through a pipe, not a temporary file); found {}; install a current bash (macOS: brew install bash) and put it on PATH; nothing was built or written",
        MIN_BASH.0,
        MIN_BASH.1,
        if seen.is_empty() {
            "none".to_owned()
        } else {
            seen.join(", ")
        }
    )))
}

/// The exit code and sub-step a failed stage 6 recorded in
/// `$RUN/delivery.failed` (`exit=<code>` and `step=packing|unpacking`), if
/// that file exists and is well formed.
pub(crate) fn stage_failure(marker: &Path) -> Option<(i32, DeliveryStep)> {
    let meta = fs::metadata(marker).ok()?;
    if !meta.is_file() || meta.len() > 64 {
        return None;
    }
    let text = fs::read_to_string(marker).ok()?;
    let code = text
        .lines()
        .find_map(|line| line.strip_prefix("exit="))
        .and_then(|code| code.trim().parse().ok())?;
    let step = match text
        .lines()
        .find_map(|line| line.strip_prefix("step="))
        .map(str::trim)
    {
        Some("packing") => DeliveryStep::Packing,
        Some("unpacking") => DeliveryStep::Unpacking,
        _ => return None,
    };
    Some((code, step))
}

/// `cargo build` arguments for the report binary (after `cargo`).
pub(crate) fn report_build_args(target_dir: &Path, host_target: &str) -> Vec<OsString> {
    vec![
        OsString::from("build"),
        OsString::from("--release"),
        OsString::from("--locked"),
        OsString::from("--target-dir"),
        target_dir.as_os_str().to_owned(),
        OsString::from("--target"),
        OsString::from(host_target),
        OsString::from("--features"),
        OsString::from("security-tools"),
        OsString::from("--bin"),
        OsString::from(REPORT_BINARY),
    ]
}

/// Draw `[0, MAX_SEED]` uniformly from the OS CSPRNG (rejection sampling).
fn os_seed() -> Result<u64, TdpError> {
    use rand::TryRngCore;
    loop {
        let v = rand::rngs::OsRng
            .try_next_u64()
            .map_err(|e| TdpError::io(format!("OS random source failed: {e}")))?
            >> 1;
        if v <= MAX_SEED {
            return Ok(v);
        }
    }
}

/// The five run seeds (§8.2): gen (stages 1+2), mix (3), split (4), cross (5,
/// used only when stage 5 starts from a circuit file) and unpack (6: the
/// delivery seed, read by `tdp_unpacking run --refine`; the name in SEEDS
/// is unchanged so format-3 files resume).
#[derive(Clone, Copy)]
pub(crate) struct RunSeeds {
    pub(crate) gen_seed: u64,
    pub(crate) mix: u64,
    pub(crate) split: u64,
    pub(crate) cross: u64,
    pub(crate) unpack: u64,
}

/// The SEEDS text (format 3, §8.2).
pub(crate) fn seeds_text(seeds: RunSeeds, run_key: &[u8; 32], source: &str) -> String {
    let key: String = run_key.iter().map(|b| format!("{b:02x}")).collect();
    format!(
        "seeds_format=3\ngen_seed={}\nmix_seed={}\nsplit_seed={}\ncross_seed={}\nunpack_seed={}\nrun_key={key}\nseed_source={source}\n",
        seeds.gen_seed, seeds.mix, seeds.split, seeds.cross, seeds.unpack
    )
}

/// Create `$RUN/SEEDS` (create_new, mode 0600) before anything else is
/// written to the run directory. The values are read from this file by
/// scripts/tdp_gen.sh and handed to gen and the mixer on inherited file
/// descriptors (`@/dev/fd/N`, `--seed-file /dev/fd/3`): never on argv (ps),
/// never in the environment, and never printed: gen's and the mixer's banners
/// in stage12.log and stage3.log show `(file)` in their place. The private
/// SEEDS file is the only on-disk copy (plus, for a calibration run, the
/// operator's own seed file).
pub(crate) fn create_seeds_file(config: &ResolvedConfig) -> Result<(), TdpError> {
    use rand::TryRngCore;
    fs::create_dir_all(&config.run_dir).map_err(|error| {
        TdpError::io(format!(
            "cannot create run_dir {}: {error}",
            config.run_dir.display()
        ))
    })?;
    let path = config.run_dir.join("SEEDS");
    let (seeds, source) = match config.seeds {
        SeedPlan::Random => (
            RunSeeds {
                gen_seed: os_seed()?,
                mix: os_seed()?,
                split: os_seed()?,
                cross: os_seed()?,
                unpack: os_seed()?,
            },
            "random",
        ),
        SeedPlan::Explicit {
            gen_seed,
            mix_seed,
            split_seed,
            cross_seed,
            unpack_seed,
        } => (
            RunSeeds {
                gen_seed,
                mix: mix_seed,
                split: match split_seed {
                    Some(s) => s,
                    None => os_seed()?,
                },
                cross: match cross_seed {
                    Some(s) => s,
                    None => os_seed()?,
                },
                unpack: match unpack_seed {
                    Some(s) => s,
                    None => os_seed()?,
                },
            },
            "explicit",
        ),
    };
    let mut key = [0u8; 32];
    rand::rngs::OsRng
        .try_fill_bytes(&mut key)
        .map_err(|e| TdpError::io(format!("OS random source failed: {e}")))?;
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(&path)
        .map_err(|error| TdpError::io(format!("cannot create {}: {error}", path.display())))?;
    file.write_all(seeds_text(seeds, &key, source).as_bytes())
        .and_then(|_| file.sync_all())
        .map_err(|error| TdpError::io(format!("cannot write {}: {error}", path.display())))
}

/// On resume, SEEDS must exist, parse, be private, and agree with an explicit
/// seed file (compared here, never printed). Format 3 is the current one; a
/// format-2 file (the four-stage driver's) is parsed too for completeness,
/// although a managed resume never meets one (recipe v10 manifests are
/// refused before this check; scripts/tdp_gen.sh upgrades them standalone).
pub(crate) fn check_seeds_file(config: &ResolvedConfig) -> Result<(), TdpError> {
    let path = config.run_dir.join("SEEDS");
    let text = fs::read_to_string(&path).map_err(|error| {
        TdpError::config(format!(
            "this run has no readable {} ({error}); use a fresh run.directory",
            path.display()
        ))
    })?;
    let lines: Vec<&str> = text.lines().collect();
    let field = |i: usize, key: &str| {
        lines
            .get(i)
            .and_then(|l| l.strip_prefix(key))
            .filter(|v| !v.is_empty())
    };
    let seed_at = |i: usize, key: &str| -> Option<u64> {
        field(i, key)?.parse().ok().filter(|v| *v <= MAX_SEED)
    };
    // (gen, mix, [split, cross, unpack]): the last three are absent in format 2.
    let parsed = (|| {
        let (names, key_at): (&[&str], usize) = match (lines.len(), lines.first().copied()) {
            (8, Some("seeds_format=3")) => (
                &[
                    "gen_seed=",
                    "mix_seed=",
                    "split_seed=",
                    "cross_seed=",
                    "unpack_seed=",
                ],
                6,
            ),
            (5, Some("seeds_format=2")) => (&["gen_seed=", "mix_seed="], 3),
            _ => return None,
        };
        let mut seeds = Vec::new();
        for (i, name) in names.iter().enumerate() {
            seeds.push(seed_at(i + 1, name)?);
        }
        let k = field(key_at, "run_key=")?;
        if k.len() != 64
            || !k
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return None;
        }
        matches!(
            field(key_at + 1, "seed_source="),
            Some("random" | "explicit")
        )
        .then_some(seeds)
    })();
    let seeds = parsed.ok_or_else(|| {
        TdpError::config(format!(
            "{} is malformed (seeds_format=3 expected)",
            path.display()
        ))
    })?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = fs::metadata(&path)
            .map(|m| m.permissions().mode())
            .unwrap_or(0);
        if mode & 0o077 != 0 {
            return Err(TdpError::config(format!(
                "{} must not be accessible by group/other (chmod 600)",
                path.display()
            )));
        }
    }
    if let SeedPlan::Explicit {
        gen_seed,
        mix_seed,
        split_seed,
        cross_seed,
        unpack_seed,
    } = config.seeds
    {
        let want = [
            Some(gen_seed),
            Some(mix_seed),
            split_seed,
            cross_seed,
            unpack_seed,
        ];
        let conflict = want
            .iter()
            .zip(seeds.iter())
            .any(|(w, have)| w.is_some_and(|w| w != *have));
        if conflict {
            return Err(TdpError::config(format!(
                "calibration.seed_file conflicts with the existing {}; use the original seeds or a fresh run.directory",
                path.display()
            )));
        }
    }
    Ok(())
}

pub(crate) fn validate_current_recipe(config: &ResolvedConfig) -> Result<(), TdpError> {
    match config.preprocessing_mode {
        PreprocessingMode::EmbeddedMasking => {
            let k = config.knobs.mask_pair_wires;
            if !(2..=64).contains(&k) {
                return Err(TdpError::config(
                    "preprocessing.mask_pair_wires must be in 2..64 for new runs",
                ));
            }
            let cycle_width = k - k % 2 + usize::from(config.knobs.balanced_masks);
            if 2 * config.n <= cycle_width {
                return Err(TdpError::config(
                    "Embedded masking requires 2 * source.wires > preprocessing.mask_pair_wires rounded down to an even number + balanced_masks; reduce preprocessing.mask_pair_wires or increase source.wires",
                ));
            }
        }
        PreprocessingMode::Nonlinear291 => {}
        PreprocessingMode::ThreeMask => {
            return Err(TdpError::config(
                "preprocessing.mode three-mask is not wired into the managed TDP driver yet; run gen_sandwich_gadget directly",
            ));
        }
    }
    Ok(())
}

/// §11.5: an inherited variable that could change output is refused (the
/// same rule as scripts/tdp_gen.sh, from the same registry).
pub(crate) fn reject_inherited_environment(
    keys: impl IntoIterator<Item = OsString>,
) -> Result<(), TdpError> {
    let names: Vec<String> = keys
        .into_iter()
        .map(|k| k.to_string_lossy().into_owned())
        .collect();
    let refused = crate::tdp::env_registry::refused_inherited(&names);
    if refused.is_empty() {
        Ok(())
    } else {
        Err(TdpError::config(crate::tdp::env_registry::refusal_message(
            &refused,
        )))
    }
}

pub(crate) fn canonical_script_flag(flag: &str) -> &str {
    match flag {
        "-n" => "--source-wires",
        "-o" => "--run-directory",
        "--mcd" => "--source-gates",
        "--expand" => "--db-mixing-target-size-factor",
        "--hold" => "--db-mixing-hold-work-units",
        "--stop-after" => "--run-stop-after-stage",
        "--force-from" => "--run-rerun-from-stage",
        "--skip-stages" => "--run-skip-stages",
        "--xr" => "--crossing-target-size-factor",
        "--xb" => "--crossing-width-penalty-base",
        "--xc" => "--crossing-width-penalty-threshold",
        "--xtdiv" => "--crossing-size-tolerance-divisor",
        "--xmf" => "--crossing-move-budget-factor",
        "--xmoves" => "--crossing-move-attempts",
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
    let mut push = |flag: &str, value: String| {
        args.push(OsString::from(flag));
        args.push(OsString::from(value));
    };
    if config.preprocessing_mode == PreprocessingMode::EmbeddedMasking {
        push("--preprocessing-preset", config.preset.name().to_owned());
        if config.preset == Preset::Custom {
            for (_, flag, value) in config.knobs.fields() {
                push(&format!("--preprocessing-{flag}"), value);
            }
        }
    }
    match config.mcd_rule {
        SourceGatesRule::Validated => {}
        SourceGatesRule::Explicit => push("--mcd", config.mcd.to_string()),
        SourceGatesRule::Formula => push("--source-gates-rule", "formula".into()),
    }
    if let Some(path) = &config.source_path {
        if config.mcd_rule == SourceGatesRule::Validated {
            push("--mcd", config.mcd.to_string());
        }
        args.push(OsString::from("--source-path"));
        args.push(path.as_os_str().to_owned());
    }
    let mut push = |flag: &str, value: String| {
        args.push(OsString::from(flag));
        args.push(OsString::from(value));
    };
    if let Schedule::Custom {
        growth,
        hold,
        expand,
    } = &config.schedule
    {
        push("--db-mixing-schedule", "custom".into());
        push("--db-mixing-growth-work-units", growth.clone());
        push("--hold", hold.clone());
        push("--expand", expand.clone());
    }
    match config.frozen_filter {
        FrozenFilter::Auto => {}
        FrozenFilter::On => push("--database-lookup-miss-filter", "on".into()),
        FrozenFilter::Off => push("--database-lookup-miss-filter", "off".into()),
    }
    for (flag, value, default) in config.packing.flags() {
        if value != default {
            push(flag, value);
        }
    }
    let c = &config.crossing;
    push_optional_arg(&mut args, "--xr", c.xr.as_deref());
    push_optional_arg(&mut args, "--xb", c.xb.as_deref());
    push_optional_arg(&mut args, "--xc", c.xc.as_deref());
    push_optional_arg(&mut args, "--xtdiv", c.xtdiv.as_deref());
    push_optional_arg(&mut args, "--xmf", c.xmf.as_deref());
    push_optional_arg(&mut args, "--xmoves", c.xmoves.as_deref());
    let mut push = |flag: &str, value: String| {
        args.push(OsString::from(flag));
        args.push(OsString::from(value));
    };
    for (flag, value, default) in config.delivery.flags() {
        if value != default {
            push(flag, value);
        }
    }
    if !config.skip_stages.is_empty() {
        push("--skip-stages", config.skip_stages_text());
    }
    if let Some(t) = &config.threads {
        push("--threads", t.clone());
    }
    if config.qc_enabled {
        push("--leakage-repair-seed", config.qc_seed.to_string());
    }
    let mut args_tail: Vec<OsString> = Vec::new();
    push_optional_arg(&mut args_tail, "--stop-after", config.stop_after.as_deref());
    push_optional_arg(&mut args_tail, "--force-from", config.force_from.as_deref());
    push_optional_arg(&mut args_tail, "--pieces", config.pieces.as_deref());
    push_optional_arg(
        &mut args_tail,
        "--min-block-size",
        config.min_block_size.as_deref(),
    );
    push_optional_arg(
        &mut args_tail,
        "--piece-threads",
        config.piece_threads.as_deref(),
    );
    args.extend(args_tail);
    // Only option positions are translated; path/value strings may equal an alias.
    for option in args.iter_mut().step_by(2) {
        if let Some(name) = option.to_str() {
            *option = OsString::from(canonical_script_flag(name));
        }
    }
    // Switches (no value) go last so the pairwise translation above holds.
    if config.qc_enabled {
        args.push(OsString::from("--leakage-repair"));
        if let Some(reference) = &config.qc_reference {
            args.push(OsString::from("--leakage-repair-reference"));
            args.push(reference.as_os_str().to_owned());
        }
    }
    if config.packing.allow_floor_miss {
        args.push(OsString::from("--packing-allow-floor-miss"));
    }
    if config.packing.allow_over_budget {
        args.push(OsString::from("--packing-allow-over-budget"));
    }
    if config.delivery.anf_only {
        args.push(OsString::from("--delivery-anf-only"));
    }
    args
}

pub(crate) fn push_optional_arg(args: &mut Vec<OsString>, flag: &str, value: Option<&str>) {
    if let Some(value) = value {
        args.push(OsString::from(flag));
        args.push(OsString::from(value));
    }
}

/// §11.5 allow-list: the bash driver starts from an EMPTY environment plus
/// PATH/HOME/TMPDIR, the locale, the binary directory, the store path and the
/// plumbing-test switch. Every recipe setting travels as a flag; bash builds
/// each stage binary's environment itself (`stage_exec`).
pub(crate) fn configure_environment(
    command: &mut ProcessCommand,
    config: &ResolvedConfig,
    binary_dir: &Path,
) {
    command.env_clear();
    for key in crate::tdp::env_registry::ALLOW_ENV {
        if let Some(value) = std::env::var_os(key) {
            command.env(key, value);
        }
    }
    command.env("LANG", "C");
    command.env("LC_ALL", "C");
    command.env("TDP_BIN_DIR", binary_dir);
    set_or_remove_env(command, "FROZEN_DB_DIR", config.frozen_db.path.as_deref());
    if config.allow_empty_store {
        command.env("TDP_GEN_ALLOW_EMPTY_STORE", "1");
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
    if config.frozen_db.path.is_some() {
        println!(
            "[tdp_gen] regular frozen store fingerprint: {}; pinned_recipe={}",
            store_fingerprint_text(config),
            pinned_recipe(config)
        );
    }
    println!(
        "[tdp_gen] store filter: {}",
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
        "[tdp_gen] seeds: {}",
        match config.seeds {
            SeedPlan::Explicit { .. } =>
                "EXPLICIT FROM PRIVATE FILE (redacted; CALIBRATION ONLY) -> $RUN/SEEDS",
            SeedPlan::Random => "CSPRNG for a fresh run -> private $RUN/SEEDS (kept on resume)",
        }
    );
    println!(
        "[tdp_gen] stage 1-2: source.gates={} ({}{}) preset={}",
        config.mcd,
        config.mcd_rule.name(),
        if config.mcd_rule == SourceGatesRule::Formula {
            ", UNVALIDATED SIZE"
        } else {
            ""
        },
        config.preset.name()
    );
    match &config.schedule {
        Schedule::Validated => println!(
            "[tdp_gen] db_mixing (stage 3): schedule validated (profile 2,2,2,2,2, moves int(G*2*2*1.3))"
        ),
        Schedule::Custom {
            growth,
            hold,
            expand,
        } => println!(
            "[tdp_gen] db_mixing (stage 3): schedule custom growth={growth} hold={hold} target_size_factor={expand}"
        ),
    }
    println!(
        "[tdp_gen] stages: {} (skipped: {})",
        config.stages_text(),
        config.skip_stages_text()
    );
    println!(
        "[tdp_gen] splitting (stage 4): archive-era flags (--split --split-stop --p-join 0.8 --split-reach-k 2 --split-fail-limit 100 --split-canaries 256 --k-max 12)"
    );
    println!(
        "[tdp_gen] crossing (stage 5): {}",
        config
            .crossing
            .fields()
            .iter()
            .map(|(key, _, value)| format!("{key}={value}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!(
        "[tdp_gen] delivery (stage 6): {}",
        if config.delivery.anf_only {
            "anf_only=true (the packer's deliverable; no resynthesis)".to_owned()
        } else {
            format!(
                "unpacking with lattice refinement: time_limit={} restarts={} beam={} max_rounds={}",
                config.delivery.time_limit,
                config.delivery.restarts,
                config.delivery.beam,
                config.delivery.max_rounds
            )
        }
    );
    println!(
        "[tdp_gen] packing (stage 6, first sub-step): {}",
        if config.packing.is_default() {
            "default (G=512, split budget 65536, cpython39 caps)".to_owned()
        } else {
            config
                .packing
                .flags()
                .into_iter()
                .filter(|(_, v, d)| v != d)
                .map(|(f, v, _)| format!("{f} {v}"))
                .chain(
                    config
                        .packing
                        .allow_floor_miss
                        .then(|| "--packing-allow-floor-miss (NOT recommended)".to_owned()),
                )
                .chain(
                    config
                        .packing
                        .allow_over_budget
                        .then(|| "--packing-allow-over-budget (NOT recommended)".to_owned()),
                )
                .collect::<Vec<_>>()
                .join(" ")
        }
    );
    if let Some(size) = &config.min_block_size {
        println!(
            "[tdp_gen] stage 3: automatic pieces target_piece_gates={} threads={}",
            size,
            config.piece_threads.as_deref().unwrap_or("available CPUs")
        );
    } else {
        println!(
            "[tdp_gen] stage 3: pieces={} threads={}",
            config.pieces.as_deref().unwrap_or("1"),
            config.piece_threads.as_deref().unwrap_or("pieces+1")
        );
    }
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
    bash: &Bash,
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
            OsString::from("tdp_packing"),
            OsString::from("--bin"),
            OsString::from("tdp_unpacking"),
        ];
        println!("[tdp_gen] would build: {}", render_command(&build));
        let mut report = vec![OsString::from("cargo")];
        report.extend(report_build_args(&config.build_target_dir, host_target));
        println!("[tdp_gen] would build: {}", render_command(&report));
    } else if managed_resume && config.build_release {
        println!(
            "[tdp_gen] release build: would be skipped for resume; the fingerprinted binaries would be reused"
        );
    } else {
        println!("[tdp_gen] release build: skipped by configuration");
    }
    let mut invocation = vec![
        bash.path.clone().into_os_string(),
        script.as_os_str().to_owned(),
    ];
    invocation.extend_from_slice(args);
    println!("[tdp_gen] would run: {}", render_command(&invocation));
}

/// Render argv for display, redacting the value after EVERY flag whose name
/// contains `seed` (except `--seed-file`, whose path is not secret; §8.4).
pub(crate) fn render_command(args: &[OsString]) -> String {
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
        let flag = canonical_script_flag(text.as_ref());
        if flag.starts_with('-') && flag.contains("seed") && flag != "--seed-file" {
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
