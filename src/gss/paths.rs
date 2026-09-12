//! Repository discovery, path separation and stage prerequisites.
use super::*;

pub(crate) fn validate_external_paths(config: &ResolvedConfig) -> Result<(), String> {
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
    let db_mixing_artifact = if config.recipe_version >= 6 {
        "db_mixing.mpmct1"
    } else {
        "phaseA.mpmct1"
    };
    let existing_db_mixing = fs::metadata(config.run_dir.join(db_mixing_artifact))
        .is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0);
    let existing_gadget = fs::metadata(config.run_dir.join("gss.mpmct1"))
        .is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0);
    let stage_three_will_run =
        stop_after >= 3 && (force_from <= 3 || !existing_gadget || !existing_db_mixing);
    if stage_three_will_run {
        if let Some(path) = &config.frozen_db.path {
            validate_frozen_store(path, "database.regular_dir")?;
            validate_requested_filter(path, "database.regular_dir", config.frozen_filter)?;
        }
        if let Some(path) = &config.frozen_curated.path {
            validate_frozen_store(path, "database.curated_dir")?;
            validate_requested_filter(path, "database.curated_dir", config.frozen_filter)?;
        }
    }
    if stage_three_will_run && config.allow_empty_store && config.frozen_db.path.is_some() {
        return Err(
            "database.allow_no_database_for_tests=true conflicts with a configured database.regular_dir; clear one so the db_mixing mode is unambiguous"
                .to_owned(),
        );
    }
    if stage_three_will_run
        && config.frozen_db.path.is_none()
        && config.frozen_curated.path.is_some()
    {
        return Err(
            "database.curated_dir cannot be used without database.regular_dir; the curated store is a cascade ahead of the regular store"
                .to_owned(),
        );
    }
    if stage_three_will_run && config.frozen_db.path.is_none() && !config.allow_empty_store {
        return Err(
            "stage 3 needs database.regular_dir (or FROZEN_DB_DIR in the environment); `database.allow_no_database_for_tests = true` is only for plumbing tests"
                .to_owned(),
        );
    }
    Ok(())
}

pub(crate) fn validate_requested_filter(
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
            "database.lookup_miss_filter=on requires a nonempty {}/filters.bin for {label}",
            path.display()
        ));
    }
    Ok(())
}

pub(crate) fn validate_frozen_store(path: &Path, label: &str) -> Result<(), String> {
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

pub(crate) fn validate_run_destination(repo_root: &Path, run_dir: &Path) -> Result<(), String> {
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

pub(crate) fn validate_build_target(repo_root: &Path, target_dir: &Path) -> Result<(), String> {
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

pub(crate) fn validate_path_separation(
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
        paths.push(("database.regular_dir", resolve_existing_prefix(path)));
    }
    if let Some(path) = frozen_curated {
        paths.push(("database.curated_dir", resolve_existing_prefix(path)));
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
pub(crate) fn resolve_existing_prefix(path: &Path) -> std::io::Result<PathBuf> {
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

pub(crate) fn fresh_default_run_dir(repo_root: &Path, n: usize, tag: &str) -> PathBuf {
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

pub(crate) fn default_run_tag() -> String {
    let epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{epoch}-{}", std::process::id())
}

pub(crate) fn resolve_path(repo_root: &Path, path: &Path) -> PathBuf {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        repo_root.join(path)
    };
    lexical_normalize(&absolute)
}

pub(crate) fn lexical_normalize(path: &Path) -> PathBuf {
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

pub(crate) fn rustc_host_triple() -> Result<String, GssError> {
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

pub(crate) fn find_repo_root() -> Result<PathBuf, GssError> {
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

pub(crate) fn is_repo_root(path: &Path) -> bool {
    path.join("Cargo.toml").is_file()
        && path.join("configs/gss.toml").is_file()
        && path.join("scripts/gss_mix.sh").is_file()
}
