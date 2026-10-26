//! Run provenance and executable fingerprints.
use super::*;

pub(crate) const MAX_MANIFEST_BYTES: u64 = 64 * 1024;
pub(crate) const CURRENT_RECIPE_VERSION: u8 = 13;
/// The managed binaries, fingerprinted in the manifest and built for fresh runs.
pub(crate) const PRODUCTION_BINARIES: [&str; 5] = [
    "gen_sandwich_gadget",
    "circuit_mixer",
    "tdp_packing",
    "tdp_unpacking",
    "tdp_adaptive",
];
/// The report-only binary behind val_chain.sh's stage-2 span check
/// (`hmap_affine --exposure ... SPAN_INTERSECT`, $RUN/stage2_span.*). Built
/// for fresh runs (it needs the `security-tools` feature) and fingerprinted
/// when present; without it the driver skips the report and the manifest says
/// `stage2_span_report=skipped`.
pub(crate) const REPORT_BINARY: &str = "hmap_affine";

pub(crate) fn prepare_recipe_manifest(
    config: &ResolvedConfig,
    binary_dir: &Path,
    script: &Path,
) -> Result<(), TdpError> {
    let fingerprints = run_binary_fingerprints(binary_dir)?;
    let script_hash = hash_file(script).map_err(|error| {
        TdpError::io(format!(
            "cannot fingerprint orchestrator {}: {error}",
            script.display()
        ))
    })?;
    let desired = recipe_manifest(config, &fingerprints, script_hash);
    fs::create_dir_all(&config.run_dir).map_err(|error| {
        TdpError::io(format!(
            "cannot create run_dir {}: {error}",
            config.run_dir.display()
        ))
    })?;
    let path = config.run_dir.join("tdp_command.conf");
    if path.exists() {
        return compare_recipe_manifest(&path, &desired);
    }

    match OpenOptions::new().write(true).create_new(true).open(&path) {
        Ok(mut file) => {
            file.write_all(desired.as_bytes()).map_err(|error| {
                TdpError::io(format!(
                    "cannot write recipe manifest {}: {error}",
                    path.display()
                ))
            })?;
            file.sync_all().map_err(|error| {
                TdpError::io(format!(
                    "cannot sync recipe manifest {}: {error}",
                    path.display()
                ))
            })?;
            println!("[tdp_gen] recipe manifest: {}", path.display());
            Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            compare_recipe_manifest(&path, &desired)
        }
        Err(error) => Err(TdpError::io(format!(
            "cannot create recipe manifest {}: {error}",
            path.display()
        ))),
    }
}

pub(crate) fn production_binary_fingerprints(
    binary_dir: &Path,
) -> Result<Vec<(&'static str, u128)>, TdpError> {
    PRODUCTION_BINARIES
        .into_iter()
        .map(|name| {
            let path = binary_dir.join(name);
            hash_file(&path)
                .map(|hash| (name, hash))
                .map_err(|error| {
                    TdpError::io(format!(
                        "cannot fingerprint production binary {}: {error}; set run.build_binaries=true or select the target directory containing all four binaries",
                        path.display()
                    ))
                })
        })
        .collect()
}

/// The production fingerprints plus the report binary's, when it is built.
pub(crate) fn run_binary_fingerprints(
    binary_dir: &Path,
) -> Result<Vec<(&'static str, u128)>, TdpError> {
    let mut fingerprints = production_binary_fingerprints(binary_dir)?;
    let report = binary_dir.join(REPORT_BINARY);
    if report.is_file() {
        let hash = hash_file(&report).map_err(|error| {
            TdpError::io(format!(
                "cannot fingerprint report binary {}: {error}",
                report.display()
            ))
        })?;
        fingerprints.push((REPORT_BINARY, hash));
    }
    Ok(fingerprints)
}

pub(crate) fn hash_file(path: &Path) -> std::io::Result<u128> {
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

pub(crate) fn compare_recipe_manifest(path: &Path, desired: &str) -> Result<(), TdpError> {
    let metadata = fs::metadata(path).map_err(|error| {
        TdpError::io(format!(
            "cannot inspect recipe manifest {}: {error}",
            path.display()
        ))
    })?;
    if !metadata.is_file() || metadata.len() > MAX_MANIFEST_BYTES {
        return Err(TdpError::config(format!(
            "recipe manifest {} must be a regular file no larger than {MAX_MANIFEST_BYTES} bytes",
            path.display()
        )));
    }
    let existing = fs::read_to_string(path).map_err(|error| {
        TdpError::io(format!(
            "cannot read recipe manifest {}: {error}",
            path.display()
        ))
    })?;
    let adopted_unverified = existing
        .lines()
        .any(|line| line == "adopted_unverified=true");
    let normalize = |text: &str| -> String {
        text.lines()
            .filter(|line| *line != "adopted_unverified=true")
            .map(|line| format!("{line}\n"))
            .collect()
    };
    let normalized_existing = normalize(&existing);
    let normalized_desired = normalize(desired);
    if normalized_existing == normalized_desired {
        println!("[tdp_gen] recipe manifest matches: {}", path.display());
        if adopted_unverified {
            eprintln!(
                "[tdp_gen] WARNING: this run was adopted without a verifiable pre-wrapper recipe"
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
    Err(TdpError::config(format!(
        "resolved recipe or production binaries differ from {}; refusing a mixed-provenance resume{}",
        path.display(),
        if differences.is_empty() {
            String::new()
        } else {
            format!(": {}", differences.join(", "))
        }
    )))
}

/// The stage-3 `FROZEN_FILTER` value scripts/tdp_gen.sh sets for this
/// config (D11): `auto` is 1 when the store has no filters.bin (the reference
/// situation: val_chain.sh's FROZEN_FILTER=1 on an unfiltered store) and 0
/// when it has one (no filtered/unfiltered neutrality comparison recorded,
/// FILTER_NEUTRALITY_RECORDED=0); without a store the variable is not set
/// (plumbing mode). The driver also records the value it set in
/// $RUN/stage3.settings.
pub(crate) fn resolved_frozen_filter(config: &ResolvedConfig) -> &'static str {
    let Some(store) = config.frozen_db.path.as_deref() else {
        return "unset";
    };
    match config.frozen_filter {
        FrozenFilter::On => "1",
        FrozenFilter::Off => "0",
        FrozenFilter::Auto => {
            let filtered =
                fs::metadata(store.join("filters.bin")).is_ok_and(|m| m.is_file() && m.len() > 0);
            if filtered { "0" } else { "1" }
        }
    }
}

/// [`store_fingerprint`] of the regular store the chosen runs used
/// (`frozen_m1_m11`: probes.bin, tables.bin and 256 shards, each with its
/// `.ok` checksum marker, no filters.bin).
pub(crate) const VALIDATED_REGULAR_STORE: u128 = 0xfd19_8367_7e4d_5b74_267d_3802_c9c3_4642;

/// An identity of a frozen store's contents that does not read its 298 GB of
/// shards: the name and size of every regular, non-hidden file in the
/// directory, and the text of each `*.ok` marker (the per-file checksum the
/// store was built with), in name order. A copy with other shards, sizes or
/// markers, or an added filters.bin, has another fingerprint. `None` when
/// the directory cannot be read.
pub(crate) fn store_fingerprint(store: &Path) -> Option<u128> {
    let mut entries: Vec<(String, u64)> = Vec::new();
    for entry in fs::read_dir(store).ok()? {
        let entry = entry.ok()?;
        let name = entry.file_name().into_string().ok()?;
        if name.starts_with('.') {
            continue;
        }
        let meta = fs::metadata(entry.path()).ok()?;
        if meta.is_file() {
            entries.push((name, meta.len()));
        }
    }
    entries.sort();
    let mut hasher = Xxh3::new();
    for (name, len) in entries {
        hasher.update(format!("{name}\t{len}\n").as_bytes());
        if name.ends_with(".ok") && len <= 4096 {
            hasher.update(&fs::read(store.join(&name)).ok()?);
            hasher.update(b"\n");
        }
    }
    Some(hasher.digest128())
}

/// The manifest's `frozen_db_fingerprint` value.
pub(crate) fn store_fingerprint_text(config: &ResolvedConfig) -> String {
    match config.frozen_db.path.as_deref() {
        None => "unset".to_owned(),
        Some(store) => match store_fingerprint(store) {
            Some(fp) => format!("{fp:032x}"),
            None => "unreadable".to_owned(),
        },
    }
}

/// The validation recipe this run reproduces, if any: val_chain.sh CONFIG=L8
/// or A as run for the chosen runs (vL64a/b, vL128a/b, vA64a, vA128a). Only
/// a run whose every recipe setting is that chain's: embedded masking with
/// preset L8 or A, the validated |C| for a generated source, schedule
/// validated in one piece, no leakage repair, default packing, stages 4 and
/// 5 off and stage 6 ending after the packer's deliverable
/// (`delivery.anf_only`: the chain had four stages and no resynthesis; the
/// unused unpacking parameters do not matter), and stage 3
/// on THE store those runs used ([`VALIDATED_REGULAR_STORE`]) unfiltered:
/// FROZEN_FILTER=1 on a store without filters.bin, which is how val_chain.sh
/// ran (a filtered run, `lookup_miss_filter = "on"` with filters.bin, was
/// never validated; D11). Anything else records `none`.
pub(crate) fn pinned_recipe(config: &ResolvedConfig) -> &'static str {
    pinned_recipe_for_store(config, VALIDATED_REGULAR_STORE)
}

/// [`pinned_recipe`] against a given reference-store fingerprint (tests).
pub(crate) fn pinned_recipe_for_store(config: &ResolvedConfig, reference: u128) -> &'static str {
    let reference_store = config.frozen_db.path.as_deref().is_some_and(|store| {
        !store.join("filters.bin").exists() && store_fingerprint(store) == Some(reference)
    });
    let matches_chain = config.preprocessing_mode == PreprocessingMode::EmbeddedMasking
        && config.mcd_rule == SourceGatesRule::Validated
        && config.source_path.is_none()
        && config.schedule == Schedule::Validated
        && config.pieces.as_deref().unwrap_or("1") == "1"
        && config.min_block_size.is_none()
        && !config.allow_empty_store
        && reference_store
        && resolved_frozen_filter(config) == "1"
        && !config.qc_enabled
        && config.packing.is_default()
        && config.source_construction == SourceConstruction::Random
        // the recipe's stages, not the lifecycle stop point: a stop before
        // stage 4 must not pin a run whose recipe includes stages 4-6
        && [4, 5].iter().all(|k| config.skip_stages.contains(k))
        && config.delivery.mode == DeliveryMode::Windowed
        && config.delivery.anf_only;
    match (matches_chain, config.preset) {
        (true, Preset::L8) => "val_chain-L8-20261007",
        (true, Preset::A) => "val_chain-A-20261007",
        _ => "none",
    }
}

/// The resolved recipe (§11.2): every value printed is the value the run
/// uses, never a hard-coded description. No seed, no seed-derived value and
/// no hash of C or of a private circuit appears here.
pub(crate) fn recipe_manifest(
    config: &ResolvedConfig,
    binaries: &[(&str, u128)],
    script_hash: u128,
) -> String {
    use std::fmt::Write as _;
    let path_or_unset = |path: &Option<PathBuf>| {
        path.as_deref()
            .map(|path| path.to_string_lossy().into_owned())
            .unwrap_or_else(|| "unset".to_owned())
    };
    let mut m = String::new();
    let mut kv = |k: &str, v: &str| {
        let _ = writeln!(m, "{k}={v}");
    };
    kv("tdp_command_recipe", &config.recipe_version.to_string());
    kv("pinned_recipe", pinned_recipe(config));
    kv("n", &config.n.to_string());
    kv("source_gates", &config.mcd.to_string());
    kv("source_gates_rule", config.mcd_rule.name());
    kv(
        "source",
        if config.source_path.is_some() {
            "path"
        } else {
            "generated"
        },
    );
    kv("source_path", &path_or_unset(&config.source_path));
    kv("source_construction", config.source_construction.name());
    kv("sandwich_variant", "classic");
    kv(
        "preprocessing_mode",
        config.preprocessing_mode.canonical_name(),
    );
    kv("preprocessing_preset", config.preset.name());
    if config.preprocessing_mode == PreprocessingMode::EmbeddedMasking {
        for (key, _, value) in config.knobs.fields() {
            kv(&format!("preprocessing_{key}"), &value);
        }
        kv("preprocessing_quadratic_fire", "1");
    }
    if config.preprocessing_mode == PreprocessingMode::ThreeMask {
        kv(
            "three_mask_params",
            if config.three_mask.is_default() {
                "default"
            } else {
                "custom"
            },
        );
        for (key, _, value, _) in config.three_mask.fields() {
            kv(&format!("three_mask_{key}"), &value);
        }
        kv("three_mask_outer_blocks", "L8");
    }
    kv("gadget_seed_rule", "gen");
    match &config.schedule {
        Schedule::Validated => {
            kv("db_mixing_schedule", "validated");
            kv("stage3_profile", "2,2,2,2,2");
            kv("stage3_moves_rule", "int(G*2*2*1.3)");
        }
        Schedule::Custom {
            growth,
            hold,
            expand,
        } => {
            kv("db_mixing_schedule", "custom");
            kv("stage3_growth_work_units", growth);
            kv("stage3_hold_work_units", hold);
            kv("stage3_target_size_factor", expand);
            kv("stage3_moves_rule", "int((n0+hold)*r*G*1.3)");
        }
    }
    kv("stage3_verify_every", "default(10000)");
    kv("stage3_report_every", "default(50000)");
    // Which stages the recipe runs (lifecycle stop/rerun points are not here).
    kv("skip_stages", &config.skip_stages_text());
    kv("stage4_split_args", "archive(ea2e516f)");
    for (key, _, value) in config.crossing.fields() {
        kv(&format!("crossing_{key}"), &value);
    }
    kv(
        "pieces",
        if config.min_block_size.is_some() {
            "auto"
        } else {
            config.pieces.as_deref().unwrap_or("1")
        },
    );
    if let Some(size) = &config.min_block_size {
        kv("min_block_size", size);
    }
    kv("frozen_db_dir", &path_or_unset(&config.frozen_db.path));
    kv("frozen_db_fingerprint", &store_fingerprint_text(config));
    kv(
        "frozen_filter",
        &format!(
            "{}({})",
            match config.frozen_filter {
                FrozenFilter::Auto => "auto",
                FrozenFilter::On => "on",
                FrozenFilter::Off => "off",
            },
            resolved_frozen_filter(config)
        ),
    );
    kv("frozen_regular_value_convention", "native");
    kv("allow_empty_store", &config.allow_empty_store.to_string());
    kv("env_policy", "allowlist-v1");
    kv("cache_mb", "lookup:2048,canon:256,xpoly:1024");
    kv("canon_caps", "unbounded");
    kv(
        "packing_params",
        if config.packing.is_default() {
            "default"
        } else {
            "custom"
        },
    );
    let p = &config.packing;
    kv("packing_window_gates", &p.window_gates.to_string());
    kv("packing_interval_cap", &p.interval_cap.to_string());
    kv("packing_monomial_budget", &p.monomial_budget.to_string());
    kv("packing_span_cap", &p.span_cap.to_string());
    kv("packing_split_budget", &p.split_budget.to_string());
    kv("packing_fix_iterations", &p.fix_iterations.to_string());
    kv(
        "packing_fallback_iterations",
        &p.fallback_iterations.to_string(),
    );
    kv("packing_anf_cap_semantics", &p.anf_cap_semantics);
    kv("packing_allow_floor_miss", &p.allow_floor_miss.to_string());
    kv(
        "packing_allow_over_budget",
        &p.allow_over_budget.to_string(),
    );
    let dl = &config.delivery;
    kv("delivery_mode", dl.mode.name());
    kv(
        "delivery_params",
        if dl.is_default() { "default" } else { "custom" },
    );
    kv("delivery_anf_only", &dl.anf_only.to_string());
    kv("delivery_time_limit", &dl.time_limit);
    kv("delivery_restarts", &dl.restarts.to_string());
    kv("delivery_beam", &dl.beam.to_string());
    kv("delivery_max_rounds", &dl.max_rounds.to_string());
    let ad = &config.adaptive;
    kv(
        "adaptive_params",
        if ad.is_default() { "default" } else { "custom" },
    );
    kv("adaptive_max_rounds", &ad.max_rounds.to_string());
    kv("adaptive_stop_fraction", &ad.stop_fraction);
    kv("adaptive_seeds_per_block", &ad.seeds_per_block.to_string());
    kv("adaptive_time_limit", &ad.time_limit);
    kv("adaptive_restarts", &ad.restarts.to_string());
    kv("adaptive_beam", &ad.beam.to_string());
    kv("adaptive_term_cap", &ad.term_cap.to_string());
    kv("adaptive_max_block_gates", &ad.max_block_gates.to_string());
    kv("adaptive_segments", &ad.segments.to_string());
    kv("adaptive_avoid_radius", &ad.avoid_radius.to_string());
    kv(
        "adaptive_refinement_levels",
        &ad.refinement_levels.to_string(),
    );
    kv(
        "adaptive_min_refine_gates",
        &ad.min_refine_gates.to_string(),
    );
    kv("adaptive_census_batches", &ad.census_batches.to_string());
    kv("adaptive_accept", &ad.accept);
    kv("adaptive_canonical_order", &ad.canonical_order.to_string());
    kv("seed_record", "SEEDS");
    kv(
        "seed_source",
        match config.seeds {
            SeedPlan::Random => "random",
            SeedPlan::Explicit { .. } => "explicit",
        },
    );
    kv("seed_bits", "63");
    kv("seeds_format", "3");
    kv("calibration_only", &config.calibration_only.to_string());
    kv("qc_enabled", &config.qc_enabled.to_string());
    kv("qc_seed", &config.qc_seed.to_string());
    kv("qc_reference", &path_or_unset(&config.qc_reference));
    for (name, fingerprint) in binaries {
        kv(
            &format!("binary_{name}_xxh3"),
            &format!("{fingerprint:032x}"),
        );
    }
    kv(
        "stage2_span_report",
        if binaries.iter().any(|(name, _)| *name == REPORT_BINARY) {
            REPORT_BINARY
        } else {
            "skipped"
        },
    );
    kv("script_tdp_gen_xxh3", &format!("{script_hash:032x}"));
    m
}

pub(crate) fn read_recipe_version(path: &Path) -> Result<u8, TdpError> {
    let meta = fs::metadata(path)
        .map_err(|e| TdpError::io(format!("cannot read {}: {e}", path.display())))?;
    if meta.len() > MAX_MANIFEST_BYTES {
        return Err(TdpError::config("recipe manifest is too large"));
    }
    let text = fs::read_to_string(path)
        .map_err(|e| TdpError::io(format!("cannot read {}: {e}", path.display())))?;
    match text.lines().next() {
        Some("tdp_command_recipe=13") => Ok(CURRENT_RECIPE_VERSION),
        Some("tdp_command_recipe=12") => Err(TdpError::config(
            "recipe v12 run (the six-stage driver whose only stage 6 was the windowed delivery): resume with the checkout that created it (recipe v13 added the adaptive stage 6, the three-mask stage 2 and the balanced source); the original run is unchanged",
        )),
        Some("tdp_command_recipe=11") => Err(TdpError::config(
            "recipe v11 run (the seven-stage driver): resume with the checkout that created it (recipe v12 merged stages 6 and 7 into the delivery stage: packing, then unpacking with lattice refinement); the original run is unchanged",
        )),
        Some("tdp_command_recipe=10") => Err(TdpError::config(
            "recipe v10 run (the four-stage driver): resume with the checkout that created it (recipe v11 added stages 4, 5 and 7 and the five-seed SEEDS format); the original run is unchanged",
        )),
        Some("tdp_command_recipe=9") => Err(TdpError::config(
            "recipe v9 run: resume with the checkout that created it (recipe v10 changed the defaults to the chosen L8 runs and added stage 4); the original run is unchanged",
        )),
        _ => Err(TdpError::config(
            "unsupported recipe version; tdp_gen requires a TDP v13 run; use a fresh run.directory or the checkout that created this run (v8 and older: branch ran/archive/curated-and-postmix); the original run is unchanged",
        )),
    }
}
