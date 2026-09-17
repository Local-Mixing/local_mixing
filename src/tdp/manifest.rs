//! Run provenance and executable fingerprints.
use super::*;

pub(crate) const MAX_MANIFEST_BYTES: u64 = 64 * 1024;
pub(crate) const CURRENT_RECIPE_VERSION: u8 = 8;

pub(crate) fn prepare_recipe_manifest(
    config: &ResolvedConfig,
    binary_dir: &Path,
    script: &Path,
) -> Result<(), TdpError> {
    let fingerprints = production_binary_fingerprints(binary_dir)?;
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
    ["gen_sandwich_gadget", "circuit_mixer", "fcompress"]
        .into_iter()
        .map(|name| {
            let path = binary_dir.join(name);
            hash_file(&path)
                .map(|hash| (name, hash))
                .map_err(|error| {
                    TdpError::io(format!(
                        "cannot fingerprint production binary {}: {error}; set run.build_binaries=true or select the target directory containing all three binaries",
                        path.display()
                    ))
                })
        })
        .collect()
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
    // Earlier v8 embedded-masking runs predate these fields and therefore
    // mean shuffling off / return home. Only those exact defaults may be
    // omitted; enabling shuffling remains a locked recipe change. Binary
    // and script fingerprints still have to match for any managed resume.
    let normalize = |text: &str| -> String {
        let embedded = text
            .lines()
            .any(|line| line == "preprocessing_mode=embedded-masking");
        text.lines()
            .filter(|line| {
                *line != "adopted_unverified=true"
                    && !(embedded
                        && matches!(
                            *line,
                            "preprocessing_shuffling_segments=0"
                                | "preprocessing_shuffling_return_home=true"
                        ))
            })
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

pub(crate) fn recipe_manifest(
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
            "tdp_command_recipe={}\n",
            "n={}\n",
            "frozen_db_dir={}\n",
            "frozen_curated_dir={}\n",
            "curated_value_convention={}\n",
            "preprocessing_mode={}\n",
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
            "pieces={}\n",
            "allow_empty_store={}\n",
            "calibration_only={}\n",
            "pinned_recipe=canon512/200000-cache256/1024/2048-native-regular\n",
        ),
        config.recipe_version,
        config.n,
        path_or_unset(&config.frozen_db.path),
        path_or_unset(&config.frozen_curated.path),
        config.curated_value_convention,
        config.preprocessing_mode.canonical_name(),
        // Reserved fields retain their explicit non-applicable values.
        "not-applicable",
        "not-applicable",
        config.mcd.as_deref().unwrap_or("derived"),
        config.expand.as_deref().unwrap_or("2"),
        config.hold.as_deref().unwrap_or("30"),
        config.xr.as_deref().unwrap_or("2"),
        config.xb.as_deref().unwrap_or("3"),
        config.xc.as_deref().unwrap_or("1"),
        config.xtdiv.as_deref().unwrap_or("25"),
        config.xmoves.as_deref().unwrap_or("6*target"),
        if config.min_block_size.is_some() {
            "auto"
        } else {
            config.pieces.as_deref().unwrap_or("1")
        },
        config.allow_empty_store,
        config.calibration_only,
    );
    {
        use std::fmt::Write as _;
        let _ = writeln!(manifest, "sandwich_variant=classic");
        if config.preprocessing_mode == PreprocessingMode::EmbeddedMasking {
            let _ = writeln!(
                manifest,
                "preprocessing_mask_pair_wires={}\npreprocessing_max_open_masks={}\npreprocessing_min_open_masks={}\npreprocessing_balanced_masks={}\npreprocessing_quadratic_fire=1\npreprocessing_extra_lgis=0\npreprocessing_encoded_io=false",
                config.mask_pair_wires,
                config.max_open_masks,
                config.min_open_masks,
                u8::from(config.balanced_masks)
            );
            let _ = writeln!(
                manifest,
                "preprocessing_shuffling_segments={}\npreprocessing_shuffling_return_home={}",
                config.shuffling_segments, config.shuffling_return_home,
            );
        }
        let _ = writeln!(
            manifest,
            "source_path={}\nsource_xxh3={}",
            path_or_unset(&config.source_path),
            config
                .source_hash
                .map(|h| format!("{h:032x}"))
                .unwrap_or_else(|| "generated".into())
        );
        let _ = writeln!(
            manifest,
            "qc_enabled={}\nqc_seed={}\nqc_reference={}\nqc_reference_xxh3={}",
            config.qc_enabled,
            config.qc_seed,
            path_or_unset(&config.qc_reference),
            config
                .qc_reference_hash
                .map(|h| format!("{h:032x}"))
                .unwrap_or_else(|| "unset".into())
        );
    }
    // Preserve fixed/serial manifest text; automatic sizing is a locked recipe value.
    if let Some(size) = &config.min_block_size {
        use std::fmt::Write as _;
        let _ = writeln!(manifest, "min_block_size={size}");
    }
    for (name, fingerprint) in binaries {
        use std::fmt::Write as _;
        let _ = writeln!(manifest, "binary_{name}_xxh3={fingerprint:032x}");
    }
    use std::fmt::Write as _;
    let _ = writeln!(manifest, "script_tdp_gen_xxh3={script_hash:032x}");
    manifest
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
        Some("tdp_command_recipe=8") => Ok(CURRENT_RECIPE_VERSION),
        _ => Err(TdpError::config(
            "unsupported recipe version; tdp_gen requires a TDP v8 run; use a fresh run.directory or the checkout that created this run; the original run is unchanged",
        )),
    }
}
