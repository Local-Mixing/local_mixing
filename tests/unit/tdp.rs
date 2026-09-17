use super::*;
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn temp_path(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "local_mixing_tdp_{label}_{}_{}",
        std::process::id(),
        TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ))
}

fn parse(body: &str) -> RawConfig {
    parse_config(&format!("config_version = 1\n{body}")).unwrap()
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
    assert_eq!(command().get_name(), "tdp_gen");
    assert!(command().get_all_aliases().next().is_none());
    let matches = command()
        .try_get_matches_from(["tdp_gen", "--dry-run", "--config", "custom.toml"])
        .unwrap();
    assert!(matches.get_flag("dry_run"));
    assert_eq!(
        matches.get_one::<PathBuf>("config"),
        Some(&PathBuf::from("custom.toml"))
    );
}

#[test]
fn omitted_values_resolve_to_current_defaults() {
    let resolved = resolve_for_test(&parse("")).unwrap();
    assert_eq!(resolved.n, 128);
    assert!(resolved.generated_run_dir);
    assert!(resolved.run_dir.ends_with("runs/tdp_gen_n128_test-run"));
    assert!(resolved.build_release);
    assert_eq!(
        resolved.preprocessing_mode,
        PreprocessingMode::EmbeddedMasking
    );
    assert_eq!(resolved.hold.as_deref(), Some("27"));
    assert!(resolved.calibration_seed.is_none());
    assert_eq!(resolved.shuffling_segments, 0);
    assert!(resolved.shuffling_return_home);
    assert!(
        !script_args(&resolved)
            .iter()
            .any(|arg| arg == OsStr::new("-s"))
    );
}

#[test]
fn nonlinear_capacity_is_validated_before_launch() {
    let too_large = parse("source.wires = 64\npreprocessing.mode = \"nonlinear291\"");
    let error = resolve_for_test(&too_large).err().unwrap();
    assert!(error.contains("wire capacity"), "{error}");
    assert!(error.contains("reduce n or set a smaller mcd"), "{error}");

    let derived_fit = parse("source.wires = 63\npreprocessing.mode = \"nonlinear291\"");
    assert!(resolve_for_test(&derived_fit).is_ok());

    let explicit_fit =
        parse("source.wires = 128\nsource.gates = 1\npreprocessing.mode = \"nonlinear291\"");
    assert!(resolve_for_test(&explicit_fit).is_ok());
}

#[test]
fn every_public_tuning_value_maps_to_a_separate_script_argument() {
    let raw = parse(
        "source.wires = 64\nrun.directory = \"runs/with spaces;$(literal)\"\nsource.gates = 900\ndb_mixing.target_size_factor = 2.5\ndb_mixing.hold_work_units = 4.5\ncrossing.target_size_factor = 2.25\ncrossing.width_penalty_base = 4\ncrossing.width_penalty_threshold = 2\ncrossing.size_tolerance_divisor = 20\ncrossing.move_attempts = 1234\nrun.stop_after_stage = 5\nrun.rerun_from_stage = 4\nparallel.pieces = 4\nparallel.threads = 6\ndatabase.allow_no_database_for_tests = true",
    );
    let resolved = resolve_for_test(&raw).unwrap();
    let args = script_args(&resolved);
    let strings: Vec<String> = args
        .iter()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect();
    for expected in [
        "--source-wires",
        "64",
        "--run-directory",
        "--source-gates",
        "900",
        "--db-mixing-target-size-factor",
        "2.5",
        "--db-mixing-hold-work-units",
        "4.5",
        "--crossing-target-size-factor",
        "2.25",
        "--crossing-width-penalty-base",
        "4",
        "--crossing-width-penalty-threshold",
        "2",
        "--crossing-size-tolerance-divisor",
        "20",
        "--crossing-move-attempts",
        "1234",
        "--run-stop-after-stage",
        "5",
        "--run-rerun-from-stage",
        "4",
        "--parallel-pieces",
        "--parallel-threads",
        "6",
    ] {
        assert!(
            strings.iter().any(|actual| actual == expected),
            "missing {expected}"
        );
    }
    // The serial recipe leaves parallel piece controls unset.
    let serial = resolve_for_test(&parse(
        "source.wires = 64\nrun.directory = \"runs/serial\"\ndatabase.allow_no_database_for_tests = true",
    ))
    .unwrap();
    assert!(
        script_args(&serial)
            .iter()
            .all(|arg| !arg.to_string_lossy().starts_with("--parallel-")),
        "omitted pieces must not emit piece flags"
    );
    assert!(resolved.run_dir.ends_with("runs/with spaces;$(literal)"));
}

#[test]
fn automatic_piece_config_validation_forwarding_and_manifest() {
    for value in ["0", "1", "-2", "2.5", "1000000001", "18446744073709551616"] {
        let result = parse_config(&format!(
            "config_version = 1\nparallel.target_piece_gates = {value}\nrun.stop_after_stage = 2"
        ))
        .and_then(|raw| resolve_for_test(&raw));
        let error = result.err().unwrap();
        // Fractions and values outside TOML's integer range fail in parsing;
        // valid TOML integers must satisfy the current piece-size bounds.
        assert!(!error.is_empty(), "accepted {value}");
    }
    for pieces in ["1", "4"] {
        let raw = parse(&format!(
            "parallel.pieces = {pieces}\nparallel.target_piece_gates = 2\nrun.stop_after_stage = 2"
        ));
        assert!(
            resolve_for_test(&raw)
                .err()
                .unwrap()
                .contains("mutually exclusive")
        );
    }
    for size in ["2", "128", "1000000000"] {
        let config = resolve_for_test(&parse(&format!(
            "parallel.target_piece_gates = {size}\nparallel.threads = 3\nrun.stop_after_stage = 2"
        )))
        .unwrap();
        let normalized = size.parse::<usize>().unwrap().to_string();
        let args = script_args(&config);
        assert!(args.windows(2).any(
            |pair| pair[0] == "--parallel-target-piece-gates" && pair[1] == normalized.as_str()
        ));
        assert!(
            args.windows(2)
                .any(|pair| pair[0] == "--parallel-threads" && pair[1] == "3")
        );
        assert!(!args.iter().any(|arg| arg == "--parallel-pieces"));
        let manifest = recipe_manifest(&config, &[], 1);
        assert!(manifest.contains(&format!("\nmin_block_size={normalized}\n")));
        assert!(manifest.contains("\npieces=auto\n"));
        assert!(!manifest.contains("piece_threads"));
        let mut changed = config.clone();
        changed.piece_threads = None;
        assert_eq!(manifest, recipe_manifest(&changed, &[], 1));
        changed.min_block_size = Some("3".into());
        assert_ne!(manifest, recipe_manifest(&changed, &[], 1));
        changed.min_block_size = None;
        assert_ne!(manifest, recipe_manifest(&changed, &[], 1));
    }
    let serial = resolve_for_test(&parse("run.stop_after_stage = 2")).unwrap();
    assert!(serial.pieces.is_none() && serial.min_block_size.is_none());
    assert!(!recipe_manifest(&serial, &[], 1).contains("min_block_size"));
    assert!(
        !script_args(&serial)
            .iter()
            .any(|arg| arg == "--parallel-target-piece-gates")
    );
}

#[test]
fn environment_store_paths_are_resolved_and_document_paths_win() {
    let raw = parse(
        "run.directory = \"runs/test\"\ndatabase.regular_dir = \"stores/from-doc\"\nrun.stop_after_stage = 2",
    );
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
fn resume_after_db_mixing_does_not_require_the_stage_three_store() {
    let run_dir = std::env::temp_dir().join(format!(
        "local_mixing_tdp_resume_{}_{}",
        std::process::id(),
        default_run_tag()
    ));
    fs::create_dir_all(&run_dir).unwrap();
    fs::write(run_dir.join("db_mixing.mpmct1"), "nonempty").unwrap();
    fs::write(run_dir.join("tdp.mpmct1"), "nonempty").unwrap();
    fs::write(run_dir.join("SEED"), "123456\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(run_dir.join("SEED"), fs::Permissions::from_mode(0o600)).unwrap();
    }

    let raw = parse(&format!("run.directory = {:?}", run_dir));
    let resolved = resolve_for_test(&raw).unwrap();
    validate_external_paths(&resolved).unwrap();

    let forced = parse(&format!(
        "run.directory = {:?}\nrun.rerun_from_stage = 3",
        run_dir
    ));
    let forced = resolve_for_test(&forced).unwrap();
    assert!(validate_external_paths(&forced).is_err());
    fs::remove_file(run_dir.join("tdp.mpmct1")).unwrap();
    assert!(
        validate_external_paths(&resolved).is_err(),
        "missing stage 2 invalidates downstream artifacts"
    );
    fs::remove_dir_all(run_dir).unwrap();
}

#[test]
fn invalid_numbers_enums_and_booleans_fail_before_launch() {
    for body in [
        "source.wires = 2",
        "db_mixing.target_size_factor = nan",
        "db_mixing.target_size_factor = 1",
        "db_mixing.hold_work_units = -1",
        "crossing.size_tolerance_divisor = 0",
        "run.stop_after_stage = 7",
        "preprocessing.mode = \"invented\"",
        "run.build_binaries = \"yes\"",
        "database.lookup_miss_filter = \"maybe\"",
        "parallel.pieces = 0",
        "parallel.pieces = 65",
        "parallel.pieces = 1.5",
        "parallel.threads = 4",
        "parallel.pieces = 1\nparallel.threads = 2",
        "parallel.pieces = 2\nparallel.threads = 0",
    ] {
        let result = parse_config(&format!("config_version = 1\n{body}"))
            .and_then(|raw| resolve_for_test(&raw));
        assert!(result.is_err(), "unexpectedly accepted {body}");
    }
}

#[test]
fn explicit_seed_requires_calibration_gate_and_is_redacted() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let seed_path = std::env::temp_dir().join(format!(
        "local_mixing_tdp_seed_{}_{}",
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
        "run.directory = \"runs/test\"\ncalibration.seed_file = {:?}\ndatabase.allow_no_database_for_tests = true",
        seed_path
    ));
    assert!(resolve_for_test(&without_gate).is_err());

    let with_gate = parse(&format!(
        "run.directory = \"runs/test\"\ncalibration.enabled = true\ncalibration.seed_file = {:?}\ndatabase.allow_no_database_for_tests = true",
        seed_path
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
fn checked_in_toml_recipe_uses_embedded_masking() {
    let parsed = parse_config(include_str!("../../configs/tdp.toml")).unwrap();
    let config = resolve_for_test(&parsed).unwrap();
    assert_eq!(
        config.preprocessing_mode,
        PreprocessingMode::EmbeddedMasking
    );
    assert_eq!(
        (
            config.mask_pair_wires,
            config.max_open_masks,
            config.min_open_masks
        ),
        (2, 3, 2)
    );
    assert_eq!(config.hold.as_deref(), Some("27"));
    assert_eq!(config.recipe_version, 8);
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
        "runs/../.git/tdp",
    ] {
        let raw = parse(&format!(
            "run.directory = {value:?}\ndatabase.allow_no_database_for_tests = true"
        ));
        assert!(resolve_config(&raw, root, |_| None, "test").is_err());
    }
}

#[test]
fn recipe_manifest_is_seed_free_normalized_and_lifecycle_independent() {
    let base = resolve_for_test(&parse(
        "run.directory = \"runs/manifest-test\"\nrun.stop_after_stage = 2\ndatabase.allow_no_database_for_tests = true",
    ))
    .unwrap();
    let fingerprints = [
        ("gen_sandwich_gadget", 1u128),
        ("circuit_mixer", 2u128),
        ("fcompress", 3u128),
    ];
    let expected = recipe_manifest(&base, &fingerprints, 4);
    assert!(expected.contains("binary_circuit_mixer_xxh3=00000000000000000000000000000002"));
    assert!(expected.contains("script_tdp_gen_xxh3=00000000000000000000000000000004"));

    let explicit_defaults = resolve_for_test(&parse(
        "run.directory = \"runs/manifest-test\"\ndb_mixing.target_size_factor = 2.0\ndb_mixing.hold_work_units = 2.7e1\ncrossing.target_size_factor = 2e0\ncrossing.width_penalty_base = 3.0\ncrossing.width_penalty_threshold = 1\ncrossing.size_tolerance_divisor = 25\nparallel.pieces = 1\nrun.stop_after_stage = 2\ndatabase.allow_no_database_for_tests = true",
    ))
    .unwrap();
    assert_eq!(
        expected,
        recipe_manifest(&explicit_defaults, &fingerprints, 4)
    );
    assert!(expected.starts_with("tdp_command_recipe=8\n"));
    assert!(expected.contains("\npieces=1\n"));
    // `pieces` is a locked recipe value; `piece_threads` is lifecycle.
    let pieced = resolve_for_test(&parse(
        "run.directory = \"runs/manifest-test\"\nparallel.pieces = 4\nparallel.threads = 8\nrun.stop_after_stage = 2\ndatabase.allow_no_database_for_tests = true",
    ))
    .unwrap();
    let pieced_manifest = recipe_manifest(&pieced, &fingerprints, 4);
    assert_ne!(expected, pieced_manifest);
    assert!(pieced_manifest.contains("\npieces=4\n"));
    assert!(!pieced_manifest.contains("piece_threads"));
    let mut retuned = pieced.clone();
    retuned.piece_threads = Some("3".to_owned());
    assert_eq!(pieced_manifest, recipe_manifest(&retuned, &fingerprints, 4));

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
    let desired = "tdp_command_recipe=8\nn=128\n";
    fs::write(&path, desired).unwrap();
    compare_recipe_manifest(&path, desired).unwrap();

    let mismatch = compare_recipe_manifest(&path, "tdp_command_recipe=8\nn=129\n").unwrap_err();
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
        "run.directory = \"runs/env-test\"\nrun.stop_after_stage = 2\ndatabase.allow_no_database_for_tests = true",
    ))
    .unwrap();
    let mut command = ProcessCommand::new("true");
    configure_environment(&mut command, &config, Path::new("/tmp/tdp-bin"));
    let environment: BTreeMap<String, Option<String>> = command
        .get_envs()
        .map(|(key, value)| {
            (
                key.to_string_lossy().into_owned(),
                value.map(|value| value.to_string_lossy().into_owned()),
            )
        })
        .collect();
    for key in ["BASH_ENV", "PROD_K", "SAT_HARDEN", "MIXER_DUMP_OUT"] {
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
        environment.get("TDP_BIN_DIR"),
        Some(&Some("/tmp/tdp-bin".to_owned()))
    );
    assert_eq!(environment.get("PROD_PRESET"), Some(&None));
    for (key, value) in [
        ("EMBEDDED_MASKING_K", "2"),
        ("EMBEDDED_MASKING_MAX_OPEN", "3"),
        ("EMBEDDED_MASKING_MIN_OPEN", "2"),
        ("EMBEDDED_MASKING_BALANCED", "1"),
        ("EMBEDDED_MASKING_QUAD_FIRE", "1"),
        ("EMBEDDED_MASKING_SHUFFLING", "0"),
        ("EMBEDDED_MASKING_SHUFFLING_RETURN_HOME", "1"),
    ] {
        assert_eq!(environment.get(key), Some(&Some(value.to_owned())));
    }

    let nonlinear = resolve_for_test(&parse(
        "run.directory = \"runs/env-test-nonlinear\"\npreprocessing.mode = \"nonlinear291\"\nsource.gates = 1\nrun.stop_after_stage = 2\ndatabase.allow_no_database_for_tests = true",
    ))
    .unwrap();
    let mut command = ProcessCommand::new("true");
    configure_environment(&mut command, &nonlinear, Path::new("/tmp/tdp-bin"));
    let environment: BTreeMap<String, Option<String>> = command
        .get_envs()
        .map(|(key, value)| {
            (
                key.to_string_lossy().into_owned(),
                value.map(|value| value.to_string_lossy().into_owned()),
            )
        })
        .collect();
    assert_eq!(environment.get("PROD_PRESET"), Some(&None));
    assert_eq!(environment.get("PROD_POST_FRAGMENT"), Some(&None));
}

#[test]
fn run_target_and_store_trees_must_not_overlap() {
    let base = temp_path("separation");
    fs::create_dir_all(&base).unwrap();
    let run = base.join("run");
    let target = base.join("target");
    let store = base.join("store");
    assert!(validate_path_separation(&run, &run.join("target"), None, None).is_err());
    assert!(validate_path_separation(&run, &target, Some(&target.join("store")), None).is_err());
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
    let config_path = base.join("recipe.toml");
    let run_dir = base.join("run");
    let target_dir = base.join("target");
    let store_dir = base.join("unused-store");
    fs::write(
        &config_path,
        format!(
            "config_version = 1\nrun.directory = {run_dir:?}\nrun.build_directory = {target_dir:?}\ndatabase.regular_dir = {store_dir:?}\nrun.stop_after_stage = 2"
        ),
    )
    .unwrap();
    let matches = command()
        .try_get_matches_from(vec![
            OsString::from("tdp_gen"),
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
fn unverified_runs_are_rejected_without_changing_saved_artifacts() {
    let base = temp_path("unverified-run");
    let run_dir = base.join("run");
    fs::create_dir_all(&run_dir).unwrap();
    let seed_path = run_dir.join("SEED");
    fs::write(&seed_path, "123456\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&seed_path, fs::Permissions::from_mode(0o600)).unwrap();
    }
    fs::write(run_dir.join("tdp.mpmct1"), "saved gadget\n").unwrap();
    let config_path = base.join("recipe.toml");
    let target_dir = base.join("target");
    for adopt in [false, true] {
        for calibration in [false, true] {
            fs::write(
                &config_path,
                format!(
                    "config_version = 1\nrun.directory = {run_dir:?}\nrun.build_directory = {target_dir:?}\nrun.stop_after_stage = 2\nrun.adopt_unverified_run = {adopt}\ncalibration.enabled = {calibration}\n"
                ),
            )
            .unwrap();
            let matches = command()
                .try_get_matches_from([
                    OsString::from("tdp_gen"),
                    OsString::from("--config"),
                    config_path.clone().into_os_string(),
                ])
                .unwrap();
            let error = run_inner(&matches).unwrap_err();
            assert!(
                error.message.contains("use a fresh run.directory"),
                "{error:?}"
            );
            assert!(!target_dir.exists());
            assert!(!run_dir.join("tdp_command.conf").exists());
            assert_eq!(fs::read_to_string(&seed_path).unwrap(), "123456\n");
            assert_eq!(
                fs::read_to_string(run_dir.join("tdp.mpmct1")).unwrap(),
                "saved gadget\n"
            );
        }
    }
    fs::remove_dir_all(base).unwrap();
}

#[test]
fn toml_rejects_unknown_keys_duplicates_and_wrong_types() {
    for input in [
        "config_version = 2",
        "config_version = 1\n[typo]",
        "config_version = 1\n[preprocessing]\nmode = \"embedded-masking\"\nmisspelled = 1",
        "config_version = 1\n[source]\nwires = \"128\"",
        "config_version = 1\n[run]\nbuild_binaries = \"true\"",
        "config_version = 1\n[source]\nwires = 4\nwires = 5",
    ] {
        assert!(parse_config(input).is_err(), "accepted {input}");
    }
}

#[test]
fn toml_default_hold_and_explicit_27_have_the_same_recipe() {
    let implicit = resolve_for_test(&parse_config("config_version = 1").unwrap()).unwrap();
    let explicit = resolve_for_test(
        &parse_config("config_version = 1\n[db_mixing]\nhold_work_units = 27.0").unwrap(),
    )
    .unwrap();
    assert_eq!(
        recipe_manifest(&implicit, &[], 7),
        recipe_manifest(&explicit, &[], 7)
    );
    assert!(recipe_manifest(&implicit, &[], 7).contains("hold=27\n"));
    assert!(recipe_manifest(&implicit, &[], 7).starts_with("tdp_command_recipe=8\n"));
}

#[test]
fn managed_v8_dry_run_preserves_manifest_and_rejects_changed_binary() {
    let base = temp_path("managed-v8");
    let run_dir = base.join("run");
    let target_dir = base.join("target");
    let binary_dir = target_dir
        .join(rustc_host_triple().unwrap())
        .join("release");
    fs::create_dir_all(&run_dir).unwrap();
    fs::create_dir_all(&binary_dir).unwrap();
    for name in ["gen_sandwich_gadget", "circuit_mixer", "fcompress"] {
        fs::write(binary_dir.join(name), format!("original {name}\n")).unwrap();
    }
    let recipe = format!(
        "config_version = 1\nsource.wires = 8\nsource.gates = 1\nrun.directory = {run_dir:?}\nrun.build_directory = {target_dir:?}\nrun.stop_after_stage = 2\n"
    );
    let config_path = base.join("original.toml");
    fs::write(&config_path, &recipe).unwrap();
    let config = resolve_for_test(&parse_config(&recipe).unwrap()).unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let original_manifest = recipe_manifest(
        &config,
        &production_binary_fingerprints(&binary_dir).unwrap(),
        hash_file(&root.join("scripts/tdp_gen.sh")).unwrap(),
    );
    let manifest_path = run_dir.join("tdp_command.conf");
    fs::write(&manifest_path, &original_manifest).unwrap();
    let matches = command()
        .try_get_matches_from([
            OsString::from("tdp_gen"),
            OsString::from("--dry-run"),
            OsString::from("--config"),
            config_path.into_os_string(),
        ])
        .unwrap();
    run_inner(&matches).unwrap();
    assert_eq!(
        fs::read_to_string(&manifest_path).unwrap(),
        original_manifest
    );
    fs::write(binary_dir.join("circuit_mixer"), "changed binary\n").unwrap();
    let error = run_inner(&matches).unwrap_err();
    assert!(
        error.message.contains("mixed-provenance resume"),
        "{error:?}"
    );
    assert_eq!(
        fs::read_to_string(&manifest_path).unwrap(),
        original_manifest
    );
    fs::remove_dir_all(base).unwrap();
}

#[test]
fn unsupported_modes_are_rejected_before_artifacts_or_builds() {
    let base = temp_path("unsupported-mode");
    fs::create_dir_all(&base).unwrap();
    let run_dir = base.join("run");
    let target_dir = base.join("target");
    let config_path = base.join("recipe");
    for mode in ["unknown-mode", "embedded_masking", "EMBEDDED-MASKING"] {
        let recipe = format!(
            "config_version = 1\nsource.wires = 4\nsource.gates = 1\nrun.directory = {run_dir:?}\nrun.build_directory = {target_dir:?}\npreprocessing.mode = {mode:?}"
        );
        fs::write(&config_path, recipe).unwrap();
        let matches = command()
            .try_get_matches_from([
                OsString::from("tdp_gen"),
                OsString::from("--dry-run"),
                OsString::from("--config"),
                config_path.clone().into_os_string(),
            ])
            .unwrap();
        let error = run_inner(&matches).unwrap_err();
        assert!(
            error
                .message
                .contains("expected embedded-masking or nonlinear291"),
            "{error:?}"
        );
        assert!(!run_dir.exists());
        assert!(!target_dir.exists());
    }
    fs::remove_dir_all(base).unwrap();
}

#[test]
fn non_toml_input_is_rejected() {
    for input in [
        "```toml\nconfig_version = 1\n```",
        r#"{"config_version": 1}"#,
    ] {
        assert!(parse_config(input).is_err(), "accepted {input}");
    }
}

#[test]
fn embedded_masking_rejects_silently_clamped_width_and_insufficient_band() {
    for (n, k, valid) in [(4, 1, false), (4, 2, true), (4, 7, true), (4, 8, false)] {
        let recipe = format!(
            "config_version = 1\n[source]\nwires = {n}\n[preprocessing]\nmode = \"embedded-masking\"\nmask_pair_wires = {k}"
        );
        let config = resolve_for_test(&parse_config(&recipe).unwrap()).unwrap();
        assert_eq!(validate_current_recipe(&config).is_ok(), valid, "{recipe}");
    }
}

#[test]
fn unsupported_recipe_versions_fail_before_building_and_leave_saved_runs_unchanged() {
    for version in [3, 4, 5, 6, 7] {
        let base = temp_path("unsupported-version");
        let run_dir = base.join("run");
        let target_dir = base.join("target");
        fs::create_dir_all(&run_dir).unwrap();
        let recipe = format!(
            "config_version = 1\nsource.wires = 4\nsource.gates = 1\nrun.directory = {run_dir:?}\nrun.build_directory = {target_dir:?}\nrun.stop_after_stage = 2\n"
        );
        let config_path = base.join("recipe.toml");
        fs::write(&config_path, recipe).unwrap();
        let original_manifest = format!("tdp_command_recipe={version}\nrecorded=unchanged\n");
        let manifest_path = run_dir.join("tdp_command.conf");
        fs::write(&manifest_path, &original_manifest).unwrap();
        for dry_run in [false, true] {
            let mut args = vec![
                OsString::from("tdp_gen"),
                OsString::from("--config"),
                config_path.clone().into_os_string(),
            ];
            if dry_run {
                args.push(OsString::from("--dry-run"));
            }
            let matches = command().try_get_matches_from(args).unwrap();
            let error = run_inner(&matches).unwrap_err();
            assert_eq!(error.exit_code, 2);
            assert!(
                error.message.contains("unsupported recipe version"),
                "{error:?}"
            );
            assert!(!target_dir.exists());
            assert_eq!(
                fs::read_to_string(&manifest_path).unwrap(),
                original_manifest
            );
            assert_eq!(fs::read_dir(&run_dir).unwrap().count(), 1);
        }
        fs::remove_dir_all(base).unwrap();
    }
}

#[test]
fn supported_toml_aliases_normalize_and_reject_duplicates() {
    for &(canonical, alias, flat, kind) in crate::tdp::config::TOML_FIELDS {
        let value = match kind {
            "string" => "\"example\"",
            "bool" => "true",
            "integer" => "4",
            "number" => "2.5",
            _ => unreachable!(),
        };
        let canonical_raw =
            parse_config(&format!("config_version = 1\n{canonical} = {value}")).unwrap();
        let (section, key) = alias.split_once('.').unwrap();
        let alias_raw =
            parse_config(&format!("config_version = 1\n[{section}]\n{key} = {value}")).unwrap();
        assert_eq!(canonical_raw.entries.len(), 1);
        assert_eq!(
            canonical_raw.value(flat),
            alias_raw.value(flat),
            "{canonical}"
        );
        if canonical != alias {
            let error = parse_config(&format!(
                "config_version = 1\n{alias} = {value}\n{canonical} = {value}"
            ))
            .unwrap_err();
            assert!(error.contains("duplicate TDP setting"), "{error}");
            assert!(error.contains(canonical), "{error}");
        }
    }
}

#[test]
fn canonical_recipe_preserves_normalized_values_and_flag_values() {
    let recipe = "config_version = 1\nrun.directory = \"runs/naming-test\"\ndb_mixing.target_size_factor = 3.0\ndb_mixing.hold_work_units = 7.0\nparallel.target_piece_gates = 10000\nparallel.threads = 4\npreprocessing.mask_pair_wires = 4\nleakage_repair.enabled = true\nleakage_repair.seed = 99\ncrossing.move_attempts = 42";
    let mut config = resolve_for_test(&parse_config(recipe).unwrap()).unwrap();
    // Preserve argument values even when they have the same spelling as an option.
    config.run_dir = PathBuf::from("-n");
    config.calibration_seed = Some("123456789".into());
    let args = script_args(&config);
    assert_eq!(args[2], "--run-directory");
    assert_eq!(args[3], "-n");
    assert!(args.contains(&OsString::from("--db-mixing-target-size-factor")));
    assert!(args.contains(&OsString::from("--parallel-target-piece-gates")));
    assert_eq!(args[0], "--source-wires");
    let rendered = render_command(&args, Some("--calibration-seed"));
    assert!(rendered.contains("<redacted>"));
    assert!(!rendered.contains("123456789"));
}

#[test]
fn completed_db_mixing_uses_current_artifact_name() {
    let root = temp_path("db-mixing-artifacts");
    fs::create_dir_all(&root).unwrap();
    let mut config = resolve_for_test(&parse_config("config_version = 1").unwrap()).unwrap();
    config.run_dir = root.clone();
    fs::write(root.join("tdp.mpmct1"), "saved gadget").unwrap();
    // An unrelated artifact cannot bypass database prerequisites.
    fs::write(root.join("unrelated.mpmct1"), "saved stage3").unwrap();
    assert!(validate_external_paths(&config).is_err());
    fs::write(root.join("db_mixing.mpmct1"), "saved stage3").unwrap();
    validate_external_paths(&config).unwrap();
    config.force_from = Some("3".into());
    assert!(validate_external_paths(&config).is_err());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn embedded_masking_config_uses_canonical_names() {
    let config = resolve_for_test(&parse("preprocessing.mode = \"embedded-masking\"")).unwrap();
    validate_current_recipe(&config).unwrap();
    assert_eq!(config.recipe_version, 8);
    let manifest = recipe_manifest(&config, &[], 1);
    assert!(manifest.contains("preprocessing_mode=embedded-masking\n"));
    assert!(manifest.contains("preprocessing_balanced_masks=1\n"));
    let args = script_args(&config);
    assert_eq!(args[4], "--preprocessing-mode");
    assert_eq!(args[5], "embedded-masking");

    for setting in [
        "[unknown]",
        "unknown.mode = \"embedded-masking\"",
        "preprocessing.misspelled_setting = 2",
    ] {
        let error = parse_config(&format!("config_version = 1\n{setting}")).unwrap_err();
        assert!(error.contains("unknown TDP"), "{error}");
    }
}

#[test]
fn nonlinear291_rejects_only_explicit_mask_controls_for_v8() {
    let base = "config_version = 1\nsource.wires = 4\nsource.gates = 1\npreprocessing.mode = \"nonlinear291\"\n";
    let config = resolve_for_test(&parse_config(base).unwrap()).unwrap();
    validate_current_recipe(&config).unwrap();
    assert!(!config.explicit_mask_controls);
    assert_eq!(config.preprocessing_mode, PreprocessingMode::Nonlinear291);
    assert!(!recipe_manifest(&config, &[], 1).contains("preprocessing_mask_pair_wires"));
    for setting in [
        "preprocessing.mask_pair_wires = 2",
        "preprocessing.max_open_masks = 3",
        "preprocessing.min_open_masks = 2",
        "preprocessing.balanced_masks = true",
        "preprocessing.shuffling_segments = 8",
        "preprocessing.shuffling_return_home = true",
    ] {
        let raw = parse_config(&format!("{base}{setting}")).unwrap();
        let config = resolve_for_test(&raw).unwrap();
        let error = validate_current_recipe(&config).unwrap_err();
        assert!(
            error.message.contains("mask controls apply only"),
            "{setting}: {error:?}"
        );
    }
}

#[test]
fn embedded_mask_balancing_is_typed_pinned_and_locked_in_v8() {
    let raw = parse_config("config_version = 1\n[preprocessing]\nbalanced_masks = false").unwrap();
    let mut config = resolve_for_test(&raw).unwrap();
    validate_current_recipe(&config).unwrap();
    assert!(!config.balanced_masks);
    let manifest = recipe_manifest(&config, &[], 1);
    assert!(manifest.contains("preprocessing_balanced_masks=0\n"));
    let mut command = ProcessCommand::new("bash");
    configure_environment(&mut command, &config, Path::new("/tmp/tdp-bin"));
    assert!(
        command.get_envs().any(
            |(key, value)| key == "EMBEDDED_MASKING_BALANCED" && value == Some(OsStr::new("0"))
        )
    );
    config.balanced_masks = true;
    assert_ne!(manifest, recipe_manifest(&config, &[], 1));
}

#[test]
fn preprocessing_shuffling_is_typed_pinned_and_preserves_tdp_output_ports() {
    let raw = parse("preprocessing.shuffling_segments = 8");
    let config = resolve_for_test(&raw).unwrap();
    validate_current_recipe(&config).unwrap();
    assert_eq!(config.shuffling_segments, 8);
    assert!(config.shuffling_return_home);
    let manifest = recipe_manifest(&config, &[], 1);
    assert!(manifest.contains("preprocessing_shuffling_segments=8\n"));
    assert!(manifest.contains("preprocessing_shuffling_return_home=true\n"));
    let mut command = ProcessCommand::new("true");
    configure_environment(&mut command, &config, Path::new("/tmp/tdp-bin"));
    for (key, value) in [
        ("EMBEDDED_MASKING_SHUFFLING", "8"),
        ("EMBEDDED_MASKING_SHUFFLING_RETURN_HOME", "1"),
    ] {
        assert!(
            command
                .get_envs()
                .any(|(k, v)| k == key && v == Some(OsStr::new(value)))
        );
    }
    for segments in 1..8 {
        let error = resolve_for_test(&parse(&format!(
            "preprocessing.shuffling_segments = {segments}"
        )))
        .err()
        .unwrap();
        assert!(error.contains("0 (off) or at least 8"), "{error}");
    }
    let error = resolve_for_test(&parse(
        "preprocessing.shuffling_segments = 8\npreprocessing.shuffling_return_home = false",
    ))
    .err()
    .unwrap();
    assert!(error.contains("fixed physical output layout"), "{error}");
}

#[test]
fn manifests_without_shuffling_settings_require_defaults_and_matching_binaries() {
    let config = resolve_for_test(&parse("")).unwrap();
    let desired = recipe_manifest(&config, &[("gen_sandwich_gadget", 1)], 2);
    let without_shuffling: String = desired
        .lines()
        .filter(|line| !line.starts_with("preprocessing_shuffling_"))
        .map(|line| format!("{line}\n"))
        .collect();
    let path = temp_path("shuffling-manifest");
    fs::write(&path, &without_shuffling).unwrap();
    compare_recipe_manifest(&path, &desired).unwrap();
    let mut enabled = config.clone();
    enabled.shuffling_segments = 8;
    let error = compare_recipe_manifest(
        &path,
        &recipe_manifest(&enabled, &[("gen_sandwich_gadget", 1)], 2),
    )
    .unwrap_err();
    assert!(error.message.contains("preprocessing_shuffling_segments"));
    assert!(
        compare_recipe_manifest(
            &path,
            &recipe_manifest(&config, &[("gen_sandwich_gadget", 3)], 2),
        )
        .is_err()
    );
    // Enabling/changing routing on an already shuffled run also changes its
    // recipe identity; it cannot inherit an unshuffled saved stage.
    fs::write(&path, recipe_manifest(&enabled, &[], 2)).unwrap();
    enabled.shuffling_segments = 16;
    assert!(compare_recipe_manifest(&path, &recipe_manifest(&enabled, &[], 2)).is_err());
    fs::remove_file(path).unwrap();
}

#[test]
fn current_manifest_records_default_shuffling_deterministically() {
    let raw = parse_config("config_version = 1\nsource.wires = 4").unwrap();
    let config = resolve_for_test(&raw).unwrap();
    // Keep the TDP v8 manifest deterministic, including every field and its order.
    let expected = concat!(
        "tdp_command_recipe=8\nn=4\nfrozen_db_dir=unset\nfrozen_curated_dir=unset\n",
        "curated_value_convention=native\npreprocessing_mode=embedded-masking\n",
        "production_preset=not-applicable\npost_fragment=not-applicable\n",
        "mcd=derived\nexpand=2\nhold=27\nxr=2\nxb=3\nxc=1\nxtdiv=25\n",
        "xmoves=6*target\npieces=1\nallow_empty_store=false\ncalibration_only=false\n",
        "pinned_recipe=canon512/200000-cache256/1024/2048-native-regular\n",
        "sandwich_variant=classic\npreprocessing_mask_pair_wires=2\n",
        "preprocessing_max_open_masks=3\npreprocessing_min_open_masks=2\n",
        "preprocessing_balanced_masks=1\npreprocessing_quadratic_fire=1\n",
        "preprocessing_extra_lgis=0\npreprocessing_encoded_io=false\n",
        "preprocessing_shuffling_segments=0\npreprocessing_shuffling_return_home=true\n",
        "source_path=unset\nsource_xxh3=generated\nqc_enabled=false\nqc_seed=20803\n",
        "qc_reference=unset\nqc_reference_xxh3=unset\n",
        "script_tdp_gen_xxh3=00000000000000000000000000000001\n",
    );
    assert_eq!(recipe_manifest(&config, &[], 1), expected);
}

#[test]
#[cfg(unix)]
fn fresh_v8_managed_runs_record_and_resume_both_preprocessing_modes() {
    use std::os::unix::fs::PermissionsExt;
    for mode in ["embedded-masking", "nonlinear291"] {
        let base = temp_path("fresh-v8");
        let run_dir = base.join("run");
        let target_dir = base.join("target");
        let binary_dir = target_dir
            .join(rustc_host_triple().unwrap())
            .join("release");
        fs::create_dir_all(&binary_dir).unwrap();
        let generator = concat!(
            "#!/bin/bash\nset -eu\n",
            "printf '4\\nx\\nx\\n' > \"$1\"\n",
            "printf '4\\nx\\nx\\n' > \"$1.sandwich.mpmct1\"\n",
            "printf 'called\\n' >> \"$1.calls\"\n",
        );
        for name in ["gen_sandwich_gadget", "circuit_mixer", "fcompress"] {
            let path = binary_dir.join(name);
            fs::write(
                &path,
                if name == "gen_sandwich_gadget" {
                    generator
                } else {
                    "#!/bin/bash\nexit 98\n"
                },
            )
            .unwrap();
            fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
        }
        let recipe = format!(
            "config_version = 1\nsource.wires = 4\nsource.gates = 1\npreprocessing.mode = {mode:?}\nrun.directory = {run_dir:?}\nrun.build_directory = {target_dir:?}\nrun.build_binaries = false\nrun.stop_after_stage = 2\n"
        );
        let config_path = base.join("recipe.toml");
        fs::write(&config_path, recipe).unwrap();
        let matches = command()
            .try_get_matches_from([
                OsString::from("tdp_gen"),
                OsString::from("--config"),
                config_path.into_os_string(),
            ])
            .unwrap();
        run_inner(&matches).unwrap();
        let manifest_path = run_dir.join("tdp_command.conf");
        let manifest = fs::read_to_string(&manifest_path).unwrap();
        assert!(manifest.starts_with("tdp_command_recipe=8\n"));
        assert!(manifest.contains(&format!("preprocessing_mode={mode}\n")));
        assert_eq!(read_recipe_version(&manifest_path).unwrap(), 8);
        let stage_marker = fs::read_to_string(run_dir.join("stage12.recipe")).unwrap();
        assert!(stage_marker.contains(&format!("preprocessing_mode={mode}\n")));
        run_inner(&matches).unwrap();
        assert_eq!(fs::read_to_string(manifest_path).unwrap(), manifest);
        assert_eq!(
            fs::read_to_string(run_dir.join("tdp.mpmct1.calls")).unwrap(),
            "called\n"
        );
        fs::remove_dir_all(base).unwrap();
    }
}
