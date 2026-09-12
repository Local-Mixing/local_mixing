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
    assert_eq!(
        resolved.preprocessing_mode,
        RecipePreprocessingMode::Product2223
    );
    assert_eq!(resolved.production_preset, "production");
    assert!(resolved.calibration_seed.is_none());
    assert!(
        !script_args(&resolved)
            .iter()
            .any(|arg| arg == OsStr::new("-s"))
    );
}

#[test]
fn historical_gadgetization_modes_are_validated_propagated_and_normalized() {
    for (value, expected) in [
        ("product-2223", RecipePreprocessingMode::Product2223),
        ("2223", RecipePreprocessingMode::Product2223),
        ("nonlinear193", RecipePreprocessingMode::Nonlinear193),
        ("nonlinear291", RecipePreprocessingMode::Nonlinear291),
    ] {
        let resolved = resolve_for_test(&parse(&format!(
            "run_dir = runs/mode-test\ngadgetization_mode = {value}\nmcd = 1\nallow_empty_store = true"
        )))
        .unwrap();
        assert_eq!(resolved.preprocessing_mode, expected);
        let args = script_args(&resolved);
        let mode_index = args
            .iter()
            .position(|arg| arg == OsStr::new("--gadgetization-mode"))
            .unwrap();
        assert_eq!(
            args.get(mode_index + 1),
            Some(&OsString::from(expected.as_str()))
        );
    }

    let canonical = resolve_for_test(&parse(
        "run_dir = runs/mode-test\ngadgetization_mode = product-2223\nallow_empty_store = true",
    ))
    .unwrap();
    let alias = resolve_for_test(&parse(
        "run_dir = runs/mode-test\ngadgetization_mode = 2223\nallow_empty_store = true",
    ))
    .unwrap();
    let fingerprints = [("gen_sandwich_gadget", 1u128)];
    assert_eq!(
        recipe_manifest(&canonical, &fingerprints, 2),
        recipe_manifest(&alias, &fingerprints, 2)
    );
    assert!(
        recipe_manifest(&alias, &fingerprints, 2).contains("gadgetization_mode=product-2223\n")
    );
}

#[test]
fn nonlinear_modes_reject_product_only_controls() {
    for product_setting in [
        "production_preset = production",
        "production_preset = micro-gray",
        "post_fragment = off",
        "post_fragment = exact",
    ] {
        let raw = parse(&format!(
            "run_dir = runs/mode-test\ngadgetization_mode = nonlinear193\n{product_setting}\nallow_empty_store = true"
        ));
        let error = resolve_for_test(&raw).err().unwrap();
        assert!(error.contains("product-2223"), "unexpected error: {error}");
    }

    let nonlinear = resolve_for_test(&parse(
        "run_dir = runs/mode-test\ngadgetization_mode = nonlinear291\nmcd = 1\nallow_empty_store = true",
    ))
    .unwrap();
    let manifest = recipe_manifest(&nonlinear, &[], 1);
    assert!(manifest.contains("gadgetization_mode=nonlinear291\n"));
    assert!(manifest.contains("production_preset=not-applicable\n"));
    assert!(manifest.contains("post_fragment=not-applicable\n"));
}

#[cfg(feature = "legacy-tools")]
#[test]
fn nonlinear_capacity_is_validated_before_launch() {
    let too_large = parse(
        "n = 64\nrun_dir = runs/mode-capacity\ngadgetization_mode = nonlinear291\nallow_empty_store = true",
    );
    let error = resolve_for_test(&too_large).err().unwrap();
    assert!(error.contains("wire capacity"), "{error}");
    assert!(error.contains("reduce n or set a smaller mcd"), "{error}");

    let derived_fit = parse(
        "n = 63\nrun_dir = runs/mode-capacity\ngadgetization_mode = nonlinear291\nallow_empty_store = true",
    );
    assert!(resolve_for_test(&derived_fit).is_ok());

    let explicit_fit = parse(
        "n = 128\nrun_dir = runs/mode-capacity\ngadgetization_mode = nonlinear291\nmcd = 1\nallow_empty_store = true",
    );
    assert!(resolve_for_test(&explicit_fit).is_ok());
}

#[test]
fn every_public_tuning_value_maps_to_a_separate_script_argument() {
    let raw = parse(
        "n = 64\nrun_dir = runs/with spaces;$(literal)\nmcd = 900\nexpand = 2.5\nhold = 4.5\nxr = 2.25\nxb = 4\nxc = 2\nxtdiv = 20\nxmoves = 1234\nstop_after = 5\nforce_from = 4\npieces = 4\npiece_threads = 6\nallow_empty_store = true",
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
        "--pieces",
        "--piece-threads",
        "6",
    ] {
        assert!(
            strings.iter().any(|actual| actual == expected),
            "missing {expected}"
        );
    }
    // The serial recipe passes no piece flags at all, so its stage command
    // lines stay byte-identical to the pre-feature driver.
    let serial = resolve_for_test(&parse(
        "n = 64\nrun_dir = runs/serial\nallow_empty_store = true",
    ))
    .unwrap();
    assert!(
        script_args(&serial)
            .iter()
            .all(|arg| !arg.to_string_lossy().starts_with("--piece")),
        "blank pieces must not emit piece flags"
    );
    assert!(resolved.run_dir.ends_with("runs/with spaces;$(literal)"));
}

#[test]
fn automatic_piece_config_validation_forwarding_and_manifest() {
    for value in ["0", "1", "-2", "2.5", "1000000001", "18446744073709551616"] {
        let raw = parse(&format!("min_block_size = {value}\nstop_after = 2"));
        assert!(
            resolve_for_test(&raw)
                .err()
                .unwrap()
                .contains("min_block_size")
        );
    }
    for pieces in ["1", "01", "4"] {
        let raw = parse(&format!(
            "pieces = {pieces}\nmin_block_size = 2\nstop_after = 2"
        ));
        assert!(
            resolve_for_test(&raw)
                .err()
                .unwrap()
                .contains("mutually exclusive")
        );
    }
    for size in ["2", "000128", "1000000000"] {
        let config = resolve_for_test(&parse(&format!(
            "pieces =\nmin_block_size = {size}\npiece_threads = 3\nstop_after = 2"
        )))
        .unwrap();
        let normalized = size.parse::<usize>().unwrap().to_string();
        let args = script_args(&config);
        assert!(
            args.windows(2)
                .any(|pair| pair[0] == "--min-block-size" && pair[1] == normalized.as_str())
        );
        assert!(
            args.windows(2)
                .any(|pair| pair[0] == "--piece-threads" && pair[1] == "3")
        );
        assert!(!args.iter().any(|arg| arg == "--pieces"));
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
    let serial = resolve_for_test(&parse("pieces =\nmin_block_size =\nstop_after = 2")).unwrap();
    assert!(serial.pieces.is_none() && serial.min_block_size.is_none());
    assert!(!recipe_manifest(&serial, &[], 1).contains("min_block_size"));
    assert!(
        !script_args(&serial)
            .iter()
            .any(|arg| arg == "--min-block-size")
    );
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
    fs::write(run_dir.join("gss.mpmct1"), "nonempty").unwrap();
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
    fs::remove_file(run_dir.join("gss.mpmct1")).unwrap();
    assert!(
        validate_external_paths(&resolved).is_err(),
        "missing stage 2 invalidates downstream artifacts"
    );
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
        "gadgetization_mode = invented",
        "production_preset = invented",
        "build_release = yes",
        "frozen_filter = maybe",
        "pieces = 0",
        "pieces = 65",
        "pieces = 1.5",
        "piece_threads = 4",
        "pieces = 1\npiece_threads = 2",
        "pieces = 2\npiece_threads = 0",
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
fn checked_in_toml_recipe_uses_quadratic_masking() {
    let parsed = parse_config(include_str!("../../configs/gss.toml")).unwrap();
    let config = resolve_for_test(&parsed).unwrap();
    assert_eq!(
        config.preprocessing_mode,
        RecipePreprocessingMode::QuadraticMasking
    );
    assert_eq!(
        (config.bv5_k, config.bv5_max_open, config.bv5_min_open),
        (2, 3, 2)
    );
    assert_eq!(config.hold.as_deref(), Some("27"));
    assert_eq!(config.recipe_version, 7);
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
        "run_dir = runs/manifest-test\nexpand = 2.0\nhold = 3e1\nxr = 2e0\nxb = 3.0\nxc = 1\nxtdiv = 25\npieces = 01\nstop_after = 2\nallow_empty_store = true",
    ))
    .unwrap();
    assert_eq!(
        expected,
        recipe_manifest(&explicit_defaults, &fingerprints, 4)
    );
    assert!(expected.starts_with("gss_command_recipe=3\n"));
    assert!(expected.contains("\npieces=1\n"));
    // `pieces` is a locked recipe value; `piece_threads` is lifecycle.
    let pieced = resolve_for_test(&parse(
        "run_dir = runs/manifest-test\npieces = 4\npiece_threads = 8\nstop_after = 2\nallow_empty_store = true",
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
    let desired = "gss_command_recipe=2\nn=128\n";
    fs::write(&path, desired).unwrap();
    compare_recipe_manifest(&path, desired).unwrap();

    let mismatch = compare_recipe_manifest(&path, "gss_command_recipe=2\nn=129\n").unwrap_err();
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
    assert_eq!(
        environment.get("PROD_PRESET"),
        Some(&Some("production".to_owned()))
    );

    let nonlinear = resolve_for_test(&parse(
        "run_dir = runs/env-test-nonlinear\ngadgetization_mode = nonlinear193\nmcd = 1\nstop_after = 2\nallow_empty_store = true",
    ))
    .unwrap();
    let mut command = ProcessCommand::new("true");
    configure_environment(&mut command, &nonlinear, Path::new("/tmp/gss-bin"));
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
    let config_path = base.join("manual.md");
    let run_dir = base.join("run");
    let target_dir = base.join("target");
    let store_dir = base.join("unused-store");
    fs::write(
        &config_path,
        document(&format!(
            "gadgetization_mode = ran-balanced\nrun_dir = {}\nbuild_target_dir = {}\nfrozen_db_dir = {}\nstop_after = 2",
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
    assert!(error.message.contains("calibration.enabled = true"));
    write_config(true);
    run_inner(&matches()).unwrap();
    assert!(!run_dir.join("gss_command.conf").exists());
    fs::remove_dir_all(base).unwrap();
}

#[test]
fn toml_rejects_unknown_keys_duplicates_and_wrong_types() {
    for input in [
        "config_version = 2",
        "config_version = 1\n[typo]",
        "config_version = 1\n[gadget]\nmode = \"ran-balanced\"\nmisspelled = 1",
        "config_version = 1\n[source]\nwires = \"128\"",
        "config_version = 1\n[run]\nbuild_release = \"true\"",
        "config_version = 1\n[source]\nwires = 4\nwires = 5",
    ] {
        assert!(parse_config(input).is_err(), "accepted {input}");
    }
}

#[test]
fn toml_default_hold_and_explicit_27_have_the_same_recipe() {
    let implicit = resolve_for_test(&parse_config("config_version = 1").unwrap()).unwrap();
    let explicit =
        resolve_for_test(&parse_config("config_version = 1\n[phase_a]\nhold = 27.0").unwrap())
            .unwrap();
    assert_eq!(
        recipe_manifest(&implicit, &[], 7),
        recipe_manifest(&explicit, &[], 7)
    );
    assert!(recipe_manifest(&implicit, &[], 7).contains("hold=27\n"));
    assert!(recipe_manifest(&implicit, &[], 7).starts_with("gss_command_recipe=7\n"));
}

#[test]
fn v3_script_remains_the_current_checkpoint_driver() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let script = script_for_recipe(root, 3);
    assert_eq!(script, root.join("scripts/compat/gss_mix_v3.sh"));
    assert_ne!(
        hash_file(&script).unwrap(),
        hash_file(&script_for_recipe(root, 4)).unwrap()
    );
    let config = resolve_for_test(&parse("run_dir = runs/compat-check")).unwrap();
    assert!(recipe_manifest(&config, &[], 7).starts_with("gss_command_recipe=3\n"));
}

#[test]
fn managed_v3_dry_run_uses_recorded_driver_and_rejects_changed_binary() {
    let base = temp_path("managed-v3");
    let run_dir = base.join("run");
    let target_dir = base.join("target");
    let binary_dir = target_dir
        .join(rustc_host_triple().unwrap())
        .join("release");
    fs::create_dir_all(&run_dir).unwrap();
    fs::create_dir_all(&binary_dir).unwrap();
    for name in ["gen_sandwich_gadget", "fmix", "fcompress"] {
        fs::write(binary_dir.join(name), format!("original {name}\n")).unwrap();
    }
    let recipe = document(&format!(
        "n = 8\nmcd = 1\nrun_dir = {}\nbuild_target_dir = {}\nfrozen_db_dir = {}\nstop_after = 2",
        run_dir.display(),
        target_dir.display(),
        base.join("unused-store").display()
    ));
    let config_path = base.join("original.md");
    fs::write(&config_path, &recipe).unwrap();
    let config = resolve_for_test(&parse_config(&recipe).unwrap()).unwrap();
    let mut child = ProcessCommand::new("bash");
    configure_environment(&mut child, &config, &binary_dir);
    assert!(child.get_envs().all(|(key, _)| !matches!(
        key.to_str(),
        Some("SANDWICH_VARIANT" | "GSS_SOURCE_C" | "DB_QC" | "DB_QC_SEED" | "DB_QC_REFERENCE")
    )));
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let original_manifest = recipe_manifest(
        &config,
        &production_binary_fingerprints(&binary_dir).unwrap(),
        hash_file(&script_for_recipe(root, 3)).unwrap(),
    );
    let manifest_path = run_dir.join("gss_command.conf");
    fs::write(&manifest_path, &original_manifest).unwrap();
    let matches = command()
        .try_get_matches_from([
            OsString::from("gss"),
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
    fs::write(binary_dir.join("fmix"), "changed binary\n").unwrap();
    assert!(run_inner(&matches).is_err());
    assert_eq!(
        fs::read_to_string(&manifest_path).unwrap(),
        original_manifest
    );
    fs::remove_dir_all(base).unwrap();
}

#[test]
fn retired_fresh_modes_are_rejected_before_artifacts_or_builds() {
    let base = temp_path("retired-product");
    fs::create_dir_all(&base).unwrap();
    let run_dir = base.join("run");
    let target_dir = base.join("target");
    let config_path = base.join("recipe");
    for mode in ["product-2223", "2223", "nonlinear193"] {
        for toml in [false, true] {
            let recipe = if toml {
                format!(
                    "config_version = 1\n[source]\nwires = 4\ngates = 1\n[run]\ndirectory = {:?}\nbuild_target_dir = {:?}\n[gadget]\nmode = {mode:?}",
                    run_dir, target_dir,
                )
            } else {
                document(&format!(
                    "n = 4\nmcd = 1\nrun_dir = {}\nbuild_target_dir = {}\ngadgetization_mode = {mode}",
                    run_dir.display(),
                    target_dir.display(),
                ))
            };
            fs::write(&config_path, recipe).unwrap();
            let matches = command()
                .try_get_matches_from([
                    OsString::from("gss"),
                    OsString::from("--dry-run"),
                    OsString::from("--config"),
                    config_path.clone().into_os_string(),
                ])
                .unwrap();
            let error = run_inner(&matches).unwrap_err();
            assert!(
                error.message.contains("supported only for existing runs"),
                "{error:?}"
            );
            assert!(!run_dir.exists());
            assert!(!target_dir.exists());
        }
    }
    fs::remove_dir_all(base).unwrap();
}

#[test]
fn current_ran_recipe_rejects_silently_clamped_k_and_insufficient_band() {
    for mode in [
        "quadratic-masking",
        "ran-balanced",
        "blinded-v5",
        "blinded_v5",
    ] {
        for (n, k, valid) in [(4, 1, false), (4, 2, true), (4, 7, true), (4, 8, false)] {
            let recipe = format!(
                "config_version = 1\n[source]\nwires = {n}\n[gadget]\nmode = {mode:?}\nk = {k}"
            );
            let config = resolve_for_test(&parse_config(&recipe).unwrap()).unwrap();
            assert_eq!(validate_current_recipe(&config).is_ok(), valid, "{recipe}");
        }
    }
}

#[test]
fn managed_v4_v5_v6_keep_their_driver_and_resume_original_recipes() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let old_script = script_for_recipe(root, 4);
    assert_eq!(old_script, root.join("scripts/compat/gss_mix_v4.sh"));
    assert_eq!(
        script_for_recipe(root, 6),
        root.join("scripts/compat/gss_mix_v6.sh")
    );
    assert_eq!(script_for_recipe(root, 7), root.join("scripts/gss_mix.sh"));
    assert_ne!(
        hash_file(&old_script).unwrap(),
        hash_file(&script_for_recipe(root, 6)).unwrap()
    );
    for (version, mode) in [
        (4, "product-2223"),
        (4, "2223"),
        (4, "ran-balanced"),
        (5, "ran-balanced"),
        (6, "ran-balanced"),
        (6, "quadratic-masking"),
        (6, "nonlinear193"),
        (6, "nonlinear291"),
    ] {
        let old_script = script_for_recipe(root, version);
        let base = temp_path("managed-compatible");
        let run_dir = base.join("run");
        let target_dir = base.join("target");
        let binary_dir = target_dir
            .join(rustc_host_triple().unwrap())
            .join("release");
        fs::create_dir_all(&run_dir).unwrap();
        fs::create_dir_all(&binary_dir).unwrap();
        for name in ["gen_sandwich_gadget", "fmix", "fcompress"] {
            fs::write(binary_dir.join(name), format!("original {name}\n")).unwrap();
        }
        let recipe = format!(
            "config_version = 1\n[source]\nwires = 4\ngates = 1\n[run]\ndirectory = {:?}\nbuild_target_dir = {:?}\nstop_after = 2\n[gadget]\nmode = {mode:?}",
            run_dir, target_dir,
        );
        let config_path = base.join("original.toml");
        fs::write(&config_path, &recipe).unwrap();
        let mut config = resolve_for_test(&parse_config(&recipe).unwrap()).unwrap();
        config.recipe_version = version;
        let original_manifest = recipe_manifest(
            &config,
            &production_binary_fingerprints(&binary_dir).unwrap(),
            hash_file(&old_script).unwrap(),
        );
        let manifest_path = run_dir.join("gss_command.conf");
        fs::write(&manifest_path, &original_manifest).unwrap();
        let matches = command()
            .try_get_matches_from([
                OsString::from("gss"),
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
        fs::write(binary_dir.join("fmix"), "changed binary\n").unwrap();
        assert!(run_inner(&matches).is_err());
        assert_eq!(
            fs::read_to_string(&manifest_path).unwrap(),
            original_manifest
        );
        fs::remove_dir_all(base).unwrap();
    }
}

#[test]
fn canonical_and_legacy_toml_names_normalize_and_reject_alias_duplicates() {
    for &(canonical, old, flat, kind) in crate::gss::config::TOML_FIELDS {
        let value = match kind {
            "string" => "\"example\"",
            "bool" => "true",
            "integer" => "4",
            "number" => "2.5",
            _ => unreachable!(),
        };
        let canonical_raw =
            parse_config(&format!("config_version = 1\n{canonical} = {value}")).unwrap();
        let (section, key) = old.split_once('.').unwrap();
        let old_raw =
            parse_config(&format!("config_version = 1\n[{section}]\n{key} = {value}")).unwrap();
        assert_eq!(canonical_raw.entries.len(), 1);
        assert_eq!(
            canonical_raw.value(flat),
            old_raw.value(flat),
            "{canonical}"
        );
        if canonical != old {
            let error = parse_config(&format!(
                "config_version = 1\n{old} = {value}\n{canonical} = {value}"
            ))
            .unwrap_err();
            assert!(error.contains("duplicate GSS setting"), "{error}");
            assert!(error.contains(canonical), "{error}");
        }
    }
}

#[test]
fn canonical_recipe_preserves_normalized_values_and_flag_values() {
    let old = "config_version = 1\nrun.directory = \"runs/naming-test\"\nphase_a.expand = 3.0\nphase_a.hold = 7.0\nparallel.min_block_size = 10000\nparallel.threads = 4\ngadget.k = 4\nquality_control.enabled = true\nquality_control.seed = 99\ncrossing.moves = 42";
    let new = "config_version = 1\nrun.directory = \"runs/naming-test\"\ndb_mixing.target_size_factor = 3.0\ndb_mixing.hold_work_units = 7.0\nparallel.target_piece_gates = 10000\nparallel.threads = 4\ngadget.mask_pair_wires = 4\nleakage_repair.enabled = true\nleakage_repair.seed = 99\ncrossing.move_attempts = 42";
    let old = resolve_for_test(&parse_config(old).unwrap()).unwrap();
    let mut new = resolve_for_test(&parse_config(new).unwrap()).unwrap();
    assert_eq!(recipe_manifest(&old, &[], 7), recipe_manifest(&new, &[], 7));
    // Translate option positions only, never a value that happens to spell an old flag.
    new.run_dir = PathBuf::from("--pieces");
    new.calibration_seed = Some("123456789".into());
    let args = script_args(&new);
    assert_eq!(args[2], "--run-directory");
    assert_eq!(args[3], "--pieces");
    assert!(args.contains(&OsString::from("--db-mixing-target-size-factor")));
    assert!(args.contains(&OsString::from("--parallel-target-piece-gates")));
    for version in [3, 4, 5, 6, 7] {
        new.recipe_version = version;
        let args = script_args(&new);
        assert_eq!(args[0], if version >= 6 { "--source-wires" } else { "-n" });
        let rendered = render_command(&args, Some("-s"));
        assert!(rendered.contains("<redacted>"));
        assert!(!rendered.contains("123456789"));
    }
}

#[test]
fn completed_db_mixing_uses_version_specific_artifacts() {
    let root = temp_path("db-mixing-artifacts");
    fs::create_dir_all(&root).unwrap();
    let mut config = resolve_for_test(&parse_config("config_version = 1").unwrap()).unwrap();
    config.run_dir = root.clone();
    fs::write(root.join("gss.mpmct1"), "saved gadget").unwrap();
    // A stale legacy name cannot bypass current-run database prerequisites.
    fs::write(root.join("phaseA.mpmct1"), "saved stage3").unwrap();
    assert!(validate_external_paths(&config).is_err());
    config.recipe_version = 5;
    validate_external_paths(&config).unwrap();
    fs::remove_file(root.join("phaseA.mpmct1")).unwrap();
    fs::write(root.join("db_mixing.mpmct1"), "saved stage3").unwrap();
    assert!(validate_external_paths(&config).is_err());
    config.recipe_version = 6;
    validate_external_paths(&config).unwrap();
    config.force_from = Some("3".into());
    assert!(validate_external_paths(&config).is_err());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn preprocessing_aliases_and_v7_identity_are_canonical() {
    let mut baseline = None;
    for section in ["preprocessing", "gadget"] {
        for mode in [
            "quadratic-masking",
            "ran-balanced",
            "blinded-v5",
            "blinded_v5",
        ] {
            let raw =
                parse_config(&format!("config_version = 1\n[{section}]\nmode = {mode:?}")).unwrap();
            let config = resolve_for_test(&raw).unwrap();
            validate_current_recipe(&config).unwrap();
            assert_eq!(config.recipe_version, 7);
            let manifest = recipe_manifest(&config, &[], 1);
            assert!(manifest.contains("preprocessing_mode=quadratic-masking\n"));
            assert!(manifest.contains("preprocessing_balanced_masks=1\n"));
            assert!(!manifest.contains("gadgetization_mode="));
            assert!(!manifest.contains("bv5_"));
            if let Some(expected) = &baseline {
                assert_eq!(expected, &manifest);
            } else {
                baseline = Some(manifest);
            }
            let args = script_args(&config);
            assert_eq!(args[4], "--preprocessing-mode");
            assert_eq!(args[5], "quadratic-masking");
        }
    }
}

#[test]
fn nonlinear291_rejects_only_explicit_mask_controls_for_v7() {
    let base = "config_version = 1\nsource.wires = 4\nsource.gates = 1\npreprocessing.mode = \"nonlinear291\"\n";
    let config = resolve_for_test(&parse_config(base).unwrap()).unwrap();
    validate_current_recipe(&config).unwrap();
    assert!(!config.explicit_mask_controls);
    assert!(config.preprocessing_mode.supported().is_some());
    assert!(!recipe_manifest(&config, &[], 1).contains("preprocessing_mask_pair_wires"));
    for setting in [
        "preprocessing.mask_pair_wires = 2",
        "preprocessing.max_open_masks = 3",
        "preprocessing.min_open_masks = 2",
        "preprocessing.balanced_masks = true",
        "gadget.k = 2",
        "gadget.mask_pair_wires = 2",
        "gadget.max_open = 3",
        "gadget.min_open_masks = 2",
        "gadget.balanced_masks = false",
    ] {
        let raw = parse_config(&format!("{base}{setting}")).unwrap();
        let mut config = resolve_for_test(&raw).unwrap();
        let error = validate_current_recipe(&config).unwrap_err();
        assert!(
            error.message.contains("mask controls apply only"),
            "{setting}: {error:?}"
        );
        // Old nonlinear recipes recorded but did not apply the three integer fields.
        if !setting.contains("balanced_masks") {
            config.recipe_version = 6;
            validate_current_recipe(&config).unwrap();
        }
    }
}

#[test]
fn quadratic_balancing_is_typed_pinned_and_locked_in_v7() {
    let raw = parse_config("config_version = 1\n[preprocessing]\nbalanced_masks = false").unwrap();
    let mut config = resolve_for_test(&raw).unwrap();
    validate_current_recipe(&config).unwrap();
    assert!(!config.bv5_balanced);
    let manifest = recipe_manifest(&config, &[], 1);
    assert!(manifest.contains("preprocessing_balanced_masks=0\n"));
    let mut command = ProcessCommand::new("bash");
    configure_environment(&mut command, &config, Path::new("/tmp/gss-bin"));
    assert!(
        command
            .get_envs()
            .any(|(key, value)| key == "BV5_BALANCED" && value == Some(OsStr::new("0")))
    );
    config.bv5_balanced = true;
    assert_ne!(manifest, recipe_manifest(&config, &[], 1));
    config.bv5_balanced = false;
    config.recipe_version = 6;
    assert!(
        validate_current_recipe(&config)
            .unwrap_err()
            .message
            .contains("pin balanced masks")
    );
}

#[test]
fn old_manifest_field_order_and_mode_spelling_are_unchanged() {
    let raw = parse_config("config_version = 1\nsource.wires = 4").unwrap();
    let mut config = resolve_for_test(&raw).unwrap();
    // Literal fixture from the v6 serializer before this refactor. Paths, source,
    // seed and binaries are absent so the fixture is independent of the checkout.
    let prefix = concat!(
        "n=4\nfrozen_db_dir=unset\nfrozen_curated_dir=unset\n",
        "curated_value_convention=native\ngadgetization_mode=ran-balanced\n",
        "production_preset=not-applicable\npost_fragment=not-applicable\n",
        "mcd=derived\nexpand=2\nhold=27\nxr=2\nxb=3\nxc=1\nxtdiv=25\n",
        "xmoves=6*target\npieces=1\nallow_empty_store=false\ncalibration_only=false\n",
        "pinned_recipe=canon512/200000-cache256/1024/2048-native-regular\n",
    );
    let v4_fields = concat!(
        "sandwich_variant=classic\nbv5_k=2\nbv5_max_open=3\nbv5_min_open=2\n",
        "bv5_balanced=1\nbv5_quad_fire=1\nbv5_extra_lgis=0\nbv5_encoded_io=false\n",
        "source_path=unset\nsource_xxh3=generated\nqc_enabled=false\nqc_seed=20803\n",
        "qc_reference=unset\nqc_reference_xxh3=unset\n",
    );
    for version in [3, 4, 5, 6] {
        config.recipe_version = version;
        let expected = format!(
            "gss_command_recipe={version}\n{prefix}{}script_gss_mix_xxh3={:032x}\n",
            if version >= 4 { v4_fields } else { "" },
            1u128
        );
        assert_eq!(recipe_manifest(&config, &[], 1), expected);
        let args = script_args(&config);
        assert_eq!(
            args[4],
            if version == 6 {
                "--gadget-mode"
            } else {
                "--gadgetization-mode"
            }
        );
        assert_eq!(args[5], "ran-balanced");
    }
}

#[test]
#[cfg(unix)]
fn fresh_v7_managed_runs_record_and_resume_both_preprocessing_modes() {
    use std::os::unix::fs::PermissionsExt;
    for mode in ["quadratic-masking", "nonlinear291"] {
        let base = temp_path("fresh-v7");
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
        for name in ["gen_sandwich_gadget", "fmix", "fcompress"] {
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
                OsString::from("gss"),
                OsString::from("--config"),
                config_path.into_os_string(),
            ])
            .unwrap();
        run_inner(&matches).unwrap();
        let manifest_path = run_dir.join("gss_command.conf");
        let manifest = fs::read_to_string(&manifest_path).unwrap();
        assert!(manifest.starts_with("gss_command_recipe=7\n"));
        assert!(manifest.contains(&format!("preprocessing_mode={mode}\n")));
        assert_eq!(read_recipe_version(&manifest_path).unwrap(), 7);
        let stage_marker = fs::read_to_string(run_dir.join("stage12.recipe")).unwrap();
        assert!(stage_marker.contains(&format!("preprocessing_mode={mode}\n")));
        run_inner(&matches).unwrap();
        assert_eq!(fs::read_to_string(manifest_path).unwrap(), manifest);
        assert_eq!(
            fs::read_to_string(run_dir.join("gss.mpmct1.calls")).unwrap(),
            "called\n"
        );
        fs::remove_dir_all(base).unwrap();
    }
}
