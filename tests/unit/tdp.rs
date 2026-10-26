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
    parse_config(&format!("config_version = 5\n{body}")).unwrap()
}

fn resolve_for_test(raw: &RawConfig) -> Result<ResolvedConfig, String> {
    resolve_config(
        raw,
        Path::new(env!("CARGO_MANIFEST_DIR")),
        |_| None,
        "test-run",
    )
}

fn resolve_body(body: &str) -> Result<ResolvedConfig, String> {
    parse_config(&format!("config_version = 5\n{body}")).and_then(|raw| resolve_for_test(&raw))
}

fn strings(args: &[OsString]) -> Vec<String> {
    args.iter()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect()
}

fn private_file(path: &Path, text: &str) {
    fs::write(path, text).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
    }
}

const ALL_CUSTOM: &str = "preprocessing.preset = \"custom\"\npreprocessing.mask_pair_wires = 2\npreprocessing.max_open_masks = 3\npreprocessing.min_open_masks = 2\npreprocessing.balanced_masks = true\npreprocessing.shuffling_segments = 8\npreprocessing.shuffling_return_home = false\npreprocessing.premask = true\npreprocessing.seed_avoid_c1 = true\npreprocessing.chain = 2\npreprocessing.chain_len = 8\n";

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
fn omitted_values_resolve_to_the_chosen_l8_runs() {
    // The defaults requirement (docs/STAGE4_RUST_DESIGN.md §1): nothing but
    // source.wires reproduces val_chain.sh CONFIG=L8.
    let resolved = resolve_for_test(&parse("")).unwrap();
    assert_eq!(resolved.n, 128);
    assert!(resolved.generated_run_dir);
    assert!(resolved.run_dir.ends_with("runs/tdp_gen_n128_test-run"));
    assert!(resolved.build_release);
    assert_eq!(
        resolved.preprocessing_mode,
        PreprocessingMode::EmbeddedMasking
    );
    assert_eq!(resolved.preset, Preset::L8);
    assert_eq!(resolved.knobs, Stage2Knobs::L8);
    assert_eq!(resolved.schedule, Schedule::Validated);
    assert_eq!(
        (resolved.mcd, resolved.mcd_rule),
        (3000, SourceGatesRule::Validated)
    );
    assert_eq!(resolved.stop_after_stage(), 6);
    assert!(resolved.skip_stages.is_empty());
    assert_eq!(resolved.stages_text(), "1-6");
    assert!(resolved.packing.is_default());
    assert!(resolved.delivery.is_default());
    assert!(resolved.adaptive.is_default());
    assert_eq!(resolved.delivery.mode, DeliveryMode::Adaptive);
    assert!(resolved.adaptive_on() && !resolved.unpacking_on());
    assert!(matches!(resolved.seeds, SeedPlan::Random));
    let args = strings(&script_args(&resolved));
    assert!(
        args.windows(2)
            .any(|w| w == ["--preprocessing-preset", "L8"])
    );
    // nothing else steers the recipe: no size, schedule, knob, packing,
    // delivery or seed flag
    for arg in &args {
        assert!(
            !arg.starts_with("--source-gates")
                && !arg.starts_with("--source-construction")
                && !arg.starts_with("--three-mask")
                && !arg.starts_with("--db-mixing")
                && !arg.starts_with("--packing")
                && !arg.starts_with("--delivery")
                && !arg.starts_with("--adaptive")
                && !arg.contains("seed")
                && !(arg.starts_with("--preprocessing-")
                    && arg != "--preprocessing-mode"
                    && arg != "--preprocessing-preset"),
            "{arg} in {args:?}"
        );
    }
    // L8 = val_chain.sh's stage-2 environment
    assert_eq!(
        (
            Stage2Knobs::L8.shuffling_segments,
            Stage2Knobs::L8.shuffling_return_home,
            Stage2Knobs::L8.premask,
            Stage2Knobs::L8.seed_avoid_c1,
            Stage2Knobs::L8.chain,
            Stage2Knobs::L8.chain_len
        ),
        (8, false, true, true, 2, 8)
    );
    assert_eq!(
        (
            Stage2Knobs::L8.mask_pair_wires,
            Stage2Knobs::L8.max_open_masks
        ),
        (2, 3)
    );
}

#[test]
fn default_source_gates_table() {
    let at = |body: &str| resolve_body(&format!("run.stop_after_stage = 2\n{body}"));
    let c = at("source.wires = 64").unwrap();
    assert_eq!((c.mcd, c.mcd_rule), (1000, SourceGatesRule::Validated));
    let c = at("source.wires = 128").unwrap();
    assert_eq!((c.mcd, c.mcd_rule), (3000, SourceGatesRule::Validated));
    let error = at("source.wires = 32").err().unwrap();
    assert!(
        error.contains("no validated source size for n=32"),
        "{error}"
    );
    let c = at("source.wires = 32\nsource.gates_rule = \"formula\"").unwrap();
    assert_eq!((c.mcd, c.mcd_rule), (800, SourceGatesRule::Formula));
    assert!(
        strings(&script_args(&c))
            .windows(2)
            .any(|w| w == ["--source-gates-rule", "formula"])
    );
    assert!(
        recipe_manifest(&c, &[], 1).contains("\nsource_gates=800\nsource_gates_rule=formula\n")
    );
    let c = at("source.wires = 64\nsource.gates = 1000").unwrap();
    assert_eq!(c.mcd_rule, SourceGatesRule::Validated);
    let c = at("source.wires = 64\nsource.gates = 999").unwrap();
    assert_eq!((c.mcd, c.mcd_rule), (999, SourceGatesRule::Explicit));
    assert!(
        strings(&script_args(&c))
            .windows(2)
            .any(|w| w == ["--source-gates", "999"])
    );
    assert!(at("source.wires = 64\nsource.gates = 999\nsource.gates_rule = \"formula\"").is_err());
    assert!(at("source.wires = 32\nsource.gates_rule = \"guess\"").is_err());
}

#[test]
fn nonlinear_capacity_is_validated_before_launch() {
    let too_large =
        parse("source.wires = 64\nsource.gates = 2304\npreprocessing.mode = \"nonlinear291\"");
    let error = resolve_for_test(&too_large).err().unwrap();
    assert!(error.contains("wire capacity"), "{error}");
    let formula_fit = parse(
        "source.wires = 63\nsource.gates_rule = \"formula\"\npreprocessing.mode = \"nonlinear291\"\nrun.stop_after_stage = 3",
    );
    assert!(resolve_for_test(&formula_fit).is_ok());
    let explicit_fit = parse(
        "source.wires = 128\nsource.gates = 1\npreprocessing.mode = \"nonlinear291\"\nrun.stop_after_stage = 3",
    );
    assert!(resolve_for_test(&explicit_fit).is_ok());
}

/// The monomial and split budgets are resource bounds, shared with
/// `tdp_packing` (MAX_MONOMIAL_BUDGET): refused before anything runs.
#[test]
fn packing_budgets_are_bounded_by_the_resource_limit() {
    for key in ["monomial_budget", "split_budget"] {
        let windowed = "delivery.mode = \"windowed\"\n";
        assert!(resolve_for_test(&parse(&format!("{windowed}packing.{key} = 262144"))).is_ok());
        let error = resolve_for_test(&parse(&format!("{windowed}packing.{key} = 262145")))
            .err()
            .unwrap();
        assert!(error.contains("262144"), "{error}");
    }
}

/// nonlinear291's width (24n + 12(2|C|+2s+n) + 29) is checked against stage
/// 4's 1024 state wires before anything runs, not after stage 3.
#[test]
fn nonlinear_width_beyond_stage_four_is_refused_up_front() {
    let wide = "source.wires = 16\nsource.gates = 64\npreprocessing.mode = \"nonlinear291\"\n";
    let error = resolve_for_test(&parse(wide)).err().unwrap();
    assert!(error.contains("3677-wire"), "{error}");
    assert!(error.contains("at most 1024 state wires"), "{error}");
    assert!(resolve_for_test(&parse(&format!("{wide}run.stop_after_stage = 3"))).is_ok());
    // n=4, |C|=1, s=8: 389 wires, within stage 4
    let narrow = "source.wires = 4\nsource.gates = 1\npreprocessing.mode = \"nonlinear291\"\n";
    assert!(resolve_for_test(&parse(narrow)).is_ok());
    // embedded masking is bounded by n <= 256 alone
    assert!(resolve_for_test(&parse("source.wires = 16\nsource.gates = 64\n")).is_ok());
}

#[test]
fn every_public_tuning_value_maps_to_a_separate_script_argument() {
    let raw = parse(
        "source.wires = 64\nrun.directory = \"runs/with spaces;$(literal)\"\nsource.gates = 900\ndb_mixing.schedule = \"custom\"\ndb_mixing.growth_work_units = 3\ndb_mixing.target_size_factor = 2.5\ndb_mixing.hold_work_units = 4.5\nrun.stop_after_stage = 3\nrun.rerun_from_stage = 2\nparallel.pieces = 4\nparallel.threads = 6\ndatabase.allow_no_database_for_tests = true\npacking.split_budget = 4096\npacking.allow_floor_miss = true\ndelivery.mode = \"windowed\"\ndelivery.time_limit = 12.5\ndelivery.max_rounds = 7\ndelivery.anf_only = true\nrun.threads = 3\nsource.construction = \"balanced\"",
    );
    let resolved = resolve_for_test(&raw).unwrap();
    let args = strings(&script_args(&resolved));
    for pair in [
        ["--source-wires", "64"],
        ["--source-gates", "900"],
        ["--db-mixing-schedule", "custom"],
        ["--db-mixing-growth-work-units", "3"],
        ["--db-mixing-target-size-factor", "2.5"],
        ["--db-mixing-hold-work-units", "4.5"],
        ["--run-stop-after-stage", "3"],
        ["--run-rerun-from-stage", "2"],
        ["--parallel-pieces", "4"],
        ["--parallel-threads", "6"],
        ["--packing-split-budget", "4096"],
        ["--delivery-mode", "windowed"],
        ["--delivery-time-limit", "12.5"],
        ["--delivery-max-rounds", "7"],
        ["--threads", "3"],
        ["--source-construction", "balanced"],
    ] {
        assert!(
            args.windows(2).any(|w| w == pair),
            "missing {pair:?} in {args:?}"
        );
    }
    assert!(args.iter().any(|a| a == "--packing-allow-floor-miss"));
    assert!(args.iter().any(|a| a == "--delivery-anf-only"));
    assert!(!args.iter().any(|a| a == "--packing-window-gates"));
    assert!(!args.iter().any(|a| a == "--delivery-restarts"));
    assert!(!resolved.unpacking_on());
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
        let result = resolve_body(&format!(
            "parallel.target_piece_gates = {value}\nrun.stop_after_stage = 2"
        ));
        assert!(result.is_err(), "accepted {value}");
    }
    for pieces in ["1", "4"] {
        let error = resolve_body(&format!(
            "parallel.pieces = {pieces}\nparallel.target_piece_gates = 2\nrun.stop_after_stage = 2"
        ))
        .err()
        .unwrap();
        assert!(error.contains("mutually exclusive"));
    }
    for size in ["2", "128", "1000000000"] {
        let config = resolve_body(&format!(
            "parallel.target_piece_gates = {size}\nparallel.threads = 3\nrun.stop_after_stage = 2"
        ))
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
        let manifest = recipe_manifest(&config, &[], 1);
        assert!(manifest.contains(&format!("\nmin_block_size={normalized}\n")));
        assert!(manifest.contains("\npieces=auto\n"));
        assert!(!manifest.contains("piece_threads"));
        let mut changed = config.clone();
        changed.piece_threads = None;
        assert_eq!(manifest, recipe_manifest(&changed, &[], 1));
        changed.min_block_size = Some("3".into());
        assert_ne!(manifest, recipe_manifest(&changed, &[], 1));
    }
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

fn seeds(gen_seed: u64, mix: u64) -> RunSeeds {
    RunSeeds {
        gen_seed,
        mix,
        split: 3,
        cross: 4,
        unpack: 5,
    }
}

#[test]
fn resume_after_db_mixing_does_not_require_the_stage_three_store() {
    let run_dir = temp_path("resume");
    fs::create_dir_all(&run_dir).unwrap();
    fs::write(run_dir.join("db_mixing.mpmct1"), "nonempty").unwrap();
    fs::write(run_dir.join("tdp.mpmct1"), "nonempty").unwrap();
    private_file(
        &run_dir.join("SEEDS"),
        &seeds_text(seeds(1, 2), &[7u8; 32], "explicit"),
    );
    let raw = parse(&format!("run.directory = {:?}", run_dir));
    let resolved = resolve_for_test(&raw).unwrap();
    // an output without a completion marker (interrupted, partial) re-runs
    // stage 3, which needs the store
    assert!(validate_external_paths(&resolved).is_err());
    fs::write(run_dir.join("stage3.recipe"), "tdp_stage3_recipe=1\n").unwrap();
    validate_external_paths(&resolved).unwrap();
    let forced = resolve_for_test(&parse(&format!(
        "run.directory = {:?}\nrun.rerun_from_stage = 3",
        run_dir
    )))
    .unwrap();
    assert!(validate_external_paths(&forced).is_err());
    // stage 4 alone needs no store
    let stage4 = resolve_for_test(&parse(&format!(
        "run.directory = {:?}\nrun.rerun_from_stage = 4",
        run_dir
    )))
    .unwrap();
    validate_external_paths(&stage4).unwrap();
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
        "source.wires = 512",
        "db_mixing.schedule = \"custom\"\ndb_mixing.growth_work_units = 3\ndb_mixing.hold_work_units = 27\ndb_mixing.target_size_factor = 1",
        "db_mixing.schedule = \"custom\"\ndb_mixing.growth_work_units = 3\ndb_mixing.hold_work_units = -1\ndb_mixing.target_size_factor = 2",
        "db_mixing.schedule = \"invented\"",
        "run.stop_after_stage = 8",
        "run.rerun_from_stage = 8",
        "run.stop_after_stage = 7",
        "run.rerun_from_stage = 7",
        "run.stop_after_stage = 1",
        "run.rerun_from_stage = 1",
        "run.skip_stages = [1]",
        "run.skip_stages = [7]",
        "run.skip_stages = [8]",
        "run.skip_stages = [\"4\"]",
        "run.skip_stages = 4",
        "crossing.size_tolerance_divisor = 0",
        "crossing.target_size_factor = 0.5",
        "crossing.target_size_factor = 17",
        "crossing.width_penalty_base = 0",
        "crossing.move_attempts = 0",
        "crossing.move_budget_factor = 0",
        "crossing.move_budget_factor = 1001",
        "crossing.move_budget_factor = 2\ncrossing.move_attempts = 5",
        "delivery.mode = \"windowed\"\ndelivery.time_limit = 0",
        "delivery.mode = \"windowed\"\ndelivery.time_limit = 86401",
        "delivery.mode = \"windowed\"\ndelivery.restarts = 0",
        "delivery.mode = \"windowed\"\ndelivery.beam = 0",
        "delivery.mode = \"windowed\"\ndelivery.max_rounds = 0",
        "delivery.mode = \"windowed\"\ndelivery.max_rounds = 1001",
        "delivery.mode = \"windowed\"\ndelivery.anf_only = \"no\"",
        "preprocessing.mode = \"invented\"",
        "preprocessing.preset = \"L9\"",
        "run.build_binaries = \"yes\"",
        "database.lookup_miss_filter = \"maybe\"",
        "parallel.pieces = 0",
        "parallel.pieces = 65",
        "parallel.pieces = 1.5",
        "parallel.threads = 4",
        "parallel.pieces = 1\nparallel.threads = 2",
        "parallel.pieces = 2\nparallel.threads = 0",
        "delivery.mode = \"windowed\"\npacking.window_gates = 0",
        "delivery.mode = \"windowed\"\npacking.anf_cap_semantics = \"sorted\"",
        "delivery.mode = \"windowed\"\npacking.allow_floor_miss = 1",
        "run.threads = 0",
        // the stage-6 mode decides which parameter family applies
        "delivery.mode = \"invented\"",
        "delivery.anf_only = true",
        "delivery.time_limit = 30",
        "packing.window_gates = 512",
        "delivery.mode = \"windowed\"\nadaptive.max_rounds = 3",
        "adaptive.max_rounds = 1001",
        "adaptive.stop_fraction = 1",
        "adaptive.stop_fraction = -0.1",
        "adaptive.seeds_per_block = 0",
        "adaptive.seeds_per_block = 17",
        "adaptive.time_limit = 0",
        "adaptive.restarts = 0",
        "adaptive.beam = 0",
        "adaptive.term_cap = 63",
        "adaptive.max_block_gates = 1",
        "adaptive.segments = 0",
        "adaptive.refinement_levels = 17",
        "adaptive.min_refine_gates = 0",
        "adaptive.census_batches = 0",
        "adaptive.accept = \"largest\"",
        "adaptive.canonical_order = 1",
        // the source construction
        "source.construction = \"tidy\"",
        "source.construction = \"balanced\"\nsource.path = \"circuits/source.g57\"",
        // [three_mask] only under its mode, and its geometry
        "three_mask.banks = 8",
        "preprocessing.mode = \"three-mask\"\npreprocessing.preset = \"L8\"",
        "preprocessing.mode = \"three-mask\"\npreprocessing.chain = 2",
        "preprocessing.mode = \"three-mask\"\nthree_mask.band_wires = 48",
        "preprocessing.mode = \"three-mask\"\nthree_mask.band_wires = 72",
        "preprocessing.mode = \"three-mask\"\nthree_mask.banks = 7",
        "preprocessing.mode = \"three-mask\"\nthree_mask.banks = 2",
        "preprocessing.mode = \"three-mask\"\nthree_mask.max_retiring = 4",
        "preprocessing.mode = \"three-mask\"\nthree_mask.mask_count = 4",
        "preprocessing.mode = \"three-mask\"\nthree_mask.random_helpers = 1",
    ] {
        assert!(resolve_body(body).is_err(), "unexpectedly accepted {body}");
    }
    // the former stage 7 is part of stage 6 now: the message says so
    let error = resolve_body("run.skip_stages = [7]").err().unwrap();
    assert!(error.contains("2..6"), "{error}");
    assert!(error.contains("one delivery stage 6"), "{error}");
    // every stage 2..6 is a valid stop and rerun point; larger n is allowed
    // when stage 6 (delivery) does not run
    for k in 2..=6 {
        resolve_body(&format!(
            "run.stop_after_stage = {k}\nrun.rerun_from_stage = {k}"
        ))
        .unwrap();
    }
    resolve_body("source.wires = 512\nsource.gates = 10\nrun.stop_after_stage = 3").unwrap();
    resolve_body("source.wires = 512\nsource.gates = 10\nrun.skip_stages = [6]").unwrap();
    assert!(resolve_body("source.wires = 512\nsource.gates = 10\nrun.skip_stages = [5]").is_err());
    // skip lists: deduplicated, sorted; 6 may be skipped alone
    let c = resolve_body("run.skip_stages = [5, 4, 5]").unwrap();
    assert_eq!(c.skip_stages, vec![4, 5]);
    assert_eq!(
        (c.stages_text(), c.skip_stages_text()),
        ("1-3,6".to_owned(), "4,5".to_owned())
    );
    assert!(
        c.adaptive_on(),
        "stage 6 runs in its adaptive mode by default"
    );
    let four_stage = resolve_body(
        "run.skip_stages = [4, 5]\ndelivery.mode = \"windowed\"\ndelivery.anf_only = true",
    )
    .unwrap();
    assert_eq!(four_stage.stages_text(), "1-3,6");
    assert!(!four_stage.unpacking_on() && four_stage.packing_on() && !four_stage.adaptive_on());
    let windowed = resolve_body("delivery.mode = \"windowed\"").unwrap();
    assert!(windowed.unpacking_on() && windowed.packing_on() && !windowed.adaptive_on());
    assert_eq!(
        resolve_body("run.skip_stages = [2, 3]")
            .unwrap()
            .stages_text(),
        "1,4-6"
    );
    let no_six = resolve_body("run.skip_stages = [6]").unwrap();
    assert!(!no_six.unpacking_on() && !no_six.adaptive_on() && !no_six.packing_on());
    assert_eq!(
        resolve_body("run.stop_after_stage = 3")
            .unwrap()
            .stages_text(),
        "1-3"
    );
    assert_eq!(
        resolve_body("run.skip_stages = []")
            .unwrap()
            .skip_stages_text(),
        "none"
    );
}

#[test]
fn explicit_seed_file_requires_calibration_and_never_reaches_argv() {
    let seed_path = temp_path("seed");
    private_file(&seed_path, "gen_seed=123456789\nmix_seed=987654321\n");
    let without_gate = parse(&format!(
        "run.directory = \"runs/test\"\ncalibration.seed_file = {:?}\ndatabase.allow_no_database_for_tests = true",
        seed_path
    ));
    assert!(resolve_for_test(&without_gate).is_err());
    let with_gate = parse(&format!(
        "run.directory = \"runs/test\"\ncalibration.enabled = true\ncalibration.seed_file = {:?}\ndatabase.allow_no_database_for_tests = true",
        seed_path
    ));
    let resolved = resolve_for_test(&with_gate).unwrap();
    assert!(matches!(
        resolved.seeds,
        SeedPlan::Explicit {
            gen_seed: 123456789,
            mix_seed: 987654321,
            split_seed: None,
            cross_seed: None,
            unpack_seed: None,
        }
    ));
    let mut invocation = vec![OsString::from("bash"), OsString::from("script")];
    invocation.extend(script_args(&resolved));
    let rendered = render_command(&invocation);
    assert!(!rendered.contains("123456789") && !rendered.contains("987654321"));
    let manifest = recipe_manifest(&resolved, &[], 1);
    assert!(!manifest.contains("123456789") && !manifest.contains("987654321"));
    assert!(manifest.contains("\nseed_source=explicit\n"));
    fs::remove_file(seed_path).unwrap();
}

#[test]
fn seed_file_format_1_and_malformed_files_are_rejected() {
    let path = temp_path("seed-v1");
    private_file(&path, "123456\n");
    let error = read_seed_file(&path).err().unwrap();
    assert!(error.contains("format 1 holds one seed"), "{error}");
    assert!(error.contains("gen_seed= and mix_seed="), "{error}");
    for text in [
        "gen_seed=1\n",
        "gen_seed=1\nmix_seed=x\n",
        "gen_seed=1\nmix_seed=9223372036854775793\n",
        "gen_seed=1\nmix_seed=2\nextra=3\n",
        "gen_seed=1\ngen_seed=1\nmix_seed=2\n",
        "gen_seed=-1\nmix_seed=2\n",
    ] {
        private_file(&path, text);
        assert!(read_seed_file(&path).is_err(), "accepted {text:?}");
    }
    private_file(&path, "gen_seed=9223372036854775792\nmix_seed=0\n");
    let s = read_seed_file(&path).unwrap();
    assert_eq!((s.gen_seed, s.mix_seed), (MAX_SEED, 0));
    assert!(s.split_seed.is_none() && s.cross_seed.is_none() && s.unpack_seed.is_none());
    // the stage-4/5/7 seeds are optional and may come in any order
    private_file(
        &path,
        "unpack_seed=5\ngen_seed=1\nsplit_seed=3\nmix_seed=2\n",
    );
    let s = read_seed_file(&path).unwrap();
    assert_eq!(
        (
            s.gen_seed,
            s.mix_seed,
            s.split_seed,
            s.cross_seed,
            s.unpack_seed
        ),
        (1, 2, Some(3), None, Some(5))
    );
    private_file(&path, "gen_seed=1\nmix_seed=2\ncross_seed=x\n");
    assert!(read_seed_file(&path).is_err());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        assert!(read_seed_file(&path).err().unwrap().contains("chmod 600"));
    }
    fs::remove_file(path).unwrap();
}

#[test]
fn all_seed_flags_redacted() {
    let args: Vec<OsString> = [
        "bash",
        "--leakage-repair-seed",
        "111",
        "--seed",
        "222",
        "--gen-seed",
        "333",
        "--seed-file",
        "/private/seeds",
        "--source-wires",
        "64",
    ]
    .iter()
    .map(OsString::from)
    .collect();
    let rendered = render_command(&args);
    for secret in ["111", "222", "333"] {
        assert!(!rendered.contains(secret), "{rendered}");
    }
    assert!(rendered.contains("/private/seeds"));
    assert!(rendered.contains("--source-wires 64"));
}

#[test]
fn checked_in_toml_recipe_is_the_l8_default() {
    let parsed = parse_config(include_str!("../../configs/tdp.toml")).unwrap();
    let config = resolve_for_test(&parsed).unwrap();
    assert_eq!(
        config.preprocessing_mode,
        PreprocessingMode::EmbeddedMasking
    );
    assert_eq!(config.preset, Preset::L8);
    assert_eq!(config.schedule, Schedule::Validated);
    assert_eq!((config.n, config.mcd), (128, 3000));
    assert_eq!(config.recipe_version, 13);
    assert_eq!(config.stop_after_stage(), 6);
    assert_eq!(config.stages_text(), "1-6");
    assert!(config.delivery.is_default() && config.adaptive.is_default());
    assert!(config.adaptive_on() && !config.unpacking_on());
    assert_eq!(config.source_construction, SourceConstruction::Random);
    let bare = resolve_for_test(&parse("")).unwrap();
    assert_eq!(
        recipe_manifest(&config, &[], 1),
        recipe_manifest(&bare, &[], 1)
    );
}

#[test]
fn v1_config_rejected_with_migration() {
    let error = parse_config("config_version = 1\n").unwrap_err();
    assert!(
        error.contains("config_version 1 is not supported by recipe v12"),
        "{error}"
    );
    assert!(error.contains("preprocessing.preset"), "{error}");
    // the four-stage template: a pointer to a later schema
    let error = parse_config("config_version = 2\n").unwrap_err();
    assert!(error.contains("config_version = 4"), "{error}");
    assert!(error.contains("skip_stages = [4, 5]"), "{error}");
    // the seven-stage template (separate packing and unpacking stages): the
    // delivery-stage pointer, and its [unpacking] table named explicitly
    let error = parse_config("config_version = 3\n").unwrap_err();
    assert!(error.contains("config_version = 4"), "{error}");
    assert!(error.contains("[delivery]"), "{error}");
    assert!(error.contains("delivery.anf_only = true"), "{error}");
    // the six-stage template whose only stage 6 was the windowed delivery:
    // the pointer to delivery.mode and the new tables
    let error = parse_config("config_version = 4\n").unwrap_err();
    assert!(error.contains("config_version = 5"), "{error}");
    assert!(error.contains("delivery.mode = \"windowed\""), "{error}");
    assert!(error.contains("[adaptive]"), "{error}");
    assert!(error.contains("three-mask"), "{error}");
    let error = parse_config("config_version = 5\n[unpacking]\ntime_limit = 30\n").unwrap_err();
    assert!(error.contains("[unpacking] moved to [delivery]"), "{error}");
    let error = parse_config("config_version = 5\n[unpacking]\n").unwrap_err();
    assert!(error.contains("[unpacking] moved to [delivery]"), "{error}");
    assert!(parse_config("config_version = 6\n").is_err());
}

#[test]
fn old_template_with_preset_l8_rejected() {
    // The v1 template's explicit knobs, with config_version bumped to 4.
    let old = "config_version = 5\n[preprocessing]\nmode = \"embedded-masking\"\nmask_pair_wires = 2\nmax_open_masks = 3\nmin_open_masks = 2\nbalanced_masks = true\nshuffling_segments = 0\nshuffling_return_home = true\n[db_mixing]\ntarget_size_factor = 2.0\nhold_work_units = 27.0\n";
    let error = resolve_for_test(&parse_config(old).unwrap()).err().unwrap();
    assert!(error.contains("is set by preset L8"), "{error}");
}

#[test]
fn preset_field_conflict_rejected() {
    for preset in ["L8", "A"] {
        for field in [
            "mask_pair_wires = 2",
            "shuffling_segments = 8",
            "shuffling_return_home = false",
            "premask = true",
            "chain = 2",
            "chain_len = 8",
        ] {
            let error = resolve_body(&format!(
                "preprocessing.preset = \"{preset}\"\npreprocessing.{field}"
            ))
            .err()
            .unwrap();
            assert!(
                error.contains(&format!("is set by preset {preset}")),
                "{error}"
            );
        }
    }
    // custom needs every knob
    let error = resolve_body("preprocessing.preset = \"custom\"\npreprocessing.chain = 2")
        .err()
        .unwrap();
    assert!(error.contains("requires every stage-2 field"), "{error}");
    let custom = resolve_body(ALL_CUSTOM).unwrap();
    assert_eq!(custom.preset, Preset::Custom);
    assert_eq!(custom.knobs, Stage2Knobs::L8);
    // return_home = false is accepted under custom (persistent migration)
    assert!(!custom.knobs.shuffling_return_home);
    let args = strings(&script_args(&custom));
    for pair in [
        ["--preprocessing-preset", "custom"],
        ["--preprocessing-shuffling-return-home", "0"],
        ["--preprocessing-chain-len", "8"],
    ] {
        assert!(args.windows(2).any(|w| w == pair), "{pair:?}");
    }
}

#[test]
fn schedule_field_conflict_rejected() {
    for field in [
        "growth_work_units = 3",
        "hold_work_units = 27",
        "target_size_factor = 2",
    ] {
        let error = resolve_body(&format!("db_mixing.{field}")).err().unwrap();
        assert!(error.contains("is set by schedule validated"), "{error}");
    }
    let error = resolve_body("db_mixing.schedule = \"custom\"\ndb_mixing.hold_work_units = 27")
        .err()
        .unwrap();
    assert!(error.contains("requires"), "{error}");
    let c = resolve_body("db_mixing.schedule = \"custom\"\ndb_mixing.growth_work_units = 3\ndb_mixing.hold_work_units = 27\ndb_mixing.target_size_factor = 2").unwrap();
    assert!(matches!(c.schedule, Schedule::Custom { .. }));
    let m = recipe_manifest(&c, &[], 1);
    assert!(
        m.contains("\ndb_mixing_schedule=custom\nstage3_growth_work_units=3\nstage3_hold_work_units=27\nstage3_target_size_factor=2\n"),
        "{m}"
    );
}

#[test]
fn preset_rejected_for_nonlinear291() {
    let base = "source.wires = 4\nsource.gates = 1\npreprocessing.mode = \"nonlinear291\"\n";
    let config = resolve_body(base).unwrap();
    assert_eq!(config.preset, Preset::None);
    assert!(!recipe_manifest(&config, &[], 1).contains("preprocessing_mask_pair_wires"));
    assert!(
        !strings(&script_args(&config))
            .iter()
            .any(|a| a == "--preprocessing-preset")
    );
    let error = resolve_body(&format!("{base}preprocessing.preset = \"L8\""))
        .err()
        .unwrap();
    assert!(
        error.contains("applies only to embedded-masking"),
        "{error}"
    );
    for setting in [
        "preprocessing.mask_pair_wires = 2",
        "preprocessing.premask = true",
        "preprocessing.chain = 2",
    ] {
        let error = resolve_body(&format!("{base}{setting}")).err().unwrap();
        assert!(
            error.contains("mask controls apply only"),
            "{setting}: {error}"
        );
    }
}

#[test]
fn preset_a_drops_chain() {
    let a = resolve_body("preprocessing.preset = \"A\"").unwrap();
    assert_eq!(a.knobs, Stage2Knobs::A);
    assert_eq!((a.knobs.chain, a.knobs.chain_len), (0, 0));
    assert_eq!(
        Stage2Knobs {
            chain: 2,
            chain_len: 8,
            ..Stage2Knobs::A
        },
        Stage2Knobs::L8
    );
    let m = recipe_manifest(&a, &[], 1);
    assert!(m.contains("\npreprocessing_preset=A\n"));
    assert!(m.contains("\npreprocessing_chain=0\npreprocessing_chain_len=0\n"));
    assert!(
        strings(&script_args(&a))
            .windows(2)
            .any(|w| w == ["--preprocessing-preset", "A"])
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
        "runs/../.git/tdp",
    ] {
        let raw = parse(&format!(
            "run.directory = {value:?}\ndatabase.allow_no_database_for_tests = true"
        ));
        assert!(resolve_config(&raw, root, |_| None, "test").is_err());
    }
}

#[test]
fn the_exact_default_manifest_text() {
    let config = resolve_for_test(&parse("source.wires = 64")).unwrap();
    let expected = concat!(
        "tdp_command_recipe=13\npinned_recipe=none\nn=64\n",
        "source_gates=1000\nsource_gates_rule=validated\nsource=generated\nsource_path=unset\n",
        "source_construction=random\n",
        "sandwich_variant=classic\npreprocessing_mode=embedded-masking\npreprocessing_preset=L8\n",
        "preprocessing_mask_pair_wires=2\npreprocessing_max_open_masks=3\n",
        "preprocessing_min_open_masks=2\npreprocessing_balanced_masks=1\n",
        "preprocessing_shuffling_segments=8\npreprocessing_shuffling_return_home=0\n",
        "preprocessing_premask=1\npreprocessing_seed_avoid_c1=1\n",
        "preprocessing_chain=2\npreprocessing_chain_len=8\npreprocessing_quadratic_fire=1\n",
        "gadget_seed_rule=gen\ndb_mixing_schedule=validated\nstage3_profile=2,2,2,2,2\n",
        "stage3_moves_rule=int(G*2*2*1.3)\nstage3_verify_every=default(10000)\n",
        "stage3_report_every=default(50000)\nskip_stages=none\nstage4_split_args=archive(ea2e516f)\n",
        "crossing_target_size_factor=2\ncrossing_width_penalty_base=3\ncrossing_width_penalty_threshold=1\n",
        "crossing_size_tolerance_divisor=25\ncrossing_move_budget_factor=3\ncrossing_move_attempts=factor*target\n",
        "pieces=1\nfrozen_db_dir=unset\nfrozen_db_fingerprint=unset\n",
        "frozen_filter=auto(unset)\n",
        "frozen_regular_value_convention=native\nallow_empty_store=false\nenv_policy=allowlist-v1\n",
        "cache_mb=lookup:2048,canon:256,xpoly:1024\ncanon_caps=unbounded\npacking_params=default\n",
        "packing_window_gates=512\npacking_interval_cap=512\npacking_monomial_budget=24576\n",
        "packing_span_cap=3000\npacking_split_budget=65536\npacking_fix_iterations=4\n",
        "packing_fallback_iterations=4\npacking_anf_cap_semantics=cpython39\n",
        "packing_allow_floor_miss=false\npacking_allow_over_budget=false\n",
        "delivery_mode=adaptive\ndelivery_params=default\ndelivery_anf_only=false\ndelivery_time_limit=30\n",
        "delivery_restarts=16\ndelivery_beam=6\ndelivery_max_rounds=16\n",
        "adaptive_params=default\nadaptive_max_rounds=0\nadaptive_stop_fraction=0.02\nadaptive_seeds_per_block=2\n",
        "adaptive_time_limit=60\nadaptive_restarts=16\nadaptive_beam=6\nadaptive_term_cap=131072\n",
        "adaptive_max_block_gates=65536\nadaptive_segments=16\nadaptive_avoid_radius=64\n",
        "adaptive_refinement_levels=2\nadaptive_min_refine_gates=128\nadaptive_census_batches=96\n",
        "adaptive_accept=smaller-not-leakier\nadaptive_canonical_order=true\nseed_record=SEEDS\n",
        "seed_source=random\nseed_bits=63\nseeds_format=3\ncalibration_only=false\nqc_enabled=false\nqc_seed=20803\n",
        "qc_reference=unset\nbinary_tdp_packing_xxh3=00000000000000000000000000000003\n",
        "stage2_span_report=skipped\nscript_tdp_gen_xxh3=00000000000000000000000000000001\n",
    );
    assert_eq!(recipe_manifest(&config, &[("tdp_packing", 3)], 1), expected);
    // no hard-coded "hold" text: every value is resolved (D18); "threshold"
    // is crossing_width_penalty_threshold, a resolved value
    assert!(!expected.replace("threshold", "").contains("hold"));
}

/// Both drivers pass a packing flag only when it differs from
/// PackingConfig::default(), and the manifest then records
/// packing_params=default: so the defaults must be what `tdp_packing run`
/// runs with no flag (PackParams::PRODUCTION, RepairParams::default()) and
/// what scripts/tdp_gen.sh assumes. The binary side is pinned to the literal
/// val_chain.sh values in tests/integration/tdp_packing.rs (run_end_to_end_n8).
#[test]
fn packing_defaults_are_the_stage_four_binary_defaults() {
    use crate::stages::packing::types::{CapSemantics, PackParams, RepairMode, RepairParams};
    let d = PackingConfig::default();
    let pack = PackParams::PRODUCTION;
    let repair = RepairParams::default();
    assert_eq!(d.window_gates, pack.g as u64);
    assert_eq!(d.interval_cap, pack.w_cap as u64);
    assert_eq!(d.monomial_budget, pack.b_mono as u64);
    assert_eq!(d.span_cap, pack.span_cap as u64);
    assert_eq!(d.split_budget, repair.split_budget as u64);
    assert_eq!(d.fix_iterations, repair.fix_iterations as u64);
    assert_eq!(d.fallback_iterations, repair.fallback_iterations as u64);
    assert_eq!(d.anf_cap_semantics, pack.cap_semantics.name());
    assert_eq!(d.anf_cap_semantics, repair.cap_semantics.name());
    assert_eq!(CapSemantics::default(), pack.cap_semantics);
    assert_eq!(repair.mode, RepairMode::Oracle);
    assert!(!d.allow_floor_miss && !d.allow_over_budget);
    // val_chain.sh's stage 4: window_pack3.py A B 512 512 24576 3000, repair.sh 512
    assert_eq!(
        (
            d.interval_cap,
            d.window_gates,
            d.monomial_budget,
            d.span_cap
        ),
        (512, 512, 24576, 3000)
    );
    assert_eq!(
        (d.split_budget, d.fix_iterations, d.fallback_iterations),
        (65536, 4, 4)
    );
    // scripts/tdp_gen.sh: its initial values and its pf() omission defaults
    let script =
        fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/tdp_gen.sh"))
            .unwrap();
    let initial = format!(
        "PACK_WINDOW_GATES={}; PACK_INTERVAL_CAP={}; PACK_MONOMIAL_BUDGET={}; PACK_SPAN_CAP={}\n\
         PACK_SPLIT_BUDGET={}; PACK_FIX_ITERATIONS={}; PACK_FALLBACK_ITERATIONS={}; PACK_CAP_SEM={}\n",
        d.window_gates,
        d.interval_cap,
        d.monomial_budget,
        d.span_cap,
        d.split_budget,
        d.fix_iterations,
        d.fallback_iterations,
        d.anf_cap_semantics
    );
    assert!(
        script.contains(&initial),
        "tdp_gen.sh initial PACK_* values"
    );
    for (flag, var, value) in [
        (
            "--window-gates",
            "PACK_WINDOW_GATES",
            d.window_gates.to_string(),
        ),
        (
            "--interval-cap",
            "PACK_INTERVAL_CAP",
            d.interval_cap.to_string(),
        ),
        (
            "--monomial-budget",
            "PACK_MONOMIAL_BUDGET",
            d.monomial_budget.to_string(),
        ),
        ("--span-cap", "PACK_SPAN_CAP", d.span_cap.to_string()),
        (
            "--split-budget",
            "PACK_SPLIT_BUDGET",
            d.split_budget.to_string(),
        ),
        (
            "--fix-iterations",
            "PACK_FIX_ITERATIONS",
            d.fix_iterations.to_string(),
        ),
        (
            "--fallback-iterations",
            "PACK_FALLBACK_ITERATIONS",
            d.fallback_iterations.to_string(),
        ),
        (
            "--anf-cap-semantics",
            "PACK_CAP_SEM",
            d.anf_cap_semantics.clone(),
        ),
    ] {
        let line = format!("pf {flag} \"${var}\" {value}\n");
        assert!(script.contains(&line), "tdp_gen.sh lacks {line:?}");
    }
    // every Rust-side flag the driver can omit is one of those
    for (flag, value, default) in d.flags() {
        assert_eq!(value, default, "{flag}");
    }
}

/// The delivery stage's synthesizer defaults are `tdp_unpacking`'s own
/// (`UnpackParams::default().synth`), and scripts/tdp_gen.sh starts from the
/// same values and omits a flag exactly at them (`dfl`); `--refine
/// --max-rounds` is always passed.
#[test]
fn delivery_defaults_are_the_unpacking_binary_defaults() {
    use crate::stages::unpacking::UnpackParams;
    let d = DeliveryConfig::default();
    let synth = UnpackParams::default().synth;
    assert_eq!(d.time_limit, synth.time_limit.to_string());
    assert_eq!(d.restarts, synth.restarts as u64);
    assert_eq!(d.beam, synth.beam as u64);
    assert!(!d.anf_only);
    assert_eq!(d.max_rounds, 16, "docs/DELIVERY_STAGE.md §2: max_rounds 16");
    let script =
        fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/tdp_gen.sh"))
            .unwrap();
    let initial = format!(
        "DELIV_TIME_LIMIT={}; DELIV_RESTARTS={}; DELIV_BEAM={}; DELIV_MAX_ROUNDS={}\n",
        d.time_limit, d.restarts, d.beam, d.max_rounds
    );
    assert!(
        script.contains(&initial),
        "tdp_gen.sh initial DELIV_* values"
    );
    assert!(script.contains("DELIVFLAGS=(--refine --max-rounds \"$DELIV_MAX_ROUNDS\")\n"));
    for (flag, var, value) in [
        ("--time-limit", "DELIV_TIME_LIMIT", d.time_limit.clone()),
        ("--restarts", "DELIV_RESTARTS", d.restarts.to_string()),
        ("--beam", "DELIV_BEAM", d.beam.to_string()),
    ] {
        let line = format!("dfl {flag} \"${var}\" {value}\n");
        assert!(script.contains(&line), "tdp_gen.sh lacks {line:?}");
    }
    for (flag, value, default) in d.flags() {
        assert_eq!(value, default, "{flag}");
    }
    // the switch and the four values are separate script arguments
    let custom = resolve_body(
        "delivery.mode = \"windowed\"\ndelivery.anf_only = true\ndelivery.time_limit = 2.5\ndelivery.restarts = 2\ndelivery.beam = 3\ndelivery.max_rounds = 4",
    )
    .unwrap();
    assert!(!custom.delivery.is_default());
    let args = strings(&script_args(&custom));
    for pair in [
        ["--delivery-time-limit", "2.5"],
        ["--delivery-restarts", "2"],
        ["--delivery-beam", "3"],
        ["--delivery-max-rounds", "4"],
    ] {
        assert!(args.windows(2).any(|w| w == pair), "{pair:?} in {args:?}");
    }
    assert_eq!(args.last().unwrap(), "--delivery-anf-only");
    let m = recipe_manifest(&custom, &[], 1);
    assert!(
        m.contains("\ndelivery_mode=windowed\ndelivery_params=custom\ndelivery_anf_only=true\ndelivery_time_limit=2.5\ndelivery_restarts=2\ndelivery_beam=3\ndelivery_max_rounds=4\n"),
        "{m}"
    );
}

#[test]
fn recipe_manifest_is_seed_free_and_lifecycle_independent() {
    let base = resolve_for_test(&parse(
        "run.directory = \"runs/manifest-test\"\nrun.stop_after_stage = 2\ndatabase.allow_no_database_for_tests = true",
    ))
    .unwrap();
    let fingerprints = [
        ("gen_sandwich_gadget", 1u128),
        ("circuit_mixer", 2u128),
        ("tdp_packing", 5u128),
    ];
    let expected = recipe_manifest(&base, &fingerprints, 4);
    assert!(expected.contains("binary_tdp_packing_xxh3=00000000000000000000000000000005"));
    let mut lifecycle = base.clone();
    lifecycle.build_release = false;
    lifecycle.build_target_dir = PathBuf::from("/different/target");
    lifecycle.stop_after = Some("3".to_owned());
    lifecycle.force_from = Some("3".to_owned());
    lifecycle.threads = Some("2".to_owned());
    assert_eq!(expected, recipe_manifest(&lifecycle, &fingerprints, 4));
    lifecycle.packing.split_budget = 1024;
    let changed = recipe_manifest(&lifecycle, &fingerprints, 4);
    assert_ne!(expected, changed);
    assert!(changed.contains("\npacking_params=custom\n"));
    lifecycle.packing = base.packing.clone();
    lifecycle.calibration_only = true;
    assert_ne!(expected, recipe_manifest(&lifecycle, &fingerprints, 4));
}

#[test]
fn manifest_comparison_accepts_exact_and_rejects_changed_or_oversized() {
    let path = temp_path("manifest");
    let desired = "tdp_command_recipe=10\nn=128\n";
    fs::write(&path, desired).unwrap();
    compare_recipe_manifest(&path, desired).unwrap();
    let mismatch = compare_recipe_manifest(&path, "tdp_command_recipe=10\nn=129\n").unwrap_err();
    assert_eq!(mismatch.exit_code, 2);
    assert!(mismatch.message.contains("mixed-provenance"));
    fs::write(&path, vec![b'x'; MAX_MANIFEST_BYTES as usize + 1]).unwrap();
    let oversized = compare_recipe_manifest(&path, desired).unwrap_err();
    assert!(oversized.message.contains("no larger"));
    fs::remove_file(path).unwrap();
}

#[test]
#[cfg(unix)]
fn env_is_allowlist_only() {
    let config = resolve_for_test(&parse(
        "run.directory = \"runs/env-test\"\nrun.stop_after_stage = 2\ndatabase.allow_no_database_for_tests = true",
    ))
    .unwrap();
    let mut command = ProcessCommand::new("/usr/bin/env");
    configure_environment(&mut command, &config, Path::new("/tmp/tdp-bin"));
    let out = command.output().unwrap();
    let keys: BTreeSet<String> = String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .filter_map(|l| l.split_once('=').map(|x| x.0.to_owned()))
        .collect();
    let allowed: BTreeSet<String> = [
        "PATH",
        "HOME",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "TDP_BIN_DIR",
        "FROZEN_DB_DIR",
        "TDP_GEN_ALLOW_EMPTY_STORE",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    assert!(keys.is_subset(&allowed), "{keys:?}");
    assert!(keys.contains("TDP_BIN_DIR") && keys.contains("TDP_GEN_ALLOW_EMPTY_STORE"));
}

#[test]
fn inherited_output_affecting_var_rejected() {
    // one or more per refused class: recipe, driver, forbidden, diagnostic
    for name in [
        "EMBEDDED_MASKING_CHAIN",
        "FROZEN_FILTER",
        "SANDWICH_VARIANT",
        "TDP_SOURCE_C",
        "MIXER_DB_LEN_BAND",
        "SLICE_GUARD_LEGACY",
        "MIXER_STOP_AT_PHASE",
        "TWIST_G57_NO_SLIDE",
        "CANON_RULE_L_BRANCH_CAP",
        "EMBEDDED_MASKING_RETIRE",
        "PROD_K",
        "BV5_X",
        "DB_QC_SEED",
        "EMBEDDED_MASKING_DUMP_GATES",
    ] {
        let error = reject_inherited_environment([OsString::from("PATH"), OsString::from(name)])
            .unwrap_err();
        assert!(error.message.contains(name), "{name}: {}", error.message);
        assert!(error.message.contains("not honoured by the managed recipe"));
    }
    // inputs, neutral and tooling variables are not refused
    reject_inherited_environment(
        [
            "FROZEN_DB_DIR",
            "TDP_BIN_DIR",
            "LOOKUP_CACHE_MB",
            "CARGO",
            "HOME",
            "UNRELATED",
        ]
        .map(OsString::from),
    )
    .unwrap();
}

/// Every literal passed to an environment read under src/ must be classified
/// (§11.5), so a new knob forces a decision.
#[test]
fn env_registry_covers_every_env_read() {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }
    let mut files = Vec::new();
    walk(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
        &mut files,
    );
    let callers = [
        "env::var(",
        "env::var_os(",
        "envu(",
        "value_convention(",
        "flag(",
        "num(",
    ];
    let mut unclassified = BTreeSet::new();
    let mut found = 0;
    for f in files {
        let text = fs::read_to_string(&f).unwrap();
        for caller in callers {
            for (i, _) in text.match_indices(caller) {
                let rest = text[i + caller.len()..].trim_start();
                let Some(rest) = rest.strip_prefix('"') else {
                    continue;
                };
                let name: String = rest
                    .chars()
                    .take_while(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || *c == '_')
                    .collect();
                if name.is_empty() {
                    continue;
                }
                found += 1;
                // the allow-list (PATH: the driver resolves bash on it)
                if crate::tdp::env_registry::classify(&name).is_none()
                    && !crate::tdp::env_registry::ALLOW_ENV.contains(&name.as_str())
                {
                    unclassified.insert(format!("{name} ({})", f.display()));
                }
            }
        }
    }
    assert!(found > 50, "the scan found only {found} reads");
    assert!(
        unclassified.is_empty(),
        "unclassified environment reads: {unclassified:?}"
    );
}

/// The `src/stages` contract: construction never reads process configuration.
/// The stage-2 knobs are typed fields since §11.4; the remaining reads are
/// the stage-3 engine's, a recorded follow-up (§13.4) that the managed driver
/// forbids.
#[test]
fn stages_library_reads_no_environment() {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    walk(&root.join("src/stages"), &mut files);
    let allowed = [root.join("src/stages/db_mixing/environment.rs")];
    for f in files {
        let text = fs::read_to_string(&f).unwrap();
        if text.contains("env::var") && !allowed.contains(&f) {
            panic!("{} reads the environment", f.display());
        }
    }
}

#[test]
fn env_registry_tsv_in_sync() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/tdp_env_registry.tsv");
    assert_eq!(
        fs::read_to_string(path).unwrap(),
        crate::tdp::env_registry::registry_tsv(),
        "regenerate scripts/tdp_env_registry.tsv from src/tdp/env_registry.rs"
    );
}

/// Both sub-steps of stage 6: the bash driver's exit tables are the
/// binaries' own (`packing::types::EXIT_CODES`, `unpacking::types::EXIT_CODES`),
/// and the managed driver's meanings agree.
#[test]
fn exit_code_tables_agree() {
    use crate::stages::packing::types::{EXIT_CODES as PACKING, Stage4Error};
    use crate::stages::unpacking::types::{EXIT_CODES as UNPACKING, Stage7Error};
    let script =
        fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/tdp_gen.sh"))
            .unwrap();
    let body_of = |function: &str| {
        let start = script.find(function).unwrap();
        &script[start..start + script[start..].find("\n}").unwrap()]
    };
    let body = body_of("packing_exit_meaning() {");
    for (code, meaning) in PACKING.iter().filter(|(c, _)| *c != 0) {
        assert!(
            body.contains(&format!("{code}) echo \"{meaning}\"")),
            "tdp_gen.sh packing_exit_meaning lacks exit {code} = {meaning}"
        );
        assert_eq!(Stage4Error::meaning(*code), *meaning);
    }
    let body = body_of("unpacking_exit_meaning() {");
    for (code, meaning) in UNPACKING.iter().filter(|(c, _)| *c != 0) {
        assert!(
            body.contains(&format!("{code}) echo \"{meaning}\"")),
            "tdp_gen.sh unpacking_exit_meaning lacks exit {code} = {meaning}"
        );
        assert_eq!(Stage7Error::meaning(*code), *meaning);
        assert_eq!(unpacking_exit_meaning(*code), *meaning);
    }
    let body = body_of("adaptive_exit_meaning() {");
    for (code, meaning) in ADAPTIVE_EXIT_CODES.iter().filter(|(c, _)| *c != 0) {
        assert!(
            body.contains(&format!("{code}) echo \"{meaning}\"")),
            "tdp_gen.sh adaptive_exit_meaning lacks exit {code} = {meaning}"
        );
        assert_eq!(adaptive_exit_meaning(*code), *meaning);
    }
}

#[test]
fn run_target_and_store_trees_must_not_overlap() {
    let base = temp_path("separation");
    fs::create_dir_all(&base).unwrap();
    let run = base.join("run");
    let target = base.join("target");
    let store = base.join("store");
    assert!(validate_path_separation(&run, &run.join("target"), None).is_err());
    assert!(validate_path_separation(&run, &target, Some(&target.join("store"))).is_err());
    validate_path_separation(&run, &target, Some(&store)).unwrap();
    fs::remove_dir_all(base).unwrap();
}

#[test]
fn pinned_recipe_and_resolved_filter_follow_the_run() {
    let manifest_pinned = |c: &ResolvedConfig| {
        recipe_manifest(c, &[], 1)
            .lines()
            .find_map(|l| l.strip_prefix("pinned_recipe=").map(str::to_owned))
            .unwrap()
    };
    let filter = |c: &ResolvedConfig| {
        recipe_manifest(c, &[], 1)
            .lines()
            .find_map(|l| l.strip_prefix("frozen_filter=").map(str::to_owned))
            .unwrap()
    };
    let store = temp_path("pinned-store");
    fs::create_dir_all(&store).unwrap();
    fs::write(store.join("shard_00.frz"), "shard").unwrap();
    fs::write(store.join("shard_00.frz.ok"), "0123\n").unwrap();
    // this test store stands in for the validated one
    let reference = store_fingerprint(&store).unwrap();
    let pinned = |c: &ResolvedConfig| pinned_recipe_for_store(c, reference).to_owned();
    let six = resolve_body(&format!(
        "source.wires = 64\ndatabase.regular_dir = {store:?}\n"
    ))
    .unwrap();
    assert_eq!(
        pinned(&six),
        "none",
        "a six-stage run is not the validated four-stage chain"
    );
    // stages 4 and 5 off but stage 6 resynthesizing: not the chain either
    let resynth = resolve_body(&format!(
        "source.wires = 64\ndatabase.regular_dir = {store:?}\nrun.skip_stages = [4, 5]\n"
    ))
    .unwrap();
    assert_eq!(pinned(&resynth), "none");
    let body = format!(
        "source.wires = 64\ndatabase.regular_dir = {store:?}\nrun.skip_stages = [4, 5]\ndelivery.mode = \"windowed\"\ndelivery.anf_only = true\n"
    );
    let l8 = resolve_body(&body).unwrap();
    assert_eq!(pinned(&l8), "val_chain-L8-20261007");
    // the balanced source and the three-mask stage 2 are not the chain
    let balanced = resolve_body(&format!("{body}source.construction = \"balanced\"\n")).unwrap();
    assert_eq!(pinned(&balanced), "none");
    let mut three_mask = l8.clone();
    three_mask.preprocessing_mode = PreprocessingMode::ThreeMask;
    three_mask.preset = Preset::None;
    assert_eq!(pinned(&three_mask), "none");
    // the unpacking parameters are unused under anf_only: they do not unpin
    let unused = resolve_body(&format!("{body}delivery.restarts = 2\n")).unwrap();
    assert_eq!(pinned(&unused), "val_chain-L8-20261007");
    // the pin follows the recipe, not the lifecycle stop point: a stop before
    // stage 4 does not pin a six-stage recipe, and a stop at 3 keeps the
    // four-stage pin (so raising the stop point later resumes the same manifest)
    let six_stop3 = resolve_body(&format!(
        "source.wires = 64\ndatabase.regular_dir = {store:?}\nrun.stop_after_stage = 3\n"
    ))
    .unwrap();
    assert_eq!(pinned(&six_stop3), "none");
    let four_stop3 = resolve_body(&format!("{body}run.stop_after_stage = 3\n")).unwrap();
    assert_eq!(pinned(&four_stop3), "val_chain-L8-20261007");
    assert_eq!(
        recipe_manifest(&four_stop3, &[], 1),
        recipe_manifest(&l8, &[], 1),
        "lifecycle settings must not change the manifest"
    );
    // ... but it is not the validated store: the manifest says so
    assert_eq!(manifest_pinned(&l8), "none");
    assert!(
        recipe_manifest(&l8, &[], 1).contains(&format!("frozen_db_fingerprint={reference:032x}\n"))
    );
    // another store (a changed marker, an extra shard): not the reference
    fs::write(store.join("shard_00.frz.ok"), "4567\n").unwrap();
    assert_eq!(pinned(&l8), "none");
    fs::write(store.join("shard_00.frz.ok"), "0123\n").unwrap();
    fs::write(store.join("shard_01.frz"), "shard").unwrap();
    assert_eq!(pinned(&l8), "none");
    fs::remove_file(store.join("shard_01.frz")).unwrap();
    assert_eq!(pinned(&l8), "val_chain-L8-20261007");
    // no store at all: not the chain
    let mut no_store = l8.clone();
    no_store.frozen_db.path = None;
    assert_eq!(pinned(&no_store), "none");
    assert_eq!(filter(&l8), "auto(1)");
    let a = resolve_body(&format!("{body}preprocessing.preset = \"A\"")).unwrap();
    assert_eq!(pinned(&a), "val_chain-A-20261007");
    let mut changed = l8.clone();
    changed.preset = Preset::Custom;
    assert_eq!(pinned(&changed), "none");
    let mut changed = l8.clone();
    changed.packing.split_budget = 1024;
    assert_eq!(pinned(&changed), "none");
    let mut changed = l8.clone();
    changed.qc_enabled = true;
    assert_eq!(pinned(&changed), "none");
    let custom = resolve_body(&format!(
        "{body}db_mixing.schedule = \"custom\"\ndb_mixing.growth_work_units = 3\ndb_mixing.hold_work_units = 27\ndb_mixing.target_size_factor = 2"
    ))
    .unwrap();
    assert_eq!(pinned(&custom), "none");
    let off = resolve_body(&format!("{body}database.lookup_miss_filter = \"off\"")).unwrap();
    assert_eq!(
        (filter(&off).as_str(), pinned(&off).as_str()),
        ("off(0)", "none")
    );
    // a store with filters.bin: auto resolves to 0, which val_chain.sh never ran
    fs::write(store.join("filters.bin"), "filter").unwrap();
    assert_eq!(filter(&l8), "auto(0)");
    assert_eq!(pinned(&l8), "none");
    // ... and "on" runs the mixer FILTERED (FROZEN_FILTER=1 with filters.bin):
    // also never validated, even against a reference that includes filters.bin
    let on = resolve_body(&format!("{body}database.lookup_miss_filter = \"on\"")).unwrap();
    assert_eq!(filter(&on), "on(1)");
    assert_eq!(pinned(&on), "none");
    let with_filter = store_fingerprint(&store).unwrap();
    assert_eq!(pinned_recipe_for_store(&on, with_filter), "none");
    fs::remove_dir_all(store).unwrap();
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
            "config_version = 5\nrun.directory = {run_dir:?}\nrun.build_directory = {target_dir:?}\ndatabase.regular_dir = {store_dir:?}\nrun.stop_after_stage = 2"
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
fn v9_runs_are_rejected_without_changing_saved_artifacts() {
    let base = temp_path("v9-run");
    let run_dir = base.join("run");
    fs::create_dir_all(&run_dir).unwrap();
    let seed_path = run_dir.join("SEED");
    private_file(&seed_path, "123456\n");
    fs::write(run_dir.join("tdp.mpmct1"), "saved gadget\n").unwrap();
    let config_path = base.join("recipe.toml");
    let target_dir = base.join("target");
    fs::write(
        &config_path,
        format!(
            "config_version = 5\nrun.directory = {run_dir:?}\nrun.build_directory = {target_dir:?}\nrun.stop_after_stage = 2\n"
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
    assert!(error.message.contains("recipe v9"), "{error:?}");
    assert!(!target_dir.exists());
    assert!(!run_dir.join("tdp_command.conf").exists());
    assert!(!run_dir.join("SEEDS").exists());
    assert_eq!(fs::read_to_string(&seed_path).unwrap(), "123456\n");
    fs::remove_dir_all(base).unwrap();
}

#[test]
fn v9_manifest_rejected_with_message() {
    let path = temp_path("v9-manifest");
    fs::write(&path, "tdp_command_recipe=9\nn=4\n").unwrap();
    let error = read_recipe_version(&path).unwrap_err();
    assert!(
        error
            .message
            .contains("recipe v9 run: resume with the checkout that created it"),
        "{error:?}"
    );
    fs::write(&path, "tdp_command_recipe=10\n").unwrap();
    let error = read_recipe_version(&path).unwrap_err();
    assert!(
        error
            .message
            .contains("recipe v10 run (the four-stage driver)"),
        "{error:?}"
    );
    fs::write(&path, "tdp_command_recipe=11\n").unwrap();
    let error = read_recipe_version(&path).unwrap_err();
    assert!(
        error
            .message
            .contains("recipe v11 run (the seven-stage driver)"),
        "{error:?}"
    );
    assert!(error.message.contains("delivery stage"), "{error:?}");
    fs::write(&path, "tdp_command_recipe=12\n").unwrap();
    let error = read_recipe_version(&path).unwrap_err();
    assert!(
        error.message.contains(
            "recipe v12 run (the six-stage driver whose only stage 6 was the windowed delivery)"
        ),
        "{error:?}"
    );
    fs::write(&path, "tdp_command_recipe=13\n").unwrap();
    assert_eq!(read_recipe_version(&path).unwrap(), 13);
    fs::remove_file(path).unwrap();
}

#[test]
fn toml_rejects_unknown_keys_duplicates_and_wrong_types() {
    for input in [
        "config_version = 6",
        "config_version = 5\n[typo]",
        "config_version = 5\n[preprocessing]\nmode = \"embedded-masking\"\nmisspelled = 1",
        "config_version = 5\n[source]\nwires = \"128\"",
        "config_version = 5\n[run]\nbuild_binaries = \"true\"",
        "config_version = 5\n[source]\nwires = 4\nwires = 5",
        "config_version = 5\n[packing]\nwindow_gate = 512",
        "config_version = 5\n[delivery]\nallow_fallback = true",
        "config_version = 5\n[unpacking]\nbeam = 6",
    ] {
        assert!(parse_config(input).is_err(), "accepted {input}");
    }
}

/// Mock stage binaries (tests/shell/lib/tdp_mock.py) in a target layout.
#[cfg(unix)]
fn mock_binaries(target_dir: &Path) -> PathBuf {
    let binary_dir = target_dir
        .join(rustc_host_triple().unwrap())
        .join("release");
    fs::create_dir_all(&binary_dir).unwrap();
    let mock = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/shell/lib/tdp_mock.py");
    for name in PRODUCTION_BINARIES {
        std::os::unix::fs::symlink(&mock, binary_dir.join(name)).unwrap();
    }
    binary_dir
}

#[test]
#[cfg(unix)]
fn managed_v10_dry_run_preserves_manifest_and_rejects_changed_binary() {
    let base = temp_path("managed-v10");
    let run_dir = base.join("run");
    let target_dir = base.join("target");
    let binary_dir = target_dir
        .join(rustc_host_triple().unwrap())
        .join("release");
    fs::create_dir_all(&run_dir).unwrap();
    fs::create_dir_all(&binary_dir).unwrap();
    for name in PRODUCTION_BINARIES {
        fs::write(binary_dir.join(name), format!("original {name}\n")).unwrap();
    }
    let recipe = format!(
        "config_version = 5\nsource.wires = 8\nsource.gates = 1\nrun.directory = {run_dir:?}\nrun.build_directory = {target_dir:?}\nrun.stop_after_stage = 2\n"
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
    private_file(
        &run_dir.join("SEEDS"),
        &seeds_text(seeds(1, 2), &[3u8; 32], "random"),
    );
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
    fs::write(binary_dir.join("tdp_packing"), "changed binary\n").unwrap();
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
            "config_version = 5\nsource.wires = 4\nsource.gates = 1\nrun.directory = {run_dir:?}\nrun.build_directory = {target_dir:?}\npreprocessing.mode = {mode:?}"
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
                .contains("expected embedded-masking, three-mask or nonlinear291"),
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
        "```toml\nconfig_version = 5\n```",
        r#"{"config_version": 2}"#,
    ] {
        assert!(parse_config(input).is_err(), "accepted {input}");
    }
}

#[test]
fn embedded_masking_rejects_insufficient_band() {
    for (n, k, valid) in [(4, 2, true), (4, 7, true), (4, 8, false)] {
        let body = ALL_CUSTOM.replace(
            "preprocessing.mask_pair_wires = 2",
            &format!("preprocessing.mask_pair_wires = {k}"),
        );
        let config =
            resolve_body(&format!("source.wires = {n}\nsource.gates = 4\n{body}")).unwrap();
        assert_eq!(validate_current_recipe(&config).is_ok(), valid, "{n} {k}");
    }
    assert!(
        resolve_body(&ALL_CUSTOM.replace(
            "preprocessing.mask_pair_wires = 2",
            "preprocessing.mask_pair_wires = 1"
        ))
        .is_err()
    );
}

#[test]
fn unsupported_recipe_versions_fail_before_building_and_leave_saved_runs_unchanged() {
    for version in [3, 4, 5, 6, 7, 8, 9, 10, 11] {
        let base = temp_path("unsupported-version");
        let run_dir = base.join("run");
        let target_dir = base.join("target");
        fs::create_dir_all(&run_dir).unwrap();
        let recipe = format!(
            "config_version = 5\nsource.wires = 4\nsource.gates = 1\nrun.directory = {run_dir:?}\nrun.build_directory = {target_dir:?}\nrun.stop_after_stage = 2\n"
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
                error.message.contains("unsupported recipe version")
                    || error.message.contains("recipe v9 run")
                    || error.message.contains("recipe v10 run")
                    || error.message.contains("recipe v11 run"),
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
            "integer_list" => "[4, 5]",
            _ => unreachable!(),
        };
        let canonical_raw =
            parse_config(&format!("config_version = 5\n{canonical} = {value}")).unwrap();
        let (section, key) = alias.split_once('.').unwrap();
        let alias_raw =
            parse_config(&format!("config_version = 5\n[{section}]\n{key} = {value}")).unwrap();
        assert_eq!(canonical_raw.entries.len(), 1);
        assert_eq!(
            canonical_raw.value(flat),
            alias_raw.value(flat),
            "{canonical}"
        );
        if canonical != alias {
            let error = parse_config(&format!(
                "config_version = 5\n{alias} = {value}\n{canonical} = {value}"
            ))
            .unwrap_err();
            assert!(error.contains("duplicate TDP setting"), "{error}");
        }
    }
}

#[test]
fn canonical_recipe_preserves_flag_values() {
    let mut config = resolve_body("run.directory = \"runs/naming-test\"\nparallel.target_piece_gates = 10000\nparallel.threads = 4\nleakage_repair.enabled = true\nleakage_repair.seed = 99").unwrap();
    // Preserve argument values even when they have the same spelling as an option.
    config.run_dir = PathBuf::from("-n");
    let args = script_args(&config);
    assert_eq!(args[0], "--source-wires");
    assert_eq!(args[2], "--run-directory");
    assert_eq!(args[3], "-n");
    assert!(args.contains(&OsString::from("--parallel-target-piece-gates")));
    assert!(args.contains(&OsString::from("--leakage-repair")));
    let rendered = render_command(&args);
    assert!(
        rendered.contains("--leakage-repair-seed <redacted>"),
        "{rendered}"
    );
}

#[test]
fn completed_db_mixing_uses_current_artifact_name() {
    let root = temp_path("db-mixing-artifacts");
    fs::create_dir_all(&root).unwrap();
    let mut config = resolve_for_test(&parse("")).unwrap();
    config.run_dir = root.clone();
    fs::write(root.join("tdp.mpmct1"), "saved gadget").unwrap();
    fs::write(root.join("unrelated.mpmct1"), "saved stage3").unwrap();
    assert!(validate_external_paths(&config).is_err());
    fs::write(root.join("db_mixing.mpmct1"), "saved stage3").unwrap();
    assert!(validate_external_paths(&config).is_err());
    fs::write(root.join("stage3.recipe"), "tdp_stage3_recipe=1\n").unwrap();
    validate_external_paths(&config).unwrap();
    config.force_from = Some("3".into());
    assert!(validate_external_paths(&config).is_err());
    fs::remove_dir_all(root).unwrap();
}

/// The mock's JSON call lines: (name, args) of each call, in order.
fn mock_calls(path: &Path) -> Vec<(String, Vec<String>)> {
    fn quoted(s: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut cur: Option<String> = None;
        let mut chars = s.chars();
        while let Some(c) = chars.next() {
            match (c, cur.as_mut()) {
                ('"', None) => cur = Some(String::new()),
                ('"', Some(_)) => out.push(cur.take().unwrap()),
                ('\\', Some(v)) => {
                    if let Some(n) = chars.next() {
                        v.push(n);
                    }
                }
                (c, Some(v)) => v.push(c),
                _ => {}
            }
        }
        out
    }
    fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .map(|line| {
            let name = quoted(&line[line.find("\"name\": ").unwrap() + 8..])[0].clone();
            let a = line.find("\"args\": [").unwrap() + 9;
            let b = a + line[a..].find(']').unwrap();
            (name, quoted(&line[a..b]))
        })
        .collect()
}

/// A fresh managed run on mock binaries: SEEDS is created (private) before
/// the manifest, no seed value reaches the manifest, the narrative log or the
/// stage-2 marker, gen gets GEN_SEED three times, and a resume reuses
/// everything.
#[test]
#[cfg(unix)]
fn seeds_file_created_before_manifest_and_never_in_narrative_log() {
    use std::os::unix::fs::PermissionsExt;
    for mode in ["embedded-masking", "nonlinear291"] {
        let base = temp_path("fresh-v10");
        let run_dir = base.join("run");
        let target_dir = base.join("target");
        mock_binaries(&target_dir);
        let recipe = format!(
            "config_version = 5\nsource.wires = 4\nsource.gates = 1\npreprocessing.mode = {mode:?}\nrun.directory = {run_dir:?}\nrun.build_directory = {target_dir:?}\nrun.build_binaries = false\nrun.stop_after_stage = 2\n"
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
        assert!(manifest.starts_with("tdp_command_recipe=13\n"));
        assert!(manifest.contains(&format!("preprocessing_mode={mode}\n")));
        let seeds_path = run_dir.join("SEEDS");
        let seeds = fs::read_to_string(&seeds_path).unwrap();
        assert_eq!(
            fs::metadata(&seeds_path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert!(
            fs::metadata(&seeds_path).unwrap().modified().unwrap()
                <= fs::metadata(&manifest_path).unwrap().modified().unwrap()
        );
        let values: Vec<String> = seeds
            .lines()
            .filter_map(|l| {
                l.strip_prefix("gen_seed=")
                    .or_else(|| l.strip_prefix("mix_seed="))
                    .or_else(|| l.strip_prefix("split_seed="))
                    .or_else(|| l.strip_prefix("cross_seed="))
                    .or_else(|| l.strip_prefix("unpack_seed="))
                    .or_else(|| l.strip_prefix("run_key="))
            })
            .map(String::from)
            .collect();
        assert_eq!(values.len(), 6);
        assert!(seeds.starts_with("seeds_format=3\n"));
        for file in ["tdp_command.conf", "tdp_gen.log", "stage12.recipe"] {
            let text = fs::read_to_string(run_dir.join(file)).unwrap();
            for v in &values {
                assert!(!text.contains(v.as_str()), "a seed value reached {file}");
            }
        }
        let marker = fs::read_to_string(run_dir.join("stage12.recipe")).unwrap();
        assert!(marker.contains(&format!("preprocessing_mode={mode}\n")));
        assert!(marker.contains("gadget_seed_rule=gen\n"));
        let calls = mock_calls(&base.join("run.calls"));
        let (name, args) = &calls[0];
        assert_eq!(name, "gen_sandwich_gadget");
        assert_eq!(args[7], args[8]);
        assert_eq!(args[8], args[9]);
        assert!(seeds.contains(&format!("gen_seed={}\n", args[7])));
        assert_eq!(args.len(), if mode == "nonlinear291" { 11 } else { 10 });
        run_inner(&matches).unwrap();
        assert_eq!(fs::read_to_string(manifest_path).unwrap(), manifest);
        assert_eq!(fs::read_to_string(&seeds_path).unwrap(), seeds);
        let gens = mock_calls(&base.join("run.calls"))
            .iter()
            .filter(|(n, _)| n == "gen_sandwich_gadget")
            .count();
        assert_eq!(gens, 1, "a resume must not regenerate stage 2");
        fs::remove_dir_all(base).unwrap();
    }
}

/// A run directory with SEEDS but no managed manifest (a standalone
/// tdp_gen.sh run) is never adopted.
#[test]
fn standalone_runs_are_not_adopted() {
    let base = temp_path("standalone-run");
    let run_dir = base.join("run");
    fs::create_dir_all(&run_dir).unwrap();
    private_file(
        &run_dir.join("SEEDS"),
        &seeds_text(seeds(1, 2), &[9u8; 32], "random"),
    );
    fs::write(run_dir.join("tdp.mpmct1"), "saved gadget\n").unwrap();
    let target_dir = base.join("target");
    let config_path = base.join("recipe.toml");
    fs::write(
        &config_path,
        format!(
            "config_version = 5\nsource.wires = 4\nsource.gates = 1\nrun.directory = {run_dir:?}\nrun.build_directory = {target_dir:?}\nrun.stop_after_stage = 2\n"
        ),
    )
    .unwrap();
    for dry in [true, false] {
        let mut args = vec![
            OsString::from("tdp_gen"),
            OsString::from("--config"),
            config_path.clone().into_os_string(),
        ];
        if dry {
            args.push(OsString::from("--dry-run"));
        }
        let error = run_inner(&command().try_get_matches_from(args).unwrap()).unwrap_err();
        assert!(error.message.contains("no managed manifest"), "{error:?}");
        assert!(!target_dir.exists());
        assert!(!run_dir.join("tdp_command.conf").exists());
    }
    fs::remove_dir_all(base).unwrap();
}

#[test]
fn bash_is_resolved_by_version_not_by_name() {
    assert_eq!(parse_bash_version("5.2\n"), Some((5, 2)));
    assert_eq!(parse_bash_version("3.2"), Some((3, 2)));
    assert_eq!(parse_bash_version(".\n"), None);
    assert_eq!(parse_bash_version(""), None);
    // macOS's 3.2 and a pre-pipe-here-string 5.0 are too old; 5.1 and later pass
    // (separate statements: clippy's redundant_comparisons folds a `&&` chain
    // against the constant)
    assert!((3, 2) < MIN_BASH);
    assert!((5, 0) < MIN_BASH);
    assert!((5, 1) >= MIN_BASH);
    assert!((6, 0) >= MIN_BASH);
    // PATH order first (relative entries skipped), then the fixed locations once
    let path = std::env::join_paths(["/x/bin", "rel/bin", "/bin"]).unwrap();
    let c = bash_candidates(Some(&path));
    assert_eq!(c[0], PathBuf::from("/x/bin/bash"));
    assert_eq!(c[1], PathBuf::from("/bin/bash"));
    assert!(!c.iter().any(|p| p.starts_with("rel")));
    assert_eq!(
        c.iter().filter(|p| **p == Path::new("/bin/bash")).count(),
        1
    );
    assert!(c.contains(&PathBuf::from("/opt/homebrew/bin/bash")));
    // the machine running the tests has a current bash (CI: ubuntu's 5.x)
    let bash = resolve_bash().unwrap();
    assert!(bash.version >= MIN_BASH);
}

/// `delivery.failed` names the exit code AND the sub-step; anything else
/// attaches no meaning (a 2 from gen, the mixer or bash is not stage 6's).
#[test]
fn stage_meaning_needs_the_failure_marker() {
    let base = temp_path("stage-marker");
    fs::create_dir_all(&base).unwrap();
    let marker = base.join("delivery.failed");
    assert_eq!(stage_failure(&marker), None);
    fs::write(&marker, "exit=5\nstep=packing\n").unwrap();
    assert_eq!(stage_failure(&marker), Some((5, DeliveryStep::Packing)));
    fs::write(&marker, "exit=4\nstep=unpacking\n").unwrap();
    assert_eq!(stage_failure(&marker), Some((4, DeliveryStep::Unpacking)));
    fs::write(&marker, "exit=4\nstep=adaptive\n").unwrap();
    assert_eq!(stage_failure(&marker), Some((4, DeliveryStep::Adaptive)));
    assert_eq!(
        adaptive_exit_meaning(4),
        "verification or synthesis failure"
    );
    fs::write(&marker, "exit=5\n").unwrap();
    assert_eq!(stage_failure(&marker), None, "no sub-step");
    fs::write(&marker, "exit=5\nstep=mixing\n").unwrap();
    assert_eq!(stage_failure(&marker), None);
    fs::write(&marker, "garbage\n").unwrap();
    assert_eq!(stage_failure(&marker), None);
    assert_eq!(
        unpacking_exit_meaning(4),
        "verification or synthesis failure"
    );
    assert_eq!(unpacking_exit_meaning(5), "unknown exit code");
    fs::remove_dir_all(base).unwrap();
}

/// Recipe v13: the three-mask stage 2 (`[three_mask]` only under its mode,
/// defaults = the campaign configuration, non-defaults as separate script
/// flags, every effective value in the manifest).
#[test]
fn three_mask_table_follows_its_mode() {
    let tm = resolve_body("preprocessing.mode = \"three-mask\"").unwrap();
    assert_eq!(tm.preprocessing_mode, PreprocessingMode::ThreeMask);
    assert_eq!(tm.preset, Preset::None);
    assert!(tm.three_mask.is_default());
    let args = strings(&script_args(&tm));
    assert!(
        args.windows(2)
            .any(|w| w == ["--preprocessing-mode", "three-mask"])
    );
    assert!(
        !args.iter().any(|a| a.starts_with("--three-mask-")),
        "defaults emit no knob flag: {args:?}"
    );
    assert!(!args.iter().any(|a| a == "--preprocessing-preset"));
    let m = recipe_manifest(&tm, &[], 1);
    assert!(m.contains("\npreprocessing_mode=three-mask\npreprocessing_preset=none\n"));
    assert!(
        m.contains(
            "\nthree_mask_params=default\nthree_mask_band_wires=0\nthree_mask_banks=8\nthree_mask_retire_budget=2\nthree_mask_max_retiring=0\nthree_mask_rotation=1\nthree_mask_maturity=2\nthree_mask_visit_sweeps=2\nthree_mask_growth=8\nthree_mask_warmup=4\nthree_mask_segments=6\nthree_mask_mask_count=3\nthree_mask_random_helpers=1\nthree_mask_unique_cover=1\nthree_mask_outer_blocks=L8\n"
        ),
        "{m}"
    );
    assert!(!m.contains("preprocessing_mask_pair_wires"));
    let custom = resolve_body(
        "preprocessing.mode = \"three-mask\"\nthree_mask.band_wires = 256\nthree_mask.random_helpers = false\nthree_mask.segments = 8",
    )
    .unwrap();
    assert!(!custom.three_mask.is_default());
    let args = strings(&script_args(&custom));
    for pair in [
        ["--three-mask-band-wires", "256"],
        ["--three-mask-random-helpers", "0"],
        ["--three-mask-segments", "8"],
    ] {
        assert!(args.windows(2).any(|w| w == pair), "{pair:?} in {args:?}");
    }
    assert!(!args.iter().any(|a| a == "--three-mask-banks"));
    let m = recipe_manifest(&custom, &[], 1);
    assert!(m.contains("\nthree_mask_params=custom\nthree_mask_band_wires=256\n"));
    assert!(m.contains("\nthree_mask_random_helpers=0\n"));
    // the embedded-masking manifest carries no three-mask line
    let em = resolve_for_test(&parse("")).unwrap();
    assert!(!recipe_manifest(&em, &[], 1).contains("three_mask"));
}

/// Recipe v13: the stage-6 mode and the `[adaptive]` table.
#[test]
fn adaptive_table_follows_the_delivery_mode() {
    let d = AdaptiveConfig::default();
    assert_eq!(
        (d.max_rounds, d.stop_fraction.as_str(), d.seeds_per_block),
        (0, "0.02", 2)
    );
    assert_eq!((d.time_limit.as_str(), d.restarts, d.beam), ("60", 16, 6));
    assert_eq!((d.term_cap, d.max_block_gates), (131_072, 65_536));
    assert_eq!((d.segments, d.avoid_radius), (16, 64));
    assert_eq!((d.refinement_levels, d.min_refine_gates), (2, 128));
    assert_eq!(d.census_batches, 96);
    assert_eq!(d.accept, "smaller-not-leakier");
    assert!(d.canonical_order);
    for (flag, value, default) in d.flags() {
        assert_eq!(value, default, "{flag}");
    }
    let custom = resolve_body(
        "adaptive.max_rounds = 3\nadaptive.stop_fraction = 0.01\nadaptive.time_limit = 30\nadaptive.accept = \"smaller\"\nadaptive.canonical_order = false",
    )
    .unwrap();
    assert!(!custom.adaptive.is_default());
    let args = strings(&script_args(&custom));
    for pair in [
        ["--adaptive-max-rounds", "3"],
        ["--adaptive-stop-fraction", "0.01"],
        ["--adaptive-time-limit", "30"],
        ["--adaptive-accept", "smaller"],
    ] {
        assert!(args.windows(2).any(|w| w == pair), "{pair:?} in {args:?}");
    }
    assert_eq!(args.last().unwrap(), "--adaptive-no-canonical-order");
    assert!(!args.iter().any(|a| a == "--delivery-mode"));
    assert!(!args.iter().any(|a| a == "--adaptive-restarts"));
    let m = recipe_manifest(&custom, &[], 1);
    assert!(m.contains("\ndelivery_mode=adaptive\n"));
    assert!(m.contains(
        "\nadaptive_params=custom\nadaptive_max_rounds=3\nadaptive_stop_fraction=0.01\n"
    ));
    assert!(m.contains("\nadaptive_accept=smaller\nadaptive_canonical_order=false\n"));
    // the windowed mode: its flag, the packer and unpacking parameters
    let windowed =
        resolve_body("delivery.mode = \"windowed\"\npacking.split_budget = 4096").unwrap();
    let args = strings(&script_args(&windowed));
    assert!(
        args.windows(2)
            .any(|w| w == ["--delivery-mode", "windowed"])
    );
    assert!(
        args.windows(2)
            .any(|w| w == ["--packing-split-budget", "4096"])
    );
    assert!(recipe_manifest(&windowed, &[], 1).contains("\ndelivery_mode=windowed\n"));
    // tdp_adaptive is a production binary (built and fingerprinted)
    assert!(PRODUCTION_BINARIES.contains(&"tdp_adaptive"));
    // the new environment names are classified
    for name in [
        "TDP_SOURCE_CONSTRUCTION",
        "THREE_MASK_BAND_WIRES",
        "THREE_MASK_UNIQUE_COVER",
    ] {
        assert!(
            crate::tdp::env_registry::classify(name).is_some_and(|e| e.class.inherited_is_fatal()),
            "{name}"
        );
    }
    assert!(
        crate::tdp::env_registry::classify("THREE_MASK_INVENTED")
            .is_some_and(|e| e.class == crate::tdp::env_registry::EnvClass::Forbidden)
    );
}
