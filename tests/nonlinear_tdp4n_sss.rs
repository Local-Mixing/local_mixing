use local_mixing::postmix::format::read_mpmct;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before Unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "local_mixing_nonlinear_tdp4n_{}_{}",
            std::process::id(),
            stamp
        ));
        std::fs::create_dir(&path).expect("create integration-test directory");
        Self(path)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn display_failure(label: &str, output: &Output) -> String {
    format!(
        "{label} failed with {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn metadata_value(path: &Path, key: &str) -> String {
    let metadata = std::fs::read_to_string(path).expect("read slice metadata");
    metadata
        .lines()
        .find_map(|line| line.strip_prefix(&format!("{key}=")))
        .unwrap_or_else(|| panic!("missing {key} in {}", path.display()))
        .to_owned()
}

fn run_nonlinear_tdp4n_sss(
    source: &Path,
    constructed: &Path,
    final_path: &Path,
    seed: Option<&str>,
    rounds: usize,
    moves: usize,
) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_local_mixing_bin"));
    command
        .current_dir(source.parent().expect("source parent"))
        .env("SSS_CNOT_MOVES_PER_ROUND", moves.to_string())
        .args(["sss", "-n", "3", "-m", "1", "-x", "2", "-r"])
        .arg(rounds.to_string())
        .arg("-s")
        .arg(source)
        .arg("-d")
        .arg(final_path)
        .args([
            "--tdp4n",
            "--nonlinear_gadgetize",
            "--slice-zero-random",
            "--cnot",
            "--M_length",
            "96",
            "--rg-frequency",
            "1",
            "--shooting_times",
            "2",
            "--collision_rounds",
            "3",
            "--stable_compressions",
            "1",
            "--gadget_path",
        ])
        .arg(constructed);
    if let Some(seed) = seed {
        command.env("SSS_CNOT_SEED", seed);
    } else {
        command.env_remove("SSS_CNOT_SEED");
    }
    command.output().expect("launch nonlinear TDP4n SSS")
}

#[test]
fn nonlinear_tdp4n_rounds_never_emit_complemented_empty_identities() {
    let temp = TempDir::new();
    let source = temp.0.join("source.txt");
    let constructed = temp.0.join("constructed.mpmct1");
    let final_path = temp.0.join("final.mpmct1");
    std::fs::write(&source, "201;102;").expect("write compact G57 source");

    let sss = run_nonlinear_tdp4n_sss(&source, &constructed, &final_path, Some("7"), 2, 3000);
    assert!(sss.status.success(), "{}", display_failure("SSS", &sss));

    let round_base = final_path.to_string_lossy();
    let artifacts = [
        constructed,
        PathBuf::from(format!("{round_base}round1.txt")),
        PathBuf::from(format!("{round_base}round2.txt")),
        final_path.clone(),
    ];
    for artifact in &artifacts {
        let (gates, wires) =
            read_mpmct(artifact.to_str().expect("UTF-8 artifact path")).expect("read artifact");
        assert_eq!(wires, 19, "unexpected namespace in {}", artifact.display());
        assert!(
            gates.iter().all(|gate| !gate.is_noop()),
            "complemented empty identity escaped into {}",
            artifact.display()
        );
    }

    let slice_metadata = PathBuf::from(format!("{}.slice_zero_random", final_path.display()));
    let fixed_y = metadata_value(&slice_metadata, "y_hex");
    let fixed_z = metadata_value(&slice_metadata, "z_hex");
    for artifact in &artifacts {
        let artifact_metadata = PathBuf::from(format!("{}.slice_zero_random", artifact.display()));
        let (artifact_gates, _) =
            read_mpmct(artifact.to_str().expect("UTF-8 artifact path")).expect("read artifact");
        assert_eq!(
            metadata_value(&artifact_metadata, "gates"),
            artifact_gates.len().to_string(),
            "sidecar gate count does not match {}",
            artifact.display()
        );
        assert_eq!(metadata_value(&artifact_metadata, "construction_seed"), "7");
        assert_eq!(metadata_value(&artifact_metadata, "rg_frequency"), "1");
        assert_eq!(metadata_value(&artifact_metadata, "source_gates"), "2");
        assert_eq!(metadata_value(&artifact_metadata, "y_hex"), fixed_y);
        assert_eq!(metadata_value(&artifact_metadata, "z_hex"), fixed_z);

        let check = Command::new(env!("CARGO_BIN_EXE_slice_check_4n"))
            .arg(artifact)
            .arg("--base-circuit")
            .arg(&source)
            .args(["--n", "3", "--fixed-y"])
            .arg(&fixed_y)
            .arg("--fixed-z")
            .arg(&fixed_z)
            .args(["--samples", "8", "--seed", "7001"])
            .output()
            .expect("launch fixed-slice checker");
        assert!(
            check.status.success(),
            "{}",
            display_failure(
                &format!("slice_check_4n for {}", artifact.display()),
                &check
            )
        );
        let evidence = String::from_utf8_lossy(&check.stdout);
        assert!(
            evidence
                .lines()
                .any(|line| line == "base_semantics_ok=true")
        );
        assert!(evidence.lines().any(|line| line == "slice_ok=true"));
    }
}

#[test]
fn advertised_random_seed_replays_a_full_sss_round() {
    let temp = TempDir::new();
    let source = temp.0.join("source.txt");
    std::fs::write(&source, "201;102;").expect("write compact G57 source");

    let first_constructed = temp.0.join("first_constructed.mpmct1");
    let first_final = temp.0.join("first_final.mpmct1");
    let first = run_nonlinear_tdp4n_sss(&source, &first_constructed, &first_final, None, 1, 500);
    assert!(
        first.status.success(),
        "{}",
        display_failure("unseeded SSS", &first)
    );
    let first_metadata = PathBuf::from(format!("{}.slice_zero_random", first_final.display()));
    let advertised_seed = metadata_value(&first_metadata, "construction_seed");

    let replay_constructed = temp.0.join("replay_constructed.mpmct1");
    let replay_final = temp.0.join("replay_final.mpmct1");
    let replay = run_nonlinear_tdp4n_sss(
        &source,
        &replay_constructed,
        &replay_final,
        Some(&advertised_seed),
        1,
        500,
    );
    assert!(
        replay.status.success(),
        "{}",
        display_failure("seed replay SSS", &replay)
    );

    let first_round = PathBuf::from(format!("{}round1.txt", first_final.display()));
    let replay_round = PathBuf::from(format!("{}round1.txt", replay_final.display()));
    for (first_path, replay_path) in [
        (&first_constructed, &replay_constructed),
        (&first_round, &replay_round),
        (&first_final, &replay_final),
    ] {
        assert_eq!(
            std::fs::read(first_path).expect("read first replay artifact"),
            std::fs::read(replay_path).expect("read replay artifact"),
            "advertised seed did not reproduce {}",
            first_path.display()
        );
    }
}

#[test]
fn fmix_accepts_an_identity_only_mpmct_tape() {
    let temp = TempDir::new();
    let input = temp.0.join("identity.mpmct1");
    let output = temp.0.join("mixed.mpmct1");
    std::fs::write(&input, "mpmct1 2 2\n0 1 0\n1 1 0\n").expect("write identity tape");

    let run = Command::new(env!("CARGO_BIN_EXE_fmix"))
        .arg("--input")
        .arg(&input)
        .arg("--output")
        .arg(&output)
        .args(["--moves", "0", "--report-every", "1", "--seed", "17"])
        .output()
        .expect("launch fmix");
    assert!(
        run.status.success(),
        "{}",
        display_failure("identity-only fmix", &run)
    );
    let (gates, wires) =
        read_mpmct(output.to_str().expect("UTF-8 output path")).expect("read fmix output");
    assert_eq!(wires, 2);
    assert!(gates.is_empty());
}
