use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);
fn scratch() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "gss_circuit_cli_{}_{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&path).unwrap();
    path
}
fn cli(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_local_mixing_bin"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn packed_and_expanded_circuits_evaluate_and_compare_through_the_grouped_cli() {
    let dir = scratch();
    let expanded = dir.join("expanded.mpmct1");
    fs::write(&expanded, "mpmct1 3 1\n0 0 1 1 1\n").unwrap();
    for (name, body) in [
        ("c.esop1", "esop1 3 1\n0 1 1 1 1\n"),
        ("c.anf1", "anf1 3 1\n0 1 1 1\n"),
    ] {
        let path = dir.join(name);
        fs::write(&path, body).unwrap();
        let result = cli(&[
            "circuit",
            "evaluate",
            "-n",
            "3",
            "-s",
            path.to_str().unwrap(),
            "-x",
            "2",
        ]);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(String::from_utf8_lossy(&result.stdout).contains("Output: 110 (0x03)"));
        let result = cli(&[
            "circuit",
            "compare",
            "-n",
            "3",
            "-i",
            "32",
            "-a",
            expanded.to_str().unwrap(),
            "-b",
            path.to_str().unwrap(),
        ]);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn comparison_mismatch_and_zero_samples_return_failure() {
    let dir = scratch();
    let identity = dir.join("identity");
    let flip = dir.join("flip");
    fs::write(&identity, "mpmct1 1 0\n").unwrap();
    fs::write(&flip, "mpmct1 1 1\n0 0 0\n").unwrap();
    for count in ["1", "0"] {
        let result = cli(&[
            "circuit",
            "compare",
            "-n",
            "1",
            "-i",
            count,
            "-a",
            identity.to_str().unwrap(),
            "-b",
            flip.to_str().unwrap(),
        ]);
        assert_eq!(result.status.code(), Some(1));
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn evaluate_requires_exactly_one_input_selection() {
    for extra in [vec![], vec!["--random", "--input", "0"]] {
        let mut args = vec!["circuit", "evaluate", "--wires", "3", "--source", "unused"];
        args.extend(extra);
        let result = cli(&args);
        assert_eq!(result.status.code(), Some(2));
    }
}

#[test]
fn generated_g57_evaluates_and_compares_with_general_format_through_cli() {
    use local_mixing::circuit::{CircuitSeq, XGate};
    let dir = scratch();
    let g57 = dir.join("generated.g57");
    let result = cli(&[
        "circuit",
        "generate",
        "--wires",
        "3",
        "--gates",
        "5",
        "--destination",
        g57.to_str().unwrap(),
    ]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let circuit = CircuitSeq::from_bytes(&fs::read(&g57).unwrap());
    assert_eq!(circuit.gates.len(), 5);
    let expected = circuit.evaluate_64(2);
    let result = cli(&[
        "circuit",
        "evaluate",
        "--wires",
        "3",
        "--source",
        g57.to_str().unwrap(),
        "--input",
        "0x2",
    ]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let bits: String = (0..3)
        .map(|i| if expected & (1 << i) != 0 { '1' } else { '0' })
        .collect();
    assert!(
        String::from_utf8_lossy(&result.stdout)
            .contains(&format!("Output: {bits} (0x{expected:02x})"))
    );
    let general = dir.join("generated.mpmct1");
    let gates: Vec<_> = circuit
        .gates
        .iter()
        .map(|&gate| XGate::from_g57(gate))
        .collect();
    local_mixing::engine::format::write_mpmct(general.to_str().unwrap(), &gates, 3).unwrap();
    let result = cli(&[
        "circuit",
        "compare",
        "--wires",
        "3",
        "--iterations",
        "32",
        "--circuit-a",
        g57.to_str().unwrap(),
        "--circuit-b",
        general.to_str().unwrap(),
    ]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn comparison_reports_a_high_wire_mismatch_through_cli() {
    let dir = scratch();
    let identity = dir.join("identity");
    let flip = dir.join("flip");
    fs::write(&identity, "mpmct1 301 0\n").unwrap();
    fs::write(&flip, "mpmct1 301 1\n300 0 0\n").unwrap();
    let result = cli(&[
        "circuit",
        "compare",
        "--wires",
        "1",
        "--iterations",
        "1",
        "--circuit-a",
        identity.to_str().unwrap(),
        "--circuit-b",
        flip.to_str().unwrap(),
    ]);
    assert_eq!(result.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&result.stderr).contains("circuits differ on sample 1: input"));
    fs::remove_dir_all(dir).unwrap();
}
