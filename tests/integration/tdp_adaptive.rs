//! `tdp_adaptive` end to end (docs/ADAPTIVE_STAGE.md): a small stage-2
//! circuit built by `gen_sandwich_gadget` (n = 8, |C| = |D| = 16, 32
//! wires) is resynthesized with small caps, the published circuit is
//! checked here for equivalence with the input on random full states and on
//! the zero slice (independently of the binary's own checks), the report
//! keys are present, a second run on the same out-dir resumes from the
//! completed rounds, the stand-alone `census` command runs, and the usage
//! errors exit 2.
//!
//! The slow part (the run itself) can be skipped by setting
//! `TDP_ADAPTIVE_SKIP_SLOW` (a test-only switch; the binary reads no
//! environment).
use local_mixing::circuit::formats::{read_g57_file, read_mpmct};
use local_mixing::circuit::xgate::XGate;
use local_mixing::stages::packing::pyrand::PyRandom;
use local_mixing::stages::packing::verify::{eval_gates_lanes16, random_lanes};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

static NEXT: AtomicU64 = AtomicU64::new(0);

fn scratch() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "tdp_adaptive_it_{}_{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

fn p(x: &Path) -> &str {
    x.to_str().unwrap()
}

fn adaptive(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_tdp_adaptive"))
        .env_clear()
        .args(args)
        .output()
        .expect("spawn tdp_adaptive")
}

fn kv(report: &str, key: &str) -> String {
    report
        .lines()
        .find_map(|l| l.strip_prefix(&format!("{key}=")))
        .unwrap_or_else(|| panic!("{key} missing from the report:\n{report}"))
        .to_string()
}

/// `OUT 8 16 16 24 1 160 7 7 7`: n = 8, |C| = |D| = 16, a 32-wire stage-2
/// circuit of ~19k gates, with `OUT.source_c.g57` and
/// `OUT.sandwich.mpmct1`.
fn gen_fixture(dir: &Path) -> (PathBuf, PathBuf, PathBuf) {
    let out = dir.join("g.mpmct1");
    let o = Command::new(env!("CARGO_BIN_EXE_gen_sandwich_gadget"))
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .arg(&out)
        .args(["8", "16", "16", "24", "1", "160", "7", "7", "7"])
        .output()
        .expect("spawn gen_sandwich_gadget");
    assert!(
        o.status.success(),
        "gen: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    let c = PathBuf::from(format!("{}.source_c.g57", out.display()));
    let s = PathBuf::from(format!("{}.sandwich.mpmct1", out.display()));
    assert!(c.is_file() && s.is_file());
    (out, c, s)
}

/// Both circuits agree on every wire of `batches` x 1024 random full states.
fn equivalent(a: &[XGate], b: &[XGate], width: usize, batches: usize) {
    let mut rng = PyRandom::new(777);
    for i in 0..batches {
        let x = random_lanes(&mut rng, width);
        let mut sa = x.clone();
        eval_gates_lanes16(a, &mut sa);
        let mut sb = x;
        eval_gates_lanes16(b, &mut sb);
        assert!(
            sa == sb,
            "circuits differ on random full states (batch {i})"
        );
    }
}

/// x on wires 0..n, every other wire 0: wires n..2n hold C(x).
fn zero_slice(c: &[XGate], gates: &[XGate], n: usize, width: usize) {
    let mut rng = PyRandom::new(778);
    for _ in 0..4 {
        let x = random_lanes(&mut rng, n);
        let mut want = x.clone();
        eval_gates_lanes16(c, &mut want);
        let mut st = x;
        st.resize(width, [0; 16]);
        eval_gates_lanes16(gates, &mut st);
        for w in 0..n {
            assert!(
                st[n + w] == want[w],
                "zero slice: wire {} != C(x) wire {w}",
                n + w
            );
        }
    }
}

const SMALL_CAPS: [&str; 12] = [
    "--term-cap",
    "4096",
    "--max-block-gates",
    "400",
    "--time-limit",
    "5",
    "--census-batches",
    "8",
    "--segments",
    "2",
    "--threads",
    "4",
];

/// Three runs on one out-dir: round 1 alone (`--max-rounds 1`), then
/// `--max-rounds 2` (a PARTIAL resume: round 1 reloaded from its report,
/// round 2 computed from round 1's kept output), then the same command
/// again (a COMPLETED rerun: both rounds reloaded, the published
/// `final.mpmct1` taken as the final circuit, nothing recomputed, the
/// non-final round outputs having been pruned).
#[test]
fn run_resynthesizes_checks_and_resumes() {
    if std::env::var_os("TDP_ADAPTIVE_SKIP_SLOW").is_some() {
        eprintln!("TDP_ADAPTIVE_SKIP_SLOW set: skipping the run");
        return;
    }
    let dir = scratch();
    let (g, c, s) = gen_fixture(&dir);
    let out = dir.join("out");
    let seed_file = dir.join("seed");
    fs::write(&seed_file, "20261026\n").unwrap();
    let base = vec![
        "run",
        "--input",
        p(&g),
        "--source-c",
        p(&c),
        "--n",
        "8",
        "--sandwich",
        p(&s),
        "--out-dir",
        p(&out),
        "--seed-file",
        p(&seed_file),
        "--restarts",
        "4",
    ];
    let mut args1 = base.clone();
    args1.extend_from_slice(&SMALL_CAPS);
    args1.extend_from_slice(&["--max-rounds", "1"]);
    let mut args2 = base.clone();
    args2.extend_from_slice(&SMALL_CAPS);
    args2.extend_from_slice(&["--max-rounds", "2"]);

    // ---- run A: round 1
    let t0 = Instant::now();
    let o = adaptive(&args1);
    let stdout = String::from_utf8_lossy(&o.stdout);
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(
        o.status.success(),
        "exit {:?}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        o.status.code()
    );
    let first = t0.elapsed().as_secs_f64();
    assert!(!stdout.contains("20261026"), "the seed leaked into stdout");

    // ---- outputs of run A
    let report = fs::read_to_string(out.join("adaptive.report")).unwrap();
    assert!(
        !report.contains("20261026"),
        "the seed leaked into the report"
    );
    assert_eq!(kv(&report, "adaptive_report"), "1");
    assert_eq!(kv(&report, "status"), "ok");
    assert_eq!(kv(&report, "n"), "8");
    assert_eq!(kv(&report, "width"), "32");
    assert_eq!(kv(&report, "param.seed_source"), "file");
    assert_eq!(kv(&report, "param.max_rounds"), "1");
    assert_eq!(kv(&report, "rounds"), "1");
    assert_eq!(kv(&report, "round.1.reloaded"), "false");
    assert_eq!(kv(&report, "final_reloaded"), "false");
    assert_eq!(kv(&report, "stop_reason"), "round cap");
    let input_gates: usize = kv(&report, "input_gates").parse().unwrap();
    let final_gates: usize = kv(&report, "final_gates").parse().unwrap();
    assert!(final_gates <= input_gates);
    for key in [
        "stop_reason",
        "plateau",
        "final_round",
        "boundary.internal",
        "boundary.excluded",
        "boundary.deeper",
        "round.1.input_gates",
        "round.1.output_gates",
        "round.1.blocks",
        "round.1.picks_seed",
        "round.1.deeper_affine",
        "stage5_leakage.validated.affine.deeper_births",
        "final_leakage.validated.affine.deeper_births",
        "final_leakage.audit_passed",
        "check.zero_slice",
        "check.payload_contract",
        "seconds",
    ] {
        kv(&report, key);
    }
    assert!(kv(&report, "check.equiv_canonical").starts_with("ok("));
    assert!(kv(&report, "check.zero_slice").starts_with("ok("));
    assert!(kv(&report, "check.payload_contract").starts_with("ok("));
    assert!(out.join("adaptive.log").is_file());
    assert!(out.join("progress.txt").is_file());
    assert!(out.join("boundary.txt").is_file());
    assert!(out.join("round1").join("result.report").is_file());
    assert!(out.join("round1").join("bounds.out").is_file());
    assert!(out.join("final_census").is_dir());
    let r1 = fs::read_to_string(out.join("round1").join("result.report")).unwrap();
    assert_eq!(kv(&r1, "round"), "1");
    assert_eq!(kv(&r1, "input_gates"), input_gates.to_string());
    kv(&r1, "picks.orig");
    kv(&r1, "coverage.synthesized_blocks");
    kv(&r1, "leakage_out.validated.direct.deeper_births");

    // ---- the published circuit of run A, checked here
    let (input, iw) = read_mpmct(p(&g)).unwrap();
    let (fin, fw) = read_mpmct(p(&out.join("final.mpmct1"))).unwrap();
    assert_eq!((iw, fw), (32, 32));
    assert_eq!(fin.len(), final_gates);
    equivalent(&input, &fin, 32, 4);
    let cg = read_g57_file(p(&c)).unwrap();
    zero_slice(&cg, &fin, 8, 32);

    // ---- run B: a partial resume (round 1 reloaded, round 2 computed)
    let t1 = Instant::now();
    let o = adaptive(&args2);
    let stdout2 = String::from_utf8_lossy(&o.stdout);
    assert!(
        o.status.success(),
        "partial resume: exit {:?}\nstdout:\n{stdout2}\nstderr:\n{}",
        o.status.code(),
        String::from_utf8_lossy(&o.stderr)
    );
    let second = t1.elapsed().as_secs_f64();
    let report2 = fs::read_to_string(out.join("adaptive.report")).unwrap();
    assert_eq!(kv(&report2, "param.max_rounds"), "2");
    assert_eq!(kv(&report2, "rounds"), "2");
    assert_eq!(kv(&report2, "round.1.reloaded"), "true");
    assert_eq!(kv(&report2, "round.2.reloaded"), "false");
    assert_eq!(kv(&report2, "final_reloaded"), "false");
    assert_eq!(
        kv(&report2, "round.1.output_gates"),
        kv(&report, "round.1.output_gates")
    );
    assert!(stdout2.contains("round 1: reloaded"), "{stdout2}");
    assert!(kv(&report2, "check.equiv_canonical").starts_with("ok("));
    assert!(kv(&report2, "check.zero_slice").starts_with("ok("));
    assert!(kv(&report2, "check.payload_contract").starts_with("ok("));
    let final2: usize = kv(&report2, "final_gates").parse().unwrap();
    assert!(final2 <= final_gates, "round 2 never grows the result");
    assert!(out.join("round2").join("result.report").is_file());
    let (fin2, _) = read_mpmct(p(&out.join("final.mpmct1"))).unwrap();
    assert_eq!(fin2.len(), final2);
    equivalent(&input, &fin2, 32, 4);
    zero_slice(&cg, &fin2, 8, 32);
    let final_round: usize = kv(&report2, "final_round").parse().unwrap();
    assert!(
        out.join(format!("round{final_round}"))
            .join("output.mpmct1")
            .is_file(),
        "the final round's output is kept"
    );

    // ---- run C: the same command on the completed out-dir reruns from
    // the reports alone (every round reloaded, final.mpmct1 taken as the
    // final circuit, nothing recomputed, the same circuit republished)
    let t2 = Instant::now();
    let o = adaptive(&args2);
    let stdout3 = String::from_utf8_lossy(&o.stdout);
    assert!(
        o.status.success(),
        "completed rerun: exit {:?}\nstdout:\n{stdout3}\nstderr:\n{}",
        o.status.code(),
        String::from_utf8_lossy(&o.stderr)
    );
    let third = t2.elapsed().as_secs_f64();
    let report3 = fs::read_to_string(out.join("adaptive.report")).unwrap();
    assert_eq!(kv(&report3, "rounds"), "2");
    assert_eq!(kv(&report3, "round.1.reloaded"), "true");
    assert_eq!(kv(&report3, "round.2.reloaded"), "true");
    assert_eq!(kv(&report3, "final_reloaded"), "true");
    assert_eq!(kv(&report3, "final_gates"), final2.to_string());
    assert_eq!(kv(&report3, "final_round"), final_round.to_string());
    assert_eq!(kv(&report3, "stop_reason"), kv(&report2, "stop_reason"));
    assert_eq!(kv(&report3, "check.equiv_canonical"), "reloaded");
    assert!(kv(&report3, "check.zero_slice").starts_with("ok("));
    assert!(kv(&report3, "check.payload_contract").starts_with("ok("));
    kv(&report3, "final_leakage.validated.affine.deeper_births");
    assert!(stdout3.contains("final circuit reloaded"), "{stdout3}");
    assert!(
        third < second && third < first,
        "completed rerun took {third:.1}s (partial resume {second:.1}s, round 1 {first:.1}s)"
    );
    let (fin3, _) = read_mpmct(p(&out.join("final.mpmct1"))).unwrap();
    assert_eq!(fin3, fin2, "the rerun publishes the same circuit");
    let progress = fs::read_to_string(out.join("progress.txt")).unwrap();
    assert_eq!(kv(&progress, "rounds"), "2");
    eprintln!(
        "timing: round 1 {first:.1}s, partial resume {second:.1}s, completed rerun {third:.1}s"
    );

    // ---- the stand-alone census
    let cdir = dir.join("census_out");
    let o = adaptive(&[
        "census",
        "--input",
        p(&out.join("final.mpmct1")),
        "--source-c",
        p(&c),
        "--n",
        "8",
        "--out-dir",
        p(&cdir),
        "--census-batches",
        "4",
        "--threads",
        "2",
    ]);
    assert!(
        o.status.success(),
        "census: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    let cr = fs::read_to_string(cdir.join("census.report")).unwrap();
    assert_eq!(kv(&cr, "adaptive_census_report"), "1");
    assert_eq!(kv(&cr, "gates"), final2.to_string());
    kv(&cr, "leakage.validated.affine.deeper_births");
    kv(&cr, "leakage.audit_passed");
    assert!(cdir.join("census").is_dir());
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn usage_errors_exit_2() {
    let dir = scratch();
    let out = dir.join("out");
    let cases: Vec<Vec<&str>> = vec![
        vec![],
        vec!["frobnicate"],
        vec!["run", "--input", "x", "--source-c", "y", "--n", "8"],
        vec![
            "run",
            "--input",
            "x",
            "--source-c",
            "y",
            "--out-dir",
            p(&out),
        ],
        vec![
            "run",
            "--input",
            "x",
            "--source-c",
            "y",
            "--n",
            "8",
            "--out-dir",
            p(&out),
            "--bogus",
            "1",
        ],
        vec![
            "run",
            "--input",
            "x",
            "--source-c",
            "y",
            "--n",
            "8",
            "--out-dir",
            p(&out),
            "--accept",
            "larger",
        ],
        vec![
            "run",
            "--input",
            "x",
            "--source-c",
            "y",
            "--n",
            "8",
            "--out-dir",
            p(&out),
            "--seed",
            "1",
            "--seed-file",
            "f",
        ],
        vec![
            "run",
            "--input",
            "x",
            "--source-c",
            "y",
            "--n",
            "8",
            "--out-dir",
            p(&out),
            "--time-limit",
            "abc",
        ],
        vec![
            "run",
            "--input",
            "x",
            "--source-c",
            "y",
            "--n",
            "8",
            "--out-dir",
            p(&out),
            "--threads",
            "0",
        ],
        vec![
            "run",
            "--input",
            "x",
            "--source-c",
            "y",
            "--n",
            "8",
            "--out-dir",
            p(&out),
            "--stop-fraction",
            "1.5",
        ],
        vec![
            "census",
            "--input",
            "x",
            "--source-c",
            "y",
            "--out-dir",
            p(&out),
        ],
    ];
    for args in cases {
        let o = adaptive(&args);
        assert_eq!(
            o.status.code(),
            Some(2),
            "{args:?}: {}",
            String::from_utf8_lossy(&o.stderr)
        );
        let err = String::from_utf8_lossy(&o.stderr);
        assert_eq!(
            err.lines().count(),
            2,
            "one-line message plus the exit line: {err}"
        );
    }
    // a missing input is an input error (3)
    let o = adaptive(&[
        "run",
        "--input",
        p(&dir.join("none.mpmct1")),
        "--source-c",
        "y",
        "--n",
        "8",
        "--out-dir",
        p(&out),
    ]);
    assert_eq!(o.status.code(), Some(3));
    let o = adaptive(&["help"]);
    assert!(o.status.success());
    assert!(String::from_utf8_lossy(&o.stdout).contains("tdp_adaptive run"));
    let _ = fs::remove_dir_all(dir);
}
