//! `tdp_unpacking` end to end (docs/STAGE7_UNPACKING.md): a deliverable built
//! by `tdp_packing run` on the committed n8_L8 inputs is unpacked, the result
//! is checked for equivalence here (independently of the binary's own
//! checks), and the exit codes and the no-fallback path are exercised on the
//! small crafted stage-4 fixtures.
use local_mixing::circuit::formats::read_g57_file;
use local_mixing::stages::packing::oracle_io::read_oracle_mpmct;
use local_mixing::stages::packing::pyrand::PyRandom;
use local_mixing::stages::packing::serialize::{Format, decode};
use local_mixing::stages::packing::verify::{eval_gates_lanes16, random_lanes};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

fn fx(p: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/stage4")
        .join(p)
}

fn scratch() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "tdp_unpacking_it_{}_{}",
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

fn unpack(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_tdp_unpacking"))
        .env_clear()
        .args(args)
        .output()
        .expect("spawn tdp_unpacking")
}

fn packing(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_tdp_packing"))
        .env_clear()
        .args(args)
        .output()
        .expect("spawn tdp_packing")
}

fn kv(report: &str, key: &str) -> String {
    report
        .lines()
        .find_map(|l| l.strip_prefix(&format!("{key}=")))
        .unwrap_or_else(|| panic!("{key} missing from unpacking.report"))
        .to_string()
}

/// The circuit and the deliverable agree on `batches` x 1024 random states.
fn equivalent(circuit: &Path, deliverable: &Path, batches: usize) {
    let c = read_oracle_mpmct(circuit).unwrap();
    let d = decode(&fs::read(deliverable).unwrap(), Format::Container).unwrap();
    let nw = d.state_wires as usize;
    assert_eq!(c.header_n, nw, "header wire count = state width");
    assert!(c.computed_n <= nw);
    let mut rng = PyRandom::new(4242);
    for _ in 0..batches {
        let x = random_lanes(&mut rng, nw);
        let mut sg = x.clone();
        eval_gates_lanes16(&c.gates, &mut sg);
        let mut sd = x;
        d.eval_lanes16(&mut sd);
        assert_eq!(sg, sd, "unpacked circuit != deliverable");
    }
}

/// The n8_L8 stage-4 run (`tdp_packing run` on the committed G2, G3, C),
/// built once per test binary: its `packing/` directory.
fn n8_packing() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = scratch();
        let out = dir.join("packing");
        let o = packing(&[
            "run",
            "--stage2",
            p(&fx("cview_n8/g2.mpmct1")),
            "--stage3",
            p(&fx("n8_L8/g3.mpmct1")),
            "--source-c",
            p(&fx("n8_L8/c.g57")),
            "--n",
            "8",
            "--out-dir",
            p(&out),
            "--label",
            "n8_L8",
            "--no-oracle-files",
        ]);
        assert!(
            o.status.success(),
            "tdp_packing run: {}",
            String::from_utf8_lossy(&o.stderr)
        );
        assert!(out.join("deliverable.tdp4").is_file());
        out
    })
}

/// A small crafted deliverable (`anf/<case>`) as a container: (scratch dir,
/// container path).
fn crafted(case: &str, source_wires: &str) -> (PathBuf, PathBuf) {
    let dir = scratch();
    let bin = dir.join(format!("{case}.bin"));
    let tdp4 = dir.join(format!("{case}.tdp4"));
    let o = packing(&[
        "serialize",
        p(&fx(&format!("anf/{case}.mpmct1"))),
        p(&fx(&format!("anf/{case}.bounds"))),
        p(&bin),
        case,
        "--container",
        p(&tdp4),
        "--source-wires",
        source_wires,
    ]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    (dir, tdp4)
}

#[test]
fn unpacks_the_n8_l8_deliverable_with_fallback_and_zero_slice() {
    let pk = n8_packing();
    let out = scratch().join("unpacking");
    let o = unpack(&[
        "run",
        "--deliverable",
        p(&pk.join("deliverable.tdp4")),
        "--packed",
        p(&pk.join("rep512.mpmct1")),
        "--bounds",
        p(&pk.join("rep512.bounds")),
        "--source-c",
        p(&fx("n8_L8/c.g57")),
        "--n",
        "8",
        "--out-dir",
        p(&out),
        "--seed",
        "3",
        "--threads",
        "4",
        "--restarts",
        "2",
        "--time-limit",
        "3",
        "--label",
        "n8",
    ]);
    assert!(
        o.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    for f in [
        "unpacked.mpmct1",
        "windows.tsv",
        "unpacking.report",
        "unpacking.log",
    ] {
        assert!(out.join(f).is_file(), "{f} missing");
        assert!(
            !out.join(format!("{f}.tmp")).exists(),
            "{f}.tmp left behind"
        );
    }
    let report = fs::read_to_string(out.join("unpacking.report")).unwrap();
    assert_eq!(kv(&report, "unpacking_report"), "1");
    assert_eq!(kv(&report, "state_wires"), "32");
    assert_eq!(kv(&report, "n"), "8");
    assert_eq!(kv(&report, "windows"), "73");
    assert_eq!(kv(&report, "gates_in"), "34825");
    assert_eq!(kv(&report, "param.seed_source"), "argv");
    assert!(
        !report.contains("seed=3"),
        "the seed value is in the report"
    );
    assert_eq!(kv(&report, "check.equiv_canonical"), "ok(4096)");
    assert_eq!(kv(&report, "check.zero_slice"), "ok(4096)");
    let syn: usize = kv(&report, "synthesized").parse().unwrap();
    let fb: usize = kv(&report, "fallback").parse().unwrap();
    assert_eq!(syn + fb, 73);
    assert_eq!(
        kv(&report, "status"),
        if fb > 0 { "fallback_used" } else { "ok" }
    );
    let tsv = fs::read_to_string(out.join("windows.tsv")).unwrap();
    assert_eq!(tsv.lines().count(), 74);
    let header: Vec<&str> = tsv.lines().next().unwrap().split('\t').collect();
    assert_eq!(header[0], "idx");
    assert_eq!(header[11], "syn_offset");
    let rows: Vec<Vec<&str>> = tsv
        .lines()
        .skip(1)
        .map(|l| l.split('\t').collect())
        .collect();
    // orig_* follow the packed bounds; syn_offset is the cumulative count
    let mut off = 0usize;
    for (i, r) in rows.iter().enumerate() {
        assert_eq!(r[0], i.to_string());
        let lo: usize = r[1].parse().unwrap();
        let hi: usize = r[2].parse().unwrap();
        assert_eq!(r[3], (hi - lo).to_string());
        assert_eq!(r[11], off.to_string());
        off += r[4].parse::<usize>().unwrap();
        assert!(matches!(r[9], "ok" | "fallback"), "{}", r[9]);
    }
    assert_eq!(kv(&report, "gates_out"), off.to_string());
    let circ = read_oracle_mpmct(&out.join("unpacked.mpmct1")).unwrap();
    assert_eq!(circ.gates.len(), off);
    assert_eq!(kv(&report, "gates_canonical"), off.to_string());
    // the canonical circuit is the deliverable's function (our own check)
    equivalent(
        &out.join("unpacked.mpmct1"),
        &pk.join("deliverable.tdp4"),
        2,
    );
    // and C sits on the zero slice
    let c = read_g57_file(p(&fx("n8_L8/c.g57"))).unwrap();
    let mut rng = PyRandom::new(9);
    let x = random_lanes(&mut rng, 8);
    let mut want = x.clone();
    eval_gates_lanes16(&c, &mut want);
    let mut st = x;
    st.resize(32, [0; 16]);
    eval_gates_lanes16(&circ.gates, &mut st);
    assert_eq!(&st[8..16], &want[..]);
    // the gate order is canonical: canonicalizing again changes nothing
    let mut again = circ.gates.clone();
    local_mixing::canonicalization::gate_order::xgate_canonicalize(&mut again);
    assert_eq!(again, circ.gates);
}

#[test]
fn crafted_cases_synthesize_without_fallback_and_reproduce_by_seed() {
    let (dir, tdp4) = crafted("cancel", "1");
    let run = |out: &Path, extra: &[&str]| {
        let mut args = vec![
            "run",
            "--deliverable",
            p(&tdp4),
            "--out-dir",
            p(out),
            "--no-fallback",
            "--threads",
            "2",
            "--restarts",
            "2",
        ];
        args.extend_from_slice(extra);
        unpack(&args)
    };
    let a = dir.join("a");
    let o = run(&a, &["--seed", "5"]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let report = fs::read_to_string(a.join("unpacking.report")).unwrap();
    assert_eq!(kv(&report, "status"), "ok");
    assert_eq!(kv(&report, "fallback"), "0");
    assert_eq!(kv(&report, "gates_in"), "unknown");
    assert_eq!(kv(&report, "ratio"), "unknown");
    assert_eq!(kv(&report, "check.zero_slice"), "skipped");
    assert_eq!(kv(&report, "windows"), "48");
    equivalent(&a.join("unpacked.mpmct1"), &tdp4, 1);
    // the same seed from a file gives the same circuit; the report says "file"
    let seed_file = dir.join("seed.txt");
    fs::write(&seed_file, "5\n").unwrap();
    let b = dir.join("b");
    let o = run(&b, &["--seed-file", p(&seed_file)]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(
        fs::read(a.join("unpacked.mpmct1")).unwrap(),
        fs::read(b.join("unpacked.mpmct1")).unwrap()
    );
    let rb = fs::read_to_string(b.join("unpacking.report")).unwrap();
    assert_eq!(kv(&rb, "param.seed_source"), "file");
    let log = fs::read_to_string(b.join("unpacking.log")).unwrap();
    assert!(log.contains("seed (file)"), "{log}");
    assert!(!log.contains("seed 5") && !rb.contains("seed=5"));
    // another seed may give another (equivalent) circuit
    let c = dir.join("c");
    let o = run(&c, &["--seed", "6"]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    equivalent(&c.join("unpacked.mpmct1"), &tdp4, 1);
    // a packed circuit with another window count is exit 3
    let o = run(
        &dir.join("d"),
        &[
            "--packed",
            p(&fx("anf/w32_dense.mpmct1")),
            "--bounds",
            p(&fx("anf/w32_dense.bounds")),
        ],
    );
    assert_eq!(
        o.status.code(),
        Some(3),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    // the packed circuit the deliverable came from is accepted
    let e = dir.join("e");
    let o = run(
        &e,
        &[
            "--packed",
            p(&fx("anf/cancel.mpmct1")),
            "--bounds",
            p(&fx("anf/cancel.bounds")),
            "--seed",
            "5",
        ],
    );
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let re = fs::read_to_string(e.join("unpacking.report")).unwrap();
    assert_eq!(kv(&re, "gates_in"), "99");
    assert_eq!(
        fs::read(a.join("unpacked.mpmct1")).unwrap(),
        fs::read(e.join("unpacked.mpmct1")).unwrap(),
        "the packed circuit must not influence a synthesized window"
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn exit_codes() {
    let dir = scratch();
    let (_, tdp4) = crafted("nogates", "1");
    let code = |args: &[&str]| unpack(args).status.code();
    // usage
    assert_eq!(code(&[]), Some(2));
    assert_eq!(code(&["bogus"]), Some(2));
    assert_eq!(code(&["run", "--out-dir", p(&dir)]), Some(2));
    let base = ["run", "--deliverable", p(&tdp4), "--out-dir", p(&dir)];
    let with = |extra: &[&str]| {
        let mut a = base.to_vec();
        a.extend_from_slice(extra);
        code(&a)
    };
    assert_eq!(with(&["--bogus", "1"]), Some(2));
    assert_eq!(with(&["--seed", "1", "--seed-file", p(&tdp4)]), Some(2));
    assert_eq!(with(&["--seed-file", p(&tdp4)]), Some(2));
    assert_eq!(with(&["--packed", p(&tdp4)]), Some(2));
    assert_eq!(with(&["--restarts", "0"]), Some(2));
    assert_eq!(with(&["--time-limit", "x"]), Some(2));
    assert_eq!(with(&["--threads", "0"]), Some(2));
    assert_eq!(with(&["--n", "1"]), Some(2));
    assert_eq!(
        with(&["--source-c", p(&fx("n8_L8/c.g57")), "--n", "7"]),
        Some(2)
    );
    // input
    assert_eq!(
        code(&[
            "run",
            "--deliverable",
            p(&dir.join("missing")),
            "--out-dir",
            p(&dir)
        ]),
        Some(3)
    );
    let raw = fx("anf/nogates.bin");
    assert_eq!(
        code(&["run", "--deliverable", p(&raw), "--out-dir", p(&dir)]),
        Some(3)
    );
    assert_eq!(
        with(&[
            "--packed",
            p(&dir.join("missing")),
            "--bounds",
            p(&dir.join("missing"))
        ]),
        Some(3)
    );
    // an empty deliverable unpacks to an empty circuit
    let out = dir.join("empty");
    let o = unpack(&["run", "--deliverable", p(&tdp4), "--out-dir", p(&out)]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let c = read_oracle_mpmct(&out.join("unpacked.mpmct1")).unwrap();
    assert!(c.gates.is_empty());
    assert_eq!(c.header_n, 8);
    // --out-dir on a file is exit 2
    assert_eq!(
        code(&["run", "--deliverable", p(&tdp4), "--out-dir", p(&tdp4)]),
        Some(2)
    );
    // selftest
    let o = unpack(&[
        "selftest",
        "--n",
        "6",
        "--m",
        "6",
        "--trials",
        "4",
        "--threads",
        "2",
    ]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(String::from_utf8_lossy(&o.stdout).contains("4 windows synthesized"));
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn thread_count_does_not_change_the_output() {
    // the small `cancel` case: the dense cases run into the per-window
    // deadline under the unoptimized test build, which makes their output
    // legitimately timing-dependent
    let (dir, tdp4) = crafted("cancel", "1");
    let mut outs = Vec::new();
    for t in ["1", "3"] {
        let out = dir.join(format!("t{t}"));
        let o = unpack(&[
            "run",
            "--deliverable",
            p(&tdp4),
            "--out-dir",
            p(&out),
            "--no-fallback",
            "--threads",
            t,
            "--restarts",
            "2",
            "--seed",
            "11",
        ]);
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        outs.push(fs::read(out.join("unpacked.mpmct1")).unwrap());
        let report = fs::read_to_string(out.join("unpacking.report")).unwrap();
        assert_eq!(kv(&report, "deadline_hits"), "0", "timing-dependent output");
        equivalent(&out.join("unpacked.mpmct1"), &tdp4, 1);
    }
    assert_eq!(outs[0], outs[1]);
    let _ = fs::remove_dir_all(dir);
}
