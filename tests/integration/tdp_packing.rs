//! `tdp_packing` against outputs of the Python ORACLE (docs/STAGE4_RUST_DESIGN.md
//! §10.3-10.4). Every expected file under `tests/fixtures/stage4/{anf,n8_L8}`
//! was produced by the unmodified oracle scripts (`anf_dump_eval.py`,
//! `anf_compose.py`, and `anf_compose.window_anf` via `anf_windows.py`) run
//! in a sandbox by `tests/fixtures/stage4/regen.sh` under an interpreter that
//! passes the CPython 3.9 set-order self-test.
//!
//! Deliverable bodies are compared byte for byte; reports exactly (only the
//! wall time in `anf_dump_eval.py`'s first line is normalized).
use local_mixing::circuit::formats::read_g57_file;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

fn fx(p: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/stage4")
        .join(p)
}

fn scratch() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "tdp_packing_it_{}_{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_tdp_packing"))
        .env_clear()
        .args(args)
        .output()
        .expect("spawn tdp_packing")
}

fn ok(args: &[&str]) -> String {
    let o = run(args);
    assert!(
        o.status.success(),
        "tdp_packing {args:?} failed: {}\n{}",
        o.status,
        String::from_utf8_lossy(&o.stderr)
    );
    String::from_utf8(o.stdout).unwrap()
}

fn p(x: &Path) -> &str {
    x.to_str().unwrap()
}

/// `anf_dump_eval.py` line 1 with the wall time normalized.
fn norm_line1(s: &str) -> String {
    let l = s.lines().next().unwrap_or("");
    let (head, tail) = l.split_once(" in ").expect("line 1 has ' in '");
    let rest = tail.split_once("s,").expect("line 1 has 's,'").1;
    format!("{head} in <T>s,{rest}")
}

struct Case {
    name: String,
    budgets: Vec<String>,
}

fn cases() -> Vec<Case> {
    fs::read_to_string(fx("anf/cases.txt"))
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let mut it = l.split_whitespace();
            Case {
                name: it.next().unwrap().to_string(),
                budgets: it.map(String::from).collect(),
            }
        })
        .collect()
}

#[test]
fn serialize_matches_oracle_crafted_cases() {
    let dir = scratch();
    let cs = cases();
    assert!(cs.len() >= 10);
    for c in &cs {
        let circ = fx(&format!("anf/{}.mpmct1", c.name));
        let bounds = fx(&format!("anf/{}.bounds", c.name));
        let out = dir.join(format!("{}.bin", c.name));
        let tdp4 = dir.join(format!("{}.tdp4", c.name));
        let stdout = ok(&[
            "serialize",
            p(&circ),
            p(&bounds),
            p(&out),
            &c.name,
            "--container",
            p(&tdp4),
            "--source-wires",
            "1",
        ]);
        let want = fs::read(fx(&format!("anf/{}.bin", c.name))).unwrap();
        assert!(
            fs::read(&out).unwrap() == want,
            "{}: body differs from anf_dump_eval.py",
            c.name
        );
        assert!(
            fs::read(&tdp4).unwrap()[48..] == want[..],
            "{}: container body",
            c.name
        );
        let line1 = fs::read_to_string(fx(&format!("anf/{}.deliver.txt", c.name))).unwrap();
        assert_eq!(norm_line1(&stdout), line1.trim_end(), "{}", c.name);
        // the oracle's evaluation check said match=True; ours ran the same check
        let m = fs::read_to_string(fx(&format!("anf/{}.match", c.name))).unwrap();
        assert_eq!(m.trim(), "match=True");
        assert!(stdout.contains("match=True"), "{}: {stdout}", c.name);
        let dec = ok(&["decode", p(&tdp4), "--format", "container"]);
        assert!(dec.starts_with("format=container"), "{dec}");
    }
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn serialize_matches_oracle_n8_l8_real_chain() {
    let dir = scratch();
    let out = dir.join("deliverable.bin");
    let tdp4 = dir.join("deliverable.tdp4");
    let stdout = ok(&[
        "serialize",
        p(&fx("n8_L8/rep.mpmct1")),
        p(&fx("n8_L8/rep.bounds")),
        p(&out),
        "n8_L8",
        "--container",
        p(&tdp4),
        "--source-wires",
        "8",
        "--zero-slice-c",
        p(&fx("n8_L8/c.g57")),
    ]);
    let want = fs::read(fx("n8_L8/deliverable.bin")).unwrap();
    assert!(
        fs::read(&out).unwrap() == want,
        "n8_L8 body differs from anf_dump_eval.py"
    );
    let cont = fs::read(&tdp4).unwrap();
    assert!(cont[48..] == want[..]);
    let hdr = fs::read(fx("n8_L8/container_header.bin")).unwrap();
    assert_eq!(cont[..48], hdr[..], "header != container_header.py");
    assert_eq!(&cont[..8], b"LMTDPANF");
    assert_eq!(u32::from_le_bytes(cont[16..20].try_into().unwrap()), 8);
    assert_eq!(u32::from_le_bytes(cont[20..24].try_into().unwrap()), 32);
    let line1 = fs::read_to_string(fx("n8_L8/deliver_line1.txt")).unwrap();
    assert_eq!(norm_line1(&stdout), line1.trim_end());
    assert!(
        stdout.contains("zero slice C(x) on wires 8..16 ok"),
        "{stdout}"
    );

    // eval on the container: x on wires 0..8 gives C(x) on 8..16
    let c = read_g57_file(p(&fx("n8_L8/c.g57"))).unwrap();
    for x in [0u64, 1, 0x5a, 0xff, 0x80, 0x37] {
        let mut st = [x];
        for g in &c {
            g.apply_limbs(&mut st);
        }
        let got = ok(&[
            "eval",
            p(&tdp4),
            "--format",
            "container",
            "--x",
            &format!("{x:x}"),
        ]);
        assert_eq!(got.trim(), format!("{:x}", st[0] & 0xff), "x={x:#x}");
    }
    // the raw body decodes with an explicit width only
    assert_eq!(run(&["decode", p(&out)]).status.code(), Some(2));
    ok(&["decode", p(&out), "--format", "raw", "--n", "8"]);
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn anf_report_matches_oracle() {
    for c in cases() {
        let circ = fx(&format!("anf/{}.mpmct1", c.name));
        let bounds = fx(&format!("anf/{}.bounds", c.name));
        for b in &c.budgets {
            let label = format!("{}-b{b}", c.name);
            let got = ok(&["anf-report", p(&circ), p(&bounds), b, &label, "2"]);
            let want = fs::read_to_string(fx(&format!("anf/{}.anf_b{b}.txt", c.name))).unwrap();
            assert_eq!(got, want, "{} budget {b}", c.name);
        }
    }
    let rep = fx("n8_L8/rep.mpmct1");
    let bounds = fx("n8_L8/rep.bounds");
    for (args, file) in [
        (["65536", "n8_L8-rep512", "500"], "n8_L8/anf_rep.txt"),
        (["64", "n8_L8-b64", "3"], "n8_L8/anf_b64.txt"),
        (["256", "n8_L8-b256", "3"], "n8_L8/anf_b256.txt"),
    ] {
        let got = ok(&["anf-report", p(&rep), p(&bounds), args[0], args[1], args[2]]);
        assert_eq!(got, fs::read_to_string(fx(file)).unwrap(), "{file}");
    }
}

#[test]
fn window_decisions_match_oracle_under_both_semantics() {
    let mut order_dependent = 0;
    for c in cases() {
        let circ = fx(&format!("anf/{}.mpmct1", c.name));
        let bounds = fx(&format!("anf/{}.bounds", c.name));
        for b in &c.budgets {
            let got = ok(&["window-anf", p(&circ), p(&bounds), b]);
            let want = fs::read_to_string(fx(&format!("anf/{}.win_b{b}.txt", c.name))).unwrap();
            assert_eq!(got, want, "{} budget {b}", c.name);
            order_dependent += want
                .lines()
                .filter(|l| {
                    let f: Vec<&str> = l.split_whitespace().collect();
                    f[3] != f[4]
                })
                .count();
        }
    }
    // the fixture must keep exercising decisions that depend on CPython order
    assert!(
        order_dependent >= 5,
        "only {order_dependent} order-dependent windows"
    );
}

#[test]
fn thread_count_does_not_change_output() {
    let dir = scratch();
    let mut bodies = Vec::new();
    for t in ["1", "4"] {
        let out = dir.join(format!("t{t}.bin"));
        ok(&[
            "serialize",
            p(&fx("anf/w300.mpmct1")),
            p(&fx("anf/w300.bounds")),
            p(&out),
            "w300",
            "--threads",
            t,
        ]);
        bodies.push(fs::read(&out).unwrap());
    }
    assert!(bodies[0] == bodies[1]);
    assert!(bodies[0] == fs::read(fx("anf/w300.bin")).unwrap());
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn exit_codes() {
    let dir = scratch();
    let bad = dir.join("bad.mpmct1");
    fs::write(&bad, "mpmct1 4 1\n0 0 2 3 1 1 1\n").unwrap();
    let empty = dir.join("empty.bounds");
    fs::write(&empty, "").unwrap();
    let single = fx("anf/single.mpmct1");
    let code = |args: &[&str]| run(args).status.code();
    // 3: non-canonical gate line; nothing is left behind
    let o = dir.join("o.bin");
    assert_eq!(
        code(&["serialize", p(&bad), p(&empty), p(&o), "x"]),
        Some(3)
    );
    assert!(!o.exists() && !dir.join("o.bin.tmp").exists());
    // 2: usage
    assert_eq!(code(&[]), Some(2));
    assert_eq!(code(&["frobnicate"]), Some(2));
    assert_eq!(code(&["run"]), Some(2));
    assert_eq!(code(&["anf-report", p(&bad)]), Some(2));
    assert_eq!(
        code(&[
            "serialize",
            p(&single),
            p(&empty),
            p(&o),
            "x",
            "--bogus",
            "1"
        ]),
        Some(2)
    );
    // 7: the output directory does not exist
    let nowhere = dir.join("missing/dir/o.bin");
    assert_eq!(
        code(&["serialize", p(&single), p(&empty), p(&nowhere), "x"]),
        Some(7)
    );
    // 3: a raw body is not a container
    let raw = fx("anf/single.bin");
    assert_eq!(code(&["decode", p(&raw), "--format", "container"]), Some(3));
    let _ = fs::remove_dir_all(dir);
}
