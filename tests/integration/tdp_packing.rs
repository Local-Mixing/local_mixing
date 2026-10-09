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

// ---------------------------------------------------------------- packer

/// `pack/cases.txt`: `name A B argv...` (`OUT` = the output circuit path).
fn pack_cases() -> Vec<(String, String, String, Vec<String>)> {
    fs::read_to_string(fx("pack/cases.txt"))
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let mut it = l.split_whitespace().map(String::from);
            let name = it.next().unwrap();
            let a = it.next().unwrap();
            let b = it.next().unwrap();
            (name, a, b, it.collect())
        })
        .collect()
}

fn digest(path: &Path) -> String {
    use local_mixing::stages::packing::blake2b::{blake2b, hex};
    hex(&blake2b(&fs::read(path).unwrap(), 16))
}

/// Run one fixture case; returns (stdout, packed path if any, order path).
fn run_pack_case(
    dir: &Path,
    a: &str,
    b: &str,
    argv: &[String],
    extra: &[&str],
) -> (String, Option<PathBuf>, PathBuf) {
    let out = dir.join("packed.mpmct1");
    let order = dir.join("order.txt");
    let mut args: Vec<String> = vec!["pack".into(), p(&fx(a)).into(), p(&fx(b)).into()];
    let mut has_out = false;
    for t in argv {
        if t == "OUT" {
            args.push(p(&out).into());
            has_out = true;
        } else {
            args.push(t.clone());
        }
    }
    args.push("--order-out".into());
    args.push(p(&order).into());
    args.extend(extra.iter().map(|s| s.to_string()));
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let stdout = ok(&refs);
    (stdout, has_out.then_some(out), order)
}

#[test]
fn pack_matches_oracle() {
    let cases = pack_cases();
    assert!(cases.len() >= 15);
    for (name, a, b, argv) in cases {
        let dir = scratch();
        let (stdout, packed, order) = run_pack_case(&dir, &a, &b, &argv, &[]);
        let want = fs::read_to_string(fx(&format!("pack/expected/{name}.txt"))).unwrap();
        assert_eq!(stdout, want, "{name}: stdout differs from window_pack3.py");
        let mut got_digest = String::new();
        if let Some(pk) = &packed {
            got_digest.push_str(&format!("packed {}\n", digest(pk)));
            let mut bp = pk.as_os_str().to_owned();
            bp.push(".bounds");
            assert_eq!(
                fs::read(PathBuf::from(bp)).unwrap(),
                fs::read(fx(&format!("pack/expected/{name}.bounds"))).unwrap(),
                "{name}: bounds differ"
            );
        }
        got_digest.push_str(&format!("order {}\n", digest(&order)));
        assert_eq!(
            got_digest,
            fs::read_to_string(fx(&format!("pack/expected/{name}.digest"))).unwrap(),
            "{name}: packed circuit or emitted order differs"
        );
        let _ = fs::remove_dir_all(dir);
    }
}

#[test]
fn pack_thread_count_does_not_change_output() {
    let case = pack_cases().into_iter().find(|c| c.0 == "n8_prod").unwrap();
    let mut outs = Vec::new();
    for t in ["1", "4"] {
        let dir = scratch();
        let (stdout, packed, order) =
            run_pack_case(&dir, &case.1, &case.2, &case.3, &["--threads", t]);
        outs.push((
            stdout,
            fs::read(packed.unwrap()).unwrap(),
            fs::read(order).unwrap(),
        ));
        let _ = fs::remove_dir_all(dir);
    }
    assert!(outs[0] == outs[1]);
}

#[test]
fn pack_exit_codes() {
    let dir = scratch();
    let code = |args: &[&str]| run(args).status.code();
    let a = fx("pack/dense16_a.mpmct1");
    let b = fx("pack/dense16_b.mpmct1");
    let out = dir.join("o.mpmct1");
    let leftovers = |d: &Path| fs::read_dir(d).unwrap().count();
    // 2: usage
    assert_eq!(code(&["pack", p(&a)]), Some(2));
    assert_eq!(code(&["pack", p(&a), p(&b), "x"]), Some(2));
    assert_eq!(code(&["pack", p(&a), p(&b), "--bogus", "1"]), Some(2));
    assert_eq!(
        code(&[
            "pack",
            p(&a),
            p(&b),
            "8",
            "8",
            "64",
            "300",
            "l",
            "o",
            "extra"
        ]),
        Some(2)
    );
    // 3: malformed gate line; empty B
    let bad = dir.join("bad.mpmct1");
    fs::write(&bad, "mpmct1 4 1\n0 0 2 3 1 1 1\n").unwrap();
    assert_eq!(code(&["pack", p(&a), p(&bad)]), Some(3));
    let empty = dir.join("empty.mpmct1");
    fs::write(&empty, "mpmct1 16 0\n").unwrap();
    assert_eq!(
        code(&[
            "pack",
            p(&a),
            p(&empty),
            "8",
            "8",
            "64",
            "300",
            "x",
            p(&out)
        ]),
        Some(3)
    );
    // 4: B is not A's function, so the oracle's VERIFY fails; no output
    let text = fs::read_to_string(&b).unwrap();
    let mut lines: Vec<String> = text.lines().map(String::from).collect();
    let g: Vec<&str> = lines[10].split_whitespace().collect();
    let flipped = format!(
        "{} {} {}",
        g[0],
        1 - g[1].parse::<u8>().unwrap(),
        g[2..].join(" ")
    );
    lines[10] = flipped;
    let wrong = dir.join("wrong.mpmct1");
    fs::write(&wrong, lines.join("\n") + "\n").unwrap();
    let before = leftovers(&dir);
    let o = run(&[
        "pack",
        p(&a),
        p(&wrong),
        "8",
        "8",
        "64",
        "300",
        "x",
        p(&out),
    ]);
    assert_eq!(o.status.code(), Some(4));
    assert!(String::from_utf8_lossy(&o.stdout).contains("(FAIL)"));
    assert_eq!(leftovers(&dir), before, "no output after a failed VERIFY");
    // 7: the output directory does not exist
    let nowhere = dir.join("missing/o.mpmct1");
    assert_eq!(
        code(&[
            "pack",
            p(&a),
            p(&b),
            "8",
            "8",
            "64",
            "300",
            "x",
            p(&nowhere)
        ]),
        Some(7)
    );
    let _ = fs::remove_dir_all(dir);
}

// ---------------------------------------------------------------- repair

/// `repair/cases.txt`: `name packed bounds c n split_budget fix fallback`
/// (`@n8` = the real n=8 pack, made here by `tdp_packing pack`, whose output
/// is the oracle's: see `pack_matches_oracle`).
struct RepairCase {
    name: String,
    packed: String,
    bounds: String,
    c: String,
    n: String,
    budget: String,
    fix: String,
    fallback: String,
}

fn repair_cases() -> Vec<RepairCase> {
    fs::read_to_string(fx("repair/cases.txt"))
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let f: Vec<String> = l.split_whitespace().map(String::from).collect();
            assert_eq!(f.len(), 8, "repair case line {l:?}");
            RepairCase {
                name: f[0].clone(),
                packed: f[1].clone(),
                bounds: f[2].clone(),
                c: f[3].clone(),
                n: f[4].clone(),
                budget: f[5].clone(),
                fix: f[6].clone(),
                fallback: f[7].clone(),
            }
        })
        .collect()
}

/// The packed input of a case: (circuit, bounds).
fn repair_input(rc: &RepairCase, dir: &Path) -> (PathBuf, PathBuf) {
    if rc.packed == "@n8" {
        let out = dir.join("packed.mpmct1");
        ok(&[
            "pack",
            p(&fx("cview_n8/g2.mpmct1")),
            p(&fx("n8_L8/g3.mpmct1")),
            "512",
            "512",
            "24576",
            "3000",
            "n8-G512",
            p(&out),
        ]);
        (out.clone(), dir.join("packed.mpmct1.bounds"))
    } else {
        (fx(&rc.packed), fx(&rc.bounds))
    }
}

fn run_repair_case(rc: &RepairCase, dir: &Path, extra: &[&str]) -> (Output, PathBuf) {
    let (packed, bounds) = repair_input(rc, dir);
    let out = dir.join("out");
    let c = fx(&rc.c);
    let mut args = vec![
        "repair",
        "--packed",
        p(&packed),
        "--bounds",
        p(&bounds),
        "--source-c",
        p(&c),
        "--n",
        &rc.n,
        "--out-dir",
        p(&out),
        "--split-budget",
        &rc.budget,
        "--fix-iterations",
        &rc.fix,
        "--fallback-iterations",
        &rc.fallback,
    ];
    args.extend_from_slice(extra);
    (run(&args), out)
}

/// The oracle runs its first `verify_equiv` in the background, so that line's
/// position in the log is a race: compare the other lines in order and the
/// verify lines as a multiset.
fn split_verify_lines(log: &str) -> (Vec<String>, Vec<String>) {
    let (mut other, mut verify): (Vec<String>, Vec<String>) = (Vec::new(), Vec::new());
    for l in log.lines() {
        if l.ends_with("EQUIVALENT (1024 random states)") {
            verify.push(l.to_string());
        } else {
            other.push(l.to_string());
        }
    }
    verify.sort();
    (other, verify)
}

#[test]
fn repair_matches_oracle() {
    let cases = repair_cases();
    assert!(cases.len() >= 6);
    for rc in &cases {
        let dir = scratch();
        let (o, out) = run_repair_case(rc, &dir, &["--allow-floor-miss", "--allow-over-budget"]);
        let name = &rc.name;
        assert!(
            o.status.success(),
            "{name}: {}\n{}",
            o.status,
            String::from_utf8_lossy(&o.stderr)
        );
        let exp = |s: &str| fx(&format!("repair/expected/{name}.{s}"));
        let got = |s: &str| out.join(format!("rep512.{s}"));
        let want_rep = if rc.packed == "@n8" {
            fx("n8_L8/rep.mpmct1")
        } else {
            exp("rep.mpmct1")
        };
        assert!(
            fs::read(got("mpmct1")).unwrap() == fs::read(want_rep).unwrap(),
            "{name}: repaired circuit differs from the oracle's"
        );
        for s in ["bounds", "exposure.tsv", "allcuts.tsv", "wit.tsv"] {
            assert!(
                fs::read(got(s)).unwrap() == fs::read(exp(s)).unwrap(),
                "{name}: rep512.{s} differs from the oracle's"
            );
        }
        let prefix = format!("{}/", out.display());
        let norm = |path: PathBuf| fs::read_to_string(path).unwrap().replace(&prefix, "<DIR>/");
        assert_eq!(
            norm(got("cview.log")),
            fs::read_to_string(exp("cview.log")).unwrap(),
            "{name}: cview.log"
        );
        let log = norm(out.join("repair512.log"));
        assert_eq!(
            split_verify_lines(&log),
            split_verify_lines(&fs::read_to_string(exp("log")).unwrap()),
            "{name}: repair log"
        );
        assert_eq!(
            String::from_utf8(o.stdout).unwrap(),
            fs::read_to_string(out.join("repair512.log")).unwrap(),
            "{name}: stdout is the log"
        );
        let _ = fs::remove_dir_all(dir);
    }
}

#[test]
fn repair_thread_count_does_not_change_output() {
    let rc = repair_cases()
        .into_iter()
        .find(|c| c.name == "leaky")
        .unwrap();
    let mut outs = Vec::new();
    for t in ["1", "4"] {
        let dir = scratch();
        let (o, out) = run_repair_case(&rc, &dir, &["--allow-floor-miss", "--threads", t]);
        assert!(o.status.success());
        let mut files = Vec::new();
        for f in [
            "rep512.mpmct1",
            "rep512.bounds",
            "rep512.exposure.tsv",
            "rep512.allcuts.tsv",
            "rep512.wit.tsv",
        ] {
            files.push(fs::read(out.join(f)).unwrap());
        }
        outs.push(files);
        let _ = fs::remove_dir_all(dir);
    }
    assert!(outs[0] == outs[1]);
}

#[test]
fn repair_single_window_case() {
    // the orderdep inputs with no boundary at all: the oracle's hmap_affine
    // panics on such a lattice ("empty plate"); here it is scanned as zero
    // cuts (visible 0) and the window is kept whole when it fits.
    let mut rc = repair_cases()
        .into_iter()
        .find(|c| c.name == "orderdep")
        .unwrap();
    let dir = scratch();
    let empty = dir.join("empty.bounds");
    fs::write(&empty, "\n").unwrap();
    rc.bounds = p(&empty).to_string();
    rc.packed = p(&fx(&rc.packed)).to_string();
    rc.budget = "65536".into();
    let (o, out) = run_repair_case(&rc, &dir, &["--stats"]);
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(o.status.success(), "{err}");
    assert!(err.contains("repair.single_window_scans=2"), "{err}");
    assert_eq!(fs::read_to_string(out.join("rep512.bounds")).unwrap(), "\n");
    let log = fs::read_to_string(out.join("repair512.log")).unwrap();
    assert!(log.contains("fix iter 1: C-view visible 0 (floor 8)\n  (single window"));
    assert!(log.contains("0 splits added (0 -> 0 boundaries)"));
    let cv = fs::read_to_string(out.join("rep512.cview.log")).unwrap();
    assert!(cv.contains(" cols=0, "), "{cv}");
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn repair_exit_codes_and_check_final() {
    let cases = repair_cases();
    let get = |n: &str| cases.iter().find(|c| c.name == n).unwrap();
    let code = |o: &Output| o.status.code();
    // 5: leaky ends with visible non-output rows; the repair outputs (oracle
    // intermediates) are still written; --allow-floor-miss gives 0
    let dir = scratch();
    let (o, out) = run_repair_case(get("leaky"), &dir, &[]);
    assert_eq!(code(&o), Some(5));
    assert!(out.join("rep512.mpmct1").exists());
    let (o, _) = run_repair_case(get("leaky"), &dir, &["--allow-floor-miss"]);
    assert_eq!(code(&o), Some(0));
    assert!(String::from_utf8_lossy(&o.stderr).contains("status=floor_miss"));
    // check-final agrees on the oracle's own repaired leaky circuit
    let cf = |rep: &Path, bounds: &Path, c: &str, extra: &[&str]| {
        let mut a = vec![
            "check-final",
            "--rep",
            p(rep),
            "--bounds",
            p(bounds),
            "--source-c",
            c,
            "--n",
            "8",
        ];
        a.extend_from_slice(extra);
        run(&a)
    };
    let leaky_c = fx("repair/leaky.c.g57");
    let o = cf(
        &fx("repair/expected/leaky.rep.mpmct1"),
        &fx("repair/expected/leaky.bounds"),
        p(&leaky_c),
        &[],
    );
    assert_eq!(code(&o), Some(5));
    assert!(String::from_utf8_lossy(&o.stdout).contains("status=floor_miss"));
    // ... and passes the real n=8 repair (the oracle's rep.mpmct1)
    let n8c = fx("n8_L8/c.g57");
    let o = cf(
        &fx("n8_L8/rep.mpmct1"),
        &fx("n8_L8/rep.bounds"),
        p(&n8c),
        &[],
    );
    assert_eq!(code(&o), Some(0), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(String::from_utf8_lossy(&o.stdout).contains("status=ok"));
    // 6: orderdep (nothing visible) with a budget no window can meet
    let mut od = repair_cases()
        .into_iter()
        .find(|c| c.name == "orderdep")
        .unwrap();
    od.budget = "2".into();
    let d2 = scratch();
    let (o, _) = run_repair_case(&od, &d2, &[]);
    assert_eq!(code(&o), Some(6), "{}", String::from_utf8_lossy(&o.stderr));
    let (o, _) = run_repair_case(&od, &d2, &["--allow-over-budget"]);
    assert_eq!(code(&o), Some(0));
    assert!(String::from_utf8_lossy(&o.stderr).contains("status=over_budget"));
    // 2: usage; 3: malformed input
    assert_eq!(run(&["repair", "--packed", "x"]).status.code(), Some(2));
    let bad = d2.join("bad.mpmct1");
    fs::write(&bad, "mpmct1 32 1\n0 0 2 3 1 1 1\n").unwrap();
    let mut bad_case = repair_cases()
        .into_iter()
        .find(|c| c.name == "leaky")
        .unwrap();
    bad_case.packed = p(&bad).to_string();
    bad_case.bounds = p(&fx("repair/leaky.packed.bounds")).to_string();
    let (o, _) = run_repair_case(&bad_case, &d2, &[]);
    assert_eq!(code(&o), Some(3));
    // 7: the output directory cannot be created
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let ro = d2.join("ro");
        fs::create_dir_all(&ro).unwrap();
        fs::set_permissions(&ro, fs::Permissions::from_mode(0o500)).unwrap();
        let rc = get("sparse");
        let o = run(&[
            "repair",
            "--packed",
            p(&fx(&rc.packed)),
            "--bounds",
            p(&fx(&rc.bounds)),
            "--source-c",
            p(&fx(&rc.c)),
            "--n",
            "8",
            "--out-dir",
            p(&ro.join("sub")),
            "--allow-floor-miss",
        ]);
        fs::set_permissions(&ro, fs::Permissions::from_mode(0o700)).unwrap();
        assert_eq!(code(&o), Some(7), "{}", String::from_utf8_lossy(&o.stderr));
    }
    let _ = fs::remove_dir_all(dir);
    let _ = fs::remove_dir_all(d2);
}
