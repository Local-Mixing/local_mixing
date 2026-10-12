//! `tdp_unpacking` end to end (docs/STAGE7_UNPACKING.md, docs/DELIVERY_STAGE.md):
//! a deliverable built by `tdp_packing run` on the committed n8_L8 inputs is
//! unpacked, the result is checked for equivalence here (independently of the
//! binary's own checks), the `--windows` subset, `canon` and the `--refine`
//! loop (with and without forced failures) run on the same fixture, and the
//! exit codes and the no-fallback path are exercised on the small crafted
//! stage-4 fixtures.
use local_mixing::circuit::formats::read_g57_file;
use local_mixing::circuit::xgate::XGate;
use local_mixing::stages::packing::oracle_io::{read_bounds, read_oracle_mpmct, window_cuts};
use local_mixing::stages::packing::pyrand::PyRandom;
use local_mixing::stages::packing::serialize::{Format, decode};
use local_mixing::stages::packing::verify::{check_zero_slice, eval_gates_lanes16, random_lanes};
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

/// The synthesis settings of every n8_L8 run here (the unoptimized test build
/// is slow: 3 s per window, 2 restarts; a window may legitimately fall back).
const N8_ARGS: [&str; 10] = [
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
];

/// `tdp_unpacking run` on the n8_L8 packing with `extra` arguments (packed
/// circuit, source C and `N8_ARGS` always given); the output directory.
fn n8_run(out: &Path, extra: &[&str]) -> Output {
    let pk = n8_packing();
    let (deliv, rep, bounds, c) = (
        pk.join("deliverable.tdp4"),
        pk.join("rep512.mpmct1"),
        pk.join("rep512.bounds"),
        fx("n8_L8/c.g57"),
    );
    let mut args = vec![
        "run",
        "--deliverable",
        p(&deliv),
        "--packed",
        p(&rep),
        "--bounds",
        p(&bounds),
        "--source-c",
        p(&c),
        "--n",
        "8",
        "--out-dir",
        p(out),
    ];
    args.extend_from_slice(&N8_ARGS);
    args.extend_from_slice(extra);
    unpack(&args)
}

/// The plain `tdp_unpacking run` on the n8_L8 deliverable (fallback allowed,
/// zero slice against C), run once per test binary: its output directory.
fn n8_plain() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let out = scratch().join("unpacking");
        let o = n8_run(&out, &[]);
        assert!(
            o.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&o.stdout),
            String::from_utf8_lossy(&o.stderr)
        );
        out
    })
}

/// The packed circuit's cuts `[0] + bounds + [m]`.
fn n8_cuts() -> (Vec<XGate>, Vec<usize>) {
    let pk = n8_packing();
    let c = read_oracle_mpmct(&pk.join("rep512.mpmct1")).unwrap();
    let b = read_bounds(&pk.join("rep512.bounds")).unwrap();
    let cuts = window_cuts(&b, c.gates.len());
    (c.gates, cuts)
}

/// Two gate lists compute the same function on 1024 random 32-wire states.
fn same_function(a: &[XGate], b: &[XGate], seed: u64) -> bool {
    let mut rng = PyRandom::new(seed);
    let x = random_lanes(&mut rng, 32);
    let mut sa = x.clone();
    eval_gates_lanes16(a, &mut sa);
    let mut sb = x;
    eval_gates_lanes16(b, &mut sb);
    sa == sb
}

fn rows_of(tsv: &str) -> Vec<Vec<String>> {
    tsv.lines()
        .skip(1)
        .map(|l| l.split('\t').map(|s| s.to_string()).collect())
        .collect()
}

#[test]
fn unpacks_the_n8_l8_deliverable_with_fallback_and_zero_slice() {
    let pk = n8_packing();
    let out = n8_plain().to_path_buf();
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
fn windows_subset_is_a_plain_concatenation_of_the_selected_windows() {
    let out = scratch().join("sel");
    // indices and a half-open range: {0, 2, 5}
    let o = n8_run(&out, &["--windows", "5,2:3,0"]);
    assert!(
        o.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    let report = fs::read_to_string(out.join("unpacking.report")).unwrap();
    assert_eq!(kv(&report, "windows_selected"), "3");
    assert_eq!(kv(&report, "windows"), "3");
    assert_eq!(kv(&report, "check.equiv_concat"), "skipped");
    assert_eq!(kv(&report, "check.equiv_canonical"), "skipped");
    assert_eq!(kv(&report, "check.zero_slice"), "skipped");
    assert_eq!(kv(&report, "check.windows_exact"), "ok(3)");
    let tsv = fs::read_to_string(out.join("windows.tsv")).unwrap();
    let rows = rows_of(&tsv);
    assert_eq!(rows.len(), 3);
    let (packed, cuts) = n8_cuts();
    let circ = read_oracle_mpmct(&out.join("unpacked.mpmct1")).unwrap();
    assert_eq!(circ.header_n, 32);
    let mut off = 0usize;
    for (r, want) in rows.iter().zip([0usize, 2, 5]) {
        assert_eq!(r[0], want.to_string(), "row idx");
        assert_eq!(r[1], cuts[want].to_string(), "orig_lo");
        assert_eq!(r[2], cuts[want + 1].to_string(), "orig_hi");
        assert_eq!(r[11], off.to_string(), "syn_offset relative to this file");
        let syn: usize = r[4].parse().unwrap();
        // the segment at syn_offset IS that window (no reorder across windows):
        // it computes the packed window's function
        let seg = &circ.gates[off..off + syn];
        assert!(
            same_function(seg, &packed[cuts[want]..cuts[want + 1]], 77 + want as u64),
            "window {want}: segment != packed window"
        );
        off += syn;
    }
    assert_eq!(circ.gates.len(), off, "exactly the selected windows' gates");
    assert_eq!(kv(&report, "gates_out"), off.to_string());
    // the same windows synthesized in the plain run give the same rows (the
    // PRNG is keyed by the lattice index, not by the selection)
    let plain = rows_of(&fs::read_to_string(n8_plain().join("windows.tsv")).unwrap());
    for (r, want) in rows.iter().zip([0usize, 2, 5]) {
        let q = &plain[want];
        assert_eq!((&r[4], &r[9], &r[12]), (&q[4], &q[9], &q[12]), "window {want}");
    }
    // usage errors
    let code = |extra: &[&str]| n8_run(&scratch().join("x"), extra).status.code();
    assert_eq!(code(&["--windows", "73"]), Some(2));
    assert_eq!(code(&["--windows", "3:3"]), Some(2));
    assert_eq!(code(&["--windows", "1", "--refine"]), Some(2));
    assert_eq!(code(&["--max-rounds", "2"]), Some(2));
}

#[test]
fn canon_is_equivalent_idempotent_and_keeps_the_header() {
    let dir = scratch();
    let plain = n8_plain();
    let canon = |inp: &Path, out: &Path| unpack(&["canon", p(inp), p(out)]);
    // the plain run's circuit is already canonical: canon reproduces it byte for byte
    let o = canon(&plain.join("unpacked.mpmct1"), &dir.join("again.mpmct1"));
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(
        fs::read(plain.join("unpacked.mpmct1")).unwrap(),
        fs::read(dir.join("again.mpmct1")).unwrap()
    );
    assert!(!dir.join("again.mpmct1.tmp").exists());
    // a non-canonical concatenation (a --windows subset): canon(X) == X as a
    // function, is a fixpoint of the canonicalizer, keeps the header width
    let sel = dir.join("sel");
    let o = n8_run(&sel, &["--windows", "1:4"]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let o = canon(&sel.join("unpacked.mpmct1"), &dir.join("c.mpmct1"));
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(String::from_utf8_lossy(&o.stdout).contains("equivalent on 4096 random states"));
    let x = read_oracle_mpmct(&sel.join("unpacked.mpmct1")).unwrap();
    let c = read_oracle_mpmct(&dir.join("c.mpmct1")).unwrap();
    assert_eq!(c.header_n, x.header_n);
    assert_eq!(c.gates.len(), x.gates.len());
    assert!(same_function(&x.gates, &c.gates, 5));
    let mut again = c.gates.clone();
    local_mixing::canonicalization::gate_order::xgate_canonicalize(&mut again);
    assert_eq!(again, c.gates);
    // exit codes
    assert_eq!(unpack(&["canon", p(&dir.join("c.mpmct1"))]).status.code(), Some(2));
    assert_eq!(
        unpack(&["canon", p(&dir.join("missing")), p(&dir.join("y"))]).status.code(),
        Some(3)
    );
    let _ = fs::remove_dir_all(dir);
}

/// Bounds files as sorted interior cuts (both writers end lines with `\n`;
/// the packer's writer prints a lone `\n` for an empty list).
fn bounds_of(path: &Path) -> Vec<usize> {
    let mut b: Vec<usize> = fs::read_to_string(path)
        .unwrap()
        .split_whitespace()
        .map(|t| t.parse().unwrap())
        .collect();
    b.sort_unstable();
    b
}

#[test]
fn refine_with_forced_failures_splits_every_window_and_converges() {
    let pk = n8_packing();
    let out = scratch().join("refine_forced");
    let o = n8_run(&out, &["--refine", "--force-fail-rounds", "1"]);
    assert!(
        o.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    for f in [
        "unpacked.mpmct1",
        "windows.tsv",
        "rounds.tsv",
        "unpacking.report",
        "unpacking.log",
        "deliverable.tdp4",
        "bounds.final",
        "deliverable.0.tdp4",
        "bounds.0",
        "deliverable.1.tdp4",
        "bounds.1",
    ] {
        assert!(out.join(f).is_file(), "{f} missing");
        assert!(!out.join(format!("{f}.tmp")).exists(), "{f}.tmp left behind");
    }
    let report = fs::read_to_string(out.join("unpacking.report")).unwrap();
    assert_eq!(kv(&report, "refine"), "1");
    assert_eq!(kv(&report, "status"), "ok");
    assert_eq!(kv(&report, "windows_initial"), "73");
    assert_eq!(kv(&report, "verbatim"), "0");
    assert_eq!(kv(&report, "over_budget"), "0");
    assert_eq!(kv(&report, "param.force_fail_rounds"), "1");
    let rounds: usize = kv(&report, "rounds").parse().unwrap();
    let splits: usize = kv(&report, "splits").parse().unwrap();
    let final_n: usize = kv(&report, "windows_final").parse().unwrap();
    assert!(rounds >= 2, "rounds {rounds}");
    assert!(splits >= 73, "every window of round 0 was split: {splits}");
    assert_eq!(final_n, 73 + splits, "no 1-gate window on this fixture");
    assert_eq!(kv(&report, "windows"), final_n.to_string());
    assert_eq!(kv(&report, "synthesized"), final_n.to_string());
    assert_eq!(kv(&report, "gates_in"), "34825");
    assert!(kv(&report, "check.final_deliverable").starts_with("ok("));
    assert_eq!(
        kv(&report, "check.new_cuts"),
        format!("ok({splits} new cuts, {} unclean)", kv(&report, "unclean_cuts"))
    );
    assert_eq!(kv(&report, "check.equiv_canonical"), "ok(4096)");
    assert_eq!(kv(&report, "check.zero_slice"), "ok(4096)");
    // rounds.tsv: round 0 failed everything and split everything
    let rt = fs::read_to_string(out.join("rounds.tsv")).unwrap();
    assert_eq!(
        rt.lines().next().unwrap(),
        "round\tpending\tconverged\tfailed\tsplit\tseconds"
    );
    let r0: Vec<&str> = rt.lines().nth(1).unwrap().split('\t').collect();
    assert_eq!(&r0[..5], &["0", "73", "0", "73", "73"]);
    assert_eq!(rt.lines().count(), rounds + 1);
    // the per-round inputs: round 0 = the packer's deliverable and bounds
    assert_eq!(
        fs::read(out.join("deliverable.0.tdp4")).unwrap(),
        fs::read(pk.join("deliverable.tdp4")).unwrap()
    );
    assert_eq!(bounds_of(&out.join("bounds.0")), bounds_of(&pk.join("rep512.bounds")));
    assert_eq!(bounds_of(&out.join("bounds.1")).len(), 73 + 72);
    // the final deliverable: the refined lattice, decodable, zero slice ok
    let d = decode(&fs::read(out.join("deliverable.tdp4")).unwrap(), Format::Container).unwrap();
    assert_eq!(d.window_count(), final_n);
    assert_eq!(d.header.unwrap().window_count as usize, final_n);
    let bf = bounds_of(&out.join("bounds.final"));
    assert_eq!(bf.len(), final_n - 1);
    assert!(bf.windows(2).all(|w| w[0] < w[1]), "bounds strictly ascending");
    assert!(bf.iter().all(|&b| 0 < b && b < 34825));
    let c = read_g57_file(p(&fx("n8_L8/c.g57"))).unwrap();
    assert_eq!(check_zero_slice(&c, &d, 8, 1024, 11).unwrap(), 1024);
    equivalent(&out.join("unpacked.mpmct1"), &out.join("deliverable.tdp4"), 2);
    // windows.tsv: the reference's columns first, one row per final window,
    // contiguous, every one synthesized in a round >= 1
    let tsv = fs::read_to_string(out.join("windows.tsv")).unwrap();
    let header: Vec<&str> = tsv.lines().next().unwrap().split('\t').collect();
    assert_eq!(
        &header[..14],
        &[
            "idx", "lo", "hi", "round", "orig_gates", "syn_gates", "status", "synth_status",
            "seconds", "monomials", "changed_wires", "input_vars", "max_degree", "syn_offset"
        ]
    );
    let rows = rows_of(&tsv);
    assert_eq!(rows.len(), final_n);
    let mut prev_hi = 0usize;
    let mut off = 0usize;
    for (i, r) in rows.iter().enumerate() {
        assert_eq!(r[0], i.to_string());
        let lo: usize = r[1].parse().unwrap();
        let hi: usize = r[2].parse().unwrap();
        assert_eq!(lo, prev_hi, "contiguous lattice");
        assert!(hi > lo);
        prev_hi = hi;
        assert!(r[3].parse::<i64>().unwrap() >= 1, "round of row {i}: {}", r[3]);
        assert_eq!(r[4], (hi - lo).to_string());
        assert_eq!(r[6], "ok");
        assert_eq!(r[7], "converged");
        assert_eq!(r[13], off.to_string());
        off += r[5].parse::<usize>().unwrap();
    }
    assert_eq!(prev_hi, 34825);
    assert_eq!(kv(&report, "gates_out"), off.to_string());
    // a rerun removes the stale per-round files of the earlier run
    fs::write(out.join("deliverable.9.tdp4"), b"stale").unwrap();
    fs::write(out.join("bounds.9"), b"stale").unwrap();
    let o = n8_run(&out, &["--refine", "--max-rounds", "0"]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(!out.join("deliverable.9.tdp4").exists());
    assert!(!out.join("bounds.9").exists());
    let r2 = fs::read_to_string(out.join("unpacking.report")).unwrap();
    assert_eq!(kv(&r2, "rounds"), "1");
    assert_eq!(kv(&r2, "splits"), "0");
    assert_eq!(kv(&r2, "windows_final"), "73");
}

#[test]
fn refine_without_the_knob_matches_the_plain_run_unless_a_window_failed() {
    let pk = n8_packing();
    let plain = n8_plain();
    let out = scratch().join("refine");
    let o = n8_run(&out, &["--refine"]);
    assert!(
        o.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    let report = fs::read_to_string(out.join("unpacking.report")).unwrap();
    let plain_report = fs::read_to_string(plain.join("unpacking.report")).unwrap();
    let fallback: usize = kv(&plain_report, "fallback").parse().unwrap();
    let rounds: usize = kv(&report, "rounds").parse().unwrap();
    let splits: usize = kv(&report, "splits").parse().unwrap();
    let final_n: usize = kv(&report, "windows_final").parse().unwrap();
    assert_eq!(final_n, 73 + splits);
    assert_eq!(kv(&report, "verbatim"), "0");
    assert_eq!(kv(&report, "status"), "ok");
    assert!(!report.contains("param.force_fail_rounds"));
    equivalent(&out.join("unpacked.mpmct1"), &out.join("deliverable.tdp4"), 2);
    // A window that hits the 3 s deadline of the unoptimized test build has a
    // timing-dependent outcome (it may converge in one run and time out in the
    // other, under the parallel tests' load), so the run-to-run comparison is
    // exact only when neither run hit it.
    let dl_plain: usize = kv(&plain_report, "deadline_hits").parse().unwrap();
    let dl_refine: usize = kv(&report, "deadline_hits").parse().unwrap();
    let timing_free = dl_plain == 0 && dl_refine == 0;
    if fallback == 0 && timing_free {
        // every window converged in round 0: one round, the packer's lattice,
        // and byte for byte the plain run's circuit (same windows, same seed,
        // same canonical order)
        assert_eq!(rounds, 1);
        assert_eq!(splits, 0);
        assert_eq!(
            fs::read(out.join("unpacked.mpmct1")).unwrap(),
            fs::read(plain.join("unpacked.mpmct1")).unwrap()
        );
        assert_eq!(
            fs::read(out.join("deliverable.tdp4")).unwrap(),
            fs::read(pk.join("deliverable.tdp4")).unwrap()
        );
        assert_eq!(bounds_of(&out.join("bounds.final")), bounds_of(&pk.join("rep512.bounds")));
    } else if timing_free {
        // the plain run emitted the failed windows verbatim; the loop split
        // them instead, so the lattice grew and the circuits differ
        assert!(rounds >= 2, "rounds {rounds}");
        assert!(splits >= fallback, "splits {splits} < fallback {fallback}");
        assert_ne!(
            fs::read(out.join("unpacked.mpmct1")).unwrap(),
            fs::read(plain.join("unpacked.mpmct1")).unwrap()
        );
    } else {
        eprintln!(
            "deadline hits (plain {dl_plain}, refine {dl_refine}): the run-to-run comparison is timing-dependent; per-window checks only"
        );
    }
    // a window that converged within the deadline in both runs converged
    // identically (round 0, same lattice index, same seed, same gate count)
    let prow = rows_of(&fs::read_to_string(plain.join("windows.tsv")).unwrap());
    let rrow = rows_of(&fs::read_to_string(out.join("windows.tsv")).unwrap());
    let (_, cuts) = n8_cuts();
    let hit = |row: &Vec<String>| row.last().unwrap() == "1";
    let mut compared = 0usize;
    for (i, q) in prow.iter().enumerate() {
        if q[9] != "ok" || hit(q) {
            continue;
        }
        let Some(r) = rrow
            .iter()
            .find(|r| r[1] == cuts[i].to_string() && r[2] == cuts[i + 1].to_string())
        else {
            assert!(
                !timing_free,
                "window {i} converged in the plain run but was split here without a deadline hit"
            );
            continue;
        };
        assert_eq!(r[3], "0", "window {i}: round");
        assert_eq!(r[6], "ok", "window {i}: status");
        if !hit(r) {
            assert_eq!(r[5], q[4], "window {i}: syn_gates");
            compared += 1;
        }
    }
    assert!(compared > 0, "no window converged within the deadline in both runs");
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
