//! `tdp_unpacking`: the stage-7 production binary (docs/STAGE7_UNPACKING.md).
//!
//! It reads NO environment variables; threads are set by `--threads` only.
//! Exit codes are `Stage7Error::exit_code`.
//!
//! ```text
//! tdp_unpacking run --deliverable <deliverable.tdp4> --out-dir DIR
//!                   [--packed <rep.mpmct1> --bounds <rep.bounds>]
//!                   [--source-c <C.g57> --n N] [--seed S=1 | --seed-file PATH]
//!                   [--threads K] [--time-limit 30] [--restarts 16] [--beam 6]
//!                   [--no-fallback] [--no-postopt] [--label tdp]
//!     Resynthesizes every window of the deliverable from its ANF alone,
//!     concatenates the windows, canonicalizes the gate order of the whole
//!     circuit and writes `unpacked.mpmct1`, `windows.tsv`,
//!     `unpacking.report` and `unpacking.log` into DIR (via *.tmp, after
//!     every check passed). With `--packed/--bounds` (stage 6's repaired
//!     circuit; its windows index the deliverable's) a window whose
//!     synthesis fails is emitted verbatim after its ANF is checked against
//!     the deliverable (`status=fallback_used`); `--no-fallback` makes such a
//!     window exit 4. With `--source-c/--n` the final circuit is also checked
//!     on the zero slice against C.
//! tdp_unpacking selftest [--n 8] [--m 12] [--trials 40] [--seed 7] [--threads K]
//!     Random small circuits: compose -> deliverable -> synthesize -> verify
//!     through the same pipeline (no fallback).
//! ```
use crate::circuit::formats::read_g57_file;
use crate::circuit::xgate::XGate;
use crate::stages::packing::oracle_io::{
    read_bounds, read_oracle_mpmct, window_cuts, write_pack_mpmct,
};
use crate::stages::packing::serialize::{Deliverable, Format, decode, encode_body};
use crate::stages::unpacking::run::{EXACT_BUDGETS, quantile, windows_tsv};
use crate::stages::unpacking::synth::TAG_NAMES;
use crate::stages::unpacking::types::MAX_STATE_WIRES;
use crate::stages::unpacking::verify::{check_equiv, check_zero_slice_gates};
use crate::stages::unpacking::{
    PackedWindows, Stage7Error, UnpackParams, UnpackSummary, WindowResult, concatenate,
    unpack_windows,
};
use std::fmt::Write as _;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

pub fn run() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match dispatch(&args) {
        Ok(()) => 0,
        Err(e) => {
            let code = e.exit_code();
            eprintln!("tdp_unpacking: {e}");
            eprintln!(
                "tdp_unpacking: exit {code} ({})",
                Stage7Error::meaning(code)
            );
            code
        }
    };
    std::process::exit(code);
}

// ---------------------------------------------------------------- argv

const SWITCHES: [&str; 4] = ["--no-fallback", "--no-postopt", "--help", "--stats"];

struct Args {
    pos: Vec<String>,
    opts: Vec<(String, Option<String>)>,
}

impl Args {
    fn parse(a: &[String]) -> Result<Args, Stage7Error> {
        let mut pos = Vec::new();
        let mut opts = Vec::new();
        let mut i = 0;
        while i < a.len() {
            let s = &a[i];
            if let Some(name) = s.strip_prefix("--") {
                if SWITCHES.contains(&s.as_str()) {
                    opts.push((name.to_string(), None));
                } else {
                    let v = a
                        .get(i + 1)
                        .ok_or_else(|| Stage7Error::Usage(format!("{s} needs a value")))?;
                    opts.push((name.to_string(), Some(v.clone())));
                    i += 1;
                }
            } else {
                pos.push(s.clone());
            }
            i += 1;
        }
        Ok(Args { pos, opts })
    }

    fn check_known(&self, known: &[&str]) -> Result<(), Stage7Error> {
        for (k, _) in &self.opts {
            if !known.contains(&k.as_str()) {
                return Err(Stage7Error::Usage(format!("unknown option --{k}")));
            }
        }
        Ok(())
    }

    fn get(&self, k: &str) -> Option<&str> {
        self.opts
            .iter()
            .rev()
            .find(|(n, _)| n == k)
            .and_then(|(_, v)| v.as_deref())
    }

    fn flag(&self, k: &str) -> bool {
        self.opts.iter().any(|(n, _)| n == k)
    }

    fn num(&self, k: &str) -> Result<Option<u64>, Stage7Error> {
        self.get(k).map(|v| parse_u64(v, k)).transpose()
    }

    fn usize_or(&self, k: &str, dflt: usize) -> Result<usize, Stage7Error> {
        Ok(self.num(k)?.map(|v| v as usize).unwrap_or(dflt))
    }

    fn req(&self, k: &str) -> Result<&str, Stage7Error> {
        self.get(k)
            .ok_or_else(|| Stage7Error::Usage(format!("--{k} is required")))
    }
}

fn parse_u64(v: &str, what: &str) -> Result<u64, Stage7Error> {
    if v.is_empty() || v.len() > 19 || !v.bytes().all(|b| b.is_ascii_digit()) {
        return Err(Stage7Error::Usage(format!(
            "{what}: {v:?} is not a nonnegative integer"
        )));
    }
    v.parse()
        .map_err(|_| Stage7Error::Usage(format!("{what}: {v:?} out of range")))
}

fn setup_threads(a: &Args) -> Result<usize, Stage7Error> {
    if let Some(t) = a.num("threads")? {
        if t == 0 || t > 4096 {
            return Err(Stage7Error::Usage("--threads must be in 1..=4096".into()));
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(t as usize)
            .build_global()
            .map_err(|e| Stage7Error::Internal(format!("thread pool: {e}")))?;
    }
    Ok(rayon::current_num_threads())
}

/// The seed and where it came from (`argv` or `file`); the value is never
/// printed.
fn seed_of(a: &Args) -> Result<(u64, &'static str), Stage7Error> {
    match (a.get("seed"), a.get("seed-file")) {
        (Some(_), Some(_)) => Err(Stage7Error::Usage(
            "--seed and --seed-file are mutually exclusive".into(),
        )),
        (Some(s), None) => Ok((parse_u64(s, "--seed")?, "argv")),
        (None, Some(p)) => Ok((
            crate::programs::read_seed_file(p).map_err(Stage7Error::Usage)?,
            "file",
        )),
        (None, None) => Ok((1, "default")),
    }
}

fn params_of(a: &Args) -> Result<UnpackParams, Stage7Error> {
    let mut p = UnpackParams::default();
    if let Some(t) = a.get("time-limit") {
        p.synth.time_limit = t
            .parse::<f64>()
            .map_err(|_| Stage7Error::Usage(format!("--time-limit {t:?}: not a number")))?;
    }
    p.synth.restarts = a.usize_or("restarts", p.synth.restarts)?;
    p.synth.beam = a.usize_or("beam", p.synth.beam)?;
    p.postopt = !a.flag("no-postopt");
    p.allow_fallback = !a.flag("no-fallback");
    p.validate().map_err(Stage7Error::Usage)?;
    Ok(p)
}

fn dispatch(argv: &[String]) -> Result<(), Stage7Error> {
    let (cmd, rest) = argv.split_first().ok_or_else(|| {
        Stage7Error::Usage("tdp_unpacking run|selftest ... (see the module docs)".into())
    })?;
    let a = Args::parse(rest)?;
    match cmd.as_str() {
        "run" => cmd_run(&a),
        "selftest" => cmd_selftest(&a),
        other => Err(Stage7Error::Usage(format!("unknown command {other:?}"))),
    }
}

// ---------------------------------------------------------------- outputs

/// A private output file written to `<path>.tmp` and renamed on `commit`.
struct Staged {
    path: PathBuf,
    tmp: PathBuf,
}

impl Staged {
    fn create(path: &Path) -> Result<(Staged, BufWriter<File>), Stage7Error> {
        let mut tmp = path.as_os_str().to_owned();
        tmp.push(".tmp");
        let tmp = PathBuf::from(tmp);
        let mut o = OpenOptions::new();
        o.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            o.mode(0o600);
        }
        let f = o.open(&tmp).map_err(|e| Stage7Error::io(&tmp, e))?;
        Ok((
            Staged {
                path: path.to_path_buf(),
                tmp,
            },
            BufWriter::with_capacity(1 << 20, f),
        ))
    }

    fn finish(w: BufWriter<File>, tmp: &Path) -> Result<(), Stage7Error> {
        let f = w
            .into_inner()
            .map_err(|e| Stage7Error::io(tmp, e.into_error()))?;
        f.sync_all().map_err(|e| Stage7Error::io(tmp, e))
    }

    fn commit(self) -> Result<(), Stage7Error> {
        fs::rename(&self.tmp, &self.path).map_err(|e| Stage7Error::io(&self.path, e))
    }

    fn abandon(&self) {
        let _ = fs::remove_file(&self.tmp);
    }
}

type FileBody<'a> = Box<dyn FnOnce(&mut BufWriter<File>) -> io::Result<()> + 'a>;

/// Write each (path, writer) via *.tmp, then rename them in order. A write
/// failure commits nothing; a rename failure leaves the earlier renames in
/// place and removes the remaining *.tmp files.
fn write_all_staged(files: Vec<(PathBuf, FileBody<'_>)>) -> Result<(), Stage7Error> {
    let mut staged: Vec<Staged> = Vec::new();
    let r = (|| -> Result<(), Stage7Error> {
        for (path, f) in files {
            let (st, mut w) = Staged::create(&path)?;
            let tmp = st.tmp.clone();
            staged.push(st);
            f(&mut w).map_err(|e| Stage7Error::io(&tmp, e))?;
            Staged::finish(w, &tmp)?;
        }
        Ok(())
    })();
    if let Err(e) = r {
        for s in &staged {
            s.abandon();
        }
        return Err(e);
    }
    let mut staged = staged.into_iter();
    while let Some(s) = staged.next() {
        if let Err(e) = s.commit() {
            for rest in staged {
                rest.abandon();
            }
            return Err(e);
        }
    }
    Ok(())
}

fn private_dir(dir: &Path) -> Result<(), Stage7Error> {
    if !dir.exists() {
        fs::create_dir_all(dir).map_err(|e| Stage7Error::io(dir, e))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(dir, fs::Permissions::from_mode(0o700))
                .map_err(|e| Stage7Error::io(dir, e))?;
        }
    } else if !dir.is_dir() {
        return Err(Stage7Error::Usage(format!(
            "--out-dir {} is not a directory",
            dir.display()
        )));
    }
    Ok(())
}

pub const OUTPUT_NAMES: [&str; 4] = [
    "unpacked.mpmct1",
    "windows.tsv",
    "unpacking.report",
    "unpacking.log",
];

/// Removes every output of an earlier run from `out_dir` (and its *.tmp).
fn remove_stale_outputs(out_dir: &Path) -> Result<(), Stage7Error> {
    for name in OUTPUT_NAMES {
        for suffix in ["", ".tmp"] {
            let path = out_dir.join(format!("{name}{suffix}"));
            match fs::remove_file(&path) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(e) => return Err(Stage7Error::io(&path, e)),
            }
        }
    }
    Ok(())
}

/// Writes every line to stdout and keeps a copy (the log file).
struct Tee {
    buf: Vec<u8>,
}

impl Tee {
    fn line(&mut self, s: &str) {
        let stdout = io::stdout();
        let mut o = stdout.lock();
        let _ = o.write_all(s.as_bytes());
        let _ = o.write_all(b"\n");
        let _ = o.flush();
        self.buf.extend_from_slice(s.as_bytes());
        self.buf.push(b'\n');
    }
}

// ---------------------------------------------------------------- run

fn xxh3_file(path: &Path) -> Option<u128> {
    let bytes = fs::read(path).ok()?;
    Some(xxhash_rust::xxh3::xxh3_128(&bytes))
}

struct Checks {
    exact_windows: usize,
    equiv_concat: usize,
    equiv_canonical: usize,
    zero_slice: Option<usize>,
}

#[allow(clippy::too_many_arguments)]
fn unpacking_report(
    status: &str,
    d: &Deliverable,
    deliverable_xxh3: u128,
    n: Option<usize>,
    p: &UnpackParams,
    seed_source: &str,
    threads: usize,
    s: &UnpackSummary,
    canonical_gates: usize,
    checks: &Checks,
    times: [f64; 4],
) -> String {
    let mut out = String::new();
    let mut kv = |k: &str, v: String| {
        let _ = writeln!(out, "{k}={v}");
    };
    let opt = |v: Option<usize>| v.map_or("unknown".to_string(), |x| x.to_string());
    kv("unpacking_report", "1".into());
    kv("status", status.into());
    kv(
        "tool",
        format!("tdp_unpacking {}", env!("CARGO_PKG_VERSION")),
    );
    kv(
        "binary_xxh3",
        std::env::current_exe()
            .ok()
            .and_then(|p| xxh3_file(&p))
            .map(|h| format!("{h:032x}"))
            .unwrap_or_else(|| "unknown".into()),
    );
    kv("deliverable_xxh3", format!("{deliverable_xxh3:032x}"));
    kv("state_wires", d.state_wires.to_string());
    kv("source_wires", d.source_wires.to_string());
    kv("n", opt(n));
    kv("param.seed_source", seed_source.into());
    kv("param.time_limit", format!("{}", p.synth.time_limit));
    kv("param.restarts", p.synth.restarts.to_string());
    kv("param.beam", p.synth.beam.to_string());
    kv("param.back_k", p.synth.back_k.to_string());
    kv("param.back_look", p.synth.back_look.to_string());
    kv("param.back_prefer", p.synth.back_prefer.to_string());
    kv("param.w0", p.synth.w0.to_string());
    kv("param.w1", p.synth.w1.to_string());
    kv("param.deg_weight", p.synth.deg_weight.to_string());
    kv("param.exact_k", p.synth.exact_k.to_string());
    kv("param.step_factor", p.synth.step_factor.to_string());
    kv("param.stall_limit", p.synth.stall_limit.to_string());
    kv("param.macro_batch", p.synth.macro_batch.to_string());
    kv("param.macros", p.synth.use_macros.to_string());
    kv("param.postopt", p.postopt.to_string());
    kv("param.fallback", p.allow_fallback.to_string());
    kv("param.ancilla_pool", p.ancilla_pool.to_string());
    kv("param.verify_seed", p.verify_seed.to_string());
    kv("param.equiv_batches", p.equiv_batches.to_string());
    kv(
        "param.exact_budgets",
        EXACT_BUDGETS
            .iter()
            .map(|b| b.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    kv("param.threads", threads.to_string());
    kv("windows", s.windows.to_string());
    kv("synthesized", s.synthesized.to_string());
    kv("fallback", s.fallback.to_string());
    kv("deadline_hits", s.deadline_hits.to_string());
    kv("verify_fail", s.verify_fail.to_string());
    kv("exact_overflow", s.exact_overflow.to_string());
    kv("exact_retried", s.exact_retried.to_string());
    kv("gates_in", opt(s.gates_in));
    kv("gates_out", s.gates_out.to_string());
    kv("gates_pre_postopt", s.gates_pre_postopt.to_string());
    kv("gates_canonical", canonical_gates.to_string());
    kv(
        "ratio",
        s.ratio()
            .map_or("unknown".to_string(), |r| format!("{r:.4}")),
    );
    for (name, count) in TAG_NAMES.iter().zip(&s.tags) {
        kv(&format!("gates_by_move.{name}"), count.to_string());
    }
    for (k, v) in &s.status_counts {
        kv(&format!("synth_status.{k}"), v.to_string());
    }
    for (name, q) in [("p10", 0.1), ("p50", 0.5), ("p90", 0.9), ("max", 1.0)] {
        kv(
            &format!("seconds.{name}"),
            format!("{:.3}", quantile(&s.secs, q)),
        );
    }
    kv("seconds.sum", format!("{:.1}", s.secs.iter().sum::<f64>()));
    for (name, q) in [("min", 0.0), ("p50", 0.5), ("max", 1.0)] {
        kv(
            &format!("restarts_used.{name}"),
            format!("{:.0}", quantile(&s.restarts_used, q)),
        );
    }
    kv(
        "check.windows_exact",
        format!("ok({})", checks.exact_windows),
    );
    kv(
        "check.windows_sampled",
        format!("ok({})", checks.exact_windows),
    );
    kv("check.equiv_concat", format!("ok({})", checks.equiv_concat));
    kv(
        "check.equiv_canonical",
        format!("ok({})", checks.equiv_canonical),
    );
    kv(
        "check.zero_slice",
        checks
            .zero_slice
            .map_or("skipped".to_string(), |n| format!("ok({n})")),
    );
    kv("time.load_s", format!("{:.1}", times[0]));
    kv("time.synth_s", format!("{:.1}", times[1]));
    kv("time.canonicalize_s", format!("{:.1}", times[2]));
    kv("time.verify_s", format!("{:.1}", times[3]));
    out
}

const RUN_OPTIONS: [&str; 15] = [
    "deliverable",
    "out-dir",
    "packed",
    "bounds",
    "source-c",
    "n",
    "seed",
    "seed-file",
    "threads",
    "time-limit",
    "restarts",
    "beam",
    "no-fallback",
    "no-postopt",
    "label",
];

fn cmd_run(a: &Args) -> Result<(), Stage7Error> {
    a.check_known(&RUN_OPTIONS)?;
    if !a.pos.is_empty() {
        return Err(Stage7Error::Usage(
            "run --deliverable F --out-dir DIR [--packed P --bounds B] [--source-c C --n N] [...]"
                .into(),
        ));
    }
    let threads = setup_threads(a)?;
    let mut params = params_of(a)?;
    let (seed, seed_source) = seed_of(a)?;
    params.seed = seed;
    let label = a.get("label").unwrap_or("tdp").to_string();
    if label.is_empty() || label.contains(['/', '\n']) {
        return Err(Stage7Error::Usage(format!("--label {label:?}")));
    }
    let out_dir = PathBuf::from(a.req("out-dir")?);
    if a.get("packed").is_some() != a.get("bounds").is_some() {
        return Err(Stage7Error::Usage(
            "--packed and --bounds go together".into(),
        ));
    }
    if a.get("source-c").is_some() != a.get("n").is_some() {
        return Err(Stage7Error::Usage("--source-c and --n go together".into()));
    }
    let t0 = Instant::now();

    // ---- inputs
    let dpath = a.req("deliverable")?;
    let bytes = fs::read(dpath).map_err(|e| Stage7Error::Input(format!("{dpath}: {e}")))?;
    let deliverable_xxh3 = xxhash_rust::xxh3::xxh3_128(&bytes);
    let d = decode(&bytes, Format::Container)
        .map_err(|e| Stage7Error::Input(format!("{dpath}: {e}")))?;
    drop(bytes);
    let nw = d.state_wires as usize;
    if nw > MAX_STATE_WIRES {
        return Err(Stage7Error::Usage(format!(
            "{dpath}: {nw} state wires; stage 7 supports at most {MAX_STATE_WIRES}"
        )));
    }
    let packed_circ = match a.get("packed") {
        Some(p) => {
            let c = read_oracle_mpmct(Path::new(p))?;
            let b = read_bounds(Path::new(a.req("bounds")?))?;
            let cuts = window_cuts(&b, c.gates.len());
            Some((c, cuts))
        }
        None => None,
    };
    let packed = packed_circ.as_ref().map(|(c, cuts)| PackedWindows {
        gates: &c.gates,
        cuts: cuts.clone(),
    });
    let n = a.num("n")?.map(|v| v as usize);
    let source_c: Option<Vec<XGate>> = match (a.get("source-c"), n) {
        (Some(path), Some(n)) => {
            if n == 0 || 2 * n > nw || n as u32 != d.source_wires {
                return Err(Stage7Error::Usage(format!(
                    "--n {n}: the deliverable has {} source wires in a {nw}-wire state",
                    d.source_wires
                )));
            }
            let c = read_g57_file(path).map_err(|e| Stage7Error::Input(format!("{path}: {e}")))?;
            if let Some(w) = c.iter().map(|g| g.max_wire() as usize).max()
                && w >= n
            {
                return Err(Stage7Error::Input(format!(
                    "{path}: C uses wire {w}, beyond n = {n}"
                )));
            }
            Some(c)
        }
        _ => None,
    };
    private_dir(&out_dir)?;
    remove_stale_outputs(&out_dir)?;
    let t_load = t0.elapsed().as_secs_f64();

    // ---- synthesis
    let log = Mutex::new(Tee { buf: Vec::new() });
    let say = |s: &str| log.lock().unwrap().line(s);
    say(&format!(
        "{label}: {} windows, {nw} state wires, {} monomials; packed {}; seed ({seed_source}); threads {threads}; \
         time-limit {}s restarts {} beam {} postopt {} fallback {}",
        d.window_count(),
        d.monomials,
        packed
            .as_ref()
            .map_or("none".to_string(), |p| format!("{} gates", p.gates.len())),
        params.synth.time_limit,
        params.synth.restarts,
        params.synth.beam,
        params.postopt,
        params.allow_fallback,
    ));
    let total = d.window_count();
    let progress = |done: usize, r: &WindowResult| {
        if done.is_multiple_of(50)
            || done == total
            || r.status != crate::stages::unpacking::WinStatus::Ok
        {
            say(&format!(
                "[{done}/{total}] window {} {} -> {} ({}, {}, {:.2}s) {}",
                r.idx,
                r.orig_gates().map_or("?".to_string(), |g| g.to_string()),
                r.gates.len(),
                r.status.name(),
                r.synth_status,
                r.secs,
                r.note
            ));
        }
    };
    let results = unpack_windows(&d, packed.as_ref(), &params, &progress)?;
    let summary = UnpackSummary::of(&results);
    let t_synth = t0.elapsed().as_secs_f64() - t_load;
    let (concat, offsets) = concatenate(&results);
    let tsv = windows_tsv(&results, &offsets);
    drop(results);

    // ---- canonical order + checks
    let t_c = Instant::now();
    let equiv_concat = check_equiv(
        &concat,
        &d,
        params.verify_seed,
        params.equiv_batches,
        "concatenation",
    )?;
    let mut gates = concat;
    crate::canonicalization::gate_order::xgate_canonicalize(&mut gates);
    let t_canon = t_c.elapsed().as_secs_f64();
    let t_v = Instant::now();
    let equiv_canonical = check_equiv(
        &gates,
        &d,
        params.verify_seed ^ 0xC0FFEE,
        params.equiv_batches,
        "canonical circuit",
    )?;
    let zero_slice = match (&source_c, n) {
        (Some(c), Some(n)) => Some(check_zero_slice_gates(
            c,
            &gates,
            n,
            nw,
            params.zero_slice_samples,
            params.verify_seed ^ 0x5A5A,
        )?),
        _ => None,
    };
    let t_verify = t_v.elapsed().as_secs_f64();
    let checks = Checks {
        exact_windows: summary.windows,
        equiv_concat,
        equiv_canonical,
        zero_slice,
    };
    let status = if summary.fallback > 0 {
        "fallback_used"
    } else {
        "ok"
    };
    say(&format!(
        "{label}: {} windows: {} synthesized, {} fallback, {} deadline hits; gates {} -> {} (ratio {}); \
         canonicalized {} gates in {t_canon:.1}s; checks: windows exact+sampled ok, equiv concat ok({equiv_concat}), \
         equiv canonical ok({equiv_canonical}), zero slice {}; status={status} ({:.1}s)",
        summary.windows,
        summary.synthesized,
        summary.fallback,
        summary.deadline_hits,
        summary.gates_in.map_or("?".to_string(), |g| g.to_string()),
        summary.gates_out,
        summary
            .ratio()
            .map_or("?".to_string(), |r| format!("{r:.4}")),
        gates.len(),
        zero_slice.map_or("skipped".to_string(), |s| format!("ok({s})")),
        t0.elapsed().as_secs_f64(),
    ));
    let report = unpacking_report(
        status,
        &d,
        deliverable_xxh3,
        n,
        &params,
        seed_source,
        threads,
        &summary,
        gates.len(),
        &checks,
        [t_load, t_synth, t_canon, t_verify],
    );
    let log_bytes = std::mem::take(&mut log.lock().unwrap().buf);
    // the circuit first, the report (the completion record) last
    write_all_staged(vec![
        (
            out_dir.join("unpacked.mpmct1"),
            Box::new(|w| write_pack_mpmct(w, nw, &gates)),
        ),
        (
            out_dir.join("windows.tsv"),
            Box::new(|w| w.write_all(tsv.as_bytes())),
        ),
        (
            out_dir.join("unpacking.log"),
            Box::new(|w| w.write_all(&log_bytes)),
        ),
        (
            out_dir.join("unpacking.report"),
            Box::new(|w| w.write_all(report.as_bytes())),
        ),
    ])?;
    Ok(())
}

// ---------------------------------------------------------------- selftest

/// Random circuits of `m` gates (k in {1, 2}, random polarities and
/// complement) on `n` wires, each a window of one deliverable.
fn random_windows(n: usize, m: usize, trials: usize, seed: u64) -> Vec<XGate> {
    use crate::stages::unpacking::mono::Rng;
    let mut rng = Rng::new(seed);
    let mut gates = Vec::with_capacity(m * trials);
    for _ in 0..trials {
        for _ in 0..m {
            let t = rng.below(n as u64) as u16;
            let k = 1 + rng.below(2) as usize;
            let mut a = rng.below(n as u64) as u16;
            while a == t {
                a = rng.below(n as u64) as u16;
            }
            let mut b = rng.below(n as u64) as u16;
            while b == t || b == a {
                b = rng.below(n as u64) as u16;
            }
            let mut lits = vec![(a, rng.below(2) == 1)];
            if k == 2 {
                lits.push((b, rng.below(2) == 1));
            }
            let mut g = XGate::conj(t, lits).expect("distinct wires");
            g.comp = rng.below(2) == 1;
            gates.push(g);
        }
    }
    gates
}

fn cmd_selftest(a: &Args) -> Result<(), Stage7Error> {
    a.check_known(&[
        "n",
        "m",
        "trials",
        "seed",
        "threads",
        "time-limit",
        "restarts",
    ])?;
    setup_threads(a)?;
    let n = a.usize_or("n", 8)?;
    let m = a.usize_or("m", 12)?;
    let trials = a.usize_or("trials", 40)?;
    let seed = a.num("seed")?.unwrap_or(7);
    if !(3..=MAX_STATE_WIRES).contains(&n) || m == 0 || trials == 0 {
        return Err(Stage7Error::Usage(
            "selftest: n in 3..=1024, m >= 1, trials >= 1".into(),
        ));
    }
    let mut params = UnpackParams::default();
    params.synth.time_limit = a
        .get("time-limit")
        .map(|t| t.parse::<f64>().unwrap_or(5.0))
        .unwrap_or(5.0);
    params.synth.restarts = a.usize_or("restarts", params.synth.restarts)?;
    params.allow_fallback = false;
    params.seed = seed;
    let gates = random_windows(n, m, trials, seed);
    let bounds: Vec<i64> = (1..trials).map(|i| (i * m) as i64).collect();
    let mut body = Vec::new();
    encode_body(&gates, &bounds, gates.len(), n, 8, &mut body)?;
    let d = decode(
        &body,
        Format::Raw {
            source_wires: 0,
            state_wires: n as u32,
        },
    )?;
    let t0 = Instant::now();
    let results = unpack_windows(&d, None, &params, &|_, _| {})?;
    let s = UnpackSummary::of(&results);
    let (concat, _) = concatenate(&results);
    check_equiv(&concat, &d, 99, 2, "selftest concatenation")?;
    let mut canon = concat;
    crate::canonicalization::gate_order::xgate_canonicalize(&mut canon);
    check_equiv(&canon, &d, 98, 2, "selftest canonical")?;
    println!(
        "selftest n={n} m={m} trials={trials}: {} windows synthesized, {} fallback, {} deadline hits; \
         gates orig {} syn {} canonical {}; {:.2}s",
        s.synthesized,
        s.fallback,
        s.deadline_hits,
        gates.len(),
        s.gates_out,
        canon.len(),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}
