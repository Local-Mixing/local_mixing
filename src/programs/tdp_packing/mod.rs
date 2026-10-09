//! `tdp_packing`: the stage-4 production binary (docs/STAGE4_RUST_DESIGN.md §5).
//!
//! It reads NO environment variables; threads are set by `--threads` only.
//! Exit codes are `Stage4Error::exit_code` (§5.4).
//!
//! Implemented so far (the ANF / deliverable phase):
//!
//! ```text
//! tdp_packing serialize <rep.mpmct1> <rep.bounds> <out.bin> <label>
//!                       [--container <out.tdp4> --source-wires N] [--no-checks]
//!                       [--zero-slice-c <C.g57> --source-wires N] [--threads K]
//!     anf_dump_eval.py parity mode: writes the raw body; line 1 of stdout is
//!     DE's line 1. Then (unless --no-checks) decodes the written body and
//!     checks it exactly against window_anf and the gates, and runs DE's
//!     1024-lane evaluation check (seed 1). Outputs are written to *.tmp and
//!     renamed only after every check passed.
//! tdp_packing anf-report <C.mpmct1> <bounds> [budget=8192] [label=anf] [check_every=500]
//!                       [--anf-cap-semantics cpython39|canonical] [--stats] [--threads K]
//!     anf_compose.py parity mode (stdout identical).
//! tdp_packing window-anf <C.mpmct1> <bounds> <budget> [--threads K]
//!     per-window decisions and ANF digests under both cap semantics
//!     (the fixture format of tests/fixtures/stage4/anf_windows.py).
//! tdp_packing decode <file> (--format raw --n N [--state-wires M] | --format container) [--stats]
//! tdp_packing eval   <file> (--format …) (--x HEX | --state HEX)
//! tdp_packing container <body.bin> <out.tdp4> --source-wires N --state-wires M
//! tdp_packing verify-equiv <A.mpmct1> <B.mpmct1> [--seed 77]
//! tdp_packing pack-replay <packed.mpmct1> <packed.bounds> [b_mono=24576]
//!                       [--anf-cap-semantics …] [--threads K]
//!     replays the packer's incremental window ANF (WindowState, product cap
//!     8·b_mono) over a packed circuit's windows: compose aborts (each must be
//!     a window's last gate), order-dependent aborts, ambiguous products.
//!
//! tdp_packing pack <A> <B> [W_cap=64] [G=64] [B_mono=3072] [span_cap=3000]
//!                       [label=pack3-G{G}] [out.mpmct1]
//!                       [--order-out P] [--anf-cap-semantics …] [--stats] [--threads K]
//!     window_pack3.py parity mode (the argv and its defaults; production is
//!     `512 512 24576 3000`): stdout is the oracle's log; `out.mpmct1` and
//!     `out.mpmct1.bounds` are written (via *.tmp) only after the oracle's
//!     VERIFY and the exact reorder certificate pass. `--order-out` writes
//!     the emitted order as WP's PACK_DUMP_ORDER does.
//!
//!
//! tdp_packing repair --packed <P.mpmct1> [--bounds <P.bounds>] --source-c <C.g57> --n N
//!                       --out-dir DIR [--g 512] [--split-budget 65536]
//!                       [--fix-iterations 4] [--fallback-iterations 4]
//!                       [--anf-cap-semantics …] [--allow-floor-miss] [--allow-over-budget]
//!                       [--no-oracle-files] [--stats] [--threads K]
//!     repair.sh parity mode on a packed circuit and its UNREPAIRED bounds
//!     (default `<P>.bounds`): cancel, fix loop, merge/split fallback loop.
//!     Writes `rep<G>.mpmct1`, `rep<G>.bounds`, the last in-loop scan as
//!     `rep<G>.{exposure,allcuts,wit}.tsv` + `rep<G>.cview.log` (unless
//!     --no-oracle-files) and `repair<G>.log` (also on stdout) into DIR, all
//!     via *.tmp after the exact repair certificate and both sampled
//!     equivalence checks passed. Then the final tests: a visible non-output
//!     row exits 5, a window failing the split's budget test exits 6 (the
//!     outputs are intermediates and stay written); `--allow-*` turns either
//!     into exit 0 with `status=floor_miss|over_budget` on stderr.
//! tdp_packing check-final --rep <R.mpmct1> --bounds <R.bounds> --source-c <C.g57> --n N
//!                       [--split-budget 65536] [--anf-cap-semantics …]
//!                       [--allow-floor-miss] [--allow-over-budget] [--threads K]
//!     the final tests on any repaired circuit (oracle or Rust).
//! ```
//!
//! `run` and `census` belong to later phases and exit 2.
use crate::circuit::formats::read_g57_file;
use crate::circuit::xgate::XGate;
use crate::stages::packing::anf::{MulStats, window_anf};
use crate::stages::packing::census::anf_size_report;
use crate::stages::packing::mono::{MonoWidthFn, with_mono_width};
use crate::stages::packing::oracle_io::{
    read_bounds, read_oracle_mpmct, window_cuts, write_bounds, write_pack_mpmct, write_repair_mpmct,
};
use crate::stages::packing::repair::split::FitCache;
use crate::stages::packing::repair::{CView, RepairReport, anf_width, final_tests, repair};
use crate::stages::packing::scheduler::pack;
use crate::stages::packing::serialize::{
    ContainerHeader, Deliverable, Format, IO_LAYOUT_CLASSIC_ZERO_SLICE, decode, encode_body,
};
use crate::stages::packing::types::{CapSemantics, PackParams, RepairParams, Stage4Error};
use crate::stages::packing::verify::{
    check_deliverable, check_windows_anf, check_zero_slice, verify_equiv,
};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

mod run;

/// Default number of windows composed concurrently (§4.1 `max_concurrent_anf`).
const MAX_CONCURRENT_ANF: usize = 8;
/// `anf_dump_eval.py`'s evaluation seed.
const DE_EVAL_SEED: u64 = 1;
/// Seed of the per-window sampled ANF-vs-gates check (no oracle counterpart).
const WINDOW_CHECK_SEED: u64 = 20_261_008;
const EXHAUSTIVE_SUPPORT_MAX: usize = 16;
const ZERO_SLICE_SAMPLES: usize = 4096;
const ZERO_SLICE_SEED: u64 = 4096;

pub fn run() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match dispatch(&args) {
        Ok(()) => 0,
        Err(e) => {
            let code = e.exit_code();
            eprintln!("tdp_packing: {e}");
            eprintln!("tdp_packing: exit {code} ({})", Stage4Error::meaning(code));
            code
        }
    };
    std::process::exit(code);
}

/// Split argv into positionals and `--flag [value]` options.
struct Args {
    pos: Vec<String>,
    opts: Vec<(String, Option<String>)>,
}

const SWITCHES: [&str; 6] = [
    "--no-checks",
    "--stats",
    "--help",
    "--allow-floor-miss",
    "--allow-over-budget",
    "--no-oracle-files",
];

impl Args {
    fn parse(a: &[String]) -> Result<Args, Stage4Error> {
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
                        .ok_or_else(|| Stage4Error::Usage(format!("{s} needs a value")))?;
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

    fn check_known(&self, known: &[&str]) -> Result<(), Stage4Error> {
        for (k, _) in &self.opts {
            if !known.contains(&k.as_str()) {
                return Err(Stage4Error::Usage(format!("unknown option --{k}")));
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

    fn num(&self, k: &str) -> Result<Option<u64>, Stage4Error> {
        self.get(k).map(|v| parse_u64(v, k)).transpose()
    }
}

fn parse_u64(v: &str, what: &str) -> Result<u64, Stage4Error> {
    if v.is_empty() || !v.bytes().all(|b| b.is_ascii_digit()) {
        return Err(Stage4Error::Usage(format!(
            "{what}: {v:?} is not a nonnegative integer"
        )));
    }
    v.parse()
        .map_err(|_| Stage4Error::Usage(format!("{what}: {v:?} out of range")))
}

fn setup_threads(a: &Args) -> Result<(), Stage4Error> {
    if let Some(t) = a.num("threads")? {
        if t == 0 {
            return Err(Stage4Error::Usage("--threads must be at least 1".into()));
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(t as usize)
            .build_global()
            .map_err(|e| Stage4Error::Internal(format!("thread pool: {e}")))?;
    }
    Ok(())
}

fn cap_semantics(a: &Args) -> Result<CapSemantics, Stage4Error> {
    match a.get("anf-cap-semantics") {
        None => Ok(CapSemantics::CPython39),
        Some(s) => CapSemantics::parse(s).ok_or_else(|| {
            Stage4Error::Usage(format!("--anf-cap-semantics {s:?}: cpython39|canonical"))
        }),
    }
}

fn dispatch(argv: &[String]) -> Result<(), Stage4Error> {
    let (cmd, rest) = argv.split_first().ok_or_else(|| {
        Stage4Error::Usage("tdp_packing <command> ... (see the module docs)".into())
    })?;
    let a = Args::parse(rest)?;
    match cmd.as_str() {
        "serialize" => cmd_serialize(&a),
        "anf-report" => cmd_anf_report(&a),
        "window-anf" => cmd_window_anf(&a),
        "decode" => cmd_decode(&a),
        "eval" => cmd_eval(&a),
        "container" => cmd_container(&a),
        "verify-equiv" => cmd_verify_equiv(&a),
        "pack-replay" => cmd_pack_replay(&a),
        "pack" => cmd_pack(&a),
        "repair" => cmd_repair(&a),
        "check-final" => cmd_check_final(&a),
        "run" => run::cmd_run(&a),
        "census" => run::cmd_census(&a),
        other => Err(Stage4Error::Usage(format!("unknown command {other:?}"))),
    }
}

// ---------------------------------------------------------------- outputs

/// A private output file written to `<path>.tmp` and renamed on `commit`.
struct Staged {
    path: PathBuf,
    tmp: PathBuf,
}

impl Staged {
    fn create(path: &Path) -> Result<(Staged, BufWriter<File>), Stage4Error> {
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
        let f = o.open(&tmp).map_err(|e| Stage4Error::io(&tmp, e))?;
        Ok((
            Staged {
                path: path.to_path_buf(),
                tmp,
            },
            BufWriter::new(f),
        ))
    }

    fn finish(w: BufWriter<File>, tmp: &Path) -> Result<(), Stage4Error> {
        let f = w
            .into_inner()
            .map_err(|e| Stage4Error::io(tmp, e.into_error()))?;
        f.sync_all().map_err(|e| Stage4Error::io(tmp, e))
    }

    fn commit(self) -> Result<(), Stage4Error> {
        fs::rename(&self.tmp, &self.path).map_err(|e| Stage4Error::io(&self.path, e))
    }

    fn abandon(&self) {
        let _ = fs::remove_file(&self.tmp);
    }
}

// ---------------------------------------------------------------- serialize

fn cmd_serialize(a: &Args) -> Result<(), Stage4Error> {
    a.check_known(&[
        "container",
        "source-wires",
        "no-checks",
        "zero-slice-c",
        "threads",
    ])?;
    if a.pos.len() != 4 {
        return Err(Stage4Error::Usage(
            "serialize <rep.mpmct1> <rep.bounds> <out.bin> <label> [--container P --source-wires N]".into(),
        ));
    }
    setup_threads(a)?;
    let t0 = Instant::now();
    let circ = read_oracle_mpmct(Path::new(&a.pos[0]))?;
    let bounds = read_bounds(Path::new(&a.pos[1]))?;
    let out = Path::new(&a.pos[2]);
    let label = &a.pos[3];
    let source_wires = a.num("source-wires")?;
    if (a.get("container").is_some() || a.get("zero-slice-c").is_some()) && source_wires.is_none() {
        return Err(Stage4Error::Usage(
            "--container/--zero-slice-c need --source-wires".into(),
        ));
    }
    let nw = circ.computed_n;
    if let Some(n) = source_wires
        && (n == 0 || 2 * n > nw as u64)
    {
        return Err(Stage4Error::Usage(format!(
            "--source-wires {n} does not fit a {nw}-wire state"
        )));
    }
    let (body, mut w) = Staged::create(out)?;
    let rep = match encode_body(
        &circ.gates,
        &bounds,
        circ.header_m,
        nw,
        MAX_CONCURRENT_ANF.max(rayon::current_num_threads()),
        &mut w,
    ) {
        Ok(r) => r,
        Err(e) => {
            body.abandon();
            return Err(e);
        }
    };
    Staged::finish(w, &body.tmp)?;
    println!(
        "{label}: serialized {:.1} MB in {:.0}s, {} windows",
        rep.body_bytes as f64 / 1e6,
        t0.elapsed().as_secs_f64(),
        rep.windows
    );
    let checked = (|| -> Result<(), Stage4Error> {
        if a.flag("no-checks") {
            return Ok(());
        }
        let bytes = fs::read(&body.tmp).map_err(|e| Stage4Error::io(&body.tmp, e))?;
        let n = source_wires.unwrap_or(0) as u32;
        let d = decode(
            &bytes,
            Format::Raw {
                source_wires: n,
                state_wires: nw as u32,
            },
        )?;
        let wc = check_windows_anf(
            &circ.gates,
            &bounds,
            &d,
            EXHAUSTIVE_SUPPORT_MAX,
            WINDOW_CHECK_SEED,
        )?;
        check_deliverable(&circ.gates, circ.computed_n, &d, DE_EVAL_SEED)?;
        let mut zs = String::new();
        if let Some(cpath) = a.get("zero-slice-c") {
            let c =
                read_g57_file(cpath).map_err(|e| Stage4Error::Input(format!("{cpath}: {e}")))?;
            let s = check_zero_slice(&c, &d, n as usize, ZERO_SLICE_SAMPLES, ZERO_SLICE_SEED)?;
            zs = format!("; zero slice C(x) on wires {n}..{} ok ({s} samples)", 2 * n);
        }
        println!(
            "checks: decode ok; windows == window_anf and == gates ({} exhaustive, {} sampled); \
             eval vs gates (1024 lanes, seed {DE_EVAL_SEED}) match=True{zs}",
            wc.exhaustive, wc.sampled
        );
        Ok(())
    })();
    if let Err(e) = checked {
        body.abandon();
        return Err(e);
    }
    if let Some(cpath) = a.get("container") {
        let h = ContainerHeader {
            source_wires: source_wires.unwrap() as u32,
            state_wires: nw as u32,
            io_layout: IO_LAYOUT_CLASSIC_ZERO_SLICE,
            window_count: rep.windows as u32,
            body_bytes: rep.body_bytes,
            body_crc32: rep.body_crc32,
        };
        if let Err(e) = write_container(&body.tmp, Path::new(cpath), &h) {
            body.abandon();
            return Err(e);
        }
    }
    body.commit()?;
    println!(
        "emit: windows={} monomials={} max_window_monomials={} empty_windows={} body_bytes={} body_crc32={:08x} body_xxh3={:032x}",
        rep.windows,
        rep.monomials,
        rep.max_window_monomials,
        rep.empty_windows,
        rep.body_bytes,
        rep.body_crc32,
        rep.body_xxh3
    );
    Ok(())
}

/// The 48-byte header, then a copy of the body file.
fn write_container(body: &Path, out: &Path, h: &ContainerHeader) -> Result<(), Stage4Error> {
    let (st, mut w) = Staged::create(out)?;
    let r = (|| -> Result<(), Stage4Error> {
        w.write_all(&h.to_bytes())
            .map_err(|e| Stage4Error::io(&st.tmp, e))?;
        let mut f = File::open(body).map_err(|e| Stage4Error::io(body, e))?;
        io::copy(&mut f, &mut w).map_err(|e| Stage4Error::io(&st.tmp, e))?;
        Ok(())
    })();
    if let Err(e) = r {
        st.abandon();
        return Err(e);
    }
    Staged::finish(w, &st.tmp)?;
    st.commit()
}

fn cmd_container(a: &Args) -> Result<(), Stage4Error> {
    a.check_known(&["source-wires", "state-wires"])?;
    if a.pos.len() != 2 {
        return Err(Stage4Error::Usage(
            "container <body.bin> <out.tdp4> --source-wires N --state-wires M".into(),
        ));
    }
    let n = a
        .num("source-wires")?
        .ok_or_else(|| Stage4Error::Usage("--source-wires".into()))?;
    let m = a
        .num("state-wires")?
        .ok_or_else(|| Stage4Error::Usage("--state-wires".into()))?;
    if m == 0 || m > 65536 || 2 * n > m {
        return Err(Stage4Error::Usage(format!(
            "source {n} / state {m} out of range"
        )));
    }
    let body = Path::new(&a.pos[0]);
    let bytes =
        fs::read(body).map_err(|e| Stage4Error::Input(format!("{}: {e}", body.display())))?;
    let d = decode(
        &bytes,
        Format::Raw {
            source_wires: n as u32,
            state_wires: m as u32,
        },
    )?;
    let h = ContainerHeader {
        source_wires: n as u32,
        state_wires: m as u32,
        io_layout: IO_LAYOUT_CLASSIC_ZERO_SLICE,
        window_count: d.window_count() as u32,
        body_bytes: bytes.len() as u64,
        body_crc32: crate::stages::packing::serialize::crc32_ieee(&bytes),
    };
    write_container(body, Path::new(&a.pos[1]), &h)
}

// ---------------------------------------------------------------- reports

fn cmd_anf_report(a: &Args) -> Result<(), Stage4Error> {
    a.check_known(&["anf-cap-semantics", "stats", "threads"])?;
    if a.pos.len() < 2 || a.pos.len() > 5 {
        return Err(Stage4Error::Usage(
            "anf-report <C.mpmct1> <bounds> [budget] [label] [check_every]".into(),
        ));
    }
    setup_threads(a)?;
    let sem = cap_semantics(a)?;
    let circ = read_oracle_mpmct(Path::new(&a.pos[0]))?;
    let bounds = read_bounds(Path::new(&a.pos[1]))?;
    let budget = a
        .pos
        .get(2)
        .map(|v| parse_u64(v, "budget"))
        .transpose()?
        .unwrap_or(8192);
    let label = a.pos.get(3).map(String::as_str).unwrap_or("anf");
    let every = a
        .pos
        .get(4)
        .map(|v| parse_u64(v, "check_every"))
        .transpose()?
        .unwrap_or(500);
    if every == 0 {
        return Err(Stage4Error::Usage("check_every must be at least 1".into()));
    }
    let r = anf_size_report(
        &circ.gates,
        circ.computed_n,
        &bounds,
        budget as usize,
        label,
        every as usize,
        sem,
    );
    print!("{}", r.text);
    if a.flag("stats") {
        eprintln!(
            "stats: overflow product={} running_sum={} final={}; mul calls={} ambiguous={} aborts={} order_dependent={} ({})",
            r.overflow_product,
            r.overflow_running_sum,
            r.overflow_final,
            r.stats.calls,
            r.stats.ambiguous,
            r.stats.aborts,
            r.stats.aborts_order_dependent,
            sem.name()
        );
    }
    if r.failed > 0 {
        return Err(Stage4Error::Verify(format!(
            "{} spot checks failed",
            r.failed
        )));
    }
    Ok(())
}

const FNV_OFF: u64 = 0xCBF2_9CE4_8422_2325;
const FNV_PRIME: u64 = 0x100_0000_01B3;

struct WindowDecisions<'a> {
    gates: &'a [XGate],
    cuts: &'a [usize],
    budget: usize,
    limbs: usize,
}

impl MonoWidthFn<Vec<String>> for WindowDecisions<'_> {
    fn call<const W: usize>(self) -> Vec<String> {
        use rayon::prelude::*;
        (0..self.cuts.len() - 1)
            .into_par_iter()
            .map(|wi| {
                let (lo, hi) = (self.cuts[wi], self.cuts[wi + 1]);
                let mut cols = Vec::new();
                for sem in [CapSemantics::CPython39, CapSemantics::Canonical] {
                    let mut st = MulStats::default();
                    cols.push(
                        match window_anf::<W>(&self.gates[lo..hi], self.budget, sem, &mut st) {
                            Err(_) => "overflow".to_string(),
                            Ok(anf) => {
                                let mut h = FNV_OFF;
                                let mut feed = |x: u64| h = (h ^ x).wrapping_mul(FNV_PRIME);
                                for (w, poly) in &anf.changed {
                                    feed(*w as u64);
                                    feed(poly.len() as u64);
                                    for m in poly {
                                        for l in &m.0[..self.limbs] {
                                            feed(*l);
                                        }
                                    }
                                }
                                format!("ok:{}:{h:016x}", anf.total)
                            }
                        },
                    );
                }
                format!("{wi} {lo} {hi} {} {}", cols[0], cols[1])
            })
            .collect()
    }
}

fn cmd_window_anf(a: &Args) -> Result<(), Stage4Error> {
    a.check_known(&["threads"])?;
    if a.pos.len() != 3 {
        return Err(Stage4Error::Usage(
            "window-anf <C.mpmct1> <bounds> <budget>".into(),
        ));
    }
    setup_threads(a)?;
    let circ = read_oracle_mpmct(Path::new(&a.pos[0]))?;
    let bounds = read_bounds(Path::new(&a.pos[1]))?;
    let budget = parse_u64(&a.pos[2], "budget")? as usize;
    let cuts = window_cuts(&bounds, circ.gates.len());
    let nw = circ.computed_n.max(1);
    if nw > 1024 {
        return Err(Stage4Error::Usage(format!("{nw} wires: at most 1024")));
    }
    let lines = with_mono_width(
        nw,
        WindowDecisions {
            gates: &circ.gates,
            cuts: &cuts,
            budget,
            limbs: nw.div_ceil(64),
        },
    );
    let stdout = io::stdout();
    let mut o = stdout.lock();
    for l in lines {
        writeln!(o, "{l}").map_err(|e| Stage4Error::io("<stdout>", e))?;
    }
    Ok(())
}

// ---------------------------------------------------------------- decode / eval

fn load_deliverable(a: &Args, path: &str) -> Result<Deliverable, Stage4Error> {
    let bytes = fs::read(path).map_err(|e| Stage4Error::Input(format!("{path}: {e}")))?;
    match a.get("format") {
        Some("container") => decode(&bytes, Format::Container),
        Some("raw") => {
            let n = a
                .num("n")?
                .ok_or_else(|| Stage4Error::Usage("--format raw needs --n".into()))?;
            let m = a.num("state-wires")?.unwrap_or(4 * n);
            if n > 65536 || m > 65536 {
                return Err(Stage4Error::Usage("widths above 65536".into()));
            }
            decode(
                &bytes,
                Format::Raw {
                    source_wires: n as u32,
                    state_wires: m as u32,
                },
            )
        }
        _ => Err(Stage4Error::Usage(
            "--format raw|container is required (no guessing)".into(),
        )),
    }
}

fn cmd_decode(a: &Args) -> Result<(), Stage4Error> {
    a.check_known(&["format", "n", "state-wires", "stats"])?;
    if a.pos.len() != 1 {
        return Err(Stage4Error::Usage(
            "decode <file> --format raw --n N | --format container".into(),
        ));
    }
    let d = load_deliverable(a, &a.pos[0])?;
    println!(
        "format={} windows={} monomials={} body_bytes={} source_wires={} state_wires={}",
        if d.header.is_some() {
            "container"
        } else {
            "raw"
        },
        d.window_count(),
        d.monomials,
        d.body_bytes,
        d.source_wires,
        d.state_wires
    );
    if a.flag("stats") {
        let (mut fmax, mut cmax, mut empty) = (0usize, 0usize, 0usize);
        for i in 0..d.window_count() {
            let w = d.window(i);
            fmax = fmax.max(w.support.len());
            cmax = cmax.max(w.changed_len());
            if w.changed_len() == 0 {
                empty += 1;
            }
        }
        println!("max_support={fmax} max_changed={cmax} empty_windows={empty}");
    }
    Ok(())
}

fn parse_hex_bits(s: &str, width: usize) -> Result<Vec<u64>, Stage4Error> {
    let s = s.strip_prefix("0x").unwrap_or(s);
    let mut limbs = vec![0u64; width.div_ceil(64).max(1)];
    for (i, c) in s.bytes().rev().enumerate() {
        let d = (c as char)
            .to_digit(16)
            .ok_or_else(|| Stage4Error::Usage(format!("bad hex digit {:?}", c as char)))?
            as u64;
        if d == 0 {
            continue;
        }
        let bit = 4 * i;
        if bit + (64 - d.leading_zeros() as usize) > width {
            return Err(Stage4Error::Usage(format!(
                "hex value wider than {width} bits"
            )));
        }
        limbs[bit / 64] |= d << (bit % 64);
    }
    Ok(limbs)
}

fn hex_bits(limbs: &[u64], lo: usize, hi: usize) -> String {
    // bits lo..hi as a Python-style hex integer
    let mut out = vec![0u64; (hi - lo).div_ceil(64).max(1)];
    for b in lo..hi {
        if (limbs[b >> 6] >> (b & 63)) & 1 == 1 {
            let k = b - lo;
            out[k >> 6] |= 1 << (k & 63);
        }
    }
    let top = (0..out.len()).rev().find(|&i| out[i] != 0);
    match top {
        None => "0".into(),
        Some(t) => {
            let mut s = format!("{:x}", out[t]);
            for i in (0..t).rev() {
                s.push_str(&format!("{:016x}", out[i]));
            }
            s
        }
    }
}

fn cmd_eval(a: &Args) -> Result<(), Stage4Error> {
    a.check_known(&["format", "n", "state-wires", "x", "state"])?;
    if a.pos.len() != 1 {
        return Err(Stage4Error::Usage(
            "eval <file> --format … (--x HEX | --state HEX)".into(),
        ));
    }
    let d = load_deliverable(a, &a.pos[0])?;
    let nw = d.state_wires as usize;
    let n = d.source_wires as usize;
    match (a.get("x"), a.get("state")) {
        (Some(x), None) => {
            if n == 0 {
                return Err(Stage4Error::Usage("--x needs a source width".into()));
            }
            let mut st = parse_hex_bits(x, n)?;
            st.resize(nw.div_ceil(64), 0);
            d.eval_limbs(&mut st);
            println!("{}", hex_bits(&st, n, 2 * n));
        }
        (None, Some(s)) => {
            let mut st = parse_hex_bits(s, nw)?;
            st.resize(nw.div_ceil(64), 0);
            d.eval_limbs(&mut st);
            println!("{}", hex_bits(&st, 0, nw));
        }
        _ => {
            return Err(Stage4Error::Usage(
                "give exactly one of --x and --state".into(),
            ));
        }
    }
    Ok(())
}

fn cmd_verify_equiv(a: &Args) -> Result<(), Stage4Error> {
    a.check_known(&["seed"])?;
    if a.pos.len() != 2 {
        return Err(Stage4Error::Usage(
            "verify-equiv <A.mpmct1> <B.mpmct1> [--seed 77]".into(),
        ));
    }
    let ca = read_oracle_mpmct(Path::new(&a.pos[0]))?;
    let cb = read_oracle_mpmct(Path::new(&a.pos[1]))?;
    let seed = a.num("seed")?.unwrap_or(77);
    verify_equiv(&ca.gates, &cb.gates, ca.header_n.max(cb.header_n), seed)?;
    println!(
        "{}: {} vs {} gates; EQUIVALENT (1024 random states)",
        a.pos[1],
        ca.gates.len(),
        cb.gates.len()
    );
    Ok(())
}

struct PackReplay<'a> {
    gates: &'a [XGate],
    cuts: &'a [usize],
    mul_cap: usize,
    sem: CapSemantics,
}

/// (aborts, aborts not at the window's last gate, windows with an ambiguous product, stats)
type ReplayOut = (usize, usize, usize, MulStats);

impl MonoWidthFn<ReplayOut> for PackReplay<'_> {
    fn call<const W: usize>(self) -> ReplayOut {
        use crate::stages::packing::anf::WindowState;
        use rayon::prelude::*;
        let per: Vec<(usize, usize, bool, MulStats)> = (0..self.cuts.len() - 1)
            .into_par_iter()
            .map(|wi| {
                let (lo, hi) = (self.cuts[wi], self.cuts[wi + 1]);
                let mut ws = WindowState::<W>::new(self.mul_cap);
                let mut st = MulStats::default();
                let (mut aborts, mut early) = (0, 0);
                for gi in lo..hi {
                    if !ws.compose(&self.gates[gi], self.sem, &mut st) {
                        aborts += 1;
                        if gi != hi - 1 {
                            early += 1;
                        }
                        break;
                    }
                }
                (aborts, early, st.ambiguous > 0, st)
            })
            .collect();
        let mut out: ReplayOut = (0, 0, 0, MulStats::default());
        for (a, e, amb, st) in per {
            out.0 += a;
            out.1 += e;
            out.2 += amb as usize;
            out.3.add(&st);
        }
        out
    }
}

fn cmd_pack_replay(a: &Args) -> Result<(), Stage4Error> {
    a.check_known(&["anf-cap-semantics", "threads"])?;
    if a.pos.len() < 2 || a.pos.len() > 3 {
        return Err(Stage4Error::Usage(
            "pack-replay <packed.mpmct1> <packed.bounds> [b_mono=24576]".into(),
        ));
    }
    setup_threads(a)?;
    let sem = cap_semantics(a)?;
    let circ = read_oracle_mpmct(Path::new(&a.pos[0]))?;
    let bounds = read_bounds(Path::new(&a.pos[1]))?;
    let b_mono = a
        .pos
        .get(2)
        .map(|v| parse_u64(v, "b_mono"))
        .transpose()?
        .unwrap_or(24576) as usize;
    let cuts = window_cuts(&bounds, circ.gates.len());
    let nw = circ.computed_n.max(1);
    if nw > 1024 {
        return Err(Stage4Error::Usage(format!("{nw} wires: at most 1024")));
    }
    let (aborts, early, amb_windows, st) = with_mono_width(
        nw,
        PackReplay {
            gates: &circ.gates,
            cuts: &cuts,
            mul_cap: 8 * b_mono,
            sem,
        },
    );
    println!(
        "windows {} aborts {} (not at window end: {}) order_dependent {} windows_with_ambiguous_mul {} mul_calls {} ambiguous {} ({})",
        cuts.len() - 1,
        aborts,
        early,
        st.aborts_order_dependent,
        amb_windows,
        st.calls,
        st.ambiguous,
        sem.name()
    );
    Ok(())
}

// ---------------------------------------------------------------- pack

fn cmd_pack(a: &Args) -> Result<(), Stage4Error> {
    a.check_known(&["order-out", "anf-cap-semantics", "stats", "threads"])?;
    if a.pos.len() < 2 || a.pos.len() > 8 {
        return Err(Stage4Error::Usage(
            "pack <A> <B> [W_cap] [G] [B_mono] [span_cap] [label] [out.mpmct1]".into(),
        ));
    }
    setup_threads(a)?;
    let mut p = PackParams::ORACLE_DEFAULTS;
    p.cap_semantics = cap_semantics(a)?;
    let num = |i: usize, what: &str, dflt: usize| -> Result<usize, Stage4Error> {
        a.pos
            .get(i)
            .map(|v| parse_u64(v, what).map(|x| x as usize))
            .transpose()
            .map(|x| x.unwrap_or(dflt))
    };
    p.w_cap = num(2, "W_cap", p.w_cap)?;
    p.g = num(3, "G", p.g)?;
    p.b_mono = num(4, "B_mono", p.b_mono)?;
    p.span_cap = num(5, "span_cap", p.span_cap)?;
    let label = a
        .pos
        .get(6)
        .cloned()
        .unwrap_or_else(|| format!("pack3-G{}", p.g));
    let t0 = Instant::now();
    let ca = read_oracle_mpmct(Path::new(&a.pos[0]))?;
    let cb = read_oracle_mpmct(Path::new(&a.pos[1]))?;
    let t_read = t0.elapsed().as_secs_f64();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let packed = pack(&ca, &cb, &p, &label, &mut out)?;
    drop(out);
    let t_pack = t0.elapsed().as_secs_f64() - t_read;
    let mut staged: Vec<Staged> = Vec::new();
    let r = (|| -> Result<(), Stage4Error> {
        if let Some(path) = a.pos.get(7) {
            let path = Path::new(path);
            let (st, mut w) = Staged::create(path)?;
            staged.push(st);
            let tmp = staged.last().unwrap().tmp.clone();
            let gates: Vec<XGate> = packed.gates(&cb.gates).cloned().collect();
            write_pack_mpmct(&mut w, packed.n_out, &gates).map_err(|e| Stage4Error::io(&tmp, e))?;
            Staged::finish(w, &tmp)?;
            let mut bp = path.as_os_str().to_owned();
            bp.push(".bounds");
            let (st, mut w) = Staged::create(Path::new(&bp))?;
            staged.push(st);
            let tmp = staged.last().unwrap().tmp.clone();
            write_bounds(&mut w, &packed.bounds).map_err(|e| Stage4Error::io(&tmp, e))?;
            Staged::finish(w, &tmp)?;
        }
        if let Some(path) = a.get("order-out") {
            let (st, mut w) = Staged::create(Path::new(path))?;
            staged.push(st);
            let tmp = staged.last().unwrap().tmp.clone();
            let order: Vec<usize> = packed.order.iter().map(|&i| i as usize).collect();
            write_bounds(&mut w, &order).map_err(|e| Stage4Error::io(&tmp, e))?;
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
    for s in staged {
        s.commit()?;
    }
    if a.flag("stats") {
        let rp = &packed.report;
        eprintln!(
            "stats: mul calls={} ambiguous={} aborts={} order_dependent={} ({}); compose_aborts={} \
             bounds={} duplicate_bounds={} windows={}; read {:.2}s pack+verify+certify {:.2}s \
             (spans {:.2}s intervals {:.2}s schedule {:.2}s verify {:.2}s certify {:.2}s)",
            rp.mul.calls,
            rp.mul.ambiguous,
            rp.mul.aborts,
            rp.mul.aborts_order_dependent,
            p.cap_semantics.name(),
            rp.compose_aborts,
            rp.bounds,
            rp.duplicate_bounds,
            rp.windows,
            t_read,
            t_pack,
            rp.time_spans_s,
            rp.time_intervals_s,
            rp.time_schedule_s,
            rp.time_verify_s,
            rp.time_certify_s
        );
    }
    Ok(())
}

// ---------------------------------------------------------------- repair

fn req<'a>(a: &'a Args, k: &str) -> Result<&'a str, Stage4Error> {
    a.get(k)
        .ok_or_else(|| Stage4Error::Usage(format!("--{k} is required")))
}

fn opt_usize(a: &Args, k: &str, dflt: usize) -> Result<usize, Stage4Error> {
    Ok(a.num(k)?.map(|v| v as usize).unwrap_or(dflt))
}

fn repair_params(a: &Args) -> Result<RepairParams, Stage4Error> {
    let mut p = RepairParams::default();
    p.split_budget = opt_usize(a, "split-budget", p.split_budget)?;
    p.fix_iterations = opt_usize(a, "fix-iterations", p.fix_iterations)?;
    p.fallback_iterations = opt_usize(a, "fallback-iterations", p.fallback_iterations)?;
    p.cap_semantics = cap_semantics(a)?;
    if p.split_budget == 0 {
        return Err(Stage4Error::Usage("--split-budget must be positive".into()));
    }
    Ok(p)
}

/// Read C (g57) and check `n` against it and the state width.
fn load_source_c(path: &str, n: u64, state_n: usize) -> Result<Vec<XGate>, Stage4Error> {
    if n == 0 || n > 256 {
        return Err(Stage4Error::Usage(format!(
            "--n {n}: stage 4 supports 1 <= n <= 256"
        )));
    }
    let n = n as usize;
    if 4 * n > state_n {
        return Err(Stage4Error::Usage(format!(
            "--n {n}: 4n exceeds the {state_n}-wire state"
        )));
    }
    let c = read_g57_file(path).map_err(|e| Stage4Error::Input(format!("{path}: {e}")))?;
    if let Some(w) = c.iter().map(|g| g.max_wire() as usize).max()
        && w >= n
    {
        return Err(Stage4Error::Input(format!(
            "{path}: C uses wire {w}, beyond n = {n}"
        )));
    }
    Ok(c)
}

/// Writes every line to stdout and keeps a copy (the log file).
struct Tee {
    buf: Vec<u8>,
}

impl Write for Tee {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        let stdout = io::stdout();
        let mut o = stdout.lock();
        o.write_all(b)?;
        o.flush()?;
        self.buf.extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        io::stdout().flush()
    }
}

fn private_dir(dir: &Path) -> Result<(), Stage4Error> {
    if !dir.exists() {
        fs::create_dir_all(dir).map_err(|e| Stage4Error::io(dir, e))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(dir, fs::Permissions::from_mode(0o700))
                .map_err(|e| Stage4Error::io(dir, e))?;
        }
    } else if !dir.is_dir() {
        return Err(Stage4Error::Usage(format!(
            "--out-dir {} is not a directory",
            dir.display()
        )));
    }
    Ok(())
}

/// Writes one output file's content.
type FileBody<'a> = Box<dyn FnOnce(&mut BufWriter<File>) -> io::Result<()> + 'a>;

/// Write each (path, writer) via *.tmp; commit all or none.
fn write_all_staged(files: Vec<(PathBuf, FileBody<'_>)>) -> Result<(), Stage4Error> {
    let mut staged: Vec<Staged> = Vec::new();
    let r = (|| -> Result<(), Stage4Error> {
        for (path, f) in files {
            let (st, mut w) = Staged::create(&path)?;
            let tmp = st.tmp.clone();
            staged.push(st);
            f(&mut w).map_err(|e| Stage4Error::io(&tmp, e))?;
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
    for s in staged {
        s.commit()?;
    }
    Ok(())
}

fn report_lines(r: &RepairReport, p: &RepairParams, status: &str) -> Vec<String> {
    let join = |v: &[usize]| {
        v.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")
    };
    vec![
        format!("status={status}"),
        format!(
            "param.split_budget={} param.fix_iterations={} param.fallback_iterations={} \
             param.cview_batches={} param.cview_seed={} param.anf_cap_semantics={} param.repair_mode={}",
            p.split_budget,
            p.fix_iterations,
            p.fallback_iterations,
            p.cview.batches,
            p.cview.seed,
            p.cap_semantics.name(),
            p.mode.name()
        ),
        format!(
            "repair.cancel_passes={} repair.cancel_pairs={} repair.fix_trajectory={} repair.fix_moved={} \
             repair.fix_exhausted={}",
            join(&r.cancel_passes),
            r.cancel_passes.iter().sum::<usize>(),
            join(&r.fix_trajectory),
            join(&r.fix_moved),
            r.fix_exhausted as u8
        ),
        format!(
            "repair.fallback_trajectory={} repair.merged={} repair.splits={} repair.forced_splits={} \
             repair.single_window_scans={} repair.scans={}",
            join(&r.fallback_trajectory),
            join(&r.merged),
            join(&r.splits),
            r.forced_splits.len(),
            r.single_window_scans,
            r.scans
        ),
        format!(
            "repair.final_gates={} repair.final_bounds={} repair.final_visible={} \
             repair.final_nonoutput_visible={} repair.final_over_budget={}",
            r.final_gates,
            r.final_bounds,
            r.final_visible,
            r.final_nonoutput_visible.len(),
            r.final_over_budget.len()
        ),
        format!(
            "repair.split_mul_calls={} repair.split_mul_ambiguous={} repair.split_aborts={} \
             repair.split_order_dependent={} repair.split_order_dependent_windows={}",
            r.split_stats.calls,
            r.split_stats.ambiguous,
            r.split_stats.aborts,
            r.split_stats.aborts_order_dependent,
            r.split_order_dependent_windows
        ),
    ]
}

fn cmd_repair(a: &Args) -> Result<(), Stage4Error> {
    a.check_known(&[
        "packed",
        "bounds",
        "source-c",
        "n",
        "out-dir",
        "g",
        "split-budget",
        "fix-iterations",
        "fallback-iterations",
        "anf-cap-semantics",
        "allow-floor-miss",
        "allow-over-budget",
        "no-oracle-files",
        "stats",
        "threads",
    ])?;
    if !a.pos.is_empty() {
        return Err(Stage4Error::Usage(
            "repair --packed P [--bounds B] --source-c C --n N --out-dir DIR [...]".into(),
        ));
    }
    setup_threads(a)?;
    let p = repair_params(a)?;
    let packed_path = req(a, "packed")?;
    let bounds_path = a
        .get("bounds")
        .map(String::from)
        .unwrap_or_else(|| format!("{packed_path}.bounds"));
    let out_dir = PathBuf::from(req(a, "out-dir")?);
    let g = opt_usize(a, "g", 512)?;
    let n = a
        .num("n")?
        .ok_or_else(|| Stage4Error::Usage("--n is required".into()))?;
    let t0 = Instant::now();
    let packed = read_oracle_mpmct(Path::new(packed_path))?;
    let bounds0 = read_bounds(Path::new(&bounds_path))?;
    let c = load_source_c(req(a, "source-c")?, n, packed.header_n)?;
    private_dir(&out_dir)?;
    let rep_path = out_dir.join(format!("rep{g}.mpmct1"));
    let cv = CView::new(&c, n as usize, &p.cview);
    let mut tee = Tee { buf: Vec::new() };
    let r = repair(
        &packed.gates,
        packed.header_n,
        &bounds0,
        &cv,
        &p,
        &rep_path.display().to_string(),
        &mut tee,
    )?;
    let t_repair = t0.elapsed().as_secs_f64();
    let status = r
        .report
        .status(a.flag("allow-floor-miss"), a.flag("allow-over-budget"));
    let status_name = match &status {
        Ok(s) => *s,
        Err(Stage4Error::Floor { .. }) => "floor_miss",
        Err(_) => "over_budget",
    };
    // outputs (the oracle's files)
    let path = |suffix: &str| out_dir.join(format!("rep{g}.{suffix}"));
    let (pe, pa, pw) = (path("exposure.tsv"), path("allcuts.tsv"), path("wit.tsv"));
    let cview_log = cv.scan_log(
        &r.last_scan,
        &pe.display().to_string(),
        &pa.display().to_string(),
        &pw.display().to_string(),
    );
    let mut files: Vec<(PathBuf, FileBody)> = vec![
        (
            rep_path.clone(),
            Box::new(|w| write_repair_mpmct(w, r.n_header, &r.gates)),
        ),
        (path("bounds"), Box::new(|w| write_bounds(w, &r.bounds))),
        (
            out_dir.join(format!("repair{g}.log")),
            Box::new(|w| w.write_all(&tee.buf)),
        ),
    ];
    if !a.flag("no-oracle-files") {
        let s = &r.last_scan.report;
        files.push((pe, Box::new(|w| s.write_exposure_tsv(w))));
        files.push((pa, Box::new(|w| s.write_allcuts_tsv(w))));
        files.push((pw, Box::new(|w| s.write_witness_tsv(w))));
        files.push((
            path("cview.log"),
            Box::new(move |w| {
                for l in &cview_log {
                    writeln!(w, "{l}")?;
                }
                Ok(())
            }),
        ));
    }
    write_all_staged(files)?;
    let fr = &r.report;
    eprintln!(
        "repair: {} -> {} gates, {} boundaries; final scan: {} visible, {} non-output visible; \
         {} windows over budget; status={status_name} ({:.1}s)",
        fr.packed_gates,
        fr.final_gates,
        fr.final_bounds,
        fr.final_visible,
        fr.final_nonoutput_visible.len(),
        fr.final_over_budget.len(),
        t_repair
    );
    if a.flag("stats") {
        for l in report_lines(fr, &p, status_name) {
            eprintln!("stats: {l}");
        }
    }
    status.map(|_| ())
}

fn cmd_check_final(a: &Args) -> Result<(), Stage4Error> {
    a.check_known(&[
        "rep",
        "bounds",
        "source-c",
        "n",
        "split-budget",
        "anf-cap-semantics",
        "allow-floor-miss",
        "allow-over-budget",
        "threads",
    ])?;
    if !a.pos.is_empty() {
        return Err(Stage4Error::Usage(
            "check-final --rep R --bounds B --source-c C --n N [...]".into(),
        ));
    }
    setup_threads(a)?;
    let p = repair_params(a)?;
    let n = a
        .num("n")?
        .ok_or_else(|| Stage4Error::Usage("--n is required".into()))?;
    let rep = read_oracle_mpmct(Path::new(req(a, "rep")?))?;
    let bounds: Vec<usize> = read_bounds(Path::new(req(a, "bounds")?))?
        .into_iter()
        .filter(|&b| b >= 0)
        .map(|b| b as usize)
        .collect();
    let c = load_source_c(req(a, "source-c")?, n, rep.header_n)?;
    let nw = anf_width(rep.header_n, &rep.gates)?;
    let cv = CView::new(&c, n as usize, &p.cview);
    let mut oracle = cv.oracle(&rep.gates);
    let mut fits = FitCache::new(&rep.gates, nw, p.split_budget, p.cap_semantics);
    let (s, nonout, over) = final_tests(&cv, &mut oracle, &mut fits, &rep.gates, &bounds);
    let fr = RepairReport {
        final_gates: rep.gates.len(),
        final_bounds: bounds.len(),
        final_visible: s.visible(),
        final_nonoutput_visible: nonout,
        final_over_budget: over,
        ..Default::default()
    };
    let status = fr.status(a.flag("allow-floor-miss"), a.flag("allow-over-budget"));
    let status_name = match &status {
        Ok(s) => *s,
        Err(Stage4Error::Floor { .. }) => "floor_miss",
        Err(_) => "over_budget",
    };
    println!(
        "check-final: {} gates, {} windows; C-view visible {} (floor {n}), {} non-output visible; \
         {} windows over budget {}; status={status_name}",
        fr.final_gates,
        s.cuts + 1,
        fr.final_visible,
        fr.final_nonoutput_visible.len(),
        fr.final_over_budget.len(),
        p.split_budget
    );
    status.map(|_| ())
}
