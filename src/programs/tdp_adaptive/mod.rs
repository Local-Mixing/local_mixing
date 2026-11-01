//! `tdp_adaptive`: the production binary of stage 6 in adaptive mode
//! (`stages::adaptive`, docs/ADAPTIVE_STAGE.md): the stage-5 circuit is
//! resynthesized in large C-aware blocks, round after round, until its
//! size plateaus.
//!
//! It reads NO environment variables; threads are set by `--threads` only.
//! The seed never appears in any output (`param.seed_source` names where
//! it came from). Exit codes are `AdaptiveError::exit_code`.
//!
//! ```text
//! tdp_adaptive run --input G.mpmct1 --source-c C.g57 --n N --out-dir DIR
//!     [--sandwich S.mpmct1] [--seed S | --seed-file PATH] [--threads T]
//!     [--max-rounds N=0] [--stop-fraction F=0.02] [--seeds-per-block K=2]
//!     [--time-limit S=60] [--restarts R=16] [--beam B=6]
//!     [--term-cap 131072] [--max-block-gates 65536] [--segments 16]
//!     [--avoid-radius 64] [--refinement-levels 2] [--min-refine-gates 128]
//!     [--census-batches 96] [--probe-inputs 8192]
//!     [--accept smaller|smaller-not-leakier] [--blind]
//!     [--no-canonical-order] [--keep-round-circuits] [--no-postopt]
//!     [--ancilla-pool 32] [--compose-cap 1048576] [--compose-time-limit 300]
//!     [--equiv-batches 8] [--label tdp]
//!     Runs the rounds into DIR (resumable: completed rounds are reloaded),
//!     publishes `final.mpmct1` after the canonical-order, zero-slice and
//!     (with --sandwich) payload-contract checks, with `adaptive.report`
//!     and `adaptive.log`. `--max-rounds 0` runs to the plateau rules.
//!     `--blind` plans and splits against the empty census (the C-blind
//!     attacker's view).
//! tdp_adaptive census --input G.mpmct1 --source-c C.g57 --n N --out-dir DIR
//!     [--census-batches 96] [--probe-inputs 8192] [--threads T] [--label tdp]
//!     The stand-alone census of a circuit against C: `DIR/census/`,
//!     `DIR/boundary.txt` and the metrics in `DIR/census.report`.
//! tdp_adaptive help
//! ```
use crate::stages::adaptive::census;
use crate::stages::adaptive::driver::{
    self, DriverInputs, DriverParams, load_inputs, load_or_classify, private_dir, source_of,
};
use crate::stages::adaptive::round::{
    Accept, AdaptiveError, EXIT_CODES, Leakage, Log, leakage_kv, write_text_atomic,
};
use std::fmt::Write as _;
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

pub fn run() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match dispatch(&args) {
        Ok(()) => 0,
        Err(e) => {
            let code = e.exit_code();
            eprintln!("tdp_adaptive: {e}");
            eprintln!(
                "tdp_adaptive: exit {code} ({})",
                AdaptiveError::meaning(code)
            );
            code
        }
    };
    std::process::exit(code);
}

const USAGE: &str = "tdp_adaptive run --input G.mpmct1 --source-c C.g57 --n N --out-dir DIR [--sandwich S.mpmct1] [--seed S | --seed-file PATH] [--threads T] [--max-rounds 0] [--stop-fraction 0.02] [--seeds-per-block 2] [--time-limit 60] [--restarts 16] [--beam 6] [--term-cap 131072] [--max-block-gates 65536] [--segments 16] [--avoid-radius 64] [--refinement-levels 2] [--min-refine-gates 128] [--census-batches 96] [--probe-inputs 8192] [--accept smaller|smaller-not-leakier] [--blind] [--no-canonical-order] [--keep-round-circuits] [--no-postopt] [--ancilla-pool 32] [--compose-cap 1048576] [--compose-time-limit 300] [--equiv-batches 8] [--label tdp]
tdp_adaptive census --input G.mpmct1 --source-c C.g57 --n N --out-dir DIR [--census-batches 96] [--probe-inputs 8192] [--threads T] [--label tdp]
tdp_adaptive help";

// ---------------------------------------------------------------- argv

const SWITCHES: [&str; 5] = [
    "--blind",
    "--no-canonical-order",
    "--keep-round-circuits",
    "--no-postopt",
    "--help",
];

const RUN_OPTIONS: [&str; 32] = [
    "input",
    "source-c",
    "n",
    "out-dir",
    "sandwich",
    "seed",
    "seed-file",
    "threads",
    "max-rounds",
    "stop-fraction",
    "seeds-per-block",
    "time-limit",
    "restarts",
    "beam",
    "term-cap",
    "max-block-gates",
    "segments",
    "avoid-radius",
    "refinement-levels",
    "min-refine-gates",
    "census-batches",
    "probe-inputs",
    "accept",
    "blind",
    "no-canonical-order",
    "keep-round-circuits",
    "no-postopt",
    "ancilla-pool",
    "compose-cap",
    "compose-time-limit",
    "equiv-batches",
    "label",
];

const CENSUS_OPTIONS: [&str; 8] = [
    "input",
    "source-c",
    "n",
    "out-dir",
    "census-batches",
    "probe-inputs",
    "threads",
    "label",
];

struct Args {
    pos: Vec<String>,
    opts: Vec<(String, Option<String>)>,
}

impl Args {
    fn parse(a: &[String]) -> Result<Args, AdaptiveError> {
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
                        .ok_or_else(|| AdaptiveError::Usage(format!("{s} needs a value")))?;
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

    fn check_known(&self, known: &[&str]) -> Result<(), AdaptiveError> {
        for (k, _) in &self.opts {
            if !known.contains(&k.as_str()) {
                return Err(AdaptiveError::Usage(format!("unknown option --{k}")));
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

    fn num(&self, k: &str) -> Result<Option<u64>, AdaptiveError> {
        self.get(k).map(|v| parse_u64(v, k)).transpose()
    }

    fn usize_or(&self, k: &str, dflt: usize) -> Result<usize, AdaptiveError> {
        Ok(self.num(k)?.map(|v| v as usize).unwrap_or(dflt))
    }

    fn f64_or(&self, k: &str, dflt: f64) -> Result<f64, AdaptiveError> {
        match self.get(k) {
            None => Ok(dflt),
            Some(v) => v
                .parse::<f64>()
                .ok()
                .filter(|x| x.is_finite())
                .ok_or_else(|| AdaptiveError::Usage(format!("--{k} {v:?}: not a number"))),
        }
    }

    fn req(&self, k: &str) -> Result<&str, AdaptiveError> {
        self.get(k)
            .ok_or_else(|| AdaptiveError::Usage(format!("--{k} is required")))
    }
}

fn parse_u64(v: &str, what: &str) -> Result<u64, AdaptiveError> {
    if v.is_empty() || v.len() > 19 || !v.bytes().all(|b| b.is_ascii_digit()) {
        return Err(AdaptiveError::Usage(format!(
            "--{what}: {v:?} is not a nonnegative integer"
        )));
    }
    v.parse()
        .map_err(|_| AdaptiveError::Usage(format!("--{what}: {v:?} out of range")))
}

fn setup_threads(a: &Args) -> Result<usize, AdaptiveError> {
    if let Some(t) = a.num("threads")? {
        if t == 0 || t > 4096 {
            return Err(AdaptiveError::Usage("--threads must be in 1..=4096".into()));
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(t as usize)
            .build_global()
            .map_err(|e| AdaptiveError::Internal(format!("thread pool: {e}")))?;
    }
    Ok(rayon::current_num_threads())
}

/// The seed and where it came from (`argv`, `file` or `default`); the
/// value is never printed.
fn seed_of(a: &Args) -> Result<(u64, &'static str), AdaptiveError> {
    match (a.get("seed"), a.get("seed-file")) {
        (Some(_), Some(_)) => Err(AdaptiveError::Usage(
            "--seed and --seed-file are mutually exclusive".into(),
        )),
        (Some(s), None) => Ok((parse_u64(s, "seed")?, "argv")),
        (None, Some(p)) => Ok((
            crate::programs::read_seed_file(p).map_err(AdaptiveError::Usage)?,
            "file",
        )),
        (None, None) => Ok((1, "default")),
    }
}

fn label_of(a: &Args) -> Result<String, AdaptiveError> {
    let label = a.get("label").unwrap_or("tdp").to_string();
    if label.is_empty() || label.contains(['/', '\n']) {
        return Err(AdaptiveError::Usage(format!("--label {label:?}")));
    }
    Ok(label)
}

fn params_of(a: &Args) -> Result<DriverParams, AdaptiveError> {
    let mut p = DriverParams::default();
    p.max_rounds = a.usize_or("max-rounds", p.max_rounds)?;
    p.stop_fraction = a.f64_or("stop-fraction", p.stop_fraction)?;
    p.canonical_order = !a.flag("no-canonical-order");
    p.keep_round_circuits = a.flag("keep-round-circuits");
    p.probe_inputs = a.usize_or("probe-inputs", p.probe_inputs)?;
    let r = &mut p.round;
    r.seeds_per_block = a.usize_or("seeds-per-block", r.seeds_per_block)?;
    r.base_time_limit = a.f64_or("time-limit", r.base_time_limit)?;
    r.block.synth.time_limit = r.base_time_limit;
    r.block.synth.restarts = a.usize_or("restarts", r.block.synth.restarts)?;
    r.block.synth.beam = a.usize_or("beam", r.block.synth.beam)?;
    r.block.postopt = !a.flag("no-postopt");
    r.block.ancilla_pool = a.usize_or("ancilla-pool", r.block.ancilla_pool)?;
    r.block.compose_cap = a.usize_or("compose-cap", r.block.compose_cap)?;
    r.block.compose_time_limit = a.f64_or("compose-time-limit", r.block.compose_time_limit)?;
    r.plan.term_cap = a.usize_or("term-cap", r.plan.term_cap)?;
    r.plan.max_block_gates = a.usize_or("max-block-gates", r.plan.max_block_gates)?;
    r.plan.segments = a.usize_or("segments", r.plan.segments)?;
    r.plan.avoid_radius = a.usize_or("avoid-radius", r.plan.avoid_radius)?;
    r.refinement_levels = a.usize_or("refinement-levels", r.refinement_levels)?;
    r.min_refine_gates = a.usize_or("min-refine-gates", r.min_refine_gates)?;
    r.census_batches = a.usize_or("census-batches", r.census_batches)?;
    r.equiv_batches = a.usize_or("equiv-batches", r.equiv_batches)?;
    r.blind = a.flag("blind");
    if let Some(s) = a.get("accept") {
        r.accept = Accept::parse(s).ok_or_else(|| {
            AdaptiveError::Usage(format!("--accept {s:?}: smaller or smaller-not-leakier"))
        })?;
    }
    p.validate().map_err(AdaptiveError::Usage)?;
    Ok(p)
}

fn dispatch(argv: &[String]) -> Result<(), AdaptiveError> {
    let (cmd, rest) = argv
        .split_first()
        .ok_or_else(|| AdaptiveError::Usage(USAGE.lines().next().unwrap_or("").into()))?;
    let a = Args::parse(rest)?;
    match cmd.as_str() {
        "run" => cmd_run(&a),
        "census" => cmd_census(&a),
        "help" | "--help" | "-h" => {
            cmd_help();
            Ok(())
        }
        other => Err(AdaptiveError::Usage(format!("unknown command {other:?}"))),
    }
}

fn cmd_help() {
    println!("{USAGE}");
    println!("exit codes:");
    for (c, m) in EXIT_CODES {
        println!("  {c:3}  {m}");
    }
}

/// Every line to stdout, prefixed by the label.
fn stdout_log(label: &str) -> impl Fn(&str) + Sync + '_ {
    move |s: &str| {
        let stdout = io::stdout();
        let mut o = stdout.lock();
        let _ = writeln!(o, "{label}: {s}");
        let _ = o.flush();
    }
}

// ---------------------------------------------------------------- run

fn cmd_run(a: &Args) -> Result<(), AdaptiveError> {
    a.check_known(&RUN_OPTIONS)?;
    if a.flag("help") {
        cmd_help();
        return Ok(());
    }
    if !a.pos.is_empty() {
        return Err(AdaptiveError::Usage(
            USAGE.lines().next().unwrap_or("").into(),
        ));
    }
    let threads = setup_threads(a)?;
    let params = params_of(a)?;
    let (seed, seed_source) = seed_of(a)?;
    let label = label_of(a)?;
    let n = a
        .num("n")?
        .ok_or_else(|| AdaptiveError::Usage("--n is required".into()))? as usize;
    let inputs = DriverInputs {
        input: PathBuf::from(a.req("input")?),
        source_c: PathBuf::from(a.req("source-c")?),
        n,
        sandwich: a.get("sandwich").map(PathBuf::from),
        out_dir: PathBuf::from(a.req("out-dir")?),
        seed,
        threads,
    };
    let log = stdout_log(&label);
    let log: Log<'_> = &log;
    let rep = driver::run(&inputs, &params, seed_source, log)?;
    log(&format!(
        "done: {} -> {} gates in {} rounds ({}); {}",
        rep.input_gates,
        rep.final_gates,
        rep.rounds.len(),
        rep.stop_reason,
        inputs.out_dir.join("adaptive.report").display()
    ));
    Ok(())
}

// ---------------------------------------------------------------- census

fn cmd_census(a: &Args) -> Result<(), AdaptiveError> {
    a.check_known(&CENSUS_OPTIONS)?;
    if !a.pos.is_empty() {
        return Err(AdaptiveError::Usage(
            "census --input G --source-c C --n N --out-dir DIR [--census-batches B] [--probe-inputs P] [--threads T]".into(),
        ));
    }
    let threads = setup_threads(a)?;
    let label = label_of(a)?;
    let n = a
        .num("n")?
        .ok_or_else(|| AdaptiveError::Usage("--n is required".into()))? as usize;
    let batches = a.usize_or("census-batches", census::DEFAULT_BATCHES)?;
    let probe_inputs = a.usize_or(
        "probe-inputs",
        crate::stages::adaptive::boundary::DEFAULT_PROBE_INPUTS,
    )?;
    if batches == 0 || probe_inputs == 0 {
        return Err(AdaptiveError::Usage(
            "--census-batches and --probe-inputs must be positive".into(),
        ));
    }
    let input = PathBuf::from(a.req("input")?);
    let source_c = PathBuf::from(a.req("source-c")?);
    let out_dir = PathBuf::from(a.req("out-dir")?);
    let log = stdout_log(&label);
    let log: Log<'_> = &log;
    let t0 = Instant::now();
    let loaded = load_inputs(&input, &source_c, n)?;
    private_dir(&out_dir)?;
    log(&format!(
        "census: {} gates on {} wires, n={n}, C {} gates, {batches} batches; threads {threads}",
        loaded.gates.len(),
        loaded.width,
        loaded.c.len()
    ));
    let b = load_or_classify(
        &loaded.c,
        n,
        probe_inputs,
        &out_dir.join("boundary.txt"),
        log,
    )?;
    let source = source_of(loaded.c.clone(), n, &b)?;
    let c = census::census(&source, &loaded.gates, loaded.width, batches);
    let cdir = out_dir.join("census");
    if cdir.exists() {
        std::fs::remove_dir_all(&cdir).map_err(|e| AdaptiveError::io(&cdir, e))?;
    }
    std::fs::create_dir_all(&cdir).map_err(|e| AdaptiveError::io(&cdir, e))?;
    census::write_dir(&c, &cdir).map_err(|e| AdaptiveError::io(&cdir, e))?;
    let leak = Leakage::of(&c, &source);
    let mut s = String::new();
    let _ = writeln!(s, "adaptive_census_report=1");
    let _ = writeln!(s, "status=ok");
    let _ = writeln!(s, "tool=tdp_adaptive {}", env!("CARGO_PKG_VERSION"));
    let _ = writeln!(s, "input={}", input.display());
    let _ = writeln!(s, "source_c={}", source_c.display());
    let _ = writeln!(s, "n={n}");
    let _ = writeln!(s, "width={}", loaded.width);
    let _ = writeln!(s, "gates={}", c.gates);
    let _ = writeln!(s, "samples={}", c.samples);
    let _ = writeln!(s, "param.census_batches={batches}");
    let _ = writeln!(s, "param.probe_inputs={probe_inputs}");
    let _ = writeln!(s, "param.threads={threads}");
    let _ = writeln!(s, "boundary.internal={}", source.internal.len());
    let _ = writeln!(s, "boundary.excluded={}", source.excluded.len());
    let _ = writeln!(s, "boundary.deeper={}", source.deeper().len());
    let _ = writeln!(s, "affine_intervals={}", c.affine.len());
    let _ = writeln!(s, "direct_intervals={}", c.direct.len());
    let _ = writeln!(s, "audit.random_inputs={}", c.audit.random_inputs);
    let _ = writeln!(s, "audit.structured_inputs={}", c.audit.structured_inputs);
    let _ = writeln!(s, "audit.failed_direct={}", c.audit.failed_direct);
    let _ = writeln!(s, "audit.failed_affine={}", c.audit.failed_affine);
    leakage_kv(&mut s, "leakage", &leak);
    let _ = writeln!(s, "census_seconds={:.3}", c.seconds);
    let _ = writeln!(s, "seconds={:.3}", t0.elapsed().as_secs_f64());
    write_text_atomic(&out_dir.join("census.report"), &s)?;
    log(&format!(
        "census done: {} affine, {} direct intervals; deeper affine {} direct {} (validated); audit {}; {:.1}s -> {}",
        c.affine.len(),
        c.direct.len(),
        leak.validated.affine.deeper_births,
        leak.validated.direct.deeper_births,
        if leak.audit_passed {
            "passed"
        } else {
            "FAILED"
        },
        t0.elapsed().as_secs_f64(),
        out_dir.join("census.report").display()
    ));
    Ok(())
}
