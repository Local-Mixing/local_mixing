//! The adaptive stage end to end (port of `main` of the prototype): the
//! boundary classification of C, round after round of `round::run_round`
//! under the stop rules, the final circuit, its checks and its report.
//!
//! Stop rules (the prototype's): a round whose output is not smaller than
//! its input is STALLED (the next round re-runs on the same circuit with
//! fresh seeds); two stalled rounds end the run; so do two consecutive
//! rounds whose joint compression is below `stop_fraction`
//! (`last.output_gates >= (1 - f) * previous.input_gates`), and
//! `max_rounds` ("capped"). The final circuit is the smallest round
//! output, or the input when no round shrank.
//!
//! Resume: a round whose `result.report` exists is reloaded; its
//! `output.mpmct1` is read only when a later round or the final needs it.
//! `progress.txt` is rewritten after every round, `adaptive.log` is
//! appended to.
//!
//! Output directory:
//! ```text
//! DIR/boundary.txt            C's birth classification (cached)
//! DIR/round<r>/               see `round`
//! DIR/progress.txt            per-round gate counts and deeper births
//! DIR/final_census/           census of the published circuit
//! DIR/final.mpmct1            the published circuit
//! DIR/adaptive.report         the completion record (key=value)
//! DIR/adaptive.log            every progress line, appended across resumes
//! ```
use super::boundary;
use super::round::{
    AdaptiveError, Leakage, Log, RoundContext, RoundParams, RoundResult, leakage_kv,
    load_or_census, path_str, read_bounds_file, run_round, write_text_atomic,
};
use super::synth::same_function;
use super::types::Source;
use crate::canonicalization::gate_order::xgate_canonicalize;
use crate::circuit::formats::{read_g57_file, read_mpmct, write_mpmct_atomic};
use crate::circuit::xgate::XGate;
use crate::stages::packing::pyrand::PyRandom;
use crate::stages::packing::verify::{Lanes, eval_gates_lanes16, random_lanes};
use crate::stages::unpacking::types::MAX_STATE_WIRES;
use crate::stages::unpacking::verify::check_zero_slice_gates;
use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

pub const DEFAULT_STOP_FRACTION: f64 = 0.02;
/// Batches of 1024 random full states of the canonical-order check.
pub const FINAL_EQUIV_BATCHES: usize = 8;
pub const ZERO_SLICE_SAMPLES: usize = 4096;
/// Random inputs per direction of the payload contract (plus the
/// structured ones).
pub const PAYLOAD_RANDOM_INPUTS: usize = 8192;
/// Seed of the final sampled checks (public, fixed).
pub const CHECK_SEED: u64 = 20_261_026 ^ 0xF1F1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DriverInputs {
    /// The stage-5 circuit (mpmct1, wires `0..width`).
    pub input: PathBuf,
    /// C in g57 text (wires `0..n`).
    pub source_c: PathBuf,
    pub n: usize,
    /// The stage-1 sandwich (mpmct1, `2n` wires) for the payload contract.
    pub sandwich: Option<PathBuf>,
    pub out_dir: PathBuf,
    /// The stage seed (never printed).
    pub seed: u64,
    /// Reported only (the caller sized the thread pool).
    pub threads: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DriverParams {
    /// 0 = until the plateau rules fire.
    pub max_rounds: usize,
    pub stop_fraction: f64,
    pub round: RoundParams,
    /// Canonical gate order of the final circuit (checked equivalent).
    pub canonical_order: bool,
    /// Keep every round's `output.mpmct1` and the synthesis group caches
    /// (otherwise only the final round's output survives the run).
    pub keep_round_circuits: bool,
    /// Probe inputs of the boundary classification.
    pub probe_inputs: usize,
}

impl Default for DriverParams {
    fn default() -> Self {
        DriverParams {
            max_rounds: 0,
            stop_fraction: DEFAULT_STOP_FRACTION,
            round: RoundParams::default(),
            canonical_order: true,
            keep_round_circuits: false,
            probe_inputs: boundary::DEFAULT_PROBE_INPUTS,
        }
    }
}

impl DriverParams {
    pub fn validate(&self) -> Result<(), String> {
        self.round.validate()?;
        if self.max_rounds > 10_000 {
            return Err("max rounds: at most 10000".into());
        }
        if !(self.stop_fraction >= 0.0 && self.stop_fraction < 1.0) {
            return Err(format!(
                "stop fraction {}: must be in [0, 1)",
                self.stop_fraction
            ));
        }
        if self.probe_inputs == 0 {
            return Err("probe inputs must be positive".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BoundaryCounts {
    /// `|J|`: internal births of C.
    pub internal: usize,
    /// Public quadratic boundary births.
    pub excluded: usize,
    /// `|J \ excluded|`.
    pub deeper: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FinalChecks {
    /// Canonical circuit == un-ordered final (samples), when ordered.
    pub equiv_canonical: Option<usize>,
    pub zero_slice: usize,
    /// Payload contract inputs (both directions), when a sandwich was given.
    pub payload_contract: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FinalReport {
    pub input_gates: usize,
    pub final_gates: usize,
    /// The round whose output is published (`None`: the input).
    pub final_round: Option<usize>,
    pub rounds: Vec<RoundResult>,
    pub stop_reason: String,
    pub plateau: bool,
    pub final_leakage: Leakage,
    pub boundary: BoundaryCounts,
    pub checks: FinalChecks,
    pub seconds: f64,
}

// ---------------------------------------------------------------- stop rules

/// `stop_fraction * 100` without float noise (`2`, `0.5`).
pub fn percent(f: f64) -> String {
    let s = format!("{:.6}", f * 100.0);
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s.is_empty() { "0".into() } else { s.into() }
}

/// The prototype's stop rules after the latest round, over the rounds'
/// `(input_gates, output_gates)`; `stalled` is the running count of rounds
/// without a decrease. `Some(reason)` ends the run.
pub fn stop_rule(
    rounds: &[(usize, usize)],
    stalled: &mut usize,
    stop_fraction: f64,
) -> Option<String> {
    let &(inp, out) = rounds.last()?;
    if out >= inp {
        *stalled += 1;
    } else {
        *stalled = 0;
    }
    if *stalled >= 2 {
        return Some("two consecutive rounds without size decrease".into());
    }
    if rounds.len() >= 2 {
        let prev_in = rounds[rounds.len() - 2].0;
        if out as f64 >= (1.0 - stop_fraction) * prev_in as f64 {
            *stalled = 2;
            return Some(format!(
                "two consecutive rounds with joint compression below {}%",
                percent(stop_fraction)
            ));
        }
    }
    None
}

// ---------------------------------------------------------------- payload contract

/// One input of the payload contract (an `n`-bit pattern on wires `0..n`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pattern {
    Zero,
    Ones,
    Unit(usize),
    NotUnit(usize),
    Pair(usize, usize),
}

impl Pattern {
    fn bit(self, w: usize) -> bool {
        match self {
            Pattern::Zero => false,
            Pattern::Ones => true,
            Pattern::Unit(a) => w == a,
            Pattern::NotUnit(a) => w != a,
            Pattern::Pair(a, b) => w == a || w == b,
        }
    }
}

/// The structured inputs of `clean_payload.rs`: 0, all ones, every unit
/// vector and its complement, every pair of units.
fn structured_patterns(n: usize) -> Vec<Pattern> {
    let mut v = vec![Pattern::Zero, Pattern::Ones];
    for a in 0..n {
        v.push(Pattern::Unit(a));
        v.push(Pattern::NotUnit(a));
        for b in a + 1..n {
            v.push(Pattern::Pair(a, b));
        }
    }
    v
}

fn patterns_lanes(ps: &[Pattern], n: usize) -> Vec<Lanes> {
    let mut lanes = vec![[0u64; 16]; n];
    for (k, p) in ps.iter().enumerate() {
        for (w, lane) in lanes.iter_mut().enumerate() {
            if p.bit(w) {
                lane[k >> 6] |= 1u64 << (k & 63);
            }
        }
    }
    lanes
}

fn eval_dir(gates: &[XGate], st: &mut [Lanes], inverse: bool) {
    if inverse {
        for g in gates.iter().rev() {
            g.apply_lanes16(st);
        }
    } else {
        eval_gates_lanes16(gates, st);
    }
}

/// The stage-1 payload contract of `clean_payload.rs`: with x on wires
/// `0..n` and every other wire 0, the circuit leaves on wires `n..2n` what
/// the sandwich leaves there, forward and (on the reversed gate lists)
/// inverse; `PAYLOAD_RANDOM_INPUTS` random inputs plus the structured
/// ones per direction. Returns the number of inputs checked.
pub fn payload_contract(
    sandwich: &[XGate],
    sandwich_width: usize,
    circuit: &[XGate],
    width: usize,
    n: usize,
) -> Result<usize, AdaptiveError> {
    let sw = sandwich_width.max(2 * n);
    if n == 0 || 2 * n > width {
        return Err(AdaptiveError::Usage(format!(
            "payload contract: n={n} does not fit a {width}-wire state"
        )));
    }
    if let Some(w) = sandwich.iter().map(|g| g.max_wire() as usize).max()
        && w >= sw
    {
        return Err(AdaptiveError::Input(format!(
            "the sandwich uses wire {w}, beyond its {sw}-wire state"
        )));
    }
    let structured = structured_patterns(n);
    let random_batches = PAYLOAD_RANDOM_INPUTS.div_ceil(1024);
    let mut checked = 0usize;
    for inverse in [false, true] {
        let mut rng = PyRandom::new(CHECK_SEED ^ 0x2026_1019 ^ inverse as u64);
        let mut batches: Vec<Vec<Lanes>> = Vec::new();
        for _ in 0..random_batches {
            batches.push(random_lanes(&mut rng, n));
        }
        for chunk in structured.chunks(1024) {
            batches.push(patterns_lanes(chunk, n));
        }
        for (bi, x) in batches.iter().enumerate() {
            let mut want = x.clone();
            want.resize(sw, [0; 16]);
            eval_dir(sandwich, &mut want, inverse);
            let mut got = x.clone();
            got.resize(width, [0; 16]);
            eval_dir(circuit, &mut got, inverse);
            if want[n..2 * n] != got[n..2 * n] {
                return Err(AdaptiveError::Verify(format!(
                    "payload contract: wires {n}..{} differ from the sandwich's ({}, batch {bi})",
                    2 * n,
                    if inverse { "inverse" } else { "forward" }
                )));
            }
            checked += if bi < random_batches {
                1024
            } else {
                structured.len() - (bi - random_batches) * 1024
            }
            .min(1024);
        }
    }
    Ok(checked)
}

// ---------------------------------------------------------------- the run

/// The circuit a round runs on: in memory, or on disk until needed.
struct Current {
    gates: Option<Vec<XGate>>,
    path: PathBuf,
    round: Option<usize>,
}

impl Current {
    fn load(&mut self, width: usize) -> Result<&[XGate], AdaptiveError> {
        if self.gates.is_none() {
            let (g, w) = read_mpmct(path_str(&self.path)?).map_err(|e| {
                AdaptiveError::Input(format!(
                    "{}: {e} (a completed run removes the outputs of non-final rounds; resume needs them or a fresh --out-dir)",
                    self.path.display()
                ))
            })?;
            if w != width {
                return Err(AdaptiveError::Input(format!(
                    "{}: {w} wires, the input has {width}",
                    self.path.display()
                )));
            }
            self.gates = Some(g);
        }
        Ok(self.gates.as_deref().expect("loaded"))
    }
}

/// Create `dir` (mode 0700) unless it exists as a directory.
pub fn private_dir(dir: &Path) -> Result<(), AdaptiveError> {
    if !dir.exists() {
        fs::create_dir_all(dir).map_err(|e| AdaptiveError::io(dir, e))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(dir, fs::Permissions::from_mode(0o700))
                .map_err(|e| AdaptiveError::io(dir, e))?;
        }
    } else if !dir.is_dir() {
        return Err(AdaptiveError::Usage(format!(
            "--out-dir {} is not a directory",
            dir.display()
        )));
    }
    Ok(())
}

/// The stage inputs, validated.
pub struct Loaded {
    pub gates: Vec<XGate>,
    pub width: usize,
    pub c: Vec<XGate>,
}

/// Read the circuit and C and validate them against `n`.
pub fn load_inputs(input: &Path, source_c: &Path, n: usize) -> Result<Loaded, AdaptiveError> {
    let (gates, width) = read_mpmct(path_str(input)?)
        .map_err(|e| AdaptiveError::Input(format!("{}: {e}", input.display())))?;
    if n == 0 || 2 * n > width {
        return Err(AdaptiveError::Usage(format!(
            "--n {n}: the circuit has {width} wires (at least 2n needed)"
        )));
    }
    if width > MAX_STATE_WIRES {
        return Err(AdaptiveError::Usage(format!(
            "{}: {width} wires; the stage supports at most {MAX_STATE_WIRES}",
            input.display()
        )));
    }
    if let Some(w) = gates.iter().map(|g| g.max_wire() as usize).max()
        && w >= width
    {
        return Err(AdaptiveError::Input(format!(
            "{}: gate on wire {w}, beyond the {width}-wire header",
            input.display()
        )));
    }
    let c = read_g57_file(path_str(source_c)?)
        .map_err(|e| AdaptiveError::Input(format!("{}: {e}", source_c.display())))?;
    if c.is_empty() {
        return Err(AdaptiveError::Input(format!(
            "{}: C has no gates",
            source_c.display()
        )));
    }
    if let Some(w) = c.iter().map(|g| g.max_wire() as usize).max()
        && w >= n
    {
        return Err(AdaptiveError::Input(format!(
            "{}: C uses wire {w}, beyond n = {n}",
            source_c.display()
        )));
    }
    Ok(Loaded { gates, width, c })
}

/// The boundary classification of C, cached at `path`.
pub fn load_or_classify(
    c: &[XGate],
    n: usize,
    probe_inputs: usize,
    path: &Path,
    log: Log<'_>,
) -> Result<boundary::Boundary, AdaptiveError> {
    if path.is_file()
        && let Ok(b) = boundary::read_file(path)
        && b.rows.len() == c.len()
    {
        log(&format!(
            "boundary classification reloaded from {}: {} excluded of {} births",
            path.display(),
            b.excluded.len(),
            b.rows.len()
        ));
        return Ok(b);
    }
    let t0 = Instant::now();
    let b = boundary::classify(c, n, probe_inputs);
    boundary::write_file(&b, path).map_err(|e| AdaptiveError::io(path, e))?;
    log(&format!(
        "boundary classification: {} excluded (public quadratic) of {} births, {} probe inputs ({:.1}s)",
        b.excluded.len(),
        b.rows.len(),
        b.probe_inputs,
        t0.elapsed().as_secs_f64()
    ));
    Ok(b)
}

/// `Source::new` from the boundary, checked against C's own output set.
pub fn source_of(c: Vec<XGate>, n: usize, b: &boundary::Boundary) -> Result<Source, AdaptiveError> {
    let source = Source::new(c, n, b.excluded.clone());
    let internal_rows: BTreeSet<usize> = b
        .rows
        .iter()
        .filter(|r| !r.is_output)
        .map(|r| r.source_gate)
        .collect();
    if internal_rows != source.internal {
        return Err(AdaptiveError::Input(
            "the boundary classification's internal births differ from C's (stale boundary.txt?)"
                .into(),
        ));
    }
    Ok(source)
}

fn progress_text(rounds: &[RoundResult]) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "adaptive_progress=1");
    let _ = writeln!(s, "rounds={}", rounds.len());
    for r in rounds {
        let _ = writeln!(s, "round.{}.input_gates={}", r.round, r.input_gates);
        let _ = writeln!(s, "round.{}.output_gates={}", r.round, r.output_gates);
        let _ = writeln!(
            s,
            "round.{}.deeper_affine={}",
            r.round, r.leakage_out.validated.affine.deeper_births
        );
        let _ = writeln!(
            s,
            "round.{}.deeper_direct={}",
            r.round, r.leakage_out.validated.direct.deeper_births
        );
    }
    let _ = writeln!(
        s,
        "updated={}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs())
    );
    s
}

/// `adaptive.report`.
pub fn final_report_text(
    rep: &FinalReport,
    inputs: &DriverInputs,
    params: &DriverParams,
    width: usize,
    seed_source: &str,
) -> String {
    let p = &params.round;
    let mut s = String::new();
    let _ = writeln!(s, "adaptive_report=1");
    let _ = writeln!(s, "status=ok");
    let _ = writeln!(s, "tool=tdp_adaptive {}", env!("CARGO_PKG_VERSION"));
    let _ = writeln!(s, "input={}", inputs.input.display());
    let _ = writeln!(s, "source_c={}", inputs.source_c.display());
    let _ = writeln!(
        s,
        "sandwich={}",
        inputs
            .sandwich
            .as_ref()
            .map_or("none".to_string(), |p| p.display().to_string())
    );
    let _ = writeln!(s, "n={}", inputs.n);
    let _ = writeln!(s, "width={width}");
    let _ = writeln!(s, "input_gates={}", rep.input_gates);
    let _ = writeln!(s, "final_gates={}", rep.final_gates);
    let _ = writeln!(
        s,
        "final_round={}",
        rep.final_round
            .map_or("input".to_string(), |r| r.to_string())
    );
    let _ = writeln!(s, "rounds={}", rep.rounds.len());
    let _ = writeln!(s, "stop_reason={}", rep.stop_reason);
    let _ = writeln!(s, "plateau={}", rep.plateau);
    let _ = writeln!(s, "param.seed_source={seed_source}");
    let _ = writeln!(s, "param.threads={}", inputs.threads);
    let _ = writeln!(s, "param.max_rounds={}", params.max_rounds);
    let _ = writeln!(s, "param.stop_fraction={}", params.stop_fraction);
    let _ = writeln!(s, "param.seeds_per_block={}", p.seeds_per_block);
    let _ = writeln!(s, "param.time_limit={}", p.base_time_limit);
    let _ = writeln!(s, "param.restarts={}", p.block.synth.restarts);
    let _ = writeln!(s, "param.beam={}", p.block.synth.beam);
    let _ = writeln!(s, "param.postopt={}", p.block.postopt);
    let _ = writeln!(s, "param.ancilla_pool={}", p.block.ancilla_pool);
    let _ = writeln!(s, "param.compose_cap={}", p.block.compose_cap);
    let _ = writeln!(s, "param.verify_batches={}", p.block.verify_batches);
    let _ = writeln!(s, "param.term_cap={}", p.plan.term_cap);
    let _ = writeln!(s, "param.max_block_gates={}", p.plan.max_block_gates);
    let _ = writeln!(s, "param.lookahead={}", p.plan.lookahead);
    let _ = writeln!(s, "param.product_pairs={}", p.plan.product_pairs);
    let _ = writeln!(s, "param.segments={}", p.plan.segments);
    let _ = writeln!(s, "param.avoid_radius={}", p.plan.avoid_radius);
    let _ = writeln!(s, "param.refinement_levels={}", p.refinement_levels);
    let _ = writeln!(s, "param.min_refine_gates={}", p.min_refine_gates);
    let _ = writeln!(s, "param.census_batches={}", p.census_batches);
    let _ = writeln!(s, "param.probe_inputs={}", params.probe_inputs);
    let _ = writeln!(s, "param.accept={}", p.accept.name());
    let _ = writeln!(s, "param.blind={}", p.blind);
    let _ = writeln!(s, "param.equiv_batches={}", p.equiv_batches);
    let _ = writeln!(s, "param.canonical_order={}", params.canonical_order);
    let _ = writeln!(
        s,
        "param.keep_round_circuits={}",
        params.keep_round_circuits
    );
    let _ = writeln!(s, "boundary.internal={}", rep.boundary.internal);
    let _ = writeln!(s, "boundary.excluded={}", rep.boundary.excluded);
    let _ = writeln!(s, "boundary.deeper={}", rep.boundary.deeper);
    for r in &rep.rounds {
        let k = format!("round.{}", r.round);
        let _ = writeln!(s, "{k}.input_gates={}", r.input_gates);
        let _ = writeln!(s, "{k}.output_gates={}", r.output_gates);
        let _ = writeln!(s, "{k}.blocks={}", r.blocks);
        let _ = writeln!(s, "{k}.initial_blocks={}", r.initial_blocks);
        let _ = writeln!(
            s,
            "{k}.synthesized_blocks={}",
            r.coverage.synthesized_blocks
        );
        let _ = writeln!(s, "{k}.picks_orig={}", r.picks.orig);
        let _ = writeln!(
            s,
            "{k}.picks_seed={}",
            r.picks
                .seed
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        let _ = writeln!(s, "{k}.rejected_leakier={}", r.picks.rejected_leakier);
        let _ = writeln!(s, "{k}.rejected_growing={}", r.picks.rejected_growing);
        let _ = writeln!(
            s,
            "{k}.deeper_affine={}",
            r.leakage_out.validated.affine.deeper_births
        );
        let _ = writeln!(
            s,
            "{k}.deeper_direct={}",
            r.leakage_out.validated.direct.deeper_births
        );
        let _ = writeln!(s, "{k}.reloaded={}", r.reloaded);
        let _ = writeln!(s, "{k}.seconds={:.3}", r.seconds);
    }
    if let Some(first) = rep.rounds.first() {
        leakage_kv(&mut s, "stage5_leakage", &first.leakage_in);
    }
    leakage_kv(&mut s, "final_leakage", &rep.final_leakage);
    let okn = |v: Option<usize>| v.map_or("skipped".to_string(), |x| format!("ok({x})"));
    let _ = writeln!(
        s,
        "check.equiv_canonical={}",
        okn(rep.checks.equiv_canonical)
    );
    let _ = writeln!(s, "check.zero_slice=ok({})", rep.checks.zero_slice);
    let _ = writeln!(
        s,
        "check.payload_contract={}",
        okn(rep.checks.payload_contract)
    );
    let _ = writeln!(s, "seconds={:.3}", rep.seconds);
    s
}

/// Remove the circuits a completed run no longer needs: every synthesis
/// group cache, and `output.mpmct1` of the rounds other than `final_round`.
fn prune_round_circuits(out_dir: &Path, rounds: &[RoundResult], final_round: Option<usize>) {
    for r in rounds {
        let rdir = out_dir.join(format!("round{}", r.round));
        if Some(r.round) != final_round {
            let _ = fs::remove_file(rdir.join("output.mpmct1"));
        }
        let Ok(rd) = fs::read_dir(&rdir) else {
            continue;
        };
        for e in rd.flatten() {
            let name = e.file_name();
            let name = name.to_string_lossy();
            if (name.starts_with("synth") || name.starts_with("refine"))
                && e.path().is_dir()
                && let Ok(files) = fs::read_dir(e.path())
            {
                for f in files.flatten() {
                    if f.path().extension().is_some_and(|x| x == "mpmct1") {
                        let _ = fs::remove_file(f.path());
                    }
                }
            }
        }
    }
}

/// Run the stage. `seed_source` names where the seed came from (`argv`,
/// `file`, `default`) for the report; the value is never written.
pub fn run(
    inputs: &DriverInputs,
    params: &DriverParams,
    seed_source: &str,
    log: Log<'_>,
) -> Result<FinalReport, AdaptiveError> {
    let t0 = Instant::now();
    params.validate().map_err(AdaptiveError::Usage)?;
    private_dir(&inputs.out_dir)?;
    let log_path = inputs.out_dir.join("adaptive.log");
    let log_file = Mutex::new(
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .map_err(|e| AdaptiveError::io(&log_path, e))?,
    );
    let say = |s: &str| {
        log(s);
        if let Ok(mut f) = log_file.lock() {
            let _ = f.write_all(s.as_bytes());
            let _ = f.write_all(b"\n");
        }
    };
    let say: Log<'_> = &say;

    // ---- inputs
    let Loaded { gates, width, c } = load_inputs(&inputs.input, &inputs.source_c, inputs.n)?;
    let n = inputs.n;
    let sandwich = match &inputs.sandwich {
        Some(p) => {
            let (sg, sw) = read_mpmct(path_str(p)?)
                .map_err(|e| AdaptiveError::Input(format!("{}: {e}", p.display())))?;
            if sw < 2 * n || sw > width {
                return Err(AdaptiveError::Input(format!(
                    "{}: the sandwich has {sw} wires, expected 2n = {} (at most the circuit's {width})",
                    p.display(),
                    2 * n
                )));
            }
            Some((sg, sw))
        }
        None => None,
    };
    say(&format!(
        "tdp_adaptive: {} gates on {width} wires, n={n}, C {} gates; sandwich {}; seed ({seed_source}); threads {}; max rounds {}, stop fraction {}",
        gates.len(),
        c.len(),
        sandwich
            .as_ref()
            .map_or("none".to_string(), |(g, _)| format!("{} gates", g.len())),
        inputs.threads,
        if params.max_rounds == 0 {
            "plateau".to_string()
        } else {
            params.max_rounds.to_string()
        },
        params.stop_fraction
    ));
    let b = load_or_classify(
        &c,
        n,
        params.probe_inputs,
        &inputs.out_dir.join("boundary.txt"),
        say,
    )?;
    let source = source_of(c.clone(), n, &b)?;
    let boundary_counts = BoundaryCounts {
        internal: source.internal.len(),
        excluded: source.excluded.len(),
        deeper: source.deeper().len(),
    };
    let ctx = RoundContext {
        source: &source,
        width,
        stage_seed: inputs.seed,
        params: &params.round,
    };

    // ---- rounds
    let input_gates = gates.len();
    let mut current = Current {
        gates: Some(gates),
        path: inputs.input.clone(),
        round: None,
    };
    let mut rounds: Vec<RoundResult> = Vec::new();
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    let mut prev_bounds: Vec<usize> = Vec::new();
    let mut stalled = 0usize;
    let mut stop_reason = String::from("round cap");
    let mut r = 0usize;
    loop {
        r += 1;
        if params.max_rounds > 0 && r > params.max_rounds {
            break;
        }
        let rdir = inputs.out_dir.join(format!("round{r}"));
        let g = current.load(width)?;
        let mut res = run_round(&ctx, r, g, &prev_bounds, &rdir, say)?;
        let output = res.output.take();
        pairs.push((res.input_gates, res.output_gates));
        let shrank = res.output_gates < res.input_gates;
        if shrank {
            prev_bounds = if res.reloaded {
                read_bounds_file(&rdir.join("bounds.out"))?
            } else {
                res.output_boundaries.clone()
            };
        }
        rounds.push(res);
        write_text_atomic(
            &inputs.out_dir.join("progress.txt"),
            &progress_text(&rounds),
        )?;
        if shrank {
            current = Current {
                gates: output,
                path: rdir.join("output.mpmct1"),
                round: Some(r),
            };
        }
        if let Some(reason) = stop_rule(&pairs, &mut stalled, params.stop_fraction) {
            stop_reason = reason;
            break;
        }
    }
    let plateau = stalled >= 2;
    say(&format!(
        "rounds done: {} rounds, {input_gates} -> {} gates ({})",
        rounds.len(),
        pairs
            .iter()
            .map(|p| p.1.min(p.0))
            .min()
            .unwrap_or(input_gates),
        stop_reason
    ));

    // ---- the final circuit and its checks
    let final_round = current.round;
    let mut final_gates: Vec<XGate> = current.load(width)?.to_vec();
    current.gates = None;
    let mut checks = FinalChecks::default();
    if params.canonical_order {
        let t = Instant::now();
        let mut canon = final_gates.clone();
        xgate_canonicalize(&mut canon);
        if !same_function(
            &final_gates,
            &canon,
            width,
            FINAL_EQUIV_BATCHES,
            CHECK_SEED ^ 0xCA,
        ) {
            return Err(AdaptiveError::Verify(
                "the canonically ordered final circuit differs from the assembled one".into(),
            ));
        }
        checks.equiv_canonical = Some(FINAL_EQUIV_BATCHES * 1024);
        final_gates = canon;
        say(&format!(
            "canonical order: {} gates, equivalence ok ({} samples, {:.1}s)",
            final_gates.len(),
            FINAL_EQUIV_BATCHES * 1024,
            t.elapsed().as_secs_f64()
        ));
    }
    checks.zero_slice = check_zero_slice_gates(
        &c,
        &final_gates,
        n,
        width,
        ZERO_SLICE_SAMPLES,
        CHECK_SEED ^ 0x5A5A,
    )?;
    say(&format!(
        "zero slice vs C: ok ({} samples)",
        checks.zero_slice
    ));
    if let Some((sg, sw)) = &sandwich {
        let k = payload_contract(sg, *sw, &final_gates, width, n)?;
        checks.payload_contract = Some(k);
        say(&format!(
            "payload contract vs the sandwich: ok ({k} inputs over both directions)"
        ));
    }
    let fc = load_or_census(
        &ctx,
        &final_gates,
        &inputs.out_dir.join("final_census"),
        say,
    )?;
    let final_leakage = Leakage::of(&fc, &source);
    drop(fc);
    let rep = FinalReport {
        input_gates,
        final_gates: final_gates.len(),
        final_round,
        rounds,
        stop_reason,
        plateau,
        final_leakage,
        boundary: boundary_counts,
        checks,
        seconds: t0.elapsed().as_secs_f64(),
    };
    let fpath = inputs.out_dir.join("final.mpmct1");
    write_mpmct_atomic(path_str(&fpath)?, &final_gates, width)
        .map_err(|e| AdaptiveError::io(&fpath, e))?;
    drop(final_gates);
    if !params.keep_round_circuits {
        prune_round_circuits(&inputs.out_dir, &rep.rounds, final_round);
    }
    say(&format!(
        "final: {} gates (round {}), deeper affine {} direct {} (validated), {} ({:.1}s)",
        rep.final_gates,
        final_round.map_or("input".to_string(), |r| r.to_string()),
        rep.final_leakage.validated.affine.deeper_births,
        rep.final_leakage.validated.direct.deeper_births,
        if plateau { "plateau" } else { "capped" },
        rep.seconds
    ));
    write_text_atomic(
        &inputs.out_dir.join("adaptive.report"),
        &final_report_text(&rep, inputs, params, width, seed_source),
    )?;
    Ok(rep)
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/adaptive/driver.rs"]
mod tests;
