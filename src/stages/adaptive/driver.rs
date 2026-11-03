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
//! Resume: a round whose `result.report` exists is reloaded from the
//! report alone; a round that must run reads the current circuit (the
//! input, or the last shrinking round's `output.mpmct1`, which a partial
//! run always still has). A COMPLETED run prunes the non-final round
//! outputs and the synthesis caches (unless `keep_round_circuits`), so a
//! rerun on a completed out-dir reloads every round, takes the published
//! `final.mpmct1` as the final circuit (gate count checked against the
//! final round's report), re-runs the zero-slice and payload checks on it
//! and republishes it unchanged (`final_reloaded=true`,
//! `check.equiv_canonical=reloaded`). `progress.txt` is rewritten after
//! every round, `adaptive.log` is appended to.
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
use super::census::{AUTO_BATCHES, batches_for_width};
use super::round::{
    Accept, AdaptiveError, Cost, EARLY_OUTPUT_FRAC, Leakage, Log, RoundContext, RoundParams,
    RoundResult, gates_digest, leakage_kv, load_or_census, parse_kv, path_str, read_bounds_file,
    reload_census, reload_round, run_round, write_text_atomic,
};
use super::synth::same_function;
use super::types::{Census, Source};
use crate::canonicalization::gate_order::xgate_canonicalize;
use crate::circuit::formats::{read_g57_file, read_mpmct, write_mpmct_atomic};
use crate::circuit::xgate::XGate;
use crate::stages::packing::pyrand::PyRandom;
use crate::stages::packing::verify::{Lanes, eval_gates_lanes16, random_lanes};
use crate::stages::post_processing::compression::{
    CompressParams, HelperChoice, compress, lower_wide_gates, lower_wide_gates_with, push_nots,
};
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
    /// Check the payload contract forward only (`--forward-only`): the
    /// stage-2 circuit was built without the reverse contract (opening
    /// guard over all data wires, no closing guard), so its inverse junks
    /// the reverse payload by design.
    pub forward_only: bool,
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
    /// Phase B (the multi-scale descent): block-gate caps run after the
    /// large-block rounds plateau, each to its own stall, with an exact
    /// NOT-push + fcompress cleanup between them. Empty = phase A only (the
    /// classic behaviour). Caps are planned C-blind (the leak guard still
    /// vetoes a leakier block); compression, not exposure, is their job.
    pub descent: Vec<usize>,
    /// The per-cap stall threshold of phase B (joint two-round compression
    /// below this ends a cap and moves to the next smaller one).
    pub descent_stop_fraction: f64,
    /// fcompress iterations in the between-cap cleanup (0 = NOT-push only).
    pub cleanup_fcompress_iters: usize,
    /// Run one exact cleanup (NOT-push + fcompress) on the input BEFORE
    /// phase A. On the n=64 stage-5 output it removed 63% in 78 s; the ANF
    /// rounds then work on a far smaller circuit (and fcompress acts on the
    /// crossing walk's local redundancy before the synthesizer rewrites it).
    pub pre_cleanup: bool,
    /// Phase C: after the descent, rewrite every gate wider than this many
    /// controls into 2-control gates (the exact dirty-ancilla Barenco
    /// decomposition) and run one more exact cleanup, so the published
    /// circuit is in the deliverable's gate set. `None` = leave stage 5's
    /// wide gates in place (the classic behaviour).
    pub lower_controls: Option<usize>,
    /// Canonical gate order of the final circuit (checked equivalent).
    pub canonical_order: bool,
    /// Keep every round's `output.mpmct1` and the synthesis group caches
    /// (otherwise only the final round's output survives the run).
    pub keep_round_circuits: bool,
    /// Probe inputs of the boundary classification.
    pub probe_inputs: usize,
    /// `round.census_batches` was given as `auto` ([`AUTO_BATCHES`]) and
    /// resolved by the census batch rule ([`batches_for_width`]) once the
    /// input's width was known.
    pub census_batches_auto: bool,
}

impl Default for DriverParams {
    fn default() -> Self {
        DriverParams {
            max_rounds: 0,
            stop_fraction: DEFAULT_STOP_FRACTION,
            round: RoundParams::default(),
            descent: Vec::new(),
            descent_stop_fraction: 0.01,
            cleanup_fcompress_iters: 10,
            pre_cleanup: false,
            lower_controls: None,
            canonical_order: true,
            keep_round_circuits: false,
            probe_inputs: boundary::DEFAULT_PROBE_INPUTS,
            census_batches_auto: false,
        }
    }
}

impl DriverParams {
    pub fn validate(&self) -> Result<(), String> {
        // `auto` census batches are resolved from the input's width in `run`
        let mut round = self.round.clone();
        if round.census_batches == AUTO_BATCHES {
            round.census_batches = 1;
        }
        round.validate()?;
        if self.max_rounds > 10_000 {
            return Err("max rounds: at most 10000".into());
        }
        if !(self.stop_fraction >= 0.0 && self.stop_fraction < 1.0) {
            return Err(format!(
                "stop fraction {}: must be in [0, 1)",
                self.stop_fraction
            ));
        }
        if !(self.descent_stop_fraction >= 0.0 && self.descent_stop_fraction < 1.0) {
            return Err(format!(
                "descent stop fraction {}: must be in [0, 1)",
                self.descent_stop_fraction
            ));
        }
        if self.descent.iter().any(|&c| c == 0) {
            return Err("descent caps must be positive".into());
        }
        if self.descent.windows(2).any(|w| w[0] <= w[1]) {
            return Err("descent caps must be strictly decreasing".into());
        }
        if self.lower_controls.is_some_and(|c| c < 2) {
            return Err("lower controls: at least 2".into());
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
    /// The INPUT circuit's zero slice against C (samples), checked before
    /// round 1 so that a wrong `--n`/`--source-c` fails in seconds.
    pub input_zero_slice: usize,
    /// The INPUT circuit's payload contract (inputs), when a sandwich was
    /// given.
    pub input_payload_contract: Option<usize>,
    /// Canonical circuit == un-ordered final (samples), when ordered.
    pub equiv_canonical: Option<usize>,
    /// The final circuit was reloaded from `final.mpmct1` (already in
    /// canonical order) and republished unchanged: the order check of
    /// the run that wrote it stands (`check.equiv_canonical=reloaded`).
    pub canonical_reloaded: bool,
    /// Whether the canonical order was published: `Some(false)` when the
    /// reordered circuit exposed more than the circuit it reorders (validated
    /// deeper births, affine or direct, or early C outputs: [`Exposure`]),
    /// or with `--no-canonical-order`; `None` when a reloaded final's
    /// previous report does not say.
    pub canonical_applied: Option<bool>,
    pub canonical_reason: String,
    /// Phase C's leak guard: `kept` (the lowering and cleanup exposed no
    /// more than phase C's input), `random_helpers:<k>` (it did, and the
    /// k-th lowering with random helper wires did not), `leakier` (no
    /// lowering avoided it: the least exposed one is published and the
    /// report's status is `leakier_lowering`), `off` (guard off); `None`
    /// when phase C did not run.
    pub phase_c_guard: Option<String>,
    pub zero_slice: usize,
    /// Payload contract inputs (both directions, or forward only), when a
    /// sandwich was given.
    pub payload_contract: Option<usize>,
    /// The payload contract was checked forward only (`--forward-only`).
    pub payload_forward_only: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FinalReport {
    pub input_gates: usize,
    pub final_gates: usize,
    /// The round whose output is published (`None`: the input).
    pub final_round: Option<usize>,
    /// The final circuit came from `final.mpmct1` of a completed run (every
    /// round reloaded, the round outputs pruned).
    pub final_reloaded: bool,
    pub rounds: Vec<RoundResult>,
    pub stop_reason: String,
    pub plateau: bool,
    pub final_leakage: Leakage,
    /// C outputs affinely exposed before EARLY_OUTPUT_FRAC of the published
    /// circuit (a copy of an output deep before its delivery region).
    pub final_early_outputs: usize,
    pub boundary: BoundaryCounts,
    pub checks: FinalChecks,
    pub seconds: f64,
}

impl FinalReport {
    /// `ok`, or `leakier_lowering` when phase C could not lower the wide
    /// gates without exposing more than its input (the published circuit is
    /// then not recommended: tdp_gen.sh records an allowed failure).
    pub fn status(&self) -> &'static str {
        if self.checks.phase_c_guard.as_deref() == Some("leakier") {
            "leakier_lowering"
        } else {
            "ok"
        }
    }
}

/// What the leak guard compares, from a validated census: deeper births of C
/// exposed affinely and directly, and C output bits affinely exposed before
/// [`EARLY_OUTPUT_FRAC`] of the circuit. Ordered early outputs first, so the
/// minimum is the least exposed circuit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Exposure {
    pub early_outputs: usize,
    pub deeper_affine: usize,
    pub deeper_direct: usize,
}

impl Exposure {
    pub fn of(census: &Census, source: &Source) -> Exposure {
        let l = Leakage::of(census, source);
        Exposure {
            early_outputs: census.early_output_count(EARLY_OUTPUT_FRAC),
            deeper_affine: l.validated.affine.deeper_births,
            deeper_direct: l.validated.direct.deeper_births,
        }
    }

    /// Any of the three counts above `base`'s: the guarded step is leakier.
    pub fn exceeds(&self, base: &Exposure) -> bool {
        self.deeper_affine > base.deeper_affine
            || self.deeper_direct > base.deeper_direct
            || self.early_outputs > base.early_outputs
    }
}

impl std::fmt::Display for Exposure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "deeper affine {}, direct {}, early outputs {}",
            self.deeper_affine, self.deeper_direct, self.early_outputs
        )
    }
}

/// Copy a census directory (flat: interval tables, summary, audit, stamp).
fn copy_census_dir(src: &Path, dst: &Path) -> Result<(), AdaptiveError> {
    let _ = fs::remove_dir_all(dst);
    fs::create_dir_all(dst).map_err(|e| AdaptiveError::io(dst, e))?;
    for entry in fs::read_dir(src).map_err(|e| AdaptiveError::io(src, e))? {
        let entry = entry.map_err(|e| AdaptiveError::io(src, e))?;
        let from = entry.path();
        if from.is_file() {
            let to = dst.join(entry.file_name());
            fs::copy(&from, &to).map_err(|e| AdaptiveError::io(&to, e))?;
        }
    }
    Ok(())
}

/// Lowerings with random helper wires phase C tries when the plain one is
/// leakier than its input.
const PHASE_C_HELPER_ATTEMPTS: u64 = 8;

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
    payload_contract_dirs(sandwich, sandwich_width, circuit, width, n, true)
}

/// [`payload_contract`] with the inverse direction optional: `inverse`
/// false checks the forward payload only (a stage 2 built without the
/// reverse contract).
pub fn payload_contract_dirs(
    sandwich: &[XGate],
    sandwich_width: usize,
    circuit: &[XGate],
    width: usize,
    n: usize,
    inverse: bool,
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
    let directions: &[bool] = if inverse { &[false, true] } else { &[false] };
    for &inverse in directions {
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
    /// Gate count, known from the reports without reading the circuit.
    count: usize,
    /// Size under the run's cost model (= `count` under `Cost::Gates`); the
    /// shrink and stop rules compare this.
    cost: usize,
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
    let _ = writeln!(s, "status={}", rep.status());
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
    let _ = writeln!(
        s,
        "param.descent={}",
        params
            .descent
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    let _ = writeln!(
        s,
        "param.descent_stop_fraction={}",
        params.descent_stop_fraction
    );
    let _ = writeln!(
        s,
        "param.cleanup_fcompress_iters={}",
        params.cleanup_fcompress_iters
    );
    let _ = writeln!(
        s,
        "param.lower_controls={}",
        params
            .lower_controls
            .map_or_else(|| "none".to_string(), |c| c.to_string())
    );
    let _ = writeln!(s, "param.cost={}", p.cost.name());
    let _ = writeln!(s, "param.pre_cleanup={}", params.pre_cleanup);
    let _ = writeln!(s, "param.seeds_per_block={}", p.seeds_per_block);
    let _ = writeln!(s, "param.time_limit={}", p.base_time_limit);
    let _ = writeln!(s, "param.restarts={}", p.block.synth.restarts);
    let _ = writeln!(s, "param.beam={}", p.block.synth.beam);
    let _ = writeln!(s, "param.postopt={}", p.block.postopt);
    let _ = writeln!(s, "param.ancilla_pool={}", p.block.ancilla_pool);
    let _ = writeln!(s, "param.compose_cap={}", p.block.compose_cap);
    let _ = writeln!(s, "param.compose_time_limit={}", p.block.compose_time_limit);
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
    let _ = writeln!(
        s,
        "param.census_batches_rule={}",
        if params.census_batches_auto {
            "auto"
        } else {
            "explicit"
        }
    );
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
    let _ = writeln!(s, "final_leakage.early_outputs={}", rep.final_early_outputs);
    let _ = writeln!(s, "final_reloaded={}", rep.final_reloaded);
    let okn = |v: Option<usize>| v.map_or("skipped".to_string(), |x| format!("ok({x})"));
    let _ = writeln!(
        s,
        "check.input_zero_slice=ok({})",
        rep.checks.input_zero_slice
    );
    let _ = writeln!(
        s,
        "check.input_payload_contract={}",
        okn(rep.checks.input_payload_contract)
    );
    let _ = writeln!(
        s,
        "check.equiv_canonical={}",
        if rep.checks.canonical_reloaded {
            "reloaded".to_string()
        } else {
            okn(rep.checks.equiv_canonical)
        }
    );
    let _ = writeln!(
        s,
        "canonical_order_applied={}",
        rep.checks
            .canonical_applied
            .map_or("unknown".to_string(), |b| b.to_string())
    );
    let _ = writeln!(s, "canonical_order_reason={}", rep.checks.canonical_reason);
    let _ = writeln!(
        s,
        "check.phase_c_guard={}",
        rep.checks.phase_c_guard.as_deref().unwrap_or("not_run")
    );
    let _ = writeln!(s, "check.zero_slice=ok({})", rep.checks.zero_slice);
    let _ = writeln!(
        s,
        "check.payload_contract={}",
        okn(rep.checks.payload_contract)
    );
    let _ = writeln!(
        s,
        "check.payload_directions={}",
        if rep.checks.payload_forward_only {
            "forward"
        } else {
            "both"
        }
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

/// The exact NOT-push + fcompress cleanup run between the descent caps.
/// Returns the cleaned circuit when it is strictly smaller than the input
/// and verified equivalent on `equiv_batches` x 1024 random full states,
/// else `None`. Resumable: `dir/cleanup.report` records the before/after
/// gate counts (and the input digest for diagnostics) and `dir/output.mpmct1`
/// the cleaned circuit; on resume the reload is keyed on the input GATE
/// COUNT (`current.count`, known from the reports), never on the input
/// circuit, which a completed run prunes. The input circuit is loaded only
/// on the compute path.
#[allow(clippy::too_many_arguments)]
fn cleanup_pass(
    current: &mut Current,
    width: usize,
    cost: Cost,
    fcompress_iters: usize,
    equiv_batches: usize,
    dir: &Path,
    guard: Option<&RoundContext<'_>>,
    log: Log<'_>,
) -> Result<Option<Vec<XGate>>, AdaptiveError> {
    let in_count = current.count;
    let in_cost = current.cost;
    let out_path = dir.join("output.mpmct1");
    let report = dir.join("cleanup.report");
    if let Ok(t) = fs::read_to_string(&report) {
        let kv = parse_kv(&t);
        let num = |k: &str| kv.get(k).and_then(|s| s.parse::<usize>().ok());
        if num("input_gates") == Some(in_count)
            && let Some(out_gates) = num("output_gates")
        {
            // reports written before the cost model compare gate counts
            let out_cost = num("output_cost").unwrap_or(out_gates);
            let rec_in_cost = num("input_cost").unwrap_or(in_count);
            if kv.get("rejected").map(String::as_str) == Some("leakier") {
                log(&format!(
                    "cleanup {} reloaded: rejected (exposure increased), kept {in_count} gates",
                    dir.display(),
                ));
                return Ok(None);
            }
            if out_cost >= rec_in_cost {
                log(&format!(
                    "cleanup {} reloaded: {in_count} gates, not smaller (kept)",
                    dir.display(),
                ));
                return Ok(None);
            }
            let (g, w) = read_mpmct(path_str(&out_path)?)
                .map_err(|e| AdaptiveError::Input(format!("{}: {e}", out_path.display())))?;
            if w == width && g.len() == out_gates {
                log(&format!(
                    "cleanup {} reloaded: {in_count} -> {} gates",
                    dir.display(),
                    g.len()
                ));
                return Ok(Some(g));
            }
        }
    }
    fs::create_dir_all(dir).map_err(|e| AdaptiveError::io(dir, e))?;
    let t0 = Instant::now();
    let input = current.load(width)?;
    let digest = format!("{:032x}", gates_digest(input));
    let (pushed, nr) = push_nots(input, width);
    let cleaned = if fcompress_iters == 0 {
        pushed
    } else {
        let params = CompressParams {
            max_iters: fcompress_iters,
            local_verify: true,
            ..CompressParams::default()
        };
        let (out, _) = compress(pushed, width, &params);
        out
    };
    // Exact by construction, but re-check globally like a round does.
    if !same_function(input, &cleaned, width, equiv_batches, CHECK_SEED ^ 0xC1EA) {
        return Err(AdaptiveError::Verify(format!(
            "cleanup {}: the cleaned circuit ({} gates) differs from its input ({} gates)",
            dir.display(),
            cleaned.len(),
            input.len()
        )));
    }
    // The leak guard: fcompress is C-blind, so census both sides and reject a
    // cleanup that exposes more (validated deeper affine / direct births, or
    // C outputs before EARLY_OUTPUT_FRAC) than its input did.
    let mut guard_kv = String::from("guard=off\n");
    let mut rejected = false;
    if let Some(ctx) = guard {
        let cin = load_or_census(ctx, input, &dir.join("input_census"), true, log)?;
        let cout = load_or_census(ctx, &cleaned, &dir.join("output_census"), true, log)?;
        let (li, lo) = (
            Leakage::of(&cin, ctx.source),
            Leakage::of(&cout, ctx.source),
        );
        let e_in = (
            li.validated.affine.deeper_births,
            li.validated.direct.deeper_births,
            cin.early_output_count(EARLY_OUTPUT_FRAC),
        );
        let e_out = (
            lo.validated.affine.deeper_births,
            lo.validated.direct.deeper_births,
            cout.early_output_count(EARLY_OUTPUT_FRAC),
        );
        rejected = e_out.0 > e_in.0 || e_out.1 > e_in.1 || e_out.2 > e_in.2;
        guard_kv = format!(
            "guard=on\ndeeper_affine_in={}\ndeeper_affine_out={}\ndeeper_direct_in={}\ndeeper_direct_out={}\nearly_outputs_in={}\nearly_outputs_out={}\n",
            e_in.0, e_out.0, e_in.1, e_out.1, e_in.2, e_out.2
        );
        if rejected {
            log(&format!(
                "cleanup {} REJECTED: exposure increased (deeper affine {} -> {}, direct {} -> {}, early outputs {} -> {}); kept the input",
                dir.display(),
                e_in.0,
                e_out.0,
                e_in.1,
                e_out.1,
                e_in.2,
                e_out.2
            ));
        }
    }
    write_mpmct_atomic(path_str(&out_path)?, &cleaned, width)
        .map_err(|e| AdaptiveError::io(&out_path, e))?;
    write_text_atomic(
        &report,
        &format!(
            "cleanup_report=1\ninput_digest={digest}\ninput_gates={}\noutput_gates={}\ncost={}\ninput_cost={in_cost}\noutput_cost={}\nnots_absorbed={}\nnoops_dropped={}\ntrailing_nots={}\n{guard_kv}rejected={}\n",
            input.len(),
            cleaned.len(),
            cost.name(),
            cost.of(&cleaned),
            nr.nots_absorbed,
            nr.noops_dropped,
            nr.trailing_nots,
            if rejected { "leakier" } else { "none" },
        ),
    )?;
    log(&format!(
        "cleanup {}: {} -> {} gates (NOT-push absorbed {}, dropped {} no-ops; fcompress {} iters; {:.1}s)",
        dir.display(),
        input.len(),
        cleaned.len(),
        nr.nots_absorbed,
        nr.noops_dropped,
        fcompress_iters,
        t0.elapsed().as_secs_f64(),
    ));
    // Accepted only when not leakier and smaller under the run's cost model
    // (fcompress may merge into wide cubes: fewer raw gates, but a higher
    // 2-control cost).
    if !rejected && cost.of(&cleaned) < in_cost {
        Ok(Some(cleaned))
    } else {
        Ok(None)
    }
}

/// Phase C: lower every gate wider than `max_controls` to 2-control gates
/// (exact Barenco), then one exact NOT-push + fcompress cleanup, and make
/// the result the published circuit. Exact and verified on `equiv_batches`
/// x 1024 states. Returns the new circuit (the gate SET changed, so this is
/// unconditional, not size-gated) with the leak guard's outcome, or `None`
/// when there were no wide gates. With `guard` (the leak guard on), the
/// result may not expose more than the input ([`Exposure`]); when it does,
/// up to [`PHASE_C_HELPER_ATTEMPTS`] lowerings with random helper wires
/// (NOT-push only) are tried, and when none avoids it the least exposed
/// candidate is published with the outcome `leakier`. Resumable keyed on the
/// input gate count (the `lower/` dir).
#[allow(clippy::too_many_arguments)]
fn lower_and_clean(
    current: &mut Current,
    width: usize,
    max_controls: usize,
    fcompress_iters: usize,
    equiv_batches: usize,
    dir: &Path,
    guard: Option<&RoundContext<'_>>,
    seed: u64,
    log: Log<'_>,
) -> Result<Option<(Vec<XGate>, String)>, AdaptiveError> {
    let in_count = current.count;
    let out_path = dir.join("output.mpmct1");
    let report = dir.join("lower.report");
    if let Ok(t) = fs::read_to_string(&report) {
        let kv = parse_kv(&t);
        if kv.get("input_gates").and_then(|s| s.parse::<usize>().ok()) == Some(in_count)
            && let Some(wide) = kv.get("wide_in").and_then(|s| s.parse::<u64>().ok())
        {
            if wide == 0 {
                log(&format!(
                    "phase C {} reloaded: no wide gates (kept)",
                    dir.display()
                ));
                return Ok(None);
            }
            let (g, w) = read_mpmct(path_str(&out_path)?)
                .map_err(|e| AdaptiveError::Input(format!("{}: {e}", out_path.display())))?;
            let outcome = kv
                .get("guard_outcome")
                .cloned()
                .unwrap_or_else(|| "off".to_string());
            if w == width {
                log(&format!(
                    "phase C {} reloaded: {in_count} -> {} gates (guard {outcome})",
                    dir.display(),
                    g.len()
                ));
                return Ok(Some((g, outcome)));
            }
        }
    }
    fs::create_dir_all(dir).map_err(|e| AdaptiveError::io(dir, e))?;
    let t0 = Instant::now();
    let input = current.load(width)?;
    let (l1, lrep) = lower_wide_gates(input, width, max_controls);
    if lrep.wide_in == 0 {
        write_text_atomic(
            &report,
            &format!(
                "lower_report=1\ninput_gates={in_count}\nwide_in=0\nmax_controls_in={}\n",
                lrep.max_controls_in
            ),
        )?;
        log(&format!(
            "phase C {}: no gates wider than {max_controls} controls (max seen {})",
            dir.display(),
            lrep.max_controls_in
        ));
        return Ok(None);
    }
    // Lower FIRST so fcompress compresses the 2-control form, then lower AGAIN
    // because fcompress may re-introduce multi-control cubes -- lowering is the
    // last width-changing step, so the output is in the gate set by
    // construction. push_nots (<= 2-control preserving) tidies the Barenco NOTs.
    let (p1, _) = push_nots(&l1, width);
    let compressed = if fcompress_iters == 0 {
        p1
    } else {
        let params = CompressParams {
            max_iters: fcompress_iters,
            local_verify: true,
            ..CompressParams::default()
        };
        compress(p1, width, &params).0
    };
    let (l2, lrep2) = lower_wide_gates(&compressed, width, max_controls);
    let (cleaned, _) = push_nots(&l2, width);
    // The guarantee: nothing wider than max_controls survives (only possible
    // if a gate had too few spare helper wires, which never happens at a
    // stage-5 width).
    if let Some(w) = cleaned.iter().map(|g| g.ctrls.len()).max()
        && w > max_controls
    {
        return Err(AdaptiveError::Internal(format!(
            "phase C {}: a {w}-control gate survived lowering on {width} wires (wide_kept {})",
            dir.display(),
            lrep2.wide_kept
        )));
    }
    if !same_function(input, &cleaned, width, equiv_batches, CHECK_SEED ^ 0x10E2) {
        return Err(AdaptiveError::Verify(format!(
            "phase C {}: the lowered circuit ({} gates) differs from its input ({in_count} gates)",
            dir.display(),
            cleaned.len(),
        )));
    }
    // The leak guard: the lowering's dirty helpers and the cleanup are C-blind,
    // so census both sides; a leakier result is retried with random helper
    // wires (lowering and NOT-push only) and, failing that, the least
    // exposed candidate is published and flagged.
    let mut cleaned = cleaned;
    let mut outcome = String::from("off");
    let mut guard_kv = String::from("guard=off\n");
    if let Some(ctx) = guard {
        let e_in = Exposure::of(
            &load_or_census(ctx, input, &dir.join("input_census"), true, log)?,
            ctx.source,
        );
        let e_out = Exposure::of(
            &load_or_census(ctx, &cleaned, &dir.join("output_census"), true, log)?,
            ctx.source,
        );
        guard_kv = format!("guard=on\nexposure_in={e_in}\nexposure_cleaned={e_out}\n");
        if !e_out.exceeds(&e_in) {
            outcome = "kept".into();
        } else {
            log(&format!(
                "phase C {}: the lowering exposes more ({e_in} -> {e_out}); retrying with random helper wires",
                dir.display()
            ));
            let mut best = (e_out, None::<Vec<XGate>>);
            let mut found = None;
            for attempt in 0..PHASE_C_HELPER_ATTEMPTS {
                let (l, lr) = lower_wide_gates_with(
                    input,
                    width,
                    max_controls,
                    HelperChoice::Seeded(seed ^ 0x10E2_C0DE ^ attempt),
                );
                let (p, _) = push_nots(&l, width);
                if lr.wide_kept > 0 || p.iter().any(|g| g.ctrls.len() > max_controls) {
                    continue;
                }
                if !same_function(
                    input,
                    &p,
                    width,
                    equiv_batches,
                    CHECK_SEED ^ 0x10E3 ^ attempt,
                ) {
                    return Err(AdaptiveError::Verify(format!(
                        "phase C {}: random-helper lowering {attempt} differs from its input",
                        dir.display()
                    )));
                }
                let e = Exposure::of(
                    &load_or_census(
                        ctx,
                        &p,
                        &dir.join(format!("attempt{attempt}_census")),
                        true,
                        log,
                    )?,
                    ctx.source,
                );
                let _ = writeln!(guard_kv, "exposure_attempt{attempt}={e}");
                if !e.exceeds(&e_in) {
                    found = Some((attempt, p));
                    break;
                }
                if e < best.0 {
                    best = (e, Some(p));
                }
            }
            match found {
                Some((attempt, p)) => {
                    cleaned = p;
                    outcome = format!("random_helpers:{attempt}");
                }
                None => {
                    if let Some(p) = best.1 {
                        cleaned = p;
                    }
                    outcome = "leakier".into();
                    log(&format!(
                        "phase C {}: WARNING no lowering avoids the extra exposure (least exposed: {}); published, report status leakier_lowering",
                        dir.display(),
                        best.0
                    ));
                }
            }
        }
    }
    write_mpmct_atomic(path_str(&out_path)?, &cleaned, width)
        .map_err(|e| AdaptiveError::io(&out_path, e))?;
    write_text_atomic(
        &report,
        &format!(
            "lower_report=1\ninput_gates={in_count}\noutput_gates={}\nwide_in={}\nmax_controls_in={}\nrewidened_by_fcompress={}\nwide_kept={}\n{guard_kv}guard_outcome={outcome}\n",
            cleaned.len(),
            lrep.wide_in,
            lrep.max_controls_in,
            lrep2.wide_in,
            lrep2.wide_kept,
        ),
    )?;
    log(&format!(
        "phase C {}: lowered {} wide gates (max {} controls), fcompress re-widened {}, {in_count} -> {} gates (all <= {max_controls} controls), guard {outcome}; {:.1}s",
        dir.display(),
        lrep.wide_in,
        lrep.max_controls_in,
        lrep2.wide_in,
        cleaned.len(),
        t0.elapsed().as_secs_f64(),
    ));
    Ok(Some((cleaned, outcome)))
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
    // `--census-batches auto`: the census batch rule at this width
    let resolved_params;
    let params = if params.round.census_batches == AUTO_BATCHES {
        let mut p = params.clone();
        p.round.census_batches = batches_for_width(width);
        p.census_batches_auto = true;
        say(&format!(
            "census batches: auto, {} for {width} wires ({} inputs)",
            p.round.census_batches,
            64 * p.round.census_batches
        ));
        resolved_params = p;
        &resolved_params
    } else {
        params
    };
    let n = inputs.n;
    // the census draws 64 inputs per batch and needs more than width + 128
    // of them (a usage error here, not a panic after hours of synthesis)
    if 64 * params.round.census_batches <= width + 128 {
        return Err(AdaptiveError::Usage(format!(
            "census batches {}: a {width}-wire circuit needs at least {}",
            params.round.census_batches,
            (width + 128) / 64 + 1
        )));
    }
    // the planner composes every block from the identity (one term per wire)
    // and asserts its term limit covers the width: refuse up front instead
    if params.round.plan.term_cap < width {
        return Err(AdaptiveError::Usage(format!(
            "term cap {}: a {width}-wire circuit needs at least {width}",
            params.round.plan.term_cap
        )));
    }
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

    // ---- fail fast: the stage-5 circuit must satisfy its own contracts
    // (a wrong --n, --source-c or --sandwich is caught here, in seconds,
    // not after every round)
    let input_zero_slice = check_zero_slice_gates(
        &c,
        &gates,
        n,
        width,
        ZERO_SLICE_SAMPLES,
        CHECK_SEED ^ 0x1A5A,
    )
    .map_err(|e| match e {
        crate::stages::unpacking::Stage7Error::Verify(m) => AdaptiveError::Verify(format!(
            "the stage-5 circuit {} does not satisfy the zero slice against C {} with n={n}: {m}",
            inputs.input.display(),
            inputs.source_c.display()
        )),
        other => other.into(),
    })?;
    say(&format!(
        "input zero slice vs C: ok ({input_zero_slice} samples)"
    ));
    let input_payload_contract = match &sandwich {
        Some((sg, sw)) => {
            let k = payload_contract_dirs(sg, *sw, &gates, width, n, !inputs.forward_only)
                .map_err(|e| match e {
                AdaptiveError::Verify(m) => AdaptiveError::Verify(format!(
                    "the stage-5 circuit {} does not satisfy the payload contract of the sandwich {}: {m}",
                    inputs.input.display(),
                    inputs
                        .sandwich
                        .as_ref()
                        .map_or(String::new(), |p| p.display().to_string())
                )),
                other => other,
            })?;
            say(&format!(
                "input payload contract vs the sandwich: ok ({k} inputs, {})",
                if inputs.forward_only {
                    "forward only"
                } else {
                    "both directions"
                }
            ));
            Some(k)
        }
        None => None,
    };

    // ---- rounds
    let input_gates = gates.len();
    let cost_model = params.round.cost;
    let input_cost = cost_model.of(&gates);
    let mut current = Current {
        gates: Some(gates),
        path: inputs.input.clone(),
        round: None,
        count: input_gates,
        cost: input_cost,
    };
    let mut rounds: Vec<RoundResult> = Vec::new();
    let mut stop_reason = String::from("round cap");
    let mut r = 0usize;
    let mut plateau = false;
    let mut hit_round_cap = false;
    // The block-cap LEVELS of this stage (not pipeline stages): level 0 is
    // phase A (the classic large-block rounds, with the run's own blind flag
    // and stop fraction), levels 1.. are phase B's descent caps (planned
    // C-blind), each run to its own stall. An exact NOT-push + fcompress
    // cleanup runs after a level when phase B is active.
    struct Level {
        cap: usize,
        blind: bool,
        stop_fraction: f64,
    }
    let mut levels: Vec<Level> = vec![Level {
        cap: params.round.plan.max_block_gates,
        blind: params.round.blind,
        stop_fraction: params.stop_fraction,
    }];
    for &cap in &params.descent {
        levels.push(Level {
            cap,
            blind: true,
            stop_fraction: params.descent_stop_fraction,
        });
    }
    let cleanup_on = !params.descent.is_empty();
    // The pre-cleanup: one exact cleanup of the input before phase A.
    if params.pre_cleanup {
        let cdir = inputs.out_dir.join("precleanup");
        let guard = (params.round.accept == Accept::SmallerNotLeakier).then_some(&ctx);
        if let Some(cleaned) = cleanup_pass(
            &mut current,
            width,
            cost_model,
            params.cleanup_fcompress_iters,
            params.round.equiv_batches,
            &cdir,
            guard,
            say,
        )? {
            current = Current {
                count: cleaned.len(),
                cost: cost_model.of(&cleaned),
                gates: Some(cleaned),
                path: cdir.join("output.mpmct1"),
                round: None,
            };
        }
    }
    'levels: for (si, level) in levels.iter().enumerate() {
        let mut rp = params.round.clone();
        rp.plan.max_block_gates = level.cap;
        rp.blind = level.blind;
        let sctx = RoundContext {
            source: &source,
            width,
            stage_seed: inputs.seed,
            params: &rp,
        };
        if si > 0 {
            say(&format!(
                "phase B descent level {si}/{}: block cap {} gates, blind, stop fraction {}",
                levels.len() - 1,
                level.cap,
                level.stop_fraction
            ));
        }
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        let mut prev_bounds: Vec<usize> = Vec::new();
        let mut stalled = 0usize;
        loop {
            r += 1;
            if params.max_rounds > 0 && r > params.max_rounds {
                hit_round_cap = true;
                break 'levels;
            }
            let rdir = inputs.out_dir.join(format!("round{r}"));
            // a completed round is reloaded from its report alone; only a
            // round that must run reads the current circuit (so a completed
            // run, whose non-final round outputs are pruned, reruns from its
            // reports)
            let mut res = match reload_round(&rdir, r, current.count, say)? {
                Some(res) => res,
                None => {
                    let g = current.load(width)?;
                    run_round(&sctx, r, g, &prev_bounds, &rdir, say)?
                }
            };
            let output = res.output.take();
            pairs.push((res.input_cost, res.output_cost));
            let shrank = res.output_cost < res.input_cost;
            if shrank {
                prev_bounds = if res.reloaded {
                    read_bounds_file(&rdir.join("bounds.out"))?
                } else {
                    res.output_boundaries.clone()
                };
            }
            let count = res.output_gates;
            let ocost = res.output_cost;
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
                    count,
                    cost: ocost,
                };
            }
            if let Some(reason) = stop_rule(&pairs, &mut stalled, level.stop_fraction) {
                stop_reason = reason;
                break;
            }
        }
        plateau = stalled >= 2;
        // An exact cleanup between/after stages: the next cap (and the final
        // publish) then works on the smaller, NOT-normalized circuit.
        if cleanup_on {
            let cdir = inputs.out_dir.join(format!("cleanup{si}"));
            let round = current.round;
            let guard = (params.round.accept == Accept::SmallerNotLeakier).then_some(&sctx);
            if let Some(cleaned) = cleanup_pass(
                &mut current,
                width,
                cost_model,
                params.cleanup_fcompress_iters,
                params.round.equiv_batches,
                &cdir,
                guard,
                say,
            )? {
                current = Current {
                    count: cleaned.len(),
                    cost: cost_model.of(&cleaned),
                    gates: Some(cleaned),
                    path: cdir.join("output.mpmct1"),
                    round,
                };
                write_text_atomic(
                    &inputs.out_dir.join("progress.txt"),
                    &progress_text(&rounds),
                )?;
            }
        }
    }
    if hit_round_cap {
        stop_reason = "round cap".into();
        plateau = false;
    }
    // Phase C: lower the leftover wide gates into the deliverable's 2-control
    // gate set, then one more exact cleanup. The gate SET changes, so this is
    // not size-gated; its output is the published circuit.
    let mut phase_c_guard: Option<String> = None;
    if let Some(mc) = params.lower_controls {
        let ldir = inputs.out_dir.join("lower");
        let round = current.round;
        let guard = (params.round.accept == Accept::SmallerNotLeakier).then_some(&ctx);
        if let Some((lowered, outcome)) = lower_and_clean(
            &mut current,
            width,
            mc,
            params.cleanup_fcompress_iters,
            params.round.equiv_batches,
            &ldir,
            guard,
            inputs.seed,
            say,
        )? {
            phase_c_guard = Some(outcome);
            current = Current {
                count: lowered.len(),
                cost: cost_model.of(&lowered),
                gates: Some(lowered),
                path: ldir.join("output.mpmct1"),
                round,
            };
            write_text_atomic(
                &inputs.out_dir.join("progress.txt"),
                &progress_text(&rounds),
            )?;
        }
    }
    say(&format!(
        "rounds done: {} rounds, {input_gates} -> {} gates ({})",
        rounds.len(),
        current.count.min(input_gates),
        stop_reason
    ));

    // ---- the final circuit and its checks
    let final_round = current.round;
    let fpath = inputs.out_dir.join("final.mpmct1");
    // the final round's output is in memory (it ran here) or on disk (a
    // partial run keeps every round output); a COMPLETED run has pruned
    // nothing but the final round's, yet when that too is gone the
    // published final.mpmct1 is the circuit (gate count checked against
    // the round's report) and is republished unchanged
    let (mut final_gates, final_reloaded): (Vec<XGate>, bool) = match current.gates.take() {
        Some(g) => (g, false),
        None if final_round.is_some() && fpath.is_file() => {
            let (g, w) = read_mpmct(path_str(&fpath)?)
                .map_err(|e| AdaptiveError::Input(format!("{}: {e}", fpath.display())))?;
            if w == width && g.len() == current.count {
                say(&format!(
                    "final circuit reloaded from {} ({} gates, round {})",
                    fpath.display(),
                    g.len(),
                    final_round.unwrap_or(0)
                ));
                (g, true)
            } else if current.path.is_file() {
                say(&format!(
                    "{}: {} gates on {w} wires, round {} reported {} on {width}: stale, reading the round output",
                    fpath.display(),
                    g.len(),
                    final_round.unwrap_or(0),
                    current.count
                ));
                (current.load(width)?.to_vec(), false)
            } else {
                return Err(AdaptiveError::Input(format!(
                    "{}: {} gates on {w} wires, round {} reported {} gates on {width} (stale final.mpmct1, and the round output is pruned)",
                    fpath.display(),
                    g.len(),
                    final_round.unwrap_or(0),
                    current.count
                )));
            }
        }
        None => (current.load(width)?.to_vec(), false),
    };
    current.gates = None;
    let mut checks = FinalChecks {
        input_zero_slice,
        input_payload_contract,
        payload_forward_only: inputs.forward_only,
        phase_c_guard,
        ..Default::default()
    };
    let final_census_dir = inputs.out_dir.join("final_census");
    let mut final_census: Option<Census> = None;
    if !params.canonical_order {
        checks.canonical_applied = Some(false);
        checks.canonical_reason = "disabled (--no-canonical-order)".into();
    } else if final_reloaded {
        // the published circuit is republished as read: the order check
        // and the exposure guard of the run that wrote it stand (its
        // report says whether the order was applied)
        checks.canonical_reloaded = true;
        let prev = fs::read_to_string(inputs.out_dir.join("adaptive.report"))
            .map(|t| parse_kv(&t))
            .unwrap_or_default();
        checks.canonical_applied = match prev.get("canonical_order_applied").map(String::as_str) {
            Some("true") => Some(true),
            Some("false") => Some(false),
            _ => None,
        };
        checks.canonical_reason = prev
            .get("canonical_order_reason")
            .cloned()
            .unwrap_or_else(|| "reloaded".into());
        say(&format!(
            "canonical order: reloaded ({} gates, published by the run that wrote {}; applied={})",
            final_gates.len(),
            fpath.display(),
            checks
                .canonical_applied
                .map_or("unknown".to_string(), |b| b.to_string())
        ));
    } else {
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
        say(&format!(
            "canonical order: {} gates, equivalence ok ({} samples, {:.1}s)",
            canon.len(),
            FINAL_EQUIV_BATCHES * 1024,
            t.elapsed().as_secs_f64()
        ));
        // the exposure guard: reordering changes which values are
        // simultaneously live, so the reordered circuit may not expose more
        // than the circuit it reorders (after the cleanups and phase C, not
        // the last round's output): validated deeper births, affine and
        // direct, and early C outputs (`Exposure`). The census of the
        // circuit being reordered is reloaded where phase C left it.
        let (base_census, base_dir) = {
            let ldir = inputs.out_dir.join("lower");
            let mut candidates = vec![ldir.join("output_census")];
            candidates.extend(
                (0..PHASE_C_HELPER_ATTEMPTS).map(|k| ldir.join(format!("attempt{k}_census"))),
            );
            match candidates
                .iter()
                .find_map(|d| reload_census(d, &final_gates).map(|c| (c, d.clone())))
            {
                Some(found) => found,
                None => {
                    let d = inputs.out_dir.join("precanonical_census");
                    (load_or_census(&ctx, &final_gates, &d, true, say)?, d)
                }
            }
        };
        let base = Exposure::of(&base_census, &source);
        let cc = load_or_census(&ctx, &canon, &final_census_dir, true, say)?;
        let ce = Exposure::of(&cc, &source);
        if ce.exceeds(&base) {
            checks.canonical_applied = Some(false);
            checks.canonical_reason = format!("exposure increased ({base} -> {ce})");
            say(&format!(
                "canonical order NOT applied: {}; publishing the circuit as assembled",
                checks.canonical_reason
            ));
            // the reordered circuit's census stays on record next to the
            // published one's
            let keep = inputs.out_dir.join("final_canonical_census");
            let _ = fs::remove_dir_all(&keep);
            fs::rename(&final_census_dir, &keep).map_err(|e| AdaptiveError::io(&keep, e))?;
            // the published circuit's census at final_census/, as always
            copy_census_dir(&base_dir, &final_census_dir)?;
            final_census = Some(base_census);
        } else {
            checks.canonical_applied = Some(true);
            checks.canonical_reason = format!("ordered ({base} -> {ce})");
            final_gates = canon;
            final_census = Some(cc);
        }
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
        let k = payload_contract_dirs(sg, *sw, &final_gates, width, n, !inputs.forward_only)?;
        checks.payload_contract = Some(k);
        say(&format!(
            "payload contract vs the sandwich: ok ({k} inputs, {})",
            if inputs.forward_only {
                "forward only"
            } else {
                "both directions"
            }
        ));
    }
    let fc = match final_census {
        Some(c) => c,
        None => load_or_census(&ctx, &final_gates, &final_census_dir, true, say)?,
    };
    let final_leakage = Leakage::of(&fc, &source);
    let final_early_outputs = fc.early_output_count(EARLY_OUTPUT_FRAC);
    drop(fc);
    let rep = FinalReport {
        input_gates,
        final_gates: final_gates.len(),
        final_round,
        final_reloaded,
        rounds,
        stop_reason,
        plateau,
        final_leakage,
        final_early_outputs,
        boundary: boundary_counts,
        checks,
        seconds: t0.elapsed().as_secs_f64(),
    };
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
