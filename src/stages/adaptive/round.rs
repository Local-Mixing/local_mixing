//! One round of the adaptive stage (port of `round_once` of the prototype,
//! variant `v2`):
//!
//! 1. census of the input (reloaded from `input_census/` when present);
//! 2. plan the partition (`planner::plan`, against the empty census when
//!    `blind`; reloaded from `partition/bounds` when present);
//! 3. synthesize every block for every seed, grouped by budget (x1 / x2 /
//!    x4 of the base time limit for blocks of <= 1000 / <= 4000 / more
//!    gates), blocks in parallel, each block single-threaded under its own
//!    deadline; every group is cached as `synth0/s<k>_t<mult>.{tsv,mpmct1}`
//!    so that a killed round resumes from its finished groups;
//! 4. refinement: a block that failed for every seed and is longer than
//!    `min_refine_gates` is split at `Profile::split` and its children are
//!    synthesized (`refine<level>/`), up to `refinement_levels` times;
//! 5. acceptance per block: the original, or a seed's synthesis that is
//!    strictly smaller and (with `SmallerNotLeakier`) exposes no more
//!    deeper internal C births at its interior cuts than the original block
//!    does — measured on a census of the candidate circuit assembled from
//!    that seed's results (`candidate_s<k>_census/`);
//! 6. assemble the output, check it equivalent to the input on random full
//!    states (exit 4 on a mismatch), census it, write `output.mpmct1`,
//!    `bounds.out` and the completion record `result.report`.
//!
//! A round whose `result.report` exists is reloaded, never recomputed.
use super::census;
use super::planner::{self, Profile};
use super::synth::{BlockParams, BlockResult, same_function, synthesize_block};
use super::types::{Census, FamilyMetrics, LeakageMetrics, PlanParams, Source};
use crate::circuit::formats::{read_mpmct, write_mpmct_atomic};
use crate::circuit::xgate::XGate;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// Progress lines of the stage (called from worker threads too).
pub type Log<'a> = &'a (dyn Fn(&str) + Sync);

/// Seed of the per-round sampled equivalence check (public, fixed).
pub const EQUIV_SEED: u64 = 20_261_026 ^ 0xE0E0;

// ---------------------------------------------------------------- errors

/// Stage errors and exit codes, the table of `unpacking::Stage7Error`.
#[derive(Debug)]
pub enum AdaptiveError {
    /// Exit 2: usage or parameter error.
    Usage(String),
    /// Exit 3: unreadable, malformed or inconsistent input.
    Input(String),
    /// Exit 4: a check failed.
    Verify(String),
    /// Exit 7: writing an output failed.
    Io { path: PathBuf, source: io::Error },
    /// Exit 8: an internal invariant was violated without panicking.
    Internal(String),
}

pub const EXIT_CODES: [(i32, &str); 7] = [
    (0, "finished"),
    (2, "usage or parameter error"),
    (3, "input error"),
    (4, "verification failure"),
    (7, "I/O error writing outputs"),
    (8, "internal invariant violated"),
    (101, "panic"),
];

impl AdaptiveError {
    pub fn exit_code(&self) -> i32 {
        match self {
            AdaptiveError::Usage(_) => 2,
            AdaptiveError::Input(_) => 3,
            AdaptiveError::Verify(_) => 4,
            AdaptiveError::Io { .. } => 7,
            AdaptiveError::Internal(_) => 8,
        }
    }

    pub fn meaning(code: i32) -> &'static str {
        EXIT_CODES
            .iter()
            .find(|(c, _)| *c == code)
            .map(|(_, m)| *m)
            .unwrap_or("unknown exit code")
    }

    pub fn io(path: impl Into<PathBuf>, source: io::Error) -> AdaptiveError {
        AdaptiveError::Io {
            path: path.into(),
            source,
        }
    }
}

impl fmt::Display for AdaptiveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AdaptiveError::Usage(m) => write!(f, "usage: {m}"),
            AdaptiveError::Input(m) => write!(f, "input: {m}"),
            AdaptiveError::Verify(m) => write!(f, "verification failed: {m}"),
            AdaptiveError::Io { path, source } => {
                write!(f, "I/O error on {}: {source}", path.display())
            }
            AdaptiveError::Internal(m) => write!(f, "internal error: {m}"),
        }
    }
}

impl std::error::Error for AdaptiveError {}

impl From<crate::stages::unpacking::Stage7Error> for AdaptiveError {
    fn from(e: crate::stages::unpacking::Stage7Error) -> Self {
        use crate::stages::unpacking::Stage7Error as E;
        match e {
            E::Usage(m) => AdaptiveError::Usage(m),
            E::Input(m) => AdaptiveError::Input(m),
            E::Verify(m) => AdaptiveError::Verify(m),
            E::Io { path, source } => AdaptiveError::Io { path, source },
            E::Internal(m) => AdaptiveError::Internal(m),
        }
    }
}

// ---------------------------------------------------------------- parameters

/// The block acceptance rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Accept {
    /// Keep a synthesis that is strictly smaller than the original block.
    Smaller,
    /// ... and exposes no more deeper internal births at its interior cuts.
    SmallerNotLeakier,
}

impl Accept {
    pub fn name(self) -> &'static str {
        match self {
            Accept::Smaller => "smaller",
            Accept::SmallerNotLeakier => "smaller-not-leakier",
        }
    }

    pub fn parse(s: &str) -> Option<Accept> {
        match s {
            "smaller" => Some(Accept::Smaller),
            "smaller-not-leakier" | "smaller_not_leakier" => Some(Accept::SmallerNotLeakier),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RoundParams {
    pub plan: PlanParams,
    pub block: BlockParams,
    /// Synthesizer seeds per block (every block is tried with each).
    pub seeds_per_block: usize,
    /// Seconds for a block of <= 1000 gates; x2 up to 4000, x4 above.
    pub base_time_limit: f64,
    pub refinement_levels: usize,
    /// Only blocks longer than this are split.
    pub min_refine_gates: usize,
    pub census_batches: usize,
    pub accept: Accept,
    /// Plan and split against the empty census (the C-blind attacker's
    /// view); the acceptance still measures exposure on real censuses.
    pub blind: bool,
    /// Batches of 1024 random full states of the output == input check.
    pub equiv_batches: usize,
}

impl Default for RoundParams {
    fn default() -> Self {
        RoundParams {
            plan: PlanParams::default(),
            block: BlockParams::default(),
            seeds_per_block: 2,
            base_time_limit: 60.0,
            refinement_levels: 2,
            min_refine_gates: 128,
            census_batches: census::DEFAULT_BATCHES,
            accept: Accept::SmallerNotLeakier,
            blind: false,
            equiv_batches: 8,
        }
    }
}

impl RoundParams {
    pub fn validate(&self) -> Result<(), String> {
        self.block.validate()?;
        if self.seeds_per_block == 0 || self.seeds_per_block > 64 {
            return Err(format!(
                "seeds per block {}: must be in 1..=64",
                self.seeds_per_block
            ));
        }
        if !(self.base_time_limit > 0.0 && self.base_time_limit.is_finite()) {
            return Err(format!(
                "time limit {}: must be a positive number of seconds",
                self.base_time_limit
            ));
        }
        if self.refinement_levels > 16 {
            return Err("refinement levels: at most 16".into());
        }
        if self.census_batches == 0 {
            return Err("census batches must be positive".into());
        }
        if self.equiv_batches == 0 || self.equiv_batches > 1024 {
            return Err(format!(
                "equivalence batches {}: must be in 1..=1024",
                self.equiv_batches
            ));
        }
        if self.plan.term_cap == 0 || self.plan.max_block_gates == 0 {
            return Err("term cap and max block gates must be positive".into());
        }
        if self.plan.segments == 0 {
            return Err("segments must be positive".into());
        }
        Ok(())
    }
}

/// What every round shares.
pub struct RoundContext<'a> {
    pub source: &'a Source,
    /// Wires of the circuit (the state width).
    pub width: usize,
    /// The stage seed (never printed); block seeds derive from it.
    pub stage_seed: u64,
    pub params: &'a RoundParams,
}

// ---------------------------------------------------------------- results

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RefinementLevel {
    pub level: usize,
    /// Blocks split at this level.
    pub split: usize,
    pub children: usize,
    /// `(child, seed)` pairs that synthesized.
    pub synthesized: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Picks {
    pub orig: usize,
    /// Blocks won by seed `k`.
    pub seed: Vec<usize>,
    pub rejected_leakier: usize,
    pub rejected_growing: usize,
}

impl Picks {
    pub fn new(seeds: usize) -> Picks {
        Picks {
            seed: vec![0; seeds],
            ..Default::default()
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Coverage {
    pub synthesized_blocks: usize,
    pub input_gates_synthesized: usize,
}

/// The two metric families of one census.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Leakage {
    pub discovery: LeakageMetrics,
    pub validated: LeakageMetrics,
    pub audit_passed: bool,
}

impl Leakage {
    pub fn of(c: &Census, source: &Source) -> Leakage {
        Leakage {
            discovery: c.metrics(source, false),
            validated: c.metrics(source, true),
            audit_passed: c.audit.all_passed(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RoundResult {
    pub round: usize,
    pub input_gates: usize,
    pub output_gates: usize,
    /// Leaves after refinement.
    pub blocks: usize,
    pub initial_blocks: usize,
    pub refinement: Vec<RefinementLevel>,
    pub picks: Picks,
    pub coverage: Coverage,
    pub leakage_in: Leakage,
    pub leakage_out: Leakage,
    /// Interior cuts of the output at the block boundaries.
    pub output_boundaries: Vec<usize>,
    pub equivalence_samples: usize,
    pub seconds: f64,
    /// Reloaded from an earlier run's `result.report`.
    pub reloaded: bool,
    /// The output gates (in memory when the round ran in this process;
    /// `None` when reloaded: the driver reads `output.mpmct1` on demand).
    pub output: Option<Vec<XGate>>,
}

// ---------------------------------------------------------------- policy

/// The synthesizer seed of seed slot `k` in round `r`: a fixed mixing of
/// the stage seed (never printed, like the stage seed).
pub fn block_seed(stage_seed: u64, round: usize, k: usize) -> u64 {
    let mut z = stage_seed
        ^ (round as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (k as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The budget multiplier of a block: 1 up to 1000 gates, 2 up to 4000, 4
/// above (the prototype's 60 / 120 / 240 s).
pub fn budget_multiplier(gates: usize) -> usize {
    if gates <= 1000 {
        1
    } else if gates <= 4000 {
        2
    } else {
        4
    }
}

pub fn budget(base: f64, gates: usize) -> f64 {
    base * budget_multiplier(gates) as f64
}

/// A seed's synthesis of one block, as the acceptance sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub seed: usize,
    pub gates: usize,
    /// Deeper internal births exposed at the block's interior cuts in that
    /// seed's candidate circuit (ignored by `Accept::Smaller`).
    pub exposure: usize,
}

/// Choose a block's replacement: `None` keeps the original (`orig_gates`,
/// exposing `orig_exposure` births; `None` = the `Smaller` rule). A
/// candidate must be strictly smaller (else `rejected_growing`) and, under
/// the exposure rule, not leakier (else `rejected_leakier`); the
/// `(exposure, gates)`-minimum wins, the original first on ties (the
/// prototype's `min` over `[original] + candidates`).
pub fn choose(
    orig_gates: usize,
    orig_exposure: Option<usize>,
    cands: &[Candidate],
    picks: &mut Picks,
) -> Option<usize> {
    let mut opts: Vec<(usize, usize, Option<usize>)> =
        vec![(orig_exposure.unwrap_or(usize::MAX), orig_gates, None)];
    for c in cands {
        if c.gates >= orig_gates {
            picks.rejected_growing += 1;
            continue;
        }
        match orig_exposure {
            Some(e_in) => {
                if c.exposure > e_in {
                    picks.rejected_leakier += 1;
                    continue;
                }
                opts.push((c.exposure, c.gates, Some(c.seed)));
            }
            None => opts.push((0, c.gates, Some(c.seed))),
        }
    }
    let best = if orig_exposure.is_some() {
        opts.iter().min_by_key(|o| (o.0, o.1))
    } else {
        opts.iter().min_by_key(|o| o.1)
    }
    .expect("the original is always an option");
    match best.2 {
        None => picks.orig += 1,
        Some(k) => picks.seed[k] += 1,
    }
    best.2
}

/// `[0] + the leaves' ends`.
pub fn edges_of(leaves: &[(usize, usize)]) -> Vec<usize> {
    let mut e = Vec::with_capacity(leaves.len() + 1);
    e.push(leaves.first().map_or(0, |l| l.0));
    e.extend(leaves.iter().map(|l| l.1));
    e
}

// ---------------------------------------------------------------- block rows

/// One block's synthesis record (a row of a group's `.tsv`).
#[derive(Clone, Debug, PartialEq)]
pub struct BlockRow {
    pub lo: usize,
    pub hi: usize,
    /// `ok` or the failure reason.
    pub status: String,
    pub synth_status: String,
    pub syn_gates: usize,
    pub secs: f64,
    pub time_limit: f64,
    pub support: usize,
    pub changed: usize,
    pub monomials: usize,
    pub max_degree: usize,
    pub pre_postopt: usize,
    pub steps: usize,
    pub restarts_used: usize,
    pub deadline_hit: bool,
    pub universe: usize,
    pub note: String,
    /// The checked replacement (`status == ok`).
    pub gates: Option<Vec<XGate>>,
}

impl BlockRow {
    pub fn is_ok(&self) -> bool {
        self.gates.is_some()
    }

    fn of(lo: usize, hi: usize, time_limit: f64, r: BlockResult) -> BlockRow {
        BlockRow {
            lo,
            hi,
            status: r.status.name().to_string(),
            synth_status: r.synth_status.to_string(),
            syn_gates: r.gates.as_ref().map_or(hi - lo, |g| g.len()),
            secs: r.secs,
            time_limit,
            support: r.support,
            changed: r.changed,
            monomials: r.monomials,
            max_degree: r.max_degree,
            pre_postopt: r.pre_postopt,
            steps: r.steps,
            restarts_used: r.restarts_used,
            deadline_hit: r.deadline_hit,
            universe: r.universe,
            note: r.note,
            gates: r.gates,
        }
    }
}

const ROW_HEADER: &str = "lo\thi\torig_gates\tstatus\tsynth_status\tsyn_gates\tsyn_offset\tseconds\ttime_limit\tsupport\tchanged\tmonomials\tmax_degree\tpre_postopt\tsteps\trestarts_used\tdeadline_hit\tuniverse\tnote";

/// The rows' tsv; `offsets[i]` locates row `i`'s gates in the group circuit
/// (-1 for a failed row).
fn rows_tsv(rows: &[BlockRow], offsets: &[i64]) -> String {
    let mut s = String::from(ROW_HEADER);
    s.push('\n');
    for (r, &off) in rows.iter().zip(offsets) {
        let _ = writeln!(
            s,
            "{}\t{}\t{}\t{}\t{}\t{}\t{off}\t{:.3}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            r.lo,
            r.hi,
            r.hi - r.lo,
            r.status,
            r.synth_status,
            r.syn_gates,
            r.secs,
            r.time_limit,
            r.support,
            r.changed,
            r.monomials,
            r.max_degree,
            r.pre_postopt,
            r.steps,
            r.restarts_used,
            r.deadline_hit as u8,
            r.universe,
            r.note.replace(['\t', '\n'], " ")
        );
    }
    s
}

/// The group's rows and the concatenation of the ok rows' gates.
fn save_group(
    tsv: &Path,
    circ: &Path,
    rows: &[BlockRow],
    width: usize,
) -> Result<(), AdaptiveError> {
    let mut gates: Vec<XGate> = Vec::new();
    let mut offsets: Vec<i64> = Vec::with_capacity(rows.len());
    for r in rows {
        match &r.gates {
            Some(g) => {
                offsets.push(gates.len() as i64);
                gates.extend_from_slice(g);
            }
            None => offsets.push(-1),
        }
    }
    write_mpmct_atomic(path_str(circ)?, &gates, width).map_err(|e| AdaptiveError::io(circ, e))?;
    write_text_atomic(tsv, &rows_tsv(rows, &offsets))
}

fn parse_rows(text: &str) -> Option<Vec<(BlockRow, i64)>> {
    let mut lines = text.lines();
    if lines.next()? != ROW_HEADER {
        return None;
    }
    let mut out = Vec::new();
    for l in lines {
        if l.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = l.split('\t').collect();
        if f.len() < 19 {
            return None;
        }
        let u = |i: usize| f[i].parse::<usize>().ok();
        let row = BlockRow {
            lo: u(0)?,
            hi: u(1)?,
            status: f[3].to_string(),
            synth_status: f[4].to_string(),
            syn_gates: u(5)?,
            secs: f[7].parse().ok()?,
            time_limit: f[8].parse().ok()?,
            support: u(9)?,
            changed: u(10)?,
            monomials: u(11)?,
            max_degree: u(12)?,
            pre_postopt: u(13)?,
            steps: u(14)?,
            restarts_used: u(15)?,
            deadline_hit: f[16] == "1",
            universe: u(17)?,
            note: f[18..].join("\t"),
            gates: None,
        };
        let off: i64 = f[6].parse().ok()?;
        if row.hi <= row.lo || row.hi - row.lo != u(2)? {
            return None;
        }
        out.push((row, off));
    }
    Some(out)
}

/// A cached group, when its tsv covers exactly `blocks` and its circuit
/// is consistent; `None` (recompute) otherwise.
fn load_group(
    tsv: &Path,
    circ: &Path,
    blocks: &[(usize, usize)],
    width: usize,
) -> Option<Vec<BlockRow>> {
    let text = fs::read_to_string(tsv).ok()?;
    let parsed = parse_rows(&text)?;
    let want: BTreeSet<(usize, usize)> = blocks.iter().copied().collect();
    let have: BTreeSet<(usize, usize)> = parsed.iter().map(|(r, _)| (r.lo, r.hi)).collect();
    if want != have || parsed.len() != blocks.len() {
        return None;
    }
    let mut rows = Vec::with_capacity(parsed.len());
    let need_circuit = parsed.iter().any(|(r, _)| r.status == "ok");
    let gates: Option<Vec<XGate>> = if need_circuit {
        let (g, w) = read_mpmct(circ.to_str()?).ok()?;
        if w != width {
            return None;
        }
        Some(g)
    } else {
        None
    };
    for (mut r, off) in parsed {
        if r.status == "ok" {
            let g = gates.as_ref()?;
            let off = usize::try_from(off).ok()?;
            let end = off.checked_add(r.syn_gates)?;
            if end > g.len() {
                return None;
            }
            r.gates = Some(g[off..end].to_vec());
        }
        rows.push(r);
    }
    Some(rows)
}

type SeedResults = FxHashMap<(usize, usize), BlockRow>;

/// Synthesize `blocks` with seed slot `k` under the budget `mult x base`,
/// in parallel, cached at `<stem>.{tsv,mpmct1}`.
#[allow(clippy::too_many_arguments)]
fn synth_group(
    ctx: &RoundContext<'_>,
    r: usize,
    input: &[XGate],
    k: usize,
    mult: usize,
    blocks: &[(usize, usize)],
    stem: &Path,
    log: Log<'_>,
) -> Result<Vec<BlockRow>, AdaptiveError> {
    let tsv = PathBuf::from(format!("{}.tsv", stem.display()));
    let circ = PathBuf::from(format!("{}.mpmct1", stem.display()));
    let tl = ctx.params.base_time_limit * mult as f64;
    if let Some(rows) = load_group(&tsv, &circ, blocks, ctx.width) {
        log(&format!(
            "round {r}: seed {k} budget {tl}s: {} blocks reloaded from {}",
            rows.len(),
            tsv.display()
        ));
        return Ok(rows);
    }
    let total = blocks.len();
    let gates_total: usize = blocks.iter().map(|b| b.1 - b.0).sum();
    log(&format!(
        "round {r}: seed {k} budget {tl}s: {total} blocks, {gates_total} gates, {} threads",
        rayon::current_num_threads()
    ));
    let seed = block_seed(ctx.stage_seed, r, k);
    let done = AtomicUsize::new(0);
    let t0 = Instant::now();
    let rows: Vec<BlockRow> = blocks
        .par_iter()
        .map(|&(lo, hi)| {
            let res = synthesize_block(&input[lo..hi], ctx.width, &ctx.params.block, tl, seed, lo);
            let row = BlockRow::of(lo, hi, tl, res);
            let d = done.fetch_add(1, Ordering::Relaxed) + 1;
            if !row.is_ok() || d.is_multiple_of(10) || d == total {
                log(&format!(
                    "round {r}: seed {k} [{d}/{total}] block [{lo},{hi}) {} -> {} {} ({}, {:.1}s) {}",
                    hi - lo,
                    row.syn_gates,
                    row.status,
                    row.synth_status,
                    row.secs,
                    row.note
                ));
            }
            row
        })
        .collect();
    let ok = rows.iter().filter(|r| r.is_ok()).count();
    log(&format!(
        "round {r}: seed {k} budget {tl}s: {ok}/{total} blocks synthesized in {:.1}s",
        t0.elapsed().as_secs_f64()
    ));
    save_group(&tsv, &circ, &rows, ctx.width)?;
    Ok(rows)
}

/// Synthesize the leaves `todo` for every seed, grouped by budget; the rows
/// go into `res[k]`.
fn synth_leaves(
    ctx: &RoundContext<'_>,
    r: usize,
    input: &[XGate],
    todo: &[(usize, usize)],
    area: &Path,
    res: &mut [SeedResults],
    log: Log<'_>,
) -> Result<(), AdaptiveError> {
    if todo.is_empty() {
        return Ok(());
    }
    fs::create_dir_all(area).map_err(|e| AdaptiveError::io(area, e))?;
    let mut by_budget: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
    for &l in todo {
        by_budget
            .entry(budget_multiplier(l.1 - l.0))
            .or_default()
            .push(l);
    }
    for (k, slot) in res.iter_mut().enumerate() {
        for (&mult, blocks) in &by_budget {
            let stem = area.join(format!("s{k}_t{mult}"));
            for row in synth_group(ctx, r, input, k, mult, blocks, &stem, log)? {
                slot.insert((row.lo, row.hi), row);
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- censuses

/// A structural digest of a gate list (which census a cached directory
/// belongs to).
pub fn gates_digest(gates: &[XGate]) -> u128 {
    let mut h = xxhash_rust::xxh3::Xxh3::new();
    let mut buf: Vec<u8> = Vec::with_capacity(32);
    for g in gates {
        buf.clear();
        buf.extend_from_slice(&g.target.to_le_bytes());
        buf.push(g.comp as u8);
        buf.push(g.ctrls.len() as u8);
        for &(w, p) in &g.ctrls {
            buf.extend_from_slice(&w.to_le_bytes());
            buf.push(p as u8);
        }
        h.update(&buf);
    }
    h.digest128()
}

const DIGEST_FILE: &str = "circuit.xxh3";

/// The census of `circuit` at `dir`: reloaded when the directory holds a
/// census of exactly this circuit, computed and written otherwise.
pub fn load_or_census(
    ctx: &RoundContext<'_>,
    circuit: &[XGate],
    dir: &Path,
    log: Log<'_>,
) -> Result<Census, AdaptiveError> {
    let digest = format!("{:032x}", gates_digest(circuit));
    let stamp = dir.join(DIGEST_FILE);
    if dir.is_dir()
        && fs::read_to_string(&stamp).is_ok_and(|s| s.trim() == digest)
        && let Ok(c) = census::read_dir(dir)
        && c.gates == circuit.len()
    {
        log(&format!(
            "census of {} gates reloaded from {}",
            circuit.len(),
            dir.display()
        ));
        return Ok(c);
    }
    let t0 = Instant::now();
    let c = census::census(ctx.source, circuit, ctx.width, ctx.params.census_batches);
    if dir.exists() {
        fs::remove_dir_all(dir).map_err(|e| AdaptiveError::io(dir, e))?;
    }
    fs::create_dir_all(dir).map_err(|e| AdaptiveError::io(dir, e))?;
    census::write_dir(&c, dir).map_err(|e| AdaptiveError::io(dir, e))?;
    write_text_atomic(&stamp, &format!("{digest}\n"))?;
    log(&format!(
        "census of {} gates: {} affine, {} direct intervals, audit {} ({:.1}s) -> {}",
        circuit.len(),
        c.affine.len(),
        c.direct.len(),
        if c.audit.all_passed() {
            "passed"
        } else {
            "FAILED"
        },
        t0.elapsed().as_secs_f64(),
        dir.display()
    ));
    Ok(c)
}

/// The planned interior cuts: reloaded from `dir/bounds` when present.
fn load_or_plan(
    ctx: &RoundContext<'_>,
    input: &[XGate],
    pcensus: &Census,
    prev_bounds: &[usize],
    dir: &Path,
    log: Log<'_>,
) -> Result<Vec<usize>, AdaptiveError> {
    let m = input.len();
    let bf = dir.join("bounds");
    if bf.is_file()
        && let Ok(b) = planner::read_bounds(&bf)
        && valid_bounds(&b, m)
    {
        log(&format!(
            "plan reloaded from {}: {} blocks",
            bf.display(),
            b.len() + 1
        ));
        return Ok(b);
    }
    let t0 = Instant::now();
    let plan = planner::plan(
        input,
        ctx.width,
        pcensus,
        ctx.source,
        &ctx.params.plan,
        prev_bounds,
    );
    if !valid_bounds(&plan.bounds, m) {
        return Err(AdaptiveError::Internal(format!(
            "planner returned {} bounds that are not strictly increasing inside (0, {m})",
            plan.bounds.len()
        )));
    }
    fs::create_dir_all(dir).map_err(|e| AdaptiveError::io(dir, e))?;
    planner::write_plan(&plan, dir).map_err(|e| AdaptiveError::io(dir, e))?;
    if !bf.is_file() {
        // the planner's own layout did not include `bounds`: keep our copy
        let mut s = String::new();
        for b in &plan.bounds {
            let _ = writeln!(s, "{b}");
        }
        write_text_atomic(&bf, &s)?;
    }
    log(&format!(
        "plan: {} blocks ({} seams, {} single-gate exceptions) in {:.1}s",
        plan.bounds.len() + 1,
        plan.seams.len(),
        plan.single_gate_exceptions,
        t0.elapsed().as_secs_f64()
    ));
    Ok(plan.bounds)
}

fn valid_bounds(b: &[usize], m: usize) -> bool {
    b.windows(2).all(|w| w[0] < w[1]) && b.iter().all(|&x| x > 0 && x < m)
}

/// Concatenate, per leaf, the replacement `pick(i, leaf)` returns or the
/// input's block; with the output's edges.
fn assemble<'b>(
    input: &[XGate],
    leaves: &[(usize, usize)],
    pick: impl Fn(usize, (usize, usize)) -> Option<&'b [XGate]>,
) -> (Vec<XGate>, Vec<usize>) {
    let mut out: Vec<XGate> = Vec::with_capacity(input.len());
    let mut edges = Vec::with_capacity(leaves.len() + 1);
    edges.push(0);
    for (i, &l) in leaves.iter().enumerate() {
        match pick(i, l) {
            Some(g) => out.extend_from_slice(g),
            None => out.extend_from_slice(&input[l.0..l.1]),
        }
        edges.push(out.len());
    }
    (out, edges)
}

// ---------------------------------------------------------------- the round

/// Run (or reload) round `r` on `input` into `out_dir`.
pub fn run_round(
    ctx: &RoundContext<'_>,
    r: usize,
    input: &[XGate],
    prev_bounds: &[usize],
    out_dir: &Path,
    log: Log<'_>,
) -> Result<RoundResult, AdaptiveError> {
    let t0 = Instant::now();
    let p = ctx.params;
    p.validate().map_err(AdaptiveError::Usage)?;
    fs::create_dir_all(out_dir).map_err(|e| AdaptiveError::io(out_dir, e))?;
    let report_path = out_dir.join("result.report");
    if report_path.is_file() {
        let text =
            fs::read_to_string(&report_path).map_err(|e| AdaptiveError::io(&report_path, e))?;
        let mut res = parse_round_report(&text)
            .map_err(|e| AdaptiveError::Input(format!("{}: {e}", report_path.display())))?;
        if res.round != r {
            return Err(AdaptiveError::Input(format!(
                "{}: holds round {}, expected round {r}",
                report_path.display(),
                res.round
            )));
        }
        if res.input_gates != input.len() {
            return Err(AdaptiveError::Input(format!(
                "{}: round {r} ran on {} gates, the input now has {}",
                report_path.display(),
                res.input_gates,
                input.len()
            )));
        }
        res.reloaded = true;
        log(&format!(
            "round {r}: reloaded from {} ({} -> {} gates)",
            report_path.display(),
            res.input_gates,
            res.output_gates
        ));
        return Ok(res);
    }
    let m = input.len();
    let width = ctx.width;
    if m == 0 {
        return Err(AdaptiveError::Input(
            "the input circuit has no gates".into(),
        ));
    }
    let seeds = p.seeds_per_block;
    log(&format!(
        "round {r}: {m} gates, {width} wires; seeds/block {seeds}, budgets {}/{}/{}s, refinement {} levels (> {} gates), accept {}, blind {}",
        p.base_time_limit,
        p.base_time_limit * 2.0,
        p.base_time_limit * 4.0,
        p.refinement_levels,
        p.min_refine_gates,
        p.accept.name(),
        p.blind
    ));

    // ---- input census and plan
    let cin = load_or_census(ctx, input, &out_dir.join("input_census"), log)?;
    let empty;
    let pcensus: &Census = if p.blind {
        empty = Census::empty(m, width);
        &empty
    } else {
        &cin
    };
    let bounds = load_or_plan(
        ctx,
        input,
        pcensus,
        prev_bounds,
        &out_dir.join("partition"),
        log,
    )?;
    let initial_blocks = bounds.len() + 1;
    let mut leaves: Vec<(usize, usize)> = {
        let mut e = Vec::with_capacity(bounds.len() + 2);
        e.push(0);
        e.extend_from_slice(&bounds);
        e.push(m);
        e.windows(2).map(|w| (w[0], w[1])).collect()
    };

    // ---- synthesis and refinement
    let mut res: Vec<SeedResults> = (0..seeds).map(|_| SeedResults::default()).collect();
    synth_leaves(
        ctx,
        r,
        input,
        &leaves,
        &out_dir.join("synth0"),
        &mut res,
        log,
    )?;
    let profile = Profile::new(pcensus, ctx.source);
    let mut refinement: Vec<RefinementLevel> = Vec::new();
    for level in 1..=p.refinement_levels {
        let failed: Vec<(usize, usize)> = leaves
            .iter()
            .copied()
            .filter(|l| {
                l.1 - l.0 > p.min_refine_gates
                    && res.iter().all(|s| !s.get(l).is_some_and(|row| row.is_ok()))
            })
            .collect();
        if failed.is_empty() {
            break;
        }
        let mut children: Vec<(usize, usize)> = Vec::with_capacity(2 * failed.len());
        for &(lo, hi) in &failed {
            let cut = profile.split(lo, hi);
            if !(lo < cut && cut < hi) {
                return Err(AdaptiveError::Internal(format!(
                    "refinement split of [{lo}, {hi}) returned {cut}"
                )));
            }
            children.push((lo, cut));
            children.push((cut, hi));
        }
        let failed_set: BTreeSet<(usize, usize)> = failed.iter().copied().collect();
        leaves.retain(|l| !failed_set.contains(l));
        leaves.extend_from_slice(&children);
        leaves.sort_unstable();
        log(&format!(
            "round {r}: refinement level {level}: {} blocks split into {} children",
            failed.len(),
            children.len()
        ));
        synth_leaves(
            ctx,
            r,
            input,
            &children,
            &out_dir.join(format!("refine{level}")),
            &mut res,
            log,
        )?;
        let synthesized = children
            .iter()
            .map(|c| {
                res.iter()
                    .filter(|s| s.get(c).is_some_and(|row| row.is_ok()))
                    .count()
            })
            .sum();
        refinement.push(RefinementLevel {
            level,
            split: failed.len(),
            children: children.len(),
            synthesized,
        });
    }
    if leaves.first().map(|l| l.0) != Some(0)
        || leaves.last().map(|l| l.1) != Some(m)
        || leaves.windows(2).any(|w| w[0].1 != w[1].0)
    {
        return Err(AdaptiveError::Internal(
            "the leaves do not partition the input".into(),
        ));
    }

    // ---- acceptance
    let edges = edges_of(&leaves);
    let mut exposure_in: Option<Vec<BTreeSet<usize>>> = None;
    let mut exposure_cand: Vec<Vec<BTreeSet<usize>>> = Vec::new();
    if p.accept == Accept::SmallerNotLeakier {
        exposure_in = Some(cin.interior_sets(&edges, ctx.source));
        for (k, slot) in res.iter().enumerate() {
            let (cand, oedges) = assemble(input, &leaves, |_, l| {
                slot.get(&l).and_then(|row| row.gates.as_deref())
            });
            log(&format!(
                "round {r}: candidate of seed {k}: {} gates",
                cand.len()
            ));
            let cs = load_or_census(
                ctx,
                &cand,
                &out_dir.join(format!("candidate_s{k}_census")),
                log,
            )?;
            drop(cand);
            exposure_cand.push(cs.interior_sets(&oedges, ctx.source));
        }
    }
    let mut picks = Picks::new(seeds);
    let mut chosen: Vec<Option<usize>> = Vec::with_capacity(leaves.len());
    for (i, &l) in leaves.iter().enumerate() {
        let cands: Vec<Candidate> = res
            .iter()
            .enumerate()
            .filter_map(|(k, slot)| {
                slot.get(&l).filter(|row| row.is_ok()).map(|row| Candidate {
                    seed: k,
                    gates: row.syn_gates,
                    exposure: exposure_cand.get(k).map_or(0, |e| e[i].len()),
                })
            })
            .collect();
        chosen.push(choose(
            l.1 - l.0,
            exposure_in.as_ref().map(|e| e[i].len()),
            &cands,
            &mut picks,
        ));
    }
    let coverage = Coverage {
        synthesized_blocks: chosen.iter().filter(|c| c.is_some()).count(),
        input_gates_synthesized: leaves
            .iter()
            .zip(&chosen)
            .filter(|(_, c)| c.is_some())
            .map(|(l, _)| l.1 - l.0)
            .sum(),
    };

    // ---- output
    let (output, oedges) = assemble(input, &leaves, |i, l| {
        chosen[i].and_then(|k| res[k].get(&l).and_then(|row| row.gates.as_deref()))
    });
    drop(res);
    drop(exposure_cand);
    drop(exposure_in);
    let equiv_seed = EQUIV_SEED ^ (r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    if !same_function(input, &output, width, p.equiv_batches, equiv_seed) {
        return Err(AdaptiveError::Verify(format!(
            "round {r}: the assembled output ({} gates) differs from the input ({m} gates) on random full states",
            output.len()
        )));
    }
    let equivalence_samples = p.equiv_batches * 1024;
    let cout = load_or_census(ctx, &output, &out_dir.join("output_census"), log)?;
    let leakage_in = Leakage::of(&cin, ctx.source);
    let leakage_out = Leakage::of(&cout, ctx.source);
    let output_boundaries: Vec<usize> = oedges[1..oedges.len() - 1].to_vec();
    let out_path = out_dir.join("output.mpmct1");
    write_mpmct_atomic(path_str(&out_path)?, &output, width)
        .map_err(|e| AdaptiveError::io(&out_path, e))?;
    let mut bs = String::new();
    for b in &output_boundaries {
        let _ = writeln!(bs, "{b}");
    }
    write_text_atomic(&out_dir.join("bounds.out"), &bs)?;
    let result = RoundResult {
        round: r,
        input_gates: m,
        output_gates: output.len(),
        blocks: leaves.len(),
        initial_blocks,
        refinement,
        picks,
        coverage,
        leakage_in,
        leakage_out,
        output_boundaries,
        equivalence_samples,
        seconds: t0.elapsed().as_secs_f64(),
        reloaded: false,
        output: Some(output),
    };
    write_text_atomic(&report_path, &round_report(&result, ctx, width))?;
    log(&format!(
        "round {r} done: {m} -> {} gates, {} blocks ({} synthesized), picks orig {} seeds {:?} rejected leakier {} growing {}; deeper affine {} -> {} (validated); {:.1}s",
        result.output_gates,
        result.blocks,
        result.coverage.synthesized_blocks,
        result.picks.orig,
        result.picks.seed,
        result.picks.rejected_leakier,
        result.picks.rejected_growing,
        result.leakage_in.validated.affine.deeper_births,
        result.leakage_out.validated.affine.deeper_births,
        result.seconds
    ));
    Ok(result)
}

// ---------------------------------------------------------------- report

fn family_kv(out: &mut String, prefix: &str, f: &FamilyMetrics) {
    let _ = writeln!(
        out,
        "{prefix}.all_internal_births={}",
        f.all_internal_births
    );
    let _ = writeln!(out, "{prefix}.deeper_births={}", f.deeper_births);
    let _ = writeln!(
        out,
        "{prefix}.cut_exposure_fraction={:.6}",
        f.cut_exposure_fraction
    );
}

/// The `leakage_*` keys of a census (shared with the final report).
pub fn leakage_kv(out: &mut String, prefix: &str, l: &Leakage) {
    for (name, m) in [("discovery", &l.discovery), ("validated", &l.validated)] {
        family_kv(out, &format!("{prefix}.{name}.direct"), &m.direct);
        family_kv(out, &format!("{prefix}.{name}.affine"), &m.affine);
    }
    let _ = writeln!(out, "{prefix}.audit_passed={}", l.audit_passed);
}

/// `result.report` (key=value; every key is read back by
/// `parse_round_report`).
pub fn round_report(res: &RoundResult, ctx: &RoundContext<'_>, width: usize) -> String {
    let p = ctx.params;
    let mut s = String::new();
    let _ = writeln!(s, "adaptive_round_report=1");
    let _ = writeln!(s, "status=ok");
    let _ = writeln!(s, "round={}", res.round);
    let _ = writeln!(s, "width={width}");
    let _ = writeln!(s, "input_gates={}", res.input_gates);
    let _ = writeln!(s, "output_gates={}", res.output_gates);
    let _ = writeln!(s, "blocks={}", res.blocks);
    let _ = writeln!(s, "initial_blocks={}", res.initial_blocks);
    let _ = writeln!(s, "refinement_levels={}", res.refinement.len());
    for l in &res.refinement {
        let _ = writeln!(s, "refinement.{}.split={}", l.level, l.split);
        let _ = writeln!(s, "refinement.{}.children={}", l.level, l.children);
        let _ = writeln!(s, "refinement.{}.synthesized={}", l.level, l.synthesized);
    }
    let _ = writeln!(s, "seeds_per_block={}", res.picks.seed.len());
    let _ = writeln!(s, "picks.orig={}", res.picks.orig);
    for (k, v) in res.picks.seed.iter().enumerate() {
        let _ = writeln!(s, "picks.seed.{k}={v}");
    }
    let _ = writeln!(s, "picks.rejected_leakier={}", res.picks.rejected_leakier);
    let _ = writeln!(s, "picks.rejected_growing={}", res.picks.rejected_growing);
    let _ = writeln!(
        s,
        "coverage.synthesized_blocks={}",
        res.coverage.synthesized_blocks
    );
    let _ = writeln!(
        s,
        "coverage.input_gates_synthesized={}",
        res.coverage.input_gates_synthesized
    );
    leakage_kv(&mut s, "leakage_in", &res.leakage_in);
    leakage_kv(&mut s, "leakage_out", &res.leakage_out);
    let _ = writeln!(s, "equivalence_samples={}", res.equivalence_samples);
    let _ = writeln!(s, "output_boundaries={}", res.output_boundaries.len());
    let _ = writeln!(s, "output_boundaries_file=bounds.out");
    let _ = writeln!(s, "param.time_limit={}", p.base_time_limit);
    let _ = writeln!(s, "param.restarts={}", p.block.synth.restarts);
    let _ = writeln!(s, "param.beam={}", p.block.synth.beam);
    let _ = writeln!(s, "param.term_cap={}", p.plan.term_cap);
    let _ = writeln!(s, "param.max_block_gates={}", p.plan.max_block_gates);
    let _ = writeln!(s, "param.segments={}", p.plan.segments);
    let _ = writeln!(s, "param.avoid_radius={}", p.plan.avoid_radius);
    let _ = writeln!(s, "param.refinement_levels={}", p.refinement_levels);
    let _ = writeln!(s, "param.min_refine_gates={}", p.min_refine_gates);
    let _ = writeln!(s, "param.census_batches={}", p.census_batches);
    let _ = writeln!(s, "param.accept={}", p.accept.name());
    let _ = writeln!(s, "param.blind={}", p.blind);
    let _ = writeln!(s, "seconds={:.3}", res.seconds);
    s
}

/// `key=value` lines (the first `=` splits; blank lines and `#` comments
/// are skipped).
pub fn parse_kv(text: &str) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    for l in text.lines() {
        let l = l.trim_end();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        if let Some((k, v)) = l.split_once('=') {
            m.insert(k.to_string(), v.to_string());
        }
    }
    m
}

fn kv_usize(m: &BTreeMap<String, String>, k: &str) -> Result<usize, String> {
    m.get(k)
        .ok_or_else(|| format!("missing {k}"))?
        .parse()
        .map_err(|_| format!("{k} is not an integer"))
}

fn kv_f64(m: &BTreeMap<String, String>, k: &str) -> Result<f64, String> {
    m.get(k)
        .ok_or_else(|| format!("missing {k}"))?
        .parse()
        .map_err(|_| format!("{k} is not a number"))
}

fn kv_bool(m: &BTreeMap<String, String>, k: &str) -> Result<bool, String> {
    match m.get(k).map(|s| s.as_str()) {
        Some("true") => Ok(true),
        Some("false") => Ok(false),
        Some(_) => Err(format!("{k} is not a boolean")),
        None => Err(format!("missing {k}")),
    }
}

fn parse_family(m: &BTreeMap<String, String>, prefix: &str) -> Result<FamilyMetrics, String> {
    Ok(FamilyMetrics {
        all_internal_births: kv_usize(m, &format!("{prefix}.all_internal_births"))?,
        deeper_births: kv_usize(m, &format!("{prefix}.deeper_births"))?,
        cut_exposure_fraction: kv_f64(m, &format!("{prefix}.cut_exposure_fraction"))?,
        ids: BTreeSet::new(),
    })
}

/// The `leakage_*` keys back (without the id sets).
pub fn parse_leakage(m: &BTreeMap<String, String>, prefix: &str) -> Result<Leakage, String> {
    let metrics = |name: &str| -> Result<LeakageMetrics, String> {
        Ok(LeakageMetrics {
            direct: parse_family(m, &format!("{prefix}.{name}.direct"))?,
            affine: parse_family(m, &format!("{prefix}.{name}.affine"))?,
        })
    };
    Ok(Leakage {
        discovery: metrics("discovery")?,
        validated: metrics("validated")?,
        audit_passed: kv_bool(m, &format!("{prefix}.audit_passed"))?,
    })
}

/// A `result.report` back to a `RoundResult` (`output_boundaries` is read
/// from `bounds.out` by the caller when needed; here it is empty).
pub fn parse_round_report(text: &str) -> Result<RoundResult, String> {
    let m = parse_kv(text);
    if m.get("adaptive_round_report").map(|s| s.as_str()) != Some("1") {
        return Err("not an adaptive round report".into());
    }
    if m.get("status").map(|s| s.as_str()) != Some("ok") {
        return Err("status is not ok".into());
    }
    let levels = kv_usize(&m, "refinement_levels")?;
    let mut refinement = Vec::with_capacity(levels);
    for level in 1..=levels {
        refinement.push(RefinementLevel {
            level,
            split: kv_usize(&m, &format!("refinement.{level}.split"))?,
            children: kv_usize(&m, &format!("refinement.{level}.children"))?,
            synthesized: kv_usize(&m, &format!("refinement.{level}.synthesized"))?,
        });
    }
    let seeds = kv_usize(&m, "seeds_per_block")?;
    let mut picks = Picks::new(seeds);
    picks.orig = kv_usize(&m, "picks.orig")?;
    for (k, slot) in picks.seed.iter_mut().enumerate() {
        *slot = kv_usize(&m, &format!("picks.seed.{k}"))?;
    }
    picks.rejected_leakier = kv_usize(&m, "picks.rejected_leakier")?;
    picks.rejected_growing = kv_usize(&m, "picks.rejected_growing")?;
    Ok(RoundResult {
        round: kv_usize(&m, "round")?,
        input_gates: kv_usize(&m, "input_gates")?,
        output_gates: kv_usize(&m, "output_gates")?,
        blocks: kv_usize(&m, "blocks")?,
        initial_blocks: kv_usize(&m, "initial_blocks")?,
        refinement,
        picks,
        coverage: Coverage {
            synthesized_blocks: kv_usize(&m, "coverage.synthesized_blocks")?,
            input_gates_synthesized: kv_usize(&m, "coverage.input_gates_synthesized")?,
        },
        leakage_in: parse_leakage(&m, "leakage_in")?,
        leakage_out: parse_leakage(&m, "leakage_out")?,
        output_boundaries: Vec::new(),
        equivalence_samples: kv_usize(&m, "equivalence_samples")?,
        seconds: kv_f64(&m, "seconds")?,
        reloaded: false,
        output: None,
    })
}

/// `bounds.out` (one interior cut per line).
pub fn read_bounds_file(path: &Path) -> Result<Vec<usize>, AdaptiveError> {
    let text = fs::read_to_string(path).map_err(|e| AdaptiveError::io(path, e))?;
    text.split_whitespace()
        .map(|t| {
            t.parse::<usize>().map_err(|_| {
                AdaptiveError::Input(format!("{}: {t:?} is not a cut", path.display()))
            })
        })
        .collect()
}

// ---------------------------------------------------------------- files

pub(crate) fn path_str(p: &Path) -> Result<&str, AdaptiveError> {
    p.to_str()
        .ok_or_else(|| AdaptiveError::Usage(format!("path {} is not valid UTF-8", p.display())))
}

/// A text file written to `<path>.tmp`, synced and renamed over `path`.
pub(crate) fn write_text_atomic(path: &Path, text: &str) -> Result<(), AdaptiveError> {
    use std::io::Write as _;
    let tmp = PathBuf::from(format!("{}.tmp", path.display()));
    let r = (|| -> io::Result<()> {
        let mut o = fs::OpenOptions::new();
        o.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            o.mode(0o600);
        }
        let mut f = o.open(&tmp)?;
        f.write_all(text.as_bytes())?;
        f.sync_all()?;
        fs::rename(&tmp, path)
    })();
    if let Err(e) = r {
        let _ = fs::remove_file(&tmp);
        return Err(AdaptiveError::io(path, e));
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/adaptive/round.rs"]
mod tests;
