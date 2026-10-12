//! The per-window pipeline and the whole-deliverable driver.
//!
//! For every window of the deliverable (in parallel):
//!
//! 1. build the local universe and the spec from the decoded window
//!    (`spec`); the synthesizer sees nothing else;
//! 2. `synth::synthesize` at the monomial width of the universe;
//! 3. map the gates to global wires and check them: sampled
//!    (`verify::check_window_sampled`) and exact
//!    (`verify::check_window_exact`, retried at a larger budget when the
//!    composition overflows);
//! 4. the pairwise peephole pass (`postopt`), re-checked the same way;
//!    kept only when it passes;
//! 5. a window that did not converge or failed a check falls back to the
//!    packed circuit's window gates (when given and allowed), after those
//!    are checked exactly against the deliverable too; otherwise it is an
//!    error (exit 4).
//!
//! The results are concatenated in window order; the caller canonicalizes
//! the whole list and runs the circuit-level checks.
use super::postopt::peephole;
use super::spec::{AnfStats, LocalUniverse, anf_stats, build_spec};
use super::synth::{NTAGS, SynthOutput, SynthStatus, synthesize};
use super::types::{Stage7Error, UnpackParams};
use super::verify::{ExactFailure, check_window_exact, check_window_sampled};
use crate::circuit::xgate::XGate;
use crate::stages::packing::mono::{MonoWidthFn, mono_words};
use crate::stages::packing::serialize::{Deliverable, EMIT_BUDGET, WindowView};
use rayon::prelude::*;
use std::time::Instant;

/// Budgets tried by the exact window check, in order: stage 4's emission
/// budget (every packed window fits it), then 4x for synthesized windows
/// whose intermediate composition is larger than the original's.
pub const EXACT_BUDGETS: [usize; 2] = [EMIT_BUDGET, 4 * EMIT_BUDGET];

/// The packed circuit (stage 6's `rep.mpmct1` + `.bounds`) as the
/// verbatim fallback source; `cuts` = `[0] + bounds + [m]`.
pub struct PackedWindows<'a> {
    pub gates: &'a [XGate],
    pub cuts: Vec<usize>,
}

impl PackedWindows<'_> {
    pub fn window_count(&self) -> usize {
        self.cuts.len().saturating_sub(1)
    }
    pub fn window(&self, i: usize) -> &[XGate] {
        &self.gates[self.cuts[i]..self.cuts[i + 1]]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WinStatus {
    /// Synthesized from the ANF and checked.
    Ok,
    /// The packed circuit's window, emitted verbatim (checked too).
    Fallback,
    /// Synthesis or a check failed and the caller asked for failures to be
    /// reported instead of handled (`report_failures`): no gates. Only the
    /// refinement loop (`refine`) sees this status.
    Failed,
    /// The packed circuit's window, emitted verbatim by the refinement loop
    /// (a 1-gate failure or a failure at `max_rounds`).
    Verbatim,
}

impl WinStatus {
    pub fn name(self) -> &'static str {
        match self {
            WinStatus::Ok => "ok",
            WinStatus::Fallback => "fallback",
            WinStatus::Failed => "failed",
            WinStatus::Verbatim => "verbatim",
        }
    }
}

/// One window's outcome (the `windows.tsv` row).
#[derive(Clone, Debug)]
pub struct WindowResult {
    pub idx: usize,
    pub gates: Vec<XGate>,
    pub status: WinStatus,
    /// `converged`, `stepcap`, `timeout`, `stalled`, `verify-fail`,
    /// `exact-overflow`.
    pub synth_status: &'static str,
    pub secs: f64,
    pub stats: AnfStats,
    /// Local universe size and its monomial words.
    pub universe: usize,
    pub width: usize,
    pub tags: [usize; NTAGS],
    pub pre_postopt: usize,
    pub steps: usize,
    pub note: String,
    pub restarts_used: usize,
    pub deadline_hit: bool,
    /// Budget at which the exact check passed (0 for a fallback window
    /// confirmed at the emission budget).
    pub exact_budget: usize,
    /// `(lo, hi)` in the packed circuit, when known.
    pub orig: Option<(usize, usize)>,
}

impl WindowResult {
    pub fn orig_gates(&self) -> Option<usize> {
        self.orig.map(|(lo, hi)| hi - lo)
    }
}

struct Synth<'a> {
    view: &'a WindowView<'a>,
    uni: &'a LocalUniverse,
    params: &'a UnpackParams,
    idx: usize,
}

impl MonoWidthFn<SynthOutput> for Synth<'_> {
    fn call<const W: usize>(self) -> SynthOutput {
        let spec = build_spec::<W>(self.view, self.uni);
        synthesize(&spec, &self.params.synth, self.params.seed, self.idx)
    }
}

/// Both window checks; `Ok(budget)` names the budget the exact check
/// passed at.
fn check_window(
    gates: &[XGate],
    view: &WindowView<'_>,
    nw: usize,
    seed: u64,
) -> Result<usize, &'static str> {
    check_window_sampled(gates, view, nw, seed).map_err(|_| "verify-fail")?;
    let mut last = ExactFailure::Mismatch;
    for &budget in &EXACT_BUDGETS {
        match check_window_exact(gates, view, nw, budget) {
            Ok(()) => return Ok(budget),
            Err(ExactFailure::Mismatch) => return Err("verify-fail"),
            Err(e) => last = e,
        }
    }
    debug_assert!(matches!(last, ExactFailure::Overflow(_)));
    Err("exact-overflow")
}

/// One window: synthesis, checks, peephole, then (on failure) the fallback,
/// an error, or — with `report_failures` — a `WinStatus::Failed` result
/// (no gates) for the caller to handle.
fn process_window(
    d: &Deliverable,
    i: usize,
    packed: Option<&PackedWindows<'_>>,
    params: &UnpackParams,
    report_failures: bool,
) -> Result<WindowResult, Stage7Error> {
    let t0 = Instant::now();
    let nw = d.state_wires as usize;
    let view = d.window(i);
    let stats = anf_stats(&view);
    let uni = LocalUniverse::for_window(&view, nw, params.ancilla_pool);
    let width = mono_words(uni.len()).ok_or_else(|| {
        Stage7Error::Internal(format!("window {i}: universe of {} wires", uni.len()))
    })?;
    let mut res = WindowResult {
        idx: i,
        gates: Vec::new(),
        status: WinStatus::Fallback,
        synth_status: "",
        secs: 0.0,
        stats,
        universe: uni.len(),
        width,
        tags: [0; NTAGS],
        pre_postopt: 0,
        steps: 0,
        note: String::new(),
        restarts_used: 0,
        deadline_hit: false,
        exact_budget: 0,
        orig: packed.map(|p| (p.cuts[i], p.cuts[i + 1])),
    };
    // ---- the only call that produces gates from the ANF
    let out = crate::stages::packing::mono::with_mono_width(
        uni.len(),
        Synth {
            view: &view,
            uni: &uni,
            params,
            idx: i,
        },
    );
    res.synth_status = out.status.name();
    res.steps = out.steps;
    res.note = out.note;
    res.restarts_used = out.restarts_used;
    res.deadline_hit = out.deadline_hit;
    let seed = params.verify_seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    if out.status == SynthStatus::Converged {
        for t in &out.tags {
            res.tags[*t as usize] += 1;
        }
        let local = out.gates;
        res.pre_postopt = local.len();
        let gates = uni.to_xgates(&local);
        match check_window(&gates, &view, nw, seed) {
            Ok(budget) => {
                res.exact_budget = budget;
                res.gates = gates;
                res.status = WinStatus::Ok;
                if params.postopt {
                    let mut opt = local;
                    if peephole(&mut opt) > 0 {
                        let g2 = uni.to_xgates(&opt);
                        match check_window(&g2, &view, nw, seed ^ 0xBEEF) {
                            Ok(budget) => {
                                res.exact_budget = budget;
                                res.gates = g2;
                            }
                            Err(why) => {
                                res.note = format!("postopt {why}; kept unoptimised");
                            }
                        }
                    }
                }
            }
            Err(why) => {
                res.synth_status = why;
                res.note = format!("{why} on the synthesized window");
            }
        }
    }
    if res.status != WinStatus::Ok && report_failures {
        res.status = WinStatus::Failed;
        res.secs = t0.elapsed().as_secs_f64();
        return Ok(res);
    }
    if res.status != WinStatus::Ok {
        let Some(p) = packed.filter(|_| params.allow_fallback) else {
            return Err(Stage7Error::Verify(format!(
                "window {i}: synthesis {} ({}); no fallback{}",
                res.synth_status,
                res.note,
                if packed.is_none() {
                    " (no --packed circuit)"
                } else {
                    " (--no-fallback)"
                }
            )));
        };
        let gates = p.window(i).to_vec();
        match check_window(&gates, &view, nw, seed ^ 0xFA11) {
            Ok(_) => {}
            Err(why) => {
                return Err(Stage7Error::Verify(format!(
                    "window {i}: the packed circuit's window [{}, {}) != the deliverable ({why})",
                    p.cuts[i],
                    p.cuts[i + 1]
                )));
            }
        }
        res.gates = gates;
        res.status = WinStatus::Fallback;
    }
    res.secs = t0.elapsed().as_secs_f64();
    Ok(res)
}

/// Synthesize every window of `d` (in parallel); `progress` is called with
/// each finished window and the count so far. Results come back in window
/// order.
pub fn unpack_windows(
    d: &Deliverable,
    packed: Option<&PackedWindows<'_>>,
    params: &UnpackParams,
    progress: &(dyn Fn(usize, &WindowResult) + Sync),
) -> Result<Vec<WindowResult>, Stage7Error> {
    let all: Vec<usize> = (0..d.window_count()).collect();
    unpack_selected(d, packed, params, &all, false, progress)
}

/// Synthesize the windows `sel` of `d` (indices in `d`'s lattice, which key
/// the tie-break PRNG and the verify seed; any order, no duplicates), in
/// parallel. With `report_failures` a window that fails comes back as
/// `WinStatus::Failed` instead of falling back or erroring. Results come
/// back in the order of `sel`.
pub fn unpack_selected(
    d: &Deliverable,
    packed: Option<&PackedWindows<'_>>,
    params: &UnpackParams,
    sel: &[usize],
    report_failures: bool,
    progress: &(dyn Fn(usize, &WindowResult) + Sync),
) -> Result<Vec<WindowResult>, Stage7Error> {
    params.validate().map_err(Stage7Error::Usage)?;
    if let Some(&i) = sel.iter().find(|&&i| i >= d.window_count()) {
        return Err(Stage7Error::Usage(format!(
            "window {i} selected, the deliverable has {} windows",
            d.window_count()
        )));
    }
    let nw = d.state_wires as usize;
    if mono_words(nw).is_none() {
        return Err(Stage7Error::Usage(format!(
            "stage 7 supports at most 1024 state wires, got {nw}"
        )));
    }
    if let Some(p) = packed {
        if p.window_count() != d.window_count() {
            return Err(Stage7Error::Input(format!(
                "the packed circuit has {} windows, the deliverable {}",
                p.window_count(),
                d.window_count()
            )));
        }
        if let Some(w) = p.gates.iter().map(|g| g.max_wire() as usize).max()
            && w >= nw
        {
            return Err(Stage7Error::Input(format!(
                "the packed circuit uses wire {w}, beyond the {nw}-wire state"
            )));
        }
    }
    let done = std::sync::atomic::AtomicUsize::new(0);
    let results: Vec<Result<WindowResult, Stage7Error>> = sel
        .par_iter()
        .map(|&i| {
            let r = process_window(d, i, packed, params, report_failures);
            if let Ok(r) = &r {
                let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                progress(n, r);
            }
            r
        })
        .collect();
    results.into_iter().collect()
}

/// The windows' gates in window order and each window's offset in it.
pub fn concatenate(results: &[WindowResult]) -> (Vec<XGate>, Vec<usize>) {
    let total: usize = results.iter().map(|r| r.gates.len()).sum();
    let mut gates = Vec::with_capacity(total);
    let mut offsets = Vec::with_capacity(results.len());
    for r in results {
        offsets.push(gates.len());
        gates.extend(r.gates.iter().cloned());
    }
    (gates, offsets)
}

/// Nearest-rank quantile of an unsorted sample (`anf_synth`'s `quantile`).
pub fn quantile(xs: &[f64], q: f64) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let k = ((q * (s.len() - 1) as f64).round() as usize).min(s.len() - 1);
    s[k]
}

/// Totals over a run's windows (the `unpacking.report` numbers).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct UnpackSummary {
    pub windows: usize,
    pub synthesized: usize,
    /// Windows emitted verbatim from the packed circuit (`Fallback` and
    /// `Verbatim`).
    pub fallback: usize,
    /// `WinStatus::Failed` windows (only under `report_failures`).
    pub failed: usize,
    pub deadline_hits: usize,
    pub verify_fail: usize,
    pub exact_overflow: usize,
    pub exact_retried: usize,
    /// Sum of the packed windows' gates, when known.
    pub gates_in: Option<usize>,
    pub gates_out: usize,
    pub gates_pre_postopt: usize,
    pub tags: [usize; NTAGS],
    pub secs: Vec<f64>,
    pub restarts_used: Vec<f64>,
    pub status_counts: Vec<(&'static str, usize)>,
}

impl UnpackSummary {
    pub fn of(results: &[WindowResult]) -> UnpackSummary {
        let mut s = UnpackSummary {
            windows: results.len(),
            gates_in: results.iter().map(|r| r.orig_gates()).sum(),
            ..Default::default()
        };
        for r in results {
            s.gates_out += r.gates.len();
            s.secs.push(r.secs);
            s.restarts_used.push(r.restarts_used as f64);
            if r.deadline_hit {
                s.deadline_hits += 1;
            }
            match r.status {
                WinStatus::Ok => {
                    s.synthesized += 1;
                    s.gates_pre_postopt += r.pre_postopt;
                    for i in 0..NTAGS {
                        s.tags[i] += r.tags[i];
                    }
                    if r.exact_budget > EXACT_BUDGETS[0] {
                        s.exact_retried += 1;
                    }
                }
                WinStatus::Fallback | WinStatus::Verbatim => s.fallback += 1,
                WinStatus::Failed => s.failed += 1,
            }
            match r.synth_status {
                "verify-fail" => s.verify_fail += 1,
                "exact-overflow" => s.exact_overflow += 1,
                _ => {}
            }
            match s
                .status_counts
                .iter_mut()
                .find(|(k, _)| *k == r.synth_status)
            {
                Some(e) => e.1 += 1,
                None => s.status_counts.push((r.synth_status, 1)),
            }
        }
        s
    }

    pub fn ratio(&self) -> Option<f64> {
        self.gates_in
            .filter(|&g| g > 0)
            .map(|g| self.gates_out as f64 / g as f64)
    }
}

/// `windows.tsv`: the `anf_synth` contract columns (`idx orig_lo orig_hi
/// orig_gates syn_gates changed_wires input_vars monomials max_degree status
/// seconds syn_offset`; `orig_*` are -1 without a packed circuit) followed
/// by `synth_status pre_postopt_gates steps note tag_* universe width
/// exact_budget restarts_used deadline_hit`. `syn_offset` locates the
/// window in the concatenation BEFORE canonical reordering.
pub fn windows_tsv(results: &[WindowResult], offsets: &[usize]) -> String {
    use std::fmt::Write as _;
    let mut s = String::from(
        "idx\torig_lo\torig_hi\torig_gates\tsyn_gates\tchanged_wires\tinput_vars\tmonomials\tmax_degree\tstatus\tseconds\tsyn_offset\tsynth_status\tpre_postopt_gates\tsteps\tnote\ttag_greedy\ttag_lookahead\ttag_sink\ttag_helper\ttag_cycle\ttag_linear\ttag_other\ttag_back\tuniverse\twidth\texact_budget\trestarts_used\tdeadline_hit\n",
    );
    for (r, &off) in results.iter().zip(offsets) {
        let (lo, hi, og) = match r.orig {
            Some((lo, hi)) => (lo as i64, hi as i64, (hi - lo) as i64),
            None => (-1, -1, -1),
        };
        let _ = write!(
            s,
            "{}\t{lo}\t{hi}\t{og}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{off}\t{}\t{}\t{}\t{}",
            r.idx,
            r.gates.len(),
            r.stats.changed_wires,
            r.stats.input_vars,
            r.stats.monomials,
            r.stats.max_degree,
            r.status.name(),
            r.secs,
            r.synth_status,
            r.pre_postopt,
            r.steps,
            r.note.replace(['\t', '\n'], " ")
        );
        for t in r.tags {
            let _ = write!(s, "\t{t}");
        }
        let _ = writeln!(
            s,
            "\t{}\t{}\t{}\t{}\t{}",
            r.universe, r.width, r.exact_budget, r.restarts_used, r.deadline_hit as u8
        );
    }
    s
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/unpacking/run.rs"]
mod tests;
