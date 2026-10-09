//! The stage-4 repair, in process (docs/STAGE4_RUST_DESIGN.md §4.11): the
//! control flow of `waist/repair.sh` over ports of `order_repair.py` (cancel,
//! fix), `merge_bounds2.py` (merge) and `anf_split2.py` (split), with the
//! C-view exposure scan of `hmap_affine` (`cview`) run on in-memory circuits.
//!
//! `RepairMode::Oracle` is `repair.sh` exactly:
//!
//! 1. cancel to a fixed point on the packed circuit (and its bounds);
//! 2. up to `fix_iterations` times: scan; stop when `visible <= n`; else fix.
//!    Every iteration restarts from `to_seq(gates, bounds)` of the previous
//!    result, as the subprocess chain reloads its files; there is no rescan
//!    after the last fix;
//! 3. up to `fallback_iterations` times: scan; if `visible > n`, merge with
//!    this scan's all-cuts rows; split (budget `split_budget`, candidates
//!    checked on the same circuit); stop when `visible <= n` and the split
//!    added nothing;
//! 4. a final, report-only scan of the final lattice and the split's budget
//!    test on every final window (they decide exits 5 and 6 in the binary).
//!
//! Defined where the oracle fails (D-b): a lattice without an interior cut
//! (the oracle's `hmap_affine` panics on an "empty plate") is scanned as zero
//! cuts, i.e. `visible = 0`, and the loop goes on as for any other count.
//!
//! Every gate keeps its ORIGIN (index in the packed circuit) through every
//! sequence operation, so the result is certified exactly
//! (`certify::certify_repair`), in addition to the oracle's sampled
//! `verify_equiv` checks. No environment reads.
pub mod cancel;
pub mod fix;
pub mod merge;
pub mod seq;
pub mod split;

use self::cancel::run_cancel;
use self::fix::{FixRow, run_fix};
use self::merge::merge_bounds;
use self::seq::{from_seq, to_seq};
use self::split::{FitCache, split_pass};
use super::anf::MulStats;
use super::certify::certify_repair;
use super::cview::{
    CRows, CutOracle, CviewParams, ExposureOptions, ExposureReport, HonestSamples, assemble,
    born_rows, prefix_indices,
};
use super::mono::mono_words;
use super::types::{RepairParams, Stage4Error};
use super::verify::verify_equiv;
use crate::circuit::xgate::{XGate, max_wire};
use rustc_hash::{FxHashMap, FxHashSet};
use std::io::Write;

/// `verify_equiv.py`'s seed.
pub const VERIFY_SEED: u64 = 77;
/// `hmap_affine --train-batches` default (printed in the scan log only).
const HA_TRAIN_BATCHES: usize = 72;

/// The C side of every scan: honest samples, the born rows of C and its
/// output gates. Built once per repair.
pub struct CView {
    pub n: usize,
    pub batches: usize,
    pub samples: HonestSamples,
    pub rows: CRows,
    /// Output gates as the oracle derives them from `exposure.tsv`: the
    /// largest C gate per wire among the born rows.
    pub outs: FxHashSet<usize>,
    pub c_len: usize,
    nw_c: usize,
}

/// One C-view scan of a (circuit, lattice) and what its log line 1 needs.
#[derive(Clone, Debug)]
pub struct Scan {
    pub report: ExposureReport,
    /// Number of distinct cuts scanned (0 = a single window, scan skipped).
    pub cuts: usize,
    pub g_len: usize,
    pub nw_g: usize,
}

impl Scan {
    pub fn visible(&self) -> usize {
        self.report.visible()
    }
}

impl CView {
    /// `hmap_affine --c C --n n --exposure --c-step 1` (batches at least 2).
    pub fn new(c: &[XGate], n: usize, p: &CviewParams) -> Self {
        let batches = p.batches.max(2);
        let samples = HonestSamples::draw(n, batches, p.seed);
        let rows = born_rows(c, n, &prefix_indices(c.len(), 1), &samples);
        let mut last: FxHashMap<usize, usize> = FxHashMap::default();
        for r in &rows.rows {
            let e = last.entry(r.wire).or_insert(r.gate);
            *e = (*e).max(r.gate);
        }
        let nw_c = if c.is_empty() {
            n
        } else {
            (max_wire(c) as usize + 1).max(n)
        };
        CView {
            n,
            batches,
            samples,
            rows,
            outs: last.into_values().collect(),
            c_len: c.len(),
            nw_c,
        }
    }

    /// `hmap_affine`'s width of P: `max(maxwire + 1, n)`.
    pub fn nw_g(&self, p: &[XGate]) -> usize {
        if p.is_empty() {
            self.n
        } else {
            (max_wire(p) as usize + 1).max(self.n)
        }
    }

    pub fn oracle<'a>(&'a self, p: &'a [XGate]) -> CutOracle<'a> {
        CutOracle::new(p, self.nw_g(p), &self.rows, &self.samples)
    }

    /// Scan `bounds` the way `hmap_affine --g-cuts` reads a bounds file:
    /// sorted, deduplicated, cuts beyond the circuit dropped; all-cuts and
    /// witness rows collected (as `repair.sh` asks).
    pub fn scan(&self, oracle: &mut CutOracle<'_>, p: &[XGate], bounds: &[usize]) -> Scan {
        let mut cuts: Vec<usize> = bounds.iter().copied().filter(|&b| b <= p.len()).collect();
        cuts.sort_unstable();
        cuts.dedup();
        let per = if cuts.is_empty() {
            Vec::new()
        } else {
            oracle.results(&cuts)
        };
        Scan {
            report: assemble(
                &self.rows,
                &cuts,
                &per,
                &ExposureOptions::historic(true, true),
            ),
            cuts: cuts.len(),
            g_len: p.len(),
            nw_g: self.nw_g(p),
        }
    }

    /// `anf_split2.leaky_cuts`: the candidates at which some non-output C
    /// value is exactly affine.
    pub fn leaky(&self, oracle: &mut CutOracle<'_>, cands: &[usize]) -> FxHashSet<usize> {
        let per = oracle.results(cands);
        per.iter()
            .filter(|res| {
                res.fits
                    .iter()
                    .any(|(r, _)| !self.outs.contains(&self.rows.rows[*r as usize].gate))
            })
            .map(|res| res.cut)
            .collect()
    }

    /// The fix pass's rows: every visible row's witness, in row order.
    pub fn fix_rows(&self, scan: &Scan) -> Vec<FixRow> {
        let r = &scan.report;
        (0..r.rows.len())
            .filter(|&i| r.best[i] >= 0)
            .map(|i| {
                let (cut, sup) = r.witness[i].as_ref().expect("visible row has a witness");
                FixRow {
                    c_gate: r.rows[i].gate,
                    cut: *cut,
                    support: sup.clone(),
                }
            })
            .collect()
    }

    /// The scan's log (`rep<G>.cview.log`) exactly as `hmap_affine` prints it,
    /// with the given output paths.
    pub fn scan_log(&self, s: &Scan, exposure: &str, allcuts: &str, witness: &str) -> Vec<String> {
        let b = self.batches;
        let tb = HA_TRAIN_BATCHES.clamp(1, b - 1);
        let mut out = vec![format!(
            "[hmap_affine] c={} gates ({}w), g={} gates ({}w), n={}; degree=1 regressors={}; rows={} cols={}, samples={} (train {})",
            self.c_len,
            self.nw_c,
            s.g_len,
            s.nw_g,
            self.n,
            s.nw_g + 1,
            self.c_len + 1,
            s.cuts,
            b * 64,
            tb * 64
        )];
        out.extend(s.report.summary_lines());
        out.push(format!("[exposure] wrote {exposure}"));
        out.push(format!(
            "[exposure] wrote all-cuts {allcuts} ({} rows)",
            s.report.allcuts.len()
        ));
        out.push(format!("[exposure] wrote witness {witness}"));
        out
    }
}

/// Everything the repair did (`packing.report` keys `repair.*`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RepairReport {
    pub packed_gates: usize,
    /// Entries of the packed bounds file (duplicates and out-of-range kept).
    pub packed_bounds: usize,
    /// Pairs per productive cancel pass.
    pub cancel_passes: Vec<usize>,
    /// Visible count per fix iteration.
    pub fix_trajectory: Vec<usize>,
    /// Gates moved per fix pass.
    pub fix_moved: Vec<usize>,
    /// Every fix iteration ran without reaching the floor (no rescan).
    pub fix_exhausted: bool,
    /// Visible count per fallback iteration.
    pub fallback_trajectory: Vec<usize>,
    /// Boundaries removed per merge (only iterations above the floor merge).
    pub merged: Vec<usize>,
    /// Splits added per fallback iteration.
    pub splits: Vec<usize>,
    /// Windows split without an exposure-free candidate (`WARNING`).
    pub forced_splits: Vec<(usize, usize)>,
    /// Scans of a lattice without an interior cut (skipped, visible 0).
    pub single_window_scans: usize,
    pub final_gates: usize,
    pub final_bounds: usize,
    pub final_visible: usize,
    /// C gates of visible non-output rows in the final scan (exit 5).
    pub final_nonoutput_visible: Vec<usize>,
    /// Final windows failing the split's budget test (exit 6).
    pub final_over_budget: Vec<(usize, usize)>,
    pub split_stats: MulStats,
    pub split_order_dependent_windows: usize,
    pub scans: usize,
}

impl RepairReport {
    /// The exit decision of the final tests: `Floor` (exit 5) first, then
    /// `OverBudget` (exit 6); `allow_*` waives them.
    pub fn status(
        &self,
        allow_floor_miss: bool,
        allow_over_budget: bool,
    ) -> Result<&'static str, Stage4Error> {
        if !self.final_nonoutput_visible.is_empty() {
            if !allow_floor_miss {
                return Err(Stage4Error::Floor {
                    nonoutput_visible: self.final_nonoutput_visible.clone(),
                });
            }
            return Ok("floor_miss");
        }
        if !self.final_over_budget.is_empty() {
            if !allow_over_budget {
                return Err(Stage4Error::OverBudget {
                    windows: self.final_over_budget.clone(),
                });
            }
            return Ok("over_budget");
        }
        Ok("ok")
    }
}

pub struct Repaired {
    /// Header n of the packed circuit, carried through (`order_repair.save`).
    pub n_header: usize,
    /// Packed index of every repaired gate, in repaired order.
    pub origins: Vec<u32>,
    pub gates: Vec<XGate>,
    pub bounds: Vec<usize>,
    /// The last in-loop scan: what the oracle leaves in `rep<G>.*.tsv`.
    pub last_scan: Scan,
    /// The report-only scan of the final lattice.
    pub final_scan: Scan,
    /// Cancelled pairs per pass (packed indices).
    pub cancelled: Vec<Vec<(u32, u32)>>,
    pub report: RepairReport,
}

fn materialize(arena: &[XGate], origins: &[u32]) -> Vec<XGate> {
    origins.iter().map(|&o| arena[o as usize].clone()).collect()
}

fn logln(log: &mut dyn Write, line: &str) -> Result<(), Stage4Error> {
    writeln!(log, "{line}").map_err(|e| Stage4Error::io("<repair log>", e))
}

/// The final tests on (gates, bounds): a scan (report only) and the split's
/// budget test on every window `[0] + sorted(0 < b < m) + [m]`.
pub fn final_tests(
    cv: &CView,
    oracle: &mut CutOracle<'_>,
    fits: &mut FitCache<'_>,
    gates: &[XGate],
    bounds: &[usize],
) -> (Scan, Vec<usize>, Vec<(usize, usize)>) {
    let s = cv.scan(oracle, gates, bounds);
    let nonout: Vec<usize> = s
        .report
        .nonoutput_visible(&cv.outs)
        .into_iter()
        .map(|r| s.report.rows[r].gate)
        .collect();
    let m = gates.len();
    let mut cuts = vec![0usize];
    let mut inner: Vec<usize> = bounds.iter().copied().filter(|&b| 0 < b && b < m).collect();
    inner.sort_unstable();
    cuts.extend(inner);
    cuts.push(m);
    let windows: Vec<(usize, usize)> = cuts.windows(2).map(|w| (w[0], w[1])).collect();
    fits.prefetch(&windows);
    let over = windows
        .into_iter()
        .filter(|&(lo, hi)| !fits.fits(lo, hi))
        .collect();
    (s, nonout, over)
}

/// The state width every window ANF of `gates` uses (`anf_compose.load`).
pub fn anf_width(n_header: usize, gates: &[XGate]) -> Result<usize, Stage4Error> {
    let nw = if gates.is_empty() {
        n_header
    } else {
        n_header.max(max_wire(gates) as usize + 1)
    };
    if mono_words(nw.max(1)).is_none() {
        return Err(Stage4Error::Usage(format!(
            "{nw}-wire state: stage 4 supports at most 1024 wires"
        )));
    }
    Ok(nw)
}

/// `repair.sh` on a packed circuit (`packed`, header `packed_n`) and its
/// unrepaired bounds. `rep_label` is the repaired circuit's path as printed
/// in the log (`$P` in the oracle). The log is the oracle's stdout, with the
/// background `verify_equiv` line in a fixed place (right after the cancel).
pub fn repair(
    packed: &[XGate],
    packed_n: usize,
    bounds0: &[i64],
    cv: &CView,
    p: &RepairParams,
    rep_label: &str,
    log: &mut dyn Write,
) -> Result<Repaired, Stage4Error> {
    let n = cv.n;
    if packed.len() >= u32::MAX as usize {
        return Err(Stage4Error::Input("packed circuit too long".into()));
    }
    let nw = anf_width(packed_n, packed)?;
    if cv.nw_g(packed) < n || nw < n {
        return Err(Stage4Error::Input(format!(
            "the packed circuit has fewer than n = {n} wires"
        )));
    }
    let mut rep = RepairReport {
        packed_gates: packed.len(),
        packed_bounds: bounds0.len(),
        ..Default::default()
    };

    // 1. cancel (order_repair.py ... cancel)
    let all: Vec<u32> = (0..packed.len() as u32).collect();
    let mut seq = to_seq(&all, bounds0);
    drop(all);
    let cancelled = run_cancel(&mut seq, packed);
    let (mut origins, mut bounds) = from_seq(&seq);
    drop(seq);
    let total: usize = cancelled.iter().map(Vec::len).sum();
    for pass in &cancelled {
        logln(log, &format!("  cancel pass: {} pairs", pass.len()))?;
        rep.cancel_passes.push(pass.len());
    }
    logln(
        log,
        &format!("cancelled {total} pairs ({} gates)", 2 * total),
    )?;
    logln(
        log,
        &format!(
            "{} -> {} gates; {} -> {} boundaries",
            packed.len(),
            origins.len(),
            bounds0.len(),
            bounds.len()
        ),
    )?;
    let equiv = |origins: &[u32], log: &mut dyn Write| -> Result<(), Stage4Error> {
        let g = materialize(packed, origins);
        verify_equiv(packed, &g, packed_n, VERIFY_SEED)?;
        logln(
            log,
            &format!(
                "{rep_label}: {} vs {} gates; EQUIVALENT (1024 random states)",
                packed.len(),
                g.len()
            ),
        )
    };
    equiv(&origins, log)?;

    // 2. fix loop
    let mut last_scan: Option<Scan> = None;
    for it in 1..=p.fix_iterations {
        let gates = materialize(packed, &origins);
        let mut oracle = cv.oracle(&gates);
        let s = cv.scan(&mut oracle, &gates, &bounds);
        drop(oracle);
        rep.scans += 1;
        let v = s.visible();
        rep.fix_trajectory.push(v);
        logln(
            log,
            &format!("fix iter {it}: C-view visible {v} (floor {n})"),
        )?;
        if s.cuts == 0 {
            rep.single_window_scans += 1;
            logln(log, "  (single window: no interior cut, scan skipped)")?;
        }
        let rows = cv.fix_rows(&s);
        last_scan = Some(s);
        if v <= n {
            break;
        }
        let mut seq = to_seq(
            &origins,
            &bounds.iter().map(|&b| b as i64).collect::<Vec<_>>(),
        );
        let fr = run_fix(&mut seq, packed, &rows, &cv.outs);
        rep.fix_moved.push(fr.moved);
        let (o2, b2) = from_seq(&seq);
        logln(
            log,
            &format!(
                "{} -> {} gates; {} -> {} boundaries",
                origins.len(),
                o2.len(),
                bounds.len(),
                b2.len()
            ),
        )?;
        origins = o2;
        bounds = b2;
        if it == p.fix_iterations {
            rep.fix_exhausted = true;
        }
    }
    equiv(&origins, log)?;

    // 3. fallback loop: the circuit is fixed from here on, so every scan and
    // every split predicate share one per-cut memo and one window memo.
    let gates = materialize(packed, &origins);
    let m = gates.len();
    let mut oracle = cv.oracle(&gates);
    let mut fits = FitCache::new(&gates, nw, p.split_budget, p.cap_semantics);
    for it in 1..=p.fallback_iterations {
        let s = cv.scan(&mut oracle, &gates, &bounds);
        rep.scans += 1;
        let v = s.visible();
        rep.fallback_trajectory.push(v);
        logln(log, &format!("fallback iter {it}: C-view visible {v}"))?;
        if s.cuts == 0 {
            rep.single_window_scans += 1;
            logln(log, "  (single window: no interior cut, scan skipped)")?;
        }
        if v > n {
            let (kept, removed) = merge_bounds(&bounds, &s.report, &cv.outs);
            logln(
                log,
                &format!("merge: removed {removed} of {} boundaries", bounds.len()),
            )?;
            rep.merged.push(removed);
            bounds = kept;
        }
        last_scan = Some(s);
        let sp = split_pass(m, &bounds, &mut fits, &mut |c| cv.leaky(&mut oracle, c));
        // `echo "  $out"`: only the first line of the split's stdout gets the
        // two-space prefix.
        let mut lines: Vec<String> = sp
            .forced
            .iter()
            .map(|(lo, hi)| format!("  WARNING: no exposure-free split in [{lo},{hi})"))
            .collect();
        lines.push(format!(
            "{rep_label}: {} splits added ({} -> {} boundaries)",
            sp.new.len(),
            sp.bounds_in,
            sp.bounds_out.len()
        ));
        lines[0] = format!("  {}", lines[0]);
        for l in &lines {
            logln(log, l)?;
        }
        rep.splits.push(sp.new.len());
        rep.forced_splits.extend(sp.forced.iter().copied());
        let none = sp.new.is_empty();
        bounds = sp.bounds_out;
        if v <= n && none {
            break;
        }
    }

    // 4. final tests (report only; no output byte depends on them)
    let (final_scan, nonout, over) = final_tests(cv, &mut oracle, &mut fits, &gates, &bounds);
    drop(oracle);
    rep.final_gates = m;
    rep.final_bounds = bounds.len();
    rep.final_visible = final_scan.visible();
    rep.final_nonoutput_visible = nonout;
    rep.final_over_budget = over;
    rep.split_stats = fits.stats;
    rep.split_order_dependent_windows = fits.order_dependent_windows;
    drop(fits);

    // exact certificate: cancellations replay, then a linear extension
    certify_repair(packed, &cancelled, &origins)?;

    let last_scan = last_scan.unwrap_or_else(|| final_scan.clone());
    Ok(Repaired {
        n_header: packed_n,
        origins,
        gates,
        bounds,
        last_scan,
        final_scan,
        cancelled,
        report: rep,
    })
}

#[cfg(test)]
#[path = "../../../../tests/unit/stages/packing/repair/mod.rs"]
mod tests;
