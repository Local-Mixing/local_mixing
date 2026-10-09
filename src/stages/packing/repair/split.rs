//! `anf_split2.py`: exposure-aware ANF split. Each window whose ANF does not
//! fit the budget is split at a candidate cut where NO interior (non-output)
//! C value is exactly affine, the candidate nearest the middle, recursively
//! until every piece fits or has at most 2 gates. When every candidate leaks,
//! the split falls back to all candidates and reports it (`WARNING`).
use crate::circuit::xgate::XGate;
use crate::stages::packing::anf::{MulStats, window_anf};
use crate::stages::packing::mono::{MonoWidthFn, with_mono_width};
use crate::stages::packing::types::CapSemantics;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

/// Number of candidate cuts tried per split (`k` in the oracle).
pub const SPLIT_CANDIDATES: usize = 32;

/// `sorted({lo + (hi - lo) * i // 33 for i in 1..=32} - {lo, hi})`.
pub fn split_candidates(lo: usize, hi: usize) -> Vec<usize> {
    let k = SPLIT_CANDIDATES;
    let mut v: Vec<usize> = (1..=k)
        .map(|i| lo + (hi - lo) * i / (k + 1))
        .filter(|&c| c != lo && c != hi)
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// The candidate nearest `(lo + hi) // 2`, the smaller one on a tie
/// (Python `min` keeps the first minimum of the sorted list).
pub fn pick_candidate(good: &[usize], lo: usize, hi: usize) -> usize {
    let mid = (lo + hi) / 2;
    *good
        .iter()
        .min_by_key(|&&x| x.abs_diff(mid))
        .expect("at least one candidate")
}

struct FitsBatch<'a> {
    gates: &'a [XGate],
    windows: &'a [(usize, usize)],
    budget: usize,
    sem: CapSemantics,
}

impl MonoWidthFn<Vec<(bool, MulStats)>> for FitsBatch<'_> {
    fn call<const W: usize>(self) -> Vec<(bool, MulStats)> {
        self.windows
            .par_iter()
            .map(|&(lo, hi)| {
                let mut st = MulStats::default();
                let ok =
                    window_anf::<W>(&self.gates[lo..hi], self.budget, self.sem, &mut st).is_ok();
                (ok, st)
            })
            .collect()
    }
}

/// The split predicate (`window_anf(gates, lo, hi, budget)[0] is not None`),
/// memoized per window for one circuit.
pub struct FitCache<'a> {
    gates: &'a [XGate],
    nw: usize,
    budget: usize,
    sem: CapSemantics,
    memo: FxHashMap<(usize, usize), bool>,
    /// Product counters of every evaluation (ambiguous products, aborts,
    /// order-dependent aborts).
    pub stats: MulStats,
    /// Windows whose decision depended on the CPython iteration order (an
    /// order-dependent product abort happened while evaluating it).
    pub order_dependent_windows: usize,
}

impl<'a> FitCache<'a> {
    /// `nw` is the circuit's computed width (at most 1024 wires).
    pub fn new(gates: &'a [XGate], nw: usize, budget: usize, sem: CapSemantics) -> Self {
        FitCache {
            gates,
            nw: nw.max(1),
            budget,
            sem,
            memo: FxHashMap::default(),
            stats: MulStats::default(),
            order_dependent_windows: 0,
        }
    }

    /// Evaluate every window not yet memoized, in parallel.
    pub fn prefetch(&mut self, windows: &[(usize, usize)]) {
        let mut todo: Vec<(usize, usize)> = windows
            .iter()
            .copied()
            .filter(|w| !self.memo.contains_key(w))
            .collect();
        todo.sort_unstable();
        todo.dedup();
        if todo.is_empty() {
            return;
        }
        let res = with_mono_width(
            self.nw,
            FitsBatch {
                gates: self.gates,
                windows: &todo,
                budget: self.budget,
                sem: self.sem,
            },
        );
        for (w, (ok, st)) in todo.into_iter().zip(res) {
            self.stats.add(&st);
            self.order_dependent_windows += (st.aborts_order_dependent > 0) as usize;
            self.memo.insert(w, ok);
        }
    }

    pub fn fits(&mut self, lo: usize, hi: usize) -> bool {
        if !self.memo.contains_key(&(lo, hi)) {
            self.prefetch(&[(lo, hi)]);
        }
        self.memo[&(lo, hi)]
    }
}

/// What one split pass did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SplitReport {
    /// New cuts in the order the recursion added them.
    pub new: Vec<usize>,
    /// Windows `[lo, hi)` split without an exposure-free candidate.
    pub forced: Vec<(usize, usize)>,
    /// Number of input bounds (`len(bounds)`).
    pub bounds_in: usize,
    /// `sorted(set(bounds) | set(new))`.
    pub bounds_out: Vec<usize>,
}

/// One `anf_split2.py` pass over `bounds` (any order; the pass sorts them
/// without deduplicating, as the oracle does). `leaky(cands)` returns the
/// candidates at which some non-output C value is exactly affine.
pub fn split_pass(
    m: usize,
    bounds: &[usize],
    fits: &mut FitCache<'_>,
    leaky: &mut dyn FnMut(&[usize]) -> FxHashSet<usize>,
) -> SplitReport {
    let mut sorted = bounds.to_vec();
    sorted.sort_unstable();
    let mut cuts = vec![0usize];
    cuts.extend(sorted.iter().copied().filter(|&b| 0 < b && b < m));
    cuts.push(m);
    let top: Vec<(usize, usize)> = cuts
        .windows(2)
        .map(|w| (w[0], w[1]))
        .filter(|&(lo, hi)| hi - lo > 2)
        .collect();
    fits.prefetch(&top);
    let mut rep = SplitReport {
        bounds_in: bounds.len(),
        ..Default::default()
    };
    fn fit(
        lo: usize,
        hi: usize,
        fits: &mut FitCache<'_>,
        leaky: &mut dyn FnMut(&[usize]) -> FxHashSet<usize>,
        rep: &mut SplitReport,
    ) {
        // The oracle evaluates the ANF first; a window of at most 2 gates
        // returns either way, so it is not evaluated here.
        if hi - lo <= 2 || fits.fits(lo, hi) {
            return;
        }
        let cands = split_candidates(lo, hi);
        let bad = leaky(&cands);
        let mut good: Vec<usize> = cands.iter().copied().filter(|c| !bad.contains(c)).collect();
        if good.is_empty() {
            good = cands;
            rep.forced.push((lo, hi));
        }
        let c = pick_candidate(&good, lo, hi);
        rep.new.push(c);
        fit(lo, c, fits, leaky, rep);
        fit(c, hi, fits, leaky, rep);
    }
    for w in cuts.windows(2) {
        fit(w[0], w[1], fits, leaky, &mut rep);
    }
    let mut out: Vec<usize> = bounds.iter().chain(rep.new.iter()).copied().collect();
    out.sort_unstable();
    out.dedup();
    rep.bounds_out = out;
    rep
}

#[cfg(test)]
#[path = "../../../../tests/unit/stages/packing/repair/split.rs"]
mod tests;
