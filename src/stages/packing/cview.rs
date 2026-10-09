//! The C-view exposure engine (stage 4), moved out of the `hmap_affine`
//! security tool so the packing stage and the tool share one implementation.
//!
//! For every value born in the source circuit C (the target of each gate,
//! restricted to wires `< n`), the exposure scan finds the minimum support t*
//! of an EXACT affine predictor of that value from the wire values of the
//! delivered circuit P at any of the given cuts: t* is the number of distinct
//! nonconstant P wire values read (complement pairs collapse, constants are
//! free), exact over all honest samples. Both circuits run on the same random
//! inputs on wires `0..n` with zeros elsewhere.
//!
//! Per cut (all of this matches the historic tool exactly):
//! * wire columns are complement-normalized (a column whose first sample bit
//!   is 1 is complemented), constant columns dropped, and duplicates collapse
//!   onto the LOWEST wire of each class;
//! * a GF(2) basis with coefficient tracking takes the representatives in
//!   wire order and then the constant column; dependent columns become
//!   nullspace vectors in arrival order;
//! * each row's particular solution is minimized over the nullspace coset
//!   with a Gray-code walk that keeps the first strict minimum; nullity above
//!   [`NULLITY_CAP`] keeps the particular solution and is counted as
//!   approximate.
//!
//! A `(row, cut)` fit depends only on `(C, P, cut, n, seed, batches)`, so
//! [`CutResult`]s are memoized per cut ([`CutOracle`]) and computed in
//! parallel. Everything that spans several cuts (strict-< best weight,
//! earliest-cut witness, `skip_settled` and its approximate-count accounting)
//! is replayed sequentially in cut order by [`assemble`].
//!
//! P is simulated in streaming fashion with one wire-major `[wire][batch]`
//! state, plus checkpoints every [`CHECKPOINT_EVERY`] gates for re-simulating
//! candidate cuts; the historic tool materialized `[batch][cut][wire]`.
//!
//! No environment reads; no feature gate (stage 4 builds with a plain
//! `cargo build --release`).

use super::gf2::{TrackedBasis, bit};
use crate::circuit::xgate::{XGate, max_wire};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::io::{self, Write};
use std::sync::Arc;

/// Default honest-sample seed (`hmap_affine --seed`).
pub const DEFAULT_SEED: u64 = 12345;
/// Default number of 64-sample batches (`hmap_affine --batches`).
pub const DEFAULT_BATCHES: usize = 96;
/// Largest nullity whose coset is minimized exactly.
pub const NULLITY_CAP: usize = 14;
/// Checkpoint interval (gates) of the streaming simulation of P.
pub const CHECKPOINT_EVERY: usize = 16384;

/// Sampling parameters of the C-view.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CviewParams {
    pub batches: usize,
    pub seed: u64,
}

impl Default for CviewParams {
    fn default() -> Self {
        Self {
            batches: DEFAULT_BATCHES,
            seed: DEFAULT_SEED,
        }
    }
}

/// The shared random inputs: `batches` x 64 samples of `n` input wires.
#[derive(Clone, Debug)]
pub struct HonestSamples {
    pub n: usize,
    pub batches: usize,
    /// `[batch][wire < n]`, bit l = sample l of the batch.
    x: Vec<Vec<u64>>,
}

impl HonestSamples {
    /// `StdRng::seed_from_u64(seed)`; per batch, `n` draws of `random::<u64>()`
    /// in wire order (the historic tool's draw order).
    pub fn draw(n: usize, batches: usize, seed: u64) -> Self {
        assert!(batches >= 1, "at least one sample batch");
        let mut rng = StdRng::seed_from_u64(seed);
        let x = (0..batches)
            .map(|_| (0..n).map(|_| rng.random::<u64>()).collect())
            .collect();
        Self { n, batches, x }
    }

    /// Input words of one batch (`n` words).
    pub fn batch(&self, b: usize) -> &[u64] {
        &self.x[b]
    }

    /// Wire-major initial state over `nw` wires: `state[w * batches + b]`.
    fn wire_major_init(&self, nw: usize) -> Vec<u64> {
        assert!(nw >= self.n, "state narrower than the input width");
        let b = self.batches;
        let mut st = vec![0u64; nw * b];
        for (bi, xb) in self.x.iter().enumerate() {
            for (w, &v) in xb.iter().enumerate() {
                st[w * b + bi] = v;
            }
        }
        st
    }
}

/// Apply `g` to a wire-major state of `b` words per wire. Per batch this is
/// bit-for-bit `XGate::apply_lanes`.
#[inline]
fn apply_wire_major(g: &XGate, st: &mut [u64], b: usize, acc: &mut [u64]) {
    acc.fill(!0u64);
    for &(w, pos) in &g.ctrls {
        let m = (pos as u64).wrapping_sub(1);
        let src = &st[w as usize * b..(w as usize + 1) * b];
        for (a, &s) in acc.iter_mut().zip(src) {
            *a &= s ^ m;
        }
    }
    let c = 0u64.wrapping_sub(g.comp as u64);
    let t = g.target as usize;
    for (d, &a) in st[t * b..(t + 1) * b].iter_mut().zip(acc.iter()) {
        *d ^= a ^ c;
    }
}

/// Prefix positions `0, step, 2*step, ... (< end), end` (historic `indices`).
pub fn prefix_indices(end: usize, step: usize) -> Vec<usize> {
    assert!(step >= 1, "prefix step must be positive");
    let mut v: Vec<usize> = (0..end).step_by(step).collect();
    v.push(end);
    v
}

/// An unparsable line of a cut file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CutParseError {
    /// 1-based line number.
    pub line: usize,
    pub source: std::num::ParseIntError,
}

impl std::fmt::Display for CutParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cut file line {}: {}", self.line, self.source)
    }
}

impl std::error::Error for CutParseError {}

/// Parse a cut file (one gate index per line, `0` = before gate 0): blank
/// lines are ignored, every other line is trimmed and parsed, cuts beyond
/// `g_len` are dropped, and the result is sorted and deduplicated.
pub fn parse_cut_file(text: &str, g_len: usize) -> Result<Vec<usize>, CutParseError> {
    let mut cuts = Vec::new();
    for (i, l) in text.lines().enumerate() {
        if l.trim().is_empty() {
            continue;
        }
        let x: usize = l.trim().parse().map_err(|e| CutParseError {
            line: i + 1,
            source: e,
        })?;
        if x <= g_len {
            cuts.push(x);
        }
    }
    cuts.sort_unstable();
    cuts.dedup();
    Ok(cuts)
}

/// One born value of C: the target wire of gate `gate`, observed at prefix
/// `prefix` (= `gate + 1` with a unit step).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BornRow {
    pub prefix: usize,
    pub gate: usize,
    pub wire: usize,
}

/// The born values of C over the honest samples.
#[derive(Clone, Debug)]
pub struct CRows {
    pub rows: Vec<BornRow>,
    batches: usize,
    /// `[row][batch]`.
    vecs: Vec<u64>,
}

impl CRows {
    pub fn len(&self) -> usize {
        self.rows.len()
    }
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
    pub fn batches(&self) -> usize {
        self.batches
    }
    /// Sample vector of row `r` (`batches` words).
    pub fn vec(&self, r: usize) -> &[u64] {
        &self.vecs[r * self.batches..(r + 1) * self.batches]
    }
}

/// The born rows of C at the given prefixes (non-decreasing, each at most
/// `c.len()`; duplicates repeat their row): for every prefix `i > 0` whose
/// gate `i - 1` targets a wire `< n`, the value of that wire after the prefix.
pub fn born_rows(c: &[XGate], n: usize, prefixes: &[usize], s: &HonestSamples) -> CRows {
    assert_eq!(s.n, n, "samples drawn for a different input width");
    assert!(
        prefixes.windows(2).all(|w| w[0] <= w[1]),
        "prefixes must be non-decreasing"
    );
    let nw = (max_wire(c) as usize + 1).max(n);
    let b = s.batches;
    let mut st = s.wire_major_init(nw);
    let mut acc = vec![0u64; b];
    let mut rows = Vec::new();
    let mut vecs = Vec::new();
    let mut pos = 0usize;
    for &i in prefixes {
        assert!(i <= c.len(), "prefix {i} beyond C ({} gates)", c.len());
        while pos < i {
            apply_wire_major(&c[pos], &mut st, b, &mut acc);
            pos += 1;
        }
        if i == 0 {
            continue;
        }
        let t = c[i - 1].target as usize;
        if t < n {
            rows.push(BornRow {
                prefix: i,
                gate: i - 1,
                wire: t,
            });
            vecs.extend_from_slice(&st[t * b..(t + 1) * b]);
        }
    }
    CRows {
        rows,
        batches: b,
        vecs,
    }
}

/// C's output gates: the last writer of every wire `< n`.
pub fn output_gates(c: &[XGate], n: usize) -> FxHashSet<usize> {
    let mut last: FxHashMap<usize, usize> = FxHashMap::default();
    for (gi, g) in c.iter().enumerate() {
        let t = g.target as usize;
        if t < n {
            last.insert(t, gi);
        }
    }
    last.into_values().collect()
}

/// A minimal exact affine predictor of one row at one cut.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CutFit {
    /// t*: number of nonconstant P wire values read.
    pub weight: u32,
    /// The P wires read (representatives, ascending).
    pub support: Vec<u32>,
}

/// The wire columns of P at one cut, ready to predict rows.
pub struct CutColumns {
    /// Representative wire of each complement class, ascending.
    rep_wires: Vec<usize>,
    /// Basis over the representatives (columns `0..m`) and the constant
    /// (column `m`).
    basis: TrackedBasis,
}

impl CutColumns {
    /// Build from a wire-major state `cols[w * batches + b]` over `nw` wires.
    pub fn build(cols: &[u64], nw: usize, batches: usize) -> Self {
        assert_eq!(cols.len(), nw * batches, "state size");
        let b = batches;
        // Complement-normalized columns; constants dropped; first wire wins.
        let mut norm = vec![0u64; nw * b];
        let mut keep: Vec<usize> = Vec::with_capacity(nw);
        for k in 0..nw {
            let src = &cols[k * b..(k + 1) * b];
            let flip = if src[0] & 1 == 1 { !0u64 } else { 0 };
            let dst = &mut norm[k * b..(k + 1) * b];
            let mut any = 0u64;
            for (d, &s) in dst.iter_mut().zip(src) {
                *d = s ^ flip;
                any |= *d;
            }
            if any != 0 {
                keep.push(k);
            }
        }
        let mut seen: FxHashMap<&[u64], ()> = FxHashMap::default();
        let mut rep_wires = Vec::new();
        for &k in &keep {
            let key = &norm[k * b..(k + 1) * b];
            if seen.insert(key, ()).is_none() {
                rep_wires.push(k);
            }
        }
        let m = rep_wires.len();
        let mut basis = TrackedBasis::new(b, m + 1);
        for (idx, &k) in rep_wires.iter().enumerate() {
            basis.insert(&norm[k * b..(k + 1) * b], idx);
        }
        basis.insert(&vec![!0u64; b], m);
        Self { rep_wires, basis }
    }

    /// Number of distinct nonconstant column classes.
    pub fn reps(&self) -> usize {
        self.rep_wires.len()
    }
    pub fn nullity(&self) -> usize {
        self.basis.nullity()
    }
    /// True when the coset minimization is skipped (counted as approximate).
    pub fn over_cap(&self) -> bool {
        self.basis.nullity() > NULLITY_CAP
    }

    fn wirewt(&self, coef: &[u64]) -> u32 {
        let m = self.rep_wires.len();
        let mut wt = 0u32;
        for (wi, &w) in coef.iter().enumerate() {
            let mut ww = w;
            if wi == m / 64 {
                ww &= !(1u64 << (m % 64));
            }
            wt += ww.count_ones();
        }
        wt
    }

    /// The minimal exact predictor of `row` (one word per batch), or `None`
    /// when the row is not exactly affine in this cut's wires.
    pub fn predict(&self, row: &[u64]) -> Option<CutFit> {
        let mut scratch = Vec::new();
        self.predict_with(row, &mut scratch)
    }

    fn predict_with(&self, row: &[u64], scratch: &mut Vec<u32>) -> Option<CutFit> {
        let cw = self.basis.coef_words();
        let mut coef = vec![0u64; cw];
        if !self.basis.solve(row, &mut coef, scratch) {
            return None;
        }
        let mut w = self.wirewt(&coef);
        let mut argmin = coef.clone();
        let k = self.basis.nullity();
        if k > 0 && k <= NULLITY_CAP {
            let mut cur = coef;
            let mut prev = 0usize;
            for g2 in 1usize..(1usize << k) {
                let gray = g2 ^ (g2 >> 1);
                let flip = (gray ^ prev).trailing_zeros() as usize;
                prev = gray;
                for (a, x) in cur.iter_mut().zip(self.basis.null(flip)) {
                    *a ^= x;
                }
                let wt = self.wirewt(&cur);
                if wt < w {
                    w = wt;
                    argmin.copy_from_slice(&cur);
                }
            }
        }
        let support = (0..self.rep_wires.len())
            .filter(|&idx| bit(&argmin, idx))
            .map(|idx| self.rep_wires[idx] as u32)
            .collect();
        Some(CutFit { weight: w, support })
    }
}

/// Every row's minimal fit at one cut.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CutResult {
    /// The cut position (gates of P before it).
    pub cut: usize,
    /// Nullity above [`NULLITY_CAP`]: fits at this cut are upper bounds.
    pub over_cap: bool,
    /// `(row, fit)` for each exactly affine row, rows ascending.
    pub fits: Vec<(u32, CutFit)>,
}

fn cut_result(cut: usize, cols: &[u64], nw: usize, rows: &CRows) -> CutResult {
    let cc = CutColumns::build(cols, nw, rows.batches);
    let mut scratch = Vec::new();
    let fits = (0..rows.len())
        .filter_map(|r| {
            cc.predict_with(rows.vec(r), &mut scratch)
                .map(|f| (r as u32, f))
        })
        .collect();
    CutResult {
        cut,
        over_cap: cc.over_cap(),
        fits,
    }
}

/// Per-cut results for one P, memoized, with a streaming simulation of P.
pub struct CutOracle<'a> {
    p: &'a [XGate],
    nw: usize,
    rows: &'a CRows,
    samples: &'a HonestSamples,
    checkpoint_every: usize,
    /// State at gate `k * checkpoint_every`, for the k created so far.
    checkpoints: Vec<Vec<u64>>,
    /// Current simulation position and state.
    pos: usize,
    state: Vec<u64>,
    memo: FxHashMap<usize, Arc<CutResult>>,
    parallel: bool,
    chunk: usize,
}

impl<'a> CutOracle<'a> {
    /// `nw` is P's wire count (at least `samples.n` and above every wire of P).
    pub fn new(p: &'a [XGate], nw: usize, rows: &'a CRows, samples: &'a HonestSamples) -> Self {
        assert!(nw >= samples.n, "P has fewer wires than n");
        assert!(
            p.is_empty() || (max_wire(p) as usize) < nw,
            "P uses a wire beyond nw = {nw}"
        );
        assert_eq!(rows.batches, samples.batches, "rows from other samples");
        let state = samples.wire_major_init(nw);
        Self {
            p,
            nw,
            rows,
            samples,
            checkpoint_every: CHECKPOINT_EVERY,
            checkpoints: vec![state.clone()],
            pos: 0,
            state,
            memo: FxHashMap::default(),
            parallel: true,
            chunk: (2 * rayon::current_num_threads()).max(8),
        }
    }

    /// Change the checkpoint interval (before the first query).
    pub fn with_checkpoint_every(mut self, every: usize) -> Self {
        assert!(every >= 1);
        assert!(
            self.pos == 0 && self.checkpoints.len() == 1,
            "set before use"
        );
        self.checkpoint_every = every;
        self
    }

    /// Compute cuts serially (for tests and reproducibility checks; the
    /// results are identical either way).
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Number of memoized cuts.
    pub fn memoized(&self) -> usize {
        self.memo.len()
    }

    fn rewind_to(&mut self, cut: usize) {
        let k = (cut / self.checkpoint_every).min(self.checkpoints.len() - 1);
        self.state.copy_from_slice(&self.checkpoints[k]);
        self.pos = k * self.checkpoint_every;
    }

    fn advance_to(&mut self, target: usize, acc: &mut [u64]) {
        let b = self.samples.batches;
        while self.pos < target {
            apply_wire_major(&self.p[self.pos], &mut self.state, b, acc);
            self.pos += 1;
            if self.pos % self.checkpoint_every == 0
                && self.checkpoints.len() == self.pos / self.checkpoint_every
            {
                self.checkpoints.push(self.state.clone());
            }
        }
    }

    fn flush(&mut self, pending: &mut Vec<(usize, Vec<u64>)>) {
        let (nw, rows) = (self.nw, self.rows);
        let done: Vec<CutResult> = if self.parallel {
            pending
                .par_iter()
                .map(|(cut, st)| cut_result(*cut, st, nw, rows))
                .collect()
        } else {
            pending
                .iter()
                .map(|(cut, st)| cut_result(*cut, st, nw, rows))
                .collect()
        };
        for r in done {
            self.memo.insert(r.cut, Arc::new(r));
        }
        pending.clear();
    }

    /// The results at `cuts` (each at most `p.len()`), in the given order;
    /// cuts not yet memoized are computed (in parallel unless disabled).
    pub fn results(&mut self, cuts: &[usize]) -> Vec<Arc<CutResult>> {
        let mut missing: Vec<usize> = cuts
            .iter()
            .copied()
            .filter(|c| !self.memo.contains_key(c))
            .collect();
        missing.sort_unstable();
        missing.dedup();
        if let Some(&last) = missing.last() {
            assert!(
                last <= self.p.len(),
                "cut {last} beyond P ({} gates)",
                self.p.len()
            );
        }
        if let Some(&first) = missing.first()
            && first < self.pos
        {
            self.rewind_to(first);
        }
        let mut acc = vec![0u64; self.samples.batches];
        let mut pending: Vec<(usize, Vec<u64>)> = Vec::new();
        for cut in missing {
            self.advance_to(cut, &mut acc);
            pending.push((cut, self.state.clone()));
            if pending.len() >= self.chunk {
                self.flush(&mut pending);
            }
        }
        self.flush(&mut pending);
        cuts.iter().map(|c| Arc::clone(&self.memo[c])).collect()
    }
}

/// What [`assemble`] collects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExposureOptions {
    /// Record every (row, cut, weight) where the row is exactly affine.
    pub collect_allcuts: bool,
    /// Record the earliest minimizing cut and its support per visible row.
    pub collect_witness: bool,
    /// Skip rows already settled at weight 0 or 1 (the historic tool does
    /// this exactly when it does not collect all cuts; it changes the
    /// approximate-minimization count).
    pub skip_settled: bool,
}

impl ExposureOptions {
    /// The historic tool's options for the given output requests.
    pub fn historic(collect_allcuts: bool, collect_witness: bool) -> Self {
        Self {
            collect_allcuts,
            collect_witness,
            skip_settled: !collect_allcuts,
        }
    }
}

/// The exposure scan's result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExposureReport {
    pub rows: Vec<BornRow>,
    /// t* per row; -1 = no exact affine predictor at any scanned cut.
    pub best: Vec<i64>,
    /// Per row: (cut position, support) of the earliest minimizer, when
    /// collected and visible.
    pub witness: Vec<Option<(usize, Vec<u32>)>>,
    /// (row, cut position, weight), in cut order then row order, when collected.
    pub allcuts: Vec<(u32, usize, u32)>,
    /// Coset minimizations skipped (nullity above the cap).
    pub approx: usize,
    pub options: ExposureOptions,
}

/// Replay per-cut results in cut order (`cuts[i]` is `per_cut[i]`'s cut).
pub fn assemble(
    rows: &CRows,
    cuts: &[usize],
    per_cut: &[Arc<CutResult>],
    o: &ExposureOptions,
) -> ExposureReport {
    assert_eq!(cuts.len(), per_cut.len(), "one result per cut");
    let nr = rows.len();
    let mut best = vec![-1i64; nr];
    let mut witness: Vec<Option<(usize, Vec<u32>)>> = vec![None; nr];
    let mut allcuts = Vec::new();
    let mut approx = 0usize;
    for (&cut, res) in cuts.iter().zip(per_cut) {
        assert_eq!(res.cut, cut, "result for another cut");
        for (r, fit) in &res.fits {
            let r = *r as usize;
            if o.skip_settled && best[r] >= 0 && best[r] <= 1 {
                continue;
            }
            if res.over_cap {
                approx += 1;
            }
            if o.collect_allcuts {
                allcuts.push((r as u32, cut, fit.weight));
            }
            if best[r] < 0 || (fit.weight as i64) < best[r] {
                best[r] = fit.weight as i64;
                if o.collect_witness {
                    witness[r] = Some((cut, fit.support.clone()));
                }
            }
        }
    }
    ExposureReport {
        rows: rows.rows.clone(),
        best,
        witness,
        allcuts,
        approx,
        options: *o,
    }
}

/// The full scan: per-cut results for P at `cuts` (in parallel), assembled.
pub fn exposure_scan(
    c_rows: &CRows,
    p: &[XGate],
    nw_p: usize,
    s: &HonestSamples,
    cuts: &[usize],
    o: &ExposureOptions,
) -> ExposureReport {
    let mut oracle = CutOracle::new(p, nw_p, c_rows, s);
    let per_cut = oracle.results(cuts);
    assemble(c_rows, cuts, &per_cut, o)
}

impl ExposureReport {
    /// Rows with an exact affine predictor at some cut.
    pub fn visible(&self) -> usize {
        self.best.iter().filter(|&&t| t >= 0).count()
    }

    /// The `[exposure]` summary and t* histogram lines, exactly as the
    /// historic tool prints them.
    pub fn summary_lines(&self) -> Vec<String> {
        let mut hist: std::collections::BTreeMap<i64, usize> = Default::default();
        for &t in &self.best {
            *hist.entry(t).or_default() += 1;
        }
        let total = self.best.len().max(1);
        let visible = self.visible();
        let mut out = vec![format!(
            "[exposure] {} born values, {} visible ({:.1}%), {} coset-minimizations skipped (nullity>14)",
            total,
            visible,
            100.0 * visible as f64 / total as f64,
            self.approx
        )];
        let mut cum = 0usize;
        for (&t, &cnt) in hist.iter().filter(|(t, _)| **t >= 0) {
            cum += cnt;
            out.push(format!(
                "[exposure] t*={:>3}: {:>5} values ({:>5.1}%)  cumulative {:>5.1}%",
                t,
                cnt,
                100.0 * cnt as f64 / total as f64,
                100.0 * cum as f64 / total as f64
            ));
        }
        out
    }

    /// `<out>.exposure.tsv`: gate, wire, t* per row.
    pub fn write_exposure_tsv(&self, w: &mut impl Write) -> io::Result<()> {
        writeln!(
            w,
            "# gate\twire\ttstar (-1 = no exact affine predictor at any sampled cut)"
        )?;
        for (r, &t) in self.rows.iter().zip(&self.best) {
            writeln!(w, "{}\t{}\t{t}", r.gate, r.wire)?;
        }
        Ok(())
    }

    /// The all-cuts TSV (requires `collect_allcuts`).
    pub fn write_allcuts_tsv(&self, w: &mut impl Write) -> io::Result<()> {
        if !self.options.collect_allcuts {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "all-cuts rows were not collected",
            ));
        }
        writeln!(
            w,
            "# gate\twire\tcut\tweight (every cut where the value is exactly affine)"
        )?;
        for &(r, cut, wt) in &self.allcuts {
            let row = self.rows[r as usize];
            writeln!(w, "{}\t{}\t{cut}\t{wt}", row.gate, row.wire)?;
        }
        Ok(())
    }

    /// The witness TSV (requires `collect_witness`).
    pub fn write_witness_tsv(&self, w: &mut impl Write) -> io::Result<()> {
        if !self.options.collect_witness {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "witnesses were not collected",
            ));
        }
        writeln!(w, "# gate\twire\ttstar\tcut\tG-wires (predictor support)")?;
        for ((r, &t), wit) in self.rows.iter().zip(&self.best).zip(&self.witness) {
            if t >= 0 {
                let (cut, wires) = wit.as_ref().expect("visible row has a witness");
                let list = wires
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                writeln!(w, "{}\t{}\t{t}\t{cut}\t{list}", r.gate, r.wire)?;
            }
        }
        Ok(())
    }

    /// Visible rows whose gate is not an output gate of C (the identity
    /// floor test: a nonempty result means the floor is missed).
    pub fn nonoutput_visible(&self, outs: &FxHashSet<usize>) -> Vec<usize> {
        (0..self.rows.len())
            .filter(|&r| self.best[r] >= 0 && !outs.contains(&self.rows[r].gate))
            .collect()
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/packing/cview.rs"]
mod tests;
