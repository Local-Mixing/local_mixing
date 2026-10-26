//! The adaptive planner: large variable-length blocks grown by incremental
//! exact ANF composition, with boundaries chosen away from exposed cuts
//! (port of the server prototype's `adaptive_bounds.rs` and the `plan()` /
//! `Profile` of `stage6_run.py`).
//!
//! A block starts at `start` and grows gate by gate while the composed ANF
//! stays within the composition limits (`Limits`: total terms 8x the cap,
//! per-wire terms 4x the cap, a product-pair work limit, the gate cap). An
//! endpoint is ELIGIBLE when the total is at most `term_cap`; the scan
//! continues `lookahead` gates past the latest eligible endpoint (a prefix
//! can get more complicated and simplify again). Only the upper half of the
//! feasible interval `[start, end]` is considered for the boundary, chosen
//! by the key (deep affine proxies at the cut, all internal affine proxies,
//! near a previous round's boundary, -signed clearance from the affine
//! intervals, direct locations, latest). A gate that alone exceeds the caps
//! becomes a single-gate block (`single_gate_exception`).
//!
//! With `segments > 1` or `avoid_radius > 0`, the circuit is first cut at
//! seams: the cuts nearest to `i * m / segments` with affine coverage zero
//! and farther than `avoid_radius` from every previous bound (searched up
//! to 20000 gates away, else the target itself), and the segments are
//! planned in parallel (rayon; the caller sets the global pool). The bounds
//! are the seams plus every segment's bounds.
//!
//! A census with no intervals (`Census::empty`) plans C-blind: every cut
//! scores equally and the latest eligible cut wins.
//!
//! Library code: no environment reads, nothing printed.
use super::census::{invalid, parse_num};
use super::types::{BlockReport, Census, Interval, Plan, PlanParams, Source};
use crate::circuit::xgate::{XGate, max_wire};
use crate::stages::packing::mono::{Mono, MonoWidthFn, with_mono_width};
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

/// Farthest a seam is searched from its target.
pub const SEAM_SEARCH: usize = 20_000;

const BLOCKS_HEADER: &str = "start\tend\tmax_feasible_end\tstop_reason\tterms\tdeep_affine_at_end\tall_internal_affine_at_end\tinternal_direct_locations_at_end\tsigned_clearance";

pub type Poly<const W: usize> = FxHashSet<Mono<W>>;

/// Composition limits during a block's growth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// Total terms over all wires.
    pub total: usize,
    /// Terms of one wire's polynomial (and of a control product).
    pub wire: usize,
    /// Product pairs per gate update.
    pub pairs: usize,
    /// Gates per block.
    pub gates: usize,
}

impl Limits {
    /// The prototype's limits: `8 * term_cap` in total and `4 * term_cap`
    /// per wire while composing; only eligible endpoints meet `term_cap`.
    pub fn from_params(p: &PlanParams) -> Limits {
        Limits {
            total: p.term_cap * 8,
            wire: p.term_cap * 4,
            pairs: p.product_pairs,
            gates: p.max_block_gates,
        }
    }
}

#[inline]
fn toggle<const W: usize>(p: &mut Poly<W>, m: Mono<W>) {
    if !p.remove(&m) {
        p.insert(m);
    }
}

/// Incremental exact ANF of a gate sequence: `p[w]` is wire `w`'s
/// polynomial in the block's input variables (variable `w` = wire `w`).
pub struct Compose<const W: usize> {
    pub p: Vec<Poly<W>>,
    pub total: usize,
    pub changed: usize,
}

impl<const W: usize> Compose<W> {
    pub fn new(width: usize) -> Self {
        Compose {
            p: (0..width)
                .map(|w| [Mono::<W>::bit(w)].into_iter().collect())
                .collect(),
            total: width,
            changed: 0,
        }
    }

    /// Apply `g`; a rejected update leaves every polynomial untouched. A
    /// negative control multiplies by `p[w] + 1`, which is `prod * p[w] +
    /// prod` (the prototype toggled 1 into a clone of `p[w]`; the product
    /// and the pair count are the same).
    pub fn add(&mut self, g: &XGate, lim: &Limits) -> Result<(), &'static str> {
        let mut prod: Poly<W> = [Mono::<W>::ZERO].into_iter().collect();
        let mut pairs = 0usize;
        for &(w, pos) in &g.ctrls {
            let cp = &self.p[w as usize];
            let cp_len = if pos {
                cp.len()
            } else if cp.contains(&Mono::<W>::ZERO) {
                cp.len() - 1
            } else {
                cp.len() + 1
            };
            pairs = pairs.saturating_add(prod.len().saturating_mul(cp_len));
            if pairs > lim.pairs {
                return Err("product_pairs");
            }
            let mut next = Poly::default();
            for a in &prod {
                for b in cp {
                    toggle(&mut next, a.or(b));
                }
            }
            if !pos {
                for a in &prod {
                    toggle(&mut next, *a);
                }
            }
            if next.len() > lim.wire {
                return Err("intermediate_terms");
            }
            prod = next;
        }
        if g.comp {
            toggle(&mut prod, Mono::<W>::ZERO);
        }
        let t = g.target as usize;
        let old = self.p[t].len();
        // In place; toggling is an involution, so a rejection rolls back.
        for m in &prod {
            toggle(&mut self.p[t], *m);
        }
        let new = self.p[t].len();
        let total = self.total - old + new;
        let rejected = if new > lim.wire {
            Some("wire_terms")
        } else if total > lim.total {
            Some("total_terms")
        } else {
            None
        };
        if let Some(reason) = rejected {
            for m in &prod {
                toggle(&mut self.p[t], *m);
            }
            return Err(reason);
        }
        self.total = total;
        self.changed += 1;
        Ok(())
    }
}

/// Per cut `0..=m`: the number of non-output intervals (minus `excluded`
/// source gates) covering it.
fn cut_counts(ivs: &[Interval], m: usize, excluded: Option<&BTreeSet<usize>>) -> Vec<i32> {
    let mut changes = vec![0i32; m + 2];
    for iv in ivs {
        if iv.is_output || excluded.is_some_and(|x| x.contains(&iv.source_gate)) {
            continue;
        }
        assert!(
            iv.start_cut < iv.end_cut && iv.end_cut <= m + 1,
            "interval outside the circuit"
        );
        changes[iv.start_cut] += 1;
        changes[iv.end_cut] -= 1;
    }
    let mut sum = 0i32;
    for x in changes.iter_mut() {
        sum += *x;
        *x = sum;
        assert!(sum >= 0);
    }
    changes.truncate(m + 1);
    changes
}

/// Distance to the nearest change of exposure status: positive outside
/// proxy intervals, negative inside. With no change anywhere every cut gets
/// the same value (no false promise of clearance).
pub fn signed_clearance(counts: &[i32]) -> Vec<i32> {
    let m = counts.len();
    let mut distance = vec![m as i32; m];
    let mut last = None;
    for i in 0..m {
        if i > 0 && (counts[i] == 0) != (counts[i - 1] == 0) {
            last = Some(i - 1);
        }
        if let Some(j) = last {
            distance[i] = (i - j) as i32;
        }
    }
    last = None;
    for i in (0..m).rev() {
        if i + 1 < m && (counts[i] == 0) != (counts[i + 1] == 0) {
            last = Some(i + 1);
        }
        if let Some(j) = last {
            distance[i] = distance[i].min((j - i) as i32);
        }
    }
    for (d, &c) in distance.iter_mut().zip(counts) {
        if c > 0 {
            *d = -*d;
        }
    }
    distance
}

/// The per-cut scores every segment's planner reads.
struct CutScores {
    /// Deep (non-excluded) internal affine proxies.
    ac: Vec<i32>,
    /// All internal affine proxies.
    all: Vec<i32>,
    /// Internal direct locations.
    dc: Vec<i32>,
    clearance: Vec<i32>,
    /// 1 within `avoid_radius` of a previous bound.
    near: Vec<u8>,
}

struct SegmentPlan {
    bounds: Vec<usize>,
    blocks: Vec<BlockReport>,
    exceptions: usize,
}

fn plan_segment<const W: usize>(
    circuit: &[XGate],
    width: usize,
    (seg_start, seg_end): (usize, usize),
    params: &PlanParams,
    lim: &Limits,
    scores: &CutScores,
) -> SegmentPlan {
    assert!(seg_start < seg_end && seg_end <= circuit.len());
    let term_cap = params.term_cap;
    let mut out = SegmentPlan {
        bounds: Vec::new(),
        blocks: Vec::new(),
        exceptions: 0,
    };
    let mut start = seg_start;
    while start < seg_end {
        let mut state = Compose::<W>::new(width);
        let mut scanned = start;
        let mut end = start;
        let max = (start + lim.gates).min(seg_end);
        let mut reason = "gate_cap";
        let mut terms = vec![width];
        while scanned < max {
            match state.add(&circuit[scanned], lim) {
                Ok(()) => {
                    scanned += 1;
                    terms.push(state.total);
                    if state.total <= term_cap {
                        end = scanned;
                    }
                    if scanned - end >= params.lookahead {
                        reason = "lookahead_limit";
                        break;
                    }
                }
                Err(r) => {
                    reason = r;
                    break;
                }
            }
        }
        if end == seg_end {
            reason = "end";
        }
        if end == start {
            // A wide single gate can exceed the symbolic cap; boundary
            // selection cannot divide it. Keep it as its own block.
            end += 1;
            if terms.len() == 1 {
                terms.push(width);
            }
            reason = "single_gate_exception";
            out.exceptions += 1;
        }
        // Only the upper half of the feasible growth interval is eligible:
        // no tiny block merely for a lucky boundary.
        let lo = start + (end - start).div_ceil(2);
        let chosen = if end == seg_end {
            seg_end
        } else {
            (lo..=end)
                .filter(|&cut| terms[cut - start] <= term_cap || end == start + 1)
                .min_by_key(|&cut| {
                    (
                        scores.ac[cut],
                        scores.all[cut],
                        scores.near[cut],
                        -scores.clearance[cut],
                        scores.dc[cut],
                        Reverse(cut),
                    )
                })
                .expect("the feasible end is eligible")
        };
        assert!(chosen > start);
        if chosen < seg_end {
            out.bounds.push(chosen);
        }
        out.blocks.push(BlockReport {
            start,
            end: chosen,
            max_feasible_end: end,
            stop_reason: reason,
            terms: terms[chosen - start],
            deep_affine_at_end: scores.ac[chosen],
            all_internal_affine_at_end: scores.all[chosen],
            direct_locations_at_end: scores.dc[chosen],
            signed_clearance: scores.clearance[chosen],
        });
        start = chosen;
    }
    out
}

struct PlanJob<'a> {
    circuit: &'a [XGate],
    width: usize,
    segments: Vec<(usize, usize)>,
    params: &'a PlanParams,
    lim: Limits,
    scores: &'a CutScores,
}

impl MonoWidthFn<Vec<SegmentPlan>> for PlanJob<'_> {
    fn call<const W: usize>(self) -> Vec<SegmentPlan> {
        let run = |seg: &(usize, usize)| {
            plan_segment::<W>(
                self.circuit,
                self.width,
                *seg,
                self.params,
                &self.lim,
                self.scores,
            )
        };
        if self.segments.len() == 1 {
            vec![run(&self.segments[0])]
        } else {
            self.segments.par_iter().map(run).collect()
        }
    }
}

/// The seams of the parallel planner (see the module doc).
fn seams(
    m: usize,
    segments: usize,
    coverage: &[i32],
    prev_bounds: &[usize],
    radius: usize,
) -> Vec<usize> {
    let mut seams = Vec::new();
    for i in 1..segments {
        let target = i * m / segments;
        let mut best = None;
        'search: for d in 0..SEAM_SEARCH {
            for t in [target.checked_sub(d), Some(target + d)]
                .into_iter()
                .flatten()
            {
                if t > 0
                    && t < m
                    && coverage[t] == 0
                    && prev_bounds.iter().all(|&b| t.abs_diff(b) > radius)
                {
                    best = Some(t);
                    break 'search;
                }
            }
        }
        seams.push(best.unwrap_or(target));
    }
    // Degenerate tiny circuits: a target of 0 or m is not a seam.
    seams.retain(|&t| t > 0 && t < m);
    seams.sort_unstable();
    seams.dedup();
    seams
}

/// Plan `circuit` (`width` wires, at most 1024) against `census` (taken on
/// the same circuit; `Census::empty` for C-blind planning) with the births
/// of `source` excluded per `source.excluded`. `prev_bounds` are the
/// previous round's boundaries (penalized within `params.avoid_radius`).
pub fn plan(
    circuit: &[XGate],
    width: usize,
    census: &Census,
    source: &Source,
    params: &PlanParams,
    prev_bounds: &[usize],
) -> Plan {
    let started = Instant::now();
    let m = circuit.len();
    assert_eq!(census.gates, m, "census of another circuit");
    assert!(
        circuit.is_empty() || (max_wire(circuit) as usize) < width,
        "circuit uses a wire beyond width {width}"
    );
    let lim = Limits::from_params(params);
    assert!(lim.total >= width, "term cap below the wire count");
    let mut near = vec![0u8; m + 1];
    if params.avoid_radius > 0 {
        for &b in prev_bounds {
            let lo = b.saturating_sub(params.avoid_radius);
            let hi = (b + params.avoid_radius).min(m);
            near[lo..=hi].fill(1);
        }
    }
    let all = cut_counts(&census.affine, m, None);
    let scores = CutScores {
        ac: cut_counts(&census.affine, m, Some(&source.excluded)),
        clearance: signed_clearance(&all),
        all,
        dc: cut_counts(&census.direct, m, None),
        near,
    };
    if m == 0 {
        return Plan {
            seconds: started.elapsed().as_secs_f64(),
            ..Default::default()
        };
    }
    let segments = params.segments.max(1);
    let sequential = segments == 1 && params.avoid_radius == 0;
    let seams = if sequential {
        Vec::new()
    } else {
        seams(
            m,
            segments,
            &census.coverage(),
            prev_bounds,
            params.avoid_radius,
        )
    };
    let mut edges = Vec::with_capacity(seams.len() + 2);
    edges.push(0);
    edges.extend_from_slice(&seams);
    edges.push(m);
    let segs: Vec<(usize, usize)> = edges.windows(2).map(|e| (e[0], e[1])).collect();
    let results = with_mono_width(
        width,
        PlanJob {
            circuit,
            width,
            segments: segs,
            params,
            lim,
            scores: &scores,
        },
    );
    let mut bounds: BTreeSet<usize> = seams.iter().copied().collect();
    let mut blocks = Vec::new();
    let mut exceptions = 0;
    for r in results {
        bounds.extend(r.bounds);
        blocks.extend(r.blocks);
        exceptions += r.exceptions;
    }
    Plan {
        bounds: bounds.into_iter().collect(),
        seams,
        blocks,
        single_gate_exceptions: exceptions,
        seconds: started.elapsed().as_secs_f64(),
    }
}

// ---------------------------------------------------------------------------
// the exposure profile (refinement splits)
// ---------------------------------------------------------------------------

/// The exposure profile of a circuit: per cut the (deep affine, all
/// affine, direct) counts of non-output intervals, and the maximal runs of
/// cuts with affine exposure (`unions`). Built from the discovery intervals
/// (not validated-only), like the prototype.
#[derive(Clone, Debug)]
pub struct Profile {
    count: usize,
    cuts: Vec<usize>,
    values: Vec<[i32; 3]>,
    unions: Vec<(usize, usize)>,
    starts: Vec<usize>,
}

impl Profile {
    pub fn new(census: &Census, source: &Source) -> Profile {
        let count = census.gates;
        let mut changes: BTreeMap<usize, [i32; 3]> = BTreeMap::new();
        for (affine, ivs) in [(true, &census.affine), (false, &census.direct)] {
            for iv in ivs {
                if iv.is_output {
                    continue;
                }
                let d = if affine {
                    [(!source.excluded.contains(&iv.source_gate)) as i32, 1, 0]
                } else {
                    [0, 0, 1]
                };
                let a = changes.entry(iv.start_cut).or_default();
                for j in 0..3 {
                    a[j] += d[j];
                }
                let b = changes.entry(iv.end_cut).or_default();
                for j in 0..3 {
                    b[j] -= d[j];
                }
            }
        }
        changes.entry(0).or_default();
        changes.entry(count + 1).or_default();
        let mut running = [0i32; 3];
        let mut cuts = Vec::with_capacity(changes.len());
        let mut values = Vec::with_capacity(changes.len());
        let mut unions = Vec::new();
        let mut begin = None;
        for (&c, d) in &changes {
            let before = running[1];
            for j in 0..3 {
                running[j] += d[j];
            }
            if before == 0 && running[1] > 0 {
                begin = Some(c);
            } else if before > 0 && running[1] == 0 {
                unions.push((begin.take().expect("an open run"), c));
            }
            cuts.push(c);
            values.push(running);
        }
        let starts = unions.iter().map(|&(a, _)| a).collect();
        Profile {
            count,
            cuts,
            values,
            unions,
            starts,
        }
    }

    /// (deep affine, all affine, direct) at `cut`.
    pub fn counts(&self, cut: usize) -> [i32; 3] {
        let i = self.cuts.partition_point(|&c| c <= cut);
        self.values[i - 1]
    }

    /// Distance from `cut` to the nearest end of affine exposure: negative
    /// inside a run, positive outside (`count + 1` with nothing in sight).
    pub fn clearance(&self, cut: usize) -> i32 {
        let i = self.starts.partition_point(|&s| s <= cut);
        let far = (self.count + 1) as i32;
        if i > 0 && self.unions[i - 1].1 > cut {
            let (a, b) = self.unions[i - 1];
            let mut d = Vec::new();
            if a > 0 {
                d.push((cut - a + 1) as i32);
            }
            if b <= self.count {
                d.push((b - cut) as i32);
            }
            return -d.into_iter().min().unwrap_or(far);
        }
        let mut d = Vec::new();
        if i > 0 {
            d.push((cut - self.unions[i - 1].1 + 1) as i32);
        }
        if i < self.unions.len() {
            d.push((self.unions[i].0 - cut) as i32);
        }
        d.into_iter().min().unwrap_or(far)
    }

    /// A refinement cut of the block `[lo, hi)` (`hi - lo >= 2`): among the
    /// candidates of the middle half (its ends and midpoint, every exposure
    /// change there and its neighbours, the midpoints between changes, the
    /// centers of the gaps between exposure runs), the one with the fewest
    /// deep affine proxies, then fewest affine proxies, then greatest
    /// clearance, fewest direct locations, nearest the midpoint, latest.
    /// Returns `lo < cut < hi`.
    pub fn split(&self, lo: usize, hi: usize) -> usize {
        assert!(hi >= lo + 2, "a block of at least two gates splits");
        let quarter = ((hi - lo) / 4).max(1);
        let (low, high) = (lo + quarter, hi - quarter);
        let middle = (lo + hi) / 2;
        let mut cand: BTreeSet<usize> = [low, high, middle].into_iter().collect();
        let ix = self.cuts.partition_point(|&c| c < low);
        let iy = self.cuts.partition_point(|&c| c <= high);
        let mut events = vec![low];
        events.extend_from_slice(&self.cuts[ix..iy.max(ix)]);
        events.push(high);
        let inside = |x: usize| low <= x && x <= high;
        for &c in &events {
            cand.extend(
                [c.wrapping_sub(1), c, c + 1]
                    .into_iter()
                    .filter(|&x| inside(x)),
            );
        }
        for e in events.windows(2) {
            let (a, b) = (e[0], e[1]);
            cand.insert((a + b) / 2);
            cand.insert((a + b).div_ceil(2));
        }
        for u in self.unions.windows(2) {
            let center = u[0].1 - 1 + u[1].0;
            cand.extend(
                [center / 2, center.div_ceil(2)]
                    .into_iter()
                    .filter(|&x| inside(x)),
            );
        }
        let cut = cand
            .into_iter()
            .min_by_key(|&c| {
                let k = self.counts(c);
                (
                    k[0],
                    k[1],
                    -self.clearance(c),
                    k[2],
                    c.abs_diff(middle),
                    Reverse(c),
                )
            })
            .expect("candidates exist");
        assert!(lo < cut && cut < hi);
        cut
    }
}

// ---------------------------------------------------------------------------
// files
// ---------------------------------------------------------------------------

/// `bounds` (one cut per line), `blocks.tsv` (the prototype's columns) and
/// `summary.txt` (`key=value` lines).
pub fn write_plan(p: &Plan, dir: &Path) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    let mut b = BufWriter::new(fs::File::create(dir.join("bounds"))?);
    for x in &p.bounds {
        writeln!(b, "{x}")?;
    }
    b.flush()?;
    let mut r = BufWriter::new(fs::File::create(dir.join("blocks.tsv"))?);
    writeln!(r, "{BLOCKS_HEADER}")?;
    for k in &p.blocks {
        writeln!(
            r,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            k.start,
            k.end,
            k.max_feasible_end,
            k.stop_reason,
            k.terms,
            k.deep_affine_at_end,
            k.all_internal_affine_at_end,
            k.direct_locations_at_end,
            k.signed_clearance
        )?;
    }
    r.flush()?;
    let mut s = BufWriter::new(fs::File::create(dir.join("summary.txt"))?);
    writeln!(s, "bounds={}", p.bounds.len())?;
    writeln!(s, "blocks={}", p.blocks.len())?;
    let seams = p
        .seams
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(",");
    writeln!(s, "seams={seams}")?;
    writeln!(s, "single_gate_exceptions={}", p.single_gate_exceptions)?;
    writeln!(s, "seconds={:?}", p.seconds)?;
    s.flush()
}

/// The cuts of a `bounds` file (one per line, blank lines ignored).
pub fn read_bounds(path: &Path) -> io::Result<Vec<usize>> {
    fs::read_to_string(path)?
        .split_whitespace()
        .map(|t| parse_num(t, "bound"))
        .collect::<io::Result<Vec<usize>>>()
        .and_then(|v| {
            if v.windows(2).all(|w| w[0] < w[1]) {
                Ok(v)
            } else {
                Err(invalid(format!("{}: bounds not ascending", path.display())))
            }
        })
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/adaptive/planner.rs"]
mod tests;
