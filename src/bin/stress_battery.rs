//! A standing battery of feasible correlates for a gadgetized circuit.
//!
//! Motivation. Every hiding claim in this project is class-relative: a
//! gadget snapshot is a bijective image of the input, so an UNBOUNDED
//! correlate reconstructs the whole original trajectory and there is nothing
//! to measure. What is measurable — and what security actually rests on — is
//! whether some FEASIBLE predictor class recovers computational-progress
//! structure. The right object is therefore not one heatmap but a battery:
//! a structured matrix of bounded correlates, run on every candidate design,
//! versioned like a test suite. This binary is that matrix.
//!
//! It scores an (original C, gadget G) pair along four axes:
//!
//!   * TARGET class (what original-side function to predict): single C-state
//!     bit; a 2-bit parity of C-state bits; a trajectory / gate-activity bit
//!     (did wire t flip over the last c-step) — the progress-clock channel no
//!     execution measure in the tree has looked at.
//!   * FEATURE class (gadget-side predictor complexity): best agreement over
//!     {const, single wire, XOR of two wires} (F1), optionally plus one AND
//!     over a declared wire set (F2). Bias-sensitive by construction — every
//!     score is an achieved agreement, never snapped to 0.5, so it sees the
//!     statistical margin the exact-span ridge is blind to.
//!   * ACCESS model: execution sampling (the heatmap probes above); plus two
//!     NON-execution probes — a deterministic syntactic census (write/read
//!     shape, unwritten-in-window, band separability) and a degree-1 GF(2)
//!     affine-invariant miner (the forward-learnable-invariant / SAT channel).
//!   * NULLS: a max-over-candidates noise floor; a column-permutation perm-z
//!     for the progress-alignment claim; and an optional DECOY-source null —
//!     run the same probe against an independently generated circuit C' that
//!     is functionally equivalent to C. A ridge that is as strong against a
//!     decoy as against the true source is leaking generic progress, not the
//!     identity of C; the true-minus-decoy contrast is the honest signal.
//!
//! Every number is a LOWER BOUND on what an adversary in the class can do:
//! predictor families are capped, grids are coarse, and "dead" always means
//! "dead against this battery at this budget", never unconditionally.
//!
//! Extensibility: a new correlate is a new row in the probe table (a target
//! generator or a feature scorer) — the sampling core, ridge readout, nulls
//! and report are shared. Adding one should not touch anything else.
//!
//! Usage:
//!   stress_battery --c C.g57 --g G.mpmct1 --n 128 \
//!       [--and-wires 512-567] [--band 512-567] [--decoy C1.g57,C2.g57] \
//!       [--pred-wires 0-567] [--c-step .. --g-step .. --samples ..] \
//!       [--out report_prefix]

use clap::Parser;
use local_mixing::postmix::format::{read_g57_file, read_mpmct};
use local_mixing::postmix::xgate::{XGate, max_wire};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::io::Write;

#[derive(Parser, Debug)]
#[command(name = "stress_battery")]
struct Args {
    /// Original circuit C.
    #[arg(long)]
    c: String,
    /// Gadgetized circuit G.
    #[arg(long)]
    g: String,
    #[arg(long, default_value = "g57")]
    c_format: String,
    #[arg(long, default_value = "mpmct1")]
    g_format: String,
    /// Logical input width n: random x on wires 0..n-1, zeros elsewhere, on
    /// both circuits. C's target bits are wires 0..n-1.
    #[arg(long)]
    n: usize,
    /// Prefix strides (coarse by default — the pair search is O(W^2)/cell).
    #[arg(long, default_value_t = 200)]
    c_step: usize,
    #[arg(long, default_value_t = 20000)]
    g_step: usize,
    /// Samples, rounded up to a multiple of 64.
    #[arg(long, default_value_t = 8192)]
    samples: usize,
    /// Target bits / parities sampled per cell (0 = all single bits).
    #[arg(long, default_value_t = 16)]
    target_bits: usize,
    /// Predictor wire set for the XOR search, e.g. "0-567" (empty = all G).
    #[arg(long, default_value = "")]
    pred_wires: String,
    /// Declared AND set for the F2 (deg-2-capable) feature, e.g. "512-567"
    /// (empty = skip F2).
    #[arg(long, default_value = "")]
    and_wires: String,
    /// Known band range for the census localization AUC, e.g. "512-567"
    /// (empty = report the separability gap only).
    #[arg(long, default_value = "")]
    band: String,
    /// Comma-separated decoy circuits C' (functionally equivalent to C, same
    /// n), for the decoy-source null. Same format as --c.
    #[arg(long, default_value = "")]
    decoy: String,
    /// Window sizes (gates) for the unwritten-in-window census sweep.
    #[arg(long, default_value = "10000,50000")]
    census_windows: String,
    #[arg(long, default_value_t = 12345)]
    seed: u64,
    /// Report prefix; writes <out>.json. Empty = stdout only.
    #[arg(long, default_value = "")]
    out: String,
}

// ---------------------------------------------------------------------------
// IO + sampling core
// ---------------------------------------------------------------------------

fn read_circuit(path: &str, fmt: &str) -> Vec<XGate> {
    match fmt {
        "g57" => read_g57_file(path).unwrap_or_else(|_| panic!("read {path} (g57)")),
        "mpmct1" => read_mpmct(path).unwrap_or_else(|_| panic!("read {path} (mpmct1)")).0,
        o => panic!("unknown format {o}"),
    }
}

fn indices(end: usize, step: usize) -> Vec<usize> {
    let mut v = Vec::new();
    let mut i = 0;
    while i < end {
        v.push(i);
        i += step;
    }
    v.push(end);
    v
}

/// Lane-state (bit l = sample l) at each prefix length in `idx`.
fn snapshots(gates: &[XGate], init: &[u64], idx: &[usize]) -> Vec<Vec<u64>> {
    let mut state = init.to_vec();
    let mut out = Vec::with_capacity(idx.len());
    let mut k = 0;
    for pos in 0..=gates.len() {
        while k < idx.len() && idx[k] == pos {
            out.push(state.clone());
            k += 1;
        }
        if pos < gates.len() {
            gates[pos].apply_lanes(&mut state);
        }
    }
    out
}

fn parse_wire_list(spec: &str, nw: usize) -> Vec<usize> {
    if spec.is_empty() {
        return (0..nw).collect();
    }
    let mut v = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if let Some((lo, hi)) = part.split_once('-') {
            let lo: usize = lo.parse().expect("bad wire range");
            let hi: usize = hi.parse().expect("bad wire range");
            v.extend(lo..=hi.min(nw - 1));
        } else if !part.is_empty() {
            v.push(part.parse().expect("bad wire entry"));
        }
    }
    v.sort_unstable();
    v.dedup();
    v
}

// ---------------------------------------------------------------------------
// Feature scorer: best agreement of a packed target with a bounded predictor
// family over G's wires. Bias-sensitive (max(a,1-a)), always a lower bound.
// Returns (F1 = best over {const, single, XOR-pair}, F2 = F1 plus one AND,
// const_agree = best CONSTANT predictor = max(p,1-p) = the target's base
// rate). Callers score the LIFT best − const_agree, which auto-corrects for a
// biased/sparse target (e.g. a rare gate-flip event): a predictor that only
// exploits the base rate scores zero lift, so prominence is not inflated by
// target bias — only genuine predictive information counts.
// ---------------------------------------------------------------------------

fn best_agreement(
    target: &[u64],
    wires: &[Vec<u64>],
    and_wires: &[Vec<u64>],
    nbits: f64,
) -> (f64, f64, f64) {
    let words = target.len();
    let agreement = |mismatch: u32| {
        let a = mismatch as f64 / nbits;
        if a < 0.5 { 1.0 - a } else { a }
    };
    let mut best = 0f64;
    let mut best_pair = (usize::MAX, usize::MAX);
    let m: u32 = target.iter().map(|w| w.count_ones()).sum();
    let const_agree = agreement(m); // best constant predictor (target base rate)
    best = best.max(const_agree);
    let dev: Vec<Vec<u64>> = wires
        .iter()
        .map(|u| (0..words).map(|w| target[w] ^ u[w]).collect())
        .collect();
    for (i, d) in dev.iter().enumerate() {
        let m: u32 = d.iter().map(|w| w.count_ones()).sum();
        let a = agreement(m);
        if a > best {
            best = a;
            best_pair = (i, usize::MAX);
        }
    }
    for (i, d) in dev.iter().enumerate() {
        for (off, u) in wires[i + 1..].iter().enumerate() {
            let m: u32 = (0..words).map(|w| (d[w] ^ u[w]).count_ones()).sum();
            let a = agreement(m);
            if a > best {
                best = a;
                best_pair = (i, i + 1 + off);
            }
        }
    }
    if and_wires.is_empty() {
        return (best, best, const_agree);
    }
    let mut best_res: Vec<u64> = target.to_vec();
    for idx in [best_pair.0, best_pair.1] {
        if idx != usize::MAX {
            for w in 0..words {
                best_res[w] ^= wires[idx][w];
            }
        }
    }
    let mut aug = best;
    for (i, p) in and_wires.iter().enumerate() {
        for q in &and_wires[i + 1..] {
            for (pa, qa) in [(false, false), (false, true), (true, false), (true, true)] {
                let m: u32 = (0..words)
                    .map(|w| {
                        let a = if pa { !p[w] } else { p[w] };
                        let b = if qa { !q[w] } else { q[w] };
                        (best_res[w] ^ (a & b)).count_ones()
                    })
                    .sum();
                let a = agreement(m);
                if a > aug {
                    aug = a;
                }
            }
        }
    }
    (best, aug, const_agree)
}

// ---------------------------------------------------------------------------
// Ridge readout for an "agreement" plate (high = leaky). Interior-only:
// both axes have unencoded ports and (0,0) reads 1.0 in every build.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Ridge {
    plate_mean: f64,
    floor: f64,
    /// Median over interior rows of (row-best agreement − floor).
    prominence: f64,
    /// Peak interior row-best − floor.
    peak: f64,
    /// Spearman(row index, argmax column) over interior rows: progress
    /// alignment. ~1 = a clean advancing diagonal, ~0 = none.
    rho: f64,
    /// z-score of rho against permuting the argmax-column assignment.
    perm_z: f64,
    rows: usize,
    cols: usize,
}

fn spearman(xs: &[f64], ys: &[f64]) -> f64 {
    let rank = |v: &[f64]| -> Vec<f64> {
        let n = v.len();
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
        let mut r = vec![0f64; n];
        let mut i = 0;
        while i < n {
            let mut j = i;
            while j + 1 < n && v[idx[j + 1]] == v[idx[i]] {
                j += 1;
            }
            let avg = (i + j) as f64 / 2.0; // average rank for ties
            for k in i..=j {
                r[idx[k]] = avg;
            }
            i = j + 1;
        }
        r
    };
    let rx = rank(xs);
    let ry = rank(ys);
    pearson(&rx, &ry)
}

fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for i in 0..xs.len() {
        let dx = xs[i] - mx;
        let dy = ys[i] - my;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if sxx <= 0.0 || syy <= 0.0 {
        return 0.0;
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}

fn read_ridge(mat: &[f64], rows: usize, cols: usize, floor: f64, rng: &mut impl Rng) -> Ridge {
    let plate_mean = mat.iter().sum::<f64>() / mat.len().max(1) as f64;
    let (r0, c0) = ((rows / 10).max(1), (cols / 10).max(1));
    let mut row_idx: Vec<f64> = Vec::new();
    let mut argmax_col: Vec<f64> = Vec::new();
    let mut row_best: Vec<f64> = Vec::new();
    for ri in r0..rows.saturating_sub(r0) {
        let mut best = f64::MIN;
        let mut bc = c0;
        for cj in c0..cols.saturating_sub(c0) {
            let v = mat[ri * cols + cj];
            if v > best {
                best = v;
                bc = cj;
            }
        }
        if best == f64::MIN {
            continue;
        }
        row_idx.push(ri as f64);
        argmax_col.push(bc as f64);
        row_best.push(best);
    }
    let mut sorted = row_best.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted.get(sorted.len() / 2).copied().unwrap_or(f64::NAN);
    let peak = sorted.last().copied().unwrap_or(f64::NAN);
    let rho = if row_idx.len() >= 3 {
        spearman(&row_idx, &argmax_col)
    } else {
        0.0
    };
    // perm-z: permute the argmax-column assignment, recompute rho.
    let perm_z = if row_idx.len() >= 3 {
        let k = 200;
        let mut perm = argmax_col.clone();
        let mut vals = Vec::with_capacity(k);
        for _ in 0..k {
            perm.shuffle(rng);
            vals.push(spearman(&row_idx, &perm));
        }
        let mu = vals.iter().sum::<f64>() / k as f64;
        let var = vals.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / k as f64;
        if var > 1e-12 { (rho - mu) / var.sqrt() } else { 0.0 }
    } else {
        0.0
    };
    Ridge {
        plate_mean,
        floor,
        prominence: median - floor,
        peak: peak - floor,
        rho,
        perm_z,
        rows,
        cols,
    }
}

// ---------------------------------------------------------------------------
// Execution-probe engine. A probe = (target generator, feature family) scored
// as an agreement plate + ridge readout. Targets are functions of C-state at
// row ri; features are G-wire predictors at column cj.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum TargetKind {
    Bit,
    Par2,
    Traj,
}

struct Sampled {
    // [batch][row][wire] for C, [batch][col][wire] for G.
    cs: Vec<Vec<Vec<u64>>>,
    gs: Vec<Vec<Vec<u64>>>,
    batches: usize,
    rows: usize,
    cols: usize,
}

/// Build the packed target vector for (row ri, target index t) of a chosen
/// kind, from a snapshot set `cs` (C or a decoy). par2/traj use `aux` (a
/// second bit / the previous row).
fn build_target(
    cs: &[Vec<Vec<u64>>],
    batches: usize,
    ri: usize,
    kind: TargetKind,
    bits: &[usize],
    pairs: &[(usize, usize)],
    t: usize,
) -> Vec<u64> {
    match kind {
        TargetKind::Bit => {
            let b = bits[t];
            (0..batches).map(|batch| cs[batch][ri][b]).collect()
        }
        TargetKind::Par2 => {
            let (a, c) = pairs[t];
            (0..batches).map(|batch| cs[batch][ri][a] ^ cs[batch][ri][c]).collect()
        }
        TargetKind::Traj => {
            let b = bits[t];
            let prev = ri.saturating_sub(1);
            (0..batches).map(|batch| cs[batch][ri][b] ^ cs[batch][prev][b]).collect()
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_probe(
    s: &Sampled,
    cs_for_targets: &[Vec<Vec<u64>>],
    kind: TargetKind,
    bits: &[usize],
    pairs: &[(usize, usize)],
    pred: &[usize],
    and_set: &[usize],
    nbits: f64,
    floor: f64,
    rng: &mut impl Rng,
) -> (Ridge, Option<Ridge>) {
    let n_t = match kind {
        TargetKind::Par2 => pairs.len(),
        _ => bits.len(),
    };
    let mut m1 = vec![0f64; s.rows * s.cols];
    let mut m2 = vec![0f64; s.rows * s.cols];
    for ri in 0..s.rows {
        for cj in 0..s.cols {
            let wires: Vec<Vec<u64>> =
                pred.iter().map(|&k| (0..s.batches).map(|b| s.gs[b][cj][k]).collect()).collect();
            let ands: Vec<Vec<u64>> =
                and_set.iter().map(|&k| (0..s.batches).map(|b| s.gs[b][cj][k]).collect()).collect();
            let (mut sum1, mut sum2) = (0f64, 0f64);
            for t in 0..n_t {
                let target = build_target(cs_for_targets, s.batches, ri, kind, bits, pairs, t);
                let (a1, a2, base) = best_agreement(&target, &wires, &ands, nbits);
                // Lift over the per-target base rate (bias-corrected).
                sum1 += a1 - base;
                sum2 += a2 - base;
            }
            m1[ri * s.cols + cj] = sum1 / n_t as f64;
            m2[ri * s.cols + cj] = sum2 / n_t as f64;
        }
    }
    let r1 = read_ridge(&m1, s.rows, s.cols, floor, rng);
    let r2 = if and_set.is_empty() {
        None
    } else {
        Some(read_ridge(&m2, s.rows, s.cols, floor, rng))
    };
    (r1, r2)
}

// ---------------------------------------------------------------------------
// Non-execution probe 1: syntactic census (deterministic, no sampling).
// ---------------------------------------------------------------------------

struct Census {
    wires: usize,
    writes_min: u64,
    writes_med: u64,
    writes_max: u64,
    reads_min: u64,
    reads_med: u64,
    reads_max: u64,
    /// Largest gap between adjacent sorted write-counts, normalized by range:
    /// a wide gap = a threshold cleanly separates two wire populations (the
    /// band-vs-carrier signature). 0 = smooth, →1 = bimodal.
    write_bimodality_gap: f64,
    /// Per window size: max over wires of "unwritten in some window of this
    /// size" — a wire never written in a window is trivially a source/band
    /// candidate. Reported as (window, unwritten_wire_count).
    unwritten_in_window: Vec<(usize, usize)>,
    /// If --band given: AUC of using each per-wire statistic to rank band vs
    /// non-band (1.0 = a statistic perfectly localizes the band). Reported for
    /// write-count and width-1-write-share.
    band_write_auc: Option<f64>,
    band_w1share_auc: Option<f64>,
}

fn median_u64(v: &mut [u64]) -> u64 {
    v.sort_unstable();
    v.get(v.len() / 2).copied().unwrap_or(0)
}

fn census(g: &[XGate], nw: usize, band: &[usize], windows: &[usize]) -> Census {
    let mut writes = vec![0u64; nw];
    let mut reads = vec![0u64; nw];
    let mut w1writes = vec![0u64; nw]; // width-1 (plain CNOT) writes
    // last write position per wire, to find max unwritten gap per window.
    for (pos, gate) in g.iter().enumerate() {
        let t = gate.target as usize;
        writes[t] += 1;
        if gate.width() <= 1 {
            w1writes[t] += 1;
        }
        for &(w, _) in gate.ctrls.iter() {
            reads[w as usize] += 1;
        }
        let _ = pos;
    }
    let mut wsort = writes.clone();
    let writes_med = median_u64(&mut wsort);
    let writes_min = *wsort.first().unwrap_or(&0);
    let writes_max = *wsort.last().unwrap_or(&0);
    // bimodality: largest normalized gap in the sorted write counts.
    let mut gap = 0f64;
    if wsort.len() >= 2 && writes_max > writes_min {
        for i in 1..wsort.len() {
            let g = (wsort[i] - wsort[i - 1]) as f64;
            gap = gap.max(g);
        }
        gap /= (writes_max - writes_min) as f64;
    }
    let mut rsort = reads.clone();
    let reads_med = median_u64(&mut rsort);
    let reads_min = *rsort.first().unwrap_or(&0);
    let reads_max = *rsort.last().unwrap_or(&0);

    // unwritten-in-window: slide a window of size W over gate positions; a
    // wire written outside every window of that size somewhere is a candidate.
    // Simpler robust proxy: count wires whose MAX gap between consecutive
    // writes (with sentinels at 0 and len) exceeds the window size.
    let mut last = vec![0usize; nw];
    let mut maxgap = vec![0usize; nw];
    let seen = &mut vec![false; nw];
    for (pos, gate) in g.iter().enumerate() {
        let t = gate.target as usize;
        if seen[t] {
            maxgap[t] = maxgap[t].max(pos - last[t]);
        } else {
            maxgap[t] = maxgap[t].max(pos); // gap from start
            seen[t] = true;
        }
        last[t] = pos;
    }
    let len = g.len();
    for w in 0..nw {
        let tail = if seen[w] { len - last[w] } else { len };
        maxgap[w] = maxgap[w].max(tail);
    }
    let unwritten_in_window: Vec<(usize, usize)> = windows
        .iter()
        .map(|&win| (win, (0..nw).filter(|&w| maxgap[w] >= win).count()))
        .collect();

    // width-1 write share per wire.
    let w1share: Vec<f64> = (0..nw)
        .map(|w| if writes[w] > 0 { w1writes[w] as f64 / writes[w] as f64 } else { 0.0 })
        .collect();

    let (band_write_auc, band_w1share_auc) = if band.is_empty() {
        (None, None)
    } else {
        let in_band = |w: usize| band.binary_search(&w).is_ok();
        let auc = |score: &dyn Fn(usize) -> f64| -> f64 {
            // AUC of ranking: P(band wire scores below a non-band wire) under
            // the hypothesis "band is the LOW-write / HIGH-w1share tail".
            // Report max(auc, 1-auc) so either direction of separation counts.
            let mut pos = Vec::new(); // band
            let mut neg = Vec::new(); // non-band
            for w in 0..nw {
                if in_band(w) { pos.push(score(w)) } else { neg.push(score(w)) }
            }
            let (mut wins, mut ties, mut tot) = (0f64, 0f64, 0f64);
            for &p in &pos {
                for &q in &neg {
                    tot += 1.0;
                    if p < q { wins += 1.0 } else if p == q { ties += 1.0 }
                }
            }
            if tot == 0.0 { return 0.5; }
            let a = (wins + 0.5 * ties) / tot;
            a.max(1.0 - a)
        };
        (
            Some(auc(&|w| writes[w] as f64)),
            Some(auc(&|w| w1share[w])),
        )
    };

    Census {
        wires: nw,
        writes_min,
        writes_med,
        writes_max,
        reads_min,
        reads_med,
        reads_max,
        write_bimodality_gap: gap,
        unwritten_in_window,
        band_write_auc,
        band_w1share_auc,
    }
}

// ---------------------------------------------------------------------------
// Non-execution probe 2: degree-1 GF(2) affine-invariant miner. At a column,
// find affine relations over G's wires that hold on all train samples, and
// report how many survive on holdout — free equations for a forward-learning
// (SAT-guiding) adversary. Ports aside, a hidden invariant is a real leak the
// reconstruction heatmap never sees.
// ---------------------------------------------------------------------------

fn bit(v: &[u64], i: usize) -> bool {
    (v[i / 64] >> (i % 64)) & 1 == 1
}
fn xor_into(dst: &mut [u64], src: &[u64]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d ^= s;
    }
}
fn first_set(v: &[u64]) -> Option<usize> {
    for (w, &word) in v.iter().enumerate() {
        if word != 0 {
            return Some(w * 64 + word.trailing_zeros() as usize);
        }
    }
    None
}

/// Count affine invariants at column cj: relations sum_{k in S} g_k (+const)
/// = 0 holding on train, verified on holdout. Reduces feature sample-vectors
/// (wires + const) to a GF(2) basis; a feature that reduces to zero yields an
/// invariant (its accumulated coefficient set), which is then checked on the
/// holdout batches.
fn invariants_at(
    gs: &[Vec<Vec<u64>>],
    cj: usize,
    nw: usize,
    train_b: usize,
    total_b: usize,
) -> usize {
    // Regressor r in 0..nw = wire r; r == nw = const.
    let n_reg = nw + 1;
    let coef_words = n_reg.div_ceil(64);
    let feat_train = |r: usize| -> Vec<u64> {
        if r == nw {
            vec![!0u64; train_b]
        } else {
            (0..train_b).map(|b| gs[b][cj][r]).collect()
        }
    };
    struct Row {
        samp: Vec<u64>,
        coef: Vec<u64>,
    }
    let mut basis: Vec<(usize, Row)> = Vec::new();
    let mut invariants: Vec<Vec<u64>> = Vec::new();
    for r in 0..n_reg {
        let mut samp = feat_train(r);
        let mut coef = vec![0u64; coef_words];
        coef[r / 64] |= 1u64 << (r % 64);
        for (piv, brow) in basis.iter() {
            if bit(&samp, *piv) {
                xor_into(&mut samp, &brow.samp);
                xor_into(&mut coef, &brow.coef);
            }
        }
        match first_set(&samp) {
            Some(p) => basis.push((p, Row { samp, coef })),
            None => invariants.push(coef), // reduced to 0 on train: a relation
        }
    }
    // Verify each candidate invariant on the holdout batches.
    let ho = total_b - train_b;
    if ho == 0 {
        return invariants.len();
    }
    let mut verified = 0;
    for coef in &invariants {
        let mut ok = true;
        'batches: for b in train_b..total_b {
            let mut acc = 0u64;
            for r in 0..n_reg {
                if (coef[r / 64] >> (r % 64)) & 1 == 1 {
                    acc ^= if r == nw { !0u64 } else { gs[b][cj][r] };
                }
            }
            if acc != 0 {
                ok = false;
                break 'batches;
            }
        }
        let _ = ho;
        if ok {
            verified += 1;
        }
    }
    verified
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

struct ProbeReport {
    target: String,
    feature: String,
    ridge: Ridge,
    /// Decoy ridges (same probe, decoy as target-source) and the honest
    /// true-minus-decoy prominence contrast.
    decoy_prominence: Vec<f64>,
    contrast_vs_decoy: Option<f64>,
}

struct Report {
    c: String,
    g: String,
    n: usize,
    c_gates: usize,
    g_gates: usize,
    g_wires: usize,
    rows: usize,
    cols: usize,
    samples: usize,
    floor: f64,
    probes: Vec<ProbeReport>,
    invariant_interior_max: usize,
    invariant_interior_median: usize,
    census: Census,
}

fn main() {
    let args = Args::parse();
    let c = read_circuit(&args.c, &args.c_format);
    let g = read_circuit(&args.g, &args.g_format);
    let n = args.n;
    let nw_g = (max_wire(&g) as usize + 1).max(n);
    let nw_c = (max_wire(&c) as usize + 1).max(n);
    let batches = args.samples.div_ceil(64).max(4);
    let train_b = (batches * 3 / 4).max(2).min(batches - 1);
    let nbits = (batches * 64) as f64;

    let pred = {
        let p = parse_wire_list(&args.pred_wires, nw_g);
        if p.is_empty() { (0..nw_g).collect() } else { p }
    };
    let and_set = parse_wire_list(&args.and_wires, nw_g);
    let and_set = if args.and_wires.is_empty() { Vec::new() } else { and_set };
    let band = parse_wire_list(&args.band, nw_g);
    let band = if args.band.is_empty() { Vec::new() } else { band };
    let windows: Vec<usize> = args
        .census_windows
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    let i_idx = indices(c.len(), args.c_step.max(1));
    let j_idx = indices(g.len(), args.g_step.max(1));
    let (rows, cols) = (i_idx.len(), j_idx.len());
    let candidates = pred.len() + pred.len() * (pred.len().saturating_sub(1)) / 2;
    println!(
        "[stress_battery] c={} g={} n={} | rows={} cols={} samples={} (train {}) | \
         pred wires={} ({} cand) and-set={} band={}",
        c.len(), g.len(), n, rows, cols, batches * 64, train_b * 64,
        pred.len(), candidates, and_set.len(), band.len()
    );

    let mut rng = StdRng::seed_from_u64(args.seed);

    // Decoys.
    let decoys: Vec<Vec<XGate>> = if args.decoy.is_empty() {
        Vec::new()
    } else {
        args.decoy.split(',').map(|p| read_circuit(p.trim(), &args.c_format)).collect()
    };

    // Sample: C at rows, G at cols, decoys at their own fraction-matched rows,
    // all driven by the SAME x per batch.
    let mut cs: Vec<Vec<Vec<u64>>> = Vec::with_capacity(batches);
    let mut gs: Vec<Vec<Vec<u64>>> = Vec::with_capacity(batches);
    let mut ds: Vec<Vec<Vec<Vec<u64>>>> = vec![Vec::with_capacity(batches); decoys.len()];
    // decoy row grids matched to C's row fractions.
    let decoy_idx: Vec<Vec<usize>> = decoys
        .iter()
        .map(|d| {
            i_idx
                .iter()
                .map(|&i| ((i as f64 / c.len().max(1) as f64) * d.len() as f64).round() as usize)
                .map(|p| p.min(d.len()))
                .collect()
        })
        .collect();
    for _ in 0..batches {
        let x: Vec<u64> = (0..n).map(|_| rng.random::<u64>()).collect();
        let mut ic = vec![0u64; nw_c];
        let mut ig = vec![0u64; nw_g];
        ic[..n].copy_from_slice(&x);
        ig[..n].copy_from_slice(&x);
        cs.push(snapshots(&c, &ic, &i_idx));
        gs.push(snapshots(&g, &ig, &j_idx));
        for (di, d) in decoys.iter().enumerate() {
            let nw_d = (max_wire(d) as usize + 1).max(n);
            let mut idd = vec![0u64; nw_d];
            idd[..n].copy_from_slice(&x);
            ds[di].push(snapshots(d, &idd, &decoy_idx[di]));
        }
    }

    let s = Sampled { cs, gs, batches, rows, cols };

    // Noise floor: the LIFT (best − base) of the same max-over-candidates
    // search against an unrelated random target — i.e. the overfit inflation
    // a max over ~W^2/2 predictors buys by chance. Cells are lifts too, so a
    // prominence above this floor is real predictive information.
    let null_target: Vec<u64> = (0..batches).map(|_| rng.random::<u64>()).collect();
    let floor = {
        let mut acc = 0.0;
        let mut cnt = 0;
        for cj in (cols / 4..cols.saturating_sub(cols / 4)).step_by((cols / 8).max(1)) {
            let wires: Vec<Vec<u64>> =
                pred.iter().map(|&k| (0..batches).map(|b| s.gs[b][cj][k]).collect()).collect();
            let (best, _, base) = best_agreement(&null_target, &wires, &[], nbits);
            acc += best - base;
            cnt += 1;
        }
        acc / cnt.max(1) as f64
    };

    // Target sets.
    let all_bits: Vec<usize> = (0..n).collect();
    let bits: Vec<usize> = if args.target_bits == 0 || args.target_bits >= n {
        all_bits.clone()
    } else {
        let mut v = all_bits.clone();
        v.shuffle(&mut rng);
        v.truncate(args.target_bits);
        v.sort_unstable();
        v
    };
    let n_pairs = args.target_bits.max(1);
    let pairs: Vec<(usize, usize)> = (0..n_pairs)
        .map(|_| {
            let a = rng.random_range(0..n);
            let mut c2 = rng.random_range(0..n);
            while c2 == a {
                c2 = rng.random_range(0..n);
            }
            (a.min(c2), a.max(c2))
        })
        .collect();

    // Execution probe matrix: (target kind × feature) with the shared floor.
    let probe_kinds = [
        ("bit", TargetKind::Bit),
        ("par2", TargetKind::Par2),
        ("traj", TargetKind::Traj),
    ];
    let mut probes = Vec::new();
    for (tname, kind) in probe_kinds {
        let (r1, r2) =
            run_probe(&s, &s.cs, kind, &bits, &pairs, &pred, &and_set, nbits, floor, &mut rng);
        // Decoy null (bit probe only — parity/traj decoy alignment is noisier).
        let mut decoy_prom = Vec::new();
        if matches!(kind, TargetKind::Bit) {
            for di in 0..decoys.len() {
                let (dr, _) = run_probe(
                    &s, &ds[di], kind, &bits, &pairs, &pred, &and_set, nbits, floor, &mut rng,
                );
                decoy_prom.push(dr.prominence);
            }
        }
        let contrast = if decoy_prom.is_empty() {
            None
        } else {
            let mean_d = decoy_prom.iter().sum::<f64>() / decoy_prom.len() as f64;
            Some(r1.prominence - mean_d)
        };
        probes.push(ProbeReport {
            target: tname.to_string(),
            feature: "F1(xor-pair)".to_string(),
            ridge: r1,
            decoy_prominence: decoy_prom.clone(),
            contrast_vs_decoy: contrast,
        });
        if let Some(r2) = r2 {
            probes.push(ProbeReport {
                target: tname.to_string(),
                feature: format!("F2(xor+1AND/{})", and_set.len()),
                ridge: r2,
                decoy_prominence: Vec::new(),
                contrast_vs_decoy: None,
            });
        }
    }

    // Invariant miner over interior columns.
    let (c0, cc) = ((cols / 10).max(1), cols.saturating_sub((cols / 10).max(1)));
    let mut inv: Vec<usize> = Vec::new();
    for cj in c0..cc {
        inv.push(invariants_at(&s.gs, cj, nw_g, train_b, batches));
    }
    inv.sort_unstable();
    let inv_max = inv.last().copied().unwrap_or(0);
    let inv_med = inv.get(inv.len() / 2).copied().unwrap_or(0);

    let cen = census(&g, nw_g, &band, &windows);

    // -------- human-readable summary --------
    println!(
        "\n[stress_battery] floor={floor:.4}  (overfit inflation of the lift-over-base-rate; \
         a per-row peak above it is real predictive information)\n"
    );
    println!(
        "  prominence/peak = predictive LIFT over the target's base rate (0 = no info); \
         rho/perm-z = progress alignment of the peak\n"
    );
    println!(
        "  {:<6} {:<16} {:>8} {:>7} {:>7} {:>7} {:>9}  {}",
        "target", "feature", "promin", "peak", "rho", "perm-z", "contrast", "verdict"
    );
    for p in &probes {
        let contrast = p.contrast_vs_decoy.map(|c| format!("{c:+.4}")).unwrap_or_else(|| "-".into());
        // Two independent gates: is there exploitable info (prominence over the
        // overfit floor), and is it progress-aligned (perm-z on the peak).
        let has_info = p.ridge.prominence > 2.0 * floor.max(0.005);
        let aligned = p.ridge.perm_z >= 3.0 && p.ridge.rho.abs() >= 0.5;
        let verdict = match (has_info, aligned) {
            (true, true) => "ALIGNED-LEAK",
            (true, false) => "bias-only",
            (false, _) => "flat",
        };
        println!(
            "  {:<6} {:<16} {:>8.4} {:>7.4} {:>7.3} {:>7.2} {:>9}  {}",
            p.target, p.feature, p.ridge.prominence, p.ridge.peak, p.ridge.rho, p.ridge.perm_z, contrast, verdict
        );
    }
    println!(
        "\n  invariants (interior cols): median={inv_med} max={inv_max}  \
         (0 = no hidden affine relations; >0 = SAT-channel leak)"
    );
    println!(
        "  census: writes/wire {}/{}/{}  reads/wire {}/{}/{}  bimodality-gap={:.3}",
        cen.writes_min, cen.writes_med, cen.writes_max,
        cen.reads_min, cen.reads_med, cen.reads_max, cen.write_bimodality_gap
    );
    for (w, cnt) in &cen.unwritten_in_window {
        println!("    unwritten in a {w}-gate window: {cnt} wires");
    }
    if let (Some(a), Some(b)) = (cen.band_write_auc, cen.band_w1share_auc) {
        println!("    band localization AUC: write-count={a:.3}  width1-share={b:.3}  (0.5=hidden, 1.0=trivially found)");
    }

    let report = Report {
        c: args.c.clone(),
        g: args.g.clone(),
        n,
        c_gates: c.len(),
        g_gates: g.len(),
        g_wires: nw_g,
        rows,
        cols,
        samples: batches * 64,
        floor,
        probes,
        invariant_interior_max: inv_max,
        invariant_interior_median: inv_med,
        census: cen,
    };
    if !args.out.is_empty() {
        let path = format!("{}.json", args.out);
        let mut f = std::fs::File::create(&path).expect("create report");
        f.write_all(report.to_json().as_bytes()).expect("write report");
        println!("\n[stress_battery] wrote {path}");
    }
}

// ---------------------------------------------------------------------------
// Hand-written JSON (the tree has no serde_json; other bins do the same).
// ---------------------------------------------------------------------------

impl Ridge {
    fn to_json(&self) -> String {
        format!(
            "{{\"plate_mean\":{:.5},\"floor\":{:.5},\"prominence\":{:.5},\"peak\":{:.5},\
             \"rho\":{:.5},\"perm_z\":{:.4},\"rows\":{},\"cols\":{}}}",
            self.plate_mean, self.floor, self.prominence, self.peak, self.rho, self.perm_z,
            self.rows, self.cols
        )
    }
}

fn f64_list(v: &[f64]) -> String {
    let parts: Vec<String> = v.iter().map(|x| format!("{x:.5}")).collect();
    format!("[{}]", parts.join(","))
}

impl Census {
    fn to_json(&self) -> String {
        let uw: Vec<String> = self
            .unwritten_in_window
            .iter()
            .map(|(w, c)| format!("[{w},{c}]"))
            .collect();
        let opt = |o: Option<f64>| o.map(|x| format!("{x:.4}")).unwrap_or_else(|| "null".into());
        format!(
            "{{\"wires\":{},\"writes\":[{},{},{}],\"reads\":[{},{},{}],\
             \"write_bimodality_gap\":{:.4},\"unwritten_in_window\":[{}],\
             \"band_write_auc\":{},\"band_w1share_auc\":{}}}",
            self.wires, self.writes_min, self.writes_med, self.writes_max,
            self.reads_min, self.reads_med, self.reads_max,
            self.write_bimodality_gap, uw.join(","),
            opt(self.band_write_auc), opt(self.band_w1share_auc)
        )
    }
}

impl ProbeReport {
    fn to_json(&self) -> String {
        let contrast = self
            .contrast_vs_decoy
            .map(|c| format!("{c:.5}"))
            .unwrap_or_else(|| "null".into());
        format!(
            "{{\"target\":\"{}\",\"feature\":\"{}\",\"ridge\":{},\
             \"decoy_prominence\":{},\"contrast_vs_decoy\":{}}}",
            self.target, self.feature, self.ridge.to_json(),
            f64_list(&self.decoy_prominence), contrast
        )
    }
}

impl Report {
    fn to_json(&self) -> String {
        let probes: Vec<String> = self.probes.iter().map(|p| p.to_json()).collect();
        format!(
            "{{\"c\":\"{}\",\"g\":\"{}\",\"n\":{},\"c_gates\":{},\"g_gates\":{},\
             \"g_wires\":{},\"rows\":{},\"cols\":{},\"samples\":{},\"floor\":{:.5},\
             \"invariant_interior_max\":{},\"invariant_interior_median\":{},\
             \"probes\":[{}],\"census\":{}}}",
            self.c, self.g, self.n, self.c_gates, self.g_gates, self.g_wires,
            self.rows, self.cols, self.samples, self.floor,
            self.invariant_interior_max, self.invariant_interior_median,
            probes.join(","), self.census.to_json()
        )
    }
}
