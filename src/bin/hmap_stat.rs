//! Statistical companion to `hmap_affine`: how well can a snapshot of the
//! gadget PREDICT the original circuit's state, as opposed to reconstruct it
//! exactly?
//!
//! `hmap_affine` asks a yes/no question — "is C_i's bit an exact GF(2)
//! function of G_j's wires?" — and scores every inexact cell at 0.5. That
//! makes it blind to bias: an adversary predicting the bit 78% of the time is
//! scored identically to one predicting it 50% of the time. The product-share
//! encoding's term count `k` lives entirely inside that blind spot, because
//! each extra mask term changes only how often the naive predictor
//! `c0 XOR c1` is wrong, never whether an exact relation exists. Piling-up
//! over mask terms firing with probability 2^-deg predicts, for the shipped
//! plans:
//!
//!   [2,2,3,3] 57%   [2,3,3] 64%   [3,3] 78%   [3] 87.5%
//!
//! and `hmap_affine` reports ~0.5 ("hidden") for all four. This tool measures
//! the left-hand side directly.
//!
//! MEASURE. Per cell (C-prefix i, G-prefix j) and per target bit, search a
//! family of cheap predictors of that bit from G_j's wires and keep the best
//! AGREEMENT (fraction of samples predicted correctly, always >= 0.5 since a
//! predictor and its complement are both available):
//!   * every single wire,
//!   * every XOR of two wires — the family that contains `c0 XOR c1`, the
//!     carrier pair, which is exactly the predictor the mask terms perturb.
//! The cell's score is the mean best-agreement over the target bits. Every
//! number this produces is a LOWER BOUND on what an adversary can do: the
//! predictor family stops at two wires and the prefix grid is coarse, so no
//! column need land exactly on the matched snapshot.
//!
//! NULL. A max over ~W^2/2 candidates is biased upward by chance, so the same
//! search is run against a random balanced target and reported as the noise
//! floor. Read the plate as "agreement minus floor"; the floor is ~0.5 +
//! 0.5*sqrt(2 ln(#candidates)/N), so keep `--samples` well above a few
//! thousand and compare like with like.
//!
//! PORTS. Both axes have unencoded ends: row 0 is C's input state, column 0 is
//! G before the encoding is ramped in (its low wires literally hold x), and
//! likewise at the far end. Cell (0,0) is therefore 1.0 in EVERY build and any
//! statistic that includes it says nothing. The summary below reports the
//! interior separately for that reason, and
//! `reports/band_hardening_20260725/stat_readout.py` is the reader — NOT
//! `plot_hmap_ridge.py`, which assumes hmap_affine's inverted convention
//! (there low is leaky; here high is) and would trace nonsense on this plate.
//!
//! COST. W^2/2 candidates x samples/64 words per (cell, bit). Default grids
//! are therefore much coarser than `hmap_affine`'s: prefer `--c-step`/
//! `--g-step` giving ~16 rows/cols and `--target-bits 16`.

use clap::Parser;
use local_mixing::postmix::format::{read_g57_file, read_mpmct};
use local_mixing::postmix::xgate::{XGate, max_wire};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::Rng;
use std::io::Write;

#[derive(Parser, Debug)]
#[command(name = "hmap_stat")]
struct Args {
    /// Original circuit C
    #[arg(long)]
    c: String,
    /// Gadgetized circuit G
    #[arg(long)]
    g: String,
    #[arg(long, default_value = "g57")]
    c_format: String,
    #[arg(long, default_value = "mpmct1")]
    g_format: String,
    /// Logical input width n: x is injected on wires 0..n-1 of both, zeros
    /// elsewhere; C_i's target bits are wires 0..n-1.
    #[arg(long)]
    n: usize,
    /// Prefix strides (coarse by default — this measure is O(W^2) per cell).
    #[arg(long, default_value_t = 200)]
    c_step: usize,
    #[arg(long, default_value_t = 20000)]
    g_step: usize,
    /// Samples, rounded up to a multiple of 64.
    #[arg(long, default_value_t = 4096)]
    samples: usize,
    /// Target bits per cell: a random subset of C's n state bits (0 = all).
    #[arg(long, default_value_t = 16)]
    target_bits: usize,
    /// Restrict the predictor's wire set, e.g. "0-511" (empty = all G wires).
    #[arg(long, default_value = "")]
    wire_list: String,
    /// Give the adversary ONE degree-2 term on top of the best XOR predictor:
    /// `(w_p ^ a) & (w_q ^ b)` over this wire set, e.g. the band "512-567"
    /// (empty = XOR predictors only). This is the deg-2-capable adversary, and
    /// it is the one that decides whether a degree-2 mask term is worth its
    /// gates: such a term is exactly what one AND can cancel.
    #[arg(long, default_value = "")]
    and_wires: String,
    #[arg(long, default_value_t = 12345)]
    seed: u64,
    #[arg(long)]
    out: String,
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

// Snapshot lane-state (bit l = sample l) at each prefix length in `idx`.
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
            let lo: usize = lo.parse().expect("bad --wire-list range");
            let hi: usize = hi.parse().expect("bad --wire-list range");
            v.extend(lo..=hi.min(nw - 1));
        } else if !part.is_empty() {
            v.push(part.parse().expect("bad --wire-list entry"));
        }
    }
    v.sort_unstable();
    v.dedup();
    assert!(
        !v.is_empty() && *v.last().unwrap() < nw,
        "--wire-list selects no wire in 0..{nw}"
    );
    v
}

/// Best agreement of `target` with any single wire or XOR of two wires, over
/// packed samples. Agreement is symmetric under complementing the predictor,
/// so the score is max(a, 1-a) and always >= 0.5. Returns that score and, when
/// `and_wires` is non-empty, the score after greedily adding ONE degree-2 term
/// `(w_p ^ a) & (w_q ^ b)` to the best XOR predictor found.
fn best_agreement(
    target: &[u64],
    wires: &[Vec<u64>],
    and_wires: &[Vec<u64>],
    nbits: f64,
) -> (f64, f64) {
    let words = target.len();
    let agreement = |mismatch: u32| {
        let a = mismatch as f64 / nbits;
        if a < 0.5 { 1.0 - a } else { a }
    };
    // Score every candidate by popcount alone — no allocation in the inner
    // loops — and remember only WHICH one won, as a pair of wire indices
    // (usize::MAX meaning "no wire", i.e. the constant predictor).
    let mut best = 0f64;
    let mut best_pair = (usize::MAX, usize::MAX);
    // The constant predictor (the target's own bias).
    let m: u32 = target.iter().map(|w| w.count_ones()).sum();
    best = best.max(agreement(m));
    // Pre-XOR each wire with the target once, then pairs are one more XOR.
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
        return (best, best);
    }
    // Materialize the winner's residual once.
    let mut best_res: Vec<u64> = target.to_vec();
    for idx in [best_pair.0, best_pair.1] {
        if idx != usize::MAX {
            for w in 0..words {
                best_res[w] ^= wires[idx][w];
            }
        }
    }
    // One degree-2 term on top, greedily: all pairs from the AND set, all four
    // literal polarities. Greedy (the XOR part is not re-optimized), so this
    // too is a lower bound on the deg-2-capable adversary.
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
    (best, aug)
}

fn main() {
    let args = Args::parse();
    let c: Vec<XGate> = match args.c_format.as_str() {
        "g57" => read_g57_file(&args.c).expect("read c (g57)"),
        "mpmct1" => read_mpmct(&args.c).expect("read c (mpmct1)").0,
        o => panic!("unknown --c-format {o}"),
    };
    let g: Vec<XGate> = match args.g_format.as_str() {
        "g57" => read_g57_file(&args.g).expect("read g (g57)"),
        "mpmct1" => read_mpmct(&args.g).expect("read g (mpmct1)").0,
        o => panic!("unknown --g-format {o}"),
    };
    let n = args.n;
    let nw_c = (max_wire(&c) as usize + 1).max(n);
    let nw_g = (max_wire(&g) as usize + 1).max(n);
    let batches = args.samples.div_ceil(64).max(2);
    let nbits = (batches * 64) as f64;
    let pred_wires = parse_wire_list(&args.wire_list, nw_g);
    let candidates = pred_wires.len() + pred_wires.len() * (pred_wires.len() - 1) / 2;
    let and_set: Vec<usize> = if args.and_wires.is_empty() {
        Vec::new()
    } else {
        parse_wire_list(&args.and_wires, nw_g)
    };

    let i_idx = indices(c.len(), args.c_step.max(1));
    let j_idx = indices(g.len(), args.g_step.max(1));
    let (rows, cols) = (i_idx.len(), j_idx.len());
    let mut rng = StdRng::seed_from_u64(args.seed);
    let targets: Vec<usize> = if args.target_bits == 0 || args.target_bits >= n {
        (0..n).collect()
    } else {
        let mut all: Vec<usize> = (0..n).collect();
        for i in 0..args.target_bits {
            let j = i + rng.random_range(0..(n - i));
            all.swap(i, j);
        }
        all.truncate(args.target_bits);
        all.sort_unstable();
        all
    };
    println!(
        "[hmap_stat] c={} gates ({}w), g={} gates ({}w), n={}; rows={} cols={}, samples={}, \
         target bits={}, predictor wires={} ({} candidates)",
        c.len(), nw_c, g.len(), nw_g, n, rows, cols, batches * 64,
        targets.len(), pred_wires.len(), candidates
    );

    // Per batch: snapshot C at each i (targets) and G at each j (predictors),
    // driven by the SAME random x on wires 0..n-1.
    let mut cs: Vec<Vec<Vec<u64>>> = Vec::with_capacity(batches);
    let mut gs: Vec<Vec<Vec<u64>>> = Vec::with_capacity(batches);
    for _ in 0..batches {
        let x: Vec<u64> = (0..n).map(|_| rng.random::<u64>()).collect();
        let mut ic = vec![0u64; nw_c];
        let mut ig = vec![0u64; nw_g];
        ic[..n].copy_from_slice(&x);
        ig[..n].copy_from_slice(&x);
        cs.push(snapshots(&c, &ic, &i_idx));
        gs.push(snapshots(&g, &ig, &j_idx));
    }
    // The noise floor: the same search against a target with no relation to
    // anything, so the reader can tell a real bias from the max-over-
    // candidates artefact.
    let null_target: Vec<u64> = (0..batches).map(|_| rng.random::<u64>()).collect();

    let mut mat = vec![0f32; rows * cols];      // best incl. the optional AND
    let mut xor_mat = vec![0f32; rows * cols];  // XOR predictors only
    let mut null_sum = 0f64;
    let mut null_cells = 0usize;
    for (ri, _) in i_idx.iter().enumerate() {
        for (cj, _) in j_idx.iter().enumerate() {
            // Repack G_j's predictor wires as [wire][batch].
            let wires: Vec<Vec<u64>> = pred_wires
                .iter()
                .map(|&k| (0..batches).map(|b| gs[b][cj][k]).collect())
                .collect();
            let ands: Vec<Vec<u64>> = and_set
                .iter()
                .map(|&k| (0..batches).map(|b| gs[b][cj][k]).collect())
                .collect();
            let (mut sum, mut sum_aug) = (0f64, 0f64);
            for &t in &targets {
                let target: Vec<u64> = (0..batches).map(|b| cs[b][ri][t]).collect();
                let (xor_only, augmented) = best_agreement(&target, &wires, &ands, nbits);
                sum += xor_only;
                sum_aug += augmented;
            }
            xor_mat[ri * cols + cj] = (sum / targets.len() as f64) as f32;
            mat[ri * cols + cj] = (sum_aug / targets.len() as f64) as f32;
            // One null draw per column is plenty and keeps the cost down.
            if ri == 0 {
                null_sum += best_agreement(&null_target, &wires, &ands, nbits).1;
                null_cells += 1;
            }
        }
        println!("[hmap_stat] row {}/{} done", ri + 1, rows);
    }

    let (mut sum, mut sumsq) = (0f64, 0f64);
    for &m in &mat {
        sum += m as f64;
        sumsq += (m as f64) * (m as f64);
    }
    let nn = (rows * cols) as f64;
    let mu = sum / nn;
    let sigma = (sumsq / nn - mu * mu).max(0.0).sqrt();
    let floor = null_sum / null_cells.max(1) as f64;
    // Interior statistics: the ports are unencoded on BOTH axes, and cell
    // (0,0) reads 1.0 in every build, so a whole-plate peak is meaningless.
    let (r0, c0) = ((rows / 10).max(1), (cols / 10).max(1));
    let mut inner: Vec<f64> = Vec::new();
    for ri in r0..rows.saturating_sub(r0) {
        let mut row_best = 0f64;
        for cj in c0..cols.saturating_sub(c0) {
            row_best = row_best.max(mat[ri * cols + cj] as f64);
        }
        inner.push(row_best);
    }
    inner.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = inner.get(inner.len() / 2).copied().unwrap_or(f64::NAN);
    let inner_peak = inner.last().copied().unwrap_or(f64::NAN);
    println!(
        "[hmap_stat] plate mean={mu:.4} std={sigma:.4} | null floor={floor:.4} | \
         INTERIOR per-row best agreement: median={median:.4} peak={inner_peak:.4} \
         (excess over floor {:+.4})",
        median - floor
    );
    if !and_set.is_empty() {
        // What the one AND term bought: the same interior statistic without it.
        let mut xor_inner: Vec<f64> = Vec::new();
        for ri in r0..rows.saturating_sub(r0) {
            let mut row_best = 0f64;
            for cj in c0..cols.saturating_sub(c0) {
                row_best = row_best.max(xor_mat[ri * cols + cj] as f64);
            }
            xor_inner.push(row_best);
        }
        xor_inner.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let xor_median = xor_inner.get(xor_inner.len() / 2).copied().unwrap_or(f64::NAN);
        println!(
            "[hmap_stat] XOR-only interior median={xor_median:.4}; one deg-2 term over \
             {} wires raises it to {median:.4} ({:+.4})",
            and_set.len(),
            median - xor_median
        );
    }

    let bin = format!("{}.bin", args.out);
    let mut f = std::fs::File::create(&bin).expect("create bin");
    let bytes: Vec<u8> = mat.iter().flat_map(|v| v.to_le_bytes()).collect();
    f.write_all(&bytes).expect("write bin");
    let meta = format!(
        "{{\"rows\":{},\"cols\":{},\"n\":{},\"mu\":{:.6},\"sigma\":{:.6},\"floor\":{:.6},\
         \"i_idx\":[{}],\"j_idx\":[{}]}}",
        rows, cols, n, mu, sigma, floor,
        i_idx.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","),
        j_idx.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","),
    );
    std::fs::write(format!("{}.meta.json", args.out), meta).expect("write meta");
    println!("[hmap_stat] wrote {bin} ({rows}x{cols} f32) and {}.meta.json", args.out);
}
