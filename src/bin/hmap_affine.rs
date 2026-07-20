//! Affine-reconstruction heatmap between an original circuit C (n wires) and a
//! wider gadgetized version G (>= n wires) that computes C in an encoded form.
//!
//! H(i,j) = mean over C_i(x)'s n logical bits of the best-effort GF(2)-affine
//! reconstruction error of that bit from ALL of G_j(x,0,..,0)'s wires, measured
//! on a held-out sample split. Both sides are deterministic functions of the
//! shared input x (aux wires fixed at 0): x is injected on G's wires 0..n-1,
//! zeros elsewhere. Per target bit:
//!   * fit an affine map over training samples (GF(2) span membership);
//!   * inconsistent (bit is not an affine function of G_j) -> error 0.5;
//!   * consistent -> measured holdout bit-error (≈0 for a genuine affine
//!     relation; ~0.5 if the training fit was a spurious overfit).
//! Low H = C_i is linearly recoverable from G_j (leakage / progress alignment);
//! H ≈ 0.5 = hidden. The measure is invariant to the affine part of the gadget
//! encoding (CNOT/negation/share-XOR), which is exactly what dissolves the
//! n-vs-4n width mismatch that a raw Hamming comparison cannot handle. When C
//! and G coincide (identity embedding) it reduces to the ordinary prefix map.
//!
//! DEGREE (`--degree`): degree 1 is the affine predictor above (linear terms +
//! constant). Degree 2 additionally offers every pairwise product g_a·g_b of
//! G_j's wires as a regressor (a product of two lane-words is their bitwise
//! AND), so the fit can reconstruct C_i by a degree-2 GF(2) function of G_j —
//! it sees through any degree-2 re-encoding, not just the affine part. Degree d
//! would add all monomials of degree <= d. The predictor degree models a
//! feasible reconstruction adversary of bounded algebraic degree; a diagonal
//! that survives to higher degree is a stronger leak.
//!
//! COMPLEXITY: the regressor count is 1 + sum_{k=1..d} C(W, k) where W is the
//! wire count used for products (`--deg2-wires`, default all G wires). The GF(2)
//! span-membership solve needs MORE training samples than regressors to avoid
//! spurious consistency, and the basis costs O(regressors^2 * samples/64) words
//! per cell. So cost blows up as W^d: full degree 2 at W=512 is ~1.3e5
//! regressors (needs >1.3e5 samples, ~TB basis) — intractable; restrict
//! `--deg2-wires` to make degree 2 a LOWER BOUND on degree-2 leakage (products
//! among excluded wires unseen), or run at small n. Degree >=3 is only
//! feasible at small n or small W. See docs/HMAP_AFFINE.* for the reasoning and
//! the higher-degree extension.
//!
//! Output: <out>.bin (row-major f32, rows = C prefixes, cols = G prefixes) plus
//! <out>.meta.json (index vectors + mean/std), same format as `hmap`.

use clap::Parser;
use local_mixing::postmix::format::{read_g57_file, read_mpmct};
use local_mixing::postmix::xgate::{XGate, max_wire};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::Rng;
use std::io::Write;

#[derive(Parser, Debug)]
#[command(name = "hmap_affine")]
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
    /// Logical input width n: x is injected on G wires 0..n-1, zeros elsewhere;
    /// C_i's target bits are wires 0..n-1.
    #[arg(long)]
    n: usize,
    /// Predictor degree: 1 = affine (linear + const); 2 = also degree-2
    /// products of G_j's wires (bitwise AND of lanes). Degree 2 adds
    /// C(deg2_wires,2) regressors, so keep deg2_wires small enough that the
    /// total stays well under the training-sample count.
    #[arg(long, default_value_t = 1)]
    degree: usize,
    /// For --degree 2: form products only among wires 0..deg2_wires (0 = all G
    /// wires). Linear terms always cover all wires. A subset makes degree-2 a
    /// LOWER BOUND on degree-2 leakage (products among excluded wires unseen).
    #[arg(long, default_value_t = 0)]
    deg2_wires: usize,
    /// Prefix strides
    #[arg(long, default_value_t = 10)]
    c_step: usize,
    #[arg(long, default_value_t = 50)]
    g_step: usize,
    /// Total sample batches (64 samples each) and the number used for training;
    /// the rest are the held-out set. train_batches must exceed (G wires + 1)/64
    /// comfortably so a non-affine bit cannot be fit by coincidence.
    #[arg(long, default_value_t = 96)]
    batches: usize,
    #[arg(long, default_value_t = 72)]
    train_batches: usize,
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

// ---- GF(2) bit-vector helpers over a Vec<u64> ----
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

// A reduced row: sample-space vector paired with the regressor coefficients that
// produced it (both GF(2), packed).
struct Row {
    samp: Vec<u64>,
    coef: Vec<u64>,
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
    assert!(nw_g >= n, "G has fewer wires than n");
    let b = args.batches.max(2);
    let tb = args.train_batches.clamp(1, b - 1);

    // Regressor descriptors: all linear wires, the constant, then (degree 2)
    // products among wires 0..dw. Both the basis build and the holdout eval
    // iterate this list, so the two stay consistent.
    #[derive(Clone, Copy)]
    enum Reg {
        Lin(usize),
        Const,
        Pair(usize, usize),
    }
    let mut regs: Vec<Reg> = (0..nw_g).map(Reg::Lin).collect();
    regs.push(Reg::Const);
    if args.degree >= 2 {
        let dw = if args.deg2_wires == 0 { nw_g } else { args.deg2_wires.min(nw_g) };
        for a in 0..dw {
            for c2 in (a + 1)..dw {
                regs.push(Reg::Pair(a, c2));
            }
        }
    }
    let n_reg = regs.len();
    assert!(
        tb * 64 > n_reg + 64,
        "train samples ({}) must exceed regressors ({}) with margin; lower --deg2-wires or raise --batches/--train-batches",
        tb * 64,
        n_reg
    );
    let coef_words = n_reg.div_ceil(64);
    // Train/holdout words for regressor r at G-column cj.
    let reg_train = |r: Reg, cj: usize, gs: &Vec<Vec<Vec<u64>>>| -> Vec<u64> {
        match r {
            Reg::Lin(k) => (0..tb).map(|batch| gs[batch][cj][k]).collect(),
            Reg::Const => vec![!0u64; tb],
            Reg::Pair(a, c2) => (0..tb).map(|batch| gs[batch][cj][a] & gs[batch][cj][c2]).collect(),
        }
    };
    let reg_ho_word = |r: Reg, cj: usize, batch: usize, gs: &Vec<Vec<Vec<u64>>>| -> u64 {
        match r {
            Reg::Lin(k) => gs[batch][cj][k],
            Reg::Const => !0u64,
            Reg::Pair(a, c2) => gs[batch][cj][a] & gs[batch][cj][c2],
        }
    };

    let i_idx = indices(c.len(), args.c_step.max(1));
    let j_idx = indices(g.len(), args.g_step.max(1));
    let (rows, cols) = (i_idx.len(), j_idx.len());
    println!(
        "[hmap_affine] c={} gates ({}w), g={} gates ({}w), n={}; degree={} regressors={}; rows={} cols={}, samples={} (train {})",
        c.len(), nw_c, g.len(), nw_g, n, args.degree, n_reg, rows, cols, b * 64, tb * 64
    );

    // Per batch: snapshot C at each i (targets) and G at each j (regressors),
    // driven by the SAME random x on wires 0..n-1.
    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut cs: Vec<Vec<Vec<u64>>> = Vec::with_capacity(b); // [batch][i][wire]
    let mut gs: Vec<Vec<Vec<u64>>> = Vec::with_capacity(b); // [batch][j][wire]
    for _ in 0..b {
        let x: Vec<u64> = (0..n).map(|_| rng.random::<u64>()).collect();
        let mut ic = vec![0u64; nw_c];
        let mut ig = vec![0u64; nw_g];
        ic[..n].copy_from_slice(&x);
        ig[..n].copy_from_slice(&x);
        cs.push(snapshots(&c, &ic, &i_idx));
        gs.push(snapshots(&g, &ig, &j_idx));
    }

    let ho = b - tb; // holdout batches
    let mut mat = vec![0f32; rows * cols];

    // Scratch reused per cell.
    for (ri, &_i) in i_idx.iter().enumerate() {
        for (cj, &_j) in j_idx.iter().enumerate() {
            // Build a GF(2) basis of G_j's regressor columns over TRAIN samples.
            let mut basis: Vec<(usize, Row)> = Vec::new();
            let mut add_col = |train: Vec<u64>, reg_idx: usize, basis: &mut Vec<(usize, Row)>| {
                let mut samp = train;
                let mut coef = vec![0u64; coef_words];
                coef[reg_idx / 64] |= 1u64 << (reg_idx % 64);
                for (piv, brow) in basis.iter() {
                    if bit(&samp, *piv) {
                        xor_into(&mut samp, &brow.samp);
                        xor_into(&mut coef, &brow.coef);
                    }
                }
                if let Some(p) = first_set(&samp) {
                    basis.push((p, Row { samp, coef }));
                }
            };
            for (ridx, &r) in regs.iter().enumerate() {
                add_col(reg_train(r, cj, &gs), ridx, &mut basis);
            }

            // Reconstruct each target bit; accumulate error.
            let mut err_sum = 0f64;
            for t in 0..n {
                let mut samp: Vec<u64> = (0..tb).map(|batch| cs[batch][ri][t]).collect();
                let mut coef = vec![0u64; coef_words];
                for (piv, brow) in basis.iter() {
                    if bit(&samp, *piv) {
                        xor_into(&mut samp, &brow.samp);
                        xor_into(&mut coef, &brow.coef);
                    }
                }
                if first_set(&samp).is_some() {
                    err_sum += 0.5; // inconsistent: not a degree-<=d function of G_j
                    continue;
                }
                // Consistent: evaluate `coef` on the holdout batches.
                let mut errbits = 0u64;
                for batch in tb..b {
                    let mut acc = 0u64;
                    for (ridx, &r) in regs.iter().enumerate() {
                        if (coef[ridx / 64] >> (ridx % 64)) & 1 == 1 {
                            acc ^= reg_ho_word(r, cj, batch, &gs);
                        }
                    }
                    errbits += (acc ^ cs[batch][ri][t]).count_ones() as u64;
                }
                err_sum += errbits as f64 / (ho as f64 * 64.0);
            }
            mat[ri * cols + cj] = (err_sum / n as f64) as f32;
        }
    }

    let (mut sum, mut sumsq) = (0f64, 0f64);
    for &m in &mat {
        sum += m as f64;
        sumsq += (m as f64) * (m as f64);
    }
    let nn = (rows * cols) as f64;
    let mu = sum / nn;
    let sigma = (sumsq / nn - mu * mu).max(0.0).sqrt();
    println!("[hmap_affine] H mean={:.4} std={:.4} (0=affine-recoverable, 0.5=hidden)", mu, sigma);

    let bin = format!("{}.bin", args.out);
    let mut f = std::fs::File::create(&bin).expect("create bin");
    let bytes: Vec<u8> = mat.iter().flat_map(|v| v.to_le_bytes()).collect();
    f.write_all(&bytes).expect("write bin");
    let meta = format!(
        "{{\"rows\":{},\"cols\":{},\"n\":{},\"mu\":{:.6},\"sigma\":{:.6},\"i_idx\":[{}],\"j_idx\":[{}]}}",
        rows, cols, n, mu, sigma,
        i_idx.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","),
        j_idx.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","),
    );
    std::fs::write(format!("{}.meta.json", args.out), meta).expect("write meta");
    println!("[hmap_affine] wrote {} ({}x{} f32) and {}.meta.json", bin, rows, cols, args.out);
}
