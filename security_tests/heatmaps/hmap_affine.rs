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
use local_mixing::circuit::formats::{read_g57_file, read_mpmct};
use local_mixing::circuit::xgate::{XGate, max_wire};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
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
    /// For --degree 2: explicit product-wire set as ranges, e.g. "0-31,256-287"
    /// (inclusive). Overrides --deg2-wires. Use to straddle blocks (data+aux)
    /// instead of the prefix rule — the encoding pairs shares across blocks, so
    /// a prefix-only set can miss every informative product.
    #[arg(long, default_value = "")]
    deg2_wire_list: String,
    /// TSV file of hot segments (wire<TAB>start_gate<TAB>end_gate, '#'
    /// comments) of G, e.g. from qc_scan --hot-out. At each G column whose
    /// cut falls inside a listed segment's live interval (start < cut <=
    /// end), that wire is EXCLUDED from the regressor set, so the map
    /// measures reconstruction with the identified hot carriers zeroed out.
    #[arg(long, default_value = "")]
    exclude_segments: String,
    /// Score only the ACTIVE wire per row: the target of C's gate i-1, the
    /// one value born at prefix i. Every other wire's value was scored at the
    /// row where it was written, so the all-wires default re-scores each
    /// value across its whole segment (length-biased row correlation); this
    /// scores each original value exactly once. Row 0 (inputs) is skipped
    /// (cells 0.5). Per-value analogue of the QC hot-segment statistic.
    #[arg(long, default_value_t = false)]
    active_wire: bool,
    /// Prefix strides
    #[arg(long, default_value_t = 10)]
    c_step: usize,
    #[arg(long, default_value_t = 50)]
    g_step: usize,
    /// Explicit G cut positions, one gate index per line (0 = before gate 0).
    /// Overrides --g-step: use for packed/windowed artifacts where the only
    /// delivered cuts are the window boundaries.
    #[arg(long, default_value = "")]
    g_cuts: String,
    /// Total sample batches (64 samples each) and the number used for training;
    /// the rest are the held-out set. train_batches must exceed (G wires + 1)/64
    /// comfortably so a non-affine bit cannot be fit by coincidence.
    #[arg(long, default_value_t = 96)]
    batches: usize,
    #[arg(long, default_value_t = 72)]
    train_batches: usize,
    #[arg(long, default_value_t = 12345)]
    seed: u64,
    /// Restrict the C-prefix (row) range to [c_from, c_to] gate indices;
    /// 0/0 = the whole circuit. With a small --c-step this gives a dense
    /// LOCAL plate instead of a coarse global one, which is what resolving a
    /// narrow feature needs.
    #[arg(long, default_value_t = 0)]
    c_from: usize,
    #[arg(long, default_value_t = 0)]
    c_to: usize,
    /// Same for the G-prefix (column) range.
    #[arg(long, default_value_t = 0)]
    g_from: usize,
    #[arg(long, default_value_t = 0)]
    g_to: usize,
    /// Dump the SUPPORT of the fitted relation for the N leakiest interior
    /// cells: how many wires the reconstruction actually uses, and which. A
    /// leak is only interpretable once you know what it reads — e.g. whether
    /// it runs through the wires that carry mask sources.
    #[arg(long, default_value_t = 0)]
    dump_best: usize,
    /// EXPOSURE SCAN (RC's refinement of --active-wire): for every value
    /// born in C, the minimum support t* of an EXACT affine predictor from
    /// any sampled G cut — t* = number of distinct nonconstant G wire
    /// values read (complement pairs collapse, constants free; exact over
    /// ALL samples). Per cut: dedup wire columns, GF(2) basis with
    /// coefficient tracking, dependent columns become nullspace vectors,
    /// and the per-row particular solution is minimized over the nullspace
    /// coset (exact for nullity <= 14, else upper bound, counted). Writes
    /// <out>.exposure.tsv (gate, wire, t*; -1 = invisible) and prints the
    /// t* histogram. Use with --c-step 1; skips the matrix outputs.
    #[arg(long, default_value_t = false)]
    exposure: bool,
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
        Triple(usize, usize, usize),
    }
    let mut regs: Vec<Reg> = (0..nw_g).map(Reg::Lin).collect();
    regs.push(Reg::Const);
    if args.degree >= 2 {
        let wires: Vec<usize> = if !args.deg2_wire_list.is_empty() {
            let mut v = Vec::new();
            for part in args.deg2_wire_list.split(',') {
                let part = part.trim();
                if let Some((lo, hi)) = part.split_once('-') {
                    let lo: usize = lo.parse().expect("bad --deg2-wire-list range");
                    let hi: usize = hi.parse().expect("bad --deg2-wire-list range");
                    v.extend(lo..=hi.min(nw_g - 1));
                } else if !part.is_empty() {
                    v.push(part.parse().expect("bad --deg2-wire-list entry"));
                }
            }
            v.sort_unstable();
            v.dedup();
            v
        } else {
            let dw = if args.deg2_wires == 0 {
                nw_g
            } else {
                args.deg2_wires.min(nw_g)
            };
            (0..dw).collect()
        };
        let mut pairs = 0usize;
        for (i, &a) in wires.iter().enumerate() {
            for &c2 in &wires[i + 1..] {
                regs.push(Reg::Pair(a, c2));
                pairs += 1;
            }
        }
        // Degree 3: all triples over the SAME wire set (the `--deg2-wire-list`
        // doubles as the product-wire set for every degree >= 2). Targets the
        // band where product-share masks live, so a degree-3 adversary can form
        // the mask monomials without paying C(all_wires, 3).
        let mut triples = 0usize;
        if args.degree >= 3 {
            for (i, &a) in wires.iter().enumerate() {
                for (j, &b2) in wires[i + 1..].iter().enumerate() {
                    for &c3 in &wires[i + 1 + j + 1..] {
                        regs.push(Reg::Triple(a, b2, c3));
                        triples += 1;
                    }
                }
            }
        }
        println!(
            "[hmap_affine] product wires: {} ({} pairs, {} triples)",
            wires.len(),
            pairs,
            triples
        );
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
            Reg::Pair(a, c2) => (0..tb)
                .map(|batch| gs[batch][cj][a] & gs[batch][cj][c2])
                .collect(),
            Reg::Triple(a, b2, c3) => (0..tb)
                .map(|batch| gs[batch][cj][a] & gs[batch][cj][b2] & gs[batch][cj][c3])
                .collect(),
        }
    };
    let reg_ho_word = |r: Reg, cj: usize, batch: usize, gs: &Vec<Vec<Vec<u64>>>| -> u64 {
        match r {
            Reg::Lin(k) => gs[batch][cj][k],
            Reg::Const => !0u64,
            Reg::Pair(a, c2) => gs[batch][cj][a] & gs[batch][cj][c2],
            Reg::Triple(a, b2, c3) => gs[batch][cj][a] & gs[batch][cj][b2] & gs[batch][cj][c3],
        }
    };

    let clip = |v: Vec<usize>, from: usize, to: usize| -> Vec<usize> {
        if from == 0 && to == 0 {
            return v;
        }
        let hi = if to == 0 { usize::MAX } else { to };
        v.into_iter().filter(|&x| x >= from && x <= hi).collect()
    };
    let i_idx = clip(indices(c.len(), args.c_step.max(1)), args.c_from, args.c_to);
    let j_idx = if args.g_cuts.is_empty() {
        clip(indices(g.len(), args.g_step.max(1)), args.g_from, args.g_to)
    } else {
        let mut cuts: Vec<usize> = std::fs::read_to_string(&args.g_cuts)
            .expect("read --g-cuts file")
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.trim().parse::<usize>().expect("cut index"))
            .filter(|&x| x <= g.len())
            .collect();
        cuts.sort_unstable();
        cuts.dedup();
        clip(cuts, args.g_from, args.g_to)
    };
    assert!(
        !i_idx.is_empty() && !j_idx.is_empty(),
        "empty plate after --c-from/--c-to/--g-from/--g-to clip"
    );
    let (rows, cols) = (i_idx.len(), j_idx.len());
    println!(
        "[hmap_affine] c={} gates ({}w), g={} gates ({}w), n={}; degree={} regressors={}; rows={} cols={}, samples={} (train {})",
        c.len(),
        nw_c,
        g.len(),
        nw_g,
        n,
        args.degree,
        n_reg,
        rows,
        cols,
        b * 64,
        tb * 64
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

    if args.exposure {
        assert!(args.degree == 1, "--exposure is affine-only (degree 1)");
        let words = b;
        let allzero = |v: &[u64]| v.iter().all(|&w| w == 0);
        // rows = born values of C (active-wire semantics)
        let mut rowinfo: Vec<(usize, usize, usize)> = Vec::new(); // (ri, gate idx, wire)
        for (ri, &i) in i_idx.iter().enumerate() {
            if i == 0 {
                continue;
            }
            let t = c[i - 1].target as usize;
            if t < n {
                rowinfo.push((ri, i - 1, t));
            }
        }
        // SPAN_INTERSECT=<path>: per G cut, the dimension of the intersection
        // of span{C born values, 1} and span{G wire values at the cut, 1},
        // minus the shared constant -- i.e. how many independent NONZERO
        // affine combinations of C values are also affine combinations of
        // the cut's wire values. dim(U∩W) = dim W - rank(W reduced mod U).
        if let Ok(spath) = std::env::var("SPAN_INTERSECT") {
            let words = b;
            let lead = |v: &[u64]| -> Option<usize> {
                v.iter()
                    .enumerate()
                    .find_map(|(wi, &w)| (w != 0).then(|| wi * 64 + w.trailing_zeros() as usize))
            };
            let reduce = |v: &mut Vec<u64>, basis: &[(usize, Vec<u64>)]| {
                for (p, bv) in basis {
                    if (v[p / 64] >> (p % 64)) & 1 == 1 {
                        for (a, x) in v.iter_mut().zip(bv) {
                            *a ^= x;
                        }
                    }
                }
            };
            let mut ubasis: Vec<(usize, Vec<u64>)> = Vec::new();
            let mut insert = |mut v: Vec<u64>, basis: &mut Vec<(usize, Vec<u64>)>| -> bool {
                reduce(&mut v, basis);
                match lead(&v) {
                    Some(p) => {
                        basis.push((p, v));
                        true
                    }
                    None => false,
                }
            };
            // I = span{inputs, 1}; U = span{C values} + I. The EXCESS
            // dim((U)∩W) - dim(I∩W) counts combinations that reveal C-value
            // information beyond what the inputs already explain.
            let ri0 = i_idx
                .iter()
                .position(|&i| i == 0)
                .expect("c-step snapshot 0");
            let mut ibasis: Vec<(usize, Vec<u64>)> = Vec::new();
            insert(vec![!0u64; words], &mut ibasis);
            for k in 0..n {
                let v: Vec<u64> = (0..b).map(|batch| cs[batch][ri0][k]).collect();
                insert(v, &mut ibasis);
            }
            // SPAN_OUTPUTS_TRIVIAL=1: also count C's outputs (last writer per
            // wire) as trivial, so only non-output combinations remain.
            if std::env::var("SPAN_OUTPUTS_TRIVIAL").is_ok_and(|v| v != "0") {
                let mut last: std::collections::HashMap<usize, usize> =
                    std::collections::HashMap::new();
                for (k, &(ri, gi, t)) in rowinfo.iter().enumerate() {
                    let _ = ri;
                    let e = last.entry(t).or_insert(k);
                    if rowinfo[*e].1 < gi {
                        *e = k;
                    }
                }
                for &k in last.values() {
                    let (ri, _, t) = rowinfo[k];
                    let v: Vec<u64> = (0..b).map(|batch| cs[batch][ri][t]).collect();
                    insert(v, &mut ibasis);
                }
            }
            ubasis = ibasis.clone();
            for &(ri, _, t) in &rowinfo {
                let v: Vec<u64> = (0..b).map(|batch| cs[batch][ri][t]).collect();
                insert(v, &mut ubasis);
            }
            let dim_u = ubasis.len();
            use std::io::Write as _;
            let mut f = std::fs::File::create(&spath).expect("write span tsv");
            writeln!(f, "# cut\tdimW\texcess intersection beyond inputs").unwrap();
            let mut hist: std::collections::BTreeMap<usize, usize> = Default::default();
            let mut first_last: Option<(usize, usize)> = None;
            for (cj, &j) in j_idx.iter().enumerate() {
                let mut wbasis: Vec<(usize, Vec<u64>)> = Vec::new();
                let mut rbasis: Vec<(usize, Vec<u64>)> = Vec::new();
                let mut qbasis: Vec<(usize, Vec<u64>)> = Vec::new();
                let cols = std::iter::once(vec![!0u64; words])
                    .chain((0..nw_g).map(|k| (0..b).map(|batch| gs[batch][cj][k]).collect()));
                for col in cols {
                    if insert(col.clone(), &mut wbasis) {
                        let mut r = col.clone();
                        reduce(&mut r, &ubasis);
                        insert(r, &mut rbasis);
                        let mut q = col;
                        reduce(&mut q, &ibasis);
                        insert(q, &mut qbasis);
                    }
                }
                // dim(U∩W) - dim(I∩W) = rank(W mod I) - rank(W mod U)
                let inter = qbasis.len() - rbasis.len();
                *hist.entry(inter).or_default() += 1;
                if inter > 0 {
                    first_last = Some(match first_last {
                        None => (j, j),
                        Some((a, _)) => (a, j),
                    });
                }
                writeln!(f, "{j}\t{}\t{inter}", wbasis.len()).unwrap();
            }
            println!(
                "[span] dim span(C values,1) = {dim_u} over {} samples; cuts by intersection dim: {:?}; nonzero from cut {:?}",
                b * 64,
                hist,
                first_last
            );
            return;
        }
        let mut best: Vec<i64> = vec![-1; rowinfo.len()];
        // EXPOSURE_WITNESS=<path>: per exposed row, dump the minimizing
        // predictor -- cut position and the participating G wires.
        let witness_path = std::env::var("EXPOSURE_WITNESS").ok();
        // Every (row, cut, weight) at which a row is exactly affine, not just
        // its best cut: the per-value exposure multiplicity over the lattice.
        let allcuts_path = std::env::var("EXPOSURE_ALLCUTS").ok();
        let mut allcuts: Vec<(usize, usize, u32)> = Vec::new();
        let mut witness: Vec<Option<(usize, Vec<usize>)>> = vec![None; rowinfo.len()];
        let mut approx = 0usize;
        for (cj, _) in j_idx.iter().enumerate() {
            // Dedup wire columns under complement; constants are free.
            let mut reps: Vec<Vec<u64>> = Vec::new();
            let mut rep_wires: Vec<usize> = Vec::new();
            let mut seen: std::collections::HashMap<Vec<u64>, usize> =
                std::collections::HashMap::new();
            for k in 0..nw_g {
                let col: Vec<u64> = (0..b).map(|batch| gs[batch][cj][k]).collect();
                let mut key = col;
                if key[0] & 1 == 1 {
                    for w in key.iter_mut() {
                        *w = !*w;
                    }
                }
                if allzero(&key) {
                    continue;
                }
                if !seen.contains_key(&key) {
                    seen.insert(key.clone(), reps.len());
                    reps.push(key);
                    rep_wires.push(k);
                }
            }
            let m = reps.len();
            let cw = (m + 1).div_ceil(64); // coef bits: reps 0..m, const at m
            let mut basis: Vec<(usize, Vec<u64>, Vec<u64>)> = Vec::new();
            let mut nulls: Vec<Vec<u64>> = Vec::new();
            {
                let mut add = |mut samp: Vec<u64>, idx: usize| {
                    let mut coef = vec![0u64; cw];
                    coef[idx / 64] |= 1 << (idx % 64);
                    for (p, bs, bc) in basis.iter() {
                        if (samp[p / 64] >> (p % 64)) & 1 == 1 {
                            for (a, x) in samp.iter_mut().zip(bs) {
                                *a ^= x;
                            }
                            for (a, x) in coef.iter_mut().zip(bc) {
                                *a ^= x;
                            }
                        }
                    }
                    match samp.iter().enumerate().find_map(|(wi, &w)| {
                        (w != 0).then(|| wi * 64 + w.trailing_zeros() as usize)
                    }) {
                        Some(p) => basis.push((p, samp, coef)),
                        None => nulls.push(coef),
                    }
                };
                for idx in 0..m {
                    add(reps[idx].clone(), idx);
                }
                add(vec![!0u64; words], m);
            }
            let wirewt = |coef: &[u64]| -> u32 {
                let mut wt = 0u32;
                for (wi, &w) in coef.iter().enumerate() {
                    let mut ww = w;
                    if wi == m / 64 {
                        ww &= !(1u64 << (m % 64));
                    }
                    wt += ww.count_ones();
                }
                wt
            };
            for (rowi, &(ri, _, t)) in rowinfo.iter().enumerate() {
                if allcuts_path.is_none() && best[rowi] >= 0 && best[rowi] <= 1 {
                    continue; // cannot improve below weight 1 (0 = constant value)
                }
                let mut samp: Vec<u64> = (0..b).map(|batch| cs[batch][ri][t]).collect();
                let mut coef = vec![0u64; cw];
                for (p, bs, bc) in basis.iter() {
                    if (samp[p / 64] >> (p % 64)) & 1 == 1 {
                        for (a, x) in samp.iter_mut().zip(bs) {
                            *a ^= x;
                        }
                        for (a, x) in coef.iter_mut().zip(bc) {
                            *a ^= x;
                        }
                    }
                }
                if !allzero(&samp) {
                    continue; // not exactly affine at this cut
                }
                let mut w = wirewt(&coef);
                let mut argmin = coef.clone();
                if !nulls.is_empty() {
                    if nulls.len() <= 14 {
                        let k = nulls.len();
                        let mut cur = coef.clone();
                        let mut prev = 0usize;
                        for g2 in 1usize..(1usize << k) {
                            let gray = g2 ^ (g2 >> 1);
                            let flip = (gray ^ prev).trailing_zeros() as usize;
                            prev = gray;
                            for (a, x) in cur.iter_mut().zip(&nulls[flip]) {
                                *a ^= x;
                            }
                            let wt = wirewt(&cur);
                            if wt < w {
                                w = wt;
                                argmin = cur.clone();
                            }
                        }
                    } else {
                        approx += 1;
                    }
                }
                if allcuts_path.is_some() {
                    allcuts.push((rowi, cj, w));
                }
                if best[rowi] < 0 || (w as i64) < best[rowi] {
                    best[rowi] = w as i64;
                    if witness_path.is_some() {
                        let mut wires = Vec::new();
                        for idx in 0..m {
                            if (argmin[idx / 64] >> (idx % 64)) & 1 == 1 {
                                wires.push(rep_wires[idx]);
                            }
                        }
                        witness[rowi] = Some((cj, wires));
                    }
                }
            }
        }
        let mut hist: std::collections::BTreeMap<i64, usize> = std::collections::BTreeMap::new();
        for &t in &best {
            *hist.entry(t).or_default() += 1;
        }
        let total = best.len().max(1);
        let visible = best.iter().filter(|&&t| t >= 0).count();
        println!(
            "[exposure] {} born values, {} visible ({:.1}%), {} coset-minimizations skipped (nullity>14)",
            total,
            visible,
            100.0 * visible as f64 / total as f64,
            approx
        );
        let mut cum = 0usize;
        for (&t, &cnt) in hist.iter().filter(|(t, _)| **t >= 0) {
            cum += cnt;
            println!(
                "[exposure] t*={:>3}: {:>5} values ({:>5.1}%)  cumulative {:>5.1}%",
                t,
                cnt,
                100.0 * cnt as f64 / total as f64,
                100.0 * cum as f64 / total as f64
            );
        }
        use std::io::Write as _;
        let path = format!("{}.exposure.tsv", args.out);
        let mut f = std::fs::File::create(&path).expect("write exposure tsv");
        writeln!(
            f,
            "# gate\twire\ttstar (-1 = no exact affine predictor at any sampled cut)"
        )
        .unwrap();
        for (&(_, gi, t), &w) in rowinfo.iter().zip(best.iter()) {
            writeln!(f, "{gi}\t{t}\t{w}").unwrap();
        }
        println!("[exposure] wrote {path}");
        if let Some(apath) = allcuts_path {
            let mut f = std::fs::File::create(&apath).expect("write exposure allcuts");
            writeln!(
                f,
                "# gate\twire\tcut\tweight (every cut where the value is exactly affine)"
            )
            .unwrap();
            for &(rowi, cj, w) in &allcuts {
                let (_, gi, t) = rowinfo[rowi];
                writeln!(f, "{gi}\t{t}\t{}\t{w}", j_idx[cj]).unwrap();
            }
            println!("[exposure] wrote all-cuts {apath} ({} rows)", allcuts.len());
        }
        if let Some(wpath) = witness_path {
            let mut f = std::fs::File::create(&wpath).expect("write exposure witness");
            writeln!(f, "# gate\twire\ttstar\tcut\tG-wires (predictor support)").unwrap();
            for ((&(_, gi, t), &w), wit) in rowinfo.iter().zip(best.iter()).zip(witness.iter()) {
                if w >= 0 {
                    let (cj, wires) = wit.as_ref().unwrap();
                    let list = wires
                        .iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join(",");
                    writeln!(f, "{gi}\t{t}\t{w}\t{}\t{list}", j_idx[*cj]).unwrap();
                }
            }
            println!("[exposure] wrote witness {wpath}");
        }
        return;
    }

    // Per-column excluded-wire sets from --exclude-segments: wire w is
    // excluded at column cut j when some listed segment (w, s, e) is live
    // there (s < j <= e).
    let excl: Vec<std::collections::HashSet<usize>> = if args.exclude_segments.is_empty() {
        vec![std::collections::HashSet::new(); cols]
    } else {
        let text =
            std::fs::read_to_string(&args.exclude_segments).expect("read --exclude-segments file");
        let mut segs: Vec<(usize, usize, usize)> = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut f = line.split_whitespace();
            let (w, s, e) = (
                f.next().and_then(|x| x.parse().ok()),
                f.next().and_then(|x| x.parse().ok()),
                f.next().and_then(|x| x.parse().ok()),
            );
            if let (Some(w), Some(s), Some(e)) = (w, s, e) {
                segs.push((w, s, e));
            }
        }
        let per: Vec<std::collections::HashSet<usize>> = j_idx
            .iter()
            .map(|&j| {
                segs.iter()
                    .filter(|&&(_, s, e)| s < j && j <= e)
                    .map(|&(w, _, _)| w)
                    .collect()
            })
            .collect();
        let total: usize = per.iter().map(|s| s.len()).sum();
        println!(
            "[hmap_affine] exclude-segments: {} segments; mean {:.1} wires excluded per column (max {})",
            segs.len(),
            total as f64 / cols as f64,
            per.iter().map(|s| s.len()).max().unwrap_or(0)
        );
        per
    };

    let ho = b - tb; // holdout batches
    let mut mat = vec![0f32; rows * cols];
    // (holdout error, row, col, target bit, linear support) for --dump-best.
    let mut fits: Vec<(f64, usize, usize, usize, Vec<usize>)> = Vec::new();

    // Scratch reused per cell.
    for (ri, &_i) in i_idx.iter().enumerate() {
        for (cj, &_j) in j_idx.iter().enumerate() {
            // Build a GF(2) basis of G_j's regressor columns over TRAIN samples.
            let mut basis: Vec<(usize, Row)> = Vec::new();
            let add_col = |train: Vec<u64>, reg_idx: usize, basis: &mut Vec<(usize, Row)>| {
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
            let ex = &excl[cj];
            for (ridx, &r) in regs.iter().enumerate() {
                if !ex.is_empty() {
                    let banned = match r {
                        Reg::Lin(k) => ex.contains(&k),
                        Reg::Const => false,
                        Reg::Pair(a, c2) => ex.contains(&a) || ex.contains(&c2),
                        Reg::Triple(a, b2, c3) => {
                            ex.contains(&a) || ex.contains(&b2) || ex.contains(&c3)
                        }
                    };
                    if banned {
                        continue;
                    }
                }
                add_col(reg_train(r, cj, &gs), ridx, &mut basis);
            }

            // Reconstruct each target bit; accumulate error. Under
            // --active-wire the row's only target is the value born at
            // C's prefix end (gate _i-1's target wire).
            let targets: Vec<usize> = if args.active_wire {
                match i_idx[ri] {
                    0 => Vec::new(),
                    i => {
                        let t = c[i - 1].target as usize;
                        if t < n { vec![t] } else { Vec::new() }
                    }
                }
            } else {
                (0..n).collect()
            };
            if targets.is_empty() {
                mat[ri * cols + cj] = 0.5;
                continue;
            }
            let mut err_sum = 0f64;
            for &t in &targets {
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
                let ho_err = errbits as f64 / (ho as f64 * 64.0);
                err_sum += ho_err;
                // Interior only: both axes have unencoded ends (cell (0,0) is
                // C's input against G's input wires and reads perfectly in
                // EVERY build, encoded or not), so port cells say nothing.
                let interior = ri >= rows / 10
                    && ri < rows - rows / 10
                    && cj >= cols / 10
                    && cj < cols - cols / 10;
                if args.dump_best > 0 && interior && ho_err < 0.10 {
                    // A relation that actually predicts. Record its support so
                    // the leak can be attributed to specific wires.
                    let used: Vec<usize> = regs
                        .iter()
                        .enumerate()
                        .filter(|(ridx, _)| (coef[ridx / 64] >> (ridx % 64)) & 1 == 1)
                        .filter_map(|(_, &r)| match r {
                            Reg::Lin(k) => Some(k),
                            _ => None,
                        })
                        .collect();
                    fits.push((ho_err, ri, cj, t, used));
                }
            }
            mat[ri * cols + cj] = (err_sum / targets.len() as f64) as f32;
        }
    }
    if args.dump_best > 0 {
        fits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        println!(
            "[hmap_affine] {} predictive relations (holdout err < 0.10); {} leakiest:",
            fits.len(),
            args.dump_best.min(fits.len())
        );
        for (err, ri, cj, t, used) in fits.iter().take(args.dump_best) {
            let mut w = used.clone();
            w.sort_unstable();
            println!(
                "  row {ri} (C prefix {}) col {cj} (G prefix {}) target bit {t}: err {err:.4}, \
                 {} wires: {:?}{}",
                i_idx[*ri],
                j_idx[*cj],
                w.len(),
                &w[..w.len().min(24)],
                if w.len() > 24 { " ..." } else { "" }
            );
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
    println!(
        "[hmap_affine] H mean={:.4} std={:.4} (0=affine-recoverable, 0.5=hidden)",
        mu, sigma
    );

    let bin = format!("{}.bin", args.out);
    let mut f = std::fs::File::create(&bin).expect("create bin");
    let bytes: Vec<u8> = mat.iter().flat_map(|v| v.to_le_bytes()).collect();
    f.write_all(&bytes).expect("write bin");
    let meta = format!(
        "{{\"rows\":{},\"cols\":{},\"n\":{},\"mu\":{:.6},\"sigma\":{:.6},\"i_idx\":[{}],\"j_idx\":[{}]}}",
        rows,
        cols,
        n,
        mu,
        sigma,
        i_idx
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(","),
        j_idx
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    std::fs::write(format!("{}.meta.json", args.out), meta).expect("write meta");
    println!(
        "[hmap_affine] wrote {} ({}x{} f32) and {}.meta.json",
        bin, rows, cols, args.out
    );
}
