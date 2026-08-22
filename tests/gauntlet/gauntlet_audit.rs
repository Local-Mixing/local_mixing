//! Gauntlet audit engine — runs the four attack classes against a dumped
//! gadget trace (uniform bundle produced by `gauntlet_gen`):
//!
//!   A1  direct wire match:      target == (or == complement of) some trace
//!                               coordinate; also min Hamming distance.
//!   A2  state-affine Gaussian:  per prefix j of G, is the target in the GF(2)
//!                               affine span of the wire state G_j?
//!                               (overdetermined: fit samples >> n_wires)
//!   A3  trace-affine Gaussian:  target in the affine span of the FULL trace
//!                               (init wires + every flip + every newval)?
//!                               fit on nfeat+256 samples, CV on 2048 held out.
//!   A4  bounded correlation scan: all features at weight 1, plus deterministic
//!                               capped/strided feature subsets at weight 2
//!                               (xor/and/or/both orders of andnot) and weight 3
//!                               (xor3/and3/or3/(a&b)^c/(a|b)^c). The weight-2
//!                               operator families are not exhaustive Boolean
//!                               searches; feature selection is additionally
//!                               non-exhaustive when the feature count exceeds
//!                               a configured cap. Uses +/-1 covariance and a
//!                               NULL noise target.
//!
//! Usage: gauntlet_audit --prefix P [--w2-cap N] [--w3-cap N] [--witnesses K]

use clap::Parser;
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    prefix: String,
    /// deterministic strided feature cap for the weight-2 scan
    #[arg(long, default_value_t = 64)]
    w2_cap: usize,
    /// deterministic strided feature cap for the weight-3 scan
    #[arg(long, default_value_t = 16)]
    w3_cap: usize,
    /// witnesses to print per leaked target
    #[arg(long, default_value_t = 10)]
    witnesses: usize,
    /// skip the global exact-linear attack when init+flips features exceed this
    #[arg(long, default_value_t = 40000)]
    a3_max_f: usize,
}

/// One attack witness, serialized to <prefix>.hits.jsonl for the heatmaps.
struct Hit {
    target: String,
    attack: String, // a1 | xrows | xtrace | w1 | w2 | w3
    op: String,     // for w2/w3: xor/and/or/anot/bnot / x3/a3/o3/anx/orx
    features: Vec<u64>,
    strength: f64,       // 1.0 for exact, |cov| for correlations
    prefix: Option<u64>, // xrows only: the state prefix at which it recovered
}

// ------------------------------------------------------------------ loading
struct Bundle {
    meta: HashMap<String, String>,
    names: Vec<String>,
    trivial: Vec<i64>,
    gate_targets: Vec<u16>,
    features: Vec<Vec<u64>>,
    targets: Vec<Vec<u64>>,
    samples: usize,
    n_wires: usize,
    n_gates: usize,
    corr_samples: usize,
}

fn load(prefix: &str) -> Bundle {
    let meta_raw = std::fs::read_to_string(format!("{prefix}.meta")).unwrap();
    let mut meta = HashMap::new();
    let mut names = Vec::new();
    let mut trivial = Vec::new();
    let mut target_lines: Vec<(usize, String, i64)> = Vec::new();
    for line in meta_raw.lines() {
        let (k, v) = line
            .split_once('\t')
            .unwrap_or_else(|| panic!("{prefix}.meta: malformed line {line:?}"));
        if let Some(rest) = k.strip_prefix("target[") {
            let idx: usize = rest
                .strip_suffix(']')
                .expect("target metadata key must end in ]")
                .parse()
                .expect("invalid target metadata index");
            let mut p = v.split('\t');
            let nm = p.next().unwrap().to_string();
            let tr: i64 = p
                .next()
                .expect("target metadata missing trivial feature")
                .parse()
                .expect("invalid target trivial feature");
            assert!(
                p.next().is_none(),
                "{prefix}.meta: target metadata has extra fields"
            );
            target_lines.push((idx, nm, tr));
        } else {
            assert!(
                meta.insert(k.to_string(), v.to_string()).is_none(),
                "{prefix}.meta: duplicate key {k}"
            );
        }
    }
    assert_eq!(
        meta.get("behavioral_ok").map(String::as_str),
        Some("true"),
        "{prefix}.meta: refusing to audit a missing or failed behavioral equivalence check"
    );
    target_lines.sort();
    for (expected, (idx, nm, tr)) in target_lines.into_iter().enumerate() {
        assert_eq!(
            idx, expected,
            "{prefix}.meta: target indices must be unique and contiguous"
        );
        names.push(nm);
        trivial.push(tr);
    }
    let samples: usize = meta["samples"].parse().unwrap();
    let k: usize = meta["k"].parse().unwrap();
    let n: usize = meta["n"].parse().unwrap();
    let n_wires: usize = meta["n_wires"].parse().unwrap();
    let n_gates: usize = meta["n_gates"].parse().unwrap();
    let n_features: usize = meta["n_features"].parse().unwrap();
    let n_targets: usize = meta["n_targets"].parse().unwrap();
    let corr_samples: usize = meta["corr_samples"].parse().unwrap();
    let gadget = meta.get("gadget").expect("missing gadget metadata");
    assert!(
        matches!(gadget.as_str(), "none" | "ss" | "semi" | "band" | "file"),
        "unsupported gadget metadata {gadget}"
    );
    let expected_features = n_wires
        .checked_add(
            n_gates
                .checked_mul(2)
                .expect("n_features gate term overflow"),
        )
        .expect("n_features overflow");
    assert_eq!(
        n_features, expected_features,
        "n_features must equal n_wires + 2*n_gates"
    );
    assert!(n > 0 && n <= n_wires, "invalid logical wire count");
    let expected_targets = k
        .checked_mul(5)
        .and_then(|value| value.checked_add(1))
        .expect("n_targets overflow");
    assert_eq!(n_targets, expected_targets, "n_targets must equal 5*k + 1");
    assert_eq!(names.len(), n_targets, "target metadata count mismatch");
    for gate in 0..k {
        for (offset, kind) in ["a", "b", "cold", "f", "cnew"].into_iter().enumerate() {
            assert_eq!(
                names[5 * gate + offset],
                format!("g{gate}:{kind}"),
                "unexpected target metadata name"
            );
        }
    }
    assert_eq!(names[5 * k], "NULL", "last target must be NULL");
    let file_mode = gadget == "file";
    let logical_limit = i64::try_from(n).expect("logical wire count exceeds i64");
    for (index, &feature) in trivial.iter().enumerate() {
        if file_mode {
            assert_eq!(
                feature, -1,
                "file-mode target {index} cannot name a raw-input trace feature"
            );
        } else {
            assert!(
                feature == -1 || (0..logical_limit).contains(&feature),
                "target {index} has an invalid trivial feature"
            );
        }
    }
    assert_eq!(trivial[5 * k], -1, "NULL target cannot be trivial");
    assert!(
        samples > 0 && samples % 64 == 0,
        "samples must be positive and 64-aligned"
    );
    assert!(
        corr_samples > 0 && corr_samples % 64 == 0,
        "corr_samples must be positive and 64-aligned"
    );
    let fit_samples = (n_features + 256).next_multiple_of(64);
    assert_eq!(
        samples,
        fit_samples + 2048 + corr_samples,
        "bundle sample layout must be fit + 2048 CV + correlation tail"
    );
    let gate_targets: Vec<u16> = if meta["gate_targets"].is_empty() {
        Vec::new()
    } else {
        meta["gate_targets"]
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect()
    };
    assert_eq!(gate_targets.len(), n_gates);
    assert!(
        gate_targets
            .iter()
            .all(|&target| (target as usize) < n_wires),
        "gate target is outside n_wires"
    );
    let w = samples / 64;
    let read_cols = |path: &str, n: usize| -> Vec<Vec<u64>> {
        let raw = std::fs::read(path).unwrap();
        assert_eq!(raw.len(), n * w * 8, "{path}: size mismatch");
        raw.chunks_exact(w * 8)
            .map(|c| {
                c.chunks_exact(8)
                    .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
                    .collect()
            })
            .collect()
    };
    let features = read_cols(&format!("{prefix}.trace.bin"), n_features);
    let targets = read_cols(&format!("{prefix}.targets.bin"), names.len());
    Bundle {
        meta,
        names,
        trivial,
        gate_targets,
        features,
        targets,
        samples,
        n_wires,
        n_gates,
        corr_samples,
    }
}

// ------------------------------------------------------------------ GF(2) basis
/// Affine-span membership over the fit window, with coefficient tracking so a
/// successful reduction yields an explicit witness relation (column indices;
/// index `const_idx` is the constant-1 column).
struct Basis<'a> {
    cols: &'a [&'a [u64]], // all available columns
    fw: usize,             // fit words
    // (pivot bit position, reduced column (fw words), coeff bitset over col idx)
    items: Vec<(u32, Vec<u64>, Vec<u64>)>,
    coeff_words: usize,
    track: bool,
}

impl<'a> Basis<'a> {
    fn new(cols: &'a [&'a [u64]], fw: usize, track: bool) -> Self {
        let coeff_words = cols.len().div_ceil(64);
        Basis {
            cols,
            fw,
            items: Vec::new(),
            coeff_words,
            track,
        }
    }
    /// Reduce an arbitrary column (given by slice) against the basis.
    /// When tracking, seeds the coeff bitset with `seed_idx` if provided.
    fn reduce(&self, col: &[u64], seed_idx: Option<usize>) -> (Vec<u64>, Vec<u64>) {
        let mut m: Vec<u64> = col[0..self.fw].to_vec();
        let mut cf = vec![0u64; self.coeff_words];
        if self.track {
            if let Some(idx) = seed_idx {
                cf[idx / 64] |= 1 << (idx % 64);
            }
        }
        for (pv, bm, bc) in &self.items {
            if (m[*pv as usize / 64] >> (*pv % 64)) & 1 == 1 {
                for i in 0..self.fw {
                    m[i] ^= bm[i];
                }
                if self.track {
                    for i in 0..self.coeff_words {
                        cf[i] ^= bc[i];
                    }
                }
            }
        }
        (m, cf)
    }
    fn insert(&mut self, col_idx: usize) {
        let (m, cf) = self.reduce(self.cols[col_idx], Some(col_idx));
        if let Some(last) = m.iter().rposition(|&x| x != 0) {
            let pv = (64 * last + 63 - m[last].leading_zeros() as usize) as u32;
            self.items.push((pv, m, cf));
        }
    }
    /// Some(coeff over basis column indices) if `col` is in the span.
    fn contains(&self, col: &[u64]) -> Option<Vec<u64>> {
        let (m, cf) = self.reduce(col, None);
        if m.iter().all(|&x| x == 0) {
            Some(cf)
        } else {
            None
        }
    }
}

/// Verify a witness relation on a disjoint sample window: XOR the selected
/// columns over [w0, w0+cw) and compare against the target; returns the number
/// of mismatching words.
fn cv_check(cols: &[&[u64]], coeff: &[u64], target: &[u64], w0: usize, cw: usize) -> usize {
    let mut bad = 0;
    'outer: for wi in w0..w0 + cw {
        let mut acc = 0u64;
        for (ci, c) in cols.iter().enumerate() {
            if (coeff[ci / 64] >> (ci % 64)) & 1 == 1 {
                acc ^= c[wi];
            }
        }
        if acc != target[wi] {
            bad += 1;
            if bad > 0 {
                break 'outer;
            }
        }
    }
    bad
}

fn feature_name(idx: usize, nw: usize) -> String {
    if idx < nw {
        format!("init[w{idx}]")
    } else {
        let g = (idx - nw) / 2;
        if (idx - nw) % 2 == 0 {
            format!("g{g}:flip")
        } else {
            format!("g{g}:new")
        }
    }
}

// ------------------------------------------------------------------ main
fn main() {
    let args = Args::parse();
    assert!(args.w2_cap >= 2, "--w2-cap must be at least 2");
    assert!(args.w3_cap >= 3, "--w3-cap must be at least 3");
    let b = load(&args.prefix);
    let s_fit: usize = (b.features.len() + 256).div_ceil(64) * 64;
    let fw = s_fit / 64;
    let cv_w0 = fw; // 2048 held-out samples
    let cw = 2048 / 64;
    let corr_tail = b.corr_samples / 64; // words
    let corr_w0 = b.samples / 64 - corr_tail;
    let nw = b.n_wires;
    let nfeat = b.features.len();
    let nt = b.targets.len();

    // column store: features ++ const-1 (index nfeat)
    let const_col: Vec<u64> = (0..b.samples / 64).map(|_| u64::MAX).collect();
    let mut colrefs: Vec<&[u64]> = b.features.iter().map(|c| c.as_slice()).collect();
    colrefs.push(&const_col);
    let const_idx = nfeat;

    println!(
        "== {} ==  S={} F={} (nw={} gates={}) targets={} | fit={} cv=2048 corr={}",
        b.meta.get("gadget").map(|s| s.as_str()).unwrap_or("?"),
        b.samples,
        nfeat,
        nw,
        b.n_gates,
        nt,
        s_fit,
        b.corr_samples
    );

    // ------------------------------------------------- A1: direct wire match
    println!("-- A1 direct wire match --");
    let mut hits: Vec<Hit> = Vec::new();
    let mut a1_lines = Vec::new();
    let mut a1_leaks = 0;
    let mut a1_leaks_nontrivial = 0;
    for (ti, t) in b.targets.iter().enumerate() {
        let mut exact = Vec::new();
        let mut min_hd = f64::MAX;
        for (fi, f) in b.features.iter().enumerate() {
            let mut hd = 0u64;
            for i in 0..t.len() {
                hd += (f[i] ^ t[i]).count_ones() as u64;
            }
            let hdc = b.samples as u64 - hd; // complement agreement
            let m = hd.min(hdc);
            if m == 0 {
                exact.push((fi, hdc == 0));
            }
            let frac = m as f64 / b.samples as f64;
            if frac < min_hd {
                min_hd = frac;
            }
        }
        let triv = b.trivial[ti];
        let nontriv = exact.iter().filter(|&&(fi, _)| fi as i64 != triv).count();
        if !exact.is_empty() {
            a1_leaks += 1;
            a1_leaks_nontrivial += (nontriv > 0) as usize;
            let fs: Vec<u64> = exact.iter().take(16).map(|&(fi, _)| fi as u64).collect();
            if nontriv > 0 {
                hits.push(Hit {
                    target: b.names[ti].clone(),
                    attack: "a1".into(),
                    op: "".into(),
                    features: fs,
                    strength: 1.0,
                    prefix: None,
                });
            }
        }
        let first = exact
            .first()
            .map(|&(fi, comp)| {
                format!(
                    "{}{}",
                    feature_name(fi, nw),
                    if comp { " (comp)" } else { "" }
                )
            })
            .unwrap_or_default();
        a1_lines.push(format!(
            "A1 {}\texact={} nontrivial={} first={} min_hd={:.4}",
            b.names[ti],
            exact.len(),
            nontriv,
            first,
            min_hd
        ));
    }
    for l in &a1_lines {
        if !l.contains("exact=0 ") {
            println!("{l}");
        }
    }
    println!(
        "A1 summary: {a1_leaks}/{} targets matched (nontrivial: {a1_leaks_nontrivial})",
        nt
    );

    // ------------------------------------------------- A2: state-affine (per prefix)
    println!(
        "-- A2 state-affine Gaussian (fit {} samples, cv 2048) --",
        fw * 64
    );
    let mut a2_first: Vec<Option<(usize, usize, bool, bool)>> = vec![None; nt]; // (prefix, weight, cv_ok, trivial)
    // parallel over prefixes, merge per-target earliest
    let per_prefix: Vec<Vec<(usize, usize, bool, bool)>> = (0..=b.n_gates)
        .into_par_iter()
        .map(|prefix| {
            let mut latest: Vec<usize> = (0..nw).collect();
            for (j, &tg) in b.gate_targets.iter().take(prefix).enumerate() {
                latest[tg as usize] = nw + 2 * j + 1;
            }
            let mut cols: Vec<&[u64]> = latest.iter().map(|&i| colrefs[i]).collect();
            cols.push(&const_col);
            let mut basis = Basis::new(&cols, fw, false);
            for i in 0..=nw {
                basis.insert(i);
            }
            let hits: Vec<usize> = (0..nt)
                .filter(|&ti| {
                    let t = &b.targets[ti];
                    let mut m: Vec<u64> = t[0..fw].to_vec();
                    for (pv, bm, _) in &basis.items {
                        if (m[*pv as usize / 64] >> (*pv % 64)) & 1 == 1 {
                            for i in 0..fw {
                                m[i] ^= bm[i];
                            }
                        }
                    }
                    m.iter().all(|&x| x == 0)
                })
                .collect();
            if hits.is_empty() {
                return Vec::new();
            }
            let mut basis = Basis::new(&cols, fw, true);
            for i in 0..=nw {
                basis.insert(i);
            }
            let mut out = Vec::new();
            for &ti in &hits {
                if let Some(cf) = basis.contains(&b.targets[ti]) {
                    let mut gcf = vec![0u64; (nfeat + 1) / 64 + 1];
                    let mut weight = 0usize;
                    let mut trivial_hit = false;
                    for ci in 0..=nw {
                        if (cf[ci / 64] >> (ci % 64)) & 1 == 1 {
                            let gi = if ci == nw { const_idx } else { latest[ci] };
                            gcf[gi / 64] |= 1 << (gi % 64);
                            weight += 1;
                            if gi as i64 == b.trivial[ti] {
                                trivial_hit = true;
                            }
                        }
                    }
                    let bad = cv_check(&colrefs, &gcf, &b.targets[ti], cv_w0, cw);
                    out.push((ti, weight, bad == 0, trivial_hit && weight == 1));
                }
            }
            // annotate with prefix via packing order: caller zips
            out.into_iter()
                .map(|(ti, wgt, ok, tv)| (prefix * 1_000_000 + ti, wgt, ok, tv))
                .collect::<Vec<_>>()
        })
        .collect();
    for (prefix, v) in per_prefix.iter().enumerate() {
        for &(packed, wgt, ok, tv) in v {
            let ti = packed % 1_000_000;
            let e = &mut a2_first[ti];
            // A fit-span candidate can be a finite-sample artifact. Keep
            // searching later prefixes until one also validates on the
            // held-out rows, then retain the earliest CV-valid recovery.
            if ok && e.is_none() {
                *e = Some((prefix, wgt, ok, tv));
            }
        }
    }
    let mut a2_leaks = 0;
    let mut a2_leaks_nt = 0;
    for ti in 0..nt {
        if let Some((prefix, wgt, ok, tv)) = a2_first[ti] {
            if ok {
                a2_leaks += 1;
                if !tv {
                    a2_leaks_nt += 1;
                }
                println!(
                    "XROWS {}\trecovered prefix={} weight={} cv=ok{}",
                    b.names[ti],
                    prefix,
                    wgt,
                    if tv { " TRIVIAL(input-wire)" } else { "" }
                );
                if !tv {
                    // witness = the state's wires: re-derive via tracked pass
                    // (cheap single-target rebuild at the recovered prefix)
                    let mut latest: Vec<usize> = (0..nw).collect();
                    for (j, &tg) in b.gate_targets.iter().take(prefix).enumerate() {
                        latest[tg as usize] = nw + 2 * j + 1;
                    }
                    let mut cols: Vec<&[u64]> = latest.iter().map(|&i| colrefs[i]).collect();
                    cols.push(&const_col);
                    let mut basis = Basis::new(&cols, fw, true);
                    for i in 0..=nw {
                        basis.insert(i);
                    }
                    if let Some(cf) = basis.contains(&b.targets[ti]) {
                        let mut feats: Vec<u64> = Vec::new();
                        for ci in 0..=nw {
                            if (cf[ci / 64] >> (ci % 64)) & 1 == 1 {
                                if ci != nw {
                                    feats.push(latest[ci] as u64);
                                }
                            }
                        }
                        hits.push(Hit {
                            target: b.names[ti].clone(),
                            attack: "xrows".into(),
                            op: "".into(),
                            features: feats,
                            strength: 1.0,
                            prefix: Some(prefix as u64),
                        });
                    }
                }
            }
        }
    }
    println!(
        "XROWS summary: {a2_leaks}/{nt} targets recoverable from some state (nontrivial: {a2_leaks_nt})"
    );

    // ------------------------------------------------- A3: global exact-linear (full trace of G)
    // Feature set: init wires + gate FLIPS only.  This has the same span as
    // the full trace (newval_j = init[tgt] ^ flips_{<=j on tgt}), at half the
    // columns: F' = nw + ng.
    println!(
        "-- XTRACE global exact-linear (features = init + flips, F' = {}, fit {} cv 2048) --",
        nw + b.n_gates,
        fw * 64
    );
    let flip_feat = |j: usize| nw + 2 * j; // map flip j -> full-trace feature idx
    let (mut a3_leaks, mut a3_leaks_nt) = (0usize, 0usize);
    let xtrace_status;
    if nw + b.n_gates > args.a3_max_f {
        xtrace_status = "skipped";
        println!(
            "XTRACE skipped: F'={} exceeds --a3-max-f {}",
            nw + b.n_gates,
            args.a3_max_f
        );
    } else {
        xtrace_status = "completed";
        // reduced columns over the fit window (block-parallel pre-reduction)
        let mut rcols: Vec<Vec<u64>> = Vec::with_capacity(nw + b.n_gates + 1);
        let mut basis: Vec<(u32, Vec<u64>)> = Vec::new(); // (pivot, reduced)
        let mut rank_list: Vec<Option<u32>> = Vec::new();
        let blk = 1024usize;
        let total = nw + b.n_gates + 1;
        for lo in (0..total).step_by(blk) {
            let hi = (lo + blk).min(total);
            // parallel pre-reduce against the CURRENT basis snapshot
            let snap = &basis;
            let mut pre: Vec<Vec<u64>> = (lo..hi)
                .into_par_iter()
                .map(|i| {
                    let src: &[u64] = if i == total - 1 {
                        &const_col
                    } else if i < nw {
                        &b.features[i]
                    } else {
                        &b.features[flip_feat(i - nw)]
                    };
                    let mut m: Vec<u64> = src[0..fw].to_vec();
                    for (pv, bm) in snap {
                        if (m[*pv as usize / 64] >> (*pv % 64)) & 1 == 1 {
                            for x in 0..fw {
                                m[x] ^= bm[x];
                            }
                        }
                    }
                    m
                })
                .collect();
            // sequential finalize
            for m0 in pre.drain(..) {
                let mut m = m0;
                for (pv, bm) in &basis {
                    if (m[*pv as usize / 64] >> (*pv % 64)) & 1 == 1 {
                        for x in 0..fw {
                            m[x] ^= bm[x];
                        }
                    }
                }
                if let Some(last) = m.iter().rposition(|&x| x != 0) {
                    let pv = (64 * last + 63 - m[last].leading_zeros() as usize) as u32;
                    basis.push((pv, m.clone()));
                    rank_list.push(Some(pv));
                } else {
                    rank_list.push(None);
                }
                rcols.push(m);
            }
        }
        let rank = basis.len();
        println!("trace rank {} / {} columns", rank, total);
        // membership for all targets (parallel)
        let in_span: Vec<bool> = (0..nt)
            .into_par_iter()
            .map(|ti| {
                let t = &b.targets[ti];
                let mut m: Vec<u64> = t[0..fw].to_vec();
                for (pv, bm) in &basis {
                    if (m[*pv as usize / 64] >> (*pv % 64)) & 1 == 1 {
                        for x in 0..fw {
                            m[x] ^= bm[x];
                        }
                    }
                }
                m.iter().all(|&x| x == 0)
            })
            .collect();
        // tracked rebuild only if some target is in span (witness extraction)
        let hit_targets: Vec<usize> = (0..nt).filter(|&ti| in_span[ti]).collect();
        let mut tb_cols: Vec<&[u64]> = Vec::with_capacity(total);
        if !hit_targets.is_empty() {
            for i in 0..total {
                tb_cols.push(if i == total - 1 {
                    const_col.as_slice()
                } else if i < nw {
                    b.features[i].as_slice()
                } else {
                    b.features[flip_feat(i - nw)].as_slice()
                });
            }
        }
        let mut tb = Basis::new(&tb_cols, fw, true);
        if !hit_targets.is_empty() {
            for i in 0..total {
                tb.insert(i);
            }
        }
        for ti in 0..nt {
            if !in_span[ti] {
                continue;
            }
            let t = &b.targets[ti];
            let tb = &tb;
            // tracked reduce of target
            let mut m: Vec<u64> = t[0..fw].to_vec();
            let mut cf = vec![0u64; tb.coeff_words];
            for (pv, bm, bc) in &tb.items {
                if (m[*pv as usize / 64] >> (*pv % 64)) & 1 == 1 {
                    for x in 0..fw {
                        m[x] ^= bm[x];
                    }
                    for x in 0..cf.len() {
                        cf[x] ^= bc[x];
                    }
                }
            }
            // map reduced-column coeff -> full-trace feature ids
            let mut gcf = vec![0u64; (nfeat + 1) / 64 + 1];
            let mut feats: Vec<u64> = Vec::new();
            let mut weight = 0usize;
            let total = nw + b.n_gates + 1;
            for ci in 0..total {
                if (cf[ci / 64] >> (ci % 64)) & 1 == 1 {
                    weight += 1;
                    let gi = if ci == total - 1 {
                        const_idx
                    } else if ci < nw {
                        ci
                    } else {
                        flip_feat(ci - nw)
                    };
                    gcf[gi / 64] |= 1 << (gi % 64);
                    if gi != const_idx {
                        feats.push(gi as u64);
                    }
                }
            }
            let bad = cv_check(&colrefs, &gcf, t, cv_w0, cw);
            let trivial_only = weight == 1
                && b.trivial[ti] >= 0
                && (gcf[b.trivial[ti] as usize / 64] >> (b.trivial[ti] as usize % 64)) & 1 == 1;
            if bad == 0 {
                a3_leaks += 1;
                if !trivial_only {
                    a3_leaks_nt += 1;
                }
                let mut wit: Vec<String> = Vec::new();
                for &gi in &feats {
                    if wit.len() < args.witnesses {
                        wit.push(feature_name(gi as usize, nw));
                    }
                }
                println!(
                    "XTRACE {}\tIN SPAN weight={} cv=ok{} witness= {}",
                    b.names[ti],
                    weight,
                    if trivial_only {
                        " TRIVIAL(input-wire)"
                    } else {
                        ""
                    },
                    wit.join(" ^ ")
                );
                if !trivial_only {
                    hits.push(Hit {
                        target: b.names[ti].clone(),
                        attack: "xtrace".into(),
                        op: "".into(),
                        features: feats,
                        strength: 1.0,
                        prefix: None,
                    });
                }
            } else {
                println!(
                    "XTRACE {}\tfit-in-span but CV FAILED (artifact, not counted)",
                    b.names[ti]
                );
            }
        }
    }
    if xtrace_status == "completed" {
        println!(
            "XTRACE summary: {a3_leaks}/{nt} targets in affine span of full trace (nontrivial: {a3_leaks_nt})"
        );
    } else {
        println!("XTRACE summary: skipped (no leak counts computed)");
    }

    // ------------------------------------------------- A4: correlations
    println!(
        "-- A4 bounded correlation scan (tail {} samples) --",
        b.corr_samples
    );
    let s_corr = b.corr_samples as f64;
    // Deterministic strided feature subsets for w2/w3; selection covers all
    // features only when nfeat is at or below the respective cap. Even then,
    // the bounded operator families are not exhaustive Boolean-function scans.
    // W1 scans all features independently of these sets.
    let stride = |cap: usize| -> Vec<usize> {
        if nfeat <= cap {
            (0..nfeat).collect()
        } else {
            let st = (nfeat + cap - 1) / cap;
            (0..nfeat).step_by(st).take(cap).collect()
        }
    };
    let w2_set = stride(args.w2_cap);
    let w3_set = stride(args.w3_cap);
    println!(
        "A4 coverage: w1_features={nfeat}/{nfeat} w2_features={}/{} w3_features={}/{} selection=deterministic-strided",
        w2_set.len(),
        nfeat,
        w3_set.len(),
        nfeat
    );
    let pops: Vec<u64> = (0..nfeat)
        .into_par_iter()
        .map(|i| {
            let mut p = 0u64;
            for wi in corr_w0..corr_w0 + corr_tail {
                p += b.features[i][wi].count_ones() as u64;
            }
            p
        })
        .collect();
    let tpops: Vec<u64> = b
        .targets
        .iter()
        .map(|t| {
            let mut p = 0u64;
            for wi in corr_w0..corr_w0 + corr_tail {
                p += t[wi].count_ones() as u64;
            }
            p
        })
        .collect();

    // cov(u,t) from popcounts on the corr window
    let cov_from = |upop: u64, u_xor_t_pop: u64, tpop: u64| -> f64 {
        let eu = 1.0 - 2.0 * upop as f64 / s_corr;
        let et = 1.0 - 2.0 * tpop as f64 / s_corr;
        let eut = 1.0 - 2.0 * u_xor_t_pop as f64 / s_corr;
        (eut - eu * et).abs()
    };

    // w1 over all features
    let w1: Vec<(f64, u64)> = (0..nt)
        .into_par_iter()
        .map(|ti| {
            let t = &b.targets[ti];
            let mut best = (0.0f64, 0u64);
            for fi in 0..nfeat {
                let f = &b.features[fi];
                let mut xp = 0u64;
                for wi in corr_w0..corr_w0 + corr_tail {
                    xp += (f[wi] ^ t[wi]).count_ones() as u64;
                }
                let c = cov_from(pops[fi], xp, tpops[ti]);
                if c > best.0 {
                    best = (c, fi as u64);
                }
            }
            best
        })
        .collect();

    // w2: pairs over the capped/strided w2_set, ops xor/and/or and both
    // directional and-not orderings. This is a bounded, non-exhaustive
    // Boolean-function family.
    // acc entry: (cov, packed witness (feat_i<<32)|feat_j)
    let w2: Vec<[(f64, u64); 5]> = (0..w2_set.len())
        .into_par_iter()
        .map(|ii| {
            let mut acc = vec![[(0.0f64, 0u64); 5]; nt];
            let a = &b.features[w2_set[ii]];
            let apop = pops[w2_set[ii]];
            let mut xw = vec![0u64; corr_tail];
            let mut aw = vec![0u64; corr_tail];
            let mut ow = vec![0u64; corr_tail];
            let mut anw = vec![0u64; corr_tail];
            let mut bnw = vec![0u64; corr_tail];
            for jj in ii + 1..w2_set.len() {
                let c = &b.features[w2_set[jj]];
                let cpop = pops[w2_set[jj]];
                let mut px = 0u64;
                let mut pa = 0u64;
                let mut po = 0u64;
                let mut pn = 0u64;
                let mut pb = 0u64;
                for k in 0..corr_tail {
                    let av = a[corr_w0 + k];
                    let cv = c[corr_w0 + k];
                    xw[k] = av ^ cv;
                    aw[k] = av & cv;
                    ow[k] = av | cv;
                    anw[k] = av & !cv;
                    bnw[k] = cv & !av;
                    px += xw[k].count_ones() as u64;
                    pa += aw[k].count_ones() as u64;
                    po += ow[k].count_ones() as u64;
                    pn += anw[k].count_ones() as u64;
                    pb += bnw[k].count_ones() as u64;
                }
                // E[s_a s_c] for and/or translation
                let eac = 1.0 - 2.0 * pa as f64 / s_corr;
                let _ = eac;
                let ea = 1.0 - 2.0 * apop as f64 / s_corr;
                let ec = 1.0 - 2.0 * cpop as f64 / s_corr;
                let ex = 1.0 - 2.0 * px as f64 / s_corr;
                let eo = 1.0 - 2.0 * po as f64 / s_corr;
                let en = 1.0 - 2.0 * pn as f64 / s_corr;
                let eb = 1.0 - 2.0 * pb as f64 / s_corr;
                let e_and = 1.0 - 2.0 * pa as f64 / s_corr;
                let wit = ((w2_set[ii] as u64) << 32) | w2_set[jj] as u64;
                for ti in 0..nt {
                    let t = &b.targets[ti];
                    let mut hx = 0u64;
                    let mut ha = 0u64;
                    let mut ho = 0u64;
                    let mut hn = 0u64;
                    let mut hb = 0u64;
                    for k in 0..corr_tail {
                        let tv = t[corr_w0 + k];
                        hx += (xw[k] ^ tv).count_ones() as u64;
                        ha += (aw[k] ^ tv).count_ones() as u64;
                        ho += (ow[k] ^ tv).count_ones() as u64;
                        hn += (anw[k] ^ tv).count_ones() as u64;
                        hb += (bnw[k] ^ tv).count_ones() as u64;
                    }
                    let et = 1.0 - 2.0 * tpops[ti] as f64 / s_corr;
                    let cs = [
                        (1.0 - 2.0 * hx as f64 / s_corr - ex * et).abs(),
                        (1.0 - 2.0 * ha as f64 / s_corr - e_and * et).abs(),
                        (1.0 - 2.0 * ho as f64 / s_corr - eo * et).abs(),
                        (1.0 - 2.0 * hn as f64 / s_corr - en * et).abs(),
                        (1.0 - 2.0 * hb as f64 / s_corr - eb * et).abs(),
                    ];
                    for o in 0..5 {
                        if cs[o] > acc[ti][o].0 {
                            acc[ti][o] = (cs[o], wit);
                        }
                    }
                }
                let _ = (ea, ec);
            }
            acc
        })
        .reduce(
            || vec![[(0.0, 0); 5]; nt],
            |mut a, b| {
                for ti in 0..nt {
                    for o in 0..5 {
                        if b[ti][o].0 > a[ti][o].0 {
                            a[ti][o] = b[ti][o];
                        }
                    }
                }
                a
            },
        );

    // w3: triples over the capped/strided w3_set, with the bounded,
    // non-exhaustive family xor3/and3/or3/(a&b)^c/(a|b)^c.
    // acc entry: (cov, packed witness i|j|k, 21 bits each)
    let w3: Vec<[(f64, u64); 5]> = (0..w3_set.len())
        .into_par_iter()
        .map(|ii| {
            let mut acc = vec![[(0.0f64, 0u64); 5]; nt];
            let a = &b.features[w3_set[ii]];
            let mut u = vec![0u64; corr_tail];
            for jj in ii + 1..w3_set.len() {
                let c = &b.features[w3_set[jj]];
                for kk in jj + 1..w3_set.len() {
                    let d = &b.features[w3_set[kk]];
                    for op in 0..5 {
                        let mut up = 0u64;
                        for k in 0..corr_tail {
                            let (av, cv, dv) = (a[corr_w0 + k], c[corr_w0 + k], d[corr_w0 + k]);
                            u[k] = match op {
                                0 => av ^ cv ^ dv,
                                1 => av & cv & dv,
                                2 => av | cv | dv,
                                3 => (av & cv) ^ dv,
                                _ => (av | cv) ^ dv,
                            };
                            up += u[k].count_ones() as u64;
                        }
                        let eu = 1.0 - 2.0 * up as f64 / s_corr;
                        let wit = ((w3_set[ii] as u64) << 42)
                            | ((w3_set[jj] as u64) << 21)
                            | w3_set[kk] as u64;
                        for ti in 0..nt {
                            let t = &b.targets[ti];
                            let mut h = 0u64;
                            for k in 0..corr_tail {
                                h += (u[k] ^ t[corr_w0 + k]).count_ones() as u64;
                            }
                            let et = 1.0 - 2.0 * tpops[ti] as f64 / s_corr;
                            let cov = (1.0 - 2.0 * h as f64 / s_corr - eu * et).abs();
                            if cov > acc[ti][op].0 {
                                acc[ti][op] = (cov, wit);
                            }
                        }
                    }
                }
            }
            acc
        })
        .reduce(
            || vec![[(0.0, 0); 5]; nt],
            |mut a, b| {
                for ti in 0..nt {
                    for o in 0..5 {
                        if b[ti][o].0 > a[ti][o].0 {
                            a[ti][o] = b[ti][o];
                        }
                    }
                }
                a
            },
        );

    let null_idx = nt - 1;
    let null_w2 = w2[null_idx].iter().map(|e| e.0).fold(0.0, f64::max);
    let null_w3 = w3[null_idx].iter().map(|e| e.0).fold(0.0, f64::max);
    let sigma = 1.0 / s_corr.sqrt();
    let thr2 = null_w2.max(6.0 * sigma);
    let thr3 = null_w3.max(6.0 * sigma);
    let thr1 = w1[null_idx].0.max(6.0 * sigma);
    let ops2 = ["xor", "and", "or", "anot", "bnot"];
    let ops3 = ["x3", "a3", "o3", "anx", "orx"];
    let mut a4_w1_flag = 0;
    let mut a4_w2_flag = 0;
    let mut a4_w3_flag = 0;
    for ti in 0..nt {
        let w2m = w2[ti].iter().map(|e| e.0).fold(0.0, f64::max);
        let w3m = w3[ti].iter().map(|e| e.0).fold(0.0, f64::max);
        let flagged = b.names[ti] != "NULL" && (w1[ti].0 > thr1 || w2m > thr2 || w3m > thr3);
        if b.names[ti] != "NULL" {
            a4_w1_flag += (w1[ti].0 > thr1) as usize;
            a4_w2_flag += (w2m > thr2) as usize;
            a4_w3_flag += (w3m > thr3) as usize;
            if w1[ti].0 > thr1 {
                hits.push(Hit {
                    target: b.names[ti].clone(),
                    attack: "w1".into(),
                    op: "id".into(),
                    features: vec![w1[ti].1],
                    strength: w1[ti].0,
                    prefix: None,
                });
            }
            for o in 0..5 {
                let (c, wit) = w2[ti][o];
                if c > thr2 {
                    hits.push(Hit {
                        target: b.names[ti].clone(),
                        attack: "w2".into(),
                        op: ops2[o].into(),
                        features: vec![wit >> 32, wit & 0xffff_ffff],
                        strength: c,
                        prefix: None,
                    });
                }
            }
            for o in 0..5 {
                let (c, wit) = w3[ti][o];
                if c > thr3 {
                    hits.push(Hit {
                        target: b.names[ti].clone(),
                        attack: "w3".into(),
                        op: ops3[o].into(),
                        features: vec![wit >> 42, (wit >> 21) & 0x1f_ffff, wit & 0x1f_ffff],
                        strength: c,
                        prefix: None,
                    });
                }
            }
        }
        println!(
            "A4 {}\tw1={:.4} w2[xor={:.4} and={:.4} or={:.4} anot={:.4} bnot={:.4}] w3[x3={:.4} a3={:.4} o3={:.4} anx={:.4} orx={:.4}]{}",
            b.names[ti],
            w1[ti].0,
            w2[ti][0].0,
            w2[ti][1].0,
            w2[ti][2].0,
            w2[ti][3].0,
            w2[ti][4].0,
            w3[ti][0].0,
            w3[ti][1].0,
            w3[ti][2].0,
            w3[ti][3].0,
            w3[ti][4].0,
            if flagged { "  <-- FLAG" } else { "" }
        );
        if flagged {
            for o in 0..5 {
                let (c, wit) = w2[ti][o];
                if c > thr2 {
                    let (fi, fj) = ((wit >> 32) as usize, (wit & 0xffff_ffff) as usize);
                    let a = feature_name(fi, nw);
                    let b = feature_name(fj, nw);
                    let expr = match o {
                        0 => format!("{a} xor {b}"),
                        1 => format!("{a} and {b}"),
                        2 => format!("{a} or {b}"),
                        3 => format!("{a} and !{b}"),
                        _ => format!("{b} and !{a}"),
                    };
                    println!("    w2-witness {}: {} (cov {:.4})", ops2[o], expr, c);
                }
            }
            for o in 0..5 {
                let (c, wit) = w3[ti][o];
                if c > thr3 {
                    let fi = (wit >> 42) as usize;
                    let fj = ((wit >> 21) & 0x1f_ffff) as usize;
                    let fk = (wit & 0x1f_ffff) as usize;
                    println!(
                        "    w3-witness {}: {} , {} , {} (cov {:.4})",
                        ops3[o],
                        feature_name(fi, nw),
                        feature_name(fj, nw),
                        feature_name(fk, nw),
                        c
                    );
                }
            }
        }
    }
    println!(
        "A4 summary: sigma={:.4} null(w1={:.4} w2={:.4} w3={:.4}) flagged: w1={} w2={} w3={} of {}",
        sigma,
        w1[null_idx].0,
        null_w2,
        null_w3,
        a4_w1_flag,
        a4_w2_flag,
        a4_w3_flag,
        nt - 1
    );

    // ---------------- hits.jsonl (witnesses for the heatmaps) ----------------
    let mut hj = String::new();
    for h in &hits {
        hj.push_str(&format!(
            "{{\"target\":\"{}\",\"attack\":\"{}\",\"op\":\"{}\",\"features\":[{}],\"strength\":{:.4}{}}}\n",
            h.target, h.attack, h.op,
            h.features.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(","),
            h.strength,
            h.prefix.map(|p| format!(",\"prefix\":{}", p)).unwrap_or_default()
        ));
    }
    std::fs::write(format!("{}.hits.jsonl", args.prefix), hj).unwrap();

    // machine-readable rollup for the orchestrator
    let (xtrace_result, xtrace_nt_result) = if xtrace_status == "completed" {
        (a3_leaks.to_string(), a3_leaks_nt.to_string())
    } else {
        ("NA".to_string(), "NA".to_string())
    };
    let result_gadget = b
        .meta
        .get("builder_gadget")
        .or_else(|| b.meta.get("gadget"))
        .map(|s| s.as_str())
        .unwrap_or("?");
    println!(
        "RESULT gadget={} k={} a1={} a1_nt={} xrows={} xrows_nt={} xtrace_status={} xtrace={} xtrace_nt={} w1flag={} w2flag={} w3flag={} w1_scanned={} w2_sampled={} w3_sampled={} n_features={} corr_selection=deterministic_strided null1={:.4} null2={:.4} null3={:.4}",
        result_gadget,
        b.meta.get("k").map(|s| s.as_str()).unwrap_or("?"),
        a1_leaks,
        a1_leaks_nontrivial,
        a2_leaks,
        a2_leaks_nt,
        xtrace_status,
        xtrace_result,
        xtrace_nt_result,
        a4_w1_flag,
        a4_w2_flag,
        a4_w3_flag,
        nfeat,
        w2_set.len(),
        w3_set.len(),
        nfeat,
        w1[null_idx].0,
        null_w2,
        null_w3
    );
}
