use super::*;
use crate::circuit::xgate::XGate;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Reference: the exposure loop of `hmap_affine` at 492564f0, transcribed with
// its materialized `[batch][cut][wire]` snapshots, `HashMap` dedup and
// non-reduced basis. The engine must agree with it on every output.
// ---------------------------------------------------------------------------

struct RefOut {
    rowinfo: Vec<(usize, usize)>, // (gate, wire)
    best: Vec<i64>,
    witness: Vec<Option<(usize, Vec<usize>)>>,
    allcuts: Vec<(usize, usize, u32)>, // (row, cut position, weight)
    approx: usize,
}

fn ref_snapshots(gates: &[XGate], init: &[u64], idx: &[usize]) -> Vec<Vec<u64>> {
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

#[allow(clippy::too_many_arguments)]
fn reference_scan(
    c: &[XGate],
    g: &[XGate],
    n: usize,
    b: usize,
    seed: u64,
    i_idx: &[usize],
    j_idx: &[usize],
    allcuts_on: bool,
    witness_on: bool,
) -> RefOut {
    let nw_c = (max_wire(c) as usize + 1).max(n);
    let nw_g = (max_wire(g) as usize + 1).max(n);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut cs: Vec<Vec<Vec<u64>>> = Vec::new();
    let mut gs: Vec<Vec<Vec<u64>>> = Vec::new();
    for _ in 0..b {
        let x: Vec<u64> = (0..n).map(|_| rng.random::<u64>()).collect();
        let mut ic = vec![0u64; nw_c];
        let mut ig = vec![0u64; nw_g];
        ic[..n].copy_from_slice(&x);
        ig[..n].copy_from_slice(&x);
        cs.push(ref_snapshots(c, &ic, i_idx));
        gs.push(ref_snapshots(g, &ig, j_idx));
    }
    let words = b;
    let allzero = |v: &[u64]| v.iter().all(|&w| w == 0);
    let mut rowinfo: Vec<(usize, usize, usize)> = Vec::new();
    for (ri, &i) in i_idx.iter().enumerate() {
        if i == 0 {
            continue;
        }
        let t = c[i - 1].target as usize;
        if t < n {
            rowinfo.push((ri, i - 1, t));
        }
    }
    let mut best: Vec<i64> = vec![-1; rowinfo.len()];
    let mut allcuts: Vec<(usize, usize, u32)> = Vec::new();
    let mut witness: Vec<Option<(usize, Vec<usize>)>> = vec![None; rowinfo.len()];
    let mut approx = 0usize;
    for (cj, _) in j_idx.iter().enumerate() {
        let mut reps: Vec<Vec<u64>> = Vec::new();
        let mut rep_wires: Vec<usize> = Vec::new();
        let mut seen: HashMap<Vec<u64>, usize> = HashMap::new();
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
        let cw = (m + 1).div_ceil(64);
        let mut basis: Vec<(usize, Vec<u64>, Vec<u64>)> = Vec::new();
        let mut nulls: Vec<Vec<u64>> = Vec::new();
        {
            let mut add =
                |mut samp: Vec<u64>, idx: usize| {
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
            if !allcuts_on && best[rowi] >= 0 && best[rowi] <= 1 {
                continue;
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
                continue;
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
            if allcuts_on {
                allcuts.push((rowi, j_idx[cj], w));
            }
            if best[rowi] < 0 || (w as i64) < best[rowi] {
                best[rowi] = w as i64;
                if witness_on {
                    let mut wires = Vec::new();
                    for idx in 0..m {
                        if (argmin[idx / 64] >> (idx % 64)) & 1 == 1 {
                            wires.push(rep_wires[idx]);
                        }
                    }
                    witness[rowi] = Some((j_idx[cj], wires));
                }
            }
        }
    }
    RefOut {
        rowinfo: rowinfo.iter().map(|&(_, g, t)| (g, t)).collect(),
        best,
        witness,
        allcuts,
        approx,
    }
}

fn assert_matches_reference(rep: &ExposureReport, r: &RefOut) {
    let rows: Vec<(usize, usize)> = rep.rows.iter().map(|x| (x.gate, x.wire)).collect();
    assert_eq!(rows, r.rowinfo, "rows");
    assert_eq!(rep.best, r.best, "best");
    assert_eq!(rep.approx, r.approx, "approx");
    let allcuts: Vec<(usize, usize, u32)> = rep
        .allcuts
        .iter()
        .map(|&(row, cut, w)| (row as usize, cut, w))
        .collect();
    assert_eq!(allcuts, r.allcuts, "allcuts");
    let wit: Vec<Option<(usize, Vec<usize>)>> = rep
        .witness
        .iter()
        .map(|o| {
            o.as_ref()
                .map(|(c, ws)| (*c, ws.iter().map(|&w| w as usize).collect()))
        })
        .collect();
    assert_eq!(wit, r.witness, "witness");
}

// ---------------------------------------------------------------------------
// Circuit builders
// ---------------------------------------------------------------------------

fn rand_gate(rng: &mut StdRng, lo: usize, hi: usize, max_ctrls: usize) -> XGate {
    loop {
        let t = rng.random_range(lo..hi) as u16;
        let lits: Vec<(u16, bool)> = (0..rng.random_range(0..=max_ctrls))
            .map(|_| (rng.random_range(lo..hi) as u16, rng.random_bool(0.5)))
            .filter(|l| l.0 != t)
            .collect();
        if let Some(mut g) = XGate::conj(t, lits) {
            g.comp = rng.random_bool(0.2);
            return g;
        }
    }
}

fn aux_gate(rng: &mut StdRng, target: u16, lo: usize, hi: usize) -> XGate {
    loop {
        let lits: Vec<(u16, bool)> = (0..rng.random_range(1..=2))
            .map(|_| (rng.random_range(lo..hi) as u16, rng.random_bool(0.5)))
            .filter(|l| l.0 != target)
            .collect();
        if let Some(mut g) = XGate::conj(target, lits) {
            g.comp = rng.random_bool(0.2);
            return g;
        }
    }
}

/// C on n wires; P = C's gates interleaved with aux traffic on wires
/// n..nw: copies (CNOT aux ^= data) and complements of data, linear
/// combinations, constants (NOT on a clean aux wire), nonlinear junk, and
/// occasional uncomputation. With `linear`, C is CNOT/NOT only and the aux
/// traffic linear, so every value is affine in x and the many distinct aux
/// combinations give nullity above the cap. Gives complement classes, constants, many
/// nullspace vectors and genuinely exposed rows.
fn build_pair(
    rng: &mut StdRng,
    n: usize,
    aux: usize,
    c_len: usize,
    linear: bool,
) -> (Vec<XGate>, Vec<XGate>) {
    let max_ctrls = if linear { 1 } else { 2 };
    let c: Vec<XGate> = (0..c_len)
        .map(|_| rand_gate(rng, 0, n, max_ctrls))
        .collect();
    let nw = n + aux;
    let mut p = Vec::new();
    for g in &c {
        p.push(g.clone());
        for _ in 0..rng.random_range(0..if linear { 8 } else { 4 }) {
            let a = rng.random_range(n..nw) as u16;
            let kinds = if linear { 4 } else { 6 };
            let gate = match rng.random_range(0..kinds) {
                0 | 1 => XGate::cnot(a, rng.random_range(0..n) as u16),
                2 => XGate::x_gate(a),
                3 => {
                    let src = rng.random_range(n..nw) as u16;
                    if src == a {
                        continue;
                    }
                    XGate::cnot(a, src)
                }
                // nonlinear: controls anywhere (data or aux), target on aux
                4 => aux_gate(rng, a, 0, nw),
                // nonlinear junk on aux only
                _ => aux_gate(rng, a, n, nw),
            };
            p.push(gate);
        }
    }
    (c, p)
}

fn scan(
    c: &[XGate],
    p: &[XGate],
    n: usize,
    b: usize,
    seed: u64,
    cuts: &[usize],
    o: ExposureOptions,
) -> ExposureReport {
    let s = HonestSamples::draw(n, b, seed);
    let prefixes = prefix_indices(c.len(), 1);
    let rows = born_rows(c, n, &prefixes, &s);
    let nw = (max_wire(p) as usize + 1).max(n);
    exposure_scan(&rows, p, nw, &s, cuts, &o)
}

fn tsvs(rep: &ExposureReport) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut e = Vec::new();
    rep.write_exposure_tsv(&mut e).unwrap();
    let mut a = Vec::new();
    if rep.options.collect_allcuts {
        rep.write_allcuts_tsv(&mut a).unwrap();
    }
    let mut w = Vec::new();
    if rep.options.collect_witness {
        rep.write_witness_tsv(&mut w).unwrap();
    }
    (e, a, w)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn matches_historic_reference_on_random_pairs() {
    let mut rng = StdRng::seed_from_u64(0xc71e_0001);
    let mut saw_visible = 0;
    let mut saw_null = 0;
    let mut saw_settled_skip = 0;
    for trial in 0..24 {
        let linear = trial % 3 == 2;
        let n = if linear { 5 + trial % 2 } else { 3 + trial % 5 };
        let aux = if linear {
            48
        } else {
            [2, 6, 24, 40][trial % 4]
        };
        let (c, p) = build_pair(&mut rng, n, aux, 10 + trial, linear);
        let b = 2 + trial % 3;
        let seed = rng.random::<u64>();
        let i_idx = prefix_indices(c.len(), 1);
        let cuts: Vec<usize> = if trial % 2 == 0 {
            prefix_indices(p.len(), 1)
        } else {
            let mut v: Vec<usize> = (0..12).map(|_| rng.random_range(0..=p.len())).collect();
            v.sort_unstable();
            v.dedup();
            v
        };
        for (allc, wit) in [(true, true), (false, true), (false, false), (true, false)] {
            let r = reference_scan(&c, &p, n, b, seed, &i_idx, &cuts, allc, wit);
            let rep = scan(
                &c,
                &p,
                n,
                b,
                seed,
                &cuts,
                ExposureOptions::historic(allc, wit),
            );
            assert_matches_reference(&rep, &r);
            saw_visible += rep.visible();
            saw_null += r.approx;
            if allc {
                let r2 = reference_scan(&c, &p, n, b, seed, &i_idx, &cuts, false, wit);
                saw_settled_skip += (r.approx != r2.approx) as usize;
            }
        }
    }
    assert!(saw_visible > 0, "the fixture exercises visible rows");
    assert!(saw_null > 0, "the fixture exercises nullity > 14");
    assert!(
        saw_settled_skip > 0,
        "skip_settled changes the approximate count"
    );
}

#[test]
fn complement_pair_collapses() {
    // wire 0 = v, wire 1 = !v, wire 2 = v: one class, lowest wire wins.
    let v = [0b1010u64, 7];
    let cols = [v[0], v[1], !v[0], !v[1], v[0], v[1]];
    let cc = CutColumns::build(&cols, 3, 2);
    assert_eq!(cc.reps(), 1);
    let fit = cc.predict(&[!v[0], !v[1]]).expect("complement is affine");
    assert_eq!(fit.weight, 1);
    assert_eq!(fit.support, vec![0]);
}

#[test]
fn constant_value_tstar_zero() {
    // C: w2 = x0 (ancilla), then w0 ^= w2 -> w0 = 0: a constant born value.
    let c = vec![XGate::cnot(2, 0), XGate::cnot(0, 2)];
    let n = 2;
    let s = HonestSamples::draw(n, 3, 5);
    let rows = born_rows(&c, n, &prefix_indices(c.len(), 1), &s);
    assert_eq!(rows.len(), 1, "only the target < n row");
    assert_eq!(
        rows.rows[0],
        BornRow {
            prefix: 2,
            gate: 1,
            wire: 0
        }
    );
    let p = vec![XGate::cnot(1, 0)];
    let rep = exposure_scan(
        &rows,
        &p,
        2,
        &s,
        &[0, 1],
        &ExposureOptions::historic(true, true),
    );
    assert_eq!(rep.best, vec![0]);
    assert_eq!(rep.allcuts, vec![(0, 0, 0), (0, 1, 0)]);
    assert_eq!(rep.witness[0], Some((0, vec![])));
}

#[test]
fn nullity_le_14_exact_minimum() {
    // columns x0, x1, x0^x1: the particular solution of x0^x1 reads 2 wires,
    // the coset minimum reads 1 (wire 2).
    let x0 = [0x0123_4567_89ab_cdeeu64];
    let x1 = [0x0fed_cba9_8765_4320u64];
    let cols = [x0[0], x1[0], x0[0] ^ x1[0]];
    let cc = CutColumns::build(&cols, 3, 1);
    assert_eq!(cc.nullity(), 1);
    assert!(!cc.over_cap());
    let fit = cc.predict(&[x0[0] ^ x1[0]]).unwrap();
    assert_eq!(
        fit,
        CutFit {
            weight: 1,
            support: vec![2]
        }
    );
    assert!(cc.predict(&[x0[0] & x1[0]]).is_none());
}

#[test]
fn nullity_gt_14_counts_approx() {
    // 5 generators (bit 0 clear: no complementing), all 31 nonzero
    // combinations as wires (wire i = combination i + 1): nullity 26 > 14.
    let mut rng = StdRng::seed_from_u64(9);
    let gens: Vec<u64> = (0..5).map(|_| rng.random::<u64>() & !1).collect();
    let cols: Vec<u64> = (1u32..32)
        .map(|mask| {
            (0..5)
                .filter(|i| mask >> i & 1 == 1)
                .fold(0u64, |a, i| a ^ gens[i])
        })
        .collect();
    let cc = CutColumns::build(&cols, cols.len(), 1);
    assert!(cc.nullity() > NULLITY_CAP, "nullity {}", cc.nullity());
    assert!(cc.over_cap());
    // Without minimization the particular solution (on the independent
    // generators) is kept even though a single wire would do.
    let target = gens[0] ^ gens[1];
    let fit = cc.predict(&[target]).unwrap();
    assert_eq!(fit.weight, 2);
    assert_eq!(fit.support, vec![0, 1]);

    // assemble counts it once per (row, cut) visited; skip_settled drops
    // visits of rows already at weight <= 1.
    let rows = CRows {
        rows: vec![
            BornRow {
                prefix: 1,
                gate: 0,
                wire: 0,
            },
            BornRow {
                prefix: 2,
                gate: 1,
                wire: 0,
            },
        ],
        batches: 1,
        vecs: vec![target, gens[2]],
    };
    let mk = |cut| {
        Arc::new(CutResult {
            cut,
            over_cap: true,
            fits: vec![
                (
                    0,
                    CutFit {
                        weight: 2,
                        support: vec![0, 1],
                    },
                ),
                (
                    1,
                    CutFit {
                        weight: 1,
                        support: vec![2],
                    },
                ),
            ],
        })
    };
    let per = vec![mk(3), mk(5)];
    let skip = assemble(
        &rows,
        &[3, 5],
        &per,
        &ExposureOptions::historic(false, false),
    );
    assert_eq!(skip.approx, 3);
    let all = assemble(
        &rows,
        &[3, 5],
        &per,
        &ExposureOptions::historic(true, false),
    );
    assert_eq!(all.approx, 4);
    assert_eq!(all.allcuts.len(), 4);
}

#[test]
fn earliest_cut_wins_tie() {
    let rows = CRows {
        rows: vec![BornRow {
            prefix: 1,
            gate: 0,
            wire: 0,
        }],
        batches: 1,
        vecs: vec![1],
    };
    let r = |cut, w: u32, s: Vec<u32>| {
        Arc::new(CutResult {
            cut,
            over_cap: false,
            fits: vec![(
                0,
                CutFit {
                    weight: w,
                    support: s,
                },
            )],
        })
    };
    let per = vec![
        r(4, 3, vec![1, 2, 3]),
        r(7, 3, vec![4, 5, 6]),
        r(9, 2, vec![8, 9]),
    ];
    let rep = assemble(
        &rows,
        &[4, 7, 9],
        &per,
        &ExposureOptions::historic(true, true),
    );
    assert_eq!(rep.best, vec![2]);
    assert_eq!(rep.witness[0], Some((9, vec![8, 9])));
    let rep = assemble(
        &rows,
        &[4, 7],
        &per[..2],
        &ExposureOptions::historic(true, true),
    );
    assert_eq!(rep.witness[0], Some((4, vec![1, 2, 3])));
}

#[test]
fn cut_file_parse_rules() {
    assert_eq!(
        parse_cut_file("5\n\n  3 \n9\n5\n0\n", 8).unwrap(),
        vec![0, 3, 5]
    );
    assert_eq!(parse_cut_file("", 8).unwrap(), Vec::<usize>::new());
    assert_eq!(parse_cut_file("8\n", 8).unwrap(), vec![8]);
    let e = parse_cut_file("1\nx\n", 8).unwrap_err();
    assert_eq!(e.line, 2);
    assert!(parse_cut_file("-1\n", 8).is_err());
    assert_eq!(prefix_indices(10, 4), vec![0, 4, 8, 10]);
    assert_eq!(prefix_indices(8, 4), vec![0, 4, 8]);
    assert_eq!(prefix_indices(0, 3), vec![0]);
    assert_eq!(prefix_indices(3, 1), vec![0, 1, 2, 3]);
}

#[test]
fn samples_pinned_seed_12345() {
    let s = HonestSamples::draw(3, 2, DEFAULT_SEED);
    // The draw order is batch-major, wire-minor, one u64 per (batch, wire).
    let mut rng = StdRng::seed_from_u64(12345);
    let flat: Vec<u64> = (0..6).map(|_| rng.random::<u64>()).collect();
    assert_eq!(s.batch(0), &flat[0..3]);
    assert_eq!(s.batch(1), &flat[3..6]);
    // Pinned values (rand 0.9 StdRng = ChaCha12): a dependency bump that
    // changes the stream changes every C-view and must be caught here.
    assert_eq!(s.batch(0)[0], PINNED_12345[0]);
    assert_eq!(s.batch(0)[1], PINNED_12345[1]);
    assert_eq!(s.batch(1)[2], PINNED_12345[2]);
    assert_eq!(
        CviewParams::default(),
        CviewParams {
            batches: 96,
            seed: 12345
        }
    );
}

const PINNED_12345: [u64; 3] = [
    5643454701289460054,
    16290607951801111627,
    15959174902109986018,
];

#[test]
fn memo_equals_full_scan() {
    let mut rng = StdRng::seed_from_u64(0xc71e_0002);
    let (c, p) = build_pair(&mut rng, 5, 24, 40, false);
    let n = 5;
    let s = HonestSamples::draw(n, 3, 77);
    let rows = born_rows(&c, n, &prefix_indices(c.len(), 1), &s);
    let nw = (max_wire(&p) as usize + 1).max(n);
    let all = prefix_indices(p.len(), 1);
    let full = CutOracle::new(&p, nw, &rows, &s).results(&all);
    // Small checkpoint interval, queries out of order and repeated: rewinds.
    let mut o = CutOracle::new(&p, nw, &rows, &s).with_checkpoint_every(7);
    let late: Vec<usize> = all.iter().copied().filter(|x| x % 3 == 0).rev().collect();
    let _ = o.results(&late);
    let _ = o.results(&[all.len() / 2, 1, all.len() / 2]);
    let memo = o.results(&all);
    assert_eq!(o.memoized(), all.len());
    for (a, b) in full.iter().zip(&memo) {
        assert_eq!(**a, **b);
    }
    let opts = ExposureOptions::historic(true, true);
    assert_eq!(
        assemble(&rows, &all, &full, &opts),
        exposure_scan(&rows, &p, nw, &s, &all, &opts)
    );
}

#[test]
fn parallel_equals_serial() {
    let mut rng = StdRng::seed_from_u64(0xc71e_0003);
    let (c, p) = build_pair(&mut rng, 6, 40, 30, true);
    let n = 6;
    let s = HonestSamples::draw(n, 2, 1);
    let rows = born_rows(&c, n, &prefix_indices(c.len(), 1), &s);
    let nw = (max_wire(&p) as usize + 1).max(n);
    let cuts = prefix_indices(p.len(), 1);
    let par = CutOracle::new(&p, nw, &rows, &s).results(&cuts);
    let ser = CutOracle::new(&p, nw, &rows, &s)
        .with_parallel(false)
        .results(&cuts);
    for (a, b) in par.iter().zip(&ser) {
        assert_eq!(**a, **b);
    }
    let o = ExposureOptions::historic(false, true);
    assert_eq!(
        tsvs(&assemble(&rows, &cuts, &par, &o)),
        tsvs(&assemble(&rows, &cuts, &ser, &o))
    );
}

#[test]
fn tsv_formats_and_summary() {
    let rows = CRows {
        rows: vec![
            BornRow {
                prefix: 1,
                gate: 0,
                wire: 3,
            },
            BornRow {
                prefix: 2,
                gate: 1,
                wire: 0,
            },
        ],
        batches: 1,
        vecs: vec![1, 2],
    };
    let per = vec![Arc::new(CutResult {
        cut: 6,
        over_cap: false,
        fits: vec![(
            1,
            CutFit {
                weight: 2,
                support: vec![4, 9],
            },
        )],
    })];
    let rep = assemble(&rows, &[6], &per, &ExposureOptions::historic(true, true));
    let (e, a, w) = tsvs(&rep);
    assert_eq!(
        String::from_utf8(e).unwrap(),
        "# gate\twire\ttstar (-1 = no exact affine predictor at any sampled cut)\n0\t3\t-1\n1\t0\t2\n"
    );
    assert_eq!(
        String::from_utf8(a).unwrap(),
        "# gate\twire\tcut\tweight (every cut where the value is exactly affine)\n1\t0\t6\t2\n"
    );
    assert_eq!(
        String::from_utf8(w).unwrap(),
        "# gate\twire\ttstar\tcut\tG-wires (predictor support)\n1\t0\t2\t6\t4,9\n"
    );
    assert_eq!(
        rep.summary_lines(),
        vec![
            "[exposure] 2 born values, 1 visible (50.0%), 0 coset-minimizations skipped (nullity>14)"
                .to_string(),
            "[exposure] t*=  2:     1 values ( 50.0%)  cumulative  50.0%".to_string(),
        ]
    );
    let none = assemble(&rows, &[6], &per, &ExposureOptions::historic(false, false));
    assert!(none.write_witness_tsv(&mut Vec::new()).is_err());
    assert!(none.write_allcuts_tsv(&mut Vec::new()).is_err());
}

#[test]
fn output_gates_and_floor_test() {
    let c = vec![
        XGate::cnot(0, 1),
        XGate::cnot(1, 0),
        XGate::cnot(0, 1),
        XGate::cnot(3, 0),
    ];
    let outs = output_gates(&c, 2);
    let mut v: Vec<usize> = outs.iter().copied().collect();
    v.sort_unstable();
    assert_eq!(v, vec![1, 2]);
    let rep = ExposureReport {
        rows: (0..3)
            .map(|g| BornRow {
                prefix: g + 1,
                gate: g,
                wire: 0,
            })
            .collect(),
        best: vec![1, -1, 0],
        witness: vec![None; 3],
        allcuts: vec![],
        approx: 0,
        options: ExposureOptions::historic(false, false),
    };
    assert_eq!(rep.nonoutput_visible(&outs), vec![0]);
}

#[test]
fn born_rows_match_lane_simulation() {
    let mut rng = StdRng::seed_from_u64(0xc71e_0004);
    let (c, _) = build_pair(&mut rng, 6, 2, 50, false);
    let s = HonestSamples::draw(6, 3, 4);
    let prefixes = vec![0, 3, 3, 10, 50];
    let rows = born_rows(&c, 6, &prefixes, &s);
    assert_eq!(rows.len(), 4, "prefix 0 has no row; duplicates repeat");
    for (r, row) in rows.rows.iter().enumerate() {
        for b in 0..3 {
            let mut st = vec![0u64; 6];
            st.copy_from_slice(s.batch(b));
            for g in &c[..row.prefix] {
                g.apply_lanes(&mut st);
            }
            assert_eq!(rows.vec(r)[b], st[row.wire]);
        }
    }
}
