use super::*;
use rand::{Rng, SeedableRng, rngs::StdRng};

type HistoricBasis = Vec<(usize, Vec<u64>, Vec<u64>)>;

/// The historic `hmap_affine` basis: not reduced, pivot = lowest set bit,
/// reduction in insertion order. Returns (independent cols, nulls).
fn historic_basis(cols: &[Vec<u64>], cw: usize) -> (HistoricBasis, Vec<Vec<u64>>) {
    let mut basis: Vec<(usize, Vec<u64>, Vec<u64>)> = Vec::new();
    let mut nulls = Vec::new();
    for (idx, col) in cols.iter().enumerate() {
        let mut samp = col.clone();
        let mut coef = vec![0u64; cw];
        coef[idx / 64] |= 1 << (idx % 64);
        for (p, bs, bc) in basis.iter() {
            if bit(&samp, *p) {
                xor_into(&mut samp, bs);
                xor_into(&mut coef, bc);
            }
        }
        match first_set(&samp) {
            Some(p) => basis.push((p, samp, coef)),
            None => nulls.push(coef),
        }
    }
    (basis, nulls)
}

fn historic_solve(basis: &[(usize, Vec<u64>, Vec<u64>)], v: &[u64], cw: usize) -> Option<Vec<u64>> {
    let mut samp = v.to_vec();
    let mut coef = vec![0u64; cw];
    for (p, bs, bc) in basis {
        if bit(&samp, *p) {
            xor_into(&mut samp, bs);
            xor_into(&mut coef, bc);
        }
    }
    first_set(&samp).is_none().then_some(coef)
}

fn combo(cols: &[Vec<u64>], coef: &[u64]) -> Vec<u64> {
    let mut acc = vec![0u64; cols[0].len()];
    for (i, c) in cols.iter().enumerate() {
        if bit(coef, i) {
            xor_into(&mut acc, c);
        }
    }
    acc
}

#[test]
fn bit_helpers() {
    let v = [0u64, 1 << 5];
    assert!(bit(&v, 69));
    assert!(!bit(&v, 5));
    assert_eq!(first_set(&v), Some(69));
    assert_eq!(first_set(&[0, 0]), None);
    let mut d = [3u64, 4];
    xor_into(&mut d, &[1, 4]);
    assert_eq!(d, [2, 0]);
}

/// Reduced and historic bases agree on independence, nulls and solutions,
/// over matrices with many dependencies (few distinct generators).
#[test]
fn reduced_basis_matches_historic_basis() {
    let mut rng = StdRng::seed_from_u64(0x6f2_0001);
    for trial in 0..200 {
        let words = 1 + trial % 3;
        let ncols = 1 + rng.random_range(0..90usize);
        let gens = 1 + rng.random_range(0..(words * 64).min(40));
        let g: Vec<Vec<u64>> = (0..gens)
            .map(|_| (0..words).map(|_| rng.random::<u64>()).collect())
            .collect();
        let cols: Vec<Vec<u64>> = (0..ncols)
            .map(|_| {
                let mut v = vec![0u64; words];
                for gv in &g {
                    if rng.random_bool(0.3) {
                        xor_into(&mut v, gv);
                    }
                }
                if rng.random_bool(0.05) {
                    v.fill(0);
                }
                v
            })
            .collect();
        let cw = ncols.div_ceil(64);
        let (hb, hn) = historic_basis(&cols, cw);
        let mut tb = TrackedBasis::new(words, ncols);
        let mut indep = Vec::new();
        for (i, c) in cols.iter().enumerate() {
            if tb.insert(c, i) {
                indep.push(i);
            }
        }
        assert_eq!(tb.independent_columns(), &indep[..]);
        assert_eq!(tb.rank(), hb.len());
        assert_eq!(tb.nullity(), hn.len());
        for (j, n) in hn.iter().enumerate() {
            assert_eq!(tb.null(j), &n[..], "null {j}");
            assert!(combo(&cols, n).iter().all(|&w| w == 0));
        }
        let mut scratch = Vec::new();
        for _ in 0..20 {
            let v: Vec<u64> = if rng.random_bool(0.5) {
                let mut v = vec![0u64; words];
                for c in &cols {
                    if rng.random_bool(0.5) {
                        xor_into(&mut v, c);
                    }
                }
                v
            } else {
                (0..words).map(|_| rng.random::<u64>()).collect()
            };
            let mut out = vec![0u64; tb.coef_words()];
            let ok = tb.solve(&v, &mut out, &mut scratch);
            let h = historic_solve(&hb, &v, cw);
            assert_eq!(ok, h.is_some());
            if let Some(h) = h {
                assert_eq!(out, h);
                assert_eq!(combo(&cols, &out), v);
            }
        }
    }
}

#[test]
fn zero_column_is_dependent_with_unit_null() {
    let mut tb = TrackedBasis::new(1, 3);
    assert!(tb.insert(&[5], 0));
    assert!(!tb.insert(&[0], 1));
    assert_eq!(tb.null(0), &[0b10]);
    assert!(!tb.insert(&[5], 2));
    assert_eq!(tb.null(1), &[0b101]);
}
