//! Sorted GF(2) monomial arithmetic and polynomial presentation.
use std::cmp::Ordering as CmpOrdering;
// Polynomial representation of circuit
pub type Monomial = u64;
pub type Polynomial = Vec<Monomial>;

pub fn polynomial_from_terms<I>(terms: I) -> Polynomial
where
    I: IntoIterator<Item = Monomial>,
{
    let mut terms: Vec<Monomial> = terms.into_iter().collect();
    normalize_polynomial(&mut terms);
    terms
}

pub fn normalize_polynomial(poly: &mut Polynomial) {
    poly.sort_unstable();

    let mut write = 0usize;
    let mut read = 0usize;
    while read < poly.len() {
        let m = poly[read];
        let mut count = 1usize;
        read += 1;
        while read < poly.len() && poly[read] == m {
            count += 1;
            read += 1;
        }
        if count % 2 == 1 {
            poly[write] = m;
            write += 1;
        }
    }
    poly.truncate(write);
}

pub fn substitute_input_negation(poly: &mut Polynomial, w: usize) {
    // Substituting x_w -> x_w + 1 toggles, for every monomial containing x_w,
    // the same monomial with x_w cleared. The rests are pairwise distinct
    // (rest | bit reconstructs its unique source), so the sequential
    // binary-search toggles the old implementation performed are exactly one
    // sorted symmetric-difference merge.
    let bit = 1u64 << w;
    let mut rests: Vec<Monomial> = poly
        .iter()
        .filter(|&&m| m & bit != 0)
        .map(|&m| m & !bit)
        .collect();
    if rests.is_empty() {
        return;
    }
    // Rests inherit the poly's sort order (the cleared bit is common to every
    // source monomial); sort defensively anyway — it is a no-op then.
    rests.sort_unstable();
    poly_xor_assign(poly, rests);
}

pub(super) fn toggle_monomial(poly: &mut Polynomial, m: Monomial) {
    match poly.binary_search(&m) {
        Ok(pos) => {
            poly.remove(pos);
        }
        Err(pos) => {
            poly.insert(pos, m);
        }
    }
}

pub(super) fn poly_xor_assign(poly: &mut Polynomial, terms: Polynomial) {
    let old = std::mem::take(poly);
    let mut merged = Vec::with_capacity(old.len().max(terms.len()));
    let mut i = 0usize;
    let mut j = 0usize;

    while i < old.len() && j < terms.len() {
        match old[i].cmp(&terms[j]) {
            CmpOrdering::Less => {
                merged.push(old[i]);
                i += 1;
            }
            CmpOrdering::Greater => {
                merged.push(terms[j]);
                j += 1;
            }
            CmpOrdering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    merged.extend_from_slice(&old[i..]);
    merged.extend_from_slice(&terms[j..]);
    *poly = merged;
}

#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn poly_and_not(poly_1: &Polynomial, poly_2: &Polynomial) -> Polynomial {
    let mut terms = Vec::with_capacity(poly_1.len() * (poly_2.len() + 1));
    for &m1 in poly_1 {
        terms.push(m1);
        for &m2 in poly_2 {
            terms.push(m1 | m2);
        }
    }
    polynomial_from_terms(terms)
}

/// Allocation-reusing form of `poly_and_not`: same raw term stream and the
/// same sort+cancel normalization, written into a caller-owned scratch vec.
pub(super) fn poly_and_not_into(poly_1: &[Monomial], poly_2: &[Monomial], out: &mut Vec<Monomial>) {
    out.clear();
    out.reserve(poly_1.len() * (poly_2.len() + 1));
    for &m1 in poly_1 {
        out.push(m1);
        for &m2 in poly_2 {
            out.push(m1 | m2);
        }
    }
    normalize_polynomial(out);
}

/// Allocation-reusing form of `poly_xor_assign`: merges the sorted symmetric
/// difference of `a` and `b` into a caller-owned scratch vec.
pub(super) fn poly_xor_merge_into(a: &[Monomial], b: &[Monomial], out: &mut Vec<Monomial>) {
    out.clear();
    let mut i = 0usize;
    let mut j = 0usize;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            CmpOrdering::Less => {
                out.push(a[i]);
                i += 1;
            }
            CmpOrdering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            CmpOrdering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
}
pub fn monomial_degree(m: u64) -> u32 {
    m.count_ones()
}

fn mono_compressed_str(m: u64, n: usize) -> String {
    if m == 0 {
        return "I".into();
    }
    (0..n)
        .filter(|&i| (m >> i) & 1 == 1)
        .map(|i| format!("{}", i))
        .collect::<Vec<_>>()
        .join("•")
}

pub fn poly_to_compressed_str(poly: &Polynomial, n: usize) -> String {
    if poly.is_empty() {
        return "i".into();
    }
    let mut terms: Vec<u64> = poly.iter().copied().collect();
    terms.sort_by_key(|&m| (monomial_degree(m), m));
    terms
        .iter()
        .map(|&m| mono_compressed_str(m, n))
        .collect::<Vec<_>>()
        .join(" ")
}
