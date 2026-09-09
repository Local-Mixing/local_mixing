//! Compiler-private multivariate polynomials over GF(2) (square-free monomials).

use std::collections::BTreeSet;

/// Square-free monomial: sorted distinct wire indices. Empty = constant 1.
pub type Monomial = Vec<u16>;

/// Sparse polynomial over GF(2).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Poly {
    terms: BTreeSet<Monomial>,
}

impl Poly {
    pub fn zero() -> Self {
        Self {
            terms: BTreeSet::new(),
        }
    }

    pub fn one() -> Self {
        let mut p = Self::zero();
        p.terms.insert(vec![]);
        p
    }

    pub fn var(w: u16) -> Self {
        let mut p = Self::zero();
        p.terms.insert(vec![w]);
        p
    }

    pub fn terms(&self) -> impl Iterator<Item = &Monomial> {
        self.terms.iter()
    }

    pub fn degree(&self) -> usize {
        self.terms.iter().map(|m| m.len()).max().unwrap_or(0)
    }

    #[allow(dead_code)]
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    fn normalize_mono(mut m: Monomial) -> Monomial {
        m.sort_unstable();
        m.dedup();
        m
    }

    pub fn toggle_mono(&mut self, m: Monomial) {
        let m = Self::normalize_mono(m);
        if !self.terms.remove(&m) {
            self.terms.insert(m);
        }
    }

    pub fn add_assign(&mut self, other: &Self) {
        for m in &other.terms {
            self.toggle_mono(m.clone());
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        let mut out = self.clone();
        out.add_assign(other);
        out
    }

    pub fn mul(&self, other: &Self) -> Self {
        let mut out = Self::zero();
        for a in &self.terms {
            for b in &other.terms {
                let mut m = Vec::with_capacity(a.len() + b.len());
                m.extend_from_slice(a);
                m.extend_from_slice(b);
                out.toggle_mono(m);
            }
        }
        out
    }

    pub fn split_degree(&self, max_deg: usize) -> (Self, Self) {
        let mut low = Self::zero();
        let mut high = Self::zero();
        for m in &self.terms {
            if m.len() <= max_deg {
                low.terms.insert(m.clone());
            } else {
                high.terms.insert(m.clone());
            }
        }
        (low, high)
    }

    #[allow(dead_code)]
    pub fn eval(&self, bits: &[bool]) -> bool {
        let mut acc = false;
        for m in &self.terms {
            let mut mon = true;
            for &w in m {
                mon &= bits[w as usize];
            }
            acc ^= mon;
        }
        acc
    }

    pub fn term_count(&self) -> usize {
        self.terms.len()
    }
}

/// Sparse random quadratic on `ancillas`: O(|A|) terms (not dense Θ(|A|²)).
///
/// Always non-empty (at least one term) so carriers are never left as bare plaintext.
pub fn random_quadratic(ancillas: &[u16], rng: &mut impl rand::Rng) -> Poly {
    let mut p = Poly::zero();
    if ancillas.is_empty() {
        p.toggle_mono(vec![]);
        return p;
    }
    if rng.random_bool(0.5) {
        p.toggle_mono(vec![]);
    }
    for &w in ancillas {
        if rng.random_bool(0.5) {
            p.toggle_mono(vec![w]);
        }
    }
    let n = ancillas.len();
    let n_pairs = n.max(1);
    for _ in 0..n_pairs {
        if n < 2 {
            break;
        }
        let i = rng.random_range(0..n);
        let mut j = rng.random_range(0..n);
        while j == i {
            j = rng.random_range(0..n);
        }
        let (a, b) = if ancillas[i] < ancillas[j] {
            (ancillas[i], ancillas[j])
        } else {
            (ancillas[j], ancillas[i])
        };
        p.toggle_mono(vec![a, b]);
    }
    // Guarantee non-empty algebraic mask.
    if p.term_count() == 0 {
        p.toggle_mono(vec![ancillas[0]]);
    }
    debug_assert!(p.degree() <= 2);
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn mul_and_degree() {
        let p = Poly::var(0).mul(&Poly::var(1)).mul(&Poly::var(2));
        assert_eq!(p.degree(), 3);
    }

    #[test]
    fn sparse_is_linear_in_n() {
        let mut rng = StdRng::seed_from_u64(0);
        let anc: Vec<u16> = (0..40).collect();
        let q = random_quadratic(&anc, &mut rng);
        // O(n): constant + ≤n linears + ≤n pairs → ≤ 2n+1
        assert!(q.term_count() <= 2 * anc.len() + 1, "terms={}", q.term_count());
        assert!(q.degree() <= 2);
    }
}
