//! GF(2) linear algebra with coefficient tracking, shared by the C-view
//! exposure engine (`cview`).
//!
//! A [`TrackedBasis`] spans "sample vectors" of `words` u64 words. Every
//! inserted vector is a column `col` of some matrix R; the basis remembers,
//! for each basis vector, which combination of columns produced it. A column
//! that depends on the earlier ones yields a nullspace vector of R instead.
//!
//! The basis is kept in reduced row-echelon form (every pivot bit is set in
//! exactly one basis vector). The set of independent columns, the nullspace
//! vectors and the particular solutions returned by [`TrackedBasis::solve`]
//! do NOT depend on the elimination order: a column is independent iff it is
//! outside the span of the columns inserted before it (greedy, insertion
//! order); the null vector of a dependent column d is the unique `e_d + c_I`
//! with `R_I c_I = R_d` over the independent columns I; and a solved
//! right-hand side gets the unique solution supported on I. These are exactly
//! the vectors the historic `hmap_affine` exposure scan computed with its
//! non-reduced, lowest-bit-pivot basis, so the two implementations agree bit
//! for bit while this one can test membership word by word.

/// Bit `i` of a packed GF(2) vector.
#[inline]
pub fn bit(v: &[u64], i: usize) -> bool {
    (v[i / 64] >> (i % 64)) & 1 == 1
}

/// `dst ^= src`, word by word.
#[inline]
pub fn xor_into(dst: &mut [u64], src: &[u64]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d ^= s;
    }
}

/// Index of the lowest set bit, if any.
#[inline]
pub fn first_set(v: &[u64]) -> Option<usize> {
    v.iter()
        .enumerate()
        .find_map(|(wi, &w)| (w != 0).then(|| wi * 64 + w.trailing_zeros() as usize))
}

/// GF(2) basis of sample vectors with coefficient tracking (see module doc).
#[derive(Clone, Debug)]
pub struct TrackedBasis {
    words: usize,
    coef_words: usize,
    coef_bits: usize,
    /// Pivot bit of each basis vector, in insertion order.
    pivots: Vec<usize>,
    /// Sample part of each basis vector, `[k][words]`.
    samp: Vec<u64>,
    /// Coefficient part of each basis vector, `[k][coef_words]`.
    coef: Vec<u64>,
    /// Null vectors, `[j][coef_words]`, in the order their columns arrived.
    nulls: Vec<u64>,
    /// Independent columns, in insertion order.
    independent: Vec<usize>,
}

impl TrackedBasis {
    /// An empty basis over `words`-word sample vectors whose columns are
    /// numbered `0..coef_bits`.
    pub fn new(words: usize, coef_bits: usize) -> Self {
        Self {
            words,
            coef_words: coef_bits.div_ceil(64).max(1),
            coef_bits,
            pivots: Vec::new(),
            samp: Vec::new(),
            coef: Vec::new(),
            nulls: Vec::new(),
            independent: Vec::new(),
        }
    }

    pub fn words(&self) -> usize {
        self.words
    }
    pub fn coef_words(&self) -> usize {
        self.coef_words
    }
    pub fn rank(&self) -> usize {
        self.pivots.len()
    }
    pub fn nullity(&self) -> usize {
        self.nulls.len() / self.coef_words
    }
    /// The `j`-th null vector (columns combination summing to zero).
    pub fn null(&self, j: usize) -> &[u64] {
        &self.nulls[j * self.coef_words..(j + 1) * self.coef_words]
    }
    /// The independent columns, in insertion order.
    pub fn independent_columns(&self) -> &[usize] {
        &self.independent
    }

    /// Insert column `col` with sample vector `v`. Returns `true` when the
    /// column is independent of the columns inserted before it; otherwise a
    /// null vector is recorded and `false` is returned.
    pub fn insert(&mut self, v: &[u64], col: usize) -> bool {
        assert_eq!(v.len(), self.words, "sample vector width");
        assert!(col < self.coef_bits, "column {col} >= {}", self.coef_bits);
        let (w, cw) = (self.words, self.coef_words);
        let mut s = v.to_vec();
        let mut c = vec![0u64; cw];
        c[col / 64] |= 1u64 << (col % 64);
        for (i, &p) in self.pivots.iter().enumerate() {
            if bit(&s, p) {
                xor_into(&mut s, &self.samp[i * w..(i + 1) * w]);
                xor_into(&mut c, &self.coef[i * cw..(i + 1) * cw]);
            }
        }
        match first_set(&s) {
            None => {
                self.nulls.extend_from_slice(&c);
                false
            }
            Some(p) => {
                // Keep the basis reduced: clear bit p from every other vector.
                for i in 0..self.pivots.len() {
                    if bit(&self.samp[i * w..(i + 1) * w], p) {
                        xor_into(&mut self.samp[i * w..(i + 1) * w], &s);
                        xor_into(&mut self.coef[i * cw..(i + 1) * cw], &c);
                    }
                }
                self.pivots.push(p);
                self.samp.extend_from_slice(&s);
                self.coef.extend_from_slice(&c);
                self.independent.push(col);
                true
            }
        }
    }

    /// Solve `R c = v`. Returns `false` when `v` is outside the span;
    /// otherwise writes into `coef_out` the unique solution supported on the
    /// independent columns. `scratch` is reused between calls.
    pub fn solve(&self, v: &[u64], coef_out: &mut [u64], scratch: &mut Vec<u32>) -> bool {
        assert_eq!(v.len(), self.words, "sample vector width");
        let w = self.words;
        // Reduced form: the basis vectors to combine are exactly those whose
        // pivot bit is set in v.
        scratch.clear();
        for (i, &p) in self.pivots.iter().enumerate() {
            if bit(v, p) {
                scratch.push(i as u32);
            }
        }
        // Membership: v equals the XOR of the selected vectors. Check from the
        // last word down (pivots cluster in the low words, so a mismatch
        // usually shows up immediately).
        for wi in (0..w).rev() {
            let mut r = v[wi];
            for &i in scratch.iter() {
                r ^= self.samp[i as usize * w + wi];
            }
            if r != 0 {
                return false;
            }
        }
        let cw = self.coef_words;
        assert_eq!(coef_out.len(), cw, "coefficient width");
        coef_out.fill(0);
        for &i in scratch.iter() {
            xor_into(coef_out, &self.coef[i as usize * cw..(i as usize + 1) * cw]);
        }
        true
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/packing/gf2.rs"]
mod tests;
