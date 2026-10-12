//! Monomial bitset over the LOCAL variable universe of one window and the
//! synthesizer's PRNG (ported from `anf_synth`'s `mono.rs`).
//!
//! Bit `v` set = factor `x_v`; the all-zero monomial is the constant 1. The
//! width `W` (u64 words) is chosen per window from the size of its local
//! universe (`spec::LocalUniverse`), so a 128-variable window searches over
//! 2-word monomials whatever the state width. The derived `Ord` is
//! lexicographic from word 0 (as in `anf_synth`): any total order serves the
//! synthesizer's deterministic tie-breaks.
use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Mono<const W: usize>(pub [u64; W]);

impl<const W: usize> Default for Mono<W> {
    fn default() -> Self {
        Mono([0; W])
    }
}

impl<const W: usize> Mono<W> {
    pub const ONE: Mono<W> = Mono([0; W]);

    /// Number of variables this width holds.
    pub const VARS: u32 = (64 * W) as u32;

    #[inline]
    pub fn var(v: u32) -> Mono<W> {
        let mut m = [0u64; W];
        m[(v >> 6) as usize] |= 1u64 << (v & 63);
        Mono(m)
    }

    #[inline]
    pub fn contains(&self, v: u32) -> bool {
        (self.0[(v >> 6) as usize] >> (v & 63)) & 1 == 1
    }

    #[inline]
    pub fn with(&self, v: u32) -> Mono<W> {
        let mut m = self.0;
        m[(v >> 6) as usize] |= 1u64 << (v & 63);
        Mono(m)
    }

    #[inline]
    pub fn without(&self, v: u32) -> Mono<W> {
        let mut m = self.0;
        m[(v >> 6) as usize] &= !(1u64 << (v & 63));
        Mono(m)
    }

    #[inline]
    pub fn or(&self, o: &Mono<W>) -> Mono<W> {
        let mut m = self.0;
        for (a, b) in m.iter_mut().zip(o.0.iter()) {
            *a |= *b;
        }
        Mono(m)
    }

    #[inline]
    pub fn and(&self, o: &Mono<W>) -> Mono<W> {
        let mut m = self.0;
        for (a, b) in m.iter_mut().zip(o.0.iter()) {
            *a &= *b;
        }
        Mono(m)
    }

    /// `self \ o`
    #[inline]
    pub fn minus(&self, o: &Mono<W>) -> Mono<W> {
        let mut m = self.0;
        for (a, b) in m.iter_mut().zip(o.0.iter()) {
            *a &= !*b;
        }
        Mono(m)
    }

    /// `self` is a subset of `o`.
    #[inline]
    pub fn is_subset(&self, o: &Mono<W>) -> bool {
        self.0.iter().zip(o.0.iter()).all(|(a, b)| a & !b == 0)
    }

    #[inline]
    pub fn is_one(&self) -> bool {
        self.0.iter().all(|&x| x == 0)
    }

    #[inline]
    pub fn degree(&self) -> u32 {
        self.0.iter().map(|x| x.count_ones()).sum()
    }

    /// Variables in ascending order.
    pub fn vars(&self) -> MonoVars<W> {
        MonoVars { m: *self, word: 0 }
    }

    pub fn to_vec(&self) -> Vec<u32> {
        self.vars().collect()
    }

    pub fn from_vars(vs: &[u32]) -> Mono<W> {
        let mut m = Mono::ONE;
        for &v in vs {
            m = m.with(v);
        }
        m
    }

    /// The lowest variable of the monomial (`None` for the constant).
    pub fn first_var(&self) -> Option<u32> {
        self.vars().next()
    }
}

impl<const W: usize> fmt::Debug for Mono<W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_one() {
            return write!(f, "1");
        }
        let vs: Vec<String> = self.vars().map(|v| format!("x{v}")).collect();
        write!(f, "{}", vs.join("*"))
    }
}

pub struct MonoVars<const W: usize> {
    m: Mono<W>,
    word: usize,
}

impl<const W: usize> Iterator for MonoVars<W> {
    type Item = u32;
    #[inline]
    fn next(&mut self) -> Option<u32> {
        while self.word < W {
            let w = self.m.0[self.word];
            if w != 0 {
                let b = w.trailing_zeros();
                self.m.0[self.word] = w & (w - 1);
                return Some((self.word as u32) * 64 + b);
            }
            self.word += 1;
        }
        None
    }
}

/// Small deterministic PRNG (splitmix64): the synthesizer seeds it from
/// `(seed, window index, restart)` only, so a run is reproducible.
#[derive(Clone, Debug)]
pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut r = Rng(seed ^ 0x9E37_79B9_7F4A_7C15);
        for _ in 0..4 {
            r.next_u64();
        }
        r
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[inline]
    pub fn below(&mut self, n: u64) -> u64 {
        if n <= 1 { 0 } else { self.next_u64() % n }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/unpacking/mono.rs"]
mod tests;
