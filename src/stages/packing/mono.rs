//! Monomial keys: a bitmask over global wires, exactly a nonnegative Python
//! int (docs/STAGE4_RUST_DESIGN.md §4.5). Limb 0 is the least significant;
//! `Ord` is numeric (most significant limb first), the CPython hash is the
//! value mod 2^61 - 1, computed by Mersenne folding.
use std::cmp::Ordering;
use std::fmt;

/// CPython's `_PyHASH_MODULUS` on 64-bit builds.
pub const M61: u64 = (1u64 << 61) - 1;

/// A key whose CPython hash is known (nonnegative ints only).
pub trait PyIntKey: Copy + Eq {
    fn py_hash(&self) -> u64;
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Mono<const W: usize>(pub [u64; W]);

impl<const W: usize> Default for Mono<W> {
    fn default() -> Self {
        Mono([0; W])
    }
}

impl<const W: usize> Mono<W> {
    pub const ZERO: Mono<W> = Mono([0; W]);

    #[inline]
    pub fn bit(w: usize) -> Self {
        let mut m = [0u64; W];
        m[w >> 6] = 1u64 << (w & 63);
        Mono(m)
    }

    #[inline]
    pub fn or(&self, o: &Self) -> Self {
        let mut m = self.0;
        for (a, b) in m.iter_mut().zip(o.0.iter()) {
            *a |= *b;
        }
        Mono(m)
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|&l| l == 0)
    }

    #[inline]
    pub fn test(&self, w: usize) -> bool {
        (self.0[w >> 6] >> (w & 63)) & 1 == 1
    }

    /// Set bits in ascending order (wire ids).
    pub fn bits(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.iter().enumerate().flat_map(|(i, &limb)| {
            let mut l = limb;
            std::iter::from_fn(move || {
                if l == 0 {
                    None
                } else {
                    let b = l.trailing_zeros() as usize;
                    l &= l - 1;
                    Some(i * 64 + b)
                }
            })
        })
    }

    pub fn count_ones(&self) -> u32 {
        self.0.iter().map(|l| l.count_ones()).sum()
    }

    /// Parse a lowercase/uppercase hex integer (no prefix); `None` if it
    /// does not fit in W limbs.
    pub fn from_hex(s: &str) -> Option<Self> {
        let mut m = [0u64; W];
        for (i, c) in s.bytes().rev().enumerate() {
            let d = (c as char).to_digit(16)? as u64;
            if d == 0 {
                continue;
            }
            let bit = i * 4;
            if bit / 64 >= W {
                return None;
            }
            // 64 is a multiple of 4: a hex digit never straddles two limbs.
            m[bit / 64] |= d << (bit % 64);
        }
        Some(Mono(m))
    }
}

/// `8 * h mod M61` for `h < M61` (2^61 ≡ 1).
#[inline(always)]
fn rot8(h: u64) -> u64 {
    ((h << 3) & M61) | (h >> 58)
}

/// CPython's int hash of the little-endian limb array, folded from the most
/// significant limb: `h <- (h * 2^64 + limb) mod M61`, using 2^64 ≡ 8.
#[inline]
pub fn py_int_hash(limbs: &[u64]) -> u64 {
    let mut h = 0u64;
    for &limb in limbs.iter().rev() {
        let x = rot8(h) + (limb >> 61) + (limb & M61);
        h = (x & M61) + (x >> 61);
        if h >= M61 {
            h -= M61;
        }
    }
    h
}

impl<const W: usize> PyIntKey for Mono<W> {
    #[inline]
    fn py_hash(&self) -> u64 {
        py_int_hash(&self.0)
    }
}

impl PyIntKey for u64 {
    #[inline]
    fn py_hash(&self) -> u64 {
        py_int_hash(&[*self])
    }
}

impl<const W: usize> Ord for Mono<W> {
    fn cmp(&self, other: &Self) -> Ordering {
        for i in (0..W).rev() {
            match self.0[i].cmp(&other.0[i]) {
                Ordering::Equal => continue,
                o => return o,
            }
        }
        Ordering::Equal
    }
}

impl<const W: usize> PartialOrd for Mono<W> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<const W: usize> fmt::Debug for Mono<W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:x}")
    }
}

impl<const W: usize> fmt::LowerHex for Mono<W> {
    /// Python's `f"{k:x}"`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let top = (0..W).rev().find(|&i| self.0[i] != 0);
        match top {
            None => write!(f, "0"),
            Some(t) => {
                write!(f, "{:x}", self.0[t])?;
                for i in (0..t).rev() {
                    write!(f, "{:016x}", self.0[i])?;
                }
                Ok(())
            }
        }
    }
}

/// Number of u64 limbs for a state of `nw` wires: the smallest of
/// {1, 2, 4, 8, 16} with 64·W ≥ nw; `None` above 1024 wires.
pub fn mono_words(nw: usize) -> Option<usize> {
    [1usize, 2, 4, 8, 16].into_iter().find(|&w| 64 * w >= nw)
}

/// A computation generic over the monomial width.
pub trait MonoWidthFn<R> {
    fn call<const W: usize>(self) -> R;
}

/// Run `f` at the monomial width for `nw` wires (§4.5). Panics above 1024
/// wires; callers validate the width first (`mono_words`).
pub fn with_mono_width<R>(nw: usize, f: impl MonoWidthFn<R>) -> R {
    match mono_words(nw) {
        Some(1) => f.call::<1>(),
        Some(2) => f.call::<2>(),
        Some(4) => f.call::<4>(),
        Some(8) => f.call::<8>(),
        Some(16) => f.call::<16>(),
        _ => panic!("stage 4 supports at most 1024 state wires, got {nw}"),
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/packing/mono.rs"]
mod tests;
