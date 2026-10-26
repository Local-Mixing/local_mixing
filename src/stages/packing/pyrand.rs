//! CPython `random.Random` (MT19937, `init_by_array` seeding, `getrandbits`
//! word order), bit-exact for nonnegative integer seeds
//! (docs/STAGE4_RUST_DESIGN.md §4.3; golden vectors in Appendix A).
//!
//! Users: the packer and link census (seed 20261004), `verify_equiv` (77),
//! `anf_compose` spot checks (20261004, 256-bit draws) and the
//! `anf_dump_eval` evaluation check (1, 1024-bit draws).

const N: usize = 624;
const M: usize = 397;

pub struct PyRandom {
    mt: [u32; N],
    idx: usize,
}

impl PyRandom {
    /// `random.Random(seed)` for an int seed: key = 32-bit little-endian words
    /// of `|seed|` (one zero word for seed 0), then `init_by_array`.
    pub fn new(seed: u64) -> Self {
        let key: Vec<u32> = if seed >> 32 == 0 {
            vec![seed as u32]
        } else {
            vec![seed as u32, (seed >> 32) as u32]
        };
        Self::from_key(&key)
    }

    pub fn from_key(key: &[u32]) -> Self {
        let mut r = PyRandom { mt: [0; N], idx: N };
        r.init_genrand(19_650_218);
        let mt = &mut r.mt;
        let (mut i, mut j) = (1usize, 0usize);
        let klen = key.len().max(1);
        for _ in 0..N.max(klen) {
            let prev = mt[i - 1] ^ (mt[i - 1] >> 30);
            mt[i] = (mt[i] ^ prev.wrapping_mul(1_664_525))
                .wrapping_add(*key.get(j).unwrap_or(&0))
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
            if j >= klen {
                j = 0;
            }
        }
        for _ in 0..N - 1 {
            let prev = mt[i - 1] ^ (mt[i - 1] >> 30);
            mt[i] = (mt[i] ^ prev.wrapping_mul(1_566_083_941)).wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
        }
        mt[0] = 0x8000_0000;
        r
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..N {
            let prev = self.mt[i - 1] ^ (self.mt[i - 1] >> 30);
            self.mt[i] = 1_812_433_253u32.wrapping_mul(prev).wrapping_add(i as u32);
        }
        self.idx = N;
    }

    fn twist(&mut self) {
        const UPPER: u32 = 0x8000_0000;
        const LOWER: u32 = 0x7fff_ffff;
        const MATRIX_A: u32 = 0x9908_b0df;
        let mt = &mut self.mt;
        for kk in 0..N {
            let y = (mt[kk] & UPPER) | (mt[(kk + 1) % N] & LOWER);
            let mag = if y & 1 == 1 { MATRIX_A } else { 0 };
            mt[kk] = mt[(kk + M) % N] ^ (y >> 1) ^ mag;
        }
        self.idx = 0;
    }

    pub fn genrand_u32(&mut self) -> u32 {
        if self.idx >= N {
            self.twist();
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `getrandbits(k)` as 32-bit words, least significant first; a partial
    /// top word keeps the output's high bits (`u32 >> (32 - r)`).
    pub fn getrandbits_words(&mut self, k: usize) -> Vec<u32> {
        assert!(k > 0, "getrandbits(0)");
        let words = (k - 1) / 32 + 1;
        let mut out = Vec::with_capacity(words);
        let mut left = k;
        for _ in 0..words {
            let mut r = self.genrand_u32();
            if left < 32 {
                r >>= 32 - left;
            }
            out.push(r);
            left = left.saturating_sub(32);
        }
        out
    }

    /// `getrandbits(64 * L)` as L little-endian u64 lanes.
    pub fn getrandbits_lanes<const L: usize>(&mut self) -> [u64; L] {
        let mut out = [0u64; L];
        for lane in out.iter_mut() {
            let lo = self.genrand_u32() as u64;
            let hi = self.genrand_u32() as u64;
            *lane = lo | (hi << 32);
        }
        out
    }

    /// `getrandbits(1)`: the top bit of one output.
    pub fn getrandbit(&mut self) -> bool {
        self.genrand_u32() >> 31 == 1
    }

    /// `getrandbits(k)` for `0 < k <= 64` as one integer: the words of
    /// [`Self::getrandbits_words`], least significant first.
    pub fn getrandbits_u64(&mut self, k: usize) -> u64 {
        assert!(k <= 64, "getrandbits_u64({k}) exceeds one u64");
        self.getrandbits_words(k)
            .iter()
            .enumerate()
            .fold(0u64, |acc, (i, &w)| acc | ((w as u64) << (32 * i)))
    }

    /// `random()`: the 53-bit double `(a * 2^26 + b) / 2^53` from
    /// `a = genrand_u32() >> 5` and `b = genrand_u32() >> 6`
    /// (`random_random` in `_randommodule.c`; the scaling by a power of two
    /// is exact).
    pub fn random(&mut self) -> f64 {
        let a = (self.genrand_u32() >> 5) as f64;
        let b = (self.genrand_u32() >> 6) as f64;
        (a * 67_108_864.0 + b) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// `_randbelow(n)` (`_randbelow_with_getrandbits`, CPython 3.2+): draw
    /// `getrandbits(n.bit_length())` until the value is below `n`; 0 for
    /// `n == 0`.
    pub fn randbelow(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        let k = (u64::BITS - n.leading_zeros()) as usize;
        loop {
            let r = self.getrandbits_u64(k);
            if r < n {
                return r;
            }
        }
    }

    /// `randrange(stop)` for `stop > 0`: one `_randbelow(stop)` draw.
    pub fn randrange(&mut self, stop: usize) -> usize {
        assert!(stop > 0, "randrange({stop}): empty range");
        self.randbelow(stop as u64) as usize
    }

    /// `choice(seq)`: `seq[_randbelow(len(seq))]`.
    pub fn choice<T: Copy>(&mut self, seq: &[T]) -> T {
        assert!(!seq.is_empty(), "choice() from an empty sequence");
        seq[self.randbelow(seq.len() as u64) as usize]
    }

    /// `choices(population, weights=weights)[0]` for integer weights:
    /// cumulative weights by `itertools.accumulate`, `total = cum[-1] + 0.0`,
    /// index `bisect.bisect(cum, random() * total, 0, len(population) - 1)`.
    /// CPython compares the float against the int cumulative weights
    /// exactly; converting them to `f64` is exact while they stay below
    /// 2^53, which is asserted.
    pub fn choices_one<T: Copy>(&mut self, population: &[T], weights: &[u64]) -> T {
        assert_eq!(
            population.len(),
            weights.len(),
            "choices(): the number of weights does not match the population"
        );
        let mut cum = Vec::with_capacity(weights.len());
        let mut acc = 0u64;
        for &w in weights {
            acc += w;
            cum.push(acc);
        }
        let total = *cum.last().expect("choices() from an empty population");
        assert!(
            total > 0,
            "choices(): total of weights must be greater than zero"
        );
        assert!(
            total < (1u64 << 53),
            "choices(): cumulative weights exceed the exact f64 range"
        );
        let x = self.random() * (total as f64);
        // bisect.bisect_right(cum, x, 0, len(population) - 1)
        let (mut lo, mut hi) = (0usize, population.len() - 1);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if x < cum[mid] as f64 {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        population[lo]
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/packing/pyrand.rs"]
mod tests;
