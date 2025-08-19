use crate::{
            rainbow::constants::{self, CONTROL_FUNC_TABLE},
            circuit::{Circuit, Gate, Permutation},
            };

use lru::LruCache;
use itertools::Itertools;
use once_cell::sync::Lazy;
use std::num::NonZeroUsize;
use std::sync::Mutex;
use std::sync::atomic::AtomicI64;

#[derive(Clone, Debug)]
pub struct Canonicalization
{
    perm: Permutation,
    shuffle: Permutation,
}


#[derive(Clone, Debug)]
pub struct CandSet
{
    candidate: Vec<Vec<bool>>,
}

// save time by caching canonicalizations
// ideally, traverse circuits in a cache-friendly way (functional equiv?)
// no need to store unpopular perms
const CACHE_SIZE: usize = 32768;

static BIT_SHUF: Lazy<Mutex<Vec<Vec<usize>>>> = Lazy::new(|| Mutex::new(Vec::new()));
static CACHE: Lazy<Mutex<LruCache<String, Canonicalization>>> = Lazy::new(|| {
    Mutex::new(LruCache::new(NonZeroUsize::new(CACHE_SIZE).unwrap()))
});

pub fn init(n: usize) {
    let perms: Vec<Vec<usize>> = (0..n).permutations(n).collect();
    let bit_shuf = perms.into_iter().skip(1).collect::<Vec<_>>();
    *BIT_SHUF.lock().unwrap() = bit_shuf;

    // reset cache if needed
    *CACHE.lock().unwrap() = LruCache::new(NonZeroUsize::new(CACHE_SIZE).unwrap());
}

fn strings_of_weight(w: usize, n: usize) -> Vec<usize> {
    let n_total = 1usize << n;

    // Compute the next integer after x with the same Hamming weight
    fn next(x: usize) -> usize {
        let c = x & (!x + 1); // equivalent to x & -x in 2's complement
        let r = x + c;
        (((r ^ x) >> 2) / c) | r
    }

    let mut a = (1 << w) - 1;

    if w == 0 {
        return vec![a];
    }

    let mut result = Vec::new();
    while a < n_total {
        result.push(a);
        a = next(a);
    }

    result
}

fn index_set(s: usize, n: usize) -> Vec<usize> {
    // "light" strings
    let mut p = strings_of_weight(s, n);

    if 2 * s == n {
        return p;
    }

    // "heavy" strings
    let mut q = strings_of_weight(n - s, n);
    p.append(&mut q);
    p
}



static PERM_CACHED: AtomicI64 = AtomicI64::new(0);
static PERM_BF_COMPUTED: AtomicI64 = AtomicI64::new(0);
static PERM_FAST_COMPUTED: AtomicI64 = AtomicI64::new(0);



impl Permutation {
    //need to test
    //Eli note, this needs t work in weight-class order
    pub fn brute_canonical(&self) -> Canonicalization {
        let n = self.data.len();
        let b = (n as u32 - 1).next_power_of_two().trailing_zeros() as usize;

        // store minimal bit permutation in here
        let mut m = self.clone().data;
        // temporary to reconstruct shuffled bits
        let mut t = vec![0; n];
        // temporary to reconstruct shuffled indices
        let mut idx = vec![0; n];
        // temporary to shuffle t into, according to idx
        let mut s = vec![0; n];

        let mut best_shuffle = Permutation::id_perm(b);

        let bit_shuf = BIT_SHUF.lock().unwrap();
        for r in bit_shuf.iter() {
            // Apply the bit shuffle
            for (src, dst) in r.iter().enumerate() {
                for (i, &val) in self.data.iter().enumerate() {
                    t[i] |= ((val >> src) & 1) << dst;
                    idx[i] |= ((i >> src) & 1) << dst;
                }
            }

            for (i, &ti) in t.iter().enumerate() {
                s[idx[i]] = ti;
            }

            // lexicographical sort in weight-order
            for w in 0..=b / 2 {
                let mut done = false;
                for i in index_set(w, b) {
                    if s[i] == m[i] {
                        continue;
                    }
                    if s[i] < m[i] {
                        m.clone_from_slice(&s);
                        best_shuffle = Permutation{ data: r.clone(), };
                    }
                    done = true;
                    break;
                }
                if done {
                    break;
                }
            }

            // clear slices out for the next round
            t.fill(0);
            idx.fill(0);
        }

        Canonicalization {
            perm: Permutation { data: m },
            shuffle: best_shuffle,
        }
    }

    

    //TODO: implement fastcanon, add PermCached, PermFastComputed, PermBFComputed
    pub fn canonical_with_retry(&self, retry: bool) -> Canonicalization {
        // Panic if BIT_SHUF hasn't been initialized
        if BIT_SHUF.lock().unwrap().is_empty() {
            panic!("Call init() first!");
        }

        let ps = self.repr(); // returns Vec<u8> for cache key

        // Check cache
        {
            let mut cache = CACHE.lock().unwrap();
            if let Some(c) = cache.get(&ps) {
                // Cached value exists
                PERM_CACHED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                if c.perm.data.is_empty() {
                    // "nil" means already canonical
                    return Canonicalization {
                        perm: self.clone(),
                        shuffle: Permutation { data: Vec::new() },
                    };
                }

                return c.clone();
            }
        }

        // Try fast canonicalization
        let mut pm = self.fast_canon(); // TODO: implement this

        if pm.perm.data.is_empty() {
            if retry {
                // Fast canon failed, retry with a random shuffle
                let n = self.data.len();
                let r = Permutation::rand_perm(n); // returns a Permutation
                return self.bit_shuffle(&r.data).canonical_with_retry(false);
            } else {
                // Retry not allowed, fall back to brute force
                pm = self.brute_canonical();
                PERM_BF_COMPUTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        } else {
            PERM_FAST_COMPUTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        // Store result in cache
        let mut cache = CACHE.lock().unwrap();
        if self.data == pm.perm.data {
            // Already canonical, store empty to indicate "nil"
            cache.put(
                ps,
                Canonicalization {
                    perm: Permutation { data: Vec::new() },
                    shuffle: Permutation { data: Vec::new() },
                },
            );
        } else {
            cache.put(ps, pm.clone());
        }

        pm
    }

    pub fn canonical(&self) -> Canonicalization {
        self.canonical_with_retry(false)
    }
    //Need to implement CandSet
    pub fn fast_canon(&self) -> Canonicalization {
        Canonicalization{ perm: self.clone(), shuffle: Permutation::id_perm(self.data.len()) } 
    }
}
