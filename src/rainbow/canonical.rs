use crate::circuit::{self, Circuit, Gate, Permutation, CircuitSeq};
use crate::circuit::par_all_circuits;
use smallvec::SmallVec;
use itertools::Itertools;
use lru::LruCache;
use once_cell::sync::Lazy;
use std::{
    collections::{HashMap, HashSet},
    num::NonZeroUsize,
    sync::Mutex,
    sync::atomic::AtomicI64,
    sync::atomic::Ordering
};


#[derive(Clone, Debug)]
pub struct Canonicalization
{
    pub perm: Permutation,
    pub shuffle: Permutation,
}


#[derive(Clone, Debug)]
pub struct CandSet
{
    pub candidate: Vec<Vec<bool>>,
}

#[derive(Clone, Debug)]
pub struct PermStore {
    pub perm: Permutation,
    pub circuits: HashMap<Vec<u8>, bool>,
    pub count: usize,
    pub contains_any_circuit: bool,
    pub contains_canonical: bool,
    pub already_visited: bool,
}

// save time by caching canonicalizations
// ideally, traverse circuits in a cache-friendly way (functional equiv?)
// no need to store unpopular perms
const CACHE_SIZE: usize = 32768;

//Use Lazy to ensure that BIT_SHUF is only initialized once. 
//Use Mutex to allow us to initialize BIT_SHUF dynamically with chosen n
//Use Lazy and Mutex for CACHE to ensure threads aren't updating cache at the same time
//Use least recently updated cache for efficiency as we don't need unpopular perms
//Cache will hold permutations and their associated canonicalizations, so if we see a permutation again, we do not need to recompute the canon perm

//TODO: Can we remove this lock on BIT_SHUF
static BIT_SHUF: Lazy<Mutex<Vec<Vec<usize>>>> = Lazy::new(|| Mutex::new(Vec::new()));
//TODO: Computing strings is inefficient
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

pub fn strings_of_weight(w: usize, n: usize) -> Vec<usize> {
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

pub fn index_set(s: usize, n: usize) -> Vec<usize> {
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
        let mut pm = self.fast_canon();

        if pm.perm.data.is_empty() {
            if retry {
                // Fast canon failed, retry with a random shuffle
                let n = self.data.len();
                let r = Permutation::rand_perm(n); // returns a Permutation
                return self.bit_shuffle(&r.data).canonical_with_retry(false);
            } else {
                // Retry not allowed, fall back to brute force
                pm = self.brute_canonical();
                PERM_BF_COMPUTED.fetch_add(1, Ordering::Relaxed);
            }
        } else {
            PERM_FAST_COMPUTED.fetch_add(1, Ordering::Relaxed);
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
    
    pub fn brute_canonical(&self) -> Canonicalization {
        let bit_shuf_global = BIT_SHUF.lock().unwrap();
        // Panic if BIT_SHUF hasn't been initialized
        if bit_shuf_global.is_empty() {
            panic!("Call init() first!");
        }
        
        // num wires
        let n = self.data.len();

        // num bits that we can shuffle for relabeling
        // wires are 0..n-1
        let num_b = std::mem::size_of::<usize>() * 8 - (n - 1).leading_zeros() as usize;

        // store the minimal bit permutation
        let mut min_perm: SmallVec<[usize; 64]> = SmallVec::from_slice(&self.data);

        // store the shuffled bits according to the potential bit_shuf
        let mut bits: SmallVec<[usize; 64]> = SmallVec::from_elem(0, n);

        // Vector to hold where old indices moved
        let mut index_shuf: SmallVec<[usize; 64]> = SmallVec::from_elem(0, n);

        // Vector to hold the perm after bit shuffling and index shuffling
        let mut perm_shuf: SmallVec<[usize; 64]> = SmallVec::from_elem(0, n);

        // hold our current best_shuffle
        let mut best_shuffle = Permutation::id_perm(num_b);

        for r in bit_shuf_global.iter() {
            for (src, &dst) in r.iter().enumerate() {
                for (i, &val) in self.data.iter().enumerate() {
                    bits[i] |= ((val >> src) & 1) << dst;
                    index_shuf[i] |= ((i >> src) & 1) << dst;
                }
            }

            for (i, &val) in bits.iter().enumerate() {
                perm_shuf[index_shuf[i]] = val;
            }

            // lexicographical sort in weight-order 
            // Only consider b/2 since the "light" and "heavy" are just complements of each other
            // See index_set
            for weight in 0..=num_b / 2 {
                let mut done = false;
                for i in index_set(weight, num_b) {
                    if perm_shuf[i] == min_perm[i] {
                        continue;
                    }
                    if perm_shuf[i] < min_perm[i] {
                        min_perm.copy_from_slice(&perm_shuf);
                        best_shuffle.data.copy_from_slice(&r);
                    }
                    done = true;
                    break;
                }
                if done {
                    break;
                }
            }

            // clear the temp vectors to check the next bit_shuf
            bits.fill(0);
            index_shuf.fill(0);
        }

        Canonicalization {
            perm: Permutation { data: min_perm.into_vec() },
            shuffle: best_shuffle,
        }
    }

    //Goal of fast canon is to produce small snippets of the best permutation (by lexi order) and determine which in canonical
    //If we can't decide between multiple, for now, we just ignore and will do brute force
    pub fn fast_canon(&self) -> Canonicalization {
        let num_bits = self.bits();
        let mut candidates = CandSet::new(num_bits);
        let mut found_identity = false;

        // Scratch buffer to avoid cloning every iteration
        let mut scratch = CandSet::new(num_bits);

        // Pre-allocate viable_sets buffer to reuse
        let mut viable_sets: Vec<CandSet> = Vec::with_capacity(4);

        for weight in 0..=num_bits/2 {
            let index_words = index_set(weight, num_bits); // Vec<usize>

            'word_loop: for &w in &index_words {
                // Determine which preimages are possible
                let preimages = candidates.preimages(w);
                if preimages.is_empty() {
                    return Canonicalization {
                        perm: Permutation { data: Vec::new() },
                        shuffle: Permutation { data: Vec::new() },
                    };
                }

                viable_sets.clear();
                let mut best_score = -1;

                for &pre_idx in &preimages {
                    let mapped_value = self.data[pre_idx];

                    if !candidates.consistent(pre_idx, w) {
                        continue;
                    }

                    // Reset scratch from candidates and enforce mapping
                    scratch.copy_from(&candidates);
                    scratch.enforce(pre_idx, w);

                    // Minimum possible value with current scratch
                    let (score, mut reduced_set) = scratch.min_consistent(mapped_value);
                    if score < 0 {
                        continue;
                    }

                    reduced_set.intersect(&candidates);
                    if !reduced_set.consistent(pre_idx, w) {
                        continue;
                    }

                    // Track best score and viable sets
                    if best_score < 0 || score < best_score {
                        best_score = score;
                        viable_sets.clear();
                        // Move reduced_set into the vector (no clone)
                        viable_sets.push(reduced_set);
                        if w as isize == score {
                            found_identity = true;
                        }
                    } else if score == best_score {
                        if w as isize == score {
                            if found_identity {
                                viable_sets.push(reduced_set);
                            } else {
                                viable_sets.clear();
                                viable_sets.push(reduced_set);
                            }
                            found_identity = true;
                        } else if !found_identity {
                            viable_sets.push(reduced_set);
                        }
                    }
                }

                match viable_sets.len() {
                    0 => continue,
                    1 => candidates = viable_sets.pop().unwrap(),
                    _ => {
                        return Canonicalization {
                            perm: Permutation { data: Vec::new() },
                            shuffle: Permutation { data: Vec::new() },
                        }
                    }
                }

                if candidates.complete() {
                    break 'word_loop;
                }
            }

            if candidates.complete() {
                break;
            }
        }

        if candidates.unconstrained() {
            return Canonicalization {
                perm: self.clone(),
                shuffle: Permutation { data: Vec::new() },
            };
        }

        if !candidates.complete() {
            println!("Incomplete!");
            println!("{:?}", self);
            println!("{:?}", candidates);
            std::process::exit(1);
        }

        let final_shuffle = match candidates.output() {
            Some(v) => Permutation { data: v },
            None => {
                eprintln!("CandSet output returned None!");
                std::process::exit(1);
            }
        };

        Canonicalization {
            perm: self.bit_shuffle(&final_shuffle.data),
            shuffle: final_shuffle,
        }
    }
}


impl Circuit {
    pub fn canonicalize(&mut self) {
        // Insertion-sort-based canonicalization
        for i in 1..self.gates.len() {
            let gi = self.gates[i].clone(); // copy for checking
            let mut to_swap: Option<usize> = None;

            let mut j = i;
            while j > 0 {
                j -= 1;
                if self.gates[j].collides(&gi) {
                    break;
                } else if !self.gates[j].ordered(&gi) {
                    to_swap = Some(j);
                }
            }

            if let Some(pos) = to_swap {
                let g = self.gates[i].clone(); // copy for insertion
                // Remove the gate at i
                self.gates.remove(i);
                // Insert at the new position
                self.gates.insert(pos, g);
            }
        }
    }
}

impl CandSet {
    pub fn new(n: usize) -> CandSet {
        //build n x n candidate set, initialize with true
        let c = vec![vec![true; n]; n];
        CandSet{ candidate: c }
    }

    /// Compute the possible preimages of `w`, given this candidate set
    pub fn preimages(&self, w: usize) -> Vec<usize> {
        let n = self.candidate.len();
        let mut p = Vec::new();

        // Hamming weight of w
        let hw = w.count_ones() as usize;

        for s in strings_of_weight(hw, n) {
            // aggregate candidate array
            let mut agg_cand = vec![false; n];

            for b in 0..n {
                if (s & (1 << b)) != 0 {
                    // bit b is set... merge candidate rows
                    for (i, &elm) in self.candidate[b].iter().enumerate() {
                        agg_cand[i] = agg_cand[i] || elm;
                    }
                }
            }

            // check if wbits ⊆ agg_cand
            let mut consistent = true;
            for b in 0..n {
                if (w & (1 << b)) != 0 && !agg_cand[b] {
                    consistent = false;
                    break;
                }
            }

            if consistent {
                p.push(s);
            }
        }
        p
    }

    pub fn enforce(&mut self, x:usize, y:usize) {
        let n = self.candidate.len();

        for (k, row) in self.candidate.iter_mut().enumerate() {
            if (x & (1 << k)) == 0 {
                // zeros must map to zeros: zero out the ones in y
                for b in 0..n {
                    row[b] &= !(y & (1 << b) != 0);
                }
            } else {
                // ones must map to ones: zero out the zeros in y
                for b in 0..n {
                    row[b] &= !(y & (1 << b) == 0);
                }
            }
        }
    }

    pub fn fix_map(&mut self, from: usize, to: usize) {
        let n = self.candidate.len();
        for i in 0..n {
            self.candidate[from][i] = false;
            self.candidate[i][to] = false;
        }

        self.candidate[from][to] = true;
    }

    // Is it possible that x maps to y, given the candset?
    // Similar to enforce.
    pub fn consistent(&self, x:usize, y:usize) -> bool {
        let n = self.candidate.len();
        // xz, xo := bit_locations(x, n)
	    // yz, yo := bit_locations(y, n)

        for (k, row) in self.candidate.iter().enumerate() {
            // for each index where x has a zero...
            if x & (1 << k) == 0 {
                // does y also have a zero somewhere?
                let mut mat = false;
                for b in 0..n {
                    if y&(1<<b) == 0 && row[b] {
                        mat = true;
                        break;
                    }
                }

                if !mat {
                    return false
                }
            }

            //where x has a one
            if x&(1<<k) != 0 {
                //does y?
                let mut mat = false;
                for b in 0..n {
                    if y&(1<<b) != 0 && row[b] {
                        mat = true;
                        break;
                    }
                }

                if !mat {
                    return false
                }
            }
        }
        true
    }

    //find the bits that could map to bit b
    pub fn bits_pre(&self, b: usize) -> Vec<usize> {
        let mut out = Vec::new();
        for (k, row) in self.candidate.iter().enumerate() {
            if row[b] {
                out.push(k);
            }
        }
        out
    }

    // Find the minimum consistent mapping for x
    // returns the minimum value and a new candidate set
    pub fn min_consistent(&self, x:usize) -> (isize, CandSet) {
        let n = self.candidate.len();
        let mut c2 = self.clone();

        let mut out: usize = 0;
        for i in (0..n).rev() {
            //which bits map to bit i
            let i_from = c2.bits_pre(i);

            if i_from.is_empty() {
                return (-1, CandSet{ candidate: Vec::new() }) //no valid mapping
            }

            //try to make this bit zero. else, one
            let mut max_zero:isize = -1;
            let mut j:isize = -1;

            for &b in &i_from {
                if (x&(1 << b) == 0) && ((b as isize) > max_zero) {
                    max_zero = b as isize;
                }
            }

            if max_zero >= 0 {
                j = max_zero;
            } else {
                //look for min one. default is invalid
                let mut min_one = n as isize + 1;
                for &b in &i_from {
                    if (x&(1 << b) != 0) && ((b as isize) < min_one) {
                        min_one = b as isize;
                    }
                }

                //min_one should be valid
                if min_one > n as isize{
                    return (-1, CandSet{ candidate: Vec::new() })
                }

                j = min_one;
            }

            //apply shuffle
            out |= ((x >> j) & 1) << i;
            c2.fix_map(j as usize, i);
        }

        let mut c3 = self.clone();
        c3.enforce(x,out);
        (out as isize, c3)
    }

    pub fn complete(&self) -> bool {
        let mut seen: HashSet<usize> = HashSet::new();
        for row in &self.candidate {
            let mut count_nonneg = 0;
            for (i,&val) in row.iter().enumerate() {
                if val {
                    count_nonneg += 1;
                    //we've already seen this column, so fail
                    if !seen.insert(i) {
                        return false
                    }
                }
            }

            //must have exactly one "true" per row
            if count_nonneg != 1 {
                return false
            }
        }
        true
    } 

    pub fn unconstrained(&self) -> bool {
        let all_true: Vec<bool> = vec![true; self.candidate.len()];
        for r in &self.candidate {
            if r != &all_true {
                return false
            }
        }
        true
    }

    //output the bit rearrangement
    pub fn output(&self) -> Option<Vec<usize>> {
        if !self.complete() {
            //can't output incomplete perm
            return None
        }

        let mut output = vec![0; self.candidate.len()];
        for (i,row) in self.candidate.iter().enumerate() {
            for (j,&x) in row.iter().enumerate() {
                if x {
                    output[i] = j;
                    break;
                }
            }
        } 
        Some(output)
    }

    pub fn intersect(&mut self, c2: &CandSet) {
        for (i,row) in self.candidate.iter_mut().enumerate() {
            let s = &c2.candidate[i];
            for (j,val) in row.iter_mut().enumerate() {
                *val &= s[j];
            }
        }
    }

    pub fn to_string(&self) -> String {
        let mut s = String::new();
        for (i,row) in self.candidate.iter().enumerate() {
            s.push_str(&format!("{}", i)); 
            for &x in row {
                if x {
                    s += "#";
                } else { 
                    s += ".";
                }
            }
            s += " ";
        }
        s
    }

    pub fn copy_from(&mut self, other: &CandSet) {
        for (dest_row, src_row) in self.candidate.iter_mut().zip(&other.candidate) {
            dest_row.copy_from_slice(src_row);
        }
    } 
}

impl PermStore {
    pub fn NewPermStore(perm: Permutation) -> Self {
        Self {
            perm,
            circuits: HashMap::new(),
            count: 0,
            contains_any_circuit: false,
            contains_canonical: false,
            already_visited: false,
        }
    }

    pub fn add_circuit(&mut self, repr: &Vec<u8>) {
        self.count += 1;
        self.circuits.insert(repr.to_vec(), true);
    }

    pub fn replace(&mut self, repr: &Vec<u8>) {
        self.count += 1;
        self.circuits.clear();
        self.circuits.insert(repr.to_vec(),true);
    }

    pub fn increment(&mut self) {
        self.count += 1;
    }
}

impl CircuitSeq {
    pub fn canonicalize(&mut self, base_gates: &Vec<[usize; 3]>) {
        for i in 1..self.gates.len() {
            //index in base_gates of current gate
            let gi_index = self.gates[i];
            let mut to_swap: Option<usize> = None;

            let mut j = i;
            while j > 0 {
                j -= 1;
                let gj_index = self.gates[j];

                if Gate::collides_index(gi_index, gj_index, base_gates) {
                    break;
                } else if !Gate::ordered_index(gj_index, gi_index, base_gates) {
                    to_swap = Some(j);
                }
            }
            if let Some(pos) = to_swap {
                let g = self.gates[i];
                self.gates.remove(i);
                self.gates.insert(pos, g);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_string_brute() {
        init(4);
        let mut c: Circuit = Circuit::from_string("0 1 2; 3 2 1; 0 2 1".to_string());
        println!("Circuit data: \n{}", c.to_string());
        let perm = c.permutation();
        println!("Permutation: \n{:?}", perm.data);
        c.canonicalize();
        let canon = perm.fast_canon();
        println!("Canonical perm: \n {:?}", canon.perm);
        println!("Shuffle: \n{:?}", canon.shuffle);
        println!("Canonical circuit: \n{}", c.to_string());
    }

    #[test]
    fn test_par() {
        //warmpup if getting unstable performance times
        for _ in 0..10 {
            let _ = par_all_circuits(2,2);
        }

        let now = std::time::Instant::now();
        let _ = circuit::par_all_circuits(3,3);
        println!("Time: {:?}", now.elapsed());
    }

    #[test]
    fn test_brute_canon() {
        let mut c: Circuit = Circuit::from_string("0 1 2; 3 2 1; 0 2 1".to_string());
        let p = c.permutation();
        for _ in 0..10 {
            let _ = p.brute_canonical();
        }
        let now = std::time::Instant::now();
        //TODO
    }
}

