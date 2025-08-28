use crate::{
    circuit::{Circuit, Gate, Permutation},
    rainbow::constants::{self, CONTROL_FUNC_TABLE},
};

use itertools::Itertools;
use lru::LruCache;
use once_cell::sync::Lazy;
use std::{
    collections::{HashMap, HashSet},
    num::NonZeroUsize,
    sync::{Arc, Mutex},
    sync::atomic::AtomicI64,
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
    pub circuits: HashMap<String, bool>,
    pub count: usize,
    pub contains_any_circuit: bool,
    pub contains_canonical: bool,
    pub already_visited: bool,
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
    
    //Average-case poly time algorithm
    //Returns none if it can't determine. Then just resort to brute force
    pub fn fast_canon(&self) -> Canonicalization {
        let n = self.bits();
        let mut cand = CandSet::new(n.try_into().unwrap());
        let mut identity = false;
        for weight in 0..n/2 {
            let s = index_set(weight.try_into().unwrap(),n.try_into().unwrap()); //a Vec<usize>
            for &w in &s {
                let p = cand.preimages(w);
                if p.len() == 0 {
                    return Canonicalization{ perm: Permutation{ data: Vec::new() },
                                             shuffle: Permutation{ data: Vec::new() } }
                }
                
                let mut passed: Vec<CandSet> = Vec::new();
                let mut best_val = -1;
                let mut best_x = 0;

                for &x in &p {
                    let y = self.data[x];
                    if !cand.consistent(x,w) {
                        continue;
                    }
                    let mut cand2 = cand.clone();
                    cand2.enforce(x,w);
                    
			        // given an input of y and the candidate set, what is the
				    // minimum possible value we can achieve?

                    let (val,mut m) = cand2.min_consistent(y);
                    if val < 0 {
                        continue;
                    }

                    m.intersect(&cand);
                    if !m.consistent(x,w) {
                        continue
                    }

                    if best_val < 0 || val < best_val {
                        best_val = val;
                        best_x = x;
                        passed = vec![m]; //reset
                        if w as isize == val {
                            identity = true;
                        }
                    } else if val == best_val {
                        if w as isize == val {
                            if identity {
                                passed.push(m);
                            } else {
                                passed = vec![m];
                                best_x = x;
                            }
                            identity = true;
                        } else if !identity {
                            passed.push(m);
                        }
                    } else {
                        continue;
                    }
                }
                match passed.len() {
                    0 => continue,
                    1 => cand = passed.remove(0),
                    _ => return Canonicalization { perm: Permutation { data: Vec::new(), },
                                                   shuffle: Permutation{ data: Vec::new() }, },
                }
                if cand.complete() {
                    break;
                }
            }
            if cand.complete() {
                break;
            }
        }
        if cand.unconstrained() {
            return Canonicalization{ perm: self.clone(), shuffle: Permutation{ data: Vec::new(), }, }
        }

        if !cand.complete() {
            println!("Incomplete!");
            println!("{:?}", self);
            println!("{:?}", cand);
            std::process::exit(1);
        }
        let final_shuffle = match cand.output() {
            Some(v) => Permutation { data: v },
            None => {
            // fallback if output is incomplete, maybe return identity or exit
            eprintln!("CandSet output returned None!");
            std::process::exit(1);
            }
        };

        Canonicalization{ perm: self.bit_shuffle(&final_shuffle.data), shuffle: final_shuffle, } 
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
            if (x & (1 << k) == 0) {
                // does y also have a zero somewhere?
                let mut mat = false;
                for b in 0..n {
                    if (y&(1<<b) == 0 && row[b]) {
                        mat = true;
                        break;
                    }
                }

                if !mat {
                    return false
                }
            }

            //where x has a one
            if (x&(1<<k) != 0) {
                //does y?
                let mut mat = false;
                for b in 0..n {
                    if (y&(1<<b) != 0 && row[b]) {
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

    pub fn add_circuit(&mut self, repr: &str) {
        self.count += 1;
        self.circuits.insert(repr.to_string(), true);
    }

    pub fn replace(&mut self, repr: &str) {
        self.count += 1;
        self.circuits.clear();
        self.circuits.insert(repr.to_string(),true);
    }

    pub fn increment(&mut self) {
        self.count += 1;
    }
}
