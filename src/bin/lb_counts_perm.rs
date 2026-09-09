use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use clap::{Parser, ValueEnum};
use crossbeam::queue::SegQueue;
use dashmap::mapref::entry::Entry;
use dashmap::{DashMap, DashSet};
use local_mixing::circuit::{CircuitSeq, Gate, Permutation, base_gates, circuit::iter_ones};
use nauty_Traces_sys::{SG_FREE, SparseGraph, optionblk, sparsegraph, sparsenauty, statsblk};
use num_bigint::BigUint;
use rustc_hash::{FxBuildHasher, FxHashSet};

fn factorial(n: usize) -> Option<usize> {
    (1..=n).try_fold(1, usize::checked_mul)
}

fn factorial_big(n: usize) -> BigUint {
    let mut f = BigUint::from(1u8);
    for i in 2..=n {
        f *= i;
    }
    f
}

/// `true` iff `p ∈ Alt_{N}` (`N = p.data.len()`), via cycle type.
fn perm_is_even(p: &Permutation) -> bool {
    let n = p.data.len();
    let mut seen = vec![false; n];
    let mut even = true;
    for start in 0..n {
        if seen[start] {
            continue;
        }
        let mut len = 0usize;
        let mut j = start;
        while !seen[j] {
            seen[j] = true;
            j = p.data[j];
            len += 1;
        }
        // An even-length cycle is an odd permutation.
        if len % 2 == 0 {
            even = !even;
        }
    }
    even
}

/// Wire-orbit canonical form of a reversible function, plus the orbit size
/// `|W·f| = n! / |Stab(f)|` as reported by nauty.
fn canonicalize_perm_sparse_graph(
    p: &Permutation,
    canonical_graph_scratch: &mut sparsegraph,
) -> (Permutation, usize) {
    #[allow(non_snake_case)]
    let NN = p.data.len();
    let n = NN.ilog2() as usize;
    assert_eq!(NN, 1 << n);

    let n_vertices = n + NN;
    let n_edges = NN * n;

    let mut lab: Vec<i32> = (0..n_vertices as i32).collect();
    let mut ptn: Vec<i32> = vec![1; n_vertices];
    // Two colour classes: wires, then point vertices.
    ptn[n - 1] = 0;
    ptn[n_vertices - 1] = 0;

    let mut v = vec![0; n_vertices];
    let mut d = vec![0i32; n_vertices];
    let mut e = Vec::<i32>::with_capacity(n_edges);

    // Wire vertex `bit` points to p(x) for each input x containing that bit.
    for bit in 0..n {
        d[bit] = (NN / 2) as i32;
        v[bit] = e.len();
        e.extend(
            p.data
                .iter()
                .enumerate()
                .filter(|(x, _)| (x >> bit) & 1 == 1)
                .map(|(_, &y)| (y + n) as i32),
        );
    }

    // Point vertex y points back to the wire vertices set in y.
    for y in 0..NN {
        let node_tag = y + n;
        d[node_tag] = y.count_ones() as i32;
        v[node_tag] = e.len();
        e.extend(iter_ones(y).map(|bit| bit as i32));
    }

    assert_eq!(v.len(), n_vertices);
    assert_eq!(e.len(), n_edges);

    let mut sg = SparseGraph { v, d, e };
    let mut opt = optionblk::default_sparse();
    // nauty only guarantees that `lab` is canonical when this is enabled.
    // Its API also requires a `canong` output buffer in that mode, even though
    // we discard the graph and retain only the labelling.
    opt.getcanon = 1;
    opt.digraph = 1;
    opt.defaultptn = 0; // honour our colour partition
    let mut stat = statsblk::default();
    let mut orbits = vec![0; n_vertices];

    unsafe {
        sparsenauty(
            &mut (&mut sg).into(),
            lab.as_mut_ptr(),
            ptn.as_mut_ptr(),
            orbits.as_mut_ptr(),
            &mut opt,
            &mut stat,
            canonical_graph_scratch,
        );
    }

    assert!(stat.grpsize2 == 0);
    let sphere = factorial(n).unwrap() / (stat.grpsize1 as usize);

    // lab[i] = original vertex now at canonical position i. The colour
    // partition keeps wire vertices in the first n positions.
    let mut wire_shuf = vec![0usize; n];
    for (canonical, &original) in lab[..n].iter().enumerate() {
        wire_shuf[original as usize] = canonical;
    }

    (p.bit_shuffle(&wire_shuf), sphere)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum CanonMethod {
    /// Fast graph canonicalization via nauty (default).
    Nauty,
    /// Slow but unambiguous lex-min over all wire permutations.
    Brute,
}

/// Dispatch to the selected canonicalizer. `scratch` is only touched by nauty.
fn canonicalize(
    method: CanonMethod,
    p: &Permutation,
    n: usize,
    scratch: &mut sparsegraph,
) -> (Permutation, usize) {
    match method {
        CanonMethod::Nauty => canonicalize_perm_sparse_graph(p, scratch),
        CanonMethod::Brute => canonicalize_perm_brute(p, n),
    }
}

/// Ground-truth wire-orbit canonicalization: the lexicographically minimal
/// conjugate `σ p σ⁻¹` over every wire permutation `σ ∈ Sₙ`, plus the orbit
/// size `|W·p|`. Unambiguous but `O(n!·2ⁿ)`, so only used for small `n` to
/// validate the nauty path.
fn canonicalize_perm_brute(p: &Permutation, n: usize) -> (Permutation, usize) {
    use itertools::Itertools;

    let mut best: Option<Permutation> = None;
    let mut orbit: std::collections::HashSet<Vec<usize>> = std::collections::HashSet::new();

    for shuffle in (0..n).permutations(n) {
        let conjugate = p.bit_shuffle(&shuffle);
        orbit.insert(conjugate.data.clone());
        match &best {
            Some(current) if current.data <= conjugate.data => {}
            _ => best = Some(conjugate),
        }
    }

    (best.unwrap(), orbit.len())
}

/// Packed truth-table. For domain size ≤ 256 (`n ≤ 8`) each image fits in a
/// `u8`; otherwise `u16` (enough through `n = 16`).
#[derive(Hash, PartialEq, Eq, PartialOrd, Ord, Clone, Debug)]
enum CompactPerm {
    U8(Box<[u8]>),
    U16(Box<[u16]>),
}

impl CompactPerm {
    fn from_permutation(p: &Permutation) -> Self {
        if p.data.len() <= 256 {
            CompactPerm::U8(p.data.iter().map(|&x| x as u8).collect())
        } else {
            CompactPerm::U16(p.data.iter().map(|&x| x as u16).collect())
        }
    }

    fn to_permutation(&self) -> Permutation {
        match self {
            CompactPerm::U8(d) => Permutation {
                data: d.iter().map(|&x| x as usize).collect(),
            },
            CompactPerm::U16(d) => Permutation {
                data: d.iter().map(|&x| x as usize).collect(),
            },
        }
    }

    fn len(&self) -> usize {
        match self {
            CompactPerm::U8(values) => values.len(),
            CompactPerm::U16(values) => values.len(),
        }
    }

    fn lcp(&self, other: &Self) -> usize {
        match (self, other) {
            (CompactPerm::U8(a), CompactPerm::U8(b)) => {
                a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
            }
            (CompactPerm::U16(a), CompactPerm::U16(b)) => {
                a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
            }
            _ => 0,
        }
    }

    fn symbol_bytes(&self) -> usize {
        match self {
            CompactPerm::U8(_) => 1,
            CompactPerm::U16(_) => 2,
        }
    }
}

/// Exact Lehmer/factoradic rank for permutations that fit in 128 bits.
///
/// Reversible functions on five wires permute 32 values, and `32! < 2^118`,
/// so their complete truth tables have a collision-free 128-bit key. The
/// remaining-values bitset makes ranking O(N), with no allocation.
fn lehmer_rank_u128(p: &CompactPerm) -> Option<u128> {
    let values: Vec<u16> = match p {
        CompactPerm::U8(values) => values.iter().map(|&value| value as u16).collect(),
        CompactPerm::U16(values) => values.to_vec(),
    };
    let n = values.len();
    if n > 64 {
        return None;
    }

    let mut remaining = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
    let mut rank = 0u128;

    for (index, &value) in values.iter().enumerate() {
        if value as usize >= n {
            return None;
        }
        let bit = 1u64 << value;
        if remaining & bit == 0 {
            return None;
        }
        let lower_mask = bit - 1;
        let smaller = (remaining & lower_mask).count_ones() as u128;
        rank = rank
            .checked_mul((n - index) as u128)?
            .checked_add(smaller)?;
        remaining &= !bit;
    }

    Some(rank)
}

/// Number of bits needed to store one image value of an `nn`-element
/// permutation, i.e. `log2(nn) = n`.
#[inline]
fn value_bits(nn: usize) -> usize {
    debug_assert!(nn.is_power_of_two());
    nn.trailing_zeros() as usize
}

/// Widest permutation that fits in a bit-packed `[u64; 6]` key: we need
/// `nn * value_bits(nn) <= 384`, which holds for `n <= 6` (64 * 6 = 384).
const PACKED_U384_MAX_VALUES: usize = 64;

/// Exact, allocation-free key: each image value is bit-packed into a fixed
/// `[u64; 6]`. Unlike the trie this has no per-permutation heap object, which
/// is what makes the `n = 6` ball (order 10^9 states) tractable.
fn pack_perm_u384(p: &CompactPerm, bits: usize) -> [u64; 6] {
    let mut packed = [0u64; 6];
    let mut place = |index: usize, value: u64| {
        let start = index * bits;
        let word = start / 64;
        let offset = start % 64;
        packed[word] |= value << offset;
        if offset + bits > 64 {
            packed[word + 1] |= value >> (64 - offset);
        }
    };
    match p {
        CompactPerm::U8(values) => {
            for (index, &value) in values.iter().enumerate() {
                place(index, value as u64);
            }
        }
        CompactPerm::U16(values) => {
            for (index, &value) in values.iter().enumerate() {
                place(index, value as u64);
            }
        }
    }
    packed
}

/// Inverse of [`pack_perm_u384`]; used only by tests to prove injectivity.
#[cfg(test)]
fn unpack_perm_u384(packed: &[u64; 6], nn: usize, bits: usize) -> CompactPerm {
    let mask = (1u64 << bits) - 1;
    let values: Vec<u8> = (0..nn)
        .map(|index| {
            let start = index * bits;
            let word = start / 64;
            let offset = start % 64;
            let mut raw = packed[word] >> offset;
            if offset + bits > 64 {
                raw |= packed[word + 1] << (64 - offset);
            }
            (raw & mask) as u8
        })
        .collect();
    CompactPerm::U8(values.into_boxed_slice())
}

fn analyze_prefixes(mut layers: Vec<Vec<CompactPerm>>, nn: usize) {
    const RESTART_INTERVAL: usize = 16;
    let mut factorial = BigUint::from(1u8);
    for i in 2..=nn {
        factorial *= i;
    }
    let lehmer_bits = (&factorial - BigUint::from(1u8)).bits() as usize;
    let lehmer_bytes = lehmer_bits.div_ceil(8);

    println!("\nPrefix/encoding analysis (cumulative through each BFS depth)");
    println!("Lehmer fixed width: {lehmer_bits} bits = {lehmer_bytes} bytes/permutation");
    println!("depth  perms   raw_bytes  frontcoded_bytes  mean_unique_prefix  p50  p90  p99  max");

    let mut cumulative = Vec::<CompactPerm>::new();
    let final_depth = layers.len() - 1;
    for (depth, layer) in layers.iter_mut().enumerate() {
        cumulative.append(layer);
        if cumulative.is_empty() {
            continue;
        }
        cumulative.sort_unstable();

        let key_len = cumulative[0].len();
        let symbol_bytes = cumulative[0].symbol_bytes();
        let mut unique_prefixes = Vec::with_capacity(cumulative.len());
        let mut frontcoded_bytes = 0usize;

        for i in 0..cumulative.len() {
            let prev_lcp = if i == 0 {
                0
            } else {
                cumulative[i].lcp(&cumulative[i - 1])
            };
            let next_lcp = if i + 1 == cumulative.len() {
                0
            } else {
                cumulative[i].lcp(&cumulative[i + 1])
            };
            let unique = if cumulative.len() == 1 {
                0
            } else {
                (prev_lcp.max(next_lcp) + 1).min(key_len)
            };
            unique_prefixes.push(unique);

            if i % RESTART_INTERVAL == 0 {
                frontcoded_bytes += key_len * symbol_bytes;
            } else {
                // u16 LCP length plus the unmatched suffix.
                frontcoded_bytes += 2 + (key_len - prev_lcp) * symbol_bytes;
            }
        }
        // Four-byte offset per restart block.
        frontcoded_bytes += cumulative.len().div_ceil(RESTART_INTERVAL) * 4;

        unique_prefixes.sort_unstable();
        let percentile = |p: f64| {
            let index = ((unique_prefixes.len() - 1) as f64 * p).round() as usize;
            unique_prefixes[index]
        };
        let mean = unique_prefixes.iter().sum::<usize>() as f64 / unique_prefixes.len() as f64;
        let raw_bytes = cumulative.len() * key_len * symbol_bytes;
        println!(
            "{depth:5} {count:7} {raw_bytes:11} {frontcoded_bytes:17} \
             {mean:18.2} {p50:4} {p90:4} {p99:4} {max:4}",
            count = cumulative.len(),
            p50 = percentile(0.50),
            p90 = percentile(0.90),
            p99 = percentile(0.99),
            max = unique_prefixes[unique_prefixes.len() - 1],
        );

        if depth == final_depth {
            let mut histogram = vec![0usize; key_len + 1];
            for &prefix in &unique_prefixes {
                histogram[prefix] += 1;
            }
            println!("Final shortest-unique-prefix histogram (length: count)");
            for (length, count) in histogram.into_iter().enumerate() {
                if count != 0 {
                    println!("{length:3}: {count}");
                }
            }
        }
    }
}

/// Concurrent value trie whose path is `π(0), π(1), ...`.
///
/// Each edge packs `(parent_node, value)` into one `u64`: 16 bits for the
/// value and 48 bits for the node id. This supports far more than billions of
/// nodes without the padding of a `(u64, u16)` key.
struct PermTrie {
    edges: DashMap<u64, u64, FxBuildHasher>,
    leaves: DashSet<u64, FxBuildHasher>,
    next_node: AtomicU64,
}

impl PermTrie {
    const VALUE_BITS: u32 = 16;
    const MAX_NODE: u64 = u64::MAX >> Self::VALUE_BITS;

    fn new() -> Self {
        Self {
            edges: DashMap::with_hasher(FxBuildHasher),
            leaves: DashSet::with_hasher(FxBuildHasher),
            next_node: AtomicU64::new(1), // node 0 is the root
        }
    }

    #[inline]
    fn edge_key(parent: u64, value: u16) -> u64 {
        assert!(
            parent <= Self::MAX_NODE,
            "permutation trie exceeded its 48-bit node-id space"
        );
        (parent << Self::VALUE_BITS) | value as u64
    }

    /// Returns `(leaf_node, newly_inserted_permutation)`.
    ///
    /// Threads can alternate winning individual edge insertions for an
    /// identical path, so a separate leaf insertion supplies the one atomic
    /// dedup winner for the complete permutation.
    fn insert(&self, p: &CompactPerm) -> (u64, bool) {
        let mut parent = 0u64;

        let mut push_value = |value: u16| {
            let edge = Self::edge_key(parent, value);
            parent = match self.edges.entry(edge) {
                Entry::Occupied(slot) => *slot.get(),
                Entry::Vacant(slot) => {
                    let child = self.next_node.fetch_add(1, Ordering::Relaxed);
                    assert!(child <= Self::MAX_NODE, "permutation trie node-id overflow");
                    slot.insert(child);
                    child
                }
            };
        };

        match p {
            CompactPerm::U8(values) => {
                for &value in values.iter() {
                    push_value(value as u16);
                }
            }
            CompactPerm::U16(values) => {
                for &value in values.iter() {
                    push_value(value);
                }
            }
        }

        let inserted = self.leaves.insert(parent);
        (parent, inserted)
    }

    fn permutation_count(&self) -> usize {
        self.leaves.len()
    }

    fn node_count(&self) -> u64 {
        self.next_node.load(Ordering::Relaxed)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum StoreMethod {
    /// Use exact Lehmer ranks when they fit in `u128`, otherwise use the trie.
    Auto,
    /// Concurrent value-prefix trie.
    Trie,
    /// Exact `u128` Lehmer ranks. Supports at most 32 values (`n <= 5`).
    Lehmer128,
    /// Exact bit-packed `[u64; 6]` keys. Supports at most 64 values (`n <= 6`).
    Packed384,
}

/// Fixed shard count so [`Visited::reserve_additional`] can convert a global
/// capacity request into the per-shard argument `DashMap::try_reserve` expects.
const VISITED_SHARDS: usize = 64;

enum Visited {
    Trie(PermTrie),
    /// `DashMap` rather than `DashSet` so we can call `try_reserve` between
    /// BFS layers and avoid mid-layer rehash spikes near the RAM ceiling.
    Lehmer128(DashMap<u128, (), FxBuildHasher>),
    Packed384 {
        map: DashMap<[u64; 6], (), FxBuildHasher>,
        bits: usize,
    },
}

impl Visited {
    fn new(method: StoreMethod, permutation_len: usize) -> Self {
        let selected = match method {
            StoreMethod::Auto if permutation_len <= 32 => StoreMethod::Lehmer128,
            StoreMethod::Auto if permutation_len <= PACKED_U384_MAX_VALUES => {
                StoreMethod::Packed384
            }
            StoreMethod::Auto => StoreMethod::Trie,
            other => other,
        };

        match selected {
            StoreMethod::Trie => Self::Trie(PermTrie::new()),
            StoreMethod::Lehmer128 => {
                assert!(
                    permutation_len <= 32,
                    "exact u128 Lehmer keys support at most 32 values (n <= 5)"
                );
                Self::Lehmer128(DashMap::with_capacity_and_hasher_and_shard_amount(
                    0,
                    FxBuildHasher,
                    VISITED_SHARDS,
                ))
            }
            StoreMethod::Packed384 => {
                assert!(
                    permutation_len <= PACKED_U384_MAX_VALUES,
                    "bit-packed [u64; 6] keys support at most 64 values (n <= 6)"
                );
                Self::Packed384 {
                    map: DashMap::with_capacity_and_hasher_and_shard_amount(
                        0,
                        FxBuildHasher,
                        VISITED_SHARDS,
                    ),
                    bits: value_bits(permutation_len),
                }
            }
            StoreMethod::Auto => unreachable!(),
        }
    }

    fn insert(&self, permutation: &CompactPerm) -> bool {
        match self {
            Self::Trie(trie) => trie.insert(permutation).1,
            Self::Lehmer128(map) => {
                let rank = lehmer_rank_u128(permutation)
                    .expect("permutation does not have an exact u128 Lehmer rank");
                map.insert(rank, ()).is_none()
            }
            Self::Packed384 { map, bits } => {
                map.insert(pack_perm_u384(permutation, *bits), ()).is_none()
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Trie(trie) => trie.permutation_count(),
            Self::Lehmer128(map) => map.len(),
            Self::Packed384 { map, .. } => map.len(),
        }
    }

    fn description(&self) -> &'static str {
        match self {
            Self::Trie(_) => "prefix trie",
            Self::Lehmer128(_) => "exact u128 Lehmer keys",
            Self::Packed384 { .. } => "exact bit-packed [u64; 6] keys",
        }
    }

    fn trie_node_count(&self) -> Option<u64> {
        match self {
            Self::Trie(trie) => Some(trie.node_count()),
            _ => None,
        }
    }

    /// Grow the visited table for up to `additional` new keys.
    ///
    /// `DashMap::try_reserve(x)` reserves `x` slots in *each* shard, so we ask
    /// for `ceil(additional / shards)` per shard. Best-effort: if the OS
    /// refuses, we keep going and accept rehash risk.
    fn reserve_additional(&mut self, additional: usize) {
        if additional == 0 {
            return;
        }
        let per_shard = additional.div_ceil(VISITED_SHARDS);
        let result = match self {
            Self::Trie(_) => return,
            Self::Lehmer128(map) => map.try_reserve(per_shard),
            Self::Packed384 { map, .. } => map.try_reserve(per_shard),
        };
        if let Err(_) = result {
            eprintln!(
                "warning: failed to reserve capacity for {additional} additional visited keys"
            );
        }
    }
}

/// One wire-orbit discovered by the min-length BFS.
#[derive(Clone, Debug)]
struct CanonOrbit {
    perm: CompactPerm,
    sphere: usize,
    min_depth: usize,
}

fn iso_bfs(
    n: usize,
    max_m: usize,
    analyze_prefixes_enabled: bool,
    requested_threads: Option<usize>,
    canon_method: CanonMethod,
    store_method: StoreMethod,
    collect_canonical: bool,
) -> (Vec<usize>, Vec<CanonOrbit>) {
    let nn = 1usize << n;
    let gen_gates = base_gates(n);
    let gens: Arc<Vec<Permutation>> = Arc::new(
        gen_gates
            .iter()
            .copied()
            .map(|g| CircuitSeq { gates: vec![g] }.perm(n))
            .collect(),
    );
    let gen_size = gens.len();

    let mut visited = Arc::new(Visited::new(store_method, nn));
    let analysis = analyze_prefixes_enabled
        .then(|| Arc::new(Mutex::new(vec![Vec::<CompactPerm>::new(); max_m + 1])));
    let dist_counts = Arc::new(DashMap::<usize, usize, FxBuildHasher>::with_hasher(
        FxBuildHasher,
    ));
    let spheres = Arc::new(DashMap::<usize, usize, FxBuildHasher>::with_hasher(
        FxBuildHasher,
    ));
    let collected = collect_canonical.then(|| {
        Arc::new(DashMap::<CompactPerm, (usize, usize), FxBuildHasher>::with_hasher(FxBuildHasher))
    });

    // Seed: identity at depth 0 (recorded, not expanded).
    let mut seed_canonical_graph = sparsegraph::default();
    let id_perm = Permutation::id_perm(nn);
    let (id_canon, id_sphere) =
        canonicalize(canon_method, &id_perm, n, &mut seed_canonical_graph);
    let id_key = CompactPerm::from_permutation(&id_canon);
    let inserted = visited.insert(&id_key);
    assert!(inserted);
    if let Some(analysis) = &analysis {
        analysis.lock().unwrap()[0].push(id_key.clone());
    }
    if let Some(collected) = &collected {
        collected.insert(id_key.clone(), (id_sphere, 0));
    }

    // Seed: one length-1 gate (all single gates are one wire-orbit).
    let base_ckt = CircuitSeq {
        gates: vec![[0, 1, 2]],
    };
    let (base_canon, base_sphere) = canonicalize(
        canon_method,
        &base_ckt.perm(n),
        n,
        &mut seed_canonical_graph,
    );
    SG_FREE(&mut seed_canonical_graph);
    let base_key = CompactPerm::from_permutation(&base_canon);
    let inserted = visited.insert(&base_key);
    assert!(inserted);
    if let Some(analysis) = &analysis {
        analysis.lock().unwrap()[1].push(base_key.clone());
    }
    if let Some(collected) = &collected {
        collected.insert(base_key.clone(), (base_sphere, 1));
    }
    *dist_counts.entry(1).or_default() += 1;
    *spheres.entry(1).or_default() += base_sphere;

    let num_threads = requested_threads.unwrap_or_else(num_cpus::get);
    assert!(num_threads > 0, "thread count must be positive");
    assert!(max_m >= 1, "maximum gate depth must be at least one");

    // Strictly level-synchronous BFS. A single mixed-depth queue is incorrect:
    // a longer path can win visited-set insertion before a shorter path,
    // making sphere counts and even the explored ball schedule-dependent.
    let mut frontier = Arc::new(SegQueue::<CompactPerm>::new());
    frontier.push(base_key);

    for depth in 2..=max_m {
        // Worst-case every generator neighbour is new. Reserving before the
        // layer avoids a mid-layer 2× rehash spike when the table is already
        // near the machine's RAM limit.
        let layer_capacity = frontier.len().saturating_mul(gen_size);
        Arc::get_mut(&mut visited)
            .expect("visited Arc uniquely owned between BFS layers")
            .reserve_additional(layer_capacity);

        let next_frontier = Arc::new(SegQueue::<CompactPerm>::new());
        let layer_started = Instant::now();
        let layer_circuits_done = Arc::new(AtomicU64::new(0));
        let layer_circuit_total = frontier.len() as u64 * gen_size as u64;

        std::thread::scope(|scope| {
            for tid in 0..num_threads {
                let frontier = frontier.clone();
                let next_frontier = next_frontier.clone();
                let visited = visited.clone();
                let analysis = analysis.clone();
                let dist_counts = dist_counts.clone();
                let spheres = spheres.clone();
                let collected = collected.clone();
                let gens = gens.clone();
                let layer_circuits_done = layer_circuits_done.clone();

                scope.spawn(move || {
                    // Reused across every canonicalization performed by this
                    // worker during the current BFS layer.
                    let mut canonical_graph_scratch = sparsegraph::default();
                    let mut new_count = 0usize;
                    let mut sphere_count = 0usize;
                    let thread_started = Instant::now();
                    let mut last_report = Instant::now();
                    let mut thread_circuits = 0u64;

                    while let Some(parent) = frontier.pop() {
                        let parent_perm = parent.to_permutation();

                        for gperm in gens.iter() {
                            let h = gperm.compose(&parent_perm);
                            let (canon, sphere) =
                                canonicalize(canon_method, &h, n, &mut canonical_graph_scratch);
                            let key = CompactPerm::from_permutation(&canon);

                            if !visited.insert(&key) {
                                continue;
                            }

                            new_count += 1;
                            sphere_count += sphere;
                            if let Some(analysis) = &analysis {
                                analysis.lock().unwrap()[depth].push(key.clone());
                            }
                            if let Some(collected) = &collected {
                                collected.insert(key.clone(), (sphere, depth));
                            }

                            // Count the final sphere but retain only states
                            // that will be expanded in the following layer.
                            if depth != max_m {
                                next_frontier.push(key);
                            }
                        }

                        thread_circuits += gen_size as u64;
                        let circuits_done = layer_circuits_done
                            .fetch_add(gen_size as u64, Ordering::Relaxed)
                            + gen_size as u64;

                        // Each worker reports independently, but never more
                        // often than once every two seconds.
                        if last_report.elapsed() >= Duration::from_secs(2) {
                            let thread_speed =
                                thread_circuits as f64 / thread_started.elapsed().as_secs_f64();
                            let layer_speed =
                                circuits_done as f64 / layer_started.elapsed().as_secs_f64();
                            let remaining = layer_circuit_total.saturating_sub(circuits_done);
                            let eta = if layer_speed > 0.0 {
                                remaining as f64 / layer_speed
                            } else {
                                f64::INFINITY
                            };

                            println!(
                                "t{tid:3} m:{depth:3} perms:{perms:10} Q:{queued:10} \
                                 next:{next:10} speed:{speed:8.1}k ckt/s eta:{eta:8.0}s",
                                perms = visited.len(),
                                queued = frontier.len(),
                                next = next_frontier.len(),
                                speed = thread_speed / 1000.0,
                            );
                            last_report = Instant::now();
                        }
                    }

                    *dist_counts.entry(depth).or_default() += new_count;
                    *spheres.entry(depth).or_default() += sphere_count;
                    SG_FREE(&mut canonical_graph_scratch);
                });
            }
        });

        frontier = next_frontier;
    }

    let can_ckt = visited.len();
    // This exceeds usize quickly (e.g. 60^11 for five wires), even when the
    // discovered ball still fits in memory.
    let total_ckt = BigUint::from(gen_size).pow(max_m as u32);
    let compr_ratio = &total_ckt / BigUint::from(can_ckt.max(1));

    println!("n={n} wires");
    println!(
        "Final: {} canonical perms, {} storage, {} total circuits, ({}x)",
        can_ckt,
        visited.description(),
        total_ckt,
        compr_ratio
    );
    if let Some(node_count) = visited.trie_node_count() {
        let trie_edges = node_count - 1;
        let unshared_values = (can_ckt as u128) * (nn as u128);
        println!(
            "Trie prefixes: {} nodes, {} edges vs {} unshared values ({:.1}% retained)",
            node_count,
            trie_edges,
            unshared_values,
            100.0 * trie_edges as f64 / unshared_values as f64
        );
    }

    let mut layer_counts = vec![0usize; max_m + 1];
    layer_counts[0] = 1;
    println!("m      count     sphere");
    for m in 1..=max_m {
        let count = dist_counts.get(&m).map(|c| *c).unwrap_or(0);
        let sp = spheres.get(&m).map(|c| *c).unwrap_or(0);
        layer_counts[m] = count;
        println!("{m} {count:10} {sp:10}");
    }

    if let Some(analysis) = analysis {
        let mut guard = analysis.lock().unwrap();
        analyze_prefixes(std::mem::take(&mut *guard), nn);
    }

    let orbits = collected
        .map(|map| {
            map.iter()
                .map(|entry| {
                    let (sphere, min_depth) = *entry.value();
                    CanonOrbit {
                        perm: entry.key().clone(),
                        sphere,
                        min_depth,
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    (layer_counts, orbits)
}

/// Expand one exact-length layer of canonical permutations: every parent in
/// `frontier` composed with every generator, then re-canonicalized.
fn expand_exact_layer(
    frontier: &[CompactPerm],
    gens: &[Permutation],
    n: usize,
    depth: usize,
    requested_threads: Option<usize>,
    canon_method: CanonMethod,
) -> Vec<(CompactPerm, usize)> {
    if frontier.is_empty() {
        return Vec::new();
    }

    let num_threads = requested_threads.unwrap_or_else(num_cpus::get).max(1);
    let gen_size = gens.len();
    let next = Arc::new(DashMap::<CompactPerm, usize, FxBuildHasher>::with_hasher(
        FxBuildHasher,
    ));
    let queue = Arc::new(SegQueue::<CompactPerm>::new());
    for parent in frontier {
        queue.push(parent.clone());
    }

    let gens = Arc::new(gens.to_vec());
    let layer_started = Instant::now();
    let layer_circuits_done = Arc::new(AtomicU64::new(0));
    let layer_circuit_total = frontier.len() as u64 * gen_size as u64;

    std::thread::scope(|scope| {
        for tid in 0..num_threads {
            let queue = queue.clone();
            let next = next.clone();
            let gens = gens.clone();
            let layer_circuits_done = layer_circuits_done.clone();

            scope.spawn(move || {
                let mut canonical_graph_scratch = sparsegraph::default();
                let thread_started = Instant::now();
                let mut last_report = Instant::now();
                let mut thread_circuits = 0u64;

                while let Some(parent) = queue.pop() {
                    let parent_perm = parent.to_permutation();
                    for gperm in gens.iter() {
                        let h = gperm.compose(&parent_perm);
                        let (canon, sphere) =
                            canonicalize(canon_method, &h, n, &mut canonical_graph_scratch);
                        next.insert(CompactPerm::from_permutation(&canon), sphere);
                    }

                    thread_circuits += gen_size as u64;
                    let circuits_done = layer_circuits_done
                        .fetch_add(gen_size as u64, Ordering::Relaxed)
                        + gen_size as u64;

                    if last_report.elapsed() >= Duration::from_secs(2) {
                        let thread_speed =
                            thread_circuits as f64 / thread_started.elapsed().as_secs_f64();
                        let layer_speed =
                            circuits_done as f64 / layer_started.elapsed().as_secs_f64();
                        let remaining = layer_circuit_total.saturating_sub(circuits_done);
                        let eta = if layer_speed > 0.0 {
                            remaining as f64 / layer_speed
                        } else {
                            f64::INFINITY
                        };
                        println!(
                            "t{tid:3} exact-m:{depth:3} layer:{layer:10} Q:{queued:10} \
                             speed:{speed:8.1}k ckt/s eta:{eta:8.0}s",
                            layer = next.len(),
                            queued = queue.len(),
                            speed = thread_speed / 1000.0,
                        );
                        last_report = Instant::now();
                    }
                }

                SG_FREE(&mut canonical_graph_scratch);
            });
        }
    });

    next.iter()
        .map(|entry| (entry.key().clone(), *entry.value()))
        .collect()
}

#[allow(dead_code)]
struct ExactCoverResult {
    /// Unique canonical permutations realized by some exact-length circuit.
    exact_counts: Vec<usize>,
    /// Orbit-weighted |W·f| sum at each exact length.
    exact_spheres: Vec<usize>,
    /// Smallest `m` such that exact-`m` covers every even generated orbit and
    /// exact-`(m+1)` covers every odd generated orbit.
    cover_m: Option<usize>,
}

/// After a min-length `iso_bfs` that lists every reachable canonical orbit,
/// walk exact-length layers by appending one gate to each canonical perm of
/// length `m-1`. That answers whether (e.g.) 9-gate and 10-gate circuits
/// cover `Alt` and `Sym \ Alt` rather than merely generating them by depth.
fn exact_cover_search(
    n: usize,
    max_m: usize,
    analyze_prefixes_enabled: bool,
    requested_threads: Option<usize>,
    canon_method: CanonMethod,
    store_method: StoreMethod,
) -> ExactCoverResult {
    let nn = 1usize << n;
    let (min_counts, orbits) = iso_bfs(
        n,
        max_m,
        analyze_prefixes_enabled,
        requested_threads,
        canon_method,
        store_method,
        true,
    );

    let mut even_targets = FxHashSet::default();
    let mut odd_targets = FxHashSet::default();
    let mut even_sphere_total = 0usize;
    let mut odd_sphere_total = 0usize;

    let mut even_min_diam = 0usize;
    let mut odd_min_diam = 0usize;
    for orbit in &orbits {
        let even = perm_is_even(&orbit.perm.to_permutation());
        if even {
            even_targets.insert(orbit.perm.clone());
            even_sphere_total += orbit.sphere;
            even_min_diam = even_min_diam.max(orbit.min_depth);
        } else {
            odd_targets.insert(orbit.perm.clone());
            odd_sphere_total += orbit.sphere;
            odd_min_diam = odd_min_diam.max(orbit.min_depth);
        }
    }

    let group_order = even_sphere_total + odd_sphere_total;
    let full_sym = factorial_big(nn);
    let full_alt = &full_sym / 2u8;
    println!(
        "\nExact-length cover of {} canonical orbits ({} even / {} odd)",
        orbits.len(),
        even_targets.len(),
        odd_targets.len()
    );
    println!(
        "Generated group order {group_order} (even {even_sphere_total}, odd {odd_sphere_total}); \
         |Sym_{nn}| = {full_sym}, |Alt_{nn}| = {full_alt}"
    );
    println!(
        "Min-length diameter: even orbits ≤ {even_min_diam}, odd orbits ≤ {odd_min_diam}"
    );
    if BigUint::from(group_order) != full_sym {
        println!(
            "warning: min-length ball of radius {max_m} is a proper subgroup/ball; \
             coverage is with respect to this generated set, not all of Sym_{nn}"
        );
    }

    let gen_gates = base_gates(n);
    let gens: Vec<Permutation> = gen_gates
        .iter()
        .copied()
        .map(|g| CircuitSeq { gates: vec![g] }.perm(n))
        .collect();

    let mut scratch = sparsegraph::default();
    let id_perm = Permutation::id_perm(nn);
    let (id_canon, id_sphere) = canonicalize(canon_method, &id_perm, n, &mut scratch);
    SG_FREE(&mut scratch);
    let id_key = CompactPerm::from_permutation(&id_canon);

    let mut layer = vec![(id_key, id_sphere)];
    let mut exact_counts = vec![0usize; max_m + 1];
    let mut exact_spheres = vec![0usize; max_m + 1];
    let mut even_complete = vec![false; max_m + 1];
    let mut odd_complete = vec![false; max_m + 1];

    let summarize = |layer: &[(CompactPerm, usize)]| {
        let mut even_hits = 0usize;
        let mut odd_hits = 0usize;
        let mut even_sp = 0usize;
        let mut odd_sp = 0usize;
        for (perm, sphere) in layer {
            if even_targets.contains(perm) {
                even_hits += 1;
                even_sp += *sphere;
            } else if odd_targets.contains(perm) {
                odd_hits += 1;
                odd_sp += *sphere;
            }
        }
        (even_hits, odd_hits, even_sp, odd_sp)
    };

    println!(
        "m     canons     sphere  even_orbits/target  odd_orbits/target  even_sphere  odd_sphere  Alt  odd"
    );
    let report_layer = |depth: usize,
                        layer: &[(CompactPerm, usize)],
                        even_complete: &mut [bool],
                        odd_complete: &mut [bool],
                        exact_counts: &mut [usize],
                        exact_spheres: &mut [usize]| {
        let (even_hits, odd_hits, even_sp, odd_sp) = summarize(layer);
        let sphere: usize = layer.iter().map(|(_, s)| *s).sum();
        exact_counts[depth] = layer.len();
        exact_spheres[depth] = sphere;
        even_complete[depth] = even_hits == even_targets.len();
        odd_complete[depth] = odd_hits == odd_targets.len();
        let alt_mark = if even_complete[depth] { "yes" } else { "no" };
        let odd_mark = if odd_complete[depth] { "yes" } else { "no" };
        println!(
            "{depth:<5} {canons:8} {sphere:10} {even_hits:8}/{even_t:<8} {odd_hits:7}/{odd_t:<8} \
             {even_sp:11} {odd_sp:10}  {alt_mark:3}  {odd_mark}",
            canons = layer.len(),
            even_t = even_targets.len(),
            odd_t = odd_targets.len(),
        );
    };

    report_layer(
        0,
        &layer,
        &mut even_complete,
        &mut odd_complete,
        &mut exact_counts,
        &mut exact_spheres,
    );

    for depth in 1..=max_m {
        let parents: Vec<CompactPerm> = layer.into_iter().map(|(p, _)| p).collect();
        layer = expand_exact_layer(
            &parents,
            &gens,
            n,
            depth,
            requested_threads,
            canon_method,
        );
        report_layer(
            depth,
            &layer,
            &mut even_complete,
            &mut odd_complete,
            &mut exact_counts,
            &mut exact_spheres,
        );
    }

    let mut cover_m = None;
    for m in 0..max_m {
        if even_complete[m] && odd_complete[m + 1] {
            cover_m = Some(m);
            break;
        }

        if odd_complete[m] && even_complete[m + 1] {
            cover_m = Some(m);
            break;
        }
    }

    let first_even = even_complete.iter().position(|&ok| ok);
    let first_odd = odd_complete.iter().position(|&ok| ok);

    println!();
    match (first_even, first_odd) {
        (Some(e), Some(o)) => {
            println!(
                "First exact length covering every even generated permutation (Alt): {e}"
            );
            println!(
                "First exact length covering every odd generated permutation (Sym \\ Alt): {o}"
            );
        }
        _ => println!(
            "Exact-length layers up to {max_m} do not yet cover both parities. Raise -m."
        ),
    }
    match cover_m {
        Some(m) => {
            println!(
                "Smallest m such that exact-length {m}-gate circuits cover all even \
                 generated permutations (Alt) and exact-length {}-gate circuits cover \
                 all odd generated permutations (Sym \\ Alt): m={m}",
                m + 1
            );
            if BigUint::from(group_order) == full_sym {
                println!(
                    "The generated group is all of Sym_{nn}, so this is the Alt_{nn} / \
                     (Sym_{nn} \\ Alt_{nn}) covering length."
                );
            }
        }
        None => {
            println!(
                "No m ≤ {max_m} has exact-length m covering every even orbit and \
                 m+1 covering every odd orbit. Raise -m."
            );
        }
    }

    // Min-length vs exact-length sanity: first-appearance counts cannot exceed
    // the exact-length layer that contains them.
    for depth in 0..=max_m {
        if min_counts[depth] > exact_counts[depth] {
            println!(
                "warning: min-length count {} at depth {depth} exceeds exact-length \
                 layer {}",
                min_counts[depth], exact_counts[depth]
            );
        }
    }

    ExactCoverResult {
        exact_counts,
        exact_spheres,
        cover_m,
    }
}

/// Injections `[8] → [16]`: images of the eight ancilla=0 data inputs.
/// `16×15×…×9 = 518_918_400`.
const ANCILLA_INJ: usize = 16 * 15 * 14 * 13 * 12 * 11 * 10 * 9;
const S8_ORDER: usize = 40320;

fn identity_ancilla_pack() -> u32 {
    let mut packed = 0u32;
    for i in 0..8u32 {
        packed |= i << (4 * i);
    }
    packed
}

fn identity_perm16() -> u64 {
    let mut packed = 0u64;
    for i in 0..16u64 {
        packed |= i << (4 * i);
    }
    packed
}

fn apply_gate_perm16(packed: u64, gate: [u16; 3]) -> u64 {
    let mut out = 0u64;
    for i in 0..16 {
        let x = ((packed >> (4 * i)) & 0xF) as usize;
        let y = Gate::evaluate_index(x, gate) as u64;
        out |= y << (4 * i);
    }
    out
}

/// `Some(π)` iff the 16-perm is `π ⊗ I`: ancilla bit 3 is identity on every
/// input, and both slices implement the same `π ∈ S_8`.
fn induced_pi_otimes_i(packed: u64) -> Option<[u8; 8]> {
    let mut pi = [0u8; 8];
    let mut seen = 0u8;
    for i in 0..8 {
        let lo = ((packed >> (4 * i)) & 0xF) as u8;
        let hi = ((packed >> (4 * (i + 8))) & 0xF) as u8;
        if lo >= 8 || hi != lo + 8 {
            return None;
        }
        seen |= 1 << lo;
        pi[i] = lo;
    }
    (seen == 0xFF).then_some(pi)
}

fn pack_pi_otimes_i(pi: [u8; 8]) -> u64 {
    let mut packed = 0u64;
    for i in 0..8 {
        let v = pi[i] as u64;
        packed |= v << (4 * i);
        packed |= (v + 8) << (4 * (i + 8));
    }
    packed
}

/// All 16-point permutations realized by some exact-length `k`-gate circuit.
fn exact_perm16_layer(k: usize, gens: &[[u16; 3]]) -> FxHashSet<u64> {
    let mut layer = FxHashSet::default();
    layer.insert(identity_perm16());
    for _ in 0..k {
        let mut next = FxHashSet::with_capacity_and_hasher(layer.len() * gens.len(), Default::default());
        for &state in &layer {
            for &gate in gens {
                next.insert(apply_gate_perm16(state, gate));
            }
        }
        layer = next;
    }
    layer
}

fn apply_gate_pack(packed: u32, gate: [u16; 3]) -> u32 {
    let mut out = 0u32;
    for i in 0..8 {
        let x = ((packed >> (4 * i)) & 0xF) as usize;
        let y = Gate::evaluate_index(x, gate) as u32;
        out |= y << (4 * i);
    }
    out
}

/// Combinadic rank of an 8-tuple of distinct values in `0..16`.
fn rank_injection(packed: u32) -> usize {
    const MUL: [usize; 8] = [
        15 * 14 * 13 * 12 * 11 * 10 * 9,
        14 * 13 * 12 * 11 * 10 * 9,
        13 * 12 * 11 * 10 * 9,
        12 * 11 * 10 * 9,
        11 * 10 * 9,
        10 * 9,
        9,
        1,
    ];
    let mut used = 0u16;
    let mut rank = 0usize;
    for i in 0..8 {
        let v = ((packed >> (4 * i)) & 0xF) as u16;
        let smaller = (!used & ((1u16 << v) - 1)).count_ones() as usize;
        rank += smaller * MUL[i];
        used |= 1u16 << v;
    }
    rank
}

fn lehmer_rank_s8(images: [u8; 8]) -> usize {
    let mut remaining = 0xFFu16;
    let mut rank = 0usize;
    for (i, &value) in images.iter().enumerate() {
        let bit = 1u16 << value;
        let smaller = (remaining & (bit - 1)).count_ones() as usize;
        rank = rank * (8 - i) + smaller;
        remaining &= !bit;
    }
    rank
}

/// If every tracked output still has ancilla bit 3 = 0, the 8-tuple is a
/// permutation of `{0,…,7}` — the induced `S_8` element.
fn induced_s8(packed: u32) -> Option<[u8; 8]> {
    let mut images = [0u8; 8];
    let mut seen = 0u8;
    for i in 0..8 {
        let v = ((packed >> (4 * i)) & 0xF) as u8;
        if v >= 8 {
            return None;
        }
        seen |= 1 << v;
        images[i] = v;
    }
    (seen == 0xFF).then_some(images)
}

fn s8_perm_is_even(images: [u8; 8]) -> bool {
    perm_is_even(&Permutation {
        data: images.iter().map(|&x| x as usize).collect(),
    })
}

fn atomic_bitset(nbits: usize) -> Vec<AtomicU64> {
    (0..nbits.div_ceil(64)).map(|_| AtomicU64::new(0)).collect()
}

fn bit_test_and_set(bits: &[AtomicU64], index: usize) -> bool {
    let word = index / 64;
    let mask = 1u64 << (index % 64);
    let old = bits[word].fetch_or(mask, Ordering::Relaxed);
    (old & mask) == 0
}

fn bit_test(bits: &[AtomicU64], index: usize) -> bool {
    let word = index / 64;
    let mask = 1u64 << (index % 64);
    bits[word].load(Ordering::Relaxed) & mask != 0
}

fn bit_set(bits: &[AtomicU64], index: usize) {
    let word = index / 64;
    let mask = 1u64 << (index % 64);
    bits[word].fetch_or(mask, Ordering::Relaxed);
}

#[allow(dead_code)]
struct AncillaS8Result {
    found: usize,
    /// Smallest `m` with every `π ∈ S_8` having min-length ≤ `m`.
    min_cover: Option<usize>,
    /// Smallest exact length at which +2 padding from min-length would place
    /// every `π ∈ S_8` (requires all min-lengths to share a parity).
    exact_cover: Option<usize>,
    hist_even: Vec<usize>,
    hist_odd: Vec<usize>,
    odd_cycle: bool,
}

/// Four-wire BFS for borrowed-ancilla embeddings of `S_8`.
///
/// Wire 3 is an ancilla promised to start at 0. Configurations are the images
/// of the eight `ancilla=0` inputs in `{0,1}^4` (`P(16,8)` states). A
/// configuration is a clean `S_8` element when every image has bit 3 clear.
/// Gates may target the ancilla; it just has to come home.
fn ancilla_s8_bfs(
    max_m: usize,
    requested_threads: Option<usize>,
    gates: &[[u16; 3]],
) -> AncillaS8Result {
    assert!(!gates.is_empty(), "need at least one generator");
    assert!(max_m >= 1, "maximum gate depth must be at least one");

    let num_threads = requested_threads.unwrap_or_else(num_cpus::get).max(1);
    let gen_size = gates.len();
    let gens = Arc::new(gates.to_vec());

    let visited = Arc::new(atomic_bitset(ANCILLA_INJ));
    let first_odd = Arc::new(atomic_bitset(ANCILLA_INJ));
    let found_at: Arc<Vec<AtomicU8>> = Arc::new((0..S8_ORDER).map(|_| AtomicU8::new(u8::MAX)).collect());
    let found_count = Arc::new(AtomicUsize::new(0));
    let odd_cycle = Arc::new(AtomicBool::new(false));
    let visited_count = Arc::new(AtomicU64::new(0));

    let start = identity_ancilla_pack();
    let start_rank = rank_injection(start);
    assert!(bit_test_and_set(&visited, start_rank));
    visited_count.fetch_add(1, Ordering::Relaxed);
    found_at[0].store(0, Ordering::Relaxed);
    found_count.store(1, Ordering::Relaxed);

    let mut frontier = Arc::new(SegQueue::<u32>::new());
    frontier.push(start);

    println!(
        "ancilla S8 BFS: 3 data wires + wire 3 borrowed, {} generators, P(16,8) = {ANCILLA_INJ}",
        gen_size
    );
    println!("m     found_S8         configs          Q  new_S8");
    println!("0 {:>10} {:>15} {:>10} {:>6}", 1, 1, 1, 1);

    for depth in 1..=max_m {
        if found_count.load(Ordering::Relaxed) == S8_ORDER {
            break;
        }

        let next_frontier = Arc::new(SegQueue::<u32>::new());
        let layer_started = Instant::now();
        let layer_done = Arc::new(AtomicU64::new(0));
        let layer_total = frontier.len() as u64 * gen_size as u64;
        let layer_new_s8 = Arc::new(AtomicUsize::new(0));

        std::thread::scope(|scope| {
            for tid in 0..num_threads {
                let frontier = frontier.clone();
                let next_frontier = next_frontier.clone();
                let visited = visited.clone();
                let first_odd = first_odd.clone();
                let found_at = found_at.clone();
                let found_count = found_count.clone();
                let odd_cycle = odd_cycle.clone();
                let visited_count = visited_count.clone();
                let gens = gens.clone();
                let layer_done = layer_done.clone();
                let layer_new_s8 = layer_new_s8.clone();

                scope.spawn(move || {
                    let thread_started = Instant::now();
                    let mut last_report = Instant::now();
                    let mut thread_ckts = 0u64;

                    while let Some(parent) = frontier.pop() {
                        if found_count.load(Ordering::Relaxed) == S8_ORDER {
                            break;
                        }
                        for &gate in gens.iter() {
                            let child = apply_gate_pack(parent, gate);
                            let rank = rank_injection(child);
                            let child_depth = depth;
                            let child_odd = child_depth % 2 == 1;

                            if !bit_test_and_set(&visited, rank) {
                                let first_was_odd = bit_test(&first_odd, rank);
                                if first_was_odd != child_odd {
                                    odd_cycle.store(true, Ordering::Relaxed);
                                }
                                continue;
                            }

                            if child_odd {
                                bit_set(&first_odd, rank);
                            }
                            visited_count.fetch_add(1, Ordering::Relaxed);
                            next_frontier.push(child);

                            if let Some(images) = induced_s8(child) {
                                let idx = lehmer_rank_s8(images);
                                if found_at[idx]
                                    .compare_exchange(
                                        u8::MAX,
                                        child_depth as u8,
                                        Ordering::Relaxed,
                                        Ordering::Relaxed,
                                    )
                                    .is_ok()
                                {
                                    found_count.fetch_add(1, Ordering::Relaxed);
                                    layer_new_s8.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }

                        thread_ckts += gen_size as u64;
                        let ckts_done =
                            layer_done.fetch_add(gen_size as u64, Ordering::Relaxed) + gen_size as u64;
                        if last_report.elapsed() >= Duration::from_secs(2) {
                            let speed = thread_ckts as f64 / thread_started.elapsed().as_secs_f64();
                            let layer_speed =
                                ckts_done as f64 / layer_started.elapsed().as_secs_f64();
                            let remaining = layer_total.saturating_sub(ckts_done);
                            let eta = if layer_speed > 0.0 {
                                remaining as f64 / layer_speed
                            } else {
                                f64::INFINITY
                            };
                            println!(
                                "t{tid:3} m:{depth:3} S8:{s8:6}/{S8_ORDER} cfg:{cfg:12} \
                                 Q:{q:10} next:{nxt:10} {speed:8.1}k ckt/s eta:{eta:8.0}s",
                                s8 = found_count.load(Ordering::Relaxed),
                                cfg = visited_count.load(Ordering::Relaxed),
                                q = frontier.len(),
                                nxt = next_frontier.len(),
                                speed = speed / 1000.0,
                            );
                            last_report = Instant::now();
                        }
                    }
                });
            }
        });

        let new_s8 = layer_new_s8.load(Ordering::Relaxed);
        let found = found_count.load(Ordering::Relaxed);
        println!(
            "{depth} {found:>10} {configs:>15} {q:>10} {new_s8:>6}",
            configs = visited_count.load(Ordering::Relaxed),
            q = next_frontier.len(),
        );

        frontier = next_frontier;
        if frontier.len() == 0 {
            break;
        }
    }

    let mut hist_even = vec![0usize; max_m + 1];
    let mut hist_odd = vec![0usize; max_m + 1];
    let mut unranked = [0u8; 8];
    // Build even/odd by unranking each found permutation... we stored only
    // lehmer ranks. Unrank, or scan found_at and unrank lehmer to a perm.
    for idx in 0..S8_ORDER {
        let d = found_at[idx].load(Ordering::Relaxed);
        if d == u8::MAX {
            continue;
        }
        unrank_s8(idx, &mut unranked);
        let depth = d as usize;
        if depth > max_m {
            continue;
        }
        if s8_perm_is_even(unranked) {
            hist_even[depth] += 1;
        } else {
            hist_odd[depth] += 1;
        }
    }

    let found = found_count.load(Ordering::Relaxed);
    println!("\nm  new_even  new_odd  cum_even  cum_odd  cum_total");
    let mut cum_even = 0usize;
    let mut cum_odd = 0usize;
    let mut min_cover = None;
    for m in 0..=max_m {
        cum_even += hist_even[m];
        cum_odd += hist_odd[m];
        let cum = cum_even + cum_odd;
        println!(
            "{m:<2} {e:8} {o:8} {cum_even:8} {cum_odd:8} {cum:9}",
            e = hist_even[m],
            o = hist_odd[m],
        );
        if min_cover.is_none() && cum == S8_ORDER {
            min_cover = Some(m);
        }
    }

    let cycle = odd_cycle.load(Ordering::Relaxed);
    if cycle {
        println!(
            "odd cycle in the configuration graph: min-length parity need not persist at exact length"
        );
    }
    match min_cover {
        Some(m) => println!("All of S_8 has a clean ancilla circuit of length ≤ {m}"),
        None => println!("Only {found}/{S8_ORDER} of S_8 found within {max_m} gates; raise -m"),
    }

    // Drop the min-length visited tables before the exact-length walk.
    drop(visited);
    drop(first_odd);

    let exact_counts = exact_ancilla_s8_layers(max_m, requested_threads, gates);
    let mut exact_cover = None;
    println!("\nExact-length clean S_8 (ancilla restored)");
    println!("m     clean_S8");
    for m in 0..=max_m {
        println!("{m:<2} {c:10}", c = exact_counts[m]);
        if exact_cover.is_none() && exact_counts[m] == S8_ORDER {
            exact_cover = Some(m);
        }
    }
    match exact_cover {
        Some(m) => println!(
            "Smallest exact length whose 4-wire circuits cover all of S_8 (ancilla restored): m={m}"
        ),
        None => println!("No exact length ≤ {max_m} covers all of S_8; raise -m"),
    }

    AncillaS8Result {
        found,
        min_cover,
        exact_cover,
        hist_even,
        hist_odd,
        odd_cycle: cycle,
    }
}

fn unrank_s8(mut rank: usize, out: &mut [u8; 8]) {
    let mut elems: Vec<u8> = (0..8).collect();
    for i in 0..8 {
        let f = factorial(7 - i).unwrap();
        let idx = rank / f;
        rank %= f;
        out[i] = elems.remove(idx);
    }
}

/// Exact-length layer walk on P(16,8). Each layer is the set of configurations
/// reachable by some circuit of length exactly `m`; clean configs contribute
/// their induced `S_8` element.
fn exact_ancilla_s8_layers(
    max_m: usize,
    requested_threads: Option<usize>,
    gates: &[[u16; 3]],
) -> Vec<usize> {
    let num_threads = requested_threads.unwrap_or_else(num_cpus::get).max(1);
    let gen_size = gates.len();
    let gens = Arc::new(gates.to_vec());

    let mut counts = vec![0usize; max_m + 1];
    counts[0] = 1;

    let mut frontier = Arc::new(SegQueue::<u32>::new());
    frontier.push(identity_ancilla_pack());

    println!("\nExact-length expansion on P(16,8) configs");

    for depth in 1..=max_m {
        let next_bits = Arc::new(atomic_bitset(ANCILLA_INJ));
        let next_frontier = Arc::new(SegQueue::<u32>::new());
        let s8_seen: Arc<Vec<AtomicU8>> =
            Arc::new((0..S8_ORDER).map(|_| AtomicU8::new(0)).collect());
        let s8_count = Arc::new(AtomicUsize::new(0));
        let layer_started = Instant::now();
        let layer_done = Arc::new(AtomicU64::new(0));
        let layer_total = frontier.len() as u64 * gen_size as u64;

        std::thread::scope(|scope| {
            for tid in 0..num_threads {
                let frontier = frontier.clone();
                let next_frontier = next_frontier.clone();
                let next_bits = next_bits.clone();
                let s8_seen = s8_seen.clone();
                let s8_count = s8_count.clone();
                let gens = gens.clone();
                let layer_done = layer_done.clone();

                scope.spawn(move || {
                    let thread_started = Instant::now();
                    let mut last_report = Instant::now();
                    let mut thread_ckts = 0u64;

                    while let Some(parent) = frontier.pop() {
                        if s8_count.load(Ordering::Relaxed) == S8_ORDER {
                            break;
                        }
                        for &gate in gens.iter() {
                            let child = apply_gate_pack(parent, gate);
                            let rank = rank_injection(child);
                            if !bit_test_and_set(&next_bits, rank) {
                                continue;
                            }
                            next_frontier.push(child);
                            if let Some(images) = induced_s8(child) {
                                let idx = lehmer_rank_s8(images);
                                if s8_seen[idx]
                                    .compare_exchange(0, 1, Ordering::Relaxed, Ordering::Relaxed)
                                    .is_ok()
                                {
                                    s8_count.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }

                        thread_ckts += gen_size as u64;
                        let ckts_done =
                            layer_done.fetch_add(gen_size as u64, Ordering::Relaxed) + gen_size as u64;
                        if last_report.elapsed() >= Duration::from_secs(2) {
                            let speed = thread_ckts as f64 / thread_started.elapsed().as_secs_f64();
                            let layer_speed =
                                ckts_done as f64 / layer_started.elapsed().as_secs_f64();
                            let remaining = layer_total.saturating_sub(ckts_done);
                            let eta = if layer_speed > 0.0 {
                                remaining as f64 / layer_speed
                            } else {
                                f64::INFINITY
                            };
                            println!(
                                "t{tid:3} exact-m:{depth:3} S8:{s8:6}/{S8_ORDER} \
                                 Q:{q:10} next:{nxt:10} {speed:8.1}k ckt/s eta:{eta:8.0}s",
                                s8 = s8_count.load(Ordering::Relaxed),
                                q = frontier.len(),
                                nxt = next_frontier.len(),
                                speed = speed / 1000.0,
                            );
                            last_report = Instant::now();
                        }
                    }
                });
            }
        });

        counts[depth] = s8_count.load(Ordering::Relaxed);
        println!(
            "exact m={depth}: {s8} clean S_8, {cfg} configs",
            s8 = counts[depth],
            cfg = next_frontier.len(),
        );
        if counts[depth] == S8_ORDER {
            for later in (depth + 1)..=max_m {
                counts[later] = 0;
            }
            break;
        }
        frontier = next_frontier;
        if frontier.len() == 0 {
            break;
        }
    }

    counts
}

/// Min-length search for `π ⊗ I`: wire 3 is the identity on every input and
/// both slices implement the same `π ∈ S_8`. A naive BFS on `S_16` explodes
/// (hundreds of millions of states by depth 7). Instead we store a forward
/// ball of radius `meet`, then meet-in-the-middle from each of the 40320
/// targets so the search is complete through `meet + extra = max_m`.
fn ancilla_s8_identity_bfs(
    max_m: usize,
    requested_threads: Option<usize>,
    gates: &[[u16; 3]],
) -> AncillaS8Result {
    assert!(!gates.is_empty(), "need at least one generator");
    assert!(max_m >= 1, "maximum gate depth must be at least one");

    let meet = max_m.min(6);
    let extra = max_m - meet;
    let num_threads = requested_threads.unwrap_or_else(num_cpus::get).max(1);
    let gen_size = gates.len();
    let gens = Arc::new(gates.to_vec());

    let ball: Arc<DashMap<u64, u8, FxBuildHasher>> = Arc::new(
        DashMap::with_capacity_and_hasher(1 << 20, FxBuildHasher),
    );
    let found_at: Arc<Vec<AtomicU8>> =
        Arc::new((0..S8_ORDER).map(|_| AtomicU8::new(u8::MAX)).collect());
    let found_count = Arc::new(AtomicUsize::new(0));

    let start = identity_perm16();
    ball.insert(start, 0);
    found_at[0].store(0, Ordering::Relaxed);
    found_count.store(1, Ordering::Relaxed);

    let mut frontier = Arc::new(SegQueue::<u64>::new());
    frontier.push(start);

    println!(
        "ancilla S8 π ⊗ I: {} generators, forward ball radius {meet}, then MITM extra {extra}",
        gen_size
    );
    println!("m     found_S8         configs          Q  new_S8");
    println!("0 {:>10} {:>15} {:>10} {:>6}", 1, 1, 1, 1);

    for depth in 1..=meet {
        if found_count.load(Ordering::Relaxed) == S8_ORDER {
            break;
        }

        let next_frontier = Arc::new(SegQueue::<u64>::new());
        let layer_started = Instant::now();
        let layer_done = Arc::new(AtomicU64::new(0));
        let layer_total = frontier.len() as u64 * gen_size as u64;
        let layer_new_s8 = Arc::new(AtomicUsize::new(0));

        std::thread::scope(|scope| {
            for tid in 0..num_threads {
                let frontier = frontier.clone();
                let next_frontier = next_frontier.clone();
                let ball = ball.clone();
                let found_at = found_at.clone();
                let found_count = found_count.clone();
                let gens = gens.clone();
                let layer_done = layer_done.clone();
                let layer_new_s8 = layer_new_s8.clone();

                scope.spawn(move || {
                    let thread_started = Instant::now();
                    let mut last_report = Instant::now();
                    let mut thread_ckts = 0u64;

                    while let Some(parent) = frontier.pop() {
                        for &gate in gens.iter() {
                            let child = apply_gate_perm16(parent, gate);
                            if ball.insert(child, depth as u8).is_some() {
                                continue;
                            }
                            next_frontier.push(child);
                            if let Some(images) = induced_pi_otimes_i(child) {
                                let idx = lehmer_rank_s8(images);
                                if found_at[idx]
                                    .compare_exchange(
                                        u8::MAX,
                                        depth as u8,
                                        Ordering::Relaxed,
                                        Ordering::Relaxed,
                                    )
                                    .is_ok()
                                {
                                    found_count.fetch_add(1, Ordering::Relaxed);
                                    layer_new_s8.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }

                        thread_ckts += gen_size as u64;
                        let ckts_done =
                            layer_done.fetch_add(gen_size as u64, Ordering::Relaxed) + gen_size as u64;
                        if last_report.elapsed() >= Duration::from_secs(2) {
                            let speed = thread_ckts as f64 / thread_started.elapsed().as_secs_f64();
                            let layer_speed =
                                ckts_done as f64 / layer_started.elapsed().as_secs_f64();
                            let remaining = layer_total.saturating_sub(ckts_done);
                            let eta = if layer_speed > 0.0 {
                                remaining as f64 / layer_speed
                            } else {
                                f64::INFINITY
                            };
                            println!(
                                "t{tid:3} m:{depth:3} S8:{s8:6}/{S8_ORDER} cfg:{cfg:12} \
                                 Q:{q:10} next:{nxt:10} {speed:8.1}k ckt/s eta:{eta:8.0}s",
                                s8 = found_count.load(Ordering::Relaxed),
                                cfg = ball.len(),
                                q = frontier.len(),
                                nxt = next_frontier.len(),
                                speed = speed / 1000.0,
                            );
                            last_report = Instant::now();
                        }
                    }
                });
            }
        });

        let new_s8 = layer_new_s8.load(Ordering::Relaxed);
        let found = found_count.load(Ordering::Relaxed);
        println!(
            "{depth} {found:>10} {configs:>15} {q:>10} {new_s8:>6}",
            configs = ball.len(),
            q = next_frontier.len(),
        );
        frontier = next_frontier;
        if frontier.len() == 0 {
            break;
        }
    }

    if found_count.load(Ordering::Relaxed) < S8_ORDER && extra > 0 {
        for extra_k in 1..=extra {
            if found_count.load(Ordering::Relaxed) == S8_ORDER {
                break;
            }
            let missing: Vec<(usize, u64)> = (0..S8_ORDER)
                .filter_map(|idx| {
                    if found_at[idx].load(Ordering::Relaxed) == u8::MAX {
                        let mut pi = [0u8; 8];
                        unrank_s8(idx, &mut pi);
                        Some((idx, pack_pi_otimes_i(pi)))
                    } else {
                        None
                    }
                })
                .collect();
            println!(
                "MITM extra={extra_k}: {} π ⊗ I still missing",
                missing.len()
            );
            let missing = Arc::new(missing);
            let chunk = (missing.len() / num_threads).max(1);
            std::thread::scope(|scope| {
                for tid in 0..num_threads {
                    let missing = missing.clone();
                    let ball = ball.clone();
                    let found_at = found_at.clone();
                    let found_count = found_count.clone();
                    let gens = gens.clone();
                    let start = tid * chunk;
                    let end = if tid + 1 == num_threads {
                        missing.len()
                    } else {
                        ((tid + 1) * chunk).min(missing.len())
                    };
                    if start >= missing.len() {
                        continue;
                    }
                    scope.spawn(move || {
                        for &(idx, target) in &missing[start..end] {
                            if let Some(dist) = meet_min_depth(target, extra_k, &ball, &gens) {
                                if dist <= max_m
                                    && found_at[idx]
                                        .compare_exchange(
                                            u8::MAX,
                                            dist as u8,
                                            Ordering::Relaxed,
                                            Ordering::Relaxed,
                                        )
                                        .is_ok()
                                {
                                    found_count.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }
                    });
                }
            });
            println!(
                "MITM extra={extra_k} done: {}/{} found",
                found_count.load(Ordering::Relaxed),
                S8_ORDER
            );
        }
    }

    let mut hist_even = vec![0usize; max_m + 1];
    let mut hist_odd = vec![0usize; max_m + 1];
    let mut unranked = [0u8; 8];
    for idx in 0..S8_ORDER {
        let d = found_at[idx].load(Ordering::Relaxed);
        if d == u8::MAX {
            continue;
        }
        unrank_s8(idx, &mut unranked);
        let depth = d as usize;
        if depth > max_m {
            continue;
        }
        if s8_perm_is_even(unranked) {
            hist_even[depth] += 1;
        } else {
            hist_odd[depth] += 1;
        }
    }

    let found = found_count.load(Ordering::Relaxed);
    println!("\nm  new_even  new_odd  cum_even  cum_odd  cum_total");
    let mut cum_even = 0usize;
    let mut cum_odd = 0usize;
    let mut min_cover = None;
    for m in 0..=max_m {
        cum_even += hist_even[m];
        cum_odd += hist_odd[m];
        let cum = cum_even + cum_odd;
        println!(
            "{m:<2} {e:8} {o:8} {cum_even:8} {cum_odd:8} {cum:9}",
            e = hist_even[m],
            o = hist_odd[m],
        );
        if min_cover.is_none() && cum == S8_ORDER {
            min_cover = Some(m);
        }
    }

    match min_cover {
        Some(m) => println!("All of S_8 has a π ⊗ I circuit of length ≤ {m}"),
        None => println!(
            "Only {found}/{S8_ORDER} of S_8 found as π ⊗ I within {max_m} gates; raise -m"
        ),
    }

    // Exact-length π ⊗ I: +2 padding on min-length, plus whether S_8 parity
    // still tracks circuit-length parity.
    let mut exact_cover = None;
    let even_only_even = (0..=max_m).step_by(2).map(|m| hist_odd[m]).sum::<usize>() == 0
        && (1..=max_m).step_by(2).map(|m| hist_even[m]).sum::<usize>() == 0;
    if even_only_even {
        println!(
            "π ⊗ I still splits by circuit parity (even π at even length, odd π at odd length); \
             no single exact m covers all of S_8"
        );
    } else {
        for m in 0..=max_m {
            let mut covered = 0usize;
            for d in (m % 2..=m).step_by(2) {
                covered += hist_even[d] + hist_odd[d];
            }
            if covered == S8_ORDER {
                exact_cover = Some(m);
                break;
            }
        }
        match exact_cover {
            Some(m) => println!(
                "Smallest exact length covering all of S_8 as π ⊗ I (same-parity +2 padding): m={m}"
            ),
            None => {
                if found == S8_ORDER {
                    println!(
                        "Min-lengths use both parities; +2 padding does not yield a single covering m ≤ {max_m}"
                    );
                }
            }
        }
    }

    AncillaS8Result {
        found,
        min_cover,
        exact_cover,
        hist_even,
        hist_odd,
        odd_cycle: !even_only_even,
    }
}

/// Shortest `dist(id, target)` using a stored forward ball: BFS from `target`
/// up to `extra` gates and take `min (d_ball(s) + d_target(s))`.
fn meet_min_depth(
    target: u64,
    extra: usize,
    ball: &DashMap<u64, u8, FxBuildHasher>,
    gens: &[[u16; 3]],
) -> Option<usize> {
    let mut best = usize::MAX;
    if let Some(d) = ball.get(&target) {
        best = *d as usize;
    }
    let mut frontier = vec![target];
    let mut seen = FxHashSet::default();
    seen.insert(target);
    for step in 1..=extra {
        if step >= best {
            break;
        }
        let mut next = Vec::new();
        for &parent in &frontier {
            for &gate in gens {
                let child = apply_gate_perm16(parent, gate);
                if !seen.insert(child) {
                    continue;
                }
                if let Some(d) = ball.get(&child) {
                    best = best.min(*d as usize + step);
                }
                next.push(child);
            }
        }
        frontier = next;
        if frontier.is_empty() {
            break;
        }
    }
    (best < usize::MAX).then_some(best)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum AncillaInit {
    /// Ancilla starts at 0; only that slice must be restored.
    Zero,
    /// Ancilla is the identity function on every input (`π ⊗ I`).
    Identity,
}

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    #[arg(short = 'n', default_value_t = 4)]
    wires: usize,

    #[arg(short = 'm', default_value = None)]
    gates: Option<usize>,

    /// Retain discovered permutations long enough to measure prefix and
    /// front-coding compression. Intended for representative-sized runs.
    #[arg(long)]
    analyze_prefixes: bool,

    /// Worker threads. Defaults to the host's logical CPU count.
    #[arg(short = 't', long)]
    threads: Option<usize>,

    /// Canonicalization backend. `brute` is slow but unambiguous; use it to
    /// validate `nauty` counts on small instances.
    #[arg(long, value_enum, default_value_t = CanonMethod::Nauty)]
    canon: CanonMethod,

    /// Exact visited-set representation. `auto` uses compact Lehmer ranks for
    /// n <= 5 and falls back to the prefix trie for larger permutations.
    #[arg(long, value_enum, default_value_t = StoreMethod::Auto)]
    store: StoreMethod,

    /// After the usual min-length BFS, expand exact-length layers of canonical
    /// permutations by appending one generator to each length-(m-1) orbit.
    /// Reports the smallest m whose m-gate circuits cover every even generated
    /// permutation and whose (m+1)-gate circuits cover every odd one.
    #[arg(long)]
    exact_cover: bool,

    /// Four-wire borrowed-ancilla cover of S_8.
    /// `zero`: ancilla starts at 0; only that slice must be restored.
    /// `identity`: ancilla is unchanged on every input (π ⊗ I on the data wires).
    /// Passing `--ancilla-s8` with no value keeps the previous `zero` behaviour.
    #[arg(
        long,
        value_enum,
        num_args = 0..=1,
        default_missing_value = "zero",
        conflicts_with = "exact_cover"
    )]
    ancilla_s8: Option<AncillaInit>,
}

fn main() {
    let args = Args::parse();
    let n = args.wires;
    let m = args.gates.unwrap_or(n * (n.ilog2() + 1) as usize);
    if let Some(init) = args.ancilla_s8 {
        let gates = base_gates(4);
        match init {
            AncillaInit::Zero => {
                let _ = ancilla_s8_bfs(m, args.threads, &gates);
            }
            AncillaInit::Identity => {
                let _ = ancilla_s8_identity_bfs(m, args.threads, &gates);
            }
        }
    } else if args.exact_cover {
        let _ = exact_cover_search(
            n,
            m,
            args.analyze_prefixes,
            args.threads,
            args.canon,
            args.store,
        );
    } else {
        let _ = iso_bfs(
            n,
            m,
            args.analyze_prefixes,
            args.threads,
            args.canon,
            args.store,
            false,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::Itertools;
    use std::{collections::HashMap, sync::atomic::AtomicUsize};

    #[test]
    fn canonical_permutation_is_wire_shuffle_invariant() {
        let n = 4;
        let circuit = CircuitSeq {
            gates: vec![[0, 1, 2], [3, 0, 1], [1, 2, 3]],
        };
        let p = circuit.perm(n);
        let mut scratch = sparsegraph::default();
        let (expected, _) = canonicalize_perm_sparse_graph(&p, &mut scratch);

        for shuffle in (0..n).permutations(n) {
            let shuffled = p.bit_shuffle(&shuffle);
            let (actual, _) = canonicalize_perm_sparse_graph(&shuffled, &mut scratch);
            assert_eq!(actual, expected);
        }

        SG_FREE(&mut scratch);
    }

    #[test]
    fn concurrent_trie_insert_has_one_winner() {
        let trie = Arc::new(PermTrie::new());
        let key = CompactPerm::U8((0..32).collect::<Vec<_>>().into_boxed_slice());
        let winners = Arc::new(AtomicUsize::new(0));

        std::thread::scope(|scope| {
            for _ in 0..8 {
                let trie = trie.clone();
                let key = key.clone();
                let winners = winners.clone();
                scope.spawn(move || {
                    for _ in 0..100 {
                        if trie.insert(&key).1 {
                            winners.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                });
            }
        });

        assert_eq!(winners.load(Ordering::Relaxed), 1);
        assert_eq!(trie.permutation_count(), 1);
        assert_eq!(trie.node_count(), 33); // root + one node per value
    }

    #[test]
    fn lehmer_rank_matches_lexicographic_order() {
        for n in 1..=8 {
            for (expected, permutation) in (0..n).permutations(n).enumerate() {
                let key = CompactPerm::U8(
                    permutation
                        .into_iter()
                        .map(|value| value as u8)
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                );
                assert_eq!(lehmer_rank_u128(&key), Some(expected as u128));
            }
        }
    }

    #[test]
    fn concurrent_lehmer_insert_has_one_winner() {
        let visited = Arc::new(Visited::new(StoreMethod::Lehmer128, 32));
        let key = CompactPerm::U8((0..32).collect::<Vec<_>>().into_boxed_slice());
        let winners = Arc::new(AtomicUsize::new(0));

        std::thread::scope(|scope| {
            for _ in 0..8 {
                let visited = visited.clone();
                let key = key.clone();
                let winners = winners.clone();
                scope.spawn(move || {
                    for _ in 0..100 {
                        if visited.insert(&key) {
                            winners.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                });
            }
        });

        assert_eq!(winners.load(Ordering::Relaxed), 1);
        assert_eq!(visited.len(), 1);
    }

    #[test]
    fn base_gates_are_involutions() {
        // Justifies frontier-only BFS: g = g^{-1} makes the Cayley graph
        // undirected, so a neighbour of a depth-d state has depth in
        // {d-1, d, d+1}.
        for n in 2..=6 {
            let identity = Permutation::id_perm(1 << n);
            for gate in base_gates(n) {
                let g = CircuitSeq { gates: vec![gate] }.perm(n);
                assert_eq!(
                    g.compose(&g),
                    identity,
                    "gate {gate:?} is not an involution"
                );
            }
        }
    }

    #[test]
    fn packed_key_round_trips() {
        for n in 3..=6 {
            let nn = 1 << n;
            let bits = value_bits(nn);
            for p in random_perms(n, 256, 8) {
                let key = CompactPerm::from_permutation(&p);
                let packed = pack_perm_u384(&key, bits);
                let restored = unpack_perm_u384(&packed, nn, bits);
                assert_eq!(restored, key, "packed key round-trip failed for n={n}");
            }
        }
    }

    #[test]
    fn packed_store_matches_known_counts() {
        // n = 6 exercises the bit-packed backend end to end.
        let expected = vec![1, 1, 31, 1536];
        for threads in [1, 4] {
            let actual = iso_bfs(
                6,
                3,
                false,
                Some(threads),
                CanonMethod::Nauty,
                StoreMethod::Packed384,
                false,
            )
            .0;
            assert_eq!(actual, expected, "wrong spheres with {threads} workers");
        }
    }

    #[test]
    fn parallel_bfs_is_level_synchronous() {
        let expected = vec![1, 1, 22, 369, 6544, 111_903];
        for threads in [1, 8] {
            let actual = iso_bfs(
                4,
                5,
                false,
                Some(threads),
                CanonMethod::Nauty,
                StoreMethod::Lehmer128,
                false,
            )
            .0;
            assert_eq!(actual, expected, "wrong spheres with {threads} workers");
        }
    }

    /// Deterministic pseudo-random reversible functions: compose `k` random
    /// base gates onto the identity using a fixed-seed xorshift.
    fn random_perms(n: usize, count: usize, gates_per: usize) -> Vec<Permutation> {
        let gate_perms: Vec<Permutation> = base_gates(n)
            .iter()
            .copied()
            .map(|g| CircuitSeq { gates: vec![g] }.perm(n))
            .collect();

        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        (0..count)
            .map(|_| {
                let mut p = Permutation::id_perm(1 << n);
                for _ in 0..gates_per {
                    let g = &gate_perms[(next() as usize) % gate_perms.len()];
                    p = g.compose(&p);
                }
                p
            })
            .collect()
    }

    /// nauty must be a pure function of its input: identical calls, identical
    /// output. A failure here indicates internal RNG / uninitialized state.
    #[test]
    fn nauty_canon_is_pure() {
        let n = 5;
        let mut scratch = sparsegraph::default();
        for p in random_perms(n, 64, 6) {
            let (first, first_sphere) = canonicalize_perm_sparse_graph(&p, &mut scratch);
            for _ in 0..50 {
                let (again, sphere) = canonicalize_perm_sparse_graph(&p, &mut scratch);
                assert_eq!(again, first, "nauty canonicalization not deterministic");
                assert_eq!(sphere, first_sphere, "nauty orbit size not deterministic");
            }
        }
        SG_FREE(&mut scratch);
    }

    /// The decisive check: nauty and the unambiguous brute-force canonicalizer
    /// must induce the *same* partition of functions into wire-orbits, and
    /// agree on orbit sizes. If nauty ever splits or merges an orbit, the two
    /// canonical maps will disagree on which perms share a representative.
    #[test]
    fn nauty_matches_brute_partition() {
        let n = 5;
        let mut scratch = sparsegraph::default();

        // Map each brute-force representative to the nauty representative we saw
        // for it. Any inconsistency means the two disagree on the partition.
        let mut brute_to_nauty: HashMap<Vec<usize>, Vec<usize>> = HashMap::new();
        let mut nauty_to_brute: HashMap<Vec<usize>, Vec<usize>> = HashMap::new();

        for p in random_perms(n, 512, 8) {
            let (nauty_canon, nauty_sphere) = canonicalize_perm_sparse_graph(&p, &mut scratch);
            let (brute_canon, brute_sphere) = canonicalize_perm_brute(&p, n);

            assert_eq!(
                nauty_sphere, brute_sphere,
                "orbit size mismatch: nauty {nauty_sphere} vs brute {brute_sphere}"
            );

            let nk = nauty_canon.data.clone();
            let bk = brute_canon.data.clone();

            if let Some(prev) = brute_to_nauty.insert(bk.clone(), nk.clone()) {
                assert_eq!(
                    prev, nk,
                    "same brute-orbit mapped to two nauty reps (split)"
                );
            }
            if let Some(prev) = nauty_to_brute.insert(nk.clone(), bk.clone()) {
                assert_eq!(prev, bk, "two brute-orbits mapped to one nauty rep (merge)");
            }
        }

        SG_FREE(&mut scratch);
    }

    #[test]
    fn identity_is_even_transposition_is_odd() {
        let id = Permutation::id_perm(8);
        assert!(perm_is_even(&id));

        let mut swap = Permutation::id_perm(8);
        swap.data.swap(0, 1);
        assert!(!perm_is_even(&swap));

        // 3-cycle is even.
        let mut three = Permutation::id_perm(8);
        three.data[0] = 1;
        three.data[1] = 2;
        three.data[2] = 0;
        assert!(perm_is_even(&three));
    }

    #[test]
    fn iso_bfs_collect_matches_layer_counts() {
        let (counts, orbits) = iso_bfs(
            3,
            4,
            false,
            Some(2),
            CanonMethod::Nauty,
            StoreMethod::Auto,
            true,
        );
        let mut by_depth = vec![0usize; counts.len()];
        for orbit in &orbits {
            by_depth[orbit.min_depth] += 1;
        }
        assert_eq!(by_depth, counts);
        assert_eq!(orbits.len(), counts.iter().sum::<usize>());
    }

    /// Enumerate every length-`m` circuit, canonicalize, and count unique orbits.
    fn brute_exact_canonical_count(n: usize, m: usize) -> usize {
        let gens: Vec<Permutation> = base_gates(n)
            .iter()
            .copied()
            .map(|g| CircuitSeq { gates: vec![g] }.perm(n))
            .collect();
        let mut scratch = sparsegraph::default();
        let mut unique = std::collections::HashSet::new();

        fn rec(
            remaining: usize,
            current: &Permutation,
            gens: &[Permutation],
            scratch: &mut sparsegraph,
            unique: &mut std::collections::HashSet<Vec<usize>>,
        ) {
            if remaining == 0 {
                let (canon, _) = canonicalize_perm_sparse_graph(current, scratch);
                unique.insert(canon.data);
                return;
            }
            for g in gens {
                rec(remaining - 1, &g.compose(current), gens, scratch, unique);
            }
        }

        rec(
            m,
            &Permutation::id_perm(1 << n),
            &gens,
            &mut scratch,
            &mut unique,
        );
        SG_FREE(&mut scratch);
        unique.len()
    }

    #[test]
    fn exact_cover_matches_brute_enumeration() {
        let result = exact_cover_search(3, 3, false, Some(2), CanonMethod::Nauty, StoreMethod::Auto);
        for m in 0..=3 {
            let brute = brute_exact_canonical_count(3, m);
            assert_eq!(
                result.exact_counts[m], brute,
                "exact-length canonical count mismatch at m={m}"
            );
        }
        // Padding with an involution pair: identity is realized at every even length.
        assert_eq!(result.exact_counts[0], 1);
        assert_eq!(result.exact_spheres[0], 1);
        assert!(result.exact_counts[2] >= 2);
        // Relative to the radius-3 ball (not all of Alt_8), even orbits are
        // covered at exact length 2 and odd orbits at exact length 3.
        assert_eq!(result.cover_m, Some(2));
    }

    #[test]
    fn exact_even_layers_are_even_permutations() {
        let n = 3;
        let max_m = 3;
        let (_, orbits) = iso_bfs(
            n,
            max_m,
            false,
            Some(2),
            CanonMethod::Nauty,
            StoreMethod::Auto,
            true,
        );
        for orbit in orbits {
            let even = perm_is_even(&orbit.perm.to_permutation());
            assert_eq!(
                even,
                orbit.min_depth % 2 == 0,
                "parity disagrees with circuit length"
            );
        }
    }

    #[test]
    fn lehmer_s8_round_trips() {
        for rank in 0..S8_ORDER {
            let mut images = [0u8; 8];
            unrank_s8(rank, &mut images);
            assert_eq!(lehmer_rank_s8(images), rank);
        }
        let mut seen = std::collections::HashSet::new();
        for rank in 0..S8_ORDER {
            let mut images = [0u8; 8];
            unrank_s8(rank, &mut images);
            assert!(seen.insert(images), "lehmer unrank collision at {rank}");
        }
    }

    #[test]
    fn rank_injection_identity_is_zero() {
        let id = identity_ancilla_pack();
        assert_eq!(rank_injection(id), 0);
        assert_eq!(induced_s8(id), Some([0, 1, 2, 3, 4, 5, 6, 7]));
        assert_eq!(lehmer_rank_s8([0, 1, 2, 3, 4, 5, 6, 7]), 0);

        let mut seen = std::collections::HashSet::new();
        for k in 0..16u8 {
            let mut vals = [0u8; 8];
            let mut used = 0u16;
            let mut slot = 0;
            for offset in 0..16u8 {
                let v = k.wrapping_add(offset) % 16;
                if used & (1 << v) == 0 && slot < 8 {
                    vals[slot] = v;
                    used |= 1 << v;
                    slot += 1;
                }
            }
            let mut packed = 0u32;
            for i in 0..8 {
                packed |= (vals[i] as u32) << (4 * i);
            }
            let r = rank_injection(packed);
            assert!(r < ANCILLA_INJ);
            assert!(seen.insert(r), "injection rank collision");
        }
    }

    #[test]
    fn ancilla_bfs_on_three_wire_gates_matches_known_spheres() {
        let gates: Vec<[u16; 3]> = base_gates(4)
            .into_iter()
            .filter(|g| g.iter().all(|&w| w < 3))
            .collect();
        assert_eq!(gates.len(), 6);
        let result = ancilla_s8_bfs(4, Some(2), &gates);
        assert_eq!(result.hist_even[0], 1);
        assert_eq!(result.hist_odd[1], 6);
        assert_eq!(result.hist_even[2], 27);
        assert_eq!(result.hist_odd[3], 120);
        assert_eq!(result.hist_even[4], 528);
        assert_eq!(result.hist_odd[0] + result.hist_odd[2] + result.hist_odd[4], 0);
        assert_eq!(result.hist_even[1] + result.hist_even[3], 0);
    }

    #[test]
    fn pi_otimes_i_identity_and_three_wire_gate() {
        let id = identity_perm16();
        assert_eq!(induced_pi_otimes_i(id), Some([0, 1, 2, 3, 4, 5, 6, 7]));

        let three_wire = apply_gate_perm16(id, [0, 1, 2]);
        let pi = induced_pi_otimes_i(three_wire).expect("3-wire gate should be π ⊗ I");
        assert_ne!(pi, [0, 1, 2, 3, 4, 5, 6, 7]);
        assert!(!s8_perm_is_even(pi));

        // Ancilla as negative control: fires always on the 0-slice and only
        // sometimes on the 1-slice, so π_0 ≠ π_1.
        let mixed = apply_gate_perm16(id, [0, 1, 3]);
        assert!(induced_pi_otimes_i(mixed).is_none());
    }

    #[test]
    fn identity_ancilla_bfs_on_three_wire_gates_matches_known_spheres() {
        let gates: Vec<[u16; 3]> = base_gates(4)
            .into_iter()
            .filter(|g| g.iter().all(|&w| w < 3))
            .collect();
        let result = ancilla_s8_identity_bfs(4, Some(2), &gates);
        assert_eq!(result.hist_even[0], 1);
        assert_eq!(result.hist_odd[1], 6);
        assert_eq!(result.hist_even[2], 27);
        assert_eq!(result.hist_odd[3], 120);
        assert_eq!(result.hist_even[4], 528);
        assert_eq!(result.hist_odd[0] + result.hist_odd[2] + result.hist_odd[4], 0);
        assert_eq!(result.hist_even[1] + result.hist_even[3], 0);
    }

    fn s8_cycle_type(pi: [u8; 8]) -> [u8; 8] {
        let mut seen = [false; 8];
        let mut lengths = [0u8; 8];
        let mut n = 0usize;
        for start in 0..8 {
            if seen[start] {
                continue;
            }
            let mut len = 0u8;
            let mut j = start;
            while !seen[j] {
                seen[j] = true;
                j = pi[j] as usize;
                len += 1;
            }
            lengths[n] = len;
            n += 1;
        }
        lengths[..n].sort_by(|a, b| b.cmp(a));
        lengths
    }

    fn classify_exact_perm16_s8(
        layer: &FxHashSet<u64>,
    ) -> (
        Vec<bool>,
        Vec<bool>,
        usize,
        usize,
        usize,
        usize,
    ) {
        let mut seen_otimes = vec![false; S8_ORDER];
        let mut seen_borrowed = vec![false; S8_ORDER];
        let mut otimes_even = 0usize;
        let mut otimes_odd = 0usize;
        let mut borrowed_even = 0usize;
        let mut borrowed_odd = 0usize;
        for &packed in layer {
            if let Some(pi) = induced_pi_otimes_i(packed) {
                let idx = lehmer_rank_s8(pi);
                if !seen_otimes[idx] {
                    seen_otimes[idx] = true;
                    if s8_perm_is_even(pi) {
                        otimes_even += 1;
                    } else {
                        otimes_odd += 1;
                    }
                }
            }
            let mut slice = 0u32;
            for i in 0..8 {
                slice |= ((packed >> (4 * i)) as u32 & 0xF) << (4 * i);
            }
            if let Some(pi) = induced_s8(slice) {
                let idx = lehmer_rank_s8(pi);
                if !seen_borrowed[idx] {
                    seen_borrowed[idx] = true;
                    if s8_perm_is_even(pi) {
                        borrowed_even += 1;
                    } else {
                        borrowed_odd += 1;
                    }
                }
            }
        }
        (
            seen_otimes,
            seen_borrowed,
            otimes_even,
            otimes_odd,
            borrowed_even,
            borrowed_odd,
        )
    }

    #[test]
    fn even_s8_in_odd_four_wire_layers() {
        let gens = base_gates(4);
        let mut layer = FxHashSet::default();
        layer.insert(identity_perm16());
        let mut even_otimes_by_k: [Option<Vec<bool>>; 6] = Default::default();
        let mut counts = [(0usize, 0usize, 0usize, 0usize); 6];
        counts[0] = (1, 0, 1, 0);

        for k in 1..=5 {
            let mut next =
                FxHashSet::with_capacity_and_hasher(layer.len() * gens.len(), Default::default());
            for &state in &layer {
                for &gate in &gens {
                    next.insert(apply_gate_perm16(state, gate));
                }
            }
            layer = next;
            let (otimes, _borrowed, oe, oo, be, bo) = classify_exact_perm16_s8(&layer);
            counts[k] = (oe, oo, be, bo);
            println!(
                "exact {k}: 16-perms={}  π⊗I even/odd={oe}/{oo}  borrowed even/odd={be}/{bo}",
                layer.len()
            );
            even_otimes_by_k[k] = Some(otimes);
        }

        assert_eq!(counts[1], (0, 6, 9, 6));
        assert_eq!(counts[2], (28, 0, 76, 78));
        assert_eq!(counts[3], (0, 132, 613, 603));
        assert_eq!(counts[4], (604, 6, 3368, 3383));
        assert_eq!(counts[5], (111, 2516, 12263, 12278));

        let at5 = even_otimes_by_k[5].as_ref().unwrap();
        let at2 = even_otimes_by_k[2].as_ref().unwrap();
        let at4 = even_otimes_by_k[4].as_ref().unwrap();
        let mut already_even_len = 0usize;
        let mut only_odd5 = 0usize;
        let mut types: std::collections::BTreeMap<[u8; 8], usize> =
            std::collections::BTreeMap::new();
        let mut unranked = [0u8; 8];
        for idx in 0..S8_ORDER {
            if !at5[idx] || !s8_perm_is_even({
                unrank_s8(idx, &mut unranked);
                unranked
            }) {
                continue;
            }
            unrank_s8(idx, &mut unranked);
            *types.entry(s8_cycle_type(unranked)).or_default() += 1;
            if at2[idx] || at4[idx] {
                already_even_len += 1;
            } else {
                only_odd5 += 1;
            }
        }
        println!(
            "exact-5 even π⊗I: also even-length ≤4: {already_even_len}; first seen at odd 5: {only_odd5}"
        );
        for (ct, n) in &types {
            let parts: Vec<String> = ct.iter().filter(|&&l| l > 0).map(|l| l.to_string()).collect();
            println!("  cycle type [{}]: {n}", parts.join(","));
        }
        assert_eq!(already_even_len, 39);
        assert_eq!(only_odd5, 72);
        assert_eq!(types.get(&[2, 2, 1, 1, 1, 1, 0, 0]).copied(), Some(15));
        assert_eq!(types.get(&[3, 2, 2, 1, 0, 0, 0, 0]).copied(), Some(24));
        assert_eq!(types.get(&[4, 2, 1, 1, 0, 0, 0, 0]).copied(), Some(72));
    }

    #[test]
    fn odd_four_wire_identity_minimum_is_seven() {
        let gens = base_gates(4);
        let id = identity_perm16();
        assert_eq!(gens.len(), 24);

        for &k in &[1usize, 3, 5] {
            let layer = exact_perm16_layer(k, &gens);
            assert!(
                !layer.contains(&id),
                "found a {k}-gate 4-wire identity; 7 would not be minimal"
            );
        }

        // Length 7: some permutation is both a 3-fold and a 4-fold product,
        // so concatenating those walks is a 7-gate identity.
        let layer3 = exact_perm16_layer(3, &gens);
        let layer4 = exact_perm16_layer(4, &gens);
        let meet = layer3.intersection(&layer4).next().is_some();
        assert!(
            meet,
            "no 7-gate 4-wire identity (3-fold ∩ 4-fold products is empty)"
        );
    }

    #[test]
    fn perm16_exact_layer_intersections() {
        let gens = base_gates(4);
        let mut layers: Vec<FxHashSet<u64>> = Vec::with_capacity(6);
        let mut layer = FxHashSet::default();
        layer.insert(identity_perm16());
        layers.push(layer.clone());
        for k in 1..=5 {
            let mut next =
                FxHashSet::with_capacity_and_hasher(layer.len() * gens.len(), Default::default());
            for &state in &layer {
                for &gate in &gens {
                    next.insert(apply_gate_perm16(state, gate));
                }
            }
            layer = next;
            for (i, old) in layers.iter().enumerate() {
                let inter: Vec<u64> = layer.intersection(old).copied().collect();
                let n_pi = inter
                    .iter()
                    .filter(|&&p| induced_pi_otimes_i(p).is_some())
                    .count();
                println!(
                    "L{i} ∩ L{k} = {} (π⊗I {n_pi}), |L{k}|={}",
                    inter.len(),
                    layer.len()
                );
            }
            layers.push(layer.clone());
        }
    }

    fn invert_perm16(packed: u64) -> u64 {
        let mut out = 0u64;
        for i in 0..16u64 {
            let dest = (packed >> (4 * i)) & 0xF;
            out |= i << (4 * dest);
        }
        out
    }

    fn compose_perm16(p: u64, q: u64) -> u64 {
        let mut out = 0u64;
        for i in 0..16u64 {
            let mid = (q >> (4 * i)) & 0xF;
            let dest = (p >> (4 * mid)) & 0xF;
            out |= dest << (4 * i);
        }
        out
    }

    fn layer_contains_via_meet(target: u64, half: &FxHashSet<u64>) -> bool {
        for &a in half {
            let b = compose_perm16(invert_perm16(a), target);
            if half.contains(&b) {
                return true;
            }
        }
        false
    }

    #[test]
    fn generators_in_l6_and_expandable_pairs() {
        let gens = base_gates(4);
        let id = identity_perm16();
        assert_eq!(compose_perm16(id, id), id);
        assert_eq!(invert_perm16(id), id);

        let l2 = exact_perm16_layer(2, &gens);
        let l3 = exact_perm16_layer(3, &gens);
        let l5 = exact_perm16_layer(5, &gens);

        let mut n_gen_l6 = 0usize;
        let mut n_pi_gen_l6 = 0usize;
        for &gate in &gens {
            let g = apply_gate_perm16(id, gate);
            if layer_contains_via_meet(g, &l3) {
                n_gen_l6 += 1;
                if induced_pi_otimes_i(g).is_some() {
                    n_pi_gen_l6 += 1;
                    println!("3-wire generator {gate:?} is in L6");
                }
            }
        }
        println!("generators in L6: {n_gen_l6}/24 (π⊗I {n_pi_gen_l6})");

        let mut expandable_pi = 0usize;
        for &p in l2.intersection(&l5) {
            if let Some(pi) = induced_pi_otimes_i(p) {
                expandable_pi += 1;
                println!(
                    "L2∩L5 π⊗I cycle {:?} even={}",
                    s8_cycle_type(pi),
                    s8_perm_is_even(pi)
                );
            }
        }
        println!("L2∩L5 π⊗I count {expandable_pi}");

        let three: Vec<[u16; 3]> = gens
            .iter()
            .copied()
            .filter(|g| g.iter().all(|&w| w < 3))
            .collect();
        assert_eq!(three.len(), 6);
        let mut three_l2 = FxHashSet::default();
        for &g in &three {
            let p = apply_gate_perm16(id, g);
            for &h in &three {
                three_l2.insert(apply_gate_perm16(p, h));
            }
        }
        let n_three_expand = three_l2.intersection(&l5).filter(|&&p| p != id).count();
        println!(
            "non-id 3-wire length-2 products also in 4-wire L5: {n_three_expand}/{}",
            three_l2.len().saturating_sub(1)
        );

        // Can every 3-wire min-length-9 odd π use one of those 3 pairs on a geodesic?
        let expandable: FxHashSet<u64> = three_l2
            .intersection(&l5)
            .copied()
            .filter(|&p| p != id)
            .collect();
        assert_eq!(expandable.len(), 3);

        let mut dist = vec![u8::MAX; S8_ORDER];
        let mut packed_at = vec![0u64; S8_ORDER];
        let mut preds: Vec<Vec<(usize, [u16; 3])>> = vec![Vec::new(); S8_ORDER];
        dist[0] = 0;
        packed_at[0] = id;
        let mut frontier = vec![0usize];
        while let Some(p_idx) = {
            if frontier.is_empty() {
                None
            } else {
                Some(frontier.remove(0))
            }
        } {
            if dist[p_idx] >= 9 {
                continue;
            }
            let p = packed_at[p_idx];
            for &gate in &three {
                let child = apply_gate_perm16(p, gate);
                let chi = induced_pi_otimes_i(child).unwrap();
                let idx = lehmer_rank_s8(chi);
                let nd = dist[p_idx] + 1;
                if nd > dist[idx] {
                    continue;
                }
                if nd < dist[idx] {
                    dist[idx] = nd;
                    packed_at[idx] = child;
                    preds[idx].clear();
                    frontier.push(idx);
                }
                if nd == dist[idx] {
                    preds[idx].push((p_idx, gate));
                }
            }
        }

        let mut last_two = vec![false; S8_ORDER];
        for idx in 0..S8_ORDER {
            if dist[idx] < 2 {
                continue;
            }
            for &(u, g) in &preds[idx] {
                for &(w, h) in &preds[u] {
                    let pair = apply_gate_perm16(apply_gate_perm16(id, h), g);
                    if expandable.contains(&pair) && dist[w] + 2 == dist[idx] {
                        last_two[idx] = true;
                    }
                }
            }
        }

        let mut can_expand = last_two.clone();
        let mut order: Vec<usize> = (0..S8_ORDER).filter(|&i| dist[i] != u8::MAX).collect();
        order.sort_by_key(|&i| dist[i]);
        for idx in order {
            if can_expand[idx] {
                continue;
            }
            for &(u, _) in &preds[idx] {
                if dist[u] + 1 == dist[idx] && can_expand[u] {
                    can_expand[idx] = true;
                    break;
                }
            }
        }

        let mut n9_odd = 0usize;
        let mut n9_odd_expand = 0usize;
        let mut stubborn = Vec::new();
        let mut unranked = [0u8; 8];
        for idx in 0..S8_ORDER {
            if dist[idx] != 9 {
                continue;
            }
            unrank_s8(idx, &mut unranked);
            if s8_perm_is_even(unranked) {
                continue;
            }
            n9_odd += 1;
            if can_expand[idx] {
                n9_odd_expand += 1;
            } else {
                stubborn.push(packed_at[idx]);
            }
        }
        println!("3-wire min-length-9 odds: {n9_odd}, geodesic with expandable pair: {n9_odd_expand}");
        assert_eq!(n9_odd - n9_odd_expand, stubborn.len());

        let l4 = exact_perm16_layer(4, &gens);
        for (i, &t) in stubborn.iter().enumerate() {
            let pi = induced_pi_otimes_i(t).unwrap();
            let in_l3 = l3.contains(&t);
            let in_l5 = l5.contains(&t);
            let mut in_l7 = false;
            for &a in &l3 {
                let b = compose_perm16(invert_perm16(a), t);
                if l4.contains(&b) {
                    in_l7 = true;
                    break;
                }
            }
            println!(
                "stubborn {i}: even={} in L3={in_l3} L5={in_l5} L7={in_l7} cycle {:?}",
                s8_perm_is_even(pi),
                s8_cycle_type(pi)
            );
        }

        drop(l2);
        drop(l3);
        drop(l4);
        drop(l5);
        let l6 = exact_perm16_layer(6, &gens);
        println!("|L6|={}", l6.len());
        for (i, &t) in stubborn.iter().enumerate() {
            let in_l12 = layer_contains_via_meet(t, &l6);
            println!("stubborn {i} in L12={in_l12}");
            assert!(in_l12, "stubborn min-length-9 odd not in L12");
        }
        assert_eq!(stubborn.len(), 4);

        assert_eq!(n_gen_l6, 24);
        assert_eq!(n_pi_gen_l6, 6);
    }

    #[test]
    fn min10_even_pi_otimes_i_not_in_l11() {
        let gens = base_gates(4);
        let three: Vec<[u16; 3]> = gens
            .iter()
            .copied()
            .filter(|g| g.iter().all(|&w| w < 3))
            .collect();
        let id = identity_perm16();
        let mut dist = vec![u8::MAX; S8_ORDER];
        let mut packed_at = vec![0u64; S8_ORDER];
        dist[0] = 0;
        packed_at[0] = id;
        let mut frontier = vec![0usize];
        let mut qhead = 0usize;
        while qhead < frontier.len() {
            let p_idx = frontier[qhead];
            qhead += 1;
            if dist[p_idx] >= 10 {
                continue;
            }
            let p = packed_at[p_idx];
            for &gate in &three {
                let child = apply_gate_perm16(p, gate);
                let chi = induced_pi_otimes_i(child).unwrap();
                let idx = lehmer_rank_s8(chi);
                let nd = dist[p_idx] + 1;
                if nd < dist[idx] {
                    dist[idx] = nd;
                    packed_at[idx] = child;
                    frontier.push(idx);
                }
            }
        }

        let mut targets = Vec::new();
        let mut types: std::collections::BTreeMap<[u8; 8], usize> =
            std::collections::BTreeMap::new();
        let mut unranked = [0u8; 8];
        for idx in 0..S8_ORDER {
            if dist[idx] != 10 {
                continue;
            }
            unrank_s8(idx, &mut unranked);
            if !s8_perm_is_even(unranked) {
                continue;
            }
            targets.push(packed_at[idx]);
            *types.entry(s8_cycle_type(unranked)).or_default() += 1;
            println!("3-wire d=10 even π={unranked:?} cycle {:?}", s8_cycle_type(unranked));
        }
        println!("3-wire min-length-10 even: {}  cycle types {:?}", targets.len(), types);
        assert_eq!(targets.len(), 56);

        let mut layer = FxHashSet::default();
        layer.insert(id);
        let mut l3 = FxHashSet::default();
        let mut l4 = FxHashSet::default();
        let mut l5 = FxHashSet::default();
        for k in 1..=6 {
            let mut next =
                FxHashSet::with_capacity_and_hasher(layer.len() * gens.len(), Default::default());
            for &state in &layer {
                for &gate in &gens {
                    next.insert(apply_gate_perm16(state, gate));
                }
            }
            layer = next;
            println!("built L{k} {}", layer.len());
            match k {
                3 => l3 = layer.clone(),
                4 => l4 = layer.clone(),
                5 => l5 = layer.clone(),
                _ => {}
            }
        }
        let l6 = layer;

        let in_meet = |t: u64, a: &FxHashSet<u64>, b: &FxHashSet<u64>| -> bool {
            a.iter().any(|&x| b.contains(&compose_perm16(invert_perm16(x), t)))
        };

        let mut bottlenecks = Vec::new();
        for &t in &targets {
            let pi = induced_pi_otimes_i(t).unwrap();
            let in_l4 = l4.contains(&t);
            let in_l6 = l6.contains(&t);
            let in_l8 = in_meet(t, &l4, &l4);
            let in_l5 = l5.contains(&t);
            let in_l7 = in_meet(t, &l3, &l4);
            let in_l9 = in_meet(t, &l4, &l5);
            let still_10 = !in_l4 && !in_l6 && !in_l8 && !in_l5 && !in_l7 && !in_l9;
            if still_10 {
                bottlenecks.push(t);
                println!(
                    "bottleneck π={:?} cycle {:?}",
                    pi,
                    s8_cycle_type(pi)
                );
            }
        }
        println!(
            "still min-length 10 as π⊗I: {} / {}",
            bottlenecks.len(),
            targets.len()
        );
        assert_eq!(bottlenecks.len(), 16);

        let mut n_hit = 0usize;
        let mut hit = vec![false; bottlenecks.len()];
        for &a in &l5 {
            let ainv = invert_perm16(a);
            for (i, &t) in bottlenecks.iter().enumerate() {
                if hit[i] {
                    continue;
                }
                if l6.contains(&compose_perm16(ainv, t)) {
                    hit[i] = true;
                    n_hit += 1;
                }
            }
            if n_hit == bottlenecks.len() {
                break;
            }
        }
        println!("bottleneck even π⊗I in L11: {n_hit}/{}", bottlenecks.len());
        assert_eq!(n_hit, bottlenecks.len(), "diameter-10 even π⊗I should all sit in L11");

        let mut d8 = Vec::new();
        for idx in 0..S8_ORDER {
            if dist[idx] != 8 {
                continue;
            }
            unrank_s8(idx, &mut unranked);
            if s8_perm_is_even(unranked) {
                d8.push(packed_at[idx]);
            }
        }
        println!("3-wire min-length-8 even: {}", d8.len());
        let mut d8_exact = Vec::new();
        let mut d8_types: std::collections::BTreeMap<[u8; 8], usize> =
            std::collections::BTreeMap::new();
        for &t in &d8 {
            let pi = induced_pi_otimes_i(t).unwrap();
            let in_l4 = l4.contains(&t);
            let in_l6 = l6.contains(&t);
            let in_l5 = l5.contains(&t);
            let in_l7 = in_meet(t, &l3, &l4);
            let in_l9 = in_meet(t, &l4, &l5);
            if !in_l4 && !in_l6 && !in_l5 && !in_l7 && !in_l9 {
                d8_exact.push(t);
                *d8_types.entry(s8_cycle_type(pi)).or_default() += 1;
            }
        }
        let bottleneck_set: FxHashSet<u64> = bottlenecks.iter().copied().collect();
        for &t in &targets {
            if bottleneck_set.contains(&t) {
                continue;
            }
            let pi = induced_pi_otimes_i(t).unwrap();
            let in_l4 = l4.contains(&t);
            let in_l6 = l6.contains(&t);
            let in_l5 = l5.contains(&t);
            let in_l7 = in_meet(t, &l3, &l4);
            let in_l9 = in_meet(t, &l4, &l5);
            if !in_l4 && !in_l6 && !in_l5 && !in_l7 && !in_l9 {
                d8_exact.push(t);
                *d8_types.entry(s8_cycle_type(pi)).or_default() += 1;
            }
        }
        println!(
            "still min-length 8 as π⊗I (no odd length ≤9): {} / {}  types {:?}",
            d8_exact.len(),
            d8.len(),
            d8_types
        );

        let mut n8_hit = 0usize;
        let mut hit8 = vec![false; d8_exact.len()];
        for &a in &l5 {
            let ainv = invert_perm16(a);
            for (i, &t) in d8_exact.iter().enumerate() {
                if hit8[i] {
                    continue;
                }
                if l6.contains(&compose_perm16(ainv, t)) {
                    hit8[i] = true;
                    n8_hit += 1;
                }
            }
            if n8_hit == d8_exact.len() && !d8_exact.is_empty() {
                break;
            }
        }
        println!("min-length-8 even π⊗I in L11: {n8_hit}/{}", d8_exact.len());
        if n8_hit < d8_exact.len() {
            for (i, &t) in d8_exact.iter().enumerate() {
                if !hit8[i] {
                    let pi = induced_pi_otimes_i(t).unwrap();
                    println!("L11 miss π={:?} cycle {:?}", pi, s8_cycle_type(pi));
                }
            }
        }
        assert_eq!(n8_hit, d8_exact.len());
    }
}
