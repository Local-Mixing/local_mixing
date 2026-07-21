use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use clap::{Parser, ValueEnum};
use crossbeam::queue::SegQueue;
use dashmap::mapref::entry::Entry;
use dashmap::{DashMap, DashSet};
use local_mixing::circuit::{CircuitSeq, Permutation, base_gates, circuit::iter_ones};
use nauty_Traces_sys::{SG_FREE, SparseGraph, optionblk, sparsegraph, sparsenauty, statsblk};
use num_bigint::BigUint;
use rustc_hash::FxBuildHasher;

fn factorial(n: usize) -> Option<usize> {
    (1..=n).try_fold(1, usize::checked_mul)
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

fn iso_bfs(
    n: usize,
    max_m: usize,
    analyze_prefixes_enabled: bool,
    requested_threads: Option<usize>,
    canon_method: CanonMethod,
    store_method: StoreMethod,
) -> Vec<usize> {
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

    // Seed: identity at depth 0 (recorded, not expanded).
    let mut seed_canonical_graph = sparsegraph::default();
    let id_perm = Permutation::id_perm(nn);
    let (id_canon, _) = canonicalize(canon_method, &id_perm, n, &mut seed_canonical_graph);
    let id_key = CompactPerm::from_permutation(&id_canon);
    let inserted = visited.insert(&id_key);
    assert!(inserted);
    if let Some(analysis) = &analysis {
        analysis.lock().unwrap()[0].push(id_key.clone());
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

    layer_counts
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
}

fn main() {
    let args = Args::parse();
    let n = args.wires;
    let m = args.gates.unwrap_or(n * (n.ilog2() + 1) as usize);
    let _ = iso_bfs(
        n,
        m,
        args.analyze_prefixes,
        args.threads,
        args.canon,
        args.store,
    );
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
            );
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
            );
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
}
