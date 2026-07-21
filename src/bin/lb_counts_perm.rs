use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    thread::sleep,
    time::{Duration, Instant},
};

use clap::Parser;
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

/// Packed truth-table. For domain size ≤ 256 (`n ≤ 8`) each image fits in a
/// `u8`; otherwise `u16` (enough through `n = 16`).
#[derive(Hash, PartialEq, Eq, Clone)]
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

/// Full permutations are retained only while on the active frontier.
struct WorkItem {
    depth: u16,
    perm: CompactPerm,
}

fn iso_bfs(n: usize, max_m: usize) {
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

    let trie = Arc::new(PermTrie::new());
    let dist_counts = Arc::new(DashMap::<usize, usize, FxBuildHasher>::with_hasher(
        FxBuildHasher,
    ));
    let spheres = Arc::new(DashMap::<usize, usize, FxBuildHasher>::with_hasher(
        FxBuildHasher,
    ));

    // Seed: identity at depth 0 (recorded, not expanded).
    let mut seed_canonical_graph = sparsegraph::default();
    let id_perm = Permutation::id_perm(nn);
    let (id_canon, _) = canonicalize_perm_sparse_graph(&id_perm, &mut seed_canonical_graph);
    let id_key = CompactPerm::from_permutation(&id_canon);
    let (_, inserted) = trie.insert(&id_key);
    assert!(inserted);

    // Seed: one length-1 gate (all single gates are one wire-orbit).
    let base_ckt = CircuitSeq {
        gates: vec![[0, 1, 2]],
    };
    let (base_canon, base_sphere) =
        canonicalize_perm_sparse_graph(&base_ckt.perm(n), &mut seed_canonical_graph);
    SG_FREE(&mut seed_canonical_graph);
    let base_key = CompactPerm::from_permutation(&base_canon);
    let (_, inserted) = trie.insert(&base_key);
    assert!(inserted);
    *dist_counts.entry(1).or_default() += 1;
    *spheres.entry(1).or_default() += base_sphere;

    let q = Arc::new(SegQueue::<WorkItem>::new());
    q.push(WorkItem {
        depth: 1,
        perm: base_key,
    });

    let circuits_stored = Arc::new(AtomicUsize::new(2)); // id + base
    // Outstanding work: items in `q` plus items a worker has popped and is
    // still expanding. Seeded with the base node.
    let pending = Arc::new(AtomicUsize::new(1));
    let num_threads = num_cpus::get();

    std::thread::scope(|s| {
        for tid in 0..num_threads {
            let q = q.clone();
            let trie = trie.clone();
            let dist_counts = dist_counts.clone();
            let spheres = spheres.clone();
            let gens = gens.clone();
            let circuits_stored = circuits_stored.clone();
            let pending = pending.clone();

            let start = Instant::now();
            let mut last_stored = 0usize;

            s.spawn(move || {
                let mut last_print = Instant::now();
                // Reused across all nauty calls on this worker. The C library
                // grows it as needed; retaining one avoids an allocation per
                // canonicalization.
                let mut canonical_graph_scratch = sparsegraph::default();
                let mut batch: Vec<(CompactPerm, usize, usize, u16)> =
                    Vec::with_capacity(gen_size);

                loop {
                    let parent = match q.pop() {
                        Some(item) => item,
                        None => {
                            if pending.load(Ordering::SeqCst) == 0 {
                                break;
                            }
                            sleep(Duration::from_micros(100));
                            continue;
                        }
                    };

                    let m = parent.depth as usize + 1;

                    if m <= max_m {
                        let parent_perm = parent.perm.to_permutation();
                        batch.clear();

                        for (gi, gperm) in gens.iter().enumerate() {
                            let h = gperm.compose(&parent_perm);
                            let (canon, sphere) =
                                canonicalize_perm_sparse_graph(&h, &mut canonical_graph_scratch);
                            batch.push((
                                CompactPerm::from_permutation(&canon),
                                m,
                                sphere,
                                gi as u16,
                            ));
                        }

                        let mut counts_update = HashMap::<usize, usize>::new();
                        let mut new_count = 0usize;

                        for (key, depth, sphere, _gate_idx) in batch.drain(..) {
                            let (_, inserted) = trie.insert(&key);
                            if !inserted {
                                continue;
                            }

                            *counts_update.entry(depth).or_default() += 1;
                            *spheres.entry(depth).or_default() += sphere;
                            new_count += 1;

                            // Count the frontier layer but do not expand it.
                            if depth != max_m {
                                pending.fetch_add(1, Ordering::SeqCst);
                                q.push(WorkItem {
                                    depth: depth as u16,
                                    perm: key,
                                });
                            }
                        }

                        let ql = q.len();
                        let ct = circuits_stored.fetch_add(new_count, Ordering::Relaxed);

                        for (depth, count) in counts_update {
                            *dist_counts.entry(depth).or_default() += count;
                        }

                        let kper_sec = (ct as f64) / start.elapsed().as_secs_f64() / 1000.0;
                        let eta = (ql as f64 / 1000.0) / kper_sec.max(1e-9);

                        if last_print.elapsed().as_secs_f32() > 2.0 {
                            if last_stored != ct {
                                println!(
                                    "t{tid:3} st:{ct:6} Q:{ql:6}    {kper_sec:.1}k/s  m={m}"
                                );
                            } else {
                                println!(
                                    "t{tid:3} st:{ct:6} Q:{ql:6}    {kper_sec:.1}k/s   eta {eta:.0} sec"
                                );
                            }
                            last_print = Instant::now();
                        }
                        last_stored = ct;
                    }

                    pending.fetch_sub(1, Ordering::SeqCst);
                }

                SG_FREE(&mut canonical_graph_scratch);
            });
        }
    });

    let can_ckt = trie.permutation_count();
    // This exceeds usize quickly (e.g. 60^11 for five wires), even when the
    // discovered ball still fits in memory.
    let total_ckt = BigUint::from(gen_size).pow(max_m as u32);
    let compr_ratio = &total_ckt / BigUint::from(can_ckt.max(1));

    println!("n={n} wires");
    println!(
        "Final: {} canonical perms, {} permutation-trie nodes, {} total circuits, ({}x)",
        can_ckt,
        trie.node_count(),
        total_ckt,
        compr_ratio
    );
    let trie_edges = trie.node_count() - 1;
    let unshared_values = (can_ckt as u128) * (nn as u128);
    println!(
        "Trie prefixes: {} edges vs {} unshared values ({:.1}% retained)",
        trie_edges,
        unshared_values,
        100.0 * trie_edges as f64 / unshared_values as f64
    );

    println!("m      count     sphere");
    for m in 1..=max_m {
        let count = dist_counts.get(&m).map(|c| *c).unwrap_or(0);
        let sp = spheres.get(&m).map(|c| *c).unwrap_or(0);
        println!("{m} {count:10} {sp:10}");
    }
}

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    #[arg(short = 'n', default_value_t = 4)]
    wires: usize,

    #[arg(short = 'm', default_value = None)]
    gates: Option<usize>,
}

fn main() {
    let args = Args::parse();
    let n = args.wires;
    let m = args.gates.unwrap_or(n * (n.ilog2() + 1) as usize);
    iso_bfs(n, m);
}

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::Itertools;

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
}
