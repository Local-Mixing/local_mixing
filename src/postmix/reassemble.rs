// Exact, barrier-free recovery of g57 gates from their plain XGate fragments.
//
// A structural g57 has firing function
//
//     1 XOR (!p AND q) = p OR !q
//
// and is represented by `comp=1` with two controls of opposite polarity.
// Splitting it into disjoint plain cubes gives
//
//     {p=s} XOR {p=!s,q=!s}.
//
// Depending on which parent literal was first in the randomized presplit, this
// is either a positive 1cc plus an all-negative 2cc, or a negative 1cc plus an
// all-positive 2cc.  A homogeneous 2cc can use either control as `p`, so a
// greedy pairing is not exact.  The barrier-free problem is a capacitated
// bipartite matching: every 2cc is a unit vertex with at most two edges, while
// all 1ccs with the same (target, control, polarity) form one capacity vertex.
// Dinic on this compact network is exact and keeps both memory and graph
// construction linear in the input size (there is no expansion across
// duplicate 1ccs).
use super::xgate::XGate;
use std::collections::HashMap;

/// True when `g` is in the native structural representation of a non-degenerate
/// g57: complemented, width two, with one positive and one negative literal.
///
/// `XGate::from_g57([a, x, x])` is an always-firing X gate and deliberately is
/// not counted as a structural width-two g57.
pub fn is_structural_g57(g: &XGate) -> bool {
    g.comp && g.ctrls.len() == 2 && g.ctrls[0].1 != g.ctrls[1].1
}

fn singleton_1cc_key(g: &XGate) -> Option<u64> {
    (!g.comp && g.ctrls.len() == 1).then(|| pack_key(g.target, g.ctrls[0].0, g.ctrls[0].1))
}

fn homogeneous_2cc(g: &XGate) -> Option<(u16, u16, bool)> {
    (!g.comp && g.ctrls.len() == 2 && g.ctrls[0].1 == g.ctrls[1].1)
        .then(|| (g.ctrls[0].0, g.ctrls[1].0, g.ctrls[0].1))
}

#[inline]
fn pack_key(target: u16, control: u16, polarity: bool) -> u64 {
    ((target as u64) << 17) | ((control as u64) << 1) | polarity as u64
}

/// Recognize the exact two-plain-gate decomposition of a g57 and return the
/// fused structural gate.  Argument order does not matter, and the positive
/// 1cc may select either wire of a homogeneous, opposite-polarity 2cc.
pub fn plain_pair_to_g57(a: &XGate, b: &XGate) -> Option<XGate> {
    if a.target != b.target {
        return None;
    }
    let (one, two) = match (singleton_1cc_key(a), homogeneous_2cc(b)) {
        (Some(_), Some(_)) => (a, b),
        _ => match (singleton_1cc_key(b), homogeneous_2cc(a)) {
            (Some(_), Some(_)) => (b, a),
            _ => return None,
        },
    };
    let p = one.ctrls[0].0;
    let singleton_polarity = one.ctrls[0].1;
    if two.ctrls[0].1 == singleton_polarity {
        return None;
    }
    let (w0, w1) = (two.ctrls[0].0, two.ctrls[1].0);
    let q = if p == w0 {
        w1
    } else if p == w1 {
        w0
    } else {
        return None;
    };
    Some(if singleton_polarity {
        XGate::from_g57([one.target, p, q])
    } else {
        XGate::from_g57([one.target, q, p])
    })
}

/// One pair in an exact maximum matching.  The indices refer to the input
/// slice passed to [`analyze_barrier_free`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReassemblyPair {
    pub one_cc: usize,
    pub two_cc: usize,
}

/// Counts useful both for whole-artifact analysis and for fcompress reporting.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReassemblyStats {
    pub total_gates: usize,
    pub structural_g57: usize,
    /// Every uncomplemented XGate, including shapes not eligible for an exact
    /// two-fragment g57 recovery.  This is the user-facing loose-fragment
    /// denominator.
    pub plain_fragments: usize,
    pub singleton_1cc: usize,
    pub homogeneous_2cc: usize,
    /// Polarity breakdowns are useful for diagnosing randomized presplit
    /// orientation; each pair must cross from one polarity to the other.
    pub positive_1cc: usize,
    pub negative_1cc: usize,
    pub positive_2cc: usize,
    pub negative_2cc: usize,
    /// Singleton instances whose full (target, wire, polarity) key occurs on
    /// at least one opposite-polarity homogeneous 2cc.
    pub compatible_singleton_1cc: usize,
    /// Homogeneous 2cc instances with at least one available singleton key.
    pub compatible_homogeneous_2cc: usize,
    /// Cardinality of the exact maximum matching.
    pub reassemblable_pairs: usize,
    /// Exactly twice `reassemblable_pairs`.
    pub reassemblable_fragments: usize,
}

impl ReassemblyStats {
    /// Existing structural gates plus the g57s recoverable from plain pairs.
    pub fn potential_g57(&self) -> usize {
        self.structural_g57 + self.reassemblable_pairs
    }

    /// Percentage of all input gates that can participate in a recovered pair.
    pub fn percent_of_all_gates_reassemblable(&self) -> f64 {
        percent(self.reassemblable_fragments, self.total_gates)
    }

    /// Percentage of all loose/plain fragments that the maximum matching can
    /// consume.  Existing structural g57s are not part of the denominator.
    pub fn percent_reassemblable_of_plain(&self) -> f64 {
        percent(self.reassemblable_fragments, self.plain_fragments)
    }

    /// Percentage among the two syntactically eligible plain-fragment shapes.
    pub fn percent_of_candidate_fragments_reassemblable(&self) -> f64 {
        percent(
            self.reassemblable_fragments,
            self.singleton_1cc + self.homogeneous_2cc,
        )
    }
}

fn percent(numer: usize, denom: usize) -> f64 {
    if denom == 0 {
        0.0
    } else {
        100.0 * numer as f64 / denom as f64
    }
}

/// Exact maximum matching and its summary.  "Barrier-free" means gate order
/// and intervening read/write conflicts are intentionally ignored.  This is a
/// valid whole-circuit *metric*.  Applying the pairs as rewrites is valid only
/// in a region already known to be freely gatherable, such as one fcompress
/// same-target group.
#[derive(Clone, Debug, Default)]
pub struct ReassemblyAnalysis {
    pub stats: ReassemblyStats,
    pub pairs: Vec<ReassemblyPair>,
}

#[derive(Clone, Copy, Debug)]
struct KeyInfo {
    capacity: u32,
    active: u32,
}

#[derive(Clone, Copy, Debug)]
struct Job {
    gate_index: usize,
    keys: [u32; 2],
    flow_edges: [u32; 2],
}

const NONE: u32 = u32::MAX;

/// Compute an exact maximum set of disjoint plain-fragment pairs.  Graph size
/// is O(gates), including when a key has millions of duplicate 1cc instances.
pub fn analyze_barrier_free(gates: &[XGate]) -> ReassemblyAnalysis {
    let mut stats = ReassemblyStats {
        total_gates: gates.len(),
        ..ReassemblyStats::default()
    };

    // Collapse duplicate 1ccs into capacities.  We recover concrete 1cc
    // indices in one final streaming pass after the flow is known.
    let mut key_of: HashMap<u64, u32> = HashMap::new();
    let mut keys: Vec<KeyInfo> = Vec::new();
    for g in gates {
        if is_structural_g57(g) {
            stats.structural_g57 += 1;
        }
        if !g.comp {
            stats.plain_fragments += 1;
        }
        if let Some(key) = singleton_1cc_key(g) {
            stats.singleton_1cc += 1;
            if g.ctrls[0].1 {
                stats.positive_1cc += 1;
            } else {
                stats.negative_1cc += 1;
            }
            let id = match key_of.get(&key) {
                Some(&id) => id,
                None => {
                    let id = u32::try_from(keys.len()).expect("too many reassembly keys");
                    key_of.insert(key, id);
                    keys.push(KeyInfo {
                        capacity: 0,
                        active: NONE,
                    });
                    id
                }
            };
            keys[id as usize].capacity = keys[id as usize]
                .capacity
                .checked_add(1)
                .expect("too many 1cc fragments for one key");
        }
    }

    // A 2cc job has one or two compact key endpoints.  Missing endpoints are
    // omitted rather than expanded into dummy vertices.
    let mut jobs: Vec<Job> = Vec::new();
    for (gate_index, g) in gates.iter().enumerate() {
        let Some((w0, w1, polarity)) = homogeneous_2cc(g) else {
            continue;
        };
        stats.homogeneous_2cc += 1;
        if polarity {
            stats.positive_2cc += 1;
        } else {
            stats.negative_2cc += 1;
        }
        let singleton_polarity = !polarity;
        let k0 = key_of
            .get(&pack_key(g.target, w0, singleton_polarity))
            .copied()
            .unwrap_or(NONE);
        let k1 = key_of
            .get(&pack_key(g.target, w1, singleton_polarity))
            .copied()
            .unwrap_or(NONE);
        if k0 == NONE && k1 == NONE {
            continue;
        }
        jobs.push(Job {
            gate_index,
            keys: [k0, k1],
            flow_edges: [NONE, NONE],
        });
    }
    stats.compatible_homogeneous_2cc = jobs.len();
    if jobs.is_empty() {
        return ReassemblyAnalysis {
            stats,
            pairs: Vec::new(),
        };
    }

    // Compact away 1cc keys with no incident 2cc before allocating the flow
    // graph.  This matters for million-gate inputs dominated by unrelated 1ccs.
    for job in &jobs {
        for &key in &job.keys {
            if key != NONE {
                keys[key as usize].active = 0;
            }
        }
    }
    let mut active_len = 0u32;
    for key in &mut keys {
        if key.active != NONE {
            key.active = active_len;
            active_len = active_len.checked_add(1).expect("too many active keys");
            stats.compatible_singleton_1cc += key.capacity as usize;
        }
    }

    // source -> each 2cc job (1) -> either key (1) -> sink (1cc count).
    let source = 0usize;
    let job_base = 1usize;
    let key_base = job_base + jobs.len();
    let sink = key_base + active_len as usize;
    assert!(
        sink < u32::MAX as usize,
        "reassembly graph exceeds u32 node indices"
    );
    let mut graph: Vec<Vec<FlowEdge>> = (0..=sink).map(|_| Vec::new()).collect();
    for (j, job) in jobs.iter_mut().enumerate() {
        let node = job_base + j;
        add_edge(&mut graph, source, node, 1);
        for side in 0..2 {
            let key = job.keys[side];
            if key != NONE {
                let active = keys[key as usize].active as usize;
                job.flow_edges[side] = add_edge(&mut graph, node, key_base + active, 1);
            }
        }
    }
    for key in &keys {
        if key.active != NONE {
            add_edge(
                &mut graph,
                key_base + key.active as usize,
                sink,
                key.capacity,
            );
        }
    }
    let matched = max_flow(&mut graph, source, sink) as usize;

    // Read which key received each matched 2cc.  Store assignments grouped by
    // key with counting-sort offsets, then map them to concrete 1cc occurrences
    // in input order without a Vec allocation per key.
    let mut assignment: Vec<(u32, usize)> = Vec::with_capacity(matched);
    for (j, job) in jobs.iter().enumerate() {
        let node = job_base + j;
        for side in 0..2 {
            let edge = job.flow_edges[side];
            if edge != NONE && graph[node][edge as usize].cap == 0 {
                assignment.push((job.keys[side], j));
                break;
            }
        }
    }
    debug_assert_eq!(assignment.len(), matched);

    let mut offsets = vec![0usize; keys.len() + 1];
    for &(key, _) in &assignment {
        offsets[key as usize + 1] += 1;
    }
    for i in 0..keys.len() {
        offsets[i + 1] += offsets[i];
    }
    let mut write = offsets[..keys.len()].to_vec();
    let mut jobs_by_key = vec![0usize; matched];
    for (key, job) in assignment {
        jobs_by_key[write[key as usize]] = job;
        write[key as usize] += 1;
    }
    let mut read = offsets[..keys.len()].to_vec();
    let mut pairs = Vec::with_capacity(matched);
    for (one_cc, g) in gates.iter().enumerate() {
        let Some(packed) = singleton_1cc_key(g) else {
            continue;
        };
        let key = key_of[&packed] as usize;
        if read[key] < offsets[key + 1] {
            let job = jobs_by_key[read[key]];
            read[key] += 1;
            pairs.push(ReassemblyPair {
                one_cc,
                two_cc: jobs[job].gate_index,
            });
        }
    }
    debug_assert_eq!(pairs.len(), matched);
    stats.reassemblable_pairs = matched;
    stats.reassemblable_fragments = matched * 2;
    ReassemblyAnalysis { stats, pairs }
}

/// Apply an already-computed set of disjoint pairs in a barrier-free region.
/// Each fused g57 is emitted at the earlier of its two fragment positions;
/// all unpaired gates retain their relative order.
pub fn fuse_pairs(gates: &[XGate], pairs: &[ReassemblyPair]) -> Vec<XGate> {
    if pairs.is_empty() {
        return gates.to_vec();
    }
    let mut consumed = vec![false; gates.len()];
    let mut fused_at: Vec<Option<XGate>> = vec![None; gates.len()];
    for pair in pairs {
        assert!(pair.one_cc < gates.len() && pair.two_cc < gates.len());
        assert!(!consumed[pair.one_cc] && !consumed[pair.two_cc]);
        let fused = plain_pair_to_g57(&gates[pair.one_cc], &gates[pair.two_cc])
            .expect("reassembly matching produced an invalid pair");
        let at = pair.one_cc.min(pair.two_cc);
        consumed[pair.one_cc] = true;
        consumed[pair.two_cc] = true;
        fused_at[at] = Some(fused);
    }
    let mut out = Vec::with_capacity(gates.len() - pairs.len());
    for i in 0..gates.len() {
        if let Some(g) = fused_at[i].take() {
            out.push(g);
        } else if !consumed[i] {
            out.push(gates[i].clone());
        }
    }
    out
}

/// Analyze and fuse an exact maximum set of pairs in a barrier-free region.
pub fn fuse_barrier_free(gates: &[XGate]) -> (Vec<XGate>, ReassemblyStats) {
    let analysis = analyze_barrier_free(gates);
    let out = fuse_pairs(gates, &analysis.pairs);
    (out, analysis.stats)
}

#[derive(Clone, Copy, Debug)]
struct FlowEdge {
    to: u32,
    rev: u32,
    cap: u32,
}

fn add_edge(graph: &mut [Vec<FlowEdge>], from: usize, to: usize, cap: u32) -> u32 {
    let fwd = u32::try_from(graph[from].len()).expect("too many flow edges on one node");
    let rev = u32::try_from(graph[to].len()).expect("too many flow edges on one node");
    graph[from].push(FlowEdge {
        to: to as u32,
        rev,
        cap,
    });
    graph[to].push(FlowEdge {
        to: from as u32,
        rev: fwd,
        cap: 0,
    });
    fwd
}

// Dinic's blocking-flow algorithm.  Every source->job edge has unit capacity,
// so one augmentation always carries one match.  The DFS is iterative: an
// adversarial alternating chain can be very long, and million-gate analysis
// must not depend on the process stack size.
fn max_flow(graph: &mut [Vec<FlowEdge>], source: usize, sink: usize) -> u32 {
    let mut flow = 0u32;
    let mut level = vec![-1i32; graph.len()];
    let mut queue = Vec::<usize>::with_capacity(graph.len());
    let mut next = vec![0usize; graph.len()];
    let mut nodes = Vec::<usize>::new();
    let mut path = Vec::<(usize, usize)>::new();
    while build_levels(graph, source, sink, &mut level, &mut queue) {
        next.fill(0);
        while send_one(
            graph, source, sink, &level, &mut next, &mut nodes, &mut path,
        ) {
            flow = flow.checked_add(1).expect("too many reassembled pairs");
        }
    }
    flow
}

fn build_levels(
    graph: &[Vec<FlowEdge>],
    source: usize,
    sink: usize,
    level: &mut [i32],
    queue: &mut Vec<usize>,
) -> bool {
    level.fill(-1);
    queue.clear();
    level[source] = 0;
    queue.push(source);
    let mut head = 0usize;
    while head < queue.len() {
        let v = queue[head];
        head += 1;
        for edge in &graph[v] {
            let to = edge.to as usize;
            if edge.cap != 0 && level[to] < 0 {
                level[to] = level[v] + 1;
                queue.push(to);
            }
        }
    }
    level[sink] >= 0
}

fn send_one(
    graph: &mut [Vec<FlowEdge>],
    source: usize,
    sink: usize,
    level: &[i32],
    next: &mut [usize],
    nodes: &mut Vec<usize>,
    path: &mut Vec<(usize, usize)>,
) -> bool {
    nodes.clear();
    path.clear();
    nodes.push(source);
    loop {
        let v = *nodes.last().expect("flow path always contains source");
        if v == sink {
            // The first edge is source->job with capacity one, so the path
            // bottleneck is exactly one even when a key has larger capacity.
            for &(from, edge_index) in path.iter() {
                let edge = graph[from][edge_index];
                debug_assert!(edge.cap > 0);
                graph[from][edge_index].cap -= 1;
                graph[edge.to as usize][edge.rev as usize].cap += 1;
            }
            return true;
        }

        while next[v] < graph[v].len() {
            let edge = graph[v][next[v]];
            if edge.cap != 0 && level[edge.to as usize] == level[v] + 1 {
                break;
            }
            next[v] += 1;
        }
        if next[v] == graph[v].len() {
            nodes.pop();
            let Some((parent, edge_index)) = path.pop() else {
                return false;
            };
            next[parent] = edge_index + 1;
        } else {
            let edge_index = next[v];
            let to = graph[v][edge_index].to as usize;
            path.push((v, edge_index));
            nodes.push(to);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postmix::rules;
    use crate::postmix::xgate::eval_lanes;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use smallvec::smallvec;

    fn conj(t: u16, lits: &[(u16, bool)]) -> XGate {
        XGate::conj(t, lits.iter().copied()).unwrap()
    }

    #[test]
    fn structural_and_plain_recognition_are_exact() {
        let native = XGate::from_g57([0, 1, 2]);
        assert!(is_structural_g57(&native));
        assert!(!is_structural_g57(&XGate::x_gate(0)));
        assert!(!is_structural_g57(&XGate {
            target: 0,
            comp: true,
            ctrls: smallvec![(1, false), (2, false)],
        }));

        let one_p = conj(0, &[(1, true)]);
        let one_q = conj(0, &[(2, true)]);
        let two = conj(0, &[(1, false), (2, false)]);
        assert_eq!(
            plain_pair_to_g57(&one_p, &two),
            Some(XGate::from_g57([0, 1, 2]))
        );
        assert_eq!(
            plain_pair_to_g57(&two, &one_q),
            Some(XGate::from_g57([0, 2, 1]))
        );
        assert!(plain_pair_to_g57(&conj(0, &[(3, true)]), &two).is_none());
        assert!(plain_pair_to_g57(&conj(0, &[(1, false)]), &two).is_none());
        assert!(plain_pair_to_g57(&one_p, &conj(4, &[(1, false), (2, false)])).is_none());

        let one_negative = conj(0, &[(1, false)]);
        let two_positive = conj(0, &[(1, true), (2, true)]);
        assert_eq!(
            plain_pair_to_g57(&one_negative, &two_positive),
            Some(XGate::from_g57([0, 2, 1]))
        );
    }

    #[test]
    fn randomized_presplit_alternate_orientation_reassembles() {
        let parent = XGate::from_g57([0, 1, 2]);
        let mut alternate = None;
        // rules::presplit shuffles the two parent literals.  Find its
        // deterministic seeded ordering that puts the positive parent literal
        // first, yielding a negative 1cc plus an all-positive 2cc.
        for seed in 0..64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let pieces = rules::presplit(&parent, &mut rng);
            if pieces[0].ctrls[0].1 == false {
                alternate = Some(pieces);
                break;
            }
        }
        let pieces = alternate.expect("seed range should exercise both presplit shuffles");
        assert_eq!(pieces[0].width(), 1);
        assert!(!pieces[0].ctrls[0].1);
        assert_eq!(pieces[1].width(), 2);
        assert!(pieces[1].ctrls.iter().all(|&(_, polarity)| polarity));
        assert_eq!(plain_pair_to_g57(&pieces[0], &pieces[1]), Some(parent));
        let a = analyze_barrier_free(&pieces);
        assert_eq!(a.stats.negative_1cc, 1);
        assert_eq!(a.stats.positive_2cc, 1);
        assert_eq!(a.stats.reassemblable_pairs, 1);
    }

    #[test]
    fn maximum_matching_repairs_a_greedy_trap() {
        // e(1,2) can use either key; e(1,3) can only use key 1 because no
        // positive-3 fragment exists.  Choosing key 1 for the first edge is a
        // greedy dead end, while the exact maximum has cardinality two.
        let gates = vec![
            conj(0, &[(1, true)]),
            conj(0, &[(2, true)]),
            conj(0, &[(1, false), (2, false)]),
            conj(0, &[(1, false), (3, false)]),
        ];
        let analysis = analyze_barrier_free(&gates);
        assert_eq!(analysis.stats.reassemblable_pairs, 2);
        assert_eq!(analysis.stats.reassemblable_fragments, 4);
        assert_eq!(analysis.pairs.len(), 2);
        let (out, stats) = fuse_barrier_free(&gates);
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(is_structural_g57));
        assert_eq!(stats.percent_of_all_gates_reassemblable(), 100.0);
    }

    #[test]
    fn capacities_and_targets_are_respected() {
        let gates = vec![
            conj(0, &[(1, true)]),
            conj(0, &[(1, true)]),
            conj(0, &[(1, false), (2, false)]),
            conj(0, &[(1, false), (3, false)]),
            conj(0, &[(1, false), (4, false)]),
            // Same controls on another target cannot consume target 0's 1ccs.
            conj(5, &[(1, false), (2, false)]),
            XGate::from_g57([6, 7, 8]),
        ];
        let a = analyze_barrier_free(&gates);
        assert_eq!(a.stats.structural_g57, 1);
        assert_eq!(a.stats.plain_fragments, 6);
        assert_eq!(a.stats.singleton_1cc, 2);
        assert_eq!(a.stats.homogeneous_2cc, 4);
        assert_eq!(a.stats.positive_1cc, 2);
        assert_eq!(a.stats.negative_1cc, 0);
        assert_eq!(a.stats.positive_2cc, 0);
        assert_eq!(a.stats.negative_2cc, 4);
        assert_eq!(a.stats.compatible_homogeneous_2cc, 3);
        assert_eq!(a.stats.compatible_singleton_1cc, 2);
        assert_eq!(a.stats.reassemblable_pairs, 2);
        assert_eq!(a.stats.potential_g57(), 3);
        assert_eq!(a.stats.percent_reassemblable_of_plain(), 100.0 * 4.0 / 6.0);
    }

    #[test]
    fn compact_flow_matches_bruteforce_on_random_small_instances() {
        fn brute(gates: &[XGate], twos: &[usize], at: usize, used: &mut [bool]) -> usize {
            if at == twos.len() {
                return 0;
            }
            let mut best = brute(gates, twos, at + 1, used);
            for one in 0..gates.len() {
                if !used[one] && plain_pair_to_g57(&gates[one], &gates[twos[at]]).is_some() {
                    used[one] = true;
                    best = best.max(1 + brute(gates, twos, at + 1, used));
                    used[one] = false;
                }
            }
            best
        }

        let mut rng = StdRng::seed_from_u64(0x57_57);
        for _ in 0..300 {
            let mut gates = Vec::new();
            for _ in 0..rng.random_range(0..=6) {
                let w = rng.random_range(1..=5);
                gates.push(conj(0, &[(w, rng.random_bool(0.5))]));
            }
            for _ in 0..rng.random_range(0..=6) {
                let a = rng.random_range(1..=5);
                let mut b = rng.random_range(1..=5);
                while b == a {
                    b = rng.random_range(1..=5);
                }
                let polarity = rng.random_bool(0.5);
                gates.push(conj(0, &[(a, polarity), (b, polarity)]));
            }
            let twos: Vec<usize> = gates
                .iter()
                .enumerate()
                .filter_map(|(i, g)| homogeneous_2cc(g).is_some().then_some(i))
                .collect();
            let expected = brute(&gates, &twos, 0, &mut vec![false; gates.len()]);
            let got = analyze_barrier_free(&gates);
            assert_eq!(got.stats.reassemblable_pairs, expected, "gates={gates:?}");
            let mut seen = vec![false; gates.len()];
            for pair in got.pairs {
                assert!(!seen[pair.one_cc] && !seen[pair.two_cc]);
                seen[pair.one_cc] = true;
                seen[pair.two_cc] = true;
                assert!(plain_pair_to_g57(&gates[pair.one_cc], &gates[pair.two_cc]).is_some());
            }
        }
    }

    #[test]
    fn fusing_a_pair_preserves_all_lanes() {
        let before = vec![conj(0, &[(1, false), (2, false)]), conj(0, &[(1, true)])];
        let (after, stats) = fuse_barrier_free(&before);
        assert_eq!(stats.reassemblable_pairs, 1);
        assert_eq!(after, vec![XGate::from_g57([0, 1, 2])]);
        for state in 0u64..8 {
            let mut a: Vec<u64> = (0..3).map(|w| ((state >> w) & 1) * !0).collect();
            let mut b = a.clone();
            eval_lanes(before.iter(), &mut a);
            eval_lanes(after.iter(), &mut b);
            assert_eq!(a, b);
        }
    }

    #[test]
    fn large_disjoint_instance_has_linear_graph_size() {
        // Large enough to catch an accidental all-pairs compatibility scan,
        // while remaining cheap in the normal unit-test suite.
        let n = 20_000u16;
        let mut gates = Vec::with_capacity(n as usize * 2);
        for target in 0..n {
            let p = target.wrapping_add(1);
            let q = target.wrapping_add(2);
            // Controls must differ from target; wrapping only affects the last
            // targets and still preserves that property here.
            gates.push(conj(target, &[(p, true)]));
            gates.push(conj(target, &[(p, false), (q, false)]));
        }
        let a = analyze_barrier_free(&gates);
        assert_eq!(a.stats.reassemblable_pairs, n as usize);
    }

    #[test]
    #[ignore = "explicit million-gate scalability regression"]
    fn million_gate_capacity_instance() {
        let pairs = 500_000usize;
        let mut gates = Vec::with_capacity(pairs * 2);
        for _ in 0..pairs {
            gates.push(conj(0, &[(1, true)]));
            gates.push(conj(0, &[(1, false), (2, false)]));
        }
        let a = analyze_barrier_free(&gates);
        assert_eq!(a.stats.total_gates, 1_000_000);
        assert_eq!(a.stats.reassemblable_pairs, pairs);
        assert_eq!(a.stats.reassemblable_fragments, 1_000_000);
    }
}
