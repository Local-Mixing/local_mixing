// Final compression pass for post-mix circuits (fcompress).
//
// The XGate calculus is closed under XOR on a shared target: a run of
// same-target gates composes to t ^= f1 XOR f2 XOR ... XOR fk, an ESOP
// (XOR of mixed-polarity cubes) whose canonical form is ANF. This pass
// gathers same-target gates that can float to a common point, reduces the
// gathered cube set (pairwise catalogue to fixed point, then an ANF
// rewrite when it wins: exact minimum ESOP on supports <= 4 wires, greedy
// multi-negation subcube covering plus maximum distance-1 matching beyond),
// re-emits the survivors as consecutive XGates, and interleaves one
// conjugation-descent pass (postprocessing::downhill) per iteration, which
// collapses R1 case-split ladders that gathering alone cannot touch.
// Everything stays in mpmct1 and downstream tooling keeps working.
// Deterministic and attacker-computable, so it never weakens hiding; the
// compressed size doubles as the honest "effective size" of a mixed
// circuit. With zero_in set the pass additionally specializes to the
// promised known-zero entry wires, and equality (and the size metric) is
// then relative to that input subspace.
//
// Gathering is one forward sweep with an open group per target wire:
//   - a read of wire t closes t's group (a reader pins the accumulated
//     value; writes may not cross it) -- unless the reader is separated from
//     every member by an opposite-polarity shared literal, in which case it
//     commutes with the group and the group floats past it,
//   - a write to a wire in a group's union-of-member-controls either
//     TRANSPORTS the group across the writer h (the downhill substitution
//     u <- u XOR fire(h) on h's target, accepted when the catalogue-reduced
//     ESOP does not grow -- this is Toffoli sliding at any distance, and it
//     is what lets a conjugated copy g' = g[u -> !u] meet g and cancel) or,
//     when h reads the target or the substitution would grow the ESOP,
//     closes the group,
//   - members then float right to the close point and are emitted there.
// A transported group's cubes live in the frame AFTER the writer, so the
// group records a dependency on the writer's group and is emitted after it:
// closing a group closes its dependencies first, and a close set is emitted
// in dependency order (ascending last-member order among independent
// groups). Groups with no dependency path between them commute, so any
// order among those is legal. The dependency graph is kept acyclic by
// construction (a transport that would close a cycle closes the group
// instead).
//
// The gather runs forward and then on the reversed gate list (every XGate is
// an involution, so the reversed list is the inverse function and gathering
// it is exact): the crossing stage floats gates both ways, and a case-split
// ladder left behind by a leftward crossing can only be folded back by a
// leftward float.
//
// Optional output-cone pruning for gadgetized circuits (equality required
// only on designated live wires): one exact backward pass in the
// XOR-accumulate model — a gate is deletable iff its target is dead at
// its position; a kept gate makes its controls live, and its target STAYS
// live (XOR never overwrites).
use crate::circuit::xgate::{Lits, XGate};
use crate::engine::format::PackedGate;
use crate::engine::mix::{Merge, merge_result};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::OnceLock;

pub struct CompressParams {
    // None = equality required on every wire. Some(mask) = only on wires
    // with mask[w] == true; dead cones are pruned.
    pub live_out: Option<Vec<bool>>,
    // None = equality required on every input. Some(mask) = wires with
    // mask[w] == true are promised zero at circuit entry; gates are
    // specialized to that input subspace (literals on known wires fold,
    // never-firing gates drop). The output then equals the input circuit
    // ONLY on the promised subspace.
    pub zero_in: Option<Vec<bool>>,
    pub max_iters: usize,
    // Groups are closed proactively at this many members.
    pub group_cap: usize,
    // ANF rewrite attempted only when the group support fits (mask bits).
    pub anf_support_cap: usize,
    // Interleave one conjugation-descent pass (postprocessing::downhill) after each
    // gather/reduce iteration.
    pub downhill: bool,
    // Float groups across writers of their control wires by conjugation
    // (Toffoli sliding) instead of closing them, when the ESOP does not grow.
    pub transport: bool,
    // Extra cubes a transport may add (0 = neutral-or-better only). Growth
    // is speculative: the pass-level guard restores the previous circuit if
    // an iteration ends larger.
    pub transport_slack: usize,
    // A reader of the target separated from every member by an opposite
    // literal commutes with the group and does not close it.
    pub sep_reads: bool,
    // Also gather on the reversed gate list each iteration (leftward float).
    pub reverse_pass: bool,
    pub local_verify: bool,
    pub seed: u64,
}

impl Default for CompressParams {
    fn default() -> CompressParams {
        CompressParams {
            live_out: None,
            zero_in: None,
            max_iters: 10,
            group_cap: 64,
            anf_support_cap: 40,
            downhill: true,
            transport: true,
            transport_slack: 0,
            sep_reads: true,
            reverse_pass: true,
            local_verify: true,
            seed: 0,
        }
    }
}

#[derive(Default, Debug)]
pub struct CompressReport {
    pub iters: usize,
    pub liveness_dropped: usize,
    pub zero_killed: usize,
    pub zero_lits_dropped: u64,
    pub groups: u64,
    pub multi_groups: u64,
    pub max_group: usize,
    pub catalogue_merges: u64,
    pub anf_wins: u64,
    pub exact_wins: u64,
    pub downhill_swaps: u64,
    // Groups floated across a writer of one of their control wires (ESOP
    // changed / unchanged), refused on cost, refused to keep the frame
    // dependencies acyclic; readers that passed a group by separation.
    pub transports: u64,
    pub transport_noops: u64,
    pub transport_refused: u64,
    pub transport_cycle_refused: u64,
    pub sep_passes: u64,
    pub verifies_skipped: u64,
    pub gates_in: usize,
    pub gates_out: usize,
    pub lits_in: u64,
    pub lits_out: u64,
}

pub fn lits_of(gates: &[XGate]) -> u64 {
    gates.iter().map(|g| g.width() as u64).sum()
}

// Exact dead-cone elimination: keep a gate iff its target is live at its
// position; kept gates make their controls live. Returns dropped count.
pub fn liveness_prune(gates: Vec<XGate>, live_out: &[bool]) -> (Vec<XGate>, usize) {
    let (out, _, dropped) = liveness_prune_anc(gates, None, live_out);
    (out, dropped)
}

// Per-gate ancestor set, in the fmix sidecar's word layout. Threaded through
// compression so the compressed circuit keeps a meaningful sidecar: gathering
// is a permutation (sets follow gates), a reduced multi-member group stamps
// every survivor with the UNION of its members' sets (each emitted cube
// derives from the whole gathered ESOP), and pruned gates just drop out.
pub type AncBits = Vec<u64>;

pub(crate) fn or_anc(dst: &mut AncBits, src: &AncBits) {
    if dst.len() < src.len() {
        dst.resize(src.len(), 0);
    }
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d |= *s;
    }
}

// Input-side specialization to the promised zero slice: with zero_in[w] wires
// known 0 at entry, run three-valued constant tracking (0 / 1 / unknown)
// forward once. A literal on a known wire either always holds (drop the
// literal) or never (the cube is dead: t ^= comp XOR 0, so the gate drops
// when comp=0 and degrades to a bare X when comp=1); a gate folded to an
// empty cube is an X (flips a known constant, stays) or a no-op comp gate
// (drops); any surviving write forgets its target's constant. The result
// equals the input circuit on every wire, but ONLY for entries inside the
// promised subspace.
fn zero_specialize_anc(
    gates: Vec<XGate>,
    anc: Option<Vec<AncBits>>,
    zero_in: &[bool],
    wires: usize,
) -> (Vec<XGate>, Option<Vec<AncBits>>, usize, u64) {
    let mut known: Vec<Option<bool>> = (0..wires)
        .map(|w| if zero_in.get(w) == Some(&true) { Some(false) } else { None })
        .collect();
    let n = gates.len();
    let had_anc = anc.is_some();
    let mut out: Vec<XGate> = Vec::with_capacity(n);
    let mut out_anc: Option<Vec<AncBits>> = anc.as_ref().map(|_| Vec::with_capacity(n));
    let mut killed = 0usize;
    let mut lits_dropped = 0u64;
    let anc_iter = anc.unwrap_or_default();
    let mut anc_it = anc_iter.into_iter();
    for g in gates {
        let tag = if had_anc { anc_it.next() } else { None };
        let mut dead = false;
        let mut lits: Lits = Lits::new();
        for &(w, p) in &g.ctrls {
            match known[w as usize] {
                Some(v) if v == p => {} // literal always true: fold it away
                Some(_) => {
                    dead = true; // literal always false: the cube never fires
                    break;
                }
                None => lits.push((w, p)),
            }
        }
        let t = g.target as usize;
        if dead {
            // t ^= comp XOR 0: gone for comp=0, a bare NOT for comp=1.
            if g.comp {
                lits_dropped += g.width() as u64;
                known[t] = known[t].map(|v| !v);
                out.push(XGate::x_gate(g.target));
                if let Some(oa) = out_anc.as_mut() {
                    oa.push(tag.expect("sidecar aligned with gates"));
                }
            } else {
                killed += 1;
            }
            continue;
        }
        if lits.is_empty() {
            // t ^= comp XOR 1: an X gate, or a never-firing comp gate.
            if g.comp {
                killed += 1;
                continue;
            }
            lits_dropped += g.width() as u64;
            known[t] = known[t].map(|v| !v);
            out.push(XGate { target: g.target, comp: false, ctrls: lits });
        } else {
            known[t] = None;
            lits_dropped += (g.width() - lits.len()) as u64;
            out.push(XGate { target: g.target, comp: g.comp, ctrls: lits });
        }
        if let Some(oa) = out_anc.as_mut() {
            oa.push(tag.expect("sidecar aligned with gates"));
        }
    }
    (out, out_anc, killed, lits_dropped)
}

fn liveness_prune_anc(
    gates: Vec<XGate>,
    anc: Option<Vec<AncBits>>,
    live_out: &[bool],
) -> (Vec<XGate>, Option<Vec<AncBits>>, usize) {
    let n = gates.len();
    let mut live = live_out.to_vec();
    let mut keep = vec![false; n];
    for i in (0..n).rev() {
        let g = &gates[i];
        if live[g.target as usize] {
            keep[i] = true;
            for &(w, _) in &g.ctrls {
                live[w as usize] = true;
            }
        }
    }
    let dropped = keep.iter().filter(|&&k| !k).count();
    let out: Vec<XGate> = gates
        .into_iter()
        .zip(keep.iter())
        .filter(|(_, k)| **k)
        .map(|(g, _)| g)
        .collect();
    let anc_out = anc.map(|a| {
        a.into_iter()
            .zip(keep.iter())
            .filter(|(_, k)| **k)
            .map(|(s, _)| s)
            .collect()
    });
    (out, anc_out, dropped)
}

// ---- exact minimum ESOP for supports of at most 4 wires -------------------
//
// One table per support size n in 1..=4. State = the function's truth table
// (2^n bits); edges = XOR one of the 3^n mixed-polarity cubes over exactly
// those n vars (the all-absent cube is the constant 1). BFS from 0 gives, for
// every function, a minimum-cube ESOP with a witness via parent pointers.
// Built lazily once per process; the largest (n=4) is 65536 states x 81 cubes.
struct ExactTab {
    from: Vec<u16>,       // parent state on a shortest path to 0
    via: Vec<u8>,         // cube index applied on that step
    cubes: Vec<(u8, u8)>, // (pos, neg) variable masks
}

static EXACT_TABS: OnceLock<[ExactTab; 4]> = OnceLock::new();

fn exact_tab_build(n: usize) -> ExactTab {
    let nbits = 1usize << n;
    let states = 1usize << nbits;
    let vmask = (nbits - 1) as u8; // n low variable bits
    let mut cubes: Vec<(u8, u8)> = Vec::new();
    for pos in 0u8..=vmask {
        for neg in 0u8..=vmask {
            if pos & neg == 0 {
                cubes.push((pos, neg));
            }
        }
    }
    let tts: Vec<u16> = cubes
        .iter()
        .map(|&(pos, neg)| {
            let mut tt: u16 = 0;
            for a in 0..nbits as u16 {
                if a as u8 & pos == pos && a as u8 & neg == 0 {
                    tt |= 1 << a;
                }
            }
            tt
        })
        .collect();
    let mut seen = vec![false; states];
    let mut from = vec![0u16; states];
    let mut via = vec![0u8; states];
    seen[0] = true;
    let mut q = VecDeque::from([0u16]);
    while let Some(s) = q.pop_front() {
        for (ci, &tt) in tts.iter().enumerate() {
            let t = s ^ tt;
            if !seen[t as usize] {
                seen[t as usize] = true;
                from[t as usize] = s;
                via[t as usize] = ci as u8;
                q.push_back(t);
            }
        }
    }
    debug_assert!(seen.iter().all(|&b| b), "minterm cubes reach every state");
    ExactTab { from, via, cubes }
}

// Minimum ESOP of the monomial set over n<=4 support vars: cubes plus a
// parity flip when the witness uses the constant cube.
fn exact_small(monos: &[u64], n: usize) -> (Vec<(u64, u64)>, bool) {
    debug_assert!((1..=4).contains(&n));
    let tabs = EXACT_TABS.get_or_init(|| {
        [
            exact_tab_build(1),
            exact_tab_build(2),
            exact_tab_build(3),
            exact_tab_build(4),
        ]
    });
    let tab = &tabs[n - 1];
    let nbits = 1u16 << n;
    let mut f: u16 = 0;
    for &m in monos {
        let m = m as u16;
        for a in 0..nbits {
            if a & m == m {
                f ^= 1 << a;
            }
        }
    }
    let mut cubes: Vec<(u64, u64)> = Vec::new();
    let mut delta = false;
    let mut s = f;
    while s != 0 {
        let (pos, neg) = tab.cubes[tab.via[s as usize] as usize];
        if pos == 0 && neg == 0 {
            delta ^= true;
        } else {
            cubes.push((pos as u64, neg as u64));
        }
        s = tab.from[s as usize];
    }
    (cubes, delta)
}

// ---- maximum distance-1 matching ------------------------------------------
//
// Monomials differing in exactly one variable pair into a single-negation
// cube: mono(m) XOR mono(m|b) = mono(m) AND NOT b. The pair graph is
// bipartite by popcount parity, so Hopcroft-Karp finds a maximum matching
// (the old greedy pairing is kept as the fallback for oversized sets).
// Returns (cubes from matched pairs, unmatched monomials).
const MATCH_CAP: usize = 1 << 15;

fn match_pairs(monos: &[u64], nbits: usize) -> (Vec<(u64, u64)>, Vec<u64>) {
    if monos.len() > MATCH_CAP {
        return greedy_pairs(monos);
    }
    let idx: HashMap<u64, usize> = monos.iter().enumerate().map(|(i, &m)| (m, i)).collect();
    let lefts: Vec<usize> = (0..monos.len())
        .filter(|&i| monos[i].count_ones() % 2 == 0)
        .collect();
    let adj: Vec<Vec<usize>> = lefts
        .iter()
        .map(|&i| {
            (0..nbits)
                .filter_map(|b| idx.get(&(monos[i] ^ (1u64 << b))).copied())
                .collect()
        })
        .collect();
    let mut pair_l: Vec<Option<usize>> = vec![None; lefts.len()]; // left pos -> mono idx
    let mut pair_r: Vec<Option<usize>> = vec![None; monos.len()]; // mono idx -> left pos
    loop {
        // BFS layers from free left vertices.
        let mut dist: Vec<Option<u32>> = vec![None; lefts.len()];
        let mut q: VecDeque<usize> = VecDeque::new();
        for (li, p) in pair_l.iter().enumerate() {
            if p.is_none() {
                dist[li] = Some(0);
                q.push_back(li);
            }
        }
        let mut found = false;
        while let Some(li) = q.pop_front() {
            for &v in &adj[li] {
                match pair_r[v] {
                    None => found = true,
                    Some(lj) => {
                        if dist[lj].is_none() {
                            dist[lj] = Some(dist[li].expect("queued has dist") + 1);
                            q.push_back(lj);
                        }
                    }
                }
            }
        }
        if !found {
            break;
        }
        fn dfs(
            li: usize,
            adj: &[Vec<usize>],
            dist: &mut [Option<u32>],
            pair_l: &mut [Option<usize>],
            pair_r: &mut [Option<usize>],
        ) -> bool {
            let d = dist[li];
            for vi in 0..adj[li].len() {
                let v = adj[li][vi];
                let ok = match pair_r[v] {
                    None => true,
                    Some(lj) => {
                        dist[lj] == d.map(|x| x + 1)
                            && dfs(lj, adj, dist, pair_l, pair_r)
                    }
                };
                if ok {
                    pair_l[li] = Some(v);
                    pair_r[v] = Some(li);
                    return true;
                }
            }
            dist[li] = None;
            false
        }
        for li in 0..lefts.len() {
            if pair_l[li].is_none() {
                dfs(li, &adj, &mut dist, &mut pair_l, &mut pair_r);
            }
        }
    }
    let mut used = vec![false; monos.len()];
    let mut cubes = Vec::new();
    for (li, p) in pair_l.iter().enumerate() {
        if let Some(v) = *p {
            let (a, b) = (monos[lefts[li]], monos[v]);
            used[lefts[li]] = true;
            used[v] = true;
            cubes.push((a & b, a ^ b));
        }
    }
    let rest: Vec<u64> = (0..monos.len()).filter(|&i| !used[i]).map(|i| monos[i]).collect();
    (cubes, rest)
}

// The pre-2026-08 greedy pairing (pair m with m minus one bit), as the
// fallback when the monomial set is too large for Hopcroft-Karp.
fn greedy_pairs(monos: &[u64]) -> (Vec<(u64, u64)>, Vec<u64>) {
    let mut order: Vec<u64> = monos.to_vec();
    order.sort_unstable_by_key(|&m| std::cmp::Reverse((m.count_ones(), m)));
    let pos_of: HashMap<u64, usize> = order.iter().enumerate().map(|(i, &m)| (m, i)).collect();
    let mut matched = vec![false; order.len()];
    let mut cubes = Vec::new();
    let mut rest = Vec::new();
    for i in 0..order.len() {
        if matched[i] {
            continue;
        }
        let m = order[i];
        let mut bits = m;
        let mut paired = false;
        while bits != 0 {
            let b = bits & bits.wrapping_neg();
            bits &= bits - 1;
            if let Some(&j) = pos_of.get(&(m & !b)) {
                if !matched[j] && j != i {
                    matched[i] = true;
                    matched[j] = true;
                    cubes.push((m & !b, b));
                    paired = true;
                    break;
                }
            }
        }
        if !paired {
            matched[i] = true;
            rest.push(m);
        }
    }
    (cubes, rest)
}

// ---- greedy subcube covering ----------------------------------------------
//
// A cube with negative-literal mask N replaces the 2^|N| monomials
// {pos|S : S subset of N} in one gate, so hunting for fully-present subcubes
// of dimension >= 2 beats any pairing. Best-first (largest dimension each
// round) up to COVER_EXHAUSTIVE_CAP monomials, single ascending-popcount pass
// beyond. Returns (cover cubes, residual monomials in input order).
const COVER_EXHAUSTIVE_CAP: usize = 1024;

fn cover_grow(m: u64, set: &HashSet<u64>, nbits: usize) -> (u64, Vec<u64>) {
    let mut nmask = 0u64;
    let mut exp = vec![m];
    for b in 0..nbits {
        let bit = 1u64 << b;
        if m & bit != 0 || nmask & bit != 0 {
            continue;
        }
        if exp.iter().all(|&e| set.contains(&(e | bit))) {
            let add: Vec<u64> = exp.iter().map(|&e| e | bit).collect();
            exp.extend(add);
            nmask |= bit;
        }
    }
    (nmask, exp)
}

fn greedy_cover(monos: &[u64], nbits: usize) -> (Vec<(u64, u64)>, Vec<u64>) {
    let mut set: HashSet<u64> = monos.iter().copied().collect();
    let mut cubes: Vec<(u64, u64)> = Vec::new();
    if monos.len() <= COVER_EXHAUSTIVE_CAP {
        loop {
            let mut best: Option<(u32, u64, u64, Vec<u64>)> = None;
            for &m in monos {
                if !set.contains(&m) {
                    continue;
                }
                let (nmask, exp) = cover_grow(m, &set, nbits);
                let dim = nmask.count_ones();
                if dim >= 2 && best.as_ref().is_none_or(|b| dim > b.0) {
                    best = Some((dim, m, nmask, exp));
                }
            }
            let Some((_, base, nmask, exp)) = best else { break };
            cubes.push((base, nmask));
            for e in exp {
                set.remove(&e);
            }
        }
    } else {
        let mut by_pop: Vec<u64> = monos.to_vec();
        by_pop.sort_unstable_by_key(|&m| (m.count_ones(), m));
        for m in by_pop {
            if !set.contains(&m) {
                continue;
            }
            let (nmask, exp) = cover_grow(m, &set, nbits);
            if nmask.count_ones() >= 2 {
                cubes.push((m, nmask));
                for e in exp {
                    set.remove(&e);
                }
            }
        }
    }
    let residual: Vec<u64> = monos.iter().copied().filter(|m| set.contains(m)).collect();
    (cubes, residual)
}

// ANF rewrite of a cube set over its support: expand mixed-polarity cubes
// into positive monomials (canonical, so all cancellation happens), then
// re-express the monomial set as few cubes as found among: an exact minimum
// ESOP (support <= 4), greedy subcube covering + maximum matching on the
// residual, and maximum matching alone (covering can lose when it strands
// monomials the matching wanted). The zero monomial may be consumed by a
// cover/pair cube (as a pure-negative cube) or left to the parity delta.
// Returns (cubes, parity_delta, exact_used) or None when the support or the
// expansion would be too large.
// Canonical ANF of a cube set: the sorted support and the sorted positive
// monomials (bit i = support[i]) that survive cancellation. None when the
// support exceeds `support_cap` (hard cap 63) or the expansion budget.
fn anf_expand(cubes: &[Lits], support_cap: usize) -> Option<(Vec<u16>, Vec<u64>)> {
    let mut support: Vec<u16> = cubes
        .iter()
        .flat_map(|c| c.iter().map(|&(w, _)| w))
        .collect();
    support.sort_unstable();
    support.dedup();
    let n = support.len();
    if n > support_cap.min(63) {
        return None;
    }
    let idx_of: FxHashMap<u16, u32> = support
        .iter()
        .enumerate()
        .map(|(i, &w)| (w, i as u32))
        .collect();
    let mut budget = 1u64 << 18;
    let mut anf = FxHashSet::<u64>::default();
    for c in cubes {
        let (mut pos, mut neg) = (0u64, 0u64);
        for &(w, p) in c {
            let b = 1u64 << idx_of[&w];
            if p { pos |= b } else { neg |= b }
        }
        let terms = 1u64 << neg.count_ones();
        if terms > budget {
            return None;
        }
        budget -= terms;
        // cube = AND(pos) * PROD(1 XOR w in neg) = XOR over subsets of neg.
        let mut sub = neg;
        loop {
            let monomial = pos | sub;
            if !anf.insert(monomial) {
                anf.remove(&monomial);
            }
            if sub == 0 {
                break;
            }
            sub = (sub - 1) & neg;
        }
    }
    let mut monos: Vec<u64> = anf.into_iter().collect();
    monos.sort_unstable();
    Some((support, monos))
}

fn anf_reduce(cubes: &[Lits], support_cap: usize) -> Option<(Vec<Lits>, bool, bool)> {
    let (support, monos) = anf_expand(cubes, support_cap)?;
    Some(esop_from_monomials(&support, &monos))
}

// Deterministic ESOP from a canonical monomial set over a sorted support
// (bit i of a monomial = support[i], n <= 63): best of the exact minimum
// table (n <= 4), greedy subcube cover + maximum matching, matching alone.
// Depends on nothing but (support, monomials), so it is a function of the
// activation function: applied to a packed ANF gate it yields one
// compacted spelling per function (postprocessing::compress::compact).
// Returns (cubes, parity_delta, exact_used).
fn esop_from_monomials(support: &[u16], monos: &[u64]) -> (Vec<Lits>, bool, bool) {
    let n = support.len();
    if monos.is_empty() {
        return (Vec::new(), false, false);
    }
    // Candidate strategies, first-listed wins ties on (cubes, lits).
    let mut cands: Vec<(Vec<(u64, u64)>, bool, bool)> = Vec::new();
    if (1..=4).contains(&n) {
        let (cs, delta) = exact_small(&monos, n);
        cands.push((cs, delta, true));
    }
    {
        let (mut cs, resid) = greedy_cover(&monos, n);
        let (pairs, singles) = match_pairs(&resid, n);
        cs.extend(pairs);
        let mut delta = false;
        for &m in &singles {
            if m == 0 { delta = true } else { cs.push((m, 0)) }
        }
        cands.push((cs, delta, false));
    }
    {
        let (mut cs, singles) = match_pairs(&monos, n);
        let mut delta = false;
        for &m in &singles {
            if m == 0 { delta = true } else { cs.push((m, 0)) }
        }
        cands.push((cs, delta, false));
    }
    let (mut best, delta, exact) = cands
        .into_iter()
        .min_by_key(|(cs, _, _)| {
            (
                cs.len(),
                cs.iter().map(|&(p, ng)| (p | ng).count_ones() as u64).sum::<u64>(),
            )
        })
        .expect("at least one strategy");
    best.sort_unstable_by_key(|&(p, ng)| (std::cmp::Reverse((p | ng).count_ones()), p, ng));
    let to_lits = |pos: u64, neg: u64| -> Lits {
        let mut l: Lits = Lits::new();
        for (i, &w) in support.iter().enumerate() {
            if pos >> i & 1 == 1 {
                l.push((w, true));
            } else if neg >> i & 1 == 1 {
                l.push((w, false));
            }
        }
        l
    };
    let out: Vec<Lits> = best.iter().map(|&(p, ng)| to_lits(p, ng)).collect();
    (out, delta, exact)
}

// Reduce one gathered group to a minimal-found cube list and emit as XGates
// (parity absorbed as comp on the first cube, or an X gate if none remain).
fn reduce_group(
    target: u16,
    members: &[XGate],
    p: &CompressParams,
    rng: &mut StdRng,
    rep: &mut CompressReport,
) -> Vec<XGate> {
    rep.groups += 1;
    rep.max_group = rep.max_group.max(members.len());
    if members.len() == 1 {
        return members.to_vec();
    }
    rep.multi_groups += 1;
    let mut parity = false;
    let mut cubes: Vec<Lits> = Vec::with_capacity(members.len());
    for m in members {
        debug_assert_eq!(m.target, target);
        if m.comp {
            parity = !parity;
        }
        cubes.push(m.ctrls.clone());
    }
    // Pairwise catalogue (cancel / drop-literal / subsume) to a fixed point.
    'outer: loop {
        for i in 0..cubes.len() {
            // `a` depends only on `i`; every merge restarts the scan from the
            // outer loop, so it can never go stale inside the `j` sweep.
            let a = XGate {
                target,
                comp: false,
                ctrls: cubes[i].clone(),
            };
            for j in i + 1..cubes.len() {
                let b = XGate {
                    target,
                    comp: false,
                    ctrls: cubes[j].clone(),
                };
                if let Some(m) = merge_result(&a, &b) {
                    rep.catalogue_merges += 1;
                    let repl = match m {
                        Merge::Cancel => None,
                        Merge::DropLit(g) | Merge::Subsume(g) | Merge::XFuse(g) => Some(g.ctrls),
                        // Both operands are built comp=false two lines above,
                        // and Absorb requires a comp=1 partner, so it cannot
                        // arise here. Assert rather than map, so that changing
                        // the operands fails loudly instead of silently
                        // dropping the comp bit the merge was carrying.
                        Merge::Absorb(_) => {
                            unreachable!("ESOP cubes are comp=0; Absorb needs a comp=1 partner")
                        }
                    };
                    cubes.swap_remove(j);
                    match repl {
                        Some(c) => cubes[i] = c,
                        None => {
                            cubes.swap_remove(i);
                        }
                    }
                    continue 'outer;
                }
            }
        }
        break;
    }
    // ANF alternative: canonical cancellation, kept when strictly smaller.
    // Two-member groups belong here too: pairs whose XOR is one COMPLEMENTED
    // cube (presplit-rejoin, const-XOR-cube) are exactly what the pairwise
    // catalogue must refuse but the parity slot absorbs for free.
    if cubes.len() >= 2 {
        if let Some((alt, delta, exact)) = anf_reduce(&cubes, p.anf_support_cap) {
            let (alt_l, cur_l) = (
                alt.iter().map(|c| c.len()).sum::<usize>(),
                cubes.iter().map(|c| c.len()).sum::<usize>(),
            );
            if (alt.len(), alt_l) < (cubes.len(), cur_l) {
                rep.anf_wins += 1;
                if exact {
                    rep.exact_wins += 1;
                }
                cubes = alt;
                parity ^= delta;
            }
        }
    }
    let mut out: Vec<XGate> = Vec::with_capacity(cubes.len().max(1));
    for (k, c) in cubes.into_iter().enumerate() {
        out.push(XGate {
            target,
            comp: parity && k == 0,
            ctrls: c,
        });
    }
    if out.is_empty() && parity {
        out.push(XGate::x_gate(target));
    }
    if p.local_verify {
        // An identity reduction (output == input gate-for-gate) is functionally
        // equal by construction; verifying it exhaustively is pure waste and
        // ~95% of multi-group reductions are identities. The verify rng feeds
        // nothing but the assertion, so skipping its draws cannot reach the
        // output bytes.
        if out == members {
            rep.verifies_skipped += 1;
        } else {
            verify_group(members, &out, rng);
        }
    }
    out
}

// Per-gate bitmask compilation over the indexed sorted support: each cube is
// flattened at `words` u64s per gate as (positive-literal, negative-literal)
// masks plus its comp bit.  Bit `i` of an assignment corresponds to
// `support[i]`.
fn cube_masks(gates: &[XGate], support: &[u16], words: usize) -> (Vec<bool>, Vec<u64>, Vec<u64>) {
    let mut comps = Vec::with_capacity(gates.len());
    let mut pos = vec![0u64; gates.len() * words];
    let mut neg = vec![0u64; gates.len() * words];
    for (i, g) in gates.iter().enumerate() {
        comps.push(g.comp);
        for &(w, p) in &g.ctrls {
            let bit = support
                .binary_search(&w)
                .expect("verify support must contain every cube wire");
            let slot = i * words + bit / 64;
            let mask = 1u64 << (bit % 64);
            if p {
                pos[slot] |= mask;
            } else {
                neg[slot] |= mask;
            }
        }
    }
    (comps, pos, neg)
}

// XOR of the compiled cubes on one assignment bitset: a cube fires iff every
// positive bit is set and every negative bit is clear.
fn masked_parity(comps: &[bool], pos: &[u64], neg: &[u64], words: usize, assign: &[u64]) -> bool {
    let mut acc = false;
    for (i, &comp) in comps.iter().enumerate() {
        let base = i * words;
        let mut fires = true;
        for word in 0..words {
            let a = assign[word];
            if a & pos[base + word] != pos[base + word] || a & neg[base + word] != 0 {
                fires = false;
                break;
            }
        }
        acc ^= comp ^ fires;
    }
    acc
}

/// Original dyn-Fn cube evaluation, retained as the equivalence reference for
/// `opt_equiv_masked_parity_matches_reference`.
#[cfg(test)]
fn parity_of_reference(gates: &[XGate], val: &dyn Fn(u16) -> bool) -> bool {
    let mut acc = false;
    for g in gates {
        acc ^= g.comp ^ g.ctrls.iter().all(|&(w, p)| val(w) == p);
    }
    acc
}

fn verify_group(before: &[XGate], after: &[XGate], rng: &mut StdRng) {
    let mut support: Vec<u16> = before
        .iter()
        .chain(after)
        .flat_map(|g| g.ctrls.iter().map(|&(w, _)| w))
        .collect();
    support.sort_unstable();
    support.dedup();
    let words = support.len().div_ceil(64).max(1);
    let (before_comps, before_pos, before_neg) = cube_masks(before, &support, words);
    let (after_comps, after_pos, after_neg) = cube_masks(after, &support, words);
    let check = |assign: &[u64]| {
        assert_eq!(
            masked_parity(&before_comps, &before_pos, &before_neg, words, assign),
            masked_parity(&after_comps, &after_pos, &after_neg, words, assign),
            "fcompress group reduction changed the function"
        );
    };
    if support.len() <= 16 {
        for a in 0u32..(1u32 << support.len()) {
            check(&[a as u64]);
        }
    } else {
        let mut assign = vec![0u64; words];
        for _ in 0..512 {
            assign.fill(0);
            // One draw per support wire in ascending wire order: the exact
            // sequence the HashMap-based sampler drew.
            for bit in 0..support.len() {
                if rng.random_bool(0.5) {
                    assign[bit / 64] |= 1u64 << (bit % 64);
                }
            }
            check(&assign);
        }
    }
}

struct Group {
    target: u16,
    members: Vec<XGate>,
    union: Vec<u16>, // sorted control-wire union across members
    last: usize,     // index of the last member in the input order
    open: bool,
    // OR of the members' ancestor sets (empty when ancestry is not threaded).
    // Every emitted survivor of the group carries this union: each cube of
    // the reduced ESOP derives from the whole gathered run.
    anc: AncBits,
    // Groups this one must be emitted AFTER: the group of every writer it
    // was transported across (its cubes are written in the frame after that
    // writer). Acyclic by construction; entries may be closed already.
    deps: SmallVec<[usize; 4]>,
}

// Does open group `a` transitively depend on `b`? A closed group's
// dependencies are closed too (closing cascades into them), so only open
// slots are walked.
fn depends_on(slots: &[Group], a: usize, b: usize) -> bool {
    let mut stack: SmallVec<[usize; 16]> = SmallVec::from_slice(&[a]);
    let mut seen: SmallVec<[usize; 16]> = SmallVec::new();
    while let Some(x) = stack.pop() {
        if x == b {
            return true;
        }
        if seen.contains(&x) {
            continue;
        }
        seen.push(x);
        for &d in &slots[x].deps {
            if slots[d].open {
                stack.push(d);
            }
        }
    }
    false
}

// Dependencies-first order of a close set (ascending last-member order among
// independent groups): `set` is already sorted by last member.
fn topo_visit(s: usize, slots: &[Group], set: &[usize], order: &mut Vec<usize>) {
    if order.contains(&s) {
        return;
    }
    for &d in slots[s].deps.iter() {
        if set.contains(&d) {
            topo_visit(d, slots, set, order);
        }
    }
    order.push(s);
}

// Float a group's ESOP across `h`, a writer of one of its union wires that
// does not read its target: the substitution u <- u XOR fire(h) on h's
// target (downhill's conjugation), accepted when the catalogue-reduced
// result is no larger than the catalogue-reduced current members in
// (gates, lits), or grows by at most `slack` gates. Returns the new member
// list and whether the ESOP changed; an unchanged ESOP means the group
// commutes with h and needs no frame dependency (the caller then keeps its
// raw members).
fn transport_across(
    members: &[XGate],
    target: u16,
    h: &XGate,
    slack: usize,
) -> Option<(Vec<XGate>, bool)> {
    use super::downhill::{conjugate, esop_equal, from_block, gate_cost, gates_of, lit_cost};
    let before = from_block(members, target);
    let after = conjugate(&before, h, target);
    let (bg, bl) = (gate_cost(&before), lit_cost(&before));
    let (ag, al) = (gate_cost(&after), lit_cost(&after));
    let ok = if slack == 0 {
        (ag, al) <= (bg, bl)
    } else {
        ag <= bg + slack
    };
    if !ok {
        return None;
    }
    let changed = !esop_equal(&before, &after);
    Some((gates_of(after, target), changed))
}

// One forward gather-and-reduce sweep. `anc` (when present) is aligned with
// `gates`; the returned tags are aligned with the returned gates.
fn gather_reduce_pass(
    gates: &[XGate],
    anc: Option<&[AncBits]>,
    wires: usize,
    p: &CompressParams,
    rng: &mut StdRng,
    rep: &mut CompressReport,
) -> (Vec<XGate>, Option<Vec<AncBits>>) {
    let mut out: Vec<XGate> = Vec::with_capacity(gates.len());
    let mut out_anc: Option<Vec<AncBits>> = anc.map(|_| Vec::with_capacity(gates.len()));
    let mut slots: Vec<Group> = Vec::new();
    let mut open_at: Vec<Option<usize>> = vec![None; wires]; // target wire -> slot
    let mut union_of: Vec<Vec<usize>> = vec![Vec::new(); wires]; // wire -> slots (stale ok)

    // Close a seed set of groups: cascade into their frame dependencies (a
    // transported group is emitted after the groups it crossed, and closing
    // a group early is always legal), order dependencies-first with ascending
    // last-member order among independent groups, reduce, emit. Groups with
    // no dependency path between them commute, so that order is legal.
    let close = |seed: SmallVec<[usize; 16]>,
                 slots: &mut Vec<Group>,
                 open_at: &mut Vec<Option<usize>>,
                 out: &mut Vec<XGate>,
                 out_anc: &mut Option<Vec<AncBits>>,
                 rng: &mut StdRng,
                 rep: &mut CompressReport| {
        let mut set: Vec<usize> = Vec::new();
        let mut queue = seed;
        while let Some(s) = queue.pop() {
            // The first visit closes the slot before enqueuing its
            // dependencies, so `open` is also the exact visited bit.
            if !slots[s].open {
                continue;
            }
            set.push(s);
            slots[s].open = false;
            queue.extend(slots[s].deps.iter().copied());
        }
        set.sort_unstable_by_key(|&s| slots[s].last);
        let mut order: Vec<usize> = Vec::with_capacity(set.len());
        for &s in &set {
            topo_visit(s, slots, &set, &mut order);
        }
        for s in order {
            let g = &slots[s];
            if open_at[g.target as usize] == Some(s) {
                open_at[g.target as usize] = None;
            }
            let cubes = reduce_group(g.target, &slots[s].members, p, rng, rep);
            if let Some(oa) = out_anc.as_mut() {
                for _ in 0..cubes.len() {
                    oa.push(slots[s].anc.clone());
                }
            }
            out.extend(cubes);
        }
    };

    for (i, g) in gates.iter().enumerate() {
        let u = g.target as usize;
        // Reads close the groups accumulating on the wires they read, unless
        // the reader is separated from every member (it then commutes with
        // the group as a whole and the group floats past it).
        let mut seed = SmallVec::<[usize; 16]>::new();
        for &(w, _) in &g.ctrls {
            if let Some(s) = open_at[w as usize] {
                if p.sep_reads && slots[s].members.iter().all(|m| !XGate::collides(m, g)) {
                    rep.sep_passes += 1;
                } else {
                    seed.push(s);
                }
            }
        }
        // The write to u: every open group with u in its union either floats
        // across g by conjugation (candidates below), commutes with g (g reads
        // its target but is separated from every member -- the read rule
        // just let it pass), or closes.
        let mut cand = SmallVec::<[usize; 8]>::new();
        for &s in &union_of[u] {
            if !slots[s].open {
                continue;
            }
            if g.reads(slots[s].target) {
                if !seed.contains(&s) {
                    // separated pass: the group commutes with g
                    continue;
                }
                seed.push(s);
            } else if p.transport {
                cand.push(s);
            } else {
                seed.push(s);
            }
        }
        union_of[u].retain(|&s| slots[s].open);
        if !seed.is_empty() {
            close(seed, &mut slots, &mut open_at, &mut out, &mut out_anc, rng, rep);
        }
        // Decide every candidate before applying any transport: a refusal
        // closes the group, and that close cascades into its dependencies --
        // possibly the open group on u (so the writer would open a fresh
        // slot) or another candidate. Nothing may be transported into the
        // post-g frame until every close of this step is done, and the slot
        // g will join or open is fixed only after those closes.
        let mut accepted: SmallVec<[(usize, Vec<XGate>, bool); 8]> = SmallVec::new();
        let mut refused = SmallVec::<[usize; 16]>::new();
        for s in cand {
            if !slots[s].open {
                continue; // closed above as somebody's dependency
            }
            match transport_across(&slots[s].members, slots[s].target, g, p.transport_slack) {
                Some((new_members, changed)) => accepted.push((s, new_members, changed)),
                None => {
                    rep.transport_refused += 1;
                    refused.push(s);
                }
            }
        }
        if !refused.is_empty() {
            close(refused, &mut slots, &mut open_at, &mut out, &mut out_anc, rng, rep);
        }
        // The slot g joins: opened now (empty) when none is open on u, so a
        // dependency recorded below always names an existing slot.
        let h_slot = match open_at[u] {
            Some(s) => s,
            None => {
                let s = slots.len();
                slots.push(Group {
                    target: g.target,
                    members: Vec::new(),
                    union: Vec::new(),
                    last: i,
                    open: true,
                    anc: AncBits::new(),
                    deps: SmallVec::new(),
                });
                open_at[u] = Some(s);
                s
            }
        };
        // A transported group depends on g's group; refuse (close) any whose
        // dependency would close a cycle. Such a close cannot reach g's group
        // (that group depends on the refused one, so it is not among the
        // refused one's dependencies) and so h_slot stays valid.
        let mut cyc = SmallVec::<[usize; 16]>::new();
        for (s, _, changed) in &accepted {
            if *changed && slots[*s].open && depends_on(&slots, h_slot, *s) {
                rep.transport_cycle_refused += 1;
                cyc.push(*s);
            }
        }
        if !cyc.is_empty() {
            close(cyc, &mut slots, &mut open_at, &mut out, &mut out_anc, rng, rep);
        }
        debug_assert_eq!(open_at[u], Some(h_slot));
        for (s, new_members, changed) in accepted {
            if !slots[s].open {
                continue;
            }
            if !changed {
                rep.transport_noops += 1;
                continue;
            }
            if p.local_verify {
                let mut before: Vec<XGate> = slots[s].members.clone();
                before.push(g.clone());
                let mut after: Vec<XGate> = vec![g.clone()];
                after.extend(new_members.iter().cloned());
                super::downhill::verify_span(&before, &after, rng);
            }
            rep.transports += 1;
            let grp = &mut slots[s];
            grp.members = new_members;
            for m in &grp.members {
                for &(w, _) in &m.ctrls {
                    if grp.union.binary_search(&w).is_err() {
                        let pos = grp.union.partition_point(|&x| x < w);
                        grp.union.insert(pos, w);
                        union_of[w as usize].push(s);
                    }
                }
            }
            if !grp.deps.contains(&h_slot) {
                grp.deps.push(h_slot);
            }
        }
        let g_anc = anc.map(|a| a[i].clone()).unwrap_or_default();
        // Join the group for this target (opened above when it did not exist).
        let grp = &mut slots[h_slot];
        for &(w, _) in &g.ctrls {
            if grp.union.binary_search(&w).is_err() {
                let pos = grp.union.partition_point(|&x| x < w);
                grp.union.insert(pos, w);
                union_of[w as usize].push(h_slot);
            }
        }
        grp.members.push(g.clone());
        or_anc(&mut grp.anc, &g_anc);
        grp.last = i;
        if grp.members.len() >= p.group_cap {
            close(
                SmallVec::from_slice(&[h_slot]),
                &mut slots,
                &mut open_at,
                &mut out,
                &mut out_anc,
                rng,
                rep,
            );
        }
    }
    let remaining: SmallVec<[usize; 16]> = (0..slots.len()).filter(|&s| slots[s].open).collect();
    close(
        remaining,
        &mut slots,
        &mut open_at,
        &mut out,
        &mut out_anc,
        rng,
        rep,
    );
    (out, out_anc)
}

// ---- packing ---------------------------------------------------------------
//
// At a fixed point of the gather every maximal run of consecutive same-target
// gates is one gathered group (a group floats to one close point and is
// emitted there, and runs that could still float together would have been
// merged). Packing spells each run as ONE generalized gate t ^= f(controls)
// with f in algebraic normal form -- the XOR of positive monomials, the
// unique representation of a Boolean function. The point is not size (the
// ANF is ~2.4x the cube count on GSS finals) but the removal of information:
// the cube list fcompress emits is the catalogue-reduced descendant of the
// cubes the mixer left, so it carries history, while the ANF depends on
// nothing but the function. Exact and attacker-computable like the rest of
// the pass. Monomials are ascending wire lists sorted by (degree, wires);
// the empty monomial is the constant 1 (a comp bit). Any support size.
fn pack_run(run: &[XGate]) -> PackedGate {
    let target = run[0].target;
    let mut set: FxHashSet<Vec<u16>> = FxHashSet::default();
    let mut toggle = |m: Vec<u16>| {
        if !set.remove(&m) {
            set.insert(m);
        }
    };
    for g in run {
        debug_assert_eq!(g.target, target);
        if g.comp {
            toggle(Vec::new());
        }
        let pos: Vec<u16> = g.ctrls.iter().filter(|l| l.1).map(|l| l.0).collect();
        let neg: Vec<u16> = g.ctrls.iter().filter(|l| !l.1).map(|l| l.0).collect();
        assert!(neg.len() <= 24, "cube with {} negative literals: expansion too large", neg.len());
        // AND(pos) * PROD(1 XOR w in neg) = XOR over subsets of neg.
        for sub in 0..(1u64 << neg.len()) {
            let mut m = pos.clone();
            for (b, &w) in neg.iter().enumerate() {
                if sub >> b & 1 == 1 {
                    m.push(w);
                }
            }
            m.sort_unstable();
            toggle(m);
        }
    }
    let mut g = PackedGate {
        target,
        terms: set
            .into_iter()
            .map(|m| m.into_iter().map(|w| (w, true)).collect())
            .collect(),
    };
    g.sort_terms();
    g
}

/// Compaction: rewrite a packed ANF gate as a mixed-polarity ESOP by the
/// deterministic reducer strategies, from the ANF ALONE (never from the
/// cubes the ANF came from), so the result is still one spelling per
/// activation function -- at ~2.3x fewer terms than the ANF. Gates whose
/// support exceeds the 63-wire mask width are left in ANF (still canonical).
pub fn compact_gate(g: &PackedGate) -> PackedGate {
    debug_assert!(g.is_anf(), "compaction starts from the ANF");
    let mut support: Vec<u16> = g.terms.iter().flatten().map(|l| l.0).collect();
    support.sort_unstable();
    support.dedup();
    if support.len() > 63 {
        return g.clone();
    }
    let mut monos: Vec<u64> = g
        .terms
        .iter()
        .map(|t| {
            t.iter().fold(0u64, |m, &(w, _)| {
                m | 1u64 << support.binary_search(&w).expect("wire in support")
            })
        })
        .collect();
    monos.sort_unstable();
    let (cubes, parity, _) = esop_from_monomials(&support, &monos);
    let mut out = PackedGate {
        target: g.target,
        terms: cubes.into_iter().map(|c| c.into_iter().collect()).collect(),
    };
    if parity {
        out.terms.push(Vec::new());
    }
    out.sort_terms();
    debug_assert!(out.terms.len() <= g.terms.len());
    out
}

pub fn compact(packed: &[PackedGate]) -> Vec<PackedGate> {
    packed.iter().map(compact_gate).collect()
}

/// Pack every maximal same-target run into one canonical ANF gate. Exact for
/// any gate list; canonical (one representation per activation function) and
/// maximally packed when the list is a gather fixed point, i.e. fcompress
/// output.
pub fn pack(gates: &[XGate]) -> Vec<PackedGate> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < gates.len() {
        let mut j = i + 1;
        while j < gates.len() && gates[j].target == gates[i].target {
            j += 1;
        }
        out.push(pack_run(&gates[i..j]));
        i = j;
    }
    out
}

// Packing census: what packing leaves and what the canonical form costs.
// Per run: current cube count, ANF monomials, support, monomial degrees;
// plus the size of the deterministic canonical ESOP (anf_reduce applied to
// the ANF, support <= 63 only). Prints a few lines.
pub fn pack_census(gates: &[XGate]) {
    fn bin(x: usize) -> usize {
        match x {
            0 => 0,
            1 => 1,
            2 => 2,
            3..=4 => 3,
            5..=8 => 4,
            9..=16 => 5,
            17..=32 => 6,
            33..=64 => 7,
            65..=256 => 8,
            _ => 9,
        }
    }
    const LABELS: [&str; 10] =
        ["0", "1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65-256", ">256"];
    let mut cubes_h = [0usize; 10];
    let mut anf_h = [0usize; 10];
    let mut sup_h = [0usize; 10];
    let mut canon_h = [0usize; 10];
    let mut deg_h = [0usize; 10];
    let (mut runs, mut multi_runs, mut multi_mass) = (0usize, 0usize, 0usize);
    let (mut anf_total, mut anf_max, mut sup_max) = (0usize, 0usize, 0usize);
    let (mut cube_lits, mut mono_degs) = (0usize, 0usize);
    let (mut blowup_runs, mut blowup_extra) = (0usize, 0usize);
    let (mut canon_total, mut canon_cubes, mut canon_skipped) = (0usize, 0usize, 0usize);
    let (mut canon_smaller, mut canon_larger, mut canon_gain, mut canon_loss) =
        (0usize, 0usize, 0usize, 0usize);
    let mut i = 0usize;
    while i < gates.len() {
        let mut j = i + 1;
        while j < gates.len() && gates[j].target == gates[i].target {
            j += 1;
        }
        let run = &gates[i..j];
        let k = run.len();
        runs += 1;
        cubes_h[bin(k)] += 1;
        if k > 1 {
            multi_runs += 1;
            multi_mass += k;
        }
        cube_lits += run.iter().map(XGate::width).sum::<usize>();
        let pg = pack_run(run);
        let m = pg.terms.len();
        anf_h[bin(m)] += 1;
        anf_total += m;
        anf_max = anf_max.max(m);
        let mut support: Vec<u16> = pg.terms.iter().flatten().map(|l| l.0).collect();
        support.sort_unstable();
        support.dedup();
        sup_h[bin(support.len())] += 1;
        sup_max = sup_max.max(support.len());
        for mo in &pg.terms {
            mono_degs += mo.len();
            deg_h[bin(mo.len())] += 1;
        }
        if m > k {
            blowup_runs += 1;
            blowup_extra += m - k;
        }
        if support.len() <= 63 {
            let canon = compact_gate(&pg).terms.len();
            canon_h[bin(canon)] += 1;
            canon_total += canon;
            canon_cubes += k;
            if canon < k {
                canon_smaller += 1;
                canon_gain += k - canon;
            } else if canon > k {
                canon_larger += 1;
                canon_loss += canon - k;
            }
        } else {
            canon_skipped += 1;
        }
        i = j;
    }
    let g = gates.len();
    println!(
        "[pack] gates={} runs(=packed gates)={} saved={} ({:.1}%) | multi runs={} ({:.1}% of runs) holding {} gates ({:.1}% of mass)",
        g, runs, g - runs, 100.0 * (g - runs) as f64 / g.max(1) as f64,
        multi_runs, 100.0 * multi_runs as f64 / runs.max(1) as f64,
        multi_mass, 100.0 * multi_mass as f64 / g.max(1) as f64
    );
    println!(
        "[pack] canonical ANF: {} monomials for {} cubes (x{:.2}); runs where ANF > cubes: {} (+{} monomials); max monomials in a run: {}; max support: {} wires",
        anf_total, g, anf_total as f64 / g.max(1) as f64, blowup_runs, blowup_extra, anf_max, sup_max
    );
    println!(
        "[pack] description mass: cubes carry {} literals, ANF carries {} wire occurrences (x{:.2}); mean cube width {:.2}, mean monomial degree {:.2}",
        cube_lits, mono_degs, mono_degs as f64 / cube_lits.max(1) as f64,
        cube_lits as f64 / g.max(1) as f64, mono_degs as f64 / anf_total.max(1) as f64
    );
    println!(
        "[pack] compacted ESOP (deterministic, from the ANF alone, support<=63; {} runs skipped): {} terms vs {} cubes now; smaller on {} runs (-{}), larger on {} runs (+{})",
        canon_skipped, canon_total, canon_cubes, canon_smaller, canon_gain, canon_larger, canon_loss
    );
    let show = |name: &str, h: &[usize; 10]| {
        let parts: Vec<String> = LABELS
            .iter()
            .zip(h.iter())
            .filter(|(_, c)| **c > 0)
            .map(|(l, c)| format!("{l}:{c}"))
            .collect();
        println!("[pack] {name}: {}", parts.join(" "));
    };
    show("cubes per run", &cubes_h);
    show("ANF monomials per run", &anf_h);
    show("monomial degree", &deg_h);
    show("compacted ESOP terms per run", &canon_h);
    show("support wires per run", &sup_h);
}

// Full pass: [liveness] -> gather+reduce, iterated to a gate-count fixed
// point. Prints one [fcompress] line per iteration.
pub fn compress(
    gates: Vec<XGate>,
    wires: usize,
    p: &CompressParams,
) -> (Vec<XGate>, CompressReport) {
    let (out, _, rep) = compress_anc(gates, None, wires, p);
    (out, rep)
}

// Same pass with per-gate ancestor sets threaded through: sets follow gates
// under gathering, group survivors carry the member union, pruned gates drop.
// `anc` must be aligned with `gates`; the returned tags align with the output.
pub fn compress_anc(
    gates: Vec<XGate>,
    anc: Option<Vec<AncBits>>,
    wires: usize,
    p: &CompressParams,
) -> (Vec<XGate>, Option<Vec<AncBits>>, CompressReport) {
    if let Some(a) = &anc {
        assert_eq!(
            a.len(),
            gates.len(),
            "ancestry tags must align with the input gates"
        );
    }
    let mut rng = StdRng::seed_from_u64(p.seed);
    let mut rep = CompressReport::default();
    rep.gates_in = gates.len();
    rep.lits_in = lits_of(&gates);
    let mut cur = gates;
    let mut cur_anc = anc;
    let mut prev = (cur.len(), lits_of(&cur));
    for iter in 1..=p.max_iters {
        let before = cur.len();
        if let Some(z) = &p.zero_in {
            let (kept, kept_anc, killed, lits_dropped) =
                zero_specialize_anc(cur, cur_anc, z, wires);
            cur = kept;
            cur_anc = kept_anc;
            rep.zero_killed += killed;
            rep.zero_lits_dropped += lits_dropped;
        }
        if let Some(lv) = &p.live_out {
            let (kept, kept_anc, dropped) = liveness_prune_anc(cur, cur_anc, lv);
            cur = kept;
            cur_anc = kept_anc;
            rep.liveness_dropped += dropped;
        }
        let snapshot = (cur.clone(), cur_anc.clone());
        let (next, next_anc) =
            gather_reduce_pass(&cur, cur_anc.as_deref(), wires, p, &mut rng, &mut rep);
        cur = next;
        cur_anc = next_anc;
        if p.reverse_pass {
            // The reversed list is the inverse function (involutions), so a
            // forward gather of it is a leftward gather of the circuit.
            cur.reverse();
            if let Some(a) = cur_anc.as_mut() {
                a.reverse();
            }
            let (next, next_anc) =
                gather_reduce_pass(&cur, cur_anc.as_deref(), wires, p, &mut rng, &mut rep);
            cur = next;
            cur_anc = next_anc;
            cur.reverse();
            if let Some(a) = cur_anc.as_mut() {
                a.reverse();
            }
        }
        let mut dh_swaps = 0usize;
        if p.downhill {
            let (next, next_anc, swaps) =
                super::downhill::apply_pass(cur, cur_anc, &mut rng, p.local_verify);
            cur = next;
            cur_anc = next_anc;
            dh_swaps = swaps;
            rep.downhill_swaps += swaps as u64;
        }
        rep.iters = iter;
        println!(
            "[fcompress] iter={} gates {} -> {} | groups={} multi={} max={} | catalogue={} anf_wins={} exact={} downhill={} transport={} (noop={} refused={} cyc={}) sep={} live_dropped={} zero_killed={} vskip={}",
            iter,
            before,
            cur.len(),
            rep.groups,
            rep.multi_groups,
            rep.max_group,
            rep.catalogue_merges,
            rep.anf_wins,
            rep.exact_wins,
            dh_swaps,
            rep.transports,
            rep.transport_noops,
            rep.transport_refused,
            rep.transport_cycle_refused,
            rep.sep_passes,
            rep.liveness_dropped,
            rep.zero_killed,
            rep.verifies_skipped
        );
        // Progress = strictly smaller (gates, lits): downhill can shrink lits
        // at equal gate count, and that still enables later gather wins. The
        // pruners and downhill never regress the pair; gathering with
        // transport can in principle (a transported group may reduce worse
        // than its parts would have separately), so an iteration that ends
        // larger is discarded and the previous circuit kept.
        let now = (cur.len(), lits_of(&cur));
        if now > prev {
            println!(
                "[fcompress] iter={} regressed {:?} -> {:?}; keeping the previous circuit",
                iter, prev, now
            );
            cur = snapshot.0;
            cur_anc = snapshot.1;
            break;
        }
        if now == prev {
            break;
        }
        prev = now;
    }
    rep.gates_out = cur.len();
    rep.lits_out = lits_of(&cur);
    (cur, cur_anc, rep)
}

#[cfg(test)]
mod compress_tests {
    use super::*;
    use crate::circuit::xgate::{XGate, eval_lanes};

    fn conj(t: u16, lits: &[(u16, bool)]) -> XGate {
        XGate::conj(t, lits.iter().copied()).unwrap()
    }

    fn equal_on(a: &[XGate], b: &[XGate], wires: usize, live: Option<&[bool]>) -> bool {
        let mut rng = StdRng::seed_from_u64(99);
        for _ in 0..64 {
            let mut sa: Vec<u64> = (0..wires).map(|_| rng.random::<u64>()).collect();
            let mut sb = sa.clone();
            eval_lanes(a.iter(), &mut sa);
            eval_lanes(b.iter(), &mut sb);
            for w in 0..wires {
                if live.is_none_or(|l| l[w]) && sa[w] != sb[w] {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn opt_equiv_masked_parity_matches_reference() {
        let mut rng = StdRng::seed_from_u64(0x5EED_CAFE);
        for case in 0..240usize {
            // Wire pools spanning verify_group's exhaustive (<=16 support) and
            // sampled (>16) branches; the largest also forces the multi-word
            // mask path (support > 64 bits).
            let pool: u16 = [6, 20, 80][case % 3];
            let n_gates = rng.random_range(1..=10usize);
            let mut gates: Vec<XGate> = Vec::new();
            for _ in 0..n_gates {
                let width = rng.random_range(0..=6usize);
                let mut lits: Vec<(u16, bool)> = Vec::with_capacity(width);
                while lits.len() < width {
                    let w = rng.random_range(0..pool);
                    if lits.iter().all(|&(seen, _)| seen != w) {
                        lits.push((w, rng.random_bool(0.5)));
                    }
                }
                let mut g = XGate::conj(pool, lits).expect("distinct literals");
                g.comp = rng.random_bool(0.5);
                gates.push(g);
            }
            if pool > 64 {
                // Singletons on every pool wire guarantee a two-word support.
                for w in 0..pool {
                    let mut g = XGate::conj(pool, [(w, rng.random_bool(0.5))]).unwrap();
                    g.comp = rng.random_bool(0.5);
                    gates.push(g);
                }
            }
            let mut support: Vec<u16> = gates
                .iter()
                .flat_map(|g| g.ctrls.iter().map(|&(w, _)| w))
                .collect();
            support.sort_unstable();
            support.dedup();
            let words = support.len().div_ceil(64).max(1);
            let (comps, pos, neg) = cube_masks(&gates, &support, words);
            let compare = |bits: &[bool]| {
                let mut assign = vec![0u64; words];
                for (i, &b) in bits.iter().enumerate() {
                    if b {
                        assign[i / 64] |= 1u64 << (i % 64);
                    }
                }
                let reference = parity_of_reference(&gates, &|w| {
                    bits[support.binary_search(&w).expect("wire in support")]
                });
                assert_eq!(
                    masked_parity(&comps, &pos, &neg, words, &assign),
                    reference,
                    "case={case} bits={bits:?}"
                );
            };
            if support.len() <= 12 {
                for a in 0u32..(1u32 << support.len()) {
                    let bits: Vec<bool> = (0..support.len()).map(|i| a >> i & 1 == 1).collect();
                    compare(&bits);
                }
            }
            for _ in 0..64 {
                let bits: Vec<bool> = (0..support.len()).map(|_| rng.random_bool(0.5)).collect();
                compare(&bits);
            }
        }
    }

    #[test]
    fn adjacent_identical_gates_cancel() {
        let g = conj(0, &[(1, true), (2, false)]);
        let gates = vec![g.clone(), conj(3, &[(1, true)]), g.clone()];
        let (out, rep) = compress(gates.clone(), 4, &CompressParams::default());
        assert_eq!(out.len(), 1, "pair cancels across a non-conflicting gate");
        assert!(equal_on(&gates, &out, 4, None));
        assert!(rep.catalogue_merges >= 1);
    }

    #[test]
    fn reader_blocks_gather() {
        let g = conj(0, &[(1, true)]);
        // Gate 1 reads wire 0, pinning the two writes apart.
        let gates = vec![g.clone(), conj(2, &[(0, true)]), g.clone()];
        let (out, _) = compress(gates.clone(), 3, &CompressParams::default());
        assert_eq!(out.len(), 3, "reader of the target wire must block merging");
        assert!(equal_on(&gates, &out, 3, None));
    }

    #[test]
    fn control_write_blocks_gather() {
        let g = conj(0, &[(1, true)]);
        // Gate 1 writes wire 1 (g's control): second copy may not join.
        let gates = vec![g.clone(), conj(1, &[(2, true)]), g.clone()];
        let (out, _) = compress(gates.clone(), 3, &CompressParams::default());
        assert_eq!(out.len(), 3);
        assert!(equal_on(&gates, &out, 3, None));
    }

    #[test]
    fn comp_parity_folds() {
        // (1 XOR c) XOR c = 1: fossil + its own monomial fuse to an X gate.
        let mut fossil = conj(0, &[(1, false), (2, true)]);
        fossil.comp = true;
        let plain = conj(0, &[(1, false), (2, true)]);
        let gates = vec![fossil.clone(), plain];
        let (out, _) = compress(gates.clone(), 3, &CompressParams::default());
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].width(), 0);
        assert!(equal_on(&gates, &out, 3, None));
        // (1 XOR c) XOR (1 XOR c) = 0: two identical fossils vanish.
        let gates2 = vec![fossil.clone(), fossil];
        let (out2, _) = compress(gates2.clone(), 3, &CompressParams::default());
        assert!(out2.is_empty());
        assert!(equal_on(&gates2, &out2, 3, None));
    }

    #[test]
    fn anf_collapses_beyond_pairwise() {
        // x AND y, x AND NOT y, x: pairwise DropLit then Cancel wipes it out;
        // add a 3-cube ANF-only case too and check function preservation.
        let gates = vec![
            conj(0, &[(1, true), (2, true)]),
            conj(0, &[(1, true), (2, false)]),
            conj(0, &[(1, true)]),
        ];
        let (out, _) = compress(gates.clone(), 3, &CompressParams::default());
        assert!(out.is_empty(), "xy + x!y + x = 0");
        let gates2 = vec![
            conj(0, &[(1, true), (2, true)]),
            conj(0, &[(2, true), (3, true)]),
            conj(0, &[(1, true), (3, true)]),
            conj(0, &[(1, true), (2, true), (3, true)]),
        ];
        let (out2, _) = compress(gates2.clone(), 4, &CompressParams::default());
        assert!(equal_on(&gates2, &out2, 4, None));
        assert!(out2.len() <= 4);
    }

    #[test]
    fn ancestry_threads_through_compression() {
        // Two same-target gates that DropLit into one (t0 ^= x1  ⊕  t0 ^= ¬x1
        // → t0 ^= 1), carrying disjoint ancestor sets: the survivor must hold
        // the union. A bystander gate keeps its own set untouched.
        let g1 = conj(0, &[(1, true)]);
        let g2 = conj(0, &[(1, false)]);
        let by = conj(2, &[(3, true)]);
        let gates = vec![g1, g2, by];
        let anc = vec![vec![0b01u64], vec![0b10u64], vec![0b100u64]];
        let p = CompressParams::default();
        let (out, out_anc, rep) = compress_anc(gates.clone(), Some(anc), 4, &p);
        let out_anc = out_anc.expect("tags threaded");
        assert_eq!(
            out.len(),
            out_anc.len(),
            "tags must align with output gates"
        );
        assert!(rep.catalogue_merges >= 1, "the pair must merge");
        let mut found_union = false;
        for (g, a) in out.iter().zip(out_anc.iter()) {
            if g.target == 0 {
                assert_eq!(a, &vec![0b11u64], "survivor carries the members' union");
                found_union = true;
            } else {
                assert_eq!(a, &vec![0b100u64], "bystander keeps its own set");
            }
        }
        assert!(
            found_union,
            "a target-0 survivor must exist (parity X gate)"
        );
        // Function must be preserved with tags threaded (same pass, same rng).
        assert!(equal_on(&gates, &out, 4, None));
    }

    #[test]
    fn two_member_complemented_pair_collapses() {
        // ab XOR !b = 1 XOR a!b: one comp'd cube. The pairwise catalogue must
        // refuse this (flipped shared polarity); the ANF path absorbs the
        // complement into the parity slot. Regression for the old >=3 gate.
        let gates = vec![
            conj(0, &[(1, true), (2, true)]),
            conj(0, &[(2, false)]),
        ];
        let (out, rep) = compress(gates.clone(), 3, &CompressParams::default());
        assert_eq!(out.len(), 1, "pair must collapse to one complemented cube");
        assert!(out[0].comp, "the complement must land in the comp bit");
        assert!(equal_on(&gates, &out, 3, None));
        assert!(rep.anf_wins >= 1);
    }

    #[test]
    fn exact_table_beats_pairing() {
        // a XOR b XOR ab = 1 XOR !a!b: three cubes into one comp'd cube. The
        // catalogue gets stuck at two (a!b, b); the exact <=4-support table
        // finds the minimum witness.
        let gates = vec![
            conj(0, &[(1, true)]),
            conj(0, &[(2, true)]),
            conj(0, &[(1, true), (2, true)]),
        ];
        let (out, rep) = compress(gates.clone(), 3, &CompressParams::default());
        assert_eq!(out.len(), 1, "exact minimum is one complemented cube");
        assert_eq!(out[0].width(), 2);
        assert!(out[0].comp);
        assert!(equal_on(&gates, &out, 3, None));
        assert!(rep.exact_wins >= 1);
    }

    #[test]
    fn random_groups_reduce_and_preserve_function() {
        // Soak the multi-strategy reducer: random same-target groups across
        // the exact (<=4), cover/matching, and large-support paths.
        let mut rng = StdRng::seed_from_u64(0xE50);
        for case in 0..200usize {
            let pool = [3u16, 4, 6, 10][case % 4];
            let n = rng.random_range(2..=10usize);
            let mut gates: Vec<XGate> = Vec::new();
            for _ in 0..n {
                let w = rng.random_range(0..=pool.min(5) as usize);
                let mut lits: Vec<(u16, bool)> = Vec::new();
                while lits.len() < w {
                    let c = rng.random_range(1..=pool);
                    if lits.iter().all(|&(seen, _)| seen != c) {
                        lits.push((c, rng.random_bool(0.5)));
                    }
                }
                let mut g = XGate::conj(0, lits).expect("distinct literals");
                g.comp = rng.random_bool(0.3);
                gates.push(g);
            }
            let (out, _) = compress(gates.clone(), pool as usize + 1, &CompressParams::default());
            assert!(out.len() <= gates.len());
            assert!(
                equal_on(&gates, &out, pool as usize + 1, None),
                "case {case} changed the function"
            );
        }
    }

    #[test]
    fn zero_specialization_folds_and_kills() {
        // zero_in = wire 0. Gate 0 can never fire (positive literal on a zero
        // wire); gate 1 folds its always-true literal away; the comp=1 gate
        // with a dead cube still applies t ^= 1 and must DEGRADE to an X, not
        // vanish, while the comp=1 gate folding to an empty cube is a no-op
        // and must vanish; the write to wire 0 ends its known-zero status, so
        // the last gate survives untouched. Equality is only promised on the
        // zero slice.
        let dead_comp = {
            let mut g = conj(5, &[(0, true), (2, true)]);
            g.comp = true;
            g
        };
        let noop_comp = {
            let mut g = conj(6, &[(0, false)]);
            g.comp = true;
            g
        };
        let gates = vec![
            conj(3, &[(0, true), (1, true)]),
            conj(3, &[(0, false), (1, true)]),
            dead_comp,
            noop_comp,
            conj(0, &[(1, true)]),
            conj(4, &[(0, true), (2, true)]),
        ];
        let mut zero = vec![false; 7];
        zero[0] = true;
        let p = CompressParams {
            zero_in: Some(zero.clone()),
            ..Default::default()
        };
        let (out, rep) = compress(gates.clone(), 7, &p);
        assert_eq!(rep.zero_killed, 2, "dead comp=0 gate and no-op comp gate");
        assert!(rep.zero_lits_dropped >= 1);
        assert_eq!(out.len(), 4);
        assert!(out.iter().any(|g| g.target == 3 && g.width() == 1));
        assert!(out.iter().any(|g| g.target == 4 && g.width() == 2));
        assert!(
            out.iter().any(|g| g.target == 5 && g.width() == 0 && !g.comp),
            "dead comp=1 cube must leave a bare X behind"
        );
        assert!(out.iter().all(|g| g.target != 6));
        // Equal on the promised subspace: zero wires forced to 0 at entry.
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..64 {
            let sa: Vec<u64> = (0..7)
                .map(|w| if zero[w] { 0 } else { rng.random::<u64>() })
                .collect();
            let mut sb = sa.clone();
            let mut sa = sa;
            eval_lanes(gates.iter(), &mut sa);
            eval_lanes(out.iter(), &mut sb);
            assert_eq!(sa, sb, "zero-slice equality violated");
        }
    }

    #[test]
    fn downhill_collapses_crossing_ladder() {
        // t ^= bx; b ^= c; t ^= bx; t ^= cx computes just b ^= c: the trailing
        // pair is the case-split ladder of floating the first write across the
        // CNOT. Gathering alone is stuck (the CNOT pins both sides); the
        // interleaved downhill pass conjugates the ladder back and the next
        // iteration cancels everything.
        let gates = vec![
            conj(0, &[(1, true), (3, true)]),
            conj(1, &[(2, true)]),
            conj(0, &[(1, true), (3, true)]),
            conj(0, &[(2, true), (3, true)]),
        ];
        let (out, rep) = compress(gates.clone(), 4, &CompressParams::default());
        assert_eq!(out.len(), 1, "everything but the CNOT must vanish");
        assert_eq!(out[0], conj(1, &[(2, true)]));
        // With the defaults the reverse-gather transport folds the ladder
        // before downhill runs; either conjugation route counts.
        assert!(rep.downhill_swaps + rep.transports >= 1);
        assert!(equal_on(&gates, &out, 4, None));
        // With downhill AND in-gather transport disabled the ladder must
        // survive: the win comes from conjugation (either the adjacent
        // downhill pass or the reverse-gather transport), not from plain
        // gathering. Legacy gathering plus downhill alone still wins.
        let (out2, _) = compress(gates.clone(), 4, &legacy());
        assert_eq!(out2.len(), 4);
        let p = CompressParams {
            downhill: true,
            ..legacy()
        };
        let (out3, rep3) = compress(gates.clone(), 4, &p);
        assert_eq!(out3.len(), 1);
        assert!(rep3.downhill_swaps >= 1);
    }

    #[test]
    fn liveness_prunes_dead_cones() {
        // Wire 0 live, wire 1 dead. Last write to 1 is deletable; the earlier
        // write to 1 feeds the live gate through 1 and must stay.
        let gates = vec![
            conj(1, &[(2, true)]), // stays: 1 read below by live gate
            conj(0, &[(1, true)]), // live
            conj(1, &[(0, true)]), // dead: nothing live reads 1 after
        ];
        let live = vec![true, false, true];
        let p = CompressParams {
            live_out: Some(live.clone()),
            ..Default::default()
        };
        let (out, rep) = compress(gates.clone(), 3, &p);
        assert_eq!(rep.liveness_dropped, 1);
        assert!(equal_on(&gates, &out, 3, Some(&live)));
    }

    #[test]
    fn random_circuit_compresses_and_preserves_function() {
        let mut rng = StdRng::seed_from_u64(5);
        let wires = 8u16;
        let mut gates: Vec<XGate> = Vec::new();
        while gates.len() < 400 {
            let t = rng.random_range(0..wires);
            let w = rng.random_range(0..=3usize);
            let lits: Vec<(u16, bool)> = (0..w)
                .map(|_| {
                    let mut c = rng.random_range(0..wires);
                    while c == t {
                        c = rng.random_range(0..wires);
                    }
                    (c, rng.random_bool(0.5))
                })
                .collect();
            if let Some(mut g) = XGate::conj(t, lits) {
                g.comp = rng.random_bool(0.2);
                gates.push(g);
            }
        }
        let (out, rep) = compress(gates.clone(), wires as usize, &CompressParams::default());
        assert!(out.len() <= gates.len());
        assert!(rep.iters >= 1);
        assert!(equal_on(&gates, &out, wires as usize, None));
        // On a dense 8-wire circuit real reduction should happen.
        assert!(
            out.len() < gates.len(),
            "expected some compression on dense circuit"
        );
    }

    // Gather-only parameters: the legacy rule set (no transport, no
    // separation passes, no reverse pass, no downhill).
    fn legacy() -> CompressParams {
        CompressParams {
            downhill: false,
            transport: false,
            sep_reads: false,
            reverse_pass: false,
            ..Default::default()
        }
    }

    #[test]
    fn toffoli_sliding_pair_cancels() {
        // The family-B triple from the K2 final (gates 837-839): a control
        // flipped under the other controls slides through with a polarity
        // change, so the two copies are one identity. Legacy gathering
        // cannot form the group (the flip writes a control); transport
        // conjugates the first copy across the flip and the pair cancels.
        let g = conj(3, &[(0, false), (1, true), (2, true)]);
        let flip = conj(1, &[(0, false), (2, true)]);
        let g2 = conj(3, &[(0, false), (1, false), (2, true)]);
        let gates = vec![g, flip.clone(), g2];
        let (out, rep) = compress(gates.clone(), 4, &CompressParams::default());
        assert_eq!(out, vec![flip], "only the flip survives");
        assert!(rep.transports >= 1);
        assert!(equal_on(&gates, &out, 4, None));
        let (legacy_out, _) = compress(gates.clone(), 4, &legacy());
        assert_eq!(legacy_out.len(), 3, "legacy gathering is blocked by the flip");
    }

    #[test]
    fn toffoli_sliding_at_distance_and_through_reads() {
        // Same relation, copies seven gates apart, with the flipped control
        // READ in between (which closes the flip's own group but must not
        // close the transported group: it depends on the flip's group, and a
        // dependency emitted early is fine). Complemented flip too.
        let g = conj(3, &[(0, false), (1, true), (2, true)]);
        let mut flip = conj(1, &[(0, false), (2, true)]);
        flip.comp = true; // u ^= NOT(!w0 & w2): flips u where g's cube holds
        // g[u] under u <- u ^ 1 ^ (!w0&w2) = g with u kept (comp adds a flip
        // everywhere, the cube undoes it where g fires): copy reads u
        // positively... work it out: on g's cube the flip fires 0 -> u
        // unchanged. So the identical copy is the identity here.
        let g2 = conj(3, &[(0, false), (1, true), (2, true)]);
        // Fillers read u and each carries its own private literal, so they
        // neither merge with each other nor touch g.
        let filler: Vec<XGate> = (0..6)
            .map(|k| conj(4 + (k % 3), &[(1, k % 2 == 0), (8 + k as u16, true)]))
            .collect();
        let mut gates = vec![g, flip.clone()];
        gates.extend(filler.iter().cloned());
        gates.push(g2);
        let (out, rep) = compress(gates.clone(), 14, &CompressParams::default());
        assert_eq!(out.len(), 7, "the pair cancels through six fillers: {out:?}");
        // The complemented flip fires 0 on g's cube, so the conjugated ESOP
        // is g itself: a no-op transport (no frame dependency needed).
        assert!(rep.transports + rep.transport_noops >= 1);
        assert!(equal_on(&gates, &out, 14, None));
    }

    #[test]
    fn transport_frame_order_respects_dependencies() {
        // g floats across the flip (dependency on the flip's group); a read
        // of t then closes g's group BEFORE the flip's group would close on
        // its own, and another writer of u arrives afterwards. The flip must
        // be emitted before the transported cube and the later writer must
        // not be gathered across it.
        let g = conj(3, &[(0, false), (1, true), (2, true)]);
        let flip = conj(1, &[(0, false), (2, true)]);
        let read_t = conj(5, &[(3, true)]);
        let later_u = conj(1, &[(6, true)]);
        let g2 = conj(3, &[(0, false), (1, false), (2, true)]);
        let gates = vec![g, flip, read_t, later_u, g2];
        let (out, _) = compress(gates.clone(), 8, &CompressParams::default());
        assert!(equal_on(&gates, &out, 8, None));
        // And the same with the flip's group forced closed by a read of u in
        // between, so the transported group outlives its dependency.
        let read_u = conj(5, &[(1, true)]);
        let g = conj(3, &[(0, false), (1, true), (2, true)]);
        let flip = conj(1, &[(0, false), (2, true)]);
        let g2 = conj(3, &[(0, false), (1, false), (2, true)]);
        let gates = vec![g, flip, read_u, g2];
        let (out, rep) = compress(gates.clone(), 8, &CompressParams::default());
        assert!(equal_on(&gates, &out, 8, None));
        assert_eq!(out.len(), 2, "pair cancels across the read of u: {out:?}");
        assert!(rep.transports >= 1);
    }

    #[test]
    fn separated_reader_does_not_close_group() {
        // r reads t but is separated from both copies on wire 2 (opposite
        // polarity), so it commutes with them and the copies cancel.
        let g = conj(0, &[(1, true), (2, true)]);
        let r = conj(3, &[(0, true), (2, false)]);
        let gates = vec![g.clone(), r.clone(), g.clone()];
        let (out, rep) = compress(gates.clone(), 4, &CompressParams::default());
        assert_eq!(out, vec![r], "separated reader passes, pair cancels");
        assert!(rep.sep_passes >= 1);
        assert!(equal_on(&gates, &out, 4, None));
        let (legacy_out, _) = compress(gates.clone(), 4, &legacy());
        assert_eq!(legacy_out.len(), 3);
    }

    #[test]
    fn reverse_pass_folds_leftward_ladder() {
        // h, then (three gates later) the case-split ladder {t^=L&u, t^=L&M}
        // left by crossing t^=L&u leftward over h (u ^= M). Downhill sees
        // only immediate neighbours and the forward gather floats the ladder
        // away from h; the reversed gather floats it back onto h and the
        // transport folds it to one cube.
        let h = conj(1, &[(4, true)]);
        let fill: Vec<XGate> = vec![
            conj(5, &[(6, true)]),
            conj(6, &[(7, false)]),
            conj(7, &[(5, true)]),
        ];
        let ladder = vec![
            conj(0, &[(1, true), (2, true)]),
            conj(0, &[(2, true), (4, true)]),
        ];
        let mut gates = vec![h];
        gates.extend(fill);
        gates.extend(ladder);
        let (out, rep) = compress(gates.clone(), 8, &CompressParams::default());
        assert_eq!(out.len(), 5, "ladder folds to one cube: {out:?}");
        assert!(rep.transports >= 1);
        assert!(equal_on(&gates, &out, 8, None));
        let p = CompressParams {
            reverse_pass: false,
            ..Default::default()
        };
        let (fwd_only, _) = compress(gates.clone(), 8, &p);
        assert_eq!(fwd_only.len(), 6, "forward-only gathering cannot reach h");
    }

    #[test]
    fn dense_random_soak_all_rules() {
        // Dense circuits on few wires exercise transports, dependency
        // cascades, separated passes and the reverse pass together; the
        // pass must preserve the function and never grow the circuit.
        let mut rng = StdRng::seed_from_u64(0xD0_5EED);
        for case in 0..120usize {
            let wires: u16 = [4, 5, 6, 8][case % 4];
            let n = 40 + (case * 7) % 260;
            let mut gates: Vec<XGate> = Vec::new();
            while gates.len() < n {
                let t = rng.random_range(0..wires);
                let w = rng.random_range(0..=3usize);
                let mut lits: Vec<(u16, bool)> = Vec::new();
                while lits.len() < w {
                    let c = rng.random_range(0..wires);
                    if c != t && lits.iter().all(|&(seen, _)| seen != c) {
                        lits.push((c, rng.random_bool(0.5)));
                    }
                }
                let mut g = XGate::conj(t, lits).expect("distinct literals");
                g.comp = rng.random_bool(0.25);
                gates.push(g);
            }
            for (label, p) in [
                ("all", CompressParams::default()),
                (
                    "slack1",
                    CompressParams {
                        transport_slack: 1,
                        ..Default::default()
                    },
                ),
                ("legacy", legacy()),
            ] {
                let (out, _) = compress(gates.clone(), wires as usize, &p);
                assert!(
                    out.len() <= gates.len(),
                    "case {case} {label} grew the circuit"
                );
                assert!(
                    equal_on(&gates, &out, wires as usize, None),
                    "case {case} {label} changed the function"
                );
            }
        }
    }

    #[test]
    fn pack_is_exact_canonical_and_unbounded() {
        use crate::engine::format::{expand_packed, read_anf1, read_mpmct, write_anf1};
        // Exactness on random compressed circuits, through the file format.
        let mut rng = StdRng::seed_from_u64(0x9AC7);
        for case in 0..40usize {
            let wires: u16 = [5, 8, 12][case % 3];
            let mut gates: Vec<XGate> = Vec::new();
            while gates.len() < 150 {
                let t = rng.random_range(0..wires);
                let w = rng.random_range(0..=3usize);
                let mut lits: Vec<(u16, bool)> = Vec::new();
                while lits.len() < w {
                    let c = rng.random_range(0..wires);
                    if c != t && lits.iter().all(|&(seen, _)| seen != c) {
                        lits.push((c, rng.random_bool(0.5)));
                    }
                }
                let mut g = XGate::conj(t, lits).expect("distinct literals");
                g.comp = rng.random_bool(0.3);
                gates.push(g);
            }
            let (out, _) = compress(gates.clone(), wires as usize, &CompressParams::default());
            let packed = pack(&out);
            assert!(packed.len() <= out.len());
            assert!(equal_on(&gates, &expand_packed(&packed), wires as usize, None), "case {case}");
            if case == 0 {
                let path = std::env::temp_dir().join(format!("fcompress_pack_test_{}.anf1", std::process::id()));
                let path = path.to_str().unwrap().to_string();
                write_anf1(&path, &packed, wires as usize).unwrap();
                let (back, w) = read_anf1(&path).unwrap();
                assert_eq!(back, packed, "anf1 round trip");
                assert_eq!(w, wires as usize);
                let (via_mpmct, _) = read_mpmct(&path).unwrap();
                assert_eq!(via_mpmct, expand_packed(&packed), "read_mpmct dispatches on the anf1 header");
                std::fs::remove_file(&path).ok();
            }
        }
        // Canonical: two spellings of one function pack identically, cubes
        // with negative literals expand, comp bits become the constant.
        let a = vec![conj(0, &[(1, true), (2, true)]), conj(0, &[(1, true), (2, false)])];
        let b = vec![conj(0, &[(1, true)])];
        assert_eq!(pack(&a), pack(&b));
        let mut c = conj(0, &[(1, false), (2, false)]);
        c.comp = true; // 1 ^ (1^a)(1^b) = a ^ b ^ ab
        let pc = pack(&[c]);
        assert_eq!(
            pc[0].terms,
            vec![vec![(1u16, true)], vec![(2u16, true)], vec![(1u16, true), (2u16, true)]]
        );
        // Unbounded support: a run over 80 wires packs in one gate.
        let wide: Vec<XGate> = (1u16..=80).map(|w| conj(0, &[(w, true), (w + 100, false)])).collect();
        let pw = pack(&wide);
        assert_eq!(pw.len(), 1);
        assert_eq!(pw[0].terms.len(), 160);
        assert!(equal_on(&wide, &expand_packed(&pw), 181, None));
    }

    #[test]
    fn compaction_is_exact_smaller_and_a_function_of_the_anf() {
        use crate::engine::format::{expand_packed, read_mpmct, write_esop1};
        // Two spellings of one function -> one ANF -> one compacted gate.
        let a = vec![conj(0, &[(1, false), (2, false), (3, true)])];
        let b: Vec<XGate> = a[0]
            .ctrls
            .iter()
            .map(|_| ())
            .take(0)
            .map(|_| a[0].clone())
            .collect::<Vec<_>>();
        drop(b);
        let pa = pack(&a);
        assert_eq!(pa[0].terms.len(), 4, "!x!y z = z + xz + yz + xyz");
        let ca = compact(&pa);
        assert_eq!(ca.len(), 1);
        assert!(ca[0].terms.len() <= pa[0].terms.len());
        assert!(equal_on(&a, &expand_packed(&ca), 4, None));
        // The same function spelled as its four monomials compacts identically.
        let mono: Vec<XGate> = vec![
            conj(0, &[(3, true)]),
            conj(0, &[(1, true), (3, true)]),
            conj(0, &[(2, true), (3, true)]),
            conj(0, &[(1, true), (2, true), (3, true)]),
        ];
        assert_eq!(compact(&pack(&mono)), ca);
        // Exactness + non-growth on random compressed circuits, through esop1.
        let mut rng = StdRng::seed_from_u64(0xE50_9);
        for case in 0..40usize {
            let wires: u16 = [5, 8, 12][case % 3];
            let mut gates: Vec<XGate> = Vec::new();
            while gates.len() < 150 {
                let t = rng.random_range(0..wires);
                let w = rng.random_range(0..=3usize);
                let mut lits: Vec<(u16, bool)> = Vec::new();
                while lits.len() < w {
                    let c = rng.random_range(0..wires);
                    if c != t && lits.iter().all(|&(seen, _)| seen != c) {
                        lits.push((c, rng.random_bool(0.5)));
                    }
                }
                let mut g = XGate::conj(t, lits).expect("distinct literals");
                g.comp = rng.random_bool(0.3);
                gates.push(g);
            }
            let (out, _) = compress(gates.clone(), wires as usize, &CompressParams::default());
            let anf = pack(&out);
            let esop = compact(&anf);
            let (ta, te): (usize, usize) = (
                anf.iter().map(|g| g.terms.len()).sum(),
                esop.iter().map(|g| g.terms.len()).sum(),
            );
            assert!(te <= ta, "case {case}: compaction grew {ta} -> {te}");
            assert!(equal_on(&gates, &expand_packed(&esop), wires as usize, None), "case {case}");
            if case == 1 {
                let path = std::env::temp_dir().join(format!("fcompress_esop_test_{}.esop1", std::process::id()));
                let path = path.to_str().unwrap().to_string();
                write_esop1(&path, &esop, wires as usize).unwrap();
                let (via_mpmct, _) = read_mpmct(&path).unwrap();
                assert_eq!(via_mpmct, expand_packed(&esop));
                std::fs::remove_file(&path).ok();
            }
        }
    }
}
