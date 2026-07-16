// Final compression pass for post-mix circuits (fcompress).
//
// The XGate calculus is closed under XOR on a shared target: a run of
// same-target gates composes to t ^= f1 XOR f2 XOR ... XOR fk, an ESOP
// (XOR of mixed-polarity cubes) whose canonical form is ANF. This pass
// gathers same-target gates that can float to a common point, reduces the
// gathered cube set (pairwise catalogue to fixed point, then an ANF
// rewrite when it wins), and re-emits the survivors as consecutive XGates,
// so the output stays in mpmct1 and downstream tooling keeps working.
// Deterministic and attacker-computable, so it never weakens hiding; the
// compressed size doubles as the honest "effective size" of a mixed
// circuit.
//
// Gathering is one forward sweep with an open group per target wire:
//   - any read of wire t closes t's group (a reader pins the accumulated
//     value; writes may not cross it),
//   - any write to a wire in a group's union-of-member-controls closes
//     that group (members' control values may not change),
//   - members then float right to the close point, which the two rules
//     make unconditionally legal, and are emitted there.
// Closures cascade (an emitted group writes its target, which may poison
// other groups) and are emitted in ascending last-member order; the two
// closure rules make that order respect every read/write constraint.
//
// Optional output-cone pruning for gadgetized circuits (equality required
// only on designated live wires): one exact backward pass in the
// XOR-accumulate model — a gate is deletable iff its target is dead at
// its position; a kept gate makes its controls live, and its target STAYS
// live (XOR never overwrites).
use super::lineage::{GroupKind, PathKind, ProvId, ProvenanceArena, ResolvedCoverage};
use super::mix::{Merge, merge_result};
use super::reassemble::{analyze_barrier_free, is_structural_g57, plain_pair_to_g57};
use super::source::{SourceClassCounts, UNKNOWN_SOURCE, merge_source, merge_sources};
use super::xgate::{Lits, XGate};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::HashMap;

pub struct CompressParams {
    // None = equality required on every wire. Some(mask) = only on wires
    // with mask[w] == true; dead cones are pruned.
    pub live_out: Option<Vec<bool>>,
    pub max_iters: usize,
    // Groups are closed proactively at this many members.
    pub group_cap: usize,
    // ANF rewrite attempted only when the group support fits (mask bits).
    pub anf_support_cap: usize,
    pub local_verify: bool,
    pub seed: u64,
}

impl Default for CompressParams {
    fn default() -> CompressParams {
        CompressParams {
            live_out: None,
            max_iters: 256,
            group_cap: 64,
            anf_support_cap: 24,
            local_verify: true,
            seed: 0,
        }
    }
}

#[derive(Default, Debug)]
pub struct CompressReport {
    pub iters: usize,
    pub liveness_dropped: usize,
    pub groups: u64,
    pub multi_groups: u64,
    pub max_group: usize,
    pub catalogue_merges: u64,
    /// Legacy direct-pair metric: exact plain 1cc+2cc pairs fused inside
    /// barrier-free groups on the first compression pass only.
    pub reassembled_pairs: u64,
    pub reassembled_fragments: u64,
    /// Direct pair fusions over every compression pass.  `*_later` is the
    /// subset that became eligible only after at least one earlier sweep (for
    /// example, a 3cc sibling pair first catalogue-merging to a 2cc).
    pub direct_pairs_total: u64,
    pub direct_pairs_later: u64,
    pub direct_fragments_total: u64,
    /// Parent-source classification of every direct structural-g57 fusion.
    pub direct_pass1_sources: SourceClassCounts,
    pub direct_later_sources: SourceClassCounts,
    pub anf_wins: u64,
    /// Structural g57 outputs produced by ANF-winning groups.  Fragment
    /// participation in these outputs is group-attributed, not one-to-one.
    pub anf_structural_g57: u64,
    pub anf_sources: SourceClassCounts,
    /// Structural g57 outputs emitted by reduced groups with at least one
    /// loose-input fragment in their provenance.
    pub esop_structural_g57: u64,
    pub esop_sources: SourceClassCounts,
    pub gates_in: usize,
    pub gates_out: usize,
    pub lits_in: u64,
    pub lits_out: u64,
    /// True only when a sweep made no gate or liveness change.  A false value
    /// means `max_iters` was exhausted and the result is not a fixed point.
    pub reached_fixed_point: bool,
}

/// Provenance roots of each recovery route.  Categories intentionally overlap;
/// [`RecoverySummary`] deduplicates the original input fragments within every
/// reported union.
#[derive(Clone, Debug, Default)]
pub struct RecoveryEvents {
    pub direct_pass1: Vec<ProvId>,
    pub direct_later: Vec<ProvId>,
    pub anf: Vec<ProvId>,
    pub esop_structural: Vec<ProvId>,
    pub database: Vec<ProvId>,
}

/// Exact lower-bound and inclusive group-attributed fragment coverage.
#[derive(Clone, Debug, Default)]
pub struct RecoverySummary {
    pub input_plain_fragments: usize,
    pub input_width_histogram: Vec<u64>,
    pub direct_pass1: ResolvedCoverage,
    pub direct_later: ResolvedCoverage,
    pub anf: ResolvedCoverage,
    pub database: ResolvedCoverage,
    pub ever: ResolvedCoverage,
    pub final_structural_g57: ResolvedCoverage,
}

/// A compressed tape plus its in-memory provenance sidecar.  The sidecar is
/// used by the optional frozen-database stage and is never written into mpmct1.
#[derive(Clone, Debug)]
pub struct TracedCircuit {
    pub gates: Vec<XGate>,
    pub roots: Vec<ProvId>,
    /// Parent-source mark aligned with `gates`/`roots`.
    pub source_marks: Vec<u32>,
    /// Original pre-split gates indexed by pristine source marks.
    pub source_parents: Vec<XGate>,
    pub provenance: ProvenanceArena,
    pub recovery: RecoveryEvents,
}

impl TracedCircuit {
    pub fn recovery_summary(&self) -> RecoverySummary {
        debug_assert_eq!(self.gates.len(), self.roots.len());
        let direct_pass1 = self
            .provenance
            .resolve(self.recovery.direct_pass1.iter().copied());
        let direct_later = self
            .provenance
            .resolve(self.recovery.direct_later.iter().copied());
        let anf = self.provenance.resolve(self.recovery.anf.iter().copied());
        let database = self
            .provenance
            .resolve(self.recovery.database.iter().copied());

        let ever_roots = self
            .recovery
            .direct_pass1
            .iter()
            .chain(&self.recovery.direct_later)
            .chain(&self.recovery.anf)
            .chain(&self.recovery.esop_structural)
            .chain(&self.recovery.database)
            .copied();
        let ever = self.provenance.resolve(ever_roots);
        let final_roots = self
            .gates
            .iter()
            .zip(&self.roots)
            .filter_map(|(gate, &root)| is_structural_g57(gate).then_some(root));
        let final_structural_g57 = self.provenance.resolve(final_roots);

        RecoverySummary {
            input_plain_fragments: self.provenance.input_plain_fragments(),
            input_width_histogram: self.provenance.input_width_histogram().to_vec(),
            direct_pass1,
            direct_later,
            anf,
            database,
            ever,
            final_structural_g57,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TrackedGate {
    gate: XGate,
    root: ProvId,
    source: u32,
}

pub fn lits_of(gates: &[XGate]) -> u64 {
    gates.iter().map(|g| g.width() as u64).sum()
}

// Exact dead-cone elimination: keep a gate iff its target is live at its
// position; kept gates make their controls live. Returns dropped count.
pub fn liveness_prune(gates: Vec<XGate>, live_out: &[bool]) -> (Vec<XGate>, usize) {
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
    let mut out = Vec::with_capacity(n);
    let mut dropped = 0usize;
    for (i, g) in gates.into_iter().enumerate() {
        if keep[i] {
            out.push(g);
        } else {
            dropped += 1;
        }
    }
    (out, dropped)
}

fn liveness_prune_tracked(gates: Vec<TrackedGate>, live_out: &[bool]) -> (Vec<TrackedGate>, usize) {
    let n = gates.len();
    let mut live = live_out.to_vec();
    let mut keep = vec![false; n];
    for i in (0..n).rev() {
        let gate = &gates[i].gate;
        if live[gate.target as usize] {
            keep[i] = true;
            for &(wire, _) in &gate.ctrls {
                live[wire as usize] = true;
            }
        }
    }
    let mut out = Vec::with_capacity(n);
    let mut dropped = 0usize;
    for (i, gate) in gates.into_iter().enumerate() {
        if keep[i] {
            out.push(gate);
        } else {
            dropped += 1;
        }
    }
    (out, dropped)
}

// XOR of the gathered control functions as (parity, cubes): comp gates
// contribute 1 XOR cube. Evaluate one assignment (support-indexed bools).
fn eval_cubes(parity: bool, cubes: &[Lits], val: &dyn Fn(u16) -> bool) -> bool {
    let mut acc = parity;
    for c in cubes {
        acc ^= c.iter().all(|&(w, p)| val(w) == p);
    }
    acc
}

// ANF rewrite of a cube set over its support: expand mixed-polarity cubes
// into positive monomials (canonical, so all cancellation happens), then
// greedily re-pair monomials differing in one wire back into single-negation
// cubes. Returns (cubes, parity_delta) or None when the support or the
// expansion would be too large.
fn anf_reduce(cubes: &[Lits], support_cap: usize) -> Option<(Vec<Lits>, bool)> {
    let mut support: Vec<u16> = cubes
        .iter()
        .flat_map(|c| c.iter().map(|&(w, _)| w))
        .collect();
    support.sort_unstable();
    support.dedup();
    if support.len() > support_cap.min(31) {
        return None;
    }
    let idx_of: HashMap<u16, u32> = support
        .iter()
        .enumerate()
        .map(|(i, &w)| (w, i as u32))
        .collect();
    let mut budget = 1u64 << 17;
    let mut anf: HashMap<u32, bool> = HashMap::new();
    for c in cubes {
        let (mut pos, mut neg) = (0u32, 0u32);
        for &(w, p) in c {
            let b = 1u32 << idx_of[&w];
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
            let e = anf.entry(pos | sub).or_insert(false);
            *e = !*e;
            if sub == 0 {
                break;
            }
            sub = (sub - 1) & neg;
        }
    }
    let mut monos: Vec<u32> = anf
        .into_iter()
        .filter(|&(_, on)| on)
        .map(|(m, _)| m)
        .collect();
    let parity_delta = monos.iter().any(|&m| m == 0);
    monos.retain(|&m| m != 0);
    monos.sort_unstable_by_key(|&m| std::cmp::Reverse((m.count_ones(), m)));
    let pos_of: HashMap<u32, usize> = monos.iter().enumerate().map(|(i, &m)| (m, i)).collect();
    let mut matched = vec![false; monos.len()];
    let mut out: Vec<Lits> = Vec::new();
    let to_lits = |pos: u32, neg: u32| -> Lits {
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
    for i in 0..monos.len() {
        if matched[i] {
            continue;
        }
        let m = monos[i];
        let mut bits = m;
        let mut paired = false;
        while bits != 0 {
            let b = bits & bits.wrapping_neg();
            bits &= bits - 1;
            if let Some(&j) = pos_of.get(&(m & !b)) {
                if !matched[j] && j != i {
                    // mono(m\b) XOR mono(m) = (m\b) AND NOT b
                    matched[i] = true;
                    matched[j] = true;
                    out.push(to_lits(m & !b, b));
                    paired = true;
                    break;
                }
            }
        }
        if !paired {
            matched[i] = true;
            out.push(to_lits(m, 0));
        }
    }
    Some((out, parity_delta))
}

#[derive(Clone, Debug)]
struct TrackedCube {
    ctrls: Lits,
    root: ProvId,
    source: u32,
}

// Apply the exact maximum direct 1cc+2cc matching while preserving the
// concrete pair indices, so every fused g57 receives an exact ancestry root.
fn fuse_tracked_pairs(
    members: &[TrackedGate],
    iteration: usize,
    source_parents: &[XGate],
    provenance: &mut ProvenanceArena,
    recovery: &mut RecoveryEvents,
    rep: &mut CompressReport,
) -> Vec<TrackedGate> {
    let gates: Vec<XGate> = members.iter().map(|member| member.gate.clone()).collect();
    let analysis = analyze_barrier_free(&gates);
    if analysis.pairs.is_empty() {
        return members.to_vec();
    }

    let mut consumed = vec![false; members.len()];
    let mut fused_at: Vec<Option<TrackedGate>> = vec![None; members.len()];
    let path = if iteration == 1 {
        PathKind::DirectPass1
    } else {
        PathKind::DirectLater
    };
    for pair in &analysis.pairs {
        debug_assert!(!consumed[pair.one_cc] && !consumed[pair.two_cc]);
        let gate = plain_pair_to_g57(&members[pair.one_cc].gate, &members[pair.two_cc].gate)
            .expect("reassembly matching produced an invalid pair");
        let root =
            provenance.exact_union(members[pair.one_cc].root, members[pair.two_cc].root, path);
        let source = merge_source(members[pair.one_cc].source, members[pair.two_cc].source);
        if iteration == 1 {
            rep.reassembled_pairs += 1;
            rep.reassembled_fragments += 2;
            recovery.direct_pass1.push(root);
            rep.direct_pass1_sources
                .record(source, &gate, source_parents);
        } else {
            rep.direct_pairs_later += 1;
            recovery.direct_later.push(root);
            rep.direct_later_sources
                .record(source, &gate, source_parents);
        }
        rep.direct_pairs_total += 1;
        rep.direct_fragments_total += 2;

        let at = pair.one_cc.min(pair.two_cc);
        consumed[pair.one_cc] = true;
        consumed[pair.two_cc] = true;
        fused_at[at] = Some(TrackedGate { gate, root, source });
    }

    let mut out = Vec::with_capacity(members.len() - analysis.pairs.len());
    for (index, member) in members.iter().enumerate() {
        if let Some(fused) = fused_at[index].take() {
            out.push(fused);
        } else if !consumed[index] {
            out.push(member.clone());
        }
    }
    out
}

// Reduce one gathered group to a minimal-found cube list and emit as XGates
// (parity absorbed as comp on the first cube, or an X gate if none remain).
fn reduce_group(
    target: u16,
    members: &[TrackedGate],
    iteration: usize,
    p: &CompressParams,
    source_parents: &[XGate],
    rng: &mut StdRng,
    provenance: &mut ProvenanceArena,
    recovery: &mut RecoveryEvents,
    rep: &mut CompressReport,
) -> Vec<TrackedGate> {
    rep.groups += 1;
    rep.max_group = rep.max_group.max(members.len());
    if members.len() == 1 {
        return members.to_vec();
    }
    rep.multi_groups += 1;

    // Same-target members gathered into this group commute and are therefore
    // a barrier-free region.  Recover the exact maximum number of g57s before
    // the general ESOP catalogue: mix::merge_result intentionally forbids
    // this presplit rejoin so fmix remains irreversible, while fcompress is
    // specifically allowed to perform it here.
    let reassembled = fuse_tracked_pairs(
        members,
        iteration,
        source_parents,
        provenance,
        recovery,
        rep,
    );
    let mut parity = false;
    let mut parity_root = ProvId::EMPTY;
    let mut parity_source: Option<u32> = None;
    let mut cubes: Vec<TrackedCube> = Vec::with_capacity(reassembled.len());
    for m in &reassembled {
        debug_assert_eq!(m.gate.target, target);
        if m.gate.comp {
            parity = !parity;
            parity_root = provenance.exact_union(parity_root, m.root, PathKind::Parity);
            parity_source = Some(match parity_source {
                Some(source) => merge_source(source, m.source),
                None => m.source,
            });
        }
        cubes.push(TrackedCube {
            ctrls: m.gate.ctrls.clone(),
            root: m.root,
            source: m.source,
        });
    }
    // Pairwise catalogue (cancel / drop-literal / subsume) to a fixed point.
    'outer: loop {
        for i in 0..cubes.len() {
            for j in i + 1..cubes.len() {
                let a = XGate {
                    target,
                    comp: false,
                    ctrls: cubes[i].ctrls.clone(),
                };
                let b = XGate {
                    target,
                    comp: false,
                    ctrls: cubes[j].ctrls.clone(),
                };
                if let Some(m) = merge_result(&a, &b) {
                    rep.catalogue_merges += 1;
                    let repl = match m {
                        Merge::Cancel => None,
                        Merge::DropLit(g) | Merge::Subsume(g) | Merge::XFuse(g) => Some(g.ctrls),
                    };
                    let merged_root =
                        provenance.exact_union(cubes[i].root, cubes[j].root, PathKind::Catalogue);
                    let merged_source = merge_source(cubes[i].source, cubes[j].source);
                    cubes.swap_remove(j);
                    match repl {
                        Some(ctrls) => {
                            cubes[i] = TrackedCube {
                                ctrls,
                                root: merged_root,
                                source: merged_source,
                            }
                        }
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
    let mut anf_applied = false;
    if cubes.len() >= 3 {
        let cube_lits: Vec<Lits> = cubes.iter().map(|cube| cube.ctrls.clone()).collect();
        if let Some((alt, delta)) = anf_reduce(&cube_lits, p.anf_support_cap) {
            let (alt_l, cur_l) = (
                alt.iter().map(|c| c.len()).sum::<usize>(),
                cubes.iter().map(|c| c.ctrls.len()).sum::<usize>(),
            );
            if (alt.len(), alt_l) < (cubes.len(), cur_l) {
                rep.anf_wins += 1;
                anf_applied = true;
                let group_root =
                    provenance.group_union(cubes.iter().map(|cube| cube.root), GroupKind::Anf);
                let group_source = merge_sources(cubes.iter().map(|cube| cube.source));
                cubes = alt
                    .into_iter()
                    .map(|ctrls| TrackedCube {
                        ctrls,
                        root: group_root,
                        source: group_source,
                    })
                    .collect();
                parity ^= delta;
                if delta {
                    // The constant term was created by the ANF rewrite, so its
                    // attribution must cross the same non-canonical group
                    // boundary even when ANF emits no cubes at all.
                    parity_root = provenance.exact_union(parity_root, group_root, PathKind::Parity);
                }
            }
        }
    }
    let mut out: Vec<TrackedGate> = Vec::with_capacity(cubes.len().max(1));
    for (index, mut cube) in cubes.into_iter().enumerate() {
        let comp = parity && index == 0;
        if comp {
            cube.root = provenance.exact_union(cube.root, parity_root, PathKind::Parity);
            if let Some(source) = parity_source {
                cube.source = merge_source(cube.source, source);
            }
        }
        out.push(TrackedGate {
            gate: XGate {
                target,
                comp,
                ctrls: cube.ctrls,
            },
            root: cube.root,
            source: cube.source,
        });
    }
    if out.is_empty() && parity {
        out.push(TrackedGate {
            gate: XGate::x_gate(target),
            root: parity_root,
            source: parity_source.unwrap_or(UNKNOWN_SOURCE),
        });
    }
    if p.local_verify {
        let before: Vec<XGate> = members.iter().map(|member| member.gate.clone()).collect();
        let after: Vec<XGate> = out.iter().map(|member| member.gate.clone()).collect();
        verify_group(&before, &after, rng);
    }
    for member in &out {
        if is_structural_g57(&member.gate) && !member.root.is_empty() {
            rep.esop_structural_g57 += 1;
            recovery.esop_structural.push(member.root);
            rep.esop_sources
                .record(member.source, &member.gate, source_parents);
            if anf_applied {
                rep.anf_structural_g57 += 1;
                recovery.anf.push(member.root);
                rep.anf_sources
                    .record(member.source, &member.gate, source_parents);
            }
        }
    }
    out
}

fn verify_group(before: &[XGate], after: &[XGate], rng: &mut StdRng) {
    let mut support: Vec<u16> = before
        .iter()
        .chain(after)
        .flat_map(|g| g.ctrls.iter().map(|&(w, _)| w))
        .collect();
    support.sort_unstable();
    support.dedup();
    let parity_of = |gates: &[XGate], val: &dyn Fn(u16) -> bool| -> bool {
        let mut acc = false;
        for g in gates {
            acc ^= g.comp ^ g.ctrls.iter().all(|&(w, p)| val(w) == p);
        }
        acc
    };
    let check = |val: &dyn Fn(u16) -> bool| {
        assert_eq!(
            parity_of(before, val),
            parity_of(after, val),
            "fcompress group reduction changed the function"
        );
    };
    if support.len() <= 16 {
        let pos: HashMap<u16, usize> = support.iter().enumerate().map(|(i, &w)| (w, i)).collect();
        for a in 0u32..(1u32 << support.len()) {
            check(&|w| a >> pos[&w] & 1 == 1);
        }
    } else {
        for _ in 0..512 {
            let assign: HashMap<u16, bool> =
                support.iter().map(|&w| (w, rng.random_bool(0.5))).collect();
            check(&|w| assign[&w]);
        }
    }
}

struct Group {
    target: u16,
    members: Vec<TrackedGate>,
    union: Vec<u16>, // sorted control-wire union across members
    last: usize,     // index of the last member in the input order
    open: bool,
}

// One forward gather-and-reduce sweep.
fn gather_reduce_pass(
    gates: &[TrackedGate],
    iteration: usize,
    wires: usize,
    p: &CompressParams,
    source_parents: &[XGate],
    rng: &mut StdRng,
    provenance: &mut ProvenanceArena,
    recovery: &mut RecoveryEvents,
    rep: &mut CompressReport,
) -> Vec<TrackedGate> {
    let mut out: Vec<TrackedGate> = Vec::with_capacity(gates.len());
    let mut slots: Vec<Group> = Vec::new();
    let mut open_at: Vec<Option<usize>> = vec![None; wires]; // target wire -> slot
    let mut union_of: Vec<Vec<usize>> = vec![Vec::new(); wires]; // wire -> slots (stale ok)

    // Close a seed set of groups: cascade poison from emitted writes, order
    // by last-member index (provably conflict-free), reduce, emit.
    let close = |seed: Vec<usize>,
                 slots: &mut Vec<Group>,
                 open_at: &mut Vec<Option<usize>>,
                 union_of: &mut Vec<Vec<usize>>,
                 out: &mut Vec<TrackedGate>,
                 rng: &mut StdRng,
                 provenance: &mut ProvenanceArena,
                 recovery: &mut RecoveryEvents,
                 rep: &mut CompressReport| {
        let mut set: Vec<usize> = Vec::new();
        let mut queue = seed;
        while let Some(s) = queue.pop() {
            if !slots[s].open || set.contains(&s) {
                continue;
            }
            set.push(s);
            slots[s].open = false;
            // Emitting this group writes its target: poison groups reading it.
            let t = slots[s].target as usize;
            queue.extend(union_of[t].iter().copied());
            // Defensive: its cubes read the union wires; groups TARGETING
            // those wires cannot be open here (their writes would already
            // have closed this group), but sweep them anyway.
            let uw: Vec<usize> = slots[s].union.iter().map(|&w| w as usize).collect();
            for w in uw {
                if let Some(s2) = open_at[w] {
                    queue.push(s2);
                }
            }
        }
        set.sort_unstable_by_key(|&s| slots[s].last);
        for s in set {
            let g = &slots[s];
            if open_at[g.target as usize] == Some(s) {
                open_at[g.target as usize] = None;
            }
            let cubes = reduce_group(
                g.target,
                &slots[s].members,
                iteration,
                p,
                source_parents,
                rng,
                provenance,
                recovery,
                rep,
            );
            out.extend(cubes);
        }
    };

    for (i, tracked) in gates.iter().enumerate() {
        let g = &tracked.gate;
        // Reads close the groups accumulating on the wires they read.
        let mut seed: Vec<usize> = Vec::new();
        for &(w, _) in &g.ctrls {
            if let Some(s) = open_at[w as usize] {
                seed.push(s);
            }
        }
        // The write closes every group whose member controls include target.
        for &s in &union_of[g.target as usize] {
            if slots[s].open {
                seed.push(s);
            }
        }
        union_of[g.target as usize].retain(|&s| slots[s].open);
        if !seed.is_empty() {
            close(
                seed,
                &mut slots,
                &mut open_at,
                &mut union_of,
                &mut out,
                rng,
                provenance,
                recovery,
                rep,
            );
        }
        // Join or open the group for this target.
        match open_at[g.target as usize] {
            Some(s) => {
                let grp = &mut slots[s];
                for &(w, _) in &g.ctrls {
                    if grp.union.binary_search(&w).is_err() {
                        let pos = grp.union.partition_point(|&x| x < w);
                        grp.union.insert(pos, w);
                        union_of[w as usize].push(s);
                    }
                }
                grp.members.push(tracked.clone());
                grp.last = i;
                if grp.members.len() >= p.group_cap {
                    close(
                        vec![s],
                        &mut slots,
                        &mut open_at,
                        &mut union_of,
                        &mut out,
                        rng,
                        provenance,
                        recovery,
                        rep,
                    );
                }
            }
            None => {
                let s = slots.len();
                let mut union: Vec<u16> = g.ctrls.iter().map(|&(w, _)| w).collect();
                union.sort_unstable();
                union.dedup();
                for &w in &union {
                    union_of[w as usize].push(s);
                }
                slots.push(Group {
                    target: g.target,
                    members: vec![tracked.clone()],
                    union,
                    last: i,
                    open: true,
                });
                open_at[g.target as usize] = Some(s);
            }
        }
    }
    let remaining: Vec<usize> = (0..slots.len()).filter(|&s| slots[s].open).collect();
    close(
        remaining,
        &mut slots,
        &mut open_at,
        &mut union_of,
        &mut out,
        rng,
        provenance,
        recovery,
        rep,
    );
    out
}

// Full pass: [liveness] -> gather+reduce, iterated to an exact gate fixed
// point.  The traced entry point retains the provenance sidecar for reporting
// and optional database compression.
pub fn compress_traced(
    gates: Vec<XGate>,
    wires: usize,
    p: &CompressParams,
) -> (TracedCircuit, CompressReport) {
    let source_marks = vec![UNKNOWN_SOURCE; gates.len()];
    compress_traced_with_sources(gates, wires, p, source_marks, Vec::new())
}

/// Compression entry point with source-parent lineage from `fsplit`.
/// `source_marks` must align with `gates`; `source_parents` is the original
/// pre-split tape indexed by each pristine mark.
pub fn compress_traced_with_sources(
    gates: Vec<XGate>,
    wires: usize,
    p: &CompressParams,
    source_marks: Vec<u32>,
    source_parents: Vec<XGate>,
) -> (TracedCircuit, CompressReport) {
    assert_eq!(
        gates.len(),
        source_marks.len(),
        "source marks must align with compression input"
    );
    let mut rng = StdRng::seed_from_u64(p.seed);
    let mut rep = CompressReport::default();
    rep.gates_in = gates.len();
    rep.lits_in = lits_of(&gates);
    let (mut provenance, roots) = ProvenanceArena::from_gates(&gates);
    let mut recovery = RecoveryEvents::default();
    let mut cur: Vec<TrackedGate> = gates
        .into_iter()
        .zip(roots)
        .zip(source_marks)
        .map(|((gate, root), source)| TrackedGate { gate, root, source })
        .collect();
    for iter in 1..=p.max_iters {
        let before = cur.len();
        let mut changed_before_gather = false;
        if let Some(lv) = &p.live_out {
            let (kept, dropped) = liveness_prune_tracked(cur, lv);
            cur = kept;
            rep.liveness_dropped += dropped;
            changed_before_gather = dropped != 0;
        }
        let next = gather_reduce_pass(
            &cur,
            iter,
            wires,
            p,
            &source_parents,
            &mut rng,
            &mut provenance,
            &mut recovery,
            &mut rep,
        );
        let fixed_point = !changed_before_gather
            && next.len() == cur.len()
            && next
                .iter()
                .zip(&cur)
                .all(|(left, right)| left.gate == right.gate);
        cur = next;
        rep.iters = iter;
        println!(
            "[fcompress] iter={} gates {} -> {} | groups={} multi={} max={} | direct_pass1={}/{}frags direct_later={} catalogue={} anf_wins={} anf_g57={} live_dropped={} fixed={}",
            iter,
            before,
            cur.len(),
            rep.groups,
            rep.multi_groups,
            rep.max_group,
            rep.reassembled_pairs,
            rep.reassembled_fragments,
            rep.direct_pairs_later,
            rep.catalogue_merges,
            rep.anf_wins,
            rep.anf_structural_g57,
            rep.liveness_dropped,
            fixed_point
        );
        if fixed_point {
            rep.reached_fixed_point = true;
            break;
        }
    }
    rep.gates_out = cur.len();
    rep.lits_out = cur.iter().map(|tracked| tracked.gate.width() as u64).sum();
    let mut gates = Vec::with_capacity(cur.len());
    let mut roots = Vec::with_capacity(cur.len());
    let mut source_marks = Vec::with_capacity(cur.len());
    for tracked in cur {
        gates.push(tracked.gate);
        roots.push(tracked.root);
        source_marks.push(tracked.source);
    }
    (
        TracedCircuit {
            gates,
            roots,
            source_marks,
            source_parents,
            provenance,
            recovery,
        },
        rep,
    )
}

/// Compatibility wrapper for callers that only need the compressed tape.
pub fn compress(
    gates: Vec<XGate>,
    wires: usize,
    p: &CompressParams,
) -> (Vec<XGate>, CompressReport) {
    let (traced, report) = compress_traced(gates, wires, p);
    (traced.gates, report)
}

#[cfg(test)]
mod compress_tests {
    use super::super::xgate::{XGate, eval_lanes};
    use super::*;

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
    fn fcompress_reassembles_exact_maximum_inside_group() {
        // The first 2cc can consume positive 1 or positive 2; the second can
        // consume only positive 1.  A first-fit choice of positive 1 would
        // recover one g57, while maximum matching recovers both.
        let gates = vec![
            conj(0, &[(1, true)]),
            conj(0, &[(2, true)]),
            conj(0, &[(1, false), (2, false)]),
            conj(0, &[(1, false), (3, false)]),
        ];
        let p = CompressParams {
            max_iters: 10,
            ..Default::default()
        };
        let (out, rep) = compress(gates.clone(), 4, &p);
        assert_eq!(out.len(), 2);
        assert_eq!(rep.reassembled_pairs, 2);
        assert_eq!(rep.reassembled_fragments, 4);
        assert!(equal_on(&gates, &out, 4, None));

        // Report fields describe first-pass input fragments and must not grow
        // merely because the fixed-point loop is allowed more iterations.
        let (_, one_pass) = compress(
            gates,
            4,
            &CompressParams {
                max_iters: 1,
                ..Default::default()
            },
        );
        assert_eq!(rep.reassembled_pairs, one_pass.reassembled_pairs);
        assert_eq!(rep.reassembled_fragments, one_pass.reassembled_fragments);
    }

    #[test]
    fn fcompress_reassembles_alternate_presplit_orientation() {
        // If presplit visits the parent's positive monomial literal first, it
        // emits a negative singleton and an all-positive 2cc.
        let gates = vec![conj(0, &[(1, false)]), conj(0, &[(1, true), (2, true)])];
        let (out, rep) = compress(gates.clone(), 3, &CompressParams::default());
        assert_eq!(rep.reassembled_pairs, 1);
        assert_eq!(rep.reassembled_fragments, 2);
        assert_eq!(out, vec![XGate::from_g57([0, 2, 1])]);
        assert!(equal_on(&gates, &out, 3, None));
    }

    #[test]
    fn source_lineage_distinguishes_parent_return_from_cross_parent_fusion() {
        let gates = vec![conj(0, &[(1, true)]), conj(0, &[(1, false), (2, false)])];
        let parent = XGate::from_g57([0, 1, 2]);
        let (returned, returned_report) = compress_traced_with_sources(
            gates.clone(),
            3,
            &CompressParams::default(),
            vec![0, 0],
            vec![parent.clone()],
        );
        assert_eq!(returned.gates, vec![parent.clone()]);
        assert_eq!(returned.source_marks, vec![0]);
        assert_eq!(returned_report.direct_pass1_sources.returned_to_parent, 1);
        assert_eq!(returned_report.direct_pass1_sources.new_total(), 0);

        let (mixed, mixed_report) = compress_traced_with_sources(
            gates,
            3,
            &CompressParams::default(),
            vec![0, 1],
            vec![parent, XGate::from_g57([0, 2, 1])],
        );
        assert_eq!(
            mixed.source_marks,
            vec![crate::postmix::source::MIXED_SOURCE]
        );
        assert_eq!(mixed_report.direct_pass1_sources.returned_to_parent, 0);
        assert_eq!(mixed_report.direct_pass1_sources.new_mixed_parents, 1);
    }

    #[test]
    fn wider_catalogue_cascade_is_counted_on_later_pass() {
        // xz + x!z -> x, while !x!yw + !x!y!w -> !x!y.  The resulting
        // 1cc+2cc pair is not present on pass one; pass two recovers g57.
        let gates = vec![
            conj(0, &[(1, true), (3, true)]),
            conj(0, &[(1, true), (3, false)]),
            conj(0, &[(1, false), (2, false), (4, true)]),
            conj(0, &[(1, false), (2, false), (4, false)]),
        ];
        let (traced, rep) = compress_traced(gates.clone(), 5, &CompressParams::default());
        assert_eq!(traced.gates, vec![XGate::from_g57([0, 1, 2])]);
        assert_eq!(
            rep.reassembled_pairs, 0,
            "legacy pass-one metric stays zero"
        );
        assert_eq!(rep.direct_pairs_later, 1);
        assert_eq!(rep.direct_pairs_total, 1);
        assert!(rep.reached_fixed_point);
        assert!(equal_on(&gates, &traced.gates, 5, None));

        let summary = traced.recovery_summary();
        assert_eq!(summary.direct_pass1.exact.total, 0);
        assert_eq!(summary.direct_later.exact.total, 4);
        assert_eq!(summary.direct_later.exact.by_initial_width[2], 2);
        assert_eq!(summary.direct_later.exact.by_initial_width[3], 2);
        assert_eq!(summary.ever.exact.total, 4);
        assert_eq!(summary.final_structural_g57.exact.total, 4);
    }

    #[test]
    fn four_control_cascade_attributes_every_wide_leaf() {
        // Shannon-expand x over z,u and !x!y over v,w.  Eight wide leaves
        // collapse through 4cc->3cc->2cc/1cc before the final direct fusion.
        let mut gates = Vec::new();
        for z in [false, true] {
            for u in [false, true] {
                gates.push(conj(0, &[(1, true), (3, z), (4, u)]));
            }
        }
        for v in [false, true] {
            for w in [false, true] {
                gates.push(conj(0, &[(1, false), (2, false), (5, v), (6, w)]));
            }
        }
        let (traced, rep) = compress_traced(gates.clone(), 7, &CompressParams::default());
        assert_eq!(traced.gates, vec![XGate::from_g57([0, 1, 2])]);
        assert_eq!(rep.reassembled_pairs, 0);
        assert_eq!(rep.direct_pairs_later, 1);
        assert!(rep.catalogue_merges >= 6);
        assert!(equal_on(&gates, &traced.gates, 7, None));

        let coverage = traced.recovery_summary().direct_later.exact;
        assert_eq!(coverage.total, 8);
        assert_eq!(coverage.by_initial_width[3], 4);
        assert_eq!(coverage.by_initial_width[4], 4);
    }

    #[test]
    fn anf_created_g57_is_group_attributed_not_exact() {
        // y XOR !x XOR (x AND !y) = x OR !y.
        let gates = vec![
            conj(0, &[(2, true)]),
            conj(0, &[(1, false)]),
            conj(0, &[(1, true), (2, false)]),
        ];
        let (traced, rep) = compress_traced(gates.clone(), 3, &CompressParams::default());
        assert_eq!(traced.gates, vec![XGate::from_g57([0, 1, 2])]);
        assert_eq!(rep.reassembled_pairs, 0);
        assert_eq!(rep.direct_pairs_total, 0);
        assert_eq!(rep.anf_wins, 1);
        assert_eq!(rep.anf_structural_g57, 1);
        assert!(equal_on(&gates, &traced.gates, 3, None));

        let summary = traced.recovery_summary();
        assert_eq!(summary.anf.exact.total, 0);
        assert_eq!(summary.anf.inclusive.total, 3);
        assert_eq!(summary.ever.exact.total, 0);
        assert_eq!(summary.ever.inclusive.total, 3);
        assert_eq!(summary.final_structural_g57.exact.total, 0);
        assert_eq!(summary.final_structural_g57.inclusive.total, 3);
    }

    #[test]
    fn cascade_recovery_still_respects_reader_barrier() {
        let gates = vec![
            conj(0, &[(1, true), (3, true)]),
            conj(0, &[(1, true), (3, false)]),
            conj(5, &[(0, true)]),
            conj(0, &[(1, false), (2, false), (4, true)]),
            conj(0, &[(1, false), (2, false), (4, false)]),
        ];
        let (traced, rep) = compress_traced(gates.clone(), 6, &CompressParams::default());
        assert_eq!(rep.direct_pairs_total, 0);
        assert_eq!(traced.recovery_summary().ever.inclusive.total, 0);
        assert!(equal_on(&gates, &traced.gates, 6, None));
    }

    #[test]
    fn report_distinguishes_iteration_cap_from_fixed_point() {
        let gates = vec![
            conj(0, &[(1, true), (3, true)]),
            conj(0, &[(1, true), (3, false)]),
            conj(0, &[(1, false), (2, false), (4, true)]),
            conj(0, &[(1, false), (2, false), (4, false)]),
        ];
        let (_, capped) = compress(
            gates,
            5,
            &CompressParams {
                max_iters: 1,
                ..Default::default()
            },
        );
        assert!(!capped.reached_fixed_point);
    }

    #[test]
    fn reassembly_does_not_cross_a_gather_barrier() {
        let one = conj(0, &[(1, true)]);
        let two = conj(0, &[(1, false), (2, false)]);
        // This gate reads target 0, so the fragments cannot be gathered into
        // one legal fcompress group even though the global barrier-free metric
        // would regard their shapes as compatible.
        let reader = conj(3, &[(0, true)]);
        let gates = vec![one, reader, two];
        let (out, rep) = compress(gates.clone(), 4, &CompressParams::default());
        assert_eq!(rep.reassembled_pairs, 0);
        assert_eq!(out.len(), gates.len());
        assert!(equal_on(&gates, &out, 4, None));
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
}
