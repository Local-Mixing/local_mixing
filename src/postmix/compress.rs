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
use super::mix::{Merge, merge_result};
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
            max_iters: 10,
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
    pub anf_wins: u64,
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
    let mut support: Vec<u16> = cubes.iter().flat_map(|c| c.iter().map(|&(w, _)| w)).collect();
    support.sort_unstable();
    support.dedup();
    if support.len() > support_cap.min(31) {
        return None;
    }
    let idx_of: HashMap<u16, u32> =
        support.iter().enumerate().map(|(i, &w)| (w, i as u32)).collect();
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
    let mut monos: Vec<u32> = anf.into_iter().filter(|&(_, on)| on).map(|(m, _)| m).collect();
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
            for j in i + 1..cubes.len() {
                let a = XGate { target, comp: false, ctrls: cubes[i].clone() };
                let b = XGate { target, comp: false, ctrls: cubes[j].clone() };
                if let Some(m) = merge_result(&a, &b) {
                    rep.catalogue_merges += 1;
                    let repl = match m {
                        Merge::Cancel => None,
                        Merge::DropLit(g) | Merge::Subsume(g) | Merge::XFuse(g) => Some(g.ctrls),
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
    if cubes.len() >= 3 {
        if let Some((alt, delta)) = anf_reduce(&cubes, p.anf_support_cap) {
            let (alt_l, cur_l) = (alt.iter().map(|c| c.len()).sum::<usize>(),
                                  cubes.iter().map(|c| c.len()).sum::<usize>());
            if (alt.len(), alt_l) < (cubes.len(), cur_l) {
                rep.anf_wins += 1;
                cubes = alt;
                parity ^= delta;
            }
        }
    }
    let mut out: Vec<XGate> = Vec::with_capacity(cubes.len().max(1));
    for (k, c) in cubes.into_iter().enumerate() {
        out.push(XGate { target, comp: parity && k == 0, ctrls: c });
    }
    if out.is_empty() && parity {
        out.push(XGate::x_gate(target));
    }
    if p.local_verify {
        verify_group(members, &out, rng);
    }
    out
}

fn verify_group(before: &[XGate], after: &[XGate], rng: &mut StdRng) {
    let mut support: Vec<u16> =
        before.iter().chain(after).flat_map(|g| g.ctrls.iter().map(|&(w, _)| w)).collect();
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
        let pos: HashMap<u16, usize> =
            support.iter().enumerate().map(|(i, &w)| (w, i)).collect();
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
    members: Vec<XGate>,
    union: Vec<u16>, // sorted control-wire union across members
    last: usize,     // index of the last member in the input order
    open: bool,
}

// One forward gather-and-reduce sweep.
fn gather_reduce_pass(
    gates: &[XGate],
    wires: usize,
    p: &CompressParams,
    rng: &mut StdRng,
    rep: &mut CompressReport,
) -> Vec<XGate> {
    let mut out: Vec<XGate> = Vec::with_capacity(gates.len());
    let mut slots: Vec<Group> = Vec::new();
    let mut open_at: Vec<Option<usize>> = vec![None; wires]; // target wire -> slot
    let mut union_of: Vec<Vec<usize>> = vec![Vec::new(); wires]; // wire -> slots (stale ok)

    // Close a seed set of groups: cascade poison from emitted writes, order
    // by last-member index (provably conflict-free), reduce, emit.
    let close = |seed: Vec<usize>,
                 slots: &mut Vec<Group>,
                 open_at: &mut Vec<Option<usize>>,
                 union_of: &mut Vec<Vec<usize>>,
                 out: &mut Vec<XGate>,
                 rng: &mut StdRng,
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
            let cubes = reduce_group(g.target, &slots[s].members, p, rng, rep);
            out.extend(cubes);
        }
    };

    for (i, g) in gates.iter().enumerate() {
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
            close(seed, &mut slots, &mut open_at, &mut union_of, &mut out, rng, rep);
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
                grp.members.push(g.clone());
                grp.last = i;
                if grp.members.len() >= p.group_cap {
                    close(vec![s], &mut slots, &mut open_at, &mut union_of, &mut out, rng, rep);
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
                slots.push(Group { target: g.target, members: vec![g.clone()], union, last: i, open: true });
                open_at[g.target as usize] = Some(s);
            }
        }
    }
    let remaining: Vec<usize> = (0..slots.len()).filter(|&s| slots[s].open).collect();
    close(remaining, &mut slots, &mut open_at, &mut union_of, &mut out, rng, rep);
    out
}

// Full pass: [liveness] -> gather+reduce, iterated to a gate-count fixed
// point. Prints one [fcompress] line per iteration.
pub fn compress(gates: Vec<XGate>, wires: usize, p: &CompressParams) -> (Vec<XGate>, CompressReport) {
    let mut rng = StdRng::seed_from_u64(p.seed);
    let mut rep = CompressReport::default();
    rep.gates_in = gates.len();
    rep.lits_in = lits_of(&gates);
    let mut cur = gates;
    for iter in 1..=p.max_iters {
        let before = cur.len();
        if let Some(lv) = &p.live_out {
            let (kept, dropped) = liveness_prune(cur, lv);
            cur = kept;
            rep.liveness_dropped += dropped;
        }
        cur = gather_reduce_pass(&cur, wires, p, &mut rng, &mut rep);
        rep.iters = iter;
        println!(
            "[fcompress] iter={} gates {} -> {} | groups={} multi={} max={} | catalogue={} anf_wins={} live_dropped={}",
            iter, before, cur.len(), rep.groups, rep.multi_groups, rep.max_group,
            rep.catalogue_merges, rep.anf_wins, rep.liveness_dropped
        );
        if cur.len() >= before {
            break;
        }
    }
    rep.gates_out = cur.len();
    rep.lits_out = lits_of(&cur);
    (cur, rep)
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
            conj(1, &[(2, true)]),          // stays: 1 read below by live gate
            conj(0, &[(1, true)]),          // live
            conj(1, &[(0, true)]),          // dead: nothing live reads 1 after
        ];
        let live = vec![true, false, true];
        let p = CompressParams { live_out: Some(live.clone()), ..Default::default() };
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
        assert!(out.len() < gates.len(), "expected some compression on dense circuit");
    }
}
