//! Gather groups in dependency order and transport them across equivalent frames.
use super::*;

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
pub(super) fn transport_across(
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
pub(super) fn gather_reduce_pass(
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
            close(
                seed,
                &mut slots,
                &mut open_at,
                &mut out,
                &mut out_anc,
                rng,
                rep,
            );
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
            close(
                refused,
                &mut slots,
                &mut open_at,
                &mut out,
                &mut out_anc,
                rng,
                rep,
            );
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
            close(
                cyc,
                &mut slots,
                &mut open_at,
                &mut out,
                &mut out_anc,
                rng,
                rep,
            );
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
