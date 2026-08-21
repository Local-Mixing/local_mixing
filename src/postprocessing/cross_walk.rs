// Production GSS Stage 5: the directional crossing walk and its
// thermostat contraction channels. The shared Mixer owns arena, provenance,
// checkpoint, and database state; this module owns the concrete cross/undo/
// merge walk operations.
use crate::circuit::xgate::XGate;
use crate::engine::arena::{Dir, NIL};
use crate::engine::mix::{Merge, Meta, Mixer, UndoEntry, key_of, merge_key, merge_result};
use crate::engine::rules::{self, BlockReason, Outcome, Role, RuleKind};
use rand::Rng;

impl Mixer {
    pub(crate) fn cross_move(&mut self) {
        let id = self.pick_cross_shot();
        self.cross_move_on(id);
    }

    /// Shot selection for the cross. With p_mincross off (the default) this
    /// is exactly the historical uniform draw and consumes no extra RNG.
    /// Armed, a coin sends the shot to the min-dgen pool: the K least-split
    /// lineages at the last cadenced rebuild, each entry consumed on draw
    /// (dead entries pruned lazily). The pool draining before the next
    /// rebuild silently degrades to uniform — size cross_pool_k above the
    /// expected biased draws per cadence.
    fn pick_cross_shot(&mut self) -> u32 {
        if self.params.p_mincross > 0.0 {
            if self.moves_done >= self.cross_pool_due {
                self.rebuild_cross_pool();
            }
            if !self.cross_pool.is_empty()
                && self.rng.random_bool(self.params.p_mincross.clamp(0.0, 1.0))
            {
                for _ in 0..8 {
                    if self.cross_pool.is_empty() {
                        break;
                    }
                    let i = self.rng.random_range(0..self.cross_pool.len());
                    let id = self.cross_pool.swap_remove(i);
                    if self.arena.is_linked(id) {
                        self.counters.cross_pool_shots += 1;
                        return id;
                    }
                }
            }
        }
        self.arena.random_linked(&mut self.rng)
    }

    /// O(n) scan + O(n) select of the K lowest-dgen linked gates. dgen is
    /// the split-generation stamp — during a DB-free walk it counts the
    /// split events in a gate's lineage, so low dgen IS "family the walk
    /// has not touched".
    fn rebuild_cross_pool(&mut self) {
        let mut cand: Vec<(u32, u32)> = self
            .arena
            .ids_in_order()
            .into_iter()
            .map(|id| (self.meta_of(id).dgen, id))
            .collect();
        let k = self.params.cross_pool_k.min(cand.len());
        if k > 0 && k < cand.len() {
            cand.select_nth_unstable(k - 1);
        }
        cand.truncate(k);
        self.cross_pool = cand.into_iter().map(|(_, id)| id).collect();
        self.cross_pool_due = self.moves_done + self.params.cross_rescan.max(1);
    }

    pub(crate) fn cross_move_on(&mut self, id: u32) {
        let dir = self.meta_of(id).dir;
        let way = self.float_to_collision(id, dir);
        let h_id = self.arena.neighbor(id, dir);
        if h_id == NIL {
            self.counters.boundary += 1;
            self.retreat(id, way, dir);
            return;
        }
        let g = self.arena.gate(id).clone();
        let h = self.arena.gate(h_id).clone();

        if g.comp {
            if !self.split_allowed(g.width()) {
                self.counters.declined += 1;
                self.retreat(id, way, dir);
                return;
            }
            let pieces = rules::presplit(&g, &mut self.rng);
            if self.params.local_verify {
                assert!(
                    rules::verify_rewrite(std::slice::from_ref(&g), &pieces),
                    "presplit verification failed: {g:?} -> {pieces:?}"
                );
            }
            let pm = self.meta_of(id);
            let ev = self.fresh_event();
            for p in &pieces {
                self.counters.width_hist[p.width().min(15)] += 1;
            }
            let ids = self.splice_replace_one(id, pieces);
            // Sibling convention (2026-08-05): pieces of one g57 split take
            // ALTERNATING directions from a fair draw — never the old
            // independent per-piece child_dir, under which siblings could
            // agree.
            let d0 = self.rand_dir();
            for (i, &pid) in ids.iter().enumerate() {
                let d = if i % 2 == 0 { d0 } else { d0.opposite() };
                self.set_meta(
                    pid,
                    Meta {
                        origin: pm.origin,
                        event: ev,
                        dir: d,
                        dgen: self.child_gen(pm.dgen),
                        litter: pm.litter,
                        litter_size: pm.litter_size,
                    },
                );
            }
            self.advance_births(&ids);
            self.counters.presplits += 1;
            return;
        }

        match rules::cross(&g, &h, self.params.k_max, &mut self.rng) {
            Outcome::R0Swap => unreachable!("R0 after floating to collision"),
            Outcome::Blocked(BlockReason::WidthCap) => {
                self.counters.blocked_width += 1;
                self.retreat(id, way, dir);
            }
            Outcome::Blocked(BlockReason::Deadlock) => {
                self.counters.blocked_deadlock += 1;
                self.retreat(id, way, dir);
            }
            Outcome::PresplitColliding => {
                // The colliding gate is a g57 that must split: pre-splitting it
                // is this move's whole effect.
                if !self.split_allowed(h.width()) {
                    self.counters.declined += 1;
                    self.retreat(id, way, dir);
                    return;
                }
                let hp = rules::presplit(&h, &mut self.rng);
                if self.params.local_verify {
                    assert!(
                        rules::verify_rewrite(std::slice::from_ref(&h), &hp),
                        "colliding presplit verification failed: {h:?} -> {hp:?}"
                    );
                }
                let hm = self.meta_of(h_id);
                let ev = self.fresh_event();
                for p in &hp {
                    self.counters.width_hist[p.width().min(15)] += 1;
                }
                let ids = self.splice_replace_one(h_id, hp);
                // Sibling convention (2026-08-05): alternating directions
                // from a fair draw, replacing the old inherit-from-the-shot
                // law for presplit fragments.
                let d0 = self.rand_dir();
                for (i, &pid) in ids.iter().enumerate() {
                    let d = if i % 2 == 0 { d0 } else { d0.opposite() };
                    self.set_meta(
                        pid,
                        Meta {
                            origin: hm.origin,
                            event: ev,
                            dir: d,
                            dgen: self.child_gen(hm.dgen),
                            litter: hm.litter,
                            litter_size: hm.litter_size,
                        },
                    );
                }
                self.advance_births(&ids);
                self.counters.presplits += 1;
            }
            Outcome::Rewrite { seq, kind, dropped } => {
                let split_width = match kind {
                    RuleKind::R1 | RuleKind::R3 => g.width(),
                    RuleKind::R2 => h.width(),
                };
                if !self.split_allowed(split_width) {
                    self.counters.declined += 1;
                    self.retreat(id, way, dir);
                    return;
                }
                if self.params.local_verify {
                    let before: Vec<XGate> = match dir {
                        Dir::R => vec![g.clone(), h.clone()],
                        Dir::L => vec![h.clone(), g.clone()],
                    };
                    let after: Vec<XGate> = match dir {
                        Dir::R => seq.iter().map(|(x, _)| x.clone()).collect(),
                        Dir::L => seq.iter().rev().map(|(x, _)| x.clone()).collect(),
                    };
                    assert!(
                        rules::verify_rewrite(&before, &after),
                        "cross verification failed ({kind:?}, {dir:?}): {g:?} x {h:?}"
                    );
                }
                self.counters.dropped_neverfire += dropped as u64;
                match kind {
                    RuleKind::R1 => self.counters.cross_r1 += 1,
                    RuleKind::R2 => self.counters.cross_r2 += 1,
                    RuleKind::R3 => self.counters.cross_r3 += 1,
                }
                for (gate, role) in &seq {
                    if *role != Role::CollidingIntact {
                        self.counters.width_hist[gate.width().min(15)] += 1;
                    }
                }
                let gm = self.meta_of(id);
                let hm = self.meta_of(h_id);
                let (g_origin, h_origin) = (gm.origin, hm.origin);
                let ev = self.fresh_event();
                let placed = self.splice_pair(id, h_id, dir, seq);
                // Ancestry treats a cross as a DB splice over the window
                // {g, h}: EVERY output — the intact pivot included — carries
                // the union of both parents' ancestor sets, because the
                // rewrite re-encodes the pair jointly even when one gate's
                // spelling survives it. The journal entry below records the
                // PRE-cross litters, so an undo reverses this too. Litter
                // inheritance is unchanged when ancestry is off, matching
                // the merge-union precedent.
                let union_litter = if self.anc_words > 0 && gm.litter != hm.litter {
                    let l = self.anc_union_litter(&[gm.litter, hm.litter]);
                    Some((l, placed.len().min(u16::MAX as usize) as u16))
                } else {
                    None
                };
                let mut fresh: Vec<u32> = Vec::new();
                for &(pid, role) in &placed {
                    match role {
                        Role::ShotPiece | Role::Core => {
                            let d = self.child_dir(dir);
                            let (litter, litter_size) =
                                union_litter.unwrap_or((gm.litter, gm.litter_size));
                            self.set_meta(
                                pid,
                                Meta {
                                    origin: g_origin,
                                    event: ev,
                                    dir: d,
                                    dgen: self.child_gen(gm.dgen),
                                    litter,
                                    litter_size,
                                },
                            );
                            fresh.push(pid);
                        }
                        Role::CollidingPiece => {
                            let d = self.child_dir(dir);
                            let (litter, litter_size) =
                                union_litter.unwrap_or((hm.litter, hm.litter_size));
                            self.set_meta(
                                pid,
                                Meta {
                                    origin: h_origin,
                                    event: ev,
                                    dir: d,
                                    dgen: self.child_gen(hm.dgen),
                                    litter,
                                    litter_size,
                                },
                            );
                            fresh.push(pid);
                        }
                        Role::CollidingIntact => {
                            // Node reused; only the ancestry-bearing fields
                            // move. Origin, event, dir and dgen stay intact so
                            // trajectories and tabu semantics are unchanged.
                            // The stamp bump makes absorption count as
                            // "further processing": any OLDER journal entry
                            // holding this node dies, since undoing it would
                            // wipe the ancestors absorbed here. This cross's
                            // own entry records the post-bump stamp below and
                            // stays undoable.
                            if let Some((l, ls)) = union_litter {
                                let m = self.meta_of(pid);
                                self.set_meta(
                                    pid,
                                    Meta {
                                        litter: l,
                                        litter_size: ls,
                                        ..m
                                    },
                                );
                                self.arena.touch(pid);
                            }
                        }
                    }
                }
                self.advance_births(&fresh);
                // Record for exact reversal (only when the undo channel is
                // live — with undo_frac == 0 the journal would be dead weight).
                // Pivot: the node every other piece collides with — h when
                // intact (R1/R3), else the passed shot (R2, the only ShotPiece
                // there).
                if self.params.undo_frac > 0.0 {
                    let pivot = placed
                        .iter()
                        .find(|(_, r)| *r == Role::CollidingIntact)
                        .or_else(|| placed.iter().find(|(_, r)| *r == Role::ShotPiece))
                        .map(|&(i, _)| i)
                        .expect("rewrite emitted no pivot");
                    let (before, origins, gens, litters, litter_sizes) = match dir {
                        Dir::R => (
                            [g.clone(), h.clone()],
                            [g_origin, h_origin],
                            [gm.dgen, hm.dgen],
                            [gm.litter, hm.litter],
                            [gm.litter_size, hm.litter_size],
                        ),
                        Dir::L => (
                            [h.clone(), g.clone()],
                            [h_origin, g_origin],
                            [hm.dgen, gm.dgen],
                            [hm.litter, gm.litter],
                            [hm.litter_size, gm.litter_size],
                        ),
                    };
                    let after: Vec<(u32, u32)> = placed
                        .iter()
                        .map(|&(i, _)| (i, self.arena.stamp(i)))
                        .collect();
                    if self.journal.len() >= self.params.journal_len {
                        self.journal.pop_front();
                    }
                    self.journal.push_back(UndoEntry {
                        before,
                        dir,
                        pivot,
                        after,
                        event: ev,
                        origins,
                        gens,
                        litters,
                        litter_sizes,
                        misses: 0,
                    });
                }
            }
        }
    }
    // ---- the contraction moves ----

    // Reverse a recorded crossing: sample journal entries until a live one is
    // found (dead ones — any piece touched since — are discarded), gather its
    // pieces back around the pivot by floating, verify the block against the
    // original pair exhaustively, and splice [g, h] back in.
    pub(crate) fn undo_move(&mut self) -> bool {
        for _ in 0..8 {
            if self.journal.is_empty() {
                return false;
            }
            let i = self.rng.random_range(0..self.journal.len());
            let alive = self.journal[i]
                .after
                .iter()
                .all(|&(id, st)| self.arena.is_linked(id) && self.arena.stamp(id) == st);
            if !alive {
                self.journal.swap_remove_back(i);
                self.counters.undo_dead += 1;
                continue;
            }
            if self.is_tabu(self.journal[i].event) {
                self.counters.undo_tabu += 1;
                continue;
            }
            let e = self
                .journal
                .swap_remove_back(i)
                .expect("journal index valid");
            if let Some(mut e) = self.try_undo(e) {
                // Gather miss: pieces only floated (function-preserving), the
                // entry is still valid — retry later, but only a few times
                // (a blocked entry usually stays blocked).
                e.misses += 1;
                if e.misses < 3 {
                    self.journal.push_back(e);
                }
                return false;
            }
            return true;
        }
        false
    }

    // Returns the entry back on a gather miss, None on success.
    fn try_undo(&mut self, e: UndoEntry) -> Option<UndoEntry> {
        let (mut edge_l, mut edge_r) = (e.pivot, e.pivot);
        for &(id, _) in &e.after {
            if id == e.pivot {
                continue;
            }
            // Locate the piece relative to the current block, then float it
            // onto the block's edge. Pieces pairwise commute (same target, no
            // reads of it), so ungathered siblings never block the float.
            // Scan the piece's own (persistent) Meta direction first: pieces
            // are advanced in their meta dir at birth, so the hint is usually
            // right and the full-reach wrong-side miss is skipped. The piece's
            // LOCATION determines side/found, not the scan order, so the
            // outcome is bit-identical.
            let hint = self.meta_of(id).dir;
            let mut side = None;
            for d in [hint, hint.opposite()] {
                let edge = match d {
                    Dir::R => edge_r,
                    Dir::L => edge_l,
                };
                let mut cur = self.arena.neighbor(edge, d);
                let mut steps = 0usize;
                while cur != NIL && steps < self.params.merge_reach {
                    if cur == id {
                        side = Some(d);
                        break;
                    }
                    cur = self.arena.neighbor(cur, d);
                    steps += 1;
                }
                if side.is_some() {
                    break;
                }
            }
            let Some(side) = side else {
                self.counters.undo_gather_miss += 1;
                return Some(e);
            };
            let (anchor, ok) = match side {
                Dir::R => {
                    self.float_until(id, Dir::L, edge_r);
                    (id, self.arena.neighbor(edge_r, Dir::R) == id)
                }
                Dir::L => {
                    self.float_until(id, Dir::R, edge_l);
                    (id, self.arena.neighbor(edge_l, Dir::L) == id)
                }
            };
            if !ok {
                self.counters.undo_gather_miss += 1;
                return Some(e);
            }
            match side {
                Dir::R => edge_r = anchor,
                Dir::L => edge_l = anchor,
            }
        }
        // Contiguous block [edge_l ..= edge_r] now holds exactly the pieces.
        let mut block: Vec<u32> = Vec::with_capacity(e.after.len());
        let mut cur = edge_l;
        loop {
            block.push(cur);
            if cur == edge_r {
                break;
            }
            cur = self.arena.neighbor(cur, Dir::R);
        }
        debug_assert_eq!(
            block.len(),
            e.after.len(),
            "gathered block is not contiguous"
        );
        if self.params.local_verify {
            let actual: Vec<XGate> = block.iter().map(|&b| self.arena.gate(b).clone()).collect();
            assert!(
                rules::verify_rewrite(&actual, &e.before),
                "undo verification failed: block {actual:?} != {:?}",
                e.before
            );
        }
        let cursor = self.arena.neighbor(edge_l, Dir::L);
        for &bid in &block {
            self.evict_taps(bid);
            self.index_remove(bid);
            self.arena.unlink(bid);
            self.arena.free_node(bid);
        }
        let mut c = cursor;
        let mut new_ids: Vec<u32> = Vec::with_capacity(2);
        for (j, gate) in e.before.iter().enumerate() {
            c = self.arena.insert_after(c, gate.clone());
            self.index_add(c);
            let d = self.rand_dir();
            self.set_meta(
                c,
                Meta {
                    origin: e.origins[j],
                    event: 0,
                    dir: d,
                    dgen: e.gens[j],
                    litter: e.litters[j],
                    litter_size: e.litter_sizes[j],
                },
            );
            new_ids.push(c);
        }
        self.counters.undos += 1;
        None
    }

    // Partner ids for `g_id` from the index: same-key gates (cancel / xfuse /
    // drop-lit) plus each one-wire-reduced key (subsume, larger side). The
    // smaller subsume partner finds the pair when IT is sampled as the larger
    // one — initiators are uniform, so coverage is symmetric.
    fn merge_candidates(&self, g_id: u32) -> smallvec::SmallVec<[u32; 8]> {
        let g = self.arena.gate(g_id);
        let mut out = smallvec::SmallVec::<[u32; 8]>::new();
        let consider = |ids: &[u32], out: &mut smallvec::SmallVec<[u32; 8]>| {
            for &c in ids {
                if c != g_id && merge_result(g, self.arena.gate(c)).is_some() {
                    out.push(c);
                }
            }
        };
        if let Some(ids) = self.index.get(&key_of(g)) {
            consider(ids, &mut out);
        }
        for skip in 0..g.ctrls.len() {
            let k = merge_key(
                g.target,
                g.ctrls
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != skip)
                    .map(|(_, &(w, _))| w),
            );
            if let Some(ids) = self.index.get(&k) {
                consider(ids, &mut out);
            }
        }
        out
    }

    pub(crate) fn merge_move(&mut self) -> bool {
        // Sample initiators until one has index partners; most gates are
        // ladder-form with unique keys, so a direct random pick rarely works.
        let mut found: Option<(u32, smallvec::SmallVec<[u32; 8]>)> = None;
        for _ in 0..8 {
            let g_id = self.arena.random_linked(&mut self.rng);
            let cands = self.merge_candidates(g_id);
            if !cands.is_empty() {
                found = Some((g_id, cands));
                break;
            }
        }
        let Some((g_id, cands)) = found else {
            self.counters.merge_no_partner += 1;
            return false;
        };
        // Far partners are usually wall-blocked (over a long span some gate
        // almost surely collides with both), so scan outward from the
        // initiator and take the NEAREST reachable candidate, tracking the
        // initiator's colliders incrementally for the wall check.
        let mut cand_set = cands;
        cand_set.sort_unstable();
        let mut chosen: Option<(u32, Dir, usize)> = None;
        for dir in [Dir::R, Dir::L] {
            // The collider list is consulted ONLY when a candidate is actually
            // reached, and the overwhelming majority of these scans reach none
            // (merge_too_far dominates the counters). Build it lazily: `built`
            // walks the same nodes in the same order and tests each at most
            // once, catching up only as far as the node before a candidate.
            // The list handed to a wall check is therefore the identical
            // prefix, in the identical order, that the eager walk would have
            // produced -- so the wall verdict, the counters and the chosen
            // partner are unchanged -- while a scan that finds no candidate now
            // costs no collides_ids calls at all instead of one per step.
            // (`Vec::new` does not allocate until the first push, so the common
            // no-collider case is also allocation-free.)
            let mut g_colliders: Vec<u32> = Vec::new();
            let mut built = self.arena.neighbor(g_id, dir);
            let mut built_steps = 0usize;
            let mut cur = self.arena.neighbor(g_id, dir);
            let mut steps = 0usize;
            while cur != NIL && steps < self.params.merge_reach {
                if chosen.is_some_and(|(_, _, d)| steps >= d) {
                    break; // the other direction already found a nearer one
                }
                if cand_set.binary_search(&cur).is_ok() {
                    // Catch up over every node strictly before `cur`, which is
                    // exactly what the eager loop had accumulated by this point.
                    while built_steps < steps {
                        if self.arena.collides_ids(g_id, built) {
                            g_colliders.push(built);
                        }
                        built = self.arena.neighbor(built, dir);
                        built_steps += 1;
                    }
                    debug_assert_eq!(
                        g_colliders,
                        {
                            let mut eager = Vec::new();
                            let mut n = self.arena.neighbor(g_id, dir);
                            for _ in 0..steps {
                                if self.arena.collides_ids(g_id, n) {
                                    eager.push(n);
                                }
                                n = self.arena.neighbor(n, dir);
                            }
                            eager
                        },
                        "lazy collider cursor diverged from the eager walk"
                    );
                    // Indexed rather than `.iter().any(|..| self.arena..)`: the
                    // closure would capture `self` while `g_colliders` is
                    // borrowed. Short-circuits at the same element as before.
                    let mut wall = false;
                    for i in 0..g_colliders.len() {
                        if self.arena.collides_ids(g_colliders[i], cur) {
                            wall = true;
                            break;
                        }
                    }
                    if wall {
                        self.counters.merge_wall_blocked += 1;
                    } else {
                        chosen = Some((cur, dir, steps));
                        break;
                    }
                }
                cur = self.arena.neighbor(cur, dir);
                steps += 1;
            }
        }
        let Some((h_id, dir, _)) = chosen else {
            self.counters.merge_too_far += 1;
            return false;
        };
        let (mg, mh) = (self.meta_of(g_id), self.meta_of(h_id));
        let sibling = mg.event != 0 && mg.event == mh.event;
        if sibling && self.is_tabu(mg.event) {
            self.counters.tabu_blocked += 1;
            return false;
        }
        if !self.bring_adjacent(g_id, h_id, dir) {
            self.counters.merge_not_adjacent += 1;
            return false;
        }
        let g = self.arena.gate(g_id).clone();
        let h = self.arena.gate(h_id).clone();
        let merged = merge_result(&g, &h).expect("candidate was mergeable");
        let out = merged.gates();
        if self.params.local_verify {
            assert!(
                rules::verify_rewrite(&[g.clone(), h.clone()], &out),
                "merge verification failed: {g:?} + {h:?}"
            );
        }
        // Splice: replace the adjacent pair by the merged gate (or nothing).
        // Same-target gates commute, so their order is irrelevant.
        let left = if self.arena.neighbor(g_id, Dir::R) == h_id {
            g_id
        } else {
            h_id
        };
        let cursor = self.arena.neighbor(left, Dir::L);
        // Evict-then-unlink per node: h's eviction runs after g's unlink so
        // its target can never be the dying g (both partners die here).
        self.evict_taps(g_id);
        self.index_remove(g_id);
        self.arena.unlink(g_id);
        self.evict_taps(h_id);
        self.index_remove(h_id);
        self.arena.unlink(h_id);
        self.arena.free_node(g_id);
        self.arena.free_node(h_id);
        let mut new_ids: Vec<u32> = Vec::new();
        let mut c = cursor;
        for gate in out {
            c = self.arena.insert_after(c, gate);
            self.index_add(c);
            new_ids.push(c);
        }
        let origin = if mg.origin == mh.origin {
            mg.origin
        } else {
            self.counters.merges_cross_origin += 1;
            mg.origin
        };
        for &nid in &new_ids {
            // Merged output stays in place (scatter is suspended) and keeps
            // shooting the way the initiating parent was headed, per dir_p.
            // Gen: the merged content depends on both parents, so it is only
            // as re-encoded as the LESS re-encoded of the two.
            let d = self.child_dir(mg.dir);
            // Litter follows the same parent the generation does: the merged
            // gate is only as re-encoded as the less re-encoded parent, so it
            // inherits that parent's provenance too.
            let (mut litter, litter_size) = if mg.dgen <= mh.dgen {
                (mg.litter, mg.litter_size)
            } else {
                (mh.litter, mh.litter_size)
            };
            // Ancestry, unlike provenance, comes from both: the merged content
            // depends on each parent, so under --ancestors the merge mints a
            // litter carrying the union rather than picking a side.
            if self.anc_words > 0 && mg.litter != mh.litter {
                litter = self.anc_union_litter(&[mg.litter, mh.litter]);
            }
            self.set_meta(
                nid,
                Meta {
                    origin,
                    event: 0,
                    dir: d,
                    dgen: mg.dgen.min(mh.dgen),
                    litter,
                    litter_size,
                },
            );
        }
        if sibling {
            self.counters.merges_sibling += 1;
        }
        match merged {
            Merge::Cancel => self.counters.merges_cancel += 1,
            Merge::XFuse(_) => self.counters.merges_xfuse += 1,
            Merge::DropLit(_) => self.counters.merges_drop += 1,
            Merge::Subsume(_) => self.counters.merges_subsume += 1,
            Merge::Absorb(_) => self.counters.merges_absorb += 1,
        }
        true
    }
    // Float g toward h (stopping at h), then h toward g (stopping at g).
    // Adjacent afterwards iff nothing colliding with both sat between them AND
    // the two one-directional floats suffice (interleaved blockers can still
    // prevent a meet; that is a harmless miss).
    fn bring_adjacent(&mut self, g_id: u32, h_id: u32, dir: Dir) -> bool {
        self.float_until(g_id, dir, h_id);
        if self.arena.neighbor(g_id, dir) == h_id {
            return true;
        }
        self.float_until(h_id, dir.opposite(), g_id);
        self.arena.neighbor(g_id, dir) == h_id
    }
    // Fragment direction law: inherit the shot gate's direction with
    // probability dir_p, else the opposite.
    fn child_dir(&mut self, shot: Dir) -> Dir {
        if self.rng.random_bool(self.params.dir_p) {
            shot
        } else {
            shot.opposite()
        }
    }
    // Failed cross: the shot gate does not stay parked at the collision — it
    // retreats floor((1 - dir_q) * way) of the way it floated in. The path is
    // passable by construction (it just floated through those gates).
    fn retreat(&mut self, id: u32, way: usize, dir: Dir) {
        let k = ((1.0 - self.params.dir_q) * way as f64).floor() as usize;
        if k == 0 {
            return;
        }
        let back = dir.opposite();
        let mut anchor = id;
        for _ in 0..k {
            anchor = self.arena.neighbor(anchor, back);
        }
        self.arena.unlink(id);
        match back {
            Dir::L => self.arena.link_before(id, anchor),
            Dir::R => self.arena.link_after(id, anchor),
        }
        self.counters.floats += 1;
        self.counters.float_steps += k as u64;
    }
    fn split_allowed(&mut self, c: usize) -> bool {
        let d = self.params.split_damp;
        if c <= d {
            return true;
        }
        let b = self.params.split_base;
        if b <= 1.0 {
            return true;
        }
        let p = b.powi(-(((c - d) as i32).min(1000)));
        self.rng.random_bool(p.min(1.0))
    }
    fn splice_pair(
        &mut self,
        g_id: u32,
        h_id: u32,
        dir: Dir,
        seq: Vec<(XGate, Role)>,
    ) -> Vec<(u32, Role)> {
        let first = match dir {
            Dir::R => g_id,
            Dir::L => h_id,
        };
        let mut cursor = self.arena.neighbor(first, Dir::L);
        // Canary re-anchoring: evict g while it is linked, but evict h only
        // AFTER g's unlink — with both still linked, h's left neighbor can BE
        // the dying g (dir R, no CollidingIntact), which would strand h's
        // taps on a freed slot. After the unlink the neighbor pointers are
        // repaired, so the eviction target is always a survivor. h keeps its
        // node (and taps) when the rewrite carries a CollidingIntact.
        let h_stays = seq.iter().any(|&(_, r)| r == Role::CollidingIntact);
        self.evict_taps(g_id);
        self.index_remove(g_id);
        self.arena.unlink(g_id);
        if !h_stays {
            self.evict_taps(h_id);
        }
        self.arena.unlink(h_id);
        let emitted: Vec<(XGate, Role)> = match dir {
            Dir::R => seq,
            Dir::L => seq.into_iter().rev().collect(),
        };
        let mut h_reused = false;
        let mut out = Vec::with_capacity(emitted.len());
        for (gate, role) in emitted {
            let id = if role == Role::CollidingIntact {
                debug_assert_eq!(&gate, self.arena.gate(h_id));
                self.arena.link_after(h_id, cursor);
                h_reused = true;
                h_id
            } else {
                let nid = self.arena.insert_after(cursor, gate);
                self.index_add(nid);
                nid
            };
            cursor = id;
            out.push((id, role));
        }
        self.arena.free_node(g_id);
        if !h_reused {
            self.index_remove(h_id);
            self.arena.free_node(h_id);
        }
        out
    }
}
