//! DB window geometry, generation-biased seed pools and deterministic sampling.
use super::*;

impl Mixer {
    // ---- DB window sampling ----
    //
    // Both samplers return a contiguous run of node ids (link order, leftmost
    // first) plus g1's stored direction (for the incoming-gate pivot rule), or
    // None when the attempt aborts (the L-cap could not be satisfied) or too few
    // gates could be gathered. Any floating they do is function-preserving, so a
    // subsequent miss leaves the circuit equivalent.

    pub(super) fn width_of(&self, id: u32) -> usize {
        self.arena.gate(id).width()
    }

    // Does any gate of the contiguous span [lo..hi] collide with (not commute
    // with) gate `x`?
    pub(super) fn span_collides(&self, lo: u32, hi: u32, x: u32) -> bool {
        let mut cur = lo;
        loop {
            if self.arena.collides_ids(cur, x) {
                return true;
            }
            if cur == hi {
                return false;
            }
            cur = self.arena.neighbor(cur, Dir::R);
        }
    }

    // Move a commuting neighbor `x` from just past the span's `dir` end to the
    // far side of the span (one function-preserving hop of the whole block).
    pub(super) fn move_across(&mut self, x: u32, lo: u32, hi: u32, dir: Dir) {
        self.arena.unlink(x);
        match dir {
            Dir::R => self.arena.link_before(x, lo), // block shifts right past x
            Dir::L => self.arena.link_after(x, hi),
        }
        self.counters.floats += 1;
        self.counters.float_steps += 1;
    }

    pub(super) fn span_ids(&self, lo: u32, hi: u32) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut cur = lo;
        loop {
            ids.push(cur);
            if cur == hi {
                break;
            }
            cur = self.arena.neighbor(cur, Dir::R);
        }
        ids
    }

    // Returns the sampled window plus WHICH geometry actually built it (the
    // coin outcome under Mixed), so records and stats can split by sampler.
    /// Draw `litter_samples` candidate windows and keep the one spanning the
    /// most distinct litters. Diversity is the point: a window drawn from many
    /// replacement events is one no single earlier splice can undo, which is
    /// the same property the full-litter ban enforces at the other end.
    pub(super) fn sample_best_window(
        &mut self,
        w: usize,
        geo: DbSample,
    ) -> Option<(Vec<u32>, Dir, DbSample)> {
        let n = self.params.litter_samples.max(1);
        let mut best: Option<(Vec<u32>, Dir, DbSample)> = None;
        let mut best_distinct = 0usize;
        for _ in 0..n {
            let Some(cand) = self.sample_window(w, geo) else {
                continue;
            };
            let d = self.litter_census(&cand.0).0;
            if best.is_none() || d > best_distinct {
                best_distinct = d;
                best = Some(cand);
            }
            if best_distinct >= w {
                break; // already maximal; more draws cannot improve it
            }
        }
        best
    }

    /// Collect one candidate window under an ALREADY-CHOSEN geometry. The coin
    /// lives in `db_attempt_inner` now (see `DbSample`).
    pub(super) fn sample_window(
        &mut self,
        w: usize,
        geo: DbSample,
    ) -> Option<(Vec<u32>, Dir, DbSample)> {
        match geo {
            DbSample::Contiguous => self
                .collect_contiguous(w)
                .map(|(ids, d)| (ids, d, DbSample::Contiguous)),
            DbSample::Convex => self
                .collect_convex(w)
                .map(|(ids, d)| (ids, d, DbSample::Convex)),
            DbSample::Pair => self.collect_pair().map(|(ids, d)| (ids, d, DbSample::Pair)),
            // Bridge is not a sampler: its two endpoint windows are built by
            // bridge_round directly. The geometry coin never draws it.
            DbSample::Bridge => None,
        }
    }

    /// Rebuild the generation pool: the `pool_k` lowest-generation gates among
    /// those that are pool-eligible AND still below the goal. An O(size) scan,
    /// so it runs on the `gen_rescan` cadence rather than every round.
    pub(super) fn rebuild_pool(&mut self) {
        let target = self.params.gen_target;
        self.pool.clear();
        let mut cands: Vec<(u32, u32)> = Vec::new();
        // Walk the linked list directly (same order as ids_in_order) without
        // materializing the O(size) id vector.
        let mut id = self.arena.head();
        while id != NIL {
            let m = self.meta_of(id);
            if m.dgen < target && self.pool_eligible(id) {
                cands.push((m.dgen, id));
            }
            id = self.arena.neighbor(id, Dir::R);
        }
        let k = self.params.pool_k.max(1);
        if cands.len() > k {
            cands.select_nth_unstable(k - 1);
            cands.truncate(k);
        }
        self.pool.extend(cands.into_iter().map(|(_, id)| id));
    }

    /// Draw from the pool, pruning entries that went stale since the rebuild
    /// (freed, re-encoded past the goal, or now too wide to seed).
    pub(super) fn draw_pool(&mut self) -> Option<u32> {
        let target = self.params.gen_target;
        for _ in 0..8 {
            if self.pool.is_empty() {
                return None;
            }
            let i = self.rng.random_range(0..self.pool.len());
            let id = self.pool[i];
            let m = self.meta_of(id);
            if !self.arena.is_linked(id) || m.dgen >= target || !self.pool_eligible(id) {
                self.pool.swap_remove(i);
                continue;
            }
            return Some(id);
        }
        None
    }

    /// True g57 shape: `comp = 1`, two controls, opposite polarity.
    pub(super) fn gate_is_g57(&self, id: u32) -> bool {
        let g = self.arena.gate(id);
        g.comp && g.ctrls.len() == 2 && g.ctrls[0].1 != g.ctrls[1].1
    }

    /// May this gate sit inside the window currently being built? The width cap
    /// always applies; a g57-only COMP attempt additionally excludes everything
    /// that is not a g57, so the window cannot contain the intruder that
    /// collapses the long-window match rate.
    pub(super) fn window_eligible(&self, id: u32) -> bool {
        let cap = self.params.w_window;
        if cap > 0 && self.width_of(id) >= cap {
            return false;
        }
        !self.db_g57_only || self.gate_is_g57(id)
    }

    /// May this gate seed a window and count toward the dose? Stricter than
    /// `window_eligible`: an ineligible gate can never be re-encoded, so its
    /// generation is pinned forever and an unfiltered pool converges on exactly
    /// that set and stays there.
    pub(super) fn pool_eligible(&self, id: u32) -> bool {
        let cap = self.params.w_pool;
        cap == 0 || self.width_of(id) < cap
    }

    pub(super) fn pick_seed(&mut self) -> Option<u32> {
        let g = self.pick_seed_inner();
        self.db_seed_home = g.map(|id| (id, self.arena.neighbor(id, Dir::L)));
        g
    }

    /// Put the seed back where it was drawn from. Retracing a float path the
    /// gate already passed through, exactly as `retreat` does for a declined
    /// cross -- the intervening gates commute with it by construction.
    pub(super) fn restore_seed(&mut self) {
        let Some((id, home)) = self.db_seed_home.take() else {
            return;
        };
        if !self.arena.is_linked(id) || self.arena.neighbor(id, Dir::L) == home {
            return;
        }
        if home != NIL && !self.arena.is_linked(home) {
            return; // its anchor was consumed; leave it where it is
        }
        // Hop back toward home ONE gate at a time, and only over neighbours the
        // seed commutes with. An unchecked relink is wrong here even though the
        // seed floated in along this path: `retreat` may reverse a float
        // immediately, with nothing intervening, but a window build moves OTHER
        // gates too -- ctrl-cap evasion parks a collider out of the way, and an
        // evaded collider is by definition one that does NOT commute. Teleport
        // the seed across that and the circuit's function changes.
        for dir in [Dir::L, Dir::R] {
            for _ in 0..Self::RESTORE_HOPS {
                if self.arena.neighbor(id, Dir::L) == home {
                    return;
                }
                let nb = self.arena.neighbor(id, dir);
                if nb == NIL {
                    break;
                }
                if XGate::collides(self.arena.gate(id), self.arena.gate(nb)) {
                    break; // blocked: leave it here rather than jump the gate
                }
                self.arena.unlink(id);
                match dir {
                    Dir::R => self.arena.link_after(id, nb),
                    Dir::L => self.arena.link_before(id, nb),
                }
            }
        }
    }

    /// Bound on the checked walk home. A seed that cannot get back within this
    /// many hops stays where it is: a partly-restored seed is still correct,
    /// only less tidy.
    const RESTORE_HOPS: usize = 512;

    /// One coin, one pool. Heads (probability `p_mingen`) draws from the
    /// generation pool; tails draws uniformly. `seed_from_pool` records which,
    /// because the canary must count only rounds that genuinely came from the
    /// pool -- a heads round that fell through because the pool had drained is
    /// a DIFFERENT failure (rebuild too slow) with the opposite remedy, and
    /// conflating them would let the brake or a slow rescan look like
    /// unreachable material.
    pub(super) fn pick_seed_inner(&mut self) -> Option<u32> {
        self.seed_from_pool = false;
        self.seed_fell_through = false;
        if self.params.gen_target > 0
            && self.active_p_mingen() > 0.0
            && self.rng.random_bool(self.active_p_mingen().clamp(0.0, 1.0))
        {
            if let Some(id) = self.draw_pool() {
                self.seed_from_pool = true;
                return Some(id);
            }
            self.seed_fell_through = true;
        }
        for _ in 0..8 {
            let g = self.arena.random_linked(&mut self.rng);
            if self.window_eligible(g) {
                return Some(g);
            }
        }
        None
    }

    // Generation of a split child (see MixParams::gen_split_inherit):
    // ratchet semantics give parent + 1, inherit semantics keep the parent
    // generation; MAXGEN stays MAXGEN either way (saturating).
    pub(crate) fn child_gen(&self, parent: u32) -> u32 {
        if self.params.gen_split_inherit {
            parent
        } else {
            parent.saturating_add(1)
        }
    }

    // Hard per-attempt bound on ctrl-cap evasion floats. Evading a wide gate
    // "succeeds" whenever it floats at least one step, so a window build whose
    // colliders are all wide can ping-pong between receding walls doing
    // unbounded arena work while the collected count never grows (observed as
    // a flat-RSS 100%-CPU livelock). Legitimate builds use a handful of
    // evasions; past this budget the attempt aborts and the round is spent.
    const EVADE_BUDGET: usize = 128;

    // Contiguous: g plus its w-1 neighbors in g's direction, spilling to the
    // other direction at the circuit end. A candidate with > L controls is first
    // floated out of the way; if it cannot float, the build reverses direction;
    // if that side is also blocked, the attempt aborts.
    pub(super) fn collect_contiguous(&mut self, w: usize) -> Option<(Vec<u32>, Dir)> {
        let g1 = self.pick_seed()?;
        let dir1 = self.meta_of(g1).dir;
        let (mut lo, mut hi) = (g1, g1);
        let mut count = 1usize;
        let mut dir = dir1;
        let mut switched = false;
        let mut evade_budget = Self::EVADE_BUDGET;
        while count < w {
            let end = if dir == Dir::R { hi } else { lo };
            let x = self.arena.neighbor(end, dir);
            if x == NIL {
                if !switched {
                    dir = dir1.opposite();
                    switched = true;
                    continue;
                }
                break; // both ends reached the circuit boundary
            }
            if !self.window_eligible(x) {
                if evade_budget == 0 {
                    self.counters.db_build_aborts += 1;
                    return None;
                }
                evade_budget -= 1;
                if self.float_to_collision(x, dir) > 0 {
                    continue; // floated the wide gate out of the slot; retry
                }
                if !switched {
                    dir = dir1.opposite();
                    switched = true;
                    continue;
                }
                return None; // wide gate unavoidable on both sides -> abort
            }
            if dir == Dir::R {
                hi = x;
            } else {
                lo = x;
            }
            count += 1;
        }
        let ids = self.span_ids(lo, hi);
        if ids.is_empty() {
            return None;
        }
        Some((ids, dir1))
    }

    // Convex: float g1 to its first collider, then grow the block by floating it
    // (in dir1 w.p. p, else the opposite) to the next collider and absorbing it.
    // The L-cap evades a wide collider the same way (float it away; else reverse;
    // else abort).
    pub(super) fn collect_convex(&mut self, w: usize) -> Option<(Vec<u32>, Dir)> {
        let p = self.params.db_convex_p;
        let g1 = self.pick_seed()?;
        let dir1 = self.meta_of(g1).dir;
        self.float_to_collision(g1, dir1);
        let (mut lo, mut hi) = (g1, g1);
        let mut count = 1usize;
        let mut evade_budget = Self::EVADE_BUDGET;

        while count < w {
            // The first collider is reached in g1's own direction (spec: float g1
            // in dir1 to hit g2); later steps randomize direction by p.
            let want = if count == 1 {
                dir1
            } else if self.rng.random_bool(p) {
                dir1
            } else {
                dir1.opposite()
            };
            // Float the block toward `want` to the next collider; if that side is
            // a boundary, try the opposite direction once.
            let (mut g3, mut dir) = self.float_block_to_collider(lo, hi, want);
            if g3 == NIL {
                let (g3b, dirb) = self.float_block_to_collider(lo, hi, want.opposite());
                if g3b == NIL {
                    break; // block is convex-maximal (both ends commute to the wall)
                }
                g3 = g3b;
                dir = dirb;
            }
            // L-cap: evade a wide collider.
            if !self.window_eligible(g3) {
                if evade_budget == 0 {
                    self.counters.db_build_aborts += 1;
                    return None;
                }
                evade_budget -= 1;
                if self.float_to_collision(g3, dir) > 0 {
                    continue; // g3 floated away; re-float the block to the next collider
                }
                // Reverse: look for a collider on the other side instead.
                let (g3r, dirr) = self.float_block_to_collider(lo, hi, dir.opposite());
                if g3r == NIL || !self.window_eligible(g3r) {
                    return None; // wide gate unavoidable -> abort
                }
                g3 = g3r;
                dir = dirr;
            }
            if dir == Dir::R {
                hi = g3;
            } else {
                lo = g3;
            }
            count += 1;
        }
        let ids = self.span_ids(lo, hi);
        if ids.is_empty() {
            return None;
        }
        Some((ids, dir1))
    }

    // Pair: fuse the seed with a far COMMUTING partner (docs/NONLOCAL_PHASE_A.md).
    // Scan the seed's commutation box — the gates it could float past in its
    // own direction, out to the first collider or pair_scan_cap — then float
    // the seed adjacent to the chosen partner and return the fused 2-gate
    // window. Every crossed gate commutes with the seed (that is what the scan
    // checked), so the relocation is function-preserving like every other
    // float, and a later miss is walked home by db_attempt's usual
    // restore_seed. The other samplers cannot produce this window: Convex
    // absorbs only COLLIDERS (commuting gates are hopped past), and Contiguous
    // pairs commuting gates only when they already sit at distance 1.
    pub(super) fn collect_pair(&mut self) -> Option<(Vec<u32>, Dir)> {
        let g1 = self.pick_seed()?;
        let dir1 = self.meta_of(g1).dir;
        let cap = self.params.pair_scan_cap.max(1);
        // Read-only box scan: partner candidates are window-eligible gates the
        // seed commutes with, recorded with their hop distance (gates crossed
        // to reach them).
        let mut cands: Vec<(u32, u64)> = Vec::new();
        let mut cur = self.arena.neighbor(g1, dir1);
        let mut hops = 0u64;
        while cur != NIL && (hops as usize) < cap && !self.arena.collides_ids(g1, cur) {
            if self.window_eligible(cur) {
                cands.push((cur, hops));
            }
            hops += 1;
            cur = self.arena.neighbor(cur, dir1);
        }
        if cur != NIL && hops as usize >= cap {
            self.counters.pair_scan_truncs += 1;
        }
        if cands.is_empty() {
            self.counters.pair_boxes_empty += 1;
            return None;
        }
        let (g2, dist) = if self.params.pair_pick_uniform {
            cands[self.rng.random_range(0..cands.len())]
        } else {
            *cands.last().unwrap()
        };
        // Fuse: float the seed adjacent to the partner. The scan cleared the
        // path, so float_until stops with the partner as the next neighbor.
        self.float_until(g1, dir1, g2);
        if self.arena.neighbor(g1, dir1) != g2 {
            // Unreachable while the scan and float_until agree on collides();
            // decline defensively rather than fuse a wrong window.
            self.counters.pair_boxes_empty += 1;
            return None;
        }
        self.counters.pair_fused += 1;
        self.counters.pair_box_sum += hops;
        self.counters.pair_box_max = self.counters.pair_box_max.max(hops);
        self.counters.pair_dist_sum += dist;
        self.counters.pair_dist_max = self.counters.pair_dist_max.max(dist);
        // Window ids in link order, leftmost first.
        let ids = match dir1 {
            Dir::R => vec![g1, g2],
            Dir::L => vec![g2, g1],
        };
        Some((ids, dir1))
    }
}
