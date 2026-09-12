//! Commuting transport and post-splice advancement over the shared arena.
use super::*;

impl Mixer {
    // ---- floating (same semantics as the fsplit engine) ----

    pub(super) fn float_distance(&self, id: u32, dir: Dir, cap: usize) -> usize {
        let mut cur = self.arena.neighbor(id, dir);
        let mut d = 0usize;
        while cur != NIL && d < cap && !self.arena.collides_ids(id, cur) {
            d += 1;
            cur = self.arena.neighbor(cur, dir);
        }
        d
    }

    pub(crate) fn float_to_collision(&mut self, id: u32, dir: Dir) -> usize {
        self.float_until(id, dir, NIL)
    }

    // Slide `id` in `dir` past non-colliders, stopping early if the next node is
    // `stop`. Needed for merges: merge partners never collide (same target), so
    // an unbounded float would sail straight past the partner.
    pub(crate) fn float_until(&mut self, id: u32, dir: Dir, stop: u32) -> usize {
        // Scan with borrows only (no gate clone); mutate after the scan scope.
        let (last, steps) = {
            let mut last = NIL;
            let mut cur = self.arena.neighbor(id, dir);
            let mut steps = 0usize;
            while cur != NIL && cur != stop && !self.arena.collides_ids(id, cur) {
                last = cur;
                steps += 1;
                cur = self.arena.neighbor(cur, dir);
            }
            (last, steps)
        };
        if steps > 0 {
            self.arena.unlink(id);
            match dir {
                Dir::R => self.arena.link_after(id, last),
                Dir::L => self.arena.link_before(id, last),
            }
            self.counters.floats += 1;
            self.counters.float_steps += steps as u64;
        }
        steps
    }

    pub(super) fn float_uniform(&mut self, id: u32) -> usize {
        let dl = self.float_distance(id, Dir::L, usize::MAX);
        let dr = self.float_distance(id, Dir::R, usize::MAX);
        if dl + dr == 0 {
            return 0;
        }
        let off = self.rng.random_range(0..=(dl + dr));
        let (dir, k) = if off < dl {
            (Dir::L, dl - off)
        } else {
            (Dir::R, off - dl)
        };
        if k == 0 {
            return 0;
        }
        let mut anchor = id;
        for _ in 0..k {
            anchor = self.arena.neighbor(anchor, dir);
        }
        self.arena.unlink(id);
        match dir {
            Dir::L => self.arena.link_before(id, anchor),
            Dir::R => self.arena.link_after(id, anchor),
        }
        k
    }

    pub(crate) fn rand_dir(&mut self) -> Dir {
        if self.rng.random_bool(0.5) {
            Dir::L
        } else {
            Dir::R
        }
    }

    // Directional birth transport (replaces the uniform scatter): a fresh
    // piece advances floor(dir_q * slack) gates in its own direction, where
    // slack is how far it could float that way before its first collision.
    pub(super) fn advance_birth(&mut self, id: u32) {
        if !self.arena.is_linked(id) {
            return;
        }
        let dir = self.meta_of(id).dir;
        let slack = self.float_distance(id, dir, usize::MAX);
        let k = (self.params.dir_q * slack as f64).floor() as usize;
        if k == 0 {
            return;
        }
        let mut anchor = id;
        for _ in 0..k {
            anchor = self.arena.neighbor(anchor, dir);
        }
        self.arena.unlink(id);
        match dir {
            Dir::L => self.arena.link_before(id, anchor),
            Dir::R => self.arena.link_after(id, anchor),
        }
        self.counters.scatters += 1;
        self.counters.scatter_steps += k as u64;
    }

    pub(crate) fn advance_births(&mut self, ids: &[u32]) {
        for &id in ids {
            self.advance_birth(id);
        }
    }

    pub(super) fn float_block_to_collider(&mut self, lo: u32, hi: u32, dir: Dir) -> (u32, Dir) {
        loop {
            let end = if dir == Dir::R { hi } else { lo };
            let x = self.arena.neighbor(end, dir);
            if x == NIL {
                return (NIL, dir);
            }
            if self.span_collides(lo, hi, x) {
                return (x, dir);
            }
            self.move_across(x, lo, hi, dir);
        }
    }

    /// GLOBAL RE-RANDOMISATION: float every gate to a uniformly random position
    /// inside its own commutation bounds.
    ///
    /// This is the whole-circuit analogue of the per-splice birth advance. Where
    /// `--db-advance` scatters one litter at birth, this re-draws the position of
    /// every gate at once, so positional structure inherited from the input --
    /// which the ancestry instruments show is what survives DB mixing -- is
    /// attacked directly rather than incidentally.
    ///
    /// Function preservation is free: each step is a float past non-colliding
    /// gates, i.e. a commutation, so no verification is needed. `float_uniform`
    /// picks the offset over the gate's full [left, right] slack, so a gate with
    /// no slack simply stays put.
    ///
    /// Cost is O(sum of per-gate slack), which is why the round rate is scaled
    /// as 1/|circuit|: the expected work per round stays O(mean slack) no matter
    /// how large the circuit grows.
    // GLOBAL re-randomisation: re-place EVERY gate uniformly inside its own
    // commutation bounds. This is exactly the terminal float applied mid-run,
    // so it is semantics-preserving (float_distance never crosses a
    // non-commuting neighbour) and size-preserving -- it moves gates and
    // nothing else. The walk is sequential, so a gate's bounds already
    // reflect the earlier gates' new positions; that is the same law
    // final_float has always used.
    pub(super) fn global_shuffle(&mut self) {
        let t0 = std::time::Instant::now();
        let (moved, disp) = self.final_float();
        self.counters.shuffle_ns += t0.elapsed().as_nanos() as u64;
        self.counters.shuffles += 1;
        self.counters.shuffle_moved += moved;
        self.counters.shuffle_steps += disp;
    }

    pub fn final_float(&mut self) -> (u64, u64) {
        let ids = self.arena.ids_in_order();
        let (mut moved, mut disp) = (0u64, 0u64);
        for id in ids {
            let k = self.float_uniform(id);
            if k > 0 {
                moved += 1;
                disp += k as u64;
            }
        }
        (moved, disp)
    }
}
