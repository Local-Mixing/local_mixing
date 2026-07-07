// The float-and-split episode loop.
//
// Episode: pick a starting gate by sampling candidates and keeping the one with
// the largest one-directional float distance (prefers big boxes AND gates at one
// end of their box — an end-of-box gate travels its whole box), float it to its
// collision point, split it past the colliding gate under the K-cap, then keep
// floating shot pieces in the same direction and colliding pieces in the other,
// until the worklist drains. Episodes repeat until the size bound B or
// K-saturation; a final pass floats every gate to a uniform random position in
// its full two-sided commutable box.
use super::arena::{Arena, Dir, NIL};
use super::rules::{self, BlockReason, Outcome, Role, RuleKind};
use super::xgate::XGate;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::VecDeque;

pub struct Params {
    pub k_max: usize,
    // Width-damping offset D: a gate with c controls splits with probability
    // min(2^(-(c-D)), 1). D=2: c<=2 always, c=3 half, c=4 quarter, ...
    pub split_damp: usize,
    pub size_bound: usize,
    pub candidates: usize,
    pub walk_cap: usize,
    pub episode_cap: usize,
    pub verify_every: usize,
    pub report_every: usize,
    pub saturation_patience: usize,
    pub local_verify: bool,
    pub seed: u64,
}

impl Default for Params {
    fn default() -> Params {
        Params {
            k_max: 4,
            split_damp: 2,
            size_bound: usize::MAX,
            candidates: 64,
            walk_cap: 4096,
            episode_cap: 10_000,
            verify_every: 64,
            report_every: 1000,
            saturation_patience: 200,
            local_verify: true,
            seed: 0,
        }
    }
}

#[derive(Default)]
pub struct Counters {
    pub episodes: u64,
    pub pops: u64,
    pub floats: u64,
    pub float_steps: u64,
    pub splits_r1: u64,
    pub splits_r2: u64,
    pub splits_r3: u64,
    pub presplit_shot: u64,
    pub presplit_colliding: u64,
    pub blocked_width: u64,
    pub blocked_deadlock: u64,
    pub split_declined: u64,
    pub scatters: u64,
    pub scatter_steps: u64,
    pub g57_shot_blocked: u64,
    pub dropped_neverfire: u64,
    pub cores: u64,
    pub boundary_retire: u64,
    pub stale_skips: u64,
    pub width_hist: [u64; 16],
}

// One visualization/debug snapshot: the full circuit after a mutation event,
// with the indices involved. Gates are dumped compactly as
// (target, comp, [(wire, positive)...]).
pub struct TraceStep {
    pub kind: &'static str, // init|float|presplit|presplit_h|r1|r2|r3|scatter|skipped|blocked|final_float
    pub label: String,
    pub gates: Vec<(u16, u8, Vec<(u16, u8)>)>,
    pub moves: Vec<(usize, usize)>, // floated gates: (index before, index after)
    pub colliding: Option<usize>,
    pub new_idx: Vec<usize>,
    pub core_idx: Vec<usize>,
}

pub struct Engine {
    pub arena: Arena,
    pub params: Params,
    pub counters: Counters,
    pub trace: Option<Vec<TraceStep>>,
    pub trace_cap: usize,
    original: Vec<XGate>,
    num_wires: usize,
    rng: StdRng,
}

pub enum StopReason {
    SizeBound,
    Saturated,
    TraceFull,
}

impl Engine {
    pub fn new(gates: Vec<XGate>, params: Params) -> Engine {
        let num_wires = super::xgate::max_wire(&gates) as usize + 1;
        let rng = StdRng::seed_from_u64(params.seed);
        Engine {
            arena: Arena::from_gates(gates.clone()),
            params,
            counters: Counters::default(),
            trace: None,
            trace_cap: 0,
            original: gates,
            num_wires,
            rng,
        }
    }

    // ---- tracing (visualization/debug; off unless `trace` is Some) ----

    pub fn trace_on(&mut self, cap: usize) {
        self.trace = Some(Vec::with_capacity(cap + 1));
        self.trace_cap = cap;
        self.emit_trace("init", "initial circuit".to_string(), vec![], None, &[], &[]);
    }

    fn trace_full(&self) -> bool {
        self.trace.as_ref().is_some_and(|t| t.len() > self.trace_cap)
    }

    fn emit_trace(
        &mut self,
        kind: &'static str,
        label: String,
        moves: Vec<(usize, usize)>,
        colliding: Option<u32>,
        new_ids: &[u32],
        core_ids: &[u32],
    ) {
        if self.trace.is_none() {
            return;
        }
        let colliding = colliding.map(|id| self.arena.index_of(id));
        let new_idx: Vec<usize> = new_ids.iter().map(|&i| self.arena.index_of(i)).collect();
        let core_idx: Vec<usize> = core_ids.iter().map(|&i| self.arena.index_of(i)).collect();
        let gates = self
            .arena
            .ids_in_order()
            .iter()
            .map(|&id| {
                let g = self.arena.gate(id);
                (g.target, g.comp as u8, g.ctrls.iter().map(|&(w, p)| (w, p as u8)).collect())
            })
            .collect();
        self.trace
            .as_mut()
            .unwrap()
            .push(TraceStep { kind, label, gates, moves, colliding, new_idx, core_idx });
    }

    fn structural_changes(&self) -> u64 {
        let c = &self.counters;
        c.splits_r1 + c.splits_r2 + c.splits_r3 + c.presplit_shot + c.presplit_colliding
    }

    pub fn run(&mut self) -> StopReason {
        let mut idle = 0usize;
        loop {
            if self.trace_full() {
                self.global_check();
                return StopReason::TraceFull;
            }
            if self.arena.len() >= self.params.size_bound {
                self.global_check();
                return StopReason::SizeBound;
            }
            // Saturation is measured by STRUCTURAL progress (splits/presplits).
            // Floats alone must not reset the counter: a fully K-blocked circuit
            // can float forever without ever growing.
            let before = self.structural_changes();
            self.episode();
            self.counters.episodes += 1;
            if self.structural_changes() > before {
                idle = 0;
            } else {
                idle += 1;
                if idle >= self.params.saturation_patience {
                    self.global_check();
                    return StopReason::Saturated;
                }
            }
            if self.counters.episodes % self.params.verify_every as u64 == 0 {
                self.global_check();
            }
            if self.counters.episodes % self.params.report_every as u64 == 0 {
                self.report();
            }
        }
    }

    // ---- selection ----

    // Float distance in `dir` without moving, capped.
    fn float_distance(&self, id: u32, dir: Dir, cap: usize) -> usize {
        let g = self.arena.gate(id);
        let mut cur = self.arena.neighbor(id, dir);
        let mut d = 0usize;
        while cur != NIL && d < cap && !XGate::collides(g, self.arena.gate(cur)) {
            d += 1;
            cur = self.arena.neighbor(cur, dir);
        }
        d
    }

    fn select_start(&mut self) -> Option<(u32, Dir)> {
        let mut best: Option<(u32, Dir, usize)> = None;
        for _ in 0..self.params.candidates {
            let id = self.arena.random_linked(&mut self.rng);
            let dl = self.float_distance(id, Dir::L, self.params.walk_cap);
            let dr = self.float_distance(id, Dir::R, self.params.walk_cap);
            let (score, dir) = if dl > dr {
                (dl, Dir::L)
            } else if dr > dl {
                (dr, Dir::R)
            } else if self.rng.random_bool(0.5) {
                (dl, Dir::L)
            } else {
                (dr, Dir::R)
            };
            if score > best.map_or(0, |(_, _, s)| s) {
                best = Some((id, dir, score));
            }
        }
        best.map(|(id, dir, _)| (id, dir))
    }

    // ---- floating ----

    // Slide `id` in `dir` past every non-colliding neighbor; returns steps taken.
    // After this, the `dir`-neighbor (if any) collides with `id`.
    fn float_to_collision(&mut self, id: u32, dir: Dir) -> usize {
        let g = self.arena.gate(id).clone();
        let mut last = NIL;
        let mut cur = self.arena.neighbor(id, dir);
        let mut steps = 0usize;
        while cur != NIL && !XGate::collides(&g, self.arena.gate(cur)) {
            last = cur;
            steps += 1;
            cur = self.arena.neighbor(cur, dir);
        }
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

    // ---- splicing ----

    // Replace node `id` by `gates` at its position; returns the fresh ids.
    fn splice_replace_one(&mut self, id: u32, gates: Vec<XGate>) -> Vec<u32> {
        let mut cursor = self.arena.neighbor(id, Dir::L);
        self.arena.unlink(id);
        self.arena.free_node(id);
        let mut ids = Vec::with_capacity(gates.len());
        for g in gates {
            cursor = self.arena.insert_after(cursor, g);
            ids.push(cursor);
        }
        ids
    }

    // Replace the adjacent pair (shot `g_id`, colliding `h_id`, orientation per
    // `dir`) by the rewrite sequence. `seq` is in rightward form; leftward
    // crossings emit it reversed (all gates are involutions, so reversing both
    // sides of the identity is exact). The CollidingIntact entry reuses h's node
    // so a queued reference to h stays valid.
    fn splice_pair(&mut self, g_id: u32, h_id: u32, dir: Dir, seq: Vec<(XGate, Role)>) -> Vec<(u32, Role)> {
        let first = match dir {
            Dir::R => g_id,
            Dir::L => h_id,
        };
        let mut cursor = self.arena.neighbor(first, Dir::L);
        self.arena.unlink(g_id);
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
                self.arena.insert_after(cursor, gate)
            };
            cursor = id;
            out.push((id, role));
        }
        self.arena.free_node(g_id);
        if !h_reused {
            self.arena.free_node(h_id);
        }
        out
    }

    // ---- the episode ----

    fn episode(&mut self) -> bool {
        let Some((start, dir)) = self.select_start() else {
            return false;
        };
        // select_start only returns candidates with a positive float distance,
        // but sampling can still land all-wedged; distance 0 means no episode.
        if self.float_distance(start, dir, 1) == 0 {
            return false;
        }
        let mut acted = false;
        let mut wl: VecDeque<(u32, u32, Dir)> = VecDeque::new();
        wl.push_back((start, self.arena.stamp(start), dir));
        let mut pops = 0usize;
        while let Some((id, stamp, dir)) = wl.pop_front() {
            if self.arena.len() >= self.params.size_bound || self.trace_full() {
                break;
            }
            pops += 1;
            if pops > self.params.episode_cap {
                break;
            }
            if !self.arena.is_linked(id) || self.arena.stamp(id) != stamp {
                self.counters.stale_skips += 1;
                continue;
            }
            self.counters.pops += 1;

            let pre_idx = if self.trace.is_some() { self.arena.index_of(id) } else { 0 };
            let fsteps = self.float_to_collision(id, dir);
            if fsteps > 0 {
                acted = true;
            }
            let arrow = if dir == Dir::R { "right" } else { "left" };
            let moved: Vec<(usize, usize)> = if fsteps > 0 && self.trace.is_some() {
                vec![(pre_idx, self.arena.index_of(id))]
            } else {
                vec![]
            };
            let h_id = self.arena.neighbor(id, dir);
            if h_id == NIL {
                self.counters.boundary_retire += 1;
                if fsteps > 0 {
                    self.emit_trace(
                        "float",
                        format!("floated {fsteps} {arrow} to the circuit boundary; retired"),
                        moved,
                        None,
                        &[],
                        &[],
                    );
                }
                continue;
            }
            let g = self.arena.gate(id).clone();
            let h = self.arena.gate(h_id).clone();

            if g.comp {
                // Shot g57: pre-split only if at least one piece can make progress.
                let pieces = rules::presplit(&g, &mut self.rng);
                let useful = pieces.iter().any(|p| {
                    !XGate::collides(p, &h)
                        || matches!(
                            rules::cross(p, &h, self.params.k_max, &mut self.rng),
                            Outcome::Rewrite { .. } | Outcome::PresplitColliding | Outcome::R0Swap
                        )
                });
                if !useful {
                    self.counters.g57_shot_blocked += 1;
                    if fsteps > 0 {
                        self.emit_trace(
                            "blocked",
                            format!("g57 floated {fsteps} {arrow}, hit a colliding gate; both pieces would be blocked (K-cap), retired intact"),
                            moved,
                            Some(h_id),
                            &[],
                            &[],
                        );
                    }
                    continue;
                }
                if !self.split_allowed(g.width()) {
                    self.counters.split_declined += 1;
                    if fsteps > 0 {
                        self.emit_trace(
                            "skipped",
                            format!("g57 floated {fsteps} {arrow}; split declined by width damping ({} controls); retired", g.width()),
                            moved,
                            Some(h_id),
                            &[],
                            &[],
                        );
                    }
                    continue;
                }
                if self.params.local_verify {
                    assert!(
                        rules::verify_rewrite(std::slice::from_ref(&g), &pieces),
                        "presplit verification failed: {g:?} -> {pieces:?}"
                    );
                }
                for p in &pieces {
                    self.counters.width_hist[p.width().min(15)] += 1;
                }
                let n_pieces = pieces.len();
                let ids = self.splice_replace_one(id, pieces);
                self.emit_trace(
                    "presplit",
                    format!("shot g57 floated {fsteps} {arrow} to its collision, pre-split into {n_pieces} exclusive conjunction pieces"),
                    moved,
                    Some(h_id),
                    &ids,
                    &[],
                );
                self.scatter(&ids, "the pre-split pieces");
                for pid in &ids {
                    wl.push_back((*pid, self.arena.stamp(*pid), dir));
                }
                self.counters.presplit_shot += 1;
                acted = true;
                continue;
            }

            match rules::cross(&g, &h, self.params.k_max, &mut self.rng) {
                Outcome::R0Swap => {
                    // float_to_collision stopped here, so this should not occur.
                    debug_assert!(false, "R0 after floating to collision");
                }
                Outcome::Blocked(reason) => {
                    let why = match reason {
                        BlockReason::WidthCap => {
                            self.counters.blocked_width += 1;
                            "a residue would exceed K controls"
                        }
                        BlockReason::Deadlock => {
                            self.counters.blocked_deadlock += 1;
                            "R3 deadlock (nothing can cross)"
                        }
                    };
                    if fsteps > 0 {
                        self.emit_trace(
                            "blocked",
                            format!("piece floated {fsteps} {arrow}, blocked: {why}; retired"),
                            moved,
                            Some(h_id),
                            &[],
                            &[],
                        );
                    }
                    continue;
                }
                Outcome::PresplitColliding => {
                    let hp = rules::presplit(&h, &mut self.rng);
                    if !self.split_allowed(h.width()) {
                        self.counters.split_declined += 1;
                        if fsteps > 0 {
                            self.emit_trace(
                                "skipped",
                                format!("shot floated {fsteps} {arrow}; colliding g57 split declined by width damping ({} controls); shot retired", h.width()),
                                moved,
                                Some(h_id),
                                &[],
                                &[],
                            );
                        }
                        continue;
                    }
                    if self.params.local_verify {
                        assert!(
                            rules::verify_rewrite(std::slice::from_ref(&h), &hp),
                            "colliding presplit verification failed: {h:?} -> {hp:?}"
                        );
                    }
                    for p in &hp {
                        self.counters.width_hist[p.width().min(15)] += 1;
                    }
                    let n_pieces = hp.len();
                    let hp_ids = self.splice_replace_one(h_id, hp);
                    self.emit_trace(
                        "presplit_h",
                        format!("shot floated {fsteps} {arrow}; colliding g57 must split (R2), pre-split it into {n_pieces} pieces first, retrying"),
                        moved,
                        None,
                        &hp_ids,
                        &[],
                    );
                    self.scatter(&hp_ids, "the colliding gate's pre-split pieces");
                    self.counters.presplit_colliding += 1;
                    // Retry g against whatever is adjacent now.
                    wl.push_front((id, self.arena.stamp(id), dir));
                    acted = true;
                }
                Outcome::Rewrite { seq, kind, dropped } => {
                    // Width damping applies to the gate that would split: the
                    // shot in R1/R3, the colliding gate in R2.
                    let split_width = match kind {
                        RuleKind::R1 | RuleKind::R3 => g.width(),
                        RuleKind::R2 => h.width(),
                    };
                    if !self.split_allowed(split_width) {
                        self.counters.split_declined += 1;
                        if fsteps > 0 {
                            self.emit_trace(
                                "skipped",
                                format!("floated {fsteps} {arrow}; {kind:?} split declined by width damping ({split_width} controls); retired"),
                                moved,
                                Some(h_id),
                                &[],
                                &[],
                            );
                        }
                        continue;
                    }
                    self.counters.dropped_neverfire += dropped as u64;
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
                    match kind {
                        RuleKind::R1 => self.counters.splits_r1 += 1,
                        RuleKind::R2 => self.counters.splits_r2 += 1,
                        RuleKind::R3 => self.counters.splits_r3 += 1,
                    }
                    for (gate, role) in &seq {
                        if *role != Role::CollidingIntact {
                            self.counters.width_hist[gate.width().min(15)] += 1;
                        }
                    }
                    let placed = self.splice_pair(id, h_id, dir, seq);
                    if self.trace.is_some() {
                        let (kind_s, verb) = match kind {
                            RuleKind::R1 => ("r1", "split into the ladder and crossed; colliding gate intact"),
                            RuleKind::R2 => ("r2", "passed intact; colliding gate split and pushed the other way"),
                            RuleKind::R3 => ("r3", "mutual collision: stay-behind core wedged, rest crossed"),
                        };
                        let h_kept: Option<u32> = placed
                            .iter()
                            .find(|(_, r)| *r == Role::CollidingIntact)
                            .map(|&(i, _)| i);
                        let new_ids: Vec<u32> = placed
                            .iter()
                            .filter(|(_, r)| matches!(r, Role::ShotPiece | Role::CollidingPiece))
                            .map(|&(i, _)| i)
                            .collect();
                        let core_ids: Vec<u32> = placed
                            .iter()
                            .filter(|(_, r)| *r == Role::Core)
                            .map(|&(i, _)| i)
                            .collect();
                        self.emit_trace(
                            kind_s,
                            format!(
                                "shot floated {fsteps} {arrow}, then {verb} ({} new pieces, {dropped} never-fire residues dropped)",
                                new_ids.len()
                            ),
                            moved,
                            h_kept,
                            &new_ids,
                            &core_ids,
                        );
                    }
                    let scatter_ids: Vec<u32> = placed
                        .iter()
                        .filter(|(_, r)| matches!(r, Role::ShotPiece | Role::CollidingPiece))
                        .map(|&(i, _)| i)
                        .collect();
                    self.scatter(&scatter_ids, "the new pieces");
                    for (pid, role) in placed {
                        match role {
                            Role::ShotPiece => wl.push_back((pid, self.arena.stamp(pid), dir)),
                            Role::CollidingPiece => {
                                wl.push_back((pid, self.arena.stamp(pid), dir.opposite()))
                            }
                            Role::Core => self.counters.cores += 1,
                            Role::CollidingIntact => {}
                        }
                    }
                    acted = true;
                }
            }
        }
        acted
    }

    // ---- uniform box floats ----

    // Float one gate to a uniform random position in its full two-sided box.
    // Returns the displacement, plus (from, to) indices when tracing.
    fn float_uniform(&mut self, id: u32) -> (usize, Option<(usize, usize)>) {
        let dl = self.float_distance(id, Dir::L, usize::MAX);
        let dr = self.float_distance(id, Dir::R, usize::MAX);
        if dl + dr == 0 {
            return (0, None);
        }
        let off = self.rng.random_range(0..=(dl + dr));
        let (dir, k) = if off < dl { (Dir::L, dl - off) } else { (Dir::R, off - dl) };
        if k == 0 {
            return (0, None);
        }
        let from = if self.trace.is_some() { self.arena.index_of(id) } else { 0 };
        // Walk to the node currently k positions away, then splice around it.
        let mut anchor = id;
        for _ in 0..k {
            anchor = self.arena.neighbor(anchor, dir);
        }
        self.arena.unlink(id);
        match dir {
            Dir::L => self.arena.link_before(id, anchor),
            Dir::R => self.arena.link_after(id, anchor),
        }
        let ft = if self.trace.is_some() { Some((from, self.arena.index_of(id))) } else { None };
        (k, ft)
    }

    // Post-event scatter: float each fresh piece of g/h to a uniform random
    // position in its box, so the next crossing site decorrelates from this one.
    fn scatter(&mut self, ids: &[u32], what: &'static str) {
        let mut moves: Vec<(usize, usize)> = Vec::new();
        let mut disp = 0usize;
        let mut moved = 0usize;
        for &id in ids {
            if !self.arena.is_linked(id) {
                continue;
            }
            let (k, ft) = self.float_uniform(id);
            if k > 0 {
                moved += 1;
                disp += k;
                if let Some(m) = ft {
                    moves.push(m);
                }
            }
        }
        self.counters.scatters += moved as u64;
        self.counters.scatter_steps += disp as u64;
        if moved > 0 && self.trace.is_some() {
            let ids_now: Vec<u32> = ids.iter().copied().filter(|&i| self.arena.is_linked(i)).collect();
            self.emit_trace(
                "scatter",
                format!("{what} floated to uniform random positions in their boxes ({moved} moved, {disp} total displacement)"),
                moves,
                None,
                &ids_now,
                &[],
            );
        }
    }

    // Width-damped split decision: a gate with c controls splits with
    // probability min(2^(-(c - split_damp)), 1); at or below split_damp it
    // always splits.
    fn split_allowed(&mut self, c: usize) -> bool {
        let d = self.params.split_damp;
        if c <= d {
            return true;
        }
        self.rng.random_bool(1.0 / (1u64 << (c - d).min(62)) as f64)
    }

    // ---- final uniform float ----

    // One pass; for each gate one uniform draw over its full two-sided box.
    pub fn final_float(&mut self) -> (u64, u64) {
        let ids = self.arena.ids_in_order();
        let (mut moved, mut disp) = (0u64, 0u64);
        for id in ids {
            let (k, _) = self.float_uniform(id);
            if k > 0 {
                moved += 1;
                disp += k as u64;
            }
        }
        (moved, disp)
    }

    // ---- verification & reporting ----

    // Sampled functional equality vs the ORIGINAL input circuit, on all wires.
    pub fn global_check(&mut self) {
        let batches = 4;
        for _ in 0..batches {
            let mut st_orig: Vec<u64> = (0..self.num_wires).map(|_| self.rng.random()).collect();
            let mut st_cur = st_orig.clone();
            super::xgate::eval_lanes(&self.original, &mut st_orig);
            let mut cur = self.arena.head();
            while cur != NIL {
                self.arena.gate(cur).apply_lanes(&mut st_cur);
                cur = self.arena.neighbor(cur, Dir::R);
            }
            assert_eq!(
                st_orig, st_cur,
                "FUNCTIONALITY BROKEN: circuit no longer equals the input (episode {})",
                self.counters.episodes
            );
        }
    }

    pub fn remaining_g57(&self) -> usize {
        let mut cur = self.arena.head();
        let mut n = 0usize;
        while cur != NIL {
            if self.arena.gate(cur).comp {
                n += 1;
            }
            cur = self.arena.neighbor(cur, Dir::R);
        }
        n
    }

    pub fn report(&self) {
        let c = &self.counters;
        let hist: Vec<String> = (0..=self.params.k_max.min(15))
            .map(|w| format!("{}:{}", w, c.width_hist[w]))
            .collect();
        println!(
            "[fsplit] ep={} size={} pops={} floats={} steps={} r1={} r2={} r3={} presplit_shot={} presplit_coll={} declined={} scatter={}/{} blocked_k={} deadlock={} g57_blocked={} cores={} drops={} boundary={} stale={} width[{}]",
            c.episodes,
            self.arena.len(),
            c.pops,
            c.floats,
            c.float_steps,
            c.splits_r1,
            c.splits_r2,
            c.splits_r3,
            c.presplit_shot,
            c.presplit_colliding,
            c.split_declined,
            c.scatters,
            c.scatter_steps,
            c.blocked_width,
            c.blocked_deadlock,
            c.g57_shot_blocked,
            c.cores,
            c.dropped_neverfire,
            c.boundary_retire,
            c.stale_skips,
            hist.join(" ")
        );
    }
}
