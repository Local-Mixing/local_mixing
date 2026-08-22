// Production GSS Stage 4: split every complemented g57 gate, optionally
// join it to a long-range absorbed NOT twist, and finish the move with one
// crossing shot supplied by the sibling `cross_walk` module. Shared arena,
// provenance, and checkpoint state remain on Mixer.
use crate::circuit::xgate::XGate;
use crate::engine::arena::{Dir, NIL};
use crate::engine::mix::{Meta, Mixer, Tap};
use crate::engine::rules;
use rand::Rng;

// Split-stage rank restamp cadence, in moves (docs/FMIX_SPLIT_TWIST.md §5).
const RANK_EVERY: u64 = 8192;

// Flip the polarity of the literal on `w`, if the gate carries one. The
// in-place sibling of `conj_by_not`: exact for comp gates too (the flip
// happens inside the complemented conjunction).
fn flip_wire_literal(g: &mut XGate, w: u16) {
    for l in g.ctrls.iter_mut() {
        if l.0 == w {
            l.1 = !l.1;
        }
    }
}

impl Mixer {
    // ---- the split twist (docs/FMIX_SPLIT_TWIST.md) ----
    //
    // One move: split a random g57 into its presplit pair, then with
    // probability p_join wrap an ABSORBED pure-NOT twist on the g57's target
    // wire between the split's 1-control piece and a bracket found across the
    // circuit (splitting the bracket too when it is a g57, and force-splitting
    // every g57 the segment conjugates), and finish with one ordinary cross
    // shot from the 2-control piece. Every sub-rewrite is function-preserving
    // on its own: presplit is exact, and the twist is the identity
    //   g1' . S' . h1' = g1 . X(w) . S' . X(w) . h1 = g1 . S . h1
    // where ' flips a bracket's control polarity (the absorbed X: a gate
    // targeting w commutes with X(w) and composes into a single gate) and S'
    // flips every w-READING pin in the open segment (gates targeting w are
    // invariant).

    pub(crate) fn split_twist_move(&mut self) {
        // Ranks drive the bracket draw, so refresh them on growth too: the
        // stage roughly doubles the circuit in a few thousand moves, far
        // inside the move cadence.
        if self.moves_done >= self.rank_due || self.arena.len() > self.rank_n + self.rank_n / 4 {
            self.restamp_ranks();
        }
        if !self.taps_planted {
            self.plant_taps();
        }
        // 1. A uniformly random g57; none anywhere = exit A. Outside a live
        // stage (standalone --p-split-twist dispatch) an empty pool is just a
        // spent round — there is no stage to end.
        if self.comp_ids.is_empty() {
            if self.split_on {
                self.end_split_stage("g57 pool exhausted");
            } else {
                self.counters.twist_skips += 1;
            }
            return;
        }
        let g_id = self.comp_ids[self.rng.random_range(0..self.comp_ids.len())];
        let w = self.arena.gate(g_id).target;
        // Twist direction (v3, 2026-08-05): drawn with probability
        // proportional to the circuit length REMAINING on each side of g, so
        // a side is picked exactly as rarely as it is short — the fix for
        // the 0-5% span spike that the own-stored-direction rule produced on
        // edge-adjacent primaries (a tiny span now needs a short side AND
        // the proportional coin to pick it: squared suppression). Stored
        // direction is only the fallback for an unstamped primary.
        let g_rank = self.rank_of(g_id);
        let g_dir = if g_rank != NIL && self.rank_n > 0 {
            let p_right = (self.rank_n as f64 - g_rank as f64) / self.rank_n as f64;
            if self.rng.random_bool(p_right.clamp(0.0, 1.0)) {
                Dir::R
            } else {
                Dir::L
            }
        } else {
            self.meta_of(g_id).dir
        };
        // 2. Split it. g1 = the 1-control rung, g2 = the widest.
        let (g1, g2) = self.split_g57(g_id);
        self.counters.split_prims += 1;
        // 3. The join coin: tails ends the move after the bare split.
        if self
            .rng
            .random_bool(1.0 - self.params.p_join.clamp(0.0, 1.0))
        {
            self.split_report_line(None);
            return;
        }
        // 4. The bracket draw (directional length-biased; reach_k = 0 keeps
        // the original cascade as the A/B arm), then the twist.
        let mut span = None;
        let picked = if self.params.split_reach_k == 0 {
            self.pick_bracket_cascade(w, g1, g_rank)
        } else {
            self.pick_bracket(w, g1, g_rank, g_dir)
        };
        match picked {
            None => {
                self.counters.split_fails += 1;
                self.split_fail_streak += 1;
                if self.split_on && self.split_fail_streak >= self.params.split_fail_limit {
                    self.end_split_stage("failure limit");
                }
            }
            Some((h_id, h_comp, crossed)) => {
                let h1 = if h_comp {
                    self.counters.split_hsplits += 1;
                    self.split_g57(h_id).0
                } else {
                    h_id
                };
                let s = self.apply_not_twist(g1, h1, w);
                self.counters.split_joins += 1;
                self.counters.split_span_sum += s as u64;
                let frac20 = (s * 20) / self.arena.len().max(1);
                self.counters.split_span_hist[frac20.min(19)] += 1;
                span = Some(s);
                if crossed {
                    self.counters.split_xmid += 1;
                }
                self.split_fail_streak = 0;
            }
        }
        // 6. One ordinary cross shot from g2, twist outcome notwithstanding.
        if self.arena.is_linked(g2) {
            self.cross_move_on(g2);
        }
        self.split_report_line(span);
    }

    /// Split a g57 in place by the randomized first-failing-literal presplit
    /// (the literal shuffle IS the design's `r` bit). Pieces stay put — no
    /// birth transport: the 1-control piece must sit where the bracket forms,
    /// and the widest piece's transport is the move's cross. First piece
    /// draws a fair direction, the rest alternate (the sibling convention).
    /// Returns (first piece, last piece).
    pub(crate) fn split_g57(&mut self, id: u32) -> (u32, u32) {
        let g = self.arena.gate(id).clone();
        debug_assert!(g.comp, "split_g57 on a non-comp gate");
        let pieces = rules::presplit(&g, &mut self.rng);
        if self.params.local_verify {
            assert!(
                rules::verify_rewrite(std::slice::from_ref(&g), &pieces),
                "split-twist presplit verification failed: {g:?} -> {pieces:?}"
            );
        }
        let pm = self.meta_of(id);
        let ev = self.fresh_event();
        for p in &pieces {
            self.counters.width_hist[p.width().min(15)] += 1;
        }
        let d0 = self.rand_dir();
        let ids = self.splice_replace_one(id, pieces);
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
        (ids[0], *ids.last().expect("presplit emitted no pieces"))
    }

    /// The bracket draw on wire `w` (docs §2.4, v2 2026-08-05): DIRECTIONAL
    /// and length-biased, replacing the halves cascade (whose other-half
    /// preference made midpoint crossing a constant 100% — an overshoot).
    /// Candidates are the bracket-eligible gates targeting w on the picked
    /// g57's OWN side (its stored direction, the cross convention); comp and
    /// 1-control candidates compete equally. Among split_reach_k uniform
    /// samples the FARTHEST (rank distance) wins: k=1 uniform, larger k
    /// prefers longer runs. Candidates born since the last rank stamp are
    /// invisible until the next stamp (growth-triggered, so the blind window
    /// is <=25% of the circuit's life). No candidate on that side = the
    /// twist fails. Returns (id, is_comp, crossed_midpoint).
    fn pick_bracket(&mut self, w: u16, g1: u32, g_rank: u32, d: Dir) -> Option<(u32, bool, bool)> {
        if g_rank == NIL {
            return None;
        }
        self.split_candidates.clear();
        for &id in &self.wt_buckets[w as usize] {
            if id == g1 {
                continue;
            }
            let r = self.rank.get(id as usize).copied().unwrap_or(NIL);
            if r == NIL {
                continue;
            }
            let on_side = match d {
                Dir::R => r > g_rank,
                Dir::L => r < g_rank,
            };
            if on_side {
                self.split_candidates.push((id, r.abs_diff(g_rank)));
            }
        }
        if self.split_candidates.is_empty() {
            return None;
        }
        let k = self.params.split_reach_k.max(1);
        let mut best = self.split_candidates[self.rng.random_range(0..self.split_candidates.len())];
        for _ in 1..k {
            let c = self.split_candidates[self.rng.random_range(0..self.split_candidates.len())];
            if c.1 > best.1 {
                best = c;
            }
        }
        let id = best.0;
        let comp = self.arena.gate(id).comp;
        let mid = (self.rank_n / 2) as u32;
        let crossed = (g_rank < mid) != (self.rank_of(id) < mid);
        Some((id, comp, crossed))
    }

    /// The ORIGINAL v1 bracket cascade, kept as the A/B comparison arm
    /// (split_reach_k = 0): other-half g57 > other-half CNOT/NCNOT >
    /// same-half g57 > same-half CNOT/NCNOT, uniform within the first
    /// non-empty class. Its hard other-half preference makes midpoint
    /// crossing ~always true.
    fn pick_bracket_cascade(&mut self, w: u16, g1: u32, g_rank: u32) -> Option<(u32, bool, bool)> {
        let g_half = if g_rank == NIL || self.rank_n == 0 {
            None
        } else {
            Some((g_rank as usize) >= self.rank_n / 2)
        };
        let mut groups: [Vec<u32>; 4] = Default::default();
        for &id in &self.wt_buckets[w as usize] {
            if id == g1 {
                continue;
            }
            let comp = self.arena.gate(id).comp;
            let r = self.rank_of(id);
            let other = match (g_half, r) {
                (Some(gh), r) if r != NIL => gh != ((r as usize) >= self.rank_n / 2),
                _ => false,
            };
            let k = match (comp, other) {
                (true, true) => 0,
                (false, true) => 1,
                (true, false) => 2,
                (false, false) => 3,
            };
            groups[k].push(id);
        }
        for (k, grp) in groups.iter().enumerate() {
            if !grp.is_empty() {
                let id = grp[self.rng.random_range(0..grp.len())];
                return Some((id, k == 0 || k == 2, k < 2));
            }
        }
        None
    }

    /// The absorbed pure-NOT twist on wire `w` between brackets `g1` and `h1`
    /// (both 1-control gates targeting w). Locates h1 by an alternating
    /// bidirectional walk from g1 (cost <= 2x the segment the flip pass walks
    /// anyway), flips both brackets' control polarity, and conjugates the open
    /// segment: g57s reading w are force-split (5a, keeping the g57+X-series
    /// closure), every w-reading pin flips, gates targeting w are invariant.
    /// Canaries on w anchored in [left, right) count one flip. Returns the
    /// span: gates strictly between the brackets.
    fn apply_not_twist(&mut self, g1: u32, h1: u32, w: u16) -> usize {
        let (left, right) = {
            let (mut l, mut r) = (g1, g1);
            loop {
                if r != NIL {
                    r = self.arena.neighbor(r, Dir::R);
                    if r == h1 {
                        break (g1, h1);
                    }
                }
                if l != NIL {
                    l = self.arena.neighbor(l, Dir::L);
                    if l == h1 {
                        break (h1, g1);
                    }
                }
                assert!(
                    l != NIL || r != NIL,
                    "split-twist bracket not reachable from g1"
                );
            }
        };
        self.absorb_flip(g1);
        self.absorb_flip(h1);
        self.bump_taps_at(left, w);
        let mut span = 0usize;
        let mut cur = self.arena.neighbor(left, Dir::R);
        while cur != right {
            span += 1;
            debug_assert!(cur != NIL, "segment walk ran off the circuit");
            let next = self.arena.neighbor(cur, Dir::R);
            // Bump before mutating: a 5a splice evicts this node's taps to an
            // already-visited neighbor, and the count rides the tap, not the
            // anchor.
            self.bump_taps_at(cur, w);
            let g = self.arena.gate(cur);
            if g.reads(w) {
                if g.comp {
                    // 5a: force-split, then flip the pieces' w-pins. Exact:
                    // conjugation commutes with the exact presplit.
                    let gc = g.clone();
                    let mut pieces = rules::presplit(&gc, &mut self.rng);
                    for p in pieces.iter_mut() {
                        flip_wire_literal(p, w);
                    }
                    // Exhaustive verify only inside verify_rewrite's support
                    // envelope; wide gates get the identical X-conjugation and
                    // stay covered by global_check.
                    if self.params.local_verify && gc.width() < 16 {
                        let x = XGate::x_gate(w);
                        let before = vec![x.clone(), gc.clone(), x];
                        assert!(
                            rules::verify_rewrite(&before, &pieces),
                            "split-twist 5a verification failed: {gc:?} on wire {w}"
                        );
                    }
                    let pm = self.meta_of(cur);
                    let ev = self.fresh_event();
                    for p in &pieces {
                        self.counters.width_hist[p.width().min(15)] += 1;
                    }
                    let d0 = self.rand_dir();
                    let ids = self.splice_replace_one(cur, pieces);
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
                    self.counters.split_segs += 1;
                } else {
                    let mut ng = g.clone();
                    flip_wire_literal(&mut ng, w);
                    if self.params.local_verify && ng.width() < 16 {
                        let x = XGate::x_gate(w);
                        let before = vec![x.clone(), g.clone(), x];
                        assert!(
                            rules::verify_rewrite(&before, std::slice::from_ref(&ng)),
                            "split-twist pin flip verification failed on wire {w}"
                        );
                    }
                    self.index_remove(cur);
                    self.arena.replace_gate(cur, ng);
                    self.index_add(cur);
                }
            }
            cur = next;
        }
        span
    }

    /// Absorb one X(w) into a 1-control gate targeting w: flip its control's
    /// polarity (CNOT <-> NCNOT). The single-gate composition identity — the
    /// reason the twist pays zero synthetic gates.
    fn absorb_flip(&mut self, id: u32) {
        let g = self.arena.gate(id).clone();
        debug_assert!(
            !g.comp && g.ctrls.len() == 1,
            "bracket must be a 1-control conjunction"
        );
        let mut ng = g.clone();
        ng.ctrls[0].1 = !ng.ctrls[0].1;
        if self.params.local_verify {
            let before = vec![XGate::x_gate(g.target), g.clone()];
            assert!(
                rules::verify_rewrite(&before, std::slice::from_ref(&ng)),
                "split-twist absorption verification failed: {g:?}"
            );
        }
        self.index_remove(id);
        self.arena.replace_gate(id, ng);
        self.index_add(id);
    }

    // ---- split-stage instrumentation ----

    /// Restamp approximate position ranks (an O(n) walk). Heuristic
    /// consumers only; see the field comment.
    fn restamp_ranks(&mut self) {
        self.rank.clear();
        self.rank.resize(self.arena.capacity(), NIL);
        for (i, id) in self.arena.ids_in_order_iter().enumerate() {
            self.rank[id as usize] = i as u32;
        }
        self.rank_n = self.arena.len();
        self.rank_due = self.moves_done + RANK_EVERY;
    }

    /// Ordinal at the last stamp; NIL = born since it.
    fn rank_of(&self, id: u32) -> u32 {
        self.rank.get(id as usize).copied().unwrap_or(NIL)
    }

    /// Plant the wire canaries: uniform anchors, wire drawn from the anchor's
    /// touched wires. Uses metrics_rng so arming canaries never perturbs the
    /// walk trajectory of a seed.
    fn plant_taps(&mut self) {
        self.taps_planted = true;
        if self.params.split_canaries == 0 {
            return;
        }
        self.restamp_ranks();
        let ids = self.arena.ids_in_order();
        let n = ids.len();
        for _ in 0..self.params.split_canaries {
            let i = self.metrics_rng.random_range(0..n);
            let id = ids[i];
            let g = self.arena.gate(id);
            let mut wires: Vec<u16> = vec![g.target];
            wires.extend(g.ctrls.iter().map(|&(w, _)| w));
            let w = wires[self.metrics_rng.random_range(0..wires.len())];
            self.tap_at
                .entry(id)
                .or_default()
                .push(self.taps.len() as u32);
            self.taps.push(Tap {
                anchor: id,
                wire: w,
                orig_permille: ((i * 1000) / n.max(1)) as u16,
                flips: 0,
            });
        }
        println!("[fmix] split: planted {} canaries", self.taps.len());
    }

    /// Count a twist flip for every canary on `w` anchored at `id`.
    fn bump_taps_at(&mut self, id: u32, w: u16) {
        if let Some(list) = self.tap_at.get(&id) {
            for &t in list {
                let tp = &mut self.taps[t as usize];
                if tp.wire == w {
                    tp.flips += 1;
                    self.counters.tap_flips += 1;
                }
            }
        }
    }

    fn split_report_line(&mut self, span: Option<usize>) {
        // The generic mixer report is cadence-controlled; detailed split
        // telemetry must obey the same contract. Previously this printed once
        // per split move, turning a production run into millions of formatted
        // writes despite `--report-every 1000000`.
        if (self.moves_done + 1) % self.params.report_every != 0 {
            return;
        }
        let c = &self.counters;
        println!(
            "[fmix] split mv={} size={} comp={} prims={} hspl={} segs={} joins={} xmid={} fails={} streak={} tapf={} span={}",
            self.moves_done,
            self.arena.len(),
            self.comp_ids.len(),
            c.split_prims,
            c.split_hsplits,
            c.split_segs,
            c.split_joins,
            c.split_xmid,
            c.split_fails,
            self.split_fail_streak,
            c.tap_flips,
            span.map_or(-1i64, |s| s as i64),
        );
    }

    /// Span distribution at the stage boundary: mean gates between the
    /// brackets and the fraction-of-circuit histogram (5% buckets, span
    /// normalized by the circuit size AT ITS MOVE).
    fn split_span_summary(&self) {
        let c = &self.counters;
        if c.split_joins == 0 {
            return;
        }
        let cells: Vec<String> = c.split_span_hist.iter().map(|&v| v.to_string()).collect();
        println!(
            "[fmix] split spans: mean={:.0} gates over {} twists; frac-of-circuit hist (5% buckets 0-100): {}",
            c.split_span_sum as f64 / c.split_joins as f64,
            c.split_joins,
            cells.join(" ")
        );
    }

    /// Stage boundary (both exits): clear the live flag, latch split_ended for
    /// run()'s split_stop check, and print the stage summary + canary report.
    fn end_split_stage(&mut self, reason: &str) {
        self.split_on = false;
        self.split_ended = true;
        self.split_done = true;
        let c = &self.counters;
        println!(
            "[fmix] split stage ENDED at move {}: {reason} — prims={} hspl={} segs={} joins={} xmid={} fails={} size={} comp={}",
            self.moves_done,
            c.split_prims,
            c.split_hsplits,
            c.split_segs,
            c.split_joins,
            c.split_xmid,
            c.split_fails,
            self.arena.len(),
            self.comp_ids.len(),
        );
        self.split_span_summary();
        self.split_tap_summary();
    }

    /// Canary report: one line per canary plus mean flips by ORIGINAL-position
    /// decile — the spread/reach read. Idempotent (prints once); public so a
    /// run that stops before the stage boundary can still dump it.
    pub fn split_tap_summary(&mut self) {
        if self.taps.is_empty() || self.taps_reported {
            return;
        }
        self.taps_reported = true;
        self.restamp_ranks();
        let n = self.rank_n.max(1);
        let mut dec_flips = [0u64; 10];
        let mut dec_n = [0u64; 10];
        for t in &self.taps {
            let now = self
                .rank
                .get(t.anchor as usize)
                .copied()
                .filter(|&r| r != NIL)
                .map(|r| (r as usize * 1000) / n);
            println!(
                "[fmix] canary wire={} orig={} now={} flips={}",
                t.wire,
                t.orig_permille,
                now.map_or(-1i64, |x| x as i64),
                t.flips
            );
            let d = (t.orig_permille as usize / 100).min(9);
            dec_flips[d] += t.flips;
            dec_n[d] += 1;
        }
        let cells: Vec<String> = (0..10)
            .map(|d| {
                if dec_n[d] == 0 {
                    "-".to_string()
                } else {
                    format!("{:.1}", dec_flips[d] as f64 / dec_n[d] as f64)
                }
            })
            .collect();
        println!(
            "[fmix] canary deciles (mean flips by ORIGINAL position, left to right): {}",
            cells.join(" ")
        );
    }
}
