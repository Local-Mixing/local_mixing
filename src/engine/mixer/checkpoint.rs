//! Versioned .state serialization; field order and v1/early-v2 readers are preserved.
use super::*;

impl MixCounters {
    /// Whitespace-separated dump of every counter, in declaration order. These
    /// are trajectory statistics rather than chain state, but a resumed run
    /// whose report restarted from zero would make its own history unreadable,
    /// and two of them -- twist_span and canary_fallthrough -- feed conditions.
    pub fn to_line(&self) -> String {
        let vals: Vec<u64> = vec![
            self.moves,
            self.db_ing_hits,
            self.db_ing_rounds,
            self.db_hard_hits,
            self.db_hard_rounds,
            self.db_hard_added,
            self.db_identity_skips,
            self.db_curated_hits,
            self.merges_absorb,
            self.db_g57_rounds,
            self.db_g57_hits,
            self.db_slot2_rounds,
            self.db_slot2_hits,
            self.db_slot2_added,
            self.brake_engagements,
            self.brake_rounds,
            self.canary_fallthrough,
            self.litter_banned,
            self.twist_placed,
            self.db_curated_rejected,
            self.twist_place_fallback,
            self.dmin_windows,
            self.dmin_shorter,
            self.litter_windows,
            self.litter_distinct_sum,
            self.litter_full_spliced,
            self.gen_misses,
            self.merges_cancel,
            self.merges_xfuse,
            self.merges_drop,
            self.merges_subsume,
            self.merges_sibling,
            self.merges_cross_origin,
            self.tabu_blocked,
            self.merge_no_partner,
            self.merge_wall_blocked,
            self.merge_too_far,
            self.merge_not_adjacent,
            self.undos,
            self.undo_dead,
            self.undo_tabu,
            self.undo_gather_miss,
            self.db_comp_hits,
            self.db_comp_misses,
            self.db_agn_hits,
            self.db_agn_misses,
            self.db_gates_removed,
            self.db_gates_added,
            self.db_wide_skip,
            self.db_attempts,
            self.db_degree_skips,
            self.db_span_skips,
            self.db_build_aborts,
            self.cross_r1,
            self.cross_r2,
            self.cross_r3,
            self.presplits,
            self.fresh_splits,
            self.unsubs,
            self.inserts,
            self.twist_negs,
            self.twist_swaps,
            self.twist_cnots,
            self.twist_relabels,
            self.twist_case_splits,
            self.twist_span,
            self.twist_skips,
            self.blocked_width,
            self.blocked_deadlock,
            self.declined,
            self.boundary,
            self.floats,
            self.float_steps,
            self.scatters,
            self.scatter_steps,
            self.dropped_neverfire,
            // Trailing fields are parsed with a zero default so pre-existing
            // .state files stay loadable; append here, never insert.
            self.tg_consumed,
            self.tg_emitted,
            // Split stage (2026-08-05).
            self.split_prims,
            self.split_hsplits,
            self.split_segs,
            self.split_joins,
            self.split_fails,
            self.split_xmid,
            self.tap_flips,
            self.split_span_sum,
            self.cross_pool_shots,
        ];
        vals.iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Inverse of `to_line`. The two histograms (`splice_sizes`, `width_hist`)
    /// are deliberately not carried: they describe SHAPES rather than totals,
    /// and a resumed run should report the shapes it produced itself.
    pub fn from_line(s: &str) -> Option<MixCounters> {
        let mut it = s.split_whitespace();
        fn next_u64<'a>(it: &mut impl Iterator<Item = &'a str>) -> Option<u64> {
            it.next()?.parse().ok()
        }
        Some(MixCounters {
            moves: next_u64(&mut it)?,
            db_ing_hits: next_u64(&mut it)?,
            db_ing_rounds: next_u64(&mut it)?,
            db_hard_hits: next_u64(&mut it)?,
            db_hard_rounds: next_u64(&mut it)?,
            db_hard_added: next_u64(&mut it)?,
            db_identity_skips: next_u64(&mut it)?,
            db_curated_hits: next_u64(&mut it)?,
            merges_absorb: next_u64(&mut it)?,
            db_g57_rounds: next_u64(&mut it)?,
            db_g57_hits: next_u64(&mut it)?,
            db_slot2_rounds: next_u64(&mut it)?,
            db_slot2_hits: next_u64(&mut it)?,
            db_slot2_added: next_u64(&mut it)?,
            brake_engagements: next_u64(&mut it)?,
            brake_rounds: next_u64(&mut it)?,
            canary_fallthrough: next_u64(&mut it)?,
            litter_banned: next_u64(&mut it)?,
            twist_placed: next_u64(&mut it)?,
            db_curated_rejected: next_u64(&mut it)?,
            twist_place_fallback: next_u64(&mut it)?,
            dmin_windows: next_u64(&mut it)?,
            dmin_shorter: next_u64(&mut it)?,
            litter_windows: next_u64(&mut it)?,
            litter_distinct_sum: next_u64(&mut it)?,
            litter_full_spliced: next_u64(&mut it)?,
            gen_misses: next_u64(&mut it)?,
            merges_cancel: next_u64(&mut it)?,
            merges_xfuse: next_u64(&mut it)?,
            merges_drop: next_u64(&mut it)?,
            merges_subsume: next_u64(&mut it)?,
            merges_sibling: next_u64(&mut it)?,
            merges_cross_origin: next_u64(&mut it)?,
            tabu_blocked: next_u64(&mut it)?,
            merge_no_partner: next_u64(&mut it)?,
            merge_wall_blocked: next_u64(&mut it)?,
            merge_too_far: next_u64(&mut it)?,
            merge_not_adjacent: next_u64(&mut it)?,
            undos: next_u64(&mut it)?,
            undo_dead: next_u64(&mut it)?,
            undo_tabu: next_u64(&mut it)?,
            undo_gather_miss: next_u64(&mut it)?,
            db_comp_hits: next_u64(&mut it)?,
            db_comp_misses: next_u64(&mut it)?,
            db_agn_hits: next_u64(&mut it)?,
            db_agn_misses: next_u64(&mut it)?,
            db_gates_removed: next_u64(&mut it)?,
            db_gates_added: next_u64(&mut it)?,
            db_wide_skip: next_u64(&mut it)?,
            db_attempts: next_u64(&mut it)?,
            db_degree_skips: next_u64(&mut it)?,
            db_span_skips: next_u64(&mut it)?,
            db_build_aborts: next_u64(&mut it)?,
            cross_r1: next_u64(&mut it)?,
            cross_r2: next_u64(&mut it)?,
            cross_r3: next_u64(&mut it)?,
            presplits: next_u64(&mut it)?,
            fresh_splits: next_u64(&mut it)?,
            unsubs: next_u64(&mut it)?,
            inserts: next_u64(&mut it)?,
            twist_negs: next_u64(&mut it)?,
            twist_swaps: next_u64(&mut it)?,
            twist_cnots: next_u64(&mut it)?,
            twist_relabels: next_u64(&mut it)?,
            twist_case_splits: next_u64(&mut it)?,
            twist_span: next_u64(&mut it)?,
            twist_skips: next_u64(&mut it)?,
            blocked_width: next_u64(&mut it)?,
            blocked_deadlock: next_u64(&mut it)?,
            declined: next_u64(&mut it)?,
            boundary: next_u64(&mut it)?,
            floats: next_u64(&mut it)?,
            float_steps: next_u64(&mut it)?,
            scatters: next_u64(&mut it)?,
            scatter_steps: next_u64(&mut it)?,
            dropped_neverfire: next_u64(&mut it)?,
            // Appended after the twist-g57 work landed: absent in older state
            // files, so they default to zero rather than failing the resume.
            tg_consumed: next_u64(&mut it).unwrap_or(0),
            tg_emitted: next_u64(&mut it).unwrap_or(0),
            split_prims: next_u64(&mut it).unwrap_or(0),
            split_hsplits: next_u64(&mut it).unwrap_or(0),
            split_segs: next_u64(&mut it).unwrap_or(0),
            split_joins: next_u64(&mut it).unwrap_or(0),
            split_fails: next_u64(&mut it).unwrap_or(0),
            split_xmid: next_u64(&mut it).unwrap_or(0),
            tap_flips: next_u64(&mut it).unwrap_or(0),
            split_span_sum: next_u64(&mut it).unwrap_or(0),
            cross_pool_shots: next_u64(&mut it).unwrap_or(0),
            split_span_hist: [0u64; 20],
            tg_solves: 0,
            tg_solve_ns: 0,
            tg_slides: 0,
            tg_retries: 0,
            shuffles: 0,
            shuffle_moved: 0,
            shuffle_steps: 0,
            shuffle_ns: 0,
            choice_splices: 0,
            choice_multi: 0,
            choice_sum: 0,
            choice_bits_milli: 0,
            db_wide_poly: 0,
            len_attempts: Vec::new(),
            len_hits: Vec::new(),
            len_removed: Vec::new(),
            len_added: Vec::new(),
            len_span_skip: Vec::new(),
            len_deg_skip: Vec::new(),
            db_mix_added: 0,
            db_mix_removed: 0,
            db_cmp_added: 0,
            db_cmp_removed: 0,
            splice_sizes: Vec::new(),
            splice_sizes_curated: Vec::new(),
            pair_rounds: 0,
            pair_boxes_empty: 0,
            pair_scan_truncs: 0,
            pair_fused: 0,
            pair_splices: 0,
            pair_box_sum: 0,
            pair_box_max: 0,
            pair_dist_sum: 0,
            pair_dist_max: 0,
            pair_perm_skips: 0,
            bridge_rounds: 0,
            bridge_short: 0,
            bridge_refused: 0,
            bridge_probe_miss: 0,
            bridge_rollbacks: 0,
            bridge_half: 0,
            bridge_committed: 0,
            bridge_span_sum: 0,
            bridge_span_max: 0,
            bridge_colliders_sum: 0,
            bridge_wake_sum: 0,
            width_hist: [0u64; 16],
            tg_net_hist: [0u64; 8],
        })
    }
}
// v2 (2026-08-05): the split-stage scalar line and the staps (canary) section
// (docs/FMIX_SPLIT_TWIST.md §7). The reader still accepts v1, defaulting both.
pub const STATE_VERSION: u32 = 2;

impl Mixer {
    /// Write everything a resumed run needs and the circuit file does not
    /// carry. A run is hours long and every measurement depends on it, so a
    /// stop -- flag, canary, dose or budget -- should be a PAUSE, not a loss.
    ///
    /// Three things make this more than a gate dump:
    ///
    /// - Per-gate `dir` is load-bearing and has no sidecar. Directions are
    ///   drawn at load and the whole directional walk rides on them, so a
    ///   resume that redrew them would restart transport rather than continue
    ///   it. Same for `dgen`, `litter` and `event`.
    /// - The undo journal references arena IDs, so the checkpoint RENUMBERS to
    ///   0..n-1 in arena order -- exactly what `Arena::from_gates` reproduces --
    ///   and remaps the entries. Entries with any dead piece are dropped; they
    ///   were already unusable.
    /// - The ORIGINAL circuit is what `global_check` compares against. A
    ///   resumed run verifying against its own resume point would verify
    ///   nothing about fidelity to the true input.
    ///
    /// `StdRng` is not serialisable, so a fresh `u64` is drawn from each
    /// generator and stored: a clean continuation, not a bit-identical replay.
    pub fn save_state(&mut self, path: &str) -> std::io::Result<()> {
        use std::fmt::Write as _;
        let ids = self.arena.ids_in_order();
        let mut newid: FxHashMap<u32, u32> =
            FxHashMap::with_capacity_and_hasher(ids.len(), Default::default());
        for (i, &id) in ids.iter().enumerate() {
            newid.insert(id, i as u32);
        }
        let gate_line = |o: &mut String, g: &XGate| {
            let _ = write!(o, "{} {} {}", g.target, g.comp as u8, g.ctrls.len());
            for &(w, p) in &g.ctrls {
                let _ = write!(o, " {} {}", w, p as u8);
            }
        };
        let mut o = String::with_capacity(ids.len() * 48);
        let _ = writeln!(o, "fmix-state {STATE_VERSION}");
        let _ = writeln!(o, "wires {}", self.num_wires);
        let _ = writeln!(o, "moves {}", self.moves_done);
        let _ = writeln!(o, "next_event {}", self.next_event);
        let _ = writeln!(o, "next_litter {}", self.next_litter);
        let _ = writeln!(o, "rng {}", self.rng.random::<u64>());
        let _ = writeln!(o, "metrics_rng {}", self.metrics_rng.random::<u64>());
        let _ = writeln!(
            o,
            "db_mode {}",
            match self.db_mode_cur {
                DbMode::Mix => "mix",
                DbMode::Compressing => "comp",
                DbMode::SizeAgnostic => "any",
                DbMode::MinGrow => "mingrow",
                DbMode::Stable => "stable",
                DbMode::StableGrow => "stable-grow",
                DbMode::StableLedger => "stable-ledger",
                DbMode::Same => "same",
                DbMode::BandLedger => "band-ledger",
                DbMode::BandShrink => "band-shrink",
                DbMode::BandGrow => "band-grow",
            }
        );
        let _ = writeln!(
            o,
            "brake {} {} {}",
            self.brake_on as u8, self.brake_mark_move, self.brake_mark_size
        );
        let _ = writeln!(o, "pool_scan_due {}", self.pool_scan_due);
        let _ = writeln!(o, "canary_failures {}", self.canary_failures);
        // Tri-state phase (0 = never armed, 1 = live, 2 = ended) so a resume
        // can tell "stage already ran" from "stage never requested", plus the
        // canary-report latch.
        let split_phase: u8 = if self.split_on {
            1
        } else if self.split_done {
            2
        } else {
            0
        };
        let _ = writeln!(
            o,
            "split {} {} {}",
            split_phase, self.split_fail_streak, self.taps_reported as u8
        );
        let _ = writeln!(o, "anc {} {}", self.anc_words, self.anc_m);
        let _ = writeln!(o, "counters {}", self.counters.to_line());

        let _ = writeln!(o, "gates {}", ids.len());
        for &id in &ids {
            let m = self.meta_of(id);
            gate_line(&mut o, self.arena.gate(id));
            let _ = writeln!(
                o,
                " | {} {} {} {} {} {}",
                m.origin,
                m.event,
                (m.dir == Dir::R) as u8,
                m.dgen,
                m.litter,
                m.litter_size
            );
        }
        let _ = writeln!(o, "original {}", self.original.len());
        for g in &self.original {
            gate_line(&mut o, g);
            o.push('\n');
        }
        let _ = writeln!(o, "tabu {}", self.tabu.len());
        for &(e, mv) in self.tabu.iter() {
            let _ = writeln!(o, "{e} {mv}");
        }
        let pool: Vec<u32> = self
            .pool
            .iter()
            .filter_map(|id| newid.get(id).copied())
            .collect();
        let _ = writeln!(o, "pool {}", pool.len());
        for id in &pool {
            let _ = writeln!(o, "{id}");
        }
        let _ = writeln!(o, "canary {}", self.canary.len());
        for b in self.canary.iter() {
            let _ = writeln!(o, "{}", *b as u8);
        }
        // Journal: keep only entries every piece of which is still live, then
        // remap. Stamps are NOT stored -- the rebuilt arena assigns its own, and
        // resume reads them back from it.
        let keep: Vec<&UndoEntry> = self
            .journal
            .iter()
            .filter(|e| {
                e.after.iter().all(|&(id, st)| {
                    self.arena.is_linked(id)
                        && self.arena.stamp(id) == st
                        && newid.contains_key(&id)
                })
            })
            .collect();
        let _ = writeln!(o, "journal {}", keep.len());
        for e in keep {
            gate_line(&mut o, &e.before[0]);
            o.push(' ');
            gate_line(&mut o, &e.before[1]);
            let _ = write!(
                o,
                " | {} {} {} {} {} {} {} {} {} {} {}",
                (e.dir == Dir::R) as u8,
                newid[&e.pivot],
                e.event,
                e.origins[0],
                e.origins[1],
                e.gens[0],
                e.gens[1],
                e.litters[0],
                e.litters[1],
                e.litter_sizes[0],
                e.litter_sizes[1]
            );
            let _ = write!(o, " | {}", e.after.len());
            for &(id, _) in &e.after {
                let _ = write!(o, " {}", newid[&id]);
            }
            let _ = writeln!(o, " {}", e.misses);
        }
        let _ = writeln!(o, "ancsets {}", self.anc.len());
        for (l, bits) in self.anc.iter() {
            let _ = write!(o, "{l}");
            for w in bits {
                let _ = write!(o, " {w}");
            }
            o.push('\n');
        }
        // Optional trailing section (2026-08-03): the sampled tracer list,
        // serialized explicitly. An IMPORTED tracer set (--anc-in) is not a
        // function of (anc_m, K, anc_sample_seed), so the regeneration the
        // resume path used to rely on would silently remap the mask bits.
        // Old states lack the section (the reader falls back to
        // regeneration) and old binaries never read this far, so the version
        // deliberately stays 1.
        if self.anc_sampled {
            let _ = write!(o, "anctracers {}", self.anc_tracers.len());
            for t in &self.anc_tracers {
                let _ = write!(o, " {t}");
            }
            o.push('\n');
        }
        // Wire canaries: anchors are serialized as POSITIONS (the checkpoint
        // renumbers ids), re-anchored by ordinal at load.
        let _ = writeln!(o, "staps {}", self.taps.len());
        if !self.taps.is_empty() {
            let pos: HashMap<u32, usize> = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
            for t in &self.taps {
                let p = pos.get(&t.anchor).copied().unwrap_or(0);
                let _ = writeln!(o, "{} {} {} {}", t.wire, t.orig_permille, t.flips, p);
            }
        }
        std::fs::write(path, o)
    }

    /// Rebuild a mixer from a state file. `params` come from the CLI as usual:
    /// a resume is free to change rates, targets, the brake or the mode, which
    /// is the point -- a paused run should be steerable. What it must NOT
    /// change is the version, since the field meanings would drift silently.
    pub fn resume_state(path: &str, params: MixParams, db: FrozenDb) -> std::io::Result<Mixer> {
        let text = std::fs::read_to_string(path)?;
        let mut lines = text.lines();
        let bad = |m: &str| std::io::Error::other(m.to_string());
        let mut hdr = lines
            .next()
            .ok_or_else(|| bad("empty state file"))?
            .split_whitespace();
        if hdr.next() != Some("fmix-state") {
            return Err(bad("missing fmix-state header"));
        }
        let v: u32 = hdr
            .next()
            .and_then(|x| x.parse().ok())
            .ok_or_else(|| bad("bad version"))?;
        // v1 is a strict prefix of v2 (split line and staps section absent),
        // so old states load with the split stage off and no canaries.
        if v != 1 && v != STATE_VERSION {
            return Err(bad(&format!(
                "state file version {v} != {STATE_VERSION}; refusing to reinterpret its fields"
            )));
        }
        // Scalars, in the order save_state writes them.
        fn scalar(lines: &mut std::str::Lines<'_>, want: &str) -> std::io::Result<Vec<String>> {
            let l = lines
                .next()
                .ok_or_else(|| std::io::Error::other(format!("missing {want}")))?;
            let mut it = l.split_whitespace();
            if it.next() != Some(want) {
                return Err(std::io::Error::other(format!(
                    "expected {want} in state file"
                )));
            }
            Ok(it.map(|x| x.to_string()).collect())
        }
        let num_wires: usize = scalar(&mut lines, "wires")?[0]
            .parse()
            .map_err(|_| bad("wires"))?;
        let moves_done: u64 = scalar(&mut lines, "moves")?[0]
            .parse()
            .map_err(|_| bad("moves"))?;
        let next_event: u64 = scalar(&mut lines, "next_event")?[0]
            .parse()
            .map_err(|_| bad("next_event"))?;
        let next_litter: u64 = scalar(&mut lines, "next_litter")?[0]
            .parse()
            .map_err(|_| bad("next_litter"))?;
        let rng_seed: u64 = scalar(&mut lines, "rng")?[0]
            .parse()
            .map_err(|_| bad("rng"))?;
        let mrng_seed: u64 = scalar(&mut lines, "metrics_rng")?[0]
            .parse()
            .map_err(|_| bad("metrics_rng"))?;
        let mode_s = scalar(&mut lines, "db_mode")?[0].clone();
        let brake = scalar(&mut lines, "brake")?;
        let pool_scan_due: u64 = scalar(&mut lines, "pool_scan_due")?[0]
            .parse()
            .map_err(|_| bad("scan"))?;
        let canary_failures: usize = scalar(&mut lines, "canary_failures")?[0]
            .parse()
            .map_err(|_| bad("cf"))?;
        let split_state: Option<(u8, u32, bool)> = if v >= 2 {
            let s = scalar(&mut lines, "split")?;
            let phase: u8 = s
                .first()
                .and_then(|x| x.parse().ok())
                .ok_or_else(|| bad("split"))?;
            let streak = s
                .get(1)
                .and_then(|x| x.parse().ok())
                .ok_or_else(|| bad("split streak"))?;
            // Third field absent in the earliest v2 files: default unreported.
            let reported = s.get(2).and_then(|x| x.parse::<u8>().ok()).unwrap_or(0) != 0;
            Some((phase, streak, reported))
        } else {
            None
        };
        let anc_hdr = scalar(&mut lines, "anc")?;
        let counters_line = scalar(&mut lines, "counters")?.join(" ");

        // Sections. Gates carry their meta on the same line after a `|`.
        fn section(lines: &mut std::str::Lines<'_>, want: &str) -> std::io::Result<usize> {
            let l = lines
                .next()
                .ok_or_else(|| std::io::Error::other(format!("missing {want}")))?;
            let mut it = l.split_whitespace();
            if it.next() != Some(want) {
                return Err(std::io::Error::other(format!("expected section {want}")));
            }
            it.next()
                .and_then(|x| x.parse().ok())
                .ok_or_else(|| std::io::Error::other(format!("{want} count")))
        }
        let parse_gate = |it: &mut std::str::SplitWhitespace| -> Option<XGate> {
            let target: u16 = it.next()?.parse().ok()?;
            let comp: u8 = it.next()?.parse().ok()?;
            let k: usize = it.next()?.parse().ok()?;
            let mut ctrls: Lits = Lits::new();
            for _ in 0..k {
                let w: u16 = it.next()?.parse().ok()?;
                let p: u8 = it.next()?.parse().ok()?;
                ctrls.push((w, p != 0));
            }
            ctrls.sort_unstable();
            Some(XGate {
                target,
                comp: comp != 0,
                ctrls,
            })
        };

        let ng = section(&mut lines, "gates")?;
        let mut gates = Vec::with_capacity(ng);
        let mut metas = Vec::with_capacity(ng);
        for _ in 0..ng {
            let l = lines.next().ok_or_else(|| bad("short gates section"))?;
            let (gp, mp) = l
                .split_once(" | ")
                .ok_or_else(|| bad("gate line missing meta"))?;
            let g = parse_gate(&mut gp.split_whitespace()).ok_or_else(|| bad("bad gate"))?;
            let m: Vec<&str> = mp.split_whitespace().collect();
            if m.len() != 6 {
                return Err(bad("gate meta must have 6 fields"));
            }
            let p = |i: usize| -> std::io::Result<u64> {
                m[i].parse().map_err(|_| bad("bad meta field"))
            };
            metas.push(Meta {
                origin: p(0)? as u32,
                event: p(1)?,
                dir: if p(2)? != 0 { Dir::R } else { Dir::L },
                dgen: p(3)? as u32,
                litter: p(4)?,
                litter_size: p(5)? as u16,
            });
            gates.push(g);
        }
        let no = section(&mut lines, "original")?;
        let mut original = Vec::with_capacity(no);
        for _ in 0..no {
            let l = lines.next().ok_or_else(|| bad("short original section"))?;
            original.push(parse_gate(&mut l.split_whitespace()).ok_or_else(|| bad("bad orig"))?);
        }
        let nt = section(&mut lines, "tabu")?;
        let mut tabu: VecDeque<(u64, u64)> = VecDeque::with_capacity(nt);
        for _ in 0..nt {
            let l = lines.next().ok_or_else(|| bad("short tabu"))?;
            let mut it = l.split_whitespace();
            let e: u64 = it
                .next()
                .and_then(|x| x.parse().ok())
                .ok_or_else(|| bad("tabu"))?;
            let mv: u64 = it
                .next()
                .and_then(|x| x.parse().ok())
                .ok_or_else(|| bad("tabu"))?;
            tabu.push_back((e, mv));
        }
        let np = section(&mut lines, "pool")?;
        let mut pool = Vec::with_capacity(np);
        for _ in 0..np {
            let l = lines.next().ok_or_else(|| bad("short pool"))?;
            pool.push(l.trim().parse::<u32>().map_err(|_| bad("pool id"))?);
        }
        let nc = section(&mut lines, "canary")?;
        let mut canary: VecDeque<bool> = VecDeque::with_capacity(nc);
        for _ in 0..nc {
            let l = lines.next().ok_or_else(|| bad("short canary"))?;
            canary.push_back(l.trim() != "0");
        }
        let nj = section(&mut lines, "journal")?;
        let mut journal_raw = Vec::with_capacity(nj);
        for _ in 0..nj {
            let l = lines.next().ok_or_else(|| bad("short journal"))?;
            let parts: Vec<&str> = l.split(" | ").collect();
            if parts.len() != 3 {
                return Err(bad("journal line shape"));
            }
            let mut gi = parts[0].split_whitespace();
            let b0 = parse_gate(&mut gi).ok_or_else(|| bad("journal before0"))?;
            let b1 = parse_gate(&mut gi).ok_or_else(|| bad("journal before1"))?;
            let f: Vec<u64> = parts[1]
                .split_whitespace()
                .map(|x| x.parse().unwrap_or(0))
                .collect();
            if f.len() != 11 {
                return Err(bad("journal meta shape"));
            }
            let mut ai = parts[2].split_whitespace();
            let n: usize = ai
                .next()
                .and_then(|x| x.parse().ok())
                .ok_or_else(|| bad("after n"))?;
            let after: Vec<u32> = (0..n)
                .filter_map(|_| ai.next().and_then(|x| x.parse().ok()))
                .collect();
            let misses: u8 = ai.next().and_then(|x| x.parse().ok()).unwrap_or(0);
            journal_raw.push((b0, b1, f, after, misses));
        }
        let na = section(&mut lines, "ancsets")?;
        let mut anc: HashMap<u64, Vec<u64>> = HashMap::with_capacity(na);
        for _ in 0..na {
            let l = lines.next().ok_or_else(|| bad("short ancsets"))?;
            let mut it = l.split_whitespace();
            let key: u64 = it
                .next()
                .and_then(|x| x.parse().ok())
                .ok_or_else(|| bad("anc key"))?;
            anc.insert(key, it.filter_map(|x| x.parse().ok()).collect());
        }
        // Optional trailing section: explicitly serialized tracers (states
        // written since 2026-08-03). Absent in older states, which regenerate
        // below instead. Read with one line of lookahead, since the v2 staps
        // section may follow (or replace) it.
        let mut pending = lines.next();
        let stored_tracers: Option<Vec<u32>> = match pending {
            Some(l) if l.starts_with("anctracers ") => {
                let mut it = l["anctracers ".len()..].split_whitespace();
                let k: usize = it
                    .next()
                    .and_then(|x| x.parse().ok())
                    .ok_or_else(|| bad("anctracers count"))?;
                let tr: Vec<u32> = it.filter_map(|x| x.parse().ok()).collect();
                if tr.len() != k {
                    return Err(bad("anctracers list length mismatch"));
                }
                pending = lines.next();
                Some(tr)
            }
            _ => None,
        };
        // v2 canaries: (wire, orig_permille, flips, position at save).
        let mut staps_raw: Vec<(u16, u16, u64, usize)> = Vec::new();
        if let Some(l) = pending {
            if let Some(rest) = l.strip_prefix("staps ") {
                let k: usize = rest.trim().parse().map_err(|_| bad("staps count"))?;
                for _ in 0..k {
                    let tl = lines.next().ok_or_else(|| bad("short staps section"))?;
                    let f: Vec<&str> = tl.split_whitespace().collect();
                    if f.len() != 4 {
                        return Err(bad("staps line shape"));
                    }
                    staps_raw.push((
                        f[0].parse().map_err(|_| bad("stap wire"))?,
                        f[1].parse().map_err(|_| bad("stap orig"))?,
                        f[2].parse().map_err(|_| bad("stap flips"))?,
                        f[3].parse().map_err(|_| bad("stap pos"))?,
                    ));
                }
            }
        }

        // Build the mixer on the SAME id assignment the checkpoint renumbered
        // to: Arena::from_gates hands out 0..n-1 in order, which is what
        // save_state mapped the journal and pool onto.
        // Construct with ancestors OFF: the sizing guard in new_with_db reads
        // the CURRENT gate count, but ancestor sets are indexed by ORIGINAL
        // input gates and their width comes from the state file. A resumed 1.4M
        // gate circuit whose input was 20k would trip a guard meant for the
        // input. Restore the real setting and the recorded widths below.
        // Same reasoning applies to --anc-samples: tracers index the ORIGINAL
        // input gates, but new_with_db would draw them against the RESUMED gate
        // count. Construct with sampling off and regenerate below from the
        // restored anc_m.
        let ancestors_wanted = params.ancestors;
        let anc_samples_wanted = params.anc_samples;
        let anc_sample_seed = params.anc_sample_seed;
        let mut mx = Mixer::new_with_db(
            gates,
            num_wires,
            MixParams {
                ancestors: false,
                anc_samples: 0,
                ..params
            },
            db,
        );
        mx.params.ancestors = ancestors_wanted;
        mx.params.anc_samples = anc_samples_wanted;
        mx.original = original;
        mx.moves_done = moves_done;
        mx.counters = MixCounters::from_line(&counters_line).ok_or_else(|| bad("counters"))?;
        mx.counters.moves = moves_done;
        mx.next_event = next_event;
        mx.next_litter = next_litter;
        mx.rng = StdRng::seed_from_u64(rng_seed);
        mx.metrics_rng = StdRng::seed_from_u64(mrng_seed);
        // The LIVE mode comes from the command line, not the file. A resume is
        // meant to be re-steerable -- changing --db-mode is the whole point of a
        // manual breathing cycle -- and letting the saved value win silently
        // ignores the flag: a resume asked for COMP would keep growing in MIX
        // and look like COMP was broken. The saved mode is kept for diagnostics
        // only. If the brake was engaged it re-engages on the next round
        // anyway, since apply_size_brake runs before slot 1.
        let _saved_mode = DbMode::parse(&mode_s);
        mx.db_mode_cur = mx.params.db_mode;
        mx.brake_on = brake[0] != "0";
        mx.brake_mark_move = brake[1].parse().unwrap_or(0);
        mx.brake_mark_size = brake[2].parse().unwrap_or(0);
        mx.pool_scan_due = pool_scan_due;
        mx.canary = canary;
        mx.canary_failures = canary_failures;
        mx.pool = pool;
        // Split stage, by recorded phase. 1 (live) continues the stage —
        // which still needs --split on the resume line per the repeat-your-
        // flags rule (warn loudly if it is missing, that is almost always a
        // mistake). 2 (ended) never re-arms: --split on the resume just means
        // "this is a split pipeline", part 2 continues. 0 (never armed) lets
        // an explicit --split start the stage fresh on the resumed circuit.
        // v1 states have no phase and take the constructor's arming.
        if let Some((phase, streak, reported)) = split_state {
            match phase {
                1 => {
                    mx.split_on = mx.params.split;
                    if !mx.params.split {
                        eprintln!(
                            "[fmix] WARNING: state file has a LIVE split stage but --split was not \
                             given — the stage stays OFF and part-2 moves run on unsplit material"
                        );
                    }
                }
                2 => {
                    mx.split_on = false;
                    mx.split_done = true;
                }
                _ => mx.split_on = mx.params.split,
            }
            mx.split_fail_streak = streak;
            mx.taps_reported = reported;
        }
        if !staps_raw.is_empty() {
            let ids = mx.arena.ids_in_order();
            for (wire, orig, flips, pos) in staps_raw {
                let anchor = ids[pos.min(ids.len() - 1)];
                mx.tap_at
                    .entry(anchor)
                    .or_default()
                    .push(mx.taps.len() as u32);
                mx.taps.push(Tap {
                    anchor,
                    wire,
                    orig_permille: orig,
                    flips,
                });
            }
            mx.taps_planted = true;
        }
        mx.anc = anc;
        mx.anc_words = anc_hdr[0].parse().unwrap_or(0);
        mx.anc_m = anc_hdr[1].parse().unwrap_or(0);
        // Tracer restoration. States since 2026-08-03 carry the list
        // explicitly (required for --anc-in imports, whose tracers are not a
        // function of anything this run knows); older states regenerate it
        // from (anc_m, K, anc_sample_seed) exactly as before. The assertions
        // catch a resume that changed K (or the seed, when that changes K's
        // rounding): the stored masks index the ORIGINAL tracer set and
        // cannot be reinterpreted.
        if let Some(tr) = stored_tracers {
            assert_eq!(
                mx.anc_words,
                tr.len().div_ceil(64),
                "state file anctracers/ancsets width mismatch ({} tracers, {} words)",
                tr.len(),
                mx.anc_words
            );
            if anc_samples_wanted > 0 {
                assert_eq!(
                    anc_samples_wanted.min(mx.anc_m),
                    tr.len(),
                    "resume changed --anc-samples ({} wanted, {} stored): the recorded masks \
                     index the ORIGINAL tracer set and cannot be reinterpreted",
                    anc_samples_wanted.min(mx.anc_m),
                    tr.len()
                );
            }
            mx.params.anc_samples = tr.len();
            mx.anc_sampled = true;
            mx.anc_tracers = tr;
        } else if anc_samples_wanted > 0 && mx.anc_m > 0 {
            let k = anc_samples_wanted.min(mx.anc_m);
            assert_eq!(
                mx.anc_words,
                k.div_ceil(64),
                "resume changed --anc-samples ({k} tracers now, {} words stored): the recorded \
                 masks index the ORIGINAL tracer set and cannot be reinterpreted",
                mx.anc_words
            );
            mx.anc_sampled = true;
            mx.anc_tracers = Self::pick_tracers(mx.anc_m, k, anc_sample_seed);
        }
        for (i, m) in metas.into_iter().enumerate() {
            mx.set_meta(i as u32, m);
        }
        // Stamps come from the freshly built arena; the checkpoint kept only
        // entries whose pieces were all live, so every id here is linked.
        mx.journal = journal_raw
            .into_iter()
            .map(|(b0, b1, f, after, misses)| UndoEntry {
                before: [b0, b1],
                dir: if f[0] != 0 { Dir::R } else { Dir::L },
                pivot: f[1] as u32,
                after: after.iter().map(|&id| (id, mx.arena.stamp(id))).collect(),
                event: f[2],
                origins: [f[3] as u32, f[4] as u32],
                gens: [f[5] as u32, f[6] as u32],
                litters: [f[7], f[8]],
                litter_sizes: [f[9] as u16, f[10] as u16],
                misses,
            })
            .collect();
        mx.tabu = tabu;
        Ok(mx)
    }
}
