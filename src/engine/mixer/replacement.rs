//! Database replacement attempts, acceptance accounting and recorded probes.
use super::*;

impl Mixer {
    // ---- DB replacement moves ----
    //
    // Sample a window of [db_min_window, db_max_window] gates (contiguous or
    // convex, per db_sample; wide gates evaded per db_ctrl_cap), look it up in
    // the frozen store by its exact function polynomial (which handles the
    // conjunction-control "toffoli" gates the walk produces, not just g57s), and
    // splice in an equivalent circuit chosen per `mode`:
    //   Compressing  -> a non-growing equivalent, uniform among the shortest;
    //   SizeAgnostic -> any equivalent, uniform over all (may grow the circuit).
    // The sampled window is a contiguous run (convex sampling floats it together
    // first), so replacing it by an equal-function block preserves the circuit.
    // When `db_verify` is on the splice is checked exhaustively first (support
    // <= 24 wires); with it off the splice rests on the key/decode invariants and
    // the periodic global_check. Every attempt is recorded if --db-record is set.
    // Returns true iff a replacement was spliced in.
    // Miss-accounting wrapper: when the attempt was seeded on a laggard
    // (last_seed recorded by draw_laggard) and that exact gate survived the
    // round unconsumed (same id, same arena stamp — a reused id fails the
    // stamp check), the seed missed: bump its counter so the tier machinery
    // can graduate it to the paid channel or retire it as unreachable.
    /// A COMP-DB attempt that may be drawn as g57-only (see `p_comp_g57`).
    pub(super) fn db_attempt_comp(&mut self) -> bool {
        let g57 = self.params.p_comp_g57 > 0.0
            && self.rng.random_bool(self.params.p_comp_g57.clamp(0.0, 1.0));
        self.db_g57_only = g57;
        if g57 {
            self.counters.db_g57_rounds += 1;
        }
        let hit = self.db_attempt(DbMode::Compressing);
        if g57 && hit {
            self.counters.db_g57_hits += 1;
        }
        self.db_g57_only = false;
        hit
    }

    pub(super) fn db_attempt(&mut self, mode: DbMode) -> bool {
        // stable-ledger: pick between the two stable selections by the flow
        // ledger — ±1 while shrink budget remains (removed < added + slack),
        // w/w+1 once exhausted. Bounds net DB drift below by -slack at any
        // size, without wasting hits on post-draw declines.
        const STABLE_LEDGER_SLACK: u64 = 64;
        let mode = if mode == DbMode::StableLedger {
            if self.stable_led_removed < self.stable_led_added + STABLE_LEDGER_SLACK {
                DbMode::Stable
            } else {
                DbMode::StableGrow
            }
        } else {
            mode
        };
        // band-ledger: full band while the size ledger is inside the slack,
        // corrective size skew once it drifts. Slack is proportional to the
        // circuit (0.2%, floor 64) so the correction scales with the run.
        self.band_led_round = mode == DbMode::BandLedger;
        let mode = if mode == DbMode::BandLedger {
            let slack = ((self.arena.len() as i64) / 500).max(64);
            if self.band_led > slack {
                DbMode::BandShrink
            } else if self.band_led < -slack {
                DbMode::BandGrow
            } else {
                DbMode::SizeAgnostic
            }
        } else {
            mode
        };
        self.db_seed_home = None;
        self.seed_from_pool = false;
        self.seed_fell_through = false;
        self.db_pair_round = false;
        let spliced = self.db_attempt_inner(mode);
        if spliced && self.db_pair_round {
            self.counters.pair_splices += 1;
        }
        // Canary accounting. A round QUALIFIES only when the seed genuinely
        // came from the pool; a heads coin that fell through because the pool
        // had drained is counted separately, because it means the rebuild is
        // too slow (scan more often) rather than that the material is
        // unreachable (stop the run) -- opposite remedies, so conflating them
        // would let a slow rescan masquerade as exhaustion.
        //
        // The canary SLEEPS under the brake: COMP declines far more often by
        // construction, and since this is a stop condition, mixing those
        // samples in would end runs for a reason that has nothing to do with
        // reachability.
        if self.seed_fell_through {
            self.counters.canary_fallthrough += 1;
        }
        if self.seed_from_pool && mode != DbMode::Compressing {
            let w = self.params.canary_window.max(1);
            if self.canary.len() == w {
                if self.canary.pop_front() == Some(true) {
                    self.canary_failures -= 1;
                }
            }
            self.canary.push_back(!spliced);
            if !spliced {
                self.canary_failures += 1;
            }
        }
        // A failed attempt must leave no trace: put the seed back. On success
        // the seed was consumed by the splice, so there is nothing to restore.
        if spliced {
            self.db_seed_home = None;
        } else {
            self.restore_seed();
        }
        spliced
    }

    /// The canary condition: the trailing window is FULL and the failure
    /// fraction in it exceeds `canary_theta`. Buffer-fullness is the
    /// minimum-sample guard, and the window is denominated in qualifying rounds
    /// rather than moves -- the right units, since the qualifying rate itself
    /// varies with the fall-through rate.
    pub fn canary_fired(&self) -> bool {
        let w = self.params.canary_window.max(1);
        if self.params.canary_theta <= 0.0 || self.canary.len() < w {
            return false;
        }
        self.canary_failures as f64 / self.canary.len() as f64 > self.params.canary_theta
    }

    /// Failure fraction currently in the canary window (0 until it fills).
    pub fn canary_frac(&self) -> f64 {
        if self.canary.is_empty() {
            0.0
        } else {
            self.canary_failures as f64 / self.canary.len() as f64
        }
    }

    pub(super) fn db_attempt_inner(&mut self, mode: DbMode) -> bool {
        let n = self.arena.len();
        let wmin = self.params.db_min_window.max(1);
        if n < wmin {
            return false;
        }
        // GEOMETRY FIRST, then length. The order used to be the other way
        // round -- the length was drawn here and each candidate window re-flipped
        // its own convex/contiguous coin inside `sample_window`. That made a
        // geometry-conditional length impossible to express, and it also let the
        // best-of-`litter_samples` selection compare windows drawn under
        // different geometries. One coin per round fixes both.
        // Pair coin first (docs/NONLOCAL_PHASE_A.md), and only for non-COMP
        // rounds: COMP admits only non-growing spellings, and with both bans
        // armed a commuting pair has no admissible same-length spelling, so a
        // COMP pair round could never splice. p_pair == 0 draws no RNG — the
        // stream is bit-identical to the pair-less chain.
        let geo = if mode != DbMode::Compressing
            && self.params.p_pair > 0.0
            && self.rng.random_bool(self.params.p_pair.clamp(0.0, 1.0))
        {
            DbSample::Pair
        } else if self.rng.random_bool(self.active_p_convex().clamp(0.0, 1.0)) {
            DbSample::Convex
        } else {
            DbSample::Contiguous
        };
        self.db_pair_round = geo == DbSample::Pair;
        if self.db_pair_round {
            self.counters.pair_rounds += 1;
        }
        let wmax = self.active_s_db(geo).max(1);
        // Prefix descent always starts at the top of the range — the descent
        // itself visits every shorter length, so sampling a shorter start
        // would only duplicate coverage.
        let wmax = if self.db_g57_only {
            self.params.s_db_g57.max(wmin)
        } else {
            wmax
        };
        let descend = self.active_prefixes();
        let len = if geo == DbSample::Pair {
            // A pair window is always the seed plus its partner; drawing a
            // shorter length would degenerate to a plain 1-gate re-spelling.
            wmax.min(n)
        } else if descend || wmin == wmax {
            wmax.min(n)
        } else {
            self.rng.random_range(wmin..=wmax.min(n))
        };

        // Sample the window under the geometry drawn above; g1dir drives the
        // incoming direction pivot below.
        let Some((ids, g1dir, smp)) = self.sample_best_window(len, geo) else {
            return false;
        };
        // Stamped into every --db-record attempt line (smp=ctg|cvx) so stats
        // can split hits by sampler geometry, esp. under --db-sample mixed.
        self.db_last_sampler = smp;
        self.geo_attempts[matches!(smp, DbSample::Convex) as usize] += 1;
        self.db_last_len = ids.len();
        Self::bump_len(&mut self.counters.len_attempts, ids.len(), 1);
        // Litter fragmentation census over the sampled window (observation only).
        let (distinct, full_litter) = self.litter_census(&ids);
        self.counters.litter_windows += 1;
        self.counters.litter_distinct_sum += distinct as u64;
        // Full-litter ban on the MAIN (no-descent) window path — the descent
        // path enforces it per rung and descends past banned rungs instead: a
        // window that is exactly one complete litter is where the store is
        // most likely to hand back the spelling that made it — refuse it.
        if self.params.litter_ban && full_litter && !descend {
            self.counters.litter_banned += 1;
            self.count_db_miss(mode);
            return false;
        }
        let window: Vec<XGate> = ids.iter().map(|&id| self.arena.gate(id).clone()).collect();

        // Prefix descent, largest first: try the full k-gate window, then the
        // (k-1)-gate prefix, and so on down to db_min_window; splice the
        // LONGEST prefix with a usable match (max rewrite per round). Every
        // prefix attempt is recorded and counted. In dry-run the descent runs
        // to the bottom recording hits without splicing (full measurement);
        // live, the first hit splices and ends the round. A span-cap or
        // wide-verify decline keeps descending — shorter prefixes span fewer
        // wires and may still match.
        if descend {
            let wmin = 1usize;
            let guard = DegreeGuard {
                max_degree: self.params.db_max_degree,
                probes: self.params.db_degree_probes,
            };
            // Shrink from whichever end is FARTHER from the seed, so the seed
            // survives to the shortest rung. Dropping from a fixed end walks
            // away from the very gate the descent exists to re-encode: the seed
            // sits at the left edge only when its own direction is R, so a
            // fixed-end descent abandons it immediately on about half of
            // contiguous windows, and on every convex one, where the block
            // grows outward in both directions around it.
            let seed = self.db_seed_home.map(|(id, _)| id);
            let k = seed
                .and_then(|sd| ids.iter().position(|&x| x == sd))
                .unwrap_or(0);
            // Store-routing schedule for this descent.
            //
            // ONE-PASS CASCADE: at each window length, probe curated, then
            // fall back to regular before shortening. So a regular hit at
            // length p wins over a curated hit at length p-1 -- the descent
            // can never see the shorter curated rung.
            //
            // --curated-exhaust (two passes): pass 0 walks EVERY length against
            // curated alone; only a descent that came up completely empty runs
            // pass 1 against the regular store. Curated material is preferred
            // at any length over regular material at a longer one. The price is
            // that a full curated miss re-canonicalizes the whole descent, and
            // db_attempts counts both passes -- both are real costs of the
            // policy, so neither is hidden from the counters.
            let cur_ok =
                self.params.curated && (mode != DbMode::Compressing || self.params.curated_in_comp);
            let passes: &[(bool, bool)] = if self.params.curated_exhaust && cur_ok {
                &[(true, false), (false, true)]
            } else {
                &[(cur_ok, true)]
            };
            for &(cur_armed, reg_fb) in passes {
                let (mut lo, mut hi) = (0usize, window.len() - 1);
                loop {
                    let p = hi - lo + 1;
                    if p < wmin {
                        break;
                    }
                    let prefix = &window[lo..=hi];
                    // Full-litter ban: this rung is exactly the set some earlier
                    // replacement emitted, so it is where the store is most likely
                    // to hand that spelling straight back. Descending past it is
                    // free -- a shorter rung is no longer a complete litter.
                    if self.params.litter_ban && self.litter_census(&ids[lo..=hi]).1 {
                        self.counters.litter_banned += 1;
                        if p == wmin {
                            break;
                        }
                        if k.saturating_sub(lo) >= hi.saturating_sub(k) {
                            lo += 1;
                        } else {
                            hi -= 1;
                        }
                        continue;
                    }
                    self.counters.db_attempts += 1;
                    if self.params.db_max_span > 0
                        && crate::canonicalization::xgate::xgate_used_wires_len(prefix)
                            > self.params.db_max_span
                    {
                        self.counters.db_span_skips += 1;
                        self.record_db_attempt(prefix, 0, None);
                        self.count_db_miss(mode);
                        // The descent must SHRINK on every exit path. Falling
                        // through to `continue` without moving lo/hi respins the
                        // identical prefix forever -- and since a store miss is
                        // the common case, that hangs the run before it completes
                        // a single move.
                        if p == wmin {
                            break;
                        }
                        if k.saturating_sub(lo) >= hi.saturating_sub(k) {
                            lo += 1;
                        } else {
                            hi -= 1;
                        }
                        continue;
                    }
                    let res = db_replace(
                        prefix,
                        self.num_wires,
                        &self.db,
                        self.db_budget,
                        mode,
                        guard,
                        cur_armed,
                        self.params.curated_in_comp,
                        reg_fb,
                        self.params.mix_pay_random,
                        self.db_pair_round,
                        &mut self.rng,
                    );
                    self.counters.db_identity_skips += res.identity_skipped as u64;
                    self.counters.pair_perm_skips += res.permutation_skipped as u64;
                    if res.chosen.is_some() && res.chosen_curated {
                        self.counters.db_curated_hits += 1;
                    }
                    if res.chosen.is_some() {
                        self.note_choice(res.choice_count);
                    }
                    if let Some(ml) = res.min_match_len {
                        self.counters.dmin_windows += 1;
                        if ml < prefix.len() {
                            self.counters.dmin_shorter += 1;
                        }
                    }
                    if res.degree_skipped {
                        self.counters.db_degree_skips += 1;
                        let k = self.db_last_len;
                        Self::bump_len(&mut self.counters.len_deg_skip, k, 1);
                    }
                    let Some(replacement) = res.chosen else {
                        self.record_db_attempt(prefix, res.match_count, None);
                        self.count_db_miss(mode);
                        // The descent must SHRINK on every exit path. Falling
                        // through to `continue` without moving lo/hi respins the
                        // identical prefix forever -- and since a store miss is
                        // the common case, that hangs the run before it completes
                        // a single move.
                        if p == wmin {
                            break;
                        }
                        if k.saturating_sub(lo) >= hi.saturating_sub(k) {
                            lo += 1;
                        } else {
                            hi -= 1;
                        }
                        continue;
                    };
                    if self.params.db_dry_run {
                        self.record_db_attempt(prefix, res.match_count, None);
                        self.count_db_hit(mode);
                        // The descent must SHRINK on every exit path. Falling
                        // through to `continue` without moving lo/hi respins the
                        // identical prefix forever -- and since a store miss is
                        // the common case, that hangs the run before it completes
                        // a single move.
                        if p == wmin {
                            break;
                        }
                        if k.saturating_sub(lo) >= hi.saturating_sub(k) {
                            lo += 1;
                        } else {
                            hi -= 1;
                        }
                        continue;
                    }
                    if self.try_db_splice_curated(
                        res.chosen_curated,
                        &ids[lo..=hi],
                        g1dir,
                        prefix,
                        replacement,
                        res.match_count,
                        mode,
                    ) {
                        return true;
                    }
                    if p == wmin {
                        break;
                    }
                    if k.saturating_sub(lo) >= hi.saturating_sub(k) {
                        lo += 1;
                    } else {
                        hi -= 1;
                    }
                }
            }
            return false;
        }

        self.counters.db_attempts += 1;

        // Span guard: canonicalizing a wide-span window is the dominant cost
        // (Rule-L over large tied wire groups), and the store holds nothing
        // that wide — record the (near-certain) miss and move on.
        if self.params.db_max_span > 0
            && crate::canonicalization::xgate::xgate_used_wires_len(&window)
                > self.params.db_max_span
        {
            self.counters.db_span_skips += 1;
            let k = self.db_last_len;
            Self::bump_len(&mut self.counters.len_span_skip, k, 1);
            self.record_db_attempt(&window, 0, None);
            match mode {
                DbMode::Compressing => self.counters.db_comp_misses += 1,
                DbMode::SizeAgnostic
                | DbMode::MinGrow
                | DbMode::Mix
                | DbMode::Stable
                | DbMode::StableGrow
                | DbMode::StableLedger
                | DbMode::Same
                | DbMode::BandLedger
                | DbMode::BandShrink
                | DbMode::BandGrow => self.counters.db_agn_misses += 1,
            }
            return false;
        }

        let guard = DegreeGuard {
            max_degree: self.params.db_max_degree,
            probes: self.params.db_degree_probes,
        };
        let res = db_replace(
            &window,
            self.num_wires,
            &self.db,
            self.db_budget,
            mode,
            guard,
            self.params.curated,
            self.params.curated_in_comp,
            true,
            self.params.mix_pay_random,
            self.db_pair_round,
            &mut self.rng,
        );
        self.counters.db_identity_skips += res.identity_skipped as u64;
        self.counters.pair_perm_skips += res.permutation_skipped as u64;
        if res.chosen.is_some() {
            self.note_choice(res.choice_count);
        }
        if let Some(ml) = res.min_match_len {
            self.counters.dmin_windows += 1;
            if ml < window.len() {
                self.counters.dmin_shorter += 1;
            }
        }
        if res.chosen.is_some() && res.chosen_curated {
            self.counters.db_curated_hits += 1;
        }
        let match_count = res.match_count;
        if res.degree_skipped {
            self.counters.db_degree_skips += 1;
        }

        // Measurement mode: record the window + match count, never mutate.
        if self.params.db_dry_run {
            self.record_db_attempt(&window, match_count, None);
            if match_count > 0 {
                match mode {
                    DbMode::Compressing => self.counters.db_comp_hits += 1,
                    DbMode::SizeAgnostic
                    | DbMode::MinGrow
                    | DbMode::Mix
                    | DbMode::Stable
                    | DbMode::StableGrow
                    | DbMode::StableLedger
                    | DbMode::Same
                    | DbMode::BandLedger
                    | DbMode::BandShrink
                    | DbMode::BandGrow => self.counters.db_agn_hits += 1,
                }
            } else {
                match mode {
                    DbMode::Compressing => self.counters.db_comp_misses += 1,
                    DbMode::SizeAgnostic
                    | DbMode::MinGrow
                    | DbMode::Mix
                    | DbMode::Stable
                    | DbMode::StableGrow
                    | DbMode::StableLedger
                    | DbMode::Same
                    | DbMode::BandLedger
                    | DbMode::BandShrink
                    | DbMode::BandGrow => self.counters.db_agn_misses += 1,
                }
            }
            return false;
        }

        let curated_pick = res.chosen_curated;
        let mlen = res.min_match_len;
        let res_fwd_key = res.fwd_key;
        let Some(replacement) = res.chosen else {
            self.record_db_attempt(&window, match_count, None);
            self.count_db_miss(mode);
            return false;
        };
        let fwd_key = res_fwd_key;
        let hit = self.try_db_splice_curated(
            curated_pick,
            &ids,
            g1dir,
            &window,
            replacement,
            match_count,
            mode,
        );
        if hit {
            self.geo_hits[matches!(self.db_last_sampler, DbSample::Convex) as usize] += 1;
            // True class of a big-pool conversion, via the reference store.
            if curated_pick && match_count > 20 {
                if let (Some(db), Some(k)) = (self.runtime.reference_db(), fwd_key) {
                    if let Some(v) = db.get_regular(&k) {
                        let mut mn = usize::MAX;
                        let mut pos = 0usize;
                        while pos < v.len() {
                            let l = v[pos] as usize;
                            if l == 0 || l % 3 != 0 || pos + 1 + l > v.len() {
                                break;
                            }
                            mn = mn.min(l / 3);
                            pos += 1 + l;
                        }
                        if mn != usize::MAX {
                            self.m123_class_hist[mn.min(31)] += 1;
                        }
                    }
                }
            }
            // Complexity of the converted permutation: the store's minimal
            // matching spelling for this window (the operational "smallest
            // number of gates that computes it").
            if let Some(ml) = mlen {
                self.dmin_success_hist[ml.min(31)] += 1;
            }
        }
        hit
    }

    /// Bump one per-length bucket, growing the vector on demand.
    pub(super) fn bump_len(v: &mut Vec<u64>, k: usize, by: u64) {
        let k = k.min(LEN_HIST_MAX);
        if v.len() <= LEN_HIST_MAX {
            v.resize(LEN_HIST_MAX + 1, 0);
        }
        v[k] += by;
    }

    pub(super) fn count_db_hit(&mut self, mode: DbMode) {
        match mode {
            DbMode::Compressing => self.counters.db_comp_hits += 1,
            DbMode::SizeAgnostic
            | DbMode::MinGrow
            | DbMode::Mix
            | DbMode::Stable
            | DbMode::StableGrow
            | DbMode::StableLedger
            | DbMode::Same
            | DbMode::BandLedger
            | DbMode::BandShrink
            | DbMode::BandGrow => self.counters.db_agn_hits += 1,
        }
        let k = self.db_last_len;
        Self::bump_len(&mut self.counters.len_hits, k, 1);
    }

    pub(super) fn count_db_miss(&mut self, mode: DbMode) {
        match mode {
            DbMode::Compressing => self.counters.db_comp_misses += 1,
            DbMode::SizeAgnostic
            | DbMode::MinGrow
            | DbMode::Mix
            | DbMode::Stable
            | DbMode::StableGrow
            | DbMode::StableLedger
            | DbMode::Same
            | DbMode::BandLedger
            | DbMode::BandShrink
            | DbMode::BandGrow => self.counters.db_agn_misses += 1,
        }
    }

    // Verify (optionally), record, and splice `replacement` over the window
    // nodes `ids` (whose gates are `window`). Returns false only when the
    // combined support exceeds verify_rewrite's 24-wire cap with verification
    // on — declined rather than spliced unchecked, recorded as a miss.
    pub(super) fn try_db_splice_curated(
        &mut self,
        from_curated: bool,
        ids: &[u32],
        g1dir: Dir,
        window: &[XGate],
        replacement: Vec<XGate>,
        match_count: usize,
        mode: DbMode,
    ) -> bool {
        // Optional exhaustive equivalence check on the combined support.
        // verify_rewrite caps support at 24 wires; with verification on, a wider
        // window is not checkable so we decline it rather than splice unchecked.
        if self.params.db_verify {
            let mut support: Vec<u16> = window
                .iter()
                .chain(replacement.iter())
                .flat_map(|g| std::iter::once(g.target).chain(g.ctrls.iter().map(|&(w, _)| w)))
                .collect();
            support.sort_unstable();
            support.dedup();
            let _vt = std::time::Instant::now();
            // Past verify_rewrite's 24-wire exhaustive ceiling, verify by ANF
            // instead of by evaluation: cost is bounded by the polynomial term
            // count rather than 2^support, and it is still a proof. Measured
            // on the production store, the >= 24-wire slice is 1.3% of entries
            // (all 10-11 gates), max degree 7, and carries at most 233 terms --
            // so this is a comparison where the exhaustive check would have
            // been 16.7M evaluations. Only an UNDECIDED result (support > 64
            // wires, or a budget hit) still declines.
            let _vok = if support.len() > 24 {
                match crate::stages::db_mixing::replacement::polys_equivalent(
                    window,
                    &replacement,
                    self.db_budget,
                ) {
                    Some(ok) => {
                        self.counters.db_wide_poly += 1;
                        ok
                    }
                    None => {
                        self.counters.db_wide_skip += 1;
                        self.record_db_attempt(window, match_count, None);
                        self.count_db_miss(mode);
                        return false;
                    }
                }
            } else {
                rules::verify_rewrite(window, &replacement)
            };
            crate::canonicalization::xgate::VERIFY_NS.fetch_add(
                _vt.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            if !_vok {
                // A CURATED replacement that fails verification is refused and
                // counted, not fatal. Curated entries are halves of split
                // minimal identities, and a half need not restore the helper
                // wires it borrows -- the observed failures write a wire the
                // window never touched. Until that invariant is pinned down,
                // treat curated as best-effort and let the run continue on the
                // regular store. A REGULAR failure stays fatal: there it would
                // mean the store or the canonicalisation is wrong, which is not
                // something to paper over.
                assert!(
                    from_curated,
                    "DB replacement verification failed: {window:?} -> {replacement:?}"
                );
                self.counters.db_curated_rejected += 1;
                self.record_db_attempt(window, match_count, None);
                self.count_db_miss(mode);
                return false;
            }
        }

        self.record_db_attempt(window, match_count, Some(&replacement));

        // Size accounting (replacement may be shorter, equal, or longer).
        let old = window.len();
        let new = replacement.len();
        // Ledger flows for the stable family (only ±1 by construction).
        if matches!(mode, DbMode::Stable | DbMode::StableGrow) {
            if new > old {
                self.stable_led_added += (new - old) as u64;
            } else if new < old {
                self.stable_led_removed += (old - new) as u64;
            }
        }
        // band-ledger drift accounting (signed, over every band conversion).
        if self.band_led_round {
            self.band_led += new as i64 - old as i64;
        }
        // Big-pool (M1/M2/M3) usage: only the swapped pools carry more than
        // the bounded contract's 20 candidates, so this count is exact.
        if from_curated && match_count > 20 {
            self.bigpool_hits += 1;
        }
        if new <= old {
            Self::bump_len(&mut self.counters.len_removed, old, (old - new) as u64);
            self.counters.db_gates_removed += (old - new) as u64;
            if mode == DbMode::Compressing {
                self.counters.db_cmp_removed += (old - new) as u64;
            } else {
                self.counters.db_mix_removed += (old - new) as u64;
            }
        } else {
            Self::bump_len(&mut self.counters.len_added, old, (new - old) as u64);
            self.counters.db_gates_added += (new - old) as u64;
            if mode == DbMode::Compressing {
                self.counters.db_cmp_added += (new - old) as u64;
            } else {
                self.counters.db_mix_added += (new - old) as u64;
            }
        }
        self.count_db_hit(mode);

        // Splice: insert the replacement after the node left of the window, then
        // unlink/free every window node (bumping stamps, which invalidates any
        // journal undo entry that referenced them).
        let cursor = self.arena.neighbor(ids[0], Dir::L);
        // Origin: keep the shared ancestor if the whole window agrees, else mark
        // the rewritten material synthetic (a DB block spans mixed lineage).
        let m0 = self.meta_of(ids[0]);
        let same_origin = ids.iter().all(|&id| self.meta_of(id).origin == m0.origin);
        let origin = if same_origin { m0.origin } else { ORIGIN_SYNTH };
        // Gen: a DB-replacement increment site. Products get g+1 where g is
        // the MEDIAN generation of the outgoing window — upper middle of the
        // sorted gens by default (median rounded up on even sizes, benchmark
        // semantics 2026-07-21), lower middle under gen_median_low.
        // Saturating: an all-fresh window stays fresh.
        let dgen = {
            let mut gens: Vec<u32> = ids.iter().map(|&id| self.meta_of(id).dgen).collect();
            gens.sort_unstable();
            let mid = if self.params.gen_median_low {
                gens.len().saturating_sub(1) / 2
            } else {
                gens.len() / 2
            };
            gens.get(mid)
                .copied()
                .unwrap_or(GEN_FRESH)
                .saturating_add(1)
        };
        // Would an ssg-style full-litter ban have refused this splice? Counted,
        // not enforced.
        if self.litter_census(ids).1 {
            self.counters.litter_full_spliced += 1;
        }
        for &id in ids {
            self.evict_taps(id);
            self.index_remove(id);
            self.arena.unlink(id);
            self.arena.free_node(id);
        }
        // Incoming-gate directions: split the replacement at a pivot so the block
        // shoots outward from a point set by g1's original direction. g1 left ->
        // pivot at floor(2m/3); g1 right -> floor(m/3). Gates up to the pivot
        // (inclusive) head left, the rest head right.
        let m = replacement.len();
        let pivot = if g1dir == Dir::L { (2 * m) / 3 } else { m / 3 };
        // The splice is the litter-creating event: every product carries one
        // fresh id and the size this replacement emitted. A later window that
        // is exactly this set is the case where the store can hand the outgoing
        // spelling straight back (A -> B -> A).
        let litter = {
            let srcs: Vec<u64> = ids.iter().map(|&id| self.meta_of(id).litter).collect();
            self.anc_union_litter(&srcs)
        };
        let litter_size = m.min(u16::MAX as usize) as u16;
        {
            // Joint (outgoing, incoming) size of this splice.
            let (o, i) = (ids.len().min(SPLICE_HIST_MAX), m.min(SPLICE_HIST_MAX));
            if self.counters.splice_sizes.is_empty() {
                self.counters.splice_sizes =
                    vec![vec![0u64; SPLICE_HIST_MAX + 1]; SPLICE_HIST_MAX + 1];
            }
            self.counters.splice_sizes[o][i] += 1;
            if from_curated {
                if self.counters.splice_sizes_curated.is_empty() {
                    self.counters.splice_sizes_curated =
                        vec![vec![0u64; SPLICE_HIST_MAX + 1]; SPLICE_HIST_MAX + 1];
                }
                self.counters.splice_sizes_curated[o][i] += 1;
            }
        }
        let mut c = cursor;
        let mut placed: Vec<u32> = Vec::with_capacity(m);
        for (i, gate) in replacement.into_iter().enumerate() {
            c = self.arena.insert_after(c, gate);
            self.index_add(c);
            let d = if i <= pivot { Dir::L } else { Dir::R };
            self.set_meta(
                c,
                Meta {
                    origin,
                    event: 0,
                    dir: d,
                    dgen,
                    litter,
                    litter_size,
                },
            );
            placed.push(c);
        }
        // Products ride their assigned direction outward, exactly as split
        // pieces do. Float-only, so the function is preserved by construction;
        // it also scatters the litter, which makes a later window less likely
        // to be exactly this replacement.
        if self.params.db_advance {
            self.advance_births(&placed);
        }
        true
    }

    // Append one record to --db-record: the outgoing window, the number of
    // equivalent DB circuits, and (on success) the replacing subcircuit. Gates
    // are printed as `target:comp:ctrl(pol)...`, one circuit per line.
    pub(super) fn record_db_attempt(
        &mut self,
        window: &[XGate],
        matches: usize,
        repl: Option<&[XGate]>,
    ) {
        use std::io::Write;
        let smp = match self.db_last_sampler {
            DbSample::Convex => "cvx",
            DbSample::Pair => "pair",
            DbSample::Bridge => "brg",
            _ => "ctg",
        };
        let Some(w) = self.db_record.as_mut() else {
            return;
        };
        fn fmt(gates: &[XGate]) -> String {
            gates
                .iter()
                .map(|g| {
                    let ctrls: Vec<String> = g
                        .ctrls
                        .iter()
                        .map(|&(wire, p)| format!("{wire}{}", if p { "+" } else { "-" }))
                        .collect();
                    format!("{}:{}:{}", g.target, g.comp as u8, ctrls.join(","))
                })
                .collect::<Vec<_>>()
                .join(" ")
        }
        let _ = writeln!(
            w,
            "attempt mv={} matches={} replaced={} smp={}",
            self.moves_done,
            matches,
            repl.is_some() as u8,
            smp
        );
        let _ = writeln!(w, "  out {}", fmt(window));
        if let Some(r) = repl {
            let _ = writeln!(w, "  in  {}", fmt(r));
        }
    }
}
