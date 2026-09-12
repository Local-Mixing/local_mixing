//! Whole-circuit scheduling, size controller, stopping and progress snapshots.
use super::*;

/// The profile's target size at effective work `eff`: linear ramp to
/// r1*s_in over [0, n0], hold to n1, linear ramp to r2*s_in over [n1, n2],
/// r2*s_in thereafter.
pub fn prof_target(n: [f64; 3], r: [f64; 2], s_in: f64, eff: f64) -> f64 {
    let (n0, n1, n2) = (n[0], n[1], n[2]);
    let (r1, r2) = (r[0], r[1]);
    if eff <= 0.0 {
        s_in
    } else if eff < n0 {
        s_in * (1.0 + (r1 - 1.0) * (eff / n0))
    } else if eff < n1 {
        s_in * r1
    } else if eff < n2 {
        s_in * (r1 + (r2 - r1) * ((eff - n1) / (n2 - n1)))
    } else {
        s_in * r2
    }
}

impl Mixer {
    // ---- the chain ----

    // Layer-2 overlay: the DB knobs for the round's LIVE mode. With the mode
    // overlay off, db_mode_cur is fixed and these return the base params, so a
    // single-mode run behaves exactly as before. Only COMP-mode reads the
    // overrides, and only when they are set.
    /// Window length for the round's live mode AND the geometry just drawn.
    /// Precedence, most specific first: mode+geometry (`s_db_comp_ctg` /
    /// `s_db_ctg`) -> mode (`s_db_comp`) -> base (`s_db`). A 0 anywhere means
    /// "not set, fall through".
    // Every one of these delegates to MixParams::db_knobs, so the resolution
    // rules exist in exactly ONE place. They are called per DB round, but the
    // work is a handful of Option::or on Copy types -- nothing next to the
    // milliseconds a canonicalization costs.
    pub(super) fn active_s_db(&self, geo: DbSample) -> usize {
        let k = self.params.db_knobs(self.db_mode_cur);
        match geo {
            DbSample::Convex => k.s_db_cvx,
            DbSample::Contiguous => k.s_db_ctg,
            // A pair window is exactly the seed plus its partner; the length
            // knobs do not apply. Bridge windows are likewise always 2 (and
            // never drawn by the geometry coin — the tag only reaches here
            // through record paths).
            DbSample::Pair | DbSample::Bridge => 2,
        }
    }
    pub(super) fn active_prefixes(&self) -> bool {
        self.params.db_knobs(self.db_mode_cur).prefixes
    }
    pub(super) fn active_p_convex(&self) -> f64 {
        self.params.db_knobs(self.db_mode_cur).p_convex
    }
    pub(super) fn active_p_mingen(&self) -> f64 {
        self.params.db_knobs(self.db_mode_cur).p_mingen
    }

    // Layer-2 mode overlay (slot 0): pick this round's DB mode by coin when
    // armed (p_mix >= 0) -- MIX-DB with probability p_mix, else COMP-DB. This is
    // the "set parameter" rule of the slot-0 condition engine, independent of
    // the size brake and the thermostat. Off (p_mix < 0) leaves db_mode_cur as
    // the fixed db_mode (possibly steered by the size brake).
    pub(super) fn apply_mode_overlay(&mut self) {
        if self.params.p_mix < 0.0 {
            return;
        }
        // Same rule as prof_tick: the MIX side is the configured re-encode
        // mode, so an explicit --db-mode survives the overlay.
        let mix_side = match self.params.db_mode {
            DbMode::Compressing => DbMode::Mix,
            m => m,
        };
        self.db_mode_cur = if self.rng.random_bool(self.params.p_mix.clamp(0.0, 1.0)) {
            mix_side
        } else {
            DbMode::Compressing
        };
    }

    /// Record the branching factor of one successful splice's selection.
    pub(super) fn note_choice(&mut self, k: usize) {
        if k == 0 {
            return;
        }
        self.counters.choice_splices += 1;
        self.counters.choice_sum += k as u64;
        if k > 1 {
            self.counters.choice_multi += 1;
            self.counters.choice_bits_milli += ((k as f64).log2() * 1000.0).round() as u64;
        }
    }

    pub(super) fn prof_init(&mut self) {
        let s_in = self.original.len() as f64;
        // Phase priors until the first plant estimates arrive: full MIX for
        // the expansion leg (matching pay-random's strong up-lever).
        self.prof = Some(ProfState {
            phase: 1,
            // Moderate prior: the plant is unknown for exactly one interval,
            // and a hot prior overshoots the expansion ramp before the first
            // update can react (measured: p_mix=1 grows ~1.7 gates/move on a
            // fresh 100k gadget, ~3x the steepest ramp anyone asks for).
            pmix: 0.5,
            integ: 0.0,
            ghat: 0.0,
            shat: 0.0,
            dhat: 0.0,
            eff: 0.0,
            // First update comes early (quarter cadence) so the plant is
            // identified before much work is spent on a guess.
            next_eff: self.params.prof_cadence_eff.max(0.05) * 0.25,
            sat: 0,
            s_in,
            base_moves: self.moves_done,
            base_size: s_in,
            base_pmix: 0.5,
            base_mix: [0; 4],
            base_cmp: [0; 4],
        });
        self.prof_snapshot();
        eprintln!(
            "[fmix] profile ON: n={:?} r={:?} s_in={} cadence_eff={} deadband={} dp_max={}",
            self.params.prof_n,
            self.params.prof_r,
            s_in,
            self.params.prof_cadence_eff,
            self.params.prof_deadband,
            self.params.prof_dp_max
        );
    }

    pub(super) fn prof_snapshot(&mut self) {
        let c = &self.counters;
        let mix = [
            c.db_agn_hits,
            c.db_agn_misses,
            c.db_mix_added,
            c.db_mix_removed,
        ];
        let cmp = [
            c.db_comp_hits,
            c.db_comp_misses,
            c.db_cmp_added,
            c.db_cmp_removed,
        ];
        let size = self.arena.len() as f64;
        let moves = self.moves_done;
        if let Some(p) = self.prof.as_mut() {
            p.base_moves = moves;
            p.base_size = size;
            p.base_pmix = p.pmix;
            p.base_mix = mix;
            p.base_cmp = cmp;
        }
    }

    /// Per-round profile bookkeeping: accumulate effective work, run the
    /// controller at its cadence, and flip the MIX/COMP coin at the
    /// controller's current lever.
    pub(super) fn prof_tick(&mut self) {
        let size = self.arena.len().max(1) as f64;
        let (due, pmix) = {
            let p = self.prof.as_mut().expect("prof_tick with prof off");
            p.eff += 1.0 / size;
            (p.eff >= p.next_eff, p.pmix)
        };
        if due {
            self.prof_update();
        }
        let pm = if due {
            self.prof.as_ref().unwrap().pmix
        } else {
            pmix
        };
        // The MIX side of the coin is the CONFIGURED re-encode mode, not a
        // hardcoded Mix: --db-mode stable/stable-grow/stable-ledger must
        // survive the profile overlay (it used to be stomped here every
        // round, silently running plain Mix). Runs without an explicit
        // --db-mode are unchanged (their configured mode IS Mix).
        let mix_side = match self.params.db_mode {
            DbMode::Compressing => DbMode::Mix,
            m => m,
        };
        self.db_mode_cur = if self.rng.random_bool(pm.clamp(0.0, 1.0)) {
            mix_side
        } else {
            DbMode::Compressing
        };
    }

    /// One controller update (every prof_cadence_eff of effective work):
    /// refresh the plant estimates from the interval's per-mode counters,
    /// advance the phase, retarget the thermostat, and move the lever by
    /// feed-forward inversion plus a small integral correction — deadbanded,
    /// rate-limited, clamped, with saturation logged (best-effort contract).
    pub(super) fn prof_update(&mut self) {
        let s = self.arena.len().max(1) as f64;
        let c = &self.counters;
        let mix_now = [
            c.db_agn_hits,
            c.db_agn_misses,
            c.db_mix_added,
            c.db_mix_removed,
        ];
        let cmp_now = [
            c.db_comp_hits,
            c.db_comp_misses,
            c.db_cmp_added,
            c.db_cmp_removed,
        ];
        let n = self.params.prof_n;
        let r = self.params.prof_r;
        let (cad, dead, dp_max, ew, ki) = (
            self.params.prof_cadence_eff.max(0.05),
            self.params.prof_deadband,
            self.params.prof_dp_max,
            self.params.prof_ewma,
            self.params.prof_ki,
        );
        if let Some(p) = self.prof.as_mut() {
            // ---- Plant identification over the interval, in gates/move ----
            // The lever splits rounds into MIX (w.p. p) and COMP (w.p. 1-p),
            // so per-move contributions are normalised by the p that was
            // actually in force; an arm that saw too little of the interval
            // keeps its previous estimate rather than dividing by ~0.
            let dmoves = (self.moves_done - p.base_moves).max(1) as f64;
            let p_used = p.base_pmix;
            let net_mix = (mix_now[2] - p.base_mix[2]) as f64 - (mix_now[3] - p.base_mix[3]) as f64;
            let net_cmp = (cmp_now[3] - p.base_cmp[3]) as f64 - (cmp_now[2] - p.base_cmp[2]) as f64;
            if p_used > 0.05 {
                let g_new = net_mix / (dmoves * p_used);
                p.ghat = if p.ghat == 0.0 {
                    g_new
                } else {
                    (1.0 - ew) * p.ghat + ew * g_new
                };
            }
            if p_used < 0.95 {
                let s_new = net_cmp / (dmoves * (1.0 - p_used));
                p.shat = if p.shat == 0.0 {
                    s_new
                } else {
                    (1.0 - ew) * p.shat + ew * s_new
                };
            }
            // The DISTURBANCE: observed total drift minus what the DB move
            // accounts for. Twists live here (they add gates at a rate the
            // controller never models), along with expansion moves and
            // thermostat contractions. Estimating it is what lets a profile
            // run at any twist rate.
            let v_obs = (s - p.base_size) / dmoves;
            let v_db = (net_mix - net_cmp) / dmoves;
            let d_new = v_obs - v_db;
            p.dhat = if p.dhat == 0.0 {
                d_new
            } else {
                (1.0 - ew) * p.dhat + ew * d_new
            };
            // Phase machine (best-effort: eff marks OR size arrival).
            let old_phase = p.phase;
            match p.phase {
                1 if p.eff >= n[0] || s >= 0.99 * r[0] * p.s_in => p.phase = 2,
                2 if p.eff >= n[1] => p.phase = 3,
                3 if p.eff >= n[2] || s <= 1.01 * r[1] * p.s_in => p.phase = 4,
                _ => {}
            }
            if p.phase != old_phase {
                p.integ = 0.0;
                p.sat = 0;
                eprintln!(
                    "[fmix] profile: phase {} -> {} at eff={:.2} size={}",
                    old_phase, p.phase, p.eff, s as usize
                );
                // Exact-point recovery: FMIX_STOP_AT_PHASE=<k> finishes the
                // run cleanly the moment the schedule enters phase k. A
                // deterministic replay (same seed/input/flags) stopped this
                // way recovers the ORIGINAL run's circuit at the leg boundary
                // with zero overshoot — the transition is detected at the
                // same controller update in both runs.
                if self.runtime.stop_at_phase() == Some(p.phase as u32) {
                    println!(
                        "[fmix] FMIX_STOP_AT_PHASE={}: stopping cleanly at the phase boundary (eff={:.2} size={})",
                        p.phase, p.eff, s as usize
                    );
                    self.stop_requested = true;
                }
            }
            // Feasibility diagnosis for the compression leg. With the lever
            // fully down the best available drift is dhat - shat; when the
            // disturbance (twists, chiefly) exceeds what COMP can remove, the
            // circuit grows no matter what the controller does. Say so once,
            // in those terms, instead of leaving a silent saturation.
            if p.phase == 3 && p.dhat > p.shat && p.sat == 4 {
                eprintln!(
                    "[fmix] profile: COMPRESSION INFEASIBLE — disturbance {:+.4} gates/move (twists et al.) exceeds max COMP removal {:.4}; the lever is pinned at 0 and the circuit still grows. Lower the twist rate or relax R2.",
                    p.dhat, p.shat
                );
            }
            let s_star = prof_target(n, r, p.s_in, p.eff);
            let s_star_next = prof_target(n, r, p.s_in, p.eff + cad);
            let err = (s - s_star) / s_star.max(1.0);
            if err.abs() >= dead {
                p.integ = (p.integ - ki * err).clamp(-0.3, 0.3);
                // Feed-forward: solve p*ghat - (1-p)*shat + dhat = v_star for
                // p, where v_star is the slope that lands on the setpoint one
                // cadence ahead. The disturbance enters as a constant offset,
                // so a twist-heavy run simply gets a lower p_mix.
                let horizon = (cad * s).max(1.0);
                let v_star = (s_star_next - s) / horizon;
                let denom = p.ghat + p.shat;
                let mut want = if denom.abs() > 1e-9 {
                    (v_star - p.dhat + p.shat) / denom
                } else {
                    p.pmix
                };
                want += p.integ;
                // Rate limit, with an escape hatch: far from the profile the
                // lever may move freely (a 0.1/step crawl cannot catch a ramp
                // that is 6 eff units long), near it the limit binds and the
                // loop stays gentle.
                let far = err.abs() > 4.0 * dead.max(1e-6);
                let lim = if far { 1.0 } else { dp_max };
                let dp = (want - p.pmix).clamp(-lim, lim);
                p.pmix = (p.pmix + dp).clamp(0.0, 1.0);
            }
            // Saturation: pinned lever while clearly behind the profile.
            if (p.pmix >= 0.999 && err < -0.10) || (p.pmix <= 0.001 && err > 0.10) {
                p.sat += 1;
                if p.sat == 5 {
                    eprintln!(
                        "[fmix] profile: SATURATED (phase {} pmix={:.3} size={} S*={:.0}) — best-effort, continuing pinned",
                        p.phase, p.pmix, s as usize, s_star
                    );
                }
            } else {
                p.sat = 0;
            }
            p.next_eff += cad;
            // The thermostat is conscripted: pull toward the moving setpoint.
            self.params.target_size = s_star.max(2.0) as usize;
        }
        self.prof_snapshot();
    }

    // One first-class twist round. The twist is always from the swap family;
    // `twist_move` rolls the (alpha, beta) negation coins internally, giving
    // swap 1/4, swap+negate-one 1/2, swap+negate-both 1/4. The legacy
    // w_twist_* weights are retired (accepted-but-ignored on the CLI).
    pub(super) fn twist_round(&mut self) {
        // Layer-1 dispatch: the split twist first (forced while the stage is
        // live), then the existing g57/swap-family choice. After a --split
        // run's boundary the live dispatch is ZERO (docs §3) — part 2 runs no
        // further split twists even if --p-split-twist was set; the CLI value
        // is the standalone (no --split) layer-1 mode.
        let p_st = if self.params.split && self.split_on {
            1.0
        } else if self.params.split && self.split_done {
            0.0
        } else {
            self.params.p_split_twist
        };
        if p_st > 0.0 && self.rng.random_bool(p_st.clamp(0.0, 1.0)) {
            self.split_twist_move();
            return;
        }
        if self.params.twist_g57 {
            self.twist_move_g57();
        } else {
            self.twist_move();
        }
    }

    /// The size brake. Growth past `size_hi` forces slot 2 into COMP; it is
    /// released back to `params.db_mode` at `size_lo`, OR earlier when COMP has
    /// stopped paying.
    ///
    /// The productivity release is what makes a WIDE band safe, and a wide band
    /// is what the transport argument wants: growth legs are where material
    /// actually moves. The danger was never the width but sitting in COMP past
    /// its usefulness, where it starves -- declines climb as the circuit nears
    /// local minimality -- and spends re-encoding diversity, since COMP draws
    /// only from minimum-size spellings and so pulls the circuit toward exactly
    /// the form `fcompress` would compute anyway. Guarding that directly means
    /// a too-wide band costs nothing: the brake lets go on its own.
    pub(super) fn apply_size_brake(&mut self) {
        if self.params.size_hi == 0 {
            return;
        }
        let size = self.arena.len();
        if !self.brake_on {
            if size >= self.params.size_hi {
                self.brake_on = true;
                self.db_mode_cur = DbMode::Compressing;
                self.brake_mark_move = self.moves_done;
                self.brake_mark_size = size;
                self.counters.brake_engagements += 1;
            }
            return;
        }
        self.counters.brake_rounds += 1;
        if size <= self.params.size_lo {
            self.brake_on = false;
            self.db_mode_cur = self.params.db_mode;
            return;
        }
        // Productivity release: COMP stops growth immediately but shrinks only
        // slowly (about 13% of its hits strictly shrink), so the shed rate is
        // the honest signal that the leg is still worth running.
        let window = self.params.comp_release_window.max(1);
        if self.moves_done.saturating_sub(self.brake_mark_move) >= window {
            let shed = self.brake_mark_size.saturating_sub(size) as f64;
            let rate = shed / window as f64;
            if rate < self.params.comp_release_eps {
                self.brake_on = false;
                self.db_mode_cur = self.params.db_mode;
                return;
            }
            self.brake_mark_move = self.moves_done;
            self.brake_mark_size = size;
        }
    }

    pub fn run(&mut self) -> MixStop {
        if self.params.prof_n[2] > 0.0 && self.prof.is_none() {
            if self.moves_done > 0 {
                // v1: the profile's effective-work clock is not serialised, so
                // a resumed profile would restart phase 1 from eff=0. Profiles
                // are whole-run single invocations; warn rather than mis-steer.
                eprintln!(
                    "[fmix] WARNING: --profile on a RESUME (moves_done={}) restarts the profile clock at eff=0 — run a profile as a single invocation",
                    self.moves_done
                );
            }
            self.prof_init();
        }
        while self.moves_done < self.params.moves {
            if self.arena.len() == 0 {
                self.global_check();
                return MixStop::CircuitEmpty;
            }
            // Generation targeting: refresh the laggard list on its cadence
            // (an O(size) scan; entries invalidated between scans are pruned
            // lazily at draw time in pick_seed).
            if self.params.gen_target > 0 && self.moves_done >= self.pool_scan_due {
                self.rebuild_pool();
                self.pool_scan_due = self.moves_done + self.params.gen_rescan.max(1);
            }
            // The split stage (docs/FMIX_SPLIT_TWIST.md): while live it owns
            // the whole round — every other slot is withheld, and the stage
            // boundary optionally ends the run (--split-stop).
            let took_split = self.params.split && self.split_on && {
                self.split_twist_move();
                true
            };
            if took_split && self.split_ended {
                self.split_ended = false;
                if self.params.split_stop {
                    self.moves_done += 1;
                    self.counters.moves = self.moves_done;
                    self.global_check();
                    self.report();
                    return MixStop::SplitDone;
                }
            }
            // Slot 0 continued. With a PROFILE active the controller is the
            // one size authority: it owns target_size, keeps the brake
            // inert, and drives the MIX/COMP coin. Otherwise: the size
            // brake, then the p_mix overlay (whose per-round coin, when
            // armed, is the binding choice; the thermostat is untouched).
            if took_split {
                // stage round: no slot-0 steering
            } else if self.prof.is_some() {
                self.prof_tick();
                if self.prof.as_ref().is_some_and(|p| p.phase == 4) {
                    self.report();
                    return MixStop::ProfileDone;
                }
            } else {
                self.apply_size_brake();
                self.apply_mode_overlay();
            }
            // Slot 1: one twist, at a FIXED rate the rest of the machinery
            // balances around.
            let took_twist = !took_split
                && self.params.p_twist > 0.0
                && self.rng.random_bool(self.params.p_twist.clamp(0.0, 1.0))
                && {
                    self.twist_round();
                    true
                };
            // Slot 1b: GLOBAL re-randomisation, sitting after the twist and
            // before the DB move. Rate is scaled by the circuit size, so the
            // DEFAULT probability is exactly 1/|circuit| -- an expected one
            // whole-circuit reshuffle per |circuit| rounds, and O(mean slack)
            // expected work per round however large the circuit gets.
            let took_shuffle = !took_split
                && !took_twist
                && self.params.shuffle_rate > 0.0
                && self.arena.len() >= 2
                && {
                    let p = (self.params.shuffle_rate / self.arena.len() as f64).clamp(0.0, 1.0);
                    self.rng.random_bool(p)
                }
                && {
                    self.global_shuffle();
                    true
                };
            // Slot 1c: one bridge fusion (docs/NONLOCAL_PHASE_A.md), at a
            // fixed rate like the twist. p_bridge == 0 draws no RNG.
            let took_bridge = !took_split
                && !took_twist
                && !took_shuffle
                && self.params.p_bridge > 0.0
                && self.arena.len() >= 4
                && self.rng.random_bool(self.params.p_bridge.clamp(0.0, 1.0))
                && {
                    self.bridge_round();
                    true
                };
            // Slot 2: ONE DB move, under the live db_mode. A round whose
            // descent finds nothing is SPENT -- there is no fallthrough -- so
            // the thermostat receives exactly (1 - p_twist)(1 - p_db) of rounds
            // no matter how hard the material is.
            let took_db = !took_split
                && !took_twist
                && !took_shuffle
                && !took_bridge
                && self.params.p_db > 0.0
                && self.arena.len() >= 1
                && self.rng.random_bool(self.params.p_db.clamp(0.0, 1.0))
                && {
                    let mode = self.db_mode_cur;
                    let before = self.arena.len();
                    let hit = self.db_attempt(mode);
                    self.counters.db_slot2_rounds += 1;
                    if hit {
                        self.counters.db_slot2_hits += 1;
                        self.counters.db_slot2_added +=
                            self.arena.len().saturating_sub(before) as u64;
                    }
                    true
                };
            // Slot 3: the thermostat.
            if !took_split && !took_twist && !took_shuffle && !took_bridge && !took_db {
                let excess = self.arena.len() as f64 - self.params.target_size as f64;
                // In steer mode the 0.98 ceiling is the binding constraint on
                // holding size: its 2% expansion floor is a structural growth
                // source (measured +0.007/move) that saturated contraction
                // cannot absorb. Above target, steered runs contract harder.
                let hi = self.params.contract_ceiling;
                let p_contract = (1.0 / (1.0 + (-excess / self.params.temp).exp())).clamp(0.02, hi);
                // Nothing to contract below two gates; every contraction channel
                // samples a linked node, so guard the empty/singleton arena (which
                // a DB move can reach on a near-identity region).
                if self.arena.len() >= 2 && self.rng.random_bool(p_contract) {
                    // Contraction channels with complementary stock; when one finds
                    // nothing, fall through to the next rather than wasting the move.
                    // The compressing DB replacement is tried first with probability
                    // w_db (the only channel that can contract non-ladder material),
                    // then undo/merge as before.
                    let did_db = self.params.p_comp > 0.0
                        && self.rng.random_bool(self.params.p_comp.clamp(0.0, 1.0))
                        && self.db_attempt_comp();
                    if did_db {
                        // done
                    } else if self.rng.random_bool(self.params.undo_frac) {
                        if !self.undo_move() {
                            self.merge_move();
                        }
                    } else if !self.merge_move() {
                        self.undo_move();
                    }
                } else {
                    self.expand_move();
                }
            }
            self.moves_done += 1;
            self.counters.moves = self.moves_done;
            // Piecewise rounds use the same 1/size-per-move clock as
            // prof_tick. Account for this move even when a stop is requested.
            if self.params.eff_budget > 0.0 {
                self.eff_done += 1.0 / self.arena.len().max(1) as f64;
            }
            // Checked every move (a plain bool - no RNG, no trajectory
            // effect) so FMIX_STOP_AT_PHASE stops at the transition move
            // rather than up to report_every moves later. The stop-FLAG
            // file poll stays at report cadence below. An explicit stop takes
            // precedence when a piece also exhausts its round budget here.
            if self.stop_requested {
                self.global_check();
                return MixStop::StopFlag;
            }
            // A piece stops once it has spent its share of size-normalised
            // work, so a round costs the serial run's moves per eff.
            if self.params.eff_budget > 0.0 && self.eff_done >= self.params.eff_budget {
                self.global_check();
                return MixStop::RoundDone;
            }
            if self.moves_done % self.params.verify_every == 0 {
                self.global_check();
            }
            if self.moves_done % self.params.report_every == 0 {
                self.report();
                self.check_flags();
                self.check_gen_snap();
                self.check_move_snap();
                if self.stop_requested {
                    self.global_check();
                    return MixStop::StopFlag;
                }
                if self.canary_fired() {
                    println!(
                        "[fmix] canary fired at move {}: {:.1}% of the last {} pool-seeded rounds failed at every rung (theta {}), fall-through {} — the pool is material the store cannot spell; stopping",
                        self.moves_done,
                        100.0 * self.canary_frac(),
                        self.canary.len(),
                        self.params.canary_theta,
                        self.counters.canary_fallthrough,
                    );
                    self.global_check();
                    return MixStop::CanaryFired;
                }
                if self.dose_reached() {
                    println!(
                        "[fmix] dose reached at move {}: all-gates laggard frac <= {} (circuit generation >= {}), twist coverage {:.1} — stopping",
                        self.moves_done,
                        self.params.gen_stop_frac,
                        self.params.gen_target,
                        self.twist_coverage(),
                    );
                    self.global_check();
                    return MixStop::DoseReached;
                }
            }
        }
        self.global_check();
        MixStop::MovesBudget
    }

    pub(super) fn check_flags(&mut self) {
        if let Some(f) = self.stop_flag.clone() {
            if std::path::Path::new(&f).exists() {
                let _ = std::fs::remove_file(&f);
                println!(
                    "[fmix] stop flag seen at move {}: finishing cleanly",
                    self.moves_done
                );
                self.stop_requested = true;
            }
        }
        if let Some(f) = self.dump_flag.clone() {
            if std::path::Path::new(&f).exists() {
                self.global_check();
                let gates = self.arena.to_vec();
                // Move-stamped filename so repeated touches build a trajectory
                // instead of overwriting; origins sidecar so the positional
                // metrics (diffusion/autocorr) are computable per snapshot.
                let out = format!("{}.mv{}", self.dump_out, self.moves_done);
                let tmp = format!("{out}.tmp");
                match crate::circuit::formats::write_mpmct(&tmp, &gates, self.num_wires) {
                    Ok(()) => {
                        if let Err(e) = std::fs::rename(&tmp, &out) {
                            eprintln!("[fmix] dump rename failed: {e}");
                        } else {
                            let mut s = String::with_capacity(gates.len() * 8);
                            for o in self.origins_in_order() {
                                s.push_str(&format!("{o}\n"));
                            }
                            if let Err(e) = std::fs::write(format!("{out}.origins"), s) {
                                eprintln!("[fmix] dump origins write failed: {e}");
                            }
                            let mut s = String::with_capacity(gates.len() * 4);
                            for g in self.gens_in_order() {
                                s.push_str(&format!("{g}\n"));
                            }
                            if let Err(e) = std::fs::write(format!("{out}.gens"), s) {
                                eprintln!("[fmix] dump gens write failed: {e}");
                            }
                            println!(
                                "[fmix] DUMP: wrote {} gates to {} at move {} (verified, continuing)",
                                gates.len(),
                                out,
                                self.moves_done
                            );
                        }
                    }
                    Err(e) => eprintln!("[fmix] dump write failed: {e}"),
                }
                let _ = std::fs::remove_file(&f);
            }
        }
    }

    // Generation-multiple snapshots (--gen-snap-every): when the circuit
    // generation crosses a fresh multiple of the interval, write ONE verified
    // state and name it for every multiple crossed this interval (a report
    // gap can jump several multiples; the same state honestly serves each).
    pub(super) fn check_gen_snap(&mut self) {
        let every = self.params.gen_snap_every;
        if every == 0 {
            return;
        }
        let Some(base) = self.gen_snap_base.clone() else {
            return;
        };
        let g = self.gen_stats().g_circ;
        let reached = (g / every) * every;
        if reached <= self.last_gen_snap {
            return;
        }
        self.global_check();
        let gates = self.arena.to_vec();
        let mut gens = String::with_capacity(gates.len() * 4);
        for gg in self.gens_in_order() {
            gens.push_str(&format!("{gg}\n"));
        }
        let mut prev: Option<String> = None;
        let mut m = self.last_gen_snap + every;
        while m <= reached {
            let out = format!("{base}.gen{m}.mpmct1");
            let ok = match &prev {
                None => {
                    let tmp = format!("{out}.tmp");
                    match crate::circuit::formats::write_mpmct(&tmp, &gates, self.num_wires) {
                        Ok(()) => match std::fs::rename(&tmp, &out) {
                            Ok(()) => true,
                            Err(e) => {
                                eprintln!("[fmix] gen-snap rename failed: {e}");
                                false
                            }
                        },
                        Err(e) => {
                            eprintln!("[fmix] gen-snap write failed: {e}");
                            false
                        }
                    }
                }
                Some(p) => match std::fs::copy(p, &out) {
                    Ok(_) => true,
                    Err(e) => {
                        eprintln!("[fmix] gen-snap copy failed: {e}");
                        false
                    }
                },
            };
            if ok {
                if let Err(e) = std::fs::write(format!("{out}.gens"), &gens) {
                    eprintln!("[fmix] gen-snap gens write failed: {e}");
                }
                println!(
                    "[fmix] GEN-SNAP: circuit generation {} >= {}: wrote {} gates to {} at move {} (verified, continuing)",
                    g,
                    m,
                    gates.len(),
                    out,
                    self.moves_done
                );
                prev = Some(out);
            }
            m += every;
        }
        self.last_gen_snap = reached;
    }

    // Move-multiple snapshots (--snap-every-moves): verified state at fixed
    // move-count multiples — the progress clock when the generation census
    // is not meaningful (pure-split runs).
    pub(super) fn check_move_snap(&mut self) {
        let every = self.params.snap_every_moves;
        if every == 0 || self.moves_done % every != 0 {
            return;
        }
        let Some(base) = self.gen_snap_base.clone() else {
            return;
        };
        self.global_check();
        let gates = self.arena.to_vec();
        let out = format!("{base}.mv{}.mpmct1", self.moves_done);
        let tmp = format!("{out}.tmp");
        match crate::circuit::formats::write_mpmct(&tmp, &gates, self.num_wires) {
            Ok(()) => match std::fs::rename(&tmp, &out) {
                Ok(()) => {
                    let mut gens = String::with_capacity(gates.len() * 4);
                    for gg in self.gens_in_order() {
                        gens.push_str(&format!("{gg}\n"));
                    }
                    if let Err(e) = std::fs::write(format!("{out}.gens"), &gens) {
                        eprintln!("[fmix] move-snap gens write failed: {e}");
                    }
                    // A resumable state alongside the circuit. A circuit
                    // snapshot on its own cannot be continued -- directions,
                    // generations, litters, the journal and the original all
                    // live outside it -- so a long run could only be restarted,
                    // not resumed, from any point but its end. Written to a
                    // temp path and renamed, so an interrupted write never
                    // leaves a half-file that looks resumable.
                    let sp = format!("{out}.state");
                    let sptmp = format!("{sp}.tmp");
                    match self.save_state(&sptmp) {
                        Ok(()) => {
                            if let Err(e) = std::fs::rename(&sptmp, &sp) {
                                eprintln!("[fmix] move-snap state rename failed: {e}");
                            }
                        }
                        Err(e) => eprintln!("[fmix] move-snap state write failed: {e}"),
                    }
                    println!(
                        "[fmix] MOVE-SNAP: wrote {} gates to {} at move {} (verified, continuing)",
                        gates.len(),
                        out,
                        self.moves_done
                    );
                }
                Err(e) => eprintln!("[fmix] move-snap rename failed: {e}"),
            },
            Err(e) => eprintln!("[fmix] move-snap write failed: {e}"),
        }
    }

    // Dose-based stop (see MixParams::gen_stop_frac): the benchmark
    // criterion — the fraction of ALL gates still below gen_target at or
    // below gen_stop_frac (0.05 = "the circuit has generation >= target"),
    // AND twist coverage at or above twist_cov_stop. Wide and written-off
    // gates count against the fraction like everyone else; the walk lifts
    // them too, since split children get parent + 1.
    pub(super) fn dose_reached(&self) -> bool {
        if self.params.gen_target == 0 || self.params.gen_stop_frac < 0.0 {
            return false;
        }
        let s = self.gen_stats();
        // Laggard fraction among the gates targeting can actually move. This
        // must NOT be all_lag/total: gates the DB channel can never re-encode
        // stay below target forever, so on material where they exceed the
        // stop fraction (a product-share gadget is ~62% wide) the all-gates
        // ratio never falls and the dose stop never fires — the run burns its
        // whole move budget after the dose is long since complete. With
        // everything targetable the two agree, so narrow runs are unaffected.
        if s.targetable == 0 {
            // Nothing is re-encodable: the dose is unmeasurable, not met.
            // Run the move budget rather than exiting immediately.
            return false;
        }
        let lag_frac = s.lag as f64 / s.targetable as f64;
        if lag_frac > self.params.gen_stop_frac {
            return false;
        }
        self.params.twist_cov_stop <= 0.0 || self.twist_coverage() >= self.params.twist_cov_stop
    }
}
