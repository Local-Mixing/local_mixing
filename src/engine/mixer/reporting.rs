//! Functional verification, generation census and trajectory reporting.
use super::*;

impl Mixer {
    // ---- verification, metrics, reporting ----

    pub fn global_check(&mut self) {
        assert_eq!(
            self.indexed_count,
            self.arena.len(),
            "merge index drifted from arena (move {})",
            self.moves_done
        );
        // 4 batches x 64 lanes = 256 samples, carried together as [u64; 4] per
        // wire so the circuit is walked ONCE instead of four times. The arena
        // leg is a dependent-load chase over the node list, so it was four
        // cache misses per gate to compute four independent results; the fused
        // form pays the traversal once and adds only register-resident ANDs.
        //
        // The draws stay batch-major (all wires of batch 0, then batch 1, ...),
        // which is exactly the order and count the four-pass loop consumed, so
        // the RNG stream is untouched. Evaluation itself never draws.
        const BATCHES: usize = 4;
        let mut st_orig = vec![[0u64; BATCHES]; self.num_wires];
        for b in 0..BATCHES {
            for w in 0..self.num_wires {
                st_orig[w][b] = self.rng.random();
            }
        }
        let mut st_cur = st_orig.clone();
        crate::circuit::xgate::eval_lanes4(&self.original, &mut st_orig);
        let mut cur = self.arena.head();
        while cur != NIL {
            self.arena.gate(cur).apply_lanes4(&mut st_cur);
            cur = self.arena.neighbor(cur, Dir::R);
        }
        if st_orig != st_cur {
            // Same trip condition as the old per-batch assert_eq!; name the
            // first differing (wire, batch) instead of dumping both states.
            let (wire, batch) = (0..self.num_wires)
                .flat_map(|w| (0..BATCHES).map(move |b| (w, b)))
                .find(|&(w, b)| st_orig[w][b] != st_cur[w][b])
                .expect("states differ but no differing wire found");
            panic!(
                "FUNCTIONALITY BROKEN: circuit no longer equals the input (move {}, wire {wire}, batch {batch})",
                self.moves_done
            );
        }
    }

    /// One arena pass for all three counts. Folded rather than split into
    /// separate walks: `report` already makes ten full pointer-chases over the
    /// circuit, and the gate is dereferenced here anyway.
    pub fn g57_census(&self) -> G57Census {
        let mut out = G57Census::default();
        let mut cur = self.arena.head();
        while cur != NIL {
            let g = self.arena.gate(cur);
            if g.comp && g.ctrls.len() == 2 {
                out.shaped += 1;
                if g.ctrls[0].1 == g.ctrls[1].1 {
                    out.same_pol += 1;
                } else {
                    out.opp_pol += 1;
                }
            }
            cur = self.arena.neighbor(cur, Dir::R);
        }
        out
    }

    /// True g57 shape: `comp = 1` with exactly two controls of OPPOSITE
    /// polarity (`a ^= b OR !c`). Distinct from `remaining_g57`, which counts
    /// every `comp = 1` gate regardless of width -- the report's `comp=` field.
    pub fn true_g57(&self) -> usize {
        self.g57_census().opp_pol
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

    // Mean |current position fraction - original position fraction| over gates
    // with a real origin: 0 at the start, drifts toward ~1/3 (the mean for
    // independent uniforms) as positional memory of the original decays.
    pub fn origin_displacement(&self) -> f64 {
        let n = self.arena.len() as f64;
        let m = self.original.len() as f64;
        let (mut acc, mut cnt) = (0.0f64, 0u64);
        let mut cur = self.arena.head();
        let mut i = 0usize;
        while cur != NIL {
            let o = self.meta_of(cur).origin;
            if o != ORIGIN_SYNTH {
                acc += ((i as f64 / n) - (o as f64 / m)).abs();
                cnt += 1;
            }
            i += 1;
            cur = self.arena.neighbor(cur, Dir::R);
        }
        if cnt == 0 { 0.0 } else { acc / cnt as f64 }
    }

    // Mean number of distinct origins in sampled 32-gate windows (max 32):
    // low = original material still clumped, high = well interleaved.
    pub fn window_origin_diversity(&mut self, samples: usize) -> f64 {
        let ids = self.arena.ids_in_order();
        if ids.len() < 32 {
            return 0.0;
        }
        let mut acc = 0.0f64;
        for _ in 0..samples {
            let s = self.metrics_rng.random_range(0..=(ids.len() - 32));
            let mut set: Vec<u32> = ids[s..s + 32]
                .iter()
                .map(|&id| self.meta_of(id).origin)
                .collect();
            set.sort_unstable();
            set.dedup();
            acc += set.len() as f64;
        }
        acc / samples as f64
    }

    // Fraction of gates whose output is never read before its wire is next
    // overwritten (see stats::fanouts).
    pub fn fanout_zero_frac(&self) -> f64 {
        let ids = self.arena.ids_in_order();
        let fan = crate::engine::stats::fanouts(
            ids.iter().map(|&id| self.arena.gate(id)),
            self.num_wires,
        );
        if fan.is_empty() {
            return 0.0;
        }
        fan.iter().filter(|&&f| f == 0).count() as f64 / fan.len() as f64
    }

    // Mean two-sided float-box size over sampled gates, capped per direction:
    // the mobility / roadblock gauge.
    pub fn mean_leeway(&mut self, samples: usize, cap: usize) -> f64 {
        if self.arena.len() == 0 || samples == 0 {
            return 0.0;
        }
        let mut acc = 0usize;
        for _ in 0..samples {
            let id = self.arena.random_linked(&mut self.metrics_rng);
            acc += self.float_distance(id, Dir::L, cap) + self.float_distance(id, Dir::R, cap);
        }
        acc as f64 / samples as f64
    }

    /// Non-zero cells of the splice size histogram, as `out->in:count`.
    pub fn splice_size_line_curated(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        for (o, row) in self.counters.splice_sizes_curated.iter().enumerate() {
            for (i, &c) in row.iter().enumerate() {
                if c > 0 {
                    parts.push(format!("{o}->{i}:{c}"));
                }
            }
        }
        parts.join(" ")
    }

    pub fn splice_size_line(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        for (o, row) in self.counters.splice_sizes.iter().enumerate() {
            for (i, &c) in row.iter().enumerate() {
                if c > 0 {
                    parts.push(format!("{o}->{i}:{c}"));
                }
            }
        }
        parts.join(" ")
    }

    pub fn report(&mut self) {
        if self.quiet {
            return;
        }
        self.anc_prune();
        // Flush the attempt recorder so a long run is inspectable mid-flight.
        if let Some(w) = self.db_record.as_mut() {
            use std::io::Write;
            let _ = w.flush();
        }
        let disp = self.origin_displacement();
        let owin = self.window_origin_diversity(64);
        let fan0 = self.fanout_zero_frac();
        let leew = self.mean_leeway(256, 4096);
        let origins = self.origins_in_order();
        let odiff = crate::engine::stats::origin_diffusion(&origins);
        let oadj = crate::engine::stats::adjacent_origin_autocorr(&origins);
        // Fraction of gates whose ancestry label has been destroyed. A DB splice
        // over a window spanning mixed lineage stamps its products ORIGIN_SYNTH,
        // and origin_diffusion / adjacent_origin_autocorr / origin_displacement
        // all SKIP those gates. So odiff, oadj and disp are computed over the
        // material mixing has failed to touch, and they get more selective the
        // better the mixing works. Without this field that bias is invisible:
        // read osyn first, and treat the other three as unusable once it is high.
        let (anc_card, anc_span) = self.anc_stats();
        let osyn = if origins.is_empty() {
            0.0
        } else {
            origins.iter().filter(|&&o| o == ORIGIN_SYNTH).count() as f64 / origins.len() as f64
        };
        let gs = self.gen_stats();
        let gmin = if gs.min == GEN_FRESH {
            "F".to_string()
        } else {
            gs.min.to_string()
        };
        let cov = self.twist_coverage();
        let g57c = self.g57_census();
        let c = &self.counters;
        let hist: Vec<String> = (0..=self.params.k_max.min(15))
            .map(|w| format!("{}:{}", w, c.width_hist[w]))
            .collect();
        println!(
            "[fmix] mv={} size={} target={} comp={} g57={} shaped={} polf={:.3} | merges c={} x={} d={} s={} a={} sib={} xorig={} tabu={} nopart={} wall={} far={} noadj={} | undo ok={} dead={} tabu={} miss={} live={} | db pdb={:.3} slot2={}/{} sadd={} comp={}/{} agn={}/{} rm={} add={} wide={} wpoly={} dsk={} ssk={} bab={} idsk={} cur={}/{} g57only={}/{} sled={}/{} m123={} bled={} | expand r1={} r2={} r3={} pre={} fresh={} unsub={} ins={} tn1={} tsw={} tn2={} twrel={} twsplit={} twspan={} twskip={} shuf={} shufmv={} shufst={} shufms={} | declined={} blockw={} dl={} bnd={} | floats={}/{} scat={}/{} | disp={:.4} owin={:.1} fan0={:.3} leew={:.0} odiff={:.4} oadj={:.4} osyn={:.3} anc={:.1} ancspan={:.3} width[{}] | gen tgt={} G={} Gall={} tgtbl={} alag={}/{} lag={}/{} wlag={} min={} cov={:.1} canary={:.3} cft={} | litter distinct={:.2} full={} ban={} tplace={}/{} dmin={:.3} dminw={} canon[poly={}ms canon={}ms calls={}] verify={}ms degprobe={}ms/{} | choice n={} multi={:.3} mean={:.2} bits/splice={:.3}",
            c.moves,
            self.arena.len(),
            self.params.target_size,
            self.remaining_g57(),
            g57c.opp_pol,
            g57c.shaped,
            g57c.pol_flipped(),
            c.merges_cancel,
            c.merges_xfuse,
            c.merges_drop,
            c.merges_subsume,
            c.merges_absorb,
            c.merges_sibling,
            c.merges_cross_origin,
            c.tabu_blocked,
            c.merge_no_partner,
            c.merge_wall_blocked,
            c.merge_too_far,
            c.merge_not_adjacent,
            c.undos,
            c.undo_dead,
            c.undo_tabu,
            c.undo_gather_miss,
            self.journal.len(),
            self.params.p_db,
            c.db_slot2_hits,
            c.db_slot2_rounds,
            c.db_slot2_added,
            c.db_comp_hits,
            c.db_comp_misses,
            c.db_agn_hits,
            c.db_agn_misses,
            c.db_gates_removed,
            c.db_gates_added,
            c.db_wide_skip,
            c.db_wide_poly,
            c.db_degree_skips,
            c.db_span_skips,
            c.db_build_aborts,
            c.db_identity_skips,
            c.db_curated_hits,
            c.db_curated_rejected,
            c.db_g57_hits,
            c.db_g57_rounds,
            self.stable_led_added,
            self.stable_led_removed,
            self.bigpool_hits,
            self.band_led,
            c.cross_r1,
            c.cross_r2,
            c.cross_r3,
            c.presplits,
            c.fresh_splits,
            c.unsubs,
            c.inserts,
            c.twist_negs,
            c.twist_swaps,
            c.twist_cnots,
            c.twist_relabels,
            c.twist_case_splits,
            c.twist_span,
            c.twist_skips,
            c.shuffles,
            c.shuffle_moved,
            c.shuffle_steps,
            c.shuffle_ns / 1_000_000,
            c.declined,
            c.blocked_width,
            c.blocked_deadlock,
            c.boundary,
            c.floats,
            c.float_steps,
            c.scatters,
            c.scatter_steps,
            disp,
            owin,
            fan0,
            leew,
            odiff,
            oadj,
            osyn,
            anc_card,
            anc_span,
            hist.join(" "),
            self.params.gen_target,
            gs.g_circ,
            gs.g_all,
            gs.targetable,
            gs.all_lag,
            gs.total,
            gs.lag,
            gs.elig,
            gs.wlag,
            gmin,
            cov,
            self.canary_frac(),
            c.canary_fallthrough,
            if c.litter_windows > 0 {
                c.litter_distinct_sum as f64 / c.litter_windows as f64
            } else {
                0.0
            },
            c.litter_full_spliced,
            c.litter_banned,
            c.twist_placed,
            c.twist_place_fallback,
            if c.dmin_windows > 0 {
                c.dmin_shorter as f64 / c.dmin_windows as f64
            } else {
                0.0
            },
            // The DENOMINATOR alongside the ratio: windows for which the
            // store held any non-identical equivalent. Without it a moving
            // dmin cannot be told from a moving sample.
            c.dmin_windows,
            crate::canonicalization::xgate::POLY_NS.load(std::sync::atomic::Ordering::Relaxed)
                / 1_000_000,
            crate::canonicalization::xgate::CANON_NS.load(std::sync::atomic::Ordering::Relaxed)
                / 1_000_000,
            crate::canonicalization::xgate::CANON_CALLS.load(std::sync::atomic::Ordering::Relaxed),
            crate::canonicalization::xgate::VERIFY_NS.load(std::sync::atomic::Ordering::Relaxed)
                / 1_000_000,
            crate::canonicalization::xgate::DEGREE_NS.load(std::sync::atomic::Ordering::Relaxed)
                / 1_000_000,
            crate::canonicalization::xgate::DEGREE_CALLS.load(std::sync::atomic::Ordering::Relaxed),
            // Selection entropy: how often a successful splice had a real
            // choice, how wide that choice was, and the bits it injected.
            c.choice_splices,
            if c.choice_splices > 0 {
                c.choice_multi as f64 / c.choice_splices as f64
            } else {
                0.0
            },
            if c.choice_splices > 0 {
                c.choice_sum as f64 / c.choice_splices as f64
            } else {
                0.0
            },
            if c.choice_splices > 0 {
                c.choice_bits_milli as f64 / 1000.0 / c.choice_splices as f64
            } else {
                0.0
            }
        );
        // Per-success complexity histogram (store-minimal spelling length of
        // each converted permutation); printed whenever any conversions exist.
        let hs: Vec<String> = self
            .dmin_success_hist
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c > 0)
            .map(|(l, &c)| format!("{l}:{c}"))
            .collect();
        if !hs.is_empty() {
            println!("[fmix] dminh mv={} {}", self.moves_done, hs.join(" "));
        }
        let ms: Vec<String> = self
            .m123_class_hist
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c > 0)
            .map(|(l, &c)| format!("M{l}:{c}"))
            .collect();
        if !ms.is_empty() {
            println!("[fmix] m123class mv={} {}", self.moves_done, ms.join(" "));
        }
        if self.geo_attempts[0] + self.geo_attempts[1] > 0 {
            let rate = |h: u64, a: u64| {
                if a > 0 {
                    100.0 * h as f64 / a as f64
                } else {
                    0.0
                }
            };
            println!(
                "[fmix] geo mv={} ctg={}/{} ({:.2}%) cvx={}/{} ({:.2}%)",
                self.moves_done,
                self.geo_hits[0],
                self.geo_attempts[0],
                rate(self.geo_hits[0], self.geo_attempts[0]),
                self.geo_hits[1],
                self.geo_attempts[1],
                rate(self.geo_hits[1], self.geo_attempts[1]),
            );
        }
        // Pair-geometry meters (docs/NONLOCAL_PHASE_A.md), only when armed.
        if self.params.p_pair > 0.0 {
            let fused = c.pair_fused.max(1) as f64;
            println!(
                "[fmix] pair rounds={} fused={} splices={} permskip={} empty={} trunc={} box avg={:.1} max={} dist avg={:.1} max={}",
                c.pair_rounds,
                c.pair_fused,
                c.pair_splices,
                c.pair_perm_skips,
                c.pair_boxes_empty,
                c.pair_scan_truncs,
                c.pair_box_sum as f64 / fused,
                c.pair_box_max,
                c.pair_dist_sum as f64 / fused,
                c.pair_dist_max,
            );
        }
        // Bridge-fusion meters (docs/NONLOCAL_PHASE_A.md), only when armed.
        if self.params.p_bridge > 0.0 {
            let commits = (c.bridge_committed + c.bridge_half).max(1) as f64;
            println!(
                "[fmix] bridge rounds={} committed={} half={} rollback={} probemiss={} refused={} short={} span avg={:.1} max={} colliders avg={:.2} wake={}",
                c.bridge_rounds,
                c.bridge_committed,
                c.bridge_half,
                c.bridge_rollbacks,
                c.bridge_probe_miss,
                c.bridge_refused,
                c.bridge_short,
                c.bridge_span_sum as f64 / commits,
                c.bridge_span_max,
                c.bridge_colliders_sum as f64 / commits,
                c.bridge_wake_sum,
            );
        }
        let anc = self.anc_report();
        if !anc.is_empty() {
            println!("{anc}");
        }
        let ga = self.gen_anc_report();
        if !ga.is_empty() {
            println!("{ga}");
        }
        let sizes = self.splice_size_line();
        if !sizes.is_empty() {
            println!("[fmix] splice sizes out->in: {sizes}");
        }
        let csizes = self.splice_size_line_curated();
        if !csizes.is_empty() {
            println!("[fmix] splice sizes (curated) out->in: {csizes}");
        }
        if let Some(p) = &self.prof {
            let s_star = prof_target(self.params.prof_n, self.params.prof_r, p.s_in, p.eff);
            println!(
                "[fmix] profile: phase={} eff={:.2} size={} S*={:.0} pmix={:.3} ghat={:+.4} shat={:+.4} dhat={:+.4} integ={:+.3} sat={}",
                p.phase,
                p.eff,
                self.arena.len(),
                s_star,
                p.pmix,
                p.ghat,
                p.shat,
                p.dhat,
                p.integ,
                p.sat
            );
        }
        {
            // PER-OUTGOING-LENGTH breakdown: the rates the headline line
            // reports are averages over a uniform length draw in 1..s_db, so
            // they conflate "this width works" with "this width was sampled".
            // This says what each width actually did.
            let c = &self.counters;
            let n = c.len_attempts.len();
            if n > 0 && c.len_attempts.iter().sum::<u64>() > 0 {
                println!(
                    "[fmix] per-length: len attempts hits hit% removed added net span_skip deg_skip"
                );
                let g = |v: &Vec<u64>, k: usize| v.get(k).copied().unwrap_or(0);
                for k in 1..n {
                    let a = c.len_attempts[k];
                    if a == 0 {
                        continue;
                    }
                    let h = g(&c.len_hits, k);
                    let rm = g(&c.len_removed, k);
                    let ad = g(&c.len_added, k);
                    println!(
                        "[fmix] len {k:>3} {a:>9} {h:>8} {:>6.2} {rm:>8} {ad:>6} {:>+6} {:>10} {:>9}",
                        100.0 * h as f64 / a as f64,
                        rm as i64 - ad as i64,
                        g(&c.len_span_skip, k),
                        g(&c.len_deg_skip, k)
                    );
                }
            }
        }
        if self.params.twist_g57 {
            let c = &self.counters;
            let us = if c.tg_solves > 0 {
                c.tg_solve_ns as f64 / 1000.0 / c.tg_solves as f64
            } else {
                0.0
            };
            let hist: Vec<String> = c.tg_net_hist.iter().map(|v| v.to_string()).collect();
            println!(
                "[fmix] twist-g57: consumed={} emitted={} net/seam[{}] solves={} avg_us={:.1} slides={} retries={}",
                c.tg_consumed,
                c.tg_emitted,
                hist.join(","),
                c.tg_solves,
                us,
                c.tg_slides,
                c.tg_retries
            );
        }
    }

    pub fn gen_stats(&self) -> GenStats {
        let target = self.params.gen_target;
        let mut s = GenStats {
            lag: 0,
            elig: 0,
            wlag: 0,
            min: GEN_FRESH,
            all_lag: 0,
            total: 0,
            g_circ: 0,
            g_all: 0,
            targetable: 0,
        };
        // Two bucketed gen histograms for the percentiles (everything at or
        // past the cap lands in the top bucket, incl. GEN_FRESH — fine, the
        // 5th percentile of interest sits far below it): `hist` over all
        // gates (g_all, kept for continuity) and `thist` over the targetable
        // ones (g_circ, the number that means something).
        const GB: usize = 1024;
        let mut hist = [0u64; GB];
        let mut thist = [0u64; GB];
        for id in self.arena.ids_in_order() {
            let m = self.meta_of(id);
            s.min = s.min.min(m.dgen);
            s.total += 1;
            hist[(m.dgen as usize).min(GB - 1)] += 1;
            if m.dgen < target {
                s.all_lag += 1;
            }
            let eligible = self.pool_eligible(id);
            if !eligible {
                if m.dgen < target {
                    s.wlag += 1;
                }
                continue;
            }
            s.elig += 1;
            if m.dgen >= target {
                // At or past target: targetable and done.
                s.targetable += 1;
                thist[(m.dgen as usize).min(GB - 1)] += 1;
                continue;
            }
            // Below target and eligible: this is the population targeting can
            // actually move, and the denominator the dose stop reads. Nothing
            // is written off any more -- with the descent reaching length 1,
            // which cannot decline for free, an eligible gate the store knows
            // at all advances; the residue that truly cannot is what the canary
            // is for, rather than a per-gate miss counter.
            s.lag += 1;
            s.targetable += 1;
            thist[(m.dgen as usize).min(GB - 1)] += 1;
        }
        // Largest G with >= 95% of the population at generation >= G: walk
        // the histogram until more than 5% lie strictly below G.
        let percentile = |h: &[u64; GB], population: u64| -> u32 {
            let allow = population / 20;
            let mut below = 0u64;
            let mut g_out = 0u32;
            for g in 0..GB {
                below += if g > 0 { h[g - 1] } else { 0 };
                if below <= allow {
                    g_out = g as u32;
                } else {
                    break;
                }
            }
            g_out
        };
        s.g_all = percentile(&hist, s.total);
        // With nothing targetable the generation census is meaningless; fall
        // back to the all-gates figure rather than reporting a bare 0.
        s.g_circ = if s.targetable == 0 {
            s.g_all
        } else {
            percentile(&thist, s.targetable)
        };
        s
    }

    /// Cumulative per-position twist coverage: total twisted window span over
    /// the current size — the db_mixing twist dose meter (target ~600x per the
    /// saturation law).
    pub fn twist_coverage(&self) -> f64 {
        self.counters.twist_span as f64 / self.arena.len().max(1) as f64
    }
}
