//! Piecewise-parallel rounds for the GSS fragmentation stages (db_mixing DB
//! mixing, the split stage), docs/FMIX_PIECEWISE.md.
//!
//! One round: cut the whole circuit into `p` contiguous pieces of roughly
//! equal length, run one `Mixer` per piece IN PARALLEL (all sharing the one
//! read-only store), and concatenate the mixed pieces back in order. The next
//! round cuts at the midpoints between the previous round's seams, so every
//! old boundary lands mid-piece — the "shift by half a slice" rule, made exact
//! under uneven per-piece growth. On a shifted round the two end fragments are
//! half a slice long, so the round has `p + 1` pieces.
//!
//! The whole-circuit mixer W keeps the truth: the stage input as `original`,
//! the counters, the move clock, the event/litter id counters, the profile
//! controller, the split tri-state and the wire canaries. A piece carries its
//! gates with their `Meta` verbatim, mints event and litter ids in a disjoint
//! band, verifies against its own slice, and is steered from outside: the
//! controller's lever as a fixed `p_mix`, a proportional share of the size
//! setpoint as `target_size`/`temp`, and a size-normalised work budget
//! (`eff_budget`). After the round W folds the pieces back (counters, moves,
//! ids, tabu, journal, canaries), verifies the whole circuit against the true
//! input, and steps the UNCHANGED profile controller once — so with the round
//! length equal to the controller cadence the schedule, the phase machine and
//! the lever updates are the serial ones.
//!
//! `min_block_size` optionally derives `p = max(1, len / min_block_size)` at
//! each round. It is a divisor, not a floor on the shifted end pieces. When
//! p changes, rebase the cuts for the new count while keeping the shift phase.
//! With no size option, `pieces = 1` keeps the original serial path.

use super::*;
use rayon::prelude::*;

/// Driver configuration (fmix `--pieces` and its `--piece-*` companions).
#[derive(Clone, Debug)]
pub struct PieceCfg {
    /// Number of pieces on an unshifted round (shifted rounds run one more).
    pub pieces: usize,
    /// Derive the piece count from the current gate count each round instead
    /// of `pieces`. Shifted end pieces may be smaller than this divisor.
    pub min_block_size: Option<usize>,
    /// Cut-point jitter as a fraction of the local seam interval, in
    /// [0, 0.25). Every old seam stays at least (0.5 - jitter) of its
    /// interval away from every new cut.
    pub jitter: f64,
    /// Round length in eff units (moves per gate). 0 = the profile
    /// controller's cadence under a profile, 0.5 otherwise.
    pub round_eff: f64,
    /// Threads in the piece pool; 0 = pieces + 1 for fixed counts, or the
    /// available CPU parallelism for automatic counts.
    pub threads: usize,
    /// Stage 4: rounds after which the stage is declared ended even if g57s
    /// remain (they are reported, never silently dropped).
    pub split_rounds_max: usize,
    /// Let pieces print their own report lines (they are silent by default).
    pub verbose: bool,
    /// Run the pieces one after another on the calling thread. Same result
    /// by construction; used to prove thread-count independence.
    pub sequential: bool,
}

impl Default for PieceCfg {
    fn default() -> PieceCfg {
        PieceCfg {
            pieces: 1,
            min_block_size: None,
            jitter: 0.125,
            round_eff: 0.0,
            threads: 0,
            split_rounds_max: 6,
            verbose: false,
            sequential: false,
        }
    }
}

impl PieceCfg {
    fn pieces_for_len(&self, len: usize) -> usize {
        match self.min_block_size {
            Some(size) => (len / size).max(1),
            None => self.pieces,
        }
    }
}

/// Actual previous output seams and the nominal count that produced them.
/// Remember the nominal count separately: a shifted round has P+1 pieces.
#[derive(Default)]
struct Partition {
    seams: Vec<usize>,
    pieces: usize,
}

impl Partition {
    fn cuts(&mut self, cfg: &PieceCfg, len: usize, round: usize, rng: &mut StdRng) -> Vec<usize> {
        let pieces = cfg.pieces_for_len(len);
        let changed = cfg.min_block_size.is_some() && pieces != self.pieces && round > 0;
        self.pieces = pieces;
        if changed {
            // Old seams encode a different P. Keeping them would silently
            // keep the old count despite recomputing len / block_size.
            if round % 2 == 0 {
                return cut_points(&[], len, pieces, 0, cfg.jitter, rng);
            }
            let base: Vec<usize> = (1..pieces).map(|k| k * len / pieces).collect();
            return cut_points(&base, len, pieces, 1, cfg.jitter, rng);
        }
        cut_points(&self.seams, len, pieces, round, cfg.jitter, rng)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Policy {
    /// Stage 3: the layer-2 profile on W, pieces steered per round.
    Profile,
    /// Stage 4: pieces split to exhaustion, rounds until no g57 is left.
    Split,
    /// Plain thermostat walk in fixed-length rounds (tests, future stage 5).
    Moves,
}

/// A wire canary in position form (anchor = ordinal in the piece).
pub(crate) struct TapRec {
    pub wire: u16,
    pub orig_permille: u16,
    pub flips: u64,
    pub pos: u32,
}

/// An undo-journal entry in position form (the state file's representation);
/// stamps are re-read from the arena the entry is installed into.
pub(crate) struct JournalRec {
    pub before: [XGate; 2],
    pub dir: Dir,
    pub pivot: u32,
    pub after: Vec<u32>,
    pub event: u64,
    pub origins: [u32; 2],
    pub gens: [u32; 2],
    pub litters: [u64; 2],
    pub litter_sizes: [u16; 2],
    pub misses: u8,
}

/// Everything a piece needs from W, and everything W folds back from a piece.
pub(crate) struct PieceParts {
    pub gates: Vec<XGate>,
    pub metas: Vec<Meta>,
    pub taps: Vec<TapRec>,
    pub tabu: Vec<(u64, u64)>,
    pub journal: Vec<JournalRec>,
    // Run bookkeeping.
    pub counters: MixCounters,
    pub moves_start: u64,
    pub moves_done: u64,
    pub eff_done: f64,
    pub len_start: usize,
    pub event_base: u64,
    pub litter_base: u64,
    pub next_event: u64,
    pub next_litter: u64,
    pub split_fail_streak: u32,
    pub split_end_reason: Option<&'static str>,
    pub stop: Option<MixStop>,
}

fn add_vec(dst: &mut Vec<u64>, src: &[u64]) {
    if dst.len() < src.len() {
        dst.resize(src.len(), 0);
    }
    for (d, s) in dst.iter_mut().zip(src) {
        *d += s;
    }
}

fn add_vv(dst: &mut Vec<Vec<u64>>, src: &[Vec<u64>]) {
    if dst.len() < src.len() {
        dst.resize(src.len(), Vec::new());
    }
    for (d, s) in dst.iter_mut().zip(src) {
        add_vec(d, s);
    }
}

fn add_arr<const N: usize>(dst: &mut [u64; N], src: &[u64; N]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d += s;
    }
}

/// Field-wise fold of a piece's counters into W's. Written as a full
/// destructuring so that adding a counter is a compile error here until it is
/// classified (sum, max, or the move clock, which W owns).
impl std::ops::AddAssign<&MixCounters> for MixCounters {
    fn add_assign(&mut self, rhs: &MixCounters) {
        let MixCounters {
            moves: _,
            db_ing_hits,
            db_ing_rounds,
            db_hard_hits,
            db_hard_rounds,
            db_hard_added,
            db_identity_skips,
            db_curated_hits,
            db_curated_rejected,
            merges_absorb,
            db_g57_rounds,
            db_g57_hits,
            db_slot2_rounds,
            db_slot2_hits,
            db_slot2_added,
            brake_engagements,
            brake_rounds,
            canary_fallthrough,
            litter_banned,
            dmin_windows,
            dmin_shorter,
            twist_placed,
            twist_place_fallback,
            splice_sizes,
            splice_sizes_curated,
            litter_windows,
            litter_distinct_sum,
            litter_full_spliced,
            gen_misses,
            merges_cancel,
            merges_xfuse,
            merges_drop,
            merges_subsume,
            merges_sibling,
            merges_cross_origin,
            tabu_blocked,
            merge_no_partner,
            merge_wall_blocked,
            merge_too_far,
            merge_not_adjacent,
            undos,
            undo_dead,
            undo_tabu,
            undo_gather_miss,
            db_comp_hits,
            db_comp_misses,
            db_agn_hits,
            db_agn_misses,
            db_gates_removed,
            db_gates_added,
            shuffles,
            shuffle_moved,
            shuffle_steps,
            shuffle_ns,
            choice_splices,
            choice_multi,
            choice_sum,
            choice_bits_milli,
            db_mix_added,
            db_mix_removed,
            db_cmp_added,
            db_cmp_removed,
            db_wide_skip,
            db_wide_poly,
            len_attempts,
            len_hits,
            len_removed,
            len_added,
            len_span_skip,
            len_deg_skip,
            db_attempts,
            db_degree_skips,
            db_span_skips,
            db_build_aborts,
            cross_r1,
            cross_r2,
            cross_r3,
            presplits,
            fresh_splits,
            unsubs,
            inserts,
            twist_negs,
            twist_swaps,
            twist_cnots,
            twist_relabels,
            twist_case_splits,
            twist_span,
            twist_skips,
            tg_consumed,
            tg_emitted,
            tg_solves,
            tg_solve_ns,
            tg_slides,
            tg_retries,
            tg_net_hist,
            blocked_width,
            blocked_deadlock,
            declined,
            boundary,
            floats,
            float_steps,
            scatters,
            scatter_steps,
            dropped_neverfire,
            split_prims,
            split_hsplits,
            split_segs,
            split_joins,
            split_fails,
            split_xmid,
            tap_flips,
            split_span_sum,
            split_span_hist,
            cross_pool_shots,
            pair_rounds,
            pair_boxes_empty,
            pair_scan_truncs,
            pair_fused,
            pair_splices,
            pair_box_sum,
            pair_box_max,
            pair_dist_sum,
            pair_dist_max,
            pair_perm_skips,
            bridge_rounds,
            bridge_short,
            bridge_refused,
            bridge_probe_miss,
            bridge_rollbacks,
            bridge_half,
            bridge_committed,
            bridge_span_sum,
            bridge_span_max,
            bridge_colliders_sum,
            bridge_wake_sum,
            width_hist,
        } = rhs;
        self.db_ing_hits += db_ing_hits;
        self.db_ing_rounds += db_ing_rounds;
        self.db_hard_hits += db_hard_hits;
        self.db_hard_rounds += db_hard_rounds;
        self.db_hard_added += db_hard_added;
        self.db_identity_skips += db_identity_skips;
        self.db_curated_hits += db_curated_hits;
        self.db_curated_rejected += db_curated_rejected;
        self.merges_absorb += merges_absorb;
        self.db_g57_rounds += db_g57_rounds;
        self.db_g57_hits += db_g57_hits;
        self.db_slot2_rounds += db_slot2_rounds;
        self.db_slot2_hits += db_slot2_hits;
        self.db_slot2_added += db_slot2_added;
        self.brake_engagements += brake_engagements;
        self.brake_rounds += brake_rounds;
        self.canary_fallthrough += canary_fallthrough;
        self.litter_banned += litter_banned;
        self.dmin_windows += dmin_windows;
        self.dmin_shorter += dmin_shorter;
        self.twist_placed += twist_placed;
        self.twist_place_fallback += twist_place_fallback;
        add_vv(&mut self.splice_sizes, splice_sizes);
        add_vv(&mut self.splice_sizes_curated, splice_sizes_curated);
        self.litter_windows += litter_windows;
        self.litter_distinct_sum += litter_distinct_sum;
        self.litter_full_spliced += litter_full_spliced;
        self.gen_misses += gen_misses;
        self.merges_cancel += merges_cancel;
        self.merges_xfuse += merges_xfuse;
        self.merges_drop += merges_drop;
        self.merges_subsume += merges_subsume;
        self.merges_sibling += merges_sibling;
        self.merges_cross_origin += merges_cross_origin;
        self.tabu_blocked += tabu_blocked;
        self.merge_no_partner += merge_no_partner;
        self.merge_wall_blocked += merge_wall_blocked;
        self.merge_too_far += merge_too_far;
        self.merge_not_adjacent += merge_not_adjacent;
        self.undos += undos;
        self.undo_dead += undo_dead;
        self.undo_tabu += undo_tabu;
        self.undo_gather_miss += undo_gather_miss;
        self.db_comp_hits += db_comp_hits;
        self.db_comp_misses += db_comp_misses;
        self.db_agn_hits += db_agn_hits;
        self.db_agn_misses += db_agn_misses;
        self.db_gates_removed += db_gates_removed;
        self.db_gates_added += db_gates_added;
        self.shuffles += shuffles;
        self.shuffle_moved += shuffle_moved;
        self.shuffle_steps += shuffle_steps;
        self.shuffle_ns += shuffle_ns;
        self.choice_splices += choice_splices;
        self.choice_multi += choice_multi;
        self.choice_sum += choice_sum;
        self.choice_bits_milli += choice_bits_milli;
        self.db_mix_added += db_mix_added;
        self.db_mix_removed += db_mix_removed;
        self.db_cmp_added += db_cmp_added;
        self.db_cmp_removed += db_cmp_removed;
        self.db_wide_skip += db_wide_skip;
        self.db_wide_poly += db_wide_poly;
        add_vec(&mut self.len_attempts, len_attempts);
        add_vec(&mut self.len_hits, len_hits);
        add_vec(&mut self.len_removed, len_removed);
        add_vec(&mut self.len_added, len_added);
        add_vec(&mut self.len_span_skip, len_span_skip);
        add_vec(&mut self.len_deg_skip, len_deg_skip);
        self.db_attempts += db_attempts;
        self.db_degree_skips += db_degree_skips;
        self.db_span_skips += db_span_skips;
        self.db_build_aborts += db_build_aborts;
        self.cross_r1 += cross_r1;
        self.cross_r2 += cross_r2;
        self.cross_r3 += cross_r3;
        self.presplits += presplits;
        self.fresh_splits += fresh_splits;
        self.unsubs += unsubs;
        self.inserts += inserts;
        self.twist_negs += twist_negs;
        self.twist_swaps += twist_swaps;
        self.twist_cnots += twist_cnots;
        self.twist_relabels += twist_relabels;
        self.twist_case_splits += twist_case_splits;
        self.twist_span += twist_span;
        self.twist_skips += twist_skips;
        self.tg_consumed += tg_consumed;
        self.tg_emitted += tg_emitted;
        self.tg_solves += tg_solves;
        self.tg_solve_ns += tg_solve_ns;
        self.tg_slides += tg_slides;
        self.tg_retries += tg_retries;
        add_arr(&mut self.tg_net_hist, tg_net_hist);
        self.blocked_width += blocked_width;
        self.blocked_deadlock += blocked_deadlock;
        self.declined += declined;
        self.boundary += boundary;
        self.floats += floats;
        self.float_steps += float_steps;
        self.scatters += scatters;
        self.scatter_steps += scatter_steps;
        self.dropped_neverfire += dropped_neverfire;
        self.split_prims += split_prims;
        self.split_hsplits += split_hsplits;
        self.split_segs += split_segs;
        self.split_joins += split_joins;
        self.split_fails += split_fails;
        self.split_xmid += split_xmid;
        self.tap_flips += tap_flips;
        self.split_span_sum += split_span_sum;
        add_arr(&mut self.split_span_hist, split_span_hist);
        self.cross_pool_shots += cross_pool_shots;
        self.pair_rounds += pair_rounds;
        self.pair_boxes_empty += pair_boxes_empty;
        self.pair_scan_truncs += pair_scan_truncs;
        self.pair_fused += pair_fused;
        self.pair_splices += pair_splices;
        self.pair_box_sum += pair_box_sum;
        self.pair_box_max = self.pair_box_max.max(*pair_box_max);
        self.pair_dist_sum += pair_dist_sum;
        self.pair_dist_max = self.pair_dist_max.max(*pair_dist_max);
        self.pair_perm_skips += pair_perm_skips;
        self.bridge_rounds += bridge_rounds;
        self.bridge_short += bridge_short;
        self.bridge_refused += bridge_refused;
        self.bridge_probe_miss += bridge_probe_miss;
        self.bridge_rollbacks += bridge_rollbacks;
        self.bridge_half += bridge_half;
        self.bridge_committed += bridge_committed;
        self.bridge_span_sum += bridge_span_sum;
        self.bridge_span_max = self.bridge_span_max.max(*bridge_span_max);
        self.bridge_colliders_sum += bridge_colliders_sum;
        self.bridge_wake_sum += bridge_wake_sum;
        add_arr(&mut self.width_hist, width_hist);
    }
}

fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The seed of piece `piece` in round `round`: a deterministic function of
/// the run seed, distinct across (round, piece), independent of the thread
/// that happens to run the piece.
pub(crate) fn seed_of(seed: u64, round: usize, piece: usize) -> u64 {
    splitmix64(
        splitmix64(seed ^ 0xA5A5_5A5A_C3C3_3C3C)
            ^ (((round as u64) + 1) << 32)
            ^ ((piece as u64) + 1),
    )
}

/// Interior cut positions for round `round` over `len` gates.
///
/// `prev_seams` are the previous round's interior seams in the CURRENT
/// concatenation (prefix sums of the previous pieces' output lengths). Round 0
/// cuts at multiples of `len / p` (p pieces). An odd round cuts at the
/// midpoints of every interval of `[0] ++ prev_seams ++ [len]` — p cuts, p + 1
/// pieces, the two end pieces half a slice long. An even round cuts at the
/// midpoints between consecutive old seams — p - 1 cuts, p pieces. Each cut is
/// then jittered by up to ±`jitter` of its local interval, so every old seam
/// stays at least (0.5 - jitter) of its interval from every new cut.
pub(crate) fn cut_points(
    prev_seams: &[usize],
    len: usize,
    p: usize,
    round: usize,
    jitter: f64,
    rng: &mut StdRng,
) -> Vec<usize> {
    if p < 2 || len < 2 {
        return Vec::new();
    }
    let j = jitter.clamp(0.0, 0.25);
    // (nominal cut, local interval) triples.
    let mut nominal: Vec<(usize, usize, usize)> = Vec::new();
    if round == 0 || prev_seams.is_empty() {
        let l = len / p;
        for k in 1..p {
            let c = (k * len) / p;
            nominal.push((c, c.saturating_sub(l / 2), (c + l / 2).min(len)));
        }
    } else {
        let mut bounds: Vec<usize> = Vec::with_capacity(prev_seams.len() + 2);
        bounds.push(0);
        bounds.extend(prev_seams.iter().copied().filter(|&s| s > 0 && s < len));
        bounds.push(len);
        bounds.sort_unstable();
        bounds.dedup();
        if round % 2 == 1 {
            for w in bounds.windows(2) {
                if w[1] > w[0] + 1 {
                    nominal.push(((w[0] + w[1]) / 2, w[0], w[1]));
                }
            }
        } else if bounds.len() >= 4 {
            for w in bounds[1..bounds.len() - 1].windows(2) {
                if w[1] > w[0] + 1 {
                    nominal.push(((w[0] + w[1]) / 2, w[0], w[1]));
                }
            }
        }
    }
    let mut cuts: Vec<usize> = nominal
        .into_iter()
        .map(|(c, a, b)| {
            if j <= 0.0 || b <= a + 2 {
                return c;
            }
            let u: f64 = rng.random_range(-j..=j);
            let d = (u * (b - a) as f64).round() as i64;
            (c as i64 + d).clamp(a as i64 + 1, b as i64 - 1) as usize
        })
        .collect();
    cuts.retain(|&c| c > 0 && c < len);
    cuts.sort_unstable();
    cuts.dedup();
    cuts
}

fn stop_name(s: Option<MixStop>) -> &'static str {
    match s {
        None => "-",
        Some(MixStop::MovesBudget) => "moves",
        Some(MixStop::StopFlag) => "stop",
        Some(MixStop::DoseReached) => "dose",
        Some(MixStop::CanaryFired) => "canary",
        Some(MixStop::ProfileDone) => "profile",
        Some(MixStop::SplitDone) => "split",
        Some(MixStop::RoundDone) => "round",
        Some(MixStop::CircuitEmpty) => "empty",
    }
}

impl Mixer {
    pub(crate) fn pos_map(&self) -> (Vec<u32>, FxHashMap<u32, u32>) {
        let ids = self.arena.ids_in_order();
        let mut pos_of: FxHashMap<u32, u32> =
            FxHashMap::with_capacity_and_hasher(ids.len(), Default::default());
        for (i, &id) in ids.iter().enumerate() {
            pos_of.insert(id, i as u32);
        }
        (ids, pos_of)
    }

    /// The slice `[a, b)` of this mixer's circuit as piece input: gates and
    /// Meta verbatim, the canaries anchored inside it, the live undo entries
    /// lying wholly inside it, and the whole tabu ring.
    fn export_range(
        &self,
        ids: &[u32],
        pos_of: &FxHashMap<u32, u32>,
        a: usize,
        b: usize,
    ) -> PieceParts {
        let gates: Vec<XGate> = ids[a..b]
            .iter()
            .map(|&id| self.arena.gate(id).clone())
            .collect();
        let metas: Vec<Meta> = ids[a..b].iter().map(|&id| self.meta_of(id)).collect();
        let inside = |id: u32| -> Option<u32> {
            pos_of
                .get(&id)
                .copied()
                .filter(|&p| (p as usize) >= a && (p as usize) < b)
                .map(|p| p - a as u32)
        };
        let mut taps = Vec::new();
        for t in &self.taps {
            if let Some(pos) = inside(t.anchor) {
                taps.push(TapRec {
                    wire: t.wire,
                    orig_permille: t.orig_permille,
                    flips: t.flips,
                    pos,
                });
            }
        }
        let mut journal = Vec::new();
        for e in &self.journal {
            let live = e
                .after
                .iter()
                .all(|&(id, st)| self.arena.is_linked(id) && self.arena.stamp(id) == st);
            if !live || !self.arena.is_linked(e.pivot) {
                continue;
            }
            let Some(pivot) = inside(e.pivot) else {
                continue;
            };
            let mut after = Vec::with_capacity(e.after.len());
            let mut whole = true;
            for &(id, _) in &e.after {
                match inside(id) {
                    Some(p) => after.push(p),
                    None => {
                        whole = false;
                        break;
                    }
                }
            }
            if !whole {
                continue;
            }
            journal.push(JournalRec {
                before: e.before.clone(),
                dir: e.dir,
                pivot,
                after,
                event: e.event,
                origins: e.origins,
                gens: e.gens,
                litters: e.litters,
                litter_sizes: e.litter_sizes,
                misses: e.misses,
            });
        }
        PieceParts {
            gates,
            metas,
            taps,
            tabu: self.tabu.iter().copied().collect(),
            journal,
            counters: MixCounters::default(),
            moves_start: self.moves_done,
            moves_done: self.moves_done,
            eff_done: 0.0,
            len_start: b - a,
            event_base: 0,
            litter_base: 0,
            next_event: self.next_event,
            next_litter: self.next_litter,
            split_fail_streak: 0,
            split_end_reason: None,
            stop: None,
        }
    }

    /// Cut this mixer's circuit at `cuts` (interior positions, ascending) into
    /// piece inputs.
    pub(crate) fn export_slices(&self, cuts: &[usize]) -> Vec<PieceParts> {
        let (ids, pos_of) = self.pos_map();
        let n = ids.len();
        let mut bounds = Vec::with_capacity(cuts.len() + 2);
        bounds.push(0);
        bounds.extend(cuts.iter().copied().filter(|&c| c > 0 && c < n));
        bounds.push(n);
        bounds.dedup();
        bounds
            .windows(2)
            .map(|w| self.export_range(&ids, &pos_of, w[0], w[1]))
            .collect()
    }

    /// This (piece) mixer's whole state as parts for the fold: gates and Meta
    /// in order, canaries and live undo entries by position, the tabu ring,
    /// and the run bookkeeping. Takes the counters.
    pub(crate) fn export_parts(&mut self) -> PieceParts {
        let (ids, pos_of) = self.pos_map();
        let n = ids.len();
        let mut p = self.export_range(&ids, &pos_of, 0, n);
        p.counters = std::mem::take(&mut self.counters);
        p.moves_done = self.moves_done;
        p.eff_done = self.eff_done;
        p.next_event = self.next_event;
        p.next_litter = self.next_litter;
        p.split_fail_streak = self.split_fail_streak;
        p.split_end_reason = self.split_end_reason;
        p
    }

    /// Install per-gate and positional state on a freshly built arena whose
    /// ids are 0..n-1 in circuit order (the `resume_state` install sequence,
    /// in memory).
    fn install_parts(
        &mut self,
        metas: Vec<Meta>,
        taps: Vec<TapRec>,
        journal: Vec<JournalRec>,
        tabu: Vec<(u64, u64)>,
    ) {
        for (i, m) in metas.into_iter().enumerate() {
            self.set_meta(i as u32, m);
        }
        let ids = self.arena.ids_in_order();
        self.taps.clear();
        self.tap_at.clear();
        for t in taps {
            let anchor = ids[(t.pos as usize).min(ids.len() - 1)];
            self.tap_at
                .entry(anchor)
                .or_default()
                .push(self.taps.len() as u32);
            self.taps.push(Tap {
                anchor,
                wire: t.wire,
                orig_permille: t.orig_permille,
                flips: t.flips,
            });
        }
        let arena = &self.arena;
        let journal: VecDeque<UndoEntry> = journal
            .into_iter()
            .map(|r| UndoEntry {
                before: r.before,
                dir: r.dir,
                pivot: ids[r.pivot as usize],
                after: r
                    .after
                    .iter()
                    .map(|&p| {
                        let id = ids[p as usize];
                        (id, arena.stamp(id))
                    })
                    .collect(),
                event: r.event,
                origins: r.origins,
                gens: r.gens,
                litters: r.litters,
                litter_sizes: r.litter_sizes,
                misses: r.misses,
            })
            .collect();
        self.journal = journal;
        self.tabu = tabu.into_iter().collect();
    }

    /// A piece mixer over `parts`: W's gates with their Meta, verifying
    /// against its own slice, on the shared clock `moves_start`, minting ids
    /// from its own band, and silent unless `quiet` is false.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts(
        parts: PieceParts,
        num_wires: usize,
        params: MixParams,
        db: Arc<FrozenDb>,
        moves_start: u64,
        next_event: u64,
        next_litter: u64,
        quiet: bool,
    ) -> Mixer {
        let PieceParts {
            gates,
            metas,
            taps,
            tabu,
            journal,
            ..
        } = parts;
        let mut mx = Mixer::new_with_shared_db(gates, num_wires, params, db);
        mx.moves_done = moves_start;
        mx.counters.moves = moves_start;
        mx.next_event = next_event;
        mx.next_litter = next_litter;
        mx.quiet = quiet;
        mx.install_parts(metas, taps, journal, tabu);
        // Canaries belong to W: a piece never plants or reports its own.
        mx.taps_planted = true;
        mx.taps_reported = true;
        mx
    }

    /// Concatenate the finished pieces back into this (whole-circuit) mixer,
    /// in index order, and fold their bookkeeping: counters summed, the move
    /// clock advanced by the pieces' spend, id counters to the band maxima,
    /// the tabu ring merged with ages translated onto the new clock, canaries
    /// and undo entries re-positioned. Everything whole-circuit (the original,
    /// the RNGs, the profile controller, the split tri-state, the flag paths)
    /// is carried over. Returns the interior seams of the new concatenation.
    pub(crate) fn rebuild_from_parts(&mut self, parts: Vec<PieceParts>) -> Vec<usize> {
        let tabu_moves = self.params.tabu_moves;
        let spent: u64 = parts
            .iter()
            .map(|p| p.moves_done.saturating_sub(p.moves_start))
            .sum();
        let m_new = self.moves_done + spent;
        let total: usize = parts.iter().map(|p| p.gates.len()).sum();
        let mut gates = Vec::with_capacity(total);
        let mut metas = Vec::with_capacity(total);
        let mut taps = Vec::new();
        let mut journal = Vec::new();
        let mut seams = Vec::with_capacity(parts.len());
        let mut tabu: Vec<(u64, u64)> = self
            .tabu
            .iter()
            .copied()
            .filter(|&(_, mv)| mv + tabu_moves > m_new)
            .collect();
        let mut counters = std::mem::take(&mut self.counters);
        let mut next_event = self.next_event;
        let mut next_litter = self.next_litter;
        let mut streak = 0u32;
        let mut off = 0usize;
        for (i, p) in parts.into_iter().enumerate() {
            if i > 0 {
                seams.push(off);
            }
            let len = p.gates.len();
            gates.extend(p.gates);
            metas.extend(p.metas);
            taps.extend(p.taps.into_iter().map(|t| TapRec {
                pos: t.pos + off as u32,
                ..t
            }));
            journal.extend(p.journal.into_iter().map(|r| JournalRec {
                pivot: r.pivot + off as u32,
                after: r.after.iter().map(|&x| x + off as u32).collect(),
                ..r
            }));
            // The piece's own entries (its band) with their age translated
            // onto the merged clock; W's inherited entries were kept above.
            for &(e, mv) in &p.tabu {
                if e >= p.event_base {
                    let age = p.moves_done.saturating_sub(mv);
                    if age < tabu_moves {
                        tabu.push((e, m_new.saturating_sub(age)));
                    }
                }
            }
            counters += &p.counters;
            next_event = next_event.max(p.next_event);
            next_litter = next_litter.max(p.next_litter);
            streak = streak.max(p.split_fail_streak);
            off += len;
        }
        debug_assert!(
            tabu.windows(2).all(|w| w[0].0 < w[1].0),
            "merged tabu unsorted"
        );
        let jl = self.params.journal_len;
        if journal.len() > jl {
            journal.drain(0..journal.len() - jl);
        }
        let mut fresh = Mixer::new_with_shared_db(
            gates,
            self.num_wires,
            self.params.clone(),
            Arc::clone(&self.db),
        );
        fresh.runtime = self.runtime.clone();
        fresh.install_parts(metas, taps, journal, tabu);
        fresh.original = std::mem::take(&mut self.original);
        fresh.rng = self.rng.clone();
        fresh.metrics_rng = self.metrics_rng.clone();
        fresh.prof = self.prof.take();
        counters.moves = m_new;
        fresh.counters = counters;
        fresh.moves_done = m_new;
        fresh.next_event = next_event;
        fresh.next_litter = next_litter;
        fresh.anc = std::mem::take(&mut self.anc);
        fresh.anc_words = self.anc_words;
        fresh.anc_m = self.anc_m;
        fresh.anc_sampled = self.anc_sampled;
        fresh.anc_tracers = std::mem::take(&mut self.anc_tracers);
        fresh.split_on = self.split_on;
        fresh.split_done = self.split_done;
        fresh.split_ended = false;
        fresh.split_fail_streak = streak;
        fresh.split_end_reason = self.split_end_reason;
        fresh.taps_planted = self.taps_planted;
        fresh.taps_reported = self.taps_reported;
        fresh.stop_flag = self.stop_flag.take();
        fresh.dump_flag = self.dump_flag.take();
        fresh.dump_out = std::mem::take(&mut self.dump_out);
        fresh.stop_requested = self.stop_requested;
        fresh.gen_snap_base = self.gen_snap_base.take();
        fresh.last_gen_snap = self.last_gen_snap;
        fresh.db_record = self.db_record.take();
        fresh.db_mode_cur = self.db_mode_cur;
        fresh.pool_scan_due = self.pool_scan_due;
        fresh.brake_on = self.brake_on;
        fresh.brake_mark_move = self.brake_mark_move;
        fresh.brake_mark_size = self.brake_mark_size;
        fresh.quiet = self.quiet;
        *self = fresh;
        seams
    }
}

/// What the steering closure knows about a piece before it runs.
struct PieceSpec {
    start: usize,
    len: usize,
    comp: usize,
    total: usize,
}

struct RoundOut {
    pieces: usize,
    len_min: usize,
    len_max: usize,
    size_before: usize,
    size_after: usize,
    moves: u64,
    eff_inc: f64,
    progress: u64,
    stops: Vec<Option<MixStop>>,
}

/// Overrides every piece gets regardless of policy.
fn common_piece_overrides(pp: &mut MixParams) {
    pp.report_every = u64::MAX;
    pp.split = false;
    pp.split_stop = false;
    pp.split_canaries = 0;
    pp.ancestors = false;
    pp.anc_samples = 0;
    pp.gen_snap_every = 0;
    pp.snap_every_moves = 0;
    pp.prof_n = [0.0; 3];
    pp.prof_r = [0.0; 2];
    pp.eff_budget = 0.0;
    pp.span_norm = 0;
    pp.rank_base = 0;
    pp.rank_total = 0;
}

/// One round: cut, run the pieces (in parallel), fold back, verify.
fn run_round(
    w: &mut Mixer,
    cfg: &PieceCfg,
    pool: Option<&rayon::ThreadPool>,
    drv: &mut StdRng,
    partition: &mut Partition,
    round: usize,
    mk: &dyn Fn(&Mixer, &PieceSpec) -> MixParams,
) -> RoundOut {
    let s = w.arena.len();
    let cuts = partition.cuts(cfg, s, round, drv);
    let parts = w.export_slices(&cuts);
    let m = w.moves_done;
    let db = w.shared_db();
    let num_wires = w.num_wires;
    let quiet = !cfg.verbose;
    let base_e = w.next_event;
    let base_l = w.next_litter;
    let mut jobs = Vec::with_capacity(parts.len());
    let mut start = 0usize;
    let (mut len_min, mut len_max) = (usize::MAX, 0usize);
    for (i, mut p) in parts.into_iter().enumerate() {
        let len = p.gates.len();
        len_min = len_min.min(len);
        len_max = len_max.max(len);
        let comp = p.gates.iter().filter(|g| g.comp).count();
        let spec = PieceSpec {
            start,
            len,
            comp,
            total: s,
        };
        let mut params = mk(w, &spec);
        params.seed = seed_of(w.params.seed, round, i);
        p.event_base = base_e + ((i as u64) << 32);
        p.litter_base = base_l + ((i as u64) << 32);
        p.moves_start = m;
        p.len_start = len;
        jobs.push((p, params));
        start += len;
    }
    let runtime = w.runtime.clone();
    let run_one = |(part, params): (PieceParts, MixParams)| -> PieceParts {
        let (eb, lb, ms, ls) = (
            part.event_base,
            part.litter_base,
            part.moves_start,
            part.len_start,
        );
        let mut mx = Mixer::from_parts(part, num_wires, params, Arc::clone(&db), ms, eb, lb, quiet);
        mx.runtime = runtime.clone();
        let stop = mx.run();
        let mut out = mx.export_parts();
        out.event_base = eb;
        out.litter_base = lb;
        out.moves_start = ms;
        out.len_start = ls;
        out.stop = Some(stop);
        assert!(
            out.next_event - eb < (1u64 << 32) && out.next_litter - lb < (1u64 << 32),
            "piece id band overflow"
        );
        out
    };
    let outs: Vec<PieceParts> = match pool {
        Some(pool) => pool.install(|| jobs.into_par_iter().map(run_one).collect()),
        None => jobs.into_iter().map(run_one).collect(),
    };
    let out = RoundOut {
        pieces: outs.len(),
        len_min,
        len_max,
        size_before: s,
        size_after: outs.iter().map(|o| o.gates.len()).sum(),
        moves: outs
            .iter()
            .map(|o| o.moves_done.saturating_sub(o.moves_start))
            .sum(),
        eff_inc: outs
            .iter()
            .map(|o| (o.len_start as f64 / s.max(1) as f64) * o.eff_done)
            .sum(),
        progress: outs
            .iter()
            .map(|o| o.counters.split_prims + o.counters.split_hsplits + o.counters.split_segs)
            .sum(),
        stops: outs.iter().map(|o| o.stop).collect(),
    };
    partition.seams = w.rebuild_from_parts(outs);
    w.global_check();
    out
}

fn driver_line(w: &Mixer, round: usize, out: &RoundOut, extra: &str) {
    let mut stops: Vec<(&'static str, usize)> = Vec::new();
    for s in &out.stops {
        let name = stop_name(*s);
        match stops.iter_mut().find(|(n, _)| *n == name) {
            Some(e) => e.1 += 1,
            None => stops.push((name, 1)),
        }
    }
    let stops: Vec<String> = stops.iter().map(|(n, c)| format!("{n}x{c}")).collect();
    println!(
        "[fmix] pieces: round {} pieces={} len=[{}..{}] size {} -> {} moves+={} (total {}) eff+={:.4} stops={} {}",
        round,
        out.pieces,
        out.len_min,
        out.len_max,
        out.size_before,
        out.size_after,
        out.moves,
        w.moves_done,
        out.eff_inc,
        stops.join(","),
        extra
    );
}

/// Run the whole-circuit mixer `w` in piecewise-parallel rounds until the
/// stage's own stop rule fires. The policy follows `w.params`: a layer-2
/// profile (`prof_n[2] > 0`) runs the Profile policy, `split` runs the split
/// stage to global g57 exhaustion, otherwise a plain thermostat walk in
/// fixed-length rounds. `w` ends holding the concatenated circuit and every
/// whole-circuit statistic, so fmix's post-run tail (report, final float,
/// state file, output) needs no change.
pub fn run_piecewise(w: &mut Mixer, cfg: &PieceCfg) -> MixStop {
    assert!(
        cfg.min_block_size.is_some() || cfg.pieces >= 2,
        "run_piecewise needs at least 2 pieces or a block-size divisor"
    );
    if let Some(size) = cfg.min_block_size {
        assert!(size >= 2, "min_block_size must be at least 2");
        assert_eq!(
            cfg.pieces, 1,
            "min_block_size and pieces are mutually exclusive"
        );
    }
    if w.arena.len() == 0 {
        w.global_check();
        return MixStop::CircuitEmpty;
    }
    let policy = if w.params.prof_n[2] > 0.0 {
        Policy::Profile
    } else if w.params.split {
        assert!(
            w.params.split_stop,
            "piecewise split stage requires split_stop (the stage boundary ends the run)"
        );
        Policy::Split
    } else {
        Policy::Moves
    };
    let threads = if cfg.threads == 0 && cfg.min_block_size.is_some() {
        std::thread::available_parallelism().map_or(1, usize::from)
    } else if cfg.threads == 0 {
        cfg.pieces + 1
    } else {
        cfg.threads
    };
    let pool = if cfg.sequential {
        None
    } else {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .stack_size(64 << 20)
                .thread_name(|i| format!("piece{i}"))
                .build()
                .expect("piece thread pool"),
        )
    };
    let initial_pieces = cfg.pieces_for_len(w.arena.len());
    if let Some(size) = cfg.min_block_size {
        println!(
            "[fmix] automatic pieces: min_block_size={size} (P=max(1,current_gates/{size}) each round)"
        );
    }
    println!(
        "[fmix] pieces ON: pieces={} (shifted rounds {}), policy={:?}, threads={}, jitter={}, round_eff={}",
        initial_pieces,
        if initial_pieces > 1 {
            initial_pieces + 1
        } else {
            1
        },
        policy,
        if cfg.sequential { 1 } else { threads },
        cfg.jitter,
        cfg.round_eff
    );
    let mut drv = StdRng::seed_from_u64(w.params.seed ^ 0x9E37_79B9_7F4A_7C15);
    let mut partition = Partition::default();
    let mut round = 0usize;
    match policy {
        Policy::Profile => {
            if w.prof.is_none() {
                w.prof_init();
            }
            let cad = w.params.prof_cadence_eff.max(0.05);
            loop {
                if w.prof.as_ref().is_some_and(|p| p.phase == 4) {
                    w.report();
                    return MixStop::ProfileDone;
                }
                if w.moves_done >= w.params.moves {
                    w.global_check();
                    return MixStop::MovesBudget;
                }
                // An empty circuit has no slices and cannot advance a round.
                if w.arena.len() == 0 {
                    w.global_check();
                    return MixStop::CircuitEmpty;
                }
                let (eff, next_eff, pmix) = {
                    let p = w.prof.as_ref().expect("profile armed");
                    (p.eff, p.next_eff, p.pmix)
                };
                let d_eff = if cfg.round_eff > 0.0 {
                    cfg.round_eff
                } else {
                    (next_eff - eff).max(cad * 0.05)
                };
                let mk = |w: &Mixer, sp: &PieceSpec| -> MixParams {
                    let frac = sp.len as f64 / sp.total.max(1) as f64;
                    let mut pp = w.params.clone();
                    common_piece_overrides(&mut pp);
                    pp.p_mix = pmix.clamp(0.0, 1.0);
                    pp.size_hi = 0;
                    pp.size_lo = 0;
                    pp.target_size = ((w.params.target_size as f64 * frac).round() as usize).max(2);
                    pp.temp = (w.params.temp * frac).max(16.0);
                    pp.eff_budget = d_eff;
                    let remaining = w.params.moves.saturating_sub(w.moves_done);
                    let share = (remaining as f64 * frac).floor() as u64;
                    let want = (3.0 * d_eff * sp.len as f64).ceil() as u64;
                    pp.moves = w.moves_done + want.min(share).max(1);
                    pp
                };
                let out = run_round(w, cfg, pool.as_ref(), &mut drv, &mut partition, round, &mk);
                let due = {
                    let p = w.prof.as_mut().expect("profile armed");
                    p.eff += out.eff_inc;
                    p.eff >= p.next_eff
                };
                if due {
                    w.prof_update();
                    let p = w.prof.as_mut().expect("profile armed");
                    while p.next_eff <= p.eff {
                        p.next_eff += cad;
                    }
                }
                let extra = {
                    let p = w.prof.as_ref().expect("profile armed");
                    format!(
                        "eff={:.3} phase={} pmix={:.3} target={}",
                        p.eff, p.phase, p.pmix, w.params.target_size
                    )
                };
                driver_line(w, round, &out, &extra);
                w.report();
                w.check_flags();
                if w.stop_requested {
                    w.global_check();
                    return MixStop::StopFlag;
                }
                round += 1;
            }
        }
        Policy::Split => {
            if !w.taps_planted {
                w.plant_taps();
            }
            let reason: &'static str;
            loop {
                if w.remaining_g57() == 0 {
                    reason = "g57 pool exhausted";
                    break;
                }
                if round >= cfg.split_rounds_max {
                    reason = "round cap";
                    break;
                }
                if w.moves_done >= w.params.moves {
                    w.global_check();
                    return MixStop::MovesBudget;
                }
                let mk = |w: &Mixer, sp: &PieceSpec| -> MixParams {
                    let frac = sp.len as f64 / sp.total.max(1) as f64;
                    let mut pp = w.params.clone();
                    common_piece_overrides(&mut pp);
                    pp.split = true;
                    pp.split_stop = true;
                    pp.span_norm = sp.total;
                    pp.rank_base = sp.start as u32;
                    pp.rank_total = sp.total as u32;
                    let remaining = w.params.moves.saturating_sub(w.moves_done);
                    let share = (remaining as f64 * frac).floor() as u64;
                    pp.moves = w.moves_done + ((sp.comp + sp.len) as u64).min(share).max(1);
                    pp
                };
                let out = run_round(w, cfg, pool.as_ref(), &mut drv, &mut partition, round, &mk);
                let extra = format!("comp={}", w.remaining_g57());
                driver_line(w, round, &out, &extra);
                w.report();
                w.check_flags();
                if w.stop_requested {
                    w.global_check();
                    return MixStop::StopFlag;
                }
                round += 1;
                if out.progress == 0 {
                    reason = "failure limit";
                    break;
                }
            }
            // The stage boundary, as run() takes it under split_stop.
            w.split_on = false;
            w.split_ended = false;
            w.split_done = true;
            w.split_end_reason = Some(reason);
            w.restamp_ranks();
            w.announce_split_end(reason);
            w.moves_done += 1;
            w.counters.moves = w.moves_done;
            w.global_check();
            w.report();
            MixStop::SplitDone
        }
        Policy::Moves => {
            let d_eff = if cfg.round_eff > 0.0 {
                cfg.round_eff
            } else {
                0.5
            };
            loop {
                if w.moves_done >= w.params.moves {
                    w.global_check();
                    return MixStop::MovesBudget;
                }
                // An empty circuit has no slices and cannot advance a round.
                if w.arena.len() == 0 {
                    w.global_check();
                    return MixStop::CircuitEmpty;
                }
                let mk = |w: &Mixer, sp: &PieceSpec| -> MixParams {
                    let frac = sp.len as f64 / sp.total.max(1) as f64;
                    let mut pp = w.params.clone();
                    common_piece_overrides(&mut pp);
                    pp.target_size = ((w.params.target_size as f64 * frac).round() as usize).max(2);
                    pp.temp = (w.params.temp * frac).max(16.0);
                    pp.size_hi = 0;
                    pp.size_lo = 0;
                    pp.eff_budget = d_eff;
                    let remaining = w.params.moves.saturating_sub(w.moves_done);
                    let share = (remaining as f64 * frac).floor() as u64;
                    let want = (3.0 * d_eff * sp.len as f64).ceil() as u64;
                    pp.moves = w.moves_done + want.min(share).max(1);
                    pp
                };
                let out = run_round(w, cfg, pool.as_ref(), &mut drv, &mut partition, round, &mk);
                driver_line(w, round, &out, "");
                w.report();
                w.check_flags();
                if w.stop_requested {
                    w.global_check();
                    return MixStop::StopFlag;
                }
                round += 1;
            }
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/engine/mixer/piecewise/tests.rs"]
mod tests;
