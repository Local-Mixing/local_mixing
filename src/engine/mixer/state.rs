//! Mixer state, counters, constructors and externally visible stopping reasons.
use super::*;

pub const ORIGIN_SYNTH: u32 = u32::MAX;

/// The width-2 `comp = 1` population, split by control polarity.
///
/// `shaped == same_pol + opp_pol` by construction. The split separates two
/// things the single `g57=` field used to conflate:
///
/// - **`shaped`** is the DB-effectiveness reading. The store emits g57 circuits
///   and nothing else, so every gate the DB has ever spliced in has this shape,
///   and `1 - shaped/size` is the material the DB did not produce. Measured:
///   with twists off, `opp_pol/comp` holds at 1.000 for a whole run.
/// - **`same_pol`** is a twist odometer, not a structural fact. A negation
///   twist conjugates a window by NOT on one wire, flipping the polarity of
///   every literal on it (`conj_by_not`). A g57's two controls sit on distinct
///   wires, so a twist touches at most one, and touching one flips the pair
///   from opposite to same: same gate, same width, same `comp`, only the
///   polarity relation moves. It random-walks toward 1/2 under twist pressure
///   while DB splices inject fresh opposite-polarity material and pull it back.
///
/// Swap twists carry polarity with the wire and leave the split alone. A/B at
/// equal twist count, 400k moves: `opp_pol/comp` = 1.000 (no twists), 0.704
/// (neg), 1.000 (swap, at 2.6x the relabel count).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct G57Census {
    /// `comp = 1` with exactly two controls, polarity ignored: the shape the
    /// store can spell, hence the shape of everything it emits.
    pub shaped: usize,
    /// Of `shaped`, both controls the SAME polarity -- not a storable g57.
    pub same_pol: usize,
    /// Of `shaped`, OPPOSITE polarity (`a ^= b OR !c`) -- a true g57.
    pub opp_pol: usize,
}

impl G57Census {
    /// Fraction of the shaped population twists have flipped out of g57 form.
    /// 0 = untwisted, 1/2 = fully scrambled. Bounded and size-independent, so
    /// unlike `cov` -- whose denominator is the growing circuit -- it cannot
    /// fall while twists keep firing.
    pub fn pol_flipped(&self) -> f64 {
        if self.shaped == 0 {
            0.0
        } else {
            self.same_pol as f64 / self.shaped as f64
        }
    }
}

#[derive(Default)]
pub struct MixCounters {
    pub moves: u64,
    // Ingest-then-pay rounds: cheap (Compressing on cheap-tier laggards) and
    // paid (MinGrow on hard-tier gates); hits = spliced. db_hard_added is the
    // growth ledger — gates the paid channel added net; gen_misses = total
    // seed-miss bumps.
    pub db_ing_hits: u64,
    pub db_ing_rounds: u64,
    pub db_hard_hits: u64,
    pub db_hard_rounds: u64,
    pub db_hard_added: u64,
    // Candidates refused by the identity guard, and splices whose replacement
    // came from the curated store.
    pub db_identity_skips: u64,
    pub db_curated_hits: u64,
    // Curated replacements refused because they did not verify (see
    // try_db_splice_curated).
    pub db_curated_rejected: u64,
    // NOT-into-comp=1 absorptions (Merge::Absorb): the channel that lets a
    // twist bracket be swallowed by a neighbouring g57 rather than paid for.
    pub merges_absorb: u64,
    // g57-only COMP attempts and their hits.
    pub db_g57_rounds: u64,
    pub db_g57_hits: u64,
    // Slot-2 rounds, their hits, and the growth they added.
    pub db_slot2_rounds: u64,
    pub db_slot2_hits: u64,
    pub db_slot2_added: u64,
    // Times the brake engaged, and rounds spent under it.
    pub brake_engagements: u64,
    pub brake_rounds: u64,
    // Heads coins that found the pool drained (see db_attempt).
    pub canary_fallthrough: u64,
    // Descent rungs refused by the full-litter ban.
    pub litter_banned: u64,
    // Distance-from-minimal: sampled windows for which the store held ANY
    // non-identical equivalent, and how many of those admitted a STRICTLY
    // SHORTER one. The ratio is a live fcompress residual without running
    // fcompress -- as COMP drives the circuit toward its locally-minimal form
    // (the form the attacker-computable compressor would reach anyway) this
    // falls toward zero, which is the spelling diversity being spent.
    pub dmin_windows: u64,
    pub dmin_shorter: u64,
    // Twists placed on a pattern match, and twists that fell back to a random
    // position because no candidate matched within the try budget.
    pub twist_placed: u64,
    pub twist_place_fallback: u64,
    // Joint size distribution of successful splices: [outgoing len][incoming
    // len]. The shape of what the store actually trades, which the scalar rm=
    // and add= totals cannot show -- a channel that swaps 3 gates for 3 and one
    // that alternates 2->5 and 5->2 report identically.
    pub splice_sizes: Vec<Vec<u64>>,
    // Same joint histogram, curated-store splices only (splice_sizes minus
    // this = regular). A shape: not carried across resumes.
    pub splice_sizes_curated: Vec<Vec<u64>>,
    // Litter census (observation only — nothing bans or prefers on these yet;
    // see docs/FMIX_MENU.md 2.6). `litter_windows`/`litter_distinct_sum` give
    // the mean distinct litters per sampled DB window, i.e. how fast churn
    // fragments litters. `litter_full_spliced` counts splices whose outgoing
    // window was exactly one COMPLETE litter — precisely the replacements an
    // ssg-style full-litter ban would have refused, and therefore the number
    // that says whether the ban is worth wiring in here.
    pub litter_windows: u64,
    pub litter_distinct_sum: u64,
    pub litter_full_spliced: u64,
    pub gen_misses: u64,
    pub merges_cancel: u64,
    pub merges_xfuse: u64,
    pub merges_drop: u64,
    pub merges_subsume: u64,
    pub merges_sibling: u64,
    pub merges_cross_origin: u64,
    pub tabu_blocked: u64,
    pub merge_no_partner: u64,
    pub merge_wall_blocked: u64,
    pub merge_too_far: u64,
    pub merge_not_adjacent: u64,
    pub undos: u64,
    pub undo_dead: u64,
    pub undo_tabu: u64,
    pub undo_gather_miss: u64,
    // DB replacement moves (compressing channel + size-agnostic move).
    pub db_comp_hits: u64, // compressing: window replaced by a non-growing friend
    pub db_comp_misses: u64, // compressing: sampled window, no non-growing friend
    pub db_agn_hits: u64,  // size-agnostic: window replaced (any length)
    pub db_agn_misses: u64, // size-agnostic: sampled window, no equivalent found
    pub db_gates_removed: u64, // gates removed by accepted DB replacements
    pub db_gates_added: u64, // gates added by accepted (growing) DB replacements
    // Per-mode split of the same deltas (Compressing vs the rest), the
    // profile controller's plant observables: ghat = (mix_added -
    // mix_removed)/mix rounds, shat = (comp_removed - comp_added)/comp
    // rounds. Session-local (zero after a resume, like the histograms).
    // SELECTION ENTROPY of successful splices (session-local). `choice_*`
    // counts splices by whether the mode's eligible set held more than one
    // candidate; `choice_sum` totals the eligible-set sizes and
    // `choice_bits_milli` totals log2(eligible) in millibits, so the run can
    // report both "how often was there a choice", "how many on average" and
    // the actual entropy those choices injected through WHICH gates were
    // spliced -- as opposed to where.
    pub shuffles: u64,
    pub shuffle_moved: u64,
    pub shuffle_steps: u64,
    /// Wall nanoseconds spent inside global_shuffle. The move's cost was
    /// being INFERRED from displacement counters and a separate final-float
    /// timing; this measures it.
    pub shuffle_ns: u64,
    pub choice_splices: u64,
    pub choice_multi: u64,
    pub choice_sum: u64,
    pub choice_bits_milli: u64,
    pub db_mix_added: u64,
    pub db_mix_removed: u64,
    pub db_cmp_added: u64,
    pub db_cmp_removed: u64,
    pub db_wide_skip: u64, // support > 64 wires or budget hit: UNDECIDABLE, skipped
    // Wide windows (> 24 wires) verified by ANF comparison instead of the
    // exhaustive evaluator. Session-local.
    pub db_wide_poly: u64,
    // PER-OUTGOING-LENGTH breakdown of the DB move, indexed by the sampled
    // window's gate count (0..=LEN_HIST_MAX). Without it every rate the run
    // reports is an average over a random length draw, which conflates "this
    // width works" with "this width was sampled often". Session-local shapes,
    // like splice_sizes.
    pub len_attempts: Vec<u64>,
    pub len_hits: Vec<u64>,
    pub len_removed: Vec<u64>,
    pub len_added: Vec<u64>,
    pub len_span_skip: Vec<u64>,
    pub len_deg_skip: Vec<u64>,
    pub db_attempts: u64,     // total DB lookups attempted (both modes)
    pub db_degree_skips: u64, // attempts skipped by the degree guard (no lookup)
    pub db_span_skips: u64,   // attempts skipped by the span guard (no lookup)
    pub db_build_aborts: u64, // window builds aborted by the evade budget
    pub cross_r1: u64,
    pub cross_r2: u64,
    pub cross_r3: u64,
    pub presplits: u64,
    pub fresh_splits: u64,
    pub unsubs: u64,
    pub inserts: u64,
    pub twist_negs: u64,
    pub twist_swaps: u64,
    pub twist_cnots: u64,
    pub twist_relabels: u64,
    pub twist_case_splits: u64,
    pub twist_span: u64,
    pub twist_skips: u64,
    // --twist-g57 seam economics: context gates the brackets consumed, g57
    // word gates they emitted (net growth = emitted - consumed vs a flat +6
    // for the legacy packets), and total MITM solve time (the online-cost
    // answer, session-local).
    pub tg_consumed: u64,
    pub tg_emitted: u64,
    pub tg_solves: u64,
    pub tg_solve_ns: u64,
    // v2 placement gauges (session-local): seams that found their home by
    // sliding the window edge, and extra window redraws the joint-acceptance
    // rule spent.
    pub tg_slides: u64,
    pub tg_retries: u64,
    // Histogram of per-seam net cost (word len - context consumed), 0..=7.
    // A shape, so like splice_sizes it is not carried across resumes.
    pub tg_net_hist: [u64; 8],
    pub blocked_width: u64,
    pub blocked_deadlock: u64,
    pub declined: u64,
    pub boundary: u64,
    pub floats: u64,
    pub float_steps: u64,
    pub scatters: u64,
    pub scatter_steps: u64,
    pub dropped_neverfire: u64,
    // ---- split stage (docs/FMIX_SPLIT_TWIST.md) ----
    // Splits of the picked g57 (step 2), of the bracket g57 (4a/4c), and the
    // forced segment splits (5a); twist successes, step-4e failures, and
    // successes whose brackets sat in different circuit halves.
    pub split_prims: u64,
    pub split_hsplits: u64,
    pub split_segs: u64,
    pub split_joins: u64,
    pub split_fails: u64,
    pub split_xmid: u64,
    // Total canary flips (each = a twist span complementing the value carried
    // at a canary's position on its wire).
    pub tap_flips: u64,
    // Sum of twist span lengths (gates strictly between the brackets), and
    // the span-as-fraction-of-circuit histogram in 5% buckets (a SHAPE, not
    // carried across resumes, like width_hist).
    pub split_span_sum: u64,
    pub split_span_hist: [u64; 20],
    // Cross shots drawn from the min-dgen pool (vs uniform).
    pub cross_pool_shots: u64,
    // ---- pair geometry (docs/NONLOCAL_PHASE_A.md; session-local) ----
    // Rounds that drew the pair geometry, scans with no eligible partner,
    // scans cut by pair_scan_cap, fused windows and their splices, box-size
    // and fused-transport-distance tallies (sum/max over fused windows), and
    // candidates refused by the reorder ban.
    pub pair_rounds: u64,
    pub pair_boxes_empty: u64,
    pub pair_scan_truncs: u64,
    pub pair_fused: u64,
    pub pair_splices: u64,
    pub pair_box_sum: u64,
    pub pair_box_max: u64,
    pub pair_dist_sum: u64,
    pub pair_dist_max: u64,
    pub pair_perm_skips: u64,
    // ---- bridge fusion (docs/NONLOCAL_PHASE_A.md; session-local) ----
    // Rounds, walks clipped at the circuit tail, plans refused (carrier
    // unbuildable / mode-c collider / collider budget), pre-insert store
    // misses, far-splice rollbacks, half commits (near window missed after
    // the far one spliced), full commits, interior-length and collider
    // tallies, and wake correction gates inserted.
    pub bridge_rounds: u64,
    pub bridge_short: u64,
    pub bridge_refused: u64,
    pub bridge_probe_miss: u64,
    pub bridge_rollbacks: u64,
    pub bridge_half: u64,
    pub bridge_committed: u64,
    pub bridge_span_sum: u64,
    pub bridge_span_max: u64,
    pub bridge_colliders_sum: u64,
    pub bridge_wake_sum: u64,
    pub width_hist: [u64; 16],
}

impl MixCounters {
    pub fn merges(&self) -> u64 {
        self.merges_cancel + self.merges_xfuse + self.merges_drop + self.merges_subsume
    }
    pub fn expands(&self) -> u64 {
        self.cross_r1
            + self.cross_r2
            + self.cross_r3
            + self.presplits
            + self.fresh_splits
            + self.unsubs
            + self.inserts
            + self.twist_negs
            + self.twist_swaps
            + self.twist_cnots
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Meta {
    pub(crate) origin: u32,
    pub(crate) event: u64, // 0 = not a split product
    // Persistent shooting direction: a cross floats this gate in `dir`.
    // Fossils draw it uniformly at birth; fragments inherit per dir_p.
    pub(crate) dir: Dir,
    // Rewrite generation (benchmark semantics, 2026-07-21): how many
    // re-encodings this gate's material has been through since the input.
    // Input gates start at 0; a DB splice stamps its products with the
    // outgoing window's upper-median generation + 1; every SPLIT (presplit,
    // cross piece, fresh-split, unsubsume, twist case-split) stamps children
    // with parent + 1; merges take the min of their parents. Born-random
    // material (insert pairs, twist bracket packets) carries no input
    // structure and gets GEN_FRESH (= the spec's MAXGEN, higher than every
    // real generation; saturating arithmetic keeps it fixed).
    pub(crate) dgen: u32,
    // Litter tag (ssg `80a2c1d2` semantics, ported for measurement): the
    // replacement event that CREATED this gate. Input gates and born-random
    // material are singleton litters. A DB splice stamps all of its products
    // with one fresh id; splits and merges PROPAGATE the parent's id, so a
    // litter fragments under churn rather than being reassigned.
    //
    // `litter_size` is the size at creation and is deliberately NOT maintained
    // under splits: a window is "a complete litter" only when every gate shares
    // one id AND the count still equals that recorded size, so churn makes the
    // test conservative (it misses, never over-fires) exactly as in ssg.
    //
    // Observation only today — nothing bans or prefers on these yet; see
    // docs/FMIX_MENU.md §2.6.
    pub(crate) litter: u64,
    pub(crate) litter_size: u16,
}

/// Generation stamp for born-random material (fresh insert pairs, twist
/// bracket packets): these gates never held any input structure, so for the
/// gen-target machinery they count as already re-encoded.
pub const GEN_FRESH: u32 = u32::MAX;

// A recorded crossing, eligible for exact reversal while every emitted node is
// still alive and untouched (checked via arena stamps — any later split, merge
// or reuse of a piece bumps its stamp and kills the entry). Pieces pairwise
// commute (same target, none reads it) and may drift to either side of the
// pivot (rungs carry a flipped ladder literal, so under the separation
// exemption most commute with the pivot too); the gather accretes from both
// sides and the restore is exhaustively verified, so drift is harmless.
pub(crate) struct UndoEntry {
    // The pre-crossing pair, in circuit order for the recorded direction.
    pub(crate) before: [XGate; 2],
    pub(crate) dir: Dir,
    pub(crate) pivot: u32,
    pub(crate) after: Vec<(u32, u32)>, // (id, stamp) of every emitted node, incl. pivot
    pub(crate) event: u64,
    pub(crate) origins: [u32; 2], // origin of before[0], before[1]
    pub(crate) gens: [u32; 2],    // gen of before[0], before[1] (restored on undo)
    // Litter of before[0], before[1], restored on undo so that reversing a
    // crossing also reverses its provenance rather than minting new litters.
    pub(crate) litters: [u64; 2],
    pub(crate) litter_sizes: [u16; 2],
    pub(crate) misses: u8, // failed gather attempts; entry dropped after a few
}

/// Layer-2 profile-controller state (slot 0). While active it is the ONE
/// size authority: it owns `target_size` (the thermostat pulls toward the
/// moving setpoint), the static size brake is inert, and the per-round
/// MIX/COMP coin is driven by `pmix`. Best-effort contract: on saturation it
/// logs and holds the pinned lever rather than fighting the profile.
pub struct ProfState {
    pub phase: u8, // 1 expand, 2 hold, 3 compress, 4 done (hold at r2)
    pub pmix: f64,
    pub integ: f64,
    // Plant, all in GATES PER MOVE at full lever: ghat = drift if p_mix
    // were 1, shat = removal rate if p_mix were 0, dhat = the DISTURBANCE —
    // everything that changes size and is NOT the DB move (twists above all,
    // plus expansion moves and thermostat contractions). The controller does
    // not model twists; it measures their effect as the residual between
    // observed drift and DB-attributed drift, which is why it works at any
    // twist rate without being told what that rate is.
    pub ghat: f64,
    pub shat: f64,
    pub dhat: f64,
    pub eff: f64,
    pub next_eff: f64,
    pub sat: u32,
    pub s_in: f64,
    pub(super) base_moves: u64,
    pub(super) base_size: f64,
    pub(super) base_pmix: f64,
    pub(super) base_mix: [u64; 4], // agn_hits, agn_misses, db_mix_added, db_mix_removed
    pub(super) base_cmp: [u64; 4], // comp_hits, comp_misses, db_cmp_added, db_cmp_removed
}

pub struct Mixer {
    pub(super) runtime: RuntimeControls,
    pub arena: Arena,
    pub params: MixParams,
    pub counters: MixCounters,
    pub(super) meta: Vec<Meta>,
    // merge-key -> linked node ids with that (target, wire-set). Kept exact by
    // the index_add/index_remove hooks in every splice; validated at verify
    // points via indexed_count.
    // FxHashMap: accessed by key only (never iterated), so the hasher choice
    // cannot leak into behavior.
    pub(crate) index: FxHashMap<u64, Vec<u32>>,
    // Node id -> position in its merge-index bucket. This mirrors the split
    // indexes' position arrays and turns every splice's index removal from a
    // linear bucket search into an O(1) swap-remove without changing bucket
    // order or merge selection.
    pub(super) index_pos: Vec<u32>,
    pub(super) indexed_count: usize,
    // Twist-g57 seam-solve memo: eng.solve(t, MAX_WORD) is a pure function of
    // the target perm `t`, and each twist round re-solves the same handful of
    // targets (both seams x wire-pair candidates x k-contexts x slides x
    // retries). Pure cache — deliberately NOT serialized by save_state and not
    // restored by load_state; a resumed run just starts cold. SmallVec: word
    // length is <= swap_words::MAX_WORD = 7, so hits never allocate.
    pub(super) solve_memo: FxHashMap<u64, Option<smallvec::SmallVec<[u8; 7]>>>,
    pub(crate) journal: VecDeque<UndoEntry>,
    pub(super) tabu: VecDeque<(u64, u64)>, // (event, move at creation)
    pub(super) next_event: u64,
    // Next litter id. Input gates take 0..n as singleton litters, so fresh ids
    // start at n and never collide with them.
    pub(super) next_litter: u64,
    // litter id -> bitset over input-gate indices (see MixParams::ancestors).
    pub(super) anc: HashMap<u64, Vec<u64>>,
    pub(crate) anc_words: usize,
    pub(super) anc_m: usize,
    // Sampled mode: the bitset universe is the K tracers rather than all
    // `anc_m` input gates, and `anc_tracers[t]` is the input-gate index that
    // bit `t` stands for (sorted). Empty in exact mode.
    pub(super) anc_sampled: bool,
    pub(super) anc_tracers: Vec<u32>,
    pub(super) original: Vec<XGate>,
    pub(super) num_wires: usize,
    pub moves_done: u64,
    pub(crate) rng: StdRng,
    // Sampling RNG for report-line gauges only. Separate from `rng` so adding
    // or re-cadencing metrics never perturbs the move trajectory of a seed.
    pub(crate) metrics_rng: StdRng,
    // Graceful stop / on-demand snapshot, both file-flag driven (checked at the
    // report cadence): touch stop_flag -> finish cleanly; touch dump_flag ->
    // verified snapshot to dump_out, continue.
    pub(super) stop_flag: Option<String>,
    pub(super) dump_flag: Option<String>,
    pub(super) dump_out: String,
    // Generation-multiple snapshots (params.gen_snap_every): output base path
    // and the highest generation multiple already written.
    pub(super) gen_snap_base: Option<String>,
    pub(super) last_gen_snap: u32,
    pub(super) stop_requested: bool,
    // Frozen replacement store for the DB moves. Opened from the environment
    // only when a DB move is enabled; an empty (miss-everything) store otherwise
    // so runs without them never require FROZEN_DB_DIR. Shared (Arc) so the
    // piece mixers of a piecewise round can all read one open store.
    pub(super) db: Arc<FrozenDb>,
    pub(super) db_budget: XPolyBudget,
    // Optional per-attempt recorder (--db-record): one block per DB attempt with
    // the outgoing window, the number of equivalent DB circuits, and (on
    // success) the replacing subcircuit.
    pub(super) db_record: Option<std::io::BufWriter<std::fs::File>>,
    // Which geometry built the window of the CURRENT db_attempt (see
    // sample_window); stamped into --db-record attempt lines.
    pub(super) db_last_sampler: DbSample,
    // Gate count of the window the current DB attempt sampled, for the
    // per-length breakdown.
    pub(super) db_last_len: usize,
    // (seed id, its left neighbour when drawn). Window building floats gates,
    // so a FAILED attempt would otherwise leave the seed displaced -- and under
    // pool targeting the same stubborn gate is drawn repeatedly, which would
    // write a characteristic displacement into the circuit.
    pub(super) db_seed_home: Option<(u32, u32)>,
    // Set for the duration of one COMP attempt drawn as g57-only.
    pub(super) db_g57_only: bool,
    // Whether the CURRENT db_attempt drew the pair geometry: arms the
    // candidate reorder ban and the pair splice counter.
    pub(super) db_pair_round: bool,
    // The LIVE slot-2 mode. params.db_mode is what the brake returns to.
    pub(super) db_mode_cur: DbMode,
    // stable-ledger flow accounting (gates added/removed by Stable/StableGrow
    // splices). Deliberately NOT in MixCounters/state: a resume restarts the
    // ledger at zero, which just re-grants the slack once.
    pub(super) stable_led_added: u64,
    pub(super) stable_led_removed: u64,
    // per-success histogram of the store-minimal spelling length (the
    // converted permutation's operational complexity), index = gates (cap 31)
    pub(super) dmin_success_hist: [u64; 32],
    // TRUE complexity class of each big-pool (M1/M2/M3...) conversion, read
    // from the REFERENCE store named by FROZEN_REF_DIR — the pool swap
    // replaced the small spellings in the live store, so the class is only
    // recoverable by re-querying the same key against the original.
    pub(super) m123_class_hist: [u64; 32],
    // successful curated splices whose candidate list exceeded the bounded
    // contract (>20) — by construction exactly the M1/M2/M3 big-pool hits.
    pub(super) bigpool_hits: u64,
    // SIZE LEDGER for --db-mode band-ledger: signed net gates this channel
    // has added. The size choice is skewed corrective whenever |ledger|
    // exceeds the slack, so the band's size variability is kept per splice
    // while the total is conserved in expectation.
    pub(super) band_led: i64,
    // true for the current round when the effective mode came from BandLedger
    pub(super) band_led_round: bool,
    // DB attempts / successful splices split by window GEOMETRY
    // ([0] = contiguous i.e. sequential, [1] = convex). Plain Mixer fields,
    // deliberately not in MixCounters: adding there breaks the .state format.
    pub(super) geo_attempts: [u64; 2],
    pub(super) geo_hits: [u64; 2],
    pub(super) brake_on: bool,
    pub(super) brake_mark_move: u64,
    pub(super) brake_mark_size: usize,
    // Next move at which the pool is rebuilt.
    pub(super) pool_scan_due: u64,
    // Layer-2 profile controller (params.prof_n[2] > 0); see ProfState.
    pub(super) prof: Option<ProfState>,
    // The generation pool (see MixParams::pool_k).
    pub(super) pool: Vec<u32>,
    // Whether the current attempt's seed genuinely came from the pool, and
    // whether a heads coin fell through because the pool had drained.
    pub(super) seed_from_pool: bool,
    pub(super) seed_fell_through: bool,
    // Trailing window of qualifying-round outcomes (true = failed at every
    // rung), with a running failure count so the fraction is O(1).
    pub(super) canary: VecDeque<bool>,
    pub(super) canary_failures: usize,
    // ---- split stage (docs/FMIX_SPLIT_TWIST.md) ----
    // Live stage flag (params.split arms it; exits clear it), the
    // consecutive-4e-failure streak, and a one-move latch run() reads to stop
    // at the boundary under split_stop.
    pub(crate) split_on: bool,
    pub(crate) split_fail_streak: u32,
    pub(crate) split_ended: bool,
    // The stage ran to its boundary at some point in this run's history
    // (persisted): distinguishes "ended" from "never armed" on resume, and
    // zeroes the live split-twist dispatch for part 2 of a --split run.
    pub(crate) split_done: bool,
    // O(1)-samplable population indexes, maintained by index_add/index_remove:
    // every comp gate, and per target wire the bracket-eligible gates (comp,
    // or non-comp with exactly one control). pos vectors map id -> slot in its
    // list (NIL = absent) for O(1) swap-removal.
    pub(crate) comp_ids: Vec<u32>,
    pub(super) comp_pos: Vec<u32>,
    pub(crate) wt_buckets: Vec<Vec<u32>>,
    pub(super) wt_pos: Vec<u32>,
    // Reused by Stage 4's directional bracket draw. It is derived scratch,
    // deliberately absent from checkpoints; a resumed run starts empty and
    // fills it in the same bucket order before drawing any RNG.
    pub(crate) split_candidates: Vec<(u32, u32)>,
    // Wire canaries: flip monitors riding the material. A tap sits on `wire`
    // immediately right of `anchor`; taps re-anchor to the live left neighbor
    // when their anchor dies (evict_taps at every node-death site).
    pub(crate) taps: Vec<Tap>,
    // FxHashMap: accessed by key only (never iterated), so the hasher choice
    // cannot leak into behavior. Probed once per node in every twist span, so
    // SipHash-1-3 showed up against a map that only ever holds `split_canaries`
    // entries.
    pub(crate) tap_at: FxHashMap<u32, Vec<u32>>,
    pub(crate) taps_planted: bool,
    pub(crate) taps_reported: bool,
    // Approximate position ranks (id -> ordinal at last stamp; NIL = unknown),
    // restamped every RANK_EVERY moves. Heuristic only: half classification,
    // the midpoint-crossing counter and canary positions read it; correctness
    // never does.
    pub(crate) rank: Vec<u32>,
    pub(crate) rank_n: usize,
    pub(crate) rank_due: u64,
    // Min-dgen cross-shot pool (p_mincross): the K least-split lineages at
    // the last rebuild, consumed on draw so no single laggard is hammered.
    pub(crate) cross_pool: Vec<u32>,
    pub(crate) cross_pool_due: u64,
    // ---- piecewise rounds (mix/piecewise.rs) ----
    // Size-normalised work done under MixParams::eff_budget.
    pub(crate) eff_done: f64,
    // A piece mixer prints nothing: the whole-circuit mixer reports once per
    // round and emits the stage summaries from the folded counters.
    pub(crate) quiet: bool,
    // The reason the split stage ended (set by end_split_stage), so a quiet
    // piece can hand it to the driver without printing.
    pub(crate) split_end_reason: Option<&'static str>,
}

/// One split-stage wire canary. This is stored with the shared mixer state
/// because every node-death path must be able to re-anchor it, even when the
/// mutation originates in db_mixing or the crossing walk.
pub(crate) struct Tap {
    pub(crate) anchor: u32,
    pub(crate) wire: u16,
    pub(crate) orig_permille: u16,
    pub(crate) flips: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MixStop {
    MovesBudget,
    StopFlag,
    // Generation + twist-coverage dose targets met (gen_stop_frac): the run
    // ended as soon as the mixing dose was achieved, spending no further
    // moves (and hence no further incidental growth) past the requirement.
    DoseReached,
    // The canary fired: over the trailing window of qualifying rounds the
    // failure fraction exceeded canary_theta, i.e. the pool has converged on
    // material the store cannot spell and further moves buy nothing.
    CanaryFired,
    // The layer-2 size profile finished its third phase (arrived at R2, or
    // reached N2). The contract is fulfilled, so the run ends there rather
    // than burning the remaining move budget outside any setpoint -- which,
    // at a twist rate whose growth the lever cannot offset, would actively
    // undo the compression leg.
    ProfileDone,
    // The split stage ended (g57 exhaustion or the failure limit) under
    // --split-stop: the run stops at the stage boundary (trial mode).
    SplitDone,
    // Piecewise rounds: the piece spent its size-normalised work budget
    // (MixParams::eff_budget); the driver concatenates and re-cuts.
    RoundDone,
    // No gates remain to mix (for example, an identity block cancelled).
    CircuitEmpty,
}

/// Version stamp for the resume file. The parameter set has changed repeatedly,
/// and silently reinterpreting fields would be worse than refusing to load.
/// A parsed per-gate ancestry sidecar (see `write_anc_sidecar`): the ancestry
/// universe plus one ancestor set per gate, in circuit order. This is the
/// cross-RUN transport format — unlike the state file, which continues the
/// same run, a sidecar lets a NEW run inherit ancestor lists via --anc-in, so
/// a phase boundary stops resetting the ancestry clock.
pub struct AncSidecar {
    pub sampled: bool,
    /// Universe size: the ORIGINAL input-gate count the set bits index (exact
    /// mode), kept for span normalisation in sampled mode too.
    pub m: usize,
    pub words: usize,
    /// Sampled mode: the original-input index each bit stands for (sorted).
    pub tracers: Vec<u32>,
    /// One set per gate, `words` u64s each.
    pub sets: Vec<Vec<u64>>,
}

/// Generation census over the linked circuit (see `Meta::dgen`).
pub struct GenStats {
    /// Cap-eligible gates still below gen_target and still targetable
    /// (cheap + hard tiers) — the dose-stop numerator.
    pub lag: u64,
    /// Cap-eligible gates in total.
    pub elig: u64,
    /// Wide (cap-ineligible) gates below gen_target: invisible to the DB
    /// channel until some other move narrows or splits them (their pieces
    /// get parent + 1, so the walk lifts them).
    pub wlag: u64,
    /// Minimum gen over ALL linked gates (GEN_FRESH when nothing lags).
    pub min: u32,
    /// ALL gates below gen_target (eligible + wide + written-off) and the
    /// total.
    pub all_lag: u64,
    pub total: u64,
    /// The circuit generation: the largest G such that at least 95% of the
    /// TARGETABLE gates have generation >= G (the 5th-percentile generation
    /// over the population the DB channel can actually move — cap-eligible
    /// and not written off).
    ///
    /// It is deliberately NOT the percentile over all gates. Generations only
    /// advance under DB re-encoding, so a gate the DB can never touch sits at
    /// generation 0 forever; if such gates exceed 5% of the circuit they pin
    /// the all-gates percentile at 0 permanently, no matter how much mixing
    /// runs. That is a structural property of the material, not a measure of
    /// progress. On a product-share gadget ~62% of gates are wide, so the
    /// all-gates figure is identically 0 and says nothing.
    ///
    /// When every gate is targetable (the common case: no ctrl cap and no
    /// write-offs) this is exactly the old all-gates percentile, so runs on
    /// uniformly narrow material are unaffected.
    pub g_circ: u32,
    /// The old all-gates percentile, kept for continuity with earlier runs
    /// and reported as Gall=. Equals g_circ when everything is targetable.
    pub g_all: u32,
    /// Gates the DB channel can actually move: cap-eligible, minus those
    /// written off as unreachable. The denominator behind g_circ.
    pub targetable: u64,
}

impl Mixer {
    pub fn new(gates: Vec<XGate>, num_wires: usize, params: MixParams) -> Mixer {
        // Open the replacement store once, only when a DB move is enabled, so
        // runs without them never require FROZEN_DB_DIR.
        // Every channel that can reach the store must be in this test. Leaving
        // one out yields FrozenDb::empty(), so every lookup misses, the run does
        // zero re-encoding, and nothing says so -- it looks like a measurement.
        // The ingest/paid channels were once missing here, which yielded an
        // empty store and a run that reported zero re-encoding as if it were a
        // result; they are gone now, but the invariant remains.
        let db = if params.p_comp > 0.0
            || params.p_db > 0.0
            // Bridge is its own slot and reaches the store independently of
            // p_db; p_pair is subordinate to p_db (it only picks the geometry
            // inside a slot-2 round) so it needs no entry here.
            || params.p_bridge > 0.0
        {
            FrozenDb::from_env()
        } else {
            FrozenDb::empty()
        };
        Mixer::new_with_db(gates, num_wires, params, db)
    }

    /// As [`Mixer::new`] but with an explicit store (tests point this at a
    /// prebuilt frozen directory instead of `FROZEN_DB_DIR`).
    pub fn new_with_db(
        gates: Vec<XGate>,
        num_wires: usize,
        params: MixParams,
        db: FrozenDb,
    ) -> Mixer {
        Mixer::new_with_shared_db(gates, num_wires, params, Arc::new(db))
    }

    /// The store handle, for piece mixers that must share it (the store is
    /// read-only; opening it more than once per process is not affordable).
    pub fn shared_db(&self) -> Arc<FrozenDb> {
        Arc::clone(&self.db)
    }

    /// As [`Mixer::new_with_db`] over an already shared store handle.
    pub fn new_with_shared_db(
        gates: Vec<XGate>,
        num_wires: usize,
        mut params: MixParams,
        db: Arc<FrozenDb>,
    ) -> Mixer {
        let n = gates.len();
        if params.target_size == 0 {
            params.target_size = n;
        }
        if params.temp <= 0.0 {
            params.temp = (params.target_size as f64 / 100.0).max(64.0);
        }
        let num_wires = num_wires.max(crate::circuit::xgate::max_wire(&gates) as usize + 1);
        let mut rng = StdRng::seed_from_u64(params.seed);
        let metrics_rng = StdRng::seed_from_u64(params.seed ^ 0x5EED_517A75);
        let meta = (0..n)
            .map(|i| Meta {
                origin: i as u32,
                event: 0,
                dir: if rng.random_bool(0.5) { Dir::L } else { Dir::R },
                dgen: 0,
                // Input gates are singleton litters, as in ssg: they were not
                // emitted by any replacement, so there is no prior spelling a
                // full-litter rule could send them back to.
                litter: i as u64,
                litter_size: 1,
            })
            .collect();
        let db_mode0 = params.db_mode;
        // Ancestry universe: SAMPLED (K tracer input gates, fixed cost, scales)
        // takes precedence over EXACT (all input gates, |input| bits/litter).
        let (anc_words0, anc_m0, anc_sampled0, anc_tracers0) = if params.anc_samples > 0 {
            let k = params.anc_samples.min(n);
            (
                k.div_ceil(64),
                n,
                true,
                Self::pick_tracers(n, k, params.anc_sample_seed),
            )
        } else if params.ancestors {
            assert!(
                n <= 20_000,
                "--ancestors stores |input| bits per litter; {n} input gates is past                  the small-input envelope this instrument is for (use --anc-samples for large inputs)"
            );
            (n.div_ceil(64), n, false, Vec::new())
        } else {
            (0, 0, false, Vec::new())
        };
        // In sampled mode the tracers' own singleton sets are stored EXPLICITLY
        // (K entries), which is what lets `anc_or_into` drop the implicit
        // singleton rule: a non-tracer input gate then contributes nothing, so
        // untracked lineage costs no memory at all.
        let mut anc0: HashMap<u64, Vec<u64>> = HashMap::new();
        for (t, &gi) in anc_tracers0.iter().enumerate() {
            let mut bits = vec![0u64; anc_words0];
            bits[t / 64] |= 1u64 << (t % 64);
            anc0.insert(gi as u64, bits);
        }
        let mut index: FxHashMap<u64, Vec<u32>> = FxHashMap::default();
        let mut index_pos = vec![NIL; n];
        for (i, g) in gates.iter().enumerate() {
            let bucket = index.entry(key_of(g)).or_default();
            index_pos[i] = bucket.len() as u32;
            bucket.push(i as u32);
        }
        let db_budget = {
            let mut b = XPolyBudget::default();
            if params.db_wire_terms > 0 {
                b.max_poly_terms = params.db_wire_terms;
            }
            if params.db_total_terms > 0 {
                b.max_total_terms = params.db_total_terms;
            }
            b
        };
        let split_on0 = params.split;
        let mut mx = Mixer {
            runtime: RuntimeControls::LegacyEnvironment,
            arena: Arena::from_gates(gates.clone()),
            params,
            counters: MixCounters::default(),
            meta,
            index,
            index_pos,
            indexed_count: n,
            solve_memo: FxHashMap::default(),
            journal: VecDeque::new(),
            tabu: VecDeque::new(),
            next_event: 1,
            next_litter: n as u64,
            anc: anc0,
            anc_words: anc_words0,
            anc_m: anc_m0,
            anc_sampled: anc_sampled0,
            anc_tracers: anc_tracers0,
            original: gates,
            num_wires,
            moves_done: 0,
            rng,
            metrics_rng,
            stop_flag: None,
            dump_flag: None,
            dump_out: String::new(),
            gen_snap_base: None,
            last_gen_snap: 0,
            stop_requested: false,
            db,
            db_budget,
            db_record: None,
            db_last_sampler: DbSample::Contiguous,
            db_last_len: 0,
            db_seed_home: None,
            db_g57_only: false,
            db_pair_round: false,
            db_mode_cur: db_mode0,
            stable_led_added: 0,
            stable_led_removed: 0,
            dmin_success_hist: [0; 32],
            m123_class_hist: [0; 32],
            bigpool_hits: 0,
            band_led: 0,
            band_led_round: false,
            geo_attempts: [0; 2],
            geo_hits: [0; 2],
            brake_on: false,
            brake_mark_move: 0,
            brake_mark_size: 0,
            pool: Vec::new(),
            seed_from_pool: false,
            seed_fell_through: false,
            canary: VecDeque::new(),
            canary_failures: 0,
            pool_scan_due: 0,
            prof: None,
            split_on: split_on0,
            split_fail_streak: 0,
            split_ended: false,
            split_done: false,
            comp_ids: Vec::new(),
            comp_pos: Vec::new(),
            wt_buckets: Vec::new(),
            wt_pos: Vec::new(),
            split_candidates: Vec::new(),
            taps: Vec::new(),
            tap_at: FxHashMap::default(),
            taps_planted: false,
            taps_reported: false,
            rank: Vec::new(),
            rank_n: 0,
            rank_due: 0,
            cross_pool: Vec::new(),
            cross_pool_due: 0,
            eff_done: 0.0,
            quiet: false,
            split_end_reason: None,
        };
        mx.rebuild_side_index();
        // Collision-mask fast path (arena.rs): every wire a run touches stays
        // below num_wires, so num_wires <= 64 * MASK_WORDS (production: 128
        // wires = 2 words) means the mask side-array is authoritative for the
        // whole run and collides_ids never falls back to XGate::collides.
        debug_assert!(
            num_wires > 64 * crate::engine::arena::MASK_WORDS || mx.arena.masks_ok(),
            "collision masks poisoned despite num_wires = {num_wires}"
        );
        mx
    }

    /// Enable per-DB-attempt recording to `path` (see [`MixCounters`] db_*).
    pub fn enable_db_record(&mut self, path: &str) {
        match std::fs::File::create(path) {
            Ok(f) => self.db_record = Some(std::io::BufWriter::new(f)),
            Err(e) => eprintln!("[fmix] could not open --db-record {path}: {e}"),
        }
    }

    pub fn enable_flags(&mut self, stop: Option<String>, dump: Option<String>, dump_out: String) {
        self.stop_flag = stop;
        self.dump_flag = dump;
        self.dump_out = dump_out;
    }

    /// Arm generation-multiple snapshots (params.gen_snap_every): files go to
    /// `<base>.gen<m>.mpmct1` (+ `.gens` sidecar).
    pub fn set_gen_snap_base(&mut self, base: String) {
        self.gen_snap_base = Some(base);
    }
}
