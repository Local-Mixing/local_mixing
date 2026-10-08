//! Resolved mixer options and their unchanged default/overlay rules.
use super::*;

/// anything longer is folded into the top bucket.
pub const SPLICE_HIST_MAX: usize = 24;

/// Largest window length tracked in the per-outgoing-length DB breakdown.
/// Sized past any plausible `s_db` so a sweep never silently folds its widest
/// windows into the top bucket -- the point of the breakdown is exactly the
/// behaviour of the widest ones.
pub const LEN_HIST_MAX: usize = 32;

/// One mode's DB knobs after the base -> mode -> mode+geometry layering has
/// been applied. Produced by `MixParams::db_knobs`; the single source of
/// truth for what a mode will actually do.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolvedDbKnobs {
    pub s_db_cvx: usize,
    pub s_db_ctg: usize,
    pub p_convex: f64,
    pub p_mingen: f64,
    pub prefixes: bool,
}

/// How a DB move samples its outgoing window. Drawn ONCE per round, at the top
/// of `db_attempt_inner`, because the window length now depends on it: the TDP
/// profile wants a wide convex probe and a narrow contiguous one in the same
/// mode (`--s-db-ctg` / `--s-db-comp-ctg`).
///
/// The old `Mixed` variant and `DbSample::parse` are gone: geometry has been a
/// per-round `p_convex` coin since the sampler knobs were split per mode, so
/// nothing constructed `Mixed` and nothing called `parse`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DbSample {
    /// Pick a gate g, take it plus its w-1 neighbors in g's own direction,
    /// falling back to the other direction when the circuit end is reached.
    Contiguous,
    /// Grow a convex (mutually-gatherable) block: float g1 to its first
    /// non-commuting neighbor, then repeatedly float the whole block — in g1's
    /// direction w.p. p, else the opposite — to the next non-commuting gate and
    /// absorb it, until w gates are collected.
    Convex,
    /// Fuse the seed with a far COMMUTING partner: scan the seed's commutation
    /// box (the gates it could float past, capped at pair_scan_cap), pick one
    /// — the farthest eligible, or uniform under pair_pick_uniform — float the
    /// seed adjacent to it, and hand back the fused 2-gate window. The other
    /// samplers cannot build this window: Convex only absorbs colliders (a
    /// commuting gate is hopped past), and Contiguous pairs commuting gates
    /// only at physical distance 1.
    Pair,
    /// Not a sampler: the tag stamped on the two endpoint splices of a bridge
    /// round (`bridge_round`), which fuses two gates that commutation CANNOT
    /// bring together by conjugating the interior through a carrier and
    /// re-spelling both carrier-adjacent windows. Never drawn by the geometry
    /// coin.
    Bridge,
}

/// One planned bridge round: the two target gates,
// The twist menu is a single SWAP FAMILY (see `twist_move`): a wire swap,
// optionally composed with a negation of one or both swapped wires. The
// variant is chosen by two independent fair coins (alpha, beta) at move time,
// so there is no longer a kind enum -- swap 1/4, swap+negate-one 1/2,
// swap+negate-both 1/4. Each operator is realised as three single-control
// gates (a 3-CNOT swap network with control polarities carrying the negations).

#[derive(Clone)]
pub struct MixParams {
    // Max controls per gate: caps bridge wake corrections (conj_wake) and
    // sizes the reported width histogram.
    pub k_max: usize,
    // Directional transport. Every gate carries a left/right direction (input
    // gates get a random one). With db_advance, and for twist-g57 brackets, a
    // fresh product advances floor(dir_q * slack) gates in its own direction
    // at birth.
    pub dir_q: f64,
    // Total move attempts.
    pub moves: u64,
    /// Piecewise rounds: stop after this much size-normalised work (the sum
    /// of 1/size over the moves made, i.e. the profile controller's eff
    /// unit), returning `MixStop::RoundDone`. 0 = off (serial runs).
    pub eff_budget: f64,
    /// Probability each of the two swapped wires is NEGATED, independent per
    /// wire. 0.5 = the swap-family default (swap 1/4, negate-one 1/2,
    /// negate-both 1/4); 0.0 = pure positive swaps only (no polarity flips,
    /// though the 3-CNOT swap brackets are still inserted).
    pub twist_neg_p: f64,
    /// GLOBAL RE-RANDOMISATION rate. Each round, with probability
    /// `shuffle_rate / |circuit|`, every gate is floated to a uniformly random
    /// position inside its own commutation bounds. Rate 1.0 (the default) is
    /// therefore "expected one whole-circuit reshuffle per |circuit| rounds".
    /// 0 disables the move. Unlike a splice this changes only WHERE gates sit,
    /// never what they are, so it is function-preserving by construction --
    /// every step is a commutation.
    pub shuffle_rate: f64,
    // twist_g57: spell twist brackets as all-g57 words instead of 3-CNOT
    // packets, siting each bracket adaptively so it absorbs neighborhood
    // gates (the hidden-SAMF mechanism, XGate-native). Pure swap only: the
    // negation arms keep the legacy packet until their word tables exist.
    // Every gate the move inserts takes the ballistic birth-advance
    // unconditionally (the db_advance treatment; legacy brackets sit tight).
    // See twist_move_g57 and swap_words.rs.
    pub twist_g57: bool,
    // Twist window lengths are log-uniform over [twist_min_len, circuit size]:
    // the all-scales dial that decorrelates computational progress at every
    // window scale, the structured analog of ssg's long-range shooting.
    pub twist_min_len: usize,
    // Frozen-DB replacement move (slot 2). It samples a window, keys it by
    // its exact function, looks it up in the store and splices an equivalent
    // chosen by the live db_mode.
    //
    // p_db: probability that a round (not taken by a twist, shuffle or bridge)
    // is spent on a DB replacement. Stage 3 runs at 1.0. Off (0.0) by default:
    // the store is then never opened (unless p_bridge is set) and every round
    // that no other slot takes is idle.
    pub p_db: f64,
    // s_db: the length the descent STARTS from. The descent itself visits every
    // shorter length down to 1, so there is no separate minimum: one parameter
    // sets the ambition and the descent handles reality.
    pub s_db: usize,
    /// Minimum DB window length (0 = no floor). With the prefix descent off,
    /// restricts drawn windows to db_min_window..=s_db exactly.
    pub db_min_window: usize,
    // p_convex: probability the window sampler is convex rather than contiguous.
    // Replaces the three-valued DbSample: contiguous is 0, convex is 1, and the
    // old `mixed` is 0.5.
    pub p_convex: f64,
    // Window sampling geometry and its guards (see DbSample / db_attempt).
    // db_ctrl_cap (L): while building a window, a gate with more than L controls
    // is evaded (floated out of the way, else the build reverses, else aborts) so
    // high-degree gates that always miss are kept out of the window. 0 = no cap.
    // db_convex_p: for Convex, the probability each growth step floats the block
    // in g1's original direction (else the opposite).
    // db_mode: the slot-2 admission rule. Deterministic, not a coin -- set by
    // slot-0 rules, and the manual size brake (COMP arrests growth while still
    // stamping generations, so it slows the dose rather than stopping it).
    pub db_mode: DbMode,
    // Layer-2 mode overlay (slot 0). When p_mix >= 0, each round's slot-2 DB
    // move picks its mode by coin -- MIX-DB with probability p_mix, else
    // COMP-DB -- overriding the fixed db_mode, and reads that mode's own knobs.
    // The *_comp values are the COMP-mode overrides; each falls back to its base
    // value when unset (s_db_comp == 0, p_convex_comp < 0, p_mingen_comp < 0).
    // p_mix < 0 disables the overlay (single db_mode, exactly as before).
    pub p_mix: f64,
    // Pay-random MIX selection (layer 2 / db_mixing): when the MIX pool holds
    // only larger spellings, pick uniformly among ALL of them instead of
    // among the minimal ones — more growth and diversity per paid splice,
    // and a stronger up-lever for the profile controller.
    pub mix_pay_random: bool,
    // Layer-2 db_mixing size profile: effective-
    // work marks [n0, n1, n2] and size ratios [r1, r2] vs the input size.
    // n2 == 0 -> no profile. While a profile is active the controller is the
    // ONLY size authority: it owns the moving size setpoint, the size brake is
    // inert, and p_mix is driven by the controller (best-effort contract;
    // saturation is logged, never fought).
    pub prof_n: [f64; 3],
    pub prof_r: [f64; 2],
    // Controller knobs: control cadence in eff units, relative deadband on
    // |S - S*|, max |Δp_mix| per update, EWMA weight for fresh per-round
    // drift estimates, integral gain.
    pub prof_cadence_eff: f64,
    pub prof_deadband: f64,
    pub prof_dp_max: f64,
    pub prof_ewma: f64,
    pub prof_ki: f64,
    // ---- Layered DB knobs: base -> mode -> mode+geometry ----
    //
    // Every override is `Option`, and `None` is the ONLY way to say "not set at
    // this level, fall through". This is deliberate and was a bug fix: these
    // used to be sentinel-encoded (0 for usize, < 0 for f64), which works only
    // while the shipped default IS the sentinel. It stopped being one --
    // `s_db_comp` ships at 12 and `p_convex_comp` at 0.9 -- so both fired
    // unconditionally and silently shadowed an explicit `--s-db` / `--p-convex`
    // in COMP rounds. Worse, 0 is a LEGITIMATE value here: `p_mingen_comp = 0`
    // is what the TDP profile wants, and a sentinel scheme cannot tell it from
    // "unset".
    //
    // Resolution is by specificity and lives entirely in the `active_*` methods
    // below. Deciding WHICH level the user actually asked for is the CLI's job
    // (circuit_mixer.rs), because only there can clap's ValueSource distinguish "the
    // user passed this value" from "this is merely the default".
    pub s_db_comp: Option<usize>,
    pub p_convex_comp: Option<f64>,
    pub p_mingen_comp: Option<f64>,
    // Per-GEOMETRY window length, on top of the per-MODE split. The two
    // samplers have very different cost curves -- a contiguous window of the
    // same gate count spans far more wires, and its canonicalization cost runs
    // 3.6x convex at length 5, 12.6x at 7 and 47.8x at 12 -- so a profile that
    // wants a wide convex probe usually wants a much narrower contiguous one.
    pub s_db_ctg: Option<usize>,
    pub s_db_comp_ctg: Option<usize>,
    // Per-MODE prefix descent. The mode overlay (--p-mix) runs MIX and COMP in
    // one process, and they want opposite settings: COMP descends (it is the
    // compression lever, worth ~600x on transport), while MIX draws one uniform
    // length (descent there just re-probes lengths whose expansion band is only
    // 1..~5).
    pub db_prefixes_mix: Option<bool>,
    pub db_prefixes_comp: Option<bool>,
    // Two eligibility thresholds, not one. w_window governs what may sit INSIDE
    // a window; w_pool governs what may SEED one and count toward the dose.
    // They want different values: width-3 gates match in context often enough
    // to be worth admitting to windows, but their end-to-end per-gate re-encode
    // rate is 0.41% against 98.98% for width <= 2, so at a shared threshold they
    // pile up at the bottom of the pool with nothing to eject them.
    pub w_window: usize,
    pub w_pool: usize,
    pub db_convex_p: f64,
    // Exhaustive per-splice equivalence check on DB replacements. Correctness
    // rests on the key/decode invariants, so this is a safety net; disabling it
    // speeds long runs (the periodic global_check still guards). Windows whose
    // support exceeds 24 wires can only be spliced with verification OFF.
    pub db_verify: bool,
    // Measurement mode: sample windows and record the DB match count via
    // --db-record but NEVER splice (the circuit stays stationary). With
    // p_db = 1.0 and all other weights 0 this makes circuit_mixer a pure match-rate
    // sampler over the input circuit.
    pub db_dry_run: bool,
    // Degree pre-filter for DB lookups: a window whose function degree exceeds
    // db_max_degree (the max ANF degree any stored circuit has) cannot match, so
    // it is skipped before the expensive canonicalization. 0 = off. This is the
    // guard that keeps the DB move cheap on the high-width windows the walk
    // produces (which almost always miss). db_degree_probes random subcubes are
    // tested per direction.
    pub db_max_degree: usize,
    pub db_degree_probes: usize,
    // Span pre-filter for DB lookups: a window touching more distinct wires
    // than this is recorded as a miss without canonicalizing. Set to the max
    // canonical support any stored function has (census: frozen_degree_scan).
    // Unlike the degree guard this is not a strict certificate — a window's
    // FUNCTION could depend on fewer wires than the window touches — but exact
    // cancellation is vanishingly rare, and canonicalizing wide-span windows
    // is the dominant cost (Rule-L over large tied wire groups). 0 = off.
    pub db_max_span: usize,
    // Term caps for the DB lookup's polynomial budget: a window whose wire
    // polys (or their sum) outgrow the largest any stored function has cannot
    // match, and the budget Err lands BEFORE the Rule-L canonicalization that
    // dominates dense-window cost. 0 = legacy XPolyBudget defaults (2^18/2^20).
    // Set from the frozen_degree_scan census of the store.
    pub db_wire_terms: usize,
    pub db_total_terms: usize,
    // Key every prefix window[..p] (p in [db_min_window, len]) of each sampled
    // window instead of only the full window: one walk, many lookup shots.
    // Dry-run measurement only for now — the splice policy when several
    // prefixes match (longest vs uniform) is an open design choice.
    pub db_prefixes: bool,
    // db_advance: give DB splice products the ballistic birth-advance that
    // twist-g57 brackets get (advance floor(dir_q * slack) along the product's own
    // direction). Without it a splice assigns directions that nothing ever acts
    // on. The historic alternative source of transport -- crossings -- widened
    // material, and width is what kills DB matching, so this is the transport
    // channel that does not fight the store. Off by default: it changes
    // trajectories, so the A/B is one flag.
    pub db_advance: bool,
    // ---- pair geometry ----
    // p_pair: probability a non-COMP DB round samples its window with the PAIR
    // geometry: the seed plus one far COMMUTING partner, floated adjacent and
    // fused into a 2-gate window. The db_mixing transport move — the fused
    // splice unions litters across the seed's whole commutation box instead
    // of one window span. 0 disables and DRAWS NO RNG: the walk is
    // move-for-move identical to the pair-less chain at equal seed.
    pub p_pair: f64,
    // Cap on the pair box scan (gates examined past the seed before giving up).
    pub pair_scan_cap: usize,
    // Partner policy: false = farthest eligible gate in the box (max transport
    // per move), true = uniform over the eligible box.
    pub pair_pick_uniform: bool,
    // ---- bridge fusion ----
    // p_bridge: per-round probability of one bridge round — jointly re-encode
    // two gates commutation cannot bring together, by conjugating the interior
    // through a carrier (wake corrections on interior colliders) and
    // re-spelling both carrier-adjacent windows through the store. 0 disables
    // and DRAWS NO RNG. Corrections are non-g57 conjunctions: this move
    // trades polf for reach, like the legacy twist packets.
    pub p_bridge: f64,
    // Interior length draw is log-uniform in [bridge_min_span,
    // bridge_max_span].
    pub bridge_min_span: usize,
    pub bridge_max_span: usize,
    // Refuse a round whose interior holds more colliders than this (each
    // collider costs one or two correction gates).
    pub bridge_max_colliders: usize,
    // ancestors: track, per litter, the SET of original input gates that
    // contributed to it -- the union of the sets of the litters the outgoing
    // window drew on. Unlike the single `origin` label (which a mixed-lineage
    // splice destroys, see osyn=), a union never loses information, so it
    // measures how far input material actually travels and what a mixed gate is
    // made of. Cost is |litters| x |input| bits, so this is a small-input
    // instrument: it refuses to arm above `ANC_MAX_INPUT` gates.
    pub ancestors: bool,
    /// SAMPLED ancestry: instead of the full ancestor set per litter (which
    /// costs |input| bits and caps the instrument at 20k input gates), track
    /// only `anc_samples` randomly chosen input gates -- "tracers" -- and for
    /// each one the set of current gates descended from it. Cost is a fixed
    /// K bits per litter regardless of input size, so this scales to production
    /// circuits. 0 = off; takes precedence over `ancestors` when both are set.
    pub anc_samples: usize,
    /// Tracer-selection seed. Default 0 means the tracer set is a function of
    /// (input size, K) alone, so it is identical across runs and across a
    /// resume -- which makes schedules comparable and makes a resumed run track
    /// the same input gates. Vary it for independent replicates.
    pub anc_sample_seed: u64,
    // Size brake (hysteresis). Growth to size_hi arms COMP; the mode is
    // released back to db_mode at size_lo OR when COMP stops paying, whichever
    // comes first. The productivity release is what makes a WIDE band safe: the
    // risk was never the width, it was sitting in COMP past its usefulness,
    // where it starves (declines rise as the circuit approaches local
    // minimality) and spends re-encoding diversity (COMP draws only from
    // minimum-size spellings, pulling toward the form a local compressor
    // would reach).
    // 0 = brake off.
    pub size_hi: usize,
    pub size_lo: usize,
    // Release COMP when its shed rate over the trailing window falls below this
    // many gates per round.
    pub comp_release_eps: f64,
    pub comp_release_window: u64,
    // Fixed top-level twist rate (slot 1): with this probability a round
    // performs one conjugation twist from the swap family (see twist_move;
    // twist_neg_p sets the negation coins). The size machinery balances
    // around the fixed twist rate: bracket mass is absorbed by the DB move
    // like any other growth source.
    pub p_twist: f64,
    // Generation targeting: drive every (cap-eligible) gate through at least
    // gen_target DB re-encodings. With gen_target > 0, a DB seed is drawn
    // from the laggard list (gates with gen < gen_target) with probability
    // gen_bias instead of uniformly, turning the coupon-collector tail of
    // uniform selection into direct work — fewer moves, hence less incidental
    // growth, for the same minimum generation. 0 = off (trajectory identical
    // to the untargeted chain at equal seed).
    pub gen_target: u32,
    // p_mingen: probability a DB seed is drawn from the generation POOL rather
    // than uniformly -- what stops the process being a coupon collector, where
    // the last few percent of gates soak up most of the moves.
    pub p_mingen: f64,
    // pool_k: the pool is the K lowest-generation gates among those that are
    // pool-eligible AND still below the goal. Both filters are load-bearing. An
    // ineligible gate can never be re-encoded, so its generation is pinned
    // forever and an unfiltered pool converges on exactly that set; and without
    // the below-goal filter a late-run pool is padded with ordinary
    // low-but-finished gates that re-encode fine, so the canary could never
    // fire. A COUNT, not a fraction: the drain rate is set by the move economy
    // (gen_rescan x p_db x p_mingen) and is independent of circuit size, so a
    // percentage over-provisions as the circuit grows and under-provisions on
    // small ones. K must exceed the draws taken between rebuilds, or the pool
    // empties and the biased coin silently degrades to uniform.
    pub pool_k: usize,
    // Canary: fire when the failure fraction over the last canary_window
    // QUALIFYING rounds exceeds canary_theta. Healthy failure rates sit well
    // under 0.2 (five rungs against ~99% per-window hit rates on width-<=2
    // material) while the pathological case drives toward 1.0, so 0.9 sits in a
    // wide gap. 0 = off.
    pub canary_theta: f64,
    pub canary_window: usize,
    // litter_ban: refuse a window that is exactly one COMPLETE litter -- the
    // unit some earlier replacement emitted, and therefore precisely where the
    // store can hand the outgoing spelling straight back (A -> B -> A).
    // Singleton litters are exempt by construction: an input gate has no
    // earlier spelling to be returned to, and banning it would also refuse the
    // descent's length-1 rung, the one rung that always makes progress.
    pub litter_ban: bool,
    // litter_samples: draw this many candidate windows and keep the one
    // spanning the MOST distinct litters. 1 = off. Discarded candidates may
    // leave ctrl-cap evasion floats behind; those are function-preserving and
    // the walk floats constantly, so the cost is arena churn, not correctness.
    pub litter_samples: usize,
    // twist_place_tries: how many candidate positions the twist placer samples
    // looking for a TWIST_PATTERNS match before giving up and placing the twist
    // uniformly at random. 0 = always random, which is the historical
    // behaviour.
    pub twist_place_tries: usize,
    // Pool rebuild cadence in moves (an O(size) scan each time).
    pub gen_rescan: u64,
    // Median variant for the DB stamp: false (default) = upper median
    // (sorted[len/2], median rounded up on even windows); true = lower
    // median (sorted[(len-1)/2], rounded down). On 2-gate windows — the most
    // common splice — the lower median IS the min, so this probes how close
    // the median rule sits to min-semantics' straggler-bound climb.
    pub gen_median_low: bool,
    // Dose-based stop: with gen_target > 0 and gen_stop_frac >= 0, the run
    // ends (MixStop::DoseReached) at the first report point where the
    // laggard fraction among the TARGETABLE gates (cap-eligible, not written
    // off — i.e. lag/targetable, the same population behind g_circ) is
    // <= gen_stop_frac AND the cumulative per-position twist coverage
    // (twist_span / size) has reached twist_cov_stop (0 = no coverage
    // requirement). The move budget becomes a ceiling: db_mixing runs exactly
    // as long as the dose requires.
    pub gen_stop_frac: f64,
    pub twist_cov_stop: f64,
    // Generation-multiple snapshots: at each report point, when the circuit
    // generation (g_circ, the 5th-percentile gate generation) crosses a fresh
    // multiple of this interval, write a verified snapshot to
    // <base>.gen<m>.mpmct1 (+ .gens sidecar); the base path is armed with
    // Mixer::set_gen_snap_base. 0 = off.
    pub gen_snap_every: u32,
    // Move-multiple snapshots: at each report point where moves_done is a
    // multiple of this interval, write a verified snapshot to
    // <base>.mv<moves>.mpmct1 (+ .gens sidecar). The progress clock for
    // regimes where the generation census is not meaningful (e.g. p_db = 0).
    // Choose a multiple of report_every. 0 = off.
    pub snap_every_moves: u64,
    pub verify_every: u64,
    pub report_every: u64,
    pub local_verify: bool,
    pub seed: u64,
}

impl MixParams {
    /// Settle the layered DB knobs for one mode: base -> mode -> mode+geometry,
    /// most specific wins, `None` meaning "this level says nothing".
    ///
    /// THE single source of these rules. The CLI banner calls it too, so what a
    /// run prints is by construction what the mixer will do -- the old banner
    /// re-derived the fall-through itself and could drift, which is how the
    /// COMP-shadowing bug stayed invisible.
    pub fn db_knobs(&self, mode: DbMode) -> ResolvedDbKnobs {
        let comp = mode == DbMode::Compressing;
        let opt = |o: Option<f64>| if comp { o } else { None };
        ResolvedDbKnobs {
            s_db_cvx: if comp { self.s_db_comp } else { None }.unwrap_or(self.s_db),
            // COMP contiguous falls back to COMP convex, not straight to the
            // base: a run that set --s-db-comp meant it for both geometries.
            s_db_ctg: if comp {
                self.s_db_comp_ctg.or(self.s_db_comp)
            } else {
                self.s_db_ctg
            }
            .unwrap_or(self.s_db),
            p_convex: opt(self.p_convex_comp).unwrap_or(self.p_convex),
            p_mingen: opt(self.p_mingen_comp).unwrap_or(self.p_mingen),
            prefixes: if comp {
                self.db_prefixes_comp
            } else {
                self.db_prefixes_mix
            }
            .unwrap_or(self.db_prefixes),
        }
    }
}

impl Default for MixParams {
    fn default() -> MixParams {
        MixParams {
            k_max: 12,
            dir_q: 0.85,
            moves: 1_000_000,
            eff_budget: 0.0,
            twist_neg_p: 0.5,
            shuffle_rate: 2.0,
            twist_g57: false,
            twist_min_len: 64,
            // Store-free by default: MixParams::default() is the test/base
            // value, and any positive DB rate here would make every construction
            // demand FROZEN_DB_DIR. The PRODUCTION defaults live on the circuit_mixer
            // CLI, where a run that wants the store asks for it -- as of
            // 2026-08-03 that means s_db 9, p_convex 0.4, s_db_comp 12,
            // p_convex_comp 0.9, db_prefixes ON. The values below stay the store-free
            // test baseline on purpose; do not "sync" them to the CLI.
            p_db: 0.0,
            s_db: 5,
            db_min_window: 0,
            p_convex: 0.5,
            db_mode: DbMode::Mix,
            p_mix: -1.0,
            mix_pay_random: false,
            prof_n: [0.0; 3],
            prof_r: [0.0; 2],
            prof_cadence_eff: 0.5,
            prof_deadband: 0.02,
            prof_dp_max: 0.1,
            prof_ewma: 0.3,
            prof_ki: 0.05,
            s_db_comp: None,
            p_convex_comp: None,
            p_mingen_comp: None,
            s_db_ctg: None,
            s_db_comp_ctg: None,
            db_prefixes_mix: None,
            db_prefixes_comp: None,
            w_window: 4,
            w_pool: 3,
            db_convex_p: 0.75,
            db_verify: true,
            db_dry_run: false,
            db_max_degree: 0,
            db_degree_probes: 6,
            db_max_span: 0,
            db_wire_terms: 0,
            db_total_terms: 0,
            db_prefixes: false,
            db_advance: false,
            p_pair: 0.0,
            pair_scan_cap: 4096,
            pair_pick_uniform: false,
            p_bridge: 0.0,
            bridge_min_span: 16,
            bridge_max_span: 512,
            bridge_max_colliders: 8,
            ancestors: false,
            anc_samples: 0,
            anc_sample_seed: 0,
            size_hi: 0,
            size_lo: 0,
            comp_release_eps: 0.0,
            comp_release_window: 250_000,
            p_twist: 0.0,
            gen_target: 0,
            p_mingen: 0.8,
            pool_k: 20_000,
            canary_theta: 0.0,
            canary_window: 2000,
            litter_ban: false,
            litter_samples: 1,
            twist_place_tries: 0,
            gen_rescan: 10_000,
            gen_median_low: false,
            gen_stop_frac: -1.0,
            twist_cov_stop: 0.0,
            gen_snap_every: 0,
            snap_every_moves: 0,
            verify_every: 10_000,
            report_every: 50_000,
            local_verify: true,
            seed: 0,
        }
    }
}

/// Resolved twist and phase-stop controls. Database-backed
/// walks retain their default replacement/cache policy. These options replace
/// the corresponding environment overrides and are deliberately not part
/// of the versioned checkpoint format; supply them again when resuming.
#[derive(Clone)]
pub struct MixRuntimeOptions {
    pub twist_g57_slide: bool,
    pub twist_g57_retry: bool,
    pub stop_at_phase: Option<u32>,
    /// Finish cleanly the moment cumulative successful slot-2 DB splices
    /// reach this count (checked every move, so the stop is exact). For
    /// matched-splice-budget comparisons across arms.
    pub stop_at_splices: Option<u64>,
}

impl Default for MixRuntimeOptions {
    fn default() -> Self {
        Self {
            twist_g57_slide: true,
            twist_g57_retry: true,
            stop_at_phase: None,
            stop_at_splices: None,
        }
    }
}
