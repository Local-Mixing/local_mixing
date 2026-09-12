//! fmix CLI options and precedence-aware resolution of DB policy layers.
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "fmix")]
pub struct Args {
    #[command(flatten)]
    pub(super) quality: QualityArgs,
    /// Input circuit file. Not required when --resume is given: a resume
    /// rebuilds the circuit, its metadata and the original from the state file.
    #[arg(long, required_unless_present = "resume")]
    pub(super) input: Option<String>,
    /// Input format: mpmct1 | g57
    #[arg(long, default_value = "mpmct1")]
    pub(super) input_format: String,
    /// Output file (mpmct1 format)
    #[arg(long)]
    pub(super) output: Option<String>,
    /// Thermostat target gate count (default: input size)
    #[arg(long)]
    pub(super) target_size: Option<usize>,
    /// Thermostat softness in gates (default: max(target/100, 64))
    #[arg(long)]
    pub(super) temp: Option<f64>,
    /// Total move attempts
    #[arg(long, default_value_t = 1_000_000)]
    pub(super) moves: u64,
    /// Max controls per gate (K)
    #[arg(long, default_value_t = 12)]
    pub(super) k_max: usize,
    /// Width-damping offset D for expansion moves (fsplit convention)
    #[arg(long, default_value_t = 2)]
    pub(super) split_damp: usize,
    /// Width-damper base B: a split of parent width c proceeds with
    /// probability B^-(c - split_damp)
    #[arg(long, default_value_t = 2.0)]
    pub(super) split_base: f64,
    /// Probability that a collision fragment inherits the shot gate's
    /// direction (else it gets the opposite)
    #[arg(long, default_value_t = 0.75)]
    pub(super) dir_p: f64,
    /// Directional transport fraction: fresh pieces advance floor(q * slack)
    /// in their own direction at birth; a failed cross retreats the shot gate
    /// floor((1-q) * way)
    #[arg(long, default_value_t = 0.85)]
    pub(super) dir_q: f64,
    /// Max distance (gates) a merge partner may sit from the initiator
    #[arg(long, default_value_t = 4096)]
    pub(super) merge_reach: usize,
    /// Undo journal capacity (recorded crossings eligible for reversal)
    #[arg(long, default_value_t = 262_144)]
    pub(super) journal_len: usize,
    /// Fraction of contraction moves that try a journal undo first
    #[arg(long, default_value_t = 0.5)]
    pub(super) undo_frac: f64,
    /// Refractory period in moves: a split event may not be undone or
    /// sibling-merged until this many moves have passed
    #[arg(long, default_value_t = 2_000)]
    pub(super) tabu_moves: u64,
    /// Expansion move weights
    #[arg(long, default_value_t = 0.70)]
    pub(super) w_cross: f64,
    /// SUSPENDED by default (covered by the twists' case-splitting); set > 0
    /// to re-enable
    #[arg(long, default_value_t = 0.0)]
    pub(super) w_fresh: f64,
    #[arg(long, default_value_t = 0.10)]
    pub(super) w_unsub: f64,
    #[arg(long, default_value_t = 0.05)]
    pub(super) w_insert: f64,
    /// Conjugation-twist weights (0 = off, trajectory-identical to the
    /// pre-twist chain). One twist conjugates a whole window (log-uniform
    /// length up to the circuit size) by a wire negation (+2 gates) or a wire
    /// swap (+6 gates), so keep these SMALL relative to the other weights:
    /// ~1e-4 gives a few thousand twists per 10M expansion moves.
    #[arg(long, default_value_t = 0.0)]
    pub(super) w_twist_neg: f64,
    #[arg(long, default_value_t = 0.0)]
    pub(super) w_twist_swap: f64,
    /// Transvection twist: conjugate a window by x_a ^= x_b (one CNOT per
    /// side, +2 gates). Affine and NOT a Hamming isometry — the rung that
    /// breaks avalanche-style distance gauges neg/swap twists provably
    /// preserve. Interior gates reading a case-split on b (count x2, width +1,
    /// K-cap enforced); b is drawn from wires the window never writes, which
    /// caps these windows at the mid scale (~n*ln n gates).
    #[arg(long, default_value_t = 0.0)]
    pub(super) w_twist_cnot: f64,
    /// Probability each swapped wire is negated (swap family). 0.5 = default
    /// (swap 1/4, negate-one 1/2, negate-both 1/4); 0.0 = pure positive swaps
    /// (no polarity flips, but the 3-CNOT brackets are still inserted).
    #[arg(long, default_value_t = 0.5)]
    pub(super) twist_neg_p: f64,
    /// SAMPLED ancestry: trace this many randomly chosen INPUT gates and report,
    /// per traced gate, the set of current gates descended from it (count,
    /// positional coverage and entropy). Fixed K-bit-per-litter cost, so unlike
    /// --ancestors it scales to production circuits. Overrides --ancestors.
    #[arg(long, default_value_t = 0)]
    pub(super) anc_samples: usize,
    /// Tracer-selection seed. 0 (default) makes the tracer set depend only on
    /// (input size, K), so runs are comparable and a resume traces the same
    /// gates; vary for independent replicates.
    #[arg(long, default_value_t = 0)]
    pub(super) anc_sample_seed: u64,
    /// Write a per-gate ancestry sidecar (final order, matching --output): a
    /// header naming the universe (exact m / sampled K + tracer list), then
    /// one ancestor-set line per gate. The cross-RUN counterpart of the state
    /// file: a later run imports it with --anc-in, so a phase boundary stops
    /// resetting the ancestry clock. Needs ancestry armed.
    #[arg(long)]
    pub(super) anc_out: Option<String>,
    /// Import per-gate ancestor lists from a sidecar written by --anc-out and
    /// use them as this run's INITIAL ancestry (fresh runs only; the gate
    /// count must match the input circuit). The file defines the universe, so
    /// --ancestors / --anc-samples must not also be given.
    #[arg(long)]
    pub(super) anc_in: Option<String>,
    /// Minimum twist window length (max is the current circuit size)
    #[arg(long, default_value_t = 64)]
    pub(super) twist_min_len: usize,
    /// LAYER-2 db_mixing preset: sets the db_mixing default block (--twist-g57
    /// --p-twist 0.0005 --db-advance --p-mingen 0.6 --mix-pay-random, COMP
    /// p_mingen 0) unless individually overridden. curated and p_convex are
    /// no longer set here: the 2026-08-03 shipped defaults (curated ON,
    /// p_convex 0.4) already cover them.
    #[arg(long = "db-mixing", alias = "phase-a", default_value_t = false)]
    pub(super) db_mixing: bool,
    /// Pay-random MIX selection: when only larger spellings exist, pick a
    /// uniformly random one instead of a minimal one (layer-2 db_mixing).
    #[arg(long, default_value_t = false)]
    pub(super) mix_pay_random: bool,
    /// LAYER-2 size profile "N0,N1,N2,R1,R2": three-phase best-effort size
    /// schedule in effective-work (moves/gate) units — expand to R1*input by
    /// N0, hold to N1, compress toward R2*input by N2. The controller reads
    /// the live monitors and steers --p-mix; it is then the ONLY size
    /// authority, so passing any of --target-size/--size-hi/--size-lo
    /// alongside --profile is an error. Empty = off.
    #[arg(long, default_value = "")]
    pub(super) profile: String,
    /// Profile controller: control cadence in effective-work units.
    #[arg(long, default_value_t = 0.5)]
    pub(super) prof_cadence_eff: f64,
    /// Profile controller: relative size deadband (no lever change within).
    #[arg(long, default_value_t = 0.02)]
    pub(super) prof_deadband: f64,
    /// Profile controller: max |Δp_mix| per update (rate limit).
    #[arg(long, default_value_t = 0.1)]
    pub(super) prof_dp_max: f64,
    /// Profile controller: EWMA weight for fresh plant (ghat/shat) estimates.
    #[arg(long, default_value_t = 0.3)]
    pub(super) prof_ewma: f64,
    /// Profile controller: integral gain on the size tracking error.
    #[arg(long, default_value_t = 0.05)]
    pub(super) prof_ki: f64,
    /// Spell twist brackets as all-g57 words sited adaptively so they absorb
    /// neighborhood gates (hidden-SAMF style), instead of 3-CNOT packets.
    /// Pure swap only (twist_neg_p is ignored on this path); every inserted
    /// gate takes the ballistic birth-advance unconditionally.
    #[arg(long, default_value_t = false)]
    pub(super) twist_g57: bool,
    /// Probability a CONTRACTION tries COMP-DB (non-growing, uniform among the
    /// shortest) before falling through to journal undo and the merge
    /// catalogue. Requires FROZEN_DB_DIR.
    #[arg(long, default_value_t = 1.0)]
    pub(super) p_comp: f64,
    /// Probability an EXPANSION is an ANY-DB move rather than a cross.
    #[arg(long, default_value_t = 0.1)]
    pub(super) p_any: f64,
    /// Slot-2 admission rule: mix | comp | any. MIX is free-if-possible else
    /// pay-the-minimum and is the db_mixing default; COMP refuses to grow and is
    /// the manual size brake; ANY takes any equivalent and accelerates growth.
    #[arg(long, default_value = "mix")]
    pub(super) db_mode: String,
    /// Size-agnostic DB move: probability that a whole round is spent replacing a
    /// sampled window with a uniform random equivalent of ANY gate count (may
    /// grow the circuit), instead of the normal contract/expand step. 0 = off.
    #[arg(long, default_value_t = 0.0)]
    pub(super) p_db: f64,
    /// Fixed top-level twist rate: each round is a conjugation twist with this
    /// probability, independent of the contract/expand economy (whose round
    /// supply collapses when the size controller holds at target). Twist type
    /// follows the w-twist-* weights as ratios (neg/swap 50/50 when all 0);
    /// with this set, the expansion mix no longer performs twists. Size
    /// machinery balances around the fixed rate. Rough sizing: coverage per
    /// move ~= p-twist x mean-window-span / size.
    #[arg(long, default_value_t = 0.0)]
    pub(super) p_twist: f64,
    /// Arm the SPLIT STAGE (docs/FMIX_SPLIT_TWIST.md): split twists — a g57
    /// split plus an absorbed long-range pure-NOT twist plus one cross — are
    /// the ONLY move until the stage exits (g57 exhaustion, or
    /// --split-fail-limit consecutive bracket failures), then the round runs
    /// under the rest of this command line as usual.
    #[arg(long, default_value_t = false)]
    pub(super) split: bool,
    /// End the RUN at the split-stage boundary (trial mode) instead of
    /// continuing into part 2.
    #[arg(long, default_value_t = false)]
    pub(super) split_stop: bool,
    /// Probability a split carries the absorbed NOT twist + cross; the rest
    /// end after the bare split.
    #[arg(long, default_value_t = 0.8)]
    pub(super) p_join: f64,
    /// Consecutive bracket-search failures (step 4e) that end the stage.
    #[arg(long, default_value_t = 100)]
    pub(super) split_fail_limit: u32,
    /// Wire canaries planted at split-stage start (0 = off): per-position flip
    /// monitors, reported by ORIGINAL position at the stage boundary.
    #[arg(long, default_value_t = 256)]
    pub(super) split_canaries: usize,
    /// Length bias of the bracket draw: k candidates sampled on the picked
    /// g57's own side (its stored direction), farthest wins. 1 = uniform,
    /// 2 ~ 2/3 of the available run, 3 ~ 3/4; larger = longer spans.
    /// 0 = the ORIGINAL other-half-first cascade (A/B comparison arm).
    #[arg(long, default_value_t = 2)]
    pub(super) split_reach_k: usize,
    /// Layer-1 dispatch weight for split twists inside the twist slot OUTSIDE
    /// the split stage (the stage itself forces 1.0).
    #[arg(long, default_value_t = 0.0)]
    pub(super) p_split_twist: f64,
    /// Min-dgen cross-shot bias: probability a cross shot is drawn from the
    /// pool of the K least-split lineages instead of uniformly (the uniform
    /// draw is rich-get-richer; this points expansion at untouched
    /// families). 0 = off, and off draws no RNG — trajectories identical.
    #[arg(long, default_value_t = 0.0)]
    pub(super) p_mincross: f64,
    /// Min-dgen pool size (a COUNT): must exceed the biased draws taken
    /// between rebuilds or the coin silently degrades to uniform.
    #[arg(long, default_value_t = 20_000)]
    pub(super) cross_pool_k: usize,
    /// Min-dgen pool rebuild cadence in moves (O(size) scan each).
    #[arg(long, default_value_t = 10_000)]
    pub(super) cross_rescan: u64,
    /// GLOBAL re-randomisation rate, in units of ONE whole-circuit reshuffle
    /// per this many circuit-sizes of rounds. The per-round probability is
    /// shuffle-rate / |circuit|, so e.g. 2.0 means "expect two full
    /// reshuffles every |circuit| rounds" at any size, and the expected work
    /// per round stays O(mean commutation slack) as the circuit grows. The
    /// move walks every gate in order and re-places it uniformly inside its
    /// own commutation bounds (the same placement rule as the terminal
    /// float), so it is semantics- and size-preserving: it only moves gates.
    /// OFF by default since 2026-08-03 (was 2.0); note 0 vs >0 reshapes the
    /// walk-RNG stream at equal seed, so it is an A/B arm, not a live toggle.
    #[arg(long, default_value_t = 0.0)]
    pub(super) shuffle_rate: f64,
    /// Two-pass store routing, ON BY DEFAULT (2026-08-03): the descent runs
    /// curated-only over every window length first, and consults the regular
    /// store only if that whole pass came up empty. (The old one-pass cascade
    /// probed curated then regular at EACH length before shortening, so a
    /// regular hit at length p beat a curated hit at length p-1.) Needs
    /// --curated; applies to compression too while --curated-in-comp is on
    /// (also the default). Disable with --no-curated-exhaust.
    #[arg(long, default_value_t = true)]
    pub(super) curated_exhaust: bool,
    /// Turn --curated-exhaust off (single-pass curated-then-regular at each
    /// window length).
    #[arg(long, default_value_t = false)]
    pub(super) no_curated_exhaust: bool,
    /// Arm the curated store for COMPRESSION as well, ON BY DEFAULT
    /// (2026-08-03): COMP first walks the whole cascade against curated, then
    /// against regular, same as MIX. The old off-by-contract rationale: an
    /// uneven split of a minimal identity gives two functionally inverse
    /// circuits of unequal length, so the store holds longer-than-minimal
    /// spellings, and curated's lexicographic priority in the selection rule
    /// compounds the bias toward growth. Under compression the size rule keeps
    /// only the spellings strictly shorter than the window -- the shorter
    /// halves. Needs --curated. Disable with --no-curated-in-comp.
    #[arg(long, default_value_t = true)]
    pub(super) curated_in_comp: bool,
    /// Turn --curated-in-comp off (compression goes regular-only, the old
    /// contract).
    #[arg(long, default_value_t = false)]
    pub(super) no_curated_in_comp: bool,
    /// Window length the descent STARTS from, in MIX mode (COMP has its own,
    /// --s-db-comp). The descent visits every shorter length down to 1, so
    /// there is no separate minimum.
    #[arg(long, default_value_t = 9)]
    pub(super) s_db: usize,
    /// Minimum DB window length (0 = off). Use with --no-db-prefixes so the
    /// descent does not visit shorter prefixes anyway.
    #[arg(long, default_value_t = 0)]
    pub(super) db_min_window: usize,
    /// Probability the window sampler is convex rather than contiguous, in MIX
    /// mode (COMP: --p-convex-comp). Default 0.4 = contiguous 60% / convex 40%.
    #[arg(long, default_value_t = 0.4)]
    pub(super) p_convex: f64,
    /// A gate with this many controls or more may not sit INSIDE a window.
    #[arg(long, default_value_t = 4)]
    pub(super) w_window: usize,
    /// A gate with this many controls or more may not SEED a window or count
    /// toward the dose. Stricter than --w-window on purpose: width-3 gates
    /// match in context but re-encode end-to-end at 0.41% against 98.98% for
    /// width <= 2, so at a shared threshold they pile up in the pool forever.
    #[arg(long, default_value_t = 3)]
    pub(super) w_pool: usize,
    /// Convex sampling: probability each growth step floats the block in g1's
    /// direction (else the opposite).
    #[arg(long, default_value_t = 0.75)]
    pub(super) db_convex_p: f64,
    /// Skip the exhaustive per-splice equivalence check on DB replacements
    /// (faster long runs; the periodic global check still guards correctness).
    /// With this set, DB windows wider than 24 wires can also be replaced.
    #[arg(long, default_value_t = false)]
    pub(super) no_db_verify: bool,
    /// Record every DB replacement attempt to this file: the outgoing window, the
    /// number of equivalent DB circuits, and (on success) the replacing circuit.
    #[arg(long)]
    pub(super) db_record: Option<String>,
    /// Measurement mode: sample windows and record DB match counts (--db-record)
    /// but never splice, so the circuit stays fixed. With --p-db 1.0 and all
    /// other move weights 0 this makes fmix a pure match-rate sampler.
    #[arg(long, default_value_t = false)]
    pub(super) db_dry_run: bool,
    /// Degree cap for the DB lookup: a window whose function degree exceeds this
    /// cannot match any stored circuit and is skipped before canonicalization
    /// (the main speed guard on high-width windows). Set to the DB's max ANF
    /// degree; must be <= 11 (the probe tests (cap+1)-dim subcubes and maxes
    /// out at 12). 0 = off (every window canonicalizes). Degree-skipped
    /// attempts are still recorded by --db-record, so measurement runs can
    /// (and should) keep the cap on.
    #[arg(long, default_value_t = 0)]
    pub(super) db_max_degree: usize,
    /// Random subcubes probed per direction by the degree cap (higher = fewer
    /// missed high-degree windows, at proportional cost).
    #[arg(long, default_value_t = 6)]
    pub(super) db_degree_probes: usize,
    /// Span cap for the DB lookup: a window touching more distinct wires than
    /// this is recorded as a miss without canonicalizing. Set to the store's
    /// max canonical support (measure with frozen_degree_scan). This is the
    /// main speed guard: Rule-L canonicalization cost explodes with tied wire
    /// count, and wide-span windows can't match anyway. 0 = off.
    #[arg(long, default_value_t = 0)]
    pub(super) db_max_span: usize,
    /// Per-wire polynomial term cap for the DB lookup budget: a window whose
    /// wire poly outgrows the largest any stored function has cannot match,
    /// and the budget bail lands before the expensive Rule-L canonicalization.
    /// Set to the store's max (frozen_degree_scan census). 0 = default 2^18.
    #[arg(long, default_value_t = 0)]
    pub(super) db_wire_terms: usize,
    /// Total-terms cap across a window's wire polys (census: per-entry total).
    /// 0 = default 2^20.
    #[arg(long, default_value_t = 0)]
    pub(super) db_total_terms: usize,
    /// Largest-first prefix descent (the size-reduction cascade), ON BY
    /// DEFAULT (2026-08-03): try the full sampled window, then its len-1
    /// prefix, etc. down to length 1, splicing the LONGEST matching prefix
    /// (live) or recording every prefix (with --db-dry-run). Span/verify
    /// declines keep descending — shorter prefixes may still fit. Disable
    /// with --no-db-prefixes.
    #[arg(long, default_value_t = true)]
    pub(super) db_prefixes: bool,
    /// Turn --db-prefixes off (one attempt at a uniformly sampled length, no
    /// descent).
    #[arg(long, default_value_t = false)]
    pub(super) no_db_prefixes: bool,
    /// Give DB splice products the ballistic birth-advance that split pieces
    /// get: each product floats floor(dir_q * slack) along its own direction.
    /// Without this a splice assigns directions nothing acts on, so a
    /// DB-dominated schedule has no directional transport at all. The other
    /// source of transport -- crossings -- widens gates, and width is what
    /// kills DB matching. Off by default (it changes trajectories).
    #[arg(long, default_value_t = false)]
    pub(super) db_advance: bool,
    /// Pair-geometry rate (docs/NONLOCAL_PHASE_A.md): probability a non-COMP
    /// DB round samples its window as the seed plus one FAR COMMUTING partner,
    /// floated adjacent and fused into a 2-gate window — the db_mixing transport
    /// experiment. The fused splice unions litters across the seed's whole
    /// commutation box, and the reorder ban forces every pair splice onto a
    /// genuinely different spelling. 0 = off, draws no RNG.
    #[arg(long, default_value_t = 0.0)]
    pub(super) p_pair: f64,
    /// Pair geometry: cap on the commutation-box scan past the seed.
    #[arg(long, default_value_t = 4096)]
    pub(super) pair_scan_cap: usize,
    /// Pair geometry: pick the partner uniformly from the eligible box instead
    /// of the farthest gate.
    #[arg(long, default_value_t = false)]
    pub(super) pair_pick_uniform: bool,
    /// Bridge-fusion rate (docs/NONLOCAL_PHASE_A.md): per-round probability of
    /// jointly re-encoding two gates that commutation CANNOT bring together —
    /// a carrier conjugates the interior (wake corrections on interior
    /// colliders, non-g57: trades polf for reach like legacy twist packets)
    /// and both carrier-adjacent windows are re-spelled through the store.
    /// 0 = off, draws no RNG.
    #[arg(long, default_value_t = 0.0)]
    pub(super) p_bridge: f64,
    /// Bridge: log-uniform interior-length draw, lower bound.
    #[arg(long, default_value_t = 16)]
    pub(super) bridge_min_span: usize,
    /// Bridge: log-uniform interior-length draw, upper bound.
    #[arg(long, default_value_t = 512)]
    pub(super) bridge_max_span: usize,
    /// Bridge: refuse rounds whose interior holds more colliders than this
    /// (each costs one or two wake correction gates).
    #[arg(long, default_value_t = 8)]
    pub(super) bridge_max_colliders: usize,
    /// Layer-2 mode overlay (slot 0): per-round probability the slot-2 DB move
    /// is MIX-DB, else COMP-DB. Each round flips this coin, overriding --db-mode
    /// and reading the chosen mode's own knobs -- MIX uses --s-db / --p-convex /
    /// --p-mingen, COMP uses the *-comp overrides below. < 0 disables the
    /// overlay (single --db-mode, as before). The thermostat is unaffected, so
    /// pair with --p-db 1.0 for a pure per-round MIX/COMP schedule.
    #[arg(long, default_value_t = -1.0)]
    pub(super) p_mix: f64,
    // The five knobs below are Option ON PURPOSE and carry no clap default.
    // They are OVERRIDES, and clap's `Option` is what records "the user asked
    // for this" -- which is exactly the distinction a sentinel default cannot
    // make. Their shipped values live in `DbLayer::shipped()`, one layer down,
    // so an explicit --s-db now beats a merely-defaulted --s-db-comp instead of
    // being silently shadowed by it. See §2.1.2 of the manual.
    //
    /// COMP-mode window length. Shipped default 12: COMP starts its descent
    /// higher than MIX's 9.
    #[arg(long)]
    pub(super) s_db_comp: Option<usize>,
    /// COMP-mode convex probability. Shipped default 0.9 = convex 90% /
    /// contiguous 10%. Convex wins compression on every axis measured (16x
    /// gates removed, 7x ancestry transport, 31x less CPU -- wide contiguous
    /// windows cost ~30x per canonicalization even when they pass the span
    /// cap), so COMP leans hard convex.
    #[arg(long)]
    pub(super) p_convex_comp: Option<f64>,
    /// COMP-mode pool-seed probability under --p-mix. Unset = use --p-mingen.
    #[arg(long)]
    pub(super) p_mingen_comp: Option<f64>,
    /// MIX-mode window length when the round draws a CONTIGUOUS window.
    /// Unset = share --s-db across both geometries. A contiguous window of the
    /// same gate count spans far more wires than a convex one and costs 3.6x
    /// its canonicalization at length 5, 12.6x at 7, 47.8x at 12 -- so a
    /// profile that wants a wide convex probe usually wants a narrow
    /// contiguous one.
    #[arg(long)]
    pub(super) s_db_ctg: Option<usize>,
    /// COMP-mode window length when the round draws a CONTIGUOUS window.
    /// Unset = use --s-db-comp.
    #[arg(long)]
    pub(super) s_db_comp_ctg: Option<usize>,
    /// Prefix descent in MIX rounds only (unset = use --db-prefixes). Under the
    /// --p-mix overlay both modes run in one process and want opposite
    /// settings, so this splits the global flag per mode.
    #[arg(long)]
    pub(super) db_prefixes_mix: Option<bool>,
    /// Prefix descent in COMP rounds only (unset = use --db-prefixes).
    #[arg(long)]
    pub(super) db_prefixes_comp: Option<bool>,
    /// GSS profile: the DB settings for running fmix on a gadgetized sliced
    /// sandwich. Curated on; COMP = descent on, p_mingen 0, convex 95% at
    /// s_db 12 / contiguous 5% at s_db 6; MIX = descent off, p_mingen 0.5,
    /// convex 50% / contiguous 50%, s_db 6 for both. Explicit flags win.
    ///
    /// Deliberately does NOT set --p-mix: the MIX/COMP balance is the layer-2
    /// controller's lever, and this profile is meant to be the right setting
    /// at every p_mix. Also deliberately g57-PRESERVING -- every DB splice
    /// re-spells a g57 word as another g57 word (polf stays 0), because
    /// breaking g57 form is a separate concern from this profile's job.
    #[arg(long, default_value_t = false)]
    pub(super) gss: bool,
    /// Also probe the curated store (FROZEN_CURATED_DIR) and prefer a
    /// non-identical curated match over a regular one REGARDLESS OF SIZE. The
    /// curated store holds circuits every strict subcircuit of which is
    /// shortest, so its replacements are routes fcompress cannot partially
    /// undo -- but it is built from splits of minimal identities and so holds
    /// the longer halves of uneven identity splits, and curated's lexicographic priority compounds it.
    /// Compression probes it too while --curated-in-comp is on (the default).
    /// ON BY DEFAULT (2026-08-03). If FROZEN_CURATED_DIR is unset the default
    /// degrades to regular-only WITH A WARNING; passing --curated explicitly
    /// makes the missing store a hard error instead. Disable with
    /// --no-curated.
    #[arg(long, default_value_t = true)]
    pub(super) curated: bool,
    /// Turn --curated off (regular store only, both modes).
    #[arg(long, default_value_t = false)]
    pub(super) no_curated: bool,
    /// Track, per litter, the SET of original input gates that contributed to
    /// it (the union of the sets its outgoing window drew on). Reports anc=
    /// (mean set size: what a mixed gate is made of) and ancspan= (mean
    /// normalised index span: how far input material travels to meet). Immune
    /// to the ORIGIN_SYNTH erosion that makes odiff/oadj unreadable. Costs
    /// |input| bits per live litter, so SMALL INPUTS ONLY -- it refuses above
    /// 20k input gates.
    #[arg(long, default_value_t = false)]
    pub(super) ancestors: bool,
    /// Probability a COMP-DB attempt restricts itself to PURE g57 material and
    /// starts its descent at --s-db-g57. Only g57-only windows survive length:
    /// measured match rate is 100% to m=5 and 94% at 6, but ANY non-g57 gate in
    /// a 6-gate window drops it to <=7%, so the long-window compression that
    /// pays is unavailable except on pure windows.
    #[arg(long, default_value_t = 0.0)]
    pub(super) p_comp_g57: f64,
    /// Upper clamp on the contraction probability. 0.98 leaves a 2% expansion
    /// floor above target -- a structural growth source, but also what keeps
    /// crossings (hence fossil erosion) running while the walk sits at target.
    #[arg(long, default_value_t = 0.98)]
    pub(super) contract_ceiling: f64,
    /// Size brake: growth to this size forces slot 2 into COMP. 0 = brake off.
    #[arg(long, default_value_t = 0)]
    pub(super) size_hi: usize,
    /// Release the brake at this size, or earlier if COMP stops paying.
    #[arg(long, default_value_t = 0)]
    pub(super) size_lo: usize,
    /// Release the brake when COMP sheds fewer than this many gates per round
    /// over the trailing window. This is what makes a WIDE band safe: the risk
    /// was never band width but sitting in COMP past its usefulness.
    #[arg(long, default_value_t = 0.0)]
    pub(super) comp_release_eps: f64,
    /// Trailing window (moves) for the productivity release.
    #[arg(long, default_value_t = 250_000)]
    pub(super) comp_release_window: u64,
    /// Starting window length for a g57-only COMP attempt.
    #[arg(long, default_value_t = 9)]
    pub(super) s_db_g57: usize,
    /// Frozen store directory. Overrides FROZEN_DB_DIR, which is env-only and
    /// in no rc file -- detached runs that miss it abort instantly.
    #[arg(long)]
    pub(super) frozen_db_dir: Option<String>,
    /// Curated store directory. Overrides FROZEN_CURATED_DIR.
    #[arg(long)]
    pub(super) frozen_curated_dir: Option<String>,
    /// Generation targeting: drive every (ctrl-cap-eligible) gate through at
    /// least this many DB re-encodings. Each gate carries a generation (input
    /// gates 0; a DB splice stamps min(window)+1; splits/merges propagate;
    /// fresh insert/bracket material counts as done). DB seeds are drawn from
    /// the below-target gates with probability --gen-bias, replacing the
    /// coupon-collector tail of uniform selection with direct work. 0 = off.
    #[arg(long, default_value_t = 0)]
    pub(super) gen_target: u32,
    /// Probability a DB seed comes from the generation POOL rather than
    /// uniformly. This is what stops the walk being a coupon collector, where
    /// the last few percent of gates soak up most of the moves.
    #[arg(long, default_value_t = 0.8)]
    pub(super) p_mingen: f64,
    /// Pool size in GATES: the K lowest-generation gates that are pool-eligible
    /// and still below the goal. A count rather than a fraction, because the
    /// drain rate is set by the move economy (gen_rescan x p_db x p_mingen) and
    /// is independent of circuit size. K must exceed the draws taken between
    /// rebuilds or the pool empties and the biased coin silently degrades to
    /// uniform -- watch the fall-through counter.
    #[arg(long, default_value_t = 20_000)]
    pub(super) pool_k: usize,
    /// Stop when the failure fraction over the last --canary-window QUALIFYING
    /// rounds exceeds this. A qualifying round is one whose seed genuinely came
    /// from the pool; heads coins that fell through a drained pool are counted
    /// separately, since those mean the rebuild is too slow rather than that the
    /// material is unreachable. Asleep while the brake holds COMP, because COMP
    /// declines far more often by construction. 0 = off.
    #[arg(long, default_value_t = 0.0)]
    pub(super) canary_theta: f64,
    /// Trailing window for the canary, in qualifying rounds.
    #[arg(long, default_value_t = 2000)]
    pub(super) canary_window: usize,
    /// Refuse a descent rung that is exactly one COMPLETE litter -- the set some
    /// earlier replacement emitted, and so where the store is most likely to
    /// hand that spelling straight back. Singleton litters are exempt: an input
    /// gate has no earlier spelling, and banning it would also refuse the
    /// length-1 rung, the one that always makes progress.
    #[arg(long, default_value_t = false)]
    pub(super) litter_ban: bool,
    /// Draw this many candidate windows and keep the one spanning the most
    /// distinct litters. 1 = off.
    #[arg(long, default_value_t = 1)]
    pub(super) litter_samples: usize,
    /// Candidate positions the twist placer samples looking for a welcoming
    /// neighbourhood (a gate that can absorb the bracket) before giving up and
    /// placing the twist uniformly at random. 0 = always random.
    #[arg(long, default_value_t = 0)]
    pub(super) twist_place_tries: usize,
    /// Split-rule variant: children INHERIT the parent generation unchanged
    /// (only DB replacements raise generations). Default off = ratchet
    /// semantics (split children get parent + 1).
    #[arg(long, default_value_t = false)]
    pub(super) gen_split_inherit: bool,
    /// Median variant for the DB generation stamp: use the LOWER median
    /// (rounded down on even windows; on 2-gate windows this is the min).
    /// Default off = upper median (rounded up).
    #[arg(long, default_value_t = false)]
    pub(super) gen_median_low: bool,
    /// Laggard-list rebuild cadence in moves (O(size) scan each rebuild;
    /// entries going stale between rebuilds are pruned at draw time).
    #[arg(long, default_value_t = 10_000)]
    pub(super) gen_rescan: u64,
    /// Dose-based stop: end the run (before the move budget) at the first
    /// report point where the below-target fraction among eligible gates is
    /// <= this AND --twist-cov-stop is met. The db_mixing "minimal growth"
    /// switch: spend exactly the moves the dose requires, no more. Negative =
    /// off. Requires --gen-target > 0.
    #[arg(long, default_value_t = -1.0)]
    pub(super) gen_stop_frac: f64,
    /// Twist-coverage requirement for the dose stop: cumulative twisted span
    /// over current size (per-position coverage; saturation target ~600).
    /// 0 = no coverage requirement.
    #[arg(long, default_value_t = 0.0)]
    pub(super) twist_cov_stop: f64,
    /// Write per-gate DB-generation stamps (final order, one per line;
    /// 4294967295 = born-random material) for dose analysis
    #[arg(long)]
    pub(super) gens_out: Option<String>,
    /// Verified snapshot each time the circuit generation (G=, the
    /// 5th-percentile gate generation) crosses a multiple of this value:
    /// <output>.gen<m>.mpmct1 + .gens sidecar. 0 = off. Meaningful only with
    /// --gen-target > 0 (generations move under DB re-encoding).
    #[arg(long, default_value_t = 0)]
    pub(super) gen_snap_every: u32,
    /// Verified snapshot at every multiple of this many moves
    /// (<output>.mv<moves>.mpmct1 + .gens sidecar): the progress clock for
    /// pure-split runs where the generation census is not meaningful. Use a
    /// multiple of --report-every. 0 = off.
    #[arg(long, default_value_t = 0)]
    pub(super) snap_every_moves: u64,
    /// Global sampled equality check every N moves
    #[arg(long, default_value_t = 10_000)]
    pub(super) verify_every: u64,
    /// Progress report (and stop/dump flag check) every N moves
    #[arg(long, default_value_t = 50_000)]
    pub(super) report_every: u64,
    /// Disable the per-move exhaustive local verification
    #[arg(long, default_value_t = false)]
    pub(super) no_local_verify: bool,
    /// Skip the final uniform float pass
    #[arg(long, default_value_t = false)]
    pub(super) skip_final_float: bool,
    /// Write per-gate origin indices (final order, one per line; 4294967295 =
    /// synthetic) for dispersion analysis
    #[arg(long)]
    pub(super) origins_out: Option<String>,
    #[arg(long, default_value_t = 0)]
    pub(super) seed: u64,
    /// Write a resume file on every stop (and at each snapshot). Holds what the
    /// circuit file cannot: per-gate direction/generation/litter/event, the undo
    /// journal, the ORIGINAL circuit that global_check verifies against, and the
    /// condition state (moves, twist coverage, canary ring, brake, pool).
    #[arg(long)]
    pub(super) state_out: Option<String>,
    /// Per-gate litter-id sidecar for the INPUT circuit (header "litter1 N",
    /// one id per gate line) — e.g. written by sgdb_substitute so each
    /// substituted block counts as a litter for --litter-ban.
    #[arg(long)]
    pub(super) litter_in: Option<String>,
    /// Resume from a state file instead of --input. Parameters still come from
    /// the command line, so a paused run can be re-steered; only the state
    /// VERSION must match, since field meanings would otherwise drift silently.
    #[arg(long)]
    pub(super) resume: Option<String>,
    /// Piecewise-parallel rounds (docs/FMIX_PIECEWISE.md): cut the circuit
    /// into this many contiguous pieces, mix them in parallel on one shared
    /// store, concatenate, shift the cuts by half a slice, repeat. 1 = the
    /// serial run unless --parallel-target-piece-gates is selected. Stages 3 and
    /// 4 only (refused with --resume).
    #[arg(long = "parallel-pieces", alias = "pieces", default_value_t = 1)]
    pub(super) pieces: usize,
    /// Automatic piecewise rounds: nominal piece count is
    /// max(1, floor(current gates / B)), recalculated every round.
    /// B only determines the count; shifted blocks may be smaller.
    /// Minimum must be >= 2. Select this option alone, without --parallel-pieces or
    /// --piece-min-len. Small circuits start with one block.
    #[arg(long = "parallel-target-piece-gates", alias = "min-block-size", value_name = "B", value_parser = parse_min_block_size,
        conflicts_with_all = ["pieces", "piece_min_len"])]
    pub(super) min_block_size: Option<usize>,
    /// Cut-point jitter as a fraction of the local seam interval, in
    /// [0, 0.25): every old seam stays >= (0.5 - jitter) of its interval away
    /// from every new cut.
    #[arg(long, default_value_t = 0.125)]
    pub(super) piece_jitter: f64,
    /// Round length in eff units (moves per gate). 0 = the profile
    /// controller's cadence under --profile (identical controller
    /// granularity), 0.5 otherwise.
    #[arg(long, default_value_t = 0.0)]
    pub(super) piece_round_eff: f64,
    /// Fixed-mode shortest piece the cuts may produce (checked against the input size at
    /// startup; a violation is an error, never a silent change of --parallel-pieces).
    #[arg(long, default_value_t = 1024)]
    pub(super) piece_min_len: usize,
    /// Threads in the piece pool; 0 = available parallelism for --parallel-target-piece-gates,
    /// or pieces + 1 in fixed mode (a shifted round has one more, half-length, piece).
    #[arg(
        long = "parallel-threads",
        alias = "piece-threads",
        default_value_t = 0
    )]
    pub(super) piece_threads: usize,
    /// --split in piecewise mode: rounds after which the stage is declared ended
    /// even if g57s remain (reported, never dropped).
    #[arg(long, default_value_t = 6)]
    pub(super) split_rounds_max: usize,
    /// Let piece mixers print their own report lines (they are silent by
    /// default; the whole-circuit mixer reports once per round).
    #[arg(long, default_value_t = false)]
    pub(super) piece_verbose: bool,
}

fn parse_min_block_size(value: &str) -> Result<usize, String> {
    let size = value
        .parse::<usize>()
        .map_err(|_| "minimum block size must be an integer >= 2".to_string())?;
    if size < 2 {
        return Err("minimum block size must be an integer >= 2".to_string());
    }
    Ok(size)
}

impl Args {
    pub(super) fn piecewise_enabled(&self) -> bool {
        self.pieces > 1 || self.min_block_size.is_some()
    }
}

/// One layer of DB-knob opinions. `None` = "this layer says nothing here".
///
/// The point of making layers explicit is that a preset used to *mutate* the
/// args struct, after which nothing downstream could tell "the user asked for
/// 12" from "--gss set 12" -- so precedence was decided by mutation order and
/// was invisible in the code. Here it is data: `cli.over(preset).over(shipped)`
/// reads in precedence order, and the banner prints what came out.
/// Which BASE knobs the user typed. Used only to withhold shipped
/// mode-level defaults that would otherwise outrank them on specificity.
#[derive(Default, Clone, Copy)]
pub(super) struct BaseGiven {
    pub(super) s_db: bool,
    pub(super) p_convex: bool,
}

#[derive(Default, Clone, Copy)]
pub(super) struct DbLayer {
    pub(super) s_db: Option<usize>,
    pub(super) s_db_ctg: Option<usize>,
    pub(super) s_db_comp: Option<usize>,
    pub(super) s_db_comp_ctg: Option<usize>,
    pub(super) p_convex: Option<f64>,
    pub(super) p_convex_comp: Option<f64>,
    pub(super) p_mingen: Option<f64>,
    pub(super) p_mingen_comp: Option<f64>,
    pub(super) prefixes: Option<bool>,
    pub(super) prefixes_mix: Option<bool>,
    pub(super) prefixes_comp: Option<bool>,
}

impl DbLayer {
    /// `self` wins wherever it has an opinion; `under` fills the rest.
    pub(super) fn over(self, under: DbLayer) -> DbLayer {
        DbLayer {
            s_db: self.s_db.or(under.s_db),
            s_db_ctg: self.s_db_ctg.or(under.s_db_ctg),
            s_db_comp: self.s_db_comp.or(under.s_db_comp),
            s_db_comp_ctg: self.s_db_comp_ctg.or(under.s_db_comp_ctg),
            p_convex: self.p_convex.or(under.p_convex),
            p_convex_comp: self.p_convex_comp.or(under.p_convex_comp),
            p_mingen: self.p_mingen.or(under.p_mingen),
            p_mingen_comp: self.p_mingen_comp.or(under.p_mingen_comp),
            prefixes: self.prefixes.or(under.prefixes),
            prefixes_mix: self.prefixes_mix.or(under.prefixes_mix),
            prefixes_comp: self.prefixes_comp.or(under.prefixes_comp),
        }
    }

    /// Bottom layer: what fmix ships when the user says nothing at all. These
    /// are the values that used to sit in `default_value_t`.
    ///
    /// `base_given` names the base knobs the user passed explicitly, and this
    /// layer WITHHOLDS its mode-level opinion for each of them. That single
    /// rule is the whole bug fix: `--db-mode comp --s-db 20` used to run at 12
    /// because a defaulted `s_db_comp` outranked an explicit `--s-db` on
    /// specificity. A shipped default is not a statement about *this* run, so
    /// it must not outrank one.
    ///
    /// Note the asymmetry with presets, which is deliberate: `--gss` DOES keep
    /// its COMP settings when you also pass `--s-db`. A named profile is a
    /// coherent unit and its mode-level choices are intentional; if you mean
    /// to move COMP too, say `--s-db-comp`.
    pub(super) fn shipped(base_given: BaseGiven) -> DbLayer {
        DbLayer {
            s_db: Some(9),
            p_convex: Some(0.4),
            p_mingen: Some(0.8),
            prefixes: Some(true),
            s_db_comp: (!base_given.s_db).then_some(12),
            p_convex_comp: (!base_given.p_convex).then_some(0.9),
            ..DbLayer::default()
        }
    }

    /// The GSS profile: the DB block for a gadgetized-sliced-sandwich input.
    /// Deliberately silent on p_mix -- that is layer 2's lever.
    pub(super) fn gss() -> DbLayer {
        DbLayer {
            prefixes_comp: Some(true),
            p_mingen_comp: Some(0.0),
            p_convex_comp: Some(0.95),
            s_db_comp: Some(12),
            s_db_comp_ctg: Some(6),
            prefixes_mix: Some(false),
            p_mingen: Some(0.5),
            p_convex: Some(0.5),
            s_db: Some(6),
            ..DbLayer::default()
        }
    }

    /// db_mixing's DB opinions (the twist/advance block stays in the preset
    /// below; only DB knobs belong here).
    pub(super) fn db_mixing() -> DbLayer {
        DbLayer {
            p_mingen: Some(0.6),
            p_mingen_comp: Some(0.0),
            ..DbLayer::default()
        }
    }
}

/// Independent end-of-mixing QC; all defaults are inert without --leakage-repair.
#[derive(clap::Args, Debug)]
pub(super) struct QualityArgs {
    /// Audit hot wire segments and attempt surgical DB repairs before saving.
    #[arg(long = "leakage-repair", alias = "qc")]
    pub(super) qc: bool,
    /// Original reference circuit in the SAME input/wire coordinates (default: walk input).
    #[arg(
        long = "leakage-repair-reference",
        alias = "qc-reference",
        requires = "qc"
    )]
    pub(super) qc_reference: Option<String>,
    #[arg(long = "leakage-repair-reference-format", alias = "qc-reference-format", default_value = "mpmct1", value_parser = ["mpmct1", "g57"])]
    pub(super) qc_reference_format: String,
    /// Detailed report path (default: <output>.qc.txt).
    #[arg(long = "leakage-repair-report", alias = "qc-report", requires = "qc")]
    pub(super) qc_report: Option<String>,
    /// Independent seed for QC input probes and DB scratch-wire assignments.
    #[arg(
        long = "leakage-repair-seed",
        alias = "qc-seed",
        default_value_t = 0x5143_u64
    )]
    pub(super) qc_seed: u64,
    #[arg(
        long = "leakage-repair-max-attempts",
        alias = "qc-max-attempts",
        default_value_t = 32
    )]
    pub(super) qc_max_attempts: usize,
    #[arg(
        long = "leakage-repair-max-repairs",
        alias = "qc-max-repairs",
        default_value_t = 8
    )]
    pub(super) qc_max_repairs: usize,
    #[arg(
        long = "leakage-repair-max-block-gates",
        alias = "qc-max-block-gates",
        default_value_t = 12
    )]
    pub(super) qc_max_block_gates: usize,
    #[arg(
        long = "leakage-repair-max-replacement-gates",
        alias = "qc-max-replacement-gates",
        default_value_t = 24
    )]
    pub(super) qc_max_replacement_gates: usize,
    #[arg(
        long = "leakage-repair-max-support",
        alias = "qc-max-support",
        default_value_t = 24
    )]
    pub(super) qc_max_support: usize,
    #[arg(
        long = "leakage-repair-max-span",
        alias = "qc-max-span",
        default_value_t = 256
    )]
    pub(super) qc_max_span: usize,
    #[arg(
        long = "leakage-repair-max-candidates",
        alias = "qc-max-candidates",
        default_value_t = 64
    )]
    pub(super) qc_max_candidates: usize,
    /// Number of 64-input batches in EACH of the training and held-out partitions.
    #[arg(
        long = "leakage-repair-sample-batches",
        alias = "qc-sample-batches",
        default_value_t = 4
    )]
    pub(super) qc_sample_batches: usize,
    /// Original internal wire features used by affine predictor (0 disables affine).
    #[arg(
        long = "leakage-repair-reference-segments",
        alias = "qc-reference-segments",
        default_value_t = 32
    )]
    pub(super) qc_reference_segments: usize,
    /// Original firing predicates checked for correlation (0 disables firing test).
    #[arg(
        long = "leakage-repair-reference-firings",
        alias = "qc-reference-firings",
        default_value_t = 128
    )]
    pub(super) qc_reference_firings: usize,
    #[arg(
        long = "leakage-repair-scan-segments",
        alias = "qc-scan-segments",
        default_value_t = 2048
    )]
    pub(super) qc_scan_segments: usize,
    /// Minimum absolute Pearson/phi correlation on training AND held-out inputs.
    #[arg(
        long = "leakage-repair-correlation",
        alias = "qc-correlation",
        default_value_t = 0.98
    )]
    pub(super) qc_correlation: f64,
}

impl QualityArgs {
    pub(super) fn config(&self) -> crate::stages::db_mixing::leakage_repair::QualityConfig {
        use crate::stages::db_mixing::leakage_repair::detect::DetectorConfig;
        use crate::stages::db_mixing::leakage_repair::{QualityConfig, blocks::BlockLimits};
        use crate::stages::db_mixing::replacement::QcCandidateLimits;
        QualityConfig {
            detector: DetectorConfig {
                seed: self.qc_seed,
                train_batches: self.qc_sample_batches,
                heldout_batches: self.qc_sample_batches,
                max_original_segments: self.qc_reference_segments,
                max_original_firings: self.qc_reference_firings,
                max_mixed_segments: self.qc_scan_segments,
                min_abs_correlation: self.qc_correlation,
                ..DetectorConfig::default()
            },
            block_limits: BlockLimits {
                max_span: self.qc_max_span,
                max_gates: self.qc_max_block_gates,
                max_support: self.qc_max_support,
            },
            candidate_limits: QcCandidateLimits {
                max_candidates: self.qc_max_candidates,
                max_gates: self.qc_max_replacement_gates,
                max_support: self.qc_max_support,
            },
            max_attempts: self.qc_max_attempts,
            max_repairs: self.qc_max_repairs,
            ..QualityConfig::default()
        }
    }
}
