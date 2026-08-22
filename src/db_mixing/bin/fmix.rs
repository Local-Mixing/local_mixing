// Multi-mode GSS mixer driver. It runs production Phase A (`--gss --phase-a`),
// the split step (`--split --split-stop`), and the resumed crossing walk
// (`--resume`); standalone mpmct1/g57 walks remain supported. The shared
// randomized, size-thermostatted rewrite walk includes R-rule crossings,
// fresh-wire case splits, unsubsume splits,
// copy-pair insertions and conjugation twists (--w-twist-neg / --w-twist-swap:
// bracket a window with a wire negation or a 3-CNOT wire swap and conjugate
// its interior — state/progress mixing, the SAMF mechanism XGate-native)
// expand; catalogue merges (cancel / X-fuse / drop-literal / subsume)
// contract. The thermostat holds the gate count near --target-size; the
// objective is churn (distance from the original description), not size.
// The split/merge chain never emits comp=1 gates, so with the DB channels and
// --twist-g57 off the g57 "fossil" count is monotone; DB splices and g57-word
// twist brackets both emit g57-form material (see the mix.rs header). Ends with
// a final uniform float, a sampled global check against the input, and an
// mpmct1 write.
//
// Graceful stop: touch $FMIX_STOP_FLAG and the run finishes cleanly (verified
// write) at the next report point. Snapshot: touch $FMIX_DUMP_FLAG to get a
// verified mid-run circuit at $FMIX_DUMP_OUT (default <output>.snapshot.txt).
//
// Example:
//   FMIX_STOP_FLAG=/tmp/fmix.stop fmix --input mixed_fsplit.txt \
//     --target-size 3000000 --moves 50000000 --output mixed_fmix.txt
use clap::Parser;
use local_mixing::circuit::xgate::{XGate, max_wire};
use local_mixing::db_mixing::db_replace::DbMode;
use local_mixing::engine::format;
use local_mixing::engine::mix::{GEN_FRESH, MixParams, MixStop, Mixer, ORIGIN_SYNTH};

#[derive(Parser, Debug)]
#[command(name = "fmix")]
struct Args {
    /// Input circuit file. Not required when --resume is given: a resume
    /// rebuilds the circuit, its metadata and the original from the state file.
    #[arg(long, required_unless_present = "resume")]
    input: Option<String>,
    /// Input format: mpmct1 | g57
    #[arg(long, default_value = "mpmct1")]
    input_format: String,
    /// Output file (mpmct1 format)
    #[arg(long)]
    output: Option<String>,
    /// Thermostat target gate count (default: input size)
    #[arg(long)]
    target_size: Option<usize>,
    /// Thermostat softness in gates (default: max(target/100, 64))
    #[arg(long)]
    temp: Option<f64>,
    /// Total move attempts
    #[arg(long, default_value_t = 1_000_000)]
    moves: u64,
    /// Max controls per gate (K)
    #[arg(long, default_value_t = 12)]
    k_max: usize,
    /// Width-damping offset D for expansion moves (fsplit convention)
    #[arg(long, default_value_t = 2)]
    split_damp: usize,
    /// Width-damper base B: a split of parent width c proceeds with
    /// probability B^-(c - split_damp)
    #[arg(long, default_value_t = 2.0)]
    split_base: f64,
    /// Probability that a collision fragment inherits the shot gate's
    /// direction (else it gets the opposite)
    #[arg(long, default_value_t = 0.75)]
    dir_p: f64,
    /// Directional transport fraction: fresh pieces advance floor(q * slack)
    /// in their own direction at birth; a failed cross retreats the shot gate
    /// floor((1-q) * way)
    #[arg(long, default_value_t = 0.85)]
    dir_q: f64,
    /// Max distance (gates) a merge partner may sit from the initiator
    #[arg(long, default_value_t = 4096)]
    merge_reach: usize,
    /// Undo journal capacity (recorded crossings eligible for reversal)
    #[arg(long, default_value_t = 262_144)]
    journal_len: usize,
    /// Fraction of contraction moves that try a journal undo first
    #[arg(long, default_value_t = 0.5)]
    undo_frac: f64,
    /// Refractory period in moves: a split event may not be undone or
    /// sibling-merged until this many moves have passed
    #[arg(long, default_value_t = 2_000)]
    tabu_moves: u64,
    /// Expansion move weights
    #[arg(long, default_value_t = 0.70)]
    w_cross: f64,
    /// SUSPENDED by default (covered by the twists' case-splitting); set > 0
    /// to re-enable
    #[arg(long, default_value_t = 0.0)]
    w_fresh: f64,
    #[arg(long, default_value_t = 0.10)]
    w_unsub: f64,
    #[arg(long, default_value_t = 0.05)]
    w_insert: f64,
    /// Conjugation-twist weights (0 = off, trajectory-identical to the
    /// pre-twist chain). One twist conjugates a whole window (log-uniform
    /// length up to the circuit size) by a wire negation (+2 gates) or a wire
    /// swap (+6 gates), so keep these SMALL relative to the other weights:
    /// ~1e-4 gives a few thousand twists per 10M expansion moves.
    #[arg(long, default_value_t = 0.0)]
    w_twist_neg: f64,
    #[arg(long, default_value_t = 0.0)]
    w_twist_swap: f64,
    /// Transvection twist: conjugate a window by x_a ^= x_b (one CNOT per
    /// side, +2 gates). Affine and NOT a Hamming isometry — the rung that
    /// breaks avalanche-style distance gauges neg/swap twists provably
    /// preserve. Interior gates reading a case-split on b (count x2, width +1,
    /// K-cap enforced); b is drawn from wires the window never writes, which
    /// caps these windows at the mid scale (~n*ln n gates).
    #[arg(long, default_value_t = 0.0)]
    w_twist_cnot: f64,
    /// Probability each swapped wire is negated (swap family). 0.5 = default
    /// (swap 1/4, negate-one 1/2, negate-both 1/4); 0.0 = pure positive swaps
    /// (no polarity flips, but the 3-CNOT brackets are still inserted).
    #[arg(long, default_value_t = 0.5)]
    twist_neg_p: f64,
    /// SAMPLED ancestry: trace this many randomly chosen INPUT gates and report,
    /// per traced gate, the set of current gates descended from it (count,
    /// positional coverage and entropy). Fixed K-bit-per-litter cost, so unlike
    /// --ancestors it scales to production circuits. Overrides --ancestors.
    #[arg(long, default_value_t = 0)]
    anc_samples: usize,
    /// Tracer-selection seed. 0 (default) makes the tracer set depend only on
    /// (input size, K), so runs are comparable and a resume traces the same
    /// gates; vary for independent replicates.
    #[arg(long, default_value_t = 0)]
    anc_sample_seed: u64,
    /// Write a per-gate ancestry sidecar (final order, matching --output): a
    /// header naming the universe (exact m / sampled K + tracer list), then
    /// one ancestor-set line per gate. The cross-RUN counterpart of the state
    /// file: a later run imports it with --anc-in, so a phase boundary stops
    /// resetting the ancestry clock. Needs ancestry armed.
    #[arg(long)]
    anc_out: Option<String>,
    /// Import per-gate ancestor lists from a sidecar written by --anc-out and
    /// use them as this run's INITIAL ancestry (fresh runs only; the gate
    /// count must match the input circuit). The file defines the universe, so
    /// --ancestors / --anc-samples must not also be given.
    #[arg(long)]
    anc_in: Option<String>,
    /// Minimum twist window length (max is the current circuit size)
    #[arg(long, default_value_t = 64)]
    twist_min_len: usize,
    /// LAYER-2 phase-A preset: sets the phase-A default block (--twist-g57
    /// --p-twist 0.0005 --db-advance --p-mingen 0.6 --mix-pay-random, COMP
    /// p_mingen 0) unless individually overridden. curated and p_convex are
    /// no longer set here: the 2026-08-03 shipped defaults (curated ON,
    /// p_convex 0.4) already cover them.
    #[arg(long, default_value_t = false)]
    phase_a: bool,
    /// Pay-random MIX selection: when only larger spellings exist, pick a
    /// uniformly random one instead of a minimal one (layer-2 phase A).
    #[arg(long, default_value_t = false)]
    mix_pay_random: bool,
    /// LAYER-2 size profile "N0,N1,N2,R1,R2": three-phase best-effort size
    /// schedule in effective-work (moves/gate) units — expand to R1*input by
    /// N0, hold to N1, compress toward R2*input by N2. The controller reads
    /// the live monitors and steers --p-mix; it is then the ONLY size
    /// authority, so passing any of --target-size/--size-hi/--size-lo
    /// alongside --profile is an error. Empty = off.
    #[arg(long, default_value = "")]
    profile: String,
    /// Profile controller: control cadence in effective-work units.
    #[arg(long, default_value_t = 0.5)]
    prof_cadence_eff: f64,
    /// Profile controller: relative size deadband (no lever change within).
    #[arg(long, default_value_t = 0.02)]
    prof_deadband: f64,
    /// Profile controller: max |Δp_mix| per update (rate limit).
    #[arg(long, default_value_t = 0.1)]
    prof_dp_max: f64,
    /// Profile controller: EWMA weight for fresh plant (ghat/shat) estimates.
    #[arg(long, default_value_t = 0.3)]
    prof_ewma: f64,
    /// Profile controller: integral gain on the size tracking error.
    #[arg(long, default_value_t = 0.05)]
    prof_ki: f64,
    /// Spell twist brackets as all-g57 words sited adaptively so they absorb
    /// neighborhood gates (hidden-SAMF style), instead of 3-CNOT packets.
    /// Pure swap only (twist_neg_p is ignored on this path); every inserted
    /// gate takes the ballistic birth-advance unconditionally.
    #[arg(long, default_value_t = false)]
    twist_g57: bool,
    /// Probability a CONTRACTION tries COMP-DB (non-growing, uniform among the
    /// shortest) before falling through to journal undo and the merge
    /// catalogue. Requires FROZEN_DB_DIR.
    #[arg(long, default_value_t = 1.0)]
    p_comp: f64,
    /// Probability an EXPANSION is an ANY-DB move rather than a cross.
    #[arg(long, default_value_t = 0.1)]
    p_any: f64,
    /// Slot-2 admission rule: mix | comp | any. MIX is free-if-possible else
    /// pay-the-minimum and is the phase-A default; COMP refuses to grow and is
    /// the manual size brake; ANY takes any equivalent and accelerates growth.
    #[arg(long, default_value = "mix")]
    db_mode: String,
    /// Size-agnostic DB move: probability that a whole round is spent replacing a
    /// sampled window with a uniform random equivalent of ANY gate count (may
    /// grow the circuit), instead of the normal contract/expand step. 0 = off.
    #[arg(long, default_value_t = 0.0)]
    p_db: f64,
    /// Fixed top-level twist rate: each round is a conjugation twist with this
    /// probability, independent of the contract/expand economy (whose round
    /// supply collapses when the size controller holds at target). Twist type
    /// follows the w-twist-* weights as ratios (neg/swap 50/50 when all 0);
    /// with this set, the expansion mix no longer performs twists. Size
    /// machinery balances around the fixed rate. Rough sizing: coverage per
    /// move ~= p-twist x mean-window-span / size.
    #[arg(long, default_value_t = 0.0)]
    p_twist: f64,
    /// Arm the SPLIT STAGE (docs/FMIX_SPLIT_TWIST.md): split twists — a g57
    /// split plus an absorbed long-range pure-NOT twist plus one cross — are
    /// the ONLY move until the stage exits (g57 exhaustion, or
    /// --split-fail-limit consecutive bracket failures), then the round runs
    /// under the rest of this command line as usual.
    #[arg(long, default_value_t = false)]
    split: bool,
    /// End the RUN at the split-stage boundary (trial mode) instead of
    /// continuing into part 2.
    #[arg(long, default_value_t = false)]
    split_stop: bool,
    /// Probability a split carries the absorbed NOT twist + cross; the rest
    /// end after the bare split.
    #[arg(long, default_value_t = 0.8)]
    p_join: f64,
    /// Consecutive bracket-search failures (step 4e) that end the stage.
    #[arg(long, default_value_t = 100)]
    split_fail_limit: u32,
    /// Wire canaries planted at split-stage start (0 = off): per-position flip
    /// monitors, reported by ORIGINAL position at the stage boundary.
    #[arg(long, default_value_t = 256)]
    split_canaries: usize,
    /// Length bias of the bracket draw: k candidates sampled on the picked
    /// g57's own side (its stored direction), farthest wins. 1 = uniform,
    /// 2 ~ 2/3 of the available run, 3 ~ 3/4; larger = longer spans.
    /// 0 = the ORIGINAL other-half-first cascade (A/B comparison arm).
    #[arg(long, default_value_t = 2)]
    split_reach_k: usize,
    /// Layer-1 dispatch weight for split twists inside the twist slot OUTSIDE
    /// the split stage (the stage itself forces 1.0).
    #[arg(long, default_value_t = 0.0)]
    p_split_twist: f64,
    /// Min-dgen cross-shot bias: probability a cross shot is drawn from the
    /// pool of the K least-split lineages instead of uniformly (the uniform
    /// draw is rich-get-richer; this points expansion at untouched
    /// families). 0 = off, and off draws no RNG — trajectories identical.
    #[arg(long, default_value_t = 0.0)]
    p_mincross: f64,
    /// Min-dgen pool size (a COUNT): must exceed the biased draws taken
    /// between rebuilds or the coin silently degrades to uniform.
    #[arg(long, default_value_t = 20_000)]
    cross_pool_k: usize,
    /// Min-dgen pool rebuild cadence in moves (O(size) scan each).
    #[arg(long, default_value_t = 10_000)]
    cross_rescan: u64,
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
    shuffle_rate: f64,
    /// Two-pass store routing, ON BY DEFAULT (2026-08-03): the descent runs
    /// curated-only over every window length first, and consults the regular
    /// store only if that whole pass came up empty. (The old one-pass cascade
    /// probed curated then regular at EACH length before shortening, so a
    /// regular hit at length p beat a curated hit at length p-1.) Needs
    /// --curated; applies to compression too while --curated-in-comp is on
    /// (also the default). Disable with --no-curated-exhaust.
    #[arg(long, default_value_t = true)]
    curated_exhaust: bool,
    /// Turn --curated-exhaust off (single-pass curated-then-regular at each
    /// window length).
    #[arg(long, default_value_t = false)]
    no_curated_exhaust: bool,
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
    curated_in_comp: bool,
    /// Turn --curated-in-comp off (compression goes regular-only, the old
    /// contract).
    #[arg(long, default_value_t = false)]
    no_curated_in_comp: bool,
    /// Window length the descent STARTS from, in MIX mode (COMP has its own,
    /// --s-db-comp). The descent visits every shorter length down to 1, so
    /// there is no separate minimum.
    #[arg(long, default_value_t = 9)]
    s_db: usize,
    /// Minimum DB window length (0 = off). Use with --no-db-prefixes so the
    /// descent does not visit shorter prefixes anyway.
    #[arg(long, default_value_t = 0)]
    db_min_window: usize,
    /// Probability the window sampler is convex rather than contiguous, in MIX
    /// mode (COMP: --p-convex-comp). Default 0.4 = contiguous 60% / convex 40%.
    #[arg(long, default_value_t = 0.4)]
    p_convex: f64,
    /// A gate with this many controls or more may not sit INSIDE a window.
    #[arg(long, default_value_t = 4)]
    w_window: usize,
    /// A gate with this many controls or more may not SEED a window or count
    /// toward the dose. Stricter than --w-window on purpose: width-3 gates
    /// match in context but re-encode end-to-end at 0.41% against 98.98% for
    /// width <= 2, so at a shared threshold they pile up in the pool forever.
    #[arg(long, default_value_t = 3)]
    w_pool: usize,
    /// Convex sampling: probability each growth step floats the block in g1's
    /// direction (else the opposite).
    #[arg(long, default_value_t = 0.75)]
    db_convex_p: f64,
    /// Skip the exhaustive per-splice equivalence check on DB replacements
    /// (faster long runs; the periodic global check still guards correctness).
    /// With this set, DB windows wider than 24 wires can also be replaced.
    #[arg(long, default_value_t = false)]
    no_db_verify: bool,
    /// Record every DB replacement attempt to this file: the outgoing window, the
    /// number of equivalent DB circuits, and (on success) the replacing circuit.
    #[arg(long)]
    db_record: Option<String>,
    /// Measurement mode: sample windows and record DB match counts (--db-record)
    /// but never splice, so the circuit stays fixed. With --p-db 1.0 and all
    /// other move weights 0 this makes fmix a pure match-rate sampler.
    #[arg(long, default_value_t = false)]
    db_dry_run: bool,
    /// Degree cap for the DB lookup: a window whose function degree exceeds this
    /// cannot match any stored circuit and is skipped before canonicalization
    /// (the main speed guard on high-width windows). Set to the DB's max ANF
    /// degree; must be <= 11 (the probe tests (cap+1)-dim subcubes and maxes
    /// out at 12). 0 = off (every window canonicalizes). Degree-skipped
    /// attempts are still recorded by --db-record, so measurement runs can
    /// (and should) keep the cap on.
    #[arg(long, default_value_t = 0)]
    db_max_degree: usize,
    /// Random subcubes probed per direction by the degree cap (higher = fewer
    /// missed high-degree windows, at proportional cost).
    #[arg(long, default_value_t = 6)]
    db_degree_probes: usize,
    /// Span cap for the DB lookup: a window touching more distinct wires than
    /// this is recorded as a miss without canonicalizing. Set to the store's
    /// max canonical support (measure with frozen_degree_scan). This is the
    /// main speed guard: Rule-L canonicalization cost explodes with tied wire
    /// count, and wide-span windows can't match anyway. 0 = off.
    #[arg(long, default_value_t = 0)]
    db_max_span: usize,
    /// Per-wire polynomial term cap for the DB lookup budget: a window whose
    /// wire poly outgrows the largest any stored function has cannot match,
    /// and the budget bail lands before the expensive Rule-L canonicalization.
    /// Set to the store's max (frozen_degree_scan census). 0 = default 2^18.
    #[arg(long, default_value_t = 0)]
    db_wire_terms: usize,
    /// Total-terms cap across a window's wire polys (census: per-entry total).
    /// 0 = default 2^20.
    #[arg(long, default_value_t = 0)]
    db_total_terms: usize,
    /// Largest-first prefix descent (the size-reduction cascade), ON BY
    /// DEFAULT (2026-08-03): try the full sampled window, then its len-1
    /// prefix, etc. down to length 1, splicing the LONGEST matching prefix
    /// (live) or recording every prefix (with --db-dry-run). Span/verify
    /// declines keep descending — shorter prefixes may still fit. Disable
    /// with --no-db-prefixes.
    #[arg(long, default_value_t = true)]
    db_prefixes: bool,
    /// Turn --db-prefixes off (one attempt at a uniformly sampled length, no
    /// descent).
    #[arg(long, default_value_t = false)]
    no_db_prefixes: bool,
    /// Give DB splice products the ballistic birth-advance that split pieces
    /// get: each product floats floor(dir_q * slack) along its own direction.
    /// Without this a splice assigns directions nothing acts on, so a
    /// DB-dominated schedule has no directional transport at all. The other
    /// source of transport -- crossings -- widens gates, and width is what
    /// kills DB matching. Off by default (it changes trajectories).
    #[arg(long, default_value_t = false)]
    db_advance: bool,
    /// Pair-geometry rate (docs/NONLOCAL_PHASE_A.md): probability a non-COMP
    /// DB round samples its window as the seed plus one FAR COMMUTING partner,
    /// floated adjacent and fused into a 2-gate window — the phase-A transport
    /// experiment. The fused splice unions litters across the seed's whole
    /// commutation box, and the reorder ban forces every pair splice onto a
    /// genuinely different spelling. 0 = off, draws no RNG.
    #[arg(long, default_value_t = 0.0)]
    p_pair: f64,
    /// Pair geometry: cap on the commutation-box scan past the seed.
    #[arg(long, default_value_t = 4096)]
    pair_scan_cap: usize,
    /// Pair geometry: pick the partner uniformly from the eligible box instead
    /// of the farthest gate.
    #[arg(long, default_value_t = false)]
    pair_pick_uniform: bool,
    /// Bridge-fusion rate (docs/NONLOCAL_PHASE_A.md): per-round probability of
    /// jointly re-encoding two gates that commutation CANNOT bring together —
    /// a carrier conjugates the interior (wake corrections on interior
    /// colliders, non-g57: trades polf for reach like legacy twist packets)
    /// and both carrier-adjacent windows are re-spelled through the store.
    /// 0 = off, draws no RNG.
    #[arg(long, default_value_t = 0.0)]
    p_bridge: f64,
    /// Bridge: log-uniform interior-length draw, lower bound.
    #[arg(long, default_value_t = 16)]
    bridge_min_span: usize,
    /// Bridge: log-uniform interior-length draw, upper bound.
    #[arg(long, default_value_t = 512)]
    bridge_max_span: usize,
    /// Bridge: refuse rounds whose interior holds more colliders than this
    /// (each costs one or two wake correction gates).
    #[arg(long, default_value_t = 8)]
    bridge_max_colliders: usize,
    /// Layer-2 mode overlay (slot 0): per-round probability the slot-2 DB move
    /// is MIX-DB, else COMP-DB. Each round flips this coin, overriding --db-mode
    /// and reading the chosen mode's own knobs -- MIX uses --s-db / --p-convex /
    /// --p-mingen, COMP uses the *-comp overrides below. < 0 disables the
    /// overlay (single --db-mode, as before). The thermostat is unaffected, so
    /// pair with --p-db 1.0 for a pure per-round MIX/COMP schedule.
    #[arg(long, default_value_t = -1.0)]
    p_mix: f64,
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
    s_db_comp: Option<usize>,
    /// COMP-mode convex probability. Shipped default 0.9 = convex 90% /
    /// contiguous 10%. Convex wins compression on every axis measured (16x
    /// gates removed, 7x ancestry transport, 31x less CPU -- wide contiguous
    /// windows cost ~30x per canonicalization even when they pass the span
    /// cap), so COMP leans hard convex.
    #[arg(long)]
    p_convex_comp: Option<f64>,
    /// COMP-mode pool-seed probability under --p-mix. Unset = use --p-mingen.
    #[arg(long)]
    p_mingen_comp: Option<f64>,
    /// MIX-mode window length when the round draws a CONTIGUOUS window.
    /// Unset = share --s-db across both geometries. A contiguous window of the
    /// same gate count spans far more wires than a convex one and costs 3.6x
    /// its canonicalization at length 5, 12.6x at 7, 47.8x at 12 -- so a
    /// profile that wants a wide convex probe usually wants a narrow
    /// contiguous one.
    #[arg(long)]
    s_db_ctg: Option<usize>,
    /// COMP-mode window length when the round draws a CONTIGUOUS window.
    /// Unset = use --s-db-comp.
    #[arg(long)]
    s_db_comp_ctg: Option<usize>,
    /// Prefix descent in MIX rounds only (unset = use --db-prefixes). Under the
    /// --p-mix overlay both modes run in one process and want opposite
    /// settings, so this splits the global flag per mode.
    #[arg(long)]
    db_prefixes_mix: Option<bool>,
    /// Prefix descent in COMP rounds only (unset = use --db-prefixes).
    #[arg(long)]
    db_prefixes_comp: Option<bool>,
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
    gss: bool,
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
    curated: bool,
    /// Turn --curated off (regular store only, both modes).
    #[arg(long, default_value_t = false)]
    no_curated: bool,
    /// Track, per litter, the SET of original input gates that contributed to
    /// it (the union of the sets its outgoing window drew on). Reports anc=
    /// (mean set size: what a mixed gate is made of) and ancspan= (mean
    /// normalised index span: how far input material travels to meet). Immune
    /// to the ORIGIN_SYNTH erosion that makes odiff/oadj unreadable. Costs
    /// |input| bits per live litter, so SMALL INPUTS ONLY -- it refuses above
    /// 20k input gates.
    #[arg(long, default_value_t = false)]
    ancestors: bool,
    /// Probability a COMP-DB attempt restricts itself to PURE g57 material and
    /// starts its descent at --s-db-g57. Only g57-only windows survive length:
    /// measured match rate is 100% to m=5 and 94% at 6, but ANY non-g57 gate in
    /// a 6-gate window drops it to <=7%, so the long-window compression that
    /// pays is unavailable except on pure windows.
    #[arg(long, default_value_t = 0.0)]
    p_comp_g57: f64,
    /// Upper clamp on the contraction probability. 0.98 leaves a 2% expansion
    /// floor above target -- a structural growth source, but also what keeps
    /// crossings (hence fossil erosion) running while the walk sits at target.
    #[arg(long, default_value_t = 0.98)]
    contract_ceiling: f64,
    /// Size brake: growth to this size forces slot 2 into COMP. 0 = brake off.
    #[arg(long, default_value_t = 0)]
    size_hi: usize,
    /// Release the brake at this size, or earlier if COMP stops paying.
    #[arg(long, default_value_t = 0)]
    size_lo: usize,
    /// Release the brake when COMP sheds fewer than this many gates per round
    /// over the trailing window. This is what makes a WIDE band safe: the risk
    /// was never band width but sitting in COMP past its usefulness.
    #[arg(long, default_value_t = 0.0)]
    comp_release_eps: f64,
    /// Trailing window (moves) for the productivity release.
    #[arg(long, default_value_t = 250_000)]
    comp_release_window: u64,
    /// Starting window length for a g57-only COMP attempt.
    #[arg(long, default_value_t = 9)]
    s_db_g57: usize,
    /// Frozen store directory. Overrides FROZEN_DB_DIR, which is env-only and
    /// in no rc file -- detached runs that miss it abort instantly.
    #[arg(long)]
    frozen_db_dir: Option<String>,
    /// Curated store directory. Overrides FROZEN_CURATED_DIR.
    #[arg(long)]
    frozen_curated_dir: Option<String>,
    /// Generation targeting: drive every (ctrl-cap-eligible) gate through at
    /// least this many DB re-encodings. Each gate carries a generation (input
    /// gates 0; a DB splice stamps min(window)+1; splits/merges propagate;
    /// fresh insert/bracket material counts as done). DB seeds are drawn from
    /// the below-target gates with probability --gen-bias, replacing the
    /// coupon-collector tail of uniform selection with direct work. 0 = off.
    #[arg(long, default_value_t = 0)]
    gen_target: u32,
    /// Probability a DB seed comes from the generation POOL rather than
    /// uniformly. This is what stops the walk being a coupon collector, where
    /// the last few percent of gates soak up most of the moves.
    #[arg(long, default_value_t = 0.8)]
    p_mingen: f64,
    /// Pool size in GATES: the K lowest-generation gates that are pool-eligible
    /// and still below the goal. A count rather than a fraction, because the
    /// drain rate is set by the move economy (gen_rescan x p_db x p_mingen) and
    /// is independent of circuit size. K must exceed the draws taken between
    /// rebuilds or the pool empties and the biased coin silently degrades to
    /// uniform -- watch the fall-through counter.
    #[arg(long, default_value_t = 20_000)]
    pool_k: usize,
    /// Stop when the failure fraction over the last --canary-window QUALIFYING
    /// rounds exceeds this. A qualifying round is one whose seed genuinely came
    /// from the pool; heads coins that fell through a drained pool are counted
    /// separately, since those mean the rebuild is too slow rather than that the
    /// material is unreachable. Asleep while the brake holds COMP, because COMP
    /// declines far more often by construction. 0 = off.
    #[arg(long, default_value_t = 0.0)]
    canary_theta: f64,
    /// Trailing window for the canary, in qualifying rounds.
    #[arg(long, default_value_t = 2000)]
    canary_window: usize,
    /// Refuse a descent rung that is exactly one COMPLETE litter -- the set some
    /// earlier replacement emitted, and so where the store is most likely to
    /// hand that spelling straight back. Singleton litters are exempt: an input
    /// gate has no earlier spelling, and banning it would also refuse the
    /// length-1 rung, the one that always makes progress.
    #[arg(long, default_value_t = false)]
    litter_ban: bool,
    /// Draw this many candidate windows and keep the one spanning the most
    /// distinct litters. 1 = off.
    #[arg(long, default_value_t = 1)]
    litter_samples: usize,
    /// Candidate positions the twist placer samples looking for a welcoming
    /// neighbourhood (a gate that can absorb the bracket) before giving up and
    /// placing the twist uniformly at random. 0 = always random.
    #[arg(long, default_value_t = 0)]
    twist_place_tries: usize,
    /// Split-rule variant: children INHERIT the parent generation unchanged
    /// (only DB replacements raise generations). Default off = ratchet
    /// semantics (split children get parent + 1).
    #[arg(long, default_value_t = false)]
    gen_split_inherit: bool,
    /// Median variant for the DB generation stamp: use the LOWER median
    /// (rounded down on even windows; on 2-gate windows this is the min).
    /// Default off = upper median (rounded up).
    #[arg(long, default_value_t = false)]
    gen_median_low: bool,
    /// Laggard-list rebuild cadence in moves (O(size) scan each rebuild;
    /// entries going stale between rebuilds are pruned at draw time).
    #[arg(long, default_value_t = 10_000)]
    gen_rescan: u64,
    /// Dose-based stop: end the run (before the move budget) at the first
    /// report point where the below-target fraction among eligible gates is
    /// <= this AND --twist-cov-stop is met. The phase-A "minimal growth"
    /// switch: spend exactly the moves the dose requires, no more. Negative =
    /// off. Requires --gen-target > 0.
    #[arg(long, default_value_t = -1.0)]
    gen_stop_frac: f64,
    /// Twist-coverage requirement for the dose stop: cumulative twisted span
    /// over current size (per-position coverage; saturation target ~600).
    /// 0 = no coverage requirement.
    #[arg(long, default_value_t = 0.0)]
    twist_cov_stop: f64,
    /// Write per-gate DB-generation stamps (final order, one per line;
    /// 4294967295 = born-random material) for dose analysis
    #[arg(long)]
    gens_out: Option<String>,
    /// Verified snapshot each time the circuit generation (G=, the
    /// 5th-percentile gate generation) crosses a multiple of this value:
    /// <output>.gen<m>.mpmct1 + .gens sidecar. 0 = off. Meaningful only with
    /// --gen-target > 0 (generations move under DB re-encoding).
    #[arg(long, default_value_t = 0)]
    gen_snap_every: u32,
    /// Verified snapshot at every multiple of this many moves
    /// (<output>.mv<moves>.mpmct1 + .gens sidecar): the progress clock for
    /// pure-split runs where the generation census is not meaningful. Use a
    /// multiple of --report-every. 0 = off.
    #[arg(long, default_value_t = 0)]
    snap_every_moves: u64,
    /// Global sampled equality check every N moves
    #[arg(long, default_value_t = 10_000)]
    verify_every: u64,
    /// Progress report (and stop/dump flag check) every N moves
    #[arg(long, default_value_t = 50_000)]
    report_every: u64,
    /// Disable the per-move exhaustive local verification
    #[arg(long, default_value_t = false)]
    no_local_verify: bool,
    /// Skip the final uniform float pass
    #[arg(long, default_value_t = false)]
    skip_final_float: bool,
    /// Write per-gate origin indices (final order, one per line; 4294967295 =
    /// synthetic) for dispersion analysis
    #[arg(long)]
    origins_out: Option<String>,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Write a resume file on every stop (and at each snapshot). Holds what the
    /// circuit file cannot: per-gate direction/generation/litter/event, the undo
    /// journal, the ORIGINAL circuit that global_check verifies against, and the
    /// condition state (moves, twist coverage, canary ring, brake, pool).
    #[arg(long)]
    state_out: Option<String>,
    /// Per-gate litter-id sidecar for the INPUT circuit (header "litter1 N",
    /// one id per gate line) — e.g. written by sgdb_substitute so each
    /// substituted block counts as a litter for --litter-ban.
    #[arg(long)]
    litter_in: Option<String>,
    /// Resume from a state file instead of --input. Parameters still come from
    /// the command line, so a paused run can be re-steered; only the state
    /// VERSION must match, since field meanings would otherwise drift silently.
    #[arg(long)]
    resume: Option<String>,
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
struct BaseGiven {
    s_db: bool,
    p_convex: bool,
}

#[derive(Default, Clone, Copy)]
struct DbLayer {
    s_db: Option<usize>,
    s_db_ctg: Option<usize>,
    s_db_comp: Option<usize>,
    s_db_comp_ctg: Option<usize>,
    p_convex: Option<f64>,
    p_convex_comp: Option<f64>,
    p_mingen: Option<f64>,
    p_mingen_comp: Option<f64>,
    prefixes: Option<bool>,
    prefixes_mix: Option<bool>,
    prefixes_comp: Option<bool>,
}

impl DbLayer {
    /// `self` wins wherever it has an opinion; `under` fills the rest.
    fn over(self, under: DbLayer) -> DbLayer {
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
    fn shipped(base_given: BaseGiven) -> DbLayer {
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
    fn gss() -> DbLayer {
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

    /// Phase A's DB opinions (the twist/advance block stays in the preset
    /// below; only DB knobs belong here).
    fn phase_a() -> DbLayer {
        DbLayer {
            p_mingen: Some(0.6),
            p_mingen_comp: Some(0.0),
            ..DbLayer::default()
        }
    }
}

fn main() {
    // Keep the raw matches so the preset below can tell "the user asked for
    // this value" from "the user said nothing" — `--p-twist 0` is a real
    // request and must not be overwritten just because 0 is also the default.
    let matches = <Args as clap::CommandFactory>::command().get_matches();
    let given =
        |name: &str| matches.value_source(name) == Some(clap::parser::ValueSource::CommandLine);
    let mut args = <Args as clap::FromArgMatches>::from_arg_matches(&matches).expect("parse args");

    // Off switches for the default-on DB knobs (2026-08-03 defaults). Applied
    // first, so everything below -- the phase-A preset included -- reads the
    // settled values.
    if args.no_db_prefixes {
        args.db_prefixes = false;
    }
    if args.no_curated {
        args.curated = false;
    }
    if args.no_curated_exhaust {
        args.curated_exhaust = false;
    }
    if args.no_curated_in_comp {
        args.curated_in_comp = false;
    }

    // LAYER-2 phase-A preset: fill the phase-A default block for any knob the
    // user did not pass explicitly. Explicit flags always win. (curated and
    // p_convex used to be set here; both are covered by the 2026-08-03
    // shipped defaults -- curated ON, p_convex 0.4 -- so phase A now just
    // inherits them.)
    if args.phase_a {
        if !given("twist_g57") {
            args.twist_g57 = true;
        }
        if !given("p_twist") {
            args.p_twist = 0.0005;
        }
        if !given("db_advance") {
            args.db_advance = true;
        }
        if !given("mix_pay_random") {
            args.mix_pay_random = true;
        }
        // Its DB opinions (p_mingen 0.6 / p_mingen_comp 0) live in
        // DbLayer::phase_a(), applied with the rest of the stack below.
        println!(
            "[fmix] phase-A preset ON: twist-g57={} p_twist={} db-advance={} curated={} mix-pay-random={} (DB knobs settled below)",
            args.twist_g57, args.p_twist, args.db_advance, args.curated, args.mix_pay_random
        );
    }

    // GSS profile: the non-DB half (its DB block is DbLayer::gss()).
    if args.gss {
        if !given("curated") {
            args.curated = true;
        }
        if !given("curated_in_comp") {
            args.curated_in_comp = true;
        }
        if !given("db_advance") {
            args.db_advance = true;
        }
        println!(
            "[fmix] GSS profile ON: curated={} db-advance={} | p_mix NOT set (layer-2 owns it) | DB knobs settled below",
            args.curated, args.db_advance
        );
    }

    // ---- DB knob precedence, in one place ----
    //
    //   explicit CLI  >  preset (--gss over --phase-a)  >  shipped defaults
    //
    // and within each layer, most specific wins at USE time (mode+geometry ->
    // mode -> base, in Mixer::active_*). The rule that matters, and that this
    // replaced: an EXPLICIT base knob now beats a merely-DEFAULTED mode knob.
    // Before, `--db-mode comp --s-db 20` silently ran at 12 because
    // `s_db_comp`'s default of 12 was indistinguishable from a real request.
    let cli = DbLayer {
        s_db: given("s_db").then_some(args.s_db),
        p_convex: given("p_convex").then_some(args.p_convex),
        p_mingen: given("p_mingen").then_some(args.p_mingen),
        // `--no-db-prefixes` is just "explicitly false" -- folding it in here
        // retires it as a separate mechanism.
        prefixes: if args.no_db_prefixes {
            Some(false)
        } else {
            given("db_prefixes").then_some(args.db_prefixes)
        },
        // These are Option at the CLI, so Some IS "the user asked".
        s_db_ctg: args.s_db_ctg,
        s_db_comp: args.s_db_comp,
        s_db_comp_ctg: args.s_db_comp_ctg,
        p_convex_comp: args.p_convex_comp,
        p_mingen_comp: args.p_mingen_comp,
        prefixes_mix: args.db_prefixes_mix,
        prefixes_comp: args.db_prefixes_comp,
    };
    let mut preset = DbLayer::default();
    if args.phase_a {
        preset = preset.over(DbLayer::phase_a());
    }
    if args.gss {
        // GSS is the more specific profile, so it sits above phase A.
        preset = DbLayer::gss().over(preset);
    }
    let base_given = BaseGiven {
        s_db: given("s_db"),
        p_convex: given("p_convex"),
    };
    let db = cli.over(preset).over(DbLayer::shipped(base_given));
    // Settle the args once, so every consumer below reads the same values.
    let must = |o: Option<f64>| o.expect("DbLayer::shipped() sets every base knob");
    args.s_db = db.s_db.expect("shipped sets s_db");
    args.p_convex = must(db.p_convex);
    args.p_mingen = must(db.p_mingen);
    args.db_prefixes = db.prefixes.expect("shipped sets prefixes");
    args.s_db_ctg = db.s_db_ctg;
    args.s_db_comp = db.s_db_comp;
    args.s_db_comp_ctg = db.s_db_comp_ctg;
    args.p_convex_comp = db.p_convex_comp;
    args.p_mingen_comp = db.p_mingen_comp;
    args.db_prefixes_mix = db.prefixes_mix;
    args.db_prefixes_comp = db.prefixes_comp;

    // A knob that cannot possibly fire is a bug in the command line, not a
    // no-op to shrug at -- silently-inert flags are exactly how the shadowing
    // bug hid for two days. With the overlay off (p_mix < 0) only one mode
    // ever runs, so the other mode's overrides can never be read.
    if args.p_db > 0.0 && args.p_mix < 0.0 {
        let comp_only = args.db_mode == "comp";
        // Read `cli`, NOT the settled args: after the layers are applied every
        // override holds a value, so blaming `args` would name flags the user
        // never typed.
        let inert: Vec<&str> = if comp_only {
            [
                ("--s-db-ctg", cli.s_db_ctg.is_some()),
                ("--db-prefixes-mix", cli.prefixes_mix.is_some()),
            ]
            .iter()
            .filter(|(_, set)| *set)
            .map(|(n, _)| *n)
            .collect()
        } else {
            [
                ("--s-db-comp", cli.s_db_comp.is_some()),
                ("--s-db-comp-ctg", cli.s_db_comp_ctg.is_some()),
                ("--p-convex-comp", cli.p_convex_comp.is_some()),
                ("--p-mingen-comp", cli.p_mingen_comp.is_some()),
                ("--db-prefixes-comp", cli.prefixes_comp.is_some()),
            ]
            .iter()
            .filter(|(_, set)| *set)
            .map(|(n, _)| *n)
            .collect()
        };
        if !inert.is_empty() {
            eprintln!(
                "[fmix] ERROR: {} can never take effect: --db-mode {} with no --p-mix overlay means {}-DB rounds never happen. Drop the flag, or arm the overlay with --p-mix.",
                inert.join(", "),
                args.db_mode,
                if comp_only { "MIX" } else { "COMP" }
            );
            std::process::exit(2);
        }
    }

    // Resolve the curated store BEFORE any banner mentions it, so the
    // printouts below describe what will actually run.
    if let Some(d) = &args.frozen_db_dir {
        unsafe { std::env::set_var("FROZEN_DB_DIR", d) };
    }
    if let Some(d) = &args.frozen_curated_dir {
        unsafe { std::env::set_var("FROZEN_CURATED_DIR", d) };
    }
    if args.curated && std::env::var("FROZEN_CURATED_DIR").is_err() {
        assert!(
            !given("curated"),
            "--curated needs FROZEN_CURATED_DIR; refusing to run, because \
             degrading silently to regular-only would look like a measurement"
        );
        // curated is only a DEFAULT here, so degrade -- but say so loudly.
        println!(
            "[fmix] WARNING: curated-first is the shipped default but FROZEN_CURATED_DIR is \
             unset -> curated OFF for this run (regular store only). Export FROZEN_CURATED_DIR \
             (or pass --frozen-curated-dir); pass --curated explicitly to make this an error."
        );
        args.curated = false;
    }
    if args.curated {
        assert!(
            !args.no_db_verify,
            "--curated with --no-db-verify is refused: the curated store has already been \
             observed returning a non-equivalent replacement (forward/reverse key confusion), \
             and the per-splice check is what caught it"
        );
    }

    assert!(
        !(args.split && !args.profile.is_empty()),
        "--split and --profile both steer the round; run the split stage as its own invocation"
    );
    // Parse the size profile and enforce single size authority.
    let profile: Option<([f64; 3], [f64; 2])> = if args.profile.is_empty() {
        None
    } else {
        let v: Vec<f64> = args
            .profile
            .split(',')
            .map(|x| {
                x.trim()
                    .parse()
                    .expect("--profile wants N0,N1,N2,R1,R2 reals")
            })
            .collect();
        assert_eq!(
            v.len(),
            5,
            "--profile wants exactly 5 comma-separated values N0,N1,N2,R1,R2"
        );
        // N2 may be given either as an ABSOLUTE effective-work mark (N2 >= N1)
        // or, as the original spec phrased it ("or N3 moves per gate"), as the
        // compression leg's BUDGET. A value below N1 can only mean the latter,
        // so read it that way and say so rather than rejecting the profile.
        let mut n = [v[0], v[1], v[2]];
        if n[2] < n[1] {
            let budget = n[2];
            n[2] = n[1] + budget;
            println!(
                "[fmix] profile: N2={budget} < N1={} read as the COMPRESSION BUDGET -> absolute end mark {}",
                n[1], n[2]
            );
        }
        let r = [v[3], v[4]];
        assert!(
            n[0] > 0.0 && n[1] >= n[0] && n[2] >= n[1],
            "--profile needs 0 < N0 <= N1 and a positive compression leg"
        );
        // R1 = 1 (no expand leg, pure hold) and R2 < 1 (compress below the
        // input size) are both valid schedules: prof_target is plain linear
        // interpolation with no (R1-1) divisions, and the controller tracks
        // err against S* symmetrically. The compress leg below x1 is only as
        // feasible as the material allows — best-effort, same as any leg.
        assert!(
            r[0] >= 1.0 && r[1] > 0.0 && r[0] >= r[1],
            "--profile needs R1 >= 1 and 0 < R2 <= R1"
        );
        assert!(
            args.target_size.is_none() && args.size_hi == 0 && args.size_lo == 0,
            "--profile is the sole size authority: remove --target-size / --size-hi / --size-lo (make up your mind)"
        );
        assert!(
            args.p_mix < 0.0,
            "--profile drives p_mix; do not also pass --p-mix"
        );
        Some((n, r))
    };

    // A resume carries its own circuit, so there is nothing to read here.
    let (gates, file_wires): (Vec<XGate>, usize) = match (&args.resume, &args.input) {
        (Some(_), _) => (Vec::new(), 0),
        (None, Some(path)) => match args.input_format.as_str() {
            "mpmct1" => format::read_mpmct(path).expect("read mpmct1 circuit"),
            "g57" => {
                let g = format::read_g57_file(path).expect("read g57 circuit");
                let w = max_wire(&g) as usize + 1;
                (g, w)
            }
            other => panic!("unknown --input-format {other}"),
        },
        (None, None) => unreachable!("clap requires --input unless --resume"),
    };
    let num_wires = file_wires.max(max_wire(&gates) as usize + 1);
    let input_len = gates.len();
    let comp0 = gates.iter().filter(|g| g.comp).count();
    let target = args.target_size.unwrap_or(input_len);
    if args.resume.is_none() {
        println!(
            "[fmix] input: {} gates ({} g57 fossils), {} wires; k_max={} split_damp={} split_base={} dir_p={} dir_q={} target={} temp={} moves={} seed={}",
            input_len,
            comp0,
            num_wires,
            args.k_max,
            args.split_damp,
            args.split_base,
            args.dir_p,
            args.dir_q,
            target,
            args.temp.unwrap_or((target as f64 / 100.0).max(64.0)),
            args.moves,
            args.seed
        );
    }
    if args.p_pair > 0.0 {
        println!(
            "[fmix] pair geometry ON: p_pair={} scan_cap={} pick={} (far-pair fusion, docs/NONLOCAL_PHASE_A.md)",
            args.p_pair,
            args.pair_scan_cap,
            if args.pair_pick_uniform {
                "uniform"
            } else {
                "far"
            }
        );
    }
    if args.p_bridge > 0.0 {
        println!(
            "[fmix] bridge fusion ON: p_bridge={} span=[{},{}] max_colliders={} — wake corrections are non-g57 (polf > 0 expected; docs/NONLOCAL_PHASE_A.md)",
            args.p_bridge, args.bridge_min_span, args.bridge_max_span, args.bridge_max_colliders
        );
    }
    if args.p_twist > 0.0 {
        println!(
            "[fmix] first-class twist rounds ON: p_twist={} (w-twist-* weights serve as type ratios)",
            args.p_twist
        );
    }
    if args.curated && args.curated_in_comp {
        println!(
            "[fmix] curated-in-comp ON: COMPRESSION probes the curated store too (the size rule keeps only curated spellings shorter than the window, i.e. the shorter halves of identity splits)"
        );
    }
    if args.curated_exhaust {
        if args.curated {
            println!(
                "[fmix] curated-exhaust ON: the prefix descent runs CURATED-ONLY over every window length, and falls back to the regular store only if that whole pass missed ({})",
                if args.curated_in_comp {
                    "MIX and COMP alike"
                } else {
                    "expansion only; compression stays regular-only"
                }
            );
        } else if given("curated_exhaust") {
            println!(
                "[fmix] WARNING: --curated-exhaust without a curated store -- it has no effect"
            );
        }
    }
    if args.shuffle_rate > 0.0 {
        println!(
            "[fmix] global re-randomisation ON: shuffle_rate={} (per-round p = {}/|circuit|, i.e. one expected whole-circuit reshuffle per |circuit|/{} rounds; semantics- and size-preserving)",
            args.shuffle_rate, args.shuffle_rate, args.shuffle_rate
        );
    } else {
        println!("[fmix] global re-randomisation OFF (--shuffle-rate 0)");
    }
    if args.p_twist > 0.0 {
        println!(
            "[fmix] twists ON (swap family): p_twist={} twist_min_len={} twist_neg_p={} -- each swapped wire negated w.p. twist_neg_p (0.5 => swap 1/4, negate-one 1/2, negate-both 1/4; 0 => pure swap, no polarity flips) (w_twist_* retired/ignored)",
            args.p_twist, args.twist_min_len, args.twist_neg_p
        );
    }
    if args.twist_g57 {
        // Force-build the engine now so its cost is paid (and printed) at
        // startup rather than silently inside the first twist round.
        let eng = local_mixing::engine::swap_words::engine();
        println!(
            "[fmix] twist-g57 ON: brackets are adaptive all-g57 words (pure swap; twist_neg_p ignored), inserted gates take the birth-advance; engine ball {} perms, built in {} ms",
            eng.back_len, eng.build_ms
        );
    }
    if args.anc_samples > 0 {
        println!(
            "[fmix] SAMPLED ancestry ON: tracing {} input gates (sample_seed={}); reports per-tracer descendant count, positional coverage and entropy. Scales to any input size; anc=/ancspan= stay 0 (see the tracers line).",
            args.anc_samples, args.anc_sample_seed
        );
    }
    if args.p_mix >= 0.0 {
        println!(
            "[fmix] mode overlay ON (slot 0): p_mix={} -> MIX-DB w.p. p_mix else COMP-DB, per round (each mode's settled knobs are on the 'DB effective per mode' line below)",
            args.p_mix
        );
    }
    if args.p_comp > 0.0 || args.p_db > 0.0 || args.p_any > 0.0 {
        println!(
            "[fmix] DB ON: p_db(slot2)={} db_mode={} p_comp(contract)={} p_any(expand)={} w_window={} w_pool={} verify={} curated={} (FROZEN_DB_DIR required)",
            args.p_db,
            args.db_mode,
            args.p_comp,
            args.p_any,
            args.w_window,
            args.w_pool,
            !args.no_db_verify,
            args.curated
        );
        // Print what each mode will ACTUALLY use, resolved by the SAME code the
        // mixer runs (MixParams::db_knobs) rather than a second copy of the
        // fall-through rules -- the old banner re-derived them itself and could
        // therefore drift from reality, which is how the COMP shadowing hid.
        let probe = MixParams {
            s_db: args.s_db,
            db_min_window: args.db_min_window,
            s_db_ctg: args.s_db_ctg,
            s_db_comp: args.s_db_comp,
            s_db_comp_ctg: args.s_db_comp_ctg,
            p_convex: args.p_convex,
            p_convex_comp: args.p_convex_comp,
            p_mingen: args.p_mingen,
            p_mingen_comp: args.p_mingen_comp,
            db_prefixes: args.db_prefixes,
            db_prefixes_mix: args.db_prefixes_mix,
            db_prefixes_comp: args.db_prefixes_comp,
            ..MixParams::default()
        };
        let km = probe.db_knobs(DbMode::Mix);
        let kc = probe.db_knobs(DbMode::Compressing);
        println!(
            "[fmix] DB effective per mode: MIX p_convex={} s_db(cvx)={} s_db(ctg)={} p_mingen={} descent={} | COMP p_convex={} s_db(cvx)={} s_db(ctg)={} p_mingen={} descent={}",
            km.p_convex,
            km.s_db_cvx,
            km.s_db_ctg,
            km.p_mingen,
            km.prefixes,
            kc.p_convex,
            kc.s_db_cvx,
            kc.s_db_ctg,
            kc.p_mingen,
            kc.prefixes
        );
        if args.p_comp_g57 > 0.0 {
            println!(
                "[fmix] g57-only COMP attempts: p={} starting at s_db_g57={}",
                args.p_comp_g57, args.s_db_g57
            );
        }
    }
    if args.curated {
        println!(
            "[fmix] curated ON: ordinary expansion probes the CURATED store only (forward key, any size); compression {}",
            if args.curated_in_comp {
                "probes it too (--curated-in-comp)"
            } else {
                "stays regular-only"
            }
        );
    }
    if args.db_advance {
        println!(
            "[fmix] db-advance ON: splice products take the ballistic birth-advance (dir_q={})",
            args.dir_q
        );
    }
    if args.gen_target > 0 {
        println!(
            "[fmix] generation targeting ON: gen_target={} p_mingen={} pool_k={} gen_rescan={} gen_stop_frac={} twist_cov_stop={} split_rule={}",
            args.gen_target,
            args.p_mingen,
            args.pool_k,
            args.gen_rescan,
            args.gen_stop_frac,
            args.twist_cov_stop,
            if args.gen_split_inherit {
                "inherit"
            } else {
                "ratchet(+1)"
            }
        );
        if args.canary_theta > 0.0 {
            println!(
                "[fmix] canary ON: theta={} window={} qualifying rounds",
                args.canary_theta, args.canary_window
            );
        }
    }

    assert!(
        args.db_max_degree == 0 || args.db_max_degree <= 11,
        "--db-max-degree {} unusable: the degree probe caps subcube dimension \
         at 12, so caps above 11 would silently disable the guard",
        args.db_max_degree
    );
    let params = MixParams {
        k_max: args.k_max,
        split_damp: args.split_damp,
        split_base: args.split_base,
        dir_p: args.dir_p,
        dir_q: args.dir_q,
        target_size: target,
        mix_pay_random: args.mix_pay_random,
        prof_n: profile.map(|(n, _)| n).unwrap_or([0.0; 3]),
        prof_r: profile.map(|(_, r)| r).unwrap_or([0.0; 2]),
        prof_cadence_eff: args.prof_cadence_eff,
        prof_deadband: args.prof_deadband,
        prof_dp_max: args.prof_dp_max,
        prof_ewma: args.prof_ewma,
        prof_ki: args.prof_ki,
        temp: args.temp.unwrap_or(0.0),
        moves: args.moves,
        merge_reach: args.merge_reach,
        journal_len: args.journal_len,
        undo_frac: args.undo_frac,
        tabu_moves: args.tabu_moves,
        w_cross: args.w_cross,
        w_fresh: args.w_fresh,
        w_unsub: args.w_unsub,
        w_insert: args.w_insert,
        w_twist_neg: args.w_twist_neg,
        w_twist_swap: args.w_twist_swap,
        w_twist_cnot: args.w_twist_cnot,
        twist_neg_p: args.twist_neg_p,
        twist_g57: args.twist_g57,
        twist_min_len: args.twist_min_len,
        p_comp: args.p_comp,
        p_any: args.p_any,
        db_mode: local_mixing::db_mixing::db_replace::DbMode::parse(&args.db_mode)
            .unwrap_or_else(|| panic!("unknown --db-mode {} (mix|comp|any|stable|stable-grow|stable-ledger|same|band-ledger)", args.db_mode)),
        p_db: args.p_db,
        p_twist: args.p_twist,
        shuffle_rate: args.shuffle_rate,
        curated_exhaust: args.curated_exhaust,
        curated_in_comp: args.curated_in_comp,
        s_db: args.s_db,
        db_min_window: args.db_min_window,
        w_window: args.w_window,
        w_pool: args.w_pool,
        p_convex: args.p_convex,
        db_convex_p: args.db_convex_p,
        db_verify: !args.no_db_verify,
        db_dry_run: args.db_dry_run,
        db_max_degree: args.db_max_degree,
        db_degree_probes: args.db_degree_probes,
        db_max_span: args.db_max_span,
        db_wire_terms: args.db_wire_terms,
        db_total_terms: args.db_total_terms,
        db_prefixes: args.db_prefixes,
        db_advance: args.db_advance,
        p_pair: args.p_pair,
        pair_scan_cap: args.pair_scan_cap,
        pair_pick_uniform: args.pair_pick_uniform,
        p_bridge: args.p_bridge,
        bridge_min_span: args.bridge_min_span,
        bridge_max_span: args.bridge_max_span,
        bridge_max_colliders: args.bridge_max_colliders,
        p_mix: args.p_mix,
        s_db_comp: args.s_db_comp,
        p_convex_comp: args.p_convex_comp,
        s_db_ctg: args.s_db_ctg,
        s_db_comp_ctg: args.s_db_comp_ctg,
        db_prefixes_mix: args.db_prefixes_mix,
        db_prefixes_comp: args.db_prefixes_comp,
        p_mingen_comp: args.p_mingen_comp,
        curated: args.curated,
        ancestors: args.ancestors,
        anc_samples: args.anc_samples,
        anc_sample_seed: args.anc_sample_seed,
        p_comp_g57: args.p_comp_g57,
        contract_ceiling: args.contract_ceiling,
        size_hi: args.size_hi,
        size_lo: args.size_lo,
        comp_release_eps: args.comp_release_eps,
        comp_release_window: args.comp_release_window,
        s_db_g57: args.s_db_g57,
        gen_target: args.gen_target,
        p_mingen: args.p_mingen,
        pool_k: args.pool_k,
        canary_theta: args.canary_theta,
        canary_window: args.canary_window,
        litter_ban: args.litter_ban,
        litter_samples: args.litter_samples,
        twist_place_tries: args.twist_place_tries,
        gen_rescan: args.gen_rescan,
        gen_split_inherit: args.gen_split_inherit,
        gen_median_low: args.gen_median_low,
        gen_stop_frac: args.gen_stop_frac,
        twist_cov_stop: args.twist_cov_stop,
        gen_snap_every: args.gen_snap_every,
        snap_every_moves: args.snap_every_moves,
        split: args.split,
        split_stop: args.split_stop,
        p_split_twist: args.p_split_twist,
        p_join: args.p_join,
        split_fail_limit: args.split_fail_limit,
        split_canaries: args.split_canaries,
        split_reach_k: args.split_reach_k,
        p_mincross: args.p_mincross,
        cross_pool_k: args.cross_pool_k,
        cross_rescan: args.cross_rescan,
        verify_every: args.verify_every,
        report_every: args.report_every,
        local_verify: !args.no_local_verify,
        seed: args.seed,
    };
    assert!(
        args.anc_in.is_none() || args.resume.is_none(),
        "--anc-in seeds a FRESH run's ancestry; a --resume carries its own (drop one)"
    );
    assert!(
        args.anc_in.is_none() || (args.anc_samples == 0 && !args.ancestors),
        "--anc-in defines the ancestry universe; drop --ancestors / --anc-samples"
    );
    assert!(
        args.anc_out.is_none()
            || args.ancestors
            || args.anc_samples > 0
            || args.anc_in.is_some()
            || args.resume.is_some(),
        "--anc-out needs ancestry armed (--ancestors, --anc-samples or --anc-in): refusing now \
         rather than at the end of the run"
    );
    let mut mixer = match &args.resume {
        Some(path) => {
            let db = if params.p_comp > 0.0 || params.p_db > 0.0 || params.p_any > 0.0 {
                local_mixing::db_mixing::frozen::FrozenDb::from_env()
            } else {
                local_mixing::db_mixing::frozen::FrozenDb::empty()
            };
            let mx = Mixer::resume_state(path, params, db).expect("resume from state file");
            println!(
                "[fmix] RESUMED from {path}: {} gates at move {}, verifying against the original",
                mx.arena.len(),
                mx.moves_done
            );
            mx
        }
        None => match &args.anc_in {
            Some(p) => {
                let sc = Mixer::read_anc_sidecar(p).expect("read ancestry sidecar");
                let (mode_s, sc_m, sc_k, sc_n) = (
                    if sc.sampled { "sampled" } else { "exact" },
                    sc.m,
                    sc.tracers.len(),
                    sc.sets.len(),
                );
                // Construct with ancestry OFF: the sidecar defines the
                // universe, and the constructor would otherwise pick its own
                // tracers against the wrong input population.
                let mut mx = Mixer::new(
                    gates,
                    num_wires,
                    MixParams {
                        ancestors: false,
                        anc_samples: 0,
                        ..params
                    },
                );
                mx.import_ancestry(sc);
                println!(
                    "[fmix] ancestry IMPORTED from {p}: {mode_s} mode, universe m={sc_m}, K={sc_k}, {sc_n} per-gate sets; anc meters continue the PRODUCING run's clock"
                );
                mx
            }
            None => Mixer::new(gates, num_wires, params),
        },
    };
    // External litter assignment (e.g. the SGDB substitution's sidecar: every
    // replaced gate's block is one litter, so --litter-ban covers the INITIAL
    // replacements too, not only the walk's own splices).
    if let Some(p) = &args.litter_in {
        let ids: Vec<u64> = std::fs::read_to_string(p)
            .unwrap_or_else(|e| panic!("read --litter-in {p}: {e}"))
            .lines()
            .skip(1) // header "litter1 N"
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.trim().parse().expect("bad litter id"))
            .collect();
        mixer.load_litters(&ids);
    }
    // A resumed run has no input file, so the summary's "before" figures must
    // come from the resumed circuit or it reports 0 -> N and reads as if the
    // chain started from nothing.
    let (input_len, comp0) = match &args.resume {
        Some(_) => (mixer.arena.len(), mixer.remaining_g57()),
        None => (input_len, comp0),
    };
    if let Some(path) = &args.db_record {
        mixer.enable_db_record(path);
        println!("[fmix] recording DB attempts to {path}");
    }
    if args.gen_snap_every > 0 || args.snap_every_moves > 0 {
        let base = args
            .output
            .clone()
            .unwrap_or_else(|| "fmix_out".to_string());
        if args.gen_snap_every > 0 {
            println!(
                "[fmix] gen snapshots armed: every {} circuit generations -> {base}.gen<m>.mpmct1",
                args.gen_snap_every
            );
        }
        if args.snap_every_moves > 0 {
            println!(
                "[fmix] move snapshots armed: every {} moves -> {base}.mv<m>.mpmct1",
                args.snap_every_moves
            );
        }
        mixer.set_gen_snap_base(base);
    }

    let stop = std::env::var("FMIX_STOP_FLAG")
        .ok()
        .filter(|s| !s.is_empty());
    let dump = std::env::var("FMIX_DUMP_FLAG")
        .ok()
        .filter(|s| !s.is_empty());
    if stop.is_some() || dump.is_some() {
        let dump_out = std::env::var("FMIX_DUMP_OUT")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| {
                args.output
                    .as_deref()
                    .map(|o| format!("{o}.snapshot.txt"))
                    .unwrap_or_else(|| "fmix_snapshot.txt".to_string())
            });
        if let Some(f) = &stop {
            println!("[fmix] stop flag armed: touch {f} -> clean finish");
        }
        if let Some(f) = &dump {
            println!("[fmix] dump signal armed: touch {f} -> snapshot to {dump_out}");
        }
        mixer.enable_flags(stop, dump, dump_out);
    }

    let t0 = std::time::Instant::now();
    let stop_reason = mixer.run();
    let secs = t0.elapsed().as_secs_f64();
    mixer.report();
    // Canary dump for runs that stopped BEFORE the stage boundary (budget,
    // stop flag); a no-op when the boundary already reported them.
    mixer.split_tap_summary();
    {
        use std::sync::atomic::Ordering;
        let rl = local_mixing::circuit::CANON_RULE_L_SKIPS.load(Ordering::Relaxed);
        let mc = local_mixing::circuit::CANON_CAP_SKIPS.load(Ordering::Relaxed);
        let rlb = local_mixing::circuit::CANON4_RULE_L_BRANCHES.load(Ordering::Relaxed);
        let rlc = local_mixing::circuit::CANON4_RULE_L_CALLS.load(Ordering::Relaxed);
        println!(
            "[fmix] canon caps: rule_l_skips={rl} monomial_skips={mc} rule_l_calls={rlc} rule_l_branches={rlb}"
        );
    }
    println!(
        "[fmix] chain done in {:.1}s: {} ({} -> {} gates, {} -> {} g57 fossils)",
        secs,
        match stop_reason {
            MixStop::MovesBudget => "moves budget spent",
            MixStop::StopFlag => "stop flag",
            MixStop::DoseReached => "dose reached (gen + twist coverage targets met)",
            MixStop::CanaryFired => "canary fired (pool is unspellable by the store)",
            MixStop::ProfileDone => "profile complete (size schedule finished)",
            MixStop::SplitDone => "split stage complete (stopped at the stage boundary)",
        },
        input_len,
        mixer.arena.len(),
        comp0,
        mixer.remaining_g57(),
    );

    if !args.skip_final_float {
        let t1 = std::time::Instant::now();
        let (moved, disp) = mixer.final_float();
        mixer.global_check();
        println!(
            "[fmix] final float: {} gates moved, {} total displacement, {:.1}s (verified)",
            moved,
            disp,
            t1.elapsed().as_secs_f64()
        );
    }

    if let Some(sp) = &args.state_out {
        mixer.save_state(sp).expect("write state file");
        println!("[fmix] wrote resume state to {sp}");
    }
    if let Some(out) = &args.output {
        let final_gates = mixer.arena.to_vec();
        format::write_mpmct(out, &final_gates, num_wires).expect("write output");
        println!("[fmix] wrote {} gates to {}", final_gates.len(), out);
    } else {
        println!("[fmix] no --output given; result discarded after verification");
    }

    if let Some(path) = &args.origins_out {
        let origins = mixer.origins_in_order();
        let mut s = String::with_capacity(origins.len() * 8);
        for o in origins {
            s.push_str(&format!("{o}\n"));
        }
        std::fs::write(path, s).expect("write origins");
        println!("[fmix] wrote origins sidecar to {path} (synthetic = {ORIGIN_SYNTH})");
    }

    if let Some(path) = &args.gens_out {
        let gens = mixer.gens_in_order();
        let mut s = String::with_capacity(gens.len() * 4);
        for g in gens {
            s.push_str(&format!("{g}\n"));
        }
        std::fs::write(path, s).expect("write gens");
        println!("[fmix] wrote gens sidecar to {path} (born-random = {GEN_FRESH})");
    }

    if let Some(path) = &args.anc_out {
        // After the final float, so line i is gate i of the written circuit.
        mixer
            .write_anc_sidecar(path)
            .expect("write ancestry sidecar");
        println!(
            "[fmix] wrote ancestry sidecar to {path} ({} per-gate sets; import with --anc-in)",
            mixer.arena.len()
        );
    }
}
