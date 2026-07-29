// Post-fsplit mixing chain driver. Takes an mpmct1 circuit (typically fsplit
// output; g57 also accepted) and runs a randomized, size-thermostatted local
// rewrite walk: R-rule crossings, fresh-wire case splits, unsubsume splits,
// copy-pair insertions and conjugation twists (--w-twist-neg / --w-twist-swap:
// bracket a window with a wire negation or a 3-CNOT wire swap and conjugate
// its interior — state/progress mixing, the SAMF mechanism XGate-native)
// expand; catalogue merges (cancel / X-fuse / drop-literal / subsume)
// contract. The thermostat holds the gate count near --target-size; the
// objective is churn (distance from the original description), not size.
// Never emits comp=1 gates, so the g57 "fossil" count is monotone. Ends with a
// final uniform float, a sampled global check against the input, and an mpmct1
// write.
//
// Graceful stop: touch $FMIX_STOP_FLAG and the run finishes cleanly (verified
// write) at the next report point. Snapshot: touch $FMIX_DUMP_FLAG to get a
// verified mid-run circuit at $FMIX_DUMP_OUT (default <output>.snapshot.txt).
//
// Example:
//   FMIX_STOP_FLAG=/tmp/fmix.stop fmix --input mixed_fsplit.txt \
//     --target-size 3000000 --moves 50000000 --output mixed_fmix.txt
use clap::Parser;
use local_mixing::postmix::format;
use local_mixing::postmix::mix::{MixParams, MixStop, Mixer, GEN_FRESH, ORIGIN_SYNTH};
use local_mixing::postmix::xgate::{XGate, max_wire};

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
    /// Minimum twist window length (max is the current circuit size)
    #[arg(long, default_value_t = 64)]
    twist_min_len: usize,
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
    /// Window length the descent STARTS from. The descent visits every shorter
    /// length down to 1, so there is no separate minimum.
    #[arg(long, default_value_t = 5)]
    s_db: usize,
    /// Probability the window sampler is convex rather than contiguous.
    #[arg(long, default_value_t = 0.5)]
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
    /// Largest-first prefix descent: try the full sampled window, then its
    /// len-1 prefix, etc. down to --db-min-window, splicing the LONGEST
    /// matching prefix (live) or recording every prefix (with --db-dry-run).
    /// Span/verify declines keep descending — shorter prefixes may still fit.
    #[arg(long, default_value_t = false)]
    db_prefixes: bool,
    /// Give DB splice products the ballistic birth-advance that split pieces
    /// get: each product floats floor(dir_q * slack) along its own direction.
    /// Without this a splice assigns directions nothing acts on, so a
    /// DB-dominated schedule has no directional transport at all. The other
    /// source of transport -- crossings -- widens gates, and width is what
    /// kills DB matching. Off by default (it changes trajectories).
    #[arg(long, default_value_t = false)]
    db_advance: bool,
    /// Also probe the curated store (FROZEN_CURATED_DIR) and prefer a
    /// non-identical curated match over a regular one REGARDLESS OF SIZE. The
    /// curated store holds circuits every strict subcircuit of which is
    /// shortest, so its replacements are routes fcompress cannot partially
    /// undo -- but it is built from splits of minimal identities and so holds
    /// LONGER equivalents, meaning this deliberately prefers growth.
    /// Compressing mode ignores it. Requires FROZEN_CURATED_DIR.
    #[arg(long, default_value_t = false)]
    curated: bool,
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
    /// Resume from a state file instead of --input. Parameters still come from
    /// the command line, so a paused run can be re-steered; only the state
    /// VERSION must match, since field meanings would otherwise drift silently.
    #[arg(long)]
    resume: Option<String>,
}

fn main() {
    let args = Args::parse();

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
    if args.p_twist > 0.0 {
        println!(
            "[fmix] first-class twist rounds ON: p_twist={} (w-twist-* weights serve as type ratios)",
            args.p_twist
        );
    }
    if args.w_twist_neg > 0.0 || args.w_twist_swap > 0.0 || args.w_twist_cnot > 0.0 {
        println!(
            "[fmix] twists ON: w_twist_neg={} w_twist_swap={} w_twist_cnot={} twist_min_len={}",
            args.w_twist_neg, args.w_twist_swap, args.w_twist_cnot, args.twist_min_len
        );
    }
    if args.p_comp > 0.0 || args.p_db > 0.0 || args.p_any > 0.0 {
        println!(
            "[fmix] DB ON: p_db(slot2)={} db_mode={} p_comp(contract)={} p_any(expand)={} s_db={} p_convex={} w_window={} w_pool={} verify={} curated={} (FROZEN_DB_DIR required)",
            args.p_db, args.db_mode, args.p_comp, args.p_any, args.s_db, args.p_convex,
            args.w_window, args.w_pool, !args.no_db_verify, args.curated
        );
        if args.p_comp_g57 > 0.0 {
            println!(
                "[fmix] g57-only COMP attempts: p={} starting at s_db_g57={}",
                args.p_comp_g57, args.s_db_g57
            );
        }
    }
    if let Some(d) = &args.frozen_db_dir {
        unsafe { std::env::set_var("FROZEN_DB_DIR", d) };
    }
    if let Some(d) = &args.frozen_curated_dir {
        unsafe { std::env::set_var("FROZEN_CURATED_DIR", d) };
    }
    if args.curated {
        assert!(
            std::env::var("FROZEN_CURATED_DIR").is_ok(),
            "--curated needs FROZEN_CURATED_DIR; refusing to run, because \
             degrading silently to regular-only would look like a measurement"
        );
        println!("[fmix] curated ON: curated matches preferred over regular, any size");
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
            args.gen_target, args.p_mingen, args.pool_k, args.gen_rescan, args.gen_stop_frac,
            args.twist_cov_stop,
            if args.gen_split_inherit { "inherit" } else { "ratchet(+1)" }
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
        twist_min_len: args.twist_min_len,
        p_comp: args.p_comp,
        p_any: args.p_any,
        db_mode: local_mixing::postmix::db_replace::DbMode::parse(&args.db_mode)
            .unwrap_or_else(|| panic!("unknown --db-mode {} (mix|comp|any)", args.db_mode)),
        p_db: args.p_db,
        p_twist: args.p_twist,
        s_db: args.s_db,
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
        curated: args.curated,
        ancestors: args.ancestors,
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
        verify_every: args.verify_every,
        report_every: args.report_every,
        local_verify: !args.no_local_verify,
        seed: args.seed,
    };
    let mut mixer = match &args.resume {
        Some(path) => {
            let db = if params.p_comp > 0.0 || params.p_db > 0.0 || params.p_any > 0.0 {
                local_mixing::replace::frozen::FrozenDb::from_env()
            } else {
                local_mixing::replace::frozen::FrozenDb::empty()
            };
            let mx = Mixer::resume_state(path, params, db).expect("resume from state file");
            println!(
                "[fmix] RESUMED from {path}: {} gates at move {}, verifying against the original",
                mx.arena.len(),
                mx.moves_done
            );
            mx
        }
        None => Mixer::new(gates, num_wires, params),
    };
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
        let base = args.output.clone().unwrap_or_else(|| "fmix_out".to_string());
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

    let stop = std::env::var("FMIX_STOP_FLAG").ok().filter(|s| !s.is_empty());
    let dump = std::env::var("FMIX_DUMP_FLAG").ok().filter(|s| !s.is_empty());
    if stop.is_some() || dump.is_some() {
        let dump_out = std::env::var("FMIX_DUMP_OUT").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| {
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
    {
        use std::sync::atomic::Ordering;
        let rl = local_mixing::circuit::circuit::CANON_RULE_L_SKIPS.load(Ordering::Relaxed);
        let mc = local_mixing::circuit::circuit::CANON_CAP_SKIPS.load(Ordering::Relaxed);
        let rlb = local_mixing::circuit::circuit::CANON4_RULE_L_BRANCHES.load(Ordering::Relaxed);
        let rlc = local_mixing::circuit::circuit::CANON4_RULE_L_CALLS.load(Ordering::Relaxed);
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
}
