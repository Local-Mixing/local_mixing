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
    /// Input circuit file
    #[arg(long)]
    input: String,
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
    /// Compressing DB move: probability that a CONTRACTION attempt first samples
    /// a contiguous window and replaces it with a non-growing equivalent from the
    /// store (uniform among the shortest), falling through to undo/merge on a
    /// miss. 0 = off. Requires FROZEN_DB_DIR (and optionally FROZEN_CURATED_DIR)
    /// when > 0 or --p-db > 0. Handles conjunction-control gates, not just g57s.
    #[arg(long, default_value_t = 0.0)]
    w_db: f64,
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
    /// Anneal p-db linearly across the move budget, ending at this value
    /// (the "splitting phase" lever: retire the DB growth engine as material
    /// turns wide). Negative = no anneal.
    #[arg(long, default_value_t = -1.0)]
    p_db_final: f64,
    /// Size-steer the agnostic move: multiply the (annealed) p-db by
    /// sigmoid(-(size-target)/temp), so one run grows to target and then
    /// holds size (pair with --w-db for the shrink side). Reported as pdb=
    /// in every progress line.
    #[arg(long, default_value_t = false)]
    p_db_steer: bool,
    /// Minimum DB-replacement window length (gates).
    #[arg(long, default_value_t = 2)]
    db_min_window: usize,
    /// Maximum DB-replacement window length (gates).
    #[arg(long, default_value_t = 12)]
    db_max_window: usize,
    /// Window sampling geometry: `contiguous` (a gate plus its neighbors in its
    /// own direction) or `convex` (float a block together, absorbing colliders).
    #[arg(long, default_value = "contiguous")]
    db_sample: String,
    /// Control cap L for window building: a candidate gate with more than L
    /// controls is evaded (floated away, else the build reverses, else aborts),
    /// keeping high-degree always-miss gates out of the window. 0 = no cap.
    #[arg(long, default_value_t = 0)]
    db_ctrl_cap: usize,
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
    /// Generation targeting: drive every (ctrl-cap-eligible) gate through at
    /// least this many DB re-encodings. Each gate carries a generation (input
    /// gates 0; a DB splice stamps min(window)+1; splits/merges propagate;
    /// fresh insert/bracket material counts as done). DB seeds are drawn from
    /// the below-target gates with probability --gen-bias, replacing the
    /// coupon-collector tail of uniform selection with direct work. 0 = off.
    #[arg(long, default_value_t = 0)]
    gen_target: u32,
    /// Probability a DB seed is drawn from the laggard (below-target) list
    /// instead of uniformly; the remainder keeps the unbiased churn.
    #[arg(long, default_value_t = 0.9)]
    gen_bias: f64,
    /// Ingest-then-pay, CHEAP channel: probability a round is a
    /// Compressing-mode DB attempt (non-growing replacements only — zero
    /// growth risk, safe to run hot) seeded on a cheap-tier laggard. A seed
    /// that keeps failing graduates to the hard tier at --gen-miss-budget.
    /// 0 = off. Requires --gen-target > 0.
    #[arg(long, default_value_t = 0.0)]
    p_db_ingest: f64,
    /// Ingest-then-pay, PAID channel: probability a round is a MinGrow-mode
    /// attempt (uniform among the SHORTEST equivalents, growing allowed)
    /// seeded on a hard-tier gate — growth is spent only on gates the cheap
    /// channel provably cannot ingest, at the minimum spelling each, and the
    /// cost is ledgered in the report's paid= field. Fires only while
    /// hard-tier gates exist. 0 = off.
    #[arg(long, default_value_t = 0.0)]
    p_db_hard: f64,
    /// Seed misses before a laggard graduates cheap -> hard tier.
    #[arg(long, default_value_t = 6)]
    gen_miss_budget: u16,
    /// Seed misses before a gate is written off as unreachable (excluded
    /// from targeting and from the dose stop, reported as u=). 0 = never.
    #[arg(long, default_value_t = 0)]
    gen_giveup: u16,
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
}

fn main() {
    let args = Args::parse();

    let (gates, file_wires): (Vec<XGate>, usize) = match args.input_format.as_str() {
        "mpmct1" => format::read_mpmct(&args.input).expect("read mpmct1 circuit"),
        "g57" => {
            let g = format::read_g57_file(&args.input).expect("read g57 circuit");
            let w = max_wire(&g) as usize + 1;
            (g, w)
        }
        other => panic!("unknown --input-format {other}"),
    };
    let num_wires = file_wires.max(max_wire(&gates) as usize + 1);
    let input_len = gates.len();
    let comp0 = gates.iter().filter(|g| g.comp).count();
    let target = args.target_size.unwrap_or(input_len);
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
    if args.w_db > 0.0 || args.p_db > 0.0 || args.p_db_final > 0.0 {
        println!(
            "[fmix] DB replacement ON: w_db(compress)={} p_db(agnostic)={} p_db_final={} steer={} window=[{},{}] verify={} (FROZEN_DB_DIR required)",
            args.w_db, args.p_db, args.p_db_final, args.p_db_steer,
            args.db_min_window, args.db_max_window, !args.no_db_verify
        );
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
            "[fmix] generation targeting ON: gen_target={} gen_bias={} gen_rescan={} gen_stop_frac={} twist_cov_stop={} split_rule={} (report: gen tgt/G/alag/lag/c/h/u/wlag/min, cov, ing, hard, paid)",
            args.gen_target, args.gen_bias, args.gen_rescan, args.gen_stop_frac,
            args.twist_cov_stop,
            if args.gen_split_inherit { "inherit" } else { "ratchet(+1)" }
        );
        if args.p_db_ingest > 0.0 || args.p_db_hard > 0.0 {
            println!(
                "[fmix] ingest-then-pay ON: p_db_ingest={} (cheap, non-growing) p_db_hard={} (paid, MinGrow) miss_budget={} giveup={}",
                args.p_db_ingest, args.p_db_hard, args.gen_miss_budget, args.gen_giveup
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
        w_db: args.w_db,
        p_db: args.p_db,
        p_twist: args.p_twist,
        p_db_final: args.p_db_final,
        p_db_steer: args.p_db_steer,
        db_min_window: args.db_min_window,
        db_max_window: args.db_max_window,
        db_sample: local_mixing::postmix::mix::DbSample::parse(&args.db_sample)
            .unwrap_or_else(|| panic!("unknown --db-sample {} (contiguous|convex)", args.db_sample)),
        db_ctrl_cap: args.db_ctrl_cap,
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
        gen_target: args.gen_target,
        gen_bias: args.gen_bias,
        gen_rescan: args.gen_rescan,
        p_db_ingest: args.p_db_ingest,
        p_db_hard: args.p_db_hard,
        gen_miss_budget: args.gen_miss_budget,
        gen_giveup: args.gen_giveup,
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
    let mut mixer = Mixer::new(gates, num_wires, params);
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
