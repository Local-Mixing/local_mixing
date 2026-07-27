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
use clap::error::ErrorKind;
use clap::{CommandFactory, Parser, ValueEnum};
use local_mixing::postmix::format;
use local_mixing::postmix::g57_pairs::{G57PairRegion, G57PairSeedConfig};
use local_mixing::postmix::mix::{
    DbSample, FmixG57PairSeedReport, GEN_FRESH, MixParams, MixStop, Mixer, ORIGIN_SYNTH,
};
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
    /// Absolute transactional gate-count ceiling. Zero disables the ceiling.
    /// Growth moves whose complete replacement would exceed it are rejected.
    #[arg(long, default_value_t = 0)]
    hard_size_cap: usize,
    /// Thermostat softness in gates (default: max(target/100, 64))
    #[arg(long)]
    temp: Option<f64>,
    /// Total move attempts for the legacy one-phase schedule (default:
    /// 1,000,000). Conflicts with --grow-moves/--churn-moves.
    #[arg(long, conflicts_with_all = ["grow_moves", "churn_moves"])]
    moves: Option<u64>,
    /// Move attempts in the growth phase. Must be paired with --churn-moves.
    /// The phase uses the ordinary target, thermostat, and move weights.
    #[arg(long, requires = "churn_moves")]
    grow_moves: Option<u64>,
    /// Additional move attempts in a fixed-target churn phase. Must be paired
    /// with --grow-moves. The same Mixer, RNG stream, provenance, undo journal,
    /// and tabu history continue across the phase boundary.
    #[arg(long, requires = "grow_moves")]
    churn_moves: Option<u64>,
    /// Treat the growth phase as new-SSS dose acquisition: transition to churn
    /// as soon as --gen-stop-frac/--twist-cov-stop are met, then run exactly
    /// --churn-moves additional moves on the same Mixer. The dose-only ingest
    /// and paid DB channels are disabled at the transition, and final output
    /// is refused if the generation laggard fraction regresses.
    #[arg(
        long,
        default_value_t = false,
        requires_all = ["grow_moves", "churn_moves", "gen_target"]
    )]
    dose_then_churn: bool,
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
    /// Persistent nonlinear-frame weight. A frame opens a reversible packet
    /// of nonlinear controlled-X gates, shoots/conjugates it through a window,
    /// and closes with its inverse. Zero preserves the historical trajectory.
    #[arg(long, default_value_t = 0.0)]
    w_nl_frame: f64,
    /// Minimum control width of a nonlinear frame gate. Width 2 is the first
    /// genuinely nonlinear (Toffoli-class) setting.
    #[arg(long, default_value_t = 2)]
    nl_frame_min_width: usize,
    /// Maximum control width of a nonlinear frame gate (must not exceed
    /// --k-max).
    #[arg(long, default_value_t = 3)]
    nl_frame_max_width: usize,
    /// Number of nonlinear controlled-X gates in each reversible frame packet.
    #[arg(long, default_value_t = 16)]
    nl_frame_packet_gates: usize,
    /// Number of transport shots attempted while a nonlinear frame is open.
    #[arg(long, default_value_t = 64)]
    nl_frame_shots: usize,
    /// Minimum move age before a nonlinear frame may be locally undone or
    /// sibling-merged.
    #[arg(long, default_value_t = 100_000)]
    nl_frame_tenure: u64,
    /// Minimum twist window length (max is the current circuit size)
    #[arg(long, default_value_t = 64)]
    twist_min_len: usize,
    /// Compressing DB move: probability that a contraction attempt first
    /// samples a window and replaces it with a non-growing equivalent.
    #[arg(long, default_value_t = 0.0)]
    w_db: f64,
    /// Size-agnostic DB replacement probability per top-level round.
    #[arg(long, default_value_t = 0.0)]
    p_db: f64,
    /// Fixed top-level conjugation-twist probability per round. Twist kind is
    /// selected using the --w-twist-* values as relative weights.
    #[arg(long, default_value_t = 0.0)]
    p_twist: f64,
    /// Anneal --p-db linearly to this probability over the move budget.
    /// Negative disables annealing.
    #[arg(long, default_value_t = -1.0)]
    p_db_final: f64,
    /// Multiply the current --p-db by a below-target sigmoid.
    #[arg(long, default_value_t = false)]
    p_db_steer: bool,
    /// Minimum DB-replacement window length.
    #[arg(long, default_value_t = 2)]
    db_min_window: usize,
    /// Maximum DB-replacement window length.
    #[arg(long, default_value_t = 12)]
    db_max_window: usize,
    /// DB window sampler: contiguous | convex | mixed.
    #[arg(long, default_value = "contiguous")]
    db_sample: String,
    /// Maximum controls on a DB-window gate. Zero disables this guard.
    #[arg(long, default_value_t = 0)]
    db_ctrl_cap: usize,
    /// Convex sampler's probability of growing in the seed gate's direction.
    #[arg(long, default_value_t = 0.75)]
    db_convex_p: f64,
    /// Skip exhaustive local equivalence verification for DB splices.
    #[arg(long, default_value_t = false)]
    no_db_verify: bool,
    /// Record every DB replacement attempt to this file.
    #[arg(long)]
    db_record: Option<String>,
    /// Sample and record DB matches without modifying the circuit.
    #[arg(long, default_value_t = false)]
    db_dry_run: bool,
    /// Reject a DB lookup whose sampled function exceeds this ANF degree.
    /// Zero disables the guard; the supported maximum is 11.
    #[arg(long, default_value_t = 0)]
    db_max_degree: usize,
    /// Random subcubes tested in each direction by the degree guard.
    #[arg(long, default_value_t = 6)]
    db_degree_probes: usize,
    /// Reject a DB lookup touching more than this many wires. Zero disables.
    #[arg(long, default_value_t = 0)]
    db_max_span: usize,
    /// Per-wire polynomial-term budget. Zero selects the core default.
    #[arg(long, default_value_t = 0)]
    db_wire_terms: usize,
    /// Total polynomial-term budget. Zero selects the core default.
    #[arg(long, default_value_t = 0)]
    db_total_terms: usize,
    /// Try sampled-window prefixes largest first down to --db-min-window.
    #[arg(long, default_value_t = false)]
    db_prefixes: bool,
    /// Drive eligible gates through at least this many DB re-encodings.
    /// Zero disables generation targeting.
    #[arg(long, default_value_t = 0)]
    gen_target: u32,
    /// Probability of selecting a below-target generation laggard as DB seed.
    #[arg(long, default_value_t = 0.9)]
    gen_bias: f64,
    /// Laggard-list rebuild cadence in moves.
    #[arg(long, default_value_t = 10_000)]
    gen_rescan: u64,
    /// Cheap ingest channel probability: non-growing DB replacement seeded
    /// from below-target cheap-tier gates.
    #[arg(long, default_value_t = 0.0)]
    p_db_ingest: f64,
    /// Paid hard channel probability: minimum-growth DB replacement seeded
    /// from hard-tier gates.
    #[arg(long, default_value_t = 0.0)]
    p_db_hard: f64,
    /// Seed misses before a laggard graduates from cheap to hard tier.
    #[arg(long, default_value_t = 6)]
    gen_miss_budget: u16,
    /// Seed misses before a laggard is marked unreachable. Zero means never.
    #[arg(long, default_value_t = 0)]
    gen_giveup: u16,
    /// Have split children inherit the parent generation instead of +1.
    #[arg(long, default_value_t = false)]
    gen_split_inherit: bool,
    /// Stamp DB replacements from the lower rather than upper median.
    #[arg(long, default_value_t = false)]
    gen_median_low: bool,
    /// Stop at a report point when the below-target fraction among targetable
    /// gates (cap-eligible and not retired by --gen-giveup) is at most this
    /// and --twist-cov-stop is met. Negative disables the dose stop.
    #[arg(long, default_value_t = -1.0)]
    gen_stop_frac: f64,
    /// Use only live gates without an identifiable seeded-G57-pair frame tag
    /// as the scope for generation dose stop and final revalidation. Within
    /// that scope, only targetable gates form the dose denominator. The
    /// historical all-gate and all-non-pair censuses are still reported.
    #[arg(long, default_value_t = false)]
    gen_dose_exclude_g57_pair_frames: bool,
    /// Minimum cumulative twisted-span/current-size coverage for dose stop.
    #[arg(long, default_value_t = 0.0)]
    twist_cov_stop: f64,
    /// Write final per-gate DB-generation stamps, one per line.
    #[arg(long)]
    gens_out: Option<String>,
    /// Churn-phase target (default: inherit --target-size).
    #[arg(long, requires = "churn_moves")]
    churn_target_size: Option<usize>,
    /// Churn-phase thermostat softness (default: inherit, or recompute the
    /// standard default when --churn-target-size changes).
    #[arg(long, requires = "churn_moves")]
    churn_temp: Option<f64>,
    /// Churn-phase merge search distance (default: inherit).
    #[arg(long, requires = "churn_moves")]
    churn_merge_reach: Option<usize>,
    /// Churn-phase journal-undo fraction (default: inherit).
    #[arg(long, requires = "churn_moves")]
    churn_undo_frac: Option<f64>,
    /// Churn-phase split-event refractory period (default: inherit).
    #[arg(long, requires = "churn_moves")]
    churn_tabu_moves: Option<u64>,
    /// Churn-phase expansion weights (each defaults to its growth value).
    #[arg(long, requires = "churn_moves")]
    churn_w_cross: Option<f64>,
    #[arg(long, requires = "churn_moves")]
    churn_w_fresh: Option<f64>,
    #[arg(long, requires = "churn_moves")]
    churn_w_unsub: Option<f64>,
    #[arg(long, requires = "churn_moves")]
    churn_w_insert: Option<f64>,
    #[arg(long, requires = "churn_moves")]
    churn_w_twist_neg: Option<f64>,
    #[arg(long, requires = "churn_moves")]
    churn_w_twist_swap: Option<f64>,
    #[arg(long, requires = "churn_moves")]
    churn_w_twist_cnot: Option<f64>,
    #[arg(long, requires = "churn_moves")]
    churn_w_nl_frame: Option<f64>,
    /// Churn-phase twist/frame geometry (each defaults to its growth value).
    #[arg(long, requires = "churn_moves")]
    churn_twist_min_len: Option<usize>,
    #[arg(long, requires = "churn_moves")]
    churn_nl_frame_min_width: Option<usize>,
    #[arg(long, requires = "churn_moves")]
    churn_nl_frame_max_width: Option<usize>,
    #[arg(long, requires = "churn_moves")]
    churn_nl_frame_packet_gates: Option<usize>,
    #[arg(long, requires = "churn_moves")]
    churn_nl_frame_shots: Option<usize>,
    #[arg(long, requires = "churn_moves")]
    churn_nl_frame_tenure: Option<u64>,
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
    /// Inject/fragment native-G57 identity pairs immediately before or after
    /// the ordinary fmix walk.
    #[arg(
        long,
        value_enum,
        requires_all = [
            "g57_pairs_per_wire",
            "g57_pair_target_wires",
            "g57_pair_control_wire_limit",
            "g57_pair_region"
        ]
    )]
    g57_pair_stage: Option<G57PairStage>,
    #[arg(long, requires = "g57_pair_stage")]
    g57_pairs_per_wire: Option<usize>,
    #[arg(long, requires = "g57_pair_stage")]
    g57_pair_target_wires: Option<usize>,
    /// Controls are limited to 0..LIMIT independently of the target range.
    #[arg(long, requires = "g57_pair_stage")]
    g57_pair_control_wire_limit: Option<usize>,
    #[arg(long, requires = "g57_pair_stage")]
    g57_pair_region: Option<G57PairRegion>,
    /// Dedicated content/anchor seed (default: inherit --seed).
    #[arg(long, requires = "g57_pair_stage")]
    g57_pair_seed: Option<u64>,
    /// Per-pair TSV evidence (default derived from --output and stage).
    #[arg(long, requires = "g57_pair_stage")]
    g57_pair_manifest: Option<String>,
    /// Write the final live G57-pair-frame and dual generation census as a
    /// machine-readable key=value sidecar.
    #[arg(long, requires = "g57_pair_stage")]
    g57_pair_census_out: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum G57PairStage {
    Pre,
    Post,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunSchedule {
    /// Historical one-phase walk. Keeping this as a distinct path ensures an
    /// invocation with no schedule flags makes exactly one Mixer::run call.
    Legacy { moves: u64 },
    /// Two cumulative budgets on one Mixer. The second run starts at
    /// `grow_moves` and stops at `grow_moves + churn_moves`.
    GrowThenChurn {
        grow_moves: u64,
        churn_moves: u64,
        total_moves: u64,
    },
}

impl RunSchedule {
    fn from_args(args: &Args) -> Result<Self, String> {
        match (args.grow_moves, args.churn_moves) {
            (None, None) => Ok(Self::Legacy {
                moves: args.moves.unwrap_or(1_000_000),
            }),
            (Some(grow_moves), Some(churn_moves)) => {
                if grow_moves == 0 || churn_moves == 0 {
                    return Err("--grow-moves and --churn-moves must both be positive".into());
                }
                let total_moves = grow_moves.checked_add(churn_moves).ok_or_else(|| {
                    "--grow-moves + --churn-moves overflows the u64 move clock".to_string()
                })?;
                Ok(Self::GrowThenChurn {
                    grow_moves,
                    churn_moves,
                    total_moves,
                })
            }
            // clap's reciprocal `requires` declarations reject these before
            // main, but keep this total for direct construction/unit tests.
            _ => Err("--grow-moves and --churn-moves must be supplied together".into()),
        }
    }

    fn first_budget(self) -> u64 {
        match self {
            Self::Legacy { moves } => moves,
            Self::GrowThenChurn { grow_moves, .. } => grow_moves,
        }
    }

    fn total_budget(self) -> u64 {
        match self {
            Self::Legacy { moves } => moves,
            Self::GrowThenChurn { total_moves, .. } => total_moves,
        }
    }
}

impl Args {
    fn validate(&self) -> Result<(), String> {
        let validate_probability = |name: &str, value: f64| {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(format!("{name} must be finite and in [0, 1]"));
            }
            Ok(())
        };
        let validate_weights = |phase: &str, weights: &[(&str, f64)]| {
            let mut total = 0.0;
            for (name, value) in weights {
                if !value.is_finite() || *value < 0.0 {
                    return Err(format!(
                        "{phase} {name} weight must be finite and nonnegative"
                    ));
                }
                total += value;
            }
            if !total.is_finite() {
                return Err(format!("{phase} expansion-weight sum must be finite"));
            }
            Ok(())
        };
        let validate_frame = |phase: &str, weight: f64, min: usize, max: usize, packet: usize| {
            if !weight.is_finite() || weight < 0.0 {
                return Err(format!(
                    "{phase} nonlinear-frame weight must be finite and nonnegative"
                ));
            }
            if weight == 0.0 {
                // Preserve historical configurations such as --k-max 1:
                // dormant frame geometry has no effect on the walk.
                return Ok(());
            }
            if min < 2 {
                return Err(format!(
                    "{phase} nonlinear-frame minimum width must be at least 2"
                ));
            }
            if min > max {
                return Err(format!(
                    "{phase} nonlinear-frame width range is invalid: min {min} > max {max}"
                ));
            }
            if max > self.k_max {
                return Err(format!(
                    "{phase} nonlinear-frame maximum width ({max}) exceeds --k-max ({})",
                    self.k_max
                ));
            }
            if packet == 0 {
                return Err(format!(
                    "{phase} nonlinear-frame packet gate count must be positive"
                ));
            }
            Ok(())
        };

        if matches!(self.target_size, Some(0)) {
            return Err("--target-size must be positive".into());
        }
        if self.hard_size_cap > 0
            && self
                .target_size
                .is_some_and(|target| target > self.hard_size_cap)
        {
            return Err(format!(
                "--target-size ({}) exceeds --hard-size-cap ({})",
                self.target_size.expect("checked Some"),
                self.hard_size_cap
            ));
        }
        if matches!(self.temp, Some(t) if t <= 0.0 || !t.is_finite()) {
            return Err("--temp must be finite and positive".into());
        }
        if !self.split_base.is_finite() || self.split_base <= 0.0 {
            return Err("--split-base must be finite and positive".into());
        }
        validate_probability("--dir-p", self.dir_p)?;
        validate_probability("--dir-q", self.dir_q)?;
        validate_probability("--undo-frac", self.undo_frac)?;
        validate_weights(
            "growth",
            &[
                ("cross", self.w_cross),
                ("fresh", self.w_fresh),
                ("unsub", self.w_unsub),
                ("insert", self.w_insert),
                ("twist-neg", self.w_twist_neg),
                ("twist-swap", self.w_twist_swap),
                ("twist-cnot", self.w_twist_cnot),
                ("nonlinear-frame", self.w_nl_frame),
            ],
        )?;

        validate_frame(
            "growth",
            self.w_nl_frame,
            self.nl_frame_min_width,
            self.nl_frame_max_width,
            self.nl_frame_packet_gates,
        )?;

        let churn_min = self
            .churn_nl_frame_min_width
            .unwrap_or(self.nl_frame_min_width);
        let churn_max = self
            .churn_nl_frame_max_width
            .unwrap_or(self.nl_frame_max_width);
        let churn_packet = self
            .churn_nl_frame_packet_gates
            .unwrap_or(self.nl_frame_packet_gates);
        let churn_weight = self.churn_w_nl_frame.unwrap_or(self.w_nl_frame);
        validate_probability(
            "--churn-undo-frac",
            self.churn_undo_frac.unwrap_or(self.undo_frac),
        )?;
        validate_weights(
            "churn",
            &[
                ("cross", self.churn_w_cross.unwrap_or(self.w_cross)),
                ("fresh", self.churn_w_fresh.unwrap_or(self.w_fresh)),
                ("unsub", self.churn_w_unsub.unwrap_or(self.w_unsub)),
                ("insert", self.churn_w_insert.unwrap_or(self.w_insert)),
                (
                    "twist-neg",
                    self.churn_w_twist_neg.unwrap_or(self.w_twist_neg),
                ),
                (
                    "twist-swap",
                    self.churn_w_twist_swap.unwrap_or(self.w_twist_swap),
                ),
                (
                    "twist-cnot",
                    self.churn_w_twist_cnot.unwrap_or(self.w_twist_cnot),
                ),
                ("nonlinear-frame", churn_weight),
            ],
        )?;
        validate_frame("churn", churn_weight, churn_min, churn_max, churn_packet)?;
        if matches!(self.churn_target_size, Some(0)) {
            return Err("--churn-target-size must be positive".into());
        }
        if self.hard_size_cap > 0
            && self
                .churn_target_size
                .is_some_and(|target| target > self.hard_size_cap)
        {
            return Err(format!(
                "--churn-target-size ({}) exceeds --hard-size-cap ({})",
                self.churn_target_size.expect("checked Some"),
                self.hard_size_cap
            ));
        }
        if matches!(self.churn_temp, Some(t) if t <= 0.0 || !t.is_finite()) {
            return Err("--churn-temp must be finite and positive".into());
        }
        if matches!(self.g57_pairs_per_wire, Some(0)) {
            return Err("--g57-pairs-per-wire must be positive".into());
        }
        for (name, probability) in [
            ("--w-db", self.w_db),
            ("--p-db", self.p_db),
            ("--p-twist", self.p_twist),
            ("--db-convex-p", self.db_convex_p),
            ("--gen-bias", self.gen_bias),
            ("--p-db-ingest", self.p_db_ingest),
            ("--p-db-hard", self.p_db_hard),
        ] {
            validate_probability(name, probability)?;
        }
        if !self.p_db_final.is_finite() || self.p_db_final > 1.0 {
            return Err("--p-db-final must be finite and either negative or in [0, 1]".into());
        }
        if self.db_min_window < 2 {
            return Err("--db-min-window must be at least 2".into());
        }
        if self.db_max_window < self.db_min_window {
            return Err(format!(
                "--db-max-window ({}) must be at least --db-min-window ({})",
                self.db_max_window, self.db_min_window
            ));
        }
        if DbSample::parse(&self.db_sample).is_none() {
            return Err(format!(
                "unknown --db-sample {} (expected contiguous|convex|mixed)",
                self.db_sample
            ));
        }
        if self.db_max_degree > 11 {
            return Err(format!(
                "--db-max-degree {} exceeds the supported maximum 11",
                self.db_max_degree
            ));
        }
        if self.db_max_degree > 0 && self.db_degree_probes == 0 {
            return Err("--db-degree-probes must be positive when the degree guard is on".into());
        }
        if self.gen_target > 0 && self.gen_rescan == 0 {
            return Err("--gen-rescan must be positive when generation targeting is on".into());
        }
        if (self.p_db_ingest > 0.0 || self.p_db_hard > 0.0) && self.gen_target == 0 {
            return Err("--p-db-ingest/--p-db-hard require --gen-target > 0".into());
        }
        if !self.gen_stop_frac.is_finite() || self.gen_stop_frac > 1.0 {
            return Err("--gen-stop-frac must be finite and either negative or in [0, 1]".into());
        }
        if self.gen_stop_frac >= 0.0 && self.gen_target == 0 {
            return Err("--gen-stop-frac requires --gen-target > 0".into());
        }
        if self.gen_dose_exclude_g57_pair_frames {
            if self.gen_stop_frac < 0.0 {
                return Err(
                    "--gen-dose-exclude-g57-pair-frames requires an enabled --gen-stop-frac".into(),
                );
            }
            if self.g57_pair_stage != Some(G57PairStage::Pre) {
                return Err(
                    "--gen-dose-exclude-g57-pair-frames requires --g57-pair-stage pre".into(),
                );
            }
        }
        if !self.twist_cov_stop.is_finite() || self.twist_cov_stop < 0.0 {
            return Err("--twist-cov-stop must be finite and nonnegative".into());
        }
        if self.twist_cov_stop > 0.0 && self.gen_stop_frac < 0.0 {
            return Err("--twist-cov-stop requires an enabled --gen-stop-frac".into());
        }
        if self.dose_then_churn {
            if self.grow_moves.is_none() || self.churn_moves.is_none() {
                return Err("--dose-then-churn requires --grow-moves and --churn-moves".into());
            }
            if self.gen_target == 0 {
                return Err("--dose-then-churn requires --gen-target > 0".into());
            }
            if self.gen_stop_frac < 0.0 {
                return Err("--dose-then-churn requires an enabled --gen-stop-frac".into());
            }
            if self.g57_pair_stage == Some(G57PairStage::Post) {
                return Err(
                    "--dose-then-churn cannot use --g57-pair-stage post because the certified final generation census precedes that hook; use a separate terminal pair invocation"
                        .into(),
                );
            }
        }
        Ok(())
    }

    fn db_requested(&self) -> bool {
        self.w_db > 0.0
            || self.p_db > 0.0
            || self.p_db_final > 0.0
            || self.p_db_ingest > 0.0
            || self.p_db_hard > 0.0
    }

    fn validate_wire_capacity(&self, num_wires: usize) -> Result<(), String> {
        if self.w_nl_frame > 0.0 && self.nl_frame_max_width >= num_wires {
            return Err(format!(
                "growth nonlinear frame needs one target plus {} controls, but the circuit has only {num_wires} wires",
                self.nl_frame_max_width
            ));
        }
        let churn_weight = self.churn_w_nl_frame.unwrap_or(self.w_nl_frame);
        let churn_max = self
            .churn_nl_frame_max_width
            .unwrap_or(self.nl_frame_max_width);
        if churn_weight > 0.0 && churn_max >= num_wires {
            return Err(format!(
                "churn nonlinear frame needs one target plus {churn_max} controls, but the circuit has only {num_wires} wires"
            ));
        }
        Ok(())
    }

    fn g57_pair_config(&self, num_wires: usize) -> Option<G57PairSeedConfig> {
        self.g57_pair_stage.map(|_| G57PairSeedConfig {
            pairs_per_target_wire: self
                .g57_pairs_per_wire
                .expect("clap requires --g57-pairs-per-wire"),
            target_wires: self
                .g57_pair_target_wires
                .expect("clap requires --g57-pair-target-wires"),
            num_wires,
            control_wire_limit: self
                .g57_pair_control_wire_limit
                .expect("clap requires --g57-pair-control-wire-limit"),
            region: self
                .g57_pair_region
                .expect("clap requires --g57-pair-region"),
            seed: self.g57_pair_seed.unwrap_or(self.seed),
        })
    }
}

fn default_temp(target: usize) -> f64 {
    (target as f64 / 100.0).max(64.0)
}

/// Change only explicitly requested churn knobs. State belonging to the walk
/// (RNG, arena, origins, journal, and retained tabu entries) lives in `mixer`
/// and is intentionally not rebuilt at the phase boundary.
fn apply_churn_overrides(mixer: &mut Mixer, args: &Args) {
    if let Some(target) = args.churn_target_size {
        assert!(target > 0, "--churn-target-size must be positive");
        let changed = target != mixer.params.target_size;
        mixer.params.target_size = target;
        if changed && args.churn_temp.is_none() {
            mixer.params.temp = default_temp(target);
        }
    }
    if let Some(temp) = args.churn_temp {
        assert!(temp > 0.0, "--churn-temp must be positive");
        mixer.params.temp = temp;
    }
    if let Some(v) = args.churn_merge_reach {
        mixer.params.merge_reach = v;
    }
    if let Some(v) = args.churn_undo_frac {
        mixer.params.undo_frac = v;
    }
    if let Some(v) = args.churn_tabu_moves {
        mixer.params.tabu_moves = v;
    }
    if let Some(v) = args.churn_w_cross {
        mixer.params.w_cross = v;
    }
    if let Some(v) = args.churn_w_fresh {
        mixer.params.w_fresh = v;
    }
    if let Some(v) = args.churn_w_unsub {
        mixer.params.w_unsub = v;
    }
    if let Some(v) = args.churn_w_insert {
        mixer.params.w_insert = v;
    }
    if let Some(v) = args.churn_w_twist_neg {
        mixer.params.w_twist_neg = v;
    }
    if let Some(v) = args.churn_w_twist_swap {
        mixer.params.w_twist_swap = v;
    }
    if let Some(v) = args.churn_w_twist_cnot {
        mixer.params.w_twist_cnot = v;
    }
    if let Some(v) = args.churn_w_nl_frame {
        mixer.params.w_nl_frame = v;
    }
    if let Some(v) = args.churn_twist_min_len {
        mixer.params.twist_min_len = v;
    }
    if let Some(v) = args.churn_nl_frame_min_width {
        mixer.params.nl_frame_min_width = v;
    }
    if let Some(v) = args.churn_nl_frame_max_width {
        mixer.params.nl_frame_max_width = v;
    }
    if let Some(v) = args.churn_nl_frame_packet_gates {
        mixer.params.nl_frame_packet_gates = v;
    }
    if let Some(v) = args.churn_nl_frame_shots {
        mixer.params.nl_frame_shots = v;
    }
    if let Some(v) = args.churn_nl_frame_tenure {
        mixer.params.nl_frame_tenure = v;
    }
}

#[derive(Clone, Copy, Debug)]
struct DoseCensus {
    lag_fraction: f64,
    twist_coverage: f64,
    lag_met: bool,
    dose_met: bool,
}

fn census_fraction(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn targetable_lag_fraction(lag: u64, targetable: u64) -> Option<f64> {
    (targetable > 0).then(|| census_fraction(lag, targetable))
}

fn report_dose_census(label: &str, mixer: &Mixer, args: &Args) -> DoseCensus {
    let stats = mixer.generation_dose_stats();
    let describe = |scope: &str, census: local_mixing::postmix::mix::GenStats| {
        format!(
            "{scope} G={} Gall={} tgtbl={} alag={}/{} afrac={:.6} lag={}/{} frac={:.6} cheap={} hard={} unreach={} width_lag={} eligible={} fresh={}",
            census.g_circ,
            census.g_all,
            census.targetable,
            census.all_lag,
            census.total,
            census_fraction(census.all_lag, census.total),
            census.lag,
            census.targetable,
            census_fraction(census.lag, census.targetable),
            census.cheap,
            census.hard,
            census.unreach,
            census.wlag,
            census.elig,
            census.fresh,
        )
    };
    let all_summary = describe("all", stats.all);
    let non_pair_summary = describe("non_pair", stats.non_pair);
    let (dose_scope, selected) = if args.gen_dose_exclude_g57_pair_frames {
        ("non-pair", stats.non_pair)
    } else {
        ("all", stats.all)
    };
    let legacy_selected_lag_fraction = census_fraction(selected.all_lag, selected.total);
    let selected_targetable_fraction = targetable_lag_fraction(selected.lag, selected.targetable);
    let lag_fraction = selected_targetable_fraction.unwrap_or(0.0);
    let twist_coverage = mixer.twist_coverage();
    let denominator_nonempty = selected_targetable_fraction.is_some();
    let lag_met =
        args.gen_stop_frac >= 0.0 && denominator_nonempty && lag_fraction <= args.gen_stop_frac;
    let dose_met = lag_met && (args.twist_cov_stop <= 0.0 || twist_coverage >= args.twist_cov_stop);
    println!(
        "[fmix] {label} census: mv={} size={} target={} | {} | {} | dose_scope={} selected_lag_fraction={:.6} dose_lag_fraction={:.6} denominator_nonempty={} required_lag<={:.6} twist_coverage={:.6} required_coverage>={:.6} dose_met={}",
        mixer.counters.moves,
        mixer.arena.len(),
        args.gen_target,
        all_summary,
        non_pair_summary,
        dose_scope,
        legacy_selected_lag_fraction,
        lag_fraction,
        denominator_nonempty,
        args.gen_stop_frac,
        twist_coverage,
        args.twist_cov_stop,
        dose_met,
    );
    DoseCensus {
        lag_fraction,
        twist_coverage,
        lag_met,
        dose_met,
    }
}

fn run_churn_phase(mixer: &mut Mixer, args: &Args, start_move: u64, churn_moves: u64) -> MixStop {
    apply_churn_overrides(mixer, args);
    let stop_move = start_move
        .checked_add(churn_moves)
        .expect("churn move budget overflows the u64 move clock");
    mixer.params.moves = stop_move;
    println!(
        "[fmix] phase churn START: mv={}..{} ({} additional) target={} temp={} merge_reach={} undo_frac={} tabu={} weights cross/fresh/unsub/insert/twn/tws/twc/nlf={}/{}/{}/{}/{}/{}/{}/{} nl_width={}..{} nl_packet={} nl_shots={} nl_tenure={}",
        start_move,
        stop_move,
        churn_moves,
        mixer.params.target_size,
        mixer.params.temp,
        mixer.params.merge_reach,
        mixer.params.undo_frac,
        mixer.params.tabu_moves,
        mixer.params.w_cross,
        mixer.params.w_fresh,
        mixer.params.w_unsub,
        mixer.params.w_insert,
        mixer.params.w_twist_neg,
        mixer.params.w_twist_swap,
        mixer.params.w_twist_cnot,
        mixer.params.w_nl_frame,
        mixer.params.nl_frame_min_width,
        mixer.params.nl_frame_max_width,
        mixer.params.nl_frame_packet_gates,
        mixer.params.nl_frame_shots,
        mixer.params.nl_frame_tenure,
    );
    let reason = mixer.run();
    println!(
        "[fmix] phase churn DONE: mv={} size={}",
        mixer.counters.moves,
        mixer.arena.len()
    );
    reason
}

fn transition_to_dose_churn(
    mixer: &mut Mixer,
    args: &Args,
    churn_moves: u64,
    boundary: &str,
) -> MixStop {
    mixer.report();
    let census = report_dose_census("new-SSS phase-A transition", mixer, args);
    if !census.dose_met {
        panic!(
            "[fmix] {boundary} before the requested new-SSS dose: lag_fraction={:.6} (required <= {:.6}) twist_coverage={:.6} (required >= {:.6}); refusing churn/output",
            census.lag_fraction, args.gen_stop_frac, census.twist_coverage, args.twist_cov_stop,
        );
    }

    let start_move = mixer.counters.moves;
    mixer.params.gen_stop_frac = -1.0;
    mixer.params.twist_cov_stop = 0.0;
    mixer.params.p_db_ingest = 0.0;
    mixer.params.p_db_hard = 0.0;
    // Phase A uses p_twist as a first-class long-range dose. Phase B follows
    // the established grow/churn schedule, where the much smaller
    // churn-w-twist-* values are ordinary expansion weights.
    mixer.params.p_twist = 0.0;
    println!(
        "[fmix] new-SSS phase transition: boundary={boundary} mv={start_move}; dose stop, DB ingest/hard, and first-class p_twist disabled; generation metadata retained; scheduling exactly {churn_moves} churn moves"
    );
    run_churn_phase(mixer, args, start_move, churn_moves)
}

fn require_complete_dose_churn(stop_reason: &MixStop, args: &Args) {
    if args.dose_then_churn && !matches!(stop_reason, MixStop::MovesBudget) {
        let terminal = match stop_reason {
            MixStop::MovesBudget => "moves-budget",
            MixStop::StopFlag => "stop-flag",
            MixStop::DoseReached => "unconsumed-dose-reached",
        };
        panic!(
            "[fmix] new-SSS composite did not complete its certified dose transition and exact post-dose churn budget ({terminal}); refusing final float/output"
        );
    }
}

fn pair_stage_name(stage: G57PairStage) -> &'static str {
    match stage {
        G57PairStage::Pre => "pre",
        G57PairStage::Post => "post",
    }
}

fn run_g57_pair_hook(
    mixer: &mut Mixer,
    config: G57PairSeedConfig,
    stage: G57PairStage,
    args: &Args,
) -> FmixG57PairSeedReport {
    let stage_name = pair_stage_name(stage);
    let report = mixer
        .seed_g57_pairs(config)
        .unwrap_or_else(|error| panic!("invalid {stage_name}-fmix G57 pair hook: {error}"));
    // Every local step has an exact truth-table proof; this is an additional
    // whole-circuit sampled guard against plumbing/index errors.
    mixer.global_check();
    let manifest = args.g57_pair_manifest.clone().unwrap_or_else(|| {
        args.output
            .as_deref()
            .map(|output| format!("{output}.{stage_name}.g57-pairs.tsv"))
            .unwrap_or_else(|| format!("fmix.{stage_name}.g57-pairs.tsv"))
    });
    std::fs::write(&manifest, report.manifest_tsv()).expect("write fmix G57 pair manifest");
    println!(
        "[fmix] g57-pair-{stage_name}: baseline={} final={} pairs={} native_g57={} fragments={} per_target={} targets={} controls=0..{} region={} frozen_gaps={}..{} seed={} intact_shot_steps_left/right={}/{} intact_stops_collision/boundary={}/{} fragment_transport_steps={} cross_attempts={} halves_with_r_crossing={}/{} verified manifest={}",
        report.baseline_gates,
        report.final_gates,
        report.pairs,
        report.inserted_native_g57,
        report.emitted_fragments,
        report.pairs_per_target_wire,
        report.target_wires,
        report.control_wire_limit,
        report.region,
        report.first_gap,
        report.gap_end_exclusive,
        report.seed,
        report.total_intact_left_shot_steps,
        report.total_intact_right_shot_steps,
        report.intact_collision_stops,
        report.intact_boundary_stops,
        report.total_transport_steps,
        report.total_cross_attempts,
        report.halves_with_r_crossing,
        report.pairs * 2,
        manifest,
    );
    report
}

fn emit_final_g57_pair_census(mixer: &Mixer, args: &Args, num_wires: usize) {
    let Some(stage) = args.g57_pair_stage else {
        return;
    };
    let config = args
        .g57_pair_config(num_wires)
        .expect("pair stage has a validated config");
    let pairs_requested = config.pair_count().expect("validated pair count");
    let copy_frames_requested = pairs_requested
        .checked_mul(2)
        .expect("validated pair copy-frame count");
    let pair = mixer.g57_pair_frame_stats();
    assert!(
        pair.distinct_pair_ids <= pairs_requested
            && pair.distinct_copy_frames <= copy_frames_requested
            && pair.complete_pairs <= pair.distinct_pair_ids
            && pair.tagged_gates <= mixer.arena.len(),
        "invalid final G57-pair frame census"
    );

    let generation = mixer.generation_dose_stats();
    let all_lag_fraction = census_fraction(generation.all.all_lag, generation.all.total);
    let non_pair_lag_fraction =
        census_fraction(generation.non_pair.all_lag, generation.non_pair.total);
    let all_dose_lag_fraction = census_fraction(generation.all.lag, generation.all.targetable);
    let non_pair_dose_lag_fraction =
        census_fraction(generation.non_pair.lag, generation.non_pair.targetable);
    let all_fresh_fraction = census_fraction(generation.all.fresh, generation.all.total);
    let non_pair_fresh_fraction =
        census_fraction(generation.non_pair.fresh, generation.non_pair.total);
    let (dose_scope, selected) = if args.gen_dose_exclude_g57_pair_frames {
        ("non-pair", generation.non_pair)
    } else {
        ("all", generation.all)
    };
    // Preserve selected_* as the historical scoped all-gate census for the
    // v1 campaign sidecar. The additive dose_targetable_* fields below record
    // the population now used by generation stopping.
    let selected_lag_fraction = census_fraction(selected.all_lag, selected.total);
    let dose_lag_fraction = census_fraction(selected.lag, selected.targetable);

    println!(
        "[fmix] g57-pair final census: stage={} requested_pairs={} live_pair_ids={} complete_pairs={} requested_copy_frames={} live_copy_frames={} tagged_gates={} union_span_coverage={:.9} | generation_all G={} Gall={} targetable_lag={}/{} frac={:.9} all_lag={}/{} frac={:.9} fresh={} fresh_frac={:.9} | generation_non_pair G={} Gall={} targetable_lag={}/{} frac={:.9} all_lag={}/{} frac={:.9} fresh={} fresh_frac={:.9} | dose_scope={} selected_lag_fraction={:.9} dose_lag_fraction={:.9}",
        pair_stage_name(stage),
        pairs_requested,
        pair.distinct_pair_ids,
        pair.complete_pairs,
        copy_frames_requested,
        pair.distinct_copy_frames,
        pair.tagged_gates,
        pair.union_span_coverage,
        generation.all.g_circ,
        generation.all.g_all,
        generation.all.lag,
        generation.all.targetable,
        all_dose_lag_fraction,
        generation.all.all_lag,
        generation.all.total,
        all_lag_fraction,
        generation.all.fresh,
        all_fresh_fraction,
        generation.non_pair.g_circ,
        generation.non_pair.g_all,
        generation.non_pair.lag,
        generation.non_pair.targetable,
        non_pair_dose_lag_fraction,
        generation.non_pair.all_lag,
        generation.non_pair.total,
        non_pair_lag_fraction,
        generation.non_pair.fresh,
        non_pair_fresh_fraction,
        dose_scope,
        selected_lag_fraction,
        dose_lag_fraction,
    );

    let Some(path) = &args.g57_pair_census_out else {
        return;
    };
    let sidecar = format!(
        concat!(
            "schema=g57_pair_frame_census_v1\n",
            "pair_stage={}\n",
            "pairs_requested={}\n",
            "pair_ids_live={}\n",
            "complete_pairs_live={}\n",
            "copy_frames_requested={}\n",
            "copy_frames_live={}\n",
            "tagged_gates_live={}\n",
            "union_span_coverage={:.12}\n",
            "generation_target={}\n",
            "generation_all_total={}\n",
            "generation_all_lag={}\n",
            "generation_all_lag_fraction={:.12}\n",
            "generation_all_G={}\n",
            "generation_all_Gall={}\n",
            "generation_all_targetable={}\n",
            "generation_all_targetable_lag={}\n",
            "generation_all_targetable_lag_fraction={:.12}\n",
            "generation_all_fresh={}\n",
            "generation_all_fresh_fraction={:.12}\n",
            "generation_non_pair_total={}\n",
            "generation_non_pair_lag={}\n",
            "generation_non_pair_lag_fraction={:.12}\n",
            "generation_non_pair_G={}\n",
            "generation_non_pair_Gall={}\n",
            "generation_non_pair_targetable={}\n",
            "generation_non_pair_targetable_lag={}\n",
            "generation_non_pair_targetable_lag_fraction={:.12}\n",
            "generation_non_pair_fresh={}\n",
            "generation_non_pair_fresh_fraction={:.12}\n",
            "generation_dose_scope={}\n",
            "generation_selected_total={}\n",
            "generation_selected_lag={}\n",
            "generation_selected_lag_fraction={:.12}\n",
            "generation_dose_targetable_total={}\n",
            "generation_dose_targetable_lag={}\n",
            "generation_dose_targetable_lag_fraction={:.12}\n",
            "generation_required_max_lag_fraction={:.12}\n",
            "twist_coverage={:.12}\n"
        ),
        pair_stage_name(stage),
        pairs_requested,
        pair.distinct_pair_ids,
        pair.complete_pairs,
        copy_frames_requested,
        pair.distinct_copy_frames,
        pair.tagged_gates,
        pair.union_span_coverage,
        args.gen_target,
        generation.all.total,
        generation.all.all_lag,
        all_lag_fraction,
        generation.all.g_circ,
        generation.all.g_all,
        generation.all.targetable,
        generation.all.lag,
        all_dose_lag_fraction,
        generation.all.fresh,
        all_fresh_fraction,
        generation.non_pair.total,
        generation.non_pair.all_lag,
        non_pair_lag_fraction,
        generation.non_pair.g_circ,
        generation.non_pair.g_all,
        generation.non_pair.targetable,
        generation.non_pair.lag,
        non_pair_dose_lag_fraction,
        generation.non_pair.fresh,
        non_pair_fresh_fraction,
        dose_scope,
        selected.total,
        selected.all_lag,
        selected_lag_fraction,
        selected.targetable,
        selected.lag,
        dose_lag_fraction,
        args.gen_stop_frac,
        mixer.twist_coverage(),
    );
    std::fs::write(path, sidecar).expect("write final G57-pair census");
    println!("[fmix] wrote final G57-pair census to {path}");
}

fn validate_hard_cap_admission(
    input_len: usize,
    target: usize,
    hard_size_cap: usize,
) -> Result<(), String> {
    if hard_size_cap == 0 {
        return Ok(());
    }
    if input_len > hard_size_cap {
        return Err(format!(
            "input gate count ({input_len}) exceeds --hard-size-cap ({hard_size_cap})"
        ));
    }
    if target > hard_size_cap {
        return Err(format!(
            "effective target size ({target}) exceeds --hard-size-cap ({hard_size_cap})"
        ));
    }
    Ok(())
}

fn main() {
    let args = Args::parse();
    let schedule = RunSchedule::from_args(&args)
        .unwrap_or_else(|e| Args::command().error(ErrorKind::ValueValidation, e).exit());
    if let Err(e) = args.validate() {
        Args::command().error(ErrorKind::ValueValidation, e).exit();
    }

    let (gates, file_wires): (Vec<XGate>, usize) = match args.input_format.as_str() {
        "mpmct1" => format::read_mpmct(&args.input).expect("read mpmct1 circuit"),
        "g57" => {
            let g = format::read_g57_file(&args.input).expect("read g57 circuit");
            let w = max_wire(&g) as usize + 1;
            (g, w)
        }
        other => panic!("unknown --input-format {other}"),
    };
    let identity_noops = gates.iter().filter(|gate| gate.is_noop()).count();
    if identity_noops > 0 {
        println!("[fmix] canonicalized away {identity_noops} complemented empty identity gates");
    }
    let canonical_input_len = gates.len() - identity_noops;
    if canonical_input_len == 0 {
        println!("[fmix] canonical input is the empty identity circuit");
    }
    let num_wires = file_wires.max(max_wire(&gates) as usize + 1);
    if let Err(e) = args.validate_wire_capacity(num_wires) {
        Args::command().error(ErrorKind::ValueValidation, e).exit();
    }
    if let Some(config) = args.g57_pair_config(num_wires)
        && let Err(e) = config.validate()
    {
        Args::command().error(ErrorKind::ValueValidation, e).exit();
    }
    let input_len = canonical_input_len;
    let comp0 = gates
        .iter()
        .filter(|gate| gate.comp && !gate.is_noop())
        .count();
    let target = args.target_size.unwrap_or(input_len);
    if let Err(error) = validate_hard_cap_admission(input_len, target, args.hard_size_cap) {
        Args::command()
            .error(ErrorKind::ValueValidation, error)
            .exit();
    }
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
        args.temp.unwrap_or(default_temp(target)),
        schedule.total_budget(),
        args.seed
    );
    match schedule {
        RunSchedule::Legacy { .. } => println!("[fmix] schedule: legacy one-phase"),
        RunSchedule::GrowThenChurn {
            grow_moves,
            churn_moves,
            total_moves,
        } => {
            if args.dose_then_churn {
                println!(
                    "[fmix] schedule: new-SSS dose for at most {} moves, then exactly {} additional churn moves (maximum total={})",
                    grow_moves, churn_moves, total_moves
                );
            } else {
                println!(
                    "[fmix] schedule: grow={} moves, then churn={} additional moves (cumulative stop={})",
                    grow_moves, churn_moves, total_moves
                );
            }
        }
    }
    if args.hard_size_cap > 0 {
        println!(
            "[fmix] hard size cap ON: {} gates (transactional growth rejection)",
            args.hard_size_cap
        );
    }
    if args.p_twist > 0.0 {
        println!(
            "[fmix] first-class twist rounds ON: p_twist={} (w-twist-* select kind)",
            args.p_twist
        );
    }
    if args.w_twist_neg > 0.0
        || args.w_twist_swap > 0.0
        || args.w_twist_cnot > 0.0
        || args.w_nl_frame > 0.0
    {
        println!(
            "[fmix] frames ON: w_twist_neg={} w_twist_swap={} w_twist_cnot={} w_nl_frame={} twist_min_len={} nl_width={}..{} nl_packet={} nl_shots={} nl_tenure={}",
            args.w_twist_neg,
            args.w_twist_swap,
            args.w_twist_cnot,
            args.w_nl_frame,
            args.twist_min_len,
            args.nl_frame_min_width,
            args.nl_frame_max_width,
            args.nl_frame_packet_gates,
            args.nl_frame_shots,
            args.nl_frame_tenure,
        );
    }
    if args.db_requested() {
        println!(
            "[fmix] DB replacement ON: w_db={} p_db={} p_db_final={} p_db_ingest={} p_db_hard={} steer={} window=[{},{}] sample={} verify={} (FROZEN_DB_DIR required)",
            args.w_db,
            args.p_db,
            args.p_db_final,
            args.p_db_ingest,
            args.p_db_hard,
            args.p_db_steer,
            args.db_min_window,
            args.db_max_window,
            args.db_sample,
            !args.no_db_verify,
        );
    }
    if args.gen_target > 0 {
        println!(
            "[fmix] generation targeting ON: target={} bias={} rescan={} stop_frac={} dose_scope={} twist_cov_stop={} miss_budget={} giveup={} split={} median={}",
            args.gen_target,
            args.gen_bias,
            args.gen_rescan,
            args.gen_stop_frac,
            if args.gen_dose_exclude_g57_pair_frames {
                "non-pair"
            } else {
                "all"
            },
            args.twist_cov_stop,
            args.gen_miss_budget,
            args.gen_giveup,
            if args.gen_split_inherit {
                "inherit"
            } else {
                "ratchet"
            },
            if args.gen_median_low {
                "lower"
            } else {
                "upper"
            },
        );
    }

    let params = MixParams {
        k_max: args.k_max,
        split_damp: args.split_damp,
        split_base: args.split_base,
        dir_p: args.dir_p,
        dir_q: args.dir_q,
        target_size: target,
        hard_size_cap: args.hard_size_cap,
        temp: args.temp.unwrap_or(0.0),
        moves: schedule.first_budget(),
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
        w_nl_frame: args.w_nl_frame,
        nl_frame_min_width: args.nl_frame_min_width,
        nl_frame_max_width: args.nl_frame_max_width,
        nl_frame_packet_gates: args.nl_frame_packet_gates,
        nl_frame_shots: args.nl_frame_shots,
        nl_frame_tenure: args.nl_frame_tenure,
        twist_min_len: args.twist_min_len,
        p_twist: args.p_twist,
        w_db: args.w_db,
        p_db: args.p_db,
        p_db_final: args.p_db_final,
        p_db_steer: args.p_db_steer,
        db_min_window: args.db_min_window,
        db_max_window: args.db_max_window,
        db_sample: DbSample::parse(&args.db_sample).expect("validated --db-sample"),
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
        gen_dose_exclude_g57_pair_frames: args.gen_dose_exclude_g57_pair_frames,
        twist_cov_stop: args.twist_cov_stop,
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
    if args.g57_pair_stage == Some(G57PairStage::Pre) {
        run_g57_pair_hook(
            &mut mixer,
            args.g57_pair_config(num_wires)
                .expect("pre pair stage has a validated config"),
            G57PairStage::Pre,
            &args,
        );
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
    let stop_reason = match schedule {
        RunSchedule::Legacy { .. } => mixer.run(),
        RunSchedule::GrowThenChurn {
            grow_moves,
            churn_moves,
            total_moves: _,
        } => {
            println!(
                "[fmix] phase grow START: mv=0..{} target={} temp={}",
                grow_moves, mixer.params.target_size, mixer.params.temp
            );
            match mixer.run() {
                MixStop::StopFlag => MixStop::StopFlag,
                MixStop::DoseReached => {
                    if args.dose_then_churn {
                        transition_to_dose_churn(
                            &mut mixer,
                            &args,
                            churn_moves,
                            "dose-reached report point",
                        )
                    } else {
                        println!(
                            "[fmix] phase grow DONE early: mv={} size={} (generation/twist dose reached; churn skipped)",
                            mixer.counters.moves,
                            mixer.arena.len()
                        );
                        MixStop::DoseReached
                    }
                }
                MixStop::MovesBudget => {
                    if args.dose_then_churn {
                        transition_to_dose_churn(
                            &mut mixer,
                            &args,
                            churn_moves,
                            "phase-A move-budget boundary",
                        )
                    } else {
                        println!(
                            "[fmix] phase grow DONE: mv={} size={}; applying churn overrides",
                            grow_moves,
                            mixer.arena.len()
                        );
                        mixer.report();
                        run_churn_phase(&mut mixer, &args, grow_moves, churn_moves)
                    }
                }
            }
        }
    };
    let secs = t0.elapsed().as_secs_f64();
    mixer.report();
    // `DoseReached` is consumed by `transition_to_dose_churn`, and a complete
    // post-dose phase terminates only at its exact move boundary. A StopFlag
    // from either Phase A or churn must therefore remain diagnostic evidence,
    // never an accepted partial composite.
    require_complete_dose_churn(&stop_reason, &args);
    if args.dose_then_churn {
        let final_census = report_dose_census("new-SSS final", &mixer, &args);
        if !final_census.lag_met {
            panic!(
                "[fmix] new-SSS churn regressed the final generation dose: target={} lag_fraction={:.6} (required <= {:.6}); refusing final float/output",
                args.gen_target, final_census.lag_fraction, args.gen_stop_frac,
            );
        }
        println!(
            "[fmix] new-SSS final laggard revalidation PASSED: {:.6} <= {:.6} (phase-A twist dose was already certified at transition)",
            final_census.lag_fraction, args.gen_stop_frac
        );
    } else if matches!(&stop_reason, MixStop::MovesBudget) && args.gen_stop_frac >= 0.0 {
        let final_census = report_dose_census("generation budget boundary", &mixer, &args);
        if !final_census.dose_met {
            panic!(
                "[fmix] generation dose shortfall at move budget: target={} lag_fraction={:.6} (required <= {:.6}) twist_coverage={:.6} (required >= {:.6}); refusing final float/output",
                args.gen_target,
                final_census.lag_fraction,
                args.gen_stop_frac,
                final_census.twist_coverage,
                args.twist_cov_stop,
            );
        }
        println!("[fmix] generation dose met at the exact move boundary");
    }
    println!(
        "[fmix] chain done in {:.1}s: {} ({} -> {} gates, {} -> {} g57 fossils)",
        secs,
        match stop_reason {
            MixStop::MovesBudget => "moves budget spent",
            MixStop::StopFlag => "stop flag",
            MixStop::DoseReached => "dose reached (generation + twist coverage targets met)",
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

    if args.g57_pair_stage == Some(G57PairStage::Post) {
        run_g57_pair_hook(
            &mut mixer,
            args.g57_pair_config(num_wires)
                .expect("post pair stage has a validated config"),
            G57PairStage::Post,
            &args,
        );
    }

    emit_final_g57_pair_census(&mixer, &args, num_wires);

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
        let generations = mixer.gens_in_order();
        let mut s = String::with_capacity(generations.len() * 4);
        for generation in generations {
            s.push_str(&format!("{generation}\n"));
        }
        std::fs::write(path, s).expect("write generations");
        println!("[fmix] wrote gens sidecar to {path} (born-random = {GEN_FRESH})");
    }
}

#[cfg(test)]
mod cli_tests {
    use super::*;

    fn parse(extra: &[&str]) -> Result<Args, clap::Error> {
        let mut argv = vec!["fmix", "--input", "unused.mpmct1"];
        argv.extend_from_slice(extra);
        Args::try_parse_from(argv)
    }

    #[test]
    fn targetable_lag_fraction_fails_closed_on_an_empty_population() {
        assert_eq!(targetable_lag_fraction(1, 4), Some(0.25));
        assert_eq!(targetable_lag_fraction(0, 0), None);
    }

    #[test]
    fn legacy_schedule_keeps_historical_default() {
        let args = parse(&[]).unwrap();
        assert_eq!(
            RunSchedule::from_args(&args).unwrap(),
            RunSchedule::Legacy { moves: 1_000_000 }
        );

        let args = parse(&["--moves", "123"]).unwrap();
        assert_eq!(
            RunSchedule::from_args(&args).unwrap(),
            RunSchedule::Legacy { moves: 123 }
        );
    }

    #[test]
    fn schedule_pair_is_required_and_moves_conflicts() {
        assert!(parse(&["--grow-moves", "10"]).is_err());
        assert!(parse(&["--churn-moves", "20"]).is_err());
        assert!(parse(&["--moves", "30", "--grow-moves", "10", "--churn-moves", "20"]).is_err());
        assert!(parse(&["--churn-w-cross", "0.5"]).is_err());
    }

    #[test]
    fn g57_pair_hook_requires_complete_explicit_geometry() {
        assert!(parse(&["--g57-pair-stage", "pre"]).is_err());
        let args = parse(&[
            "--g57-pair-stage",
            "post",
            "--g57-pairs-per-wire",
            "100",
            "--g57-pair-target-wires",
            "384",
            "--g57-pair-control-wire-limit",
            "128",
            "--g57-pair-region",
            "middle-quarter",
            "--g57-pair-seed",
            "57",
        ])
        .unwrap();
        assert_eq!(args.g57_pair_stage, Some(G57PairStage::Post));
        let config = args.g57_pair_config(384).unwrap();
        assert_eq!(config.pairs_per_target_wire, 100);
        assert_eq!(config.target_wires, 384);
        assert_eq!(config.control_wire_limit, 128);
        assert_eq!(config.seed, 57);
        config.validate().unwrap();
    }

    #[test]
    fn pair_census_and_non_pair_dose_flags_are_fail_closed() {
        assert!(parse(&["--g57-pair-census-out", "pair.env"]).is_err());

        let missing_stop = parse(&[
            "--gen-dose-exclude-g57-pair-frames",
            "--gen-target",
            "100",
            "--g57-pair-stage",
            "pre",
            "--g57-pairs-per-wire",
            "50",
            "--g57-pair-target-wires",
            "512",
            "--g57-pair-control-wire-limit",
            "512",
            "--g57-pair-region",
            "uniform",
        ])
        .unwrap();
        assert!(missing_stop.validate().is_err());

        let missing_pair_stage = parse(&[
            "--gen-dose-exclude-g57-pair-frames",
            "--gen-target",
            "100",
            "--gen-stop-frac",
            "0.02",
        ])
        .unwrap();
        assert!(missing_pair_stage.validate().is_err());

        let args = parse(&[
            "--gen-dose-exclude-g57-pair-frames",
            "--gen-target",
            "100",
            "--gen-stop-frac",
            "0.02",
            "--g57-pair-stage",
            "pre",
            "--g57-pairs-per-wire",
            "50",
            "--g57-pair-target-wires",
            "512",
            "--g57-pair-control-wire-limit",
            "512",
            "--g57-pair-region",
            "uniform",
            "--g57-pair-census-out",
            "pair.env",
        ])
        .unwrap();
        args.validate().unwrap();
        assert!(args.gen_dose_exclude_g57_pair_frames);
        assert_eq!(args.g57_pair_census_out.as_deref(), Some("pair.env"));
    }

    #[test]
    fn schedule_uses_cumulative_move_clock() {
        let args = parse(&["--grow-moves", "10", "--churn-moves", "20"]).unwrap();
        assert_eq!(
            RunSchedule::from_args(&args).unwrap(),
            RunSchedule::GrowThenChurn {
                grow_moves: 10,
                churn_moves: 20,
                total_moves: 30,
            }
        );
    }

    #[test]
    fn nonlinear_width_ranges_are_validated_per_phase() {
        let args = parse(&[
            "--w-nl-frame",
            "0.1",
            "--nl-frame-min-width",
            "4",
            "--nl-frame-max-width",
            "3",
        ])
        .unwrap();
        assert!(args.validate().is_err());

        let args = parse(&[
            "--grow-moves",
            "10",
            "--churn-moves",
            "20",
            "--churn-w-nl-frame",
            "0.1",
            "--nl-frame-min-width",
            "2",
            "--nl-frame-max-width",
            "3",
            "--churn-nl-frame-min-width",
            "4",
        ])
        .unwrap();
        assert!(args.validate().is_err());
    }

    #[test]
    fn growth_and_churn_probabilities_and_weights_are_validated_up_front() {
        let assert_rejected = |extra: &[&str]| match parse(extra) {
            Ok(args) => assert!(args.validate().is_err(), "accepted invalid args: {extra:?}"),
            Err(_) => {
                // clap rejecting a syntactically negative option value is also
                // a valid up-front failure, before any growth work begins.
            }
        };
        for extra in [
            vec!["--undo-frac", "1.01"],
            vec!["--w-cross", "-0.01"],
            vec!["--w-insert", "NaN"],
            vec!["--temp", "inf"],
        ] {
            assert_rejected(&extra);
        }

        for extra in [
            vec![
                "--grow-moves",
                "10",
                "--churn-moves",
                "20",
                "--churn-undo-frac",
                "-0.01",
            ],
            vec![
                "--grow-moves",
                "10",
                "--churn-moves",
                "20",
                "--churn-undo-frac",
                "1.01",
            ],
            vec![
                "--grow-moves",
                "10",
                "--churn-moves",
                "20",
                "--churn-w-cross",
                "-0.01",
            ],
            vec![
                "--grow-moves",
                "10",
                "--churn-moves",
                "20",
                "--churn-w-insert",
                "NaN",
            ],
        ] {
            assert_rejected(&extra);
        }
    }

    #[test]
    fn dormant_frame_geometry_does_not_break_legacy_low_k() {
        let args = parse(&["--k-max", "1"]).unwrap();
        assert_eq!(args.w_nl_frame, 0.0);
        assert!(args.validate().is_ok());
        assert!(args.validate_wire_capacity(1).is_ok());
    }

    #[test]
    fn active_frame_requires_target_plus_controls() {
        let args = parse(&[
            "--w-nl-frame",
            "0.1",
            "--nl-frame-min-width",
            "2",
            "--nl-frame-max-width",
            "3",
        ])
        .unwrap();
        args.validate().unwrap();
        assert!(args.validate_wire_capacity(3).is_err());
        assert!(args.validate_wire_capacity(4).is_ok());
    }

    #[test]
    fn churn_overrides_inherit_unspecified_values() {
        let args = parse(&[
            "--grow-moves",
            "10",
            "--churn-moves",
            "20",
            "--churn-target-size",
            "20000",
            "--churn-w-cross",
            "0.45",
            "--churn-w-nl-frame",
            "0.2",
            "--churn-nl-frame-packet-gates",
            "8",
        ])
        .unwrap();
        args.validate().unwrap();

        let params = MixParams {
            target_size: 10,
            temp: 77.0,
            moves: 10,
            w_cross: 0.70,
            w_fresh: 0.123,
            ..MixParams::default()
        };
        let mut mixer = Mixer::new(vec![XGate::x_gate(0)], 4, params);
        apply_churn_overrides(&mut mixer, &args);

        assert_eq!(mixer.params.target_size, 20_000);
        assert_eq!(mixer.params.temp, 200.0);
        assert_eq!(mixer.params.w_cross, 0.45);
        assert_eq!(mixer.params.w_fresh, 0.123);
        assert_eq!(mixer.params.w_nl_frame, 0.2);
        assert_eq!(mixer.params.nl_frame_packet_gates, 8);
    }

    #[test]
    fn db_and_generation_cli_surface_parses_as_one_configuration() {
        let args = parse(&[
            "--target-size",
            "1000",
            "--hard-size-cap",
            "2000",
            "--w-db",
            "0.2",
            "--p-db",
            "0.3",
            "--p-twist",
            "0.01",
            "--p-db-final",
            "0.1",
            "--p-db-steer",
            "--db-min-window",
            "2",
            "--db-max-window",
            "5",
            "--db-sample",
            "mixed",
            "--db-ctrl-cap",
            "2",
            "--db-convex-p",
            "0.75",
            "--no-db-verify",
            "--db-record",
            "attempts.tsv",
            "--db-dry-run",
            "--db-max-degree",
            "9",
            "--db-degree-probes",
            "8",
            "--db-max-span",
            "30",
            "--db-wire-terms",
            "1024",
            "--db-total-terms",
            "2048",
            "--db-prefixes",
            "--gen-target",
            "100",
            "--gen-bias",
            "0.98",
            "--gen-rescan",
            "50",
            "--p-db-ingest",
            "0.5",
            "--p-db-hard",
            "0.05",
            "--gen-miss-budget",
            "6",
            "--gen-giveup",
            "128",
            "--gen-split-inherit",
            "--gen-median-low",
            "--gen-stop-frac",
            "0.02",
            "--twist-cov-stop",
            "600",
            "--gens-out",
            "final.gens",
        ])
        .unwrap();
        args.validate().unwrap();

        assert_eq!(args.hard_size_cap, 2000);
        assert_eq!(args.w_db, 0.2);
        assert_eq!(args.p_db, 0.3);
        assert_eq!(args.p_twist, 0.01);
        assert_eq!(args.p_db_final, 0.1);
        assert!(args.p_db_steer);
        assert_eq!(DbSample::parse(&args.db_sample), Some(DbSample::Mixed));
        assert_eq!(args.db_ctrl_cap, 2);
        assert!(args.no_db_verify);
        assert_eq!(args.gen_target, 100);
        assert_eq!(args.p_db_ingest, 0.5);
        assert_eq!(args.p_db_hard, 0.05);
        assert_eq!(args.gen_stop_frac, 0.02);
        assert_eq!(args.twist_cov_stop, 600.0);
        assert_eq!(args.gens_out.as_deref(), Some("final.gens"));
        assert!(args.db_requested());
    }

    #[test]
    fn no_db_defaults_match_the_inert_core_configuration() {
        let args = parse(&[]).unwrap();
        let defaults = MixParams::default();
        args.validate().unwrap();

        assert_eq!(args.hard_size_cap, defaults.hard_size_cap);
        assert_eq!(args.w_db, defaults.w_db);
        assert_eq!(args.p_db, defaults.p_db);
        assert_eq!(args.p_twist, defaults.p_twist);
        assert_eq!(args.p_db_final, defaults.p_db_final);
        assert_eq!(args.p_db_ingest, defaults.p_db_ingest);
        assert_eq!(args.p_db_hard, defaults.p_db_hard);
        assert_eq!(args.gen_target, defaults.gen_target);
        assert_eq!(DbSample::parse(&args.db_sample), Some(defaults.db_sample));
        assert!(!args.db_requested());
    }

    #[test]
    fn ingest_and_paid_channels_count_as_db_activation() {
        let ingest = parse(&["--gen-target", "1", "--p-db-ingest", "0.5"]).unwrap();
        ingest.validate().unwrap();
        assert!(ingest.db_requested());

        let hard = parse(&["--gen-target", "1", "--p-db-hard", "0.05"]).unwrap();
        hard.validate().unwrap();
        assert!(hard.db_requested());
    }

    #[test]
    fn db_and_generation_ranges_are_rejected_up_front() {
        let assert_invalid = |extra: &[&str]| {
            let args = parse(extra).unwrap();
            assert!(
                args.validate().is_err(),
                "accepted invalid DB/generation args: {extra:?}"
            );
        };

        assert_invalid(&["--w-db", "1.01"]);
        assert_invalid(&["--p-db", "NaN"]);
        assert_invalid(&["--p-twist", "1.01"]);
        assert_invalid(&["--p-db-final", "1.01"]);
        assert_invalid(&["--db-min-window", "1"]);
        assert_invalid(&["--db-min-window", "5", "--db-max-window", "4"]);
        assert_invalid(&["--db-sample", "diagonal"]);
        assert_invalid(&["--db-convex-p", "1.01"]);
        assert_invalid(&["--db-max-degree", "12"]);
        assert_invalid(&["--db-max-degree", "9", "--db-degree-probes", "0"]);
        assert_invalid(&["--gen-bias", "1.01"]);
        assert_invalid(&["--p-db-ingest", "0.1"]);
        assert_invalid(&["--p-db-hard", "0.1"]);
        assert_invalid(&["--gen-target", "1", "--gen-rescan", "0"]);
        assert_invalid(&["--gen-stop-frac", "0.02"]);
        assert_invalid(&["--gen-target", "1", "--gen-stop-frac", "1.01"]);
        assert_invalid(&["--twist-cov-stop", "1"]);
        assert_invalid(&["--target-size", "2001", "--hard-size-cap", "2000"]);
        assert_invalid(&[
            "--grow-moves",
            "10",
            "--churn-moves",
            "20",
            "--churn-target-size",
            "2001",
            "--hard-size-cap",
            "2000",
        ]);
    }

    #[test]
    fn every_supported_db_sampler_parses() {
        for sampler in ["contiguous", "convex", "mixed"] {
            let args = parse(&["--db-sample", sampler]).unwrap();
            args.validate().unwrap();
            assert!(DbSample::parse(&args.db_sample).is_some());
        }
    }

    #[test]
    fn dose_then_churn_requires_an_explicit_enabled_dose_schedule() {
        assert!(parse(&["--dose-then-churn"]).is_err());
        assert!(
            parse(&[
                "--grow-moves",
                "10",
                "--churn-moves",
                "20",
                "--dose-then-churn",
            ])
            .is_err()
        );

        let missing_stop = parse(&[
            "--grow-moves",
            "10",
            "--churn-moves",
            "20",
            "--dose-then-churn",
            "--gen-target",
            "100",
        ])
        .unwrap();
        assert!(missing_stop.validate().is_err());

        let args = parse(&[
            "--grow-moves",
            "10",
            "--churn-moves",
            "20",
            "--dose-then-churn",
            "--gen-target",
            "100",
            "--gen-stop-frac",
            "0.02",
        ])
        .unwrap();
        args.validate().unwrap();
        assert!(args.dose_then_churn);

        let post_pair = parse(&[
            "--grow-moves",
            "10",
            "--churn-moves",
            "20",
            "--dose-then-churn",
            "--gen-target",
            "100",
            "--gen-stop-frac",
            "0.02",
            "--g57-pair-stage",
            "post",
            "--g57-pairs-per-wire",
            "1",
            "--g57-pair-target-wires",
            "4",
            "--g57-pair-control-wire-limit",
            "4",
            "--g57-pair-region",
            "uniform",
        ])
        .unwrap();
        assert!(post_pair.validate().is_err());
    }

    #[test]
    fn dose_transition_runs_exactly_the_requested_additional_moves() {
        let args = parse(&[
            "--grow-moves",
            "10",
            "--churn-moves",
            "3",
            "--dose-then-churn",
            "--gen-target",
            "1",
            "--gen-stop-frac",
            "0",
            "--report-every",
            "0",
            "--verify-every",
            "0",
        ])
        .unwrap();
        args.validate().unwrap();

        let params = MixParams {
            target_size: 1,
            moves: 10,
            report_every: 0,
            verify_every: 0,
            local_verify: false,
            gen_target: 1,
            gen_stop_frac: 0.0,
            ..MixParams::default()
        };
        let mut mixer = Mixer::new(vec![XGate::x_gate(0)], 2, params);
        assert_eq!(mixer.advance_all_generations_after_global_pass(), 1);
        // Demonstrate that the transition, rather than core construction,
        // retires these phase-A-only channels.
        mixer.params.p_db_ingest = 0.5;
        mixer.params.p_db_hard = 0.05;
        mixer.params.p_twist = 0.0016;

        let reason = transition_to_dose_churn(&mut mixer, &args, 3, "unit-test dose boundary");
        assert!(matches!(reason, MixStop::MovesBudget));
        assert_eq!(mixer.counters.moves, 3);
        assert_eq!(mixer.params.moves, 3);
        assert_eq!(mixer.params.gen_stop_frac, -1.0);
        assert_eq!(mixer.params.p_db_ingest, 0.0);
        assert_eq!(mixer.params.p_db_hard, 0.0);
        assert_eq!(mixer.params.p_twist, 0.0);
    }

    #[test]
    #[should_panic(expected = "did not complete its certified dose transition")]
    fn dose_then_churn_refuses_a_stop_flag_terminal() {
        let args = parse(&[
            "--grow-moves",
            "10",
            "--churn-moves",
            "3",
            "--dose-then-churn",
            "--gen-target",
            "1",
            "--gen-stop-frac",
            "0",
        ])
        .unwrap();
        args.validate().unwrap();
        require_complete_dose_churn(&MixStop::StopFlag, &args);
    }

    #[test]
    fn hard_cap_rejects_an_oversized_input_before_the_walk() {
        assert!(validate_hard_cap_admission(2_001, 1_000, 2_000).is_err());
        assert!(validate_hard_cap_admission(1_000, 2_001, 2_000).is_err());
        assert!(validate_hard_cap_admission(2_000, 2_000, 2_000).is_ok());
        assert!(validate_hard_cap_admission(usize::MAX, usize::MAX, 0).is_ok());
    }
}
