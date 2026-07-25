use clap::{Arg, Command};

mod commands;

// Shared argument set for the `sss` and `ssg` subcommands.
fn add_shoot_args(c: Command) -> Command {
    c.arg(Arg::new("n").short('n').long("n").required(true).value_parser(clap::value_parser!(usize)))
                .arg(Arg::new("m").short('m').long("m").required(true).value_parser(clap::value_parser!(usize)))
                .arg(Arg::new("x").short('x').long("x").required(true).value_parser(clap::value_parser!(usize)))
                .arg(
                    Arg::new("source")
                        .short('s')
                        .long("source")
                        .required(true)
                        .value_parser(clap::value_parser!(String))
                        .help("Path to the source circuit file"),
                )
                .arg(
                    Arg::new("rounds")
                        .short('r')
                        .long("rounds")
                        .required(true)
                        .value_parser(clap::value_parser!(usize))
                        .help("Number of rounds"),
                )
                .arg(
                    Arg::new("destination")
                        .short('d')
                        .long("destination")
                        .required(true)
                        .value_parser(clap::value_parser!(String))
                        .help("Path to the new circuit file"),
                )
                .arg(
                    Arg::new("interleave")
                        .long("interleave")
                        .help("Use interleaving")
                        .required(false)
                        .action(clap::ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("gadgetize")
                        .long("gadgetize")
                        .help("Gadgetize the circuit at the start (input becomes 2n wires)")
                        .required(false)
                        .action(clap::ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("feistalize")
                        .long("feistalize")
                        .help("Feistalize the circuit at the start (input becomes 3n wires)")
                        .required(false)
                        .conflicts_with("gadgetize")
                        .action(clap::ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("slice_zero")
                        .long("slice-zero")
                        .alias("slice_zero")
                        .help("Before feistalization, insert M that preserves the (y,z)=0 slice and randomizes x off it")
                        .required(false)
                        .requires("feistalize")
                        .action(clap::ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("slice_zero_random")
                        .long("slice-zero-random")
                        .alias("slice_zero_random")
                        .help("Before feistalization, insert M that preserves a public random (y,z) slice and randomizes x off it")
                        .required(false)
                        .requires("feistalize")
                        .conflicts_with("slice_zero")
                        .action(clap::ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("slice_zero_hardcoded")
                        .long("slice-zero-hardcoded")
                        .alias("slice_zero_hardcoded")
                        .help("Before feistalization, insert hardcoded M that preserves the (y,z)=0 slice")
                        .required(false)
                        .requires("feistalize")
                        .conflicts_with("slice_zero")
                        .conflicts_with("slice_zero_random")
                        .action(clap::ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("slice_zero_random_gates")
                        .long("slice-zero-random-gates")
                        .alias("slice_zero_random_gates")
                        .required(false)
                        .value_parser(clap::value_parser!(usize))
                        .help("Number of random M gates for --slice-zero-random (default: 32n)"),
                )
                .arg(
                    Arg::new("slice_zero_hardcoded_rounds")
                        .long("slice-zero-hardcoded-rounds")
                        .alias("slice_zero_hardcoded_rounds")
                        .required(false)
                        .default_value("1")
                        .value_parser(clap::value_parser!(usize))
                        .help("Number of hardcoded M rounds for --slice-zero-hardcoded"),
                )
                .arg(
                    Arg::new("slice_zero_ccnot")
                        .long("slice-zero-ccnot")
                        .alias("slice_zero_ccnot")
                        .help("(--cnot) Before gadgetization, insert an M of positive CNOT/CCNOTs (targets on data wires, every gate reading >=1 NON-DATA wire — aux AND the product-share band) that preserves the all-zero non-data slice and provably disturbs x on every other slice (pinned target/control split, rank-checked over the linear rows); three-control gates make the off-slice disturbance quadratic in x rather than affine")
                        .required(false)
                        .requires("gadgetize")
                        .requires("cnot")
                        .action(clap::ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("slice_zero_ccnot_gates")
                        .long("slice-zero-ccnot-gates")
                        .alias("slice_zero_ccnot_gates")
                        .required(false)
                        .value_parser(clap::value_parser!(usize))
                        .help("Number of M gates for --slice-zero-ccnot (default: 10n; ~1/3 CNOTs and ~2/3 CCNOTs in one uniformly random order). Must be at least n+band: every non-data wire gets a pin gate, which is what makes the disturbance exact"),
                )
                .arg(
                    Arg::new("sliced_sandwich")
                        .long("sliced-sandwich")
                        .alias("sliced_sandwich")
                        .help("(--cnot) Sliced-sandwich construction on 2n wires: [C interleaved with S1] ; y^=x ; [D interleaved with S2], where C is the source, D a random reversible circuit of m gates, and S1/S2 slice blocks of s CNOT/CCNOT gates (targets on wires 0..n, reading the second half). On the zero slice A(x,0)=(junk, C(x)) with the answer on wires n..2n")
                        .required(false)
                        .requires("cnot")
                        .conflicts_with("gadgetize")
                        .conflicts_with("feistalize")
                        .action(clap::ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("sandwich_m")
                        .long("sandwich-m")
                        .alias("sandwich_m")
                        .required(false)
                        .value_parser(clap::value_parser!(usize))
                        .help("Gates in the random D computation for --sliced-sandwich (default: n*(log2 n)^2)"),
                )
                .arg(
                    Arg::new("sandwich_s")
                        .long("sandwich-s")
                        .alias("sandwich_s")
                        .required(false)
                        .value_parser(clap::value_parser!(usize))
                        .help("Gates in each slice block S1, S2 for --sliced-sandwich (default: n*log2 n)"),
                )
                .arg(
                    Arg::new("gadget_path")
                        .long("gadget_path")
                        .required(false)
                        .value_parser(clap::value_parser!(String))
                        .help("Path to write the gadgetized/feistalized circuit (default: ./gadgetized/{source filename})"),
                )
                .arg(
                    Arg::new("full-shuffle")
                        .long("full-shuffle")
                        .help("Insert n SAMFs between every gate after each round's shooting insertion and before compression")
                        .required(false)
                        .conflicts_with("full-shuffle-early")
                        .action(clap::ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("full-shuffle-early")
                        .long("full-shuffle-early")
                        .help("Insert n SAMFs between every gate once after gadgetization/feistalization and before the main loop")
                        .required(false)
                        .conflicts_with("full-shuffle")
                        .action(clap::ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("gates_ahead_expand")
                        .long("gates_ahead_expand")
                        .required(false)
                        .default_value("2")
                        .value_parser(clap::value_parser!(usize))
                        .help("Gates in each curated expansion window, anchored at the colliding pair (2 = pair; >2 shrinks by 1 on a curated-DB miss down to the pair)"),
                )
                .arg(
                    Arg::new("gates_ahead_samf")
                        .long("gates_ahead_samf")
                        .required(false)
                        .default_value("3")
                        .value_parser(clap::value_parser!(usize))
                        .help("Context gates ending at the expansion tail (reaching into preceding output when the expansion is shorter) prepended to the 3 SAMF gates when hiding a SAMF"),
                )
                .arg(
                    Arg::new("type_attempts")
                        .long("type_attempts")
                        .required(false)
                        .default_value("1")
                        .value_parser(clap::value_parser!(usize))
                        .help("Distinct SAMF gate types to try per collision before giving up (each tries one random hardcoded SAMF of a not-yet-tried type)"),
                )
                .arg(
                    Arg::new("shooting_times")
                        .long("shooting_times")
                        .required(false)
                        .default_value("1")
                        .value_parser(clap::value_parser!(usize))
                        .help("Number of shooting rounds; each round runs one collision game then one plain SAMF insertion before the final unsamf"),
                )
                .arg(
                    Arg::new("rg_frequency")
                        .long("rg-frequency")
                        .required(false)
                        .default_value("2")
                        .value_parser(clap::value_parser!(usize))
                        .help("RG randomization rate. --cnot --gadgetize: number of nonlinear RGs (uniform RG1/RG2/RG3 draws) between consecutive SG gadgets (defaults to 1 there). Other paths: number of SG gadgets between each single RG (default 2)"),
                )
                .arg(
                    Arg::new("mask_cov")
                        .long("mask-cov")
                        .alias("mask_cov")
                        .required(false)
                        .default_value("0.0")
                        .value_parser(clap::value_parser!(f64))
                        .help("(--cnot gadgetize) Deferred-mask (RG4) coverage: target fraction of logical values carrying pending masks (0 = off, the current default). Actual coverage self-limits below 1 (masks need unmasked source carriers); ~0.75 is a workable validated setting. See docs/NONLINEAR_SHARE_ENCODING.md"),
                )
                .arg(
                    Arg::new("mask_k")
                        .long("mask-k")
                        .alias("mask_k")
                        .required(false)
                        .default_value("1")
                        .value_parser(clap::value_parser!(usize))
                        .help("(--cnot gadgetize) Pending mask terms per masked value (piling-up: k=1 caps degree-1 readout error at 0.25, k=3 at 0.4375)"),
                )
                .arg(
                    Arg::new("mask_depth")
                        .long("mask-depth")
                        .alias("mask_depth")
                        .required(false)
                        .default_value("2")
                        .value_parser(clap::value_parser!(usize))
                        .help("(--cnot gadgetize) Reserved for the deferred mask-tower thread (degree-3+ masks); a no-op in the v1 cascade-free ledger, which keeps masks at degree 2"),
                )
                .arg(
                    Arg::new("mask_taper")
                        .long("mask-taper")
                        .alias("mask_taper")
                        .required(false)
                        .value_parser(clap::value_parser!(usize))
                        .help("(--cnot gadgetize) Gaps before the body's end to stop re-injecting masks (default: max(4, n/5))"),
                )
                .arg(
                    Arg::new("prod_k")
                        .long("prod-k")
                        .alias("prod_k")
                        .required(false)
                        .default_value("0")
                        .value_parser(clap::value_parser!(usize))
                        .help("(--cnot gadgetize) Product-share encoding: permanent multiplicative mask terms per value, sourced on a frozen read-only band (0 = off). Degree-1 readout error floor: k=1 -> 0.25, k=2 -> 0.375, k=3 -> 0.4375. Replaces the CG menu with the share-native ANF fold (no operand reconstruction). Mutually exclusive with --mask-cov"),
                )
                .arg(
                    Arg::new("prod_deg")
                        .long("prod-deg")
                        .alias("prod_deg")
                        .required(false)
                        .default_value("2")
                        .value_parser(clap::value_parser!(usize))
                        .help("(--cnot gadgetize) Base product-mask degree = literals per term (min 2). deg=2 is the base encoding; deg=d hides the value from any reconstruction adversary of degree < d, at the cost of fold fragments up to width d*(gate arity)"),
                )
                .arg(
                    Arg::new("prod_k_hi")
                        .long("prod-k-hi")
                        .alias("prod_k_hi")
                        .required(false)
                        .default_value("0")
                        .value_parser(clap::value_parser!(usize))
                        .help("(--cnot gadgetize) Additional higher-degree tower mask terms per value (mixed design: k base deg-D terms + k_hi deg-D_hi terms; statistical strength from the base tier, algebraic hiding to deg_hi-1 from the tower tier)"),
                )
                .arg(
                    Arg::new("prod_deg_hi")
                        .long("prod-deg-hi")
                        .alias("prod_deg_hi")
                        .required(false)
                        .default_value("3")
                        .value_parser(clap::value_parser!(usize))
                        .help("(--cnot gadgetize) Degree of the --prod-k-hi tower terms (default 3)"),
                )
                .arg(
                    Arg::new("prod_band")
                        .long("prod-band")
                        .alias("prod_band")
                        .required(false)
                        .default_value("0")
                        .value_parser(clap::value_parser!(usize))
                        .help("(--cnot gadgetize) Source-band width in extra wires for --prod-k (0 = auto ~max(sqrt(4nk), deg+3))"),
                )
                .arg(
                    Arg::new("prod_rsrc")
                        .long("prod-rsrc")
                        .alias("prod_rsrc")
                        .required(false)
                        .default_value("1")
                        .value_parser(clap::value_parser!(usize))
                        .help("(--cnot gadgetize) Product-mask re-source moves per inter-SG gap (churn; 0 = off)"),
                )
                .arg(
                    Arg::new("prod_max_width")
                        .long("prod-max-width")
                        .alias("prod_max_width")
                        .required(false)
                        .default_value("0")
                        .value_parser(clap::value_parser!(usize))
                        .help("(--cnot gadgetize) Cap the encoding's emitted control width by laddering wider conjunctions over dedicated zero scratch wires (0 = legacy wide fragments; 2 = full narrow mode, every emitted gate a g57/CNOT — the phase-A DB vocabulary). Narrow mode is exact on the pinned zero-aux slice"),
                )
                .arg(
                    Arg::new("prod_fill_nl")
                        .long("prod-fill-nl")
                        .alias("prod_fill_nl")
                        .required(false)
                        .default_value("0")
                        .value_parser(clap::value_parser!(usize))
                        .help("(--cnot gadgetize) Nonlinear band fill: product terms per band wire, cascading over earlier band wires (0 = legacy linear fill). Kills learnable affine band invariants; input-degree multiplies up the band with only 2-control gates"),
                )
                .arg(
                    Arg::new("prod_roll")
                        .long("prod-roll")
                        .alias("prod_roll")
                        .required(false)
                        .default_value("0")
                        .value_parser(clap::value_parser!(usize))
                        .help("(--cnot gadgetize) Rolling band: band-variable relocations per inter-SG gap (0 = band stays on its home wires). One roll is RG2's 3-CNOT swap applied across the carrier/band boundary, so the band is not a body-static, statically identifiable wire set. Costs 3 CNOTs per roll"),
                )
                .arg(
                    Arg::new("egg")
                        .long("egg")
                        .help("Use expansion game (expand_loop 2x) instead of the shuffled shooting game")
                        .required(false)
                        .action(clap::ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("equality_check")
                        .long("equality_check")
                        .help("Run probabilistic equality/functionality checks after each round and at the end")
                        .required(false)
                        .action(clap::ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("single-end")
                        .long("single-end")
                        .help("Accumulate SAMFs/NOTs across ALL rounds (functionality is broken between rounds) and undo them in a single pass after the last round, before its compression")
                        .required(false)
                        .action(clap::ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("light_compression")
                        .long("light-compression")
                        .visible_alias("lc")
                        .short('l')
                        .help("Light compression: between rounds, stop compressing once the circuit is at most half its max (post-shooting) size")
                        .required(false)
                        .action(clap::ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("record_replacements")
                        .long("record")
                        .visible_alias("rc")
                        .help("Record every expansion/compression replacement (out-gate range, wires touched, incoming-gate count) to <destination>.replacements")
                        .required(false)
                        .action(clap::ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("track_survivors")
                        .long("track-survivors")
                        .visible_alias("ts")
                        .help("Record which pre-mixing gates are never part of any replacement, to <destination>.survivors")
                        .required(false)
                        .action(clap::ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("cnot")
                        .long("cnot")
                        .help("Keep all post-ingress stages in heterogeneous mpmct1 form, using native CNOTs/fragments where safe (the source remains G57). Routes to the CNOT gadgetizer + fmix mixer.")
                        .required(false)
                        .action(clap::ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("collision_rounds")
                        .long("collision_rounds")
                        .required(false)
                        .default_value("1")
                        .value_parser(clap::value_parser!(usize))
                        .help("(--cnot) Shuffled collision-game rounds before each plain SAMF insertion"),
                )
                .arg(
                    Arg::new("stable_compressions")
                        .long("stable_compressions")
                        .required(false)
                        .default_value("6")
                        .value_parser(clap::value_parser!(usize))
                        .help("(--cnot) Base stable-compression window; the final round uses 2x this"),
                )
}

fn main() {
    let matches = Command::new("local_mixing")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(add_shoot_args(
            Command::new("sss")
                .about("Shuffle-shoot-shuffle obfuscation + compression game"),
        ))
        .subcommand(
            add_shoot_args(Command::new("ssg").about(
                "ssg: generation-mixing variant of sss (generation tags + fanout/leeway selection)",
            ))
            .arg(
                Arg::new("max_fanout")
                    .long("max-fanout")
                    .required(false)
                    .default_value("50")
                    .value_parser(clap::value_parser!(usize))
                    .help("Hard cap on per-gate fanout (gen mode)"),
            )
            .arg(
                Arg::new("min_median_leeway")
                    .long("min-median-leeway")
                    .required(false)
                    .default_value("10")
                    .value_parser(clap::value_parser!(usize))
                    .help("Raise low-leeway gates when median leeway < this (gen mode)"),
            )
            .arg(
                Arg::new("min_gen")
                    .long("min-gen")
                    .required(false)
                    .default_value("1")
                    .value_parser(clap::value_parser!(usize))
                    .help("Stage B: keep shooting passes until every gate's generation >= this"),
            )
            .arg(
                Arg::new("min_gen_fraction")
                    .long("min-gen-fraction")
                    .required(false)
                    .default_value("0.99")
                    .value_parser(clap::value_parser!(f64))
                    .help("Stage B: stop once this fraction of gates reach --min-gen (default 0.99)"),
            )
            .arg(
                Arg::new("pass_length")
                    .long("pass-length")
                    .required(false)
                    .default_value("0")
                    .value_parser(clap::value_parser!(usize))
                    .help("Stage B: max successful replacements per shooting pass (0 = unbounded)"),
            )
            .arg(
                Arg::new("max_passes")
                    .long("max-passes")
                    .required(false)
                    .default_value("100000")
                    .value_parser(clap::value_parser!(usize))
                    .help("Stage B: safety cap on shooting passes per round"),
            )
            .arg(
                Arg::new("samf_target")
                    .long("samf-target")
                    .required(false)
                    .default_value("0")
                    .value_parser(clap::value_parser!(usize))
                    .help("If a round hides >= this many SAMFs, skip plain-SAMF insertion (m->0) for later rounds (0 = disabled)"),
            )
            .arg(
                Arg::new("grow_threshold")
                    .long("grow-threshold")
                    .required(false)
                    .default_value("0")
                    .value_parser(clap::value_parser!(f64))
                    .help("Stage D: size-threshold compression cadence. When > 0, ignore -r and instead alternate shoot/compress stages until the min-gen condition is met; each stage shoots until the circuit is this many PERCENT larger than the previous compressed size. Each compression stage is saved (<dest>stage<k>.txt). 0 = use fixed -r rounds."),
            )
            .arg(
                Arg::new("compress_fraction")
                    .long("compress-fraction")
                    .required(false)
                    .default_value("0")
                    .value_parser(clap::value_parser!(f64))
                    .help("Stage D only: compress each stage down to this FRACTION of the post-shooting size (e.g. 0.55), instead of compressing fully. With a 2x grow threshold, 0.55 nets +10% growth per round. 0 = compress fully each stage. Also the 'x' in --target-size."),
            )
            .arg(
                Arg::new("target_size")
                    .long("target-size")
                    .required(false)
                    .default_value("0")
                    .value_parser(clap::value_parser!(usize))
                    .help("Stage D absolute final/held size: each stage shoots until the circuit reaches TARGET-SIZE, then compresses back to (--compress-fraction * TARGET-SIZE); at the incompressibility ceiling the circuit pins at TARGET-SIZE. Overrides --grow-threshold. 0 = off."),
            )
            .arg(
                Arg::new("outgoing_rank_script")
                    .long("outgoing-rank-script")
                    .required(false)
                    .value_parser(clap::value_parser!(String))
                    .help("Rhai script providing rank(cands) for outgoing window selection (#11)"),
            )
            .arg(
                Arg::new("incoming_rank_script")
                    .long("incoming-rank-script")
                    .required(false)
                    .value_parser(clap::value_parser!(String))
                    .help("Rhai script providing rank(cands) for incoming replacement selection (#9)"),
            ),
        )
        .subcommand(
            Command::new("compress")
                .about("Run compression trials on a circuit file")
                .arg(
                    Arg::new("s")
                        .short('s')
                        .long("source")
                        .required(true)
                        .value_parser(clap::value_parser!(String))
                        .help("Path to the starting circuit file"),
                )
                .arg(
                    Arg::new("d")
                        .short('d')
                        .long("destination")
                        .required(true)
                        .value_parser(clap::value_parser!(String))
                        .help("Path to the new circuit file"),
                )
                .arg(
                    Arg::new("n")
                        .short('n')
                        .long("wires")
                        .required(true)
                        .value_parser(clap::value_parser!(usize))
                        .help("Number of wires in the circuit"),
                )
                .arg(
                    Arg::new("seq")
                        .long("seq")
                        .help("Enable seq mode")
                        .required(false)
                        .action(clap::ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("target_fraction")
                        .long("target-fraction")
                        .required(false)
                        .value_parser(clap::value_parser!(f64))
                        .help("Stop compressing early once the circuit reaches this fraction of its initial size (e.g. 0.5 = half). Combined with the usual no-progress stop; whichever triggers first wins. Absent = compress to convergence."),
                ),
        )
        .subcommand(
            Command::new("genran")
                .about("Generate a random circuit with n wires and m gates")
                .arg(
                    Arg::new("d")
                        .short('d')
                        .long("destination")
                        .required(true)
                        .value_parser(clap::value_parser!(String))
                        .help("Path to the new circuit file"),
                )
                .arg(
                    Arg::new("n")
                        .short('n')
                        .long("wires")
                        .required(true)
                        .value_parser(clap::value_parser!(usize))
                        .help("Number of wires in the circuit"),
                )
                .arg(
                    Arg::new("m")
                        .short('m')
                        .long("gates")
                        .required(true)
                        .value_parser(clap::value_parser!(usize))
                        .help("Number of gates in the circuit"),
                ),
        )
        .subcommand(
            Command::new("shuffle")
                .about("Shuffle a circuit")
                .arg(Arg::new("n").short('n').long("n").required(true).value_parser(clap::value_parser!(usize)))
                .arg(
                    Arg::new("s")
                        .short('s')
                        .long("source")
                        .required(true)
                        .value_parser(clap::value_parser!(String))
                        .help("Path to the source circuit file"),
                )
                .arg(Arg::new("i").short('i').long("iterations").required(true).value_parser(clap::value_parser!(usize)))
                .arg(
                    Arg::new("d")
                        .short('d')
                        .long("destination")
                        .required(true)
                        .value_parser(clap::value_parser!(String))
                        .help("Path to the new circuit file"),
                )
                .arg(
                    Arg::new("knuth")
                        .long("knuth")
                        .help("Use Knuth shuffle instead of simple")
                        .required(false)
                        .action(clap::ArgAction::SetTrue),
                ),
        )
        .subcommand(
            Command::new("shoot")
                .about("Shoot random gates through a circuit")
                .arg(Arg::new("i").short('i').long("iterations").required(true).value_parser(clap::value_parser!(usize)))
                .arg(
                    Arg::new("s")
                        .short('s')
                        .long("source")
                        .required(true)
                        .value_parser(clap::value_parser!(String))
                        .help("Path to the source circuit file"),
                )
                .arg(
                    Arg::new("d")
                        .short('d')
                        .long("destination")
                        .required(true)
                        .value_parser(clap::value_parser!(String))
                        .help("Path to the new circuit file"),
                ),
        )
        .subcommand(
            Command::new("equal")
                .about("Check if two circuits are functionally equivalent")
                .arg(
                    Arg::new("wires")
                        .short('n')
                        .long("wires")
                        .required(true)
                        .value_parser(clap::value_parser!(usize))
                        .help("Number of wires"),
                )
                .arg(
                    Arg::new("iterations")
                        .short('i')
                        .long("iterations")
                        .required(true)
                        .value_parser(clap::value_parser!(usize))
                        .help("Number of test iterations"),
                )
                .arg(
                    Arg::new("circuit_a")
                        .short('a')
                        .long("circuit-a")
                        .required(true)
                        .value_parser(clap::value_parser!(String))
                        .help("Path to first circuit file"),
                )
                .arg(
                    Arg::new("circuit_b")
                        .short('b')
                        .long("circuit-b")
                        .required(true)
                        .value_parser(clap::value_parser!(String))
                        .help("Path to second circuit file"),
                ),
        )
        .subcommand(
            Command::new("evaluate")
                .about("Evaluate a circuit on an input and print the output")
                .arg(
                    Arg::new("source")
                        .short('s')
                        .long("source")
                        .required(true)
                        .value_parser(clap::value_parser!(String))
                        .help("Path to the circuit file"),
                )
                .arg(
                    Arg::new("n")
                        .short('n')
                        .long("n")
                        .required(true)
                        .value_parser(clap::value_parser!(usize))
                        .help("Number of wires"),
                )
                .arg(
                    Arg::new("input")
                        .short('x')
                        .long("input")
                        .required(false)
                        .value_parser(clap::value_parser!(String))
                        .help("Input value (decimal or 0x-prefixed hex)"),
                )
                .arg(
                    Arg::new("random")
                        .short('r')
                        .long("random")
                        .required(false)
                        .action(clap::ArgAction::SetTrue)
                        .help("Use a random input (prints the chosen input)"),
                ),
        )
        .get_matches();

    match matches.subcommand() {
        Some(("sss", sub)) => commands::sss::run(sub),
        Some(("ssg", sub)) => commands::ssg::run(sub),
        Some(("compress", sub)) => commands::compress::run(sub),
        Some(("genran", sub)) => commands::genran::run(sub),
        Some(("shuffle", sub)) => commands::shuffle::run(sub),
        Some(("shoot", sub)) => commands::shoot::run(sub),
        Some(("equal", sub)) => commands::equal::run(sub),
        Some(("evaluate", sub)) => commands::evaluate::run(sub),
        _ => unreachable!("subcommand_required guarantees a match"),
    }
}
