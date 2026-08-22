use clap::{Arg, Command};

mod commands;

fn parse_regular_gate_count(raw: &str) -> Result<usize, String> {
    let value = raw
        .parse::<usize>()
        .map_err(|_| format!("expected an integer in 1..=21, got {raw:?}"))?;
    if !(1..=21).contains(&value) {
        return Err(format!(
            "gate count must be in 1..=21 so 3m fits u64 monomials, got {value}"
        ));
    }
    Ok(value)
}

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
                        .help("Gadgetize the circuit at the start (wire count depends on the selected representation)")
                        .required(false)
                        .action(clap::ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("five_carrier")
                        .long("five-carrier")
                        .alias("five_carrier")
                        .help("(--cnot --gadgetize) Use the five-carrier nonlinear product-share representation instead of the default single-carrier representation; output has 5n carrier wires plus the product band")
                        .required(false)
                        .requires("gadgetize")
                        .requires("cnot")
                        .conflicts_with("prod_single")
                        .action(clap::ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("strong_five_carrier")
                        .long("strong-five-carrier")
                        .alias("strong_five_carrier")
                        .help("(--cnot --gadgetize) Use the experimental cubic five-carrier representation: zero endpoint parity correlation through weight two and no exact degree-two recovery; output has 5n carrier wires plus the product band")
                        .required(false)
                        .requires("gadgetize")
                        .requires("cnot")
                        .conflicts_with("five_carrier")
                        .conflicts_with("prod_single")
                        .action(clap::ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("six_carrier")
                        .long("six-carrier")
                        .alias("six_carrier")
                        .help("(--cnot --gadgetize) Use the six-carrier nonlinear product-share representation; its endpoint trace has zero parity correlation through weight three and no exact degree-two recovery; output has 6n carrier wires plus the product band")
                        .required(false)
                        .requires("gadgetize")
                        .requires("cnot")
                        .conflicts_with("five_carrier")
                        .conflicts_with("strong_five_carrier")
                        .conflicts_with("prod_single")
                        .action(clap::ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("strong_six_carrier")
                        .long("strong-six-carrier")
                        .alias("strong_six_carrier")
                        .help("(--cnot --gadgetize) Use the experimental structural six-carrier representation: the cubic decode and endpoint spectrum are retained, while the update has full affine graph rank and no frozen carrier lane; output has 6n carrier wires plus the product band")
                        .required(false)
                        .requires("gadgetize")
                        .requires("cnot")
                        .conflicts_with("five_carrier")
                        .conflicts_with("strong_five_carrier")
                        .conflicts_with("six_carrier")
                        .conflicts_with("seven_carrier")
                        .conflicts_with("prod_single")
                        .action(clap::ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("seven_carrier")
                        .long("seven-carrier")
                        .alias("seven_carrier")
                        .help("(--cnot --gadgetize) Use the seven-carrier nonlinear product-share representation; its endpoint trace has zero parity correlation through weight three and no exact degree-three recovery; output has 7n carrier wires plus the product band")
                        .required(false)
                        .requires("gadgetize")
                        .requires("cnot")
                        .conflicts_with("five_carrier")
                        .conflicts_with("strong_five_carrier")
                        .conflicts_with("six_carrier")
                        .conflicts_with("strong_six_carrier")
                        .conflicts_with("prod_single")
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
                        .help("(--cnot) Before gadgetization, insert an M of positive CNOT/CCNOTs (targets on data wires, every gate reading >=1 NON-DATA wire — auxiliary carriers AND the product-share band) that preserves the all-zero non-data slice and provably disturbs x on every other slice (pinned target/control split, rank-checked over the linear rows); three-control gates make the off-slice disturbance quadratic in x rather than affine")
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
                        .help("Number of M gates for --slice-zero-ccnot (default: 10n; ~1/3 CNOTs and ~2/3 CCNOTs in one uniformly random order). The effective block covers every non-data wire: band in single-carrier mode, n+band in paired mode, 4n+band with either five-carrier mode, 5n+band with either six-carrier mode, or 6n+band with --seven-carrier; smaller requests are raised to that minimum"),
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
                        .value_parser(clap::value_parser!(usize))
                        .help("(--cnot gadgetize) Product-share encoding: permanent multiplicative mask terms per value, sourced on a band (0 = off). The --prod-* flags inherit the selected representation's coherent production preset (single-carrier by default, nonlinear under a five/six/seven-carrier flag), and an explicitly passed flag overrides just that field. Pass --prod-k 0 and --prod-k-hi 0 for no product masks in the legacy representations; nonlinear carrier modes require a nonempty mask plan. Replaces the CG menu with a share-native fold (no operand reconstruction). Mutually exclusive with --mask-cov"),
                )
                .arg(
                    Arg::new("prod_deg")
                        .long("prod-deg")
                        .alias("prod_deg")
                        .required(false)
                        .value_parser(clap::value_parser!(usize))
                        .help("(--cnot gadgetize) Base product-mask degree = literals per term (min 2). deg=2 is the base encoding; deg=d hides the value from any reconstruction adversary of degree < d, at the cost of fold fragments up to width d*(gate arity)"),
                )
                .arg(
                    Arg::new("prod_k_hi")
                        .long("prod-k-hi")
                        .alias("prod_k_hi")
                        .required(false)
                        .value_parser(clap::value_parser!(usize))
                        .help("(--cnot gadgetize) Additional higher-degree tower mask terms per value (mixed design: k base deg-D terms + k_hi deg-D_hi terms; statistical strength from the base tier, algebraic hiding to deg_hi-1 from the tower tier)"),
                )
                .arg(
                    Arg::new("prod_deg_hi")
                        .long("prod-deg-hi")
                        .alias("prod_deg_hi")
                        .required(false)
                        .value_parser(clap::value_parser!(usize))
                        .help("(--cnot gadgetize) Degree of the --prod-k-hi tower terms (default 3)"),
                )
                .arg(
                    Arg::new("prod_band")
                        .long("prod-band")
                        .alias("prod_band")
                        .required(false)
                        .value_parser(clap::value_parser!(usize))
                        .help("(--cnot gadgetize) Source-band width in extra wires for --prod-k. DEFAULT 0 = match the value count, giving a 1:1 carrier/band split -- that is what makes the write census fail to separate the two populations (185/452/847 against 180/428/848 at n=128), where a narrow band is a minority a windowed census still isolates. Band = n also leaves --prod-fill-pivots no room, which is the homogeneity-versus-provable-uniformity trade. Pass a number for any other sizing; the pre-2026-07-26 auto rule was ceil(sqrt(4nk)) = 56 at n=256"),
                )
                .arg(
                    Arg::new("prod_rsrc")
                        .long("prod-rsrc")
                        .alias("prod_rsrc")
                        .required(false)
                        .value_parser(clap::value_parser!(usize))
                        .help("(--cnot gadgetize) Product-mask re-source moves per inter-SG gap (churn; 0 = off)"),
                )
                .arg(
                    Arg::new("prod_max_width")
                        .long("prod-max-width")
                        .alias("prod_max_width")
                        .required(false)
                        .value_parser(clap::value_parser!(usize))
                        .help("(--cnot gadgetize) Cap legacy product-fold conjunction width by laddering wider conjunctions over dedicated zero scratch wires (0 = legacy wide fragments; 2 = full narrow mode, every legacy fold gate a g57/CNOT — the phase-A DB vocabulary). The nonlinear carrier decodes have their own exact update, port, and fallback gates and do not currently honor this legacy fold cap. Narrow mode is exact on the pinned zero-aux slice"),
                )
                .arg(
                    Arg::new("prod_fill_nl")
                        .long("prod-fill-nl")
                        .alias("prod_fill_nl")
                        .required(false)
                        .value_parser(clap::value_parser!(usize))
                        .help("(--cnot gadgetize) Nonlinear band fill: product terms per band wire, cascading over earlier band wires (0 = legacy linear fill). Kills learnable affine band invariants; input-degree multiplies up the band with only 2-control gates"),
                )
                .arg(
                    Arg::new("prod_roll")
                        .long("prod-roll")
                        .alias("prod_roll")
                        .required(false)
                        .value_parser(clap::value_parser!(usize))
                        .help("(--cnot gadgetize) Rolling band: band-variable relocations per inter-SG gap (0 = band stays on its home wires). One roll is RG2's 3-CNOT swap applied across the carrier/band boundary, so the band is not a body-static, statically identifiable wire set. Costs 3 CNOTs per roll"),
                )
                .arg(
                    Arg::new("prod_single")
                        .long("prod-single")
                        .alias("prod_single")
                        .required(false)
                        .value_parser(clap::value_parser!(usize))
                        .help("(--cnot gadgetize) Single-carrier decode: ONE degree-1 term per value instead of two (0 = the legacy carrier pair). The second carrier is free to an affine adversary, so it adds nothing to the piling-up product; dropping it halves the carriers and cuts the fold to (1+k)^arity. Spend the freed atom on a mask: --prod-k 1 --prod-deg 2 --prod-k-hi 2 --prod-deg-hi 3 is [1,2,3,3] (recommended), --prod-k 2 --prod-k-hi 1 is [1,2,2,3] (better degree-1 statistics, half the degree-2 margin). Requires --prod-rsrc >= 1: with one carrier, re-sourcing is what refreshes the representation"),
                )
                .arg(
                    Arg::new("prod_gray_fold")
                        .long("prod-gray-fold")
                        .alias("prod_gray_fold")
                        .required(false)
                        .value_parser(clap::value_parser!(usize))
                        .help("(--cnot gadgetize) Product-fold mode: 0=expanded/no aggregate; 1=aggregate Gray (default and smallest, but one accumulator before/after gather reveals the complete operand mask); 2=four-share micro Gray (16 restored rectangles, removes the single-interval witness but the four public share deltas recombine); 3=max-degree-sentinel Gray (gathers only lower-degree Q and keeps maximum-degree H out of every accumulator; experimental). Modes 2/3 trade substantially more gates for a higher-order temporal witness. All modes preserve arbitrary dirty helpers; these are measured hardening options, not cryptographic proofs"),
                )
                .arg(
                    Arg::new("prod_g57_narrow")
                        .long("prod-g57-narrow")
                        .alias("prod_g57_narrow")
                        .required(false)
                        .value_parser(clap::value_parser!(usize))
                        .help("(--cnot gadgetize) Realize width-<=2 fold fragments in the g57/CNOT vocabulary instead of as bare conjunctions. MEASURED at n=128 band 256: DB match rate 35.1% -> 41.5% for +4.5% gates, via the pure sampler. The MECHANISM is open: an earlier rationale claimed a comp=0 width-2 conjunction is unreachable by a g57-built store because it lies outside the X-free span <x, y, 1^xy> over {h,x,y}; that span identity is real (it is the f with const = coeff_xy, and it is why three of four polarity patterns owe a ledger constant) but it only covers circuits that never write x or y -- g57+CNOT on three wires generates all of S8, so h ^= xy is reachable in 5 gates. Keep the flag for the measurement, not the argument"),
                )
                .arg(
                    Arg::new("prod_ladder_cap")
                        .long("prod-ladder-cap")
                        .alias("prod_ladder_cap")
                        .required(false)
                        .value_parser(clap::value_parser!(usize))
                        .help("(--cnot gadgetize) Ladder fold fragments of width in (2, cap] down to <=2 controls over borrowed dirty carriers; wider fragments stay as single wide gates (0 = no laddering). This is the knob that reduces the ABSOLUTE count of >2-control gates, which --prod-max-width 2 does only by laddering everything at roughly 6.2x the fold"),
                )
                .arg(
                    Arg::new("prod_cg_jitter")
                        .long("prod-cg-jitter")
                        .alias("prod_cg_jitter")
                        .required(false)
                        .value_parser(clap::value_parser!(usize))
                        .help("(--cnot gadgetize) Percent of values carrying one EXTRA LOW-degree mask term. Breaks the fixed (1+k_total)^arity fragment count per CG block, which otherwise segments the circuit into blocks and reveals each source gate's arity by counting. Jitter is extra terms only, never fewer, so the committed operating point (the weakest value) does not move. NOTE: the per-fragment ESOP re-cover conj(L+u)^conj(L+!u) was tried for this and REFUTED -- the two halves share every literal but one polarity and target the same carrier, so the twin is greppable within the block whatever order they are emitted in"),
                )
                .arg(
                    Arg::new("prod_rung_menu")
                        .long("prod-rung-menu")
                        .alias("prod_rung_menu")
                        .required(false)
                        .value_parser(clap::value_parser!(usize))
                        .help("(--cnot gadgetize) Spelling variability for the ladder's double sweep, which emits its rung AND its target gate twice: identical copies plant exact gate pairs, so a `sort | uniq -c` census with no execution locates every laddered fragment, its borrowed wire and two of its three literals. 0 = one fixed spelling. 1 = vary only where the equivalent spellings are the SAME SIZE, which over this gate set means same-polarity emissions (a 2-subset has two 2-gate spellings; a mixed-polarity function IS a generator, so its spellings are 1 and 3) -- measured n=16 at ladder_cap 3, identical gate count to level 0 with width-2 duplicate groups 60.7% -> 52.6%. 2 = every spelling including the longer ones: 41.0%, but +18.8% gates. Correctness is unaffected at every level, since all spellings of one emission contribute the same function"),
                )
                .arg(
                    Arg::new("prod_fill_pivots")
                        .long("prod-fill-pivots")
                        .alias("prod_fill_pivots")
                        .required(false)
                        .value_parser(clap::value_parser!(usize))
                        .help("(--cnot gadgetize) Reserve a distinct pivot per band wire during the nonlinear fill, making the band exactly uniform on {0,1}^b at the port. Requires band <~ 3n/4 (the pivots leave the fill no legal non-pivot data wire otherwise). NOTE: --prod-refill-data > 0 forfeits the guarantee past the port"),
                )
                .arg(
                    Arg::new("prod_epoch")
                        .long("prod-epoch")
                        .alias("prod_epoch")
                        .required(false)
                        .value_parser(clap::value_parser!(usize))
                        .help("(--cnot gadgetize) Retire-and-refill epochs: a band variable is re-sourced everywhere it is named and then rewritten, so band values stop being frozen functions of the input (0 = frozen band). Without this the band is recoverable by FUNCTION LIFETIME alone -- rolling relocates a variable but preserves its function. Smaller = more turnovers = more cost; merging the population fully costs about +41% gates"),
                )
                .arg(
                    Arg::new("prod_refill_data")
                        .long("prod-refill-data")
                        .alias("prod_refill_data")
                        .required(false)
                        .value_parser(clap::value_parser!(usize))
                        .help("(--cnot gadgetize) Percent of refill sources drawn from CARRIERS rather than from other band values. Band-only refills keep the band statistically independent of the data (good for correlation) but a band that never touches the data is itself a signature; carrier-sourced refills inject data at the cost of the port's joint-uniformity guarantee"),
                )
                .arg(
                    Arg::new("prod_src_dist")
                        .long("prod-src-dist")
                        .alias("prod_src_dist")
                        .required(false)
                        .value_parser(clap::value_parser!(usize))
                        .help("(--cnot gadgetize) Distributed sourcing: drop the dedicated band and source mask literals on ordinary carriers of other values, migrating a mask before anything writes its source (0 = band, 1 = no band). Costs no extra wires and leaves no globally quiet wire"),
                )
                .arg(
                    Arg::new("prod_src_horizon")
                        .long("prod-src-horizon")
                        .alias("prod_src_horizon")
                        .required(false)
                        .value_parser(clap::value_parser!(usize))
                        .help("(--cnot gadgetize) Lookahead horizon for --prod-src-dist, in source-gate positions (0 = auto n/2): prefer sources whose owning value is not targeted within this many gates, so masks usually die of ordinary churn rather than a forced migration"),
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
        .subcommand(commands::gss::command())
        // Offline regular replacement-DB generation. These commands retain
        // their historical names and working-directory path contract so old
        // rebuild recipes continue to work.
        .subcommand(
            Command::new("rocksdb_1")
                .hide(!cfg!(feature = "legacy-db-tools"))
                .about("Build an m-gate RocksDB by extending the (m-1)-gate DB")
                .arg(
                    Arg::new("m")
                        .short('m')
                        .long("m")
                        .required(true)
                        .value_parser(parse_regular_gate_count)
                        .help("Number of gates (1..=21; 3m must fit u64 monomials)"),
                )
                .arg(
                    Arg::new("min_n")
                        .long("min_n")
                        .required(false)
                        .default_value("0")
                        .value_parser(clap::value_parser!(usize))
                        .help("Minimum used-wire count (0 = no lower bound)"),
                )
                .arg(
                    Arg::new("max_n")
                        .long("max_n")
                        .required(false)
                        .default_value("0")
                        .value_parser(clap::value_parser!(usize))
                        .help("Maximum used-wire count (0 = no upper bound)"),
                )
                .arg(
                    Arg::new("no_L")
                        .long("no_L")
                        .action(clap::ArgAction::SetTrue)
                        .help("Skip candidates whose canonicalization requires Rule L (no effect for m=1)"),
                ),
        )
        .subcommand(
            Command::new("rocksdb_2")
                .hide(!cfg!(feature = "legacy-db-tools"))
                .about("Build an (m1+m2)-gate RocksDB by combining two source DBs")
                .arg(
                    Arg::new("m1")
                        .long("m1")
                        .required(true)
                        .value_parser(parse_regular_gate_count)
                        .help("Gate count of the first source DB"),
                )
                .arg(
                    Arg::new("m2")
                        .long("m2")
                        .required(true)
                        .value_parser(parse_regular_gate_count)
                        .help("Gate count of the second source DB"),
                )
                .arg(
                    Arg::new("min_n")
                        .long("min_n")
                        .required(false)
                        .default_value("0")
                        .value_parser(clap::value_parser!(usize))
                        .help("Minimum used-wire count (0 = no lower bound)"),
                ),
        )
        .subcommand(
            Command::new("rocks_to_lmdb")
                .hide(!cfg!(feature = "legacy-db-tools"))
                .about("Convert a combined RocksDB into 256 sharded LMDB databases")
                .arg(
                    Arg::new("source")
                        .short('s')
                        .long("source")
                        .required(true)
                        .value_parser(clap::value_parser!(String))
                        .help("Source RocksDB path"),
                )
                .arg(
                    Arg::new("path")
                        .short('p')
                        .long("path")
                        .required(true)
                        .value_parser(clap::value_parser!(String))
                        .help("Output LMDB directory"),
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
        Some(("gss", sub)) => commands::gss::run(sub),
        Some(("rocksdb_1", sub)) => commands::db_generation::run_rocksdb_1(sub),
        Some(("rocksdb_2", sub)) => commands::db_generation::run_rocksdb_2(sub),
        Some(("rocks_to_lmdb", sub)) => commands::db_generation::run_rocks_to_lmdb(sub),
        _ => unreachable!("subcommand_required guarantees a match"),
    }
}

#[cfg(test)]
mod cli_tests {
    use super::*;

    #[test]
    fn regular_gate_count_parser_enforces_the_monomial_abi() {
        assert_eq!(parse_regular_gate_count("1"), Ok(1));
        assert_eq!(parse_regular_gate_count("21"), Ok(21));
        assert!(parse_regular_gate_count("0").is_err());
        assert!(parse_regular_gate_count("22").is_err());
        assert!(parse_regular_gate_count("not-a-number").is_err());
    }

    fn parse_shoot_args(extra: &[&str]) -> Result<clap::ArgMatches, clap::Error> {
        let mut args = vec![
            "sss",
            "--n",
            "3",
            "--m",
            "1",
            "--x",
            "1",
            "--source",
            "source.g57",
            "--rounds",
            "0",
            "--destination",
            "out.mpmct1",
        ];
        args.extend_from_slice(extra);
        add_shoot_args(Command::new("sss")).try_get_matches_from(args)
    }

    #[test]
    fn five_carrier_flag_parses_for_the_cnot_gadgetizer() {
        let matches = parse_shoot_args(&["--cnot", "--gadgetize", "--five-carrier"]).unwrap();
        assert!(matches.get_flag("five_carrier"));
    }

    #[test]
    fn five_carrier_flag_requires_cnot_and_gadgetize() {
        assert!(parse_shoot_args(&["--gadgetize", "--five-carrier"]).is_err());
        assert!(parse_shoot_args(&["--cnot", "--five-carrier"]).is_err());
    }

    #[test]
    fn five_carrier_flag_rejects_the_single_carrier_override() {
        assert!(
            parse_shoot_args(&[
                "--cnot",
                "--gadgetize",
                "--five-carrier",
                "--prod-single",
                "1",
            ])
            .is_err()
        );
    }

    #[test]
    fn strong_five_carrier_flag_parses_and_conflicts_with_legacy_five() {
        let matches =
            parse_shoot_args(&["--cnot", "--gadgetize", "--strong-five-carrier"]).unwrap();
        assert!(matches.get_flag("strong_five_carrier"));
        assert!(
            parse_shoot_args(&[
                "--cnot",
                "--gadgetize",
                "--five-carrier",
                "--strong-five-carrier",
            ])
            .is_err()
        );
        assert!(
            parse_shoot_args(&[
                "--cnot",
                "--gadgetize",
                "--strong-five-carrier",
                "--prod-single",
                "1",
            ])
            .is_err()
        );
    }

    #[test]
    fn six_carrier_flag_parses_for_the_cnot_gadgetizer() {
        let matches = parse_shoot_args(&["--cnot", "--gadgetize", "--six-carrier"]).unwrap();
        assert!(matches.get_flag("six_carrier"));
    }

    #[test]
    fn six_carrier_flag_requires_cnot_and_gadgetize() {
        assert!(parse_shoot_args(&["--gadgetize", "--six-carrier"]).is_err());
        assert!(parse_shoot_args(&["--cnot", "--six-carrier"]).is_err());
    }

    #[test]
    fn strong_six_carrier_flag_parses_and_conflicts_with_legacy_six() {
        let matches = parse_shoot_args(&["--cnot", "--gadgetize", "--strong-six-carrier"]).unwrap();
        assert!(matches.get_flag("strong_six_carrier"));
        assert!(
            parse_shoot_args(&[
                "--cnot",
                "--gadgetize",
                "--six-carrier",
                "--strong-six-carrier",
            ])
            .is_err()
        );
        assert!(
            parse_shoot_args(&[
                "--cnot",
                "--gadgetize",
                "--strong-six-carrier",
                "--prod-single",
                "1",
            ])
            .is_err()
        );
    }

    #[test]
    fn seven_carrier_flag_parses_for_the_cnot_gadgetizer() {
        let matches = parse_shoot_args(&["--cnot", "--gadgetize", "--seven-carrier"]).unwrap();
        assert!(matches.get_flag("seven_carrier"));
    }

    #[test]
    fn seven_carrier_flag_requires_cnot_and_gadgetize() {
        assert!(parse_shoot_args(&["--gadgetize", "--seven-carrier"]).is_err());
        assert!(parse_shoot_args(&["--cnot", "--seven-carrier"]).is_err());
    }

    #[test]
    fn nonlinear_carrier_flags_are_mutually_exclusive() {
        assert!(
            parse_shoot_args(&["--cnot", "--gadgetize", "--five-carrier", "--six-carrier",])
                .is_err()
        );
        assert!(
            parse_shoot_args(&[
                "--cnot",
                "--gadgetize",
                "--six-carrier",
                "--prod-single",
                "1",
            ])
            .is_err()
        );
        assert!(
            parse_shoot_args(&["--cnot", "--gadgetize", "--six-carrier", "--seven-carrier",])
                .is_err()
        );
        assert!(
            parse_shoot_args(&["--cnot", "--gadgetize", "--five-carrier", "--seven-carrier",])
                .is_err()
        );
        assert!(
            parse_shoot_args(&[
                "--cnot",
                "--gadgetize",
                "--seven-carrier",
                "--prod-single",
                "1",
            ])
            .is_err()
        );
    }
}
