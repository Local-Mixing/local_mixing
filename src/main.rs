use clap::{Arg, Command};

mod commands;

fn main() {
    let matches = Command::new("local_mixing")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(
            Command::new("sss")
                .about("Shuffle-shoot-shuffle obfuscation + compression game")
                .arg(Arg::new("n").short('n').long("n").required(true).value_parser(clap::value_parser!(usize)))
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
                    Arg::new("gadget_path")
                        .long("gadget_path")
                        .required(false)
                        .value_parser(clap::value_parser!(String))
                        .help("Path to write the gadgetized/feistalized circuit (default: ./gadgetized/{source filename})"),
                )
                .arg(
                    Arg::new("full-shuffle")
                        .long("full-shuffle")
                        .help("Insert n SAMFs between every gate once before the main loop")
                        .required(false)
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
                        .help("Number of shuffled shooting passes to run before SAMF insertion"),
                )
                .arg(
                    Arg::new("rg_frequency")
                        .long("rg-frequency")
                        .required(false)
                        .default_value("2")
                        .value_parser(clap::value_parser!(usize))
                        .help("Number of SG gadgets between each RG gadget (2 = two SGs then one RG)"),
                )
                .arg(
                    Arg::new("feistal_masked_sg")
                        .long("feistal-masked-sg")
                        .help("Feistalize with randomized masked SG updates")
                        .required(false)
                        .action(clap::ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("feistal_rg_refresh")
                        .long("feistal-rg-refresh")
                        .help("Feistalize with nonlinear null-refresh RG steps")
                        .required(false)
                        .action(clap::ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("feistal_slice_scramble_rounds")
                        .long("feistal-slice-scramble-rounds")
                        .required(false)
                        .default_value("0")
                        .value_parser(clap::value_parser!(usize))
                        .help("Valid-slice scrambler rounds before feistalization sharing"),
                )
                .arg(
                    Arg::new("egg")
                        .long("egg")
                        .help("Use expansion game (expand_loop 2x) instead of the shuffled shooting game")
                        .required(false)
                        .action(clap::ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("two_sided_mixing")
                        .long("two-sided-mixing")
                        .help("Run each local-mixing round once forward and once reversed")
                        .required(false)
                        .action(clap::ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("two_sided_candidates")
                        .long("two-sided-candidates")
                        .help("Run forward and reversed local-mixing candidates independently, then keep the smaller candidate")
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
                    Arg::new("compression_stable_max")
                        .long("compression-stable-max")
                        .required(false)
                        .default_value("6")
                        .value_parser(clap::value_parser!(usize))
                        .help("Compression early-stop window length"),
                )
                .arg(
                    Arg::new("compression_min_reduction")
                        .long("compression-min-reduction")
                        .required(false)
                        .default_value("50")
                        .value_parser(clap::value_parser!(usize))
                        .help("Stop compression when the stable window reduces by less than this many gates"),
                )
                .arg(
                    Arg::new("compression_max_iters")
                        .long("compression-max-iters")
                        .required(false)
                        .default_value("0")
                        .value_parser(clap::value_parser!(usize))
                        .help("Maximum compression passes per round; 0 means no explicit cap"),
                )
                .arg(
                    Arg::new("compression_max_seconds")
                        .long("compression-max-seconds")
                        .required(false)
                        .default_value("0")
                        .value_parser(clap::value_parser!(u64))
                        .help("Maximum seconds spent in compression per round; 0 means no explicit cap"),
                )
                .arg(
                    Arg::new("check_samples")
                        .long("check-samples")
                        .required(false)
                        .default_value("100000")
                        .value_parser(clap::value_parser!(usize))
                        .help("Random samples for per-round/final functionality checks"),
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
        .get_matches();

    match matches.subcommand() {
        Some(("sss", sub)) => commands::sss::run(sub),
        Some(("compress", sub)) => commands::compress::run(sub),
        Some(("genran", sub)) => commands::genran::run(sub),
        Some(("shuffle", sub)) => commands::shuffle::run(sub),
        Some(("shoot", sub)) => commands::shoot::run(sub),
        Some(("equal", sub)) => commands::equal::run(sub),
        _ => unreachable!("subcommand_required guarantees a match"),
    }
}
