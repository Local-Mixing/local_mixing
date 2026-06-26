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
                        .requires("slice_zero_random")
                        .conflicts_with("M_length")
                        .value_parser(clap::value_parser!(usize))
                        .help("Deprecated alias for --M_length"),
                )
                .arg(
                    Arg::new("M_length")
                        .long("M_length")
                        .required(false)
                        .requires("slice_zero_random")
                        .conflicts_with("slice_zero_random_gates")
                        .value_parser(clap::value_parser!(usize))
                        .help("Number of random M gates for --slice-zero-random (default: 20n)"),
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
                        .help("Number of collision-game passes; each pass runs collision_rounds collision games then one plain SAMF insertion before the final unsamf"),
                )
                .arg(
                    Arg::new("collision_rounds")
                        .long("collision_rounds")
                        .required(false)
                        .default_value("1")
                        .value_parser(clap::value_parser!(usize))
                        .help("Number of shuffled collision-game rounds to run before each plain SAMF insertion"),
                )
                .arg(
                    Arg::new("stable_compressions")
                        .long("stable_compressions")
                        .required(false)
                        .default_value("6")
                        .value_parser(clap::value_parser!(usize))
                        .help("Base stable-compression window before stopping; the final round uses 2x this value"),
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
                    Arg::new("expansion_game")
                        .long("expansion_game")
                        .alias("egg")
                        .help("Before each collision-game pass, run one curated expansion loop pass")
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
                    Arg::new("stable_compressions")
                        .long("stable_compressions")
                        .required(false)
                        .default_value("6")
                        .value_parser(clap::value_parser!(usize))
                        .help("Base stable-compression window before stopping; this final compression uses 2x this value"),
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
        Some(("compress", sub)) => commands::compress::run(sub),
        Some(("genran", sub)) => commands::genran::run(sub),
        Some(("shuffle", sub)) => commands::shuffle::run(sub),
        Some(("shoot", sub)) => commands::shoot::run(sub),
        Some(("equal", sub)) => commands::equal::run(sub),
        Some(("evaluate", sub)) => commands::evaluate::run(sub),
        _ => unreachable!("subcommand_required guarantees a match"),
    }
}
