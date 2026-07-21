use clap::{Arg, ArgGroup, Command};

mod commands;

fn main() {
    let matches = Command::new("local_mixing")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(
            Command::new("sss")
                .about("Shuffle-shoot-shuffle obfuscation + compression game")
                .group(
                    ArgGroup::new("fixed_slice_transform")
                        .args(["feistalize", "tdp4n"])
                        .multiple(false),
                )
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
                    Arg::new("cnot")
                        .long("cnot")
                        .help("Keep all post-ingress stages in heterogeneous mpmct1 form, using native CNOTs/fragments where safe (the source remains G57)")
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
                    Arg::new("tdp4n")
                        .long("tdp4n")
                        .help("Build C; native CNOT X->Y; random D on X, then two-share gadgetize the 2n logical wires (4n physical wires)")
                        .required(false)
                        .requires("cnot")
                        .conflicts_with("gadgetize")
                        .conflicts_with("feistalize")
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
                        .help("Before Feistal/4n-TDP construction, insert M that preserves a public random (y,z) slice and randomizes x off it")
                        .required(false)
                        .requires("fixed_slice_transform")
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
                )
                .arg(
                    Arg::new("record_replacements")
                        .long("record")
                        .alias("record_replacements")
                        .help("Record expansion/compression replacements to <destination>.replacements")
                        .required(false)
                        .action(clap::ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("generation_tags")
                        .long("generation-tags")
                        .alias("gen-tags")
                        .help("Maintain generation tags and write <destination>.generations")
                        .required(false)
                        .action(clap::ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("outgoing_mode")
                        .long("outgoing-mode")
                        .required(false)
                        .default_value("legacy")
                        .value_parser(["legacy", "gen"])
                        .help("Outgoing shooting-window mode: legacy or gen"),
                )
                .arg(
                    Arg::new("incoming_rank")
                        .long("incoming-rank")
                        .required(false)
                        .default_value("sat")
                        .value_parser(["sat", "fanout", "hybrid"])
                        .help("Incoming replacement ranker: sat, fanout, or hybrid"),
                )
                .arg(
                    Arg::new("max_fanout")
                        .long("max-fanout")
                        .required(false)
                        .default_value("50")
                        .value_parser(clap::value_parser!(usize))
                        .help("Fanout cap used by fanout/hybrid incoming ranking"),
                )
                .arg(
                    Arg::new("min_median_leeway")
                        .long("min-median-leeway")
                        .required(false)
                        .default_value("10")
                        .value_parser(clap::value_parser!(usize))
                        .help("Low-leeway threshold used by fanout/hybrid ranking"),
                )
                .arg(
                    Arg::new("samf_target")
                        .long("samf-target")
                        .required(false)
                        .default_value("0")
                        .value_parser(clap::value_parser!(usize))
                        .help("If a round hides at least this many SAMFs, set m=0 for later rounds; 0 disables"),
                )
                .arg(
                    Arg::new("min_gen")
                        .long("min-gen")
                        .required(false)
                        .default_value("0")
                        .value_parser(clap::value_parser!(usize))
                        .help("Stage B: keep low-generation shooting passes until this generation; 0 disables"),
                )
                .arg(
                    Arg::new("min_gen_fraction")
                        .long("min-gen-fraction")
                        .required(false)
                        .default_value("0.99")
                        .value_parser(clap::value_parser!(f64))
                        .help("Stage B: stop once this fraction of gates reach --min-gen"),
                )
                .arg(
                    Arg::new("pass_length")
                        .long("pass-length")
                        .required(false)
                        .default_value("0")
                        .value_parser(clap::value_parser!(usize))
                        .help("Stage B: max successful outgoing replacements per low-gen pass; 0 is unbounded"),
                )
                .arg(
                    Arg::new("max_passes")
                        .long("max-passes")
                        .required(false)
                        .default_value("100000")
                        .value_parser(clap::value_parser!(usize))
                        .help("Stage B: safety cap on low-generation shooting passes per round"),
                )
                .arg(
                    Arg::new("grow_threshold")
                        .long("grow-threshold")
                        .required(false)
                        .default_value("0")
                        .value_parser(clap::value_parser!(f64))
                        .help("SSS/SSG hybrid cadence: when >0, ignore --rounds and compress whenever a stage grows by this percentage from its previous compressed size"),
                )
                .arg(
                    Arg::new("compress_fraction")
                        .long("compress-fraction")
                        .required(false)
                        .default_value("0")
                        .value_parser(clap::value_parser!(f64))
                        .help("SSS/SSG hybrid cadence: stop each compression at this fraction of the post-shooting size, or of --target-size; 0 compresses to the normal stall point"),
                )
                .arg(
                    Arg::new("target_size")
                        .long("target-size")
                        .required(false)
                        .default_value("0")
                        .value_parser(clap::value_parser!(usize))
                        .help("SSS/SSG hybrid cadence: absolute per-stage shooting cap and held final size; overrides --grow-threshold and stops only on the min-generation condition"),
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
        // Rainbow-table (replacement DB) generation pipeline; see
        // src/rainbow_table/mod.rs. Paths are relative to the working
        // directory: reads ./rocks_db_m*, writes ./test_rocks_db_m*.
        .subcommand(
            Command::new("rocksdb_1")
                .hide(!cfg!(feature = "legacy-db-tools"))
                .about("Create m sized rocks_db based on m-1")
                .arg(
                    Arg::new("m")
                        .short('m')
                        .long("m")
                        .required(true)
                        .value_parser(clap::value_parser!(usize))
                        .help("Number of gates"),
                )
                .arg(
                    Arg::new("min_n")
                        .long("min_n")
                        .required(false)
                        .default_value("0")
                        .value_parser(clap::value_parser!(usize))
                        .help("Minimum number of used wires; candidates with fewer are skipped"),
                )
                .arg(
                    Arg::new("max_n")
                        .long("max_n")
                        .required(false)
                        .default_value("0")
                        .value_parser(clap::value_parser!(usize))
                        .help("Maximum number of used wires; candidates with more are skipped (0 = no limit)"),
                )
                .arg(
                    Arg::new("no_L")
                        .long("no_L")
                        .action(clap::ArgAction::SetTrue)
                        .help("Skip circuits that require Rule L during canonicalization"),
                ),
        )
        .subcommand(
            Command::new("rocksdb_2")
                .hide(!cfg!(feature = "legacy-db-tools"))
                .about("Create m sized rocks_db with m1+m2 method")
                .arg(
                    Arg::new("m1")
                        .long("m1")
                        .required(true)
                        .value_parser(clap::value_parser!(usize))
                        .help("Number of gates"),
                )
                .arg(
                    Arg::new("m2")
                        .long("m2")
                        .required(true)
                        .value_parser(clap::value_parser!(usize))
                        .help("Number of gates"),
                )
                .arg(
                    Arg::new("min_n")
                        .long("min_n")
                        .required(false)
                        .default_value("0")
                        .value_parser(clap::value_parser!(usize))
                        .help("Minimum number of used wires in c1; candidates with fewer are skipped"),
                ),
        )
        .subcommand(
            Command::new("combine_rocks")
                .hide(!cfg!(feature = "legacy-db-tools"))
                .about("Combine all rocks_db_m* databases into a single output DB")
                .arg(
                    Arg::new("path")
                        .short('p')
                        .long("path")
                        .required(true)
                        .value_parser(clap::value_parser!(String))
                        .help("Output path for the combined database"),
                ),
        )
        .subcommand(
            Command::new("rocks_to_lmdb")
                .hide(!cfg!(feature = "legacy-db-tools"))
                .about("Convert a RocksDB into an LMDB store")
                .arg(
                    Arg::new("source")
                        .short('s')
                        .long("source")
                        .required(true)
                        .value_parser(clap::value_parser!(String))
                        .help("Path to the source RocksDB"),
                )
                .arg(
                    Arg::new("path")
                        .short('p')
                        .long("path")
                        .required(true)
                        .value_parser(clap::value_parser!(String))
                        .help("Output path for the LMDB store"),
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
        Some(("rocksdb_1", sub)) => commands::rainbow_table::run_rocksdb_1(sub),
        Some(("rocksdb_2", sub)) => commands::rainbow_table::run_rocksdb_2(sub),
        Some(("combine_rocks", sub)) => commands::rainbow_table::run_combine_rocks(sub),
        Some(("rocks_to_lmdb", sub)) => commands::rainbow_table::run_rocks_to_lmdb(sub),
        _ => unreachable!("subcommand_required guarantees a match"),
    }
}
