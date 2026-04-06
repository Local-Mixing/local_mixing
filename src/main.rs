use clap::{Arg, ArgAction, Command};
use itertools::Itertools;
use plotters::prelude::*;
use rocksdb::{DB, Options, BlockBasedOptions, Cache};

use std::{
    fs,
    io::Write,
    path::Path,
};

use local_mixing::{
    circuit::CircuitSeq,
    random::random_data::{build_from_sql, main_random, random_circuit, random_sulking, random_walk_no_skeleton, shoot_random_gate},
    replace::{
        identities::{get_random_wide_identity, random_canonical_id}, main_mix::{
            // main_butterfly, 
            main_butterfly_big, main_interleave_big, 
            // main_mix, 
            main_rac_big, main_rac_big_distance, main_sequential_butterfly, main_shuffle_rcs_big, open_all_dbs, main_shooting_game, main_shuffle_shoot_shuffle
        }, mixing::install_kill_handler, pairs::{GatePair, gate_pair_taxonomy}, replace::{
            compress_big_ancillas,
            sequential_compress_big_ancillas,
            compress_loop,
        }, transpositions::{generate_reversible, insert_wire_shuffles_knuth, insert_wire_shuffles_simple, insert_wire_shuffles_x}
    },
};

fn main() {
    let matches = Command::new("rainbow")
        .about("Rainbow circuit generator")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(
            Command::new("new")
                .about("Build a new database")
                .arg(Arg::new("n").short('n').long("n").required(true).value_parser(clap::value_parser!(usize)))
                .arg(Arg::new("m").short('m').long("m").required(true).value_parser(clap::value_parser!(usize))),
        )
        .subcommand(
            Command::new("load")
                .about("Load an existing database")
                .arg(Arg::new("n").short('n').long("n").required(true).value_parser(clap::value_parser!(usize)))
                .arg(Arg::new("m").short('m').long("m").required(true).value_parser(clap::value_parser!(usize))),
        )
        .subcommand(
            Command::new("explore")
                .about("Explore an existing database")
                .arg(Arg::new("n").short('n').long("n").required(true).value_parser(clap::value_parser!(usize)))
                .arg(Arg::new("m").short('m').long("m").required(true).value_parser(clap::value_parser!(usize))),
        )
        .subcommand(
            Command::new("random")
                .about("Generate random circuits and store in DB")
                .arg(Arg::new("n").short('n').long("n").required(true).value_parser(clap::value_parser!(usize)))
                .arg(Arg::new("m").short('m').long("m").required(true).value_parser(clap::value_parser!(usize)))
                .arg(
                    Arg::new("count")
                        .short('c')
                        .long("count")
                        .value_parser(clap::value_parser!(usize))
                        .conflicts_with("sliding"),
                )
                .arg(
                    Arg::new("sliding")
                        .short('C')
                        .long("sliding")
                        .action(ArgAction::SetTrue)
                        .conflicts_with("count"),
                ),
        )
        .subcommand(
            Command::new("mix")
                .about("Obfuscate and compress an existing circuit")
                .arg(
                    Arg::new("rounds")
                        .short('r')
                        .long("rounds")
                        .required(true)
                        .value_parser(clap::value_parser!(usize))
                ),
        )
        .subcommand(
            Command::new("butterfly")
                .about("Obfuscate and compress an existing circuit via butterfly method")
                .arg(
                    Arg::new("rounds")
                        .short('r')
                        .long("rounds")
                        .required(true)
                        .value_parser(clap::value_parser!(usize))
                ),
        )
        .subcommand(
        Command::new("bbutterfly")
            .about("Obfuscate and compress an existing circuit via butterfly_big method")
            .arg(
                Arg::new("rounds")
                    .short('r')
                    .long("rounds")
                    .required(true)
                    .value_parser(clap::value_parser!(usize)),
            )
            .arg(
                Arg::new("path")
                    .short('p')
                    .long("path")
                    .required(true)
                    .value_parser(clap::value_parser!(String))
                    .help("Path to the circuit file"),
            )
            .arg(
                Arg::new("n")
                    .short('n')
                    .long("n")
                    .required(false)
                    .default_value("32")
                    .value_parser(clap::value_parser!(usize))
                    .help("Number of wires (default: 32)"),
            ),
    )
    .subcommand(
        Command::new("abbutterfly")
            .about("Obfuscate and compress an existing circuit via asymmetric butterfly_big method")
            .arg(
                Arg::new("rounds")
                    .short('r')
                    .long("rounds")
                    .required(true)
                    .value_parser(clap::value_parser!(usize)),
            )
            .arg(
                Arg::new("path")
                    .short('p')
                    .long("path")
                    .required(true)
                    .value_parser(clap::value_parser!(String))
                    .help("Path to the circuit file"),
            )
            .arg(
                Arg::new("n")
                    .short('n')
                    .long("n")
                    .required(false)
                    .default_value("32")
                    .value_parser(clap::value_parser!(usize))
                    .help("Number of wires (default: 32)"),
            )
            .arg(
                Arg::new("bookendless")
                    .short('b')
                    .long("bookendless")
                    .help("Enable bookendless mode")
                    .action(clap::ArgAction::SetTrue),
            ),
    )
    .subcommand(
        Command::new("rcs")
            .about("Obfuscate and compress an existing circuit via replace and compress sequential method")
            .arg(
                Arg::new("rounds")
                    .short('r')
                    .long("rounds")
                    .required(true)
                    .value_parser(clap::value_parser!(usize)),
            )
            .arg(
                Arg::new("source")
                    .short('s')
                    .long("source")
                    .required(true)
                    .value_parser(clap::value_parser!(String))
                    .help("Path to the input circuit file"),
            )
            .arg(
                Arg::new("destination")
                    .short('d')
                    .long("destination")
                    .required(true)
                    .value_parser(clap::value_parser!(String))
                    .help("Path to the output circuit file"),
            )
            .arg(
                Arg::new("n")
                    .short('n')
                    .long("n")
                    .required(false)
                    .default_value("32")
                    .value_parser(clap::value_parser!(usize))
                    .help("Number of wires (default: 32)"),
            )
            .arg(
                Arg::new("id_len")
                    .long("id_len")
                    .required(true)
                    .value_parser(clap::value_parser!(usize))
                    .help("ID length"),
            )
            .arg(
                Arg::new("tower")
                    .short('t')
                    .long("tower")
                    .help("Use tower identities over singles")
                    .required(false)
                    .action(clap::ArgAction::SetTrue),
            )
            .arg(
                Arg::new("intermediate")
                    .short('i')
                    .long("intermediate")
                    .required(true)
                    .value_parser(clap::value_parser!(String))
                    .help("Path to the intermediate circuit file"),
            ),
    )
    .subcommand(
        Command::new("srcs")
            .about("Obfuscate and compress an existing circuit via replace and compress sequential method with wire shuffling")
            .arg(
                Arg::new("rounds")
                    .short('r')
                    .long("rounds")
                    .required(true)
                    .value_parser(clap::value_parser!(usize)),
            )
            .arg(
                Arg::new("source")
                    .short('s')
                    .long("source")
                    .required(true)
                    .value_parser(clap::value_parser!(String))
                    .help("Path to the input circuit file"),
            )
            .arg(
                Arg::new("destination")
                    .short('d')
                    .long("destination")
                    .required(true)
                    .value_parser(clap::value_parser!(String))
                    .help("Path to the output circuit file"),
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
                Arg::new("id_len")
                    .long("id_len")
                    .required(true)
                    .value_parser(clap::value_parser!(usize))
                    .help("ID length"),
            )
            .arg(
                Arg::new("tower")
                    .short('t')
                    .long("tower")
                    .help("Use tower identities over singles")
                    .required(false)
                    .action(clap::ArgAction::SetTrue),
            )
            .arg(
                Arg::new("intermediate")
                    .short('i')
                    .long("intermediate")
                    .required(true)
                    .value_parser(clap::value_parser!(String))
                    .help("Path to the intermediate circuit file"),
            )
            .arg(
                Arg::new("x")
                    .short('x')
                    .long("x")
                    .required(false)
                    .value_parser(clap::value_parser!(usize))
                    .help("Number of shuffles"),
            ),
    )
    .subcommand(
        Command::new("interleave")
            .about("Obfuscate and compress an existing circuit via replace and compress sequential method")
            .arg(
                Arg::new("rounds")
                    .short('r')
                    .long("rounds")
                    .required(true)
                    .value_parser(clap::value_parser!(usize)),
            )
            .arg(
                Arg::new("source")
                    .short('s')
                    .long("source")
                    .required(true)
                    .value_parser(clap::value_parser!(String))
                    .help("Path to the input circuit file"),
            )
            .arg(
                Arg::new("destination")
                    .short('d')
                    .long("destination")
                    .required(true)
                    .value_parser(clap::value_parser!(String))
                    .help("Path to the output circuit file"),
            )
            .arg(
                Arg::new("n")
                    .short('n')
                    .long("n")
                    .required(false)
                    .default_value("32")
                    .value_parser(clap::value_parser!(usize))
                    .help("Number of wires (default: 32)"),
            )
            .arg(
                Arg::new("tower")
                    .short('t')
                    .long("tower")
                    .help("Use tower identities over singles")
                    .required(false)
                    .action(clap::ArgAction::SetTrue),
            )
            .arg(
                Arg::new("id_len")
                    .long("id_len")
                    .required(true)
                    .value_parser(clap::value_parser!(usize))
                    .help("ID length"),
            )
            .arg(
                Arg::new("intermediate")
                    .short('i')
                    .long("intermediate")
                    .required(true)
                    .value_parser(clap::value_parser!(String))
                    .help("Path to the intermediate circuit file"),
            ),
    )
    .subcommand(
        Command::new("rcd")
            .about("Obfuscate and compress an existing circuit via replace and compress distance method")
            .arg(
                Arg::new("rounds")
                    .short('r')
                    .long("rounds")
                    .required(true)
                    .value_parser(clap::value_parser!(usize)),
            )
            .arg(
                Arg::new("source")
                    .short('s')
                    .long("source")
                    .required(true)
                    .value_parser(clap::value_parser!(String))
                    .help("Path to the input circuit file"),
            )
            .arg(
                Arg::new("destination")
                    .short('d')
                    .long("destination")
                    .required(true)
                    .value_parser(clap::value_parser!(String))
                    .help("Path to the output circuit file"),
            )
            .arg(
                Arg::new("n")
                    .short('n')
                    .long("n")
                    .required(false)
                    .default_value("32")
                    .value_parser(clap::value_parser!(usize))
                    .help("Number of wires (default: 32)"),
            )
            .arg(
                Arg::new("id_len")
                    .long("id_len")
                    .required(true)
                    .value_parser(clap::value_parser!(usize))
                    .help("ID length"),
            )
            .arg(
                Arg::new("intermediate")
                    .short('i')
                    .long("intermediate")
                    .required(true)
                    .value_parser(clap::value_parser!(String))
                    .help("Path to the intermediate circuit file"),
            )
            .arg(
                Arg::new("m")
                    .short('m')
                    .long("min")
                    .required(false)
                    .default_value("30")
                    .value_parser(clap::value_parser!(usize))
                    .help("Minimum distance"),
            )
            .arg(
                Arg::new("tower")
                    .short('t')
                    .long("tower")
                    .help("Use tower identities over singles")
                    .required(false)
                    .action(clap::ArgAction::SetTrue),
            ),
    )
    .subcommand(
        Command::new("heatmap")
            .about("Run the circuit distinguisher and produce a heatmap")
            .arg(
                Arg::new("inputs")
                    .short('i')
                    .long("inputs")
                    .required(true)
                    .value_parser(clap::value_parser!(usize))
                    .help("Number of random inputs to test"),
            )
            .arg(
                Arg::new("num_wires")
                    .short('n')
                    .long("num_wires")
                    .required(true)
                    .value_parser(clap::value_parser!(usize)),
            )
            .arg(
                Arg::new("xlabel")
                    .short('x')
                    .long("xlabel")
                    .value_parser(clap::value_parser!(String))
                    .help("Label for X axis"),
            )
            .arg(
                Arg::new("ylabel")
                    .short('y')
                    .long("ylabel")
                    .value_parser(clap::value_parser!(String))
                    .help("Label for Y axis"),
            )
            .arg(
                Arg::new("std")
                    .short('s')
                    .help("Use standard deviation (if given) or raw otherwise")
                    .action(ArgAction::SetTrue)
            ),
    )
    .subcommand(
        Command::new("reverse")
            .about("Reverse the order of gates in a circuit file")
            .arg(
                Arg::new("source")
                    .short('s')
                    .long("source")
                    .required(true)
                    .value_parser(clap::value_parser!(String))
                    .help("Path to the source circuit file"),
            )
            .arg(
                Arg::new("dest")
                    .short('d')
                    .long("dest")
                    .required(true)
                    .value_parser(clap::value_parser!(String))
                    .help("Path to write the reversed circuit file"),
            ),
    )
    .subcommand(
        Command::new("gen_reversible")
            .about("Generate reversible circuit")
            .arg(
                Arg::new("source")
                    .short('s')
                    .long("source")
                    .required(true)
                    .value_parser(clap::value_parser!(String))
                    .help("Path to the source circuit file"),
            )
            .arg(
                Arg::new("dest")
                    .short('d')
                    .long("dest")
                    .required(true)
                    .value_parser(clap::value_parser!(String))
                    .help("Path to write the reversed circuit file"),
            )
            .arg(
                Arg::new("n")
                    .short('n')
                    .long("n")
                    .required(true)
                    .value_parser(clap::value_parser!(usize))
                    .help("Number of wires in the circuit"),
            )
    )
    .subcommand(
        Command::new("binload")
            .about("Load a binary circuit file")
            .arg(
                Arg::new("n")
                    .short('n')
                    .long("n")
                    .required(true)
                    .value_parser(clap::value_parser!(usize))
                    .help("Number of wires in the circuit"),
            )
            .arg(
                Arg::new("m")
                    .short('m')
                    .long("m")
                    .required(true)
                    .value_parser(clap::value_parser!(usize))
                    .help("Number of gates in the circuit"),
            )
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
                    .short('s')
                    .long("seq")
                    .help("Enable seq mode")
                    .required(false)
                    .action(clap::ArgAction::SetTrue),
            ),
    )
    .subcommand(
        Command::new("wiredot")
            .about("Run the circuit counter and produce a dotplot")
            .arg(
                Arg::new("num_wires")
                    .short('n')
                    .long("num_wires")
                    .required(true)
                    .value_parser(clap::value_parser!(usize)),
            )
            .arg(
                Arg::new("xlabel")
                    .short('x')
                    .long("xlabel")
                    .value_parser(clap::value_parser!(String))
                    .help("Label for X axis"),
            )
            .arg(
                Arg::new("path")
                    .short('p')
                    .long("path")
                    .value_parser(clap::value_parser!(String))
                    .help("Circuit to analyze path"),
            ),
            
    )
    .subcommand(
        Command::new("lmdb")
            .about("Explore an existing database")
            .arg(Arg::new("n").short('n').long("n").required(true).value_parser(clap::value_parser!(usize)))
            .arg(Arg::new("m").short('m').long("m").required(true).value_parser(clap::value_parser!(usize))),
    )
    .subcommand(
        Command::new("lmdbp")
            .about("Explore an existing database")
            .arg(Arg::new("n").short('n').long("n").required(true).value_parser(clap::value_parser!(usize)))
            .arg(Arg::new("m").short('m').long("m").required(true).value_parser(clap::value_parser!(usize))),
    )
    .subcommand(
        Command::new("lmdbcounts")
        .about("Generate table for generating canon ids")
    )
    .subcommand(
        Command::new("lmdbid")
        .about("Generate table for generating canon ids")
        .arg(Arg::new("n").short('n').long("n").required(true).value_parser(clap::value_parser!(usize)))
    )
    .subcommand(
        Command::new("lmdbnid")
        .about("Generate table for generating canon ids for n wires")
        .arg(Arg::new("n").short('n').long("n").required(true).value_parser(clap::value_parser!(usize)))
    )
    .subcommand(
        Command::new("string")
            .about("Reverse the order of gates in a circuit file")
            .arg(
                Arg::new("source")
                    .short('s')
                    .long("source")
                    .required(true)
                    .value_parser(clap::value_parser!(String))
                    .help("Path to the source circuit file"),
            )
            .arg(
                Arg::new("dest")
                    .short('d')
                    .long("dest")
                    .required(true)
                    .value_parser(clap::value_parser!(String))
                    .help("Path to write the reversed circuit file"),
            ),
    )
    .subcommand(
        Command::new("degree")
            .about("Compute an upper bound on the algebraic degree of each wire")
            .arg(
                Arg::new("source")
                    .short('s')
                    .long("source")
                    .required(true)
                    .value_parser(clap::value_parser!(String))
                    .help("Path to the source circuit file"),
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
                Arg::new("start")
                    .long("start")
                    .required(true)
                    .value_parser(clap::value_parser!(usize))
                    .help("Starting index"),
            )
            .arg(
                Arg::new("end")
                    .long("end")
                    .required(true)
                    .value_parser(clap::value_parser!(usize))
                    .help("Ending index"),
            )
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
                    .action(clap::ArgAction::SetTrue)
            )
    )
    .subcommand(
        Command::new("seq_butterfly")
            .about("Do sequential butterfly on a circuit")
            .arg(Arg::new("n").short('n').long("n").required(true).value_parser(clap::value_parser!(usize)))
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
                    .help("Number of rounds")
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
                Arg::new("id_len")
                    .long("id_len")
                    .required(true)
                    .value_parser(clap::value_parser!(usize))
                    .help("Number of wires per identity")
            )
            .arg(
                Arg::new("rev_left")
                    .long("rev_left")
                    .help("Use reverse order for shoot+collide left on R")
                    .required(false) 
                    .action(clap::ArgAction::SetTrue)
            )
            .arg(
                Arg::new("for_right")
                    .long("for_right")
                    .help("Use forward order for shoot+collide right on R*")
                    .required(false) 
                    .action(clap::ArgAction::SetTrue)
            )
            .arg(
                Arg::new("tower_left")
                    .long("tower_left")
                    .help("Use tower identities for collide left on R")
                    .required(false) 
                    .action(clap::ArgAction::SetTrue)
            )
            .arg(
                Arg::new("tower_right")
                    .long("tower_right")
                    .help("Use tower identities for collide right on R*")
                    .required(false) 
                    .action(clap::ArgAction::SetTrue)
            )
            .arg(
                Arg::new("add_rounds_left")
                    .long("add_rounds_left")
                    .required(false)
                    .default_value("0")
                    .value_parser(clap::value_parser!(u8))
                    .help("Add shoot+collision rounds for R*. This is currently unsupported"),
            )
            .arg(
                Arg::new("add_rounds_right")
                    .long("add_rounds_right")
                    .required(false)
                    .default_value("0")
                    .value_parser(clap::value_parser!(u8))
                    .help("Add shoot+collision rounds for R*. This is currently unsupported"),
            )
    )
    .subcommand(
        Command::new("ssg")
            .about("Do simple shooting game on a circuit")
            .arg(Arg::new("n").short('n').long("n").required(true).value_parser(clap::value_parser!(usize)))
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
                    .help("Number of rounds")
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
                Arg::new("id_len")
                    .long("id_len")
                    .required(true)
                    .value_parser(clap::value_parser!(usize))
                    .help("Number of wires per identity")
            )
            .arg(
                Arg::new("tower")
                    .long("tower")
                    .help("Use tower identities")
                    .required(false) 
                    .action(clap::ArgAction::SetTrue)
            )
            .arg(
                Arg::new("stop")
                    .long("stop")
                    .required(true)
                    .value_parser(clap::value_parser!(usize))
                    .help("When to stop the game")
            )
            .arg(
                Arg::new("intermediate")
                    .short('i')
                    .long("intermediate")
                    .required(true)
                    .value_parser(clap::value_parser!(String))
                    .help("Path to the intermediate circuit file"),
            ),
    )
    .subcommand(
        Command::new("sss")
            .about("Do simple shooting game on a circuit")
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
                    .help("Number of rounds")
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
                Arg::new("id_len")
                    .long("id_len")
                    .required(true)
                    .value_parser(clap::value_parser!(usize))
                    .help("Number of wires per identity")
            )
            .arg(
                Arg::new("tower")
                    .long("tower")
                    .help("Use tower identities")
                    .required(false) 
                    .action(clap::ArgAction::SetTrue)
            )
            .arg(
                Arg::new("interleave")
                    .long("interleave")
                    .help("Use interleaving")
                    .required(false) 
                    .action(clap::ArgAction::SetTrue)
            )
            .arg(
                Arg::new("stop")
                    .long("stop")
                    .required(true)
                    .value_parser(clap::value_parser!(usize))
                    .help("When to stop the game")
            )
            .arg(
                Arg::new("intermediate")
                    .short('i')
                    .long("intermediate")
                    .required(true)
                    .value_parser(clap::value_parser!(String))
                    .help("Path to the intermediate circuit file"),
            ),
    )
    .subcommand(
        Command::new("shoot")
            .about("Shuffle a circuit")
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
            )
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
            )
    )
    .get_matches();

    match matches.subcommand() {
        // Some(("load", sub)) => {
        //     let n: usize = *sub.get_one("n").unwrap();
        //     let m: usize = *sub.get_one("m").unwrap();
        //     // main_rainbow_load(n, m, "./db");
            
        //     // Open DB connection
        //     let config = Config::default().access_mode(AccessMode::ReadOnly).unwrap();
        //     let conn = Connection::open_with_flags("circuits.duckdb", config).unwrap();
        //     let perms: Vec<Vec<usize>> = (0..n).permutations(n).collect();
        //     let bit_shuf = perms.into_iter().skip(1).collect::<Vec<_>>();
        //     build_from_sql(&conn, n,m, &bit_shuf).expect("Unknown error occured");
        // }
        Some(("random", sub)) => {
            let n: usize = *sub.get_one("n").unwrap();
            let m: usize = *sub.get_one("m").unwrap();

            if let Some(count) = sub.get_one::<usize>("count") {
                // Fixed-count mode
                main_random(n, m, *count, false);
            } else if sub.get_flag("sliding") {
                // Sliding-window fail-rate mode
                main_random(n, m, 0, true);
            } else {
                panic!("You must provide either -c <count> or -C for sliding-window mode");
            }
        }
        // Some(("mix", sub)) => {
        //     let rounds: usize = *sub.get_one("rounds").unwrap();

        //     let data = fs::read_to_string("initial.txt").expect("Failed to read initial.txt");
        //     // let seed = OsRng.try_next_u64().unwrap_or_else(|e| {
        //     //     panic!("Failed to generate random seed: {}", e);
        //     // });
        //     // println!("Using seed: {}", seed);
        //     if data.trim().is_empty() {
        //         // Open DB connection
        //         let config = Config::default().access_mode(AccessMode::ReadOnly).unwrap();
        //         let conn = Connection::open_with_flags("circuits.duckdb", config).unwrap();
        //         let lmdb = "./db";
        //         let env = Environment::new()
        //         .set_max_dbs(266)      
        //         .set_map_size(700 * 1024 * 1024 * 1024) 
        //         .open(Path::new(lmdb))
        //         .expect("Failed to open lmdb");

        //         // Fallback when file is empty
        //         let c1= random_canonical_id(&env, &conn, 5).unwrap();
        //         println!("{:?} Starting Len: {}", c1.permutation(5).data, c1.gates.len());
        //         main_mix(&c1, rounds, &conn, 5);
        //     } else {
                
        //         let c = CircuitSeq::from_string(&data);

        //         // Open DB connection
        //         let config = Config::default().access_mode(AccessMode::ReadOnly).unwrap();
        //         let conn = Connection::open_with_flags("circuits.duckdb", config).unwrap();
        //         main_mix(&c, rounds, &conn, 5);
        //     }
        // }
        // Some(("butterfly", sub)) => {
        //     let rounds: usize = *sub.get_one("rounds").unwrap();
        //     let data = fs::read_to_string("initial.txt").expect("Failed to read initial.txt");
        //     // let seed = OsRng.try_next_u64().unwrap_or_else(|e| {
        //     //     panic!("Failed to generate random seed: {}", e);
        //     // });
        //     // println!("Using seed: {}", seed);
        //     if data.trim().is_empty() {
        //         // Open DB connection
        //         let config = Config::default().access_mode(AccessMode::ReadOnly).unwrap();
        //         let conn = Connection::open_with_flags("circuits.duckdb", config).unwrap();
        //         // Fallback when file is empty
        //         println!("Generating random");
        //         let c1= random_circuit(6,30);
        //         // let perms: Vec<Vec<usize>> = (0..5).permutations(5).collect();
        //         // let bit_shuf = perms.into_iter().skip(1).collect::<Vec<_>>();
        //         // let c1 = compress(&random_circuit(5,128), 100_000, &conn, &bit_shuf,5 );
        //         println!("{:?} Starting Len: {}", c1.permutation(6).data, c1.gates.len());
        //         main_butterfly(&c1, rounds, &conn, 6);
        //     } else {
                
        //         let c = CircuitSeq::from_string(&data);

        //         // Open DB connection
        //         let config = Config::default().access_mode(AccessMode::ReadOnly).unwrap();
        //         let conn = Connection::open_with_flags("circuits.duckdb", config).unwrap();
        //         main_butterfly(&c, rounds, &conn, 6);
        //     }
        // }
        Some(("bbutterfly", sub)) => {
            let rounds: usize = *sub.get_one("rounds").unwrap();
            let path: &str = sub.get_one::<String>("path").unwrap().as_str();
            let n: usize = *sub.get_one("n").unwrap_or(&32); // default to 32 if not provided
            let data = fs::read_to_string("initial.txt").expect("Failed to read initial.txt");

            let lmdb = "./db";
            let _ = std::fs::create_dir_all(lmdb);

            let env = Environment::new()
                .set_max_dbs(266)      
                .set_map_size(700 * 1024 * 1024 * 1024) 
                .open(Path::new(lmdb))
                .expect("Failed to open lmdb");

            let db_n6m5 = DB::open_for_read_only(&Options::default(), "rocksdb_n6m5perms", false)
                .expect("Failed to open RocksDB n6m5");
            let db_n7m4 = DB::open_for_read_only(&Options::default(), "rocksdb_n7m4perms", false)
                .expect("Failed to open RocksDB n7m4");

            if data.trim().is_empty() {
                println!("Generating random");
                let c1 = random_circuit(n, 30);
                println!("Starting Len: {}", c1.gates.len());
                main_butterfly_big(&c1, rounds, &db_n6m5, &db_n7m4, n, false, path, &env);
            } else {
                let c = CircuitSeq::from_string(&data);
                main_butterfly_big(&c, rounds, &db_n6m5, &db_n7m4, n, false, path, &env);
            }
        }

        Some(("abbutterfly", sub)) => {
            let rounds: usize = *sub.get_one("rounds").unwrap();
            let path: &str = sub.get_one::<String>("path").unwrap().as_str();
            let n: usize = *sub.get_one("n").unwrap_or(&32); // default to 32 if not provided
            let data = fs::read_to_string("initial.txt").expect("Failed to read initial.txt");
            let bookendless = sub.get_flag("bookendless"); 

            let lmdb = "./db";
            let _ = std::fs::create_dir_all(lmdb);

            let env = Environment::new()
                .set_max_readers(10000) 
                .set_max_dbs(266)      
                .set_map_size(800 * 1024 * 1024 * 1024) 
                .open(Path::new(lmdb))
                .expect("Failed to open lmdb");

            let db_n6m5 = DB::open_for_read_only(&Options::default(), "rocksdb_n6m5perms", false)
                .expect("Failed to open RocksDB n6m5");
            let db_n7m4 = DB::open_for_read_only(&Options::default(), "rocksdb_n7m4perms", false)
                .expect("Failed to open RocksDB n7m4");

            install_kill_handler();
            if data.trim().is_empty() {
                println!("Generating random");
                let c1 = random_circuit(n, 30);
                println!("Starting Len: {}", c1.gates.len());
                if bookendless {
                    // main_butterfly_big_bookendsless(&c1, rounds, &db_n6m5, &db_n7m4, n, true, path, &env);
                } else {
                    main_butterfly_big(&c1, rounds, &db_n6m5, &db_n7m4, n, true, path, &env);
                }
            } else {
                let c = CircuitSeq::from_string(&data);
                if bookendless {
                    // main_butterfly_big_bookendsless(&c, rounds, &db_n6m5, &db_n7m4, n, true, path, &env);
                } else {
                    main_butterfly_big(&c, rounds, &db_n6m5, &db_n7m4, n, true, path, &env);
                }
            }
        }
        Some(("rcs", sub)) => {
            let rounds: usize = *sub.get_one("rounds").unwrap();
            let s: &str = sub.get_one::<String>("source").unwrap().as_str();
            let i: &str = sub.get_one::<String>("intermediate").unwrap().as_str();
            let d: &str = sub.get_one::<String>("destination").unwrap().as_str();
            let tower = sub.get_flag("tower");
            let n: usize = *sub.get_one("n").unwrap_or(&32); // default to 32 if not provided
            let id_len: usize = *sub.get_one("id_len").unwrap(); 
            let data = fs::read_to_string(s).expect("Failed to read initial.txt");

            let lmdb = "./db";
            let _ = std::fs::create_dir_all(lmdb);

            let env = Environment::new()
                .set_max_readers(10000) 
                .set_max_dbs(266)      
                .set_map_size(800 * 1024 * 1024 * 1024) 
                .open(Path::new(lmdb))
                .expect("Failed to open lmdb");

            let db_n6m5 = DB::open_for_read_only(&Options::default(), "rocksdb_n6m5perms", false)
                .expect("Failed to open RocksDB n6m5");
            let db_n7m4 = DB::open_for_read_only(&Options::default(), "rocksdb_n7m4perms", false)
                .expect("Failed to open RocksDB n7m4");

            install_kill_handler();
            if data.trim().is_empty() {
                println!("Empty file");
            } else {
                let c = CircuitSeq::from_string(&data);
                main_rac_big(&c, rounds, &db_n6m5, &db_n7m4, n, d, &env, i, tower, id_len);
                let x_label = {
                    let stem = std::path::Path::new(s).file_stem().unwrap().to_str().unwrap();
                    let num = stem.strip_prefix("circuit").unwrap_or(stem);
                    format!("Circuit {}", num)
                };

                let y_label = {
                    let stem = std::path::Path::new(d).file_stem().unwrap().to_str().unwrap();
                    let num = stem.strip_prefix("circuit").unwrap_or(stem);
                    format!("Circuit {}", num)
                };
                let path_s = std::path::Path::new(s).file_stem().unwrap().to_str().unwrap();
                let path_d = std::path::Path::new(d).file_stem().unwrap().to_str().unwrap();
                println!(
                    "For generating heatmaps:\n\
                    python3 ./heatmap/heatmap.py \
                    --n {} \
                    --i 100 \
                    --x \"{}\" \
                    --y \"{}\" \
                    --c1 \"{}\" \
                    --c2 \"{}\" \
                    --path ./{}{}.png",
                        n, x_label, y_label, s, d, path_s, path_d
                );
            }
        }
        Some(("srcs", sub)) => {
            let rounds: usize = *sub.get_one("rounds").unwrap();
            let s: &str = sub.get_one::<String>("source").unwrap().as_str();
            let i: &str = sub.get_one::<String>("intermediate").unwrap().as_str();
            let d: &str = sub.get_one::<String>("destination").unwrap().as_str();
            let tower = sub.get_flag("tower");
            let n: usize = *sub.get_one("n").unwrap();
            let x: usize = *sub.get_one("x").unwrap_or(&0);
            let id_len: usize = *sub.get_one("id_len").unwrap();
            let data = fs::read_to_string(s).expect("Failed to read initial.txt");

            let lmdb = "./db";
            let _ = std::fs::create_dir_all(lmdb);

            let env = Environment::new()
                .set_max_readers(10000) 
                .set_max_dbs(266)      
                .set_map_size(800 * 1024 * 1024 * 1024) 
                .open(Path::new(lmdb))
                .expect("Failed to open lmdb");

            let db_n6m5 = DB::open_for_read_only(&Options::default(), "rocksdb_n6m5perms", false)
                .expect("Failed to open RocksDB n6m5");
            let db_n7m4 = DB::open_for_read_only(&Options::default(), "rocksdb_n7m4perms", false)
                .expect("Failed to open RocksDB n7m4");

            install_kill_handler();
            if data.trim().is_empty() {
                println!("Empty file");
            } else {
                let c = CircuitSeq::from_string(&data);
                main_shuffle_rcs_big(&c, rounds, &db_n6m5, &db_n7m4, n, d, &env, i, tower, x, id_len);
                let x_label = {
                    let stem = std::path::Path::new(s).file_stem().unwrap().to_str().unwrap();
                    let num = stem.strip_prefix("circuit").unwrap_or(stem);
                    format!("Circuit {}", num)
                };

                let y_label = {
                    let stem = std::path::Path::new(d).file_stem().unwrap().to_str().unwrap();
                    let num = stem.strip_prefix("circuit").unwrap_or(stem);
                    format!("Circuit {}", num)
                };
                let path_s = std::path::Path::new(s).file_stem().unwrap().to_str().unwrap();
                let path_d = std::path::Path::new(d).file_stem().unwrap().to_str().unwrap();
                println!(
                    "For generating heatmaps:\n\
                    python3 ./heatmap/heatmap.py \
                    --n {} \
                    --i 100 \
                    --x \"{}\" \
                    --y \"{}\" \
                    --c1 \"{}\" \
                    --c2 \"{}\" \
                    --path ./{}{}.png",
                        n, x_label, y_label, s, d, path_s, path_d
                );
            }
        }
        Some(("interleave", sub)) => {
            let rounds: usize = *sub.get_one("rounds").unwrap();
            let s: &str = sub.get_one::<String>("source").unwrap().as_str();
            let i: &str = sub.get_one::<String>("intermediate").unwrap().as_str();
            let d: &str = sub.get_one::<String>("destination").unwrap().as_str();
            let tower = sub.get_flag("tower");
            let n: usize = *sub.get_one("n").unwrap_or(&32); // default to 32 if not provided
            let id_len: usize = *sub.get_one("id_len").unwrap();
            let data = fs::read_to_string(s).expect("Failed to read initial.txt");

            let lmdb = "./db";
            let _ = std::fs::create_dir_all(lmdb);

            let env = Environment::new()
                .set_max_readers(10000) 
                .set_max_dbs(266)      
                .set_map_size(800 * 1024 * 1024 * 1024) 
                .open(Path::new(lmdb))
                .expect("Failed to open lmdb");

            let db_n6m5 = DB::open_for_read_only(&Options::default(), "rocksdb_n6m5perms", false)
                .expect("Failed to open RocksDB n6m5");
            let db_n7m4 = DB::open_for_read_only(&Options::default(), "rocksdb_n7m4perms", false)
                .expect("Failed to open RocksDB n7m4");

            install_kill_handler();
            if data.trim().is_empty() {
                println!("Empty file");
            } else {
                let c = CircuitSeq::from_string(&data);
                main_interleave_big(&c, rounds, &db_n6m5, &db_n7m4, n, d, &env, i, tower, id_len);
                let x_label = {
                    let stem = std::path::Path::new(s).file_stem().unwrap().to_str().unwrap();
                    let num = stem.strip_prefix("circuit").unwrap_or(stem);
                    format!("Circuit {}", num)
                };

                let y_label = {
                    let stem = std::path::Path::new(d).file_stem().unwrap().to_str().unwrap();
                    let num = stem.strip_prefix("circuit").unwrap_or(stem);
                    format!("Circuit {}", num)
                };
                let path_s = std::path::Path::new(s).file_stem().unwrap().to_str().unwrap();
                let path_d = std::path::Path::new(d).file_stem().unwrap().to_str().unwrap();
                println!(
                    "For generating heatmaps:\n\
                    python3 ./heatmap/heatmap.py \
                    --n {} \
                    --i 100 \
                    --x \"{}\" \
                    --y \"{}\" \
                    --c1 \"{}\" \
                    --c2 \"{}\" \
                    --path ./{}{}.png",
                        n, x_label, y_label, s, d, path_s, path_d
                );
            }
        }
        Some(("rcd", sub)) => {
            let rounds: usize = *sub.get_one("rounds").unwrap();
            let s: &str = sub.get_one::<String>("source").unwrap().as_str();
            let i: &str = sub.get_one::<String>("intermediate").unwrap().as_str();
            let d: &str = sub.get_one::<String>("destination").unwrap().as_str();
            let n: usize = *sub.get_one("n").unwrap_or(&32); // default to 32 if not provided
            let m: usize = *sub.get_one("m").unwrap_or(&30); // default to 30f not provided
            let id_len: usize = *sub.get_one("id_len").unwrap();
            let tower = sub.get_flag("tower");
            let data = fs::read_to_string(s).expect("Failed to read initial.txt");

            let lmdb = "./db";
            let _ = std::fs::create_dir_all(lmdb);

            let env = Environment::new()
                .set_max_readers(10000) 
                .set_max_dbs(266)      
                .set_map_size(800 * 1024 * 1024 * 1024) 
                .open(Path::new(lmdb))
                .expect("Failed to open lmdb");

            let db_n6m5 = DB::open_for_read_only(&Options::default(), "rocksdb_n6m5perms", false)
                .expect("Failed to open RocksDB n6m5");
            let db_n7m4 = DB::open_for_read_only(&Options::default(), "rocksdb_n7m4perms", false)
                .expect("Failed to open RocksDB n7m4");

            install_kill_handler();
            if data.trim().is_empty() {
                println!("Empty file");
            } else {
                let c = CircuitSeq::from_string(&data);
                main_rac_big_distance(&c, rounds, &db_n6m5, &db_n7m4, n, d, &env, i, m, tower, id_len);
                let x_label = {
                    let stem = std::path::Path::new(s).file_stem().unwrap().to_str().unwrap();
                    let num = stem.strip_prefix("circuit").unwrap_or(stem);
                    format!("Circuit {}", num)
                };

                let y_label = {
                    let stem = std::path::Path::new(d).file_stem().unwrap().to_str().unwrap();
                    let num = stem.strip_prefix("circuit").unwrap_or(stem);
                    format!("Circuit {}", num)
                };
                let path_s = std::path::Path::new(s).file_stem().unwrap().to_str().unwrap();
                let path_d = std::path::Path::new(d).file_stem().unwrap().to_str().unwrap();
                println!(
                    "For generating heatmaps:\n\
                    python3 ./heatmap/heatmap.py \
                    --n {} \
                    --i 100 \
                    --x \"{}\" \
                    --y \"{}\" \
                    --c1 \"{}\" \
                    --c2 \"{}\" \
                    --path ./{}{}.png",
                        n, x_label, y_label, s, d, path_s, path_d
                );
            }
        }
        Some(("seq_butterfly", sub)) => {
            let rounds: usize = *sub.get_one("rounds").unwrap();
            let s: &str = sub.get_one::<String>("source").unwrap().as_str();
            let d: &str = sub.get_one::<String>("destination").unwrap().as_str();
            let n: usize = *sub.get_one("n").unwrap();
            let id_len: usize = *sub.get_one("id_len").unwrap();
            let tower_left = sub.get_flag("tower_left");
            let tower_right = sub.get_flag("tower_right");
            let more_left = *sub.get_one("add_rounds_left").unwrap_or(&0);
            let more_right = *sub.get_one("add_rounds_right").unwrap_or(&0);
            let rev_left = sub.get_flag("rev_left");
            let for_right = sub.get_flag("for_right");
            let data = fs::read_to_string(s).expect("Failed to read initial.txt");

            let lmdb = "./db";
            let _ = std::fs::create_dir_all(lmdb);

            let env = Environment::new()
                .set_max_readers(10000) 
                .set_max_dbs(266)      
                .set_map_size(800 * 1024 * 1024 * 1024) 
                .open(Path::new(lmdb))
                .expect("Failed to open lmdb");

            let db_n6m5 = DB::open_for_read_only(&Options::default(), "rocksdb_n6m5perms", false)
                .expect("Failed to open RocksDB n6m5");
            let db_n7m4 = DB::open_for_read_only(&Options::default(), "rocksdb_n7m4perms", false)
                .expect("Failed to open RocksDB n7m4");

            install_kill_handler();
            if data.trim().is_empty() {
                println!("Empty file");
            } else {
                let c = CircuitSeq::from_string(&data);
                main_sequential_butterfly(
                    &c, 
                    rounds, 
                    &db_n6m5,
                    &db_n7m4,
                    n, 
                    d, 
                    &env, 
                    id_len, 
                    rev_left, 
                    tower_left, 
                    more_left,
                    !for_right,
                    tower_right,
                    more_right,
                );
                let x_label = {
                    let stem = std::path::Path::new(s).file_stem().unwrap().to_str().unwrap();
                    let num = stem.strip_prefix("circuit").unwrap_or(stem);
                    format!("Circuit {}", num)
                };

                let y_label = {
                    let stem = std::path::Path::new(d).file_stem().unwrap().to_str().unwrap();
                    let num = stem.strip_prefix("circuit").unwrap_or(stem);
                    format!("Circuit {}", num)
                };
                let path_s = std::path::Path::new(s).file_stem().unwrap().to_str().unwrap();
                let path_d = std::path::Path::new(d).file_stem().unwrap().to_str().unwrap();
                println!(
                    "For generating heatmaps:\n\
                    python3 ./heatmap/heatmap.py \
                    --n {} \
                    --i 100 \
                    --x \"{}\" \
                    --y \"{}\" \
                    --c1 \"{}\" \
                    --c2 \"{}\" \
                    --path ./{}{}.png",
                        n, x_label, y_label, s, d, path_s, path_d
                );
            }
        }
        Some(("ssg", sub)) => {
            let rounds: usize = *sub.get_one("rounds").unwrap();
            let s: &str = sub.get_one::<String>("source").unwrap().as_str();
            let d: &str = sub.get_one::<String>("destination").unwrap().as_str();
            let n: usize = *sub.get_one("n").unwrap();
            let id_len: usize = *sub.get_one("id_len").unwrap();
            let tower = sub.get_flag("tower");
            let stop: usize = *sub.get_one("stop").unwrap();
            let i: &str = sub.get_one::<String>("intermediate").unwrap().as_str();
            let data = fs::read_to_string(s).expect("Failed to read initial.txt");

            let lmdb = "./db";
            let _ = std::fs::create_dir_all(lmdb);

            let env = Environment::new()
                .set_max_readers(10000) 
                .set_max_dbs(266)      
                .set_map_size(800 * 1024 * 1024 * 1024) 
                .open(Path::new(lmdb))
                .expect("Failed to open lmdb");

            let cache = Cache::new_lru_cache(25 * 1024 * 1024 * 1024); 

            let mut block_opts = BlockBasedOptions::default();
            block_opts.set_block_cache(&cache);
            block_opts.set_bloom_filter(10.0, false);
            block_opts.set_cache_index_and_filter_blocks(true);
            block_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);

            let mut opts = Options::default();
            opts.set_block_based_table_factory(&block_opts);
            opts.set_compression_type(rocksdb::DBCompressionType::None);

            let db_n6m5 = DB::open_for_read_only(&opts, "rocksdb_n6m5perms", false)
                .expect("Failed to open RocksDB n6m5");
            let db_n7m4 = DB::open_for_read_only(&opts, "rocksdb_n7m4perms", false)
                .expect("Failed to open RocksDB n7m4");

            install_kill_handler();
            if data.trim().is_empty() {
                println!("Empty file");
            } else {
                let c = CircuitSeq::from_string(&data);
                main_shooting_game(
                    &c, 
                    rounds, 
                    &db_n6m5,
                    &db_n7m4,
                    n, 
                    d, 
                    &env, 
                    id_len, 
                    tower,
                    stop,
                    i,
                );
                let x_label = {
                    let stem = std::path::Path::new(s).file_stem().unwrap().to_str().unwrap();
                    let num = stem.strip_prefix("circuit").unwrap_or(stem);
                    format!("Circuit {}", num)
                };

                let y_label = {
                    let stem = std::path::Path::new(d).file_stem().unwrap().to_str().unwrap();
                    let num = stem.strip_prefix("circuit").unwrap_or(stem);
                    format!("Circuit {}", num)
                };
                let path_s = std::path::Path::new(s).file_stem().unwrap().to_str().unwrap();
                let path_d = std::path::Path::new(d).file_stem().unwrap().to_str().unwrap();
                println!(
                    "For generating heatmaps:\n\
                    python3 ./heatmap/heatmap.py \
                    --n {} \
                    --i 100 \
                    --x \"{}\" \
                    --y \"{}\" \
                    --c1 \"{}\" \
                    --c2 \"{}\" \
                    --path ./{}{}.png",
                        n, x_label, y_label, s, d, path_s, path_d
                );
            }
        }
        Some(("sss", sub)) => {
            let rounds: usize = *sub.get_one("rounds").unwrap();
            let s: &str = sub.get_one::<String>("source").unwrap().as_str();
            let d: &str = sub.get_one::<String>("destination").unwrap().as_str();
            let n: usize = *sub.get_one("n").unwrap();
            let m: usize = *sub.get_one("m").unwrap();
            let x: usize = *sub.get_one("x").unwrap();
            let id_len: usize = *sub.get_one("id_len").unwrap();
            let tower = sub.get_flag("tower");
            let leave = sub.get_flag("interleave");
            let stop: usize = *sub.get_one("stop").unwrap();
            let i: &str = sub.get_one::<String>("intermediate").unwrap().as_str();
            let data = fs::read_to_string(s).expect("Failed to read initial.txt");

            let lmdb = "./db";
            let _ = std::fs::create_dir_all(lmdb);

            let env = Environment::new()
                .set_max_readers(10000) 
                .set_max_dbs(266)      
                .set_map_size(800 * 1024 * 1024 * 1024) 
                .open(Path::new(lmdb))
                .expect("Failed to open lmdb");

            let cache = Cache::new_lru_cache(25 * 1024 * 1024 * 1024); 

            let mut block_opts = BlockBasedOptions::default();
            block_opts.set_block_cache(&cache);
            block_opts.set_bloom_filter(10.0, false);
            block_opts.set_cache_index_and_filter_blocks(true);
            block_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);

            let mut opts = Options::default();
            opts.set_block_based_table_factory(&block_opts);
            opts.set_compression_type(rocksdb::DBCompressionType::None);

            let db_n6m5 = DB::open_for_read_only(&opts, "rocksdb_n6m5perms", false)
                .expect("Failed to open RocksDB n6m5");
            let db_n7m4 = DB::open_for_read_only(&opts, "rocksdb_n7m4perms", false)
                .expect("Failed to open RocksDB n7m4");

            install_kill_handler();
            if data.trim().is_empty() {
                println!("Empty file");
            } else {
                let c = CircuitSeq::from_string(&data);
                main_shuffle_shoot_shuffle(
                    &c, 
                    rounds, 
                    &db_n6m5,
                    &db_n7m4,
                    n, 
                    m,
                    x,
                    d, 
                    &env, 
                    id_len, 
                    tower,
                    stop,
                    i,
                    leave,
                );
                let x_label = {
                    let stem = std::path::Path::new(s).file_stem().unwrap().to_str().unwrap();
                    let num = stem.strip_prefix("circuit").unwrap_or(stem);
                    format!("Circuit {}", num)
                };

                let y_label = {
                    let stem = std::path::Path::new(d).file_stem().unwrap().to_str().unwrap();
                    let num = stem.strip_prefix("circuit").unwrap_or(stem);
                    format!("Circuit {}", num)
                };
                let path_s = std::path::Path::new(s).file_stem().unwrap().to_str().unwrap();
                let path_d = std::path::Path::new(d).file_stem().unwrap().to_str().unwrap();
                println!(
                    "For generating heatmaps:\n\
                    python3 ./heatmap/heatmap.py \
                    --n {} \
                    --i 100 \
                    --x \"{}\" \
                    --y \"{}\" \
                    --c1 \"{}\" \
                    --c2 \"{}\" \
                    --path ./{}{}.png",
                        n, x_label, y_label, s, d, path_s, path_d
                );
            }
        }
        Some(("reverse", sub)) => {
            let from_path = sub.get_one::<String>("source").unwrap();
            let dest_path = sub.get_one::<String>("dest").unwrap();
            reverse(from_path, dest_path);
        }
        Some(("gen_reversible", sub)) => {
            let from_path = sub.get_one::<String>("source").unwrap();
            let dest_path = sub.get_one::<String>("dest").unwrap();
            let n: usize = *sub.get_one("n").expect("Missing -n <wires>");
            let lmdb = "./db";
            let env = Environment::new()
                .set_max_dbs(266)      
                .set_map_size(800 * 1024 * 1024 * 1024) 
                .open(Path::new(lmdb))
                .expect("Failed to open lmdb");
            let dbs = open_all_dbs(&env);

            let contents = fs::read_to_string(from_path)
                .unwrap_or_else(|_| panic!("Failed to read circuit file at {}", from_path));

            let c = CircuitSeq::from_string(&contents);
            println!("Creating reversible circuit");
            let reversible = generate_reversible(&c, n, &env, &dbs);
            let mut file = fs::File::create(dest_path)
                .expect("Failed to create new file");
            write!(file, "{}", reversible.repr())
                .expect("Failed to write compressed circuit to file");

            println!("Reversible circuit written to {}", dest_path);
        }
        Some(("compress", sub)) => {
            let s: &String = sub.get_one("s").expect("Missing -s <source>");
            let n: usize = *sub.get_one("n").expect("Missing -n <wires>");
            let d: &String = sub.get_one("d").expect("Missing -d <destination>");
            let seq = sub.get_flag("seq"); 
            let contents = fs::read_to_string(s)
                .unwrap_or_else(|_| panic!("Failed to read circuit file at {}", s));

            let mut acc = CircuitSeq::from_string(&contents);

            let lmdb = "./db";
            let _ = std::fs::create_dir_all(lmdb);

            let env = Environment::new()
                .set_max_dbs(266)      
                .set_map_size(800 * 1024 * 1024 * 1024) 
                .open(Path::new(lmdb))
                .expect("Failed to open lmdb");
            let dbs = open_all_dbs(&env);

            println!("Opening RocksDB");
            let cache = Cache::new_lru_cache(25 * 1024 * 1024 * 1024); 

            let mut block_opts = BlockBasedOptions::default();
            block_opts.set_block_cache(&cache);
            block_opts.set_bloom_filter(10.0, false);
            block_opts.set_cache_index_and_filter_blocks(true);
            block_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);

            let mut opts = Options::default();
            opts.set_block_based_table_factory(&block_opts);
            opts.set_compression_type(rocksdb::DBCompressionType::None);

            let db_n6m5 = DB::open_for_read_only(&opts, "rocksdb_n6m5perms", false)
                .expect("Failed to open RocksDB n6m5");
            let db_n7m4 = DB::open_for_read_only(&opts, "rocksdb_n7m4perms", false)
                .expect("Failed to open RocksDB n7m4");

            let bit_shuf_list = (3..=7)
                .map(|n| {
                    (0..n)
                        .permutations(n)
                        .filter(|p| !p.iter().enumerate().all(|(i, &x)| i == x))
                        .collect::<Vec<Vec<usize>>>()
                })
                .collect();
            // Call compression logic
            println!("Starting compression");
            acc = compress_loop(
                &acc,
                n,
                &db_n6m5,
                &db_n7m4,
                &env,
                &bit_shuf_list,
                &dbs,
                12,
                1,
                1
            );
            let mut file = fs::File::create(d)
                .expect("Failed to create new file");
            write!(file, "{}", acc.repr())
                .expect("Failed to write compressed circuit to file");

            println!("Compressed circuit written to {}", d);
        }
        Some(("shuffle", sub)) => {
            let from_path = sub.get_one::<String>("s").unwrap();
            let dest_path = sub.get_one::<String>("d").unwrap();
            let n: usize = *sub.get_one("n").expect("Missing -n <wires>");
            let i: usize = *sub.get_one("i").expect("Missing -i <insertions>");
            let knuth = sub.get_flag("knuth");
            let lmdb = "./db";
            let env = Environment::new()
                .set_max_dbs(266)      
                .set_map_size(800 * 1024 * 1024 * 1024) 
                .open(Path::new(lmdb))
                .expect("Failed to open lmdb");
            let dbs = open_all_dbs(&env);

            let contents = fs::read_to_string(from_path)
                .unwrap_or_else(|_| panic!("Failed to read circuit file at {}", from_path));

            let mut c = CircuitSeq::from_string(&contents);
            println!("Creating shuffled circuit");
            if i == 0 {
                if knuth {
                    insert_wire_shuffles_knuth(&mut c, n, &env, &dbs);
                } else {
                    insert_wire_shuffles_simple(&mut c, n, &env, &dbs);
                }
            } else {
                insert_wire_shuffles_x(&mut c, n, &env, &dbs, i);
            }
            let mut file = fs::File::create(dest_path)
                .expect("Failed to create new file");
            write!(file, "{}", c.repr())
                .expect("Failed to write compressed circuit to file");

            println!("Shuffled circuit written to {}", dest_path);
        }
        Some(("shoot", sub)) => {
            let from_path = sub.get_one::<String>("s").unwrap();
            let dest_path = sub.get_one::<String>("d").unwrap();
            let i: usize = *sub.get_one("i").expect("Missing -i <iterations>");

            let contents = fs::read_to_string(from_path)
                .unwrap_or_else(|_| panic!("Failed to read circuit file at {}", from_path));

            let mut c = CircuitSeq::from_string(&contents);
            // let mut rng = rand::rng();
            println!("Creating shot circuit");
            shoot_random_gate(&mut c, i);
            // random_sulking(&mut c);
            // c = random_walk_no_skeleton(&mut c, &mut rng);
            let mut file = fs::File::create(dest_path)
                .expect("Failed to create new file");
            write!(file, "{}", c.repr())
                .expect("Failed to write compressed circuit to file");

            println!("Shot circuit written to {}", dest_path);
        }
        Some(("equal", sub)) => {
            let c1_path = sub.get_one::<String>("circuit_a").unwrap();
            let c2_path = sub.get_one::<String>("circuit_b").unwrap();
            let i: usize = *sub.get_one::<usize>("iterations").expect("Missing --iterations");
            let n: usize = *sub.get_one::<usize>("wires").expect("Missing --wires");
            let contents1 = fs::read_to_string(c1_path)
                .unwrap_or_else(|_| panic!("Failed to read circuit file at {}", c1_path));
            let contents2 = fs::read_to_string(c2_path)
                .unwrap_or_else(|_| panic!("Failed to read circuit file at {}", c2_path));
            let c1 = CircuitSeq::from_string(&contents1);
            let c2 = CircuitSeq::from_string(&contents2);
            println!("Checking for equivalence between {} and {}", c1_path, c2_path);
            println!("{}: {} gates", c1_path, c1.gates.len());
            println!("{}: {} gates", c2_path, c2.gates.len());
            if c1.probably_equal(&c2, n, i).is_ok() {
                println!("Circuits are equal!");
            } else {
                println!("Circuits are not equal");
            }
        }
        Some(("wiredot", sub)) => {
            let n: usize = *sub.get_one("num_wires").unwrap();
            
            let path = sub
                .get_one::<String>("path")
                .map(|s| s.as_str())
                .unwrap();

            let xlabel = sub
                .get_one::<String>("xlabel")
                .map(|s| s.as_str())
                .unwrap_or("Circuit 1 gate index");

            let e = format!("Failed to read {}", path);
            let c = fs::read_to_string(path)
                .expect(&e);
            let c = CircuitSeq::from_string(&c);

            analyze_gate_to_wires(&c, n, xlabel).unwrap();
        }
        Some(("lmdb", sub)) => {
            let n: usize = *sub.get_one("n").unwrap();
            let m: usize = *sub.get_one("m").unwrap();
            let _ = sql_to_lmdb(n,m);
        }
        Some(("lmdbp", sub)) => {
            let n: usize = *sub.get_one("n").unwrap();
            let m: usize = *sub.get_one("m").unwrap();
            let _ = sql_to_lmdb_perms(n, m);
        }
        Some(("lmdbcounts", _)) => {
            let env_path = "./db";

            let env = Environment::new()
                .set_max_dbs(266)
                .set_map_size(64 * 1024 * 1024 * 1024)
                .open(Path::new(env_path))
                .expect("Failed to open lmdb");

            let ns_and_ms = [
                (3, 10),
                (4, 6),
                (5, 5),
                (6, 5),
                (7, 4),
            ];

            for (n, max_m) in ns_and_ms {
                let tables: Vec<String> = (1..=max_m)
                    .map(|m| format!("n{}m{}", n, m))
                    .collect();

                // println!("tables: {:?}", tables);
                let perms_to_m =
                    perm_tables_with_duplicates(&env, &tables)
                        .expect("Failed to compute perms");

                let db_name = format!("perm_tables_n{}", n);
                save_perm_tables_to_lmdb(&env_path, &db_name, &perms_to_m)
                    .expect("Failed to save perms");

                println!("Saved perm_tables_n{}", n);
            }
        }
        Some(("lmdbid", sub)) => {
            let n: usize = *sub.get_one("n").unwrap();
            let env_path = "./db";

            let env = Environment::new()
                .set_max_dbs(266)
                .set_map_size(800 * 1024 * 1024 * 1024)
                .open(Path::new(env_path))
                .expect("Failed to open lmdb");
            let ns_and_ms = [
                (5, 5),
                (6, 5),
                (7, 4),
            ];
            let ns_and_ms = vec![ns_and_ms[n - 5]];
            for (n, max_m) in ns_and_ms {
                let tables: Vec<String> = (1..=max_m)
                    .map(|m| format!("n{}m{}", n, m))
                    .collect();

                let perm_circuit_table =
                    circuit_tables_gen(&env, &tables)
                        .expect("Failed to compute perms");

                let tax_id_table = create_tax_id_table(perm_circuit_table);
                let db_name = format!("{}", n);
                save_tax_id_tables_to_lmdb(&env_path, &db_name, &tax_id_table)
                    .expect("Failed to save perms");

                println!("Saved ids_n{}", n);
            }
        }
        Some(("lmdbnid", sub)) => {
            let n: usize = *sub.get_one("n").unwrap();
            fill_n_id(n);
        }
        Some(("string", sub)) => {
            let from_path = sub.get_one::<String>("source").unwrap();
            let dest_path = sub.get_one::<String>("dest").unwrap();
            let input_str = fs::read_to_string(from_path)
                .unwrap_or_else(|e| panic!("Failed to read {}: {}", from_path, e));
            let circuit = CircuitSeq::from_string(input_str.trim());
            let string = circuit.to_string(circuit.used_wires().len());
            fs::write(dest_path, string)
             .unwrap_or_else(|e| panic!("Failed to write {}: {}", dest_path, e));
        }
        Some(("degree", sub)) => {
            let from_path = sub.get_one::<String>("source").unwrap();
            let n: usize = *sub.get_one("n").expect("Missing -n <wires>");
            let start: usize = *sub.get_one("start").expect("Starting index");
            let end: usize = *sub.get_one("end").expect("Ending index");
            let input_str = fs::read_to_string(from_path)
                .unwrap_or_else(|e| panic!("Failed to read {}: {}", from_path, e));
            let circuit = CircuitSeq::from_string(input_str.trim());
            let end = if end == 0 {
                circuit.gates.len()
            } else {
                end
            };
            let degrees = circuit.to_degree_upper(n, start, end);
            for i in 0..n {
                println!("wire {}: {} degree", i, degrees[i]);
            }
        }
        Some(("genran", sub)) => {
            let d: &String = sub.get_one("d").expect("Missing -d <path>");
            let n: usize = *sub.get_one("n").expect("Missing -n <wires>");
            let m: usize = *sub.get_one("m").expect("Missing -n <wires>");
            
            let circuit = random_circuit(n, m);
            let mut file = fs::File::create(d)
                .expect("Failed to create new file");
            write!(file, "{}", circuit.repr())
                .expect("Failed to write random circuit to file");
        }
        _ => unreachable!(),
    }
}

// Simply reverse a circuit and then save it to a given path
pub fn reverse(from_path: &str, dest_path: &str) {
    if !Path::new(from_path).exists() {
        panic!("Source file {} does not exist", from_path);
    }

    // Read circuit string
    let input_str = fs::read_to_string(from_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", from_path, e));

    // Parse into CircuitSeq
    let mut circuit = CircuitSeq::from_string(input_str.trim());

    // Reverse the gates
    circuit.gates.reverse();

    // Convert back to string
    let reversed_str = circuit.repr();

    // Write to destination file
    fs::write(dest_path, reversed_str)
        .unwrap_or_else(|e| panic!("Failed to write {}: {}", dest_path, e));

    println!("Reversed circuit written to {}", dest_path);
}

// Find the number of gates on a particular pin
pub fn analyze_gate_to_wires(circuit: &CircuitSeq, num_wires: usize, x: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut total_counts = vec![0u32; num_wires];
    let mut active_counts = vec![0u32; num_wires];
    for gate in &circuit.gates {
        for (i, &w) in gate.iter().enumerate() {
            total_counts[w as usize] += 1;
            if i == 0 {
                active_counts[w as usize] += 1;
            }
        }
    }

    let root = BitMapBackend::new("wire_plot.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_count = *total_counts.iter().max().unwrap_or(&1);

    let mut chart = ChartBuilder::on(&root)
        .caption("Gate Touches per Wire", "sans-serif, 24")
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0f64..num_wires as f64, 0f64..(max_count as f64 + 1.0))?;

    let x_label = format!("Wire Index ({})", x);
    let _ = chart.configure_mesh()
        .x_desc(x_label)
        .y_desc("Gate Touch Count")
        .draw();

    chart.draw_series(
        (0..num_wires).map(|i| Circle::new((i as f64, total_counts[i] as f64), 6, BLUE.filled())),

    )?
    .label("Total gate count")
    .legend(|(x,y)| Circle::new((x,y), 5, BLUE.filled()));

    chart.draw_series(
        (0..num_wires).map(|i| Circle::new((i as f64, active_counts[i] as f64), 6, RED.filled())),

    )?
    .label("Active gate count")
    .legend(|(x,y)| Circle::new((x,y), 5, BLUE.filled()));

    root.present()?;
    println!("Saved to wire_plot.png");
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Helper code to create LMDB
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

use lmdb::{Environment, Database, WriteFlags, Transaction};
use local_mixing::circuit::Permutation;
use lmdb::Cursor;

// Helper code to convert to lmdb from the old sql db
pub fn sql_to_lmdb(n: usize, m: usize) -> Result<(), ()> {
    let lmdb_path = "./db";
    let map_size_bytes: usize = 800 * 1024 * 1024 * 1024;
    let batch_max_entries: usize = 100_000;

    let sqlite_conn = rusqlite::Connection::open("circuits.db")
        .expect("Failed to open SQLite database");
    let table = format!("n{}m{}", n, m);

    let query = format!("SELECT * FROM {}", table);
    let mut stmt = sqlite_conn.prepare(&query).expect("Failed to prepare SQLite query");
    let mut rows = stmt.query([]).expect("Failed to query SQLite rows");

    fs::create_dir_all(lmdb_path).expect("Failed to create LMDB directory");
    let env = Environment::new()
        .set_max_dbs(266)
        .set_map_size(map_size_bytes)
        .open(Path::new(lmdb_path))
        .expect("Failed to open LMDB environment");

    let db = env.create_db(Some(&table), lmdb::DatabaseFlags::empty())
        .expect("Failed to create LMDB database");

    let mut batch: Vec<Vec<u8>> = Vec::with_capacity(batch_max_entries);
    let mut rows_processed: u64 = 0;

    let flush = |env: &Environment, db: Database, batch: &mut Vec<Vec<u8>>| {
        if batch.is_empty() { return; }
        let mut txn = env.begin_rw_txn().expect("Failed to begin LMDB RW transaction");
        for key in batch.iter() {
            txn.put(db, key, &[], WriteFlags::empty())
                .expect("Failed to write LMDB entry");
        }
        txn.commit().expect("Failed to commit LMDB transaction");
        batch.clear();
    };

    while let Some(row) = rows.next().expect("Failed getting next SQLite row") {
        rows_processed += 1;

        let circuit: Vec<u8> = row.get(0).expect("Failed to read column 'circuit'");
        let perm: Vec<u8> = row.get(1).expect("Failed to read column 'perm'");
        let shuf: Vec<u8> = row.get(2).expect("Failed to read column 'shuf'");

        // check inverse
        let inv = crate::Permutation::from_blob(&perm).invert().repr_blob();
        let mut inv_key = inv.clone();
        inv_key.extend_from_slice(&0u32.to_le_bytes());
        let ro_txn = env.begin_ro_txn().expect("Failed to begin LMDB RO txn");
        if ro_txn.get(db, &inv_key).is_ok() {
            continue
        }

        let mut key = perm.clone();

        let mut circuit_seq = CircuitSeq::from_blob(&circuit);
        circuit_seq.rewire(&Permutation::from_blob(&shuf), n);
        circuit_seq.canonicalize();
        if circuit_seq.gates.windows(2).any(|w| w[0] == w[1]) {
            continue
        }
        // compute key = perm || circuit 
        key.extend_from_slice(&circuit_seq.repr_blob());

        batch.push(key);

        if batch.len() >= batch_max_entries {
            flush(&env, db, &mut batch);
        }

        if rows_processed % 100_000 == 0 {
            println!("Processed {} in {}", rows_processed, table);
        }
    }

    if !batch.is_empty() {
        flush(&env, db, &mut batch);
    }

    println!("Finished copying {} rows into LMDB table {}", rows_processed, table);

    Ok(())
}

pub fn sql_to_lmdb_perms(n: usize, m: usize) -> Result<(), ()> {
    let lmdb_path = "./db";
    let map_size_bytes: usize = 800 * 1024 * 1024 * 1024;
    let batch_max_entries: usize = 100_000;

    // Open SQLite
    let sqlite_conn = rusqlite::Connection::open("circuits.db")
        .expect("Failed to open SQLite database");
    let table = format!("n{}m{}", n, m);
    let table2 = format!("n{}m{}perms", n, m);
    let query = format!("SELECT * FROM {}", table);
    let mut stmt = sqlite_conn.prepare(&query).expect("Failed to prepare SQLite query");
    let mut rows = stmt.query([]).expect("Failed to query SQLite rows");

    // Open LMDB
    fs::create_dir_all(lmdb_path).expect("Failed to create LMDB directory");
    let env = Environment::new()
        .set_max_dbs(266)
        .set_map_size(map_size_bytes)
        .open(Path::new(lmdb_path))
        .expect("Failed to open LMDB environment");

    let db = env.create_db(Some(&table2), lmdb::DatabaseFlags::empty())
        .expect("Failed to create LMDB database");

    let mut batch: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(batch_max_entries);
    let mut rows_processed: u64 = 0;

    // Flush function writes batch to LMDB
    let flush = |env: &Environment, db: Database, batch: &mut Vec<(Vec<u8>, Vec<u8>)>| {
        if batch.is_empty() { return; }
        let mut txn = env.begin_rw_txn().expect("Failed to begin LMDB RW transaction");
        for (key, val) in batch.iter() {
            txn.put(db, key, val, WriteFlags::empty())
                .expect("Failed to write LMDB entry");
        }
        txn.commit().expect("Failed to commit LMDB transaction");
        batch.clear();
    };

    // Iterate SQLite rows
    while let Some(row) = rows.next().expect("Failed getting next SQLite row") {
        rows_processed += 1;

        let circuit: Vec<u8> = row.get(0).expect("Failed to read 'circuit'");
        let perm: Vec<u8> = row.get(1).expect("Failed to read 'perm'");
        let shuf: Vec<u8> = row.get(2).expect("Failed to read 'shuf'");

        // Serialize (perm, shuf) together
        let mut val = Vec::with_capacity(perm.len() + shuf.len());
        val.extend_from_slice(&perm);
        val.extend_from_slice(&shuf);

        batch.push((circuit, val));

        if batch.len() >= batch_max_entries {
            flush(&env, db, &mut batch);
        }

        if rows_processed % 100_000 == 0 {
            println!("Processed {} rows in {}", rows_processed, table);
        }
    }

    if !batch.is_empty() {
        flush(&env, db, &mut batch);
    }

    println!("Finished copying {} rows into LMDB table {}", rows_processed, table);
    Ok(())
}

/// Scans all tables and creates a DB of perms with multiple circuits
use std::collections::HashMap;
use std::collections::HashSet;
fn perm_tables_with_duplicates(
    env: &Environment,
    tables: &[String], // tables like n{num_wires}m{m}
) -> Result<HashMap<Vec<u8>, Vec<u8>>, lmdb::Error> {
    let mut perms_to_m: HashMap<Vec<u8>, Vec<u8>> = HashMap::new();

    for table in tables {
        // parse num_wires and m from table name
        let t = table.strip_prefix('n').unwrap();
        let (n_str, m_str) = t.split_once('m').unwrap();
        let num_wires: usize = n_str.parse().unwrap();
        let m: u8 = m_str.parse().unwrap();

        let perm_len = 1usize << num_wires;

        let db = env.open_db(Some(table))?;
        let ro_txn = env.begin_ro_txn()?;
        let mut cursor = ro_txn.open_ro_cursor(db)?;

        for (k, _) in cursor.iter() {
            let perm = &k[..perm_len];

            // push every occurrence of perm, even duplicates in same table
            perms_to_m
                .entry(perm.to_vec())
                .or_default()
                .push(m);
        }
    }

    perms_to_m.retain(|_, ms| ms.len() > 1);

    Ok(perms_to_m)
}

fn save_perm_tables_to_lmdb(
    env_path: &str,
    db_name: &str,
    perms_to_m: &HashMap<Vec<u8>, Vec<u8>>,
) -> Result<(), Box<dyn std::error::Error>> {

    std::fs::create_dir_all(env_path)?;
    let env = Environment::new()
        .set_max_dbs(266)
        .set_map_size(800 * 1024 * 1024 * 1024)
        .open(Path::new(env_path))?;

    let db = env.create_db(Some(db_name), lmdb::DatabaseFlags::empty())?;

    let batch_size = 100_000;
    let mut batch: Vec<(&Vec<u8>, Vec<u8>)> = Vec::with_capacity(batch_size);

    let flush_batch = |env: &Environment, db: Database, batch: &mut Vec<(&Vec<u8>, Vec<u8>)>| {
        if batch.is_empty() { return; }
        let mut txn = env.begin_rw_txn().expect("Failed to begin LMDB txn");
        for (key, value) in batch.iter() {
            txn.put(db, key, value, WriteFlags::empty())
                .expect("Failed to write LMDB entry");
        }
        txn.commit().expect("Failed to commit LMDB txn");
        batch.clear();
    };

    for (perm, ms) in perms_to_m.iter() {
        let value = bincode::serialize(ms)?;
        batch.push((perm, value));

        if batch.len() >= batch_size {
            flush_batch(&env, db, &mut batch);
        }
    }

    flush_batch(&env, db, &mut batch);

    Ok(())
}

fn circuit_tables_gen(
    env: &Environment,
    tables: &[String], // tables like n{num_wires}m{m}
) -> Result<HashMap<Vec<u8>, Vec<Vec<u8>>>, lmdb::Error> {
    let mut perms_to_circuits: HashMap<Vec<u8>, Vec<Vec<u8>>> = HashMap::new();

    for table in tables {
        let t = table.strip_prefix('n').unwrap();
        let (n_str, _m_str) = t.split_once('m').unwrap();
        let num_wires: usize = n_str.parse().unwrap();

        let perm_len = 1usize << num_wires;

        let db = env.open_db(Some(table))?;
        let ro_txn = env.begin_ro_txn()?;
        let mut cursor = ro_txn.open_ro_cursor(db)?;

        for (k, _) in cursor.iter() {
            let perm = &k[..perm_len];
            let circuit = &k[perm_len..];
            perms_to_circuits
                .entry(perm.to_vec())
                .or_default()
                .push(circuit.to_vec());
        }
    }

    // keep only perms that appear in more than one circuit
    perms_to_circuits.retain(|_, circuits| circuits.len() > 1);

    Ok(perms_to_circuits)
}

fn create_tax_id_table(circuit_table: HashMap<Vec<u8>, Vec<Vec<u8>>>) -> HashMap<GatePair, Vec<Vec<u8>>> {
    let mut tax_table: HashMap<GatePair, HashSet<Vec<u8>>> = HashMap::new();
    for (_, circuits) in circuit_table {
        let n = circuits.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let mut curr_tax_f: HashSet<GatePair> = HashSet::new();
                let mut curr_tax_b: HashSet<GatePair> = HashSet::new();
                let c1 = &circuits[i];
                let c2 = &circuits[j];

                let mut c1 = CircuitSeq::from_blob(&c1);
                let mut c2 = CircuitSeq::from_blob(&c2);
                c2.gates.reverse();
                let mut forward = c1.concat(&c2);
                let mut i = 0;
                while i < forward.gates.len().saturating_sub(1) {
                    if forward.gates[i] == forward.gates[i + 1] {
                        forward.gates.drain(i..=i + 1);
                        i = i.saturating_sub(2);
                    } else {
                        i += 1;
                    }
                }
                c1.gates.reverse();
                c2.gates.reverse();
                let mut back = c2.concat(&c1);
                let mut i = 0;
                while i < back.gates.len().saturating_sub(1) {
                    if back.gates[i] == back.gates[i + 1] {
                        back.gates.drain(i..=i + 1);
                        i = i.saturating_sub(2);
                    } else {
                        i += 1;
                    }
                }
                let len = forward.gates.len();
                for _ in 0..len {
                    let g1 = forward.gates[0];
                    let g2 = forward.gates[1];
                    let ftax = gate_pair_taxonomy(&g1, &g2);
                    let g1 = back.gates[0];
                    let g2 = back.gates[1];
                    let btax = gate_pair_taxonomy(&g1, &g2);
                    if curr_tax_f.insert(ftax) {
                        tax_table
                            .entry(ftax)
                            .or_default()
                            .insert(forward.clone().repr_blob());
                    }

                    if curr_tax_b.insert(btax) {
                        tax_table
                            .entry(btax)
                            .or_default()
                            .insert(back.clone().repr_blob());
                    }

                    let first = forward.gates.remove(0);
                    forward.gates.push(first);
                    let first = back.gates.remove(0);
                    back.gates.push(first);
                }
            }
        }
    }

    tax_table
        .into_iter()
        .map(|(k, v)| (k, v.into_iter().collect()))
        .collect()

}

fn save_tax_id_tables_to_lmdb(
    env_path: &str,
    db_name: &str,
    perms_to_m: &HashMap<GatePair, Vec<Vec<u8>>>,
) -> Result<(), Box<dyn std::error::Error>> {
    use lmdb::{Environment, Database, WriteFlags};
    use std::path::Path;

    // Open environment
    let env = Environment::new()
        .set_max_dbs(266)
        .set_map_size(800 * 1024 * 1024 * 1024)
        .open(Path::new(env_path))?;

    // let dbs_to_delete = [
    
    // ];

    // for db_name in dbs_to_delete.iter() {
    //     if let Ok(db) = env.open_db(Some(db_name)) {
    //         let mut txn = env.begin_rw_txn()?;
    //         // SAFETY: ensure no other transactions or handles are active
    //         unsafe {
    //             txn.drop_db(db)?;
    //         }
    //         txn.commit()?;
    //         println!("Dropped DB: {}", db_name);
    //     } else {
    //         println!("DB not found: {}", db_name);
    //     }
    // }

    let batch_size = 100;
    let mut batch: Vec<Vec<u8>> = Vec::with_capacity(batch_size);

    let flush_batch = |env: &Environment, db: Database, batch: &mut Vec<Vec<u8>>| {
        if batch.is_empty() {
            return;
        }
        println!("Flushing batch");
        let mut txn = env.begin_rw_txn().expect("Failed to begin LMDB txn");
        for key in batch.iter() {
            txn.put(db, key, &[], WriteFlags::empty())
                .expect("Failed to write LMDB key");
        }
        txn.commit().expect("Failed to commit LMDB txn");
        batch.clear();
    };

    for (gatepair, circuits) in perms_to_m.iter() {
        let g = GatePair::to_int(gatepair); // your conversion function
        let dynamic_db_name = format!("ids_n{}g{}", db_name, g);
        let db = env.create_db(Some(&dynamic_db_name), lmdb::DatabaseFlags::empty())?;

        for circuit in circuits {
            batch.push(circuit.clone());

            if batch.len() >= batch_size {
                flush_batch(&env, db, &mut batch);
            }
        }

        flush_batch(&env, db, &mut batch);
    }

    Ok(())
}

use rand::Rng;
fn gen_mean(circuit: &CircuitSeq, num_wires: usize) -> f64 {
    let circuit_one = circuit.clone();
    let circuit_two = circuit;

    let circuit_one_len = circuit_one.gates.len();
    let circuit_two_len = circuit_two.gates.len();

    let num_points = (circuit_one_len + 1) * (circuit_two_len + 1);
    let mut average = vec![0f64; num_points * 3];

    let mut rng = rand::rng();
    let num_inputs = 20;

    for _ in 0..num_inputs {
        // if i % 10 == 0 {
        //     // println!("{}/{}", i, num_inputs);
        //     io::stdout().flush().unwrap();
        // }

        let input_bits: u128 = if num_wires < u128::BITS as usize {
            rng.random_range(0..(1u128 << num_wires))
        } else {
            rng.random_range(0..=u128::MAX)
        };

        let evolution_one = circuit_one.evaluate_evolution_128(input_bits);
        let evolution_two = circuit_two.evaluate_evolution_128(input_bits);

        for i1 in 0..=circuit_one_len {
            for i2 in 0..=circuit_two_len {
                let diff = evolution_one[i1] ^ evolution_two[i2];
                let hamming_dist = diff.count_ones() as f64;
                let overlap = hamming_dist / num_wires as f64;

                let index = i1 * (circuit_two_len + 1) + i2;
                average[index * 3] = i1 as f64;
                average[index * 3 + 1] = i2 as f64;
                average[index * 3 + 2] += overlap / num_inputs as f64;
            }
        }
    }

    let mut sum = 0.0;
    for i in 0..num_points {
        sum += average[i * 3 + 2];
    }

    sum / num_points as f64
}

pub fn fill_n_id(n: usize) {
    use std::{
        collections::HashMap,
        path::Path,
        sync::{
            atomic::{AtomicU64, Ordering},
            Arc,
        },
        thread,
        time::Instant,
    };

    use crossbeam_channel::{bounded, Receiver, Sender};
    use lmdb::{Database, Environment, WriteFlags};


    const WORKERS: usize = 60;
    const BATCH_SIZE: usize = 10;
    let env = Arc::new(
        Environment::new()
            .set_max_dbs(266)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .set_max_readers(512)
            .open(Path::new("./db")).expect("Failed to open env")
    );
    let dbs = Arc::new(open_all_dbs(&env));
    // Drop existing DBs
    for g in 0..34 {
        let db_name = format!("ids_n{}g{}single", n, g);
        if let Ok(db) = env.open_db(Some(&db_name)) {
            let mut txn = env.begin_rw_txn().unwrap();
            let _ = unsafe { txn.drop_db(db) };
        let _ = txn.commit();
        }
        let db_name = format!("ids_n{}g{}tower", n, g);
        if let Ok(db) = env.open_db(Some(&db_name)) {
            let mut txn = env.begin_rw_txn().unwrap();
            let _ = unsafe { txn.drop_db(db) };
        let _ = txn.commit();
        }
    }

    let bit_shuf_list = Arc::new(
        (3..=7)
            .map(|n| {
                (0..n)
                    .permutations(n)
                    .filter(|p| !p.iter().enumerate().all(|(i, &x)| i == x))
                    .collect::<Vec<Vec<usize>>>()
            })
            .collect::<Vec<_>>(),
    );

    let total_written = Arc::new(AtomicU64::new(0));

    let (tx, rx): (Sender<((u8, bool), Vec<u8>)>, Receiver<((u8, bool), Vec<u8>)>) =
        bounded(100_000);

    //flush

    {
        let env = Arc::clone(&env);
        let total_written = Arc::clone(&total_written);

        thread::spawn(move || {
            let mut batches: HashMap<(u8, bool), Vec<Vec<u8>>> = HashMap::new();
            let mut db_cache: HashMap<(u8, bool), Database> = HashMap::new();
            let mut written_per_g: HashMap<(u8, bool), u64> = HashMap::new();
            let mut last_print = Instant::now();

            loop {
                let ((g, tower), key) = rx.recv().unwrap();

                let db = *db_cache.entry((g, tower)).or_insert_with(|| {
                    let suffix = if tower { "tower" } else { "single" };
                    let name = format!("ids_n{}g{}{}", n, g, suffix);
                    env.create_db(Some(&name), lmdb::DatabaseFlags::empty())
                        .unwrap()
                });

                let batch = batches.entry((g, tower)).or_default();
                batch.push(key);

                if batch.len() >= BATCH_SIZE {
                    let mut txn = env.begin_rw_txn().unwrap();

                    for v in batch.drain(..) {
                        txn.put(db, &v, &[], WriteFlags::empty())
                            .unwrap();
                    }

                    txn.commit().unwrap();
                    total_written.fetch_add(BATCH_SIZE as u64, Ordering::Relaxed);
                    *written_per_g.entry((g, tower)).or_insert(0) += BATCH_SIZE as u64;
                }

                if last_print.elapsed().as_secs() >= 60 {
                    println!("total written: {}",
                        total_written.load(Ordering::Relaxed)
                    );

                    for g in 0..34 {
                        let single = written_per_g
                            .get(&(g as u8, false))
                            .copied()
                            .unwrap_or(0);
                        let tower = written_per_g
                            .get(&(g as u8, true))
                            .copied()
                            .unwrap_or(0);

                        println!("g {:02}: single {:>8} | tower {:>8}", g, single, tower);
                    }
                    last_print = Instant::now();
                }
            }
        });
    }

    
    //workers
    let mut handles = Vec::new();

    for _ in 0..WORKERS {
        let tx = tx.clone();
        let env = env.clone();
        let dbs = dbs.clone();
        let bit_shuf_list = bit_shuf_list.clone();

        handles.push(thread::spawn(move || {
            loop {
                let tower = rand::rng().random_bool(0.5);

                let mut id = get_random_wide_identity(
                    n,
                    &env,
                    &dbs,
                    &bit_shuf_list,
                    tower,
                );

                let len = id.gates.len();

                for _ in 0..len {
                    if gen_mean(&id, n) < 0.335 {
                        let first = id.gates.remove(0);
                        id.gates.push(first);
                        continue;
                    }

                    let g1 = id.gates[0];
                    let g2 = id.gates[1];
                    let gp = gate_pair_taxonomy(&g1, &g2);
                    let g = GatePair::to_int(&gp) as u8;

                    tx.send(((g, tower), id.repr_blob())).unwrap();

                    let first = id.gates.remove(0);
                    id.gates.push(first);
                }
            }
        }));
    }

    for h in handles {
        let _ = h.join();
    }
}