use clap::{Arg, ArgAction, Command};
use itertools::Itertools;
use plotters::prelude::*;
use rusqlite::{Connection, OpenFlags};

use std::{
    fs,
    io::Write,
    path::Path,
};

use local_mixing::{
    circuit::CircuitSeq,
    random::random_data::{build_from_sql, main_random, random_circuit},
    replace::{
        mixing::{
            install_kill_handler,
        },
        main_mix::{
            main_butterfly,
            main_butterfly_big,
            main_interleave_big,
            main_mix,
            main_rac_big,
            main_rac_big_distance,
            main_shuffle_rcs_big,
            open_all_dbs,
        },
        pairs::{gate_pair_taxonomy, GatePair},
        identities::{get_random_wide_identity, random_canonical_id},
        transpositions::generate_reversible,
        replace::{
            compress_big_ancillas,
            sequential_compress_big_ancillas,
        },
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
                Arg::new("intermediate")
                    .short('i')
                    .long("intermediate")
                    .required(true)
                    .value_parser(clap::value_parser!(String))
                    .help("Path to the intermediate circuit file"),
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
                    Arg::new("p")
                        .short('p')
                        .long("path")
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
            Command::new("lmdbcounts")
            .about("Generate table for generating canon ids")
        )
        .subcommand(
            Command::new("lmdbid")
            .about("Generate table for generating canon ids")
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
        .get_matches();

    match matches.subcommand() {
        Some(("load", sub)) => {
            let n: usize = *sub.get_one("n").unwrap();
            let m: usize = *sub.get_one("m").unwrap();
            // main_rainbow_load(n, m, "./db");
            
            // Open DB connection
            let mut conn = Connection::open("./circuits.db").expect("Failed to open DB");
            conn.execute_batch(
                    "
                    PRAGMA synchronous = OFF;
                    PRAGMA journal_mode = WAL;
                    PRAGMA temp_store = MEMORY;
                    PRAGMA cache_size = -200000;
                    PRAGMA locking_mode = EXCLUSIVE;
                    "
                ).unwrap();
            let perms: Vec<Vec<usize>> = (0..n).permutations(n).collect();
            let bit_shuf = perms.into_iter().skip(1).collect::<Vec<_>>();
            build_from_sql(&mut conn, n,m, &bit_shuf).expect("Unknown error occured");
        }
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
        Some(("mix", sub)) => {
            let rounds: usize = *sub.get_one("rounds").unwrap();

            let data = fs::read_to_string("initial.txt").expect("Failed to read initial.txt");
            // let seed = OsRng.try_next_u64().unwrap_or_else(|e| {
            //     panic!("Failed to generate random seed: {}", e);
            // });
            // println!("Using seed: {}", seed);
            if data.trim().is_empty() {
                // Open DB connection
                let mut conn = Connection::open_with_flags("./circuits.db",OpenFlags::SQLITE_OPEN_READ_ONLY,).expect("Failed to open DB (read-only)");
                conn.execute_batch(
                    "
                    PRAGMA synchronous = NORMAL;
                    PRAGMA journal_mode = WAL;
                    PRAGMA temp_store = MEMORY;
                    PRAGMA cache_size = -200000;
                    PRAGMA locking_mode = EXCLUSIVE;
                    "
                ).unwrap();
                let lmdb = "./db";
                let env = Environment::new()
                .set_max_dbs(263)      
                .set_map_size(700 * 1024 * 1024 * 1024) 
                .open(Path::new(lmdb))
                .expect("Failed to open lmdb");

                // Fallback when file is empty
                let c1= random_canonical_id(&env, &conn, 5).unwrap();
                println!("{:?} Starting Len: {}", c1.permutation(5).data, c1.gates.len());
                main_mix(&c1, rounds, &mut conn, 5);
            } else {
                
                let c = CircuitSeq::from_string(&data);

                // Open DB connection
                let mut conn = Connection::open_with_flags("./circuits.db",OpenFlags::SQLITE_OPEN_READ_ONLY,).expect("Failed to open DB (read-only)");
                conn.execute_batch(
                    "
                    PRAGMA synchronous = NORMAL;
                    PRAGMA journal_mode = WAL;
                    PRAGMA temp_store = MEMORY;
                    PRAGMA cache_size = -200000;
                    PRAGMA locking_mode = EXCLUSIVE;
                    "
                ).unwrap();
                main_mix(&c, rounds, &mut conn, 5);
            }
        }
        Some(("butterfly", sub)) => {
            let rounds: usize = *sub.get_one("rounds").unwrap();
            let data = fs::read_to_string("initial.txt").expect("Failed to read initial.txt");
            // let seed = OsRng.try_next_u64().unwrap_or_else(|e| {
            //     panic!("Failed to generate random seed: {}", e);
            // });
            // println!("Using seed: {}", seed);
            if data.trim().is_empty() {
                // Open DB connection
                let mut conn = Connection::open_with_flags("./db/circuits.db",OpenFlags::SQLITE_OPEN_READ_ONLY,).expect("Failed to open DB (read-only)");
                conn.execute_batch(
                    "
                    PRAGMA synchronous = NORMAL;
                    PRAGMA journal_mode = WAL;
                    PRAGMA temp_store = MEMORY;
                    PRAGMA cache_size = -200000;
                    PRAGMA locking_mode = EXCLUSIVE;
                    "
                ).unwrap();
                // Fallback when file is empty
                println!("Generating random");
                let c1= random_circuit(6,30);
                // let perms: Vec<Vec<usize>> = (0..5).permutations(5).collect();
                // let bit_shuf = perms.into_iter().skip(1).collect::<Vec<_>>();
                // let c1 = compress(&random_circuit(5,128), 100_000, &mut conn, &bit_shuf,5 );
                println!("{:?} Starting Len: {}", c1.permutation(6).data, c1.gates.len());
                main_butterfly(&c1, rounds, &mut conn, 6);
            } else {
                
                let c = CircuitSeq::from_string(&data);

                // Open DB connection
                let mut conn = Connection::open_with_flags("./db/circuits.db",OpenFlags::SQLITE_OPEN_READ_ONLY,).expect("Failed to open DB (read-only)");
                conn.execute_batch(
                    "
                    PRAGMA synchronous = NORMAL;
                    PRAGMA journal_mode = WAL;
                    PRAGMA temp_store = MEMORY;
                    PRAGMA cache_size = -200000;
                    PRAGMA locking_mode = EXCLUSIVE;
                    "
                ).unwrap();
                main_butterfly(&c, rounds, &mut conn, 6);
            }
        }
        Some(("bbutterfly", sub)) => {
            let rounds: usize = *sub.get_one("rounds").unwrap();
            let path: &str = sub.get_one::<String>("path").unwrap().as_str();
            let n: usize = *sub.get_one("n").unwrap_or(&32); // default to 32 if not provided
            let data = fs::read_to_string("initial.txt").expect("Failed to read initial.txt");

            let mut conn = Connection::open("./circuits.db").expect("Failed to open DB");
            conn.execute_batch(
                "
                PRAGMA temp_store = MEMORY;
                PRAGMA cache_size = -200000;
                "
            ).unwrap();

            let lmdb = "./db";
            let _ = std::fs::create_dir_all(lmdb);

            let env = Environment::new()
                .set_max_dbs(263)      
                .set_map_size(700 * 1024 * 1024 * 1024) 
                .open(Path::new(lmdb))
                .expect("Failed to open lmdb");

            if data.trim().is_empty() {
                println!("Generating random");
                let c1 = random_circuit(n as u8, 30);
                println!("Starting Len: {}", c1.gates.len());
                main_butterfly_big(&c1, rounds, &mut conn, n, false, path, &env);
            } else {
                let c = CircuitSeq::from_string(&data);
                main_butterfly_big(&c, rounds, &mut conn, n, false, path, &env);
            }
        }

        Some(("abbutterfly", sub)) => {
            let rounds: usize = *sub.get_one("rounds").unwrap();
            let path: &str = sub.get_one::<String>("path").unwrap().as_str();
            let n: usize = *sub.get_one("n").unwrap_or(&32); // default to 32 if not provided
            let data = fs::read_to_string("initial.txt").expect("Failed to read initial.txt");
            let bookendless = sub.get_flag("bookendless"); 

            let mut conn = Connection::open("./circuits.db").expect("Failed to open DB");
            conn.execute_batch(
                "
                PRAGMA temp_store = MEMORY;
                PRAGMA cache_size = -200000;
                "
            ).unwrap();
            let lmdb = "./db";
            let _ = std::fs::create_dir_all(lmdb);

            let env = Environment::new()
                .set_max_readers(10000) 
                .set_max_dbs(263)      
                .set_map_size(800 * 1024 * 1024 * 1024) 
                .open(Path::new(lmdb))
                .expect("Failed to open lmdb");
            install_kill_handler();
            if data.trim().is_empty() {
                println!("Generating random");
                let c1 = random_circuit(n as u8, 30);
                println!("Starting Len: {}", c1.gates.len());
                if bookendless {
                    // main_butterfly_big_bookendsless(&c1, rounds, &mut conn, n, true, path, &env);
                } else {
                    main_butterfly_big(&c1, rounds, &mut conn, n, true, path, &env);
                }
            } else {
                let c = CircuitSeq::from_string(&data);
                if bookendless {
                    // main_butterfly_big_bookendsless(&c, rounds, &mut conn, n, true, path, &env);
                } else {
                    main_butterfly_big(&c, rounds, &mut conn, n, true, path, &env);
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
            let data = fs::read_to_string(s).expect("Failed to read initial.txt");

            let mut conn = Connection::open("./circuits.db").expect("Failed to open DB");
            conn.execute_batch(
                "
                PRAGMA temp_store = MEMORY;
                PRAGMA cache_size = -200000;
                "
            ).unwrap();
            let lmdb = "./db";
            let _ = std::fs::create_dir_all(lmdb);

            let env = Environment::new()
                .set_max_readers(10000) 
                .set_max_dbs(263)      
                .set_map_size(800 * 1024 * 1024 * 1024) 
                .open(Path::new(lmdb))
                .expect("Failed to open lmdb");
            install_kill_handler();
            if data.trim().is_empty() {
                println!("Empty file");
            } else {
                let c = CircuitSeq::from_string(&data);
                main_rac_big(&c, rounds, &mut conn, n, d, &env, i, tower);
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
                    python3 ./heatmap/heatmap_raw.py \
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
            let n: usize = *sub.get_one("n").unwrap_or(&32); // default to 32 if not provided
            let data = fs::read_to_string(s).expect("Failed to read initial.txt");

            let mut conn = Connection::open("./circuits.db").expect("Failed to open DB");
            conn.execute_batch(
                "
                PRAGMA temp_store = MEMORY;
                PRAGMA cache_size = -200000;
                "
            ).unwrap();
            let lmdb = "./db";
            let _ = std::fs::create_dir_all(lmdb);

            let env = Environment::new()
                .set_max_readers(10000) 
                .set_max_dbs(263)      
                .set_map_size(800 * 1024 * 1024 * 1024) 
                .open(Path::new(lmdb))
                .expect("Failed to open lmdb");
            install_kill_handler();
            if data.trim().is_empty() {
                println!("Empty file");
            } else {
                let c = CircuitSeq::from_string(&data);
                main_shuffle_rcs_big(&c, rounds, &mut conn, n, d, &env, i, tower);
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
                    python3 ./heatmap/heatmap_raw.py \
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
            let data = fs::read_to_string(s).expect("Failed to read initial.txt");

            let mut conn = Connection::open("./circuits.db").expect("Failed to open DB");
            conn.execute_batch(
                "
                PRAGMA temp_store = MEMORY;
                PRAGMA cache_size = -200000;
                "
            ).unwrap();
            let lmdb = "./db";
            let _ = std::fs::create_dir_all(lmdb);

            let env = Environment::new()
                .set_max_readers(10000) 
                .set_max_dbs(263)      
                .set_map_size(800 * 1024 * 1024 * 1024) 
                .open(Path::new(lmdb))
                .expect("Failed to open lmdb");
            install_kill_handler();
            if data.trim().is_empty() {
                println!("Empty file");
            } else {
                let c = CircuitSeq::from_string(&data);
                main_interleave_big(&c, rounds, &mut conn, n, d, &env, i, tower);
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
                    python3 ./heatmap/heatmap_raw.py \
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
            let tower = sub.get_flag("tower");
            let data = fs::read_to_string(s).expect("Failed to read initial.txt");

            let mut conn = Connection::open("./circuits.db").expect("Failed to open DB");
            conn.execute_batch(
                "
                PRAGMA temp_store = MEMORY;
                PRAGMA cache_size = -200000;
                "
            ).unwrap();
            let lmdb = "./db";
            let _ = std::fs::create_dir_all(lmdb);

            let env = Environment::new()
                .set_max_readers(10000) 
                .set_max_dbs(263)      
                .set_map_size(800 * 1024 * 1024 * 1024) 
                .open(Path::new(lmdb))
                .expect("Failed to open lmdb");
            install_kill_handler();
            if data.trim().is_empty() {
                println!("Empty file");
            } else {
                let c = CircuitSeq::from_string(&data);
                main_rac_big_distance(&c, rounds, &mut conn, n, d, &env, i, m, tower);
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
                    python3 ./heatmap/heatmap_raw.py \
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
                .set_max_dbs(263)      
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
            let p: &String = sub.get_one("p").expect("Missing -p <path>");
            let n: usize = *sub.get_one("n").expect("Missing -n <wires>");
            let d: &String = sub.get_one("d").expect("Missing -d <path>");
            let seq = sub.get_flag("seq"); 
            let contents = fs::read_to_string(p)
                .unwrap_or_else(|_| panic!("Failed to read circuit file at {}", p));

            let mut acc = CircuitSeq::from_string(&contents);

            let mut conn = Connection::open_with_flags("./circuits.db",rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,)
            .expect("Failed to open ./circuits.db in read-only mode");
            let lmdb = "./db";
            let _ = std::fs::create_dir_all(lmdb);

            let env = Environment::new()
                .set_max_dbs(263)      
                .set_map_size(800 * 1024 * 1024 * 1024) 
                .open(Path::new(lmdb))
                .expect("Failed to open lmdb");
            let dbs = open_all_dbs(&env);
            let bit_shuf_list = (3..=7)
                .map(|n| {
                    (0..n)
                        .permutations(n)
                        .filter(|p| !p.iter().enumerate().all(|(i, &x)| i == x))
                        .collect::<Vec<Vec<usize>>>()
                })
                .collect();
            // Call compression logic
            let mut stable_count = 0;
            while stable_count < 6 {
                let before = acc.gates.len();
                acc = if !seq {
                    compress_big_ancillas(&acc, 1_000, n, &mut conn, &env, &bit_shuf_list, &dbs)
                } else {
                    sequential_compress_big_ancillas(&acc, n, &mut conn, &env, &bit_shuf_list, &dbs);
                    acc.gates.reverse();
                    sequential_compress_big_ancillas(&acc, n, &mut conn, &env, &bit_shuf_list, &dbs);
                    acc.gates.reverse();
                    acc
                };
                let after = acc.gates.len();

                if after == before {
                    stable_count += 1;
                    println!("  Final compression stable {}/3 at {} gates", stable_count, after);
                } else {
                    println!("  Final compression reduced: {} → {} gates", before, after);
                    stable_count = 0;
                }
            }
            let mut file = fs::File::create(d)
                .expect("Failed to create new file");
            write!(file, "{}", acc.repr())
                .expect("Failed to write compressed circuit to file");

            println!("Compressed circuit written to {}", d);
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
            let _ = sql_to_lmdb_perms(n, m);
        }
        Some(("lmdbcounts", _)) => {
            let env_path = "./db";

            let env = Environment::new()
                .set_max_dbs(263)
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
        Some(("lmdbid", _)) => {
            let env_path = "./db";

            let env = Environment::new()
                .set_max_dbs(263)
                .set_map_size(800 * 1024 * 1024 * 1024)
                .open(Path::new(env_path))
                .expect("Failed to open lmdb");

            let ns_and_ms = [
                // (5, 5),
                // (6, 5),
                (7, 4),
            ];

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
        Some(("genran", sub)) => {
            let d: &String = sub.get_one("d").expect("Missing -d <path>");
            let n: usize = *sub.get_one("n").expect("Missing -n <wires>");
            let m: usize = *sub.get_one("m").expect("Missing -n <wires>");
            
            let circuit = random_circuit(n as u8, m);
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
    let sqlite_path = "circuits.db";
    let lmdb_path = "./db";
    let map_size_bytes: usize = 800 * 1024 * 1024 * 1024;
    let batch_max_entries: usize = 100_000;

    let conn = Connection::open(sqlite_path).expect("Failed to open sqlite database");
    let table = format!("n{}m{}", n, m);

    let query = format!("SELECT * FROM {}", table);
    let mut stmt = conn.prepare(&query).expect("Failed to prepare SQLite query");
    let mut rows = stmt.query([]).expect("Failed to query SQLite rows");

    fs::create_dir_all(lmdb_path).expect("Failed to create LMDB directory");
    let env = Environment::new()
        .set_max_dbs(263)
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
    let sqlite_path = "circuits.db";
    let lmdb_path = "./db";
    let map_size_bytes: usize = 800 * 1024 * 1024 * 1024;
    let batch_max_entries: usize = 100_000;

    // Open SQLite
    let conn = Connection::open(sqlite_path).expect("Failed to open SQLite database");
    let table = format!("n{}m{}", n, m);
    let table2 = format!("n{}m{}perms", n, m);
    let query = format!("SELECT * FROM {}", table);
    let mut stmt = conn.prepare(&query).expect("Failed to prepare SQLite query");
    let mut rows = stmt.query([]).expect("Failed to query SQLite rows");

    // Open LMDB
    fs::create_dir_all(lmdb_path).expect("Failed to create LMDB directory");
    let env = Environment::new()
        .set_max_dbs(263)
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
        .set_max_dbs(263)
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
                c1.gates.reverse();
                c2.gates.reverse();
                let mut back = c2.concat(&c1);

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
        .set_max_dbs(263)
        .set_map_size(800 * 1024 * 1024 * 1024)
        .open(Path::new(env_path))?;

    let dbs_to_delete = [
        "ids_nids_n5g14",
        "ids_nids_n5g18",
        "ids_nids_n5g19",
        "ids_nids_n5g23",
        "ids_nids_n5g3",
        "ids_nids_n5g30",
        "ids_nids_n5g6",
        "ids_nids_n5g9",
    ];

    for db_name in dbs_to_delete.iter() {
        if let Ok(db) = env.open_db(Some(db_name)) {
            let mut txn = env.begin_rw_txn()?;
            // SAFETY: ensure no other transactions or handles are active
            unsafe {
                txn.drop_db(db)?;
            }
            txn.commit()?;
            println!("Dropped DB: {}", db_name);
        } else {
            println!("DB not found: {}", db_name);
        }
    }

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
    use rusqlite::{Connection, OpenFlags};

    const WORKERS: usize = 60;
    const BATCH_SIZE: usize = 10;


    let env_path = "./db";
    let env = Environment::new()
        .set_max_dbs(263)
        .set_map_size(800 * 1024 * 1024 * 1024)
        .open(Path::new(env_path))
        .expect("Failed to open db");

    // Drop existing DBs
    // for g in 0..34 {
    //     let db_name = format!("ids_n{}g{}", n, g);
    //     if let Ok(db) = env.open_db(Some(&db_name)) {
    //         let mut txn = env.begin_rw_txn().unwrap();
    //         unsafe { txn.drop_db(db).unwrap() };
    //         txn.commit().unwrap();
    //     }
    // }

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

    let key_counter = Arc::new(AtomicU64::new(0));
    let total_written = Arc::new(AtomicU64::new(0));

    let (tx, rx): (Sender<((u8, bool), Vec<u8>)>, Receiver<((u8, bool), Vec<u8>)>) =
        bounded(100_000);

    //flush

    let env_flush = env;
    let total_written_flush = total_written.clone();
    let key_counter_flush = key_counter.clone();
    let flush_handle = thread::spawn(move || {
        let mut batches: HashMap<(u8, bool), Vec<Vec<u8>>> = HashMap::new();
        let mut db_cache: HashMap<(u8, bool), Database> = HashMap::new();
        let mut written_per_g: HashMap<(u8, bool), u64> = HashMap::new();
        let mut last_print = Instant::now();

        loop {
            let ((g, tower), key) = rx.recv().expect("worker hung up");

            let db = *db_cache.entry((g, tower)).or_insert_with(|| {
                let suffix = if tower { "tower" } else { "single" };
                let name = format!("ids_n{}g{}{}", n, g, suffix);

                env_flush
                    .create_db(Some(&name), lmdb::DatabaseFlags::empty())
                    .expect("create db")
            });

            let batch = batches.entry((g, tower)).or_default();
            batch.push(key);

            if batch.len() >= BATCH_SIZE {
                let already_written = written_per_g
                    .get(&(g, tower))
                    .copied()
                    .unwrap_or(0);

                if already_written >= 10_000{
                    batch.clear();
                    continue;
                }
                let mut txn = env_flush.begin_rw_txn().unwrap();
                for v in batch.iter() {
                    let k = key_counter_flush.fetch_add(1, Ordering::Relaxed);
                    let key_bytes = k.to_be_bytes();
                    txn.put(db, &key_bytes, v, WriteFlags::empty()).unwrap();
                }
                txn.commit().unwrap();

                let written = batch.len() as u64;
                batch.clear();

                total_written_flush.fetch_add(written, Ordering::Relaxed);
                *written_per_g.entry((g, tower)).or_insert(0) += written;
            }

            if last_print.elapsed().as_secs() >= 60 {
                println!("total written: {}", total_written_flush.load(Ordering::Relaxed));
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

    
    //workers
    let mut handles = Vec::new();

    for _ in 0..WORKERS {
        let tx = tx.clone();
        let bit_shuf_list = bit_shuf_list.clone();

        handles.push(thread::spawn(move || {
            let mut conn = Connection::open_with_flags(
                "circuits.db",
                OpenFlags::SQLITE_OPEN_READ_ONLY,
            )
            .expect("sqlite open");
            let env_path = "./db";
            let env = Environment::new()
                .set_max_dbs(263)
                .set_map_size(800 * 1024 * 1024 * 1024)
                .open(Path::new(env_path))
                .expect("Failed to open db");
            let dbs = open_all_dbs(&env);
            loop {
                let tower = rand::rng().random_bool(0.5);
                let mut id = get_random_wide_identity(
                    n,
                    &env,
                    &dbs,
                    &mut conn,
                    &bit_shuf_list,
                    tower,
                );

                let len = id.gates.len();

                for _ in 0..len {
                    if gen_mean(&id, n) < 0.235 {
                        let first = id.gates.remove(0);
                        id.gates.push(first);
                        continue;
                    }

                    let g1 = id.gates[0];
                    let g2 = id.gates[1];
                    let gp = gate_pair_taxonomy(&g1, &g2);
                    let g = GatePair::to_int(&gp) as u8;

                    let key = id.repr_blob();
                    tx.send(((g, tower), key)).unwrap();

                    let first = id.gates.remove(0);
                    id.gates.push(first);
                }
            }
        }));
    }

    let _ = flush_handle.join();
    for h in handles {
        let _ = h.join();
    }
}

