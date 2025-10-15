use local_mixing::{
    circuit::CircuitSeq,
    rainbow::{
        explore::explore_db,
        rainbow::{main_rainbow_generate, main_rainbow_load},
    },
    random::random_data::{build_from_sql, main_random, random_circuit},
    replace::{
        mixing::main_mix,
        replace::{random_canonical_id, random_id, compress},
    },
};
use rusqlite::OpenFlags;
use local_mixing::replace::mixing::main_butterfly;
use local_mixing::replace::mixing::main_butterfly_big;
use clap::{Arg, ArgAction, Command};
use itertools::Itertools;
use rand::rngs::OsRng;
use rand::TryRngCore;
use rusqlite::Connection;
use std::fs;

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
                    .value_parser(clap::value_parser!(usize))
            ),
    )

        .get_matches();

    match matches.subcommand() {
        Some(("new", sub)) => {
            let n: usize = *sub.get_one("n").unwrap();
            let m: usize = *sub.get_one("m").unwrap();
            main_rainbow_generate(n, m);
        }
        Some(("load", sub)) => {
            let n: usize = *sub.get_one("n").unwrap();
            let m: usize = *sub.get_one("m").unwrap();
            // main_rainbow_load(n, m, "./db");
            
            // Open DB connection
            let mut conn = Connection::open("./db/circuits.db").expect("Failed to open DB");
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
        Some(("explore", sub)) => {
            let n: usize = *sub.get_one("n").unwrap();
            let m: usize = *sub.get_one("m").unwrap();
            explore_db(n, m);
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
                let c1= random_canonical_id(&conn, 5).unwrap();
                println!("{:?} Starting Len: {}", c1.permutation(5).data, c1.gates.len());
                main_mix(&c1, rounds, &mut conn, 5);
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

        let data = fs::read_to_string("initial.txt").expect("Failed to read initial.txt");

        let mut conn = Connection::open("./circuits.db").expect("Failed to open DB");
        conn.execute_batch(
            "
            PRAGMA temp_store = MEMORY;
            PRAGMA cache_size = -200000;
            "
        ).unwrap();

        if data.trim().is_empty() {
            println!("Generating random");
            let c1 = random_circuit(16,30);
            println!("{:?} Starting Len: {}", c1.permutation(16).data, c1.gates.len());
            main_butterfly_big(&c1, rounds, &mut conn, 16);
        } else {
            let c = CircuitSeq::from_string(&data);
            main_butterfly_big(&c, rounds, &mut conn, 16);
        }
    }
        _ => unreachable!(),
    }
}