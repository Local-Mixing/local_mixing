use clap::{Arg, ArgAction, Command};
use itertools::Itertools;
use plotters::prelude::*;
use rand::{Rng};
use rusqlite::{Connection, OpenFlags};
use serde_json::json;
use std::{
    fs::{self},
    io::Write,
    path::Path,
    time::Instant,
    process::Stdio,
};
use local_mixing::replace::replace::expand_big;
use local_mixing::replace::replace::compress_big;
use local_mixing::{
    circuit::CircuitSeq,
    rainbow::{
        explore::explore_db,
        rainbow::{main_rainbow_load},
    },
    random::random_data::{build_from_sql, main_random, random_circuit},
    replace::{
        mixing::{main_butterfly, main_butterfly_big, main_mix, main_butterfly_big_bookendsless},
        replace::{random_canonical_id},
    },
};

use local_mixing::replace::replace::compress_big_ancillas;
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
                    Arg::new("n")
                        .short('n')
                        .long("wires")
                        .required(true)
                        .value_parser(clap::value_parser!(usize))
                        .help("Number of wires in the circuit"),
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
        Some(("binload", sub)) => {
            let n: usize = *sub.get_one("n").unwrap();
            let m: usize = *sub.get_one("m").unwrap();
            main_rainbow_load(n, m, "./db");
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
                // Fallback when file is empty
                let c1= random_canonical_id(&conn, 5).unwrap();
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
                .set_max_dbs(33)      
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
                .set_max_dbs(33)      
                .set_map_size(700 * 1024 * 1024 * 1024) 
                .open(Path::new(lmdb))
                .expect("Failed to open lmdb");

            if data.trim().is_empty() {
                println!("Generating random");
                let c1 = random_circuit(n as u8, 30);
                println!("Starting Len: {}", c1.gates.len());
                if bookendless {
                    main_butterfly_big_bookendsless(&c1, rounds, &mut conn, n, true, path, &env);
                } else {
                    main_butterfly_big(&c1, rounds, &mut conn, n, true, path, &env);
                }
            } else {
                let c = CircuitSeq::from_string(&data);
                if bookendless {
                    main_butterfly_big_bookendsless(&c, rounds, &mut conn, n, true, path, &env);
                } else {
                    main_butterfly_big(&c, rounds, &mut conn, n, true, path, &env);
                }
            }
        }
        Some(("heatmap", sub)) => {
            let num_inputs: usize = *sub.get_one("inputs").unwrap();
            let n: usize = *sub.get_one("num_wires").unwrap();

            let xlabel = sub
                .get_one::<String>("xlabel")
                .map(|s| s.as_str())
                .unwrap_or("Circuit 1 gate index");
            let ylabel = sub
                .get_one::<String>("ylabel")
                .map(|s| s.as_str())
                .unwrap_or("Circuit 2 gate index");

            let flag = sub.get_flag("std"); // true if -s was given

            println!(
                "Running distinguisher with {} inputs...",
                num_inputs
            );
            // heatmap(n, num_inputs, flag);
        }
        Some(("reverse", sub)) => {
            let from_path = sub.get_one::<String>("source").unwrap();
            let dest_path = sub.get_one::<String>("dest").unwrap();
            reverse(from_path, dest_path);
        }
        Some(("compress", sub)) => {
            let p: &String = sub.get_one("p").expect("Missing -p <path>");
            let n: usize = *sub.get_one("n").expect("Missing -n <wires>");

            let contents = fs::read_to_string(p)
                .unwrap_or_else(|_| panic!("Failed to read circuit file at {}", p));

            let mut acc = CircuitSeq::from_string(&contents);

            let mut conn = Connection::open_with_flags("./circuits.db",rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,)
            .expect("Failed to open ./circuits.db in read-only mode");
            let lmdb = "./db";
            let _ = std::fs::create_dir_all(lmdb);

            let env = Environment::new()
                .set_max_dbs(33)      
                .set_map_size(700 * 1024 * 1024 * 1024) 
                .open(Path::new(lmdb))
                .expect("Failed to open lmdb");
            // Call compression logic
            let mut stable_count = 0;
            while stable_count < 3 {
                let before = acc.gates.len();
                acc = compress_big_ancillas(&acc, 1_000, n, &mut conn, &env);
                let after = acc.gates.len();

                if after == before {
                    stable_count += 1;
                    println!("  Final compression stable {}/3 at {} gates", stable_count, after);
                } else {
                    println!("  Final compression reduced: {} → {} gates", before, after);
                    stable_count = 0;
                }
            }
            let mut file = fs::File::create("compressed.txt")
                .expect("Failed to create compressed.txt");
            write!(file, "{}", acc.repr())
                .expect("Failed to write compressed circuit to file");

            println!("Compressed circuit written to compressed.txt");
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
            let _ = sql_to_lmdb(n, m);
        }
        _ => unreachable!(),
    }
}

pub fn heatmap(circuit_one: &CircuitSeq, circuit_two: &CircuitSeq, num_wires: usize, num_inputs: usize, flag: bool) -> f64 {
    let mut circuit_one = circuit_one.clone();
    let mut circuit_two = circuit_two.clone();
    if flag {
        circuit_one.canonicalize();
        circuit_two.canonicalize();
    }
    let circuit_one_len = circuit_one.gates.len();
    let circuit_two_len = circuit_two.gates.len();

    let mut average = vec![[0f64, 0f64, 0f64]; (circuit_one_len + 1) * (circuit_two_len + 1)];
    let mut rng = rand::rng();
    let start_time = Instant::now();

    for _ in 0..num_inputs {
        let input_bits: usize = if num_wires < usize::BITS as usize {
            rng.random_range(0..(1usize << num_wires))
        } else {
            rng.random_range(0..=usize::MAX)
        };

        let evolution_one = circuit_one.evaluate_evolution(input_bits);
        let evolution_two = circuit_two.evaluate_evolution(input_bits);
        for i1 in 0..=circuit_one_len {
            for i2 in 0..=circuit_two_len {
                let diff = evolution_one[i1] ^ evolution_two[i2];
                let hamming_dist = diff.count_ones() as f64;
                let overlap = (2.0 * hamming_dist / num_wires as f64) - 1.0;
                let abs_overlap = overlap.abs();

                let index = i1 * (circuit_two_len + 1) + i2;
                average[index][2] += abs_overlap / num_inputs as f64;
            }
        }
    }

    println!("Time elapsed: {:?}", Instant::now() - start_time);

    let total_points = average.len();
    let mean_all: f64 = average.iter().map(|p| p[2]).sum::<f64>() / total_points as f64;

    mean_all
}

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

use lmdb::{Environment, Database, WriteFlags, Transaction};
use local_mixing::circuit::Permutation;
use std::collections::HashMap;

pub fn sql_to_lmdb(
    n: usize,
    m: usize,
) -> Result<(), ()> {

    let sqlite_path = "circuits.db";
    let lmdb_path = "./db";
    let map_size_bytes: usize = 800usize * 1024 * 1024 * 1024;
    let batch_max_entries: usize = 100000;
    
    let conn = Connection::open(sqlite_path).expect("Failed to open sqlite database");
    let table = format!("n{}m{}", n, m);

    let query = format!("SELECT * FROM {}", table);
    let mut stmt = conn.prepare(&query).expect("Failed to prepare SQLite query");
    let mut rows = stmt.query([]).expect("Failed to query SQLite rows");

    fs::create_dir_all(lmdb_path).expect("Failed to create LMDB directory");
    let env = Environment::new()
        .set_max_dbs(33)
        .set_map_size(map_size_bytes)
        .open(Path::new(lmdb_path))
        .expect("Failed to open LMDB environment");

    let db = env.create_db(Some(&table), lmdb::DatabaseFlags::empty())
        .expect("Failed to create LMDB database");

    // track count per perm
    let mut perm_counts: HashMap<Vec<u8>, u32> = HashMap::new();

    let mut batch: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(batch_max_entries);
    let mut rows_processed: u64 = 0;

    let flush = |env: &Environment, db: Database, batch: &mut Vec<(Vec<u8>, Vec<u8>)>| {
        if batch.is_empty() { return; }
        let mut txn = env.begin_rw_txn().expect("Failed to begin LMDB RW transaction");
        for (key, value) in batch.iter() {
            txn.put(db, key, value, WriteFlags::empty()).expect("Failed to write LMDB entry");
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
            continue;
        }

        // get count for this perm
        let count = perm_counts.entry(perm.clone()).or_insert(0);
        let key = [perm.clone(), count.to_le_bytes().to_vec()].concat();
        *count += 1;

        // serialize value (circuit + shuf)
        let value = bincode::serialize(&(circuit, shuf)).expect("Failed to serialize value");

        batch.push((key, value));

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
