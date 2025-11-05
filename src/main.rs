use clap::{Arg, ArgAction, Command};
use colorgrad::{CustomGradient, Gradient};
use itertools::Itertools;
use plotters::prelude::*;
use rand::{rngs::OsRng, Rng, TryRngCore};
use rusqlite::{Connection, OpenFlags};
use serde_json::json;
use std::{
    fs::{self, File},
    io::Write,
    path::Path,
    time::Instant,
    process::Stdio,
};
use local_mixing::replace::replace::expand_big;
use local_mixing::{
    circuit::CircuitSeq,
    rainbow::{
        explore::explore_db,
        rainbow::{main_rainbow_load},
    },
    random::random_data::{build_from_sql, main_random, random_circuit},
    replace::{
        mixing::{main_butterfly, main_butterfly_big, main_mix},
        replace::{compress, random_canonical_id, random_id},
    },
};
use local_mixing::replace::replace::compress_big;
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
                    Arg::new("r")
                        .short('r')
                        .long("trials")
                        .required(true)
                        .value_parser(clap::value_parser!(usize))
                        .help("Number of compression trials"),
                )
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
        .get_matches();

    match matches.subcommand() {
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

            if data.trim().is_empty() {
                println!("Generating random");
                let c1 = random_circuit(n as u8, 30);
                println!("Starting Len: {}", c1.gates.len());
                main_butterfly_big(&c1, rounds, &mut conn, n, false, path);
            } else {
                let c = CircuitSeq::from_string(&data);
                main_butterfly_big(&c, rounds, &mut conn, n, false, path);
            }
        }

        Some(("abbutterfly", sub)) => {
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

            if data.trim().is_empty() {
                println!("Generating random");
                let c1 = random_circuit(n as u8, 30);
                println!("Starting Len: {}", c1.gates.len());
                main_butterfly_big(&c1, rounds, &mut conn, n, true, path);
            } else {
                let c = CircuitSeq::from_string(&data);
                main_butterfly_big(&c, rounds, &mut conn, n, true, path);
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
            heatmap(n, num_inputs, xlabel, ylabel, flag);
        }
        Some(("reverse", sub)) => {
            let from_path = sub.get_one::<String>("source").unwrap();
            let dest_path = sub.get_one::<String>("dest").unwrap();
            reverse(from_path, dest_path);
        }
        Some(("compress", sub)) => {
            let r: usize = *sub.get_one("r").expect("Missing -r <trials>");
            let p: &String = sub.get_one("p").expect("Missing -p <path>");
            let n: usize = *sub.get_one("n").expect("Missing -n <wires>");

            let contents = fs::read_to_string(p)
                .unwrap_or_else(|_| panic!("Failed to read circuit file at {}", p));

            let circuit = CircuitSeq::from_string(&contents);

            let mut conn = Connection::open_with_flags("./circuits.db",rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,)
            .expect("Failed to open ./circuits.db in read-only mode");

            // Call your compression logic
            let compressed = compress_big(&expand_big(&circuit, r/100, n, &mut conn), r, n, &mut conn);

            let mut file = fs::File::create("compressed.txt")
                .expect("Failed to create compressed.txt");
            write!(file, "{}", compressed.repr())
                .expect("Failed to write compressed circuit to file");

            println!("Compressed circuit written to compressed.txt");
        }
        _ => unreachable!(),
    }
}


pub fn heatmap(num_wires: usize, num_inputs: usize, xlabel: &str, ylabel: &str, flag: bool) {
    // Load circuits
    let contents = fs::read_to_string("butterfly_recent.txt")
        .expect("Failed to read butterfly_recent.txt");

    let (circuit_one_str, circuit_two_str) = contents
        .split_once(':')
        .expect("Invalid format in butterfly_recent.txt");

    let mut circuit_one = CircuitSeq::from_string(circuit_one_str);
    let mut circuit_two = CircuitSeq::from_string(circuit_two_str);
    circuit_one.canonicalize();
    circuit_two.canonicalize();
    let circuit_one_len = circuit_one.gates.len();
    let circuit_two_len = circuit_two.gates.len();

    let mut average = vec![[0f64, 0f64, 0f64]; (circuit_one_len + 1) * (circuit_two_len + 1)];
    let mut rng = rand::rng();
    let start_time = Instant::now();

    for i in 0..num_inputs {
        if i % 10 == 0 {
            println!("{}/{}", i, num_inputs);
        }

        let input_bits: usize = rng.random_range(0..(1 << num_wires));
        let evolution_one = circuit_one.evaluate_evolution(input_bits);
        let evolution_two = circuit_two.evaluate_evolution(input_bits);
        if !flag {
            for i1 in 0..=circuit_one_len {
                for i2 in 0..=circuit_two_len {
                    let diff = evolution_one[i1] ^ evolution_two[i2];
                    let hamming_dist = diff.count_ones() as f64;
                    let overlap = hamming_dist / num_wires as f64;

                    let index = i1 * (circuit_two_len + 1) + i2;
                    average[index][0] = i1 as f64;
                    average[index][1] = i2 as f64;
                    average[index][2] += overlap / num_inputs as f64;
                }
            }
        } else {
            for i1 in 0..=circuit_one_len {
                for i2 in 0..=circuit_two_len {
                    let diff = evolution_one[i1] ^ evolution_two[i2];
                    let hamming_dist = diff.count_ones() as f64;
                    let overlap = (2.0 * hamming_dist / num_wires as f64) - 1.0;
                    let abs_overlap = overlap.abs();

                    let index = i1 * (circuit_two_len + 1) + i2;
                    average[index][0] = i1 as f64;
                    average[index][1] = i2 as f64;
                    average[index][2] += abs_overlap / num_inputs as f64;
                }
            }
        }
    }

    println!("Time elapsed: {:?}", Instant::now() - start_time);

    // Prepare JSON for Python
    let data = json!({
        "results": average,
        "xlabel": xlabel,
        "ylabel": ylabel
    });

    // Choose Python script based on flag
    let script_path = if flag {
        "./heatmap/heatmap.py"       // std dev version
    } else {
        "./heatmap/heatmap_raw.py"   // raw values version
    };

    let mut child = std::process::Command::new("python3")
        .arg(script_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("Failed to spawn Python script");

    child.stdin.as_mut()
        .expect("Failed to open stdin")
        .write_all(data.to_string().as_bytes())
        .expect("Failed to write JSON to Python stdin");

    let status = child.wait().expect("Failed to wait on Python process");
    if !status.success() {
        eprintln!("Python script failed with status {:?}", status.code());
    }
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