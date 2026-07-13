use std::fs;
use std::io::Write;
use std::path::Path;

use lmdb::{Environment, EnvironmentFlags};

use local_mixing::circuit::CircuitSeq;
use local_mixing::replace::main_mix::open_all_dbs;
use local_mixing::replace::replace::{compress_loop, print_compress_timers};

/// Run compression on a circuit file against the sharded LMDB.
pub fn run(sub: &clap::ArgMatches) {
    let s: &String = sub.get_one("s").expect("Missing -s <source>");
    let n: usize = *sub.get_one("n").expect("Missing -n <wires>");
    let d: &String = sub.get_one("d").expect("Missing -d <destination>");
    let _seq = sub.get_flag("seq");
    let stable_compressions: usize = *sub
        .get_one("stable_compressions")
        .expect("Missing --stable_compressions");

    let contents =
        fs::read_to_string(s).unwrap_or_else(|_| panic!("Failed to read circuit file at {}", s));
    let mut acc = CircuitSeq::from_string(&contents);

    let lmdb_path = "./db";
    let _ = std::fs::create_dir_all(lmdb_path);
    // Frozen-table mode: see sss.rs — ./db becomes a stub for plumbing only.
    if std::env::var("FROZEN_DB_DIR").is_ok() {
        local_mixing::replace::frozen::ensure_stub_lmdb(lmdb_path);
    }
    // NO_READAHEAD: random point lookups into a DB larger than RAM; see sss.rs.
    // LMDB_READAHEAD=1 restores kernel readahead.
    let mut env_flags = EnvironmentFlags::READ_ONLY | EnvironmentFlags::NO_LOCK;
    if std::env::var("LMDB_READAHEAD").map(|v| v == "1") != Ok(true) {
        env_flags |= EnvironmentFlags::NO_READAHEAD;
    }
    let env = Environment::new()
        .set_flags(env_flags)
        .set_max_dbs(556)
        .set_max_readers(10000)
        .set_map_size(800 * 1024 * 1024 * 1024)
        .open(Path::new(lmdb_path))
        .expect("Failed to open lmdb");

    // Print timers on Ctrl+C.
    ctrlc::set_handler(|| {
        print_compress_timers();
        std::process::exit(0);
    })
    .expect("Failed to set Ctrl+C handler");

    println!("Starting compression");
    let (shard_dbs, _curated_shard_dbs) = open_all_dbs(&env);
    acc = compress_loop(
        &acc,
        n,
        &env,
        &shard_dbs,
        stable_compressions,
        1,
        1,
        d,
        &mut Vec::new(),
    );
    print_compress_timers();

    let mut file = fs::File::create(d).expect("Failed to create new file");
    write!(file, "{}", acc.repr()).expect("Failed to write compressed circuit to file");
    println!("Compressed circuit written to {}", d);
}
