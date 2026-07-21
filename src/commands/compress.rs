use std::fs;
use std::io::Write;

use local_mixing::circuit::CircuitSeq;
use local_mixing::replace::frozen::FrozenDb;
use local_mixing::replace::replace::{compress_loop, print_compress_timers};

/// Run compression on a circuit file against the frozen replacement store.
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

    let db = FrozenDb::from_env();

    // Print timers on Ctrl+C.
    ctrlc::set_handler(|| {
        print_compress_timers();
        std::process::exit(0);
    })
    .expect("Failed to set Ctrl+C handler");

    println!("Starting compression");
    acc = compress_loop(
        &acc,
        n,
        &db,
        stable_compressions,
        1,
        1,
        d,
        None,
        &mut Vec::new(),
    );
    print_compress_timers();

    let mut file = fs::File::create(d).expect("Failed to create new file");
    write!(file, "{}", acc.repr()).expect("Failed to write compressed circuit to file");
    println!("Compressed circuit written to {}", d);
}
