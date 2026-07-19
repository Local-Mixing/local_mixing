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
    // Optional early-stop: stop compressing once the circuit reaches this fraction of its initial
    // size (in addition to the usual no-progress stop). Absent => compress to convergence.
    let target_fraction: Option<f64> = sub.get_one::<f64>("target_fraction").copied();

    let contents =
        fs::read_to_string(s).unwrap_or_else(|_| panic!("Failed to read circuit file at {}", s));
    let mut acc = CircuitSeq::from_string(&contents);

    let initial_len = acc.gates.len();
    let early_stop_target: Option<usize> = target_fraction.map(|f| {
        let f = f.clamp(0.0, 1.0);
        let t = (initial_len as f64 * f).floor() as usize;
        println!(
            "Compression early-stop target: {:.1}% of {} = {} gates",
            f * 100.0,
            initial_len,
            t
        );
        t
    });

    // Compression reads the regular frozen store (FROZEN_DB_DIR).
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
        12,
        1,
        1,
        d,
        false,
        early_stop_target,
        // The standalone compress command is always a delivery-strength ("final") compression.
        true,
        &mut Vec::new(),
    );
    print_compress_timers();

    let mut file = fs::File::create(d).expect("Failed to create new file");
    write!(file, "{}", acc.repr()).expect("Failed to write compressed circuit to file");
    println!("Compressed circuit written to {}", d);
}
