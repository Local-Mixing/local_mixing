use std::fs;
use std::io::Write;

use local_mixing::circuit::CircuitSeq;
use local_mixing::db_mixing::frozen::FrozenDb;
use local_mixing::db_mixing::transpositions::{
    insert_wire_shuffles_knuth, insert_wire_shuffles_simple, insert_wire_shuffles_x,
};

/// Insert wire shuffles (SAMFs) into a circuit.
pub fn run(sub: &clap::ArgMatches) {
    let from_path = sub.get_one::<String>("s").unwrap();
    let dest_path = sub.get_one::<String>("d").unwrap();
    let n: usize = *sub.get_one("n").expect("Missing -n <wires>");
    let i: usize = *sub.get_one("i").expect("Missing -i <insertions>");
    let knuth = sub.get_flag("knuth");

    // Shuffle insertion reads FROZEN_DB_DIR / FROZEN_CURATED_DIR.
    let db = FrozenDb::from_env();

    let contents = fs::read_to_string(from_path)
        .unwrap_or_else(|_| panic!("Failed to read circuit file at {}", from_path));

    let mut c = CircuitSeq::from_string(&contents);
    println!("Creating shuffled circuit");
    if i == 0 {
        if knuth {
            insert_wire_shuffles_knuth(&mut c, n, &db, true, true);
        } else {
            insert_wire_shuffles_simple(&mut c, n, &db, true, true);
        }
    } else {
        insert_wire_shuffles_x(&mut c, n, i, &db, true, true);
    }

    let mut file = fs::File::create(dest_path).expect("Failed to create new file");
    write!(file, "{}", c.repr()).expect("Failed to write shuffled circuit to file");
    println!("Shuffled circuit written to {}", dest_path);
}
