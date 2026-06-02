use std::fs;
use std::io::Write;

use local_mixing::circuit::CircuitSeq;
use local_mixing::random::random_data::shoot_random_gate;

/// Shoot random gates through a circuit `i` times.
pub fn run(sub: &clap::ArgMatches) {
    let from_path = sub.get_one::<String>("s").unwrap();
    let dest_path = sub.get_one::<String>("d").unwrap();
    let i: usize = *sub.get_one("i").expect("Missing -i <iterations>");

    let contents = fs::read_to_string(from_path)
        .unwrap_or_else(|_| panic!("Failed to read circuit file at {}", from_path));

    let mut c = CircuitSeq::from_string(&contents);
    println!("Creating shot circuit");
    shoot_random_gate(&mut c, i);

    let mut file = fs::File::create(dest_path).expect("Failed to create new file");
    write!(file, "{}", c.repr()).expect("Failed to write shot circuit to file");
    println!("Shot circuit written to {}", dest_path);
}
