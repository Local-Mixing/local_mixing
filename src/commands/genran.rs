use std::fs;
use std::io::Write;

use local_mixing::circuit::random_circuit;

/// Generate a random circuit with `n` wires and `m` gates.
pub fn run(sub: &clap::ArgMatches) {
    let d: &String = sub.get_one("d").expect("Missing -d <path>");
    let n: usize = *sub.get_one("n").expect("Missing -n <wires>");
    let m: usize = *sub.get_one("m").expect("Missing -m <gates>");

    let circuit = random_circuit(n, m);
    let mut file = fs::File::create(d).expect("Failed to create new file");
    write!(file, "{}", circuit.repr()).expect("Failed to write random circuit to file");
}
