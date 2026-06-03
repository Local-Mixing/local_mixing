use std::fs;

use local_mixing::circuit::CircuitSeq;

/// Check whether two circuit files are functionally equivalent (probabilistic).
pub fn run(sub: &clap::ArgMatches) {
    let c1_path = sub.get_one::<String>("circuit_a").unwrap();
    let c2_path = sub.get_one::<String>("circuit_b").unwrap();
    let i: usize = *sub.get_one::<usize>("iterations").expect("Missing --iterations");
    let n: usize = *sub.get_one::<usize>("wires").expect("Missing --wires");

    let contents1 = fs::read_to_string(c1_path)
        .unwrap_or_else(|_| panic!("Failed to read circuit file at {}", c1_path));
    let contents2 = fs::read_to_string(c2_path)
        .unwrap_or_else(|_| panic!("Failed to read circuit file at {}", c2_path));

    let c1 = CircuitSeq::from_string(&contents1);
    let c2 = CircuitSeq::from_string(&contents2);

    println!("Checking for equivalence between {} and {}", c1_path, c2_path);
    println!("{}: {} gates", c1_path, c1.gates.len());
    println!("{}: {} gates", c2_path, c2.gates.len());

    if c1.probably_equal(&c2, n, i).is_ok() {
        println!("Circuits are equal!");
    } else {
        println!("Circuits are not equal");
    }
}
