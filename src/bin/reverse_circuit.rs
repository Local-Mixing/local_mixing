use local_mixing::circuit::CircuitSeq;
use std::{env, fs::File, io::Write};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: reverse_circuit <input.txt> <output.txt>");
        std::process::exit(2);
    }
    let text = std::fs::read_to_string(&args[1]).unwrap_or_else(|e| panic!("failed to read {}: {e}", args[1]));
    let mut circuit = CircuitSeq::from_string(&text);
    circuit.gates.reverse();
    let mut out = File::create(&args[2]).unwrap_or_else(|e| panic!("failed to create {}: {e}", args[2]));
    write!(out, "{}", circuit.repr()).expect("failed to write reversed circuit");
    println!("input_gates {}", circuit.gates.len());
    println!("path {}", args[2]);
}
