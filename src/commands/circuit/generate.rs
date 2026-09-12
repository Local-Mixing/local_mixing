//! Generate a random G57 circuit.
use super::shared::parse_wires;
use clap::{Arg, ArgMatches, Command};
use local_mixing::circuit::random_circuit;
use std::fs;

pub fn command() -> Command {
    Command::new("generate")
        .about("Generate a random circuit with n wires and m gates")
        .arg(
            Arg::new("d")
                .short('d')
                .long("destination")
                .required(true)
                .value_parser(clap::value_parser!(String))
                .help("Path to the new circuit file"),
        )
        .arg(
            Arg::new("n")
                .short('n')
                .long("wires")
                .required(true)
                .value_parser(parse_wires)
                .help("Number of wires in the circuit"),
        )
        .arg(
            Arg::new("m")
                .short('m')
                .long("gates")
                .required(true)
                .value_parser(clap::value_parser!(usize))
                .help("Number of gates in the circuit"),
        )
}

pub fn run(sub: &ArgMatches) -> Result<(), String> {
    let destination = sub.get_one::<String>("d").unwrap();
    let n = *sub.get_one::<usize>("n").unwrap();
    let m = *sub.get_one::<usize>("m").unwrap();
    if !(3..=1024).contains(&n) {
        return Err("random G57 generation requires 3..=1024 wires".into());
    }
    let circuit = random_circuit(n, m);
    fs::write(destination, circuit.repr()).map_err(|e| format!("cannot write {destination}: {e}"))
}
