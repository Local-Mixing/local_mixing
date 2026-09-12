//! Compare complete circuit functions on sampled inputs.
use super::shared::{format_bits, parse_wires};
use clap::{Arg, ArgMatches, Command};
use local_mixing::circuit::operations::CircuitSource;

pub fn command() -> Command {
    Command::new("compare")
        .about(
            "Compare complete circuit functions on random inputs; a mismatch exits with status 1",
        )
        .arg(
            Arg::new("wires")
                .short('n')
                .long("wires")
                .required(true)
                .value_parser(parse_wires)
                .help("Number of wires"),
        )
        .arg(
            Arg::new("iterations")
                .short('i')
                .long("iterations")
                .required(true)
                .value_parser(clap::value_parser!(usize))
                .help("Number of test iterations"),
        )
        .arg(
            Arg::new("circuit_a")
                .short('a')
                .long("circuit-a")
                .required(true)
                .value_parser(clap::value_parser!(String))
                .help("Path to first circuit file"),
        )
        .arg(
            Arg::new("circuit_b")
                .short('b')
                .long("circuit-b")
                .required(true)
                .value_parser(clap::value_parser!(String))
                .help("Path to second circuit file"),
        )
}

pub fn run(sub: &ArgMatches) -> Result<(), String> {
    let a = sub.get_one::<String>("circuit_a").unwrap();
    let b = sub.get_one::<String>("circuit_b").unwrap();
    let n = *sub.get_one::<usize>("wires").unwrap();
    let iterations = *sub.get_one::<usize>("iterations").unwrap();
    if !(1..=1024).contains(&n) {
        return Err("wires must be in 1..=1024".into());
    }
    if iterations == 0 {
        return Err("comparison requires at least one iteration".into());
    }
    let left = CircuitSource::read(a)?;
    let right = CircuitSource::read(b)?;
    let comparison = left.compare_sampled(&right, n, iterations, &mut rand::rng())?;
    let width = comparison.wires;
    if let Some(counterexample) = comparison.counterexample {
        return Err(format!(
            "circuits differ on sample {}: input {}",
            counterexample.sample,
            format_bits(&counterexample.input, width)
        ));
    }
    println!(
        "No mismatch in {iterations} random inputs over {width} wires (sampled comparison, not an exhaustive proof)."
    );
    Ok(())
}
