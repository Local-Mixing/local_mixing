//! Evaluate a circuit on an explicit or random input.
use super::shared::{format_bits, parse_wires};
use clap::{Arg, ArgGroup, ArgMatches, Command};
use local_mixing::circuit::U1024;
use local_mixing::circuit::operations::{CircuitSource, CircuitState, mask_state, random_state};

pub fn command() -> Command {
    Command::new("evaluate")
        .group(
            ArgGroup::new("input_choice")
                .args(["input", "random"])
                .required(true)
                .multiple(false),
        )
        .about("Evaluate a circuit on an input and print the output")
        .arg(
            Arg::new("source")
                .short('s')
                .long("source")
                .required(true)
                .value_parser(clap::value_parser!(String))
                .help("Path to the circuit file"),
        )
        .arg(
            Arg::new("n")
                .short('n')
                .long("wires")
                .visible_alias("n")
                .required(true)
                .value_parser(parse_wires)
                .help("Number of wires"),
        )
        .arg(
            Arg::new("input")
                .short('x')
                .long("input")
                .required(false)
                .value_parser(clap::value_parser!(String))
                .help("Input value (decimal or 0x-prefixed hex)"),
        )
        .arg(
            Arg::new("random")
                .short('r')
                .long("random")
                .required(false)
                .action(clap::ArgAction::SetTrue)
                .help("Use a random input (prints the chosen input)"),
        )
}

pub fn run(sub: &ArgMatches) -> Result<(), String> {
    let path = sub.get_one::<String>("source").unwrap();
    let n = *sub.get_one::<usize>("n").unwrap();
    if !(1..=1024).contains(&n) {
        return Err("wires must be in 1..=1024".into());
    }
    let explicit = sub.get_one::<String>("input").is_some();
    if explicit == sub.get_flag("random") {
        return Err("choose exactly one of --input and --random".into());
    }
    let source = CircuitSource::read(path)?;
    let input = read_input(sub, n)?;
    let mut state = input;
    source.evaluate(&mut state);
    println!("n: {n}");
    println!("Input:  {}", format_bits(&input, n));
    println!("Output: {}", format_bits(&state, n));
    Ok(())
}

fn read_input(sub: &clap::ArgMatches, n: usize) -> Result<CircuitState, String> {
    if sub.get_flag("random") {
        return Ok(random_state(n, &mut rand::rng()));
    }
    let raw = sub
        .get_one::<String>("input")
        .expect("-x required when not using -r");
    let mut state = parse_u1024(raw)?.0;
    mask_state(&mut state, n);
    Ok(state)
}

fn parse_u1024(s: &str) -> Result<U1024, String> {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        U1024::from_str_radix(hex, 16).map_err(|e| format!("invalid hexadecimal input: {e}"))
    } else {
        U1024::from_dec_str(s).map_err(|e| format!("invalid decimal input: {e}"))
    }
}
