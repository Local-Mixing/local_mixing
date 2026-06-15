use local_mixing::circuit::CircuitSeq;
use local_mixing::replace::gadgets::{FeistalOptions, feistalize_with_options};
use std::{env, fs::File, io::Write};

fn usage(program: &str) -> ! {
    eprintln!(
        "usage: {program} <source.txt> <dest.txt> [n=128] [rg_freq=2] [--sg-masked] [--rg-refresh] [--slice-scramble-rounds N]"
    );
    std::process::exit(2);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        usage(&args[0]);
    }
    let mut pos = 3;
    let n: usize = if args.get(pos).is_some_and(|s| !s.starts_with("--")) {
        let value = args[pos].parse().unwrap_or_else(|_| usage(&args[0]));
        pos += 1;
        value
    } else {
        128
    };
    let rg_freq: usize = if args.get(pos).is_some_and(|s| !s.starts_with("--")) {
        let value = args[pos].parse().unwrap_or_else(|_| usage(&args[0]));
        pos += 1;
        value
    } else {
        2
    };
    let mut options = FeistalOptions::default();
    while pos < args.len() {
        match args[pos].as_str() {
            "--sg-masked" => options.masked_sg = true,
            "--rg-refresh" => options.rg_null_refresh = true,
            "--slice-scramble-rounds" => {
                pos += 1;
                options.slice_scramble_rounds = args
                    .get(pos)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| usage(&args[0]));
            }
            _ => usage(&args[0]),
        }
        pos += 1;
    }
    let text = std::fs::read_to_string(&args[1])
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", args[1]));
    let circuit = CircuitSeq::from_string(&text);
    let mut rng = rand::rng();
    let transformed = feistalize_with_options(&circuit, n, rg_freq, options, &mut rng);
    let mut out =
        File::create(&args[2]).unwrap_or_else(|e| panic!("failed to create {}: {e}", args[2]));
    write!(out, "{}", transformed.repr()).expect("failed to write feistalized circuit");
    println!("source_gates {}", circuit.gates.len());
    println!("dest_gates {}", transformed.gates.len());
    println!("dest_wires {}", 3 * n);
    println!("rg_freq {}", rg_freq);
    println!("masked_sg {}", options.masked_sg);
    println!("rg_null_refresh {}", options.rg_null_refresh);
    println!("slice_scramble_rounds {}", options.slice_scramble_rounds);
    println!("path {}", args[2]);
}
