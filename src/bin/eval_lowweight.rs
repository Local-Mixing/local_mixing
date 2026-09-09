//! Tiny evaluator: print weight for --x-lo / --x-hi on a challenge.
use clap::Parser;
use local_mixing::sandwich::{hamming_weight, ChunkedCircuit};
use std::fs;
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "challenges/lowweight128_obf.txt")]
    challenge: PathBuf,
    #[arg(long)]
    x_lo: String,
    #[arg(long, default_value = "0")]
    x_hi: String,
}

fn main() {
    let args = Args::parse();
    let s = fs::read_to_string(&args.challenge).unwrap();
    let mut prefix = 64usize;
    let mut body = String::new();
    for line in s.lines() {
        if let Some(r) = line.trim().strip_prefix("challenge_prefix_zeros ") {
            prefix = r.parse().unwrap();
            continue;
        }
        if line.trim().starts_with("challenge_seed ") {
            continue;
        }
        body.push_str(line);
        body.push('\n');
    }
    let circ = ChunkedCircuit::from_readable_string(&body).unwrap();
    let free_n = circ.data_wires - prefix;
    let lo = u64::from_str_radix(args.x_lo.trim_start_matches("0x"), 16).unwrap();
    let hi = u64::from_str_radix(args.x_hi.trim_start_matches("0x"), 16).unwrap();
    let mut input = vec![false; circ.data_wires];
    for i in 0..free_n.min(64) {
        input[prefix + i] = ((lo >> i) & 1) != 0;
    }
    for i in 64..free_n {
        input[prefix + i] = ((hi >> (i - 64)) & 1) != 0;
    }
    let wt = hamming_weight(&circ.evaluate_data_bits(&input));
    println!("{wt}");
}
