//! Attack the sealed low-weight challenge: minimize wt(C(0^{prefix} || x)).
//!
//! This binary only loads the obfuscated representation — no plaintext.

use clap::{Parser, ValueEnum};
use local_mixing::sandwich::{hamming_weight, ChunkedCircuit};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Clone, Debug, ValueEnum)]
enum Mode {
    /// Uniform random free inputs
    Random,
    /// Random-restart first-improvement hill climb (1-bit / 2-bit flips)
    Hill,
    /// Simulated annealing on free bits
    Anneal,
    /// Exhaustive k-flip neighborhood around --start-x
    Polish,
}

#[derive(Parser, Debug)]
#[command(about = "Search for low Hamming-weight outputs under a fixed zero prefix")]
struct Args {
    #[arg(long, default_value = "challenges/lowweight_obf.txt")]
    challenge: PathBuf,

    #[arg(long, default_value = "hill")]
    mode: Mode,

    #[arg(long, default_value_t = 10_000)]
    samples: usize,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Print progress every this many evaluations
    #[arg(long, default_value_t = 500)]
    report_every: usize,

    /// Starting free input for polish mode (hex, e.g. e7603c21)
    #[arg(long, default_value = "0")]
    start_x: String,

    /// Max flip radius for polish (1..=3)
    #[arg(long, default_value_t = 3)]
    polish_k: usize,
}

struct Challenge {
    prefix_zeros: usize,
    circuit: ChunkedCircuit,
}

fn load_challenge(path: &PathBuf) -> Challenge {
    let s = fs::read_to_string(path).expect("read challenge");
    let mut prefix_zeros = 32usize;
    let mut circuit_text = String::new();
    for line in s.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("challenge_prefix_zeros ") {
            prefix_zeros = rest.trim().parse().unwrap();
            continue;
        }
        if t.starts_with("challenge_seed ") {
            continue;
        }
        circuit_text.push_str(line);
        circuit_text.push('\n');
    }
    let circuit = ChunkedCircuit::from_readable_string(&circuit_text).expect("parse circuit");
    Challenge {
        prefix_zeros,
        circuit,
    }
}

fn eval_wt(ch: &Challenge, free: u64) -> usize {
    let y = ch.circuit.evaluate_prefixed_zeros(ch.prefix_zeros, free);
    hamming_weight(&y)
}

fn main() {
    let args = Args::parse();
    let ch = load_challenge(&args.challenge);
    let free_bits = ch.circuit.data_wires - ch.prefix_zeros;
    assert!(free_bits <= 64);
    let free_mask = if free_bits == 64 {
        u64::MAX
    } else {
        (1u64 << free_bits) - 1
    };

    println!(
        "loaded challenge: data_wires={} total_wires={} chunks={} prefix_zeros={} free_bits={}",
        ch.circuit.data_wires,
        ch.circuit.total_wires,
        ch.circuit.chunks.len(),
        ch.prefix_zeros,
        free_bits
    );
    println!("mode={:?} samples={} seed={}", args.mode, args.samples, args.seed);

    let mut rng = StdRng::seed_from_u64(args.seed);
    let t0 = Instant::now();
    let mut evals = 0usize;
    let mut best_x = 0u64;
    let mut best_wt = usize::MAX;

    let mut consider = |x: u64, evals: &mut usize, best_x: &mut u64, best_wt: &mut usize| {
        let x = x & free_mask;
        let wt = eval_wt(&ch, x);
        *evals += 1;
        if wt < *best_wt {
            *best_wt = wt;
            *best_x = x;
            println!(
                "  new best wt={best_wt}  x=0x{best_x:08x}  after {evals} evals  ({:.1}s)",
                t0.elapsed().as_secs_f64()
            );
        }
        if args.report_every > 0 && *evals % args.report_every == 0 {
            println!(
                "  progress evals={evals} best_wt={best_wt} x=0x{best_x:08x} ({:.1}s)",
                t0.elapsed().as_secs_f64()
            );
        }
        wt
    };

    match args.mode {
        Mode::Random => {
            for _ in 0..args.samples {
                let x = rng.random::<u64>() & free_mask;
                consider(x, &mut evals, &mut best_x, &mut best_wt);
            }
        }
        Mode::Hill => {
            let restarts = (args.samples / 400).max(1);
            let budget_per = (args.samples / restarts).max(50);
            for r in 0..restarts {
                let mut x = rng.random::<u64>() & free_mask;
                let mut cur = consider(x, &mut evals, &mut best_x, &mut best_wt);
                // Phase 1: random 1-/2-bit proposals
                for _ in 0..(budget_per / 2) {
                    if evals >= args.samples {
                        break;
                    }
                    let mut y = x;
                    let i = rng.random_range(0..free_bits);
                    y ^= 1u64 << i;
                    if rng.random_bool(0.3) {
                        let j = rng.random_range(0..free_bits);
                        y ^= 1u64 << j;
                    }
                    let wt = consider(y, &mut evals, &mut best_x, &mut best_wt);
                    if wt <= cur {
                        x = y;
                        cur = wt;
                    }
                }
                // Phase 2: deterministic coordinate descent to a 1-flip local min
                loop {
                    if evals >= args.samples {
                        break;
                    }
                    let mut improved = false;
                    for i in 0..free_bits {
                        if evals >= args.samples {
                            break;
                        }
                        let y = x ^ (1u64 << i);
                        let wt = consider(y, &mut evals, &mut best_x, &mut best_wt);
                        if wt < cur {
                            x = y;
                            cur = wt;
                            improved = true;
                        }
                    }
                    if !improved {
                        break;
                    }
                }
                if r % 10 == 0 {
                    println!(
                        "  restart {r}/{restarts} best_wt={best_wt} ({:.1}s)",
                        t0.elapsed().as_secs_f64()
                    );
                }
            }
        }
        Mode::Anneal => {
            let mut x = rng.random::<u64>() & free_mask;
            let mut cur = consider(x, &mut evals, &mut best_x, &mut best_wt);
            let mut temp = (ch.circuit.data_wires as f64) / 4.0;
            let cooling = (0.1_f64 / temp.max(1.0)).powf(1.0 / args.samples as f64);
            for _ in 1..args.samples {
                let i = rng.random_range(0..free_bits);
                let y = x ^ (1u64 << i);
                let wt = consider(y, &mut evals, &mut best_x, &mut best_wt);
                let delta = wt as f64 - cur as f64;
                if delta <= 0.0 || rng.random::<f64>() < (-delta / temp).exp() {
                    x = y;
                    cur = wt;
                }
                temp *= cooling;
            }
        }
        Mode::Polish => {
            let start = u64::from_str_radix(args.start_x.trim_start_matches("0x"), 16)
                .expect("start_x hex")
                & free_mask;
            best_wt = consider(start, &mut evals, &mut best_x, &mut best_wt);
            let k_max = args.polish_k.clamp(1, 4);
            println!("polishing start=0x{start:08x} wt={best_wt} k_max={k_max}");

            // All XOR masks with popcount in 1..=k_max
            let mut stack = vec![(0u64, 0usize, 0usize)]; // (xor_mask, depth, next_bit)
            while let Some((xor_mask, depth, next)) = stack.pop() {
                if depth >= 1 {
                    consider(start ^ xor_mask, &mut evals, &mut best_x, &mut best_wt);
                }
                if depth == k_max {
                    continue;
                }
                for i in (next..free_bits).rev() {
                    stack.push((xor_mask ^ (1u64 << i), depth + 1, i + 1));
                }
            }
        }
    }

    println!(
        "DONE evals={evals} best_wt={best_wt} best_x=0x{best_x:08x} elapsed={:.2}s",
        t0.elapsed().as_secs_f64()
    );
}
