//! Black-box / hybrid search for n=128 challenges (64 free bits) using bit vectors.

use clap::{Parser, ValueEnum};
use local_mixing::sandwich::{hamming_weight, ChunkedCircuit};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Clone, ValueEnum, Debug)]
enum Mode {
    Random,
    Hill,
    Anneal,
}

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "challenges/lowweight128_obf.txt")]
    challenge: PathBuf,
    #[arg(long, default_value = "anneal")]
    mode: Mode,
    #[arg(long, default_value_t = 50_000)]
    samples: usize,
    #[arg(long, default_value_t = 7)]
    seed: u64,
    #[arg(long, default_value_t = 2000)]
    report_every: usize,
    /// Optional hex seed for free bits [0..64) (lo). Empty = random start.
    #[arg(long, default_value = "")]
    start_x_lo: String,
    #[arg(long, default_value = "")]
    start_x_hi: String,
}

fn load(path: &PathBuf) -> (ChunkedCircuit, usize) {
    let s = fs::read_to_string(path).unwrap();
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
    (ChunkedCircuit::from_readable_string(&body).unwrap(), prefix)
}

fn eval_wt(c: &ChunkedCircuit, prefix: usize, free: &[bool]) -> usize {
    let mut input = vec![false; c.data_wires];
    input[prefix..].copy_from_slice(free);
    hamming_weight(&c.evaluate_data_bits(&input))
}

fn consider(
    circ: &ChunkedCircuit,
    prefix: usize,
    free_bits: usize,
    free: &[bool],
    best_wt: &mut usize,
    best_free: &mut Vec<bool>,
    evals: &mut usize,
    report_every: usize,
    t0: Instant,
) -> usize {
    let wt = eval_wt(circ, prefix, free);
    *evals += 1;
    if wt < *best_wt {
        *best_wt = wt;
        *best_free = free.to_vec();
        let mut x0 = 0u64;
        let mut x1 = 0u64;
        for i in 0..free_bits.min(64) {
            if best_free[i] {
                x0 |= 1 << i;
            }
        }
        for i in 64..free_bits {
            if best_free[i] {
                x1 |= 1 << (i - 64);
            }
        }
        println!(
            "  new best wt={best_wt} x_lo=0x{x0:016x} x_hi=0x{x1:016x} evals={evals} ({:.1}s)",
            t0.elapsed().as_secs_f64()
        );
    }
    if report_every > 0 && *evals % report_every == 0 {
        println!(
            "  progress evals={evals} best={best_wt} ({:.1}s)",
            t0.elapsed().as_secs_f64()
        );
    }
    wt
}

fn main() {
    let args = Args::parse();
    let (circ, prefix) = load(&args.challenge);
    let free_bits = circ.data_wires - prefix;
    println!(
        "n={} total={} chunks={} prefix={} free={}",
        circ.data_wires,
        circ.total_wires,
        circ.chunks.len(),
        prefix,
        free_bits
    );

    let mut rng = StdRng::seed_from_u64(args.seed);
    let t0 = Instant::now();
    let mut best_free = vec![false; free_bits];
    let mut best_wt = usize::MAX;
    let mut evals = 0usize;

    let start_free = if !args.start_x_lo.is_empty() {
        let lo = u64::from_str_radix(args.start_x_lo.trim_start_matches("0x"), 16).unwrap();
        let hi = if args.start_x_hi.is_empty() {
            0u64
        } else {
            u64::from_str_radix(args.start_x_hi.trim_start_matches("0x"), 16).unwrap()
        };
        let mut free = vec![false; free_bits];
        for i in 0..free_bits.min(64) {
            free[i] = ((lo >> i) & 1) != 0;
        }
        for i in 64..free_bits {
            free[i] = ((hi >> (i - 64)) & 1) != 0;
        }
        Some(free)
    } else {
        None
    };

    match args.mode {
        Mode::Random => {
            for _ in 0..args.samples {
                let free: Vec<bool> = (0..free_bits).map(|_| rng.random()).collect();
                consider(
                    &circ,
                    prefix,
                    free_bits,
                    &free,
                    &mut best_wt,
                    &mut best_free,
                    &mut evals,
                    args.report_every,
                    t0,
                );
            }
        }
        Mode::Hill => {
            let restarts = 50;
            for _ in 0..restarts {
                let mut free: Vec<bool> = (0..free_bits).map(|_| rng.random()).collect();
                let mut cur = consider(
                    &circ,
                    prefix,
                    free_bits,
                    &free,
                    &mut best_wt,
                    &mut best_free,
                    &mut evals,
                    args.report_every,
                    t0,
                );
                for _ in 0..(args.samples / restarts) {
                    let i = rng.random_range(0..free_bits);
                    free[i] = !free[i];
                    let wt = consider(
                        &circ,
                        prefix,
                        free_bits,
                        &free,
                        &mut best_wt,
                        &mut best_free,
                        &mut evals,
                        args.report_every,
                        t0,
                    );
                    if wt <= cur {
                        cur = wt;
                    } else {
                        free[i] = !free[i];
                    }
                }
                loop {
                    let mut improved = false;
                    for i in 0..free_bits {
                        free[i] = !free[i];
                        let wt = consider(
                            &circ,
                            prefix,
                            free_bits,
                            &free,
                            &mut best_wt,
                            &mut best_free,
                            &mut evals,
                            args.report_every,
                            t0,
                        );
                        if wt < cur {
                            cur = wt;
                            improved = true;
                        } else {
                            free[i] = !free[i];
                        }
                    }
                    if !improved {
                        break;
                    }
                }
            }
        }
        Mode::Anneal => {
            let mut free: Vec<bool> = start_free
                .clone()
                .unwrap_or_else(|| (0..free_bits).map(|_| rng.random()).collect());
            let mut cur = consider(
                &circ,
                prefix,
                free_bits,
                &free,
                &mut best_wt,
                &mut best_free,
                &mut evals,
                args.report_every,
                t0,
            );
            let mut temp = (circ.data_wires as f64) / 3.0;
            let cooling = (0.05_f64 / temp).powf(1.0 / args.samples as f64);
            for _ in 1..args.samples {
                let nflip = if rng.random_bool(0.25) { 2 } else { 1 };
                let mut flipped = Vec::with_capacity(nflip);
                for _ in 0..nflip {
                    let i = rng.random_range(0..free_bits);
                    free[i] = !free[i];
                    flipped.push(i);
                }
                let wt = consider(
                    &circ,
                    prefix,
                    free_bits,
                    &free,
                    &mut best_wt,
                    &mut best_free,
                    &mut evals,
                    args.report_every,
                    t0,
                );
                let delta = wt as f64 - cur as f64;
                if delta <= 0.0 || rng.random::<f64>() < (-delta / temp).exp() {
                    cur = wt;
                } else {
                    for i in flipped {
                        free[i] = !free[i];
                    }
                }
                temp *= cooling;
            }
            for kick in 0..200 {
                free = best_free.clone();
                for _ in 0..(3 + kick % 5) {
                    let i = rng.random_range(0..free_bits);
                    free[i] = !free[i];
                }
                cur = consider(
                    &circ,
                    prefix,
                    free_bits,
                    &free,
                    &mut best_wt,
                    &mut best_free,
                    &mut evals,
                    args.report_every,
                    t0,
                );
                for _ in 0..5000 {
                    let i = rng.random_range(0..free_bits);
                    free[i] = !free[i];
                    let wt = consider(
                        &circ,
                        prefix,
                        free_bits,
                        &free,
                        &mut best_wt,
                        &mut best_free,
                        &mut evals,
                        args.report_every,
                        t0,
                    );
                    if wt <= cur {
                        cur = wt;
                    } else {
                        free[i] = !free[i];
                    }
                }
                loop {
                    let mut improved = false;
                    for i in 0..free_bits {
                        free[i] = !free[i];
                        let wt = consider(
                            &circ,
                            prefix,
                            free_bits,
                            &free,
                            &mut best_wt,
                            &mut best_free,
                            &mut evals,
                            args.report_every,
                            t0,
                        );
                        if wt < cur {
                            cur = wt;
                            improved = true;
                        } else {
                            free[i] = !free[i];
                        }
                    }
                    if !improved {
                        break;
                    }
                }
            }
        }
    }

    println!(
        "DONE best_wt={best_wt} evals={evals} ({:.2}s)",
        t0.elapsed().as_secs_f64()
    );
}
