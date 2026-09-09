//! Warm-up: same low-weight search on a *plaintext* r57 circuit (much easier structure).

use clap::Parser;
use local_mixing::circuit::CircuitSeq;
use local_mixing::sandwich::hamming_weight;
use primitive_types::U256 as u256;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

#[derive(Parser, Debug)]
struct Args {
    #[arg(short = 'n', long, default_value_t = 64)]
    wires: usize,
    #[arg(short = 'm', long, default_value_t = 512)]
    gates: usize,
    #[arg(long, default_value_t = 32)]
    prefix_zeros: usize,
    #[arg(long, default_value_t = 0xC0FFEE)]
    seed: u64,
    #[arg(long, default_value_t = 50_000)]
    samples: usize,
}

fn eval_plain_prefixed(c: &CircuitSeq, n: usize, prefix: usize, free: u64) -> usize {
    let mut state = u256::zero();
    let free_bits = n - prefix;
    for i in 0..free_bits {
        if ((free >> i) & 1) != 0 {
            state |= u256::one() << (prefix + i);
        }
    }
    let out = c.evaluate_256(state);
    let mut bits = Vec::with_capacity(n);
    for i in 0..n {
        bits.push(((out >> i) & u256::one()) == u256::one());
    }
    hamming_weight(&bits)
}

fn random_circuit(n: usize, m: usize, rng: &mut impl Rng) -> CircuitSeq {
    let mut gates = Vec::with_capacity(m);
    for _ in 0..m {
        loop {
            let mut g = [0u16; 3];
            let mut used = vec![false; n];
            for j in 0..3 {
                loop {
                    let v = rng.random_range(0..n) as u16;
                    if !used[v as usize] {
                        used[v as usize] = true;
                        g[j] = v;
                        break;
                    }
                }
            }
            if gates.last() == Some(&g) {
                continue;
            }
            gates.push(g);
            break;
        }
    }
    CircuitSeq { gates }
}

fn main() {
    let args = Args::parse();
    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut c = random_circuit(args.wires, args.gates, &mut rng);
    c.canonicalize();

    let free_bits = args.wires - args.prefix_zeros;
    let free_mask = (1u64 << free_bits) - 1;
    println!(
        "plaintext warm-up n={} m={} prefix={} free_bits={}",
        args.wires, args.gates, args.prefix_zeros, free_bits
    );

    let t0 = Instant::now();
    let mut best_x = 0u64;
    let mut best_wt = usize::MAX;
    let mut evals = 0usize;

    // Random baseline
    for _ in 0..args.samples / 5 {
        let x = rng.random::<u64>() & free_mask;
        let wt = eval_plain_prefixed(&c, args.wires, args.prefix_zeros, x);
        evals += 1;
        if wt < best_wt {
            best_wt = wt;
            best_x = x;
            println!("  [rand] best wt={best_wt} x=0x{best_x:08x} evals={evals}");
        }
    }

    // Hill climb
    let restarts = 40;
    for r in 0..restarts {
        let mut x = rng.random::<u64>() & free_mask;
        let mut cur = eval_plain_prefixed(&c, args.wires, args.prefix_zeros, x);
        evals += 1;
        if cur < best_wt {
            best_wt = cur;
            best_x = x;
            println!("  [hill] best wt={best_wt} x=0x{best_x:08x} evals={evals}");
        }
        for _ in 0..args.samples / restarts {
            let i = rng.random_range(0..free_bits);
            let y = x ^ (1u64 << i);
            let wt = eval_plain_prefixed(&c, args.wires, args.prefix_zeros, y);
            evals += 1;
            if wt <= cur {
                x = y;
                cur = wt;
                if wt < best_wt {
                    best_wt = wt;
                    best_x = x;
                    println!("  [hill] best wt={best_wt} x=0x{best_x:08x} evals={evals}");
                }
            }
        }
        if r % 10 == 0 {
            println!("  restart {r} best={best_wt}");
        }
    }

    // Exhaustive on a few free bits if we greedily fix others — skip for now.

    println!(
        "DONE plaintext best_wt={best_wt} x=0x{best_x:08x} evals={evals} ({:.2}s)",
        t0.elapsed().as_secs_f64()
    );
}
