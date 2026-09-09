//! Structural probe + ILP-style greedy: estimate which free bits turn which
//! output bits on, then pick a free assignment that covers few output 1s.
//!
//! Still only uses the obfuscated circuit (forward eval). No plaintext.

use clap::Parser;
use local_mixing::sandwich::{hamming_weight, ChunkedCircuit};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "challenges/lowweight_obf.txt")]
    challenge: PathBuf,
    #[arg(long, default_value_t = 5)]
    seed: u64,
    /// Number of base points for influence probing
    #[arg(long, default_value_t = 64)]
    probes: usize,
    #[arg(long, default_value_t = 50_000)]
    polish_samples: usize,
}

fn load(path: &PathBuf) -> (ChunkedCircuit, usize) {
    let s = fs::read_to_string(path).unwrap();
    let mut prefix = 32usize;
    let mut body = String::new();
    for line in s.lines() {
        if let Some(r) = line.trim().strip_prefix("challenge_prefix_zeros ") {
            prefix = r.trim().parse().unwrap();
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

fn eval(c: &ChunkedCircuit, prefix: usize, x: u64) -> Vec<bool> {
    c.evaluate_prefixed_zeros(prefix, x)
}

fn main() {
    let args = Args::parse();
    let (circ, prefix) = load(&args.challenge);
    let free_bits = circ.data_wires - prefix;
    let n_out = circ.data_wires;
    let free_mask = (1u64 << free_bits) - 1;
    let mut rng = StdRng::seed_from_u64(args.seed);
    let t0 = Instant::now();

    println!(
        "structural attack: free_bits={free_bits} probes={} polish={}",
        args.probes, args.polish_samples
    );

    // Influence: for each free bit i and output bit o, how often flipping i
    // turns o from 0→1 or 1→0 (correlation magnitude).
    let mut flip_on = vec![vec![0u32; n_out]; free_bits]; // 0->1 counts
    let mut flip_off = vec![vec![0u32; n_out]; free_bits]; // 1->0 counts
    let mut best_x = 0u64;
    let mut best_wt = usize::MAX;

    for p in 0..args.probes {
        let x = rng.random::<u64>() & free_mask;
        let y0 = eval(&circ, prefix, x);
        let wt = hamming_weight(&y0);
        if wt < best_wt {
            best_wt = wt;
            best_x = x;
            println!("  probe best wt={best_wt} x=0x{best_x:08x}");
        }
        for i in 0..free_bits {
            let y1 = eval(&circ, prefix, x ^ (1u64 << i));
            for o in 0..n_out {
                if !y0[o] && y1[o] {
                    flip_on[i][o] += 1;
                }
                if y0[o] && !y1[o] {
                    flip_off[i][o] += 1;
                }
            }
        }
        if (p + 1) % 16 == 0 {
            println!("  probed {}/{} ({:.1}s)", p + 1, args.probes, t0.elapsed().as_secs_f64());
        }
    }

    // Greedy: start from best_x; repeatedly flip the bit that most reduces
    // predicted weight using average influence (prefer flips that turn bits off).
    let mut x = best_x;
    for round in 0..200 {
        let y = eval(&circ, prefix, x);
        let cur = hamming_weight(&y);
        if cur < best_wt {
            best_wt = cur;
            best_x = x;
            println!("  greedy best wt={best_wt} x=0x{best_x:08x}");
        }
        let mut best_i = None;
        let mut best_score = 0.0;
        for i in 0..free_bits {
            // Score: expected #outputs turned off minus turned on at this point.
            let mut score = 0.0;
            for o in 0..n_out {
                let on = flip_on[i][o] as f64;
                let off = flip_off[i][o] as f64;
                let tot = (on + off).max(1.0);
                if y[o] {
                    score += off / tot; // hope to turn off
                } else {
                    score -= on / tot; // risk turning on
                }
            }
            if score > best_score {
                best_score = score;
                best_i = Some(i);
            }
        }
        if let Some(i) = best_i {
            if best_score <= 0.01 {
                break;
            }
            x ^= 1u64 << i;
        } else {
            break;
        }
        if round % 20 == 0 {
            println!("  greedy round {round} cur_wt={cur} score={best_score:.2}");
        }
    }

    // Polish with hill climb from best
    println!("  polishing from wt={best_wt} ...");
    for _ in 0..args.polish_samples {
        let i = rng.random_range(0..free_bits);
        let y = x ^ (1u64 << i);
        let wt = hamming_weight(&eval(&circ, prefix, y));
        if wt <= hamming_weight(&eval(&circ, prefix, x)) {
            x = y;
        }
        if wt < best_wt {
            best_wt = wt;
            best_x = y;
            println!("  polish best wt={best_wt} x=0x{best_x:08x}");
        }
    }

    // Exhaustive 2-flip neighborhood around best (C(32,2)=496)
    println!("  2-flip polish around best...");
    let base = best_x;
    for i in 0..free_bits {
        for j in (i + 1)..free_bits {
            let y = base ^ (1u64 << i) ^ (1u64 << j);
            let wt = hamming_weight(&eval(&circ, prefix, y));
            if wt < best_wt {
                best_wt = wt;
                best_x = y;
                println!("  2flip best wt={best_wt} x=0x{best_x:08x}");
            }
        }
        let y = base ^ (1u64 << i);
        let wt = hamming_weight(&eval(&circ, prefix, y));
        if wt < best_wt {
            best_wt = wt;
            best_x = y;
            println!("  1flip best wt={best_wt} x=0x{best_x:08x}");
        }
    }

    println!(
        "DONE structural best_wt={best_wt} best_x=0x{best_x:08x} ({:.2}s)",
        t0.elapsed().as_secs_f64()
    );
}
