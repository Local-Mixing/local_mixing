use std::{env, fs};

use primitive_types::U512;
use rand::{RngCore, seq::SliceRandom};
use rayon::prelude::*;

use local_mixing::circuit::{CircuitSeq, Gate};

fn parse_u128_hex(s: &str) -> u128 {
    let hex = s.trim().trim_start_matches("0x").trim_start_matches("0X");
    u128::from_str_radix(hex, 16).expect("bad target hex")
}

fn low_u128(v: U512) -> u128 {
    let bytes = v.to_little_endian();
    let mut lo = [0u8; 16];
    lo.copy_from_slice(&bytes[..16]);
    u128::from_le_bytes(lo)
}

fn reverse_eval(gates: &[[u16; 3]], left: u128, target: u128, right: u128) -> U512 {
    let mut state = U512::from(left) | (U512::from(target) << 128) | (U512::from(right) << 256);
    for &g in gates.iter().rev() {
        state = Gate::evaluate_index_512(state, g);
    }
    state
}

fn hamming(x: u128) -> u32 {
    x.count_ones()
}

fn bit(row: &[u64; 4], col: usize) -> bool {
    ((row[col / 64] >> (col % 64)) & 1) == 1
}

fn xor_row(dst: &mut [u64; 4], src: &[u64; 4]) {
    for i in 0..4 {
        dst[i] ^= src[i];
    }
}

fn solve_linear_samples(
    mut rows: Vec<[u64; 4]>,
    mut rhs: Vec<bool>,
    samples: usize,
    rng: &mut impl RngCore,
) -> Option<Vec<[u64; 4]>> {
    let m = rows.len();
    let mut where_col = [usize::MAX; 256];
    let mut rank = 0usize;
    for col in 0..256 {
        let pivot = (rank..m).find(|&r| bit(&rows[r], col));
        let Some(p) = pivot else { continue };
        rows.swap(rank, p);
        rhs.swap(rank, p);
        where_col[col] = rank;
        for r in 0..m {
            if r != rank && bit(&rows[r], col) {
                let pivot_row = rows[rank];
                xor_row(&mut rows[r], &pivot_row);
                rhs[r] ^= rhs[rank];
            }
        }
        rank += 1;
        if rank == m {
            break;
        }
    }
    for r in rank..m {
        if rows[r] == [0; 4] && rhs[r] {
            return None;
        }
    }
    let mut sol = [0u64; 4];
    for col in 0..256 {
        let r = where_col[col];
        if r != usize::MAX && rhs[r] {
            sol[col / 64] |= 1u64 << (col % 64);
        }
    }
    let free_cols: Vec<usize> = (0..256).filter(|&c| where_col[c] == usize::MAX).collect();
    let mut out = Vec::with_capacity(samples + 1);
    out.push(sol);
    for _ in 0..samples {
        let mut s = sol;
        for &free in &free_cols {
            if (rng.next_u64() & 1) == 0 {
                continue;
            }
            s[free / 64] ^= 1u64 << (free % 64);
            for pivot_col in 0..256 {
                let r = where_col[pivot_col];
                if r != usize::MAX && bit(&rows[r], free) {
                    s[pivot_col / 64] ^= 1u64 << (pivot_col % 64);
                }
            }
        }
        out.push(s);
    }
    Some(out)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: newton_feistal <circuit> <target_hex> [restarts] [iters] [samples]");
        std::process::exit(2);
    }
    let data = fs::read_to_string(&args[1]).expect("read circuit");
    let circuit = CircuitSeq::from_string(&data);
    let gates = circuit.gates;
    let target = parse_u128_hex(&args[2]);
    let restarts: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(100);
    let iters: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(64);
    let samples: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(32);
    let mask = (U512::one() << 128) - U512::one();
    let mut rng = rand::rng();
    let mut global_best = (129u32, 0u128, 0u128);

    println!("gates: {}", gates.len());
    for restart in 0..restarts {
        let mut left = rng.next_u64() as u128 | ((rng.next_u64() as u128) << 64);
        let mut right = rng.next_u64() as u128 | ((rng.next_u64() as u128) << 64);
        for iter in 0..iters {
            let base = reverse_eval(&gates, left, target, right);
            let y = low_u128((base >> 128) & mask);
            let wt = hamming(y);
            println!("restart {restart} iter {iter} wt {wt}");
            if wt < global_best.0 {
                global_best = (wt, left, right);
                println!("best wt={wt} left=0x{left:032x} right=0x{right:032x}");
            }
            if y == 0 {
                let x = low_u128(base & mask);
                let z = low_u128((base >> 256) & mask);
                let input = U512::from(x) | (U512::from(z) << 256);
                let out = Gate::evaluate_index_list_512(input, &gates);
                let middle = low_u128((out >> 128) & mask);
                println!("x=0x{x:032x}");
                println!("z=0x{z:032x}");
                println!("middle=0x{middle:032x}");
                println!("ok={}", middle == target);
                return;
            }

            let cols: Vec<u128> = (0..256)
                .into_par_iter()
                .map(|col| {
                    let (l2, r2) = if col < 128 {
                        (left ^ (1u128 << col), right)
                    } else {
                        (left, right ^ (1u128 << (col - 128)))
                    };
                    let st = reverse_eval(&gates, l2, target, r2);
                    low_u128(((st >> 128) & mask) ^ U512::from(y))
                })
                .collect();
            let mut rows = vec![[0u64; 4]; 128];
            for (col, &dy) in cols.iter().enumerate() {
                for row in 0..128 {
                    if ((dy >> row) & 1) == 1 {
                        rows[row][col / 64] |= 1u64 << (col % 64);
                    }
                }
            }

            let mut best_flip = None;
            for (col, &dy) in cols.iter().enumerate() {
                let wt2 = hamming(y ^ dy);
                if best_flip.map_or(true, |(_, best_wt)| wt2 < best_wt) {
                    best_flip = Some((col, wt2));
                }
            }
            if let Some((col, wt2)) = best_flip {
                if wt2 < wt {
                    if col < 128 {
                        left ^= 1u128 << col;
                    } else {
                        right ^= 1u128 << (col - 128);
                    }
                    println!("  greedy flip {col} wt {wt} -> {wt2}");
                    continue;
                }
            }

            let rhs: Vec<bool> = (0..128).map(|i| ((y >> i) & 1) == 1).collect();
            let candidates = match solve_linear_samples(rows.clone(), rhs, samples, &mut rng) {
                Some(c) => c,
                None => {
                    let ones: Vec<usize> = (0..128).filter(|&i| ((y >> i) & 1) == 1).collect();
                    let sub_rows: Vec<[u64; 4]> = ones.iter().map(|&i| rows[i]).collect();
                    let sub_rhs = vec![true; sub_rows.len()];
                    match solve_linear_samples(sub_rows, sub_rhs, samples * 2, &mut rng) {
                        Some(c) => {
                            println!(
                                "full linear system inconsistent; using {} wrong-bit rows",
                                ones.len()
                            );
                            c
                        }
                        None => {
                            println!("wrong-bit subsystem inconsistent");
                            break;
                        }
                    }
                }
            };
            let mut best = None;
            for delta in candidates {
                let dl = delta[0] as u128 | ((delta[1] as u128) << 64);
                let dr = delta[2] as u128 | ((delta[3] as u128) << 64);
                if dl == 0 && dr == 0 {
                    continue;
                }
                let st = reverse_eval(&gates, left ^ dl, target, right ^ dr);
                let y2 = low_u128((st >> 128) & mask);
                let wt2 = hamming(y2);
                if best.map_or(true, |(_, _, best_wt)| wt2 < best_wt) {
                    best = Some((dl, dr, wt2));
                }
            }
            let Some((mut dl, mut dr, mut best_wt)) = best else {
                println!("zero deltas");
                break;
            };
            if best_wt > wt {
                let ones: Vec<usize> = (0..128).filter(|&i| ((y >> i) & 1) == 1).collect();
                for _ in 0..16 {
                    let mut subset = ones.clone();
                    subset.shuffle(&mut rng);
                    let keep = (subset.len() / 2).max(1);
                    subset.truncate(keep);
                    let sub_rows: Vec<[u64; 4]> = subset.iter().map(|&i| rows[i]).collect();
                    let sub_rhs = vec![true; sub_rows.len()];
                    if let Some(cands) =
                        solve_linear_samples(sub_rows, sub_rhs, samples / 2 + 1, &mut rng)
                    {
                        for delta in cands {
                            let ndl = delta[0] as u128 | ((delta[1] as u128) << 64);
                            let ndr = delta[2] as u128 | ((delta[3] as u128) << 64);
                            if ndl == 0 && ndr == 0 {
                                continue;
                            }
                            let st = reverse_eval(&gates, left ^ ndl, target, right ^ ndr);
                            let y2 = low_u128((st >> 128) & mask);
                            let wt2 = hamming(y2);
                            if wt2 < best_wt {
                                dl = ndl;
                                dr = ndr;
                                best_wt = wt2;
                            }
                        }
                    }
                    if best_wt <= wt {
                        break;
                    }
                }
                if best_wt > wt {
                    println!("no improving affine/subset sample; best {best_wt}");
                    break;
                }
            }
            println!("  step wt {wt} -> {best_wt}");
            left ^= dl;
            right ^= dr;
        }
    }
    println!(
        "no solution found; best wt={} left=0x{:032x} right=0x{:032x}",
        global_best.0, global_best.1, global_best.2
    );
}
