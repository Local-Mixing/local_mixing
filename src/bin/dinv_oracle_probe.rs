//! Structural probe of the D^-1 oracle that the REVERSED artifact exposes.
//!
//! `attack_surface_probe` establishes (by exhaustive n=3 enumeration) that the
//! gate-order-reversed 6n artifact returns, in its Y block,
//!
//!     rev_Y(u, v, z, w, band) XOR v  =  D^-1(u)
//!
//! for arbitrary chosen u and arbitrary everything else. That is a chosen-input
//! oracle for D^-1 at one circuit evaluation per query. This binary builds the
//! artifact at a realistic n and measures how structured that oracle is:
//! affinity, algebraic degree, and per-output-bit linear bias — i.e. how hard it
//! is to invert D^-1 at the single point t, which is the whole remaining problem.

use local_mixing::circuit::CircuitSeq;
use local_mixing::circuit::circuit::U1024;
use local_mixing::postmix::xgate::{XGate, eval_lanes, eval_u1024};
use local_mixing::replace::gadgets::tdp4n_nonlinear_with_slice_zero_random_cnot;
use rand::{Rng, SeedableRng, rngs::StdRng};

/// Read the n-bit block starting at `lo` out of a 6n-wire lane state, lane 0.
fn block_from_u1024(state: &U1024, lo: usize, n: usize) -> u128 {
    let one = U1024::one();
    let mut v = 0u128;
    for b in 0..n {
        if ((*state >> (lo + b)) & one) != U1024::zero() {
            v |= 1u128 << b;
        }
    }
    v
}

fn u1024_from_blocks(vals: &[(usize, u128)], n: usize) -> U1024 {
    let one = U1024::one();
    let mut s = U1024::zero();
    for &(lo, v) in vals {
        for b in 0..n {
            if (v >> b) & 1 == 1 {
                s = s | (one << (lo + b));
            }
        }
    }
    s
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(32);
    let c_gates_count: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(4096);
    let seed: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let trials: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(4096);
    assert!(n <= 128);
    let mut rng = StdRng::seed_from_u64(seed);

    let source = CircuitSeq {
        gates: (0..c_gates_count)
            .map(|_| {
                let a = rng.random_range(0..n) as u16;
                let x = loop {
                    let w = rng.random_range(0..n) as u16;
                    if w != a {
                        break w;
                    }
                };
                let y = loop {
                    let w = rng.random_range(0..n) as u16;
                    if w != a && w != x {
                        break w;
                    }
                };
                [a, x, y]
            })
            .collect(),
    };

    let built = tdp4n_nonlinear_with_slice_zero_random_cnot(&source, n, 1, 4 * n, &mut rng);
    let gates = built.circuit.gates;
    let total = built.circuit.num_wires;
    assert_eq!(total, 6 * n);
    let rev: Vec<XGate> = gates.iter().rev().cloned().collect();
    println!(
        "n={n} wires={total} |C|={c_gates_count} |artifact|={} gates",
        gates.len()
    );

    // one reverse-oracle query = one circuit traversal
    let d_inv = |u: u128| -> u128 {
        let s = u1024_from_blocks(&[(0, u)], n);
        let o = eval_u1024(rev.iter(), s);
        block_from_u1024(&o, n, n) // rev Y block, with v = 0
    };

    // ---- is D^-1 affine? second derivative test -------------------------
    // affine  <=>  f(a)^f(b)^f(c)^f(a^b^c) == 0 for all a,b,c
    let mask: u128 = if n == 128 {
        u128::MAX
    } else {
        (1u128 << n) - 1
    };
    let mut affine_violations = 0usize;
    let mut second_deriv_weight_total = 0usize;
    for _ in 0..trials.min(512) {
        let a: u128 = rng.random::<u128>() & mask;
        let b: u128 = rng.random::<u128>() & mask;
        let c: u128 = rng.random::<u128>() & mask;
        let d = d_inv(a) ^ d_inv(b) ^ d_inv(c) ^ d_inv(a ^ b ^ c);
        if d != 0 {
            affine_violations += 1;
        }
        second_deriv_weight_total += d.count_ones() as usize;
    }
    let t2 = trials.min(512);
    println!(
        "D^-1 second-derivative test: nonzero on {affine_violations}/{t2} random triples, \
         mean hamming weight of the residue = {:.2} of {n} bits",
        second_deriv_weight_total as f64 / t2 as f64
    );

    // ---- first derivative: is any single-bit derivative constant? --------
    // A constant derivative in direction e means a linear structure the
    // attacker could quotient out.
    let base_pts: Vec<u128> = (0..32).map(|_| rng.random::<u128>() & mask).collect();
    let mut constant_dirs = 0usize;
    for bit in 0..n {
        let e = 1u128 << bit;
        let d0 = d_inv(base_pts[0]) ^ d_inv(base_pts[0] ^ e);
        if base_pts.iter().all(|&p| d_inv(p) ^ d_inv(p ^ e) == d0) {
            constant_dirs += 1;
        }
    }
    println!("D^-1 linear structures among the {n} unit directions: {constant_dirs}");

    // ---- per-output-bit correlation with the best single input bit -------
    // 64-lane bitsliced sampling of the reverse oracle.
    let samples = trials.max(64) / 64 * 64;
    let mut counts = vec![vec![0i64; n]; n]; // counts[out][in] = agreements - disagreements
    let mut batches = 0usize;
    while batches * 64 < samples {
        let mut state = vec![0u64; total];
        let mut ins = [0u128; 64];
        for (lane, item) in ins.iter_mut().enumerate() {
            let u: u128 = rng.random::<u128>() & mask;
            *item = u;
            for b in 0..n {
                if (u >> b) & 1 == 1 {
                    state[b] |= 1u64 << lane;
                }
            }
        }
        eval_lanes(rev.iter(), &mut state);
        for ob in 0..n {
            let out_word = state[n + ob];
            for ib in 0..n {
                let mut in_word = 0u64;
                for (lane, &u) in ins.iter().enumerate() {
                    if (u >> ib) & 1 == 1 {
                        in_word |= 1u64 << lane;
                    }
                }
                let agree = (!(out_word ^ in_word)).count_ones() as i64;
                counts[ob][ib] += 2 * agree - 64;
            }
        }
        batches += 1;
    }
    let total_samples = (batches * 64) as f64;
    let mut worst = 0f64;
    for ob in 0..n {
        for ib in 0..n {
            let bias = (counts[ob][ib] as f64 / total_samples).abs();
            if bias > worst {
                worst = bias;
            }
        }
    }
    println!(
        "D^-1 max |correlation| between any single output bit and any single input bit \
         over {} samples: {worst:.4}  (random ~ {:.4})",
        total_samples,
        1.0 / total_samples.sqrt() * 3.0
    );

    // ---- balance: does each output bit look balanced? --------------------
    let mut min_bal = 1f64;
    for ob in 0..n {
        let mut ones = 0i64;
        let mut seen = 0i64;
        let mut b2 = 0usize;
        while b2 * 64 < samples.min(4096) {
            let mut state = vec![0u64; total];
            for lane in 0..64 {
                let u: u128 = rng.random::<u128>() & mask;
                for b in 0..n {
                    if (u >> b) & 1 == 1 {
                        state[b] |= 1u64 << lane;
                    }
                }
            }
            eval_lanes(rev.iter(), &mut state);
            ones += state[n + ob].count_ones() as i64;
            seen += 64;
            b2 += 1;
        }
        let bal = (ones as f64 / seen as f64 - 0.5).abs();
        if ob == 0 || bal < min_bal {
            min_bal = min_bal.min(bal);
        }
        if bal > 0.05 {
            println!("  output bit {ob} imbalance {bal:.3}");
        }
    }
    println!(
        "D^-1 output-bit balance: worst deviation from 1/2 reported above (none printed = all < 0.05)"
    );
}
