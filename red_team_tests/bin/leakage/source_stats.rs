//! Are the bits that mask terms are built from actually good mask sources?
//!
//! Every statistical claim the product-share encoding makes assumes its source
//! bits are jointly uniform: a degree-`d` term fires at exactly `2^-d`, and the
//! terms of a value pile up independently. The band fill was designed for that
//! (a private pivot per wire makes each bit MARGINALLY balanced), but nothing
//! in the tree has ever checked it, and marginal balance is not joint
//! uniformity — a mask multiplies THREE bits together.
//!
//! This tool measures the source ensemble directly, at a chosen snapshot:
//!
//!   * per-wire bias        |Pr[w = 1] - 1/2|, the marginal the pivot trick
//!                          is supposed to make exactly 0;
//!   * pairwise correlation max |Pr[a = b] - 1/2| over sampled pairs, the
//!                          first thing marginal balance does not control;
//!   * triple AND rate      Pr[w_a & w_b & w_c], which is exactly a degree-3
//!                          mask term's firing rate and should be 1/8.
//!
//! Read it against the sampling floor it prints: with `S` samples a rate is
//! resolved to about `1/(2*sqrt(S))`, so a deviation only means something when
//! it clears that. Comparing two builds is more informative than one number.
//!
//! Usage: source_stats --g <circuit> --n <logical n> [--prefix-frac 0.5]
//!                     [--wires 512-557] [--samples 65536] [--tuples 20000]
//!
//! `--wires` selects the candidate source set: the band range for a banded
//! build, or the whole carrier space for a distributed one. Inputs are the
//! zero-slice convention (random x on wires 0..n, zeros elsewhere), matching
//! every other measurement in the project.

use clap::Parser;
use local_mixing::engine::format::{read_g57_file, read_mpmct};
use local_mixing::circuit::xgate::{XGate, max_wire};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Parser, Debug)]
#[command(name = "source_stats")]
struct Args {
    #[arg(long)]
    g: String,
    #[arg(long, default_value = "mpmct1")]
    g_format: String,
    /// Logical input width: random x on wires 0..n-1, zeros elsewhere.
    #[arg(long)]
    n: usize,
    /// Where to snapshot, as a fraction of the gate list (0.5 = mid-body).
    #[arg(long, default_value_t = 0.5)]
    prefix_frac: f64,
    /// Candidate source wires, e.g. "512-557" (empty = all wires).
    #[arg(long, default_value = "")]
    wires: String,
    #[arg(long, default_value_t = 65536)]
    samples: usize,
    /// Random pairs/triples to sample from the candidate set.
    #[arg(long, default_value_t = 20000)]
    tuples: usize,
    /// Also scan EVERY pair and list the near-perfect ones. A pair that always
    /// agrees (or always differs) is an exact linear relation between two
    /// wires — a free equation for an affine adversary, and invisible to any
    /// marginal-balance check.
    #[arg(long, default_value_t = false)]
    exhaustive_pairs: bool,
    #[arg(long, default_value_t = 12345)]
    seed: u64,
}

fn parse_wires(spec: &str, nw: usize) -> Vec<usize> {
    if spec.is_empty() {
        return (0..nw).collect();
    }
    let mut v = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if let Some((lo, hi)) = part.split_once('-') {
            let lo: usize = lo.parse().expect("bad --wires range");
            let hi: usize = hi.parse().expect("bad --wires range");
            v.extend(lo..=hi.min(nw - 1));
        } else if !part.is_empty() {
            v.push(part.parse().expect("bad --wires entry"));
        }
    }
    v.sort_unstable();
    v.dedup();
    assert!(!v.is_empty(), "--wires selects nothing");
    v
}

fn main() {
    let args = Args::parse();
    let gates: Vec<XGate> = match args.g_format.as_str() {
        "g57" => read_g57_file(&args.g).expect("read g (g57)"),
        "mpmct1" => read_mpmct(&args.g).expect("read g (mpmct1)").0,
        o => panic!("unknown --g-format {o}"),
    };
    let nw = (max_wire(&gates) as usize + 1).max(args.n);
    let cut = ((gates.len() as f64) * args.prefix_frac) as usize;
    let batches = args.samples.div_ceil(64).max(2);
    let total_bits = (batches * 64) as f64;
    let cand = parse_wires(&args.wires, nw);
    let mut rng = StdRng::seed_from_u64(args.seed);

    // Snapshot every candidate wire at the cut, bit-sliced over samples.
    let mut lanes: Vec<Vec<u64>> = vec![Vec::with_capacity(batches); cand.len()];
    for _ in 0..batches {
        let mut state = vec![0u64; nw];
        for w in 0..args.n {
            state[w] = rng.random::<u64>();
        }
        for gate in &gates[..cut] {
            gate.apply_lanes(&mut state);
        }
        for (slot, &w) in cand.iter().enumerate() {
            lanes[slot].push(state[w]);
        }
    }

    let ones = |v: &Vec<u64>| -> f64 {
        v.iter().map(|w| w.count_ones() as f64).sum::<f64>() / total_bits
    };
    let floor = 1.0 / (2.0 * (total_bits).sqrt());

    // Marginals.
    let biases: Vec<f64> = lanes.iter().map(|v| (ones(v) - 0.5).abs()).collect();
    let mut sorted = biases.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = sorted[sorted.len() / 2];
    let worst = *sorted.last().unwrap();
    let over = biases.iter().filter(|&&b| b > 3.0 * floor).count();

    // Pairs: |Pr[a == b] - 1/2|, the correlation marginal balance cannot see.
    let mut pair_worst = 0f64;
    let mut pair_sum = 0f64;
    let mut pair_over = 0usize;
    for _ in 0..args.tuples {
        let i = rng.random_range(0..cand.len());
        let j = loop {
            let j = rng.random_range(0..cand.len());
            if j != i {
                break j;
            }
        };
        let agree: f64 = (0..batches)
            .map(|b| (!(lanes[i][b] ^ lanes[j][b])).count_ones() as f64)
            .sum::<f64>()
            / total_bits;
        let dev = (agree - 0.5).abs();
        pair_sum += dev;
        if dev > pair_worst {
            pair_worst = dev;
        }
        if dev > 3.0 * floor {
            pair_over += 1;
        }
    }

    // Triples: the actual degree-3 mask firing rate, which should be 1/8.
    // Also count DEAD terms: a mask is `PROD (w ^ a)` with compile-time
    // offsets, so if two of its factor wires happen to carry the SAME bit and
    // the draw gave them opposite offsets, the product is identically zero —
    // the value silently loses that mask term. The gadgetizer cannot see this,
    // because it picks factors by wire id, never by value.
    let mut t_min = 1f64;
    let mut t_max = 0f64;
    let mut t_sum = 0f64;
    let mut t_over = 0usize;
    let mut dead = 0usize;
    let mut dead_tuples = 0usize;
    for _ in 0..args.tuples {
        let mut idx = [0usize; 3];
        let mut k = 0;
        while k < 3 {
            let c = rng.random_range(0..cand.len());
            if !idx[..k].contains(&c) {
                idx[k] = c;
                k += 1;
            }
        }
        // All 8 offset assignments, as the ledger would draw them.
        let mut any_dead = false;
        for pol in 0..8u8 {
            let lit = |slot: usize, bit: u8, b: usize| -> u64 {
                let w = lanes[idx[slot]][b];
                if pol >> bit & 1 == 1 { !w } else { w }
            };
            let fires: u32 = (0..batches)
                .map(|b| (lit(0, 0, b) & lit(1, 1, b) & lit(2, 2, b)).count_ones())
                .sum();
            if fires == 0 {
                dead += 1;
                any_dead = true;
            }
        }
        if any_dead {
            dead_tuples += 1;
        }
        let fire: f64 = (0..batches)
            .map(|b| (lanes[idx[0]][b] & lanes[idx[1]][b] & lanes[idx[2]][b]).count_ones() as f64)
            .sum::<f64>()
            / total_bits;
        t_sum += fire;
        if fire < t_min {
            t_min = fire;
        }
        if fire > t_max {
            t_max = fire;
        }
        if (fire - 0.125).abs() > 3.0 * floor {
            t_over += 1;
        }
    }

    println!(
        "[source_stats] {} gates ({nw}w), cut at {cut} ({:.2}), {} candidate wires, {} samples; \
         3-sigma floor = {:.5}",
        gates.len(),
        args.prefix_frac,
        cand.len(),
        batches * 64,
        3.0 * floor
    );
    println!(
        "  marginal   |bias|: median {med:.5}  worst {worst:.5}  wires over floor: {over}/{}",
        cand.len()
    );
    println!(
        "  pairwise   |Pr[a=b]-1/2|: mean {:.5}  worst {pair_worst:.5}  pairs over floor: {pair_over}/{}",
        pair_sum / args.tuples as f64,
        args.tuples
    );
    println!(
        "  triple AND rate (want 0.125): mean {:.5}  min {t_min:.5}  max {t_max:.5}  \
         triples over floor: {t_over}/{}",
        t_sum / args.tuples as f64,
        args.tuples
    );

    println!(
        "  DEAD terms (identically-zero products): {dead}/{} offset assignments, \
         {dead_tuples}/{} wire triples admit one",
        args.tuples * 8,
        args.tuples
    );

    if args.exhaustive_pairs {
        let mut exact = Vec::new();
        let mut near = 0usize;
        for i in 0..cand.len() {
            for j in (i + 1)..cand.len() {
                let agree: f64 = (0..batches)
                    .map(|b| (!(lanes[i][b] ^ lanes[j][b])).count_ones() as f64)
                    .sum::<f64>()
                    / total_bits;
                let dev = (agree - 0.5).abs();
                if dev > 0.49 {
                    exact.push((cand[i], cand[j], agree));
                } else if dev > 0.25 {
                    near += 1;
                }
            }
        }
        println!(
            "  EXHAUSTIVE pairs: {} exact linear relations (|dev| > 0.49), {near} strong (> 0.25), \
             out of {} pairs",
            exact.len(),
            cand.len() * (cand.len() - 1) / 2
        );
        for (a, b, agree) in exact.iter().take(20) {
            println!("    w{a} {} w{b}   (agreement {agree:.4})", if *agree > 0.5 { "==" } else { "!=" });
        }
        if exact.len() > 20 {
            println!("    ... and {} more", exact.len() - 20);
        }
    }
}
