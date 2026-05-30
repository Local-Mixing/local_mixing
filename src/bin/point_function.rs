// Build a point function (or an identity that looks like one!)

use std::{ops::Range, slice::IterMut};

use clap::Parser;
use itertools::chain;
use local_mixing::circuit::{CircuitSeq, Gate};

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    #[arg(short = 'n', default_value_t = 128)]
    wires: u8,

    #[arg(short, long, default_value_t = 42)]
    key: usize,

    #[arg(short, long, default_value_t = false)]
    identity: bool,
}

fn tof_to_g57(wires: u8, g: &[u8; 3]) -> Vec<[u8; 3]> {
    // Let r be a random wire from the range 0..wires, NOT including any of the values in g.
    let mut r = fastrand::u8(0..wires);
    while g.contains(&r) {
        r = fastrand::u8(0..wires);
    }

    let active = g[0] as u8;
    let ctrl1 = g[1] as u8;
    let ctrl2 = g[2] as u8;

    vec![
        [active, ctrl1, r],
        [ctrl2, ctrl1, r],
        [active, ctrl1, ctrl2],
        [ctrl2, ctrl1, r],
    ]
}

fn big_tof(n: u8, active: u8, controls: Range<u8>) -> CircuitSeq {
    // Build the staircase
    let mut empty: Vec<u8> = (0..n + 2).collect();

    // Delete index `active`, and all indices in `controls` from the vector
    empty.retain(|x| *x != active && !controls.contains(x));

    // Shuffle the remaining indices
    fastrand::shuffle(&mut empty);

    let mut ctrl: Vec<u8> = controls.collect();

    let stair: Vec<[u8; 3]> = (1..(ctrl.len() - 2))
        .map(|i| [empty[i - 1], ctrl[i], empty[i]])
        .collect();

    let mut rev_stair = stair.clone();
    rev_stair.reverse();

    let last_tof = empty[ctrl.len() - 3];

    let base_gate = [active, ctrl[0], empty[0]];
    let top_gate = [last_tof, ctrl[ctrl.len() - 2], ctrl[ctrl.len() - 1]];

    CircuitSeq {
        gates: chain![
            tof_to_g57(n, &base_gate),
            stair.iter().flat_map(|g| { tof_to_g57(n, g) }),
            tof_to_g57(n, &top_gate),
            rev_stair.iter().flat_map(|g| { tof_to_g57(n, g) }),
            tof_to_g57(n, &base_gate),
            stair.iter().flat_map(|g| { tof_to_g57(n, g) }),
            tof_to_g57(n, &top_gate),
            rev_stair.iter().flat_map(|g| { tof_to_g57(n, g) }),
        ]
        .collect(),
    }
}

fn main() {
    let args = Args::parse();

    let n = args.wires;

    println!("{:?}", args);

    let key_bits = (usize::BITS as u32 - args.key.leading_zeros()) as u8;
    assert!(key_bits < n);

    let b = n.div_ceil(2);

    // TODO: Insert the NOTs

    let pf = big_tof(n, n, 0..b)
        .concat(&big_tof(n, n + 1, b..n + 1))
        .concat(&big_tof(n, n, 0..b))
        .concat(&big_tof(n, n + 1, b..n + 1));

    // TODO: id or pf
    // Insert the NOTs

    println!("{}", pf.repr());

    println!("0 => {}", pf.evaluate(0));
    let r = fastrand::usize(0..(1 << n));
    println!("{} => {}", r, pf.evaluate(r));

}
