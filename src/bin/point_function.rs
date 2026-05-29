// Build a point function (or an identity that looks like one!)

use std::ops::Range;

use clap::{Parser};
use local_mixing::circuit::{CircuitSeq, Gate};

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    #[arg(short = 'n', default_value_t = 128)]
    wires: u8,

    #[arg(short, long, default_value_t = 42)]
    key: usize,

    #[arg(short, long, default_value_t = false)]
    identity: bool
}

fn tof_to_g57(wires: u8, g: Gate) -> CircuitSeq {
    // Let r be a random wire from the range 0..wires, NOT including any of the values in g.
    let mut r = fastrand::u8(0..wires);
    while g.pins.contains(&r) {
        r = fastrand::u8(0..wires);
    }

    let active = g.pins[0] as u8;
    let ctrl1 = g.pins[1] as u8;
    let ctrl2 = g.pins[2] as u8;

    CircuitSeq { gates: vec![
        [active, ctrl1, r],
        [ctrl2, ctrl1, r],
        [active, ctrl1, ctrl2],
        [ctrl2, ctrl2, r]
    ]}
}

fn big_tof(active: u8, controls: Range<u8>) {
    // Build the staircase
}


fn main() {
    let args = Args::parse();

    let n = args.wires;

    println!("{:?}", args);

    let key_bits = (usize::BITS as u32 - args.key.leading_zeros()) as u8;
    assert!(key_bits < n);

    let b = n.div_ceil(2);

    // Insert the NOTs

    big_tof(n, 0..b);
    big_tof(n+1, b..n+1);
    big_tof(n, 0..b);
    big_tof(n+1, b..n+1);

    // Insert the NOTs
}