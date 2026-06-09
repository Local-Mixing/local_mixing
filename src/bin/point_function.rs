// Build a point function (or an identity that looks like one!)

use std::{ops::Range, path::Path};

use clap::Parser;
use itertools::chain;
use lmdb::Environment;
use local_mixing::{
    circuit::CircuitSeq,
    open_shard_dbs,
    replace::replace::{compress_lmdb, compress_loop},
};

const LMDB_PATH: &str = "./db";

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

// 6 gates on 4 wires
fn not_to_g57(wires: u8, a: u8) -> Vec<[u8; 3]> {
    let mut chosen = [a; 3];

    for i in 0..3 {
        loop {
            let w = fastrand::u8(0..wires);
            if !chosen[..=i].contains(&w) {
                chosen[i] = w;
                break;
            }
        }
    }

    let [b, c, d] = chosen;

    vec![
        [a, c, d],
        [c, d, b],
        [a, c, b],
        [a, d, c],
        [c, d, b],
        [a, b, c],
    ]
}

// 6 gate nontrivial on 4 wires
// TODO: choose different ones each time
fn id_to_g57(wires: u8, a: u8) -> Vec<[u8; 3]> {
    // one of the wires will be the active one
    // Choose three random values from the range 0..wires
    let mut chosen = [a; 3];

    for i in 0..3 {
        loop {
            let w = fastrand::u8(0..wires);
            if !chosen[..=i].contains(&w) {
                chosen[i] = w;
                break;
            }
        }
    }

    let [b, c, d] = chosen;
    vec![
        [d, c, a],
        [a, b, c],
        [c, b, a],
        [d, a, c],
        [c, b, a],
        [a, b, c],
    ]
}

fn big_tof(n: u8, active: u8, controls: Range<u8>) -> CircuitSeq {
    // Build the staircase
    let mut empty: Vec<u8> = (0..n + 2).collect();

    // Delete index `active`, and all indices in `controls` from the vector
    empty.retain(|x| *x != active && !controls.contains(x));

    // Shuffle the remaining indices
    fastrand::shuffle(&mut empty);

    let ctrl: Vec<u8> = controls.collect();

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

fn key_to_gates(wires: u8, key: usize) -> CircuitSeq {
    let mut c = CircuitSeq { gates: vec![] };

    for i in 0..wires {
        let gates = if (key >> i) & 1 == 0 {
            id_to_g57(wires, i)
        } else {
            not_to_g57(wires, i)
        };

        c.gates.extend(gates);
    }

    c
}

fn main() {
    let args = Args::parse();

    let env = Environment::new()
        .set_max_readers(10000)
        .set_max_dbs(256 + 40)
        .set_map_size(800 * 1024 * 1024 * 1024)
        .open(Path::new(LMDB_PATH))
        .expect("Failed to open database.");

    let shard_dbs = open_shard_dbs(&env);

    let n = args.wires;

    println!("{:?}", args);

    let key_bits = (usize::BITS as u32 - args.key.leading_zeros()) as u8;
    assert!(key_bits <= n);

    let b = n.div_ceil(2);

    let mut pf = key_to_gates(n, args.key)
        .concat(&big_tof(n, n, 0..b))
        .concat(&big_tof(n, n + 1, b..n + 1))
        .concat(&big_tof(n, n, 0..b))
        .concat(&big_tof(n, n + 1, b..n + 1))
        .concat(&key_to_gates(n, args.key));

    pf.canonicalize();

    println!("{}", pf.repr());

    println!("len = {}", pf.gates.len());

    let comp = compress_loop(&pf, pf.max_wire() + 1, &env, &shard_dbs, 6, 0, 0, ".");

    println!("{}", comp.repr());

    println!("0 => {}", pf.evaluate_256(0.into()));
    println!("{} => {}", args.key, pf.evaluate_256(args.key.into()));

    for _ in 0..10 {
        let r = if n < 64 {
            fastrand::usize(0..(1usize << n))
        } else {
            fastrand::usize(..)
        };
        println!(" {} => {}", r, pf.evaluate_256(r.into()));
    }
}
