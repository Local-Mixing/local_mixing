// Build a point function (or an identity that looks like one!)

use std::{fs::File, io::Write, ops::Range, path::Path};

use clap::Parser;
use itertools::chain;
use lmdb::Environment;
use local_mixing::{
    circuit::CircuitSeq,
    open_shard_dbs,
    replace::replace::compress_loop,
};

const LMDB_PATH: &str = "./db";

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    #[arg(short = 'n', default_value_t = 128)]
    wires: u16,

    #[arg(short, long, default_value_t = 42)]
    key: usize,

    #[arg(short, long, default_value_t = false, help="whether to compute an identity")]
    identity: bool,

    #[arg(short, long, default_value_t = false, help="whether to compress")]
    compress: bool,

    #[arg(short, long, help="file to save final circuit to")]
    output: Option<String>,
}

fn tof_to_g57(wires: u16, g: &[u16; 3]) -> Vec<[u16; 3]> {
    // Let r be a random wire from the range 0..wires, NOT including any of the values in g.
    let mut r = fastrand::u16(0..wires);
    while g.contains(&r) {
        r = fastrand::u16(0..wires);
    }

    let active = g[0] as u16;
    let ctrl1 = g[1] as u16;
    let ctrl2 = g[2] as u16;

    vec![
        [active, ctrl1, r],
        [ctrl2, ctrl1, r],
        [active, ctrl1, ctrl2],
        [ctrl2, ctrl1, r],
    ]
}

// 6 gates on 4 wires
fn not_to_g57(wires: u16, a: u16) -> Vec<[u16; 3]> {
    let mut chosen = [a; 3];

    for i in 0..3 {
        loop {
            let w = fastrand::u16(0..wires);
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
fn id_to_g57(wires: u16, a: u16) -> Vec<[u16; 3]> {
    // one of the wires will be the active one
    // Choose three random values from the range 0..wires
    let mut chosen = [a; 3];

    for i in 0..3 {
        loop {
            let w = fastrand::u16(0..wires);
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

fn big_tof(n: u16, active: u16, controls: Range<u16>) -> CircuitSeq {
    // Build the staircase
    let mut empty: Vec<u16> = (0..n + 2).collect();

    // Delete index `active`, and all indices in `controls` from the vector
    empty.retain(|x| *x != active && !controls.contains(x));

    // Shuffle the remaining indices
    fastrand::shuffle(&mut empty);

    let ctrl: Vec<u16> = controls.collect();

    let stair: Vec<[u16; 3]> = (1..(ctrl.len() - 2))
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

fn key_to_gates(wires: u16, key: usize) -> CircuitSeq {
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

    let n = args.wires;

    println!("{:?}", args);

    let key_bits = (usize::BITS as u32 - args.key.leading_zeros()) as u16;
    assert!(key_bits <= n);

    let b = n.div_ceil(2);

    let pf_gen = |k| -> CircuitSeq {
        key_to_gates(n, k)
            .concat(&big_tof(n, n, 0..b))
            .concat(&big_tof(n, n + 1, b..n + 1))
            .concat(&big_tof(n, n, 0..b))
            .concat(&big_tof(n, n + 1, b..n + 1))
            .concat(&key_to_gates(n, k))
    };

    let mut pf = pf_gen(args.key).concat(&pf_gen(args.key + if args.identity { 0 } else { 1 }));

    pf.canonicalize();

    println!("{}", pf.repr());

    println!("len = {}", pf.gates.len());

    if args.compress {
        let env = Environment::new()
            .set_max_readers(10000)
            .set_max_dbs(256 + 40)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new(LMDB_PATH))
            .expect("Failed to open database.");

        let shard_dbs = open_shard_dbs(&env);
        let comp = compress_loop(&pf, pf.max_wire() + 1, &env, &shard_dbs, 6, 0, 0, ".");

        println!("{}", comp.repr());
        let pl = pf.gates.len();
        let cl = comp.gates.len();
        println!(
            "Size: {} => {} ({}%)",
            pl,
            cl,
            ((cl as f64) / (pl as f64) * 100.0).round()
        );

        if args.output.is_some() {
            let mut f = File::create(args.output.unwrap()).expect("failed to open");
            f.write_all(comp.repr().as_bytes()).expect("failed to write");
            let _ = f.write(b"\n");
        }
    } else {
        if args.output.is_some() {
            let mut f = File::create(args.output.unwrap()).expect("failed to open");
            f.write_all(pf.repr().as_bytes()).expect("failed to write");
            let _ = f.write(b"\n");
        }
    }

    println!("0 => {}", pf.evaluate_256(0.into()));

    let mut r = None;
    while r != Some(args.key) {
        let s = r.unwrap_or(args.key);
        r = Some(pf.evaluate(s.into()));
        println!("{} => {}", s, r.unwrap());
    }

    for _ in 0..10 {
        let r = if n < 64 {
            fastrand::usize(0..(1usize << n))
        } else {
            fastrand::usize(..)
        };
        println!(" {} => {}", r, pf.evaluate_256(r.into()));
    }
}
