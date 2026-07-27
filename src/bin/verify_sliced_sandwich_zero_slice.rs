//! Independently verify a delivered sliced-sandwich product-share circuit.
//!
//! Contract:
//! input x on wires 0..n and zero on every other physical wire; output
//! C(x) on wires n..2n.

use clap::Parser;
use local_mixing::{
    circuit::circuit::U1024,
    postmix::{
        format::{read_g57_file, read_mpmct},
        xgate::eval_u1024,
    },
};
use rand::{RngCore, SeedableRng, rngs::StdRng};

#[derive(Debug, Parser)]
#[command(name = "verify_sliced_sandwich_zero_slice")]
struct Args {
    #[arg(long)]
    source: String,
    #[arg(long)]
    circuit: String,
    #[arg(long, default_value_t = 128)]
    n: usize,
    #[arg(long, default_value_t = 8192)]
    samples: usize,
    #[arg(long, default_value_t = 1)]
    seed: u64,
}

fn bit_mask(bits: usize) -> U1024 {
    if bits >= 1024 {
        U1024::MAX
    } else {
        (U1024::one() << bits) - U1024::one()
    }
}

fn main() {
    let args = Args::parse();
    let source = read_g57_file(&args.source).expect("read source C");
    let (circuit, wires) = read_mpmct(&args.circuit).expect("read gadget circuit");
    assert!(wires <= 1024, "checker supports at most 1024 wires");
    assert!(2 * args.n <= wires, "answer block is outside circuit");

    let low = bit_mask(args.n);
    let answer = low << args.n;
    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut bytes = [0u8; 128];

    // Include structured endpoint cases before independently random samples.
    let structured = [
        U1024::zero(),
        low,
        U1024::from(0xaaaa_aaaa_aaaa_aaaau64) & low,
        U1024::from(0x5555_5555_5555_5555u64) & low,
    ];
    let mut checked = 0usize;
    for input in structured.into_iter().chain((0..args.samples).map(|_| {
        rng.fill_bytes(&mut bytes);
        U1024::from_little_endian(&bytes) & low
    })) {
        let expected = eval_u1024(&source, input) & low;
        let got = (eval_u1024(&circuit, input) & answer) >> args.n;
        assert_eq!(
            got, expected,
            "sliced-sandwich contract mismatch at case {checked}"
        );
        checked += 1;
    }

    println!("n={}", args.n);
    println!("source_gates={}", source.len());
    println!("circuit_gates={}", circuit.len());
    println!("circuit_wires={wires}");
    println!("random_samples={}", args.samples);
    println!("structured_samples={}", structured.len());
    println!("checked_cases={checked}");
    println!("seed={}", args.seed);
    println!("fixed_zero_wires={}..{}", args.n, wires);
    println!("answer_wires={}..{}", args.n, 2 * args.n);
    println!("slice_ok=true");
}
