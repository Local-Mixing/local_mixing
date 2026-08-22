//! Manual end-to-end acceptance check for a delivered pipeline artifact: does the
//! final circuit still compute the original circuit on the zero slice?
//!
//! The contract a sliced-sandwich gadget ships under is
//!   inject x on wires 0..n, every other wire 0  ->  C(x) on wires n..2n,
//! and it must survive gadgetization, mixing and growth unchanged. fmix
//! verifies each chain link against ITS OWN input, so correctness of the
//! delivered circuit is otherwise only a transitive argument across three
//! processes; this closes it directly against the original C.
//!
//! Usage: verify_zero_slice <C.g57> <final.mpmct1> <n> [samples=200] [seed=1]
//!
//! Prints one PASS/FAIL line. Requires the circuit to fit in 1024 wires.

use local_mixing::circuit::U1024;
use local_mixing::circuit::xgate::eval_u1024;
use local_mixing::engine::format::{read_g57_file, read_mpmct};
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

fn mask_bits(bits: usize) -> U1024 {
    if bits >= 1024 {
        U1024::MAX
    } else {
        (U1024::one() << bits) - U1024::one()
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: verify_zero_slice <C.g57> <final.mpmct1> <n> [samples] [seed]");
        std::process::exit(2);
    }
    let c = read_g57_file(&a[1]).expect("read C (g57)");
    let (g, wires) = read_mpmct(&a[2]).expect("read final (mpmct1)");
    let n: usize = a[3].parse().expect("n");
    let samples: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(200);
    let seed: u64 = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(1);
    assert!(
        wires <= 1024,
        "verify_zero_slice supports at most 1024 wires"
    );

    // C is an n-wire g57 circuit; the sandwich puts its answer on the SECOND
    // half of its 2n wires, and the gadget preserves that on its low 2n.
    let low = mask_bits(n);
    let answer = mask_bits(2 * n) ^ low; // wires n..2n
    let mut rng = StdRng::seed_from_u64(seed);
    let mut bytes = [0u8; 128];
    let mut bad = 0usize;
    for i in 0..samples {
        rng.fill_bytes(&mut bytes);
        // x on wires 0..n, EVERY other wire zero: that is the slice.
        let x = U1024::from_little_endian(&bytes) & low;
        let want = eval_u1024(&c, x) & low;
        let got = (eval_u1024(&g, x) & answer) >> n;
        if got != want {
            if bad < 3 {
                eprintln!("[verify] sample {i}: got != C(x)");
            }
            bad += 1;
        }
    }
    if bad == 0 {
        println!(
            "[verify] PASS: {} samples, C(x) on wires {}..{} of the {}-wire final on the zero slice",
            samples,
            n,
            2 * n,
            wires
        );
    } else {
        println!("[verify] FAIL: {bad}/{samples} samples mismatched");
        std::process::exit(1);
    }
}
