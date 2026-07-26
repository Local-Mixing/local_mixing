//! One-off: regenerate the inner C of a sandwich_compare run (C is derived
//! deterministically from the fastrand seed and was never written to disk)
//! and write it in g57 text format for hmap_affine.
//!
//! Usage: regen_sandwich_c [n] [m_C] [seed] [out]   (defaults 128 3000 1
//! circuits/new_sliced_sandwich_C_n128.g57.txt) — must mirror
//! sandwich_compare's construction order exactly: seed fastrand, draw C
//! first (D is drawn after C, so C is unaffected).

use local_mixing::random::random_data::random_circuit;

fn main() {
    let mut args = std::env::args().skip(1);
    let n: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(128);
    let m_c: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(3000);
    let seed: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let out = args
        .next()
        .unwrap_or_else(|| "circuits/new_sliced_sandwich_C_n128.g57.txt".to_string());
    fastrand::seed(seed);
    let c = random_circuit(n, m_c);
    // Compact g57 encoding, the only form CircuitSeq::from_string parses:
    // per wire, `~`-overflow prefixes then the base-83 wire char.
    const MAP: &str =
        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_=+[]{}<>?";
    let chars: Vec<char> = MAP.chars().collect();
    let mut s = String::with_capacity(c.gates.len() * 6);
    for g in &c.gates {
        for &w in g.iter() {
            for _ in 0..(w as usize / 83) {
                s.push('~');
            }
            s.push(chars[w as usize % 83]);
        }
        s.push(';');
    }
    std::fs::write(&out, s).expect("write C");
    println!("[regen] wrote {} gates ({}w, seed {}) to {}", c.gates.len(), n, seed, out);
}
