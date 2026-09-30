//! Generate a seeded random reversible G57 circuit and its mpmct1 twin.
//!
//! Usage: gen_collision_circuit OUT_PREFIX N GATES SEED
//! Writes OUT_PREFIX.g57, OUT_PREFIX.mpmct1, and OUT_PREFIX.meta.json.

use local_mixing::circuit::formats::write_mpmct;
use local_mixing::circuit::random_circuit;
use local_mixing::circuit::xgate::XGate;
use std::env;
use std::fs;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 5 {
        eprintln!("usage: gen_collision_circuit OUT_PREFIX N GATES SEED");
        process::exit(2);
    }
    let prefix = &args[1];
    let n: usize = args[2].parse().unwrap_or_else(|_| {
        eprintln!("invalid wire count");
        process::exit(2);
    });
    let m: usize = args[3].parse().unwrap_or_else(|_| {
        eprintln!("invalid gate count");
        process::exit(2);
    });
    let seed: u64 = args[4].parse().unwrap_or_else(|_| {
        eprintln!("invalid seed");
        process::exit(2);
    });
    if !(3..=1024).contains(&n) {
        eprintln!("wire count must be in 3..=1024");
        process::exit(2);
    }

    fastrand::seed(seed);
    let circuit = random_circuit(n, m);
    let g57_path = format!("{prefix}.g57");
    let mpmct_path = format!("{prefix}.mpmct1");
    let meta_path = format!("{prefix}.meta.json");

    fs::write(&g57_path, circuit.repr()).unwrap_or_else(|e| {
        eprintln!("write {g57_path}: {e}");
        process::exit(2);
    });
    let xgates: Vec<XGate> = circuit.gates.iter().map(|g| XGate::from_g57(*g)).collect();
    write_mpmct(&mpmct_path, &xgates, n).unwrap_or_else(|e| {
        eprintln!("write {mpmct_path}: {e}");
        process::exit(2);
    });

    // Infer λ-layout from wire count: width = 3λ with pad=λ, in=2λ, out=λ.
    let (lambda, pad_bits, in_bits, out_bits) = if n % 3 == 0 {
        let lambda = n / 3;
        (lambda, lambda, 2 * lambda, lambda)
    } else {
        (32usize, 32, 64, 32)
    };
    let meta = format!(
        "{{\n  \"wires\": {n},\n  \"gates\": {m},\n  \"seed\": {seed},\n  \
         \"lambda\": {lambda},\n  \
         \"g57\": \"{g57_path}\",\n  \"mpmct1\": \"{mpmct_path}\",\n  \
         \"hash_layout\": {{\n    \"pad_bits\": {pad_bits},\n    \"in_bits\": {in_bits},\n    \
         \"out_bits\": {out_bits},\n    \"description\": \"H(x)=C(0^λ || x)_λ for λ={lambda} (2λ-bit inputs, λ-bit outputs; wire 0 = LSB)\"\n  }}\n}}\n"
    );
    fs::write(&meta_path, meta).unwrap_or_else(|e| {
        eprintln!("write {meta_path}: {e}");
        process::exit(2);
    });
    println!("[gen] wrote {g57_path}, {mpmct_path}, {meta_path} (n={n}, m={m}, seed={seed})");
}
