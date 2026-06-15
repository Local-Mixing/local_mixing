use local_mixing::circuit::{CircuitSeq, Gate};
use primitive_types::U512;
use rand::RngCore;
use std::{env, fs::File, io::Write};

fn usage(program: &str) -> ! {
    eprintln!(
        "usage:\n  {program} verify <circuit> [samples]\n  {program} eval <circuit> <hex128>\n  {program} samples <circuit> <count> <out.tsv>\n  {program} compare <reference> <candidate> [samples]"
    );
    std::process::exit(2);
}

fn parse_hex128(mut s: &str) -> U512 {
    if let Some(rest) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        s = rest;
    }
    if s.len() > 32 {
        panic!("hex value is wider than 128 bits");
    }
    U512::from_str_radix(s, 16).expect("bad hex value")
}

fn hex128(x: U512) -> String {
    format!("0x{:032x}", x.low_u128())
}

fn eval_full(circuit: &CircuitSeq, x: U512, y: U512, z: U512) -> U512 {
    let input = x | (y << 128) | (z << 256);
    Gate::evaluate_index_list_512(input, &circuit.gates)
}

fn middle(output: U512) -> U512 {
    (output >> 128) & ((U512::one() << 128) - U512::one())
}

fn projected_c(circuit: &CircuitSeq, x: U512) -> U512 {
    middle(eval_full(circuit, x, U512::zero(), U512::zero()))
}

fn random_u128() -> U512 {
    let mut bytes = [0u8; 16];
    rand::rng().fill_bytes(&mut bytes);
    U512::from_little_endian(&bytes)
}

fn load(path: &str) -> CircuitSeq {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
    CircuitSeq::from_string(&text)
}

fn verify(path: &str, samples: usize) {
    let circuit = load(path);
    let mask = (U512::one() << 128) - U512::one();
    for i in 0..samples {
        let x = random_u128();
        let y = random_u128();
        let z0 = random_u128();
        let z1 = random_u128();
        let c0 = projected_c(&circuit, x);
        let m0 = middle(eval_full(&circuit, x, U512::zero(), z0));
        let m1 = middle(eval_full(&circuit, x, U512::zero(), z1));
        let my = middle(eval_full(&circuit, x, y, z0));
        if c0 != m0 || c0 != m1 || my != ((y ^ c0) & mask) {
            panic!(
                "projection check failed at sample {i}: x={} y={} z0={} z1={}",
                hex128(x),
                hex128(y),
                hex128(z0),
                hex128(z1)
            );
        }
    }
    println!("verified_samples {samples}");
    println!("projected_function C(x) = middle(TDP0(x, 0, 0))");
}

fn write_samples(path: &str, count: usize, out_path: &str) {
    let circuit = load(path);
    let mut out = File::create(out_path).unwrap_or_else(|e| panic!("failed to create {out_path}: {e}"));
    writeln!(out, "x\tc_x").expect("failed to write header");
    for _ in 0..count {
        let x = random_u128();
        let cx = projected_c(&circuit, x);
        writeln!(out, "{}\t{}", hex128(x), hex128(cx)).expect("failed to write sample");
    }
    println!("wrote_samples {count}");
    println!("path {out_path}");
}

fn compare(reference_path: &str, candidate_path: &str, samples: usize) {
    let reference = load(reference_path);
    let candidate = load(candidate_path);
    for i in 0..samples {
        let x = random_u128();
        let expected = projected_c(&reference, x);
        let actual = projected_c(&candidate, x);
        if expected != actual {
            panic!(
                "projection compare failed at sample {i}: x={} ref={} candidate={}",
                hex128(x),
                hex128(expected),
                hex128(actual)
            );
        }
    }
    println!("projection_compare_ok {samples}");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        usage(&args[0]);
    }
    match args[1].as_str() {
        "verify" => {
            let samples = args.get(3).map_or(Ok(1000usize), |s| s.parse()).unwrap_or_else(|_| usage(&args[0]));
            verify(&args[2], samples);
        }
        "eval" => {
            if args.len() != 4 {
                usage(&args[0]);
            }
            let circuit = load(&args[2]);
            println!("{}", hex128(projected_c(&circuit, parse_hex128(&args[3]))));
        }
        "samples" => {
            if args.len() != 5 {
                usage(&args[0]);
            }
            let count = args[3].parse().unwrap_or_else(|_| usage(&args[0]));
            write_samples(&args[2], count, &args[4]);
        }
        "compare" => {
            if !(4..=5).contains(&args.len()) {
                usage(&args[0]);
            }
            let samples = args.get(4).map_or(Ok(1000usize), |s| s.parse()).unwrap_or_else(|_| usage(&args[0]));
            compare(&args[2], &args[3], samples);
        }
        _ => usage(&args[0]),
    }
}
