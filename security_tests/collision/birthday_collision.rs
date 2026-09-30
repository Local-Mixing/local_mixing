//! Birthday / brute-force collision finder for the 2λ→λ hash
//!
//!   H(x) = C(0^{pad} ‖ x)_{out}
//!
//! Usage:
//!   birthday_collision CIRCUIT.g57
//!       [--pad 32] [--in-bits 64] [--out-bits 32]
//!       [--samples N] [--seed S] [--out report.json]
//!       [--bench-evals N]

use local_mixing::circuit::{CircuitSeq, Gate};
use primitive_types::U256;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::process;
use std::time::Instant;

struct Args {
    circuit: String,
    pad: usize,
    in_bits: usize,
    out_bits: usize,
    samples: u64,
    seed: u64,
    out: Option<String>,
    bench_evals: Option<u64>,
}

fn usage() -> ! {
    eprintln!(
        "usage: birthday_collision CIRCUIT.g57 [--pad 32] [--in-bits 64] \
         [--out-bits 32] [--samples N] [--seed S] [--out report.json] \
         [--bench-evals N]"
    );
    process::exit(2);
}

fn parse_usize(s: &str, name: &str) -> usize {
    s.parse().unwrap_or_else(|_| {
        eprintln!("invalid {name}: {s}");
        process::exit(2);
    })
}

fn parse_u64(s: &str, name: &str) -> u64 {
    s.parse().unwrap_or_else(|_| {
        eprintln!("invalid {name}: {s}");
        process::exit(2);
    })
}

fn parse_args() -> Args {
    let mut argv = env::args().skip(1);
    let circuit = match argv.next() {
        Some(c) if !c.starts_with('-') => c,
        _ => usage(),
    };
    let mut args = Args {
        circuit,
        pad: 32,
        in_bits: 64,
        out_bits: 32,
        samples: 300_000,
        seed: 1,
        out: None,
        bench_evals: None,
    };
    while let Some(flag) = argv.next() {
        match flag.as_str() {
            "--pad" => args.pad = parse_usize(&argv.next().unwrap_or_else(|| usage()), "pad"),
            "--in-bits" => {
                args.in_bits = parse_usize(&argv.next().unwrap_or_else(|| usage()), "in-bits")
            }
            "--out-bits" => {
                args.out_bits = parse_usize(&argv.next().unwrap_or_else(|| usage()), "out-bits")
            }
            "--samples" => {
                args.samples = parse_u64(&argv.next().unwrap_or_else(|| usage()), "samples")
            }
            "--seed" => args.seed = parse_u64(&argv.next().unwrap_or_else(|| usage()), "seed"),
            "--out" => args.out = Some(argv.next().unwrap_or_else(|| usage())),
            "--bench-evals" => {
                args.bench_evals = Some(parse_u64(
                    &argv.next().unwrap_or_else(|| usage()),
                    "bench-evals",
                ))
            }
            _ => usage(),
        }
    }
    args
}

fn hash(gates: &[[u16; 3]], x: u64, in_bits: usize, out_bits: usize) -> u64 {
    let in_mask = if in_bits >= 64 {
        u64::MAX
    } else {
        (1u64 << in_bits) - 1
    };
    let out_mask = if out_bits >= 64 {
        u64::MAX
    } else {
        (1u64 << out_bits) - 1
    };
    let input = U256::from(x & in_mask);
    let out = Gate::evaluate_index_list_256(input, gates);
    out.low_u64() & out_mask
}

fn main() {
    let args = parse_args();
    let width = args.pad + args.in_bits;
    if !(1..=64).contains(&args.in_bits) || !(1..=64).contains(&args.out_bits) {
        eprintln!("require 1..=64 in-bits and 1..=64 out-bits");
        process::exit(2);
    }
    if width > 256 {
        eprintln!("pad + in-bits must fit in 256 wires for this tool");
        process::exit(2);
    }

    let raw = fs::read(&args.circuit).unwrap_or_else(|e| {
        eprintln!("failed to read {}: {e}", args.circuit);
        process::exit(2);
    });
    let circuit = CircuitSeq::from_bytes(&raw);

    if let Some(bench_n) = args.bench_evals {
        let mut x = 1u64;
        for _ in 0..1000 {
            x = hash(&circuit.gates, x, args.in_bits, args.out_bits).wrapping_add(1);
        }
        let t0 = Instant::now();
        for i in 0..bench_n {
            let _ = hash(&circuit.gates, i, args.in_bits, args.out_bits);
        }
        let secs = t0.elapsed().as_secs_f64().max(1e-12);
        let meval = bench_n as f64 / secs / 1e6;
        println!(
            "{{\"mode\":\"bench\",\"evals\":{bench_n},\"seconds\":{secs:.6},\
             \"meval_per_sec\":{meval:.4},\"workers\":1,\"out_bits\":{},\
             \"in_bits\":{},\"pad\":{},\"width\":{width}}}",
            args.out_bits, args.in_bits, args.pad
        );
        eprintln!("[bench] {bench_n} scalar evals in {secs:.3}s ({meval:.2} Meval/s)");
        return;
    }

    eprintln!(
        "[birthday] circuit={} gates={} H: {}-bit → {}-bit \
         (zero-pad high {} of {} wires); samples={} seed={}",
        args.circuit,
        circuit.gates.len(),
        args.in_bits,
        args.out_bits,
        args.pad,
        width,
        args.samples,
        args.seed
    );

    fastrand::seed(args.seed);
    let mut seen: HashMap<u64, u64> = HashMap::with_capacity((args.samples as usize).min(1 << 20));
    let t0 = Instant::now();
    let mut evaluated = 0u64;
    let mut collision: Option<(u64, u64, u64)> = None;

    for _ in 0..args.samples {
        let x = if args.in_bits == 64 {
            fastrand::u64(..)
        } else {
            fastrand::u64(..(1u64 << args.in_bits))
        };
        let h = hash(&circuit.gates, x, args.in_bits, args.out_bits);
        evaluated += 1;
        if let Some(x1) = seen.insert(h, x) {
            if x1 != x {
                collision = Some((x1, x, h));
                break;
            }
        }
    }

    let elapsed = t0.elapsed();
    let Some((x1, x2, h)) = collision else {
        eprintln!(
            "[birthday] no collision in {evaluated} samples ({elapsed:.2?}; {:.1} Meval/s)",
            evaluated as f64 / elapsed.as_secs_f64().max(1e-9) / 1e6
        );
        process::exit(1);
    };

    let h1 = hash(&circuit.gates, x1, args.in_bits, args.out_bits);
    let h2 = hash(&circuit.gates, x2, args.in_bits, args.out_bits);
    assert_eq!(h1, h);
    assert_eq!(h2, h);
    assert_ne!(x1, x2);

    let report = format!(
        "{{\n  \"circuit\": \"{}\",\n  \"gates\": {},\n  \"width\": {},\n  \
         \"pad_bits\": {},\n  \"in_bits\": {},\n  \"out_bits\": {},\n  \
         \"seed\": {},\n  \"samples_evaluated\": {},\n  \
         \"x1_hex\": \"0x{:016x}\",\n  \"x2_hex\": \"0x{:016x}\",\n  \
         \"digest_hex\": \"0x{:016x}\",\n  \"elapsed_secs\": {:.6}\n}}\n",
        args.circuit.replace('\\', "\\\\").replace('"', "\\\""),
        circuit.gates.len(),
        width,
        args.pad,
        args.in_bits,
        args.out_bits,
        args.seed,
        evaluated,
        x1,
        x2,
        h,
        elapsed.as_secs_f64()
    );
    print!("{report}");
    eprintln!(
        "[birthday] collision after {evaluated} samples ({elapsed:.2?}): \
         H(0x{x1:016x}) = H(0x{x2:016x}) = 0x{h:016x}"
    );
    if let Some(path) = args.out {
        fs::write(&path, &report).unwrap_or_else(|e| {
            eprintln!("failed to write {path}: {e}");
            process::exit(2);
        });
    }
}
