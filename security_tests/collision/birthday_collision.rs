//! Birthday / brute-force collision finder for the 2n→n hash
//!
//!   H(x) = C(0^{pad} ‖ x)_{out}
//!
//! built from a reversible G57 circuit C on `pad + in_bits` wires. Defaults
//! match the 96-wire experiment: `pad=32`, `in_bits=64`, `out_bits=32`.
//! High `pad` input wires are fixed to zero; the message occupies the low
//! `in_bits` wires; the digest is the low `out_bits` output wires (wire 0 =
//! LSB).
//!
//! Usage:
//!   birthday_collision CIRCUIT.g57
//!       [--pad 32] [--in-bits 64] [--out-bits 32]
//!       [--samples N] [--seed S] [--out report.json]
//!
//! Exit 0 on a verified collision, 1 if the sample budget is exhausted
//! without one, 2 on usage / I/O errors.

use local_mixing::circuit::{CircuitSeq, Gate};
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
}

fn usage() -> ! {
    eprintln!(
        "usage: birthday_collision CIRCUIT.g57 [--pad 32] [--in-bits 64] \
         [--out-bits 32] [--samples N] [--seed S] [--out report.json]"
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
            _ => usage(),
        }
    }
    args
}

fn hash(gates: &[[u16; 3]], x: u64, in_bits: usize, out_bits: usize) -> u32 {
    let in_mask = if in_bits == 64 {
        u64::MAX
    } else {
        (1u64 << in_bits) - 1
    };
    let out_mask = if out_bits == 32 {
        u32::MAX as u128
    } else {
        (1u128 << out_bits) - 1
    };
    let out = Gate::evaluate_index_list_128((x & in_mask) as u128, gates);
    (out & out_mask) as u32
}

fn main() {
    let args = parse_args();
    let width = args.pad + args.in_bits;
    if !(1..=64).contains(&args.in_bits) || !(1..=32).contains(&args.out_bits) {
        eprintln!("require 1..=64 in-bits and 1..=32 out-bits");
        process::exit(2);
    }
    if width > 128 {
        eprintln!("pad + in-bits must fit in 128 wires for this tool");
        process::exit(2);
    }

    let raw = fs::read(&args.circuit).unwrap_or_else(|e| {
        eprintln!("failed to read {}: {e}", args.circuit);
        process::exit(2);
    });
    let circuit = CircuitSeq::from_bytes(&raw);
    let wires = circuit
        .gates
        .iter()
        .flat_map(|g| g.iter().copied())
        .max()
        .map(|w| w as usize + 1)
        .unwrap_or(0);
    if wires > width {
        eprintln!(
            "warning: circuit touches wire {} but hash layout uses {width} wires",
            wires - 1
        );
    }
    if wires < width {
        // Random generation with exactly `width` wires still may not touch
        // every index; require only that generation targeted this width.
        eprintln!(
            "[birthday] note: highest touched wire is {} (layout width {width})",
            wires.saturating_sub(1)
        );
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
    let mut seen: HashMap<u32, u64> = HashMap::with_capacity((args.samples as usize).min(1 << 20));
    let t0 = Instant::now();
    let mut evaluated = 0u64;
    let mut collision: Option<(u64, u64, u32)> = None;

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
         \"digest_hex\": \"0x{:08x}\",\n  \"elapsed_secs\": {:.6}\n}}\n",
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
         H(0x{x1:016x}) = H(0x{x2:016x}) = 0x{h:08x}"
    );
    if let Some(path) = args.out {
        fs::write(&path, &report).unwrap_or_else(|e| {
            eprintln!("failed to write {path}: {e}");
            process::exit(2);
        });
    }
}
