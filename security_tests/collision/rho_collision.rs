//! Parallel distinguished-point (van Oorschot–Wiener) collision search for
//!
//!   H(x) = C(0^{pad} ‖ x)_{out}
//!
//! on a reversible G57 circuit. Defaults target the n=64 experiment:
//! `pad=64`, message width 128 (search uses a 64-bit subspace), `out=64`
//! on a 192-wire circuit. Iteration is `f(s) = H(embed(s))` with
//! `embed(s) = s` as a 128-bit message (high message bits zero). A collision
//! of `f` is a collision of `H`.
//!
//! Bit-sliced walks evaluate 64 trails per circuit pass; Rayon runs several
//! such bundles across CPU cores. Distinguished points (low `--dp-bits` of
//! the digest clear) are stored in a shared table; a repeated DP with a
//! different start is replayed to a concrete preimage pair.
//!
//! Usage:
//!   rho_collision CIRCUIT.g57
//!       [--pad 64] [--out-bits 64] [--dp-bits 16]
//!       [--workers N] [--seed S] [--max-evals N]
//!       [--out report.json]

use local_mixing::circuit::{CircuitSeq, Gate, lane_state_len};
use primitive_types::U256;
use rayon::prelude::*;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::env;
use std::fs;
use std::process;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
struct TrailRec {
    start: u64,
    steps: u64,
}

struct Args {
    circuit: String,
    pad: usize,
    out_bits: usize,
    dp_bits: u32,
    workers: usize,
    seed: u64,
    max_evals: u64,
    out: Option<String>,
    self_check: bool,
}

fn usage() -> ! {
    eprintln!(
        "usage: rho_collision CIRCUIT.g57 [--pad 64] [--out-bits 64] \
         [--dp-bits 16] [--workers N] [--seed S] [--max-evals N] \
         [--out report.json] [--self-check]"
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
        pad: 64,
        out_bits: 64,
        dp_bits: 16,
        workers: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4),
        seed: 1,
        max_evals: 100_000_000_000,
        out: None,
        self_check: false,
    };
    while let Some(flag) = argv.next() {
        match flag.as_str() {
            "--pad" => args.pad = parse_usize(&argv.next().unwrap_or_else(|| usage()), "pad"),
            "--out-bits" => {
                args.out_bits = parse_usize(&argv.next().unwrap_or_else(|| usage()), "out-bits")
            }
            "--dp-bits" => {
                args.dp_bits = parse_u64(&argv.next().unwrap_or_else(|| usage()), "dp-bits") as u32
            }
            "--workers" => {
                args.workers = parse_usize(&argv.next().unwrap_or_else(|| usage()), "workers")
            }
            "--seed" => args.seed = parse_u64(&argv.next().unwrap_or_else(|| usage()), "seed"),
            "--max-evals" => {
                args.max_evals = parse_u64(&argv.next().unwrap_or_else(|| usage()), "max-evals")
            }
            "--out" => args.out = Some(argv.next().unwrap_or_else(|| usage())),
            "--self-check" => args.self_check = true,
            _ => usage(),
        }
    }
    args
}

fn out_mask(out_bits: usize) -> u64 {
    if out_bits == 64 {
        u64::MAX
    } else {
        (1u64 << out_bits) - 1
    }
}

fn hash_scalar(gates: &[[u16; 3]], s: u64, out_bits: usize) -> u64 {
    let out = Gate::evaluate_index_list_256(U256::from(s), gates);
    out.low_u64() & out_mask(out_bits)
}

fn hash_lanes(
    gates: &[[u16; 3]],
    states: &[u64; 64],
    out_bits: usize,
    wire_slots: usize,
) -> [u64; 64] {
    let mut lane_state = vec![0u64; wire_slots];
    for bit in 0..64 {
        let mut word = 0u64;
        for lane in 0..64 {
            word |= ((states[lane] >> bit) & 1) << lane;
        }
        lane_state[bit] = word;
    }
    Gate::eval_lanes_index_list(gates, &mut lane_state);
    let mut out = [0u64; 64];
    let bits = out_bits.min(64);
    let mask = out_mask(out_bits);
    for bit in 0..bits {
        let word = lane_state[bit];
        for lane in 0..64 {
            out[lane] |= ((word >> lane) & 1) << bit;
        }
    }
    for lane in 0..64 {
        out[lane] &= mask;
    }
    out
}

/// Given two trails that end at the same distinguished point, recover x1≠x2
/// with f(x1)=f(x2).
fn reconstruct(
    gates: &[[u16; 3]],
    out_bits: usize,
    a: TrailRec,
    b: TrailRec,
) -> Option<(u64, u64, u64)> {
    let f = |s: u64| hash_scalar(gates, s, out_bits);
    let mut x = a.start;
    let mut y = b.start;
    if a.steps > b.steps {
        for _ in 0..(a.steps - b.steps) {
            x = f(x);
        }
    } else if b.steps > a.steps {
        for _ in 0..(b.steps - a.steps) {
            y = f(y);
        }
    }
    if x == y {
        // One start lies on the other trail; not an independent collision.
        return None;
    }
    // Step until images collide; predecessors are the preimages.
    for _ in 0..a.steps.max(b.steps) + 2 {
        let nx = f(x);
        let ny = f(y);
        if nx == ny {
            return if x != y { Some((x, y, nx)) } else { None };
        }
        x = nx;
        y = ny;
    }
    None
}

fn self_check(gates: &[[u16; 3]], out_bits: usize, width: usize) {
    let slots = lane_state_len(width);
    let mut states = [0u64; 64];
    for (i, s) in states.iter_mut().enumerate() {
        *s = (i as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(0x1234_5678_9ABC_DEF0);
    }
    let lane = hash_lanes(gates, &states, out_bits, slots);
    let mut mism = 0usize;
    for i in 0..64 {
        let sc = hash_scalar(gates, states[i], out_bits);
        if sc != lane[i] {
            mism += 1;
            if mism <= 4 {
                eprintln!(
                    "[self-check] lane {i}: scalar={sc:016x} lanes={:016x}",
                    lane[i]
                );
            }
        }
    }
    if mism != 0 {
        eprintln!("[self-check] FAILED: {mism}/64 mismatches");
        process::exit(2);
    }
    eprintln!("[self-check] lane/scalar OK (64 lanes, out_bits={out_bits})");
}

fn main() {
    let args = parse_args();
    if !(1..=64).contains(&args.out_bits) {
        eprintln!("out-bits must be in 1..=64");
        process::exit(2);
    }
    if args.dp_bits >= args.out_bits as u32 {
        eprintln!("dp-bits must be < out-bits");
        process::exit(2);
    }
    let width = args.pad + 128;
    let raw = fs::read(&args.circuit).unwrap_or_else(|e| {
        eprintln!("failed to read {}: {e}", args.circuit);
        process::exit(2);
    });
    let circuit = CircuitSeq::from_bytes(&raw);
    let touched = circuit
        .gates
        .iter()
        .flat_map(|g| g.iter().copied())
        .max()
        .map(|w| w as usize + 1)
        .unwrap_or(0);
    let wire_slots = lane_state_len(width.max(touched));

    if args.self_check {
        self_check(&circuit.gates, args.out_bits, width.max(touched));
    }

    let dp_mask = if args.dp_bits == 0 {
        0u64
    } else {
        (1u64 << args.dp_bits) - 1
    };
    let max_trail = 40u64 << args.dp_bits;

    eprintln!(
        "[rho] circuit={} gates={} layout pad={} message_subspace=64 out={} \
         dp_bits={} workers={} seed={} max_evals={}",
        args.circuit,
        circuit.gates.len(),
        args.pad,
        args.out_bits,
        args.dp_bits,
        args.workers,
        args.seed,
        args.max_evals
    );

    let gates = Arc::new(circuit.gates.clone());
    let table: Arc<Mutex<HashMap<u64, TrailRec>>> = Arc::new(Mutex::new(HashMap::new()));
    let found = Arc::new(AtomicBool::new(false));
    let evals = Arc::new(AtomicU64::new(0));
    let dps = Arc::new(AtomicU64::new(0));
    let dp_conflicts = Arc::new(AtomicU64::new(0));
    let reconstruct_fail = Arc::new(AtomicU64::new(0));
    let result: Arc<Mutex<Option<(u64, u64, u64)>>> = Arc::new(Mutex::new(None));
    let t0 = Instant::now();

    let worker_seeds: Vec<u64> = (0..args.workers)
        .map(|i| args.seed.wrapping_add(i as u64 * 0x9E37_79B9_7F4A_7C15))
        .collect();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.workers)
        .build()
        .expect("thread pool");

    pool.install(|| {
        worker_seeds.par_iter().for_each(|&worker_seed| {
            let mut rng_state = worker_seed;
            let mut next_u64 = || {
                rng_state = rng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = rng_state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^ (z >> 31)
            };

            let mut states = [0u64; 64];
            let mut starts = [0u64; 64];
            let mut steps = [0u64; 64];
            for i in 0..64 {
                let s = next_u64();
                states[i] = s;
                starts[i] = s;
                steps[i] = 0;
            }

            let mut local_evals = 0u64;
            while !found.load(Ordering::Relaxed) {
                if evals.load(Ordering::Relaxed) >= args.max_evals {
                    break;
                }
                let digests = hash_lanes(&gates, &states, args.out_bits, wire_slots);
                local_evals += 64;
                if local_evals >= 65536 {
                    evals.fetch_add(local_evals, Ordering::Relaxed);
                    local_evals = 0;
                }

                for lane in 0..64 {
                    let h = digests[lane];
                    steps[lane] += 1;
                    if (h & dp_mask) == 0 {
                        dps.fetch_add(1, Ordering::Relaxed);
                        let rec = TrailRec {
                            start: starts[lane],
                            steps: steps[lane],
                        };
                        let conflict = {
                            let mut tbl = table.lock().unwrap();
                            match tbl.entry(h) {
                                Entry::Occupied(e) => {
                                    let prev = *e.get();
                                    if prev.start != rec.start {
                                        Some(prev)
                                    } else {
                                        None
                                    }
                                }
                                Entry::Vacant(v) => {
                                    v.insert(rec);
                                    None
                                }
                            }
                        };
                        if let Some(prev) = conflict {
                            dp_conflicts.fetch_add(1, Ordering::Relaxed);
                            match reconstruct(&gates, args.out_bits, prev, rec) {
                                Some((x1, x2, digest)) if x1 != x2 => {
                                    let h1 = hash_scalar(&gates, x1, args.out_bits);
                                    let h2 = hash_scalar(&gates, x2, args.out_bits);
                                    if h1 == h2 && h1 == digest {
                                        *result.lock().unwrap() = Some((x1, x2, digest));
                                        found.store(true, Ordering::Relaxed);
                                        evals.fetch_add(local_evals, Ordering::Relaxed);
                                        return;
                                    }
                                    reconstruct_fail.fetch_add(1, Ordering::Relaxed);
                                }
                                _ => {
                                    reconstruct_fail.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }
                        let s = next_u64();
                        states[lane] = s;
                        starts[lane] = s;
                        steps[lane] = 0;
                    } else if steps[lane] >= max_trail {
                        let s = next_u64();
                        states[lane] = s;
                        starts[lane] = s;
                        steps[lane] = 0;
                    } else {
                        states[lane] = h;
                    }
                }

                let e = evals.load(Ordering::Relaxed);
                if e > 0 && e % 25_000_000 == 0 {
                    let elapsed = t0.elapsed().as_secs_f64().max(1e-9);
                    eprintln!(
                        "[rho] evals={e} ({:.2} Meval/s) dps={} conflicts={} \
                         recon_fail={} table={} elapsed={:.1}s",
                        e as f64 / elapsed / 1e6,
                        dps.load(Ordering::Relaxed),
                        dp_conflicts.load(Ordering::Relaxed),
                        reconstruct_fail.load(Ordering::Relaxed),
                        table.lock().unwrap().len(),
                        elapsed
                    );
                }
            }
            evals.fetch_add(local_evals, Ordering::Relaxed);
        });
    });

    let elapsed = t0.elapsed();
    let evaluated = evals.load(Ordering::Relaxed);
    match *result.lock().unwrap() {
        None => {
            eprintln!(
                "[rho] no collision after {evaluated} evals ({elapsed:.2?}; {:.2} Meval/s); \
                 dps={} conflicts={} recon_fail={}",
                evaluated as f64 / elapsed.as_secs_f64().max(1e-9) / 1e6,
                dps.load(Ordering::Relaxed),
                dp_conflicts.load(Ordering::Relaxed),
                reconstruct_fail.load(Ordering::Relaxed)
            );
            process::exit(1);
        }
        Some((x1, x2, digest)) => {
            let report = format!(
                "{{\n  \"circuit\": \"{}\",\n  \"gates\": {},\n  \"width\": {},\n  \
                 \"pad_bits\": {},\n  \"in_bits\": 128,\n  \"out_bits\": {},\n  \
                 \"search\": \"rho_dp_subspace64\",\n  \"dp_bits\": {},\n  \
                 \"seed\": {},\n  \"samples_evaluated\": {},\n  \
                 \"x1_hex\": \"0x{:032x}\",\n  \"x2_hex\": \"0x{:032x}\",\n  \
                 \"digest_hex\": \"0x{:016x}\",\n  \"elapsed_secs\": {:.6}\n}}\n",
                args.circuit.replace('\\', "\\\\").replace('"', "\\\""),
                gates.len(),
                width,
                args.pad,
                args.out_bits,
                args.dp_bits,
                args.seed,
                evaluated,
                x1 as u128,
                x2 as u128,
                digest,
                elapsed.as_secs_f64()
            );
            print!("{report}");
            eprintln!(
                "[rho] collision after {evaluated} evals ({elapsed:.2?}): \
                 H(0x{x1:016x}) = H(0x{x2:016x}) = 0x{digest:016x}"
            );
            if let Some(path) = args.out {
                fs::write(&path, &report).unwrap_or_else(|e| {
                    eprintln!("failed to write {path}: {e}");
                    process::exit(2);
                });
            }
        }
    }
}
