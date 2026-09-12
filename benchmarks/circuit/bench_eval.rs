// Timing harness for the circuit evaluation paths: the g57 base-83 parser and
// its fixed-width kernels (`CircuitSeq::evaluate_*`), the mpmct1 parser, and
// the general XGate evaluators (`apply_u64` / `apply_u1024` / `apply_lanes`).
//
// Reports ns/gate so numbers are comparable across circuit sizes. The XGate
// arms deliberately run on circuits that CONTAIN X gates (k=0), CNOTs (k=1),
// complemented g57s (k=2, comp=1) and wider conjunctions, because that is the
// gate-width mix a post-fsplit / post-fmix mpmct1 artifact actually has: a
// harness that only ever sees g57 measures the wrong loop.
//
// Examples:
//   bench_eval --synthetic --wires 128 --gates 2000000
//   bench_eval --g57 rantestn128m800/rantestn128m800round11.txt --wires 128
//   bench_eval --mpmct1 runs/blind_prod_s3/final.mpmct1
use std::time::Instant;

use clap::Parser;
use local_mixing::circuit::xgate::{XGate, eval_lanes, eval_u64, eval_u1024, max_wire};
use local_mixing::circuit::{CircuitSeq, Gate, U1024, lane_state_len};
use local_mixing::engine::format;
use primitive_types::U256;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::{Rng, RngCore};

#[derive(Parser, Debug)]
#[command(name = "bench_eval")]
struct Args {
    /// g57 base-83 circuit file (parser + CircuitSeq kernels + XGate kernels)
    #[arg(long)]
    g57: Option<String>,
    /// mpmct1 circuit file (parser + XGate kernels)
    #[arg(long)]
    mpmct1: Option<String>,
    /// Benchmark a synthetic circuit instead of a file
    #[arg(long)]
    synthetic: bool,
    /// Wires for --synthetic, and the evaluation width for --g57
    #[arg(long, default_value_t = 128)]
    wires: usize,
    /// Gates for --synthetic
    #[arg(long, default_value_t = 2_000_000)]
    gates: usize,
    /// Timed repetitions per arm (best time wins)
    #[arg(long, default_value_t = 3)]
    reps: usize,
    /// Inputs for the probably_equal arm (0 = skip)
    #[arg(long, default_value_t = 1000)]
    equal_inputs: usize,
    #[arg(long, default_value_t = 0xB0BA_CAFE)]
    seed: u64,
}

fn best<T>(reps: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut lo = f64::INFINITY;
    let mut last = f();
    for _ in 0..reps {
        let t = Instant::now();
        last = f();
        lo = lo.min(t.elapsed().as_secs_f64());
    }
    (lo, last)
}

fn line(label: &str, secs: f64, gates: usize, lanes: usize) {
    let per_gate = secs / gates as f64 * 1e9;
    let per_app = per_gate / lanes as f64;
    if lanes == 1 {
        println!("[bench] {label:28} {secs:9.4} s   {per_gate:8.3} ns/gate");
    } else {
        println!(
            "[bench] {label:28} {secs:9.4} s   {per_gate:8.3} ns/gate   \
             {per_app:7.4} ns/gate-lane ({lanes} lanes)"
        );
    }
}

/// A gate-width mix that mirrors post-split mpmct1 artifacts: mostly narrow
/// (X / CNOT / g57), with a tail of wider conjunctions.
fn synthetic_xgates(wires: usize, gates: usize, rng: &mut StdRng) -> Vec<XGate> {
    let mut out = Vec::with_capacity(gates);
    while out.len() < gates {
        let target = rng.random_range(0..wires) as u16;
        let roll: u32 = rng.random_range(0..100);
        let k = match roll {
            0..=7 => 0,   // X gate
            8..=27 => 1,  // CNOT
            28..=74 => 2, // g57 / CCNOT shaped
            75..=91 => 3,
            92..=97 => 4,
            _ => 6,
        };
        if k == 0 {
            out.push(XGate::x_gate(target));
            continue;
        }
        let mut lits: Vec<(u16, bool)> = Vec::with_capacity(k);
        for _ in 0..k {
            let mut w = rng.random_range(0..wires) as u16;
            while w == target {
                w = rng.random_range(0..wires) as u16;
            }
            lits.push((w, rng.random_bool(0.5)));
        }
        let comp = k == 2 && rng.random_bool(0.5);
        if let Some(mut g) = XGate::conj(target, lits) {
            g.comp = comp;
            out.push(g);
        }
    }
    out
}

fn synthetic_g57(wires: usize, gates: usize, rng: &mut StdRng) -> CircuitSeq {
    // ~8% of gates are spelled as X gates (control wires equal), matching the
    // X-gate density the XGate arms use.
    let mut v = Vec::with_capacity(gates);
    for _ in 0..gates {
        let a = rng.random_range(0..wires) as u16;
        let mut x = rng.random_range(0..wires) as u16;
        while x == a {
            x = rng.random_range(0..wires) as u16;
        }
        let y = if rng.random_range(0..100) < 8 {
            x // control wires equal => X gate on `a`
        } else {
            let mut y = rng.random_range(0..wires) as u16;
            while y == a {
                y = rng.random_range(0..wires) as u16;
            }
            y
        };
        v.push([a, x, y]);
    }
    CircuitSeq { gates: v }
}

fn bench_g57_kernels(c: &CircuitSeq, wires: usize, reps: usize) {
    let n = c.gates.len();
    let mut bytes = [0u8; 128];
    rand::rng().fill_bytes(&mut bytes);

    let x64 = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    let x128 = u128::from_le_bytes(bytes[..16].try_into().unwrap());
    let x256 = U256::from_little_endian(&bytes[..32]);
    let x1024 = U1024::from_little_endian(&bytes);

    if wires <= 64 {
        let (t, out) = best(reps, || c.evaluate(x64 as usize));
        line("g57 evaluate(usize)", t, n, 1);
        std::hint::black_box(out);
    }
    if wires <= 128 {
        let (t, out) = best(reps, || c.evaluate_128(x128));
        line("g57 evaluate_128", t, n, 1);
        std::hint::black_box(out);
    }
    if wires <= 256 {
        let (t, out) = best(reps, || c.evaluate_256(x256));
        line("g57 evaluate_256", t, n, 1);
        std::hint::black_box(out);
    }
    let (t, out) = best(reps, || c.evaluate_1024(x1024));
    line("g57 evaluate_1024", t, n, 1);
    std::hint::black_box(out);

    let len = lane_state_len(wires.max(c.max_wire() + 1));
    let mut rng = StdRng::seed_from_u64(11);
    let base: Vec<u64> = (0..len).map(|_| rng.random()).collect();
    let (t, _) = best(reps, || {
        let mut st = base.clone();
        Gate::eval_lanes_index_list(&c.gates, &mut st);
        st[0]
    });
    line("g57 eval_lanes (64)", t, n, 64);
}

fn bench_xgate_kernels(gates: &[XGate], wires: usize, reps: usize) {
    let n = gates.len();
    let mut bytes = [0u8; 128];
    rand::rng().fill_bytes(&mut bytes);

    if wires <= 64 {
        let x64 = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        let (t, out) = best(reps, || eval_u64(gates, x64));
        line("xgate eval_u64", t, n, 1);
        std::hint::black_box(out);
    }
    if wires <= 1024 {
        let x1024 = U1024::from_little_endian(&bytes);
        let (t, out) = best(reps, || eval_u1024(gates, x1024));
        line("xgate eval_u1024", t, n, 1);
        std::hint::black_box(out);
    }

    let mut rng = StdRng::seed_from_u64(7);
    let base: Vec<u64> = (0..wires).map(|_| rng.random()).collect();
    let (t, _) = best(reps, || {
        let mut st = base.clone();
        eval_lanes(gates.iter(), &mut st);
        st[0]
    });
    line("xgate eval_lanes (64)", t, n, 64);
}

fn main() {
    let args = Args::parse();
    let mut rng = StdRng::seed_from_u64(args.seed);

    if let Some(path) = args.g57.as_deref() {
        let raw = std::fs::read(path).expect("read g57 circuit");
        println!("[bench] g57 file {path} ({} bytes)", raw.len());
        let text = String::from_utf8(raw).expect("g57 circuit is UTF-8");
        let (t, c) = best(args.reps, || CircuitSeq::from_string(&text));
        line("g57 parse (from_string)", t, c.gates.len(), 1);
        // Faithfulness check on real input: re-encoding the parse must
        // reproduce the file byte for byte.
        assert_eq!(
            c.repr().as_bytes(),
            text.trim().as_bytes(),
            "parse/repr round trip must be byte-exact"
        );
        println!("[bench] gates={} max_wire={}", c.gates.len(), c.max_wire());
        bench_g57_kernels(&c, args.wires, args.reps);

        let (t, xs) = best(args.reps, || {
            c.gates
                .iter()
                .map(|&g| XGate::from_g57(g))
                .collect::<Vec<_>>()
        });
        line("g57 -> XGate lift", t, xs.len(), 1);
        // Cross-check the two evaluators against each other on the real
        // circuit: the g57 kernel and the XGate kernel share no code, so
        // agreement here is independent evidence that both the parser and the
        // limb kernels are right.
        {
            let mut probe = [0u8; 128];
            rand::rng().fill_bytes(&mut probe);
            let x = U1024::from_little_endian(&probe);
            assert_eq!(
                c.evaluate_1024(x),
                eval_u1024(xs.iter(), x),
                "g57 kernel and lifted-XGate kernel disagree"
            );
        }
        bench_xgate_kernels(&xs, max_wire(xs.iter()) as usize + 1, args.reps);

        if args.equal_inputs > 0 {
            let t = Instant::now();
            let r = c.probably_equal(&c, args.wires, args.equal_inputs);
            let secs = t.elapsed().as_secs_f64();
            assert!(r.is_ok(), "self-equality must hold");
            line(
                &format!("g57 probably_equal x{}", args.equal_inputs),
                secs,
                c.gates.len() * args.equal_inputs,
                1,
            );
        }
    }

    if let Some(path) = args.mpmct1.as_deref() {
        let meta = std::fs::metadata(path).expect("stat mpmct1 circuit");
        println!("[bench] mpmct1 file {path} ({} bytes)", meta.len());
        let (t, (gates, wires)) =
            best(args.reps, || format::read_mpmct(path).expect("read mpmct1"));
        line("mpmct1 parse (read_mpmct)", t, gates.len(), 1);
        let mw = max_wire(gates.iter()) as usize + 1;
        println!(
            "[bench] gates={} wires={wires} max_wire+1={mw}",
            gates.len()
        );
        let widths = gates.iter().fold([0usize; 8], |mut acc, g| {
            acc[g.width().min(7)] += 1;
            acc
        });
        println!(
            "[bench] width histogram k=0..6,7+: {:?}  comp=1: {}",
            widths,
            gates.iter().filter(|g| g.comp).count()
        );
        bench_xgate_kernels(&gates, mw.max(wires), args.reps);
    }

    if args.synthetic {
        println!(
            "[bench] synthetic wires={} gates={}",
            args.wires, args.gates
        );
        let c = synthetic_g57(args.wires, args.gates, &mut rng);
        let text = c.repr();
        println!("[bench] g57 repr {} bytes", text.len());
        let (t, parsed) = best(args.reps, || CircuitSeq::from_string(&text));
        line("g57 parse (from_string)", t, parsed.gates.len(), 1);
        assert_eq!(parsed.gates, c.gates, "parser must round-trip repr()");
        bench_g57_kernels(&c, args.wires, args.reps);

        let xs = synthetic_xgates(args.wires, args.gates, &mut rng);
        let widths = xs.iter().fold([0usize; 8], |mut acc, g| {
            acc[g.width().min(7)] += 1;
            acc
        });
        println!("[bench] xgate width histogram k=0..6,7+: {:?}", widths);
        bench_xgate_kernels(&xs, args.wires, args.reps);
    }
}
