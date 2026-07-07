// Traced fsplit run for visualization: generate a random g57 circuit, run the
// float-and-split process for --steps mutation events, and dump a JSON trace
// (full circuit snapshot per step, with the changed gates marked).
//
//   fsplit_trace --wires 32 --gates 64 --k-max 4 --steps 100 --seed 1 --out trace.json
use clap::Parser;
use local_mixing::postmix::engine::{Engine, Params};
use local_mixing::postmix::xgate::XGate;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::fmt::Write as _;

#[derive(Parser, Debug)]
#[command(name = "fsplit_trace")]
struct Args {
    #[arg(long, default_value_t = 32)]
    wires: u16,
    #[arg(long, default_value_t = 64)]
    gates: usize,
    #[arg(long, default_value_t = 4)]
    k_max: usize,
    #[arg(long, default_value_t = 2)]
    split_damp: usize,
    #[arg(long, default_value_t = 100)]
    steps: usize,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    #[arg(long)]
    out: String,
}

fn json_str(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

fn main() {
    let args = Args::parse();
    let mut rng = StdRng::seed_from_u64(args.seed.wrapping_add(0x9e3779b97f4a7c15));
    let gates: Vec<XGate> = (0..args.gates)
        .map(|_| {
            loop {
                let a = rng.random_range(0..args.wires);
                let x = rng.random_range(0..args.wires);
                let y = rng.random_range(0..args.wires);
                if a != x && a != y && x != y {
                    return XGate::from_g57([a, x, y]);
                }
            }
        })
        .collect();

    let params = Params {
        k_max: args.k_max,
        split_damp: args.split_damp,
        size_bound: usize::MAX >> 1,
        seed: args.seed,
        report_every: usize::MAX >> 1,
        verify_every: 16,
        ..Params::default()
    };
    let mut engine = Engine::new(gates, params);
    engine.trace_on(args.steps);
    engine.run();
    engine.global_check();
    let trace = engine.trace.take().unwrap();
    println!(
        "[fsplit_trace] {} steps captured, final size {} gates (verified equal)",
        trace.len() - 1,
        engine.arena.len()
    );

    let mut out = String::new();
    write!(
        out,
        "{{\"num_wires\":{},\"k_max\":{},\"split_damp\":{},\"seed\":{},\"steps\":[",
        args.wires, args.k_max, args.split_damp, args.seed
    )
    .unwrap();
    for (si, st) in trace.iter().enumerate() {
        if si > 0 {
            out.push(',');
        }
        write!(out, "{{\"kind\":{},\"label\":{},\"gates\":[", json_str(st.kind), json_str(&st.label)).unwrap();
        for (gi, (t, c, ctrls)) in st.gates.iter().enumerate() {
            if gi > 0 {
                out.push(',');
            }
            write!(out, "[{},{},[", t, c).unwrap();
            for (ci, (w, p)) in ctrls.iter().enumerate() {
                if ci > 0 {
                    out.push(',');
                }
                write!(out, "[{},{}]", w, p).unwrap();
            }
            out.push_str("]]");
        }
        out.push_str("],\"moves\":[");
        for (mi, (f, t)) in st.moves.iter().enumerate() {
            if mi > 0 {
                out.push(',');
            }
            write!(out, "[{},{}]", f, t).unwrap();
        }
        out.push_str("],");
        match st.colliding {
            Some(c) => write!(out, "\"colliding\":{},", c).unwrap(),
            None => out.push_str("\"colliding\":null,"),
        }
        let ints = |v: &[usize]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
        write!(out, "\"new\":[{}],\"core\":[{}]}}", ints(&st.new_idx), ints(&st.core_idx)).unwrap();
    }
    out.push_str("]}");
    std::fs::write(&args.out, out).expect("write trace json");
    println!("[fsplit_trace] wrote {}", args.out);
}
