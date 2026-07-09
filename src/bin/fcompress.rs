// Final compression pass for fmix/fsplit output (see postmix/compress.rs):
// gather same-target gates that can float together, XOR-reduce each group in
// ESOP/ANF space, re-emit surviving cubes as consecutive XGates, iterate to a
// fixed point. Deterministic and attacker-computable, so it cannot weaken the
// hiding; the compressed size is the honest "effective size" of the artifact.
//
// Optional dead-cone pruning for gadgetized circuits, where equality is
// required only on designated output wires: --live-wires upper-half (or
// lower-half, or an explicit list "0-255,300"). Default all wires live.
//
// Example:
//   fcompress --input mixed_fmix.txt --output mixed_fcmp.txt
//   fcompress --input gadget.txt --output gadget_fcmp.txt --live-wires upper-half
use clap::Parser;
use local_mixing::postmix::compress::{CompressParams, compress, lits_of};
use local_mixing::postmix::format;
use local_mixing::postmix::xgate::{XGate, eval_lanes, max_wire};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

#[derive(Parser, Debug)]
#[command(name = "fcompress")]
struct Args {
    /// Input circuit file
    #[arg(long)]
    input: String,
    /// Input format: mpmct1 | g57
    #[arg(long, default_value = "mpmct1")]
    input_format: String,
    /// Output file (mpmct1); omit for a dry run (verify + report only)
    #[arg(long)]
    output: Option<String>,
    /// Wires whose final value must be preserved: all | upper-half |
    /// lower-half | explicit list like "0-255,300,510"
    #[arg(long, default_value = "all")]
    live_wires: String,
    #[arg(long, default_value_t = 10)]
    max_iters: usize,
    #[arg(long, default_value_t = 64)]
    group_cap: usize,
    #[arg(long, default_value_t = 24)]
    anf_support_cap: usize,
    /// Rounds of the 64-lane sampled global check against the input
    #[arg(long, default_value_t = 64)]
    verify_rounds: usize,
    /// Disable the per-group exhaustive/sampled verification
    #[arg(long, default_value_t = false)]
    no_local_verify: bool,
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

fn parse_live(spec: &str, wires: usize) -> Option<Vec<bool>> {
    match spec {
        "all" => None,
        "upper-half" => {
            let mut v = vec![false; wires];
            v[wires / 2..].iter_mut().for_each(|b| *b = true);
            Some(v)
        }
        "lower-half" => {
            let mut v = vec![false; wires];
            v[..wires / 2].iter_mut().for_each(|b| *b = true);
            Some(v)
        }
        list => {
            let mut v = vec![false; wires];
            for part in list.split(',') {
                let part = part.trim();
                if let Some((a, b)) = part.split_once('-') {
                    let (a, b): (usize, usize) =
                        (a.parse().expect("live range"), b.parse().expect("live range"));
                    v[a..=b].iter_mut().for_each(|x| *x = true);
                } else {
                    v[part.parse::<usize>().expect("live wire")] = true;
                }
            }
            Some(v)
        }
    }
}

fn main() {
    let args = Args::parse();
    let (gates, file_wires): (Vec<XGate>, usize) = match args.input_format.as_str() {
        "mpmct1" => format::read_mpmct(&args.input).expect("read mpmct1 circuit"),
        "g57" => {
            let g = format::read_g57_file(&args.input).expect("read g57 circuit");
            let w = max_wire(&g) as usize + 1;
            (g, w)
        }
        other => panic!("unknown --input-format {other}"),
    };
    let wires = file_wires.max(max_wire(&gates) as usize + 1);
    let live = parse_live(&args.live_wires, wires);
    let n_live = live.as_ref().map_or(wires, |v| v.iter().filter(|&&b| b).count());
    println!(
        "[fcompress] input: {} gates ({} lits), {} wires ({} live); max_iters={} group_cap={} anf_cap={} seed={}",
        gates.len(),
        lits_of(&gates),
        wires,
        n_live,
        args.max_iters,
        args.group_cap,
        args.anf_support_cap,
        args.seed
    );

    let params = CompressParams {
        live_out: live.clone(),
        max_iters: args.max_iters,
        group_cap: args.group_cap,
        anf_support_cap: args.anf_support_cap,
        local_verify: !args.no_local_verify,
        seed: args.seed,
    };
    let original = gates.clone();
    let t0 = std::time::Instant::now();
    let (out, rep) = compress(gates, wires, &params);
    let secs = t0.elapsed().as_secs_f64();

    // Sampled global check against the untouched input, on live wires only.
    let mut rng = StdRng::seed_from_u64(args.seed ^ 0xC0FFEE);
    for round in 0..args.verify_rounds {
        let sa: Vec<u64> = (0..wires).map(|_| rng.random::<u64>()).collect();
        let mut sb = sa.clone();
        let mut sa = sa;
        eval_lanes(original.iter(), &mut sa);
        eval_lanes(out.iter(), &mut sb);
        for w in 0..wires {
            if live.as_ref().is_none_or(|v| v[w]) {
                assert_eq!(
                    sa[w], sb[w],
                    "global check failed on wire {w} (round {round})"
                );
            }
        }
    }
    println!(
        "[fcompress] done in {:.1}s (verified {} rounds x64 lanes): gates {} -> {} ({:.1}%), lits {} -> {} ({:.1}%), live_dropped={}",
        secs,
        args.verify_rounds,
        rep.gates_in,
        rep.gates_out,
        100.0 * rep.gates_out as f64 / rep.gates_in.max(1) as f64,
        rep.lits_in,
        rep.lits_out,
        100.0 * rep.lits_out as f64 / rep.lits_in.max(1) as f64,
        rep.liveness_dropped
    );

    if let Some(path) = &args.output {
        format::write_mpmct(path, &out, wires).expect("write output");
        println!("[fcompress] wrote {} gates to {}", out.len(), path);
    } else {
        println!("[fcompress] no --output given; result discarded after verification");
    }
}
