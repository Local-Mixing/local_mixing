// Float-and-split post-mixing driver. Takes a mixed g57 circuit (or an mpmct1
// circuit from a previous run), grows it to --size-bound by floating gates to
// their collision points and splitting them past colliding gates under the
// --k-max control cap, then floats every gate to a uniform random position in
// its commutable box. Output is mpmct1 (mixed-polarity multi-controlled
// Toffolis + surviving g57s).
//
// Examples:
//   fsplit --gen-random 24,200 --k-max 4 --size-bound 600 --seed 7 --output out.txt
//   fsplit --input mixed.txt --size-bound 1700000 --output mixed_fsplit.txt
use clap::Parser;
use local_mixing::postmix::engine::{Engine, Params, StopReason};
use local_mixing::postmix::format;
use local_mixing::postmix::xgate::{XGate, max_wire};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

#[derive(Parser, Debug)]
#[command(name = "fsplit")]
struct Args {
    /// Input circuit file (g57 base-83 by default, see --input-format)
    #[arg(long)]
    input: Option<String>,
    /// Input format: g57 | mpmct1
    #[arg(long, default_value = "g57")]
    input_format: String,
    /// Generate a random g57 circuit "wires,gates" instead of reading --input
    #[arg(long)]
    gen_random: Option<String>,
    /// Output file (mpmct1 format)
    #[arg(long)]
    output: Option<String>,
    /// Max controls per gate (K)
    #[arg(long, default_value_t = 4)]
    k_max: usize,
    /// Width-damping offset D: a gate with c controls splits with probability
    /// min(2^-(c-D), 1); at or below D it always splits
    #[arg(long, default_value_t = 2)]
    split_damp: usize,
    /// Window (gates) to search each side of a g57 start for a g57 collision
    /// partner to shoot toward; 0 disables g57xg57 targeting
    #[arg(long, default_value_t = 64)]
    g57_target_window: usize,
    /// Stop splitting when the circuit reaches this many gates (default 2x input)
    #[arg(long)]
    size_bound: Option<usize>,
    /// Candidates sampled per episode; the largest one-directional float wins
    #[arg(long, default_value_t = 64)]
    candidates: usize,
    /// Cap on the selection walk when scoring candidates
    #[arg(long, default_value_t = 4096)]
    walk_cap: usize,
    /// Max worklist pops per episode
    #[arg(long, default_value_t = 10_000)]
    episode_cap: usize,
    /// Global sampled equality check every N episodes
    #[arg(long, default_value_t = 64)]
    verify_every: usize,
    /// Progress report every N episodes
    #[arg(long, default_value_t = 1000)]
    report_every: usize,
    /// Stop after this many consecutive no-action episodes
    #[arg(long, default_value_t = 200)]
    saturation_patience: usize,
    /// Disable the per-split exhaustive local verification
    #[arg(long, default_value_t = false)]
    no_local_verify: bool,
    /// Skip the final uniform float pass
    #[arg(long, default_value_t = false)]
    skip_final_float: bool,
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

fn main() {
    let args = Args::parse();

    let gates: Vec<XGate> = if let Some(spec) = &args.gen_random {
        let (n, m) = spec
            .split_once(',')
            .and_then(|(a, b)| Some((a.trim().parse::<u16>().ok()?, b.trim().parse::<usize>().ok()?)))
            .expect("--gen-random expects \"wires,gates\"");
        assert!(n >= 3, "need at least 3 wires");
        let mut rng = StdRng::seed_from_u64(args.seed.wrapping_add(0x9e3779b97f4a7c15));
        (0..m)
            .map(|_| {
                loop {
                    let a = rng.random_range(0..n);
                    let x = rng.random_range(0..n);
                    let y = rng.random_range(0..n);
                    if a != x && a != y && x != y {
                        return XGate::from_g57([a, x, y]);
                    }
                }
            })
            .collect()
    } else {
        let input = args.input.as_deref().expect("--input or --gen-random required");
        match args.input_format.as_str() {
            "g57" => format::read_g57_file(input).expect("read g57 circuit"),
            "mpmct1" => format::read_mpmct(input).expect("read mpmct1 circuit").0,
            other => panic!("unknown --input-format {other}"),
        }
    };

    let num_wires = max_wire(&gates) as usize + 1;
    let input_len = gates.len();
    let size_bound = args.size_bound.unwrap_or(input_len * 2);
    println!(
        "[fsplit] input: {} gates, {} wires; k_max={} split_damp={} size_bound={} seed={}",
        input_len, num_wires, args.k_max, args.split_damp, size_bound, args.seed
    );

    let params = Params {
        k_max: args.k_max,
        split_damp: args.split_damp,
        g57_target_window: args.g57_target_window,
        size_bound,
        candidates: args.candidates,
        walk_cap: args.walk_cap,
        episode_cap: args.episode_cap,
        verify_every: args.verify_every,
        report_every: args.report_every,
        saturation_patience: args.saturation_patience,
        local_verify: !args.no_local_verify,
        seed: args.seed,
    };
    let mut engine = Engine::new(gates, params);

    // On-demand snapshot via an environment signal: set FSPLIT_DUMP_FLAG to a
    // path; `touch` that path from outside and the run writes the current
    // circuit to FSPLIT_DUMP_OUT (default "<output>.snapshot.txt", or
    // "fsplit_snapshot.txt" if no --output) and continues. FSPLIT_DUMP_EVERY
    // sets the check cadence in episodes (default 200).
    if let Ok(flag) = std::env::var("FSPLIT_DUMP_FLAG") {
        if !flag.is_empty() {
            let out = std::env::var("FSPLIT_DUMP_OUT").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| {
                args.output.as_deref().map(|o| format!("{o}.snapshot.txt")).unwrap_or_else(|| "fsplit_snapshot.txt".to_string())
            });
            let every: usize = std::env::var("FSPLIT_DUMP_EVERY").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
            println!("[fsplit] dump signal armed: touch {flag} -> snapshot to {out} (checked every {every} episodes)");
            engine.enable_dump(flag, out, every);
        }
    }

    let t0 = std::time::Instant::now();
    let stop = engine.run();
    let split_secs = t0.elapsed().as_secs_f64();
    engine.report();
    println!(
        "[fsplit] splitting done in {:.1}s: {} ({} -> {} gates, {} g57s remain)",
        split_secs,
        match stop {
            StopReason::SizeBound => "size bound reached",
            StopReason::Saturated => "K-SATURATED before size bound",
            StopReason::TraceFull => "trace cap reached",
        },
        input_len,
        engine.arena.len(),
        engine.remaining_g57(),
    );

    if !args.skip_final_float {
        let t1 = std::time::Instant::now();
        let (moved, disp) = engine.final_float();
        engine.global_check();
        println!(
            "[fsplit] final float: {} gates moved, {} total displacement, {:.1}s (verified)",
            moved,
            disp,
            t1.elapsed().as_secs_f64()
        );
    }

    if let Some(out) = &args.output {
        let final_gates = engine.arena.to_vec();
        format::write_mpmct(out, &final_gates, num_wires).expect("write output");
        println!("[fsplit] wrote {} gates to {}", final_gates.len(), out);
    } else {
        println!("[fsplit] no --output given; result discarded after verification");
    }
}
