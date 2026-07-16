// Standalone driver for the float-and-split engine.  This is the bridge from
// the legacy, all-g57 SSS artifact to the heterogeneous mpmct1 representation
// consumed by fmix/fcompress and the mixed-gate database compressor.
use clap::Parser;
use local_mixing::postmix::compress::lits_of;
use local_mixing::postmix::engine::{Engine, Params, StopReason};
use local_mixing::postmix::format;
use local_mixing::postmix::reassemble::is_structural_g57;
use local_mixing::postmix::source::{self, MIXED_SOURCE};
use local_mixing::postmix::xgate::{XGate, max_wire};

#[derive(Parser, Debug)]
#[command(name = "fsplit")]
struct Args {
    /// Input circuit.
    #[arg(long)]
    input: String,
    /// Input format: g57 (the base-83 SSS format) or mpmct1.
    #[arg(long, default_value = "g57")]
    input_format: String,
    /// Heterogeneous mpmct1 output.
    #[arg(long)]
    output: String,
    /// Optional fsource1 sidecar aligned with the heterogeneous output.  Each
    /// row names the original input-gate parent or records `mixed` after a
    /// cross-parent rewrite.
    #[arg(long)]
    sources_out: Option<String>,
    /// Absolute gate-count stop bound. Overrides --growth when supplied.
    #[arg(long)]
    target_size: Option<usize>,
    /// Relative gate-count stop bound when --target-size is omitted.
    #[arg(long, default_value_t = 1.25)]
    growth: f64,
    #[arg(long, default_value_t = 4)]
    k_max: usize,
    #[arg(long, default_value_t = 2)]
    split_damp: usize,
    #[arg(long, default_value_t = 64)]
    g57_target_window: usize,
    #[arg(long, default_value_t = 64)]
    candidates: usize,
    #[arg(long, default_value_t = 4096)]
    walk_cap: usize,
    #[arg(long, default_value_t = 10_000)]
    episode_cap: usize,
    #[arg(long, default_value_t = 50_000)]
    verify_every: usize,
    #[arg(long, default_value_t = 10_000)]
    report_every: usize,
    #[arg(long, default_value_t = 10_000)]
    saturation_patience: usize,
    /// Skip exhaustive verification of each local rewrite (global sampled
    /// verification at checkpoints and on exit remains enabled).
    #[arg(long, default_value_t = false)]
    no_local_verify: bool,
    /// Run the optional final uniform float pass before writing.
    #[arg(long, default_value_t = false)]
    final_float: bool,
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

fn main() {
    let args = Args::parse();
    if args.target_size.is_none() {
        assert!(
            args.growth.is_finite() && args.growth >= 1.0,
            "--growth must be finite and >= 1"
        );
    }

    let (gates, file_wires): (Vec<XGate>, usize) = match args.input_format.as_str() {
        "mpmct1" => format::read_mpmct(&args.input).expect("read mpmct1 circuit"),
        "g57" => {
            let gates = format::read_g57_file(&args.input).expect("read g57 circuit");
            let wires = max_wire(&gates) as usize + 1;
            (gates, wires)
        }
        other => panic!("unknown --input-format {other}; expected g57 or mpmct1"),
    };
    assert!(
        !gates.is_empty(),
        "fsplit requires a nonempty input circuit"
    );
    let wires = file_wires.max(max_wire(&gates) as usize + 1);
    let gates_in = gates.len();
    let lits_in = lits_of(&gates);
    let g57_in = gates.iter().filter(|g| is_structural_g57(g)).count();
    let other_comp_in = gates
        .iter()
        .filter(|g| g.comp && !is_structural_g57(g))
        .count();
    let growth_target = ((gates_in as f64) * args.growth).ceil() as usize;
    let size_bound = args.target_size.unwrap_or(growth_target).max(gates_in);

    println!(
        "[fsplit] input={} gates={} lits={} g57={} wires={} target={} growth={:.4} k_max={} split_damp={} seed={}",
        args.input,
        gates_in,
        lits_in,
        g57_in,
        wires,
        size_bound,
        size_bound as f64 / gates_in.max(1) as f64,
        args.k_max,
        args.split_damp,
        args.seed,
    );

    let params = Params {
        k_max: args.k_max,
        split_damp: args.split_damp,
        g57_target_window: args.g57_target_window,
        size_bound,
        candidates: args.candidates.max(1),
        walk_cap: args.walk_cap.max(1),
        episode_cap: args.episode_cap.max(1),
        verify_every: args.verify_every.max(1),
        report_every: args.report_every.max(1),
        saturation_patience: args.saturation_patience.max(1),
        local_verify: !args.no_local_verify,
        seed: args.seed,
    };
    let total_started = std::time::Instant::now();
    let mut engine = Engine::new(gates, params);
    let split_started = std::time::Instant::now();
    let stop = engine.run();
    let split_seconds = split_started.elapsed().as_secs_f64();
    engine.report();

    let final_float_started = std::time::Instant::now();
    let (final_float_moved, final_float_steps) = if args.final_float {
        let result = engine.final_float();
        engine.global_check();
        result
    } else {
        (0, 0)
    };
    let final_float_seconds = final_float_started.elapsed().as_secs_f64();
    let output = engine.arena.to_vec();
    let source_marks = engine.source_marks_in_order();
    assert_eq!(output.len(), source_marks.len());
    let gates_out = output.len();
    let lits_out = lits_of(&output);
    let g57_out = output.iter().filter(|g| is_structural_g57(g)).count();
    let other_comp_out = output
        .iter()
        .filter(|g| g.comp && !is_structural_g57(g))
        .count();
    format::write_mpmct(&args.output, &output, wires).expect("write mpmct1 output");
    if let Some(path) = &args.sources_out {
        source::write_source_marks(path, &source_marks, engine.num_source_parents())
            .expect("write fsource1 sidecar");
    }
    let mixed_out = source_marks
        .iter()
        .filter(|&&mark| mark == MIXED_SOURCE)
        .count();
    let pristine_out = gates_out - mixed_out;
    let total_seconds = total_started.elapsed().as_secs_f64();

    let reason = match stop {
        StopReason::SizeBound => "size_bound",
        StopReason::Saturated => "saturated",
        StopReason::TraceFull => "trace_full",
    };
    println!(
        "[fsplit] wrote {} gates to {} (reason={}, split_seconds={:.3}, final_float_moved={}, final_float_steps={})",
        gates_out, args.output, reason, split_seconds, final_float_moved, final_float_steps,
    );
    println!("fsource_summary_csv,output_gates,pristine_parent,mixed_parents,sidecar");
    println!(
        "fsource_summary_csv,{},{},{},{}",
        gates_out,
        pristine_out,
        mixed_out,
        args.sources_out.as_deref().unwrap_or("")
    );
    println!(
        "fsplit_summary_csv,input_gates,output_gates,input_lits,output_lits,input_structural_g57,output_structural_g57,input_other_complemented,output_other_complemented,episodes,presplit_shot,presplit_colliding,r1,r2,r3,reason,split_seconds,final_float_seconds,total_seconds,final_float_moved,final_float_steps"
    );
    println!(
        "fsplit_summary_csv,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{},{}",
        gates_in,
        gates_out,
        lits_in,
        lits_out,
        g57_in,
        g57_out,
        other_comp_in,
        other_comp_out,
        engine.counters.episodes,
        engine.counters.presplit_shot,
        engine.counters.presplit_colliding,
        engine.counters.splits_r1,
        engine.counters.splits_r2,
        engine.counters.splits_r3,
        reason,
        split_seconds,
        final_float_seconds,
        total_seconds,
        final_float_moved,
        final_float_steps,
    );
}
