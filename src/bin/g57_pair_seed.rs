//! Insert deterministic adjacent native-G57 identity pairs into a G57 circuit.
//! This is intended for the pre-old-SSS and post-old-SSS/pre-fsplit factorial
//! hooks.  Fragmenting/shooting an already-XGate circuit is provided by fmix's
//! matching pair-seed hook.

use clap::Parser;
use local_mixing::circuit::CircuitSeq;
use local_mixing::postmix::g57_pairs::{
    G57PairRegion, G57PairSeedConfig, insert_g57_identity_pairs_and_shoot,
};

#[derive(Parser, Debug)]
#[command(name = "g57_pair_seed")]
struct Args {
    #[arg(long)]
    input: String,
    #[arg(long)]
    output: String,
    /// Total wire namespace available to the target and random controls.
    #[arg(long)]
    num_wires: usize,
    /// Seed each target in 0..TARGET_WIRES (normally the 128 functional wires).
    #[arg(long)]
    target_wires: usize,
    /// Controls are sampled only from 0..CONTROL_WIRE_LIMIT.  This may be
    /// smaller than --num-wires (for example 128 controls over 384 targets).
    #[arg(long)]
    control_wire_limit: usize,
    /// Number of identical G57 pairs generated for every target wire.
    #[arg(long)]
    pairs_per_wire: usize,
    /// first-quarter | middle-quarter | last-quarter | uniform
    #[arg(long)]
    region: G57PairRegion,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Maximum proven-commuting swaps for each copy: first copy left, second
    /// copy right.  Zero means maximal shooting to the first collider or
    /// boundary; a positive value is a hard commuting-swap cap.
    #[arg(long, default_value_t = 0)]
    shoot_steps: usize,
    /// Machine-readable per-pair evidence (default: OUTPUT.pairs.tsv).
    #[arg(long)]
    manifest: Option<String>,
}

fn main() {
    let args = Args::parse();
    let source = std::fs::read_to_string(&args.input).expect("read input G57 circuit");
    let input = CircuitSeq::from_string(&source);
    let config = G57PairSeedConfig {
        pairs_per_target_wire: args.pairs_per_wire,
        target_wires: args.target_wires,
        num_wires: args.num_wires,
        control_wire_limit: args.control_wire_limit,
        region: args.region,
        seed: args.seed,
    };
    let (output, report) = insert_g57_identity_pairs_and_shoot(&input, config, args.shoot_steps)
        .unwrap_or_else(|error| panic!("invalid G57 pair seed request: {error}"));
    std::fs::write(&args.output, output.repr()).expect("write seeded G57 circuit");
    let manifest = args
        .manifest
        .unwrap_or_else(|| format!("{}.pairs.tsv", args.output));
    std::fs::write(&manifest, report.manifest_tsv()).expect("write G57 pair manifest");
    println!(
        "[g57-pair-seed] baseline={} final={} pairs={} inserted_gates={} per_target={} targets={} controls=0..{} region={} frozen_gaps={}..{} seed={} shoot_steps={} left_distance={} right_distance={} collision_stops={} boundary_stops={} adjacent_remaining={} output={} manifest={}",
        report.baseline_gates,
        report.final_gates,
        report.pairs,
        report.inserted_gates,
        report.pairs_per_target_wire,
        report.target_wires,
        report.control_wire_limit,
        report.region,
        report.first_gap,
        report.gap_end_exclusive,
        report.seed,
        report.shoot_steps_per_copy,
        report.total_left_distance,
        report.total_right_distance,
        report.collision_stops,
        report.boundary_stops,
        report.adjacent_pairs_remaining,
        args.output,
        manifest,
    );
}
