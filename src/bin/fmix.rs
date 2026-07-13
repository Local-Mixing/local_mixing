// Post-fsplit mixing chain driver. Takes an mpmct1 circuit (typically fsplit
// output; g57 also accepted) and runs a randomized, size-thermostatted local
// rewrite walk: R-rule crossings, fresh-wire case splits, unsubsume splits,
// copy-pair insertions and conjugation twists (--w-twist-neg / --w-twist-swap:
// bracket a window with a wire negation or a 3-CNOT wire swap and conjugate
// its interior — state/progress mixing, the SAMF mechanism XGate-native)
// expand; catalogue merges (cancel / X-fuse / drop-literal / subsume)
// contract. The thermostat holds the gate count near --target-size; the
// objective is churn (distance from the original description), not size.
// Never emits comp=1 gates, so the g57 "fossil" count is monotone. Ends with a
// final uniform float, a sampled global check against the input, and an mpmct1
// write.
//
// Graceful stop: touch $FMIX_STOP_FLAG and the run finishes cleanly (verified
// write) at the next report point. Snapshot: touch $FMIX_DUMP_FLAG to get a
// verified mid-run circuit at $FMIX_DUMP_OUT (default <output>.snapshot.txt).
//
// Example:
//   FMIX_STOP_FLAG=/tmp/fmix.stop fmix --input mixed_fsplit.txt \
//     --target-size 3000000 --moves 50000000 --output mixed_fmix.txt
use clap::Parser;
use local_mixing::postmix::format;
use local_mixing::postmix::mix::{MixParams, MixStop, Mixer, ORIGIN_SYNTH};
use local_mixing::postmix::xgate::{XGate, max_wire};

#[derive(Parser, Debug)]
#[command(name = "fmix")]
struct Args {
    /// Input circuit file
    #[arg(long)]
    input: String,
    /// Input format: mpmct1 | g57
    #[arg(long, default_value = "mpmct1")]
    input_format: String,
    /// Output file (mpmct1 format)
    #[arg(long)]
    output: Option<String>,
    /// Thermostat target gate count (default: input size)
    #[arg(long)]
    target_size: Option<usize>,
    /// Thermostat softness in gates (default: max(target/100, 64))
    #[arg(long)]
    temp: Option<f64>,
    /// Total move attempts
    #[arg(long, default_value_t = 1_000_000)]
    moves: u64,
    /// Max controls per gate (K)
    #[arg(long, default_value_t = 12)]
    k_max: usize,
    /// Width-damping offset D for expansion moves (fsplit convention)
    #[arg(long, default_value_t = 2)]
    split_damp: usize,
    /// Width-damper base B: a split of parent width c proceeds with
    /// probability B^-(c - split_damp)
    #[arg(long, default_value_t = 2.0)]
    split_base: f64,
    /// Probability that a collision fragment inherits the shot gate's
    /// direction (else it gets the opposite)
    #[arg(long, default_value_t = 0.75)]
    dir_p: f64,
    /// Directional transport fraction: fresh pieces advance floor(q * slack)
    /// in their own direction at birth; a failed cross retreats the shot gate
    /// floor((1-q) * way)
    #[arg(long, default_value_t = 0.85)]
    dir_q: f64,
    /// Max distance (gates) a merge partner may sit from the initiator
    #[arg(long, default_value_t = 4096)]
    merge_reach: usize,
    /// Undo journal capacity (recorded crossings eligible for reversal)
    #[arg(long, default_value_t = 262_144)]
    journal_len: usize,
    /// Fraction of contraction moves that try a journal undo first
    #[arg(long, default_value_t = 0.5)]
    undo_frac: f64,
    /// Refractory period in moves: a split event may not be undone or
    /// sibling-merged until this many moves have passed
    #[arg(long, default_value_t = 2_000)]
    tabu_moves: u64,
    /// Expansion move weights
    #[arg(long, default_value_t = 0.70)]
    w_cross: f64,
    /// SUSPENDED by default (covered by the twists' case-splitting); set > 0
    /// to re-enable
    #[arg(long, default_value_t = 0.0)]
    w_fresh: f64,
    #[arg(long, default_value_t = 0.10)]
    w_unsub: f64,
    #[arg(long, default_value_t = 0.05)]
    w_insert: f64,
    /// Conjugation-twist weights (0 = off, trajectory-identical to the
    /// pre-twist chain). One twist conjugates a whole window (log-uniform
    /// length up to the circuit size) by a wire negation (+2 gates) or a wire
    /// swap (+6 gates), so keep these SMALL relative to the other weights:
    /// ~1e-4 gives a few thousand twists per 10M expansion moves.
    #[arg(long, default_value_t = 0.0)]
    w_twist_neg: f64,
    #[arg(long, default_value_t = 0.0)]
    w_twist_swap: f64,
    /// Transvection twist: conjugate a window by x_a ^= x_b (one CNOT per
    /// side, +2 gates). Affine and NOT a Hamming isometry — the rung that
    /// breaks avalanche-style distance gauges neg/swap twists provably
    /// preserve. Interior gates reading a case-split on b (count x2, width +1,
    /// K-cap enforced); b is drawn from wires the window never writes, which
    /// caps these windows at the mid scale (~n*ln n gates).
    #[arg(long, default_value_t = 0.0)]
    w_twist_cnot: f64,
    /// Minimum twist window length (max is the current circuit size)
    #[arg(long, default_value_t = 64)]
    twist_min_len: usize,
    /// Global sampled equality check every N moves
    #[arg(long, default_value_t = 10_000)]
    verify_every: u64,
    /// Progress report (and stop/dump flag check) every N moves
    #[arg(long, default_value_t = 50_000)]
    report_every: u64,
    /// Disable the per-move exhaustive local verification
    #[arg(long, default_value_t = false)]
    no_local_verify: bool,
    /// Skip the final uniform float pass
    #[arg(long, default_value_t = false)]
    skip_final_float: bool,
    /// Write per-gate origin indices (final order, one per line; 4294967295 =
    /// synthetic) for dispersion analysis
    #[arg(long)]
    origins_out: Option<String>,
    #[arg(long, default_value_t = 0)]
    seed: u64,
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
    let num_wires = file_wires.max(max_wire(&gates) as usize + 1);
    let input_len = gates.len();
    let comp0 = gates.iter().filter(|g| g.comp).count();
    let target = args.target_size.unwrap_or(input_len);
    println!(
        "[fmix] input: {} gates ({} g57 fossils), {} wires; k_max={} split_damp={} split_base={} dir_p={} dir_q={} target={} temp={} moves={} seed={}",
        input_len,
        comp0,
        num_wires,
        args.k_max,
        args.split_damp,
        args.split_base,
        args.dir_p,
        args.dir_q,
        target,
        args.temp.unwrap_or((target as f64 / 100.0).max(64.0)),
        args.moves,
        args.seed
    );
    if args.w_twist_neg > 0.0 || args.w_twist_swap > 0.0 || args.w_twist_cnot > 0.0 {
        println!(
            "[fmix] twists ON: w_twist_neg={} w_twist_swap={} w_twist_cnot={} twist_min_len={}",
            args.w_twist_neg, args.w_twist_swap, args.w_twist_cnot, args.twist_min_len
        );
    }

    let params = MixParams {
        k_max: args.k_max,
        split_damp: args.split_damp,
        split_base: args.split_base,
        dir_p: args.dir_p,
        dir_q: args.dir_q,
        target_size: target,
        temp: args.temp.unwrap_or(0.0),
        moves: args.moves,
        merge_reach: args.merge_reach,
        journal_len: args.journal_len,
        undo_frac: args.undo_frac,
        tabu_moves: args.tabu_moves,
        w_cross: args.w_cross,
        w_fresh: args.w_fresh,
        w_unsub: args.w_unsub,
        w_insert: args.w_insert,
        w_twist_neg: args.w_twist_neg,
        w_twist_swap: args.w_twist_swap,
        w_twist_cnot: args.w_twist_cnot,
        twist_min_len: args.twist_min_len,
        verify_every: args.verify_every,
        report_every: args.report_every,
        local_verify: !args.no_local_verify,
        seed: args.seed,
    };
    let mut mixer = Mixer::new(gates, num_wires, params);

    let stop = std::env::var("FMIX_STOP_FLAG").ok().filter(|s| !s.is_empty());
    let dump = std::env::var("FMIX_DUMP_FLAG").ok().filter(|s| !s.is_empty());
    if stop.is_some() || dump.is_some() {
        let dump_out = std::env::var("FMIX_DUMP_OUT").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| {
            args.output
                .as_deref()
                .map(|o| format!("{o}.snapshot.txt"))
                .unwrap_or_else(|| "fmix_snapshot.txt".to_string())
        });
        if let Some(f) = &stop {
            println!("[fmix] stop flag armed: touch {f} -> clean finish");
        }
        if let Some(f) = &dump {
            println!("[fmix] dump signal armed: touch {f} -> snapshot to {dump_out}");
        }
        mixer.enable_flags(stop, dump, dump_out);
    }

    let t0 = std::time::Instant::now();
    let stop_reason = mixer.run();
    let secs = t0.elapsed().as_secs_f64();
    mixer.report();
    println!(
        "[fmix] chain done in {:.1}s: {} ({} -> {} gates, {} -> {} g57 fossils)",
        secs,
        match stop_reason {
            MixStop::MovesBudget => "moves budget spent",
            MixStop::StopFlag => "stop flag",
        },
        input_len,
        mixer.arena.len(),
        comp0,
        mixer.remaining_g57(),
    );

    if !args.skip_final_float {
        let t1 = std::time::Instant::now();
        let (moved, disp) = mixer.final_float();
        mixer.global_check();
        println!(
            "[fmix] final float: {} gates moved, {} total displacement, {:.1}s (verified)",
            moved,
            disp,
            t1.elapsed().as_secs_f64()
        );
    }

    if let Some(out) = &args.output {
        let final_gates = mixer.arena.to_vec();
        format::write_mpmct(out, &final_gates, num_wires).expect("write output");
        println!("[fmix] wrote {} gates to {}", final_gates.len(), out);
    } else {
        println!("[fmix] no --output given; result discarded after verification");
    }

    if let Some(path) = &args.origins_out {
        let origins = mixer.origins_in_order();
        let mut s = String::with_capacity(origins.len() * 8);
        for o in origins {
            s.push_str(&format!("{o}\n"));
        }
        std::fs::write(path, s).expect("write origins");
        println!("[fmix] wrote origins sidecar to {path} (synthetic = {ORIGIN_SYNTH})");
    }
}
