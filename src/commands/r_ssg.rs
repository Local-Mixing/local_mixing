//! Command entry point for the isolated Ran-style generation mixer.
//!
//! Unlike NH `sss`, this command enters the vendored R-SSG Stage B/C runtime
//! directly. The two algorithms intentionally do not share their tag vectors,
//! transposition scheduler, or main mixing driver.

use std::fs;
use std::sync::atomic::Ordering;

use local_mixing::circuit::CircuitSeq;
use local_mixing::r_ssg::gadgets::{
    SLICE_ZERO_HARDCODED_DEFAULT_ROUNDS, SLICE_ZERO_RANDOM_GATES_PER_WIRE,
};
use local_mixing::r_ssg::main_mix::main_shuffle_shoot_shuffle;
use local_mixing::r_ssg::ranking::{ScriptRanker, set_incoming, set_outgoing};
use local_mixing::r_ssg::replace::{
    COMPRESS_FRACTION_PERMILLE, GEN_MODE, GROW_THRESHOLD_PERMILLE, MAX_FANOUT, MAX_PASSES, MIN_GEN,
    MIN_GEN_PERMILLE, MIN_MEDIAN_LEEWAY, PASS_LENGTH, SAMF_TARGET, TARGET_SIZE, TRACK_SURVIVORS,
    print_compress_timers, record_finish, record_init, write_compression_histogram,
    write_expansion_histogram, write_expansion_wire_histogram,
};
use local_mixing::replace::frozen::FrozenDb;
use local_mixing::replace::main_mix_cnot::{CnotSssParams, main_shuffle_shoot_shuffle_cnot};
use local_mixing::replace::mixing::install_kill_handler;

pub fn run(sub: &clap::ArgMatches) {
    // Gadget/TDP construction is engine-agnostic: `main_shuffle_shoot_shuffle_cnot`
    // depends only on `postmix` and `replace::gadgets`, never on either
    // generation mixer's Stage B/C runtime. Origin's `ssg` reaches it the same
    // way — it takes the identical arg set as `sss` and delegates — so the
    // sliced sandwich, [2,2,2,3] gadgetization and Gray fold are built first
    // and the mixer then runs on the result.
    let do_nonlinear_gadgetize = sub.get_flag("nonlinear_gadgetize");
    if sub.get_flag("cnot") || do_nonlinear_gadgetize || sub.get_flag("tdp4n") {
        run_cnot(sub, do_nonlinear_gadgetize);
        return;
    }

    // Authoritative R-SSG process globals. These are private to the r_ssg
    // namespace and cannot alter NH SSS state.
    GEN_MODE.store(true, Ordering::Relaxed);
    TRACK_SURVIVORS.store(true, Ordering::Relaxed);

    let max_fanout = *sub.get_one::<usize>("max_fanout").unwrap();
    let min_median_leeway = *sub.get_one::<usize>("min_median_leeway").unwrap();
    let samf_target = *sub.get_one::<usize>("samf_target").unwrap();
    let min_gen = *sub.get_one::<usize>("min_gen").unwrap();
    let pass_length = *sub.get_one::<usize>("pass_length").unwrap();
    let max_passes = *sub.get_one::<usize>("max_passes").unwrap();
    let min_gen_fraction = *sub.get_one::<f64>("min_gen_fraction").unwrap();
    let grow_threshold = *sub.get_one::<f64>("grow_threshold").unwrap();
    let compress_fraction = *sub.get_one::<f64>("compress_fraction").unwrap();
    let target_size = *sub.get_one::<usize>("target_size").unwrap();

    MAX_FANOUT.store(max_fanout, Ordering::Relaxed);
    MIN_MEDIAN_LEEWAY.store(min_median_leeway, Ordering::Relaxed);
    SAMF_TARGET.store(samf_target, Ordering::Relaxed);
    MIN_GEN.store(min_gen, Ordering::Relaxed);
    PASS_LENGTH.store(pass_length, Ordering::Relaxed);
    MAX_PASSES.store(max_passes, Ordering::Relaxed);
    MIN_GEN_PERMILLE.store(
        (min_gen_fraction.clamp(0.0, 1.0) * 1000.0).round() as usize,
        Ordering::Relaxed,
    );
    GROW_THRESHOLD_PERMILLE.store(
        (grow_threshold.max(0.0) * 10.0).round() as usize,
        Ordering::Relaxed,
    );
    COMPRESS_FRACTION_PERMILLE.store(
        (compress_fraction.clamp(0.0, 1.0) * 1000.0).round() as usize,
        Ordering::Relaxed,
    );
    TARGET_SIZE.store(target_size, Ordering::Relaxed);

    if let Some(path) = sub.get_one::<String>("outgoing_rank_script") {
        match ScriptRanker::from_file(path) {
            Ok(ranker) => {
                set_outgoing(Box::new(ranker));
                println!("[r-ssg] outgoing ranking <- script {path}");
            }
            Err(error) => {
                eprintln!("[r-ssg] FATAL: outgoing rank script: {error}");
                std::process::exit(1);
            }
        }
    }
    if let Some(path) = sub.get_one::<String>("incoming_rank_script") {
        match ScriptRanker::from_file(path) {
            Ok(ranker) => {
                set_incoming(Box::new(ranker));
                println!("[r-ssg] incoming ranking <- script {path}");
            }
            Err(error) => {
                eprintln!("[r-ssg] FATAL: incoming rank script: {error}");
                std::process::exit(1);
            }
        }
    }

    let rounds = *sub.get_one::<usize>("rounds").unwrap();
    let source = sub.get_one::<String>("source").unwrap();
    let destination = sub.get_one::<String>("destination").unwrap();
    let n = *sub.get_one::<usize>("n").unwrap();
    let m = *sub.get_one::<usize>("m").unwrap();
    let x = *sub.get_one::<usize>("x").unwrap();
    let slice_zero_random_gates = sub
        .get_one::<usize>("M_length")
        .copied()
        .or_else(|| sub.get_one::<usize>("slice_zero_random_gates").copied())
        .unwrap_or(SLICE_ZERO_RANDOM_GATES_PER_WIRE * n);
    let slice_zero_hardcoded_rounds = sub
        .get_one::<usize>("slice_zero_hardcoded_rounds")
        .copied()
        .unwrap_or(SLICE_ZERO_HARDCODED_DEFAULT_ROUNDS);

    let data = fs::read_to_string(source).expect("Failed to read source circuit");
    if data.trim().is_empty() {
        println!("Empty file");
        return;
    }
    install_kill_handler();
    let db = FrozenDb::from_env();
    let circuit = CircuitSeq::from_string(&data);

    let record = sub.get_flag("record_replacements");
    if record {
        record_init(&format!("{destination}.replacements"));
    }

    println!(
        "[r-ssg] isolated generation mixer | min_gen={} ({:.1}%) | pass_length={} | \
         max_passes={} | max_fanout={} | min_median_leeway={}",
        min_gen,
        min_gen_fraction * 100.0,
        pass_length,
        max_passes,
        max_fanout,
        min_median_leeway
    );

    main_shuffle_shoot_shuffle(
        &circuit,
        rounds,
        n,
        m,
        x,
        destination,
        source,
        &db,
        sub.get_flag("interleave"),
        sub.get_flag("gadgetize"),
        sub.get_flag("feistalize"),
        sub.get_flag("slice_zero"),
        sub.get_flag("slice_zero_random"),
        slice_zero_random_gates,
        sub.get_flag("slice_zero_hardcoded"),
        slice_zero_hardcoded_rounds,
        sub.get_one::<String>("gadget_path").map(String::as_str),
        sub.get_flag("full-shuffle"),
        sub.get_flag("full-shuffle-early"),
        *sub.get_one::<usize>("gates_ahead_expand").unwrap(),
        *sub.get_one::<usize>("gates_ahead_samf").unwrap(),
        *sub.get_one::<usize>("type_attempts").unwrap(),
        *sub.get_one::<usize>("shooting_times").unwrap(),
        sub.get_flag("expansion_game"),
        sub.get_flag("equality_check"),
        *sub.get_one::<usize>("rg_frequency").unwrap(),
        sub.get_flag("single-end"),
        sub.get_flag("light_compression"),
    );

    if record {
        record_finish();
    }
    print_compress_timers();
    write_compression_histogram("r_ssg_compression_histogram.csv");
    write_expansion_histogram("r_ssg_expansion_histogram.csv");
    write_expansion_wire_histogram("r_ssg_expansion_wire_histogram.csv");
}

/// Gadget/TDP path: construct the sliced sandwich / [2,2,2,3] gadget / Gray
/// fold, then mix the result with the heterogeneous XGate driver.
///
/// This mirrors `sss`'s cnot branch exactly, and for the same reason: once the
/// circuit carries mixed-polarity multi-control gates it is no longer in the
/// G57 `[u16; 3]` vocabulary either generation mixer's Stage B/C runtime reads,
/// so the shared `postmix`-based driver does the mixing. The Stage B/C globals
/// above are deliberately NOT set here — they steer a runtime this path never
/// enters, and setting them would misreport the configuration.
fn run_cnot(sub: &clap::ArgMatches, do_nonlinear_gadgetize: bool) {
    let s: &str = sub.get_one::<String>("source").unwrap().as_str();
    let d: &str = sub.get_one::<String>("destination").unwrap().as_str();
    let data = fs::read_to_string(s).expect("Failed to read source circuit");
    install_kill_handler();
    if data.trim().is_empty() {
        println!("Empty file");
        return;
    }
    let c = CircuitSeq::from_string(&data);

    // A nonlinear gadget defaults to one RG draw between consecutive source
    // gates; an explicit --rg-frequency still wins.
    let rg_freq: usize = *sub.get_one("rg_frequency").unwrap();
    let rg_freq = if do_nonlinear_gadgetize
        && sub.value_source("rg_frequency") == Some(clap::parser::ValueSource::DefaultValue)
    {
        1
    } else {
        rg_freq
    };

    // Options that belong to the legacy G57 collision game have no meaning
    // here; refuse them rather than silently ignoring them.
    assert!(
        !sub.get_flag("interleave"),
        "--cnot does not yet support --interleave"
    );
    assert!(
        !sub.get_flag("single-end"),
        "--single-end is a legacy G57-only mode; the --cnot mixer preserves functionality after every move"
    );
    assert!(
        !sub.get_flag("record_replacements"),
        "--record is not available for the heterogeneous --cnot replacement engine"
    );
    assert!(
        *sub.get_one::<usize>("samf_target").unwrap() == 0,
        "--samf-target belongs to the legacy G57 collision game and is not available with --cnot"
    );
    let grow_threshold: f64 = *sub.get_one("grow_threshold").unwrap();
    let target_size: usize = *sub.get_one("target_size").unwrap();
    let compress_fraction: f64 = *sub.get_one("compress_fraction").unwrap();
    assert!(
        grow_threshold == 0.0 && target_size == 0 && compress_fraction == 0.0,
        "SSS/SSG cadence options are not yet available with --cnot"
    );

    main_shuffle_shoot_shuffle_cnot(
        &c,
        &CnotSssParams {
            rounds: *sub.get_one("rounds").unwrap(),
            n: *sub.get_one("n").unwrap(),
            m: *sub.get_one("m").unwrap(),
            x: *sub.get_one("x").unwrap(),
            save: d,
            source: s,
            do_gadgetize: sub.get_flag("gadgetize"),
            do_nonlinear_gadgetize,
            do_feistalize: sub.get_flag("feistalize"),
            do_tdp4n: sub.get_flag("tdp4n"),
            slice_zero: sub.get_flag("slice_zero"),
            slice_zero_random: sub.get_flag("slice_zero_random"),
            slice_zero_random_gates: sub
                .get_one::<usize>("slice_zero_random_gates")
                .copied()
                .unwrap_or(SLICE_ZERO_RANDOM_GATES_PER_WIRE * *sub.get_one::<usize>("n").unwrap()),
            slice_zero_hardcoded: sub.get_flag("slice_zero_hardcoded"),
            slice_zero_hardcoded_rounds: sub
                .get_one("slice_zero_hardcoded_rounds")
                .copied()
                .unwrap_or(SLICE_ZERO_HARDCODED_DEFAULT_ROUNDS),
            gadget_path: sub.get_one::<String>("gadget_path").map(|s| s.as_str()),
            full_shuffle: sub.get_flag("full-shuffle"),
            full_shuffle_early: sub.get_flag("full-shuffle-early"),
            shooting_times: *sub.get_one("shooting_times").unwrap(),
            collision_rounds: *sub.get_one("collision_rounds").unwrap(),
            stable_compressions: *sub.get_one("stable_compressions").unwrap(),
            expansion_game: sub.get_flag("expansion_game"),
            equality_check: sub.get_flag("equality_check"),
            rg_freq,
        },
    );
}
