use std::sync::atomic::Ordering;

use local_mixing::replace::replace::{
    COMPRESS_FRACTION_PERMILLE, GEN_MODE, GROW_THRESHOLD_PERMILLE, MAX_FANOUT, MAX_PASSES, MIN_GEN,
    MIN_GEN_PERMILLE, MIN_MEDIAN_LEEWAY, PASS_LENGTH, SAMF_TARGET, TRACK_SURVIVORS,
};

/// ssg: generation-mixing variant of `sss`.
///
/// Same control flow as `sss` for now (rounds / shooting / compression), but with:
///   * generation tags (each gate carries its generation; a replacement's new gates get
///     floor(median(removed window generations)) + 1), and
///   * fanout/leeway-driven incoming-subcircuit selection (SAT-hardness scoring disabled).
/// Stages B/C/D (min-gen-anchored bidirectional bounded passes, partial-pass SAMF translation
/// tables, size-threshold compression cadence) will be layered on next.
pub fn run(sub: &clap::ArgMatches) {
    GEN_MODE.store(true, Ordering::Relaxed);
    TRACK_SURVIVORS.store(true, Ordering::Relaxed); // keep the tag vector maintained
    let max_fanout: usize = *sub.get_one("max_fanout").unwrap();
    let min_median_leeway: usize = *sub.get_one("min_median_leeway").unwrap();
    MAX_FANOUT.store(max_fanout, Ordering::Relaxed);
    MIN_MEDIAN_LEEWAY.store(min_median_leeway, Ordering::Relaxed);
    let samf_target: usize = *sub.get_one("samf_target").unwrap();
    SAMF_TARGET.store(samf_target, Ordering::Relaxed);
    let min_gen: usize = *sub.get_one("min_gen").unwrap();
    let pass_length: usize = *sub.get_one("pass_length").unwrap();
    let max_passes: usize = *sub.get_one("max_passes").unwrap();
    let min_gen_fraction: f64 = *sub.get_one("min_gen_fraction").unwrap();
    let min_gen_permille = ((min_gen_fraction.clamp(0.0, 1.0)) * 1000.0).round() as usize;
    MIN_GEN.store(min_gen, Ordering::Relaxed);
    PASS_LENGTH.store(pass_length, Ordering::Relaxed);
    MAX_PASSES.store(max_passes, Ordering::Relaxed);
    MIN_GEN_PERMILLE.store(min_gen_permille, Ordering::Relaxed);
    // Stage D: size-threshold compression cadence (percent growth per stage, stored as permille).
    let grow_threshold: f64 = *sub.get_one("grow_threshold").unwrap();
    let grow_permille = (grow_threshold.max(0.0) * 10.0).round() as usize;
    GROW_THRESHOLD_PERMILLE.store(grow_permille, Ordering::Relaxed);
    // Stage D per-stage compression target, as a fraction of the post-shooting size (0 = full).
    let compress_fraction: f64 = *sub.get_one("compress_fraction").unwrap();
    let compress_fraction_permille = (compress_fraction.clamp(0.0, 1.0) * 1000.0).round() as usize;
    COMPRESS_FRACTION_PERMILLE.store(compress_fraction_permille, Ordering::Relaxed);
    if grow_permille > 0 {
        let compress_desc = if compress_fraction_permille > 0 {
            format!("compress to {:.1}% of post-shoot size", compress_fraction_permille as f64 / 10.0)
        } else {
            "compress fully".to_string()
        };
        println!(
            "[ssg] stage D: size-threshold cadence ON | grow {:.1}% per stage, {} (-r ignored; stop on min-gen)",
            grow_permille as f64 / 10.0,
            compress_desc
        );
    }
    println!(
        "[ssg] generation mode ON | max_fanout={} | min_median_leeway={} | samf_target={}",
        max_fanout, min_median_leeway, samf_target
    );
    println!(
        "[ssg] stage B: min_gen={} (>= {:.0}% of gates) pass_length={} max_passes={}",
        min_gen,
        min_gen_fraction * 100.0,
        pass_length,
        max_passes
    );

    // Optional runtime-pluggable ranking functions (no recompile). Built-in defaults are used
    // when these flags are absent: ParetoOutgoing (#11) and FanoutTargetIncoming (#9).
    use local_mixing::replace::ranking::{set_incoming, set_outgoing, ScriptRanker};
    if let Some(path) = sub.get_one::<String>("outgoing_rank_script") {
        match ScriptRanker::from_file(path) {
            Ok(r) => {
                set_outgoing(Box::new(r));
                println!("[ssg] outgoing ranking <- script {path}");
            }
            Err(e) => {
                eprintln!("[ssg] FATAL: outgoing rank script: {e}");
                std::process::exit(1);
            }
        }
    }
    if let Some(path) = sub.get_one::<String>("incoming_rank_script") {
        match ScriptRanker::from_file(path) {
            Ok(r) => {
                set_incoming(Box::new(r));
                println!("[ssg] incoming ranking <- script {path}");
            }
            Err(e) => {
                eprintln!("[ssg] FATAL: incoming rank script: {e}");
                std::process::exit(1);
            }
        }
    }
    // Delegate to the shared shuffle-shoot-shuffle driver; gen-mode behavior is gated on the
    // globals set above.
    crate::commands::sss::run(sub);
}
