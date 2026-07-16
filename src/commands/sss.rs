use std::fs;
use std::path::Path;

use local_mixing::circuit::CircuitSeq;
use local_mixing::replace::frozen::FrozenDb;
use local_mixing::replace::gadgets::{
    SLICE_ZERO_HARDCODED_DEFAULT_ROUNDS, SLICE_ZERO_RANDOM_GATES_PER_WIRE,
};
use local_mixing::replace::main_mix::main_shuffle_shoot_shuffle;
use local_mixing::replace::main_mix_cnot::{CnotSssParams, main_shuffle_shoot_shuffle_cnot};
use local_mixing::replace::mixing::install_kill_handler;
use local_mixing::replace::replace::{
    GEN_MODE, INCOMING_RANK_MODE, IncomingRankMode, MAX_FANOUT, MIN_MEDIAN_LEEWAY,
    OUTGOING_GEN_MODE, SAMF_TARGET, TRACK_SURVIVORS, print_compress_timers, record_finish,
    record_init, write_compression_histogram, write_expansion_histogram,
    write_expansion_wire_histogram,
};
use std::sync::atomic::Ordering;

/// Shuffle-shoot-shuffle: the main obfuscation+compression game.
pub fn run(sub: &clap::ArgMatches) {
    let rounds: usize = *sub.get_one("rounds").unwrap();
    let s: &str = sub.get_one::<String>("source").unwrap().as_str();
    let d: &str = sub.get_one::<String>("destination").unwrap().as_str();
    let n: usize = *sub.get_one("n").unwrap();
    let m: usize = *sub.get_one("m").unwrap();
    let x: usize = *sub.get_one("x").unwrap();
    let leave = sub.get_flag("interleave");
    let do_gadgetize = sub.get_flag("gadgetize");
    let do_feistalize = sub.get_flag("feistalize");
    let do_cnot = sub.get_flag("cnot");
    let slice_zero = sub.get_flag("slice_zero");
    let slice_zero_random = sub.get_flag("slice_zero_random");
    let slice_zero_hardcoded = sub.get_flag("slice_zero_hardcoded");
    let slice_zero_random_gates: usize = sub
        .get_one::<usize>("M_length")
        .copied()
        .or_else(|| sub.get_one::<usize>("slice_zero_random_gates").copied())
        .unwrap_or(SLICE_ZERO_RANDOM_GATES_PER_WIRE * n);
    let slice_zero_hardcoded_rounds: usize = sub
        .get_one("slice_zero_hardcoded_rounds")
        .copied()
        .unwrap_or(SLICE_ZERO_HARDCODED_DEFAULT_ROUNDS);
    let gadget_path = sub.get_one::<String>("gadget_path").map(|s| s.as_str());
    let full_shuffle = sub.get_flag("full-shuffle");
    let full_shuffle_early = sub.get_flag("full-shuffle-early");
    let gates_ahead_expand: usize = *sub.get_one("gates_ahead_expand").unwrap();
    let gates_ahead_samf: usize = *sub.get_one("gates_ahead_samf").unwrap();
    let type_attempts: usize = *sub.get_one("type_attempts").unwrap();
    let shooting_times: usize = *sub.get_one("shooting_times").unwrap();
    let collision_rounds: usize = *sub.get_one("collision_rounds").unwrap();
    let stable_compressions: usize = *sub.get_one("stable_compressions").unwrap();
    let expansion_game = sub.get_flag("expansion_game");
    let equality_check = sub.get_flag("equality_check");
    let single_end = sub.get_flag("single-end");
    let record_replacements = sub.get_flag("record_replacements");
    let generation_tags = sub.get_flag("generation_tags");
    let outgoing_mode = sub.get_one::<String>("outgoing_mode").unwrap().as_str();
    let incoming_rank = sub.get_one::<String>("incoming_rank").unwrap().as_str();
    let max_fanout: usize = *sub.get_one("max_fanout").unwrap();
    let min_median_leeway: usize = *sub.get_one("min_median_leeway").unwrap();
    let samf_target: usize = *sub.get_one("samf_target").unwrap();
    let min_gen: usize = *sub.get_one("min_gen").unwrap();
    let min_gen_fraction: f64 = *sub.get_one("min_gen_fraction").unwrap();
    let pass_length: usize = *sub.get_one("pass_length").unwrap();
    let max_passes: usize = *sub.get_one("max_passes").unwrap();
    let rg_freq: usize = *sub.get_one("rg_frequency").unwrap();
    let data = fs::read_to_string(s).expect("Failed to read source circuit");

    install_kill_handler();
    if data.trim().is_empty() {
        println!("Empty file");
        return;
    }

    let c = CircuitSeq::from_string(&data);
    if do_cnot {
        assert!(!leave, "--cnot does not yet support --interleave");
        assert!(
            !single_end,
            "--single-end is a legacy G57-only mode; the --cnot mixer preserves functionality after every move"
        );
        assert!(
            !record_replacements,
            "--record is not available for the heterogeneous --cnot replacement engine"
        );
        assert!(
            !generation_tags && min_gen == 0 && outgoing_mode == "legacy" && incoming_rank == "sat",
            "generation-tag/Stage-B options are not available with --cnot"
        );
        assert!(
            samf_target == 0,
            "--samf-target belongs to the legacy G57 collision game and is not available with --cnot"
        );
        main_shuffle_shoot_shuffle_cnot(
            &c,
            &CnotSssParams {
                rounds,
                n,
                m,
                x,
                save: d,
                source: s,
                do_gadgetize,
                do_feistalize,
                slice_zero,
                slice_zero_random,
                slice_zero_random_gates,
                slice_zero_hardcoded,
                slice_zero_hardcoded_rounds,
                gadget_path,
                full_shuffle,
                full_shuffle_early,
                shooting_times,
                collision_rounds,
                stable_compressions,
                expansion_game,
                equality_check,
                rg_freq,
            },
        );
        return;
    }

    let db = FrozenDb::from_env();
    if record_replacements {
        record_init(&format!("{}.replacements", d));
    }
    let gen_tags_enabled = generation_tags
        || min_gen > 0
        || outgoing_mode == "gen"
        || incoming_rank == "fanout"
        || incoming_rank == "hybrid";
    GEN_MODE.store(gen_tags_enabled, Ordering::Relaxed);
    TRACK_SURVIVORS.store(gen_tags_enabled, Ordering::Relaxed);
    OUTGOING_GEN_MODE.store(outgoing_mode == "gen", Ordering::Relaxed);
    MAX_FANOUT.store(max_fanout, Ordering::Relaxed);
    MIN_MEDIAN_LEEWAY.store(min_median_leeway, Ordering::Relaxed);
    SAMF_TARGET.store(samf_target, Ordering::Relaxed);
    INCOMING_RANK_MODE.store(
        match incoming_rank {
            "fanout" => IncomingRankMode::Fanout as usize,
            "hybrid" => IncomingRankMode::Hybrid as usize,
            _ => IncomingRankMode::Sat as usize,
        },
        Ordering::Relaxed,
    );
    if gen_tags_enabled {
        println!(
            "[sss] generation tags ON | outgoing_mode={} | incoming_rank={} | max_fanout={} | min_median_leeway={}",
            outgoing_mode, incoming_rank, max_fanout, min_median_leeway
        );
    }
    if samf_target > 0 {
        println!(
            "[sss] samf-target ON: disable plain m after >= {} hidden SAMFs in a round",
            samf_target
        );
    }
    if min_gen > 0 {
        println!(
            "[sss] Stage B ON: min_gen={} min_gen_fraction={:.4} pass_length={} max_passes={}",
            min_gen, min_gen_fraction, pass_length, max_passes
        );
    }
    main_shuffle_shoot_shuffle(
        &c,
        rounds,
        n,
        m,
        x,
        d,
        s,
        &db,
        leave,
        do_gadgetize,
        do_feistalize,
        slice_zero,
        slice_zero_random,
        slice_zero_random_gates,
        slice_zero_hardcoded,
        slice_zero_hardcoded_rounds,
        gadget_path,
        full_shuffle,
        full_shuffle_early,
        gates_ahead_expand,
        gates_ahead_samf,
        type_attempts,
        shooting_times,
        collision_rounds,
        stable_compressions,
        expansion_game,
        equality_check,
        rg_freq,
        single_end,
        min_gen,
        min_gen_fraction,
        pass_length,
        max_passes,
    );
    if record_replacements {
        record_finish();
        println!("Replacement record written to {}.replacements", d);
    }
    print_compress_timers();
    write_compression_histogram("compression_histogram.csv");
    write_expansion_histogram("expansion_histogram.csv");
    write_expansion_wire_histogram("expansion_wire_histogram.csv");

    println!("\n=== Plot commands (copy/paste one at a time) ===");
    println!(
        "python3 ./heatmap/compression_hist.py --csv compression_histogram.csv --out compression_histogram.png"
    );
    println!(
        "python3 ./heatmap/compression_heatmap.py --csv compression_histogram.csv --out compression_heatmap.png"
    );
    println!(
        "python3 ./heatmap/compression_hist.py --csv expansion_histogram.csv --out expansion_histogram.png"
    );
    println!(
        "python3 ./heatmap/compression_heatmap.py --csv expansion_histogram.csv --out expansion_heatmap.png"
    );
    println!(
        "python3 ./heatmap/compression_hist.py --csv expansion_wire_histogram.csv --out expansion_wire_histogram.png"
    );
    println!(
        "python3 ./heatmap/compression_heatmap.py --csv expansion_wire_histogram.csv --out expansion_wire_heatmap.png"
    );
    println!("=== end plot commands ===\n");

    let x_label = {
        let stem = Path::new(s).file_stem().unwrap().to_str().unwrap();
        let num = stem.strip_prefix("circuit").unwrap_or(stem);
        format!("Circuit {}", num)
    };
    let y_label = {
        let stem = Path::new(d).file_stem().unwrap().to_str().unwrap();
        let num = stem.strip_prefix("circuit").unwrap_or(stem);
        format!("Circuit {}", num)
    };
    let path_s = Path::new(s).file_stem().unwrap().to_str().unwrap();
    let path_d = Path::new(d).file_stem().unwrap().to_str().unwrap();
    println!(
        "For generating heatmaps:\n\
        python3 ./heatmap/heatmap.py \
        --n {} \
        --i 100 \
        --x \"{}\" \
        --y \"{}\" \
        --c1 \"{}\" \
        --c2 \"{}\" \
        --path ./{}{}.png",
        n, x_label, y_label, s, d, path_s, path_d
    );
}
