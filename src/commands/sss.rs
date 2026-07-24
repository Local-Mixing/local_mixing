use std::fs;
use std::path::Path;

use local_mixing::circuit::CircuitSeq;
use local_mixing::replace::frozen::FrozenDb;
use local_mixing::replace::gadgets::{
    SLICE_ZERO_CCNOT_GATES_PER_WIRE, SLICE_ZERO_HARDCODED_DEFAULT_ROUNDS,
    SLICE_ZERO_RANDOM_GATES_PER_WIRE, sandwich_default_m, sandwich_default_s,
};
use local_mixing::replace::main_mix::main_shuffle_shoot_shuffle;
use local_mixing::replace::gadgets::{MaskConfig, ProdConfig};
use local_mixing::replace::main_mix_cnot::{CnotSssParams, main_shuffle_shoot_shuffle_cnot};
use local_mixing::replace::mixing::install_kill_handler;
use local_mixing::replace::replace::{
    print_compress_timers, record_finish, record_init, write_compression_histogram,
    write_expansion_histogram, write_expansion_wire_histogram,
};

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
    let slice_zero = sub.get_flag("slice_zero");
    let slice_zero_random = sub.get_flag("slice_zero_random");
    let slice_zero_hardcoded = sub.get_flag("slice_zero_hardcoded");
    let slice_zero_random_gates: usize = sub
        .get_one("slice_zero_random_gates")
        .copied()
        .unwrap_or(SLICE_ZERO_RANDOM_GATES_PER_WIRE * n);
    let slice_zero_hardcoded_rounds: usize = sub
        .get_one("slice_zero_hardcoded_rounds")
        .copied()
        .unwrap_or(SLICE_ZERO_HARDCODED_DEFAULT_ROUNDS);
    let slice_zero_ccnot = sub.get_flag("slice_zero_ccnot");
    let slice_zero_ccnot_gates: usize = sub
        .get_one("slice_zero_ccnot_gates")
        .copied()
        .unwrap_or(SLICE_ZERO_CCNOT_GATES_PER_WIRE * n);
    let sliced_sandwich = sub.get_flag("sliced_sandwich");
    let sandwich_m: usize = sub
        .get_one("sandwich_m")
        .copied()
        .unwrap_or_else(|| sandwich_default_m(n));
    let sandwich_s: usize = sub
        .get_one("sandwich_s")
        .copied()
        .unwrap_or_else(|| sandwich_default_s(n));
    let gadget_path = sub.get_one::<String>("gadget_path").map(|s| s.as_str());
    let full_shuffle = sub.get_flag("full-shuffle");
    let full_shuffle_early = sub.get_flag("full-shuffle-early");
    let gates_ahead_expand: usize = *sub.get_one("gates_ahead_expand").unwrap();
    let gates_ahead_samf: usize = *sub.get_one("gates_ahead_samf").unwrap();
    let type_attempts: usize = *sub.get_one("type_attempts").unwrap();
    let shooting_times: usize = *sub.get_one("shooting_times").unwrap();
    let egg = sub.get_flag("egg");
    let equality_check = sub.get_flag("equality_check");
    let single_end = sub.get_flag("single-end");
    let light_compression = sub.get_flag("light_compression");
    let record_replacements = sub.get_flag("record_replacements");
    let track_survivors = sub.get_flag("track_survivors");
    let rg_freq: usize = *sub.get_one("rg_frequency").unwrap();
    let data = fs::read_to_string(s).expect("Failed to read source circuit");

    // Heterogeneous --cnot path: gadgetize with native CNOTs/fragments and mix
    // with the fmix engine (no replacement DB). Handled before the FrozenDb open
    // so --cnot does not require FROZEN_DB_DIR.
    if sub.get_flag("cnot") {
        install_kill_handler();
        if data.trim().is_empty() {
            println!("Empty file");
            return;
        }
        // The cnot gadget path settles on ONE nonlinear RG per SG unless
        // --rg-frequency is given explicitly (the clap default of 2 belongs
        // to the legacy every-K-SGs meaning on the other paths).
        let rg_freq = if sub.value_source("rg_frequency")
            == Some(clap::parser::ValueSource::DefaultValue)
        {
            1
        } else {
            rg_freq
        };
        let masks = MaskConfig {
            cov: *sub.get_one("mask_cov").unwrap(),
            k: *sub.get_one("mask_k").unwrap(),
            depth: *sub.get_one("mask_depth").unwrap(),
            taper: sub.get_one("mask_taper").copied(),
        };
        let prod = ProdConfig {
            k: *sub.get_one("prod_k").unwrap(),
            deg: *sub.get_one("prod_deg").unwrap(),
            k_hi: *sub.get_one("prod_k_hi").unwrap(),
            deg_hi: *sub.get_one("prod_deg_hi").unwrap(),
            band: *sub.get_one("prod_band").unwrap(),
            rsrc: *sub.get_one("prod_rsrc").unwrap(),
            max_width: *sub.get_one("prod_max_width").unwrap(),
            fill_nl: *sub.get_one("prod_fill_nl").unwrap(),
        };
        let c = CircuitSeq::from_string(&data);
        let collision_rounds: usize = *sub.get_one("collision_rounds").unwrap();
        let stable_compressions: usize = *sub.get_one("stable_compressions").unwrap();
        let params = CnotSssParams {
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
            slice_zero_ccnot,
            slice_zero_ccnot_gates,
            sliced_sandwich,
            sandwich_m,
            sandwich_s,
            gadget_path,
            full_shuffle,
            full_shuffle_early,
            shooting_times,
            collision_rounds,
            stable_compressions,
            expansion_game: egg,
            equality_check,
            rg_freq,
            masks,
            prod,
        };
        main_shuffle_shoot_shuffle_cnot(&c, &params);
        return;
    }

    // Replacement lookups come from the immutable frozen stores: FROZEN_DB_DIR
    // (regular, required) and FROZEN_CURATED_DIR (curated, optional).
    let db = FrozenDb::from_env();
    install_kill_handler();
    if data.trim().is_empty() {
        println!("Empty file");
        return;
    }

    let c = CircuitSeq::from_string(&data);
    if record_replacements {
        record_init(&format!("{}.replacements", d));
    }
    if track_survivors {
        local_mixing::replace::replace::TRACK_SURVIVORS
            .store(true, std::sync::atomic::Ordering::Relaxed);
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
        egg,
        equality_check,
        rg_freq,
        single_end,
        light_compression,
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
