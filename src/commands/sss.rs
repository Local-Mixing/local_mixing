use std::fs;
use std::path::Path;

use lmdb::Environment;

use local_mixing::circuit::CircuitSeq;
use local_mixing::replace::main_mix::{main_shuffle_shoot_shuffle, open_all_dbs};
use local_mixing::replace::mixing::install_kill_handler;
use local_mixing::replace::replace::{
    print_compress_timers, write_compression_histogram, write_expansion_histogram,
    write_expansion_wire_histogram,
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
    let gadget_path = sub.get_one::<String>("gadget_path").map(|s| s.as_str());
    let full_shuffle = sub.get_flag("full-shuffle");
    let gates_ahead_expand: usize = *sub.get_one("gates_ahead_expand").unwrap();
    let gates_ahead_samf: usize = *sub.get_one("gates_ahead_samf").unwrap();
    let type_attempts: usize = *sub.get_one("type_attempts").unwrap();
    let shooting_times: usize = *sub.get_one("shooting_times").unwrap();
    let egg = sub.get_flag("egg");
    let single_end = sub.get_flag("single-end");
    let rg_freq: usize = *sub.get_one("rg_frequency").unwrap();
    let data = fs::read_to_string(s).expect("Failed to read source circuit");

    let lmdb_path = "./db";
    let _ = std::fs::create_dir_all(lmdb_path);

    let env = Environment::new()
        .set_max_readers(10000)
        .set_max_dbs(556)
        .set_map_size(800 * 1024 * 1024 * 1024)
        .open(Path::new(lmdb_path))
        .expect("Failed to open lmdb");

    let (shard_dbs, curated_shard_dbs) = open_all_dbs(&env);
    install_kill_handler();
    if data.trim().is_empty() {
        println!("Empty file");
        return;
    }

    let c = CircuitSeq::from_string(&data);
    main_shuffle_shoot_shuffle(
        &c,
        rounds,
        n,
        m,
        x,
        d,
        s,
        &env,
        &shard_dbs,
        &curated_shard_dbs,
        leave,
        do_gadgetize,
        gadget_path,
        full_shuffle,
        gates_ahead_expand,
        gates_ahead_samf,
        type_attempts,
        shooting_times,
        egg,
        rg_freq,
        single_end,
    );
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
