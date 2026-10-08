//! CLI wrapper over `local_mixing::db_generation::frozen_build`: convert a sharded
//! regular replacement store into a frozen-table directory.
//!
//! Subcommands (run tables, then write, then validate):
//!   frozen_from_lmdb tables   <input> <out_dir> [--bands] [per_range]
//!   frozen_from_lmdb write    <input> <out_dir> [--bands]
//!   frozen_from_lmdb validate <input> <out_dir> [--bands]
//!
//! The default input is an LMDB directory with the regular "XX" shard dbs
//! (build FROZEN_DB_DIR). --bands instead reads a manifest file listing
//! regular RocksDB band directories (one per line) and k-way merges them.

use local_mixing::db_generation::frozen_build::{
    LmdbShards, ShardReader, stage_tables, stage_validate, stage_write,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bands = args.iter().any(|a| a == "--bands");
    let pos: Vec<&String> = args.iter().filter(|a| *a != "--bands").collect();

    let Some(input) = pos.get(2) else {
        usage();
    };
    let Some(out_dir) = pos.get(3) else {
        usage();
    };
    let stage = pos
        .get(1)
        .map(|value| value.as_str())
        .unwrap_or_else(|| usage());
    let per_range = match (stage, pos.get(4)) {
        ("tables", Some(value)) => value
            .parse::<usize>()
            .ok()
            .filter(|&value| value > 0)
            .unwrap_or_else(|| invalid("per_range must be a positive integer")),
        ("tables", None) => 400_000,
        ("write" | "validate", None) => 0,
        ("write" | "validate", Some(_)) => invalid("unexpected positional argument"),
        _ => usage(),
    };
    prepare_output(stage, out_dir);

    let source: Box<dyn ShardReader> = if bands {
        #[cfg(feature = "db-tools")]
        {
            let dirs: Vec<String> = std::fs::read_to_string(input)
                .unwrap_or_else(|e| {
                    eprintln!("read bands manifest {input}: {e}");
                    std::process::exit(2);
                })
                .lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .collect();
            eprintln!("[source] {} regular bands (direct k-way merge)", dirs.len());
            Box::new(local_mixing::db_generation::frozen_build::MultiBandShards::open(&dirs))
        }
        #[cfg(not(feature = "db-tools"))]
        {
            eprintln!("--bands requires the db-tools feature");
            std::process::exit(2);
        }
    } else {
        Box::new(LmdbShards::open(input, ""))
    };

    match stage {
        "tables" => stage_tables(source.as_ref(), out_dir, per_range),
        "write" => stage_write(source.as_ref(), out_dir),
        "validate" => stage_validate(source.as_ref(), out_dir),
        _ => usage(),
    }
}

fn prepare_output(stage: &str, out_dir: &str) {
    let directory = std::path::Path::new(out_dir);
    match stage {
        "tables" => {
            if directory.exists() {
                invalid(&format!(
                    "tables output must not already exist: {}",
                    directory.display()
                ));
            }
            std::fs::create_dir(directory).unwrap_or_else(|error| {
                invalid(&format!(
                    "cannot create fresh output directory {}: {error}",
                    directory.display()
                ))
            });
        }
        "write" => {
            if !directory.join("tables.bin").is_file() {
                invalid("write requires an existing tables.bin from the tables stage");
            }
            if directory.join("filters.bin").exists() {
                invalid("write refuses an existing filters.bin");
            }
            for shard in 0..256usize {
                let path = directory.join(format!("shard_{shard:02x}.frz"));
                if path.exists() {
                    invalid(&format!(
                        "write refuses existing shard output: {}",
                        path.display()
                    ));
                }
            }
        }
        "validate" => {
            if !directory.join("tables.bin").is_file() {
                invalid("validate requires tables.bin");
            }
            for shard in 0..256usize {
                let path = directory.join(format!("shard_{shard:02x}.frz"));
                if !path.is_file() {
                    invalid(&format!("validate requires shard: {}", path.display()));
                }
            }
        }
        _ => usage(),
    }
}

fn invalid(message: &str) -> ! {
    eprintln!("error: {message}");
    std::process::exit(2);
}

fn usage() -> ! {
    eprintln!(
        "usage: frozen_from_lmdb tables|write|validate <input> <out_dir> [--bands] [per_range]"
    );
    std::process::exit(2);
}
