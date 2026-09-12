//! CLI wrapper over `local_mixing::db_generation::frozen_build`: convert a sharded
//! replacement store into a frozen-table directory.
//!
//! Subcommands (run tables, then write, then validate):
//!   frozen_from_lmdb tables   <input> <out_dir> [--curated] [--composite] [per_range]
//!   frozen_from_lmdb write    <input> <out_dir> [--curated] [--composite]
//!   frozen_from_lmdb validate <input> <out_dir> [--curated] [--composite]
//!
//! --curated reads the "curated_XX" dbs (build FROZEN_CURATED_DIR); default
//! reads the regular "XX" dbs (build FROZEN_DB_DIR). Curated conversion
//! requires the completion marker written by `build_curated_full to-lmdb`.
//!
//! --composite reads a curated-full composite RocksDB store directly instead
//! of an LMDB directory. This is the only route for the uncapped curated
//! database: LMDB caps a non-DUPSORT value at `MAXDATASIZE` (4 GiB - 1) and the
//! largest curated key needs ~16 GiB, so `to-lmdb` cannot represent it at all.
//! The frozen encoding is identical either way -- the composite store yields
//! the same keys in the same order with the same value bytes.

use local_mixing::db_generation::frozen_build::{
    LmdbShards, ShardReader, require_curated_full_manifest, stage_tables, stage_validate,
    stage_write,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let curated = args.iter().any(|a| a == "--curated");
    let composite = args.iter().any(|a| a == "--composite");
    let prefix = if curated { "curated_" } else { "" };
    // Drop keys whose shortest circuit has fewer than N gates. Composite input
    // only -- an LMDB value is already a flat blob with no per-key rebuild step
    // to filter in. Must be identical across tables/write/validate or the
    // stages disagree about which keys exist.
    let min_gates_position = args.iter().position(|a| a == "--min-gates");
    let min_gates: usize = match min_gates_position {
        None => 0,
        Some(index) => args
            .get(index + 1)
            .unwrap_or_else(|| invalid("--min-gates requires a value"))
            .parse()
            .unwrap_or_else(|_| invalid("--min-gates must be a nonnegative integer")),
    };
    if min_gates > 0 && !composite {
        eprintln!("--min-gates requires --composite");
        std::process::exit(2);
    }
    let skip: Vec<String> = args
        .iter()
        .position(|a| a == "--min-gates")
        .map(|i| {
            vec![
                args[i].clone(),
                args.get(i + 1).cloned().unwrap_or_default(),
            ]
        })
        .unwrap_or_default();
    let pos: Vec<&String> = args
        .iter()
        .filter(|a| *a != "--curated" && *a != "--composite" && !skip.contains(a))
        .collect();

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

    if composite && curated {
        // A composite store carries its own completion manifest, checked on open.
    } else if curated {
        require_curated_full_manifest(input);
    }

    let source: Box<dyn ShardReader> = if composite {
        #[cfg(feature = "legacy-db-tools")]
        {
            if min_gates > 0 {
                eprintln!("[source] composite {input}, dropping keys with min-gates < {min_gates}");
            }
            Box::new(
                local_mixing::db_generation::frozen_build::CompositeShards::open_with_min_gates(
                    input, min_gates,
                ),
            )
        }
        #[cfg(not(feature = "legacy-db-tools"))]
        {
            eprintln!("--composite requires the legacy-db-tools feature");
            std::process::exit(2);
        }
    } else {
        Box::new(LmdbShards::open(input, prefix))
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
        "usage: frozen_from_lmdb tables|write|validate <input> <out_dir> [--curated] [--composite] [per_range]"
    );
    std::process::exit(2);
}
