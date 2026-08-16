//! CLI wrapper over `local_mixing::replace::frozen_build`: convert a sharded
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

use local_mixing::replace::frozen_build::{
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
    let min_gates: usize = args
        .iter()
        .position(|a| a == "--min-gates")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if min_gates > 0 && !composite {
        eprintln!("--min-gates requires --composite");
        std::process::exit(2);
    }
    let skip: Vec<String> = args
        .iter()
        .position(|a| a == "--min-gates")
        .map(|i| vec![args[i].clone(), args.get(i + 1).cloned().unwrap_or_default()])
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
                local_mixing::replace::frozen_build::CompositeShards::open_with_min_gates(
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

    match pos.get(1).map(|s| s.as_str()) {
        Some("tables") => {
            let per: usize = pos.get(4).and_then(|s| s.parse().ok()).unwrap_or(400_000);
            stage_tables(source.as_ref(), out_dir, per);
        }
        Some("write") => stage_write(source.as_ref(), out_dir),
        Some("validate") => stage_validate(source.as_ref(), out_dir),
        _ => usage(),
    }
}

fn usage() -> ! {
    eprintln!(
        "usage: frozen_from_lmdb tables|write|validate <input> <out_dir> [--curated] [--composite] [per_range]"
    );
    std::process::exit(2);
}
