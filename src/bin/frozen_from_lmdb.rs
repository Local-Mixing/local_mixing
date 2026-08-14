//! CLI wrapper over `local_mixing::replace::frozen_build`: convert the legacy
//! sharded LMDB replacement store into a frozen-table directory.
//!
//! Subcommands (run tables, then write, then validate):
//!   frozen_from_lmdb tables   <lmdb_dir> <out_dir> [--curated] [per_range]
//!   frozen_from_lmdb write    <lmdb_dir> <out_dir> [--curated]
//!   frozen_from_lmdb validate <lmdb_dir> <out_dir> [--curated]
//!
//! --curated reads the "curated_XX" dbs (build FROZEN_CURATED_DIR); default
//! reads the regular "XX" dbs (build FROZEN_DB_DIR). Curated conversion
//! requires the completion marker written by `build_curated_full to-lmdb`.

use local_mixing::replace::frozen_build::{
    require_curated_full_manifest, stage_tables, stage_validate, stage_write,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let curated = args.iter().any(|a| a == "--curated");
    let prefix = if curated { "curated_" } else { "" };
    let pos: Vec<&String> = args.iter().filter(|a| *a != "--curated").collect();
    if curated {
        let Some(lmdb_dir) = pos.get(2) else {
            eprintln!(
                "usage: frozen_from_lmdb tables|write|validate <lmdb_dir> <out_dir> [--curated] [per_range]"
            );
            std::process::exit(2);
        };
        require_curated_full_manifest(lmdb_dir);
    }
    match pos.get(1).map(|s| s.as_str()) {
        Some("tables") => {
            let per: usize = pos.get(4).and_then(|s| s.parse().ok()).unwrap_or(400_000);
            stage_tables(pos[2], prefix, pos[3], per);
        }
        Some("write") => stage_write(pos[2], prefix, pos[3]),
        Some("validate") => stage_validate(pos[2], prefix, pos[3]),
        _ => {
            eprintln!(
                "usage: frozen_from_lmdb tables|write|validate <lmdb_dir> <out_dir> [--curated] [per_range]"
            );
            std::process::exit(2);
        }
    }
}
