//! Rebuild a curated frozen store with per-permutation candidate pools
//! swapped in (see frozen_build::pool_swap_upto): entries with minimal
//! retained friend 1..=max_min gates get their pools from the mgdb manifest
//! (m1_pool file overrides the mn==1 entry when given; pass "-" to route M1
//! through the manifest too). Layerable over an already-swapped store.
//!
//! Usage: frozen_pool_swap <src_store> <out_store> <m1_pool.sgdb1|-> <mgdb_dir> [max_min=3]
use local_mixing::db_generation::frozen_build::pool_swap_upto;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 5 {
        eprintln!("usage: frozen_pool_swap <src_store> <out_store> <m1_pool.sgdb1|-> <mgdb_dir> [max_min=3]");
        std::process::exit(2);
    }
    let m1 = if a[3] == "-" { None } else { Some(a[3].as_str()) };
    let max_min = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(3);
    pool_swap_upto(&a[1], &a[2], m1, &a[4], max_min);
}
