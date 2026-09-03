//! Standalone driver for the blinded-V5 computation-stage gadgetizer
//! (see `local_mixing::preprocessing::blinded_v5`). Reads an mpmct1 source A,
//! gadgetizes it, writes the result. The same gadgetizer is wired into
//! `gen_sandwich_gadget` as the `blinded-v5` mode with the production preset.
//!
//! Usage: blinded_v5_gadgetize <src.mpmct1> <out.mpmct1> [K=16] [R=0(auto=n)]
//!            [seed=1] [rerand_level=0] [max_open=3] [active_wires=0]

use local_mixing::engine::format::{read_mpmct, write_mpmct};
use local_mixing::preprocessing::blinded_v5::{BlindedV5Params, gadgetize_blinded_v5};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!(
            "usage: blinded_v5_gadgetize <src> <out> [K=16] [R=0(auto=n)] [seed=1] \
             [rerand_level=0] [max_open=3] \
             [active_wires=0(all; set to n for a 2n-wire zero-slice sandwich)]"
        );
        std::process::exit(2);
    }
    let src_path = &a[1];
    let out_path = &a[2];
    let k: usize = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(16);
    let r: usize = a.get(4).map(|s| s.parse().unwrap()).unwrap_or(0);
    let seed: u64 = a.get(5).map(|s| s.parse().unwrap()).unwrap_or(1);
    let rerand_level: usize = a.get(6).map(|s| s.parse().unwrap()).unwrap_or(0);
    let max_open: usize = a.get(7).map(|s| s.parse().unwrap()).unwrap_or(3);
    let active_wires: usize = a.get(8).map(|s| s.parse().unwrap()).unwrap_or(0);

    let (src, np) = read_mpmct(src_path).expect("read source");
    let params = BlindedV5Params {
        k,
        r,
        seed,
        rerand_level,
        max_open,
        active_wires,
    };
    let g = gadgetize_blinded_v5(&src, np, &params);
    write_mpmct(out_path, &g.gates, g.num_wires).expect("write out");
    println!(
        "{out_path}: K={k} R={} rerand={}/{rerand_level} max_open={max_open} | \
         {} gates, {} atoms, {} wires (src {} gates / {np} wires)",
        g.r_used,
        g.rerand_done,
        g.gates.len(),
        g.atoms,
        g.num_wires,
        src.len()
    );
}
