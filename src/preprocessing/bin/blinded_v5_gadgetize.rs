//! Standalone driver for the blinded-V5 computation-stage gadgetizer
//! (see `local_mixing::preprocessing::blinded_v5`). Reads an mpmct1 source,
//! gadgetizes it, writes the result. The same gadgetizer is wired into
//! `gen_sandwich_gadget` as the `blinded-v5` mode with the settled production
//! preset (K=16, R=n, N=floor, masked-scratch discipline, blinded read).
//!
//! Usage: blinded_v5_gadgetize <src.mpmct1> <out.mpmct1> [K=16] [R=0(auto=n)]
//!            [N=0(floor)] [seed=1] [mode=blinded|plain] [rerand_dose=0]
//!            [discipline=0|1|2] [rerand_mode=structural|adaptive]

use local_mixing::engine::format::{read_mpmct, write_mpmct};
use local_mixing::preprocessing::blinded_v5::{BlindedV5Params, gadgetize_blinded_v5};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!(
            "usage: blinded_v5_gadgetize <src> <out> [K=16] [R=0(auto=n)] [N=0(floor)] \
             [seed=1] [blinded|plain] [rerand_dose=0] [discipline=0|1|2] [structural|adaptive]"
        );
        std::process::exit(2);
    }
    let src_path = &a[1];
    let out_path = &a[2];
    let k: usize = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(16);
    let r: usize = a.get(4).map(|s| s.parse().unwrap()).unwrap_or(0);
    let n_target: usize = a.get(5).map(|s| s.parse().unwrap()).unwrap_or(0);
    let seed: u64 = a.get(6).map(|s| s.parse().unwrap()).unwrap_or(1);
    let mode = a.get(7).map(|s| s.as_str()).unwrap_or("blinded");
    let rerand_dose: usize = a.get(8).map(|s| s.parse().unwrap()).unwrap_or(0);
    let discipline: usize = a.get(9).map(|s| s.parse().unwrap()).unwrap_or(2);
    let rerand_mode = a.get(10).map(|s| s.as_str()).unwrap_or("structural");

    let (src, np) = read_mpmct(src_path).expect("read source");
    let params = BlindedV5Params {
        k,
        r,
        n_target,
        seed,
        blinded: mode == "blinded",
        rerand_dose,
        discipline,
        adaptive_rerand: rerand_mode == "adaptive",
    };
    let g = gadgetize_blinded_v5(&src, np, &params);
    write_mpmct(out_path, &g.gates, g.num_wires).expect("write out");
    println!(
        "{out_path}: mode={mode} K={k} R={} N_target={n_target} rerand={}/{rerand_dose} \
         discipline={discipline} | {} gates, {} atoms, {} wires (src {} gates / {np} wires)",
        g.r_used,
        g.rerand_done,
        g.gates.len(),
        g.atoms,
        g.num_wires,
        src.len()
    );
}
