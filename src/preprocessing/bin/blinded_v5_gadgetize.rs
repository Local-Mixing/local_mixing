//! Standalone driver for the blinded-V5 computation-stage gadgetizer
//! (see `local_mixing::preprocessing::blinded_v5`). Reads an mpmct1 source,
//! gadgetizes it, writes the result. The same gadgetizer is wired into
//! `gen_sandwich_gadget` as the `blinded-v5` mode with the settled production
//! preset (K=16, R=n, N=floor, masked-scratch discipline, blinded read).
//!
//! Usage: blinded_v5_gadgetize <src.mpmct1> <out.mpmct1> [K=16] [R=0(auto=n)]
//!            [N=0(floor)] [seed=1] [mode=blinded|cnot|plain] [rerand_dose=0]
//!            [discipline=0|1|2] [rerand_mode=repair|clearing|adaptive]
//!            [active_wires=0]

use local_mixing::engine::format::{read_mpmct, write_mpmct};
use local_mixing::preprocessing::blinded_v5::{
    BlindedV5Params, RerandMode, gadgetize_blinded_v5,
};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!(
            "usage: blinded_v5_gadgetize <src> <out> [K=16] [R=0(auto=n)] [N=0(floor)] \
             [seed=1] [blinded|cnot|plain] [rerand_dose=0] [discipline=0|1|2] \
             [repair|clearing|adaptive] \
             [active_wires=0(all; set to n for a 2n-wire zero-slice sandwich)]\n\
             modes: blinded = t-pin read (k2->k3); cnot = single-CNOT blinded read (leaks d+e at k2); \
             plain = unmask-read-remask (leaks k1)\n\
             rerand: repair = generation-time band refresh with live-mask repair (default; \
             always achieves the dose, spread evenly through the run); \
             clearing = the post-pass commutation clearing (slow, ~0.6% accept, \
             lands at the head of the gate list); adaptive = float-then-trial"
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
    let rerand_mode = match a.get(10).map(|s| s.as_str()).unwrap_or("repair") {
        "repair" => RerandMode::Repair,
        "clearing" | "structural" => RerandMode::Clearing,
        "adaptive" => RerandMode::Adaptive,
        other => panic!("unknown rerand mode {other:?} (repair|clearing|adaptive)"),
    };
    let active_wires: usize = a.get(11).map(|s| s.parse().unwrap()).unwrap_or(0);

    let (src, np) = read_mpmct(src_path).expect("read source");
    let params = BlindedV5Params {
        k,
        r,
        n_target,
        seed,
        blinded: mode == "blinded" || mode == "cnot",
        tpin: mode == "blinded",
        rerand_dose,
        discipline,
        rerand_mode,
        active_wires,
    };
    let g = gadgetize_blinded_v5(&src, np, &params);
    write_mpmct(out_path, &g.gates, g.num_wires).expect("write out");
    println!(
        "{out_path}: mode={mode} K={k} R={} N_target={n_target} \
         rerand={}/{rerand_dose} ({:?}, {} gates) discipline={discipline} | \
         {} gates, {} atoms, {} wires (src {} gates / {np} wires)",
        g.r_used,
        g.rerand_done,
        params.rerand_mode,
        g.rerand_gates,
        g.gates.len(),
        g.atoms,
        g.num_wires,
        src.len()
    );
}
