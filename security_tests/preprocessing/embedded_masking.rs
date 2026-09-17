//! Standalone driver for the embedded-masking computation-stage gadgetizer
//! (see `local_mixing::stages::preprocessing::embedded_masking`). Reads an mpmct1 source A,
//! gadgetizes it, writes the result. The same gadgetizer is wired into
//! `gen_sandwich_gadget` as the `embedded-masking` mode with the production preset.
//!
//! Usage: embedded_masking_gadgetize <src.mpmct1> <out.mpmct1> [K=2] [R=0(auto=n)]
//!            [seed=1] [rerand_level=0(auto=m/4k slots)] [max_open=3]
//!            [active_wires=0] [extra_lgis=2] [rerand_repair=0] [rerand_burst=0(auto=8k)]
//!            [min_mask=0(auto=max_open)]
//! EMBEDDED_MASKING_SHUFFLING=8 enables internal carrier transfers. This driver
//! requires return-home layout so its emitted circuit keeps physical data ports.

use local_mixing::circuit::formats::{read_mpmct, write_mpmct};
use local_mixing::stages::preprocessing::embedded_masking::{
    EmbeddedMaskingParams, preprocess_embedded_masking, seed_band_mode,
};

fn main() {
    let mut obsolete_mask_controls: Vec<String> = std::env::vars_os()
        .filter_map(|(key, _)| key.into_string().ok())
        .filter(|key| key.starts_with("BV5_"))
        .collect();
    obsolete_mask_controls.sort();
    if !obsolete_mask_controls.is_empty() {
        eprintln!(
            "obsolete masking controls: {}; use EMBEDDED_MASKING_* instead",
            obsolete_mask_controls.join(", ")
        );
        std::process::exit(2);
    }
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!(
            "usage: embedded_masking_gadgetize <src> <out> [K=2] [R=0(auto=n)] [seed=1] \
             [rerand_level=0(auto=m/4k slots)] [max_open=3] \
             [active_wires=0(all; set to n for a 2n-wire zero-slice sandwich)] \
             [extra_lgis=2] [rerand_repair=0] [rerand_burst=0(auto=8k)] \
             [min_mask=0(auto=max_open)]"
        );
        std::process::exit(2);
    }
    let src_path = &a[1];
    let out_path = &a[2];
    let k: usize = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(2);
    let r: usize = a.get(4).map(|s| s.parse().unwrap()).unwrap_or(0);
    let seed: u64 = a.get(5).map(|s| s.parse().unwrap()).unwrap_or(1);
    let rerand_level: usize = a.get(6).map(|s| s.parse().unwrap()).unwrap_or(0);
    let max_open: usize = a.get(7).map(|s| s.parse().unwrap()).unwrap_or(3);
    let active_wires: usize = a.get(8).map(|s| s.parse().unwrap()).unwrap_or(0);
    let extra_lgis: usize = a.get(9).map(|s| s.parse().unwrap()).unwrap_or(2);
    let rerand_repair: usize = a.get(10).map(|s| s.parse().unwrap()).unwrap_or(0);
    let rerand_burst: usize = a.get(11).map(|s| s.parse().unwrap()).unwrap_or(0);
    let min_mask: usize = a.get(12).map(|s| s.parse().unwrap()).unwrap_or(0);
    let shuffling_segments = std::env::var("EMBEDDED_MASKING_SHUFFLING")
        .map(|value| {
            value
                .parse::<usize>()
                .expect("EMBEDDED_MASKING_SHUFFLING must be 0 or at least 8")
        })
        .unwrap_or(0);
    assert!(
        shuffling_segments == 0 || shuffling_segments >= 8,
        "EMBEDDED_MASKING_SHUFFLING must be 0 or at least 8"
    );
    let shuffling_return_home = std::env::var("EMBEDDED_MASKING_SHUFFLING_RETURN_HOME")
        .map(|value| match value.as_str() {
            "1" | "true" => true,
            "0" | "false" => false,
            _ => panic!("EMBEDDED_MASKING_SHUFFLING_RETURN_HOME must be 0, 1, false, or true"),
        })
        .unwrap_or(true);
    assert!(
        shuffling_segments == 0 || shuffling_return_home,
        "standalone gadgetization requires return-home shuffling to preserve physical data ports"
    );

    let (src, np) = read_mpmct(src_path).expect("read source");
    let params = EmbeddedMaskingParams {
        k,
        r,
        seed,
        rerand_level,
        rerand_repair,
        rerand_burst,
        max_open,
        min_mask,
        active_wires,
        extra_lgis,
        shuffling_segments,
        shuffling_return_home,
        quad_fire: std::env::var("EMBEDDED_MASKING_QUAD_FIRE").map_or(true, |v| v != "0"),
        balanced: std::env::var("EMBEDDED_MASKING_BALANCED").map_or(true, |v| v != "0"),
        burst_band_only: std::env::var("EMBEDDED_MASKING_BURST_BANDONLY").is_ok_and(|v| v != "0"),
        encoded_io: false,
        min_open: std::env::var("EMBEDDED_MASKING_MIN_OPEN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2),
        ..EmbeddedMaskingParams::production(seed)
    };
    let g = preprocess_embedded_masking(&src, np, &params);
    // Band-seeding module pipelined in front (the compute only reads the band).
    let r_used = if params.r == 0 { np } else { params.r };
    // Modules 2 and 4: band seed BEFORE the compute and re-seed AFTER it. The
    // five parts are always five separate modules; the compute's internal rerand
    // bursts do not discharge stage 4.
    let mut gates = seed_band_mode(
        np,
        r_used,
        active_wires,
        seed ^ 0x5EED_B00C,
        params.balanced,
    );
    gates.extend(g.gates.iter().cloned());
    gates.extend(seed_band_mode(
        np,
        r_used,
        active_wires,
        seed ^ 0xB00C_5EED,
        params.balanced,
    ));
    write_mpmct(out_path, &gates, g.num_wires).expect("write out");
    println!(
        "{out_path}: K={k} R={} rerand={} gates \
         (straddle {rerand_level}+repair {rerand_repair} slots x burst {rerand_burst}[0=auto 8k]) \
         max_open={max_open} | \
         {} gates ({} +band-seed), {} atoms, {} wires (src {} gates / {np} wires)",
        g.r_used,
        g.rerand_done,
        gates.len(),
        g.gates.len(),
        g.atoms,
        g.num_wires,
        src.len()
    );
    if shuffling_segments > 0 {
        println!(
            "embedded-masking shuffling: segments={shuffling_segments} return_home=true transfers={} added_gates={} skipped_cuts={}",
            g.shuffling_stats.transfer_count,
            g.shuffling_stats.added_gates,
            g.shuffling_stats.skipped_cut_count,
        );
    }
}
