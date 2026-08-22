//! Throwaway harness for the product-share encoding's n=16 degree-1 ridge grid
//! (mirrors the RG4 validation methodology: gadgetize ONCE, no mixing, so the
//! heatmap isolates the ENCODING's effect on the progress diagonal). Reads a
//! g57 C, gadgetizes with the product-share encoding at a given k, writes the
//! pre-mixing G as mpmct1 for hmap_affine. Also prints the endpoint Hamming
//! self-check the ridge reader needs (H≈0 reference validity).
//!
//! Usage: prod_grid <c.g57> <n> <k> <band> <seed> <out.mpmct1> [deg] [k_hi] [deg_hi] [max_width] [fill_nl] [roll]
//!        (band 0 = auto; k 0 = plain gadget; deg defaults to 2; k_hi tower
//!         terms of degree deg_hi for a mixed deg-k + deg_hi-k_hi design;
//!         max_width 2 = narrow g57 mode; fill_nl > 0 = nonlinear band fill;
//!         roll > 0 = rolling band relocations per inter-SG gap)

use local_mixing::circuit::CircuitSeq;
use local_mixing::circuit::U1024;
use local_mixing::engine::format::{read_g57_file, write_mpmct};
use local_mixing::circuit::xgate::eval_u1024;
use local_mixing::preprocessing::gadgets::{
    gadgetize_cnot, gadgetize_cnot_single, MaskConfig, ProdConfig,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn low_mask(n: usize) -> U1024 {
    (U1024::one() << n) - U1024::one()
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let c_path = &a[0];
    let n: usize = a[1].parse().unwrap();
    let k: usize = a[2].parse().unwrap();
    let band: usize = a[3].parse().unwrap();
    let seed: u64 = a[4].parse().unwrap();
    let out = &a[5];
    let deg: usize = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(2);
    let k_hi: usize = a.get(7).and_then(|s| s.parse().ok()).unwrap_or(0);
    let deg_hi: usize = a.get(8).and_then(|s| s.parse().ok()).unwrap_or(3);

    let c_gates = read_g57_file(c_path).expect("read c g57");
    // g57 file -> CircuitSeq triples (g57 gate = target,x,y).
    let triples: Vec<[u16; 3]> = c_gates
        .iter()
        .map(|g| {
            // from_g57 stores ctrls sorted; recover [target, x(neg), y(pos)]
            // where fires = x OR !y. as_g57 convention: neg literal = x, pos = y.
            let mut xw = 0u16;
            let mut yw = 0u16;
            for &(w, pol) in &g.ctrls {
                if pol {
                    yw = w
                } else {
                    xw = w
                }
            }
            [g.target, xw, yw]
        })
        .collect();
    let main = CircuitSeq { gates: triples };

    let max_width: usize = a.get(9).and_then(|s| s.parse().ok()).unwrap_or(0);
    let fill_nl: usize = a.get(10).and_then(|s| s.parse().ok()).unwrap_or(0);
    let roll: usize = a.get(11).and_then(|s| s.parse().ok()).unwrap_or(0);
    let src_dist: usize = a.get(12).and_then(|s| s.parse().ok()).unwrap_or(0);
    let prod = ProdConfig {
        k,
        deg,
        k_hi,
        deg_hi,
        band,
        rsrc: 1,
        max_width,
        fill_nl,
        roll,
        src_dist,
        src_horizon: 0,
        src_lo: a.get(13).and_then(|s| s.parse().ok()).unwrap_or(0),
        src_hi: a.get(14).and_then(|s| s.parse().ok()).unwrap_or(0),
        single: a.get(15).and_then(|s| s.parse().ok()).unwrap_or(0),
        fill_pivots: a.get(16).and_then(|s| s.parse().ok()).unwrap_or(0),
        epoch: a.get(17).and_then(|s| s.parse().ok()).unwrap_or(0),
        refill_data: a.get(18).and_then(|s| s.parse().ok()).unwrap_or(0),
        g57_narrow: a.get(19).and_then(|s| s.parse().ok()).unwrap_or(0),
        ladder_cap: a.get(20).and_then(|s| s.parse().ok()).unwrap_or(0),
        cg_jitter: a.get(21).and_then(|s| s.parse().ok()).unwrap_or(0),
        rung_menu: a.get(22).and_then(|s| s.parse().ok()).unwrap_or(0),
        gray_fold: a.get(23).and_then(|s| s.parse().ok()).unwrap_or(0),
        swap_refresh: a.get(24).and_then(|s| s.parse().ok()).unwrap_or(0),
        close_slice: 0,
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let g = if prod.single_carrier() {
        gadgetize_cnot_single(&main, n, 1, &prod, &mut rng)
    } else {
        gadgetize_cnot(&main, n, 1, &MaskConfig::off(), &prod, &mut rng)
    };

    // Endpoint self-check: G(x,0..) low n wires == C(x) for random x.
    let mut chk = StdRng::seed_from_u64(0xa11ce);
    let mut mismatches = 0;
    for _ in 0..2000 {
        let x = {
            use rand::Rng;
            U1024::from(chk.random::<u64>() as u128 & ((1u128 << n.min(63)) - 1)) & low_mask(n)
        };
        let expected = main.evaluate_1024(x) & low_mask(n);
        let got = eval_u1024(&g.gates, x) & low_mask(n);
        if got != expected {
            mismatches += 1;
        }
    }
    write_mpmct(out, &g.gates, g.num_wires).expect("write mpmct1");
    println!(
        "[prod_grid] n={n} k={k} deg={deg} k_hi={k_hi} deg_hi={deg_hi} band={} -> {} gates, {} wires, endpoint_mismatches={mismatches} -> {out}",
        prod.band_size(n),
        g.gates.len(),
        g.num_wires
    );
}
