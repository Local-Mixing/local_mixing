//! Sample a sliced sandwich and gadgetize it with the current diversified
//! scheme, writing the gadgetized circuit (mpmct1) ready for fmix.
//!
//! Pipeline: fresh random g57 C on n wires -> sliced_sandwich_cnot (samples a
//! random D, interleaves the two slice blocks, floats the N column) -> a 2n-wire
//! sandwich A with A(x,0)=(junk, C(x)) on the zero slice ->
//! gadgetize_xgates_with_slice_zero_ccnot (the diversified CG menu + nonlinear
//! {RG1,RG2,RG3} RGs + final commuting shuffle, behind a zero-slice preblock)
//! -> a 4n-wire gadget whose low 2n output equals A on the gadget's zero slice.
//!
//! Usage: gen_sandwich_gadget <out> [n=128] [m_C=3000] [m_D=3000]
//!                            [s=n*log2 n] [rg_freq=1] [slice_gates=10*2n]
//!                            [seed=1] [gadget_seed=seed] [sandwich_seed=seed]
//!
//! `seed` fixes C only (fastrand); `sandwich_seed` drives D + slicing +
//! N-float (default = seed); `gadget_seed` drives the gadgetization only.
//! So: vary gadget_seed alone = re-gadgetize the SAME sandwich A; vary
//! sandwich_seed (+ gadget_seed) with seed fixed = a FRESH sandwich around
//! the SAME C. The sandwich is dumped to `<out>.sandwich.mpmct1` so same-A /
//! fresh-A across runs is checkable byte-for-byte.

use local_mixing::circuit::circuit::U1024;
use local_mixing::postmix::format::write_mpmct;
use local_mixing::postmix::xgate::eval_u1024;
use local_mixing::random::random_data::random_circuit;
use local_mixing::replace::gadgets::{
    gadgetize_xgates_with_slice_zero_ccnot, gadgetize_xgates_with_slice_zero_ccnot_single,
    sandwich_default_s, sliced_sandwich_cnot, MaskConfig, ProdConfig,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn mask_bits(bits: usize) -> U1024 {
    if bits >= 1024 {
        U1024::MAX
    } else {
        (U1024::one() << bits) - U1024::one()
    }
}

fn main() {
    let mut a = std::env::args().skip(1);
    let out = a
        .next()
        .expect("usage: gen_sandwich_gadget <out> [n m_C m_D s rg_freq slice_gates seed]");
    let n: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(128);
    let m_c: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(3000);
    let m_d: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(3000);
    let s: usize = a
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| sandwich_default_s(n));
    let rg_freq: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let sandwich_n = 2 * n;
    let slice_gates: usize = a
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10 * sandwich_n);
    let seed: u64 = a.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let gadget_seed: u64 = a.next().and_then(|s| s.parse().ok()).unwrap_or(seed);
    let sandwich_seed: u64 = a.next().and_then(|s| s.parse().ok()).unwrap_or(seed);

    println!(
        "[gen] n={n} |C|={m_c} |D|={m_d} s={s} rg_freq={rg_freq} slice_gates={slice_gates} seed={seed} gadget_seed={gadget_seed} sandwich_seed={sandwich_seed}"
    );

    // Fresh random g57 source C (fastrand-seeded, matching sandwich_compare).
    fastrand::seed(seed);
    let c = random_circuit(n, m_c);
    // Also dump C in g57 format so hmap_affine can reconstruct against the
    // ORIGINAL computation. Regenerating with the same seed reproduces C (and
    // the whole gadget) bit-for-bit, so this recovers a past run's source.
    let c_path = format!("{out}.source_c.g57");
    std::fs::write(&c_path, c.repr()).expect("write source C");
    println!(
        "[gen] wrote source C ({} g57 gates) to {c_path}",
        c.gates.len()
    );
    let mut rng = StdRng::seed_from_u64(sandwich_seed ^ 0x5150_1CED);

    let sandwich = sliced_sandwich_cnot(&c, n, m_d, s, &mut rng);
    println!(
        "[gen] sliced sandwich: {} gates, {} wires",
        sandwich.gates.len(),
        sandwich.num_wires
    );
    let a_path = format!("{out}.sandwich.mpmct1");
    write_mpmct(&a_path, &sandwich.gates, sandwich.num_wires).expect("write sandwich");
    println!("[gen] wrote sandwich A to {a_path}");

    // Gadgetization runs on its own stream: same `seed` + different
    // `gadget_seed` = fresh gadgetization of the identical sandwich.
    let mut rng = StdRng::seed_from_u64(gadget_seed ^ 0x6AD6_E75E);

    // Product-share encoding via env vars (PROD_K base deg-PROD_DEG terms +
    // PROD_K_HI tower deg-PROD_DEG_HI terms). Default off = plain gadget.
    let env = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    // Same rule as the sss path: the validated production setting is the
    // default, and an environment variable overrides one field of it.
    let preset = ProdConfig::production_single();
    let prod = ProdConfig {
        k: env("PROD_K", preset.k),
        deg: env("PROD_DEG", preset.deg),
        k_hi: env("PROD_K_HI", preset.k_hi),
        deg_hi: env("PROD_DEG_HI", preset.deg_hi),
        band: env("PROD_BAND", preset.band),
        rsrc: env("PROD_RSRC", preset.rsrc),
        max_width: env("PROD_MAX_WIDTH", preset.max_width),
        fill_nl: env("PROD_FILL_NL", preset.fill_nl),
        roll: env("PROD_ROLL", preset.roll),
        src_dist: env("PROD_SRC_DIST", preset.src_dist),
        src_horizon: env("PROD_SRC_HORIZON", preset.src_horizon),
        src_lo: env("PROD_SRC_LO", preset.src_lo),
        src_hi: env("PROD_SRC_HI", preset.src_hi),
        fill_pivots: env("PROD_FILL_PIVOTS", preset.fill_pivots),
        g57_narrow: env("PROD_G57_NARROW", preset.g57_narrow),
        ladder_cap: env("PROD_LADDER_CAP", preset.ladder_cap),
        cg_jitter: env("PROD_CG_JITTER", preset.cg_jitter),
        rung_menu: env("PROD_RUNG_MENU", preset.rung_menu),
        epoch: env("PROD_EPOCH", preset.epoch),
        refill_data: env("PROD_REFILL_DATA", preset.refill_data),
        single: env("PROD_SINGLE", preset.single),
        gray_fold: env("PROD_GRAY_FOLD", preset.gray_fold),
    };
    if prod.enabled() {
        println!(
            "[gen] product-share encoding ON: k={} deg={} k_hi={} deg_hi={} band(auto)={} max_width={} fill_nl={} roll={}",
            prod.k,
            prod.deg,
            prod.k_hi,
            prod.deg_hi,
            prod.band_size(sandwich_n),
            prod.max_width,
            prod.fill_nl,
            prod.roll
        );
    }
    let gadget = if prod.single_carrier() {
        gadgetize_xgates_with_slice_zero_ccnot_single(
            &sandwich.gates,
            sandwich_n,
            rg_freq,
            slice_gates,
            &prod,
            &mut rng,
        )
    } else {
        gadgetize_xgates_with_slice_zero_ccnot(
            &sandwich.gates,
            sandwich_n,
            rg_freq,
            slice_gates,
            &MaskConfig::off(),
            &prod,
            &mut rng,
        )
    };
    println!(
        "[gen] diversified gadget: {} gates, {} wires",
        gadget.gates.len(),
        gadget.num_wires
    );

    // Sample-verify the gadget's low-2n output equals the sandwich, on the
    // gadget zero slice (upper 2n wires pinned to 0). Requires 4n <= 1024.
    if gadget.num_wires <= 1024 {
        let low = mask_bits(sandwich_n);
        let mut vrng = StdRng::seed_from_u64(seed ^ 0xA11CE);
        let mut bytes = [0u8; 128];
        for i in 0..200 {
            use rand::RngCore;
            vrng.fill_bytes(&mut bytes);
            let input = U1024::from_little_endian(&bytes) & low;
            let got = eval_u1024(&gadget.gates, input) & low;
            let want = eval_u1024(&sandwich.gates, input) & low;
            assert_eq!(got, want, "gadget-low != sandwich on sample {i}");
        }
        println!("[gen] verify PASSED (200 samples, low {sandwich_n} wires)");
    } else {
        println!(
            "[gen] skipped u1024 verify ({} wires > 1024)",
            gadget.num_wires
        );
    }

    write_mpmct(&out, &gadget.gates, gadget.num_wires).expect("write mpmct1");
    println!("[gen] wrote {out}");
}
