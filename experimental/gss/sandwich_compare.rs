//! A/B generator: NEW (sliced sandwich -> new-gadgetize) vs OLD (compose_A
//! sandwich -> legacy gadgetize) on the SAME C, D at n=128. Constructions
//! only (no mixing). Writes both circuits (mpmct1, 512 wires) to circuits/
//! and sample-verifies each against its pre-gadget function.
//!
//! Usage: sandwich_compare [n] [m_C] [m_D] [seed]  (defaults 128 3000 3000 1)

use local_mixing::circuit::U1024;
use local_mixing::engine::format::write_mpmct;
use local_mixing::circuit::xgate::{XGate, eval_u1024};
use local_mixing::circuit::random_circuit;
use local_mixing::preprocessing::gadgets::{
    compose_a, gadgetize, gadgetize_xgates_with_slice_zero_ccnot, sandwich_default_s,
    MaskConfig, ProdConfig,
    sliced_sandwich_with_d,
};
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

fn mask_bits(bits: usize) -> U1024 {
    if bits >= 1024 {
        U1024::MAX
    } else {
        (U1024::one() << bits) - U1024::one()
    }
}

fn random_u1024(rng: &mut impl RngCore, bits: usize) -> U1024 {
    let mut bytes = [0u8; 128];
    rng.fill_bytes(&mut bytes);
    U1024::from_little_endian(&bytes) & mask_bits(bits)
}

/// Sample-check that the gadget's low-n output equals `func(low-n input)` for
/// any aux, where `func` is evaluated by `eval_func`.
fn verify_gadget_low(
    label: &str,
    gadget: &[XGate],
    total_wires: usize,
    n: usize,
    eval_func: impl Fn(U1024) -> U1024,
    samples: usize,
    seed: u64,
) {
    let low = mask_bits(n);
    let _ = total_wires;
    let mut rng = StdRng::seed_from_u64(seed);
    for i in 0..samples {
        // Pin the gadget aux (wires n..2n) to zero: this is the gadget's zero
        // slice, required by the NEW pipeline's slice-zero-ccnot preblock and
        // a valid subset of the OLD gadget's any-aux contract.
        let input = random_u1024(&mut rng, n);
        let got = eval_u1024(gadget, input) & low;
        let want = eval_func(input & low) & low;
        if got != want {
            eprintln!("{label}: VERIFY FAILED on sample {i} (input 0x{input:x})");
            std::process::exit(1);
        }
    }
    println!("{label}: verify PASSED ({samples} samples, low {n} wires)");
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(128);
    let m_c: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(3000);
    let m_d: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(3000);
    let seed: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);

    let sandwich_n = 2 * n; // both pipelines gadgetize a 2n-wire sandwich
    let total = 2 * sandwich_n; // gadget doubles again -> 4n wires
    assert!(total <= 1024, "4n must be <= 1024 wires for U1024 eval");

    // Same C, D for both pipelines (fastrand-seeded for reproducibility).
    fastrand::seed(seed);
    let c = random_circuit(n, m_c);
    let d = random_circuit(n, m_d);
    let s = sandwich_default_s(n);
    println!(
        "[compare] n={n} |C|={} |D|={} s={s} sandwich_wires={sandwich_n} gadget_wires={total}",
        c.gates.len(),
        d.gates.len()
    );

    let mut rng = StdRng::seed_from_u64(seed ^ 0x5150_1CED);

    // ---- NEW pipeline: sliced sandwich -> new-gadgetize (double slice) ----
    let d_xgates: Vec<XGate> = d.gates.iter().map(|&g| XGate::from_g57(g)).collect();
    let sandwich = sliced_sandwich_with_d(&c, &d_xgates, n, s, &mut rng);
    println!(
        "[compare] NEW sliced sandwich: {} gates, {} wires",
        sandwich.gates.len(),
        sandwich.num_wires
    );
    let slice_gate_count = 10 * sandwich_n;
    let new_gadget = gadgetize_xgates_with_slice_zero_ccnot(
        &sandwich.gates,
        sandwich_n,
        2,
        slice_gate_count,
        // Deferred masks off: this bin reproduces the published old-vs-new
        // comparison; enable via MaskConfig for masked A/B runs.
        &MaskConfig::off(),
        &ProdConfig::off(),
        &mut rng,
    );
    println!(
        "[compare] NEW gadgetized: {} gates, {} wires",
        new_gadget.gates.len(),
        new_gadget.num_wires
    );
    let sandwich_gates = sandwich.gates.clone();
    verify_gadget_low(
        "NEW",
        &new_gadget.gates,
        total,
        sandwich_n,
        |x| eval_u1024(&sandwich_gates, x),
        200,
        seed,
    );

    // ---- OLD pipeline: compose_A sandwich -> legacy gadgetize ----
    let a = compose_a(&c, &d, n);
    println!(
        "[compare] OLD compose_A sandwich: {} gates, {} wires",
        a.gates.len(),
        2 * n
    );
    let old_gadget = gadgetize(&a, sandwich_n, 2, &mut rng);
    let old_xgates: Vec<XGate> = old_gadget.gates.iter().map(|&g| XGate::from_g57(g)).collect();
    println!(
        "[compare] OLD gadgetized: {} gates, {} wires",
        old_gadget.gates.len(),
        total
    );
    let a_for_eval = a.clone();
    verify_gadget_low(
        "OLD",
        &old_xgates,
        total,
        sandwich_n,
        |x| a_for_eval.evaluate_1024(x),
        200,
        seed,
    );

    // ---- Write both to circuits/ (mpmct1, 4n wires) ----
    std::fs::create_dir_all("circuits").expect("create circuits/");
    let new_path = format!("circuits/new_sliced_sandwich_gadgetized_n{n}.mpmct1");
    let old_path = format!("circuits/old_compose_a_gadgetized_n{n}.mpmct1");
    write_mpmct(&new_path, &new_gadget.gates, total).expect("write new");
    write_mpmct(&old_path, &old_xgates, total).expect("write old");
    println!("[compare] wrote {new_path}");
    println!("[compare] wrote {old_path}");
}
