//! Stage-specific forward and reverse payload checks on the documented zero slice.

use crate::circuit::Circuit;
use crate::stages::sandwich::SandwichVariant;
use rand::{SeedableRng, rngs::StdRng};

/// Verify 256 bit-sliced samples using an independent random stream. Auxiliary
/// outputs are intentionally unconstrained; the classic guarded construction
/// also promises the inverse payload on its low-input zero slice.
pub fn verify_payload(
    sandwich: &Circuit,
    gadget: &Circuit,
    n: usize,
    guarded: bool,
    variant: SandwichVariant,
    seed: u64,
) -> (usize, usize, bool) {
    verify_payload_ports(sandwich, gadget, n, guarded, variant, seed, true)
}

/// [`verify_payload`] for an assembly built with or without the reverse
/// contract (`preprocess_sandwich_ports`): without it the reverse check is
/// skipped (the reversed artifact is junk by design) and the third value
/// is false.
pub fn verify_payload_ports(
    sandwich: &Circuit,
    gadget: &Circuit,
    n: usize,
    guarded: bool,
    variant: SandwichVariant,
    seed: u64,
    reverse_contract: bool,
) -> (usize, usize, bool) {
    verify_payload_inputs(
        sandwich,
        gadget,
        n,
        guarded,
        variant,
        seed,
        reverse_contract,
        sandwich.num_wires,
    )
}

/// [`verify_payload_ports`] with the forward samples random on the data
/// wires `0..data_inputs` only, zeros elsewhere: all sandwich wires for the
/// band-keyed guards (they leave the data alone whenever the band is zero),
/// the input register `0..n` (the honest slice) for the two-layer opening
/// guard, which scrambles A whenever B is nonzero.
#[allow(clippy::too_many_arguments)]
pub fn verify_payload_inputs(
    sandwich: &Circuit,
    gadget: &Circuit,
    n: usize,
    guarded: bool,
    variant: SandwichVariant,
    seed: u64,
    reverse_contract: bool,
    data_inputs: usize,
) -> (usize, usize, bool) {
    let sandwich_n = sandwich.num_wires;
    // state[w] = lane word for wire w; bit L = wire w's value in sample L.
    fn eval_lanes(gates: &[crate::circuit::xgate::XGate], state: &mut [u64]) {
        for g in gates {
            let mut acc = !0u64;
            for &(w, pol) in &g.ctrls {
                let v = state[w as usize];
                acc &= if pol { v } else { !v };
            }
            state[g.target as usize] ^= if g.comp { !acc } else { acc };
        }
    }
    let mut vrng = StdRng::seed_from_u64(seed ^ 0xA11CE);
    let total_wires = gadget.num_wires.max(sandwich.num_wires);
    // With a closing zero-slice guard, the composite preserves only the
    // UPPER half of the sandwich state on the honest slice: the closing
    // guard fires against the junked band and perturbs the low (forward-
    // junk) half by design. The payload contract is unchanged — C(x) lives
    // on the upper half (see verify_zero_slice). Embedded masking is now wrapped
    // by the same junk guards, so it too verifies the upper half and runs
    // the reverse-honesty check.
    // Which data half carries the forward payload after the closing guard.
    // Classic junks the LOW half so the payload survives on the UPPER half;
    // the balanced close guard is mirrored (junks HIGH), so its payload
    // survives on the LOW half. Unguarded gadgets preserve all 2n wires.
    let (verify_from, verify_to) = if !guarded {
        (0, sandwich_n)
    } else if variant.is_balanced() {
        (0, sandwich_n / 2)
    } else {
        (sandwich_n / 2, sandwich_n)
    };
    for round in 0..4 {
        use rand::RngCore;
        let mut ga = vec![0u64; total_wires];
        for w in 0..data_inputs.min(sandwich_n) {
            ga[w] = vrng.next_u64(); // upper wires stay pinned to 0
        }
        let mut sa = ga.clone();
        eval_lanes(&gadget.gates, &mut ga);
        eval_lanes(&sandwich.gates, &mut sa);
        for w in verify_from..verify_to {
            assert_eq!(
                ga[w], sa[w],
                "gadget != sandwich on payload wire {w}, round {round}"
            );
        }
    }
    // Reverse-honesty verify (symmetric ports): the REVERSED gadget on
    // (a on the low half, zeros elsewhere) must reproduce the REVERSED
    // sandwich's upper half — the gadget-level mirror of the sandwich's
    // A^-1(a,0) = (junk, D^-1(a)) contract. Every XGate is an involution,
    // so the reversed gate list is the inverse circuit.
    let reverse_checked = guarded && !variant.is_balanced() && reverse_contract;
    if reverse_checked {
        let rev_gadget: Vec<crate::circuit::xgate::XGate> =
            gadget.gates.iter().rev().cloned().collect();
        let rev_sandwich: Vec<crate::circuit::xgate::XGate> =
            sandwich.gates.iter().rev().cloned().collect();
        for round in 0..4 {
            use rand::RngCore;
            let mut ga = vec![0u64; total_wires];
            for w in 0..n {
                ga[w] = vrng.next_u64(); // a on the low half, zeros elsewhere
            }
            let mut sa = ga.clone();
            eval_lanes(&rev_gadget, &mut ga);
            eval_lanes(&rev_sandwich, &mut sa);
            for w in n..sandwich_n {
                assert_eq!(
                    ga[w], sa[w],
                    "reverse gadget != reverse sandwich on wire {w}, round {round}"
                );
            }
        }
    }
    (verify_from, verify_to, reverse_checked)
}
