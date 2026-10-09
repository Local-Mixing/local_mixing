//! Stage-2 assembly: independent guard RNG, input-derived band seed,
//! masked computation, independent reseed and output guard.

use super::embedded_masking::{
    preprocess_embedded_masking_with_execution, seed_band_mode, seed_band_rich,
};
use super::nonlinear291::preprocess_nonlinear291;
use super::slice_guards::{
    slice_zero_feistel_guard_dims, slice_zero_feistel_guard_dims_high, slice_zero_junk_guard_dims,
    slice_zero_junk_guard_dims_high,
};
use super::types::*;
use crate::circuit::Circuit;
use crate::stages::sandwich::SandwichVariant;

pub fn preprocess_sandwich(
    sandwich: &Circuit,
    source_wires: usize,
    slice_gate_count: usize,
    variant: SandwichVariant,
    options: &PreprocessingParams,
    rng: &mut impl rand::Rng,
) -> Result<PreprocessingOutput, String> {
    let n = source_wires;
    match options {
        PreprocessingParams::EmbeddedMasking {
            params,
            execution,
            balanced_seed,
            seed_fill_nl,
        } => {
            let gadget_seed = params.seed;
            let masked = preprocess_embedded_masking_with_execution(
                &sandwich.gates,
                sandwich.num_wires,
                params,
                execution,
            );
            // Both shuffling modes preserve the sandwich output contract:
            // per-box return home trivially, persistent migration via the
            // stream-tail port restoration. Assert the contract itself.
            assert!(
                masked.final_layout[..sandwich.num_wires]
                    .iter()
                    .enumerate()
                    .all(|(role, &wire)| role == wire as usize),
                "data roles must be physically home after preprocessing"
            );
            // Full 5-step delivery: wrap the compute (parts 2-4) with the junk-half
            // zero-slice guard (parts 1 & 5), unchanged from the drip delivery. The
            // guard targets the sandwich's forward-junk half (low n) and is keyed on
            // the band (n..2n): dead at the input port (band 0 before the compute
            // seeds it), fires at the output port (band junked) but misses the
            // payload on the upper half -> the composite is reverse-honest.
            let np = sandwich.num_wires; // = 2n
            let nondata = masked.r_used; // band width
            let gc = slice_gate_count.max(nondata);
            // Per-port guard selection. The OPENING guard always junks the LOW half
            // (dead forward on the honest slice; its reverse junks the low half). The
            // CLOSING guard junks the sandwich's forward-junk half at the output port:
            // the LOW half for classic (payload on the upper), the HIGH half for
            // balanced (payload on the low). This is not masking-specific — it's the
            // gadgetize guard adapting to which half the sandwich designates as junk.
            // Punchlist #7 (2026-10-06): the accumulate-and-inject guard
            // (width <= 2 everywhere) is the default; SLICE_GUARD_LEGACY=1
            // restores the width-3 quad generator for comparison arms.
            let legacy_guard = params.legacy_slice_guard;
            let open = if legacy_guard {
                slice_zero_junk_guard_dims(np, nondata, gc, rng)
            } else {
                slice_zero_feistel_guard_dims(np, nondata, gc, rng)
            };
            let close = match (variant.is_balanced(), legacy_guard) {
                (true, true) => slice_zero_junk_guard_dims_high(np, nondata, gc, rng),
                (true, false) => slice_zero_feistel_guard_dims_high(np, nondata, gc, rng),
                (false, true) => slice_zero_junk_guard_dims(np, nondata, gc, rng),
                (false, false) => slice_zero_feistel_guard_dims(np, nondata, gc, rng),
            };
            // Module 2: band-seeding pipelined between the input slice guard (dead
            // on the zero band) and the compute (which only reads the band).
            // EMBEDDED_MASKING_BAL_SEED=0 keeps the AND-of-literals seed with balanced masks (diagnostic)
            let bal_seed = *balanced_seed;
            // seed_fill_nl > 0 selects the revived rich fill (punchlist #5:
            // pivot + linear sources + nonlinear products, slice-guard-sized);
            // 0 keeps the legacy thin 2-CNOT seed honoring `balanced_seed`.
            // Fringe fix 2: keep the forward seed's raw-pair products off the
            // control pairs of the sandwich's first-layer gates (target and
            // both controls still untouched inputs).
            let avoid_c1 = params.seed_avoid_c1;
            let mut first_layer_pairs: std::collections::HashSet<(u16, u16)> =
                std::collections::HashSet::new();
            if avoid_c1 {
                let mut written = vec![false; sandwich.num_wires];
                for gate in &sandwich.gates {
                    // Slice-block gates read the zero half with positive
                    // polarity: identity on the honest slice, not a write.
                    let dead =
                        !gate.comp && gate.ctrls.iter().any(|&(c, pos)| pos && c as usize >= n);
                    if dead {
                        continue;
                    }
                    let raw = !written[gate.target as usize]
                        && gate.ctrls.iter().all(|&(c, _)| !written[c as usize]);
                    if raw && gate.ctrls.len() == 2 {
                        let (a, b) = (gate.ctrls[0].0, gate.ctrls[1].0);
                        first_layer_pairs.insert((a.min(b), a.max(b)));
                    }
                    written[gate.target as usize] = true;
                }
            }
            let band_seed = if *seed_fill_nl > 0 {
                seed_band_rich(
                    np,
                    masked.r_used,
                    n,
                    gadget_seed ^ 0x5EED_B00C,
                    *seed_fill_nl,
                    &first_layer_pairs,
                )
            } else {
                seed_band_mode(np, masked.r_used, n, gadget_seed ^ 0x5EED_B00C, bal_seed)
            };
            // Module 4: the band RE-SEED after the compute. The five parts are always
            // five separate modules: the compute's own rerand bursts are masking
            // hygiene and do NOT discharge stage 4 (see the design doc's "Where it
            // fits in the pipeline"). Not an inverse of module 2 — the band is junk
            // at BOTH ports — so it draws a different seed.
            let band_reseed = if *seed_fill_nl > 0 {
                seed_band_rich(
                    np,
                    masked.r_used,
                    n,
                    gadget_seed ^ 0xB00C_5EED,
                    *seed_fill_nl,
                    &std::collections::HashSet::new(),
                )
            } else {
                seed_band_mode(np, masked.r_used, n, gadget_seed ^ 0xB00C_5EED, bal_seed)
            };
            let report = EmbeddedMaskingReport {
                band_wires: masked.r_used,
                atoms: masked.atoms,
                band_seed_gates: band_seed.len(),
                band_reseed_gates: band_reseed.len(),
                guard_gates: open.gates.len(),
                compute_gates: masked.gates.len(),
                original_compute_gates: masked.shuffling_stats.original_gates,
                hot_intervals: masked.role_hot_intervals,
                rerand_done: masked.rerand_done,
                transfer_count: masked.shuffling_stats.transfer_count,
                shuffling_added_gates: masked.shuffling_stats.added_gates,
                restored_ports: masked.shuffling_stats.restored_ports,
                skipped_cuts: masked.shuffling_stats.skipped_cut_count,
                jit_remasks: masked.shuffling_stats.jit_remasks,
            };
            let mut gates = open.gates;
            gates.extend(band_seed);
            gates.extend(masked.gates);
            gates.extend(band_reseed);
            gates.extend(close.gates);
            Ok(PreprocessingOutput {
                circuit: Circuit {
                    gates,
                    num_wires: masked.num_wires.max(np + nondata),
                },
                guarded: true,
                embedded_masking_report: Some(report),
            })
        }
        PreprocessingParams::Nonlinear291 => Ok(PreprocessingOutput {
            circuit: preprocess_nonlinear291(
                &sandwich.gates,
                sandwich.num_wires,
                n,
                slice_gate_count,
                rng,
            )?,
            guarded: false,
            embedded_masking_report: None,
        }),
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/preprocessing/construct.rs"]
mod tests;
