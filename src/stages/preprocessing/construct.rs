//! Stage-2 assembly: independent guard RNG, input-derived band seed,
//! masked computation, independent reseed and output guard.

use super::nonlinear291::preprocess_nonlinear291;
use super::quadratic_masking::{
    hot_intervals, preprocess_quadratic_masking_with_execution, seed_band_mode,
};
use super::slice_guards::{slice_zero_junk_guard_dims, slice_zero_junk_guard_dims_high};
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
        PreprocessingParams::QuadraticMasking {
            params,
            execution,
            balanced_seed,
            record_hot_intervals,
        } => {
            let gadget_seed = params.seed;
            let bv5 = preprocess_quadratic_masking_with_execution(
                &sandwich.gates,
                sandwich.num_wires,
                params,
                execution,
            );
            // Full 5-step delivery: wrap the compute (parts 2-4) with the junk-half
            // zero-slice guard (parts 1 & 5), unchanged from the drip delivery. The
            // guard targets the sandwich's forward-junk half (low n) and is keyed on
            // the band (n..2n): dead at the input port (band 0 before the compute
            // seeds it), fires at the output port (band junked) but misses the
            // payload on the upper half -> the composite is reverse-honest.
            let np = sandwich.num_wires; // = 2n
            let nondata = bv5.r_used; // band width
            let gc = slice_gate_count.max(nondata);
            // Per-port guard selection. The OPENING guard always junks the LOW half
            // (dead forward on the honest slice; its reverse junks the low half). The
            // CLOSING guard junks the sandwich's forward-junk half at the output port:
            // the LOW half for classic (payload on the upper), the HIGH half for
            // balanced (payload on the low). This is not V5-specific — it's the
            // gadgetize guard adapting to which half the sandwich designates as junk.
            let open = slice_zero_junk_guard_dims(np, nondata, gc, rng);
            let close = if variant.is_balanced() {
                slice_zero_junk_guard_dims_high(np, nondata, gc, rng)
            } else {
                slice_zero_junk_guard_dims(np, nondata, gc, rng)
            };
            // Module 2: band-seeding pipelined between the input slice guard (dead
            // on the zero band) and the compute (which only reads the band).
            // BV5_BAL_SEED=0 keeps the AND-of-literals seed with balanced masks (diagnostic)
            let bal_seed = *balanced_seed;
            let band_seed = seed_band_mode(np, bv5.r_used, n, gadget_seed ^ 0x5EED_B00C, bal_seed);
            // Module 4: the band RE-SEED after the compute. The five parts are always
            // five separate modules: the compute's own rerand bursts are masking
            // hygiene and do NOT discharge stage 4 (see the design doc's "Where it
            // fits in the pipeline"). Not an inverse of module 2 — the band is junk
            // at BOTH ports — so it draws a different seed.
            let band_reseed =
                seed_band_mode(np, bv5.r_used, n, gadget_seed ^ 0xB00C_5EED, bal_seed);
            let report = QuadraticMaskingReport {
                band_wires: bv5.r_used,
                atoms: bv5.atoms,
                band_seed_gates: band_seed.len(),
                band_reseed_gates: band_reseed.len(),
                guard_gates: open.gates.len(),
                compute_gates: bv5.gates.len(),
                hot_intervals: if *record_hot_intervals {
                    hot_intervals(&bv5.gates, np, bv5.r_used)
                } else {
                    Vec::new()
                },
            };
            let mut gates = open.gates;
            gates.extend(band_seed);
            gates.extend(bv5.gates);
            gates.extend(band_reseed);
            gates.extend(close.gates);
            Ok(PreprocessingOutput {
                circuit: Circuit {
                    gates,
                    num_wires: bv5.num_wires.max(np + nondata),
                },
                guarded: true,
                quadratic_masking_report: Some(report),
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
            quadratic_masking_report: None,
        }),
    }
}

#[cfg(test)]
#[path = "../../../tests/stages/preprocessing/managed_construction.rs"]
mod tests;
