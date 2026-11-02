//! Stage-2 assembly: independent guard RNG, input-derived band seed,
//! masked computation, independent reseed and output guard.

use super::embedded_masking::{
    EmbeddedMaskingParams, band_width, preprocess_embedded_masking_with_execution, seed_band_g57,
    seed_band_g57_gates, seed_band_mode, seed_band_rich,
};
use super::nonlinear291::preprocess_nonlinear291;
use super::slice_guards::{
    slice_zero_feistel_guard_dims, slice_zero_feistel_guard_dims_high, slice_zero_junk_guard_dims,
    slice_zero_junk_guard_dims_high, two_layer_opening_guard_dims,
};
use super::three_mask::{self, ThreeMaskParams};
use super::types::*;
use crate::circuit::{Circuit, XGate};
use crate::stages::sandwich::SandwichVariant;

/// The four outer blocks of the five-part assembly, around the compute.
struct OuterBlocks {
    open: Vec<XGate>,
    band_seed: Vec<XGate>,
    band_reseed: Vec<XGate>,
    close: Vec<XGate>,
}

/// Modules 1, 2, 4 and 5 of the stage: the opening and closing zero-slice
/// guards (drawn from `rng` in that order), the input-derived band seed and
/// the independently seeded band reseed, for a band of `nondata` wires
/// (`np..np + nondata`). Shared by every mode that wraps a compute; the
/// compute itself never touches `rng`, so the blocks do not depend on it.
///
/// `reverse_contract` false (stage-2 `reverse_contract`, RC 2026-11-01):
/// there is no closing guard (`close` is empty), so only the forward
/// payload is kept, and the opening guard is the two-layer guard keyed on
/// the register B and the band (RC 2026-11-02, recipe v15; recipe v14's
/// guard was keyed on the band only, so with the band zero it let every
/// value of B through).
#[allow(clippy::too_many_arguments)]
fn outer_blocks(
    sandwich: &Circuit,
    n: usize,
    nondata: usize,
    slice_gate_count: usize,
    variant: SandwichVariant,
    params: &EmbeddedMaskingParams,
    balanced_seed: bool,
    seed_fill_nl: usize,
    reverse_contract: bool,
    rng: &mut impl rand::Rng,
) -> OuterBlocks {
    let gadget_seed = params.seed;
    // Full 5-step delivery: wrap the compute (parts 2-4) with the junk-half
    // zero-slice guard (parts 1 & 5), unchanged from the drip delivery. The
    // guard targets the sandwich's forward-junk half (low n) and is keyed on
    // the band (n..2n): dead at the input port (band 0 before the compute
    // seeds it), fires at the output port (band junked) but misses the
    // payload on the upper half -> the composite is reverse-honest.
    let np = sandwich.num_wires; // = 2n
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
    // The two-layer guard has its own size (n log n three-control gates per
    // layer) and is width <= 2 by construction; SLICE_GUARD_LEGACY applies
    // to the junk-half guards of the reverse contract only.
    let open = match (reverse_contract, legacy_guard) {
        (true, true) => slice_zero_junk_guard_dims(np, nondata, gc, rng),
        (true, false) => slice_zero_feistel_guard_dims(np, nondata, gc, rng),
        (false, _) => two_layer_opening_guard_dims(n, nondata, rng),
    };
    let close = match (reverse_contract, variant.is_balanced(), legacy_guard) {
        (false, _, _) => Circuit {
            gates: Vec::new(),
            num_wires: np + nondata,
        },
        (true, true, true) => slice_zero_junk_guard_dims_high(np, nondata, gc, rng),
        (true, true, false) => slice_zero_feistel_guard_dims_high(np, nondata, gc, rng),
        (true, false, true) => slice_zero_junk_guard_dims(np, nondata, gc, rng),
        (true, false, false) => slice_zero_feistel_guard_dims(np, nondata, gc, rng),
    };
    // Module 2: band-seeding pipelined between the input slice guard (dead
    // on the zero band) and the compute (which only reads the band).
    // EMBEDDED_MASKING_BAL_SEED=0 keeps the AND-of-literals seed with balanced masks (diagnostic)
    // seed_fill_nl > 0 selects the revived rich fill (punchlist #5:
    // pivot + linear sources + nonlinear products, slice-guard-sized);
    // 0 keeps the legacy thin 2-CNOT seed honoring `balanced_seed`.
    // Fringe fix 2: keep the forward seed's raw-pair products off the
    // control pairs of the sandwich's first-layer gates (target and
    // both controls still untouched inputs).
    let mut first_layer_pairs: std::collections::HashSet<(u16, u16)> =
        std::collections::HashSet::new();
    if params.seed_avoid_c1 {
        let mut written = vec![false; sandwich.num_wires];
        for gate in &sandwich.gates {
            // Slice-block gates read the zero half with positive
            // polarity: identity on the honest slice, not a write.
            let dead = !gate.comp && gate.ctrls.iter().any(|&(c, pos)| pos && c as usize >= n);
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
    // Without the reverse contract (recipe v15, RC 2026-11-02): 8 n log n
    // random g57 gates writing the band, redrawn until every band wire comes
    // out non-constant on the honest slice behind the opening guard.
    let band_seed = if !reverse_contract {
        let mut seed = gadget_seed ^ 0x5EED_B00C;
        loop {
            let gates = seed_band_g57(
                np,
                nondata,
                seed_band_g57_gates(n),
                seed,
                &first_layer_pairs,
            );
            if band_nonconstant_on_slice(&open.gates, &gates, n, np + nondata) {
                break gates;
            }
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        }
    } else if seed_fill_nl > 0 {
        seed_band_rich(
            np,
            nondata,
            n,
            gadget_seed ^ 0x5EED_B00C,
            seed_fill_nl,
            &first_layer_pairs,
        )
    } else {
        seed_band_mode(np, nondata, n, gadget_seed ^ 0x5EED_B00C, balanced_seed)
    };
    // Module 4: the band RE-SEED after the compute. The five parts are always
    // five separate modules: the compute's own rerand bursts are masking
    // hygiene and do NOT discharge stage 4 (see the design doc's "Where it
    // fits in the pipeline"). Not an inverse of module 2 — the band is junk
    // at BOTH ports — so it draws a different seed.
    let band_reseed = if !reverse_contract {
        seed_band_g57(
            np,
            nondata,
            seed_band_g57_gates(n),
            gadget_seed ^ 0xB00C_5EED,
            &std::collections::HashSet::new(),
        )
    } else if seed_fill_nl > 0 {
        seed_band_rich(
            np,
            nondata,
            n,
            gadget_seed ^ 0xB00C_5EED,
            seed_fill_nl,
            &std::collections::HashSet::new(),
        )
    } else {
        seed_band_mode(np, nondata, n, gadget_seed ^ 0xB00C_5EED, balanced_seed)
    };
    OuterBlocks {
        open: open.gates,
        band_seed,
        band_reseed,
        close: close.gates,
    }
}

/// Whether every band wire (`2n..total`) takes both values over 256 random
/// honest-slice inputs (x on `0..n`, zeros elsewhere) run through `prefix`
/// and then `seed`.
fn band_nonconstant_on_slice(prefix: &[XGate], seed: &[XGate], n: usize, total: usize) -> bool {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xBA5D_5EED);
    let mut ones = vec![false; total];
    let mut zeros = vec![false; total];
    for _ in 0..4 {
        let mut state = vec![0u64; total];
        for lane in state.iter_mut().take(n) {
            *lane = rng.random();
        }
        for g in prefix.iter().chain(seed) {
            g.apply_lanes(&mut state);
        }
        for w in 2 * n..total {
            ones[w] |= state[w] != 0;
            zeros[w] |= state[w] != !0u64;
        }
    }
    (2 * n..total).all(|w| ones[w] && zeros[w])
}

pub fn preprocess_sandwich(
    sandwich: &Circuit,
    source_wires: usize,
    slice_gate_count: usize,
    variant: SandwichVariant,
    options: &PreprocessingParams,
    rng: &mut impl rand::Rng,
) -> Result<PreprocessingOutput, String> {
    preprocess_sandwich_ports(
        sandwich,
        source_wires,
        slice_gate_count,
        variant,
        options,
        true,
        rng,
    )
}

/// [`preprocess_sandwich`] with the port contract explicit:
/// `reverse_contract` true keeps both junk-half guards (the reversed
/// artifact reproduces the reversed sandwich's payload half); false opens
/// with the two-layer guard and drops the closing guard (forward payload
/// only). Nonlinear291 has no guards and ignores it.
pub fn preprocess_sandwich_ports(
    sandwich: &Circuit,
    source_wires: usize,
    slice_gate_count: usize,
    variant: SandwichVariant,
    options: &PreprocessingParams,
    reverse_contract: bool,
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
            let np = sandwich.num_wires; // = 2n
            let nondata = masked.r_used; // band width
            let blocks = outer_blocks(
                sandwich,
                n,
                nondata,
                slice_gate_count,
                variant,
                params,
                *balanced_seed,
                *seed_fill_nl,
                reverse_contract,
                rng,
            );
            let report = EmbeddedMaskingReport {
                band_wires: masked.r_used,
                atoms: masked.atoms,
                band_seed_gates: blocks.band_seed.len(),
                band_reseed_gates: blocks.band_reseed.len(),
                guard_gates: blocks.open.len(),
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
            let mut gates = blocks.open;
            gates.extend(blocks.band_seed);
            gates.extend(masked.gates);
            gates.extend(blocks.band_reseed);
            gates.extend(blocks.close);
            Ok(PreprocessingOutput {
                circuit: Circuit {
                    gates,
                    num_wires: masked.num_wires.max(np + nondata),
                },
                guarded: true,
                embedded_masking_report: Some(report),
                three_mask_report: None,
            })
        }
        PreprocessingParams::ThreeMask {
            params,
            outer,
            balanced_seed,
            seed_fill_nl,
        } => {
            let np = sandwich.num_wires; // = 2n
            // The outer blocks are sized by the embedded-masking band they
            // would wrap (`outer.r`, auto = np), so they are byte-identical
            // to the embedded-masking branch's for the same `outer`.
            let seeded = band_width(np, outer);
            let band_wires = three_mask::resolve_band_wires(np, params.band_wires);
            if band_wires < seeded {
                return Err(format!(
                    "three-mask band ({band_wires} wires) must cover the band the outer blocks seed ({seeded} wires)"
                ));
            }
            let blocks = outer_blocks(
                sandwich,
                n,
                seeded,
                slice_gate_count,
                variant,
                outer,
                *balanced_seed,
                *seed_fill_nl,
                reverse_contract,
                rng,
            );
            // Extra band wires beyond the seeded prefix get the compute's
            // reversible XOR ingress from that prefix (their honest input is
            // zero); a band of exactly the seeded width needs none.
            let cfg = ThreeMaskParams {
                seeded_band_wires: if band_wires > seeded { seeded } else { 0 },
                // the wire-turnover mode forms its cohorts itself (banks stay 0)
                banks: if params.wire_turnover {
                    params.banks
                } else {
                    three_mask::resolve_banks(band_wires, params.banks)
                },
                ..params.clone()
            };
            let compute = three_mask::compute(&sandwich.gates, np, band_wires, &cfg)?;
            let s = compute.stats;
            let report = ThreeMaskReport {
                band_wires,
                seeded_band_wires: cfg.seeded_band_wires,
                guard_gates: blocks.open.len(),
                band_seed_gates: blocks.band_seed.len(),
                band_reseed_gates: blocks.band_reseed.len(),
                compute_gates: s.compute_gates,
                before_migration: s.before_migration,
                fire_core_gates: s.fire_core_gates,
                mask_gates: s.mask_gates,
                growth_gates: s.growth_gates,
                migration_gates: s.migration_gates,
                transfers: s.transfers,
                skipped_cuts: s.skipped_cuts,
                rotations: s.rotations.len(),
                retired_banks: s.retired_banks,
                retiring_replacements: s.retiring_replacements,
                band_ingress_gates: s.band_ingress_gates,
                promotion_stalls: s.promotion_stalls,
                local_mask_fallbacks: s.local_mask_fallbacks,
                mask_rank_rejections: s.mask_rank_rejections,
                wire_turnover: cfg.wire_turnover,
                wire_limbo_mean: if s.wire_limbo_waits.is_empty() {
                    0.0
                } else {
                    s.wire_limbo_waits.iter().sum::<usize>() as f64
                        / s.wire_limbo_waits.len() as f64
                },
                wire_limbo_max: s.wire_limbo_waits.iter().copied().max().unwrap_or(0),
                interleave_window: s.interleave_window,
                interleave_mean_displacement: s.interleave_mean_displacement,
                interleave_span_250: s.interleave_span_250,
                interleave_depth: s.interleave_depth,
                fire_len_mean: s.fire_len_mean,
                fire_span_measured: s.fire_span_measured,
                fire_span_mean: s.fire_span_mean,
                fire_span_p50: s.fire_span_p50,
                fire_span_p90: s.fire_span_p90,
                fires_per_250: s.fires_per_250,
                fire: s.fire.clone(),
                stats: s,
            };
            let mut gates = blocks.open;
            gates.extend(blocks.band_seed);
            gates.extend(compute.gates);
            gates.extend(blocks.band_reseed);
            gates.extend(blocks.close);
            Ok(PreprocessingOutput {
                circuit: Circuit {
                    gates,
                    num_wires: np + band_wires,
                },
                guarded: true,
                embedded_masking_report: None,
                three_mask_report: Some(report),
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
            three_mask_report: None,
        }),
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/preprocessing/construct.rs"]
mod tests;
