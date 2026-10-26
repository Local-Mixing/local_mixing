//! Resolved stage-2 inputs and the public slice contract.

use super::embedded_masking::{EmbeddedMaskingExecution, EmbeddedMaskingParams};
use super::three_mask::{ThreeMaskParams, ThreeMaskStats};
use crate::circuit::Circuit;

/// The supported choices for new TDP preprocessing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PreprocessingMode {
    #[default]
    EmbeddedMasking,
    /// The embedded-masking assembly (guards, band seed and reseed) around
    /// the three-mask compute (`three_mask`).
    ThreeMask,
    Nonlinear291,
}

impl PreprocessingMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "embedded-masking" => Some(Self::EmbeddedMasking),
            "three-mask" => Some(Self::ThreeMask),
            "nonlinear291" => Some(Self::Nonlinear291),
            _ => None,
        }
    }

    pub fn canonical_name(self) -> &'static str {
        match self {
            Self::EmbeddedMasking => "embedded-masking",
            Self::ThreeMask => "three-mask",
            Self::Nonlinear291 => "nonlinear291",
        }
    }
}

/// Fully resolved options; construction never reads process configuration.
pub enum PreprocessingParams {
    EmbeddedMasking {
        params: EmbeddedMaskingParams,
        execution: EmbeddedMaskingExecution,
        balanced_seed: bool,
        /// Nonlinear product terms per band wire in the band seed/reseed
        /// (punchlist #5). `0` keeps the legacy 2-CNOT thin seed; the
        /// managed default (4) makes each seed block slice-guard-sized,
        /// reviving the pre-reorganization `emit_band_fill_nl_pivots_src`
        /// generator (pivot + 1..=7 linear sources + nonlinear products).
        seed_fill_nl: usize,
    },
    /// The five-part assembly with the three-mask compute in the middle.
    /// `outer` holds the parameters of the outer blocks (guards, band seed
    /// and reseed), which are exactly those of the embedded-masking branch
    /// for the same `outer`, `balanced_seed`, `seed_fill_nl` and guard RNG;
    /// only `outer.seed`, `outer.r`, `outer.seed_avoid_c1` and
    /// `outer.legacy_slice_guard` reach them. `params.band_wires` (0 = auto)
    /// is the compute band; `params.seeded_band_wires` is overridden from it.
    ThreeMask {
        params: ThreeMaskParams,
        outer: EmbeddedMaskingParams,
        balanced_seed: bool,
        seed_fill_nl: usize,
    },
    Nonlinear291,
}

impl PreprocessingParams {
    pub fn managed_tdp(seed: u64, source_wires: usize) -> Self {
        Self::EmbeddedMasking {
            params: EmbeddedMaskingParams::managed_tdp(seed, source_wires),
            execution: EmbeddedMaskingExecution::default(),
            balanced_seed: true,
            seed_fill_nl: 4,
        }
    }
}

/// Measurements used by the executable's progress and optional trace reports.
pub struct EmbeddedMaskingReport {
    pub band_wires: usize,
    pub atoms: usize,
    pub band_seed_gates: usize,
    pub band_reseed_gates: usize,
    pub guard_gates: usize,
    pub compute_gates: usize,
    /// Compute length before optional role transfers, for coverage coordinates.
    pub original_compute_gates: usize,
    /// (Logical data role, original gate start, original gate end), before
    /// optional physical-role shuffling and before the surrounding modules.
    pub hot_intervals: Vec<(u16, usize, usize)>,
    /// Rerand bursts executed inside the compute (masking hygiene).
    pub rerand_done: usize,
    /// Carrier transfers emitted by preprocessing shuffling (2 CNOTs each).
    pub transfer_count: usize,
    /// Total gates shuffling added to the compute (transfers + per-box
    /// pending-mask drains + the persistent-mode port-restoration tail).
    pub shuffling_added_gates: usize,
    /// Data roles swapped home by the persistent-mode tail (3 CNOTs each).
    pub restored_ports: usize,
    /// Transfer cuts skipped for lack of an eligible partner.
    pub skipped_cuts: usize,
    /// Fresh masks applied by just-in-time drain management (punchlist #8).
    pub jit_remasks: usize,
}

/// Measurements of the three-mask assembly: the outer blocks and the
/// compute's own statistics (`ThreeMaskStats`).
#[derive(Clone, Debug)]
pub struct ThreeMaskReport {
    /// Resolved compute band width (the circuit has `2n + band_wires` wires).
    pub band_wires: usize,
    /// Band prefix seeded by the outer blocks when `band_wires` exceeds it
    /// (the compute's extra-wire ingress reads it); 0 when the whole band is
    /// seeded.
    pub seeded_band_wires: usize,
    pub guard_gates: usize,
    pub band_seed_gates: usize,
    pub band_reseed_gates: usize,
    /// Compute length after carrier migration.
    pub compute_gates: usize,
    pub before_migration: usize,
    pub fire_core_gates: usize,
    pub mask_gates: usize,
    pub growth_gates: usize,
    pub migration_gates: usize,
    pub transfers: usize,
    pub skipped_cuts: usize,
    pub rotations: usize,
    pub retired_banks: usize,
    pub retiring_replacements: usize,
    pub band_ingress_gates: usize,
    pub promotion_stalls: usize,
    pub local_mask_fallbacks: usize,
    pub mask_rank_rejections: usize,
    pub fire: super::three_mask::fire::FireStats,
    /// The complete compute statistics the summary fields were taken from.
    pub stats: ThreeMaskStats,
}

/// The low sandwich-width wires retain the logical layout. Auxiliary wires
/// start at zero. With embedded masking's closing guard, only the sandwich's
/// payload half is promised; nonlinear291 preserves every logical output wire.
pub struct PreprocessingOutput {
    pub circuit: Circuit,
    /// A closing junk guard narrows the output guarantee to the payload half.
    /// Nonlinear291's ingress-only guard does not set this flag.
    pub guarded: bool,
    pub embedded_masking_report: Option<EmbeddedMaskingReport>,
    pub three_mask_report: Option<ThreeMaskReport>,
}
