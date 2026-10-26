//! Resolved stage-2 inputs and the public slice contract.

use super::embedded_masking::{EmbeddedMaskingExecution, EmbeddedMaskingParams};
use crate::circuit::Circuit;

/// The two supported choices for new TDP preprocessing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PreprocessingMode {
    #[default]
    EmbeddedMasking,
    /// The three-mask compute (x_i = d_i + three cubic band monomials)
    /// inside the embedded-masking outer blocks.
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

/// The low sandwich-width wires retain the logical layout. Auxiliary wires
/// start at zero. With embedded masking's closing guard, only the sandwich's
/// payload half is promised; nonlinear291 preserves every logical output wire.
pub struct PreprocessingOutput {
    pub circuit: Circuit,
    /// A closing junk guard narrows the output guarantee to the payload half.
    /// Nonlinear291's ingress-only guard does not set this flag.
    pub guarded: bool,
    pub embedded_masking_report: Option<EmbeddedMaskingReport>,
}
