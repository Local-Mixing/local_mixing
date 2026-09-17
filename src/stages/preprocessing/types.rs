//! Resolved stage-2 inputs and the public slice contract.

use super::embedded_masking::{EmbeddedMaskingExecution, EmbeddedMaskingParams};
use crate::circuit::Circuit;

/// The two supported choices for new TDP preprocessing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PreprocessingMode {
    #[default]
    EmbeddedMasking,
    Nonlinear291,
}

impl PreprocessingMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "embedded-masking" => Some(Self::EmbeddedMasking),
            "nonlinear291" => Some(Self::Nonlinear291),
            _ => None,
        }
    }

    pub fn canonical_name(self) -> &'static str {
        match self {
            Self::EmbeddedMasking => "embedded-masking",
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
    },
    Nonlinear291,
}

impl PreprocessingParams {
    pub fn managed_tdp(seed: u64, source_wires: usize) -> Self {
        Self::EmbeddedMasking {
            params: EmbeddedMaskingParams::managed_tdp(seed, source_wires),
            execution: EmbeddedMaskingExecution::default(),
            balanced_seed: true,
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
