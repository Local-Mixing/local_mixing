//! Resolved stage-2 inputs and the public slice contract.

use super::quadratic_masking::{QuadraticMaskingExecution, QuadraticMaskingParams};
use crate::circuit::Circuit;

/// The two supported choices for new GSS preprocessing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PreprocessingMode {
    #[default]
    QuadraticMasking,
    Nonlinear291,
}

impl PreprocessingMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "quadratic-masking" | "ran-balanced" | "blinded-v5" | "blinded_v5" => {
                Some(Self::QuadraticMasking)
            }
            "nonlinear291" => Some(Self::Nonlinear291),
            _ => None,
        }
    }

    pub fn canonical_name(self) -> &'static str {
        match self {
            Self::QuadraticMasking => "quadratic-masking",
            Self::Nonlinear291 => "nonlinear291",
        }
    }
}

/// Fully resolved options; construction never reads process configuration.
pub enum PreprocessingParams {
    QuadraticMasking {
        params: QuadraticMaskingParams,
        execution: QuadraticMaskingExecution,
        balanced_seed: bool,
        record_hot_intervals: bool,
    },
    Nonlinear291,
}

impl PreprocessingParams {
    pub fn managed_gss(seed: u64, source_wires: usize) -> Self {
        Self::QuadraticMasking {
            params: QuadraticMaskingParams::managed_gss(seed, source_wires),
            execution: QuadraticMaskingExecution::default(),
            balanced_seed: true,
            record_hot_intervals: false,
        }
    }
}

/// Measurements used by the executable's progress and optional trace reports.
pub struct QuadraticMaskingReport {
    pub band_wires: usize,
    pub atoms: usize,
    pub band_seed_gates: usize,
    pub band_reseed_gates: usize,
    pub guard_gates: usize,
    pub compute_gates: usize,
    pub hot_intervals: Vec<(u16, usize, usize)>,
}

/// The low sandwich-width wires retain the logical layout. Auxiliary wires
/// start at zero. With quadratic masking's closing guard, only the sandwich's
/// payload half is promised; nonlinear291 preserves every logical output wire.
pub struct PreprocessingOutput {
    pub circuit: Circuit,
    /// A closing junk guard narrows the output guarantee to the payload half.
    /// Nonlinear291's ingress-only guard does not set this flag.
    pub guarded: bool,
    pub quadratic_masking_report: Option<QuadraticMaskingReport>,
}
