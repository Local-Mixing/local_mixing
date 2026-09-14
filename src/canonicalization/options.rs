//! Explicit options for reusable canonicalization computations.
use super::environment;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CanonicalizationOptions {
    /// Total candidates charged across the entire Rule-L recursion tree.
    /// None is unbounded; Some(0) rejects any search requiring Rule L.
    pub rule_l_branch_cap: Option<u64>,
    /// Emit timing diagnostics at or above this threshold; None is silent.
    pub trace_threshold_ms: Option<u128>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct G57CanonicalizationOptions {
    /// Maximum reduced monomials per wire; None is unbounded.
    pub monomial_cap: Option<usize>,
    pub canonicalization: CanonicalizationOptions,
}

// None selects the process environment; explicit callers supply Some. Read the
// environment trace threshold lazily, only when tracing is enabled.
pub(super) fn trace_enabled(options: Option<&CanonicalizationOptions>) -> bool {
    options.map_or_else(environment::compression_trace_enabled, |value| {
        value.trace_threshold_ms.is_some()
    })
}

pub(super) fn trace_threshold_ms(options: Option<&CanonicalizationOptions>) -> u128 {
    options.map_or_else(environment::compression_trace_threshold_ms, |value| {
        value.trace_threshold_ms.unwrap_or(u128::MAX)
    })
}
