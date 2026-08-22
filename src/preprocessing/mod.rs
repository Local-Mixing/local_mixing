// Pipeline stages 1+2 — circuit construction: the sliced sandwich and its
// gadgetization (product-share encoding, carrier representations, zero-slice
// preblocks), plus masked-swap SAMF insertion before mixing starts. Generic
// wide-gate fragmentation now belongs to the shared circuit layer.
//
// `gen_sandwich_gadget` is the production stage-1+2 binary used by
// scripts/gss_mix.sh. Manual validation lives under tests/manual/gss, while
// A/B probes live under experimental/gss; the standalone wide-fragment
// wrapper lives under experimental/circuit.
pub mod gadgets;
pub mod nonlinear_gss;
pub mod samf;

// Compatibility path for callers that predate the ownership move.
#[doc(hidden)]
pub use crate::circuit::wide_fragment as fragment;
