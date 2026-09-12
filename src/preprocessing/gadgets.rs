//! Historical import facade. Runtime construction has dedicated modules;
//! comparison-only families are available with `legacy-tools`.

pub use super::guards::{slice_zero_junk_guard_dims, slice_zero_junk_guard_dims_high};
pub use super::sandwich::{
    SandwichVariant, sandwich_default_m, sandwich_default_s, sliced_sandwich_cnot,
    sliced_sandwich_with_d,
};
pub use super::types::CnotCircuit;

#[cfg(feature = "legacy-tools")]
pub use super::legacy_gadgets::*;
