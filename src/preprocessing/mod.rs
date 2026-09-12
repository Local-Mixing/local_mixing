//! Compatibility imports for the former combined stages 1 and 2 module.
//! New code uses `stages::sandwich` and `stages::preprocessing`.

pub mod blinded_v5;
pub mod gadgets;
pub use crate::stages::preprocessing::slice_guards as guards;
pub use crate::stages::sandwich::construct as sandwich;
pub mod types {
    pub use crate::circuit::Circuit as CnotCircuit;
}
#[cfg(feature = "legacy-tools")]
mod shared {
    pub(super) use crate::circuit::randomize::random_wire_except;
}

#[cfg(feature = "legacy-tools")]
#[path = "../../security_tests/support/preprocessing/gadgets.rs"]
pub mod legacy_gadgets;
pub use crate::stages::preprocessing::nonlinear291 as nonlinear_gss;
pub use crate::stages::preprocessing::{PreprocessingMode, quadratic_masking};
#[cfg(feature = "legacy-tools")]
#[path = "../../security_tests/support/preprocessing/samf.rs"]
pub mod samf;

// Compatibility path for callers that predate the ownership move.
#[doc(hidden)]
#[cfg(feature = "legacy-tools")]
pub use crate::circuit::wide_fragment as fragment;
