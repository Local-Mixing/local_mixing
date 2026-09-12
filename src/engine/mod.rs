//! Shared mutable circuit state, scheduling, checkpoints and primitive moves.
//! Stage algorithms live under crate::stages; old module names forward below.
pub mod arena;
#[doc(hidden)]
pub use crate::circuit::formats as format;
pub mod mixer;
#[doc(hidden)]
pub use mixer as mix;
pub mod moves;
#[cfg(feature = "legacy-db-tools")]
#[path = "../../db_gen/support/mpx1.rs"]
pub mod mpx1;
#[doc(hidden)]
pub use moves::{rules, swap_words};
pub mod stats;
#[doc(hidden)]
pub use crate::canonicalization::xgate as xpoly;

// Compatibility path for callers that predate XGate's move into the shared
// circuit layer. New code should import `crate::circuit::xgate`.
#[doc(hidden)]
pub use crate::circuit::xgate;

#[cfg(test)]
#[path = "../../tests/unit/engine.rs"]
mod tests;
