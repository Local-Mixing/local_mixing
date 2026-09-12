//! Circuit representations, evaluation, formats and generic randomization.
//! Canonicalization exports below preserve old public imports.
pub mod evaluate;
pub mod formats;
pub mod g57;
pub mod operations;
mod permutation;
pub mod randomize;
pub mod types;

#[cfg(feature = "legacy-tools")]
#[path = "../../security_tests/support/circuit/wide_fragment.rs"]
pub mod wide_fragment;
#[cfg(feature = "legacy-db-tools")]
#[path = "../../db_gen/support/xcanon.rs"]
pub mod xcanon;
pub mod xgate;

pub use evaluate::{U1024, lane_state_len};
pub use g57::*;
pub use permutation::Permutation;
pub use types::{Circuit, CnotCircuit};
// Compatibility exports for callers predating dedicated canonicalization ownership.
pub use crate::canonicalization::*;
pub use randomize::{random_circuit, shoot_random_gate};
#[cfg(feature = "legacy-tools")]
pub use wide_fragment::{FragmentStats, FragmentStyle, fragment_wide_post_shuffle};
pub use xgate::{Lits, XGate, eval_lanes, eval_lanes4, eval_limbs, eval_u64, eval_u1024, max_wire};
