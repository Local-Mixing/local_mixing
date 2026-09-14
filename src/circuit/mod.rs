//! Circuit representations, evaluation, formats and generic randomization.
pub mod evaluate;
pub mod formats;
pub mod g57;
pub mod operations;
mod permutation;
pub mod randomize;
pub mod types;

pub mod xgate;

pub use evaluate::{U1024, lane_state_len};
pub use g57::*;
pub use permutation::Permutation;
pub use randomize::{random_circuit, shoot_random_gate};
pub use types::Circuit;
pub use xgate::{Lits, XGate, eval_lanes, eval_lanes4, eval_limbs, eval_u64, eval_u1024, max_wire};
