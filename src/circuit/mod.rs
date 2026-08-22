// Shared circuit core: the legacy G57 CircuitSeq representation, the general
// XGate representation, canonicalization, generic randomization, and generic
// wide-gate fragmentation. Stage-specific orchestration belongs above this
// module.
mod circuit;
mod randomize;

pub mod wide_fragment;
pub mod xgate;

pub use circuit::*;
pub use randomize::{random_circuit, shoot_random_gate};
pub use wide_fragment::{FragmentStats, FragmentStyle, fragment_wide_post_shuffle};
pub use xgate::{Lits, XGate, eval_lanes, eval_lanes4, eval_limbs, eval_u64, eval_u1024, max_wire};
