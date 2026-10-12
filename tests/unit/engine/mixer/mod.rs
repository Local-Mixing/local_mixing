// Topic files share this module so test names and access to mixer internals
// remain unchanged. Keep reusable circuit fixtures in fixtures.rs.
use super::*;
use crate::circuit::xgate::XGate;

include!("fixtures.rs");
include!("serial_baselines.rs");
include!("piecewise_rounds.rs");
include!("scheduling.rs");
include!("gate_algebra.rs");
include!("twists.rs");
include!("splitting.rs");
include!("database_policy.rs");
include!("generations.rs");
include!("ancestry.rs");
include!("checkpoints.rs");
