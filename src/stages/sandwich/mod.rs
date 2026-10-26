//! Stage 1: source preparation and sliced-sandwich construction.

pub mod balanced;
pub mod construct;
pub use balanced::{BalancedAudit, audit_balanced, balanced_source};
pub use construct::*;
