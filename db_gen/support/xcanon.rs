//! XGate-level circuit canonicalization for the database builders.
//!
//! The implementation lives in [`crate::canonicalization::gate_order`], where
//! the runtime identity guard also uses it; this module keeps the builders'
//! historical import path.

pub use crate::canonicalization::gate_order::{xgate_adjacent_id, xgate_canonicalize, xgate_order};

#[cfg(test)]
#[path = "../../tests/unit/db_gen/support/xcanon/tests.rs"]
mod tests;
