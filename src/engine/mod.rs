//! Shared mutable circuit state, scheduling, checkpoints and primitive moves.
//! Stage algorithms live under crate::stages.
pub mod arena;
pub mod mixer;
pub mod moves;
pub mod stats;

#[cfg(test)]
#[path = "../../tests/unit/engine.rs"]
mod tests;
