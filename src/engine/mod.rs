// Shared state and equivalence-walk machinery behind fmix. One Mixer drives
// every mode, while stage-specific algorithms live with their pipeline stage:
// Phase A's splice channel is in db_mixing::db_replace, production Stage-4
// splitting is in postprocessing::splitting, the Stage-5 crossing walk is in
// postprocessing::cross_walk, and final compression is in
// postprocessing::compress.
pub mod arena;
pub mod format;
pub mod mix;
pub mod rules;
pub mod stats;
pub mod swap_words;
pub mod xpoly;

// Compatibility path for callers that predate XGate's move into the shared
// circuit layer. New code should import `crate::circuit::xgate`.
#[doc(hidden)]
pub use crate::circuit::xgate;

#[cfg(test)]
mod tests;
