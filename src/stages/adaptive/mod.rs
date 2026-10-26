//! Stage 6, adaptive mode: C-aware resynthesis of the stage-5 circuit in
//! large variable-length blocks, repeated round after round until the size
//! plateaus (the server prototype `py6/stage6_run.py` variant `v2`, ported;
//! docs/ADAPTIVE_STAGE.md).
//!
//! Analysis half: `census` (every-cut direct/affine exposure of C's births,
//! with the fresh-input audit), `boundary` (which births are public
//! quadratic functions and therefore not "deeper"), `planner` (large blocks
//! grown by exact ANF composition, boundaries chosen away from exposed
//! cuts, segments planned in parallel; the exposure profile used to split a
//! block that does not synthesize).
//!
//! Synthesis half: `synth` (one block: compose its ANF, synthesize it with
//! the unpacking synthesizer, check and peephole), `round` (one round: plan,
//! synthesize every block with every seed under size-scaled budgets, refine
//! failures, accept per block the smallest candidate that is not leakier,
//! assemble and check), `driver` (the rounds, the stop rules, the final
//! circuit and its report).
pub mod types;

pub mod boundary;
pub mod census;
pub mod planner;

pub mod driver;
pub mod round;
pub mod synth;

pub use types::*;
