//! Stage 7 (unpacking): resynthesize a circuit from the deliverable's ANF
//! windows (docs/STAGE7_UNPACKING.md). Library code here reads no
//! environment variables (the `src/stages` contract).
//!
//! Each window of `deliverable.tdp4` is handed, as its ANF alone, to a port
//! of the `anf_synth` bidirectional search (`synth`, over a per-window
//! local variable universe, `spec`, with a width-generic monomial `mono`
//! and the pairwise peephole `postopt`); the synthesized gates are checked
//! exactly against the window (`verify`), the windows are concatenated in
//! order (`run`) and the caller canonicalizes the whole gate list with
//! `canonicalization::gate_order::xgate_canonicalize`, so that the window
//! boundaries vanish. A window that cannot be synthesized falls back to
//! the packed circuit's gates (verbatim, after the same exact check) or is
//! an error — or, in the delivery stage's refinement loop (`refine`,
//! docs/DELIVERY_STAGE.md), is cut in two, re-emitted and unpacked again.
//!
//! The production entry point is `tdp_unpacking run`
//! (`src/programs/tdp_unpacking/mod.rs`).
pub mod mono;
pub mod postopt;
pub mod refine;
pub mod run;
pub mod spec;
pub mod synth;
pub mod types;
pub mod verify;

pub use refine::{RefineOutcome, RefineParams};
pub use run::{
    PackedWindows, UnpackSummary, WinStatus, WindowResult, concatenate, unpack_selected,
    unpack_windows,
};
pub use synth::{SynthParams, SynthStatus};
pub use types::{Stage7Error, UnpackParams};
