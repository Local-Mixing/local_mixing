//! Stage 4 (packing): windowed ANF delivery. The canonical oracle is the
//! Python prototype set in `scripts/windowed_delivery/`; the Rust port
//! follows `docs/STAGE4_RUST_DESIGN.md`. Library code here reads no
//! environment variables (the `src/stages` contract).
//!
//! Ported so far:
//! * the C-view exposure engine (`gf2`, `cview`), shared with the
//!   `hmap_affine` security tool;
//! * window ANF composition and its caps on a CPython 3.9 set emulator
//!   (`pyrand`, `pyset`, `mono`, `anf`), the oracle-format readers
//!   (`oracle_io`), the deliverable body and container (`serialize`), the
//!   checks (`verify`) and the `anf_compose.py` size report (`census`).
pub mod anf;
pub mod census;
pub mod cview;
pub mod gf2;
pub mod mono;
pub mod oracle_io;
pub mod pyrand;
pub mod pyset;
pub mod serialize;
pub mod types;
pub mod verify;
