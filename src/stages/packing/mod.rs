//! Stage 4 (packing): windowed ANF delivery. The canonical oracle is the
//! Python prototype set in `scripts/windowed_delivery/`; the Rust port
//! follows `docs/STAGE4_RUST_DESIGN.md`. Library code here reads no
//! environment variables (the `src/stages` contract).
//!
//! Ported so far: the C-view exposure engine (`gf2`, `cview`), shared with
//! the `hmap_affine` security tool.
pub mod cview;
pub mod gf2;
