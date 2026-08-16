//! Replacement-database construction and migration primitives.
//!
//! The dependency-light validation code is always available; the RocksDB
//! builder binary is gated behind the `legacy-db-tools` feature.

pub mod curated_full;
