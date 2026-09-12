//! Offline database builders and candidate generation.
//! Shared storage and validation live under crate::database.
#[cfg(any(test, feature = "legacy-db-tools"))]
pub mod curated_full;
#[cfg(any(test, feature = "legacy-db-tools"))]
pub mod frozen_build;
#[cfg(feature = "legacy-db-tools")]
pub mod regular;
pub mod wide_gates;
