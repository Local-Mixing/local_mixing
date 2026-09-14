//! Offline database builders and candidate generation.
//! Shared storage and validation live under crate::database.
#[cfg(any(test, feature = "db-tools"))]
pub mod curated_full;
#[cfg(any(test, feature = "db-tools"))]
pub mod frozen_build;
#[cfg(feature = "db-tools")]
pub mod regular;
pub mod wide_gates;

#[cfg(feature = "db-tools")]
#[path = "support/mpx1.rs"]
pub mod mpx1;
#[cfg(feature = "db-tools")]
#[path = "support/wide_db.rs"]
pub mod wide_db;
#[cfg(feature = "db-tools")]
#[path = "support/xcanon.rs"]
pub mod xcanon;
