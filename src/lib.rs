pub mod canonicalization;
pub mod circuit;
pub mod database;
#[cfg(any(test, feature = "legacy-db-tools", feature = "benchmark-tools"))]
#[path = "../db_gen/mod.rs"]
pub mod db_generation;
pub mod db_mixing;
pub mod engine;
#[cfg(feature = "legacy-tools")]
#[path = "../security_tests/support/experimental/mod.rs"]
pub mod experimental;
pub mod postprocessing;
pub mod preprocessing;
pub mod programs;
pub mod stages;

// Compatibility path for downstream callers using the pre-reorganization
// module name. New code should use `postprocessing`.
#[doc(hidden)]
pub use postprocessing as fragmentation;

// Optional analysis extension; implementation and API registration live together.
#[cfg(feature = "python-extension")]
#[path = "../security_tests/python/mod.rs"]
mod python;

pub mod gss;
