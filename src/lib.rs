pub mod canonicalization;
pub mod circuit;
pub mod database;
#[cfg(any(test, feature = "db-tools", feature = "benchmark-tools"))]
#[path = "../db_gen/mod.rs"]
pub mod db_generation;
pub mod engine;
pub mod programs;
pub mod stages;

// Optional analysis extension; implementation and API registration live together.
#[cfg(feature = "python-extension")]
#[path = "../security_tests/python/mod.rs"]
mod python;

pub mod gss;
