//! Polynomial construction and deterministic database-key canonicalization.
mod cache;
mod legacy_environment;
mod options;
pub use options::{CanonicalizationOptions, G57CanonicalizationOptions};
pub mod canonicalize;
pub mod keys;
pub mod polynomial;
mod window;
pub mod xgate;
pub use cache::{CANON_CACHE_HITS, CANON_CACHE_QUERIES};
pub use canonicalize::*;
pub use keys::*;
pub use polynomial::*;

#[cfg(test)]
#[path = "../../tests/unit/canonicalization/options.rs"]
mod options_tests;
