//! GSS db_mixing and the immutable replacement-store runtime.

// Compatibility names; algorithms and storage each have a single owner.
pub use crate::database::{frozen, lookup_cache};
pub use crate::stages::db_mixing::{leakage_repair as quality, replacement as db_replace};

// Historical mixing drivers remain available to explicit comparison builds.
#[cfg(feature = "legacy-tools")]
#[path = "../../security_tests/support/db_mixing/convex.rs"]
pub mod convex;
#[cfg(feature = "legacy-tools")]
#[path = "../../security_tests/support/db_mixing/main_mix.rs"]
pub mod main_mix;
#[cfg(feature = "legacy-tools")]
#[path = "../../security_tests/support/db_mixing/main_mix_cnot.rs"]
pub mod main_mix_cnot;
#[cfg(feature = "legacy-tools")]
#[path = "../../security_tests/support/db_mixing/pairs.rs"]
pub mod pairs;
#[cfg(feature = "legacy-tools")]
#[path = "../../security_tests/support/db_mixing/ranking.rs"]
pub mod ranking;
#[cfg(feature = "legacy-tools")]
#[path = "../../security_tests/support/db_mixing/replace.rs"]
pub mod replace;
#[cfg(feature = "legacy-tools")]
#[path = "../../security_tests/support/db_mixing/sat_score.rs"]
pub mod sat_score;
#[cfg(feature = "legacy-tools")]
#[path = "../../security_tests/support/db_mixing/segcircuit.rs"]
pub mod segcircuit;
#[cfg(feature = "legacy-tools")]
#[path = "../../security_tests/support/db_mixing/transpositions.rs"]
pub mod transpositions;
#[cfg(feature = "legacy-tools")]
#[path = "../../security_tests/support/db_mixing/util.rs"]
pub mod util;

#[cfg(feature = "legacy-db-tools")]
#[path = "../../db_gen/support/wide_db.rs"]
pub mod wide_db;
