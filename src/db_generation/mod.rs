// Replacement-database construction and migration: everything needed to
// recreate the frozen regular and curated stores. See README.md in this
// folder for the end-to-end recipe.
//
// regular.rs is the feature-gated regular RocksDB -> sharded-LMDB generator;
// curated_full.rs is the uncapped curated composite build library
// (bin/build_curated_full.rs is its CLI); frozen_build.rs converts a sharded
// store into a frozen-table directory (bin/frozen_from_lmdb.rs is its CLI,
// including the composite-direct route the uncapped store requires);
// bin/frozen_filters_build.rs writes the optional filters.bin miss filter.
// The dependency-light validation code is always available; RocksDB-linked
// builders are gated behind the `legacy-db-tools` feature.
pub mod curated_full;
pub mod frozen_build;
#[cfg(feature = "legacy-db-tools")]
pub mod regular;
