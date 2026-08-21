// Pipeline stage 3 — DB mixing (phase A) and the frozen replacement store's
// runtime read path.
//
// frozen.rs is the immutable frozen-store reader (FROZEN_DB_DIR /
// FROZEN_CURATED_DIR); db_replace.rs is the fmix DB splice channel that keys
// heterogeneous windows against it. The remaining modules are the sss/ssg
// shuffle-shoot-shuffle replacement game (the older DB-mixing driver, still
// maintained): main_mix / main_mix_cnot are the round-loop drivers invoked by
// the sss subcommand, replace.rs the expand/compress engine, pairs.rs the
// window replacement primitives, transpositions.rs the SAMF + shooting game,
// ranking.rs / sat_score.rs candidate selection, segcircuit.rs chunked
// storage, util.rs shared odds and ends.
//
// Entry binary: bin/fmix.rs (phase A via --gss --phase-a; its --split /
// --resume modes belong to postprocessing but share this engine).
pub mod convex;
pub mod db_replace;
pub mod frozen;
pub mod main_mix;
pub mod main_mix_cnot;
pub mod pairs;
pub mod ranking;
pub mod replace;
pub mod sat_score;
pub mod segcircuit;
pub mod transpositions;
pub mod util;
