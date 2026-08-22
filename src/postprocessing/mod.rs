// Postprocessing stages 4-6 — structure breaking and the final pass: the
// production split stage (fmix --split), the crossing walk (fmix --resume),
// and fcompress.
//
// `splitting` owns production GSS Stage 4, `cross_walk` owns the production
// Stage-5 crossing/undo/merge walk, and `compress` owns the attacker-computable
// greedy Stage-6 compressor. Alternative walks live under `experimental`.
pub mod compress;
pub mod cross_walk;
pub mod splitting;
