// Float-and-split post-mixing stage: take a mixed g57 circuit, repeatedly float a
// gate to its collision point and split it past the colliding gate using the
// exclusivity-preserving commutation rules (gate_commutation_rules.html), under a
// max-controls cap K, until a size bound B or K-saturation; then float every gate
// to a uniform random position in its commutable box.
//
// Self-contained: does not touch the ssg/sss replacement paths or the DB.
pub mod arena;
pub mod engine;
pub mod format;
pub mod mix;
pub mod rules;
pub mod xgate;

#[cfg(test)]
mod tests;
