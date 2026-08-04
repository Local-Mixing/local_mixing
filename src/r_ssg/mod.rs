//! Isolated Ran-style generation mixer (R SSG).
//!
//! This namespace vendors the behaviorally authoritative implementation from
//! `f8afe640` without replacing the NH `replace`, `transpositions`, or
//! `main_mix` modules used by the existing `sss` command.

pub mod gadgets;
pub mod main_mix;
pub mod pairs;
pub mod ranking;
pub mod replace;
pub mod sat_score;
pub mod segcircuit;
pub mod transpositions;
