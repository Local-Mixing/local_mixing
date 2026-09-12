//! Shared arena, provenance, and resumable state for database mixing, splitting,
//! and crossing. Child modules own state, scheduling, transformations and I/O;
//! stage algorithms mutate this state through the existing narrow operations.
use crate::canonicalization::xgate::XPolyBudget;
use crate::circuit::xgate::{Lits, XGate};
use crate::database::frozen::FrozenDb;
use crate::engine::arena::{Arena, Dir, NIL};
use crate::engine::moves::rules;
use crate::engine::moves::swap_words;
use crate::stages::db_mixing::replacement::{DbMode, DegreeGuard, db_replace};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rustc_hash::FxHashMap;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

mod checkpoint;
mod indices;
mod leakage_repair;
mod legacy_environment;
mod params;
mod piecewise;
mod provenance;
mod replacement;
mod reporting;
mod runtime;
mod sampling;
mod scheduling;
mod state;
use runtime::RuntimeControls;
mod transformations;
mod transport;

pub use checkpoint::STATE_VERSION;
pub use params::*;
#[cfg(test)]
pub(crate) use piecewise::PieceParts;
pub use piecewise::{PieceCfg, run_piecewise};
pub use scheduling::prof_target;
pub use state::*;
#[cfg(test)]
use transformations::*;
pub use transformations::{
    HIDDEN_SWAP_IDENTITY, Merge, TG_ACCEPT_NET, TG_RETRIES, TG_SLIDE_CAP, TG_SLIDE_TRIES,
    TWIST_PATTERNS, TwistPattern, merge_result,
};
pub(crate) use transformations::{key_of, merge_key};

#[cfg(test)]
#[path = "../../../tests/unit/engine/mixer/mix_tests.rs"]
mod mix_tests;
