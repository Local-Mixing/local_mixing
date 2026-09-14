//! Typed GSS recipes. Parsing, validation and saved-run compatibility have separate owners.
mod parse;
mod validate;
use crate::gss::paths::*;
pub(crate) use crate::stages::preprocessing::PreprocessingMode;
use crate::stages::preprocessing::nonlinear291::nonlinear_gss_resource_plan;
use crate::stages::sandwich::{sandwich_default_m, sandwich_default_s};
pub(crate) use parse::*;
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
pub(crate) use validate::*;

pub(crate) const MAX_SEED_FILE_BYTES: u64 = 64;

#[derive(Clone, Debug)]
pub(crate) struct ConfigEntry {
    pub(crate) value: Option<String>,
    pub(crate) line: usize,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct RawConfig {
    pub(crate) entries: BTreeMap<String, ConfigEntry>,
}

impl RawConfig {
    pub(crate) fn value(&self, key: &str) -> Option<&str> {
        self.entries
            .get(key)
            .and_then(|entry| entry.value.as_deref())
    }

    pub(crate) fn line(&self, key: &str) -> Option<usize> {
        self.entries.get(key).map(|entry| entry.line)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ValueSource {
    Document,
    Environment,
    Unset,
}

impl ValueSource {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Document => "configuration",
            Self::Environment => "environment",
            Self::Unset => "unset",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SourcedPath {
    pub(crate) path: Option<PathBuf>,
    pub(crate) source: ValueSource,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FrozenFilter {
    Auto,
    On,
    Off,
}

/// Deliberately does not derive `Debug`: `calibration_seed` is secret.
#[derive(Clone)]
pub(crate) struct ResolvedConfig {
    pub(crate) recipe_version: u8,
    pub(crate) source_path: Option<PathBuf>,
    pub(crate) source_hash: Option<u128>,
    pub(crate) bv5_k: usize,
    pub(crate) bv5_max_open: usize,
    pub(crate) bv5_min_open: usize,
    pub(crate) bv5_balanced: bool,
    pub(crate) explicit_mask_controls: bool,
    pub(crate) qc_enabled: bool,
    pub(crate) qc_seed: u64,
    pub(crate) qc_reference: Option<PathBuf>,
    pub(crate) qc_reference_hash: Option<u128>,
    pub(crate) n: usize,
    pub(crate) run_dir: PathBuf,
    pub(crate) generated_run_dir: bool,
    pub(crate) build_release: bool,
    pub(crate) build_target_dir: PathBuf,
    pub(crate) adopt_existing_run: bool,
    pub(crate) frozen_db: SourcedPath,
    pub(crate) frozen_curated: SourcedPath,
    pub(crate) curated_value_convention: String,
    pub(crate) preprocessing_mode: PreprocessingMode,
    pub(crate) calibration_only: bool,
    pub(crate) calibration_seed: Option<String>,
    pub(crate) mcd: Option<String>,
    pub(crate) expand: Option<String>,
    pub(crate) hold: Option<String>,
    pub(crate) xr: Option<String>,
    pub(crate) xb: Option<String>,
    pub(crate) xc: Option<String>,
    pub(crate) xtdiv: Option<String>,
    pub(crate) xmoves: Option<String>,
    pub(crate) stop_after: Option<String>,
    pub(crate) force_from: Option<String>,
    pub(crate) allow_empty_store: bool,
    pub(crate) frozen_filter: FrozenFilter,
    // Stages 3-4 piecewise-parallel rounds. `pieces`
    // and `min_block_size` are locked recipe values; threads are lifecycle.
    pub(crate) pieces: Option<String>,
    pub(crate) min_block_size: Option<String>,
    pub(crate) piece_threads: Option<String>,
}
