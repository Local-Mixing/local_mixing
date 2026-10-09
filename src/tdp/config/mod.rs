//! Typed TDP recipes. Parsing, validation and saved-run compatibility have separate owners.
mod parse;
mod validate;
pub(crate) use crate::stages::preprocessing::PreprocessingMode;
use crate::stages::preprocessing::nonlinear291::nonlinear_tdp_resource_plan;
use crate::stages::sandwich::sandwich_default_s;
use crate::tdp::paths::*;
pub(crate) use parse::*;
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
pub(crate) use validate::*;

pub(crate) const MAX_SEED_FILE_BYTES: u64 = 256;
/// Largest seed the driver draws or accepts (`2^63 - 16`, recorded as
/// `seed_bits=63`; docs/STAGE4_RUST_DESIGN.md §2.3).
pub(crate) const MAX_SEED: u64 = 9_223_372_036_854_775_792;
/// The source sizes the validation runs used (§2.4): n -> |C| = |D|.
pub(crate) const VALIDATED_SOURCE_GATES: &[(usize, usize)] = &[(64, 1000), (128, 3000)];

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

/// Stage-2 preset (docs/STAGE4_RUST_DESIGN.md §11.2; docs/STAGE2_LINKABILITY.md §6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Preset {
    /// The recommended configuration: A plus bounded chains (CHAIN=2, LEN=8).
    L8,
    /// SHUFFLING=8, RETURN_HOME=0, PREMASK=1, SEED_AVOID_C1=1.
    A,
    /// Every knob explicit in the config.
    Custom,
    /// nonlinear291: no embedded-masking knobs.
    None,
}

impl Preset {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Preset::L8 => "L8",
            Preset::A => "A",
            Preset::Custom => "custom",
            Preset::None => "none",
        }
    }
}

/// The effective embedded-masking knobs (what gen sees: a variable unset is
/// its base value).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Stage2Knobs {
    pub(crate) mask_pair_wires: usize,
    pub(crate) max_open_masks: usize,
    pub(crate) min_open_masks: usize,
    pub(crate) balanced_masks: bool,
    pub(crate) shuffling_segments: usize,
    pub(crate) shuffling_return_home: bool,
    pub(crate) premask: bool,
    pub(crate) seed_avoid_c1: bool,
    pub(crate) chain: usize,
    pub(crate) chain_len: usize,
}

impl Stage2Knobs {
    /// gen's base parameters (`EmbeddedMaskingParams::managed_tdp`).
    pub(crate) const BASE: Stage2Knobs = Stage2Knobs {
        mask_pair_wires: 2,
        max_open_masks: 3,
        min_open_masks: 2,
        balanced_masks: true,
        shuffling_segments: 0,
        shuffling_return_home: true,
        premask: false,
        seed_avoid_c1: false,
        chain: 0,
        chain_len: 0,
    };
    pub(crate) const A: Stage2Knobs = Stage2Knobs {
        shuffling_segments: 8,
        shuffling_return_home: false,
        premask: true,
        seed_avoid_c1: true,
        ..Stage2Knobs::BASE
    };
    pub(crate) const L8: Stage2Knobs = Stage2Knobs {
        chain: 2,
        chain_len: 8,
        ..Stage2Knobs::A
    };

    /// `(config key, script flag suffix, value)` in the manifest order.
    pub(crate) fn fields(&self) -> [(&'static str, &'static str, String); 10] {
        let b = |v: bool| u8::from(v).to_string();
        [
            (
                "mask_pair_wires",
                "mask-pair-wires",
                self.mask_pair_wires.to_string(),
            ),
            (
                "max_open_masks",
                "max-open-masks",
                self.max_open_masks.to_string(),
            ),
            (
                "min_open_masks",
                "min-open-masks",
                self.min_open_masks.to_string(),
            ),
            ("balanced_masks", "balanced-masks", b(self.balanced_masks)),
            (
                "shuffling_segments",
                "shuffling-segments",
                self.shuffling_segments.to_string(),
            ),
            (
                "shuffling_return_home",
                "shuffling-return-home",
                b(self.shuffling_return_home),
            ),
            ("premask", "premask", b(self.premask)),
            ("seed_avoid_c1", "seed-avoid-c1", b(self.seed_avoid_c1)),
            ("chain", "chain", self.chain.to_string()),
            ("chain_len", "chain-len", self.chain_len.to_string()),
        ]
    }
}

/// The stage-3 schedule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Schedule {
    /// `--profile 2,2,2,2,2`, `--moves int(G*2*2*1.3)` (val_chain.sh).
    Validated,
    /// profile `n0,n0+hold,n0+hold,r,r`, moves `int((n0+hold)*r*G*1.3)`.
    Custom {
        growth: String,
        hold: String,
        expand: String,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SourceGatesRule {
    Validated,
    Explicit,
    Formula,
}

impl SourceGatesRule {
    pub(crate) fn name(self) -> &'static str {
        match self {
            SourceGatesRule::Validated => "validated",
            SourceGatesRule::Explicit => "explicit",
            SourceGatesRule::Formula => "formula",
        }
    }
}

/// `[packing]`: `tdp_packing run` parameters (defaults = val_chain.sh).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PackingConfig {
    pub(crate) window_gates: u64,
    pub(crate) interval_cap: u64,
    pub(crate) monomial_budget: u64,
    pub(crate) span_cap: u64,
    pub(crate) split_budget: u64,
    pub(crate) fix_iterations: u64,
    pub(crate) fallback_iterations: u64,
    pub(crate) anf_cap_semantics: String,
    pub(crate) allow_floor_miss: bool,
    pub(crate) allow_over_budget: bool,
}

impl Default for PackingConfig {
    fn default() -> Self {
        PackingConfig {
            window_gates: 512,
            interval_cap: 512,
            monomial_budget: 24576,
            span_cap: 3000,
            split_budget: 65536,
            fix_iterations: 4,
            fallback_iterations: 4,
            anf_cap_semantics: "cpython39".into(),
            allow_floor_miss: false,
            allow_over_budget: false,
        }
    }
}

impl PackingConfig {
    /// `(script flag, value, default)` for the numeric/enum parameters.
    pub(crate) fn flags(&self) -> Vec<(&'static str, String, String)> {
        let d = PackingConfig::default();
        vec![
            (
                "--packing-window-gates",
                self.window_gates.to_string(),
                d.window_gates.to_string(),
            ),
            (
                "--packing-interval-cap",
                self.interval_cap.to_string(),
                d.interval_cap.to_string(),
            ),
            (
                "--packing-monomial-budget",
                self.monomial_budget.to_string(),
                d.monomial_budget.to_string(),
            ),
            (
                "--packing-span-cap",
                self.span_cap.to_string(),
                d.span_cap.to_string(),
            ),
            (
                "--packing-split-budget",
                self.split_budget.to_string(),
                d.split_budget.to_string(),
            ),
            (
                "--packing-fix-iterations",
                self.fix_iterations.to_string(),
                d.fix_iterations.to_string(),
            ),
            (
                "--packing-fallback-iterations",
                self.fallback_iterations.to_string(),
                d.fallback_iterations.to_string(),
            ),
            (
                "--packing-anf-cap-semantics",
                self.anf_cap_semantics.clone(),
                d.anf_cap_semantics.clone(),
            ),
        ]
    }

    pub(crate) fn is_default(&self) -> bool {
        *self == PackingConfig::default()
    }
}

/// Where the run's seeds come from (§8.2). Deliberately not `Debug`.
#[derive(Clone)]
pub(crate) enum SeedPlan {
    /// Fresh OS CSPRNG draws (the default).
    Random,
    /// From `[calibration] seed_file` (format 2): calibration only.
    Explicit { gen_seed: u64, mix_seed: u64 },
}

/// Deliberately does not derive `Debug`: `seeds` may hold explicit seeds.
#[derive(Clone)]
pub(crate) struct ResolvedConfig {
    pub(crate) recipe_version: u8,
    pub(crate) source_path: Option<PathBuf>,
    pub(crate) preset: Preset,
    pub(crate) knobs: Stage2Knobs,
    pub(crate) qc_enabled: bool,
    pub(crate) qc_seed: u64,
    pub(crate) qc_reference: Option<PathBuf>,
    pub(crate) n: usize,
    pub(crate) run_dir: PathBuf,
    pub(crate) generated_run_dir: bool,
    pub(crate) build_release: bool,
    pub(crate) build_target_dir: PathBuf,
    pub(crate) adopt_existing_run: bool,
    pub(crate) frozen_db: SourcedPath,
    pub(crate) preprocessing_mode: PreprocessingMode,
    pub(crate) calibration_only: bool,
    pub(crate) seeds: SeedPlan,
    /// The resolved |C| = |D| and how it was chosen.
    pub(crate) mcd: usize,
    pub(crate) mcd_rule: SourceGatesRule,
    pub(crate) schedule: Schedule,
    pub(crate) packing: PackingConfig,
    pub(crate) threads: Option<String>,
    pub(crate) stop_after: Option<String>,
    pub(crate) force_from: Option<String>,
    pub(crate) allow_empty_store: bool,
    pub(crate) frozen_filter: FrozenFilter,
    // Stage-3 piecewise-parallel rounds. `pieces`
    // and `min_block_size` are locked recipe values; threads are lifecycle.
    pub(crate) pieces: Option<String>,
    pub(crate) min_block_size: Option<String>,
    pub(crate) piece_threads: Option<String>,
}

impl ResolvedConfig {
    pub(crate) fn stop_after_stage(&self) -> u64 {
        self.stop_after
            .as_deref()
            .unwrap_or("4")
            .parse()
            .expect("validated stop_after")
    }
}
