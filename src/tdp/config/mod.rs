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

/// `[delivery]`: the unpacking sub-step of stage 6 (`tdp_unpacking run
/// --refine`, docs/DELIVERY_STAGE.md). The packing sub-step's parameters are
/// `[packing]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DeliveryConfig {
    /// Adaptive (the default) or windowed; the fields below apply to the
    /// windowed mode only (the adaptive mode's are `[adaptive]`).
    pub(crate) mode: DeliveryMode,
    /// Stop the stage after the packer's deliverable: no resynthesis, no
    /// lattice refinement, no `unpacked.mpmct1`.
    pub(crate) anf_only: bool,
    /// Synthesis wall-clock limit per window per round, seconds (a
    /// normalized number).
    pub(crate) time_limit: String,
    pub(crate) restarts: u64,
    pub(crate) beam: u64,
    /// Refinement rounds: a window that does not resynthesize is cut in two
    /// and both halves are repacked and unpacked, up to this many rounds.
    pub(crate) max_rounds: u64,
}

impl Default for DeliveryConfig {
    fn default() -> Self {
        DeliveryConfig {
            mode: DeliveryMode::Adaptive,
            anf_only: false,
            time_limit: "30".into(),
            restarts: 16,
            beam: 6,
            max_rounds: 16,
        }
    }
}

impl DeliveryConfig {
    /// `(script flag, value, default)` for the numeric parameters
    /// (`anf_only` is the switch `--delivery-anf-only`).
    pub(crate) fn flags(&self) -> Vec<(&'static str, String, String)> {
        let d = DeliveryConfig::default();
        vec![
            (
                "--delivery-time-limit",
                self.time_limit.clone(),
                d.time_limit,
            ),
            (
                "--delivery-restarts",
                self.restarts.to_string(),
                d.restarts.to_string(),
            ),
            ("--delivery-beam", self.beam.to_string(), d.beam.to_string()),
            (
                "--delivery-max-rounds",
                self.max_rounds.to_string(),
                d.max_rounds.to_string(),
            ),
        ]
    }

    pub(crate) fn is_default(&self) -> bool {
        *self == DeliveryConfig::default()
    }
}

/// How stage 6 (delivery) is run: the adaptive C-aware resynthesis (the
/// default since recipe v13) or the windowed packing + unpacking with
/// lattice refinement (recipe v12's only stage 6).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum DeliveryMode {
    #[default]
    Adaptive,
    Windowed,
}

impl DeliveryMode {
    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value {
            "adaptive" => Some(Self::Adaptive),
            "windowed" => Some(Self::Windowed),
            _ => None,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Adaptive => "adaptive",
            Self::Windowed => "windowed",
        }
    }
}

/// How the source C is drawn when `source.path` is unset: uniformly random
/// g57 gates, or RC's balanced construction (every internal value read
/// exactly twice).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum SourceConstruction {
    #[default]
    Random,
    Balanced,
}

impl SourceConstruction {
    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value {
            "random" => Some(Self::Random),
            "balanced" => Some(Self::Balanced),
            _ => None,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::Balanced => "balanced",
        }
    }
}

/// `[three_mask]`: the knobs of the three-mask stage-2 compute
/// (`preprocessing.mode = "three-mask"`). The defaults are the campaign
/// configuration "three-mask modified / v2" (random helpers, unique cover).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ThreeMaskConfig {
    /// 0 = auto: the larger of 2n and 128, rounded up to a multiple of 16.
    pub(crate) band_wires: u64,
    /// 0 = auto: banks of 16 wires (`band_wires / 16`, the design's geometry).
    pub(crate) banks: u64,
    pub(crate) retire_budget: u64,
    pub(crate) max_retiring: u64,
    pub(crate) rotation: u64,
    pub(crate) maturity: u64,
    pub(crate) visit_sweeps: u64,
    pub(crate) growth: u64,
    pub(crate) warmup: u64,
    pub(crate) segments: u64,
    pub(crate) mask_count: u64,
    pub(crate) random_helpers: bool,
    pub(crate) unique_cover: bool,
}

impl Default for ThreeMaskConfig {
    fn default() -> Self {
        ThreeMaskConfig {
            band_wires: 0,
            banks: 0,
            retire_budget: 2,
            max_retiring: 0,
            rotation: 1,
            maturity: 2,
            visit_sweeps: 2,
            growth: 8,
            warmup: 4,
            segments: 6,
            mask_count: 3,
            random_helpers: true,
            unique_cover: true,
        }
    }
}

impl ThreeMaskConfig {
    /// `(config key, script flag suffix, value, default)` in the manifest order.
    pub(crate) fn fields(&self) -> [(&'static str, &'static str, String, String); 13] {
        let d = ThreeMaskConfig::default();
        let b = |v: bool| u8::from(v).to_string();
        [
            (
                "band_wires",
                "band-wires",
                self.band_wires.to_string(),
                d.band_wires.to_string(),
            ),
            (
                "banks",
                "banks",
                self.banks.to_string(),
                d.banks.to_string(),
            ),
            (
                "retire_budget",
                "retire-budget",
                self.retire_budget.to_string(),
                d.retire_budget.to_string(),
            ),
            (
                "max_retiring",
                "max-retiring",
                self.max_retiring.to_string(),
                d.max_retiring.to_string(),
            ),
            (
                "rotation",
                "rotation",
                self.rotation.to_string(),
                d.rotation.to_string(),
            ),
            (
                "maturity",
                "maturity",
                self.maturity.to_string(),
                d.maturity.to_string(),
            ),
            (
                "visit_sweeps",
                "visit-sweeps",
                self.visit_sweeps.to_string(),
                d.visit_sweeps.to_string(),
            ),
            (
                "growth",
                "growth",
                self.growth.to_string(),
                d.growth.to_string(),
            ),
            (
                "warmup",
                "warmup",
                self.warmup.to_string(),
                d.warmup.to_string(),
            ),
            (
                "segments",
                "segments",
                self.segments.to_string(),
                d.segments.to_string(),
            ),
            (
                "mask_count",
                "mask-count",
                self.mask_count.to_string(),
                d.mask_count.to_string(),
            ),
            (
                "random_helpers",
                "random-helpers",
                b(self.random_helpers),
                b(d.random_helpers),
            ),
            (
                "unique_cover",
                "unique-cover",
                b(self.unique_cover),
                b(d.unique_cover),
            ),
        ]
    }

    pub(crate) fn is_default(&self) -> bool {
        *self == ThreeMaskConfig::default()
    }
}

/// `[adaptive]`: stage 6 in its adaptive mode (`tdp_adaptive run`,
/// docs/ADAPTIVE_STAGE.md). The defaults are the server prototype's policy
/// (variant v2, 2 % two-round stop rule).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AdaptiveConfig {
    /// Rounds to run at most; 0 = until the size plateaus.
    pub(crate) max_rounds: u64,
    /// End of mixing when two consecutive rounds compress jointly by less
    /// than this fraction (a normalized number in [0, 1)).
    pub(crate) stop_fraction: String,
    pub(crate) seeds_per_block: u64,
    /// Synthesis wall limit per block, seconds, for blocks of at most 1000
    /// gates (x2 up to 4000 gates, x4 above).
    pub(crate) time_limit: String,
    pub(crate) restarts: u64,
    pub(crate) beam: u64,
    pub(crate) term_cap: u64,
    pub(crate) max_block_gates: u64,
    pub(crate) segments: u64,
    pub(crate) avoid_radius: u64,
    pub(crate) refinement_levels: u64,
    pub(crate) min_refine_gates: u64,
    pub(crate) census_batches: u64,
    /// `smaller-not-leakier` (v2) or `smaller`.
    pub(crate) accept: String,
    pub(crate) canonical_order: bool,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        AdaptiveConfig {
            max_rounds: 0,
            stop_fraction: "0.02".into(),
            seeds_per_block: 2,
            time_limit: "60".into(),
            restarts: 16,
            beam: 6,
            term_cap: 131_072,
            max_block_gates: 65_536,
            segments: 16,
            avoid_radius: 64,
            refinement_levels: 2,
            min_refine_gates: 128,
            census_batches: 96,
            accept: "smaller-not-leakier".into(),
            canonical_order: true,
        }
    }
}

impl AdaptiveConfig {
    /// `(script flag, value, default)` for the value parameters
    /// (`canonical_order` is the switch `--adaptive-no-canonical-order`).
    pub(crate) fn flags(&self) -> Vec<(&'static str, String, String)> {
        let d = AdaptiveConfig::default();
        vec![
            (
                "--adaptive-max-rounds",
                self.max_rounds.to_string(),
                d.max_rounds.to_string(),
            ),
            (
                "--adaptive-stop-fraction",
                self.stop_fraction.clone(),
                d.stop_fraction,
            ),
            (
                "--adaptive-seeds-per-block",
                self.seeds_per_block.to_string(),
                d.seeds_per_block.to_string(),
            ),
            (
                "--adaptive-time-limit",
                self.time_limit.clone(),
                d.time_limit,
            ),
            (
                "--adaptive-restarts",
                self.restarts.to_string(),
                d.restarts.to_string(),
            ),
            ("--adaptive-beam", self.beam.to_string(), d.beam.to_string()),
            (
                "--adaptive-term-cap",
                self.term_cap.to_string(),
                d.term_cap.to_string(),
            ),
            (
                "--adaptive-max-block-gates",
                self.max_block_gates.to_string(),
                d.max_block_gates.to_string(),
            ),
            (
                "--adaptive-segments",
                self.segments.to_string(),
                d.segments.to_string(),
            ),
            (
                "--adaptive-avoid-radius",
                self.avoid_radius.to_string(),
                d.avoid_radius.to_string(),
            ),
            (
                "--adaptive-refinement-levels",
                self.refinement_levels.to_string(),
                d.refinement_levels.to_string(),
            ),
            (
                "--adaptive-min-refine-gates",
                self.min_refine_gates.to_string(),
                d.min_refine_gates.to_string(),
            ),
            (
                "--adaptive-census-batches",
                self.census_batches.to_string(),
                d.census_batches.to_string(),
            ),
            ("--adaptive-accept", self.accept.clone(), d.accept),
        ]
    }

    pub(crate) fn is_default(&self) -> bool {
        *self == AdaptiveConfig::default()
    }
}

/// The stage-5 crossing walk's controls (`[crossing]`), each `None` = the
/// default (target factor 2, penalty base 3, threshold 1, tolerance divisor
/// 25: the archive-era driver's; move budget 3 * target: RC 2026-10-12, it was
/// 2 * target on 2026-10-11 and the archive ran 6 * target). `xmoves` is an
/// absolute budget overriding the factor.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct CrossingConfig {
    pub(crate) xr: Option<String>,
    pub(crate) xb: Option<String>,
    pub(crate) xc: Option<String>,
    pub(crate) xtdiv: Option<String>,
    pub(crate) xmf: Option<String>,
    pub(crate) xmoves: Option<String>,
}

impl CrossingConfig {
    pub(crate) const DEFAULTS: [(&'static str, &'static str); 6] = [
        ("target_size_factor", "2"),
        ("width_penalty_base", "3"),
        ("width_penalty_threshold", "1"),
        ("size_tolerance_divisor", "25"),
        ("move_budget_factor", "3"),
        ("move_attempts", "factor*target"),
    ];

    /// `(manifest key, script flag, value)`; the value is the default when unset.
    pub(crate) fn fields(&self) -> [(&'static str, &'static str, String); 6] {
        let v = |o: &Option<String>, d: &str| o.clone().unwrap_or_else(|| d.to_owned());
        let d = Self::DEFAULTS;
        [
            (d[0].0, "--crossing-target-size-factor", v(&self.xr, d[0].1)),
            (d[1].0, "--crossing-width-penalty-base", v(&self.xb, d[1].1)),
            (
                d[2].0,
                "--crossing-width-penalty-threshold",
                v(&self.xc, d[2].1),
            ),
            (
                d[3].0,
                "--crossing-size-tolerance-divisor",
                v(&self.xtdiv, d[3].1),
            ),
            (
                d[4].0,
                "--crossing-move-budget-factor",
                v(&self.xmf, d[4].1),
            ),
            (d[5].0, "--crossing-move-attempts", v(&self.xmoves, d[5].1)),
        ]
    }
}

/// Where the run's seeds come from (§8.2). Deliberately not `Debug`.
#[derive(Clone)]
pub(crate) enum SeedPlan {
    /// Fresh OS CSPRNG draws (the default).
    Random,
    /// From `[calibration] seed_file`: calibration only. The stage-4/5/6
    /// seeds (split, cross, unpack = the delivery seed) are optional in the
    /// file (drawn fresh when absent).
    Explicit {
        gen_seed: u64,
        mix_seed: u64,
        split_seed: Option<u64>,
        cross_seed: Option<u64>,
        unpack_seed: Option<u64>,
    },
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
    /// `[three_mask]` (its defaults when the mode is not three-mask).
    pub(crate) three_mask: ThreeMaskConfig,
    /// `source.construction` (random unless `balanced`; irrelevant with a
    /// supplied `source.path`).
    pub(crate) source_construction: SourceConstruction,
    pub(crate) calibration_only: bool,
    pub(crate) seeds: SeedPlan,
    /// The resolved |C| = |D| and how it was chosen.
    pub(crate) mcd: usize,
    pub(crate) mcd_rule: SourceGatesRule,
    pub(crate) schedule: Schedule,
    pub(crate) packing: PackingConfig,
    pub(crate) crossing: CrossingConfig,
    pub(crate) delivery: DeliveryConfig,
    /// `[adaptive]` (its defaults under the windowed mode).
    pub(crate) adaptive: AdaptiveConfig,
    pub(crate) threads: Option<String>,
    pub(crate) stop_after: Option<String>,
    pub(crate) force_from: Option<String>,
    /// `run.skip_stages`: stages 2..6 not run (sorted, deduplicated).
    pub(crate) skip_stages: Vec<u64>,
    pub(crate) allow_empty_store: bool,
    pub(crate) frozen_filter: FrozenFilter,
    // Stage-3 piecewise-parallel rounds. `pieces`
    // and `min_block_size` are locked recipe values; threads are lifecycle.
    pub(crate) pieces: Option<String>,
    pub(crate) min_block_size: Option<String>,
    pub(crate) piece_threads: Option<String>,
}

/// Stage 6, delivery (adaptive resynthesis, or packing then unpacking with
/// lattice refinement), is the last stage.
pub(crate) const LAST_STAGE: u64 = 6;

impl ResolvedConfig {
    pub(crate) fn stop_after_stage(&self) -> u64 {
        self.stop_after
            .as_deref()
            .unwrap_or("6")
            .parse()
            .expect("validated stop_after")
    }

    /// Whether stage `k` runs: not past the stop point and not skipped.
    pub(crate) fn stage_on(&self, k: u64) -> bool {
        k <= self.stop_after_stage() && !self.skip_stages.contains(&k)
    }

    /// Whether stage 6 runs in its adaptive mode.
    pub(crate) fn adaptive_on(&self) -> bool {
        self.stage_on(6) && self.delivery.mode == DeliveryMode::Adaptive
    }

    /// Whether stage 6 runs the windowed packing sub-step.
    pub(crate) fn packing_on(&self) -> bool {
        self.stage_on(6) && self.delivery.mode == DeliveryMode::Windowed
    }

    /// Whether the windowed delivery runs its unpacking sub-step (stage 6
    /// on, windowed, and not `delivery.anf_only`).
    pub(crate) fn unpacking_on(&self) -> bool {
        self.packing_on() && !self.delivery.anf_only
    }

    /// The stages that run, as the driver prints them: "1-6", "1-3,6", ...
    pub(crate) fn stages_text(&self) -> String {
        let stages: Vec<u64> = (1..=LAST_STAGE).filter(|&k| self.stage_on(k)).collect();
        let mut out = Vec::new();
        let mut i = 0;
        while i < stages.len() {
            let mut j = i;
            while j + 1 < stages.len() && stages[j + 1] == stages[j] + 1 {
                j += 1;
            }
            out.push(if i == j {
                stages[i].to_string()
            } else {
                format!("{}-{}", stages[i], stages[j])
            });
            i = j + 1;
        }
        out.join(",")
    }

    pub(crate) fn skip_stages_text(&self) -> String {
        if self.skip_stages.is_empty() {
            "none".to_owned()
        } else {
            self.skip_stages
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(",")
        }
    }
}
