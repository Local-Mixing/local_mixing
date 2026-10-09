//! Environment registry (docs/STAGE4_RUST_DESIGN.md §11.5): every environment
//! name the crate reads, with its class. The managed driver (this module and
//! `scripts/tdp_gen.sh`, which reads the same table from
//! `scripts/tdp_env_registry.tsv`) starts stage binaries from an EMPTY
//! environment plus an allow-list and the recipe/driver variables, and
//! refuses any inherited variable that could change output.
//!
//! `env_registry_covers_every_env_read` (tests/unit/tdp.rs) scans `src/` for
//! every literal passed to an environment read and fails on an unclassified
//! name, so a new knob forces a classification decision.

/// How the managed driver treats a variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EnvClass {
    /// Output-affecting, set by the preset only. Inherited: fatal.
    Recipe,
    /// Output-affecting or run-plumbing, set by the driver from its config.
    /// Inherited: fatal (the driver has a flag or config field instead).
    Driver,
    /// The driver's own inputs (store path, binary directory, plumbing-test
    /// switch). Inherited: accepted; never passed to a stage binary as such.
    Input,
    /// Proven output-neutral (named test). Inherited: ignored with a note;
    /// the driver passes its own values.
    Neutral,
    /// Output-affecting and part of no recipe. Inherited: fatal.
    Forbidden,
    /// Diagnostics of construction experiments (traces, dumps). Inherited:
    /// fatal (they are never passed, so honouring them is impossible).
    Diagnostic,
    /// Not read by stage binaries (build tooling, trace switches). Ignored.
    Tooling,
}

impl EnvClass {
    #[cfg_attr(not(test), allow(dead_code))] // the registry TSV (tests keep it in sync)
    pub(crate) fn name(self) -> &'static str {
        match self {
            EnvClass::Recipe => "recipe",
            EnvClass::Driver => "driver",
            EnvClass::Input => "input",
            EnvClass::Neutral => "neutral",
            EnvClass::Forbidden => "forbidden",
            EnvClass::Diagnostic => "diagnostic",
            EnvClass::Tooling => "tooling",
        }
    }

    /// Whether an inherited value is a fatal error.
    pub(crate) fn inherited_is_fatal(self) -> bool {
        matches!(
            self,
            EnvClass::Recipe | EnvClass::Driver | EnvClass::Forbidden | EnvClass::Diagnostic
        )
    }
}

/// One entry: an exact name, or a prefix (`prefix = true`). `default` is the
/// effective value when unset ("-" when there is none).
#[derive(Clone, Copy, Debug)]
pub(crate) struct EnvEntry {
    pub(crate) name: &'static str,
    pub(crate) prefix: bool,
    pub(crate) class: EnvClass,
    #[cfg_attr(not(test), allow(dead_code))] // recorded in the registry TSV
    pub(crate) default: &'static str,
}

const fn exact(name: &'static str, class: EnvClass, default: &'static str) -> EnvEntry {
    EnvEntry {
        name,
        prefix: false,
        class,
        default,
    }
}

const fn prefix(name: &'static str, class: EnvClass) -> EnvEntry {
    EnvEntry {
        name,
        prefix: true,
        class,
        default: "-",
    }
}

use EnvClass::*;

pub(crate) const ENV_REGISTRY: &[EnvEntry] = &[
    // stage 2 (gen_sandwich_gadget): the preset knobs
    exact("EMBEDDED_MASKING_SHUFFLING", Recipe, "0"),
    exact("EMBEDDED_MASKING_SHUFFLING_RETURN_HOME", Recipe, "1"),
    exact("EMBEDDED_MASKING_PREMASK", Recipe, "0"),
    exact("EMBEDDED_MASKING_SEED_AVOID_C1", Recipe, "0"),
    exact("EMBEDDED_MASKING_CHAIN", Recipe, "0"),
    exact("EMBEDDED_MASKING_CHAIN_LEN", Recipe, "0"),
    exact("EMBEDDED_MASKING_K", Recipe, "2"),
    exact("EMBEDDED_MASKING_MAX_OPEN", Recipe, "3"),
    exact("EMBEDDED_MASKING_MIN_OPEN", Recipe, "2"),
    exact("EMBEDDED_MASKING_BALANCED", Recipe, "1"),
    // stage 2 diagnostics
    exact("EMBEDDED_MASKING_LGI_DEBUG", Diagnostic, "-"),
    exact("EMBEDDED_MASKING_STRADDLE_CENSUS", Diagnostic, "-"),
    exact("EMBEDDED_MASKING_DRIFT_DEBUG", Diagnostic, "-"),
    exact("EMBEDDED_MASKING_DUMP_GATES", Diagnostic, "-"),
    exact("EMBEDDED_MASKING_DIAG", Diagnostic, "-"),
    exact("EMBEDDED_MASKING_HOT_MANIFEST", Diagnostic, "-"),
    // every other stage-2 construction knob (QUAD_FIRE, RERAND, RETIRE*, ...)
    prefix("EMBEDDED_MASKING_", Forbidden),
    exact("SLICE_GUARD_LEGACY", Forbidden, "0"),
    exact("SANDWICH_VARIANT", Driver, "classic"),
    exact("TDP_SOURCE_C", Driver, "-"),
    // stage 3 (circuit_mixer) and the frozen store
    exact("FROZEN_DB_DIR", Input, "-"),
    exact("FROZEN_FILTER", Driver, "0"),
    exact("FROZEN_REGULAR_VALUE_CONVENTION", Driver, "native"),
    exact("MIXER_STOP_FLAG", Driver, "-"),
    exact("MIXER_DUMP_FLAG", Driver, "-"),
    exact("MIXER_DUMP_OUT", Forbidden, "-"),
    exact("MIXER_STOP_AT_PHASE", Forbidden, "-"),
    exact("MIXER_STOP_AT_SPLICES", Forbidden, "-"),
    exact("MIXER_DB_LEN_BAND", Forbidden, "-"),
    exact("TWIST_G57_NO_RETRY", Forbidden, "-"),
    exact("TWIST_G57_NO_SLIDE", Forbidden, "-"),
    exact("MIN_DIR_LOOKUP", Forbidden, "-"),
    exact("CANON_RULE_L_BRANCH_CAP", Forbidden, "unbounded"),
    exact("CANON_MONOMIAL_CAP", Forbidden, "unbounded"),
    exact("LOOKUP_CACHE_MB", Neutral, "512"),
    exact("CANON_CACHE_MB", Neutral, "256"),
    exact("XPOLY_CANON_CACHE_MB", Neutral, "1024"),
    exact("FROZEN_REF_DIR", Forbidden, "-"),
    prefix("FROZEN_CURATED_", Forbidden),
    // retired control namespaces
    prefix("PROD_", Forbidden),
    prefix("SAT_", Forbidden),
    prefix("DB_QC", Forbidden),
    prefix("BV5_", Forbidden),
    // the driver's own inputs
    exact("TDP_BIN_DIR", Input, "-"),
    exact("TDP_GEN_ALLOW_EMPTY_STORE", Input, "0"),
    // tooling and traces
    exact("COMPRESSION_TRACE", Tooling, "-"),
    exact("COMPRESSION_TRACE_MS", Tooling, "-"),
    exact("CARGO", Tooling, "-"),
    exact("CARGO_TARGET_DIR", Tooling, "-"),
    exact("RUSTC", Tooling, "-"),
    exact("RAYON_NUM_THREADS", Tooling, "-"),
];

/// The registry entry governing `name`: an exact entry, else the longest
/// matching prefix.
pub(crate) fn classify(name: &str) -> Option<&'static EnvEntry> {
    ENV_REGISTRY
        .iter()
        .find(|e| !e.prefix && e.name == name)
        .or_else(|| {
            ENV_REGISTRY
                .iter()
                .filter(|e| e.prefix && name.starts_with(e.name))
                .max_by_key(|e| e.name.len())
        })
}

/// The variables the managed driver passes to every stage binary besides the
/// recipe and driver variables (§11.1 `ALLOW_ENV`).
pub(crate) const ALLOW_ENV: &[&str] = &["PATH", "HOME", "TMPDIR"];

/// `scripts/tdp_env_registry.tsv`: `name<TAB>exact|prefix<TAB>class<TAB>default`.
#[cfg_attr(not(test), allow(dead_code))] // tests keep the TSV in sync
pub(crate) fn registry_tsv() -> String {
    let mut s = String::from(
        "# generated from src/tdp/env_registry.rs; checked by cargo test (env_registry_tsv_in_sync)\n",
    );
    for e in ENV_REGISTRY {
        s.push_str(&format!(
            "{}\t{}\t{}\t{}\n",
            e.name,
            if e.prefix { "prefix" } else { "exact" },
            e.class.name(),
            e.default
        ));
    }
    s
}

/// Inherited variables the managed driver refuses, with the message both
/// entry points print.
pub(crate) fn refused_inherited<I, K>(keys: I) -> Vec<String>
where
    I: IntoIterator<Item = K>,
    K: AsRef<str>,
{
    let mut out: Vec<String> = keys
        .into_iter()
        .filter_map(|k| {
            let k = k.as_ref();
            classify(k)
                .filter(|e| e.class.inherited_is_fatal())
                .map(|_| k.to_string())
        })
        .collect();
    out.sort();
    out.dedup();
    out
}

pub(crate) fn refusal_message(names: &[String]) -> String {
    format!(
        "{} {} not honoured by the managed recipe; unset {} (use flags/config)",
        names.join(", "),
        if names.len() == 1 { "is" } else { "are" },
        if names.len() == 1 { "it" } else { "them" }
    )
}
