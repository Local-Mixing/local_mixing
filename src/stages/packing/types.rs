//! Stage-4 parameter enums, errors and exit codes (docs/STAGE4_RUST_DESIGN.md §4.1, §5.4).
use std::fmt;
use std::io;
use std::path::PathBuf;

/// How a monomial-product cap decides (§7.2).
///
/// `CPython39` reproduces the oracle: `mul` aborts as soon as a PREFIX of the
/// smaller operand's CPython 3.9 iteration order pushes the running product
/// above the cap. `Canonical` completes the product and aborts only when the
/// finished product exceeds the cap; it is an explicit deviation, never the
/// default.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CapSemantics {
    #[default]
    CPython39,
    Canonical,
}

impl CapSemantics {
    pub fn parse(s: &str) -> Option<CapSemantics> {
        match s {
            "cpython39" => Some(CapSemantics::CPython39),
            "canonical" => Some(CapSemantics::Canonical),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            CapSemantics::CPython39 => "cpython39",
            CapSemantics::Canonical => "canonical",
        }
    }
}

/// Packer parameters (docs/STAGE4_RUST_DESIGN.md §4.1). `PRODUCTION` is the
/// recommended configuration (`window_pack3.py A B 512 512 24576 3000`);
/// `ORACLE_DEFAULTS` are `window_pack3.py`'s own argv defaults.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackParams {
    /// Interval cap (argv 3).
    pub w_cap: usize,
    /// Soft window gate cap G (argv 4; hard 2G inside a burst).
    pub g: usize,
    /// Soft window monomial budget (argv 5; hard 2·b_mono in a burst);
    /// the product cap is 8·b_mono.
    pub b_mono: usize,
    /// Longest span considered (argv 6).
    pub span_cap: usize,
    /// Pre-emptive close slack (WP const PAD).
    pub pad: usize,
    /// Contagion threshold (WP const CMAX).
    pub cmax: usize,
    /// Fingerprint seed (WP const SEED).
    pub fp_seed: u64,
    pub cap_semantics: CapSemantics,
}

impl PackParams {
    pub const PRODUCTION: PackParams = PackParams {
        w_cap: 512,
        g: 512,
        b_mono: 24576,
        span_cap: 3000,
        pad: 4,
        cmax: 24,
        fp_seed: 20_261_004,
        cap_semantics: CapSemantics::CPython39,
    };

    pub const ORACLE_DEFAULTS: PackParams = PackParams {
        w_cap: 64,
        g: 64,
        b_mono: 3072,
        span_cap: 3000,
        ..PackParams::PRODUCTION
    };

    pub fn mul_cap(&self) -> usize {
        self.b_mono.saturating_mul(8)
    }

    /// Bounds shared with the managed driver (`[packing]`,
    /// src/tdp/config/validate.rs): the scheduler's caps (`2·G`, `2·b_mono`,
    /// `8·b_mono`, `G − load`) are plain usize/i64 arithmetic, which wraps in
    /// release builds beyond these.
    pub fn validate(&self) -> Result<(), String> {
        for (v, name, max) in [
            (self.g, "--window-gates", MAX_WINDOW_GATES),
            (self.w_cap, "--interval-cap", MAX_WINDOW_GATES),
            (self.b_mono, "--monomial-budget", MAX_MONOMIAL_BUDGET),
            (self.span_cap, "--span-cap", MAX_SPAN_CAP),
        ] {
            if v == 0 || v > max {
                return Err(format!("{name} {v}: must be in 1..={max}"));
            }
        }
        Ok(())
    }
}

/// Widest state stage 4 accepts: the monomial width tops out at 16 words
/// (`mono::with_mono_width`), so the managed driver refuses wider plans up front.
pub const MAX_STATE_WIRES: usize = 1024;

/// Largest window gate cap / interval cap accepted (the driver's bound).
pub const MAX_WINDOW_GATES: usize = 1_000_000;
/// Largest monomial budget and split budget accepted. These are RESOURCE
/// bounds: a window's ANF may hold up to 4·budget monomials and a product
/// 8·budget, so memory grows with the budget (budgets of 1e6 to 1e9 reached
/// 5-9 GB on the n=8 fixture within two minutes; at 2^20 the packer still
/// held 2-3 GB at the narrowest monomial width; at 2^18 it peaked at 1.2 GB
/// and finished, with window gates and interval cap 1e6). 2^18 = 262,144 is
/// 5x the largest measured packer budget (49,152 at G=1024) and 4x the
/// recommended split budget 65,536. The deliverable's emission uses the same
/// ceiling (`serialize::EMIT_BUDGET`), so every window that a split at an
/// accepted budget leaves fitting also emits.
pub const MAX_MONOMIAL_BUDGET: usize = 1 << 18;
/// Largest span cap accepted (a gate count, not a monomial count).
pub const MAX_SPAN_CAP: usize = 1_000_000_000;
/// Largest fix / fallback iteration count accepted.
pub const MAX_ITERATIONS: usize = 1000;

/// The repair algorithm. Later improvements are new variants behind flags,
/// never the default until RC re-validates (§13.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RepairMode {
    /// `repair.sh` exactly: cancel, fix x4, merge/split fallback x4.
    #[default]
    Oracle,
}

impl RepairMode {
    pub fn name(self) -> &'static str {
        match self {
            RepairMode::Oracle => "oracle",
        }
    }
}

/// Repair parameters (docs/STAGE4_RUST_DESIGN.md §4.1). The defaults are the
/// recommended configuration (`repair.sh`): split budget 65536, 4 fix and 4
/// fallback iterations, C-view 96 batches with seed 12345, CPython 3.9 caps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepairParams {
    pub split_budget: usize,
    pub fix_iterations: usize,
    pub fallback_iterations: usize,
    pub cview: crate::stages::packing::cview::CviewParams,
    pub cap_semantics: CapSemantics,
    pub mode: RepairMode,
}

impl RepairParams {
    /// The driver's bounds (see [`PackParams::validate`]).
    pub fn validate(&self) -> Result<(), String> {
        if self.split_budget == 0 || self.split_budget > MAX_MONOMIAL_BUDGET {
            return Err(format!(
                "--split-budget {}: must be in 1..={MAX_MONOMIAL_BUDGET}",
                self.split_budget
            ));
        }
        for (v, name) in [
            (self.fix_iterations, "--fix-iterations"),
            (self.fallback_iterations, "--fallback-iterations"),
        ] {
            if v > MAX_ITERATIONS {
                return Err(format!("{name} {v}: must be in 0..={MAX_ITERATIONS}"));
            }
        }
        Ok(())
    }
}

impl Default for RepairParams {
    fn default() -> Self {
        RepairParams {
            split_budget: 65536,
            fix_iterations: 4,
            fallback_iterations: 4,
            cview: Default::default(),
            cap_semantics: CapSemantics::CPython39,
            mode: RepairMode::Oracle,
        }
    }
}

/// Every stage-4 failure, mapped one-to-one onto the binary's exit codes.
#[derive(Debug)]
pub enum Stage4Error {
    /// Exit 2: usage or parameter error.
    Usage(String),
    /// Exit 3: unreadable, malformed or inconsistent input.
    Input(String),
    /// Exit 4: a certificate or equivalence check failed.
    Verify(String),
    /// Exit 5: the final scan has a visible non-output row.
    Floor { nonoutput_visible: Vec<usize> },
    /// Exit 6: some final window fails the split's budget test.
    OverBudget { windows: Vec<(usize, usize)> },
    /// Exit 7: writing an output failed.
    Io { path: PathBuf, source: io::Error },
    /// Exit 8: an internal invariant was violated without panicking.
    Internal(String),
}

/// (exit code, meaning): the single table printed by the binary and the driver.
pub const EXIT_CODES: [(i32, &str); 9] = [
    (0, "finished"),
    (2, "usage or parameter error"),
    (3, "input error"),
    (4, "verification failure"),
    (5, "exposure floor missed"),
    (6, "over budget"),
    (7, "I/O error writing outputs"),
    (8, "internal invariant violated"),
    (101, "panic"),
];

impl Stage4Error {
    pub fn exit_code(&self) -> i32 {
        match self {
            Stage4Error::Usage(_) => 2,
            Stage4Error::Input(_) => 3,
            Stage4Error::Verify(_) => 4,
            Stage4Error::Floor { .. } => 5,
            Stage4Error::OverBudget { .. } => 6,
            Stage4Error::Io { .. } => 7,
            Stage4Error::Internal(_) => 8,
        }
    }

    pub fn meaning(code: i32) -> &'static str {
        EXIT_CODES
            .iter()
            .find(|(c, _)| *c == code)
            .map(|(_, m)| *m)
            .unwrap_or("unknown exit code")
    }

    pub fn io(path: impl Into<PathBuf>, source: io::Error) -> Stage4Error {
        Stage4Error::Io {
            path: path.into(),
            source,
        }
    }
}

impl fmt::Display for Stage4Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Stage4Error::Usage(m) => write!(f, "usage: {m}"),
            Stage4Error::Input(m) => write!(f, "input: {m}"),
            Stage4Error::Verify(m) => write!(f, "verification failed: {m}"),
            Stage4Error::Floor { nonoutput_visible } => write!(
                f,
                "exposure floor missed: {} visible non-output rows",
                nonoutput_visible.len()
            ),
            Stage4Error::OverBudget { windows } => {
                write!(f, "over budget: {} windows", windows.len())?;
                if let Some((lo, hi)) = windows.first() {
                    write!(f, " (first [{lo}, {hi}))")?;
                }
                Ok(())
            }
            Stage4Error::Io { path, source } => {
                write!(f, "I/O error on {}: {source}", path.display())
            }
            Stage4Error::Internal(m) => write!(f, "internal error: {m}"),
        }
    }
}

impl std::error::Error for Stage4Error {}

#[cfg(test)]
#[path = "../../../tests/unit/stages/packing/types.rs"]
mod tests;
