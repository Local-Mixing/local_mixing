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
}

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
                write!(f, "over budget: {} windows", windows.len())
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
