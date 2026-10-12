//! Stage-7 parameters, errors and exit codes (docs/STAGE7_UNPACKING.md).
use super::synth::SynthParams;
use std::fmt;
use std::io;
use std::path::PathBuf;

/// Everything `unpack` needs besides its inputs.
#[derive(Clone, Debug, PartialEq)]
pub struct UnpackParams {
    pub synth: SynthParams,
    /// Run the pairwise peephole pass on every synthesized window.
    pub postopt: bool,
    /// Emit the packed circuit's window verbatim when synthesis fails
    /// (needs the packed circuit); otherwise any failed window is an error.
    pub allow_fallback: bool,
    /// Tie-break seed of the synthesizer (never reported).
    pub seed: u64,
    /// Outside wires offered to the synthesizer as ancilla candidates per
    /// window (`spec::LocalUniverse`).
    pub ancilla_pool: usize,
    /// Seed of the sampled checks (reported; nothing secret).
    pub verify_seed: u64,
    /// Batches of 1024 random states for the sampled whole-circuit checks.
    pub equiv_batches: usize,
    /// Samples of the zero-slice check (rounded up to 1024s).
    pub zero_slice_samples: usize,
}

/// Outside wires offered as ancilla candidates per window.
pub const DEFAULT_ANCILLA_POOL: usize = 32;
/// Seed of every sampled check (public, fixed).
pub const DEFAULT_VERIFY_SEED: u64 = 20_261_011;
pub const DEFAULT_EQUIV_BATCHES: usize = 4;
pub const DEFAULT_ZERO_SLICE_SAMPLES: usize = 4096;

impl Default for UnpackParams {
    fn default() -> Self {
        UnpackParams {
            synth: SynthParams::default(),
            postopt: true,
            allow_fallback: true,
            seed: 1,
            ancilla_pool: DEFAULT_ANCILLA_POOL,
            verify_seed: DEFAULT_VERIFY_SEED,
            equiv_batches: DEFAULT_EQUIV_BATCHES,
            zero_slice_samples: DEFAULT_ZERO_SLICE_SAMPLES,
        }
    }
}

impl UnpackParams {
    pub fn validate(&self) -> Result<(), String> {
        self.synth.validate()?;
        if self.ancilla_pool > 1024 {
            return Err(format!("ancilla pool {}: at most 1024", self.ancilla_pool));
        }
        if self.equiv_batches == 0 || self.equiv_batches > 1024 {
            return Err(format!(
                "equivalence batches {}: must be in 1..=1024",
                self.equiv_batches
            ));
        }
        Ok(())
    }
}

/// Widest state stage 7 accepts (the monomial width of the exact window
/// check tops out at 16 words, as in stage 4).
pub const MAX_STATE_WIRES: usize = 1024;

#[derive(Debug)]
pub enum Stage7Error {
    /// Exit 2: usage or parameter error.
    Usage(String),
    /// Exit 3: unreadable, malformed or inconsistent input.
    Input(String),
    /// Exit 4: a window could not be synthesized (no fallback allowed) or a
    /// check failed.
    Verify(String),
    /// Exit 7: writing an output failed.
    Io { path: PathBuf, source: io::Error },
    /// Exit 8: an internal invariant was violated without panicking.
    Internal(String),
}

/// (exit code, meaning): the single table printed by the binary.
pub const EXIT_CODES: [(i32, &str); 7] = [
    (0, "finished"),
    (2, "usage or parameter error"),
    (3, "input error"),
    (4, "verification or synthesis failure"),
    (7, "I/O error writing outputs"),
    (8, "internal invariant violated"),
    (101, "panic"),
];

impl Stage7Error {
    pub fn exit_code(&self) -> i32 {
        match self {
            Stage7Error::Usage(_) => 2,
            Stage7Error::Input(_) => 3,
            Stage7Error::Verify(_) => 4,
            Stage7Error::Io { .. } => 7,
            Stage7Error::Internal(_) => 8,
        }
    }

    pub fn meaning(code: i32) -> &'static str {
        EXIT_CODES
            .iter()
            .find(|(c, _)| *c == code)
            .map(|(_, m)| *m)
            .unwrap_or("unknown exit code")
    }

    pub fn io(path: impl Into<PathBuf>, source: io::Error) -> Stage7Error {
        Stage7Error::Io {
            path: path.into(),
            source,
        }
    }
}

impl fmt::Display for Stage7Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Stage7Error::Usage(m) => write!(f, "usage: {m}"),
            Stage7Error::Input(m) => write!(f, "input: {m}"),
            Stage7Error::Verify(m) => write!(f, "verification failed: {m}"),
            Stage7Error::Io { path, source } => {
                write!(f, "I/O error on {}: {source}", path.display())
            }
            Stage7Error::Internal(m) => write!(f, "internal error: {m}"),
        }
    }
}

impl std::error::Error for Stage7Error {}

/// Stage-4 errors raised by the shared decoder / composer, mapped onto the
/// stage-7 codes (`Floor` and `OverBudget` cannot occur on this path).
impl From<crate::stages::packing::types::Stage4Error> for Stage7Error {
    fn from(e: crate::stages::packing::types::Stage4Error) -> Self {
        use crate::stages::packing::types::Stage4Error as E;
        match e {
            E::Usage(m) => Stage7Error::Usage(m),
            E::Input(m) => Stage7Error::Input(m),
            E::Verify(m) => Stage7Error::Verify(m),
            E::Io { path, source } => Stage7Error::Io { path, source },
            E::Internal(m) => Stage7Error::Internal(m),
            other => Stage7Error::Internal(format!("unexpected stage-4 error: {other}")),
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/unpacking/types.rs"]
mod tests;
