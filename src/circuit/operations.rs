//! File-based circuit operations shared by commands and other library callers.
//! G57 evaluation uses CircuitSeq's existing kernels; generalized and packed
//! files use XGate evaluation. No command-line parsing or printing belongs here.
use super::{CircuitSeq, U1024, XGate, eval_limbs, max_wire};
use crate::circuit::formats as format;
use primitive_types::U256;
use rand::RngCore;
use std::fs;

/// Little-endian state covering up to 1024 wires, one bit per wire.
pub type CircuitState = [u64; 16];

enum Representation {
    G57(CircuitSeq),
    General(Vec<XGate>),
}

/// A loaded circuit with its declared and actually used wire width accounted for.
pub struct CircuitSource {
    representation: Representation,
    wires: usize,
}

/// Outcome of comparing all wires of two circuits on sampled inputs.
/// A missing counterexample is a sampled result, not a proof of equivalence.
pub struct SampledComparison {
    pub wires: usize,
    pub counterexample: Option<Counterexample>,
}

/// The first sampled input producing different outputs.
pub struct Counterexample {
    /// One-based sample number.
    pub sample: usize,
    pub input: CircuitState,
}

impl CircuitSource {
    /// Load G57, mpmct1, esop1 or anf1 using their existing readers.
    /// Files requiring more than 1024 wires are rejected.
    pub fn read(path: &str) -> Result<Self, String> {
        let raw = fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
        let first = raw
            .iter()
            .position(|b| !b.is_ascii_whitespace())
            .unwrap_or(raw.len());
        let bytes = &raw[first..];
        let (representation, declared) = if [b"mpmct1".as_slice(), b"esop1", b"anf1"]
            .iter()
            .any(|header| bytes.starts_with(header))
        {
            let (gates, wires) =
                format::read_mpmct(path).map_err(|e| format!("cannot read {path}: {e}"))?;
            (Representation::General(gates), wires as usize)
        } else {
            (Representation::G57(CircuitSeq::from_bytes(&raw)), 0)
        };
        let touched = match &representation {
            Representation::G57(c) => {
                if c.gates.is_empty() {
                    0
                } else {
                    c.max_wire() + 1
                }
            }
            Representation::General(g) => {
                if g.is_empty() {
                    0
                } else {
                    max_wire(g.iter()) as usize + 1
                }
            }
        };
        let wires = declared.max(touched);
        if wires > 1024 {
            return Err(format!(
                "{path} needs {wires} wires; circuit utilities support at most 1024"
            ));
        }
        Ok(Self {
            representation,
            wires,
        })
    }

    /// Maximum of the declared width and highest touched wire plus one.
    pub fn wire_count(&self) -> usize {
        self.wires
    }

    /// Evaluate in place, preserving every untouched bit of the supplied state.
    /// The loaded circuit determines the evaluator width, including auxiliary wires.
    pub fn evaluate(&self, state: &mut CircuitState) {
        match &self.representation {
            Representation::G57(c) => evaluate_g57(c, state, self.wires),
            Representation::General(g) => eval_limbs(g.iter(), state),
        }
    }

    /// Compare complete circuit functions on `samples` random inputs.
    ///
    /// Samples cover max(minimum_wires, both circuits' widths), and all output
    /// wires are compared. This deliberately differs from
    /// CircuitSeq::probably_equal, whose extra input wires start at zero and
    /// whose output comparison covers only the requested low wires.
    /// `minimum_wires` must be 1..=1024 and `samples` must be positive.
    pub fn compare_sampled(
        &self,
        other: &Self,
        minimum_wires: usize,
        samples: usize,
        rng: &mut impl RngCore,
    ) -> Result<SampledComparison, String> {
        if !(1..=1024).contains(&minimum_wires) {
            return Err("wires must be in 1..=1024".into());
        }
        if samples == 0 {
            return Err("comparison requires at least one iteration".into());
        }
        let wires = minimum_wires.max(self.wire_count()).max(other.wire_count());
        for sample in 1..=samples {
            let input = random_state(wires, rng);
            let (mut left, mut right) = (input, input);
            self.evaluate(&mut left);
            other.evaluate(&mut right);
            if left != right {
                return Ok(SampledComparison {
                    wires,
                    counterexample: Some(Counterexample { sample, input }),
                });
            }
        }
        Ok(SampledComparison {
            wires,
            counterexample: None,
        })
    }
}

fn evaluate_g57(circuit: &CircuitSeq, state: &mut CircuitState, wires: usize) {
    if wires <= 64 {
        state[0] = circuit.evaluate_64(state[0]);
    } else if wires <= 128 {
        let out = circuit.evaluate_128((state[0] as u128) | ((state[1] as u128) << 64));
        state[0] = out as u64;
        state[1] = (out >> 64) as u64;
    } else if wires <= 256 {
        let mut value = U256::zero();
        value.0.copy_from_slice(&state[..4]);
        state[..4].copy_from_slice(&circuit.evaluate_256(value).0);
    } else {
        let output = circuit.evaluate_1024(U1024(*state));
        state.copy_from_slice(&output.0);
    }
}

/// Draw a state and zero all bits at or above `wires` (at most 1024).
pub fn random_state(wires: usize, rng: &mut impl RngCore) -> CircuitState {
    let mut state = [0; 16];
    for limb in &mut state {
        *limb = rng.next_u64();
    }
    mask_state(&mut state, wires);
    state
}

/// Zero all bits at or above `wires`. Panics if `wires` exceeds 1024.
pub fn mask_state(state: &mut CircuitState, wires: usize) {
    assert!(wires <= 1024, "state supports at most 1024 wires");
    for (i, limb) in state.iter_mut().enumerate() {
        let low = i * 64;
        *limb &= if wires >= low + 64 {
            u64::MAX
        } else if wires <= low {
            0
        } else {
            (1u64 << (wires - low)) - 1
        };
    }
}

#[cfg(test)]
#[path = "../../tests/unit/circuit/operations.rs"]
mod tests;
