//! Canonical-key validation shared by database import and candidate generation.
use super::codec::{CuratedError, FUNCTION_KEY_BYTES, checked_blob};
use crate::circuit::CircuitSeq;
pub fn canonical_key(circuit: &CircuitSeq) -> Result<[u8; FUNCTION_KEY_BYTES], CuratedError> {
    let (key, _, used) = circuit.canonicalize_polys_single_hashed(false);
    key.ok_or(CuratedError::CanonicalizationSkipped {
        gates: circuit.gates.len(),
        wires: used.len(),
    })
}

pub fn validate_and_emit<F>(
    expected: [u8; FUNCTION_KEY_BYTES],
    circuit: &CircuitSeq,
    emit: &mut F,
) -> Result<(), CuratedError>
where
    F: FnMut([u8; FUNCTION_KEY_BYTES], Vec<u8>) -> Result<(), CuratedError>,
{
    let actual = canonical_key(circuit)?;
    if actual != expected {
        return Err(CuratedError::EquivalenceMismatch { expected, actual });
    }
    emit(expected, checked_blob(circuit)?)
}
