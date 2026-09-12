//! Lossless curated-database primitives.
//!
//! The historical value representation is a chain of `[u8 byte_len][blob]`
//! records. A composite build store uses `[16-byte function key][blob]` as its
//! RocksDB key and an empty value. That representation gives RocksDB exact,
//! global candidate deduplication without accumulating unbounded hot-key
//! values in memory or relying on a probabilistic digest.

use crate::circuit::CircuitSeq;
use std::error::Error;
use std::fmt;

pub const FUNCTION_KEY_BYTES: usize = 16;
pub const MAX_LEGACY_BLOB_BYTES: usize = u8::MAX as usize;
// A valid composite key is always longer than 16 bytes, so this one-byte
// metadata key cannot collide with any `(function, circuit)` record.
pub const COMPOSITE_FORMAT_MARKER: &[u8] = b"\0";
pub const COMPOSITE_COMPLETE_MARKER: &[u8] = b"\0complete";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CuratedError {
    MalformedValue {
        offset: usize,
        value_len: usize,
    },
    EmptyCompositeKey,
    InvalidCompositeKey {
        len: usize,
    },
    EmptyCircuit,
    CircuitTooLong {
        gates: usize,
        bytes: usize,
    },
    WireTooLarge {
        wire: u16,
    },
    CanonicalizationSkipped {
        gates: usize,
        wires: usize,
    },
    EquivalenceMismatch {
        expected: [u8; FUNCTION_KEY_BYTES],
        actual: [u8; FUNCTION_KEY_BYTES],
    },
}

impl fmt::Display for CuratedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MalformedValue { offset, value_len } => write!(
                f,
                "malformed curated value at byte {offset} (value length {value_len})"
            ),
            Self::EmptyCompositeKey => write!(f, "composite record is the format marker"),
            Self::InvalidCompositeKey { len } => write!(
                f,
                "composite record has {len} bytes; expected a 16-byte key and a nonempty circuit"
            ),
            Self::EmptyCircuit => write!(f, "empty circuits cannot be stored in a curated value"),
            Self::CircuitTooLong { gates, bytes } => write!(
                f,
                "{gates}-gate circuit needs {bytes} bytes, beyond the legacy 255-byte record limit"
            ),
            Self::WireTooLarge { wire } => write!(
                f,
                "wire {wire} cannot be represented by the legacy one-byte wire format"
            ),
            Self::CanonicalizationSkipped { gates, wires } => write!(
                f,
                "canonicalization skipped a {gates}-gate/{wires}-wire circuit; a full build cannot continue"
            ),
            Self::EquivalenceMismatch { expected, actual } => write!(
                f,
                "derived candidate failed equivalence validation: expected {}, got {}",
                hex_key(expected),
                hex_key(actual)
            ),
        }
    }
}

impl Error for CuratedError {}

fn hex_key(key: &[u8; FUNCTION_KEY_BYTES]) -> String {
    let mut text = String::with_capacity(FUNCTION_KEY_BYTES * 2);
    for byte in key {
        use fmt::Write as _;
        let _ = write!(text, "{byte:02x}");
    }
    text
}

pub struct LegacyValueIter<'a> {
    value: &'a [u8],
    pos: usize,
    finished: bool,
}

impl<'a> Iterator for LegacyValueIter<'a> {
    type Item = Result<&'a [u8], CuratedError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished || self.pos == self.value.len() {
            return None;
        }
        let record_offset = self.pos;
        let len = self.value[self.pos] as usize;
        self.pos += 1;
        if len == 0 || self.pos + len > self.value.len() || len % 3 != 0 {
            self.finished = true;
            return Some(Err(CuratedError::MalformedValue {
                offset: record_offset,
                value_len: self.value.len(),
            }));
        }
        let blob = &self.value[self.pos..self.pos + len];
        self.pos += len;
        Some(Ok(blob))
    }
}

/// Iterate every circuit in a legacy value without allocating a candidate
/// list. Truncated records are errors rather than the old best-effort behavior
/// that silently discarded the tail.
pub fn legacy_value_blobs(value: &[u8]) -> LegacyValueIter<'_> {
    LegacyValueIter {
        value,
        pos: 0,
        finished: false,
    }
}

pub fn decode_legacy_value(value: &[u8]) -> Result<Vec<Vec<u8>>, CuratedError> {
    legacy_value_blobs(value)
        .map(|blob| blob.map(ToOwned::to_owned))
        .collect()
}

/// Checked serialization for the legacy circuit record format.
pub fn checked_blob(circuit: &CircuitSeq) -> Result<Vec<u8>, CuratedError> {
    if circuit.gates.is_empty() {
        return Err(CuratedError::EmptyCircuit);
    }
    let bytes = circuit.gates.len().checked_mul(3).unwrap_or(usize::MAX);
    if bytes > MAX_LEGACY_BLOB_BYTES {
        return Err(CuratedError::CircuitTooLong {
            gates: circuit.gates.len(),
            bytes,
        });
    }
    let mut blob = Vec::with_capacity(bytes);
    for gate in &circuit.gates {
        for &wire in gate {
            let wire = u8::try_from(wire).map_err(|_| CuratedError::WireTooLarge { wire })?;
            blob.push(wire);
        }
    }
    Ok(blob)
}

pub fn encode_legacy_record(blob: &[u8]) -> Result<Vec<u8>, CuratedError> {
    if blob.is_empty() || blob.len() % 3 != 0 {
        return Err(CuratedError::MalformedValue {
            offset: 0,
            value_len: blob.len(),
        });
    }
    let len = u8::try_from(blob.len()).map_err(|_| CuratedError::CircuitTooLong {
        gates: blob.len() / 3,
        bytes: blob.len(),
    })?;
    let mut encoded = Vec::with_capacity(blob.len() + 1);
    encoded.push(len);
    encoded.extend_from_slice(blob);
    Ok(encoded)
}

/// Exact-dedup build key. Identical `(function, circuit)` pairs become the
/// same RocksDB key; distinct circuits can never disappear through a hash
/// collision in an auxiliary dedup set.
pub fn composite_key(key: &[u8; FUNCTION_KEY_BYTES], blob: &[u8]) -> Result<Vec<u8>, CuratedError> {
    encode_legacy_record(blob)?;
    let mut composite = Vec::with_capacity(FUNCTION_KEY_BYTES + blob.len());
    composite.extend_from_slice(key);
    composite.extend_from_slice(blob);
    Ok(composite)
}

pub fn split_composite_key(
    composite: &[u8],
) -> Result<([u8; FUNCTION_KEY_BYTES], &[u8]), CuratedError> {
    if composite == COMPOSITE_FORMAT_MARKER {
        return Err(CuratedError::EmptyCompositeKey);
    }
    if composite.len() <= FUNCTION_KEY_BYTES || (composite.len() - FUNCTION_KEY_BYTES) % 3 != 0 {
        return Err(CuratedError::InvalidCompositeKey {
            len: composite.len(),
        });
    }
    let mut key = [0u8; FUNCTION_KEY_BYTES];
    key.copy_from_slice(&composite[..FUNCTION_KEY_BYTES]);
    Ok((key, &composite[FUNCTION_KEY_BYTES..]))
}
