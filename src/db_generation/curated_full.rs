//! Lossless curated-database primitives.
//!
//! The historical value representation is a chain of `[u8 byte_len][blob]`
//! records. A composite build store uses `[16-byte function key][blob]` as its
//! RocksDB key and an empty value. That representation gives RocksDB exact,
//! global candidate deduplication without accumulating unbounded hot-key
//! values in memory or relying on a probabilistic digest.

use crate::circuit::CircuitSeq;
use std::collections::HashMap;
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

/// First-use wire relabelling of a gate word; the deterministic serialization
/// dihedral canonicalization compares, and the normal form under which the
/// sieve's shingles are matched.
///
/// Control positions are PRESERVED: a g57 gate `[a, x, y]` fires on
/// `(NOT x) AND y`, so `[a, x, y]` and `[a, y, x]` are different functions and
/// must never be conflated (sorting controls here corrupts the word).
pub fn relabel_word(word: &[[u16; 3]]) -> Vec<[u16; 3]> {
    let mut map: HashMap<u16, u16> = HashMap::new();
    let mut next = 0u16;
    let mut out = Vec::with_capacity(word.len());
    for gate in word {
        let mut mapped = [0u16; 3];
        for (slot, &wire) in mapped.iter_mut().zip(gate.iter()) {
            *slot = *map.entry(wire).or_insert_with(|| {
                let v = next;
                next += 1;
                v
            });
        }
        out.push(mapped);
    }
    out
}

/// Dihedral-orbit canonical representative of a cyclic gate word: the minimum
/// over both directions and all rotations of the relabelled serialization.
/// Rotations and reversals of an identity are identities of the same orbit, so
/// deriving splits from this one word reaches the entire orbit's candidates
/// exactly once.
pub fn dihedral_canonical_word(gates: &[[u16; 3]]) -> Vec<[u16; 3]> {
    let n = gates.len();
    if n == 0 {
        return Vec::new();
    }
    let mut best: Option<Vec<[u16; 3]>> = None;
    let mut reversed: Vec<[u16; 3]> = gates.to_vec();
    reversed.reverse();
    for seq in [gates, reversed.as_slice()] {
        let mut rotated = Vec::with_capacity(n);
        for start in 0..n {
            rotated.clear();
            rotated.extend_from_slice(&seq[start..]);
            rotated.extend_from_slice(&seq[..start]);
            let form = relabel_word(&rotated);
            if best.as_ref().is_none_or(|b| form < *b) {
                best = Some(form);
            }
        }
    }
    best.unwrap()
}

/// Stable byte serialization of a gate word, for orbit-hashing.
pub fn word_bytes(word: &[[u16; 3]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(word.len() * 6);
    for gate in word {
        for &wire in gate {
            out.extend_from_slice(&wire.to_le_bytes());
        }
    }
    out
}

fn map_wire(
    wire: u16,
    used_map: &HashMap<u16, u16>,
    extra_map: &mut HashMap<u16, u16>,
    next_extra: &mut u16,
) -> u16 {
    if let Some(&mapped) = used_map.get(&wire) {
        mapped
    } else if let Some(&mapped) = extra_map.get(&wire) {
        mapped
    } else {
        let mapped = *next_extra;
        *next_extra = next_extra
            .checked_add(1)
            .expect("u16 wire space exhausted while deriving curated circuit");
        extra_map.insert(wire, mapped);
        mapped
    }
}

fn canonical_key(circuit: &CircuitSeq) -> Result<[u8; FUNCTION_KEY_BYTES], CuratedError> {
    let (key, _, used) = circuit.canonicalize_polys_single_hashed(false);
    key.ok_or(CuratedError::CanonicalizationSkipped {
        gates: circuit.gates.len(),
        wires: used.len(),
    })
}

fn validate_and_emit<F>(
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

/// Emit every prefix/reversed-suffix friend produced by every direction,
/// rotation, and split of one already-tested minimal identity.
///
/// Unlike the shortcut builder, this function has no source-friend cap, no
/// value-byte cap, and no half-split pruning. The composite output store is
/// responsible for exact global deduplication.
pub fn derive_identity_candidates<F>(identity: &CircuitSeq, emit: F) -> Result<u64, CuratedError>
where
    F: FnMut([u8; FUNCTION_KEY_BYTES], Vec<u8>) -> Result<(), CuratedError>,
{
    derive_identity_candidates_where(identity, |_, _, _| true, emit)
}

/// As [`derive_identity_candidates`], but `accept(reverse, rotation_start,
/// split)` decides which splits are emitted.
///
/// A split is the unit that matters for replacement quality: it produces the
/// pair `(prefix, reversed suffix)` that becomes two entries under one key. A
/// caller can therefore keep only the splits whose halves are worth storing --
/// e.g. neither half containing a locally compressible window -- without
/// discarding the whole identity, which is what makes an identity-level
/// quality filter viable at all.
pub fn derive_identity_candidates_where<A, F>(
    identity: &CircuitSeq,
    mut accept: A,
    mut emit: F,
) -> Result<u64, CuratedError>
where
    A: FnMut(bool, usize, usize) -> bool,
    F: FnMut([u8; FUNCTION_KEY_BYTES], Vec<u8>) -> Result<(), CuratedError>,
{
    let n = identity.gates.len();
    if n < 2 {
        return Ok(0);
    }
    let mut emitted = 0u64;
    for reverse in [false, true] {
        let directed: Vec<[u16; 3]> = if reverse {
            identity.gates.iter().rev().copied().collect()
        } else {
            identity.gates.clone()
        };
        for rotation_start in 0..n {
            let rotation: Vec<[u16; 3]> = directed[rotation_start..]
                .iter()
                .chain(&directed[..rotation_start])
                .copied()
                .collect();

            for split in 1..n {
                if !accept(reverse, rotation_start, split) {
                    continue;
                }
                let prefix = CircuitSeq {
                    gates: rotation[..split].to_vec(),
                };
                let (key, permutation, used) = prefix.canonicalize_polys_single_hashed(false);
                let key = key.ok_or(CuratedError::CanonicalizationSkipped {
                    gates: prefix.gates.len(),
                    wires: used.len(),
                })?;
                let inverse = permutation.invert();
                let used_map: HashMap<u16, u16> = used
                    .iter()
                    .enumerate()
                    .map(|(dense, &original)| (original, inverse.data[dense] as u16))
                    .collect();

                let mut prefix_db = CircuitSeq {
                    gates: rotation[..split]
                        .iter()
                        .map(|&[target, control_a, control_b]| {
                            [
                                used_map[&target],
                                used_map[&control_a],
                                used_map[&control_b],
                            ]
                        })
                        .collect(),
                };
                prefix_db.canonicalize();

                let mut extra_map = HashMap::new();
                let mut next_extra = used.len() as u16;
                let mut tail_db = CircuitSeq {
                    gates: rotation[split..]
                        .iter()
                        .rev()
                        .map(|&[target, control_a, control_b]| {
                            [
                                map_wire(target, &used_map, &mut extra_map, &mut next_extra),
                                map_wire(control_a, &used_map, &mut extra_map, &mut next_extra),
                                map_wire(control_b, &used_map, &mut extra_map, &mut next_extra),
                            ]
                        })
                        .collect(),
                };
                tail_db.canonicalize();

                validate_and_emit(key, &prefix_db, &mut emit)?;
                validate_and_emit(key, &tail_db, &mut emit)?;
                emitted += 2;
            }
        }
    }
    Ok(emitted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    #[test]
    fn malformed_legacy_value_is_not_partially_accepted() {
        let err = decode_legacy_value(&[3, 0, 1]).unwrap_err();
        assert!(matches!(err, CuratedError::MalformedValue { .. }));
    }

    #[test]
    fn checked_blob_rejects_each_legacy_format_overflow() {
        let too_many_gates = CircuitSeq {
            gates: vec![[0, 1, 2]; 86],
        };
        assert!(matches!(
            checked_blob(&too_many_gates),
            Err(CuratedError::CircuitTooLong { gates: 86, .. })
        ));
        let wide_wire = CircuitSeq {
            gates: vec![[256, 1, 2]],
        };
        assert_eq!(
            checked_blob(&wide_wire),
            Err(CuratedError::WireTooLarge { wire: 256 })
        );
    }

    #[test]
    fn composite_records_preserve_more_than_all_historical_caps() {
        let key = [0x5au8; FUNCTION_KEY_BYTES];
        let mut records = BTreeSet::new();
        for i in 0..300u16 {
            let blob = vec![
                (i >> 8) as u8,
                i as u8,
                1,
                9,
                (i.wrapping_mul(17) >> 8) as u8,
                i.wrapping_mul(17) as u8,
            ];
            records.insert(composite_key(&key, &blob).unwrap());
        }
        assert_eq!(records.len(), 300);

        let mut grouped: BTreeMap<[u8; FUNCTION_KEY_BYTES], Vec<u8>> = BTreeMap::new();
        for record in records {
            let (record_key, blob) = split_composite_key(&record).unwrap();
            grouped
                .entry(record_key)
                .or_default()
                .extend(encode_legacy_record(blob).unwrap());
        }
        let value = &grouped[&key];
        assert_eq!(decode_legacy_value(value).unwrap().len(), 300);
        assert!(value.len() > 512);
    }

    #[test]
    fn identity_split_candidates_rehash_to_their_emitted_keys() {
        // A B A B is an identity because A and B act on disjoint wires and
        // therefore commute; neither adjacent pair is equal in this spelling.
        let identity = CircuitSeq {
            gates: vec![[0, 1, 2], [3, 4, 5], [0, 1, 2], [3, 4, 5]],
        };
        let mut records = BTreeSet::new();
        let emitted = derive_identity_candidates(&identity, |key, blob| {
            records.insert(composite_key(&key, &blob)?);
            Ok(())
        })
        .unwrap();
        assert_eq!(emitted, 2 * 2 * 4 * 3);
        assert!(!records.is_empty());
        for record in records {
            let (key, blob) = split_composite_key(&record).unwrap();
            assert_eq!(canonical_key(&CircuitSeq::from_blob(blob)).unwrap(), key);
        }
    }
}
