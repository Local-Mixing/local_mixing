//! Curated candidate generation; storage encodings live in database::codec.
use crate::circuit::CircuitSeq;
pub use crate::database::codec::*;
#[cfg(test)]
use crate::database::validation::canonical_key;
use crate::database::validation::validate_and_emit;
use std::collections::HashMap;
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
#[path = "../tests/db_gen/curated_full/tests.rs"]
mod tests;
