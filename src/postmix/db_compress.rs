//! Frozen-store compression for heterogeneous [`XGate`] tapes.
//!
//! The legacy compressor is deeply coupled to `CircuitSeq<[u16; 3]>`.  This
//! module keeps the mixed tape as XGates, samples only contiguous windows (so
//! no g57-specific convexity or collision code is needed), keys each window by
//! its exact heterogeneous polynomial function, and maps shorter g57 friends
//! from the regular frozen store back into XGate form.

use super::compress::{RecoveryEvents, TracedCircuit};
use super::lineage::{GroupKind, ProvId, ProvenanceArena};
use super::source::{MIXED_SOURCE, SourceClassCounts, UNKNOWN_SOURCE, merge_sources};
use super::xgate::XGate;
use super::xpoly::{CanonicalXPolys, XPolyBudget, XPolyError, canonicalize_xgates_single};
use crate::circuit::circuit::{CircuitSeq, Permutation, polys_repr_blob};
use crate::replace::frozen::FrozenDb;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;
use xxhash_rust::xxh3::xxh3_128;

#[derive(Clone, Debug)]
pub struct DbCompressParams {
    /// Number of random contiguous windows to try.  The tape shrinks in place,
    /// so subsequent trials see earlier accepted replacements.
    pub trials: usize,
    pub min_window: usize,
    pub max_window: usize,
    /// Probe the inverse (reversed gate order) as well as the forward function.
    pub probe_reverse: bool,
    pub poly_budget: XPolyBudget,
    pub seed: u64,
}

impl Default for DbCompressParams {
    fn default() -> Self {
        Self {
            trials: 10_000,
            min_window: 2,
            max_window: 12,
            probe_reverse: true,
            poly_budget: XPolyBudget::default(),
            seed: 0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DbCompressReport {
    pub gates_in: usize,
    pub gates_out: usize,
    pub lits_in: u64,
    pub lits_out: u64,
    pub trials_requested: usize,
    pub trials_attempted: usize,
    /// Sampled windows containing at least one gate that is not a structural
    /// g57 (`comp=true`, exactly two controls, opposite polarities).
    pub heterogeneous_windows_attempted: u64,
    /// Sampled windows made entirely of structural g57 gates.
    pub all_g57_windows_attempted: u64,
    pub canonical_directions_attempted: u64,
    /// Directional polynomial/canonicalization failures.  These are safe skips,
    /// not circuit errors or partial replacements.
    pub polynomial_skips: u64,
    pub budget_skips: u64,
    /// Windows for which neither forward nor reverse canonicalization succeeded.
    pub windows_without_key: u64,
    pub lookups: u64,
    pub lookup_hits: u64,
    /// Directional frozen-store hits produced by heterogeneous windows.
    pub heterogeneous_lookup_hits: u64,
    /// Directional frozen-store hits produced by all-g57 windows.
    pub all_g57_lookup_hits: u64,
    pub malformed_value_entries: u64,
    pub candidates_decoded: u64,
    pub shorter_candidates: u64,
    pub windows_with_shorter_candidate: u64,
    pub replacements: u64,
    /// Accepted replacements whose sampled input window was heterogeneous.
    pub heterogeneous_replacements: u64,
    /// Accepted replacements whose sampled input window was all structural g57.
    pub all_g57_replacements: u64,
    pub gates_removed_by_db: u64,
    /// Number of gates removed as identical adjacent involution pairs.
    pub adjacent_gates_cancelled: u64,
    /// Structural g57 gates emitted by accepted replacement windows carrying
    /// at least one loose-input fragment in their group-attributed ancestry.
    pub attributed_structural_g57_outputs: u64,
    /// Accepted windows with nonempty loose-fragment ancestry and at least one
    /// structural g57 in the frozen friend.
    pub attributed_g57_replacement_windows: u64,
    /// Source-parent composition of accepted replacement windows.
    pub single_parent_replacement_windows: u64,
    pub mixed_parent_replacement_windows: u64,
    pub unknown_source_replacement_windows: u64,
    /// Exact parent-vs-new classification of structural G57 friends emitted
    /// by the frozen database.
    pub structural_sources: SourceClassCounts,
}

#[derive(Clone, Debug)]
struct DbTrackedGate {
    gate: XGate,
    root: ProvId,
    source: u32,
}

/// Recognize the exact gate shape representable by one legacy g57 triple.
/// Wire ordering is irrelevant here; [`XGate`] construction already enforces
/// that the target is not also a control.
fn is_structural_g57(gate: &XGate) -> bool {
    gate.comp && gate.ctrls.len() == 2 && gate.ctrls[0].1 != gate.ctrls[1].1
}

/// Cancel all identical adjacent involution pairs, including cascades exposed
/// by an earlier cancellation, while dropping their matching trace roots.
fn cancel_adjacent_traced(gates: &mut Vec<DbTrackedGate>) -> usize {
    let old = std::mem::take(gates);
    let mut out: Vec<DbTrackedGate> = Vec::with_capacity(old.len());
    let mut cancelled = 0usize;
    for gate in old {
        if out.last().is_some_and(|last| last.gate == gate.gate) {
            out.pop();
            cancelled += 2;
        } else {
            out.push(gate);
        }
    }
    *gates = out;
    cancelled
}

fn key_of(canonical: &CanonicalXPolys) -> [u8; 16] {
    xxh3_128(&polys_repr_blob(&canonical.polys)).to_le_bytes()
}

#[derive(Clone)]
struct CandidateSpec {
    circuit: CircuitSeq,
    reversed: bool,
    order: Permutation,
    used_wires: Vec<u16>,
}

fn candidate_wire_slots(candidate: &CircuitSeq) -> usize {
    candidate
        .gates
        .iter()
        .flatten()
        .copied()
        .max()
        .map_or(0, |wire| wire as usize + 1)
}

/// Parse the frozen value's sequence of `[byte_len][three-byte g57 blob]`
/// entries without panicking on a damaged/truncated value.
fn decode_value(value: &[u8], report: &mut DbCompressReport) -> Vec<CircuitSeq> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < value.len() {
        let len = value[pos] as usize;
        pos += 1;
        if len % 3 != 0 || pos.checked_add(len).is_none_or(|end| end > value.len()) {
            report.malformed_value_entries += 1;
            break;
        }
        out.push(CircuitSeq::from_blob(&value[pos..pos + len]));
        pos += len;
    }
    out
}

/// Undo polynomial canonicalization and dense remapping, then convert the g57
/// friend to XGates.  This mirrors `candidate_to_circuit_space` in the legacy
/// compressor but is fallible rather than panicking when a friend needs more
/// global wires than are available.
fn candidate_to_xgates(
    mut candidate: CircuitSeq,
    reversed: bool,
    order: &Permutation,
    used_wires: &[u16],
    num_wires: usize,
    rng: &mut StdRng,
) -> Option<Vec<XGate>> {
    if candidate.gates.is_empty() {
        return Some(Vec::new());
    }
    if used_wires.iter().any(|&wire| wire as usize >= num_wires) {
        return None;
    }
    if reversed {
        candidate.gates.reverse();
    }

    let canonical_slots = candidate_wire_slots(&candidate);
    if canonical_slots > num_wires {
        return None;
    }
    let mut canonical_to_dense = order.data.clone();
    while canonical_to_dense.len() < canonical_slots {
        canonical_to_dense.push(canonical_to_dense.len());
    }
    if candidate
        .gates
        .iter()
        .flatten()
        .any(|&wire| wire as usize >= canonical_to_dense.len())
    {
        return None;
    }
    for gate in &mut candidate.gates {
        for wire in gate {
            *wire = canonical_to_dense[*wire as usize] as u16;
        }
    }

    let dense_slots = candidate_wire_slots(&candidate);
    let mut dense_to_global = used_wires.to_vec();
    if dense_to_global.len() < dense_slots {
        let mut occupied = vec![false; num_wires];
        for &wire in &dense_to_global {
            occupied[wire as usize] = true;
        }
        let mut available: Vec<u16> = (0..num_wires)
            .filter(|&wire| !occupied[wire])
            .map(|wire| u16::try_from(wire).ok())
            .collect::<Option<Vec<_>>>()?;
        available.shuffle(rng);
        let need = dense_slots - dense_to_global.len();
        if available.len() < need {
            return None;
        }
        dense_to_global.extend(available.into_iter().take(need));
    }

    let mut out = Vec::with_capacity(candidate.gates.len());
    for [target, positive, negative] in candidate.gates {
        let mapped = [
            *dense_to_global.get(target as usize)?,
            *dense_to_global.get(positive as usize)?,
            *dense_to_global.get(negative as usize)?,
        ];
        // Frozen friends are built from valid g57 base gates.  Reject corrupt
        // values that put the active wire in a control position.
        if mapped[0] == mapped[1] || mapped[0] == mapped[2] {
            return None;
        }
        out.push(XGate::from_g57(mapped));
    }
    Some(out)
}

fn record_poly_error(error: XPolyError, report: &mut DbCompressReport) {
    report.polynomial_skips += 1;
    if matches!(error, XPolyError::BudgetExceeded { .. }) {
        report.budget_skips += 1;
    }
}

/// Lineage-aware, testable implementation parameterized by a point lookup.
fn compress_contiguous_traced_with_lookup<F>(
    mut circuit: TracedCircuit,
    num_wires: usize,
    params: &DbCompressParams,
    mut lookup: F,
) -> (TracedCircuit, DbCompressReport)
where
    F: FnMut(&[u8; 16]) -> Option<Vec<u8>>,
{
    assert_eq!(circuit.gates.len(), circuit.roots.len());
    assert_eq!(circuit.gates.len(), circuit.source_marks.len());
    let mut gates: Vec<DbTrackedGate> = circuit
        .gates
        .drain(..)
        .zip(circuit.roots.drain(..))
        .zip(circuit.source_marks.drain(..))
        .map(|((gate, root), source)| DbTrackedGate { gate, root, source })
        .collect();
    let mut report = DbCompressReport {
        gates_in: gates.len(),
        lits_in: gates.iter().map(|gate| gate.gate.width() as u64).sum(),
        trials_requested: params.trials,
        ..DbCompressReport::default()
    };
    report.adjacent_gates_cancelled += cancel_adjacent_traced(&mut gates) as u64;

    let mut rng = StdRng::seed_from_u64(params.seed);
    let min_window = params.min_window.max(1);
    let max_window = params.max_window.max(min_window);

    for _ in 0..params.trials {
        let upper = max_window.min(gates.len());
        if upper < min_window {
            break;
        }
        report.trials_attempted += 1;
        let window_len = if min_window == upper {
            upper
        } else {
            rng.random_range(min_window..=upper)
        };
        let start = rng.random_range(0..=gates.len() - window_len);
        let end = start + window_len;
        let window = &gates[start..end];
        let window_gates: Vec<XGate> = window.iter().map(|gate| gate.gate.clone()).collect();
        let heterogeneous = window_gates.iter().any(|gate| !is_structural_g57(gate));
        if heterogeneous {
            report.heterogeneous_windows_attempted += 1;
        } else {
            report.all_g57_windows_attempted += 1;
        }

        let mut directions: Vec<(bool, CanonicalXPolys)> = Vec::with_capacity(2);
        report.canonical_directions_attempted += 1;
        match canonicalize_xgates_single(&window_gates, false, params.poly_budget) {
            Ok(canonical) => directions.push((false, canonical)),
            Err(error) => record_poly_error(error, &mut report),
        }
        if params.probe_reverse {
            report.canonical_directions_attempted += 1;
            match canonicalize_xgates_single(&window_gates, true, params.poly_budget) {
                Ok(canonical) => directions.push((true, canonical)),
                Err(error) => record_poly_error(error, &mut report),
            }
        }
        if directions.is_empty() {
            report.windows_without_key += 1;
            continue;
        }
        // Min-direction first matches the regular frozen table's build policy;
        // probing the other distinct key as well recovers coverage from older
        // or asymmetric stores.
        directions.sort_by(|a, b| a.1.polys.cmp(&b.1.polys).then(a.0.cmp(&b.0)));

        let mut seen_keys = HashSet::new();
        let mut candidates = Vec::<CandidateSpec>::new();
        for (reversed, canonical) in directions {
            let key = key_of(&canonical);
            if !seen_keys.insert(key) {
                continue;
            }
            report.lookups += 1;
            let Some(value) = lookup(&key) else {
                continue;
            };
            report.lookup_hits += 1;
            if heterogeneous {
                report.heterogeneous_lookup_hits += 1;
            } else {
                report.all_g57_lookup_hits += 1;
            }
            for candidate in decode_value(&value, &mut report) {
                report.candidates_decoded += 1;
                if candidate.gates.len() >= window_len {
                    continue; // strict gate-count reductions only
                }
                if candidate_wire_slots(&candidate) > num_wires {
                    continue;
                }
                report.shorter_candidates += 1;
                candidates.push(CandidateSpec {
                    circuit: candidate,
                    reversed,
                    order: canonical.order.clone(),
                    used_wires: canonical.used_wires.clone(),
                });
            }
        }
        if candidates.is_empty() {
            continue;
        }
        report.windows_with_shorter_candidate += 1;

        let best_len = candidates
            .iter()
            .map(|candidate| candidate.circuit.gates.len())
            .min()
            .expect("nonempty candidates");
        candidates.retain(|candidate| candidate.circuit.gates.len() == best_len);
        let pick = rng.random_range(0..candidates.len());
        let picked = candidates.swap_remove(pick);
        let Some(replacement) = candidate_to_xgates(
            picked.circuit,
            picked.reversed,
            &picked.order,
            &picked.used_wires,
            num_wires,
            &mut rng,
        ) else {
            continue;
        };
        if replacement.len() >= window_len {
            continue;
        }

        report.replacements += 1;
        if heterogeneous {
            report.heterogeneous_replacements += 1;
        } else {
            report.all_g57_replacements += 1;
        }
        report.gates_removed_by_db += (window_len - replacement.len()) as u64;
        let window_root = circuit.provenance.group_union(
            gates[start..end].iter().map(|gate| gate.root),
            GroupKind::Database,
        );
        let window_source = merge_sources(gates[start..end].iter().map(|gate| gate.source));
        match window_source {
            UNKNOWN_SOURCE => report.unknown_source_replacement_windows += 1,
            MIXED_SOURCE => report.mixed_parent_replacement_windows += 1,
            _ => report.single_parent_replacement_windows += 1,
        }
        let structural_outputs = replacement
            .iter()
            .filter(|gate| is_structural_g57(gate))
            .count();
        if !window_root.is_empty() && structural_outputs > 0 {
            report.attributed_g57_replacement_windows += 1;
            report.attributed_structural_g57_outputs += structural_outputs as u64;
            circuit
                .recovery
                .database
                .extend(std::iter::repeat_n(window_root, structural_outputs));
        }
        for gate in replacement.iter().filter(|gate| is_structural_g57(gate)) {
            report
                .structural_sources
                .record(window_source, gate, &circuit.source_parents);
        }
        let replacement = replacement
            .into_iter()
            .map(|gate| DbTrackedGate {
                gate,
                root: window_root,
                source: window_source,
            })
            .collect::<Vec<_>>();
        gates.splice(start..end, replacement);
    }

    // Catch duplicate pairs exposed across replacement boundaries.  Doing one
    // final linear pass avoids an O(n) full scan after every accepted window.
    report.adjacent_gates_cancelled += cancel_adjacent_traced(&mut gates) as u64;
    report.gates_out = gates.len();
    report.lits_out = gates.iter().map(|gate| gate.gate.width() as u64).sum();
    circuit.gates.reserve(gates.len());
    circuit.roots.reserve(gates.len());
    circuit.source_marks.reserve(gates.len());
    for gate in gates {
        circuit.gates.push(gate.gate);
        circuit.roots.push(gate.root);
        circuit.source_marks.push(gate.source);
    }
    (circuit, report)
}

/// Compatibility wrapper used by focused lookup tests and untraced callers.
fn compress_contiguous_with_lookup<F>(
    gates: Vec<XGate>,
    num_wires: usize,
    params: &DbCompressParams,
    lookup: F,
) -> (Vec<XGate>, DbCompressReport)
where
    F: FnMut(&[u8; 16]) -> Option<Vec<u8>>,
{
    let (provenance, roots) = ProvenanceArena::from_gates(&gates);
    let source_marks = vec![UNKNOWN_SOURCE; gates.len()];
    let traced = TracedCircuit {
        gates,
        roots,
        source_marks,
        source_parents: Vec::new(),
        provenance,
        recovery: RecoveryEvents::default(),
    };
    let (traced, report) =
        compress_contiguous_traced_with_lookup(traced, num_wires, params, lookup);
    (traced.gates, report)
}

/// Compress a heterogeneous tape against the regular frozen replacement store.
pub fn compress_frozen_contiguous(
    gates: Vec<XGate>,
    num_wires: usize,
    db: &FrozenDb,
    params: &DbCompressParams,
) -> (Vec<XGate>, DbCompressReport) {
    compress_contiguous_with_lookup(gates, num_wires, params, |key| db.get_regular(key))
}

/// Lineage-aware frozen-store compression used by `fcompress`.
pub fn compress_frozen_contiguous_traced(
    circuit: TracedCircuit,
    num_wires: usize,
    db: &FrozenDb,
    params: &DbCompressParams,
) -> (TracedCircuit, DbCompressReport) {
    compress_contiguous_traced_with_lookup(circuit, num_wires, params, |key| db.get_regular(key))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postmix::xgate::eval_lanes;
    use crate::postmix::xpoly::canonicalize_xgates_single;
    use std::collections::HashMap;

    fn encoded_value(circuits: &[CircuitSeq]) -> Vec<u8> {
        let mut value = Vec::new();
        for circuit in circuits {
            let blob = circuit.repr_blob();
            value.push(u8::try_from(blob.len()).unwrap());
            value.extend(blob);
        }
        value
    }

    fn exhaustively_equal(a: &[XGate], b: &[XGate], n: usize) -> bool {
        for input in 0..(1u64 << n) {
            let mut sa: Vec<u64> = (0..n)
                .map(|wire| if input >> wire & 1 == 1 { 1 } else { 0 })
                .collect();
            let mut sb = sa.clone();
            eval_lanes(a.iter(), &mut sa);
            eval_lanes(b.iter(), &mut sb);
            if sa.iter().zip(&sb).any(|(x, y)| (x ^ y) & 1 != 0) {
                return false;
            }
        }
        true
    }

    #[test]
    fn adjacent_identical_pairs_cancel_without_a_lookup() {
        let gate = XGate::conj(0, [(1, true), (2, false)]).unwrap();
        let params = DbCompressParams {
            trials: 0,
            ..DbCompressParams::default()
        };
        let (out, report) =
            compress_contiguous_with_lookup(vec![gate.clone(), gate], 3, &params, |_| None);
        assert!(out.is_empty());
        assert_eq!(report.adjacent_gates_cancelled, 2);
    }

    #[test]
    fn frozen_friend_reassembles_two_fragments_to_one_g57() {
        let before = vec![
            XGate::conj(0, [(1, true)]).unwrap(),
            XGate::conj(0, [(1, false), (2, false)]).unwrap(),
        ];
        let legacy = CircuitSeq {
            gates: vec![[0, 1, 2]],
        };

        // Frozen values are expressed in canonical wire space.  Construct the
        // one-gate friend exactly as the builder/legacy canonicalizer does.
        let (legacy_polys, legacy_order, _) = legacy.canonicalize_polys_single(false);
        let mut stored = legacy.clone();
        stored.rewire(&legacy_order.invert(), 3);
        stored.canonicalize();
        let key = xxh3_128(&polys_repr_blob(&legacy_polys)).to_le_bytes();
        let values = HashMap::from([(key, encoded_value(&[stored]))]);

        let params = DbCompressParams {
            trials: 1,
            min_window: 2,
            max_window: 2,
            seed: 7,
            ..DbCompressParams::default()
        };
        let (provenance, roots) = ProvenanceArena::from_gates(&before);
        let traced = TracedCircuit {
            gates: before.clone(),
            roots,
            source_marks: vec![0, 1],
            source_parents: vec![XGate::from_g57([0, 1, 2]), XGate::from_g57([0, 2, 1])],
            provenance,
            recovery: RecoveryEvents::default(),
        };
        let (traced, report) =
            compress_contiguous_traced_with_lookup(traced, 3, &params, |query| {
                values.get(query).cloned()
            });
        let summary = traced.recovery_summary();
        let out = traced.gates;
        assert_eq!(out.len(), 1);
        assert_eq!(report.replacements, 1);
        assert_eq!(report.heterogeneous_windows_attempted, 1);
        assert_eq!(report.heterogeneous_lookup_hits, 1);
        assert_eq!(report.heterogeneous_replacements, 1);
        assert_eq!(report.all_g57_windows_attempted, 0);
        assert_eq!(report.all_g57_lookup_hits, 0);
        assert_eq!(report.all_g57_replacements, 0);
        assert_eq!(report.gates_removed_by_db, 1);
        assert_eq!(report.attributed_g57_replacement_windows, 1);
        assert_eq!(report.attributed_structural_g57_outputs, 1);
        assert_eq!(report.mixed_parent_replacement_windows, 1);
        assert_eq!(report.single_parent_replacement_windows, 0);
        assert_eq!(report.structural_sources.new_mixed_parents, 1);
        assert_eq!(report.structural_sources.returned_to_parent, 0);
        assert_eq!(summary.database.exact.total, 0);
        assert_eq!(summary.database.inclusive.total, 2);
        assert_eq!(summary.final_structural_g57.exact.total, 0);
        assert_eq!(summary.final_structural_g57.inclusive.total, 2);
        assert!(out[0].comp);
        assert!(exhaustively_equal(&before, &out, 3));
    }

    #[test]
    fn structural_g57_recognition_is_exact() {
        assert!(is_structural_g57(&XGate::from_g57([0, 1, 2])));
        assert!(!is_structural_g57(
            &XGate::conj(0, [(1, true), (2, false)]).unwrap()
        ));
        let mut one_control = XGate::from_g57([0, 1, 2]);
        one_control.ctrls.pop();
        assert!(!is_structural_g57(&one_control));
        let mut same_polarity = XGate::from_g57([0, 1, 2]);
        same_polarity.ctrls[0].1 = same_polarity.ctrls[1].1;
        assert!(!is_structural_g57(&same_polarity));
    }

    #[test]
    fn equal_length_friend_is_never_accepted() {
        let before = vec![XGate::from_g57([0, 1, 2]), XGate::from_g57([1, 0, 2])];
        let canonical = canonicalize_xgates_single(&before, false, XPolyBudget::default()).unwrap();
        let key = key_of(&canonical);
        let same_len = CircuitSeq {
            gates: vec![[0, 1, 2], [1, 0, 2]],
        };
        let value = encoded_value(&[same_len]);
        let params = DbCompressParams {
            trials: 1,
            min_window: 2,
            max_window: 2,
            probe_reverse: false,
            ..DbCompressParams::default()
        };
        let (out, report) = compress_contiguous_with_lookup(before.clone(), 3, &params, |query| {
            (query == &key).then(|| value.clone())
        });
        assert_eq!(out, before);
        assert_eq!(report.replacements, 0);
        assert_eq!(report.shorter_candidates, 0);
    }

    #[test]
    fn polynomial_budget_failure_skips_the_window_and_lookup() {
        let before = vec![XGate::from_g57([0, 1, 2])];
        let params = DbCompressParams {
            trials: 1,
            min_window: 1,
            max_window: 1,
            probe_reverse: false,
            poly_budget: XPolyBudget {
                max_mul_terms: 0,
                max_poly_terms: 16,
                max_total_terms: 32,
            },
            ..DbCompressParams::default()
        };
        let mut lookups = 0;
        let (out, report) = compress_contiguous_with_lookup(before.clone(), 3, &params, |_| {
            lookups += 1;
            None
        });
        assert_eq!(out, before);
        assert_eq!(lookups, 0);
        assert_eq!(report.budget_skips, 1);
        assert_eq!(report.windows_without_key, 1);
    }
}
