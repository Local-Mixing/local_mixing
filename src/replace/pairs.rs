use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    circuit::circuit::CircuitSeq,
    random::random_data::random_circuit,
    replace::sat_score::{
        compression_selection_score, expansion_selection_score, sat_bcp_enabled,
        sat_bcp_min_resistance, sat_compress_preserve_delta, sat_compress_protect_enabled,
        sat_cone_aware_enabled, sat_cone_min_fraction, sat_expand_min_delta, sat_score_seed,
        sat_score_slack, sat_scoring_enabled, score_subcircuit,
    },
    replace::{
        frozen::FrozenDb,
        replace::{note_wire_use, shuffled_unused_wires},
    },
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Gate taxonomy and Replacement Pair
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CollisionType {
    OnActive,
    OnCtrl1,
    OnCtrl2,
    OnNew,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GatePair {
    pub a: CollisionType,
    pub c1: CollisionType,
    pub c2: CollisionType,
}

/// Which frozen namespace(s) a pair lookup may consult.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LookupScope {
    CuratedOnly,
    RegularOnly,
    CuratedThenRegular,
}

impl GatePair {
    pub fn new() -> Self {
        GatePair {
            a: CollisionType::OnNew,
            c1: CollisionType::OnNew,
            c2: CollisionType::OnNew,
        }
    }

    pub fn is_none(gate_pair: &Self) -> bool {
        gate_pair.a == CollisionType::OnNew
            && gate_pair.c1 == CollisionType::OnNew
            && gate_pair.c2 == CollisionType::OnNew
    }
}

pub fn expand_curated_frozen(gates: &[[u16; 3]], n: usize, db: &FrozenDb) -> Option<Vec<[u16; 3]>> {
    expand_curated_frozen_neg(gates, n, db, &[], LookupScope::CuratedThenRegular)
}

/// Shared lookup: probe the curated frozen store with the possibly adjusted
/// forward key, then optionally fall back to the regular frozen store.
/// The shard fallback honours MIN_DIR_LOOKUP exactly like expand_frozen /
/// compress_frozen (see replace.rs), and every probe goes through the exact
/// process-wide lookup cache. Returns (value, final_order, is_reversed).
fn curated_then_shard_lookup(
    sub: &CircuitSeq,
    db: &FrozenDb,
    fwd_polys: Vec<crate::circuit::circuit::Polynomial>,
    fwd_order: crate::circuit::circuit::Permutation,
    scope: LookupScope,
) -> Option<(
    std::sync::Arc<[u8]>,
    crate::circuit::circuit::Permutation,
    bool,
)> {
    use crate::circuit::circuit::polys_repr_blob;
    use crate::replace::replace::{
        LOOKUP_NS_CURATED, LOOKUP_NS_SHARD, MIN_DIR_VALIDATE_PROBES, MIN_DIR_VIOLATIONS,
        MinDirLookup, cached_db_get, min_dir_lookup_mode,
    };
    use std::sync::atomic::Ordering;
    use xxhash_rust::xxh3::xxh3_128;

    let fwd_key = xxh3_128(&polys_repr_blob(&fwd_polys)).to_le_bytes();
    // The curated store is probed forward-only, matching the original behaviour.
    if scope != LookupScope::RegularOnly && db.has_curated() {
        if let Some(v) = cached_db_get(db, LOOKUP_NS_CURATED, &fwd_key) {
            return Some((v, fwd_order, false));
        }
    }

    if scope == LookupScope::CuratedOnly {
        return None;
    }

    match min_dir_lookup_mode() {
        MinDirLookup::Legacy => {
            if let Some(v) = cached_db_get(db, LOOKUP_NS_SHARD, &fwd_key) {
                return Some((v, fwd_order, false));
            }
            let (rev_polys, rev_order, _) = sub.canonicalize_polys_single(true);
            if rev_polys.is_empty() {
                return None;
            }
            let rev_key = xxh3_128(&polys_repr_blob(&rev_polys)).to_le_bytes();
            cached_db_get(db, LOOKUP_NS_SHARD, &rev_key).map(|v| (v, rev_order, true))
        }
        mode => {
            let (rev_polys, rev_order, _) = sub.canonicalize_polys_single(true);
            if rev_polys.is_empty() {
                return None;
            }
            let rev_is_min = rev_polys < fwd_polys;
            let (min_polys, min_order, min_reversed, alt_polys, alt_order, alt_reversed) =
                if rev_is_min {
                    (rev_polys, rev_order, true, fwd_polys, fwd_order, false)
                } else {
                    (fwd_polys, fwd_order, false, rev_polys, rev_order, true)
                };
            let min_key = xxh3_128(&polys_repr_blob(&min_polys)).to_le_bytes();
            if let Some(v) = cached_db_get(db, LOOKUP_NS_SHARD, &min_key) {
                return Some((v, min_order, min_reversed));
            }
            if mode == MinDirLookup::Validate {
                MIN_DIR_VALIDATE_PROBES.fetch_add(1, Ordering::Relaxed);
                let alt_key = xxh3_128(&polys_repr_blob(&alt_polys)).to_le_bytes();
                if let Some(v) = cached_db_get(db, LOOKUP_NS_SHARD, &alt_key) {
                    MIN_DIR_VIOLATIONS.fetch_add(1, Ordering::Relaxed);
                    eprintln!(
                        "[min-dir-violation] pairs: non-min canonical key present while min key absent (gates={})",
                        sub.gates.len()
                    );
                    return Some((v, alt_order, alt_reversed));
                }
            }
            None
        }
    }
}

pub fn expand_curated_frozen_neg(
    gates: &[[u16; 3]],
    n: usize,
    db: &FrozenDb,
    negated_inputs: &[u16],
    scope: LookupScope,
) -> Option<Vec<[u16; 3]>> {
    use crate::circuit::circuit::Permutation;

    let mut rng = rand::rng();
    let sub = CircuitSeq {
        gates: gates.to_vec(),
    };

    let (fwd_polys, fwd_order, used) = if negated_inputs.is_empty() {
        sub.canonicalize_polys_single(false)
    } else {
        sub.canonicalize_polys_single_neg(negated_inputs)
    };
    if fwd_polys.is_empty() {
        return None;
    }

    // Curated store first (forward-only), then the regular store (cached,
    // MIN_DIR_LOOKUP-aware). The shard fallback is only valid for the plain
    // canonical form, so it is disabled when inputs are negated.
    let (value, final_order, is_reversed) = curated_then_shard_lookup(
        &sub,
        db,
        fwd_polys,
        fwd_order,
        if negated_inputs.is_empty() {
            scope
        } else {
            LookupScope::CuratedOnly
        },
    )?;

    let mut candidates: Vec<CircuitSeq> = Vec::new();
    let mut pos = 0;
    while pos < value.len() {
        let len = value[pos] as usize;
        pos += 1;
        if pos + len > value.len() {
            break;
        }
        let candidate = CircuitSeq::from_blob(&value[pos..pos + len]);
        pos += len;
        if candidate.gates.len() > gates.len() {
            candidates.push(candidate);
        }
    }
    if candidates.is_empty() {
        return None;
    }

    let mut best: Vec<CircuitSeq> = if sat_scoring_enabled() {
        let seed = sat_score_seed();
        let base_n = sub.max_wire() + 1;
        let base_score =
            expansion_selection_score(&score_subcircuit(&sub.gates, base_n, seed ^ 0xBAD5_EED));
        let required_score = base_score + sat_expand_min_delta();
        let scored: Vec<(f64, CircuitSeq)> = candidates
            .into_iter()
            .enumerate()
            .filter_map(|(idx, candidate)| {
                let score_n = candidate.max_wire() + 1;
                let sat_score = score_subcircuit(&candidate.gates, score_n, seed ^ idx as u64);
                if sat_cone_aware_enabled()
                    && sat_score.output_cone_fraction < sat_cone_min_fraction()
                {
                    return None;
                }
                if sat_bcp_enabled() && sat_score.bcp_resistance < sat_bcp_min_resistance() {
                    return None;
                }
                Some((expansion_selection_score(&sat_score), candidate))
            })
            .filter(|(score, _)| *score > required_score)
            .collect();
        if scored.is_empty() {
            return None;
        }
        let max_score = scored
            .iter()
            .map(|(score, _)| *score)
            .fold(f64::NEG_INFINITY, f64::max);
        scored
            .into_iter()
            .filter(|(score, _)| (*score - max_score).abs() <= 1e-9)
            .map(|(_, candidate)| candidate)
            .collect()
    } else {
        // Favor expansions that are both LONG (many gates) and WIDE (many distinct wires),
        // not solely the largest gate count. Score each candidate by gates + wires and keep
        // the top-scoring set, breaking ties at random.
        let score = |c: &CircuitSeq| c.gates.len() + c.used_wires().len();
        let max_score = candidates.iter().map(|c| score(c)).max().unwrap();
        candidates
            .into_iter()
            .filter(|c| score(c) == max_score)
            .collect()
    };
    let idx = if matches!(
        crate::replace::replace::incoming_rank_mode(),
        crate::replace::replace::IncomingRankMode::Fanout
            | crate::replace::replace::IncomingRankMode::Hybrid
    ) {
        let features: Vec<_> = best
            .iter()
            .map(|candidate| crate::replace::replace::cand_features(&candidate.gates, &[], &[]))
            .collect();
        crate::replace::ranking::incoming()
            .order(&features)
            .first()
            .copied()
            .unwrap_or(0)
    } else {
        rng.random_range(0..best.len())
    };
    let mut repl = best.swap_remove(idx);

    if is_reversed {
        repl.gates.reverse();
    }

    let repl_n = repl.max_wire() + 1;
    let mut order_data = final_order.data.clone();
    while order_data.len() < repl_n {
        let i = order_data.len();
        order_data.push(i);
    }
    repl.rewire(
        &Permutation { data: order_data },
        std::cmp::max(repl_n, final_order.data.len()),
    );

    let repl_n_b = repl.max_wire() + 1;
    let mut used_ext = used.clone();
    if used_ext.len() < repl_n_b {
        let available = shuffled_unused_wires(n, &used_ext, &mut rng);
        let mut avail = available.into_iter();
        while used_ext.len() < repl_n_b {
            avail.next().map(|w| used_ext.push(w))?;
        }
    }

    let out = CircuitSeq::unrewire_subcircuit(&repl, &used_ext).gates;
    note_wire_use(&out);
    Some(out)
}

// Like expand_curated_frozen but returns the minimum-gate replacement with <= gates.len() gates.
pub fn compress_curated_frozen(
    gates: &[[u16; 3]],
    n: usize,
    db: &FrozenDb,
    scope: LookupScope,
) -> Option<Vec<[u16; 3]>> {
    use crate::circuit::circuit::Permutation;

    let mut rng = rand::rng();
    let sub = CircuitSeq {
        gates: gates.to_vec(),
    };

    let (fwd_polys, fwd_order, used) = sub.canonicalize_polys_single(false);
    if fwd_polys.is_empty() {
        return None;
    }

    let (value, final_order, is_reversed) =
        curated_then_shard_lookup(&sub, db, fwd_polys, fwd_order, scope)?;

    let mut candidates: Vec<CircuitSeq> = Vec::new();
    let mut pos = 0;
    while pos < value.len() {
        let len = value[pos] as usize;
        pos += 1;
        if pos + len > value.len() {
            break;
        }
        let candidate = CircuitSeq::from_blob(&value[pos..pos + len]);
        pos += len;
        if candidate.gates.len() <= gates.len() {
            candidates.push(candidate);
        }
    }
    if candidates.is_empty() {
        return None;
    }

    let min_gates = candidates.iter().map(|c| c.gates.len()).min().unwrap();
    let mut best: Vec<CircuitSeq> = if sat_scoring_enabled() {
        let max_len = min_gates.saturating_add(sat_score_slack()).min(gates.len());
        let seed = sat_score_seed();
        let base_score = compression_selection_score(&score_subcircuit(
            &sub.gates,
            sub.max_wire() + 1,
            seed ^ 0xc0de_1234,
        ));
        let scored: Vec<(f64, CircuitSeq)> = candidates
            .into_iter()
            .filter(|c| c.gates.len() <= max_len)
            .enumerate()
            .filter_map(|(idx, candidate)| {
                let score_n = candidate.max_wire() + 1;
                let sat_score = score_subcircuit(&candidate.gates, score_n, seed ^ idx as u64);
                if sat_cone_aware_enabled()
                    && sat_score.output_cone_fraction < sat_cone_min_fraction()
                {
                    return None;
                }
                if sat_bcp_enabled() && sat_score.bcp_resistance < sat_bcp_min_resistance() {
                    return None;
                }
                let candidate_score = compression_selection_score(&sat_score);
                if sat_compress_protect_enabled()
                    && candidate_score + sat_compress_preserve_delta() < base_score
                {
                    return None;
                }
                Some((candidate_score, candidate))
            })
            .collect();
        if scored.is_empty() {
            return None;
        }
        let max_score = scored
            .iter()
            .map(|(score, _)| *score)
            .fold(f64::NEG_INFINITY, f64::max);
        scored
            .into_iter()
            .filter(|(score, _)| (*score - max_score).abs() <= 1e-9)
            .map(|(_, candidate)| candidate)
            .collect()
    } else {
        // Pick minimum-gate replacement for maximum compression.
        candidates
            .into_iter()
            .filter(|c| c.gates.len() == min_gates)
            .collect()
    };

    // Equal-length replacement only counts if there are multiple friends (alternatives).
    // A single equal-length option adds no obfuscation value.
    if min_gates == gates.len() && best.len() <= 1 {
        return None;
    }

    let idx = if matches!(
        crate::replace::replace::incoming_rank_mode(),
        crate::replace::replace::IncomingRankMode::Fanout
            | crate::replace::replace::IncomingRankMode::Hybrid
    ) {
        let features: Vec<_> = best
            .iter()
            .map(|candidate| crate::replace::replace::cand_features(&candidate.gates, &[], &[]))
            .collect();
        crate::replace::ranking::incoming()
            .order(&features)
            .first()
            .copied()
            .unwrap_or(0)
    } else {
        rng.random_range(0..best.len())
    };
    let mut repl = best.swap_remove(idx);

    if is_reversed {
        repl.gates.reverse();
    }

    let repl_n = repl.max_wire() + 1;
    let mut order_data = final_order.data.clone();
    while order_data.len() < repl_n {
        let i = order_data.len();
        order_data.push(i);
    }
    repl.rewire(
        &Permutation { data: order_data },
        std::cmp::max(repl_n, final_order.data.len()),
    );

    let repl_n_b = repl.max_wire() + 1;
    let mut used_ext = used.clone();
    if used_ext.len() < repl_n_b {
        let available = shuffled_unused_wires(n, &used_ext, &mut rng);
        let mut avail = available.into_iter();
        while used_ext.len() < repl_n_b {
            avail.next().map(|w| used_ext.push(w))?;
        }
    }

    let out = CircuitSeq::unrewire_subcircuit(&repl, &used_ext).gates;
    note_wire_use(&out);
    Some(out)
}

/// Like compress_curated_frozen, but returns ANY equivalent replacement for `gates` from the
/// selected frozen stores — NOT required to be a compression. Among the stored friends it prefers the
/// fewest gates (so it still compresses when possible) but accepts equal-length or longer
/// replacements, and does not reject lone equal-length options. Used as the relaxed fallback
/// in the unsamfing stage so an undo SAMF can still be hidden even when no strictly-shorter
/// curated replacement exists. Equivalence-preserving: all stored friends share the window's
/// canonical polynomial form.
pub fn find_any_replacement_frozen(
    gates: &[[u16; 3]],
    n: usize,
    db: &FrozenDb,
    scope: LookupScope,
) -> Option<Vec<[u16; 3]>> {
    use crate::circuit::circuit::Permutation;

    let mut rng = rand::rng();
    let sub = CircuitSeq {
        gates: gates.to_vec(),
    };

    let (fwd_polys, fwd_order, used) = sub.canonicalize_polys_single(false);
    if fwd_polys.is_empty() {
        return None;
    }

    let (value, final_order, is_reversed) =
        curated_then_shard_lookup(&sub, db, fwd_polys, fwd_order, scope)?;

    // Accept ALL stored friends (any length), preferring the fewest gates.
    let mut candidates: Vec<CircuitSeq> = Vec::new();
    let mut pos = 0;
    while pos < value.len() {
        let len = value[pos] as usize;
        pos += 1;
        if pos + len > value.len() {
            break;
        }
        candidates.push(CircuitSeq::from_blob(&value[pos..pos + len]));
        pos += len;
    }
    if candidates.is_empty() {
        return None;
    }

    let min_gates = candidates.iter().map(|c| c.gates.len()).min().unwrap();
    let mut best: Vec<CircuitSeq> = candidates
        .into_iter()
        .filter(|c| c.gates.len() == min_gates)
        .collect();

    let idx = rng.random_range(0..best.len());
    let mut repl = best.swap_remove(idx);

    if is_reversed {
        repl.gates.reverse();
    }

    let repl_n = repl.max_wire() + 1;
    let mut order_data = final_order.data.clone();
    while order_data.len() < repl_n {
        let i = order_data.len();
        order_data.push(i);
    }
    repl.rewire(
        &Permutation { data: order_data },
        std::cmp::max(repl_n, final_order.data.len()),
    );

    let repl_n_b = repl.max_wire() + 1;
    let mut used_ext = used.clone();
    if used_ext.len() < repl_n_b {
        let available = shuffled_unused_wires(n, &used_ext, &mut rng);
        let mut avail = available.into_iter();
        while used_ext.len() < repl_n_b {
            avail.next().map(|w| used_ext.push(w))?;
        }
    }

    let out = CircuitSeq::unrewire_subcircuit(&repl, &used_ext).gates;
    note_wire_use(&out);
    Some(out)
}

// returns the replacement gates and the id length (0 when using curated path)
pub fn replace_single_pair(
    left: &[u16; 3],
    right: &[u16; 3],
    num_wires: usize,
    db: &FrozenDb,
) -> (Vec<[u16; 3]>, usize) {
    match expand_curated_frozen(&[*left, *right], num_wires, db) {
        Some(repl) => (repl, 0),
        None => (vec![], 0),
    }
}

// Replace triple of gates
// Largely unused as if replacing pairs is effective, replacing triples would largely be the same

// Used in the interleave method
// Create a circuit on n..2n wires and then interleave them
pub fn interleave(circuit: &CircuitSeq, n: usize, db: &FrozenDb) -> CircuitSeq {
    let m = circuit.gates.len();
    let mut random = random_circuit(n, m);
    let mut rng = rand::rng();
    let mut gates = Vec::new();
    for gate in random.gates.iter_mut() {
        for pin in gate.iter_mut() {
            *pin += n as u16;
        }
    }
    for i in 0..m {
        // Choose between pair replacemnt or CNOT
        // NOTs not currently supported
        let choice = rng.random_range(0..2);
        if choice == 0 {
            let replaced_pair =
                replace_single_pair(&circuit.gates[i], &random.gates[i], 2 * n, db).0;
            if replaced_pair.is_empty() {
                gates.push(circuit.gates[i]);
                gates.push(random.gates[i]);
            } else {
                gates.extend_from_slice(&replaced_pair);
            }
        } else {
            gates.push(circuit.gates[i]);
            gates.push(random.gates[i]);
        }
    }

    CircuitSeq { gates }
}
