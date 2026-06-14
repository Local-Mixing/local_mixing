use rand::{Rng, prelude::SliceRandom};
use serde::{Deserialize, Serialize};

extern crate lmdb_sys;

use crate::{
    circuit::circuit::CircuitSeq,
    random::random_data::random_circuit,
    replace::sat_score::{
        compression_selection_score, expansion_selection_score, sat_bcp_enabled,
        sat_bcp_min_resistance, sat_compress_preserve_delta, sat_compress_protect_enabled,
        sat_cone_aware_enabled, sat_cone_min_fraction, sat_expand_min_delta, sat_score_seed,
        sat_score_slack, sat_scoring_enabled, score_subcircuit,
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

pub fn expand_curated_lmdb(
    gates: &[[u16; 3]],
    n: usize,
    env: &lmdb::Environment,
    curated_shard_dbs: &[lmdb::Database],
    shard_dbs: &[lmdb::Database],
) -> Option<Vec<[u16; 3]>> {
    use crate::circuit::circuit::{Permutation, polys_repr_blob};
    use lmdb::Transaction;
    use rand::prelude::SliceRandom;
    use xxhash_rust::xxh3::xxh3_128;

    let mut rng = rand::rng();
    let sub = CircuitSeq {
        gates: gates.to_vec(),
    };

    let (fwd_polys, fwd_order, used) = sub.canonicalize_polys_single(false);
    if fwd_polys.is_empty() {
        return None;
    }

    let fwd_key = xxh3_128(&polys_repr_blob(&fwd_polys))
        .to_le_bytes()
        .to_vec();
    let fwd_shard = fwd_key[0] as usize;

    let txn = env.begin_ro_txn().ok()?;

    // Try curated DBs first (forward direction only — no reversal needed for curated).
    let curated_hit = if !curated_shard_dbs.is_empty() {
        txn.get(curated_shard_dbs[fwd_shard], &fwd_key)
            .map(|v: &[u8]| v.to_vec())
            .ok()
    } else {
        None
    };

    let (value, final_order, is_reversed) = if let Some(v) = curated_hit {
        (v, fwd_order, false)
    } else if !shard_dbs.is_empty() {
        // Fallback: try regular shard DBs (both forward and reverse, same as expand_lmdb).
        if let Ok(v) = txn
            .get(shard_dbs[fwd_shard], &fwd_key)
            .map(|v: &[u8]| v.to_vec())
        {
            (v, fwd_order, false)
        } else {
            let (rev_polys, rev_order, _) = sub.canonicalize_polys_single(true);
            if rev_polys.is_empty() {
                return None;
            }
            let rev_key = xxh3_128(&polys_repr_blob(&rev_polys))
                .to_le_bytes()
                .to_vec();
            let rev_shard = rev_key[0] as usize;
            match txn
                .get(shard_dbs[rev_shard], &rev_key)
                .map(|v: &[u8]| v.to_vec())
            {
                Ok(v) => (v, rev_order, true),
                Err(_) => return None,
            }
        }
    } else {
        return None;
    };

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
        let mut available: Vec<u16> = (0..n as u16).filter(|w| !used_ext.contains(w)).collect();
        available.shuffle(&mut rng);
        let mut avail = available.into_iter();
        while used_ext.len() < repl_n_b {
            avail.next().map(|w| used_ext.push(w))?;
        }
    }

    Some(CircuitSeq::unrewire_subcircuit(&repl, &used_ext).gates)
}

// Like expand_curated_lmdb but returns the minimum-gate replacement with <= gates.len() gates.
pub fn compress_curated_lmdb(
    gates: &[[u16; 3]],
    n: usize,
    env: &lmdb::Environment,
    curated_shard_dbs: &[lmdb::Database],
    shard_dbs: &[lmdb::Database],
) -> Option<Vec<[u16; 3]>> {
    use crate::circuit::circuit::{Permutation, polys_repr_blob};
    use lmdb::Transaction;
    use xxhash_rust::xxh3::xxh3_128;

    let mut rng = rand::rng();
    let sub = CircuitSeq {
        gates: gates.to_vec(),
    };

    let (fwd_polys, fwd_order, used) = sub.canonicalize_polys_single(false);
    if fwd_polys.is_empty() {
        return None;
    }

    let fwd_key = xxh3_128(&polys_repr_blob(&fwd_polys))
        .to_le_bytes()
        .to_vec();
    let fwd_shard = fwd_key[0] as usize;

    let txn = env.begin_ro_txn().ok()?;

    let curated_hit = if !curated_shard_dbs.is_empty() {
        txn.get(curated_shard_dbs[fwd_shard], &fwd_key)
            .map(|v: &[u8]| v.to_vec())
            .ok()
    } else {
        None
    };

    let (value, final_order, is_reversed) = if let Some(v) = curated_hit {
        (v, fwd_order, false)
    } else if !shard_dbs.is_empty() {
        if let Ok(v) = txn
            .get(shard_dbs[fwd_shard], &fwd_key)
            .map(|v: &[u8]| v.to_vec())
        {
            (v, fwd_order, false)
        } else {
            let (rev_polys, rev_order, _) = sub.canonicalize_polys_single(true);
            if rev_polys.is_empty() {
                return None;
            }
            let rev_key = xxh3_128(&polys_repr_blob(&rev_polys))
                .to_le_bytes()
                .to_vec();
            let rev_shard = rev_key[0] as usize;
            match txn
                .get(shard_dbs[rev_shard], &rev_key)
                .map(|v: &[u8]| v.to_vec())
            {
                Ok(v) => (v, rev_order, true),
                Err(_) => return None,
            }
        }
    } else {
        return None;
    };

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
        let mut available: Vec<u16> = (0..n as u16).filter(|w| !used_ext.contains(w)).collect();
        available.shuffle(&mut rng);
        let mut avail = available.into_iter();
        while used_ext.len() < repl_n_b {
            avail.next().map(|w| used_ext.push(w))?;
        }
    }

    Some(CircuitSeq::unrewire_subcircuit(&repl, &used_ext).gates)
}

/// Like compress_curated_lmdb, but returns ANY equivalent replacement for `gates` from the
/// given DBs — NOT required to be a compression. Among the stored friends it prefers the
/// fewest gates (so it still compresses when possible) but accepts equal-length or longer
/// replacements, and does not reject lone equal-length options. Used as the relaxed fallback
/// in the unsamfing stage so an undo SAMF can still be hidden even when no strictly-shorter
/// curated replacement exists. Equivalence-preserving: all stored friends share the window's
/// canonical polynomial form.
pub fn find_any_replacement_lmdb(
    gates: &[[u16; 3]],
    n: usize,
    env: &lmdb::Environment,
    curated_shard_dbs: &[lmdb::Database],
    shard_dbs: &[lmdb::Database],
) -> Option<Vec<[u16; 3]>> {
    use crate::circuit::circuit::{Permutation, polys_repr_blob};
    use lmdb::Transaction;
    use xxhash_rust::xxh3::xxh3_128;

    let mut rng = rand::rng();
    let sub = CircuitSeq {
        gates: gates.to_vec(),
    };

    let (fwd_polys, fwd_order, used) = sub.canonicalize_polys_single(false);
    if fwd_polys.is_empty() {
        return None;
    }

    let fwd_key = xxh3_128(&polys_repr_blob(&fwd_polys))
        .to_le_bytes()
        .to_vec();
    let fwd_shard = fwd_key[0] as usize;

    let txn = env.begin_ro_txn().ok()?;

    let curated_hit = if !curated_shard_dbs.is_empty() {
        txn.get(curated_shard_dbs[fwd_shard], &fwd_key)
            .map(|v: &[u8]| v.to_vec())
            .ok()
    } else {
        None
    };

    let (value, final_order, is_reversed) = if let Some(v) = curated_hit {
        (v, fwd_order, false)
    } else if !shard_dbs.is_empty() {
        if let Ok(v) = txn
            .get(shard_dbs[fwd_shard], &fwd_key)
            .map(|v: &[u8]| v.to_vec())
        {
            (v, fwd_order, false)
        } else {
            let (rev_polys, rev_order, _) = sub.canonicalize_polys_single(true);
            if rev_polys.is_empty() {
                return None;
            }
            let rev_key = xxh3_128(&polys_repr_blob(&rev_polys))
                .to_le_bytes()
                .to_vec();
            let rev_shard = rev_key[0] as usize;
            match txn
                .get(shard_dbs[rev_shard], &rev_key)
                .map(|v: &[u8]| v.to_vec())
            {
                Ok(v) => (v, rev_order, true),
                Err(_) => return None,
            }
        }
    } else {
        return None;
    };

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
        let mut available: Vec<u16> = (0..n as u16).filter(|w| !used_ext.contains(w)).collect();
        available.shuffle(&mut rng);
        let mut avail = available.into_iter();
        while used_ext.len() < repl_n_b {
            avail.next().map(|w| used_ext.push(w))?;
        }
    }

    Some(CircuitSeq::unrewire_subcircuit(&repl, &used_ext).gates)
}

// returns the replacement gates and the id length (0 when using curated path)
pub fn replace_single_pair(
    left: &[u16; 3],
    right: &[u16; 3],
    num_wires: usize,
    env: &lmdb::Environment,
    curated_shard_dbs: &[lmdb::Database],
    shard_dbs: &[lmdb::Database],
) -> (Vec<[u16; 3]>, usize) {
    if curated_shard_dbs.is_empty() {
        return (vec![], 0);
    }
    match expand_curated_lmdb(
        &[*left, *right],
        num_wires,
        env,
        curated_shard_dbs,
        shard_dbs,
    ) {
        Some(repl) => (repl, 0),
        None => (vec![], 0),
    }
}

// Replace triple of gates
// Largely unused as if replacing pairs is effective, replacing triples would largely be the same

// Used in the interleave method
// Create a circuit on n..2n wires and then interleave them
pub fn interleave(circuit: &CircuitSeq, n: usize, env: &lmdb::Environment) -> CircuitSeq {
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
                replace_single_pair(&circuit.gates[i], &random.gates[i], 2 * n, env, &[], &[]).0;
            gates.extend_from_slice(&replaced_pair);
        } else {
            gates.push(circuit.gates[i]);
            gates.push(random.gates[i]);
        }
    }

    CircuitSeq { gates }
}
