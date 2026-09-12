//! S9 (v1) of the wide-gate design: the wide sidecar store's read path, as
//! FEATURE-GATED PROBE TOOLING. The runtime invariant "lookup does not pull
//! in RocksDB" stands — production fmix arming waits for the frozen-wide
//! container (FRZWID01); until then this module serves verification bins and
//! offline experiments, and the shipped runtime binary is byte-identical.
//!
//! Every candidate returned by [`wide_probe`] is PROVEN equivalent to the
//! window by ANF comparison before it is handed back — the strongest
//! correctness contract any consumer could ask for.
use crate::circuit::polys_repr_blob;
use crate::circuit::xgate::XGate;
use crate::db_generation::regular::append_merge_wide;
use crate::db_mixing::db_replace::polys_equivalent;
use crate::engine::mpx1;
use crate::engine::xpoly::{XPolyBudget, canonicalize_xgates_single};
use rand::Rng;
use rand::seq::SliceRandom;
use rocksdb::{DB, Options};
use xxhash_rust::xxh3::xxh3_128;

/// Read handle for the wide RocksDB sidecar. `None` inside = every probe
/// misses (mirrors `FrozenDb::empty`'s convention).
pub struct WideDb {
    db: Option<DB>,
}

impl WideDb {
    pub fn open(dir: &str) -> Self {
        let mut opts = Options::default();
        opts.create_if_missing(false);
        opts.set_max_open_files(-1);
        opts.set_merge_operator_associative("append_merge_wide", append_merge_wide);
        let db = DB::open_for_read_only(&opts, dir, false)
            .unwrap_or_else(|e| panic!("open wide store {dir}: {e}"));
        Self { db: Some(db) }
    }

    /// `WIDE_DB_DIR` unset => a handle where every lookup misses.
    pub fn from_env() -> Self {
        match std::env::var("WIDE_DB_DIR") {
            Ok(dir) => Self::open(&dir),
            Err(_) => Self { db: None },
        }
    }

    pub fn get(&self, key: &[u8; 16]) -> Option<Vec<u8>> {
        self.db.as_ref()?.get(key).ok().flatten()
    }
}

/// Place a stored wide friend (canonical wire space) into the window's global
/// wire space: canonical -> dense via `order`, dense -> global via
/// `used_wires` plus randomly drawn scratch wires. The XGate analogue of
/// `db_replace::friend_to_xgates`.
pub fn wide_friend_to_xgates(
    friend: &[XGate],
    order: &[u16],
    used_wires: &[u16],
    num_wires: usize,
    rng: &mut impl Rng,
) -> Option<Vec<XGate>> {
    if friend.is_empty() {
        return Some(Vec::new());
    }
    let canonical_slots = friend
        .iter()
        .flat_map(|g| std::iter::once(g.target).chain(g.ctrls.iter().map(|&(w, _)| w)))
        .max()
        .map_or(0, |w| w as usize + 1);
    let mut canonical_to_dense: Vec<u16> = order.to_vec();
    while canonical_to_dense.len() < canonical_slots {
        canonical_to_dense.push(canonical_to_dense.len() as u16);
    }

    let dense_slots = canonical_to_dense
        .iter()
        .take(canonical_slots)
        .map(|&d| d as usize + 1)
        .max()
        .unwrap_or(0)
        .max(canonical_slots);
    let mut dense_to_global: Vec<u16> = used_wires.to_vec();
    if dense_to_global.len() < dense_slots {
        let mut occupied = vec![false; num_wires];
        for &w in &dense_to_global {
            if (w as usize) < num_wires {
                occupied[w as usize] = true;
            } else {
                return None;
            }
        }
        let mut available: Vec<u16> = (0..num_wires as u16)
            .filter(|&w| !occupied[w as usize])
            .collect();
        available.shuffle(rng);
        let need = dense_slots - dense_to_global.len();
        if available.len() < need {
            return None;
        }
        dense_to_global.extend(available.into_iter().take(need));
    }

    let map = |w: u16| -> Option<u16> {
        let dense = *canonical_to_dense.get(w as usize)? as usize;
        dense_to_global.get(dense).copied()
    };
    let mut out = Vec::with_capacity(friend.len());
    for g in friend {
        let target = map(g.target)?;
        let mut ctrls = crate::circuit::xgate::Lits::new();
        for &(w, p) in &g.ctrls {
            let mw = map(w)?;
            if mw == target {
                return None; // corrupt friend: control on its own target
            }
            ctrls.push((mw, p));
        }
        crate::circuit::xgate::sort_lits(&mut ctrls);
        out.push(XGate {
            target,
            comp: g.comp,
            ctrls,
        });
    }
    Some(out)
}

/// Probe the wide store for `window` (forward frame — the store's keying) and
/// return every candidate PROVEN equivalent by ANF comparison.
pub fn wide_probe(
    window: &[XGate],
    num_wires: usize,
    db: &WideDb,
    budget: XPolyBudget,
    rng: &mut impl Rng,
) -> Vec<Vec<XGate>> {
    // Min-dir keying, mirroring the store's build convention: the key is the
    // smaller of the two directional canonical serializations. When the
    // REVERSED direction wins, the stored representative implements the
    // window's inverse and the placed friend must be reversed.
    let Ok(canon_f) = canonicalize_xgates_single(window, false, budget) else {
        return Vec::new();
    };
    let Ok(canon_r) = canonicalize_xgates_single(window, true, budget) else {
        return Vec::new();
    };
    let blob_f = polys_repr_blob(&canon_f.polys);
    let blob_r = polys_repr_blob(&canon_r.polys);
    let (canon, blob, matched_reversed) = if blob_r < blob_f {
        (canon_r, blob_r, true)
    } else {
        (canon_f, blob_f, false)
    };
    let key = xxh3_128(&blob).to_le_bytes();
    let Some(value) = db.get(&key) else {
        return Vec::new();
    };
    let Ok(friends) = mpx1::decode_value(&value) else {
        return Vec::new();
    };
    let order: Vec<u16> = canon.order.data.iter().map(|&d| d as u16).collect();
    let mut out = Vec::new();
    for friend in &friends {
        if let Some(mut placed) =
            wide_friend_to_xgates(friend, &order, &canon.used_wires, num_wires, rng)
        {
            if matched_reversed {
                placed.reverse();
            }
            if polys_equivalent(window, &placed, budget) == Some(true) {
                out.push(placed);
            }
        }
    }
    out
}

#[cfg(test)]
#[path = "../../tests/db_gen/support/wide_db/tests.rs"]
mod tests;
