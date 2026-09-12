//! V3 of the wide-gate design — THE GO/NO-GO MEASUREMENT.
//!
//! Enumerates all 32 concrete m1-wide candidates (single 3-control
//! conjunction gates on a 4-wire universe: 4 targets x 8 polarity patterns),
//! canonicalizes each, and reports the distinct function keys (predicted: 4,
//! by S3 control-relabeling symmetry — verified here, not assumed). Also
//! cross-checks that `wide_gates_for_circuit_filtered`'s class
//! representatives reach exactly the same key set.
//!
//! With `FROZEN_DB_DIR` set, additionally probes the regular (and optional
//! curated) store for every distinct key in both direction frames and
//! reports the yield: how many of these functions are NOT already reachable
//! in the existing g57-only store. Without the var, prints keys only.
use local_mixing::circuit::polys_repr_blob;
use local_mixing::circuit::xgate::XGate;
use local_mixing::db_generation::wide_gates::wide_gates_for_circuit_filtered;
use local_mixing::db_mixing::frozen::FrozenDb;
use local_mixing::engine::xpoly::{XPolyBudget, canonicalize_xgates_single};
use std::collections::BTreeMap;
use xxhash_rust::xxh3::xxh3_128;

fn key_for(g: &XGate, reversed: bool, budget: XPolyBudget) -> [u8; 16] {
    let canon = canonicalize_xgates_single(std::slice::from_ref(g), reversed, budget)
        .expect("single wide gate canonicalizes");
    xxh3_128(&polys_repr_blob(&canon.polys)).to_le_bytes()
}

fn main() {
    let budget = XPolyBudget::default();

    // All 32 concrete candidates on wires {0,1,2,3}.
    let mut concrete = Vec::new();
    for t in 0..4u16 {
        let ctrls: Vec<u16> = (0..4u16).filter(|&w| w != t).collect();
        for pols in 0..8u8 {
            let g = XGate::conj(
                t,
                [
                    (ctrls[0], pols & 1 != 0),
                    (ctrls[1], pols & 2 != 0),
                    (ctrls[2], pols & 4 != 0),
                ],
            )
            .expect("distinct wires");
            concrete.push(g);
        }
    }
    assert_eq!(concrete.len(), 32);

    // Group by forward key.
    let mut classes: BTreeMap<[u8; 16], (usize, XGate)> = BTreeMap::new();
    for g in &concrete {
        let k = key_for(g, false, budget);
        classes
            .entry(k)
            .and_modify(|e| e.0 += 1)
            .or_insert((1, g.clone()));
    }
    println!(
        "m1-wide: 32 concrete candidates -> {} distinct forward keys",
        classes.len()
    );
    for (k, (count, rep)) in &classes {
        println!(
            "  key {}  members {:2}  rep target={} ctrls={:?} ",
            hex(k),
            count,
            rep.target,
            rep.ctrls
        );
    }

    // Cross-check: the S5 enumerator's representatives cover the same keys.
    let (reps, _skipped) = wide_gates_for_circuit_filtered(&[], 4, 0, 0);
    let rep_keys: std::collections::BTreeSet<[u8; 16]> =
        reps.iter().map(|g| key_for(g, false, budget)).collect();
    let all_keys: std::collections::BTreeSet<[u8; 16]> = classes.keys().copied().collect();
    assert_eq!(
        rep_keys, all_keys,
        "S5 representatives must reach exactly the concrete key set"
    );
    println!(
        "S5 cross-check: {} representatives cover all {} keys exactly",
        reps.len(),
        all_keys.len()
    );

    if std::env::var("FROZEN_DB_DIR").is_err() {
        println!("FROZEN_DB_DIR unset: key report only (run near a store for the yield).");
        return;
    }
    let db = FrozenDb::from_env();
    let mut new_keys = 0usize;
    for (k, (_, rep)) in &classes {
        let krev = key_for(rep, true, budget);
        let hit_fwd_reg = db.get_regular(k).is_some();
        let hit_rev_reg = db.get_regular(&krev).is_some();
        let hit_fwd_cur = db.get_curated(k).is_some();
        let hit_rev_cur = db.get_curated(&krev).is_some();
        let hit = hit_fwd_reg || hit_rev_reg || hit_fwd_cur || hit_rev_cur;
        if !hit {
            new_keys += 1;
        }
        println!(
            "  key {}  regular fwd/rev {}/{}  curated fwd/rev {}/{}  => {}",
            hex(k),
            hit_fwd_reg as u8,
            hit_rev_reg as u8,
            hit_fwd_cur as u8,
            hit_rev_cur as u8,
            if hit { "KNOWN" } else { "NEW" }
        );
    }
    println!(
        "V3 YIELD: {new_keys} NEW of {} distinct m1-wide functions",
        classes.len()
    );
}

fn hex(k: &[u8; 16]) -> String {
    k.iter().map(|b| format!("{b:02x}")).collect()
}
