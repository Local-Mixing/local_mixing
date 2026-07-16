// Integration check: corrected-convention polynomial keys vs the (regenerated) frozen DB.
//
// For random m-gate windows this verifies three things end to end:
//   1. HIT RATE — canonicalize_polys_single (corrected to_polynomial) -> polys_repr_blob ->
//      xxh3_128 keys actually hit the frozen shard/curated stores (fwd, then rev fallback,
//      mirroring pairs.rs Legacy lookup).
//   2. FRAGMENT KEYS — expanding the window into cube-gate fragments, transporting them
//      (commute walk + splits + recombine), then keying via fragment_polys +
//      canonicalize_polys_4 yields the *identical* canonical polynomials and key. This is the
//      precondition for DB-backed fragment reassembly.
//   3. VALUE CONVENTION — friends decoded from hit values, re-keyed through the same
//      corrected pipeline, land back on the lookup key (stored circuits match the corrected
//      convention).
//
// Usage (frozen dirs from env, filter optional):
//   FROZEN_DB_DIR=~/frozen_m1_m11 FROZEN_CURATED_DIR=~/frozen_curated_m1_m11 \
//     cargo run --release --bin frozen_key_check [probes_per_config]
//
// Exits nonzero on any key/value mismatch; hit rates are reported, not asserted.

use local_mixing::circuit::circuit::{
    CircuitSeq, Polynomial, canonicalize_polys_4, polys_repr_blob,
};
use local_mixing::random::random_data::random_circuit;
use local_mixing::replace::fragment::{
    CubeGate, commute, expand_g57_circuit, fragment_polys, recombine, split,
};
use local_mixing::replace::frozen::FrozenDb;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use xxhash_rust::xxh3::xxh3_128;

fn key_of(polys: &Vec<Polynomial>) -> [u8; 16] {
    xxh3_128(&polys_repr_blob(polys)).to_le_bytes()
}

/// Random equivalence-preserving fragment transport within k dense wires:
/// commute walk, a few splits on free pivots, more commuting, then recombine.
fn transport(mut frags: Vec<CubeGate>, k: usize, rng: &mut StdRng) -> Vec<CubeGate> {
    let cap = k.saturating_sub(1);
    for phase in 0..2 {
        for _ in 0..300 {
            if frags.len() < 2 {
                break;
            }
            let i = rng.random_range(0..frags.len() - 1);
            if commute(&frags[i], &frags[i + 1]) {
                frags.swap(i, i + 1);
            }
        }
        if phase == 0 {
            for _ in 0..4 {
                let i = rng.random_range(0..frags.len());
                if frags[i].arity() >= cap {
                    continue;
                }
                let free: Vec<u16> = (0..k as u16)
                    .filter(|&w| {
                        w != frags[i].target && !frags[i].lits.iter().any(|&(x, _)| x == w)
                    })
                    .collect();
                if free.is_empty() {
                    continue;
                }
                let p = free[rng.random_range(0..free.len())];
                let (lo, hi) = split(&frags[i], p);
                frags[i] = lo;
                frags.insert(i + 1, hi);
            }
        }
    }
    recombine(frags)
}

fn main() {
    let probes: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(400);
    let db = FrozenDb::from_env();
    let mut rng = StdRng::seed_from_u64(0xf90ce9);
    let mut frag_key_mismatch = 0usize;
    let mut value_rekey_bad = 0usize;
    let mut value_rekey_ok = 0usize;

    println!(
        "{:>2} {:>3} | {:>6} {:>8} {:>8} {:>8} | {:>9} {:>9}",
        "m", "n", "probes", "cur_hit", "fwd_hit", "any_hit", "frag_ok", "val_ok"
    );
    for &(m, n) in &[
        (1usize, 3usize),
        (2, 4),
        (3, 5),
        (4, 6),
        (5, 7),
        (6, 8),
        (7, 9),
        (8, 10),
    ] {
        let (mut cur_hits, mut fwd_hits, mut any_hits, mut frag_ok, mut val_checked) =
            (0usize, 0usize, 0usize, 0usize, 0usize);
        let mut probed = 0usize;
        for _ in 0..probes {
            let c = random_circuit(n, m);
            let (fwd_polys, _ord, used) = c.canonicalize_polys_single(false);
            if fwd_polys.is_empty() {
                continue;
            }
            probed += 1;
            let fwd_key = key_of(&fwd_polys);

            // --- 1. hit rate (curated fwd; shard fwd; shard rev fallback) ---
            let cur_v = db.get_curated(&fwd_key);
            let fwd_v = db.get_regular(&fwd_key);
            let mut any_v = cur_v.clone().or_else(|| fwd_v.clone());
            if any_v.is_none() {
                let (rev_polys, _ro, _) = c.canonicalize_polys_single(true);
                if !rev_polys.is_empty() {
                    any_v = db.get_regular(&key_of(&rev_polys));
                }
            }
            cur_hits += cur_v.is_some() as usize;
            fwd_hits += fwd_v.is_some() as usize;
            any_hits += any_v.is_some() as usize;

            // --- 2. fragment-path key equality ---
            // Remap to dense wires exactly like canonicalize_polys_single, expand, transport,
            // rebuild polynomials from fragments, canonicalize: must equal fwd_polys.
            let k = used.len();
            let wire_map: std::collections::HashMap<u16, u16> = used
                .iter()
                .enumerate()
                .map(|(i, &w)| (w, i as u16))
                .collect();
            let dense = CircuitSeq {
                gates: c
                    .gates
                    .iter()
                    .map(|&[a, b, cc]| [wire_map[&a], wire_map[&b], wire_map[&cc]])
                    .collect(),
            };
            let frags = transport(expand_g57_circuit(&dense.gates), k, &mut rng);
            let fp = fragment_polys(&frags, k);
            match canonicalize_polys_4(fp, true) {
                Ok((frag_canon, _)) if frag_canon == fwd_polys => frag_ok += 1,
                _ => {
                    frag_key_mismatch += 1;
                    if frag_key_mismatch <= 3 {
                        eprintln!("FRAG KEY MISMATCH: m={m} n={n} gates={:?}", c.gates);
                    }
                }
            }

            // --- 3. value convention: re-key decoded friends (cap the work) ---
            if let Some(v) = any_v {
                if val_checked < 40 {
                    val_checked += 1;
                    let mut pos = 0usize;
                    let mut checked_friend = false;
                    while pos < v.len() && !checked_friend {
                        let len = v[pos] as usize;
                        pos += 1;
                        if pos + len > v.len() {
                            break;
                        }
                        let friend = CircuitSeq::from_blob(&v[pos..pos + len]);
                        pos += len;
                        if friend.gates.is_empty() {
                            continue;
                        }
                        checked_friend = true;
                        let (fr_polys, _, _) = friend.canonicalize_polys_single(false);
                        let fr_fwd = key_of(&fr_polys);
                        let mut ok = fr_fwd == fwd_key;
                        if !ok {
                            let (fr_rev, _, _) = friend.canonicalize_polys_single(true);
                            if !fr_rev.is_empty() {
                                ok = key_of(&fr_rev) == fwd_key;
                            }
                        }
                        if ok {
                            value_rekey_ok += 1;
                        } else {
                            value_rekey_bad += 1;
                            if value_rekey_bad <= 3 {
                                eprintln!(
                                    "VALUE REKEY MISMATCH: m={m} n={n} friend={:?}",
                                    friend.gates
                                );
                            }
                        }
                    }
                }
            }
        }
        println!(
            "{:>2} {:>3} | {:>6} {:>7.1}% {:>7.1}% {:>7.1}% | {:>8.1}% {:>9}",
            m,
            n,
            probed,
            100.0 * cur_hits as f64 / probed.max(1) as f64,
            100.0 * fwd_hits as f64 / probed.max(1) as f64,
            100.0 * any_hits as f64 / probed.max(1) as f64,
            100.0 * frag_ok as f64 / probed.max(1) as f64,
            format!("{}/{}", value_rekey_ok, value_rekey_ok + value_rekey_bad),
        );
    }

    println!(
        "\nfragment-key mismatches: {frag_key_mismatch}; value re-key failures: {value_rekey_bad}"
    );
    if frag_key_mismatch > 0 || value_rekey_bad > 0 {
        std::process::exit(1);
    }
}
