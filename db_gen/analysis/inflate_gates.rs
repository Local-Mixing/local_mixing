//! Per-gate inflation: replace EVERY gate of a circuit with a random long
//! curated-store equivalent of that single gate (a true friend of the 1-gate
//! circuit). The output computes the same permutation, spelled entirely in
//! cdb vocabulary -- the "fattened" start state for store-ablation mixing
//! experiments (kills syntactic obstructions of the original spelling).
//!
//! Candidates come from the QC enumeration path (`qc_candidates_with`), so
//! each one is already proved functionally equivalent; the lookup closure
//! only serves the curated store. Key mode follows FROZEN_CURATED_KEY, as in
//! production (set `circuit` for the v3 true-friend stores).
//!
//! ```text
//! FROZEN_DB_DIR=... FROZEN_CURATED_DIR=... FROZEN_CURATED_KEY=circuit \
//!   inflate_gates IN.mpmct1 OUT.mpmct1 [--target-len 11] [--seed S]
//! ```

use local_mixing::canonicalization::xgate::XPolyBudget;
use local_mixing::circuit::formats::{read_mpmct, write_mpmct};
use local_mixing::database::frozen::FrozenDb;
use local_mixing::stages::db_mixing::replacement::{
    QC_MAX_BUCKET_BYTES, QcCandidateLimits, qc_candidates_with,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::error::Error;

type AnyResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

fn main() {
    if let Err(error) = run() {
        eprintln!("inflate_gates: {error}");
        std::process::exit(1);
    }
}

fn run() -> AnyResult<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: inflate_gates IN.mpmct1 OUT.mpmct1 [--target-len N] [--seed S]");
        std::process::exit(2);
    }
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse::<u64>().ok())
    };
    let target_len = flag("--target-len").unwrap_or(11) as usize;
    let seed = flag("--seed").unwrap_or_else(|| rand::rng().random());
    // --share F: chain blocks by wire reuse. Each block computes F ⊗ identity
    // on its scratch wires, so relabelling scratch (injective, avoiding the
    // block's own anchors) is function-preserving; we relabel so ~F of each
    // block's wires are drawn from the PREVIOUS block's wire set, making
    // cross-block windows wire-compact (the db-max-span cap is on wires).
    let share = args
        .iter()
        .position(|a| a == "--share")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);
    let mut rng = StdRng::seed_from_u64(seed);

    let (gates, num_wires) = read_mpmct(&args[1]).map_err(|e| format!("read {}: {e}", args[1]))?;
    let regular = std::env::var("FROZEN_DB_DIR").map_err(|_| "FROZEN_DB_DIR unset")?;
    let curated = std::env::var("FROZEN_CURATED_DIR").map_err(|_| "FROZEN_CURATED_DIR unset")?;
    let db = FrozenDb::open(&regular, Some(&curated));

    let budget = XPolyBudget {
        max_mul_terms: 1 << 16,
        max_poly_terms: 4096,
        max_total_terms: 16384,
    };
    let limits = QcCandidateLimits {
        max_candidates: 4096,
        max_gates: 32,
        max_support: 24,
    };

    let debug = flag("--debug").unwrap_or(0) as usize;
    let mut out = Vec::with_capacity(gates.len() * target_len);
    let mut hist = std::collections::BTreeMap::<usize, usize>::new();
    let mut kept = 0usize;
    let mut prev_wires: std::collections::HashSet<u16> = Default::default();
    let (mut shared_sum, mut wires_sum, mut blocks) = (0f64, 0usize, 0usize);
    // --arity K: partition into consecutive K-tuples and respell each tuple
    // AS A UNIT first (the original values interior to the tuple are then
    // never materialized -- arity erases (K-1)/K of the original waypoint
    // lattice at preprocessing time), then grow in-block to target_len.
    let arity = flag("--arity").unwrap_or(1) as usize;
    let mut tuple_keep = 0usize;
    // Single lookups return ~5-6-gate spellings (the store keeps a few
    // representatives per class), so target length is reached by COMPOSITION:
    // grow each gate's block by replacing a random gate inside it until the
    // block is ~target_len gates. Every step is a verified equivalent, so the
    // block always spells exactly its original gate's function.
    for (gi, chunk) in gates.chunks(arity.max(1)).enumerate() {
        let mut block = chunk.to_vec();
        let mut stalled = 0usize;
        if block.len() > 1 {
            // Whole-tuple respell: lookup the K-gate window as one function.
            let res =
                qc_candidates_with(&block, num_wires, budget, limits, &mut rng, |key, cur| {
                    if cur {
                        None
                    } else {
                        db.get_regular_qc(key, 65536, QC_MAX_BUCKET_BYTES)
                            .ok()
                            .flatten()
                    }
                });
            let pool: Vec<_> = res
                .candidates
                .iter()
                .filter(|c| !c.gates.is_empty() && c.gates.len() <= target_len + 2)
                .collect();
            if pool.is_empty() {
                tuple_keep += 1;
            } else {
                block = pool[rng.random_range(0..pool.len())].gates.clone();
            }
        }
        while block.len() < target_len && stalled < 12 {
            let at = rng.random_range(0..block.len());
            let window = [block[at].clone()];
            let res =
                qc_candidates_with(&window, num_wires, budget, limits, &mut rng, |key, cur| {
                    if cur {
                        None
                    } else {
                        db.get_regular_qc(key, 65536, QC_MAX_BUCKET_BYTES)
                            .ok()
                            .flatten()
                    }
                });
            let room = target_len + 2 - block.len();
            let pool: Vec<_> = res
                .candidates
                .iter()
                .filter(|c| c.gates.len() > 1 && c.gates.len() - 1 <= room)
                .collect();
            if pool.is_empty() {
                stalled += 1;
                continue;
            }
            let pick = pool[rng.random_range(0..pool.len())].gates.clone();
            block.splice(at..=at, pick);
        }
        if share > 0.0 {
            use rand::seq::SliceRandom;
            use std::collections::{HashMap, HashSet};
            let anchors: HashSet<u16> = chunk
                .iter()
                .flat_map(|g| std::iter::once(g.target).chain(g.ctrls.iter().map(|&(w, _)| w)))
                .collect();
            let wires_of = |gs: &[local_mixing::circuit::xgate::XGate]| -> HashSet<u16> {
                gs.iter()
                    .flat_map(|g| std::iter::once(g.target).chain(g.ctrls.iter().map(|&(w, _)| w)))
                    .collect()
            };
            if !prev_wires.is_empty() {
                let used = wires_of(&block);
                let mut scratch: Vec<u16> = used
                    .iter()
                    .copied()
                    .filter(|w| !anchors.contains(w))
                    .collect();
                scratch.shuffle(&mut rng);
                let want = ((used.len() as f64) * share).round() as usize;
                let already = anchors.iter().filter(|w| prev_wires.contains(w)).count();
                let mut pool: Vec<u16> = prev_wires
                    .iter()
                    .copied()
                    .filter(|w| !anchors.contains(w))
                    .collect();
                pool.shuffle(&mut rng);
                let need = want
                    .saturating_sub(already)
                    .min(scratch.len())
                    .min(pool.len());
                let mut map: HashMap<u16, u16> = HashMap::new();
                let mut taken: HashSet<u16> = anchors.clone();
                for i in 0..need {
                    map.insert(scratch[i], pool[i]);
                    taken.insert(pool[i]);
                }
                for &s in &scratch[need..] {
                    let mut fresh: Vec<u16> = (0..num_wires as u16)
                        .filter(|w| !taken.contains(w) && !prev_wires.contains(w))
                        .collect();
                    if fresh.is_empty() {
                        fresh = (0..num_wires as u16)
                            .filter(|w| !taken.contains(w))
                            .collect();
                    }
                    let pick = fresh[rng.random_range(0..fresh.len())];
                    map.insert(s, pick);
                    taken.insert(pick);
                }
                for g in &mut block {
                    if let Some(&nw) = map.get(&g.target) {
                        g.target = nw;
                    }
                    for c in &mut g.ctrls {
                        if let Some(&nw) = map.get(&c.0) {
                            c.0 = nw;
                        }
                    }
                }
                let new_wires = wires_of(&block);
                shared_sum +=
                    new_wires.intersection(&prev_wires).count() as f64 / new_wires.len() as f64;
                wires_sum += new_wires.len();
                blocks += 1;
                prev_wires = new_wires;
            } else {
                prev_wires = wires_of(&block);
            }
        }
        *hist.entry(block.len()).or_default() += 1;
        if block.len() == 1 {
            kept += 1;
        }
        if gi < debug {
            eprintln!("[debug g{gi}] block_len={}", block.len());
        }
        out.extend(block);
    }

    write_mpmct(&args[2], &out, num_wires).map_err(|e| format!("write {}: {e}", args[2]))?;
    let info = format!(
        "inflate_gates in={} target_len={} share={} arity={} tuple_keep={} seed={} gates {}->{} \
         kept_original={} len_hist={:?} mean_wires_per_block={:.1} mean_shared_frac={:.3}\n",
        args[1],
        target_len,
        share,
        arity,
        tuple_keep,
        seed,
        gates.len(),
        out.len(),
        kept,
        hist,
        wires_sum as f64 / blocks.max(1) as f64,
        shared_sum / blocks.max(1) as f64
    );
    std::fs::write(
        format!("{}.info", args[2].trim_end_matches(".mpmct1")),
        &info,
    )?;
    print!("{info}");
    Ok(())
}
