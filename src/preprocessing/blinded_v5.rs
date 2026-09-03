//! Blinded-V5 computation stage: an alternative to the drip `route_fire`
//! compute. Takes a circuit `A` on `n` wires and builds an equivalent circuit
//! on `2n` wires (data `0..n`, band `n..2n`) whose middle is a shuffled cloud
//! of locally-geodesic-identity (LGI) masks with `A`'s gates threaded through
//! it. Only the compute changes; the surrounding pipeline (slice guards, band
//! fill, band rerand, final slice) is unchanged.
//!
//! Structure (RC spec, 2026-09-03):
//!   0. Band seed: each band wire `x_i & !x_j` from the honest active inputs
//!      (this is the "band wire seeding module"; the compute then only READS
//!      the band).
//!   1. LGI scaffold: for each data wire `w`, `u_w+1` K-cycle LGIs, where
//!      `u_w` = number of uses of `w` in `A` (as control or target). Each LGI
//!      is a K-cycle `w ^= sum_i r_i & r_{i+1}` (apply) and its removal; all
//!      `2K*(3m+n)` gates commute (data targets, band controls).
//!   2. Order the scaffold with <= `max_open` LGIs open per wire.
//!   3. Sprinkle `rerand_level` band-refresh gates `b ^= data & aux`, placed by
//!      the SORTED left-to-right filler: at each pre-chosen position pull every
//!      still-un-emitted reader of `b` forward (they lie in the untouched,
//!      all-commuting remainder, so it never blocks) then emit the rerand -- so
//!      no open mask straddles it and the full dose is always achieved.
//!   4. Interleave `A` evenly; at each gate unmask its controls (re-emit their
//!      open monomials), fire, remask. (Plain read; the linear-mask blinded
//!      read is a separate step.)
//!   5. No discipline pass.
//!
//! Correctness is verified EXHAUSTIVELY (all 2^n inputs, many band settings)
//! for small n in `scratchpad/v6`; the logic is n-independent.

use crate::circuit::xgate::XGate;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BTreeSet;

/// Configuration for [`gadgetize_blinded_v5`].
#[derive(Clone, Copy, Debug)]
pub struct BlindedV5Params {
    /// Control wires per LGI (mask cycle length).
    pub k: usize,
    /// Band pool size. `0` selects `r = np` (band as wide as the data).
    pub r: usize,
    /// Deterministic seed.
    pub seed: u64,
    /// Band-refresh dose (`b ^= data & aux` updates). `0` = off.
    pub rerand_level: usize,
    /// Soft cap on simultaneously-open LGIs per wire (RC: `<= 3`).
    pub max_open: usize,
    /// Seed the band ONLY from data wires `0..active_wires` (the wires carrying
    /// live input on the honest distribution). `0` (or `>= np`) means all `np`.
    /// For a 2n-wire sliced sandwich used on its zero slice, set this to `n`.
    pub active_wires: usize,
}

impl BlindedV5Params {
    /// Settled preset: K=16, R=n (auto), max_open=3, no rerand.
    pub fn production(seed: u64) -> Self {
        Self {
            k: 16,
            r: 0,
            seed,
            rerand_level: 0,
            max_open: 3,
            active_wires: 0,
        }
    }
}

/// Result of [`gadgetize_blinded_v5`].
pub struct BlindedV5Output {
    /// The gadgetized circuit.
    pub gates: Vec<XGate>,
    /// Total wire count: `np + r`.
    pub num_wires: usize,
    /// LGI atoms laid down (`3m + n`).
    pub atoms: usize,
    /// Band-refresh updates inserted (== `rerand_level`; the sorted filler
    /// always achieves the dose).
    pub rerand_done: usize,
    /// Effective band pool used (`r`, resolved from `0`-means-`np`).
    pub r_used: usize,
}

/// K-cycle monomials (band pairs) of a cycle on ancillas `cy`: gate `i` is
/// `w ^= cy[i] & cy[i+1]`. A wire `b = cy[j]` appears in gates `j` and `j-1`.
fn cycle_pairs(cy: &[u16]) -> Vec<(u16, u16)> {
    let k = cy.len();
    (0..k).map(|i| (cy[i], cy[(i + 1) % k])).collect()
}

/// One monomial gate `w ^= bi & bj` (dedups to a CNOT when bi == bj, i.e. K=1).
fn mono(w: u16, bi: u16, bj: u16) -> XGate {
    XGate::conj(w, [(bi, true), (bj, true)]).expect("distinct target/controls")
}

/// Random K-subset of `pool` (partial Fisher-Yates).
fn sample_k(pool: &[u16], k: usize, rng: &mut StdRng) -> Vec<u16> {
    let mut v = pool.to_vec();
    let n = v.len();
    for i in 0..k {
        let j = i + rng.random_range(0..(n - i));
        v.swap(i, j);
    }
    v.truncate(k);
    v
}

/// Recover a scaffold cycle monomial `(target, (bi, bj))` from a gate: target a
/// data wire (`< np`), positive band control(s). K>=2 gives two distinct band
/// controls; K=1 dedups to one, canonicalised as `(b, b)`.
fn as_cycle(g: &XGate, np: usize) -> Option<(usize, (u16, u16))> {
    if (g.target as usize) >= np || g.comp {
        return None;
    }
    match g.ctrls.as_slice() {
        [(a0, true), (a1, true)] if (*a0 as usize) >= np && (*a1 as usize) >= np => {
            Some((g.target as usize, (*a0, *a1)))
        }
        [(a0, true)] if (*a0 as usize) >= np => Some((g.target as usize, (*a0, *a0))),
        _ => None,
    }
}

/// Gadgetize `A` (`src`, on wires `0..np`) into a `np + r`-wire circuit. See the
/// module docs. The `np` data wires end holding `A`'s output; the band wires
/// are seeded from the input and left dirty (the slice stages clear them).
pub fn gadgetize_blinded_v5(src: &[XGate], np: usize, p: &BlindedV5Params) -> BlindedV5Output {
    let k = p.k.max(1);
    let r = if p.r == 0 { np } else { p.r };
    assert!(r >= k, "R must be >= K");
    assert!(np >= 2, "need at least two data wires");
    let active = if p.active_wires == 0 || p.active_wires > np {
        np
    } else {
        p.active_wires
    };
    let total = np + r;
    assert!(total < u16::MAX as usize, "too many wires");
    let band: Vec<u16> = (np as u16..total as u16).collect();
    let max_open = p.max_open.max(1);
    let mut rng = StdRng::seed_from_u64(p.seed);
    let mut out: Vec<XGate> = Vec::new();

    // ---- 0. Band seed (the band wire seeding module) ---------------------
    for &aw in &band {
        let i1 = rng.random_range(0..active) as u16;
        let mut i2 = rng.random_range(0..active) as u16;
        while i2 == i1 {
            i2 = rng.random_range(0..active) as u16;
        }
        out.push(XGate::conj(aw, [(i1, true), (i2, false)]).unwrap());
    }

    // ---- 1. LGI pool: u_w+1 cycles per data wire -------------------------
    let mut u = vec![0usize; np];
    for g in src {
        u[g.target as usize] += 1;
        for &(w, _) in &g.ctrls {
            if (w as usize) < np {
                u[w as usize] += 1;
            }
        }
    }
    let atoms: usize = u.iter().map(|&c| c + 1).sum();
    let mut pool: Vec<Vec<Vec<u16>>> = vec![Vec::new(); np];
    for w in 0..np {
        for _ in 0..(u[w] + 1) {
            pool[w].push(sample_k(&band, k, &mut rng));
        }
    }

    // ---- 2. Pure scaffold, rolling <= max_open open per wire, merged -----
    let mut streams: Vec<Vec<XGate>> = vec![Vec::new(); np];
    for w in 0..np {
        let cys = &pool[w];
        let mut open: Vec<usize> = Vec::new();
        for i in 0..cys.len() {
            for (bi, bj) in cycle_pairs(&cys[i]) {
                streams[w].push(mono(w as u16, bi, bj));
            }
            open.push(i);
            while open.len() > max_open {
                let old = open.remove(0);
                for (bi, bj) in cycle_pairs(&cys[old]) {
                    streams[w].push(mono(w as u16, bi, bj));
                }
            }
        }
        for &i in &open {
            for (bi, bj) in cycle_pairs(&cys[i]) {
                streams[w].push(mono(w as u16, bi, bj));
            }
        }
    }
    let mut ptr = vec![0usize; np];
    let mut remaining: usize = streams.iter().map(|s| s.len()).sum();
    let mut scaffold: Vec<XGate> = Vec::with_capacity(remaining);
    while remaining > 0 {
        let mut pick = rng.random_range(0..remaining);
        let mut w = 0;
        loop {
            let left = streams[w].len() - ptr[w];
            if pick < left {
                break;
            }
            pick -= left;
            w += 1;
        }
        scaffold.push(streams[w][ptr[w]].clone());
        ptr[w] += 1;
        remaining -= 1;
    }

    // ---- 3. Rerand: sorted, left-to-right pull-forward filler ------------
    let mut plan: Vec<(usize, XGate, u16)> = (0..p.rerand_level)
        .map(|_| {
            let pos = rng.random_range(0..scaffold.len().max(1));
            let b = band[rng.random_range(0..band.len())];
            let d = rng.random_range(0..np) as u16;
            let mut c = band[rng.random_range(0..band.len())];
            while c == b {
                c = band[rng.random_range(0..band.len())];
            }
            let cand =
                XGate::conj(b, [(d, rng.random_bool(0.5)), (c, rng.random_bool(0.5))]).unwrap();
            (pos, cand, b)
        })
        .collect();
    plan.sort_by_key(|x| x.0);
    let mut rerand_done = 0usize;
    if !plan.is_empty() {
        let mut consumed = vec![false; scaffold.len()];
        let mut built: Vec<XGate> = Vec::with_capacity(scaffold.len() + plan.len());
        let mut cursor = 0usize;
        for (pos, cand, b) in &plan {
            while cursor < *pos {
                if !consumed[cursor] {
                    built.push(scaffold[cursor].clone());
                    consumed[cursor] = true;
                }
                cursor += 1;
            }
            for j in cursor..scaffold.len() {
                if !consumed[j] && scaffold[j].reads(*b) {
                    built.push(scaffold[j].clone());
                    consumed[j] = true;
                }
            }
            built.push(cand.clone());
            rerand_done += 1;
        }
        for j in cursor..scaffold.len() {
            if !consumed[j] {
                built.push(scaffold[j].clone());
            }
        }
        scaffold = built;
    }

    // ---- 4. Interleave A, tracking OPEN monomials per wire ---------------
    // A-gate j is placed after scaffold position pos[j] (even spread).
    let m = src.len();
    let mut pos: Vec<usize> = (0..m)
        .map(|j| ((j as u64 * scaffold.len() as u64) / (m.max(1) as u64)) as usize)
        .collect();
    pos.sort_unstable();

    let mut open_mono: Vec<BTreeSet<(u16, u16)>> = vec![BTreeSet::new(); np];
    let mut ai = 0usize;
    for idx in 0..=scaffold.len() {
        while ai < m && pos[ai] == idx {
            let g = &src[ai];
            ai += 1;
            let ctrls: Vec<u16> = g.ctrls.iter().map(|&(w, _)| w).collect();
            for &c in &ctrls {
                for &(bi, bj) in &open_mono[c as usize] {
                    out.push(mono(c, bi, bj)); // unmask -> true control
                }
            }
            out.push(g.clone()); // fire
            for &c in &ctrls {
                for &(bi, bj) in &open_mono[c as usize] {
                    out.push(mono(c, bi, bj)); // remask
                }
            }
        }
        if idx < scaffold.len() {
            let g = &scaffold[idx];
            if let Some((w, pair)) = as_cycle(g, np) {
                if !open_mono[w].insert(pair) {
                    open_mono[w].remove(&pair);
                }
            }
            out.push(g.clone());
        }
    }
    debug_assert!(ai == m, "not all A-gates were placed");

    BlindedV5Output {
        gates: out,
        num_wires: total,
        atoms,
        rerand_done,
        r_used: r,
    }
}
