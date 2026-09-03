//! Blinded-V5 computation stage: an alternative to the drip `route_fire`
//! compute. Takes a circuit `A` on `n` wires and builds an equivalent circuit
//! on `2n` wires (data `0..n`, band `n..2n`) whose middle is a shuffled cloud
//! of locally-geodesic-identity (LGI) masks with `A`'s gates threaded through
//! it via a MASKED read. Only the compute changes; the surrounding pipeline
//! (slice guards, band fill, band rerand, final slice) is unchanged.
//!
//! Structure (RC spec, 2026-09-03):
//!   0. Band seed: each band wire `x_i & !x_j` from the honest active inputs.
//!   1. LGI scaffold: per data wire `w`, `u_w+1` K-cycle LGIs (u_w = uses of w
//!      in A). Each LGI is a K-cycle of g57 gates `g57(w,r_i,r_{i+1})`
//!      (`w ^= 1 ^ (!r_i & r_{i+1})`). g57 gates are the ASYMMETRIC OR form, so
//!      a gate and its reverse LINEARISE: `g57(w,r1,r2)+g57(w,r2,r1) = w+r1+r2`.
//!      All `2K*(3m+n)` gates commute (data targets, band controls).
//!   2. Order the scaffold with <= `max_open` LGIs open per wire.
//!   3. Sprinkle `rerand_level` band-refresh gates `b ^= data & aux`, placed by
//!      a SORTED left-to-right pull-forward filler (full dose, never blocks).
//!   4. Interleave A evenly. At each gate the MASKED read: LINEARISE each
//!      control (emit the reverse g57 of every net-open mask so the wire carries
//!      `operand + rho`, rho linear in band wires), then realise
//!      `c ^= comp ^ lit(a)&lit(b)` via `(a'+rho_a)(b'+rho_b)` using ONLY the
//!      masked control wires and band wires -- never a bare operand or `a^b` --
//!      then DE-LINEARISE. Single-control gates reduce to the linear part.
//!   5. No discipline pass.
//!
//! Verified EXHAUSTIVELY (all 2^n inputs x many band settings, k in 2..=5, all
//! max_open and rerand levels) in `scratchpad/v6` (compute_g57); n-independent.
//! K must be >= 2 (a g57 1-cycle is a degenerate constant flip).

use crate::circuit::xgate::XGate;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::{BTreeSet, HashMap, HashSet};

/// Configuration for [`gadgetize_blinded_v5`].
#[derive(Clone, Copy, Debug)]
pub struct BlindedV5Params {
    /// Control wires per LGI (mask cycle length). Must be `>= 2`.
    pub k: usize,
    /// Band pool size. `0` selects `r = np`.
    pub r: usize,
    /// Deterministic seed.
    pub seed: u64,
    /// Band-refresh dose. `0` = off.
    pub rerand_level: usize,
    /// Soft cap on simultaneously-open LGIs per wire (RC: `<= 3`).
    pub max_open: usize,
    /// Seed the band ONLY from data wires `0..active_wires`. `0` = all `np`.
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
    /// Band-refresh updates inserted (== `rerand_level`).
    pub rerand_done: usize,
    /// Effective band pool used.
    pub r_used: usize,
}

/// g57(w,x,y) = `w ^= 1 ^ (!x & y)`; `XGate::from_g57([w,x,y])` builds exactly
/// this (comp=1, monomial `!x & y`).
fn g57(w: u16, x: u16, y: u16) -> XGate {
    XGate::from_g57([w, x, y])
}

/// Recover `(w, (x,y))` from a scaffold g57 gate: data target, comp, one
/// negative + one positive band control. `x` = negative-polarity wire.
fn as_g57(g: &XGate, np: usize) -> Option<(usize, (u16, u16))> {
    if (g.target as usize) >= np || !g.comp || g.ctrls.len() != 2 {
        return None;
    }
    let (a0, p0) = g.ctrls[0];
    let (a1, p1) = g.ctrls[1];
    if (a0 as usize) < np || (a1 as usize) < np {
        return None;
    }
    match (p0, p1) {
        (false, true) => Some((g.target as usize, (a0, a1))),
        (true, false) => Some((g.target as usize, (a1, a0))),
        _ => None,
    }
}

fn cycle_g57(w: u16, cy: &[u16]) -> Vec<XGate> {
    let k = cy.len();
    (0..k).map(|i| g57(w, cy[i], cy[(i + 1) % k])).collect()
}

fn conj(t: u16, lits: &[(u16, bool)]) -> XGate {
    XGate::conj(t, lits.iter().copied()).expect("valid conj")
}

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

/// Linearise wire `w` masked by the net-open g57 ordered pairs `netopen`:
/// complete every net-open g57 into its reverse pair so `w = w_true ^ rho`.
/// Returns (rho band wires, the reverse gates emitted -- undo after the read).
fn linearize(w: u16, netopen: &[(u16, u16)], out: &mut Vec<XGate>) -> (Vec<u16>, Vec<XGate>) {
    let netset: HashSet<(u16, u16)> = netopen.iter().copied().collect();
    let mut added = Vec::new();
    for &(x, y) in netopen {
        if !netset.contains(&(y, x)) {
            let g = g57(w, y, x);
            out.push(g.clone());
            added.push(g);
        }
    }
    let mut cnt: HashMap<u16, usize> = HashMap::new();
    let mut seen: HashSet<(u16, u16)> = HashSet::new();
    for &(x, y) in netopen {
        let key = (x.min(y), x.max(y));
        if seen.insert(key) {
            *cnt.entry(x).or_insert(0) += 1;
            *cnt.entry(y).or_insert(0) += 1;
        }
    }
    let mut rho: Vec<u16> = cnt
        .iter()
        .filter(|&(_, &c)| c % 2 == 1)
        .map(|(&k, _)| k)
        .collect();
    rho.sort_unstable();
    (rho, added)
}

/// Emit `c ^= comp ^ prod(lit(w_i,p_i))` for <=2 (already linearised) controls
/// `(wire, pol, rho)`, using only the masked control wires and band wires.
fn masked_fire(c: u16, ctrls: &[(u16, bool, Vec<u16>)], comp: bool, out: &mut Vec<XGate>) {
    match ctrls.len() {
        0 => {
            if comp {
                out.push(XGate::x_gate(c));
            }
        }
        1 => {
            let (w1, p1, ra) = (ctrls[0].0, ctrls[0].1, &ctrls[0].2);
            out.push(conj(c, &[(w1, true)]));
            for &s in ra {
                out.push(conj(c, &[(s, true)]));
            }
            if comp ^ !p1 {
                out.push(XGate::x_gate(c));
            }
        }
        2 => {
            let (w1, p1, ra) = (ctrls[0].0, ctrls[0].1, &ctrls[0].2);
            let (w2, p2, rb) = (ctrls[1].0, ctrls[1].1, &ctrls[1].2);
            let (ca, cb) = (!p1, !p2);
            out.push(conj(c, &[(w1, true), (w2, true)]));
            for &r in rb {
                out.push(conj(c, &[(w1, true), (r, true)]));
            }
            if cb {
                out.push(conj(c, &[(w1, true)]));
            }
            for &s in ra {
                out.push(conj(c, &[(s, true), (w2, true)]));
            }
            for &s in ra {
                for &r in rb {
                    out.push(conj(c, &[(s, true), (r, true)]));
                }
            }
            if cb {
                for &s in ra {
                    out.push(conj(c, &[(s, true)]));
                }
            }
            if ca {
                out.push(conj(c, &[(w2, true)]));
            }
            if ca {
                for &r in rb {
                    out.push(conj(c, &[(r, true)]));
                }
            }
            if comp ^ (ca & cb) {
                out.push(XGate::x_gate(c));
            }
        }
        _ => panic!("masked read supports <= 2 controls"),
    }
}

/// Gadgetize `A` (`src`, on wires `0..np`) into a `np + r`-wire circuit with the
/// masked read. The `np` data wires end holding `A`'s output; the band wires are
/// seeded from the input and left dirty.
pub fn gadgetize_blinded_v5(src: &[XGate], np: usize, p: &BlindedV5Params) -> BlindedV5Output {
    let k = p.k.max(2); // g57 1-cycle is a degenerate constant flip
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

    // 0. band seed
    for &aw in &band {
        let i1 = rng.random_range(0..active) as u16;
        let mut i2 = rng.random_range(0..active) as u16;
        while i2 == i1 {
            i2 = rng.random_range(0..active) as u16;
        }
        out.push(conj(aw, &[(i1, true), (i2, false)]));
    }

    // 1. LGI pool
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

    // 2. scaffold, rolling <= max_open per wire, merged
    let mut streams: Vec<Vec<XGate>> = vec![Vec::new(); np];
    for w in 0..np {
        let cys = &pool[w];
        let mut open: Vec<usize> = Vec::new();
        for i in 0..cys.len() {
            streams[w].extend(cycle_g57(w as u16, &cys[i]));
            open.push(i);
            while open.len() > max_open {
                let old = open.remove(0);
                streams[w].extend(cycle_g57(w as u16, &cys[old]));
            }
        }
        for &i in &open {
            streams[w].extend(cycle_g57(w as u16, &cys[i]));
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

    // 3. rerand: sorted pull-forward filler
    let mut plan: Vec<(usize, XGate, u16)> = (0..p.rerand_level)
        .map(|_| {
            let pos = rng.random_range(0..scaffold.len().max(1));
            let b = band[rng.random_range(0..band.len())];
            let d = rng.random_range(0..np) as u16;
            let mut c = band[rng.random_range(0..band.len())];
            while c == b {
                c = band[rng.random_range(0..band.len())];
            }
            (pos, conj(b, &[(d, rng.random_bool(0.5)), (c, rng.random_bool(0.5))]), b)
        })
        .collect();
    plan.sort_by_key(|x| x.0);
    let mut rerand_done = 0usize;
    if !plan.is_empty() {
        let mut consumed = vec![false; scaffold.len()];
        let mut built = Vec::with_capacity(scaffold.len() + plan.len());
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

    // 4. interleave A with the masked read
    let m = src.len();
    let mut pos: Vec<usize> = (0..m)
        .map(|j| ((j as u64 * scaffold.len() as u64) / (m.max(1) as u64)) as usize)
        .collect();
    pos.sort_unstable();
    let mut open_g57: Vec<BTreeSet<(u16, u16)>> = vec![BTreeSet::new(); np];
    let mut ai = 0usize;
    for idx in 0..=scaffold.len() {
        while ai < m && pos[ai] == idx {
            let g = &src[ai];
            ai += 1;
            let mut ctrls: Vec<(u16, bool, Vec<u16>)> = Vec::new();
            let mut undo: Vec<XGate> = Vec::new();
            for &(w, pol) in &g.ctrls {
                let netopen: Vec<(u16, u16)> = open_g57[w as usize].iter().copied().collect();
                let (rho, added) = linearize(w, &netopen, &mut out);
                for gg in added.into_iter().rev() {
                    undo.push(gg);
                }
                ctrls.push((w, pol, rho));
            }
            masked_fire(g.target, &ctrls, g.comp, &mut out);
            for gg in undo {
                out.push(gg);
            }
        }
        if idx < scaffold.len() {
            let g = &scaffold[idx];
            if let Some((w, pair)) = as_g57(g, np) {
                if !open_g57[w].insert(pair) {
                    open_g57[w].remove(&pair);
                }
            }
            out.push(g.clone());
        }
    }
    debug_assert!(ai == m);

    BlindedV5Output {
        gates: out,
        num_wires: total,
        atoms,
        rerand_done,
        r_used: r,
    }
}
