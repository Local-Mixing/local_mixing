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
//!   3. Sprinkle band-refresh gates `b ^= lit & lit` (half with a live-data
//!      control, half band-only) at sorted positions, in two kinds: `rerand_level`
//!      STRADDLE rerands (close the masks straddling the update -- thins masking
//!      past a ~1024 knee) and `rerand_repair` REPAIR rerands (re-derive each
//!      straddling mask across the update so it stays open -- no thinning).
//!   4. Place each A-gate STRADDLING an existing scaffold LGI on its active wire
//!      (hidden-firing fix). An atomic, mask-restoring gate module would leak
//!      which gate fired: the active wire's before/after XOR across it equals the
//!      true gate increment. So each A-gate is emitted with its fire split around
//!      one scaffold g57 on its target wire `c` (its dependency-respecting slot):
//!      that LGI toggles `c`'s mask mid-fire, so the module's net XOR on `c` is
//!      `Delta ^ (that LGI's secret band-mask)`, never a bare increment. These
//!      B-gates are already present (rerand-protected) -- no injected masks, no
//!      repair, no size cost. At each gate the
//!      MASKED read: LINEARISE each control (emit the reverse g57 of every
//!      net-open mask so the wire carries `operand + rho`, rho linear in band
//!      wires), then realise `c ^= comp ^ lit(a)&lit(b)` via
//!      `(a'+rho_a)(b'+rho_b)` using ONLY the masked control wires and band wires
//!      -- never a bare operand or `a^b` -- then DE-LINEARISE. Single-control
//!      gates reduce to the linear part.
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
    /// STRADDLE-only band-refresh dose (total; split half data-control, half
    /// band-only). Each such rerand CLOSES the masks straddling it, so heavy
    /// doses thin the masking (knee ~1024). `0` = off.
    pub rerand_level: usize,
    /// REPAIR band-refresh dose (total; split half data-control, half band-only).
    /// A repair rerand re-derives every straddling mask across the update (old-b
    /// cancels, new-b re-adds), so masks STAY open -- no thinning. Stack these on
    /// top of the straddle dose for extra band turnover. `0` = off.
    pub rerand_repair: usize,
    /// Soft cap on simultaneously-open LGIs per wire (RC: `<= 3`).
    pub max_open: usize,
    /// Seed the band ONLY from data wires `0..active_wires`. `0` = all `np`.
    pub active_wires: usize,
    /// Extra LGIs per wire beyond `u_w+1`. Adds straddle SLOTS for the
    /// hidden-firing fix (each A-gate is placed straddling a scaffold LGI on its
    /// active wire); more slots => fewer gates left with their firing exposed.
    pub extra_lgis: usize,
}

impl BlindedV5Params {
    /// Settled preset: K=2 (band wires per LGI -> 1 disjoint pair; affine- and
    /// deg-2-neutral across K, so smallest is best), R=n (auto), max_open=3.
    /// Rerand: 1K STRADDLE (at the safe knee -- straddle-only thins masking past
    /// ~1024) topped off with 3K REPAIR (repair doesn't thin, so it adds band
    /// turnover cheaply), each split half data-control / half band-only (RC,
    /// 2026-09-03). `extra_lgis` adds straddle slots for the hidden-firing fix.
    pub fn production(seed: u64) -> Self {
        Self {
            k: 2,
            r: 0,
            seed,
            rerand_level: 1000,
            rerand_repair: 3000,
            max_open: 3,
            active_wires: 0,
            extra_lgis: 0,
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
    /// Band-refresh updates inserted (straddle + repair).
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

/// One LGI on wire `w`: DISJOINT g57 pairs over `cy`, i.e. `g57(w,cy[2i],cy[2i+1])`
/// for each pair. A cycle would telescope to 0 under the read's pair-completion
/// (`Σ cy[i]⊕cy[i+1]` around a cycle = 0 -> a BARE operand); disjoint pairs
/// linearise to a nonzero `Σ cy[2i]⊕cy[2i+1]` instead. The deg-2 masking each
/// gate `1⊕(¬cy[2i]∧cy[2i+1])` gives is disjoint deg-2 piling-up (the optimal
/// sparse-mask shape). Needs `k >= 2`; an odd trailing wire is dropped.
fn cycle_g57(w: u16, cy: &[u16]) -> Vec<XGate> {
    (0..cy.len() / 2).map(|i| g57(w, cy[2 * i], cy[2 * i + 1])).collect()
}

/// The disjoint-pair g57 of an LGI on wire `w` (cycle `cy`) that READS band `b`,
/// i.e. the applied gate whose value changes when `b` flips. `None` if `b` is
/// only the odd-K trailing (unpaired) wire -- then no applied gate reads it, so
/// a `b`-update needs no repair on this mask.
fn b_g57(w: u16, cy: &[u16], b: u16) -> Option<XGate> {
    for i in 0..cy.len() / 2 {
        if cy[2 * i] == b || cy[2 * i + 1] == b {
            return Some(g57(w, cy[2 * i], cy[2 * i + 1]));
        }
    }
    None
}

/// A band-refresh update `b ^= lit & lit`. `data_ctrl`: one control is a LIVE
/// data wire (`0..active`, so it carries moving data, not a dead 0); otherwise
/// both controls are band wires. `active` = the low honest half in the sandwich.
fn rerand_gate(b: u16, data_ctrl: bool, active: usize, band: &[u16], rng: &mut StdRng) -> XGate {
    let (c1, c2) = if data_ctrl {
        let d = rng.random_range(0..active) as u16;
        let mut c = band[rng.random_range(0..band.len())];
        while c == b {
            c = band[rng.random_range(0..band.len())];
        }
        (d, c)
    } else {
        let mut a1 = band[rng.random_range(0..band.len())];
        while a1 == b {
            a1 = band[rng.random_range(0..band.len())];
        }
        let mut a2 = band[rng.random_range(0..band.len())];
        while a2 == b || a2 == a1 {
            a2 = band[rng.random_range(0..band.len())];
        }
        (a1, a2)
    };
    conj(b, &[(c1, rng.random_bool(0.5)), (c2, rng.random_bool(0.5))])
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
            // 0 controls: fire = comp ^ AND(empty) = comp ^ 1 = !comp
            if !comp {
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

/// The band-seeding module (module 2 of the 5-step pipeline): each band wire
/// `np..np+r` set to `x_i & !x_j` from the honest active input wires. Emitted
/// SEPARATELY and pipelined in front of the compute -- the compute only READS
/// the band. `active_wires` = 0 means all `np` data wires.
pub fn seed_band(np: usize, r: usize, active_wires: usize, seed: u64) -> Vec<XGate> {
    let active = if active_wires == 0 || active_wires > np {
        np
    } else {
        active_wires
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(r);
    for aw in (np as u16)..((np + r) as u16) {
        let i1 = rng.random_range(0..active) as u16;
        let mut i2 = rng.random_range(0..active) as u16;
        while i2 == i1 {
            i2 = rng.random_range(0..active) as u16;
        }
        out.push(XGate::conj(aw, [(i1, true), (i2, false)]).unwrap());
    }
    out
}

/// Data-hazard dependencies of `A`: any linear extension of them is a valid
/// reordering, since the only reorderings allowed are among commuting gates and
/// this permutation preserves `A`'s semantics. `deps[k]` = gates that must
/// precede `k`.
/// Same-target XOR writes COMMUTE (no WAW edge), so writers of a wire are not
/// chained -- both hazards run against the FULL per-wire history (never cleared):
/// a read is ordered after every earlier write of the wire, and a write after
/// every earlier read of it (incl. reads of the wire's initial value -- clearing
/// on write would let a later commuting writer slip before such a read).
fn compute_deps(src: &[XGate], np: usize) -> Vec<Vec<usize>> {
    let m = src.len();
    let mut deps: Vec<Vec<usize>> = vec![Vec::new(); m];
    let mut wrs: Vec<Vec<usize>> = vec![Vec::new(); np];
    let mut rds: Vec<Vec<usize>> = vec![Vec::new(); np];
    for k in 0..m {
        for &(cw, _) in &src[k].ctrls {
            let wi = cw as usize;
            if wi < np {
                for &g in &wrs[wi] {
                    deps[k].push(g); // RAW
                }
                rds[wi].push(k);
            }
        }
        let t = src[k].target as usize;
        for &r in &rds[t] {
            deps[k].push(r); // WAR
        }
        wrs[t].push(k);
    }
    for d in deps.iter_mut() {
        d.sort_unstable();
        d.dedup();
    }
    deps
}


/// Gadgetize `A` (`src`, on wires `0..np`) into a `np + r`-wire circuit with the
/// masked read. The band wires `np..np+r` are READ, never seeded here -- the
/// caller pipelines in [`seed_band`] (or the pipeline's band-fill module). The
/// `np` data wires end holding `A`'s output; the band is left dirty.
pub fn gadgetize_blinded_v5(src: &[XGate], np: usize, p: &BlindedV5Params) -> BlindedV5Output {
    let k = p.k.max(2); // g57 1-cycle is a degenerate constant flip
    let r = if p.r == 0 { np } else { p.r };
    assert!(r >= k, "R must be >= K");
    assert!(np >= 2, "need at least two data wires");
    let total = np + r;
    assert!(total < u16::MAX as usize, "too many wires");
    let band: Vec<u16> = (np as u16..total as u16).collect();
    let max_open = p.max_open.max(1);
    // Data controls for rerand draw from live-data wires only (0..active). In the
    // sandwich the honest input is the low half; the high half is a dead zero
    // slice, so a control there would be a constant 0 (useless as a refresh).
    let active = if p.active_wires == 0 || p.active_wires > np {
        np
    } else {
        p.active_wires
    };
    let mut rng = StdRng::seed_from_u64(p.seed);
    let mut out: Vec<XGate> = Vec::new();

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
        for _ in 0..(u[w] + 1 + p.extra_lgis) {
            pool[w].push(sample_k(&band, k, &mut rng));
        }
    }

    // 2. LGIs with ids + per-wire Open/Close event streams (rolling max_open),
    // then merge. Working at event granularity (not flat gates) lets the rerand
    // pass do the STRADDLE-ONLY close (below), which needs to know which masks
    // reading a band wire are open at each rerand.
    let mut lgi_w: Vec<u16> = Vec::new();
    let mut lgi_cy: Vec<Vec<u16>> = Vec::new();
    let mut wire_lgis: Vec<Vec<usize>> = vec![Vec::new(); np];
    for w in 0..np {
        for cy in &pool[w] {
            let id = lgi_w.len();
            lgi_w.push(w as u16);
            lgi_cy.push(cy.clone());
            wire_lgis[w].push(id);
        }
    }
    #[derive(Clone)]
    enum Ev {
        Open(usize),
        Close(usize),
        Rerand(u16, bool, bool), // (band wire, data-control?, is_repair?)
    }
    let mut streams: Vec<Vec<Ev>> = vec![Vec::new(); np];
    for w in 0..np {
        let mut open: Vec<usize> = Vec::new();
        for &id in &wire_lgis[w] {
            streams[w].push(Ev::Open(id));
            open.push(id);
            while open.len() > max_open {
                streams[w].push(Ev::Close(open.remove(0)));
            }
        }
        for &id in &open {
            streams[w].push(Ev::Close(id));
        }
    }
    let mut ptr = vec![0usize; np];
    let mut remaining: usize = streams.iter().map(|s| s.len()).sum();
    let mut events: Vec<Ev> = Vec::with_capacity(remaining);
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
        events.push(streams[w][ptr[w]].clone());
        ptr[w] += 1;
        remaining -= 1;
    }

    // 3. rerand events: `rerand_level` STRADDLE + `rerand_repair` REPAIR, each
    // total split half data-control (`b ^= data & aux`) / half band-only
    // (`b ^= aux & aux`) by index parity, spliced at sorted positions.
    let mut rplan: Vec<(usize, u16, bool, bool)> = Vec::new(); // (pos, b, data_ctrl, is_repair)
    for &(count, is_repair) in &[(p.rerand_level, false), (p.rerand_repair, true)] {
        for i in 0..count {
            let pos = rng.random_range(0..=events.len());
            let b = band[rng.random_range(0..band.len())];
            rplan.push((pos, b, i % 2 == 0, is_repair));
        }
    }
    rplan.sort_by_key(|x| x.0);
    let mut merged: Vec<Ev> = Vec::with_capacity(events.len() + rplan.len());
    let mut ri = 0;
    for (i, ev) in events.into_iter().enumerate() {
        while ri < rplan.len() && rplan[ri].0 == i {
            merged.push(Ev::Rerand(rplan[ri].1, rplan[ri].2, rplan[ri].3));
            ri += 1;
        }
        merged.push(ev);
    }
    while ri < rplan.len() {
        merged.push(Ev::Rerand(rplan[ri].1, rplan[ri].2, rplan[ri].3));
        ri += 1;
    }

    // forward-process: emit the scaffold. At each rerand, handle the masks
    // reading b that are open -- STRADDLE closes them (they'd otherwise carry a
    // stale b across the update), REPAIR re-derives them across the update so
    // they stay open (no masking collapse). See the Ev::Rerand arm.
    let mut scaffold: Vec<XGate> = Vec::new();
    let mut open_reading: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); r];
    let mut closed = vec![false; lgi_w.len()];
    let mut rerand_done = 0usize;
    for ev in merged {
        match ev {
            Ev::Open(id) => {
                if closed[id] {
                    continue;
                }
                scaffold.extend(cycle_g57(lgi_w[id], &lgi_cy[id]));
                for &bw in &lgi_cy[id] {
                    open_reading[bw as usize - np].insert(id);
                }
            }
            Ev::Close(id) => {
                if closed[id] {
                    continue;
                }
                closed[id] = true;
                scaffold.extend(cycle_g57(lgi_w[id], &lgi_cy[id]));
                for &bw in &lgi_cy[id] {
                    open_reading[bw as usize - np].remove(&id);
                }
            }
            Ev::Rerand(b, data_ctrl, is_repair) => {
                let bi = b as usize - np;
                let ids: Vec<usize> = open_reading[bi].iter().copied().collect();
                if is_repair {
                    // REPAIR: emit each straddling mask's b-reading g57 with the OLD
                    // b (removing its contribution), then the update, then again
                    // with the NEW b (re-adding it). Old-b cancels, new-b re-derives
                    // -- the mask stays open (no thinning), b just evolves under it.
                    for &id in &ids {
                        if let Some(g) = b_g57(lgi_w[id], &lgi_cy[id], b) {
                            scaffold.push(g);
                        }
                    }
                    scaffold.push(rerand_gate(b, data_ctrl, active, &band, &mut rng));
                    for &id in &ids {
                        if let Some(g) = b_g57(lgi_w[id], &lgi_cy[id], b) {
                            scaffold.push(g);
                        }
                    }
                } else {
                    // STRADDLE-only: close the masks reading b so none straddles the
                    // update, then apply it (heavy doses thin the masking).
                    for id in ids {
                        if closed[id] {
                            continue;
                        }
                        closed[id] = true;
                        scaffold.extend(cycle_g57(lgi_w[id], &lgi_cy[id]));
                        for &bw in &lgi_cy[id] {
                            open_reading[bw as usize - np].remove(&id);
                        }
                    }
                    scaffold.push(rerand_gate(b, data_ctrl, active, &band, &mut rng));
                }
                rerand_done += 1;
            }
        }
    }

    // 4. Place each A-gate STRADDLING an existing scaffold LGI on its active wire
    // (hidden-firing fix, RC): the straddled LGI half toggles c's mask INSIDE the
    // gate's fire, so the module's net XOR on c is Delta ^ (that LGI's secret
    // band-mask) -- never a bare increment -- for EVERY gate, reusing B-gates that
    // are already there (rerand-protected, already tracked in open_g57). No
    // injected masks, no repair, no size cost.
    let diag = std::env::var("BV5_DIAG").is_ok();
    // 4a/4b. LIST-SCHEDULE the placement during the walk: a gate is READY once all
    // its data-hazard dependencies are emitted; at each scaffold LGI slot on wire
    // c we place one ready gate targeting c (straddling that slot). This uses each
    // slot as soon as a gate needs it, so a gate is only left unplaced when it
    // genuinely becomes ready after the last usable slot on its wire.
    let deps = compute_deps(src, np);
    let mut indeg = vec![0usize; src.len()];
    let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); src.len()];
    for k in 0..src.len() {
        for &g in &deps[k] {
            dependents[g].push(k);
            indeg[k] += 1;
        }
    }
    // ready gates queued by target wire (FIFO for determinism)
    let mut ready: Vec<std::collections::VecDeque<usize>> =
        vec![std::collections::VecDeque::new(); np];
    for gi in 0..src.len() {
        if indeg[gi] == 0 {
            ready[src[gi].target as usize].push_back(gi);
        }
    }
    let mut placed = vec![false; src.len()];
    // 4c. emit one A-gate's masked read/fire, optionally straddling scaffold[idx]
    // (the LGI is emitted between the two fire halves; being on the target wire it
    // never touches the operands, so both halves share one rho -- no re-linearize).
    let mut open_g57: Vec<BTreeSet<(u16, u16)>> = vec![BTreeSet::new(); np];
    let mut diagc = (0usize, 0usize, 0usize); // (reads, bare, rho_sum)
    let mut emit_gate = |gi: usize,
                         straddle: Option<usize>,
                         open_g57: &mut Vec<BTreeSet<(u16, u16)>>,
                         out: &mut Vec<XGate>,
                         rng: &mut StdRng,
                         diagc: &mut (usize, usize, usize)| {
        let g = &src[gi];
        let mut ctrls: Vec<(u16, bool, Vec<u16>)> = Vec::new();
        let mut undo: Vec<XGate> = Vec::new();
        for &(w, pol) in &g.ctrls {
            let netopen: Vec<(u16, u16)> = open_g57[w as usize].iter().copied().collect();
            let (mut rho, added) = linearize(w, &netopen, out);
            for gg in added.into_iter().rev() {
                undo.push(gg);
            }
            // On-demand: add COMPLETE fresh linear pairs (self-contained r1^r2, fed
            // straight into rho) so the mask spans >= max_open pairs and rho is
            // NEVER empty (no bare read).
            let mut extra = max_open.saturating_sub(netopen.len());
            if extra == 0 && rho.is_empty() {
                extra = 1;
            }
            let mut guard = 0;
            loop {
                for _ in 0..extra {
                    let r1 = band[rng.random_range(0..r)];
                    let mut r2 = band[rng.random_range(0..r)];
                    while r2 == r1 {
                        r2 = band[rng.random_range(0..r)];
                    }
                    out.push(g57(w, r1, r2));
                    out.push(g57(w, r2, r1));
                    undo.push(g57(w, r2, r1));
                    undo.push(g57(w, r1, r2));
                    for x in [r1, r2] {
                        if let Some(pp) = rho.iter().position(|&v| v == x) {
                            rho.remove(pp);
                        } else {
                            rho.push(x);
                        }
                    }
                }
                extra = 0;
                if !rho.is_empty() || guard > 3 {
                    break;
                }
                extra = 1;
                guard += 1;
            }
            if diag {
                diagc.0 += 1;
                if rho.is_empty() {
                    diagc.1 += 1;
                }
                diagc.2 += rho.len();
            }
            ctrls.push((w, pol, rho));
        }
        let mut fires = Vec::new();
        masked_fire(g.target, &ctrls, g.comp, &mut fires);
        match straddle {
            Some(idx) => {
                let cut = ((fires.len() + 1) / 2).min(fires.len());
                out.extend_from_slice(&fires[..cut]);
                if let Some((w, pair)) = as_g57(&scaffold[idx], np) {
                    if !open_g57[w].insert(pair) {
                        open_g57[w].remove(&pair);
                    }
                }
                out.push(scaffold[idx].clone());
                out.extend_from_slice(&fires[cut..]);
            }
            None => out.extend(fires),
        }
        for gg in undo {
            out.push(gg);
        }
    };
    // 4d. walk the scaffold; at each LGI slot on c, straddle it with a ready gate
    // targeting c (firing hidden); emit the rest normally.
    let mut placed_count = 0usize;
    for idx in 0..scaffold.len() {
        if let Some((c, _)) = as_g57(&scaffold[idx], np) {
            if let Some(gi) = ready[c as usize].pop_front() {
                emit_gate(gi, Some(idx), &mut open_g57, &mut out, &mut rng, &mut diagc);
                placed[gi] = true;
                placed_count += 1;
                for &d in &dependents[gi] {
                    indeg[d] -= 1;
                    if indeg[d] == 0 {
                        ready[src[d].target as usize].push_back(d);
                    }
                }
                continue; // scaffold[idx] was emitted inside the straddle
            }
        }
        if let Some((w, pair)) = as_g57(&scaffold[idx], np) {
            if !open_g57[w].insert(pair) {
                open_g57[w].remove(&pair);
            }
        }
        out.push(scaffold[idx].clone());
    }
    // 4e. every gate not placed during the walk (ready after its last usable slot,
    // or freed only now) is emitted atomically at the end -- correct, but its
    // firing is NOT hidden. Drain the remaining DAG in a valid order.
    let mut unplaced_count = 0usize;
    loop {
        let mut progressed = false;
        for c in 0..np {
            while let Some(gi) = ready[c].pop_front() {
                emit_gate(gi, None, &mut open_g57, &mut out, &mut rng, &mut diagc);
                placed[gi] = true;
                unplaced_count += 1;
                progressed = true;
                for &d in &dependents[gi] {
                    indeg[d] -= 1;
                    if indeg[d] == 0 {
                        ready[src[d].target as usize].push_back(d);
                    }
                }
            }
        }
        if !progressed {
            break;
        }
    }
    debug_assert!(placed.iter().all(|&p| p));
    let (n_reads, n_bare, rho_sum) = diagc;
    if diag {
        eprintln!(
            "[bv5-diag] A-gates={} straddled(firing hidden)={} unplaced(firing exposed)={}",
            src.len(),
            placed_count,
            unplaced_count
        );
        eprintln!(
            "[bv5-diag] control reads={n_reads}  BARE (rho empty)={n_bare} ({:.3}%)  \
             mean |rho| (band wires masking the operand)={:.2}",
            100.0 * n_bare as f64 / n_reads.max(1) as f64,
            rho_sum as f64 / n_reads.max(1) as f64
        );
    }

    BlindedV5Output {
        gates: out,
        num_wires: total,
        atoms,
        rerand_done,
        r_used: r,
    }
}
