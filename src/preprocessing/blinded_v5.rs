//! Blinded-V5 computation stage: an alternative to the drip `route_fire`
//! compute. Takes a circuit `A` on `n` wires and builds an equivalent circuit
//! on `2n` wires (data `0..n`, band `n..2n`) whose middle is a shuffled cloud
//! of locally-geodesic-identity (LGI) masks with `A`'s gates threaded through
//! it via a MASKED read. Only the compute changes; the surrounding pipeline
//! (slice guards, band fill, band rerand, final slice) is unchanged.
//!
//! The masking atom is the g57 gate `g57(w,x,y) = w ^= 1 ^ (!x & y)` (data
//! target, band controls). A DISJOINT-PAIR LGI on `w`, `g57(w,cy[2i],cy[2i+1])`,
//! is a deg-2 mask; a g57 and its reverse LINEARISE -- `g57(w,r1,r2) ^
//! g57(w,r2,r1) = w ^ r1 ^ r2` -- which the read exploits (AND monomials are
//! symmetric and do NOT linearise). K must be >= 2 (a 1-cycle is a degenerate
//! constant flip).
//!
//! CO-SAMPLED build (RC, 2026-09-04): the LGI masks, the rerand gates, and the
//! A-gate placements are produced TOGETHER in one forward pass, so every A-gate
//! is straddled by a real (rerand-protected) LGI. Per active wire `w`, `u_w+1`
//! LGIs (u_w = uses of w) with <= `max_open` open at once -- SAME masking budget
//! as before, so the same statistics, at no cost. Of `w`'s opens, `w_w` are
//! STRADDLE opens generated ON DEMAND when an A-gate on `w` is placed, and the
//! rest are FILLER opens (masking during reads):
//!   * MASKED READ: for each control, LINEARISE the net-open masks (emit each
//!     reverse g57 so the wire carries `operand ^ rho`, rho a linear XOR of band
//!     wires; a fresh-pair top-up keeps rho non-empty -> never a bare operand),
//!     realise `c ^= comp ^ lit(a)&lit(b)` as `(a'^rho_a)(b'^rho_b)` over ONLY
//!     the masked control wires and band wires (never bare `a`,`b`,`a^b`; 0/1/2
//!     controls, all polarities), then DE-LINEARISE.
//!   * HIDDEN FIRING: the gate's fire is split and the STRADDLE-OPEN of one of
//!     `c`'s LGIs is emitted between the halves. That mask toggles `c` mid-fire,
//!     so the module's net XOR on `c` is `Delta ^ (secret band-mask)`, never the
//!     bare gate increment -- otherwise the active wire's before/after XOR across
//!     an atomic module would expose which gate of A fired.
//!   * RERAND: band-refresh gates `b ^= lit & lit` (half live-data control, half
//!     band-only) woven through the pass in two kinds -- `rerand_level` STRADDLE
//!     (close the masks reading `b` before the update; thins masking past a ~1024
//!     knee) and `rerand_repair` REPAIR (re-derive each mask reading `b` across
//!     the update so it stays open -- no thinning). Both cover the straddle opens
//!     automatically since everything is in one pass.
//! A-gates are placed in a data-hazard-valid order (compute_deps).
//!
//! Only the compute changes; the surrounding pipeline (slice guards, band seed,
//! final slice) is unchanged. Verified EXHAUSTIVELY (all 2^n inputs x many band
//! settings, k in 2..=5, all max_open and rerand levels) in `scratchpad/v6`
//! (compute_g57); n-independent.

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
/// Emit an LGI's disjoint-pair g57s on wire `w` and TOGGLE its pairs in `pairs`
/// (opening if closed, closing if open -- self-inverse either way).
fn emit_lgi(w: u16, cy: &[u16], out: &mut Vec<XGate>, pairs: &mut BTreeSet<(u16, u16)>) {
    out.extend(cycle_g57(w, cy));
    for i in 0..cy.len() / 2 {
        let pr = (cy[2 * i], cy[2 * i + 1]);
        if !pairs.insert(pr) {
            pairs.remove(&pr);
        }
    }
}

/// Weighted random index into `weights` (sum == `total`), or None if total == 0.
fn pick_weighted(weights: &[usize], total: usize, rng: &mut StdRng) -> Option<usize> {
    if total == 0 {
        return None;
    }
    let mut r = rng.random_range(0..total);
    for (w, &wt) in weights.iter().enumerate() {
        if r < wt {
            return Some(w);
        }
        r -= wt;
    }
    None
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
    // Data controls for rerand draw from live-data wires only (0..active).
    let active = if p.active_wires == 0 || p.active_wires > np {
        np
    } else {
        p.active_wires
    };
    let mut rng = StdRng::seed_from_u64(p.seed);
    let m = src.len();

    // Per-wire LGI budget: u_w+1(+extra) LGIs -> w_w STRADDLE opens (one per write
    // to w, generated on demand at the gate's fire so it hides that firing) and
    // the rest FILLER opens (masking during reads). Same masking budget as before,
    // just co-sampled with the A-gate placement -> same statistics, no cost.
    let mut u = vec![0usize; np];
    let mut writes = vec![0usize; np];
    for g in src {
        u[g.target as usize] += 1;
        writes[g.target as usize] += 1;
        for &(w, _) in &g.ctrls {
            if (w as usize) < np {
                u[w as usize] += 1;
            }
        }
    }
    let atoms: usize = u.iter().map(|&c| c + 1 + p.extra_lgis).sum();
    let mut filler_left = vec![0usize; np];
    let mut straddles_left = vec![0usize; np];
    for w in 0..np {
        straddles_left[w] = writes[w];
        filler_left[w] = (u[w] + 1 + p.extra_lgis) - writes[w];
    }

    // dependency readiness (Kahn); any linear extension preserves A's semantics.
    let deps = compute_deps(src, np);
    let mut indeg = vec![0usize; m];
    let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); m];
    for i in 0..m {
        for &g in &deps[i] {
            dependents[g].push(i);
            indeg[i] += 1;
        }
    }
    let mut ready: std::collections::VecDeque<usize> =
        (0..m).filter(|&i| indeg[i] == 0).collect();

    // rerand plan: rerand_level STRADDLE + rerand_repair REPAIR, each half
    // data-control / half band-only; shuffled, spread through the build.
    let mut rplan: Vec<(bool, bool)> = Vec::new();
    for &(count, is_repair) in &[(p.rerand_level, false), (p.rerand_repair, true)] {
        for i in 0..count {
            rplan.push((is_repair, i % 2 == 0));
        }
    }
    for i in (1..rplan.len()).rev() {
        let j = rng.random_range(0..=i);
        rplan.swap(i, j);
    }

    let mut open_cy: Vec<Vec<Vec<u16>>> = vec![Vec::new(); np];
    let mut open_pairs: Vec<BTreeSet<(u16, u16)>> = vec![BTreeSet::new(); np];
    let mut out: Vec<XGate> = Vec::with_capacity(30 * (3 * m + np));
    let mut rerand_done = 0usize;
    let diag = std::env::var("BV5_DIAG").is_ok();
    let (mut n_reads, mut n_bare, mut rho_sum) = (0usize, 0usize, 0usize);

    let total_opens: usize =
        straddles_left.iter().sum::<usize>() + filler_left.iter().sum::<usize>();
    let step_target = (2 * total_opens).max(1);
    let mut ri = 0usize;
    let mut steps = 0usize;
    let rgap = if rplan.is_empty() {
        usize::MAX
    } else {
        (step_target / rplan.len()).max(1)
    };
    let fillers_per_gate = (filler_left.iter().sum::<usize>() / m.max(1)).max(1);

    macro_rules! emit_one_rerand {
        () => {{
            if ri < rplan.len() {
                let (is_repair, data_ctrl) = rplan[ri];
                ri += 1;
                let b = band[rng.random_range(0..band.len())];
                if is_repair {
                    for w in 0..np {
                        for cy in &open_cy[w] {
                            if let Some(gg) = b_g57(w as u16, cy, b) {
                                out.push(gg);
                            }
                        }
                    }
                    out.push(rerand_gate(b, data_ctrl, active, &band, &mut rng));
                    for w in 0..np {
                        for cy in &open_cy[w] {
                            if let Some(gg) = b_g57(w as u16, cy, b) {
                                out.push(gg);
                            }
                        }
                    }
                } else {
                    for w in 0..np {
                        let mut idx = 0;
                        while idx < open_cy[w].len() {
                            if open_cy[w][idx].iter().any(|&x| x == b) {
                                let cy = open_cy[w].remove(idx);
                                emit_lgi(w as u16, &cy, &mut out, &mut open_pairs[w]);
                            } else {
                                idx += 1;
                            }
                        }
                    }
                    out.push(rerand_gate(b, data_ctrl, active, &band, &mut rng));
                }
                rerand_done += 1;
            }
        }};
    }
    macro_rules! maybe_rerand {
        () => {{
            steps += 1;
            if steps % rgap == 0 {
                emit_one_rerand!();
            }
        }};
    }
    macro_rules! filler_open {
        ($w:expr) => {{
            let w = $w;
            if open_cy[w].len() >= max_open {
                let cy = open_cy[w].remove(0);
                emit_lgi(w as u16, &cy, &mut out, &mut open_pairs[w]);
            }
            let cy = sample_k(&band, k, &mut rng);
            emit_lgi(w as u16, &cy, &mut out, &mut open_pairs[w]);
            open_cy[w].push(cy);
            filler_left[w] -= 1;
            maybe_rerand!();
        }};
    }

    let mut placed = 0usize;
    while placed < m {
        let gi = match ready.pop_front() {
            Some(g) => g,
            None => break,
        };
        let c = src[gi].target as usize;
        let mut ctrls: Vec<(u16, bool, Vec<u16>)> = Vec::new();
        let mut undo: Vec<XGate> = Vec::new();
        for &(w, pol) in &src[gi].ctrls {
            let netopen: Vec<(u16, u16)> = open_pairs[w as usize].iter().copied().collect();
            let (mut rho, added) = linearize(w, &netopen, &mut out);
            for gg in added.into_iter().rev() {
                undo.push(gg);
            }
            let mut extra = max_open.saturating_sub(netopen.len());
            if extra == 0 && rho.is_empty() {
                extra = 1;
            }
            let mut guard = 0;
            loop {
                for _ in 0..extra {
                    let r1 = band[rng.random_range(0..band.len())];
                    let mut r2 = band[rng.random_range(0..band.len())];
                    while r2 == r1 {
                        r2 = band[rng.random_range(0..band.len())];
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
                n_reads += 1;
                if rho.is_empty() {
                    n_bare += 1;
                }
                rho_sum += rho.len();
            }
            ctrls.push((w, pol, rho));
        }
        let mut fires = Vec::new();
        masked_fire(src[gi].target, &ctrls, src[gi].comp, &mut fires);
        if open_cy[c].len() >= max_open {
            let cy = open_cy[c].remove(0);
            emit_lgi(c as u16, &cy, &mut out, &mut open_pairs[c]);
        }
        let cut = ((fires.len() + 1) / 2).min(fires.len());
        out.extend_from_slice(&fires[..cut]);
        let cy = sample_k(&band, k, &mut rng);
        emit_lgi(c as u16, &cy, &mut out, &mut open_pairs[c]);
        open_cy[c].push(cy);
        out.extend_from_slice(&fires[cut..]);
        for gg in undo {
            out.push(gg);
        }
        straddles_left[c] -= 1;
        placed += 1;
        maybe_rerand!();
        for &d in &dependents[gi] {
            indeg[d] -= 1;
            if indeg[d] == 0 {
                ready.push_back(d);
            }
        }
        for _ in 0..fillers_per_gate {
            let tf: usize = filler_left.iter().sum();
            match pick_weighted(&filler_left, tf, &mut rng) {
                Some(w) => filler_open!(w),
                None => break,
            }
        }
    }
    debug_assert!(placed == m);
    loop {
        let tf: usize = filler_left.iter().sum();
        match pick_weighted(&filler_left, tf, &mut rng) {
            Some(w) => filler_open!(w),
            None => break,
        }
    }
    while ri < rplan.len() {
        emit_one_rerand!();
    }
    for w in 0..np {
        while let Some(cy) = open_cy[w].pop() {
            emit_lgi(w as u16, &cy, &mut out, &mut open_pairs[w]);
        }
    }
    let _ = straddles_left;
    if diag {
        eprintln!(
            "[bv5-diag] A-gates={m} straddled(firing hidden)={m} unplaced=0  control reads={n_reads}  \
             BARE (rho empty)={n_bare} ({:.3}%)  mean |rho|={:.2}",
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
