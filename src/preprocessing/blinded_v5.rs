//! Blinded-V5 computation-stage gadgetizer.
//!
//! Persistent geodesic-identity (LGI) additive masking of a source circuit,
//! with a cofactor-*blinded* read that never materialises a masked operand
//! (internal k=1 exposure = 0). This is the "V5 augmented with blinded read +
//! control-only write" computation stage of the sliced-sandwich pipeline.
//!
//! Three knobs (see [`BlindedV5Params`]):
//!   * `k` — control wires per LGI (mask cycle length; the mask is a sum of `k`
//!     ancilla-pair monomials, degree-2, self-inverse). `k` is the affine-leak
//!     lever: larger `k` = wider mask = harder to strip linearly.
//!   * `r` — ancilla / band pool (LGI cycle controls; `r >= k`). `r = n` is the
//!     tight Hamming-ridge floor; below it the HD ridge reappears.
//!   * `n_target` — injected-LGI churn target. The blinded per-gate operand
//!     refresh already provides a floor of ~2.2·|src| atoms, so any
//!     `n_target` below that floor is a no-op.
//!
//! Two structural options:
//!   * `discipline` (0/1/2) routes every data-write-with-data-control through a
//!     scratch aux so data-writes have only aux controls; level 2 masks the
//!     scratch so the payload delta is never bare.
//!   * `rerand` (dose, structural or adaptive) rerandomises free aux from data.
//!
//! The n=256 sliced-sandwich calibration settled on [`BlindedV5Params::production`]:
//! K=16 (affine knee), R=n (auto), N=floor, masked-scratch discipline, blinded
//! read, no rerand.

use crate::circuit::xgate::XGate;
use crate::preprocessing::gadgets::commuting_shuffle;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;

/// Configuration for [`gadgetize_blinded_v5`].
#[derive(Clone, Copy, Debug)]
pub struct BlindedV5Params {
    /// Control wires per LGI (mask cycle length). The affine-leak lever.
    pub k: usize,
    /// Ancilla / band pool size. `0` selects the settled default `r = np`
    /// (the tight HD-ridge floor); otherwise `r` must be `>= k`.
    pub r: usize,
    /// Injected-LGI churn target. `0` (or anything below the per-gate refresh
    /// floor of ~2.2·|src|) takes just the floor.
    pub n_target: usize,
    /// Deterministic seed for the masking / gadget stream.
    pub seed: u64,
    /// `true` = cofactor-blinded read (internal k=1 = 0); `false` = plain V5
    /// unmask-read-remask (leaks k=1).
    pub blinded: bool,
    /// `true` = t-pin blinded read (a fresh pin splits the cofactor into two
    /// stages so the operand XOR `d+e` is never on a wire; raises the
    /// comeback degree, k=2 -> k=3, at ~2x the read cost). Only meaningful
    /// with `blinded`. `false` = the single-CNOT blinded read (leaks `d+e`
    /// at k=2).
    pub tpin: bool,
    /// Rerand dose (aux rerandomisation gates). `0` = off.
    pub rerand_dose: usize,
    /// Data/aux discipline level: 0 = off, 1 = bare scratch, 2 = masked scratch.
    pub discipline: usize,
    /// `true` = local-adaptive rerand (trial-and-error, shuffle-agnostic);
    /// `false` = structural rerand at mask-free boundaries.
    pub adaptive_rerand: bool,
    /// Seed mask ancillas ONLY from data wires `0..active_wires` — the wires
    /// that carry live input on the honest input distribution. `0` (or
    /// `>= np`) means all `np` data wires. For a 2n-wire sliced sandwich used
    /// on its zero slice (x on the low n, zeros on the high n), set this to `n`:
    /// otherwise ~half the aux are seeded from the zeroed high wires and
    /// collapse to constant 0, gutting the masking on the actual usage.
    pub active_wires: usize,
}

impl BlindedV5Params {
    /// The settled production configuration for the n=256 sliced sandwich:
    /// K=16 (affine knee), R=n (auto), N=floor, masked-scratch discipline,
    /// blinded read, no rerand.
    pub fn production(seed: u64) -> Self {
        Self {
            k: 16,
            r: 0, // auto = np (R = n)
            n_target: 0, // floor
            seed,
            blinded: true,
            tpin: true,
            rerand_dose: 0,
            discipline: 2,
            adaptive_rerand: false,
            active_wires: 0, // caller sets = n for a zero-slice sandwich
        }
    }
}

/// Result of [`gadgetize_blinded_v5`].
pub struct BlindedV5Output {
    /// The gadgetized circuit (native g57 gates).
    pub gates: Vec<XGate>,
    /// Total wire count: `np + r + (discipline > 0)`.
    pub num_wires: usize,
    /// LGI atoms laid down (masking cost driver).
    pub atoms: usize,
    /// Rerand gates actually inserted (<= `rerand_dose`).
    pub rerand_done: usize,
    /// Effective ancilla pool used (`r`, resolved from the `0`-means-`np` rule).
    pub r_used: usize,
}

fn sample_k(anc: &[u16], k: usize, rng: &mut StdRng) -> Vec<u16> {
    let mut pool = anc.to_vec();
    let n = pool.len();
    for i in 0..k {
        let j = i + rng.random_range(0..(n - i));
        pool.swap(i, j);
    }
    pool.truncate(k);
    pool
}

/// K-cycle LGI half on wire `w` from ancilla sequence `cy`: w ^= a_i & a_{i+1}.
fn cycle_on(w: u16, cy: &[u16]) -> Vec<XGate> {
    let k = cy.len();
    (0..k)
        .map(|i| XGate::conj(w, [(cy[i], true), (cy[(i + 1) % k], true)]).expect("distinct"))
        .collect()
}

/// Change 1: route data-writes-with-data-controls through a scratch aux `s`.
/// level 1 = BARE scratch (s briefly = fires, exposes the payload product at k=2);
/// level 2 = MASKED scratch (s = M ^ fires, never bare; M removed from the data
/// target by an aux-controlled gate). Both keep the data/aux discipline.
fn apply_discipline(
    gates: &[XGate],
    np: usize,
    s: u16,
    level: usize,
    anc: &[u16],
    rng: &mut StdRng,
) -> Vec<XGate> {
    let mut out = Vec::with_capacity(gates.len());
    for g in gates {
        let writes_data = (g.target as usize) < np;
        let has_data_ctrl = g.ctrls.iter().any(|&(w, _)| (w as usize) < np);
        if writes_data && has_data_ctrl {
            let mut fires = g.clone();
            fires.target = s; // s ^= comp XOR (lits): data ctrls -> aux target
            if level >= 2 {
                // single aux-monomial mask M = a_i AND a_j on the scratch
                let ai = anc[rng.random_range(0..anc.len())];
                let mut aj = anc[rng.random_range(0..anc.len())];
                while aj == ai {
                    aj = anc[rng.random_range(0..anc.len())];
                }
                let mask_s = XGate::conj(s, [(ai, true), (aj, true)]).unwrap();
                let unmask_c = XGate::conj(g.target, [(ai, true), (aj, true)]).unwrap();
                out.push(mask_s.clone()); // s = M
                out.push(fires.clone()); // s = M ^ fires (masked delta, never bare)
                out.push(XGate::cnot(g.target, s)); // data ^= (M ^ fires)
                out.push(unmask_c); // data ^= M  => net data ^= fires
                out.push(fires); // s = M
                out.push(mask_s); // s = 0
            } else {
                out.push(fires.clone());
                out.push(XGate::cnot(g.target, s)); // data ^= s
                out.push(fires); // restore s to 0
            }
        } else {
            out.push(g.clone());
        }
    }
    out
}

/// Bit-sliced fire value of `g` on a 4-lane (256-sample) state: comp XOR the AND
/// of the (possibly negated) control lanes.
#[inline]
fn gate_fire(g: &XGate, s: &[[u64; 4]]) -> [u64; 4] {
    let mut acc = [!0u64; 4];
    for &(w, p) in &g.ctrls {
        let v = s[w as usize];
        if p {
            for l in 0..4 {
                acc[l] &= v[l];
            }
        } else {
            for l in 0..4 {
                acc[l] &= !v[l];
            }
        }
    }
    if g.comp {
        for l in 0..4 {
            acc[l] = !acc[l];
        }
    }
    acc
}

/// Sound, incremental validity check for a candidate rerand gate inserted just
/// before `fwd[0]`, given the base state `st` at that point. Propagates the
/// candidate's aux perturbation forward on both the base and perturbed branches
/// and REJECTS the instant it would reach a DATA wire (0..np); ACCEPTS if the
/// perturbation stays clear of every data wire through to the end (or fully
/// reconverges first). An accepted candidate provably preserves the data output
/// — it only rerandomises aux values the computation never reads downstream, the
/// data-neutral churn we want after a global float — which is why the check can
/// stop at the perturbation's forward reach instead of re-simulating the whole
/// circuit and comparing full states per trial. (Aux that ARE read into a
/// balanced mask are rejected here: they only rebalance at their unmask and the
/// safety margin is not worth chasing; the free-aux churn suffices.)
fn rerand_preserves_data(st: &[[u64; 4]], fwd: &[XGate], cand: &XGate, np: usize) -> bool {
    let mut sb = st.to_vec(); // base branch
    let mut sc = st.to_vec(); // candidate branch
    let f = gate_fire(cand, &sc);
    let t = cand.target as usize;
    for l in 0..4 {
        sc[t][l] ^= f[l];
    }
    if sc[t] == sb[t] {
        return false; // degenerate no-op candidate
    }
    let mut ndiff = 1usize; // wires where sb != sc
    for g in fwd {
        let tt = g.target as usize;
        let before = sb[tt] != sc[tt];
        let fb = gate_fire(g, &sb);
        let fc = gate_fire(g, &sc);
        for l in 0..4 {
            sb[tt][l] ^= fb[l];
            sc[tt][l] ^= fc[l];
        }
        let after = sb[tt] != sc[tt];
        if before != after {
            if after {
                ndiff += 1;
            } else {
                ndiff -= 1;
            }
        }
        if tt < np && after {
            return false; // perturbation reached a data output
        }
        if ndiff == 0 {
            return true; // fully reconverged: data (and all else) provably preserved
        }
    }
    true // reached the end with data never disturbed
}

/// CHANGE 2, local-adaptive form: with NO structural knowledge, sweep a cursor
/// forward maintaining the base state; at spread-out positions try a random
/// rerand gate aux_t ^= data_d & aux_c and keep it iff it provably preserves the
/// data output ([`rerand_preserves_data`], a 256-sample check). Meant to run
/// AFTER a global gate-location shuffle so it cannot use the mask schedule.
/// Returns (rewritten gate list, accepted count).
fn adaptive_rerand_incremental(
    gates: &[XGate],
    np: usize,
    nw: usize,
    anc: &[u16],
    dose: usize,
    inputs: &[[u64; 4]],
    rng: &mut StdRng,
) -> (Vec<XGate>, usize) {
    let r = anc.len();
    let mut st = vec![[0u64; 4]; nw];
    st[..np].copy_from_slice(&inputs[..np]); // aux start at 0, seeded by the gates
    let mut out = Vec::with_capacity(gates.len() + dose);
    let mut accepted = 0usize;
    let len = gates.len();
    for idx in 0..=len {
        if accepted < dose {
            let need = dose - accepted;
            let remaining = len + 1 - idx;
            // Attempt at a fraction of positions rising toward the tail (need
            // per remaining position, oversampled): most positions reject fast
            // (data diverges on first mask), and accepts land where an aux is
            // dead-forward to the data. Rejects are cheap so no attempt cap is
            // needed; the achievable dose is bounded by how many safely
            // rerandomisable aux positions the floated circuit actually has.
            let p_try = ((need as f64) / (remaining.max(1) as f64) * 8.0).min(1.0);
            if rng.random_bool(p_try) {
                let tgt = anc[rng.random_range(0..r)];
                let dd = rng.random_range(0..np) as u16;
                let mut ac = anc[rng.random_range(0..r)];
                while ac == tgt {
                    ac = anc[rng.random_range(0..r)];
                }
                let cand =
                    XGate::conj(tgt, [(dd, rng.random_bool(0.5)), (ac, rng.random_bool(0.5))])
                        .unwrap();
                if rerand_preserves_data(&st, &gates[idx..], &cand, np) {
                    let f = gate_fire(&cand, &st);
                    let tt = cand.target as usize;
                    for l in 0..4 {
                        st[tt][l] ^= f[l];
                    }
                    out.push(cand);
                    accepted += 1;
                }
            }
        }
        if idx == len {
            break;
        }
        let g = &gates[idx];
        let f = gate_fire(g, &st);
        let tt = g.target as usize;
        for l in 0..4 {
            st[tt][l] ^= f[l];
        }
        out.push(g.clone());
    }
    (out, accepted)
}

fn blinded_pols(p1: bool, p2: bool) -> (bool, bool, bool) {
    match (p1, p2) {
        (true, false) => (false, true, false),
        (false, true) => (false, true, true),
        (true, true) => (true, false, true),
        (false, false) => (true, false, false),
    }
}

/// Local-clearing band refresh (RC's algorithm, 2026-09-03): sprinkle `dose`
/// band-update gates `b ^= data_d & aux_c` MID-computation to continuously
/// refresh the band values as the computation proceeds. Each update is inserted
/// at a random point P and made safe PER-IDENTITY: only the identities (an LGI
/// mask's apply+remove, or a read-gadget block, tagged by `id_of`) whose uses of
/// `b` straddle P have those uses commuted to one side — every other identity is
/// left in place, so identities before P keep the old `b` and identities after
/// see the new `b`, and `b` genuinely evolves through the run. (Consolidating
/// ALL of b's uses instead would shove the update to b's lifetime edge and
/// refresh nothing — the rejected "all-one-side" variant.) `b` is chosen light
/// (fewest control-gates) so few gates move; a commutation blocker aborts the
/// attempt (the partial, commutation-valid moves keep the circuit ≡) and retries.
/// `id_of` is kept parallel to `gates` through every swap. Returns (accepted, blockers).
fn local_clearing_per_identity(
    gates: &mut Vec<XGate>,
    id_of: &mut Vec<i64>,
    np: usize,
    band: &[u16],
    dose: usize,
    rng: &mut StdRng,
) -> (usize, usize) {
    use std::collections::{HashMap, HashSet};
    const SENT: u16 = u16::MAX; // sentinel marks the insert point (replaced before return)
    let band_lo = band.iter().copied().min().unwrap_or(0);
    // Precompute each band wire's control-gate count ONCE, then keep it current
    // incrementally (each accepted update adds one control-use of its `c` wire).
    let mut counts: HashMap<u16, usize> = band.iter().map(|&w| (w, 0usize)).collect();
    for g in gates.iter() {
        for &(w, _) in &g.ctrls {
            if let Some(c) = counts.get_mut(&w) {
                *c += 1;
            }
        }
    }
    let _ = band_lo;
    let mut accepted = 0usize;
    let mut blockers = 0usize;
    let cap = dose.saturating_mul(60).max(dose + 30_000);
    let mut attempts = 0usize;
    while accepted < dose && attempts < cap {
        attempts += 1;
        let len = gates.len();
        if len < 2 {
            break;
        }
        let p = rng.random_range(1..len);
        // light band wire: fewest control-gates among a few samples (precomputed)
        let mut b = band[rng.random_range(0..band.len())];
        let mut best = usize::MAX;
        for _ in 0..6 {
            let cand = band[rng.random_range(0..band.len())];
            let cnt = counts[&cand];
            if cnt < best {
                best = cnt;
                b = cand;
            }
        }
        gates.insert(p, XGate::x_gate(SENT));
        id_of.insert(p, -2);
        let mut sp = p;
        // which tagged identities have b-uses on both sides of sp?
        let mut sides: HashMap<i64, (bool, bool)> = HashMap::new();
        for (i, g) in gates.iter().enumerate() {
            if i == sp || id_of[i] < 0 {
                continue;
            }
            if g.ctrls.iter().any(|&(w, _)| w == b) {
                let e = sides.entry(id_of[i]).or_insert((false, false));
                if i < sp {
                    e.0 = true;
                } else {
                    e.1 = true;
                }
            }
        }
        let straddle: HashSet<i64> = sides
            .iter()
            .filter(|(_, (bf, af))| *bf && *af)
            .map(|(id, _)| *id)
            .collect();
        let mut good = true;
        for &id in &straddle {
            // b-gates of this identity, split by the CURRENT sentinel position
            let mut before = Vec::new();
            let mut after = Vec::new();
            for (i, g) in gates.iter().enumerate() {
                if i == sp {
                    continue;
                }
                if id_of[i] == id && g.ctrls.iter().any(|&(w, _)| w == b) {
                    if i < sp {
                        before.push(i);
                    } else {
                        after.push(i);
                    }
                }
            }
            if before.is_empty() || after.is_empty() {
                continue;
            }
            if before.len() <= after.len() {
                before.sort_unstable_by(|a, c| c.cmp(a));
                for &start in &before {
                    let mut i = start;
                    while i < sp {
                        if XGate::collides(&gates[i], &gates[i + 1]) {
                            good = false;
                            break;
                        }
                        gates.swap(i, i + 1);
                        id_of.swap(i, i + 1);
                        i += 1;
                    }
                    if !good {
                        break;
                    }
                    sp -= 1;
                }
            } else {
                after.sort_unstable();
                for &start in &after {
                    let mut i = start;
                    while i > sp {
                        if XGate::collides(&gates[i], &gates[i - 1]) {
                            good = false;
                            break;
                        }
                        gates.swap(i, i - 1);
                        id_of.swap(i, i - 1);
                        i -= 1;
                    }
                    if !good {
                        break;
                    }
                    sp += 1;
                }
            }
            if !good {
                break;
            }
        }
        let spos = gates.iter().position(|g| g.target == SENT).unwrap();
        if !good {
            gates.remove(spos);
            id_of.remove(spos);
            blockers += 1;
            continue;
        }
        let d = rng.random_range(0..np) as u16;
        let mut c = band[rng.random_range(0..band.len())];
        while c == b {
            c = band[rng.random_range(0..band.len())];
        }
        gates[spos] =
            XGate::conj(b, [(d, rng.random_bool(0.5)), (c, rng.random_bool(0.5))]).unwrap();
        id_of[spos] = -1;
        if let Some(cc) = counts.get_mut(&c) {
            *cc += 1; // the new update reads aux_c as a control
        }
        accepted += 1;
    }
    (accepted, blockers)
}

/// Gadgetize `src` (a circuit on wires `0..np`) with persistent LGI masking and
/// the blinded/plain read. See [`BlindedV5Params`]. The output computes the same
/// function as `src` on its `np` data wires (aux start and end at 0).
pub fn gadgetize_blinded_v5(src: &[XGate], np: usize, p: &BlindedV5Params) -> BlindedV5Output {
    let k = p.k;
    let r = if p.r == 0 { np } else { p.r };
    let n_target = p.n_target;
    let discipline = p.discipline;
    let adaptive = p.adaptive_rerand;
    let rerand_dose = p.rerand_dose;
    assert!(r >= k, "R must be >= K");

    // Seed mask ancillas only from the honest-input active wires (0..active),
    // so none are seeded from wires that are constant on the actual usage.
    let active = if p.active_wires == 0 || p.active_wires > np {
        np
    } else {
        p.active_wires
    };
    let np_u = np as u16;
    let anc: Vec<u16> = (np_u..np_u + r as u16).collect();
    let scratch = np_u + r as u16; // change-1 scratch aux (=0)
    let nw = np + r + if discipline > 0 { 1 } else { 0 };
    let mut rng = StdRng::seed_from_u64(p.seed);
    let mut out: Vec<XGate> = Vec::new();
    let mut atoms = 0usize;

    // Identity tag per emitted gate (for the local-clearing band refresh). A
    // persistent mask (apply + its later remove, re-emissions of the same cycle)
    // shares one id via `live_id[w]`; each blinded/plain read gadget is one block
    // id; seeds and structural rerand are untagged (-1). `emit`/`emitn` push a
    // gate + its tag in lock-step so `id_of` stays parallel to `out`.
    let mut id_of: Vec<i64> = Vec::new();
    let mut next_id: i64 = 0;
    let mut live_id: Vec<i64> = vec![-1; np];
    macro_rules! emit {
        ($g:expr, $t:expr) => {{
            out.push($g);
            id_of.push($t);
        }};
    }
    macro_rules! emitn {
        ($gs:expr, $t:expr) => {{
            for g in $gs.iter().cloned() {
                out.push(g);
                id_of.push($t);
            }
        }};
    }

    // seed ancillas: each = (x_i AND NOT x_j), i,j drawn from the active wires
    for &aw in &anc {
        let i1 = rng.random_range(0..active) as u16;
        let mut i2 = rng.random_range(0..active) as u16;
        while i2 == i1 {
            i2 = rng.random_range(0..active) as u16;
        }
        emit!(XGate::conj(aw, [(i1, true), (i2, false)]).unwrap(), -1);
    }

    let mut live: Vec<Vec<XGate>> = vec![Vec::new(); np];
    let mut live_cy: Vec<Vec<u16>> = vec![Vec::new(); np]; // ancilla set per live cycle
    for w in 0..np {
        let cy = sample_k(&anc, k, &mut rng);
        let h = cycle_on(w as u16, &cy);
        let id = next_id;
        next_id += 1;
        emitn!(h, id);
        live[w] = h;
        live_cy[w] = cy;
        live_id[w] = id;
        atoms += 1;
    }

    let base_est = np + 3 * src.len() + np;
    let extra = n_target.saturating_sub(base_est);
    let slots = src.len() + 1;
    let churn_per_slot = extra as f64 / slots as f64;
    // rerand is a post-pass now (clearing by default, adaptive if selected); the
    // old inline free-aux structural rerand is disabled (superseded).
    let rerand_per_slot = 0.0f64;
    let mut churn_acc = 0.0f64;
    let mut rerand_acc = 0.0f64;
    let mut rerand_done = 0usize;

    let bwire = anc[0];

    // main loop
    for gi in 0..=src.len() {
        // change 2: rerandomize free aux at this boundary
        rerand_acc += rerand_per_slot;
        let mut want = rerand_acc as usize;
        rerand_acc -= want as f64;
        if want > 0 {
            let committed: HashSet<u16> = live_cy.iter().flatten().copied().collect();
            let free: Vec<u16> = anc.iter().copied().filter(|w| !committed.contains(w)).collect();
            want = want.min(free.len());
            for &at in free.iter().take(want) {
                let dd = rng.random_range(0..np) as u16;
                // control aux != at
                let mut ac = anc[rng.random_range(0..r)];
                while ac == at {
                    ac = anc[rng.random_range(0..r)];
                }
                emit!(
                    XGate::conj(at, [(dd, rng.random_bool(0.5)), (ac, rng.random_bool(0.5))])
                        .unwrap(),
                    -1
                );
                rerand_done += 1;
            }
        }

        // churn refreshes (change-2 N density)
        churn_acc += churn_per_slot;
        let nref = churn_acc as usize;
        churn_acc -= nref as f64;
        for _ in 0..nref {
            let w = rng.random_range(0..np);
            let cy = sample_k(&anc, k, &mut rng);
            let nh = cycle_on(w as u16, &cy);
            let id = next_id;
            next_id += 1;
            emitn!(nh, id); // apply new mask
            emitn!(live[w], live_id[w]); // remove old mask (shares its id)
            live[w] = nh;
            live_cy[w] = cy;
            live_id[w] = id;
            atoms += 1;
        }

        if gi == src.len() {
            break;
        }
        let g = &src[gi];
        let mut touched: Vec<u16> = vec![g.target];
        for &(w, _) in &g.ctrls {
            if !touched.contains(&w) {
                touched.push(w);
            }
        }

        if p.blinded && g.ctrls.len() == 2 {
            let (w1, p1) = (g.ctrls[0].0, g.ctrls[0].1);
            let (w2, p2) = (g.ctrls[1].0, g.ctrls[1].1);
            let cy = sample_k(&anc, k, &mut rng);
            let s1 = cycle_on(w1, &cy);
            let s2 = cycle_on(w2, &cy);
            let id1 = next_id;
            next_id += 1;
            emitn!(s1, id1); // new mask on w1
            emitn!(live[w1 as usize], live_id[w1 as usize]); // remove old w1 mask
            live[w1 as usize] = s1;
            live_cy[w1 as usize] = cy.clone();
            live_id[w1 as usize] = id1;
            atoms += 1;
            let id2 = next_id;
            next_id += 1;
            emitn!(s2, id2); // new mask on w2
            emitn!(live[w2 as usize], live_id[w2 as usize]); // remove old w2 mask
            live[w2 as usize] = s2.clone();
            live_cy[w2 as usize] = cy.clone();
            live_id[w2 as usize] = id2;
            atoms += 1;
            let block = next_id; // the read gadget is one identity block
            next_id += 1;
            let (bpol, f1p, f2p) = blinded_pols(p1, p2);
            if p.tpin {
                // t-pin read: a fresh pin `tp_w` (a band wire holding a stable
                // seeded value w, restored) splits the cofactor S=w1+w2 into two
                // stages (w+w1),(w+w2), so the operand XOR d+e is never on a
                // wire. The pin must avoid the shared mask cycle `cy` (its wires
                // are read to unmask w2) and the blind wire. Stage i uses pin
                // polarity `tp` = f1p for stage 0 and true for stage 1: the two
                // stages sum to F1.F2 = (S or !S).F2 for either f1p, and `comp`
                // is applied once (stage 0).
                let mut tp_w = anc[rng.random_range(0..r)];
                while cy.contains(&tp_w) || tp_w == bwire {
                    tp_w = anc[rng.random_range(0..r)];
                }
                for (i, &op) in [w1, w2].iter().enumerate() {
                    let tp = if i == 0 { f1p } else { true };
                    emit!(XGate::cnot(tp_w, op), block); // t ^= op  (t = w+op+u)
                    // blind w2 with b.lit(t,!tp) (annihilated by the cofactor lit(t,tp))
                    emit!(XGate::conj(w2, [(bwire, true), (tp_w, !tp)]).unwrap(), block);
                    emitn!(s2, id2); // unmask w2 (w2 = e) -- w2 mask toggle
                    let mut pay = XGate::conj(g.target, [(tp_w, tp), (w2, f2p)]).unwrap();
                    if i == 0 {
                        pay.comp = g.comp; // complement applied once
                    }
                    emit!(pay, block); // c ^= lit(t,tp) . lit(e,f2p)
                    emitn!(s2, id2); // remask w2 -- w2 mask toggle
                    emit!(XGate::conj(w2, [(bwire, true), (tp_w, !tp)]).unwrap(), block); // unblind
                    emit!(XGate::cnot(tp_w, op), block); // restore t (t = w)
                }
            } else {
                // single-CNOT blinded read (leaks d+e at k=2)
                emit!(XGate::cnot(w1, w2), block);
                emit!(XGate::conj(w2, [(bwire, true), (w1, bpol)]).unwrap(), block);
                emitn!(s2, id2);
                let mut pay = XGate::conj(g.target, [(w1, f1p), (w2, f2p)]).unwrap();
                pay.comp = g.comp;
                emit!(pay, block);
                emitn!(s2, id2);
                emit!(XGate::conj(w2, [(bwire, true), (w1, bpol)]).unwrap(), block);
                emit!(XGate::cnot(w1, w2), block);
            }
        } else {
            let block = next_id;
            next_id += 1;
            for &w in &touched {
                emitn!(live[w as usize], live_id[w as usize]); // unmask
                atoms += 1;
            }
            emit!(g.clone(), block); // the raw source gate
            for &w in &touched {
                let cy = sample_k(&anc, k, &mut rng);
                let h = cycle_on(w, &cy);
                let id = next_id;
                next_id += 1;
                emitn!(h, id); // remask
                live[w as usize] = h;
                live_cy[w as usize] = cy;
                live_id[w as usize] = id;
                atoms += 1;
            }
        }
    }

    // final unmask
    for w in 0..np {
        emitn!(live[w], live_id[w]);
        atoms += 1;
    }

    // CHANGE 2 (band refresh): sprinkle rerand gates through the compute. Default
    // = the local-clearing PER-IDENTITY pass (mid-computation refresh, uses the
    // `id_of` identity tags). `adaptive` selects the older shuffle-then-incremental
    // path instead (float first, then a structure-blind sound check — lower yield).
    if rerand_dose > 0 {
        if adaptive {
            commuting_shuffle(&mut out, &mut rng);
            let inputs: Vec<[u64; 4]> = (0..np)
                .map(|_| [rng.random(), rng.random(), rng.random(), rng.random()])
                .collect();
            let (new_out, acc) =
                adaptive_rerand_incremental(&out, np, np + r, &anc, rerand_dose, &inputs, &mut rng);
            out = new_out;
            rerand_done = acc;
        } else {
            let (acc, _blk) =
                local_clearing_per_identity(&mut out, &mut id_of, np, &anc, rerand_dose, &mut rng);
            rerand_done = acc;
        }
    }

    if discipline > 0 {
        out = apply_discipline(&out, np, scratch, discipline, &anc, &mut rng);
    }

    BlindedV5Output {
        gates: out,
        num_wires: nw,
        atoms,
        rerand_done,
        r_used: r,
    }
}
