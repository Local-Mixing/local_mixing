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
//!   * `rerand` (dose + [`RerandMode`]) refreshes band values from data as the
//!     computation proceeds.
//!
//! The n=256 sliced-sandwich calibration settled on [`BlindedV5Params::production`]:
//! K=16 (affine knee), R=n (auto), N=floor, masked-scratch discipline, blinded
//! read, no rerand.
//!
//! Note on the output contract: the `np` DATA wires are restored to exactly the
//! source circuit's outputs, but the ancilla wires are NOT returned to zero —
//! they are seeded from the input and keep their (possibly refreshed) values.
//! The surrounding slice stages own clearing them.

use crate::circuit::xgate::XGate;
use crate::preprocessing::gadgets::commuting_shuffle;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// How the band-refresh dose is placed (see `rerand_dose`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RerandMode {
    /// Generation-time refresh with live-mask repair. At source-gate
    /// boundaries, spread uniformly over the run, emit `b ^= data_d & aux_c`
    /// bracketed by the `b`-terms of every live mask that reads `b`. That
    /// mask's `b`-contribution then telescopes to zero across
    /// apply(`b_old`) / pre-repair(`b_old`) / post-repair(`b_new`) /
    /// remove(`b_new`), so it still cancels exactly however `b` changed.
    /// O(1) per update and never blocked, so the dose is always achieved and
    /// the updates land evenly through the circuit.
    Repair,
    /// Post-pass local commutation clearing ([`local_clearing_per_identity`]).
    /// Correct, but at production scale it costs O(circuit) scans and long
    /// commutation runs per insert, and only ~0.6% of attempts survive — the
    /// ones that do are those near the start of the gate list, so it delivers
    /// endpoint re-seeding rather than the mid-run refresh it was designed
    /// for. Kept for comparison; prefer `Repair`.
    Clearing,
    /// Global commuting float, then structure-blind trial insertion checked by
    /// forward perturbation propagation. Shuffle-agnostic by construction but
    /// low-yield: safe churn needs an ancilla that is dead-forward to the data.
    Adaptive,
}

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
    /// Band-refresh dose (`b ^= data & aux` updates). `0` = off.
    pub rerand_dose: usize,
    /// Data/aux discipline level: 0 = off, 1 = bare scratch, 2 = masked scratch.
    pub discipline: usize,
    /// How the dose is placed. See [`RerandMode`].
    pub rerand_mode: RerandMode,
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
            r: 0,        // auto = np (R = n)
            n_target: 0, // floor
            seed,
            blinded: true,
            tpin: true,
            rerand_dose: 0,
            discipline: 2,
            rerand_mode: RerandMode::Repair,
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
    /// Band-refresh updates actually inserted (<= `rerand_dose`).
    pub rerand_done: usize,
    /// Gates the band refresh cost in total: the updates plus their mask repair.
    pub rerand_gates: usize,
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

/// A fresh permutation of `0..n` (Fisher-Yates on the shared `rng`).
fn shuffled_indices(n: usize, rng: &mut StdRng) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.random_range(0..=i);
        v.swap(i, j);
    }
    v
}

/// K-cycle LGI half on wire `w` from ancilla sequence `cy`: w ^= a_i & a_{i+1}.
/// Gate `i` reads `cy[i]` and `cy[i+1 mod k]`, so `cy[j]` appears in exactly
/// the two gates `j` and `j-1 mod k` — the pair the band refresh repairs.
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

/// Validity check for a candidate rerand gate inserted just before `fwd[0]`,
/// given the base state `st` at that point. ACCEPTS only if the candidate's
/// perturbation provably cannot reach any DATA wire (`0..np`).
///
/// This is a STRUCTURAL over-approximation, deliberately: it tracks the set of
/// wires that could differ between the base and perturbed runs, ignoring the
/// values. A wire joins the set when a gate reading a set member writes it, and
/// never leaves — so "data wire never enters the set" is a sound guarantee for
/// every input, not just for sampled ones.
///
/// The previous formulation propagated two concrete 256-sample branches and
/// rejected only when a data wire actually differed on those lanes. That is not
/// a proof, and it does not hold: a divergence that shows up on a rare input is
/// never sampled, so the candidate is accepted and the gadget silently computes
/// the wrong function (an audit fuzz over 3000 gadgets, each verified
/// exhaustively, found 3 wrong circuits — each wrong on only 1-2 of 64 inputs).
/// A sampled test catches a divergence of probability p with confidence
/// 1-(1-p)^256, so p = 0.1% slips through 77% of the time.
///
/// The cost of soundness is yield: reconvergence (a perturbation that cancels
/// itself before reaching data) is invisible structurally, so fewer candidates
/// are accepted. [`RerandMode::Repair`] supersedes this path and achieves the
/// full dose exactly, so the trade is worth taking.
fn rerand_preserves_data(_st: &[[u64; 4]], fwd: &[XGate], cand: &XGate, np: usize) -> bool {
    let mut tainted = vec![false; np + fwd.iter().map(|g| g.target as usize).max().unwrap_or(0) + 1];
    let t = cand.target as usize;
    if t < np {
        return false; // a candidate that writes data directly is never safe
    }
    if t >= tainted.len() {
        tainted.resize(t + 1, false);
    }
    tainted[t] = true;
    for g in fwd {
        if g.ctrls.iter().any(|&(w, _)| {
            let w = w as usize;
            w < tainted.len() && tainted[w]
        }) {
            let tt = g.target as usize;
            if tt < np {
                return false; // the perturbation could reach a data wire
            }
            if tt >= tainted.len() {
                tainted.resize(tt + 1, false);
            }
            tainted[tt] = true;
        }
    }
    true // no data wire can ever differ
}

/// [`RerandMode::Adaptive`]: with NO structural knowledge, sweep a cursor
/// forward maintaining the base state; at spread-out positions try a random
/// rerand gate aux_t ^= data_d & aux_c and keep it iff it preserves the data
/// output ([`rerand_preserves_data`], a 256-sample check). Meant to run AFTER a
/// global gate-location shuffle so it cannot use the mask schedule.
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

/// [`RerandMode::Clearing`]: RC's original post-pass formulation of the band
/// refresh. Insert `dose` band updates `b ^= data_d & aux_c` at random points
/// P, made safe PER-IDENTITY: only the identities (an LGI mask's apply+remove,
/// or a read-gadget block, tagged by `id_of`) whose uses of `b` straddle P have
/// those uses commuted to one side, so identities before P keep the old `b` and
/// identities after see the new one. `b` is chosen light (fewest control-gates)
/// so few gates move; a commutation blocker aborts the attempt (the partial,
/// commutation-valid moves keep the circuit equivalent) and retries. `id_of` is
/// kept parallel to `gates` through every swap. Returns (accepted, blockers).
///
/// Measured on the production n=256 half sandwich (1.125M gates): ~0.8 s per
/// accepted insert, ~0.6% of attempts survive, and the survivors sit at median
/// 0.2% into the gate list — so in practice this re-seeds the band at the start
/// rather than refreshing it mid-run. [`RerandMode::Repair`] is the usable path.
fn local_clearing_per_identity(
    gates: &mut Vec<XGate>,
    id_of: &mut Vec<i64>,
    np: usize,
    band: &[u16],
    dose: usize,
    rng: &mut StdRng,
) -> (usize, usize) {
    use std::collections::HashMap;
    const SENT: u16 = u16::MAX; // sentinel marks the insert point (replaced before return)
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
        // Sorted, not a HashSet: the clearing order decides which gates move
        // and therefore which attempts survive, so leaving it to hash iteration
        // order would make the artifact differ run to run at a fixed seed.
        let mut straddle: Vec<i64> = sides
            .iter()
            .filter(|(_, (bf, af))| *bf && *af)
            .map(|(id, _)| *id)
            .collect();
        straddle.sort_unstable();
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
/// function as `src` on its `np` data wires; the ancillas are left dirty (see
/// the module note).
pub fn gadgetize_blinded_v5(src: &[XGate], np: usize, p: &BlindedV5Params) -> BlindedV5Output {
    let k = p.k;
    let r = if p.r == 0 { np } else { p.r };
    let n_target = p.n_target;
    let discipline = p.discipline;
    let rerand_dose = p.rerand_dose;
    assert!(r >= k, "R must be >= K");
    // The t-pin read draws its pin from outside the shared cycle and distinct
    // from `bwire`; with fewer spare wires than that the draw spins forever.
    assert!(
        !(p.blinded && p.tpin) || r >= k + 2,
        "the t-pin read needs R >= K + 2 (a pin outside the shared cycle and != bwire)"
    );
    // k = 2 emits `w ^= a0 & a1` twice (i=0 and i=1 give the same monomial), so
    // the two gates cancel and the "mask" is identically zero: the wire ends up
    // unmasked with no other symptom. Reject it rather than silently unmask.
    assert!(k != 2, "K = 2 makes cycle_on emit a self-cancelling (zero) mask");
    assert!(k >= 1, "K must be >= 1");
    // Several draws below are rejection samples that need at least two choices.
    assert!(r >= 2, "R must be >= 2");
    assert!(
        p.active_wires != 1 && np >= 2,
        "at least two active data wires are needed to seed an ancilla"
    );
    // `local_clearing_per_identity` marks its insert point with a sentinel gate
    // on wire u16::MAX and finds it by target, so no real wire may collide.
    assert!(
        np + r + 1 < u16::MAX as usize,
        "wire count must stay below the u16::MAX clearing sentinel"
    );

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

    // Identity tag per emitted gate (used by [`RerandMode::Clearing`]). A
    // persistent mask (apply + its later remove, re-emissions of the same cycle)
    // shares one id via `live_id[w]`; each blinded/plain read gadget is one block
    // id; seeds and band updates are untagged (-1). `emit`/`emitn` push a
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
    // How many live cycles currently read each ancilla, kept current by
    // `set_cycle!`. The band refresh repairs 4 gates per reader, so this lets
    // it pick a `b` that is cheap to refresh.
    let mut holders: Vec<u32> = vec![0; r];
    macro_rules! set_cycle {
        ($w:expr, $cy:expr) => {{
            let w_ = $w as usize;
            for &a in &live_cy[w_] {
                holders[a as usize - np] -= 1;
            }
            for &a in $cy.iter() {
                holders[a as usize - np] += 1;
            }
        }};
    }

    for w in 0..np {
        let cy = sample_k(&anc, k, &mut rng);
        let h = cycle_on(w as u16, &cy);
        let id = next_id;
        next_id += 1;
        emitn!(h, id);
        live[w] = h;
        set_cycle!(w, cy);
        live_cy[w] = cy;
        live_id[w] = id;
        atoms += 1;
    }

    let base_est = np + 3 * src.len() + np;
    let extra = n_target.saturating_sub(base_est);
    let slots = src.len() + 1;
    let churn_per_slot = extra as f64 / slots as f64;
    let mut churn_acc = 0.0f64;

    // Band refresh dose, spread uniformly over the source-gate boundaries so
    // the band evolves throughout the run rather than at its edges. Only
    // `Repair` is emitted inline; the other modes run as post-passes. The
    // split is integer-exact (slot i gets `dose*(i+1)/slots - dose*i/slots`)
    // so the dose is hit precisely -- a float accumulator drops the last
    // update to rounding.
    let refresh_total = if p.rerand_mode == RerandMode::Repair {
        rerand_dose
    } else {
        0
    };
    let mut rerand_done = 0usize;
    let mut rerand_gates = 0usize;

    let bwire = anc[0];

    // main loop
    for gi in 0..=src.len() {
        // ---- band refresh at this boundary --------------------------------
        // Emitting `[b-terms of every live mask reading b] ; b ^= d & c ;
        // [the same b-terms]` leaves each such mask's b-contribution as
        //   b_old (apply) ^ b_old (pre) ^ b_new (post) ^ b_new (remove) = 0,
        // so the mask still cancels exactly whatever the update did to b, and
        // every other identity is untouched. No gate moves, nothing to block.
        // A source-gate boundary is used because no read gadget is in flight
        // there: the pin toggles and blind/unblind pairs are all closed, so the
        // live masks are the only structure holding a band value.
        let nrf = refresh_total * (gi + 1) / slots - refresh_total * gi / slots;
        for _ in 0..nrf {
            // best-of-8: cheap to repair, while still reaching the whole pool.
            // (Taking the global argmin instead is ~40% cheaper per update but
            // starves a third of the band, which defeats the refresh.)
            let mut bi = rng.random_range(0..r);
            for _ in 1..8 {
                let cand = rng.random_range(0..r);
                if holders[cand] < holders[bi] {
                    bi = cand;
                }
            }
            let b = anc[bi];
            let mut repair: Vec<(XGate, i64)> = Vec::new();
            let mut affected = vec![false; np];
            for w in 0..np {
                if let Some(j) = live_cy[w].iter().position(|&a| a == b) {
                    affected[w] = true;
                    let kk = live_cy[w].len();
                    let j2 = (j + kk - 1) % kk;
                    repair.push((live[w][j].clone(), live_id[w]));
                    if j2 != j {
                        // k = 1 degenerates to a single b-term
                        repair.push((live[w][j2].clone(), live_id[w]));
                    }
                }
            }
            // Prefer a data source the repair is not currently perturbing.
            // Correctness does not depend on this (the telescoping holds for
            // any update value), so give up after a few tries rather than spin.
            let mut d = rng.random_range(0..active);
            for _ in 0..8 {
                if !affected[d] {
                    break;
                }
                d = rng.random_range(0..active);
            }
            let mut c = anc[rng.random_range(0..r)];
            while c == b {
                c = anc[rng.random_range(0..r)];
            }
            // The repair gates commute with each other (distinct data targets,
            // ancilla controls only), so order each side independently and the
            // block is not a literal palindrome around the update.
            let pre = shuffled_indices(repair.len(), &mut rng);
            let post = shuffled_indices(repair.len(), &mut rng);
            for &i in &pre {
                emit!(repair[i].0.clone(), repair[i].1);
            }
            emit!(
                XGate::conj(
                    b,
                    [
                        (d as u16, rng.random_bool(0.5)),
                        (c, rng.random_bool(0.5))
                    ]
                )
                .unwrap(),
                -1
            );
            for &i in &post {
                emit!(repair[i].0.clone(), repair[i].1);
            }
            rerand_done += 1;
            rerand_gates += 2 * repair.len() + 1;
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
            set_cycle!(w, cy);
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
            set_cycle!(w1, cy);
            live_cy[w1 as usize] = cy.clone();
            live_id[w1 as usize] = id1;
            atoms += 1;
            let id2 = next_id;
            next_id += 1;
            emitn!(s2, id2); // new mask on w2
            emitn!(live[w2 as usize], live_id[w2 as usize]); // remove old w2 mask
            live[w2 as usize] = s2.clone();
            set_cycle!(w2, cy);
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
                set_cycle!(w, cy);
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

    // The non-inline band-refresh modes run as post-passes.
    if rerand_dose > 0 {
        match p.rerand_mode {
            RerandMode::Repair => {} // already emitted, spread through the run
            RerandMode::Clearing => {
                // `bwire` is excluded from the clearing band. It is the blind
                // wire of EVERY read gadget, and the blind/unblind pair
                // `w2 ^= bwire & lit(tp_w,!tp)` shares `tp_w` with the payload
                // at the opposite polarity -- so `XGate::collides` (correctly)
                // exempts them and the clearing is free to slide an unblind
                // back next to its blind, where the identical pair annihilates
                // and the payload then runs on a bare, unmasked operand. That
                // is exactly the k=1 exposure the blinded read exists to
                // prevent, and it grows linearly with the dose.
                let (acc, _blk) = local_clearing_per_identity(
                    &mut out,
                    &mut id_of,
                    np,
                    &anc[1..],
                    rerand_dose,
                    &mut rng,
                );
                rerand_done = acc;
                rerand_gates = acc;
            }
            RerandMode::Adaptive => {
                commuting_shuffle(&mut out, &mut rng);
                let inputs: Vec<[u64; 4]> = (0..np)
                    .map(|_| [rng.random(), rng.random(), rng.random(), rng.random()])
                    .collect();
                let (new_out, acc) = adaptive_rerand_incremental(
                    &out,
                    np,
                    np + r,
                    &anc,
                    rerand_dose,
                    &inputs,
                    &mut rng,
                );
                out = new_out;
                rerand_done = acc;
                rerand_gates = acc;
            }
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
        rerand_gates,
        r_used: r,
    }
}
