//! ANF-only window synthesis: the bidirectional search of `anf_synth`
//! (`runs/ncdb_ab_20261001/synth/src/synth.rs`, production since
//! 2026-10-08; its README and REPORT.md describe the algorithm and its
//! measured results), ported onto a width-generic monomial.
//!
//! SEPARATION GUARANTEE. [`synthesize`] receives a [`WindowSpec`] (the size
//! of the local variable universe and the delivered polynomials, i.e.
//! exactly what a window ships as) plus the run seed and the window index.
//! Nothing derived from the original gates reaches this module: it imports
//! no circuit type, and every iteration order is a deterministic function
//! of the spec content:
//!   * the spec is canonicalised on entry (wires ascending, monomials
//!     sorted, even multiplicities cancelled),
//!   * candidate moves are collected into vectors, deduplicated and sorted
//!     by their canonical key before they are scored,
//!   * ties are broken by a PRNG seeded from (seed, window index, restart)
//!     only.
//!
//! The only timing dependence is the per-window deadline
//! (`SynthParams::time_limit`): when it truncates the restarts the kept
//! circuit may depend on machine load, and the window is flagged
//! `deadline_hit`.
//!
//! Model. The state is the residual specification `P_w` (one polynomial per
//! wire; identity `{x_w}` for untouched wires). Emitting the NEXT gate
//! `g = (t; L)` of the synthesized circuit (execution order) substitutes
//! `x_t <- x_t + L(x)` in every polynomial (`F = F' o g`, so `F' = F o g`
//! because `g` is an involution). A BACKWARD gate (appended after the
//! remaining residual circuit) sets `P_t <- P_t + L(P_a, P_b)` instead. The
//! circuit is complete when every `P_w == {x_w}`. Distance
//! `D = sum_w |P_w xor {x_w}|`, degree-weighted.
//!
//! Gates are emitted over the LOCAL variable ids of the spec; the caller
//! maps them back to global wires (`spec::LocalUniverse`).
use super::mono::{Mono, Rng};
use rustc_hash::{FxHashMap, FxHashSet};
use std::time::{Duration, Instant};

/// A synthesized gate over local variable ids: `x_t ^= comp XOR AND_i lit_i`,
/// at most two controls, controls in ascending order for `k = 2`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SGate {
    pub t: u32,
    pub comp: bool,
    pub k: u8,
    pub c: [u32; 2],
    pub p: [bool; 2],
}

impl SGate {
    /// NOT: the gate computes `comp XOR AND(literals)` with the empty AND = 1,
    /// so `(t, comp=false, k=0)` is a NOT and `(t, comp=true, k=0)` a no-op.
    pub fn not(t: u32) -> SGate {
        SGate {
            t,
            comp: false,
            k: 0,
            c: [0; 2],
            p: [false; 2],
        }
    }

    pub fn cnot(t: u32, a: u32, pa: bool, comp: bool) -> SGate {
        SGate {
            t,
            comp,
            k: 1,
            c: [a, 0],
            p: [pa, false],
        }
    }

    /// Canonical control order (`a < b`) so identical gates compare equal.
    pub fn tof(t: u32, a: u32, pa: bool, b: u32, pb: bool, comp: bool) -> SGate {
        let ((x, px), (y, py)) = if a <= b {
            ((a, pa), (b, pb))
        } else {
            ((b, pb), (a, pa))
        };
        SGate {
            t,
            comp,
            k: 2,
            c: [x, y],
            p: [px, py],
        }
    }

    #[inline]
    pub fn target(&self) -> u32 {
        self.t
    }

    #[inline]
    pub fn controls(&self) -> &[u32] {
        &self.c[..self.k as usize]
    }

    #[inline]
    pub fn has_control(&self, w: u32) -> bool {
        self.controls().contains(&w)
    }

    /// Two gates commute iff neither target is a control of the other.
    #[inline]
    pub fn commutes(&self, o: &SGate) -> bool {
        !self.has_control(o.t) && !o.has_control(self.t)
    }

    pub fn is_noop(&self) -> bool {
        self.k == 0 && self.comp
    }
}

/// What the synthesizer sees: the number of local variables and, per
/// changed wire, its exit polynomial as a monomial list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowSpec<const W: usize> {
    pub num_wires: u32,
    pub polys: Vec<(u32, Vec<Mono<W>>)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SynthParams {
    /// Number of top candidates expanded by the 2-step lookahead.
    pub beam: usize,
    /// Wall-clock limit per window (seconds), shared by all restarts.
    pub time_limit: f64,
    pub restarts: usize,
    /// Step cap = `step_factor * monomials + 64` emitted gates.
    pub step_factor: usize,
    /// Consecutive non-improving (perturbation) moves tolerated.
    pub stall_limit: usize,
    /// Extra weight per degree above 2 (a monomial weighs `4 + (deg-2)*deg_weight`).
    pub deg_weight: i64,
    /// Maximum members of one amortised helper batch.
    pub macro_batch: usize,
    /// Consider helper macros before perturbing.
    pub use_macros: bool,
    /// Number of best-estimated forward candidates rescored exactly per step.
    pub exact_k: usize,
    /// Score noise added to every candidate (restart r uses `noise*r` when
    /// `noise_schedule` is set); 4 = one monomial.
    pub noise: i64,
    pub noise_schedule: bool,
    /// Number of best-estimated BACKWARD candidates scored exactly per step;
    /// 0 disables backward moves (the forward-only baseline).
    pub back_k: usize,
    /// On a tie between the best forward and backward move prefer the backward one.
    pub back_prefer: bool,
    /// Cap on the per-monomial lookup list length in backward candidate generation.
    pub back_look: usize,
    /// Weight of a constant / linear error monomial (degree 2 weighs 4).
    pub w0: i64,
    pub w1: i64,
    /// Alternate the forward/backward tie preference across restarts.
    pub back_alt: bool,
}

impl Default for SynthParams {
    /// The production defaults of `anf_synth` (README "Defaults").
    fn default() -> Self {
        SynthParams {
            beam: 6,
            time_limit: 30.0,
            restarts: 16,
            step_factor: 20,
            stall_limit: 60,
            deg_weight: 4,
            macro_batch: 32,
            use_macros: true,
            exact_k: 48,
            noise: 0,
            noise_schedule: false,
            back_k: 32,
            back_prefer: true,
            back_look: 8,
            w0: 1,
            w1: 2,
            back_alt: false,
        }
    }
}

impl SynthParams {
    /// Bounds shared with the binary's argument parser.
    pub fn validate(&self) -> Result<(), String> {
        if !(self.time_limit > 0.0 && self.time_limit.is_finite() && self.time_limit <= 1e7) {
            return Err(format!(
                "--time-limit {}: must be a positive number of seconds",
                self.time_limit
            ));
        }
        if self.restarts == 0 || self.restarts > 10_000 {
            return Err(format!(
                "--restarts {}: must be in 1..=10000",
                self.restarts
            ));
        }
        if self.beam > 1_000 {
            return Err(format!("--beam {}: must be at most 1000", self.beam));
        }
        if self.exact_k == 0 || self.exact_k > 1_000_000 {
            return Err(format!("exact_k {}: must be in 1..=1000000", self.exact_k));
        }
        if self.back_k > 1_000_000 || self.back_look == 0 {
            return Err("back_k / back_look out of range".into());
        }
        if self.w0 < 0 || self.w1 < 0 || self.deg_weight < 0 {
            return Err("weights must be nonnegative".into());
        }
        if self.step_factor == 0 || self.stall_limit == 0 || self.macro_batch == 0 {
            return Err("step_factor, stall_limit and macro_batch must be positive".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MoveTag {
    Greedy = 0,
    Lookahead = 1,
    Sink = 2,
    Helper = 3,
    Cycle = 4,
    Linear = 5,
    Other = 6,
    Back = 7,
}
pub const NTAGS: usize = 8;
pub const TAG_NAMES: [&str; NTAGS] = [
    "greedy",
    "lookahead",
    "sink",
    "helper",
    "cycle",
    "linear",
    "other",
    "back",
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SynthStatus {
    Converged,
    StepCap,
    Timeout,
    Stalled,
}

impl SynthStatus {
    pub fn name(self) -> &'static str {
        match self {
            SynthStatus::Converged => "converged",
            SynthStatus::StepCap => "stepcap",
            SynthStatus::Timeout => "timeout",
            SynthStatus::Stalled => "stalled",
        }
    }
}

#[derive(Clone, Debug)]
pub struct SynthOutput {
    pub gates: Vec<SGate>,
    pub tags: Vec<MoveTag>,
    pub status: SynthStatus,
    pub steps: usize,
    /// Restarts actually run (< `params.restarts` when the deadline cut them).
    pub restarts_used: usize,
    /// The per-window wall-clock limit truncated the search: a run ended
    /// with status `Timeout`, or the restart loop stopped before
    /// `params.restarts` because the deadline had passed. Such a window's
    /// output depends on timing, not on the spec alone.
    pub deadline_hit: bool,
    pub note: String,
}

type Poly<const W: usize> = FxHashSet<Mono<W>>;

// ---------------------------------------------------------------------------
// residual specification state
// ---------------------------------------------------------------------------
struct State<const W: usize> {
    nw: u32,
    polys: Vec<Poly<W>>,
    /// `occ[v]`: wire -> the monomials of `P_wire` containing variable `v`.
    occ: Vec<FxHashMap<u32, FxHashSet<Mono<W>>>>,
    /// `|P_w xor {x_w}|`
    dist: Vec<u32>,
    errs: i64,
    score: i64,
    dw: i64,
    w0: i64,
    w1: i64,
    /// Modification stamps: `stamp[w]` = tick of the last toggle on `P_w`.
    stamp: Vec<u64>,
    tick: u64,
}

impl<const W: usize> State<W> {
    fn new(spec: &WindowSpec<W>, dw: i64, w0: i64, w1: i64) -> State<W> {
        let nw = spec.num_wires as usize;
        let mut st = State {
            nw: spec.num_wires,
            polys: vec![Poly::default(); nw],
            occ: vec![FxHashMap::default(); nw],
            dist: vec![0; nw],
            errs: 0,
            score: 0,
            dw,
            w0,
            w1,
            stamp: vec![0; nw],
            tick: 0,
        };
        for w in 0..nw {
            st.polys[w].insert(Mono::var(w as u32));
            let mut set = FxHashSet::default();
            set.insert(Mono::var(w as u32));
            st.occ[w].insert(w as u32, set);
        }
        for (w, p) in &spec.polys {
            st.toggle(*w, Mono::var(*w)); // drop the identity monomial
            for m in p {
                st.toggle(*w, *m);
            }
        }
        st
    }

    #[inline]
    fn wt_deg(&self, d: i64) -> i64 {
        if d > 2 {
            4 + (d - 2) * self.dw
        } else if d == 2 {
            4
        } else if d == 1 {
            self.w1
        } else {
            self.w0
        }
    }

    #[inline]
    fn wt(&self, m: &Mono<W>) -> i64 {
        self.wt_deg(m.degree() as i64)
    }

    fn toggle(&mut self, w: u32, q: Mono<W>) {
        let wanted = q == Mono::var(w);
        let wq = self.wt(&q);
        self.stamp[w as usize] = self.tick;
        if self.polys[w as usize].remove(&q) {
            for v in q.vars() {
                let e = self.occ[v as usize].get_mut(&w).expect("occ");
                e.remove(&q);
                if e.is_empty() {
                    self.occ[v as usize].remove(&w);
                }
            }
            if wanted {
                self.dist[w as usize] += 1;
                self.errs += 1;
                self.score += wq;
            } else {
                self.dist[w as usize] -= 1;
                self.errs -= 1;
                self.score -= wq;
            }
        } else {
            self.polys[w as usize].insert(q);
            for v in q.vars() {
                self.occ[v as usize].entry(w).or_default().insert(q);
            }
            if wanted {
                self.dist[w as usize] -= 1;
                self.errs -= 1;
                self.score -= wq;
            } else {
                self.dist[w as usize] += 1;
                self.errs += 1;
                self.score += wq;
            }
        }
    }

    /// Substitute `x_t <- x_t + sum(l)` in every polynomial.
    fn apply(&mut self, t: u32, l: &[Mono<W>]) {
        self.tick += 1;
        let mut wires: Vec<u32> = self.occ[t as usize].keys().copied().collect();
        wires.sort_unstable();
        let mut ms: Vec<Mono<W>> = Vec::new();
        for w in wires {
            ms.clear();
            ms.extend(self.occ[t as usize][&w].iter().copied());
            ms.sort_unstable();
            for m in &ms {
                let r = m.without(t);
                for lm in l {
                    self.toggle(w, r.or(lm));
                }
            }
        }
    }

    /// Backward (output-side) gate: `P_t <- P_t + Q` where `Q = L(P_a, P_b)`
    /// is already expanded (deduplicated) into monomials. Only `P_t` changes.
    fn apply_back(&mut self, t: u32, q: &[Mono<W>]) {
        self.tick += 1;
        for m in q {
            self.toggle(t, *m);
        }
    }

    /// Exact score change of `apply_back(t, q)` (`q` deduplicated).
    fn back_delta(&self, t: u32, q: &[Mono<W>]) -> i64 {
        let p = &self.polys[t as usize];
        let xt = Mono::var(t);
        let mut ds = 0i64;
        for m in q {
            let present = p.contains(m);
            let wanted = *m == xt;
            if present != wanted {
                ds -= self.wt(m);
            } else {
                ds += self.wt(m);
            }
        }
        ds
    }

    /// Exact (score, error-count) change of `apply(t, l)` without mutating.
    fn delta(&self, t: u32, l: &[Mono<W>], scratch: &mut Vec<Mono<W>>) -> (i64, i64) {
        let mut ds = 0i64;
        let mut de = 0i64;
        for (&w, ms) in &self.occ[t as usize] {
            scratch.clear();
            let p = &self.polys[w as usize];
            let xw = Mono::var(w);
            for m in ms {
                let r = m.without(t);
                for lm in l {
                    scratch.push(r.or(lm));
                }
            }
            scratch.sort_unstable();
            let mut i = 0;
            while i < scratch.len() {
                let mut j = i + 1;
                while j < scratch.len() && scratch[j] == scratch[i] {
                    j += 1;
                }
                if (j - i) & 1 == 1 {
                    let q = scratch[i];
                    let present = p.contains(&q);
                    let wanted = q == xw;
                    let sgn = if present != wanted { -1 } else { 1 };
                    ds += sgn * self.wt(&q);
                    de += sgn;
                }
                i = j;
            }
        }
        (ds, de)
    }

    /// Latest stamp among the polynomials that contain `x_t`.
    fn max_stamp(&self, t: u32) -> u64 {
        self.occ[t as usize]
            .keys()
            .map(|&w| self.stamp[w as usize])
            .max()
            .unwrap_or(0)
    }

    /// `x_t` occurs only as the lone monomial `{x_t}` of `P_t`.
    fn is_sink(&self, t: u32) -> bool {
        let o = &self.occ[t as usize];
        o.len() == 1
            && o.get(&t).is_some_and(|s| s.len() == 1)
            && self.polys[t as usize].contains(&Mono::var(t))
    }

    /// Identity polynomial and `x_h` appears in no other polynomial.
    fn is_clean(&self, h: u32) -> bool {
        self.dist[h as usize] == 0 && self.occ[h as usize].len() == 1
    }

    fn target_affine(&self, t: u32) -> bool {
        self.polys[t as usize].iter().all(|m| m.degree() <= 1)
    }
}

// ---------------------------------------------------------------------------
// elementary candidates (one gate, <= 2 controls, any polarity / complement)
// ---------------------------------------------------------------------------
/// `L = x_a x_b + alpha x_a + beta x_b + gamma` (k=2), `x_a + gamma` (k=1), `1` (k=0).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
struct Cand {
    t: u32,
    k: u8,
    a: u32,
    b: u32,
    /// bit0 gamma, bit1 alpha, bit2 beta
    flags: u8,
}

impl Cand {
    fn not(t: u32) -> Cand {
        Cand {
            t,
            k: 0,
            a: u32::MAX,
            b: u32::MAX,
            flags: 0,
        }
    }
    fn cnot(t: u32, a: u32, gamma: bool) -> Cand {
        Cand {
            t,
            k: 1,
            a,
            b: u32::MAX,
            flags: gamma as u8,
        }
    }
    fn tof(t: u32, a: u32, b: u32, alpha: bool, beta: bool, gamma: bool) -> Cand {
        let (a, b, alpha, beta) = if a < b {
            (a, b, alpha, beta)
        } else {
            (b, a, beta, alpha)
        };
        Cand {
            t,
            k: 2,
            a,
            b,
            flags: (gamma as u8) | ((alpha as u8) << 1) | ((beta as u8) << 2),
        }
    }
    fn l<const W: usize>(&self, out: &mut [Mono<W>; 4]) -> usize {
        let gamma = self.flags & 1 != 0;
        let alpha = self.flags & 2 != 0;
        let beta = self.flags & 4 != 0;
        let mut n;
        match self.k {
            0 => {
                out[0] = Mono::ONE;
                n = 1;
            }
            1 => {
                out[0] = Mono::var(self.a);
                n = 1;
                if gamma {
                    out[1] = Mono::ONE;
                    n = 2;
                }
            }
            _ => {
                out[0] = Mono::var(self.a).with(self.b);
                n = 1;
                if alpha {
                    out[n] = Mono::var(self.a);
                    n += 1;
                }
                if beta {
                    out[n] = Mono::var(self.b);
                    n += 1;
                }
                if gamma {
                    out[n] = Mono::ONE;
                    n += 1;
                }
            }
        }
        n
    }
    fn gate(&self) -> SGate {
        family_gate(self.t, self.k, self.a, self.b, self.flags)
    }
}

/// The gate of the family with polynomial `x_a x_b + alpha x_a + beta x_b +
/// gamma` (k=2), `x_a + gamma` (k=1) or `1` (k=0); `flags` = bit0 gamma,
/// bit1 alpha, bit2 beta.
fn family_gate(t: u32, k: u8, a: u32, b: u32, flags: u8) -> SGate {
    let gamma = flags & 1 != 0;
    let alpha = flags & 2 != 0;
    let beta = flags & 4 != 0;
    match k {
        0 => SGate::not(t),
        1 => SGate::cnot(t, a, true, gamma),
        _ => {
            // L = (x_a + pa')(x_b + pb') + c = x_a x_b + pb' x_a + pa' x_b + pa'pb' + c
            let pa = !beta;
            let pb = !alpha;
            let comp = gamma ^ (alpha && beta);
            SGate::tof(t, a, pa, b, pb, comp)
        }
    }
}

/// The polynomial `L` of an arbitrary gate of the family (for replaying
/// macro gates through the exact substitution).
pub(super) fn gate_l<const W: usize>(g: &SGate, out: &mut [Mono<W>; 4]) -> usize {
    match g.k {
        0 => {
            // empty AND = 1, so L = 1 + comp
            out[0] = Mono::ONE;
            if g.comp { 0 } else { 1 }
        }
        1 => {
            // (x_a + pa') + comp
            out[0] = Mono::var(g.c[0]);
            if !g.p[0] ^ g.comp {
                out[1] = Mono::ONE;
                2
            } else {
                1
            }
        }
        _ => {
            let a = g.c[0];
            let b = g.c[1];
            let pa_ = !g.p[0];
            let pb_ = !g.p[1];
            let mut n = 0;
            out[n] = Mono::var(a).with(b);
            n += 1;
            if pb_ {
                out[n] = Mono::var(a);
                n += 1;
            }
            if pa_ {
                out[n] = Mono::var(b);
                n += 1;
            }
            if (pa_ && pb_) ^ g.comp {
                out[n] = Mono::ONE;
                n += 1;
            }
            n
        }
    }
}

struct Buffers<const W: usize> {
    cands: Vec<Cand>,
    pbuf: Vec<Mono<W>>,
    ebuf: Vec<Mono<W>>,
    scratch: Vec<Mono<W>>,
    scored: Vec<(i64, u8, u64, u32)>,
    /// candidate -> (score delta, tick at evaluation); valid while no
    /// polynomial containing the target changed since
    cache: FxHashMap<Cand, (i64, u64)>,
    /// single-term candidate -> -2 * weighted number of cancelled error monomials
    approx: FxHashMap<Cand, i64>,
}

impl<const W: usize> Buffers<W> {
    fn new() -> Buffers<W> {
        Buffers {
            cands: Vec::new(),
            pbuf: Vec::new(),
            ebuf: Vec::new(),
            scratch: Vec::new(),
            scored: Vec::new(),
            cache: FxHashMap::default(),
            approx: FxHashMap::default(),
        }
    }
}

/// Focused candidate generation with free approximate scoring.
///
/// For every error monomial `m2` of a wire `w` and every subset `s` of `m2`
/// with `|s| <= 2`, `r = m2 \ s`: if `P_w` holds a monomial `m1 = r * x_t`
/// with `t` not in `m2`, every single-term gate `(t; l)` with
/// `s <= l <= m2`, `|l| <= 2` turns `m1` into `m1 + (r u l) = m1 + m2` and
/// cancels `m2`. The sweep therefore counts, for every single-term
/// candidate `(t; l)`, the weighted number of error monomials it cancels
/// (`neg`). Since the move toggles exactly one monomial per occurrence of
/// `x_t`, its score delta is `N_t(l) - 2*neg(t,l)` up to duplicate-toggle
/// collisions (`N_t(l)` = weighted number of occurrences of `x_t`).
/// Multi-term gates (negative polarities / complement) add their terms'
/// estimates. Only the best `exact_k` estimates are then rescored exactly.
/// Index `r -> {t}` over `P_w` keeps the sweep at `|E_w| * O(d^2)`.
fn gen_cands<const W: usize>(st: &State<W>, b: &mut Buffers<W>) {
    b.cands.clear();
    b.approx.clear();
    let mut index: FxHashMap<Mono<W>, Vec<u32>> = FxHashMap::default();
    for w in 0..st.nw {
        if st.dist[w as usize] == 0 {
            continue;
        }
        let xw = Mono::var(w);
        let p = &st.polys[w as usize];
        b.ebuf.clear();
        b.ebuf.extend(p.iter().filter(|m| **m != xw).copied());
        if !p.contains(&xw) {
            b.ebuf.push(xw);
        }
        b.ebuf.sort_unstable();
        b.pbuf.clear();
        b.pbuf.extend(p.iter().copied());
        b.pbuf.sort_unstable();
        index.clear();
        for m in &b.pbuf {
            for t in m.vars() {
                index.entry(m.without(t)).or_default().push(t);
            }
        }
        let e = std::mem::take(&mut b.ebuf);
        for m2 in &e {
            let wq = 2 * st.wt(m2);
            let mv = m2.to_vec();
            let d = mv.len();
            for si in 0..=(d * (d + 1) / 2) {
                let (r, s0, s1): (Mono<W>, Option<u32>, Option<u32>) = if si == 0 {
                    (*m2, None, None)
                } else if si <= d {
                    let a = mv[si - 1];
                    (m2.without(a), Some(a), None)
                } else {
                    let mut pi = si - d - 1;
                    let mut i = 0;
                    let mut found = None;
                    while i < d {
                        let cnt = d - 1 - i;
                        if pi < cnt {
                            found = Some((i, i + 1 + pi));
                            break;
                        }
                        pi -= cnt;
                        i += 1;
                    }
                    match found {
                        Some((i, j)) => {
                            (m2.without(mv[i]).without(mv[j]), Some(mv[i]), Some(mv[j]))
                        }
                        None => break,
                    }
                };
                let Some(ts) = index.get(&r) else { continue };
                let rdeg = r.degree();
                for &t in ts {
                    if m2.contains(t) {
                        continue;
                    }
                    match (s0, s1) {
                        (None, None) => {
                            *b.approx.entry(Cand::not(t)).or_insert(0) -= wq;
                            if rdeg <= 6 {
                                let rv = r.to_vec();
                                for (i, &c) in rv.iter().enumerate() {
                                    *b.approx.entry(Cand::cnot(t, c, false)).or_insert(0) -= wq;
                                    for &dd in &rv[i + 1..] {
                                        *b.approx
                                            .entry(Cand::tof(t, c, dd, false, false, false))
                                            .or_insert(0) -= wq;
                                    }
                                }
                            }
                        }
                        (Some(a), None) => {
                            *b.approx.entry(Cand::cnot(t, a, false)).or_insert(0) -= wq;
                            if rdeg <= 6 {
                                for c in r.vars() {
                                    *b.approx
                                        .entry(Cand::tof(t, a, c, false, false, false))
                                        .or_insert(0) -= wq;
                                }
                            }
                        }
                        (Some(a), Some(bb)) => {
                            *b.approx
                                .entry(Cand::tof(t, a, bb, false, false, false))
                                .or_insert(0) -= wq;
                        }
                        _ => {}
                    }
                }
            }
        }
        b.ebuf = e;
    }
    // N_t(k): weighted number of occurrences of x_t when the toggled
    // monomial has degree deg(m) - 1 + k
    let mut nt: FxHashMap<(u32, u8), i64> = FxHashMap::default();
    let mut n_of = |t: u32, k: u8| -> i64 {
        *nt.entry((t, k)).or_insert_with(|| {
            let mut n = 0i64;
            for ms in st.occ[t as usize].values() {
                for m in ms {
                    let d = m.degree() as i64 - 1 + k as i64;
                    n += st.wt_deg(d);
                }
            }
            n
        })
    };
    // single-term estimates
    let mut singles: Vec<(Cand, i64)> = Vec::with_capacity(b.approx.len());
    for (c, neg) in b.approx.iter() {
        let k = c.k.min(2);
        singles.push((*c, n_of(c.t, k) + *neg));
    }
    singles.sort_unstable();
    let est: FxHashMap<Cand, i64> = singles.iter().copied().collect();
    b.scored.clear();
    for (c, d) in &singles {
        b.cands.push(*c);
        b.scored.push((*d, c.k, 0, (b.cands.len() - 1) as u32));
        // multi-term variants whose extra terms each help
        let d1 = est
            .get(&Cand::not(c.t))
            .copied()
            .unwrap_or_else(|| n_of(c.t, 0));
        match c.k {
            1 => {
                if d1 < 0 {
                    b.cands.push(Cand::cnot(c.t, c.a, true));
                    b.scored.push((*d + d1, 1, 0, (b.cands.len() - 1) as u32));
                }
            }
            2 => {
                let da = est
                    .get(&Cand::cnot(c.t, c.a, false))
                    .copied()
                    .unwrap_or_else(|| n_of(c.t, 1));
                let db = est
                    .get(&Cand::cnot(c.t, c.b, false))
                    .copied()
                    .unwrap_or_else(|| n_of(c.t, 1));
                for flags in 1u8..8 {
                    let gamma = flags & 1 != 0;
                    let alpha = flags & 2 != 0;
                    let beta = flags & 4 != 0;
                    if (gamma && d1 >= 0) || (alpha && da >= 0) || (beta && db >= 0) {
                        continue;
                    }
                    let dd = *d
                        + if gamma { d1 } else { 0 }
                        + if alpha { da } else { 0 }
                        + if beta { db } else { 0 };
                    b.cands.push(Cand::tof(c.t, c.a, c.b, alpha, beta, gamma));
                    b.scored.push((dd, 2, 0, (b.cands.len() - 1) as u32));
                }
            }
            _ => {}
        }
    }
}

/// Exact rescoring of the best `exact_k` estimates (stamp-validated cache);
/// `scored` ends sorted by (exact-or-estimated delta, controls, rnd).
fn score_cands<const W: usize>(
    st: &State<W>,
    b: &mut Buffers<W>,
    rng: &mut Rng,
    exact_k: usize,
    noise: i64,
) {
    for s in b.scored.iter_mut() {
        s.2 = rng.next_u64();
        if noise > 0 {
            s.0 += (s.2 % (noise as u64 + 1)) as i64;
        }
    }
    b.scored.sort_unstable();
    let mut lbuf = [Mono::<W>::ONE; 4];
    if b.cache.len() > 1_000_000 {
        b.cache.clear();
    }
    let n = b.scored.len().min(exact_k);
    for i in 0..n {
        let c = b.cands[b.scored[i].3 as usize];
        let stamp = st.max_stamp(c.t);
        let ds = match b.cache.get(&c) {
            Some(&(ds, tk)) if tk >= stamp => ds,
            _ => {
                let nl = c.l(&mut lbuf);
                let (ds, _de) = st.delta(c.t, &lbuf[..nl], &mut b.scratch);
                b.cache.insert(c, (ds, st.tick));
                ds
            }
        };
        b.scored[i].0 = ds
            + if noise > 0 {
                (b.scored[i].2 % (noise as u64 + 1)) as i64
            } else {
                0
            };
    }
    // the exactly scored block first; the estimated tail follows
    b.scored[..n].sort_unstable();
}

// ---------------------------------------------------------------------------
// backward (output-side) candidates. Appending the gate (t; L) AFTER the
// residual circuit F' gives F = g o F'' with P''_t = P_t + L(P_a, P_b):
// only P_t changes, but L is applied to the residual POLYNOMIALS of the
// control wires, so a product structure P_a * P_b inside P_t is removed by
// one gate. Candidates are voted by the error monomials of P_t that split
// as m1 * m2 with m1 in P_a and m2 in P_b (identity wires included), the
// best `back_k` estimates (|P_a||P_b| created minus votes) are scored
// exactly over all polarity/complement variants.
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
struct BCand {
    t: u32,
    k: u8,
    a: u32,
    b: u32,
    /// bit0 gamma (+1), bit1 alpha (+P_a), bit2 beta (+P_b)
    flags: u8,
}

impl BCand {
    fn gate(&self) -> SGate {
        family_gate(self.t, self.k, self.a, self.b, self.flags)
    }
}

struct BackBuffers<const W: usize> {
    idx: FxHashMap<Mono<W>, Vec<u32>>,
    votes: FxHashMap<(u32, u8, u32, u32), i64>,
    est: Vec<(i64, u32, u8, u32, u32)>,
    scored: Vec<(i64, u8, u64, BCand)>,
    pa: Vec<Mono<W>>,
    pb: Vec<Mono<W>>,
    q: Vec<Mono<W>>,
}

impl<const W: usize> BackBuffers<W> {
    fn new() -> BackBuffers<W> {
        BackBuffers {
            idx: FxHashMap::default(),
            votes: FxHashMap::default(),
            est: Vec::new(),
            scored: Vec::new(),
            pa: Vec::new(),
            pb: Vec::new(),
            q: Vec::new(),
        }
    }
}

/// Sort and cancel even multiplicities (GF(2) sum of a monomial multiset).
fn cancel_sorted<const W: usize>(v: &mut Vec<Mono<W>>) {
    v.sort_unstable();
    let mut w = 0usize;
    let mut i = 0usize;
    while i < v.len() {
        let mut j = i + 1;
        while j < v.len() && v[j] == v[i] {
            j += 1;
        }
        if (j - i) & 1 == 1 {
            v[w] = v[i];
            w += 1;
        }
        i = j;
    }
    v.truncate(w);
}

fn sorted_poly<const W: usize>(st: &State<W>, w: u32, out: &mut Vec<Mono<W>>) {
    out.clear();
    out.extend(st.polys[w as usize].iter().copied());
    out.sort_unstable();
}

/// `Q = L(P_a, P_b)` for the candidate, deduplicated; `false` if too large.
fn back_poly<const W: usize>(
    st: &State<W>,
    c: &BCand,
    bb: &mut BackBuffers<W>,
    cap: usize,
) -> bool {
    bb.q.clear();
    match c.k {
        0 => {
            bb.q.push(Mono::ONE);
            true
        }
        1 => {
            sorted_poly(st, c.a, &mut bb.pa);
            bb.q.extend_from_slice(&bb.pa);
            if c.flags & 1 != 0 {
                bb.q.push(Mono::ONE);
            }
            cancel_sorted(&mut bb.q);
            true
        }
        _ => {
            sorted_poly(st, c.a, &mut bb.pa);
            sorted_poly(st, c.b, &mut bb.pb);
            if bb.pa.len() * bb.pb.len() > cap {
                return false;
            }
            for x in &bb.pa {
                for y in &bb.pb {
                    bb.q.push(x.or(y));
                }
            }
            if c.flags & 2 != 0 {
                bb.q.extend_from_slice(&bb.pa);
            }
            if c.flags & 4 != 0 {
                bb.q.extend_from_slice(&bb.pb);
            }
            if c.flags & 1 != 0 {
                bb.q.push(Mono::ONE);
            }
            cancel_sorted(&mut bb.q);
            true
        }
    }
}

/// Wires whose residual polynomial contains monomial `m` (identity wires for
/// `m = x_v` included), capped at `cap`, deterministic order.
fn back_lookup<const W: usize>(
    st: &State<W>,
    idx: &FxHashMap<Mono<W>, Vec<u32>>,
    m: &Mono<W>,
    cap: usize,
    out: &mut Vec<u32>,
) {
    out.clear();
    if m.degree() == 1 {
        let v = m.first_var().unwrap();
        if st.dist[v as usize] == 0 {
            out.push(v);
        }
    }
    if let Some(ws) = idx.get(m) {
        for &w in ws {
            if out.len() >= cap {
                break;
            }
            out.push(w);
        }
    }
}

fn gen_back<const W: usize>(
    st: &State<W>,
    bb: &mut BackBuffers<W>,
    params: &SynthParams,
    rng: &mut Rng,
) {
    bb.scored.clear();
    bb.votes.clear();
    bb.est.clear();
    // exact-monomial index over the changed wires
    bb.idx.clear();
    let changed: Vec<u32> = (0..st.nw).filter(|&w| st.dist[w as usize] != 0).collect();
    for &w in &changed {
        let mut ms: Vec<Mono<W>> = st.polys[w as usize].iter().copied().collect();
        ms.sort_unstable();
        for m in ms {
            bb.idx.entry(m).or_default().push(w);
        }
    }
    let mut la: Vec<u32> = Vec::new();
    let mut lb: Vec<u32> = Vec::new();
    let mut ebuf: Vec<Mono<W>> = Vec::new();
    for &t in &changed {
        let xt = Mono::var(t);
        let p = &st.polys[t as usize];
        ebuf.clear();
        ebuf.extend(p.iter().filter(|m| **m != xt).copied());
        if !p.contains(&xt) {
            ebuf.push(xt);
        }
        ebuf.sort_unstable();
        for m in &ebuf {
            let wq = st.wt(m);
            let d = m.degree() as usize;
            if d == 0 {
                *bb.votes.entry((t, 0, u32::MAX, u32::MAX)).or_insert(0) += wq;
                continue;
            }
            // k = 1: m in P_a
            back_lookup(st, &bb.idx, m, params.back_look, &mut la);
            for &a in &la {
                if a != t {
                    *bb.votes.entry((t, 1, a, u32::MAX)).or_insert(0) += wq;
                }
            }
            if d > 6 {
                continue;
            }
            // k = 2: disjoint splits m = m1 * m2, m1 holding the lowest var
            let vs = m.to_vec();
            let full = (1usize << d) - 1;
            let mut mask = 1usize;
            while mask <= full {
                if mask & 1 == 1 {
                    let mut m1 = Mono::ONE;
                    let mut m2 = Mono::ONE;
                    for (i, &v) in vs.iter().enumerate() {
                        if (mask >> i) & 1 == 1 {
                            m1 = m1.with(v);
                        } else {
                            m2 = m2.with(v);
                        }
                    }
                    back_lookup(st, &bb.idx, &m1, params.back_look, &mut la);
                    if !la.is_empty() {
                        back_lookup(st, &bb.idx, &m2, params.back_look, &mut lb);
                        for &a in &la {
                            if a == t {
                                continue;
                            }
                            for &b in &lb {
                                if b == t || b == a {
                                    continue;
                                }
                                let (x, y) = if a < b { (a, b) } else { (b, a) };
                                *bb.votes.entry((t, 2, x, y)).or_insert(0) += wq;
                            }
                        }
                    }
                }
                mask += 1;
            }
        }
    }
    // estimates: 4 * created - 2 * votes (weighted votes already)
    for (&(t, k, a, b), &v) in bb.votes.iter() {
        let size = match k {
            0 => 1i64,
            1 => st.polys[a as usize].len() as i64,
            _ => (st.polys[a as usize].len() * st.polys[b as usize].len()) as i64,
        };
        bb.est.push((4 * size - 2 * v, t, k, a, b));
    }
    bb.est.sort_unstable();
    let n = bb.est.len().min(params.back_k);
    for i in 0..n {
        let (_, t, k, a, b) = bb.est[i];
        let nvar = match k {
            0 => 1u8,
            1 => 2,
            _ => 8,
        };
        let mut best: Option<(i64, BCand)> = None;
        for flags in 0..nvar {
            let c = BCand { t, k, a, b, flags };
            if !back_poly(st, &c, bb, 16384) {
                break;
            }
            let ds = st.back_delta(t, &bb.q);
            if best.is_none_or(|(bd, _)| ds < bd) {
                best = Some((ds, c));
            }
        }
        if let Some((ds, c)) = best {
            bb.scored.push((ds, k, rng.next_u64(), c));
        }
    }
    bb.scored.sort_unstable();
}

// ---------------------------------------------------------------------------
// helper macros: x_t += product(S) for |S| >= 3 using dirty ancillas
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
struct MCand<const W: usize> {
    t: u32,
    s: Mono<W>,
}

fn gen_macros<const W: usize>(st: &State<W>, out: &mut Vec<MCand<W>>) {
    out.clear();
    let mut seen: FxHashSet<MCand<W>> = FxHashSet::default();
    let mut e: Vec<Mono<W>> = Vec::new();
    let mut index: FxHashMap<Mono<W>, Vec<u32>> = FxHashMap::default();
    for w in 0..st.nw {
        if st.dist[w as usize] == 0 {
            continue;
        }
        let xw = Mono::var(w);
        let p = &st.polys[w as usize];
        e.clear();
        e.extend(p.iter().filter(|m| **m != xw && m.degree() >= 3).copied());
        if e.is_empty() {
            continue;
        }
        e.sort_unstable();
        index.clear();
        for m in p.iter() {
            if m.degree() <= 2 {
                for t in m.vars() {
                    index.entry(m.without(t)).or_default().push(t);
                }
            }
        }
        // r = {} (m1 = x_t) and r = {c} (m1 = x_t x_c)
        for m2 in &e {
            if let Some(ts) = index.get(&Mono::ONE) {
                for &t in ts {
                    if !m2.contains(t) {
                        let c = MCand { t, s: *m2 };
                        if seen.insert(c) {
                            out.push(c);
                        }
                    }
                }
            }
            if m2.degree() >= 4 {
                for c in m2.vars() {
                    if let Some(ts) = index.get(&Mono::var(c)) {
                        for &t in ts {
                            if !m2.contains(t) {
                                let mc = MCand {
                                    t,
                                    s: m2.without(c),
                                };
                                if seen.insert(mc) {
                                    out.push(mc);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out.sort_unstable();
}

/// Barenco (Lemma 7.2): `x_t += c_1 ... c_k` with `k >= 3` using `k-2` dirty
/// ancillas `h`, `4(k-2)` positive Toffolis.
pub fn barenco(t: u32, c: &[u32], h: &[u32], out: &mut Vec<SGate>) {
    let k = c.len();
    debug_assert!(k >= 3 && h.len() == k - 2);
    fn ladder(m: usize, c: &[u32], h: &[u32], out: &mut Vec<SGate>) {
        if m == 1 {
            out.push(SGate::tof(h[0], c[0], true, c[1], true, false));
        } else {
            // level m collects c_1..c_{m+1} into h_m (0-based: c[m] with h[m-2])
            out.push(SGate::tof(h[m - 1], c[m], true, h[m - 2], true, false));
            ladder(m - 1, c, h, out);
            out.push(SGate::tof(h[m - 1], c[m], true, h[m - 2], true, false));
        }
    }
    out.push(SGate::tof(t, c[k - 1], true, h[k - 3], true, false));
    ladder(k - 2, c, h, out);
    out.push(SGate::tof(t, c[k - 1], true, h[k - 3], true, false));
    ladder(k - 2, c, h, out);
}

fn barenco_cost(k: usize) -> usize {
    if k <= 2 { 1 } else { 4 * (k - 2) }
}

/// Deterministic ancilla choice: clean wires first (identity polynomial,
/// `x_h` nowhere else), then wires with identity polynomial, then anything.
fn pick_ancillas<const W: usize>(
    st: &State<W>,
    n: usize,
    forbidden: &FxHashSet<u32>,
) -> Option<Vec<u32>> {
    let mut out = Vec::with_capacity(n);
    for pass in 0..3 {
        for h in 0..st.nw {
            if out.len() == n {
                return Some(out);
            }
            if forbidden.contains(&h) || out.contains(&h) {
                continue;
            }
            let ok = match pass {
                0 => st.is_clean(h),
                1 => st.dist[h as usize] == 0,
                _ => true,
            };
            if ok {
                out.push(h);
            }
        }
    }
    if out.len() == n { Some(out) } else { None }
}

/// Plan the best helper macro batch in the current state; returns the gate
/// sequence to emit or `None`.
fn plan_macros<const W: usize>(
    st: &mut State<W>,
    params: &SynthParams,
    rng: &mut Rng,
    mc: &mut Vec<MCand<W>>,
    scratch: &mut Vec<Mono<W>>,
) -> Option<Vec<SGate>> {
    gen_macros(st, mc);
    if mc.is_empty() {
        return None;
    }
    // rank by score delta per gate
    let mut ranked: Vec<(i64, u32, u64, usize)> = Vec::new();
    for (i, c) in mc.iter().enumerate() {
        let (ds, _) = st.delta(c.t, &[c.s], scratch);
        if ds < 0 {
            let k = c.s.degree() as usize;
            let per = ds * 1024 / barenco_cost(k) as i64;
            ranked.push((per, k as u32, rng.next_u64(), i));
        }
    }
    if ranked.is_empty() {
        return None;
    }
    ranked.sort_unstable();
    let first = mc[ranked[0].3];
    // choose the pair (a,b) of first.s shared by the most other negative candidates
    let sv = first.s.to_vec();
    let mut best_pair = (sv[0], sv[1]);
    let mut best_cnt = -1i64;
    for i in 0..sv.len() {
        for j in i + 1..sv.len() {
            let (a, bb) = (sv[i], sv[j]);
            let cnt = ranked
                .iter()
                .skip(1)
                .filter(|r| {
                    let c = &mc[r.3];
                    c.s.contains(a) && c.s.contains(bb) && c.t != a && c.t != bb
                })
                .count() as i64;
            if cnt > best_cnt {
                best_cnt = cnt;
                best_pair = (a, bb);
            }
        }
    }
    let (a, bb) = best_pair;
    // greedy batch under commuting constraints (virtual application)
    let mut batch: Vec<MCand<W>> = vec![first];
    st.apply(first.t, &[first.s]);
    let mut all_t: FxHashSet<u32> = FxHashSet::default();
    let mut all_s = first.s;
    all_t.insert(first.t);
    for r in ranked.iter().skip(1) {
        if batch.len() >= params.macro_batch {
            break;
        }
        let c = mc[r.3];
        if !(c.s.contains(a) && c.s.contains(bb)) || c.t == a || c.t == bb {
            continue;
        }
        if all_s.contains(c.t) {
            continue;
        }
        if all_t.iter().any(|&t| c.s.contains(t)) {
            continue;
        }
        let (ds, _) = st.delta(c.t, &[c.s], scratch);
        if ds < 0 {
            st.apply(c.t, &[c.s]);
            batch.push(c);
            all_t.insert(c.t);
            all_s = all_s.or(&c.s);
        }
    }
    for c in batch.iter().rev() {
        st.apply(c.t, &[c.s]); // undo (involution)
    }
    // cost comparison
    let indiv: usize = batch
        .iter()
        .map(|c| barenco_cost(c.s.degree() as usize))
        .sum();
    let amort: usize = 2 + 2 * batch
        .iter()
        .map(|c| {
            let k = c.s.degree() as usize;
            if k == 3 { 1 } else { barenco_cost(k - 1) }
        })
        .sum::<usize>();
    let mut forbidden: FxHashSet<u32> = all_t.clone();
    for v in all_s.vars() {
        forbidden.insert(v);
    }
    let mut gates: Vec<SGate> = Vec::new();
    if batch.len() == 1 || indiv <= amort {
        for c in &batch {
            let cv = c.s.to_vec();
            let h = pick_ancillas(st, cv.len() - 2, &forbidden)?;
            barenco(c.t, &cv, &h, &mut gates);
        }
    } else {
        let h = pick_ancillas(st, 1, &forbidden)?[0];
        forbidden.insert(h);
        let mut inner: Vec<SGate> = Vec::new();
        for c in &batch {
            let rest: Vec<u32> = c.s.without(a).without(bb).to_vec();
            if rest.len() == 1 {
                inner.push(SGate::tof(c.t, h, true, rest[0], true, false));
            } else {
                let mut ctr = vec![h];
                ctr.extend_from_slice(&rest);
                let anc = pick_ancillas(st, ctr.len() - 2, &forbidden)?;
                barenco(c.t, &ctr, &anc, &mut inner);
            }
        }
        gates.push(SGate::tof(h, a, true, bb, true, false));
        gates.extend_from_slice(&inner);
        gates.push(SGate::tof(h, a, true, bb, true, false));
        gates.extend_from_slice(&inner);
    }
    Some(gates)
}

// ---------------------------------------------------------------------------
// linear finisher: Gauss-Jordan column elimination on the affine sub-system
// ---------------------------------------------------------------------------
/// Rows = wires with an affine residual polynomial that still has errors (A)
/// plus every variable they mention (W). A forward `CNOT(t; a)` toggles
/// column `a` in every row containing `t` (`x_t <- x_t + x_a`); a forward
/// `NOT(t)` toggles the constant of every row containing `t`. The plan
/// brings each A row to `{x_w}` on the tracked sub-matrix; the caller
/// executes it through the exact substitution and keeps it only if the
/// error count drops. Returns (gates, pivot-fix flags).
fn plan_linear<const W: usize>(st: &State<W>, max_ops: usize) -> Option<(Vec<SGate>, Vec<bool>)> {
    let mut a_rows: Vec<u32> = Vec::new();
    for w in 0..st.nw {
        if st.dist[w as usize] == 0 {
            continue;
        }
        let p = &st.polys[w as usize];
        if p.iter().all(|m| m.degree() <= 1) && p.iter().any(|m| m.degree() == 1) {
            a_rows.push(w);
        }
    }
    if a_rows.is_empty() {
        return None;
    }
    let mut wset: FxHashSet<u32> = a_rows.iter().copied().collect();
    for &w in &a_rows {
        for m in &st.polys[w as usize] {
            if let Some(v) = m.first_var() {
                wset.insert(v);
            }
        }
    }
    let mut ws: Vec<u32> = wset.into_iter().collect();
    ws.sort_unstable();
    let k = ws.len();
    // the row bitsets are monomials over POSITIONS: they must fit the width
    if k > Mono::<W>::VARS as usize {
        return None;
    }
    let mut pos: FxHashMap<u32, usize> = FxHashMap::default();
    for (i, &w) in ws.iter().enumerate() {
        pos.insert(w, i);
    }
    // rows[i] = bitset (over positions) of the variables in lin(P_ws[i])
    let mut rows: Vec<Mono<W>> = vec![Mono::ONE; k];
    let mut consts: Vec<bool> = vec![false; k];
    let mut is_a: Vec<bool> = vec![false; k];
    for &w in &a_rows {
        is_a[pos[&w]] = true;
    }
    for (i, &w) in ws.iter().enumerate() {
        for m in &st.polys[w as usize] {
            match m.degree() {
                0 => consts[i] ^= true,
                1 => {
                    if let Some(&j) = pos.get(&m.first_var().unwrap()) {
                        rows[i] = if rows[i].contains(j as u32) {
                            rows[i].without(j as u32)
                        } else {
                            rows[i].with(j as u32)
                        };
                    }
                }
                _ => {}
            }
        }
    }
    let mut gates: Vec<SGate> = Vec::new();
    let mut pivot_flag: Vec<bool> = Vec::new();
    let mut processed = vec![false; k];
    // op(t -> a): rows containing t toggle a   == CNOT(target ws[t]; control ws[a])
    let op = |rows: &mut Vec<Mono<W>>, t: usize, a: usize| {
        for r in rows.iter_mut() {
            if r.contains(t as u32) {
                *r = if r.contains(a as u32) {
                    r.without(a as u32)
                } else {
                    r.with(a as u32)
                };
            }
        }
    };
    for i in 0..k {
        if !is_a[i] {
            continue;
        }
        if gates.len() > max_ops {
            break;
        }
        if !rows[i].contains(i as u32) {
            // pivot fix: pick an unprocessed t in row i whose column is sparsest
            let mut best: Option<(usize, usize)> = None;
            for t in rows[i].vars() {
                let t = t as usize;
                if t == i || processed[t] {
                    continue;
                }
                let colw = rows.iter().filter(|r| r.contains(t as u32)).count();
                if best.is_none_or(|(bw, _)| colw < bw) {
                    best = Some((colw, t));
                }
            }
            match best {
                Some((_, t)) => {
                    op(&mut rows, t, i);
                    gates.push(SGate::cnot(ws[t], ws[i], true, false));
                    pivot_flag.push(true);
                }
                None => continue, // singular on the tracked sub-system: leave the row
            }
        }
        let others: Vec<u32> = rows[i].vars().filter(|&a| a as usize != i).collect();
        for a in others {
            op(&mut rows, i, a as usize);
            gates.push(SGate::cnot(ws[i], ws[a as usize], true, false));
            pivot_flag.push(false);
        }
        processed[i] = true;
    }
    // constants: NOT(w) toggles the constant of every row containing x_w
    for i in 0..k {
        if is_a[i] && consts[i] && rows[i] == Mono::var(i as u32) {
            for r in 0..k {
                if rows[r].contains(i as u32) {
                    consts[r] ^= true;
                }
            }
            gates.push(SGate::not(ws[i]));
            pivot_flag.push(false);
        }
    }
    if gates.is_empty() {
        None
    } else {
        Some((gates, pivot_flag))
    }
}

// ---------------------------------------------------------------------------
// one synthesis run
// ---------------------------------------------------------------------------
struct RunOut {
    gates: Vec<SGate>,
    tags: Vec<MoveTag>,
    status: SynthStatus,
    steps: usize,
    note: String,
}

fn apply_cand<const W: usize>(st: &mut State<W>, c: &Cand) {
    let mut lbuf = [Mono::<W>::ONE; 4];
    let n = c.l(&mut lbuf);
    st.apply(c.t, &lbuf[..n]);
}

fn apply_gate<const W: usize>(st: &mut State<W>, g: &SGate) {
    let mut lbuf = [Mono::<W>::ONE; 4];
    let n = gate_l(g, &mut lbuf);
    if n > 0 {
        st.apply(g.target(), &lbuf[..n]);
    }
}

#[derive(Clone, Copy)]
enum Mv {
    F(Cand),
    B(BCand),
}

fn push_tabu<T: Copy>(tabu: &mut Vec<T>, c: T) {
    tabu.push(c);
    while tabu.len() > 8 {
        tabu.remove(0);
    }
}

fn run_once<const W: usize>(
    spec: &WindowSpec<W>,
    params: &SynthParams,
    rng: &mut Rng,
    deadline: Instant,
) -> RunOut {
    let monos: usize = spec.polys.iter().map(|(_, p)| p.len()).sum();
    let step_cap = params.step_factor * monos + 64;
    let mut st = State::new(spec, params.deg_weight, params.w0, params.w1);
    let mut gates: Vec<SGate> = Vec::new();
    let mut tags: Vec<MoveTag> = Vec::new();
    // backward gates in peel order (the first peeled gate executes last)
    let mut suffix: Vec<SGate> = Vec::new();
    let mut suffix_tags: Vec<MoveTag> = Vec::new();
    let mut bb = BackBuffers::new();
    let mut bb2 = BackBuffers::new();
    let mut tabu_b: Vec<BCand> = Vec::new();
    let mut b = Buffers::new();
    let mut b2 = Buffers::new();
    let mut mc: Vec<MCand<W>> = Vec::new();
    let mut best_errs = st.errs;
    let mut stall = 0usize;
    let mut tabu: Vec<Cand> = Vec::new();
    let mut steps = 0usize;
    let mut note = String::new();
    let mut macro_fail = 0usize;
    let mut linear_rejected_at: i64 = -1;
    let finish = |mut gates: Vec<SGate>,
                  mut tags: Vec<MoveTag>,
                  suffix: &Vec<SGate>,
                  suffix_tags: &Vec<MoveTag>,
                  status: SynthStatus,
                  steps: usize,
                  note: String|
     -> RunOut {
        for (g, t) in suffix.iter().rev().zip(suffix_tags.iter().rev()) {
            gates.push(*g);
            tags.push(*t);
        }
        RunOut {
            gates,
            tags,
            status,
            steps,
            note,
        }
    };
    loop {
        if st.errs == 0 {
            return finish(
                gates,
                tags,
                &suffix,
                &suffix_tags,
                SynthStatus::Converged,
                steps,
                note,
            );
        }
        if gates.len() + suffix.len() > step_cap {
            return finish(
                gates,
                tags,
                &suffix,
                &suffix_tags,
                SynthStatus::StepCap,
                steps,
                note,
            );
        }
        if steps.is_multiple_of(8) && Instant::now() >= deadline {
            return finish(
                gates,
                tags,
                &suffix,
                &suffix_tags,
                SynthStatus::Timeout,
                steps,
                note,
            );
        }
        steps += 1;
        if st.errs < best_errs {
            best_errs = st.errs;
            stall = 0;
        }
        gen_cands(&st, &mut b);
        score_cands(&st, &mut b, rng, params.exact_k, params.noise);
        // backward (output-side) candidates
        let mut back_best: Option<(i64, BCand)> = None;
        if params.back_k > 0 {
            gen_back(&st, &mut bb, params, rng);
            if let Some(&(ds, _, _, c)) = bb.scored.first() {
                back_best = Some((ds, c));
            }
        }
        if let Some((dsb, cb)) = back_best {
            let fwd = b.scored.first().map(|s| s.0).unwrap_or(i64::MAX);
            if dsb < 0 && (dsb < fwd || (dsb == fwd && params.back_prefer)) {
                let ok = back_poly(&st, &cb, &mut bb, usize::MAX);
                debug_assert!(ok);
                let q: Vec<Mono<W>> = bb.q.clone();
                st.apply_back(cb.t, &q);
                suffix.push(cb.gate());
                suffix_tags.push(MoveTag::Back);
                continue;
            }
        }
        if let Some(&(ds, _, _, i)) = b.scored.first()
            && ds < 0
        {
            let c = b.cands[i as usize];
            let tag = if st.is_sink(c.t) {
                MoveTag::Sink
            } else if c.k <= 1 && st.target_affine(c.t) {
                MoveTag::Linear
            } else {
                MoveTag::Greedy
            };
            apply_cand(&mut st, &c);
            gates.push(c.gate());
            tags.push(tag);
            push_tabu(&mut tabu, c);
            continue;
        }
        // 2-step lookahead over the top candidates (forward and backward)
        if params.beam > 0 && (!b.scored.is_empty() || !bb.scored.is_empty()) {
            let mut firsts: Vec<(i64, Mv)> = b
                .scored
                .iter()
                .take(params.beam)
                .map(|s| (s.0, Mv::F(b.cands[s.3 as usize])))
                .collect();
            if params.back_k > 0 {
                firsts.extend(
                    bb.scored
                        .iter()
                        .take(params.beam)
                        .map(|s| (s.0, Mv::B(s.3))),
                );
            }
            let mut best: Option<(i64, i64, Mv, Mv)> = None;
            let mut q1: Vec<Mono<W>> = Vec::new();
            for &(ds1, m1) in &firsts {
                match m1 {
                    Mv::F(c1) => apply_cand(&mut st, &c1),
                    Mv::B(c1) => {
                        if !back_poly(&st, &c1, &mut bb, usize::MAX) {
                            continue;
                        }
                        q1.clear();
                        q1.extend_from_slice(&bb.q);
                        st.apply_back(c1.t, &q1);
                    }
                }
                gen_cands(&st, &mut b2);
                score_cands(&st, &mut b2, rng, params.exact_k, params.noise);
                let mut second: Option<(i64, Mv)> = b2
                    .scored
                    .first()
                    .map(|s| (s.0, Mv::F(b2.cands[s.3 as usize])));
                if params.back_k > 0 {
                    gen_back(&st, &mut bb2, params, rng);
                    if let Some(&(ds2, _, _, c2)) = bb2.scored.first()
                        && second.is_none_or(|(d, _)| ds2 < d)
                    {
                        second = Some((ds2, Mv::B(c2)));
                    }
                }
                if let Some((ds2, m2)) = second {
                    let total = ds1 + ds2;
                    let better = match &best {
                        None => true,
                        Some((bt, bds2, _, _)) => total < *bt || (total == *bt && ds2 < *bds2),
                    };
                    if better {
                        best = Some((total, ds2, m1, m2));
                    }
                }
                match m1 {
                    Mv::F(c1) => apply_cand(&mut st, &c1), // undo
                    Mv::B(c1) => st.apply_back(c1.t, &q1),
                }
            }
            if let Some((total, _, m1, m2)) = best
                && total < 0
            {
                for m in [m1, m2] {
                    match m {
                        Mv::F(c) => {
                            apply_cand(&mut st, &c);
                            gates.push(c.gate());
                            tags.push(MoveTag::Lookahead);
                            push_tabu(&mut tabu, c);
                        }
                        Mv::B(c) => {
                            let ok = back_poly(&st, &c, &mut bb, usize::MAX);
                            debug_assert!(ok);
                            let q: Vec<Mono<W>> = bb.q.clone();
                            st.apply_back(c.t, &q);
                            suffix.push(c.gate());
                            suffix_tags.push(MoveTag::Back);
                            push_tabu(&mut tabu_b, c);
                        }
                    }
                }
                continue;
            }
        }
        // linear finisher (cycles / permutation structure of the affine part)
        if linear_rejected_at != st.errs {
            if let Some((lg, pf)) = plan_linear(&st, 4 * (monos + 16)) {
                let before = st.errs;
                for g in &lg {
                    apply_gate(&mut st, g);
                }
                if st.errs < before {
                    for (g, is_pivot) in lg.iter().zip(pf.iter()) {
                        gates.push(*g);
                        tags.push(if *is_pivot {
                            MoveTag::Cycle
                        } else {
                            MoveTag::Linear
                        });
                    }
                    continue;
                }
                for g in lg.iter().rev() {
                    apply_gate(&mut st, g);
                }
                linear_rejected_at = before;
            } else {
                linear_rejected_at = st.errs;
            }
        }
        // helper macros for degree >= 3 monomials
        if params.use_macros && macro_fail < 4 {
            let before = st.score;
            if let Some(mg) = plan_macros(&mut st, params, rng, &mut mc, &mut b.scratch) {
                for g in &mg {
                    apply_gate(&mut st, g);
                    gates.push(*g);
                    tags.push(MoveTag::Helper);
                }
                if st.score >= before {
                    // the batch was planned with negative deltas, so this only
                    // happens through interaction inside the batch; back off
                    // for a few steps instead of disabling macros for good
                    macro_fail += 1;
                    if note.is_empty() {
                        note = format!("macro did not reduce score ({} -> {})", before, st.score);
                    }
                } else {
                    macro_fail = 0;
                }
                continue;
            }
        } else if macro_fail >= 4 {
            macro_fail = 0; // retry macros next time we are stuck
        }
        // perturbation: least-bad non-tabu move (forward or backward)
        let mut picked: Option<(i64, Cand)> = None;
        for s in &b.scored {
            let c = b.cands[s.3 as usize];
            if !tabu.contains(&c) {
                picked = Some((s.0, c));
                break;
            }
        }
        let mut picked_b: Option<(i64, BCand)> = None;
        if params.back_k > 0 {
            for s in &bb.scored {
                if !tabu_b.contains(&s.3) {
                    picked_b = Some((s.0, s.3));
                    break;
                }
            }
        }
        let use_b = match (picked, picked_b) {
            (Some((df, _)), Some((db, _))) => db < df || (db == df && params.back_prefer),
            (None, Some(_)) => true,
            _ => false,
        };
        if use_b {
            let (_, c) = picked_b.unwrap();
            let ok = back_poly(&st, &c, &mut bb, usize::MAX);
            debug_assert!(ok);
            let q: Vec<Mono<W>> = bb.q.clone();
            st.apply_back(c.t, &q);
            suffix.push(c.gate());
            suffix_tags.push(MoveTag::Other);
            push_tabu(&mut tabu_b, c);
            stall += 1;
            if stall > params.stall_limit {
                return finish(
                    gates,
                    tags,
                    &suffix,
                    &suffix_tags,
                    SynthStatus::Stalled,
                    steps,
                    note,
                );
            }
            continue;
        }
        match picked {
            Some((_, c)) => {
                apply_cand(&mut st, &c);
                gates.push(c.gate());
                tags.push(MoveTag::Other);
                push_tabu(&mut tabu, c);
                stall += 1;
                if stall > params.stall_limit {
                    return finish(
                        gates,
                        tags,
                        &suffix,
                        &suffix_tags,
                        SynthStatus::Stalled,
                        steps,
                        note,
                    );
                }
            }
            None => {
                note = format!("no candidates at errs={}", st.errs);
                return finish(
                    gates,
                    tags,
                    &suffix,
                    &suffix_tags,
                    SynthStatus::Stalled,
                    steps,
                    note,
                );
            }
        }
    }
}

/// Wires ascending, monomials sorted, even multiplicities cancelled,
/// identity polynomials dropped.
pub fn canonical<const W: usize>(spec: &WindowSpec<W>) -> WindowSpec<W> {
    let mut polys: Vec<(u32, Vec<Mono<W>>)> = Vec::new();
    for (w, p) in &spec.polys {
        let mut out = p.clone();
        cancel_sorted(&mut out);
        if !(out.len() == 1 && out[0] == Mono::var(*w)) {
            polys.push((*w, out));
        }
    }
    polys.sort_by_key(|(w, _)| *w);
    WindowSpec {
        num_wires: spec.num_wires,
        polys,
    }
}

fn mix(seed: u64, idx: u64, r: u64) -> u64 {
    let mut z =
        seed ^ idx.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ r.wrapping_mul(0xD1B5_4A32_D192_ED03);
    z = (z ^ (z >> 32)).wrapping_mul(0xD6E8_FEB8_6659_FD93);
    z ^ (z >> 32)
}

/// Synthesize a window from its ANF only. `seed` and `idx` (the window
/// index) only seed the tie-break PRNG.
pub fn synthesize<const W: usize>(
    spec: &WindowSpec<W>,
    params: &SynthParams,
    seed: u64,
    idx: usize,
) -> SynthOutput {
    let spec = canonical(spec);
    let t0 = Instant::now();
    let deadline = t0 + Duration::from_secs_f64(params.time_limit.max(0.01));
    if spec.polys.is_empty() {
        return SynthOutput {
            gates: vec![],
            tags: vec![],
            status: SynthStatus::Converged,
            steps: 0,
            restarts_used: 0,
            deadline_hit: false,
            note: String::new(),
        };
    }
    let mut best: Option<RunOut> = None;
    let mut last: Option<RunOut> = None;
    let mut used = 0usize;
    let mut lens: Vec<usize> = Vec::new();
    let mut deadline_hit = false;
    for r in 0..params.restarts.max(1) {
        if r > 0 && Instant::now() >= deadline {
            deadline_hit = true;
            break;
        }
        let mut rng = Rng::new(mix(seed, idx as u64, r as u64));
        let mut pr = params.clone();
        if params.noise_schedule {
            // restart 0 is the plain search; later restarts add growing score noise
            pr.noise = params.noise + 3 * r as i64;
        }
        // degree-weight schedule: restarts cycle through dw, dw+2, 0 (plain D)
        // and dw-2 so that the best of several scoring regimes is kept
        pr.deg_weight = match r % 4 {
            0 => params.deg_weight,
            1 => params.deg_weight + 2,
            2 => 0,
            _ => (params.deg_weight - 2).max(0),
        };
        if params.back_alt {
            pr.back_prefer = r.is_multiple_of(2);
        }
        let out = run_once(&spec, &pr, &mut rng, deadline);
        used += 1;
        if out.status == SynthStatus::Timeout {
            deadline_hit = true;
        }
        if out.status == SynthStatus::Converged {
            lens.push(out.gates.len());
            let better = match &best {
                None => true,
                Some(b) => out.gates.len() < b.gates.len(),
            };
            if better {
                best = Some(out);
                continue;
            }
        }
        last = Some(out);
    }
    let out = match best {
        Some(b) => b,
        None => last.expect("at least one run"),
    };
    let mut note = out.note;
    if note.is_empty() && !lens.is_empty() {
        lens.sort_unstable();
        note = format!(
            "rs={}/{}/{}/{}",
            lens.len(),
            lens[0],
            lens[lens.len() / 2],
            lens[lens.len() - 1]
        );
    }
    SynthOutput {
        gates: out.gates,
        tags: out.tags,
        status: out.status,
        steps: out.steps,
        restarts_used: used,
        deadline_hit,
        note,
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/unpacking/synth.rs"]
mod tests;
