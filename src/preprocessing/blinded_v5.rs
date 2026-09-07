//! Blinded-V5 computation stage: an alternative to the drip `route_fire`
//! compute. Takes a circuit `A` on `n` wires and builds an equivalent circuit
//! on `2n` wires (data `0..n`, band `n..2n`) whose middle is a shuffled cloud
//! of locally-geodesic-identity (LGI) masks with `A`'s gates threaded through
//! it via a MASKED read. Only the compute changes; the surrounding pipeline
//! (slice guards, band fill, band rerand, final slice) is unchanged.
//!
//! The masking atom is the g57 gate `g57(w,x,y) = w ^= 1 ^ (!x & y)` (data
//! target, band controls), i.e. the mask term `1 ^ y ^ x&y`. A DISJOINT-PAIR
//! LGI on `w`, `g57(w,cy[2i],cy[2i+1])`, is a deg-2 mask. K must be >= 2 (a
//! 1-cycle is a degenerate constant flip).
//!
//! READ MODE (RC, 2026-09-06 -- QUAD-FIRE is the default). The original read
//! LINEARISED the operand: it completed each open g57 with its reverse
//! (`g57(w,r1,r2) ^ g57(w,r2,r1) = w ^ r1 ^ r2`) so the wire carried
//! `operand ^ rho`, rho a plain XOR of band wires, fired over that, and undid
//! the reverses. That leaves the wire EXACTLY affine in band wires for the
//! window between the reverses and the undo; the undo is a write pinned only by
//! the wire's next read, so every reordering stage of the pipeline (DB
//! re-encoding, fmix's final float, the crossing walk) stretches the window to
//! the operand's idle interval -- measured as the C-vs-G affine ridge (~6k exact
//! relations, rho 1.00 through the whole pipeline). QUAD-FIRE never linearises:
//! the operand is the ANF polynomial `w ^ sum_(x,y) (1 ^ y ^ x&y)` over its open
//! pairs (topped up to `max_open` fresh quadratic pairs, undone after the fire)
//! and the fire is the polynomial product (degree <= 4 at K=2). No read ever
//! leaves a wire affine, so there is no window to stretch: relations stay at
//! the public I/O fringe (~150-500) through phase A, split, crossing and
//! compression, and the gadget is ~12% smaller (no linearise/undo gates).
//! `quad_fire = false` (env `BV5_QUAD_FIRE=0`) keeps the legacy linear read.
//!
//! CO-SAMPLED build (RC, 2026-09-04): the LGI masks, the rerand gates, and the
//! A-gate placements are produced TOGETHER in one forward pass, so every A-gate
//! is straddled by a real (rerand-protected) LGI. Per active wire `w`, `u_w+1`
//! LGIs (u_w = uses of w) with <= `max_open` open at once -- SAME masking budget
//! as before, so the same statistics, at no cost. Of `w`'s opens, `w_w` are
//! STRADDLE opens generated ON DEMAND when an A-gate on `w` is placed, and the
//! rest are FILLER opens (masking during reads):
//!   * MASKED READ (quad-fire): for each control, keep its net-open deg-2 masks
//!     as they are, top up to `max_open` quadratic terms with fresh single g57s,
//!     and realise `c ^= comp ^ lit(a)&lit(b)` as the product of the two operand
//!     polynomials over ONLY the masked control wires and band wires (never
//!     bare `a`,`b`,`a^b`; 0/1/2 controls, all polarities); then undo the
//!     temporary top-up pairs. (Legacy linear read: linearise, fire over
//!     `(a'^rho_a)(b'^rho_b)`, de-linearise -- see the module header.)
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
    /// STRADDLE rerand SLOTS. Each slot is a BURST of `rerand_burst` band-refresh
    /// gates on ONE band wire; a straddle slot CLOSES the masks reading it once,
    /// so the SLOT count is what thins the masking (knee ~1024). `0` = auto =
    /// `m/(4k)` (~875 for the n=128 sandwich, just under the knee).
    pub rerand_level: usize,
    /// REPAIR rerand SLOTS (bursts). A repair slot re-derives every mask reading
    /// the band wire once around the whole burst (old-b cancels, new-b re-adds),
    /// so masks stay open -- no thinning. `0` = off.
    pub rerand_repair: usize,
    /// Rerand BURST size F -- band-refresh gates per slot, all on the same band
    /// wire with independently-random data-or-band controls (so the band shows a
    /// data-wire-like activity burst). `0` = auto = `8k`; slots*F ~ 2m gates.
    pub rerand_burst: usize,
    /// Soft cap on simultaneously-open LGIs per wire (RC: `<= 3`).
    pub max_open: usize,
    /// HARD floor on `|rho|` -- the number of band wires masking each operand at
    /// its read. The read tops up fresh disjoint g57 pairs until `|rho| >=
    /// min_mask`, so even a low-probability draw where the open masks cancel down
    /// can never expose an operand under fewer than `min_mask` masking wires. `0`
    /// = auto = `max_open` (mean |rho| ~6 at max_open=3, so the floor rarely adds
    /// gates). Must stay `>= 1` (a 0 floor would permit a bare read).
    pub min_mask: usize,
    /// Seed the band ONLY from data wires `0..active_wires`. `0` = all `np`.
    pub active_wires: usize,
    /// Extra LGIs per wire beyond `u_w+1`. Adds straddle SLOTS for the
    /// hidden-firing fix (each A-gate is placed straddling a scaffold LGI on its
    /// active wire); more slots => fewer gates left with their firing exposed.
    pub extra_lgis: usize,
    /// QUAD-FIRE (default ON since 2026-09-06): fire from INSIDE the quadratic
    /// masks. Reads do not linearise the operand's open g57 masks (no reverse
    /// gates, no undo); the operand is the ANF polynomial `w ⊕ Σ_(x,y) (1 ⊕ y ⊕
    /// xy)` over its net-open pairs (topped up to `max_open` fresh quadratic
    /// pairs, undone after the fire) and the fire is the polynomial product
    /// (monomials of degree ≤ 4). A linearised read leaves the wire EXACTLY
    /// affine in band wires between the reverses and the undo, and every
    /// reordering stage stretches that window to the operand's idle interval
    /// -- the C-vs-G affine ridge; with no linearisation there is no window.
    /// `false` = the legacy linear read (env `BV5_QUAD_FIRE=0`).
    pub quad_fire: bool,
    /// BALANCED masks (default ON since 2026-09-07; env `BV5_BALANCED=0` for
    /// the plain g57 masks): every LGI (and every read-time top-up) carries one
    /// extra fresh band wire `z` applied as a CNOT `w ^= z`, so each mask term
    /// is `z ⊕ 1 ⊕ ¬x∧y` -- balanced (zero linear correlation with the
    /// plaintext when `z` is uniform) while still quadratic (no exact affine
    /// recovery). Motivation: a bare g57 term is 1 with probability 3/4, so a
    /// wire under one or two of them keeps a linear correlation of 1/2..1/4
    /// with its plaintext, which a statistical adversary (fire_corr, the
    /// gauntlet's w1) reads; the exact-GF(2) adversary never sees it. Cost is
    /// the read polynomial (11x11 instead of 8x8 monomials per fire at
    /// max_open 3: +92% gates). The band seed becomes `x_i ⊕ x_j` (balanced)
    /// instead of `x_i ∧ ¬x_j`.
    pub balanced: bool,
    /// Rerand bursts read band wires only (no data-wire controls). With
    /// balanced masks a burst `b ^= lit(a')∧lit(b')` over masked DATA wires
    /// puts a product correlated with a C gate's fire (¬a∧b) into the band,
    /// and the balancing CNOT copies that band wire linearly onto a data wire
    /// (env `BV5_BURST_BANDONLY=1`).
    pub burst_band_only: bool,
    /// ENCODED I/O (test harnesses, e.g. the gadget gauntlet): the data wires
    /// enter and leave the circuit MASKED. `max_open` LGIs per wire are opened
    /// before the first gate and the final drain is skipped; the gates that
    /// would open them are returned as `pre_gates` (apply to the plaintext
    /// state to encode) and the closing gates as `post_gates` (apply to the
    /// final state to decode) -- neither is part of `gates`, so an observer of
    /// the circuit's trace never sees a plain input or output. Production
    /// leaves this off: the sandwich's I/O is public.
    pub encoded_io: bool,
    /// MINIMUM open masks per data wire at every instant between the wire's
    /// first and last mask (default 2; 1 = the coverage rules alone). Enforced
    /// at every place a wire's open count can drop or start low: a rerand
    /// burst opens enough replacements before its closes, a filler or straddle
    /// open on a thin wire is followed by further opens, and a read on a thin
    /// operand keeps that many of its top-ups as real masks. Motivation: with
    /// one open mask a wire is one uniform term away from its plaintext, which a
    /// single visible monomial can cancel (gauntlet w2); measured 26% of covered
    /// wire-time at one mask before this rule (3% on the payload half). Must be
    /// below `max_open` (the rolling cap closes one when the cap is reached).
    pub min_open: usize,
}

impl BlindedV5Params {
    /// Settled preset: K=2 (band wires per LGI -> 1 disjoint pair; affine- and
    /// deg-2-neutral across K, so smallest is best), R=n (auto), max_open=3.
    /// Rerand: auto STRADDLE bursts -- `m/(4k)` slots (~875, under the ~1024 knee)
    /// x `8k` gates each ~= 2m band-refresh gates, no repair (RC, 2026-09-04);
    /// bursts concentrate the band mixing in few slots (less LGI interference) and
    /// give the band a data-wire-like activity signature.
    pub fn production(seed: u64) -> Self {
        Self {
            k: 2,
            r: 0,
            seed,
            rerand_level: 0,
            rerand_repair: 0,
            rerand_burst: 0,
            max_open: 3,
            min_mask: 0,
            active_wires: 0,
            extra_lgis: 0,
            quad_fire: true,
            balanced: true,
            burst_band_only: false,
            encoded_io: false,
            min_open: 2,
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
    /// `encoded_io`: the off-circuit LGI opens that encode the input (empty
    /// otherwise). Apply to the plaintext state BEFORE `gates`.
    pub pre_gates: Vec<XGate>,
    /// `encoded_io`: the off-circuit LGI closes that decode the output (empty
    /// otherwise). Apply to the final state AFTER `gates`.
    pub post_gates: Vec<XGate>,
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

/// The applied gate of an LGI on wire `w` (cycle `cy`) that READS band `b`:
/// the disjoint-pair g57 containing `b`, or the balancing CNOT `w ^= b` when
/// `b` is the odd trailing (balancing) wire. `None` if the LGI does not read `b`.
fn b_g57(w: u16, cy: &[u16], b: u16) -> Option<XGate> {
    for i in 0..cy.len() / 2 {
        if cy[2 * i] == b || cy[2 * i + 1] == b {
            return Some(g57(w, cy[2 * i], cy[2 * i + 1]));
        }
    }
    // balanced LGI: the odd trailing wire is the CNOT term `w ^= z`
    if cy.len() % 2 == 1 && cy[cy.len() - 1] == b {
        return Some(XGate::cnot(w, b));
    }
    None
}

/// One band-refresh gate `b ^= lit(c1) & lit(c2)` for a rerand BURST: each
/// control is independently a LIVE data wire (`0..active`) or a band wire, so a
/// burst spans all of data-data / band-band / data-band -- making the band wire
/// look like a data wire (a burst of activity with mixed controls). `active` =
/// the low honest half in the sandwich.
fn burst_gate(b: u16, active: usize, band: &[u16], rng: &mut StdRng) -> XGate {
    let pick = |rng: &mut StdRng| -> u16 {
        if active > 0 && rng.random_bool(0.5) {
            rng.random_range(0..active) as u16
        } else {
            band[rng.random_range(0..band.len())]
        }
    };
    let mut c1 = pick(rng);
    while c1 == b {
        c1 = pick(rng);
    }
    let mut c2 = pick(rng);
    while c2 == b || c2 == c1 {
        c2 = pick(rng);
    }
    conj(b, &[(c1, rng.random_bool(0.5)), (c2, rng.random_bool(0.5))])
}

/// In-place Fisher-Yates shuffle (the masked-fire monomials commute, so each
/// half of a fire batch can be emitted in an independently random order).
fn shuffle_slice(s: &mut [XGate], rng: &mut StdRng) {
    for i in (1..s.len()).rev() {
        let j = rng.random_range(0..=i);
        s.swap(i, j);
    }
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

/// LEGACY linear read (`quad_fire = false`). Linearise wire `w` masked by the
/// net-open g57 ordered pairs `netopen`: complete every net-open g57 into its
/// reverse pair so `w = w_true ^ rho`. Returns (rho band wires, the reverse
/// gates emitted -- undo after the read).
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

/// LEGACY linear read: emit `c ^= comp ^ prod(lit(w_i,p_i))` for <=2 (already
/// linearised) controls `(wire, pol, rho)`, using only the masked control wires
/// and band wires.
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

/// ANF polynomial over wires: a set of monomials (sorted distinct wire lists;
/// the empty monomial is the constant 1), XOR-toggled.
#[derive(Clone, Default)]
struct Poly(BTreeSet<Vec<u16>>);

impl Poly {
    fn toggle(&mut self, mut m: Vec<u16>) {
        m.sort_unstable();
        m.dedup();
        if !self.0.remove(&m) {
            self.0.insert(m);
        }
    }
    /// XOR in the g57(w,x,y) mask term `1 ⊕ y ⊕ x·y` (w ^= 1 ^ (!x & y)).
    fn add_g57_mask(&mut self, x: u16, y: u16) {
        self.toggle(Vec::new());
        self.toggle(vec![y]);
        self.toggle(vec![x, y]);
    }
    fn deg2_terms(&self) -> usize {
        self.0.iter().filter(|m| m.len() == 2).count()
    }
    fn mul(&self, other: &Poly) -> Poly {
        let mut out = Poly::default();
        for a in &self.0 {
            for b in &other.0 {
                let mut m = a.clone();
                m.extend_from_slice(b);
                out.toggle(m);
            }
        }
        out
    }
}

/// QUAD-FIRE emission: `c ^= comp ⊕ Π polys` with each control already a
/// polynomial (polarity applied), as one conjunction gate per monomial of the
/// product (an empty monomial is an X gate). Degree ≤ 4 at K=2.
fn quad_fire(c: u16, polys: &[Poly], comp: bool, out: &mut Vec<XGate>) {
    let mut prod = Poly::default();
    prod.toggle(Vec::new()); // the constant 1
    for p in polys {
        prod = prod.mul(p);
    }
    if comp {
        prod.toggle(Vec::new());
    }
    for m in &prod.0 {
        if m.is_empty() {
            out.push(XGate::x_gate(c));
        } else {
            let lits: Vec<(u16, bool)> = m.iter().map(|&w| (w, true)).collect();
            out.push(conj(c, &lits));
        }
    }
}

/// The band-seeding module (module 2 of the 5-step pipeline): each band wire
/// `np..np+r` set to `x_i & !x_j` from the honest active input wires. Emitted
/// SEPARATELY and pipelined in front of the compute -- the compute only READS
/// the band. `active_wires` = 0 means all `np` data wires.
pub fn seed_band(np: usize, r: usize, active_wires: usize, seed: u64) -> Vec<XGate> {
    seed_band_mode(np, r, active_wires, seed, false)
}

/// `balanced`: seed each band wire as `x_i ⊕ x_j` (uniform on a uniform input)
/// instead of `x_i ∧ ¬x_j` (biased to 0 three times in four).
pub fn seed_band_mode(np: usize, r: usize, active_wires: usize, seed: u64, balanced: bool) -> Vec<XGate> {
    let active = if active_wires == 0 || active_wires > np {
        np
    } else {
        active_wires
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(if balanced { 2 * r } else { r });
    for aw in (np as u16)..((np + r) as u16) {
        let i1 = rng.random_range(0..active) as u16;
        let mut i2 = rng.random_range(0..active) as u16;
        while i2 == i1 {
            i2 = rng.random_range(0..active) as u16;
        }
        if balanced {
            out.push(XGate::cnot(aw, i1));
            out.push(XGate::cnot(aw, i2));
        } else {
            out.push(XGate::conj(aw, [(i1, true), (i2, false)]).unwrap());
        }
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
/// A balanced LGI carries an odd trailing wire `z` = the CNOT term `w ^= z`,
/// toggled in `lin` (the wire's net-open linear band terms).
fn emit_lgi(
    w: u16,
    cy: &[u16],
    out: &mut Vec<XGate>,
    pairs: &mut BTreeSet<(u16, u16)>,
    lin: &mut BTreeSet<u16>,
) {
    out.extend(cycle_g57(w, cy));
    for i in 0..cy.len() / 2 {
        let pr = (cy[2 * i], cy[2 * i + 1]);
        if !pairs.insert(pr) {
            pairs.remove(&pr);
        }
    }
    if cy.len() % 2 == 1 {
        let z = cy[cy.len() - 1];
        out.push(XGate::cnot(w, z));
        if !lin.insert(z) {
            lin.remove(&z);
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
    // Band wires per LGI cycle: `K` rounded down to disjoint pairs (odd K wastes
    // a wire: K3 == K2), plus the balancing wire `z` when balanced.
    let lgi_k = (k - k % 2) + usize::from(p.balanced);
    // A rerand burst on band wire `b` must leave a full cycle of OTHER band wires
    // for the cover replacement, and a read top-up needs two (three when
    // balanced) distinct wires: `r > lgi_k` is the true precondition (with
    // `r == lgi_k` the replacement draw could never avoid `b`).
    assert!(
        r > lgi_k,
        "R must exceed K (+1 with balanced masks): got R={r}, cycle width {lgi_k}"
    );
    assert!(np >= 2, "need at least two data wires");
    let total = np + r;
    assert!(total < u16::MAX as usize, "too many wires");
    let band: Vec<u16> = (np as u16..total as u16).collect();
    let max_open = p.max_open.max(1);
    let min_open = p.min_open.clamp(1, max_open.saturating_sub(1).max(1));
    assert!(
        max_open == 1 || p.min_open < max_open,
        "min_open ({}) must be below max_open ({max_open})",
        p.min_open
    );
    // Hard floor on masking wires per read (never below 1 = never bare). `|ρ|`
    // toggles in steps of 2 (disjoint pairs) so it is structurally EVEN, and the
    // band supplies at most `r` distinct wires: clamp the floor to the largest
    // even value `≤ r` so it is always reachable (`r ≥ k ≥ 2` ⇒ cap ≥ 2). At
    // production `r = n` (=128) this is a no-op.
    let min_mask = (if p.min_mask == 0 { max_open } else { p.min_mask })
        .max(1)
        .min(r - (r % 2));
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

    // rerand SLOTS (bursts): `straddle_slots` STRADDLE (auto = m/(4k), within the
    // ~1024 thinning knee) + `rerand_repair` REPAIR (default off). Each slot is a
    // BURST of `burst` (=F=8k) band-refresh gates on ONE band wire, so total gates
    // ~ slots*F ~ 2m and the band shows a data-wire-like activity burst; straddle
    // closes the masks reading b once per slot, repair re-derives them once around
    // the whole burst. Shuffled, spread through the build.
    let burst = if p.rerand_burst > 0 { p.rerand_burst } else { 8 * k };
    let straddle_slots = if p.rerand_level > 0 {
        p.rerand_level
    } else {
        m / (4 * k)
    };
    let repair_slots = p.rerand_repair;
    let mut slot_plan: Vec<bool> = Vec::new(); // is_repair per slot
    slot_plan.extend(std::iter::repeat(false).take(straddle_slots));
    slot_plan.extend(std::iter::repeat(true).take(repair_slots));
    for i in (1..slot_plan.len()).rev() {
        let j = rng.random_range(0..=i);
        slot_plan.swap(i, j);
    }

    let mut open_cy: Vec<Vec<Vec<u16>>> = vec![Vec::new(); np];
    let mut open_pairs: Vec<BTreeSet<(u16, u16)>> = vec![BTreeSet::new(); np];
    let mut open_lin: Vec<BTreeSet<u16>> = vec![BTreeSet::new(); np]; // balanced CNOT terms
    let mut out: Vec<XGate> = Vec::with_capacity(30 * (3 * m + np));
    let mut rerand_done = 0usize;
    let diag = std::env::var("BV5_DIAG").is_ok();
    let (mut n_reads, mut n_bare, mut rho_sum) = (0usize, 0usize, 0usize);
    let mut rho_min = usize::MAX;
    let mut in_drain = false; // true once all A-gates are placed (post-compute)
    let mut slots_emitted = 0usize;
    let mut drain_slots = 0usize; // rerand slots emitted AFTER the last A-gate

    // Calibrate the rate AHEAD: one slot every `rgap` primitive steps (A-gate
    // placements + filler opens), so the last slot lands DURING the pass -- no
    // end-flush (emitting rerands after the last A-gate is pointless).
    let total_fillers = filler_left.iter().sum::<usize>();
    let total_steps = m + total_fillers;
    let mut si = 0usize;
    let mut steps = 0usize;
    let rgap = if slot_plan.is_empty() {
        usize::MAX
    } else {
        (total_steps / slot_plan.len()).max(1)
    };
    // Filler opens are placed on an even running schedule (below) so that ALL of
    // them — and therefore all rerand slots — land during the compute, leaving no
    // post-compute drain for rerand to fall into. After `placed` A-gates the
    // cumulative filler target is `placed * total_fillers / m`.
    let mut fillers_done = 0usize;

    // A rerand burst on band wire `b` forces every open LGI that reads `b` shut
    // (straddle) or stripped of its `b` term (repair). When those were ALL of a
    // wire's open masks the wire would sit BARE — holding its plaintext value —
    // until its next filler/straddle open, typically thousands of gates later
    // (measured: every interior bare interval of the K=2 build came from this,
    // ~450 per build, median ~10k gates, and every operand read inside one is an
    // exact copy of a C state bit). So before the close, open a replacement LGI
    // that avoids `b`: same-target XOR writes commute, so open-then-close never
    // leaves an instant with no mask on the wire. Extra to the u_w+1 budget
    // (~3 gates per event, ≈0.4% of the build).
    let mut replacements = 0usize;
    let mut read_covers = 0usize; // top-ups kept open on an otherwise uncovered operand
    let mut fire_covers = 0usize; // extra target masks opened at a fire (fire-cover rule)
    // Sample an LGI cycle for wire `w` whose band wires are DISJOINT from every
    // band wire already used by `w`'s open cycles (pairs and balancing wires)
    // and, when given, from `avoid` (the band wire a burst is about to
    // refresh). Disjointness is what makes the wire's mask sum uniform under
    // balanced masks: two identical pairs cancel (`1⊕y⊕xy` twice is 0 — the
    // wire is functionally bare while the bookkeeping counts two masks), two
    // identical `z`s cancel (`z⊕z = 0`, no uniform term left), and a `z` equal
    // to another open pair's wire folds the linear and the quadratic term into
    // an OR (`x ⊕ ¬x∧y = x∨y`, biased 3:1). Each was measured as a phi ≈ 0.25
    // segment population in the gauntlet at 32 band wires; at 256 the last
    // one still occurs on ~10% of opens.
    // Disjointness is best-effort: a band too small to hold `max_open`
    // disjoint cycles (e.g. the n=6 exhaustive test, r=6) relaxes after a
    // bounded number of draws to the weaker "no identical pair, no identical
    // z" rule, then to any cycle; `avoid` is never relaxed (a replacement that
    // read the burst wire would be closed again at once). Relaxations are
    // counted in the diag line; production bands (r = n ≥ 128) never relax.
    let mut relaxed = 0usize;
    // `$extra`: further band wires to stay away from (soft, like `used`) — at a
    // fire, the wires of the operands' polynomials (see the fire-cover rule).
    macro_rules! sample_fresh {
        ($w:expr, $avoid:expr, $extra:expr) => {{
            let w: usize = $w;
            let avoid: Option<u16> = $avoid;
            let extra: &BTreeSet<u16> = $extra;
            let mut used: BTreeSet<u16> = open_cy[w].iter().flatten().copied().collect();
            used.extend(extra.iter().copied());
            let mut tries = 0usize;
            loop {
                let cy = sample_k(&band, lgi_k, &mut rng);
                tries += 1;
                if avoid.is_some_and(|b| cy.iter().any(|&x| x == b)) {
                    assert!(
                        tries <= 1 << 16,
                        "blinded_v5: no LGI cycle on wire {w} avoids band wire {} (R={r}, cycle width {lgi_k})",
                        avoid.unwrap()
                    );
                    continue;
                }
                let clash = cy.iter().any(|x| used.contains(x));
                let weak_clash = (0..cy.len() / 2).any(|i| {
                    open_pairs[w].contains(&(cy[2 * i], cy[2 * i + 1]))
                        || open_pairs[w].contains(&(cy[2 * i + 1], cy[2 * i]))
                }) || (cy.len() % 2 == 1 && open_lin[w].contains(&cy[cy.len() - 1]));
                if !clash || (tries > 256 && !weak_clash) || tries > 4096 {
                    if clash {
                        relaxed += 1;
                    }
                    break cy;
                }
            }
        }};
    }
    macro_rules! keep_covered {
        ($w:expr, $b:expr) => {{
            let w = $w;
            let b = $b;
            let hits = open_cy[w].iter().filter(|cy| cy.iter().any(|&x| x == b)).count();
            let remain = open_cy[w].len() - hits;
            if hits > 0 && remain < min_open {
                // open enough replacements (away from `b`) BEFORE the closes so the
                // wire never drops below `min_open` (never below one at least)
                for _ in remain..min_open {
                    let cy = sample_fresh!(w, Some(b), &BTreeSet::new());
                    emit_lgi(w as u16, &cy, &mut out, &mut open_pairs[w], &mut open_lin[w]);
                    open_cy[w].push(cy);
                    replacements += 1;
                }
            }
        }};
    }
    // A burst gate `b ^= lit(c1) ∧ lit(c2)` with a DATA control `w` reads `w`
    // under its masks; if the other control is one of `w`'s own mask wires the
    // product strips that mask's uniform term (`(x ⊕ z ⊕ q) ∧ ¬z`), and two
    // data controls whose masks share a band wire correlate the same way
    // (measured: phi 0.14–0.26 on the burst's flip vs the plaintext, gauntlet
    // n=256, max_open 2). Redraw until no such coincidence (bounded).
    macro_rules! burst_gate_safe {
        ($b:expr, $active:expr) => {{
            let b: u16 = $b;
            let act: usize = $active;
            let mut tries = 0usize;
            loop {
                let g = burst_gate(b, act, &band, &mut rng);
                let (c1, c2) = (g.ctrls[0].0, g.ctrls[1].0);
                let mask_wires = |w: u16| -> BTreeSet<u16> {
                    if (w as usize) < np {
                        open_cy[w as usize].iter().flatten().copied().collect()
                    } else {
                        BTreeSet::new()
                    }
                };
                let (m1, m2) = (mask_wires(c1), mask_wires(c2));
                let clash = m1.contains(&c2) || m2.contains(&c1) || m1.iter().any(|x| m2.contains(x));
                tries += 1;
                if !clash || tries > 256 {
                    break g;
                }
            }
        }};
    }
    macro_rules! emit_slot {
        () => {{
            let is_repair = slot_plan[si];
            si += 1;
            slots_emitted += 1;
            if in_drain {
                drain_slots += 1;
            }
            let b = band[rng.random_range(0..band.len())];
            if is_repair {
                for w in 0..np {
                    keep_covered!(w, b);
                    for cy in &open_cy[w] {
                        if let Some(gg) = b_g57(w as u16, cy, b) {
                            out.push(gg);
                        }
                    }
                }
                for _ in 0..burst {
                    out.push(burst_gate_safe!(b, if p.burst_band_only { 0 } else { active }));
                    rerand_done += 1;
                }
                for w in 0..np {
                    for cy in &open_cy[w] {
                        if let Some(gg) = b_g57(w as u16, cy, b) {
                            out.push(gg);
                        }
                    }
                }
            } else {
                for w in 0..np {
                    keep_covered!(w, b);
                    let mut idx = 0;
                    while idx < open_cy[w].len() {
                        if open_cy[w][idx].iter().any(|&x| x == b) {
                            let cy = open_cy[w].remove(idx);
                            emit_lgi(w as u16, &cy, &mut out, &mut open_pairs[w], &mut open_lin[w]);
                        } else {
                            idx += 1;
                        }
                    }
                }
                for _ in 0..burst {
                    out.push(burst_gate_safe!(b, if p.burst_band_only { 0 } else { active }));
                    rerand_done += 1;
                }
            }
        }};
    }
    macro_rules! maybe_rerand {
        () => {{
            steps += 1;
            if si < slot_plan.len() && steps % rgap == 0 {
                emit_slot!();
            }
        }};
    }
    // After an open on wire `w`, bring it up to `min_open` open masks (extra to
    // the u_w+1 budget, counted in `min_open_opens`); `$extra` = band wires to
    // stay away from (the fire's wires when called mid-fire).
    let mut min_open_opens = 0usize;
    macro_rules! ensure_min_open {
        ($w:expr, $extra:expr) => {{
            let w: usize = $w;
            while open_cy[w].len() < min_open {
                let cy = sample_fresh!(w, None, $extra);
                emit_lgi(w as u16, &cy, &mut out, &mut open_pairs[w], &mut open_lin[w]);
                open_cy[w].push(cy);
                min_open_opens += 1;
            }
        }};
    }
    macro_rules! filler_open {
        ($w:expr) => {{
            let w = $w;
            if open_cy[w].len() >= max_open {
                let cy = open_cy[w].remove(0);
                emit_lgi(w as u16, &cy, &mut out, &mut open_pairs[w], &mut open_lin[w]);
            }
            let cy = sample_fresh!(w, None, &BTreeSet::new());
            emit_lgi(w as u16, &cy, &mut out, &mut open_pairs[w], &mut open_lin[w]);
            open_cy[w].push(cy);
            ensure_min_open!(w, &BTreeSet::new());
            filler_left[w] -= 1;
            maybe_rerand!();
        }};
    }

    // Encoded I/O: open `max_open` LGIs per wire OFF-circuit (into `pre`), so
    // the trace starts with every data wire already masked.
    let mut pre: Vec<XGate> = Vec::new();
    if p.encoded_io {
        for w in 0..np {
            for _ in 0..max_open {
                let cy = sample_fresh!(w, None, &BTreeSet::new());
                emit_lgi(w as u16, &cy, &mut pre, &mut open_pairs[w], &mut open_lin[w]);
                open_cy[w].push(cy);
            }
        }
    }
    let mut placed = 0usize;
    while placed < m {
        let gi = match ready.pop_front() {
            Some(g) => g,
            None => break,
        };
        let c = src[gi].target as usize;
        let mut ctrls: Vec<(u16, bool, Vec<u16>)> = Vec::new();
        let mut polys: Vec<Poly> = Vec::new();
        let mut undo: Vec<XGate> = Vec::new();
        for &(w, pol) in &src[gi].ctrls {
            let netopen: Vec<(u16, u16)> = open_pairs[w as usize].iter().copied().collect();
            if p.quad_fire {
                // No linearisation: the operand stays under its quadratic masks.
                let mut poly = Poly::default();
                poly.toggle(vec![w]);
                for &(x, y) in &netopen {
                    poly.add_g57_mask(x, y);
                }
                for &z in &open_lin[w as usize] {
                    poly.toggle(vec![z]);
                }
                // Top up with fresh quadratic pairs (single g57s, undone after the
                // fire) until at least max_open quadratic terms mask the operand.
                // An operand with NO open LGI at all (a seldom-used wire between
                // its sparse fillers) keeps its first top-up as a real LGI instead
                // of undoing it: otherwise the wire returns to holding its plain
                // value for the whole idle stretch (measured: tens of thousands of
                // gates on high-half wires). Costs nothing (the undo is dropped).
                let mut guard = 0;
                // top-ups to KEEP as real masks: enough to reach `min_open`
                let mut keep = min_open.saturating_sub(open_cy[w as usize].len());
                // Band wires in use on this operand: its open cycles plus the
                // top-ups of this read. Every top-up draws OUTSIDE this set (see
                // `sample_fresh`: any shared wire biases the mask sum).
                let mut used: BTreeSet<u16> =
                    open_cy[w as usize].iter().flatten().copied().collect();
                // best-effort disjointness (bounded), see `sample_fresh`; the
                // `hard` wires (this top-up's own pair) are never reused
                let mut draw = |used: &BTreeSet<u16>, hard: &[u16], rng: &mut StdRng| -> u16 {
                    let mut tries = 0usize;
                    loop {
                        let x = band[rng.random_range(0..band.len())];
                        tries += 1;
                        if hard.contains(&x) {
                            assert!(tries <= 1 << 16, "blinded_v5: band too small for a read top-up (R={r})");
                            continue;
                        }
                        if !used.contains(&x) || tries > 256 {
                            return x;
                        }
                    }
                };
                while poly.deg2_terms() < max_open.max(1) && guard < 64 {
                    let r1 = draw(&used, &[], &mut rng);
                    used.insert(r1);
                    let r2 = draw(&used, &[r1], &mut rng);
                    used.insert(r2);
                    if keep > 0 {
                        keep -= 1;
                        let mut cy = vec![r1, r2];
                        if p.balanced {
                            let z = draw(&used, &[r1, r2], &mut rng);
                            used.insert(z);
                            cy.push(z);
                        }
                        emit_lgi(w, &cy, &mut out, &mut open_pairs[w as usize], &mut open_lin[w as usize]);
                        poly.add_g57_mask(r1, r2);
                        if let Some(&z) = cy.get(2) {
                            poly.toggle(vec![z]);
                        }
                        open_cy[w as usize].push(cy);
                        read_covers += 1;
                        guard += 1;
                        continue;
                    }
                    out.push(g57(w, r1, r2));
                    undo.push(g57(w, r1, r2));
                    poly.add_g57_mask(r1, r2);
                    if p.balanced {
                        let z = draw(&used, &[r1, r2], &mut rng);
                        used.insert(z);
                        out.push(XGate::cnot(w, z));
                        undo.push(XGate::cnot(w, z));
                        poly.toggle(vec![z]);
                    }
                    guard += 1;
                }
                if diag {
                    n_reads += 1;
                    let q = poly.deg2_terms();
                    if q == 0 {
                        n_bare += 1;
                    }
                    rho_min = rho_min.min(2 * q);
                    rho_sum += 2 * q;
                }
                if !pol {
                    poly.toggle(Vec::new());
                }
                polys.push(poly);
                continue;
            }
            let (mut rho, added) = linearize(w, &netopen, &mut out);
            for gg in added.into_iter().rev() {
                undo.push(gg);
            }
            for &z in &open_lin[w as usize] {
                if let Some(pp) = rho.iter().position(|&v| v == z) {
                    rho.remove(pp);
                } else {
                    rho.push(z);
                }
            }
            rho.sort_unstable();
            // Top up fresh disjoint g57 pairs until |rho| >= min_mask (each pair
            // toggles two band wires in rho). First bring the open count up to
            // max_open, then keep adding single pairs until the masking floor is
            // met -- so no operand is ever read under fewer than min_mask wires.
            let mut extra = max_open.saturating_sub(netopen.len());
            if extra == 0 && rho.len() < min_mask {
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
                if rho.len() >= min_mask || guard > 40 {
                    break;
                }
                extra = 1;
                guard += 1;
            }
            // Deterministic floor finisher: the random top-up above usually meets
            // the floor, but `|ρ|` is structurally even and a random walk can
            // stall below the target (odd `min_mask`, or a `min_mask` near `r`).
            // If it fell short, add guaranteed-FRESH disjoint pairs (both wires
            // not currently in `ρ`) so `|ρ|` provably grows by 2 each step until
            // `≥ min_mask` — making the floor a real, fail-closed guarantee. It is
            // a no-op at production params (the loop already reaches `|ρ| ≥ 4`),
            // so the production circuit is unchanged; it fires only in the
            // degenerate/misconfig regime the clamp on `min_mask` keeps feasible.
            // (Fresh pairs need two band wires outside `ρ`; with balanced `z`
            // terms folded into `ρ` that can be impossible on a tiny band, so
            // stop rather than spin — the floor is then bounded by the band.)
            while rho.len() < min_mask && rho.len() + 2 <= r {
                let mut r1 = band[rng.random_range(0..band.len())];
                while rho.contains(&r1) {
                    r1 = band[rng.random_range(0..band.len())];
                }
                let mut r2 = band[rng.random_range(0..band.len())];
                while r2 == r1 || rho.contains(&r2) {
                    r2 = band[rng.random_range(0..band.len())];
                }
                out.push(g57(w, r1, r2));
                out.push(g57(w, r2, r1));
                undo.push(g57(w, r2, r1));
                undo.push(g57(w, r1, r2));
                rho.push(r1);
                rho.push(r2);
            }
            if diag {
                n_reads += 1;
                if rho.is_empty() {
                    n_bare += 1;
                }
                rho_min = rho_min.min(rho.len());
                rho_sum += rho.len();
            }
            ctrls.push((w, pol, rho));
        }
        let mut fires = Vec::new();
        if p.quad_fire {
            quad_fire(src[gi].target, &polys, src[gi].comp, &mut fires);
        } else {
            masked_fire(src[gi].target, &ctrls, src[gi].comp, &mut fires);
        }
        if open_cy[c].len() >= max_open {
            let cy = open_cy[c].remove(0);
            emit_lgi(c as u16, &cy, &mut out, &mut open_pairs[c], &mut open_lin[c]);
        }
        // Fire-cover bracket. During the fire the target's segments are
        // `c_old ⊕ M_c ⊕ partial-sum-of-monomials`; whenever a band wire of one
        // of `c`'s open masks also occurs in the operands' polynomials, the
        // monomials cancel or fold that mask's uniform term (`z ⊕ z`,
        // `z ⊕ z∧u`) and the segment turns biased toward `c_new` (measured phi
        // 0.1–0.25 on 1–2% of targets at 32–64 band wires; at 256 band wires a
        // collision occurs at ~2/3 of the fires). So every monomial block is
        // BRACKETED by a temporary mask on `c` drawn away from every band wire
        // of the fire: opened before the first monomial, closed after the last
        // (~6 gates per fire, no change to any read polynomial). The mid-fire
        // straddle open is drawn away from those wires as well.
        let mut fire_wires: BTreeSet<u16> = polys
            .iter()
            .flat_map(|q| q.0.iter().flatten().copied())
            .filter(|&x| (x as usize) >= np)
            .collect();
        let cover = sample_fresh!(c, None, &fire_wires);
        emit_lgi(c as u16, &cover, &mut out, &mut open_pairs[c], &mut open_lin[c]);
        fire_wires.extend(cover.iter().copied());
        fire_covers += 1;
        // straddle: fire-part-1, OPEN a fresh LGI on c (mid-fire), fire-part-2 --
        // each half emitted in an independently random order (monomials commute).
        let cut = ((fires.len() + 1) / 2).min(fires.len());
        shuffle_slice(&mut fires[..cut], &mut rng);
        shuffle_slice(&mut fires[cut..], &mut rng);
        out.extend_from_slice(&fires[..cut]);
        let cy = sample_fresh!(c, None, &fire_wires);
        emit_lgi(c as u16, &cy, &mut out, &mut open_pairs[c], &mut open_lin[c]);
        open_cy[c].push(cy);
        ensure_min_open!(c, &fire_wires);
        out.extend_from_slice(&fires[cut..]);
        // close the bracket (toggles its pairs/linear term back off)
        emit_lgi(c as u16, &cover, &mut out, &mut open_pairs[c], &mut open_lin[c]);
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
        // Even running schedule: catch up to `placed * total_fillers / m` opens.
        let filler_target = placed * total_fillers / m.max(1);
        while fillers_done < filler_target {
            let tf: usize = filler_left.iter().sum();
            match pick_weighted(&filler_left, tf, &mut rng) {
                Some(w) => {
                    filler_open!(w);
                    fillers_done += 1;
                }
                None => break,
            }
        }
    }
    debug_assert!(placed == m);
    in_drain = true; // everything below is emitted after the last A-gate placement
    // Safety drain for any rounding remainder (0 in practice — the running
    // schedule reaches `total_fillers` by the last placement); emits no rerand.
    loop {
        let tf: usize = filler_left.iter().sum();
        match pick_weighted(&filler_left, tf, &mut rng) {
            Some(w) => filler_open!(w),
            None => break,
        }
    }
    // No end-flush: the rate above calibrates the slots to land during the pass,
    // so `si` reaches `slot_plan.len()` whenever the plan fits the step budget
    // (`slots ≤ total_steps`, always true for the auto `m/(4K)` plan). If an
    // explicit over-large `rerand_level`/`rerand_repair` requested more slots than
    // there are placement steps, `rgap` floors to 1 and the surplus cannot land;
    // warn rather than truncate the band mixing silently (RC: no silent caps).
    if si < slot_plan.len() {
        eprintln!(
            "[bv5] WARN: {} of {} rerand slots undelivered (requested slots > {} \
             placement steps); band under-mixed — lower rerand_level/rerand_repair \
             or raise the circuit size",
            slot_plan.len() - si,
            slot_plan.len(),
            total_steps
        );
    }
    // Final drain: close every still-open LGI so the data wires hold A's
    // output -- ON the circuit normally, OFF-circuit (`post`) for encoded I/O.
    let mut post: Vec<XGate> = Vec::new();
    for w in 0..np {
        while let Some(cy) = open_cy[w].pop() {
            let sink = if p.encoded_io { &mut post } else { &mut out };
            emit_lgi(w as u16, &cy, sink, &mut open_pairs[w], &mut open_lin[w]);
        }
    }
    let _ = straddles_left;
    if diag {
        eprintln!(
            "[bv5-diag] A-gates={m} straddled(firing hidden)={m} unplaced=0  control reads={n_reads}  \
             BARE (rho empty)={n_bare} ({:.3}%)  min |rho|={} (floor {min_mask})  mean |rho|={:.2}",
            100.0 * n_bare as f64 / n_reads.max(1) as f64,
            if n_reads == 0 { 0 } else { rho_min },
            rho_sum as f64 / n_reads.max(1) as f64
        );
        eprintln!(
            "[bv5-diag] rerand slots: emitted={slots_emitted} of plan={}  \
             DRAIN (after last A-gate)={drain_slots} ({:.1}%)  main-loop={}  \
             cover-replacement LGIs={replacements}  read-cover LGIs={read_covers}  \
             fire-cover LGIs={fire_covers}  min-open opens={min_open_opens}  relaxed (non-disjoint) samples={relaxed}",
            slot_plan.len(),
            100.0 * drain_slots as f64 / slots_emitted.max(1) as f64,
            slots_emitted - drain_slots
        );
    }

    BlindedV5Output {
        gates: out,
        num_wires: total,
        atoms,
        rerand_done,
        pre_gates: pre,
        post_gates: post,
        r_used: r,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    // A random g57 circuit on `n` wires (from_g57 triples with distinct wires).
    fn random_a(n: u16, m: usize, rng: &mut StdRng) -> Vec<XGate> {
        let mut out = Vec::with_capacity(m);
        while out.len() < m {
            let a = rng.random_range(0..n);
            let x = rng.random_range(0..n);
            let y = rng.random_range(0..n);
            if a != x && a != y && x != y {
                out.push(XGate::from_g57([a, x, y]));
            }
        }
        out
    }

    // The gadget must compute A on the data wires for EVERY data input and
    // EVERY band state (masks open and close symmetrically; band updates never
    // straddle the masks that read them). Exhaustive over the data, sampled
    // over the band, both read modes, several seeds.
    #[test]
    fn gadget_computes_a_exhaustively_in_both_read_modes() {
        let n: u16 = 6;
        let np = n as usize;
        for seed in 0..6u64 {
            let mut rng = StdRng::seed_from_u64(0xB5_0000 + seed);
            let a = random_a(n, 24, &mut rng);
            // (quad_fire, balanced, encoded_io, k, repair slots): both read modes,
            // both mask kinds, encoded I/O, odd K (K3 == K2 + the balancing wire),
            // K4 (two pairs per LGI) and REPAIR-kind rerand slots.
            for (quad_fire, balanced, encoded_io, k, repair) in [
                (true, false, false, 2, 0),
                (false, false, false, 2, 0),
                (true, true, false, 2, 0),
                (false, true, false, 2, 0),
                (true, false, true, 2, 0),
                (true, true, true, 2, 0),
                (true, true, false, 3, 0),
                (true, false, false, 3, 0),
                (true, true, false, 4, 2),
                (false, true, false, 4, 2),
                (true, true, true, 2, 2),
            ] {
                let p = BlindedV5Params {
                    quad_fire,
                    balanced,
                    encoded_io,
                    k,
                    rerand_repair: repair,
                    // exercise both the min-open rule (default 2) and its absence
                    min_open: if repair > 0 { 1 } else { 2 },
                    ..BlindedV5Params::production(100 + seed)
                };
                let out = gadgetize_blinded_v5(&a, np, &p);
                assert_eq!(out.num_wires, 2 * np);
                assert_eq!(out.pre_gates.is_empty(), !encoded_io);
                assert_eq!(out.post_gates.is_empty(), !encoded_io);
                if encoded_io {
                    // every data wire is masked at the start and at the end
                    for w in 0..np as u16 {
                        assert!(out.pre_gates.iter().any(|g| g.target == w));
                        assert!(out.post_gates.iter().any(|g| g.target == w));
                    }
                }
                for band in 0..8u64 {
                    let band_bits = if band == 0 { 0 } else { rng.random::<u64>() >> (64 - np) };
                    for data in 0..(1u64 << np) {
                        let mut expect = data;
                        for g in &a {
                            expect = g.apply_u64(expect);
                        }
                        let mut st = data | (band_bits << np);
                        for g in out.pre_gates.iter().chain(&out.gates).chain(&out.post_gates) {
                            st = g.apply_u64(st);
                        }
                        assert_eq!(
                            st & ((1u64 << np) - 1),
                            expect,
                            "seed {seed} quad_fire={quad_fire} balanced={balanced} encoded_io={encoded_io} k={k} repair={repair} band {band:#x} data {data:#x}"
                        );
                    }
                }
                if quad_fire {
                    // quad-fire never emits the legacy reverse-pair brackets:
                    // every fire monomial is a plain conjunction, degree <= 4
                    assert!(out.gates.iter().all(|g| g.width() <= 4));
                }
            }
        }
    }
}

