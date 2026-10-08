//! Embedded masking computation. Takes a circuit `A` on `n` wires and builds an equivalent circuit
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
//! re-encoding, circuit_mixer's final float, the crossing walk) stretches the window to
//! the operand's idle interval -- measured as the C-vs-G affine ridge (~6k exact
//! relations, rho 1.00 through the whole pipeline). QUAD-FIRE never linearises:
//! the operand is the ANF polynomial `w ^ sum_(x,y) (1 ^ y ^ x&y)` over its open
//! pairs (topped up to `max_open` fresh quadratic pairs, undone after the fire)
//! and the fire is the polynomial product (degree <= 4 at K=2). No read ever
//! leaves a wire affine, so there is no window to stretch: relations stay at
//! the public I/O fringe (~150-500) through db_mixing, split, crossing and
//! compression, and the gadget is ~12% smaller (no linearise/undo gates).
//! `quad_fire = false` (env `EMBEDDED_MASKING_QUAD_FIRE=0`) keeps the legacy linear read.
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

use super::preprocessing_shuffling::{FireBox, ShufflingStats, apply_shuffling, remap_gate};
use crate::circuit::xgate::XGate;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::{BTreeSet, HashMap, HashSet};

/// Configuration for [`preprocess_embedded_masking`].
#[derive(Clone, Copy, Debug)]
pub struct EmbeddedMaskingParams {
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
    /// `false` = the legacy linear read (env `EMBEDDED_MASKING_QUAD_FIRE=0`).
    pub quad_fire: bool,
    /// BALANCED masks (default ON since 2026-09-07; env `EMBEDDED_MASKING_BALANCED=0` for
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
    /// (env `EMBEDDED_MASKING_BURST_BANDONLY=1`).
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
    /// Internal active-role transfers per fire box. `0` disables preprocessing
    /// shuffling and preserves the original seeded circuit; `1` is one carrier
    /// transfer per ORIGINAL-circuit write (punchlist #4's intent); larger
    /// doses additionally shorten intra-box partial-sum residencies and supply
    /// read-pressure refreshes. Safe unit boundaries and available partners can
    /// reduce the number of actual transfers.
    pub shuffling_segments: usize,
    /// `true`: restore each data role to its original wire before its box
    /// ends. `false`: PERSISTENT carrier migration -- data roles hop across
    /// wires for the whole stream and are swapped home once, at the stream
    /// tail, so port contracts hold in both modes. Band roles remain
    /// permuted either way.
    pub shuffling_return_home: bool,
    /// `0`: eager per-box mask drains (legacy). Positive: JUST-IN-TIME drains
    /// (punchlist #8) -- transfer masks outlive their fire box and drain
    /// right before a read of the masked role or a write of the holder, and
    /// any bare data role whose next touch is more than this many source
    /// gates away is immediately re-masked. Bare role values then live only
    /// in short read-adjacent stretches instead of being reborn at every box
    /// end (the migration-drain celebrity class). Requires persistent
    /// migration (`shuffling_return_home == false`).
    pub shuffling_jit_gap: usize,
}

impl EmbeddedMaskingParams {
    /// Effective managed TDP recipe: seed from the honest source prefix while
    /// retaining the established data-and-band burst refresh policy. Dirty fire
    /// helpers use band wires independently (`EmbeddedMaskingExecution`).
    pub fn managed_tdp(seed: u64, active_wires: usize) -> Self {
        Self {
            active_wires,
            ..Self::production(seed)
        }
    }

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
            shuffling_segments: 0,
            shuffling_return_home: true,
            shuffling_jit_gap: 0,
        }
    }
}

/// Resolved comparison controls. Environment parsing belongs to executable adapters.
#[derive(Clone, Copy, Debug)]
pub struct EmbeddedMaskingExecution {
    pub ancilla_band_only: bool,
    pub diagnostics: bool,
    /// Record mask-coverage intervals in the original logical-role stream.
    pub record_hot_intervals: bool,
}

impl Default for EmbeddedMaskingExecution {
    fn default() -> Self {
        Self {
            ancilla_band_only: true,
            diagnostics: false,
            record_hot_intervals: false,
        }
    }
}

/// Result of [`preprocess_embedded_masking`].
pub struct EmbeddedMaskingOutput {
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
    /// Always 0: the fire uses dirty band wires, never clean ancillas. Kept so
    /// callers that zeroed a scratch range keep compiling (nothing to zero).
    pub scratch_wires: usize,
    /// Effective band pool used.
    pub r_used: usize,
    /// Logical role -> physical wire after compute and the mapped decoder.
    /// Data entries are identity in both shuffling modes (persistent mode
    /// restores ports at the stream tail). Encoded I/O callers using a
    /// carried layout must read outputs through this map.
    pub final_layout: Vec<u16>,
    /// Target-write distribution and gate cost of the optional transform.
    pub shuffling_stats: ShufflingStats,
    /// Optional mask-coverage diagnostics before physical-role shuffling.
    /// Tuples are (logical data role, original gate start, original gate end),
    /// not physical wires or positions in the transformed gate stream.
    pub role_hot_intervals: Vec<(u16, usize, usize)>,
    #[cfg(test)]
    pub(crate) shuffling_boxes: Vec<FireBox>,
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
    (0..cy.len() / 2)
        .map(|i| g57(w, cy[2 * i], cy[2 * i + 1]))
        .collect()
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

/// In-place Fisher-Yates shuffle of fire UNITS (they commute).
fn shuffle_units(s: &mut [Vec<XGate>], rng: &mut StdRng) {
    for i in (1..s.len()).rev() {
        let j = rng.random_range(0..=i);
        s.swap(i, j);
    }
}

/// In-place Fisher-Yates shuffle (the masked-fire monomials commute, so each
/// half of a fire batch can be emitted in an independently random order).
#[allow(dead_code)]
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

/// One operand of a masked fire: the data wire, its literal polarity and the
/// LGI cycles currently open on it (`[x, y]` or `[x, y, z]`: the mask term is
/// `1 ⊕ ¬x∧y ⊕ z = 1 ⊕ y ⊕ x·y ⊕ z`).
struct Operand {
    w: u16,
    pol: bool,
    cycles: Vec<Vec<u16>>,
}

impl Operand {
    /// Constant of the operand's plaintext polynomial: one per g57 PAIR (a K=4
    /// cycle carries two pairs), plus one for a negative literal.
    fn constant(&self) -> bool {
        let pairs: usize = self.cycles.iter().map(|cy| cy.len() / 2).sum();
        (pairs % 2 == 1) ^ !self.pol
    }
    /// Linear band terms: the `y` of every pair and the balancing `z` (odd
    /// trailing wire) of every open mask.
    fn lin(&self) -> Vec<u16> {
        self.cycles
            .iter()
            .flat_map(|cy| {
                let ys = (0..cy.len() / 2).map(move |i| cy[2 * i + 1]);
                let z = if cy.len() % 2 == 1 {
                    Some(cy[cy.len() - 1])
                } else {
                    None
                };
                ys.chain(z)
            })
            .collect()
    }
    /// Quadratic band terms `x·y` of every pair of every open mask.
    fn quad(&self) -> Vec<(u16, u16)> {
        self.cycles
            .iter()
            .flat_map(|cy| (0..cy.len() / 2).map(move |i| (cy[2 * i], cy[2 * i + 1])))
            .collect()
    }
}

/// HOT-VALUE MANIFEST: the gate intervals during which a data wire's value is an
/// AFFINE function of the plaintext and the (visible) band wires — i.e. the wire
/// carries a sustained linear relation with a wire segment of the input circuit.
///
/// A wire under masks `1 ⊕ y_i ⊕ x_i·y_i ⊕ z_i` is affine in the wire state
/// exactly when its net set of open `g57` PAIRS is empty: the `y` and `z` terms
/// are single band wires (visible, hence affine), so only the quadratic `x_i·y_i`
/// terms hide the value. The coverage rules keep `min_open` pairs open between a
/// wire's first open and its last close, so with a correct build these intervals
/// are exactly the public I/O fringe: `[0, first open)` and `[last close, end)`.
/// Anything else in the list is a defect.
///
/// This is the list to hand to the DB-mixing stage as splice seeds. It is also a
/// map of the module's weak points, so it must never ship with a deliverable.
pub fn hot_intervals(gates: &[XGate], np: usize, r: usize) -> Vec<(u16, usize, usize)> {
    let (lo, hi) = (np as u16, (np + r) as u16);
    let is_pair = |g: &XGate| -> Option<(u16, u16)> {
        if !g.comp || g.ctrls.len() != 2 {
            return None;
        }
        let ((w0, p0), (w1, p1)) = (g.ctrls[0], g.ctrls[1]);
        if p0 == p1 || w0 < lo || w0 >= hi || w1 < lo || w1 >= hi {
            return None;
        }
        Some(norm_pair(w0, w1))
    };
    let mut open: Vec<BTreeSet<(u16, u16)>> = vec![BTreeSet::new(); np];
    let mut since: Vec<usize> = vec![0; np]; // start of the current uncovered run
    let mut out: Vec<(u16, usize, usize)> = Vec::new();
    for (i, g) in gates.iter().enumerate() {
        let t = g.target as usize;
        if t >= np {
            continue;
        }
        if let Some(pr) = is_pair(g) {
            let was = open[t].is_empty();
            if !open[t].insert(pr) {
                open[t].remove(&pr);
            }
            let now = open[t].is_empty();
            if was && !now {
                out.push((t as u16, since[t], i)); // covered from here on
            } else if !was && now {
                since[t] = i + 1;
            }
        }
    }
    for w in 0..np {
        if open[w].is_empty() {
            out.push((w as u16, since[w], gates.len()));
        }
    }
    out.sort_unstable();
    out
}

/// Unordered band pair, normalised.
fn norm_pair(a: u16, b: u16) -> (u16, u16) {
    if a <= b { (a, b) } else { (b, a) }
}

/// `t ^= u ∧ v` as a 2-control gate, collapsing `u == v` to a CNOT.
fn and2(t: u16, u: u16, v: u16) -> XGate {
    if u == v {
        XGate::cnot(t, u)
    } else {
        conj(t, &[(u, true), (v, true)])
    }
}

/// MASKED FIRE with two-control gates only, and NO clean ancillas (RC
/// 2026-09-08: "we cannot have any clean ancillas here").
///
/// The store is a `g57` 2-control identity ball, so a gate with three or more
/// controls is never spliced and would carry `C`'s structure through mixing
/// verbatim. The product of the two operand polynomials therefore has to be
/// realised with 2-control gates. An earlier version borrowed four CLEAN
/// scratch wires for the partial products; that was wrong twice over: a wire
/// that is 0 at every instant outside a fire is a FUNCTION-level invariant (it
/// survives any equivalent rewriting, so mixing cannot hide it) and its
/// non-zero stretches delimit exactly the fire blocks — the segmentation the
/// hidden-firing design exists to prevent — and it silently required the
/// evaluator to zero them.
///
/// Instead the partial products go on DIRTY band wires, via the identity
/// `t ^= h∧y; h ^= P∧x; t ^= h∧y; h ^= P∧x`, whose net effect is
/// `t ^= P·x·y` for ANY prior value of `h` (`h_0·y ⊕ (h_0⊕Px)·y = Pxy`) and
/// which leaves `h` restored. The prior value also blinds the intermediate for
/// free, so the explicit blinders the clean version needed are gone.
///
/// Writing `a = a' ⊕ c_a ⊕ s_a ⊕ Q_a` (constant, linear band sum `Σ(y_i⊕z_i)`,
/// quadratic part `Σ x_i y_i`), the 16 cross terms of the product are emitted
/// as: single 2-control gates wherever both factors are wires or single band
/// literals (the `s` sums are expanded term by term, so no aggregate is ever
/// materialised on a wire); the 4-gate bracket above for the 3-literal terms;
/// and an 8-gate two-ancilla bracket for `x_iy_i · x'_jy'_j`. Groupings there
/// are CROSS-operand and checked against the pairs open on any data wire, so no
/// ancilla ever holds a mask's own quadratic term.
#[allow(clippy::too_many_arguments)]
fn product_fire(
    t: u16,
    ops: &[Operand],
    comp: bool,
    open_pair_set: &BTreeSet<(u16, u16)>,
    open_pairs: &[BTreeSet<(u16, u16)>],
    band: &[u16],
    np: usize,
    r: usize,
    rng: &mut StdRng,
    blinded: &mut usize,
    census: &mut [usize; 8],
    ancilla_band_only: bool,
) -> (Vec<Vec<XGate>>, Vec<(usize, u16)>) {
    let mut units: Vec<Vec<XGate>> = Vec::new();
    // (unit index, band ancilla) of every bracket whose FINAL gate is a pure
    // ancilla restore: the caller may drop it when the ancilla is outside the
    // fire's full stability set (open masks + cover + straddle), letting the
    // ancilla drift instead of re-materialising its prior value (the
    // celebrity-value surface, RC 2026-10-06).
    let mut droppable: Vec<(usize, u16)> = Vec::new();
    match ops.len() {
        0 => {
            if comp {
                units.push(vec![XGate::x_gate(t)]);
            }
        }
        1 => {
            let a = &ops[0];
            units.push(vec![XGate::cnot(t, a.w)]);
            for l in a.lin() {
                units.push(vec![XGate::cnot(t, l)]);
            }
            for (x, y) in a.quad() {
                units.push(vec![and2(t, x, y)]);
            }
            if comp ^ a.constant() {
                units.push(vec![XGate::x_gate(t)]);
            }
        }
        _ => {
            let (a, b) = (&ops[0], &ops[1]);
            let (ca, cb) = (a.constant(), b.constant());
            let (la, lb) = (a.lin(), b.lin());
            let (qa, qb) = (a.quad(), b.quad());
            // dirty ancillas: band wires OUTSIDE the fire's own wires, so that
            // restoring them before the block ends cannot disturb any mask
            let used: BTreeSet<u16> = ops
                .iter()
                .flat_map(|o| o.cycles.iter().flatten().copied())
                .collect();
            // A dirty ancilla is picked PER TERM: it must differ from every wire
            // that term touches (otherwise a gate would control on its own
            // target, and the bracket algebra would not hold), preferring one
            // outside the operands' masks so the choice carries no structure.
            // Any wire may serve, DATA wires included (RC 2026-09-08): the
            // bracket restores whatever was there, so while it is borrowed the
            // wire carries its own masked value XOR the partial product — more
            // masking, not less — and drawing from all 2n wires keeps the fire's
            // ancilla traffic from concentrating on the band. Excluded: the
            // target, this gate's operands, and the term's own wires.
            // Ancilla pool: BAND wires by default; `EMBEDDED_MASKING_ANC_POOL=all` also allows
            // data wires. Data wires work functionally (the bracket restores any
            // prior value) and enlarge the pool, but a product XORed onto a
            // MASKED data wire can partially cancel that wire's own mask —
            // band wires are not mutually independent, so the borrowed product
            // need not be independent of the wire's mask sum — which leaves the
            // wire correlated with its plaintext. Measured over three gauntlet
            // instances at 16,384 samples: pool=all flags w1 on 6 targets in one
            // instance (phi 0.134), pool=band flags no w1 in any; the
            // wide-monomial control is clean in all three.
            let band_only = ancilla_band_only;
            let all: Vec<u16> = if band_only {
                band.to_vec()
            } else {
                (0..(np + r) as u16).collect()
            };
            // `xored` is the band pair this term will XOR onto the ancilla. A
            // data wire whose OWN open masks contain that pair must be excluded:
            // XORing `x_i∧y_i` onto it would cancel that mask's quadratic part
            // and leave the wire correlated with its own plaintext (measured:
            // phi 0.134, gauntlet w1, n=64 k=66).
            let pick = |rng: &mut StdRng, must_avoid: &[u16], xored: (u16, u16)| -> u16 {
                let pr = norm_pair(xored.0, xored.1);
                let free: Vec<u16> = all
                    .iter()
                    .copied()
                    .filter(|w| {
                        *w != t
                            && *w != a.w
                            && *w != b.w
                            && !must_avoid.contains(w)
                            && ((*w as usize) >= np || !open_pairs[*w as usize].contains(&pr))
                    })
                    .collect();
                assert!(
                    !free.is_empty(),
                    "no wire free for a dirty fire ancilla (R={}, term needs {} wires)",
                    band.len(),
                    must_avoid.len()
                );
                let pref: Vec<u16> = free.iter().copied().filter(|w| !used.contains(w)).collect();
                let pool = if pref.is_empty() { &free } else { &pref };
                pool[rng.random_range(0..pool.len())]
            };
            // `t ^= P · x · y` on a dirty `h`. The net target update is
            // complete after the third gate; the fourth only restores `h`.
            // For BAND ancillas the restore is dropped (RC 2026-10-06): it
            // re-materialised the ancilla's prior value verbatim once per
            // bracket (~1,000x per hot wire = the celebrity-value surface,
            // and recurrence >= 4 implies boundary exposure deterministically).
            // Without it the ancilla DRIFTS by `P∧x` -- band junk has no
            // value contract, and consecutive borrows then differ by data
            // QUADRATICS, never linear relations. Data ancillas
            // (ANC_POOL=all) still restore: their values are real.
            // The net target update is complete after the third gate; the
            // fourth only restores `h`. Emit it always -- the caller strips it
            // for band ancillas outside the fire's stability set (see
            // `droppable`), because band values must stay stable while any
            // open mask, cover or straddle references them.
            let and3 = |p: u16, x: u16, y: u16, h: u16| -> Vec<XGate> {
                vec![and2(t, h, y), and2(h, p, x), and2(t, h, y), and2(h, p, x)]
            };
            // wires × wires / wires × band literals: plain 2-control gates
            units.push(vec![and2(t, a.w, b.w)]);
            census[0] += 1; // a'b'
            for &l in &lb {
                units.push(vec![and2(t, a.w, l)]);
                census[1] += 1; // wire x linear
            }
            for &l in &la {
                units.push(vec![and2(t, l, b.w)]);
                census[1] += 1;
            }
            for &li in &la {
                for &lj in &lb {
                    units.push(vec![and2(t, li, lj)]);
                    census[2] += 1; // linear x linear
                }
            }
            if ca {
                units.push(vec![XGate::cnot(t, b.w)]);
                for &l in &lb {
                    units.push(vec![XGate::cnot(t, l)]);
                }
                for &(x, y) in &qb {
                    units.push(vec![and2(t, x, y)]);
                }
                census[3] += 1 + lb.len() + qb.len(); // constant x everything
            }
            if cb {
                units.push(vec![XGate::cnot(t, a.w)]);
                for &l in &la {
                    units.push(vec![XGate::cnot(t, l)]);
                }
                for &(x, y) in &qa {
                    units.push(vec![and2(t, x, y)]);
                }
                census[3] += 1 + la.len() + qa.len();
            }
            // 3-literal terms: (a' or one l_i) × x'y', and the mirror
            for &(x, y) in &qb {
                let h = pick(rng, &[x, y, a.w], (a.w, x));
                let u = and3(a.w, x, y, h);
                census[4] += u.len(); // wire x quadratic (bracket)
                if (h as usize) >= np {
                    droppable.push((units.len(), h));
                }
                units.push(u);
                for &l in &la {
                    let h = pick(rng, &[x, y, l], (l, x));
                    let u = and3(l, x, y, h);
                    census[5] += u.len(); // linear x quadratic
                    if (h as usize) >= np {
                        droppable.push((units.len(), h));
                    }
                    units.push(u);
                }
            }
            for &(x, y) in &qa {
                let h = pick(rng, &[x, y, b.w], (b.w, x));
                let u = and3(b.w, x, y, h);
                census[4] += u.len();
                if (h as usize) >= np {
                    droppable.push((units.len(), h));
                }
                units.push(u);
                for &l in &lb {
                    let h = pick(rng, &[x, y, l], (l, x));
                    let u = and3(l, x, y, h);
                    census[5] += u.len();
                    if (h as usize) >= np {
                        droppable.push((units.len(), h));
                    }
                    units.push(u);
                }
            }
            // Q_a·Q_b on two dirty ancillas: 8 gates, cross-operand grouping,
            // checked against the pairs open anywhere
            let ok = |p: (u16, u16)| -> bool {
                p.0 == p.1 || !open_pair_set.contains(&norm_pair(p.0, p.1))
            };
            for &(x, y) in &qa {
                for &(x2, y2) in &qb {
                    let cands = [((x, x2), (y, y2)), ((x, y2), (y, x2))];
                    let pick2 = cands
                        .iter()
                        .find(|(p1, p2)| ok(*p1) && ok(*p2))
                        .unwrap_or_else(|| {
                            *blinded += 1;
                            &cands[0]
                        });
                    let (p1, p2) = *pick2;
                    let h1 = pick(rng, &[p1.0, p1.1, p2.0, p2.1], p1);
                    let h2 = pick(rng, &[p1.0, p1.1, p2.0, p2.1, h1], p2);
                    // The Gray walk needs the mid-walk restore of the FIRST
                    // ancilla (the fourth t-read uses its original value), so
                    // only the LAST toggle is a pure restore. Randomise which
                    // ancilla toggles last (spreading the inherent mid-walk
                    // rebirth across wires) and drop its final restore when it
                    // is a band wire, as in `and3`.
                    let ((fa, fp), (sa, sp)) = if rng.random_bool(0.5) {
                        ((h1, p1), (h2, p2))
                    } else {
                        ((h2, p2), (h1, p1))
                    };
                    let u = vec![
                        and2(t, h1, h2),
                        and2(fa, fp.0, fp.1),
                        and2(t, h1, h2),
                        and2(sa, sp.0, sp.1),
                        and2(t, h1, h2),
                        and2(fa, fp.0, fp.1),
                        and2(t, h1, h2),
                        and2(sa, sp.0, sp.1),
                    ];
                    census[6] += u.len(); // quadratic x quadratic (bracket)
                    if (sa as usize) >= np {
                        droppable.push((units.len(), sa));
                    }
                    units.push(u);
                }
            }
            if comp ^ (ca && cb) {
                units.push(vec![XGate::x_gate(t)]);
                census[7] += 1;
            }
        }
    }
    (units, droppable)
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
pub fn seed_band_mode(
    np: usize,
    r: usize,
    active_wires: usize,
    seed: u64,
    balanced: bool,
) -> Vec<XGate> {
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

/// Emit `target ^= F(lits)` for 1..=2 literals using g57-form pieces.
/// Returns whether a residual constant complement was folded into `F`
/// (irrelevant for band seeding: constants only complement junk).
/// Revived verbatim from the pre-reorganization tree
/// (5f386381:src/preprocessing/gadgets.rs).
fn emit_g57_form(
    target: u16,
    lits: &[(u16, bool)],
    rng: &mut impl Rng,
    out: &mut Vec<XGate>,
) -> bool {
    match *lits {
        [(w, true)] => {
            out.push(XGate::cnot(target, w));
            false
        }
        [(w, false)] => {
            // ¬w = w ^ 1.
            out.push(XGate::cnot(target, w));
            true
        }
        [a, b] => {
            let ((xw, xp), (yw, yp)) = if rng.random_bool(0.5) { (a, b) } else { (b, a) };
            match (xp, yp) {
                // 1 ^ ¬x·y is exactly the g57 monomial.
                (false, true) => {
                    out.push(XGate::from_g57([target, xw, yw]));
                    true
                }
                (true, false) => {
                    out.push(XGate::from_g57([target, yw, xw]));
                    true
                }
                // x·y = (1 ^ ¬x·y) ^ y ^ 1.
                (true, true) => {
                    out.push(XGate::from_g57([target, xw, yw]));
                    out.push(XGate::cnot(target, yw));
                    true
                }
                // ¬x·¬y = (1 ^ ¬x·y) ^ x — no residual constant.
                (false, false) => {
                    out.push(XGate::from_g57([target, xw, yw]));
                    out.push(XGate::cnot(target, xw));
                    false
                }
            }
        }
        _ => unreachable!("emit_g57_form takes 1..=2 literals"),
    }
}

/// Rich band seed (punchlist #5): per band wire, a private pivot CNOT,
/// 1..=7 extra linear source CNOTs, and `fill_nl` nonlinear g57-form
/// product terms cascading over earlier band wires. Revived from the
/// pre-reorganization production fill `emit_band_fill_nl_pivots_src`
/// (5f386381:src/preprocessing/gadgets.rs; measured leakage-neutral).
/// Sources are drawn from `0..active`; band wires are `np..np+r`.
/// `avoid` lists raw data-wire pairs (min, max) the nonlinear products must not
/// read together: a seed product on a first-layer C gate's control pair is a
/// ready-made copy of that gate's nonlinear term (fringe fix 2).
pub fn seed_band_rich(
    np: usize,
    r: usize,
    active_wires: usize,
    seed: u64,
    fill_nl: usize,
    avoid: &std::collections::HashSet<(u16, u16)>,
) -> Vec<XGate> {
    let n = if active_wires == 0 || active_wires > np {
        np
    } else {
        active_wires
    };
    let band: Vec<u16> = ((np as u16)..((np + r) as u16)).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(r * (5 + 2 * fill_nl));
    // Reserving needs room to spare, not just b <= n: every reserved pivot is
    // removed from the linear pool AND from the product-source draw, so at
    // b == n there is no legal non-pivot data wire left and the draw cannot
    // terminate. Leave at least a quarter of the data wires unreserved.
    let reserve = band.len() + band.len() / 3 + 2 <= n;
    // One private pivot per band wire, drawn without replacement.
    let pivots: Vec<usize> = if reserve {
        let mut all: Vec<usize> = (0..n).collect();
        for i in 0..band.len() {
            let j = i + rng.random_range(0..(n - i));
            all.swap(i, j);
        }
        all.truncate(band.len());
        all
    } else {
        Vec::new()
    };
    let reserved: std::collections::HashSet<usize> = pivots.iter().copied().collect();
    // Transitive data support per already-filled band wire (pivot included).
    let mut supports: Vec<std::collections::HashSet<usize>> = Vec::new();
    for (index, &band_wire) in band.iter().enumerate() {
        let pivot = if reserve {
            pivots[index]
        } else {
            rng.random_range(0..n)
        };
        let mut support: std::collections::HashSet<usize> = std::iter::once(pivot).collect();
        out.push(XGate::cnot(band_wire, pivot as u16));
        // Small linear part: 1..=min(7, n-1) extra data wires besides the pivot
        // (and, when reserved, besides every OTHER wire's pivot — that is the
        // whole difference, and it is what makes the map triangular).
        let lin_max = (n - 1).min(7);
        let lin_w = 1 + rng.random_range(0..lin_max);
        let mut pool: Vec<usize> = (0..n)
            .filter(|&w| w != pivot && !(reserve && reserved.contains(&w)))
            .collect();
        let lin_w = lin_w.min(pool.len());
        for _ in 0..lin_w {
            let i = rng.random_range(0..pool.len());
            let w = pool.swap_remove(i);
            support.insert(w);
            out.push(XGate::cnot(band_wire, w as u16));
        }
        // Nonlinear products, cascading over pivot-free earlier band wires.
        let eligible_band: Vec<usize> = (0..index)
            .filter(|&i| !supports[i].contains(&pivot))
            .collect();
        for _ in 0..fill_nl {
            let mut draw = |exclude: Option<u16>| loop {
                let wire = if !eligible_band.is_empty() && rng.random_bool(0.5) {
                    band[eligible_band[rng.random_range(0..eligible_band.len())]]
                } else {
                    loop {
                        let w = rng.random_range(0..n);
                        if w != pivot && !(reserve && reserved.contains(&w)) {
                            break w as u16;
                        }
                    }
                };
                if Some(wire) != exclude {
                    break wire;
                }
            };
            let (s1, s2) = loop {
                let s1 = draw(None);
                let s2 = draw(Some(s1));
                if !avoid.contains(&(s1.min(s2), s1.max(s2))) {
                    break (s1, s2);
                }
            };
            let lits = [(s1, rng.random::<bool>()), (s2, rng.random::<bool>())];
            // Residual constants just complement F — balance is unaffected.
            emit_g57_form(band_wire, &lits, &mut rng, &mut out);
            for source in [s1, s2] {
                match band[..index].iter().position(|&w| w == source) {
                    Some(earlier) => {
                        let earlier_support: Vec<usize> =
                            supports[earlier].iter().copied().collect();
                        support.extend(earlier_support);
                    }
                    None => {
                        support.insert(source as usize);
                    }
                }
            }
        }
        supports.push(support);
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
fn emit_lgi_tagged(
    tag: &'static str,
    w: u16,
    cy: &[u16],
    out: &mut Vec<XGate>,
    pairs: &mut BTreeSet<(u16, u16)>,
    lin: &mut BTreeSet<u16>,
) {
    if std::env::var("EMBEDDED_MASKING_LGI_DEBUG").is_ok() && (cy.contains(&7) || cy.contains(&29))
    {
        eprintln!("[lgi] {tag} at out.len()={} w={w} cy={cy:?}", out.len());
    }
    emit_lgi(w, cy, out, pairs, lin)
}

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
/// Data roles end holding `A`'s output; the band is left dirty. Normally those
/// roles occupy wires `0..np`. Carried-layout shuffling instead returns their
/// physical locations in `final_layout`; encoded I/O additionally needs the
/// returned encoder/decoder gates.
pub fn preprocess_embedded_masking(
    src: &[XGate],
    np: usize,
    p: &EmbeddedMaskingParams,
) -> EmbeddedMaskingOutput {
    preprocess_embedded_masking_with_execution(src, np, p, &EmbeddedMaskingExecution::default())
}

/// Explicit comparison controls for retained linear-read and ancilla-pool studies.
pub fn preprocess_embedded_masking_with_execution(
    src: &[XGate],
    np: usize,
    p: &EmbeddedMaskingParams,
    execution: &EmbeddedMaskingExecution,
) -> EmbeddedMaskingOutput {
    assert!(
        true,
        "preprocessing shuffling accepts any segment count (0 = off; 1 = one transfer per original write)"
    );
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
    // the quadratic fire term touches up to 4 band wires and needs two further
    // DIRTY wires as ancillas; they may be data wires too, so `np + r` must
    // exceed the widest term (`product_fire`)
    assert!(
        !p.quad_fire || np + r >= 8,
        "quad-fire needs at least 8 wires for its dirty ancillas; got {}",
        np + r
    );
    assert!(np >= 2, "need at least two data wires");
    // No scratch wires: the masked fire borrows DIRTY band wires for its partial
    // products and restores them (see `product_fire`). A clean ancilla would be
    // 0 at every instant outside a fire — a function-level invariant that no
    // amount of mixing can hide, and whose non-zero stretches delimit the fire
    // blocks (RC 2026-09-08).
    let total = np + r;
    assert!(total < u16::MAX as usize, "too many wires");
    let band: Vec<u16> = (np as u16..(np + r) as u16).collect(); // scratch wires are NOT band
    // Reserved ancilla pool (linkable-mass lever, RC 2026-10-07): the fire's
    // dirty ancillas come only from `anc_band`, which no mask cycle or fire
    // cover pair ever uses, so every bracket restore on them is droppable --
    // each borrow then evolves the ancilla instead of re-bearing its prior
    // value (bracket restores were 84% of the recurring-value births).
    // Drawn from its own seeded stream: 0 (default) leaves the circuit as is.
    let anc_reserve: usize = std::env::var("EMBEDDED_MASKING_ANC_RESERVE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
        .min(r / 2);
    let (mask_band, anc_band): (Vec<u16>, Vec<u16>) = if anc_reserve > 0 {
        let mut prng = StdRng::seed_from_u64(p.seed ^ 0xA9C1_11A5);
        let mut all = band.clone();
        for i in (1..all.len()).rev() {
            let j = prng.random_range(0..=i);
            all.swap(i, j);
        }
        let pool: Vec<u16> = all[..anc_reserve].to_vec();
        let rest: Vec<u16> = band.iter().copied().filter(|w| !pool.contains(w)).collect();
        (rest, pool)
    } else {
        (band.clone(), band.clone())
    };
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
    let min_mask = (if p.min_mask == 0 {
        max_open
    } else {
        p.min_mask
    })
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
    let mut ready: std::collections::VecDeque<usize> = (0..m).filter(|&i| indeg[i] == 0).collect();

    // rerand SLOTS (bursts): `straddle_slots` STRADDLE (auto = m/(4k), within the
    // ~1024 thinning knee) + `rerand_repair` REPAIR (default off). Each slot is a
    // BURST of `burst` (=F=8k) band-refresh gates on ONE band wire, so total gates
    // ~ slots*F ~ 2m and the band shows a data-wire-like activity burst; straddle
    // closes the masks reading b once per slot, repair re-derives them once around
    // the whole burst. Shuffled, spread through the build.
    let burst = if p.rerand_burst > 0 {
        p.rerand_burst
    } else {
        8 * k
    };
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
    // Sparse, exact emission ledger: complete LGIs and borrowed-ancilla units
    // supply the only legal cuts. Mask snapshots include the temporary cover,
    // which intentionally is not a persistent member of `open_cy`.
    let mut shuffling_boxes: Vec<FireBox> = Vec::new();
    let mut current_box: Option<FireBox> = None;
    let mut box_cover: Vec<u16> = Vec::new();
    // Band ancillas drifted (restore dropped) inside the current fire box:
    // excluded from shuffling partner selection via every mask_support entry.
    let mut box_drifted: Vec<u16> = Vec::new();
    macro_rules! record_box_boundary {
        ($masks_changed:expr) => {
            if let Some(b) = current_box.as_mut() {
                b.unit_cuts.push(out.len());
                if $masks_changed {
                    let support: BTreeSet<u16> = open_cy[b.target as usize]
                        .iter()
                        .flatten()
                        .chain(box_cover.iter())
                        .chain(box_drifted.iter())
                        .copied()
                        .collect();
                    b.mask_support
                        .push((out.len(), support.into_iter().collect()));
                }
            }
        };
    }
    let mut rerand_done = 0usize;
    // EMBEDDED_MASKING_STRADDLE_CENSUS: per bracket borrow, how many live
    // ledger entries (open cycles / open_lin terms over all data wires)
    // contain the borrowed ancilla -- the injection count a restore-drop or
    // retirement at that point would pay (RC's compensation scheme pricing).
    let straddle_census_on = std::env::var("EMBEDDED_MASKING_STRADDLE_CENSUS").is_ok();
    let mut straddle_hist: Vec<usize> = Vec::new();
    let mut straddle_firelocal = 0usize;
    // Bracket-ancilla borrows per band wire since its last burst write:

    // rerand slots target the hottest wire (punchlist #8, hot-targeting leg)
    // so the celebrity base values are re-randomised every few borrows
    // instead of recurring ~1,000x. Bursts legally rewrite ANY band wire --
    // the repair machinery closes/reopens every referencing mask.
    let mut band_uses: Vec<usize> = vec![0usize; total];
    let diag = execution.diagnostics;
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
    let read_covers = 0usize; // reads now cover thin operands via ensure_min_open (counted there)
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
                let cy = sample_k(&mask_band, lgi_k, &mut rng);
                tries += 1;
                if avoid.is_some_and(|b| cy.iter().any(|&x| x == b)) {
                    assert!(
                        tries <= 1 << 16,
                        "embedded-masking: no LGI cycle on wire {w} avoids band wire {} (R={r}, cycle width {lgi_k})",
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
            let hits = open_cy[w]
                .iter()
                .filter(|cy| cy.iter().any(|&x| x == b))
                .count();
            let remain = open_cy[w].len() - hits;
            if hits > 0 && remain < min_open {
                // open enough replacements (away from `b`) BEFORE the closes so the
                // wire never drops below `min_open` (never below one at least)
                for _ in remain..min_open {
                    let cy = sample_fresh!(w, Some(b), &BTreeSet::new());
                    emit_lgi_tagged(
                        "site0",
                        w as u16,
                        &cy,
                        &mut out,
                        &mut open_pairs[w],
                        &mut open_lin[w],
                    );
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
                let clash =
                    m1.contains(&c2) || m2.contains(&c1) || m1.iter().any(|x| m2.contains(x));
                tries += 1;
                if !clash || tries > 256 {
                    break g;
                }
            }
        }};
    }
    macro_rules! emit_slot {
        () => {{
            assert!(
                current_box.is_none(),
                "band refresh cannot occur inside a shuffling box"
            );
            let is_repair = slot_plan[si];
            si += 1;
            slots_emitted += 1;
            if in_drain {
                drain_slots += 1;
            }
            let b = {
                let hottest = band
                    .iter()
                    .map(|&w| band_uses[w as usize])
                    .max()
                    .unwrap_or(0);
                let pool: Vec<u16> = band
                    .iter()
                    .copied()
                    .filter(|&w| band_uses[w as usize] == hottest)
                    .collect();
                pool[rng.random_range(0..pool.len())]
            };
            band_uses[b as usize] = 0;
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
                    out.push(burst_gate_safe!(
                        b,
                        if p.burst_band_only { 0 } else { active }
                    ));
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
                            emit_lgi_tagged(
                                "site1",
                                w as u16,
                                &cy,
                                &mut out,
                                &mut open_pairs[w],
                                &mut open_lin[w],
                            );
                        } else {
                            idx += 1;
                        }
                    }
                }
                for _ in 0..burst {
                    out.push(burst_gate_safe!(
                        b,
                        if p.burst_band_only { 0 } else { active }
                    ));
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
    // Threshold retirement (punchlist #8, RC 2026-10-07): any band wire
    // borrowed >= retire_k times since its last write gets its value retired
    // at the next box boundary via a mini-slot -- early-close the open
    // cycles containing it (the close is MOVED, not added; min_open
    // machinery reopens coverage), then one fresh nonlinear write. Caps
    // every bracket-restore chain near the recurrence-law threshold without
    // any compensation gates, by reusing the hot-targeting slot mechanics
    // at per-wire cadence instead of once per rerand slot.
    let chain_k: usize = std::env::var("EMBEDDED_MASKING_CHAIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut chained_brackets = 0usize;
    let chain_rotate = std::env::var("EMBEDDED_MASKING_CHAIN_ROTATE").is_ok_and(|v| v != "0");
    let chain_len: usize = std::env::var("EMBEDDED_MASKING_CHAIN_LEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let retire_k: usize = std::env::var("EMBEDDED_MASKING_RETIRE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    // Input-fringe guard (ret32w probe): no retirement during the first
    // N% of the compute, so the band accumulates nonlinear depth before
    // churn begins -- the early-C t*=2..8 leak in the delivered artifact
    // tracked retirement material in the input fringe.
    let retire_warmup_pct: usize = std::env::var("EMBEDDED_MASKING_RETIRE_WARMUP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    // A retirement's new band value differs from the old by its write
    // gates alone; with one gate whose two band controls are some data
    // wire's open mask pair, old ^ new IS that mask product (up to affine
    // terms), and the long-lived old value's copies make it strippable at a
    // boundary. SAFE redraws such gates; BURST writes several products so
    // old ^ new is never a single mask term.
    let retire_safe = std::env::var("EMBEDDED_MASKING_RETIRE_SAFE").map_or(true, |v| v != "0");
    let retire_burst: usize = std::env::var("EMBEDDED_MASKING_RETIRE_BURST")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
        .max(1);
    let retire_repair = std::env::var("EMBEDDED_MASKING_RETIRE_MODE").is_ok_and(|v| v == "repair");
    // Pinned-coverage gate: writes to one data wire commute, so between two
    // reads of `w` the mixer may reorder all of its mask opens and closes.
    // Only masks opened BEFORE `w`'s last read survive every such reordering
    // up to the next read. Retire `b` only when each data wire whose open
    // cycle reads `b` keeps `min_open` pinned cycles without it; otherwise
    // defer. Tracked by an incremental scan of `out`: last read per data
    // wire, last write position per (wire, mask pair) gate.
    let retire_pin_need: usize = std::env::var("EMBEDDED_MASKING_RETIRE_PIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let retire_pin = retire_pin_need > 0;
    // Count-only variant: no replacement opens at retirement (so no open
    // sits next to a close on the same wire); defer while any affected wire
    // would drop below `min_open` open cycles.
    let retire_norep = std::env::var("EMBEDDED_MASKING_RETIRE_NOREPLACE").is_ok_and(|v| v != "0");
    // Early replacement: once a band wire reaches PREOPEN% of the threshold,
    // open the replacement cycles its retirement will need on every affected
    // data wire NOW, so the eventual close is never adjacent to an open on
    // the same wire (a local rewrite could otherwise merge the two and expose
    // an under-masked intermediate). Same opens as at retirement, moved
    // earlier: no added gates.
    let retire_preopen_pct: usize = std::env::var("EMBEDDED_MASKING_RETIRE_PREOPEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut retire_preopens = 0usize;
    let mut pin_scan = 0usize;
    let mut pin_last_read: Vec<Option<usize>> = vec![None; np];
    let mut pin_pair_pos: std::collections::HashMap<(u16, u16, u16), usize> =
        std::collections::HashMap::new();
    let mut retire_deferrals = 0usize;
    let mut retire_redraws = 0usize;
    let mut retired = 0usize;
    macro_rules! retire_gate {
        ($b:expr) => {{
            let b: u16 = $b;
            let mut tries = 0usize;
            loop {
                let g = burst_gate_safe!(b, if p.burst_band_only { 0 } else { active });
                let (c1, c2) = (g.ctrls[0].0, g.ctrls[1].0);
                let mask_pair = retire_safe
                    && (c1 as usize) >= np
                    && (c2 as usize) >= np
                    && (0..np).any(|w| {
                        open_pairs[w].contains(&(c1, c2)) || open_pairs[w].contains(&(c2, c1))
                    });
                tries += 1;
                if !mask_pair || tries > 256 {
                    break g;
                }
                retire_redraws += 1;
            }
        }};
    }
    macro_rules! retire_sweep {
        () => {{
            if retire_k > 0 {
                if retire_pin {
                    while pin_scan < out.len() {
                        let g = &out[pin_scan];
                        for &(c, _) in g.ctrls.iter() {
                            if (c as usize) < np {
                                pin_last_read[c as usize] = Some(pin_scan);
                            }
                        }
                        if (g.target as usize) < np {
                            let key = match g.ctrls.len() {
                                1 => Some((g.target, g.ctrls[0].0, u16::MAX)),
                                2 => {
                                    let (a, c) = (g.ctrls[0].0, g.ctrls[1].0);
                                    Some((g.target, a.min(c), a.max(c)))
                                }
                                _ => None,
                            };
                            if let Some(key) = key {
                                pin_pair_pos.insert(key, pin_scan);
                            }
                        }
                        pin_scan += 1;
                    }
                }
                if retire_preopen_pct > 0 {
                    let pre_at = (retire_k * retire_preopen_pct / 100).max(1);
                    let near: Vec<u16> = band
                        .iter()
                        .copied()
                        .filter(|&w| {
                            let u = band_uses[w as usize];
                            u >= pre_at && u < retire_k
                        })
                        .collect();
                    for b in near {
                        for w in 0..np {
                            let before = open_cy[w].len();
                            keep_covered!(w, b);
                            retire_preopens += open_cy[w].len() - before;
                        }
                    }
                }
                let due: Vec<u16> = band
                    .iter()
                    .copied()
                    .filter(|&w| band_uses[w as usize] >= retire_k)
                    .collect();
                for b in due {
                    if retire_pin {
                        let pinned = |w: usize, cy: &[u16]| -> bool {
                            let Some(read) = pin_last_read[w] else {
                                return false;
                            };
                            let mut keys: Vec<(u16, u16, u16)> = (0..cy.len() / 2)
                                .map(|i| {
                                    let (a, c) = (cy[2 * i], cy[2 * i + 1]);
                                    (w as u16, a.min(c), a.max(c))
                                })
                                .collect();
                            if cy.len() % 2 == 1 {
                                keys.push((w as u16, cy[cy.len() - 1], u16::MAX));
                            }
                            keys.iter()
                                .all(|key| pin_pair_pos.get(key).is_none_or(|&at| at < read))
                        };
                        let blocked = (0..np).any(|w| {
                            open_cy[w].iter().any(|cy| cy.contains(&b))
                                && open_cy[w]
                                    .iter()
                                    .filter(|cy| !cy.contains(&b) && pinned(w, cy))
                                    .count()
                                    < retire_pin_need
                        });
                        if blocked {
                            retire_deferrals += 1;
                            continue;
                        }
                    }
                    if retire_norep
                        && (0..np).any(|w| {
                            let hits = open_cy[w].iter().filter(|cy| cy.contains(&b)).count();
                            hits > 0 && open_cy[w].len() - hits < min_open
                        })
                    {
                        retire_deferrals += 1;
                        continue;
                    }
                    band_uses[b as usize] = 0;
                    if retire_repair {
                        // Repair-slot form: masks reading `b` stay open and are
                        // re-keyed across the write (each LGI's term containing
                        // `b` is applied before and after), so no cycle is
                        // closed early.
                        for w in 0..np {
                            keep_covered!(w, b);
                            for cy in &open_cy[w] {
                                if let Some(gg) = b_g57(w as u16, cy, b) {
                                    out.push(gg);
                                }
                            }
                        }
                        for _ in 0..retire_burst {
                            out.push(retire_gate!(b));
                        }
                        for w in 0..np {
                            for cy in &open_cy[w] {
                                if let Some(gg) = b_g57(w as u16, cy, b) {
                                    out.push(gg);
                                }
                            }
                        }
                        retired += 1;
                        continue;
                    }
                    for w in 0..np {
                        keep_covered!(w, b);
                        let mut idx = 0;
                        while idx < open_cy[w].len() {
                            if open_cy[w][idx].iter().any(|&x| x == b) {
                                let cy = open_cy[w].remove(idx);
                                emit_lgi_tagged(
                                    "site10",
                                    w as u16,
                                    &cy,
                                    &mut out,
                                    &mut open_pairs[w],
                                    &mut open_lin[w],
                                );
                            } else {
                                idx += 1;
                            }
                        }
                    }
                    for _ in 0..retire_burst {
                        out.push(retire_gate!(b));
                    }
                    retired += 1;
                }
            }
        }};
    }

    // After an open on wire `w`, bring it up to `min_open` open masks (extra to
    // the u_w+1 budget, counted in `min_open_opens`); `$extra` = band wires to
    // stay away from (the fire's wires when called mid-fire).
    let mut min_open_opens = 0usize;
    let mut qq_blinded = 0usize; // fire ancillas blinded away from an open pair
    let mut fire_census = [0usize; 8]; // per-term-class gate counts (EMBEDDED_MASKING_DIAG)
    macro_rules! ensure_min_open {
        ($w:expr, $extra:expr) => {{
            let w: usize = $w;
            while open_cy[w].len() < min_open {
                let cy = sample_fresh!(w, None, $extra);
                emit_lgi_tagged(
                    "site2",
                    w as u16,
                    &cy,
                    &mut out,
                    &mut open_pairs[w],
                    &mut open_lin[w],
                );
                open_cy[w].push(cy);
                min_open_opens += 1;
                record_box_boundary!(true);
            }
        }};
    }
    macro_rules! filler_open {
        ($w:expr) => {{
            let w = $w;
            if std::env::var("EMBEDDED_MASKING_LGI_DEBUG").is_ok() {
                eprintln!("[trace] filler w={w} at out.len()={}", out.len());
            }
            if open_cy[w].len() >= max_open {
                let cy = open_cy[w].remove(0);
                emit_lgi_tagged(
                    "site3",
                    w as u16,
                    &cy,
                    &mut out,
                    &mut open_pairs[w],
                    &mut open_lin[w],
                );
            }
            let cy = sample_fresh!(w, None, &BTreeSet::new());
            emit_lgi_tagged(
                "site4",
                w as u16,
                &cy,
                &mut out,
                &mut open_pairs[w],
                &mut open_lin[w],
            );
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
                emit_lgi_tagged(
                    "site5",
                    w as u16,
                    &cy,
                    &mut pre,
                    &mut open_pairs[w],
                    &mut open_lin[w],
                );
                open_cy[w].push(cy);
            }
        }
    }
    // Up-front masking (fringe fix 1): otherwise a data wire is first masked
    // only when first used, so raw input copies survive deep into the compute
    // and, with a reordered fire ancilla holding a bare operand product,
    // recombine into a first-layer C value at a delivered boundary.
    let premask = std::env::var("EMBEDDED_MASKING_PREMASK").is_ok_and(|v| v != "0");
    if premask && !p.encoded_io {
        let mut order: Vec<usize> = (0..np).collect();
        for i in (1..order.len()).rev() {
            let j = rng.random_range(0..=i);
            order.swap(i, j);
        }
        for w in order {
            while open_cy[w].len() < min_open {
                let cy = sample_fresh!(w, None, &BTreeSet::new());
                emit_lgi_tagged(
                    "site11",
                    w as u16,
                    &cy,
                    &mut out,
                    &mut open_pairs[w],
                    &mut open_lin[w],
                );
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
        let mut operands: Vec<Operand> = Vec::new();
        let mut undo: Vec<XGate> = Vec::new();
        if std::env::var("EMBEDDED_MASKING_LGI_DEBUG").is_ok() {
            eprintln!(
                "[trace] fire gi={gi} target={} at out.len()={}",
                src[gi].target,
                out.len()
            );
        }
        for &(w, pol) in &src[gi].ctrls {
            if p.quad_fire {
                // Masked fire: the operand is only READ, under whatever masks it
                // carries. A thin operand (its first read, or a seldom-used wire)
                // is first brought up to `min_open` real masks — the read-cover
                // rule — so no wire is ever read bare.
                ensure_min_open!(w as usize, &BTreeSet::new());
                if diag {
                    n_reads += 1;
                    let q = open_cy[w as usize].len();
                    if q == 0 {
                        n_bare += 1;
                    }
                    rho_min = rho_min.min(2 * q);
                    rho_sum += 2 * q;
                }
                operands.push(Operand {
                    w,
                    pol,
                    cycles: open_cy[w as usize].clone(),
                });
                continue;
            }
            let netopen: Vec<(u16, u16)> = open_pairs[w as usize].iter().copied().collect();
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
                    let r1 = mask_band[rng.random_range(0..mask_band.len())];
                    let mut r2 = mask_band[rng.random_range(0..mask_band.len())];
                    while r2 == r1 {
                        r2 = mask_band[rng.random_range(0..mask_band.len())];
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
                let mut r1 = mask_band[rng.random_range(0..mask_band.len())];
                while rho.contains(&r1) {
                    r1 = mask_band[rng.random_range(0..mask_band.len())];
                }
                let mut r2 = mask_band[rng.random_range(0..mask_band.len())];
                while r2 == r1 || rho.contains(&r2) {
                    r2 = mask_band[rng.random_range(0..mask_band.len())];
                }
                out.push(g57(w, r1, r2));
                out.push(g57(w, r2, r1));
                undo.push(g57(w, r2, r1));
                undo.push(g57(w, r1, r2));
                rho.push(r1);
                rho.push(r2);
                rho.sort_unstable();
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
        // The fire as commuting UNITS (masked fire) or single monomials (legacy).
        let mut fire_droppable: Vec<(usize, u16)> = Vec::new();
        let mut fire_support: BTreeSet<u16> = BTreeSet::new();
        let mut units: Vec<Vec<XGate>> = if p.quad_fire {
            // pairs open on ANY data wire right now: an ancilla must never hold
            // one of them (see `product_fire`)
            let open_pair_set: BTreeSet<(u16, u16)> = open_cy
                .iter()
                .flatten()
                .flat_map(|cy| (0..cy.len() / 2).map(move |i| norm_pair(cy[2 * i], cy[2 * i + 1])))
                .collect();
            // Every band wire referenced by ANY currently-open mask (cycles
            // incl. balancing terms, plus the balanced linear parts): bracket
            // ancillas inside this set must be restored exactly.
            let open_support: BTreeSet<u16> = open_cy
                .iter()
                .flatten()
                .flatten()
                .copied()
                .chain(open_lin.iter().flat_map(|l| l.iter().copied()))
                .collect();
            let (fire_units, droppable) = product_fire(
                src[gi].target,
                &operands,
                src[gi].comp,
                &open_pair_set,
                &open_pairs,
                &anc_band,
                np,
                r,
                &mut rng,
                &mut qq_blinded,
                &mut fire_census,
                execution.ancilla_band_only,
            );
            fire_droppable = droppable;
            fire_support = open_support;
            fire_units
        } else {
            let mut fires = Vec::new();
            masked_fire(src[gi].target, &ctrls, src[gi].comp, &mut fires);
            fires.into_iter().map(|g| vec![g]).collect()
        };
        if p.shuffling_segments != 0 {
            box_drifted.clear();
            let support: BTreeSet<u16> = open_cy[c].iter().flatten().copied().collect();
            current_box = Some(FireBox {
                target: c as u16,
                start: out.len(),
                end: out.len(),
                unit_cuts: vec![out.len()],
                mask_support: vec![(out.len(), support.into_iter().collect())],
            });
        }
        if open_cy[c].len() >= max_open {
            let cy = open_cy[c].remove(0);
            emit_lgi_tagged(
                "site6",
                c as u16,
                &cy,
                &mut out,
                &mut open_pairs[c],
                &mut open_lin[c],
            );
            record_box_boundary!(true);
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
        // (~4 gates per fire, no change to any read polynomial). The mid-fire
        // straddle open is drawn away from those wires as well.
        let mut fire_wires: BTreeSet<u16> = if p.quad_fire {
            operands
                .iter()
                .flat_map(|o| o.cycles.iter().flatten().copied())
                .collect()
        } else {
            ctrls
                .iter()
                .flat_map(|(_, _, rho)| rho.iter().copied())
                .collect()
        };
        let cover = sample_fresh!(c, None, &fire_wires);
        emit_lgi_tagged(
            "site7",
            c as u16,
            &cover,
            &mut out,
            &mut open_pairs[c],
            &mut open_lin[c],
        );
        if current_box.is_some() {
            box_cover.clone_from(&cover);
        }
        record_box_boundary!(true);
        fire_wires.extend(cover.iter().copied());
        fire_covers += 1;
        // Draw the mid-fire straddle BEFORE emitting any unit so the full
        // stability set (open masks + cover + straddle) is known for the
        // restore-drop pass below. Same avoid set as before, so the draw
        // distribution is unchanged.
        let cy = sample_fresh!(c, None, &fire_wires);
        // The mid-fire ensure_min_open on the TARGET (after the straddle,
        // before the second unit half) also opens cycles whose wires must
        // not drift -- and it draws AFTER this pass. Pre-draw those cycles
        // here (same avoid set, same count) so the stability set is
        // complete; they are emitted at their original position below.
        let midfire_opens: Vec<Vec<u16>> = {
            let have = open_cy[c].len() + 1; // straddle joins before the ensure
            (have..min_open.max(have))
                .map(|_| sample_fresh!(c, None, &fire_wires))
                .collect()
        };
        // Chained brackets (RC 2026-10-07): a bracket's first gates add its
        // product to the target WHATEVER value the ancilla holds, so the
        // fire's brackets can share `chain_k` ancillas with no restore in
        // between -- each ancilla accumulates its terms, and all restores run
        // once, after the last unit, in emission order (the ancilla passes
        // through fresh partial sums; its starting value comes back once per
        // fire instead of once per bracket). Ancillas are drawn from band
        // wires that no other gate of this fire reads or writes (units,
        // cover, straddle, mid-fire opens, cover-pair undo), so every reader
        // outside the brackets sees them unchanged; restored before the
        // cover closes. Chained brackets never drift.
        let mut chain_restore: Vec<Option<XGate>> = vec![None; units.len()];
        let mut chain_ancillas: Vec<u16> = Vec::new();
        let mut seg_tail = 0usize;
        if chain_k > 0 && !fire_droppable.is_empty() {
            let bracket_ancillas = |ui: usize, sa: u16| -> Vec<u16> {
                let u = &units[ui];
                if u.len() == 4 {
                    vec![u[1].target]
                } else {
                    let fa = if u[1].target != sa {
                        u[1].target
                    } else {
                        u[3].target
                    };
                    vec![fa, sa]
                }
            };
            let anc_of: HashMap<usize, Vec<u16>> = fire_droppable
                .iter()
                .map(|&(ui, sa)| (ui, bracket_ancillas(ui, sa)))
                .collect();
            let mut reads: BTreeSet<u16> = BTreeSet::new();
            reads.insert(c as u16);
            for (ui, unit) in units.iter().enumerate() {
                let own: &[u16] = anc_of.get(&ui).map_or(&[], |v| v.as_slice());
                for g in unit {
                    if !own.contains(&g.target) {
                        reads.insert(g.target);
                    }
                    for &(w, _) in &g.ctrls {
                        if !own.contains(&w) {
                            reads.insert(w);
                        }
                    }
                }
            }
            reads.extend(fire_wires.iter().copied());
            reads.extend(cy.iter().copied());
            reads.extend(midfire_opens.iter().flatten().copied());
            for g in &undo {
                reads.insert(g.target);
                reads.extend(g.ctrls.iter().map(|&(w, _)| w));
            }
            let mut cands: Vec<u16> = band
                .iter()
                .copied()
                .filter(|w| !reads.contains(w))
                .collect();
            if cands.len() >= 2 {
                for i in (1..cands.len()).rev() {
                    let j = rng.random_range(0..=i);
                    cands.swap(i, j);
                }
                if chain_rotate && chain_len > 0 {
                    // Rotating segments: every L brackets form one ATOMIC unit
                    // on the next pair of fire-safe ancillas, closed by its own
                    // restores -- displacement never leaves the segment, and
                    // each band value serves (and is re-born) at most once per
                    // segment, spread over every free band wire.
                    let bracket_idx: Vec<usize> =
                        fire_droppable.iter().map(|&(ui, _)| ui).collect();
                    let mut layout: Vec<u16> = (0..total as u16).collect();
                    let mut segs: Vec<Vec<XGate>> = Vec::new();
                    let mut next = 0usize;
                    for chunk in bracket_idx.chunks(chain_len) {
                        let a1 = cands[next % cands.len()];
                        let a2 = cands[(next + 1) % cands.len()];
                        next += 2;
                        let mut body: Vec<XGate> = Vec::new();
                        let mut rest: Vec<XGate> = Vec::new();
                        let mut used2 = false;
                        for &ui in chunk {
                            let ancs = &anc_of[&ui];
                            for (k, &h) in ancs.iter().enumerate() {
                                layout[h as usize] = if k == 0 { a1 } else { a2 };
                            }
                            used2 |= ancs.len() > 1;
                            let mut mapped: Vec<XGate> =
                                units[ui].iter().map(|g| remap_gate(g, &layout)).collect();
                            for &h in ancs {
                                layout[h as usize] = h;
                            }
                            rest.push(mapped.pop().expect("bracket restore"));
                            body.extend(mapped);
                        }
                        body.extend(rest);
                        segs.push(body);
                        chain_ancillas.push(a1);
                        if used2 {
                            chain_ancillas.push(a2);
                        }
                    }
                    let bset: HashSet<usize> = bracket_idx.into_iter().collect();
                    let mut kept: Vec<Vec<XGate>> = std::mem::take(&mut units)
                        .into_iter()
                        .enumerate()
                        .filter(|(i, _)| !bset.contains(i))
                        .map(|(_, u)| u)
                        .collect();
                    seg_tail = segs.len();
                    kept.extend(segs);
                    units = kept;
                    chain_restore = vec![None; units.len()];
                    chained_brackets += fire_droppable.len();
                    fire_droppable.clear();
                } else {
                    cands.truncate(chain_k.max(2));
                    let mut layout: Vec<u16> = (0..total as u16).collect();
                    for &(ui, _) in &fire_droppable {
                        let ancs = &anc_of[&ui];
                        let first = rng.random_range(0..cands.len());
                        for (k, &h) in ancs.iter().enumerate() {
                            layout[h as usize] = cands[(first + k) % cands.len()];
                        }
                        let mapped: Vec<XGate> =
                            units[ui].iter().map(|g| remap_gate(g, &layout)).collect();
                        for &h in ancs {
                            layout[h as usize] = h;
                        }
                        units[ui] = mapped;
                        chain_restore[ui] = units[ui].pop();
                    }
                    chained_brackets += fire_droppable.len();
                    fire_droppable.clear();
                    chain_ancillas = cands;
                }
            }
        }
        // Restore-drop pass (RC 2026-10-06): strip the final ancilla restore
        // of every bracket whose band ancilla is OUTSIDE the stability set.
        // The ancilla then drifts by the bracket's product instead of
        // re-materialising its prior value (the celebrity-value surface);
        // consecutive borrows differ by data QUADRATICS, never linear
        // relations. Drifted wires are appended to the fire box's mask
        // support so the shuffling pass never uses them as transfer
        // partners (a pending transfer mask must stay constant to its
        // drain).
        // A chained ancilla is re-born once per fire however many brackets it
        // serves: count it as one borrow (2 toggles, matching a restored
        // bracket) rather than per toggle.
        let seg_from = units.len() - seg_tail;
        for (ui, unit) in units.iter().enumerate() {
            if chain_restore[ui].is_some() || ui >= seg_from {
                continue;
            }
            for g in unit {
                if (g.target as usize) >= np {
                    band_uses[g.target as usize] += 1;
                }
            }
        }
        for &a in &chain_ancillas {
            band_uses[a as usize] += 2;
        }
        // The half-shuffle below relies on units commuting, which holds only
        // while every bracket restores its ancilla: two units touching the
        // SAME ancilla stop commuting once one of them drifts. A restore may
        // therefore be dropped only when its ancilla appears in no other
        // unit of this fire.
        let mut band_unit_of: HashMap<u16, Option<usize>> = HashMap::new();
        for (ui, unit) in units.iter().enumerate() {
            for g in unit {
                let mut touch = |wire: u16| {
                    if (wire as usize) >= np {
                        band_unit_of
                            .entry(wire)
                            .and_modify(|e| {
                                if *e != Some(ui) {
                                    *e = None; // shared across units
                                }
                            })
                            .or_insert(Some(ui));
                    }
                };
                touch(g.target);
                for &(w, _) in &g.ctrls {
                    touch(w);
                }
            }
        }
        let mut drifted: Vec<u16> = Vec::new();
        // The stability set must be LIVE at drop time: the target's own
        // ensure_min_open / overflow bookkeeping runs between the
        // product_fire snapshot and this pass (that stale snapshot was the
        // 2026-10-06 leak: a cycle opened on the target right after the
        // snapshot straddled a drift). Union with the snapshot anyway.
        let mut live_support: BTreeSet<u16> = open_cy
            .iter()
            .flatten()
            .flatten()
            .copied()
            .chain(open_lin.iter().flat_map(|l| l.iter().copied()))
            .collect();
        live_support.extend(fire_support.iter().copied());
        // DEFAULT ON (punchlist #8, RC sign-off pending 2026-10-07): the
        // ancilla-restore drop kills the celebrity-value rebirth surface.
        // Stability layers, each required (violations found empirically):
        // live open-mask support incl. open_lin, the fire cover, the
        // straddle (drawn early), the PRE-DRAWN mid-fire min-open cycles,
        // per-unit ancilla uniqueness (shuffled units stop commuting if a
        // drifted ancilla is shared), and in-box-written partner exclusion
        // in the shuffling pass. EMBEDDED_MASKING_DRIFT=0 restores the
        // legacy always-restore behaviour.
        let drift_enabled = std::env::var("EMBEDDED_MASKING_DRIFT").map_or(true, |v| v != "0")
            && std::env::var("EMBEDDED_MASKING_DRIFT_ONLY").map_or(true, |w| {
                w.split_once(':').is_some_and(|(lo, hi)| {
                    lo.parse::<usize>().is_ok_and(|l| gi >= l)
                        && hi.parse::<usize>().is_ok_and(|h| gi < h)
                })
            });
        if straddle_census_on {
            for &(_, h) in &fire_droppable {
                let injections = open_cy
                    .iter()
                    .flatten()
                    .filter(|cycle| cycle.contains(&h))
                    .count()
                    + open_lin.iter().filter(|l| l.contains(&h)).count();
                straddle_hist.push(injections);
                if cover.contains(&h)
                    || cy.contains(&h)
                    || midfire_opens.iter().flatten().any(|&mw| mw == h)
                {
                    straddle_firelocal += 1;
                }
            }
        }
        for &(ui, h) in &fire_droppable {
            if !drift_enabled {
                break;
            }

            if !live_support.contains(&h)
                && !cover.contains(&h)
                && !cy.contains(&h)
                && !midfire_opens.iter().flatten().any(|&mw| mw == h)
                && band_unit_of.get(&h) == Some(&Some(ui))
            {
                units[ui].pop();
                drifted.push(h);
            }
        }
        if std::env::var("EMBEDDED_MASKING_DRIFT_DEBUG").is_ok() && !drifted.is_empty() {
            eprintln!(
                "[drift] fire gi={gi} c={c} drifted={drifted:?} support={fire_support:?} cover={cover:?} straddle={cy:?} open_cy[c]={:?}",
                open_cy[c]
            );
        }
        box_drifted = drifted.clone();
        if let Some(fire_box) = current_box.as_mut() {
            if !drifted.is_empty() {
                for (_, support) in fire_box.mask_support.iter_mut() {
                    support.extend(drifted.iter().copied());
                    support.sort_unstable();
                    support.dedup();
                }
            }
        }
        // straddle: fire-part-1, OPEN a fresh LGI on c (mid-fire), fire-part-2 --
        // each half emitted in an independently random order (units commute).
        let cut = ((units.len() + 1) / 2).min(units.len());
        // Shuffle each half with its deferred restores attached (same draw
        // sequence as `shuffle_units`), so restores follow emission order.
        let mut paired: Vec<(Vec<XGate>, Option<XGate>)> = std::mem::take(&mut units)
            .into_iter()
            .zip(chain_restore)
            .collect();
        for range in [0..cut, cut..paired.len()] {
            let half = &mut paired[range];
            for i in (1..half.len()).rev() {
                let j = rng.random_range(0..=i);
                half.swap(i, j);
            }
        }
        // Deferred restores, queued per ancilla (the restore's target). With
        // CHAIN_LEN = L > 0 an ancilla's queue is flushed after every L of its
        // brackets (bounded displacement); otherwise once, after the last unit.
        let mut pending_restores: Vec<(u16, Vec<XGate>)> = Vec::new();
        macro_rules! queue_restore {
            ($r:expr) => {{
                if let Some(g) = $r {
                    let a = g.target;
                    let slot = match pending_restores.iter().position(|(w, _)| *w == a) {
                        Some(i) => i,
                        None => {
                            pending_restores.push((a, Vec::new()));
                            pending_restores.len() - 1
                        }
                    };
                    pending_restores[slot].1.push(g.clone());
                    if chain_len > 0 && pending_restores[slot].1.len() >= chain_len {
                        let (_, gs) = pending_restores.remove(slot);
                        out.extend(gs);
                    }
                }
            }};
        }
        for (u, r) in &paired[..cut] {
            out.extend_from_slice(u);
            queue_restore!(r.as_ref());
            record_box_boundary!(false);
        }
        emit_lgi_tagged(
            "site8",
            c as u16,
            &cy,
            &mut out,
            &mut open_pairs[c],
            &mut open_lin[c],
        );
        open_cy[c].push(cy);
        record_box_boundary!(true);
        for mcy in &midfire_opens {
            emit_lgi_tagged(
                "site2-mid",
                c as u16,
                mcy,
                &mut out,
                &mut open_pairs[c],
                &mut open_lin[c],
            );
            open_cy[c].push(mcy.clone());
            min_open_opens += 1;
            record_box_boundary!(true);
        }
        for (u, r) in &paired[cut..] {
            out.extend_from_slice(u);
            queue_restore!(r.as_ref());
            record_box_boundary!(false);
        }
        if !pending_restores.is_empty() {
            for (_, gs) in pending_restores.drain(..) {
                out.extend(gs);
            }
            record_box_boundary!(false);
        }
        // close the bracket (toggles its pairs/linear term back off)
        emit_lgi_tagged(
            "site9",
            c as u16,
            &cover,
            &mut out,
            &mut open_pairs[c],
            &mut open_lin[c],
        );
        box_cover.clear();
        record_box_boundary!(true);
        for gg in undo {
            out.push(gg);
        }
        record_box_boundary!(false);
        if let Some(mut b) = current_box.take() {
            b.end = out.len();
            b.unit_cuts.dedup();
            shuffling_boxes.push(b);
        }
        straddles_left[c] -= 1;
        placed += 1;
        if placed * 100 >= retire_warmup_pct * m {
            retire_sweep!();
        }
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
            "[embedded-masking] WARN: {} of {} rerand slots undelivered (requested slots > {} \
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
            "[embedded-masking-diag] A-gates={m} straddled(firing hidden)={m} unplaced=0  control reads={n_reads}  \
             BARE (rho empty)={n_bare} ({:.3}%)  min |rho|={} (floor {min_mask})  mean |rho|={:.2}",
            100.0 * n_bare as f64 / n_reads.max(1) as f64,
            if n_reads == 0 { 0 } else { rho_min },
            rho_sum as f64 / n_reads.max(1) as f64
        );
        eprintln!(
            "[embedded-masking-diag] rerand slots: emitted={slots_emitted} of plan={}  \
             DRAIN (after last A-gate)={drain_slots} ({:.1}%)  main-loop={}  \
             cover-replacement LGIs={replacements}  read-cover LGIs={read_covers}  \
             fire-cover LGIs={fire_covers}  min-open opens={min_open_opens}  \
             blinded fire ancillas={qq_blinded}  relaxed (non-disjoint) samples={relaxed}",
            slot_plan.len(),
            100.0 * drain_slots as f64 / slots_emitted.max(1) as f64,
            slots_emitted - drain_slots
        );
    }

    if diag {
        let names = [
            "a'b'                       ",
            "wire x linear              ",
            "linear x linear            ",
            "constant x everything      ",
            "wire x quadratic  (4-gate) ",
            "linear x quadratic(4-gate) ",
            "quadratic^2       (8-gate) ",
            "constant flip              ",
        ];
        let tot: usize = fire_census.iter().sum();
        eprintln!("[embedded-masking-fire] per-fire gate census over {m} fires (total {tot}):");
        for (n, c) in names.iter().zip(fire_census.iter()) {
            eprintln!(
                "[embedded-masking-fire]   {n} {c:>10}  {:>6.1} per fire  {:>5.1}%",
                *c as f64 / m as f64,
                100.0 * *c as f64 / tot as f64
            );
        }
    }
    // Mask tracking is defined in logical roles. After transfers a data role
    // can live on a band-index wire, so inspecting the transformed stream would
    // miss its opens and misinterpret its closes.
    let role_hot_intervals = if execution.record_hot_intervals {
        hot_intervals(&out, np, r)
    } else {
        Vec::new()
    };
    // Include the entire drain in the transform. Even with data returned home,
    // later bursts, fillers and decoder gates must map their BAND controls.
    let (out, final_layout, shuffling_stats) = if p.shuffling_segments == 0 {
        let stats = ShufflingStats {
            original_gates: out.len(),
            shuffled_gates: out.len(),
            ..ShufflingStats::default()
        };
        (out, (0..total as u16).collect(), stats)
    } else {
        let transformed = apply_shuffling(
            &out,
            &shuffling_boxes,
            np,
            total,
            p.shuffling_segments,
            p.shuffling_return_home,
            p.shuffling_jit_gap,
            p.seed,
        );
        post = post
            .iter()
            .map(|gate| remap_gate(gate, &transformed.final_layout))
            .collect();
        (
            transformed.gates,
            transformed.final_layout,
            transformed.stats,
        )
    };
    if chain_k > 0 {
        eprintln!("[chain] k={chain_k}: {chained_brackets} brackets chained");
    }
    if retire_k > 0 {
        eprintln!(
            "[retire] threshold={retire_k} warmup={retire_warmup_pct}% safe={retire_safe} burst={retire_burst} repair={retire_repair} pin={retire_pin_need} norep={retire_norep} preopen={retire_preopen_pct}%: {retired} band-wire retirements, {retire_redraws} mask-pair redraws, {retire_deferrals} pin deferrals, {retire_preopens} early opens"
        );
    }
    if straddle_census_on && !straddle_hist.is_empty() {
        straddle_hist.sort_unstable();
        let n = straddle_hist.len();
        let total: usize = straddle_hist.iter().sum();
        let zero = straddle_hist.iter().filter(|&&x| x == 0).count();
        eprintln!(
            "[straddle-census] borrows={} total injections={} (avg {:.2}) zero={} ({:.0}%) med={} p90={} max={} fire-local-pinned={} ({:.0}%)",
            n,
            total,
            total as f64 / n as f64,
            zero,
            100.0 * zero as f64 / n as f64,
            straddle_hist[n / 2],
            straddle_hist[(9 * n) / 10],
            straddle_hist[n - 1],
            straddle_firelocal,
            100.0 * straddle_firelocal as f64 / n as f64,
        );
    }
    if let Ok(path) = std::env::var("EMBEDDED_MASKING_DUMP_GATES") {
        use std::io::Write;
        let mut body = format!(
            "=== call np={np} r={r} segments={} seed={}\n",
            p.shuffling_segments, p.seed
        );
        for (tag, gs) in [("pre", &pre), ("main", &out), ("post", &post)] {
            for g in gs.iter() {
                body.push_str(&format!(
                    "{tag} {} {} {:?}\n",
                    g.target, g.comp as u8, g.ctrls
                ));
            }
        }
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let _ = f.write_all(body.as_bytes());
        }
    }
    EmbeddedMaskingOutput {
        gates: out,
        num_wires: total,
        atoms,
        rerand_done,
        pre_gates: pre,
        post_gates: post,
        scratch_wires: 0,
        r_used: r,
        final_layout,
        shuffling_stats,
        role_hot_intervals,
        #[cfg(test)]
        shuffling_boxes,
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/preprocessing/embedded_masking.rs"]
mod tests;
