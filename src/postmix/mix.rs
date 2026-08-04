// The post-fsplit mixing chain (fmix): a randomized, size-thermostatted random
// walk over equivalent circuits in the XGate set. The objective is NOT size —
// it is to keep churning the circuit away from its original description while a
// thermostat holds the gate count near a target.
//
// Expansion moves: one R-rule crossing (a Hurwitz-style conjugation step),
// fresh-wire case split (R -> xR, !xR on a uniformly random uninvolved wire),
// unsubsume (!l R -> R, lR), copy-pair insertion (an existing gate inserted
// twice at a random position), and conjugation twists (the SAMF mechanism from
// ssg, XGate-native: bracket a window with a wire negation or a 3-CNOT wire
// swap and conjugate its interior — see twist_move). Contraction move: a
// pairwise merge from the closed-form catalogue (see merge_result). Every move
// is exhaustively verified on its support; the chain never emits comp=1 gates,
// so the comp count is a monotone "fossil" count of surviving original g57s.
//
// Provenance: every gate carries (origin, event) — the original-gate index its
// material descends from, and the split event that created it. A merge whose
// partners share an event is a sibling re-merge (the undo of one split);
// recent events are tabu to keep freshly split pairs from instantly rejoining.
use super::arena::{Arena, Dir, NIL};
use super::db_replace::{DbMode, DbResult, DegreeGuard, db_replace};
use super::g57_pairs::{G57PairPlan, G57PairSeedConfig, G57PairSpec, G57ShotStop};
use super::rules::{self, BlockReason, Outcome, Role, RuleKind};
use super::swap_words;
use super::xgate::{Lits, XGate};
use super::xpoly::XPolyBudget;
use crate::replace::frozen::FrozenDb;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::hash::{Hash, Hasher};

pub const ORIGIN_SYNTH: u32 = u32::MAX;
/// Generation stamp for born-random identity material.
///
/// This follows the established new-SSS/fmix convention: synthetic material
/// carries no structure from the input circuit, so it is treated as already
/// above every finite re-encoding target. Saturating generation updates keep
/// the sentinel fixed.
pub const GEN_FRESH: u32 = u32::MAX;
const FRAME_G57_PAIR_FLAG: u64 = 1u64 << 63;

/// How a DB move gathers its outgoing XGate window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DbSample {
    Contiguous,
    Convex,
    Mixed,
}

impl DbSample {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "contiguous" => Some(Self::Contiguous),
            "convex" => Some(Self::Convex),
            "mixed" => Some(Self::Mixed),
            _ => None,
        }
    }
}

const TG_CONTEXT: usize = 3;
/// Maximum distance an all-G57 bracket may slide outward looking for an
/// attachment gate that pins both twist wires.
pub const TG_SLIDE_CAP: usize = 512;
/// Improving attachment positions solved per seam slide.
pub const TG_SLIDE_TRIES: usize = 3;
/// Candidate windows considered by joint two-seam acceptance.
pub const TG_RETRIES: usize = 4;
/// Accept a candidate immediately once the combined two-seam net reaches this
/// bound. This admits two +4 attached seams, or one bare +6 seam paid for by a
/// partner at +2 or better.
pub const TG_ACCEPT_NET: isize = 8;

fn tg_slide_on() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("TWIST_G57_NO_SLIDE").is_none())
}

fn tg_retry_on() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("TWIST_G57_NO_RETRY").is_none())
}

struct TgSeam {
    /// Final interior window edge. Sliding moves this outward from the drawn
    /// boundary and the newly covered gates join the conjugated window.
    edge: u32,
    ids: Vec<u32>,
    replacement: Vec<XGate>,
}

impl TgSeam {
    fn net_growth(&self) -> isize {
        self.replacement.len() as isize - self.ids.len() as isize
    }
}

struct TgPlan {
    a: u16,
    b: u16,
    left: TgSeam,
    right: TgSeam,
    slides: u64,
}

impl TgPlan {
    fn score(&self) -> (isize, isize) {
        let consumed = (self.left.ids.len() + self.right.ids.len()) as isize;
        (self.left.net_growth() + self.right.net_growth(), -consumed)
    }
}

/// The width-two complemented population split by control-polarity shape.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct G57Census {
    pub shaped: usize,
    pub same_pol: usize,
    pub opp_pol: usize,
}

impl G57Census {
    pub fn pol_flipped(&self) -> f64 {
        if self.shaped == 0 {
            0.0
        } else {
            self.same_pol as f64 / self.shaped as f64
        }
    }
}

fn g57_pair_frame(pair_index: usize, copy: usize) -> u64 {
    assert!(copy < 2);
    let payload = (pair_index as u64)
        .checked_mul(2)
        .and_then(|value| value.checked_add(copy as u64))
        .expect("G57 pair frame id overflow");
    assert!(payload < FRAME_G57_PAIR_FLAG, "too many G57 pair frames");
    FRAME_G57_PAIR_FLAG | payload
}

fn decode_g57_pair_frame(frame: u64) -> Option<(usize, usize)> {
    if frame & FRAME_G57_PAIR_FLAG == 0 {
        return None;
    }
    let payload = frame & !FRAME_G57_PAIR_FLAG;
    Some(((payload / 2) as usize, (payload % 2) as usize))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FmixG57PairRecord {
    pub pair_index: usize,
    pub spec: G57PairSpec,
    pub left_intact_shot_steps: u64,
    pub right_intact_shot_steps: u64,
    pub left_intact_stop: G57ShotStop,
    pub right_intact_stop: G57ShotStop,
    pub left_transport_steps: u64,
    pub right_transport_steps: u64,
    pub left_cross_attempts: u64,
    pub right_cross_attempts: u64,
    pub left_r_crossings: u64,
    pub right_r_crossings: u64,
    pub left_fragment_positions: Vec<usize>,
    pub right_fragment_positions: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FmixG57PairSeedReport {
    pub baseline_gates: usize,
    pub final_gates: usize,
    pub pairs: usize,
    pub inserted_native_g57: usize,
    pub emitted_fragments: usize,
    pub target_wires: usize,
    pub control_wire_limit: usize,
    pub pairs_per_target_wire: usize,
    pub region: super::g57_pairs::G57PairRegion,
    pub first_gap: usize,
    pub gap_end_exclusive: usize,
    pub seed: u64,
    pub total_intact_left_shot_steps: u64,
    pub total_intact_right_shot_steps: u64,
    pub intact_collision_stops: usize,
    pub intact_boundary_stops: usize,
    pub total_transport_steps: u64,
    pub total_cross_attempts: u64,
    pub halves_with_r_crossing: usize,
    pub records: Vec<FmixG57PairRecord>,
}

/// Live descendant census for the identity-pair frames seeded by
/// [`Mixer::seed_g57_pairs`].
///
/// A logical pair has two copy frames (the left- and right-shot copies).
/// Rewrites may split one copy into several tagged gates or erase a tag when
/// material from incompatible frames merges.  Counting both pair IDs and copy
/// frames therefore distinguishes "some material from this pair survived"
/// from "both directional copies are still identifiable".
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct G57PairFrameStats {
    /// Distinct pair indices with at least one live tagged descendant.
    pub distinct_pair_ids: usize,
    /// Distinct `(pair_index, copy)` frames with at least one live descendant.
    pub distinct_copy_frames: usize,
    /// Pair indices for which both copy frames remain live.
    pub complete_pairs: usize,
    /// Total live gates carrying a G57-pair frame tag.
    pub tagged_gates: usize,
    /// Fraction of circuit positions covered by the union of each live copy
    /// frame's outermost descendant interval.
    pub union_span_coverage: f64,
}

impl FmixG57PairSeedReport {
    pub fn manifest_tsv(&self) -> String {
        let mut out = String::new();
        out.push_str("pair_index\tround\ttarget\tcontrol_1\tcontrol_2\tfrozen_gap\tleft_intact_shot_steps\tright_intact_shot_steps\tleft_intact_stop\tright_intact_stop\tleft_fragment_transport_steps\tright_fragment_transport_steps\tleft_cross_attempts\tright_cross_attempts\tleft_r_crossings\tright_r_crossings\tleft_fragment_positions\tright_fragment_positions\n");
        for record in &self.records {
            let join = |positions: &[usize]| {
                positions
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            };
            out.push_str(&format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                record.pair_index,
                record.spec.round,
                record.spec.target,
                record.spec.first_control,
                record.spec.second_control,
                record.spec.gap,
                record.left_intact_shot_steps,
                record.right_intact_shot_steps,
                record.left_intact_stop,
                record.right_intact_stop,
                record.left_transport_steps,
                record.right_transport_steps,
                record.left_cross_attempts,
                record.right_cross_attempts,
                record.left_r_crossings,
                record.right_r_crossings,
                join(&record.left_fragment_positions),
                join(&record.right_fragment_positions),
            ));
        }
        out
    }
}

// Closed-form pairwise merge catalogue: for same-target gates, f_g XOR f_h is a
// (possibly complemented) monomial in exactly these cases. Results are always
// comp=0 (or a cancellation): pairs whose fusion would be complemented — which
// is precisely the rejoin of a g57's presplit pieces — return None. That guard
// is what makes g57 erosion irreversible under this chain.
pub enum Merge {
    // f_g == f_h: both gates vanish.
    Cancel,
    // Same controls, opposite comp: (c XOR m) XOR (!c XOR m) = 1, a NOT gate.
    XFuse(XGate),
    // Same wires, one polarity flipped, equal comp: xR XOR !xR = R.
    DropLit(XGate),
    // Wire sets differ by one literal, shared literals equal, equal comp:
    // R XOR lR = !lR.
    Subsume(XGate),
}

impl Merge {
    pub fn gates(&self) -> Vec<XGate> {
        match self {
            Merge::Cancel => vec![],
            Merge::XFuse(g) | Merge::DropLit(g) | Merge::Subsume(g) => vec![g.clone()],
        }
    }
}

pub fn merge_result(g: &XGate, h: &XGate) -> Option<Merge> {
    if g.target != h.target {
        return None;
    }
    if g.ctrls == h.ctrls {
        return Some(if g.comp == h.comp {
            Merge::Cancel
        } else {
            Merge::XFuse(XGate::x_gate(g.target))
        });
    }
    // Below here the monomials differ; a complemented result is banned, and
    // comp1 != comp2 always complements the residual monomial.
    if g.comp != h.comp {
        return None;
    }
    let (gl, hl) = (g.ctrls.len(), h.ctrls.len());
    if gl == hl {
        // Same wire multiset with exactly one polarity flipped -> drop that wire.
        if !g.ctrls.iter().zip(h.ctrls.iter()).all(|(a, b)| a.0 == b.0) {
            return None;
        }
        let mut diff = None;
        for i in 0..gl {
            if g.ctrls[i].1 != h.ctrls[i].1 {
                if diff.is_some() {
                    return None;
                }
                diff = Some(i);
            }
        }
        let d = diff?;
        let lits = g
            .ctrls
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != d)
            .map(|(_, &l)| l);
        return Some(Merge::DropLit(
            XGate::conj(g.target, lits).expect("drop-lit merge"),
        ));
    }
    if gl.abs_diff(hl) != 1 {
        return None;
    }
    // Subset-plus-one-literal with ALL shared polarities equal. A flipped shared
    // polarity would complement the result (the presplit-rejoin case): banned.
    let (small, big) = if gl < hl { (g, h) } else { (h, g) };
    let mut extra = None;
    let mut si = small.ctrls.iter().peekable();
    for &(w, p) in &big.ctrls {
        match si.peek() {
            Some(&&(sw, sp)) if sw == w => {
                if sp != p {
                    return None;
                }
                si.next();
            }
            _ => {
                if extra.is_some() {
                    return None;
                }
                extra = Some((w, p));
            }
        }
    }
    if si.peek().is_some() {
        return None;
    }
    let (w, _) = extra?;
    let lits = big
        .ctrls
        .iter()
        .map(|&(cw, cp)| if cw == w { (cw, !cp) } else { (cw, cp) });
    Some(Merge::Subsume(
        XGate::conj(big.target, lits).expect("subsume merge"),
    ))
}

// Merge-partner index key: target + control WIRE SET (polarities and comp
// excluded — cancel/xfuse partners share it exactly, drop-lit partners differ
// only in a polarity, and subsume partners are found by looking up the key
// with one wire removed). Hash collisions are harmless: merge_result rechecks.
fn merge_key(target: u16, wires: impl Iterator<Item = u16>) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    target.hash(&mut h);
    for w in wires {
        w.hash(&mut h);
    }
    h.finish()
}

fn key_of(g: &XGate) -> u64 {
    merge_key(g.target, g.ctrls.iter().map(|&(w, _)| w))
}

// Conjugation of a single gate by NOT(w): N g N. A gate reading w sees the
// wire flipped on both sides, which is exactly a polarity flip of its
// w-literal; a gate TARGETING w is invariant (the two flips cancel through the
// XOR: !(!a ^ f) = a ^ f). Width and comp are preserved. None = g is invariant.
fn conj_by_not(g: &XGate, w: u16) -> Option<XGate> {
    if !g.reads(w) {
        return None;
    }
    let mut out = g.clone();
    for l in out.ctrls.iter_mut() {
        if l.0 == w {
            l.1 = !l.1;
        }
    }
    Some(out)
}

// Conjugation by SWAP(a, b): relabel the two wires wherever they occur, target
// or control, polarities travelling with their wire. Distinct wires stay
// distinct and the target stays outside its own controls, so the result is a
// well-formed XGate of the same width and comp. None = g touches neither wire.
fn conj_by_swap(g: &XGate, a: u16, b: u16) -> Option<XGate> {
    if g.target != a && g.target != b && !g.reads(a) && !g.reads(b) {
        return None;
    }
    let m = |w: u16| {
        if w == a {
            b
        } else if w == b {
            a
        } else {
            w
        }
    };
    let mut ctrls: Lits = g.ctrls.iter().map(|&(w, p)| (m(w), p)).collect();
    ctrls.sort_unstable();
    Some(XGate {
        target: m(g.target),
        comp: g.comp,
        ctrls,
    })
}

// Conjugation by the transvection T = CNOT(b -> a), i.e. x_a ^= x_b: T g T.
// T is linear but NOT a Hamming isometry — this is the twist rung that breaks
// distance-preserving self-gauges (avalanche profiles) that negations and
// swaps provably cannot move. The substitution is x_a -> x_a ^ x_b in every
// READ of a; b must not be the gate's target (a gate writing b and reading a
// would have to read its own target — inexpressible), which the window-level
// b-selection guarantees. Cases:
//  - g does not read a: invariant. Gates TARGETING a are invariant too (the
//    two T's toggle a by the same x_b and cancel through the XOR), and gates
//    merely reading b see it unchanged (T writes only a).
//  - g reads a and carries a b-literal of polarity q: on the gate's firing
//    slice x_b == q, so x_a ^ x_b == x_a ^ q — flip the a-literal's polarity
//    iff q. One gate, width and comp preserved (exact for comp gates too:
//    the substitution happens inside the conjunction).
//  - g reads a with no b-literal: case-split on b. lit_a(x_a ^ x_b) fires on
//    the disjoint pair (b=0 AND lit_a) / (b=1 AND !lit_a), so g becomes two
//    gates: count x2, width +1 — the structural cost of an affine frame,
//    charged conceptually against w_fresh (it is a fresh-wire split in
//    disguise). Inexpressible for comp gates (the split literal would land
//    inside a complemented conjunction): Blocked, the caller skips the window.
enum CnotConj {
    Invariant,
    Flip(XGate),
    Split(XGate, XGate),
    Blocked,
}

fn conj_by_cnot(g: &XGate, a: u16, b: u16) -> CnotConj {
    debug_assert!(
        g.target != b,
        "cnot twist requires b unwritten in the window"
    );
    if !g.reads(a) {
        return CnotConj::Invariant;
    }
    match g.ctrls.iter().find(|&&(w, _)| w == b).map(|&(_, q)| q) {
        Some(false) => CnotConj::Invariant,
        Some(true) => {
            let mut out = g.clone();
            for l in out.ctrls.iter_mut() {
                if l.0 == a {
                    l.1 = !l.1;
                }
            }
            CnotConj::Flip(out)
        }
        None => {
            if g.comp {
                return CnotConj::Blocked;
            }
            let with = |bp: bool, flip: bool| {
                XGate::conj(
                    g.target,
                    g.ctrls
                        .iter()
                        .map(|&(w, p)| if flip && w == a { (w, !p) } else { (w, p) })
                        .chain([(b, bp)]),
                )
                .expect("b is fresh to the gate")
            };
            CnotConj::Split(with(false, false), with(true, true))
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum TwistKind {
    Neg,
    Swap,
    Cnot,
}

#[derive(Clone)]
pub struct MixParams {
    pub k_max: usize,
    // Width damping for expansion moves, same convention as fsplit.
    pub split_damp: usize,
    // Base B of the width damper: a split of parent width c is allowed with
    // probability B^-(c - split_damp) (historically hardcoded at 2).
    pub split_base: f64,
    // Directional walk. Every gate carries a left/right direction (fossils get
    // a random one). A cross shoots the gate in its OWN direction; every
    // fragment born in a collision inherits the shot gate's direction with
    // probability dir_p, else the opposite — regardless of whether it is a
    // piece of the shot or of the colliding gate. A fresh piece advances
    // floor(dir_q * slack) gates in its own direction at birth (this replaces
    // the uniform scatter), and a failed cross retreats the shot gate
    // floor((1 - dir_q) * way) of the way it floated in.
    pub dir_p: f64,
    pub dir_q: f64,
    // Thermostat: p(contract) = sigmoid((size - target) / temp), clamped to
    // [0.02, 0.98] so the chain never fully stops expanding or contracting.
    pub target_size: usize,
    /// Absolute transactional growth boundary. Zero disables it.
    pub hard_size_cap: usize,
    pub temp: f64,
    // Total move attempts.
    pub moves: u64,
    // Max distance (gates) a merge partner may sit from the initiator; the
    // locating scan and the wall check both walk at most this far.
    pub merge_reach: usize,
    // Undo journal capacity (recorded crossings eligible for reversal) and the
    // fraction of contraction moves that try a journal undo first. Crossings
    // are the one expansion move the pairwise merge catalogue cannot invert
    // (ladder rungs are pairwise unmergeable), so without the journal the size
    // creeps up at the crossing rate no matter what the thermostat does.
    pub journal_len: usize,
    pub undo_frac: f64,
    // Refractory period: a split event may not be undone (journal) or
    // sibling-merged (catalogue) until this many moves have passed.
    pub tabu_moves: u64,
    // Relative weights of the expansion moves.
    pub w_cross: f64,
    pub w_fresh: f64,
    pub w_unsub: f64,
    pub w_insert: f64,
    // Conjugation-twist weights (state/progress mixing; see twist_move). Off by
    // default: with both at 0 the walk is move-for-move identical to the
    // pre-twist chain at equal seed. Keep small when on — one twist rewrites
    // O(window) gates, and window lengths run up to the whole circuit.
    pub w_twist_neg: f64,
    pub w_twist_swap: f64,
    // Transvection twist: conjugate the window by x_a ^= x_b (one CNOT per
    // side). Affine and non-isometric — the rung that breaks Hamming-distance-
    // preserving self-gauges — at the cost of case-splitting interior
    // a-readers (count x2, width +1, K-cap enforced). Windows are capped at
    // the mid scale by the need for an unwritten b wire.
    pub w_twist_cnot: f64,
    /// Layer-1 swap-family polarity coin. Each swapped wire is independently
    /// negated with this probability by first-class (`p_twist`) twists.
    pub twist_neg_p: f64,
    /// Spell first-class pure-swap brackets as adaptive all-G57 words.
    pub twist_g57: bool,
    // Persistent nonlinear frame. A move builds a reversible packet P from
    // width-2/3 controlled-X (Toffoli-class) gates over a random carrier set,
    // inserts P . P^-1, then transports tagged material in opposite directions
    // with the ordinary, locally verified crossing rules. The frame material
    // is protected from merge/undo for `nl_frame_tenure` moves so the nonlinear
    // state encoding survives long enough to overlap later frames.
    pub w_nl_frame: f64,
    // Per-gate control width inside P. Defaults stay at active quadratic/cubic
    // degree: very wide conjunctions almost never fire and can produce a
    // cosmetically flat affine heatmap without genuinely masking the state.
    pub nl_frame_min_width: usize,
    pub nl_frame_max_width: usize,
    // Number of low-degree gates in P. Packet breadth is independent of the
    // control width so a frame can touch many carriers without sparse gates.
    pub nl_frame_packet_gates: usize,
    // Total crossing shots attempted across the two separating packet halves.
    pub nl_frame_shots: usize,
    pub nl_frame_tenure: u64,
    // Twist window lengths are log-uniform over [twist_min_len, circuit size]:
    // the all-scales dial that decorrelates computational progress at every
    // window scale, the structured analog of ssg's long-range shooting.
    pub twist_min_len: usize,
    // Optional first-class twist cadence. At zero the expansion-weight
    // trajectory is unchanged.
    pub p_twist: f64,
    // Frozen-store replacement channels. All zero by default, which leaves the
    // store unopened and preserves the pre-DB RNG trajectory.
    pub w_db: f64,
    pub p_db: f64,
    // Upper clamp on the contraction probability. The old 0.98 left a 2%
    // expansion floor that is a structural growth source (measured +0.007
    // gates/move); it used to be tightened only under p_db_steer, which is
    // gone, so it is a parameter now rather than a side effect of a retired
    // flag.
    pub contract_ceiling: f64,
    pub db_min_window: usize,
    pub db_max_window: usize,
    pub db_sample: DbSample,
    // Two eligibility thresholds, not one. w_window governs what may sit INSIDE
    // a window; w_pool governs what may SEED one and count toward the dose.
    // They want different values: width-3 gates match in context often enough
    // to be worth admitting to windows, but their end-to-end per-gate re-encode
    // rate is 0.41% against 98.98% for width <= 2, so at a shared threshold they
    // pile up at the bottom of the pool with nothing to eject them.
    pub w_window: usize,
    pub w_pool: usize,
    pub db_convex_p: f64,
    pub db_verify: bool,
    pub db_dry_run: bool,
    pub db_max_degree: usize,
    pub db_degree_probes: usize,
    pub db_max_span: usize,
    pub db_wire_terms: usize,
    pub db_total_terms: usize,
    pub db_prefixes: bool,
    /// Fixed slot-2 mode when the layer-2 overlay is disabled.
    pub db_mode: DbMode,
    /// Per-round probability of MIX rather than COMP. Negative disables the
    /// overlay and preserves the historical fixed-mode trajectory.
    pub p_mix: f64,
    /// When MIX has to grow, choose from every growing spelling rather than
    /// only the shortest spellings.
    pub mix_pay_random: bool,
    /// Layer-2 start length. Zero falls back to `db_max_window`.
    pub s_db: usize,
    /// Layer-2 convex-sampler probability. Negative falls back to `db_sample`.
    pub p_convex: f64,
    /// COMP-specific overrides; zero/negative mean use the base value.
    pub s_db_comp: usize,
    pub p_convex_comp: f64,
    // p_mingen: probability a DB seed is drawn from the generation POOL rather
    // than uniformly -- what stops the process being a coupon collector, where
    // the last few percent of gates soak up most of the moves.
    pub p_mingen: f64,
    pub p_mingen_comp: f64,
    // pool_k: the pool is the K lowest-generation gates among those that are
    // pool-eligible AND still below the goal. Both filters are load-bearing. An
    // ineligible gate can never be re-encoded, so its generation is pinned
    // forever and an unfiltered pool converges on exactly that set; and without
    // the below-goal filter a late-run pool is padded with ordinary
    // low-but-finished gates that re-encode fine, so the canary could never
    // fire. A COUNT, not a fraction: the drain rate is set by the move economy
    // (gen_rescan x p_db x p_mingen) and is independent of circuit size, so a
    // percentage over-provisions as the circuit grows and under-provisions on
    // small ones. K must exceed the draws taken between rebuilds, or the pool
    // empties and the biased coin silently degrades to uniform.
    pub pool_k: usize,
    // Canary: fire when the failure fraction over the last canary_window
    // QUALIFYING rounds exceeds canary_theta. Healthy failure rates sit well
    // under 0.2 (five rungs against ~99% per-window hit rates on width-<=2
    // material) while the pathological case drives toward 1.0, so 0.9 sits in a
    // wide gap. 0 = off.
    pub canary_theta: f64,
    pub canary_window: usize,
    // Size brake (hysteresis). Growth to size_hi arms COMP; the mode is
    // released back to db_mode at size_lo OR when COMP stops paying, whichever
    // comes first. The productivity release is what makes a WIDE band safe: the
    // risk was never the width, it was sitting in COMP past its usefulness,
    // where it starves (declines rise as the circuit approaches local
    // minimality) and spends re-encoding diversity (COMP draws only from
    // minimum-size spellings, pulling toward the form fcompress would reach).
    // 0 = brake off.
    pub size_hi: usize,
    pub size_lo: usize,
    // Release COMP when its shed rate over the trailing window falls below this
    // many gates per round.
    pub comp_release_eps: f64,
    pub comp_release_window: u64,
    // litter_ban: refuse a window that is exactly one COMPLETE litter -- the
    // unit some earlier replacement emitted, and therefore precisely where the
    // store can hand the outgoing spelling straight back (A -> B -> A).
    // Singleton litters are exempt by construction: an input gate has no
    // earlier spelling to be returned to, and banning it would also refuse the
    // descent's length-1 rung, the one rung that always makes progress.
    pub litter_ban: bool,
    // Move-cadence snapshots: at each multiple, write circuit+gens+state.
    // 0 = off. snap_base is the filename prefix.
    pub snap_every_moves: u64,
    pub snap_base: String,
    // litter_samples: draw this many candidate windows and keep the one
    // spanning the MOST distinct litters. 1 = off. Discarded candidates may
    // leave ctrl-cap evasion floats behind; those are function-preserving and
    // the walk floats constantly, so the cost is arena churn, not correctness.
    pub litter_samples: usize,
    /// Curated-first routing and the two-pass prefix-descent policy.
    pub curated: bool,
    pub curated_exhaust: bool,
    pub curated_in_comp: bool,
    /// Give DB products the existing ballistic birth advance.
    pub db_advance: bool,
    /// Layer-2 size profile. `prof_n[2] == 0` disables it.
    pub prof_n: [f64; 3],
    pub prof_r: [f64; 2],
    pub prof_cadence_eff: f64,
    pub prof_deadband: f64,
    pub prof_dp_max: f64,
    pub prof_ewma: f64,
    pub prof_ki: f64,
    // Generation goal: the pool targets gates whose dgen sits below this.
    pub gen_target: u32,
    pub gen_rescan: u64,
    pub gen_split_inherit: bool,
    pub gen_median_low: bool,
    /// Dose stop threshold over targetable gates: cap-eligible gates that
    /// have not been retired by `gen_giveup`.
    pub gen_stop_frac: f64,
    /// When true, generation dose stop/revalidation uses only gates that do
    /// not carry an identifiable G57-pair frame tag. The all-gate census is
    /// still reported. Default false preserves the historical denominator.
    pub gen_dose_exclude_g57_pair_frames: bool,
    pub twist_cov_stop: f64,
    pub verify_every: u64,
    pub report_every: u64,
    pub local_verify: bool,
    pub seed: u64,
}

impl Default for MixParams {
    fn default() -> MixParams {
        MixParams {
            k_max: 12,
            split_damp: 2,
            split_base: 2.0,
            dir_p: 0.75,
            dir_q: 0.85,
            target_size: 0, // 0 -> input size, resolved by Mixer::new
            hard_size_cap: 0,
            temp: 0.0, // 0 -> max(target/100, 64), resolved by Mixer::new
            moves: 1_000_000,
            merge_reach: 4096,
            journal_len: 1 << 18,
            // Reinstated after the clock audit (2026-07-13): undo reverses
            // only sterile crossings (stamp-liveness protects anything that
            // fed later moves) and is the size valve for crossing ladders,
            // which the pairwise catalogue cannot invert. Raw crossing
            // counters overcount net work ~2x — read r1/r2/r3 minus undos.
            undo_frac: 0.5,
            tabu_moves: 2_000,
            w_cross: 0.70,
            // SUSPENDED: fresh-wire case splits are covered by the twists'
            // interior case-splitting; set > 0 to re-enable.
            w_fresh: 0.0,
            w_unsub: 0.10,
            w_insert: 0.05,
            w_twist_neg: 0.0,
            w_twist_swap: 0.0,
            w_twist_cnot: 0.0,
            twist_neg_p: 0.5,
            twist_g57: false,
            w_nl_frame: 0.0,
            nl_frame_min_width: 2,
            nl_frame_max_width: 3,
            nl_frame_packet_gates: 16,
            nl_frame_shots: 64,
            nl_frame_tenure: 100_000,
            twist_min_len: 64,
            p_twist: 0.0,
            w_db: 0.0,
            p_db: 0.0,
            contract_ceiling: 0.98,
            db_min_window: 2,
            db_max_window: 12,
            db_sample: DbSample::Contiguous,
            w_window: 4,
            w_pool: 3,
            db_convex_p: 0.75,
            db_verify: true,
            db_dry_run: false,
            db_max_degree: 0,
            db_degree_probes: 6,
            db_max_span: 0,
            db_wire_terms: 0,
            db_total_terms: 0,
            db_prefixes: false,
            db_mode: DbMode::SizeAgnostic,
            p_mix: -1.0,
            mix_pay_random: false,
            s_db: 0,
            p_convex: -1.0,
            s_db_comp: 0,
            p_convex_comp: -1.0,
            p_mingen: 0.8,
            p_mingen_comp: -1.0,
            pool_k: 20_000,
            canary_theta: 0.0,
            canary_window: 2000,
            size_hi: 0,
            size_lo: 0,
            comp_release_eps: 0.0,
            comp_release_window: 250_000,
            litter_ban: false,
            snap_every_moves: 0,
            snap_base: String::new(),
            litter_samples: 1,
            curated: false,
            curated_exhaust: false,
            curated_in_comp: false,
            db_advance: false,
            prof_n: [0.0; 3],
            prof_r: [0.0; 2],
            prof_cadence_eff: 0.5,
            prof_deadband: 0.02,
            prof_dp_max: 0.1,
            prof_ewma: 0.3,
            prof_ki: 0.05,
            gen_target: 0,
            gen_rescan: 10_000,
            gen_split_inherit: false,
            gen_median_low: false,
            gen_stop_frac: -1.0,
            gen_dose_exclude_g57_pair_frames: false,
            twist_cov_stop: 0.0,
            verify_every: 10_000,
            report_every: 50_000,
            local_verify: true,
            seed: 0,
        }
    }
}

#[derive(Default)]
pub struct MixCounters {
    pub moves: u64,
    pub db_slot2_rounds: u64,
    pub db_slot2_hits: u64,
    pub db_slot2_added: u64,
    /// Heads coins that found the pool drained.
    pub canary_fallthrough: u64,
    pub brake_engagements: u64,
    pub brake_rounds: u64,
    pub litter_banned: u64,
    pub litter_full_spliced: u64,
    pub litter_windows: u64,
    pub litter_distinct_sum: u64,
    pub db_identity_skips: u64,
    pub db_curated_hits: u64,
    pub dmin_windows: u64,
    pub dmin_shorter: u64,
    pub choice_splices: u64,
    pub choice_multi: u64,
    pub choice_sum: u64,
    pub choice_bits_milli: u64,
    pub db_mix_added: u64,
    pub db_mix_removed: u64,
    pub db_cmp_added: u64,
    pub db_cmp_removed: u64,
    pub db_comp_hits: u64,
    pub db_comp_misses: u64,
    pub db_agn_hits: u64,
    pub db_agn_misses: u64,
    pub db_gates_removed: u64,
    pub db_gates_added: u64,
    pub db_wide_skip: u64,
    pub db_attempts: u64,
    pub db_degree_skips: u64,
    pub db_span_skips: u64,
    pub db_build_aborts: u64,
    pub db_protected_skips: u64,
    pub db_frame_skips: u64,
    pub db_cap_skips: u64,
    pub blocked_size_cap: u64,
    pub merges_cancel: u64,
    pub merges_xfuse: u64,
    pub merges_drop: u64,
    pub merges_subsume: u64,
    pub merges_sibling: u64,
    pub merges_cross_origin: u64,
    pub tabu_blocked: u64,
    pub merge_no_partner: u64,
    pub merge_wall_blocked: u64,
    pub merge_too_far: u64,
    pub merge_not_adjacent: u64,
    pub undos: u64,
    pub undo_dead: u64,
    pub undo_tabu: u64,
    pub undo_gather_miss: u64,
    pub cross_r1: u64,
    pub cross_r2: u64,
    pub cross_r3: u64,
    pub presplits: u64,
    pub fresh_splits: u64,
    pub unsubs: u64,
    pub inserts: u64,
    pub twist_negs: u64,
    pub twist_swaps: u64,
    pub twist_cnots: u64,
    pub twist_relabels: u64,
    pub twist_case_splits: u64,
    pub twist_span: u64,
    pub twist_skips: u64,
    pub tg_consumed: u64,
    pub tg_emitted: u64,
    pub tg_solves: u64,
    pub tg_solve_ns: u64,
    pub tg_slides: u64,
    pub tg_retries: u64,
    /// Per-seam net growth, with shrinking seams folded into bucket zero and
    /// values above seven saturated into the last bucket.
    pub tg_net_hist: [u64; 8],
    pub nl_frame_attempts: u64,
    pub nl_frames: u64,
    pub nl_frame_skips: u64,
    pub nl_frame_packet_gates: u64,
    pub nl_frame_shot_attempts: u64,
    pub nl_frame_crossings: u64,
    pub nl_frame_preparatory_rewrites: u64,
    pub nl_frame_blocked: u64,
    pub nl_frame_span: u64,
    pub nl_frame_nodes: u64,
    pub nl_frame_protected_blocks: u64,
    pub g57_pairs_seeded: u64,
    pub g57_seed_gates: u64,
    pub g57_seed_shots: u64,
    pub g57_seed_fragments: u64,
    pub g57_seed_r_crossings: u64,
    pub blocked_width: u64,
    pub blocked_deadlock: u64,
    pub declined: u64,
    pub boundary: u64,
    pub floats: u64,
    pub float_steps: u64,
    pub scatters: u64,
    pub scatter_steps: u64,
    pub dropped_neverfire: u64,
    pub width_hist: [u64; 16],
}

impl MixCounters {
    pub fn merges(&self) -> u64 {
        self.merges_cancel + self.merges_xfuse + self.merges_drop + self.merges_subsume
    }
    pub fn expands(&self) -> u64 {
        self.cross_r1
            + self.cross_r2
            + self.cross_r3
            + self.presplits
            + self.fresh_splits
            + self.unsubs
            + self.inserts
            + self.twist_negs
            + self.twist_swaps
            + self.twist_cnots
            + self.nl_frames
    }
}

/// A new-SSS-style minimum-generation objective over the live XGate circuit.
///
/// `fraction` is clamped to `[0, 1]` by the statistics/control methods.  A
/// target is met when at least that fraction of the live gates has generation
/// `>= min_generation`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenerationTarget {
    pub min_generation: u32,
    pub fraction: f64,
}

/// Exact generation summary over the live arena.
///
/// `quantile_floor` ignores the bottom `(1 - target.fraction)` tail.  Thus at
/// a 98% target it is the generation immediately above the lowest 2%, matching
/// the Stage-D progress floor used by new SSS.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenerationStats {
    pub total: usize,
    pub reached: usize,
    pub reached_fraction: f64,
    pub min: u32,
    pub quantile_floor: u32,
    pub median: u32,
    pub max: u32,
}

/// Generation-target census over the pool predicate: dose accounting and pool
/// targeting share one eligibility test (`pool_eligible`). Nothing is written
/// off any more — with the descent reaching length 1, which cannot decline for
/// free, an eligible gate the store knows at all advances; the residue that
/// truly cannot is what the canary is for, rather than a per-gate miss counter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GenStats {
    /// Pool-eligible gates still below gen_target — the dose-stop numerator.
    pub lag: u64,
    /// Pool-eligible gates in total.
    pub elig: u64,
    /// Wide (pool-ineligible) gates below gen_target: invisible to the DB
    /// channel until some other move narrows or splits them.
    pub wlag: u64,
    pub min: u32,
    pub all_lag: u64,
    /// Live gates carrying the born-random generation sentinel.
    pub fresh: u64,
    pub total: u64,
    /// Largest generation reached by at least 95% of targetable gates.
    pub g_circ: u32,
    /// Legacy 5th-percentile generation over every gate in this scope.
    pub g_all: u32,
    /// Gates the DB channel can actually move; the denominator behind
    /// `g_circ` and the generation-dose stop.
    pub targetable: u64,
}

/// The two generation denominators reported for a dose decision.
///
/// `all` includes the whole circuit. `non_pair` excludes only gates whose live
/// metadata still identifies them as descendants of a seeded G57 pair.
/// Within either scope, generation and dose use that scope's targetable
/// population; the historical all-gate census remains available in `g_all`,
/// `all_lag`, and `total`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GenerationDoseStats {
    pub all: GenStats,
    pub non_pair: GenStats,
}

/// One bounded, low-generation anti-simplicity pass.
///
/// Every admitted rewrite is a complete, locally verified one-to-two
/// expansion.  Since each successful rewrite grows by exactly one gate, the
/// optional hard cap is never crossed or partially committed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenerationPassConfig {
    pub target: GenerationTarget,
    pub pass_length: usize,
    pub size_cap: Option<usize>,
    /// Random probes used to find a fresh case-split wire before falling back
    /// to a deterministic scan.  Zero means one random probe.
    pub candidate_trials: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenerationPassReport {
    pub attempted: usize,
    pub applied: usize,
    pub blocked: usize,
    pub before: GenerationStats,
    pub after: GenerationStats,
    pub cap_reached: bool,
}

/// Result of a contraction-only cadence.  Unlike the ordinary thermostat this
/// path never spends a residual probability mass on expansion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContractionReport {
    pub start_size: usize,
    pub end_size: usize,
    pub target_size: usize,
    pub attempts: u64,
    pub sweeps: usize,
    pub merges: u64,
    pub undos: u64,
    pub target_reached: bool,
    pub stalled: bool,
}

fn median_floor_u32(values: &mut [u32]) -> u32 {
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    values[(values.len() - 1) / 2]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Meta {
    origin: u32,
    event: u64, // 0 = not a split product
    // Persistent shooting direction: a cross floats this gate in `dir`.
    // Fossils draw it uniformly at birth; fragments inherit per dir_p.
    dir: Dir,
    // Non-zero for material descended from a persistent nonlinear frame.
    // `protected_until` blocks the two destructive contraction paths but not
    // further verified crossings/splits, so protected material can continue
    // diffusing while its trivial inverse remains unavailable.
    frame: u64,
    protected_until: u64,
    // Replacement-event provenance. Input gates and born-random material are
    // singleton litters; a DB splice stamps all products with one fresh id;
    // splits and merges propagate. litter_size is size at creation and is
    // deliberately NOT maintained under splits, so the complete-litter test
    // is conservative under churn. A NEW FIELD BESIDE `frame`, never
    // overloaded into it: try_db_splice folds/inherits frames while a litter
    // is minted fresh per splice — opposite lifecycles.
    litter: u64,
    litter_size: u16,
    // Rewrite generation (new-SSS/fmix semantics). Input gates start at zero;
    // a split child gets parent+1, a merge keeps the minimum parent
    // generation, and born-random identity material gets GEN_FRESH. Pure
    // transport/reordering leaves the stamp unchanged.
    dgen: u32,
}

struct CrossTrace {
    changed: bool,
    // True only for an R1/R2/R3 crossing. Presplitting either participant is
    // a useful preparatory rewrite but does not yet transport the shot.
    crossed: bool,
    // Descendants of the shot gate which crossed the collider and can keep
    // travelling in the same direction. R3 stay-behind cores retain frame
    // metadata but intentionally are not returned as transport frontiers.
    shot_descendants: Vec<u32>,
}

impl CrossTrace {
    fn unchanged(id: u32) -> CrossTrace {
        CrossTrace {
            changed: false,
            crossed: false,
            shot_descendants: vec![id],
        }
    }
}

struct SeedRShotTrace {
    descendants: Vec<u32>,
    attempts: u64,
    crossings: u64,
}

// A recorded crossing, eligible for exact reversal while every emitted node is
// still alive and untouched (checked via arena stamps — any later split, merge
// or reuse of a piece bumps its stamp and kills the entry). Pieces pairwise
// commute (same target, none reads it) and may drift to either side of the
// pivot (rungs carry a flipped ladder literal, so under the separation
// exemption most commute with the pivot too); the gather accretes from both
// sides and the restore is exhaustively verified, so drift is harmless.
struct UndoEntry {
    // The pre-crossing pair, in circuit order for the recorded direction.
    before: [XGate; 2],
    dir: Dir,
    pivot: u32,
    after: Vec<(u32, u32)>, // (id, stamp) of every emitted node, incl. pivot
    event: u64,
    origins: [u32; 2], // origin of before[0], before[1]
    frames: [u64; 2],
    litters: [u64; 2],
    litter_sizes: [u16; 2],
    protected_until: [u64; 2],
    gens: [u32; 2],
    misses: u8, // failed gather attempts; entry dropped after a few
}

/// Layer-2 profile-controller state. Rates are gates per move at full lever;
/// `dhat` captures size drift not attributed to the DB channel.
#[derive(Clone, Debug)]
pub struct ProfState {
    pub phase: u8,
    pub pmix: f64,
    pub integ: f64,
    pub ghat: f64,
    pub shat: f64,
    pub dhat: f64,
    pub eff: f64,
    pub next_eff: f64,
    pub sat: u32,
    pub s_in: f64,
    base_moves: u64,
    base_size: f64,
    base_pmix: f64,
    base_mix: [u64; 4],
    base_comp: [u64; 4],
}

/// Three-leg size target: expand, hold, then compress.
pub fn prof_target(n: [f64; 3], r: [f64; 2], input_size: f64, eff: f64) -> f64 {
    if eff <= 0.0 {
        input_size
    } else if eff < n[0] {
        input_size * (1.0 + (r[0] - 1.0) * eff / n[0])
    } else if eff < n[1] {
        input_size * r[0]
    } else if eff < n[2] {
        input_size * (r[0] + (r[1] - r[0]) * (eff - n[1]) / (n[2] - n[1]))
    } else {
        input_size * r[1]
    }
}

pub struct Mixer {
    pub arena: Arena,
    pub params: MixParams,
    pub counters: MixCounters,
    meta: Vec<Meta>,
    // merge-key -> linked node ids with that (target, wire-set). Kept exact by
    // the index_add/index_remove hooks in every splice; validated at verify
    // points via indexed_count.
    index: HashMap<u64, Vec<u32>>,
    indexed_count: usize,
    journal: VecDeque<UndoEntry>,
    tabu: VecDeque<(u64, u64)>, // (event, move at creation)
    next_event: u64,
    original: Vec<XGate>,
    num_wires: usize,
    moves_done: u64,
    rng: StdRng,
    // Sampling RNG for report-line gauges only. Separate from `rng` so adding
    // or re-cadencing metrics never perturbs the move trajectory of a seed.
    metrics_rng: StdRng,
    // Graceful stop / on-demand snapshot, both file-flag driven (checked at the
    // report cadence): touch stop_flag -> finish cleanly; touch dump_flag ->
    // verified snapshot to dump_out, continue.
    stop_flag: Option<String>,
    dump_flag: Option<String>,
    dump_out: String,
    stop_requested: bool,
    // None is the zero-cost/default path: no environment lookup and no store
    // handle when every DB channel is disabled.
    db: Option<FrozenDb>,
    db_budget: XPolyBudget,
    db_record: Option<std::io::BufWriter<std::fs::File>>,
    db_last_sampler: DbSample,
    // The pool: below-target, pool-eligible gates, rebuilt on the gen_rescan
    // cadence; drawn lazily with stale-entry pruning.
    pool: Vec<u32>,
    pool_scan_due: u64,
    seed_from_pool: bool,
    seed_fell_through: bool,
    // (seed id, left neighbour at draw). A failed attempt walks the seed home
    // so repeated pool draws of a stubborn gate leave no displacement.
    db_seed_home: Option<(u32, u32)>,
    // Canary ring: outcome of the last canary_window pool-seeded non-COMP
    // rounds (true = the store had no spelling at any rung).
    canary: VecDeque<bool>,
    canary_failures: usize,
    // Size-brake hysteresis state.
    next_litter: u64,
    brake_on: bool,
    brake_mark_move: u64,
    brake_mark_size: usize,
    db_mode_cur: DbMode,
    prof: Option<ProfState>,
}

pub enum MixStop {
    MovesBudget,
    StopFlag,
    DoseReached,
    ProfileDone,
    /// The canary fired: over the last `canary_window` pool-seeded growth
    /// rounds, the store failed to spell more than `canary_theta` of them at
    /// every rung. The pool's residue is unspellable — more moves cannot
    /// advance it, so stopping loses nothing.
    CanaryFired,
}

impl Mixer {
    pub fn new(gates: Vec<XGate>, num_wires: usize, params: MixParams) -> Mixer {
        let db_enabled = params.w_db > 0.0 || params.p_db > 0.0;
        let db = db_enabled.then(FrozenDb::from_env);
        Self::new_with_optional_db(gates, num_wires, params, db)
    }

    /// Construct with an explicitly opened store. Tests and embedded callers
    /// use this path without mutating process environment.
    pub fn new_with_db(
        gates: Vec<XGate>,
        num_wires: usize,
        params: MixParams,
        db: FrozenDb,
    ) -> Mixer {
        Self::new_with_optional_db(gates, num_wires, params, Some(db))
    }

    fn new_with_optional_db(
        gates: Vec<XGate>,
        num_wires: usize,
        mut params: MixParams,
        db: Option<FrozenDb>,
    ) -> Mixer {
        // Complemented empty gates are exact identities.  Older fcompress
        // outputs may contain them, and the complemented-gate crossing path
        // otherwise mistakes them for splittable G57 fossils.  Canonicalize
        // them away at the mixer boundary before building arena metadata.
        // Retain the raw input index as the origin of every surviving gate.
        let (gates, input_origins): (Vec<XGate>, Vec<u32>) = gates
            .into_iter()
            .enumerate()
            .filter_map(|(index, gate)| {
                (!gate.is_noop())
                    .then_some((gate, u32::try_from(index).expect("too many input gates")))
            })
            .unzip();
        let n = gates.len();
        if params.target_size == 0 {
            params.target_size = if params.hard_size_cap > 0 {
                n.min(params.hard_size_cap)
            } else {
                n
            };
        }
        if params.hard_size_cap > 0 {
            assert!(
                params.target_size <= params.hard_size_cap,
                "target_size {} exceeds hard_size_cap {}",
                params.target_size,
                params.hard_size_cap
            );
        }
        if params.temp <= 0.0 {
            params.temp = (params.target_size as f64 / 100.0).max(64.0);
        }
        if params.prof_n[2] > 0.0 {
            assert!(
                params.prof_n[0] > 0.0
                    && params.prof_n[1] >= params.prof_n[0]
                    && params.prof_n[2] > params.prof_n[1],
                "invalid profile effective-work marks"
            );
            assert!(
                params.prof_r[0] > 1.0
                    && params.prof_r[1] >= 1.0
                    && params.prof_r[0] >= params.prof_r[1],
                "invalid profile size ratios"
            );
        }
        let num_wires = num_wires.max(super::xgate::max_wire(&gates) as usize + 1);
        if params.w_nl_frame > 0.0 {
            assert!(
                params.nl_frame_packet_gates > 0,
                "nl_frame_packet_gates must be positive when nonlinear frames are enabled"
            );
            assert!(
                params.nl_frame_min_width >= 2,
                "nonlinear frame gates need at least two controls"
            );
            assert!(
                params.nl_frame_min_width <= params.nl_frame_max_width,
                "nl_frame_min_width exceeds nl_frame_max_width"
            );
            assert!(
                params.nl_frame_max_width <= params.k_max,
                "nonlinear frame control width exceeds k_max"
            );
            assert!(
                params.nl_frame_max_width < num_wires,
                "nonlinear frame needs a target plus distinct control wires"
            );
        }
        let mut rng = StdRng::seed_from_u64(params.seed);
        let metrics_rng = StdRng::seed_from_u64(params.seed ^ 0x5EED_517A75);
        let meta: Vec<Meta> = input_origins
            .into_iter()
            .enumerate()
            .map(|(i, origin)| Meta {
                origin,
                event: 0,
                dir: if rng.random_bool(0.5) { Dir::L } else { Dir::R },
                frame: 0,
                protected_until: 0,
                // Every input gate is its own singleton litter.
                litter: i as u64,
                litter_size: 1,
                dgen: 0,
            })
            .collect();
        let next_litter = meta.len() as u64;
        let mut index: HashMap<u64, Vec<u32>> = HashMap::new();
        for (i, g) in gates.iter().enumerate() {
            index.entry(key_of(g)).or_default().push(i as u32);
        }
        let db_budget = {
            let mut budget = XPolyBudget::default();
            if params.db_wire_terms > 0 {
                budget.max_poly_terms = params.db_wire_terms;
            }
            if params.db_total_terms > 0 {
                budget.max_total_terms = params.db_total_terms;
            }
            budget
        };
        let db_mode_cur = params.db_mode;
        Mixer {
            arena: Arena::from_gates(gates.clone()),
            params,
            counters: MixCounters::default(),
            meta,
            index,
            indexed_count: n,
            journal: VecDeque::new(),
            tabu: VecDeque::new(),
            next_event: 1,
            next_litter,
            original: gates,
            num_wires,
            moves_done: 0,
            rng,
            metrics_rng,
            stop_flag: None,
            dump_flag: None,
            dump_out: String::new(),
            stop_requested: false,
            db,
            db_budget,
            db_record: None,
            db_last_sampler: DbSample::Contiguous,
            pool: Vec::new(),
            pool_scan_due: 0,
            seed_from_pool: false,
            seed_fell_through: false,
            db_seed_home: None,
            canary: VecDeque::new(),
            canary_failures: 0,
            brake_on: false,
            brake_mark_move: 0,
            brake_mark_size: 0,
            db_mode_cur,
            prof: None,
        }
    }

    pub fn enable_db_record(&mut self, path: &str) {
        match std::fs::File::create(path) {
            Ok(file) => self.db_record = Some(std::io::BufWriter::new(file)),
            Err(error) => eprintln!("[fmix] could not open --db-record {path}: {error}"),
        }
    }

    pub fn enable_flags(&mut self, stop: Option<String>, dump: Option<String>, dump_out: String) {
        self.stop_flag = stop;
        self.dump_flag = dump;
        self.dump_out = dump_out;
    }

    /// Per-gate rewrite-generation stamps in circuit order.
    ///
    /// `GEN_FRESH` denotes born-random identity material.
    pub fn gens_in_order(&self) -> Vec<u32> {
        self.arena
            .ids_in_order()
            .into_iter()
            .map(|id| self.meta_of(id).dgen)
            .collect()
    }

    /// Descriptive alias retained for callers of the first instrumentation
    /// draft.
    pub fn generations_in_order(&self) -> Vec<u32> {
        self.gens_in_order()
    }

    pub fn generation_histogram(&self) -> BTreeMap<u32, usize> {
        let mut histogram = BTreeMap::new();
        for generation in self.gens_in_order() {
            *histogram.entry(generation).or_insert(0) += 1;
        }
        histogram
    }

    pub fn generation_stats(&self, target: GenerationTarget) -> GenerationStats {
        let mut generations = self.gens_in_order();
        if generations.is_empty() {
            return GenerationStats {
                total: 0,
                reached: 0,
                reached_fraction: 1.0,
                min: 0,
                quantile_floor: 0,
                median: 0,
                max: 0,
            };
        }
        generations.sort_unstable();
        let total = generations.len();
        let reached = generations
            .iter()
            .filter(|&&generation| generation >= target.min_generation)
            .count();
        let fraction = if target.fraction.is_finite() {
            target.fraction.clamp(0.0, 1.0)
        } else {
            1.0
        };
        let skip =
            (((1.0 - fraction) * total as f64).floor() as usize).min(total.saturating_sub(1));
        GenerationStats {
            total,
            reached,
            reached_fraction: reached as f64 / total as f64,
            min: generations[0],
            quantile_floor: generations[skip],
            median: generations[total / 2],
            max: generations[total - 1],
        }
    }

    /// DB-targeting census over the pool predicate.
    pub fn gen_stats_for(&self, target: u32) -> GenStats {
        self.gen_stats_for_scope(target, false)
    }

    /// Generation-target census excluding live descendants of seeded G57
    /// identity pairs. Exclusion is metadata based: if a later rewrite erases
    /// an incompatible frame tag, that gate re-enters the non-pair census.
    pub fn gen_stats_for_non_pair(&self, target: u32) -> GenStats {
        self.gen_stats_for_scope(target, true)
    }

    fn gen_stats_for_scope(&self, target: u32, exclude_g57_pair_frames: bool) -> GenStats {
        let ids = self.arena.ids_in_order();
        let mut stats = GenStats {
            min: GEN_FRESH,
            ..GenStats::default()
        };
        let mut all_generations = Vec::with_capacity(ids.len());
        let mut targetable_generations = Vec::with_capacity(ids.len());
        for id in ids {
            let meta = self.meta_of(id);
            if exclude_g57_pair_frames && decode_g57_pair_frame(meta.frame).is_some() {
                continue;
            }
            stats.total += 1;
            let eligible = self.pool_eligible(id);
            stats.min = stats.min.min(meta.dgen);
            all_generations.push(meta.dgen);
            stats.fresh += u64::from(meta.dgen == GEN_FRESH);
            if eligible {
                stats.elig += 1;
            }
            if meta.dgen >= target {
                if eligible {
                    targetable_generations.push(meta.dgen);
                }
                continue;
            }
            stats.all_lag += 1;
            if !eligible {
                stats.wlag += 1;
            } else {
                stats.lag += 1;
                targetable_generations.push(meta.dgen);
            }
        }
        stats.targetable = targetable_generations.len() as u64;
        let percentile = |generations: &mut Vec<u32>| {
            if generations.is_empty() {
                0
            } else {
                generations.sort_unstable();
                generations[(generations.len() / 20).min(generations.len() - 1)]
            }
        };
        stats.g_all = percentile(&mut all_generations);
        stats.g_circ = if targetable_generations.is_empty() {
            stats.g_all
        } else {
            percentile(&mut targetable_generations)
        };
        stats
    }

    pub fn gen_stats(&self) -> GenStats {
        self.gen_stats_for(self.params.gen_target)
    }

    pub fn gen_stats_non_pair(&self) -> GenStats {
        self.gen_stats_for_non_pair(self.params.gen_target)
    }

    pub fn generation_dose_stats(&self) -> GenerationDoseStats {
        GenerationDoseStats {
            all: self.gen_stats(),
            non_pair: self.gen_stats_non_pair(),
        }
    }

    pub fn twist_coverage(&self) -> f64 {
        self.counters.twist_span as f64 / self.arena.len().max(1) as f64
    }

    fn dose_reached(&self) -> bool {
        if self.params.gen_target == 0 || self.params.gen_stop_frac < 0.0 {
            return false;
        }
        let stats = if self.params.gen_dose_exclude_g57_pair_frames {
            self.gen_stats_non_pair()
        } else {
            self.gen_stats()
        };
        if stats.targetable == 0 {
            return false;
        }
        let lag_fraction = stats.lag as f64 / stats.targetable as f64;
        lag_fraction <= self.params.gen_stop_frac
            && (self.params.twist_cov_stop <= 0.0
                || self.twist_coverage() >= self.params.twist_cov_stop)
    }

    pub fn generation_goal_met(&self, target: GenerationTarget) -> bool {
        let fraction = if target.fraction.is_finite() {
            target.fraction.clamp(0.0, 1.0)
        } else {
            1.0
        };
        self.generation_stats(target).reached_fraction >= fraction
    }

    /// Reproduce historical new-SSS `expansion_game` retagging: after a
    /// scored global expansion, every resulting gate receives one shared tag,
    /// `median_floor(all previous live tags) + 1`.
    ///
    /// This is deliberately separate from physical per-rewrite generation
    /// tracking because it is a global-pass accounting convention, not local
    /// lineage.  It consumes no randomness and returns the assigned tag.
    pub fn advance_all_generations_after_global_pass(&mut self) -> u32 {
        let ids = self.arena.ids_in_order();
        if ids.is_empty() {
            return 0;
        }
        let mut generations: Vec<u32> = ids.iter().map(|&id| self.meta_of(id).dgen).collect();
        let generation = median_floor_u32(&mut generations).saturating_add(1);
        for id in ids {
            let mut meta = self.meta_of(id);
            meta.dgen = generation;
            self.set_meta(id, meta);
        }
        generation
    }

    /// Apply exact one-to-two expansions to a bounded sample of the current
    /// lowest-generation live gates below `config.target.min_generation`.
    pub fn harden_low_generation_pass(
        &mut self,
        config: GenerationPassConfig,
    ) -> GenerationPassReport {
        let before = self.generation_stats(config.target);
        let mut cap_reached = config.size_cap.is_some_and(|cap| self.arena.len() >= cap);
        if before.total == 0
            || config.pass_length == 0
            || self.generation_goal_met(config.target)
            || cap_reached
        {
            return GenerationPassReport {
                attempted: 0,
                applied: 0,
                blocked: 0,
                before,
                after: before,
                cap_reached,
            };
        }

        let ids = self.arena.ids_in_order();
        let Some(lowest) = ids
            .iter()
            .map(|&id| self.meta_of(id).dgen)
            .filter(|&generation| generation < config.target.min_generation)
            .min()
        else {
            return GenerationPassReport {
                attempted: 0,
                applied: 0,
                blocked: 0,
                before,
                after: before,
                cap_reached,
            };
        };
        let mut candidates: Vec<u32> = ids
            .into_iter()
            .filter(|&id| self.meta_of(id).dgen == lowest)
            .collect();
        candidates.shuffle(&mut self.rng);

        let mut attempted = 0usize;
        let mut applied = 0usize;
        for id in candidates.into_iter().take(config.pass_length) {
            if config.size_cap.is_some_and(|cap| self.arena.len() >= cap) {
                cap_reached = true;
                break;
            }
            if !self.arena.is_linked(id)
                || self.meta_of(id).dgen != lowest
                || self.meta_of(id).dgen >= config.target.min_generation
            {
                continue;
            }
            attempted += 1;
            if self.harden_gate_exact(id, config.candidate_trials.max(1)) {
                applied += 1;
                if self.generation_goal_met(config.target) {
                    break;
                }
            }
        }
        if config.size_cap.is_some_and(|cap| self.arena.len() >= cap) {
            cap_reached = true;
        }
        let after = self.generation_stats(config.target);
        GenerationPassReport {
            attempted,
            applied,
            blocked: attempted - applied,
            before,
            after,
            cap_reached,
        }
    }

    /// Run only the existing exact contraction channels (merge and sterile
    /// crossing undo) until `target_size`, `max_attempts`, or a sweep-based
    /// no-progress condition. This advances the Mixer's move clock so tabu and
    /// protection tenure retain their ordinary semantics.
    pub fn contract_to(
        &mut self,
        target_size: usize,
        max_attempts: u64,
        stall_window: usize,
        min_reduction_fraction: f64,
    ) -> ContractionReport {
        let start_size = self.arena.len();
        let merges_before = self.counters.merges();
        let undos_before = self.counters.undos;
        let mut attempts = 0u64;
        let mut sweeps = 0usize;
        let mut stalled = false;
        let window = stall_window.max(1);
        let reduction_fraction = if min_reduction_fraction.is_finite() {
            min_reduction_fraction.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let mut recent = VecDeque::with_capacity(window + 1);
        recent.push_back(start_size);

        while self.arena.len() > target_size && attempts < max_attempts {
            let sweep_attempts =
                (self.arena.len().max(1) as u64).min(max_attempts.saturating_sub(attempts));
            for _ in 0..sweep_attempts {
                if self.arena.len() <= target_size {
                    break;
                }
                if self.rng.random_bool(self.params.undo_frac) {
                    if !self.undo_move() {
                        self.merge_move();
                    }
                } else if !self.merge_move() {
                    self.undo_move();
                }
                self.moves_done = self.moves_done.saturating_add(1);
                self.counters.moves = self.moves_done;
                attempts += 1;
                if self.params.verify_every > 0 && self.moves_done % self.params.verify_every == 0 {
                    self.global_check();
                }
            }
            sweeps += 1;
            recent.push_back(self.arena.len());
            if recent.len() > window + 1 {
                recent.pop_front();
            }
            if recent.len() == window + 1 {
                let reduction = recent
                    .front()
                    .copied()
                    .unwrap()
                    .saturating_sub(self.arena.len());
                let threshold = ((reduction_fraction * self.arena.len() as f64) as usize).max(1);
                if reduction < threshold {
                    stalled = true;
                    break;
                }
            }
        }
        self.global_check();
        ContractionReport {
            start_size,
            end_size: self.arena.len(),
            target_size,
            attempts,
            sweeps,
            merges: self.counters.merges() - merges_before,
            undos: self.counters.undos - undos_before,
            target_reached: self.arena.len() <= target_size,
            stalled,
        }
    }

    /// Insert true native-G57 identity pairs at frozen baseline gaps and
    /// immediately fragment/transport the two copies in opposite directions.
    /// Both stages use the same verified fmix primitives as the ordinary walk:
    /// exact `rules::presplit`, indexed splicing, and directional birth
    /// transport through proven-commuting gates.
    pub fn seed_g57_pairs(
        &mut self,
        config: G57PairSeedConfig,
    ) -> Result<FmixG57PairSeedReport, String> {
        if config.num_wires != self.num_wires {
            return Err(format!(
                "G57 pair plan declares {} wires but this Mixer has {}",
                config.num_wires, self.num_wires
            ));
        }
        let baseline_ids = self.arena.ids_in_order();
        let baseline_gates = baseline_ids.len();
        let plan = G57PairPlan::generate(baseline_gates, config)?;
        if self.params.hard_size_cap > 0 {
            let minimum_final = plan
                .specs
                .len()
                .checked_mul(4)
                .and_then(|added| baseline_gates.checked_add(added));
            if minimum_final.is_none_or(|size| size > self.params.hard_size_cap) {
                return Err(format!(
                    "G57 pair seed requires at least {} + 4*{} gates, above hard_size_cap {}",
                    baseline_gates,
                    plan.specs.len(),
                    self.params.hard_size_cap
                ));
            }
        }
        let mut by_gap: Vec<Vec<usize>> = vec![Vec::new(); baseline_gates + 1];
        for (pair_index, spec) in plan.specs.iter().enumerate() {
            by_gap[spec.gap].push(pair_index);
        }

        // Insert every pair before shooting any copy.  The anchors below are
        // immutable original-node ids, so region coordinates never drift as
        // the arena grows.
        let mut copies = vec![[NIL; 2]; plan.specs.len()];
        for (gap, pair_indices) in by_gap.iter().enumerate() {
            let mut anchor = if gap == 0 { NIL } else { baseline_ids[gap - 1] };
            for &pair_index in pair_indices {
                let gate = XGate::from_g57(plan.specs[pair_index].gate());
                debug_assert!(gate.comp && gate.width() == 2);
                if self.params.local_verify {
                    assert!(
                        rules::verify_rewrite(&[], &[gate.clone(), gate.clone()]),
                        "seeded native-G57 pair is not an identity: {gate:?}"
                    );
                }
                let left = self.arena.insert_after(anchor, gate.clone());
                self.index_add(left);
                let right = self.arena.insert_after(left, gate);
                self.index_add(right);
                let event = self.fresh_event();
                // Singleton litters: a singleton can never trip the ban, and
                // sharing one id would make the pair a complete litter.
                let left_litter = self.fresh_litter();
                let right_litter = self.fresh_litter();
                self.set_meta(
                    left,
                    Meta {
                        origin: ORIGIN_SYNTH,
                        event,
                        dir: Dir::L,
                        frame: g57_pair_frame(pair_index, 0),
                        protected_until: 0,
                        litter: left_litter,
                        litter_size: 1,
                        dgen: GEN_FRESH,
                    },
                );
                self.set_meta(
                    right,
                    Meta {
                        origin: ORIGIN_SYNTH,
                        event,
                        dir: Dir::R,
                        frame: g57_pair_frame(pair_index, 1),
                        protected_until: 0,
                        litter: right_litter,
                        litter_size: 1,
                        dgen: GEN_FRESH,
                    },
                );
                self.counters.width_hist[2] += 2;
                self.counters.g57_pairs_seeded += 1;
                self.counters.g57_seed_gates += 2;
                copies[pair_index] = [left, right];
                anchor = right;
            }
        }

        // First shoot the two intact copies apart through the maximal exact
        // commuting window.  This preserves the experiment's requested order:
        // adjacent identical G57s -> intact left/right shots -> fmix
        // presplitting and fragment transport.  `float_to_collision` only
        // permutes across gates that the conservative collision predicate
        // proves commute.
        let mut intact_shot_steps: Vec<[u64; 2]> = vec![[0, 0]; plan.specs.len()];
        let mut intact_stops: Vec<[G57ShotStop; 2]> =
            vec![[G57ShotStop::Boundary; 2]; plan.specs.len()];
        for pair_index in 0..plan.specs.len() {
            for copy in 0..2 {
                let direction = if copy == 0 { Dir::L } else { Dir::R };
                let id = copies[pair_index][copy];
                intact_shot_steps[pair_index][copy] = self.float_to_collision(id, direction) as u64;
                intact_stops[pair_index][copy] = if self.arena.neighbor(id, direction) == NIL {
                    G57ShotStop::Boundary
                } else {
                    G57ShotStop::Collision
                };
            }
        }

        // Every intact copy is then directly presplit. A normal cross can stop
        // at a boundary without fragmenting, while this explicit experimental
        // hook promises that every copy becomes the exact exclusive fragments
        // used by fmix.
        let mut descendants: Vec<[Vec<u32>; 2]> = (0..plan.specs.len())
            .map(|_| [Vec::new(), Vec::new()])
            .collect();
        let mut transport: Vec<[u64; 2]> = vec![[0, 0]; plan.specs.len()];
        for pair_index in 0..plan.specs.len() {
            for copy in 0..2 {
                let before = self.counters.scatter_steps;
                descendants[pair_index][copy] = self.presplit_g57_exact(
                    copies[pair_index][copy],
                    if copy == 0 { Dir::L } else { Dir::R },
                );
                transport[pair_index][copy] = self.counters.scatter_steps - before;
                self.counters.g57_seed_fragments += descendants[pair_index][copy].len() as u64;
            }
        }

        // Presplitting is preparatory, not a shot.  Give each directional half
        // a targeted opportunity to execute one real R crossing, retaining an
        // explicit zero when its entire frontier reaches only blockers or a
        // boundary.
        let mut cross_attempts: Vec<[u64; 2]> = vec![[0, 0]; plan.specs.len()];
        let mut r_crossings: Vec<[u64; 2]> = vec![[0, 0]; plan.specs.len()];
        for pair_index in 0..plan.specs.len() {
            for copy in 0..2 {
                let shot = self.seed_fragment_r_shot(
                    std::mem::take(&mut descendants[pair_index][copy]),
                    if copy == 0 { Dir::L } else { Dir::R },
                );
                descendants[pair_index][copy] = shot.descendants;
                cross_attempts[pair_index][copy] = shot.attempts;
                r_crossings[pair_index][copy] = shot.crossings;
                self.counters.g57_seed_shots += shot.attempts;
                self.counters.g57_seed_r_crossings += shot.crossings;
            }
        }

        let mut frame_positions: Vec<[Vec<usize>; 2]> = (0..plan.specs.len())
            .map(|_| [Vec::new(), Vec::new()])
            .collect();
        for (position, id) in self.arena.ids_in_order().into_iter().enumerate() {
            if let Some((pair_index, copy)) = decode_g57_pair_frame(self.meta_of(id).frame)
                && pair_index < frame_positions.len()
            {
                frame_positions[pair_index][copy].push(position);
            }
        }
        let mut records = Vec::with_capacity(plan.specs.len());
        for pair_index in 0..plan.specs.len() {
            records.push(FmixG57PairRecord {
                pair_index,
                spec: plan.specs[pair_index].clone(),
                left_intact_shot_steps: intact_shot_steps[pair_index][0],
                right_intact_shot_steps: intact_shot_steps[pair_index][1],
                left_intact_stop: intact_stops[pair_index][0],
                right_intact_stop: intact_stops[pair_index][1],
                left_transport_steps: transport[pair_index][0],
                right_transport_steps: transport[pair_index][1],
                left_cross_attempts: cross_attempts[pair_index][0],
                right_cross_attempts: cross_attempts[pair_index][1],
                left_r_crossings: r_crossings[pair_index][0],
                right_r_crossings: r_crossings[pair_index][1],
                left_fragment_positions: frame_positions[pair_index][0].clone(),
                right_fragment_positions: frame_positions[pair_index][1].clone(),
            });
        }
        let final_gates = self.arena.len();
        let inserted_native_g57 = plan.specs.len() * 2;
        let emitted_fragments = plan.specs.len() * 4;
        Ok(FmixG57PairSeedReport {
            baseline_gates,
            final_gates,
            pairs: plan.specs.len(),
            inserted_native_g57,
            emitted_fragments,
            target_wires: plan.config.target_wires,
            control_wire_limit: plan.config.control_wire_limit,
            pairs_per_target_wire: plan.config.pairs_per_target_wire,
            region: plan.config.region,
            first_gap: plan.gap_range.start,
            gap_end_exclusive: plan.gap_range.end,
            seed: plan.config.seed,
            total_intact_left_shot_steps: intact_shot_steps.iter().map(|steps| steps[0]).sum(),
            total_intact_right_shot_steps: intact_shot_steps.iter().map(|steps| steps[1]).sum(),
            intact_collision_stops: intact_stops
                .iter()
                .flatten()
                .filter(|&&stop| stop == G57ShotStop::Collision)
                .count(),
            intact_boundary_stops: intact_stops
                .iter()
                .flatten()
                .filter(|&&stop| stop == G57ShotStop::Boundary)
                .count(),
            total_transport_steps: transport.iter().flatten().sum(),
            total_cross_attempts: cross_attempts.iter().flatten().sum(),
            halves_with_r_crossing: r_crossings
                .iter()
                .flatten()
                .filter(|&&count| count > 0)
                .count(),
            records,
        })
    }

    fn set_meta(&mut self, id: u32, m: Meta) {
        let i = id as usize;
        if i >= self.meta.len() {
            self.meta.resize(
                i + 1,
                Meta {
                    origin: ORIGIN_SYNTH,
                    event: 0,
                    dir: Dir::R,
                    frame: 0,
                    protected_until: 0,
                    litter: 0,
                    litter_size: 1,
                    dgen: GEN_FRESH,
                },
            );
        }
        self.meta[i] = m;
    }

    fn meta_of(&self, id: u32) -> Meta {
        self.meta.get(id as usize).copied().unwrap_or(Meta {
            origin: ORIGIN_SYNTH,
            event: 0,
            dir: Dir::R,
            frame: 0,
            protected_until: 0,
            litter: 0,
            litter_size: 1,
            dgen: GEN_FRESH,
        })
    }

    fn is_protected(&self, id: u32) -> bool {
        self.arena.is_linked(id) && self.moves_done < self.meta_of(id).protected_until
    }

    #[inline]
    fn child_gen(&self, parent: u32) -> u32 {
        if self.params.gen_split_inherit {
            parent
        } else {
            parent.saturating_add(1)
        }
    }

    fn admits_rewrite_size(&self, removed: usize, added: usize) -> bool {
        if added <= removed || self.params.hard_size_cap == 0 {
            return true;
        }
        self.arena
            .len()
            .checked_sub(removed)
            .and_then(|base| base.checked_add(added))
            .is_some_and(|size| size <= self.params.hard_size_cap)
    }

    /// Generation stamp used by the tested XGate DB replacement path.
    ///
    /// Products are one level above the outgoing window's upper median by
    /// default; `median_low` selects the lower median experiment. An all-fresh
    /// window remains `GEN_FRESH` by saturation.
    pub(crate) fn db_replacement_generation(&self, ids: &[u32], median_low: bool) -> u32 {
        let mut gens: Vec<u32> = ids.iter().map(|&id| self.meta_of(id).dgen).collect();
        gens.sort_unstable();
        let middle = if median_low {
            gens.len().saturating_sub(1) / 2
        } else {
            gens.len() / 2
        };
        gens.get(middle)
            .copied()
            .unwrap_or(GEN_FRESH)
            .saturating_add(1)
    }

    /// Litter ids come from one counter, never reused; input gates own
    /// 0..n-1, so fresh ids start at the input count.
    fn fresh_litter(&mut self) -> u64 {
        let l = self.next_litter;
        self.next_litter += 1;
        l
    }

    // Litter census of a window: (distinct litters, is-exactly-one-complete-
    // litter). "Complete" requires every gate to share one id AND the count to
    // still equal the size recorded when that litter was emitted — so a litter
    // that has since been split or partly merged reads as incomplete, making
    // the test conservative under churn.
    //
    // Singleton litters are excluded by construction: input gates and
    // born-random material carry no earlier spelling to be returned to, and a
    // ban on them would also refuse the descent's length-1 rung, which is the
    // one rung that always makes progress.
    fn litter_census(&self, ids: &[u32]) -> (usize, bool) {
        if ids.is_empty() {
            return (0, false);
        }
        // Runs once per DB round (and once per rung under the ban), so it is
        // alloc-free for every real window: DB windows are <= db_max_window
        // (12-13) gates, which fits the stack buffer; anything larger spills
        // to the heap and stays correct.
        let mut buf = [0u64; 16];
        let mut n = 0usize;
        let mut distinct = 0usize;
        let mut spill: Option<Vec<u64>> = None;
        for &id in ids {
            let l = self.meta_of(id).litter;
            let seen = buf[..n].contains(&l) || spill.as_ref().is_some_and(|v| v.contains(&l));
            if !seen {
                distinct += 1;
                if n < buf.len() {
                    buf[n] = l;
                    n += 1;
                } else {
                    spill.get_or_insert_with(Vec::new).push(l);
                }
            }
        }
        let size = self.meta_of(ids[0]).litter_size;
        let full = distinct == 1 && size >= 2 && ids.len() == size as usize;
        (distinct, full)
    }

    fn fresh_event(&mut self) -> u64 {
        let e = self.next_event;
        self.next_event += 1;
        self.tabu.push_back((e, self.moves_done));
        while let Some(&(_, mv)) = self.tabu.front() {
            if mv + self.params.tabu_moves <= self.moves_done {
                self.tabu.pop_front();
            } else {
                break;
            }
        }
        e
    }

    fn is_tabu(&self, event: u64) -> bool {
        event != 0
            && self
                .tabu
                .iter()
                .any(|&(ev, mv)| ev == event && mv + self.params.tabu_moves > self.moves_done)
    }

    // ---- merge-partner index maintenance ----

    fn index_add(&mut self, id: u32) {
        let k = key_of(self.arena.gate(id));
        self.index.entry(k).or_default().push(id);
        self.indexed_count += 1;
    }

    fn index_remove(&mut self, id: u32) {
        let k = key_of(self.arena.gate(id));
        let bucket = self.index.get_mut(&k).expect("index bucket missing");
        let pos = bucket
            .iter()
            .position(|&x| x == id)
            .expect("id missing from index bucket");
        bucket.swap_remove(pos);
        if bucket.is_empty() {
            self.index.remove(&k);
        }
        self.indexed_count -= 1;
    }

    // ---- the chain ----

    fn active_s_db(&self) -> usize {
        if self.db_mode_cur == DbMode::Compressing && self.params.s_db_comp > 0 {
            self.params.s_db_comp
        } else if self.params.s_db > 0 {
            self.params.s_db
        } else {
            self.params.db_max_window
        }
    }

    fn active_db_minimum(&self) -> usize {
        if self.params.s_db > 0 {
            1
        } else {
            self.params.db_min_window.max(2)
        }
    }

    fn active_p_convex(&self) -> f64 {
        if self.db_mode_cur == DbMode::Compressing && self.params.p_convex_comp >= 0.0 {
            self.params.p_convex_comp
        } else {
            self.params.p_convex
        }
    }

    fn active_p_mingen(&self) -> f64 {
        if self.db_mode_cur == DbMode::Compressing && self.params.p_mingen_comp >= 0.0 {
            self.params.p_mingen_comp
        } else {
            self.params.p_mingen
        }
    }

    // Layer-2 mode overlay (slot 0): pick this round's DB mode by coin when
    // armed (p_mix >= 0) -- MIX-DB with probability p_mix, else COMP-DB. This is
    // the "set parameter" rule of the slot-0 condition engine, independent of
    // the size brake and the thermostat. Off (p_mix < 0) leaves db_mode_cur as
    // the fixed db_mode (possibly steered by the size brake); resetting it here
    // would clobber the brake every round.
    fn apply_mode_overlay(&mut self) {
        if self.params.p_mix < 0.0 {
            return;
        }
        self.db_mode_cur = if self.rng.random_bool(self.params.p_mix.clamp(0.0, 1.0)) {
            DbMode::Mix
        } else {
            DbMode::Compressing
        };
    }

    /// The size brake. Growth past `size_hi` forces slot 2 into COMP; it is
    /// released back to `params.db_mode` at `size_lo`, OR earlier when COMP has
    /// stopped paying.
    ///
    /// The productivity release is what makes a WIDE band safe, and a wide band
    /// is what the transport argument wants: growth legs are where material
    /// actually moves. The danger was never the width but sitting in COMP past
    /// its usefulness, where it starves -- declines climb as the circuit nears
    /// local minimality -- and spends re-encoding diversity, since COMP draws
    /// only from minimum-size spellings and so pulls the circuit toward exactly
    /// the form `fcompress` would compute anyway. Guarding that directly means
    /// a too-wide band costs nothing: the brake lets go on its own.
    fn apply_size_brake(&mut self) {
        if self.params.size_hi == 0 {
            return;
        }
        let size = self.arena.len();
        if !self.brake_on {
            if size >= self.params.size_hi {
                self.brake_on = true;
                self.db_mode_cur = DbMode::Compressing;
                self.brake_mark_move = self.moves_done;
                self.brake_mark_size = size;
                self.counters.brake_engagements += 1;
            }
            return;
        }
        self.counters.brake_rounds += 1;
        if size <= self.params.size_lo {
            self.brake_on = false;
            self.db_mode_cur = self.params.db_mode;
            return;
        }
        // Productivity release: COMP stops growth immediately but shrinks only
        // slowly (about 13% of its hits strictly shrink), so the shed rate is
        // the honest signal that the leg is still worth running.
        let window = self.params.comp_release_window.max(1);
        if self.moves_done.saturating_sub(self.brake_mark_move) >= window {
            let shed = self.brake_mark_size.saturating_sub(size) as f64;
            let rate = shed / window as f64;
            if rate < self.params.comp_release_eps {
                self.brake_on = false;
                self.db_mode_cur = self.params.db_mode;
                return;
            }
            self.brake_mark_move = self.moves_done;
            self.brake_mark_size = size;
        }
    }

    fn note_choice(&mut self, count: usize) {
        if count == 0 {
            return;
        }
        self.counters.choice_splices += 1;
        self.counters.choice_sum += count as u64;
        if count > 1 {
            self.counters.choice_multi += 1;
            self.counters.choice_bits_milli += ((count as f64).log2() * 1000.0).round() as u64;
        }
    }

    fn prof_init(&mut self) {
        let input_size = self.original.len().max(1) as f64;
        self.prof = Some(ProfState {
            phase: 1,
            pmix: 0.5,
            integ: 0.0,
            ghat: 0.0,
            shat: 0.0,
            dhat: 0.0,
            eff: 0.0,
            next_eff: self.params.prof_cadence_eff.max(0.05) * 0.25,
            sat: 0,
            s_in: input_size,
            base_moves: self.moves_done,
            base_size: input_size,
            base_pmix: 0.5,
            base_mix: [0; 4],
            base_comp: [0; 4],
        });
        self.prof_snapshot();
    }

    fn prof_snapshot(&mut self) {
        let mix = [
            self.counters.db_agn_hits,
            self.counters.db_agn_misses,
            self.counters.db_mix_added,
            self.counters.db_mix_removed,
        ];
        let comp = [
            self.counters.db_comp_hits,
            self.counters.db_comp_misses,
            self.counters.db_cmp_added,
            self.counters.db_cmp_removed,
        ];
        if let Some(profile) = self.prof.as_mut() {
            profile.base_moves = self.moves_done;
            profile.base_size = self.arena.len() as f64;
            profile.base_pmix = profile.pmix;
            profile.base_mix = mix;
            profile.base_comp = comp;
        }
    }

    fn prof_tick(&mut self) {
        let size = self.arena.len().max(1) as f64;
        let due = {
            let profile = self.prof.as_mut().expect("profile tick while disabled");
            profile.eff += 1.0 / size;
            profile.eff >= profile.next_eff
        };
        if due {
            self.prof_update();
        }
        let pmix = self.prof.as_ref().expect("profile disappeared").pmix;
        self.db_mode_cur = if self.rng.random_bool(pmix.clamp(0.0, 1.0)) {
            DbMode::Mix
        } else {
            DbMode::Compressing
        };
    }

    fn prof_update(&mut self) {
        let size = self.arena.len().max(1) as f64;
        let mix_now = [
            self.counters.db_agn_hits,
            self.counters.db_agn_misses,
            self.counters.db_mix_added,
            self.counters.db_mix_removed,
        ];
        let comp_now = [
            self.counters.db_comp_hits,
            self.counters.db_comp_misses,
            self.counters.db_cmp_added,
            self.counters.db_cmp_removed,
        ];
        let cadence = self.params.prof_cadence_eff.max(0.05);
        let deadband = self.params.prof_deadband.max(0.0);
        let dp_max = self.params.prof_dp_max.max(0.0);
        let ewma = self.params.prof_ewma.clamp(0.0, 1.0);
        let ki = self.params.prof_ki.max(0.0);
        let n = self.params.prof_n;
        let r = self.params.prof_r;
        if let Some(profile) = self.prof.as_mut() {
            let moves = self.moves_done.saturating_sub(profile.base_moves).max(1) as f64;
            let used = profile.base_pmix;
            let net_mix = (mix_now[2] - profile.base_mix[2]) as f64
                - (mix_now[3] - profile.base_mix[3]) as f64;
            let net_comp = (comp_now[3] - profile.base_comp[3]) as f64
                - (comp_now[2] - profile.base_comp[2]) as f64;
            if used > 0.05 {
                let fresh = net_mix / (moves * used);
                profile.ghat = if profile.ghat == 0.0 {
                    fresh
                } else {
                    (1.0 - ewma) * profile.ghat + ewma * fresh
                };
            }
            if used < 0.95 {
                let fresh = net_comp / (moves * (1.0 - used));
                profile.shat = if profile.shat == 0.0 {
                    fresh
                } else {
                    (1.0 - ewma) * profile.shat + ewma * fresh
                };
            }
            let observed = (size - profile.base_size) / moves;
            let disturbance = observed - (net_mix - net_comp) / moves;
            profile.dhat = if profile.dhat == 0.0 {
                disturbance
            } else {
                (1.0 - ewma) * profile.dhat + ewma * disturbance
            };

            let old_phase = profile.phase;
            match profile.phase {
                1 if profile.eff >= n[0] || size >= 0.99 * r[0] * profile.s_in => profile.phase = 2,
                2 if profile.eff >= n[1] => profile.phase = 3,
                3 if profile.eff >= n[2] || size <= 1.01 * r[1] * profile.s_in => profile.phase = 4,
                _ => {}
            }
            if profile.phase != old_phase {
                profile.integ = 0.0;
                profile.sat = 0;
            }

            let target = prof_target(n, r, profile.s_in, profile.eff);
            let next_target = prof_target(n, r, profile.s_in, profile.eff + cadence);
            let error = (size - target) / target.max(1.0);
            if error.abs() >= deadband {
                profile.integ = (profile.integ - ki * error).clamp(-0.3, 0.3);
                let wanted_velocity = (next_target - size) / (cadence * size).max(1.0);
                let denominator = profile.ghat + profile.shat;
                let mut wanted = if denominator.abs() > 1e-9 {
                    (wanted_velocity - profile.dhat + profile.shat) / denominator
                } else {
                    profile.pmix
                };
                wanted += profile.integ;
                let limit = if error.abs() > 4.0 * deadband.max(1e-6) {
                    1.0
                } else {
                    dp_max
                };
                profile.pmix =
                    (profile.pmix + (wanted - profile.pmix).clamp(-limit, limit)).clamp(0.0, 1.0);
            }
            if (profile.pmix >= 0.999 && error < -0.10) || (profile.pmix <= 0.001 && error > 0.10) {
                profile.sat += 1;
            } else {
                profile.sat = 0;
            }
            profile.next_eff += cadence;
            self.params.target_size = target.max(2.0) as usize;
        }
        self.prof_snapshot();
    }

    fn twist_round(&mut self) {
        if self.params.twist_g57 {
            self.twist_move_g57();
        } else {
            self.twist_swap_family_move();
        }
    }

    pub fn run(&mut self) -> MixStop {
        // The canonical form of an identity-only input is the empty tape.
        // Nothing can be mixed, but it remains a valid circuit and must pass
        // through the same final equivalence contract.
        if self.arena.len() == 0 {
            self.moves_done = self.params.moves;
            self.counters.moves = self.moves_done;
            self.global_check();
            return MixStop::MovesBudget;
        }
        if self.params.prof_n[2] > 0.0 && self.prof.is_none() {
            self.prof_init();
        }
        while self.moves_done < self.params.moves {
            if self.params.gen_target > 0 && self.moves_done >= self.pool_scan_due {
                self.rebuild_pool();
                self.pool_scan_due = self
                    .moves_done
                    .saturating_add(self.params.gen_rescan.max(1));
            }

            if self.prof.is_some() {
                self.prof_tick();
                if self.prof.as_ref().is_some_and(|profile| profile.phase == 4) {
                    self.global_check();
                    return MixStop::ProfileDone;
                }
            } else {
                // One size authority: while a profile is active it owns
                // target_size and the brake is inert (this branch never runs).
                self.apply_size_brake();
                self.apply_mode_overlay();
            }

            // ORIG parity gap: origin's slot 1b (`global_shuffle`/`shuffle_rate`,
            // ORIG mix.rs:3071-3086) is deliberately not ported — it is a
            // separate move type outside the five layer-1 pieces and would
            // confound the canary calibration.
            let took_twist = self.params.p_twist > 0.0
                && self.rng.random_bool(self.params.p_twist.clamp(0.0, 1.0))
                && {
                    self.twist_round();
                    true
                };
            let minimum_window = self.active_db_minimum();
            let p_db_now = self.params.p_db.clamp(0.0, 1.0);
            // Slot 2: the one DB round. One pool, one probability — the
            // ingest/paid tier machinery is gone (ORIG d92e1e0b).
            let took_db = !took_twist
                && p_db_now > 0.0
                && self.arena.len() >= minimum_window
                && self.rng.random_bool(p_db_now)
                && {
                    self.counters.db_slot2_rounds += 1;
                    let before = self.arena.len();
                    if self.db_attempt(self.db_mode_cur) {
                        self.counters.db_slot2_hits += 1;
                        self.counters.db_slot2_added +=
                            self.arena.len().saturating_sub(before) as u64;
                    }
                    true
                };

            if !took_twist && !took_db {
                let excess = self.arena.len() as f64 - self.params.target_size as f64;
                let high = self.params.contract_ceiling;
                let p_contract =
                    (1.0 / (1.0 + (-excess / self.params.temp).exp())).clamp(0.02, high);
                if self.arena.len() > 0 && self.rng.random_bool(p_contract) {
                    let did_db = self.params.w_db > 0.0
                        && self.rng.random_bool(self.params.w_db.clamp(0.0, 1.0))
                        && self.db_attempt(DbMode::Compressing);
                    if !did_db {
                        if self.rng.random_bool(self.params.undo_frac) {
                            if !self.undo_move() {
                                self.merge_move();
                            }
                        } else if !self.merge_move() {
                            self.undo_move();
                        }
                    }
                } else if self.arena.len() > 0 {
                    self.expand_move();
                }
            }
            self.moves_done = self.moves_done.saturating_add(1);
            self.counters.moves = self.moves_done;
            if self.params.verify_every > 0 && self.moves_done % self.params.verify_every == 0 {
                self.global_check();
            }
            if self.params.report_every > 0 && self.moves_done % self.params.report_every == 0 {
                self.report();
                self.check_flags();
                self.check_move_snap();
                if self.stop_requested {
                    self.global_check();
                    return MixStop::StopFlag;
                }
                if self.canary_fired() {
                    println!(
                        "[fmix] CANARY fired: frac={:.3} over {} pool-seeded rounds (theta={}) — the pool is unspellable by the store",
                        self.canary_frac(),
                        self.canary.len(),
                        self.params.canary_theta
                    );
                    self.global_check();
                    return MixStop::CanaryFired;
                }
                if self.dose_reached() {
                    self.global_check();
                    return MixStop::DoseReached;
                }
            }
        }
        self.global_check();
        MixStop::MovesBudget
    }

    fn check_flags(&mut self) {
        if let Some(f) = self.stop_flag.clone() {
            if std::path::Path::new(&f).exists() {
                let _ = std::fs::remove_file(&f);
                println!(
                    "[fmix] stop flag seen at move {}: finishing cleanly",
                    self.moves_done
                );
                self.stop_requested = true;
            }
        }
        if let Some(f) = self.dump_flag.clone() {
            if std::path::Path::new(&f).exists() {
                self.global_check();
                let gates = self.arena.to_vec();
                // Move-stamped filename so repeated touches build a trajectory
                // instead of overwriting; origins sidecar so the positional
                // metrics (diffusion/autocorr) are computable per snapshot.
                let out = format!("{}.mv{}", self.dump_out, self.moves_done);
                let tmp = format!("{out}.tmp");
                match super::format::write_mpmct(&tmp, &gates, self.num_wires) {
                    Ok(()) => {
                        if let Err(e) = std::fs::rename(&tmp, &out) {
                            eprintln!("[fmix] dump rename failed: {e}");
                        } else {
                            let mut s = String::with_capacity(gates.len() * 8);
                            for o in self.origins_in_order() {
                                s.push_str(&format!("{o}\n"));
                            }
                            if let Err(e) = std::fs::write(format!("{out}.origins"), s) {
                                eprintln!("[fmix] dump origins write failed: {e}");
                            }
                            let mut generations = String::with_capacity(gates.len() * 4);
                            for generation in self.gens_in_order() {
                                generations.push_str(&format!("{generation}\n"));
                            }
                            if let Err(e) = std::fs::write(format!("{out}.gens"), generations) {
                                eprintln!("[fmix] dump generations write failed: {e}");
                            }
                            println!(
                                "[fmix] DUMP: wrote {} gates to {} at move {} (verified, continuing)",
                                gates.len(),
                                out,
                                self.moves_done
                            );
                        }
                    }
                    Err(e) => eprintln!("[fmix] dump write failed: {e}"),
                }
                let _ = std::fs::remove_file(&f);
            }
        }
    }

    // ---- expansion moves ----

    fn expand_move(&mut self) {
        let p = &self.params;
        // A positive p_twist makes affine twists first-class top-level rounds.
        // In that mode w_twist_* are type ratios only and must not also consume
        // ordinary expansion slots. Keep nonlinear frames independent: they
        // remain an explicit expansion move in either scheduling mode.
        let (tw_n, tw_s, tw_c) = if p.p_twist > 0.0 {
            (0.0, 0.0, 0.0)
        } else {
            (p.w_twist_neg, p.w_twist_swap, p.w_twist_cnot)
        };
        let total =
            p.w_cross + p.w_fresh + p.w_unsub + p.w_insert + tw_n + tw_s + tw_c + p.w_nl_frame;
        if total <= 0.0 {
            return;
        }
        let mut r = self.rng.random_range(0.0..total);
        if r < p.w_cross {
            self.cross_move();
            return;
        }
        r -= p.w_cross;
        if r < p.w_fresh {
            self.fresh_split_move();
            return;
        }
        r -= p.w_fresh;
        if r < p.w_unsub {
            self.unsub_move();
            return;
        }
        r -= p.w_unsub;
        // The >0 guards keep floating-point dust in the subtractions from ever
        // reaching a zero-weight move: with all twist weights at 0 the
        // selection (and hence every seed's trajectory) is identical to the
        // pre-twist chain, and with only neg/swap set it is identical to the
        // pre-cnot chain.
        if tw_n > 0.0 && r - p.w_insert < tw_n && r >= p.w_insert {
            self.twist_move(TwistKind::Neg);
            return;
        }
        let ns = p.w_insert + tw_n;
        if tw_s > 0.0 && r - ns < tw_s && r >= ns {
            self.twist_move(TwistKind::Swap);
            return;
        }
        let cnot_start = ns + tw_s;
        if tw_c > 0.0 && r >= cnot_start && r < cnot_start + tw_c {
            self.twist_move(TwistKind::Cnot);
            return;
        }
        if p.w_nl_frame > 0.0 && r >= cnot_start + tw_c {
            self.nonlinear_frame_move();
            return;
        }
        self.insert_move();
    }

    // One R-rule crossing: float a random gate to its collision point and cross
    // it once. g57 shots pre-split (that IS the move); no cascade follow-up —
    // the chain's later moves pick the pieces up with fresh randomness.
    // Directional: the gate shoots in its OWN stored direction; fragments
    // inherit the shot direction per dir_p and advance per dir_q; a failed
    // cross retreats the shot gate instead of leaving it parked.
    fn cross_move(&mut self) {
        let id = self.arena.random_linked(&mut self.rng);
        self.cross_move_on(id);
    }

    /// Exact anti-simplicity rewrite for one selected low-generation gate.
    /// Every successful branch replaces one gate by two, verifies the local
    /// identity unconditionally, and assigns both children parent+1.
    fn harden_gate_exact(&mut self, id: u32, candidate_trials: usize) -> bool {
        if !self.arena.is_linked(id) {
            return false;
        }
        if !self.admits_rewrite_size(1, 2) {
            self.counters.blocked_size_cap += 1;
            return false;
        }
        let gate = self.arena.gate(id).clone();
        let parent = self.meta_of(id);
        if gate.comp {
            if gate.width() != 2 {
                return false;
            }
            let before = self.arena.len();
            let ids = self.presplit_g57_exact_impl(id, parent.dir, true);
            assert_eq!(
                ids.len(),
                2,
                "generation hardening requires a one-to-two G57 presplit"
            );
            assert_eq!(self.arena.len(), before + 1);
            return true;
        }

        // Prefer a fresh-wire partition because it increases support as well
        // as gate count. Random probes retain diversity; the deterministic
        // fallback guarantees that a legal fresh wire is not missed.
        let mut fresh_wire = None;
        if gate.width() < self.params.k_max {
            for _ in 0..candidate_trials.max(1) {
                let wire = self.rng.random_range(0..self.num_wires) as u16;
                if wire != gate.target && !gate.reads(wire) {
                    fresh_wire = Some(wire);
                    break;
                }
            }
            if fresh_wire.is_none() {
                fresh_wire = (0..self.num_wires as u16)
                    .find(|&wire| wire != gate.target && !gate.reads(wire));
            }
        }

        let pieces = if let Some(wire) = fresh_wire {
            let make = |polarity: bool| {
                XGate::conj(
                    gate.target,
                    gate.ctrls.iter().copied().chain([(wire, polarity)]),
                )
                .expect("fresh hardening wire cannot contradict")
            };
            self.counters.fresh_splits += 1;
            vec![make(true), make(false)]
        } else if !gate.ctrls.is_empty() {
            // Width-capped fallback: !lR = R XOR lR. This still adds one exact
            // structural layer without exceeding the parent's width.
            let index = self.rng.random_range(0..gate.ctrls.len());
            let (wire, _) = gate.ctrls[index];
            let without = XGate::conj(gate.target, gate.ctrls_without(wire)).expect("valid subset");
            let flipped = XGate::conj(
                gate.target,
                gate.ctrls.iter().map(|&(control, polarity)| {
                    if control == wire {
                        (control, !polarity)
                    } else {
                        (control, polarity)
                    }
                }),
            )
            .expect("valid polarity flip");
            self.counters.unsubs += 1;
            if self.rng.random_bool(0.5) {
                vec![without, flipped]
            } else {
                vec![flipped, without]
            }
        } else {
            return false;
        };
        assert!(
            rules::verify_rewrite(std::slice::from_ref(&gate), &pieces),
            "generation hardening rewrite is not exact: {gate:?} -> {pieces:?}"
        );
        let event = self.fresh_event();
        for piece in &pieces {
            self.counters.width_hist[piece.width().min(15)] += 1;
        }
        let ids = self.splice_replace_one(id, pieces);
        for child in ids {
            self.set_meta(
                child,
                Meta {
                    origin: parent.origin,
                    event,
                    dir: parent.dir,
                    frame: parent.frame,
                    protected_until: parent.protected_until,
                    litter: parent.litter,
                    litter_size: parent.litter_size,
                    dgen: self.child_gen(parent.dgen),
                },
            );
        }
        true
    }

    /// Exact exclusive pre-split shared by ordinary g57 crossings and the
    /// explicit pair-seed hook.  `shot_dir` controls fragment inheritance;
    /// callers remain responsible for any width-damper policy.
    fn presplit_g57_exact(&mut self, id: u32, shot_dir: Dir) -> Vec<u32> {
        self.presplit_g57_exact_impl(id, shot_dir, false)
    }

    fn presplit_g57_exact_impl(&mut self, id: u32, shot_dir: Dir, force_verify: bool) -> Vec<u32> {
        assert!(self.arena.is_linked(id), "presplit of an unlinked gate");
        let gate = self.arena.gate(id).clone();
        assert!(gate.comp, "presplit helper requires a complemented G57");
        if !self.admits_rewrite_size(1, gate.width()) {
            self.counters.blocked_size_cap += 1;
            return Vec::new();
        }
        let source_meta = self.meta_of(id);
        let pieces = rules::presplit(&gate, &mut self.rng);
        if force_verify || self.params.local_verify {
            assert!(
                rules::verify_rewrite(std::slice::from_ref(&gate), &pieces),
                "presplit verification failed: {gate:?} -> {pieces:?}"
            );
        }
        let event = self.fresh_event();
        for piece in &pieces {
            self.counters.width_hist[piece.width().min(15)] += 1;
        }
        let ids = self.splice_replace_one(id, pieces);
        for &piece_id in &ids {
            let direction = self.child_dir(shot_dir);
            self.set_meta(
                piece_id,
                Meta {
                    origin: source_meta.origin,
                    event,
                    dir: direction,
                    frame: source_meta.frame,
                    protected_until: source_meta.protected_until,
                    litter: source_meta.litter,
                    litter_size: source_meta.litter_size,
                    dgen: self.child_gen(source_meta.dgen),
                },
            );
        }
        self.advance_births(&ids);
        self.counters.presplits += 1;
        ids
    }

    /// Drive one seeded fragment half until it performs an actual R1/R2/R3
    /// crossing, or until every live frontier is blocked/boundary.  Encountered
    /// complemented colliders are pre-split by `cross_move_on` and retried;
    /// that preparatory rewrite is never misreported as the requested shot.
    fn seed_fragment_r_shot(&mut self, initial: Vec<u32>, direction: Dir) -> SeedRShotTrace {
        let mut descendants = initial;
        let mut queue: VecDeque<u32> = descendants.iter().copied().collect();
        let mut attempts = 0u64;
        // Every preparatory retry removes at least one complemented collider.
        // This guard is defensive against future rewrite kinds that might
        // otherwise return a changed self-frontier indefinitely.
        let attempt_cap = self
            .arena
            .len()
            .saturating_add(descendants.len())
            .saturating_add(1);
        while let Some(id) = queue.pop_front() {
            if attempts as usize >= attempt_cap || !self.arena.is_linked(id) {
                continue;
            }
            let mut meta = self.meta_of(id);
            meta.dir = direction;
            self.set_meta(id, meta);
            let floated = self.float_to_collision(id, direction);
            let collider = self.arena.neighbor(id, direction);
            if collider == NIL {
                self.retreat(id, floated, direction);
                attempts += 1;
                continue;
            }
            if decode_g57_pair_frame(self.meta_of(collider).frame).is_some() {
                // Seeded halves are independent experimental subjects.  A
                // tagged collider is a genuine roadblock, but must not be
                // consumed while trying to give this half its own R shot.
                self.retreat(id, floated, direction);
                attempts += 1;
                continue;
            }
            let trace = self.cross_move_on(id);
            attempts += 1;
            if trace.changed {
                descendants.retain(|&candidate| candidate != id);
                descendants.extend(trace.shot_descendants.iter().copied());
            }
            if trace.crossed {
                return SeedRShotTrace {
                    descendants,
                    attempts,
                    crossings: 1,
                };
            }
            if trace.changed {
                // Most commonly: a complemented collider was pre-split while
                // this shot stayed intact. Retry that exact live frontier.
                for child in trace.shot_descendants.into_iter().rev() {
                    if self.arena.is_linked(child) {
                        queue.push_front(child);
                    }
                }
            }
        }
        SeedRShotTrace {
            descendants,
            attempts,
            crossings: 0,
        }
    }

    fn cross_move_on(&mut self, id: u32) -> CrossTrace {
        let g_meta = self.meta_of(id);
        let dir = g_meta.dir;
        // Defensive support for non-canonical legacy inputs.  Treating a
        // complemented empty identity as a G57 presplit yields zero pieces;
        // the ordinary "split was blocked" path would then try to retreat an
        // arena id that splice_replace_one had already freed.  Delete the
        // identity before any floating instead.
        if self.arena.gate(id).is_noop() {
            self.splice_replace_one(id, Vec::new());
            return CrossTrace {
                changed: true,
                crossed: false,
                shot_descendants: Vec::new(),
            };
        }
        let way = self.float_to_collision(id, dir);
        let h_id = self.arena.neighbor(id, dir);
        if h_id == NIL {
            self.counters.boundary += 1;
            self.retreat(id, way, dir);
            return CrossTrace::unchanged(id);
        }
        let g = self.arena.gate(id).clone();
        let h = self.arena.gate(h_id).clone();
        let h_meta = self.meta_of(h_id);

        if g.comp {
            if !self.split_allowed(g.width()) {
                self.counters.declined += 1;
                self.retreat(id, way, dir);
                return CrossTrace::unchanged(id);
            }
            let ids = self.presplit_g57_exact(id, dir);
            if ids.is_empty() {
                self.retreat(id, way, dir);
                return CrossTrace::unchanged(id);
            }
            return CrossTrace {
                changed: true,
                crossed: false,
                shot_descendants: ids,
            };
        }

        match rules::cross(&g, &h, self.params.k_max, &mut self.rng) {
            Outcome::R0Swap => unreachable!("R0 after floating to collision"),
            Outcome::Blocked(BlockReason::WidthCap) => {
                self.counters.blocked_width += 1;
                self.retreat(id, way, dir);
                CrossTrace::unchanged(id)
            }
            Outcome::Blocked(BlockReason::Deadlock) => {
                self.counters.blocked_deadlock += 1;
                self.retreat(id, way, dir);
                CrossTrace::unchanged(id)
            }
            Outcome::PresplitColliding => {
                // The colliding gate is a g57 that must split: pre-splitting it
                // is this move's whole effect.
                if !self.split_allowed(h.width()) {
                    self.counters.declined += 1;
                    self.retreat(id, way, dir);
                    return CrossTrace::unchanged(id);
                }
                // Colliding-gate fragments still inherit from the SHOT gate's
                // direction (per spec: regardless of parent).
                if self.presplit_g57_exact(h_id, dir).is_empty() {
                    self.retreat(id, way, dir);
                    return CrossTrace::unchanged(id);
                }
                CrossTrace {
                    changed: true,
                    crossed: false,
                    // Only the collider changed; the shot remains the active
                    // transport frontier and can retry against the pieces.
                    shot_descendants: vec![id],
                }
            }
            Outcome::Rewrite { seq, kind, dropped } => {
                if !self.admits_rewrite_size(2, seq.len()) {
                    self.counters.blocked_size_cap += 1;
                    self.retreat(id, way, dir);
                    return CrossTrace::unchanged(id);
                }
                let split_width = match kind {
                    RuleKind::R1 | RuleKind::R3 => g.width(),
                    RuleKind::R2 => h.width(),
                };
                if !self.split_allowed(split_width) {
                    self.counters.declined += 1;
                    self.retreat(id, way, dir);
                    return CrossTrace::unchanged(id);
                }
                if self.params.local_verify {
                    let before: Vec<XGate> = match dir {
                        Dir::R => vec![g.clone(), h.clone()],
                        Dir::L => vec![h.clone(), g.clone()],
                    };
                    let after: Vec<XGate> = match dir {
                        Dir::R => seq.iter().map(|(x, _)| x.clone()).collect(),
                        Dir::L => seq.iter().rev().map(|(x, _)| x.clone()).collect(),
                    };
                    assert!(
                        rules::verify_rewrite(&before, &after),
                        "cross verification failed ({kind:?}, {dir:?}): {g:?} x {h:?}"
                    );
                }
                self.counters.dropped_neverfire += dropped as u64;
                match kind {
                    RuleKind::R1 => self.counters.cross_r1 += 1,
                    RuleKind::R2 => self.counters.cross_r2 += 1,
                    RuleKind::R3 => self.counters.cross_r3 += 1,
                }
                for (gate, role) in &seq {
                    if *role != Role::CollidingIntact {
                        self.counters.width_hist[gate.width().min(15)] += 1;
                    }
                }
                let ev = self.fresh_event();
                let placed = self.splice_pair(id, h_id, dir, seq);
                let mut fresh: Vec<u32> = Vec::new();
                let mut shot_descendants: Vec<u32> = Vec::new();
                for &(pid, role) in &placed {
                    match role {
                        Role::ShotPiece | Role::Core => {
                            let d = self.child_dir(dir);
                            self.set_meta(
                                pid,
                                Meta {
                                    origin: g_meta.origin,
                                    event: ev,
                                    dir: d,
                                    frame: g_meta.frame,
                                    protected_until: g_meta.protected_until,
                                    litter: g_meta.litter,
                                    litter_size: g_meta.litter_size,
                                    dgen: self.child_gen(g_meta.dgen),
                                },
                            );
                            fresh.push(pid);
                            if role == Role::ShotPiece {
                                shot_descendants.push(pid);
                            }
                        }
                        Role::CollidingPiece => {
                            let d = self.child_dir(dir);
                            self.set_meta(
                                pid,
                                Meta {
                                    origin: h_meta.origin,
                                    event: ev,
                                    dir: d,
                                    frame: h_meta.frame,
                                    protected_until: h_meta.protected_until,
                                    litter: h_meta.litter,
                                    litter_size: h_meta.litter_size,
                                    dgen: self.child_gen(h_meta.dgen),
                                },
                            );
                            fresh.push(pid);
                        }
                        Role::CollidingIntact => {} // node reused, meta intact
                    }
                }
                self.advance_births(&fresh);
                // Record for exact reversal (only when the undo channel is
                // live — with undo_frac == 0 the journal would be dead weight).
                // Pivot: the node every other piece collides with — h when
                // intact (R1/R3), else the passed shot (R2, the only ShotPiece
                // there).
                if self.params.undo_frac > 0.0 {
                    let pivot = placed
                        .iter()
                        .find(|(_, r)| *r == Role::CollidingIntact)
                        .or_else(|| placed.iter().find(|(_, r)| *r == Role::ShotPiece))
                        .map(|&(i, _)| i)
                        .expect("rewrite emitted no pivot");
                    let (before, origins, frames, litters, litter_sizes, protected_until, gens) =
                        match dir {
                            Dir::R => (
                                [g.clone(), h.clone()],
                                [g_meta.origin, h_meta.origin],
                                [g_meta.frame, h_meta.frame],
                                [g_meta.litter, h_meta.litter],
                                [g_meta.litter_size, h_meta.litter_size],
                                [g_meta.protected_until, h_meta.protected_until],
                                [g_meta.dgen, h_meta.dgen],
                            ),
                            Dir::L => (
                                [h.clone(), g.clone()],
                                [h_meta.origin, g_meta.origin],
                                [h_meta.frame, g_meta.frame],
                                [h_meta.litter, g_meta.litter],
                                [h_meta.litter_size, g_meta.litter_size],
                                [h_meta.protected_until, g_meta.protected_until],
                                [h_meta.dgen, g_meta.dgen],
                            ),
                        };
                    let after: Vec<(u32, u32)> = placed
                        .iter()
                        .map(|&(i, _)| (i, self.arena.stamp(i)))
                        .collect();
                    if self.journal.len() >= self.params.journal_len {
                        self.journal.pop_front();
                    }
                    self.journal.push_back(UndoEntry {
                        before,
                        dir,
                        pivot,
                        after,
                        event: ev,
                        origins,
                        frames,
                        litters,
                        litter_sizes,
                        protected_until,
                        gens,
                        misses: 0,
                    });
                }
                CrossTrace {
                    changed: true,
                    crossed: true,
                    shot_descendants,
                }
            }
        }
    }

    // Case-split a conjunction on a uniformly random wire it does not touch:
    // R -> xR, !xR. Injects dependence on a wire the gate never read — the
    // entropy move fsplit structurally lacks (its splits only use collision-
    // forced wires). The sibling pair trivially re-merges (DropLit), hence the
    // event tabu.
    fn fresh_split_move(&mut self) {
        let id = self.arena.random_linked(&mut self.rng);
        let g = self.arena.gate(id).clone();
        if g.comp || g.width() + 1 > self.params.k_max {
            self.counters.blocked_width += 1;
            return;
        }
        if !self.admits_rewrite_size(1, 2) {
            self.counters.blocked_size_cap += 1;
            return;
        }
        if !self.split_allowed(g.width() + 1) {
            self.counters.declined += 1;
            return;
        }
        let mut x = None;
        for _ in 0..16 {
            let w = self.rng.random_range(0..self.num_wires) as u16;
            if w != g.target && !g.reads(w) {
                x = Some(w);
                break;
            }
        }
        let Some(x) = x else { return };
        let mk = |pol: bool| {
            XGate::conj(g.target, g.ctrls.iter().copied().chain([(x, pol)]))
                .expect("fresh wire cannot contradict")
        };
        let pieces = vec![mk(true), mk(false)];
        if self.params.local_verify {
            assert!(
                rules::verify_rewrite(std::slice::from_ref(&g), &pieces),
                "fresh split verification failed: {g:?} on wire {x}"
            );
        }
        let m = self.meta_of(id);
        let ev = self.fresh_event();
        for p in &pieces {
            self.counters.width_hist[p.width().min(15)] += 1;
        }
        let ids = self.splice_replace_one(id, pieces);
        for &pid in &ids {
            let d = self.child_dir(m.dir);
            self.set_meta(
                pid,
                Meta {
                    origin: m.origin,
                    event: ev,
                    dir: d,
                    frame: m.frame,
                    protected_until: m.protected_until,
                    litter: m.litter,
                    litter_size: m.litter_size,
                    dgen: self.child_gen(m.dgen),
                },
            );
        }
        self.advance_births(&ids);
        self.counters.fresh_splits += 1;
    }

    // Inverse of the Subsume merge: !lR -> R, lR (random literal). Count +1,
    // widths bounded by the original.
    fn unsub_move(&mut self) {
        let id = self.arena.random_linked(&mut self.rng);
        let g = self.arena.gate(id).clone();
        if g.comp || g.width() == 0 {
            return;
        }
        if !self.admits_rewrite_size(1, 2) {
            self.counters.blocked_size_cap += 1;
            return;
        }
        if !self.split_allowed(g.width()) {
            self.counters.declined += 1;
            return;
        }
        let j = self.rng.random_range(0..g.ctrls.len());
        let (w, p) = g.ctrls[j];
        let without = XGate::conj(g.target, g.ctrls_without(w)).expect("subset is satisfiable");
        let flipped = XGate::conj(
            g.target,
            g.ctrls
                .iter()
                .map(|&(cw, cp)| if cw == w { (cw, !cp) } else { (cw, cp) }),
        )
        .expect("polarity flip is satisfiable");
        let _ = p;
        let pieces = if self.rng.random_bool(0.5) {
            vec![without, flipped]
        } else {
            vec![flipped, without]
        };
        if self.params.local_verify {
            assert!(
                rules::verify_rewrite(std::slice::from_ref(&g), &pieces),
                "unsubsume verification failed: {g:?}"
            );
        }
        let m = self.meta_of(id);
        let ev = self.fresh_event();
        for x in &pieces {
            self.counters.width_hist[x.width().min(15)] += 1;
        }
        let ids = self.splice_replace_one(id, pieces);
        for &pid in &ids {
            let d = self.child_dir(m.dir);
            self.set_meta(
                pid,
                Meta {
                    origin: m.origin,
                    event: ev,
                    dir: d,
                    frame: m.frame,
                    protected_until: m.protected_until,
                    litter: m.litter,
                    litter_size: m.litter_size,
                    dgen: self.child_gen(m.dgen),
                },
            );
        }
        self.advance_births(&ids);
        self.counters.unsubs += 1;
    }

    // Insert an adjacent identity pair of a FRESH random conjunction: width
    // uniform in [1, k_max], random distinct wires, random polarities. Width 1
    // is admitted (welded material is ~transcript-neutral across width — the
    // damper's survival exponential cancels the firing discount — so there is
    // no reason to exclude the cheapest, most weldable class). The two copies
    // get opposite directions and each is immediately shot once (one cross
    // move per copy), so the pair separates directionally; each sub-step is
    // independently function-preserving.
    fn insert_move(&mut self) {
        if !self.admits_rewrite_size(0, 2) {
            self.counters.blocked_size_cap += 1;
            return;
        }
        let kmax = self.params.k_max.min(self.num_wires.saturating_sub(1));
        if kmax < 1 {
            return;
        }
        let k = self.rng.random_range(1..=kmax);
        let mut wires: Vec<u16> = Vec::with_capacity(k + 1);
        while wires.len() < k + 1 {
            let w = self.rng.random_range(0..self.num_wires) as u16;
            if !wires.contains(&w) {
                wires.push(w);
            }
        }
        let lits: Vec<(u16, bool)> = wires[1..]
            .iter()
            .map(|&w| (w, self.rng.random_bool(0.5)))
            .collect();
        let g = XGate::conj(wires[0], lits).expect("distinct wires cannot contradict");
        if self.params.local_verify {
            assert!(
                rules::verify_rewrite(&[], &[g.clone(), g.clone()]),
                "insert pair is not an identity: {g:?}"
            );
        }
        let pos = self.arena.random_linked(&mut self.rng);
        let a = self.arena.insert_after(pos, g.clone());
        self.index_add(a);
        let b = self.arena.insert_after(a, g);
        self.index_add(b);
        let ev = self.fresh_event();
        let da = self.rand_dir();
        let minted_litter = self.fresh_litter();
        self.set_meta(
            a,
            Meta {
                origin: ORIGIN_SYNTH,
                event: ev,
                dir: da,
                frame: 0,
                protected_until: 0,
                // Born-random material: a fresh singleton litter per gate.
                litter: minted_litter,
                litter_size: 1,
                dgen: GEN_FRESH,
            },
        );
        let minted_litter = self.fresh_litter();
        self.set_meta(
            b,
            Meta {
                origin: ORIGIN_SYNTH,
                event: ev,
                dir: da.opposite(),
                frame: 0,
                protected_until: 0,
                // Born-random material: a fresh singleton litter per gate.
                litter: minted_litter,
                litter_size: 1,
                dgen: GEN_FRESH,
            },
        );
        self.counters.width_hist[self.arena.gate(a).width().min(15)] += 2;
        self.counters.inserts += 1;
        // Each copy is shot once as part of the insert.
        self.cross_move_on(a);
        if self.arena.is_linked(b) {
            self.cross_move_on(b);
        }
    }

    // Open a persistent nonlinear frame without introducing a new rewrite
    // axiom. Construct a broad low-degree reversible packet P, insert the exact
    // identity P . P^-1, and then move material from its two halves apart using
    // the same exhaustively checked R-rule crossings as the ordinary walk.
    //
    // P contains many active width-2/3 controlled-X gates rather than a single
    // high-width conjunction: its components fire on 1/4 or 1/8 of random
    // states, so a flat affine diagnostic cannot be obtained merely through a
    // vanishingly rare perturbation. Since every XGate is an involution,
    // reverse(P) is exactly P^-1. Each subsequent crossing is independently
    // function preserving; together they turn the initially trivial identity
    // into a spatially distributed nonlinear state frame.
    fn nonlinear_frame_move(&mut self) {
        self.counters.nl_frame_attempts += 1;
        if self.arena.len() == 0 {
            self.counters.nl_frame_skips += 1;
            return;
        }
        let min_width = self.params.nl_frame_min_width;
        let max_width = self
            .params
            .nl_frame_max_width
            .min(self.params.k_max)
            .min(self.num_wires.saturating_sub(1));
        let packet_len = self.params.nl_frame_packet_gates;
        if packet_len == 0 || min_width < 2 || min_width > max_width {
            self.counters.nl_frame_skips += 1;
            return;
        }
        let Some(frame_gates) = packet_len.checked_mul(2) else {
            self.counters.blocked_size_cap += 1;
            self.counters.nl_frame_skips += 1;
            return;
        };
        if !self.admits_rewrite_size(0, frame_gates) {
            self.counters.blocked_size_cap += 1;
            self.counters.nl_frame_skips += 1;
            return;
        }

        // Cycle shuffled targets so the packet covers many carrier directions;
        // controls are freshly sampled for every gate and never include target.
        let all_wires: Vec<u16> = (0..self.num_wires).map(|w| w as u16).collect();
        let mut targets = all_wires.clone();
        targets.shuffle(&mut self.rng);
        let mut packet: Vec<XGate> = Vec::with_capacity(packet_len);
        for i in 0..packet_len {
            if i > 0 && i % targets.len() == 0 {
                targets.shuffle(&mut self.rng);
            }
            let target = targets[i % targets.len()];
            let width = self.rng.random_range(min_width..=max_width);
            let mut controls: Vec<u16> =
                all_wires.iter().copied().filter(|&w| w != target).collect();
            controls.shuffle(&mut self.rng);
            let lits = controls[..width]
                .iter()
                .map(|&w| (w, self.rng.random_bool(0.5)));
            packet.push(XGate::conj(target, lits).expect("distinct nonlinear frame literals"));
        }
        debug_assert!(packet.iter().all(|g| (2..=max_width).contains(&g.width())));

        let ev = self.fresh_event();
        let protected_until = self.moves_done.saturating_add(self.params.nl_frame_tenure);
        let frame = ev;
        let pos = self.arena.random_linked(&mut self.rng);
        let mut anchor = pos;
        let mut left_frontier: Vec<u32> = Vec::with_capacity(packet_len);
        for gate in &packet {
            self.counters.width_hist[gate.width().min(15)] += 1;
            anchor = self.arena.insert_after(anchor, gate.clone());
            self.index_add(anchor);
            let minted_litter = self.fresh_litter();
            self.set_meta(
                anchor,
                Meta {
                    origin: ORIGIN_SYNTH,
                    event: ev,
                    dir: Dir::L,
                    frame,
                    protected_until,
                    // Born-random material: a fresh singleton litter per gate.
                    litter: minted_litter,
                    litter_size: 1,
                    dgen: GEN_FRESH,
                },
            );
            left_frontier.push(anchor);
        }
        let mut right_frontier: Vec<u32> = Vec::with_capacity(packet_len);
        for gate in packet.iter().rev() {
            self.counters.width_hist[gate.width().min(15)] += 1;
            anchor = self.arena.insert_after(anchor, gate.clone());
            self.index_add(anchor);
            let minted_litter = self.fresh_litter();
            self.set_meta(
                anchor,
                Meta {
                    origin: ORIGIN_SYNTH,
                    event: ev,
                    dir: Dir::R,
                    frame,
                    protected_until,
                    // Born-random material: a fresh singleton litter per gate.
                    litter: minted_litter,
                    litter_size: 1,
                    dgen: GEN_FRESH,
                },
            );
            right_frontier.push(anchor);
        }
        // Structural proof of the inserted identity: each right-half gate is
        // the corresponding left-half gate in reverse order.
        if self.params.local_verify {
            let ids = left_frontier
                .iter()
                .chain(right_frontier.iter())
                .copied()
                .collect::<Vec<_>>();
            let inserted = ids
                .iter()
                .map(|&id| self.arena.gate(id))
                .collect::<Vec<_>>();
            for i in 0..packet_len {
                assert_eq!(
                    inserted[i],
                    inserted[2 * packet_len - 1 - i],
                    "nonlinear frame packet is not P followed by reverse(P)"
                );
            }
        }

        self.counters.nl_frames += 1;
        self.counters.nl_frame_packet_gates += (2 * packet_len) as u64;
        for shot in 0..self.params.nl_frame_shots {
            let prefer_left = shot % 2 == 0;
            let moved = if prefer_left {
                if left_frontier.is_empty() {
                    self.nonlinear_frame_shot(&mut right_frontier, Dir::R, frame)
                } else {
                    self.nonlinear_frame_shot(&mut left_frontier, Dir::L, frame)
                }
            } else {
                if right_frontier.is_empty() {
                    self.nonlinear_frame_shot(&mut left_frontier, Dir::L, frame)
                } else {
                    self.nonlinear_frame_shot(&mut right_frontier, Dir::R, frame)
                }
            };
            if !moved && left_frontier.is_empty() && right_frontier.is_empty() {
                break;
            }
        }

        let mut first = None;
        let mut last = 0usize;
        let mut nodes = 0usize;
        for (i, id) in self.arena.ids_in_order().into_iter().enumerate() {
            if self.meta_of(id).frame == frame {
                first.get_or_insert(i);
                last = i;
                nodes += 1;
            }
        }
        if let Some(first) = first {
            self.counters.nl_frame_span += (last - first + 1) as u64;
            self.counters.nl_frame_nodes += nodes as u64;
        }
    }

    fn nonlinear_frame_shot(&mut self, frontier: &mut Vec<u32>, dir: Dir, frame: u64) -> bool {
        while !frontier.is_empty() {
            let j = self.rng.random_range(0..frontier.len());
            let id = frontier.swap_remove(j);
            if !self.arena.is_linked(id) || self.meta_of(id).frame != frame {
                continue;
            }
            let mut meta = self.meta_of(id);
            meta.dir = dir;
            self.set_meta(id, meta);
            self.counters.nl_frame_shot_attempts += 1;
            let trace = self.cross_move_on(id);
            if !trace.changed {
                self.counters.nl_frame_blocked += 1;
                return false;
            }
            if trace.crossed {
                self.counters.nl_frame_crossings += 1;
            } else {
                self.counters.nl_frame_preparatory_rewrites += 1;
            }
            for child in trace.shot_descendants {
                if self.arena.is_linked(child) && self.meta_of(child).frame == frame {
                    let mut child_meta = self.meta_of(child);
                    child_meta.dir = dir;
                    self.set_meta(child, child_meta);
                    frontier.push(child);
                }
            }
            return true;
        }
        false
    }

    // Conjugation twist — the SAMF mechanism from ssg, XGate-native. Pick a
    // window W and an involution P (a wire negation; a wire swap realized as
    // 3 CNOTs; or a transvection x_a ^= x_b, a single CNOT — the affine,
    // non-isometric rung) and rewrite
    //
    //     P . (P W P) . P     ==  W
    //
    // conjugating every interior gate in place and bracketing the window with
    // one P-packet per side. Every interior STATE of the window becomes its
    // image under P while the function and everything outside stay exactly
    // unchanged: the edit is bounded (+2 or +6 gates, widths and comps
    // preserved, K-cap respected) but the trajectory effect is global — it is
    // the move that collapses the prefix-progress diagonal, which no
    // support-local move can touch. Interior gates keep their node, position
    // and provenance (no material moves — only the basis); the in-place
    // rewrite bumps arena stamps, so undo-journal entries over relabeled
    // pieces correctly die. Window lengths are log-uniform over
    // [twist_min_len, circuit size] to decorrelate progress at all scales.
    fn solve_g57_seam(&mut self, edge: u32, direction: Dir, a: u16, b: u16) -> TgSeam {
        let engine = swap_words::engine();
        let mut ids = Vec::new();
        let mut support = vec![a, b];
        let mut current = self.arena.neighbor(edge, direction);
        while ids.len() < TG_CONTEXT && current != NIL {
            // Preserve the newer frame/protection contracts: adaptive seam
            // absorption never consumes tagged or protected material.
            if self.is_protected(current) || self.meta_of(current).frame != 0 {
                break;
            }
            let gate = self.arena.gate(current);
            let mut next_support = support.clone();
            for wire in std::iter::once(gate.target).chain(gate.ctrls.iter().map(|&(wire, _)| wire))
            {
                if !next_support.contains(&wire) {
                    next_support.push(wire);
                }
            }
            if next_support.len() > swap_words::ABS_WIRES {
                break;
            }
            support = next_support;
            ids.push(current);
            current = self.arena.neighbor(current, direction);
        }

        let mut wires = [a, b, a, a];
        for index in 2..swap_words::ABS_WIRES {
            if let Some(&wire) = support.get(index) {
                wires[index] = wire;
            } else {
                let offset = self.rng.random_range(0..self.num_wires);
                for delta in 0..self.num_wires {
                    let wire = ((offset + delta) % self.num_wires) as u16;
                    if !wires[..index].contains(&wire) {
                        wires[index] = wire;
                        break;
                    }
                }
            }
        }
        let abstract_wire = |wire: u16| {
            wires
                .iter()
                .position(|&candidate| candidate == wire)
                .expect("seam support was bound") as u8
        };
        let permutations: Vec<u64> = ids
            .iter()
            .map(|&id| {
                let gate = self.arena.gate(id);
                let controls: Vec<(u8, bool)> = gate
                    .ctrls
                    .iter()
                    .map(|&(wire, polarity)| (abstract_wire(wire), polarity))
                    .collect();
                swap_words::xgate_perm(abstract_wire(gate.target), &controls, gate.comp)
            })
            .collect();
        let mut best = (0usize, engine.bare_word().to_vec());
        let started = std::time::Instant::now();
        for count in 1..=permutations.len() {
            let mut target = match direction {
                Dir::L => {
                    let mut context = swap_words::IDENT;
                    for index in (0..count).rev() {
                        context = swap_words::compose(context, permutations[index]);
                    }
                    swap_words::compose(context, engine.s_ab)
                }
                Dir::R => engine.s_ab,
            };
            if direction == Dir::R {
                for permutation in permutations.iter().take(count) {
                    target = swap_words::compose(target, *permutation);
                }
            }
            self.counters.tg_solves += 1;
            if let Some(word) = engine.solve(target, swap_words::MAX_WORD) {
                let net = word.len() as isize - count as isize;
                let best_net = best.1.len() as isize - best.0 as isize;
                if net < best_net || (net == best_net && count > best.0) {
                    best = (count, word);
                }
            }
        }
        self.counters.tg_solve_ns += started.elapsed().as_nanos() as u64;
        TgSeam {
            edge,
            ids: ids[..best.0].to_vec(),
            replacement: engine.decode(&best.1, &wires),
        }
    }

    /// Interior edge positions farther out whose next gate is a width-two
    /// complemented gate pinning both twist wires. Sliding to one merely
    /// extends the conjugated window over the traversed gates, while giving
    /// the all-G57 bracket a high-probability one-gate attachment target.
    fn g57_slide_candidates(&self, from: u32, direction: Dir, a: u16, b: u16) -> Vec<u32> {
        let mut candidates = Vec::new();
        let mut edge = from;
        for _ in 0..TG_SLIDE_CAP {
            edge = self.arena.neighbor(edge, direction);
            if edge == NIL {
                break;
            }
            let outside = self.arena.neighbor(edge, direction);
            if outside == NIL {
                break;
            }
            // Never propose an attachment that solve_g57_seam is forbidden to
            // consume under the current frame/protection contracts.
            if self.is_protected(outside) || self.meta_of(outside).frame != 0 {
                continue;
            }
            let gate = self.arena.gate(outside);
            if gate.comp && gate.ctrls.len() == 2 {
                let pins = [gate.target, gate.ctrls[0].0, gate.ctrls[1].0];
                if pins.contains(&a) && pins.contains(&b) {
                    candidates.push(edge);
                    if candidates.len() >= TG_SLIDE_TRIES {
                        break;
                    }
                }
            }
        }
        candidates
    }

    /// Evaluate a seam at its drawn edge and, if it remains the bare six-gate
    /// word, retry at a few promising outward slide positions. The first +4
    /// attachment is already optimal for a one-gate context, while a deeper
    /// context may improve further and remains eligible.
    fn eval_g57_seam(&mut self, edge: u32, direction: Dir, a: u16, b: u16) -> (TgSeam, u64) {
        let mut best = self.solve_g57_seam(edge, direction, a, b);
        let mut slides = 0u64;
        if tg_slide_on() && best.net_growth() >= 6 {
            for candidate_edge in self.g57_slide_candidates(edge, direction, a, b) {
                let candidate = self.solve_g57_seam(candidate_edge, direction, a, b);
                if candidate.net_growth() < best.net_growth() {
                    let attached = candidate.net_growth() <= 4;
                    best = candidate;
                    slides = 1;
                    if attached {
                        break;
                    }
                }
            }
        }
        (best, slides)
    }

    /// Adaptive all-G57 realization of a pure-swap conjugation bracket. Bare
    /// seams may slide outward to an attachment gate, and both ends are scored
    /// jointly across several candidate windows before any edit is committed.
    fn twist_move_g57(&mut self) {
        if self.arena.len() < 2 || self.num_wires < swap_words::ABS_WIRES {
            self.counters.twist_skips += 1;
            return;
        }

        // Evaluate complete two-seam plans without changing the arena. The
        // first plan at the joint acceptance threshold wins; otherwise commit
        // the cheapest plan seen over all redraws.
        let attempts = if tg_retry_on() { TG_RETRIES } else { 1 };
        let mut draws = 0u64;
        let mut best: Option<TgPlan> = None;
        for _ in 0..attempts {
            draws += 1;
            let n = self.arena.len();
            let minimum = self.params.twist_min_len.max(2).min(n) as f64;
            let len = (self
                .rng
                .random_range(minimum.ln()..=(n as f64).ln())
                .exp()
                .round() as usize)
                .clamp(2, n);
            let draw = self.rng.random_range(0..n + len - 1);
            let (start, len) = if draw < len - 1 {
                (self.arena.head(), draw + 1)
            } else {
                (self.arena.random_linked(&mut self.rng), len)
            };
            let mut touched = Vec::new();
            let mut seen = vec![false; self.num_wires];
            let mut end = start;
            let mut span = 0usize;
            let mut current = start;
            while current != NIL && span < len {
                let gate = self.arena.gate(current);
                for wire in
                    std::iter::once(gate.target).chain(gate.ctrls.iter().map(|&(wire, _)| wire))
                {
                    if !seen[wire as usize] {
                        seen[wire as usize] = true;
                        touched.push(wire);
                    }
                }
                end = current;
                span += 1;
                current = self.arena.neighbor(current, Dir::R);
            }
            if touched.is_empty() {
                continue;
            }

            // Boundary-supported pairs are much likelier to absorb a real
            // neighbor than a uniformly random second wire. Keep one uniform
            // pair so fresh-wire routing remains available.
            let mut pairs = Vec::new();
            for outside in [
                self.arena.neighbor(start, Dir::L),
                self.arena.neighbor(end, Dir::R),
            ] {
                if outside == NIL {
                    continue;
                }
                let gate = self.arena.gate(outside);
                let pins: Vec<u16> = std::iter::once(gate.target)
                    .chain(gate.ctrls.iter().map(|&(wire, _)| wire))
                    .collect();
                for &a in &pins {
                    if !seen[a as usize] {
                        continue;
                    }
                    for &b in &pins {
                        if a != b && !pairs.contains(&(a, b)) {
                            pairs.push((a, b));
                        }
                    }
                }
            }
            pairs.truncate(6);
            let a = touched[self.rng.random_range(0..touched.len())];
            for _ in 0..16 {
                let b = self.rng.random_range(0..self.num_wires) as u16;
                if a != b {
                    if !pairs.contains(&(a, b)) {
                        pairs.push((a, b));
                    }
                    break;
                }
            }
            if pairs.is_empty() {
                continue;
            }

            let mut pair_best: Option<TgPlan> = None;
            for (a, b) in pairs {
                let plan = TgPlan {
                    a,
                    b,
                    left: self.solve_g57_seam(start, Dir::L, a, b),
                    right: self.solve_g57_seam(end, Dir::R, a, b),
                    slides: 0,
                };
                if pair_best
                    .as_ref()
                    .is_none_or(|previous| plan.score() < previous.score())
                {
                    pair_best = Some(plan);
                }
            }
            let mut plan = pair_best.expect("at least one wire pair");
            if plan.left.net_growth() >= 6 {
                let (candidate, slid) = self.eval_g57_seam(start, Dir::L, plan.a, plan.b);
                if candidate.net_growth() < plan.left.net_growth() {
                    plan.left = candidate;
                    plan.slides += slid;
                }
            }
            if plan.right.net_growth() >= 6 {
                let (candidate, slid) = self.eval_g57_seam(end, Dir::R, plan.a, plan.b);
                if candidate.net_growth() < plan.right.net_growth() {
                    plan.right = candidate;
                    plan.slides += slid;
                }
            }
            if best
                .as_ref()
                .is_none_or(|previous| plan.score() < previous.score())
            {
                best = Some(plan);
            }
            if best.as_ref().expect("a plan was evaluated").score().0 <= TG_ACCEPT_NET {
                break;
            }
        }
        let Some(plan) = best else {
            self.counters.twist_skips += 1;
            return;
        };
        self.counters.tg_retries += draws.saturating_sub(1);
        self.counters.tg_slides += plan.slides;
        let bucket = |net: isize| net.clamp(0, 7) as usize;
        self.counters.tg_net_hist[bucket(plan.left.net_growth())] += 1;
        self.counters.tg_net_hist[bucket(plan.right.net_growth())] += 1;

        let removed = plan.left.ids.len() + plan.right.ids.len();
        let added = plan.left.replacement.len() + plan.right.replacement.len();
        if !self.admits_rewrite_size(removed, added) {
            self.counters.blocked_size_cap += 1;
            self.counters.twist_skips += 1;
            return;
        }

        let (a, b) = (plan.a, plan.b);
        let swap_packet = vec![XGate::cnot(b, a), XGate::cnot(a, b), XGate::cnot(b, a)];
        let mut relabeled = 0u64;
        let mut span_walked = 0u64;
        let mut current = plan.left.edge;
        loop {
            let is_last = current == plan.right.edge;
            span_walked += 1;
            let next = self.arena.neighbor(current, Dir::R);
            let gate = self.arena.gate(current).clone();
            if let Some(replacement) = conj_by_swap(&gate, a, b) {
                if self.params.local_verify {
                    let lhs: Vec<XGate> = swap_packet
                        .iter()
                        .cloned()
                        .chain(std::iter::once(gate))
                        .chain(swap_packet.iter().cloned())
                        .collect();
                    assert!(rules::verify_rewrite(
                        &lhs,
                        std::slice::from_ref(&replacement)
                    ));
                }
                let mut meta = self.meta_of(current);
                self.index_remove(current);
                self.arena.replace_gate(current, replacement);
                self.index_add(current);
                meta.dgen = self.child_gen(meta.dgen);
                self.set_meta(current, meta);
                relabeled += 1;
            }
            if is_last {
                break;
            }
            current = next;
        }
        if self.params.local_verify {
            let mut old_left: Vec<XGate> = plan
                .left
                .ids
                .iter()
                .rev()
                .map(|&id| self.arena.gate(id).clone())
                .collect();
            old_left.extend(swap_packet.iter().cloned());
            assert!(rules::verify_rewrite(&old_left, &plan.left.replacement));
            let mut old_right = swap_packet.clone();
            old_right.extend(plan.right.ids.iter().map(|&id| self.arena.gate(id).clone()));
            assert!(rules::verify_rewrite(&old_right, &plan.right.replacement));
        }

        let left_anchor = plan.left.ids.last().map_or_else(
            || self.arena.neighbor(plan.left.edge, Dir::L),
            |&far| self.arena.neighbor(far, Dir::L),
        );
        for &id in plan.left.ids.iter().chain(&plan.right.ids) {
            self.index_remove(id);
            self.arena.unlink(id);
            self.arena.free_node(id);
        }
        let event = self.fresh_event();
        let mut inserted = Vec::with_capacity(added);
        let mut anchor = left_anchor;
        // Seam consumption mints one fresh litter PER SIDE, exactly as a DB
        // splice would (ORIG:4226-4242).
        let left_litter = self.fresh_litter();
        let left_size = plan.left.replacement.len().min(u16::MAX as usize) as u16;
        for gate in &plan.left.replacement {
            self.counters.width_hist[gate.width().min(15)] += 1;
            anchor = self.arena.insert_after(anchor, gate.clone());
            self.index_add(anchor);
            self.set_meta(
                anchor,
                Meta {
                    origin: ORIGIN_SYNTH,
                    event,
                    dir: Dir::L,
                    frame: 0,
                    protected_until: 0,
                    litter: left_litter,
                    litter_size: left_size,
                    dgen: GEN_FRESH,
                },
            );
            inserted.push(anchor);
        }
        let mut anchor = plan.right.edge;
        let right_litter = self.fresh_litter();
        let right_size = plan.right.replacement.len().min(u16::MAX as usize) as u16;
        for gate in &plan.right.replacement {
            self.counters.width_hist[gate.width().min(15)] += 1;
            anchor = self.arena.insert_after(anchor, gate.clone());
            self.index_add(anchor);
            self.set_meta(
                anchor,
                Meta {
                    origin: ORIGIN_SYNTH,
                    event,
                    dir: Dir::R,
                    frame: 0,
                    protected_until: 0,
                    litter: right_litter,
                    litter_size: right_size,
                    dgen: GEN_FRESH,
                },
            );
            inserted.push(anchor);
        }
        self.advance_births(&inserted);
        self.counters.twist_swaps += 1;
        self.counters.twist_span += span_walked;
        self.counters.twist_relabels += relabeled;
        self.counters.tg_consumed += removed as u64;
        self.counters.tg_emitted += added as u64;
    }

    /// Layer-1 first-class twist: swap two wires and independently negate the
    /// two outputs. The polarized three-CNOT packet implements
    /// `(a,b) -> (b ^ alpha, a ^ beta)`; reversing it is the exact inverse.
    fn twist_swap_family_move(&mut self) {
        let n = self.arena.len();
        if n < 2 || self.num_wires < 2 {
            self.counters.twist_skips += 1;
            return;
        }
        if !self.admits_rewrite_size(0, 6) {
            self.counters.blocked_size_cap += 1;
            self.counters.twist_skips += 1;
            return;
        }
        let minimum = self.params.twist_min_len.max(2).min(n) as f64;
        let len = (self
            .rng
            .random_range(minimum.ln()..=(n as f64).ln())
            .exp()
            .round() as usize)
            .clamp(2, n);
        let draw = self.rng.random_range(0..n + len - 1);
        let (start, len) = if draw < len - 1 {
            (self.arena.head(), draw + 1)
        } else {
            (self.arena.random_linked(&mut self.rng), len)
        };

        let mut seen = vec![false; self.num_wires];
        let mut touched = Vec::new();
        let mut end = start;
        let mut span = 0usize;
        let mut current = start;
        while current != NIL && span < len {
            let gate = self.arena.gate(current);
            for wire in std::iter::once(gate.target).chain(gate.ctrls.iter().map(|&(wire, _)| wire))
            {
                if !seen[wire as usize] {
                    seen[wire as usize] = true;
                    touched.push(wire);
                }
            }
            end = current;
            span += 1;
            current = self.arena.neighbor(current, Dir::R);
        }
        if touched.is_empty() {
            self.counters.twist_skips += 1;
            return;
        }
        let a = touched[self.rng.random_range(0..touched.len())];
        let mut b = a;
        for _ in 0..16 {
            let candidate = self.rng.random_range(0..self.num_wires) as u16;
            if candidate != a {
                b = candidate;
                break;
            }
        }
        if a == b {
            self.counters.twist_skips += 1;
            return;
        }
        let q = self.params.twist_neg_p.clamp(0.0, 1.0);
        let alpha = self.rng.random_bool(q);
        let beta = self.rng.random_bool(q);
        let polarized = |target: u16, control: u16, negate: bool| {
            XGate::conj(target, [(control, !negate)]).expect("distinct swap-family wires")
        };
        let packet = vec![
            polarized(b, a, alpha),
            polarized(a, b, false),
            polarized(b, a, beta),
        ];
        let inverse: Vec<XGate> = packet.iter().rev().cloned().collect();
        if self.params.local_verify {
            let identity: Vec<XGate> = packet.iter().chain(&inverse).cloned().collect();
            assert!(rules::verify_rewrite(&identity, &[]));
        }

        let mut relabeled = 0u64;
        let mut current = start;
        loop {
            let is_last = current == end;
            let next = self.arena.neighbor(current, Dir::R);
            let gate = self.arena.gate(current).clone();
            if let Some(mut replacement) = conj_by_swap(&gate, a, b) {
                if alpha {
                    if let Some(changed) = conj_by_not(&replacement, a) {
                        replacement = changed;
                    }
                }
                if beta {
                    if let Some(changed) = conj_by_not(&replacement, b) {
                        replacement = changed;
                    }
                }
                if self.params.local_verify {
                    let lhs: Vec<XGate> = inverse
                        .iter()
                        .cloned()
                        .chain(std::iter::once(gate.clone()))
                        .chain(packet.iter().cloned())
                        .collect();
                    assert!(
                        rules::verify_rewrite(&lhs, std::slice::from_ref(&replacement)),
                        "swap-family conjugation failed"
                    );
                }
                let mut meta = self.meta_of(current);
                self.index_remove(current);
                self.arena.replace_gate(current, replacement);
                self.index_add(current);
                meta.dgen = self.child_gen(meta.dgen);
                self.set_meta(current, meta);
                relabeled += 1;
            }
            if is_last {
                break;
            }
            current = next;
        }

        let event = self.fresh_event();
        let mut anchor = self.arena.neighbor(start, Dir::L);
        for gate in &packet {
            self.counters.width_hist[gate.width().min(15)] += 1;
            anchor = self.arena.insert_after(anchor, gate.clone());
            self.index_add(anchor);
            let direction = self.rand_dir();
            let minted_litter = self.fresh_litter();
            self.set_meta(
                anchor,
                Meta {
                    origin: ORIGIN_SYNTH,
                    event,
                    dir: direction,
                    frame: 0,
                    protected_until: 0,
                    // Born-random material: a fresh singleton litter per gate.
                    litter: minted_litter,
                    litter_size: 1,
                    dgen: GEN_FRESH,
                },
            );
        }
        let mut anchor = end;
        for gate in &inverse {
            self.counters.width_hist[gate.width().min(15)] += 1;
            anchor = self.arena.insert_after(anchor, gate.clone());
            self.index_add(anchor);
            let direction = self.rand_dir();
            let minted_litter = self.fresh_litter();
            self.set_meta(
                anchor,
                Meta {
                    origin: ORIGIN_SYNTH,
                    event,
                    dir: direction,
                    frame: 0,
                    protected_until: 0,
                    // Born-random material: a fresh singleton litter per gate.
                    litter: minted_litter,
                    litter_size: 1,
                    dgen: GEN_FRESH,
                },
            );
        }
        match (alpha, beta) {
            (false, false) => self.counters.twist_swaps += 1,
            (true, true) => self.counters.twist_cnots += 1,
            _ => self.counters.twist_negs += 1,
        }
        self.counters.twist_span += span as u64;
        self.counters.twist_relabels += relabeled;
    }

    fn twist_move(&mut self, kind: TwistKind) {
        let n = self.arena.len();
        if n < 2 {
            return;
        }
        // Transvection windows are capped at the mid scale: b must be a wire
        // the window never writes, and a window of L random-target gates
        // leaves ~nw*exp(-L/nw) wires unwritten — beyond ~nw*ln(nw) gates the
        // pool is empty and the move could only skip. Neg/swap own the global
        // octaves; cnot frames compose across overlapping mid-scale windows.
        let cap = match kind {
            TwistKind::Cnot => {
                let nw = self.num_wires as f64;
                ((nw * nw.ln()).ceil() as usize).clamp(2, n)
            }
            _ => n,
        };
        let lmin = (self.params.twist_min_len.max(2).min(cap)) as f64;
        let len = (self
            .rng
            .random_range(lmin.ln()..=(cap as f64).ln())
            .exp()
            .round() as usize)
            .clamp(2, cap);
        // Symmetric truncation: the window's virtual start is uniform over
        // [-(len-1), n-1] and the window is clamped to the circuit, so
        // left-overshooting draws pile their opening packets at the head
        // exactly as right-overshoots pile closings at the tail. Uniform
        // start with right-only truncation starves the head of straddling
        // frames (measured 100x head/tail decorrelation asymmetry, fx1tw).
        let (start, len) = {
            let draw = self.rng.random_range(0..n + len - 1);
            if draw < len - 1 {
                (self.arena.head(), draw + 1) // left-truncated: [0, draw+1)
            } else {
                (self.arena.random_linked(&mut self.rng), len)
            }
        };

        // Pass 1: locate the window end (truncated at the tail) and collect
        // the wires it reads / writes / touches, so P can be drawn from wires
        // the window actually uses (a P acting on none of them is a no-op
        // twist) and, for cnot, b from wires it never writes.
        let mut read_seen = vec![false; self.num_wires];
        let mut write_seen = vec![false; self.num_wires];
        let mut touch_seen = vec![false; self.num_wires];
        let mut reads: Vec<u16> = Vec::new();
        let mut touches: Vec<u16> = Vec::new();
        let mut end = start;
        let mut span = 0usize;
        let mut cur = start;
        while cur != NIL && span < len {
            let g = self.arena.gate(cur);
            write_seen[g.target as usize] = true;
            if !touch_seen[g.target as usize] {
                touch_seen[g.target as usize] = true;
                touches.push(g.target);
            }
            for &(w, _) in &g.ctrls {
                if !read_seen[w as usize] {
                    read_seen[w as usize] = true;
                    reads.push(w);
                }
                if !touch_seen[w as usize] {
                    touch_seen[w as usize] = true;
                    touches.push(w);
                }
            }
            end = cur;
            span += 1;
            cur = self.arena.neighbor(cur, Dir::R);
        }

        // The involution P as a gate packet. P == P^-1, so the same packet
        // brackets both sides. Negation needs a wire the window READS (gates
        // targeting w are invariant); a swap needs one touched wire, the other
        // may be any wire — routing the window's material through a fresh
        // physical wire is a legitimate (and strong) relabeling; a cnot
        // transvection needs a read wire a and a wire b the window never
        // WRITES (see conj_by_cnot).
        let cnot = |t: u16, c: u16| XGate::conj(t, [(c, true)]).expect("cnot literal");
        let (pa, pb, packet): (u16, u16, Vec<XGate>) = match kind {
            TwistKind::Swap => {
                if touches.is_empty() {
                    self.counters.twist_skips += 1;
                    return;
                }
                let a = touches[self.rng.random_range(0..touches.len())];
                let mut b = a;
                for _ in 0..16 {
                    let c = self.rng.random_range(0..self.num_wires) as u16;
                    if c != a {
                        b = c;
                        break;
                    }
                }
                if b == a {
                    self.counters.twist_skips += 1;
                    return;
                }
                (a, b, vec![cnot(b, a), cnot(a, b), cnot(b, a)])
            }
            TwistKind::Neg => {
                if reads.is_empty() {
                    self.counters.twist_skips += 1;
                    return;
                }
                let w = reads[self.rng.random_range(0..reads.len())];
                (w, w, vec![XGate::x_gate(w)])
            }
            TwistKind::Cnot => {
                if reads.is_empty() {
                    self.counters.twist_skips += 1;
                    return;
                }
                let a = reads[self.rng.random_range(0..reads.len())];
                let pool: Vec<u16> = (0..self.num_wires as u16)
                    .filter(|&w| !write_seen[w as usize] && w != a)
                    .collect();
                if pool.is_empty() {
                    self.counters.twist_skips += 1;
                    return;
                }
                let b = pool[self.rng.random_range(0..pool.len())];
                (a, b, vec![cnot(a, b)])
            }
        };
        if self.params.local_verify {
            let double: Vec<XGate> = packet.iter().chain(packet.iter()).cloned().collect();
            assert!(
                rules::verify_rewrite(&double, &[]),
                "twist packet is not an involution: {packet:?}"
            );
        }

        // Cnot feasibility: every interior a-reader must be rewritable — a
        // comp gate without a b-literal cannot case-split (Blocked), and a
        // split must respect the K-cap. Checked BEFORE any mutation so a
        // twist either applies whole or not at all.
        let mut case_split_growth = 0usize;
        if kind == TwistKind::Cnot {
            let mut cur = start;
            loop {
                match conj_by_cnot(self.arena.gate(cur), pa, pb) {
                    CnotConj::Blocked => {
                        self.counters.twist_skips += 1;
                        return;
                    }
                    CnotConj::Split(g0, _) => {
                        if g0.width() > self.params.k_max {
                            self.counters.twist_skips += 1;
                            return;
                        }
                        case_split_growth += 1;
                    }
                    _ => {}
                }
                if cur == end {
                    break;
                }
                cur = self.arena.neighbor(cur, Dir::R);
            }
        }
        let Some(bracket_growth) = packet.len().checked_mul(2) else {
            self.counters.blocked_size_cap += 1;
            self.counters.twist_skips += 1;
            return;
        };
        let Some(total_growth) = bracket_growth.checked_add(case_split_growth) else {
            self.counters.blocked_size_cap += 1;
            self.counters.twist_skips += 1;
            return;
        };
        if !self.admits_rewrite_size(0, total_growth) {
            self.counters.blocked_size_cap += 1;
            self.counters.twist_skips += 1;
            return;
        }

        // Pass 2: conjugate every interior gate P acts on. Each per-gate
        // identity P g P == g' (or the split pair) is exhaustively verified;
        // the segment identity then telescopes (P g1 P P g2 P ... = P W P), so
        // together with the bracket packets the window computes exactly W
        // again. Neg/swap rewrite strictly in place (nodes, positions and
        // provenance survive); cnot case-splits replace one node by two at the
        // same position — material still does not move, and the pieces inherit
        // their parent's origin and share a fresh tabu event like any split.
        enum Out {
            Keep,
            One(XGate),
            Two(XGate, XGate),
        }
        let mut relabeled = 0u64;
        let mut case_splits = 0u64;
        let mut w_start = start;
        let mut first = true;
        let mut cur = start;
        let w_end = loop {
            let is_last = cur == end;
            let next = self.arena.neighbor(cur, Dir::R);
            let g = self.arena.gate(cur).clone();
            let out = match kind {
                TwistKind::Neg => conj_by_not(&g, pa).map_or(Out::Keep, Out::One),
                TwistKind::Swap => conj_by_swap(&g, pa, pb).map_or(Out::Keep, Out::One),
                TwistKind::Cnot => match conj_by_cnot(&g, pa, pb) {
                    CnotConj::Invariant => Out::Keep,
                    CnotConj::Flip(x) => Out::One(x),
                    CnotConj::Split(x, y) => Out::Two(x, y),
                    CnotConj::Blocked => unreachable!("feasibility scan admits no Blocked gate"),
                },
            };
            let (head, tail) = match out {
                Out::Keep => (cur, cur),
                Out::One(g2) => {
                    let mut m = self.meta_of(cur);
                    if self.params.local_verify {
                        let mut seq = packet.clone();
                        seq.push(g.clone());
                        seq.extend(packet.iter().cloned());
                        assert!(
                            rules::verify_rewrite(&seq, std::slice::from_ref(&g2)),
                            "twist conjugation failed: {g:?} under {packet:?}"
                        );
                    }
                    self.index_remove(cur);
                    self.arena.replace_gate(cur, g2);
                    self.index_add(cur);
                    m.dgen = self.child_gen(m.dgen);
                    self.set_meta(cur, m);
                    relabeled += 1;
                    (cur, cur)
                }
                Out::Two(g0, g1) => {
                    // The two halves fire on disjoint b-slices, so they
                    // commute: emit in random order.
                    let pieces = if self.rng.random_bool(0.5) {
                        vec![g0, g1]
                    } else {
                        vec![g1, g0]
                    };
                    if self.params.local_verify {
                        let mut seq = packet.clone();
                        seq.push(g.clone());
                        seq.extend(packet.iter().cloned());
                        assert!(
                            rules::verify_rewrite(&seq, &pieces),
                            "twist case-split failed: {g:?} under {packet:?}"
                        );
                    }
                    let m = self.meta_of(cur);
                    let ev = self.fresh_event();
                    for x in &pieces {
                        self.counters.width_hist[x.width().min(15)] += 1;
                    }
                    let ids = self.splice_replace_one(cur, pieces);
                    for &pid in &ids {
                        // Twist material does not move: pieces keep the
                        // conjugated gate's direction exactly.
                        self.set_meta(
                            pid,
                            Meta {
                                origin: m.origin,
                                event: ev,
                                dir: m.dir,
                                frame: m.frame,
                                protected_until: m.protected_until,
                                litter: m.litter,
                                litter_size: m.litter_size,
                                dgen: self.child_gen(m.dgen),
                            },
                        );
                    }
                    relabeled += 1;
                    case_splits += 1;
                    (ids[0], ids[1])
                }
            };
            if first {
                w_start = head;
                first = false;
            }
            if is_last {
                break tail;
            }
            cur = next;
        };

        // Bracket the window: P before start, P after end. Packet gates are
        // fresh synthetic material (no origin) sharing one event, so the
        // trivial bracket-cancel is tabu like any fresh sibling pair. They are
        // NOT scattered — they must sit tight on the window edges; later churn
        // is free to float or merge them (every such move is independently
        // function-preserving).
        let ev = self.fresh_event();
        let mut anchor = self.arena.neighbor(w_start, Dir::L);
        for g in &packet {
            self.counters.width_hist[g.width().min(15)] += 1;
            anchor = self.arena.insert_after(anchor, g.clone());
            self.index_add(anchor);
            let d = self.rand_dir();
            let minted_litter = self.fresh_litter();
            self.set_meta(
                anchor,
                Meta {
                    origin: ORIGIN_SYNTH,
                    event: ev,
                    dir: d,
                    frame: 0,
                    protected_until: 0,
                    // Born-random material: a fresh singleton litter per gate.
                    litter: minted_litter,
                    litter_size: 1,
                    dgen: GEN_FRESH,
                },
            );
        }
        let mut anchor = w_end;
        for g in &packet {
            self.counters.width_hist[g.width().min(15)] += 1;
            anchor = self.arena.insert_after(anchor, g.clone());
            self.index_add(anchor);
            let d = self.rand_dir();
            let minted_litter = self.fresh_litter();
            self.set_meta(
                anchor,
                Meta {
                    origin: ORIGIN_SYNTH,
                    event: ev,
                    dir: d,
                    frame: 0,
                    protected_until: 0,
                    // Born-random material: a fresh singleton litter per gate.
                    litter: minted_litter,
                    litter_size: 1,
                    dgen: GEN_FRESH,
                },
            );
        }

        match kind {
            TwistKind::Neg => self.counters.twist_negs += 1,
            TwistKind::Swap => self.counters.twist_swaps += 1,
            TwistKind::Cnot => self.counters.twist_cnots += 1,
        }
        self.counters.twist_span += span as u64;
        self.counters.twist_relabels += relabeled;
        self.counters.twist_case_splits += case_splits;
    }

    // ---- the contraction moves ----

    // Reverse a recorded crossing: sample journal entries until a live one is
    // found (dead ones — any piece touched since — are discarded), gather its
    // pieces back around the pivot by floating, verify the block against the
    // original pair exhaustively, and splice [g, h] back in.
    fn undo_move(&mut self) -> bool {
        for _ in 0..8 {
            if self.journal.is_empty() {
                return false;
            }
            let i = self.rng.random_range(0..self.journal.len());
            let alive = self.journal[i]
                .after
                .iter()
                .all(|&(id, st)| self.arena.is_linked(id) && self.arena.stamp(id) == st);
            if !alive {
                self.journal.swap_remove_back(i);
                self.counters.undo_dead += 1;
                continue;
            }
            if self.journal[i]
                .after
                .iter()
                .any(|&(id, _)| self.is_protected(id))
            {
                self.counters.nl_frame_protected_blocks += 1;
                continue;
            }
            if self.is_tabu(self.journal[i].event) {
                self.counters.undo_tabu += 1;
                continue;
            }
            let e = self
                .journal
                .swap_remove_back(i)
                .expect("journal index valid");
            if let Some(mut e) = self.try_undo(e) {
                // Gather miss: pieces only floated (function-preserving), the
                // entry is still valid — retry later, but only a few times
                // (a blocked entry usually stays blocked).
                e.misses += 1;
                if e.misses < 3 {
                    self.journal.push_back(e);
                }
                return false;
            }
            return true;
        }
        false
    }

    // Returns the entry back on a gather miss, None on success.
    fn try_undo(&mut self, e: UndoEntry) -> Option<UndoEntry> {
        let (mut edge_l, mut edge_r) = (e.pivot, e.pivot);
        for &(id, _) in &e.after {
            if id == e.pivot {
                continue;
            }
            // Locate the piece relative to the current block, then float it
            // onto the block's edge. Pieces pairwise commute (same target, no
            // reads of it), so ungathered siblings never block the float.
            let mut side = None;
            let mut cur = self.arena.neighbor(edge_r, Dir::R);
            let mut steps = 0usize;
            while cur != NIL && steps < self.params.merge_reach {
                if cur == id {
                    side = Some(Dir::R);
                    break;
                }
                cur = self.arena.neighbor(cur, Dir::R);
                steps += 1;
            }
            if side.is_none() {
                let mut cur = self.arena.neighbor(edge_l, Dir::L);
                let mut steps = 0usize;
                while cur != NIL && steps < self.params.merge_reach {
                    if cur == id {
                        side = Some(Dir::L);
                        break;
                    }
                    cur = self.arena.neighbor(cur, Dir::L);
                    steps += 1;
                }
            }
            let Some(side) = side else {
                self.counters.undo_gather_miss += 1;
                return Some(e);
            };
            let (anchor, ok) = match side {
                Dir::R => {
                    self.float_until(id, Dir::L, edge_r);
                    (id, self.arena.neighbor(edge_r, Dir::R) == id)
                }
                Dir::L => {
                    self.float_until(id, Dir::R, edge_l);
                    (id, self.arena.neighbor(edge_l, Dir::L) == id)
                }
            };
            if !ok {
                self.counters.undo_gather_miss += 1;
                return Some(e);
            }
            match side {
                Dir::R => edge_r = anchor,
                Dir::L => edge_l = anchor,
            }
        }
        // Contiguous block [edge_l ..= edge_r] now holds exactly the pieces.
        let mut block: Vec<u32> = Vec::with_capacity(e.after.len());
        let mut cur = edge_l;
        loop {
            block.push(cur);
            if cur == edge_r {
                break;
            }
            cur = self.arena.neighbor(cur, Dir::R);
        }
        debug_assert_eq!(
            block.len(),
            e.after.len(),
            "gathered block is not contiguous"
        );
        if self.params.local_verify {
            let actual: Vec<XGate> = block.iter().map(|&b| self.arena.gate(b).clone()).collect();
            assert!(
                rules::verify_rewrite(&actual, &e.before),
                "undo verification failed: block {actual:?} != {:?}",
                e.before
            );
        }
        let cursor = self.arena.neighbor(edge_l, Dir::L);
        for &bid in &block {
            self.index_remove(bid);
            self.arena.unlink(bid);
            self.arena.free_node(bid);
        }
        let mut c = cursor;
        let mut new_ids: Vec<u32> = Vec::with_capacity(2);
        for (j, gate) in e.before.iter().enumerate() {
            c = self.arena.insert_after(c, gate.clone());
            self.index_add(c);
            let d = self.rand_dir();
            self.set_meta(
                c,
                Meta {
                    origin: e.origins[j],
                    event: 0,
                    dir: d,
                    frame: e.frames[j],
                    protected_until: e.protected_until[j],
                    // Reversing a crossing also reverses its provenance.
                    litter: e.litters[j],
                    litter_size: e.litter_sizes[j],
                    dgen: e.gens[j],
                },
            );
            new_ids.push(c);
        }
        self.counters.undos += 1;
        None
    }

    // Partner ids for `g_id` from the index: same-key gates (cancel / xfuse /
    // drop-lit) plus each one-wire-reduced key (subsume, larger side). The
    // smaller subsume partner finds the pair when IT is sampled as the larger
    // one — initiators are uniform, so coverage is symmetric.
    fn merge_candidates(&self, g_id: u32) -> Vec<u32> {
        let g = self.arena.gate(g_id);
        let mut out: Vec<u32> = Vec::new();
        let consider = |ids: &[u32], out: &mut Vec<u32>| {
            for &c in ids {
                if c != g_id && merge_result(g, self.arena.gate(c)).is_some() {
                    out.push(c);
                }
            }
        };
        if let Some(ids) = self.index.get(&key_of(g)) {
            consider(ids, &mut out);
        }
        for skip in 0..g.ctrls.len() {
            let k = merge_key(
                g.target,
                g.ctrls
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != skip)
                    .map(|(_, &(w, _))| w),
            );
            if let Some(ids) = self.index.get(&k) {
                consider(ids, &mut out);
            }
        }
        out
    }

    fn merge_move(&mut self) -> bool {
        // Sample initiators until one has index partners; most gates are
        // ladder-form with unique keys, so a direct random pick rarely works.
        let mut found: Option<(u32, Vec<u32>)> = None;
        for _ in 0..8 {
            let g_id = self.arena.random_linked(&mut self.rng);
            let cands = self.merge_candidates(g_id);
            if !cands.is_empty() {
                found = Some((g_id, cands));
                break;
            }
        }
        let Some((g_id, cands)) = found else {
            self.counters.merge_no_partner += 1;
            return false;
        };
        // Far partners are usually wall-blocked (over a long span some gate
        // almost surely collides with both), so scan outward from the
        // initiator and take the NEAREST reachable candidate, tracking the
        // initiator's colliders incrementally for the wall check.
        let cand_set: std::collections::HashSet<u32> = cands.into_iter().collect();
        let g = self.arena.gate(g_id).clone();
        let mut chosen: Option<(u32, Dir, usize)> = None;
        for dir in [Dir::R, Dir::L] {
            let mut g_colliders: Vec<u32> = Vec::new();
            let mut cur = self.arena.neighbor(g_id, dir);
            let mut steps = 0usize;
            while cur != NIL && steps < self.params.merge_reach {
                if chosen.is_some_and(|(_, _, d)| steps >= d) {
                    break; // the other direction already found a nearer one
                }
                if cand_set.contains(&cur) {
                    let hg = self.arena.gate(cur);
                    let wall = g_colliders
                        .iter()
                        .any(|&b| XGate::collides(self.arena.gate(b), hg));
                    if wall {
                        self.counters.merge_wall_blocked += 1;
                    } else {
                        chosen = Some((cur, dir, steps));
                        break;
                    }
                }
                if XGate::collides(&g, self.arena.gate(cur)) {
                    g_colliders.push(cur);
                }
                cur = self.arena.neighbor(cur, dir);
                steps += 1;
            }
        }
        let Some((h_id, dir, _)) = chosen else {
            self.counters.merge_too_far += 1;
            return false;
        };
        let (mg, mh) = (self.meta_of(g_id), self.meta_of(h_id));
        if self.is_protected(g_id) || self.is_protected(h_id) {
            self.counters.nl_frame_protected_blocks += 1;
            return false;
        }
        let sibling = mg.event != 0 && mg.event == mh.event;
        if sibling && self.is_tabu(mg.event) {
            self.counters.tabu_blocked += 1;
            return false;
        }
        if !self.bring_adjacent(g_id, h_id, dir) {
            self.counters.merge_not_adjacent += 1;
            return false;
        }
        let g = self.arena.gate(g_id).clone();
        let h = self.arena.gate(h_id).clone();
        let merged = merge_result(&g, &h).expect("candidate was mergeable");
        let out = merged.gates();
        if self.params.local_verify {
            assert!(
                rules::verify_rewrite(&[g.clone(), h.clone()], &out),
                "merge verification failed: {g:?} + {h:?}"
            );
        }
        // Splice: replace the adjacent pair by the merged gate (or nothing).
        // Same-target gates commute, so their order is irrelevant.
        let left = if self.arena.neighbor(g_id, Dir::R) == h_id {
            g_id
        } else {
            h_id
        };
        let cursor = self.arena.neighbor(left, Dir::L);
        self.index_remove(g_id);
        self.index_remove(h_id);
        self.arena.unlink(g_id);
        self.arena.unlink(h_id);
        self.arena.free_node(g_id);
        self.arena.free_node(h_id);
        let mut new_ids: Vec<u32> = Vec::new();
        let mut c = cursor;
        for gate in out {
            c = self.arena.insert_after(c, gate);
            self.index_add(c);
            new_ids.push(c);
        }
        let origin = if mg.origin == mh.origin {
            mg.origin
        } else {
            self.counters.merges_cross_origin += 1;
            mg.origin
        };
        let frame = if mg.frame == mh.frame {
            mg.frame
        } else if mg.frame == 0 {
            mh.frame
        } else if mh.frame == 0 {
            mg.frame
        } else {
            // Material from two distinct, expired frames no longer has a
            // single meaningful frame interval for coverage telemetry.
            0
        };
        let protected_until = mg.protected_until.max(mh.protected_until);
        let dgen = mg.dgen.min(mh.dgen);
        // The litter follows the same parent the generation does.
        let (litter, litter_size) = if mg.dgen <= mh.dgen {
            (mg.litter, mg.litter_size)
        } else {
            (mh.litter, mh.litter_size)
        };
        for &nid in &new_ids {
            // Merged output stays in place (scatter is suspended) and keeps
            // shooting the way the initiating parent was headed, per dir_p.
            let d = self.child_dir(mg.dir);
            self.set_meta(
                nid,
                Meta {
                    origin,
                    event: 0,
                    dir: d,
                    frame,
                    protected_until,
                    litter,
                    litter_size,
                    dgen,
                },
            );
        }
        if sibling {
            self.counters.merges_sibling += 1;
        }
        match merged {
            Merge::Cancel => self.counters.merges_cancel += 1,
            Merge::XFuse(_) => self.counters.merges_xfuse += 1,
            Merge::DropLit(_) => self.counters.merges_drop += 1,
            Merge::Subsume(_) => self.counters.merges_subsume += 1,
        }
        true
    }

    // ---- DB replacement moves ----

    fn note_db_result(&mut self, result: &DbResult, outgoing_len: usize) {
        self.counters.db_identity_skips += result.identity_skipped as u64;
        if result.chosen.is_some() {
            self.note_choice(result.choice_count);
            if result.chosen_curated {
                self.counters.db_curated_hits += 1;
            }
        }
        if let Some(minimum) = result.min_match_len {
            self.counters.dmin_windows += 1;
            if minimum < outgoing_len {
                self.counters.dmin_shorter += 1;
            }
        }
    }

    fn db_attempt(&mut self, mode: DbMode) -> bool {
        let previous_mode = self.db_mode_cur;
        self.db_mode_cur = mode;
        self.db_seed_home = None;
        self.seed_from_pool = false;
        self.seed_fell_through = false;
        let spliced = self.db_attempt_inner(mode);
        // Fall-through is a REBUILD-RATE signal, not a stop signal: a heads
        // coin that found the pool drained says the rescan cadence is too
        // slow, not that the material is unspellable.
        if self.seed_fell_through {
            self.counters.canary_fallthrough += 1;
        }
        // The canary counts only rounds that genuinely came from the pool,
        // and SLEEPS under the brake: COMP declines far more often by
        // construction, so Compressing rounds are excluded.
        if self.seed_from_pool && mode != DbMode::Compressing {
            let w = self.params.canary_window.max(1);
            if self.canary.len() == w {
                if self.canary.pop_front() == Some(true) {
                    self.canary_failures -= 1;
                }
            }
            self.canary.push_back(!spliced);
            if !spliced {
                self.canary_failures += 1;
            }
        }
        if !spliced {
            self.restore_seed();
        }
        self.db_mode_cur = previous_mode;
        spliced
    }

    /// Fires iff armed (`canary_theta > 0`), the ring holds a full window
    /// (the minimum-sample guard), and the failure fraction strictly exceeds
    /// theta. Denominated in qualifying rounds, not moves.
    fn canary_fired(&self) -> bool {
        self.params.canary_theta > 0.0
            && self.canary.len() >= self.params.canary_window.max(1)
            && self.canary_frac() > self.params.canary_theta
    }

    fn canary_frac(&self) -> f64 {
        if self.canary.is_empty() {
            return 0.0;
        }
        self.canary_failures as f64 / self.canary.len() as f64
    }

    fn db_attempt_inner(&mut self, mode: DbMode) -> bool {
        if self.db.is_none() {
            return false;
        }
        let count = self.arena.len();
        let minimum = self.active_db_minimum();
        let maximum = self.active_s_db().max(minimum);
        if count < minimum {
            return false;
        }
        let len = if self.params.db_prefixes || minimum == maximum {
            maximum.min(count)
        } else {
            self.rng.random_range(minimum..=maximum.min(count))
        };
        let Some((ids, first_direction, sampler)) = self.sample_best_window(len) else {
            return false;
        };
        self.counters.litter_windows += 1;
        self.counters.litter_distinct_sum += self.litter_census(&ids).0 as u64;
        self.db_last_sampler = sampler;
        let window: Vec<XGate> = ids.iter().map(|&id| self.arena.gate(id).clone()).collect();
        let guard = DegreeGuard {
            max_degree: self.params.db_max_degree,
            probes: self.params.db_degree_probes,
        };

        if self.params.db_prefixes {
            let curated_armed =
                self.params.curated && (mode != DbMode::Compressing || self.params.curated_in_comp);
            let passes: &[(bool, bool)] = if curated_armed && self.params.curated_exhaust {
                &[(true, false), (false, true)]
            } else if curated_armed {
                &[(true, true)]
            } else {
                &[(false, true)]
            };
            // Seed-preserving descent (ORIG mix.rs:4697-4860): shrink from
            // whichever end is farther from the seed so a pool-drawn gate
            // survives to the final rung — a fixed-end descent abandons the
            // seed on about half of contiguous windows and on every convex
            // one. Every exit path of the loop body MUST shrink lo or hi;
            // anything else respins the identical rung forever and hangs the
            // run before it completes a single move.
            let k = self
                .db_seed_home
                .and_then(|(seed, _)| ids.iter().position(|&id| id == seed))
                .unwrap_or(0);
            for &(curated, regular_fallback) in passes {
                let (mut lo, mut hi) = (0usize, window.len());
                while hi - lo >= minimum {
                    // Shrink from the end farther from the seed (hi is
                    // exclusive, so the right end sits at hi - 1).
                    macro_rules! shrink {
                        () => {
                            if k.saturating_sub(lo) >= (hi - 1).saturating_sub(k) {
                                lo += 1;
                            } else {
                                hi -= 1;
                            }
                        };
                    }
                    let prefix_ids = &ids[lo..hi];
                    let prefix = &window[lo..hi];
                    if self.params.litter_ban && self.litter_census(prefix_ids).1 {
                        self.counters.litter_banned += 1;
                        // Descending past it is free — a shorter rung is no
                        // longer a complete litter.
                        shrink!();
                        continue;
                    }
                    self.counters.db_attempts += 1;
                    if prefix_ids.iter().any(|&id| self.is_protected(id)) {
                        self.counters.db_protected_skips += 1;
                        self.record_db_attempt(prefix, 0, None);
                        self.count_db_miss(mode);
                        shrink!();
                        continue;
                    }
                    if self.params.db_max_span > 0
                        && super::xpoly::xgate_used_wires(prefix).len() > self.params.db_max_span
                    {
                        self.counters.db_span_skips += 1;
                        self.record_db_attempt(prefix, 0, None);
                        self.count_db_miss(mode);
                        shrink!();
                        continue;
                    }
                    let result = db_replace(
                        prefix,
                        self.num_wires,
                        self.db.as_ref().expect("DB checked above"),
                        self.db_budget,
                        mode,
                        guard,
                        curated,
                        true,
                        regular_fallback,
                        self.params.mix_pay_random,
                        &mut self.rng,
                    );
                    self.note_db_result(&result, prefix.len());
                    if result.degree_skipped {
                        self.counters.db_degree_skips += 1;
                    }
                    let Some(replacement) = result.chosen else {
                        self.record_db_attempt(prefix, result.match_count, None);
                        self.count_db_miss(mode);
                        shrink!();
                        continue;
                    };
                    if self.params.db_dry_run {
                        self.record_db_attempt(prefix, result.match_count, None);
                        self.count_db_hit(mode);
                        shrink!();
                        continue;
                    }
                    if self.try_db_splice(
                        prefix_ids,
                        first_direction,
                        prefix,
                        replacement,
                        result.match_count,
                        mode,
                    ) {
                        return true;
                    }
                    // Fifth shrink site: a refused splice (frame conflict,
                    // hard-cap refusal, wide-verify skip, stale window) must
                    // also shrink, or the refusal is deterministic and the
                    // loop never exits.
                    shrink!();
                }
            }
            return false;
        }

        // Non-prefix path: no rung to descend to, so the ban refuses outright.
        if self.params.litter_ban && self.litter_census(&ids).1 {
            self.counters.litter_banned += 1;
            return false;
        }
        self.counters.db_attempts += 1;
        if ids.iter().any(|&id| self.is_protected(id)) {
            self.counters.db_protected_skips += 1;
            self.record_db_attempt(&window, 0, None);
            self.count_db_miss(mode);
            return false;
        }
        if self.params.db_max_span > 0
            && super::xpoly::xgate_used_wires(&window).len() > self.params.db_max_span
        {
            self.counters.db_span_skips += 1;
            self.record_db_attempt(&window, 0, None);
            self.count_db_miss(mode);
            return false;
        }
        let result = db_replace(
            &window,
            self.num_wires,
            self.db.as_ref().expect("DB checked above"),
            self.db_budget,
            mode,
            guard,
            self.params.curated,
            self.params.curated_in_comp,
            true,
            self.params.mix_pay_random,
            &mut self.rng,
        );
        self.note_db_result(&result, window.len());
        if result.degree_skipped {
            self.counters.db_degree_skips += 1;
        }
        if self.params.db_dry_run {
            self.record_db_attempt(&window, result.match_count, None);
            if result.match_count > 0 {
                self.count_db_hit(mode);
            } else {
                self.count_db_miss(mode);
            }
            return false;
        }
        let Some(replacement) = result.chosen else {
            self.record_db_attempt(&window, result.match_count, None);
            self.count_db_miss(mode);
            return false;
        };
        self.try_db_splice(
            &ids,
            first_direction,
            &window,
            replacement,
            result.match_count,
            mode,
        )
    }

    fn count_db_hit(&mut self, mode: DbMode) {
        match mode {
            DbMode::Compressing => self.counters.db_comp_hits += 1,
            DbMode::SizeAgnostic | DbMode::MinGrow | DbMode::Mix => self.counters.db_agn_hits += 1,
        }
    }

    fn count_db_miss(&mut self, mode: DbMode) {
        match mode {
            DbMode::Compressing => self.counters.db_comp_misses += 1,
            DbMode::SizeAgnostic | DbMode::MinGrow | DbMode::Mix => {
                self.counters.db_agn_misses += 1
            }
        }
    }

    fn try_db_splice(
        &mut self,
        ids: &[u32],
        first_direction: Dir,
        window: &[XGate],
        replacement: Vec<XGate>,
        match_count: usize,
        mode: DbMode,
    ) -> bool {
        if ids.is_empty()
            || ids.len() != window.len()
            || ids.iter().any(|&id| !self.arena.is_linked(id))
            || ids
                .windows(2)
                .any(|pair| self.arena.neighbor(pair[0], Dir::R) != pair[1])
            || ids
                .iter()
                .zip(window)
                .any(|(&id, gate)| self.arena.gate(id) != gate)
        {
            self.record_db_attempt(window, match_count, None);
            self.count_db_miss(mode);
            return false;
        }
        if ids.iter().any(|&id| self.is_protected(id)) {
            self.counters.db_protected_skips += 1;
            self.record_db_attempt(window, match_count, None);
            self.count_db_miss(mode);
            return false;
        }
        let protected_until = ids
            .iter()
            .map(|&id| self.meta_of(id).protected_until)
            .max()
            .unwrap_or(0);
        let mut folded_frame: Option<u64> = None;
        let mut frame_conflict = false;
        let mut has_g57_frame = false;
        for frame in ids
            .iter()
            .map(|&id| self.meta_of(id).frame)
            .filter(|&frame| frame != 0)
        {
            has_g57_frame |= decode_g57_pair_frame(frame).is_some();
            match folded_frame {
                None => folded_frame = Some(frame),
                Some(existing) if existing == frame => {}
                Some(_) => frame_conflict = true,
            }
        }
        if frame_conflict && has_g57_frame {
            self.counters.db_frame_skips += 1;
            self.record_db_attempt(window, match_count, None);
            self.count_db_miss(mode);
            return false;
        }
        let frame = if frame_conflict {
            0
        } else {
            folded_frame.unwrap_or(0)
        };
        let old = window.len();
        let new = replacement.len();
        if new > old && self.params.hard_size_cap > 0 {
            let final_size = self
                .arena
                .len()
                .checked_sub(old)
                .and_then(|base| base.checked_add(new));
            if final_size.is_none_or(|size| size > self.params.hard_size_cap) {
                self.counters.db_cap_skips += 1;
                self.record_db_attempt(window, match_count, None);
                self.count_db_miss(mode);
                return false;
            }
        }
        if self.params.db_verify {
            let mut support: Vec<u16> = window
                .iter()
                .chain(replacement.iter())
                .flat_map(|gate| {
                    std::iter::once(gate.target).chain(gate.ctrls.iter().map(|&(wire, _)| wire))
                })
                .collect();
            support.sort_unstable();
            support.dedup();
            if support.len() > 24 {
                self.counters.db_wide_skip += 1;
                self.record_db_attempt(window, match_count, None);
                self.count_db_miss(mode);
                return false;
            }
            assert!(
                rules::verify_rewrite(window, &replacement),
                "DB replacement verification failed: {window:?} -> {replacement:?}"
            );
        }

        self.record_db_attempt(window, match_count, Some(&replacement));
        if new <= old {
            self.counters.db_gates_removed += (old - new) as u64;
        } else {
            self.counters.db_gates_added += (new - old) as u64;
        }
        match mode {
            DbMode::Compressing => {
                self.counters.db_cmp_removed += old.saturating_sub(new) as u64;
                self.counters.db_cmp_added += new.saturating_sub(old) as u64;
            }
            DbMode::SizeAgnostic | DbMode::MinGrow | DbMode::Mix => {
                self.counters.db_mix_removed += old.saturating_sub(new) as u64;
                self.counters.db_mix_added += new.saturating_sub(old) as u64;
            }
        }
        self.count_db_hit(mode);

        let cursor = self.arena.neighbor(ids[0], Dir::L);
        let first_meta = self.meta_of(ids[0]);
        let same_origin = ids
            .iter()
            .all(|&id| self.meta_of(id).origin == first_meta.origin);
        let origin = if same_origin {
            first_meta.origin
        } else {
            ORIGIN_SYNTH
        };
        let dgen = self.db_replacement_generation(ids, self.params.gen_median_low);
        // The splice is the litter-creating event: every product carries one
        // fresh id, with the litter size fixed at creation.
        if self.litter_census(ids).1 {
            self.counters.litter_full_spliced += 1;
        }
        let litter = self.fresh_litter();

        for &id in ids {
            self.index_remove(id);
            self.arena.unlink(id);
            self.arena.free_node(id);
        }
        let incoming = replacement.len();
        let litter_size = incoming.min(u16::MAX as usize) as u16;
        let pivot = if first_direction == Dir::L {
            (2 * incoming) / 3
        } else {
            incoming / 3
        };
        let mut anchor = cursor;
        let mut inserted = Vec::with_capacity(replacement.len());
        for (index, gate) in replacement.into_iter().enumerate() {
            anchor = self.arena.insert_after(anchor, gate);
            self.index_add(anchor);
            inserted.push(anchor);
            self.set_meta(
                anchor,
                Meta {
                    origin,
                    event: 0,
                    dir: if index <= pivot { Dir::L } else { Dir::R },
                    frame,
                    protected_until,
                    litter,
                    litter_size,
                    dgen,
                },
            );
        }
        if self.params.db_advance {
            self.advance_births(&inserted);
        }
        true
    }

    fn record_db_attempt(
        &mut self,
        window: &[XGate],
        matches: usize,
        replacement: Option<&[XGate]>,
    ) {
        use std::io::Write;
        let sampler = match self.db_last_sampler {
            DbSample::Convex => "cvx",
            DbSample::Contiguous | DbSample::Mixed => "ctg",
        };
        let Some(writer) = self.db_record.as_mut() else {
            return;
        };
        fn format_gates(gates: &[XGate]) -> String {
            gates
                .iter()
                .map(|gate| {
                    let controls = gate
                        .ctrls
                        .iter()
                        .map(|&(wire, polarity)| {
                            format!("{wire}{}", if polarity { "+" } else { "-" })
                        })
                        .collect::<Vec<_>>()
                        .join(",");
                    format!("{}:{}:{controls}", gate.target, gate.comp as u8)
                })
                .collect::<Vec<_>>()
                .join(" ")
        }
        let _ = writeln!(
            writer,
            "attempt mv={} matches={} replaced={} smp={}",
            self.moves_done,
            matches,
            replacement.is_some() as u8,
            sampler
        );
        let _ = writeln!(writer, "  out {}", format_gates(window));
        if let Some(replacement) = replacement {
            let _ = writeln!(writer, "  in  {}", format_gates(replacement));
        }
    }

    // ---- floating (same semantics as the fsplit engine) ----

    fn float_distance(&self, id: u32, dir: Dir, cap: usize) -> usize {
        let g = self.arena.gate(id);
        let mut cur = self.arena.neighbor(id, dir);
        let mut d = 0usize;
        while cur != NIL && d < cap && !XGate::collides(g, self.arena.gate(cur)) {
            d += 1;
            cur = self.arena.neighbor(cur, dir);
        }
        d
    }

    fn float_to_collision(&mut self, id: u32, dir: Dir) -> usize {
        self.float_until(id, dir, NIL)
    }

    // Slide `id` in `dir` past non-colliders, stopping early if the next node is
    // `stop`. Needed for merges: merge partners never collide (same target), so
    // an unbounded float would sail straight past the partner.
    fn float_until(&mut self, id: u32, dir: Dir, stop: u32) -> usize {
        let g = self.arena.gate(id).clone();
        let mut last = NIL;
        let mut cur = self.arena.neighbor(id, dir);
        let mut steps = 0usize;
        while cur != NIL && cur != stop && !XGate::collides(&g, self.arena.gate(cur)) {
            last = cur;
            steps += 1;
            cur = self.arena.neighbor(cur, dir);
        }
        if steps > 0 {
            self.arena.unlink(id);
            match dir {
                Dir::R => self.arena.link_after(id, last),
                Dir::L => self.arena.link_before(id, last),
            }
            self.counters.floats += 1;
            self.counters.float_steps += steps as u64;
        }
        steps
    }

    // Float g toward h (stopping at h), then h toward g (stopping at g).
    // Adjacent afterwards iff nothing colliding with both sat between them AND
    // the two one-directional floats suffice (interleaved blockers can still
    // prevent a meet; that is a harmless miss).
    fn bring_adjacent(&mut self, g_id: u32, h_id: u32, dir: Dir) -> bool {
        self.float_until(g_id, dir, h_id);
        if self.arena.neighbor(g_id, dir) == h_id {
            return true;
        }
        self.float_until(h_id, dir.opposite(), g_id);
        self.arena.neighbor(g_id, dir) == h_id
    }

    fn float_uniform(&mut self, id: u32) -> usize {
        let dl = self.float_distance(id, Dir::L, usize::MAX);
        let dr = self.float_distance(id, Dir::R, usize::MAX);
        if dl + dr == 0 {
            return 0;
        }
        let off = self.rng.random_range(0..=(dl + dr));
        let (dir, k) = if off < dl {
            (Dir::L, dl - off)
        } else {
            (Dir::R, off - dl)
        };
        if k == 0 {
            return 0;
        }
        let mut anchor = id;
        for _ in 0..k {
            anchor = self.arena.neighbor(anchor, dir);
        }
        self.arena.unlink(id);
        match dir {
            Dir::L => self.arena.link_before(id, anchor),
            Dir::R => self.arena.link_after(id, anchor),
        }
        k
    }

    fn rand_dir(&mut self) -> Dir {
        if self.rng.random_bool(0.5) {
            Dir::L
        } else {
            Dir::R
        }
    }

    // Fragment direction law: inherit the shot gate's direction with
    // probability dir_p, else the opposite.
    fn child_dir(&mut self, shot: Dir) -> Dir {
        if self.rng.random_bool(self.params.dir_p) {
            shot
        } else {
            shot.opposite()
        }
    }

    // Directional birth transport (replaces the uniform scatter): a fresh
    // piece advances floor(dir_q * slack) gates in its own direction, where
    // slack is how far it could float that way before its first collision.
    fn advance_birth(&mut self, id: u32) {
        if !self.arena.is_linked(id) {
            return;
        }
        let dir = self.meta_of(id).dir;
        let slack = self.float_distance(id, dir, usize::MAX);
        let k = (self.params.dir_q * slack as f64).floor() as usize;
        if k == 0 {
            return;
        }
        let mut anchor = id;
        for _ in 0..k {
            anchor = self.arena.neighbor(anchor, dir);
        }
        self.arena.unlink(id);
        match dir {
            Dir::L => self.arena.link_before(id, anchor),
            Dir::R => self.arena.link_after(id, anchor),
        }
        self.counters.scatters += 1;
        self.counters.scatter_steps += k as u64;
    }

    fn advance_births(&mut self, ids: &[u32]) {
        for &id in ids {
            self.advance_birth(id);
        }
    }

    // Failed cross: the shot gate does not stay parked at the collision — it
    // retreats floor((1 - dir_q) * way) of the way it floated in. The path is
    // passable by construction (it just floated through those gates).
    fn retreat(&mut self, id: u32, way: usize, dir: Dir) {
        let k = ((1.0 - self.params.dir_q) * way as f64).floor() as usize;
        if k == 0 {
            return;
        }
        let back = dir.opposite();
        let mut anchor = id;
        for _ in 0..k {
            anchor = self.arena.neighbor(anchor, back);
        }
        self.arena.unlink(id);
        match back {
            Dir::L => self.arena.link_before(id, anchor),
            Dir::R => self.arena.link_after(id, anchor),
        }
        self.counters.floats += 1;
        self.counters.float_steps += k as u64;
    }

    // ---- DB window sampling and generation targeting ----

    fn width_of(&self, id: u32) -> usize {
        self.arena.gate(id).width()
    }

    fn span_collides(&self, lo: u32, hi: u32, candidate: u32) -> bool {
        let candidate_gate = self.arena.gate(candidate);
        let mut current = lo;
        loop {
            if XGate::collides(self.arena.gate(current), candidate_gate) {
                return true;
            }
            if current == hi {
                return false;
            }
            current = self.arena.neighbor(current, Dir::R);
        }
    }

    fn move_across(&mut self, candidate: u32, lo: u32, hi: u32, direction: Dir) {
        self.arena.unlink(candidate);
        match direction {
            Dir::R => self.arena.link_before(candidate, lo),
            Dir::L => self.arena.link_after(candidate, hi),
        }
        self.counters.floats += 1;
        self.counters.float_steps += 1;
    }

    fn span_ids(&self, lo: u32, hi: u32) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut current = lo;
        loop {
            ids.push(current);
            if current == hi {
                break;
            }
            current = self.arena.neighbor(current, Dir::R);
        }
        ids
    }

    /// Draw `litter_samples` candidate windows and keep the one spanning the
    /// most distinct litters. Diversity is the point: a window drawn from many
    /// replacement events is one no single earlier splice can undo, which is
    /// the same property the full-litter ban enforces at the other end.
    fn sample_best_window(&mut self, w: usize) -> Option<(Vec<u32>, Dir, DbSample)> {
        let n = self.params.litter_samples.max(1);
        let mut best: Option<(Vec<u32>, Dir, DbSample)> = None;
        let mut best_distinct = 0usize;
        for _ in 0..n {
            let Some(cand) = self.sample_window(w) else {
                continue;
            };
            let d = self.litter_census(&cand.0).0;
            if best.is_none() || d > best_distinct {
                best_distinct = d;
                best = Some(cand);
            }
            if best_distinct >= w {
                break; // already maximal; more draws cannot improve it
            }
        }
        best
    }

    fn sample_window(&mut self, width: usize) -> Option<(Vec<u32>, Dir, DbSample)> {
        let convex_probability = self.active_p_convex();
        let sampler = if convex_probability >= 0.0 {
            if self.rng.random_bool(convex_probability.clamp(0.0, 1.0)) {
                DbSample::Convex
            } else {
                DbSample::Contiguous
            }
        } else {
            self.params.db_sample
        };
        match sampler {
            DbSample::Contiguous => self
                .collect_contiguous(width)
                .map(|(ids, direction)| (ids, direction, DbSample::Contiguous)),
            DbSample::Convex => self
                .collect_convex(width)
                .map(|(ids, direction)| (ids, direction, DbSample::Convex)),
            DbSample::Mixed => {
                if self.rng.random_bool(0.5) {
                    self.collect_contiguous(width)
                        .map(|(ids, direction)| (ids, direction, DbSample::Contiguous))
                } else {
                    self.collect_convex(width)
                        .map(|(ids, direction)| (ids, direction, DbSample::Convex))
                }
            }
        }
    }

    /// Rebuild the generation pool: the `pool_k` lowest-generation gates among
    /// those that are pool-eligible AND still below the goal. An O(size) scan,
    /// so it runs on the `gen_rescan` cadence rather than every round.
    fn rebuild_pool(&mut self) {
        let target = self.params.gen_target;
        self.pool.clear();
        let mut cands: Vec<(u32, u32)> = Vec::new();
        for id in self.arena.ids_in_order() {
            let m = self.meta_of(id);
            if m.dgen >= target || !self.pool_eligible(id) {
                continue;
            }
            cands.push((m.dgen, id));
        }
        let k = self.params.pool_k.max(1);
        if cands.len() > k {
            cands.select_nth_unstable(k - 1);
            cands.truncate(k);
        }
        self.pool.extend(cands.into_iter().map(|(_, id)| id));
    }

    /// Draw from the pool, pruning entries that went stale since the rebuild
    /// (freed, re-encoded past the goal, or now too wide to seed).
    fn draw_pool(&mut self) -> Option<u32> {
        let target = self.params.gen_target;
        for _ in 0..8 {
            if self.pool.is_empty() {
                return None;
            }
            let i = self.rng.random_range(0..self.pool.len());
            let id = self.pool[i];
            let m = self.meta_of(id);
            if !self.arena.is_linked(id) || m.dgen >= target || !self.pool_eligible(id) {
                self.pool.swap_remove(i);
                continue;
            }
            return Some(id);
        }
        None
    }

    /// May this gate sit inside the window currently being built? WT addition
    /// to ORIG:5458-5461: framed/protected material is never window-targetable
    /// (the G57 pair census depends on it surviving).
    fn window_eligible(&self, id: u32) -> bool {
        if self.is_protected(id) {
            return false;
        }
        let cap = self.params.w_window;
        cap == 0 || self.width_of(id) < cap
    }

    /// May this gate seed a window and count toward the dose? Stricter than
    /// `window_eligible`: an ineligible gate can never be re-encoded, so its
    /// generation is pinned forever and an unfiltered pool converges on exactly
    /// that set and stays there.
    fn pool_eligible(&self, id: u32) -> bool {
        if self.is_protected(id) {
            return false;
        }
        let cap = self.params.w_pool;
        cap == 0 || self.width_of(id) < cap
    }

    fn pick_seed(&mut self) -> Option<u32> {
        let g = self.pick_seed_inner();
        self.db_seed_home = g.map(|id| (id, self.arena.neighbor(id, Dir::L)));
        g
    }

    /// Put the seed back where it was drawn from. Retracing a float path the
    /// gate already passed through, exactly as `retreat` does for a declined
    /// cross -- the intervening gates commute with it by construction.
    fn restore_seed(&mut self) {
        let Some((id, home)) = self.db_seed_home.take() else {
            return;
        };
        if !self.arena.is_linked(id) || self.arena.neighbor(id, Dir::L) == home {
            return;
        }
        if home != NIL && !self.arena.is_linked(home) {
            return; // its anchor was consumed; leave it where it is
        }
        // Hop back toward home ONE gate at a time, and only over neighbours the
        // seed commutes with. An unchecked relink is wrong here even though the
        // seed floated in along this path: `retreat` may reverse a float
        // immediately, with nothing intervening, but a window build moves OTHER
        // gates too -- ctrl-cap evasion parks a collider out of the way, and an
        // evaded collider is by definition one that does NOT commute. Teleport
        // the seed across that and the circuit's function changes.
        for dir in [Dir::L, Dir::R] {
            for _ in 0..Self::RESTORE_HOPS {
                if self.arena.neighbor(id, Dir::L) == home {
                    return;
                }
                let nb = self.arena.neighbor(id, dir);
                if nb == NIL {
                    break;
                }
                if XGate::collides(self.arena.gate(id), self.arena.gate(nb)) {
                    break; // blocked: leave it here rather than jump the gate
                }
                self.arena.unlink(id);
                match dir {
                    Dir::R => self.arena.link_after(id, nb),
                    Dir::L => self.arena.link_before(id, nb),
                }
            }
        }
    }

    /// Bound on the checked walk home. A seed that cannot get back within this
    /// many hops stays where it is: a partly-restored seed is still correct,
    /// only less tidy.
    const RESTORE_HOPS: usize = 512;

    /// One coin, one pool. Heads (probability `p_mingen`) draws from the
    /// generation pool; tails draws uniformly. `seed_from_pool` records which,
    /// because the canary must count only rounds that genuinely came from the
    /// pool -- a heads round that fell through because the pool had drained is
    /// a DIFFERENT failure (rebuild too slow) with the opposite remedy, and
    /// conflating them would let the brake or a slow rescan look like
    /// unreachable material.
    fn pick_seed_inner(&mut self) -> Option<u32> {
        self.seed_from_pool = false;
        self.seed_fell_through = false;
        if self.params.gen_target > 0
            && self.active_p_mingen() > 0.0
            && self.rng.random_bool(self.active_p_mingen().clamp(0.0, 1.0))
        {
            if let Some(id) = self.draw_pool() {
                self.seed_from_pool = true;
                return Some(id);
            }
            self.seed_fell_through = true;
        }
        for _ in 0..8 {
            let g = self.arena.random_linked(&mut self.rng);
            if self.window_eligible(g) {
                return Some(g);
            }
        }
        None
    }

    const EVADE_BUDGET: usize = 128;

    fn collect_contiguous(&mut self, width: usize) -> Option<(Vec<u32>, Dir)> {
        let first = self.pick_seed()?;
        let first_direction = self.meta_of(first).dir;
        let (mut lo, mut hi) = (first, first);
        let mut count = 1usize;
        let mut direction = first_direction;
        let mut switched = false;
        let mut evade_budget = Self::EVADE_BUDGET;
        while count < width {
            let end = if direction == Dir::R { hi } else { lo };
            let candidate = self.arena.neighbor(end, direction);
            if candidate == NIL {
                if !switched {
                    direction = first_direction.opposite();
                    switched = true;
                    continue;
                }
                break;
            }
            // Evade anything window-ineligible (too wide, or protected framed
            // material) by floating it aside — function-preserving, per
            // ORIG:5582. `< w_window` semantics replace the old `<= ctrl_cap`.
            if !self.window_eligible(candidate) {
                if evade_budget == 0 {
                    self.counters.db_build_aborts += 1;
                    return None;
                }
                evade_budget -= 1;
                if self.float_to_collision(candidate, direction) > 0 {
                    continue;
                }
                if !switched {
                    direction = first_direction.opposite();
                    switched = true;
                    continue;
                }
                return None;
            }
            if direction == Dir::R {
                hi = candidate;
            } else {
                lo = candidate;
            }
            count += 1;
        }
        let ids = self.span_ids(lo, hi);
        (ids.len() >= self.active_db_minimum()).then_some((ids, first_direction))
    }

    fn collect_convex(&mut self, width: usize) -> Option<(Vec<u32>, Dir)> {
        let direction_probability = self.params.db_convex_p;
        let first = self.pick_seed()?;
        let first_direction = self.meta_of(first).dir;
        self.float_to_collision(first, first_direction);
        let (mut lo, mut hi) = (first, first);
        let mut count = 1usize;
        let mut evade_budget = Self::EVADE_BUDGET;
        while count < width {
            let wanted =
                if count == 1 || self.rng.random_bool(direction_probability.clamp(0.0, 1.0)) {
                    first_direction
                } else {
                    first_direction.opposite()
                };
            let (mut candidate, mut direction) = self.float_block_to_collider(lo, hi, wanted);
            if candidate == NIL {
                (candidate, direction) = self.float_block_to_collider(lo, hi, wanted.opposite());
                if candidate == NIL {
                    break;
                }
            }
            if !self.window_eligible(candidate) {
                if evade_budget == 0 {
                    self.counters.db_build_aborts += 1;
                    return None;
                }
                evade_budget -= 1;
                if self.float_to_collision(candidate, direction) > 0 {
                    continue;
                }
                (candidate, direction) = self.float_block_to_collider(lo, hi, direction.opposite());
                if candidate == NIL || !self.window_eligible(candidate) {
                    return None;
                }
            }
            if direction == Dir::R {
                hi = candidate;
            } else {
                lo = candidate;
            }
            count += 1;
        }
        let ids = self.span_ids(lo, hi);
        (ids.len() >= self.active_db_minimum()).then_some((ids, first_direction))
    }

    fn float_block_to_collider(&mut self, lo: u32, hi: u32, direction: Dir) -> (u32, Dir) {
        loop {
            let end = if direction == Dir::R { hi } else { lo };
            let candidate = self.arena.neighbor(end, direction);
            if candidate == NIL {
                return (NIL, direction);
            }
            if self.span_collides(lo, hi, candidate) {
                return (candidate, direction);
            }
            self.move_across(candidate, lo, hi, direction);
        }
    }

    pub fn final_float(&mut self) -> (u64, u64) {
        let ids = self.arena.ids_in_order();
        let (mut moved, mut disp) = (0u64, 0u64);
        for id in ids {
            let k = self.float_uniform(id);
            if k > 0 {
                moved += 1;
                disp += k as u64;
            }
        }
        (moved, disp)
    }

    fn split_allowed(&mut self, c: usize) -> bool {
        let d = self.params.split_damp;
        if c <= d {
            return true;
        }
        let b = self.params.split_base;
        if b <= 1.0 {
            return true;
        }
        let p = b.powi(-(((c - d) as i32).min(1000)));
        self.rng.random_bool(p.min(1.0))
    }

    // ---- splicing (fsplit engine semantics) ----

    fn splice_replace_one(&mut self, id: u32, gates: Vec<XGate>) -> Vec<u32> {
        let mut cursor = self.arena.neighbor(id, Dir::L);
        self.index_remove(id);
        self.arena.unlink(id);
        self.arena.free_node(id);
        let mut ids = Vec::with_capacity(gates.len());
        for g in gates {
            cursor = self.arena.insert_after(cursor, g);
            self.index_add(cursor);
            ids.push(cursor);
        }
        ids
    }

    fn splice_pair(
        &mut self,
        g_id: u32,
        h_id: u32,
        dir: Dir,
        seq: Vec<(XGate, Role)>,
    ) -> Vec<(u32, Role)> {
        let first = match dir {
            Dir::R => g_id,
            Dir::L => h_id,
        };
        let mut cursor = self.arena.neighbor(first, Dir::L);
        self.index_remove(g_id);
        self.arena.unlink(g_id);
        self.arena.unlink(h_id);
        let emitted: Vec<(XGate, Role)> = match dir {
            Dir::R => seq,
            Dir::L => seq.into_iter().rev().collect(),
        };
        let mut h_reused = false;
        let mut out = Vec::with_capacity(emitted.len());
        for (gate, role) in emitted {
            let id = if role == Role::CollidingIntact {
                debug_assert_eq!(&gate, self.arena.gate(h_id));
                self.arena.link_after(h_id, cursor);
                h_reused = true;
                h_id
            } else {
                let nid = self.arena.insert_after(cursor, gate);
                self.index_add(nid);
                nid
            };
            cursor = id;
            out.push((id, role));
        }
        self.arena.free_node(g_id);
        if !h_reused {
            self.index_remove(h_id);
            self.arena.free_node(h_id);
        }
        out
    }

    // ---- verification, metrics, reporting ----

    pub fn global_check(&mut self) {
        assert_eq!(
            self.indexed_count,
            self.arena.len(),
            "merge index drifted from arena (move {})",
            self.moves_done
        );
        let batches = 4;
        for _ in 0..batches {
            let mut st_orig: Vec<u64> = (0..self.num_wires).map(|_| self.rng.random()).collect();
            let mut st_cur = st_orig.clone();
            super::xgate::eval_lanes(&self.original, &mut st_orig);
            let mut cur = self.arena.head();
            while cur != NIL {
                self.arena.gate(cur).apply_lanes(&mut st_cur);
                cur = self.arena.neighbor(cur, Dir::R);
            }
            assert_eq!(
                st_orig, st_cur,
                "FUNCTIONALITY BROKEN: circuit no longer equals the input (move {})",
                self.moves_done
            );
        }
    }

    pub fn remaining_g57(&self) -> usize {
        let mut cur = self.arena.head();
        let mut n = 0usize;
        while cur != NIL {
            if self.arena.gate(cur).comp {
                n += 1;
            }
            cur = self.arena.neighbor(cur, Dir::R);
        }
        n
    }

    pub fn g57_census(&self) -> G57Census {
        let mut census = G57Census::default();
        let mut current = self.arena.head();
        while current != NIL {
            let gate = self.arena.gate(current);
            if gate.comp && gate.ctrls.len() == 2 {
                census.shaped += 1;
                if gate.ctrls[0].1 == gate.ctrls[1].1 {
                    census.same_pol += 1;
                } else {
                    census.opp_pol += 1;
                }
            }
            current = self.arena.neighbor(current, Dir::R);
        }
        census
    }

    pub fn true_g57(&self) -> usize {
        self.g57_census().opp_pol
    }

    // Mean |current position fraction - original position fraction| over gates
    // with a real origin: 0 at the start, drifts toward ~1/3 (the mean for
    // independent uniforms) as positional memory of the original decays.
    pub fn origin_displacement(&self) -> f64 {
        let n = self.arena.len() as f64;
        let m = self.original.len() as f64;
        let (mut acc, mut cnt) = (0.0f64, 0u64);
        let mut cur = self.arena.head();
        let mut i = 0usize;
        while cur != NIL {
            let o = self.meta_of(cur).origin;
            if o != ORIGIN_SYNTH {
                acc += ((i as f64 / n) - (o as f64 / m)).abs();
                cnt += 1;
            }
            i += 1;
            cur = self.arena.neighbor(cur, Dir::R);
        }
        if cnt == 0 { 0.0 } else { acc / cnt as f64 }
    }

    // Mean number of distinct origins in sampled 32-gate windows (max 32):
    // low = original material still clumped, high = well interleaved.
    pub fn window_origin_diversity(&mut self, samples: usize) -> f64 {
        let ids = self.arena.ids_in_order();
        if ids.len() < 32 {
            return 0.0;
        }
        let mut acc = 0.0f64;
        for _ in 0..samples {
            let s = self.metrics_rng.random_range(0..=(ids.len() - 32));
            let mut set: Vec<u32> = ids[s..s + 32]
                .iter()
                .map(|&id| self.meta_of(id).origin)
                .collect();
            set.sort_unstable();
            set.dedup();
            acc += set.len() as f64;
        }
        acc / samples as f64
    }

    // Fraction of gates whose output is never read before its wire is next
    // overwritten (see stats::fanouts).
    pub fn fanout_zero_frac(&self) -> f64 {
        let ids = self.arena.ids_in_order();
        let fan = super::stats::fanouts(ids.iter().map(|&id| self.arena.gate(id)), self.num_wires);
        if fan.is_empty() {
            return 0.0;
        }
        fan.iter().filter(|&&f| f == 0).count() as f64 / fan.len() as f64
    }

    // Mean two-sided float-box size over sampled gates, capped per direction:
    // the mobility / roadblock gauge.
    pub fn mean_leeway(&mut self, samples: usize, cap: usize) -> f64 {
        if self.arena.len() == 0 || samples == 0 {
            return 0.0;
        }
        let mut acc = 0usize;
        for _ in 0..samples {
            let id = self.arena.random_linked(&mut self.metrics_rng);
            acc += self.float_distance(id, Dir::L, cap) + self.float_distance(id, Dir::R, cap);
        }
        acc as f64 / samples as f64
    }

    // Current frame gauge: distinct live frame tags, tagged gate count, and
    // the union fraction of circuit positions lying inside their outermost
    // tagged descendants. This is a delivered-coverage metric, not merely a
    // count of requested frame moves. `protected_only` selects the tenure
    // subset; an expired frame remains active until later rewrites actually
    // merge its tagged descendants away.
    fn nonlinear_frame_stats(&self, protected_only: bool) -> (usize, usize, f64) {
        let ids = self.arena.ids_in_order();
        if ids.is_empty() {
            return (0, 0, 0.0);
        }
        let mut spans: HashMap<u64, (usize, usize)> = HashMap::new();
        let mut gates = 0usize;
        for (i, &id) in ids.iter().enumerate() {
            let m = self.meta_of(id);
            if m.frame == 0
                || decode_g57_pair_frame(m.frame).is_some()
                || (protected_only && self.moves_done >= m.protected_until)
            {
                continue;
            }
            gates += 1;
            spans
                .entry(m.frame)
                .and_modify(|span| span.1 = i)
                .or_insert((i, i));
        }
        let frames = spans.len();
        let mut ranges: Vec<(usize, usize)> = spans.into_values().collect();
        ranges.sort_unstable();
        let mut covered = 0usize;
        let mut current: Option<(usize, usize)> = None;
        for (lo, hi) in ranges {
            match current {
                Some((clo, chi)) if lo <= chi.saturating_add(1) => {
                    current = Some((clo, chi.max(hi)));
                }
                Some((clo, chi)) => {
                    covered += chi - clo + 1;
                    current = Some((lo, hi));
                }
                None => current = Some((lo, hi)),
            }
        }
        if let Some((lo, hi)) = current {
            covered += hi - lo + 1;
        }
        (frames, gates, covered as f64 / ids.len() as f64)
    }

    pub fn active_nonlinear_frame_stats(&self) -> (usize, usize, f64) {
        self.nonlinear_frame_stats(false)
    }

    pub fn protected_nonlinear_frame_stats(&self) -> (usize, usize, f64) {
        self.nonlinear_frame_stats(true)
    }

    /// Census the currently identifiable descendants of seeded G57 identity
    /// pairs. Generic nonlinear frames are deliberately excluded.
    pub fn g57_pair_frame_stats(&self) -> G57PairFrameStats {
        let ids = self.arena.ids_in_order();
        if ids.is_empty() {
            return G57PairFrameStats::default();
        }

        let mut spans: HashMap<u64, (usize, usize)> = HashMap::new();
        let mut pair_copies: HashMap<usize, u8> = HashMap::new();
        let mut tagged_gates = 0usize;
        for (position, &id) in ids.iter().enumerate() {
            let frame = self.meta_of(id).frame;
            let Some((pair_index, copy)) = decode_g57_pair_frame(frame) else {
                continue;
            };
            debug_assert!(copy < 2);
            tagged_gates += 1;
            spans
                .entry(frame)
                .and_modify(|span| span.1 = position)
                .or_insert((position, position));
            *pair_copies.entry(pair_index).or_insert(0) |= 1u8 << copy;
        }

        let distinct_copy_frames = spans.len();
        let distinct_pair_ids = pair_copies.len();
        let complete_pairs = pair_copies
            .values()
            .filter(|&&copies| copies == 0b11)
            .count();
        let mut ranges: Vec<(usize, usize)> = spans.into_values().collect();
        ranges.sort_unstable();
        let mut covered = 0usize;
        let mut current: Option<(usize, usize)> = None;
        for (lo, hi) in ranges {
            match current {
                Some((clo, chi)) if lo <= chi.saturating_add(1) => {
                    current = Some((clo, chi.max(hi)));
                }
                Some((clo, chi)) => {
                    covered += chi - clo + 1;
                    current = Some((lo, hi));
                }
                None => current = Some((lo, hi)),
            }
        }
        if let Some((lo, hi)) = current {
            covered += hi - lo + 1;
        }

        G57PairFrameStats {
            distinct_pair_ids,
            distinct_copy_frames,
            complete_pairs,
            tagged_gates,
            union_span_coverage: covered as f64 / ids.len() as f64,
        }
    }

    pub fn report(&mut self) {
        let disp = self.origin_displacement();
        let owin = self.window_origin_diversity(64);
        let fan0 = self.fanout_zero_frac();
        let leew = self.mean_leeway(256, 4096);
        let origins = self.origins_in_order();
        let odiff = super::stats::origin_diffusion(&origins);
        let oadj = super::stats::adjacent_origin_autocorr(&origins);
        let (nl_live, nl_gates, nl_coverage) = self.active_nonlinear_frame_stats();
        let (nl_protected, nl_protected_gates, nl_protected_coverage) =
            self.protected_nonlinear_frame_stats();
        let generation = self.gen_stats();
        let db_probability = self.params.p_db.clamp(0.0, 1.0);
        let c = &self.counters;
        let hist: Vec<String> = (0..=self.params.k_max.min(15))
            .map(|w| format!("{}:{}", w, c.width_hist[w]))
            .collect();
        println!(
            "[fmix] mv={} size={} target={} comp={} | merges c={} x={} d={} s={} sib={} xorig={} tabu={} nopart={} wall={} far={} noadj={} | undo ok={} dead={} tabu={} miss={} live={} | db p={:.3} comp={}/{} agn={}/{} rm={} add={} attempts={} degree={} span={} wide={} build={} protected={} frame={} cap={} slot2={}/{} add={} brake={}/{} | gen tgt={} G={} Gall={} tgtbl={} alag={}/{} lag={}/{} elig={} wlag={} min={} cov={:.1} canary={:.2} fallthru={} | litter distinct={:.2} full={} ban={} | expand r1={} r2={} r3={} pre={} fresh={} unsub={} ins={} twn={} tws={} twc={} twrel={} twsplit={} twspan={} twskip={} | nl try={} ok={} skip={} packet={} shots={} crossed={} prep={} blocked={} span={} nodes={} protect-blocks={} live={}/{} cover={:.3} protected={}/{} pcover={:.3} | declined={} blockw={} blockcap={} dl={} bnd={} | floats={}/{} scat={}/{} | disp={:.4} owin={:.1} fan0={:.3} leew={:.0} odiff={:.4} oadj={:.4} width[{}]",
            c.moves,
            self.arena.len(),
            self.params.target_size,
            self.remaining_g57(),
            c.merges_cancel,
            c.merges_xfuse,
            c.merges_drop,
            c.merges_subsume,
            c.merges_sibling,
            c.merges_cross_origin,
            c.tabu_blocked,
            c.merge_no_partner,
            c.merge_wall_blocked,
            c.merge_too_far,
            c.merge_not_adjacent,
            c.undos,
            c.undo_dead,
            c.undo_tabu,
            c.undo_gather_miss,
            self.journal.len(),
            db_probability,
            c.db_comp_hits,
            c.db_comp_misses,
            c.db_agn_hits,
            c.db_agn_misses,
            c.db_gates_removed,
            c.db_gates_added,
            c.db_attempts,
            c.db_degree_skips,
            c.db_span_skips,
            c.db_wide_skip,
            c.db_build_aborts,
            c.db_protected_skips,
            c.db_frame_skips,
            c.db_cap_skips,
            c.db_slot2_hits,
            c.db_slot2_rounds,
            c.db_slot2_added,
            c.brake_engagements,
            c.brake_rounds,
            self.params.gen_target,
            generation.g_circ,
            generation.g_all,
            generation.targetable,
            generation.all_lag,
            generation.total,
            generation.lag,
            generation.targetable,
            generation.elig,
            generation.wlag,
            generation.min,
            self.twist_coverage(),
            self.canary_frac(),
            c.canary_fallthrough,
            if c.litter_windows > 0 {
                c.litter_distinct_sum as f64 / c.litter_windows as f64
            } else {
                0.0
            },
            c.litter_full_spliced,
            c.litter_banned,
            c.cross_r1,
            c.cross_r2,
            c.cross_r3,
            c.presplits,
            c.fresh_splits,
            c.unsubs,
            c.inserts,
            c.twist_negs,
            c.twist_swaps,
            c.twist_cnots,
            c.twist_relabels,
            c.twist_case_splits,
            c.twist_span,
            c.twist_skips,
            c.nl_frame_attempts,
            c.nl_frames,
            c.nl_frame_skips,
            c.nl_frame_packet_gates,
            c.nl_frame_shot_attempts,
            c.nl_frame_crossings,
            c.nl_frame_preparatory_rewrites,
            c.nl_frame_blocked,
            c.nl_frame_span,
            c.nl_frame_nodes,
            c.nl_frame_protected_blocks,
            nl_live,
            nl_gates,
            nl_coverage,
            nl_protected,
            nl_protected_gates,
            nl_protected_coverage,
            c.declined,
            c.blocked_width,
            c.blocked_size_cap,
            c.blocked_deadlock,
            c.boundary,
            c.floats,
            c.float_steps,
            c.scatters,
            c.scatter_steps,
            disp,
            owin,
            fan0,
            leew,
            odiff,
            oadj,
            hist.join(" ")
        );
        let census = self.g57_census();
        let mean_choice = if c.choice_splices == 0 {
            0.0
        } else {
            c.choice_sum as f64 / c.choice_splices as f64
        };
        let bits_per_splice = if c.choice_splices == 0 {
            0.0
        } else {
            c.choice_bits_milli as f64 / (1000.0 * c.choice_splices as f64)
        };
        let dmin = if c.dmin_windows == 0 {
            0.0
        } else {
            c.dmin_shorter as f64 / c.dmin_windows as f64
        };
        let tg_net = c
            .tg_net_hist
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "[fmix-layer] mode={:?} p_mix={:.3} shaped={} g57={} polf={:.3} curated={} idskip={} dmin={:.3} choice_splices={} multi={} mean={:.2} bits/splice={:.3} tg={}/{} net=[{}] solves={} solve_ms={:.1} slides={} retries={}{}",
            self.db_mode_cur,
            self.params.p_mix,
            census.shaped,
            census.opp_pol,
            census.pol_flipped(),
            c.db_curated_hits,
            c.db_identity_skips,
            dmin,
            c.choice_splices,
            c.choice_multi,
            mean_choice,
            bits_per_splice,
            c.tg_consumed,
            c.tg_emitted,
            tg_net,
            c.tg_solves,
            c.tg_solve_ns as f64 / 1_000_000.0,
            c.tg_slides,
            c.tg_retries,
            self.prof
                .as_ref()
                .map_or_else(String::new, |profile| format!(
                    " profile=phase{} eff={:.3} pmix={:.3} ghat={:+.4} shat={:+.4} dhat={:+.4}",
                    profile.phase,
                    profile.eff,
                    profile.pmix,
                    profile.ghat,
                    profile.shat,
                    profile.dhat
                ))
        );
    }

    pub fn origins_in_order(&self) -> Vec<u32> {
        self.arena
            .ids_in_order()
            .iter()
            .map(|&id| self.meta_of(id).origin)
            .collect()
    }

    /// The true input circuit the periodic `global_check` verifies against —
    /// on a resumed run, the original from the state file, not the resume
    /// point.
    pub fn original(&self) -> &[XGate] {
        &self.original
    }

    pub fn num_wires(&self) -> usize {
        self.num_wires
    }
}

/// Version stamp for the resume file. The parameter set has changed repeatedly,
/// and silently reinterpreting fields would be worse than refusing to load.
/// Version 2: this tree's layout (frame/protected_until in gate meta and
/// journal, a prof section, no ancestry) — origin's v1 files refuse cleanly.
pub const STATE_VERSION: u32 = 2;

/// The persistent-counter roster: the single authority both checkpoint
/// serializers use, so a field added here is checkpointed everywhere at once.
/// Histograms (`width_hist`, `tg_net_hist`) and session-local timing
/// (`tg_solves`, `tg_solve_ns`) are shapes, not totals — deliberately excluded.
macro_rules! with_counter_fields {
    ($m:ident) => {
        $m!(
            moves,
            db_slot2_rounds,
            db_slot2_hits,
            db_slot2_added,
            canary_fallthrough,
            brake_engagements,
            brake_rounds,
            litter_banned,
            litter_full_spliced,
            litter_windows,
            litter_distinct_sum,
            db_identity_skips,
            db_curated_hits,
            dmin_windows,
            dmin_shorter,
            choice_splices,
            choice_multi,
            choice_sum,
            choice_bits_milli,
            db_mix_added,
            db_mix_removed,
            db_cmp_added,
            db_cmp_removed,
            db_comp_hits,
            db_comp_misses,
            db_agn_hits,
            db_agn_misses,
            db_gates_removed,
            db_gates_added,
            db_wide_skip,
            db_attempts,
            db_degree_skips,
            db_span_skips,
            db_build_aborts,
            db_protected_skips,
            db_frame_skips,
            db_cap_skips,
            blocked_size_cap,
            merges_cancel,
            merges_xfuse,
            merges_drop,
            merges_subsume,
            merges_sibling,
            merges_cross_origin,
            tabu_blocked,
            merge_no_partner,
            merge_wall_blocked,
            merge_too_far,
            merge_not_adjacent,
            undos,
            undo_dead,
            undo_tabu,
            undo_gather_miss,
            cross_r1,
            cross_r2,
            cross_r3,
            presplits,
            fresh_splits,
            unsubs,
            inserts,
            twist_negs,
            twist_swaps,
            twist_cnots,
            twist_relabels,
            twist_case_splits,
            twist_span,
            twist_skips,
            tg_consumed,
            tg_emitted,
            tg_slides,
            tg_retries,
            nl_frame_attempts,
            nl_frames,
            nl_frame_skips,
            nl_frame_packet_gates,
            nl_frame_shot_attempts,
            nl_frame_crossings,
            nl_frame_preparatory_rewrites,
            nl_frame_blocked,
            nl_frame_span,
            nl_frame_nodes,
            nl_frame_protected_blocks,
            g57_pairs_seeded,
            g57_seed_gates,
            g57_seed_shots,
            g57_seed_fragments,
            g57_seed_r_crossings,
            blocked_width,
            blocked_deadlock,
            declined,
            boundary,
            floats,
            float_steps,
            scatters,
            scatter_steps,
            dropped_neverfire
        )
    };
}

impl MixCounters {
    /// Checkpoint form: `name=value` pairs on one line. Self-describing, so a
    /// counter added later reads as zero from an older v2 file and unknown
    /// names in a newer file are ignored — no positional fragility.
    pub fn to_line(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        macro_rules! emit {
            ($($n:ident),* $(,)?) => { $(
                {
                    if !out.is_empty() {
                        out.push(' ');
                    }
                    let _ = write!(out, "{}={}", stringify!($n), self.$n);
                }
            )* }
        }
        with_counter_fields!(emit);
        out
    }

    pub fn from_line(line: &str) -> Option<MixCounters> {
        let mut map: HashMap<&str, u64> = HashMap::new();
        for pair in line.split_whitespace() {
            let (name, value) = pair.split_once('=')?;
            map.insert(name, value.parse().ok()?);
        }
        let mut counters = MixCounters::default();
        macro_rules! load {
            ($($n:ident),* $(,)?) => { $(
                counters.$n = map.get(stringify!($n)).copied().unwrap_or(0);
            )* }
        }
        with_counter_fields!(load);
        Some(counters)
    }
}

impl Mixer {
    /// Write everything a resumed run needs and the circuit file does not
    /// carry. A run is hours long and every measurement depends on it, so a
    /// stop — flag, canary, dose or budget — should be a PAUSE, not a loss.
    ///
    /// Three things make this more than a gate dump:
    ///
    /// - Per-gate `dir` is load-bearing and has no sidecar. Directions are
    ///   drawn at load and the whole directional walk rides on them, so a
    ///   resume that redrew them would restart transport rather than continue
    ///   it. Same for `dgen`, `litter`, `frame`, `protected_until` and `event`.
    /// - The undo journal references arena IDs, so the checkpoint RENUMBERS to
    ///   0..n-1 in arena order — exactly what `Arena::from_gates` reproduces —
    ///   and remaps the entries. Entries with any dead piece are dropped; they
    ///   were already unusable.
    /// - The ORIGINAL circuit is what `global_check` compares against. A
    ///   resumed run verifying against its own resume point would verify
    ///   nothing about fidelity to the true input.
    ///
    /// `StdRng` is not serialisable, so a fresh `u64` is drawn from each
    /// generator and stored: a clean continuation, not a bit-identical replay.
    pub fn save_state(&mut self, path: &str) -> std::io::Result<()> {
        use std::fmt::Write as _;
        let ids = self.arena.ids_in_order();
        let mut newid: HashMap<u32, u32> = HashMap::with_capacity(ids.len());
        for (i, &id) in ids.iter().enumerate() {
            newid.insert(id, i as u32);
        }
        let gate_line = |o: &mut String, g: &XGate| {
            let _ = write!(o, "{} {} {}", g.target, g.comp as u8, g.ctrls.len());
            for &(w, p) in &g.ctrls {
                let _ = write!(o, " {} {}", w, p as u8);
            }
        };
        let mut o = String::with_capacity(ids.len() * 56);
        let _ = writeln!(o, "fmix-state {STATE_VERSION}");
        let _ = writeln!(o, "wires {}", self.num_wires);
        let _ = writeln!(o, "moves {}", self.moves_done);
        let _ = writeln!(o, "next_event {}", self.next_event);
        let _ = writeln!(o, "next_litter {}", self.next_litter);
        let _ = writeln!(o, "rng {}", self.rng.random::<u64>());
        let _ = writeln!(o, "metrics_rng {}", self.metrics_rng.random::<u64>());
        let _ = writeln!(
            o,
            "db_mode {}",
            match self.db_mode_cur {
                DbMode::Mix => "mix",
                DbMode::Compressing => "comp",
                DbMode::SizeAgnostic => "any",
                DbMode::MinGrow => "mingrow",
            }
        );
        let _ = writeln!(
            o,
            "brake {} {} {}",
            self.brake_on as u8, self.brake_mark_move, self.brake_mark_size
        );
        let _ = writeln!(o, "pool_scan_due {}", self.pool_scan_due);
        let _ = writeln!(o, "canary_failures {}", self.canary_failures);
        let _ = writeln!(o, "counters {}", self.counters.to_line());

        let _ = writeln!(o, "gates {}", ids.len());
        for &id in &ids {
            let m = self.meta_of(id);
            let g = self.arena.gate(id).clone();
            gate_line(&mut o, &g);
            let _ = writeln!(
                o,
                " | {} {} {} {} {} {} {} {}",
                m.origin,
                m.event,
                (m.dir == Dir::R) as u8,
                m.dgen,
                m.litter,
                m.litter_size,
                m.frame,
                m.protected_until
            );
        }
        let _ = writeln!(o, "original {}", self.original.len());
        for g in self.original.clone().iter() {
            gate_line(&mut o, g);
            o.push('\n');
        }
        let _ = writeln!(o, "tabu {}", self.tabu.len());
        for &(e, mv) in self.tabu.iter() {
            let _ = writeln!(o, "{e} {mv}");
        }
        let pool: Vec<u32> = self
            .pool
            .iter()
            .filter_map(|id| newid.get(id).copied())
            .collect();
        let _ = writeln!(o, "pool {}", pool.len());
        for id in &pool {
            let _ = writeln!(o, "{id}");
        }
        let _ = writeln!(o, "canary {}", self.canary.len());
        for b in self.canary.iter() {
            let _ = writeln!(o, "{}", *b as u8);
        }
        // Layer 2 is the primary run mode here, so ProfState IS serialized
        // (unlike origin's v1 punt): a resume that restarted the profile clock
        // at eff = 0 would make checkpointing useless for exactly the runs
        // that need it.
        match &self.prof {
            None => {
                let _ = writeln!(o, "prof 0");
            }
            Some(p) => {
                let _ = write!(
                    o,
                    "prof 1 {} {} {} {} {} {} {} {} {} {} {} {} {}",
                    p.phase,
                    p.pmix,
                    p.integ,
                    p.ghat,
                    p.shat,
                    p.dhat,
                    p.eff,
                    p.next_eff,
                    p.sat,
                    p.s_in,
                    p.base_moves,
                    p.base_size,
                    p.base_pmix
                );
                for v in p.base_mix.iter().chain(p.base_comp.iter()) {
                    let _ = write!(o, " {v}");
                }
                o.push('\n');
            }
        }
        // Journal: keep only entries every piece of which is still live, then
        // remap. Stamps are NOT stored — the rebuilt arena assigns its own, and
        // resume reads them back from it.
        let keep: Vec<&UndoEntry> = self
            .journal
            .iter()
            .filter(|e| {
                e.after.iter().all(|&(id, st)| {
                    self.arena.is_linked(id)
                        && self.arena.stamp(id) == st
                        && newid.contains_key(&id)
                })
            })
            .collect();
        let _ = writeln!(o, "journal {}", keep.len());
        for e in keep {
            gate_line(&mut o, &e.before[0]);
            o.push(' ');
            gate_line(&mut o, &e.before[1]);
            let _ = write!(
                o,
                " | {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}",
                (e.dir == Dir::R) as u8,
                newid[&e.pivot],
                e.event,
                e.origins[0],
                e.origins[1],
                e.gens[0],
                e.gens[1],
                e.litters[0],
                e.litters[1],
                e.litter_sizes[0],
                e.litter_sizes[1],
                e.frames[0],
                e.frames[1],
                e.protected_until[0],
                e.protected_until[1]
            );
            let _ = write!(o, " | {}", e.after.len());
            for &(id, _) in &e.after {
                let _ = write!(o, " {}", newid[&id]);
            }
            let _ = writeln!(o, " {}", e.misses);
        }
        std::fs::write(path, o)
    }

    /// Rebuild a mixer from a state file. `params` come from the CLI as usual:
    /// a resume is free to change rates, targets, the brake or the mode, which
    /// is the point — a paused run should be steerable. What it must NOT
    /// change is the version, since the field meanings would drift silently.
    pub fn resume_state(
        path: &str,
        params: MixParams,
        db: Option<FrozenDb>,
    ) -> std::io::Result<Mixer> {
        let text = std::fs::read_to_string(path)?;
        let mut lines = text.lines();
        let bad = |m: &str| std::io::Error::other(m.to_string());
        let mut hdr = lines
            .next()
            .ok_or_else(|| bad("empty state file"))?
            .split_whitespace();
        if hdr.next() != Some("fmix-state") {
            return Err(bad("missing fmix-state header"));
        }
        let v: u32 = hdr
            .next()
            .and_then(|x| x.parse().ok())
            .ok_or_else(|| bad("bad version"))?;
        if v != STATE_VERSION {
            return Err(bad(&format!(
                "state file version {v} != {STATE_VERSION}; refusing to reinterpret its fields"
            )));
        }
        fn scalar(lines: &mut std::str::Lines<'_>, want: &str) -> std::io::Result<Vec<String>> {
            let l = lines
                .next()
                .ok_or_else(|| std::io::Error::other(format!("missing {want}")))?;
            let mut it = l.split_whitespace();
            if it.next() != Some(want) {
                return Err(std::io::Error::other(format!(
                    "expected {want} in state file"
                )));
            }
            Ok(it.map(|x| x.to_string()).collect())
        }
        let num_wires: usize = scalar(&mut lines, "wires")?[0]
            .parse()
            .map_err(|_| bad("wires"))?;
        let moves_done: u64 = scalar(&mut lines, "moves")?[0]
            .parse()
            .map_err(|_| bad("moves"))?;
        let next_event: u64 = scalar(&mut lines, "next_event")?[0]
            .parse()
            .map_err(|_| bad("next_event"))?;
        let next_litter: u64 = scalar(&mut lines, "next_litter")?[0]
            .parse()
            .map_err(|_| bad("next_litter"))?;
        let rng_seed: u64 = scalar(&mut lines, "rng")?[0]
            .parse()
            .map_err(|_| bad("rng"))?;
        let mrng_seed: u64 = scalar(&mut lines, "metrics_rng")?[0]
            .parse()
            .map_err(|_| bad("metrics_rng"))?;
        let mode_s = scalar(&mut lines, "db_mode")?[0].clone();
        let brake = scalar(&mut lines, "brake")?;
        if brake.len() != 3 {
            return Err(bad("brake shape"));
        }
        let pool_scan_due: u64 = scalar(&mut lines, "pool_scan_due")?[0]
            .parse()
            .map_err(|_| bad("scan"))?;
        let canary_failures: usize = scalar(&mut lines, "canary_failures")?[0]
            .parse()
            .map_err(|_| bad("cf"))?;
        let counters_line = scalar(&mut lines, "counters")?.join(" ");

        fn section(lines: &mut std::str::Lines<'_>, want: &str) -> std::io::Result<usize> {
            let l = lines
                .next()
                .ok_or_else(|| std::io::Error::other(format!("missing {want}")))?;
            let mut it = l.split_whitespace();
            if it.next() != Some(want) {
                return Err(std::io::Error::other(format!("expected section {want}")));
            }
            it.next()
                .and_then(|x| x.parse().ok())
                .ok_or_else(|| std::io::Error::other(format!("{want} count")))
        }
        let parse_gate = |it: &mut std::str::SplitWhitespace| -> Option<XGate> {
            let target: u16 = it.next()?.parse().ok()?;
            let comp: u8 = it.next()?.parse().ok()?;
            let k: usize = it.next()?.parse().ok()?;
            let mut ctrls: Lits = Lits::new();
            for _ in 0..k {
                let w: u16 = it.next()?.parse().ok()?;
                let p: u8 = it.next()?.parse().ok()?;
                ctrls.push((w, p != 0));
            }
            ctrls.sort_unstable();
            Some(XGate {
                target,
                comp: comp != 0,
                ctrls,
            })
        };

        let ng = section(&mut lines, "gates")?;
        let mut gates = Vec::with_capacity(ng);
        let mut metas = Vec::with_capacity(ng);
        for _ in 0..ng {
            let l = lines.next().ok_or_else(|| bad("short gates section"))?;
            let (gp, mp) = l
                .split_once(" | ")
                .ok_or_else(|| bad("gate line missing meta"))?;
            let g = parse_gate(&mut gp.split_whitespace()).ok_or_else(|| bad("bad gate"))?;
            let m: Vec<&str> = mp.split_whitespace().collect();
            if m.len() != 8 {
                return Err(bad("gate meta must have 8 fields"));
            }
            let p =
                |i: usize| -> std::io::Result<u64> { m[i].parse().map_err(|_| bad("meta field")) };
            metas.push(Meta {
                origin: p(0)? as u32,
                event: p(1)?,
                dir: if p(2)? != 0 { Dir::R } else { Dir::L },
                dgen: p(3)? as u32,
                litter: p(4)?,
                litter_size: p(5)? as u16,
                frame: p(6)?,
                protected_until: p(7)?,
            });
            gates.push(g);
        }
        let no = section(&mut lines, "original")?;
        let mut original = Vec::with_capacity(no);
        for _ in 0..no {
            let l = lines.next().ok_or_else(|| bad("short original section"))?;
            original.push(parse_gate(&mut l.split_whitespace()).ok_or_else(|| bad("bad orig"))?);
        }
        let nt = section(&mut lines, "tabu")?;
        let mut tabu: VecDeque<(u64, u64)> = VecDeque::with_capacity(nt);
        for _ in 0..nt {
            let l = lines.next().ok_or_else(|| bad("short tabu"))?;
            let mut it = l.split_whitespace();
            let e: u64 = it
                .next()
                .and_then(|x| x.parse().ok())
                .ok_or_else(|| bad("tabu"))?;
            let mv: u64 = it
                .next()
                .and_then(|x| x.parse().ok())
                .ok_or_else(|| bad("tabu"))?;
            tabu.push_back((e, mv));
        }
        let np = section(&mut lines, "pool")?;
        let mut pool = Vec::with_capacity(np);
        for _ in 0..np {
            let l = lines.next().ok_or_else(|| bad("short pool"))?;
            pool.push(l.trim().parse::<u32>().map_err(|_| bad("pool id"))?);
        }
        let nc = section(&mut lines, "canary")?;
        let mut canary: VecDeque<bool> = VecDeque::with_capacity(nc);
        for _ in 0..nc {
            let l = lines.next().ok_or_else(|| bad("short canary"))?;
            canary.push_back(l.trim() != "0");
        }
        let prof_line = scalar(&mut lines, "prof")?;
        let prof = if prof_line.first().map(String::as_str) == Some("1") {
            if prof_line.len() != 1 + 13 + 8 {
                return Err(bad("prof shape"));
            }
            let pf = |i: usize| -> f64 { prof_line[i].parse().unwrap_or(0.0) };
            let pu = |i: usize| -> u64 { prof_line[i].parse().unwrap_or(0) };
            let mut base_mix = [0u64; 4];
            let mut base_comp = [0u64; 4];
            for i in 0..4 {
                base_mix[i] = pu(14 + i);
                base_comp[i] = pu(18 + i);
            }
            Some(ProfState {
                phase: pu(1) as u8,
                pmix: pf(2),
                integ: pf(3),
                ghat: pf(4),
                shat: pf(5),
                dhat: pf(6),
                eff: pf(7),
                next_eff: pf(8),
                sat: pu(9) as u32,
                s_in: pf(10),
                base_moves: pu(11),
                base_size: pf(12),
                base_pmix: pf(13),
                base_mix,
                base_comp,
            })
        } else {
            None
        };
        let nj = section(&mut lines, "journal")?;
        let mut journal_raw = Vec::with_capacity(nj);
        for _ in 0..nj {
            let l = lines.next().ok_or_else(|| bad("short journal"))?;
            let parts: Vec<&str> = l.split(" | ").collect();
            if parts.len() != 3 {
                return Err(bad("journal line shape"));
            }
            let mut gi = parts[0].split_whitespace();
            let b0 = parse_gate(&mut gi).ok_or_else(|| bad("journal before0"))?;
            let b1 = parse_gate(&mut gi).ok_or_else(|| bad("journal before1"))?;
            let f: Vec<u64> = parts[1]
                .split_whitespace()
                .map(|x| x.parse().unwrap_or(0))
                .collect();
            if f.len() != 15 {
                return Err(bad("journal meta shape"));
            }
            let mut ai = parts[2].split_whitespace();
            let n: usize = ai
                .next()
                .and_then(|x| x.parse().ok())
                .ok_or_else(|| bad("after n"))?;
            let after: Vec<u32> = (0..n)
                .filter_map(|_| ai.next().and_then(|x| x.parse().ok()))
                .collect();
            let misses: u8 = ai.next().and_then(|x| x.parse().ok()).unwrap_or(0);
            journal_raw.push((b0, b1, f, after, misses));
        }

        // Build the mixer on the SAME id assignment the checkpoint renumbered
        // to: Arena::from_gates hands out 0..n-1 in order, which is what
        // save_state mapped the journal and pool onto.
        let mut mx = Mixer::new_with_optional_db(gates, num_wires, params, db);
        mx.original = original;
        mx.moves_done = moves_done;
        mx.counters = MixCounters::from_line(&counters_line).ok_or_else(|| bad("counters"))?;
        mx.counters.moves = moves_done;
        mx.next_event = next_event;
        mx.next_litter = next_litter;
        mx.rng = StdRng::seed_from_u64(rng_seed);
        mx.metrics_rng = StdRng::seed_from_u64(mrng_seed);
        // The LIVE mode comes from the command line, not the file. A resume is
        // meant to be re-steerable — changing --db-mode is the whole point of a
        // manual breathing cycle — and letting the saved value win silently
        // ignores the flag. The saved mode is kept for diagnostics only. If
        // the brake was engaged it re-engages on the next round anyway, since
        // apply_size_brake runs in slot 0.
        let _saved_mode = DbMode::parse(&mode_s);
        mx.db_mode_cur = mx.params.db_mode;
        mx.brake_on = brake[0] != "0";
        mx.brake_mark_move = brake[1].parse().unwrap_or(0);
        mx.brake_mark_size = brake[2].parse().unwrap_or(0);
        mx.pool_scan_due = pool_scan_due;
        mx.canary = canary;
        mx.canary_failures = canary_failures;
        mx.pool = pool;
        mx.prof = prof;
        for (i, m) in metas.into_iter().enumerate() {
            mx.set_meta(i as u32, m);
        }
        // Stamps come from the freshly built arena; the checkpoint kept only
        // entries whose pieces were all live, so every id here is linked.
        mx.journal = journal_raw
            .into_iter()
            .map(|(b0, b1, f, after, misses)| UndoEntry {
                before: [b0, b1],
                dir: if f[0] != 0 { Dir::R } else { Dir::L },
                pivot: f[1] as u32,
                after: after.iter().map(|&id| (id, mx.arena.stamp(id))).collect(),
                event: f[2],
                origins: [f[3] as u32, f[4] as u32],
                gens: [f[5] as u32, f[6] as u32],
                litters: [f[7], f[8]],
                litter_sizes: [f[9] as u16, f[10] as u16],
                frames: [f[11], f[12]],
                protected_until: [f[13], f[14]],
                misses,
            })
            .collect();
        mx.tabu = tabu;
        Ok(mx)
    }

    /// Move-cadence snapshots: at each multiple of `snap_every_moves`, write
    /// `<base>.mv<m>.mpmct1` + `.gens` + `.state`. The state goes through a
    /// temp path + rename so an interrupted write never leaves a half-file
    /// that looks resumable.
    fn check_move_snap(&mut self) {
        if self.params.snap_every_moves == 0
            || self.params.snap_base.is_empty()
            || self.moves_done == 0
            || self.moves_done % self.params.snap_every_moves != 0
        {
            return;
        }
        let base = format!("{}.mv{}", self.params.snap_base, self.moves_done);
        let gates = self.arena.to_vec();
        if let Err(e) =
            super::format::write_mpmct(&format!("{base}.mpmct1"), &gates, self.num_wires)
        {
            eprintln!("[fmix] snap circuit write failed: {e}");
            return;
        }
        let mut generations = String::with_capacity(gates.len() * 4);
        for generation in self.gens_in_order() {
            generations.push_str(&format!("{generation}\n"));
        }
        if let Err(e) = std::fs::write(format!("{base}.gens"), generations) {
            eprintln!("[fmix] snap gens write failed: {e}");
        }
        let tmp = format!("{base}.state.tmp");
        match self.save_state(&tmp) {
            Ok(()) => {
                if let Err(e) = std::fs::rename(&tmp, format!("{base}.state")) {
                    eprintln!("[fmix] snap state rename failed: {e}");
                }
            }
            Err(e) => eprintln!("[fmix] snap state write failed: {e}"),
        }
        println!(
            "[fmix] SNAP: mv={} size={} -> {base}.{{mpmct1,gens,state}}",
            self.moves_done,
            gates.len()
        );
    }
}

#[cfg(test)]
mod mix_tests {
    use super::super::xgate::{XGate, eval_u64};
    use super::*;

    fn rand_gate(rng: &mut StdRng, wires: u16, max_w: usize, allow_comp: bool) -> XGate {
        loop {
            let target = rng.random_range(0..wires);
            let w = rng.random_range(0..=max_w);
            let lits: Vec<(u16, bool)> = (0..w)
                .map(|_| (rng.random_range(0..wires), rng.random_bool(0.5)))
                .filter(|&(cw, _)| cw != target)
                .collect();
            if let Some(mut g) = XGate::conj(target, lits) {
                if allow_comp && g.width() == 2 && rng.random_bool(0.3) {
                    g.comp = true;
                }
                return g;
            }
        }
    }

    #[test]
    fn mixer_canonicalizes_complemented_empty_input() {
        let noop = XGate {
            target: 1,
            comp: true,
            ctrls: Default::default(),
        };
        let raw = vec![XGate::x_gate(0), noop, XGate::cnot(1, 0)];
        let canonical = vec![XGate::x_gate(0), XGate::cnot(1, 0)];
        let params = MixParams {
            moves: 1_000,
            verify_every: 1,
            report_every: u64::MAX,
            seed: 0x1D_E17,
            ..MixParams::default()
        };
        let mut mixer = Mixer::new(raw.clone(), 2, params);
        assert_eq!(mixer.original, canonical);
        assert_eq!(mixer.arena.to_vec(), canonical);
        assert_eq!(mixer.origins_in_order(), vec![0, 2]);
        mixer.run();
        let mixed = mixer.arena.to_vec();
        assert!(mixed.iter().all(|gate| !gate.is_noop()));
        for state in 0..4 {
            assert_eq!(eval_u64(&raw, state), eval_u64(&mixed, state));
        }
    }

    #[test]
    fn mixer_accepts_an_identity_only_input_as_an_empty_tape() {
        let noop = |target| XGate {
            target,
            comp: true,
            ctrls: Default::default(),
        };
        let raw = vec![noop(0), noop(1)];
        let mut mixer = Mixer::new(
            raw.clone(),
            2,
            MixParams {
                moves: 100,
                verify_every: 1,
                report_every: 1,
                ..MixParams::default()
            },
        );
        assert!(mixer.original.is_empty());
        assert!(mixer.arena.to_vec().is_empty());
        assert!(matches!(mixer.run(), MixStop::MovesBudget));
        assert_eq!(mixer.counters.moves, 100);
        assert!(mixer.arena.to_vec().is_empty());
        for state in 0..4 {
            assert_eq!(eval_u64(&raw, state), state);
        }
    }

    #[test]
    fn crossing_defensively_deletes_an_injected_complemented_empty() {
        let baseline = vec![
            XGate::x_gate(0),
            // Commutes with the injected target-1 identity, forcing that
            // identity to float before it reaches the target-1 reader.
            XGate::cnot(3, 4),
            XGate::cnot(2, 1),
        ];
        let mut mixer = Mixer::new(
            baseline.clone(),
            5,
            MixParams {
                // The historical empty-presplit path tried to retreat the
                // already-freed arena id by the full one-gate float distance.
                dir_q: 0.0,
                report_every: u64::MAX,
                ..MixParams::default()
            },
        );
        let noop = XGate {
            target: 1,
            comp: true,
            ctrls: Default::default(),
        };
        let noop_id = mixer.arena.insert_after(mixer.arena.head(), noop);
        mixer.index_add(noop_id);
        mixer.set_meta(
            noop_id,
            Meta {
                origin: ORIGIN_SYNTH,
                event: 0,
                dir: Dir::R,
                frame: 0,
                protected_until: 0,
                litter: 0,
                litter_size: 1,
                dgen: 0,
            },
        );

        let trace = mixer.cross_move_on(noop_id);
        assert!(trace.changed);
        assert!(!trace.crossed);
        assert!(trace.shot_descendants.is_empty());
        assert!(!mixer.arena.is_linked(noop_id));
        assert_eq!(mixer.arena.to_vec(), baseline);
        assert_eq!(mixer.arena.len(), mixer.arena.ids_in_order().len());
        mixer.global_check();
    }

    #[test]
    fn seeded_native_g57_pairs_fragment_shoot_and_preserve_function() {
        let baseline = random_mixed_circuit(0x5757, 8, 96);
        for region in [
            super::super::g57_pairs::G57PairRegion::FirstQuarter,
            super::super::g57_pairs::G57PairRegion::MiddleQuarter,
            super::super::g57_pairs::G57PairRegion::LastQuarter,
            super::super::g57_pairs::G57PairRegion::Uniform,
        ] {
            let params = MixParams {
                k_max: 6,
                split_base: 1.0,
                dir_q: 1.0,
                report_every: u64::MAX,
                seed: 1234,
                ..MixParams::default()
            };
            let mut mixer = Mixer::new(baseline.clone(), 8, params);
            let fossils_before = mixer.remaining_g57();
            let report = mixer
                .seed_g57_pairs(G57PairSeedConfig {
                    pairs_per_target_wire: 2,
                    target_wires: 4,
                    num_wires: 8,
                    control_wire_limit: 8,
                    region,
                    seed: 20260723,
                })
                .unwrap();
            let mixed = mixer.arena.to_vec();
            for state in 0..(1u64 << 8) {
                assert_eq!(
                    eval_u64(&baseline, state),
                    eval_u64(&mixed, state),
                    "seed hook changed function: region={region} state={state:#x}"
                );
            }
            assert_eq!(report.pairs, 8);
            assert_eq!(report.inserted_native_g57, 16);
            assert_eq!(report.records.len(), 8);
            assert_eq!(
                report.intact_collision_stops + report.intact_boundary_stops,
                report.pairs * 2
            );
            assert!(
                report.total_intact_left_shot_steps + report.total_intact_right_shot_steps > 0,
                "no intact seeded G57 moved before presplitting in {region}"
            );
            assert!(report.total_cross_attempts >= 16);
            assert!(
                report.halves_with_r_crossing > 0,
                "targeted fragments never completed an R crossing in {region}"
            );
            assert!(mixer.remaining_g57() <= fossils_before);
            assert_eq!(report.manifest_tsv().lines().count(), report.pairs + 1);
            mixer.global_check();
        }
    }

    #[test]
    fn g57_pair_frame_census_tracks_rewrites_and_excludes_generic_frames() {
        let baseline = random_mixed_circuit(0x5757_cafe, 8, 64);
        let params = MixParams {
            k_max: 6,
            split_base: 1.0,
            nl_frame_packet_gates: 4,
            nl_frame_shots: 8,
            report_every: u64::MAX,
            verify_every: 0,
            seed: 0x1234_5678,
            ..MixParams::default()
        };
        let mut mixer = Mixer::new(baseline, 8, params);

        // Ordinary nonlinear frames use a disjoint tag namespace and must not
        // appear in the pair census.
        mixer.nonlinear_frame_move();
        assert!(mixer.active_nonlinear_frame_stats().0 > 0);
        assert_eq!(mixer.g57_pair_frame_stats(), G57PairFrameStats::default());

        let seeded = mixer
            .seed_g57_pairs(G57PairSeedConfig {
                pairs_per_target_wire: 2,
                target_wires: 4,
                num_wires: 8,
                control_wire_limit: 8,
                region: super::super::g57_pairs::G57PairRegion::Uniform,
                seed: 20260723,
            })
            .unwrap();
        let initial = mixer.g57_pair_frame_stats();
        assert_eq!(initial.distinct_pair_ids, seeded.pairs);
        assert_eq!(initial.distinct_copy_frames, seeded.pairs * 2);
        assert_eq!(initial.complete_pairs, seeded.pairs);
        assert!(initial.tagged_gates >= seeded.emitted_fragments);
        assert!((0.0..=1.0).contains(&initial.union_span_coverage));
        assert!(initial.union_span_coverage > 0.0);
        assert!(
            mixer.active_nonlinear_frame_stats().0 > 0,
            "pair census accidentally consumed the generic nonlinear frame"
        );

        // Exercise ordinary rewrites followed by the contraction-only path.
        // Frame tags may legitimately be folded or erased, so assert census
        // accounting invariants rather than requiring every tag to survive.
        mixer.params.target_size = mixer.arena.len() + 16;
        mixer.params.temp = 64.0;
        mixer.params.moves = 128;
        mixer.params.w_cross = 0.70;
        mixer.params.w_fresh = 0.0;
        mixer.params.w_unsub = 0.10;
        mixer.params.w_insert = 0.05;
        mixer.run();
        let after_rewrites = mixer.g57_pair_frame_stats();
        assert!(after_rewrites.distinct_pair_ids > 0);
        assert!(after_rewrites.distinct_pair_ids <= seeded.pairs);
        assert!(after_rewrites.distinct_copy_frames <= seeded.pairs * 2);
        assert!(after_rewrites.complete_pairs <= after_rewrites.distinct_pair_ids);
        assert!(after_rewrites.tagged_gates <= mixer.arena.len());

        mixer.params.tabu_moves = 0;
        let contraction = mixer.contract_to(mixer.arena.len().saturating_sub(8), 5_000, 2, 0.0);
        assert!(contraction.attempts > 0);
        let after_contraction = mixer.g57_pair_frame_stats();
        assert!(after_contraction.distinct_pair_ids <= seeded.pairs);
        assert!(after_contraction.distinct_copy_frames <= seeded.pairs * 2);
        assert!(after_contraction.complete_pairs <= after_contraction.distinct_pair_ids);
        assert!(after_contraction.tagged_gates <= mixer.arena.len());
        assert!((0.0..=1.0).contains(&after_contraction.union_span_coverage));
        mixer.global_check();
    }

    #[test]
    fn non_pair_dose_denominator_is_invariant_to_fresh_pair_frames() {
        let params = MixParams {
            gen_target: 100,
            gen_stop_frac: 0.02,
            gen_dose_exclude_g57_pair_frames: true,
            report_every: u64::MAX,
            verify_every: 0,
            seed: 0x5155,
            ..MixParams::default()
        };
        let mut mixer = Mixer::new(vec![XGate::x_gate(0)], 8, params);
        let excluded_before = mixer.dose_reached();
        assert!(!excluded_before);
        assert_eq!(
            mixer.generation_dose_stats().all,
            mixer.generation_dose_stats().non_pair
        );

        mixer
            .seed_g57_pairs(G57PairSeedConfig {
                pairs_per_target_wire: 50,
                target_wires: 4,
                num_wires: 8,
                control_wire_limit: 8,
                region: super::super::g57_pairs::G57PairRegion::FirstQuarter,
                seed: 0x5050,
            })
            .unwrap();
        let stats = mixer.generation_dose_stats();
        assert!(stats.all.fresh >= 800);
        assert_eq!(stats.non_pair.fresh, 0);
        assert!(stats.non_pair.total > 0);
        assert_eq!(stats.non_pair.all_lag, stats.non_pair.total);
        assert_eq!(
            mixer.dose_reached(),
            excluded_before,
            "adding GEN_FRESH pair frames changed the non-pair dose decision"
        );

        // Historical behavior remains opt-out and demonstrates the exact
        // dilution this campaign-specific mode is designed to avoid.
        mixer.params.gen_dose_exclude_g57_pair_frames = false;
        assert!(
            mixer.dose_reached(),
            "legacy all-gate denominator no longer treats GEN_FRESH material as reached"
        );
        assert!(!MixParams::default().gen_dose_exclude_g57_pair_frames);
    }

    // Every merge the catalogue accepts is a verified identity with comp=0
    // output; the presplit-pair rejoin (complemented result) is rejected.
    #[test]
    fn merge_catalogue_sound_and_comp_guarded() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut accepted = 0usize;
        for _ in 0..20_000 {
            let g = rand_gate(&mut rng, 6, 3, true);
            let h = rand_gate(&mut rng, 6, 3, true);
            if let Some(m) = merge_result(&g, &h) {
                let out = m.gates();
                assert!(
                    out.iter().all(|x| !x.comp),
                    "merge emitted comp: {g:?}+{h:?}"
                );
                assert!(
                    rules::verify_rewrite(&[g.clone(), h.clone()], &out),
                    "unsound merge: {g:?} + {h:?} -> {out:?}"
                );
                accepted += 1;
            }
        }
        assert!(
            accepted > 50,
            "catalogue accepted too few pairs to be tested: {accepted}"
        );

        // The presplit pieces of a g57 (x and !x!y on the same target) XOR to
        // the complemented parent: must be rejected.
        let p0 = XGate::conj(0, [(1u16, true)]).unwrap();
        let p1 = XGate::conj(0, [(1u16, false), (2u16, false)]).unwrap();
        assert!(
            merge_result(&p0, &p1).is_none(),
            "presplit rejoin must be comp-guarded"
        );

        // Two g57s differing in one polarity fuse into a conjunction (fossil
        // erosion), and a g57 plus its own monomial fuse into a NOT gate.
        let g57a = XGate {
            target: 0,
            comp: true,
            ctrls: p1.ctrls.clone(),
        };
        let mut g57b = g57a.clone();
        g57b.ctrls[0].1 = true;
        match merge_result(&g57a, &g57b) {
            Some(Merge::DropLit(m)) => assert!(!m.comp && m.width() == 1),
            other => panic!(
                "comp-comp polarity pair should DropLit, got {:?}",
                other.map(|m| m.gates())
            ),
        }
        let mono = XGate {
            target: 0,
            comp: false,
            ctrls: g57a.ctrls.clone(),
        };
        match merge_result(&g57a, &mono) {
            Some(Merge::XFuse(m)) => assert!(!m.comp && m.width() == 0),
            other => panic!(
                "g57 + own monomial should XFuse, got {:?}",
                other.map(|m| m.gates())
            ),
        }
    }

    // fresh-wire split and unsubsume each round-trip through the catalogue back
    // to the exact original gate.
    #[test]
    fn split_merge_roundtrips() {
        let mut rng = StdRng::seed_from_u64(11);
        for _ in 0..2_000 {
            let g = rand_gate(&mut rng, 8, 3, false);
            // fresh split on a wire the gate does not touch
            let x = (0..8u16).find(|&w| w != g.target && !g.reads(w)).unwrap();
            let a = XGate::conj(g.target, g.ctrls.iter().copied().chain([(x, true)])).unwrap();
            let b = XGate::conj(g.target, g.ctrls.iter().copied().chain([(x, false)])).unwrap();
            match merge_result(&a, &b) {
                Some(Merge::DropLit(m)) => assert_eq!(m, g),
                _ => panic!("fresh-split pieces must DropLit back to the parent"),
            }
            // unsubsume round-trip
            if g.width() > 0 {
                let (w, p) = g.ctrls[rng.random_range(0..g.ctrls.len())];
                let without = XGate::conj(g.target, g.ctrls_without(w)).unwrap();
                let flipped = XGate::conj(
                    g.target,
                    g.ctrls
                        .iter()
                        .map(|&(cw, cp)| if cw == w { (cw, !cp) } else { (cw, cp) }),
                )
                .unwrap();
                let _ = p;
                match merge_result(&without, &flipped) {
                    Some(Merge::Subsume(m)) => assert_eq!(m, g),
                    _ => panic!("unsubsume pieces must Subsume back to the parent"),
                }
            }
        }
    }

    fn random_g57_circuit(seed: u64, wires: u16, gates: usize) -> Vec<XGate> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..gates)
            .map(|_| {
                loop {
                    let a = rng.random_range(0..wires);
                    let x = rng.random_range(0..wires);
                    let y = rng.random_range(0..wires);
                    if a != x && a != y && x != y {
                        break XGate::from_g57([a, x, y]);
                    }
                }
            })
            .collect()
    }

    // A conjunction-dominated circuit like real fmix input (fsplit output is
    // ~90% eroded); a few g57 fossils sprinkled in.
    fn random_mixed_circuit(seed: u64, wires: u16, gates: usize) -> Vec<XGate> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..gates)
            .map(|i| {
                if i % 10 == 0 {
                    loop {
                        let a = rng.random_range(0..wires);
                        let x = rng.random_range(0..wires);
                        let y = rng.random_range(0..wires);
                        if a != x && a != y && x != y {
                            break XGate::from_g57([a, x, y]);
                        }
                    }
                } else {
                    loop {
                        let g = rand_gate(&mut rng, wires, 3, false);
                        if g.width() >= 1 {
                            break g;
                        }
                    }
                }
            })
            .collect()
    }

    // The chain holds size near the target, preserves the function (run() and
    // the end-of-run global check assert internally), does both kinds of moves,
    // and never increases the fossil count.
    #[test]
    fn mixer_holds_size_and_function() {
        let gates = random_mixed_circuit(3, 16, 300);
        let comp0 = gates.iter().filter(|g| g.comp).count();
        let params = MixParams {
            k_max: 5,
            moves: 20_000,
            target_size: 300,
            temp: 20.0,
            verify_every: 2_000,
            report_every: u64::MAX,
            seed: 5,
            ..MixParams::default()
        };
        let mut mx = Mixer::new(gates, 16, params);
        mx.run();
        // The chain equilibrates at a content-dependent floor above the target
        // (dead journal entries are permanently unmergeable by the pairwise
        // catalogue); the contract is BOUNDED drift under heavy churn, not
        // exact adherence. This run churns ~67 moves/gate.
        let n = mx.arena.len();
        assert!((200..=520).contains(&n), "size drifted from target: {n}");
        assert!(mx.counters.merges() > 0, "no merges happened");
        assert!(mx.counters.undos > 0, "no crossing undos happened");
        assert!(mx.counters.expands() > 0, "no expansions happened");
        assert!(mx.remaining_g57() <= comp0, "fossil count increased");
        assert!(
            mx.origin_displacement() > 0.01,
            "no positional mixing at all"
        );
    }

    // Thermostat up (catalogue-invertible moves only: fresh/unsub/insert, no
    // crossings), then pure-drain mode down: with expansion weights zeroed the
    // merge channel alone must dig most of the growth back out. Fresh-split
    // trees are hierarchically recoverable (child pairs DropLit-merge back
    // into parents, which can then merge with THEIR siblings), so this stock
    // stays recyclable — unlike crossing ladders, which are permanent once
    // their journal entries die (measured recoverable slack there: ~6%).
    #[test]
    fn mixer_thermostat_grows_then_drains() {
        let gates = random_mixed_circuit(9, 16, 200);
        let grow = MixParams {
            k_max: 5,
            moves: 30_000,
            target_size: 500,
            temp: 20.0,
            // Catalogue-invertible stock only. Insert is excluded since the
            // directional redesign: it embeds a cross per copy, and crossing
            // ladders are not pairwise-recoverable.
            w_cross: 0.0,
            w_fresh: 0.7,
            w_unsub: 0.3,
            w_insert: 0.0,
            verify_every: 5_000,
            report_every: u64::MAX,
            seed: 1,
            ..MixParams::default()
        };
        let mut mx = Mixer::new(gates, 16, grow);
        mx.run();
        let grown = mx.arena.len();
        assert!(grown >= 400, "did not grow toward target: {grown}");

        mx.params.target_size = 250;
        mx.params.w_fresh = 0.0;
        mx.params.w_unsub = 0.0;
        mx.params.w_insert = 0.0;
        mx.params.moves += 30_000;
        mx.run();
        let drained = mx.arena.len();
        assert!(
            (drained as f64) < grown as f64 * 0.7,
            "drain did not contract: {grown} -> {drained}"
        );
        assert!(mx.counters.merges() > 0, "no merges during drain");
    }

    // On an all-g57 input the chain erodes fossils; erosion inherently costs
    // +1 gate per presplit and is irreversible by design, so size may grow up
    // to initial + fossil count (plus thermostat slack) while comp declines.
    #[test]
    fn mixer_erodes_fossils() {
        let gates = random_g57_circuit(3, 16, 300);
        let comp0 = gates.iter().filter(|g| g.comp).count();
        let params = MixParams {
            k_max: 5,
            moves: 20_000,
            target_size: 300,
            temp: 20.0,
            verify_every: 2_000,
            report_every: u64::MAX,
            seed: 5,
            ..MixParams::default()
        };
        let mut mx = Mixer::new(gates, 16, params);
        mx.run();
        let comp_now = mx.remaining_g57();
        assert!(
            comp_now < comp0 / 2,
            "erosion too slow: {comp0} -> {comp_now}"
        );
        let n = mx.arena.len();
        assert!(n <= 300 + comp0 + 100, "grew past the erosion budget: {n}");
    }

    // Soundness of the collision predicate: ANY pair it calls non-colliding —
    // no read of the other's target, or separated by an opposite shared
    // control literal — must actually commute. (The converse is not claimed:
    // collides() may stay conservatively true on commuting pairs.)
    #[test]
    fn collides_separation_exemption_sound() {
        let mut rng = StdRng::seed_from_u64(23);
        let mut exempted = 0usize;
        for _ in 0..30_000 {
            let a = rand_gate(&mut rng, 6, 3, true);
            let b = rand_gate(&mut rng, 6, 3, true);
            let reads = a.reads(b.target) || b.reads(a.target);
            if !XGate::collides(&a, &b) {
                assert!(
                    rules::verify_rewrite(&[a.clone(), b.clone()], &[b.clone(), a.clone()]),
                    "non-colliding pair does not commute: {a:?} / {b:?}"
                );
                if reads {
                    exempted += 1; // separated despite a read of a target
                    assert!(!a.comp && !b.comp, "comp gate got the exemption");
                }
            }
        }
        assert!(
            exempted > 20,
            "exemption never fired in the sample: {exempted}"
        );
    }

    // Per-gate conjugation identities behind the twist move: N g N == neg(g)
    // and S g S == swap(g), exhaustively verified on random gates (comp gates
    // included — fossils relabel like anything else and stay fossils).
    #[test]
    fn twist_conjugation_units() {
        let mut rng = StdRng::seed_from_u64(31);
        let cnot = |t: u16, c: u16| XGate::conj(t, [(c, true)]).unwrap();
        for _ in 0..5_000 {
            let g = rand_gate(&mut rng, 8, 4, true);
            let w = rng.random_range(0..8u16);
            match conj_by_not(&g, w) {
                Some(g2) => {
                    let nw = XGate::x_gate(w);
                    assert!(
                        rules::verify_rewrite(
                            &[nw.clone(), g.clone(), nw],
                            std::slice::from_ref(&g2)
                        ),
                        "neg conjugation wrong: {g:?} on wire {w}"
                    );
                    assert_eq!(g2.width(), g.width());
                    assert_eq!(g2.comp, g.comp);
                }
                None => assert!(!g.reads(w), "invariant gate must not read the negated wire"),
            }
            let a = rng.random_range(0..8u16);
            let b = rng.random_range(0..8u16);
            if a == b {
                continue;
            }
            if let Some(g2) = conj_by_swap(&g, a, b) {
                let packet = [cnot(b, a), cnot(a, b), cnot(b, a)];
                let mut seq: Vec<XGate> = packet.to_vec();
                seq.push(g.clone());
                seq.extend(packet.to_vec());
                assert!(
                    rules::verify_rewrite(&seq, std::slice::from_ref(&g2)),
                    "swap conjugation wrong: {g:?} on wires {a},{b}"
                );
                assert_eq!(g2.width(), g.width());
                assert_eq!(g2.comp, g.comp);
                // Involution: conjugating back restores the gate exactly.
                assert_eq!(conj_by_swap(&g2, a, b).unwrap(), g);
            } else {
                assert!(g.target != a && g.target != b && !g.reads(a) && !g.reads(b));
            }
            // Transvection T = cnot(b -> a); gates writing b are excluded by
            // the move's b-selection, so they are out of scope here too.
            if g.target != b {
                let t = cnot(a, b);
                let sandwich = |pieces: &[XGate]| {
                    rules::verify_rewrite(&[t.clone(), g.clone(), t.clone()], pieces)
                };
                match conj_by_cnot(&g, a, b) {
                    CnotConj::Invariant => {
                        assert!(
                            sandwich(std::slice::from_ref(&g)),
                            "cnot-invariant gate is not invariant: {g:?} a={a} b={b}"
                        );
                    }
                    CnotConj::Flip(g2) => {
                        assert!(
                            sandwich(std::slice::from_ref(&g2)),
                            "cnot flip wrong: {g:?} a={a} b={b}"
                        );
                        assert_eq!(g2.width(), g.width());
                        assert_eq!(g2.comp, g.comp);
                    }
                    CnotConj::Split(x, y) => {
                        assert!(!g.comp, "comp gates must be Blocked, not Split");
                        assert!(
                            sandwich(&[x.clone(), y.clone()]),
                            "cnot split wrong: {g:?} a={a} b={b}"
                        );
                        // Disjoint b-slices: the pair commutes.
                        assert!(
                            sandwich(&[y.clone(), x.clone()]),
                            "cnot split pair does not commute"
                        );
                        assert_eq!(x.width(), g.width() + 1);
                        assert_eq!(y.width(), g.width() + 1);
                    }
                    CnotConj::Blocked => {
                        assert!(
                            g.comp && g.reads(a) && !g.reads(b),
                            "spurious Blocked: {g:?}"
                        );
                    }
                }
            }
        }
    }

    // The chain with twist moves enabled at high weight keeps the function
    // (run() global-checks internally), erodes rather than grows fossils, and
    // actually relabels interior gates. Also exercises the journal-stamp
    // interaction: undo entries over relabeled pieces must die, not fire.
    // Target above input: twists add gates without catalogue-invertible bulk,
    // so at target the thermostat pegs near-full contraction and would starve
    // the expansion channel this test is exercising.
    #[test]
    fn mixer_twists_preserve_function() {
        let gates = random_mixed_circuit(17, 16, 300);
        let comp0 = gates.iter().filter(|g| g.comp).count();
        let params = MixParams {
            k_max: 5,
            moves: 20_000,
            target_size: 600,
            temp: 20.0,
            w_twist_neg: 0.10,
            w_twist_swap: 0.10,
            twist_min_len: 4,
            verify_every: 1_000,
            report_every: u64::MAX,
            seed: 5,
            ..MixParams::default()
        };
        let mut mx = Mixer::new(gates, 16, params);
        mx.run();
        // Rate calibration: twist packets are hard for the catalogue to dig
        // out (brackets are wall-blocked by their own window), so at these
        // weights the thermostat pegs near-full contraction and expansions
        // run at ~2% of moves — expect twist counts near 50, not hundreds.
        assert!(
            mx.counters.twist_negs > 30,
            "negation twists barely ran: {}",
            mx.counters.twist_negs
        );
        assert!(
            mx.counters.twist_swaps > 30,
            "swap twists barely ran: {}",
            mx.counters.twist_swaps
        );
        assert!(
            mx.counters.twist_relabels > 0,
            "twists never relabeled a gate"
        );
        assert!(mx.remaining_g57() <= comp0, "fossil count increased");
        assert!(mx.counters.merges() > 0, "no merges alongside twists");
        mx.global_check();
    }

    // The directional walk at its defaults (fresh-split suspended, undo live,
    // directional insert with embedded crosses, birth advance + failed-cross
    // retreat replacing the uniform scatter): function is preserved through
    // every sub-step (local_verify + periodic global checks), inserts really
    // do shoot both copies, and size stays bounded (the bound is a loose
    // runaway canary, not a promise).
    #[test]
    fn directional_walk_preserves_function() {
        let gates = random_mixed_circuit(31, 16, 300);
        let params = MixParams {
            k_max: 5,
            moves: 20_000,
            target_size: 400,
            temp: 20.0,
            w_insert: 0.25,
            verify_every: 2_000,
            report_every: u64::MAX,
            seed: 7,
            ..MixParams::default()
        };
        let mut mx = Mixer::new(gates, 16, params);
        mx.run();
        assert!(
            mx.counters.inserts > 50,
            "inserts barely ran: {}",
            mx.counters.inserts
        );
        assert!(
            mx.counters.cross_r1 + mx.counters.cross_r2 + mx.counters.cross_r3 > 100,
            "crossings barely ran"
        );
        assert!(mx.counters.scatters > 0, "no directional birth advances");
        assert!(
            mx.counters.fresh_splits == 0,
            "fresh splits ran despite suspension"
        );
        let n = mx.arena.len();
        assert!(
            n < 4 * 400,
            "size ran away under merge-only contraction: {n}"
        );
        mx.global_check();
    }

    // Symmetric truncation: with twist_min_len at circuit scale every draw is
    // near-full-length, so ~half the windows left-truncate (virtual start < 0)
    // and their opening packets land at the head. Function preservation +
    // global_check through thousands of such windows exercises the boundary
    // insert path (brackets before the arena head) and the short-window skip
    // paths (len as small as 1).
    #[test]
    fn twist_left_truncated_windows_preserve_function() {
        let gates = random_mixed_circuit(23, 16, 300);
        let params = MixParams {
            k_max: 5,
            moves: 20_000,
            target_size: 600,
            temp: 20.0,
            w_twist_neg: 0.10,
            w_twist_swap: 0.10,
            twist_min_len: usize::MAX, // clamped to circuit size -> len == n
            verify_every: 1_000,
            report_every: u64::MAX,
            seed: 11,
            ..MixParams::default()
        };
        let mut mx = Mixer::new(gates, 16, params);
        mx.run();
        assert!(
            mx.counters.twist_negs + mx.counters.twist_swaps > 50,
            "twists barely ran: {}+{}",
            mx.counters.twist_negs,
            mx.counters.twist_swaps
        );
        mx.global_check();
    }

    // The chain with ONLY the transvection twist enabled keeps the function,
    // case-splits interior a-readers (the affine cost), never grows fossils,
    // and interoperates with the journal/merge machinery. Windows holding a
    // fossil that reads a (without a b-literal) must skip whole — on a mixed
    // circuit both outcomes occur.
    #[test]
    fn mixer_cnot_twists_preserve_function() {
        let gates = random_mixed_circuit(19, 16, 300);
        let comp0 = gates.iter().filter(|g| g.comp).count();
        let params = MixParams {
            k_max: 6,
            moves: 20_000,
            target_size: 600,
            temp: 20.0,
            w_twist_cnot: 0.50,
            twist_min_len: 4,
            verify_every: 1_000,
            report_every: u64::MAX,
            seed: 7,
            ..MixParams::default()
        };
        let mut mx = Mixer::new(gates, 16, params);
        mx.run();
        assert!(
            mx.counters.twist_cnots > 20,
            "cnot twists barely ran: {}",
            mx.counters.twist_cnots
        );
        assert!(
            mx.counters.twist_case_splits > 0,
            "no interior case-split ever happened"
        );
        assert!(mx.remaining_g57() <= comp0, "fossil count increased");
        assert!(mx.counters.merges() > 0, "no merges alongside cnot twists");
        mx.global_check();
    }

    // A nonlinear frame is introduced only as P . reverse(P), then changed by
    // already-proved crossing rewrites. Exhaustive evaluation on a small state
    // space checks the complete move, including transported descendants.
    #[test]
    fn nonlinear_frame_transport_is_exact_and_delivers_coverage() {
        let gates = random_mixed_circuit(41, 8, 120);
        let params = MixParams {
            k_max: 6,
            split_base: 1.0,
            w_nl_frame: 1.0,
            nl_frame_min_width: 2,
            nl_frame_max_width: 3,
            nl_frame_packet_gates: 8,
            nl_frame_shots: 128,
            nl_frame_tenure: 1_000,
            report_every: u64::MAX,
            seed: 29,
            ..MixParams::default()
        };
        let mut mx = Mixer::new(gates.clone(), 8, params);
        mx.nonlinear_frame_move();
        let mixed = mx.arena.to_vec();
        for state in 0..(1u64 << 8) {
            assert_eq!(
                eval_u64(&gates, state),
                eval_u64(&mixed, state),
                "nonlinear frame changed the function on state {state:#x}"
            );
        }
        assert_eq!(mx.counters.nl_frame_attempts, 1);
        assert_eq!(mx.counters.nl_frames, 1);
        assert_eq!(mx.counters.nl_frame_packet_gates, 16);
        assert!(mx.counters.nl_frame_shot_attempts <= 128);
        assert!(
            mx.counters.nl_frame_crossings > 0,
            "frame transport never completed an R-rule crossing"
        );
        assert!(mx.counters.nl_frame_span > 0);
        let (live, tagged, coverage) = mx.active_nonlinear_frame_stats();
        assert_eq!(live, 1);
        assert!(tagged > 0 && coverage > 0.0);
        mx.global_check();
    }

    // Every descendant of a protected frame inherits the deadline. Journal
    // undo sees at least one protected descendant and must leave the entry
    // alone; the same is_protected gate guards catalogue merges.
    #[test]
    fn nonlinear_frame_tenure_protects_descendants_and_journal() {
        let gates = random_mixed_circuit(43, 8, 120);
        let tenure = 100;
        let params = MixParams {
            k_max: 6,
            split_base: 1.0,
            w_nl_frame: 1.0,
            nl_frame_min_width: 2,
            nl_frame_max_width: 3,
            nl_frame_packet_gates: 8,
            nl_frame_shots: 128,
            nl_frame_tenure: tenure,
            undo_frac: 1.0,
            report_every: u64::MAX,
            seed: 31,
            ..MixParams::default()
        };
        let mut mx = Mixer::new(gates, 8, params);
        mx.nonlinear_frame_move();
        let frame_ids: Vec<u32> = mx
            .arena
            .ids_in_order()
            .into_iter()
            .filter(|&id| mx.meta_of(id).frame != 0)
            .collect();
        assert!(!frame_ids.is_empty());
        assert!(frame_ids.iter().all(|&id| mx.is_protected(id)));
        assert!(!mx.journal.is_empty(), "transport created no undo evidence");
        let blocked0 = mx.counters.nl_frame_protected_blocks;
        assert!(!mx.undo_move(), "protected frame crossing was undone early");
        assert!(mx.counters.nl_frame_protected_blocks > blocked0);
        assert!(frame_ids.iter().all(|&id| mx.arena.is_linked(id)));

        mx.moves_done = tenure;
        assert!(frame_ids.iter().all(|&id| !mx.is_protected(id)));
        let (live, tagged, coverage) = mx.active_nonlinear_frame_stats();
        assert!(live > 0 && tagged > 0 && coverage > 0.0);
        assert_eq!(mx.protected_nonlinear_frame_stats(), (0, 0, 0.0));
        mx.global_check();
    }

    // With twist weights at zero no twist path may ever be taken — not even a
    // skipped attempt from floating-point dust in the weight subtractions —
    // so per-move RNG consumption (and hence every seed's trajectory) matches
    // the pre-twist chain exactly.
    #[test]
    fn twist_weights_zero_is_inert() {
        let gates = random_mixed_circuit(3, 16, 300);
        let params = MixParams {
            k_max: 5,
            moves: 10_000,
            target_size: 300,
            temp: 20.0,
            verify_every: 5_000,
            report_every: u64::MAX,
            seed: 5,
            ..MixParams::default()
        };
        let mut inert_geometry = params.clone();
        inert_geometry.nl_frame_min_width = 4;
        inert_geometry.nl_frame_max_width = 5;
        inert_geometry.nl_frame_packet_gates = 37;
        inert_geometry.nl_frame_shots = 999;
        inert_geometry.nl_frame_tenure = 7;
        // Every layer-1/2 geometry/routing knob may vary freely while its
        // admission controls remain disabled. None may consume move RNG.
        inert_geometry.twist_neg_p = 0.93;
        inert_geometry.twist_g57 = true;
        inert_geometry.db_mode = DbMode::Mix;
        inert_geometry.s_db = 19;
        inert_geometry.s_db_comp = 23;
        inert_geometry.p_convex = 0.81;
        inert_geometry.p_convex_comp = 0.07;
        inert_geometry.p_mingen = 0.42;
        inert_geometry.p_mingen_comp = 0.12;
        inert_geometry.mix_pay_random = true;
        inert_geometry.curated = true;
        inert_geometry.curated_exhaust = true;
        inert_geometry.curated_in_comp = true;
        inert_geometry.db_advance = true;
        inert_geometry.prof_cadence_eff = 7.0;
        inert_geometry.prof_deadband = 0.9;
        inert_geometry.prof_dp_max = 0.7;
        inert_geometry.prof_ewma = 0.01;
        inert_geometry.prof_ki = 0.6;
        let mut a = Mixer::new(gates.clone(), 16, params);
        let mut b = Mixer::new(gates, 16, inert_geometry);
        a.run();
        b.run();
        assert_eq!(
            a.arena.to_vec(),
            b.arena.to_vec(),
            "zero nonlinear-frame weight perturbed the legacy RNG trajectory"
        );
        assert_eq!(a.origins_in_order(), b.origins_in_order());
        assert_eq!(
            a.counters.twist_negs
                + a.counters.twist_swaps
                + a.counters.twist_cnots
                + a.counters.twist_skips,
            0
        );
        assert_eq!(a.counters.nl_frame_attempts, 0);
        assert_eq!(b.counters.nl_frame_attempts, 0);
    }

    #[test]
    fn first_class_twist_probability_does_not_double_dose_expansion_twists() {
        let gates = vec![
            XGate::conj(0, [(1, true)]).unwrap(),
            XGate::conj(2, [(3, false)]).unwrap(),
        ];
        let params = MixParams {
            w_cross: 0.0,
            w_fresh: 0.0,
            w_unsub: 0.0,
            w_insert: 1.0,
            // With p_twist enabled this is a kind ratio for twist_round only,
            // not an additional expansion weight.
            w_twist_neg: 1.0,
            p_twist: 0.0016,
            local_verify: false,
            report_every: u64::MAX,
            seed: 57,
            ..MixParams::default()
        };
        let mut mx = Mixer::new(gates, 4, params);
        for _ in 0..32 {
            mx.expand_move();
        }
        assert_eq!(mx.counters.twist_negs, 0);
        assert_eq!(mx.counters.twist_swaps, 0);
        assert_eq!(mx.counters.twist_cnots, 0);
        assert!(
            mx.counters.inserts > 0,
            "ordinary expansion selection did not reach insert moves"
        );
        mx.global_check();
    }

    #[test]
    fn swap_family_twists_preserve_function_and_zero_q_is_pure_swap() {
        let gates = random_mixed_circuit(0x51_57, 8, 80);
        let mut mixer = Mixer::new(
            gates,
            8,
            MixParams {
                twist_neg_p: 0.0,
                twist_min_len: 3,
                hard_size_cap: 1_000,
                report_every: u64::MAX,
                seed: 91,
                ..MixParams::default()
            },
        );
        for _ in 0..12 {
            mixer.twist_swap_family_move();
        }
        assert!(mixer.counters.twist_swaps > 0);
        assert_eq!(mixer.counters.twist_negs, 0);
        assert_eq!(mixer.counters.twist_cnots, 0);
        assert!(mixer.counters.twist_relabels > 0);
        mixer.global_check();
    }

    #[test]
    fn g57_v2_slide_scan_finds_a_supported_attachment() {
        let gates = vec![
            XGate::x_gate(0),
            XGate::x_gate(3),
            XGate::from_g57([2, 0, 1]),
        ];
        let mixer = Mixer::new(gates, 4, MixParams::default());
        let ids = mixer.arena.ids_in_order();
        assert_eq!(
            mixer.g57_slide_candidates(ids[0], Dir::R, 0, 1),
            vec![ids[1]]
        );
    }

    #[test]
    fn g57_v2_joint_plans_preserve_function_and_report_seams() {
        let gates = random_mixed_circuit(0x83ae_0d11, 8, 80);
        let mut mixer = Mixer::new(
            gates,
            8,
            MixParams {
                twist_g57: true,
                twist_min_len: 3,
                hard_size_cap: 1_000,
                local_verify: true,
                report_every: u64::MAX,
                seed: 83,
                ..MixParams::default()
            },
        );
        for _ in 0..8 {
            mixer.twist_move_g57();
        }
        let applied = mixer.counters.twist_swaps;
        assert!(applied > 0);
        assert!(mixer.counters.tg_solves > 0);
        assert_eq!(
            mixer.counters.tg_net_hist.iter().sum::<u64>(),
            applied * 2,
            "each committed twist contributes two seam shapes"
        );
        assert!(mixer.counters.tg_slides <= applied * 2);
        assert!(mixer.counters.tg_retries <= applied * (TG_RETRIES as u64 - 1));
        mixer.global_check();
    }

    #[test]
    fn profile_target_and_controller_complete_all_three_legs() {
        let n = [0.02, 0.04, 0.06];
        let r = [1.2, 1.0];
        assert_eq!(prof_target(n, r, 100.0, 0.0), 100.0);
        assert!((prof_target(n, r, 100.0, 0.02) - 120.0).abs() < 1e-9);
        assert!((prof_target(n, r, 100.0, 0.04) - 120.0).abs() < 1e-9);
        assert!((prof_target(n, r, 100.0, 0.06) - 100.0).abs() < 1e-9);

        let gates = random_mixed_circuit(44, 8, 64);
        let mut mixer = Mixer::new(
            gates,
            8,
            MixParams {
                moves: 1_000,
                prof_n: n,
                prof_r: r,
                prof_cadence_eff: 0.01,
                report_every: u64::MAX,
                verify_every: 100,
                seed: 4,
                ..MixParams::default()
            },
        );
        assert!(matches!(mixer.run(), MixStop::ProfileDone));
        let profile = mixer.prof.as_ref().expect("profile state retained");
        assert_eq!(profile.phase, 4);
        assert!((0.0..=1.0).contains(&profile.pmix));
        mixer.global_check();
    }

    #[test]
    fn generation_census_and_global_pass_retag_are_exact() {
        let gates = vec![
            XGate::conj(0, [(1, true)]).unwrap(),
            XGate::conj(2, [(3, false)]).unwrap(),
            XGate::conj(4, [(5, true)]).unwrap(),
        ];
        let mut mx = Mixer::new(gates, 8, MixParams::default());
        assert_eq!(mx.gens_in_order(), vec![0, 0, 0]);
        assert_eq!(mx.generation_histogram(), BTreeMap::from([(0, 3)]));
        assert_eq!(
            mx.gen_stats_for(1),
            GenStats {
                lag: 3,
                elig: 3,
                wlag: 0,
                min: 0,
                all_lag: 3,
                fresh: 0,
                total: 3,
                g_circ: 0,
                g_all: 0,
                targetable: 3,
            }
        );

        assert_eq!(mx.advance_all_generations_after_global_pass(), 1);
        assert_eq!(mx.gens_in_order(), vec![1, 1, 1]);
        assert_eq!(mx.advance_all_generations_after_global_pass(), 2);
        let target = GenerationTarget {
            min_generation: 2,
            fraction: 1.0,
        };
        let stats = mx.generation_stats(target);
        assert_eq!(stats.total, 3);
        assert_eq!(stats.reached, 3);
        assert_eq!(stats.reached_fraction, 1.0);
        assert_eq!(
            (stats.min, stats.quantile_floor, stats.median, stats.max),
            (2, 2, 2, 2)
        );
        assert!(mx.generation_goal_met(target));
        assert_eq!(mx.gen_stats_for(2).g_circ, 2);
    }

    #[test]
    fn wide_majority_does_not_pin_targetable_generation_or_dose() {
        let mut gates = Vec::new();
        let mut rng = StdRng::seed_from_u64(4242);
        while gates.len() < 30 {
            let gate = rand_gate(&mut rng, 16, 3, false);
            if gate.width() == 3 {
                gates.push(gate);
            }
        }
        while gates.len() < 40 {
            let gate = rand_gate(&mut rng, 16, 2, false);
            if (1..=2).contains(&gate.width()) {
                gates.push(gate);
            }
        }
        let params = MixParams {
            gen_target: 100,
            gen_stop_frac: 0.02,
            // Old `db_ctrl_cap: 2` (width <= 2) becomes `w_pool: 3` (width < 3).
            w_pool: 3,
            report_every: u64::MAX,
            ..MixParams::default()
        };
        let mut mixer = Mixer::new(gates, 16, params);
        for id in mixer.arena.ids_in_order() {
            if mixer.width_of(id) <= 2 {
                let meta = mixer.meta_of(id);
                mixer.set_meta(id, Meta { dgen: 100, ..meta });
            }
        }

        let stats = mixer.gen_stats();
        assert_eq!((stats.all_lag, stats.total), (30, 40));
        assert_eq!(stats.wlag, 30);
        assert_eq!(stats.elig, 10);
        assert_eq!(stats.targetable, 10);
        assert_eq!(stats.lag, 0);
        assert_eq!(
            stats.g_all, 0,
            "the legacy all-gate percentile stays pinned"
        );
        assert_eq!(
            stats.g_circ, 100,
            "G must measure the population the DB channel can move"
        );
        assert!(mixer.dose_reached());
    }

    #[test]
    fn an_empty_targetable_population_never_meets_the_dose() {
        let gates = vec![
            XGate::conj(0, [(1, true)]).unwrap(),
            XGate::conj(2, [(3, false)]).unwrap(),
        ];
        let params = MixParams {
            gen_target: 10,
            gen_stop_frac: 1.0,
            // With the miss/give-up tiers gone, an empty targetable
            // population is built by width: w_pool = 1 makes every gate
            // (width >= 2) pool-ineligible.
            w_pool: 1,
            report_every: u64::MAX,
            ..MixParams::default()
        };
        let mixer = Mixer::new(gates, 4, params);

        let stats = mixer.gen_stats();
        assert_eq!(stats.wlag, 2);
        assert_eq!(stats.targetable, 0);
        assert_eq!(
            stats.g_circ, stats.g_all,
            "display fallback remains defined"
        );
        assert!(
            !mixer.dose_reached(),
            "an unmeasurable dose must fail closed"
        );
    }

    #[test]
    fn synthetic_identity_material_uses_fresh_generation_sentinel() {
        let gate = XGate::conj(0, [(1, true)]).unwrap();
        let mut mx = Mixer::new(
            vec![gate],
            8,
            MixParams {
                k_max: 4,
                split_base: 1.0,
                local_verify: true,
                seed: 17,
                ..MixParams::default()
            },
        );
        mx.insert_move();
        let ids = mx.arena.ids_in_order();
        let synthetic: Vec<u32> = ids
            .into_iter()
            .filter(|&id| mx.meta_of(id).origin == ORIGIN_SYNTH)
            .collect();
        assert!(
            !synthetic.is_empty(),
            "insert emitted no synthetic descendants"
        );
        assert!(
            synthetic.iter().all(|&id| mx.meta_of(id).dgen == GEN_FRESH),
            "born-random identity material or a split descendant lost GEN_FRESH"
        );
        mx.global_check();
    }

    #[test]
    fn db_generation_stamp_uses_selectable_median_and_saturates() {
        let gates = vec![
            XGate::conj(0, [(1, true)]).unwrap(),
            XGate::conj(2, [(3, true)]).unwrap(),
        ];
        let mut mx = Mixer::new(gates, 4, MixParams::default());
        let ids = mx.arena.ids_in_order();
        let mut left = mx.meta_of(ids[0]);
        left.dgen = 3;
        mx.set_meta(ids[0], left);
        let mut right = mx.meta_of(ids[1]);
        right.dgen = 7;
        mx.set_meta(ids[1], right);
        assert_eq!(mx.db_replacement_generation(&ids, false), 8);
        assert_eq!(mx.db_replacement_generation(&ids, true), 4);

        for &id in &ids {
            let mut meta = mx.meta_of(id);
            meta.dgen = GEN_FRESH;
            mx.set_meta(id, meta);
        }
        assert_eq!(
            mx.db_replacement_generation(&ids, false),
            GEN_FRESH,
            "an all-fresh DB window must remain fresh"
        );
    }

    #[test]
    fn db_splice_preserves_g57_frame_metadata_and_generation() {
        let payload = XGate::from_g57([0, 1, 2]);
        let pad = XGate::conj(3, [(4, true), (5, false)]).unwrap();
        let window = vec![pad.clone(), pad, payload.clone()];
        let mut mx = Mixer::new(
            window.clone(),
            8,
            MixParams {
                db_verify: true,
                ..MixParams::default()
            },
        );
        mx.moves_done = 20;
        let ids = mx.arena.ids_in_order();
        let frame = g57_pair_frame(9, 1);
        for (index, &id) in ids.iter().enumerate() {
            let mut meta = mx.meta_of(id);
            meta.origin = 77;
            meta.frame = frame;
            meta.protected_until = 10;
            meta.dgen = [1, 7, 3][index];
            mx.set_meta(id, meta);
        }
        assert!(mx.try_db_splice(
            &ids,
            Dir::R,
            &window,
            vec![payload.clone()],
            1,
            DbMode::Compressing,
        ));
        assert_eq!(mx.arena.to_vec(), vec![payload]);
        let output = mx.arena.ids_in_order()[0];
        let meta = mx.meta_of(output);
        assert_eq!(meta.origin, 77);
        assert_eq!(meta.frame, frame);
        assert_eq!(meta.protected_until, 10);
        assert_eq!(meta.dgen, 4, "median(1,3,7)+1");
        assert_eq!(mx.counters.db_comp_hits, 1);
        assert_eq!(mx.counters.db_gates_removed, 2);
        mx.global_check();
    }

    /// The coexistence contract: a splice preserves the folded frame AND mints
    /// one fresh litter for every product — opposite lifecycles in separate
    /// fields, never overloaded into each other.
    #[test]
    fn litters_coexist_with_frames_and_the_ban_refuses_full_litters() {
        let payload = XGate::from_g57([0, 1, 2]);
        let tail = XGate::from_g57([3, 4, 5]);
        let pad = XGate::conj(6, [(7, true)]).unwrap();
        // pad·pad cancels, so the window is exactly [payload, tail].
        let window = vec![pad.clone(), pad.clone(), payload.clone(), tail.clone()];
        let mut mx = Mixer::new(window.clone(), 8, MixParams::default());
        let input_count = 4u64;
        let ids = mx.arena.ids_in_order();
        let frame = g57_pair_frame(4, 0);
        for &id in &ids {
            let mut meta = mx.meta_of(id);
            meta.frame = frame;
            mx.set_meta(id, meta);
        }
        let input_litters: Vec<u64> = ids.iter().map(|&id| mx.meta_of(id).litter).collect();
        assert_eq!(
            input_litters,
            vec![0, 1, 2, 3],
            "inputs are singleton litters"
        );

        let replacement = vec![payload.clone(), tail.clone()];
        assert!(mx.try_db_splice(&ids, Dir::R, &window, replacement, 1, DbMode::Compressing,));
        let out = mx.arena.ids_in_order();
        assert_eq!(out.len(), 2);
        let m0 = mx.meta_of(out[0]);
        let m1 = mx.meta_of(out[1]);
        assert_eq!(m0.frame, frame, "frame preserved through the splice");
        assert_eq!(m1.frame, frame);
        assert_eq!(m0.litter, m1.litter, "one litter per splice");
        assert!(
            m0.litter >= input_count,
            "the splice litter is fresh, not an input's"
        );
        assert_eq!(m0.litter_size, 2, "size fixed at creation");

        // The census the ban reads: the products form exactly one complete
        // litter, and any strict sub-window reads incomplete — descending past
        // a banned rung is free.
        assert_eq!(mx.litter_census(&out), (1, true));
        assert_eq!(mx.litter_census(&out[..1]), (1, false));
        mx.global_check();
    }

    #[test]
    fn checkpoint_round_trip_preserves_chain_state() {
        let gates = random_mixed_circuit(0xc4e0, 10, 120);
        let params = MixParams {
            moves: 400,
            k_max: 5,
            gen_target: 50,
            p_mingen: 0.7,
            canary_theta: 0.9,
            canary_window: 8,
            size_hi: 4000,
            size_lo: 100,
            prof_n: [2.0, 3.0, 4.0],
            prof_r: [1.3, 1.0],
            report_every: u64::MAX,
            verify_every: u64::MAX,
            seed: 77,
            ..MixParams::default()
        };
        let mut mx = Mixer::new(gates, 10, params.clone());
        mx.run();
        // Decorate with state every subsystem owns: frames, protection,
        // a part-filled canary ring, an armed brake, a live profile clock.
        let ids = mx.arena.ids_in_order();
        let mut meta = mx.meta_of(ids[0]);
        meta.frame = g57_pair_frame(3, 1);
        meta.protected_until = u64::MAX;
        mx.set_meta(ids[0], meta);
        mx.canary.push_back(true);
        mx.canary.push_back(false);
        mx.canary_failures = 1;
        mx.brake_on = true;
        mx.brake_mark_move = 123;
        mx.brake_mark_size = 456;
        mx.rebuild_pool();
        mx.prof_init();

        let dir = std::env::temp_dir().join(format!("fmix_ckpt_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p1 = dir.join("a.state");
        let p1s = p1.to_str().unwrap().to_string();
        mx.save_state(&p1s).unwrap();

        let mut back = Mixer::resume_state(&p1s, params, None).unwrap();
        assert_eq!(back.arena.to_vec(), mx.arena.to_vec(), "gates in order");
        assert_eq!(back.original().to_vec(), mx.original().to_vec());
        assert_eq!(back.moves_done, mx.moves_done);
        assert_eq!(back.counters.moves, mx.counters.moves);
        assert_eq!(back.next_event, mx.next_event);
        assert_eq!(back.next_litter, mx.next_litter);
        assert_eq!(back.pool_scan_due, mx.pool_scan_due);
        assert_eq!(back.canary, mx.canary);
        assert_eq!(back.canary_failures, mx.canary_failures);
        assert!(back.brake_on);
        assert_eq!(back.brake_mark_move, 123);
        assert_eq!(back.brake_mark_size, 456);
        assert_eq!(back.tabu, mx.tabu);
        assert_eq!(back.counters.to_line(), mx.counters.to_line());
        // Metadata: every field of every gate, including the two frame fields
        // and litters.
        let old_ids = mx.arena.ids_in_order();
        let new_ids = back.arena.ids_in_order();
        for (&o, &n) in old_ids.iter().zip(new_ids.iter()) {
            assert_eq!(mx.meta_of(o), back.meta_of(n), "meta preserved");
        }
        // The pool survives (remapped): same gates by position.
        let old_pos: HashMap<u32, usize> =
            old_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let new_pos: HashMap<u32, usize> =
            new_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let mut old_pool: Vec<usize> = mx.pool.iter().map(|id| old_pos[id]).collect();
        let mut new_pool: Vec<usize> = back.pool.iter().map(|id| new_pos[id]).collect();
        old_pool.sort_unstable();
        new_pool.sort_unstable();
        assert_eq!(old_pool, new_pool, "pool survives the renumbering");
        // Profile clock continues rather than restarting at eff = 0.
        let (a, b) = (mx.prof.as_ref().unwrap(), back.prof.as_ref().unwrap());
        assert_eq!(a.phase, b.phase);
        assert!((a.eff - b.eff).abs() < 1e-12);
        assert!((a.pmix - b.pmix).abs() < 1e-12);
        assert!((a.s_in - b.s_in).abs() < 1e-12);
        back.global_check();

        // A second save differs only in the two rng lines.
        let p2 = dir.join("b.state");
        let p2s = p2.to_str().unwrap().to_string();
        back.save_state(&p2s).unwrap();
        let t1 = std::fs::read_to_string(&p1s).unwrap();
        let t2 = std::fs::read_to_string(&p2s).unwrap();
        let differing: Vec<(&str, &str)> =
            t1.lines().zip(t2.lines()).filter(|(x, y)| x != y).collect();
        // rng lines redraw by design; db_mode is diagnostic-only and the
        // resumed value comes from the CLI (params.db_mode), not the file.
        assert!(
            differing.len() <= 3,
            "only rng and db_mode may differ: {differing:?}"
        );
        assert!(differing.iter().all(|(x, _)| x.starts_with("rng")
            || x.starts_with("metrics_rng")
            || x.starts_with("db_mode")));

        // A v1 (origin-format) file refuses cleanly.
        let p3 = dir.join("v1.state");
        std::fs::write(&p3, "fmix-state 1\nwires 4\n").unwrap();
        let err = match Mixer::resume_state(p3.to_str().unwrap(), MixParams::default(), None) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a v1 state file must refuse to load"),
        };
        assert!(err.contains("refusing to reinterpret"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn db_splice_rejects_conflicting_g57_pair_frames() {
        let payload = XGate::from_g57([0, 1, 2]);
        let pad = XGate::conj(3, [(4, true), (5, false)]).unwrap();
        let window = vec![pad.clone(), pad, payload.clone()];
        let mut mx = Mixer::new(
            window.clone(),
            8,
            MixParams {
                db_verify: true,
                ..MixParams::default()
            },
        );
        let ids = mx.arena.ids_in_order();
        for (&id, frame) in ids
            .iter()
            .zip([g57_pair_frame(9, 0), g57_pair_frame(10, 1), 0])
        {
            let mut meta = mx.meta_of(id);
            meta.frame = frame;
            mx.set_meta(id, meta);
        }

        assert!(!mx.try_db_splice(&ids, Dir::R, &window, vec![payload], 1, DbMode::Compressing,));
        assert_eq!(mx.arena.to_vec(), window);
        assert_eq!(mx.counters.db_frame_skips, 1);
        assert_eq!(mx.counters.db_comp_hits, 0);
        assert_eq!(mx.counters.db_comp_misses, 1);
        mx.global_check();
    }

    #[test]
    fn db_splice_rejects_protected_and_over_cap_transactions() {
        let gate = XGate::from_g57([0, 1, 2]);
        let mut protected = Mixer::new(vec![gate.clone()], 6, MixParams::default());
        let ids = protected.arena.ids_in_order();
        let mut meta = protected.meta_of(ids[0]);
        meta.frame = 42;
        meta.protected_until = 100;
        protected.set_meta(ids[0], meta);
        let before = protected.arena.to_vec();
        assert!(!protected.try_db_splice(
            &ids,
            Dir::R,
            &before,
            before.clone(),
            1,
            DbMode::SizeAgnostic,
        ));
        assert_eq!(protected.arena.to_vec(), before);
        assert_eq!(protected.counters.db_protected_skips, 1);
        assert_eq!(protected.counters.db_agn_hits, 0);

        let pad = XGate::from_g57([3, 4, 5]);
        let replacement = vec![pad.clone(), pad, gate.clone()];
        let mut capped = Mixer::new(
            vec![gate.clone()],
            6,
            MixParams {
                hard_size_cap: 1,
                db_verify: true,
                ..MixParams::default()
            },
        );
        let ids = capped.arena.ids_in_order();
        let window = capped.arena.to_vec();
        assert!(!capped.try_db_splice(&ids, Dir::L, &window, replacement, 1, DbMode::MinGrow,));
        assert_eq!(capped.arena.to_vec(), vec![gate]);
        assert_eq!(capped.counters.db_cap_skips, 1);
        assert_eq!(capped.counters.db_gates_added, 0);
        assert_eq!(capped.counters.db_agn_hits, 0);
        capped.global_check();
    }

    #[test]
    fn generation_pool_and_dose_stop_are_live_without_a_db() {
        let gates = vec![
            XGate::conj(0, [(1, true)]).unwrap(),
            XGate::conj(2, [(3, true)]).unwrap(),
        ];
        let mut mx = Mixer::new(
            gates,
            4,
            MixParams {
                gen_target: 2,
                gen_stop_frac: 1.0,
                moves: 10,
                report_every: 1,
                verify_every: u64::MAX,
                ..MixParams::default()
            },
        );
        mx.rebuild_pool();
        assert_eq!(mx.pool.len(), 2);
        // A gate re-encoded past the target leaves the pool on rebuild.
        let first = mx.arena.ids_in_order()[0];
        let meta = mx.meta_of(first);
        mx.set_meta(first, Meta { dgen: 2, ..meta });
        mx.rebuild_pool();
        assert_eq!(mx.pool.len(), 1);
        assert!(matches!(mx.run(), MixStop::DoseReached));
        assert_eq!(mx.counters.moves, 1);
    }

    #[test]
    fn pool_keeps_only_k_lowest_generations() {
        let gates: Vec<XGate> = (0..6)
            .map(|i| XGate::conj(2 * i, [(2 * i + 1, true)]).unwrap())
            .collect();
        let mut mx = Mixer::new(
            gates,
            12,
            MixParams {
                gen_target: 100,
                pool_k: 3,
                ..MixParams::default()
            },
        );
        for (rank, id) in mx.arena.ids_in_order().into_iter().enumerate() {
            let meta = mx.meta_of(id);
            mx.set_meta(
                id,
                Meta {
                    dgen: rank as u32,
                    ..meta
                },
            );
        }
        mx.rebuild_pool();
        assert_eq!(mx.pool.len(), 3);
        let mut gens: Vec<u32> = mx.pool.iter().map(|&id| mx.meta_of(id).dgen).collect();
        gens.sort_unstable();
        assert_eq!(gens, vec![0, 1, 2], "exactly the k lowest generations");
    }

    #[test]
    fn pick_seed_targets_pool_and_prunes_stale() {
        let gates = vec![
            XGate::conj(0, [(1, true)]).unwrap(),
            XGate::conj(2, [(3, true)]).unwrap(),
        ];
        let mut mx = Mixer::new(
            gates,
            4,
            MixParams {
                gen_target: 5,
                p_mingen: 1.0,
                ..MixParams::default()
            },
        );
        mx.rebuild_pool();
        assert_eq!(mx.pool.len(), 2);
        // Unlink one pooled gate; the draw must prune it rather than seed it.
        let stale = mx.pool[0];
        mx.arena.unlink(stale);
        for _ in 0..16 {
            if let Some(seed) = mx.pick_seed() {
                assert_ne!(seed, stale, "stale entries must be pruned, not drawn");
                assert!(mx.seed_from_pool, "p_mingen=1.0 must draw from the pool");
            }
        }
        assert!(!mx.pool.contains(&stale), "stale entry pruned lazily");
    }

    /// Guards Decision D forever: protected (framed) and wide gates never
    /// enter the pool, so pool seeding cannot target the G57 pair census.
    #[test]
    fn pool_excludes_protected_and_wide_gates() {
        let narrow = XGate::conj(0, [(1, true)]).unwrap();
        let protected = XGate::conj(2, [(3, true)]).unwrap();
        let wide = XGate::conj(4, [(5, true), (6, false), (7, true)]).unwrap();
        let mut mx = Mixer::new(
            vec![narrow, protected, wide],
            10,
            MixParams {
                gen_target: 100,
                w_pool: 3,
                ..MixParams::default()
            },
        );
        let ids = mx.arena.ids_in_order();
        let meta = mx.meta_of(ids[1]);
        mx.set_meta(
            ids[1],
            Meta {
                protected_until: u64::MAX,
                ..meta
            },
        );
        mx.rebuild_pool();
        assert_eq!(mx.pool, vec![ids[0]], "only the narrow unprotected gate");
    }

    #[test]
    fn canary_fires_when_the_pool_is_unspellable() {
        let gates: Vec<XGate> = (0..4)
            .map(|i| XGate::conj(2 * i, [(2 * i + 1, true)]).unwrap())
            .collect();
        let mut mx = Mixer::new_with_optional_db(
            gates,
            8,
            MixParams {
                gen_target: 100,
                p_mingen: 1.0,
                canary_theta: 0.5,
                canary_window: 4,
                db_min_window: 2,
                db_prefixes: true,
                report_every: u64::MAX,
                ..MixParams::default()
            },
            Some(FrozenDb::empty()),
        );
        mx.rebuild_pool();
        assert!(!mx.pool.is_empty());
        // The empty store misses every rung, so each pool-seeded Mix attempt
        // is a canary failure.
        for _ in 0..4 {
            assert!(!mx.db_attempt(DbMode::Mix));
        }
        assert_eq!(mx.canary.len(), 4, "four qualifying rounds recorded");
        assert_eq!(mx.canary_failures, 4);
        assert!(mx.canary_fired());
        // The sleep contract: Compressing rounds do not qualify.
        let ring_before = mx.canary.len();
        mx.db_attempt(DbMode::Compressing);
        assert_eq!(mx.canary.len(), ring_before);
        // Theta is the master switch.
        mx.params.canary_theta = 0.0;
        assert!(!mx.canary_fired());
        mx.global_check();
    }

    #[test]
    fn size_brake_engages_and_releases() {
        let gates: Vec<XGate> = (0..6)
            .map(|i| XGate::conj(2 * i, [(2 * i + 1, true)]).unwrap())
            .collect();
        let mut mx = Mixer::new(
            gates,
            12,
            MixParams {
                size_hi: 4,
                size_lo: 2,
                // The release condition is strict (`rate < eps`), so eps must
                // be positive for a zero shed rate to release.
                comp_release_eps: 0.001,
                comp_release_window: 100,
                db_mode: DbMode::SizeAgnostic,
                ..MixParams::default()
            },
        );
        assert_eq!(mx.db_mode_cur, DbMode::SizeAgnostic);

        // Engage: 6 gates >= size_hi = 4.
        mx.apply_size_brake();
        assert!(mx.brake_on);
        assert_eq!(mx.db_mode_cur, DbMode::Compressing);
        assert_eq!(mx.counters.brake_engagements, 1);

        // The 3c trap, pinned forever: an off overlay must NOT clobber the
        // brake's steer.
        mx.params.p_mix = -1.0;
        mx.apply_mode_overlay();
        assert_eq!(mx.db_mode_cur, DbMode::Compressing);

        // Productivity release: a full window with zero shed lets go.
        mx.moves_done += 100;
        mx.apply_size_brake();
        assert!(!mx.brake_on, "zero shed rate must release");
        assert_eq!(mx.db_mode_cur, DbMode::SizeAgnostic);

        // Re-engage, then hysteresis release at size_lo.
        mx.apply_size_brake();
        assert!(mx.brake_on);
        let ids = mx.arena.ids_in_order();
        for &id in &ids[2..] {
            mx.arena.unlink(id);
        }
        mx.apply_size_brake();
        assert!(!mx.brake_on, "size_lo hysteresis must release");
        assert_eq!(mx.db_mode_cur, DbMode::SizeAgnostic);

        // size_hi == 0 is the off switch.
        mx.params.size_hi = 0;
        let mode_before = mx.db_mode_cur;
        mx.apply_size_brake();
        assert_eq!(mx.db_mode_cur, mode_before);
        assert!(!mx.brake_on);
    }

    #[test]
    fn disabled_db_configuration_is_rng_trajectory_inert() {
        let gates = random_mixed_circuit(0xdbee, 12, 180);
        let base = MixParams {
            k_max: 5,
            moves: 5_000,
            target_size: 240,
            temp: 20.0,
            verify_every: 2_500,
            report_every: u64::MAX,
            seed: 44,
            ..MixParams::default()
        };
        let mut inert_db = base.clone();
        inert_db.db_min_window = 7;
        inert_db.db_max_window = 31;
        inert_db.db_sample = DbSample::Mixed;
        inert_db.w_window = 4;
        inert_db.w_pool = 3;
        inert_db.pool_k = 7;
        inert_db.db_convex_p = 0.11;
        inert_db.db_verify = false;
        inert_db.db_dry_run = true;
        inert_db.db_max_degree = 9;
        inert_db.db_degree_probes = 17;
        inert_db.db_max_span = 13;
        inert_db.db_prefixes = true;
        inert_db.gen_rescan = 3;

        let mut baseline = Mixer::new(gates.clone(), 12, base);
        let mut instrumented = Mixer::new(gates, 12, inert_db);
        baseline.run();
        instrumented.run();
        assert_eq!(baseline.arena.to_vec(), instrumented.arena.to_vec());
        assert_eq!(baseline.origins_in_order(), instrumented.origins_in_order());
        assert_eq!(baseline.gens_in_order(), instrumented.gens_in_order());
        assert_eq!(instrumented.counters.db_attempts, 0);
    }

    #[test]
    fn hard_size_cap_blocks_non_db_growth_atomically() {
        let gate = XGate::conj(0, [(1, true)]).unwrap();
        let mut mx = Mixer::new(
            vec![gate.clone()],
            8,
            MixParams {
                hard_size_cap: 1,
                k_max: 4,
                ..MixParams::default()
            },
        );
        mx.insert_move();
        assert_eq!(mx.arena.to_vec(), vec![gate]);
        assert_eq!(mx.counters.blocked_size_cap, 1);
        mx.global_check();
    }

    #[test]
    fn bounded_generation_hardening_respects_cap_and_lifts_children() {
        let gates = vec![
            XGate::conj(0, [(1, true)]).unwrap(),
            XGate::conj(2, [(3, false)]).unwrap(),
            XGate::conj(4, [(5, true)]).unwrap(),
            XGate::conj(6, [(7, false)]).unwrap(),
        ];
        let mut mx = Mixer::new(
            gates,
            8,
            MixParams {
                k_max: 2,
                split_base: 1.0,
                local_verify: true,
                seed: 91,
                ..MixParams::default()
            },
        );
        let report = mx.harden_low_generation_pass(GenerationPassConfig {
            target: GenerationTarget {
                min_generation: 1,
                fraction: 1.0,
            },
            pass_length: 100,
            size_cap: Some(6),
            candidate_trials: 4,
        });
        assert_eq!(report.attempted, 2);
        assert_eq!(report.applied, 2);
        assert_eq!(report.blocked, 0);
        assert!(report.cap_reached);
        assert_eq!(mx.arena.len(), 6, "hard cap was crossed or under-filled");
        assert_eq!(
            mx.gens_in_order()
                .into_iter()
                .filter(|&generation| generation == 1)
                .count(),
            4,
            "each one-to-two rewrite must stamp both children parent+1"
        );
        mx.global_check();
    }

    #[test]
    fn contraction_only_merge_uses_min_parent_generation() {
        let gates = vec![
            XGate::conj(0, [(1, true)]).unwrap(),
            XGate::conj(0, [(1, false)]).unwrap(),
        ];
        let mut mx = Mixer::new(
            gates,
            2,
            MixParams {
                undo_frac: 0.0,
                local_verify: true,
                seed: 5,
                ..MixParams::default()
            },
        );
        let ids = mx.arena.ids_in_order();
        let mut left = mx.meta_of(ids[0]);
        left.dgen = 3;
        mx.set_meta(ids[0], left);
        let mut right = mx.meta_of(ids[1]);
        right.dgen = 7;
        mx.set_meta(ids[1], right);

        let report = mx.contract_to(1, 8, 2, 0.0);
        assert!(report.target_reached);
        assert!(!report.stalled);
        assert_eq!((report.start_size, report.end_size), (2, 1));
        assert_eq!(report.merges, 1);
        assert_eq!(mx.gens_in_order(), vec![3]);
        mx.global_check();
    }

    #[test]
    fn tabu_age_semantics() {
        let gates = random_g57_circuit(1, 8, 20);
        let params = MixParams {
            tabu_moves: 100,
            ..MixParams::default()
        };
        let mut mx = Mixer::new(gates, 8, params);
        let e1 = mx.fresh_event();
        mx.moves_done = 50;
        let e2 = mx.fresh_event();
        assert!(mx.is_tabu(e1) && mx.is_tabu(e2));
        assert!(!mx.is_tabu(0), "event 0 (no provenance) is never tabu");
        mx.moves_done = 120; // e1 aged out (created at move 0), e2 still fresh
        assert!(!mx.is_tabu(e1), "aged-out event must not be tabu");
        assert!(mx.is_tabu(e2));
        mx.moves_done = 200;
        let _ = mx.fresh_event(); // push triggers eviction of expired entries
        assert!(mx.tabu.len() <= 2, "expired tabu entries must be evicted");
    }
}
