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
use super::db_replace::{db_replace, DbMode, DegreeGuard};
use super::rules::{self, BlockReason, Outcome, Role, RuleKind};
use super::xgate::{Lits, XGate};
use super::xpoly::XPolyBudget;
use crate::replace::frozen::FrozenDb;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};

pub const ORIGIN_SYNTH: u32 = u32::MAX;

/// A gate pattern the twist placer looks for when choosing where to put a
/// twist's bracket. Absorption is the point: a bracket dropped next to a gate
/// that can swallow it costs nothing, where one dropped at a uniform-random
/// position has to be paid for. The set of welcoming shapes is small and known
/// ahead of time, so it is a table rather than a search.
///
/// `matches` inspects `span` consecutive gates and, on a hit, names the wire
/// the twist should prefer for its conjugating involution -- normally the
/// absorbing gate's target, since that is the wire whose bracket it can eat.
pub struct TwistPattern {
    pub name: &'static str,
    pub span: usize,
    pub matches: fn(&[XGate]) -> Option<u16>,
}

/// A verified hidden-swap identity, in g57 notation `[x,y,z]` = `x ^= y OR !z`
/// (exhaustively checked over all 8 inputs on wires a,b,c):
///
/// ```text
///   [a,b,c] . swap(a,b) . [b,c,a]  ==  [b,a,c] . [b,c,a] . [a,b,c] . [a,c,b]
/// ```
///
/// The left side is two g57s bracketing a wire swap; the right is four g57s and
/// no swap at all. So a swap conjugation sited between a matching g57 pair
/// costs +2 gates in this form against +6 for the three-CNOT realisation, and
/// leaves no swap-shaped fingerprint behind -- the whole neighbourhood is
/// ordinary g57 material afterwards.
///
/// NOT YET CONSUMED. Taking it needs the rewrite path (emit the four-g57 form
/// in place of the pair-plus-bracket), which is a different operation from
/// choosing where to put a twist; the placer below only sites brackets. Kept
/// here so the identity is not lost between sessions.
pub const HIDDEN_SWAP_IDENTITY: &str =
    "[a,b,c].swap(a,b).[b,c,a] == [b,a,c].[b,c,a].[a,b,c].[a,c,b]";

/// The pattern table. Deliberately small for now: this is the machinery, and
/// the full menu of welcoming neighbourhood configurations is precomputed
/// separately. The one entry here is the case the merge catalogue already
/// settles -- a comp=1 gate absorbs a NOT on its target and comes out comp=0,
/// which erodes a fossil in the bargain (see Merge::Absorb).
pub static TWIST_PATTERNS: &[TwistPattern] = &[TwistPattern {
    name: "comp1-absorber",
    span: 1,
    matches: |gs| {
        let g = gs.first()?;
        if g.comp && !g.ctrls.is_empty() { Some(g.target) } else { None }
    },
}];

/// Largest window/replacement length tracked in the splice size histogram;
/// anything longer is folded into the top bucket.
pub const SPLICE_HIST_MAX: usize = 16;

/// How a DB move samples its outgoing window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DbSample {
    /// Pick a gate g, take it plus its w-1 neighbors in g's own direction,
    /// falling back to the other direction when the circuit end is reached.
    Contiguous,
    /// Grow a convex (mutually-gatherable) block: float g1 to its first
    /// non-commuting neighbor, then repeatedly float the whole block — in g1's
    /// direction w.p. p, else the opposite — to the next non-commuting gate and
    /// absorb it, until w gates are collected.
    Convex,
    /// Flip a fair coin per attempt between Contiguous and Convex.
    Mixed,
}

impl DbSample {
    pub fn parse(s: &str) -> Option<DbSample> {
        match s {
            "contiguous" => Some(DbSample::Contiguous),
            "convex" => Some(DbSample::Convex),
            "mixed" => Some(DbSample::Mixed),
            _ => None,
        }
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
    // A bare NOT absorbed into a comp=1 gate of ANY width: 1 XOR (1 XOR M) = M,
    // i.e. the partner with its comp bit cleared. Always a single monomial, so
    // it is always a legal merge -- but the catalogue used to refuse it for
    // width >= 2 because the comp guard rejects every comp-differing pair. That
    // guard is right for a comp=0 partner (the result would CREATE a fossil);
    // for a comp=1 partner the result clears one, which is the allowed
    // direction. This is what lets a twist bracket be swallowed by a
    // neighbouring g57 instead of being paid for.
    Absorb(XGate),
}

impl Merge {
    pub fn gates(&self) -> Vec<XGate> {
        match self {
            Merge::Cancel => vec![],
            Merge::XFuse(g) | Merge::DropLit(g) | Merge::Subsume(g) | Merge::Absorb(g) => {
                vec![g.clone()]
            }
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
    // A bare NOT (empty control set, comp = 0, so f = 1) absorbs into a comp=1
    // partner of any width, clearing its comp bit. The generic comp guard below
    // would refuse this along with the genuinely banned direction.
    for (a, b) in [(g, h), (h, g)] {
        if a.ctrls.is_empty() && !a.comp && b.comp && !b.ctrls.is_empty() {
            let mut out = b.clone();
            out.comp = false;
            return Some(Merge::Absorb(out));
        }
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
        let lits = g.ctrls.iter().enumerate().filter(|&(i, _)| i != d).map(|(_, &l)| l);
        return Some(Merge::DropLit(XGate::conj(g.target, lits).expect("drop-lit merge")));
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
    let (w, p) = extra?;
    let lits = big.ctrls.iter().map(|&(cw, cp)| if cw == w { (cw, !cp) } else { (cw, cp) });
    Some(Merge::Subsume(XGate::conj(big.target, lits).expect("subsume merge")))
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
    Some(XGate { target: m(g.target), comp: g.comp, ctrls })
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
    debug_assert!(g.target != b, "cnot twist requires b unwritten in the window");
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
    // Twist window lengths are log-uniform over [twist_min_len, circuit size]:
    // the all-scales dial that decorrelates computational progress at every
    // window scale, the structured analog of ssg's long-range shooting.
    pub twist_min_len: usize,
    // Frozen-DB replacement moves. Both sample a contiguous window of
    // [db_min_window, db_max_window] gates, key it by its exact function, and
    // look it up in the store; they differ in how they pick a replacement.
    //
    // w_db (COMPRESSING channel): probability that a contraction attempt tries
    // the DB channel first — accepting only a non-growing equivalent, chosen
    // uniformly among the SHORTEST — falling through to undo/merge on a miss.
    //
    // p_db (SIZE-AGNOSTIC move): probability that a whole round is spent on a DB
    // replacement instead of the normal contract/expand step; the replacement is
    // a uniform random equivalent of ANY gate count (it may grow the circuit).
    //
    // Both off (0.0) by default: the store is then never opened and the
    // trajectory is identical to the pre-DB chain.
    // p_comp: probability a CONTRACTION tries COMP-DB before undo/merge.
    pub p_comp: f64,
    // p_any: probability an EXPANSION is an ANY-DB move rather than a cross.
    pub p_any: f64,
    pub p_db: f64,
    // s_db: the length the descent STARTS from. The descent itself visits every
    // shorter length down to 1, so there is no separate minimum: one parameter
    // sets the ambition and the descent handles reality.
    pub s_db: usize,
    // p_convex: probability the window sampler is convex rather than contiguous.
    // Replaces the three-valued DbSample: contiguous is 0, convex is 1, and the
    // old `mixed` is 0.5.
    pub p_convex: f64,
    // Window sampling geometry and its guards (see DbSample / db_attempt).
    // db_ctrl_cap (L): while building a window, a gate with more than L controls
    // is evaded (floated out of the way, else the build reverses, else aborts) so
    // high-degree gates that always miss are kept out of the window. 0 = no cap.
    // db_convex_p: for Convex, the probability each growth step floats the block
    // in g1's original direction (else the opposite).
    // db_mode: the slot-2 admission rule. Deterministic, not a coin -- set by
    // slot-0 rules, and the manual size brake (COMP arrests growth while still
    // stamping generations, so it slows the dose rather than stopping it).
    pub db_mode: DbMode,
    // Two eligibility thresholds, not one. w_window governs what may sit INSIDE
    // a window; w_pool governs what may SEED one and count toward the dose.
    // They want different values: width-3 gates match in context often enough
    // to be worth admitting to windows, but their end-to-end per-gate re-encode
    // rate is 0.41% against 98.98% for width <= 2, so at a shared threshold they
    // pile up at the bottom of the pool with nothing to eject them.
    pub w_window: usize,
    pub w_pool: usize,
    pub db_convex_p: f64,
    // Exhaustive per-splice equivalence check on DB replacements. Correctness
    // rests on the key/decode invariants, so this is a safety net; disabling it
    // speeds long runs (the periodic global_check still guards). Windows whose
    // support exceeds 24 wires can only be spliced with verification OFF.
    pub db_verify: bool,
    // Measurement mode: sample windows and record the DB match count via
    // --db-record but NEVER splice (the circuit stays stationary). With
    // p_db = 1.0 and all other weights 0 this makes fmix a pure match-rate
    // sampler over the input circuit.
    pub db_dry_run: bool,
    // Degree pre-filter for DB lookups: a window whose function degree exceeds
    // db_max_degree (the max ANF degree any stored circuit has) cannot match, so
    // it is skipped before the expensive canonicalization. 0 = off. This is the
    // guard that keeps the DB move cheap on the high-width windows the walk
    // produces (which almost always miss). db_degree_probes random subcubes are
    // tested per direction.
    pub db_max_degree: usize,
    pub db_degree_probes: usize,
    // Span pre-filter for DB lookups: a window touching more distinct wires
    // than this is recorded as a miss without canonicalizing. Set to the max
    // canonical support any stored function has (census: frozen_degree_scan).
    // Unlike the degree guard this is not a strict certificate — a window's
    // FUNCTION could depend on fewer wires than the window touches — but exact
    // cancellation is vanishingly rare, and canonicalizing wide-span windows
    // is the dominant cost (Rule-L over large tied wire groups). 0 = off.
    pub db_max_span: usize,
    // Term caps for the DB lookup's polynomial budget: a window whose wire
    // polys (or their sum) outgrow the largest any stored function has cannot
    // match, and the budget Err lands BEFORE the Rule-L canonicalization that
    // dominates dense-window cost. 0 = legacy XPolyBudget defaults (2^18/2^20).
    // Set from the frozen_degree_scan census of the store.
    pub db_wire_terms: usize,
    pub db_total_terms: usize,
    // Key every prefix window[..p] (p in [db_min_window, len]) of each sampled
    // window instead of only the full window: one walk, many lookup shots.
    // Dry-run measurement only for now — the splice policy when several
    // prefixes match (longest vs uniform) is an open design choice.
    pub db_prefixes: bool,
    // db_advance: give DB splice products the same ballistic birth-advance the
    // split moves get (advance floor(dir_q * slack) along the product's own
    // direction). Without it a splice assigns directions that nothing ever acts
    // on: `advance_births` fires at every split site but not here, so under a
    // DB-dominated schedule the directional walk is written and never read.
    // The alternative source of transport -- crossings -- widens material, and
    // width is what kills DB matching, so this is the transport channel that
    // does not fight the store. Off by default: it changes trajectories, so the
    // A/B is one flag.
    pub db_advance: bool,
    // curated: also probe the curated store (FROZEN_CURATED_DIR) and prefer a
    // non-identical curated match over a regular one regardless of size. The
    // curated store holds circuits every strict subcircuit of which is
    // shortest, so a curated replacement is one whose pieces are not locally
    // compressible -- a route fcompress cannot partially undo. Compressing mode
    // ignores it (shrinking is that branch's job).
    pub curated: bool,
    // ancestors: track, per litter, the SET of original input gates that
    // contributed to it -- the union of the sets of the litters the outgoing
    // window drew on. Unlike the single `origin` label (which a mixed-lineage
    // splice destroys, see osyn=), a union never loses information, so it
    // measures how far input material actually travels and what a mixed gate is
    // made of. Cost is |litters| x |input| bits, so this is a small-input
    // instrument: it refuses to arm above `ANC_MAX_INPUT` gates.
    pub ancestors: bool,
    // p_comp_g57: probability that a COMP-DB attempt restricts itself to PURE
    // g57 material and starts its descent at s_db_g57 instead of the usual
    // window. Pure-g57 windows are the only ones that survive length: the
    // measured decay is 100% at m<=5, 94% at 6, then 56/31/20/8/3/0 through 12,
    // whereas ANY non-g57 intruder in a 6-gate window drops it to <=7%. So the
    // long-window compression that actually pays is only available on g57-only
    // windows, and it needs its own coin and its own length.
    // Size brake (hysteresis). Growth to size_hi arms COMP; the mode is
    // released back to db_mode at size_lo OR when COMP stops paying, whichever
    // comes first. The productivity release is what makes a WIDE band safe: the
    // risk was never the width, it was sitting in COMP past its usefulness,
    // where it starves (declines rise as the circuit approaches local
    // minimality) and spends re-encoding diversity (COMP draws only from
    // minimum-size spellings, pulling toward the form fcompress would reach).
    // 0 = brake off.
    // Upper clamp on the contraction probability. The old 0.98 left a 2%
    // expansion floor that is a structural growth source (measured +0.007
    // gates/move); it used to be tightened only under p_db_steer, which is
    // gone, so it is a parameter now rather than a side effect of a retired
    // flag.
    pub contract_ceiling: f64,
    pub size_hi: usize,
    pub size_lo: usize,
    // Release COMP when its shed rate over the trailing window falls below this
    // many gates per round.
    pub comp_release_eps: f64,
    pub comp_release_window: u64,
    pub p_comp_g57: f64,
    pub s_db_g57: usize,
    // Fixed top-level twist rate: with this probability a round performs one
    // conjugation twist directly, decoupling twist supply (set by mixing
    // needs) from the expansion-move economy (whose round supply collapses
    // when the size controller holds at target — measured starvation: 57
    // twists per 700k moves at deep equilibrium). The twist TYPE is drawn
    // from the w_twist_* weights as ratios (neg/swap 50/50 when all are 0),
    // and with p_twist > 0 the expansion mix no longer performs twists (the
    // weights serve as ratios only). The size machinery balances around the
    // fixed twist rate: bracket mass is absorbed by the steered DB throttle
    // and walk contraction like any other growth source.
    pub p_twist: f64,
    // Linear anneal of p_db across the move budget: the effective value runs
    // Generation targeting: drive every (cap-eligible) gate through at least
    // gen_target DB re-encodings. With gen_target > 0, a DB seed is drawn
    // from the laggard list (gates with gen < gen_target) with probability
    // gen_bias instead of uniformly, turning the coupon-collector tail of
    // uniform selection into direct work — fewer moves, hence less incidental
    // growth, for the same minimum generation. 0 = off (trajectory identical
    // to the untargeted chain at equal seed).
    pub gen_target: u32,
    // p_mingen: probability a DB seed is drawn from the generation POOL rather
    // than uniformly -- what stops the process being a coupon collector, where
    // the last few percent of gates soak up most of the moves.
    pub p_mingen: f64,
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
    // litter_ban: refuse a window that is exactly one COMPLETE litter -- the
    // unit some earlier replacement emitted, and therefore precisely where the
    // store can hand the outgoing spelling straight back (A -> B -> A).
    // Singleton litters are exempt by construction: an input gate has no
    // earlier spelling to be returned to, and banning it would also refuse the
    // descent's length-1 rung, the one rung that always makes progress.
    pub litter_ban: bool,
    // litter_samples: draw this many candidate windows and keep the one
    // spanning the MOST distinct litters. 1 = off. Discarded candidates may
    // leave ctrl-cap evasion floats behind; those are function-preserving and
    // the walk floats constantly, so the cost is arena churn, not correctness.
    pub litter_samples: usize,
    // twist_place_tries: how many candidate positions the twist placer samples
    // looking for a TWIST_PATTERNS match before giving up and placing the twist
    // uniformly at random. 0 = always random, which is the historical
    // behaviour.
    pub twist_place_tries: usize,
    // Pool rebuild cadence in moves (an O(size) scan each time).
    pub gen_rescan: u64,
    // Split-rule variant for the generation benchmark: false (default) =
    // ratchet semantics, split children get parent + 1; true = children
    // inherit the parent generation unchanged, so ONLY DB replacements raise
    // generations (isolates DB re-encoding depth from walk rewrite depth).
    pub gen_split_inherit: bool,
    // Median variant for the DB stamp: false (default) = upper median
    // (sorted[len/2], median rounded up on even windows); true = lower
    // median (sorted[(len-1)/2], rounded down). On 2-gate windows — the most
    // common splice — the lower median IS the min, so this probes how close
    // the median rule sits to min-semantics' straggler-bound climb.
    pub gen_median_low: bool,
    // Dose-based stop: with gen_target > 0 and gen_stop_frac >= 0, the run
    // ends (MixStop::DoseReached) at the first report point where the
    // laggard fraction among the TARGETABLE gates (cap-eligible, not written
    // off — i.e. lag/targetable, the same population behind g_circ) is
    // <= gen_stop_frac AND the cumulative per-position twist coverage
    // (twist_span / size) has reached twist_cov_stop (0 = no coverage
    // requirement). The move budget becomes a ceiling: phase A runs exactly
    // as long as the dose requires.
    pub gen_stop_frac: f64,
    pub twist_cov_stop: f64,
    // Generation-multiple snapshots: at each report point, when the circuit
    // generation (g_circ, the 5th-percentile gate generation) crosses a fresh
    // multiple of this interval, write a verified snapshot to
    // <base>.gen<m>.mpmct1 (+ .gens sidecar); the base path is armed with
    // Mixer::set_gen_snap_base. 0 = off.
    pub gen_snap_every: u32,
    // Move-multiple snapshots: at each report point where moves_done is a
    // multiple of this interval, write a verified snapshot to
    // <base>.mv<moves>.mpmct1 (+ .gens sidecar). The progress clock for
    // regimes where the generation census is not meaningful (e.g. pure-split
    // phase B, p_db = 0). Choose a multiple of report_every. 0 = off.
    pub snap_every_moves: u64,
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
            temp: 0.0,      // 0 -> max(target/100, 64), resolved by Mixer::new
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
            twist_min_len: 64,
            // Store-free by default: MixParams::default() is the test/base
            // value, and any positive DB rate here would make every construction
            // demand FROZEN_DB_DIR. The SPEC defaults (p_comp 1.0, p_any 0.1)
            // live on the CLI, where a run that wants the store asks for it.
            p_comp: 0.0,
            p_any: 0.0,
            p_db: 0.0,
            s_db: 5,
            p_convex: 0.5,
            db_mode: DbMode::Mix,
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
            db_advance: false,
            curated: false,
            ancestors: false,
            // 0.98 is the historical value: its 2% expansion floor above
            // target is a structural growth source, but it is ALSO what keeps
            // crossings running when the walk sits at target -- and crossings
            // are what erode fossils. Tightening it to 0.9995 cuts expansion
            // 40x, so that belongs in a recipe that wants it, not here.
            contract_ceiling: 0.98,
            size_hi: 0,
            size_lo: 0,
            comp_release_eps: 0.0,
            comp_release_window: 250_000,
            p_comp_g57: 0.0,
            s_db_g57: 9,
            p_twist: 0.0,
            gen_target: 0,
            p_mingen: 0.8,
            pool_k: 20_000,
            canary_theta: 0.0,
            canary_window: 2000,
            litter_ban: false,
            litter_samples: 1,
            twist_place_tries: 0,
            gen_rescan: 10_000,
            gen_split_inherit: false,
            gen_median_low: false,
            gen_stop_frac: -1.0,
            twist_cov_stop: 0.0,
            gen_snap_every: 0,
            snap_every_moves: 0,
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
    // Ingest-then-pay rounds: cheap (Compressing on cheap-tier laggards) and
    // paid (MinGrow on hard-tier gates); hits = spliced. db_hard_added is the
    // growth ledger — gates the paid channel added net; gen_misses = total
    // seed-miss bumps.
    pub db_ing_hits: u64,
    pub db_ing_rounds: u64,
    pub db_hard_hits: u64,
    pub db_hard_rounds: u64,
    pub db_hard_added: u64,
    // Candidates refused by the identity guard, and splices whose replacement
    // came from the curated store.
    pub db_identity_skips: u64,
    pub db_curated_hits: u64,
    // Curated replacements refused because they did not verify (see
    // try_db_splice_curated).
    pub db_curated_rejected: u64,
    // NOT-into-comp=1 absorptions (Merge::Absorb): the channel that lets a
    // twist bracket be swallowed by a neighbouring g57 rather than paid for.
    pub merges_absorb: u64,
    // g57-only COMP attempts and their hits.
    pub db_g57_rounds: u64,
    pub db_g57_hits: u64,
    // Slot-2 rounds, their hits, and the growth they added.
    pub db_slot2_rounds: u64,
    pub db_slot2_hits: u64,
    pub db_slot2_added: u64,
    // Times the brake engaged, and rounds spent under it.
    pub brake_engagements: u64,
    pub brake_rounds: u64,
    // Heads coins that found the pool drained (see db_attempt).
    pub canary_fallthrough: u64,
    // Descent rungs refused by the full-litter ban.
    pub litter_banned: u64,
    // Distance-from-minimal: sampled windows for which the store held ANY
    // non-identical equivalent, and how many of those admitted a STRICTLY
    // SHORTER one. The ratio is a live fcompress residual without running
    // fcompress -- as COMP drives the circuit toward its locally-minimal form
    // (the form the attacker-computable compressor would reach anyway) this
    // falls toward zero, which is the spelling diversity being spent.
    pub dmin_windows: u64,
    pub dmin_shorter: u64,
    // Twists placed on a pattern match, and twists that fell back to a random
    // position because no candidate matched within the try budget.
    pub twist_placed: u64,
    pub twist_place_fallback: u64,
    // Joint size distribution of successful splices: [outgoing len][incoming
    // len]. The shape of what the store actually trades, which the scalar rm=
    // and add= totals cannot show -- a channel that swaps 3 gates for 3 and one
    // that alternates 2->5 and 5->2 report identically.
    pub splice_sizes: Vec<Vec<u64>>,
    // Litter census (observation only — nothing bans or prefers on these yet;
    // see docs/FMIX_MENU.md 2.6). `litter_windows`/`litter_distinct_sum` give
    // the mean distinct litters per sampled DB window, i.e. how fast churn
    // fragments litters. `litter_full_spliced` counts splices whose outgoing
    // window was exactly one COMPLETE litter — precisely the replacements an
    // ssg-style full-litter ban would have refused, and therefore the number
    // that says whether the ban is worth wiring in here.
    pub litter_windows: u64,
    pub litter_distinct_sum: u64,
    pub litter_full_spliced: u64,
    pub gen_misses: u64,
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
    // DB replacement moves (compressing channel + size-agnostic move).
    pub db_comp_hits: u64,     // compressing: window replaced by a non-growing friend
    pub db_comp_misses: u64,   // compressing: sampled window, no non-growing friend
    pub db_agn_hits: u64,      // size-agnostic: window replaced (any length)
    pub db_agn_misses: u64,    // size-agnostic: sampled window, no equivalent found
    pub db_gates_removed: u64, // gates removed by accepted DB replacements
    pub db_gates_added: u64,   // gates added by accepted (growing) DB replacements
    pub db_wide_skip: u64,     // support > 24 wires with db_verify on: not verifiable, skipped
    pub db_attempts: u64,      // total DB lookups attempted (both modes)
    pub db_degree_skips: u64,  // attempts skipped by the degree guard (no lookup)
    pub db_span_skips: u64,    // attempts skipped by the span guard (no lookup)
    pub db_build_aborts: u64,  // window builds aborted by the evade budget
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
        self.cross_r1 + self.cross_r2 + self.cross_r3 + self.presplits + self.fresh_splits
            + self.unsubs
            + self.inserts
            + self.twist_negs
            + self.twist_swaps
            + self.twist_cnots
    }
}

#[derive(Clone, Copy)]
struct Meta {
    origin: u32,
    event: u64, // 0 = not a split product
    // Persistent shooting direction: a cross floats this gate in `dir`.
    // Fossils draw it uniformly at birth; fragments inherit per dir_p.
    dir: Dir,
    // Rewrite generation (benchmark semantics, 2026-07-21): how many
    // re-encodings this gate's material has been through since the input.
    // Input gates start at 0; a DB splice stamps its products with the
    // outgoing window's upper-median generation + 1; every SPLIT (presplit,
    // cross piece, fresh-split, unsubsume, twist case-split) stamps children
    // with parent + 1; merges take the min of their parents. Born-random
    // material (insert pairs, twist bracket packets) carries no input
    // structure and gets GEN_FRESH (= the spec's MAXGEN, higher than every
    // real generation; saturating arithmetic keeps it fixed).
    dgen: u32,
    // Litter tag (ssg `80a2c1d2` semantics, ported for measurement): the
    // replacement event that CREATED this gate. Input gates and born-random
    // material are singleton litters. A DB splice stamps all of its products
    // with one fresh id; splits and merges PROPAGATE the parent's id, so a
    // litter fragments under churn rather than being reassigned.
    //
    // `litter_size` is the size at creation and is deliberately NOT maintained
    // under splits: a window is "a complete litter" only when every gate shares
    // one id AND the count still equals that recorded size, so churn makes the
    // test conservative (it misses, never over-fires) exactly as in ssg.
    //
    // Observation only today — nothing bans or prefers on these yet; see
    // docs/FMIX_MENU.md §2.6.
    litter: u64,
    litter_size: u16,
}


/// Generation stamp for born-random material (fresh insert pairs, twist
/// bracket packets): these gates never held any input structure, so for the
/// gen-target machinery they count as already re-encoded.
pub const GEN_FRESH: u32 = u32::MAX;

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
    gens: [u32; 2],    // gen of before[0], before[1] (restored on undo)
    // Litter of before[0], before[1], restored on undo so that reversing a
    // crossing also reverses its provenance rather than minting new litters.
    litters: [u64; 2],
    litter_sizes: [u16; 2],
    misses: u8, // failed gather attempts; entry dropped after a few
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
    // Next litter id. Input gates take 0..n as singleton litters, so fresh ids
    // start at n and never collide with them.
    next_litter: u64,
    // litter id -> bitset over input-gate indices (see MixParams::ancestors).
    anc: HashMap<u64, Vec<u64>>,
    anc_words: usize,
    anc_m: usize,
    original: Vec<XGate>,
    num_wires: usize,
    pub moves_done: u64,
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
    // Generation-multiple snapshots (params.gen_snap_every): output base path
    // and the highest generation multiple already written.
    gen_snap_base: Option<String>,
    last_gen_snap: u32,
    stop_requested: bool,
    // Frozen replacement store for the DB moves. Opened from the environment
    // only when a DB move is enabled; an empty (miss-everything) store otherwise
    // so runs without them never require FROZEN_DB_DIR.
    db: FrozenDb,
    db_budget: XPolyBudget,
    // Optional per-attempt recorder (--db-record): one block per DB attempt with
    // the outgoing window, the number of equivalent DB circuits, and (on
    // success) the replacing subcircuit.
    db_record: Option<std::io::BufWriter<std::fs::File>>,
    // Which geometry built the window of the CURRENT db_attempt (see
    // sample_window); stamped into --db-record attempt lines.
    db_last_sampler: DbSample,
    // (seed id, its left neighbour when drawn). Window building floats gates,
    // so a FAILED attempt would otherwise leave the seed displaced -- and under
    // pool targeting the same stubborn gate is drawn repeatedly, which would
    // write a characteristic displacement into the circuit.
    db_seed_home: Option<(u32, u32)>,
    // Set for the duration of one COMP attempt drawn as g57-only.
    db_g57_only: bool,
    // The LIVE slot-2 mode. params.db_mode is what the brake returns to.
    db_mode_cur: DbMode,
    brake_on: bool,
    brake_mark_move: u64,
    brake_mark_size: usize,
    // Next move at which the pool is rebuilt.
    pool_scan_due: u64,
    // The generation pool (see MixParams::pool_k).
    pool: Vec<u32>,
    // Whether the current attempt's seed genuinely came from the pool, and
    // whether a heads coin fell through because the pool had drained.
    seed_from_pool: bool,
    seed_fell_through: bool,
    // Trailing window of qualifying-round outcomes (true = failed at every
    // rung), with a running failure count so the fraction is O(1).
    canary: VecDeque<bool>,
    canary_failures: usize,
}

pub enum MixStop {
    MovesBudget,
    StopFlag,
    // Generation + twist-coverage dose targets met (gen_stop_frac): the run
    // ended as soon as the mixing dose was achieved, spending no further
    // moves (and hence no further incidental growth) past the requirement.
    DoseReached,
    // The canary fired: over the trailing window of qualifying rounds the
    // failure fraction exceeded canary_theta, i.e. the pool has converged on
    // material the store cannot spell and further moves buy nothing.
    CanaryFired,
}

impl MixCounters {
    /// Whitespace-separated dump of every counter, in declaration order. These
    /// are trajectory statistics rather than chain state, but a resumed run
    /// whose report restarted from zero would make its own history unreadable,
    /// and two of them -- twist_span and canary_fallthrough -- feed conditions.
    pub fn to_line(&self) -> String {
        let vals: Vec<u64> = vec![
            self.moves,
            self.db_ing_hits,
            self.db_ing_rounds,
            self.db_hard_hits,
            self.db_hard_rounds,
            self.db_hard_added,
            self.db_identity_skips,
            self.db_curated_hits,
            self.merges_absorb,
            self.db_g57_rounds,
            self.db_g57_hits,
            self.db_slot2_rounds,
            self.db_slot2_hits,
            self.db_slot2_added,
            self.brake_engagements,
            self.brake_rounds,
            self.canary_fallthrough,
            self.litter_banned,
            self.twist_placed,
            self.db_curated_rejected,
            self.twist_place_fallback,
            self.dmin_windows,
            self.dmin_shorter,
            self.litter_windows,
            self.litter_distinct_sum,
            self.litter_full_spliced,
            self.gen_misses,
            self.merges_cancel,
            self.merges_xfuse,
            self.merges_drop,
            self.merges_subsume,
            self.merges_sibling,
            self.merges_cross_origin,
            self.tabu_blocked,
            self.merge_no_partner,
            self.merge_wall_blocked,
            self.merge_too_far,
            self.merge_not_adjacent,
            self.undos,
            self.undo_dead,
            self.undo_tabu,
            self.undo_gather_miss,
            self.db_comp_hits,
            self.db_comp_misses,
            self.db_agn_hits,
            self.db_agn_misses,
            self.db_gates_removed,
            self.db_gates_added,
            self.db_wide_skip,
            self.db_attempts,
            self.db_degree_skips,
            self.db_span_skips,
            self.db_build_aborts,
            self.cross_r1,
            self.cross_r2,
            self.cross_r3,
            self.presplits,
            self.fresh_splits,
            self.unsubs,
            self.inserts,
            self.twist_negs,
            self.twist_swaps,
            self.twist_cnots,
            self.twist_relabels,
            self.twist_case_splits,
            self.twist_span,
            self.twist_skips,
            self.blocked_width,
            self.blocked_deadlock,
            self.declined,
            self.boundary,
            self.floats,
            self.float_steps,
            self.scatters,
            self.scatter_steps,
            self.dropped_neverfire
        ];
        vals.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" ")
    }

    /// Inverse of `to_line`. The two histograms (`splice_sizes`, `width_hist`)
    /// are deliberately not carried: they describe SHAPES rather than totals,
    /// and a resumed run should report the shapes it produced itself.
    pub fn from_line(s: &str) -> Option<MixCounters> {
        let mut it = s.split_whitespace();
        fn next_u64<'a>(it: &mut impl Iterator<Item = &'a str>) -> Option<u64> {
            it.next()?.parse().ok()
        }
        Some(MixCounters {
            moves: next_u64(&mut it)?,
            db_ing_hits: next_u64(&mut it)?,
            db_ing_rounds: next_u64(&mut it)?,
            db_hard_hits: next_u64(&mut it)?,
            db_hard_rounds: next_u64(&mut it)?,
            db_hard_added: next_u64(&mut it)?,
            db_identity_skips: next_u64(&mut it)?,
            db_curated_hits: next_u64(&mut it)?,
            merges_absorb: next_u64(&mut it)?,
            db_g57_rounds: next_u64(&mut it)?,
            db_g57_hits: next_u64(&mut it)?,
            db_slot2_rounds: next_u64(&mut it)?,
            db_slot2_hits: next_u64(&mut it)?,
            db_slot2_added: next_u64(&mut it)?,
            brake_engagements: next_u64(&mut it)?,
            brake_rounds: next_u64(&mut it)?,
            canary_fallthrough: next_u64(&mut it)?,
            litter_banned: next_u64(&mut it)?,
            twist_placed: next_u64(&mut it)?,
            db_curated_rejected: next_u64(&mut it)?,
            twist_place_fallback: next_u64(&mut it)?,
            dmin_windows: next_u64(&mut it)?,
            dmin_shorter: next_u64(&mut it)?,
            litter_windows: next_u64(&mut it)?,
            litter_distinct_sum: next_u64(&mut it)?,
            litter_full_spliced: next_u64(&mut it)?,
            gen_misses: next_u64(&mut it)?,
            merges_cancel: next_u64(&mut it)?,
            merges_xfuse: next_u64(&mut it)?,
            merges_drop: next_u64(&mut it)?,
            merges_subsume: next_u64(&mut it)?,
            merges_sibling: next_u64(&mut it)?,
            merges_cross_origin: next_u64(&mut it)?,
            tabu_blocked: next_u64(&mut it)?,
            merge_no_partner: next_u64(&mut it)?,
            merge_wall_blocked: next_u64(&mut it)?,
            merge_too_far: next_u64(&mut it)?,
            merge_not_adjacent: next_u64(&mut it)?,
            undos: next_u64(&mut it)?,
            undo_dead: next_u64(&mut it)?,
            undo_tabu: next_u64(&mut it)?,
            undo_gather_miss: next_u64(&mut it)?,
            db_comp_hits: next_u64(&mut it)?,
            db_comp_misses: next_u64(&mut it)?,
            db_agn_hits: next_u64(&mut it)?,
            db_agn_misses: next_u64(&mut it)?,
            db_gates_removed: next_u64(&mut it)?,
            db_gates_added: next_u64(&mut it)?,
            db_wide_skip: next_u64(&mut it)?,
            db_attempts: next_u64(&mut it)?,
            db_degree_skips: next_u64(&mut it)?,
            db_span_skips: next_u64(&mut it)?,
            db_build_aborts: next_u64(&mut it)?,
            cross_r1: next_u64(&mut it)?,
            cross_r2: next_u64(&mut it)?,
            cross_r3: next_u64(&mut it)?,
            presplits: next_u64(&mut it)?,
            fresh_splits: next_u64(&mut it)?,
            unsubs: next_u64(&mut it)?,
            inserts: next_u64(&mut it)?,
            twist_negs: next_u64(&mut it)?,
            twist_swaps: next_u64(&mut it)?,
            twist_cnots: next_u64(&mut it)?,
            twist_relabels: next_u64(&mut it)?,
            twist_case_splits: next_u64(&mut it)?,
            twist_span: next_u64(&mut it)?,
            twist_skips: next_u64(&mut it)?,
            blocked_width: next_u64(&mut it)?,
            blocked_deadlock: next_u64(&mut it)?,
            declined: next_u64(&mut it)?,
            boundary: next_u64(&mut it)?,
            floats: next_u64(&mut it)?,
            float_steps: next_u64(&mut it)?,
            scatters: next_u64(&mut it)?,
            scatter_steps: next_u64(&mut it)?,
            dropped_neverfire: next_u64(&mut it)?,
            splice_sizes: Vec::new(),
            width_hist: [0u64; 16],
        })
    }
}

/// Version stamp for the resume file. The parameter set has changed repeatedly,
/// and silently reinterpreting fields would be worse than refusing to load.
pub const STATE_VERSION: u32 = 1;

impl Mixer {
    /// Write everything a resumed run needs and the circuit file does not
    /// carry. A run is hours long and every measurement depends on it, so a
    /// stop -- flag, canary, dose or budget -- should be a PAUSE, not a loss.
    ///
    /// Three things make this more than a gate dump:
    ///
    /// - Per-gate `dir` is load-bearing and has no sidecar. Directions are
    ///   drawn at load and the whole directional walk rides on them, so a
    ///   resume that redrew them would restart transport rather than continue
    ///   it. Same for `dgen`, `litter` and `event`.
    /// - The undo journal references arena IDs, so the checkpoint RENUMBERS to
    ///   0..n-1 in arena order -- exactly what `Arena::from_gates` reproduces --
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
        let mut o = String::with_capacity(ids.len() * 48);
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
        let _ = writeln!(o, "anc {} {}", self.anc_words, self.anc_m);
        let _ = writeln!(o, "counters {}", self.counters.to_line());

        let _ = writeln!(o, "gates {}", ids.len());
        for &id in &ids {
            let m = self.meta_of(id);
            let g = self.arena.gate(id).clone();
            gate_line(&mut o, &g);
            let _ = writeln!(
                o,
                " | {} {} {} {} {} {}",
                m.origin,
                m.event,
                (m.dir == Dir::R) as u8,
                m.dgen,
                m.litter,
                m.litter_size
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
        let pool: Vec<u32> = self.pool.iter().filter_map(|id| newid.get(id).copied()).collect();
        let _ = writeln!(o, "pool {}", pool.len());
        for id in &pool {
            let _ = writeln!(o, "{id}");
        }
        let _ = writeln!(o, "canary {}", self.canary.len());
        for b in self.canary.iter() {
            let _ = writeln!(o, "{}", *b as u8);
        }
        // Journal: keep only entries every piece of which is still live, then
        // remap. Stamps are NOT stored -- the rebuilt arena assigns its own, and
        // resume reads them back from it.
        let keep: Vec<&UndoEntry> = self
            .journal
            .iter()
            .filter(|e| e.after.iter().all(|&(id, st)| {
                self.arena.is_linked(id) && self.arena.stamp(id) == st && newid.contains_key(&id)
            }))
            .collect();
        let _ = writeln!(o, "journal {}", keep.len());
        for e in keep {
            gate_line(&mut o, &e.before[0]);
            o.push(' ');
            gate_line(&mut o, &e.before[1]);
            let _ = write!(
                o,
                " | {} {} {} {} {} {} {} {} {} {} {}",
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
                e.litter_sizes[1]
            );
            let _ = write!(o, " | {}", e.after.len());
            for &(id, _) in &e.after {
                let _ = write!(o, " {}", newid[&id]);
            }
            let _ = writeln!(o, " {}", e.misses);
        }
        let _ = writeln!(o, "ancsets {}", self.anc.len());
        for (l, bits) in self.anc.iter() {
            let _ = write!(o, "{l}");
            for w in bits {
                let _ = write!(o, " {w}");
            }
            o.push('\n');
        }
        std::fs::write(path, o)
    }

    /// Rebuild a mixer from a state file. `params` come from the CLI as usual:
    /// a resume is free to change rates, targets, the brake or the mode, which
    /// is the point -- a paused run should be steerable. What it must NOT
    /// change is the version, since the field meanings would drift silently.
    pub fn resume_state(path: &str, params: MixParams, db: FrozenDb) -> std::io::Result<Mixer> {
        let text = std::fs::read_to_string(path)?;
        let mut lines = text.lines();
        let bad = |m: &str| std::io::Error::other(m.to_string());
        let mut hdr = lines.next().ok_or_else(|| bad("empty state file"))?.split_whitespace();
        if hdr.next() != Some("fmix-state") {
            return Err(bad("missing fmix-state header"));
        }
        let v: u32 = hdr.next().and_then(|x| x.parse().ok()).ok_or_else(|| bad("bad version"))?;
        if v != STATE_VERSION {
            return Err(bad(&format!(
                "state file version {v} != {STATE_VERSION}; refusing to reinterpret its fields"
            )));
        }
        // Scalars, in the order save_state writes them.
        fn scalar(
            lines: &mut std::str::Lines<'_>,
            want: &str,
        ) -> std::io::Result<Vec<String>> {
            let l = lines
                .next()
                .ok_or_else(|| std::io::Error::other(format!("missing {want}")))?;
            let mut it = l.split_whitespace();
            if it.next() != Some(want) {
                return Err(std::io::Error::other(format!("expected {want} in state file")));
            }
            Ok(it.map(|x| x.to_string()).collect())
        }
        let num_wires: usize = scalar(&mut lines, "wires")?[0].parse().map_err(|_| bad("wires"))?;
        let moves_done: u64 = scalar(&mut lines, "moves")?[0].parse().map_err(|_| bad("moves"))?;
        let next_event: u64 = scalar(&mut lines, "next_event")?[0].parse().map_err(|_| bad("next_event"))?;
        let next_litter: u64 = scalar(&mut lines, "next_litter")?[0].parse().map_err(|_| bad("next_litter"))?;
        let rng_seed: u64 = scalar(&mut lines, "rng")?[0].parse().map_err(|_| bad("rng"))?;
        let mrng_seed: u64 = scalar(&mut lines, "metrics_rng")?[0].parse().map_err(|_| bad("metrics_rng"))?;
        let mode_s = scalar(&mut lines, "db_mode")?[0].clone();
        let brake = scalar(&mut lines, "brake")?;
        let pool_scan_due: u64 = scalar(&mut lines, "pool_scan_due")?[0].parse().map_err(|_| bad("scan"))?;
        let canary_failures: usize = scalar(&mut lines, "canary_failures")?[0].parse().map_err(|_| bad("cf"))?;
        let anc_hdr = scalar(&mut lines, "anc")?;
        let counters_line = scalar(&mut lines, "counters")?.join(" ");

        // Sections. Gates carry their meta on the same line after a `|`.
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
            Some(XGate { target, comp: comp != 0, ctrls })
        };

        let ng = section(&mut lines, "gates")?;
        let mut gates = Vec::with_capacity(ng);
        let mut metas = Vec::with_capacity(ng);
        for _ in 0..ng {
            let l = lines.next().ok_or_else(|| bad("short gates section"))?;
            let (gp, mp) = l.split_once(" | ").ok_or_else(|| bad("gate line missing meta"))?;
            let g = parse_gate(&mut gp.split_whitespace()).ok_or_else(|| bad("bad gate"))?;
            let m: Vec<&str> = mp.split_whitespace().collect();
            if m.len() != 6 {
                return Err(bad("gate meta must have 6 fields"));
            }
            let p = |i: usize| -> std::io::Result<u64> {
                m[i].parse().map_err(|_| bad("bad meta field"))
            };
            metas.push(Meta {
                origin: p(0)? as u32,
                event: p(1)?,
                dir: if p(2)? != 0 { Dir::R } else { Dir::L },
                dgen: p(3)? as u32,
                litter: p(4)?,
                litter_size: p(5)? as u16,
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
            let e: u64 = it.next().and_then(|x| x.parse().ok()).ok_or_else(|| bad("tabu"))?;
            let mv: u64 = it.next().and_then(|x| x.parse().ok()).ok_or_else(|| bad("tabu"))?;
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
            if f.len() != 11 {
                return Err(bad("journal meta shape"));
            }
            let mut ai = parts[2].split_whitespace();
            let n: usize = ai.next().and_then(|x| x.parse().ok()).ok_or_else(|| bad("after n"))?;
            let after: Vec<u32> =
                (0..n).filter_map(|_| ai.next().and_then(|x| x.parse().ok())).collect();
            let misses: u8 = ai.next().and_then(|x| x.parse().ok()).unwrap_or(0);
            journal_raw.push((b0, b1, f, after, misses));
        }
        let na = section(&mut lines, "ancsets")?;
        let mut anc: HashMap<u64, Vec<u64>> = HashMap::with_capacity(na);
        for _ in 0..na {
            let l = lines.next().ok_or_else(|| bad("short ancsets"))?;
            let mut it = l.split_whitespace();
            let key: u64 = it.next().and_then(|x| x.parse().ok()).ok_or_else(|| bad("anc key"))?;
            anc.insert(key, it.filter_map(|x| x.parse().ok()).collect());
        }

        // Build the mixer on the SAME id assignment the checkpoint renumbered
        // to: Arena::from_gates hands out 0..n-1 in order, which is what
        // save_state mapped the journal and pool onto.
        // Construct with ancestors OFF: the sizing guard in new_with_db reads
        // the CURRENT gate count, but ancestor sets are indexed by ORIGINAL
        // input gates and their width comes from the state file. A resumed 1.4M
        // gate circuit whose input was 20k would trip a guard meant for the
        // input. Restore the real setting and the recorded widths below.
        let ancestors_wanted = params.ancestors;
        let mut mx = Mixer::new_with_db(gates, num_wires, MixParams { ancestors: false, ..params }, db);
        mx.params.ancestors = ancestors_wanted;
        mx.original = original;
        mx.moves_done = moves_done;
        mx.counters = MixCounters::from_line(&counters_line).ok_or_else(|| bad("counters"))?;
        mx.counters.moves = moves_done;
        mx.next_event = next_event;
        mx.next_litter = next_litter;
        mx.rng = StdRng::seed_from_u64(rng_seed);
        mx.metrics_rng = StdRng::seed_from_u64(mrng_seed);
        // The LIVE mode comes from the command line, not the file. A resume is
        // meant to be re-steerable -- changing --db-mode is the whole point of a
        // manual breathing cycle -- and letting the saved value win silently
        // ignores the flag: a resume asked for COMP would keep growing in MIX
        // and look like COMP was broken. The saved mode is kept for diagnostics
        // only. If the brake was engaged it re-engages on the next round
        // anyway, since apply_size_brake runs before slot 1.
        let _saved_mode = DbMode::parse(&mode_s);
        mx.db_mode_cur = mx.params.db_mode;
        mx.brake_on = brake[0] != "0";
        mx.brake_mark_move = brake[1].parse().unwrap_or(0);
        mx.brake_mark_size = brake[2].parse().unwrap_or(0);
        mx.pool_scan_due = pool_scan_due;
        mx.canary = canary;
        mx.canary_failures = canary_failures;
        mx.pool = pool;
        mx.anc = anc;
        mx.anc_words = anc_hdr[0].parse().unwrap_or(0);
        mx.anc_m = anc_hdr[1].parse().unwrap_or(0);
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
                misses,
            })
            .collect();
        mx.tabu = tabu;
        Ok(mx)
    }
}

/// Generation census over the linked circuit (see `Meta::dgen`).
pub struct GenStats {
    /// Cap-eligible gates still below gen_target and still targetable
    /// (cheap + hard tiers) — the dose-stop numerator.
    pub lag: u64,
    /// Cap-eligible gates in total.
    pub elig: u64,
    /// Wide (cap-ineligible) gates below gen_target: invisible to the DB
    /// channel until some other move narrows or splits them (their pieces
    /// get parent + 1, so the walk lifts them).
    pub wlag: u64,
    /// Minimum gen over ALL linked gates (GEN_FRESH when nothing lags).
    pub min: u32,
    /// ALL gates below gen_target (eligible + wide + written-off) and the
    /// total.
    pub all_lag: u64,
    pub total: u64,
    /// The circuit generation: the largest G such that at least 95% of the
    /// TARGETABLE gates have generation >= G (the 5th-percentile generation
    /// over the population the DB channel can actually move — cap-eligible
    /// and not written off).
    ///
    /// It is deliberately NOT the percentile over all gates. Generations only
    /// advance under DB re-encoding, so a gate the DB can never touch sits at
    /// generation 0 forever; if such gates exceed 5% of the circuit they pin
    /// the all-gates percentile at 0 permanently, no matter how much mixing
    /// runs. That is a structural property of the material, not a measure of
    /// progress. On a product-share gadget ~62% of gates are wide, so the
    /// all-gates figure is identically 0 and says nothing.
    ///
    /// When every gate is targetable (the common case: no ctrl cap and no
    /// write-offs) this is exactly the old all-gates percentile, so runs on
    /// uniformly narrow material are unaffected.
    pub g_circ: u32,
    /// The old all-gates percentile, kept for continuity with earlier runs
    /// and reported as Gall=. Equals g_circ when everything is targetable.
    pub g_all: u32,
    /// Gates the DB channel can actually move: cap-eligible, minus those
    /// written off as unreachable. The denominator behind g_circ.
    pub targetable: u64,
}

impl Mixer {
    pub fn new(gates: Vec<XGate>, num_wires: usize, params: MixParams) -> Mixer {
        // Open the replacement store once, only when a DB move is enabled, so
        // runs without them never require FROZEN_DB_DIR.
        // Every channel that can reach the store must be in this test. Leaving
        // one out yields FrozenDb::empty(), so every lookup misses, the run does
        // zero re-encoding, and nothing says so -- it looks like a measurement.
        // The ingest/paid channels were once missing here, which yielded an
        // empty store and a run that reported zero re-encoding as if it were a
        // result; they are gone now, but the invariant remains.
        let db = if params.p_comp > 0.0
            || params.p_db > 0.0
        {
            FrozenDb::from_env()
        } else {
            FrozenDb::empty()
        };
        Mixer::new_with_db(gates, num_wires, params, db)
    }

    /// As [`Mixer::new`] but with an explicit store (tests point this at a
    /// prebuilt frozen directory instead of `FROZEN_DB_DIR`).
    pub fn new_with_db(
        gates: Vec<XGate>,
        num_wires: usize,
        mut params: MixParams,
        db: FrozenDb,
    ) -> Mixer {
        let n = gates.len();
        if params.target_size == 0 {
            params.target_size = n;
        }
        if params.temp <= 0.0 {
            params.temp = (params.target_size as f64 / 100.0).max(64.0);
        }
        let num_wires = num_wires.max(super::xgate::max_wire(&gates) as usize + 1);
        let mut rng = StdRng::seed_from_u64(params.seed);
        let metrics_rng = StdRng::seed_from_u64(params.seed ^ 0x5EED_517A75);
        let meta = (0..n)
            .map(|i| Meta {
                origin: i as u32,
                event: 0,
                dir: if rng.random_bool(0.5) { Dir::L } else { Dir::R },
                dgen: 0,
                // Input gates are singleton litters, as in ssg: they were not
                // emitted by any replacement, so there is no prior spelling a
                // full-litter rule could send them back to.
                litter: i as u64,
                litter_size: 1,
            })
            .collect();
        let db_mode0 = params.db_mode;
        let (anc_words0, anc_m0) = if params.ancestors {
            assert!(
                n <= 20_000,
                "--ancestors stores |input| bits per litter; {n} input gates is past                  the small-input envelope this instrument is for"
            );
            (n.div_ceil(64), n)
        } else {
            (0, 0)
        };
        let mut index: HashMap<u64, Vec<u32>> = HashMap::new();
        for (i, g) in gates.iter().enumerate() {
            index.entry(key_of(g)).or_default().push(i as u32);
        }
        let db_budget = {
            let mut b = XPolyBudget::default();
            if params.db_wire_terms > 0 {
                b.max_poly_terms = params.db_wire_terms;
            }
            if params.db_total_terms > 0 {
                b.max_total_terms = params.db_total_terms;
            }
            b
        };
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
            next_litter: n as u64,
            anc: HashMap::new(),
            anc_words: anc_words0,
            anc_m: anc_m0,
            original: gates,
            num_wires,
            moves_done: 0,
            rng,
            metrics_rng,
            stop_flag: None,
            dump_flag: None,
            dump_out: String::new(),
            gen_snap_base: None,
            last_gen_snap: 0,
            stop_requested: false,
            db,
            db_budget,
            db_record: None,
            db_last_sampler: DbSample::Contiguous,
            db_seed_home: None,
            db_g57_only: false,
            db_mode_cur: db_mode0,
            brake_on: false,
            brake_mark_move: 0,
            brake_mark_size: 0,
            pool: Vec::new(),
            seed_from_pool: false,
            seed_fell_through: false,
            canary: VecDeque::new(),
            canary_failures: 0,
            pool_scan_due: 0,
        }
    }

    /// Enable per-DB-attempt recording to `path` (see [`MixCounters`] db_*).
    pub fn enable_db_record(&mut self, path: &str) {
        match std::fs::File::create(path) {
            Ok(f) => self.db_record = Some(std::io::BufWriter::new(f)),
            Err(e) => eprintln!("[fmix] could not open --db-record {path}: {e}"),
        }
    }

    pub fn enable_flags(&mut self, stop: Option<String>, dump: Option<String>, dump_out: String) {
        self.stop_flag = stop;
        self.dump_flag = dump;
        self.dump_out = dump_out;
    }

    /// Arm generation-multiple snapshots (params.gen_snap_every): files go to
    /// `<base>.gen<m>.mpmct1` (+ `.gens` sidecar).
    pub fn set_gen_snap_base(&mut self, base: String) {
        self.gen_snap_base = Some(base);
    }

    fn set_meta(&mut self, id: u32, m: Meta) {
        let i = id as usize;
        if i >= self.meta.len() {
            self.meta
                .resize(i + 1, Meta { origin: ORIGIN_SYNTH, event: 0, dir: Dir::R, dgen: GEN_FRESH, litter: 0, litter_size: 1 });
        }
        self.meta[i] = m;
    }

    fn meta_of(&self, id: u32) -> Meta {
        self.meta
            .get(id as usize)
            .copied()
            .unwrap_or(Meta { origin: ORIGIN_SYNTH, event: 0, dir: Dir::R, dgen: GEN_FRESH, litter: 0, litter_size: 1 })
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
        let mut distinct: Vec<u64> = Vec::with_capacity(ids.len());
        for &id in ids {
            let l = self.meta_of(id).litter;
            if !distinct.contains(&l) {
                distinct.push(l);
            }
        }
        let size = self.meta_of(ids[0]).litter_size;
        let full = distinct.len() == 1 && size >= 2 && ids.len() == size as usize;
        (distinct.len(), full)
    }

    // A new litter id. Unlike events these carry no tabu bookkeeping — a litter
    // is pure provenance.
    /// OR litter `l`'s ancestor set into `out`. Singleton sets are NOT stored:
    /// input gate `i` is litter `i` by construction, so any id below `anc_m`
    /// with no map entry denotes `{id}`. Ids at or above it with no entry are
    /// born-random material (twist brackets, insert pairs) and contribute
    /// nothing. That keeps init O(1) instead of O(input^2).
    fn anc_or_into(&self, l: u64, out: &mut [u64]) {
        if let Some(v) = self.anc.get(&l) {
            for (o, x) in out.iter_mut().zip(v.iter()) {
                *o |= *x;
            }
        } else if (l as usize) < self.anc_m {
            out[l as usize / 64] |= 1u64 << (l as usize % 64);
        }
    }

    /// Union the ancestor sets of `srcs`' litters and record it under a fresh
    /// litter id, which is returned. The union is what makes this survive
    /// mixed-lineage replacement, where the scalar `origin` label is discarded.
    fn anc_union_litter(&mut self, srcs: &[u64]) -> u64 {
        let l = self.fresh_litter();
        if self.anc_words == 0 {
            return l;
        }
        let mut bits = vec![0u64; self.anc_words];
        for &src in srcs {
            self.anc_or_into(src, &mut bits);
        }
        self.anc.insert(l, bits);
        l
    }

    /// Mean ancestor-set cardinality and mean normalised ancestor SPAN over
    /// live gates. Cardinality answers "what is a mixed gate made of"; span --
    /// (max index - min index) / (input - 1) -- answers "how far has input
    /// material travelled to meet". Both are immune to the ORIGIN_SYNTH erosion
    /// that makes odiff/oadj unreadable (see osyn=).
    fn anc_stats(&self) -> (f64, f64) {
        if self.anc_words == 0 {
            return (0.0, 0.0);
        }
        let (mut card_sum, mut span_sum, mut n) = (0f64, 0f64, 0u64);
        let mut bits = vec![0u64; self.anc_words];
        let mut cur = self.arena.head();
        while cur != NIL {
            bits.iter_mut().for_each(|w| *w = 0);
            self.anc_or_into(self.meta_of(cur).litter, &mut bits);
            let card: u32 = bits.iter().map(|w| w.count_ones()).sum();
            if card > 0 {
                let lo = bits
                    .iter()
                    .enumerate()
                    .find(|(_, w)| **w != 0)
                    .map(|(i, w)| i * 64 + w.trailing_zeros() as usize)
                    .unwrap_or(0);
                let hi = bits
                    .iter()
                    .enumerate()
                    .rev()
                    .find(|(_, w)| **w != 0)
                    .map(|(i, w)| i * 64 + 63 - w.leading_zeros() as usize)
                    .unwrap_or(0);
                card_sum += card as f64;
                span_sum += (hi.saturating_sub(lo)) as f64 / (self.anc_m.max(2) - 1) as f64;
                n += 1;
            }
            cur = self.arena.neighbor(cur, Dir::R);
        }
        if n == 0 { (0.0, 0.0) } else { (card_sum / n as f64, span_sum / n as f64) }
    }

    /// Log-bucketed histogram, rendered as `lo-hi:count` for non-empty buckets.
    fn log_hist(vals: &[u64]) -> String {
        let mut b = [0u64; 32];
        for &v in vals {
            let k = if v == 0 { 0 } else { 64 - (v.leading_zeros() as usize) };
            b[k.min(31)] += 1;
        }
        let mut out: Vec<String> = Vec::new();
        for (k, &c) in b.iter().enumerate() {
            if c == 0 {
                continue;
            }
            if k == 0 {
                out.push(format!("0:{c}"));
            } else {
                let lo = 1u64 << (k - 1);
                let hi = (1u64 << k) - 1;
                if lo == hi { out.push(format!("{lo}:{c}")) } else { out.push(format!("{lo}-{hi}:{c}")) }
            }
        }
        out.join(" ")
    }

    /// Ancestry in absolute units, with shapes rather than means.
    ///
    /// Three views of the same sets. `anc` is per-gate cardinality -- how many
    /// ORIGINAL gates a current gate descends from. `span` is per-gate, the
    /// index distance between the first and last of those ancestors, in
    /// original-circuit gate positions: how far apart in the input the material
    /// meeting in one gate came from. `fanout` is the dual, per INPUT gate: how
    /// many current gates carry any information about it. The two are linked by
    /// double counting, mean_fanout = mean_anc * gates / inputs, so quoting only
    /// the mean of one hides nothing -- but the distributions differ, and it is
    /// the tails that say whether spreading is uniform or a few gates are doing
    /// all the mixing.
    pub fn anc_report(&self) -> String {
        if self.anc_words == 0 {
            return String::new();
        }
        let mut cards: Vec<u64> = Vec::new();
        let mut spans: Vec<u64> = Vec::new();
        let mut fanout = vec![0u64; self.anc_m];
        let mut bits = vec![0u64; self.anc_words];
        let mut cur = self.arena.head();
        while cur != NIL {
            bits.iter_mut().for_each(|w| *w = 0);
            self.anc_or_into(self.meta_of(cur).litter, &mut bits);
            let mut lo = usize::MAX;
            let mut hi = 0usize;
            let mut card = 0u64;
            for (wi, &w) in bits.iter().enumerate() {
                let mut x = w;
                while x != 0 {
                    let b = x.trailing_zeros() as usize;
                    let idx = wi * 64 + b;
                    if idx < self.anc_m {
                        fanout[idx] += 1;
                        card += 1;
                        lo = lo.min(idx);
                        hi = hi.max(idx);
                    }
                    x &= x - 1;
                }
            }
            if card > 0 {
                cards.push(card);
                spans.push((hi - lo) as u64);
            }
            cur = self.arena.neighbor(cur, Dir::R);
        }
        let mean = |v: &[u64]| if v.is_empty() { 0.0 } else { v.iter().sum::<u64>() as f64 / v.len() as f64 };
        format!(
            "[fmix] ancestry: anc mean={:.1} [{}] | span(input gates) mean={:.0} [{}] | fanout/input mean={:.0} [{}]",
            mean(&cards),
            Self::log_hist(&cards),
            mean(&spans),
            Self::log_hist(&spans),
            mean(&fanout),
            Self::log_hist(&fanout)
        )
    }

    /// Drop ancestor sets for litters with no live gates. Without this the map
    /// grows with every splice for the whole run; with it, it is bounded by the
    /// live litter count.
    fn anc_prune(&mut self) {
        if self.anc_words == 0 {
            return;
        }
        let mut live: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut cur = self.arena.head();
        while cur != NIL {
            live.insert(self.meta_of(cur).litter);
            cur = self.arena.neighbor(cur, Dir::R);
        }
        self.anc.retain(|k, _| live.contains(k));
    }

    fn fresh_litter(&mut self) -> u64 {
        let l = self.next_litter;
        self.next_litter += 1;
        l
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
        let pos = bucket.iter().position(|&x| x == id).expect("id missing from index bucket");
        bucket.swap_remove(pos);
        if bucket.is_empty() {
            self.index.remove(&k);
        }
        self.indexed_count -= 1;
    }

    // ---- the chain ----


    // One first-class twist round: type drawn from the w_twist_* weights as
    // ratios, neg/swap 50/50 when all are zero.
    fn twist_round(&mut self) {
        let (wn, ws, wc) = (
            self.params.w_twist_neg.max(0.0),
            self.params.w_twist_swap.max(0.0),
            self.params.w_twist_cnot.max(0.0),
        );
        let total = wn + ws + wc;
        let kind = if total <= 0.0 {
            if self.rng.random_bool(0.5) { TwistKind::Neg } else { TwistKind::Swap }
        } else {
            let r = self.rng.random_range(0.0..total);
            if r < wn {
                TwistKind::Neg
            } else if r < wn + ws {
                TwistKind::Swap
            } else {
                TwistKind::Cnot
            }
        };
        self.twist_move(kind);
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

    pub fn run(&mut self) -> MixStop {
        while self.moves_done < self.params.moves {
            // Generation targeting: refresh the laggard list on its cadence
            // (an O(size) scan; entries invalidated between scans are pruned
            // lazily at draw time in pick_seed).
            if self.params.gen_target > 0 && self.moves_done >= self.pool_scan_due {
                self.rebuild_pool();
                self.pool_scan_due = self.moves_done + self.params.gen_rescan.max(1);
            }
            // Slot 0 continued: the size brake.
            self.apply_size_brake();
            // Slot 1: one twist, at a FIXED rate the rest of the machinery
            // balances around.
            let took_twist = self.params.p_twist > 0.0
                && self.rng.random_bool(self.params.p_twist.clamp(0.0, 1.0))
                && { self.twist_round(); true };
            // Slot 2: ONE DB move, under the live db_mode. A round whose
            // descent finds nothing is SPENT -- there is no fallthrough -- so
            // the thermostat receives exactly (1 - p_twist)(1 - p_db) of rounds
            // no matter how hard the material is.
            let took_db = !took_twist
                && self.params.p_db > 0.0
                && self.arena.len() >= 1
                && self.rng.random_bool(self.params.p_db.clamp(0.0, 1.0))
                && {
                    let mode = self.db_mode_cur;
                    let before = self.arena.len();
                    let hit = self.db_attempt(mode);
                    self.counters.db_slot2_rounds += 1;
                    if hit {
                        self.counters.db_slot2_hits += 1;
                        self.counters.db_slot2_added +=
                            self.arena.len().saturating_sub(before) as u64;
                    }
                    true
                };
            // Slot 3: the thermostat.
            if !took_twist && !took_db {
                let excess = self.arena.len() as f64 - self.params.target_size as f64;
                // In steer mode the 0.98 ceiling is the binding constraint on
                // holding size: its 2% expansion floor is a structural growth
                // source (measured +0.007/move) that saturated contraction
                // cannot absorb. Above target, steered runs contract harder.
                let hi = self.params.contract_ceiling;
                let p_contract =
                    (1.0 / (1.0 + (-excess / self.params.temp).exp())).clamp(0.02, hi);
                // Nothing to contract below two gates; every contraction channel
                // samples a linked node, so guard the empty/singleton arena (which
                // a DB move can reach on a near-identity region).
                if self.arena.len() >= 2 && self.rng.random_bool(p_contract) {
                    // Contraction channels with complementary stock; when one finds
                    // nothing, fall through to the next rather than wasting the move.
                    // The compressing DB replacement is tried first with probability
                    // w_db (the only channel that can contract non-ladder material),
                    // then undo/merge as before.
                    let did_db = self.params.p_comp > 0.0
                        && self.rng.random_bool(self.params.p_comp.clamp(0.0, 1.0))
                        && self.db_attempt_comp();
                    if did_db {
                        // done
                    } else if self.rng.random_bool(self.params.undo_frac) {
                        if !self.undo_move() {
                            self.merge_move();
                        }
                    } else if !self.merge_move() {
                        self.undo_move();
                    }
                } else {
                    self.expand_move();
                }
            }
            self.moves_done += 1;
            self.counters.moves = self.moves_done;
            if self.moves_done % self.params.verify_every == 0 {
                self.global_check();
            }
            if self.moves_done % self.params.report_every == 0 {
                self.report();
                self.check_flags();
                self.check_gen_snap();
                self.check_move_snap();
                if self.stop_requested {
                    self.global_check();
                    return MixStop::StopFlag;
                }
                if self.canary_fired() {
                    println!(
                        "[fmix] canary fired at move {}: {:.1}% of the last {} pool-seeded rounds failed at every rung (theta {}), fall-through {} — the pool is material the store cannot spell; stopping",
                        self.moves_done,
                        100.0 * self.canary_frac(),
                        self.canary.len(),
                        self.params.canary_theta,
                        self.counters.canary_fallthrough,
                    );
                    self.global_check();
                    return MixStop::CanaryFired;
                }
                if self.dose_reached() {
                    println!(
                        "[fmix] dose reached at move {}: all-gates laggard frac <= {} (circuit generation >= {}), twist coverage {:.1} — stopping",
                        self.moves_done,
                        self.params.gen_stop_frac,
                        self.params.gen_target,
                        self.twist_coverage(),
                    );
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
                println!("[fmix] stop flag seen at move {}: finishing cleanly", self.moves_done);
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
                            let mut s = String::with_capacity(gates.len() * 4);
                            for g in self.gens_in_order() {
                                s.push_str(&format!("{g}\n"));
                            }
                            if let Err(e) = std::fs::write(format!("{out}.gens"), s) {
                                eprintln!("[fmix] dump gens write failed: {e}");
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

    // Generation-multiple snapshots (--gen-snap-every): when the circuit
    // generation crosses a fresh multiple of the interval, write ONE verified
    // state and name it for every multiple crossed this interval (a report
    // gap can jump several multiples; the same state honestly serves each).
    fn check_gen_snap(&mut self) {
        let every = self.params.gen_snap_every;
        if every == 0 {
            return;
        }
        let Some(base) = self.gen_snap_base.clone() else {
            return;
        };
        let g = self.gen_stats().g_circ;
        let reached = (g / every) * every;
        if reached <= self.last_gen_snap {
            return;
        }
        self.global_check();
        let gates = self.arena.to_vec();
        let mut gens = String::with_capacity(gates.len() * 4);
        for gg in self.gens_in_order() {
            gens.push_str(&format!("{gg}\n"));
        }
        let mut prev: Option<String> = None;
        let mut m = self.last_gen_snap + every;
        while m <= reached {
            let out = format!("{base}.gen{m}.mpmct1");
            let ok = match &prev {
                None => {
                    let tmp = format!("{out}.tmp");
                    match super::format::write_mpmct(&tmp, &gates, self.num_wires) {
                        Ok(()) => match std::fs::rename(&tmp, &out) {
                            Ok(()) => true,
                            Err(e) => {
                                eprintln!("[fmix] gen-snap rename failed: {e}");
                                false
                            }
                        },
                        Err(e) => {
                            eprintln!("[fmix] gen-snap write failed: {e}");
                            false
                        }
                    }
                }
                Some(p) => match std::fs::copy(p, &out) {
                    Ok(_) => true,
                    Err(e) => {
                        eprintln!("[fmix] gen-snap copy failed: {e}");
                        false
                    }
                },
            };
            if ok {
                if let Err(e) = std::fs::write(format!("{out}.gens"), &gens) {
                    eprintln!("[fmix] gen-snap gens write failed: {e}");
                }
                println!(
                    "[fmix] GEN-SNAP: circuit generation {} >= {}: wrote {} gates to {} at move {} (verified, continuing)",
                    g,
                    m,
                    gates.len(),
                    out,
                    self.moves_done
                );
                prev = Some(out);
            }
            m += every;
        }
        self.last_gen_snap = reached;
    }

    // Move-multiple snapshots (--snap-every-moves): verified state at fixed
    // move-count multiples — the progress clock when the generation census
    // is not meaningful (pure-split runs).
    fn check_move_snap(&mut self) {
        let every = self.params.snap_every_moves;
        if every == 0 || self.moves_done % every != 0 {
            return;
        }
        let Some(base) = self.gen_snap_base.clone() else {
            return;
        };
        self.global_check();
        let gates = self.arena.to_vec();
        let out = format!("{base}.mv{}.mpmct1", self.moves_done);
        let tmp = format!("{out}.tmp");
        match super::format::write_mpmct(&tmp, &gates, self.num_wires) {
            Ok(()) => match std::fs::rename(&tmp, &out) {
                Ok(()) => {
                    let mut gens = String::with_capacity(gates.len() * 4);
                    for gg in self.gens_in_order() {
                        gens.push_str(&format!("{gg}\n"));
                    }
                    if let Err(e) = std::fs::write(format!("{out}.gens"), &gens) {
                        eprintln!("[fmix] move-snap gens write failed: {e}");
                    }
                    // A resumable state alongside the circuit. A circuit
                    // snapshot on its own cannot be continued -- directions,
                    // generations, litters, the journal and the original all
                    // live outside it -- so a long run could only be restarted,
                    // not resumed, from any point but its end. Written to a
                    // temp path and renamed, so an interrupted write never
                    // leaves a half-file that looks resumable.
                    let sp = format!("{out}.state");
                    let sptmp = format!("{sp}.tmp");
                    match self.save_state(&sptmp) {
                        Ok(()) => {
                            if let Err(e) = std::fs::rename(&sptmp, &sp) {
                                eprintln!("[fmix] move-snap state rename failed: {e}");
                            }
                        }
                        Err(e) => eprintln!("[fmix] move-snap state write failed: {e}"),
                    }
                    println!(
                        "[fmix] MOVE-SNAP: wrote {} gates to {} at move {} (verified, continuing)",
                        gates.len(),
                        out,
                        self.moves_done
                    );
                }
                Err(e) => eprintln!("[fmix] move-snap rename failed: {e}"),
            },
            Err(e) => eprintln!("[fmix] move-snap write failed: {e}"),
        }
    }

    // ---- expansion moves ----

    fn expand_move(&mut self) {
        // Expansion is cross-or-ANY-DB. Unsubsume, insert and fresh-split are
        // retired: the syntactic variety they supplied is supplied better by DB
        // re-spelling, and insert was the only source of material not descended
        // from the input, so retiring it also removes the born-random MAXGEN
        // case from everything except twist bracket packets. Twists are slot 1
        // now, so the expansion mix no longer performs them either.
        if self.params.p_any > 0.0
            && self.arena.len() >= 1
            && self.rng.random_bool(self.params.p_any.clamp(0.0, 1.0))
        {
            self.db_attempt(DbMode::SizeAgnostic);
            return;
        }
        self.cross_move();
    }

    fn cross_move(&mut self) {
        let id = self.arena.random_linked(&mut self.rng);
        self.cross_move_on(id);
    }

    fn cross_move_on(&mut self, id: u32) {
        let dir = self.meta_of(id).dir;
        let way = self.float_to_collision(id, dir);
        let h_id = self.arena.neighbor(id, dir);
        if h_id == NIL {
            self.counters.boundary += 1;
            self.retreat(id, way, dir);
            return;
        }
        let g = self.arena.gate(id).clone();
        let h = self.arena.gate(h_id).clone();

        if g.comp {
            if !self.split_allowed(g.width()) {
                self.counters.declined += 1;
                self.retreat(id, way, dir);
                return;
            }
            let pieces = rules::presplit(&g, &mut self.rng);
            if self.params.local_verify {
                assert!(
                    rules::verify_rewrite(std::slice::from_ref(&g), &pieces),
                    "presplit verification failed: {g:?} -> {pieces:?}"
                );
            }
            let pm = self.meta_of(id);
            let ev = self.fresh_event();
            for p in &pieces {
                self.counters.width_hist[p.width().min(15)] += 1;
            }
            let ids = self.splice_replace_one(id, pieces);
            for &pid in &ids {
                let d = self.child_dir(dir);
                self.set_meta(pid, Meta { origin: pm.origin, event: ev, dir: d, dgen: self.child_gen(pm.dgen), litter: pm.litter, litter_size: pm.litter_size });
            }
            self.advance_births(&ids);
            self.counters.presplits += 1;
            return;
        }

        match rules::cross(&g, &h, self.params.k_max, &mut self.rng) {
            Outcome::R0Swap => unreachable!("R0 after floating to collision"),
            Outcome::Blocked(BlockReason::WidthCap) => {
                self.counters.blocked_width += 1;
                self.retreat(id, way, dir);
            }
            Outcome::Blocked(BlockReason::Deadlock) => {
                self.counters.blocked_deadlock += 1;
                self.retreat(id, way, dir);
            }
            Outcome::PresplitColliding => {
                // The colliding gate is a g57 that must split: pre-splitting it
                // is this move's whole effect.
                if !self.split_allowed(h.width()) {
                    self.counters.declined += 1;
                    self.retreat(id, way, dir);
                    return;
                }
                let hp = rules::presplit(&h, &mut self.rng);
                if self.params.local_verify {
                    assert!(
                        rules::verify_rewrite(std::slice::from_ref(&h), &hp),
                        "colliding presplit verification failed: {h:?} -> {hp:?}"
                    );
                }
                let hm = self.meta_of(h_id);
                let ev = self.fresh_event();
                for p in &hp {
                    self.counters.width_hist[p.width().min(15)] += 1;
                }
                let ids = self.splice_replace_one(h_id, hp);
                for &pid in &ids {
                    // Colliding-gate fragments still inherit from the SHOT
                    // gate's direction (per spec: regardless of parent).
                    let d = self.child_dir(dir);
                    self.set_meta(pid, Meta { origin: hm.origin, event: ev, dir: d, dgen: self.child_gen(hm.dgen), litter: hm.litter, litter_size: hm.litter_size });
                }
                self.advance_births(&ids);
                self.counters.presplits += 1;
            }
            Outcome::Rewrite { seq, kind, dropped } => {
                let split_width = match kind {
                    RuleKind::R1 | RuleKind::R3 => g.width(),
                    RuleKind::R2 => h.width(),
                };
                if !self.split_allowed(split_width) {
                    self.counters.declined += 1;
                    self.retreat(id, way, dir);
                    return;
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
                let gm = self.meta_of(id);
                let hm = self.meta_of(h_id);
                let (g_origin, h_origin) = (gm.origin, hm.origin);
                let ev = self.fresh_event();
                let placed = self.splice_pair(id, h_id, dir, seq);
                let mut fresh: Vec<u32> = Vec::new();
                for &(pid, role) in &placed {
                    match role {
                        Role::ShotPiece | Role::Core => {
                            let d = self.child_dir(dir);
                            self.set_meta(
                                pid,
                                Meta { origin: g_origin, event: ev, dir: d, dgen: self.child_gen(gm.dgen), litter: gm.litter, litter_size: gm.litter_size },
                            );
                            fresh.push(pid);
                        }
                        Role::CollidingPiece => {
                            let d = self.child_dir(dir);
                            self.set_meta(
                                pid,
                                Meta { origin: h_origin, event: ev, dir: d, dgen: self.child_gen(hm.dgen), litter: hm.litter, litter_size: hm.litter_size },
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
                    let (before, origins, gens, litters, litter_sizes) = match dir {
                        Dir::R => (
                            [g.clone(), h.clone()],
                            [g_origin, h_origin],
                            [gm.dgen, hm.dgen],
                            [gm.litter, hm.litter],
                            [gm.litter_size, hm.litter_size],
                        ),
                        Dir::L => (
                            [h.clone(), g.clone()],
                            [h_origin, g_origin],
                            [hm.dgen, gm.dgen],
                            [hm.litter, gm.litter],
                            [hm.litter_size, gm.litter_size],
                        ),
                    };
                    let after: Vec<(u32, u32)> =
                        placed.iter().map(|&(i, _)| (i, self.arena.stamp(i))).collect();
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
                        gens,
                        litters,
                        litter_sizes,
                        misses: 0,
                    });
                }
            }
        }
    }

    /// Look for a welcoming neighbourhood: sample up to `twist_place_tries`
    /// candidate positions and return the first that matches any entry of
    /// TWIST_PATTERNS, together with the wire that pattern prefers. `None`
    /// means no candidate matched and the caller should place the twist
    /// uniformly at random, exactly as before.
    fn find_twist_site(&mut self) -> Option<(u32, u16)> {
        let tries = self.params.twist_place_tries;
        if tries == 0 || TWIST_PATTERNS.is_empty() || self.arena.len() < 2 {
            return None;
        }
        let max_span = TWIST_PATTERNS.iter().map(|p| p.span).max().unwrap_or(1);
        let mut run: Vec<XGate> = Vec::with_capacity(max_span);
        for _ in 0..tries {
            let at = self.arena.random_linked(&mut self.rng);
            run.clear();
            let mut cur = at;
            while run.len() < max_span && cur != NIL {
                run.push(self.arena.gate(cur).clone());
                cur = self.arena.neighbor(cur, Dir::R);
            }
            for pat in TWIST_PATTERNS {
                if run.len() < pat.span {
                    continue;
                }
                if let Some(w) = (pat.matches)(&run[..pat.span]) {
                    return Some((at, w));
                }
            }
        }
        None
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
        let len = (self.rng.random_range(lmin.ln()..=(cap as f64).ln()).exp().round() as usize)
            .clamp(2, cap);
        // Symmetric truncation: the window's virtual start is uniform over
        // [-(len-1), n-1] and the window is clamped to the circuit, so
        // left-overshooting draws pile their opening packets at the head
        // exactly as right-overshoots pile closings at the tail. Uniform
        // start with right-only truncation starves the head of straddling
        // frames (measured 100x head/tail decorrelation asymmetry, fx1tw).
        // Placement: prefer a position where a bracket can be absorbed, and
        // fall back to the symmetric random draw when none is found within the
        // try budget. `prefer_wire` is the absorbing gate's target, offered to
        // the P draw below.
        let site = self.find_twist_site();
        let prefer_wire = site.map(|(_, w)| w);
        if self.params.twist_place_tries > 0 {
            if site.is_some() {
                self.counters.twist_placed += 1;
            } else {
                self.counters.twist_place_fallback += 1;
            }
        }
        let (start, len) = match site {
            Some((at, _)) => (at, len),
            None => {
                let draw = self.rng.random_range(0..n + len - 1);
                if draw < len - 1 {
                    (self.arena.head(), draw + 1) // left-truncated: [0, draw+1)
                } else {
                    (self.arena.random_linked(&mut self.rng), len)
                }
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
                // Take the pattern's preferred wire when the window actually
                // reads it: that is the absorbing gate's target, so the bracket
                // this twist emits can be swallowed rather than paid for.
                let w = match prefer_wire {
                    Some(pw) if reads.contains(&pw) => pw,
                    _ => reads[self.rng.random_range(0..reads.len())],
                };
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
        if kind == TwistKind::Cnot {
            let mut cur = start;
            loop {
                match conj_by_cnot(self.arena.gate(cur), pa, pb) {
                    CnotConj::Blocked => {
                        self.counters.twist_skips += 1;
                        return;
                    }
                    CnotConj::Split(g0, _) if g0.width() > self.params.k_max => {
                        self.counters.twist_skips += 1;
                        return;
                    }
                    _ => {}
                }
                if cur == end {
                    break;
                }
                cur = self.arena.neighbor(cur, Dir::R);
            }
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
        let (mut w_start, mut w_end) = (start, end);
        let mut first = true;
        let mut cur = start;
        loop {
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
                    relabeled += 1;
                    (cur, cur)
                }
                Out::Two(g0, g1) => {
                    // The two halves fire on disjoint b-slices, so they
                    // commute: emit in random order.
                    let pieces =
                        if self.rng.random_bool(0.5) { vec![g0, g1] } else { vec![g1, g0] };
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
                            Meta { origin: m.origin, event: ev, dir: m.dir, dgen: self.child_gen(m.dgen), litter: m.litter, litter_size: m.litter_size },
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
                w_end = tail;
                break;
            }
            cur = next;
        }

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
            let lit = self.fresh_litter();
            self.set_meta(anchor, Meta { origin: ORIGIN_SYNTH, event: ev, dir: d, dgen: GEN_FRESH, litter: lit, litter_size: 1 });
        }
        let mut anchor = w_end;
        for g in &packet {
            self.counters.width_hist[g.width().min(15)] += 1;
            anchor = self.arena.insert_after(anchor, g.clone());
            self.index_add(anchor);
            let d = self.rand_dir();
            let lit = self.fresh_litter();
            self.set_meta(anchor, Meta { origin: ORIGIN_SYNTH, event: ev, dir: d, dgen: GEN_FRESH, litter: lit, litter_size: 1 });
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
            if self.is_tabu(self.journal[i].event) {
                self.counters.undo_tabu += 1;
                continue;
            }
            let e = self.journal.swap_remove_back(i).expect("journal index valid");
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
        debug_assert_eq!(block.len(), e.after.len(), "gathered block is not contiguous");
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
            self.set_meta(c, Meta { origin: e.origins[j], event: 0, dir: d, dgen: e.gens[j], litter: e.litters[j], litter_size: e.litter_sizes[j] });
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
        let mut consider = |ids: &[u32], out: &mut Vec<u32>| {
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
                g.ctrls.iter().enumerate().filter(|&(i, _)| i != skip).map(|(_, &(w, _))| w),
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
                    let wall =
                        g_colliders.iter().any(|&b| XGate::collides(self.arena.gate(b), hg));
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
        let left = if self.arena.neighbor(g_id, Dir::R) == h_id { g_id } else { h_id };
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
        for &nid in &new_ids {
            // Merged output stays in place (scatter is suspended) and keeps
            // shooting the way the initiating parent was headed, per dir_p.
            // Gen: the merged content depends on both parents, so it is only
            // as re-encoded as the LESS re-encoded of the two.
            let d = self.child_dir(mg.dir);
            // Litter follows the same parent the generation does: the merged
            // gate is only as re-encoded as the less re-encoded parent, so it
            // inherits that parent's provenance too.
            let (mut litter, litter_size) = if mg.dgen <= mh.dgen {
                (mg.litter, mg.litter_size)
            } else {
                (mh.litter, mh.litter_size)
            };
            // Ancestry, unlike provenance, comes from both: the merged content
            // depends on each parent, so under --ancestors the merge mints a
            // litter carrying the union rather than picking a side.
            if self.anc_words > 0 && mg.litter != mh.litter {
                litter = self.anc_union_litter(&[mg.litter, mh.litter]);
            }
            self.set_meta(nid, Meta { origin, event: 0, dir: d, dgen: mg.dgen.min(mh.dgen), litter, litter_size });
        }
        if sibling {
            self.counters.merges_sibling += 1;
        }
        match merged {
            Merge::Cancel => self.counters.merges_cancel += 1,
            Merge::XFuse(_) => self.counters.merges_xfuse += 1,
            Merge::DropLit(_) => self.counters.merges_drop += 1,
            Merge::Subsume(_) => self.counters.merges_subsume += 1,
            Merge::Absorb(_) => self.counters.merges_absorb += 1,
        }
        true
    }

    // ---- DB replacement moves ----
    //
    // Sample a window of [db_min_window, db_max_window] gates (contiguous or
    // convex, per db_sample; wide gates evaded per db_ctrl_cap), look it up in
    // the frozen store by its exact function polynomial (which handles the
    // conjunction-control "toffoli" gates the walk produces, not just g57s), and
    // splice in an equivalent circuit chosen per `mode`:
    //   Compressing  -> a non-growing equivalent, uniform among the shortest;
    //   SizeAgnostic -> any equivalent, uniform over all (may grow the circuit).
    // The sampled window is a contiguous run (convex sampling floats it together
    // first), so replacing it by an equal-function block preserves the circuit.
    // When `db_verify` is on the splice is checked exhaustively first (support
    // <= 24 wires); with it off the splice rests on the key/decode invariants and
    // the periodic global_check. Every attempt is recorded if --db-record is set.
    // Returns true iff a replacement was spliced in.
    // Miss-accounting wrapper: when the attempt was seeded on a laggard
    // (last_seed recorded by draw_laggard) and that exact gate survived the
    // round unconsumed (same id, same arena stamp — a reused id fails the
    // stamp check), the seed missed: bump its counter so the tier machinery
    // can graduate it to the paid channel or retire it as unreachable.
    /// A COMP-DB attempt that may be drawn as g57-only (see `p_comp_g57`).
    fn db_attempt_comp(&mut self) -> bool {
        let g57 = self.params.p_comp_g57 > 0.0
            && self.rng.random_bool(self.params.p_comp_g57.clamp(0.0, 1.0));
        self.db_g57_only = g57;
        if g57 {
            self.counters.db_g57_rounds += 1;
        }
        let hit = self.db_attempt(DbMode::Compressing);
        if g57 && hit {
            self.counters.db_g57_hits += 1;
        }
        self.db_g57_only = false;
        hit
    }

    fn db_attempt(&mut self, mode: DbMode) -> bool {
        self.db_seed_home = None;
        self.seed_from_pool = false;
        self.seed_fell_through = false;
        let spliced = self.db_attempt_inner(mode);
        // Canary accounting. A round QUALIFIES only when the seed genuinely
        // came from the pool; a heads coin that fell through because the pool
        // had drained is counted separately, because it means the rebuild is
        // too slow (scan more often) rather than that the material is
        // unreachable (stop the run) -- opposite remedies, so conflating them
        // would let a slow rescan masquerade as exhaustion.
        //
        // The canary SLEEPS under the brake: COMP declines far more often by
        // construction, and since this is a stop condition, mixing those
        // samples in would end runs for a reason that has nothing to do with
        // reachability.
        if self.seed_fell_through {
            self.counters.canary_fallthrough += 1;
        }
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
        // A failed attempt must leave no trace: put the seed back. On success
        // the seed was consumed by the splice, so there is nothing to restore.
        if spliced {
            self.db_seed_home = None;
        } else {
            self.restore_seed();
        }
        spliced
    }

    /// The canary condition: the trailing window is FULL and the failure
    /// fraction in it exceeds `canary_theta`. Buffer-fullness is the
    /// minimum-sample guard, and the window is denominated in qualifying rounds
    /// rather than moves -- the right units, since the qualifying rate itself
    /// varies with the fall-through rate.
    pub fn canary_fired(&self) -> bool {
        let w = self.params.canary_window.max(1);
        if self.params.canary_theta <= 0.0 || self.canary.len() < w {
            return false;
        }
        self.canary_failures as f64 / self.canary.len() as f64 > self.params.canary_theta
    }

    /// Failure fraction currently in the canary window (0 until it fills).
    pub fn canary_frac(&self) -> f64 {
        if self.canary.is_empty() {
            0.0
        } else {
            self.canary_failures as f64 / self.canary.len() as f64
        }
    }

    fn db_attempt_inner(&mut self, mode: DbMode) -> bool {
        let n = self.arena.len();
        let wmin = 1usize;
        let wmax = self.params.s_db.max(1);
        if n < wmin {
            return false;
        }
        // Prefix descent always starts at the top of the range — the descent
        // itself visits every shorter length, so sampling a shorter start
        // would only duplicate coverage.
        let wmax = if self.db_g57_only { self.params.s_db_g57.max(wmin) } else { wmax };
        let len = if self.params.db_prefixes || wmin == wmax {
            wmax.min(n)
        } else {
            self.rng.random_range(wmin..=wmax.min(n))
        };

        // Sample the window (contiguous or convex); g1dir drives the incoming
        // direction pivot below.
        let Some((ids, g1dir, smp)) = self.sample_best_window(len) else {
            return false;
        };
        // Stamped into every --db-record attempt line (smp=ctg|cvx) so stats
        // can split hits by sampler geometry, esp. under --db-sample mixed.
        self.db_last_sampler = smp;
        // Litter fragmentation census over the sampled window (observation only).
        let (distinct, _) = self.litter_census(&ids);
        self.counters.litter_windows += 1;
        self.counters.litter_distinct_sum += distinct as u64;
        let window: Vec<XGate> = ids.iter().map(|&id| self.arena.gate(id).clone()).collect();

        // Prefix descent, largest first: try the full k-gate window, then the
        // (k-1)-gate prefix, and so on down to db_min_window; splice the
        // LONGEST prefix with a usable match (max rewrite per round). Every
        // prefix attempt is recorded and counted. In dry-run the descent runs
        // to the bottom recording hits without splicing (full measurement);
        // live, the first hit splices and ends the round. A span-cap or
        // wide-verify decline keeps descending — shorter prefixes span fewer
        // wires and may still match.
        if self.params.db_prefixes {
            let wmin = 1usize;
            let guard = DegreeGuard {
                max_degree: self.params.db_max_degree,
                probes: self.params.db_degree_probes,
            };
            // Shrink from whichever end is FARTHER from the seed, so the seed
            // survives to the shortest rung. Dropping from a fixed end walks
            // away from the very gate the descent exists to re-encode: the seed
            // sits at the left edge only when its own direction is R, so a
            // fixed-end descent abandons it immediately on about half of
            // contiguous windows, and on every convex one, where the block
            // grows outward in both directions around it.
            let seed = self.db_seed_home.map(|(id, _)| id);
            let k = seed.and_then(|sd| ids.iter().position(|&x| x == sd)).unwrap_or(0);
            let (mut lo, mut hi) = (0usize, window.len() - 1);
            loop {
                let p = hi - lo + 1;
                if p < wmin {
                    break;
                }
                let prefix = &window[lo..=hi];
                // Full-litter ban: this rung is exactly the set some earlier
                // replacement emitted, so it is where the store is most likely
                // to hand that spelling straight back. Descending past it is
                // free -- a shorter rung is no longer a complete litter.
                if self.params.litter_ban && self.litter_census(&ids[lo..=hi]).1 {
                    self.counters.litter_banned += 1;
                    if p == wmin {
                        break;
                    }
                    if k.saturating_sub(lo) >= hi.saturating_sub(k) {
                        lo += 1;
                    } else {
                        hi -= 1;
                    }
                    continue;
                }
                self.counters.db_attempts += 1;
                if self.params.db_max_span > 0
                    && super::xpoly::xgate_used_wires(prefix).len() > self.params.db_max_span
                {
                    self.counters.db_span_skips += 1;
                    self.record_db_attempt(prefix, 0, None);
                    self.count_db_miss(mode);
                    continue;
                }
                let res = db_replace(
                    prefix,
                    self.num_wires,
                    &self.db,
                    self.db_budget,
                    mode,
                    guard,
                    self.params.curated,
                    &mut self.rng,
                );
                self.counters.db_identity_skips += res.identity_skipped as u64;
                if res.chosen.is_some() && res.chosen_curated {
                    self.counters.db_curated_hits += 1;
                }
                if let Some(ml) = res.min_match_len {
                    self.counters.dmin_windows += 1;
                    if ml < prefix.len() {
                        self.counters.dmin_shorter += 1;
                    }
                }
                if res.degree_skipped {
                    self.counters.db_degree_skips += 1;
                }
                let Some(replacement) = res.chosen else {
                    self.record_db_attempt(prefix, res.match_count, None);
                    self.count_db_miss(mode);
                    continue;
                };
                if self.params.db_dry_run {
                    self.record_db_attempt(prefix, res.match_count, None);
                    self.count_db_hit(mode);
                    continue;
                }
                if self.try_db_splice_curated(
                    res.chosen_curated,
                    &ids[lo..=hi],
                    g1dir,
                    prefix,
                    replacement,
                    res.match_count,
                    mode,
                ) {
                    return true;
                }
                if p == wmin {
                    break;
                }
                if k.saturating_sub(lo) >= hi.saturating_sub(k) {
                    lo += 1;
                } else {
                    hi -= 1;
                }
            }
            return false;
        }

        self.counters.db_attempts += 1;

        // Span guard: canonicalizing a wide-span window is the dominant cost
        // (Rule-L over large tied wire groups), and the store holds nothing
        // that wide — record the (near-certain) miss and move on.
        if self.params.db_max_span > 0
            && super::xpoly::xgate_used_wires(&window).len() > self.params.db_max_span
        {
            self.counters.db_span_skips += 1;
            self.record_db_attempt(&window, 0, None);
            match mode {
                DbMode::Compressing => self.counters.db_comp_misses += 1,
                DbMode::SizeAgnostic | DbMode::MinGrow | DbMode::Mix => self.counters.db_agn_misses += 1,
            }
            return false;
        }

        let guard = DegreeGuard {
            max_degree: self.params.db_max_degree,
            probes: self.params.db_degree_probes,
        };
        let res = db_replace(
            &window,
            self.num_wires,
            &self.db,
            self.db_budget,
            mode,
            guard,
            self.params.curated,
            &mut self.rng,
        );
        self.counters.db_identity_skips += res.identity_skipped as u64;
        if let Some(ml) = res.min_match_len {
            self.counters.dmin_windows += 1;
            if ml < window.len() {
                self.counters.dmin_shorter += 1;
            }
        }
        if res.chosen.is_some() && res.chosen_curated {
            self.counters.db_curated_hits += 1;
        }
        let match_count = res.match_count;
        if res.degree_skipped {
            self.counters.db_degree_skips += 1;
        }

        // Measurement mode: record the window + match count, never mutate.
        if self.params.db_dry_run {
            self.record_db_attempt(&window, match_count, None);
            if match_count > 0 {
                match mode {
                    DbMode::Compressing => self.counters.db_comp_hits += 1,
                    DbMode::SizeAgnostic | DbMode::MinGrow | DbMode::Mix => self.counters.db_agn_hits += 1,
                }
            } else {
                match mode {
                    DbMode::Compressing => self.counters.db_comp_misses += 1,
                    DbMode::SizeAgnostic | DbMode::MinGrow | DbMode::Mix => self.counters.db_agn_misses += 1,
                }
            }
            return false;
        }

        let curated_pick = res.chosen_curated;
        let Some(replacement) = res.chosen else {
            self.record_db_attempt(&window, match_count, None);
            self.count_db_miss(mode);
            return false;
        };
        self.try_db_splice_curated(curated_pick, &ids, g1dir, &window, replacement, match_count, mode)
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
            DbMode::SizeAgnostic | DbMode::MinGrow | DbMode::Mix => self.counters.db_agn_misses += 1,
        }
    }

    // Verify (optionally), record, and splice `replacement` over the window
    // nodes `ids` (whose gates are `window`). Returns false only when the
    // combined support exceeds verify_rewrite's 24-wire cap with verification
    // on — declined rather than spliced unchecked, recorded as a miss.
    fn try_db_splice_curated(
        &mut self,
        from_curated: bool,
        ids: &[u32],
        g1dir: Dir,
        window: &[XGate],
        replacement: Vec<XGate>,
        match_count: usize,
        mode: DbMode,
    ) -> bool {
        // Optional exhaustive equivalence check on the combined support.
        // verify_rewrite caps support at 24 wires; with verification on, a wider
        // window is not checkable so we decline it rather than splice unchecked.
        if self.params.db_verify {
            let mut support: Vec<u16> = window
                .iter()
                .chain(replacement.iter())
                .flat_map(|g| std::iter::once(g.target).chain(g.ctrls.iter().map(|&(w, _)| w)))
                .collect();
            support.sort_unstable();
            support.dedup();
            if support.len() > 24 {
                self.counters.db_wide_skip += 1;
                self.record_db_attempt(window, match_count, None);
                self.count_db_miss(mode);
                return false;
            }
            if !rules::verify_rewrite(window, &replacement) {
                // A CURATED replacement that fails verification is refused and
                // counted, not fatal. Curated entries are halves of split
                // minimal identities, and a half need not restore the helper
                // wires it borrows -- the observed failures write a wire the
                // window never touched. Until that invariant is pinned down,
                // treat curated as best-effort and let the run continue on the
                // regular store. A REGULAR failure stays fatal: there it would
                // mean the store or the canonicalisation is wrong, which is not
                // something to paper over.
                assert!(
                    from_curated,
                    "DB replacement verification failed: {window:?} -> {replacement:?}"
                );
                self.counters.db_curated_rejected += 1;
                self.record_db_attempt(window, match_count, None);
                self.count_db_miss(mode);
                return false;
            }
        }

        self.record_db_attempt(window, match_count, Some(&replacement));

        // Size accounting (replacement may be shorter, equal, or longer).
        let old = window.len();
        let new = replacement.len();
        if new <= old {
            self.counters.db_gates_removed += (old - new) as u64;
        } else {
            self.counters.db_gates_added += (new - old) as u64;
        }
        self.count_db_hit(mode);

        // Splice: insert the replacement after the node left of the window, then
        // unlink/free every window node (bumping stamps, which invalidates any
        // journal undo entry that referenced them).
        let cursor = self.arena.neighbor(ids[0], Dir::L);
        // Origin: keep the shared ancestor if the whole window agrees, else mark
        // the rewritten material synthetic (a DB block spans mixed lineage).
        let m0 = self.meta_of(ids[0]);
        let same_origin = ids.iter().all(|&id| self.meta_of(id).origin == m0.origin);
        let origin = if same_origin { m0.origin } else { ORIGIN_SYNTH };
        // Gen: a DB-replacement increment site. Products get g+1 where g is
        // the MEDIAN generation of the outgoing window — upper middle of the
        // sorted gens by default (median rounded up on even sizes, benchmark
        // semantics 2026-07-21), lower middle under gen_median_low.
        // Saturating: an all-fresh window stays fresh.
        let dgen = {
            let mut gens: Vec<u32> = ids.iter().map(|&id| self.meta_of(id).dgen).collect();
            gens.sort_unstable();
            let mid = if self.params.gen_median_low {
                gens.len().saturating_sub(1) / 2
            } else {
                gens.len() / 2
            };
            gens.get(mid).copied().unwrap_or(GEN_FRESH).saturating_add(1)
        };
        // Would an ssg-style full-litter ban have refused this splice? Counted,
        // not enforced.
        if self.litter_census(ids).1 {
            self.counters.litter_full_spliced += 1;
        }
        for &id in ids {
            self.index_remove(id);
            self.arena.unlink(id);
            self.arena.free_node(id);
        }
        // Incoming-gate directions: split the replacement at a pivot so the block
        // shoots outward from a point set by g1's original direction. g1 left ->
        // pivot at floor(2m/3); g1 right -> floor(m/3). Gates up to the pivot
        // (inclusive) head left, the rest head right.
        let m = replacement.len();
        let pivot = if g1dir == Dir::L { (2 * m) / 3 } else { m / 3 };
        // The splice is the litter-creating event: every product carries one
        // fresh id and the size this replacement emitted. A later window that
        // is exactly this set is the case where the store can hand the outgoing
        // spelling straight back (A -> B -> A).
        let litter = {
            let srcs: Vec<u64> = ids.iter().map(|&id| self.meta_of(id).litter).collect();
            self.anc_union_litter(&srcs)
        };
        let litter_size = m.min(u16::MAX as usize) as u16;
        {
            // Joint (outgoing, incoming) size of this splice.
            let (o, i) = (ids.len().min(SPLICE_HIST_MAX), m.min(SPLICE_HIST_MAX));
            if self.counters.splice_sizes.is_empty() {
                self.counters.splice_sizes =
                    vec![vec![0u64; SPLICE_HIST_MAX + 1]; SPLICE_HIST_MAX + 1];
            }
            self.counters.splice_sizes[o][i] += 1;
        }
        let mut c = cursor;
        let mut placed: Vec<u32> = Vec::with_capacity(m);
        for (i, gate) in replacement.into_iter().enumerate() {
            c = self.arena.insert_after(c, gate);
            self.index_add(c);
            let d = if i <= pivot { Dir::L } else { Dir::R };
            self.set_meta(c, Meta { origin, event: 0, dir: d, dgen, litter, litter_size });
            placed.push(c);
        }
        // Products ride their assigned direction outward, exactly as split
        // pieces do. Float-only, so the function is preserved by construction;
        // it also scatters the litter, which makes a later window less likely
        // to be exactly this replacement.
        if self.params.db_advance {
            self.advance_births(&placed);
        }
        true
    }

    // Append one record to --db-record: the outgoing window, the number of
    // equivalent DB circuits, and (on success) the replacing subcircuit. Gates
    // are printed as `target:comp:ctrl(pol)...`, one circuit per line.
    fn record_db_attempt(&mut self, window: &[XGate], matches: usize, repl: Option<&[XGate]>) {
        use std::io::Write;
        let smp = match self.db_last_sampler {
            DbSample::Convex => "cvx",
            _ => "ctg",
        };
        let Some(w) = self.db_record.as_mut() else { return };
        fn fmt(gates: &[XGate]) -> String {
            gates
                .iter()
                .map(|g| {
                    let ctrls: Vec<String> =
                        g.ctrls.iter().map(|&(wire, p)| format!("{wire}{}", if p { "+" } else { "-" })).collect();
                    format!("{}:{}:{}", g.target, g.comp as u8, ctrls.join(","))
                })
                .collect::<Vec<_>>()
                .join(" ")
        }
        let _ = writeln!(
            w,
            "attempt mv={} matches={} replaced={} smp={}",
            self.moves_done,
            matches,
            repl.is_some() as u8,
            smp
        );
        let _ = writeln!(w, "  out {}", fmt(window));
        if let Some(r) = repl {
            let _ = writeln!(w, "  in  {}", fmt(r));
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
        let (dir, k) = if off < dl { (Dir::L, dl - off) } else { (Dir::R, off - dl) };
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
        if self.rng.random_bool(0.5) { Dir::L } else { Dir::R }
    }

    // Fragment direction law: inherit the shot gate's direction with
    // probability dir_p, else the opposite.
    fn child_dir(&mut self, shot: Dir) -> Dir {
        if self.rng.random_bool(self.params.dir_p) { shot } else { shot.opposite() }
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

    // ---- DB window sampling ----
    //
    // Both samplers return a contiguous run of node ids (link order, leftmost
    // first) plus g1's stored direction (for the incoming-gate pivot rule), or
    // None when the attempt aborts (the L-cap could not be satisfied) or too few
    // gates could be gathered. Any floating they do is function-preserving, so a
    // subsequent miss leaves the circuit equivalent.

    fn width_of(&self, id: u32) -> usize {
        self.arena.gate(id).width()
    }

    // Does any gate of the contiguous span [lo..hi] collide with (not commute
    // with) gate `x`?
    fn span_collides(&self, lo: u32, hi: u32, x: u32) -> bool {
        let xg = self.arena.gate(x);
        let mut cur = lo;
        loop {
            if XGate::collides(self.arena.gate(cur), xg) {
                return true;
            }
            if cur == hi {
                return false;
            }
            cur = self.arena.neighbor(cur, Dir::R);
        }
    }

    // Move a commuting neighbor `x` from just past the span's `dir` end to the
    // far side of the span (one function-preserving hop of the whole block).
    fn move_across(&mut self, x: u32, lo: u32, hi: u32, dir: Dir) {
        self.arena.unlink(x);
        match dir {
            Dir::R => self.arena.link_before(x, lo), // block shifts right past x
            Dir::L => self.arena.link_after(x, hi),
        }
        self.counters.floats += 1;
        self.counters.float_steps += 1;
    }

    fn span_ids(&self, lo: u32, hi: u32) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut cur = lo;
        loop {
            ids.push(cur);
            if cur == hi {
                break;
            }
            cur = self.arena.neighbor(cur, Dir::R);
        }
        ids
    }

    // Returns the sampled window plus WHICH geometry actually built it (the
    // coin outcome under Mixed), so records and stats can split by sampler.
    /// Draw `litter_samples` candidate windows and keep the one spanning the
    /// most distinct litters. Diversity is the point: a window drawn from many
    /// replacement events is one no single earlier splice can undo, which is
    /// the same property the full-litter ban enforces at the other end.
    fn sample_best_window(&mut self, w: usize) -> Option<(Vec<u32>, Dir, DbSample)> {
        let n = self.params.litter_samples.max(1);
        let mut best: Option<(Vec<u32>, Dir, DbSample)> = None;
        let mut best_distinct = 0usize;
        for _ in 0..n {
            let Some(cand) = self.sample_window(w) else { continue };
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

    fn sample_window(&mut self, w: usize) -> Option<(Vec<u32>, Dir, DbSample)> {
        match if self.rng.random_bool(self.params.p_convex.clamp(0.0, 1.0)) {
            DbSample::Convex
        } else {
            DbSample::Contiguous
        } {
            DbSample::Contiguous => {
                self.collect_contiguous(w).map(|(ids, d)| (ids, d, DbSample::Contiguous))
            }
            DbSample::Convex => {
                self.collect_convex(w).map(|(ids, d)| (ids, d, DbSample::Convex))
            }
            DbSample::Mixed => {
                if self.rng.random_bool(0.5) {
                    self.collect_contiguous(w).map(|(ids, d)| (ids, d, DbSample::Contiguous))
                } else {
                    self.collect_convex(w).map(|(ids, d)| (ids, d, DbSample::Convex))
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

    /// True g57 shape: `comp = 1`, two controls, opposite polarity.
    fn gate_is_g57(&self, id: u32) -> bool {
        let g = self.arena.gate(id);
        g.comp && g.ctrls.len() == 2 && g.ctrls[0].1 != g.ctrls[1].1
    }

    /// May this gate sit inside the window currently being built? The width cap
    /// always applies; a g57-only COMP attempt additionally excludes everything
    /// that is not a g57, so the window cannot contain the intruder that
    /// collapses the long-window match rate.
    fn window_eligible(&self, id: u32) -> bool {
        let cap = self.params.w_window;
        if cap > 0 && self.width_of(id) >= cap {
            return false;
        }
        !self.db_g57_only || self.gate_is_g57(id)
    }

    /// May this gate seed a window and count toward the dose? Stricter than
    /// `window_eligible`: an ineligible gate can never be re-encoded, so its
    /// generation is pinned forever and an unfiltered pool converges on exactly
    /// that set and stays there.
    fn pool_eligible(&self, id: u32) -> bool {
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
            && self.params.p_mingen > 0.0
            && self.rng.random_bool(self.params.p_mingen.clamp(0.0, 1.0))
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

    // Generation of a split child (see MixParams::gen_split_inherit):
    // ratchet semantics give parent + 1, inherit semantics keep the parent
    // generation; MAXGEN stays MAXGEN either way (saturating).
    fn child_gen(&self, parent: u32) -> u32 {
        if self.params.gen_split_inherit { parent } else { parent.saturating_add(1) }
    }

    // Hard per-attempt bound on ctrl-cap evasion floats. Evading a wide gate
    // "succeeds" whenever it floats at least one step, so a window build whose
    // colliders are all wide can ping-pong between receding walls doing
    // unbounded arena work while the collected count never grows (observed as
    // a flat-RSS 100%-CPU livelock). Legitimate builds use a handful of
    // evasions; past this budget the attempt aborts and the round is spent.
    const EVADE_BUDGET: usize = 128;

    // Contiguous: g plus its w-1 neighbors in g's direction, spilling to the
    // other direction at the circuit end. A candidate with > L controls is first
    // floated out of the way; if it cannot float, the build reverses direction;
    // if that side is also blocked, the attempt aborts.
    fn collect_contiguous(&mut self, w: usize) -> Option<(Vec<u32>, Dir)> {
        let g1 = self.pick_seed()?;
        let dir1 = self.meta_of(g1).dir;
        let (mut lo, mut hi) = (g1, g1);
        let mut count = 1usize;
        let mut dir = dir1;
        let mut switched = false;
        let mut evade_budget = Self::EVADE_BUDGET;
        while count < w {
            let end = if dir == Dir::R { hi } else { lo };
            let x = self.arena.neighbor(end, dir);
            if x == NIL {
                if !switched {
                    dir = dir1.opposite();
                    switched = true;
                    continue;
                }
                break; // both ends reached the circuit boundary
            }
            if !self.window_eligible(x) {
                if evade_budget == 0 {
                    self.counters.db_build_aborts += 1;
                    return None;
                }
                evade_budget -= 1;
                if self.float_to_collision(x, dir) > 0 {
                    continue; // floated the wide gate out of the slot; retry
                }
                if !switched {
                    dir = dir1.opposite();
                    switched = true;
                    continue;
                }
                return None; // wide gate unavoidable on both sides -> abort
            }
            if dir == Dir::R {
                hi = x;
            } else {
                lo = x;
            }
            count += 1;
        }
        let ids = self.span_ids(lo, hi);
        if ids.is_empty() {
            return None;
        }
        Some((ids, dir1))
    }

    // Convex: float g1 to its first collider, then grow the block by floating it
    // (in dir1 w.p. p, else the opposite) to the next collider and absorbing it.
    // The L-cap evades a wide collider the same way (float it away; else reverse;
    // else abort).
    fn collect_convex(&mut self, w: usize) -> Option<(Vec<u32>, Dir)> {
        let p = self.params.db_convex_p;
        let g1 = self.pick_seed()?;
        let dir1 = self.meta_of(g1).dir;
        self.float_to_collision(g1, dir1);
        let (mut lo, mut hi) = (g1, g1);
        let mut count = 1usize;
        let mut evade_budget = Self::EVADE_BUDGET;

        while count < w {
            // The first collider is reached in g1's own direction (spec: float g1
            // in dir1 to hit g2); later steps randomize direction by p.
            let want = if count == 1 {
                dir1
            } else if self.rng.random_bool(p) {
                dir1
            } else {
                dir1.opposite()
            };
            // Float the block toward `want` to the next collider; if that side is
            // a boundary, try the opposite direction once.
            let (mut g3, mut dir) = self.float_block_to_collider(lo, hi, want);
            if g3 == NIL {
                let (g3b, dirb) = self.float_block_to_collider(lo, hi, want.opposite());
                if g3b == NIL {
                    break; // block is convex-maximal (both ends commute to the wall)
                }
                g3 = g3b;
                dir = dirb;
            }
            // L-cap: evade a wide collider.
            if !self.window_eligible(g3) {
                if evade_budget == 0 {
                    self.counters.db_build_aborts += 1;
                    return None;
                }
                evade_budget -= 1;
                if self.float_to_collision(g3, dir) > 0 {
                    continue; // g3 floated away; re-float the block to the next collider
                }
                // Reverse: look for a collider on the other side instead.
                let (g3r, dirr) = self.float_block_to_collider(lo, hi, dir.opposite());
                if g3r == NIL || !self.window_eligible(g3r) {
                    return None; // wide gate unavoidable -> abort
                }
                g3 = g3r;
                dir = dirr;
            }
            if dir == Dir::R {
                hi = g3;
            } else {
                lo = g3;
            }
            count += 1;
        }
        let ids = self.span_ids(lo, hi);
        if ids.is_empty() {
            return None;
        }
        Some((ids, dir1))
    }

    // Float the contiguous span [lo..hi] toward `dir` past commuting neighbors,
    // returning the first neighbor that collides with the block (and `dir`), or
    // (NIL, dir) at the boundary. Commuting neighbors are hopped to the far side.
    fn float_block_to_collider(&mut self, lo: u32, hi: u32, dir: Dir) -> (u32, Dir) {
        loop {
            let end = if dir == Dir::R { hi } else { lo };
            let x = self.arena.neighbor(end, dir);
            if x == NIL {
                return (NIL, dir);
            }
            if self.span_collides(lo, hi, x) {
                return (x, dir);
            }
            self.move_across(x, lo, hi, dir);
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

    fn splice_pair(&mut self, g_id: u32, h_id: u32, dir: Dir, seq: Vec<(XGate, Role)>) -> Vec<(u32, Role)> {
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

    /// True g57 shape: `comp = 1` with exactly two controls of OPPOSITE
    /// polarity (`a ^= b OR !c`). Distinct from `remaining_g57`, which counts
    /// every `comp = 1` gate regardless of width -- the report's `comp=` field.
    /// The DB stores g57 circuits, so this is the population it can spell.
    pub fn true_g57(&self) -> usize {
        let mut cur = self.arena.head();
        let mut n = 0usize;
        while cur != NIL {
            let g = self.arena.gate(cur);
            if g.comp && g.ctrls.len() == 2 && g.ctrls[0].1 != g.ctrls[1].1 {
                n += 1;
            }
            cur = self.arena.neighbor(cur, Dir::R);
        }
        n
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
            let mut set: Vec<u32> = ids[s..s + 32].iter().map(|&id| self.meta_of(id).origin).collect();
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

    /// Non-zero cells of the splice size histogram, as `out->in:count`.
    pub fn splice_size_line(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        for (o, row) in self.counters.splice_sizes.iter().enumerate() {
            for (i, &c) in row.iter().enumerate() {
                if c > 0 {
                    parts.push(format!("{o}->{i}:{c}"));
                }
            }
        }
        parts.join(" ")
    }

    pub fn report(&mut self) {
        self.anc_prune();
        // Flush the attempt recorder so a long run is inspectable mid-flight.
        if let Some(w) = self.db_record.as_mut() {
            use std::io::Write;
            let _ = w.flush();
        }
        let disp = self.origin_displacement();
        let owin = self.window_origin_diversity(64);
        let fan0 = self.fanout_zero_frac();
        let leew = self.mean_leeway(256, 4096);
        let origins = self.origins_in_order();
        let odiff = super::stats::origin_diffusion(&origins);
        let oadj = super::stats::adjacent_origin_autocorr(&origins);
        // Fraction of gates whose ancestry label has been destroyed. A DB splice
        // over a window spanning mixed lineage stamps its products ORIGIN_SYNTH,
        // and origin_diffusion / adjacent_origin_autocorr / origin_displacement
        // all SKIP those gates. So odiff, oadj and disp are computed over the
        // material mixing has failed to touch, and they get more selective the
        // better the mixing works. Without this field that bias is invisible:
        // read osyn first, and treat the other three as unusable once it is high.
        let (anc_card, anc_span) = self.anc_stats();
        let osyn = if origins.is_empty() {
            0.0
        } else {
            origins.iter().filter(|&&o| o == ORIGIN_SYNTH).count() as f64 / origins.len() as f64
        };
        let gs = self.gen_stats();
        let gmin = if gs.min == GEN_FRESH { "F".to_string() } else { gs.min.to_string() };
        let cov = self.twist_coverage();
        let c = &self.counters;
        let hist: Vec<String> = (0..=self.params.k_max.min(15))
            .map(|w| format!("{}:{}", w, c.width_hist[w]))
            .collect();
        println!(
            "[fmix] mv={} size={} target={} comp={} g57={} | merges c={} x={} d={} s={} a={} sib={} xorig={} tabu={} nopart={} wall={} far={} noadj={} | undo ok={} dead={} tabu={} miss={} live={} | db pdb={:.3} comp={}/{} agn={}/{} rm={} add={} wide={} dsk={} ssk={} bab={} idsk={} cur={}/{} g57only={}/{} | expand r1={} r2={} r3={} pre={} fresh={} unsub={} ins={} twn={} tws={} twc={} twrel={} twsplit={} twspan={} twskip={} | declined={} blockw={} dl={} bnd={} | floats={}/{} scat={}/{} | disp={:.4} owin={:.1} fan0={:.3} leew={:.0} odiff={:.4} oadj={:.4} osyn={:.3} anc={:.1} ancspan={:.3} width[{}] | gen tgt={} G={} Gall={} tgtbl={} alag={}/{} lag={}/{} wlag={} min={} cov={:.1} canary={:.3} cft={} | litter distinct={:.2} full={} ban={} tplace={}/{} dmin={:.3}",
            c.moves,
            self.arena.len(),
            self.params.target_size,
            self.remaining_g57(),
            self.true_g57(),
            c.merges_cancel,
            c.merges_xfuse,
            c.merges_drop,
            c.merges_subsume,
            c.merges_absorb,
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
            self.params.p_db,
            c.db_comp_hits,
            c.db_comp_misses,
            c.db_agn_hits,
            c.db_agn_misses,
            c.db_gates_removed,
            c.db_gates_added,
            c.db_wide_skip,
            c.db_degree_skips,
            c.db_span_skips,
            c.db_build_aborts,
            c.db_identity_skips,
            c.db_curated_hits,
            c.db_curated_rejected,
            c.db_g57_hits,
            c.db_g57_rounds,
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
            c.declined,
            c.blocked_width,
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
            osyn,
            anc_card,
            anc_span,
            hist.join(" "),
            self.params.gen_target,
            gs.g_circ,
            gs.g_all,
            gs.targetable,
            gs.all_lag,
            gs.total,
            gs.lag,
            gs.elig,
            gs.wlag,
            gmin,
            cov,
            self.canary_frac(),
            c.canary_fallthrough,
            if c.litter_windows > 0 {
                c.litter_distinct_sum as f64 / c.litter_windows as f64
            } else {
                0.0
            },
            c.litter_full_spliced,
            c.litter_banned,
            c.twist_placed,
            c.twist_place_fallback,
            if c.dmin_windows > 0 {
                c.dmin_shorter as f64 / c.dmin_windows as f64
            } else {
                0.0
            }
        );
        let anc = self.anc_report();
        if !anc.is_empty() {
            println!("{anc}");
        }
        let sizes = self.splice_size_line();
        if !sizes.is_empty() {
            println!("[fmix] splice sizes out->in: {sizes}");
        }
    }

    pub fn origins_in_order(&self) -> Vec<u32> {
        self.arena.ids_in_order().iter().map(|&id| self.meta_of(id).origin).collect()
    }

    /// Per-gate DB-generation stamps in circuit order (GEN_FRESH = born-random
    /// material that never held input structure).
    pub fn gens_in_order(&self) -> Vec<u32> {
        self.arena.ids_in_order().iter().map(|&id| self.meta_of(id).dgen).collect()
    }

    pub fn gen_stats(&self) -> GenStats {
        let target = self.params.gen_target;
        let mut s = GenStats {
            lag: 0,
            elig: 0,
            wlag: 0,
            min: GEN_FRESH,
            all_lag: 0,
            total: 0,
            g_circ: 0,
            g_all: 0,
            targetable: 0,
        };
        // Two bucketed gen histograms for the percentiles (everything at or
        // past the cap lands in the top bucket, incl. GEN_FRESH — fine, the
        // 5th percentile of interest sits far below it): `hist` over all
        // gates (g_all, kept for continuity) and `thist` over the targetable
        // ones (g_circ, the number that means something).
        const GB: usize = 1024;
        let mut hist = [0u64; GB];
        let mut thist = [0u64; GB];
        for id in self.arena.ids_in_order() {
            let m = self.meta_of(id);
            s.min = s.min.min(m.dgen);
            s.total += 1;
            hist[(m.dgen as usize).min(GB - 1)] += 1;
            if m.dgen < target {
                s.all_lag += 1;
            }
            let eligible = self.pool_eligible(id);
            if !eligible {
                if m.dgen < target {
                    s.wlag += 1;
                }
                continue;
            }
            s.elig += 1;
            if m.dgen >= target {
                // At or past target: targetable and done.
                s.targetable += 1;
                thist[(m.dgen as usize).min(GB - 1)] += 1;
                continue;
            }
            // Below target and eligible: this is the population targeting can
            // actually move, and the denominator the dose stop reads. Nothing
            // is written off any more -- with the descent reaching length 1,
            // which cannot decline for free, an eligible gate the store knows
            // at all advances; the residue that truly cannot is what the canary
            // is for, rather than a per-gate miss counter.
            s.lag += 1;
            s.targetable += 1;
            thist[(m.dgen as usize).min(GB - 1)] += 1;
        }
        // Largest G with >= 95% of the population at generation >= G: walk
        // the histogram until more than 5% lie strictly below G.
        let percentile = |h: &[u64; GB], population: u64| -> u32 {
            let allow = population / 20;
            let mut below = 0u64;
            let mut g_out = 0u32;
            for g in 0..GB {
                below += if g > 0 { h[g - 1] } else { 0 };
                if below <= allow {
                    g_out = g as u32;
                } else {
                    break;
                }
            }
            g_out
        };
        s.g_all = percentile(&hist, s.total);
        // With nothing targetable the generation census is meaningless; fall
        // back to the all-gates figure rather than reporting a bare 0.
        s.g_circ = if s.targetable == 0 {
            s.g_all
        } else {
            percentile(&thist, s.targetable)
        };
        s
    }

    /// Cumulative per-position twist coverage: total twisted window span over
    /// the current size — the phase-A twist dose meter (target ~600x per the
    /// saturation law).
    pub fn twist_coverage(&self) -> f64 {
        self.counters.twist_span as f64 / self.arena.len().max(1) as f64
    }

    // Dose-based stop (see MixParams::gen_stop_frac): the benchmark
    // criterion — the fraction of ALL gates still below gen_target at or
    // below gen_stop_frac (0.05 = "the circuit has generation >= target"),
    // AND twist coverage at or above twist_cov_stop. Wide and written-off
    // gates count against the fraction like everyone else; the walk lifts
    // them too, since split children get parent + 1.
    fn dose_reached(&self) -> bool {
        if self.params.gen_target == 0 || self.params.gen_stop_frac < 0.0 {
            return false;
        }
        let s = self.gen_stats();
        // Laggard fraction among the gates targeting can actually move. This
        // must NOT be all_lag/total: gates the DB channel can never re-encode
        // stay below target forever, so on material where they exceed the
        // stop fraction (a product-share gadget is ~62% wide) the all-gates
        // ratio never falls and the dose stop never fires — the run burns its
        // whole move budget after the dose is long since complete. With
        // everything targetable the two agree, so narrow runs are unaffected.
        if s.targetable == 0 {
            // Nothing is re-encodable: the dose is unmeasurable, not met.
            // Run the move budget rather than exiting immediately.
            return false;
        }
        let lag_frac = s.lag as f64 / s.targetable as f64;
        if lag_frac > self.params.gen_stop_frac {
            return false;
        }
        self.params.twist_cov_stop <= 0.0 || self.twist_coverage() >= self.params.twist_cov_stop
    }
}

#[cfg(test)]
mod mix_tests {
    use super::super::xgate::XGate;
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
                assert!(out.iter().all(|x| !x.comp), "merge emitted comp: {g:?}+{h:?}");
                assert!(
                    rules::verify_rewrite(&[g.clone(), h.clone()], &out),
                    "unsound merge: {g:?} + {h:?} -> {out:?}"
                );
                accepted += 1;
            }
        }
        assert!(accepted > 50, "catalogue accepted too few pairs to be tested: {accepted}");

        // The presplit pieces of a g57 (x and !x!y on the same target) XOR to
        // the complemented parent: must be rejected.
        let p0 = XGate::conj(0, [(1u16, true)]).unwrap();
        let p1 = XGate::conj(0, [(1u16, false), (2u16, false)]).unwrap();
        assert!(merge_result(&p0, &p1).is_none(), "presplit rejoin must be comp-guarded");

        // Two g57s differing in one polarity fuse into a conjunction (fossil
        // erosion), and a g57 plus its own monomial fuse into a NOT gate.
        let g57a = XGate { target: 0, comp: true, ctrls: p1.ctrls.clone() };
        let mut g57b = g57a.clone();
        g57b.ctrls[0].1 = true;
        match merge_result(&g57a, &g57b) {
            Some(Merge::DropLit(m)) => assert!(!m.comp && m.width() == 1),
            other => panic!("comp-comp polarity pair should DropLit, got {:?}", other.map(|m| m.gates())),
        }
        let mono = XGate { target: 0, comp: false, ctrls: g57a.ctrls.clone() };
        match merge_result(&g57a, &mono) {
            Some(Merge::XFuse(m)) => assert!(!m.comp && m.width() == 0),
            other => panic!("g57 + own monomial should XFuse, got {:?}", other.map(|m| m.gates())),
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
                    g.ctrls.iter().map(|&(cw, cp)| if cw == w { (cw, !cp) } else { (cw, cp) }),
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
            .map(|_| loop {
                let a = rng.random_range(0..wires);
                let x = rng.random_range(0..wires);
                let y = rng.random_range(0..wires);
                if a != x && a != y && x != y {
                    break XGate::from_g57([a, x, y]);
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
            // Erosion is presplit-driven and presplits are crossings, so this
            // needs enough rounds now that crossings are the only expansion.
            moves: 40_000,
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
        assert!(mx.origin_displacement() > 0.01, "no positional mixing at all");
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
            // Catalogue-invertible expansion no longer exists: fresh, unsub and
            // insert are retired, so crossings are the only growth channel and
            // the journal, not the pairwise catalogue, is what reverses them.
            w_cross: 1.0,
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
        // Expansion is cross-only now, so the drain is the thermostat pushing
        // contraction against crossings rather than against catalogue-invertible
        // stock. Give it the moves that costs.
        mx.params.moves += 60_000;
        mx.run();
        let drained = mx.arena.len();
        // Only a modest drain is available here, and that is the design rather
        // than a weakness of the test. Expansion is crossings now, and crossing
        // ladders are not pairwise-recoverable -- the catalogue cannot undo
        // them, so without a store the drain has the journal alone. Deep
        // contraction is COMP-DB's job (p_comp, which MixParams::default leaves
        // off precisely so tests need no store), and the size brake depends on
        // it. This asserts the thermostat still pushes the right way.
        assert!(drained < grown, "drain did not contract: {grown} -> {drained}");
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
        assert!(comp_now < comp0 / 2, "erosion too slow: {comp0} -> {comp_now}");
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
        assert!(exempted > 20, "exemption never fired in the sample: {exempted}");
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
                        rules::verify_rewrite(&[nw.clone(), g.clone(), nw], std::slice::from_ref(&g2)),
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
                        assert!(sandwich(std::slice::from_ref(&g2)), "cnot flip wrong: {g:?} a={a} b={b}");
                        assert_eq!(g2.width(), g.width());
                        assert_eq!(g2.comp, g.comp);
                    }
                    CnotConj::Split(x, y) => {
                        assert!(!g.comp, "comp gates must be Blocked, not Split");
                        assert!(sandwich(&[x.clone(), y.clone()]), "cnot split wrong: {g:?} a={a} b={b}");
                        // Disjoint b-slices: the pair commutes.
                        assert!(sandwich(&[y.clone(), x.clone()]), "cnot split pair does not commute");
                        assert_eq!(x.width(), g.width() + 1);
                        assert_eq!(y.width(), g.width() + 1);
                    }
                    CnotConj::Blocked => {
                        assert!(g.comp && g.reads(a) && !g.reads(b), "spurious Blocked: {g:?}");
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
            p_twist: 0.2, // slot 1 owns twists now; the w_* are type ratios
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
        assert!(mx.counters.twist_negs > 30, "negation twists barely ran: {}", mx.counters.twist_negs);
        assert!(mx.counters.twist_swaps > 30, "swap twists barely ran: {}", mx.counters.twist_swaps);
        assert!(mx.counters.twist_relabels > 0, "twists never relabeled a gate");
        assert!(mx.remaining_g57() <= comp0, "fossil count increased");
        assert!(mx.counters.merges() > 0, "no merges alongside twists");
        mx.global_check();
    }

    // The directional walk at its defaults (undo live, birth advance +
    // failed-cross retreat replacing the uniform scatter): function is
    // preserved through every sub-step (local_verify + periodic global checks),
    // crossings really run, and size stays bounded (the bound is a loose
    // runaway canary, not a promise). Inserts are retired, so crossings are the
    // whole expansion channel.
    #[test]
    fn directional_walk_preserves_function() {
        let gates = random_mixed_circuit(31, 16, 300);
        let params = MixParams {
            k_max: 5,
            moves: 20_000,
            target_size: 400,
            temp: 20.0,
            w_cross: 1.0,
            verify_every: 2_000,
            report_every: u64::MAX,
            seed: 7,
            ..MixParams::default()
        };
        let mut mx = Mixer::new(gates, 16, params);
        mx.run();
        let crossings = mx.counters.cross_r1 + mx.counters.cross_r2 + mx.counters.cross_r3;
        assert!(crossings > 50, "crossings barely ran: {crossings}");
        assert!(
            mx.counters.cross_r1 + mx.counters.cross_r2 + mx.counters.cross_r3 > 100,
            "crossings barely ran"
        );
        assert!(mx.counters.scatters > 0, "no directional birth advances");
        assert!(mx.counters.fresh_splits == 0, "fresh splits ran despite suspension");
        let n = mx.arena.len();
        assert!(n < 4 * 400, "size ran away under merge-only contraction: {n}");
        mx.global_check();
    }

    // Sampled windows are contiguous in link order and never break the circuit
    // function (both samplers only float commuting gates), and the control cap
    // keeps wide gates out of the collected window.
    #[test]
    fn window_samplers_are_contiguous_capped_and_function_preserving() {
        fn same_fn(a: &[XGate], b: &[XGate], nw: usize, seed: u64) -> bool {
            let mut rng = StdRng::seed_from_u64(seed);
            for _ in 0..8 {
                let init: Vec<u64> = (0..nw).map(|_| rng.random()).collect();
                let (mut sa, mut sb) = (init.clone(), init);
                super::super::xgate::eval_lanes(a.iter(), &mut sa);
                super::super::xgate::eval_lanes(b.iter(), &mut sb);
                if sa != sb {
                    return false;
                }
            }
            true
        }
        for sample in [DbSample::Contiguous, DbSample::Convex] {
            let gates = random_mixed_circuit(19, 16, 400);
            let reference = gates.clone();
            let params = MixParams {
                p_convex: if sample == DbSample::Convex { 1.0 } else { 0.0 },
                w_window: 3, // no window gate may reach width 3
                w_pool: 3,
                s_db: 6,
                db_convex_p: 0.75,
                report_every: u64::MAX,
                seed: 5,
                ..MixParams::default()
            };
            let mut mx = Mixer::new(gates, 16, params);
            let mut got = 0usize;
            for w in 0..4000 {
                let win = (w % 5) + 2; // window sizes 2..=6
                if let Some((ids, _dir, _smp)) = mx.sample_window(win) {
                    got += 1;
                    // contiguous in link order
                    for pair in ids.windows(2) {
                        assert_eq!(
                            mx.arena.neighbor(pair[0], Dir::R),
                            pair[1],
                            "{sample:?} window ids not contiguous"
                        );
                    }
                    assert!(ids.len() >= 2 && ids.len() <= win);
                    // control cap respected
                    for &id in &ids {
                        assert!(
                            mx.arena.gate(id).width() <= 2,
                            "{sample:?} window kept a gate wider than the cap"
                        );
                    }
                }
            }
            assert!(got > 500, "{sample:?} sampler almost never produced a window: {got}");
            // Sampling floats only commuting gates -> whole-circuit function intact.
            assert!(
                same_fn(&reference, &mx.arena.to_vec(), 16, 1),
                "{sample:?} sampling changed the circuit function"
            );
        }
    }

    #[test]
    fn contiguous_uses_gate_direction_and_spills_at_boundary() {
        // A 4-gate circuit on disjoint wires (all commute); a window of 4 from any
        // start must gather all four regardless of direction / boundary.
        let gates = vec![
            XGate::conj(0, [(1, true)]).unwrap(),
            XGate::conj(2, [(3, true)]).unwrap(),
            XGate::conj(4, [(5, true)]).unwrap(),
            XGate::conj(6, [(7, true)]).unwrap(),
        ];
        let params = MixParams { s_db: 2, report_every: u64::MAX, seed: 2, ..MixParams::default() };
        let mut mx = Mixer::new(gates, 8, params);
        for _ in 0..50 {
            let (ids, _d) = mx.collect_contiguous(4).expect("window");
            assert_eq!(ids.len(), 4, "boundary spill did not reach the quota");
        }
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
            p_twist: 0.2, // slot 1 owns twists now; the w_* are type ratios
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
            p_twist: 0.3, // slot 1 owns twists now
            w_twist_cnot: 0.50,
            twist_min_len: 4,
            verify_every: 1_000,
            report_every: u64::MAX,
            seed: 7,
            ..MixParams::default()
        };
        let mut mx = Mixer::new(gates, 16, params);
        mx.run();
        assert!(mx.counters.twist_cnots > 20, "cnot twists barely ran: {}", mx.counters.twist_cnots);
        assert!(mx.counters.twist_case_splits > 0, "no interior case-split ever happened");
        assert!(mx.remaining_g57() <= comp0, "fossil count increased");
        assert!(mx.counters.merges() > 0, "no merges alongside cnot twists");
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
        let mut a = Mixer::new(gates, 16, params);
        a.run();
        assert_eq!(
            a.counters.twist_negs
                + a.counters.twist_swaps
                + a.counters.twist_cnots
                + a.counters.twist_skips,
            0
        );
    }

    #[test]
    fn tabu_age_semantics() {
        let gates = random_g57_circuit(1, 8, 20);
        let params = MixParams { tabu_moves: 100, ..MixParams::default() };
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

    // A DB splice stamps its products with the outgoing window's
    // upper-median generation + 1 (benchmark semantics: median rounded up
    // on even window sizes).
    #[test]
    fn db_splice_stamps_upper_median_plus_one() {
        let g = XGate::conj(0, [(1, true)]).unwrap();
        let h = XGate::conj(2, [(3, true)]).unwrap();
        // Two adjacent identity pairs; we splice over the first pair.
        let gates = vec![g.clone(), g.clone(), h.clone(), h.clone()];
        let params = MixParams { report_every: u64::MAX, ..MixParams::default() };
        let mut mx = Mixer::new(gates, 8, params);
        let ids = mx.arena.ids_in_order();
        // Unequal gens across the window: the products must take min + 1.
        let m0 = mx.meta_of(ids[0]);
        mx.set_meta(ids[0], Meta { dgen: 3, ..m0 });
        let m1 = mx.meta_of(ids[1]);
        mx.set_meta(ids[1], Meta { dgen: 7, ..m1 });
        let window = vec![g.clone(), g.clone()];
        let replacement = vec![h.clone(), h.clone()]; // also an identity: verifies
        assert!(mx.try_db_splice_curated(
            false,
            &ids[..2],
            Dir::R,
            &window,
            replacement,
            1,
            DbMode::SizeAgnostic
        ));
        mx.global_check();
        let gens = mx.gens_in_order();
        assert_eq!(&gens[..2], &[8, 8], "products carry upper-median(3,7)+1 = 8: {gens:?}");
        assert_eq!(&gens[2..], &[0, 0], "untouched gates keep their gen: {gens:?}");
        // Lower-median variant: same {3,7} spread now stamps min+1 on the
        // 2-gate window.
        mx.params.gen_median_low = true;
        let ids = mx.arena.ids_in_order();
        let m0 = mx.meta_of(ids[0]);
        mx.set_meta(ids[0], Meta { dgen: 3, ..m0 });
        let m1 = mx.meta_of(ids[1]);
        mx.set_meta(ids[1], Meta { dgen: 7, ..m1 });
        let window = vec![h.clone(), h.clone()];
        let replacement = vec![g.clone(), g.clone()];
        assert!(mx.try_db_splice_curated(
            false,
            &ids[..2],
            Dir::R,
            &window,
            replacement,
            1,
            DbMode::SizeAgnostic
        ));
        assert_eq!(&mx.gens_in_order()[..2], &[4, 4], "lower median of (3,7) is 3 -> products 4");
        // Saturation: an all-fresh window stays fresh (either median).
        mx.params.gen_median_low = false;
        let ids = mx.arena.ids_in_order();
        for &id in &ids[..2] {
            let m = mx.meta_of(id);
            mx.set_meta(id, Meta { dgen: GEN_FRESH, ..m });
        }
        let window = vec![g.clone(), g.clone()];
        let replacement = vec![h.clone(), h.clone()];
        assert!(mx.try_db_splice_curated(
            false,
            &ids[..2],
            Dir::R,
            &window,
            replacement,
            1,
            DbMode::SizeAgnostic
        ));
        assert_eq!(mx.gens_in_order()[0], GEN_FRESH, "fresh window must stay fresh");
    }

    // Even without DB moves the walk lifts generations: split children get
    // parent + 1, so heavy churn mints intermediate generations strictly
    // between 0 and GEN_FRESH, while fresh material stays at GEN_FRESH.
    #[test]
    fn walk_splits_lift_generations() {
        let gates = random_mixed_circuit(29, 16, 300);
        let params = MixParams {
            k_max: 5,
            moves: 20_000,
            target_size: 600,
            temp: 20.0,
            p_twist: 0.1, // twist brackets are the only born-random source now
            w_twist_neg: 0.05,
            w_twist_swap: 0.05,
            verify_every: 5_000,
            report_every: u64::MAX,
            seed: 13,
            ..MixParams::default()
        };
        let mut mx = Mixer::new(gates, 16, params);
        mx.run();
        let gens = mx.gens_in_order();
        assert!(
            gens.iter().any(|&g| g > 0 && g != GEN_FRESH),
            "split children must climb above generation 0"
        );
        assert!(gens.contains(&GEN_FRESH), "twist brackets must be marked fresh");
        let s = mx.gen_stats();
        assert_eq!(s.total as usize, gens.len());
    }

    // The inherit split-rule variant: without DB moves nothing ever
    // increments, so after the same heavy churn every gate is either
    // still-original material (gen 0) or born-random (GEN_FRESH) — the
    // clean isolation of DB re-encoding depth from walk rewrite depth.
    #[test]
    fn inherit_split_rule_keeps_gens_binary_without_db() {
        let gates = random_mixed_circuit(29, 16, 300);
        let params = MixParams {
            k_max: 5,
            moves: 20_000,
            target_size: 600,
            temp: 20.0,
            w_twist_neg: 0.05,
            w_twist_swap: 0.05,
            gen_split_inherit: true,
            verify_every: 5_000,
            report_every: u64::MAX,
            seed: 13,
            ..MixParams::default()
        };
        let mut mx = Mixer::new(gates, 16, params);
        mx.run();
        let gens = mx.gens_in_order();
        assert!(
            gens.iter().all(|&g| g == 0 || g == GEN_FRESH),
            "under inherit semantics only DB splices may mint generations"
        );
        assert!(gens.contains(&0), "original material cannot all vanish here");
    }

// Checkpoint round-trip: a resumed chain must continue rather than restart.
    // The circuit file alone cannot do this -- directions, generations, litters,
    // the journal and the original are all outside it -- so the test checks the
    // state that has no other home, and that the resumed run still verifies
    // against the TRUE original rather than against its own resume point.
    #[test]
    fn checkpoint_round_trip_preserves_chain_state() {
        let gates = random_mixed_circuit(37, 16, 200);
        let params = MixParams {
            k_max: 5,
            moves: 4_000,
            target_size: 260,
            temp: 20.0,
            p_twist: 0.05,
            w_twist_neg: 1.0,
            gen_target: 3,
            verify_every: 1_000,
            report_every: u64::MAX,
            seed: 11,
            ..MixParams::default()
        };
        let mut mx = Mixer::new(gates.clone(), 16, params.clone());
        mx.run();
        let before_gates = mx.arena.to_vec();
        let before_meta: Vec<(u32, u32, u64)> = mx
            .arena
            .ids_in_order()
            .iter()
            .map(|&id| {
                let m = mx.meta_of(id);
                (m.dgen, m.origin, m.litter)
            })
            .collect();
        let before_moves = mx.moves_done;
        let before_twspan = mx.counters.twist_span;
        let dir_r = mx
            .arena
            .ids_in_order()
            .iter()
            .filter(|&&id| mx.meta_of(id).dir == Dir::R)
            .count();

        let path = std::env::temp_dir().join("fmix_ckpt_test.state");
        let path = path.to_str().unwrap();
        mx.save_state(path).expect("save");

        let mut rs = Mixer::resume_state(path, params, FrozenDb::empty()).expect("resume");
        assert_eq!(rs.arena.to_vec(), before_gates, "circuit must survive verbatim");
        assert_eq!(rs.moves_done, before_moves, "move counter must continue");
        assert_eq!(rs.counters.twist_span, before_twspan, "twist coverage feeds the dose stop");
        let after_meta: Vec<(u32, u32, u64)> = rs
            .arena
            .ids_in_order()
            .iter()
            .map(|&id| {
                let m = rs.meta_of(id);
                (m.dgen, m.origin, m.litter)
            })
            .collect();
        assert_eq!(after_meta, before_meta, "generations, origins and litters must survive");
        let after_dir_r = rs
            .arena
            .ids_in_order()
            .iter()
            .filter(|&&id| rs.meta_of(id).dir == Dir::R)
            .count();
        assert_eq!(after_dir_r, dir_r, "directions have no sidecar and must survive");
        // The resumed chain still verifies against the TRUE original.
        rs.global_check();
        rs.params.moves = before_moves + 2_000;
        rs.run();
        rs.global_check();
        assert!(rs.moves_done > before_moves, "resumed run must make progress");
        let _ = std::fs::remove_file(path);
    }

// The seed restore must be COLLISION-CHECKED, not an unchecked relink.
    // Window building floats gates other than the seed -- ctrl-cap evasion
    // parks a collider out of the way, and an evaded collider is by definition
    // one that does not commute with the window -- so restoring the seed by
    // teleporting it back to its recorded home can jump it across a gate it
    // does not commute with, changing the circuit's function. This reproduces
    // the regression: wide gates plus a low w_window make evasion fire on
    // nearly every attempt, and an empty store makes every attempt FAIL, so the
    // restore path runs constantly.
    #[test]
    fn failed_db_attempts_restore_seed_without_breaking_function() {
        let gates = random_mixed_circuit(97, 12, 400);
        let params = MixParams {
            k_max: 6,
            moves: 20_000,
            target_size: 400,
            temp: 20.0,
            p_db: 1.0,          // every round is a slot-2 attempt
            db_mode: DbMode::Mix,
            p_convex: 0.5,
            w_window: 2,        // width >= 2 is evaded: evasion on almost every build
            w_pool: 0,
            s_db: 5,
            verify_every: 1_000, // global_check catches any functional drift
            report_every: u64::MAX,
            seed: 5,
            ..MixParams::default()
        };
        // Empty store: every attempt misses, so every attempt restores its seed.
        let mut mx = Mixer::new_with_db(gates, 12, params, FrozenDb::empty());
        mx.run();
        mx.global_check();
        assert!(mx.counters.db_build_aborts > 0 || mx.counters.db_attempts > 0);
    }

    // The generation pool: with p_mingen 1.0 the seed comes from the pool
    // while one exists, and entries that got re-encoded (or freed) between
    // rebuilds are pruned at draw time.
    #[test]
    fn pick_seed_targets_pool_and_prunes_stale() {
        let gates = random_mixed_circuit(31, 16, 60);
        let params = MixParams {
            gen_target: 4,
            p_mingen: 1.0,
            w_pool: 0,
            report_every: u64::MAX,
            seed: 17,
            ..MixParams::default()
        };
        let mut mx = Mixer::new(gates, 16, params);
        let ids = mx.arena.ids_in_order();
        // Everyone re-encoded past target except three laggards.
        let lag: Vec<u32> = vec![ids[5], ids[20], ids[40]];
        for &id in &ids {
            let m = mx.meta_of(id);
            let g = if lag.contains(&id) { 1 } else { 9 };
            mx.set_meta(id, Meta { dgen: g, ..m });
        }
        mx.rebuild_pool();
        assert_eq!(mx.pool.len(), 3);
        for _ in 0..50 {
            let s = mx.pick_seed().expect("seed");
            assert!(lag.contains(&s), "p_mingen 1.0 must draw pool seeds while any exist");
        }
        // One crosses the target between rebuilds: draws prune it.
        let m = mx.meta_of(lag[0]);
        mx.set_meta(lag[0], Meta { dgen: 4, ..m });
        for _ in 0..200 {
            let s = mx.pick_seed().expect("seed");
            assert!(s != lag[0], "re-encoded gate must not be picked from the pool");
        }
        assert_eq!(mx.pool.len(), 2, "stale entry must be pruned at draw time");
    }

    // The pool is capped at pool_k, holding the LOWEST-generation gates: the
    // count is what bounds the drain between rebuilds, so it must be honoured
    // exactly rather than approximately.
    #[test]
    fn pool_keeps_only_k_lowest_generations() {
        let gates = random_mixed_circuit(43, 16, 60);
        let params = MixParams {
            gen_target: 100,
            pool_k: 5,
            w_pool: 0,
            report_every: u64::MAX,
            seed: 23,
            ..MixParams::default()
        };
        let mut mx = Mixer::new(gates, 16, params);
        let ids = mx.arena.ids_in_order();
        for (i, &id) in ids.iter().enumerate() {
            let m = mx.meta_of(id);
            mx.set_meta(id, Meta { dgen: i as u32, ..m });
        }
        mx.rebuild_pool();
        assert_eq!(mx.pool.len(), 5, "pool must be capped at pool_k");
        let mut gens: Vec<u32> = mx.pool.iter().map(|&id| mx.meta_of(id).dgen).collect();
        gens.sort_unstable();
        assert_eq!(gens, vec![0, 1, 2, 3, 4], "pool must hold the K lowest generations");
    }

    #[test]
    fn dose_stop_and_generation_count_targetable_gates() {
        let gates = random_mixed_circuit(37, 16, 40);
        let params = MixParams {
            gen_target: 2,
            gen_stop_frac: 0.0,
            w_window: 3,
            w_pool: 3,
            report_every: u64::MAX,
            ..MixParams::default()
        };
        let mut mx = Mixer::new(gates, 16, params);
        assert!(!mx.dose_reached(), "all-gen-0 input cannot be at dose");
        assert_eq!(mx.gen_stats().g_circ, 0);
        let ids = mx.arena.ids_in_order();
        let narrow: Vec<u32> =
            ids.iter().copied().filter(|&id| mx.width_of(id) <= 2).collect();
        let wide: Vec<u32> =
            ids.iter().copied().filter(|&id| mx.width_of(id) > 2).collect();
        assert!(
            !narrow.is_empty() && !wide.is_empty(),
            "fixture must contain both eligible and cap-ineligible gates"
        );
        for &id in &ids {
            let m = mx.meta_of(id);
            mx.set_meta(id, Meta { dgen: 3, ..m });
        }
        let s = mx.gen_stats();
        assert_eq!(s.all_lag, 0);
        assert_eq!(s.targetable, s.elig, "nothing written off yet");
        assert_eq!(s.g_circ, 3, "everything at 3 -> circuit generation 3");
        assert!(mx.dose_reached());

        // A WIDE straggler must not block the stop nor sink the generation:
        // the DB channel can never re-encode it, so it would pin both forever.
        // This is the product-share case, where ~62% of gates are wide.
        let m = mx.meta_of(wide[0]);
        mx.set_meta(wide[0], Meta { dgen: 0, ..m });
        let s = mx.gen_stats();
        assert_eq!(s.wlag, 1, "the wide gate lags");
        assert_eq!(s.lag, 0, "...but is not targetable");
        assert_eq!(s.g_circ, 3, "so the circuit generation is untouched");
        assert!(mx.dose_reached(), "a wide laggard cannot block the dose");
        // (One straggler in 40 is 2.5%, inside g_all's own 5% allowance, so
        // g_all does not move here either. The divergence between the two
        // shows up once the wide laggards exceed 5% — see the next test.)

        // An ELIGIBLE straggler does block at frac 0, and is inside a 5%
        // allowance.
        let m = mx.meta_of(narrow[0]);
        mx.set_meta(narrow[0], Meta { dgen: 0, ..m });
        assert!(!mx.dose_reached(), "an eligible laggard blocks at frac 0");
        let s = mx.gen_stats();
        assert_eq!(s.lag, 1);
        mx.params.gen_stop_frac = 1.0 / s.targetable as f64;
        assert!(mx.dose_reached(), "one laggard is within its own fraction");

        // Coverage requirement gates the stop until twists supply it.
        mx.params.twist_cov_stop = 10.0;
        assert!(!mx.dose_reached());
        mx.counters.twist_span = (mx.arena.len() as u64) * 11;
        assert!(mx.dose_reached());
        // Off switches.
        mx.params.gen_stop_frac = -1.0;
        assert!(!mx.dose_reached());
    }

    // The regression this fix exists for: on majority-wide material (a
    // product-share gadget) the old all-gates percentile pinned G= at 0 and
    // the dose stop could never fire, however complete the dose actually was.
    #[test]
    fn wide_majority_does_not_pin_the_generation_or_the_dose() {
        // 30 wide (3-control) gates + 10 narrow: 75% cap-ineligible, far past
        // the 5% allowance, so the all-gates percentile is stuck at 0.
        let mut gates: Vec<XGate> = Vec::new();
        let mut rng = StdRng::seed_from_u64(4242);
        while gates.len() < 30 {
            let g = rand_gate(&mut rng, 16, 3, false);
            if g.width() == 3 {
                gates.push(g);
            }
        }
        while gates.len() < 40 {
            let g = rand_gate(&mut rng, 16, 2, false);
            if g.width() <= 2 && g.width() >= 1 {
                gates.push(g);
            }
        }
        let params = MixParams {
            gen_target: 100,
            gen_stop_frac: 0.02,
            w_window: 3,
            w_pool: 3,
            report_every: u64::MAX,
            ..MixParams::default()
        };
        let mut mx = Mixer::new(gates, 16, params);
        // Drive every ELIGIBLE gate past the target, as a completed dose does.
        for &id in &mx.arena.ids_in_order() {
            if mx.width_of(id) <= 2 {
                let m = mx.meta_of(id);
                mx.set_meta(id, Meta { dgen: 100, ..m });
            }
        }
        let s = mx.gen_stats();
        assert!(s.wlag >= 30, "the wide majority still lags by construction");
        assert_eq!(s.lag, 0, "but the dose over eligible gates is complete");
        assert_eq!(s.g_all, 0, "the all-gates percentile is pinned at 0 (the bug)");
        assert_eq!(s.g_circ, 100, "the circuit generation reflects the real dose");
        assert!(
            mx.dose_reached(),
            "and the dose stop fires instead of burning the whole move budget"
        );
    }
}
