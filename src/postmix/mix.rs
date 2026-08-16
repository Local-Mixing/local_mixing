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
// is exhaustively verified on its support. The split/merge chain never emits
// comp=1 gates, so with the DB and g57 twists off the comp count is a monotone
// "fossil" count of surviving original g57s; DB splices (p_db > 0) and
// --twist-g57 brackets both emit g57-form material (real origins and
// ORIGIN_SYNTH respectively), so under either the comp/shaped censuses read
// population form, not fossils — tg_emitted is the twist-side odometer.
//
// Provenance: every gate carries (origin, event) — the original-gate index its
// material descends from, and the split event that created it. A merge whose
// partners share an event is a sibling re-merge (the undo of one split);
// recent events are tabu to keep freshly split pairs from instantly rejoining.
use super::arena::{Arena, Dir, NIL};
use super::db_replace::{db_replace, DbMode, DegreeGuard};
use super::rules::{self, BlockReason, Outcome, Role, RuleKind};
use super::swap_words;
use super::xgate::{Lits, XGate};
use super::xpoly::XPolyBudget;
use crate::replace::frozen::FrozenDb;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rustc_hash::FxHashMap;
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
pub const SPLICE_HIST_MAX: usize = 24;

/// Largest window length tracked in the per-outgoing-length DB breakdown.
/// Sized past any plausible `s_db` so a sweep never silently folds its widest
/// windows into the top bucket -- the point of the breakdown is exactly the
/// behaviour of the widest ones.
pub const LEN_HIST_MAX: usize = 32;

/// One mode's DB knobs after the base -> mode -> mode+geometry layering has
/// been applied. Produced by `Mixer::resolved_db_knobs`; the single source of
/// truth for what a mode will actually do.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolvedDbKnobs {
    pub s_db_cvx: usize,
    pub s_db_ctg: usize,
    pub p_convex: f64,
    pub p_mingen: f64,
    pub prefixes: bool,
}

/// How a DB move samples its outgoing window. Drawn ONCE per round, at the top
/// of `db_attempt_inner`, because the window length now depends on it: the GSS
/// profile wants a wide convex probe and a narrow contiguous one in the same
/// mode (`--s-db-ctg` / `--s-db-comp-ctg`).
///
/// The old `Mixed` variant and `DbSample::parse` are gone: geometry has been a
/// per-round `p_convex` coin since the sampler knobs were split per mode, so
/// nothing constructed `Mixed` and nothing called `parse`.
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
    /// Fuse the seed with a far COMMUTING partner: scan the seed's commutation
    /// box (the gates it could float past, capped at pair_scan_cap), pick one
    /// — the farthest eligible, or uniform under pair_pick_uniform — float the
    /// seed adjacent to it, and hand back the fused 2-gate window. The other
    /// samplers cannot build this window: Convex only absorbs colliders (a
    /// commuting gate is hopped past), and Contiguous pairs commuting gates
    /// only at physical distance 1. See docs/NONLOCAL_PHASE_A.md.
    Pair,
    /// Not a sampler: the tag stamped on the two endpoint splices of a bridge
    /// round (`bridge_round`), which fuses two gates that commutation CANNOT
    /// bring together by conjugating the interior through a carrier and
    /// re-spelling both carrier-adjacent windows. Never drawn by the geometry
    /// coin.
    Bridge,
}

/// One planned bridge round (docs/NONLOCAL_PHASE_A.md): the two target gates,
/// the carrier, and the exact conjugation wake for every interior collider.
struct BridgePlan {
    g1: u32,
    g2: u32,
    g1g: XGate,
    g2g: XGate,
    u: XGate,
    /// (interior collider id, its correction gates): the conjugate u·h·u is
    /// [h, corrections], verified exhaustively per collider before commit.
    wake: Vec<(u32, Vec<XGate>)>,
    interior_len: usize,
}

/// Outcome of merging a correction gate's literal list.
enum MergedConj {
    Gate(XGate),
    /// Contradictory literals: the correction never fires — the conjugate is
    /// the original gate unchanged.
    Never,
    /// Unbuildable (target among controls, or wider than the K-cap).
    Invalid,
}

fn merged_conj(target: u16, lits: &[(u16, bool)], k_max: usize) -> MergedConj {
    let mut merged: Vec<(u16, bool)> = Vec::with_capacity(lits.len());
    for &(w, p) in lits {
        if w == target {
            return MergedConj::Invalid;
        }
        match merged.iter().find(|&&(mw, _)| mw == w) {
            Some(&(_, mp)) if mp != p => return MergedConj::Never,
            Some(_) => {}
            None => merged.push((w, p)),
        }
    }
    if merged.len() > k_max {
        return MergedConj::Invalid;
    }
    match XGate::conj(target, merged) {
        Some(g) => MergedConj::Gate(g),
        None => MergedConj::Never,
    }
}

/// The exact conjugate u·h·u of one interior gate by the (comp = 0, single
/// monomial m) carrier u, expressed as [h, corrections] — the "adjust the
/// rest of the gates" half of a bridge round. Returns the corrections alone:
/// empty when h commutes with u or every correction is contradictory (the
/// conjugate is h unchanged), None when the pair is refused (mutual
/// collision, or a correction wider than k_max).
///
/// Modes (u fires on m = its control monomial; u's target is t_u):
/// - h READS t_u, literal λ there, other literals L (h's comp bit rides on
///   the kept copy): (λ⊕m)·L = λL ⊕ m∧L  →  corr = (t_h; m∧L).
/// - h WRITES one of m's wires, other u-literal ρ: the net t_u delta is
///   f_h∧ρ  →  corr = (t_h→t_u; c_h∧ρ); a comp-1 h has f_h = 1⊕mon, so two
///   corrections: (t_u; ρ) and (t_u; mon∧ρ).
/// - Both at once: None (the expansions interact; rare, skip the round).
fn conj_wake(u: &XGate, h: &XGate, k_max: usize) -> Option<Vec<XGate>> {
    debug_assert!(!u.comp && u.ctrls.len() == 2, "carrier is a 2-control conjunction");
    let tu = u.target;
    let reads_tu = h.reads(tu);
    let writes_m = u.reads(h.target);
    if !reads_tu && !writes_m {
        return Some(Vec::new()); // commuting (equal targets included)
    }
    if reads_tu && writes_m {
        return None;
    }
    let mut corrs: Vec<XGate> = Vec::new();
    if reads_tu {
        let lits: Vec<(u16, bool)> = u
            .ctrls
            .iter()
            .copied()
            .chain(h.ctrls.iter().copied().filter(|&(w, _)| w != tu))
            .collect();
        match merged_conj(h.target, &lits, k_max) {
            MergedConj::Gate(c) => corrs.push(c),
            MergedConj::Never => {}
            MergedConj::Invalid => return None,
        }
    } else {
        let rho = u.ctrls.iter().copied().find(|&(w, _)| w != h.target).expect("distinct wires");
        if h.comp {
            match merged_conj(tu, &[rho], k_max) {
                MergedConj::Gate(c) => corrs.push(c),
                MergedConj::Never => {}
                MergedConj::Invalid => return None,
            }
        }
        let lits: Vec<(u16, bool)> =
            h.ctrls.iter().copied().chain(std::iter::once(rho)).collect();
        match merged_conj(tu, &lits, k_max) {
            MergedConj::Gate(c) => corrs.push(c),
            MergedConj::Never => {}
            MergedConj::Invalid => return None,
        }
    }
    Some(corrs)
}

// ---- --twist-g57 placer tuning ----
// The two v2 placement features default ON; the env vars are kill-switches
// for factorial A/Bs (mirrors the SAMF_HIDE_PAIRS pattern in ssg).
/// How far a bracket may slide outward looking for an attachment gate.
pub const TG_SLIDE_CAP: usize = 512;
/// Attachment candidates actually solved per slide (first improving <= +4 wins).
pub const TG_SLIDE_TRIES: usize = 3;
/// Window redraws before settling for the best plan seen.
pub const TG_RETRIES: usize = 4;
/// A window is accepted outright when both seams total at most this net —
/// either both sides found homes (<= +4 each) or one side's match is good
/// enough (<= +2) to be worth the other staying bare.
pub const TG_ACCEPT_NET: i64 = 8;

fn tg_slide_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("TWIST_G57_NO_SLIDE").is_none())
}

fn tg_retry_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("TWIST_G57_NO_RETRY").is_none())
}

/// One planned bracket seam: the window boundary node it ends up at (possibly
/// slid outward from the drawn boundary), the context gates it consumes, and
/// the word that replaces context-plus-bracket.
struct TgSeam {
    edge: u32,
    ids: Vec<u32>,
    repl: Vec<XGate>,
}

impl TgSeam {
    fn net(&self) -> i64 {
        self.repl.len() as i64 - self.ids.len() as i64
    }
}

/// A fully-evaluated candidate twist: wires, final window, both seams.
struct TgPlan {
    a: u16,
    b: u16,
    l: TgSeam,
    r: TgSeam,
    slides: u64,
}

impl TgPlan {
    fn score(&self) -> (i64, i64) {
        // (total net, -consumed): cheapest first, more absorption on ties.
        let consumed = (self.l.ids.len() + self.r.ids.len()) as i64;
        (self.l.net() + self.r.net(), -consumed)
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
    let (w, _p) = extra?;
    let lits = big.ctrls.iter().map(|&(cw, cp)| if cw == w { (cw, !cp) } else { (cw, cp) });
    Some(Merge::Subsume(XGate::conj(big.target, lits).expect("subsume merge")))
}

// Merge-partner index key: target + control WIRE SET (polarities and comp
// excluded — cancel/xfuse partners share it exactly, drop-lit partners differ
// only in a polarity, and subsume partners are found by looking up the key
// with one wire removed). Hash collisions are harmless: merge_result rechecks.
fn merge_key(target: u16, wires: impl Iterator<Item = u16>) -> u64 {
    // FxHasher: the key is internal-only (never serialized) and collisions are
    // rechecked (above), so the hash function choice is unobservable.
    let mut h = rustc_hash::FxHasher::default();
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
// Split-stage rank restamp cadence, in moves (docs/FMIX_SPLIT_TWIST.md §5).
const RANK_EVERY: u64 = 8192;

// Flip the polarity of the literal on `w`, if the gate carries one. The
// in-place sibling of `conj_by_not`: exact for comp gates too (the flip
// happens inside the complemented conjunction).
fn flip_wire_literal(g: &mut XGate, w: u16) {
    for l in g.ctrls.iter_mut() {
        if l.0 == w {
            l.1 = !l.1;
        }
    }
}

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

// The twist menu is a single SWAP FAMILY (see `twist_move`): a wire swap,
// optionally composed with a negation of one or both swapped wires. The
// variant is chosen by two independent fair coins (alpha, beta) at move time,
// so there is no longer a kind enum -- swap 1/4, swap+negate-one 1/2,
// swap+negate-both 1/4. Each operator is realised as three single-control
// gates (a 3-CNOT swap network with control polarities carrying the negations).

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
    /// Probability each of the two swapped wires is NEGATED, independent per
    /// wire. 0.5 = the swap-family default (swap 1/4, negate-one 1/2,
    /// negate-both 1/4); 0.0 = pure positive swaps only (no polarity flips,
    /// though the 3-CNOT swap brackets are still inserted).
    pub twist_neg_p: f64,
    /// GLOBAL RE-RANDOMISATION rate. Each round, with probability
    /// `shuffle_rate / |circuit|`, every gate is floated to a uniformly random
    /// position inside its own commutation bounds. Rate 1.0 (the default) is
    /// therefore "expected one whole-circuit reshuffle per |circuit| rounds".
    /// 0 disables the move. Unlike a splice this changes only WHERE gates sit,
    /// never what they are, so it is function-preserving by construction --
    /// every step is a commutation.
    pub shuffle_rate: f64,
    /// Two-pass store routing: exhaust the curated store over EVERY window
    /// length before the regular store is consulted at any length. Only
    /// affects the prefix descent (there is no "every length" without one)
    /// and only while curated is armed for the mode -- expansion always,
    /// compression when `curated_in_comp` arms it too.
    pub curated_exhaust: bool,
    /// Arm the curated store for COMPRESSION too. Off by contract: uneven
    /// splits of a minimal identity give unequal halves, so the store holds
    /// longer-than-minimal spellings, and curated's lexicographic priority in
    /// `choose_ref` compounds that. On, curated joins the compression pool and
    /// the size rule keeps only what is strictly shorter than the window --
    /// the shorter halves, which are the ones worth having here.
    pub curated_in_comp: bool,
    // twist_g57: spell twist brackets as all-g57 words instead of 3-CNOT
    // packets, siting each bracket adaptively so it absorbs neighborhood
    // gates (the hidden-SAMF mechanism, XGate-native). Pure swap only: the
    // negation arms keep the legacy packet until their word tables exist.
    // Every gate the move inserts takes the ballistic birth-advance
    // unconditionally (the db_advance treatment; legacy brackets sit tight).
    // See twist_move_g57 and swap_words.rs.
    pub twist_g57: bool,
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
    /// Minimum DB window length (0 = no floor). With the prefix descent off,
    /// restricts drawn windows to db_min_window..=s_db exactly.
    pub db_min_window: usize,
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
    // Layer-2 mode overlay (slot 0). When p_mix >= 0, each round's slot-2 DB
    // move picks its mode by coin -- MIX-DB with probability p_mix, else
    // COMP-DB -- overriding the fixed db_mode, and reads that mode's own knobs.
    // The *_comp values are the COMP-mode overrides; each falls back to its base
    // value when unset (s_db_comp == 0, p_convex_comp < 0, p_mingen_comp < 0).
    // p_mix < 0 disables the overlay (single db_mode, exactly as before).
    pub p_mix: f64,
    // Pay-random MIX selection (layer 2 / phase A): when the MIX pool holds
    // only larger spellings, pick uniformly among ALL of them instead of
    // among the minimal ones — more growth and diversity per paid splice,
    // and a stronger up-lever for the profile controller.
    pub mix_pay_random: bool,
    // Layer-2 phase-A size profile (docs/POSTMIX_MANUAL §2.1.2): effective-
    // work marks [n0, n1, n2] and size ratios [r1, r2] vs the input size.
    // n2 == 0 -> no profile. While a profile is active the controller is the
    // ONLY size authority: it owns target_size (the thermostat pulls toward
    // the moving setpoint), the size brake is inert, and p_mix is driven by
    // the controller (best-effort contract; saturation is logged, never
    // fought).
    pub prof_n: [f64; 3],
    pub prof_r: [f64; 2],
    // Controller knobs: control cadence in eff units, relative deadband on
    // |S - S*|, max |Δp_mix| per update, EWMA weight for fresh per-round
    // drift estimates, integral gain.
    pub prof_cadence_eff: f64,
    pub prof_deadband: f64,
    pub prof_dp_max: f64,
    pub prof_ewma: f64,
    pub prof_ki: f64,
    // ---- Layered DB knobs: base -> mode -> mode+geometry ----
    //
    // Every override is `Option`, and `None` is the ONLY way to say "not set at
    // this level, fall through". This is deliberate and was a bug fix: these
    // used to be sentinel-encoded (0 for usize, < 0 for f64), which works only
    // while the shipped default IS the sentinel. It stopped being one --
    // `s_db_comp` ships at 12 and `p_convex_comp` at 0.9 -- so both fired
    // unconditionally and silently shadowed an explicit `--s-db` / `--p-convex`
    // in COMP rounds. Worse, 0 is a LEGITIMATE value here: `p_mingen_comp = 0`
    // is what the GSS profile wants, and a sentinel scheme cannot tell it from
    // "unset".
    //
    // Resolution is by specificity and lives entirely in the `active_*` methods
    // below. Deciding WHICH level the user actually asked for is the CLI's job
    // (fmix.rs), because only there can clap's ValueSource distinguish "the
    // user passed this value" from "this is merely the default".
    pub s_db_comp: Option<usize>,
    pub p_convex_comp: Option<f64>,
    pub p_mingen_comp: Option<f64>,
    // Per-GEOMETRY window length, on top of the per-MODE split. The two
    // samplers have very different cost curves -- a contiguous window of the
    // same gate count spans far more wires, and its canonicalization cost runs
    // 3.6x convex at length 5, 12.6x at 7 and 47.8x at 12 -- so a profile that
    // wants a wide convex probe usually wants a much narrower contiguous one.
    pub s_db_ctg: Option<usize>,
    pub s_db_comp_ctg: Option<usize>,
    // Per-MODE prefix descent. The mode overlay (--p-mix) runs MIX and COMP in
    // one process, and they want opposite settings: COMP descends (it is the
    // compression lever, worth ~600x on transport), while MIX draws one uniform
    // length (descent there just re-probes lengths whose expansion band is only
    // 1..~5).
    pub db_prefixes_mix: Option<bool>,
    pub db_prefixes_comp: Option<bool>,
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
    // ---- pair geometry (docs/NONLOCAL_PHASE_A.md) ----
    // p_pair: probability a non-COMP DB round samples its window with the PAIR
    // geometry: the seed plus one far COMMUTING partner, floated adjacent and
    // fused into a 2-gate window. The phase-A transport move — the fused
    // splice unions litters across the seed's whole commutation box instead
    // of one window span. 0 disables and DRAWS NO RNG: the walk is
    // move-for-move identical to the pair-less chain at equal seed.
    pub p_pair: f64,
    // Cap on the pair box scan (gates examined past the seed before giving up).
    pub pair_scan_cap: usize,
    // Partner policy: false = farthest eligible gate in the box (max transport
    // per move), true = uniform over the eligible box.
    pub pair_pick_uniform: bool,
    // ---- bridge fusion (docs/NONLOCAL_PHASE_A.md) ----
    // p_bridge: per-round probability of one bridge round — jointly re-encode
    // two gates commutation cannot bring together, by conjugating the interior
    // through a carrier (wake corrections on interior colliders) and
    // re-spelling both carrier-adjacent windows through the store. 0 disables
    // and DRAWS NO RNG. Corrections are non-g57 conjunctions: this move
    // trades polf for reach, like the legacy twist packets.
    pub p_bridge: f64,
    // Interior length draw is log-uniform in [bridge_min_span,
    // bridge_max_span].
    pub bridge_min_span: usize,
    pub bridge_max_span: usize,
    // Refuse a round whose interior holds more colliders than this (each
    // collider costs one or two correction gates).
    pub bridge_max_colliders: usize,
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
    /// SAMPLED ancestry: instead of the full ancestor set per litter (which
    /// costs |input| bits and caps the instrument at 20k input gates), track
    /// only `anc_samples` randomly chosen input gates -- "tracers" -- and for
    /// each one the set of current gates descended from it. Cost is a fixed
    /// K bits per litter regardless of input size, so this scales to production
    /// circuits. 0 = off; takes precedence over `ancestors` when both are set.
    pub anc_samples: usize,
    /// Tracer-selection seed. Default 0 means the tracer set is a function of
    /// (input size, K) alone, so it is identical across runs and across a
    /// resume -- which makes schedules comparable and makes a resumed run track
    /// the same input gates. Vary it for independent replicates.
    pub anc_sample_seed: u64,
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
    // ---- the split stage (docs/FMIX_SPLIT_TWIST.md) ----
    // split: arm the split stage. While it is live the split twist is the ONLY
    // move running (the round's other slots are withheld); the stage ends on
    // g57 exhaustion or split_fail_limit consecutive bracket failures, after
    // which the round runs under the parameters below as usual.
    pub split: bool,
    // split_stop: end the RUN at the stage boundary (MixStop::SplitDone)
    // instead of continuing into part 2 — the trial mode.
    pub split_stop: bool,
    // p_split_twist: layer-1 dispatch weight for split twists inside the twist
    // slot OUTSIDE the split stage (the stage itself forces 1.0).
    pub p_split_twist: f64,
    // p_join: probability a split carries the absorbed NOT twist + cross
    // (step 3 of the move); 1 - p_join of splits end after the split alone.
    pub p_join: f64,
    // Consecutive step-4e bracket failures that end the stage (exit B).
    pub split_fail_limit: u32,
    // Wire canaries planted at stage start (0 = off): flip monitors riding the
    // material, reported by ORIGINAL position at stage end.
    pub split_canaries: usize,
    // Length bias of the bracket draw: k candidates sampled on the picked
    // g57's own side, farthest wins. 1 = uniform, larger = longer spans.
    pub split_reach_k: usize,
    // ---- min-dgen cross-shot bias (docs/FMIX_SPLIT_TWIST.md addendum) ----
    // p_mincross: probability a cross shot is drawn from the min-dgen pool
    // (the K least-split lineages) instead of uniformly. The uniform draw is
    // a rich-get-richer sampler — families that already split carry more
    // gates and soak up proportionally more shots — so the median family
    // stays untouched while the tail grows; this coin points expansion work
    // at exactly the untouched families. 0 = off, and OFF DRAWS NO RNG: the
    // walk is move-for-move identical to the unbiased chain at equal seed.
    pub p_mincross: f64,
    // Pool size (a COUNT, like pool_k): must exceed the biased draws taken
    // between rebuilds (~ p_mincross x cross_rescan x share of cross rounds)
    // or the pool drains and the coin silently degrades to uniform.
    pub cross_pool_k: usize,
    // Pool rebuild cadence in moves (an O(size) scan + O(size) select each).
    pub cross_rescan: u64,
    pub verify_every: u64,
    pub report_every: u64,
    pub local_verify: bool,
    pub seed: u64,
}

impl MixParams {
    /// Settle the layered DB knobs for one mode: base -> mode -> mode+geometry,
    /// most specific wins, `None` meaning "this level says nothing".
    ///
    /// THE single source of these rules. The CLI banner calls it too, so what a
    /// run prints is by construction what the mixer will do -- the old banner
    /// re-derived the fall-through itself and could drift, which is how the
    /// COMP-shadowing bug stayed invisible.
    pub fn db_knobs(&self, mode: DbMode) -> ResolvedDbKnobs {
        let comp = mode == DbMode::Compressing;
        let opt = |o: Option<f64>| if comp { o } else { None };
        ResolvedDbKnobs {
            s_db_cvx: if comp { self.s_db_comp } else { None }.unwrap_or(self.s_db),
            // COMP contiguous falls back to COMP convex, not straight to the
            // base: a run that set --s-db-comp meant it for both geometries.
            s_db_ctg: if comp {
                self.s_db_comp_ctg.or(self.s_db_comp)
            } else {
                self.s_db_ctg
            }
            .unwrap_or(self.s_db),
            p_convex: opt(self.p_convex_comp).unwrap_or(self.p_convex),
            p_mingen: opt(self.p_mingen_comp).unwrap_or(self.p_mingen),
            prefixes: if comp { self.db_prefixes_comp } else { self.db_prefixes_mix }
                .unwrap_or(self.db_prefixes),
        }
    }
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
            twist_neg_p: 0.5,
            shuffle_rate: 2.0,
            curated_exhaust: false,
            curated_in_comp: false,
            twist_g57: false,
            twist_min_len: 64,
            // Store-free by default: MixParams::default() is the test/base
            // value, and any positive DB rate here would make every construction
            // demand FROZEN_DB_DIR. The PRODUCTION defaults live on the fmix
            // CLI, where a run that wants the store asks for it -- as of
            // 2026-08-03 that means s_db 9, p_convex 0.4, s_db_comp 12,
            // p_convex_comp 0.9, db_prefixes/curated/curated_exhaust/
            // curated_in_comp all ON. The values below stay the store-free
            // test baseline on purpose; do not "sync" them to the CLI.
            p_comp: 0.0,
            p_any: 0.0,
            p_db: 0.0,
            s_db: 5,
            db_min_window: 0,
            p_convex: 0.5,
            db_mode: DbMode::Mix,
            p_mix: -1.0,
            mix_pay_random: false,
            prof_n: [0.0; 3],
            prof_r: [0.0; 2],
            prof_cadence_eff: 0.5,
            prof_deadband: 0.02,
            prof_dp_max: 0.1,
            prof_ewma: 0.3,
            prof_ki: 0.05,
            s_db_comp: None,
            p_convex_comp: None,
            p_mingen_comp: None,
            s_db_ctg: None,
            s_db_comp_ctg: None,
            db_prefixes_mix: None,
            db_prefixes_comp: None,
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
            p_pair: 0.0,
            pair_scan_cap: 4096,
            pair_pick_uniform: false,
            p_bridge: 0.0,
            bridge_min_span: 16,
            bridge_max_span: 512,
            bridge_max_colliders: 8,
            curated: false,
            ancestors: false,
            anc_samples: 0,
            anc_sample_seed: 0,
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
            split: false,
            split_stop: false,
            p_split_twist: 0.0,
            p_join: 0.8,
            split_fail_limit: 100,
            split_canaries: 256,
            split_reach_k: 2,
            p_mincross: 0.0,
            cross_pool_k: 20_000,
            cross_rescan: 10_000,
            verify_every: 10_000,
            report_every: 50_000,
            local_verify: true,
            seed: 0,
        }
    }
}

/// The width-2 `comp = 1` population, split by control polarity.
///
/// `shaped == same_pol + opp_pol` by construction. The split separates two
/// things the single `g57=` field used to conflate:
///
/// - **`shaped`** is the DB-effectiveness reading. The store emits g57 circuits
///   and nothing else, so every gate the DB has ever spliced in has this shape,
///   and `1 - shaped/size` is the material the DB did not produce. Measured:
///   with twists off, `opp_pol/comp` holds at 1.000 for a whole run.
/// - **`same_pol`** is a twist odometer, not a structural fact. A negation
///   twist conjugates a window by NOT on one wire, flipping the polarity of
///   every literal on it (`conj_by_not`). A g57's two controls sit on distinct
///   wires, so a twist touches at most one, and touching one flips the pair
///   from opposite to same: same gate, same width, same `comp`, only the
///   polarity relation moves. It random-walks toward 1/2 under twist pressure
///   while DB splices inject fresh opposite-polarity material and pull it back.
///
/// Swap twists carry polarity with the wire and leave the split alone. A/B at
/// equal twist count, 400k moves: `opp_pol/comp` = 1.000 (no twists), 0.704
/// (neg), 1.000 (swap, at 2.6x the relabel count).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct G57Census {
    /// `comp = 1` with exactly two controls, polarity ignored: the shape the
    /// store can spell, hence the shape of everything it emits.
    pub shaped: usize,
    /// Of `shaped`, both controls the SAME polarity -- not a storable g57.
    pub same_pol: usize,
    /// Of `shaped`, OPPOSITE polarity (`a ^= b OR !c`) -- a true g57.
    pub opp_pol: usize,
}

impl G57Census {
    /// Fraction of the shaped population twists have flipped out of g57 form.
    /// 0 = untwisted, 1/2 = fully scrambled. Bounded and size-independent, so
    /// unlike `cov` -- whose denominator is the growing circuit -- it cannot
    /// fall while twists keep firing.
    pub fn pol_flipped(&self) -> f64 {
        if self.shaped == 0 { 0.0 } else { self.same_pol as f64 / self.shaped as f64 }
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
    // Same joint histogram, curated-store splices only (splice_sizes minus
    // this = regular). A shape: not carried across resumes.
    pub splice_sizes_curated: Vec<Vec<u64>>,
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
    // Per-mode split of the same deltas (Compressing vs the rest), the
    // profile controller's plant observables: ghat = (mix_added -
    // mix_removed)/mix rounds, shat = (comp_removed - comp_added)/comp
    // rounds. Session-local (zero after a resume, like the histograms).
    // SELECTION ENTROPY of successful splices (session-local). `choice_*`
    // counts splices by whether the mode's eligible set held more than one
    // candidate; `choice_sum` totals the eligible-set sizes and
    // `choice_bits_milli` totals log2(eligible) in millibits, so the run can
    // report both "how often was there a choice", "how many on average" and
    // the actual entropy those choices injected through WHICH gates were
    // spliced -- as opposed to where.
    pub shuffles: u64,
    pub shuffle_moved: u64,
    pub shuffle_steps: u64,
    /// Wall nanoseconds spent inside global_shuffle. The move's cost was
    /// being INFERRED from displacement counters and a separate final-float
    /// timing; this measures it.
    pub shuffle_ns: u64,
    pub choice_splices: u64,
    pub choice_multi: u64,
    pub choice_sum: u64,
    pub choice_bits_milli: u64,
    pub db_mix_added: u64,
    pub db_mix_removed: u64,
    pub db_cmp_added: u64,
    pub db_cmp_removed: u64,
    pub db_wide_skip: u64,     // support > 64 wires or budget hit: UNDECIDABLE, skipped
    // Wide windows (> 24 wires) verified by ANF comparison instead of the
    // exhaustive evaluator. Session-local.
    pub db_wide_poly: u64,
    // PER-OUTGOING-LENGTH breakdown of the DB move, indexed by the sampled
    // window's gate count (0..=LEN_HIST_MAX). Without it every rate the run
    // reports is an average over a random length draw, which conflates "this
    // width works" with "this width was sampled often". Session-local shapes,
    // like splice_sizes.
    pub len_attempts: Vec<u64>,
    pub len_hits: Vec<u64>,
    pub len_removed: Vec<u64>,
    pub len_added: Vec<u64>,
    pub len_span_skip: Vec<u64>,
    pub len_deg_skip: Vec<u64>,
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
    // --twist-g57 seam economics: context gates the brackets consumed, g57
    // word gates they emitted (net growth = emitted - consumed vs a flat +6
    // for the legacy packets), and total MITM solve time (the online-cost
    // answer, session-local).
    pub tg_consumed: u64,
    pub tg_emitted: u64,
    pub tg_solves: u64,
    pub tg_solve_ns: u64,
    // v2 placement gauges (session-local): seams that found their home by
    // sliding the window edge, and extra window redraws the joint-acceptance
    // rule spent.
    pub tg_slides: u64,
    pub tg_retries: u64,
    // Histogram of per-seam net cost (word len - context consumed), 0..=7.
    // A shape, so like splice_sizes it is not carried across resumes.
    pub tg_net_hist: [u64; 8],
    pub blocked_width: u64,
    pub blocked_deadlock: u64,
    pub declined: u64,
    pub boundary: u64,
    pub floats: u64,
    pub float_steps: u64,
    pub scatters: u64,
    pub scatter_steps: u64,
    pub dropped_neverfire: u64,
    // ---- split stage (docs/FMIX_SPLIT_TWIST.md) ----
    // Splits of the picked g57 (step 2), of the bracket g57 (4a/4c), and the
    // forced segment splits (5a); twist successes, step-4e failures, and
    // successes whose brackets sat in different circuit halves.
    pub split_prims: u64,
    pub split_hsplits: u64,
    pub split_segs: u64,
    pub split_joins: u64,
    pub split_fails: u64,
    pub split_xmid: u64,
    // Total canary flips (each = a twist span complementing the value carried
    // at a canary's position on its wire).
    pub tap_flips: u64,
    // Sum of twist span lengths (gates strictly between the brackets), and
    // the span-as-fraction-of-circuit histogram in 5% buckets (a SHAPE, not
    // carried across resumes, like width_hist).
    pub split_span_sum: u64,
    pub split_span_hist: [u64; 20],
    // Cross shots drawn from the min-dgen pool (vs uniform).
    pub cross_pool_shots: u64,
    // ---- pair geometry (docs/NONLOCAL_PHASE_A.md; session-local) ----
    // Rounds that drew the pair geometry, scans with no eligible partner,
    // scans cut by pair_scan_cap, fused windows and their splices, box-size
    // and fused-transport-distance tallies (sum/max over fused windows), and
    // candidates refused by the reorder ban.
    pub pair_rounds: u64,
    pub pair_boxes_empty: u64,
    pub pair_scan_truncs: u64,
    pub pair_fused: u64,
    pub pair_splices: u64,
    pub pair_box_sum: u64,
    pub pair_box_max: u64,
    pub pair_dist_sum: u64,
    pub pair_dist_max: u64,
    pub pair_perm_skips: u64,
    // ---- bridge fusion (docs/NONLOCAL_PHASE_A.md; session-local) ----
    // Rounds, walks clipped at the circuit tail, plans refused (carrier
    // unbuildable / mode-c collider / collider budget), pre-insert store
    // misses, far-splice rollbacks, half commits (near window missed after
    // the far one spliced), full commits, interior-length and collider
    // tallies, and wake correction gates inserted.
    pub bridge_rounds: u64,
    pub bridge_short: u64,
    pub bridge_refused: u64,
    pub bridge_probe_miss: u64,
    pub bridge_rollbacks: u64,
    pub bridge_half: u64,
    pub bridge_committed: u64,
    pub bridge_span_sum: u64,
    pub bridge_span_max: u64,
    pub bridge_colliders_sum: u64,
    pub bridge_wake_sum: u64,
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

/// Layer-2 profile-controller state (slot 0). While active it is the ONE
/// size authority: it owns `target_size` (the thermostat pulls toward the
/// moving setpoint), the static size brake is inert, and the per-round
/// MIX/COMP coin is driven by `pmix`. Best-effort contract: on saturation it
/// logs and holds the pinned lever rather than fighting the profile.
pub struct ProfState {
    pub phase: u8, // 1 expand, 2 hold, 3 compress, 4 done (hold at r2)
    pub pmix: f64,
    pub integ: f64,
    // Plant, all in GATES PER MOVE at full lever: ghat = drift if p_mix
    // were 1, shat = removal rate if p_mix were 0, dhat = the DISTURBANCE —
    // everything that changes size and is NOT the DB move (twists above all,
    // plus expansion moves and thermostat contractions). The controller does
    // not model twists; it measures their effect as the residual between
    // observed drift and DB-attributed drift, which is why it works at any
    // twist rate without being told what that rate is.
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
    base_mix: [u64; 4], // agn_hits, agn_misses, db_mix_added, db_mix_removed
    base_cmp: [u64; 4], // comp_hits, comp_misses, db_cmp_added, db_cmp_removed
}

/// The profile's target size at effective work `eff`: linear ramp to
/// r1*s_in over [0, n0], hold to n1, linear ramp to r2*s_in over [n1, n2],
/// r2*s_in thereafter.
pub fn prof_target(n: [f64; 3], r: [f64; 2], s_in: f64, eff: f64) -> f64 {
    let (n0, n1, n2) = (n[0], n[1], n[2]);
    let (r1, r2) = (r[0], r[1]);
    if eff <= 0.0 {
        s_in
    } else if eff < n0 {
        s_in * (1.0 + (r1 - 1.0) * (eff / n0))
    } else if eff < n1 {
        s_in * r1
    } else if eff < n2 {
        s_in * (r1 + (r2 - r1) * ((eff - n1) / (n2 - n1)))
    } else {
        s_in * r2
    }
}

/// Read-only reference store for class attribution (FROZEN_REF_DIR): opened
/// once, queried as a REGULAR store — only spelling LENGTHS are read, and
/// those are convention-independent.
fn reference_db() -> Option<&'static crate::replace::frozen::FrozenDb> {
    static REF: std::sync::OnceLock<Option<crate::replace::frozen::FrozenDb>> =
        std::sync::OnceLock::new();
    REF.get_or_init(|| {
        let dir = std::env::var("FROZEN_REF_DIR").ok()?;
        println!("[fmix] class-attribution reference store: {dir}");
        Some(crate::replace::frozen::FrozenDb::open(&dir, None))
    })
    .as_ref()
}

pub struct Mixer {
    pub arena: Arena,
    pub params: MixParams,
    pub counters: MixCounters,
    meta: Vec<Meta>,
    // merge-key -> linked node ids with that (target, wire-set). Kept exact by
    // the index_add/index_remove hooks in every splice; validated at verify
    // points via indexed_count.
    // FxHashMap: accessed by key only (never iterated), so the hasher choice
    // cannot leak into behavior.
    index: FxHashMap<u64, Vec<u32>>,
    indexed_count: usize,
    // Twist-g57 seam-solve memo: eng.solve(t, MAX_WORD) is a pure function of
    // the target perm `t`, and each twist round re-solves the same handful of
    // targets (both seams x wire-pair candidates x k-contexts x slides x
    // retries). Pure cache — deliberately NOT serialized by save_state and not
    // restored by load_state; a resumed run just starts cold. SmallVec: word
    // length is <= swap_words::MAX_WORD = 7, so hits never allocate.
    solve_memo: FxHashMap<u64, Option<smallvec::SmallVec<[u8; 7]>>>,
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
    // Sampled mode: the bitset universe is the K tracers rather than all
    // `anc_m` input gates, and `anc_tracers[t]` is the input-gate index that
    // bit `t` stands for (sorted). Empty in exact mode.
    anc_sampled: bool,
    anc_tracers: Vec<u32>,
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
    // Gate count of the window the current DB attempt sampled, for the
    // per-length breakdown.
    db_last_len: usize,
    // (seed id, its left neighbour when drawn). Window building floats gates,
    // so a FAILED attempt would otherwise leave the seed displaced -- and under
    // pool targeting the same stubborn gate is drawn repeatedly, which would
    // write a characteristic displacement into the circuit.
    db_seed_home: Option<(u32, u32)>,
    // Set for the duration of one COMP attempt drawn as g57-only.
    db_g57_only: bool,
    // Whether the CURRENT db_attempt drew the pair geometry: arms the
    // candidate reorder ban and the pair splice counter.
    db_pair_round: bool,
    // The LIVE slot-2 mode. params.db_mode is what the brake returns to.
    db_mode_cur: DbMode,
    // stable-ledger flow accounting (gates added/removed by Stable/StableGrow
    // splices). Deliberately NOT in MixCounters/state: a resume restarts the
    // ledger at zero, which just re-grants the slack once.
    stable_led_added: u64,
    stable_led_removed: u64,
    // per-success histogram of the store-minimal spelling length (the
    // converted permutation's operational complexity), index = gates (cap 31)
    dmin_success_hist: [u64; 32],
    // TRUE complexity class of each big-pool (M1/M2/M3...) conversion, read
    // from the REFERENCE store named by FROZEN_REF_DIR — the pool swap
    // replaced the small spellings in the live store, so the class is only
    // recoverable by re-querying the same key against the original.
    m123_class_hist: [u64; 32],
    // successful curated splices whose candidate list exceeded the bounded
    // contract (>20) — by construction exactly the M1/M2/M3 big-pool hits.
    bigpool_hits: u64,
    // SIZE LEDGER for --db-mode band-ledger: signed net gates this channel
    // has added. The size choice is skewed corrective whenever |ledger|
    // exceeds the slack, so the band's size variability is kept per splice
    // while the total is conserved in expectation.
    band_led: i64,
    // true for the current round when the effective mode came from BandLedger
    band_led_round: bool,
    // DB attempts / successful splices split by window GEOMETRY
    // ([0] = contiguous i.e. sequential, [1] = convex). Plain Mixer fields,
    // deliberately not in MixCounters: adding there breaks the .state format.
    geo_attempts: [u64; 2],
    geo_hits: [u64; 2],
    brake_on: bool,
    brake_mark_move: u64,
    brake_mark_size: usize,
    // Next move at which the pool is rebuilt.
    pool_scan_due: u64,
    // Layer-2 profile controller (params.prof_n[2] > 0); see ProfState.
    prof: Option<ProfState>,
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
    // ---- split stage (docs/FMIX_SPLIT_TWIST.md) ----
    // Live stage flag (params.split arms it; exits clear it), the
    // consecutive-4e-failure streak, and a one-move latch run() reads to stop
    // at the boundary under split_stop.
    split_on: bool,
    split_fail_streak: u32,
    split_ended: bool,
    // The stage ran to its boundary at some point in this run's history
    // (persisted): distinguishes "ended" from "never armed" on resume, and
    // zeroes the live split-twist dispatch for part 2 of a --split run.
    split_done: bool,
    // O(1)-samplable population indexes, maintained by index_add/index_remove:
    // every comp gate, and per target wire the bracket-eligible gates (comp,
    // or non-comp with exactly one control). pos vectors map id -> slot in its
    // list (NIL = absent) for O(1) swap-removal.
    comp_ids: Vec<u32>,
    comp_pos: Vec<u32>,
    wt_buckets: Vec<Vec<u32>>,
    wt_pos: Vec<u32>,
    // Wire canaries: flip monitors riding the material. A tap sits on `wire`
    // immediately right of `anchor`; taps re-anchor to the live left neighbor
    // when their anchor dies (evict_taps at every node-death site).
    taps: Vec<Tap>,
    tap_at: HashMap<u32, Vec<u32>>,
    taps_planted: bool,
    taps_reported: bool,
    // Approximate position ranks (id -> ordinal at last stamp; NIL = unknown),
    // restamped every RANK_EVERY moves. Heuristic only: half classification,
    // the midpoint-crossing counter and canary positions read it; correctness
    // never does.
    rank: Vec<u32>,
    rank_n: usize,
    rank_due: u64,
    // Min-dgen cross-shot pool (p_mincross): the K least-split lineages at
    // the last rebuild, consumed on draw so no single laggard is hammered.
    cross_pool: Vec<u32>,
    cross_pool_due: u64,
}

/// One wire canary (docs/FMIX_SPLIT_TWIST.md §5): counts how many twist spans
/// complemented the value carried on `wire` at its position. `orig_permille`
/// is the plant position as n/1000 of the circuit, the axis the stage report
/// buckets by.
struct Tap {
    anchor: u32,
    wire: u16,
    orig_permille: u16,
    flips: u64,
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
    // The layer-2 size profile finished its third phase (arrived at R2, or
    // reached N2). The contract is fulfilled, so the run ends there rather
    // than burning the remaining move budget outside any setpoint -- which,
    // at a twist rate whose growth the lever cannot offset, would actively
    // undo the compression leg.
    ProfileDone,
    // The split stage ended (g57 exhaustion or the failure limit) under
    // --split-stop: the run stops at the stage boundary (trial mode).
    SplitDone,
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
            self.dropped_neverfire,
            // Trailing fields are parsed with a zero default so pre-existing
            // .state files stay loadable; append here, never insert.
            self.tg_consumed,
            self.tg_emitted,
            // Split stage (2026-08-05).
            self.split_prims,
            self.split_hsplits,
            self.split_segs,
            self.split_joins,
            self.split_fails,
            self.split_xmid,
            self.tap_flips,
            self.split_span_sum,
            self.cross_pool_shots,
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
            // Appended after the twist-g57 work landed: absent in older state
            // files, so they default to zero rather than failing the resume.
            tg_consumed: next_u64(&mut it).unwrap_or(0),
            tg_emitted: next_u64(&mut it).unwrap_or(0),
            split_prims: next_u64(&mut it).unwrap_or(0),
            split_hsplits: next_u64(&mut it).unwrap_or(0),
            split_segs: next_u64(&mut it).unwrap_or(0),
            split_joins: next_u64(&mut it).unwrap_or(0),
            split_fails: next_u64(&mut it).unwrap_or(0),
            split_xmid: next_u64(&mut it).unwrap_or(0),
            tap_flips: next_u64(&mut it).unwrap_or(0),
            split_span_sum: next_u64(&mut it).unwrap_or(0),
            cross_pool_shots: next_u64(&mut it).unwrap_or(0),
            split_span_hist: [0u64; 20],
            tg_solves: 0,
            tg_solve_ns: 0,
            tg_slides: 0,
            tg_retries: 0,
            shuffles: 0,
            shuffle_moved: 0,
            shuffle_steps: 0,
            shuffle_ns: 0,
            choice_splices: 0,
            choice_multi: 0,
            choice_sum: 0,
            choice_bits_milli: 0,
            db_wide_poly: 0,
            len_attempts: Vec::new(),
            len_hits: Vec::new(),
            len_removed: Vec::new(),
            len_added: Vec::new(),
            len_span_skip: Vec::new(),
            len_deg_skip: Vec::new(),
            db_mix_added: 0,
            db_mix_removed: 0,
            db_cmp_added: 0,
            db_cmp_removed: 0,
            splice_sizes: Vec::new(),
            splice_sizes_curated: Vec::new(),
            pair_rounds: 0,
            pair_boxes_empty: 0,
            pair_scan_truncs: 0,
            pair_fused: 0,
            pair_splices: 0,
            pair_box_sum: 0,
            pair_box_max: 0,
            pair_dist_sum: 0,
            pair_dist_max: 0,
            pair_perm_skips: 0,
            bridge_rounds: 0,
            bridge_short: 0,
            bridge_refused: 0,
            bridge_probe_miss: 0,
            bridge_rollbacks: 0,
            bridge_half: 0,
            bridge_committed: 0,
            bridge_span_sum: 0,
            bridge_span_max: 0,
            bridge_colliders_sum: 0,
            bridge_wake_sum: 0,
            width_hist: [0u64; 16],
            tg_net_hist: [0u64; 8],
        })
    }
}

/// Version stamp for the resume file. The parameter set has changed repeatedly,
/// and silently reinterpreting fields would be worse than refusing to load.
/// A parsed per-gate ancestry sidecar (see `write_anc_sidecar`): the ancestry
/// universe plus one ancestor set per gate, in circuit order. This is the
/// cross-RUN transport format — unlike the state file, which continues the
/// same run, a sidecar lets a NEW run inherit ancestor lists via --anc-in, so
/// a phase boundary stops resetting the ancestry clock.
pub struct AncSidecar {
    pub sampled: bool,
    /// Universe size: the ORIGINAL input-gate count the set bits index (exact
    /// mode), kept for span normalisation in sampled mode too.
    pub m: usize,
    pub words: usize,
    /// Sampled mode: the original-input index each bit stands for (sorted).
    pub tracers: Vec<u32>,
    /// One set per gate, `words` u64s each.
    pub sets: Vec<Vec<u64>>,
}

// v2 (2026-08-05): the split-stage scalar line and the staps (canary) section
// (docs/FMIX_SPLIT_TWIST.md §7). The reader still accepts v1, defaulting both.
pub const STATE_VERSION: u32 = 2;

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
        let mut newid: FxHashMap<u32, u32> =
            FxHashMap::with_capacity_and_hasher(ids.len(), Default::default());
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
                DbMode::Stable => "stable",
                DbMode::StableGrow => "stable-grow",
                DbMode::StableLedger => "stable-ledger",
                DbMode::Same => "same",
                DbMode::BandLedger => "band-ledger",
                DbMode::BandShrink => "band-shrink",
                DbMode::BandGrow => "band-grow",
            }
        );
        let _ = writeln!(
            o,
            "brake {} {} {}",
            self.brake_on as u8, self.brake_mark_move, self.brake_mark_size
        );
        let _ = writeln!(o, "pool_scan_due {}", self.pool_scan_due);
        let _ = writeln!(o, "canary_failures {}", self.canary_failures);
        // Tri-state phase (0 = never armed, 1 = live, 2 = ended) so a resume
        // can tell "stage already ran" from "stage never requested", plus the
        // canary-report latch.
        let split_phase: u8 = if self.split_on { 1 } else if self.split_done { 2 } else { 0 };
        let _ = writeln!(
            o,
            "split {} {} {}",
            split_phase, self.split_fail_streak, self.taps_reported as u8
        );
        let _ = writeln!(o, "anc {} {}", self.anc_words, self.anc_m);
        let _ = writeln!(o, "counters {}", self.counters.to_line());

        let _ = writeln!(o, "gates {}", ids.len());
        for &id in &ids {
            let m = self.meta_of(id);
            gate_line(&mut o, self.arena.gate(id));
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
        for g in &self.original {
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
        // Optional trailing section (2026-08-03): the sampled tracer list,
        // serialized explicitly. An IMPORTED tracer set (--anc-in) is not a
        // function of (anc_m, K, anc_sample_seed), so the regeneration the
        // resume path used to rely on would silently remap the mask bits.
        // Old states lack the section (the reader falls back to
        // regeneration) and old binaries never read this far, so the version
        // deliberately stays 1.
        if self.anc_sampled {
            let _ = write!(o, "anctracers {}", self.anc_tracers.len());
            for t in &self.anc_tracers {
                let _ = write!(o, " {t}");
            }
            o.push('\n');
        }
        // Wire canaries: anchors are serialized as POSITIONS (the checkpoint
        // renumbers ids), re-anchored by ordinal at load.
        let _ = writeln!(o, "staps {}", self.taps.len());
        if !self.taps.is_empty() {
            let pos: HashMap<u32, usize> =
                ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
            for t in &self.taps {
                let p = pos.get(&t.anchor).copied().unwrap_or(0);
                let _ = writeln!(o, "{} {} {} {}", t.wire, t.orig_permille, t.flips, p);
            }
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
        // v1 is a strict prefix of v2 (split line and staps section absent),
        // so old states load with the split stage off and no canaries.
        if v != 1 && v != STATE_VERSION {
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
        let split_state: Option<(u8, u32, bool)> = if v >= 2 {
            let s = scalar(&mut lines, "split")?;
            let phase: u8 =
                s.first().and_then(|x| x.parse().ok()).ok_or_else(|| bad("split"))?;
            let streak = s.get(1).and_then(|x| x.parse().ok()).ok_or_else(|| bad("split streak"))?;
            // Third field absent in the earliest v2 files: default unreported.
            let reported = s.get(2).and_then(|x| x.parse::<u8>().ok()).unwrap_or(0) != 0;
            Some((phase, streak, reported))
        } else {
            None
        };
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
        // Optional trailing section: explicitly serialized tracers (states
        // written since 2026-08-03). Absent in older states, which regenerate
        // below instead. Read with one line of lookahead, since the v2 staps
        // section may follow (or replace) it.
        let mut pending = lines.next();
        let stored_tracers: Option<Vec<u32>> = match pending {
            Some(l) if l.starts_with("anctracers ") => {
                let mut it = l["anctracers ".len()..].split_whitespace();
                let k: usize =
                    it.next().and_then(|x| x.parse().ok()).ok_or_else(|| bad("anctracers count"))?;
                let tr: Vec<u32> = it.filter_map(|x| x.parse().ok()).collect();
                if tr.len() != k {
                    return Err(bad("anctracers list length mismatch"));
                }
                pending = lines.next();
                Some(tr)
            }
            _ => None,
        };
        // v2 canaries: (wire, orig_permille, flips, position at save).
        let mut staps_raw: Vec<(u16, u16, u64, usize)> = Vec::new();
        if let Some(l) = pending {
            if let Some(rest) = l.strip_prefix("staps ") {
                let k: usize = rest.trim().parse().map_err(|_| bad("staps count"))?;
                for _ in 0..k {
                    let tl = lines.next().ok_or_else(|| bad("short staps section"))?;
                    let f: Vec<&str> = tl.split_whitespace().collect();
                    if f.len() != 4 {
                        return Err(bad("staps line shape"));
                    }
                    staps_raw.push((
                        f[0].parse().map_err(|_| bad("stap wire"))?,
                        f[1].parse().map_err(|_| bad("stap orig"))?,
                        f[2].parse().map_err(|_| bad("stap flips"))?,
                        f[3].parse().map_err(|_| bad("stap pos"))?,
                    ));
                }
            }
        }

        // Build the mixer on the SAME id assignment the checkpoint renumbered
        // to: Arena::from_gates hands out 0..n-1 in order, which is what
        // save_state mapped the journal and pool onto.
        // Construct with ancestors OFF: the sizing guard in new_with_db reads
        // the CURRENT gate count, but ancestor sets are indexed by ORIGINAL
        // input gates and their width comes from the state file. A resumed 1.4M
        // gate circuit whose input was 20k would trip a guard meant for the
        // input. Restore the real setting and the recorded widths below.
        // Same reasoning applies to --anc-samples: tracers index the ORIGINAL
        // input gates, but new_with_db would draw them against the RESUMED gate
        // count. Construct with sampling off and regenerate below from the
        // restored anc_m.
        let ancestors_wanted = params.ancestors;
        let anc_samples_wanted = params.anc_samples;
        let anc_sample_seed = params.anc_sample_seed;
        let mut mx = Mixer::new_with_db(
            gates,
            num_wires,
            MixParams { ancestors: false, anc_samples: 0, ..params },
            db,
        );
        mx.params.ancestors = ancestors_wanted;
        mx.params.anc_samples = anc_samples_wanted;
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
        // Split stage, by recorded phase. 1 (live) continues the stage —
        // which still needs --split on the resume line per the repeat-your-
        // flags rule (warn loudly if it is missing, that is almost always a
        // mistake). 2 (ended) never re-arms: --split on the resume just means
        // "this is a split pipeline", part 2 continues. 0 (never armed) lets
        // an explicit --split start the stage fresh on the resumed circuit.
        // v1 states have no phase and take the constructor's arming.
        if let Some((phase, streak, reported)) = split_state {
            match phase {
                1 => {
                    mx.split_on = mx.params.split;
                    if !mx.params.split {
                        eprintln!(
                            "[fmix] WARNING: state file has a LIVE split stage but --split was not \
                             given — the stage stays OFF and part-2 moves run on unsplit material"
                        );
                    }
                }
                2 => {
                    mx.split_on = false;
                    mx.split_done = true;
                }
                _ => mx.split_on = mx.params.split,
            }
            mx.split_fail_streak = streak;
            mx.taps_reported = reported;
        }
        if !staps_raw.is_empty() {
            let ids = mx.arena.ids_in_order();
            for (wire, orig, flips, pos) in staps_raw {
                let anchor = ids[pos.min(ids.len() - 1)];
                mx.tap_at.entry(anchor).or_default().push(mx.taps.len() as u32);
                mx.taps.push(Tap { anchor, wire, orig_permille: orig, flips });
            }
            mx.taps_planted = true;
        }
        mx.anc = anc;
        mx.anc_words = anc_hdr[0].parse().unwrap_or(0);
        mx.anc_m = anc_hdr[1].parse().unwrap_or(0);
        // Tracer restoration. States since 2026-08-03 carry the list
        // explicitly (required for --anc-in imports, whose tracers are not a
        // function of anything this run knows); older states regenerate it
        // from (anc_m, K, anc_sample_seed) exactly as before. The assertions
        // catch a resume that changed K (or the seed, when that changes K's
        // rounding): the stored masks index the ORIGINAL tracer set and
        // cannot be reinterpreted.
        if let Some(tr) = stored_tracers {
            assert_eq!(
                mx.anc_words,
                tr.len().div_ceil(64),
                "state file anctracers/ancsets width mismatch ({} tracers, {} words)",
                tr.len(),
                mx.anc_words
            );
            if anc_samples_wanted > 0 {
                assert_eq!(
                    anc_samples_wanted.min(mx.anc_m),
                    tr.len(),
                    "resume changed --anc-samples ({} wanted, {} stored): the recorded masks \
                     index the ORIGINAL tracer set and cannot be reinterpreted",
                    anc_samples_wanted.min(mx.anc_m),
                    tr.len()
                );
            }
            mx.params.anc_samples = tr.len();
            mx.anc_sampled = true;
            mx.anc_tracers = tr;
        } else if anc_samples_wanted > 0 && mx.anc_m > 0 {
            let k = anc_samples_wanted.min(mx.anc_m);
            assert_eq!(
                mx.anc_words,
                k.div_ceil(64),
                "resume changed --anc-samples ({k} tracers now, {} words stored): the recorded \
                 masks index the ORIGINAL tracer set and cannot be reinterpreted",
                mx.anc_words
            );
            mx.anc_sampled = true;
            mx.anc_tracers = Self::pick_tracers(mx.anc_m, k, anc_sample_seed);
        }
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

    /// Write the per-gate ancestry sidecar: a header naming the universe, the
    /// tracer list in sampled mode, then one line per gate (current arena
    /// order — call after the final float so the order matches the written
    /// circuit) holding the gate's ancestor set as `anc_words` decimal u64s.
    /// Exact-mode implicit singletons are materialised, so the file is
    /// self-contained.
    pub fn write_anc_sidecar(&self, path: &str) -> std::io::Result<()> {
        use std::fmt::Write as _;
        assert!(
            self.anc_words > 0,
            "--anc-out needs ancestry armed (--ancestors, --anc-samples or --anc-in)"
        );
        let mut o = String::with_capacity(self.arena.len() * self.anc_words * 8);
        let _ = writeln!(
            o,
            "fmix-anc 1 {} m={} words={} gates={}",
            if self.anc_sampled { "sampled" } else { "exact" },
            self.anc_m,
            self.anc_words,
            self.arena.len()
        );
        if self.anc_sampled {
            let _ = write!(o, "tracers {}", self.anc_tracers.len());
            for t in &self.anc_tracers {
                let _ = write!(o, " {t}");
            }
            o.push('\n');
        }
        let mut bits = vec![0u64; self.anc_words];
        let mut cur = self.arena.head();
        while cur != NIL {
            bits.iter_mut().for_each(|w| *w = 0);
            self.anc_or_into(self.meta_of(cur).litter, &mut bits);
            let mut first = true;
            for w in &bits {
                if !first {
                    o.push(' ');
                }
                let _ = write!(o, "{w}");
                first = false;
            }
            o.push('\n');
            cur = self.arena.neighbor(cur, Dir::R);
        }
        std::fs::write(path, o)
    }

    /// Parse a sidecar written by `write_anc_sidecar`.
    pub fn read_anc_sidecar(path: &str) -> std::io::Result<AncSidecar> {
        let bad = |m: &str| std::io::Error::other(m.to_string());
        let text = std::fs::read_to_string(path)?;
        let mut lines = text.lines();
        let hdr = lines.next().ok_or_else(|| bad("empty ancestry sidecar"))?;
        let f: Vec<&str> = hdr.split_whitespace().collect();
        if f.len() != 6 || f[0] != "fmix-anc" || f[1] != "1" {
            return Err(bad("ancestry sidecar header: want `fmix-anc 1 <mode> m= words= gates=`"));
        }
        let sampled = match f[2] {
            "sampled" => true,
            "exact" => false,
            _ => return Err(bad("ancestry sidecar mode: want exact|sampled")),
        };
        let num = |s: &str, pre: &str| -> std::io::Result<usize> {
            s.strip_prefix(pre)
                .and_then(|x| x.parse().ok())
                .ok_or_else(|| bad(&format!("ancestry sidecar field {pre}")))
        };
        let m = num(f[3], "m=")?;
        let words = num(f[4], "words=")?;
        let gates = num(f[5], "gates=")?;
        let tracers: Vec<u32> = if sampled {
            let tl = lines.next().ok_or_else(|| bad("missing tracers line"))?;
            let mut it = tl.split_whitespace();
            if it.next() != Some("tracers") {
                return Err(bad("want `tracers K t0 t1 ...`"));
            }
            let k: usize =
                it.next().and_then(|x| x.parse().ok()).ok_or_else(|| bad("tracer count"))?;
            let v: Vec<u32> = it.filter_map(|x| x.parse().ok()).collect();
            if v.len() != k {
                return Err(bad("tracer list length mismatch"));
            }
            v
        } else {
            Vec::new()
        };
        let want_words = if sampled { tracers.len().div_ceil(64) } else { m.div_ceil(64) };
        if words != want_words {
            return Err(bad("ancestry sidecar words= inconsistent with its universe"));
        }
        let mut sets: Vec<Vec<u64>> = Vec::with_capacity(gates);
        for l in lines {
            let row: Vec<u64> = l.split_whitespace().filter_map(|x| x.parse().ok()).collect();
            if row.len() != words {
                return Err(bad("ancestry sidecar row width mismatch"));
            }
            sets.push(row);
        }
        if sets.len() != gates {
            return Err(bad("ancestry sidecar gate count mismatch"));
        }
        Ok(AncSidecar { sampled, m, words, tracers, sets })
    }

    /// Install imported ancestor lists as this run's INITIAL ancestry. Fresh
    /// runs only: the mixer must have been constructed with ancestry OFF
    /// (ancestors false, anc_samples 0), so input gates already sit on their
    /// singleton litters 0..n; this replaces the universe and attaches one
    /// imported set per input litter.
    pub fn import_ancestry(&mut self, sc: AncSidecar) {
        assert_eq!(
            sc.sets.len(),
            self.arena.len(),
            "--anc-in: sidecar has {} sets but the input circuit has {} gates",
            sc.sets.len(),
            self.arena.len()
        );
        assert!(
            self.anc_words == 0,
            "--anc-in replaces the ancestry universe; construct with --ancestors/--anc-samples off"
        );
        assert!(
            sc.sampled || sc.m <= 20_000,
            "--anc-in exact mode stores m={} bits per litter, past the small-input envelope \
             (re-run the producing chain with --anc-samples)",
            sc.m
        );
        self.anc_sampled = sc.sampled;
        self.anc_m = sc.m;
        self.anc_words = sc.words;
        self.anc_tracers = sc.tracers;
        self.anc.clear();
        for (i, bits) in sc.sets.into_iter().enumerate() {
            // Exact mode stores EVERY row, an all-zero one included: the
            // implicit-singleton rule reads a missing id < anc_m as {id}, and
            // after an import gate index i is unrelated to original-input
            // index i. Sampled mode keeps the missing-means-empty convention.
            if !self.anc_sampled || bits.iter().any(|&w| w != 0) {
                self.anc.insert(i as u64, bits);
            }
        }
        // Fresh litter ids must clear the exact-mode implicit-singleton range
        // [0, anc_m): a circuit smaller than the ORIGINAL input would
        // otherwise mint union ids that alias it.
        self.next_litter = self.next_litter.max(self.anc_m as u64);
        // Reporting reads these fields, not the universe params.
        self.params.ancestors = !self.anc_sampled;
        self.params.anc_samples = self.anc_tracers.len();
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
            // Bridge is its own slot and reaches the store independently of
            // p_db; p_pair is subordinate to p_db (it only picks the geometry
            // inside a slot-2 round) so it needs no entry here.
            || params.p_bridge > 0.0
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
        // Ancestry universe: SAMPLED (K tracer input gates, fixed cost, scales)
        // takes precedence over EXACT (all input gates, |input| bits/litter).
        let (anc_words0, anc_m0, anc_sampled0, anc_tracers0) = if params.anc_samples > 0 {
            let k = params.anc_samples.min(n);
            (k.div_ceil(64), n, true, Self::pick_tracers(n, k, params.anc_sample_seed))
        } else if params.ancestors {
            assert!(
                n <= 20_000,
                "--ancestors stores |input| bits per litter; {n} input gates is past                  the small-input envelope this instrument is for (use --anc-samples for large inputs)"
            );
            (n.div_ceil(64), n, false, Vec::new())
        } else {
            (0, 0, false, Vec::new())
        };
        // In sampled mode the tracers' own singleton sets are stored EXPLICITLY
        // (K entries), which is what lets `anc_or_into` drop the implicit
        // singleton rule: a non-tracer input gate then contributes nothing, so
        // untracked lineage costs no memory at all.
        let mut anc0: HashMap<u64, Vec<u64>> = HashMap::new();
        for (t, &gi) in anc_tracers0.iter().enumerate() {
            let mut bits = vec![0u64; anc_words0];
            bits[t / 64] |= 1u64 << (t % 64);
            anc0.insert(gi as u64, bits);
        }
        let mut index: FxHashMap<u64, Vec<u32>> = FxHashMap::default();
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
        let split_on0 = params.split;
        let mut mx = Mixer {
            arena: Arena::from_gates(gates.clone()),
            params,
            counters: MixCounters::default(),
            meta,
            index,
            indexed_count: n,
            solve_memo: FxHashMap::default(),
            journal: VecDeque::new(),
            tabu: VecDeque::new(),
            next_event: 1,
            next_litter: n as u64,
            anc: anc0,
            anc_words: anc_words0,
            anc_m: anc_m0,
            anc_sampled: anc_sampled0,
            anc_tracers: anc_tracers0,
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
            db_last_len: 0,
            db_seed_home: None,
            db_g57_only: false,
            db_pair_round: false,
            db_mode_cur: db_mode0,
            stable_led_added: 0,
            stable_led_removed: 0,
            dmin_success_hist: [0; 32],
            m123_class_hist: [0; 32],
            bigpool_hits: 0,
            band_led: 0,
            band_led_round: false,
            geo_attempts: [0; 2],
            geo_hits: [0; 2],
            brake_on: false,
            brake_mark_move: 0,
            brake_mark_size: 0,
            pool: Vec::new(),
            seed_from_pool: false,
            seed_fell_through: false,
            canary: VecDeque::new(),
            canary_failures: 0,
            pool_scan_due: 0,
            prof: None,
            split_on: split_on0,
            split_fail_streak: 0,
            split_ended: false,
            split_done: false,
            comp_ids: Vec::new(),
            comp_pos: Vec::new(),
            wt_buckets: Vec::new(),
            wt_pos: Vec::new(),
            taps: Vec::new(),
            tap_at: HashMap::new(),
            taps_planted: false,
            taps_reported: false,
            rank: Vec::new(),
            rank_n: 0,
            rank_due: 0,
            cross_pool: Vec::new(),
            cross_pool_due: 0,
        };
        mx.rebuild_side_index();
        // Collision-mask fast path (arena.rs): every wire a run touches stays
        // below num_wires, so num_wires <= 64 * MASK_WORDS (production: 128
        // wires = 2 words) means the mask side-array is authoritative for the
        // whole run and collides_ids never falls back to XGate::collides.
        debug_assert!(
            num_wires > 64 * super::arena::MASK_WORDS || mx.arena.masks_ok(),
            "collision masks poisoned despite num_wires = {num_wires}"
        );
        mx
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
        } else if !self.anc_sampled && (l as usize) < self.anc_m {
            // Exact mode only: bit `l` IS input gate `l`. In sampled mode the
            // bit space is the tracer set, tracer singletons are stored
            // explicitly, and a missing entry means "descends from no tracer".
            out[l as usize / 64] |= 1u64 << (l as usize % 64);
        }
    }

    /// Choose `k` distinct input-gate indices to trace, uniformly without
    /// replacement, from a DEDICATED rng: tracer choice must not perturb the
    /// mixing trajectory, so an exact-mode and a sampled-mode run with the same
    /// `--seed` follow the identical chain and can be compared gate for gate.
    /// Rejection sampling is O(k) expected for k << n and needs no O(n) buffer,
    /// which matters at production input sizes.
    fn pick_tracers(n: usize, k: usize, sample_seed: u64) -> Vec<u32> {
        let mut rng = StdRng::seed_from_u64(
            sample_seed ^ 0x7ACE_5EED_0000_0000 ^ ((n as u64) << 17) ^ ((k as u64) << 3),
        );
        let mut set: std::collections::HashSet<u32> = std::collections::HashSet::with_capacity(k);
        while set.len() < k {
            set.insert(rng.random_range(0..n) as u32);
        }
        let mut v: Vec<u32> = set.into_iter().collect();
        v.sort_unstable();
        v
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
        // An all-zero union needs no entry: fresh litter ids are always >= anc_m
        // (next_litter starts at the input count), so a missing entry can never
        // alias an implicit singleton and reads back as empty either way. In
        // sampled mode this is the main memory win -- only litters that actually
        // carry a tracer are stored, and most carry none.
        if bits.iter().any(|&w| w != 0) {
            self.anc.insert(l, bits);
        }
        l
    }

    /// Mean ancestor-set cardinality and mean normalised ancestor SPAN over
    /// live gates. Cardinality answers "what is a mixed gate made of"; span --
    /// (max index - min index) / (input - 1) -- answers "how far has input
    /// material travelled to meet". Both are immune to the ORIGIN_SYNTH erosion
    /// that makes odiff/oadj unreadable (see osyn=).
    fn anc_stats(&self) -> (f64, f64) {
        // Sampled mode reports through `tracer_report` instead: a sampled
        // popcount is not `anc` and a sampled index range is a biased `span`,
        // so leaving anc=/ancspan= at zero keeps those fields from silently
        // changing meaning (the mistake the g57=/shaped= split had to undo).
        if self.anc_words == 0 || self.anc_sampled {
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
        if self.anc_sampled {
            return self.tracer_report();
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

    /// Total gate x input-gate incidence: the sum over live gates of how many
    /// original gates each descends from. This is the one transport quantity
    /// both modes can report on the same footing -- exactly in exact mode, and
    /// in sampled mode as the Horvitz-Thompson estimate (every input has
    /// inclusion probability K/m, so the sampled sum scaled by m/K is unbiased).
    /// It is also the schedule-invariant measure: `anc` per gate is this divided
    /// by the gate count, which compression inflates.
    pub fn anc_incidence(&self) -> f64 {
        if self.anc_words == 0 {
            return 0.0;
        }
        let mut sum = 0u64;
        let mut bits = vec![0u64; self.anc_words];
        let mut cur = self.arena.head();
        while cur != NIL {
            bits.iter_mut().for_each(|w| *w = 0);
            self.anc_or_into(self.meta_of(cur).litter, &mut bits);
            sum += bits.iter().map(|w| w.count_ones() as u64).sum::<u64>();
            cur = self.arena.neighbor(cur, Dir::R);
        }
        if self.anc_sampled {
            sum as f64 * self.anc_m as f64 / self.anc_tracers.len().max(1) as f64
        } else {
            sum as f64
        }
    }

    /// Sampled-ancestry readout: for each traced input gate, the set of current
    /// gates descended from it, summarised three ways.
    ///
    /// - **`desc`** -- how many current gates descend from one input gate. This
    ///   is the per-input FANOUT, measured exactly for the traced gates, so its
    ///   mean over tracers is an unbiased estimate of the mean fanout over all
    ///   input gates (each input has inclusion probability K/m). Everything else
    ///   global follows from it: `incid = desc x m` is the total gate x input
    ///   incidence, and `anc = incid / size` is the mean ancestors per gate --
    ///   the exact-mode `anc`, estimated without storing |input| bits anywhere.
    /// - **`cov`** -- of `POS_BUCKETS` equal slices of the CURRENT circuit, the
    ///   fraction that hold at least one descendant. 1.0 means one input gate's
    ///   influence is present everywhere in the mixed circuit.
    /// - **`ent`** -- normalised entropy of the descendant positions over those
    ///   buckets. `cov` says how far the influence reaches, `ent` says how
    ///   evenly: cov can be 1.0 while the mass sits in one slice.
    ///
    /// `cov`/`ent` are the security-facing quantities and have no exact-mode
    /// analogue -- they ask directly whether an adversary can localise which
    /// part of the mixed circuit a given original gate went to. They are also
    /// natively samplable, unlike `span`, which a column sample can only
    /// underestimate.
    pub fn tracer_report(&self) -> String {
        const POS_BUCKETS: usize = 64;
        let k = self.anc_tracers.len();
        if k == 0 {
            return String::new();
        }
        let size = self.arena.len().max(1);
        let mut cnt = vec![0u64; k];
        let (mut lo, mut hi) = (vec![usize::MAX; k], vec![0usize; k]);
        let mut buckets = vec![0u32; k * POS_BUCKETS];
        let mut sampled_card_sum = 0u64;
        let mut carriers = 0u64;
        // BACKWARD direction: for each gate in the CURRENT circuit, how spread
        // out are its ancestors in the INPUT circuit? The forward measures
        // (reach/cov/ent) answer the mirror question -- where a given input
        // gate's descendants ended up -- and say nothing about whether an
        // output gate draws on a narrow band of the original or on all of it.
        //
        // A min/max RANGE cannot be sampled: with K of m tracers the sample's
        // extremes sit strictly inside the true ones, and the bias grows as the
        // ancestor count shrinks, so the statistic would mean different things
        // at different points in a run (which is why ancspan= is switched off
        // in sampled mode). Bucket occupancy and entropy degrade gracefully
        // instead, and the sample standard deviation is outright unbiased for
        // the population one. All three are capped by the ancestor count, so
        // they are only readable next to sampled_card.
        let mut aspan_cov = 0f64;
        let mut aspan_ent = 0f64;
        let mut aspan_sd = 0f64;
        let mut aspan_sd_n = 0u64;
        let m_in = self.anc_m.max(1);
        let mut in_buckets = [0u32; POS_BUCKETS];
        let mut bits = vec![0u64; self.anc_words];
        let mut cur = self.arena.head();
        let mut pos = 0usize;
        while cur != NIL {
            bits.iter_mut().for_each(|w| *w = 0);
            self.anc_or_into(self.meta_of(cur).litter, &mut bits);
            let b = (pos * POS_BUCKETS / size).min(POS_BUCKETS - 1);
            in_buckets.iter_mut().for_each(|c| *c = 0);
            let (mut ppos_sum, mut ppos_sq) = (0f64, 0f64);
            let mut card = 0u64;
            for (wi, &w) in bits.iter().enumerate() {
                let mut x = w;
                while x != 0 {
                    let t = wi * 64 + x.trailing_zeros() as usize;
                    if t < k {
                        cnt[t] += 1;
                        lo[t] = lo[t].min(pos);
                        hi[t] = hi[t].max(pos);
                        buckets[t * POS_BUCKETS + b] += 1;
                        // This ancestor's home in the INPUT circuit.
                        let gp = self.anc_tracers[t] as usize;
                        in_buckets[(gp * POS_BUCKETS / m_in).min(POS_BUCKETS - 1)] += 1;
                        ppos_sum += gp as f64;
                        ppos_sq += (gp as f64) * (gp as f64);
                        card += 1;
                    }
                    x &= x - 1;
                }
            }
            if card > 0 {
                carriers += 1;
                sampled_card_sum += card;
                let c = card as f64;
                aspan_cov +=
                    in_buckets.iter().filter(|&&x| x > 0).count() as f64 / POS_BUCKETS as f64;
                let h: f64 = in_buckets
                    .iter()
                    .filter(|&&x| x > 0)
                    .map(|&x| {
                        let q = x as f64 / c;
                        -q * q.log2()
                    })
                    .sum();
                aspan_ent += h / (POS_BUCKETS as f64).log2();
                if card >= 2 {
                    let mean = ppos_sum / c;
                    // Unbiased sample variance: with tracers drawn uniformly
                    // from the input, this estimates the spread of the gate's
                    // TRUE ancestor set, not just of the sampled part.
                    let var = (ppos_sq / c - mean * mean) * c / (c - 1.0);
                    aspan_sd += var.max(0.0).sqrt() / m_in as f64;
                    aspan_sd_n += 1;
                }
            }
            pos += 1;
            cur = self.arena.neighbor(cur, Dir::R);
        }
        let (mut reach, mut cov, mut ent) = (Vec::new(), Vec::new(), Vec::new());
        for t in 0..k {
            if cnt[t] == 0 {
                reach.push(0.0);
                cov.push(0.0);
                ent.push(0.0);
                continue;
            }
            reach.push((hi[t] - lo[t] + 1) as f64 / size as f64);
            let row = &buckets[t * POS_BUCKETS..(t + 1) * POS_BUCKETS];
            cov.push(row.iter().filter(|&&c| c > 0).count() as f64 / POS_BUCKETS as f64);
            let n_t = cnt[t] as f64;
            let h: f64 = row
                .iter()
                .filter(|&&c| c > 0)
                .map(|&c| {
                    let q = c as f64 / n_t;
                    -q * q.log2()
                })
                .sum();
            ent.push(h / (POS_BUCKETS as f64).log2());
        }
        let meanf = |v: &[f64]| if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 };
        // desc: exact per-tracer fanout, so its mean estimates mean fanout/input.
        let desc = cnt.iter().sum::<u64>() as f64 / k as f64;
        let incid = desc * self.anc_m as f64;
        let anc_all = incid / size as f64;
        // Reported for honesty about the sample's resolution: gates whose
        // ancestry misses every tracer look empty, and at small K most do.
        let hit = carriers as f64 / size as f64;
        format!(
            "[fmix] tracers: K={} of m={} | desc mean={:.0} [{}] | cov mean={:.3} ent mean={:.3} reach mean={:.3} | est anc={:.1} incid={:.3e} | carriers={:.3} sampled_card={:.2} | ancspan cov={:.3} ent={:.3} sd={:.3}",
            k,
            self.anc_m,
            desc,
            Self::log_hist(&cnt),
            meanf(&cov),
            meanf(&ent),
            meanf(&reach),
            anc_all,
            incid,
            hit,
            if carriers > 0 { sampled_card_sum as f64 / carriers as f64 } else { 0.0 },
            if carriers > 0 { aspan_cov / carriers as f64 } else { 0.0 },
            if carriers > 0 { aspan_ent / carriers as f64 } else { 0.0 },
            if aspan_sd_n > 0 { aspan_sd / aspan_sd_n as f64 } else { 0.0 },
        )
    }

    /// JOINT census of re-encoding depth against ancestry: for each generation
    /// band, how many gates are in it and what their mean ancestor count and
    /// mean ancestor span are.
    ///
    /// `anc` and `dgen` have only ever been reported as separate marginals, which
    /// cannot answer the question that matters: does depth BUY ancestry? A
    /// protocol where anc rises steeply with generation is compounding -- each
    /// re-encoding folds in genuinely new lineage. One where anc is flat across
    /// generations is re-spelling the same material over and over, and its depth
    /// counter is measuring effort rather than mixing. Different mode schedules
    /// can produce the same mean anc with very different shapes here.
    ///
    /// `r` is the Pearson correlation of (dgen, anc) over gates with a real
    /// generation. GEN_FRESH is a sentinel (born-random material: twist
    /// brackets, insert pairs), not a large number, so it is excluded from `r`
    /// and reported as its own band. In sampled mode the per-gate ancestor count
    /// is scaled by m/K, so the bands are comparable to exact mode; `span` is
    /// omitted there (a column sample can only underestimate it).
    pub fn gen_anc_report(&self) -> String {
        if self.anc_words == 0 {
            return String::new();
        }
        // Upper bound of each band; the last band is GEN_FRESH alone.
        const EDGES: [u32; 9] = [0, 1, 2, 4, 8, 16, 32, 64, u32::MAX - 1];
        const NAMES: [&str; 9] =
            ["g0", "g1", "g2", "g3-4", "g5-8", "g9-16", "g17-32", "g33-64", "g65+"];
        let nb = EDGES.len();
        let mut n = vec![0u64; nb + 1];
        let mut anc_sum = vec![0f64; nb + 1];
        let mut span_sum = vec![0f64; nb + 1];
        // Pearson accumulators over real-generation gates.
        let (mut cn, mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
        let scale = if self.anc_sampled {
            self.anc_m as f64 / self.anc_tracers.len().max(1) as f64
        } else {
            1.0
        };
        let mut bits = vec![0u64; self.anc_words];
        let mut cur = self.arena.head();
        while cur != NIL {
            bits.iter_mut().for_each(|w| *w = 0);
            self.anc_or_into(self.meta_of(cur).litter, &mut bits);
            let mut card = 0u64;
            let (mut lo, mut hi) = (usize::MAX, 0usize);
            for (wi, &w) in bits.iter().enumerate() {
                let mut x = w;
                while x != 0 {
                    let idx = wi * 64 + x.trailing_zeros() as usize;
                    card += 1;
                    lo = lo.min(idx);
                    hi = hi.max(idx);
                    x &= x - 1;
                }
            }
            let g = self.meta_of(cur).dgen;
            let bi = if g == GEN_FRESH {
                nb // the born-random band
            } else {
                EDGES.iter().position(|&e| g <= e).unwrap_or(nb - 1)
            };
            let a = card as f64 * scale;
            n[bi] += 1;
            anc_sum[bi] += a;
            if !self.anc_sampled && card > 0 {
                span_sum[bi] += (hi - lo) as f64;
            }
            if g != GEN_FRESH {
                let (x, y) = (g as f64, a);
                cn += 1.0;
                sx += x;
                sy += y;
                sxx += x * x;
                syy += y * y;
                sxy += x * y;
            }
            cur = self.arena.neighbor(cur, Dir::R);
        }
        let r = {
            let num = cn * sxy - sx * sy;
            let den = ((cn * sxx - sx * sx) * (cn * syy - sy * sy)).sqrt();
            if den > 0.0 { num / den } else { 0.0 }
        };
        let mut parts: Vec<String> = Vec::new();
        for bi in 0..=nb {
            if n[bi] == 0 {
                continue;
            }
            let label = if bi == nb { "FRESH" } else { NAMES[bi] };
            let cnt = n[bi] as f64;
            if self.anc_sampled {
                parts.push(format!("{label}:n={} anc={:.1}", n[bi], anc_sum[bi] / cnt));
            } else {
                parts.push(format!(
                    "{label}:n={} anc={:.1} span={:.0}",
                    n[bi],
                    anc_sum[bi] / cnt,
                    span_sum[bi] / cnt
                ));
            }
        }
        format!("[fmix] gen-anc: r={r:.3} (n={:.0} real-gen gates) | {}", cn, parts.join(" | "))
    }

    /// Drop ancestor sets for litters with no live gates. Without this the map
    /// grows with every splice for the whole run; with it, it is bounded by the
    /// live litter count (plus the litters restorable journal entries hold).
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
        // A restorable journal entry will put its parents' litters back, and
        // since a cross relabels ALL of its outputs to the union litter, those
        // pre-cross litters may have no live gate left. Dropping their sets
        // would make an undo restore ancestry-less litters silently. Dead
        // entries (any piece touched) can never be restored, so they hold
        // nothing.
        for e in self.journal.iter() {
            if e.after
                .iter()
                .all(|&(id, st)| self.arena.is_linked(id) && self.arena.stamp(id) == st)
            {
                live.insert(e.litters[0]);
                live.insert(e.litters[1]);
            }
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
        // `tabu` is only push_back'd with strictly increasing events
        // (fresh_event) and popped from the front, so it stays sorted by event:
        // a binary search finds the only possible match.
        event != 0
            && self
                .tabu
                .binary_search_by_key(&event, |&(ev, _)| ev)
                .is_ok_and(|i| self.tabu[i].1 + self.params.tabu_moves > self.moves_done)
    }

    // ---- merge-partner index maintenance ----

    fn index_add(&mut self, id: u32) {
        let k = key_of(self.arena.gate(id));
        self.index.entry(k).or_default().push(id);
        self.indexed_count += 1;
        self.side_add(id);
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
        self.side_remove(id);
    }

    // ---- split-stage population indexes ----
    //
    // Piggyback on the merge-index hooks, which every splice and in-place
    // rewrite already calls in remove-old / add-new order, so both lists stay
    // exact without new call sites. comp_ids: every comp gate (step-1
    // sampling and the exhaustion exit). wt_buckets[w]: bracket-eligible
    // gates targeting w — comp of any width (split on selection) or non-comp
    // with exactly one control (absorbs directly). pos vectors give O(1)
    // swap-removal; NIL = absent.

    fn side_add(&mut self, id: u32) {
        let idu = id as usize;
        if self.comp_pos.len() <= idu {
            self.comp_pos.resize(idu + 1, NIL);
            self.wt_pos.resize(idu + 1, NIL);
        }
        let g = self.arena.gate(id);
        let (comp, elig, t) = (g.comp, g.comp || g.ctrls.len() == 1, g.target as usize);
        if comp {
            self.comp_pos[idu] = self.comp_ids.len() as u32;
            self.comp_ids.push(id);
        }
        if elig {
            let b = &mut self.wt_buckets[t];
            self.wt_pos[idu] = b.len() as u32;
            b.push(id);
        }
    }

    fn side_remove(&mut self, id: u32) {
        let idu = id as usize;
        let p = self.comp_pos[idu];
        if p != NIL {
            self.comp_ids.swap_remove(p as usize);
            if let Some(&moved) = self.comp_ids.get(p as usize) {
                self.comp_pos[moved as usize] = p;
            }
            self.comp_pos[idu] = NIL;
        }
        let q = self.wt_pos[idu];
        if q != NIL {
            let t = self.arena.gate(id).target as usize;
            let b = &mut self.wt_buckets[t];
            b.swap_remove(q as usize);
            if let Some(&moved) = b.get(q as usize) {
                self.wt_pos[moved as usize] = q;
            }
            self.wt_pos[idu] = NIL;
        }
    }

    // Bulk (re)build: construction and resume, where the merge index is built
    // without going through index_add.
    fn rebuild_side_index(&mut self) {
        self.comp_ids.clear();
        self.comp_pos = vec![NIL; self.arena.capacity()];
        self.wt_buckets = vec![Vec::new(); self.num_wires];
        self.wt_pos = vec![NIL; self.arena.capacity()];
        for id in self.arena.ids_in_order() {
            self.side_add(id);
        }
    }

    // ---- the chain ----


    // Layer-2 overlay: the DB knobs for the round's LIVE mode. With the mode
    // overlay off, db_mode_cur is fixed and these return the base params, so a
    // single-mode run behaves exactly as before. Only COMP-mode reads the
    // overrides, and only when they are set.
    /// Window length for the round's live mode AND the geometry just drawn.
    /// Precedence, most specific first: mode+geometry (`s_db_comp_ctg` /
    /// `s_db_ctg`) -> mode (`s_db_comp`) -> base (`s_db`). A 0 anywhere means
    /// "not set, fall through".
    // Every one of these delegates to MixParams::db_knobs, so the resolution
    // rules exist in exactly ONE place. They are called per DB round, but the
    // work is a handful of Option::or on Copy types -- nothing next to the
    // milliseconds a canonicalization costs.
    fn active_s_db(&self, geo: DbSample) -> usize {
        let k = self.params.db_knobs(self.db_mode_cur);
        match geo {
            DbSample::Convex => k.s_db_cvx,
            DbSample::Contiguous => k.s_db_ctg,
            // A pair window is exactly the seed plus its partner; the length
            // knobs do not apply. Bridge windows are likewise always 2 (and
            // never drawn by the geometry coin — the tag only reaches here
            // through record paths).
            DbSample::Pair | DbSample::Bridge => 2,
        }
    }
    fn active_prefixes(&self) -> bool {
        self.params.db_knobs(self.db_mode_cur).prefixes
    }
    fn active_p_convex(&self) -> f64 {
        self.params.db_knobs(self.db_mode_cur).p_convex
    }
    fn active_p_mingen(&self) -> f64 {
        self.params.db_knobs(self.db_mode_cur).p_mingen
    }

    // Layer-2 mode overlay (slot 0): pick this round's DB mode by coin when
    // armed (p_mix >= 0) -- MIX-DB with probability p_mix, else COMP-DB. This is
    // the "set parameter" rule of the slot-0 condition engine, independent of
    // the size brake and the thermostat. Off (p_mix < 0) leaves db_mode_cur as
    // the fixed db_mode (possibly steered by the size brake).
    fn apply_mode_overlay(&mut self) {
        if self.params.p_mix < 0.0 {
            return;
        }
        // Same rule as prof_tick: the MIX side is the configured re-encode
        // mode, so an explicit --db-mode survives the overlay.
        let mix_side = match self.params.db_mode {
            DbMode::Compressing => DbMode::Mix,
            m => m,
        };
        self.db_mode_cur = if self.rng.random_bool(self.params.p_mix.clamp(0.0, 1.0)) {
            mix_side
        } else {
            DbMode::Compressing
        };
    }

    /// Record the branching factor of one successful splice's selection.
    fn note_choice(&mut self, k: usize) {
        if k == 0 {
            return;
        }
        self.counters.choice_splices += 1;
        self.counters.choice_sum += k as u64;
        if k > 1 {
            self.counters.choice_multi += 1;
            self.counters.choice_bits_milli += ((k as f64).log2() * 1000.0).round() as u64;
        }
    }

    fn prof_init(&mut self) {
        let s_in = self.original.len() as f64;
        // Phase priors until the first plant estimates arrive: full MIX for
        // the expansion leg (matching pay-random's strong up-lever).
        self.prof = Some(ProfState {
            phase: 1,
            // Moderate prior: the plant is unknown for exactly one interval,
            // and a hot prior overshoots the expansion ramp before the first
            // update can react (measured: p_mix=1 grows ~1.7 gates/move on a
            // fresh 100k gadget, ~3x the steepest ramp anyone asks for).
            pmix: 0.5,
            integ: 0.0,
            ghat: 0.0,
            shat: 0.0,
            dhat: 0.0,
            eff: 0.0,
            // First update comes early (quarter cadence) so the plant is
            // identified before much work is spent on a guess.
            next_eff: self.params.prof_cadence_eff.max(0.05) * 0.25,
            sat: 0,
            s_in,
            base_moves: self.moves_done,
            base_size: s_in,
            base_pmix: 0.5,
            base_mix: [0; 4],
            base_cmp: [0; 4],
        });
        self.prof_snapshot();
        eprintln!(
            "[fmix] profile ON: n={:?} r={:?} s_in={} cadence_eff={} deadband={} dp_max={}",
            self.params.prof_n,
            self.params.prof_r,
            s_in,
            self.params.prof_cadence_eff,
            self.params.prof_deadband,
            self.params.prof_dp_max
        );
    }

    fn prof_snapshot(&mut self) {
        let c = &self.counters;
        let mix = [c.db_agn_hits, c.db_agn_misses, c.db_mix_added, c.db_mix_removed];
        let cmp = [c.db_comp_hits, c.db_comp_misses, c.db_cmp_added, c.db_cmp_removed];
        let size = self.arena.len() as f64;
        let moves = self.moves_done;
        if let Some(p) = self.prof.as_mut() {
            p.base_moves = moves;
            p.base_size = size;
            p.base_pmix = p.pmix;
            p.base_mix = mix;
            p.base_cmp = cmp;
        }
    }

    /// Per-round profile bookkeeping: accumulate effective work, run the
    /// controller at its cadence, and flip the MIX/COMP coin at the
    /// controller's current lever.
    fn prof_tick(&mut self) {
        let size = self.arena.len().max(1) as f64;
        let (due, pmix) = {
            let p = self.prof.as_mut().expect("prof_tick with prof off");
            p.eff += 1.0 / size;
            (p.eff >= p.next_eff, p.pmix)
        };
        if due {
            self.prof_update();
        }
        let pm = if due { self.prof.as_ref().unwrap().pmix } else { pmix };
        // The MIX side of the coin is the CONFIGURED re-encode mode, not a
        // hardcoded Mix: --db-mode stable/stable-grow/stable-ledger must
        // survive the profile overlay (it used to be stomped here every
        // round, silently running plain Mix). Runs without an explicit
        // --db-mode are unchanged (their configured mode IS Mix).
        let mix_side = match self.params.db_mode {
            DbMode::Compressing => DbMode::Mix,
            m => m,
        };
        self.db_mode_cur =
            if self.rng.random_bool(pm.clamp(0.0, 1.0)) { mix_side } else { DbMode::Compressing };
    }

    /// One controller update (every prof_cadence_eff of effective work):
    /// refresh the plant estimates from the interval's per-mode counters,
    /// advance the phase, retarget the thermostat, and move the lever by
    /// feed-forward inversion plus a small integral correction — deadbanded,
    /// rate-limited, clamped, with saturation logged (best-effort contract).
    fn prof_update(&mut self) {
        let s = self.arena.len().max(1) as f64;
        let c = &self.counters;
        let mix_now = [c.db_agn_hits, c.db_agn_misses, c.db_mix_added, c.db_mix_removed];
        let cmp_now = [c.db_comp_hits, c.db_comp_misses, c.db_cmp_added, c.db_cmp_removed];
        let n = self.params.prof_n;
        let r = self.params.prof_r;
        let (cad, dead, dp_max, ew, ki) = (
            self.params.prof_cadence_eff.max(0.05),
            self.params.prof_deadband,
            self.params.prof_dp_max,
            self.params.prof_ewma,
            self.params.prof_ki,
        );
        if let Some(p) = self.prof.as_mut() {
            // ---- Plant identification over the interval, in gates/move ----
            // The lever splits rounds into MIX (w.p. p) and COMP (w.p. 1-p),
            // so per-move contributions are normalised by the p that was
            // actually in force; an arm that saw too little of the interval
            // keeps its previous estimate rather than dividing by ~0.
            let dmoves = (self.moves_done - p.base_moves).max(1) as f64;
            let p_used = p.base_pmix;
            let net_mix =
                (mix_now[2] - p.base_mix[2]) as f64 - (mix_now[3] - p.base_mix[3]) as f64;
            let net_cmp =
                (cmp_now[3] - p.base_cmp[3]) as f64 - (cmp_now[2] - p.base_cmp[2]) as f64;
            if p_used > 0.05 {
                let g_new = net_mix / (dmoves * p_used);
                p.ghat = if p.ghat == 0.0 { g_new } else { (1.0 - ew) * p.ghat + ew * g_new };
            }
            if p_used < 0.95 {
                let s_new = net_cmp / (dmoves * (1.0 - p_used));
                p.shat = if p.shat == 0.0 { s_new } else { (1.0 - ew) * p.shat + ew * s_new };
            }
            // The DISTURBANCE: observed total drift minus what the DB move
            // accounts for. Twists live here (they add gates at a rate the
            // controller never models), along with expansion moves and
            // thermostat contractions. Estimating it is what lets a profile
            // run at any twist rate.
            let v_obs = (s - p.base_size) / dmoves;
            let v_db = (net_mix - net_cmp) / dmoves;
            let d_new = v_obs - v_db;
            p.dhat = if p.dhat == 0.0 { d_new } else { (1.0 - ew) * p.dhat + ew * d_new };
            // Phase machine (best-effort: eff marks OR size arrival).
            let old_phase = p.phase;
            match p.phase {
                1 if p.eff >= n[0] || s >= 0.99 * r[0] * p.s_in => p.phase = 2,
                2 if p.eff >= n[1] => p.phase = 3,
                3 if p.eff >= n[2] || s <= 1.01 * r[1] * p.s_in => p.phase = 4,
                _ => {}
            }
            if p.phase != old_phase {
                p.integ = 0.0;
                p.sat = 0;
                eprintln!(
                    "[fmix] profile: phase {} -> {} at eff={:.2} size={}",
                    old_phase, p.phase, p.eff, s as usize
                );
                // Exact-point recovery: FMIX_STOP_AT_PHASE=<k> finishes the
                // run cleanly the moment the schedule enters phase k. A
                // deterministic replay (same seed/input/flags) stopped this
                // way recovers the ORIGINAL run's circuit at the leg boundary
                // with zero overshoot — the transition is detected at the
                // same controller update in both runs.
                static STOP_AT: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
                let target = STOP_AT.get_or_init(|| {
                    std::env::var("FMIX_STOP_AT_PHASE").ok().and_then(|v| v.parse().ok())
                });
                if *target == Some(p.phase as u32) {
                    println!(
                        "[fmix] FMIX_STOP_AT_PHASE={}: stopping cleanly at the phase boundary (eff={:.2} size={})",
                        p.phase, p.eff, s as usize
                    );
                    self.stop_requested = true;
                }
            }
            // Feasibility diagnosis for the compression leg. With the lever
            // fully down the best available drift is dhat - shat; when the
            // disturbance (twists, chiefly) exceeds what COMP can remove, the
            // circuit grows no matter what the controller does. Say so once,
            // in those terms, instead of leaving a silent saturation.
            if p.phase == 3 && p.dhat > p.shat && p.sat == 4 {
                eprintln!(
                    "[fmix] profile: COMPRESSION INFEASIBLE — disturbance {:+.4} gates/move (twists et al.) exceeds max COMP removal {:.4}; the lever is pinned at 0 and the circuit still grows. Lower the twist rate or relax R2.",
                    p.dhat, p.shat
                );
            }
            let s_star = prof_target(n, r, p.s_in, p.eff);
            let s_star_next = prof_target(n, r, p.s_in, p.eff + cad);
            let err = (s - s_star) / s_star.max(1.0);
            if err.abs() >= dead {
                p.integ = (p.integ - ki * err).clamp(-0.3, 0.3);
                // Feed-forward: solve p*ghat - (1-p)*shat + dhat = v_star for
                // p, where v_star is the slope that lands on the setpoint one
                // cadence ahead. The disturbance enters as a constant offset,
                // so a twist-heavy run simply gets a lower p_mix.
                let horizon = (cad * s).max(1.0);
                let v_star = (s_star_next - s) / horizon;
                let denom = p.ghat + p.shat;
                let mut want = if denom.abs() > 1e-9 {
                    (v_star - p.dhat + p.shat) / denom
                } else {
                    p.pmix
                };
                want += p.integ;
                // Rate limit, with an escape hatch: far from the profile the
                // lever may move freely (a 0.1/step crawl cannot catch a ramp
                // that is 6 eff units long), near it the limit binds and the
                // loop stays gentle.
                let far = err.abs() > 4.0 * dead.max(1e-6);
                let lim = if far { 1.0 } else { dp_max };
                let dp = (want - p.pmix).clamp(-lim, lim);
                p.pmix = (p.pmix + dp).clamp(0.0, 1.0);
            }
            // Saturation: pinned lever while clearly behind the profile.
            if (p.pmix >= 0.999 && err < -0.10) || (p.pmix <= 0.001 && err > 0.10) {
                p.sat += 1;
                if p.sat == 5 {
                    eprintln!(
                        "[fmix] profile: SATURATED (phase {} pmix={:.3} size={} S*={:.0}) — best-effort, continuing pinned",
                        p.phase, p.pmix, s as usize, s_star
                    );
                }
            } else {
                p.sat = 0;
            }
            p.next_eff += cad;
            // The thermostat is conscripted: pull toward the moving setpoint.
            self.params.target_size = s_star.max(2.0) as usize;
        }
        self.prof_snapshot();
    }

    // One first-class twist round. The twist is always from the swap family;
    // `twist_move` rolls the (alpha, beta) negation coins internally, giving
    // swap 1/4, swap+negate-one 1/2, swap+negate-both 1/4. The legacy
    // w_twist_* weights are retired (accepted-but-ignored on the CLI).
    fn twist_round(&mut self) {
        // Layer-1 dispatch: the split twist first (forced while the stage is
        // live), then the existing g57/swap-family choice. After a --split
        // run's boundary the live dispatch is ZERO (docs §3) — part 2 runs no
        // further split twists even if --p-split-twist was set; the CLI value
        // is the standalone (no --split) layer-1 mode.
        let p_st = if self.params.split && self.split_on {
            1.0
        } else if self.params.split && self.split_done {
            0.0
        } else {
            self.params.p_split_twist
        };
        if p_st > 0.0 && self.rng.random_bool(p_st.clamp(0.0, 1.0)) {
            self.split_twist_move();
            return;
        }
        if self.params.twist_g57 {
            self.twist_move_g57();
        } else {
            self.twist_move();
        }
    }

    // ---- the split twist (docs/FMIX_SPLIT_TWIST.md) ----
    //
    // One move: split a random g57 into its presplit pair, then with
    // probability p_join wrap an ABSORBED pure-NOT twist on the g57's target
    // wire between the split's 1-control piece and a bracket found across the
    // circuit (splitting the bracket too when it is a g57, and force-splitting
    // every g57 the segment conjugates), and finish with one ordinary cross
    // shot from the 2-control piece. Every sub-rewrite is function-preserving
    // on its own: presplit is exact, and the twist is the identity
    //   g1' . S' . h1' = g1 . X(w) . S' . X(w) . h1 = g1 . S . h1
    // where ' flips a bracket's control polarity (the absorbed X: a gate
    // targeting w commutes with X(w) and composes into a single gate) and S'
    // flips every w-READING pin in the open segment (gates targeting w are
    // invariant).

    fn split_twist_move(&mut self) {
        // Ranks drive the bracket draw, so refresh them on growth too: the
        // stage roughly doubles the circuit in a few thousand moves, far
        // inside the move cadence.
        if self.moves_done >= self.rank_due || self.arena.len() > self.rank_n + self.rank_n / 4 {
            self.restamp_ranks();
        }
        if !self.taps_planted {
            self.plant_taps();
        }
        // 1. A uniformly random g57; none anywhere = exit A. Outside a live
        // stage (standalone --p-split-twist dispatch) an empty pool is just a
        // spent round — there is no stage to end.
        if self.comp_ids.is_empty() {
            if self.split_on {
                self.end_split_stage("g57 pool exhausted");
            } else {
                self.counters.twist_skips += 1;
            }
            return;
        }
        let g_id = self.comp_ids[self.rng.random_range(0..self.comp_ids.len())];
        let w = self.arena.gate(g_id).target;
        // Twist direction (v3, 2026-08-05): drawn with probability
        // proportional to the circuit length REMAINING on each side of g, so
        // a side is picked exactly as rarely as it is short — the fix for
        // the 0-5% span spike that the own-stored-direction rule produced on
        // edge-adjacent primaries (a tiny span now needs a short side AND
        // the proportional coin to pick it: squared suppression). Stored
        // direction is only the fallback for an unstamped primary.
        let g_rank = self.rank_of(g_id);
        let g_dir = if g_rank != NIL && self.rank_n > 0 {
            let p_right = (self.rank_n as f64 - g_rank as f64) / self.rank_n as f64;
            if self.rng.random_bool(p_right.clamp(0.0, 1.0)) { Dir::R } else { Dir::L }
        } else {
            self.meta_of(g_id).dir
        };
        // 2. Split it. g1 = the 1-control rung, g2 = the widest.
        let (g1, g2) = self.split_g57(g_id);
        self.counters.split_prims += 1;
        // 3. The join coin: tails ends the move after the bare split.
        if self.rng.random_bool(1.0 - self.params.p_join.clamp(0.0, 1.0)) {
            self.split_report_line(None);
            return;
        }
        // 4. The bracket draw (directional length-biased; reach_k = 0 keeps
        // the original cascade as the A/B arm), then the twist.
        let mut span = None;
        let picked = if self.params.split_reach_k == 0 {
            self.pick_bracket_cascade(w, g1, g_rank)
        } else {
            self.pick_bracket(w, g1, g_rank, g_dir)
        };
        match picked {
            None => {
                self.counters.split_fails += 1;
                self.split_fail_streak += 1;
                if self.split_on && self.split_fail_streak >= self.params.split_fail_limit {
                    self.end_split_stage("failure limit");
                }
            }
            Some((h_id, h_comp, crossed)) => {
                let h1 = if h_comp {
                    self.counters.split_hsplits += 1;
                    self.split_g57(h_id).0
                } else {
                    h_id
                };
                let s = self.apply_not_twist(g1, h1, w);
                self.counters.split_joins += 1;
                self.counters.split_span_sum += s as u64;
                let frac20 = (s * 20) / self.arena.len().max(1);
                self.counters.split_span_hist[frac20.min(19)] += 1;
                span = Some(s);
                if crossed {
                    self.counters.split_xmid += 1;
                }
                self.split_fail_streak = 0;
            }
        }
        // 6. One ordinary cross shot from g2, twist outcome notwithstanding.
        if self.arena.is_linked(g2) {
            self.cross_move_on(g2);
        }
        self.split_report_line(span);
    }

    /// Split a g57 in place by the randomized first-failing-literal presplit
    /// (the literal shuffle IS the design's `r` bit). Pieces stay put — no
    /// birth transport: the 1-control piece must sit where the bracket forms,
    /// and the widest piece's transport is the move's cross. First piece
    /// draws a fair direction, the rest alternate (the sibling convention).
    /// Returns (first piece, last piece).
    fn split_g57(&mut self, id: u32) -> (u32, u32) {
        let g = self.arena.gate(id).clone();
        debug_assert!(g.comp, "split_g57 on a non-comp gate");
        let pieces = rules::presplit(&g, &mut self.rng);
        if self.params.local_verify {
            assert!(
                rules::verify_rewrite(std::slice::from_ref(&g), &pieces),
                "split-twist presplit verification failed: {g:?} -> {pieces:?}"
            );
        }
        let pm = self.meta_of(id);
        let ev = self.fresh_event();
        for p in &pieces {
            self.counters.width_hist[p.width().min(15)] += 1;
        }
        let d0 = self.rand_dir();
        let ids = self.splice_replace_one(id, pieces);
        for (i, &pid) in ids.iter().enumerate() {
            let d = if i % 2 == 0 { d0 } else { d0.opposite() };
            self.set_meta(pid, Meta { origin: pm.origin, event: ev, dir: d, dgen: self.child_gen(pm.dgen), litter: pm.litter, litter_size: pm.litter_size });
        }
        (ids[0], *ids.last().expect("presplit emitted no pieces"))
    }

    /// The bracket draw on wire `w` (docs §2.4, v2 2026-08-05): DIRECTIONAL
    /// and length-biased, replacing the halves cascade (whose other-half
    /// preference made midpoint crossing a constant 100% — an overshoot).
    /// Candidates are the bracket-eligible gates targeting w on the picked
    /// g57's OWN side (its stored direction, the cross convention); comp and
    /// 1-control candidates compete equally. Among split_reach_k uniform
    /// samples the FARTHEST (rank distance) wins: k=1 uniform, larger k
    /// prefers longer runs. Candidates born since the last rank stamp are
    /// invisible until the next stamp (growth-triggered, so the blind window
    /// is <=25% of the circuit's life). No candidate on that side = the
    /// twist fails. Returns (id, is_comp, crossed_midpoint).
    fn pick_bracket(&mut self, w: u16, g1: u32, g_rank: u32, d: Dir) -> Option<(u32, bool, bool)> {
        if g_rank == NIL {
            return None;
        }
        let mut cands: Vec<(u32, u32)> = Vec::new();
        for &id in &self.wt_buckets[w as usize] {
            if id == g1 {
                continue;
            }
            let r = self.rank_of(id);
            if r == NIL {
                continue;
            }
            let on_side = match d {
                Dir::R => r > g_rank,
                Dir::L => r < g_rank,
            };
            if on_side {
                cands.push((id, r.abs_diff(g_rank)));
            }
        }
        if cands.is_empty() {
            return None;
        }
        let k = self.params.split_reach_k.max(1);
        let mut best = cands[self.rng.random_range(0..cands.len())];
        for _ in 1..k {
            let c = cands[self.rng.random_range(0..cands.len())];
            if c.1 > best.1 {
                best = c;
            }
        }
        let id = best.0;
        let comp = self.arena.gate(id).comp;
        let mid = (self.rank_n / 2) as u32;
        let crossed = (g_rank < mid) != (self.rank_of(id) < mid);
        Some((id, comp, crossed))
    }

    /// The ORIGINAL v1 bracket cascade, kept as the A/B comparison arm
    /// (split_reach_k = 0): other-half g57 > other-half CNOT/NCNOT >
    /// same-half g57 > same-half CNOT/NCNOT, uniform within the first
    /// non-empty class. Its hard other-half preference makes midpoint
    /// crossing ~always true.
    fn pick_bracket_cascade(&mut self, w: u16, g1: u32, g_rank: u32) -> Option<(u32, bool, bool)> {
        let g_half = if g_rank == NIL || self.rank_n == 0 {
            None
        } else {
            Some((g_rank as usize) >= self.rank_n / 2)
        };
        let mut groups: [Vec<u32>; 4] = Default::default();
        for &id in &self.wt_buckets[w as usize] {
            if id == g1 {
                continue;
            }
            let comp = self.arena.gate(id).comp;
            let r = self.rank_of(id);
            let other = match (g_half, r) {
                (Some(gh), r) if r != NIL => gh != ((r as usize) >= self.rank_n / 2),
                _ => false,
            };
            let k = match (comp, other) {
                (true, true) => 0,
                (false, true) => 1,
                (true, false) => 2,
                (false, false) => 3,
            };
            groups[k].push(id);
        }
        for (k, grp) in groups.iter().enumerate() {
            if !grp.is_empty() {
                let id = grp[self.rng.random_range(0..grp.len())];
                return Some((id, k == 0 || k == 2, k < 2));
            }
        }
        None
    }

    /// The absorbed pure-NOT twist on wire `w` between brackets `g1` and `h1`
    /// (both 1-control gates targeting w). Locates h1 by an alternating
    /// bidirectional walk from g1 (cost <= 2x the segment the flip pass walks
    /// anyway), flips both brackets' control polarity, and conjugates the open
    /// segment: g57s reading w are force-split (5a, keeping the g57+X-series
    /// closure), every w-reading pin flips, gates targeting w are invariant.
    /// Canaries on w anchored in [left, right) count one flip. Returns the
    /// span: gates strictly between the brackets.
    fn apply_not_twist(&mut self, g1: u32, h1: u32, w: u16) -> usize {
        let (left, right) = {
            let (mut l, mut r) = (g1, g1);
            loop {
                if r != NIL {
                    r = self.arena.neighbor(r, Dir::R);
                    if r == h1 {
                        break (g1, h1);
                    }
                }
                if l != NIL {
                    l = self.arena.neighbor(l, Dir::L);
                    if l == h1 {
                        break (h1, g1);
                    }
                }
                assert!(l != NIL || r != NIL, "split-twist bracket not reachable from g1");
            }
        };
        self.absorb_flip(g1);
        self.absorb_flip(h1);
        self.bump_taps_at(left, w);
        let mut span = 0usize;
        let mut cur = self.arena.neighbor(left, Dir::R);
        while cur != right {
            span += 1;
            debug_assert!(cur != NIL, "segment walk ran off the circuit");
            let next = self.arena.neighbor(cur, Dir::R);
            // Bump before mutating: a 5a splice evicts this node's taps to an
            // already-visited neighbor, and the count rides the tap, not the
            // anchor.
            self.bump_taps_at(cur, w);
            let g = self.arena.gate(cur);
            if g.reads(w) {
                if g.comp {
                    // 5a: force-split, then flip the pieces' w-pins. Exact:
                    // conjugation commutes with the exact presplit.
                    let gc = g.clone();
                    let mut pieces = rules::presplit(&gc, &mut self.rng);
                    for p in pieces.iter_mut() {
                        flip_wire_literal(p, w);
                    }
                    // Exhaustive verify only inside verify_rewrite's support
                    // envelope; wide gates get the identical X-conjugation and
                    // stay covered by global_check.
                    if self.params.local_verify && gc.width() < 16 {
                        let x = XGate::x_gate(w);
                        let before = vec![x.clone(), gc.clone(), x];
                        assert!(
                            rules::verify_rewrite(&before, &pieces),
                            "split-twist 5a verification failed: {gc:?} on wire {w}"
                        );
                    }
                    let pm = self.meta_of(cur);
                    let ev = self.fresh_event();
                    for p in &pieces {
                        self.counters.width_hist[p.width().min(15)] += 1;
                    }
                    let d0 = self.rand_dir();
                    let ids = self.splice_replace_one(cur, pieces);
                    for (i, &pid) in ids.iter().enumerate() {
                        let d = if i % 2 == 0 { d0 } else { d0.opposite() };
                        self.set_meta(pid, Meta { origin: pm.origin, event: ev, dir: d, dgen: self.child_gen(pm.dgen), litter: pm.litter, litter_size: pm.litter_size });
                    }
                    self.counters.split_segs += 1;
                } else {
                    let mut ng = g.clone();
                    flip_wire_literal(&mut ng, w);
                    if self.params.local_verify && ng.width() < 16 {
                        let x = XGate::x_gate(w);
                        let before = vec![x.clone(), g.clone(), x];
                        assert!(
                            rules::verify_rewrite(&before, std::slice::from_ref(&ng)),
                            "split-twist pin flip verification failed on wire {w}"
                        );
                    }
                    self.index_remove(cur);
                    self.arena.replace_gate(cur, ng);
                    self.index_add(cur);
                }
            }
            cur = next;
        }
        span
    }

    /// Absorb one X(w) into a 1-control gate targeting w: flip its control's
    /// polarity (CNOT <-> NCNOT). The single-gate composition identity — the
    /// reason the twist pays zero synthetic gates.
    fn absorb_flip(&mut self, id: u32) {
        let g = self.arena.gate(id).clone();
        debug_assert!(!g.comp && g.ctrls.len() == 1, "bracket must be a 1-control conjunction");
        let mut ng = g.clone();
        ng.ctrls[0].1 = !ng.ctrls[0].1;
        if self.params.local_verify {
            let before = vec![XGate::x_gate(g.target), g.clone()];
            assert!(
                rules::verify_rewrite(&before, std::slice::from_ref(&ng)),
                "split-twist absorption verification failed: {g:?}"
            );
        }
        self.index_remove(id);
        self.arena.replace_gate(id, ng);
        self.index_add(id);
    }

    // ---- split-stage instrumentation ----

    /// Restamp approximate position ranks (an O(n) walk). Heuristic
    /// consumers only; see the field comment.
    fn restamp_ranks(&mut self) {
        self.rank.clear();
        self.rank.resize(self.arena.capacity(), NIL);
        for (i, id) in self.arena.ids_in_order().into_iter().enumerate() {
            self.rank[id as usize] = i as u32;
        }
        self.rank_n = self.arena.len();
        self.rank_due = self.moves_done + RANK_EVERY;
    }

    /// Ordinal at the last stamp; NIL = born since it.
    fn rank_of(&self, id: u32) -> u32 {
        self.rank.get(id as usize).copied().unwrap_or(NIL)
    }

    /// Plant the wire canaries: uniform anchors, wire drawn from the anchor's
    /// touched wires. Uses metrics_rng so arming canaries never perturbs the
    /// walk trajectory of a seed.
    fn plant_taps(&mut self) {
        self.taps_planted = true;
        if self.params.split_canaries == 0 {
            return;
        }
        self.restamp_ranks();
        let ids = self.arena.ids_in_order();
        let n = ids.len();
        for _ in 0..self.params.split_canaries {
            let i = self.metrics_rng.random_range(0..n);
            let id = ids[i];
            let g = self.arena.gate(id);
            let mut wires: Vec<u16> = vec![g.target];
            wires.extend(g.ctrls.iter().map(|&(w, _)| w));
            let w = wires[self.metrics_rng.random_range(0..wires.len())];
            self.tap_at.entry(id).or_default().push(self.taps.len() as u32);
            self.taps.push(Tap {
                anchor: id,
                wire: w,
                orig_permille: ((i * 1000) / n.max(1)) as u16,
                flips: 0,
            });
        }
        println!("[fmix] split: planted {} canaries", self.taps.len());
    }

    /// Count a twist flip for every canary on `w` anchored at `id`.
    fn bump_taps_at(&mut self, id: u32, w: u16) {
        if let Some(list) = self.tap_at.get(&id) {
            for &t in list {
                let tp = &mut self.taps[t as usize];
                if tp.wire == w {
                    tp.flips += 1;
                    self.counters.tap_flips += 1;
                }
            }
        }
    }

    /// Re-anchor canaries off a node about to die (call while it is still
    /// linked): to the live left neighbor, right at the head, dropped only on
    /// a single-gate circuit.
    fn evict_taps(&mut self, id: u32) {
        if self.taps.is_empty() {
            return;
        }
        let Some(list) = self.tap_at.remove(&id) else { return };
        let mut to = self.arena.neighbor(id, Dir::L);
        if to == NIL {
            to = self.arena.neighbor(id, Dir::R);
        }
        if to == NIL {
            return;
        }
        for &t in &list {
            self.taps[t as usize].anchor = to;
        }
        self.tap_at.entry(to).or_default().extend(list);
    }

    fn split_report_line(&mut self, span: Option<usize>) {
        let c = &self.counters;
        println!(
            "[fmix] split mv={} size={} comp={} prims={} hspl={} segs={} joins={} xmid={} fails={} streak={} tapf={} span={}",
            self.moves_done,
            self.arena.len(),
            self.comp_ids.len(),
            c.split_prims,
            c.split_hsplits,
            c.split_segs,
            c.split_joins,
            c.split_xmid,
            c.split_fails,
            self.split_fail_streak,
            c.tap_flips,
            span.map_or(-1i64, |s| s as i64),
        );
    }

    /// Span distribution at the stage boundary: mean gates between the
    /// brackets and the fraction-of-circuit histogram (5% buckets, span
    /// normalized by the circuit size AT ITS MOVE).
    fn split_span_summary(&self) {
        let c = &self.counters;
        if c.split_joins == 0 {
            return;
        }
        let cells: Vec<String> =
            c.split_span_hist.iter().map(|&v| v.to_string()).collect();
        println!(
            "[fmix] split spans: mean={:.0} gates over {} twists; frac-of-circuit hist (5% buckets 0-100): {}",
            c.split_span_sum as f64 / c.split_joins as f64,
            c.split_joins,
            cells.join(" ")
        );
    }

    /// Stage boundary (both exits): clear the live flag, latch split_ended for
    /// run()'s split_stop check, and print the stage summary + canary report.
    fn end_split_stage(&mut self, reason: &str) {
        self.split_on = false;
        self.split_ended = true;
        self.split_done = true;
        let c = &self.counters;
        println!(
            "[fmix] split stage ENDED at move {}: {reason} — prims={} hspl={} segs={} joins={} xmid={} fails={} size={} comp={}",
            self.moves_done,
            c.split_prims,
            c.split_hsplits,
            c.split_segs,
            c.split_joins,
            c.split_xmid,
            c.split_fails,
            self.arena.len(),
            self.comp_ids.len(),
        );
        self.split_span_summary();
        self.split_tap_summary();
    }

    /// Canary report: one line per canary plus mean flips by ORIGINAL-position
    /// decile — the spread/reach read. Idempotent (prints once); public so a
    /// run that stops before the stage boundary can still dump it.
    pub fn split_tap_summary(&mut self) {
        if self.taps.is_empty() || self.taps_reported {
            return;
        }
        self.taps_reported = true;
        self.restamp_ranks();
        let n = self.rank_n.max(1);
        let mut dec_flips = [0u64; 10];
        let mut dec_n = [0u64; 10];
        for t in &self.taps {
            let now = self
                .rank
                .get(t.anchor as usize)
                .copied()
                .filter(|&r| r != NIL)
                .map(|r| (r as usize * 1000) / n);
            println!(
                "[fmix] canary wire={} orig={} now={} flips={}",
                t.wire,
                t.orig_permille,
                now.map_or(-1i64, |x| x as i64),
                t.flips
            );
            let d = (t.orig_permille as usize / 100).min(9);
            dec_flips[d] += t.flips;
            dec_n[d] += 1;
        }
        let cells: Vec<String> = (0..10)
            .map(|d| {
                if dec_n[d] == 0 {
                    "-".to_string()
                } else {
                    format!("{:.1}", dec_flips[d] as f64 / dec_n[d] as f64)
                }
            })
            .collect();
        println!(
            "[fmix] canary deciles (mean flips by ORIGINAL position, left to right): {}",
            cells.join(" ")
        );
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
        if self.params.prof_n[2] > 0.0 && self.prof.is_none() {
            if self.moves_done > 0 {
                // v1: the profile's effective-work clock is not serialised, so
                // a resumed profile would restart phase 1 from eff=0. Profiles
                // are whole-run single invocations; warn rather than mis-steer.
                eprintln!(
                    "[fmix] WARNING: --profile on a RESUME (moves_done={}) restarts the profile clock at eff=0 — run a profile as a single invocation",
                    self.moves_done
                );
            }
            self.prof_init();
        }
        while self.moves_done < self.params.moves {
            // Generation targeting: refresh the laggard list on its cadence
            // (an O(size) scan; entries invalidated between scans are pruned
            // lazily at draw time in pick_seed).
            if self.params.gen_target > 0 && self.moves_done >= self.pool_scan_due {
                self.rebuild_pool();
                self.pool_scan_due = self.moves_done + self.params.gen_rescan.max(1);
            }
            // The split stage (docs/FMIX_SPLIT_TWIST.md): while live it owns
            // the whole round — every other slot is withheld, and the stage
            // boundary optionally ends the run (--split-stop).
            let took_split = self.params.split && self.split_on && {
                self.split_twist_move();
                true
            };
            if took_split && self.split_ended {
                self.split_ended = false;
                if self.params.split_stop {
                    self.moves_done += 1;
                    self.counters.moves = self.moves_done;
                    self.global_check();
                    self.report();
                    return MixStop::SplitDone;
                }
            }
            // Slot 0 continued. With a PROFILE active the controller is the
            // one size authority: it owns target_size, keeps the brake
            // inert, and drives the MIX/COMP coin. Otherwise: the size
            // brake, then the p_mix overlay (whose per-round coin, when
            // armed, is the binding choice; the thermostat is untouched).
            if took_split {
                // stage round: no slot-0 steering
            } else if self.prof.is_some() {
                self.prof_tick();
                if self.prof.as_ref().is_some_and(|p| p.phase == 4) {
                    self.report();
                    return MixStop::ProfileDone;
                }
            } else {
                self.apply_size_brake();
                self.apply_mode_overlay();
            }
            // Slot 1: one twist, at a FIXED rate the rest of the machinery
            // balances around.
            let took_twist = !took_split
                && self.params.p_twist > 0.0
                && self.rng.random_bool(self.params.p_twist.clamp(0.0, 1.0))
                && { self.twist_round(); true };
            // Slot 1b: GLOBAL re-randomisation, sitting after the twist and
            // before the DB move. Rate is scaled by the circuit size, so the
            // DEFAULT probability is exactly 1/|circuit| -- an expected one
            // whole-circuit reshuffle per |circuit| rounds, and O(mean slack)
            // expected work per round however large the circuit gets.
            let took_shuffle = !took_split
                && !took_twist
                && self.params.shuffle_rate > 0.0
                && self.arena.len() >= 2
                && {
                    let p = (self.params.shuffle_rate / self.arena.len() as f64).clamp(0.0, 1.0);
                    self.rng.random_bool(p)
                }
                && {
                    self.global_shuffle();
                    true
                };
            // Slot 1c: one bridge fusion (docs/NONLOCAL_PHASE_A.md), at a
            // fixed rate like the twist. p_bridge == 0 draws no RNG.
            let took_bridge = !took_split
                && !took_twist
                && !took_shuffle
                && self.params.p_bridge > 0.0
                && self.arena.len() >= 4
                && self.rng.random_bool(self.params.p_bridge.clamp(0.0, 1.0))
                && {
                    self.bridge_round();
                    true
                };
            // Slot 2: ONE DB move, under the live db_mode. A round whose
            // descent finds nothing is SPENT -- there is no fallthrough -- so
            // the thermostat receives exactly (1 - p_twist)(1 - p_db) of rounds
            // no matter how hard the material is.
            let took_db = !took_split
                && !took_twist
                && !took_shuffle
                && !took_bridge
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
            if !took_split && !took_twist && !took_shuffle && !took_bridge && !took_db {
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
            // Checked every move (a plain bool — no RNG, no trajectory
            // effect) so FMIX_STOP_AT_PHASE stops at the transition move
            // rather than up to report_every moves later. The stop-FLAG
            // file poll stays at report cadence below.
            if self.stop_requested {
                self.global_check();
                return MixStop::StopFlag;
            }
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
        let id = self.pick_cross_shot();
        self.cross_move_on(id);
    }

    /// Shot selection for the cross. With p_mincross off (the default) this
    /// is exactly the historical uniform draw and consumes no extra RNG.
    /// Armed, a coin sends the shot to the min-dgen pool: the K least-split
    /// lineages at the last cadenced rebuild, each entry consumed on draw
    /// (dead entries pruned lazily). The pool draining before the next
    /// rebuild silently degrades to uniform — size cross_pool_k above the
    /// expected biased draws per cadence.
    fn pick_cross_shot(&mut self) -> u32 {
        if self.params.p_mincross > 0.0 {
            if self.moves_done >= self.cross_pool_due {
                self.rebuild_cross_pool();
            }
            if !self.cross_pool.is_empty()
                && self.rng.random_bool(self.params.p_mincross.clamp(0.0, 1.0))
            {
                for _ in 0..8 {
                    if self.cross_pool.is_empty() {
                        break;
                    }
                    let i = self.rng.random_range(0..self.cross_pool.len());
                    let id = self.cross_pool.swap_remove(i);
                    if self.arena.is_linked(id) {
                        self.counters.cross_pool_shots += 1;
                        return id;
                    }
                }
            }
        }
        self.arena.random_linked(&mut self.rng)
    }

    /// O(n) scan + O(n) select of the K lowest-dgen linked gates. dgen is
    /// the split-generation stamp — during a DB-free walk it counts the
    /// split events in a gate's lineage, so low dgen IS "family the walk
    /// has not touched".
    fn rebuild_cross_pool(&mut self) {
        let mut cand: Vec<(u32, u32)> = self
            .arena
            .ids_in_order()
            .into_iter()
            .map(|id| (self.meta_of(id).dgen, id))
            .collect();
        let k = self.params.cross_pool_k.min(cand.len());
        if k > 0 && k < cand.len() {
            cand.select_nth_unstable(k - 1);
        }
        cand.truncate(k);
        self.cross_pool = cand.into_iter().map(|(_, id)| id).collect();
        self.cross_pool_due = self.moves_done + self.params.cross_rescan.max(1);
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
            // Sibling convention (2026-08-05): pieces of one g57 split take
            // ALTERNATING directions from a fair draw — never the old
            // independent per-piece child_dir, under which siblings could
            // agree.
            let d0 = self.rand_dir();
            for (i, &pid) in ids.iter().enumerate() {
                let d = if i % 2 == 0 { d0 } else { d0.opposite() };
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
                // Sibling convention (2026-08-05): alternating directions
                // from a fair draw, replacing the old inherit-from-the-shot
                // law for presplit fragments.
                let d0 = self.rand_dir();
                for (i, &pid) in ids.iter().enumerate() {
                    let d = if i % 2 == 0 { d0 } else { d0.opposite() };
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
                // Ancestry treats a cross as a DB splice over the window
                // {g, h}: EVERY output — the intact pivot included — carries
                // the union of both parents' ancestor sets, because the
                // rewrite re-encodes the pair jointly even when one gate's
                // spelling survives it. The journal entry below records the
                // PRE-cross litters, so an undo reverses this too. Litter
                // inheritance is unchanged when ancestry is off, matching
                // the merge-union precedent.
                let union_litter = if self.anc_words > 0 && gm.litter != hm.litter {
                    let l = self.anc_union_litter(&[gm.litter, hm.litter]);
                    Some((l, placed.len().min(u16::MAX as usize) as u16))
                } else {
                    None
                };
                let mut fresh: Vec<u32> = Vec::new();
                for &(pid, role) in &placed {
                    match role {
                        Role::ShotPiece | Role::Core => {
                            let d = self.child_dir(dir);
                            let (litter, litter_size) =
                                union_litter.unwrap_or((gm.litter, gm.litter_size));
                            self.set_meta(
                                pid,
                                Meta { origin: g_origin, event: ev, dir: d, dgen: self.child_gen(gm.dgen), litter, litter_size },
                            );
                            fresh.push(pid);
                        }
                        Role::CollidingPiece => {
                            let d = self.child_dir(dir);
                            let (litter, litter_size) =
                                union_litter.unwrap_or((hm.litter, hm.litter_size));
                            self.set_meta(
                                pid,
                                Meta { origin: h_origin, event: ev, dir: d, dgen: self.child_gen(hm.dgen), litter, litter_size },
                            );
                            fresh.push(pid);
                        }
                        Role::CollidingIntact => {
                            // Node reused; only the ancestry-bearing fields
                            // move. Origin, event, dir and dgen stay intact so
                            // trajectories and tabu semantics are unchanged.
                            // The stamp bump makes absorption count as
                            // "further processing": any OLDER journal entry
                            // holding this node dies, since undoing it would
                            // wipe the ancestors absorbed here. This cross's
                            // own entry records the post-bump stamp below and
                            // stays undoable.
                            if let Some((l, ls)) = union_litter {
                                let m = self.meta_of(pid);
                                self.set_meta(pid, Meta { litter: l, litter_size: ls, ..m });
                                self.arena.touch(pid);
                            }
                        }
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

    // One conjugation twist from the swap family. The twist operator T acts on
    // two wires (a, b): a wire SWAP, optionally composed with a negation of one
    // or both wires. `alpha` negates wire a, `beta` wire b, each an independent
    // fair coin -- so the menu is
    //   swap                (0,0)  p = 1/4
    //   swap + negate one   (1,0) or (0,1)  p = 1/2
    //   swap + negate both  (1,1)  p = 1/4
    // Each T is realised as THREE single-control gates (a 3-CNOT swap network
    // with control polarities chosen so the outer CNOTs carry the negations):
    //   G1: b ^= (a | !a)   ctrl a -> tgt b, negative control iff alpha
    //   G2: a ^= b          ctrl b -> tgt a, always positive
    //   G3: b ^= (a | !a)   ctrl a -> tgt b, negative control iff beta
    // which realises T(a,b) = (b ^ alpha, a ^ beta). T is an involution iff
    // alpha == beta, so the closing bracket is P^-1 (the reversed packet), which
    // differs from the opening P exactly in the negate-one case. Interior gates
    // are relabelled by conj_by_swap then conj_by_not on each negated wire -- a
    // pure 1->1 relabel (no case-splits, no width change), so the whole family
    // is function-preserving by commutation, verified per gate under
    // local_verify (P^-1 . g . P == g').
    fn twist_move(&mut self) {
        let n = self.arena.len();
        if n < 2 {
            return;
        }
        let cap = n;
        let lmin = (self.params.twist_min_len.max(2).min(cap)) as f64;
        let len = (self.rng.random_range(lmin.ln()..=(cap as f64).ln()).exp().round() as usize)
            .clamp(2, cap);
        // Symmetric truncation: the window's virtual start is uniform over
        // [-(len-1), n-1] and clamped to the circuit, so left-overshooting draws
        // pile their opening packets at the head exactly as right-overshoots
        // pile closings at the tail. `find_twist_site` may bias the start toward
        // an absorbing neighbourhood; its preferred wire is unused by the swap
        // family (the swap picks its own wires below).
        let site = self.find_twist_site();
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

        // Pass 1: locate the window end (truncated at the tail) and collect the
        // wires it touches, so `a` can be a wire the window actually uses (a T
        // acting on none of them is a no-op twist).
        let mut touch_seen = vec![false; self.num_wires];
        let mut touches: Vec<u16> = Vec::new();
        let mut end = start;
        let mut span = 0usize;
        let mut cur = start;
        while cur != NIL && span < len {
            let g = self.arena.gate(cur);
            if !touch_seen[g.target as usize] {
                touch_seen[g.target as usize] = true;
                touches.push(g.target);
            }
            for &(w, _) in &g.ctrls {
                if !touch_seen[w as usize] {
                    touch_seen[w as usize] = true;
                    touches.push(w);
                }
            }
            end = cur;
            span += 1;
            cur = self.arena.neighbor(cur, Dir::R);
        }

        // Wire pair: `a` from a wire the window touches, `b` any other wire
        // (routing the window's material through a fresh physical wire is a
        // legitimate and strong relabeling).
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

        // The negation pattern: each wire negated independently with
        // probability twist_neg_p (0.5 = the 1/4:1/2:1/4 family; 0 = pure swap,
        // no polarity flips).
        let q = self.params.twist_neg_p.clamp(0.0, 1.0);
        let alpha = self.rng.random_bool(q);
        let beta = self.rng.random_bool(q);

        // The packet P (three single-control gates) and its inverse P^-1
        // (reverse order; each polarised single-control XOR is its own inverse).
        // A negative control (polarity false) is `target ^= !wire`, i.e. it
        // folds the wire negation into the CNOT -- so no extra NOT gate is
        // needed and the packet stays exactly three gates.
        let pol_cnot = |t: u16, c: u16, neg: bool| {
            XGate::conj(t, [(c, !neg)]).expect("single-control cnot literal")
        };
        let packet: Vec<XGate> =
            vec![pol_cnot(b, a, alpha), pol_cnot(a, b, false), pol_cnot(b, a, beta)];
        let packet_inv: Vec<XGate> = packet.iter().rev().cloned().collect();

        if self.params.local_verify {
            // P then P^-1 is the identity.
            let round: Vec<XGate> = packet.iter().chain(packet_inv.iter()).cloned().collect();
            assert!(
                rules::verify_rewrite(&round, &[]),
                "twist packet is not invertible by its reverse: {packet:?}"
            );
        }

        // Pass 2: conjugate every interior gate. g' = conj_by_swap(g) then
        // conj_by_not on a (if alpha) and on b (if beta). The swap family never
        // case-splits, so every gate is rewritten strictly in place -- nodes,
        // positions and provenance survive.
        let mut relabeled = 0u64;
        let mut cur = start;
        loop {
            let is_last = cur == end;
            let next = self.arena.neighbor(cur, Dir::R);
            let g = self.arena.gate(cur).clone();
            let relabeled_gate = match conj_by_swap(&g, a, b) {
                None => None, // touches neither wire: invariant (the negation is too)
                Some(gs) => {
                    let mut out = gs;
                    if alpha {
                        if let Some(x) = conj_by_not(&out, a) {
                            out = x;
                        }
                    }
                    if beta {
                        if let Some(x) = conj_by_not(&out, b) {
                            out = x;
                        }
                    }
                    Some(out)
                }
            };
            if let Some(g2) = relabeled_gate {
                if self.params.local_verify {
                    // The per-gate identity: opening P, closing P^-1, so the
                    // interior gate is P^-1 . g . P.
                    let mut seq = packet_inv.clone();
                    seq.push(g.clone());
                    seq.extend(packet.iter().cloned());
                    assert!(
                        rules::verify_rewrite(&seq, std::slice::from_ref(&g2)),
                        "twist conjugation failed: {g:?} a={a} b={b} alpha={alpha} beta={beta}"
                    );
                }
                self.index_remove(cur);
                self.arena.replace_gate(cur, g2);
                self.index_add(cur);
                relabeled += 1;
            }
            if is_last {
                break;
            }
            cur = next;
        }

        // Bracket the window: opening packet P before `start`, closing packet
        // P^-1 after `end`. Packet gates are fresh synthetic material (no
        // origin) sharing one event, so the trivial bracket-cancel is tabu like
        // any fresh sibling pair. They are NOT scattered -- they sit tight on
        // the window edges; later churn is free to float or merge them (every
        // such move is independently function-preserving).
        let ev = self.fresh_event();
        let mut anchor = self.arena.neighbor(start, Dir::L);
        for g in &packet {
            self.counters.width_hist[g.width().min(15)] += 1;
            anchor = self.arena.insert_after(anchor, g.clone());
            self.index_add(anchor);
            let d = self.rand_dir();
            let lit = self.fresh_litter();
            self.set_meta(anchor, Meta { origin: ORIGIN_SYNTH, event: ev, dir: d, dgen: GEN_FRESH, litter: lit, litter_size: 1 });
        }
        let mut anchor = end;
        for g in &packet_inv {
            self.counters.width_hist[g.width().min(15)] += 1;
            anchor = self.arena.insert_after(anchor, g.clone());
            self.index_add(anchor);
            let d = self.rand_dir();
            let lit = self.fresh_litter();
            self.set_meta(anchor, Meta { origin: ORIGIN_SYNTH, event: ev, dir: d, dgen: GEN_FRESH, litter: lit, litter_size: 1 });
        }

        // Counters. Field names are retained for .state compatibility, meaning
        // repurposed: twist_swaps = pure swap, twist_negs = negate one wire,
        // twist_cnots = negate both. (Reported as tsw / tn1 / tn2.)
        match (alpha, beta) {
            (false, false) => self.counters.twist_swaps += 1,
            (true, true) => self.counters.twist_cnots += 1,
            _ => self.counters.twist_negs += 1,
        }
        self.counters.twist_span += span as u64;
        self.counters.twist_relabels += relabeled;
    }

    /// One seam of a --twist-g57 bracket: gather up to 3 neighborhood gates
    /// outward from `edge` (support capped at 4 wires including a, b), then
    /// for every context depth k ask the swap-word engine for the shortest
    /// all-g57 word realizing [ctx . S] (left seam) or [S . ctx] (right
    /// seam), and keep the cheapest by net cost (word len - k), deeper
    /// context on ties. k = 0 always solves (dist(S_ab) = 6), so a bracket
    /// always exists. Returns (consumed ids nearest-first, replacement gates
    /// in circuit order).
    fn solve_seam(&mut self, edge: u32, dir: Dir, a: u16, b: u16) -> (Vec<u32>, Vec<XGate>) {
        let eng = swap_words::engine();
        // Context gather: a gate joins while the combined support (with a, b)
        // still fits the engine's 4 abstract wires.
        let mut ids: Vec<u32> = Vec::new();
        let mut sup: Vec<u16> = vec![a, b];
        let mut cur = self.arena.neighbor(edge, dir);
        while ids.len() < 3 && cur != NIL {
            let g = self.arena.gate(cur);
            let mut s2 = sup.clone();
            for w in std::iter::once(g.target).chain(g.ctrls.iter().map(|&(w, _)| w)) {
                if !s2.contains(&w) {
                    s2.push(w);
                }
            }
            if s2.len() > 4 {
                break;
            }
            sup = s2;
            ids.push(cur);
            cur = self.arena.neighbor(cur, dir);
        }
        // Bind the abstract wires: 0 = a, 1 = b, then the context's own
        // wires, padded with fresh random wires (the engine may route
        // through a helper the context never touched).
        let mut wires: [u16; 4] = [a, b, a, a];
        for i in 2..4 {
            if let Some(&w) = sup.get(i) {
                wires[i] = w;
            } else {
                // Random start, deterministic scan: with num_wires >= 4 a
                // free wire always exists, so this cannot fail.
                let off = self.rng.random_range(0..self.num_wires);
                for d in 0..self.num_wires {
                    let w = ((off + d) % self.num_wires) as u16;
                    if !wires[..i].contains(&w) {
                        wires[i] = w;
                        break;
                    }
                }
            }
        }
        let abs_of = |w: u16| wires.iter().position(|&x| x == w).unwrap() as u8;
        let perms: Vec<u64> = ids
            .iter()
            .map(|&id| {
                let g = self.arena.gate(id);
                let ctrls: Vec<(u8, bool)> =
                    g.ctrls.iter().map(|&(w, p)| (abs_of(w), p)).collect();
                swap_words::xgate_perm(abs_of(g.target), &ctrls, g.comp)
            })
            .collect();
        // k = 0 is the precomputed bare spelling of S_ab — no scan needed.
        let mut best: Option<(usize, smallvec::SmallVec<[u8; 7]>)> =
            Some((0, smallvec::SmallVec::from_slice(eng.bare_word())));
        let t0 = std::time::Instant::now();
        for k in 1..=perms.len() {
            // Segment perm in circuit (= apply) order. Context ids are
            // nearest-first, so the left seam's circuit order is h_k..h_1
            // then the bracket; the right seam is the bracket then h_1..h_k.
            let mut t = match dir {
                Dir::L => {
                    let mut t = swap_words::IDENT;
                    for i in (0..k).rev() {
                        t = swap_words::compose(t, perms[i]);
                    }
                    swap_words::compose(t, eng.s_ab)
                }
                Dir::R => eng.s_ab,
            };
            if dir == Dir::R {
                for p in perms.iter().take(k) {
                    t = swap_words::compose(t, *p);
                }
            }
            self.counters.tg_solves += 1;
            // Memoized solve: byte-neutral because solve() is a pure function
            // of (t, MAX_WORD) — the memo only skips recomputing it.
            let solved = self
                .solve_memo
                .entry(t)
                .or_insert_with(|| {
                    eng.solve(t, swap_words::MAX_WORD).map(smallvec::SmallVec::from_vec)
                })
                .clone();
            if let Some(w) = solved {
                let better = match &best {
                    None => true,
                    Some((bk, bw)) => {
                        // Signed: non-g57 context gates are worth several
                        // g57s each, so a seam can solve BELOW its consumed
                        // count (net < 0 — a twist that shrinks the circuit).
                        let net = w.len() as i64 - k as i64;
                        let bnet = bw.len() as i64 - *bk as i64;
                        net < bnet || (net == bnet && k > *bk)
                    }
                };
                if better {
                    best = Some((k, w));
                }
            }
        }
        self.counters.tg_solve_ns += t0.elapsed().as_nanos() as u64;
        let (k, word) = best.expect("k = 0 always solves: dist(S_ab) = 6");
        (ids[..k].to_vec(), eng.decode(&word, &wires))
    }

    /// Bracket positions further out than `from` whose next-outward gate is a
    /// g57 pinning both twist wires — the only shape a k=1 attachment can
    /// cancel against. Sliding a bracket outward just extends the conjugated
    /// window over the gates stepped past, which is free (window length was a
    /// random draw, and a relabel costs far less than the word the slide
    /// saves), so the scan may roam TG_SLIDE_CAP gates.
    fn slide_candidates(&self, from: u32, dir: Dir, a: u16, b: u16) -> Vec<u32> {
        let mut out = Vec::new();
        let mut e = from;
        for _ in 0..TG_SLIDE_CAP {
            let nxt = self.arena.neighbor(e, dir);
            if nxt == NIL {
                break;
            }
            e = nxt;
            let h = self.arena.neighbor(e, dir);
            if h == NIL {
                break;
            }
            let g = self.arena.gate(h);
            if g.comp && g.ctrls.len() == 2 {
                let pins = [g.target, g.ctrls[0].0, g.ctrls[1].0];
                if pins.contains(&a) && pins.contains(&b) {
                    out.push(e);
                    if out.len() >= TG_SLIDE_TRIES {
                        break;
                    }
                }
            }
        }
        out
    }

    /// Solve one seam at its drawn boundary and, when that stays bare and
    /// sliding is enabled, retry at up to TG_SLIDE_TRIES attachment positions
    /// further out. First position reaching +4 wins (a k=1 cancel cannot be
    /// beaten by another single attachment; deeper context can, and is kept
    /// when found). Returns the seam and how many slides were adopted.
    fn eval_seam(&mut self, edge: u32, dir: Dir, a: u16, b: u16) -> (TgSeam, u64) {
        let (ids, repl) = self.solve_seam(edge, dir, a, b);
        let mut seam = TgSeam { edge, ids, repl };
        let mut slid = 0u64;
        if tg_slide_on() && seam.net() >= 6 {
            for e in self.slide_candidates(edge, dir, a, b) {
                let (ids2, repl2) = self.solve_seam(e, dir, a, b);
                let cand = TgSeam { edge: e, ids: ids2, repl: repl2 };
                if cand.net() < seam.net() {
                    let good = cand.net() <= 4;
                    seam = cand;
                    slid = 1;
                    if good {
                        break;
                    }
                }
            }
        }
        (seam, slid)
    }

    /// The --twist-g57 realization of a pure-swap conjugation twist: same
    /// window draw and interior relabel as twist_move, but each bracket is an
    /// all-g57 word sited by solve_seam so it absorbs neighborhood gates —
    /// the ssg hidden-SAMF mechanism, XGate-native. The left seam spells
    /// [ctx . S] as one word (consuming ctx), the right spells [S . ctx]:
    /// the segment becomes R . W' . R' = ctx_l . S . W' . S . ctx_r, which is
    /// the original since W' is the swap-conjugated interior. All inserted
    /// gates take the ballistic birth-advance unconditionally, aimed outward.
    ///
    /// v2 placement: seams that stay bare may SLIDE outward to an attachment
    /// gate (extending the conjugated window), and the two ends are chosen
    /// TOGETHER — a window whose best plan still totals worse than
    /// TG_ACCEPT_NET is redrawn (up to TG_RETRIES), so a side is left bare
    /// only when its partner's match pays for it or every redraw failed.
    fn twist_move_g57(&mut self) {
        if self.arena.len() < 2 || self.num_wires < 4 {
            self.counters.twist_skips += 1;
            return;
        }

        // Draw-and-evaluate loop: each attempt draws a window, seeds wire
        // pairs from its boundary gates, solves both seams (with slides for
        // whichever side stays bare), and the round commits the first plan
        // reaching TG_ACCEPT_NET — else the best plan any attempt produced.
        // Evaluation is read-only, so plans stay valid across attempts.
        let mut best: Option<TgPlan> = None;
        let mut draws = 0u64;
        let attempts = if tg_retry_on() { TG_RETRIES } else { 1 };
        for _ in 0..attempts {
            draws += 1;
            let n = self.arena.len();
            let cap = n;
            let lmin = (self.params.twist_min_len.max(2).min(cap)) as f64;
            let len = (self.rng.random_range(lmin.ln()..=(cap as f64).ln()).exp().round()
                as usize)
                .clamp(2, cap);
            // Symmetric truncation, exactly as in twist_move.
            let draw = self.rng.random_range(0..n + len - 1);
            let (start, len) = if draw < len - 1 {
                (self.arena.head(), draw + 1)
            } else {
                (self.arena.random_linked(&mut self.rng), len)
            };

            // Pass 1: window end + touched wires (a must touch the window or
            // the conjugation is a no-op).
            let mut touch_seen = vec![false; self.num_wires];
            let mut touches: Vec<u16> = Vec::new();
            let mut end = start;
            let mut span = 0usize;
            let mut cur = start;
            while cur != NIL && span < len {
                let g = self.arena.gate(cur);
                for w in std::iter::once(g.target).chain(g.ctrls.iter().map(|&(w, _)| w)) {
                    if !touch_seen[w as usize] {
                        touch_seen[w as usize] = true;
                        touches.push(w);
                    }
                }
                end = cur;
                span += 1;
                cur = self.arena.neighbor(cur, Dir::R);
            }
            if touches.is_empty() {
                continue;
            }
            // Candidate wire pairs, anchor-first: a pair drawn from a
            // boundary gate's own pins makes that gate consumable at its
            // seam (a uniform random b almost never lands inside the 4-wire
            // support). The uniform pair keeps fresh-wire routing on the
            // menu; it wins whenever no boundary pair beats its net.
            let mut cands: Vec<(u16, u16)> = Vec::new();
            for edge in [self.arena.neighbor(start, Dir::L), self.arena.neighbor(end, Dir::R)] {
                if edge == NIL {
                    continue;
                }
                let g = self.arena.gate(edge);
                let pins: Vec<u16> =
                    std::iter::once(g.target).chain(g.ctrls.iter().map(|&(w, _)| w)).collect();
                for &pa in &pins {
                    if !touch_seen[pa as usize] {
                        continue; // `a` must touch the window
                    }
                    for &pb in &pins {
                        if pb != pa && !cands.contains(&(pa, pb)) {
                            cands.push((pa, pb));
                        }
                    }
                }
            }
            cands.truncate(6);
            let a0 = touches[self.rng.random_range(0..touches.len())];
            for _ in 0..16 {
                let c = self.rng.random_range(0..self.num_wires) as u16;
                if c != a0 {
                    if !cands.contains(&(a0, c)) {
                        cands.push((a0, c));
                    }
                    break;
                }
            }
            if cands.is_empty() {
                continue;
            }

            // Both seams for every pair, no slides yet; cheapest total wins.
            let mut pair_best: Option<TgPlan> = None;
            for &(a, b) in &cands {
                let (l_ids, l_repl) = self.solve_seam(start, Dir::L, a, b);
                let (r_ids, r_repl) = self.solve_seam(end, Dir::R, a, b);
                let plan = TgPlan {
                    a,
                    b,
                    l: TgSeam { edge: start, ids: l_ids, repl: l_repl },
                    r: TgSeam { edge: end, ids: r_ids, repl: r_repl },
                    slides: 0,
                };
                if pair_best.as_ref().map_or(true, |p| plan.score() < p.score()) {
                    pair_best = Some(plan);
                }
            }
            let mut plan = pair_best.expect("cands is non-empty");
            // Slides, for whichever side of the winning pair stayed bare.
            if plan.l.net() >= 6 {
                let (seam, s) = self.eval_seam(start, Dir::L, plan.a, plan.b);
                if seam.net() < plan.l.net() {
                    plan.l = seam;
                    plan.slides += s;
                }
            }
            if plan.r.net() >= 6 {
                let (seam, s) = self.eval_seam(end, Dir::R, plan.a, plan.b);
                if seam.net() < plan.r.net() {
                    plan.r = seam;
                    plan.slides += s;
                }
            }
            if best.as_ref().map_or(true, |p| plan.score() < p.score()) {
                best = Some(plan);
            }
            if best.as_ref().expect("just set").score().0 <= TG_ACCEPT_NET {
                break;
            }
        }
        let Some(plan) = best else {
            self.counters.twist_skips += 1;
            return;
        };
        self.counters.tg_retries += draws - 1;
        self.counters.tg_slides += plan.slides;
        let (a, b) = (plan.a, plan.b);
        // Negative nets (shrinking seams) fold into bucket 0.
        self.counters.tg_net_hist[plan.l.net().clamp(0, 7) as usize] += 1;
        self.counters.tg_net_hist[plan.r.net().clamp(0, 7) as usize] += 1;

        // The reference bracket for verification: the known-correct 3-CNOT
        // swap packet (a palindrome, so it is its own inverse).
        let packet3 = vec![
            XGate::cnot(b, a),
            XGate::cnot(a, b),
            XGate::cnot(b, a),
        ];

        // Pass 2: conjugate the interior by the swap — a pure 1->1 relabel.
        // Slid seams extended the window, so the walk runs between the PLAN's
        // edges (l.edge <= drawn start, r.edge >= drawn end, both real nodes).
        let mut relabeled = 0u64;
        let mut span_walked = 0u64;
        let mut cur = plan.l.edge;
        loop {
            let is_last = cur == plan.r.edge;
            span_walked += 1;
            let next = self.arena.neighbor(cur, Dir::R);
            let g = self.arena.gate(cur).clone();
            if let Some(g2) = conj_by_swap(&g, a, b) {
                if self.params.local_verify {
                    let mut seq = packet3.clone();
                    seq.push(g.clone());
                    seq.extend(packet3.iter().cloned());
                    assert!(
                        rules::verify_rewrite(&seq, std::slice::from_ref(&g2)),
                        "g57-twist conjugation failed: {g:?} a={a} b={b}"
                    );
                }
                self.index_remove(cur);
                self.arena.replace_gate(cur, g2);
                self.index_add(cur);
                relabeled += 1;
            }
            if is_last {
                break;
            }
            cur = next;
        }

        // Splice the seams. Consumed ids are nearest-first, so the left run's
        // circuit order is ids reversed; anchors are read before unlinking.
        if self.params.local_verify {
            let mut old: Vec<XGate> =
                plan.l.ids.iter().rev().map(|&id| self.arena.gate(id).clone()).collect();
            old.extend(packet3.iter().cloned());
            assert!(
                rules::verify_rewrite(&old, &plan.l.repl),
                "g57-twist left seam failed: a={a} b={b}"
            );
            let mut old: Vec<XGate> = packet3.clone();
            old.extend(plan.r.ids.iter().map(|&id| self.arena.gate(id).clone()));
            assert!(
                rules::verify_rewrite(&old, &plan.r.repl),
                "g57-twist right seam failed: a={a} b={b}"
            );
        }
        let ev = self.fresh_event();
        let mut inserted: Vec<u32> =
            Vec::with_capacity(plan.l.repl.len() + plan.r.repl.len());
        let l_anchor = match plan.l.ids.last() {
            Some(&far) => self.arena.neighbor(far, Dir::L),
            None => self.arena.neighbor(plan.l.edge, Dir::L),
        };
        // Consumed context carried real lineage: union its litters' ancestor
        // sets into the replacement's litter, exactly as a DB splice would —
        // v1 dropped them, which silently deflated anc under consumption.
        let mut l_srcs: Vec<u64> =
            plan.l.ids.iter().map(|&id| self.meta_of(id).litter).collect();
        l_srcs.sort_unstable();
        l_srcs.dedup();
        let mut r_srcs: Vec<u64> =
            plan.r.ids.iter().map(|&id| self.meta_of(id).litter).collect();
        r_srcs.sort_unstable();
        r_srcs.dedup();
        for &id in plan.l.ids.iter().chain(plan.r.ids.iter()) {
            self.evict_taps(id);
            self.index_remove(id);
            self.arena.unlink(id);
        }
        let l_lit = self.anc_union_litter(&l_srcs);
        let r_lit = self.anc_union_litter(&r_srcs);
        let mut anchor = l_anchor;
        for g in &plan.l.repl {
            self.counters.width_hist[g.width().min(15)] += 1;
            anchor = self.arena.insert_after(anchor, g.clone());
            self.index_add(anchor);
            self.set_meta(anchor, Meta { origin: ORIGIN_SYNTH, event: ev, dir: Dir::L, dgen: GEN_FRESH, litter: l_lit, litter_size: plan.l.repl.len() as u16 });
            inserted.push(anchor);
        }
        let mut anchor = plan.r.edge;
        for g in &plan.r.repl {
            self.counters.width_hist[g.width().min(15)] += 1;
            anchor = self.arena.insert_after(anchor, g.clone());
            self.index_add(anchor);
            self.set_meta(anchor, Meta { origin: ORIGIN_SYNTH, event: ev, dir: Dir::R, dgen: GEN_FRESH, litter: r_lit, litter_size: plan.r.repl.len() as u16 });
            inserted.push(anchor);
        }

        // Part (b): every inserted gate rides its (outward) direction, the
        // db_advance treatment applied unconditionally. Same-support g57s
        // mostly collide pairwise, so the packet spreads caterpillar-style —
        // outer gates travel, inner ones stop at their siblings.
        self.advance_births(&inserted);

        self.counters.twist_swaps += 1;
        self.counters.twist_span += span_walked;
        self.counters.twist_relabels += relabeled;
        self.counters.tg_consumed += (plan.l.ids.len() + plan.r.ids.len()) as u64;
        self.counters.tg_emitted += inserted.len() as u64;
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
            // Scan the piece's own (persistent) Meta direction first: pieces
            // are advanced in their meta dir at birth, so the hint is usually
            // right and the full-reach wrong-side miss is skipped. The piece's
            // LOCATION determines side/found, not the scan order, so the
            // outcome is bit-identical.
            let hint = self.meta_of(id).dir;
            let mut side = None;
            for d in [hint, hint.opposite()] {
                let edge = match d {
                    Dir::R => edge_r,
                    Dir::L => edge_l,
                };
                let mut cur = self.arena.neighbor(edge, d);
                let mut steps = 0usize;
                while cur != NIL && steps < self.params.merge_reach {
                    if cur == id {
                        side = Some(d);
                        break;
                    }
                    cur = self.arena.neighbor(cur, d);
                    steps += 1;
                }
                if side.is_some() {
                    break;
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
            self.evict_taps(bid);
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
        let mut cand_set: smallvec::SmallVec<[u32; 8]> = smallvec::SmallVec::from_vec(cands);
        cand_set.sort_unstable();
        let mut chosen: Option<(u32, Dir, usize)> = None;
        for dir in [Dir::R, Dir::L] {
            let mut g_colliders: Vec<u32> = Vec::new();
            let mut cur = self.arena.neighbor(g_id, dir);
            let mut steps = 0usize;
            while cur != NIL && steps < self.params.merge_reach {
                if chosen.is_some_and(|(_, _, d)| steps >= d) {
                    break; // the other direction already found a nearer one
                }
                if cand_set.binary_search(&cur).is_ok() {
                    let wall =
                        g_colliders.iter().any(|&b| self.arena.collides_ids(b, cur));
                    if wall {
                        self.counters.merge_wall_blocked += 1;
                    } else {
                        chosen = Some((cur, dir, steps));
                        break;
                    }
                }
                if self.arena.collides_ids(g_id, cur) {
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
        // Evict-then-unlink per node: h's eviction runs after g's unlink so
        // its target can never be the dying g (both partners die here).
        self.evict_taps(g_id);
        self.index_remove(g_id);
        self.arena.unlink(g_id);
        self.evict_taps(h_id);
        self.index_remove(h_id);
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
        // stable-ledger: pick between the two stable selections by the flow
        // ledger — ±1 while shrink budget remains (removed < added + slack),
        // w/w+1 once exhausted. Bounds net DB drift below by -slack at any
        // size, without wasting hits on post-draw declines.
        const STABLE_LEDGER_SLACK: u64 = 64;
        let mode = if mode == DbMode::StableLedger {
            if self.stable_led_removed < self.stable_led_added + STABLE_LEDGER_SLACK {
                DbMode::Stable
            } else {
                DbMode::StableGrow
            }
        } else {
            mode
        };
        // band-ledger: full band while the size ledger is inside the slack,
        // corrective size skew once it drifts. Slack is proportional to the
        // circuit (0.2%, floor 64) so the correction scales with the run.
        self.band_led_round = mode == DbMode::BandLedger;
        let mode = if mode == DbMode::BandLedger {
            let slack = ((self.arena.len() as i64) / 500).max(64);
            if self.band_led > slack {
                DbMode::BandShrink
            } else if self.band_led < -slack {
                DbMode::BandGrow
            } else {
                DbMode::SizeAgnostic
            }
        } else {
            mode
        };
        self.db_seed_home = None;
        self.seed_from_pool = false;
        self.seed_fell_through = false;
        self.db_pair_round = false;
        let spliced = self.db_attempt_inner(mode);
        if spliced && self.db_pair_round {
            self.counters.pair_splices += 1;
        }
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
        let wmin = self.params.db_min_window.max(1);
        if n < wmin {
            return false;
        }
        // GEOMETRY FIRST, then length. The order used to be the other way
        // round -- the length was drawn here and each candidate window re-flipped
        // its own convex/contiguous coin inside `sample_window`. That made a
        // geometry-conditional length impossible to express, and it also let the
        // best-of-`litter_samples` selection compare windows drawn under
        // different geometries. One coin per round fixes both.
        // Pair coin first (docs/NONLOCAL_PHASE_A.md), and only for non-COMP
        // rounds: COMP admits only non-growing spellings, and with both bans
        // armed a commuting pair has no admissible same-length spelling, so a
        // COMP pair round could never splice. p_pair == 0 draws no RNG — the
        // stream is bit-identical to the pair-less chain.
        let geo = if mode != DbMode::Compressing
            && self.params.p_pair > 0.0
            && self.rng.random_bool(self.params.p_pair.clamp(0.0, 1.0))
        {
            DbSample::Pair
        } else if self.rng.random_bool(self.active_p_convex().clamp(0.0, 1.0)) {
            DbSample::Convex
        } else {
            DbSample::Contiguous
        };
        self.db_pair_round = geo == DbSample::Pair;
        if self.db_pair_round {
            self.counters.pair_rounds += 1;
        }
        let wmax = self.active_s_db(geo).max(1);
        // Prefix descent always starts at the top of the range — the descent
        // itself visits every shorter length, so sampling a shorter start
        // would only duplicate coverage.
        let wmax = if self.db_g57_only { self.params.s_db_g57.max(wmin) } else { wmax };
        let descend = self.active_prefixes();
        let len = if geo == DbSample::Pair {
            // A pair window is always the seed plus its partner; drawing a
            // shorter length would degenerate to a plain 1-gate re-spelling.
            wmax.min(n)
        } else if descend || wmin == wmax {
            wmax.min(n)
        } else {
            self.rng.random_range(wmin..=wmax.min(n))
        };

        // Sample the window under the geometry drawn above; g1dir drives the
        // incoming direction pivot below.
        let Some((ids, g1dir, smp)) = self.sample_best_window(len, geo) else {
            return false;
        };
        // Stamped into every --db-record attempt line (smp=ctg|cvx) so stats
        // can split hits by sampler geometry, esp. under --db-sample mixed.
        self.db_last_sampler = smp;
        self.geo_attempts[matches!(smp, DbSample::Convex) as usize] += 1;
        self.db_last_len = ids.len();
        Self::bump_len(&mut self.counters.len_attempts, ids.len(), 1);
        // Litter fragmentation census over the sampled window (observation only).
        let (distinct, full_litter) = self.litter_census(&ids);
        self.counters.litter_windows += 1;
        self.counters.litter_distinct_sum += distinct as u64;
        // Full-litter ban on the MAIN (no-descent) window path — the descent
        // path enforces it per rung and descends past banned rungs instead: a
        // window that is exactly one complete litter is where the store is
        // most likely to hand back the spelling that made it — refuse it.
        if self.params.litter_ban && full_litter && !descend {
            self.counters.litter_banned += 1;
            self.count_db_miss(mode);
            return false;
        }
        let window: Vec<XGate> = ids.iter().map(|&id| self.arena.gate(id).clone()).collect();

        // Prefix descent, largest first: try the full k-gate window, then the
        // (k-1)-gate prefix, and so on down to db_min_window; splice the
        // LONGEST prefix with a usable match (max rewrite per round). Every
        // prefix attempt is recorded and counted. In dry-run the descent runs
        // to the bottom recording hits without splicing (full measurement);
        // live, the first hit splices and ends the round. A span-cap or
        // wide-verify decline keeps descending — shorter prefixes span fewer
        // wires and may still match.
        if descend {
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
            // Store-routing schedule for this descent.
            //
            // ONE-PASS CASCADE: at each window length, probe curated, then
            // fall back to regular before shortening. So a regular hit at
            // length p wins over a curated hit at length p-1 -- the descent
            // can never see the shorter curated rung.
            //
            // --curated-exhaust (two passes): pass 0 walks EVERY length against
            // curated alone; only a descent that came up completely empty runs
            // pass 1 against the regular store. Curated material is preferred
            // at any length over regular material at a longer one. The price is
            // that a full curated miss re-canonicalizes the whole descent, and
            // db_attempts counts both passes -- both are real costs of the
            // policy, so neither is hidden from the counters.
            let cur_ok = self.params.curated
                && (mode != DbMode::Compressing || self.params.curated_in_comp);
            let passes: &[(bool, bool)] = if self.params.curated_exhaust && cur_ok {
                &[(true, false), (false, true)]
            } else {
                &[(cur_ok, true)]
            };
            for &(cur_armed, reg_fb) in passes {
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
                    // The descent must SHRINK on every exit path. Falling
                    // through to `continue` without moving lo/hi respins the
                    // identical prefix forever -- and since a store miss is
                    // the common case, that hangs the run before it completes
                    // a single move.
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
                let res = db_replace(
                    prefix,
                    self.num_wires,
                    &self.db,
                    self.db_budget,
                    mode,
                    guard,
                    cur_armed,
                    self.params.curated_in_comp,
                    reg_fb,
                    self.params.mix_pay_random,
                    self.db_pair_round,
                    &mut self.rng,
                );
                self.counters.db_identity_skips += res.identity_skipped as u64;
                self.counters.pair_perm_skips += res.permutation_skipped as u64;
                if res.chosen.is_some() && res.chosen_curated {
                    self.counters.db_curated_hits += 1;
                }
                if res.chosen.is_some() {
                    self.note_choice(res.choice_count);
                }
                if let Some(ml) = res.min_match_len {
                    self.counters.dmin_windows += 1;
                    if ml < prefix.len() {
                        self.counters.dmin_shorter += 1;
                    }
                }
                if res.degree_skipped {
                    self.counters.db_degree_skips += 1;
                    let k = self.db_last_len;
                    Self::bump_len(&mut self.counters.len_deg_skip, k, 1);
                }
                let Some(replacement) = res.chosen else {
                    self.record_db_attempt(prefix, res.match_count, None);
                    self.count_db_miss(mode);
                    // The descent must SHRINK on every exit path. Falling
                    // through to `continue` without moving lo/hi respins the
                    // identical prefix forever -- and since a store miss is
                    // the common case, that hangs the run before it completes
                    // a single move.
                    if p == wmin {
                        break;
                    }
                    if k.saturating_sub(lo) >= hi.saturating_sub(k) {
                        lo += 1;
                    } else {
                        hi -= 1;
                    }
                    continue;
                };
                if self.params.db_dry_run {
                    self.record_db_attempt(prefix, res.match_count, None);
                    self.count_db_hit(mode);
                    // The descent must SHRINK on every exit path. Falling
                    // through to `continue` without moving lo/hi respins the
                    // identical prefix forever -- and since a store miss is
                    // the common case, that hangs the run before it completes
                    // a single move.
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
            let k = self.db_last_len;
            Self::bump_len(&mut self.counters.len_span_skip, k, 1);
            self.record_db_attempt(&window, 0, None);
            match mode {
                DbMode::Compressing => self.counters.db_comp_misses += 1,
                DbMode::SizeAgnostic | DbMode::MinGrow | DbMode::Mix | DbMode::Stable | DbMode::StableGrow | DbMode::StableLedger | DbMode::Same | DbMode::BandLedger | DbMode::BandShrink | DbMode::BandGrow => self.counters.db_agn_misses += 1,
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
            self.params.curated_in_comp,
            true,
            self.params.mix_pay_random,
            self.db_pair_round,
            &mut self.rng,
        );
        self.counters.db_identity_skips += res.identity_skipped as u64;
        self.counters.pair_perm_skips += res.permutation_skipped as u64;
        if res.chosen.is_some() {
            self.note_choice(res.choice_count);
        }
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
                    DbMode::SizeAgnostic | DbMode::MinGrow | DbMode::Mix | DbMode::Stable | DbMode::StableGrow | DbMode::StableLedger | DbMode::Same | DbMode::BandLedger | DbMode::BandShrink | DbMode::BandGrow => self.counters.db_agn_hits += 1,
                }
            } else {
                match mode {
                    DbMode::Compressing => self.counters.db_comp_misses += 1,
                    DbMode::SizeAgnostic | DbMode::MinGrow | DbMode::Mix | DbMode::Stable | DbMode::StableGrow | DbMode::StableLedger | DbMode::Same | DbMode::BandLedger | DbMode::BandShrink | DbMode::BandGrow => self.counters.db_agn_misses += 1,
                }
            }
            return false;
        }

        let curated_pick = res.chosen_curated;
        let mlen = res.min_match_len;
        let res_fwd_key = res.fwd_key;
        let Some(replacement) = res.chosen else {
            self.record_db_attempt(&window, match_count, None);
            self.count_db_miss(mode);
            return false;
        };
        let fwd_key = res_fwd_key;
        let hit = self.try_db_splice_curated(curated_pick, &ids, g1dir, &window, replacement, match_count, mode);
        if hit {
            self.geo_hits[matches!(self.db_last_sampler, DbSample::Convex) as usize] += 1;
            // True class of a big-pool conversion, via the reference store.
            if curated_pick && match_count > 20 {
                if let (Some(db), Some(k)) = (reference_db(), fwd_key) {
                    if let Some(v) = db.get_regular(&k) {
                        let mut mn = usize::MAX;
                        let mut pos = 0usize;
                        while pos < v.len() {
                            let l = v[pos] as usize;
                            if l == 0 || l % 3 != 0 || pos + 1 + l > v.len() {
                                break;
                            }
                            mn = mn.min(l / 3);
                            pos += 1 + l;
                        }
                        if mn != usize::MAX {
                            self.m123_class_hist[mn.min(31)] += 1;
                        }
                    }
                }
            }
            // Complexity of the converted permutation: the store's minimal
            // matching spelling for this window (the operational "smallest
            // number of gates that computes it").
            if let Some(ml) = mlen {
                self.dmin_success_hist[ml.min(31)] += 1;
            }
        }
        hit
    }

    /// Bump one per-length bucket, growing the vector on demand.
    fn bump_len(v: &mut Vec<u64>, k: usize, by: u64) {
        let k = k.min(LEN_HIST_MAX);
        if v.len() <= LEN_HIST_MAX {
            v.resize(LEN_HIST_MAX + 1, 0);
        }
        v[k] += by;
    }

    fn count_db_hit(&mut self, mode: DbMode) {
        match mode {
            DbMode::Compressing => self.counters.db_comp_hits += 1,
            DbMode::SizeAgnostic | DbMode::MinGrow | DbMode::Mix | DbMode::Stable | DbMode::StableGrow | DbMode::StableLedger | DbMode::Same | DbMode::BandLedger | DbMode::BandShrink | DbMode::BandGrow => self.counters.db_agn_hits += 1,
        }
        let k = self.db_last_len;
        Self::bump_len(&mut self.counters.len_hits, k, 1);
    }

    fn count_db_miss(&mut self, mode: DbMode) {
        match mode {
            DbMode::Compressing => self.counters.db_comp_misses += 1,
            DbMode::SizeAgnostic | DbMode::MinGrow | DbMode::Mix | DbMode::Stable | DbMode::StableGrow | DbMode::StableLedger | DbMode::Same | DbMode::BandLedger | DbMode::BandShrink | DbMode::BandGrow => self.counters.db_agn_misses += 1,
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
            let _vt = std::time::Instant::now();
            // Past verify_rewrite's 24-wire exhaustive ceiling, verify by ANF
            // instead of by evaluation: cost is bounded by the polynomial term
            // count rather than 2^support, and it is still a proof. Measured
            // on the production store, the >= 24-wire slice is 1.3% of entries
            // (all 10-11 gates), max degree 7, and carries at most 233 terms --
            // so this is a comparison where the exhaustive check would have
            // been 16.7M evaluations. Only an UNDECIDED result (support > 64
            // wires, or a budget hit) still declines.
            let _vok = if support.len() > 24 {
                match super::db_replace::polys_equivalent(window, &replacement, self.db_budget) {
                    Some(ok) => {
                        self.counters.db_wide_poly += 1;
                        ok
                    }
                    None => {
                        self.counters.db_wide_skip += 1;
                        self.record_db_attempt(window, match_count, None);
                        self.count_db_miss(mode);
                        return false;
                    }
                }
            } else {
                rules::verify_rewrite(window, &replacement)
            };
            super::xpoly::VERIFY_NS.fetch_add(
                _vt.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            if !_vok {
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
        // Ledger flows for the stable family (only ±1 by construction).
        if matches!(mode, DbMode::Stable | DbMode::StableGrow) {
            if new > old {
                self.stable_led_added += (new - old) as u64;
            } else if new < old {
                self.stable_led_removed += (old - new) as u64;
            }
        }
        // band-ledger drift accounting (signed, over every band conversion).
        if self.band_led_round {
            self.band_led += new as i64 - old as i64;
        }
        // Big-pool (M1/M2/M3) usage: only the swapped pools carry more than
        // the bounded contract's 20 candidates, so this count is exact.
        if from_curated && match_count > 20 {
            self.bigpool_hits += 1;
        }
        if new <= old {
            Self::bump_len(&mut self.counters.len_removed, old, (old - new) as u64);
            self.counters.db_gates_removed += (old - new) as u64;
            if mode == DbMode::Compressing {
                self.counters.db_cmp_removed += (old - new) as u64;
            } else {
                self.counters.db_mix_removed += (old - new) as u64;
            }
        } else {
            Self::bump_len(&mut self.counters.len_added, old, (new - old) as u64);
            self.counters.db_gates_added += (new - old) as u64;
            if mode == DbMode::Compressing {
                self.counters.db_cmp_added += (new - old) as u64;
            } else {
                self.counters.db_mix_added += (new - old) as u64;
            }
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
            self.evict_taps(id);
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
            if from_curated {
                if self.counters.splice_sizes_curated.is_empty() {
                    self.counters.splice_sizes_curated =
                        vec![vec![0u64; SPLICE_HIST_MAX + 1]; SPLICE_HIST_MAX + 1];
                }
                self.counters.splice_sizes_curated[o][i] += 1;
            }
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
            DbSample::Pair => "pair",
            DbSample::Bridge => "brg",
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
        let mut cur = self.arena.neighbor(id, dir);
        let mut d = 0usize;
        while cur != NIL && d < cap && !self.arena.collides_ids(id, cur) {
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
        // Scan with borrows only (no gate clone); mutate after the scan scope.
        let (last, steps) = {
            let mut last = NIL;
            let mut cur = self.arena.neighbor(id, dir);
            let mut steps = 0usize;
            while cur != NIL && cur != stop && !self.arena.collides_ids(id, cur) {
                last = cur;
                steps += 1;
                cur = self.arena.neighbor(cur, dir);
            }
            (last, steps)
        };
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
        let mut cur = lo;
        loop {
            if self.arena.collides_ids(cur, x) {
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
    fn sample_best_window(&mut self, w: usize, geo: DbSample) -> Option<(Vec<u32>, Dir, DbSample)> {
        let n = self.params.litter_samples.max(1);
        let mut best: Option<(Vec<u32>, Dir, DbSample)> = None;
        let mut best_distinct = 0usize;
        for _ in 0..n {
            let Some(cand) = self.sample_window(w, geo) else { continue };
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

    /// Collect one candidate window under an ALREADY-CHOSEN geometry. The coin
    /// lives in `db_attempt_inner` now (see `DbSample`).
    fn sample_window(&mut self, w: usize, geo: DbSample) -> Option<(Vec<u32>, Dir, DbSample)> {
        match geo {
            DbSample::Contiguous => {
                self.collect_contiguous(w).map(|(ids, d)| (ids, d, DbSample::Contiguous))
            }
            DbSample::Convex => {
                self.collect_convex(w).map(|(ids, d)| (ids, d, DbSample::Convex))
            }
            DbSample::Pair => {
                self.collect_pair().map(|(ids, d)| (ids, d, DbSample::Pair))
            }
            // Bridge is not a sampler: its two endpoint windows are built by
            // bridge_round directly. The geometry coin never draws it.
            DbSample::Bridge => None,
        }
    }

    /// Rebuild the generation pool: the `pool_k` lowest-generation gates among
    /// those that are pool-eligible AND still below the goal. An O(size) scan,
    /// so it runs on the `gen_rescan` cadence rather than every round.
    fn rebuild_pool(&mut self) {
        let target = self.params.gen_target;
        self.pool.clear();
        let mut cands: Vec<(u32, u32)> = Vec::new();
        // Walk the linked list directly (same order as ids_in_order) without
        // materializing the O(size) id vector.
        let mut id = self.arena.head();
        while id != NIL {
            let m = self.meta_of(id);
            if m.dgen < target && self.pool_eligible(id) {
                cands.push((m.dgen, id));
            }
            id = self.arena.neighbor(id, Dir::R);
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

    // Pair: fuse the seed with a far COMMUTING partner (docs/NONLOCAL_PHASE_A.md).
    // Scan the seed's commutation box — the gates it could float past in its
    // own direction, out to the first collider or pair_scan_cap — then float
    // the seed adjacent to the chosen partner and return the fused 2-gate
    // window. Every crossed gate commutes with the seed (that is what the scan
    // checked), so the relocation is function-preserving like every other
    // float, and a later miss is walked home by db_attempt's usual
    // restore_seed. The other samplers cannot produce this window: Convex
    // absorbs only COLLIDERS (commuting gates are hopped past), and Contiguous
    // pairs commuting gates only when they already sit at distance 1.
    fn collect_pair(&mut self) -> Option<(Vec<u32>, Dir)> {
        let g1 = self.pick_seed()?;
        let dir1 = self.meta_of(g1).dir;
        let cap = self.params.pair_scan_cap.max(1);
        // Read-only box scan: partner candidates are window-eligible gates the
        // seed commutes with, recorded with their hop distance (gates crossed
        // to reach them).
        let mut cands: Vec<(u32, u64)> = Vec::new();
        let mut cur = self.arena.neighbor(g1, dir1);
        let mut hops = 0u64;
        while cur != NIL && (hops as usize) < cap && !self.arena.collides_ids(g1, cur) {
            if self.window_eligible(cur) {
                cands.push((cur, hops));
            }
            hops += 1;
            cur = self.arena.neighbor(cur, dir1);
        }
        if cur != NIL && hops as usize >= cap {
            self.counters.pair_scan_truncs += 1;
        }
        if cands.is_empty() {
            self.counters.pair_boxes_empty += 1;
            return None;
        }
        let (g2, dist) = if self.params.pair_pick_uniform {
            cands[self.rng.random_range(0..cands.len())]
        } else {
            *cands.last().unwrap()
        };
        // Fuse: float the seed adjacent to the partner. The scan cleared the
        // path, so float_until stops with the partner as the next neighbor.
        self.float_until(g1, dir1, g2);
        if self.arena.neighbor(g1, dir1) != g2 {
            // Unreachable while the scan and float_until agree on collides();
            // decline defensively rather than fuse a wrong window.
            self.counters.pair_boxes_empty += 1;
            return None;
        }
        self.counters.pair_fused += 1;
        self.counters.pair_box_sum += hops;
        self.counters.pair_box_max = self.counters.pair_box_max.max(hops);
        self.counters.pair_dist_sum += dist;
        self.counters.pair_dist_max = self.counters.pair_dist_max.max(dist);
        // Window ids in link order, leftmost first.
        let ids = match dir1 {
            Dir::R => vec![g1, g2],
            Dir::L => vec![g2, g1],
        };
        Some((ids, dir1))
    }

    // ---- bridge fusion (docs/NONLOCAL_PHASE_A.md) ----
    //
    // Jointly re-encode two gates that commutation CANNOT bring together.
    // Any correct two-site rewrite is X = g1·P at the left site and
    // conj_M(P⁻¹)·g2 at the right (M the interior); this move takes P = one
    // 2-control conjunction carrier u and realises the conjugation by
    // ADJUSTING the interior — every interior collider h becomes its exact
    // conjugate u·h·u = [h, correction(s)] (see conj_wake) — while the two
    // carrier copies land adjacent to g1 and g2 and the store re-spells both
    // fused windows. Telescoping makes the whole move exact:
    //   g1·u·(u·M·u)·u·g2 = g1·M·g2.
    // The carrier collides with both endpoints by construction (u reads t_g1;
    // g2 reads t_u), so the two respelled sites are correlated through u —
    // one joint replacement whose halves only compose to the original
    // through the shared carrier. Corrections are conjunction gates outside
    // strict g57 form (the polf trade the legacy twist packets already make);
    // their count is bounded by bridge_max_colliders and metered.
    fn bridge_plan(&mut self) -> Option<BridgePlan> {
        let min_span = self.params.bridge_min_span.max(1);
        let max_span = self.params.bridge_max_span.max(min_span);
        if self.arena.len() < min_span + 2 {
            return None;
        }
        let mut g1 = NIL;
        for _ in 0..8 {
            let g = self.arena.random_linked(&mut self.rng);
            if self.window_eligible(g) {
                g1 = g;
                break;
            }
        }
        if g1 == NIL {
            return None;
        }
        // Log-uniform interior length: the all-scales dial, like the twist.
        let span = {
            let lo = min_span as f64;
            let hi = max_span as f64;
            (lo * (hi / lo).powf(self.rng.random::<f64>())).round() as usize
        }
        .clamp(min_span, max_span);
        // Walk the interior, counting per-wire readers and writers so the
        // carrier can be sited where it collides least.
        let nw = self.num_wires;
        let mut rd = vec![0u32; nw];
        let mut wr = vec![0u32; nw];
        let mut interior: Vec<u32> = Vec::with_capacity(span);
        let mut cur = self.arena.neighbor(g1, Dir::R);
        while cur != NIL && interior.len() < span {
            let g = self.arena.gate(cur);
            wr[g.target as usize] += 1;
            for &(w, _) in &g.ctrls {
                rd[w as usize] += 1;
            }
            interior.push(cur);
            cur = self.arena.neighbor(cur, Dir::R);
        }
        if cur == NIL {
            self.counters.bridge_short += 1;
            return None;
        }
        let g2 = cur;
        if !self.window_eligible(g2) {
            self.counters.bridge_short += 1;
            return None;
        }
        let g1g = self.arena.gate(g1).clone();
        let g2g = self.arena.gate(g2).clone();
        // Carrier u = (t_u; x ∧ y): x on t_g1 (u reads g1's target — window 1
        // is dependency-connected), t_u = g2's least-read control wire (g2
        // reads t_u — window 2 is connected), y = a least-written other wire
        // (fewest interior colliders). Polarities random.
        let xw = g1g.target;
        let Some(tu) = g2g
            .ctrls
            .iter()
            .map(|&(w, _)| w)
            .filter(|&w| w != xw)
            .min_by_key(|&w| rd[w as usize])
        else {
            self.counters.bridge_refused += 1;
            return None;
        };
        let mut yw: Option<u16> = None;
        for w in 0..nw as u16 {
            if w == xw || w == tu {
                continue;
            }
            if yw.is_none_or(|b| wr[w as usize] < wr[b as usize]) {
                yw = Some(w);
            }
        }
        let Some(yw) = yw else {
            self.counters.bridge_refused += 1;
            return None;
        };
        let xp = self.rng.random_bool(0.5);
        let yp = self.rng.random_bool(0.5);
        let Some(u) = XGate::conj(tu, [(xw, xp), (yw, yp)]) else {
            self.counters.bridge_refused += 1;
            return None;
        };
        // The wake: exact conjugates for every interior collider, each one
        // verified exhaustively over its own support before anything mutates.
        let mut wake: Vec<(u32, Vec<XGate>)> = Vec::new();
        let mut colliders = 0usize;
        for &hid in &interior {
            let h = self.arena.gate(hid).clone();
            let Some(corrs) = conj_wake(&u, &h, self.params.k_max) else {
                self.counters.bridge_refused += 1;
                return None;
            };
            if XGate::collides(&u, &h) {
                colliders += 1;
                if colliders > self.params.bridge_max_colliders {
                    self.counters.bridge_refused += 1;
                    return None;
                }
                let mut after = vec![h.clone()];
                after.extend(corrs.iter().cloned());
                if !rules::verify_rewrite(&[u.clone(), h.clone(), u.clone()], &after) {
                    debug_assert!(false, "conj_wake produced a wrong conjugate: {u:?} x {h:?}");
                    self.counters.bridge_refused += 1;
                    return None;
                }
            }
            if !corrs.is_empty() {
                wake.push((hid, corrs));
            }
        }
        Some(BridgePlan { g1, g2, g1g, g2g, u, wake, interior_len: interior.len() })
    }

    /// Commit the insertions of a bridge plan: the wake corrections (each
    /// immediately after its collider, which it commutes with) and the two
    /// carrier copies (after g1, before g2). Function-preserving by the
    /// telescoping identity; every inserted gate is returned so a declined
    /// far-window splice can roll the circuit back exactly.
    fn bridge_insert(&mut self, plan: &BridgePlan) -> (u32, u32, Vec<u32>) {
        let ev = self.fresh_event();
        let mut inserted: Vec<u32> = Vec::new();
        let mut stamp = |mx: &mut Self, id: u32| {
            mx.index_add(id);
            let d = mx.rand_dir();
            let lit = mx.fresh_litter();
            mx.set_meta(
                id,
                Meta {
                    origin: ORIGIN_SYNTH,
                    event: ev,
                    dir: d,
                    dgen: GEN_FRESH,
                    litter: lit,
                    litter_size: 1,
                },
            );
        };
        // Counters (width_hist, bridge_wake_sum) are NOT bumped here: the wake
        // is only permanent once the far window splices. bridge_round tallies
        // them after ok2, so a rolled-back insertion leaves no metering trace.
        for (hid, corrs) in &plan.wake {
            for c in corrs {
                let id = self.arena.insert_after(*hid, c.clone());
                stamp(self, id);
                inserted.push(id);
            }
        }
        let u2 = self.arena.insert_after(self.arena.neighbor(plan.g2, Dir::L), plan.u.clone());
        stamp(self, u2);
        inserted.push(u2);
        let u1 = self.arena.insert_after(plan.g1, plan.u.clone());
        stamp(self, u1);
        inserted.push(u1);
        (u1, u2, inserted)
    }

    fn bridge_round(&mut self) {
        self.counters.bridge_rounds += 1;
        let Some(plan) = self.bridge_plan() else {
            return;
        };
        // Probe BOTH endpoint windows before anything mutates: a store miss
        // leaves no trace at all. The reorder ban is armed so a carrier that
        // happens to commute with an endpoint cannot splice trivially.
        let guard = DegreeGuard {
            max_degree: self.params.db_max_degree,
            probes: self.params.db_degree_probes,
        };
        let w1 = [plan.g1g.clone(), plan.u.clone()];
        let w2 = [plan.u.clone(), plan.g2g.clone()];
        self.counters.db_attempts += 2;
        let p1 = db_replace(
            &w1,
            self.num_wires,
            &self.db,
            self.db_budget,
            DbMode::Mix,
            guard,
            self.params.curated,
            self.params.curated_in_comp,
            true,
            self.params.mix_pay_random,
            true,
            &mut self.rng,
        );
        if p1.chosen.is_none() {
            self.counters.bridge_probe_miss += 1;
            return;
        }
        let p2 = db_replace(
            &w2,
            self.num_wires,
            &self.db,
            self.db_budget,
            DbMode::Mix,
            guard,
            self.params.curated,
            self.params.curated_in_comp,
            true,
            self.params.mix_pay_random,
            true,
            &mut self.rng,
        );
        if p2.chosen.is_none() {
            self.counters.bridge_probe_miss += 1;
            return;
        }
        let (u1, u2, inserted) = self.bridge_insert(&plan);
        // Far window first: a decline there rolls back to the exact
        // pre-insert circuit (every inserted gate is still bare).
        self.db_last_sampler = DbSample::Bridge;
        self.db_last_len = 2;
        let w2v = vec![plan.u.clone(), plan.g2g.clone()];
        self.counters.db_attempts += 1;
        let d2 = self.meta_of(u2).dir;
        let r2 = db_replace(
            &w2v,
            self.num_wires,
            &self.db,
            self.db_budget,
            DbMode::Mix,
            guard,
            self.params.curated,
            self.params.curated_in_comp,
            true,
            self.params.mix_pay_random,
            true,
            &mut self.rng,
        );
        let ok2 = match r2.chosen {
            Some(rep) => self.try_db_splice_curated(
                r2.chosen_curated,
                &[u2, plan.g2],
                d2,
                &w2v,
                rep,
                r2.match_count,
                DbMode::Mix,
            ),
            None => false,
        };
        if !ok2 {
            for &id in inserted.iter().rev() {
                self.index_remove(id);
                self.arena.unlink(id);
                self.arena.free_node(id);
            }
            self.counters.bridge_rollbacks += 1;
            return;
        }
        // The far window committed, so the wake is now permanent — tally it
        // here (not in bridge_insert) so a rolled-back insertion never leaves
        // a metering trace, matching bridge_span_sum / bridge_colliders_sum.
        for (_hid, corrs) in &plan.wake {
            for c in corrs {
                self.counters.width_hist[c.width().min(15)] += 1;
                self.counters.bridge_wake_sum += 1;
            }
        }
        // Near window. On the rare post-insert miss the bare carrier stays —
        // it is load-bearing now (site 2 already computes u·g2) and the
        // circuit is still exact.
        self.db_last_sampler = DbSample::Bridge;
        self.db_last_len = 2;
        let w1v = vec![plan.g1g.clone(), plan.u.clone()];
        self.counters.db_attempts += 1;
        let d1 = self.meta_of(plan.g1).dir;
        let r1 = db_replace(
            &w1v,
            self.num_wires,
            &self.db,
            self.db_budget,
            DbMode::Mix,
            guard,
            self.params.curated,
            self.params.curated_in_comp,
            true,
            self.params.mix_pay_random,
            true,
            &mut self.rng,
        );
        let ok1 = match r1.chosen {
            Some(rep) => self.try_db_splice_curated(
                r1.chosen_curated,
                &[plan.g1, u1],
                d1,
                &w1v,
                rep,
                r1.match_count,
                DbMode::Mix,
            ),
            None => false,
        };
        if ok1 {
            self.counters.bridge_committed += 1;
        } else {
            self.counters.bridge_half += 1;
        }
        self.counters.bridge_span_sum += plan.interior_len as u64;
        self.counters.bridge_span_max =
            self.counters.bridge_span_max.max(plan.interior_len as u64);
        self.counters.bridge_colliders_sum += plan.wake.len() as u64;
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

    /// GLOBAL RE-RANDOMISATION: float every gate to a uniformly random position
    /// inside its own commutation bounds.
    ///
    /// This is the whole-circuit analogue of the per-splice birth advance. Where
    /// `--db-advance` scatters one litter at birth, this re-draws the position of
    /// every gate at once, so positional structure inherited from the input --
    /// which the ancestry instruments show is what survives DB mixing -- is
    /// attacked directly rather than incidentally.
    ///
    /// Function preservation is free: each step is a float past non-colliding
    /// gates, i.e. a commutation, so no verification is needed. `float_uniform`
    /// picks the offset over the gate's full [left, right] slack, so a gate with
    /// no slack simply stays put.
    ///
    /// Cost is O(sum of per-gate slack), which is why the round rate is scaled
    /// as 1/|circuit|: the expected work per round stays O(mean slack) no matter
    /// how large the circuit grows.
    // GLOBAL re-randomisation: re-place EVERY gate uniformly inside its own
    // commutation bounds. This is exactly the terminal float applied mid-run,
    // so it is semantics-preserving (float_distance never crosses a
    // non-commuting neighbour) and size-preserving -- it moves gates and
    // nothing else. The walk is sequential, so a gate's bounds already
    // reflect the earlier gates' new positions; that is the same law
    // final_float has always used.
    fn global_shuffle(&mut self) {
        let t0 = std::time::Instant::now();
        let (moved, disp) = self.final_float();
        self.counters.shuffle_ns += t0.elapsed().as_nanos() as u64;
        self.counters.shuffles += 1;
        self.counters.shuffle_moved += moved;
        self.counters.shuffle_steps += disp;
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
        self.evict_taps(id);
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
        // Canary re-anchoring: evict g while it is linked, but evict h only
        // AFTER g's unlink — with both still linked, h's left neighbor can BE
        // the dying g (dir R, no CollidingIntact), which would strand h's
        // taps on a freed slot. After the unlink the neighbor pointers are
        // repaired, so the eviction target is always a survivor. h keeps its
        // node (and taps) when the rewrite carries a CollidingIntact.
        let h_stays = seq.iter().any(|&(_, r)| r == Role::CollidingIntact);
        self.evict_taps(g_id);
        self.index_remove(g_id);
        self.arena.unlink(g_id);
        if !h_stays {
            self.evict_taps(h_id);
        }
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

    /// One arena pass for all three counts. Folded rather than split into
    /// separate walks: `report` already makes ten full pointer-chases over the
    /// circuit, and the gate is dereferenced here anyway.
    pub fn g57_census(&self) -> G57Census {
        let mut out = G57Census::default();
        let mut cur = self.arena.head();
        while cur != NIL {
            let g = self.arena.gate(cur);
            if g.comp && g.ctrls.len() == 2 {
                out.shaped += 1;
                if g.ctrls[0].1 == g.ctrls[1].1 {
                    out.same_pol += 1;
                } else {
                    out.opp_pol += 1;
                }
            }
            cur = self.arena.neighbor(cur, Dir::R);
        }
        out
    }

    /// True g57 shape: `comp = 1` with exactly two controls of OPPOSITE
    /// polarity (`a ^= b OR !c`). Distinct from `remaining_g57`, which counts
    /// every `comp = 1` gate regardless of width -- the report's `comp=` field.
    pub fn true_g57(&self) -> usize {
        self.g57_census().opp_pol
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
    pub fn splice_size_line_curated(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        for (o, row) in self.counters.splice_sizes_curated.iter().enumerate() {
            for (i, &c) in row.iter().enumerate() {
                if c > 0 {
                    parts.push(format!("{o}->{i}:{c}"));
                }
            }
        }
        parts.join(" ")
    }

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
        let g57c = self.g57_census();
        let c = &self.counters;
        let hist: Vec<String> = (0..=self.params.k_max.min(15))
            .map(|w| format!("{}:{}", w, c.width_hist[w]))
            .collect();
        println!(
            "[fmix] mv={} size={} target={} comp={} g57={} shaped={} polf={:.3} | merges c={} x={} d={} s={} a={} sib={} xorig={} tabu={} nopart={} wall={} far={} noadj={} | undo ok={} dead={} tabu={} miss={} live={} | db pdb={:.3} slot2={}/{} sadd={} comp={}/{} agn={}/{} rm={} add={} wide={} wpoly={} dsk={} ssk={} bab={} idsk={} cur={}/{} g57only={}/{} sled={}/{} m123={} bled={} | expand r1={} r2={} r3={} pre={} fresh={} unsub={} ins={} tn1={} tsw={} tn2={} twrel={} twsplit={} twspan={} twskip={} shuf={} shufmv={} shufst={} shufms={} | declined={} blockw={} dl={} bnd={} | floats={}/{} scat={}/{} | disp={:.4} owin={:.1} fan0={:.3} leew={:.0} odiff={:.4} oadj={:.4} osyn={:.3} anc={:.1} ancspan={:.3} width[{}] | gen tgt={} G={} Gall={} tgtbl={} alag={}/{} lag={}/{} wlag={} min={} cov={:.1} canary={:.3} cft={} | litter distinct={:.2} full={} ban={} tplace={}/{} dmin={:.3} dminw={} canon[poly={}ms canon={}ms calls={}] verify={}ms degprobe={}ms/{} | choice n={} multi={:.3} mean={:.2} bits/splice={:.3}",
            c.moves,
            self.arena.len(),
            self.params.target_size,
            self.remaining_g57(),
            g57c.opp_pol,
            g57c.shaped,
            g57c.pol_flipped(),
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
            c.db_slot2_hits,
            c.db_slot2_rounds,
            c.db_slot2_added,
            c.db_comp_hits,
            c.db_comp_misses,
            c.db_agn_hits,
            c.db_agn_misses,
            c.db_gates_removed,
            c.db_gates_added,
            c.db_wide_skip,
            c.db_wide_poly,
            c.db_degree_skips,
            c.db_span_skips,
            c.db_build_aborts,
            c.db_identity_skips,
            c.db_curated_hits,
            c.db_curated_rejected,
            c.db_g57_hits,
            c.db_g57_rounds,
            self.stable_led_added,
            self.stable_led_removed,
            self.bigpool_hits,
            self.band_led,
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
            c.shuffles,
            c.shuffle_moved,
            c.shuffle_steps,
            c.shuffle_ns / 1_000_000,
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
            },
            // The DENOMINATOR alongside the ratio: windows for which the
            // store held any non-identical equivalent. Without it a moving
            // dmin cannot be told from a moving sample.
            c.dmin_windows,
            super::xpoly::POLY_NS.load(std::sync::atomic::Ordering::Relaxed) / 1_000_000,
            super::xpoly::CANON_NS.load(std::sync::atomic::Ordering::Relaxed) / 1_000_000,
            super::xpoly::CANON_CALLS.load(std::sync::atomic::Ordering::Relaxed),
            super::xpoly::VERIFY_NS.load(std::sync::atomic::Ordering::Relaxed) / 1_000_000,
            super::xpoly::DEGREE_NS.load(std::sync::atomic::Ordering::Relaxed) / 1_000_000,
            super::xpoly::DEGREE_CALLS.load(std::sync::atomic::Ordering::Relaxed),
            // Selection entropy: how often a successful splice had a real
            // choice, how wide that choice was, and the bits it injected.
            c.choice_splices,
            if c.choice_splices > 0 {
                c.choice_multi as f64 / c.choice_splices as f64
            } else {
                0.0
            },
            if c.choice_splices > 0 {
                c.choice_sum as f64 / c.choice_splices as f64
            } else {
                0.0
            },
            if c.choice_splices > 0 {
                c.choice_bits_milli as f64 / 1000.0 / c.choice_splices as f64
            } else {
                0.0
            }
        );
        // Per-success complexity histogram (store-minimal spelling length of
        // each converted permutation); printed whenever any conversions exist.
        let hs: Vec<String> = self
            .dmin_success_hist
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c > 0)
            .map(|(l, &c)| format!("{l}:{c}"))
            .collect();
        if !hs.is_empty() {
            println!("[fmix] dminh mv={} {}", self.moves_done, hs.join(" "));
        }
        let ms: Vec<String> = self
            .m123_class_hist
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c > 0)
            .map(|(l, &c)| format!("M{l}:{c}"))
            .collect();
        if !ms.is_empty() {
            println!("[fmix] m123class mv={} {}", self.moves_done, ms.join(" "));
        }
        if self.geo_attempts[0] + self.geo_attempts[1] > 0 {
            let rate = |h: u64, a: u64| if a > 0 { 100.0 * h as f64 / a as f64 } else { 0.0 };
            println!(
                "[fmix] geo mv={} ctg={}/{} ({:.2}%) cvx={}/{} ({:.2}%)",
                self.moves_done,
                self.geo_hits[0], self.geo_attempts[0], rate(self.geo_hits[0], self.geo_attempts[0]),
                self.geo_hits[1], self.geo_attempts[1], rate(self.geo_hits[1], self.geo_attempts[1]),
            );
        }
        // Pair-geometry meters (docs/NONLOCAL_PHASE_A.md), only when armed.
        if self.params.p_pair > 0.0 {
            let fused = c.pair_fused.max(1) as f64;
            println!(
                "[fmix] pair rounds={} fused={} splices={} permskip={} empty={} trunc={} box avg={:.1} max={} dist avg={:.1} max={}",
                c.pair_rounds,
                c.pair_fused,
                c.pair_splices,
                c.pair_perm_skips,
                c.pair_boxes_empty,
                c.pair_scan_truncs,
                c.pair_box_sum as f64 / fused,
                c.pair_box_max,
                c.pair_dist_sum as f64 / fused,
                c.pair_dist_max,
            );
        }
        // Bridge-fusion meters (docs/NONLOCAL_PHASE_A.md), only when armed.
        if self.params.p_bridge > 0.0 {
            let commits = (c.bridge_committed + c.bridge_half).max(1) as f64;
            println!(
                "[fmix] bridge rounds={} committed={} half={} rollback={} probemiss={} refused={} short={} span avg={:.1} max={} colliders avg={:.2} wake={}",
                c.bridge_rounds,
                c.bridge_committed,
                c.bridge_half,
                c.bridge_rollbacks,
                c.bridge_probe_miss,
                c.bridge_refused,
                c.bridge_short,
                c.bridge_span_sum as f64 / commits,
                c.bridge_span_max,
                c.bridge_colliders_sum as f64 / commits,
                c.bridge_wake_sum,
            );
        }
        let anc = self.anc_report();
        if !anc.is_empty() {
            println!("{anc}");
        }
        let ga = self.gen_anc_report();
        if !ga.is_empty() {
            println!("{ga}");
        }
        let sizes = self.splice_size_line();
        if !sizes.is_empty() {
            println!("[fmix] splice sizes out->in: {sizes}");
        }
        let csizes = self.splice_size_line_curated();
        if !csizes.is_empty() {
            println!("[fmix] splice sizes (curated) out->in: {csizes}");
        }
        if let Some(p) = &self.prof {
            let s_star =
                prof_target(self.params.prof_n, self.params.prof_r, p.s_in, p.eff);
            println!(
                "[fmix] profile: phase={} eff={:.2} size={} S*={:.0} pmix={:.3} ghat={:+.4} shat={:+.4} dhat={:+.4} integ={:+.3} sat={}",
                p.phase,
                p.eff,
                self.arena.len(),
                s_star,
                p.pmix,
                p.ghat,
                p.shat,
                p.dhat,
                p.integ,
                p.sat
            );
        }
        {
            // PER-OUTGOING-LENGTH breakdown: the rates the headline line
            // reports are averages over a uniform length draw in 1..s_db, so
            // they conflate "this width works" with "this width was sampled".
            // This says what each width actually did.
            let c = &self.counters;
            let n = c.len_attempts.len();
            if n > 0 && c.len_attempts.iter().sum::<u64>() > 0 {
                println!(
                    "[fmix] per-length: len attempts hits hit% removed added net span_skip deg_skip"
                );
                let g = |v: &Vec<u64>, k: usize| v.get(k).copied().unwrap_or(0);
                for k in 1..n {
                    let a = c.len_attempts[k];
                    if a == 0 {
                        continue;
                    }
                    let h = g(&c.len_hits, k);
                    let rm = g(&c.len_removed, k);
                    let ad = g(&c.len_added, k);
                    println!(
                        "[fmix] len {k:>3} {a:>9} {h:>8} {:>6.2} {rm:>8} {ad:>6} {:>+6} {:>10} {:>9}",
                        100.0 * h as f64 / a as f64,
                        rm as i64 - ad as i64,
                        g(&c.len_span_skip, k),
                        g(&c.len_deg_skip, k)
                    );
                }
            }
        }
        if self.params.twist_g57 {
            let c = &self.counters;
            let us = if c.tg_solves > 0 {
                c.tg_solve_ns as f64 / 1000.0 / c.tg_solves as f64
            } else {
                0.0
            };
            let hist: Vec<String> = c.tg_net_hist.iter().map(|v| v.to_string()).collect();
            println!(
                "[fmix] twist-g57: consumed={} emitted={} net/seam[{}] solves={} avg_us={:.1} slides={} retries={}",
                c.tg_consumed,
                c.tg_emitted,
                hist.join(","),
                c.tg_solves,
                us,
                c.tg_slides,
                c.tg_retries
            );
        }
    }

    /// Install per-gate litter ids (circuit order) from an external stage —
    /// e.g. the SGDB substitution, where each replaced gate's block becomes
    /// one litter. litter_size is recomputed per id; next_litter continues
    /// above the max.
    pub fn load_litters(&mut self, ids: &[u64]) {
        let order = self.arena.ids_in_order();
        assert_eq!(order.len(), ids.len(), "litter sidecar length != gate count");
        let mut sizes: std::collections::HashMap<u64, u16> = std::collections::HashMap::new();
        for &l in ids {
            *sizes.entry(l).or_insert(0) += 1;
        }
        for (&id, &l) in order.iter().zip(ids.iter()) {
            let mut m = self.meta_of(id);
            m.litter = l;
            m.litter_size = sizes[&l];
            self.set_meta(id, m);
        }
        self.next_litter = ids.iter().copied().max().unwrap_or(0) + 1;
        println!(
            "[fmix] litters loaded: {} gates, {} litters, largest {}",
            ids.len(),
            sizes.len(),
            sizes.values().max().copied().unwrap_or(0)
        );
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

    #[test]
    fn prof_target_traces_the_three_phases() {
        let n = [5.0, 20.0, 40.0];
        let r = [4.0, 2.0];
        let s = 1000.0;
        assert!((prof_target(n, r, s, 0.0) - 1000.0).abs() < 1e-9); // start = input
        assert!((prof_target(n, r, s, 2.5) - 2500.0).abs() < 1e-6); // mid-ramp1
        assert!((prof_target(n, r, s, 5.0) - 4000.0).abs() < 1e-6); // top of expand
        assert!((prof_target(n, r, s, 12.0) - 4000.0).abs() < 1e-6); // hold
        assert!((prof_target(n, r, s, 20.0) - 4000.0).abs() < 1e-6); // hold end
        assert!((prof_target(n, r, s, 30.0) - 3000.0).abs() < 1e-6); // mid-ramp3
        assert!((prof_target(n, r, s, 40.0) - 2000.0).abs() < 1e-6); // bottom
        assert!((prof_target(n, r, s, 99.0) - 2000.0).abs() < 1e-6); // after
        // monotone non-decreasing over [0,n0], flat over hold, non-increasing after n1
        let up = (0..=50).map(|i| prof_target(n, r, s, i as f64 * 0.1)).collect::<Vec<_>>();
        assert!(up.windows(2).all(|w| w[1] >= w[0] - 1e-9));

        // R1 = 1 (pure hold) with a sub-x1 compression end: flat at the input
        // through N1, then linear down to half size.
        let n = [1.0, 30.0, 35.0];
        let r = [1.0, 0.5];
        assert!((prof_target(n, r, s, 0.5) - 1000.0).abs() < 1e-9);
        assert!((prof_target(n, r, s, 15.0) - 1000.0).abs() < 1e-9);
        assert!((prof_target(n, r, s, 32.5) - 750.0).abs() < 1e-6);
        assert!((prof_target(n, r, s, 35.0) - 500.0).abs() < 1e-9);
        assert!((prof_target(n, r, s, 99.0) - 500.0).abs() < 1e-9);
    }

    // The controller must actually track a moderate, feasible profile on a
    // real circuit: expand to ~4x, hold, compress toward ~2x. Asserts the
    // phase peaks/troughs land near the targets (best-effort tolerance) and
    // that the phase machine advances all the way to phase 4.
    #[test]
    fn profile_controller_tracks_a_feasible_ramp() {
        let gates = random_mixed_circuit(41, 16, 400);
        let s_in = gates.len() as f64;
        let params = MixParams {
            k_max: 6,
            moves: 4_000_000,
            temp: 20.0,
            p_db: 1.0,
            p_comp: 1.0,
            p_any: 0.1,
            s_db: 5,
            db_min_window: 0,
            p_convex: 0.5,
            mix_pay_random: true,
            prof_n: [6.0, 18.0, 34.0],
            prof_r: [4.0, 2.0],
            prof_cadence_eff: 0.5,
            report_every: u64::MAX,
            verify_every: 200_000,
            seed: 3,
            ..MixParams::default()
        };
        // No FROZEN_DB_DIR in tests -> empty store, every DB lookup misses, so
        // the plant has ZERO authority and the controller cannot move size.
        // This test therefore exercises the controller MATH and phase machine
        // against a null plant: it must still advance phases on the eff marks
        // and never panic / never leave p_mix out of [0,1].
        let mut mx = Mixer::new_with_db(gates, 16, params, FrozenDb::empty());
        let stop = mx.run();
        let p = mx.prof.as_ref().expect("profile armed");
        assert_eq!(p.phase, 4, "phase machine must reach compress-done");
        assert!(p.pmix >= 0.0 && p.pmix <= 1.0, "lever stayed in range");
        assert!(matches!(stop, MixStop::ProfileDone), "a finished profile ends the run");
        // A null plant cannot grow, so the expansion leg saturates (flagged,
        // not hidden) and the compression leg is trivially already-arrived:
        // size <= R2 * input the moment phase 3 opens, which is exactly when
        // the contract says that leg is done.
        assert!(p.sat > 0, "saturation must be tracked when the lever cannot move size");
        assert!(mx.arena.len() as f64 <= 2.0 * s_in, "ended at or under the R2 setpoint");
        assert!(p.eff >= 18.0, "ran through the hold phase before finishing");
        mx.global_check();
    }

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

    // The mask-algebra collides predicate (arena.rs GateMask/mask_collides,
    // behind Arena::collides_ids) must equal XGate::collides on every pair.
    // Wire universes rotate through dense-low (shared control wires and the
    // opposite-polarity separation exemption fire constantly), mid, one-word
    // boundary, and > 64 (exercises the second mask word). g57s (comp = true)
    // come from rand_gate's allow_comp draw.
    #[test]
    fn mask_collides_matches_xgate_collides() {
        use super::super::arena::{Arena, GateMask, MASK_WORDS};
        let mut rng = StdRng::seed_from_u64(0x2026_0809);
        let mut collided = 0usize;
        let mut separated = 0usize;
        for i in 0..1_000_000usize {
            let wires: u16 = match i % 4 {
                0 => 5,
                1 => 16,
                2 => 64,
                _ => 127,
            };
            let g = rand_gate(&mut rng, wires, 4, true);
            let h = rand_gate(&mut rng, wires, 4, true);
            let mg = GateMask::of(&g).expect("wire < 64 * MASK_WORDS has a mask");
            let mh = GateMask::of(&h).expect("wire < 64 * MASK_WORDS has a mask");
            let want = XGate::collides(&g, &h);
            assert_eq!(Arena::mask_collides(&mg, &mh), want, "mask mismatch: {g:?} vs {h:?}");
            if want {
                collided += 1;
            } else if g.reads(h.target) || h.reads(g.target) {
                separated += 1; // commuted only via the polarity exemption
            }
        }
        // The draw must actually exercise both hard branches.
        assert!(collided > 10_000, "too few colliding pairs: {collided}");
        assert!(separated > 1_000, "too few exemption-separated pairs: {separated}");
        // Out-of-range wires have no mask: collides_ids falls back to
        // XGate::collides (arena poisons masks_ok on such an alloc).
        let lim = (64 * MASK_WORDS) as u16;
        assert!(GateMask::of(&XGate::cnot(0, lim)).is_none());
        assert!(GateMask::of(&XGate::cnot(lim, 0)).is_none());
        assert!(GateMask::of(&XGate::cnot(0, lim - 1)).is_some());
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
            // The global reshuffle dilutes every other menu slot by exactly
            // shuffle_rate/|circuit|. At production size that is ~2e-5 and
            // invisible; on this 300-gate circuit it is ~0.4% of rounds taken
            // from the thermostat, which is enough to push the drift band.
            // This test is the thermostat's contract, so hold the new slot
            // out of it -- the move has its own test.
            shuffle_rate: 0.0,
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
        let twists = mx.counters.twist_swaps + mx.counters.twist_negs + mx.counters.twist_cnots;
        assert!(twists > 30, "swap-family twists barely ran: {twists}");
        assert!(mx.counters.twist_relabels > 0, "twists never relabeled a gate");
        assert!(mx.remaining_g57() <= comp0, "fossil count increased");
        assert!(mx.counters.merges() > 0, "no merges alongside twists");
        mx.global_check();
    }

    // The g57 census partitions the width-2 comp population, and the two halves
    // measure different things: `shaped` is structure (what the store can
    // spell) and `same_pol` is a twist odometer. The regression this guards is
    // the one that made the old single `g57=` field misleading -- a NEGATION
    // twist flips one control's polarity and moves a gate from opp_pol to
    // same_pol WITHOUT changing its comp, width or count, while a SWAP twist
    // carries polarity with the wire and moves nothing.
    #[test]
    fn g57_census_splits_structure_from_twist_polarity() {
        // The swap family flips control polarity only through its negation coins
        // (3/4 of twists carry a negation), so same_pol is a twist odometer:
        // twists ON drive it up, OFF leaves it at zero, while shaped (structure)
        // partitions correctly in both. p_db = 0 here, so nothing injects fresh
        // opposite-polarity material -- the only mover is the twist.
        let base = |p_twist: f64| MixParams {
            k_max: 5,
            moves: 20_000,
            target_size: 600,
            temp: 20.0,
            p_twist,
            twist_min_len: 4,
            verify_every: 1_000,
            report_every: u64::MAX,
            seed: 5,
            ..MixParams::default()
        };
        let run = |p_twist| {
            let mut mx = Mixer::new(random_g57_circuit(17, 16, 400), 16, base(p_twist));
            mx.run();
            let cen = mx.g57_census();
            // The partition identity, and agreement with the old accessor.
            assert_eq!(cen.shaped, cen.same_pol + cen.opp_pol, "census does not partition");
            assert_eq!(cen.opp_pol, mx.true_g57(), "true_g57 diverged from the census");
            assert!(cen.shaped <= mx.remaining_g57(), "shaped exceeds the comp population");
            (cen, mx.counters.twist_relabels)
        };

        let (off, off_rel) = run(0.0);
        let (on, on_rel) = run(0.2);
        assert_eq!(off_rel, 0, "twists-off relabeled a gate: {off_rel}");
        assert!(on_rel > 0, "twists-on never relabeled a gate: {on_rel}");

        // Twists off: no polarity flips and no fresh material, so every shaped
        // gate stays a true g57. This is the load-bearing half -- it proves
        // same_pol tracks the twist, not mixing in general.
        assert_eq!(off.same_pol, 0, "twists-off flipped polarity: {off:?}");
        assert_eq!(off.pol_flipped(), 0.0);

        // Twists on: the negation coins flip it, on the same circuit. No upper
        // bound -- with p_db = 0 the small width-2 population can saturate at
        // 1.0; the sub-1/2 equilibrium seen in production comes from DB splices
        // this test deliberately does not have.
        assert!(on.shaped > 0 && off.shaped > 0, "no width-2 population: {on:?} {off:?}");
        assert!(on.same_pol > 0, "swap-family twists flipped no polarity: {on:?}");
        assert!(on.pol_flipped() > off.pol_flipped(), "{on:?} vs {off:?}");
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
                // Geometry is a parameter now, not an internal coin; the test
                // already knows which one it is exercising.
                if let Some((ids, _dir, _smp)) = mx.sample_window(win, sample) {
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

    // The backward ancestor-span statistic must be populated under SAMPLED
    // ancestry (where the old min/max ancspan= is switched off) and must stay
    // inside its definitional bounds. It is bucket occupancy / entropy over
    // the INPUT circuit, so all three live in [0,1].
    #[test]
    fn sampled_ancestor_span_is_populated_and_bounded() {
        let gates = random_mixed_circuit(53, 16, 400);
        let params = MixParams {
            k_max: 6,
            moves: 4_000,
            target_size: 400,
            temp: 20.0,
            anc_samples: 64,
            p_twist: 0.0,
            shuffle_rate: 0.0,
            report_every: u64::MAX,
            seed: 17,
            ..MixParams::default()
        };
        let mut mx = Mixer::new_with_db(gates, 16, params, FrozenDb::empty());
        mx.run();
        let line = mx.tracer_report();
        assert!(line.contains("ancspan cov="), "ancspan block missing: {line}");
        let grab = |k: &str| -> f64 {
            let i = line.find(k).unwrap_or_else(|| panic!("no {k} in {line}")) + k.len();
            line[i..]
                .split(|c: char| c == ' ' || c == '|')
                .next()
                .unwrap()
                .parse()
                .unwrap()
        };
        let (cov, ent, sd) = (grab("ancspan cov="), grab("ent=")
            , grab("sd="));
        for (n, v) in [("cov", cov), ("ent", ent), ("sd", sd)] {
            assert!((0.0..=1.0).contains(&v), "ancspan {n}={v} out of [0,1]");
        }
        // The old min/max form stays OFF under sampling -- this replaces it in
        // the tracers line, it does not resurrect it in the mv= line.
        assert_eq!(mx.anc_stats(), (0.0, 0.0), "sampled mode must still leave anc=/ancspan= at 0");
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

    // The tg_* counter fields are APPENDED to the state line and parsed with
    // a zero default, so a .state written before they existed must still
    // load. Simulated here by stripping the trailing tokens.
    // The prefix descent must TERMINATE. Every exit path inside it has to
    // shrink the window before looping; a `continue` that leaves lo/hi alone
    // respins the identical prefix forever. Store misses are the common case,
    // so before this was fixed a --db-prefixes run hung before completing a
    // single move. With no DB attached every lookup misses, which is exactly
    // the path that used to spin -- if this test ever hangs, the descent has
    // regressed.
    #[test]
    fn prefix_descent_terminates_when_every_lookup_misses() {
        let gates = random_mixed_circuit(43, 16, 300);
        let params = MixParams {
            k_max: 6,
            moves: 5_000,
            target_size: 300,
            temp: 20.0,
            p_db: 1.0,
            db_prefixes: true,
            s_db: 9,
            db_min_window: 0,
            db_max_span: 4, // deliberately tight: forces the span-skip path too
            p_twist: 0.0,
            shuffle_rate: 0.0,
            report_every: u64::MAX,
            seed: 13,
            ..MixParams::default()
        };
        // FrozenDb::empty() -> every lookup misses, which IS the path that used
        // to spin. No store needed to prove termination.
        let mut mx = Mixer::new_with_db(gates, 16, params, FrozenDb::empty());
        mx.run();
        assert_eq!(mx.counters.moves, 5_000, "the run did not consume its move budget");
        assert!(
            mx.counters.db_attempts > 5_000,
            "the descent should make several attempts per round, got {}",
            mx.counters.db_attempts
        );
        assert!(mx.counters.db_span_skips > 0, "the tight span cap should have skipped windows");
        mx.global_check();
    }

    #[test]
    fn counters_line_tolerates_missing_trailing_fields() {
        let mut c = MixCounters::default();
        c.moves = 7;
        c.tg_consumed = 3;
        c.tg_emitted = 9;
        c.split_prims = 11;
        c.tap_flips = 13;
        c.split_span_sum = 17;
        c.cross_pool_shots = 19;
        let line = c.to_line();
        let full = MixCounters::from_line(&line).expect("roundtrip");
        assert_eq!((full.tg_consumed, full.tg_emitted), (3, 9));
        assert_eq!((full.split_prims, full.tap_flips), (11, 13));
        assert_eq!((full.split_span_sum, full.cross_pool_shots), (17, 19));
        let old: Vec<&str> = line.split_whitespace().collect();
        // A pre-split-stage line (drop the NINE fields appended since: the 7
        // split counters + span sum + pool shots): they default to zero, the
        // tg pair survives.
        let truncated = old[..old.len() - 9].join(" ");
        let parsed = MixCounters::from_line(&truncated).expect("pre-split state must load");
        assert_eq!(parsed.moves, 7);
        assert_eq!((parsed.tg_consumed, parsed.tg_emitted), (3, 9));
        assert_eq!((parsed.split_prims, parsed.cross_pool_shots), (0, 0));
        // A pre-tg line (drop those too): everything appended defaults.
        let truncated = old[..old.len() - 11].join(" ");
        let parsed = MixCounters::from_line(&truncated).expect("pre-tg state must load");
        assert_eq!(parsed.moves, 7);
        assert_eq!((parsed.tg_consumed, parsed.tg_emitted), (0, 0));
        assert_eq!((parsed.split_prims, parsed.tap_flips), (0, 0));
    }

    // ---- the split stage (docs/FMIX_SPLIT_TWIST.md) ----

    // The whole stage on a mixed circuit, every sub-rewrite locally verified
    // (presplits, absorptions, segment conjugations), ending by g57
    // exhaustion: the run stops at the boundary under split_stop, no comp
    // gate survives, twists actually landed, and the circuit still computes
    // the input.
    #[test]
    fn split_stage_runs_to_exhaustion_and_preserves_function() {
        let gates = random_mixed_circuit(11, 16, 400);
        let comp0 = gates.iter().filter(|g| g.comp).count();
        assert!(comp0 > 10, "test input must carry g57s, got {comp0}");
        let params = MixParams {
            k_max: 8,
            moves: 200_000,
            split: true,
            split_stop: true,
            split_canaries: 32,
            report_every: u64::MAX,
            verify_every: 1_000,
            seed: 5,
            ..MixParams::default()
        };
        let mut mx = Mixer::new_with_db(gates, 16, params, FrozenDb::empty());
        let stop = mx.run();
        assert!(matches!(stop, MixStop::SplitDone), "stage must end the run under split_stop");
        assert_eq!(mx.remaining_g57(), 0, "exit A means no comp gate survives");
        let splits =
            (mx.counters.split_prims + mx.counters.split_hsplits + mx.counters.split_segs) as usize;
        assert!(splits >= comp0, "every input g57 splits through SOME channel: {splits} < {comp0}");
        assert!(mx.counters.split_joins > 0, "p_join 0.8 must land twists");
        mx.global_check();
    }

    // Sibling convention (2026-08-05): the two pieces of a g57 split take
    // opposite directions.
    #[test]
    fn split_g57_pieces_take_opposite_directions() {
        for seed in 0..16 {
            let gates =
                vec![XGate::from_g57([0, 1, 2]), XGate::conj(3, [(1u16, true)]).unwrap()];
            let params =
                MixParams { seed, moves: 0, report_every: u64::MAX, ..MixParams::default() };
            let mut mx = Mixer::new_with_db(gates, 4, params, FrozenDb::empty());
            let (g1, g2) = mx.split_g57(0);
            assert_ne!(mx.meta_of(g1).dir, mx.meta_of(g2).dir, "siblings must oppose");
            mx.global_check();
        }
    }

    // v2 state roundtrip mid-stage: the live flag, the failure streak and the
    // canaries all ride the checkpoint; the resumed run finishes the stage
    // and still verifies.
    #[test]
    fn split_stage_state_roundtrip() {
        let gates = random_mixed_circuit(13, 16, 300);
        let params = MixParams {
            k_max: 8,
            moves: 4,
            split: true,
            split_canaries: 16,
            report_every: u64::MAX,
            verify_every: u64::MAX,
            seed: 9,
            ..MixParams::default()
        };
        let mut mx = Mixer::new_with_db(gates, 16, params.clone(), FrozenDb::empty());
        let stop = mx.run();
        assert!(matches!(stop, MixStop::MovesBudget));
        let taps0 = mx.taps.len();
        assert!(taps0 > 0, "canaries plant on the first stage move");
        let path = std::env::temp_dir().join("fmix_split_state_roundtrip.txt");
        let path = path.to_str().unwrap().to_string();
        mx.save_state(&path).expect("save state");
        let params2 = MixParams { moves: 200_000, split_stop: true, ..params };
        let mut mx2 = Mixer::resume_state(&path, params2, FrozenDb::empty()).expect("resume");
        assert!(mx2.split_on, "live stage flag must survive the checkpoint");
        assert_eq!(mx2.taps.len(), taps0, "canaries must survive the checkpoint");
        let stop = mx2.run();
        assert!(matches!(stop, MixStop::SplitDone), "resumed stage must finish");
        assert_eq!(mx2.remaining_g57(), 0);
        mx2.global_check();
        let _ = std::fs::remove_file(&path);
    }

    // Min-dgen cross-shot bias: armed, the walk still preserves function and
    // the pool actually supplies shots; off, the constructor path draws no
    // extra RNG so the trajectory is byte-identical to the historical chain.
    #[test]
    fn mincross_pool_supplies_shots_and_preserves_function() {
        let gates = random_mixed_circuit(17, 16, 400);
        let params = MixParams {
            k_max: 8,
            moves: 60_000,
            p_mincross: 0.9,
            cross_pool_k: 500,
            cross_rescan: 2_000,
            report_every: u64::MAX,
            verify_every: 10_000,
            seed: 21,
            ..MixParams::default()
        };
        let mut mx = Mixer::new_with_db(gates.clone(), 16, params, FrozenDb::empty());
        mx.run();
        assert!(mx.counters.cross_pool_shots > 1_000, "the pool must actually supply shots");
        mx.global_check();

        // Off = identical trajectory: same seed, with and without the (inert)
        // pool knobs, ends in the same circuit.
        let base = MixParams {
            k_max: 8,
            moves: 30_000,
            report_every: u64::MAX,
            verify_every: u64::MAX,
            seed: 22,
            ..MixParams::default()
        };
        let mut a = Mixer::new_with_db(gates.clone(), 16, base.clone(), FrozenDb::empty());
        let mut b = Mixer::new_with_db(
            gates,
            16,
            MixParams { cross_pool_k: 7, cross_rescan: 55, ..base },
            FrozenDb::empty(),
        );
        a.run();
        b.run();
        assert_eq!(a.arena.to_vec(), b.arena.to_vec(), "p_mincross 0 must not perturb the walk");
    }

    // --twist-g57: brackets become adaptive all-g57 words solved online per
    // seam. Thousands of twists under local_verify (every seam splice checked
    // exhaustively against the reference 3-CNOT packet) plus periodic full
    // verification and a final global_check. The seams must also actually
    // absorb neighborhood material — a run where tg_consumed stays 0 means
    // the placer degenerated to bare words and the mechanism is dead.
    // The global re-randomisation move must MOVE gates and change NOTHING
    // else: same function, same size, same multiset of gates. It is scheduled
    // at 1/|circuit| per round, so the rate is pushed hard here to make it
    // fire often enough to be worth asserting on.
    #[test]
    fn global_shuffle_moves_gates_but_preserves_function_and_size() {
        let gates = random_mixed_circuit(37, 16, 300);
        let before = gates.clone();
        let params = MixParams {
            k_max: 6,
            moves: 20_000,
            target_size: 300,
            temp: 20.0,
            // 300 * the size, so the per-round p is ~1: a reshuffle nearly
            // every round.
            shuffle_rate: 300.0,
            p_twist: 0.0,
            p_db: 0.0,
            local_verify: true,
            verify_every: 500,
            report_every: u64::MAX,
            seed: 11,
            ..MixParams::default()
        };
        let mut mx = Mixer::new(gates, 16, params);
        mx.run();
        assert!(mx.counters.shuffles > 1_000, "shuffle never fired: {}", mx.counters.shuffles);
        assert!(
            mx.counters.shuffle_moved > 0,
            "shuffles ran but no gate ever changed position"
        );
        let after: Vec<XGate> =
            mx.arena.ids_in_order().iter().map(|&id| mx.arena.gate(id).clone()).collect();
        assert_eq!(after.len(), before.len(), "shuffle changed the circuit SIZE");
        let mut a: Vec<_> = after.iter().map(|g| format!("{g:?}")).collect();
        let mut b: Vec<_> = before.iter().map(|g| format!("{g:?}")).collect();
        a.sort();
        b.sort();
        assert_eq!(a, b, "shuffle changed the gate MULTISET, not just the order");
        assert_ne!(
            after.iter().map(|g| format!("{g:?}")).collect::<Vec<_>>(),
            before.iter().map(|g| format!("{g:?}")).collect::<Vec<_>>(),
            "shuffle left the order untouched"
        );
        mx.global_check();
    }

    #[test]
    fn mixer_g57_twists_preserve_function_and_absorb() {
        let gates = random_mixed_circuit(29, 16, 300);
        let params = MixParams {
            k_max: 6,
            moves: 20_000,
            target_size: 600,
            temp: 20.0,
            p_twist: 0.3,
            twist_min_len: 4,
            twist_g57: true,
            local_verify: true,
            verify_every: 1_000,
            report_every: u64::MAX,
            seed: 7,
            ..MixParams::default()
        };
        let mut mx = Mixer::new(gates, 16, params);
        mx.run();
        assert!(mx.counters.twist_swaps > 100, "g57 twists barely ran: {}", mx.counters.twist_swaps);
        assert!(mx.counters.tg_emitted > 0, "brackets emitted nothing");
        assert!(
            mx.counters.tg_consumed > 0,
            "no seam ever absorbed a neighbor -- the adaptive placement is dead"
        );
        // The solver minimizes NET (word len minus context consumed), so a
        // seam may emit up to 7 gates while consuming 3; what can never
        // happen is a twist NET above the 12-gate bare spelling, since k=0
        // always offers the 6-word per seam.
        let net = (mx.counters.tg_emitted as i64 - mx.counters.tg_consumed as i64) as f64
            / mx.counters.twist_swaps as f64;
        assert!(net <= 12.0 + 1e-9, "twist net cost exceeded the bare-word bound: {net}");
        mx.global_check();
    }

    // v2 seam consumption must PROPAGATE ancestry: a bracket word that
    // consumed real context takes the union of the consumed litters' ancestor
    // sets (DB-splice semantics). v1 dropped them, silently deflating anc.
    #[test]
    fn g57_twist_consumption_inherits_ancestry() {
        let gates = random_mixed_circuit(31, 16, 300);
        let params = MixParams {
            k_max: 6,
            moves: 20_000,
            target_size: 600,
            temp: 20.0,
            p_twist: 0.3,
            twist_min_len: 4,
            twist_g57: true,
            local_verify: true,
            ancestors: true,
            verify_every: 1_000,
            report_every: u64::MAX,
            seed: 9,
            ..MixParams::default()
        };
        let mut mx = Mixer::new(gates, 16, params);
        mx.run();
        assert!(mx.counters.tg_consumed > 0, "no consumption to test");
        let inherited = mx.arena.ids_in_order().iter().any(|&id| {
            let m = mx.meta_of(id);
            m.origin == ORIGIN_SYNTH
                && mx.anc.get(&m.litter).is_some_and(|bits| bits.iter().any(|&w| w != 0))
        });
        assert!(inherited, "no synthetic gate carries inherited ancestry");
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
        let twists = mx.counters.twist_swaps + mx.counters.twist_negs + mx.counters.twist_cnots;
        assert!(twists > 50, "twists barely ran: {twists}");
        mx.global_check();
    }

    // The negate-both variant is a genuine involution (T^2 = id) like the pure
    // swap, whereas negate-one is not (T^2 = negate-both); this test drives the
    // family hard so all three variants -- and the non-involutive closing
    // bracket P^-1 -- get exercised, keeps the function through thousands of
    // twists, and never grows fossils. All three variant counters must fire.
    #[test]
    fn mixer_swap_family_twists_preserve_function() {
        let gates = random_mixed_circuit(19, 16, 300);
        let comp0 = gates.iter().filter(|g| g.comp).count();
        let params = MixParams {
            k_max: 6,
            moves: 20_000,
            target_size: 600,
            temp: 20.0,
            p_twist: 0.3,
            twist_min_len: 4,
            verify_every: 1_000,
            report_every: u64::MAX,
            seed: 7,
            ..MixParams::default()
        };
        let mut mx = Mixer::new(gates, 16, params);
        mx.run();
        // The (alpha, beta) coins are 1/4 : 1/2 : 1/4, so at this rate all three
        // variants fire many times.
        assert!(mx.counters.twist_swaps > 5, "pure swaps barely ran: {}", mx.counters.twist_swaps);
        assert!(mx.counters.twist_negs > 5, "negate-one twists barely ran: {}", mx.counters.twist_negs);
        assert!(mx.counters.twist_cnots > 5, "negate-both twists barely ran: {}", mx.counters.twist_cnots);
        assert!(mx.counters.twist_relabels > 0, "twists never relabeled a gate");
        assert!(mx.remaining_g57() <= comp0, "fossil count increased");
        assert!(mx.counters.merges() > 0, "no merges alongside twists");
        mx.global_check();
    }

    // Sampled ancestry must agree with exact ancestry on the quantity they both
    // measure. Tracer choice comes from a dedicated rng, so the two runs follow
    // the IDENTICAL chain (asserted gate-for-gate) and the only difference is
    // the instrument -- which makes this a real calibration rather than two
    // independent samples that happen to be close.
    #[test]
    fn sampled_ancestry_calibrates_to_exact() {
        let gates = random_mixed_circuit(31, 16, 400);
        let base = MixParams {
            k_max: 5,
            moves: 20_000,
            target_size: 600,
            temp: 20.0,
            p_twist: 0.05,
            verify_every: 5_000,
            report_every: u64::MAX,
            seed: 9,
            ..MixParams::default()
        };
        let mut ex = Mixer::new(gates.clone(), 16, MixParams { ancestors: true, ..base.clone() });
        ex.run();
        let mut sa =
            Mixer::new(gates.clone(), 16, MixParams { anc_samples: 128, ..base.clone() });
        sa.run();

        // Same chain: the instrument may not perturb the walk.
        assert_eq!(ex.arena.to_vec(), sa.arena.to_vec(), "tracer selection changed the trajectory");

        // Exact mode still reports anc/span; sampled mode deliberately does not.
        assert!(ex.anc_stats().0 > 0.0, "exact mode lost its anc reading");
        assert_eq!(sa.anc_stats(), (0.0, 0.0), "sampled mode must not fill anc=/ancspan=");
        assert!(sa.tracer_report().contains("tracers: K=128"), "{}", sa.tracer_report());

        let exact = ex.anc_incidence();
        let est = sa.anc_incidence();
        assert!(exact > 0.0, "no incidence to compare");
        let rel = (est - exact).abs() / exact;
        assert!(rel < 0.25, "sampled incidence off by {rel:.3} (exact {exact:.0}, est {est:.0})");
    }

    // The joint gen x anc census must partition the circuit exactly (every gate
    // lands in exactly one band, including the GEN_FRESH sentinel band) and must
    // work in BOTH ancestry modes.
    #[test]
    fn gen_anc_census_partitions_the_circuit() {
        let gates = random_mixed_circuit(23, 16, 400);
        let base = MixParams {
            k_max: 5,
            moves: 20_000,
            target_size: 600,
            temp: 20.0,
            p_twist: 0.1, // mint some GEN_FRESH bracket material
            gen_target: 5,
            verify_every: 5_000,
            report_every: u64::MAX,
            seed: 4,
            ..MixParams::default()
        };
        for (label, p) in [
            ("exact", MixParams { ancestors: true, ..base.clone() }),
            ("sampled", MixParams { anc_samples: 64, ..base.clone() }),
        ] {
            let mut mx = Mixer::new(gates.clone(), 16, p);
            mx.run();
            let line = mx.gen_anc_report();
            assert!(line.starts_with("[fmix] gen-anc: r="), "{label}: {line}");
            // Band counts must sum to the circuit size.
            let total: usize = line
                .split('|')
                .filter_map(|s| s.split("n=").nth(1))
                .filter_map(|s| s.split_whitespace().next())
                .filter_map(|s| s.parse::<usize>().ok())
                .sum();
            // The first "n=" is the real-gen count in the header, so subtract it.
            let hdr: usize = line
                .split("(n=")
                .nth(1)
                .and_then(|s| s.split_whitespace().next())
                .and_then(|s| s.parse::<f64>().ok())
                .map(|f| f as usize)
                .expect("header count");
            assert_eq!(total - hdr, mx.arena.len(), "{label} bands do not partition: {line}");
            assert!(hdr <= mx.arena.len(), "{label}: more real-gen gates than gates");
            // Exact mode reports span per band, sampled mode must not.
            assert_eq!(line.contains("span="), label == "exact", "{label}: {line}");
        }
    }

    // The whole point of sampling: it runs on inputs the exact instrument
    // refuses (it asserts n <= 20_000). Cost is K bits per litter regardless of
    // input size, so the ancestor map stays far smaller than the circuit.
    #[test]
    fn sampled_ancestry_runs_past_the_exact_cap() {
        let n = 25_000;
        let gates = random_mixed_circuit(5, 24, n);
        let params = MixParams {
            k_max: 5,
            moves: 3_000,
            target_size: n + 200,
            temp: 50.0,
            anc_samples: 64,
            verify_every: 1_500,
            report_every: u64::MAX,
            seed: 3,
            ..MixParams::default()
        };
        let mut mx = Mixer::new(gates, 24, params);
        mx.run();
        assert!(mx.anc_incidence() > 0.0, "sampled ancestry recorded nothing");
        let rep = mx.tracer_report();
        assert!(rep.contains("K=64 of m=25000"), "{rep}");
        // Only litters that actually carry a tracer are stored, so the map is a
        // small fraction of the circuit -- this is what makes it scale.
        assert!(
            mx.anc.len() < mx.arena.len(),
            "ancestor map ({}) is not smaller than the circuit ({})",
            mx.anc.len(),
            mx.arena.len()
        );
        mx.global_check();
    }

    // twist_neg_p = 0 gives PURE positive swaps: no wire is ever negated, so no
    // interior polarity flips (same_pol stays 0) and only the pure-swap counter
    // moves -- yet the 3-CNOT brackets are still inserted (comp=0 material
    // present). This is the control that separates "foreign CNOTs" from
    // "polarity scrambling".
    #[test]
    fn twist_neg_p_zero_is_pure_swap() {
        let params = MixParams {
            k_max: 5,
            moves: 20_000,
            target_size: 600,
            temp: 20.0,
            p_twist: 0.3,
            twist_neg_p: 0.0,
            twist_min_len: 4,
            verify_every: 1_000,
            report_every: u64::MAX,
            seed: 7,
            ..MixParams::default()
        };
        let mut mx = Mixer::new(random_g57_circuit(17, 16, 400), 16, params);
        mx.run();
        assert!(mx.counters.twist_swaps > 20, "no pure swaps ran: {}", mx.counters.twist_swaps);
        assert_eq!(mx.counters.twist_negs, 0, "negate-one fired at twist_neg_p=0");
        assert_eq!(mx.counters.twist_cnots, 0, "negate-both fired at twist_neg_p=0");
        assert_eq!(mx.g57_census().same_pol, 0, "pure swap flipped polarity");
        assert!(mx.counters.twist_relabels > 0, "pure swap never relabeled");
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

    // The GSS profile needs the window length to depend on BOTH the live mode
    // and the geometry drawn for that round -- COMP convex 12 / COMP
    // contiguous 6 / MIX 6 for either. Precedence is most-specific-first, and
    // an unset level must fall through rather than clamp the window to nothing.
    #[test]
    fn s_db_resolves_by_mode_and_geometry() {
        let gates = random_mixed_circuit(11, 8, 40);
        let params = MixParams {
            s_db: 6,
            s_db_comp: Some(12),
            s_db_comp_ctg: Some(6),
            s_db_ctg: None, // MIX shares one length across both geometries
            moves: 0,
            report_every: u64::MAX,
            ..MixParams::default()
        };
        let mut mx = Mixer::new(gates, 8, params);

        mx.db_mode_cur = DbMode::Compressing;
        assert_eq!(mx.active_s_db(DbSample::Convex), 12, "COMP convex takes s_db_comp");
        assert_eq!(mx.active_s_db(DbSample::Contiguous), 6, "COMP contiguous takes s_db_comp_ctg");

        mx.db_mode_cur = DbMode::Mix;
        assert_eq!(mx.active_s_db(DbSample::Convex), 6, "MIX takes the base s_db");
        assert_eq!(
            mx.active_s_db(DbSample::Contiguous),
            6,
            "s_db_ctg=0 falls through to the base s_db, it does not zero the window"
        );

        // A geometry override only applies to its own mode.
        mx.params.s_db_ctg = Some(3);
        assert_eq!(mx.active_s_db(DbSample::Contiguous), 3, "MIX contiguous now overridden");
        mx.db_mode_cur = DbMode::Compressing;
        assert_eq!(
            mx.active_s_db(DbSample::Contiguous),
            6,
            "COMP contiguous still reads s_db_comp_ctg, not the MIX override"
        );
    }

    // Descent is per-mode: the overlay runs MIX and COMP in one process and
    // GSS wants it on in COMP and off in MIX. None must fall back to the
    // global flag so single-mode runs behave exactly as before.
    #[test]
    fn prefix_descent_resolves_per_mode() {
        let gates = random_mixed_circuit(11, 8, 40);
        let params = MixParams {
            db_prefixes: true,
            db_prefixes_comp: Some(true),
            db_prefixes_mix: Some(false),
            moves: 0,
            report_every: u64::MAX,
            ..MixParams::default()
        };
        let mut mx = Mixer::new(gates, 8, params);
        mx.db_mode_cur = DbMode::Compressing;
        assert!(mx.active_prefixes(), "COMP descends");
        mx.db_mode_cur = DbMode::Mix;
        assert!(!mx.active_prefixes(), "MIX does not, even though db_prefixes is true");

        // Unset per-mode overrides inherit the global flag, both ways.
        mx.params.db_prefixes_mix = None;
        mx.params.db_prefixes_comp = None;
        assert!(mx.active_prefixes(), "None inherits db_prefixes=true");
        mx.params.db_prefixes = false;
        assert!(!mx.active_prefixes(), "None inherits db_prefixes=false");
    }

    // Geometry is now drawn ONCE per round, before the length. A run pinned to
    // one geometry must therefore only ever report that geometry, and the
    // per-length histogram must respect that geometry's own s_db ceiling.
    #[test]
    fn geometry_is_drawn_before_the_length_and_bounds_it() {
        let gates = random_mixed_circuit(31, 16, 400);
        let base = MixParams {
            moves: 4_000,
            target_size: 400,
            temp: 20.0,
            p_db: 0.0, // store-free: exercise the sampler, not the store
            s_db: 9,
            db_min_window: 0,
            s_db_ctg: Some(3),
            verify_every: u64::MAX,
            report_every: u64::MAX,
            seed: 7,
            ..MixParams::default()
        };
        // All-contiguous: every window must obey s_db_ctg=3, not s_db=9.
        let mut ctg = Mixer::new(gates.clone(), 16, MixParams { p_convex: 0.0, ..base.clone() });
        ctg.db_mode_cur = DbMode::Mix;
        for _ in 0..200 {
            let w = ctg.active_s_db(DbSample::Contiguous);
            assert!(w <= 3, "contiguous window ceiling leaked: {w}");
        }
        // All-convex: the contiguous override must not apply.
        let mut cvx = Mixer::new(gates, 16, MixParams { p_convex: 1.0, ..base });
        cvx.db_mode_cur = DbMode::Mix;
        assert_eq!(cvx.active_s_db(DbSample::Convex), 9);
    }

    // Every layering rule in one place, asserted against MixParams::db_knobs --
    // the function both the mixer and the CLI banner go through.
    #[test]
    fn db_knobs_layering_rules() {
        let base = MixParams { s_db: 9, p_convex: 0.4, p_mingen: 0.8, db_prefixes: true,
            ..MixParams::default() };

        // Nothing overridden: both modes see the base.
        let k = base.db_knobs(DbMode::Mix);
        let c = base.db_knobs(DbMode::Compressing);
        assert_eq!((k.s_db_cvx, k.s_db_ctg, k.p_convex), (9, 9, 0.4));
        assert_eq!((c.s_db_cvx, c.s_db_ctg, c.p_convex), (9, 9, 0.4));

        // A mode override moves only that mode, and reaches BOTH its geometries.
        let p = MixParams { s_db_comp: Some(12), ..base.clone() };
        assert_eq!(p.db_knobs(DbMode::Mix).s_db_cvx, 9);
        assert_eq!(p.db_knobs(DbMode::Compressing).s_db_cvx, 12);
        assert_eq!(
            p.db_knobs(DbMode::Compressing).s_db_ctg,
            12,
            "COMP contiguous inherits COMP convex, not the base"
        );

        // A geometry override is narrower still.
        let p = MixParams { s_db_comp: Some(12), s_db_comp_ctg: Some(6), ..base.clone() };
        let c = p.db_knobs(DbMode::Compressing);
        assert_eq!((c.s_db_cvx, c.s_db_ctg), (12, 6));
        assert_eq!(p.db_knobs(DbMode::Mix).s_db_ctg, 9, "MIX untouched by COMP overrides");

        // ZERO AND FALSE MEAN THEMSELVES. This is the whole reason these are
        // Option: under the old sentinel encoding a legitimate 0 -- which is
        // exactly what GSS wants for p_mingen_comp -- was indistinguishable
        // from "unset", so it silently fell through to the base.
        let p = MixParams { p_mingen: 0.8, p_mingen_comp: Some(0.0), ..base.clone() };
        assert_eq!(p.db_knobs(DbMode::Compressing).p_mingen, 0.0);
        assert_eq!(p.db_knobs(DbMode::Mix).p_mingen, 0.8);
        let p = MixParams { db_prefixes: true, db_prefixes_mix: Some(false), ..base.clone() };
        assert!(!p.db_knobs(DbMode::Mix).prefixes);
        assert!(p.db_knobs(DbMode::Compressing).prefixes);

        // The GSS profile, end to end.
        let gss = MixParams {
            s_db: 6, p_convex: 0.5, p_mingen: 0.5, db_prefixes: true,
            db_prefixes_mix: Some(false), db_prefixes_comp: Some(true),
            p_mingen_comp: Some(0.0), p_convex_comp: Some(0.95),
            s_db_comp: Some(12), s_db_comp_ctg: Some(6),
            ..MixParams::default()
        };
        let m = gss.db_knobs(DbMode::Mix);
        let c = gss.db_knobs(DbMode::Compressing);
        assert_eq!((m.s_db_cvx, m.s_db_ctg, m.p_convex, m.p_mingen, m.prefixes),
                   (6, 6, 0.5, 0.5, false));
        assert_eq!((c.s_db_cvx, c.s_db_ctg, c.p_convex, c.p_mingen, c.prefixes),
                   (12, 6, 0.95, 0.0, true));
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
            db_min_window: 0,
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

    // Ancestry treats a cross as a DB splice over the window {g, h}: every
    // output of a crossing — the intact pivot included — carries the UNION of
    // both parents' ancestor sets. Verified through the journal, which
    // records the pre-cross litters: every piece of a live entry must read
    // exactly union(set(litters[0]), set(litters[1])).
    #[test]
    fn cross_outputs_carry_union_ancestry() {
        let gates = random_mixed_circuit(43, 16, 200);
        let n = gates.len();
        let params = MixParams {
            k_max: 6,
            moves: 6_000,
            target_size: 4 * n,
            temp: 20.0,
            ancestors: true,
            report_every: u64::MAX,
            seed: 5,
            ..MixParams::default()
        };
        let mut mx = Mixer::new(gates, 16, params);
        mx.run();
        let mut checked = 0usize;
        let mut expected = vec![0u64; mx.anc_words];
        let mut got = vec![0u64; mx.anc_words];
        for e in mx.journal.iter() {
            let live = e
                .after
                .iter()
                .all(|&(id, st)| mx.arena.is_linked(id) && mx.arena.stamp(id) == st);
            if !live {
                continue;
            }
            expected.iter_mut().for_each(|w| *w = 0);
            mx.anc_or_into(e.litters[0], &mut expected);
            mx.anc_or_into(e.litters[1], &mut expected);
            for &(id, _) in &e.after {
                got.iter_mut().for_each(|w| *w = 0);
                mx.anc_or_into(mx.meta_of(id).litter, &mut got);
                assert_eq!(
                    got, expected,
                    "a cross output (intact pivot included) must carry the parents' union"
                );
            }
            checked += 1;
        }
        assert!(checked > 0, "the run must leave live journal entries to check");
        mx.global_check();
    }

    // The pre-cross litters a live journal entry will restore must survive
    // anc_prune: the cross relabels EVERY output to the union litter, so the
    // parents' litters can go extinct among live gates, and pruning their
    // sets would make a later undo restore ancestry-less litters silently.
    #[test]
    fn anc_prune_keeps_litters_live_journal_entries_restore() {
        let gates = random_mixed_circuit(47, 16, 200);
        let n = gates.len();
        let params = MixParams {
            k_max: 6,
            moves: 6_000,
            target_size: 4 * n,
            temp: 20.0,
            ancestors: true,
            report_every: u64::MAX,
            seed: 6,
            ..MixParams::default()
        };
        let mut mx = Mixer::new(gates, 16, params);
        mx.run();
        let live_entries: Vec<[u64; 2]> = mx
            .journal
            .iter()
            .filter(|e| {
                e.after
                    .iter()
                    .all(|&(id, st)| mx.arena.is_linked(id) && mx.arena.stamp(id) == st)
            })
            .map(|e| e.litters)
            .collect();
        assert!(!live_entries.is_empty(), "need live journal entries to make the test bite");
        let resolve = |mx: &Mixer, l: u64| {
            let mut bits = vec![0u64; mx.anc_words];
            mx.anc_or_into(l, &mut bits);
            bits
        };
        let before: Vec<[Vec<u64>; 2]> = live_entries
            .iter()
            .map(|ls| [resolve(&mx, ls[0]), resolve(&mx, ls[1])])
            .collect();
        mx.anc_prune();
        for (ls, want) in live_entries.iter().zip(before.iter()) {
            assert_eq!(resolve(&mx, ls[0]), want[0], "prune dropped a restorable litter's set");
            assert_eq!(resolve(&mx, ls[1]), want[1], "prune dropped a restorable litter's set");
        }
    }

    // Sidecar round trip in both universes: write -> read must reproduce the
    // per-gate resolved sets verbatim, and importing into a FRESH mixer over
    // the same circuit must resolve identically (the phase-boundary use).
    #[test]
    fn anc_sidecar_round_trips_exact_and_sampled() {
        for sampled in [false, true] {
            let gates = random_mixed_circuit(51, 16, 150);
            let params = MixParams {
                k_max: 6,
                moves: 4_000,
                target_size: 3 * gates.len(),
                temp: 20.0,
                ancestors: !sampled,
                anc_samples: if sampled { 32 } else { 0 },
                report_every: u64::MAX,
                seed: 7,
                ..MixParams::default()
            };
            let mut mx = Mixer::new(gates, 16, params);
            mx.run();
            let path = std::env::temp_dir().join(format!("fmix_anc_test_{sampled}.anc"));
            let path = path.to_str().unwrap().to_string();
            mx.write_anc_sidecar(&path).expect("write sidecar");
            let sc = Mixer::read_anc_sidecar(&path).expect("read sidecar");
            assert_eq!(sc.sampled, sampled);
            assert_eq!(sc.m, mx.anc_m, "universe size must survive");
            assert_eq!(sc.tracers, mx.anc_tracers, "tracer list must survive");
            let resolved: Vec<Vec<u64>> = {
                let mut v = Vec::new();
                let mut bits = vec![0u64; mx.anc_words];
                let mut cur = mx.arena.head();
                while cur != NIL {
                    bits.iter_mut().for_each(|w| *w = 0);
                    mx.anc_or_into(mx.meta_of(cur).litter, &mut bits);
                    v.push(bits.clone());
                    cur = mx.arena.neighbor(cur, Dir::R);
                }
                v
            };
            assert_eq!(sc.sets, resolved, "sidecar rows must equal the resolved per-gate sets");

            // Import into a fresh run over the SAME circuit (ancestry off at
            // construction; the sidecar defines the universe).
            let out_gates = mx.arena.to_vec();
            let params2 = MixParams {
                k_max: 6,
                moves: 2_000,
                temp: 20.0,
                report_every: u64::MAX,
                seed: 8,
                ..MixParams::default()
            };
            let mut mx2 = Mixer::new(out_gates, 16, params2);
            let sc2 = Mixer::read_anc_sidecar(&path).expect("re-read sidecar");
            mx2.import_ancestry(sc2);
            assert_eq!(mx2.anc_m, mx.anc_m, "imported universe must be the ORIGINAL m");
            assert_eq!(mx2.anc_tracers, mx.anc_tracers);
            let resolved2: Vec<Vec<u64>> = {
                let mut v = Vec::new();
                let mut bits = vec![0u64; mx2.anc_words];
                let mut cur = mx2.arena.head();
                while cur != NIL {
                    bits.iter_mut().for_each(|w| *w = 0);
                    mx2.anc_or_into(mx2.meta_of(cur).litter, &mut bits);
                    v.push(bits.clone());
                    cur = mx2.arena.neighbor(cur, Dir::R);
                }
                v
            };
            assert_eq!(resolved2, resolved, "imported ancestry must resolve identically");
            // The imported run must keep walking and unioning without issue.
            mx2.run();
            mx2.global_check();
            assert_eq!(mx2.anc_m, mx.anc_m, "the universe must not drift during the run");
            let _ = std::fs::remove_file(&path);
        }
    }

    // A state file written by an ancestry-IMPORTED run must restore the
    // imported tracer set verbatim: it is not a function of (anc_m, K, seed),
    // so only the explicit anctracers section can reproduce it.
    #[test]
    fn state_round_trips_imported_tracers() {
        let gates = random_mixed_circuit(53, 16, 150);
        let params = MixParams {
            k_max: 6,
            moves: 3_000,
            target_size: 3 * gates.len(),
            temp: 20.0,
            anc_samples: 24,
            report_every: u64::MAX,
            seed: 9,
            ..MixParams::default()
        };
        let mut mx = Mixer::new(gates, 16, params);
        mx.run();
        let anc_path = std::env::temp_dir().join("fmix_anc_state_test.anc");
        let anc_path = anc_path.to_str().unwrap().to_string();
        mx.write_anc_sidecar(&anc_path).expect("write sidecar");

        // Fresh run over the output, universe imported: its tracer indices
        // point into the ORIGINAL input, which this run has never seen.
        let out_gates = mx.arena.to_vec();
        let params2 = MixParams {
            k_max: 6,
            moves: 2_000,
            temp: 20.0,
            report_every: u64::MAX,
            seed: 10,
            ..MixParams::default()
        };
        let mut mx2 = Mixer::new(out_gates, 16, params2.clone());
        mx2.import_ancestry(Mixer::read_anc_sidecar(&anc_path).expect("read sidecar"));
        mx2.run();
        let want_tracers = mx2.anc_tracers.clone();
        let want_m = mx2.anc_m;

        let st_path = std::env::temp_dir().join("fmix_anc_state_test.state");
        let st_path = st_path.to_str().unwrap().to_string();
        mx2.save_state(&st_path).expect("save state");
        // Resume without any ancestry flags: the stored section must arm it.
        let rs = Mixer::resume_state(&st_path, params2, FrozenDb::empty()).expect("resume");
        assert!(rs.anc_sampled, "stored anctracers must re-arm sampled mode");
        assert_eq!(rs.anc_tracers, want_tracers, "imported tracers must survive the state file");
        assert_eq!(rs.anc_m, want_m);
        let _ = std::fs::remove_file(&anc_path);
        let _ = std::fs::remove_file(&st_path);
    }

    // Pair geometry (docs/NONLOCAL_PHASE_A.md): with an empty store every
    // attempt misses, so a pair round reduces to scan + fuse-float + the
    // restore walk — all commutations. The function must survive, and the
    // geometry must actually fire.
    #[test]
    fn pair_geometry_preserves_function_on_misses() {
        let gates = random_mixed_circuit(7, 16, 300);
        let params = MixParams {
            p_db: 1.0,
            p_pair: 1.0,
            s_db: 5,
            moves: 30_000,
            temp: 20.0,
            report_every: u64::MAX,
            verify_every: 5_000,
            seed: 11,
            ..MixParams::default()
        };
        let mut mx = Mixer::new_with_db(gates, 16, params, FrozenDb::empty());
        mx.run();
        assert!(mx.counters.pair_rounds > 0, "pair geometry never fired");
        assert!(mx.counters.pair_fused > 0, "no pair was ever fused");
        assert_eq!(mx.counters.pair_splices, 0, "empty store cannot splice");
        mx.global_check();
    }

    // collect_pair's window contract: two physically adjacent gates in link
    // order that COMMUTE — the window shape the convex and contiguous
    // samplers cannot produce — and the fusing floats must preserve the
    // function.
    #[test]
    fn collect_pair_fuses_an_adjacent_commuting_pair() {
        let gates = random_mixed_circuit(13, 16, 200);
        let params = MixParams {
            p_pair: 1.0,
            report_every: u64::MAX,
            seed: 5,
            ..MixParams::default()
        };
        let mut mx = Mixer::new_with_db(gates, 16, params, FrozenDb::empty());
        let mut fused = 0usize;
        for _ in 0..200 {
            if let Some((ids, _dir)) = mx.collect_pair() {
                assert_eq!(ids.len(), 2, "a pair window is exactly two gates");
                assert_eq!(
                    mx.arena.neighbor(ids[0], Dir::R),
                    ids[1],
                    "fused pair must be physically adjacent, leftmost first"
                );
                assert!(
                    !mx.arena.collides_ids(ids[0], ids[1]),
                    "a pair window is a COMMUTING pair by construction"
                );
                fused += 1;
            }
        }
        assert!(fused > 0, "no pair fused in 200 attempts");
        assert_eq!(fused as u64, mx.counters.pair_fused);
        mx.global_check();
    }

    // The bridge wake algebra: for random carrier/interior-gate pairs, the
    // claimed conjugate u·h·u = [h, corrections] must hold exactly — checked
    // against exhaustive evaluation, with coverage over both collision modes
    // (h reads the carrier's target / h writes a carrier control wire) and
    // the commuting and contradictory (correction-vanishes) cases.
    #[test]
    fn conj_wake_is_the_exact_conjugate() {
        let n: u16 = 8;
        let mut rng = StdRng::seed_from_u64(0xb41d6e);
        let mut seen = [0usize; 3]; // commuting/vanished, mode-a, mode-b
        let mut refused = 0usize;
        for i in 0..6000 {
            let tu = rng.random_range(0..n);
            let mut xw = rng.random_range(0..n);
            let mut yw = rng.random_range(0..n);
            while xw == tu {
                xw = rng.random_range(0..n);
            }
            while yw == tu || yw == xw {
                yw = rng.random_range(0..n);
            }
            let u = XGate::conj(tu, [(xw, rng.random_bool(0.5)), (yw, rng.random_bool(0.5))])
                .unwrap();
            // Alternate g57 and conjunction interiors.
            let h = if i % 2 == 0 {
                let t = rng.random_range(0..n);
                let mut a = rng.random_range(0..n);
                let mut b = rng.random_range(0..n);
                while a == t {
                    a = rng.random_range(0..n);
                }
                while b == t || b == a {
                    b = rng.random_range(0..n);
                }
                XGate::from_g57([t, a, b])
            } else {
                let t = rng.random_range(0..n);
                let w = rng.random_range(1..=3);
                let mut wires: Vec<u16> = (0..n).filter(|&x| x != t).collect();
                for k in 0..wires.len() {
                    let j = rng.random_range(k..wires.len());
                    wires.swap(k, j);
                }
                XGate::conj(t, wires[..w].iter().map(|&x| (x, rng.random_bool(0.5)))).unwrap()
            };
            let Some(corrs) = conj_wake(&u, &h, 12) else {
                // Mode c (mutual collision) — must really be mutual.
                assert!(h.reads(tu) && u.reads(h.target), "spurious refusal: {u:?} x {h:?}");
                refused += 1;
                continue;
            };
            let mut after = vec![h.clone()];
            after.extend(corrs.iter().cloned());
            assert!(
                rules::verify_rewrite(&[u.clone(), h.clone(), u.clone()], &after),
                "conjugate wrong: {u:?} x {h:?} -> {after:?}"
            );
            if corrs.is_empty() {
                seen[0] += 1;
            } else if h.reads(tu) {
                seen[1] += 1;
            } else {
                seen[2] += 1;
            }
        }
        assert!(
            seen.iter().all(|&c| c > 100) && refused > 0,
            "coverage too thin: {seen:?} refused={refused}"
        );
    }

    // Bridge insertion exactness without any store: plan a bridge on a real
    // random circuit, apply the insertions (wake corrections + the two
    // carrier copies), and demand exact global functional equality — the
    // telescoping identity g1·u·(u·M·u)·u·g2 = g1·M·g2, on material where
    // the interior genuinely collides with the carrier.
    #[test]
    fn bridge_insertion_preserves_function() {
        let mut planned = 0usize;
        let mut with_wake = 0usize;
        for seed in 0..20u64 {
            let gates = random_mixed_circuit(100 + seed, 12, 160);
            let params = MixParams {
                bridge_min_span: 8,
                bridge_max_span: 64,
                bridge_max_colliders: 12,
                report_every: u64::MAX,
                seed: 40 + seed,
                ..MixParams::default()
            };
            let mut mx = Mixer::new_with_db(gates, 12, params, FrozenDb::empty());
            for _ in 0..40 {
                let Some(plan) = mx.bridge_plan() else { continue };
                planned += 1;
                if !plan.wake.is_empty() {
                    with_wake += 1;
                }
                mx.bridge_insert(&plan);
                mx.global_check();
            }
        }
        assert!(planned > 20, "too few plans succeeded: {planned}");
        assert!(with_wake > 5, "no plan ever needed a wake: colliders untested");
    }

    // A bridge round against the empty store must leave literally no trace:
    // both endpoint probes run before anything mutates.
    #[test]
    fn bridge_round_empty_store_leaves_no_trace() {
        let gates = random_mixed_circuit(23, 16, 200);
        let params = MixParams {
            bridge_min_span: 8,
            bridge_max_span: 64,
            report_every: u64::MAX,
            seed: 9,
            ..MixParams::default()
        };
        let mut mx = Mixer::new_with_db(gates.clone(), 16, params, FrozenDb::empty());
        let before = mx.arena.to_vec();
        for _ in 0..100 {
            mx.bridge_round();
        }
        assert_eq!(mx.counters.bridge_rounds, 100);
        assert_eq!(mx.counters.bridge_committed, 0, "empty store cannot commit");
        assert_eq!(mx.counters.bridge_rollbacks, 0, "probe precedes every insertion");
        assert_eq!(mx.arena.to_vec(), before, "a probe miss must leave no trace");
    }
}
