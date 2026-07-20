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
    pub w_db: f64,
    pub p_db: f64,
    pub db_min_window: usize,
    pub db_max_window: usize,
    // Window sampling geometry and its guards (see DbSample / db_attempt).
    // db_ctrl_cap (L): while building a window, a gate with more than L controls
    // is evaded (floated out of the way, else the build reverses, else aborts) so
    // high-degree gates that always miss are kept out of the window. 0 = no cap.
    // db_convex_p: for Convex, the probability each growth step floats the block
    // in g1's original direction (else the opposite).
    pub db_sample: DbSample,
    pub db_ctrl_cap: usize,
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
            w_db: 0.0,
            p_db: 0.0,
            db_min_window: 2,
            db_max_window: 12,
            db_sample: DbSample::Contiguous,
            db_ctrl_cap: 0,
            db_convex_p: 0.75,
            db_verify: true,
            db_dry_run: false,
            db_max_degree: 0,
            db_degree_probes: 6,
            db_max_span: 0,
            db_wire_terms: 0,
            db_total_terms: 0,
            db_prefixes: false,
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
    misses: u8,        // failed gather attempts; entry dropped after a few
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
}

pub enum MixStop {
    MovesBudget,
    StopFlag,
}

impl Mixer {
    pub fn new(gates: Vec<XGate>, num_wires: usize, params: MixParams) -> Mixer {
        // Open the replacement store once, only when a DB move is enabled, so
        // runs without them never require FROZEN_DB_DIR.
        let db = if params.w_db > 0.0 || params.p_db > 0.0 {
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
            })
            .collect();
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

    fn set_meta(&mut self, id: u32, m: Meta) {
        let i = id as usize;
        if i >= self.meta.len() {
            self.meta.resize(i + 1, Meta { origin: ORIGIN_SYNTH, event: 0, dir: Dir::R });
        }
        self.meta[i] = m;
    }

    fn meta_of(&self, id: u32) -> Meta {
        self.meta
            .get(id as usize)
            .copied()
            .unwrap_or(Meta { origin: ORIGIN_SYNTH, event: 0, dir: Dir::R })
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

    pub fn run(&mut self) -> MixStop {
        while self.moves_done < self.params.moves {
            // Top-level: with probability p_db the whole round is a size-agnostic
            // DB replacement move (a uniform random equivalent of any gate count),
            // regardless of the contract/expand decision. On a miss the round is
            // spent (no fallthrough) — that IS the chosen move.
            let took_agnostic = self.params.p_db > 0.0
                && self.arena.len() >= self.params.db_min_window.max(2)
                && self.rng.random_bool(self.params.p_db.clamp(0.0, 1.0))
                && { self.db_attempt(DbMode::SizeAgnostic); true };
            if !took_agnostic {
                let excess = self.arena.len() as f64 - self.params.target_size as f64;
                let p_contract =
                    (1.0 / (1.0 + (-excess / self.params.temp).exp())).clamp(0.02, 0.98);
                // Nothing to contract below two gates; every contraction channel
                // samples a linked node, so guard the empty/singleton arena (which
                // a DB move can reach on a near-identity region).
                if self.arena.len() >= 2 && self.rng.random_bool(p_contract) {
                    // Contraction channels with complementary stock; when one finds
                    // nothing, fall through to the next rather than wasting the move.
                    // The compressing DB replacement is tried first with probability
                    // w_db (the only channel that can contract non-ladder material),
                    // then undo/merge as before.
                    let did_db = self.params.w_db > 0.0
                        && self.rng.random_bool(self.params.w_db.clamp(0.0, 1.0))
                        && self.db_attempt(DbMode::Compressing);
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
                if self.stop_requested {
                    self.global_check();
                    return MixStop::StopFlag;
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
        let total = p.w_cross
            + p.w_fresh
            + p.w_unsub
            + p.w_insert
            + p.w_twist_neg
            + p.w_twist_swap
            + p.w_twist_cnot;
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
        if p.w_twist_neg > 0.0 && r - p.w_insert < p.w_twist_neg && r >= p.w_insert {
            self.twist_move(TwistKind::Neg);
            return;
        }
        let ns = p.w_insert + p.w_twist_neg;
        if p.w_twist_swap > 0.0 && r - ns < p.w_twist_swap && r >= ns {
            self.twist_move(TwistKind::Swap);
            return;
        }
        if p.w_twist_cnot > 0.0 && r >= ns + p.w_twist_swap {
            self.twist_move(TwistKind::Cnot);
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
            let origin = self.meta_of(id).origin;
            let ev = self.fresh_event();
            for p in &pieces {
                self.counters.width_hist[p.width().min(15)] += 1;
            }
            let ids = self.splice_replace_one(id, pieces);
            for &pid in &ids {
                let d = self.child_dir(dir);
                self.set_meta(pid, Meta { origin, event: ev, dir: d });
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
                let origin = self.meta_of(h_id).origin;
                let ev = self.fresh_event();
                for p in &hp {
                    self.counters.width_hist[p.width().min(15)] += 1;
                }
                let ids = self.splice_replace_one(h_id, hp);
                for &pid in &ids {
                    // Colliding-gate fragments still inherit from the SHOT
                    // gate's direction (per spec: regardless of parent).
                    let d = self.child_dir(dir);
                    self.set_meta(pid, Meta { origin, event: ev, dir: d });
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
                let g_origin = self.meta_of(id).origin;
                let h_origin = self.meta_of(h_id).origin;
                let ev = self.fresh_event();
                let placed = self.splice_pair(id, h_id, dir, seq);
                let mut fresh: Vec<u32> = Vec::new();
                for &(pid, role) in &placed {
                    match role {
                        Role::ShotPiece | Role::Core => {
                            let d = self.child_dir(dir);
                            self.set_meta(pid, Meta { origin: g_origin, event: ev, dir: d });
                            fresh.push(pid);
                        }
                        Role::CollidingPiece => {
                            let d = self.child_dir(dir);
                            self.set_meta(pid, Meta { origin: h_origin, event: ev, dir: d });
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
                    let (before, origins) = match dir {
                        Dir::R => ([g.clone(), h.clone()], [g_origin, h_origin]),
                        Dir::L => ([h.clone(), g.clone()], [h_origin, g_origin]),
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
                        misses: 0,
                    });
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
            self.set_meta(pid, Meta { origin: m.origin, event: ev, dir: d });
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
        if !self.split_allowed(g.width()) {
            self.counters.declined += 1;
            return;
        }
        let j = self.rng.random_range(0..g.ctrls.len());
        let (w, p) = g.ctrls[j];
        let without = XGate::conj(g.target, g.ctrls_without(w)).expect("subset is satisfiable");
        let flipped = XGate::conj(
            g.target,
            g.ctrls.iter().map(|&(cw, cp)| if cw == w { (cw, !cp) } else { (cw, cp) }),
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
            self.set_meta(pid, Meta { origin: m.origin, event: ev, dir: d });
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
        self.set_meta(a, Meta { origin: ORIGIN_SYNTH, event: ev, dir: da });
        self.set_meta(b, Meta { origin: ORIGIN_SYNTH, event: ev, dir: da.opposite() });
        self.counters.width_hist[self.arena.gate(a).width().min(15)] += 2;
        self.counters.inserts += 1;
        // Each copy is shot once as part of the insert.
        self.cross_move_on(a);
        if self.arena.is_linked(b) {
            self.cross_move_on(b);
        }
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
                        self.set_meta(pid, Meta { origin: m.origin, event: ev, dir: m.dir });
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
            self.set_meta(anchor, Meta { origin: ORIGIN_SYNTH, event: ev, dir: d });
        }
        let mut anchor = w_end;
        for g in &packet {
            self.counters.width_hist[g.width().min(15)] += 1;
            anchor = self.arena.insert_after(anchor, g.clone());
            self.index_add(anchor);
            let d = self.rand_dir();
            self.set_meta(anchor, Meta { origin: ORIGIN_SYNTH, event: ev, dir: d });
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
            self.set_meta(c, Meta { origin: e.origins[j], event: 0, dir: d });
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
            let d = self.child_dir(mg.dir);
            self.set_meta(nid, Meta { origin, event: 0, dir: d });
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
    fn db_attempt(&mut self, mode: DbMode) -> bool {
        let n = self.arena.len();
        let wmin = self.params.db_min_window.max(2);
        let wmax = self.params.db_max_window.max(wmin);
        if n < wmin {
            return false;
        }
        // Prefix descent always starts at the top of the range — the descent
        // itself visits every shorter length, so sampling a shorter start
        // would only duplicate coverage.
        let len = if self.params.db_prefixes || wmin == wmax {
            wmax.min(n)
        } else {
            self.rng.random_range(wmin..=wmax.min(n))
        };

        // Sample the window (contiguous or convex); g1dir drives the incoming
        // direction pivot below.
        let Some((ids, g1dir, smp)) = self.sample_window(len) else {
            return false;
        };
        // Stamped into every --db-record attempt line (smp=ctg|cvx) so stats
        // can split hits by sampler geometry, esp. under --db-sample mixed.
        self.db_last_sampler = smp;
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
            let wmin = self.params.db_min_window.max(2);
            let guard = DegreeGuard {
                max_degree: self.params.db_max_degree,
                probes: self.params.db_degree_probes,
            };
            for p in (wmin..=window.len()).rev() {
                let prefix = &window[..p];
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
                    &mut self.rng,
                );
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
                if self.try_db_splice(&ids[..p], g1dir, prefix, replacement, res.match_count, mode)
                {
                    return true;
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
                DbMode::SizeAgnostic => self.counters.db_agn_misses += 1,
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
            &mut self.rng,
        );
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
                    DbMode::SizeAgnostic => self.counters.db_agn_hits += 1,
                }
            } else {
                match mode {
                    DbMode::Compressing => self.counters.db_comp_misses += 1,
                    DbMode::SizeAgnostic => self.counters.db_agn_misses += 1,
                }
            }
            return false;
        }

        let Some(replacement) = res.chosen else {
            self.record_db_attempt(&window, match_count, None);
            self.count_db_miss(mode);
            return false;
        };
        self.try_db_splice(&ids, g1dir, &window, replacement, match_count, mode)
    }

    fn count_db_hit(&mut self, mode: DbMode) {
        match mode {
            DbMode::Compressing => self.counters.db_comp_hits += 1,
            DbMode::SizeAgnostic => self.counters.db_agn_hits += 1,
        }
    }

    fn count_db_miss(&mut self, mode: DbMode) {
        match mode {
            DbMode::Compressing => self.counters.db_comp_misses += 1,
            DbMode::SizeAgnostic => self.counters.db_agn_misses += 1,
        }
    }

    // Verify (optionally), record, and splice `replacement` over the window
    // nodes `ids` (whose gates are `window`). Returns false only when the
    // combined support exceeds verify_rewrite's 24-wire cap with verification
    // on — declined rather than spliced unchecked, recorded as a miss.
    fn try_db_splice(
        &mut self,
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
            assert!(
                rules::verify_rewrite(window, &replacement),
                "DB replacement verification failed: {window:?} -> {replacement:?}"
            );
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
        let mut c = cursor;
        for (i, gate) in replacement.into_iter().enumerate() {
            c = self.arena.insert_after(c, gate);
            self.index_add(c);
            let d = if i <= pivot { Dir::L } else { Dir::R };
            self.set_meta(c, Meta { origin, event: 0, dir: d });
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
    fn sample_window(&mut self, w: usize) -> Option<(Vec<u32>, Dir, DbSample)> {
        match self.params.db_sample {
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

    // Seed gate for a window: a random linked node whose width is within the
    // control cap (retried a few times), or None if the cap keeps rejecting.
    fn pick_seed(&mut self) -> Option<u32> {
        let cap = self.params.db_ctrl_cap;
        for _ in 0..8 {
            let g = self.arena.random_linked(&mut self.rng);
            if cap == 0 || self.width_of(g) <= cap {
                return Some(g);
            }
        }
        None
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
        let cap = self.params.db_ctrl_cap;
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
            if cap > 0 && self.width_of(x) > cap {
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
        if ids.len() < self.params.db_min_window.max(2) {
            return None;
        }
        Some((ids, dir1))
    }

    // Convex: float g1 to its first collider, then grow the block by floating it
    // (in dir1 w.p. p, else the opposite) to the next collider and absorbing it.
    // The L-cap evades a wide collider the same way (float it away; else reverse;
    // else abort).
    fn collect_convex(&mut self, w: usize) -> Option<(Vec<u32>, Dir)> {
        let cap = self.params.db_ctrl_cap;
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
            if cap > 0 && self.width_of(g3) > cap {
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
                if g3r == NIL || (cap > 0 && self.width_of(g3r) > cap) {
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
        if ids.len() < self.params.db_min_window.max(2) {
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

    pub fn report(&mut self) {
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
        let c = &self.counters;
        let hist: Vec<String> = (0..=self.params.k_max.min(15))
            .map(|w| format!("{}:{}", w, c.width_hist[w]))
            .collect();
        println!(
            "[fmix] mv={} size={} target={} comp={} | merges c={} x={} d={} s={} sib={} xorig={} tabu={} nopart={} wall={} far={} noadj={} | undo ok={} dead={} tabu={} miss={} live={} | db comp={}/{} agn={}/{} rm={} add={} wide={} dsk={} ssk={} bab={} | expand r1={} r2={} r3={} pre={} fresh={} unsub={} ins={} twn={} tws={} twc={} twrel={} twsplit={} twspan={} twskip={} | declined={} blockw={} dl={} bnd={} | floats={}/{} scat={}/{} | disp={:.4} owin={:.1} fan0={:.3} leew={:.0} odiff={:.4} oadj={:.4} width[{}]",
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
            hist.join(" ")
        );
    }

    pub fn origins_in_order(&self) -> Vec<u32> {
        self.arena.ids_in_order().iter().map(|&id| self.meta_of(id).origin).collect()
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
        assert!(mx.counters.inserts > 50, "inserts barely ran: {}", mx.counters.inserts);
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
                db_sample: sample,
                db_ctrl_cap: 2, // no window gate may exceed width 2
                db_min_window: 2,
                db_max_window: 6,
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
        let params = MixParams { db_min_window: 2, report_every: u64::MAX, seed: 2, ..MixParams::default() };
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
}
