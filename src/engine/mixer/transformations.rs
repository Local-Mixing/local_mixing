//! Verified expansion, twist placement and bridge transformations over mixer state.
use super::*;

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
        if g.comp && !g.ctrls.is_empty() {
            Some(g.target)
        } else {
            None
        }
    },
}];

/// Largest window/replacement length tracked in the splice size histogram;
/// the carrier, and the exact conjugation wake for every interior collider.
pub(super) struct BridgePlan {
    pub(super) g1: u32,
    pub(super) g2: u32,
    pub(super) g1g: XGate,
    pub(super) g2g: XGate,
    pub(super) u: XGate,
    /// (interior collider id, its correction gates): the conjugate u·h·u is
    /// [h, corrections], verified exhaustively per collider before commit.
    pub(super) wake: Vec<(u32, Vec<XGate>)>,
    pub(super) interior_len: usize,
}

/// Outcome of merging a correction gate's literal list.
pub(super) enum MergedConj {
    Gate(XGate),
    /// Contradictory literals: the correction never fires — the conjugate is
    /// the original gate unchanged.
    Never,
    /// Unbuildable (target among controls, or wider than the K-cap).
    Invalid,
}

pub(super) fn merged_conj(target: u16, lits: &[(u16, bool)], k_max: usize) -> MergedConj {
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
pub(super) fn conj_wake(u: &XGate, h: &XGate, k_max: usize) -> Option<Vec<XGate>> {
    debug_assert!(
        !u.comp && u.ctrls.len() == 2,
        "carrier is a 2-control conjunction"
    );
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
        let rho = u
            .ctrls
            .iter()
            .copied()
            .find(|&(w, _)| w != h.target)
            .expect("distinct wires");
        if h.comp {
            match merged_conj(tu, &[rho], k_max) {
                MergedConj::Gate(c) => corrs.push(c),
                MergedConj::Never => {}
                MergedConj::Invalid => return None,
            }
        }
        let lits: Vec<(u16, bool)> = h
            .ctrls
            .iter()
            .copied()
            .chain(std::iter::once(rho))
            .collect();
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

/// One planned bracket seam: the window boundary node it ends up at (possibly
/// slid outward from the drawn boundary), the context gates it consumes, and
/// the word that replaces context-plus-bracket.
pub(super) struct TgSeam {
    pub(super) edge: u32,
    pub(super) ids: Vec<u32>,
    pub(super) repl: Vec<XGate>,
}

impl TgSeam {
    pub(super) fn net(&self) -> i64 {
        self.repl.len() as i64 - self.ids.len() as i64
    }
}

/// A fully-evaluated candidate twist: wires, final window, both seams.
pub(super) struct TgPlan {
    pub(super) a: u16,
    pub(super) b: u16,
    pub(super) l: TgSeam,
    pub(super) r: TgSeam,
    pub(super) slides: u64,
}

impl TgPlan {
    pub(super) fn score(&self) -> (i64, i64) {
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
    let (w, _p) = extra?;
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
pub(crate) fn merge_key(target: u16, wires: impl Iterator<Item = u16>) -> u64 {
    // FxHasher: the key is internal-only (never serialized) and collisions are
    // rechecked (above), so the hash function choice is unobservable.
    let mut h = rustc_hash::FxHasher::default();
    target.hash(&mut h);
    for w in wires {
        w.hash(&mut h);
    }
    h.finish()
}

pub(crate) fn key_of(g: &XGate) -> u64 {
    merge_key(g.target, g.ctrls.iter().map(|&(w, _)| w))
}

// Conjugation of a single gate by NOT(w): N g N. A gate reading w sees the
// wire flipped on both sides, which is exactly a polarity flip of its
// w-literal; a gate TARGETING w is invariant (the two flips cancel through the
// XOR: !(!a ^ f) = a ^ f). Width and comp are preserved. None = g is invariant.

pub(super) fn conj_by_not(g: &XGate, w: u16) -> Option<XGate> {
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
pub(super) fn conj_by_swap(g: &XGate, a: u16, b: u16) -> Option<XGate> {
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
#[cfg(test)]
pub(super) enum CnotConj {
    Invariant,
    Flip(XGate),
    Split(XGate, XGate),
    Blocked,
}

#[cfg(test)]
pub(super) fn conj_by_cnot(g: &XGate, a: u16, b: u16) -> CnotConj {
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

impl Mixer {
    // ---- expansion moves ----

    pub(super) fn expand_move(&mut self) {
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

    /// Look for a welcoming neighbourhood: sample up to `twist_place_tries`
    /// candidate positions and return the first that matches any entry of
    /// TWIST_PATTERNS, together with the wire that pattern prefers. `None`
    /// means no candidate matched and the caller should place the twist
    /// uniformly at random, exactly as before.
    pub(super) fn find_twist_site(&mut self) -> Option<(u32, u16)> {
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
    pub(super) fn twist_move(&mut self) {
        let n = self.arena.len();
        if n < 2 {
            return;
        }
        let cap = n;
        let lmin = (self.params.twist_min_len.max(2).min(cap)) as f64;
        let len = (self
            .rng
            .random_range(lmin.ln()..=(cap as f64).ln())
            .exp()
            .round() as usize)
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
        let packet: Vec<XGate> = vec![
            pol_cnot(b, a, alpha),
            pol_cnot(a, b, false),
            pol_cnot(b, a, beta),
        ];
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
            self.set_meta(
                anchor,
                Meta {
                    origin: ORIGIN_SYNTH,
                    event: ev,
                    dir: d,
                    dgen: GEN_FRESH,
                    litter: lit,
                    litter_size: 1,
                },
            );
        }
        let mut anchor = end;
        for g in &packet_inv {
            self.counters.width_hist[g.width().min(15)] += 1;
            anchor = self.arena.insert_after(anchor, g.clone());
            self.index_add(anchor);
            let d = self.rand_dir();
            let lit = self.fresh_litter();
            self.set_meta(
                anchor,
                Meta {
                    origin: ORIGIN_SYNTH,
                    event: ev,
                    dir: d,
                    dgen: GEN_FRESH,
                    litter: lit,
                    litter_size: 1,
                },
            );
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
    pub(super) fn solve_seam(
        &mut self,
        edge: u32,
        dir: Dir,
        a: u16,
        b: u16,
    ) -> (Vec<u32>, Vec<XGate>) {
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
                let ctrls: Vec<(u8, bool)> = g.ctrls.iter().map(|&(w, p)| (abs_of(w), p)).collect();
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
                    eng.solve(t, swap_words::MAX_WORD)
                        .map(smallvec::SmallVec::from_vec)
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
    pub(super) fn slide_candidates(&self, from: u32, dir: Dir, a: u16, b: u16) -> Vec<u32> {
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
    pub(super) fn eval_seam(&mut self, edge: u32, dir: Dir, a: u16, b: u16) -> (TgSeam, u64) {
        let (ids, repl) = self.solve_seam(edge, dir, a, b);
        let mut seam = TgSeam { edge, ids, repl };
        let mut slid = 0u64;
        if self.runtime.twist_slide() && seam.net() >= 6 {
            for e in self.slide_candidates(edge, dir, a, b) {
                let (ids2, repl2) = self.solve_seam(e, dir, a, b);
                let cand = TgSeam {
                    edge: e,
                    ids: ids2,
                    repl: repl2,
                };
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
    pub(super) fn twist_move_g57(&mut self) {
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
        let attempts = if self.runtime.twist_retry() {
            TG_RETRIES
        } else {
            1
        };
        for _ in 0..attempts {
            draws += 1;
            let n = self.arena.len();
            let cap = n;
            let lmin = (self.params.twist_min_len.max(2).min(cap)) as f64;
            let len = (self
                .rng
                .random_range(lmin.ln()..=(cap as f64).ln())
                .exp()
                .round() as usize)
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
            for edge in [
                self.arena.neighbor(start, Dir::L),
                self.arena.neighbor(end, Dir::R),
            ] {
                if edge == NIL {
                    continue;
                }
                let g = self.arena.gate(edge);
                let pins: Vec<u16> = std::iter::once(g.target)
                    .chain(g.ctrls.iter().map(|&(w, _)| w))
                    .collect();
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
                    l: TgSeam {
                        edge: start,
                        ids: l_ids,
                        repl: l_repl,
                    },
                    r: TgSeam {
                        edge: end,
                        ids: r_ids,
                        repl: r_repl,
                    },
                    slides: 0,
                };
                if pair_best
                    .as_ref()
                    .map_or(true, |p| plan.score() < p.score())
                {
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
        let packet3 = vec![XGate::cnot(b, a), XGate::cnot(a, b), XGate::cnot(b, a)];

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
            let mut old: Vec<XGate> = plan
                .l
                .ids
                .iter()
                .rev()
                .map(|&id| self.arena.gate(id).clone())
                .collect();
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
        let mut inserted: Vec<u32> = Vec::with_capacity(plan.l.repl.len() + plan.r.repl.len());
        let l_anchor = match plan.l.ids.last() {
            Some(&far) => self.arena.neighbor(far, Dir::L),
            None => self.arena.neighbor(plan.l.edge, Dir::L),
        };
        // Consumed context carried real lineage: union its litters' ancestor
        // sets into the replacement's litter, exactly as a DB splice would —
        // v1 dropped them, which silently deflated anc under consumption.
        let mut l_srcs: Vec<u64> = plan
            .l
            .ids
            .iter()
            .map(|&id| self.meta_of(id).litter)
            .collect();
        l_srcs.sort_unstable();
        l_srcs.dedup();
        let mut r_srcs: Vec<u64> = plan
            .r
            .ids
            .iter()
            .map(|&id| self.meta_of(id).litter)
            .collect();
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
            self.set_meta(
                anchor,
                Meta {
                    origin: ORIGIN_SYNTH,
                    event: ev,
                    dir: Dir::L,
                    dgen: GEN_FRESH,
                    litter: l_lit,
                    litter_size: plan.l.repl.len() as u16,
                },
            );
            inserted.push(anchor);
        }
        let mut anchor = plan.r.edge;
        for g in &plan.r.repl {
            self.counters.width_hist[g.width().min(15)] += 1;
            anchor = self.arena.insert_after(anchor, g.clone());
            self.index_add(anchor);
            self.set_meta(
                anchor,
                Meta {
                    origin: ORIGIN_SYNTH,
                    event: ev,
                    dir: Dir::R,
                    dgen: GEN_FRESH,
                    litter: r_lit,
                    litter_size: plan.r.repl.len() as u16,
                },
            );
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
    pub(super) fn bridge_plan(&mut self) -> Option<BridgePlan> {
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
        Some(BridgePlan {
            g1,
            g2,
            g1g,
            g2g,
            u,
            wake,
            interior_len: interior.len(),
        })
    }

    /// Commit the insertions of a bridge plan: the wake corrections (each
    /// immediately after its collider, which it commutes with) and the two
    /// carrier copies (after g1, before g2). Function-preserving by the
    /// telescoping identity; every inserted gate is returned so a declined
    /// far-window splice can roll the circuit back exactly.
    pub(super) fn bridge_insert(&mut self, plan: &BridgePlan) -> (u32, u32, Vec<u32>) {
        let ev = self.fresh_event();
        let mut inserted: Vec<u32> = Vec::new();
        let stamp = |mx: &mut Self, id: u32| {
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
        let u2 = self
            .arena
            .insert_after(self.arena.neighbor(plan.g2, Dir::L), plan.u.clone());
        stamp(self, u2);
        inserted.push(u2);
        let u1 = self.arena.insert_after(plan.g1, plan.u.clone());
        stamp(self, u1);
        inserted.push(u1);
        (u1, u2, inserted)
    }

    pub(super) fn bridge_round(&mut self) {
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
        self.counters.bridge_span_max = self.counters.bridge_span_max.max(plan.interior_len as u64);
        self.counters.bridge_colliders_sum += plan.wake.len() as u64;
    }

    // Float the contiguous span [lo..hi] toward `dir` past commuting neighbors,
    // returning the first neighbor that collides with the block (and `dir`), or
    // (NIL, dir) at the boundary. Commuting neighbors are hopped to the far side.
}
