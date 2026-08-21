// The splitting rules (gate_commutation_rules.html), rightward form: given the
// adjacent pair [g, h] with shot gate g (a pure conjunction; g57 shots are
// pre-split first) and colliding gate h, produce a functionally equal sequence.
// Leftward crossings use the inverse trick: every gate here is an involution, so
// [h, g] = reverse(rewrite of [g, h]); the engine reverses the emitted sequence.
//
// Residue widths are checked against `k_max` BEFORE anything is emitted; a
// Blocked outcome leaves the circuit untouched.
use crate::circuit::xgate::{Lits, XGate};
use rand::Rng;
use rand::seq::SliceRandom;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    // A piece of the shot gate: keeps floating in the shot's direction.
    ShotPiece,
    // A piece of the colliding gate: floats in the opposite direction.
    CollidingPiece,
    // The colliding gate itself, unchanged (R1/R3): keeps its arena node.
    CollidingIntact,
    // R3 stay-behind residue: permanently wedged against h, retire it.
    Core,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuleKind {
    R1,
    R2,
    R3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockReason {
    // Some residue would exceed k_max controls.
    WidthCap,
    // R3 with empty sensitivity set (facing-CNOT hard core) or no crossable
    // residue: the rewrite would make no progress.
    Deadlock,
}

#[derive(Debug)]
pub enum Outcome {
    // The pair does not actually collide (defensive; the engine floats to
    // collision before calling).
    R0Swap,
    // Replacement for [g, h], in rightward order, with enqueue roles.
    Rewrite {
        seq: Vec<(XGate, Role)>,
        kind: RuleKind,
        dropped: usize,
    },
    // h is a complemented gate (g57) that would have to split (R2): the engine
    // should pre-split h in place and retry.
    PresplitColliding,
    Blocked(BlockReason),
}

// Exclusive pre-split of a complemented gate: (t; NOT(m_1..m_r)) fires iff some
// literal fails; case-split on the FIRST failing literal (order randomized).
// piece_j = (t; m_1 .. m_{j-1} AND NOT m_j). Exactly one piece fires whenever
// the parent fires, none otherwise. Pieces share a target they never read, so
// they pairwise commute.
pub fn presplit(g: &XGate, rng: &mut impl Rng) -> Vec<XGate> {
    assert!(g.comp, "presplit is only for complemented (g57) gates");
    let mut m: Lits = g.ctrls.clone();
    m.shuffle(rng);
    let mut pieces = Vec::with_capacity(m.len());
    for j in 0..m.len() {
        let lits = m[..j].iter().copied().chain([(m[j].0, !m[j].1)]);
        // Distinct wires by construction: always satisfiable.
        pieces.push(XGate::conj(g.target, lits).expect("presplit piece contradiction"));
    }
    pieces
}

pub fn cross(g: &XGate, h: &XGate, k_max: usize, rng: &mut impl Rng) -> Outcome {
    assert!(
        !g.comp,
        "shot gate must be a conjunction (pre-split g57 shots first)"
    );
    let a = g.target;
    let b = h.target;
    if a == b {
        return Outcome::R0Swap;
    }
    let h_writes_g = g.reads(b);
    let g_writes_h = h.reads(a);
    match (h_writes_g, g_writes_h) {
        (false, false) => Outcome::R0Swap,
        (true, false) => r1(g, h, k_max, rng),
        (false, true) => {
            if h.comp {
                Outcome::PresplitColliding
            } else {
                r2(g, h, k_max, rng)
            }
        }
        (true, true) => r3(g, h, k_max, rng),
    }
}

// R1 — h writes a control of g; g splits into the first-failing-literal ladder
// over h's monomial and crosses; h stays intact (a colliding g57 costs only a
// polarity flip: comp swaps which rungs read b flipped).
fn r1(g: &XGate, h: &XGate, k_max: usize, rng: &mut impl Rng) -> Outcome {
    let a = g.target;
    let b = h.target;
    let pb = g.lit_on(b).unwrap();
    let s: Lits = g.ctrls_without(b);
    let mut m: Lits = h.ctrls.clone();
    m.shuffle(rng); // ladder order is a free choice
    let flip_full = !h.comp; // h's monomial holds => h fired iff comp=0
    let flip_rung = h.comp; //  some literal failed => h fired iff comp=1

    let mut residues: Vec<XGate> = Vec::with_capacity(m.len() + 1);
    let mut dropped = 0usize;
    // Rung "monomial held": b-literal (flipped per comp) + all of m + S.
    match XGate::conj(
        a,
        [(b, pb ^ flip_full)]
            .into_iter()
            .chain(m.iter().copied())
            .chain(s.iter().copied()),
    ) {
        Some(r) => residues.push(r),
        None => dropped += 1,
    }
    // Rung j: literals m_1..m_{j-1} held, m_j failed.
    for j in 0..m.len() {
        let lits = [(b, pb ^ flip_rung)]
            .into_iter()
            .chain(m[..j].iter().copied())
            .chain([(m[j].0, !m[j].1)])
            .chain(s.iter().copied());
        match XGate::conj(a, lits) {
            Some(r) => residues.push(r),
            None => dropped += 1,
        }
    }
    debug_assert!(
        !residues.is_empty(),
        "satisfiable shot gate lost all residues"
    );
    if residues.iter().any(|r| r.width() > k_max) {
        return Outcome::Blocked(BlockReason::WidthCap);
    }
    let mut seq = vec![(h.clone(), Role::CollidingIntact)];
    seq.extend(residues.into_iter().map(|r| (r, Role::ShotPiece)));
    Outcome::Rewrite {
        seq,
        kind: RuleKind::R1,
        dropped,
    }
}

// R2 (strict mirror) — g writes a control of h; g passes intact and h splits,
// pushed to the other side. h is a conjunction here (g57 colliders are
// pre-split by the engine before this runs). Pieces case-split on whether g is
// about to flip a, via the first-failing literal of g's control F.
fn r2(g: &XGate, h: &XGate, k_max: usize, rng: &mut impl Rng) -> Outcome {
    let a = g.target;
    let b = h.target;
    let pa = h.lit_on(a).unwrap();
    let r: Lits = h.ctrls_without(a);
    let mut f: Lits = g.ctrls.clone();
    f.shuffle(rng);

    let mut pieces: Vec<XGate> = Vec::with_capacity(f.len() + 1);
    let mut dropped = 0usize;
    // F held: g flips a, so the a-literal reads flipped.
    match XGate::conj(
        b,
        [(a, !pa)]
            .into_iter()
            .chain(f.iter().copied())
            .chain(r.iter().copied()),
    ) {
        Some(p) => pieces.push(p),
        None => dropped += 1,
    }
    for j in 0..f.len() {
        let lits = [(a, pa)]
            .into_iter()
            .chain(f[..j].iter().copied())
            .chain([(f[j].0, !f[j].1)])
            .chain(r.iter().copied());
        match XGate::conj(b, lits) {
            Some(p) => pieces.push(p),
            None => dropped += 1,
        }
    }
    debug_assert!(
        !pieces.is_empty(),
        "satisfiable colliding gate lost all pieces"
    );
    if pieces.iter().any(|p| p.width() > k_max) {
        return Outcome::Blocked(BlockReason::WidthCap);
    }
    let mut seq: Vec<(XGate, Role)> = pieces
        .into_iter()
        .map(|p| (p, Role::CollidingPiece))
        .collect();
    seq.push((g.clone(), Role::ShotPiece));
    Outcome::Rewrite {
        seq,
        kind: RuleKind::R2,
        dropped,
    }
}

// R3 (primary form) — mutual collision; h stays intact. g splits into the
// stay-behind residue (g's firing restricted to h's sensitivity set R, executes
// before h, permanently wedged) and crossed residues on NOT R (first-failing
// literal ladder; b-literal flipped iff h is complemented, since on NOT R a
// complemented h always fires).
fn r3(g: &XGate, h: &XGate, k_max: usize, rng: &mut impl Rng) -> Outcome {
    let a = g.target;
    let b = h.target;
    let pb = g.lit_on(b).unwrap();
    let s: Lits = g.ctrls_without(b);
    let mut r: Lits = h.ctrls_without(a);
    if r.is_empty() {
        // h reads only a: always sensitive, nothing crosses (facing-CNOT core).
        return Outcome::Blocked(BlockReason::Deadlock);
    }
    r.shuffle(rng);

    let mut dropped = 0usize;
    let stay = match XGate::conj(
        a,
        [(b, pb)]
            .into_iter()
            .chain(s.iter().copied())
            .chain(r.iter().copied()),
    ) {
        Some(g) => Some(g),
        None => {
            dropped += 1;
            None
        }
    };
    let mut crossed: Vec<XGate> = Vec::with_capacity(r.len());
    for j in 0..r.len() {
        let lits = [(b, pb ^ h.comp)]
            .into_iter()
            .chain(r[..j].iter().copied())
            .chain([(r[j].0, !r[j].1)])
            .chain(s.iter().copied());
        match XGate::conj(a, lits) {
            Some(g) => crossed.push(g),
            None => dropped += 1,
        }
    }
    if crossed.is_empty() {
        // Nothing would cross: progress-free rewrite, leave the pair alone.
        return Outcome::Blocked(BlockReason::Deadlock);
    }
    if stay.iter().chain(crossed.iter()).any(|g| g.width() > k_max) {
        return Outcome::Blocked(BlockReason::WidthCap);
    }
    let mut seq: Vec<(XGate, Role)> = Vec::with_capacity(crossed.len() + 2);
    if let Some(st) = stay {
        seq.push((st, Role::Core));
    }
    seq.push((h.clone(), Role::CollidingIntact));
    seq.extend(crossed.into_iter().map(|g| (g, Role::ShotPiece)));
    Outcome::Rewrite {
        seq,
        kind: RuleKind::R3,
        dropped,
    }
}

// Exhaustive local equivalence check over the support wires of both sides.
// Support is small (two gates of width <= k_max), so 2^support is cheap; run on
// every rewrite in production. Returns true iff the two sequences compute the
// same function on all wires.
pub fn verify_rewrite(before: &[XGate], after: &[XGate]) -> bool {
    let mut wires: Vec<u16> = Vec::new();
    for g in before.iter().chain(after.iter()) {
        wires.push(g.target);
        wires.extend(g.ctrls.iter().map(|&(w, _)| w));
    }
    wires.sort_unstable();
    wires.dedup();
    let k = wires.len();
    assert!(k <= 24, "rewrite support unexpectedly large: {k} wires");
    let dense = |g: &XGate| {
        let map = |w: u16| wires.binary_search(&w).unwrap() as u16;
        XGate {
            target: map(g.target),
            comp: g.comp,
            ctrls: g.ctrls.iter().map(|&(w, p)| (map(w), p)).collect(),
        }
    };
    let bg: Vec<XGate> = before.iter().map(dense).collect();
    let ag: Vec<XGate> = after.iter().map(dense).collect();

    let total: u64 = 1u64 << k;
    // 64 assignments per batch: lane l is assignment v + l, i.e. bit l of wire
    // i's lane word is ((v+l)>>i)&1. With v a multiple of 64 the low six wires
    // are the constant patterns below and each higher wire is all-ones or
    // all-zeros according to bit i of v (no carry: l < 64).
    const LANE: [u64; 6] = [
        0xAAAA_AAAA_AAAA_AAAA,
        0xCCCC_CCCC_CCCC_CCCC,
        0xF0F0_F0F0_F0F0_F0F0,
        0xFF00_FF00_FF00_FF00,
        0xFFFF_0000_FFFF_0000,
        0xFFFF_FFFF_0000_0000,
    ];
    let mut st_b = vec![0u64; k];
    let mut st_a = vec![0u64; k];
    let mut v = 0u64;
    while v < total {
        for i in 0..k.min(6) {
            st_b[i] = LANE[i];
        }
        for i in 6..k {
            st_b[i] = 0u64.wrapping_sub((v >> i) & 1);
        }
        st_a.copy_from_slice(&st_b);
        crate::circuit::xgate::eval_lanes(&bg, &mut st_b);
        crate::circuit::xgate::eval_lanes(&ag, &mut st_a);
        // Ignore lanes beyond `total` when k < 6.
        let valid: u64 = if total - v >= 64 {
            !0
        } else {
            (1u64 << (total - v)) - 1
        };
        for i in 0..k {
            if (st_b[i] ^ st_a[i]) & valid != 0 {
                return false;
            }
        }
        v += 64;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    // The hoisted constant lane basis in verify_rewrite must match the
    // original per-bit construction for every wire index and batch base.
    #[test]
    fn opt_equiv_lane_basis_matches_bitwise_construction() {
        const LANE: [u64; 6] = [
            0xAAAA_AAAA_AAAA_AAAA,
            0xCCCC_CCCC_CCCC_CCCC,
            0xF0F0_F0F0_F0F0_F0F0,
            0xFF00_FF00_FF00_FF00,
            0xFFFF_0000_FFFF_0000,
            0xFFFF_FFFF_0000_0000,
        ];
        for k in 1usize..=12 {
            let total = 1u64 << k;
            let mut v = 0u64;
            while v < total {
                for i in 0..k {
                    let mut acc = 0u64;
                    for l in 0..64u64 {
                        if ((v + l) >> i) & 1 == 1 {
                            acc |= 1 << l;
                        }
                    }
                    let fast = if i < 6 {
                        LANE[i]
                    } else {
                        0u64.wrapping_sub((v >> i) & 1)
                    };
                    assert_eq!(acc, fast, "k={k} v={v} i={i}");
                }
                v += 64;
            }
        }
    }
}
