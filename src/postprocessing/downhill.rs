// Conjugation descent ("downhill"): the exact inverse of the mixer's R1
// crossing move, usable inside fcompress because it needs nothing but the
// released circuit (deterministic, attacker-computable — so like the rest of
// fcompress it can only make the reported "effective size" more honest).
//
// For a maximal same-target run B (target t) and an adjacent gate h that does
// not read t, commuting B across h rewrites B's ESOP by the substitution
// b <- b XOR fire(h) on h's target wire b (h itself is unchanged: B writes
// only t and h does not read t).  Forward mixing uses that substitution to
// expand one cube into a case-split ladder; applied in the profitable
// direction it collapses such ladders again.  A pass scans both neighbors of
// every maximal run, keeps candidates whose catalogue-reduced ESOP strictly
// shrinks in (gates, lits), applies a maximal non-overlapping subset, and
// optionally lane-verifies every rewritten span against its original.
//
// Shared by the fmix_downhill binary (scan/report/apply on a whole file) and
// by postprocessing::compress, which interleaves one pass per gather/reduce
// iteration — gathering makes runs contiguous, which is what feeds this pass.
use super::compress::{AncBits, or_anc};
use crate::engine::mix::{Merge, merge_result};
use crate::circuit::xgate::{Lits, XGate, eval_lanes};
use rand::Rng;
use rand::rngs::StdRng;

#[derive(Clone)]
struct Esop {
    parity: bool,
    cubes: Vec<XGate>,
}

#[derive(Clone, Debug)]
pub struct Candidate {
    pub lo: usize,
    pub hi: usize,
    pub neighbor: usize,
    pub side: &'static str,
    pub block_target: u16,
    pub neighbor_target: u16,
    pub before_gates: usize,
    pub after_gates: usize,
    pub before_lits: usize,
    pub after_lits: usize,
    pub span_lo: usize,
    pub span_hi: usize,
    pub replacement: Vec<XGate>,
}

fn xor_add(esop: &mut Esop, mut cube: XGate) {
    debug_assert!(!cube.comp);
    if cube.ctrls.is_empty() {
        esop.parity ^= true;
        return;
    }
    cube.comp = false;
    if let Some(i) = esop.cubes.iter().position(|g| g == &cube) {
        esop.cubes.swap_remove(i);
    } else {
        esop.cubes.push(cube);
    }
}

fn from_block(block: &[XGate], target: u16) -> Esop {
    let mut out = Esop {
        parity: false,
        cubes: Vec::new(),
    };
    for g in block {
        debug_assert_eq!(g.target, target);
        out.parity ^= g.comp;
        xor_add(
            &mut out,
            XGate {
                target,
                comp: false,
                ctrls: g.ctrls.clone(),
            },
        );
    }
    reduce(out)
}

fn reduce(mut esop: Esop) -> Esop {
    loop {
        let mut found = None;
        'outer: for i in 0..esop.cubes.len() {
            for j in i + 1..esop.cubes.len() {
                if let Some(m) = merge_result(&esop.cubes[i], &esop.cubes[j]) {
                    found = Some((i, j, m));
                    break 'outer;
                }
            }
        }
        let Some((i, j, merge)) = found else { break };
        esop.cubes.swap_remove(j);
        esop.cubes.swap_remove(i);
        match merge {
            Merge::Cancel => {}
            Merge::XFuse(g) | Merge::DropLit(g) | Merge::Subsume(g) | Merge::Absorb(g) => {
                xor_add(&mut esop, g)
            }
        }
    }
    esop
}

fn product(target: u16, a: &Lits, b: &Lits) -> Option<XGate> {
    XGate::conj(target, a.iter().copied().chain(b.iter().copied()))
}

/// Conjugate `phi` by h.  This is an involution because h is an involution.
fn conjugate(phi: &Esop, h: &XGate, target: u16) -> Esop {
    debug_assert!(!h.reads(target));
    let b = h.target;
    let mut out = Esop {
        parity: phi.parity,
        cubes: Vec::new(),
    };
    for cube in &phi.cubes {
        xor_add(&mut out, cube.clone());
        if !cube.reads(b) {
            continue;
        }
        let stripped: Lits = cube
            .ctrls
            .iter()
            .copied()
            .filter(|&(w, _)| w != b)
            .collect();
        // fire(h) = h.comp XOR product(h.ctrls).  Substituting b XOR fire(h)
        // into either polarity of b changes the literal by exactly fire(h).
        if h.comp {
            xor_add(
                &mut out,
                XGate {
                    target,
                    comp: false,
                    ctrls: stripped.clone(),
                },
            );
        }
        if let Some(g) = product(target, &stripped, &h.ctrls) {
            xor_add(&mut out, g);
        }
    }
    reduce(out)
}

// Costs are measured on the EMITTED form, where parity rides free in the
// first cube's comp bit (the reduce_group convention) and only an otherwise
// empty ESOP pays a gate for it. Anything else lets a candidate look
// profitable on ESOP arithmetic while being neutral or worse in the circuit:
// from_block(raw slice) never exceeds the raw slice on (gates, lits), so
// strict decrease under these costs is a strict circuit-level decrease.
fn gate_cost(esop: &Esop) -> usize {
    esop.cubes.len() + usize::from(esop.parity && esop.cubes.is_empty())
}

fn lit_cost(esop: &Esop) -> usize {
    esop.cubes.iter().map(XGate::width).sum()
}

fn gates_of(esop: Esop, target: u16) -> Vec<XGate> {
    let mut out = esop.cubes;
    if esop.parity {
        match out.first_mut() {
            Some(first) => first.comp = !first.comp,
            None => out.push(XGate::x_gate(target)),
        }
    }
    out
}

fn consider(
    gates: &[XGate],
    lo: usize,
    hi: usize,
    neighbor: usize,
    side: &'static str,
    out: &mut Vec<Candidate>,
) {
    let target = gates[lo].target;
    let h = &gates[neighbor];
    if h.target == target || h.reads(target) {
        return;
    }
    let before = from_block(&gates[lo..hi], target);
    if !before.cubes.iter().any(|g| g.reads(h.target)) {
        return;
    }
    let after = conjugate(&before, h, target);
    let bg = gate_cost(&before);
    let ag = gate_cost(&after);
    let bl = lit_cost(&before);
    let al = lit_cost(&after);
    if ag < bg || (ag == bg && al < bl) {
        let mut replacement = gates_of(after, target);
        let (span_lo, span_hi) = if side == "left" {
            replacement.push(h.clone());
            (neighbor, hi)
        } else {
            replacement.insert(0, h.clone());
            (lo, neighbor + 1)
        };
        out.push(Candidate {
            lo,
            hi,
            neighbor,
            side,
            block_target: target,
            neighbor_target: h.target,
            before_gates: bg,
            after_gates: ag,
            before_lits: bl,
            after_lits: al,
            span_lo,
            span_hi,
            replacement,
        });
    }
}

/// Enumerate profitable conjugations of every maximal same-target run across
/// its immediate left and right neighbor.  Returns (runs, multi_runs,
/// candidates sorted best-first).
pub fn scan(gates: &[XGate]) -> (usize, usize, Vec<Candidate>) {
    let mut candidates = Vec::new();
    let mut runs = 0usize;
    let mut multi_runs = 0usize;
    let mut i = 0usize;
    while i < gates.len() {
        let mut j = i + 1;
        while j < gates.len() && gates[j].target == gates[i].target {
            j += 1;
        }
        runs += 1;
        multi_runs += usize::from(j - i > 1);
        if i > 0 {
            consider(gates, i, j, i - 1, "left", &mut candidates);
        }
        if j < gates.len() {
            consider(gates, i, j, j, "right", &mut candidates);
        }
        i = j;
    }
    candidates.sort_unstable_by_key(|c| {
        (
            std::cmp::Reverse(c.before_gates - c.after_gates),
            std::cmp::Reverse(c.before_lits.saturating_sub(c.after_lits)),
            c.lo,
        )
    });
    (runs, multi_runs, candidates)
}

// The span rewrite is exact algebra, but assert it anyway on 8x64 random
// lanes over the span's wires.  Feeds nothing but the assertion.
fn verify_span(before: &[XGate], after: &[XGate], rng: &mut StdRng) {
    let top = before
        .iter()
        .chain(after)
        .map(|g| {
            g.ctrls
                .iter()
                .map(|&(w, _)| w)
                .chain([g.target])
                .max()
                .unwrap_or(0)
        })
        .max()
        .unwrap_or(0);
    let w = top as usize + 1;
    for round in 0..8 {
        let sa: Vec<u64> = (0..w).map(|_| rng.random()).collect();
        let mut sb = sa.clone();
        let mut sa = sa;
        eval_lanes(before.iter(), &mut sa);
        eval_lanes(after.iter(), &mut sb);
        assert_eq!(
            sa, sb,
            "downhill span rewrite changed the function (round {round})"
        );
    }
}

/// Apply a maximal non-overlapping best-first subset of `candidates` in one
/// linear rebuild.  Ancestry (when threaded) follows the rewrite: every
/// conjugated cube carries the union of the block members' sets (each output
/// cube derives from the whole block ESOP), the crossed neighbor keeps its
/// own set.  Returns the rewritten circuit and the number of swaps applied.
pub fn apply_candidates(
    gates: Vec<XGate>,
    anc: Option<Vec<AncBits>>,
    candidates: &[Candidate],
    rng: &mut StdRng,
    local_verify: bool,
) -> (Vec<XGate>, Option<Vec<AncBits>>, usize) {
    if candidates.is_empty() {
        return (gates, anc, 0);
    }
    let mut ranked: Vec<&Candidate> = candidates.iter().collect();
    ranked.sort_unstable_by_key(|c| {
        (
            std::cmp::Reverse(c.before_gates - c.after_gates),
            std::cmp::Reverse(c.before_lits.saturating_sub(c.after_lits)),
            c.span_lo,
        )
    });
    let mut occupied = vec![false; gates.len()];
    let mut chosen: Vec<&Candidate> = Vec::new();
    for c in ranked {
        if occupied[c.span_lo..c.span_hi].iter().any(|&x| x) {
            continue;
        }
        occupied[c.span_lo..c.span_hi].fill(true);
        chosen.push(c);
    }
    chosen.sort_unstable_by_key(|c| c.span_lo);
    let swaps = chosen.len();
    let mut next: Vec<XGate> = Vec::with_capacity(gates.len());
    let mut next_anc: Option<Vec<AncBits>> = anc.as_ref().map(|_| Vec::with_capacity(gates.len()));
    let mut cursor = 0usize;
    for c in chosen {
        if local_verify {
            verify_span(&gates[c.span_lo..c.span_hi], &c.replacement, rng);
        }
        next.extend_from_slice(&gates[cursor..c.span_lo]);
        if let Some(na) = next_anc.as_mut() {
            let a = anc.as_ref().expect("next_anc implies anc");
            na.extend(a[cursor..c.span_lo].iter().cloned());
            let mut block_union: AncBits = AncBits::new();
            for tags in &a[c.lo..c.hi] {
                or_anc(&mut block_union, tags);
            }
            let h_tag = a[c.neighbor].clone();
            let h_index = if c.side == "left" {
                c.replacement.len() - 1
            } else {
                0
            };
            for k in 0..c.replacement.len() {
                na.push(if k == h_index {
                    h_tag.clone()
                } else {
                    block_union.clone()
                });
            }
        }
        next.extend(c.replacement.iter().cloned());
        cursor = c.span_hi;
    }
    next.extend_from_slice(&gates[cursor..]);
    if let Some(na) = next_anc.as_mut() {
        let a = anc.as_ref().expect("next_anc implies anc");
        na.extend(a[cursor..].iter().cloned());
    }
    (next, next_anc, swaps)
}

/// One scan + apply.  Strictly (gates, lits)-lexicographically decreasing
/// whenever it applies anything, so iterating it terminates.
pub fn apply_pass(
    gates: Vec<XGate>,
    anc: Option<Vec<AncBits>>,
    rng: &mut StdRng,
    local_verify: bool,
) -> (Vec<XGate>, Option<Vec<AncBits>>, usize) {
    let (_, _, candidates) = scan(&gates);
    apply_candidates(gates, anc, &candidates, rng, local_verify)
}
