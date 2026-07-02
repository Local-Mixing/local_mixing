// Fragment mixing: adaptive Shannon decomposition of g57 gates into variable-arity cube
// gates, so bulky gates can be split into smaller pieces that slide past each other.
//
// A cube gate is
//
//     T(t, C):  x_t <- x_t XOR 1_C(x)
//
// where the cube C is a partial assignment (a conjunction of literals) that never mentions
// the target t. This is the natural fragment object: a k-literal cube gate touches 2^(n-k-1)
// transpositions, so higher-arity fragments affect fewer transpositions and commute more
// freely. The whole g57 gate set embeds as the k=2 case with one positive and one negative
// literal.
//
// The algebra implemented here is exactly the one worked out in the design discussion:
//
//   * split (Shannon):  T(t,C) = T(t, C & p=0) ; T(t, C & p=1)          [Phi-neutral]
//   * two-clause commute law: two distinct-target gates commute iff their cubes are
//     inconsistent (never fire together) OR neither target lies in the other's cube.
//   * cross-with-correction: reordering two gates that DO interfere leaves a single
//     correction cube; commutation is the special case where that correction is empty.
//     Mutual dependency (each reads the other's target) falls outside the single-cube
//     family and is refused here (it would need multi-gate expansion, e.g. a wire swap).
//
// Every identity above is brute-force verified against the real g57 evaluator
// (`Gate::evaluate_index_list`) in the test module, on all 2^n inputs for small n.

use crate::circuit::circuit::{Gate, Monomial, Polynomial, polynomial_from_terms};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// A cube gate `T(target, cube)`: toggle `target` on inputs where every literal of `cube`
/// holds. `lits` is kept sorted by wire, has no repeated wire, and never contains `target`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CubeGate {
    pub target: u16,
    /// `(wire, required_value)` literals, sorted by wire. Target never appears here.
    pub lits: Vec<(u16, bool)>,
}

impl CubeGate {
    /// Build a cube gate, normalizing the literal order. Panics on a malformed cube
    /// (target used as a control, or a wire constrained twice) since those never arise from
    /// valid g57 gates or from the moves below.
    pub fn new(target: u16, mut lits: Vec<(u16, bool)>) -> Self {
        lits.sort_by_key(|&(w, _)| w);
        for pair in lits.windows(2) {
            assert_ne!(pair[0].0, pair[1].0, "cube constrains wire {} twice", pair[0].0);
        }
        assert!(
            !lits.iter().any(|&(w, _)| w == target),
            "cube gate cannot control on its own target {target}"
        );
        CubeGate { target, lits }
    }

    /// Number of literals (the "k" of a k-cc gate).
    #[inline]
    pub fn arity(&self) -> usize {
        self.lits.len()
    }

    /// Does this gate fire (toggle its target) on `state`? One bit per wire.
    #[inline]
    pub fn fires(&self, state: u64) -> bool {
        self.lits
            .iter()
            .all(|&(w, v)| (((state >> w) & 1) == 1) == v)
    }

    /// Apply the gate to a state.
    #[inline]
    pub fn apply(&self, state: u64) -> u64 {
        if self.fires(state) {
            state ^ (1u64 << self.target)
        } else {
            state
        }
    }

    /// Transposition mass exponent: a k-literal cube gate on n wires is `2^(n-k-1)`
    /// transpositions. Always >= 0 because a cube omits the target, so k <= n-1.
    #[inline]
    pub fn mass_exp(&self, n: usize) -> u32 {
        (n - self.arity() - 1) as u32
    }
}

/// Intersect two cubes (sorted literal lists). Returns the merged cube, or `None` if they
/// disagree on a shared wire (inconsistent — the two gates never fire together).
fn cube_and(a: &[(u16, bool)], b: &[(u16, bool)]) -> Option<Vec<(u16, bool)>> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i].0 == b[j].0 {
            if a[i].1 != b[j].1 {
                return None; // shared wire, opposite polarity -> inconsistent
            }
            out.push(a[i]);
            i += 1;
            j += 1;
        } else if a[i].0 < b[j].0 {
            out.push(a[i]);
            i += 1;
        } else {
            out.push(b[j]);
            j += 1;
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    Some(out)
}

/// Drop the literal on `wire` from a cube (if present).
fn without(lits: &[(u16, bool)], wire: u16) -> Vec<(u16, bool)> {
    lits.iter().copied().filter(|&(w, _)| w != wire).collect()
}

/// The two-clause commutation law. Two cube gates commute iff:
///   - they share a target (both just XOR into it), OR
///   - their cubes are inconsistent (never fire together), OR
///   - neither gate's target lies in the other's cube (structural independence).
///
/// The middle clause is the semantic mobility that plain support-based collision checks
/// (`Gate::collides_index`) miss, and it is where fragmentation buys freedom.
pub fn commute(g: &CubeGate, h: &CubeGate) -> bool {
    if g.target == h.target {
        return true;
    }
    let h_target_in_g = g.lits.iter().any(|&(w, _)| w == h.target);
    let g_target_in_h = h.lits.iter().any(|&(w, _)| w == g.target);
    if !h_target_in_g && !g_target_in_h {
        return true; // structural independence
    }
    cube_and(&g.lits, &h.lits).is_none() // else: commute iff cubes never both fire
}

/// Shannon split of a gate on `pivot`: `T(t,C) = T(t, C & pivot=0) ; T(t, C & pivot=1)`.
/// `pivot` must differ from the target and not already appear in the cube. Phi-neutral.
pub fn split(g: &CubeGate, pivot: u16) -> (CubeGate, CubeGate) {
    assert_ne!(pivot, g.target, "cannot pivot on the target");
    assert!(
        !g.lits.iter().any(|&(w, _)| w == pivot),
        "cube already constrains pivot {pivot}"
    );
    let mut lo = g.lits.clone();
    lo.push((pivot, false));
    let mut hi = g.lits.clone();
    hi.push((pivot, true));
    (CubeGate::new(g.target, lo), CubeGate::new(g.target, hi))
}

/// Reorder the adjacent ordered pair `[a, b]` (a applied first) into a sequence that begins
/// with `b` moved to the front and computes the same permutation. Returns:
///   - `[b, a]` when they commute (empty correction),
///   - `[b, a, correction]` when one gate reads the other's target — a single correction cube
///     on the reader's target, per `K;M = M;K; T(b_K, (C_K\{t_M}) & C_M)`,
///   - `None` on mutual dependency (each reads the other's target): that conjugate leaves the
///     single-cube family (e.g. a wire swap) and must be handled by further expansion.
pub fn cross(a: &CubeGate, b: &CubeGate) -> Option<Vec<CubeGate>> {
    if a.target == b.target {
        return Some(vec![b.clone(), a.clone()]);
    }
    let b_target_in_a = a.lits.iter().any(|&(w, _)| w == b.target);
    let a_target_in_b = b.lits.iter().any(|&(w, _)| w == a.target);

    if !b_target_in_a && !a_target_in_b {
        return Some(vec![b.clone(), a.clone()]); // structural independence
    }
    if b_target_in_a && a_target_in_b {
        return None; // mutual dependency
    }

    // Exactly one side reads the other's target. The correction toggles the READER's target;
    // its cube is the reader's cube minus the shared-target literal, intersected with the
    // writer's cube. An inconsistent intersection means the pair actually commutes.
    let (reader_target, reader_lits, shared, writer_lits) = if b_target_in_a {
        // `a` reads b's target; a is the reader.
        (a.target, &a.lits, b.target, &b.lits)
    } else {
        // `b` reads a's target; b is the reader.
        (b.target, &b.lits, a.target, &a.lits)
    };
    let reader_wo = without(reader_lits, shared);
    match cube_and(&reader_wo, writer_lits) {
        None => Some(vec![b.clone(), a.clone()]),
        Some(cube) => Some(vec![
            b.clone(),
            a.clone(),
            CubeGate::new(reader_target, cube),
        ]),
    }
}

/// Split a g57 gate `[active, pos_ctrl, neg_ctrl]` (fires when pos=1 OR neg=0) into its two
/// disjoint cube fragments: the 1cc `{pos=1}` and the 2cc `{pos=0, neg=0}`. Their XOR is the
/// g57 firing function, and they commute (disjoint on `pos`).
pub fn from_g57(gate: [u16; 3]) -> [CubeGate; 2] {
    let [active, pos, neg] = gate;
    let one_cc = CubeGate::new(active, vec![(pos, true)]);
    let two_cc = CubeGate::new(active, vec![(pos, false), (neg, false)]);
    [one_cc, two_cc]
}

/// Expand a whole g57 circuit into its cube-gate fragments, in order.
pub fn expand_g57_circuit(gates: &[[u16; 3]]) -> Vec<CubeGate> {
    let mut out = Vec::with_capacity(gates.len() * 2);
    for &g in gates {
        let [a, b] = from_g57(g);
        out.push(a);
        out.push(b);
    }
    out
}

/// Total transposition mass Phi = sum over gates of `2^(n-k-1)`. Conserved by split/merge,
/// raised only by cross corrections, lowered only by recombination.
pub fn phi(frags: &[CubeGate], n: usize) -> f64 {
    frags
        .iter()
        .map(|g| 2f64.powi(g.mass_exp(n) as i32))
        .sum()
}

/// Apply a fragment list to a state.
pub fn apply_frags(frags: &[CubeGate], mut state: u64) -> u64 {
    for f in frags {
        state = f.apply(state);
    }
    state
}

/// Brute-force check that a fragment list computes the same permutation as a g57 circuit,
/// over all 2^n inputs. Only for small n (verification).
pub fn frags_match_g57(frags: &[CubeGate], gates: &[[u16; 3]], n: usize) -> bool {
    (0..(1u64 << n)).all(|s| apply_frags(frags, s) as usize == Gate::evaluate_index_list(s as usize, gates))
}

/// Brute-force check that two fragment lists compute the same permutation, over all 2^n inputs.
pub fn frags_equivalent(a: &[CubeGate], b: &[CubeGate], n: usize) -> bool {
    (0..(1u64 << n)).all(|s| apply_frags(a, s) == apply_frags(b, s))
}

// --- Fragment -> polynomial, for DB keying ------------------------------------------------
// The DB is keyed on the per-wire output polynomials (up to canonicalization). A window of
// fragments computes a function just like a g57 window does, so we build its polynomials the
// same way `CircuitSeq::to_polynomial` does — mirroring its exact representation (Monomial =
// u64 variable-bitmask, Polynomial = sorted XOR-reduced monomials, wire i starts as x_i, and
// the identity part is left in). The only difference is per-gate: a cube gate XORs the product
// of its literal polynomials into the target, using the true (evaluate_index) semantics.

/// Product of two polynomials over GF(2): each monomial pair multiplies to the union of their
/// variable sets (`|`), then XOR-reduce duplicates.
fn poly_mul(a: &Polynomial, b: &Polynomial) -> Polynomial {
    let mut terms: Vec<Monomial> = Vec::with_capacity(a.len() * b.len());
    for &m1 in a {
        for &m2 in b {
            terms.push(m1 | m2);
        }
    }
    polynomial_from_terms(terms)
}

/// NOT(p) = p + 1 (toggle the constant term).
fn poly_not(a: &Polynomial) -> Polynomial {
    let mut terms: Vec<Monomial> = a.clone();
    terms.push(0u64); // the constant-1 monomial (empty product)
    polynomial_from_terms(terms)
}

/// Per-wire output polynomials of a fragment list on `n` wires, matching the representation of
/// `CircuitSeq::to_polynomial` so the same canonicalization + hash yields the same DB key.
pub fn fragment_polys(frags: &[CubeGate], n: usize) -> Vec<Polynomial> {
    let mut polys: Vec<Polynomial> = (0..n).map(|i| vec![1u64 << i]).collect();
    for f in frags {
        // product of the literal polynomials (positive -> wire poly, negative -> its NOT)
        let mut product: Polynomial = vec![0u64]; // constant 1
        for &(w, v) in &f.lits {
            let lit = if v {
                polys[w as usize].clone()
            } else {
                poly_not(&polys[w as usize])
            };
            product = poly_mul(&product, &lit);
        }
        // polys[target] ^= product
        let mut merged = polys[f.target as usize].clone();
        merged.extend_from_slice(&product);
        polys[f.target as usize] = polynomial_from_terms(merged);
    }
    polys
}

/// The result of trying to fuse two same-target gates.
enum Combine {
    /// Identical gates annihilate (two toggles of the same cube = identity).
    Cancel,
    /// Sibling cubes differing in one pivot merge into a lower-arity gate (inverse Shannon).
    Merged(CubeGate),
}

/// Try to fuse two gates exactly. Requires the same target. Cancels identical gates; merges
/// siblings that share the same wire set and differ in exactly one literal's polarity.
fn try_combine(g: &CubeGate, h: &CubeGate) -> Option<Combine> {
    if g.target != h.target {
        return None;
    }
    if g.lits == h.lits {
        return Some(Combine::Cancel);
    }
    if g.lits.len() != h.lits.len() {
        return None;
    }
    // Both sorted; siblings share the wire set and disagree on exactly one polarity.
    let mut differ_wire = None;
    for (x, y) in g.lits.iter().zip(h.lits.iter()) {
        if x.0 != y.0 {
            return None; // different wire sets
        }
        if x.1 != y.1 {
            if differ_wire.is_some() {
                return None; // more than one disagreement
            }
            differ_wire = Some(x.0);
        }
    }
    let w = differ_wire?;
    Some(Combine::Merged(CubeGate::new(g.target, without(&g.lits, w))))
}

/// Local exact recombination: the "surface tension" that pulls fragment mass back down.
/// Repeatedly finds a pair `(i, j)` where `frags[i]` can slide right to meet `frags[j]`
/// (every gate in between commutes with it) and the two exactly cancel or sibling-merge.
/// Equivalence-preserving by construction: sliding is a chain of exact commutes, and the
/// fuse is exact. O(m^3) worst case — fine for a verification-scale driver.
pub fn recombine(mut frags: Vec<CubeGate>) -> Vec<CubeGate> {
    loop {
        let mut fused = false;
        'search: for i in 0..frags.len() {
            // Slide frags[i] rightward through gates it commutes with, trying to fuse with each
            // gate it reaches. Stop at the first gate it can neither fuse with nor pass — no
            // later gate is reachable either. The fused gate is placed at the far position so
            // the passed (commuting) gates stay correctly ordered before it.
            let mut j = i + 1;
            while j < frags.len() {
                match try_combine(&frags[i], &frags[j]) {
                    Some(Combine::Cancel) => {
                        frags.remove(j);
                        frags.remove(i);
                        fused = true;
                        break 'search;
                    }
                    Some(Combine::Merged(m)) => {
                        frags[j] = m;
                        frags.remove(i);
                        fused = true;
                        break 'search;
                    }
                    None => {
                        if commute(&frags[i], &frags[j]) {
                            j += 1;
                        } else {
                            break;
                        }
                    }
                }
            }
        }
        if !fused {
            break;
        }
    }
    frags
}

/// Count how many fragments could be reassembled into g57 gates by pairing a 1cc `{p=1}` with
/// a 2cc `{p=0, q=0}` of the same target (order-free: same-target gates always commute). This
/// is a reporting metric for how close local recombination gets to pure g57 form; it does not
/// attempt the general (DB-backed) re-compression.
pub fn g57_reassemblable_pairs(frags: &[CubeGate]) -> usize {
    let ones: Vec<&CubeGate> = frags.iter().filter(|g| g.arity() == 1).collect();
    let twos: Vec<&CubeGate> = frags
        .iter()
        .filter(|g| g.arity() == 2 && g.lits.iter().all(|&(_, v)| !v))
        .collect();
    let mut pairs = 0;
    let mut used_one = vec![false; ones.len()];
    for two in &twos {
        // 2cc {p=0, q=0}: matches a 1cc {p=1} on the same target with p one of its wires.
        let candidate_wires = [two.lits[0].0, two.lits[1].0];
        if let Some(idx) = ones.iter().enumerate().position(|(k, one)| {
            !used_one[k]
                && one.target == two.target
                && one.lits[0].1
                && candidate_wires.contains(&one.lits[0].0)
        }) {
            used_one[idx] = true;
            pairs += 1;
        }
    }
    pairs
}

/// A summary of a transport-mix run.
#[derive(Debug, Clone)]
pub struct TransportReport {
    pub n: usize,
    pub input_g57_gates: usize,
    pub initial_frags: usize,
    pub phi_initial: f64,
    pub phi_peak: f64,
    pub phi_final: f64,
    pub final_frags: usize,
    pub g57_pairs_final: usize,
    /// True iff the final fragment list still computes the input permutation (brute-forced).
    pub equivalent: bool,
}

/// A minimal, equivalence-preserving transport-mix driver, to exercise the machinery and
/// watch Phi. It expands a g57 circuit to fragments and repeatedly applies:
///   - COMMUTE: swap a random adjacent pair when the law permits (free transport, Phi-neutral),
///   - SPLIT: split a random gate on a pivot drawn from a neighbor's cube, which is what
///     manufactures the contradiction that lets a fragment slide (Phi-neutral),
///   - RECOMBINE: periodically pull mass back down (the Phi sink).
/// Cross-with-correction is implemented and tested but kept out of this default loop so the
/// demo cannot proliferate unboundedly; the point here is that the walk stays exactly
/// equivalent while fragments move and Phi stays bounded.
pub fn transport_mix(gates: &[[u16; 3]], n: usize, steps: usize, seed: u64) -> TransportReport {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut frags = expand_g57_circuit(gates);
    let phi_initial = phi(&frags, n);
    let initial_frags = frags.len();
    let mut phi_peak = phi_initial;

    for step in 0..steps {
        if frags.len() < 2 {
            break;
        }
        let roll = rng.random_range(0u8..100);
        if roll < 55 {
            // COMMUTE: try a random adjacent swap.
            let i = rng.random_range(0..frags.len() - 1);
            if commute(&frags[i], &frags[i + 1]) {
                frags.swap(i, i + 1);
            }
        } else if roll < 85 {
            // SPLIT: pick a gate and a pivot from a neighbor's cube (or any free wire).
            let i = rng.random_range(0..frags.len());
            if frags[i].arity() < n.saturating_sub(1) {
                let pivot = pick_pivot(&frags, i, n, &mut rng);
                if let Some(p) = pivot {
                    let (lo, hi) = split(&frags[i], p);
                    frags[i] = lo;
                    frags.insert(i + 1, hi);
                }
            }
        } else {
            // RECOMBINE (the sink), run occasionally to bound Phi.
            frags = recombine(std::mem::take(&mut frags));
        }
        // Periodic recombine independent of the roll, so Phi cannot run away unwatched.
        if step % 256 == 255 {
            frags = recombine(std::mem::take(&mut frags));
        }
        phi_peak = phi_peak.max(phi(&frags, n));
    }

    frags = recombine(frags);
    let phi_final = phi(&frags, n);
    let equivalent = frags_match_g57(&frags, gates, n);
    TransportReport {
        n,
        input_g57_gates: gates.len(),
        initial_frags,
        phi_initial,
        phi_peak,
        phi_final,
        final_frags: frags.len(),
        g57_pairs_final: g57_reassemblable_pairs(&frags),
        equivalent,
    }
}

/// Choose a split pivot for `frags[i]`: prefer a wire that a neighbor's cube constrains (so
/// one child gains a contradiction and becomes mobile), else any free wire.
fn pick_pivot(frags: &[CubeGate], i: usize, n: usize, rng: &mut StdRng) -> Option<u16> {
    let g = &frags[i];
    let is_free = |w: u16| w != g.target && !g.lits.iter().any(|&(x, _)| x == w);
    // Candidate pivots from an adjacent neighbor's cube.
    let mut candidates: Vec<u16> = Vec::new();
    for off in [i.wrapping_sub(1), i + 1] {
        if let Some(neigh) = frags.get(off) {
            for &(w, _) in &neigh.lits {
                if is_free(w) && !candidates.contains(&w) {
                    candidates.push(w);
                }
            }
        }
    }
    if candidates.is_empty() {
        let free: Vec<u16> = (0..n as u16).filter(|&w| is_free(w)).collect();
        if free.is_empty() {
            return None;
        }
        return Some(free[rng.random_range(0..free.len())]);
    }
    Some(candidates[rng.random_range(0..candidates.len())])
}

// ---------------------------------------------------------------------------------------
// End-to-end: decompose a g57 circuit into fragments, shoot them around, reassemble to g57.
// ---------------------------------------------------------------------------------------

use crate::circuit::circuit::CircuitSeq;

/// A slot in the reassembly buffer: either a recovered g57 gate or a still-loose fragment.
enum Slot {
    G57([u16; 3]),
    Frag(CubeGate),
}

/// If `f` and `g` are exactly the two fragments of a g57 gate (a 1cc `{p=1}` and a 2cc
/// `{p=0, q=0}` on the same target), return that g57 gate `[target, p, q]`. This only matches
/// when the two cubes ARE a g57 decomposition, so replacing them by the gate is always exact.
fn g57_pair(f: &CubeGate, g: &CubeGate) -> Option<[u16; 3]> {
    if f.target != g.target {
        return None;
    }
    let (one, two) = match (f.arity(), g.arity()) {
        (1, 2) => (f, g),
        (2, 1) => (g, f),
        _ => return None,
    };
    if !one.lits[0].1 {
        return None; // the 1cc must be a positive literal {p=1}
    }
    let p = one.lits[0].0;
    if two.lits[0].1 || two.lits[1].1 {
        return None; // the 2cc must be all-negative {..=0, ..=0}
    }
    let (w0, w1) = (two.lits[0].0, two.lits[1].0);
    let q = if w0 == p {
        w1
    } else if w1 == p {
        w0
    } else {
        return None; // the 2cc must contain the 1cc's wire p
    };
    Some([one.target, p, q])
}

/// Can `frag` commute past a slot? Past a loose fragment it's the two-clause law; past a
/// reassembled g57 gate it must commute with both of that gate's fragments.
fn slot_commute(frag: &CubeGate, slot: &Slot) -> bool {
    match slot {
        Slot::Frag(h) => commute(frag, h),
        Slot::G57(g) => {
            let [a, b] = from_g57(*g);
            commute(frag, &a) && commute(frag, &b)
        }
    }
}

/// Reassemble a fragment list into g57 gates: repeatedly find two loose fragments that form a
/// g57 pair and can be brought adjacent by valid commutes, and fuse them in place. Returns the
/// recovered g57 circuit when every fragment is consumed, or `Err(residual_fragments)` when
/// some fragments cannot be paired (they encode a permutation with no pure-g57 window here —
/// the DB re-compression case). Equivalence-preserving by construction.
pub fn reassemble_g57(frags: Vec<CubeGate>, _n: usize) -> Result<Vec<[u16; 3]>, usize> {
    let mut slots: Vec<Slot> = frags.into_iter().map(Slot::Frag).collect();
    loop {
        let mut found: Option<(usize, usize, [u16; 3], bool)> = None;
        'outer: for i in 0..slots.len() {
            let Slot::Frag(fi) = &slots[i] else { continue };
            for j in (i + 1)..slots.len() {
                let Slot::Frag(fj) = &slots[j] else { continue };
                let Some(gate) = g57_pair(fi, fj) else {
                    continue;
                };
                // Prefer sliding fi right to meet fj; else fj left to meet fi.
                if (i + 1..j).all(|k| slot_commute(fi, &slots[k])) {
                    found = Some((i, j, gate, true));
                    break 'outer;
                }
                if (i + 1..j).all(|k| slot_commute(fj, &slots[k])) {
                    found = Some((i, j, gate, false));
                    break 'outer;
                }
            }
        }
        match found {
            Some((i, j, gate, slide_i)) => {
                if slide_i {
                    slots[j] = Slot::G57(gate);
                    slots.remove(i);
                } else {
                    slots[i] = Slot::G57(gate);
                    slots.remove(j);
                }
            }
            None => break,
        }
    }
    let mut circuit = Vec::new();
    let mut residual = 0usize;
    for s in slots {
        match s {
            Slot::G57(g) => circuit.push(g),
            Slot::Frag(_) => residual += 1,
        }
    }
    if residual == 0 {
        Ok(circuit)
    } else {
        Err(residual)
    }
}

/// Exact equality of two g57 circuits over all 2^n inputs (small n only).
fn g57_circuits_equal(a: &[[u16; 3]], b: &[[u16; 3]], n: usize) -> bool {
    (0..(1u64 << n))
        .all(|s| Gate::evaluate_index_list(s as usize, a) == Gate::evaluate_index_list(s as usize, b))
}

/// Report from `shoot_and_reassemble`.
#[derive(Debug, Clone)]
pub struct ShotReport {
    pub n: usize,
    pub input_gates: usize,
    pub steps: usize,
    pub phi_initial: f64,
    pub phi_peak: f64,
    pub phi_final: f64,
    pub crossings: usize,
    /// True iff the shot fragments reassembled fully into g57 (no residual, so `circuit` is
    /// the genuinely mixed output). False means we fell back to a valid circuit unchanged.
    pub fully_reconstructed: bool,
    pub residual_fragments: usize,
    pub output_gates: usize,
    /// Brute-forced equivalence of the returned circuit to the input (only for small n).
    pub equivalent: bool,
}

/// Decompose a g57 circuit into cube-gate fragments, shoot the fragments around (transport by
/// commutation, split to enable mobility, and — when `allow_cross` — shoot fragments through
/// collisions leaving corrections), then recombine and reassemble into g57.
///
/// Always returns a circuit functionally equal to the input: when the shot fragments
/// reassemble fully, that mixed circuit is returned (`fully_reconstructed = true`); otherwise
/// the input is handed back unchanged and the report says so. `max_arity` caps fragment size
/// to bound blow-up.
pub fn shoot_and_reassemble(
    circuit: &CircuitSeq,
    n: usize,
    steps: usize,
    seed: u64,
    allow_cross: bool,
    max_arity: usize,
) -> (CircuitSeq, ShotReport) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut frags = expand_g57_circuit(&circuit.gates);
    let phi_initial = phi(&frags, n);
    let mut phi_peak = phi_initial;
    let mut crossings = 0usize;
    let cap = max_arity.min(n.saturating_sub(1)).max(2);

    for step in 0..steps {
        if frags.len() < 2 {
            break;
        }
        let i = rng.random_range(0..frags.len() - 1);
        if !allow_cross {
            // DEFAULT: pure commutation transport. The fragment multiset is preserved, so the
            // shot circuit always reassembles into g57 — but fragments reach positions no
            // whole-g57 reordering could, yielding a genuinely different gate sequence.
            if commute(&frags[i], &frags[i + 1]) {
                frags.swap(i, i + 1);
            }
        } else {
            // ADVANCED: also split to unlock mobility and shoot fragments through collisions,
            // leaving corrections. This can compress to a non-g57 form (recombine is a Phi
            // sink), in which case reassembly leaves residuals and we fall back.
            let roll = rng.random_range(0u8..100);
            if roll < 45 {
                if commute(&frags[i], &frags[i + 1]) {
                    frags.swap(i, i + 1);
                }
            } else if roll < 70 {
                if frags[i].arity() < cap {
                    if let Some(p) = pick_pivot(&frags, i, n, &mut rng) {
                        let (lo, hi) = split(&frags[i], p);
                        frags[i] = lo;
                        frags.insert(i + 1, hi);
                    }
                }
            } else if roll < 85 {
                if !commute(&frags[i], &frags[i + 1]) {
                    if let Some(repl) = cross(&frags[i], &frags[i + 1]) {
                        if repl.iter().all(|g| g.arity() <= cap) {
                            frags.splice(i..=i + 1, repl);
                            crossings += 1;
                        }
                    }
                }
            } else {
                frags = recombine(std::mem::take(&mut frags));
            }
            if step % 200 == 199 {
                frags = recombine(std::mem::take(&mut frags));
            }
        }
        phi_peak = phi_peak.max(phi(&frags, n));
    }
    if allow_cross {
        frags = recombine(frags);
    }
    let phi_final = phi(&frags, n);

    let (out_gates, fully, residual) = match reassemble_g57(frags, n) {
        Ok(gates) => (gates, true, 0),
        Err(res) => (circuit.gates.clone(), false, res),
    };
    let output = CircuitSeq { gates: out_gates };
    let equivalent = n <= 20 && g57_circuits_equal(&circuit.gates, &output.gates, n);

    let report = ShotReport {
        n,
        input_gates: circuit.gates.len(),
        steps,
        phi_initial,
        phi_peak,
        phi_final,
        crossings,
        fully_reconstructed: fully,
        residual_fragments: residual,
        output_gates: output.gates.len(),
        equivalent,
    };
    (output, report)
}

// ---------------------------------------------------------------------------------------
// Directed collision shooting — the strategy as originally stated: shoot a fragment until it
// collides, split it AT the collision so a piece slips past the blocker and continues its
// journey. (Random gate/direction choice over many shots makes fragments of colliding gates
// travel into each other's sides.)
// ---------------------------------------------------------------------------------------

/// A wire that blocker `b` constrains but mover `f` does not (and that is not `f`'s target):
/// splitting `f` on it produces a child whose literal contradicts `b`, so that child commutes
/// past `b`. Returns `(wire, value_the_passing_child_takes)` = the opposite of `b`'s literal.
fn passing_pivot(f: &CubeGate, b: &CubeGate) -> Option<(u16, bool)> {
    for &(w, bv) in &b.lits {
        if w != f.target && !f.lits.iter().any(|&(x, _)| x == w) {
            return Some((w, !bv));
        }
    }
    None
}

/// Shoot the fragment at `start` in direction `dir` (+1 = right/later, -1 = left/earlier):
/// slide by commutation until it collides, then split at the collision so the contradicting
/// child passes the blocker and keeps going. Recurses until the travelling child reaches an
/// end, hits the arity cap, or meets a blocker it cannot split past (there it crosses with a
/// correction if `allow_correction`, else stops). Every step is equivalence-preserving.
fn shoot_fragment(
    frags: &mut Vec<CubeGate>,
    start: usize,
    dir: isize,
    max_arity: usize,
    allow_correction: bool,
    crossings: &mut usize,
) {
    let mut i = start;
    loop {
        let jj = i as isize + dir;
        if jj < 0 || jj as usize >= frags.len() {
            return; // travelled to an end of the circuit
        }
        let j = jj as usize;
        if commute(&frags[i], &frags[j]) {
            frags.swap(i, j);
            i = j;
            continue;
        }
        // Collision with blocker frags[j]. Try to split so a child slips past it.
        if frags[i].arity() < max_arity {
            if let Some((w, val)) = passing_pivot(&frags[i], &frags[j]) {
                let (lo, hi) = split(&frags[i], w);
                let (passing, staying) = if val { (hi, lo) } else { (lo, hi) };
                if dir > 0 {
                    // [.. staying@i, B@i+1, passing@i+2 ..] — passing commuted past B.
                    frags[i] = staying;
                    frags.insert(i + 2, passing);
                    i += 2;
                } else {
                    // [.. passing@i-1, B@i, staying@i+1 ..]
                    frags[i] = staying;
                    frags.insert(i - 1, passing);
                    i -= 1;
                }
                continue;
            }
        }
        // Cannot split past (blocker fires on the mover's target alone, or arity capped).
        if allow_correction {
            let repl = if dir > 0 {
                cross(&frags[i], &frags[j]) // [B, F, corr]: F ends up right of B
            } else {
                cross(&frags[j], &frags[i]) // [F, B, corr]: F ends up left of B
            };
            if let Some(repl) = repl {
                if repl.iter().all(|g| g.arity() <= max_arity) {
                    let (lo, hi) = (i.min(j), i.max(j));
                    frags.splice(lo..=hi, repl);
                    *crossings += 1;
                }
            }
        }
        return; // blocked
    }
}

/// Decompose a g57 circuit into fragments and run the directed collision-shooting strategy:
/// repeatedly pick a fragment and a direction and shoot it until it collides, splitting at
/// collisions so pieces travel through. Then recombine and reassemble to g57.
pub fn collision_shoot(
    circuit: &CircuitSeq,
    n: usize,
    shots: usize,
    seed: u64,
    allow_correction: bool,
    max_arity: usize,
) -> (CircuitSeq, ShotReport) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut frags = expand_g57_circuit(&circuit.gates);
    let phi_initial = phi(&frags, n);
    let mut phi_peak = phi_initial;
    let cap = max_arity.min(n.saturating_sub(1)).max(2);
    let mut crossings = 0usize;

    for shot in 0..shots {
        if frags.len() < 2 {
            break;
        }
        let i = rng.random_range(0..frags.len());
        let dir: isize = if rng.random_bool(0.5) { 1 } else { -1 };
        shoot_fragment(&mut frags, i, dir, cap, allow_correction, &mut crossings);
        // Keep the tape small so recombine (~O(m^4)) stays cheap: fold often.
        if shot % 32 == 31 {
            phi_peak = phi_peak.max(phi(&frags, n));
            frags = recombine(std::mem::take(&mut frags));
        }
    }
    frags = recombine(frags);
    let phi_final = phi(&frags, n);

    let (out_gates, fully, residual) = match reassemble_g57(frags, n) {
        Ok(gates) => (gates, true, 0),
        Err(res) => (circuit.gates.clone(), false, res),
    };
    let output = CircuitSeq { gates: out_gates };
    let equivalent = n <= 20 && g57_circuits_equal(&circuit.gates, &output.gates, n);
    let report = ShotReport {
        n,
        input_gates: circuit.gates.len(),
        steps: shots,
        phi_initial,
        phi_peak,
        phi_final,
        crossings,
        fully_reconstructed: fully,
        residual_fragments: residual,
        output_gates: output.gates.len(),
        equivalent,
    };
    (output, report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::random_data::random_circuit;

    // A hand fixture matching the design-note example, on wires x1=1, x2=2, x3=3, x4=4.
    fn g(target: u16, lits: &[(u16, bool)]) -> CubeGate {
        CubeGate::new(target, lits.to_vec())
    }

    #[test]
    fn from_g57_reproduces_the_gate() {
        // Every g57 gate on <=5 wires equals its two cube fragments, all inputs.
        let n = 5;
        for a in 0..n as u16 {
            for p in 0..n as u16 {
                for q in 0..n as u16 {
                    if a == p || a == q || p == q {
                        continue;
                    }
                    let gate = [a, p, q];
                    let frags = from_g57(gate);
                    assert!(
                        frags_match_g57(&frags, &[gate], n),
                        "g57 {gate:?} != its fragments"
                    );
                }
            }
        }
    }

    #[test]
    fn expand_preserves_random_circuits() {
        let n = 8;
        for seed in 0..20 {
            let c = random_circuit(n, 40);
            let frags = expand_g57_circuit(&c.gates);
            assert!(
                frags_match_g57(&frags, &c.gates, n),
                "expansion changed the function (seed {seed})"
            );
        }
    }

    #[test]
    fn split_is_phi_neutral_and_exact() {
        let n = 6;
        let base = g(0, &[(2, false), (3, false)]); // 2cc
        for pivot in [1u16, 4, 5] {
            let (lo, hi) = split(&base, pivot);
            // Exactness: the pair equals the original over all inputs.
            for s in 0..(1u64 << n) {
                let orig = base.apply(s);
                let two = hi.apply(lo.apply(s));
                assert_eq!(orig, two, "split changed function at state {s}");
            }
            // Phi neutrality.
            assert!((phi(&[base.clone()], n) - phi(&[lo.clone(), hi.clone()], n)).abs() < 1e-9);
        }
    }

    #[test]
    fn commute_law_matches_brute_force() {
        // For random cube-gate pairs, `commute` must agree with actual order-independence.
        let n = 6;
        let mut rng = StdRng::seed_from_u64(7);
        let rand_gate = |rng: &mut StdRng| -> CubeGate {
            let target = rng.random_range(0..n as u16);
            let mut lits = Vec::new();
            for w in 0..n as u16 {
                if w != target && rng.random_bool(0.35) {
                    lits.push((w, rng.random_bool(0.5)));
                }
            }
            CubeGate::new(target, lits)
        };
        let mut checked = 0;
        for _ in 0..4000 {
            let a = rand_gate(&mut rng);
            let b = rand_gate(&mut rng);
            let actually_commutes =
                (0..(1u64 << n)).all(|s| b.apply(a.apply(s)) == a.apply(b.apply(s)));
            assert_eq!(
                commute(&a, &b),
                actually_commutes,
                "commute law disagreed with brute force on {a:?} vs {b:?}"
            );
            checked += 1;
        }
        assert!(checked > 0);
    }

    #[test]
    fn commute_catches_disjoint_firing_that_support_check_misses() {
        // The design example: x1 += (x2+1)(x3+1) and x2 += x3 structurally collide (the
        // second writes x2, which the first reads) but never fire together (x3 disagreement).
        let a = g(1, &[(2, false), (3, false)]); // x1 += (x2+1)(x3+1)
        let b = g(2, &[(3, true)]); // x2 += x3
        assert!(commute(&a, &b), "disjoint-firing pair should commute");
        // And it is a genuine structural collision (b's target x2 is in a's cube).
        assert!(a.lits.iter().any(|&(w, _)| w == b.target));
    }

    #[test]
    fn cross_results_are_exact_and_agree_with_commute() {
        let n = 6;
        let mut rng = StdRng::seed_from_u64(11);
        let rand_gate = |rng: &mut StdRng| -> CubeGate {
            let target = rng.random_range(0..n as u16);
            let mut lits = Vec::new();
            for w in 0..n as u16 {
                if w != target && rng.random_bool(0.4) {
                    lits.push((w, rng.random_bool(0.5)));
                }
            }
            CubeGate::new(target, lits)
        };
        let (mut crossed, mut refused) = (0, 0);
        for _ in 0..4000 {
            let a = rand_gate(&mut rng);
            let b = rand_gate(&mut rng);
            match cross(&a, &b) {
                None => {
                    // Refusal must be exactly the mutual-dependency case.
                    let mutual = a.lits.iter().any(|&(w, _)| w == b.target)
                        && b.lits.iter().any(|&(w, _)| w == a.target);
                    assert!(mutual, "cross refused a non-mutual pair {a:?} {b:?}");
                    refused += 1;
                }
                Some(repl) => {
                    // Exactness: replacement equals [a, b] over all inputs.
                    let original = vec![a.clone(), b.clone()];
                    assert!(
                        frags_equivalent(&repl, &original, n),
                        "cross changed function: {a:?} ; {b:?} -> {repl:?}"
                    );
                    // Empty correction (length 2) iff the pair commutes.
                    assert_eq!(
                        repl.len() == 2,
                        commute(&a, &b),
                        "correction presence must match commute law for {a:?} {b:?}"
                    );
                    crossed += 1;
                }
            }
        }
        assert!(crossed > 0 && refused > 0, "want both outcomes exercised");
    }

    #[test]
    fn cross_arity_grows_with_richer_blocker() {
        // Blocker x1 += x2*x5 read by mover x2 += x3: crossing leaves x1 += x3*x5 (arity 2),
        // not a same-arity correction.
        let blocker = g(1, &[(2, true), (5, true)]); // a (first)
        let mover = g(2, &[(3, true)]); // b (second)
        let repl = cross(&blocker, &mover).expect("single-sided, not mutual");
        assert_eq!(repl.len(), 3, "expected a correction");
        let corr = &repl[2];
        assert_eq!(corr.target, 1);
        assert_eq!(corr.arity(), 2, "correction arity should be |C_K|+|C_M|-1 = 2");
    }

    #[test]
    fn cross_refuses_mutual_dependency_swap() {
        // x1 += x2 and x2 += x1 conjugate to a wire swap — outside the single-cube family.
        let a = g(1, &[(2, true)]);
        let b = g(2, &[(1, true)]);
        assert!(cross(&a, &b).is_none());
    }

    #[test]
    fn recombine_is_exact_and_reduces() {
        let n = 8;
        for seed in 0..10 {
            let c = random_circuit(n, 30);
            let frags = expand_g57_circuit(&c.gates);
            // Split every gate once to create recombination opportunities, then recombine.
            let mut split_frags = Vec::new();
            let mut rng = StdRng::seed_from_u64(seed);
            for f in &frags {
                let free: Vec<u16> = (0..n as u16)
                    .filter(|&w| w != f.target && !f.lits.iter().any(|&(x, _)| x == w))
                    .collect();
                if f.arity() < n - 1 && !free.is_empty() {
                    let (lo, hi) = split(f, free[rng.random_range(0..free.len())]);
                    split_frags.push(lo);
                    split_frags.push(hi);
                } else {
                    split_frags.push(f.clone());
                }
            }
            assert!(frags_match_g57(&split_frags, &c.gates, n));
            let recombined = recombine(split_frags.clone());
            // Exactness preserved and mass did not increase.
            assert!(
                frags_match_g57(&recombined, &c.gates, n),
                "recombine changed function (seed {seed})"
            );
            assert!(recombined.len() <= split_frags.len());
        }
    }

    #[test]
    fn transport_mix_stays_equivalent_and_bounds_phi() {
        let n = 8;
        for seed in 0..8 {
            let c = random_circuit(n, 25);
            let report = transport_mix(&c.gates, n, 3000, seed);
            assert!(
                report.equivalent,
                "transport-mix broke equivalence (seed {seed}): {report:?}"
            );
            // Split and commute are Phi-neutral and the default loop has no crossings, so
            // nothing can pump Phi above the initial g57 mass...
            assert!(
                report.phi_peak <= report.phi_initial + 1e-6,
                "Phi rose without a crossing (seed {seed}): {report:?}"
            );
            // ...and recombination is a strict sink, so the final mass never exceeds the
            // initial and typically drops below it (the walk compresses as it mixes).
            assert!(
                report.phi_final <= report.phi_initial + 1e-6,
                "recombination increased Phi (seed {seed}): {report:?}"
            );
        }
    }

    #[test]
    fn reassemble_inverts_expansion() {
        let n = 8;
        for seed in 0..20 {
            let c = random_circuit(n, 40);
            let frags = expand_g57_circuit(&c.gates);
            let rebuilt = reassemble_g57(frags, n).expect("fresh expansion must reassemble");
            assert!(
                g57_circuits_equal(&c.gates, &rebuilt, n),
                "reassembly changed the function (seed {seed})"
            );
        }
    }

    #[test]
    fn shoot_and_reassemble_always_equivalent() {
        // The returned circuit must always compute the input permutation, both with and
        // without crossings, whether or not reassembly was clean.
        let n = 8;
        let mut clean = 0;
        let total = 24;
        for seed in 0..total {
            let c = random_circuit(n, 30);
            let allow_cross = seed % 2 == 0;
            let (out, report) = shoot_and_reassemble(&c, n, 4000, seed, allow_cross, 4);
            assert!(
                report.equivalent,
                "shot circuit not equivalent (seed {seed}, cross {allow_cross}): {report:?}"
            );
            assert!(
                c.probably_equal(&out, n, 4000).is_ok(),
                "probabilistic check failed (seed {seed})"
            );
            if report.fully_reconstructed {
                clean += 1;
            }
        }
        // Reassembly should usually be clean; this pins that the pipeline actually mixes
        // rather than always falling back. (It is not required to be clean every time.)
        assert!(
            clean > 0,
            "expected at least some runs to reassemble into a mixed g57 circuit"
        );
        eprintln!("shoot_and_reassemble: {clean}/{total} runs reassembled cleanly");
    }

    // A structured "hard" fixture like disperse.rs uses: blocks of mutually-colliding gates on
    // disjoint wire triples. Interior gates are pinned at g57 granularity (leeway ~0), but whole
    // blocks commute, so g57 dispersal can only shuffle blocks, not unpin them.
    fn pinned_block_circuit(blocks: usize, gates_per_block: usize) -> Vec<[u16; 3]> {
        let mut gates = Vec::new();
        for b in 0..blocks {
            let w = (3 * b) as u16;
            for g in 0..gates_per_block {
                if g % 2 == 0 {
                    gates.push([w, w + 1, w + 2]);
                } else {
                    gates.push([w + 1, w, w + 2]);
                }
            }
        }
        gates
    }

    fn leeway_triple(gates: &[[u16; 3]]) -> (usize, f64, usize) {
        let s = crate::replace::disperse::leeway_stats(gates, 1);
        (s.median, s.avg, s.p99)
    }

    #[test]
    #[ignore = "measurement; run with --ignored --nocapture"]
    fn effectiveness_report() {
        use std::collections::HashSet;
        let seeds = 24u64;

        // --- Case 1: random circuit ---
        let n = 10;
        let random = random_circuit(n, 80);
        // --- Case 2: structured, low-leeway circuit (the case fragments are meant to help) ---
        let structured = CircuitSeq {
            gates: pinned_block_circuit(4, 6),
        };
        let n_struct = 12;

        for (label, circ, wires) in [
            ("random", &random, n),
            ("structured", &structured, n_struct),
        ] {
            let (im, ia, ip) = leeway_triple(&circ.gates);
            eprintln!("\n===== {label}  (n={wires}, {} g57 gates) =====", circ.gates.len());
            eprintln!(
                "{:<22} {:>7} {:>8} {:>7} {:>8} {:>8}",
                "variant", "lee.med", "lee.avg", "lee.p99", "gates", "notes"
            );
            eprintln!(
                "{:<22} {:>7} {:>8.2} {:>7} {:>8} {:>8}",
                "input", im, ia, ip, circ.gates.len(), ""
            );

            // g57-level dispersal baseline.
            let mut dset: HashSet<String> = HashSet::new();
            let (mut dm, mut da, mut dp) = (0usize, 0f64, 0usize);
            for s in 0..seeds {
                let mut d = circ.gates.clone();
                crate::replace::disperse::disperse_random_topo(&mut d, None, 1_000_000, s);
                let (m, a, p) = leeway_triple(&d);
                dm += m;
                da += a;
                dp += p;
                dset.insert(CircuitSeq { gates: d }.repr());
            }
            eprintln!(
                "{:<22} {:>7} {:>8.2} {:>7} {:>8} {:>8}",
                "disperse (g57)",
                dm / seeds as usize,
                da / seeds as f64,
                dp / seeds as usize,
                circ.gates.len(),
                format!("{}/{} distinct", dset.len(), seeds)
            );

            // Existing g57-level single-gate shooter (random_data::shoot_random_gate): slide a
            // random gate until a structural collision, land at a random spot in that run.
            {
                let mut rset: HashSet<String> = HashSet::new();
                let (mut rm, mut ra, mut rp) = (0usize, 0f64, 0usize);
                for _ in 0..seeds {
                    let mut c2 = circ.clone();
                    crate::random::random_data::shoot_random_gate(&mut c2, 8000);
                    let (m, a, p) = leeway_triple(&c2.gates);
                    rm += m;
                    ra += a;
                    rp += p;
                    rset.insert(c2.repr());
                }
                eprintln!(
                    "{:<22} {:>7} {:>8.2} {:>7} {:>8} {:>8}",
                    "shoot_random_gate",
                    rm / seeds as usize,
                    ra / seeds as f64,
                    rp / seeds as usize,
                    circ.gates.len(),
                    format!("{}/{} distinct", rset.len(), seeds)
                );
            }

            // Random-walk fragment shooting (commute-only) vs. the directed collision-shooting
            // strategy (shoot to collision, split at the collision so a piece passes through).
            let run = |name: &str, shoot: &dyn Fn(u64) -> (CircuitSeq, ShotReport)| {
                let mut sset: HashSet<String> = HashSet::new();
                let (mut sm, mut sa, mut sp) = (0usize, 0f64, 0usize);
                let mut clean = 0usize;
                for s in 0..seeds {
                    let (out, rep) = shoot(s);
                    assert!(rep.equivalent, "{label}/{name} broke equivalence: {rep:?}");
                    let (m, a, p) = leeway_triple(&out.gates);
                    sm += m;
                    sa += a;
                    sp += p;
                    if rep.fully_reconstructed {
                        clean += 1;
                    }
                    sset.insert(out.repr());
                }
                eprintln!(
                    "{:<22} {:>7} {:>8.2} {:>7} {:>8} {:>8}",
                    name,
                    sm / seeds as usize,
                    sa / seeds as f64,
                    sp / seeds as usize,
                    circ.gates.len(),
                    format!("{}/{} distinct, {}/{} clean", sset.len(), seeds, clean, seeds)
                );
            };
            run("shoot commute (walk)", &|s| shoot_and_reassemble(circ, wires, 6000, s, false, 4));
            run("collision_shoot", &|s| collision_shoot(circ, wires, 2000, s, false, 3));
            run("collision_shoot+corr", &|s| collision_shoot(circ, wires, 2000, s, true, 3));
        }
    }

    // Evaluate per-wire polynomials at input state `s` into an output state.
    fn eval_polys(polys: &[Polynomial], s: u64, n: usize) -> usize {
        let mut out = 0usize;
        for i in 0..n {
            let mut bit = 0u64;
            for &m in &polys[i] {
                bit ^= ((s & m) == m) as u64;
            }
            if bit & 1 == 1 {
                out |= 1 << i;
            }
        }
        out
    }

    fn norm(p: &Polynomial) -> Polynomial {
        let mut q = p.clone();
        crate::circuit::circuit::normalize_polynomial(&mut q);
        q
    }

    #[test]
    fn fragment_polys_match_true_function_and_corrected_to_polynomial() {
        let n = 6;
        for _ in 0..20 {
            let c = random_circuit(n, 30);
            let frags = expand_g57_circuit(&c.gates);
            let fp = fragment_polys(&frags, n);

            // (1) The fragment polynomials evaluate to the true circuit function.
            for s in 0..(1u64 << n) {
                assert_eq!(
                    eval_polys(&fp, s, n),
                    c.evaluate(s as usize),
                    "fragment_polys disagrees with the true function at {s}"
                );
            }

            // (2) They equal what the *corrected* to_polynomial will produce. to_polynomial
            // currently swaps the two control pins, so feeding it the control-swapped circuit
            // yields the true polynomial — the same one fragment_polys builds. Equal polys =>
            // identical DB key after canonicalization/hash.
            let swapped = CircuitSeq {
                gates: c.gates.iter().map(|&[a, b, cc]| [a, cc, b]).collect(),
            };
            let tp = swapped.to_polynomial(n, 0, swapped.gates.len());
            for i in 0..n {
                assert_eq!(
                    norm(&fp[i]),
                    norm(&tp[i]),
                    "fragment_polys[{i}] != corrected to_polynomial[{i}]"
                );
            }
        }
    }

    #[test]
    fn to_polynomial_convention_differs_from_evaluate_by_control_swap() {
        // to_polynomial builds each gate's toggle as b*NOT(c)+1, but evaluate_index flips on
        // b=1 OR c=0 — which is the SAME gate with the two control pins swapped. This test
        // pins that down: to_polynomial equals evaluate only if we swap the two controls.
        let n = 4;
        let (mut mism_true, mut mism_swap) = (0, 0);
        for _ in 0..8 {
            let c = random_circuit(n, 15);
            let polys = c.to_polynomial(n, 0, c.gates.len());
            let swapped = CircuitSeq {
                gates: c.gates.iter().map(|&[a, b, cc]| [a, cc, b]).collect(),
            };
            for s in 0..(1u64 << n) {
                let mut out = 0usize;
                for i in 0..n {
                    let mut bit = 0u64;
                    for &m in &polys[i] {
                        bit ^= ((s & m) == m) as u64;
                    }
                    if bit & 1 == 1 {
                        out |= 1 << i;
                    }
                }
                mism_true += (out != c.evaluate(s as usize)) as i32;
                mism_swap += (out != swapped.evaluate(s as usize)) as i32;
            }
        }
        eprintln!(
            "to_polynomial: {mism_true} mismatches vs evaluate, {mism_swap} vs control-swapped evaluate"
        );
        assert!(mism_true > 0, "expected to_polynomial to differ from evaluate");
        assert_eq!(
            mism_swap, 0,
            "to_polynomial should equal evaluate with the two control pins swapped"
        );
    }

    #[test]
    fn collision_shoot_always_equivalent() {
        let n = 8;
        for seed in 0..12 {
            let c = random_circuit(n, 24);
            let allow_corr = seed % 2 == 0;
            let (_out, report) = collision_shoot(&c, n, 1500, seed, allow_corr, 4);
            assert!(
                report.equivalent,
                "collision_shoot broke equivalence (seed {seed}, corr {allow_corr}): {report:?}"
            );
        }
    }

    #[test]
    fn shooting_actually_reorders_gates() {
        // A cleanly-reassembled mix should generally differ from the input gate sequence:
        // fragments travel to positions unreachable at whole-g57 granularity.
        let n = 8;
        let c = random_circuit(n, 40);
        let mut any_reordered = false;
        for seed in 0..12 {
            let (out, report) = shoot_and_reassemble(&c, n, 6000, seed, false, 4);
            assert!(report.equivalent);
            if report.fully_reconstructed && out.gates != c.gates {
                any_reordered = true;
            }
        }
        assert!(any_reordered, "shooting never changed the gate order");
    }
}
