//! Frozen-store lookup for a single heterogeneous [`XGate`] window, used by the
//! fmix DB contraction move (see [`crate::postmix::mix`]).
//!
//! A convex/contiguous window of arbitrary-width, mixed-polarity XGates is keyed
//! by its exact function polynomial ([`crate::postmix::xpoly`]) — identical to
//! the legacy g57 key path, so the frozen stores serve both. Compressing moves
//! use the regular store; growth-capable moves try curated first, then fall back
//! to regular. Stored friends are g57 circuits, decoded back into XGates before
//! one is selected according to the requested [`DbMode`].
//!
//! The frozen store's keys and values share this crate's g57 convention: a
//! triple `[a,b,c]` is `a ^= (NOT b AND c) XOR 1` (the `evaluate_index` gate,
//! matched by both `XGate::from_g57` and `CircuitSeq::to_polynomial`). So a
//! stored triple decodes with plain [`XGate::from_g57`], and a window's true
//! function (via `xgates_to_polynomial`) canonicalizes to exactly the key the
//! DB was built under. The caller may still verify each replacement for
//! functional equivalence before splicing (optional; see the mixer's db_move).

use super::xgate::XGate;
use super::xpoly::{
    CanonicalXPolys, XPolyBudget, XPolyError, canonicalize_xgates_single_capped, xgate_used_wires,
};
use crate::circuit::circuit::{CircuitSeq, Permutation, polys_repr_blob};
use crate::replace::frozen::FrozenDb;
use rand::Rng;
use rand::seq::SliceRandom;
use xxhash_rust::xxh3::xxh3_128;

/// Decode a frozen-store g57 triple into its XGate. The store and this crate
/// share the g57 convention (see the module note), so this is plain
/// [`XGate::from_g57`]; the wrapper documents the store contract at each use.
#[inline]
pub fn db_g57_to_xgate(t: [u16; 3]) -> XGate {
    XGate::from_g57(t)
}

fn key_of(canonical: &CanonicalXPolys) -> [u8; 16] {
    xxh3_128(&polys_repr_blob(&canonical.polys)).to_le_bytes()
}

/// Cheap ANF-degree pre-filter (the XGate analog of the legacy g57
/// `degree_exceeds_dir`): does the `window` function, in the chosen direction,
/// have algebraic degree strictly greater than `d`? A window whose degree
/// exceeds every stored circuit's degree cannot match, so it is skipped before
/// the far more expensive canonicalization — this is the guard that keeps the
/// DB move from paying full canon cost on high-width windows that always miss.
///
/// Probabilistic (may return false on a true high-degree window): tests `k`
/// random (d+1)-dimensional affine subcubes and looks for a nonzero (d+1)-th
/// derivative (odd parity of some wire over the subcube). Bit-packs the
/// 2^(d+1) subcube points, 64 per word, and applies each XGate's exact
/// function. `false` also when the window touches <= d wires (degree <= #inputs
/// <= d) or when d+1 > 12 (subcube too large to probe cheaply).
pub fn xgate_degree_exceeds(
    window: &[XGate],
    reversed: bool,
    d: usize,
    k: usize,
    rng: &mut impl Rng,
) -> bool {
    let used = xgate_used_wires(window);
    let nw = used.len();
    if nw <= d || d + 1 > 12 {
        return false;
    }
    let dense = dense_remap_window(window, &used, reversed);
    let m = d + 1;
    let total = 1usize << m;
    let words = total.div_ceil(64);

    // Axis columns: col[i] bit p set iff (p >> i) & 1 (value of direction i at point p).
    let mut col = vec![vec![0u64; words]; m];
    for (i, ci) in col.iter_mut().enumerate() {
        for p in 0..total {
            if (p >> i) & 1 == 1 {
                ci[p / 64] |= 1u64 << (p % 64);
            }
        }
    }
    let mask_last = if total % 64 == 0 {
        u64::MAX
    } else {
        (1u64 << (total % 64)) - 1
    };

    for _ in 0..k {
        let dirs: Vec<Vec<bool>> = (0..m)
            .map(|_| {
                let mut v: Vec<bool> = (0..nw).map(|_| rng.random_bool(0.5)).collect();
                if v.iter().all(|&b| !b) {
                    v[rng.random_range(0..nw)] = true;
                }
                v
            })
            .collect();
        let base: Vec<bool> = (0..nw).map(|_| rng.random_bool(0.5)).collect();

        // Wire value across all points = base XOR of the direction columns hitting it.
        let mut state = vec![vec![0u64; words]; nw];
        for w in 0..nw {
            if base[w] {
                state[w].iter_mut().for_each(|x| *x = u64::MAX);
            }
            for (i, di) in dirs.iter().enumerate() {
                if di[w] {
                    for wi in 0..words {
                        state[w][wi] ^= col[i][wi];
                    }
                }
            }
        }

        // Apply each XGate: target ^= comp XOR product(literal). Literal(w,pos) is
        // state[w] (pos) or its complement (neg); product is the AND over controls.
        let mut prod = vec![0u64; words];
        for g in &dense {
            prod.iter_mut().for_each(|x| *x = u64::MAX);
            for &(w, pos) in &g.ctrls {
                let s = &state[w as usize];
                if pos {
                    for wi in 0..words {
                        prod[wi] &= s[wi];
                    }
                } else {
                    for wi in 0..words {
                        prod[wi] &= !s[wi];
                    }
                }
            }
            let t = g.target as usize;
            if g.comp {
                for wi in 0..words {
                    state[t][wi] ^= !prod[wi];
                }
            } else {
                for wi in 0..words {
                    state[t][wi] ^= prod[wi];
                }
            }
        }

        // A nonzero (d+1)-th derivative = odd parity of some wire over the subcube.
        for w in 0..nw {
            let mut par = 0u32;
            for wi in 0..words {
                let word = if wi + 1 == words {
                    state[w][wi] & mask_last
                } else {
                    state[w][wi]
                };
                par ^= word.count_ones() & 1;
            }
            if par & 1 == 1 {
                return true;
            }
        }
    }
    false
}

/// Dense-remap a window's XGates onto `[0, used.len())` (used sorted), reversing
/// gate order when `reversed`. Shared by the degree test and the key path.
fn dense_remap_window(window: &[XGate], used: &[u16], reversed: bool) -> Vec<XGate> {
    let map = |w: u16| used.binary_search(&w).expect("wire from used set") as u16;
    let mut dense: Vec<XGate> = window
        .iter()
        .map(|g| XGate {
            target: map(g.target),
            comp: g.comp,
            ctrls: g.ctrls.iter().map(|&(w, p)| (map(w), p)).collect(),
        })
        .collect();
    if reversed {
        dense.reverse();
    }
    dense
}

/// Map a canonical g57 friend back into global XGate wire space, mirroring the
/// legacy `candidate_to_circuit_space`: undo canonicalization (`order`) and the
/// dense window remap (`used_wires`), drawing fresh scratch wires when the
/// friend needs more than the window touched. Fallible (returns None) rather
/// than panicking when a friend cannot be placed. Emits XGates in this crate's
/// convention via [`db_g57_to_xgate`].
fn friend_to_xgates(
    mut friend: CircuitSeq,
    reversed: bool,
    order: &Permutation,
    used_wires: &[u16],
    num_wires: usize,
    rng: &mut impl Rng,
) -> Option<Vec<XGate>> {
    if friend.gates.is_empty() {
        return Some(Vec::new());
    }
    if used_wires.iter().any(|&w| w as usize >= num_wires) {
        return None;
    }
    if reversed {
        friend.gates.reverse();
    }

    let slots = |c: &CircuitSeq| {
        c.gates
            .iter()
            .flatten()
            .copied()
            .max()
            .map_or(0, |w| w as usize + 1)
    };

    // canonical wire -> dense window wire
    let canonical_slots = slots(&friend);
    if canonical_slots > num_wires {
        return None;
    }
    let mut canonical_to_dense = order.data.clone();
    while canonical_to_dense.len() < canonical_slots {
        canonical_to_dense.push(canonical_to_dense.len());
    }
    if friend
        .gates
        .iter()
        .flatten()
        .any(|&w| w as usize >= canonical_to_dense.len())
    {
        return None;
    }
    for gate in &mut friend.gates {
        for w in gate {
            *w = canonical_to_dense[*w as usize] as u16;
        }
    }

    // dense window wire -> global wire (scratch wires drawn at random)
    let dense_slots = slots(&friend);
    let mut dense_to_global = used_wires.to_vec();
    if dense_to_global.len() < dense_slots {
        let mut occupied = vec![false; num_wires];
        for &w in &dense_to_global {
            occupied[w as usize] = true;
        }
        let mut available: Vec<u16> = (0..num_wires)
            .filter(|&w| !occupied[w])
            .map(|w| u16::try_from(w).ok())
            .collect::<Option<Vec<_>>>()?;
        available.shuffle(rng);
        let need = dense_slots - dense_to_global.len();
        if available.len() < need {
            return None;
        }
        dense_to_global.extend(available.into_iter().take(need));
    }

    let mut out = Vec::with_capacity(friend.gates.len());
    for [t, p, n] in friend.gates {
        let mapped = [
            *dense_to_global.get(t as usize)?,
            *dense_to_global.get(p as usize)?,
            *dense_to_global.get(n as usize)?,
        ];
        // Reject a corrupt friend whose active wire is also a control.
        if mapped[0] == mapped[1] || mapped[0] == mapped[2] {
            return None;
        }
        out.push(db_g57_to_xgate(mapped));
    }
    Some(out)
}

/// Which replacement to pick from the equivalents the store returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DbMode {
    /// Compressing contraction: accept only friends that do not grow the window
    /// (len <= window), and pick uniformly among the SHORTEST of those.
    Compressing,
    /// Size-agnostic: pick uniformly among ALL equivalents, whatever their gate
    /// count (may grow the circuit).
    SizeAgnostic,
    /// Minimal-growth: pick uniformly among the SHORTEST equivalents, with no
    /// length restriction — pays the least possible growth to re-encode a
    /// window that has no non-growing spelling. The paid channel of the
    /// ingest-then-pay generation policy.
    MinGrow,
    /// Re-encode without growth when possible; otherwise pay for a spelling.
    /// When `pay_random` is false the paid spelling is shortest, otherwise it
    /// is drawn from all growing candidates.
    Mix,
}

impl DbMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "mix" => Some(Self::Mix),
            "comp" => Some(Self::Compressing),
            "any" => Some(Self::SizeAgnostic),
            _ => None,
        }
    }
}

/// Outcome of a store lookup for one window.
pub struct DbResult {
    /// Number of placeable equivalent circuits the store returned for this
    /// window (both canonical directions, deduped by key). This is the
    /// "how many matches" figure for the attempt recorder.
    pub match_count: usize,
    /// The replacement selected per [`DbMode`], if any qualified.
    pub chosen: Option<Vec<XGate>>,
    /// Candidates decoded to the exact outgoing window and discarded.
    pub identity_skipped: usize,
    /// Number of catalogued candidates supplied by the curated store.
    pub curated_matches: usize,
    /// Whether the selected candidate came from the curated store.
    pub chosen_curated: bool,
    /// Size of the eligible set from which the successful winner was drawn.
    pub choice_count: usize,
    /// Shortest non-decoded candidate length returned by the selected store.
    pub min_match_len: Option<usize>,
    /// True when at least one lookup direction was skipped because its exact
    /// ANF degree exceeds the frozen store's maximum represented degree.
    pub degree_skipped: bool,
}

/// Degree pre-filter configuration for a DB lookup.
#[derive(Clone, Copy, Debug)]
pub struct DegreeGuard {
    /// Max ANF degree any stored circuit can have; a window whose degree in a
    /// direction exceeds this cannot match, so that direction is skipped before
    /// canonicalization. 0 disables the guard (every direction canonicalizes).
    pub max_degree: usize,
    /// Retained for CLI/config compatibility with the former randomized
    /// pre-filter. Runtime lookups now use an exact degree cap while composing
    /// the canonical polynomial and do not consume these probes.
    pub probes: usize,
}

impl DegreeGuard {
    pub const OFF: DegreeGuard = DegreeGuard {
        max_degree: 0,
        probes: 0,
    };
}

/// Look up `window` in the frozen store and select a replacement per `mode`.
/// `num_wires` is the full circuit wire count (for scratch-wire assignment).
/// `guard` cheaply skips over-degree directions before canonicalization.
///
/// Correctness is NOT assumed from the DB: the caller may still verify the
/// returned gates are equivalent to `window` before splicing.
pub fn db_replace(
    window: &[XGate],
    num_wires: usize,
    db: &FrozenDb,
    budget: XPolyBudget,
    mode: DbMode,
    guard: DegreeGuard,
    curated: bool,
    curated_in_comp: bool,
    regular_fallback: bool,
    pay_random: bool,
    rng: &mut impl Rng,
) -> DbResult {
    let curated_armed = curated_armed_for(curated, mode, curated_in_comp);
    db_replace_with(
        window,
        num_wires,
        budget,
        mode,
        guard,
        curated_armed,
        regular_fallback,
        pay_random,
        rng,
        |key, want_curated| {
            if want_curated && curated {
                db.get_curated(key)
            } else if want_curated {
                None
            } else {
                db.get_regular(key)
            }
        },
    )
}

/// Compression normally avoids curated alternatives; the explicit override
/// lets layer-2 runs apply curated-first routing there as well.
pub fn curated_armed_for(curated: bool, mode: DbMode, curated_in_comp: bool) -> bool {
    curated && (mode != DbMode::Compressing || curated_in_comp)
}

/// Testable core: `lookup` stands in for the frozen store.
#[allow(clippy::too_many_arguments)]
pub fn db_replace_with<F>(
    window: &[XGate],
    num_wires: usize,
    budget: XPolyBudget,
    mode: DbMode,
    guard: DegreeGuard,
    curated_armed: bool,
    regular_fallback: bool,
    pay_random: bool,
    rng: &mut impl Rng,
    mut lookup: F,
) -> DbResult
where
    F: FnMut(&[u8; 16], bool) -> Option<Vec<u8>>,
{
    let miss = |degree_skipped| DbResult {
        match_count: 0,
        chosen: None,
        identity_skipped: 0,
        curated_matches: 0,
        chosen_curated: false,
        choice_count: 0,
        min_match_len: None,
        degree_skipped,
    };
    let window_len = window.len();
    if window_len == 0 {
        return miss(false);
    }

    // Compose and degree-check forward first. Curated is forward-only; reverse
    // is deferred until the regular cascade stage. Exact degree rejection
    // happens before canonical wire ordering, the expensive part of a miss.
    let mut degree_skipped = false;
    let mut directions: Vec<(bool, CanonicalXPolys)> = Vec::with_capacity(2);
    match canonicalize_xgates_single_capped(window, false, budget, guard.max_degree) {
        Ok(canonical) => directions.push((false, canonical)),
        Err(XPolyError::DegreeExceeded { .. }) => degree_skipped = true,
        Err(_) => {}
    }

    struct CandidateRef {
        value_index: usize,
        offset: usize,
        byte_len: usize,
        gates: usize,
        curated: bool,
        direction_index: usize,
    }
    impl CandidateLen for CandidateRef {
        fn gate_count(&self) -> usize {
            self.gates
        }
    }
    fn catalogue(
        value: Vec<u8>,
        curated: bool,
        direction_index: usize,
        values: &mut Vec<Vec<u8>>,
        candidates: &mut Vec<CandidateRef>,
    ) {
        let value_index = values.len();
        let mut pos = 0usize;
        while pos < value.len() {
            let byte_len = value[pos] as usize;
            pos += 1;
            if byte_len % 3 != 0
                || pos
                    .checked_add(byte_len)
                    .is_none_or(|end| end > value.len())
            {
                break;
            }
            candidates.push(CandidateRef {
                value_index,
                offset: pos,
                byte_len,
                gates: byte_len / 3,
                curated,
                direction_index,
            });
            pos += byte_len;
        }
        values.push(value);
    }

    let mut values = Vec::new();
    let mut candidates = Vec::new();

    // Stage A: curated, using the forward canonical key only. A failed or
    // over-degree forward composition simply makes this stage a miss.
    if curated_armed {
        if let Some((direction_index, (_, canonical))) = directions
            .iter()
            .enumerate()
            .find(|(_, (reversed, _))| !reversed)
        {
            let key = key_of(canonical);
            if let Some(value) = lookup(&key, true) {
                catalogue(value, true, direction_index, &mut values, &mut candidates);
            }
        }
    }

    // Stage B: regular forward and reverse, only after a complete curated miss
    // (or unconditionally when curated is not armed). `regular_fallback=false`
    // creates the curated-only primitive used by prefix-exhaustion.
    if candidates.is_empty() && (regular_fallback || !curated_armed) {
        match canonicalize_xgates_single_capped(window, true, budget, guard.max_degree) {
            Ok(canonical) => directions.push((true, canonical)),
            Err(XPolyError::DegreeExceeded { .. }) => degree_skipped = true,
            Err(_) => {}
        }
        let mut seen_keys = std::collections::HashSet::new();
        for direction_index in 0..directions.len() {
            let key = key_of(&directions[direction_index].1);
            if !seen_keys.insert(key) {
                continue;
            }
            if let Some(value) = lookup(&key, false) {
                catalogue(value, false, direction_index, &mut values, &mut candidates);
            }
        }
    }

    let match_count = candidates.len();
    let curated_matches = candidates
        .iter()
        .filter(|candidate| candidate.curated)
        .count();
    let min_match_len = candidates.iter().map(|candidate| candidate.gates).min();
    let mut chosen = None;
    let mut chosen_curated = false;
    let mut identity_skipped = 0usize;
    let mut choice_count = 0usize;
    while !candidates.is_empty() {
        let Some((pick, eligible)) = choose_ref(&candidates, window_len, mode, pay_random, rng)
        else {
            break;
        };
        let candidate = candidates.swap_remove(pick);
        let (reversed, canonical) = &directions[candidate.direction_index];
        let friend = CircuitSeq::from_blob(
            &values[candidate.value_index][candidate.offset..candidate.offset + candidate.byte_len],
        );
        let Some(gates) = friend_to_xgates(
            friend,
            *reversed,
            &canonical.order,
            &canonical.used_wires,
            num_wires,
            rng,
        ) else {
            continue;
        };
        if gates == window {
            identity_skipped += 1;
            continue;
        }
        chosen_curated = candidate.curated;
        choice_count = eligible;
        chosen = Some(gates);
        break;
    }

    DbResult {
        match_count,
        chosen,
        degree_skipped,
        identity_skipped,
        curated_matches,
        chosen_curated,
        choice_count,
        min_match_len,
    }
}

trait CandidateLen {
    fn gate_count(&self) -> usize;
}

/// Select a candidate by length without decoding it. Returns both its index
/// and the branching factor of the eligible set.
fn choose_ref<T: CandidateLen>(
    candidates: &[T],
    window_len: usize,
    mode: DbMode,
    pay_random: bool,
    rng: &mut impl Rng,
) -> Option<(usize, usize)> {
    let len_of = |index: usize| candidates[index].gate_count();
    let eligible: Vec<usize> = match mode {
        DbMode::Compressing => {
            let min = (0..candidates.len())
                .map(len_of)
                .filter(|&l| l <= window_len)
                .min()?;
            (0..candidates.len())
                .filter(|&i| len_of(i) == min)
                .collect()
        }
        DbMode::SizeAgnostic => {
            if candidates.is_empty() {
                return None;
            }
            (0..candidates.len()).collect()
        }
        DbMode::MinGrow => {
            let min = (0..candidates.len()).map(len_of).min()?;
            (0..candidates.len())
                .filter(|&i| len_of(i) == min)
                .collect()
        }
        DbMode::Mix => {
            let free: Vec<usize> = (0..candidates.len())
                .filter(|&i| len_of(i) <= window_len)
                .collect();
            if !free.is_empty() {
                free
            } else if pay_random {
                (0..candidates.len()).collect()
            } else {
                let min = (0..candidates.len()).map(len_of).min()?;
                (0..candidates.len())
                    .filter(|&i| len_of(i) == min)
                    .collect()
            }
        }
    };
    if eligible.is_empty() {
        return None;
    }
    let pick = eligible[rng.random_range(0..eligible.len())];
    Some((pick, eligible.len()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postmix::xgate::eval_lanes;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use std::collections::HashMap;

    fn exhaustively_equal(a: &[XGate], b: &[XGate], n: usize) -> bool {
        for input in 0..(1u64 << n) {
            let mut sa: Vec<u64> = (0..n).map(|w| (input >> w) & 1).collect();
            let mut sb = sa.clone();
            eval_lanes(a.iter(), &mut sa);
            eval_lanes(b.iter(), &mut sb);
            if sa.iter().zip(&sb).any(|(x, y)| (x ^ y) & 1 != 0) {
                return false;
            }
        }
        true
    }

    // Encode a value the way the frozen store does: [len][len bytes] per friend.
    fn encode_value(friends: &[CircuitSeq]) -> Vec<u8> {
        let mut v = Vec::new();
        for f in friends {
            let blob: Vec<u8> = f.gates.iter().flatten().map(|&w| w as u8).collect();
            v.push(blob.len() as u8);
            v.extend(blob);
        }
        v
    }

    // Store one 1-gate g57 friend for `legacy`'s key, in the builder's canonical
    // wire space, and return (key, value).
    fn store_friend(legacy: &CircuitSeq) -> ([u8; 16], Vec<u8>) {
        let (polys, order, _used) = legacy.canonicalize_polys_single(false);
        let key = xxh3_128(&polys_repr_blob(&polys)).to_le_bytes();
        let mut stored = legacy.clone();
        stored.rewire(&order.invert(), stored.max_wire() as usize + 1);
        stored.canonicalize();
        (key, encode_value(&[stored]))
    }

    #[test]
    fn curated_routing_rule_respects_compression_override() {
        assert!(curated_armed_for(true, DbMode::Mix, false));
        assert!(curated_armed_for(true, DbMode::SizeAgnostic, false));
        assert!(!curated_armed_for(true, DbMode::Compressing, false));
        assert!(curated_armed_for(true, DbMode::Compressing, true));
        assert!(!curated_armed_for(false, DbMode::Mix, true));
    }

    #[test]
    fn curated_only_suppresses_regular_fallback() {
        let window = vec![db_g57_to_xgate([0, 1, 2])];
        let mut rng = StdRng::seed_from_u64(77);
        let mut calls = Vec::new();
        let result = db_replace_with(
            &window,
            8,
            XPolyBudget::default(),
            DbMode::Mix,
            DegreeGuard::OFF,
            true,
            false,
            false,
            &mut rng,
            |_, curated| {
                calls.push(curated);
                None
            },
        );
        assert!(result.chosen.is_none());
        assert_eq!(calls, [true]);
    }

    #[test]
    fn mix_selection_reports_eligible_entropy_and_pay_random_pool() {
        struct Length(usize);
        impl CandidateLen for Length {
            fn gate_count(&self) -> usize {
                self.0
            }
        }
        let mut rng = StdRng::seed_from_u64(8);
        let (_, free_choices) = choose_ref(
            &[Length(1), Length(2), Length(4)],
            2,
            DbMode::Mix,
            false,
            &mut rng,
        )
        .unwrap();
        assert_eq!(free_choices, 2);

        let (_, cheapest_choices) = choose_ref(
            &[Length(3), Length(4), Length(4)],
            2,
            DbMode::Mix,
            false,
            &mut rng,
        )
        .unwrap();
        assert_eq!(cheapest_choices, 1);
        let (_, random_paid_choices) = choose_ref(
            &[Length(3), Length(4), Length(4)],
            2,
            DbMode::Mix,
            true,
            &mut rng,
        )
        .unwrap();
        assert_eq!(random_paid_choices, 3);
    }

    // A 3-gate window (one real g57 plus a cancelling involution pad) that a
    // stored 1-gate g57 friend replaces. Keyed/valued exactly as the builder.
    #[test]
    fn compressing_returns_equivalent_shorter_friend() {
        let g = db_g57_to_xgate([0, 1, 2]);
        let pad = XGate::conj(0, [(1, true), (2, false)]).unwrap();
        let window = vec![pad.clone(), pad, g]; // pad;pad = identity, so window == g

        let legacy = CircuitSeq {
            gates: vec![[0, 1, 2]],
        };
        let (key, value) = store_friend(&legacy);
        let store = HashMap::from([(key, value)]);

        let mut rng = StdRng::seed_from_u64(1);
        let res = db_replace_with(
            &window,
            8,
            XPolyBudget::default(),
            DbMode::Compressing,
            DegreeGuard::OFF,
            false,
            true,
            false,
            &mut rng,
            |k, curated| (!curated).then(|| store.get(k).cloned()).flatten(),
        );
        assert_eq!(res.match_count, 1);
        let repl = res.chosen.expect("a shorter friend exists");
        assert!(repl.len() < window.len());
        assert!(
            exhaustively_equal(&window, &repl, 8),
            "returned replacement must compute the window's function"
        );
    }

    #[test]
    fn no_hit_returns_none() {
        let window = vec![db_g57_to_xgate([0, 1, 2])];
        let mut rng = StdRng::seed_from_u64(2);
        let res = db_replace_with(
            &window,
            8,
            XPolyBudget::default(),
            DbMode::SizeAgnostic,
            DegreeGuard::OFF,
            false,
            true,
            false,
            &mut rng,
            |_, _| None,
        );
        assert_eq!(res.match_count, 0);
        assert!(res.chosen.is_none());
    }

    #[test]
    fn degree_guard_skips_high_degree_window_without_a_lookup() {
        use crate::postmix::xgate::Lits;
        use smallvec::SmallVec;
        // A single width-8 conjunction gate has ANF degree 8 — above a degree-6
        // cap, so the guard must skip it (both directions) before any lookup.
        let ctrls: Lits = (1u16..=8).map(|w| (w, true)).collect::<SmallVec<_>>();
        let wide = XGate {
            target: 0,
            comp: false,
            ctrls,
        };
        let window = vec![wide];
        let guard = DegreeGuard {
            max_degree: 6,
            probes: 6,
        };
        let mut rng = StdRng::seed_from_u64(4);
        let mut lookups = 0;
        let res = db_replace_with(
            &window,
            16,
            XPolyBudget::default(),
            DbMode::SizeAgnostic,
            guard,
            false,
            true,
            false,
            &mut rng,
            |_, _| {
                lookups += 1;
                None
            },
        );
        assert!(
            res.degree_skipped,
            "degree-8 window must be degree-skipped under a cap of 6"
        );
        assert_eq!(
            lookups, 0,
            "no store lookup should happen when degree-skipped"
        );

        // A low-degree window (two width-2 gates, degree 2) must NOT be skipped.
        let low = vec![db_g57_to_xgate([0, 1, 2]), db_g57_to_xgate([3, 4, 5])];
        let mut rng = StdRng::seed_from_u64(4);
        let res = db_replace_with(
            &low,
            16,
            XPolyBudget::default(),
            DbMode::SizeAgnostic,
            guard,
            false,
            true,
            false,
            &mut rng,
            |_, _| None,
        );
        assert!(
            !res.degree_skipped,
            "a degree-2 window must not be degree-skipped"
        );
    }

    #[test]
    fn compressing_rejects_growth_but_size_agnostic_accepts_it() {
        // Window = 1 gate; the only stored friend is 3 gates (equivalent, longer).
        let window = vec![db_g57_to_xgate([0, 1, 2])];
        let g = db_g57_to_xgate([0, 1, 2]);
        let pad = XGate::conj(0, [(1, true), (2, false)]).unwrap();
        // Store the 3-gate equivalent [pad,pad,g] under the window's key, as a
        // g57 blob — build it from a g57 circuit equal to the window's function.
        let legacy = CircuitSeq {
            gates: vec![[3, 4, 5], [3, 4, 5], [0, 1, 2]],
        };
        // legacy computes the same function as `window` (the [3,4,5] pair cancels).
        assert!(exhaustively_equal(
            &window,
            &[
                db_g57_to_xgate([3, 4, 5]),
                db_g57_to_xgate([3, 4, 5]),
                db_g57_to_xgate([0, 1, 2])
            ],
            6
        ));
        let (key, value) = store_friend(&legacy);
        let store = HashMap::from([(key, value)]);

        // Compressing: the only friend (3 gates) grows the 1-gate window -> reject.
        let mut rng = StdRng::seed_from_u64(3);
        let comp = db_replace_with(
            &window,
            8,
            XPolyBudget::default(),
            DbMode::Compressing,
            DegreeGuard::OFF,
            false,
            true,
            false,
            &mut rng,
            |k, curated| (!curated).then(|| store.get(k).cloned()).flatten(),
        );
        assert_eq!(comp.match_count, 1);
        assert!(
            comp.chosen.is_none(),
            "compressing must reject a growing friend"
        );

        // Size-agnostic: accept the longer equivalent.
        let mut rng = StdRng::seed_from_u64(3);
        let agn = db_replace_with(
            &window,
            8,
            XPolyBudget::default(),
            DbMode::SizeAgnostic,
            DegreeGuard::OFF,
            false,
            true,
            false,
            &mut rng,
            |k, curated| (!curated).then(|| store.get(k).cloned()).flatten(),
        );
        assert_eq!(agn.match_count, 1);
        let repl = agn.chosen.expect("size-agnostic accepts any length");
        assert!(repl.len() > window.len(), "this friend grows the window");
        let _ = (g, pad);
        assert!(exhaustively_equal(&window, &repl, 8));

        // MinGrow: also accepts it — the shortest spelling that exists is the
        // paid channel's whole point when nothing non-growing is available.
        let mut rng = StdRng::seed_from_u64(3);
        let mg = db_replace_with(
            &window,
            8,
            XPolyBudget::default(),
            DbMode::MinGrow,
            DegreeGuard::OFF,
            false,
            true,
            false,
            &mut rng,
            |k, curated| (!curated).then(|| store.get(k).cloned()).flatten(),
        );
        let repl = mg
            .chosen
            .expect("min-grow accepts the shortest growing friend");
        assert_eq!(repl.len(), 3);
        assert!(exhaustively_equal(&window, &repl, 8));
    }
}
