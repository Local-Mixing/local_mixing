//! Frozen-store lookup for a single heterogeneous [`XGate`] window, used by the
//! fmix DB contraction move (see [`crate::postmix::mix`]).
//!
//! A convex/contiguous window of arbitrary-width, mixed-polarity XGates is keyed
//! by its exact function polynomial ([`crate::postmix::xpoly`]) — identical to
//! the legacy g57 key path, so one frozen store serves both — and looked up in
//! the regular store. Stored friends are g57 circuits; strictly shorter ones are
//! decoded back into XGates and one is returned at random.
//!
//! The frozen store's keys and values share this crate's g57 convention: a
//! triple `[a,b,c]` is `a ^= (NOT b AND c) XOR 1` (the `evaluate_index` gate,
//! matched by both `XGate::from_g57` and `CircuitSeq::to_polynomial`). So a
//! stored triple decodes with plain [`XGate::from_g57`], and a window's true
//! function (via `xgates_to_polynomial`) canonicalizes to exactly the key the
//! DB was built under. The caller may still verify each replacement for
//! functional equivalence before splicing (optional; see the mixer's db_move).

use super::xgate::XGate;
use super::xpoly::{CanonicalXPolys, XPolyBudget, canonicalize_xgates_single, xgate_used_wires};
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
    let mask_last = if total % 64 == 0 { u64::MAX } else { (1u64 << (total % 64)) - 1 };

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
                let word = if wi + 1 == words { state[w][wi] & mask_last } else { state[w][wi] };
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

/// Parse the frozen value's `[byte_len][three-byte g57 blob]` entries without
/// panicking on a truncated/damaged value.
fn decode_value(value: &[u8]) -> Vec<CircuitSeq> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < value.len() {
        let len = value[pos] as usize;
        pos += 1;
        if len % 3 != 0 || pos.checked_add(len).is_none_or(|end| end > value.len()) {
            break;
        }
        out.push(CircuitSeq::from_blob(&value[pos..pos + len]));
        pos += len;
    }
    out
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

/// Diagnostic: return EVERY candidate the store decodes for `window`, each
/// tagged with whether it came from the curated store and which canonical
/// direction matched. No identity filtering, no mode selection, no scratch-wire
/// refusal -- this is for asking whether the STORE's answers are equivalent to
/// the window at all, which the normal path only ever answers for the one
/// candidate it happens to pick.
pub fn db_probe(
    window: &[XGate],
    num_wires: usize,
    db: &FrozenDb,
    budget: XPolyBudget,
    rng: &mut impl Rng,
) -> Vec<(Vec<XGate>, bool, bool)> {
    let mut out = Vec::new();
    for reversed in [false, true] {
        let Ok(canonical) = canonicalize_xgates_single(window, reversed, budget) else {
            continue;
        };
        let key = key_of(&canonical);
        for from_curated in [true, false] {
            let value = if from_curated { db.get_curated(&key) } else { db.get_regular(&key) };
            let Some(value) = value else { continue };
            for friend in decode_value(&value) {
                if let Some(g) = friend_to_xgates(
                    friend,
                    reversed,
                    &canonical.order,
                    &canonical.used_wires,
                    num_wires,
                    rng,
                ) {
                    out.push((g, from_curated, reversed));
                }
            }
        }
    }
    out
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
    /// Free if possible, else pay the minimum: uniform over all non-growing
    /// equivalents when any exist, otherwise uniform over the shortest. This
    /// makes the ingest-versus-pay decision PER WINDOW, from the match list the
    /// lookup already returned, which is what replaces the per-gate cheap/hard
    /// tier machinery. The asymmetry against Compressing is deliberate:
    /// Compressing is a contraction move so minimum size is its job, while Mix
    /// is a re-encoding move so entropy is — Mix therefore maximises the draw
    /// pool exactly when re-encoding is free and minimises cost only when it is
    /// not.
    Mix,
}

impl DbMode {
    pub fn parse(s: &str) -> Option<DbMode> {
        match s {
            "mix" => Some(DbMode::Mix),
            "comp" => Some(DbMode::Compressing),
            "any" => Some(DbMode::SizeAgnostic),
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
    /// Candidates dropped because they were gate-for-gate identical to the
    /// outgoing window. Splicing one is a no-op that still costs a round and
    /// still stamps a generation, so the dose meter would count a re-encoding
    /// that did not happen. ssg measured 79.6% of compressing hits as trivial
    /// identity/reorder, which is why this is excluded rather than merely
    /// counted.
    pub identity_skipped: usize,
    /// Of the surviving candidates, how many came from the curated store.
    pub curated_matches: usize,
    /// Whether the selected replacement came from the curated store.
    pub chosen_curated: bool,
    /// Length of the SHORTEST non-identical equivalent the store returned, if
    /// any. Comparing it to the window length says whether this window still
    /// admits a strictly shorter spelling -- the adversary-aligned quantity,
    /// since `fcompress` is attacker-computable and a circuit driven to its
    /// locally-minimal form has spent the spelling diversity that re-encoding
    /// buys. Reported as dmin=.
    pub min_match_len: Option<usize>,
    /// True when BOTH directions were skipped by the degree guard (a certain
    /// miss reached without any canonicalization or store lookup).
    pub degree_skipped: bool,
}

/// Degree pre-filter configuration for a DB lookup.
#[derive(Clone, Copy, Debug)]
pub struct DegreeGuard {
    /// Max ANF degree any stored circuit can have; a window whose degree in a
    /// direction exceeds this cannot match, so that direction is skipped before
    /// canonicalization. 0 disables the guard (every direction canonicalizes).
    pub max_degree: usize,
    /// Random subcubes probed per direction (more = fewer missed high-degree
    /// windows, at proportional cost).
    pub probes: usize,
}

impl DegreeGuard {
    pub const OFF: DegreeGuard = DegreeGuard { max_degree: 0, probes: 0 };
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
    rng: &mut impl Rng,
) -> DbResult {
    db_replace_with(window, num_wires, budget, mode, guard, curated, rng, |key, want_curated| {
        if want_curated {
            if curated { db.get_curated(key) } else { None }
        } else {
            db.get_regular(key)
        }
    })
}

/// Testable core: `lookup` stands in for the frozen store. `curated_armed`
/// drives the routing contract of the bounded curated DB (2026-07-30):
/// ordinary expansion (Mix / MinGrow / SizeAgnostic) probes the CURATED
/// store only when it is armed — no regular fallback — and every
/// Compressing lookup probes the REGULAR store only. Unarmed processes use
/// regular for everything, as before.
#[allow(clippy::too_many_arguments)]
pub fn db_replace_with<F>(
    window: &[XGate],
    num_wires: usize,
    budget: XPolyBudget,
    mode: DbMode,
    guard: DegreeGuard,
    curated_armed: bool,
    rng: &mut impl Rng,
    mut lookup: F,
) -> DbResult
where
    F: FnMut(&[u8; 16], bool) -> Option<Vec<u8>>,
{
    // Cascade routing (2026-07-30 selection rule): expansion probes the
    // CURATED store first (forward key only); only on a complete curated
    // miss does it fall back to the REGULAR store (forward + reverse keys).
    // The mode's size rule then applies within whichever store answered —
    // for Mix: random among no-larger spellings, else random among the
    // minimal ones. Compression (and unarmed processes) go straight to
    // regular. Reverse canonicalization is computed only if the regular
    // stage runs, so the curated fast path never pays for it.
    let curated_first = curated_armed && mode != DbMode::Compressing;
    let miss = |degree_skipped| DbResult {
        match_count: 0,
        chosen: None,
        degree_skipped,
        identity_skipped: 0,
        curated_matches: 0,
        chosen_curated: false,
        min_match_len: None,
    };
    let window_len = window.len();
    if window_len == 0 {
        return miss(false);
    }

    // Degree pre-filter: a direction certified over-degree cannot match any
    // stored circuit, so skip its (expensive) canonicalization entirely.
    let fwd_over = guard.max_degree > 0
        && xgate_degree_exceeds(window, false, guard.max_degree, guard.probes, rng);
    let rev_over = guard.max_degree > 0
        && xgate_degree_exceeds(window, true, guard.max_degree, guard.probes, rng);
    if fwd_over && rev_over {
        return miss(true); // both directions certain misses — no lookup at all
    }

    // Canonicalize the forward direction now; the reverse (regular-only, a
    // window shorter under its inverse still keys the way the builder
    // recorded it) is deferred to the regular stage below.
    let mut directions: Vec<(bool, CanonicalXPolys)> = Vec::with_capacity(2);
    if !fwd_over {
        if let Ok(c) = canonicalize_xgates_single(window, false, budget) {
            directions.push((false, c));
        }
    }

    // Every placeable equivalent circuit across both distinct keys, ANY length,
    // from both stores. Each candidate carries whether it came from the curated
    // store, because curated-ness is a lexicographic first key in `choose`.
    // Catalogue candidates WITHOUT decoding them. The stored value is a flat
    // [len][len bytes] sequence, so gate counts -- everything the selection
    // rules need -- can be read by walking offsets. Decoding was the whole cost
    // of curated: one window there offered 430,568 candidates against the
    // regular store's 6, and friend_to_xgates builds an occupancy vector and a
    // shuffled availability list PER CANDIDATE, so a single lookup did roughly
    // 70,000x the work of a regular one and curated runs never reached their
    // first checkpoint. Only the chosen candidate is decoded now.
    struct CandRef {
        vi: usize,
        off: usize,
        nbytes: usize,
        gates: usize,
        curated: bool,
        reversed: bool,
        dir_ix: usize,
    }
    impl CandLen for CandRef {
        fn gate_count(&self) -> usize {
            self.gates
        }
        fn curated(&self) -> bool {
            self.curated
        }
    }
    fn catalogue(
        value: Vec<u8>,
        from_curated: bool,
        reversed: bool,
        dir_ix: usize,
        values: &mut Vec<Vec<u8>>,
        refs: &mut Vec<CandRef>,
    ) {
        let vi = values.len();
        let mut pos = 0usize;
        while pos < value.len() {
            let len = value[pos] as usize;
            pos += 1;
            if len % 3 != 0 || pos.checked_add(len).is_none_or(|e| e > value.len()) {
                break;
            }
            refs.push(CandRef { vi, off: pos, nbytes: len, gates: len / 3, curated: from_curated, reversed, dir_ix });
            pos += len;
        }
        values.push(value);
    }

    let mut values: Vec<Vec<u8>> = Vec::new();
    let mut refs: Vec<CandRef> = Vec::new();
    let mut identity_skipped = 0usize;

    // Stage A: the curated store, FORWARD KEY ONLY. The store docs say so
    // outright: "curated lookup itself uses the forward canonical form; the
    // regular fallback may also try the reversed form." Probing curated with
    // the reverse key returns entries belonging to a different permutation --
    // measured at 430,568 candidates for one window, none of them equivalent.
    if curated_first {
        if let Some((dir_ix, (_, canonical))) =
            directions.iter().enumerate().find(|(_, (rev, _))| !rev)
        {
            let key = key_of(canonical);
            if let Some(value) = lookup(&key, true) {
                catalogue(value, true, false, dir_ix, &mut values, &mut refs);
            }
        }
    }

    // Stage B: the regular store (both keys), on a complete curated miss or
    // whenever the cascade does not apply (compression, unarmed).
    if refs.is_empty() {
        if !rev_over {
            if let Ok(c) = canonicalize_xgates_single(window, true, budget) {
                directions.push((true, c));
            }
        }
        let mut seen_keys = std::collections::HashSet::new();
        for (dir_ix, (reversed, canonical)) in directions.iter().enumerate() {
            let key = key_of(canonical);
            if !seen_keys.insert(key) {
                continue;
            }
            if let Some(value) = lookup(&key, false) {
                catalogue(value, false, *reversed, dir_ix, &mut values, &mut refs);
            }
        }
    }

    let match_count = refs.len();
    let curated_matches = refs.iter().filter(|r| r.curated).count();
    let min_match_len = refs.iter().map(|r| r.gates).min();

    // Select on gate counts alone, then decode ONLY the winner. A candidate
    // that fails to place, or that turns out to be the window itself, is
    // dropped and the choice retried -- the identity guard still applies, it
    // just no longer costs a decode of every sibling to enforce.
    let mut chosen: Option<Vec<XGate>> = None;
    let mut chosen_curated = false;
    while !refs.is_empty() {
        let Some(pick) = choose_ref(&refs, window_len, mode, rng) else { break };
        let r = refs.swap_remove(pick);
        let (rev, canonical) = &directions[r.dir_ix];
        let friend = CircuitSeq::from_blob(&values[r.vi][r.off..r.off + r.nbytes]);
        let Some(gates) = friend_to_xgates(
            friend,
            *rev,
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
        chosen = Some(gates);
        chosen_curated = r.curated;
        break;
    }
    DbResult {
        match_count,
        chosen,
        degree_skipped: false,
        identity_skipped,
        curated_matches,
        chosen_curated,
        min_match_len,
    }
}

/// Pick which candidate to decode, from gate counts alone.
///
/// Curated-ness is a lexicographic FIRST key: when any curated candidate
/// survived, the mode's size rule is applied within the curated class only,
/// regardless of size. The curated store is built from splits of minimal
/// identities, so it holds LONGER equivalents -- preferring it therefore prefers
/// growth, deliberately, to buy a route whose pieces are not locally
/// compressible. Compressing mode is exempt: its job is to shrink.
fn choose_ref<R>(refs: &[R], window_len: usize, mode: DbMode, rng: &mut impl Rng) -> Option<usize>
where
    R: CandLen,
{
    let restrict_curated = mode != DbMode::Compressing && refs.iter().any(|r| r.curated());
    let pool: Vec<usize> = (0..refs.len())
        .filter(|&i| !restrict_curated || refs[i].curated())
        .collect();
    if pool.is_empty() {
        return None;
    }
    let len_of = |i: usize| refs[i].gate_count();
    let eligible: Vec<usize> = match mode {
        // Non-growing only, then narrow to the shortest.
        DbMode::Compressing => {
            let min = pool.iter().map(|&i| len_of(i)).filter(|&l| l <= window_len).min()?;
            pool.into_iter().filter(|&i| len_of(i) == min).collect()
        }
        // Anything the store returned.
        DbMode::SizeAgnostic => pool,
        // The shortest spelling that exists, growing or not.
        DbMode::MinGrow => {
            let min = pool.iter().map(|&i| len_of(i)).min()?;
            pool.into_iter().filter(|&i| len_of(i) == min).collect()
        }
        // Free if any free spelling exists, else the cheapest paid one.
        DbMode::Mix => {
            let free: Vec<usize> =
                pool.iter().copied().filter(|&i| len_of(i) <= window_len).collect();
            if free.is_empty() {
                let min = pool.iter().map(|&i| len_of(i)).min()?;
                pool.into_iter().filter(|&i| len_of(i) == min).collect()
            } else {
                free
            }
        }
    };
    if eligible.is_empty() {
        return None;
    }
    Some(eligible[rng.random_range(0..eligible.len())])
}

/// Just enough of a candidate to select on, so selection never decodes.
trait CandLen {
    fn gate_count(&self) -> usize;
    fn curated(&self) -> bool;
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

    // A 3-gate window (one real g57 plus a cancelling involution pad) that a
    // stored 1-gate g57 friend replaces. Keyed/valued exactly as the builder.
    #[test]
    fn compressing_returns_equivalent_shorter_friend() {
        let g = db_g57_to_xgate([0, 1, 2]);
        let pad = XGate::conj(0, [(1, true), (2, false)]).unwrap();
        let window = vec![pad.clone(), pad, g]; // pad;pad = identity, so window == g

        let legacy = CircuitSeq { gates: vec![[0, 1, 2]] };
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
            &mut rng,
            |k, cur| if cur { None } else { store.get(k).cloned() },
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
            &mut rng,
            |_, _| None,
        );
        assert_eq!(res.match_count, 0);
        assert!(res.chosen.is_none());
    }

    // The 2026-07-30 cascade: expansion with curated armed probes curated
    // (forward key) FIRST and, on a hit, never touches regular; on a miss it
    // falls back to regular. Compression never probes curated.
    #[test]
    fn cascade_probes_curated_first_then_regular_on_miss() {
        let window = vec![db_g57_to_xgate([0, 1, 2]), db_g57_to_xgate([3, 4, 5])];
        // Curated HIT: value = one 2-gate circuit (identical to nothing we
        // check here — probe order is the point).
        let hit_value = vec![6u8, 0, 1, 2, 3, 4, 5];
        let mut probes: Vec<bool> = Vec::new();
        let mut rng = StdRng::seed_from_u64(7);
        let _ = db_replace_with(
            &window,
            16,
            XPolyBudget::default(),
            DbMode::Mix,
            DegreeGuard::OFF,
            true,
            &mut rng,
            |_, cur| {
                probes.push(cur);
                if cur { Some(hit_value.clone()) } else { None }
            },
        );
        assert_eq!(probes, vec![true], "curated hit must suppress the regular probe");

        // Curated MISS: regular must be probed (both keys where distinct).
        let mut probes: Vec<bool> = Vec::new();
        let mut rng = StdRng::seed_from_u64(7);
        let _ = db_replace_with(
            &window,
            16,
            XPolyBudget::default(),
            DbMode::Mix,
            DegreeGuard::OFF,
            true,
            &mut rng,
            |_, cur| {
                probes.push(cur);
                None
            },
        );
        assert!(probes.first() == Some(&true), "curated probed first");
        assert!(probes.iter().skip(1).all(|&c| !c), "fallback probes are regular");
        assert!(probes.len() >= 2, "regular fallback must actually fire");

        // Compression: never curated, even when armed.
        let mut probes: Vec<bool> = Vec::new();
        let mut rng = StdRng::seed_from_u64(7);
        let _ = db_replace_with(
            &window,
            16,
            XPolyBudget::default(),
            DbMode::Compressing,
            DegreeGuard::OFF,
            true,
            &mut rng,
            |_, cur| {
                probes.push(cur);
                None
            },
        );
        assert!(!probes.is_empty() && probes.iter().all(|&c| !c), "COMP is regular-only");
    }

    #[test]
    fn degree_guard_skips_high_degree_window_without_a_lookup() {
        use crate::postmix::xgate::Lits;
        use smallvec::SmallVec;
        // A single width-8 conjunction gate has ANF degree 8 — above a degree-6
        // cap, so the guard must skip it (both directions) before any lookup.
        let ctrls: Lits = (1u16..=8).map(|w| (w, true)).collect::<SmallVec<_>>();
        let wide = XGate { target: 0, comp: false, ctrls };
        let window = vec![wide];
        let guard = DegreeGuard { max_degree: 6, probes: 6 };
        let mut rng = StdRng::seed_from_u64(4);
        let mut lookups = 0;
        let res = db_replace_with(
            &window,
            16,
            XPolyBudget::default(),
            DbMode::SizeAgnostic,
            guard,
            false,
            &mut rng,
            |_, _| {
                lookups += 1;
                None
            },
        );
        assert!(res.degree_skipped, "degree-8 window must be degree-skipped under a cap of 6");
        assert_eq!(lookups, 0, "no store lookup should happen when degree-skipped");

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
            &mut rng,
            |_, _| None,
        );
        assert!(!res.degree_skipped, "a degree-2 window must not be degree-skipped");
    }

    #[test]
    fn compressing_rejects_growth_but_size_agnostic_accepts_it() {
        // Window = 1 gate; the only stored friend is 3 gates (equivalent, longer).
        let window = vec![db_g57_to_xgate([0, 1, 2])];
        let g = db_g57_to_xgate([0, 1, 2]);
        let pad = XGate::conj(0, [(1, true), (2, false)]).unwrap();
        // Store the 3-gate equivalent [pad,pad,g] under the window's key, as a
        // g57 blob — build it from a g57 circuit equal to the window's function.
        let legacy = CircuitSeq { gates: vec![[3, 4, 5], [3, 4, 5], [0, 1, 2]] };
        // legacy computes the same function as `window` (the [3,4,5] pair cancels).
        assert!(exhaustively_equal(&window, &[
            db_g57_to_xgate([3,4,5]), db_g57_to_xgate([3,4,5]), db_g57_to_xgate([0,1,2])
        ], 6));
        let (key, value) = store_friend(&legacy);
        let store = HashMap::from([(key, value)]);

        // Compressing: the only friend (3 gates) grows the 1-gate window -> reject.
        let mut rng = StdRng::seed_from_u64(3);
        let comp = db_replace_with(&window, 8, XPolyBudget::default(), DbMode::Compressing, DegreeGuard::OFF, false, &mut rng, |k, cur| if cur { None } else { store.get(k).cloned() });
        assert_eq!(comp.match_count, 1);
        assert!(comp.chosen.is_none(), "compressing must reject a growing friend");

        // Size-agnostic: accept the longer equivalent.
        let mut rng = StdRng::seed_from_u64(3);
        let agn = db_replace_with(&window, 8, XPolyBudget::default(), DbMode::SizeAgnostic, DegreeGuard::OFF, false, &mut rng, |k, cur| if cur { None } else { store.get(k).cloned() });
        assert_eq!(agn.match_count, 1);
        let repl = agn.chosen.expect("size-agnostic accepts any length");
        assert!(repl.len() > window.len(), "this friend grows the window");
        let _ = (g, pad);
        assert!(exhaustively_equal(&window, &repl, 8));

        // MinGrow: also accepts it — the shortest spelling that exists is the
        // paid channel's whole point when nothing non-growing is available.
        let mut rng = StdRng::seed_from_u64(3);
        let mg = db_replace_with(&window, 8, XPolyBudget::default(), DbMode::MinGrow, DegreeGuard::OFF, false, &mut rng, |k, cur| if cur { None } else { store.get(k).cloned() });
        let repl = mg.chosen.expect("min-grow accepts the shortest growing friend");
        assert_eq!(repl.len(), 3);
        assert!(exhaustively_equal(&window, &repl, 8));
    }
}
