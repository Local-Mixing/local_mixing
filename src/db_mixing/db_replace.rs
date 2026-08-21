//! Frozen-store lookup for a single heterogeneous [`XGate`] window, used by the
//! fmix DB contraction move (see [`crate::engine::mix`]).
//!
//! A convex/contiguous window of arbitrary-width, mixed-polarity XGates is keyed
//! by its exact function polynomial ([`crate::engine::xpoly`]) — identical to
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

use crate::circuit::xgate::XGate;
use crate::circuit::{CircuitSeq, Permutation, polys_repr_blob};
use crate::db_mixing::frozen::FrozenDb;
use crate::engine::xpoly::{
    CanonicalXPolys, XPolyBudget, XPolyError, canonicalize_xgates_single,
    canonicalize_xgates_single_capped, xgates_to_polynomial,
};
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

/// EXACT equivalence by ANF comparison, for windows too wide to verify
/// exhaustively.
///
/// `rules::verify_rewrite` evaluates both sides on every assignment, so its
/// cost is `2^support` and it is capped at 24 wires -- past that a replacement
/// cannot be checked and has to be declined. But two gate sequences compute the
/// same function exactly when their per-wire output polynomials are equal, and
/// the polynomial machinery is already on the lookup path (it builds the store
/// key). So a wide window can be verified by composing both sides over their
/// combined support and comparing: the cost is bounded by the polynomial TERM
/// count, not by the support size. Measured on the production store, entries
/// spanning >= 24 wires carry at most 99 terms per wire and 233 in total -- a
/// comparison, against 16.7M evaluations at the exhaustive cap.
///
/// This is a proof, not a probabilistic check: the ANF *is* the function.
///
/// Returns `None` when the support exceeds the 64-wire polynomial variable
/// limit or the budget is hit -- undecided, so the caller must decline rather
/// than assume.
pub fn polys_equivalent(a: &[XGate], b: &[XGate], budget: XPolyBudget) -> Option<bool> {
    let mut used: Vec<u16> = a
        .iter()
        .chain(b.iter())
        .flat_map(|g| std::iter::once(g.target).chain(g.ctrls.iter().map(|&(w, _)| w)))
        .collect();
    used.sort_unstable();
    used.dedup();
    if used.len() > 64 {
        return None;
    }
    let nw = used.len();
    let da = dense_remap_window(a, &used, false);
    let db = dense_remap_window(b, &used, false);
    let pa = xgates_to_polynomial(&da, nw, budget).ok()?;
    let pb = xgates_to_polynomial(&db, nw, budget).ok()?;
    // Polynomials are normalised (sorted, XOR-cancelled) by construction, so
    // structural equality is functional equality.
    Some(pa == pb)
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
            let value = if from_curated {
                db.get_curated(&key)
            } else {
                db.get_regular(&key)
            };
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
    /// Candidates dropped by the pair-window reorder ban (`ban_reorder`): a
    /// permutation of the outgoing gates computes the same function only
    /// because they commute — the reorder half of the identity/reorder
    /// pathology, which the gate-for-gate guard above cannot see. Armed only
    /// for pair-geometry windows; zero everywhere else.
    pub permutation_skipped: usize,
    /// Of the surviving candidates, how many came from the curated store.
    pub curated_matches: usize,
    /// Whether the selected replacement came from the curated store.
    pub chosen_curated: bool,
    /// SELECTION ENTROPY, per successful splice: the size of the eligible
    /// set the winner was actually drawn from, i.e. the candidates the
    /// mode's size rule admitted (not `match_count`, which is everything
    /// the store returned before filtering). 1 means the replacement was
    /// forced; k > 1 means log2(k) bits of choice entered the circuit
    /// through WHICH gates were spliced -- distinct from the entropy of
    /// where the splice happened. 0 when nothing was chosen.
    pub choice_count: usize,
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
    ban_reorder: bool,
    rng: &mut impl Rng,
) -> DbResult {
    let armed = curated_armed_for(curated, mode, curated_in_comp);
    db_replace_with_value(
        window,
        num_wires,
        budget,
        mode,
        guard,
        armed,
        regular_fallback,
        pay_random,
        ban_reorder,
        rng,
        |key, want_curated| {
            // Every probe goes through the exact process-wide lookup cache
            // (see src/db_mixing/replace.rs): the store is immutable, so cached
            // hits AND misses are byte-identical to raw probes.
            use crate::db_mixing::replace::{LOOKUP_NS_CURATED, LOOKUP_NS_SHARD, cached_db_get};
            if want_curated {
                if curated {
                    cached_db_get(db, LOOKUP_NS_CURATED, key)
                } else {
                    None
                }
            } else {
                cached_db_get(db, LOOKUP_NS_SHARD, key)
            }
        },
    )
}

/// THE ROUTING RULE, and its override. Compression does not probe the curated
/// store by default.
///
/// WHY THE CURATED STORE EXISTS. Not length -- difference. A conversion is
/// only worth its cost if the incoming circuit is MEANINGFULLY different from
/// the outgoing one, and the curated store is built to raise the probability
/// that it is. Split a minimal identity `C = A.B`: then `perm(A) =
/// perm(B)^-1`, so B^-1 is an alternative spelling of A's function -- and
/// because C is a MINIMAL identity, A and B^-1 cannot be closely related by
/// local rewriting, or C would have reduced. Every portion of a minimal
/// identity is meaningfully different from its complement, so swapping one for
/// the other is a good conversion by construction.
///
/// Two consequences worth keeping straight:
///
/// (a) Size is a side effect, not the point. An uneven split gives halves of
///     unequal length, so the store holds longer-than-minimal spellings, and
///     `choose_ref` compounds that by giving curated lexicographic priority --
///     a free (non-growing) regular candidate is discarded unseen whenever
///     curated answers. Growth is the price of difference, not a goal.
///
/// (b) `bits/splice` UNDER-STATES this store. That measure is the entropy of
///     the eligible set, which treats a trivial respelling and a
///     minimal-identity complement as equally good alternatives. Curated
///     candidates are different by construction; regular ones need not be. The
///     measured 3.4x on curated-exhaust is therefore a lower bound on what the
///     store buys.
///
/// (Tempting but false: "every split piece is minimal, else a shorter piece
/// would give a shorter identity". The shorter piece can be B^-1 itself, and
/// B^-1.B is a trivial identity, so nothing is contradicted.)
///
/// `curated_in_comp` arms curated for compression, where the size rule keeps
/// only the spellings strictly shorter than the window -- the shorter halves.
///
/// This lives in the policy wrapper, not in `db_replace_with`: the mechanism
/// takes `curated_armed` to mean "probe curated for THIS call", full stop, so
/// the rule is stated once and can be overridden without threading a mode
/// exception through the lookup path.
pub fn curated_armed_for(curated: bool, mode: DbMode, curated_in_comp: bool) -> bool {
    curated && (mode != DbMode::Compressing || curated_in_comp)
}

/// Testable core: `lookup` stands in for the frozen store. `curated_armed`
/// means "probe curated for THIS call", regardless of mode — the caller owns
/// the mode rule (`curated_armed_for`: expansion always, Compressing only
/// when `curated_in_comp` arms it). An armed call probes the CURATED store
/// first (forward key only) and consults regular only per
/// `regular_fallback`; unarmed calls use regular for everything.
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
    ban_reorder: bool,
    rng: &mut impl Rng,
    lookup: F,
) -> DbResult
where
    F: FnMut(&[u8; 16], bool) -> Option<Vec<u8>>,
{
    db_replace_with_value(
        window,
        num_wires,
        budget,
        mode,
        guard,
        curated_armed,
        regular_fallback,
        pay_random,
        ban_reorder,
        rng,
        lookup,
    )
}

/// Generic lookup core. Production keeps the cached `Arc<[u8]>` alive through
/// candidate catalogue/decode, while the public test seam above retains its
/// historical `Vec<u8>` closure API (including inference for bare `None`).
#[allow(clippy::too_many_arguments)]
fn db_replace_with_value<F, V>(
    window: &[XGate],
    num_wires: usize,
    budget: XPolyBudget,
    mode: DbMode,
    guard: DegreeGuard,
    curated_armed: bool,
    regular_fallback: bool,
    pay_random: bool,
    ban_reorder: bool,
    rng: &mut impl Rng,
    mut lookup: F,
) -> DbResult
where
    F: FnMut(&[u8; 16], bool) -> Option<V>,
    V: AsRef<[u8]>,
{
    // Cascade routing (2026-07-30 selection rule): expansion probes the
    // CURATED store first (forward key only); only on a complete curated
    // miss does it fall back to the REGULAR store (forward + reverse keys).
    // The mode's size rule then applies within whichever store answered —
    // for Mix: random among no-larger spellings, else random among the
    // minimal ones. Compression (and unarmed processes) go straight to
    // regular. Reverse canonicalization is computed only if the regular
    // stage runs, so the curated fast path never pays for it.
    // The caller owns the mode rule (see `db_replace`): `curated_armed` means
    // "probe curated for THIS call", full stop.
    let curated_first = curated_armed;
    let miss = |degree_skipped| DbResult {
        match_count: 0,
        chosen: None,
        degree_skipped,
        identity_skipped: 0,
        permutation_skipped: 0,
        curated_matches: 0,
        chosen_curated: false,
        choice_count: 0,
        min_match_len: None,
    };
    let window_len = window.len();
    if window_len == 0 {
        return miss(false);
    }

    // Degree filter: a direction whose ANF degree exceeds the store's maximum
    // cannot match any stored circuit, so its (expensive) canonicalization is
    // skipped. The check now sits INSIDE canonicalization, between composing
    // the polynomial and canonicalizing it -- exact, and free, because that
    // polynomial is needed regardless. It replaces a randomized subspace probe
    // that cost 17x the polynomial it guarded and never fired in production.
    let mut degree_skipped = false;

    // Canonicalize the forward direction now; the reverse (regular-only, a
    // window shorter under its inverse still keys the way the builder
    // recorded it) is deferred to the regular stage below.
    let mut directions: Vec<(bool, CanonicalXPolys)> = Vec::with_capacity(2);
    match canonicalize_xgates_single_capped(window, false, budget, guard.max_degree) {
        Ok(c) => directions.push((false, c)),
        Err(XPolyError::DegreeExceeded { .. }) => degree_skipped = true,
        Err(_) => {}
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
    fn catalogue<V: AsRef<[u8]>>(
        value: V,
        from_curated: bool,
        dir_ix: usize,
        values: &mut Vec<V>,
        refs: &mut Vec<CandRef>,
        curated_matches: &mut usize,
        min_match_len: &mut Option<usize>,
    ) {
        let vi = values.len();
        let bytes = value.as_ref();
        // A full-coverage curated value can hold hundreds of thousands of
        // candidates; growing `refs` from empty re-copied it ~19 times on the
        // way up. Each candidate costs at least 4 bytes here (a length byte
        // plus a 3-byte gate), so this is an estimate, not a bound -- capacity
        // only, so an over- or under-shoot changes nothing but allocator work.
        refs.reserve(bytes.len() / 4);
        let mut pos = 0usize;
        while pos < bytes.len() {
            let len = bytes[pos] as usize;
            pos += 1;
            if len % 3 != 0 || pos.checked_add(len).is_none_or(|e| e > bytes.len()) {
                break;
            }
            let gates = len / 3;
            refs.push(CandRef {
                vi,
                off: pos,
                nbytes: len,
                gates,
                curated: from_curated,
                dir_ix,
            });
            *curated_matches += usize::from(from_curated);
            *min_match_len = Some(min_match_len.map_or(gates, |min| min.min(gates)));
            pos += len;
        }
        values.push(value);
    }

    let mut values: Vec<V> = Vec::new();
    let mut refs: Vec<CandRef> = Vec::new();
    let mut curated_matches = 0usize;
    let mut min_match_len = None;
    let mut identity_skipped = 0usize;
    let mut permutation_skipped = 0usize;

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
                catalogue(
                    value,
                    true,
                    dir_ix,
                    &mut values,
                    &mut refs,
                    &mut curated_matches,
                    &mut min_match_len,
                );
            }
        }
    }

    // Stage B: the regular store (both keys), on a complete curated miss or
    // whenever the cascade does not apply (compression, unarmed).
    //
    // `regular_fallback = false` SUPPRESSES this stage while the cascade is
    // live, turning the call into a curated-only probe. That is what the
    // two-pass policy needs: exhaust curated over every window length before
    // the regular store is consulted at any length. It can never suppress
    // the regular store when the cascade does not apply -- with no curated
    // stage to fall back FROM, stage B is the only stage there is.
    if refs.is_empty() && (regular_fallback || !curated_first) {
        match canonicalize_xgates_single_capped(window, true, budget, guard.max_degree) {
            Ok(c) => directions.push((true, c)),
            Err(XPolyError::DegreeExceeded { .. }) => degree_skipped = true,
            Err(_) => {}
        }
        // The regular store is keyed by min(canon_fwd, canon_rev) (see the
        // MIN_DIR_LOOKUP note in src/db_mixing/replace.rs): when both directions
        // composed, the default Min mode probes only the min direction — the
        // non-min key can only exist if it equals the min key, so the candidate
        // set is unchanged. Legacy restores the historical probe-both cascade;
        // Validate probes the other direction on a miss and counts violations.
        use crate::db_mixing::replace::{
            MIN_DIR_VALIDATE_PROBES, MIN_DIR_VIOLATIONS, MinDirLookup, min_dir_lookup_mode,
        };
        use std::sync::atomic::Ordering;
        let fwd_idx = directions.iter().position(|(reversed, _)| !reversed);
        let rev_idx = directions.iter().position(|(reversed, _)| *reversed);
        let min_mode = match (fwd_idx, rev_idx) {
            (Some(_), Some(_)) => min_dir_lookup_mode(),
            _ => MinDirLookup::Legacy,
        };
        match min_mode {
            MinDirLookup::Legacy => {
                // At most two directions exist; dedup the second key against the first.
                let mut first_key: Option<[u8; 16]> = None;
                for (dir_ix, (_, canonical)) in directions.iter().enumerate() {
                    let key = key_of(canonical);
                    if first_key == Some(key) {
                        continue;
                    }
                    if first_key.is_none() {
                        first_key = Some(key);
                    }
                    if let Some(value) = lookup(&key, false) {
                        catalogue(
                            value,
                            false,
                            dir_ix,
                            &mut values,
                            &mut refs,
                            &mut curated_matches,
                            &mut min_match_len,
                        );
                    }
                }
            }
            mode => {
                let fi = fwd_idx.expect("min mode requires both directions");
                let ri = rev_idx.expect("min mode requires both directions");
                let rev_is_min = directions[ri].1.polys < directions[fi].1.polys;
                let (min_idx, alt_idx) = if rev_is_min { (ri, fi) } else { (fi, ri) };
                let min_key = key_of(&directions[min_idx].1);
                if let Some(value) = lookup(&min_key, false) {
                    catalogue(
                        value,
                        false,
                        min_idx,
                        &mut values,
                        &mut refs,
                        &mut curated_matches,
                        &mut min_match_len,
                    );
                } else if mode == MinDirLookup::Validate {
                    let alt_key = key_of(&directions[alt_idx].1);
                    if alt_key != min_key {
                        MIN_DIR_VALIDATE_PROBES.fetch_add(1, Ordering::Relaxed);
                        if let Some(value) = lookup(&alt_key, false) {
                            MIN_DIR_VIOLATIONS.fetch_add(1, Ordering::Relaxed);
                            eprintln!(
                                "[min-dir-violation] fmix db_replace: non-min canonical key present while min key absent (window={})",
                                window.len()
                            );
                            catalogue(
                                value,
                                false,
                                alt_idx,
                                &mut values,
                                &mut refs,
                                &mut curated_matches,
                                &mut min_match_len,
                            );
                        }
                    }
                }
            }
        }
    }

    let match_count = refs.len();

    // Select on gate counts alone, then decode ONLY the winner. A candidate
    // that fails to place, or that turns out to be the window itself, is
    // dropped and the choice retried -- the identity guard still applies, it
    // just no longer costs a decode of every sibling to enforce.
    let mut chosen: Option<Vec<XGate>> = None;
    let mut chosen_curated = false;
    let mut choice_count = 0usize;
    while !refs.is_empty() {
        let Some((pick, n_eligible)) = choose_ref(&refs, window_len, mode, pay_random, rng) else {
            break;
        };
        let r = refs.swap_remove(pick);
        let (rev, canonical) = &directions[r.dir_ix];
        let value = values[r.vi].as_ref();
        let friend = CircuitSeq::from_blob(&value[r.off..r.off + r.nbytes]);
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
        // Pair-window reorder ban: a candidate that merely permutes the
        // outgoing gates is a re-spelling commutation gives away for free.
        // The ban lives INSIDE this retry loop on purpose — the banned
        // candidate is consumed and choose_ref re-evaluates free-vs-pay on
        // the survivors, so with both bans armed a commuting 2-gate pair has
        // no admissible same-length spelling and the splice is forced onto a
        // genuinely different (usually paid/curated) one.
        if ban_reorder && is_reorder(&gates, window) {
            permutation_skipped += 1;
            continue;
        }
        chosen = Some(gates);
        chosen_curated = r.curated;
        choice_count = n_eligible;
        break;
    }
    DbResult {
        match_count,
        chosen,
        degree_skipped,
        identity_skipped,
        permutation_skipped,
        curated_matches,
        chosen_curated,
        choice_count,
        min_match_len,
    }
}

/// Gate-multiset equality: `a` is a reordering of `b` (including the identical
/// order, which the identity guard catches first). O(n²) with a used-mask —
/// windows here are tiny.
fn is_reorder(a: &[XGate], b: &[XGate]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut used = vec![false; b.len()];
    'outer: for g in a {
        for (i, h) in b.iter().enumerate() {
            if !used[i] && g == h {
                used[i] = true;
                continue 'outer;
            }
        }
        return false;
    }
    true
}

/// Pick which candidate to decode, from gate counts alone.
///
/// Curated-ness is a lexicographic FIRST key: when any curated candidate
/// survived, the mode's size rule is applied within the curated class only,
/// regardless of size. That is deliberate, and it prefers growth two ways: the
/// store holds the longer halves of uneven identity splits, and a free
/// (non-growing) regular candidate is discarded unseen whenever curated
/// answers at all. What it buys is a route whose pieces are not locally
/// compressible by the REGULAR store. Compressing mode is exempt: its job is
/// to shrink.
fn choose_ref<R>(
    refs: &[R],
    window_len: usize,
    mode: DbMode,
    pay_random: bool,
    rng: &mut impl Rng,
) -> Option<(usize, usize)>
where
    R: CandLen,
{
    let restrict_curated = mode != DbMode::Compressing && refs.iter().any(|r| r.curated());
    let in_pool = |r: &R| !restrict_curated || r.curated();

    // First determine the rule's threshold without materialising either the
    // curated pool or the eligible pool. The second scan counts candidates,
    // and the third maps the uniformly drawn rank back to the same slice index
    // the old Vec-based implementation returned. Candidate order and the one
    // RNG call are therefore unchanged, including after swap_remove retries.
    let min_pool_len = || {
        refs.iter()
            .filter(|r| in_pool(r))
            .map(CandLen::gate_count)
            .min()
    };
    let free_exists = mode == DbMode::Mix
        && refs
            .iter()
            .any(|r| in_pool(r) && r.gate_count() <= window_len);
    let target_len = match mode {
        DbMode::Compressing => refs
            .iter()
            .filter(|r| in_pool(r))
            .map(CandLen::gate_count)
            .filter(|&len| len <= window_len)
            .min(),
        DbMode::MinGrow => min_pool_len(),
        DbMode::Mix if !free_exists && !pay_random => min_pool_len(),
        DbMode::SizeAgnostic | DbMode::Mix => None,
    };
    if matches!(mode, DbMode::Compressing | DbMode::MinGrow) && target_len.is_none() {
        return None;
    }

    let eligible = |r: &R| {
        if !in_pool(r) {
            return false;
        }
        match mode {
            DbMode::Compressing | DbMode::MinGrow => Some(r.gate_count()) == target_len,
            DbMode::SizeAgnostic => true,
            DbMode::Mix if free_exists => r.gate_count() <= window_len,
            DbMode::Mix if pay_random => true,
            DbMode::Mix => Some(r.gate_count()) == target_len,
        }
    };
    let choice_count = refs.iter().filter(|r| eligible(r)).count();
    if choice_count == 0 {
        return None;
    }

    // The eligible-set size travels with the pick: it is the branching
    // factor of this selection, and hence the choice entropy the splice
    // injects.
    let rank = rng.random_range(0..choice_count);
    let pick = refs
        .iter()
        .enumerate()
        .filter(|(_, r)| eligible(r))
        .nth(rank)
        .map(|(i, _)| i)
        .expect("rank is within the counted eligible set");
    Some((pick, choice_count))
}

/// Allocation-heavy selector retained only as a determinism oracle for the
/// scan-based production implementation above.
#[cfg(test)]
fn choose_ref_reference<R>(
    refs: &[R],
    window_len: usize,
    mode: DbMode,
    pay_random: bool,
    rng: &mut impl Rng,
) -> Option<(usize, usize)>
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
        DbMode::Compressing => {
            let min = pool
                .iter()
                .map(|&i| len_of(i))
                .filter(|&l| l <= window_len)
                .min()?;
            pool.into_iter().filter(|&i| len_of(i) == min).collect()
        }
        DbMode::SizeAgnostic => pool,
        DbMode::MinGrow => {
            let min = pool.iter().map(|&i| len_of(i)).min()?;
            pool.into_iter().filter(|&i| len_of(i) == min).collect()
        }
        DbMode::Mix => {
            let free: Vec<usize> = pool
                .iter()
                .copied()
                .filter(|&i| len_of(i) <= window_len)
                .collect();
            if free.is_empty() {
                if pay_random {
                    pool
                } else {
                    let min = pool.iter().map(|&i| len_of(i)).min()?;
                    pool.into_iter().filter(|&i| len_of(i) == min).collect()
                }
            } else {
                free
            }
        }
    };
    if eligible.is_empty() {
        return None;
    }
    Some((
        eligible[rng.random_range(0..eligible.len())],
        eligible.len(),
    ))
}

/// Just enough of a candidate to select on, so selection never decodes.
trait CandLen {
    fn gate_count(&self) -> usize;
    fn curated(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::xgate::eval_lanes;
    use rand::RngCore;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use std::collections::HashMap;

    #[derive(Clone, Debug)]
    struct SelectorCandidate {
        gates: usize,
        curated: bool,
    }

    impl CandLen for SelectorCandidate {
        fn gate_count(&self) -> usize {
            self.gates
        }

        fn curated(&self) -> bool {
            self.curated
        }
    }

    #[test]
    fn scan_selector_matches_reference_selection_and_rng_state() {
        let modes = [
            DbMode::Compressing,
            DbMode::SizeAgnostic,
            DbMode::MinGrow,
            DbMode::Mix,
        ];

        // Exercise empty/all-regular/all-curated/mixed catalogues, repeated
        // lengths, every mode/pay_random combination, and the exact ordering
        // produced by both selected and unrelated swap-removals.
        for mode in modes {
            for pay_random in [false, true] {
                for case in 0..128u64 {
                    let initial_len = (case as usize * 17 + 3) % 37;
                    let mut refs: Vec<SelectorCandidate> = (0..initial_len)
                        .map(|i| SelectorCandidate {
                            gates: (i * 11 + case as usize * 7) % 13,
                            curated: match case % 4 {
                                0 => false,
                                1 => true,
                                2 => i % 2 == 0,
                                _ => (i * 5 + case as usize) % 7 < 3,
                            },
                        })
                        .collect();
                    let window_len = (case as usize * 19 + 1) % 12;
                    let seed = case
                        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                        .wrapping_add((mode as u64) << 9)
                        .wrapping_add(u64::from(pay_random));
                    let mut reference_rng = StdRng::seed_from_u64(seed);
                    let mut scan_rng = StdRng::seed_from_u64(seed);

                    for step in 0..initial_len + 2 {
                        let expected = choose_ref_reference(
                            &refs,
                            window_len,
                            mode,
                            pay_random,
                            &mut reference_rng,
                        );
                        let actual = choose_ref(&refs, window_len, mode, pay_random, &mut scan_rng);
                        assert_eq!(
                            actual, expected,
                            "selection changed: mode={mode:?} pay_random={pay_random} \
                             case={case} step={step} window_len={window_len} refs={refs:?}"
                        );

                        let mut reference_probe = reference_rng.clone();
                        let mut scan_probe = scan_rng.clone();
                        assert_eq!(
                            scan_probe.next_u64(),
                            reference_probe.next_u64(),
                            "RNG state changed: mode={mode:?} pay_random={pay_random} \
                             case={case} step={step}"
                        );

                        if let Some((pick, _)) = actual {
                            refs.swap_remove(pick);
                        } else if !refs.is_empty() {
                            // No candidate qualified (normally Compressing
                            // with only growing entries); still exercise the
                            // next selector call after an external removal.
                            let remove = (case as usize + step * 3) % refs.len();
                            refs.swap_remove(remove);
                        }
                        if step % 3 == 1 && !refs.is_empty() {
                            let remove = (case as usize * 5 + step) % refs.len();
                            refs.swap_remove(remove);
                        }
                    }
                }
            }
        }
    }

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
    fn is_reorder_is_multiset_equality() {
        let a = db_g57_to_xgate([0, 1, 2]);
        let b = db_g57_to_xgate([3, 4, 5]);
        let c = db_g57_to_xgate([0, 1, 3]);
        assert!(is_reorder(&[a.clone(), b.clone()], &[b.clone(), a.clone()]));
        assert!(is_reorder(&[a.clone(), b.clone()], &[a.clone(), b.clone()]));
        assert!(!is_reorder(
            &[a.clone(), b.clone()],
            &[a.clone(), c.clone()]
        ));
        assert!(!is_reorder(&[a.clone()], &[a.clone(), b.clone()]));
        // Duplicates must pair off one-to-one.
        assert!(!is_reorder(&[a.clone(), a.clone()], &[a.clone(), b]));
        assert!(is_reorder(&[a.clone(), a.clone()], &[a.clone(), a]));
    }

    // The pair-window reorder ban: for a commuting 2-gate window whose stored
    // spellings are exactly its two orderings, the identity guard kills one
    // and ban_reorder must kill the other, so nothing is chosen — the
    // situation that forces a real pair splice onto a longer spelling. With
    // the ban off, the reordered spelling is (deliberately) still admissible.
    // A commuting involution pair is its own inverse, so both canonical
    // directions share one key and the single stored value serves either
    // probe direction.
    #[test]
    fn ban_reorder_refuses_the_permuted_pair() {
        let g1 = db_g57_to_xgate([0, 1, 2]);
        let g2 = db_g57_to_xgate([3, 4, 5]);
        let window = vec![g1.clone(), g2.clone()];

        let legacy = CircuitSeq {
            gates: vec![[0, 1, 2], [3, 4, 5]],
        };
        let (polys, order, _used) = legacy.canonicalize_polys_single(false);
        let key = xxh3_128(&polys_repr_blob(&polys)).to_le_bytes();
        let mut fwd = legacy.clone();
        fwd.rewire(&order.invert(), fwd.max_wire() as usize + 1);
        let mut rev = fwd.clone();
        rev.gates.reverse();
        let store = HashMap::from([(key, encode_value(&[fwd, rev]))]);

        for ban in [false, true] {
            let mut rng = StdRng::seed_from_u64(7);
            let res = db_replace_with(
                &window,
                8,
                XPolyBudget::default(),
                DbMode::Mix,
                DegreeGuard::OFF,
                false,
                true,
                false,
                ban,
                &mut rng,
                |k, cur| if cur { None } else { store.get(k).cloned() },
            );
            assert_eq!(res.match_count, 2);
            if ban {
                assert!(res.chosen.is_none(), "both orderings must be refused");
                assert_eq!(res.identity_skipped, 1);
                assert_eq!(res.permutation_skipped, 1);
            } else {
                let repl = res
                    .chosen
                    .expect("without the ban the reorder is admissible");
                assert_eq!(
                    repl,
                    vec![g2.clone(), g1.clone()],
                    "the non-identity ordering wins"
                );
                assert_eq!(res.permutation_skipped, 0);
                assert!(
                    exhaustively_equal(&window, &repl, 8),
                    "the reorder computes the same function (the pair commutes)"
                );
            }
        }
    }

    // regular_fallback=false must SUPPRESS the regular stage while the
    // curated cascade is live -- that is the primitive the two-pass
    // (--curated-exhaust) descent is built on. Same window, same store, the
    // only difference is the flag.
    // The degree filter is now EXACT, read off the ANF, not a randomized probe
    // over affine subspaces. The property that buys: it cannot flake. The old
    // probe was one-sided -- it could certify "over" but silently miss it on an
    // unlucky draw -- so the same window could be skipped or not depending on
    // the rng. Run one over-degree window across many seeds and demand the same
    // answer every time.
    #[test]
    fn degree_filter_is_exact_and_seed_independent() {
        // Two chained g57s on shared wires: degree climbs above a cap of 2.
        let window = vec![
            db_g57_to_xgate([0, 1, 2]),
            db_g57_to_xgate([2, 3, 4]),
            db_g57_to_xgate([4, 5, 6]),
        ];
        let guard = DegreeGuard {
            max_degree: 2,
            probes: 6,
        };
        let mut skipped = 0;
        let mut looked_up = 0;
        for seed in 0..64u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let res = db_replace_with(
                &window,
                16,
                XPolyBudget::default(),
                DbMode::SizeAgnostic,
                guard,
                false,
                true,
                false,
                false,
                &mut rng,
                |_, _| {
                    looked_up += 1;
                    None
                },
            );
            if res.degree_skipped {
                skipped += 1;
            }
        }
        assert!(
            skipped == 0 || skipped == 64,
            "degree verdict flipped with the seed: skipped on {skipped}/64 -- the filter is not exact"
        );
        assert_eq!(
            skipped, 64,
            "this window is above the cap and must always be skipped"
        );
        // The FORWARD direction is over the cap, but a permutation's inverse
        // can be lower-degree, and this window's reverse is: stage B rightly
        // canonicalizes and probes it. The old probe behaved the same way --
        // it only short-circuited when BOTH directions were certified over --
        // so at most one key per run may reach the store, never two.
        assert!(
            looked_up <= 64,
            "more than one key per run reached the store: {looked_up} over 64 runs"
        );

        // And the same window under a cap that admits it must never be skipped.
        let guard_ok = DegreeGuard {
            max_degree: 9,
            probes: 6,
        };
        for seed in 0..16u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let res = db_replace_with(
                &window,
                16,
                XPolyBudget::default(),
                DbMode::SizeAgnostic,
                guard_ok,
                false,
                true,
                false,
                false,
                &mut rng,
                |_, _| None,
            );
            assert!(
                !res.degree_skipped,
                "in-range window skipped at seed {seed}"
            );
        }
    }

    #[test]
    fn regular_fallback_false_suppresses_the_regular_stage() {
        let g = db_g57_to_xgate([0, 1, 2]);
        let pad = XGate::conj(0, [(1, true), (2, false)]).unwrap();
        let window = vec![pad.clone(), pad, g];

        let legacy = CircuitSeq {
            gates: vec![[0, 1, 2]],
        };
        let (key, value) = store_friend(&legacy);
        // REGULAR store holds the friend; curated holds nothing.
        let store = HashMap::from([(key, value)]);
        let lookup = |k: &[u8; 16], cur: bool| if cur { None } else { store.get(k).cloned() };

        // Cascade armed, fallback ALLOWED: curated misses, regular answers.
        let mut rng = StdRng::seed_from_u64(1);
        let with_fb = db_replace_with(
            &window,
            8,
            XPolyBudget::default(),
            DbMode::Mix,
            DegreeGuard::OFF,
            true,
            true,
            false,
            false,
            &mut rng,
            lookup,
        );
        assert_eq!(with_fb.match_count, 1, "regular stage should have answered");

        // Cascade armed, fallback SUPPRESSED: curated misses and nothing else runs.
        let mut rng = StdRng::seed_from_u64(1);
        let no_fb = db_replace_with(
            &window,
            8,
            XPolyBudget::default(),
            DbMode::Mix,
            DegreeGuard::OFF,
            true,
            false,
            false,
            false,
            &mut rng,
            lookup,
        );
        assert_eq!(no_fb.match_count, 0, "regular stage must be suppressed");
        assert!(no_fb.chosen.is_none());

        // UNARMED: there is no curated stage to fall back FROM, so the flag
        // must not be able to switch the regular store off.
        let mut rng = StdRng::seed_from_u64(1);
        let unarmed = db_replace_with(
            &window,
            8,
            XPolyBudget::default(),
            DbMode::Mix,
            DegreeGuard::OFF,
            false,
            false,
            false,
            false,
            &mut rng,
            lookup,
        );
        assert_eq!(
            unarmed.match_count, 1,
            "unarmed processes must still reach regular"
        );

        // COMPRESSION is regular-only by contract; the flag must not touch it.
        let mut rng = StdRng::seed_from_u64(1);
        let comp = db_replace_with(
            &window,
            8,
            XPolyBudget::default(),
            DbMode::Compressing,
            DegreeGuard::OFF,
            // The mode rule now lives in `db_replace`; apply it as that caller
            // would. Its own contract is asserted directly below.
            curated_armed_for(true, DbMode::Compressing, false),
            false,
            false,
            false,
            &mut rng,
            lookup,
        );
        assert_eq!(
            comp.match_count, 1,
            "compression must stay regular-only regardless"
        );
        assert!(
            !curated_armed_for(true, DbMode::Compressing, false),
            "COMP must not arm curated by default"
        );
        assert!(
            curated_armed_for(true, DbMode::Compressing, true),
            "--curated-in-comp must be able to arm it"
        );
        assert!(
            curated_armed_for(true, DbMode::Mix, false),
            "expansion arms curated without any override"
        );
        assert!(
            !curated_armed_for(false, DbMode::Mix, true),
            "the override must not arm curated when curated itself is off"
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
            true,
            false,
            false,
            &mut rng,
            |_, cur| {
                probes.push(cur);
                if cur { Some(hit_value.clone()) } else { None }
            },
        );
        assert_eq!(
            probes,
            vec![true],
            "curated hit must suppress the regular probe"
        );

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
            true,
            false,
            false,
            &mut rng,
            |_, cur| {
                probes.push(cur);
                None
            },
        );
        assert!(probes.first() == Some(&true), "curated probed first");
        assert!(
            probes.iter().skip(1).all(|&c| !c),
            "fallback probes are regular"
        );
        assert!(probes.len() >= 2, "regular fallback must actually fire");

        // Compression: never curated. The rule lives in `db_replace` now, so
        // apply it the way that caller does rather than expecting the
        // mechanism to second-guess its argument.
        let mut probes: Vec<bool> = Vec::new();
        let mut rng = StdRng::seed_from_u64(7);
        let _ = db_replace_with(
            &window,
            16,
            XPolyBudget::default(),
            DbMode::Compressing,
            DegreeGuard::OFF,
            curated_armed_for(true, DbMode::Compressing, false),
            true,
            false,
            false,
            &mut rng,
            |_, cur| {
                probes.push(cur);
                None
            },
        );
        assert!(
            !probes.is_empty() && probes.iter().all(|&c| !c),
            "COMP is regular-only"
        );
    }

    // The polynomial verifier must agree with the exhaustive one wherever the
    // exhaustive one can run. That agreement is the whole warrant for using it
    // ABOVE the 24-wire ceiling, where nothing can cross-check it.
    #[test]
    fn poly_verifier_agrees_with_the_exhaustive_one() {
        let mut rng = StdRng::seed_from_u64(31);
        let budget = XPolyBudget::default();
        let (mut same, mut diff) = (0, 0);
        for _ in 0..400 {
            let n = 6;
            let k = rng.random_range(1..=4);
            let mk = |rng: &mut StdRng| -> Vec<XGate> {
                (0..k)
                    .map(|_| {
                        let t = rng.random_range(0..n) as u16;
                        let mut c: Vec<(u16, bool)> = Vec::new();
                        for w in 0..n as u16 {
                            if w != t && rng.random_bool(0.35) {
                                c.push((w, rng.random_bool(0.5)));
                            }
                        }
                        XGate {
                            target: t,
                            comp: rng.random_bool(0.5),
                            ctrls: c.into(),
                        }
                    })
                    .collect()
            };
            let a = mk(&mut rng);
            // Half the trials compare a sequence with itself (must be equal),
            // half with an independent one (usually unequal).
            let b = if rng.random_bool(0.5) {
                a.clone()
            } else {
                mk(&mut rng)
            };
            let exhaustive = crate::engine::rules::verify_rewrite(&a, &b);
            let poly = polys_equivalent(&a, &b, budget).expect("6 wires is decidable");
            assert_eq!(
                exhaustive, poly,
                "verifiers disagree on {a:?} vs {b:?}: exhaustive={exhaustive} poly={poly}"
            );
            if exhaustive { same += 1 } else { diff += 1 }
        }
        assert!(
            same > 20 && diff > 20,
            "trial mix was degenerate: {same} equal, {diff} unequal"
        );
    }

    // A window far past the exhaustive ceiling still verifies, and still
    // distinguishes: the point of the whole exercise.
    #[test]
    fn poly_verifier_handles_windows_past_the_exhaustive_cap() {
        // 30 wires: 2^30 evaluations is out of reach for verify_rewrite, which
        // refuses above 24.
        let gates: Vec<XGate> = (0..10)
            .map(|i| {
                XGate::conj(
                    i as u16,
                    [((i + 10) as u16, true), ((i + 20) as u16, false)],
                )
                .expect("distinct pins")
            })
            .collect();
        let budget = XPolyBudget::default();
        assert_eq!(polys_equivalent(&gates, &gates, budget), Some(true));
        let mut changed = gates.clone();
        changed[3].comp = !changed[3].comp;
        assert_eq!(polys_equivalent(&gates, &changed, budget), Some(false));
        // Reordering two gates that share no wires is function-preserving.
        let mut swapped = gates.clone();
        swapped.swap(0, 1);
        assert_eq!(polys_equivalent(&gates, &swapped, budget), Some(true));
    }

    #[test]
    fn degree_guard_skips_high_degree_window_without_a_lookup() {
        use crate::circuit::xgate::Lits;
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
            false,
            &mut rng,
            |k, cur| if cur { None } else { store.get(k).cloned() },
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
            false,
            &mut rng,
            |k, cur| if cur { None } else { store.get(k).cloned() },
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
            false,
            &mut rng,
            |k, cur| if cur { None } else { store.get(k).cloned() },
        );
        let repl = mg
            .chosen
            .expect("min-grow accepts the shortest growing friend");
        assert_eq!(repl.len(), 3);
        assert!(exhaustively_equal(&window, &repl, 8));
    }
}
