//! Frozen-store lookup for a single heterogeneous [`XGate`] window, used by the
//! circuit_mixer DB contraction move (see [`crate::engine::mixer`]).
//!
//! A convex/contiguous window of arbitrary-width, mixed-polarity XGates is keyed
//! by its exact function polynomial ([`crate::canonicalization::xgate`]) — identical to
//! the G57 key path, so one frozen store serves both — and looked up in
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

use crate::canonicalization::gate_order::xgate_canonical;
use crate::canonicalization::polys_repr_blob;
use crate::canonicalization::xgate::{
    CanonicalXPolys, XPolyBudget, XPolyError, canonicalize_xgates_single,
    canonicalize_xgates_single_capped, xgates_to_polynomial,
};
use crate::circuit::xgate::XGate;
use crate::circuit::{CircuitSeq, Permutation};
use crate::database::frozen::{FrozenDb, QcLookupLimit};
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

/// Candidates the identity guard rejected only because they equal the window
/// in canonical gate order (a commuting re-spelling), not verbatim.
pub static REORDER_IDENTITY_SKIPS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

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
/// tagged with which canonical direction matched. No identity filtering, no mode selection, no scratch-wire
/// refusal -- this is for asking whether the STORE's answers are equivalent to
/// the window at all, which the normal path only ever answers for the one
/// candidate it happens to pick.
pub fn db_probe(
    window: &[XGate],
    num_wires: usize,
    db: &FrozenDb,
    budget: XPolyBudget,
    rng: &mut impl Rng,
) -> Vec<(Vec<XGate>, bool)> {
    let mut out = Vec::new();
    for reversed in [false, true] {
        let Ok(canonical) = canonicalize_xgates_single(window, reversed, budget) else {
            continue;
        };
        let key = key_of(&canonical);
        let Some(value) = db.get_regular(&key) else {
            continue;
        };
        for friend in decode_value(&value) {
            if let Some(g) = friend_to_xgates(
                friend,
                reversed,
                &canonical.order,
                &canonical.used_wires,
                num_wires,
                rng,
            ) {
                out.push((g, reversed));
            }
        }
    }
    out
}

/// Work and size caps for the independent gate-wise quality-control lookup.
/// All caps are hard limits; zero does not disable a limit.
#[derive(Clone, Copy, Debug)]
pub struct QcCandidateLimits {
    /// Maximum encoded candidate records examined across both directions, including rejected and duplicate records.
    pub max_candidates: usize,
    pub max_gates: usize,
    pub max_support: usize,
}

/// Bound compressed bucket reads and decoded traversal work for QC only.
/// The ordinary replacement lookup keeps its existing store/cache policy.
pub const QC_MAX_BUCKET_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Debug)]
pub struct QcCandidate {
    /// Every returned candidate has passed exact polynomial equivalence.
    pub gates: Vec<XGate>,
    pub reversed: bool,
}

/// Diagnostics deliberately distinguish a missing entry, incomplete search,
/// and entries that only contain unsuitable replacements. In particular,
/// `truncated` must never be reported as proof that all equivalents are hot.
#[derive(Clone, Debug, Default)]
pub struct QcCandidates {
    pub candidates: Vec<QcCandidate>,
    pub examined: usize,
    pub entries_found: usize,
    pub lookups: usize,
    pub truncated: bool,
    /// `(reversed, error)` for each direction that could not be keyed.
    pub canonicalization_errors: Vec<(bool, XPolyError)>,
    /// Bounded store reads that could not be completed.
    pub lookup_limits: Vec<QcLookupLimit>,
    pub malformed_values: usize,
    pub unplaceable: usize,
    pub identity_skipped: usize,
    pub duplicate_skipped: usize,
    pub size_skipped: usize,
    pub non_equivalent: usize,
    pub equivalence_budget_exceeded: usize,
}

/// Enumerate a bounded QC candidate set without changing normal DB selection
/// or its sampling distribution. Both canonical directions are considered,
/// round-robin. Identity spellings are skipped and every returned circuit is
/// proved equivalent before exposure.
pub fn qc_candidates(
    window: &[XGate],
    num_wires: usize,
    db: &FrozenDb,
    budget: XPolyBudget,
    limits: QcCandidateLimits,
    rng: &mut impl Rng,
) -> QcCandidates {
    let mut lookup_limits = Vec::new();
    let mut result = qc_candidates_with_value(window, num_wires, budget, limits, rng, |key| {
        // One extra record makes the global enumeration's truncation signal
        // exact even when this value alone consumes the entire cap.
        let cap = limits.max_candidates.saturating_add(1);
        match db.get_regular_qc(key, cap, QC_MAX_BUCKET_BYTES) {
            Ok(value) => value,
            Err(reason) => {
                lookup_limits.push(reason);
                None
            }
        }
    });
    result.lookup_limits = lookup_limits;
    result
}

/// In-memory store seam for QC experiments and deterministic fixtures.
pub fn qc_candidates_with<F>(
    window: &[XGate],
    num_wires: usize,
    budget: XPolyBudget,
    limits: QcCandidateLimits,
    rng: &mut impl Rng,
    lookup: F,
) -> QcCandidates
where
    F: FnMut(&[u8; 16]) -> Option<Vec<u8>>,
{
    qc_candidates_with_value(window, num_wires, budget, limits, rng, lookup)
}

fn qc_candidates_with_value<F, V>(
    window: &[XGate],
    num_wires: usize,
    budget: XPolyBudget,
    limits: QcCandidateLimits,
    rng: &mut impl Rng,
    mut lookup: F,
) -> QcCandidates
where
    F: FnMut(&[u8; 16]) -> Option<V>,
    V: AsRef<[u8]>,
{
    struct Cursor<V> {
        value: V,
        position: usize,
        canonical: CanonicalXPolys,
        reversed: bool,
    }
    let mut result = QcCandidates::default();
    if limits.max_candidates == 0 {
        result.truncated = true;
        return result;
    }
    let mut cursors = Vec::with_capacity(2);
    for reversed in [false, true] {
        let canonical = match canonicalize_xgates_single(window, reversed, budget) {
            Ok(canonical) => canonical,
            Err(error) => {
                result.canonicalization_errors.push((reversed, error));
                continue;
            }
        };
        let key = key_of(&canonical);
        result.lookups += 1;
        if let Some(value) = lookup(&key) {
            result.entries_found += 1;
            cursors.push(Cursor {
                value,
                position: 0,
                canonical,
                reversed,
            });
        }
    }
    // Identity and duplicates are judged in canonical gate order: a candidate
    // that differs from the window (or from an earlier candidate) only by
    // commuting swaps is the same circuit.
    let window_canonical = xgate_canonical(window);
    let mut seen = std::collections::HashSet::new();
    loop {
        let mut advanced = false;
        for cursor in &mut cursors {
            let value = cursor.value.as_ref();
            if cursor.position == value.len() {
                continue;
            }
            if result.examined == limits.max_candidates {
                result.truncated = true;
                return result;
            }
            advanced = true;
            result.examined += 1;
            let len = value[cursor.position] as usize;
            cursor.position += 1;
            if len % 3 != 0 || len > value.len() - cursor.position {
                result.malformed_values += 1;
                cursor.position = value.len();
                continue;
            }
            let blob = &value[cursor.position..cursor.position + len];
            cursor.position += len;
            if len / 3 > limits.max_gates {
                result.size_skipped += 1;
                continue;
            }
            // Decode one bounded record, never the entire store value into a
            // vector of circuits as the unbounded diagnostic db_probe does.
            let friend = CircuitSeq::from_blob(blob);
            let Some(gates) = friend_to_xgates(
                friend,
                cursor.reversed,
                &cursor.canonical.order,
                &cursor.canonical.used_wires,
                num_wires,
                rng,
            ) else {
                result.unplaceable += 1;
                continue;
            };
            let canonical_gates = xgate_canonical(&gates);
            if canonical_gates == window_canonical {
                result.identity_skipped += 1;
                continue;
            }
            if !seen.insert(canonical_gates) {
                result.duplicate_skipped += 1;
                continue;
            }
            let mut support: Vec<u16> = gates
                .iter()
                .flat_map(|g| std::iter::once(g.target).chain(g.ctrls.iter().map(|&(w, _)| w)))
                .collect();
            support.sort_unstable();
            support.dedup();
            if support.len() > limits.max_support {
                result.size_skipped += 1;
                continue;
            }
            match polys_equivalent(window, &gates, budget) {
                Some(true) => result.candidates.push(QcCandidate {
                    gates,
                    reversed: cursor.reversed,
                }),
                Some(false) => result.non_equivalent += 1,
                None => result.equivalence_budget_exceeded += 1,
            }
        }
        if !advanced {
            return result;
        }
    }
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
    /// Stable re-encode: uniform over every equivalent within ONE gate of the
    /// window (w-1, w, w+1); a window with no such spelling is a miss. The
    /// point is maximum draw entropy subject to keeping the size where it is,
    /// so that FUTURE windows keep re-encodable material to draw on — neither
    /// the compressive drift of Mix's free branch (which prefers nothing and
    /// so shrinks whenever shorter spellings exist) nor unpaid growth.
    Stable,
    /// Stable without the shrink option: uniform over equivalents of size w
    /// or w+1 only. Never loses a gate by construction; the availability bias
    /// that drove Stable downhill (the store thins above its 7-gate density
    /// peak, so w-1 spellings outnumber w+1) becomes a slow upward ratchet
    /// that stalls where windows run out of larger spellings.
    StableGrow,
    /// Flow-balanced stable: behaves as Stable while the walk's shrink ledger
    /// has budget (gates added by +1 splices cover the gates removed, within
    /// a small slack) and as StableGrow once it is exhausted. The walk maps
    /// this to one of the two BEFORE selection (mix.rs db_attempt), so
    /// choose_ref treats it like Stable if it ever sees it. Net DB drift is
    /// bounded below by -slack at any size.
    StableLedger,
    /// Exact-size only: uniform over equivalents with len == window (the
    /// same-size conversion probe); anything else is a miss.
    Same,
    /// SIZE-BALANCED band draw: the full band is offered while the walk's
    /// size ledger is within slack, and the size choice is skewed corrective
    /// once it drifts — so size VARIABILITY per splice is preserved while the
    /// total is conserved in expectation. Mapped to BandShrink/BandGrow/
    /// SizeAgnostic by the walk before selection (mix.rs db_attempt); if it
    /// reaches choose_ref untranslated it reads as the unconstrained draw.
    BandLedger,
    /// Corrective arm: strictly SHORTER equivalents when any exist, else
    /// same-size — never grows. (Still several sizes wide inside a band.)
    BandShrink,
    /// Corrective arm: strictly LONGER equivalents when any exist, else
    /// same-size — never shrinks.
    BandGrow,
}

impl DbMode {
    pub fn parse(s: &str) -> Option<DbMode> {
        match s {
            "mix" => Some(DbMode::Mix),
            "comp" => Some(DbMode::Compressing),
            "any" => Some(DbMode::SizeAgnostic),
            "stable" => Some(DbMode::Stable),
            "stable-grow" => Some(DbMode::StableGrow),
            "stable-ledger" => Some(DbMode::StableLedger),
            "same" => Some(DbMode::Same),
            "band-ledger" => Some(DbMode::BandLedger),
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

/// Resolved replacement controls; ordinary defaults select the minimum key
/// direction and impose no extra incoming-length or canonical-search limits.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReplacementOptions {
    pub direction: crate::database::lookup_cache::MinDirLookup,
    pub incoming_length_band: Option<(usize, usize)>,
    pub canonicalization: crate::canonicalization::CanonicalizationOptions,
}

fn canonicalize_replacement(
    window: &[XGate],
    reversed: bool,
    budget: XPolyBudget,
    max_degree: usize,
    options: Option<&ReplacementOptions>,
) -> Result<CanonicalXPolys, XPolyError> {
    match options {
        None => canonicalize_xgates_single_capped(window, reversed, budget, max_degree),
        Some(options) => crate::canonicalization::xgate::canonicalize_xgates_single_with_options(
            window,
            reversed,
            &crate::canonicalization::xgate::XGateCanonicalizationOptions {
                budget,
                max_degree,
                canonicalization: options.canonicalization,
            },
        ),
    }
}

/// Look up `window` in the frozen store and select a replacement per `mode`.
/// `num_wires` is the full circuit wire count (for scratch-wire assignment).
/// `guard` cheaply skips over-degree directions before canonicalization.
///
/// Correctness is NOT assumed from the DB: the caller may still verify the
/// returned gates are equivalent to `window` before splicing.
#[allow(clippy::too_many_arguments)]
pub fn db_replace(
    window: &[XGate],
    num_wires: usize,
    db: &FrozenDb,
    budget: XPolyBudget,
    mode: DbMode,
    guard: DegreeGuard,
    pay_random: bool,
    ban_reorder: bool,
    rng: &mut impl Rng,
) -> DbResult {
    db_replace_with_value(
        window,
        num_wires,
        budget,
        mode,
        guard,
        pay_random,
        ban_reorder,
        rng,
        |key| {
            // Every probe goes through the exact process-wide lookup cache
            // in database::lookup_cache: the store is immutable, so cached hits
            // and misses are byte-identical to raw probes.
            crate::database::lookup_cache::cached_db_get(db, key)
        },
    )
}

/// Testable core: `lookup` stands in for the frozen store.
#[allow(clippy::too_many_arguments)]
pub fn db_replace_with<F>(
    window: &[XGate],
    num_wires: usize,
    budget: XPolyBudget,
    mode: DbMode,
    guard: DegreeGuard,
    pay_random: bool,
    ban_reorder: bool,
    rng: &mut impl Rng,
    lookup: F,
) -> DbResult
where
    F: FnMut(&[u8; 16]) -> Option<Vec<u8>>,
{
    db_replace_with_value(
        window,
        num_wires,
        budget,
        mode,
        guard,
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
    pay_random: bool,
    ban_reorder: bool,
    rng: &mut impl Rng,
    lookup: F,
) -> DbResult
where
    F: FnMut(&[u8; 16]) -> Option<V>,
    V: AsRef<[u8]>,
{
    db_replace_with_value_options(
        window,
        num_wires,
        budget,
        mode,
        guard,
        pay_random,
        ban_reorder,
        rng,
        lookup,
        None,
    )
}

/// Explicit lookup/selection boundary. Supply a raw store or a caller-owned
/// LookupCache closure; neither this core nor canonicalization reads the environment.
#[allow(clippy::too_many_arguments)]
pub fn db_replace_with_options<F, V>(
    window: &[XGate],
    num_wires: usize,
    budget: XPolyBudget,
    mode: DbMode,
    guard: DegreeGuard,
    pay_random: bool,
    ban_reorder: bool,
    rng: &mut impl Rng,
    lookup: F,
    options: &ReplacementOptions,
) -> DbResult
where
    F: FnMut(&[u8; 16]) -> Option<V>,
    V: AsRef<[u8]>,
{
    db_replace_with_value_options(
        window,
        num_wires,
        budget,
        mode,
        guard,
        pay_random,
        ban_reorder,
        rng,
        lookup,
        Some(options),
    )
}

/// Explicit frozen-store counterpart to db_replace. Uses raw lookups; callers
/// wanting a cache can use db_replace_with_options with their LookupCache.
#[allow(clippy::too_many_arguments)]
pub fn db_replace_options(
    window: &[XGate],
    num_wires: usize,
    db: &FrozenDb,
    budget: XPolyBudget,
    mode: DbMode,
    guard: DegreeGuard,
    pay_random: bool,
    ban_reorder: bool,
    rng: &mut impl Rng,
    options: &ReplacementOptions,
) -> DbResult {
    db_replace_with_options(
        window,
        num_wires,
        budget,
        mode,
        guard,
        pay_random,
        ban_reorder,
        rng,
        |key| db.get_regular(key),
        options,
    )
}

#[allow(clippy::too_many_arguments)]
fn db_replace_with_value_options<F, V>(
    window: &[XGate],
    num_wires: usize,
    budget: XPolyBudget,
    mode: DbMode,
    guard: DegreeGuard,
    pay_random: bool,
    ban_reorder: bool,
    rng: &mut impl Rng,
    mut lookup: F,
    options: Option<&ReplacementOptions>,
) -> DbResult
where
    F: FnMut(&[u8; 16]) -> Option<V>,
    V: AsRef<[u8]>,
{
    let miss = |degree_skipped| DbResult {
        match_count: 0,
        chosen: None,
        degree_skipped,
        identity_skipped: 0,
        permutation_skipped: 0,
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

    // Canonicalize the forward direction, then the reverse (a window shorter
    // under its inverse still keys the way the builder recorded it).
    let mut directions: Vec<(bool, CanonicalXPolys)> = Vec::with_capacity(2);
    match canonicalize_replacement(window, false, budget, guard.max_degree, options) {
        Ok(c) => directions.push((false, c)),
        Err(XPolyError::DegreeExceeded { .. }) => degree_skipped = true,
        Err(_) => {}
    }

    // Every placeable equivalent circuit across both distinct keys, ANY length.
    // Catalogue candidates WITHOUT decoding them. The stored value is a flat
    // [len][len bytes] sequence, so gate counts -- everything the selection
    // rules need -- can be read by walking offsets. friend_to_xgates builds an
    // occupancy vector and a shuffled availability list PER CANDIDATE, so only
    // the chosen candidate is decoded.
    struct CandRef {
        vi: usize,
        off: usize,
        nbytes: usize,
        gates: usize,
        dir_ix: usize,
    }
    impl CandLen for CandRef {
        fn gate_count(&self) -> usize {
            self.gates
        }
    }
    fn catalogue<V: AsRef<[u8]>>(
        value: V,
        dir_ix: usize,
        values: &mut Vec<V>,
        refs: &mut Vec<CandRef>,
        min_match_len: &mut Option<usize>,
    ) {
        let vi = values.len();
        let bytes = value.as_ref();
        // Each candidate costs at least 4 bytes here (a length byte plus a
        // 3-byte gate), so this is an estimate, not a bound -- capacity only,
        // so an over- or under-shoot changes nothing but allocator work.
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
                dir_ix,
            });
            *min_match_len = Some(min_match_len.map_or(gates, |min| min.min(gates)));
            pos += len;
        }
        values.push(value);
    }

    let mut values: Vec<V> = Vec::new();
    let mut refs: Vec<CandRef> = Vec::new();
    let mut min_match_len = None;
    let mut identity_skipped = 0usize;
    let mut permutation_skipped = 0usize;
    match canonicalize_replacement(window, true, budget, guard.max_degree, options) {
        Ok(c) => directions.push((true, c)),
        Err(XPolyError::DegreeExceeded { .. }) => degree_skipped = true,
        Err(_) => {}
    }
    // The regular store is keyed by min(canon_fwd, canon_rev). When both
    // directions compose, the default Min mode probes only the min direction — the
    // non-min key can only exist if it equals the min key, so the candidate
    // set is unchanged. Legacy restores the historical probe-both cascade;
    // Validate probes the other direction on a miss and counts violations.
    use crate::database::lookup_cache::{
        MIN_DIR_VALIDATE_PROBES, MIN_DIR_VIOLATIONS, MinDirLookup, min_dir_lookup_mode,
    };
    use std::sync::atomic::Ordering;
    let fwd_idx = directions.iter().position(|(reversed, _)| !reversed);
    let rev_idx = directions.iter().position(|(reversed, _)| *reversed);
    let min_mode = match (fwd_idx, rev_idx) {
        (Some(_), Some(_)) => {
            options.map_or_else(min_dir_lookup_mode, |options| options.direction)
        }
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
                if let Some(value) = lookup(&key) {
                    catalogue(value, dir_ix, &mut values, &mut refs, &mut min_match_len);
                }
            }
        }
        mode => {
            let fi = fwd_idx.expect("min mode requires both directions");
            let ri = rev_idx.expect("min mode requires both directions");
            let rev_is_min = directions[ri].1.polys < directions[fi].1.polys;
            let (min_idx, alt_idx) = if rev_is_min { (ri, fi) } else { (fi, ri) };
            let min_key = key_of(&directions[min_idx].1);
            if let Some(value) = lookup(&min_key) {
                catalogue(value, min_idx, &mut values, &mut refs, &mut min_match_len);
            } else if mode == MinDirLookup::Validate {
                let alt_key = key_of(&directions[alt_idx].1);
                if alt_key != min_key {
                    MIN_DIR_VALIDATE_PROBES.fetch_add(1, Ordering::Relaxed);
                    if let Some(value) = lookup(&alt_key) {
                        MIN_DIR_VIOLATIONS.fetch_add(1, Ordering::Relaxed);
                        eprintln!(
                            "[min-dir-violation] circuit_mixer db_replace: non-min canonical key present while min key absent (window={})",
                            window.len()
                        );
                        catalogue(value, alt_idx, &mut values, &mut refs, &mut min_match_len);
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
    let mut choice_count = 0usize;
    let mut window_canonical: Option<Vec<XGate>> = None;
    while !refs.is_empty() {
        let Some((pick, n_eligible)) =
            choose_ref_with_options(&refs, window_len, mode, pay_random, rng, options)
        else {
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
        // Identity guard, in canonical gate order: the store holds candidates
        // canonically ordered, so a candidate that is the window up to
        // commuting swaps is the window itself, at every window length.
        let window_canonical = window_canonical.get_or_insert_with(|| xgate_canonical(window));
        if gates == window {
            identity_skipped += 1;
            continue;
        }
        if xgate_canonical(&gates) == *window_canonical {
            identity_skipped += 1;
            REORDER_IDENTITY_SKIPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            continue;
        }
        // Pair-window reorder ban: a candidate that merely permutes the
        // outgoing gates is a re-spelling commutation gives away for free.
        // The ban lives INSIDE this retry loop on purpose — the banned
        // candidate is consumed and choose_ref re-evaluates free-vs-pay on
        // the survivors, so with both bans armed a commuting 2-gate pair has
        // no admissible same-length spelling and the splice is forced onto a
        // genuinely different (usually paid) one.
        if ban_reorder && is_reorder(&gates, window) {
            permutation_skipped += 1;
            continue;
        }
        chosen = Some(gates);
        choice_count = n_eligible;
        break;
    }
    DbResult {
        match_count,
        chosen,
        degree_skipped,
        identity_skipped,
        permutation_skipped,
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

/// Optional ABSOLUTE incoming-length band, `MIXER_DB_LEN_BAND=lo,hi` (read
/// once): candidates outside lo..=hi are dropped before any mode logic.
/// Experiment knob — composes with every mode (e.g. `--db-mode any` +
/// band = uniform draw over the banded lengths).
fn incoming_length_band(options: Option<&ReplacementOptions>) -> Option<(usize, usize)> {
    options.map_or_else(super::environment::len_band, |options| {
        options.incoming_length_band
    })
}

#[cfg(test)]
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
    choose_ref_with_options(refs, window_len, mode, pay_random, rng, None)
}

/// Pick which candidate to decode, from gate counts alone.
fn choose_ref_with_options<R>(
    refs: &[R],
    window_len: usize,
    mode: DbMode,
    pay_random: bool,
    rng: &mut impl Rng,
    options: Option<&ReplacementOptions>,
) -> Option<(usize, usize)>
where
    R: CandLen,
{
    let in_pool = |r: &R| {
        incoming_length_band(options).is_none_or(|(blo, bhi)| {
            let l = r.gate_count();
            l >= blo && l <= bhi
        })
    };

    // First determine the rule's threshold without materialising the eligible
    // pool. The second scan counts candidates,
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
    // Corrective band arms prefer the drift-correcting side and fall back to
    // same-size, so a correcting round is never wasted as a miss.
    let shrink_exists = mode == DbMode::BandShrink
        && refs
            .iter()
            .any(|r| in_pool(r) && r.gate_count() < window_len);
    let grow_exists = mode == DbMode::BandGrow
        && refs
            .iter()
            .any(|r| in_pool(r) && r.gate_count() > window_len);
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
        DbMode::Stable
        | DbMode::StableGrow
        | DbMode::StableLedger
        | DbMode::Same
        | DbMode::BandLedger
        | DbMode::BandShrink
        | DbMode::BandGrow => None,
    };
    if matches!(mode, DbMode::Compressing | DbMode::MinGrow) && target_len.is_none() {
        return None;
    }

    let eligible = |r: &R| {
        if !in_pool(r) {
            return false;
        }
        let l = r.gate_count();
        match mode {
            DbMode::Compressing | DbMode::MinGrow => Some(l) == target_len,
            DbMode::SizeAgnostic => true,
            DbMode::Mix if free_exists => l <= window_len,
            DbMode::Mix if pay_random => true,
            DbMode::Mix => Some(l) == target_len,
            // Everything within one gate of the window; no fallback — a
            // window with no near-size spelling is a miss, not a shrink or a
            // growth. StableLedger reaching here untranslated reads as
            // Stable's +-1 (the walk maps it before selection).
            DbMode::Stable | DbMode::StableLedger => l + 1 >= window_len && l <= window_len + 1,
            // Same-size or one larger only: replacement never loses a gate.
            DbMode::StableGrow => l >= window_len && l <= window_len + 1,
            // Exact-size re-encode only: the probe mode for same-size
            // conversion experiments.
            DbMode::Same => l == window_len,
            // Unmapped band-ledger: the unconstrained band draw.
            DbMode::BandLedger => true,
            DbMode::BandShrink if shrink_exists => l < window_len,
            DbMode::BandShrink => l == window_len,
            DbMode::BandGrow if grow_exists => l > window_len,
            DbMode::BandGrow => l == window_len,
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
    let mut pool: Vec<usize> = (0..refs.len()).collect();
    if let Some((blo, bhi)) = incoming_length_band(None) {
        pool.retain(|&i| {
            let l = refs[i].gate_count();
            l >= blo && l <= bhi
        });
    }
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
        // Everything within one gate of the window; no fallback — a window
        // with no near-size spelling is a miss, not a shrink or a growth.
        DbMode::Stable => pool
            .into_iter()
            .filter(|&i| len_of(i) + 1 >= window_len && len_of(i) <= window_len + 1)
            .collect(),
        // Same-size or one larger only: replacement never loses a gate.
        DbMode::StableGrow => pool
            .into_iter()
            .filter(|&i| len_of(i) >= window_len && len_of(i) <= window_len + 1)
            .collect(),
        // Mapped to Stable/StableGrow by the walk before selection; if it
        // reaches here untranslated, the safe reading is Stable's ±1.
        DbMode::StableLedger => pool
            .into_iter()
            .filter(|&i| len_of(i) + 1 >= window_len && len_of(i) <= window_len + 1)
            .collect(),
        // Exact-size re-encode only: the probe mode for same-size conversion
        // experiments.
        DbMode::Same => pool
            .into_iter()
            .filter(|&i| len_of(i) == window_len)
            .collect(),
        // Unmapped band-ledger: the unconstrained band draw.
        DbMode::BandLedger => pool,
        // Corrective arms: prefer the drift-correcting side, fall back to
        // same-size so a correcting round is never wasted as a miss.
        DbMode::BandShrink => {
            let smaller: Vec<usize> = pool
                .iter()
                .copied()
                .filter(|&i| len_of(i) < window_len)
                .collect();
            if smaller.is_empty() {
                pool.into_iter()
                    .filter(|&i| len_of(i) == window_len)
                    .collect()
            } else {
                smaller
            }
        }
        DbMode::BandGrow => {
            let larger: Vec<usize> = pool
                .iter()
                .copied()
                .filter(|&i| len_of(i) > window_len)
                .collect();
            if larger.is_empty() {
                pool.into_iter()
                    .filter(|&i| len_of(i) == window_len)
                    .collect()
            } else {
                larger
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
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/db_mixing/replacement/tests.rs"]
mod tests;
