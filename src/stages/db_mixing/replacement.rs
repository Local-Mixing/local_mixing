//! Frozen-store lookup for a single heterogeneous [`XGate`] window, used by the
//! fmix DB contraction move (see [`crate::engine::mix`]).
//!
//! A convex/contiguous window of arbitrary-width, mixed-polarity XGates is keyed
//! by its exact function polynomial ([`crate::canonicalization::xgate`]) — identical to
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

use crate::canonicalization::xgate::{
    CanonicalXPolys, XPolyBudget, XPolyError, canonicalize_xgates_single,
    canonicalize_xgates_single_capped, xgates_to_polynomial,
};
use crate::circuit::xgate::XGate;
use crate::circuit::{CircuitSeq, Permutation, polys_repr_blob};
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

/// Work and size caps for the independent gate-wise quality-control lookup.
/// All caps are hard limits; zero does not disable a limit.
#[derive(Clone, Copy, Debug)]
pub struct QcCandidateLimits {
    /// Maximum encoded candidate records examined across both stores and both
    /// directions, including rejected and duplicate records.
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
    pub from_curated: bool,
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
    /// `(curated, reason)` for bounded store reads that could not be completed.
    pub lookup_limits: Vec<(bool, QcLookupLimit)>,
    pub malformed_values: usize,
    pub unplaceable: usize,
    pub identity_skipped: usize,
    pub duplicate_skipped: usize,
    pub size_skipped: usize,
    pub non_equivalent: usize,
    pub equivalence_budget_exceeded: usize,
}

/// Enumerate a bounded QC candidate set without changing normal DB selection
/// or its sampling distribution. Both canonical directions and both stores are
/// considered, round-robin, so a large curated value cannot starve regular
/// candidates while the examination budget remains. Identity spellings are
/// skipped and every returned circuit is proved equivalent before exposure.
pub fn qc_candidates(
    window: &[XGate],
    num_wires: usize,
    db: &FrozenDb,
    budget: XPolyBudget,
    limits: QcCandidateLimits,
    rng: &mut impl Rng,
) -> QcCandidates {
    let mut lookup_limits = Vec::new();
    let mut result =
        qc_candidates_with_value(window, num_wires, budget, limits, rng, |key, curated| {
            // One extra record makes the global enumeration's truncation signal
            // exact even when this value alone consumes the entire cap.
            let cap = limits.max_candidates.saturating_add(1);
            let value = if curated {
                db.get_curated_qc(key, cap, QC_MAX_BUCKET_BYTES)
            } else {
                db.get_regular_qc(key, cap, QC_MAX_BUCKET_BYTES)
            };
            match value {
                Ok(value) => value,
                Err(reason) => {
                    lookup_limits.push((curated, reason));
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
    F: FnMut(&[u8; 16], bool) -> Option<Vec<u8>>,
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
    F: FnMut(&[u8; 16], bool) -> Option<V>,
    V: AsRef<[u8]>,
{
    struct Cursor<V> {
        value: V,
        position: usize,
        canonical: CanonicalXPolys,
        curated: bool,
        reversed: bool,
    }
    let mut result = QcCandidates::default();
    if limits.max_candidates == 0 {
        result.truncated = true;
        return result;
    }
    let mut cursors = Vec::with_capacity(4);
    for reversed in [false, true] {
        let canonical = match canonicalize_xgates_single(window, reversed, budget) {
            Ok(canonical) => canonical,
            Err(error) => {
                result.canonicalization_errors.push((reversed, error));
                continue;
            }
        };
        let key = key_of(&canonical);
        for curated in [true, false] {
            result.lookups += 1;
            if let Some(value) = lookup(&key, curated) {
                result.entries_found += 1;
                cursors.push(Cursor {
                    value,
                    position: 0,
                    canonical: canonical.clone(),
                    curated,
                    reversed,
                });
            }
        }
    }
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
            if gates == window {
                result.identity_skipped += 1;
                continue;
            }
            if !seen.insert(gates.clone()) {
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
                    from_curated: cursor.curated,
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
    /// The FORWARD canonical key of this window, when it was computed. Lets a
    /// caller re-query another store for the same permutation — e.g. to read
    /// the true complexity class of a big-pool hit from a reference store
    /// whose small spellings the pool swap replaced.
    pub fwd_key: Option<[u8; 16]>,
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
            // (see security_tests/support/db_mixing/replace.rs): the store is immutable, so cached
            // hits AND misses are byte-identical to raw probes.
            use crate::database::lookup_cache::{
                LOOKUP_NS_CURATED, LOOKUP_NS_SHARD, cached_db_get,
            };
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
    lookup: F,
) -> DbResult
where
    F: FnMut(&[u8; 16], bool) -> Option<V>,
    V: AsRef<[u8]>,
{
    db_replace_with_value_options(
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
    curated_armed: bool,
    regular_fallback: bool,
    pay_random: bool,
    ban_reorder: bool,
    rng: &mut impl Rng,
    lookup: F,
    options: &ReplacementOptions,
) -> DbResult
where
    F: FnMut(&[u8; 16], bool) -> Option<V>,
    V: AsRef<[u8]>,
{
    db_replace_with_value_options(
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
    curated: bool,
    curated_in_comp: bool,
    regular_fallback: bool,
    pay_random: bool,
    ban_reorder: bool,
    rng: &mut impl Rng,
    options: &ReplacementOptions,
) -> DbResult {
    let armed = curated_armed_for(curated, mode, curated_in_comp);
    db_replace_with_options(
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
            if want_curated {
                if curated { db.get_curated(key) } else { None }
            } else {
                db.get_regular(key)
            }
        },
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
    curated_armed: bool,
    regular_fallback: bool,
    pay_random: bool,
    ban_reorder: bool,
    rng: &mut impl Rng,
    mut lookup: F,
    options: Option<&ReplacementOptions>,
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
        fwd_key: None,
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
    match canonicalize_replacement(window, false, budget, guard.max_degree, options) {
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
    // Forward canonical key, exposed on DbResult so a caller can re-query
    // another store for the same permutation (class attribution).
    let mut fwd_key: Option<[u8; 16]> = directions
        .iter()
        .find(|(rev, _)| !rev)
        .map(|(_, c)| key_of(c));

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
        match canonicalize_replacement(window, true, budget, guard.max_degree, options) {
            Ok(c) => directions.push((true, c)),
            Err(XPolyError::DegreeExceeded { .. }) => degree_skipped = true,
            Err(_) => {}
        }
        // The regular store is keyed by min(canon_fwd, canon_rev) (see the
        // MIN_DIR_LOOKUP note in security_tests/support/db_mixing/replace.rs): when both directions
        // composed, the default Min mode probes only the min direction — the
        // non-min key can only exist if it equals the min key, so the candidate
        // set is unchanged. Legacy restores the historical probe-both cascade;
        // Validate probes the other direction on a miss and counts violations.
        use crate::database::lookup_cache::{
            MIN_DIR_VALIDATE_PROBES, MIN_DIR_VIOLATIONS, MinDirLookup, min_dir_lookup_mode,
        };
        use std::sync::atomic::Ordering;
        let fwd_idx = directions.iter().position(|(reversed, _)| !reversed);
        let rev_idx = directions.iter().position(|(reversed, _)| *reversed);
        // Stage B may be the first place the forward direction exists.
        if fwd_key.is_none() {
            if let Some(i) = fwd_idx {
                fwd_key = Some(key_of(&directions[i].1));
            }
        }
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
        fwd_key,
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
/// to shrink. Stable is deliberately NOT exempt: the stable-mixing experiment
/// tests whether staying WITHIN curated (highly inflated) material is stable,
/// so a curated answer with no near-size member is a miss — the same-size
/// regular spelling stays discarded unseen, by design.
/// Optional ABSOLUTE incoming-length band, `FMIX_DB_LEN_BAND=lo,hi` (read
/// once): candidates outside lo..=hi are dropped before any mode logic.
/// Experiment knob — composes with every mode (e.g. `--db-mode any` +
/// band = uniform draw over the banded lengths).
fn incoming_length_band(options: Option<&ReplacementOptions>) -> Option<(usize, usize)> {
    options.map_or_else(super::legacy_environment::len_band, |options| {
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
    let restrict_curated = mode != DbMode::Compressing && refs.iter().any(|r| r.curated());
    let in_pool = |r: &R| {
        (!restrict_curated || r.curated())
            && incoming_length_band(options).is_none_or(|(blo, bhi)| {
                let l = r.gate_count();
                l >= blo && l <= bhi
            })
    };

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
    let restrict_curated = mode != DbMode::Compressing && refs.iter().any(|r| r.curated());
    let mut pool: Vec<usize> = (0..refs.len())
        .filter(|&i| !restrict_curated || refs[i].curated())
        .collect();
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
    fn curated(&self) -> bool;
}

#[cfg(test)]
#[path = "../../../tests/stages/db_mixing/replacement/tests.rs"]
mod tests;
