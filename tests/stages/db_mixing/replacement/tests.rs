use super::*;
use crate::circuit::xgate::eval_lanes;
use rand::RngCore;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::HashMap;

struct FakeCand {
    n: usize,
    cur: bool,
}
impl CandLen for FakeCand {
    fn gate_count(&self) -> usize {
        self.n
    }
    fn curated(&self) -> bool {
        self.cur
    }
}

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
fn stable_mode_picks_only_near_size() {
    let mk = |ns: &[usize]| -> Vec<FakeCand> {
        ns.iter().map(|&n| FakeCand { n, cur: false }).collect()
    };
    let refs = mk(&[3, 4, 5, 6, 7, 9]);
    let mut rng = StdRng::seed_from_u64(7);
    let mut seen = std::collections::HashSet::new();
    for _ in 0..200 {
        let (i, pool) = choose_ref(&refs, 5, DbMode::Stable, false, &mut rng)
            .expect("near-size spellings exist");
        assert_eq!(pool, 3, "eligible set must be exactly {{4,5,6}}");
        seen.insert(refs[i].n);
    }
    assert_eq!(seen, std::collections::HashSet::from([4, 5, 6]));

    // No spelling within one gate of the window: a MISS, never a far pick.
    let far = mk(&[2, 9]);
    assert!(choose_ref(&far, 5, DbMode::Stable, false, &mut rng).is_none());

    // Stable keeps the STRICT curated lexicographic-first rule (the
    // stable-mixing experiment stays within curated material): a curated
    // far-size candidate hides the regular same-size one — a MISS.
    let curmix = vec![FakeCand { n: 5, cur: false }, FakeCand { n: 9, cur: true }];
    assert!(choose_ref(&curmix, 5, DbMode::Stable, false, &mut rng).is_none());
    // With a near-size CURATED candidate present, the draw is within the
    // curated class only — the regular same-size spelling stays unseen.
    let joint = vec![
        FakeCand { n: 4, cur: true },
        FakeCand { n: 5, cur: false },
        FakeCand { n: 9, cur: true },
    ];
    for _ in 0..50 {
        let (i, pool) = choose_ref(&joint, 5, DbMode::Stable, false, &mut rng).unwrap();
        assert_eq!((joint[i].n, joint[i].cur, pool), (4, true, 1));
    }

    // StableGrow: the shrink class is excluded — only w and w+1 draw.
    let refs = mk(&[3, 4, 5, 6, 7, 9]);
    let mut seen = std::collections::HashSet::new();
    for _ in 0..200 {
        let (i, pool) = choose_ref(&refs, 5, DbMode::StableGrow, false, &mut rng).unwrap();
        assert_eq!(pool, 2, "eligible must be exactly {{5,6}}");
        seen.insert(refs[i].n);
    }
    assert_eq!(seen, std::collections::HashSet::from([5, 6]));
    // Only a smaller spelling available: MISS — a gate is never lost.
    let only_shrink = mk(&[4]);
    assert!(choose_ref(&only_shrink, 5, DbMode::StableGrow, false, &mut rng).is_none());
}

#[test]
fn scan_selector_matches_reference_selection_and_rng_state() {
    let modes = [
        DbMode::Compressing,
        DbMode::SizeAgnostic,
        DbMode::MinGrow,
        DbMode::Mix,
        DbMode::Stable,
        DbMode::StableGrow,
        DbMode::StableLedger,
        DbMode::Same,
        DbMode::BandLedger,
        DbMode::BandShrink,
        DbMode::BandGrow,
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
fn qc_limits() -> QcCandidateLimits {
    QcCandidateLimits {
        max_candidates: 32,
        max_gates: 12,
        max_support: 12,
    }
}

#[test]
fn qc_returns_only_exact_equivalents_and_deduplicates_directions() {
    let g = db_g57_to_xgate([0, 1, 2]);
    let window = vec![g.clone(), g];
    // Empty circuits are equivalent; a lone g57 is a corrupt answer for
    // the identity key and must be rejected even when the store says yes.
    let value = vec![0, 3, 0, 1, 2];
    let result = qc_candidates_with(
        &window,
        3,
        XPolyBudget::default(),
        qc_limits(),
        &mut StdRng::seed_from_u64(1),
        |_, _| Some(value.clone()),
    );
    assert_eq!(result.entries_found, 4);
    assert_eq!(result.lookups, 4);
    assert_eq!(result.examined, 8);
    assert!(!result.truncated);
    assert_eq!(result.candidates.len(), 1);
    assert!(result.candidates[0].gates.is_empty());
    assert!(result.non_equivalent > 0);
    assert!(result.duplicate_skipped > 0);
}

#[test]
fn qc_round_robin_candidate_cap_reaches_regular_store() {
    let g = db_g57_to_xgate([0, 1, 2]);
    let window = vec![g.clone(), g];
    let mut limits = qc_limits();
    limits.max_candidates = 2;
    let result = qc_candidates_with(
        &window,
        3,
        XPolyBudget::default(),
        limits,
        &mut StdRng::seed_from_u64(2),
        |_, curated| {
            Some(if curated {
                vec![3, 0, 1, 2, 3, 0, 1, 2]
            } else {
                vec![0]
            })
        },
    );
    assert_eq!(result.examined, 2);
    assert!(result.truncated);
    assert_eq!(result.candidates.len(), 1);
    assert!(!result.candidates[0].from_curated);
}

#[test]
fn qc_distinguishes_no_entry_canonicalization_and_record_budget() {
    let window = vec![db_g57_to_xgate([0, 1, 2])];
    let mut rng = StdRng::seed_from_u64(3);
    let miss = qc_candidates_with(
        &window,
        3,
        XPolyBudget::default(),
        qc_limits(),
        &mut rng,
        |_, _| None,
    );
    assert_eq!(miss.entries_found, 0);
    assert_eq!(miss.lookups, 4);
    assert!(miss.canonicalization_errors.is_empty());
    assert!(!miss.truncated);
    let budget = XPolyBudget {
        max_mul_terms: 0,
        max_poly_terms: 0,
        max_total_terms: 0,
    };
    let capped = qc_candidates_with(&window, 3, budget, qc_limits(), &mut rng, |_, _| {
        panic!("must not query an undecided key")
    });
    assert_eq!(capped.canonicalization_errors.len(), 2);
    assert_eq!(capped.lookups, 0);
    let limits = QcCandidateLimits {
        max_candidates: 0,
        ..qc_limits()
    };
    let skipped = qc_candidates_with(
        &window,
        3,
        XPolyBudget::default(),
        limits,
        &mut rng,
        |_, _| panic!("zero budget must not query"),
    );
    assert!(skipped.truncated);
    assert_eq!(skipped.examined, 0);
}

#[test]
fn qc_bounds_rejected_records_and_handles_malformed_values() {
    let window = vec![db_g57_to_xgate([0, 1, 2])];
    let mut rng = StdRng::seed_from_u64(4);
    let malformed = qc_candidates_with(
        &window,
        3,
        XPolyBudget::default(),
        qc_limits(),
        &mut rng,
        |_, _| Some(vec![6, 0]),
    );
    assert_eq!(malformed.malformed_values, 4);
    assert_eq!(malformed.examined, 4);
    assert!(malformed.candidates.is_empty());
    let limits = QcCandidateLimits {
        max_candidates: 3,
        max_gates: 0,
        ..qc_limits()
    };
    let oversized = qc_candidates_with(
        &window,
        3,
        XPolyBudget::default(),
        limits,
        &mut rng,
        |_, _| Some(vec![3, 0, 1, 2]),
    );
    assert_eq!(oversized.examined, 3);
    assert_eq!(oversized.size_skipped, 3);
    assert!(oversized.truncated);
}

#[test]
fn qc_reads_reverse_only_friend_and_restores_global_wire_order() {
    let original = CircuitSeq {
        gates: vec![[0, 1, 2], [1, 2, 3], [2, 0, 3]],
    };
    let mut inverse = original.clone();
    inverse.gates.reverse();
    let (reverse_key, value) = store_friend(&inverse);
    let mut window: Vec<XGate> = original
        .gates
        .iter()
        .copied()
        .map(db_g57_to_xgate)
        .collect();
    // A cancelling pad keeps the function while making the stored friend
    // a real alternative, rather than an identity spelling.
    window.extend([XGate::x_gate(0), XGate::x_gate(0)]);
    let result = qc_candidates_with(
        &window,
        4,
        XPolyBudget::default(),
        qc_limits(),
        &mut StdRng::seed_from_u64(5),
        |key, curated| {
            if !curated && *key == reverse_key {
                Some(value.clone())
            } else {
                None
            }
        },
    );
    assert!(result.candidates.iter().any(|candidate| candidate.reversed));
    for candidate in result.candidates {
        assert!(exhaustively_equal(&window, &candidate.gates, 4));
    }
}

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

#[test]
fn explicit_length_bands_are_independent_and_keep_rng_draw_order() {
    let refs: Vec<FakeCand> = [3, 4, 5, 6, 7]
        .into_iter()
        .map(|n| FakeCand { n, cur: false })
        .collect();
    for (lo, hi) in [(3, 4), (5, 7)] {
        let options = ReplacementOptions {
            incoming_length_band: Some((lo, hi)),
            ..ReplacementOptions::default()
        };
        let eligible: Vec<usize> = refs
            .iter()
            .enumerate()
            .filter(|(_, value)| (lo..=hi).contains(&value.n))
            .map(|(i, _)| i)
            .collect();
        let mut rng = StdRng::seed_from_u64(91);
        let mut reference = StdRng::seed_from_u64(91);
        for _ in 0..30 {
            let expected = eligible[reference.random_range(0..eligible.len())];
            let actual = choose_ref_with_options(
                &refs,
                5,
                DbMode::SizeAgnostic,
                false,
                &mut rng,
                Some(&options),
            );
            assert_eq!(actual, Some((expected, eligible.len())));
            assert_eq!(rng.clone().next_u64(), reference.clone().next_u64());
        }
    }
}

#[test]
fn explicit_direction_policy_controls_real_canonical_lookup_order() {
    use crate::database::lookup_cache::MinDirLookup;
    let window: Vec<XGate> = [[0, 1, 2], [1, 2, 3], [2, 0, 3]]
        .into_iter()
        .map(db_g57_to_xgate)
        .collect();
    let mut observed = Vec::new();
    for direction in [
        MinDirLookup::Legacy,
        MinDirLookup::Min,
        MinDirLookup::Validate,
    ] {
        let mut calls = Vec::new();
        let options = ReplacementOptions {
            direction,
            ..ReplacementOptions::default()
        };
        let result = db_replace_with_options(
            &window,
            4,
            XPolyBudget::default(),
            DbMode::SizeAgnostic,
            DegreeGuard::OFF,
            false,
            true,
            false,
            false,
            &mut StdRng::seed_from_u64(5),
            |key, curated| {
                assert!(!curated);
                calls.push(*key);
                None::<Vec<u8>>
            },
            &options,
        );
        assert!(result.chosen.is_none());
        observed.push(calls);
    }
    assert_eq!(
        observed[0].len(),
        2,
        "the fixture must have distinct forward/reverse keys"
    );
    assert_eq!(observed[1].len(), 1);
    assert_eq!(observed[2].len(), 2);
    assert!(observed[0].contains(&observed[1][0]));
    assert_eq!(observed[1][0], observed[2][0]);
}
