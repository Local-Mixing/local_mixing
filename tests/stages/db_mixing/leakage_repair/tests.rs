use super::*;
use crate::circuit::polys_repr_blob;
use crate::db_mixing::db_replace::{polys_equivalent, qc_candidates_with};
use crate::db_mixing::frozen::FrozenDb;
use crate::engine::mix::MixParams;
use crate::engine::xpoly::canonicalize_xgates_single;
use xxhash_rust::xxh3::xxh3_128;

fn input() -> Vec<XGate> {
    let p = XGate::conj(0, [(1, true), (2, false)]).unwrap();
    vec![p.clone(), p, XGate::x_gate(3)]
}

fn mixer(gates: Vec<XGate>) -> Mixer {
    Mixer::new_with_db(
        gates,
        4,
        MixParams {
            seed: 7,
            ..MixParams::default()
        },
        FrozenDb::empty(),
    )
}

// Real canonicalization/decoding/proof, fed a tiny encoded store through
// the QC lookup seam. FrozenDb::open allocates 1.3 GB of shard offsets even
// for one entry; that file-format integration is covered in fmix_db_move.
fn run_with_value(
    mixer: &mut Mixer,
    config: &QualityConfig,
    value: Option<Vec<u8>>,
) -> QualityReport {
    run_quality_control_with(
        mixer,
        None,
        config,
        |window, num_wires, budget, limits, rng| {
            qc_candidates_with(window, num_wires, budget, limits, rng, |_, _| value.clone())
        },
    )
    .unwrap()
}

#[test]
fn controller_repairs_hot_identity_pair_with_exact_db_replacement() {
    let original = input();
    let mut mixer = mixer(original.clone());
    let report = run_with_value(&mut mixer, &QualityConfig::default(), Some(vec![0]));
    assert_eq!(report.repairs, 1);
    assert!(report.before.hot_segment_count > 0);
    assert_eq!(report.after.hot_segment_count, 0);
    assert_eq!(report.fresh_audit.hot_segment_count, 0);
    assert_eq!(report.events[0].status, "repaired");
    assert_eq!(mixer.arena.to_vec(), vec![XGate::x_gate(3)]);
    assert_eq!(mixer.moves_done, 0, "QC must not spend ordinary walk moves");
    assert_eq!(
        polys_equivalent(&original, &mixer.arena.to_vec(), XPolyBudget::default()),
        Some(true)
    );
}

#[test]
fn controller_separates_missing_entry_from_too_large_convex_block() {
    let original = input();
    let mut missing = mixer(original.clone());
    let miss = run_with_value(&mut missing, &QualityConfig::default(), None);
    assert_eq!(miss.repairs, 0);
    assert_eq!(miss.events[0].status, "no_db_entry");
    assert_eq!(missing.arena.to_vec(), original);
    let mut config = QualityConfig::default();
    config.block_limits.max_support = 2;
    let mut limited = mixer(original.clone());
    let report = run_quality_control_with(&mut limited, None, &config, |_, _, _, _, _| {
        panic!("oversized closure must not reach the database")
    })
    .unwrap();
    assert_eq!(report.events[0].status, "block_limit");
    assert!(report.events[0].detail.contains("SupportCap"));
    assert_eq!(limited.arena.to_vec(), original);
}

#[test]
fn controller_rejects_equivalent_replacement_with_same_affine_leakage() {
    let original = input();
    let window = &original[..2];
    let canonical = canonicalize_xgates_single(window, false, XPolyBudget::default()).unwrap();
    let key = xxh3_128(&polys_repr_blob(&canonical.polys)).to_le_bytes();
    // Complementing the original conjunction predicate changes its middle
    // wire value by a constant, so the affine detector must still reject.
    let inverse = canonical.order.invert();
    let mut blob = Vec::new();
    for _ in 0..2 {
        for wire in [0u16, 2, 1] {
            let dense = canonical.used_wires.binary_search(&wire).unwrap();
            blob.push(inverse.data[dense] as u8);
        }
    }
    let mut value = vec![blob.len() as u8];
    value.extend(blob);
    let mut mixer = mixer(original.clone());
    let report = run_quality_control_with(
        &mut mixer,
        None,
        &QualityConfig::default(),
        |window, num_wires, budget, limits, rng| {
            qc_candidates_with(window, num_wires, budget, limits, rng, |query, _| {
                if *query == key {
                    Some(value.clone())
                } else {
                    None
                }
            })
        },
    )
    .unwrap();
    assert_eq!(report.repairs, 0);
    assert_eq!(report.events[0].status, "examined_replacements_hot");
    assert_eq!(mixer.arena.to_vec(), original);
}

#[test]
fn controller_declines_empty_live_tape_before_region_tracing() {
    let original = input()[..2].to_vec();
    let mut mixer = mixer(original.clone());
    let report = run_with_value(&mut mixer, &QualityConfig::default(), Some(vec![0]));
    assert_eq!(report.repairs, 0);
    assert_eq!(report.events[0].status, "application_refused");
    assert!(report.events[0].detail.contains("empty_circuit"));
    assert_eq!(mixer.arena.to_vec(), original);
}

#[test]
fn controller_rejects_hot_firing_moved_to_final_wire_write() {
    let p = XGate::conj(0, [(1, true), (2, false)]).unwrap();
    let original = vec![p.clone(), p.clone(), p, XGate::x_gate(3)];
    let mut mixer = mixer(original.clone());
    let report = run_with_value(&mut mixer, &QualityConfig::default(), Some(vec![0]));
    assert_eq!(report.repairs, 0);
    assert!(
        report
            .events
            .iter()
            .all(|event| event.status == "examined_replacements_hot")
    );
    assert_eq!(mixer.arena.to_vec(), original);
}

#[test]
fn controller_preserves_candidate_and_polynomial_budget_failure_causes() {
    let original = input();
    let mut config = QualityConfig::default();
    config.candidate_limits.max_candidates = 1;
    let mut limited = mixer(original.clone());
    // Corrupt answers are proved unequal; the remaining unread store
    // records mean the failure cannot be called an exhausted hot class.
    let report = run_with_value(&mut limited, &config, Some(vec![3, 0, 1, 2]));
    assert_eq!(report.events[0].status, "candidate_budget");
    assert_eq!(limited.arena.to_vec(), original);
    let mut config = QualityConfig::default();
    config.polynomial_budget = XPolyBudget {
        max_mul_terms: 0,
        max_poly_terms: 0,
        max_total_terms: 0,
    };
    let mut limited = mixer(original.clone());
    let report = run_with_value(&mut limited, &config, Some(vec![0]));
    assert_eq!(report.events[0].status, "lookup_or_verification_limit");
    assert_eq!(limited.arena.to_vec(), original);
}

#[test]
fn controller_refreshes_segment_indices_after_each_repair() {
    let p = XGate::conj(0, [(1, true), (2, false)]).unwrap();
    let q = XGate::conj(3, [(4, true), (5, false)]).unwrap();
    let original = vec![
        p.clone(),
        p,
        XGate::x_gate(6),
        q.clone(),
        q,
        XGate::x_gate(7),
    ];
    let mut mixer =
        Mixer::new_with_db(original.clone(), 8, MixParams::default(), FrozenDb::empty());
    let report = run_with_value(&mut mixer, &QualityConfig::default(), Some(vec![0]));
    assert_eq!(report.repairs, 2);
    assert_eq!(
        report
            .events
            .iter()
            .map(|e| (e.start_gate, e.end_gate))
            .collect::<Vec<_>>(),
        vec![(0, 1), (1, 2)]
    );
    assert_eq!(
        mixer.arena.to_vec(),
        vec![XGate::x_gate(6), XGate::x_gate(7)]
    );
    assert_eq!(
        polys_equivalent(&original, &mixer.arena.to_vec(), XPolyBudget::default()),
        Some(true)
    );
}
