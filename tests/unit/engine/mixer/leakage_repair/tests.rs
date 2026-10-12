use super::*;
use crate::database::frozen::FrozenDb;
use crate::engine::mixer::{Meta, MixParams, ORIGIN_SYNTH, Tap, UndoEntry};
use rand::Rng;

fn mixer(gates: Vec<XGate>) -> Mixer {
    Mixer::new_with_db(
        gates,
        8,
        MixParams {
            ancestors: true,
            db_advance: true,
            db_verify: false,
            report_every: u64::MAX,
            ..MixParams::default()
        },
        FrozenDb::empty(),
    )
}

fn meta_tuple(meta: Meta) -> (u32, u64, Dir, u32, u64, u16) {
    (
        meta.origin,
        meta.event,
        meta.dir,
        meta.dgen,
        meta.litter,
        meta.litter_size,
    )
}

fn journal_pair(mx: &Mixer, ids: [u32; 2]) -> UndoEntry {
    let first = mx.meta_of(ids[0]);
    let second = mx.meta_of(ids[1]);
    UndoEntry {
        before: ids.map(|id| mx.arena.gate(id).clone()),
        dir: Dir::R,
        pivot: ids[0],
        after: ids.iter().map(|&id| (id, mx.arena.stamp(id))).collect(),
        event: 42,
        origins: [first.origin, second.origin],
        gens: [first.dgen, second.dgen],
        litters: [first.litter, second.litter],
        litter_sizes: [first.litter_size, second.litter_size],
        misses: 0,
    }
}

#[test]
fn quality_repair_preserves_live_metadata_random_stream_and_sampling_state() {
    let gate = XGate::cnot(0, 1);
    let middle = XGate::cnot(2, 3);
    let prefix = XGate::x_gate(6);
    let suffix = XGate::from_g57([4, 5, 7]);
    let original = vec![
        prefix.clone(),
        gate.clone(),
        middle.clone(),
        gate,
        suffix.clone(),
    ];
    let mut mx = mixer(original.clone());
    let ids = mx.arena.ids_in_order();
    mx.set_meta(
        ids[1],
        Meta {
            dgen: 3,
            ..mx.meta_of(ids[1])
        },
    );
    mx.set_meta(
        ids[3],
        Meta {
            dgen: 7,
            ..mx.meta_of(ids[3])
        },
    );
    mx.moves_done = 81;
    mx.counters.moves = 81;
    mx.counters.db_agn_hits = 17;
    mx.counters.len_hits = vec![9, 8, 7];
    mx.band_led_round = true;
    mx.band_led = 5;
    mx.pool = vec![ids[0], ids[2], ids[4]];
    mx.journal.push_back(journal_pair(&mx, [ids[1], ids[3]]));
    mx.journal.push_back(journal_pair(&mx, [ids[0], ids[4]]));
    let untouched: Vec<_> = [ids[0], ids[2], ids[4]]
        .into_iter()
        .map(|id| (id, mx.arena.stamp(id), meta_tuple(mx.meta_of(id))))
        .collect();
    let counters = mx.counters.to_line();
    let len_hits = mx.counters.len_hits.clone();
    let pool = mx.pool.clone();
    let next_event = mx.next_event;
    let mut expected_rng = mx.rng.clone();
    let mut expected_metrics_rng = mx.metrics_rng.clone();
    let plan = ConvexBlock {
        span: 1..4,
        selected: vec![1, 3],
        permutation: vec![1, 3, 2],
        block: 1..3,
        support: 2,
    };
    let fresh = XGate::from_g57([0, 1, 2]);
    // Two identity pairs add material as well as exercising side indexes.
    let replacement = vec![fresh.clone(); 4];
    mx.apply_quality_repair(&plan, replacement.clone()).unwrap();
    let expected = [vec![prefix], replacement, vec![middle, suffix]].concat();
    assert_eq!(mx.arena.to_vec(), expected);
    assert_eq!(mx.quality_reference(), original);
    assert_eq!(mx.quality_num_wires(), 8);
    assert_eq!(mx.moves_done, 81);
    assert_eq!(mx.next_event, next_event);
    assert_eq!(mx.counters.to_line(), counters);
    assert_eq!(mx.counters.len_hits, len_hits);
    assert_eq!(mx.pool, pool);
    assert_eq!(mx.band_led, 5);
    assert!(mx.band_led_round && mx.params.db_advance);
    assert!(!mx.params.db_verify);
    assert_eq!(mx.rng.clone().random::<u64>(), expected_rng.random::<u64>());
    assert_eq!(
        mx.metrics_rng.clone().random::<u64>(),
        expected_metrics_rng.random::<u64>()
    );
    for (id, stamp, meta) in untouched {
        assert!(mx.arena.is_linked(id));
        assert_eq!(mx.arena.stamp(id), stamp);
        assert_eq!(meta_tuple(mx.meta_of(id)), meta);
    }
    let after = mx.arena.ids_in_order();
    let litter = mx.meta_of(after[1]).litter;
    for &id in &after[1..5] {
        let meta = mx.meta_of(id);
        assert_eq!(meta.dgen, 8);
        assert_eq!(meta.origin, ORIGIN_SYNTH);
        assert_eq!((meta.litter, meta.litter_size, meta.event), (litter, 4, 0));
        let mut ancestry = vec![0; mx.anc_words];
        mx.anc_or_into(meta.litter, &mut ancestry);
        assert_eq!(ancestry, vec![(1 << 1) | (1 << 3)]);
    }
    assert_eq!(mx.journal.len(), 2);
    assert!(
        mx.journal[0]
            .after
            .iter()
            .any(|&(id, stamp)| mx.arena.stamp(id) != stamp)
    );
    assert!(
        mx.journal[1]
            .after
            .iter()
            .all(|&(id, stamp)| mx.arena.stamp(id) == stamp)
    );
    assert_eq!(mx.indexed_count, mx.arena.len());
    for &id in &after {
        let key = super::super::key_of(mx.arena.gate(id));
        assert_eq!(mx.index[&key][mx.index_pos[id as usize] as usize], id);
        assert_eq!(mx.comp_pos[id as usize] != NIL, mx.arena.gate(id).comp);
        assert_eq!(
            mx.wt_pos[id as usize] != NIL,
            mx.arena.gate(id).comp || mx.arena.gate(id).ctrls.len() == 1
        );
    }
    mx.global_check();

    let checkpoint = std::env::temp_dir().join(format!(
        "local_mixing_quality_metadata_{}.state",
        std::process::id()
    ));
    let checkpoint_path = checkpoint.to_str().unwrap();
    mx.save_state(checkpoint_path).unwrap();
    let loaded = Mixer::resume_state(checkpoint_path, mx.params.clone(), FrozenDb::empty());
    std::fs::remove_file(checkpoint).unwrap();
    let mut resumed = loaded.unwrap();
    assert_eq!(resumed.quality_reference(), original);
    assert_eq!(resumed.arena.to_vec(), expected);
    assert_eq!(resumed.moves_done, mx.moves_done);
    assert_eq!(resumed.counters.to_line(), mx.counters.to_line());
    assert_eq!(
        resumed.journal.len(),
        1,
        "only the untouched entry survives"
    );
    for (before, after) in mx
        .arena
        .ids_in_order_iter()
        .zip(resumed.arena.ids_in_order_iter())
    {
        assert_eq!(
            meta_tuple(mx.meta_of(before)),
            meta_tuple(resumed.meta_of(after))
        );
        let mut expected_ancestry = vec![0; mx.anc_words];
        let mut actual_ancestry = vec![0; resumed.anc_words];
        mx.anc_or_into(mx.meta_of(before).litter, &mut expected_ancestry);
        resumed.anc_or_into(resumed.meta_of(after).litter, &mut actual_ancestry);
        assert_eq!(actual_ancestry, expected_ancestry);
    }
    resumed.global_check();
}

#[test]
fn quality_repair_rejects_invalid_gather_and_replacement_atomically() {
    let original = vec![XGate::cnot(0, 1), XGate::cnot(1, 2), XGate::x_gate(3)];
    let mut mx = mixer(original.clone());
    let ids = mx.arena.ids_in_order();
    let metadata: Vec<_> = ids.iter().map(|&id| meta_tuple(mx.meta_of(id))).collect();
    let counters = mx.counters.to_line();
    let mut rng = mx.rng.clone();
    let mut plan = ConvexBlock {
        span: 0..2,
        selected: vec![0, 1],
        permutation: vec![1, 0],
        block: 0..2,
        support: 3,
    };
    let reversed = vec![original[1].clone(), original[0].clone()];
    assert!(
        mx.apply_quality_repair(&plan, reversed)
            .unwrap_err()
            .contains("noncommuting")
    );
    plan.permutation = vec![0, 0];
    assert!(
        mx.apply_quality_repair(&plan, original[..2].to_vec())
            .is_err()
    );
    plan.permutation = vec![0, 1];
    assert!(
        mx.apply_quality_repair(&plan, vec![XGate::x_gate(0)])
            .unwrap_err()
            .contains("not functionally equivalent")
    );
    assert_eq!(mx.arena.to_vec(), original);
    assert_eq!(mx.arena.ids_in_order(), ids);
    assert_eq!(
        ids.iter()
            .map(|&id| meta_tuple(mx.meta_of(id)))
            .collect::<Vec<_>>(),
        metadata
    );
    assert_eq!(mx.counters.to_line(), counters);
    assert_eq!(mx.rng.clone().random::<u64>(), rng.random::<u64>());
}

#[test]
fn quality_repair_keeps_whole_circuit_tap_anchors_live() {
    let gate = XGate::cnot(0, 1);
    let mut mx = mixer(vec![gate.clone(), gate]);
    let ids = mx.arena.ids_in_order();
    mx.taps.push(Tap {
        anchor: ids[1],
        wire: 0,
        orig_permille: 500,
        flips: 3,
    });
    mx.tap_at.insert(ids[1], vec![0]);
    let plan = ConvexBlock {
        span: 0..2,
        selected: vec![0, 1],
        permutation: vec![0, 1],
        block: 0..2,
        support: 2,
    };
    assert!(mx.apply_quality_repair(&plan, vec![]).is_err());
    mx.apply_quality_repair(&plan, vec![XGate::x_gate(2); 4])
        .unwrap();
    assert!(mx.arena.is_linked(mx.taps[0].anchor));
    assert_eq!(mx.tap_at[&mx.taps[0].anchor], vec![0]);
    assert_eq!(
        (mx.taps[0].wire, mx.taps[0].orig_permille, mx.taps[0].flips),
        (0, 500, 3)
    );
    mx.global_check();
}
