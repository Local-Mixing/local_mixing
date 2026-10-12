// ---- the split stage (docs/tdp_pipeline.md) ----

// The whole stage on a mixed circuit, every sub-rewrite locally verified
// (presplits, absorptions, segment conjugations), ending by g57
// exhaustion: the run stops at the boundary under split_stop, no comp
// gate survives, twists actually landed, and the circuit still computes
// the input.
#[test]
fn split_stage_runs_to_exhaustion_and_preserves_function() {
    let gates = random_mixed_circuit(11, 16, 400);
    let comp0 = gates.iter().filter(|g| g.comp).count();
    assert!(comp0 > 10, "test input must carry g57s, got {comp0}");
    let params = MixParams {
        k_max: 8,
        moves: 200_000,
        split: true,
        split_stop: true,
        split_canaries: 32,
        report_every: u64::MAX,
        verify_every: 1_000,
        seed: 5,
        ..MixParams::default()
    };
    let mut mx = Mixer::new_with_db(gates, 16, params, FrozenDb::empty());
    let stop = mx.run();
    assert!(
        matches!(stop, MixStop::SplitDone),
        "stage must end the run under split_stop"
    );
    assert_eq!(mx.remaining_g57(), 0, "exit A means no comp gate survives");
    let splits =
        (mx.counters.split_prims + mx.counters.split_hsplits + mx.counters.split_segs) as usize;
    assert!(
        splits >= comp0,
        "every input g57 splits through SOME channel: {splits} < {comp0}"
    );
    assert!(mx.counters.split_joins > 0, "p_join 0.8 must land twists");
    mx.global_check();
}

// Sibling convention: the two pieces of a g57 split take
// opposite directions.
#[test]
fn split_g57_pieces_take_opposite_directions() {
    for seed in 0..16 {
        let gates = vec![
            XGate::from_g57([0, 1, 2]),
            XGate::conj(3, [(1u16, true)]).unwrap(),
        ];
        let params = MixParams {
            seed,
            moves: 0,
            report_every: u64::MAX,
            ..MixParams::default()
        };
        let mut mx = Mixer::new_with_db(gates, 4, params, FrozenDb::empty());
        let (g1, g2) = mx.split_g57(0);
        assert_ne!(
            mx.meta_of(g1).dir,
            mx.meta_of(g2).dir,
            "siblings must oppose"
        );
        mx.global_check();
    }
}

// Min-dgen cross-shot bias: armed, the walk still preserves function and
// the pool actually supplies shots; off, the constructor path draws no
// extra RNG so the trajectory is byte-identical to the same seeded chain.
#[test]
fn mincross_pool_supplies_shots_and_preserves_function() {
    let gates = random_mixed_circuit(17, 16, 400);
    let params = MixParams {
        k_max: 8,
        moves: 60_000,
        p_mincross: 0.9,
        cross_pool_k: 500,
        cross_rescan: 2_000,
        report_every: u64::MAX,
        verify_every: 10_000,
        seed: 21,
        ..MixParams::default()
    };
    let mut mx = Mixer::new_with_db(gates.clone(), 16, params, FrozenDb::empty());
    mx.run();
    assert!(
        mx.counters.cross_pool_shots > 1_000,
        "the pool must actually supply shots"
    );
    mx.global_check();

    // Off = identical trajectory: same seed, with and without the (inert)
    // pool knobs, ends in the same circuit.
    let base = MixParams {
        k_max: 8,
        moves: 30_000,
        report_every: u64::MAX,
        verify_every: u64::MAX,
        seed: 22,
        ..MixParams::default()
    };
    let mut a = Mixer::new_with_db(gates.clone(), 16, base.clone(), FrozenDb::empty());
    let mut b = Mixer::new_with_db(
        gates,
        16,
        MixParams {
            cross_pool_k: 7,
            cross_rescan: 55,
            ..base
        },
        FrozenDb::empty(),
    );
    a.run();
    b.run();
    assert_eq!(
        a.arena.to_vec(),
        b.arena.to_vec(),
        "p_mincross 0 must not perturb the walk"
    );
}
