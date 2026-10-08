// The profile policy on a null plant: the whole-circuit controller walks
// its eff marks through the rounds exactly as the serial run does (phase
// 4 reached, hold phase traversed), the pieces never break the function,
// and the merged circuit still verifies against the stage input.
#[test]
fn piecewise_profile_null_plant_reaches_phase_4() {
    let gates = random_mixed_circuit(41, 16, 2000);
    let s_in = gates.len() as f64;
    let params = MixParams {
        k_max: 6,
        moves: 4_000_000,
        p_db: 1.0,
        s_db: 5,
        p_convex: 0.5,
        mix_pay_random: true,
        prof_n: [6.0, 18.0, 34.0],
        prof_r: [4.0, 2.0],
        prof_cadence_eff: 0.5,
        report_every: u64::MAX,
        verify_every: 200_000,
        seed: 3,
        ..MixParams::default()
    };
    let mut w = Mixer::new_with_db(gates, 16, params, FrozenDb::empty());
    let stop = run_piecewise(&mut w, &piece_cfg(4, false, 0));
    assert!(matches!(stop, MixStop::ProfileDone), "profile ends the run");
    let p = w.prof.as_ref().expect("profile armed");
    assert_eq!(p.phase, 4, "phase machine must reach compress-done");
    assert!(p.eff >= 18.0, "ran through the hold phase: eff {}", p.eff);
    assert!(p.pmix >= 0.0 && p.pmix <= 1.0);
    assert!(w.arena.len() as f64 <= 2.0 * s_in);
    assert!(w.moves_done > 0);
    assert_eq!(w.counters.moves, w.moves_done);
    w.global_check();
}

// Output is a function of (input, params, seed) only: sequential pieces,
// a 2-thread pool and a 5-thread pool produce identical circuits,
// provenance, counters and clocks; a different seed does not. Driven by the
// stage-3 DB walk with real store hits inside the pieces.
#[test]
fn piecewise_result_is_independent_of_thread_count() {
    let gates = random_g57_circuit(23, 16, 1500);
    let params = MixParams {
        k_max: 5,
        moves: 30_000,
        p_twist: 0.02,
        verify_every: 10_000,
        report_every: u64::MAX,
        seed: 9,
        ..db_walk_params()
    };
    let run = |seed: u64, sequential: bool, threads: usize| {
        let mut w = Mixer::new_with_db(
            gates.clone(),
            16,
            MixParams {
                seed,
                ..params.clone()
            },
            g57_fixture_store(),
        );
        let cfg = PieceCfg {
            pieces: 3,
            round_eff: 0.5,
            sequential,
            threads,
            ..PieceCfg::default()
        };
        let stop = run_piecewise(&mut w, &cfg);
        assert!(matches!(stop, MixStop::MovesBudget));
        w.global_check();
        (
            w.arena.to_vec(),
            meta_snapshot(&w),
            w.counters.to_line(),
            w.counters.db_slot2_hits,
            w.moves_done,
            w.next_litter,
        )
    };
    let a = run(9, true, 0);
    let b = run(9, false, 2);
    let c = run(9, false, 5);
    assert!(a == b, "sequential vs 2 threads differ");
    assert!(a == c, "sequential vs 5 threads differ");
    assert!(a.3 > 1_000, "the pieces must splice: {}", a.3);
    assert!(a.4 >= 30_000, "the moves budget is spent: {}", a.4);
    let d = run(10, true, 0);
    assert!(a.0 != d.0, "a different seed must walk differently");
}

#[test]
fn piecewise_auto_matches_fixed_when_count_stays_constant() {
    let gates = random_mixed_circuit(23, 6, 400);
    let params = MixParams {
        // An empty DB consumes the DB slot without changing the gates,
        // so both drivers must have exactly the same seeded trajectory.
        p_db: 1.0,
        p_twist: 0.0,
        moves: 2_000,
        report_every: u64::MAX,
        verify_every: 1_000,
        seed: 9,
        ..MixParams::default()
    };
    let run = |cfg: PieceCfg| {
        let mut w = Mixer::new_with_db(gates.clone(), 6, params.clone(), FrozenDb::empty());
        assert!(matches!(run_piecewise(&mut w, &cfg), MixStop::MovesBudget));
        w.global_check();
        (
            w.arena.to_vec(),
            meta_snapshot(&w),
            w.counters.to_line(),
            w.moves_done,
        )
    };
    let fixed = run(piece_cfg(4, true, 0));
    let auto = run(PieceCfg {
        min_block_size: Some(100),
        threads: 2,
        ..PieceCfg::default()
    });
    assert_eq!(auto, fixed);
}

#[test]
fn piecewise_auto_one_block_completes_profile() {
    let gates = random_mixed_circuit(11, 8, 160);
    let cfg = PieceCfg {
        min_block_size: Some(1_000),
        threads: 2,
        ..PieceCfg::default()
    };
    let profile = MixParams {
        p_db: 1.0,
        p_twist: 0.0,
        moves: 20_000,
        prof_n: [0.25, 0.5, 0.75],
        prof_r: [2.0, 2.0],
        report_every: u64::MAX,
        seed: 3,
        ..MixParams::default()
    };
    let mut w = Mixer::new_with_db(gates, 8, profile, FrozenDb::empty());
    assert!(matches!(run_piecewise(&mut w, &cfg), MixStop::ProfileDone));
    w.global_check();
}

#[test]
fn piecewise_auto_empty_circuits_terminate() {
    let cfg = PieceCfg {
        min_block_size: Some(8),
        threads: 2,
        ..PieceCfg::default()
    };
    for profile in [false, true] {
        let params = MixParams {
            moves: 20,
            prof_n: if profile { [0.25, 0.5, 0.75] } else { [0.0; 3] },
            prof_r: [2.0, 2.0],
            report_every: u64::MAX,
            ..MixParams::default()
        };
        let mut w = Mixer::new_with_db(Vec::new(), 6, params, FrozenDb::empty());
        let stop = run_piecewise(&mut w, &cfg);
        assert!(matches!(stop, MixStop::CircuitEmpty));
        assert_eq!(w.arena.len(), 0);
        w.global_check();
    }
    // An actual cancelling walk reaches zero inside a round, so the
    // next round must also terminate instead of scheduling zero jobs. The
    // compressing DB move contracts every [g, g] window to nothing.
    let params = MixParams {
        moves: 5_000,
        db_mode: DbMode::Compressing,
        p_twist: 0.0,
        shuffle_rate: 0.0,
        report_every: u64::MAX,
        seed: 7,
        ..db_walk_params()
    };
    let gates = vec![XGate::from_g57([0, 1, 2]); 64];
    let mut w = Mixer::new_with_db(gates, 6, params, g57_fixture_store());
    assert!(matches!(run_piecewise(&mut w, &cfg), MixStop::CircuitEmpty));
    assert_eq!(
        w.arena.len(),
        0,
        "the fixture must actually contract to zero"
    );
    w.global_check();
}

// Slicing a mixer into pieces and concatenating them back without any
// moves is an identity on the gates and every Meta field, and the litter
// counter lands on the highest piece band.
#[test]
fn piece_transport_round_trip_is_an_identity() {
    let gates = random_g57_circuit(37, 16, 600);
    let params = MixParams {
        k_max: 5,
        moves: 3_000,
        p_twist: 0.05,
        verify_every: 1_000,
        report_every: u64::MAX,
        seed: 11,
        ..db_walk_params()
    };
    let mut w = Mixer::new_with_db(gates, 16, params.clone(), g57_fixture_store());
    w.run();
    assert!(
        w.counters.db_slot2_hits > 0,
        "the walk must have spliced, so the metadata is not all input-fresh"
    );
    let gates_before = w.arena.to_vec();
    let metas_before = meta_snapshot(&w);
    let n = gates_before.len();
    let cuts = vec![n / 3, (2 * n) / 3];
    let parts = w.export_slices(&cuts);
    assert_eq!(parts.len(), 3);
    let db = w.shared_db();
    let m = w.moves_done;
    let bl = w.next_litter;
    let outs: Vec<PieceParts> = parts
        .into_iter()
        .enumerate()
        .map(|(i, mut p)| {
            let lb = bl + ((i as u64) << 32);
            p.litter_base = lb;
            let mut mx = Mixer::from_parts(p, 16, params.clone(), Arc::clone(&db), m, lb, true);
            mx.global_check();
            let mut o = mx.export_parts();
            o.litter_base = lb;
            o.moves_start = m;
            o
        })
        .collect();
    let seams = w.rebuild_from_parts(outs);
    assert_eq!(seams, cuts);
    assert_eq!(w.arena.to_vec(), gates_before);
    assert_eq!(meta_snapshot(&w), metas_before);
    assert_eq!(w.moves_done, m);
    assert_eq!(w.next_litter, bl + (2u64 << 32));
    w.global_check();
    // The rebuilt mixer keeps walking.
    w.params.moves = w.moves_done + 1_000;
    w.run();
    w.global_check();
}

#[test]
fn resolved_runtime_phase_stop_survives_piecewise_reassembly() {
    for piecewise in [false, true] {
        let gates = random_mixed_circuit(41, 16, 400);
        let params = MixParams {
            k_max: 6,
            moves: 4_000_000,
            p_db: 1.0,
            s_db: 5,
            p_convex: 0.5,
            mix_pay_random: true,
            prof_n: [2.0, 4.0, 6.0],
            prof_r: [4.0, 2.0],
            prof_cadence_eff: 0.5,
            report_every: u64::MAX,
            verify_every: 200_000,
            seed: 3,
            ..MixParams::default()
        };
        let mut mixer = Mixer::new_with_runtime_options(
            gates,
            16,
            params,
            Arc::new(FrozenDb::empty()),
            MixRuntimeOptions {
                stop_at_phase: Some(2),
                ..MixRuntimeOptions::default()
            },
        );
        let stop = if piecewise {
            run_piecewise(&mut mixer, &piece_cfg(4, false, 2))
        } else {
            mixer.run()
        };
        assert_eq!(stop, MixStop::StopFlag);
        assert_eq!(mixer.prof.as_ref().unwrap().phase, 2);
        mixer.global_check();
    }
}
