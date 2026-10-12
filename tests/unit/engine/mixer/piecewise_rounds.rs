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
        temp: 20.0,
        p_db: 1.0,
        p_comp: 1.0,
        p_any: 0.1,
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

// The split policy: pieces split to their own exhaustion, rounds repeat
// until the merged circuit has no g57 left, the canaries planted once on
// the whole circuit survive the slicing, and the merged state resumes
// into a stage-5-style thermostat walk unchanged.
#[test]
fn piecewise_split_stage_exhausts_g57s_and_resumes() {
    let gates = random_mixed_circuit(11, 16, 1200);
    let comp0 = gates.iter().filter(|g| g.comp).count();
    assert!(comp0 > 30, "test input must carry g57s, got {comp0}");
    let params = MixParams {
        k_max: 8,
        moves: 400_000,
        split: true,
        split_stop: true,
        split_canaries: 32,
        report_every: u64::MAX,
        verify_every: 1_000,
        seed: 5,
        ..MixParams::default()
    };
    let mut w = Mixer::new_with_db(gates, 16, params.clone(), FrozenDb::empty());
    let stop = run_piecewise(&mut w, &piece_cfg(3, false, 0));
    assert!(
        matches!(stop, MixStop::SplitDone),
        "stage boundary ends the run"
    );
    assert_eq!(
        w.remaining_g57(),
        0,
        "global exhaustion: no comp gate survives"
    );
    let splits =
        (w.counters.split_prims + w.counters.split_hsplits + w.counters.split_segs) as usize;
    assert!(
        splits >= comp0,
        "every input g57 splits: {splits} < {comp0}"
    );
    assert!(w.counters.split_joins > 0, "p_join 0.8 must land twists");
    assert_eq!(w.taps.len(), 32, "canaries planted once on W and carried");
    assert!(w.split_done && !w.split_on);
    assert_eq!(w.split_end_reason, Some("g57 pool exhausted"));
    w.global_check();

    let path = std::env::temp_dir().join("circuit_mixer_piecewise_split.state");
    let path = path.to_str().unwrap();
    w.save_state(path).expect("save");
    let resume_params = MixParams {
        split: false,
        split_stop: false,
        ..params
    };
    let mut rs = Mixer::resume_state(path, resume_params, FrozenDb::empty()).expect("resume");
    assert_eq!(
        rs.arena.to_vec(),
        w.arena.to_vec(),
        "circuit survives verbatim"
    );
    assert_eq!(meta_snapshot(&rs), meta_snapshot(&w), "provenance survives");
    assert_eq!(
        rs.moves_done, w.moves_done,
        "the merged move clock is the resume baseline"
    );
    assert!(rs.split_done && !rs.split_on, "phase 2 never re-arms");
    assert_eq!(rs.taps.len(), 32);
    rs.global_check();
    rs.params.moves = rs.moves_done + 5_000;
    rs.params.target_size = rs.arena.len() * 2;
    rs.params.temp = 64.0;
    let st = rs.run();
    assert!(matches!(st, MixStop::MovesBudget));
    rs.global_check();
    let _ = std::fs::remove_file(path);
}

// Output is a function of (input, params, seed) only: sequential pieces,
// a 2-thread pool and a 5-thread pool produce identical circuits,
// provenance, counters and clocks; a different seed does not.
#[test]
fn piecewise_result_is_independent_of_thread_count() {
    let gates = random_mixed_circuit(23, 16, 1500);
    let params = MixParams {
        k_max: 5,
        moves: 30_000,
        target_size: 1800,
        temp: 40.0,
        p_twist: 0.02,
        w_twist_neg: 1.0,
        verify_every: 10_000,
        report_every: u64::MAX,
        seed: 9,
        ..MixParams::default()
    };
    let run = |seed: u64, sequential: bool, threads: usize| {
        let mut w = Mixer::new_with_db(
            gates.clone(),
            16,
            MixParams {
                seed,
                ..params.clone()
            },
            FrozenDb::empty(),
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
            w.moves_done,
            w.next_event,
            w.next_litter,
        )
    };
    let a = run(9, true, 0);
    let b = run(9, false, 2);
    let c = run(9, false, 5);
    assert!(a == b, "sequential vs 2 threads differ");
    assert!(a == c, "sequential vs 5 threads differ");
    assert!(a.3 >= 30_000, "the moves budget is spent: {}", a.3);
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
fn piecewise_auto_one_block_completes_profile_and_split() {
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
    let mut w = Mixer::new_with_db(gates.clone(), 8, profile, FrozenDb::empty());
    assert!(matches!(run_piecewise(&mut w, &cfg), MixStop::ProfileDone));
    w.global_check();
    let split = MixParams {
        moves: 20_000,
        split: true,
        split_stop: true,
        report_every: u64::MAX,
        seed: 5,
        ..MixParams::default()
    };
    let mut w = Mixer::new_with_db(gates, 8, split, FrozenDb::empty());
    assert!(w.remaining_g57() > 0);
    assert!(matches!(run_piecewise(&mut w, &cfg), MixStop::SplitDone));
    assert_eq!(w.remaining_g57(), 0);
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
    // next round must also terminate instead of scheduling zero jobs.
    let params = MixParams {
        moves: 5_000,
        target_size: 1,
        temp: 1.0,
        w_cross: 0.0,
        w_fresh: 0.0,
        w_unsub: 0.0,
        w_insert: 0.0,
        p_twist: 0.0,
        shuffle_rate: 0.0,
        undo_frac: 0.0,
        report_every: u64::MAX,
        seed: 7,
        ..MixParams::default()
    };
    let gates = vec![XGate::conj(0, []).unwrap(); 64];
    let mut w = Mixer::new_with_db(gates, 6, params, FrozenDb::empty());
    assert!(matches!(run_piecewise(&mut w, &cfg), MixStop::CircuitEmpty));
    assert_eq!(
        w.arena.len(),
        0,
        "the fixture must actually contract to zero"
    );
    w.global_check();
}

// Slicing a mixer into pieces and concatenating them back without any
// moves is an identity on the gates, every Meta field, the canaries, the
// live tabu ring and the merge index; the undo journal keeps every entry
// that lies wholly inside a piece.
#[test]
fn piece_transport_round_trip_is_an_identity() {
    let gates = random_mixed_circuit(37, 16, 600);
    let params = MixParams {
        k_max: 5,
        moves: 3_000,
        target_size: 700,
        temp: 20.0,
        p_twist: 0.05,
        w_twist_neg: 1.0,
        split_canaries: 16,
        verify_every: 1_000,
        report_every: u64::MAX,
        seed: 11,
        ..MixParams::default()
    };
    let mut w = Mixer::new_with_db(gates, 16, params.clone(), FrozenDb::empty());
    w.run();
    w.plant_taps();
    assert_eq!(w.taps.len(), 16);
    assert!(
        !w.journal.is_empty(),
        "the walk must have journalled crossings"
    );
    let (ids, pos_of) = w.pos_map();
    let taps_before: Vec<(u16, u16, u64, u32)> = w
        .taps
        .iter()
        .map(|t| (t.wire, t.orig_permille, t.flips, pos_of[&t.anchor]))
        .collect();
    let live_tabu: Vec<(u64, u64)> = w
        .tabu
        .iter()
        .copied()
        .filter(|&(_, mv)| mv + w.params.tabu_moves > w.moves_done)
        .collect();
    let gates_before = w.arena.to_vec();
    let metas_before = meta_snapshot(&w);
    let n = ids.len();
    let cuts = vec![n / 3, (2 * n) / 3];
    let parts = w.export_slices(&cuts);
    assert_eq!(parts.len(), 3);
    let inside_count: usize = parts.iter().map(|p| p.journal.len()).sum();
    assert!(inside_count <= w.journal.len());
    let db = w.shared_db();
    let m = w.moves_done;
    let (be, bl) = (w.next_event, w.next_litter);
    let outs: Vec<PieceParts> = parts
        .into_iter()
        .enumerate()
        .map(|(i, mut p)| {
            let eb = be + ((i as u64) << 32);
            let lb = bl + ((i as u64) << 32);
            p.event_base = eb;
            p.litter_base = lb;
            let mut mx = Mixer::from_parts(p, 16, params.clone(), Arc::clone(&db), m, eb, lb, true);
            mx.global_check();
            let mut o = mx.export_parts();
            o.event_base = eb;
            o.litter_base = lb;
            o.moves_start = m;
            o
        })
        .collect();
    let seams = w.rebuild_from_parts(outs);
    assert_eq!(seams, cuts);
    assert_eq!(w.arena.to_vec(), gates_before);
    assert_eq!(meta_snapshot(&w), metas_before);
    let (_, pos_after) = w.pos_map();
    // Canaries come back grouped by piece (position order); the order of
    // the tap list is not a contract, the set of (wire, origin, flips,
    // position) is.
    let mut taps_after: Vec<(u16, u16, u64, u32)> = w
        .taps
        .iter()
        .map(|t| (t.wire, t.orig_permille, t.flips, pos_after[&t.anchor]))
        .collect();
    taps_after.sort_unstable();
    let mut taps_before = taps_before;
    taps_before.sort_unstable();
    assert_eq!(taps_after, taps_before);
    let tabu_after: Vec<(u64, u64)> = w.tabu.iter().copied().collect();
    assert_eq!(tabu_after, live_tabu);
    assert_eq!(w.journal.len(), inside_count);
    assert!(w.journal.iter().all(|e| {
        e.after
            .iter()
            .all(|&(id, st)| w.arena.is_linked(id) && w.arena.stamp(id) == st)
    }));
    assert_eq!(w.moves_done, m);
    assert_eq!(w.next_event, be + (2u64 << 32));
    w.global_check();
    // The merge index is exact after the rebuild (global_check asserts
    // indexed_count == arena.len()); the walk can continue.
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
            temp: 20.0,
            p_db: 1.0,
            p_comp: 1.0,
            p_any: 0.1,
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
