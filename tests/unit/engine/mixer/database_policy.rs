// Sampled windows are contiguous in link order and never break the circuit
// function (both samplers only float commuting gates), and the control cap
// keeps wide gates out of the collected window.
#[test]
fn window_samplers_are_contiguous_capped_and_function_preserving() {
    fn same_fn(a: &[XGate], b: &[XGate], nw: usize, seed: u64) -> bool {
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..8 {
            let init: Vec<u64> = (0..nw).map(|_| rng.random()).collect();
            let (mut sa, mut sb) = (init.clone(), init);
            crate::circuit::xgate::eval_lanes(a.iter(), &mut sa);
            crate::circuit::xgate::eval_lanes(b.iter(), &mut sb);
            if sa != sb {
                return false;
            }
        }
        true
    }
    for sample in [DbSample::Contiguous, DbSample::Convex] {
        let gates = random_mixed_circuit(19, 16, 400);
        let reference = gates.clone();
        let params = MixParams {
            p_convex: if sample == DbSample::Convex { 1.0 } else { 0.0 },
            w_window: 3, // no window gate may reach width 3
            w_pool: 3,
            s_db: 6,
            db_convex_p: 0.75,
            report_every: u64::MAX,
            seed: 5,
            ..MixParams::default()
        };
        let mut mx = Mixer::new(gates, 16, params);
        let mut got = 0usize;
        for w in 0..4000 {
            let win = (w % 5) + 2; // window sizes 2..=6
            // Geometry is a parameter now, not an internal coin; the test
            // already knows which one it is exercising.
            if let Some((ids, _dir, _smp)) = mx.sample_window(win, sample) {
                got += 1;
                // contiguous in link order
                for pair in ids.windows(2) {
                    assert_eq!(
                        mx.arena.neighbor(pair[0], Dir::R),
                        pair[1],
                        "{sample:?} window ids not contiguous"
                    );
                }
                assert!(ids.len() >= 2 && ids.len() <= win);
                // control cap respected
                for &id in &ids {
                    assert!(
                        mx.arena.gate(id).width() <= 2,
                        "{sample:?} window kept a gate wider than the cap"
                    );
                }
            }
        }
        assert!(
            got > 500,
            "{sample:?} sampler almost never produced a window: {got}"
        );
        // Sampling floats only commuting gates -> whole-circuit function intact.
        assert!(
            same_fn(&reference, &mx.arena.to_vec(), 16, 1),
            "{sample:?} sampling changed the circuit function"
        );
    }
}

#[test]
fn contiguous_uses_gate_direction_and_spills_at_boundary() {
    // A 4-gate circuit on disjoint wires (all commute); a window of 4 from any
    // start must gather all four regardless of direction / boundary.
    let gates = vec![
        XGate::conj(0, [(1, true)]).unwrap(),
        XGate::conj(2, [(3, true)]).unwrap(),
        XGate::conj(4, [(5, true)]).unwrap(),
        XGate::conj(6, [(7, true)]).unwrap(),
    ];
    let params = MixParams {
        s_db: 2,
        report_every: u64::MAX,
        seed: 2,
        ..MixParams::default()
    };
    let mut mx = Mixer::new(gates, 8, params);
    for _ in 0..50 {
        let (ids, _d) = mx.collect_contiguous(4).expect("window");
        assert_eq!(ids.len(), 4, "boundary spill did not reach the quota");
    }
}

// Prefix descent must terminate: each loop must shrink the window even
// when every lookup misses. An empty database exercises that path without
// relying on any stored entries.
#[test]
fn prefix_descent_terminates_when_every_lookup_misses() {
    let gates = random_mixed_circuit(43, 16, 300);
    let params = MixParams {
        k_max: 6,
        moves: 5_000,
        target_size: 300,
        temp: 20.0,
        p_db: 1.0,
        db_prefixes: true,
        s_db: 9,
        db_min_window: 0,
        db_max_span: 4, // deliberately tight: forces the span-skip path too
        p_twist: 0.0,
        shuffle_rate: 0.0,
        report_every: u64::MAX,
        seed: 13,
        ..MixParams::default()
    };
    // FrozenDb::empty() -> every lookup misses, which IS the path that used
    // to spin. No store needed to prove termination.
    let mut mx = Mixer::new_with_db(gates, 16, params, FrozenDb::empty());
    mx.run();
    assert_eq!(
        mx.counters.moves, 5_000,
        "the run did not consume its move budget"
    );
    assert!(
        mx.counters.db_attempts > 5_000,
        "the descent should make several attempts per round, got {}",
        mx.counters.db_attempts
    );
    assert!(
        mx.counters.db_span_skips > 0,
        "the tight span cap should have skipped windows"
    );
    mx.global_check();
}

// A DB splice stamps its products with the outgoing window's
// upper-median generation + 1 (benchmark semantics: median rounded up
// on even window sizes).
#[test]
fn db_splice_stamps_upper_median_plus_one() {
    let g = XGate::conj(0, [(1, true)]).unwrap();
    let h = XGate::conj(2, [(3, true)]).unwrap();
    // Two adjacent identity pairs; we splice over the first pair.
    let gates = vec![g.clone(), g.clone(), h.clone(), h.clone()];
    let params = MixParams {
        report_every: u64::MAX,
        ..MixParams::default()
    };
    let mut mx = Mixer::new(gates, 8, params);
    let ids = mx.arena.ids_in_order();
    // Unequal gens across the window: the products must take min + 1.
    let m0 = mx.meta_of(ids[0]);
    mx.set_meta(ids[0], Meta { dgen: 3, ..m0 });
    let m1 = mx.meta_of(ids[1]);
    mx.set_meta(ids[1], Meta { dgen: 7, ..m1 });
    let window = vec![g.clone(), g.clone()];
    let replacement = vec![h.clone(), h.clone()]; // also an identity: verifies
    assert!(mx.try_db_splice_curated(
        false,
        &ids[..2],
        Dir::R,
        &window,
        replacement,
        1,
        DbMode::SizeAgnostic
    ));
    mx.global_check();
    let gens = mx.gens_in_order();
    assert_eq!(
        &gens[..2],
        &[8, 8],
        "products carry upper-median(3,7)+1 = 8: {gens:?}"
    );
    assert_eq!(
        &gens[2..],
        &[0, 0],
        "untouched gates keep their gen: {gens:?}"
    );
    // Lower-median variant: same {3,7} spread stamps min+1 on the
    // 2-gate window.
    mx.params.gen_median_low = true;
    let ids = mx.arena.ids_in_order();
    let m0 = mx.meta_of(ids[0]);
    mx.set_meta(ids[0], Meta { dgen: 3, ..m0 });
    let m1 = mx.meta_of(ids[1]);
    mx.set_meta(ids[1], Meta { dgen: 7, ..m1 });
    let window = vec![h.clone(), h.clone()];
    let replacement = vec![g.clone(), g.clone()];
    assert!(mx.try_db_splice_curated(
        false,
        &ids[..2],
        Dir::R,
        &window,
        replacement,
        1,
        DbMode::SizeAgnostic
    ));
    assert_eq!(
        &mx.gens_in_order()[..2],
        &[4, 4],
        "lower median of (3,7) is 3 -> products 4"
    );
    // Saturation: an all-fresh window stays fresh (either median).
    mx.params.gen_median_low = false;
    let ids = mx.arena.ids_in_order();
    for &id in &ids[..2] {
        let m = mx.meta_of(id);
        mx.set_meta(
            id,
            Meta {
                dgen: GEN_FRESH,
                ..m
            },
        );
    }
    let window = vec![g.clone(), g.clone()];
    let replacement = vec![h.clone(), h.clone()];
    assert!(mx.try_db_splice_curated(
        false,
        &ids[..2],
        Dir::R,
        &window,
        replacement,
        1,
        DbMode::SizeAgnostic
    ));
    assert_eq!(
        mx.gens_in_order()[0],
        GEN_FRESH,
        "fresh window must stay fresh"
    );
}

// The TDP profile needs the window length to depend on BOTH the live mode
// and the geometry drawn for that round -- COMP convex 12 / COMP
// contiguous 6 / MIX 6 for either. Precedence is most-specific-first, and
// an unset level must fall through rather than clamp the window to nothing.
#[test]
fn s_db_resolves_by_mode_and_geometry() {
    let gates = random_mixed_circuit(11, 8, 40);
    let params = MixParams {
        s_db: 6,
        s_db_comp: Some(12),
        s_db_comp_ctg: Some(6),
        s_db_ctg: None, // MIX shares one length across both geometries
        moves: 0,
        report_every: u64::MAX,
        ..MixParams::default()
    };
    let mut mx = Mixer::new(gates, 8, params);

    mx.db_mode_cur = DbMode::Compressing;
    assert_eq!(
        mx.active_s_db(DbSample::Convex),
        12,
        "COMP convex takes s_db_comp"
    );
    assert_eq!(
        mx.active_s_db(DbSample::Contiguous),
        6,
        "COMP contiguous takes s_db_comp_ctg"
    );

    mx.db_mode_cur = DbMode::Mix;
    assert_eq!(
        mx.active_s_db(DbSample::Convex),
        6,
        "MIX takes the base s_db"
    );
    assert_eq!(
        mx.active_s_db(DbSample::Contiguous),
        6,
        "s_db_ctg=0 falls through to the base s_db, it does not zero the window"
    );

    // A geometry override only applies to its own mode.
    mx.params.s_db_ctg = Some(3);
    assert_eq!(
        mx.active_s_db(DbSample::Contiguous),
        3,
        "MIX contiguous overridden"
    );
    mx.db_mode_cur = DbMode::Compressing;
    assert_eq!(
        mx.active_s_db(DbSample::Contiguous),
        6,
        "COMP contiguous still reads s_db_comp_ctg, not the MIX override"
    );
}

// Descent is per-mode: the overlay runs MIX and COMP in one process and
// TDP wants it on in COMP and off in MIX. None must fall back to the
// global flag so single-mode runs behave exactly as before.
#[test]
fn prefix_descent_resolves_per_mode() {
    let gates = random_mixed_circuit(11, 8, 40);
    let params = MixParams {
        db_prefixes: true,
        db_prefixes_comp: Some(true),
        db_prefixes_mix: Some(false),
        moves: 0,
        report_every: u64::MAX,
        ..MixParams::default()
    };
    let mut mx = Mixer::new(gates, 8, params);
    mx.db_mode_cur = DbMode::Compressing;
    assert!(mx.active_prefixes(), "COMP descends");
    mx.db_mode_cur = DbMode::Mix;
    assert!(
        !mx.active_prefixes(),
        "MIX does not, even though db_prefixes is true"
    );

    // Unset per-mode overrides inherit the global flag, both ways.
    mx.params.db_prefixes_mix = None;
    mx.params.db_prefixes_comp = None;
    assert!(mx.active_prefixes(), "None inherits db_prefixes=true");
    mx.params.db_prefixes = false;
    assert!(!mx.active_prefixes(), "None inherits db_prefixes=false");
}

// Geometry is drawn once per round, before the length. A run pinned to
// one geometry must therefore only ever report that geometry, and the
// per-length histogram must respect that geometry's own s_db ceiling.
#[test]
fn geometry_is_drawn_before_the_length_and_bounds_it() {
    let gates = random_mixed_circuit(31, 16, 400);
    let base = MixParams {
        moves: 4_000,
        target_size: 400,
        temp: 20.0,
        p_db: 0.0, // store-free: exercise the sampler, not the store
        s_db: 9,
        db_min_window: 0,
        s_db_ctg: Some(3),
        verify_every: u64::MAX,
        report_every: u64::MAX,
        seed: 7,
        ..MixParams::default()
    };
    // All-contiguous: every window must obey s_db_ctg=3, not s_db=9.
    let mut ctg = Mixer::new(
        gates.clone(),
        16,
        MixParams {
            p_convex: 0.0,
            ..base.clone()
        },
    );
    ctg.db_mode_cur = DbMode::Mix;
    for _ in 0..200 {
        let w = ctg.active_s_db(DbSample::Contiguous);
        assert!(w <= 3, "contiguous window ceiling leaked: {w}");
    }
    // All-convex: the contiguous override must not apply.
    let mut cvx = Mixer::new(
        gates,
        16,
        MixParams {
            p_convex: 1.0,
            ..base
        },
    );
    cvx.db_mode_cur = DbMode::Mix;
    assert_eq!(cvx.active_s_db(DbSample::Convex), 9);
}

// Every layering rule in one place, asserted against MixParams::db_knobs --
// the function both the mixer and the CLI banner go through.
#[test]
fn db_knobs_layering_rules() {
    let base = MixParams {
        s_db: 9,
        p_convex: 0.4,
        p_mingen: 0.8,
        db_prefixes: true,
        ..MixParams::default()
    };

    // Nothing overridden: both modes see the base.
    let k = base.db_knobs(DbMode::Mix);
    let c = base.db_knobs(DbMode::Compressing);
    assert_eq!((k.s_db_cvx, k.s_db_ctg, k.p_convex), (9, 9, 0.4));
    assert_eq!((c.s_db_cvx, c.s_db_ctg, c.p_convex), (9, 9, 0.4));

    // A mode override moves only that mode, and reaches BOTH its geometries.
    let p = MixParams {
        s_db_comp: Some(12),
        ..base.clone()
    };
    assert_eq!(p.db_knobs(DbMode::Mix).s_db_cvx, 9);
    assert_eq!(p.db_knobs(DbMode::Compressing).s_db_cvx, 12);
    assert_eq!(
        p.db_knobs(DbMode::Compressing).s_db_ctg,
        12,
        "COMP contiguous inherits COMP convex, not the base"
    );

    // A geometry override is narrower still.
    let p = MixParams {
        s_db_comp: Some(12),
        s_db_comp_ctg: Some(6),
        ..base.clone()
    };
    let c = p.db_knobs(DbMode::Compressing);
    assert_eq!((c.s_db_cvx, c.s_db_ctg), (12, 6));
    assert_eq!(
        p.db_knobs(DbMode::Mix).s_db_ctg,
        9,
        "MIX untouched by COMP overrides"
    );

    // Option distinguishes explicit zero and false values from unset
    // overrides. Those values must take precedence over the base settings.
    let p = MixParams {
        p_mingen: 0.8,
        p_mingen_comp: Some(0.0),
        ..base.clone()
    };
    assert_eq!(p.db_knobs(DbMode::Compressing).p_mingen, 0.0);
    assert_eq!(p.db_knobs(DbMode::Mix).p_mingen, 0.8);
    let p = MixParams {
        db_prefixes: true,
        db_prefixes_mix: Some(false),
        ..base.clone()
    };
    assert!(!p.db_knobs(DbMode::Mix).prefixes);
    assert!(p.db_knobs(DbMode::Compressing).prefixes);

    // The TDP profile, end to end.
    let tdp = MixParams {
        s_db: 6,
        p_convex: 0.5,
        p_mingen: 0.5,
        db_prefixes: true,
        db_prefixes_mix: Some(false),
        db_prefixes_comp: Some(true),
        p_mingen_comp: Some(0.0),
        p_convex_comp: Some(0.95),
        s_db_comp: Some(12),
        s_db_comp_ctg: Some(6),
        ..MixParams::default()
    };
    let m = tdp.db_knobs(DbMode::Mix);
    let c = tdp.db_knobs(DbMode::Compressing);
    assert_eq!(
        (m.s_db_cvx, m.s_db_ctg, m.p_convex, m.p_mingen, m.prefixes),
        (6, 6, 0.5, 0.5, false)
    );
    assert_eq!(
        (c.s_db_cvx, c.s_db_ctg, c.p_convex, c.p_mingen, c.prefixes),
        (12, 6, 0.95, 0.0, true)
    );
}

// The seed restore must be COLLISION-CHECKED, not an unchecked relink.
// Window building floats gates other than the seed -- ctrl-cap evasion
// parks a collider out of the way, and an evaded collider is by definition
// one that does not commute with the window -- so restoring the seed by
// teleporting it back to its recorded home can jump it across a gate it
// does not commute with, changing the circuit's function. This reproduces
// the regression: wide gates plus a low w_window make evasion fire on
// nearly every attempt, and an empty store makes every attempt FAIL, so the
// restore path runs constantly.
#[test]
fn failed_db_attempts_restore_seed_without_breaking_function() {
    let gates = random_mixed_circuit(97, 12, 400);
    let params = MixParams {
        k_max: 6,
        moves: 20_000,
        target_size: 400,
        temp: 20.0,
        p_db: 1.0, // every round is a slot-2 attempt
        db_mode: DbMode::Mix,
        p_convex: 0.5,
        w_window: 2, // width >= 2 is evaded: evasion on almost every build
        w_pool: 0,
        s_db: 5,
        db_min_window: 0,
        verify_every: 1_000, // global_check catches any functional drift
        report_every: u64::MAX,
        seed: 5,
        ..MixParams::default()
    };
    // Empty store: every attempt misses, so every attempt restores its seed.
    let mut mx = Mixer::new_with_db(gates, 12, params, FrozenDb::empty());
    mx.run();
    mx.global_check();
    assert!(mx.counters.db_build_aborts > 0 || mx.counters.db_attempts > 0);
}

// Pair geometry: with an empty store every
// attempt misses, so a pair round reduces to scan + fuse-float + the
// restore walk — all commutations. The function must survive, and the
// geometry must actually fire.
#[test]
fn pair_geometry_preserves_function_on_misses() {
    let gates = random_mixed_circuit(7, 16, 300);
    let params = MixParams {
        p_db: 1.0,
        p_pair: 1.0,
        s_db: 5,
        moves: 30_000,
        temp: 20.0,
        report_every: u64::MAX,
        verify_every: 5_000,
        seed: 11,
        ..MixParams::default()
    };
    let mut mx = Mixer::new_with_db(gates, 16, params, FrozenDb::empty());
    mx.run();
    assert!(mx.counters.pair_rounds > 0, "pair geometry never fired");
    assert!(mx.counters.pair_fused > 0, "no pair was ever fused");
    assert_eq!(mx.counters.pair_splices, 0, "empty store cannot splice");
    mx.global_check();
}

// collect_pair's window contract: two physically adjacent gates in link
// order that COMMUTE — the window shape the convex and contiguous
// samplers cannot produce — and the fusing floats must preserve the
// function.
#[test]
fn collect_pair_fuses_an_adjacent_commuting_pair() {
    let gates = random_mixed_circuit(13, 16, 200);
    let params = MixParams {
        p_pair: 1.0,
        report_every: u64::MAX,
        seed: 5,
        ..MixParams::default()
    };
    let mut mx = Mixer::new_with_db(gates, 16, params, FrozenDb::empty());
    let mut fused = 0usize;
    for _ in 0..200 {
        if let Some((ids, _dir)) = mx.collect_pair() {
            assert_eq!(ids.len(), 2, "a pair window is exactly two gates");
            assert_eq!(
                mx.arena.neighbor(ids[0], Dir::R),
                ids[1],
                "fused pair must be physically adjacent, leftmost first"
            );
            assert!(
                !mx.arena.collides_ids(ids[0], ids[1]),
                "a pair window is a COMMUTING pair by construction"
            );
            fused += 1;
        }
    }
    assert!(fused > 0, "no pair fused in 200 attempts");
    assert_eq!(fused as u64, mx.counters.pair_fused);
    mx.global_check();
}

// Bridge insertion exactness without any store: plan a bridge on a real
// random circuit, apply the insertions (wake corrections + the two
// carrier copies), and demand exact global functional equality — the
// telescoping identity g1·u·(u·M·u)·u·g2 = g1·M·g2, on material where
// the interior genuinely collides with the carrier.
#[test]
fn bridge_insertion_preserves_function() {
    let mut planned = 0usize;
    let mut with_wake = 0usize;
    for seed in 0..20u64 {
        let gates = random_mixed_circuit(100 + seed, 12, 160);
        let params = MixParams {
            bridge_min_span: 8,
            bridge_max_span: 64,
            bridge_max_colliders: 12,
            report_every: u64::MAX,
            seed: 40 + seed,
            ..MixParams::default()
        };
        let mut mx = Mixer::new_with_db(gates, 12, params, FrozenDb::empty());
        for _ in 0..40 {
            let Some(plan) = mx.bridge_plan() else {
                continue;
            };
            planned += 1;
            if !plan.wake.is_empty() {
                with_wake += 1;
            }
            mx.bridge_insert(&plan);
            mx.global_check();
        }
    }
    assert!(planned > 20, "too few plans succeeded: {planned}");
    assert!(
        with_wake > 5,
        "no plan ever needed a wake: colliders untested"
    );
}

// A bridge round against the empty store must leave literally no trace:
// both endpoint probes run before anything mutates.
#[test]
fn bridge_round_empty_store_leaves_no_trace() {
    let gates = random_mixed_circuit(23, 16, 200);
    let params = MixParams {
        bridge_min_span: 8,
        bridge_max_span: 64,
        report_every: u64::MAX,
        seed: 9,
        ..MixParams::default()
    };
    let mut mx = Mixer::new_with_db(gates.clone(), 16, params, FrozenDb::empty());
    let before = mx.arena.to_vec();
    for _ in 0..100 {
        mx.bridge_round();
    }
    assert_eq!(mx.counters.bridge_rounds, 100);
    assert_eq!(mx.counters.bridge_committed, 0, "empty store cannot commit");
    assert_eq!(
        mx.counters.bridge_rollbacks, 0,
        "probe precedes every insertion"
    );
    assert_eq!(
        mx.arena.to_vec(),
        before,
        "a probe miss must leave no trace"
    );
}
