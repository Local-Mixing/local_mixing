// Even without DB moves the walk lifts generations: split children get
// parent + 1, so heavy churn mints intermediate generations strictly
// between 0 and GEN_FRESH, while fresh material stays at GEN_FRESH.
#[test]
fn walk_splits_lift_generations() {
    let gates = random_mixed_circuit(29, 16, 300);
    let params = MixParams {
        k_max: 5,
        moves: 20_000,
        target_size: 600,
        temp: 20.0,
        p_twist: 0.1, // twist brackets are the only born-random source now
        w_twist_neg: 0.05,
        w_twist_swap: 0.05,
        verify_every: 5_000,
        report_every: u64::MAX,
        seed: 13,
        ..MixParams::default()
    };
    let mut mx = Mixer::new(gates, 16, params);
    mx.run();
    let gens = mx.gens_in_order();
    assert!(
        gens.iter().any(|&g| g > 0 && g != GEN_FRESH),
        "split children must climb above generation 0"
    );
    assert!(
        gens.contains(&GEN_FRESH),
        "twist brackets must be marked fresh"
    );
    let s = mx.gen_stats();
    assert_eq!(s.total as usize, gens.len());
}

// The inherit split-rule variant: without DB moves nothing ever
// increments, so after the same heavy churn every gate is either
// still-original material (gen 0) or born-random (GEN_FRESH) — the
// clean isolation of DB re-encoding depth from walk rewrite depth.
#[test]
fn inherit_split_rule_keeps_gens_binary_without_db() {
    let gates = random_mixed_circuit(29, 16, 300);
    let params = MixParams {
        k_max: 5,
        moves: 20_000,
        target_size: 600,
        temp: 20.0,
        w_twist_neg: 0.05,
        w_twist_swap: 0.05,
        gen_split_inherit: true,
        verify_every: 5_000,
        report_every: u64::MAX,
        seed: 13,
        ..MixParams::default()
    };
    let mut mx = Mixer::new(gates, 16, params);
    mx.run();
    let gens = mx.gens_in_order();
    assert!(
        gens.iter().all(|&g| g == 0 || g == GEN_FRESH),
        "under inherit semantics only DB splices may mint generations"
    );
    assert!(
        gens.contains(&0),
        "original material cannot all vanish here"
    );
}

// The generation pool: with p_mingen 1.0 the seed comes from the pool
// while one exists, and entries that got re-encoded (or freed) between
// rebuilds are pruned at draw time.
#[test]
fn pick_seed_targets_pool_and_prunes_stale() {
    let gates = random_mixed_circuit(31, 16, 60);
    let params = MixParams {
        gen_target: 4,
        p_mingen: 1.0,
        w_pool: 0,
        report_every: u64::MAX,
        seed: 17,
        ..MixParams::default()
    };
    let mut mx = Mixer::new(gates, 16, params);
    let ids = mx.arena.ids_in_order();
    // Everyone re-encoded past target except three laggards.
    let lag: Vec<u32> = vec![ids[5], ids[20], ids[40]];
    for &id in &ids {
        let m = mx.meta_of(id);
        let g = if lag.contains(&id) { 1 } else { 9 };
        mx.set_meta(id, Meta { dgen: g, ..m });
    }
    mx.rebuild_pool();
    assert_eq!(mx.pool.len(), 3);
    for _ in 0..50 {
        let s = mx.pick_seed().expect("seed");
        assert!(
            lag.contains(&s),
            "p_mingen 1.0 must draw pool seeds while any exist"
        );
    }
    // One crosses the target between rebuilds: draws prune it.
    let m = mx.meta_of(lag[0]);
    mx.set_meta(lag[0], Meta { dgen: 4, ..m });
    for _ in 0..200 {
        let s = mx.pick_seed().expect("seed");
        assert!(
            s != lag[0],
            "re-encoded gate must not be picked from the pool"
        );
    }
    assert_eq!(mx.pool.len(), 2, "stale entry must be pruned at draw time");
}

// The pool is capped at pool_k, holding the LOWEST-generation gates: the
// count is what bounds the drain between rebuilds, so it must be honoured
// exactly rather than approximately.
#[test]
fn pool_keeps_only_k_lowest_generations() {
    let gates = random_mixed_circuit(43, 16, 60);
    let params = MixParams {
        gen_target: 100,
        pool_k: 5,
        w_pool: 0,
        report_every: u64::MAX,
        seed: 23,
        ..MixParams::default()
    };
    let mut mx = Mixer::new(gates, 16, params);
    let ids = mx.arena.ids_in_order();
    for (i, &id) in ids.iter().enumerate() {
        let m = mx.meta_of(id);
        mx.set_meta(
            id,
            Meta {
                dgen: i as u32,
                ..m
            },
        );
    }
    mx.rebuild_pool();
    assert_eq!(mx.pool.len(), 5, "pool must be capped at pool_k");
    let mut gens: Vec<u32> = mx.pool.iter().map(|&id| mx.meta_of(id).dgen).collect();
    gens.sort_unstable();
    assert_eq!(
        gens,
        vec![0, 1, 2, 3, 4],
        "pool must hold the K lowest generations"
    );
}

#[test]
fn dose_stop_and_generation_count_targetable_gates() {
    let gates = random_mixed_circuit(37, 16, 40);
    let params = MixParams {
        gen_target: 2,
        gen_stop_frac: 0.0,
        w_window: 3,
        w_pool: 3,
        report_every: u64::MAX,
        ..MixParams::default()
    };
    let mut mx = Mixer::new(gates, 16, params);
    assert!(!mx.dose_reached(), "all-gen-0 input cannot be at dose");
    assert_eq!(mx.gen_stats().g_circ, 0);
    let ids = mx.arena.ids_in_order();
    let narrow: Vec<u32> = ids
        .iter()
        .copied()
        .filter(|&id| mx.width_of(id) <= 2)
        .collect();
    let wide: Vec<u32> = ids
        .iter()
        .copied()
        .filter(|&id| mx.width_of(id) > 2)
        .collect();
    assert!(
        !narrow.is_empty() && !wide.is_empty(),
        "fixture must contain both eligible and cap-ineligible gates"
    );
    for &id in &ids {
        let m = mx.meta_of(id);
        mx.set_meta(id, Meta { dgen: 3, ..m });
    }
    let s = mx.gen_stats();
    assert_eq!(s.all_lag, 0);
    assert_eq!(s.targetable, s.elig, "nothing written off yet");
    assert_eq!(s.g_circ, 3, "everything at 3 -> circuit generation 3");
    assert!(mx.dose_reached());

    // A WIDE straggler must not block the stop nor sink the generation:
    // the DB channel can never re-encode it, so it would pin both forever.
    let m = mx.meta_of(wide[0]);
    mx.set_meta(wide[0], Meta { dgen: 0, ..m });
    let s = mx.gen_stats();
    assert_eq!(s.wlag, 1, "the wide gate lags");
    assert_eq!(s.lag, 0, "...but is not targetable");
    assert_eq!(s.g_circ, 3, "so the circuit generation is untouched");
    assert!(mx.dose_reached(), "a wide laggard cannot block the dose");
    // (One straggler in 40 is 2.5%, inside g_all's own 5% allowance, so
    // g_all does not move here either. The divergence between the two
    // shows up once the wide laggards exceed 5% — see the next test.)

    // An ELIGIBLE straggler does block at frac 0, and is inside a 5%
    // allowance.
    let m = mx.meta_of(narrow[0]);
    mx.set_meta(narrow[0], Meta { dgen: 0, ..m });
    assert!(!mx.dose_reached(), "an eligible laggard blocks at frac 0");
    let s = mx.gen_stats();
    assert_eq!(s.lag, 1);
    mx.params.gen_stop_frac = 1.0 / s.targetable as f64;
    assert!(mx.dose_reached(), "one laggard is within its own fraction");

    // Coverage requirement gates the stop until twists supply it.
    mx.params.twist_cov_stop = 10.0;
    assert!(!mx.dose_reached());
    mx.counters.twist_span = (mx.arena.len() as u64) * 11;
    assert!(mx.dose_reached());
    // Off switches.
    mx.params.gen_stop_frac = -1.0;
    assert!(!mx.dose_reached());
}

// On majority-wide material, generation and dose statistics must use
// eligible gates. Including cap-ineligible gates would pin the percentile
// at zero even when every eligible gate has received the required dose.
#[test]
fn wide_majority_does_not_pin_the_generation_or_the_dose() {
    // 30 wide (3-control) gates + 10 narrow: 75% cap-ineligible, far past
    // the 5% allowance, so the all-gates percentile is stuck at 0.
    let mut gates: Vec<XGate> = Vec::new();
    let mut rng = StdRng::seed_from_u64(4242);
    while gates.len() < 30 {
        let g = rand_gate(&mut rng, 16, 3, false);
        if g.width() == 3 {
            gates.push(g);
        }
    }
    while gates.len() < 40 {
        let g = rand_gate(&mut rng, 16, 2, false);
        if g.width() <= 2 && g.width() >= 1 {
            gates.push(g);
        }
    }
    let params = MixParams {
        gen_target: 100,
        gen_stop_frac: 0.02,
        w_window: 3,
        w_pool: 3,
        report_every: u64::MAX,
        ..MixParams::default()
    };
    let mut mx = Mixer::new(gates, 16, params);
    // Drive every ELIGIBLE gate past the target, as a completed dose does.
    for &id in &mx.arena.ids_in_order() {
        if mx.width_of(id) <= 2 {
            let m = mx.meta_of(id);
            mx.set_meta(id, Meta { dgen: 100, ..m });
        }
    }
    let s = mx.gen_stats();
    assert!(s.wlag >= 30, "the wide majority still lags by construction");
    assert_eq!(s.lag, 0, "but the dose over eligible gates is complete");
    assert_eq!(
        s.g_all, 0,
        "the all-gates percentile is pinned at 0 (the bug)"
    );
    assert_eq!(
        s.g_circ, 100,
        "the circuit generation reflects the real dose"
    );
    assert!(
        mx.dose_reached(),
        "and the dose stop fires instead of burning the whole move budget"
    );
}
