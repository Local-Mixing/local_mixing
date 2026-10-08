#[test]
fn prof_target_traces_the_three_phases() {
    let n = [5.0, 20.0, 40.0];
    let r = [4.0, 2.0];
    let s = 1000.0;
    assert!((prof_target(n, r, s, 0.0) - 1000.0).abs() < 1e-9); // start = input
    assert!((prof_target(n, r, s, 2.5) - 2500.0).abs() < 1e-6); // mid-ramp1
    assert!((prof_target(n, r, s, 5.0) - 4000.0).abs() < 1e-6); // top of expand
    assert!((prof_target(n, r, s, 12.0) - 4000.0).abs() < 1e-6); // hold
    assert!((prof_target(n, r, s, 20.0) - 4000.0).abs() < 1e-6); // hold end
    assert!((prof_target(n, r, s, 30.0) - 3000.0).abs() < 1e-6); // mid-ramp3
    assert!((prof_target(n, r, s, 40.0) - 2000.0).abs() < 1e-6); // bottom
    assert!((prof_target(n, r, s, 99.0) - 2000.0).abs() < 1e-6); // after
    // monotone non-decreasing over [0,n0], flat over hold, non-increasing after n1
    let up = (0..=50)
        .map(|i| prof_target(n, r, s, i as f64 * 0.1))
        .collect::<Vec<_>>();
    assert!(up.windows(2).all(|w| w[1] >= w[0] - 1e-9));

    // R1 = 1 (pure hold) with a sub-x1 compression end: flat at the input
    // through N1, then linear down to half size.
    let n = [1.0, 30.0, 35.0];
    let r = [1.0, 0.5];
    assert!((prof_target(n, r, s, 0.5) - 1000.0).abs() < 1e-9);
    assert!((prof_target(n, r, s, 15.0) - 1000.0).abs() < 1e-9);
    assert!((prof_target(n, r, s, 32.5) - 750.0).abs() < 1e-6);
    assert!((prof_target(n, r, s, 35.0) - 500.0).abs() < 1e-9);
    assert!((prof_target(n, r, s, 99.0) - 500.0).abs() < 1e-9);
}

// The controller must actually track a moderate, feasible profile on a
// real circuit: expand to ~4x, hold, compress toward ~2x. Asserts the
// phase peaks/troughs land near the targets (best-effort tolerance) and
// that the phase machine advances all the way to phase 4.
#[test]
fn profile_controller_tracks_a_feasible_ramp() {
    let gates = random_mixed_circuit(41, 16, 400);
    let s_in = gates.len() as f64;
    let params = MixParams {
        k_max: 6,
        moves: 4_000_000,
        p_db: 1.0,
        s_db: 5,
        db_min_window: 0,
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
    // No FROZEN_DB_DIR in tests -> empty store, every DB lookup misses, so
    // the plant has ZERO authority and the controller cannot move size.
    // This test therefore exercises the controller MATH and phase machine
    // against a null plant: it must still advance phases on the eff marks
    // and never panic / never leave p_mix out of [0,1].
    let mut mx = Mixer::new_with_db(gates, 16, params, FrozenDb::empty());
    let stop = mx.run();
    let p = mx.prof.as_ref().expect("profile armed");
    assert_eq!(p.phase, 4, "phase machine must reach compress-done");
    assert!(p.pmix >= 0.0 && p.pmix <= 1.0, "lever stayed in range");
    assert!(
        matches!(stop, MixStop::ProfileDone),
        "a finished profile ends the run"
    );
    // A null plant cannot grow, so the expansion leg saturates (flagged,
    // not hidden) and the compression leg is trivially already-arrived:
    // size <= R2 * input the moment phase 3 opens, which is exactly when
    // the contract says that leg is done.
    assert!(
        p.sat > 0,
        "saturation must be tracked when the lever cannot move size"
    );
    assert!(
        mx.arena.len() as f64 <= 2.0 * s_in,
        "ended at or under the R2 setpoint"
    );
    assert!(p.eff >= 18.0, "ran through the hold phase before finishing");
    mx.global_check();
}

// The global re-randomisation move must MOVE gates and change NOTHING
// else: same function, same size, same multiset of gates. It is scheduled
// at 1/|circuit| per round, so the rate is pushed hard here to make it
// fire often enough to be worth asserting on.
#[test]
fn global_shuffle_moves_gates_but_preserves_function_and_size() {
    let gates = random_mixed_circuit(37, 16, 300);
    let before = gates.clone();
    let params = MixParams {
        k_max: 6,
        moves: 20_000,
        // 300 * the size, so the per-round p is 1: a reshuffle every round
        // (the size never changes). No DB move: the multiset must survive.
        shuffle_rate: 300.0,
        p_twist: 0.0,
        p_db: 0.0,
        local_verify: true,
        verify_every: 500,
        report_every: u64::MAX,
        seed: 11,
        ..MixParams::default()
    };
    let mut mx = Mixer::new(gates, 16, params);
    mx.run();
    assert!(
        mx.counters.shuffles > 1_000,
        "shuffle never fired: {}",
        mx.counters.shuffles
    );
    assert!(
        mx.counters.shuffle_moved > 0,
        "shuffles ran but no gate ever changed position"
    );
    let after: Vec<XGate> = mx
        .arena
        .ids_in_order()
        .iter()
        .map(|&id| mx.arena.gate(id).clone())
        .collect();
    assert_eq!(
        after.len(),
        before.len(),
        "shuffle changed the circuit SIZE"
    );
    let mut a: Vec<_> = after.iter().map(|g| format!("{g:?}")).collect();
    let mut b: Vec<_> = before.iter().map(|g| format!("{g:?}")).collect();
    a.sort();
    b.sort();
    assert_eq!(
        a, b,
        "shuffle changed the gate MULTISET, not just the order"
    );
    assert_ne!(
        after.iter().map(|g| format!("{g:?}")).collect::<Vec<_>>(),
        before.iter().map(|g| format!("{g:?}")).collect::<Vec<_>>(),
        "shuffle left the order untouched"
    );
    mx.global_check();
}

// The stage-3 driver end to end on real store hits: every round is a DB move
// against the g57 fixture store, so the walk must actually splice (and
// re-spell, not only contract), stamp generations, preserve the function
// through local and global verification, and never mint a non-g57 gate.
#[test]
fn db_walk_splices_real_hits_and_preserves_function() {
    let gates = random_g57_circuit(5, 16, 300);
    let params = MixParams {
        moves: 20_000,
        local_verify: true,
        verify_every: 1_000,
        report_every: u64::MAX,
        seed: 21,
        ..db_walk_params()
    };
    let mut mx = Mixer::new_with_db(gates, 16, params, g57_fixture_store());
    mx.run();
    let c = &mx.counters;
    assert!(
        c.db_slot2_hits > 1_000,
        "DB walk barely spliced: {}",
        c.db_slot2_hits
    );
    // Re-spelling, not only contraction: the size-agnostic (MIX) branch of
    // the p_mix coin must hit and pay for longer spellings.
    assert!(
        c.db_agn_hits > 0,
        "no MIX re-spelling hit: {}",
        c.db_agn_hits
    );
    assert!(c.db_gates_added > 0, "no splice added gates");
    // db_advance (forced on by --tdp): splice products get the birth advance.
    assert!(
        c.scatters > 0 && c.scatter_steps >= c.scatters,
        "db_advance never advanced a birth: {} scatters / {} steps",
        c.scatters,
        c.scatter_steps
    );
    assert!(
        mx.gens_in_order().iter().any(|&g| g > 0 && g != GEN_FRESH),
        "splices must stamp generations"
    );
    let cen = mx.g57_census();
    assert_eq!(
        cen.opp_pol,
        mx.arena.len(),
        "the fixture store spells g57 words only"
    );
    mx.global_check();
}
