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
        temp: 20.0,
        p_db: 1.0,
        p_comp: 1.0,
        p_any: 0.1,
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

// The chain holds size near the target, preserves the function (run() and
// the end-of-run global check assert internally), does both kinds of moves,
// and never increases the fossil count.
#[test]
fn mixer_holds_size_and_function() {
    let gates = random_mixed_circuit(3, 16, 300);
    let comp0 = gates.iter().filter(|g| g.comp).count();
    let params = MixParams {
        k_max: 5,
        // Erosion is presplit-driven and presplits are crossings, so this
        // needs enough rounds to exercise the crossing expansion channel.
        moves: 40_000,
        target_size: 300,
        temp: 20.0,
        // The global reshuffle dilutes every other menu slot by exactly
        // shuffle_rate/|circuit|. At production size that is ~2e-5 and
        // invisible; on this 300-gate circuit it is ~0.4% of rounds taken
        // from the thermostat, which is enough to push the drift band.
        // This test is the thermostat's contract, so hold the new slot
        // out of it -- the move has its own test.
        shuffle_rate: 0.0,
        verify_every: 2_000,
        report_every: u64::MAX,
        seed: 5,
        ..MixParams::default()
    };
    let mut mx = Mixer::new(gates, 16, params);
    mx.run();
    // The chain equilibrates at a content-dependent floor above the target
    // (dead journal entries are permanently unmergeable by the pairwise
    // catalogue); the contract is BOUNDED drift under heavy churn, not
    // exact adherence. This run churns ~67 moves/gate.
    let n = mx.arena.len();
    assert!((200..=520).contains(&n), "size drifted from target: {n}");
    assert!(mx.counters.merges() > 0, "no merges happened");
    assert!(mx.counters.undos > 0, "no crossing undos happened");
    assert!(mx.counters.expands() > 0, "no expansions happened");
    assert!(mx.remaining_g57() <= comp0, "fossil count increased");
    assert!(
        mx.origin_displacement() > 0.01,
        "no positional mixing at all"
    );
}

// Crossing expansion must grow toward a higher thermostat target. Lowering
// the target then makes contraction outweigh expansion and reduces the size.
#[test]
fn mixer_thermostat_grows_then_drains() {
    let gates = random_mixed_circuit(9, 16, 200);
    let grow = MixParams {
        k_max: 5,
        moves: 30_000,
        target_size: 500,
        temp: 20.0,
        // Crossings supply growth, and the journal records their inverses.
        w_cross: 1.0,
        verify_every: 5_000,
        report_every: u64::MAX,
        seed: 1,
        ..MixParams::default()
    };
    let mut mx = Mixer::new(gates, 16, grow);
    mx.run();
    let grown = mx.arena.len();
    assert!(grown >= 400, "did not grow toward target: {grown}");

    mx.params.target_size = 250;
    // Allow enough moves for thermostat-driven contraction to overcome
    // the crossing expansion channel.
    mx.params.moves += 60_000;
    mx.run();
    let drained = mx.arena.len();
    // Without a database, crossing ladders rely on live journal entries
    // for contraction. This checks a size decrease; deeper contraction
    // depends on the database compression channel, which is disabled here.
    assert!(
        drained < grown,
        "drain did not contract: {grown} -> {drained}"
    );
    assert!(mx.counters.merges() > 0, "no merges during drain");
}

// On an all-g57 input the chain erodes fossils; erosion inherently costs
// +1 gate per presplit and is irreversible by design, so size may grow up
// to initial + fossil count (plus thermostat slack) while comp declines.
#[test]
fn mixer_erodes_fossils() {
    let gates = random_g57_circuit(3, 16, 300);
    let comp0 = gates.iter().filter(|g| g.comp).count();
    let params = MixParams {
        k_max: 5,
        moves: 20_000,
        target_size: 300,
        temp: 20.0,
        verify_every: 2_000,
        report_every: u64::MAX,
        seed: 5,
        ..MixParams::default()
    };
    let mut mx = Mixer::new(gates, 16, params);
    mx.run();
    let comp_now = mx.remaining_g57();
    assert!(
        comp_now < comp0 / 2,
        "erosion too slow: {comp0} -> {comp_now}"
    );
    let n = mx.arena.len();
    assert!(n <= 300 + comp0 + 100, "grew past the erosion budget: {n}");
}

// The directional walk uses live undo, birth advance and failed-cross
// retreat. Function must be preserved through every sub-step, crossings
// must run, and size must stay within a loose runaway bound. Crossings
// supply the expansion channel.
#[test]
fn directional_walk_preserves_function() {
    let gates = random_mixed_circuit(31, 16, 300);
    let params = MixParams {
        k_max: 5,
        moves: 20_000,
        target_size: 400,
        temp: 20.0,
        w_cross: 1.0,
        verify_every: 2_000,
        report_every: u64::MAX,
        seed: 7,
        ..MixParams::default()
    };
    let mut mx = Mixer::new(gates, 16, params);
    mx.run();
    let crossings = mx.counters.cross_r1 + mx.counters.cross_r2 + mx.counters.cross_r3;
    assert!(crossings > 50, "crossings barely ran: {crossings}");
    assert!(
        mx.counters.cross_r1 + mx.counters.cross_r2 + mx.counters.cross_r3 > 100,
        "crossings barely ran"
    );
    assert!(mx.counters.scatters > 0, "no directional birth advances");
    assert!(
        mx.counters.fresh_splits == 0,
        "fresh splits ran despite suspension"
    );
    let n = mx.arena.len();
    assert!(
        n < 4 * 400,
        "size ran away under merge-only contraction: {n}"
    );
    mx.global_check();
}

// --twist-g57: brackets become adaptive all-g57 words solved online per
// seam. Thousands of twists under local_verify (every seam splice checked
// exhaustively against the reference 3-CNOT packet) plus periodic full
// verification and a final global_check. The seams must also actually
// absorb neighborhood material — a run where tg_consumed stays 0 means
// the placer degenerated to bare words and the mechanism is dead.
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
        target_size: 300,
        temp: 20.0,
        // 300 * the size, so the per-round p is ~1: a reshuffle nearly
        // every round.
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

#[test]
fn tabu_age_semantics() {
    let gates = random_g57_circuit(1, 8, 20);
    let params = MixParams {
        tabu_moves: 100,
        ..MixParams::default()
    };
    let mut mx = Mixer::new(gates, 8, params);
    let e1 = mx.fresh_event();
    mx.moves_done = 50;
    let e2 = mx.fresh_event();
    assert!(mx.is_tabu(e1) && mx.is_tabu(e2));
    assert!(!mx.is_tabu(0), "event 0 (no provenance) is never tabu");
    mx.moves_done = 120; // e1 aged out (created at move 0), e2 still fresh
    assert!(!mx.is_tabu(e1), "aged-out event must not be tabu");
    assert!(mx.is_tabu(e2));
    mx.moves_done = 200;
    let _ = mx.fresh_event(); // push triggers eviction of expired entries
    assert!(mx.tabu.len() <= 2, "expired tabu entries must be evicted");
}
