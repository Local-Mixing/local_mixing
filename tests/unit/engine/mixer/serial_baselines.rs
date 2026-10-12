// ---- golden baselines for the serial (pieces = 1) path ----
//
// A serial run must follow the same byte-for-byte trajectory for the same
// seed and flags. These hashes only change when the walk itself is
// deliberately changed. The hash covers every gate in circuit order with
// its full Meta, the counters line and the move clock.
fn golden_hash(mx: &Mixer) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut h = DefaultHasher::new();
    for id in mx.arena.ids_in_order() {
        mx.arena.gate(id).hash(&mut h);
        let m = mx.meta_of(id);
        (
            m.origin,
            m.event,
            m.dir == Dir::R,
            m.dgen,
            m.litter,
            m.litter_size,
        )
            .hash(&mut h);
    }
    mx.counters.to_line().hash(&mut h);
    mx.moves_done.hash(&mut h);
    h.finish()
}

// Re-baselined 2026-10-08 (three times) when the curated, split-stage and
// crossing-walk fields left the hashed state (each time the trajectory was
// verified unchanged by re-inserting the retired fields as zeros), and
// re-pinned 2026-10-11 when the split stage and the crossing walk came back
// (state v6 = v3's layout: the pre-cleanup constant 0x8624858b7434eace).
// TRAJECTORY_PROFILE_NULL_PLANT (version-neutral, see trajectory_hash) is
// reproduced at ea2e516f by docs/notes/golden_repin_ea2e516f.patch; how to
// re-run it: docs/notes/golden_repin_check_20261008.md.
const GOLDEN_PROFILE_NULL_PLANT: u64 = 0x8624858b7434eace;
const TRAJECTORY_PROFILE_NULL_PLANT: u64 = 0xbaa42bedc74dc44e;
const GOLDEN_SPLIT_STOP: u64 = 0x32684402a67eeea0;
const GOLDEN_THERMOSTAT_WALK: u64 = 0x60a50a097a564f19;

#[test]
fn golden_profile_null_plant_is_stable() {
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
        prof_n: [6.0, 18.0, 34.0],
        prof_r: [4.0, 2.0],
        prof_cadence_eff: 0.5,
        report_every: u64::MAX,
        verify_every: 200_000,
        seed: 3,
        ..MixParams::default()
    };
    let mut mx = Mixer::new_with_db(gates, 16, params, FrozenDb::empty());
    let stop = mx.run();
    assert!(matches!(stop, MixStop::ProfileDone));
    let h = golden_hash(&mx);
    let t = trajectory_hash(&mut mx);
    assert_eq!(
        (h, t),
        (GOLDEN_PROFILE_NULL_PLANT, TRAJECTORY_PROFILE_NULL_PLANT),
        "serial profile trajectory drifted: got ({h:#018x}, {t:#018x})"
    );
}

// A version-neutral fingerprint of a trajectory: the gates and per-gate meta
// in circuit order, the move clock, the next draw of the main RNG and the
// counters of the kept stage-3 path -- nothing whose layout a cleanup may
// change. Used to check a re-pin against an older build.
fn trajectory_hash(mx: &mut Mixer) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut h = DefaultHasher::new();
    for id in mx.arena.ids_in_order() {
        mx.arena.gate(id).hash(&mut h);
        let m = mx.meta_of(id);
        (m.origin, m.dir == Dir::R, m.dgen, m.litter, m.litter_size).hash(&mut h);
    }
    mx.moves_done.hash(&mut h);
    let c = &mx.counters;
    (
        c.db_slot2_hits,
        c.db_agn_hits,
        c.db_comp_hits,
        c.db_gates_added,
        c.scatters,
        c.scatter_steps,
        c.twist_relabels,
    )
        .hash(&mut h);
    mx.rng.random::<u64>().hash(&mut h);
    h.finish()
}

// The stage-3 path with real mutations: DB splices against the g57 fixture
// store (MIX re-spellings and COMP contractions, db_advance birth transport,
// pay-random MIX, convex windows) plus twist-g57 brackets. Pinned 2026-10-08
// on the cleanup branch. TRAJECTORY_STAGE3_FIXTURE is also reproduced at
// ea2e516f (the pre-cleanup tip, with the in-memory fixture store backported
// and its stage-3 flags incl. p_comp 1.0 / p_any 0.1), so the cleanup left
// this trajectory unchanged. The backport and the command are committed:
// docs/notes/golden_repin_ea2e516f.patch, docs/notes/golden_repin_check_20261008.md.
// Re-pinned 2026-10-11 when the split and crossing counters came back into
// the hashed counters line (state v6); the version-neutral trajectory hash
// below was unchanged (old state hash 0xb7c7ebe60b41be59).
const GOLDEN_STAGE3_FIXTURE: u64 = 0x1d2d452400f2279a;
const TRAJECTORY_STAGE3_FIXTURE: u64 = 0xec5a286a7b0a742f;

fn stage3_fixture_params() -> MixParams {
    MixParams {
        k_max: 6,
        moves: 20_000,
        p_convex: 0.5,
        mix_pay_random: true,
        twist_g57: true,
        p_twist: 0.05,
        twist_min_len: 4,
        verify_every: 5_000,
        report_every: u64::MAX,
        seed: 11,
        ..db_walk_params()
    }
}

#[test]
fn golden_stage3_fixture_walk_is_stable() {
    let gates = random_g57_circuit(43, 16, 300);
    let mut mx = Mixer::new_with_db(gates, 16, stage3_fixture_params(), g57_fixture_store());
    mx.run();
    let c = &mx.counters;
    assert!(
        c.db_agn_hits > 0 && c.db_comp_hits > 0,
        "both DB branches must hit"
    );
    assert!(c.scatters > 0, "db_advance must move births");
    assert!(c.twist_relabels > 0, "twists must fire");
    mx.global_check();
    let g = golden_hash(&mx);
    let t = trajectory_hash(&mut mx);
    assert_eq!(
        (g, t),
        (GOLDEN_STAGE3_FIXTURE, TRAJECTORY_STAGE3_FIXTURE),
        "stage-3 fixture trajectory drifted: got ({g:#018x}, {t:#018x})"
    );
}

#[test]
fn golden_split_stop_is_stable() {
    let gates = random_mixed_circuit(11, 16, 400);
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
    assert!(matches!(stop, MixStop::SplitDone));
    let h = golden_hash(&mx);
    assert_eq!(
        h, GOLDEN_SPLIT_STOP,
        "serial split-stage trajectory drifted: got {h:#018x}"
    );
}

#[test]
fn golden_thermostat_walk_is_stable() {
    let gates = random_mixed_circuit(37, 16, 200);
    let params = MixParams {
        k_max: 5,
        moves: 4_000,
        target_size: 260,
        temp: 20.0,
        p_twist: 0.05,
        w_twist_neg: 1.0,
        gen_target: 3,
        verify_every: 1_000,
        report_every: u64::MAX,
        seed: 11,
        ..MixParams::default()
    };
    let mut mx = Mixer::new_with_db(gates, 16, params, FrozenDb::empty());
    let stop = mx.run();
    assert!(matches!(stop, MixStop::MovesBudget));
    let h = golden_hash(&mx);
    assert_eq!(
        h, GOLDEN_THERMOSTAT_WALK,
        "serial thermostat trajectory drifted: got {h:#018x}"
    );
}
