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
        (m.origin, m.dir == Dir::R, m.dgen, m.litter, m.litter_size).hash(&mut h);
    }
    mx.counters.to_line().hash(&mut h);
    mx.moves_done.hash(&mut h);
    h.finish()
}

// Re-baselined 2026-10-08 (three times): when the two curated counters,
// then the eight split-stage counters, then the crossing-walk counters and
// the per-gate event field left the hashed state. Each time the gate/meta/
// counter trajectory was verified unchanged against the previous constant by
// re-inserting the retired fields as zeros (step 4: old 0xc277215de595bf62
// reproduced exactly). The thermostat-walk golden is gone with the walk.
const GOLDEN_PROFILE_NULL_PLANT: u64 = 0xaed23f37b193b604;

#[test]
fn golden_profile_null_plant_is_stable() {
    let gates = random_mixed_circuit(41, 16, 400);
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
    let mut mx = Mixer::new_with_db(gates, 16, params, FrozenDb::empty());
    let stop = mx.run();
    assert!(matches!(stop, MixStop::ProfileDone));
    let h = golden_hash(&mx);
    assert_eq!(
        h, GOLDEN_PROFILE_NULL_PLANT,
        "serial profile trajectory drifted: got {h:#018x}"
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
// on the cleanup branch. TRAJECTORY_STAGE3_FIXTURE was also reproduced at
// ea2e516f (the pre-cleanup tip, with the in-memory fixture store backported
// and its stage-3 flags incl. p_comp 1.0 / p_any 0.1), so the cleanup left
// this trajectory unchanged.
const GOLDEN_STAGE3_FIXTURE: u64 = 0xb7c7ebe60b41be59;
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
    assert!(c.db_agn_hits > 0 && c.db_comp_hits > 0, "both DB branches must hit");
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
