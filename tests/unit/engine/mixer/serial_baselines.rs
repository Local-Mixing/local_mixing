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

// Re-baselined 2026-10-08 (twice): when the two curated counters and then
// the eight split-stage counters left the counters line. Each time the
// gate/meta trajectory was verified unchanged against the previous constants
// by re-inserting the retired (zero) fields.
const GOLDEN_PROFILE_NULL_PLANT: u64 = 0xc277215de595bf62;
const GOLDEN_THERMOSTAT_WALK: u64 = 0x85489b717f732b6e;

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
    assert_eq!(
        h, GOLDEN_PROFILE_NULL_PLANT,
        "serial profile trajectory drifted: got {h:#018x}"
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
