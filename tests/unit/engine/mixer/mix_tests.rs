use super::*;
use crate::circuit::xgate::XGate;

// ---- golden baselines for the serial (pieces = 1) path ----
//
// Recorded on the tree BEFORE the piecewise-parallel engine edits
// (2026-09-07). The contract is that a serial run is byte-for-byte the
// same trajectory for the same seed and flags, so these hashes never
// move unless the walk itself is deliberately changed. The hash covers
// every gate in circuit order with its full Meta, the counters line and
// the move clock.
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

const GOLDEN_PROFILE_NULL_PLANT: u64 = 0xa5a0889c5985fa88;
const GOLDEN_SPLIT_STOP: u64 = 0x2b4e93a0d195a7c2;
const GOLDEN_THERMOSTAT_WALK: u64 = 0xbc178ac2df496fe3;

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

// ---- piecewise-parallel rounds (mix/piecewise.rs) ----

fn piece_cfg(pieces: usize, sequential: bool, threads: usize) -> PieceCfg {
    PieceCfg {
        pieces,
        sequential,
        threads,
        ..PieceCfg::default()
    }
}

fn meta_snapshot(mx: &Mixer) -> Vec<(u32, u64, bool, u32, u64, u16)> {
    mx.arena
        .ids_in_order()
        .iter()
        .map(|&id| {
            let m = mx.meta_of(id);
            (
                m.origin,
                m.event,
                m.dir == Dir::R,
                m.dgen,
                m.litter,
                m.litter_size,
            )
        })
        .collect()
}

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

    let path = std::env::temp_dir().join("fmix_piecewise_split.state");
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

fn rand_gate(rng: &mut StdRng, wires: u16, max_w: usize, allow_comp: bool) -> XGate {
    loop {
        let target = rng.random_range(0..wires);
        let w = rng.random_range(0..=max_w);
        let lits: Vec<(u16, bool)> = (0..w)
            .map(|_| (rng.random_range(0..wires), rng.random_bool(0.5)))
            .filter(|&(cw, _)| cw != target)
            .collect();
        if let Some(mut g) = XGate::conj(target, lits) {
            if allow_comp && g.width() == 2 && rng.random_bool(0.3) {
                g.comp = true;
            }
            return g;
        }
    }
}

// The mask-algebra collides predicate (arena.rs GateMask/mask_collides,
// behind Arena::collides_ids) must equal XGate::collides on every pair.
// Wire universes rotate through dense-low (shared control wires and the
// opposite-polarity separation exemption fire constantly), mid, one-word
// boundary, and > 64 (exercises the second mask word). g57s (comp = true)
// come from rand_gate's allow_comp draw.
#[test]
fn mask_collides_matches_xgate_collides() {
    use super::super::arena::{Arena, GateMask, MASK_WORDS};
    let mut rng = StdRng::seed_from_u64(0x2026_0809);
    let mut collided = 0usize;
    let mut separated = 0usize;
    for i in 0..1_000_000usize {
        let wires: u16 = match i % 4 {
            0 => 5,
            1 => 16,
            2 => 64,
            _ => 127,
        };
        let g = rand_gate(&mut rng, wires, 4, true);
        let h = rand_gate(&mut rng, wires, 4, true);
        let mg = GateMask::of(&g).expect("wire < 64 * MASK_WORDS has a mask");
        let mh = GateMask::of(&h).expect("wire < 64 * MASK_WORDS has a mask");
        let want = XGate::collides(&g, &h);
        assert_eq!(
            Arena::mask_collides(&mg, &mh),
            want,
            "mask mismatch: {g:?} vs {h:?}"
        );
        if want {
            collided += 1;
        } else if g.reads(h.target) || h.reads(g.target) {
            separated += 1; // commuted only via the polarity exemption
        }
    }
    // The draw must actually exercise both hard branches.
    assert!(collided > 10_000, "too few colliding pairs: {collided}");
    assert!(
        separated > 1_000,
        "too few exemption-separated pairs: {separated}"
    );
    // Out-of-range wires have no mask: collides_ids falls back to
    // XGate::collides (arena poisons masks_ok on such an alloc).
    let lim = (64 * MASK_WORDS) as u16;
    assert!(GateMask::of(&XGate::cnot(0, lim)).is_none());
    assert!(GateMask::of(&XGate::cnot(lim, 0)).is_none());
    assert!(GateMask::of(&XGate::cnot(0, lim - 1)).is_some());
}

// Every merge the catalogue accepts is a verified identity with comp=0
// output; the presplit-pair rejoin (complemented result) is rejected.
#[test]
fn merge_catalogue_sound_and_comp_guarded() {
    let mut rng = StdRng::seed_from_u64(7);
    let mut accepted = 0usize;
    for _ in 0..20_000 {
        let g = rand_gate(&mut rng, 6, 3, true);
        let h = rand_gate(&mut rng, 6, 3, true);
        if let Some(m) = merge_result(&g, &h) {
            let out = m.gates();
            assert!(
                out.iter().all(|x| !x.comp),
                "merge emitted comp: {g:?}+{h:?}"
            );
            assert!(
                rules::verify_rewrite(&[g.clone(), h.clone()], &out),
                "unsound merge: {g:?} + {h:?} -> {out:?}"
            );
            accepted += 1;
        }
    }
    assert!(
        accepted > 50,
        "catalogue accepted too few pairs to be tested: {accepted}"
    );

    // The presplit pieces of a g57 (x and !x!y on the same target) XOR to
    // the complemented parent: must be rejected.
    let p0 = XGate::conj(0, [(1u16, true)]).unwrap();
    let p1 = XGate::conj(0, [(1u16, false), (2u16, false)]).unwrap();
    assert!(
        merge_result(&p0, &p1).is_none(),
        "presplit rejoin must be comp-guarded"
    );

    // Two g57s differing in one polarity fuse into a conjunction (fossil
    // erosion), and a g57 plus its own monomial fuse into a NOT gate.
    let g57a = XGate {
        target: 0,
        comp: true,
        ctrls: p1.ctrls.clone(),
    };
    let mut g57b = g57a.clone();
    g57b.ctrls[0].1 = true;
    match merge_result(&g57a, &g57b) {
        Some(Merge::DropLit(m)) => assert!(!m.comp && m.width() == 1),
        other => panic!(
            "comp-comp polarity pair should DropLit, got {:?}",
            other.map(|m| m.gates())
        ),
    }
    let mono = XGate {
        target: 0,
        comp: false,
        ctrls: g57a.ctrls.clone(),
    };
    match merge_result(&g57a, &mono) {
        Some(Merge::XFuse(m)) => assert!(!m.comp && m.width() == 0),
        other => panic!(
            "g57 + own monomial should XFuse, got {:?}",
            other.map(|m| m.gates())
        ),
    }
}

// fresh-wire split and unsubsume each round-trip through the catalogue back
// to the exact original gate.
#[test]
fn split_merge_roundtrips() {
    let mut rng = StdRng::seed_from_u64(11);
    for _ in 0..2_000 {
        let g = rand_gate(&mut rng, 8, 3, false);
        // fresh split on a wire the gate does not touch
        let x = (0..8u16).find(|&w| w != g.target && !g.reads(w)).unwrap();
        let a = XGate::conj(g.target, g.ctrls.iter().copied().chain([(x, true)])).unwrap();
        let b = XGate::conj(g.target, g.ctrls.iter().copied().chain([(x, false)])).unwrap();
        match merge_result(&a, &b) {
            Some(Merge::DropLit(m)) => assert_eq!(m, g),
            _ => panic!("fresh-split pieces must DropLit back to the parent"),
        }
        // unsubsume round-trip
        if g.width() > 0 {
            let (w, p) = g.ctrls[rng.random_range(0..g.ctrls.len())];
            let without = XGate::conj(g.target, g.ctrls_without(w)).unwrap();
            let flipped = XGate::conj(
                g.target,
                g.ctrls
                    .iter()
                    .map(|&(cw, cp)| if cw == w { (cw, !cp) } else { (cw, cp) }),
            )
            .unwrap();
            let _ = p;
            match merge_result(&without, &flipped) {
                Some(Merge::Subsume(m)) => assert_eq!(m, g),
                _ => panic!("unsubsume pieces must Subsume back to the parent"),
            }
        }
    }
}

fn random_g57_circuit(seed: u64, wires: u16, gates: usize) -> Vec<XGate> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..gates)
        .map(|_| {
            loop {
                let a = rng.random_range(0..wires);
                let x = rng.random_range(0..wires);
                let y = rng.random_range(0..wires);
                if a != x && a != y && x != y {
                    break XGate::from_g57([a, x, y]);
                }
            }
        })
        .collect()
}

// A conjunction-dominated circuit like real fmix input (fsplit output is
// ~90% eroded); a few g57 fossils sprinkled in.
fn random_mixed_circuit(seed: u64, wires: u16, gates: usize) -> Vec<XGate> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..gates)
        .map(|i| {
            if i % 10 == 0 {
                loop {
                    let a = rng.random_range(0..wires);
                    let x = rng.random_range(0..wires);
                    let y = rng.random_range(0..wires);
                    if a != x && a != y && x != y {
                        break XGate::from_g57([a, x, y]);
                    }
                }
            } else {
                loop {
                    let g = rand_gate(&mut rng, wires, 3, false);
                    if g.width() >= 1 {
                        break g;
                    }
                }
            }
        })
        .collect()
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
        // needs enough rounds now that crossings are the only expansion.
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

// Thermostat up (catalogue-invertible moves only: fresh/unsub/insert, no
// crossings), then pure-drain mode down: with expansion weights zeroed the
// merge channel alone must dig most of the growth back out. Fresh-split
// trees are hierarchically recoverable (child pairs DropLit-merge back
// into parents, which can then merge with THEIR siblings), so this stock
// stays recyclable — unlike crossing ladders, which are permanent once
// their journal entries die (measured recoverable slack there: ~6%).
#[test]
fn mixer_thermostat_grows_then_drains() {
    let gates = random_mixed_circuit(9, 16, 200);
    let grow = MixParams {
        k_max: 5,
        moves: 30_000,
        target_size: 500,
        temp: 20.0,
        // Catalogue-invertible expansion no longer exists: fresh, unsub and
        // insert are retired, so crossings are the only growth channel and
        // the journal, not the pairwise catalogue, is what reverses them.
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
    // Expansion is cross-only now, so the drain is the thermostat pushing
    // contraction against crossings rather than against catalogue-invertible
    // stock. Give it the moves that costs.
    mx.params.moves += 60_000;
    mx.run();
    let drained = mx.arena.len();
    // Only a modest drain is available here, and that is the design rather
    // than a weakness of the test. Expansion is crossings now, and crossing
    // ladders are not pairwise-recoverable -- the catalogue cannot undo
    // them, so without a store the drain has the journal alone. Deep
    // contraction is COMP-DB's job (p_comp, which MixParams::default leaves
    // off precisely so tests need no store), and the size brake depends on
    // it. This asserts the thermostat still pushes the right way.
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

// Soundness of the collision predicate: ANY pair it calls non-colliding —
// no read of the other's target, or separated by an opposite shared
// control literal — must actually commute. (The converse is not claimed:
// collides() may stay conservatively true on commuting pairs.)
#[test]
fn collides_separation_exemption_sound() {
    let mut rng = StdRng::seed_from_u64(23);
    let mut exempted = 0usize;
    for _ in 0..30_000 {
        let a = rand_gate(&mut rng, 6, 3, true);
        let b = rand_gate(&mut rng, 6, 3, true);
        let reads = a.reads(b.target) || b.reads(a.target);
        if !XGate::collides(&a, &b) {
            assert!(
                rules::verify_rewrite(&[a.clone(), b.clone()], &[b.clone(), a.clone()]),
                "non-colliding pair does not commute: {a:?} / {b:?}"
            );
            if reads {
                exempted += 1; // separated despite a read of a target
                assert!(!a.comp && !b.comp, "comp gate got the exemption");
            }
        }
    }
    assert!(
        exempted > 20,
        "exemption never fired in the sample: {exempted}"
    );
}

// Per-gate conjugation identities behind the twist move: N g N == neg(g)
// and S g S == swap(g), exhaustively verified on random gates (comp gates
// included — fossils relabel like anything else and stay fossils).
#[test]
fn twist_conjugation_units() {
    let mut rng = StdRng::seed_from_u64(31);
    let cnot = |t: u16, c: u16| XGate::conj(t, [(c, true)]).unwrap();
    for _ in 0..5_000 {
        let g = rand_gate(&mut rng, 8, 4, true);
        let w = rng.random_range(0..8u16);
        match conj_by_not(&g, w) {
            Some(g2) => {
                let nw = XGate::x_gate(w);
                assert!(
                    rules::verify_rewrite(&[nw.clone(), g.clone(), nw], std::slice::from_ref(&g2)),
                    "neg conjugation wrong: {g:?} on wire {w}"
                );
                assert_eq!(g2.width(), g.width());
                assert_eq!(g2.comp, g.comp);
            }
            None => assert!(!g.reads(w), "invariant gate must not read the negated wire"),
        }
        let a = rng.random_range(0..8u16);
        let b = rng.random_range(0..8u16);
        if a == b {
            continue;
        }
        if let Some(g2) = conj_by_swap(&g, a, b) {
            let packet = [cnot(b, a), cnot(a, b), cnot(b, a)];
            let mut seq: Vec<XGate> = packet.to_vec();
            seq.push(g.clone());
            seq.extend(packet.to_vec());
            assert!(
                rules::verify_rewrite(&seq, std::slice::from_ref(&g2)),
                "swap conjugation wrong: {g:?} on wires {a},{b}"
            );
            assert_eq!(g2.width(), g.width());
            assert_eq!(g2.comp, g.comp);
            // Involution: conjugating back restores the gate exactly.
            assert_eq!(conj_by_swap(&g2, a, b).unwrap(), g);
        } else {
            assert!(g.target != a && g.target != b && !g.reads(a) && !g.reads(b));
        }
        // Transvection T = cnot(b -> a); gates writing b are excluded by
        // the move's b-selection, so they are out of scope here too.
        if g.target != b {
            let t = cnot(a, b);
            let sandwich = |pieces: &[XGate]| {
                rules::verify_rewrite(&[t.clone(), g.clone(), t.clone()], pieces)
            };
            match conj_by_cnot(&g, a, b) {
                CnotConj::Invariant => {
                    assert!(
                        sandwich(std::slice::from_ref(&g)),
                        "cnot-invariant gate is not invariant: {g:?} a={a} b={b}"
                    );
                }
                CnotConj::Flip(g2) => {
                    assert!(
                        sandwich(std::slice::from_ref(&g2)),
                        "cnot flip wrong: {g:?} a={a} b={b}"
                    );
                    assert_eq!(g2.width(), g.width());
                    assert_eq!(g2.comp, g.comp);
                }
                CnotConj::Split(x, y) => {
                    assert!(!g.comp, "comp gates must be Blocked, not Split");
                    assert!(
                        sandwich(&[x.clone(), y.clone()]),
                        "cnot split wrong: {g:?} a={a} b={b}"
                    );
                    // Disjoint b-slices: the pair commutes.
                    assert!(
                        sandwich(&[y.clone(), x.clone()]),
                        "cnot split pair does not commute"
                    );
                    assert_eq!(x.width(), g.width() + 1);
                    assert_eq!(y.width(), g.width() + 1);
                }
                CnotConj::Blocked => {
                    assert!(
                        g.comp && g.reads(a) && !g.reads(b),
                        "spurious Blocked: {g:?}"
                    );
                }
            }
        }
    }
}

// The chain with twist moves enabled at high weight keeps the function
// (run() global-checks internally), erodes rather than grows fossils, and
// actually relabels interior gates. Also exercises the journal-stamp
// interaction: undo entries over relabeled pieces must die, not fire.
// Target above input: twists add gates without catalogue-invertible bulk,
// so at target the thermostat pegs near-full contraction and would starve
// the expansion channel this test is exercising.
#[test]
fn mixer_twists_preserve_function() {
    let gates = random_mixed_circuit(17, 16, 300);
    let comp0 = gates.iter().filter(|g| g.comp).count();
    let params = MixParams {
        k_max: 5,
        moves: 20_000,
        target_size: 600,
        temp: 20.0,
        p_twist: 0.2, // slot 1 owns twists now; the w_* are type ratios
        w_twist_neg: 0.10,
        w_twist_swap: 0.10,
        twist_min_len: 4,
        verify_every: 1_000,
        report_every: u64::MAX,
        seed: 5,
        ..MixParams::default()
    };
    let mut mx = Mixer::new(gates, 16, params);
    mx.run();
    // Rate calibration: twist packets are hard for the catalogue to dig
    // out (brackets are wall-blocked by their own window), so at these
    // weights the thermostat pegs near-full contraction and expansions
    // run at ~2% of moves — expect twist counts near 50, not hundreds.
    let twists = mx.counters.twist_swaps + mx.counters.twist_negs + mx.counters.twist_cnots;
    assert!(twists > 30, "swap-family twists barely ran: {twists}");
    assert!(
        mx.counters.twist_relabels > 0,
        "twists never relabeled a gate"
    );
    assert!(mx.remaining_g57() <= comp0, "fossil count increased");
    assert!(mx.counters.merges() > 0, "no merges alongside twists");
    mx.global_check();
}

// The g57 census partitions the width-2 comp population, and the two halves
// measure different things: `shaped` is structure (what the store can
// spell) and `same_pol` is a twist odometer. The regression this guards is
// the one that made the old single `g57=` field misleading -- a NEGATION
// twist flips one control's polarity and moves a gate from opp_pol to
// same_pol WITHOUT changing its comp, width or count, while a SWAP twist
// carries polarity with the wire and moves nothing.
#[test]
fn g57_census_splits_structure_from_twist_polarity() {
    // The swap family flips control polarity only through its negation coins
    // (3/4 of twists carry a negation), so same_pol is a twist odometer:
    // twists ON drive it up, OFF leaves it at zero, while shaped (structure)
    // partitions correctly in both. p_db = 0 here, so nothing injects fresh
    // opposite-polarity material -- the only mover is the twist.
    let base = |p_twist: f64| MixParams {
        k_max: 5,
        moves: 20_000,
        target_size: 600,
        temp: 20.0,
        p_twist,
        twist_min_len: 4,
        verify_every: 1_000,
        report_every: u64::MAX,
        seed: 5,
        ..MixParams::default()
    };
    let run = |p_twist| {
        let mut mx = Mixer::new(random_g57_circuit(17, 16, 400), 16, base(p_twist));
        mx.run();
        let cen = mx.g57_census();
        // The partition identity, and agreement with the old accessor.
        assert_eq!(
            cen.shaped,
            cen.same_pol + cen.opp_pol,
            "census does not partition"
        );
        assert_eq!(
            cen.opp_pol,
            mx.true_g57(),
            "true_g57 diverged from the census"
        );
        assert!(
            cen.shaped <= mx.remaining_g57(),
            "shaped exceeds the comp population"
        );
        (cen, mx.counters.twist_relabels)
    };

    let (off, off_rel) = run(0.0);
    let (on, on_rel) = run(0.2);
    assert_eq!(off_rel, 0, "twists-off relabeled a gate: {off_rel}");
    assert!(on_rel > 0, "twists-on never relabeled a gate: {on_rel}");

    // Twists off: no polarity flips and no fresh material, so every shaped
    // gate stays a true g57. This is the load-bearing half -- it proves
    // same_pol tracks the twist, not mixing in general.
    assert_eq!(off.same_pol, 0, "twists-off flipped polarity: {off:?}");
    assert_eq!(off.pol_flipped(), 0.0);

    // Twists on: the negation coins flip it, on the same circuit. No upper
    // bound -- with p_db = 0 the small width-2 population can saturate at
    // 1.0; the sub-1/2 equilibrium seen in production comes from DB splices
    // this test deliberately does not have.
    assert!(
        on.shaped > 0 && off.shaped > 0,
        "no width-2 population: {on:?} {off:?}"
    );
    assert!(
        on.same_pol > 0,
        "swap-family twists flipped no polarity: {on:?}"
    );
    assert!(on.pol_flipped() > off.pol_flipped(), "{on:?} vs {off:?}");
}

// The directional walk at its defaults (undo live, birth advance +
// failed-cross retreat replacing the uniform scatter): function is
// preserved through every sub-step (local_verify + periodic global checks),
// crossings really run, and size stays bounded (the bound is a loose
// runaway canary, not a promise). Inserts are retired, so crossings are the
// whole expansion channel.
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

// The backward ancestor-span statistic must be populated under SAMPLED
// ancestry (where the old min/max ancspan= is switched off) and must stay
// inside its definitional bounds. It is bucket occupancy / entropy over
// the INPUT circuit, so all three live in [0,1].
#[test]
fn sampled_ancestor_span_is_populated_and_bounded() {
    let gates = random_mixed_circuit(53, 16, 400);
    let params = MixParams {
        k_max: 6,
        moves: 4_000,
        target_size: 400,
        temp: 20.0,
        anc_samples: 64,
        p_twist: 0.0,
        shuffle_rate: 0.0,
        report_every: u64::MAX,
        seed: 17,
        ..MixParams::default()
    };
    let mut mx = Mixer::new_with_db(gates, 16, params, FrozenDb::empty());
    mx.run();
    let line = mx.tracer_report();
    assert!(
        line.contains("ancspan cov="),
        "ancspan block missing: {line}"
    );
    let grab = |k: &str| -> f64 {
        let i = line.find(k).unwrap_or_else(|| panic!("no {k} in {line}")) + k.len();
        line[i..]
            .split(|c: char| c == ' ' || c == '|')
            .next()
            .unwrap()
            .parse()
            .unwrap()
    };
    let (cov, ent, sd) = (grab("ancspan cov="), grab("ent="), grab("sd="));
    for (n, v) in [("cov", cov), ("ent", ent), ("sd", sd)] {
        assert!((0.0..=1.0).contains(&v), "ancspan {n}={v} out of [0,1]");
    }
    // The old min/max form stays OFF under sampling -- this replaces it in
    // the tracers line, it does not resurrect it in the mv= line.
    assert_eq!(
        mx.anc_stats(),
        (0.0, 0.0),
        "sampled mode must still leave anc=/ancspan= at 0"
    );
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

// The tg_* counter fields are APPENDED to the state line and parsed with
// a zero default, so a .state written before they existed must still
// load. Simulated here by stripping the trailing tokens.
// The prefix descent must TERMINATE. Every exit path inside it has to
// shrink the window before looping; a `continue` that leaves lo/hi alone
// respins the identical prefix forever. Store misses are the common case,
// so before this was fixed a --db-prefixes run hung before completing a
// single move. With no DB attached every lookup misses, which is exactly
// the path that used to spin -- if this test ever hangs, the descent has
// regressed.
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

#[test]
fn counters_line_tolerates_missing_trailing_fields() {
    let mut c = MixCounters::default();
    c.moves = 7;
    c.tg_consumed = 3;
    c.tg_emitted = 9;
    c.split_prims = 11;
    c.tap_flips = 13;
    c.split_span_sum = 17;
    c.cross_pool_shots = 19;
    let line = c.to_line();
    let full = MixCounters::from_line(&line).expect("roundtrip");
    assert_eq!((full.tg_consumed, full.tg_emitted), (3, 9));
    assert_eq!((full.split_prims, full.tap_flips), (11, 13));
    assert_eq!((full.split_span_sum, full.cross_pool_shots), (17, 19));
    let old: Vec<&str> = line.split_whitespace().collect();
    // A pre-split-stage line (drop the NINE fields appended since: the 7
    // split counters + span sum + pool shots): they default to zero, the
    // tg pair survives.
    let truncated = old[..old.len() - 9].join(" ");
    let parsed = MixCounters::from_line(&truncated).expect("pre-split state must load");
    assert_eq!(parsed.moves, 7);
    assert_eq!((parsed.tg_consumed, parsed.tg_emitted), (3, 9));
    assert_eq!((parsed.split_prims, parsed.cross_pool_shots), (0, 0));
    // A pre-tg line (drop those too): everything appended defaults.
    let truncated = old[..old.len() - 11].join(" ");
    let parsed = MixCounters::from_line(&truncated).expect("pre-tg state must load");
    assert_eq!(parsed.moves, 7);
    assert_eq!((parsed.tg_consumed, parsed.tg_emitted), (0, 0));
    assert_eq!((parsed.split_prims, parsed.tap_flips), (0, 0));
}

// ---- the split stage (docs/FMIX_SPLIT_TWIST.md) ----

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

// Sibling convention (2026-08-05): the two pieces of a g57 split take
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

// v2 state roundtrip mid-stage: the live flag, the failure streak and the
// canaries all ride the checkpoint; the resumed run finishes the stage
// and still verifies.
#[test]
fn split_stage_state_roundtrip() {
    let gates = random_mixed_circuit(13, 16, 300);
    let params = MixParams {
        k_max: 8,
        moves: 4,
        split: true,
        split_canaries: 16,
        report_every: u64::MAX,
        verify_every: u64::MAX,
        seed: 9,
        ..MixParams::default()
    };
    let mut mx = Mixer::new_with_db(gates, 16, params.clone(), FrozenDb::empty());
    let stop = mx.run();
    assert!(matches!(stop, MixStop::MovesBudget));
    let taps0 = mx.taps.len();
    assert!(taps0 > 0, "canaries plant on the first stage move");
    let path = std::env::temp_dir().join("fmix_split_state_roundtrip.txt");
    let path = path.to_str().unwrap().to_string();
    mx.save_state(&path).expect("save state");
    let params2 = MixParams {
        moves: 200_000,
        split_stop: true,
        ..params
    };
    let mut mx2 = Mixer::resume_state(&path, params2, FrozenDb::empty()).expect("resume");
    assert!(mx2.split_on, "live stage flag must survive the checkpoint");
    assert_eq!(
        mx2.taps.len(),
        taps0,
        "canaries must survive the checkpoint"
    );
    let stop = mx2.run();
    assert!(
        matches!(stop, MixStop::SplitDone),
        "resumed stage must finish"
    );
    assert_eq!(mx2.remaining_g57(), 0);
    mx2.global_check();
    let _ = std::fs::remove_file(&path);
}

// Min-dgen cross-shot bias: armed, the walk still preserves function and
// the pool actually supplies shots; off, the constructor path draws no
// extra RNG so the trajectory is byte-identical to the historical chain.
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
fn mixer_g57_twists_preserve_function_and_absorb() {
    let gates = random_mixed_circuit(29, 16, 300);
    let params = MixParams {
        k_max: 6,
        moves: 20_000,
        target_size: 600,
        temp: 20.0,
        p_twist: 0.3,
        twist_min_len: 4,
        twist_g57: true,
        local_verify: true,
        verify_every: 1_000,
        report_every: u64::MAX,
        seed: 7,
        ..MixParams::default()
    };
    let mut mx = Mixer::new(gates, 16, params);
    mx.run();
    assert!(
        mx.counters.twist_swaps > 100,
        "g57 twists barely ran: {}",
        mx.counters.twist_swaps
    );
    assert!(mx.counters.tg_emitted > 0, "brackets emitted nothing");
    assert!(
        mx.counters.tg_consumed > 0,
        "no seam ever absorbed a neighbor -- the adaptive placement is dead"
    );
    // The solver minimizes NET (word len minus context consumed), so a
    // seam may emit up to 7 gates while consuming 3; what can never
    // happen is a twist NET above the 12-gate bare spelling, since k=0
    // always offers the 6-word per seam.
    let net = (mx.counters.tg_emitted as i64 - mx.counters.tg_consumed as i64) as f64
        / mx.counters.twist_swaps as f64;
    assert!(
        net <= 12.0 + 1e-9,
        "twist net cost exceeded the bare-word bound: {net}"
    );
    mx.global_check();
}

// v2 seam consumption must PROPAGATE ancestry: a bracket word that
// consumed real context takes the union of the consumed litters' ancestor
// sets (DB-splice semantics). v1 dropped them, silently deflating anc.
#[test]
fn g57_twist_consumption_inherits_ancestry() {
    let gates = random_mixed_circuit(31, 16, 300);
    let params = MixParams {
        k_max: 6,
        moves: 20_000,
        target_size: 600,
        temp: 20.0,
        p_twist: 0.3,
        twist_min_len: 4,
        twist_g57: true,
        local_verify: true,
        ancestors: true,
        verify_every: 1_000,
        report_every: u64::MAX,
        seed: 9,
        ..MixParams::default()
    };
    let mut mx = Mixer::new(gates, 16, params);
    mx.run();
    assert!(mx.counters.tg_consumed > 0, "no consumption to test");
    let inherited = mx.arena.ids_in_order().iter().any(|&id| {
        let m = mx.meta_of(id);
        m.origin == ORIGIN_SYNTH
            && mx
                .anc
                .get(&m.litter)
                .is_some_and(|bits| bits.iter().any(|&w| w != 0))
    });
    assert!(inherited, "no synthetic gate carries inherited ancestry");
    mx.global_check();
}

// Symmetric truncation: with twist_min_len at circuit scale every draw is
// near-full-length, so ~half the windows left-truncate (virtual start < 0)
// and their opening packets land at the head. Function preservation +
// global_check through thousands of such windows exercises the boundary
// insert path (brackets before the arena head) and the short-window skip
// paths (len as small as 1).
#[test]
fn twist_left_truncated_windows_preserve_function() {
    let gates = random_mixed_circuit(23, 16, 300);
    let params = MixParams {
        k_max: 5,
        moves: 20_000,
        target_size: 600,
        temp: 20.0,
        p_twist: 0.2, // slot 1 owns twists now; the w_* are type ratios
        w_twist_neg: 0.10,
        w_twist_swap: 0.10,
        twist_min_len: usize::MAX, // clamped to circuit size -> len == n
        verify_every: 1_000,
        report_every: u64::MAX,
        seed: 11,
        ..MixParams::default()
    };
    let mut mx = Mixer::new(gates, 16, params);
    mx.run();
    let twists = mx.counters.twist_swaps + mx.counters.twist_negs + mx.counters.twist_cnots;
    assert!(twists > 50, "twists barely ran: {twists}");
    mx.global_check();
}

// The negate-both variant is a genuine involution (T^2 = id) like the pure
// swap, whereas negate-one is not (T^2 = negate-both); this test drives the
// family hard so all three variants -- and the non-involutive closing
// bracket P^-1 -- get exercised, keeps the function through thousands of
// twists, and never grows fossils. All three variant counters must fire.
#[test]
fn mixer_swap_family_twists_preserve_function() {
    let gates = random_mixed_circuit(19, 16, 300);
    let comp0 = gates.iter().filter(|g| g.comp).count();
    let params = MixParams {
        k_max: 6,
        moves: 20_000,
        target_size: 600,
        temp: 20.0,
        p_twist: 0.3,
        twist_min_len: 4,
        verify_every: 1_000,
        report_every: u64::MAX,
        seed: 7,
        ..MixParams::default()
    };
    let mut mx = Mixer::new(gates, 16, params);
    mx.run();
    // The (alpha, beta) coins are 1/4 : 1/2 : 1/4, so at this rate all three
    // variants fire many times.
    assert!(
        mx.counters.twist_swaps > 5,
        "pure swaps barely ran: {}",
        mx.counters.twist_swaps
    );
    assert!(
        mx.counters.twist_negs > 5,
        "negate-one twists barely ran: {}",
        mx.counters.twist_negs
    );
    assert!(
        mx.counters.twist_cnots > 5,
        "negate-both twists barely ran: {}",
        mx.counters.twist_cnots
    );
    assert!(
        mx.counters.twist_relabels > 0,
        "twists never relabeled a gate"
    );
    assert!(mx.remaining_g57() <= comp0, "fossil count increased");
    assert!(mx.counters.merges() > 0, "no merges alongside twists");
    mx.global_check();
}

// Sampled ancestry must agree with exact ancestry on the quantity they both
// measure. Tracer choice comes from a dedicated rng, so the two runs follow
// the IDENTICAL chain (asserted gate-for-gate) and the only difference is
// the instrument -- which makes this a real calibration rather than two
// independent samples that happen to be close.
#[test]
fn sampled_ancestry_calibrates_to_exact() {
    let gates = random_mixed_circuit(31, 16, 400);
    let base = MixParams {
        k_max: 5,
        moves: 20_000,
        target_size: 600,
        temp: 20.0,
        p_twist: 0.05,
        verify_every: 5_000,
        report_every: u64::MAX,
        seed: 9,
        ..MixParams::default()
    };
    let mut ex = Mixer::new(
        gates.clone(),
        16,
        MixParams {
            ancestors: true,
            ..base.clone()
        },
    );
    ex.run();
    let mut sa = Mixer::new(
        gates.clone(),
        16,
        MixParams {
            anc_samples: 128,
            ..base.clone()
        },
    );
    sa.run();

    // Same chain: the instrument may not perturb the walk.
    assert_eq!(
        ex.arena.to_vec(),
        sa.arena.to_vec(),
        "tracer selection changed the trajectory"
    );

    // Exact mode still reports anc/span; sampled mode deliberately does not.
    assert!(ex.anc_stats().0 > 0.0, "exact mode lost its anc reading");
    assert_eq!(
        sa.anc_stats(),
        (0.0, 0.0),
        "sampled mode must not fill anc=/ancspan="
    );
    assert!(
        sa.tracer_report().contains("tracers: K=128"),
        "{}",
        sa.tracer_report()
    );

    let exact = ex.anc_incidence();
    let est = sa.anc_incidence();
    assert!(exact > 0.0, "no incidence to compare");
    let rel = (est - exact).abs() / exact;
    assert!(
        rel < 0.25,
        "sampled incidence off by {rel:.3} (exact {exact:.0}, est {est:.0})"
    );
}

// The joint gen x anc census must partition the circuit exactly (every gate
// lands in exactly one band, including the GEN_FRESH sentinel band) and must
// work in BOTH ancestry modes.
#[test]
fn gen_anc_census_partitions_the_circuit() {
    let gates = random_mixed_circuit(23, 16, 400);
    let base = MixParams {
        k_max: 5,
        moves: 20_000,
        target_size: 600,
        temp: 20.0,
        p_twist: 0.1, // mint some GEN_FRESH bracket material
        gen_target: 5,
        verify_every: 5_000,
        report_every: u64::MAX,
        seed: 4,
        ..MixParams::default()
    };
    for (label, p) in [
        (
            "exact",
            MixParams {
                ancestors: true,
                ..base.clone()
            },
        ),
        (
            "sampled",
            MixParams {
                anc_samples: 64,
                ..base.clone()
            },
        ),
    ] {
        let mut mx = Mixer::new(gates.clone(), 16, p);
        mx.run();
        let line = mx.gen_anc_report();
        assert!(line.starts_with("[fmix] gen-anc: r="), "{label}: {line}");
        // Band counts must sum to the circuit size.
        let total: usize = line
            .split('|')
            .filter_map(|s| s.split("n=").nth(1))
            .filter_map(|s| s.split_whitespace().next())
            .filter_map(|s| s.parse::<usize>().ok())
            .sum();
        // The first "n=" is the real-gen count in the header, so subtract it.
        let hdr: usize = line
            .split("(n=")
            .nth(1)
            .and_then(|s| s.split_whitespace().next())
            .and_then(|s| s.parse::<f64>().ok())
            .map(|f| f as usize)
            .expect("header count");
        assert_eq!(
            total - hdr,
            mx.arena.len(),
            "{label} bands do not partition: {line}"
        );
        assert!(
            hdr <= mx.arena.len(),
            "{label}: more real-gen gates than gates"
        );
        // Exact mode reports span per band, sampled mode must not.
        assert_eq!(line.contains("span="), label == "exact", "{label}: {line}");
    }
}

// The whole point of sampling: it runs on inputs the exact instrument
// refuses (it asserts n <= 20_000). Cost is K bits per litter regardless of
// input size, so the ancestor map stays far smaller than the circuit.
#[test]
fn sampled_ancestry_runs_past_the_exact_cap() {
    let n = 25_000;
    let gates = random_mixed_circuit(5, 24, n);
    let params = MixParams {
        k_max: 5,
        moves: 3_000,
        target_size: n + 200,
        temp: 50.0,
        anc_samples: 64,
        verify_every: 1_500,
        report_every: u64::MAX,
        seed: 3,
        ..MixParams::default()
    };
    let mut mx = Mixer::new(gates, 24, params);
    mx.run();
    assert!(
        mx.anc_incidence() > 0.0,
        "sampled ancestry recorded nothing"
    );
    let rep = mx.tracer_report();
    assert!(rep.contains("K=64 of m=25000"), "{rep}");
    // Only litters that actually carry a tracer are stored, so the map is a
    // small fraction of the circuit -- this is what makes it scale.
    assert!(
        mx.anc.len() < mx.arena.len(),
        "ancestor map ({}) is not smaller than the circuit ({})",
        mx.anc.len(),
        mx.arena.len()
    );
    mx.global_check();
}

// twist_neg_p = 0 gives PURE positive swaps: no wire is ever negated, so no
// interior polarity flips (same_pol stays 0) and only the pure-swap counter
// moves -- yet the 3-CNOT brackets are still inserted (comp=0 material
// present). This is the control that separates "foreign CNOTs" from
// "polarity scrambling".
#[test]
fn twist_neg_p_zero_is_pure_swap() {
    let params = MixParams {
        k_max: 5,
        moves: 20_000,
        target_size: 600,
        temp: 20.0,
        p_twist: 0.3,
        twist_neg_p: 0.0,
        twist_min_len: 4,
        verify_every: 1_000,
        report_every: u64::MAX,
        seed: 7,
        ..MixParams::default()
    };
    let mut mx = Mixer::new(random_g57_circuit(17, 16, 400), 16, params);
    mx.run();
    assert!(
        mx.counters.twist_swaps > 20,
        "no pure swaps ran: {}",
        mx.counters.twist_swaps
    );
    assert_eq!(
        mx.counters.twist_negs, 0,
        "negate-one fired at twist_neg_p=0"
    );
    assert_eq!(
        mx.counters.twist_cnots, 0,
        "negate-both fired at twist_neg_p=0"
    );
    assert_eq!(mx.g57_census().same_pol, 0, "pure swap flipped polarity");
    assert!(mx.counters.twist_relabels > 0, "pure swap never relabeled");
    mx.global_check();
}

// With twist weights at zero no twist path may ever be taken — not even a
// skipped attempt from floating-point dust in the weight subtractions —
// so per-move RNG consumption (and hence every seed's trajectory) matches
// the pre-twist chain exactly.
#[test]
fn twist_weights_zero_is_inert() {
    let gates = random_mixed_circuit(3, 16, 300);
    let params = MixParams {
        k_max: 5,
        moves: 10_000,
        target_size: 300,
        temp: 20.0,
        verify_every: 5_000,
        report_every: u64::MAX,
        seed: 5,
        ..MixParams::default()
    };
    let mut a = Mixer::new(gates, 16, params);
    a.run();
    assert_eq!(
        a.counters.twist_negs
            + a.counters.twist_swaps
            + a.counters.twist_cnots
            + a.counters.twist_skips,
        0
    );
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
    // Lower-median variant: same {3,7} spread now stamps min+1 on the
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

// The GSS profile needs the window length to depend on BOTH the live mode
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
        "MIX contiguous now overridden"
    );
    mx.db_mode_cur = DbMode::Compressing;
    assert_eq!(
        mx.active_s_db(DbSample::Contiguous),
        6,
        "COMP contiguous still reads s_db_comp_ctg, not the MIX override"
    );
}

// Descent is per-mode: the overlay runs MIX and COMP in one process and
// GSS wants it on in COMP and off in MIX. None must fall back to the
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

// Geometry is now drawn ONCE per round, before the length. A run pinned to
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

    // ZERO AND FALSE MEAN THEMSELVES. This is the whole reason these are
    // Option: under the old sentinel encoding a legitimate 0 -- which is
    // exactly what GSS wants for p_mingen_comp -- was indistinguishable
    // from "unset", so it silently fell through to the base.
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

    // The GSS profile, end to end.
    let gss = MixParams {
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
    let m = gss.db_knobs(DbMode::Mix);
    let c = gss.db_knobs(DbMode::Compressing);
    assert_eq!(
        (m.s_db_cvx, m.s_db_ctg, m.p_convex, m.p_mingen, m.prefixes),
        (6, 6, 0.5, 0.5, false)
    );
    assert_eq!(
        (c.s_db_cvx, c.s_db_ctg, c.p_convex, c.p_mingen, c.prefixes),
        (12, 6, 0.95, 0.0, true)
    );
}

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

// Checkpoint round-trip: a resumed chain must continue rather than restart.
// The circuit file alone cannot do this -- directions, generations, litters,
// the journal and the original are all outside it -- so the test checks the
// state that has no other home, and that the resumed run still verifies
// against the TRUE original rather than against its own resume point.
#[test]
fn checkpoint_round_trip_preserves_chain_state() {
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
    let mut mx = Mixer::new(gates.clone(), 16, params.clone());
    mx.run();
    let before_gates = mx.arena.to_vec();
    let before_meta: Vec<(u32, u32, u64)> = mx
        .arena
        .ids_in_order()
        .iter()
        .map(|&id| {
            let m = mx.meta_of(id);
            (m.dgen, m.origin, m.litter)
        })
        .collect();
    let before_moves = mx.moves_done;
    let before_twspan = mx.counters.twist_span;
    let dir_r = mx
        .arena
        .ids_in_order()
        .iter()
        .filter(|&&id| mx.meta_of(id).dir == Dir::R)
        .count();

    let path = std::env::temp_dir().join("fmix_ckpt_test.state");
    let path = path.to_str().unwrap();
    mx.save_state(path).expect("save");

    let mut rs = Mixer::resume_state(path, params, FrozenDb::empty()).expect("resume");
    assert_eq!(
        rs.arena.to_vec(),
        before_gates,
        "circuit must survive verbatim"
    );
    assert_eq!(rs.moves_done, before_moves, "move counter must continue");
    assert_eq!(
        rs.counters.twist_span, before_twspan,
        "twist coverage feeds the dose stop"
    );
    let after_meta: Vec<(u32, u32, u64)> = rs
        .arena
        .ids_in_order()
        .iter()
        .map(|&id| {
            let m = rs.meta_of(id);
            (m.dgen, m.origin, m.litter)
        })
        .collect();
    assert_eq!(
        after_meta, before_meta,
        "generations, origins and litters must survive"
    );
    let after_dir_r = rs
        .arena
        .ids_in_order()
        .iter()
        .filter(|&&id| rs.meta_of(id).dir == Dir::R)
        .count();
    assert_eq!(
        after_dir_r, dir_r,
        "directions have no sidecar and must survive"
    );
    // The resumed chain still verifies against the TRUE original.
    rs.global_check();
    rs.params.moves = before_moves + 2_000;
    rs.run();
    rs.global_check();
    assert!(
        rs.moves_done > before_moves,
        "resumed run must make progress"
    );
    let _ = std::fs::remove_file(path);
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
    // This is the product-share case, where ~62% of gates are wide.
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

// The regression this fix exists for: on majority-wide material (a
// product-share gadget) the old all-gates percentile pinned G= at 0 and
// the dose stop could never fire, however complete the dose actually was.
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

// Ancestry treats a cross as a DB splice over the window {g, h}: every
// output of a crossing — the intact pivot included — carries the UNION of
// both parents' ancestor sets. Verified through the journal, which
// records the pre-cross litters: every piece of a live entry must read
// exactly union(set(litters[0]), set(litters[1])).
#[test]
fn cross_outputs_carry_union_ancestry() {
    let gates = random_mixed_circuit(43, 16, 200);
    let n = gates.len();
    let params = MixParams {
        k_max: 6,
        moves: 6_000,
        target_size: 4 * n,
        temp: 20.0,
        ancestors: true,
        report_every: u64::MAX,
        seed: 5,
        ..MixParams::default()
    };
    let mut mx = Mixer::new(gates, 16, params);
    mx.run();
    let mut checked = 0usize;
    let mut expected = vec![0u64; mx.anc_words];
    let mut got = vec![0u64; mx.anc_words];
    for e in mx.journal.iter() {
        let live = e
            .after
            .iter()
            .all(|&(id, st)| mx.arena.is_linked(id) && mx.arena.stamp(id) == st);
        if !live {
            continue;
        }
        expected.iter_mut().for_each(|w| *w = 0);
        mx.anc_or_into(e.litters[0], &mut expected);
        mx.anc_or_into(e.litters[1], &mut expected);
        for &(id, _) in &e.after {
            got.iter_mut().for_each(|w| *w = 0);
            mx.anc_or_into(mx.meta_of(id).litter, &mut got);
            assert_eq!(
                got, expected,
                "a cross output (intact pivot included) must carry the parents' union"
            );
        }
        checked += 1;
    }
    assert!(
        checked > 0,
        "the run must leave live journal entries to check"
    );
    mx.global_check();
}

// The pre-cross litters a live journal entry will restore must survive
// anc_prune: the cross relabels EVERY output to the union litter, so the
// parents' litters can go extinct among live gates, and pruning their
// sets would make a later undo restore ancestry-less litters silently.
#[test]
fn anc_prune_keeps_litters_live_journal_entries_restore() {
    let gates = random_mixed_circuit(47, 16, 200);
    let n = gates.len();
    let params = MixParams {
        k_max: 6,
        moves: 6_000,
        target_size: 4 * n,
        temp: 20.0,
        ancestors: true,
        report_every: u64::MAX,
        seed: 6,
        ..MixParams::default()
    };
    let mut mx = Mixer::new(gates, 16, params);
    mx.run();
    let live_entries: Vec<[u64; 2]> = mx
        .journal
        .iter()
        .filter(|e| {
            e.after
                .iter()
                .all(|&(id, st)| mx.arena.is_linked(id) && mx.arena.stamp(id) == st)
        })
        .map(|e| e.litters)
        .collect();
    assert!(
        !live_entries.is_empty(),
        "need live journal entries to make the test bite"
    );
    let resolve = |mx: &Mixer, l: u64| {
        let mut bits = vec![0u64; mx.anc_words];
        mx.anc_or_into(l, &mut bits);
        bits
    };
    let before: Vec<[Vec<u64>; 2]> = live_entries
        .iter()
        .map(|ls| [resolve(&mx, ls[0]), resolve(&mx, ls[1])])
        .collect();
    mx.anc_prune();
    for (ls, want) in live_entries.iter().zip(before.iter()) {
        assert_eq!(
            resolve(&mx, ls[0]),
            want[0],
            "prune dropped a restorable litter's set"
        );
        assert_eq!(
            resolve(&mx, ls[1]),
            want[1],
            "prune dropped a restorable litter's set"
        );
    }
}

// Sidecar round trip in both universes: write -> read must reproduce the
// per-gate resolved sets verbatim, and importing into a FRESH mixer over
// the same circuit must resolve identically (the phase-boundary use).
#[test]
fn anc_sidecar_round_trips_exact_and_sampled() {
    for sampled in [false, true] {
        let gates = random_mixed_circuit(51, 16, 150);
        let params = MixParams {
            k_max: 6,
            moves: 4_000,
            target_size: 3 * gates.len(),
            temp: 20.0,
            ancestors: !sampled,
            anc_samples: if sampled { 32 } else { 0 },
            report_every: u64::MAX,
            seed: 7,
            ..MixParams::default()
        };
        let mut mx = Mixer::new(gates, 16, params);
        mx.run();
        let path = std::env::temp_dir().join(format!("fmix_anc_test_{sampled}.anc"));
        let path = path.to_str().unwrap().to_string();
        mx.write_anc_sidecar(&path).expect("write sidecar");
        let sc = Mixer::read_anc_sidecar(&path).expect("read sidecar");
        assert_eq!(sc.sampled, sampled);
        assert_eq!(sc.m, mx.anc_m, "universe size must survive");
        assert_eq!(sc.tracers, mx.anc_tracers, "tracer list must survive");
        let resolved: Vec<Vec<u64>> = {
            let mut v = Vec::new();
            let mut bits = vec![0u64; mx.anc_words];
            let mut cur = mx.arena.head();
            while cur != NIL {
                bits.iter_mut().for_each(|w| *w = 0);
                mx.anc_or_into(mx.meta_of(cur).litter, &mut bits);
                v.push(bits.clone());
                cur = mx.arena.neighbor(cur, Dir::R);
            }
            v
        };
        assert_eq!(
            sc.sets, resolved,
            "sidecar rows must equal the resolved per-gate sets"
        );

        // Import into a fresh run over the SAME circuit (ancestry off at
        // construction; the sidecar defines the universe).
        let out_gates = mx.arena.to_vec();
        let params2 = MixParams {
            k_max: 6,
            moves: 2_000,
            temp: 20.0,
            report_every: u64::MAX,
            seed: 8,
            ..MixParams::default()
        };
        let mut mx2 = Mixer::new(out_gates, 16, params2);
        let sc2 = Mixer::read_anc_sidecar(&path).expect("re-read sidecar");
        mx2.import_ancestry(sc2);
        assert_eq!(
            mx2.anc_m, mx.anc_m,
            "imported universe must be the ORIGINAL m"
        );
        assert_eq!(mx2.anc_tracers, mx.anc_tracers);
        let resolved2: Vec<Vec<u64>> = {
            let mut v = Vec::new();
            let mut bits = vec![0u64; mx2.anc_words];
            let mut cur = mx2.arena.head();
            while cur != NIL {
                bits.iter_mut().for_each(|w| *w = 0);
                mx2.anc_or_into(mx2.meta_of(cur).litter, &mut bits);
                v.push(bits.clone());
                cur = mx2.arena.neighbor(cur, Dir::R);
            }
            v
        };
        assert_eq!(
            resolved2, resolved,
            "imported ancestry must resolve identically"
        );
        // The imported run must keep walking and unioning without issue.
        mx2.run();
        mx2.global_check();
        assert_eq!(
            mx2.anc_m, mx.anc_m,
            "the universe must not drift during the run"
        );
        let _ = std::fs::remove_file(&path);
    }
}

// A state file written by an ancestry-IMPORTED run must restore the
// imported tracer set verbatim: it is not a function of (anc_m, K, seed),
// so only the explicit anctracers section can reproduce it.
#[test]
fn state_round_trips_imported_tracers() {
    let gates = random_mixed_circuit(53, 16, 150);
    let params = MixParams {
        k_max: 6,
        moves: 3_000,
        target_size: 3 * gates.len(),
        temp: 20.0,
        anc_samples: 24,
        report_every: u64::MAX,
        seed: 9,
        ..MixParams::default()
    };
    let mut mx = Mixer::new(gates, 16, params);
    mx.run();
    let anc_path = std::env::temp_dir().join("fmix_anc_state_test.anc");
    let anc_path = anc_path.to_str().unwrap().to_string();
    mx.write_anc_sidecar(&anc_path).expect("write sidecar");

    // Fresh run over the output, universe imported: its tracer indices
    // point into the ORIGINAL input, which this run has never seen.
    let out_gates = mx.arena.to_vec();
    let params2 = MixParams {
        k_max: 6,
        moves: 2_000,
        temp: 20.0,
        report_every: u64::MAX,
        seed: 10,
        ..MixParams::default()
    };
    let mut mx2 = Mixer::new(out_gates, 16, params2.clone());
    mx2.import_ancestry(Mixer::read_anc_sidecar(&anc_path).expect("read sidecar"));
    mx2.run();
    let want_tracers = mx2.anc_tracers.clone();
    let want_m = mx2.anc_m;

    let st_path = std::env::temp_dir().join("fmix_anc_state_test.state");
    let st_path = st_path.to_str().unwrap().to_string();
    mx2.save_state(&st_path).expect("save state");
    // Resume without any ancestry flags: the stored section must arm it.
    let rs = Mixer::resume_state(&st_path, params2, FrozenDb::empty()).expect("resume");
    assert!(rs.anc_sampled, "stored anctracers must re-arm sampled mode");
    assert_eq!(
        rs.anc_tracers, want_tracers,
        "imported tracers must survive the state file"
    );
    assert_eq!(rs.anc_m, want_m);
    let _ = std::fs::remove_file(&anc_path);
    let _ = std::fs::remove_file(&st_path);
}

// Pair geometry (docs/NONLOCAL_PHASE_A.md): with an empty store every
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

// The bridge wake algebra: for random carrier/interior-gate pairs, the
// claimed conjugate u·h·u = [h, corrections] must hold exactly — checked
// against exhaustive evaluation, with coverage over both collision modes
// (h reads the carrier's target / h writes a carrier control wire) and
// the commuting and contradictory (correction-vanishes) cases.
#[test]
fn conj_wake_is_the_exact_conjugate() {
    let n: u16 = 8;
    let mut rng = StdRng::seed_from_u64(0xb41d6e);
    let mut seen = [0usize; 3]; // commuting/vanished, mode-a, mode-b
    let mut refused = 0usize;
    for i in 0..6000 {
        let tu = rng.random_range(0..n);
        let mut xw = rng.random_range(0..n);
        let mut yw = rng.random_range(0..n);
        while xw == tu {
            xw = rng.random_range(0..n);
        }
        while yw == tu || yw == xw {
            yw = rng.random_range(0..n);
        }
        let u = XGate::conj(tu, [(xw, rng.random_bool(0.5)), (yw, rng.random_bool(0.5))]).unwrap();
        // Alternate g57 and conjunction interiors.
        let h = if i % 2 == 0 {
            let t = rng.random_range(0..n);
            let mut a = rng.random_range(0..n);
            let mut b = rng.random_range(0..n);
            while a == t {
                a = rng.random_range(0..n);
            }
            while b == t || b == a {
                b = rng.random_range(0..n);
            }
            XGate::from_g57([t, a, b])
        } else {
            let t = rng.random_range(0..n);
            let w = rng.random_range(1..=3);
            let mut wires: Vec<u16> = (0..n).filter(|&x| x != t).collect();
            for k in 0..wires.len() {
                let j = rng.random_range(k..wires.len());
                wires.swap(k, j);
            }
            XGate::conj(t, wires[..w].iter().map(|&x| (x, rng.random_bool(0.5)))).unwrap()
        };
        let Some(corrs) = conj_wake(&u, &h, 12) else {
            // Mode c (mutual collision) — must really be mutual.
            assert!(
                h.reads(tu) && u.reads(h.target),
                "spurious refusal: {u:?} x {h:?}"
            );
            refused += 1;
            continue;
        };
        let mut after = vec![h.clone()];
        after.extend(corrs.iter().cloned());
        assert!(
            rules::verify_rewrite(&[u.clone(), h.clone(), u.clone()], &after),
            "conjugate wrong: {u:?} x {h:?} -> {after:?}"
        );
        if corrs.is_empty() {
            seen[0] += 1;
        } else if h.reads(tu) {
            seen[1] += 1;
        } else {
            seen[2] += 1;
        }
    }
    assert!(
        seen.iter().all(|&c| c > 100) && refused > 0,
        "coverage too thin: {seen:?} refused={refused}"
    );
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

#[test]
fn checkpoint_v1_and_early_v2_still_load_and_continue() {
    let gates = random_mixed_circuit(29, 8, 20);
    let params = MixParams {
        moves: 40,
        p_db: 0.0,
        verify_every: u64::MAX,
        report_every: u64::MAX,
        ..MixParams::default()
    };
    let mut original = Mixer::new_with_db(gates, 8, params.clone(), FrozenDb::empty());
    let path = std::env::temp_dir().join(format!("gss_old_state_{}.state", std::process::id()));
    original.save_state(path.to_str().unwrap()).unwrap();
    let v2 = std::fs::read_to_string(&path).unwrap();
    for version in [1, 2] {
        let mut old = String::new();
        for line in v2.lines() {
            if line.starts_with("staps ") {
                break;
            }
            if line.starts_with("fmix-state ") {
                old.push_str(&format!("fmix-state {version}\n"));
            } else if line.starts_with("split ") {
                if version == 2 {
                    old.push_str(
                        &line
                            .split_whitespace()
                            .take(3)
                            .collect::<Vec<_>>()
                            .join(" "),
                    );
                    old.push('\n');
                }
            } else {
                old.push_str(line);
                old.push('\n');
            }
        }
        std::fs::write(&path, old).unwrap();
        let mut resumed =
            Mixer::resume_state(path.to_str().unwrap(), params.clone(), FrozenDb::empty()).unwrap();
        assert_eq!(resumed.arena.to_vec(), original.arena.to_vec());
        resumed.run();
        resumed.global_check();
        assert!(resumed.moves_done > 0);
    }
    std::fs::remove_file(path).unwrap();
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
