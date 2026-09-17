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
    let fields: Vec<&str> = line.split_whitespace().collect();
    // Dropping the seven split counters, span sum and pool shots must
    // default those fields to zero while preserving the tg pair.
    let truncated = fields[..fields.len() - 9].join(" ");
    let parsed =
        MixCounters::from_line(&truncated).expect("state without split counters must load");
    assert_eq!(parsed.moves, 7);
    assert_eq!((parsed.tg_consumed, parsed.tg_emitted), (3, 9));
    assert_eq!((parsed.split_prims, parsed.cross_pool_shots), (0, 0));
    // Dropping the tg pair as well defaults all trailing fields.
    let truncated = fields[..fields.len() - 11].join(" ");
    let parsed = MixCounters::from_line(&truncated).expect("state without tg counters must load");
    assert_eq!(parsed.moves, 7);
    assert_eq!((parsed.tg_consumed, parsed.tg_emitted), (0, 0));
    assert_eq!((parsed.split_prims, parsed.tap_flips), (0, 0));
}

// State roundtrip mid-stage: the live flag, the failure streak and the
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
    let path = std::env::temp_dir().join("circuit_mixer_split_state_roundtrip.txt");
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

    let path = std::env::temp_dir().join("circuit_mixer_ckpt_test.state");
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
    let anc_path = std::env::temp_dir().join("circuit_mixer_anc_state_test.anc");
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

    let st_path = std::env::temp_dir().join("circuit_mixer_anc_state_test.state");
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

#[test]
fn checkpoint_versions_without_optional_sections_load_and_continue() {
    let gates = random_mixed_circuit(29, 8, 20);
    let params = MixParams {
        moves: 40,
        p_db: 0.0,
        verify_every: u64::MAX,
        report_every: u64::MAX,
        ..MixParams::default()
    };
    let mut original = Mixer::new_with_db(gates, 8, params.clone(), FrozenDb::empty());
    let path = std::env::temp_dir().join(format!("tdp_compact_state_{}.state", std::process::id()));
    original.save_state(path.to_str().unwrap()).unwrap();
    let serialized = std::fs::read_to_string(&path).unwrap();
    assert!(serialized.starts_with("circuit-mixer-state 2\n"));
    for version in [1, 2] {
        let mut compact = String::new();
        for line in serialized.lines() {
            if line.starts_with("staps ") {
                break;
            }
            if line.starts_with("circuit-mixer-state ") {
                compact.push_str(&format!("circuit-mixer-state {version}\n"));
            } else if line.starts_with("split ") {
                if version == 2 {
                    compact.push_str(
                        &line
                            .split_whitespace()
                            .take(3)
                            .collect::<Vec<_>>()
                            .join(" "),
                    );
                    compact.push('\n');
                }
            } else {
                compact.push_str(line);
                compact.push('\n');
            }
        }
        std::fs::write(&path, compact).unwrap();
        let mut resumed =
            Mixer::resume_state(path.to_str().unwrap(), params.clone(), FrozenDb::empty()).unwrap();
        assert_eq!(resumed.arena.to_vec(), original.arena.to_vec());
        resumed.run();
        resumed.global_check();
        assert!(resumed.moves_done > 0);
    }
    std::fs::remove_file(path).unwrap();
}
