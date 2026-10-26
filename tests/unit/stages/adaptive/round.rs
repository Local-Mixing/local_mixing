use super::*;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

fn scratch() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "adaptive_round_ut_{}_{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

fn cand(seed: usize, gates: usize, exposure: usize) -> Candidate {
    Candidate {
        seed,
        gates,
        exposure,
    }
}

#[test]
fn choose_smaller_not_leakier() {
    let mut picks = Picks::new(2);
    // seed 1 is smaller and not leakier; seed 0 is smaller but leakier
    let c = choose(100, Some(3), &[cand(0, 50, 4), cand(1, 70, 3)], &mut picks);
    assert_eq!(c, Some(1));
    assert_eq!(picks.rejected_leakier, 1);
    assert_eq!(picks.seed, vec![0, 1]);
    // lower exposure beats fewer gates
    let c = choose(100, Some(3), &[cand(0, 50, 3), cand(1, 90, 1)], &mut picks);
    assert_eq!(c, Some(1));
    // equal exposure: fewer gates
    let c = choose(100, Some(3), &[cand(0, 50, 2), cand(1, 60, 2)], &mut picks);
    assert_eq!(c, Some(0));
    // equal (exposure, gates): the first seed
    let c = choose(100, Some(3), &[cand(0, 50, 2), cand(1, 50, 2)], &mut picks);
    assert_eq!(c, Some(0));
    assert_eq!(picks.seed, vec![2, 2]);
    assert_eq!(picks.orig, 0);
}

#[test]
fn choose_rejects_growing_and_keeps_original() {
    let mut picks = Picks::new(2);
    // same size is "growing" (strictly smaller required)
    let c = choose(
        100,
        Some(3),
        &[cand(0, 100, 0), cand(1, 120, 0)],
        &mut picks,
    );
    assert_eq!(c, None);
    assert_eq!(picks.rejected_growing, 2);
    assert_eq!(picks.orig, 1);
    // no candidates at all
    assert_eq!(choose(10, Some(0), &[], &mut picks), None);
    assert_eq!(picks.orig, 2);
    // a smaller candidate that exposes MORE than the original loses to the original
    let c = choose(100, Some(0), &[cand(0, 10, 1)], &mut picks);
    assert_eq!(c, None);
    assert_eq!(picks.rejected_leakier, 1);
    assert_eq!(picks.orig, 3);
}

#[test]
fn choose_smaller_rule_ignores_exposure() {
    let mut picks = Picks::new(3);
    let c = choose(
        100,
        None,
        &[cand(0, 80, 999), cand(1, 60, 999), cand(2, 60, 0)],
        &mut picks,
    );
    assert_eq!(c, Some(1));
    assert_eq!(picks.rejected_leakier, 0);
    let c = choose(100, None, &[cand(0, 100, 0)], &mut picks);
    assert_eq!(c, None);
    assert_eq!(picks.rejected_growing, 1);
    assert_eq!(picks.seed, vec![0, 1, 0]);
}

#[test]
fn budget_rule() {
    assert_eq!(budget_multiplier(1), 1);
    assert_eq!(budget_multiplier(1000), 1);
    assert_eq!(budget_multiplier(1001), 2);
    assert_eq!(budget_multiplier(4000), 2);
    assert_eq!(budget_multiplier(4001), 4);
    assert_eq!(budget_multiplier(65_536), 4);
    assert_eq!(budget(60.0, 500), 60.0);
    assert_eq!(budget(60.0, 2000), 120.0);
    assert_eq!(budget(60.0, 5000), 240.0);
}

#[test]
fn block_seeds_differ_per_round_and_slot() {
    let a = block_seed(1, 1, 0);
    assert_ne!(a, block_seed(1, 1, 1));
    assert_ne!(a, block_seed(1, 2, 0));
    assert_ne!(a, block_seed(2, 1, 0));
    assert_eq!(a, block_seed(1, 1, 0));
}

#[test]
fn edges_of_leaves() {
    assert_eq!(edges_of(&[(0, 3), (3, 7), (7, 9)]), vec![0, 3, 7, 9]);
    assert_eq!(edges_of(&[]), vec![0]);
}

#[test]
fn accept_names_round_trip() {
    for a in [Accept::Smaller, Accept::SmallerNotLeakier] {
        assert_eq!(Accept::parse(a.name()), Some(a));
    }
    assert_eq!(
        Accept::parse("smaller_not_leakier"),
        Some(Accept::SmallerNotLeakier)
    );
    assert_eq!(Accept::parse("larger"), None);
}

fn sample_source() -> Source {
    // C on 4 wires: wire 3 written twice (gate 0 internal, gate 2 output)
    let c = vec![XGate::cnot(3, 0), XGate::cnot(2, 1), XGate::cnot(3, 1)];
    Source::new(c, 4, BTreeSet::new())
}

fn family(a: usize, d: usize, f: f64) -> FamilyMetrics {
    FamilyMetrics {
        all_internal_births: a,
        deeper_births: d,
        cut_exposure_fraction: f,
        ids: BTreeSet::new(),
    }
}

fn sample_result(round: usize, input_gates: usize, output_gates: usize) -> RoundResult {
    RoundResult {
        round,
        input_gates,
        output_gates,
        blocks: 5,
        initial_blocks: 4,
        refinement: vec![
            RefinementLevel {
                level: 1,
                split: 1,
                children: 2,
                synthesized: 3,
            },
            RefinementLevel {
                level: 2,
                split: 0,
                children: 0,
                synthesized: 0,
            },
        ],
        picks: Picks {
            orig: 2,
            seed: vec![1, 2],
            rejected_leakier: 4,
            rejected_growing: 5,
        },
        coverage: Coverage {
            synthesized_blocks: 3,
            input_gates_synthesized: 77,
        },
        leakage_in: Leakage {
            discovery: LeakageMetrics {
                direct: family(1, 1, 0.25),
                affine: family(2, 1, 0.5),
            },
            validated: LeakageMetrics {
                direct: family(1, 0, 0.125),
                affine: family(2, 1, 0.375),
            },
            audit_passed: true,
        },
        leakage_out: Leakage {
            discovery: LeakageMetrics {
                direct: family(0, 0, 0.0),
                affine: family(1, 1, 0.1),
            },
            validated: LeakageMetrics {
                direct: family(0, 0, 0.0),
                affine: family(1, 1, 0.1),
            },
            audit_passed: false,
        },
        output_boundaries: vec![3, 9],
        equivalence_samples: 8192,
        seconds: 1.5,
        reloaded: false,
        output: None,
    }
}

#[test]
fn round_report_round_trips() {
    let params = RoundParams::default();
    let source = sample_source();
    let ctx = RoundContext {
        source: &source,
        width: 16,
        stage_seed: 12345,
        params: &params,
    };
    let res = sample_result(3, 100, 90);
    let text = round_report(&res, &ctx, 16);
    assert!(
        !text.contains("12345"),
        "the stage seed must not appear in the report"
    );
    for key in [
        "adaptive_round_report=1",
        "status=ok",
        "round=3",
        "input_gates=100",
        "output_gates=90",
        "blocks=5",
        "initial_blocks=4",
        "refinement_levels=2",
        "refinement.1.split=1",
        "picks.seed.1=2",
        "coverage.input_gates_synthesized=77",
        "leakage_in.validated.affine.deeper_births=1",
        "leakage_out.audit_passed=false",
        "output_boundaries=2",
        "output_boundaries_file=bounds.out",
        "equivalence_samples=8192",
        "param.accept=smaller-not-leakier",
    ] {
        assert!(text.lines().any(|l| l == key), "missing {key} in\n{text}");
    }
    let back = parse_round_report(&text).unwrap();
    let mut want = res.clone();
    want.output_boundaries.clear();
    assert_eq!(back, want);
    assert!(parse_round_report("status=ok\n").is_err());
    assert!(parse_round_report(&text.replace("status=ok", "status=failed")).is_err());
}

#[test]
fn parse_kv_basics() {
    let m = parse_kv("# comment\na=1\nb=x=y\n\nc=\n");
    assert_eq!(m.get("a").map(String::as_str), Some("1"));
    assert_eq!(m.get("b").map(String::as_str), Some("x=y"));
    assert_eq!(m.get("c").map(String::as_str), Some(""));
    assert_eq!(m.len(), 3);
}

#[test]
fn group_cache_round_trips() {
    let dir = scratch();
    let tsv = dir.join("g.tsv");
    let circ = dir.join("g.mpmct1");
    let ok_gates = vec![XGate::cnot(1, 0), XGate::x_gate(2)];
    let rows = vec![
        BlockRow {
            lo: 0,
            hi: 5,
            status: "ok".into(),
            synth_status: "converged".into(),
            syn_gates: 2,
            secs: 0.5,
            time_limit: 60.0,
            support: 3,
            changed: 2,
            monomials: 4,
            max_degree: 2,
            pre_postopt: 3,
            steps: 10,
            restarts_used: 2,
            deadline_hit: false,
            universe: 8,
            note: "rs=2/2/2/2".into(),
            gates: Some(ok_gates.clone()),
        },
        BlockRow {
            lo: 5,
            hi: 9,
            status: "stepcap".into(),
            synth_status: "stepcap".into(),
            syn_gates: 4,
            secs: 1.0,
            time_limit: 60.0,
            support: 1,
            changed: 1,
            monomials: 1,
            max_degree: 1,
            pre_postopt: 0,
            steps: 99,
            restarts_used: 1,
            deadline_hit: true,
            universe: 4,
            note: "a\tnote".into(),
            gates: None,
        },
    ];
    save_group(&tsv, &circ, &rows, 8).unwrap();
    let blocks = [(0, 5), (5, 9)];
    let back = load_group(&tsv, &circ, &blocks, 8).expect("cache reloads");
    assert_eq!(back.len(), 2);
    assert_eq!(back[0].gates, Some(ok_gates));
    assert_eq!(back[0].note, "rs=2/2/2/2");
    assert_eq!(back[1].gates, None);
    assert!(back[1].deadline_hit);
    assert_eq!(back[1].note, "a note");
    // a different block set or width does not reload
    assert!(load_group(&tsv, &circ, &[(0, 5)], 8).is_none());
    assert!(load_group(&tsv, &circ, &[(0, 5), (5, 10)], 8).is_none());
    assert!(load_group(&tsv, &circ, &blocks, 16).is_none());
    assert!(load_group(&dir.join("missing.tsv"), &circ, &blocks, 8).is_none());
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn completed_round_is_reloaded_not_recomputed() {
    let dir = scratch();
    let params = RoundParams::default();
    let source = sample_source();
    let ctx = RoundContext {
        source: &source,
        width: 16,
        stage_seed: 1,
        params: &params,
    };
    let input: Vec<XGate> = (0..10)
        .map(|i| XGate::cnot((i % 4) as u16 + 8, (i % 3) as u16))
        .collect();
    let res = sample_result(2, input.len(), 7);
    write_text_atomic(&dir.join("result.report"), &round_report(&res, &ctx, 16)).unwrap();
    let lines = std::sync::Mutex::new(Vec::<String>::new());
    let log = |s: &str| lines.lock().unwrap().push(s.to_string());
    // the analysis half is never called: a reload needs no census or plan
    let back = run_round(&ctx, 2, &input, &[], &dir, &log).unwrap();
    assert!(back.reloaded);
    assert!(back.output.is_none());
    assert_eq!(back.output_gates, 7);
    assert_eq!(back.input_gates, input.len());
    assert!(lines.lock().unwrap().iter().any(|l| l.contains("reloaded")));
    // a report of another round, or of another input size, is an input error
    let e = run_round(&ctx, 3, &input, &[], &dir, &log).unwrap_err();
    assert!(matches!(e, AdaptiveError::Input(_)), "{e}");
    assert_eq!(e.exit_code(), 3);
    let e = run_round(&ctx, 2, &input[..5], &[], &dir, &log).unwrap_err();
    assert!(matches!(e, AdaptiveError::Input(_)), "{e}");
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn bounds_file_round_trips() {
    let dir = scratch();
    let p = dir.join("bounds.out");
    write_text_atomic(&p, "3\n9\n").unwrap();
    assert_eq!(read_bounds_file(&p).unwrap(), vec![3, 9]);
    write_text_atomic(&p, "").unwrap();
    assert_eq!(read_bounds_file(&p).unwrap(), Vec::<usize>::new());
    write_text_atomic(&p, "x\n").unwrap();
    assert!(read_bounds_file(&p).is_err());
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn error_codes_mirror_stage7() {
    assert_eq!(AdaptiveError::Usage("u".into()).exit_code(), 2);
    assert_eq!(AdaptiveError::Input("i".into()).exit_code(), 3);
    assert_eq!(AdaptiveError::Verify("v".into()).exit_code(), 4);
    assert_eq!(AdaptiveError::io("p", io::Error::other("x")).exit_code(), 7);
    assert_eq!(AdaptiveError::Internal("n".into()).exit_code(), 8);
    assert_eq!(AdaptiveError::meaning(4), "verification failure");
    let e: AdaptiveError = crate::stages::unpacking::Stage7Error::Verify("z".into()).into();
    assert_eq!(e.exit_code(), 4);
    assert_eq!(e.to_string(), "verification failed: z");
}

#[test]
fn params_validate() {
    assert!(RoundParams::default().validate().is_ok());
    let p = RoundParams {
        seeds_per_block: 0,
        ..Default::default()
    };
    assert!(p.validate().is_err());
    let p = RoundParams {
        base_time_limit: 0.0,
        ..Default::default()
    };
    assert!(p.validate().is_err());
    let p = RoundParams {
        equiv_batches: 0,
        ..Default::default()
    };
    assert!(p.validate().is_err());
    let p = RoundParams {
        plan: PlanParams {
            segments: 0,
            ..Default::default()
        },
        ..Default::default()
    };
    assert!(p.validate().is_err());
}

#[test]
fn reload_round_needs_no_circuit() {
    let dir = scratch();
    let params = RoundParams::default();
    let source = sample_source();
    let ctx = RoundContext {
        source: &source,
        width: 16,
        stage_seed: 1,
        params: &params,
    };
    let log = |_: &str| {};
    // nothing completed: None, and nothing else is read
    assert!(reload_round(&dir, 1, 10, &log).unwrap().is_none());
    let res = sample_result(4, 10, 7);
    write_text_atomic(&dir.join("result.report"), &round_report(&res, &ctx, 16)).unwrap();
    // the report alone suffices (no output.mpmct1, no census, no plan)
    let back = reload_round(&dir, 4, 10, &log).unwrap().expect("reloaded");
    assert!(back.reloaded);
    assert!(back.output.is_none());
    assert_eq!((back.input_gates, back.output_gates), (10, 7));
    // another round number or input size: an input error (exit 3)
    assert_eq!(reload_round(&dir, 5, 10, &log).unwrap_err().exit_code(), 3);
    assert_eq!(reload_round(&dir, 4, 11, &log).unwrap_err().exit_code(), 3);
    let _ = fs::remove_dir_all(dir);
}
