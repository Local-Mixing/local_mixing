use super::*;
use crate::stages::adaptive::round::{Coverage, Picks, RoundResult};
use crate::stages::unpacking::mono::Rng;

#[test]
fn stop_rule_two_rounds_without_decrease() {
    let mut stalled = 0;
    let mut rounds = Vec::new();
    rounds.push((1000, 900));
    assert_eq!(stop_rule(&rounds, &mut stalled, 0.02), None);
    assert_eq!(stalled, 0);
    rounds.push((900, 800));
    assert_eq!(stop_rule(&rounds, &mut stalled, 0.02), None);
    rounds.push((800, 800));
    assert_eq!(stop_rule(&rounds, &mut stalled, 0.02), None);
    assert_eq!(stalled, 1);
    rounds.push((800, 700));
    assert_eq!(stop_rule(&rounds, &mut stalled, 0.02), None);
    assert_eq!(stalled, 0, "a decrease resets the stall count");
    rounds.push((700, 700));
    assert_eq!(stop_rule(&rounds, &mut stalled, 0.02), None);
    rounds.push((700, 710));
    assert_eq!(
        stop_rule(&rounds, &mut stalled, 0.02).as_deref(),
        Some("two consecutive rounds without size decrease")
    );
    assert_eq!(stalled, 2);
}

#[test]
fn stop_rule_joint_compression() {
    let mut stalled = 0;
    let mut rounds = vec![(1000, 995)];
    assert_eq!(stop_rule(&rounds, &mut stalled, 0.02), None);
    rounds.push((995, 990));
    assert_eq!(
        stop_rule(&rounds, &mut stalled, 0.02).as_deref(),
        Some("two consecutive rounds with joint compression below 2%")
    );
    assert_eq!(stalled, 2, "the joint rule counts as a plateau");
    // exactly at the threshold stops too (>=)
    let mut stalled = 0;
    let rounds = [(1000, 990), (990, 980)];
    assert!(stop_rule(&rounds[..1], &mut stalled, 0.02).is_none());
    assert!(stop_rule(&rounds, &mut stalled, 0.02).is_some());
    // just below continues
    let mut stalled = 0;
    let rounds = [(1000, 990), (990, 979)];
    assert!(stop_rule(&rounds, &mut stalled, 0.02).is_none());
    // stop fraction 0: only a non-decreasing pair over two rounds stops
    let mut stalled = 0;
    let rounds = [(1000, 990), (990, 989)];
    assert!(stop_rule(&rounds, &mut stalled, 0.0).is_none());
    let rounds = [(1000, 990), (990, 1000)];
    let mut stalled = 0;
    assert!(stop_rule(&rounds[..1], &mut stalled, 0.0).is_none());
    assert!(stop_rule(&rounds, &mut stalled, 0.0).is_some());
    assert!(stop_rule(&[], &mut stalled, 0.0).is_none());
}

#[test]
fn percent_formatting() {
    assert_eq!(percent(0.02), "2");
    assert_eq!(percent(0.005), "0.5");
    assert_eq!(percent(0.1), "10");
    assert_eq!(percent(0.0), "0");
}

#[test]
fn structured_patterns_count_and_bits() {
    let ps = structured_patterns(8);
    assert_eq!(ps.len(), 2 + 16 + 28);
    assert!(ps.contains(&Pattern::Pair(2, 5)));
    assert!(!ps.contains(&Pattern::Pair(5, 2)));
    assert!(Pattern::NotUnit(3).bit(0) && !Pattern::NotUnit(3).bit(3));
    let lanes = patterns_lanes(&ps, 8);
    assert_eq!(lanes.len(), 8);
    // input 1 (all ones) sets bit 1 of every wire's lane
    assert!(lanes.iter().all(|l| (l[0] >> 1) & 1 == 1));
    // input 0 (zero) sets no bit 0
    assert!(lanes.iter().all(|l| l[0] & 1 == 0));
}

fn random_gates(width: u16, m: usize, rng: &mut Rng) -> Vec<XGate> {
    (0..m)
        .map(|_| {
            let t = rng.below(width as u64) as u16;
            let mut a = rng.below(width as u64) as u16;
            while a == t {
                a = rng.below(width as u64) as u16;
            }
            let mut x = XGate::conj(t, [(a, rng.below(2) == 1)]).unwrap();
            x.comp = rng.below(2) == 1;
            x
        })
        .collect()
}

#[test]
fn payload_contract_holds_and_fails() {
    let n = 4;
    let mut rng = Rng::new(5);
    let sandwich = random_gates(2 * n as u16, 30, &mut rng);
    // the circuit IS the sandwich inside a 4n-wire state: the contract holds
    let k = payload_contract(&sandwich, 2 * n, &sandwich, 4 * n, n).unwrap();
    assert_eq!(k, 2 * (8192 + 2 + 2 * n + n * (n - 1) / 2));
    // gates on wires outside the payload band do not matter
    let mut c = sandwich.clone();
    c.push(XGate::cnot(13, 0));
    c.push(XGate::x_gate(15));
    assert!(payload_contract(&sandwich, 2 * n, &c, 4 * n, n).is_ok());
    // a gate that flips a payload wire breaks it (forward)
    let mut c = sandwich.clone();
    c.push(XGate::x_gate(n as u16));
    let e = payload_contract(&sandwich, 2 * n, &c, 4 * n, n).unwrap_err();
    assert!(matches!(e, AdaptiveError::Verify(_)), "{e}");
    assert_eq!(e.exit_code(), 4);
    // a gate at the FRONT that flips a payload wire breaks the inverse direction
    let mut c = vec![XGate::x_gate(n as u16 + 1)];
    c.extend(sandwich.iter().cloned());
    assert!(payload_contract(&sandwich, 2 * n, &c, 4 * n, n).is_err());
    // the sandwich must fit its state
    assert!(payload_contract(&[XGate::cnot(9, 0)], 2 * n, &sandwich, 4 * n, n).is_err());
    assert!(payload_contract(&sandwich, 2 * n, &sandwich, 2 * n - 1, n).is_err());
}

#[test]
fn final_report_lists_rounds_and_checks() {
    let inputs = DriverInputs {
        input: PathBuf::from("G.mpmct1"),
        source_c: PathBuf::from("C.g57"),
        n: 8,
        sandwich: None,
        out_dir: PathBuf::from("out"),
        seed: 987_654_321,
        threads: 4,
    };
    let params = DriverParams::default();
    let round = RoundResult {
        round: 1,
        input_gates: 100,
        output_gates: 80,
        blocks: 3,
        initial_blocks: 3,
        refinement: vec![],
        picks: Picks {
            orig: 1,
            seed: vec![1, 1],
            rejected_leakier: 0,
            rejected_growing: 0,
        },
        coverage: Coverage {
            synthesized_blocks: 2,
            input_gates_synthesized: 60,
        },
        leakage_in: Leakage::default(),
        leakage_out: Leakage::default(),
        output_boundaries: vec![30, 60],
        equivalence_samples: 8192,
        seconds: 2.0,
        reloaded: false,
        output: None,
    };
    let rep = FinalReport {
        input_gates: 100,
        final_gates: 80,
        final_round: Some(1),
        final_reloaded: false,
        rounds: vec![round],
        stop_reason: "round cap".into(),
        plateau: false,
        final_leakage: Leakage::default(),
        boundary: BoundaryCounts {
            internal: 5,
            excluded: 2,
            deeper: 3,
        },
        checks: FinalChecks {
            equiv_canonical: Some(8192),
            canonical_reloaded: false,
            zero_slice: 4096,
            payload_contract: None,
        },
        seconds: 3.0,
    };
    let text = final_report_text(&rep, &inputs, &params, 32, "file");
    assert!(!text.contains("987654321"), "the seed must not appear");
    for key in [
        "adaptive_report=1",
        "status=ok",
        "rounds=1",
        "stop_reason=round cap",
        "plateau=false",
        "input_gates=100",
        "final_gates=80",
        "final_round=1",
        "param.seed_source=file",
        "param.threads=4",
        "boundary.deeper=3",
        "round.1.output_gates=80",
        "round.1.picks_seed=1,1",
        "check.equiv_canonical=ok(8192)",
        "check.zero_slice=ok(4096)",
        "check.payload_contract=skipped",
        "final_reloaded=false",
        "final_leakage.validated.affine.deeper_births=0",
    ] {
        assert!(text.lines().any(|l| l == key), "missing {key} in\n{text}");
    }
}

#[test]
fn params_validate() {
    assert!(DriverParams::default().validate().is_ok());
    let p = DriverParams {
        stop_fraction: 1.0,
        ..Default::default()
    };
    assert!(p.validate().is_err());
    let p = DriverParams {
        probe_inputs: 0,
        ..Default::default()
    };
    assert!(p.validate().is_err());
}
