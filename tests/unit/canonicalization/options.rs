use super::*;
use crate::circuit::{CircuitSeq, xgate::XGate};
use std::process::Command;

#[test]
fn explicit_rule_l_budget_covers_recursion_and_resets_per_call() {
    // Six indistinguishable identity wires force Rule L at multiple depths.
    let polys = (0..6).map(|wire| vec![1u64 << wire]).collect::<Vec<_>>();
    let unlimited = CanonicalizationOptions::default();
    let expected = canonicalize_polys_4_with_options(polys.clone(), true, &unlimited).unwrap();
    assert!(canonicalize_polys_4_with_options(polys.clone(), false, &unlimited).is_err());
    for _ in 0..3 {
        let root_only = CanonicalizationOptions {
            rule_l_branch_cap: Some(6),
            ..unlimited
        };
        assert!(canonicalize_polys_4_with_options(polys.clone(), true, &root_only).is_err());
        let ample = CanonicalizationOptions {
            rule_l_branch_cap: Some(1_000),
            ..unlimited
        };
        assert_eq!(
            canonicalize_polys_4_with_options(polys.clone(), true, &ample).unwrap(),
            expected
        );
    }
    std::thread::scope(|scope| {
        for cap in [0, 6, 1_000, 1_000] {
            let polys = &polys;
            let expected = &expected;
            scope.spawn(move || {
                let options = CanonicalizationOptions {
                    rule_l_branch_cap: Some(cap),
                    ..unlimited
                };
                let result = canonicalize_polys_4_with_options(polys.clone(), true, &options);
                if cap <= 6 {
                    assert!(result.is_err());
                } else {
                    assert_eq!(&result.unwrap(), expected);
                }
            });
        }
    });
}

#[test]
fn explicit_window_budgets_do_not_replay_cached_successes() {
    let circuit = CircuitSeq {
        gates: vec![[0, 1, 2], [1, 0, 2], [2, 0, 1]],
    };
    let roomy = G57CanonicalizationOptions::default();
    for reversed in [false, true] {
        let expected = circuit.canonicalize_polys_single_with_options(reversed, &roomy);
        assert!(!expected.0.is_empty());
        let hashed = circuit.canonicalize_polys_single_hashed_with_options(reversed, &roomy);
        assert_eq!(
            hashed.0,
            Some(xxhash_rust::xxh3::xxh3_128(&polys_repr_blob(&expected.0)).to_le_bytes())
        );
        assert_eq!(
            (hashed.1, hashed.2),
            (expected.1.clone(), expected.2.clone())
        );
        let tight = G57CanonicalizationOptions {
            monomial_cap: Some(1),
            ..roomy
        };
        assert!(
            circuit
                .canonicalize_polys_single_with_options(reversed, &tight)
                .0
                .is_empty()
        );
        assert_eq!(
            circuit.canonicalize_polys_single_with_options(reversed, &roomy),
            expected
        );
    }
    let neg = circuit.canonicalize_polys_single_neg_with_options(&[1], &roomy);
    assert!(!neg.0.is_empty());
    assert!(
        circuit
            .canonicalize_polys_single_neg_with_options(
                &[1],
                &G57CanonicalizationOptions {
                    monomial_cap: Some(1),
                    ..roomy
                }
            )
            .0
            .is_empty()
    );
    assert_eq!(
        circuit.canonicalize_polys_single_neg_with_options(&[1], &roomy),
        neg
    );

    let gates = vec![XGate::conj(0, [(1, true), (2, true), (3, true)]).unwrap()];
    let mut options = xgate::XGateCanonicalizationOptions::default();
    let expected = xgate::canonicalize_xgates_single_with_options(&gates, false, &options).unwrap();
    options.max_degree = 2;
    assert!(matches!(
        xgate::canonicalize_xgates_single_with_options(&gates, false, &options),
        Err(xgate::XPolyError::DegreeExceeded { .. })
    ));
    options.max_degree = 0;
    let actual = xgate::canonicalize_xgates_single_with_options(&gates, false, &options).unwrap();
    assert_eq!(
        (actual.polys, actual.order, actual.used_wires),
        (expected.polys, expected.order, expected.used_wires)
    );
}

#[test]
fn explicit_calls_leave_legacy_first_reads_lazy_and_cached() {
    const CHILD: &str = "CANON_OPTIONS_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "canonicalization::options_tests::explicit_calls_leave_legacy_first_reads_lazy_and_cached", "--test-threads=1"])
            .env(CHILD, "1")
            .env("CANON_RULE_L_BRANCH_CAP", "1")
            .env("CANON_MONOMIAL_CAP", "1")
            .env("CANON_CACHE_MB", "1")
            .env("XPOLY_CANON_CACHE_MB", "1")
            .env_remove("COMPRESSION_TRACE")
            .env("COMPRESSION_TRACE_MS", "1")
            .env_remove("BENCH_CANON")
            .output().unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    // The child process runs this test alone. Environment mutations below
    // cannot race with another test or affect the parent test process.
    assert!(canonicalize_polys_4(Vec::new(), true).is_ok());
    let circuit = CircuitSeq {
        gates: vec![[0, 1, 2]],
    };
    assert!(
        !circuit
            .canonicalize_polys_single_with_options(false, &G57CanonicalizationOptions::default())
            .0
            .is_empty()
    );
    let gates = [XGate::conj(0, [(1, true), (2, false)]).unwrap()];
    assert!(
        xgate::canonicalize_xgates_single_with_options(
            &gates,
            false,
            &xgate::XGateCanonicalizationOptions::default()
        )
        .is_ok()
    );
    unsafe {
        std::env::set_var("CANON_RULE_L_BRANCH_CAP", " 23 ");
        std::env::set_var("CANON_MONOMIAL_CAP", " 29 ");
        std::env::set_var("CANON_CACHE_MB", "0");
        std::env::set_var("XPOLY_CANON_CACHE_MB", "invalid");
        std::env::set_var("COMPRESSION_TRACE", "");
        std::env::set_var("COMPRESSION_TRACE_MS", "37");
        std::env::set_var("BENCH_CANON", "");
    }
    assert_eq!(canon_rule_l_branch_cap(), Some(23));
    assert_eq!(canon_monomial_cap(), Some(29));
    assert_eq!(legacy_environment::canon_cache_cap_bytes(), 0);
    assert_eq!(
        legacy_environment::xpoly_canon_cache_cap_bytes(),
        1024 * 1024 * 1024
    );
    assert!(legacy_environment::compression_trace_enabled());
    assert_eq!(legacy_environment::compression_trace_threshold_ms(), 37);
    #[cfg(feature = "legacy-tools")]
    assert!(legacy_environment::bench_canon_enabled());
    unsafe {
        std::env::set_var("CANON_RULE_L_BRANCH_CAP", "31");
        std::env::set_var("CANON_MONOMIAL_CAP", "31");
        std::env::set_var("CANON_CACHE_MB", "31");
        std::env::set_var("XPOLY_CANON_CACHE_MB", "31");
        std::env::remove_var("COMPRESSION_TRACE");
        std::env::set_var("COMPRESSION_TRACE_MS", "31");
        std::env::remove_var("BENCH_CANON");
    }
    assert_eq!(canon_rule_l_branch_cap(), Some(23));
    assert_eq!(canon_monomial_cap(), Some(29));
    assert_eq!(legacy_environment::canon_cache_cap_bytes(), 0);
    assert_eq!(
        legacy_environment::xpoly_canon_cache_cap_bytes(),
        1024 * 1024 * 1024
    );
    assert!(legacy_environment::compression_trace_enabled());
    assert_eq!(legacy_environment::compression_trace_threshold_ms(), 37);
    #[cfg(feature = "legacy-tools")]
    assert!(legacy_environment::bench_canon_enabled());
    assert_eq!(
        circuit.canonicalize_polys_single(false),
        circuit
            .canonicalize_polys_single_with_options(false, &G57CanonicalizationOptions::default())
    );
}
