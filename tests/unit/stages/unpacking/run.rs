use super::*;
use crate::canonicalization::gate_order::xgate_canonicalize;
use crate::stages::packing::serialize::{Format, decode, encode_body};
use crate::stages::unpacking::mono::Rng;
use crate::stages::unpacking::verify::check_equiv;
use smallvec::SmallVec;

fn g(t: u16, comp: bool, ctrls: &[(u16, bool)]) -> XGate {
    XGate {
        target: t,
        comp,
        ctrls: SmallVec::from_slice(ctrls),
    }
}

fn deliver(gates: &[XGate], bounds: &[i64], nw: usize) -> Deliverable {
    let mut body = Vec::new();
    encode_body(gates, bounds, gates.len(), nw, 2, &mut body).unwrap();
    decode(
        &body,
        Format::Raw {
            source_wires: 0,
            state_wires: nw as u32,
        },
    )
    .unwrap()
}

fn random_gates(n: u16, m: usize, rng: &mut Rng) -> Vec<XGate> {
    (0..m)
        .map(|_| {
            let t = rng.below(n as u64) as u16;
            let mut a = rng.below(n as u64) as u16;
            while a == t {
                a = rng.below(n as u64) as u16;
            }
            let mut b = rng.below(n as u64) as u16;
            while b == t || b == a {
                b = rng.below(n as u64) as u16;
            }
            let mut lits = vec![(a, rng.below(2) == 1)];
            if rng.below(2) == 1 {
                lits.push((b, rng.below(2) == 1));
            }
            let mut x = XGate::conj(t, lits).unwrap();
            x.comp = rng.below(2) == 1;
            x
        })
        .collect()
}

fn quick_params() -> UnpackParams {
    let mut p = UnpackParams::default();
    p.synth.restarts = 2;
    p.synth.time_limit = 10.0;
    p.equiv_batches = 1;
    p
}

#[test]
fn unpacks_a_small_deliverable_and_the_canonical_circuit_is_equivalent() {
    let mut rng = Rng::new(21);
    let gates = random_gates(10, 40, &mut rng);
    let bounds = vec![10i64, 20, 30];
    let d = deliver(&gates, &bounds, 16);
    let p = quick_params();
    let seen = std::sync::Mutex::new(0usize);
    let results = unpack_windows(&d, None, &p, &|_, _| *seen.lock().unwrap() += 1).unwrap();
    assert_eq!(results.len(), 4);
    assert_eq!(*seen.lock().unwrap(), 4);
    for (i, r) in results.iter().enumerate() {
        assert_eq!(r.idx, i);
        assert_eq!(r.status, WinStatus::Ok, "window {i}: {}", r.note);
        assert_eq!(r.synth_status, "converged");
        assert!(r.exact_budget >= EXACT_BUDGETS[0]);
        assert_eq!(r.orig, None);
        assert_eq!(r.width, 1);
        assert_eq!(r.universe, 16); // pool covers the whole 16-wire state
    }
    let s = UnpackSummary::of(&results);
    assert_eq!((s.windows, s.synthesized, s.fallback), (4, 4, 0));
    assert_eq!(s.gates_in, None);
    assert_eq!(s.ratio(), None);
    let (concat, offsets) = concatenate(&results);
    assert_eq!(offsets[0], 0);
    assert_eq!(concat.len(), s.gates_out);
    check_equiv(&concat, &d, 1, 1, "concat").unwrap();
    let mut canon = concat;
    xgate_canonicalize(&mut canon);
    check_equiv(&canon, &d, 2, 1, "canonical").unwrap();
    let tsv = windows_tsv(&results, &offsets);
    let header: Vec<&str> = tsv.lines().next().unwrap().split('\t').collect();
    assert_eq!(
        &header[..12],
        &[
            "idx",
            "orig_lo",
            "orig_hi",
            "orig_gates",
            "syn_gates",
            "changed_wires",
            "input_vars",
            "monomials",
            "max_degree",
            "status",
            "seconds",
            "syn_offset"
        ]
    );
    assert_eq!(tsv.lines().count(), 5);
    let row: Vec<&str> = tsv.lines().nth(1).unwrap().split('\t').collect();
    assert_eq!(row.len(), header.len());
    assert_eq!(&row[1..4], &["-1", "-1", "-1"]);
    assert_eq!(row[9], "ok");
    assert_eq!(row[11], "0");
}

#[test]
fn fallback_uses_the_packed_window_only_when_it_matches() {
    let mut rng = Rng::new(5);
    let gates = random_gates(8, 20, &mut rng);
    let bounds = vec![10i64];
    let d = deliver(&gates, &bounds, 8);
    let cuts = vec![0usize, 10, 20];
    // a synthesizer that cannot succeed: no restarts to speak of and a
    // step cap of 64 gates with a stall limit of 1 -> may still converge on
    // easy windows, so force the failure path through `beam 0`, no macros
    // and a tiny step factor
    let mut p = quick_params();
    p.synth.step_factor = 0;
    p.synth.restarts = 1;
    p.synth.use_macros = false;
    p.synth.beam = 0;
    p.synth.back_k = 0;
    // (validate() rejects step_factor 0: unpack_windows must refuse it)
    assert_eq!(
        unpack_windows(&d, None, &p, &|_, _| {})
            .unwrap_err()
            .exit_code(),
        2
    );
    p.synth.step_factor = 1;
    p.synth.stall_limit = 1;
    // no packed circuit: a failed window is exit 4 (or every window converges)
    let r = unpack_windows(&d, None, &p, &|_, _| {});
    let packed = PackedWindows {
        gates: &gates,
        cuts: cuts.clone(),
    };
    let with = unpack_windows(&d, Some(&packed), &p, &|_, _| {}).unwrap();
    match r {
        Ok(ok) => assert!(ok.iter().all(|w| w.status == WinStatus::Ok)),
        Err(e) => {
            assert_eq!(e.exit_code(), 4);
            assert!(with.iter().any(|w| w.status == WinStatus::Fallback));
            // --no-fallback: the same failure with a packed circuit
            let mut nf = p.clone();
            nf.allow_fallback = false;
            assert_eq!(
                unpack_windows(&d, Some(&packed), &nf, &|_, _| {})
                    .unwrap_err()
                    .exit_code(),
                4
            );
        }
    }
    for w in &with {
        assert_eq!(w.orig, Some((cuts[w.idx], cuts[w.idx + 1])));
        if w.status == WinStatus::Fallback {
            assert_eq!(w.gates, gates[cuts[w.idx]..cuts[w.idx + 1]].to_vec());
        }
    }
    let s = UnpackSummary::of(&with);
    assert_eq!(s.gates_in, Some(20));
    assert!(s.ratio().is_some());
    let (concat, _) = concatenate(&with);
    check_equiv(&concat, &d, 9, 1, "concat").unwrap();
    // a packed circuit that does not realize the deliverable is exit 4
    let mut wrong = gates.clone();
    wrong[3] = g(0, true, &[(1, true)]);
    wrong[15] = g(2, false, &[]);
    let bad = PackedWindows {
        gates: &wrong,
        cuts: cuts.clone(),
    };
    let e = unpack_windows(&d, Some(&bad), &p, &|_, _| {});
    if let Err(e) = e {
        assert_eq!(e.exit_code(), 4);
    }
    // a packed circuit with another window count is exit 3
    let bad = PackedWindows {
        gates: &gates,
        cuts: vec![0, 5, 10, 20],
    };
    assert_eq!(
        unpack_windows(&d, Some(&bad), &p, &|_, _| {})
            .unwrap_err()
            .exit_code(),
        3
    );
}

#[test]
fn empty_windows_give_no_gates() {
    let gates = vec![g(0, false, &[(1, true)]), g(0, false, &[(1, true)])];
    let d = deliver(&gates, &[1], 4);
    let p = quick_params();
    let results = unpack_windows(&d, None, &p, &|_, _| {}).unwrap();
    assert_eq!(results.len(), 2);
    // window 0 = x0 ^= x1; window 1 = the same (together the identity)
    assert!(!results[0].gates.is_empty());
    let d2 = deliver(&gates, &[], 4);
    let results = unpack_windows(&d2, None, &p, &|_, _| {}).unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].gates.is_empty());
    assert_eq!(results[0].status, WinStatus::Ok);
}

#[test]
fn quantile_is_nearest_rank() {
    let xs = [5.0, 1.0, 3.0, 2.0, 4.0];
    assert_eq!(quantile(&xs, 0.0), 1.0);
    assert_eq!(quantile(&xs, 0.5), 3.0);
    assert_eq!(quantile(&xs, 1.0), 5.0);
    assert_eq!(quantile(&xs, 0.9), 5.0);
    assert!(quantile(&[], 0.5).is_nan());
}
