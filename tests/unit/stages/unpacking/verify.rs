use super::*;
use crate::stages::packing::serialize::{Format, decode, encode_body};
use smallvec::SmallVec;

fn g(t: u16, comp: bool, ctrls: &[(u16, bool)]) -> XGate {
    XGate {
        target: t,
        comp,
        ctrls: SmallVec::from_slice(ctrls),
    }
}

fn deliver(gates: &[XGate], bounds: &[i64], n: u32, nw: usize) -> Deliverable {
    let mut body = Vec::new();
    encode_body(gates, bounds, gates.len(), nw, 2, &mut body).unwrap();
    decode(
        &body,
        Format::Raw {
            source_wires: n,
            state_wires: nw as u32,
        },
    )
    .unwrap()
}

#[test]
fn window_checks_accept_an_equivalent_respelling_and_reject_others() {
    // original: x0 ^= x1 x2 ; x3 ^= NOT x0
    let orig = vec![
        g(0, false, &[(1, true), (2, true)]),
        g(3, false, &[(0, false)]),
    ];
    let d = deliver(&orig, &[], 0, 8);
    let v = d.window(0);
    // a respelling: x3 ^= 1 ; x3 ^= x0 ... after x0's update: same function
    let other = vec![
        g(0, false, &[(1, true), (2, true)]),
        g(3, false, &[]),
        g(3, false, &[(0, true)]),
    ];
    assert_eq!(check_window_exact(&other, &v, 8, 1 << 10), Ok(()));
    assert_eq!(check_window_sampled(&other, &v, 8, 1), Ok(()));
    // wrong polarity
    let bad = vec![
        g(0, false, &[(1, true), (2, false)]),
        g(3, false, &[(0, false)]),
    ];
    assert_eq!(
        check_window_exact(&bad, &v, 8, 1 << 10),
        Err(ExactFailure::Mismatch)
    );
    assert!(check_window_sampled(&bad, &v, 8, 1).is_err());
    // an unrestored helper wire
    let dirty = vec![
        g(7, false, &[(1, true)]),
        g(0, false, &[(1, true), (2, true)]),
        g(3, false, &[(0, false)]),
    ];
    assert_eq!(
        check_window_exact(&dirty, &v, 8, 1 << 10),
        Err(ExactFailure::Mismatch)
    );
    assert!(
        check_window_sampled(&dirty, &v, 8, 1)
            .unwrap_err()
            .contains("wire 7")
    );
    // a gate outside the state
    let out = vec![g(9, false, &[])];
    assert!(check_window_sampled(&out, &v, 8, 1).is_err());
    // a tiny budget overflows
    assert!(matches!(
        check_window_exact(&orig, &v, 8, 0),
        Err(ExactFailure::Overflow(_))
    ));
}

#[test]
fn eval_view_lanes_matches_the_deliverable_evaluator() {
    let gates = vec![
        g(0, false, &[(1, true), (2, true)]),
        g(3, true, &[(0, false), (4, true)]),
        g(5, false, &[]),
    ];
    let d = deliver(&gates, &[], 0, 8);
    let v = d.window(0);
    let mut rng = PyRandom::new(5);
    let inp = random_lanes(&mut rng, 8);
    let mut st = inp.clone();
    d.eval_lanes16(&mut st);
    let got = eval_view_lanes(&v, &inp);
    assert_eq!(got.len(), 3);
    for (w, lanes) in got {
        assert_eq!(st[w as usize], lanes, "wire {w}");
    }
}

#[test]
fn whole_circuit_equivalence_and_zero_slice() {
    // C = x0 ^= x1 on 2 source wires; state 8: inputs 0..2, answer 2..4
    let c = vec![g(0, false, &[(1, true)])];
    // the "deliverable": copy x to the answer wires, apply C there
    let gates = vec![
        g(2, false, &[(0, true)]),
        g(3, false, &[(1, true)]),
        g(2, false, &[(3, true)]),
        g(7, false, &[(6, true)]), // scratch noise, zero on the slice
    ];
    let d = deliver(&gates, &[2], 2, 8);
    assert_eq!(d.window_count(), 2);
    assert_eq!(check_equiv(&gates, &d, 1, 2, "t").unwrap(), 2048);
    let mut reordered = gates.clone();
    reordered.swap(0, 1);
    assert!(check_equiv(&reordered, &d, 1, 1, "t").is_ok());
    let mut wrong = gates.clone();
    wrong[2] = g(2, false, &[(3, false)]);
    assert_eq!(
        check_equiv(&wrong, &d, 1, 1, "t").unwrap_err().exit_code(),
        4
    );
    assert_eq!(
        check_zero_slice_gates(&c, &gates, 2, 8, 1, 3).unwrap(),
        1024
    );
    assert_eq!(
        check_zero_slice_gates(&c, &wrong, 2, 8, 1, 3)
            .unwrap_err()
            .exit_code(),
        4
    );
    // n too large for the state
    assert_eq!(
        check_zero_slice_gates(&c, &gates, 5, 8, 1, 3)
            .unwrap_err()
            .exit_code(),
        2
    );
    // a gate beyond the state
    let mut wide = gates.clone();
    wide.push(g(9, false, &[]));
    assert_eq!(
        check_equiv(&wide, &d, 1, 1, "t").unwrap_err().exit_code(),
        4
    );
}
