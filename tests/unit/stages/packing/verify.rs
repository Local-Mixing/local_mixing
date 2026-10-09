use super::*;
use crate::stages::packing::oracle_io::parse_oracle_mpmct;
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

/// A toy "sandwich": copy x (wires 0..2) to the answer wires 2..4 through
/// scratch wires 4..8, with C = (x0 ^= x1; x1 ^= NOT x0 AND x1) as g57s.
fn toy() -> (Vec<XGate>, Vec<XGate>) {
    let c = vec![g(0, false, &[(1, true)]), g(1, true, &[(0, false)])];
    let p = vec![
        g(4, false, &[(0, true)]),
        g(5, false, &[(1, true)]),
        g(4, false, &[(5, true)]),
        g(5, true, &[(4, false)]),
        g(2, false, &[(4, true)]),
        g(3, false, &[(5, true)]),
        g(6, false, &[(2, true), (3, false)]),
        g(6, false, &[(2, true), (3, false)]),
        g(7, false, &[]),
    ];
    (c, p)
}

#[test]
fn zero_slice_passes_and_detects_single_flip() {
    let (c, p) = toy();
    let d = deliver(&p, &[3, 6], 2, 8);
    assert_eq!(check_zero_slice(&c, &d, 2, 2048, 9).unwrap(), 2048);
    let mut bad = p.clone();
    bad[5] = g(3, false, &[(4, true)]);
    let d2 = deliver(&bad, &[3, 6], 2, 8);
    assert!(matches!(
        check_zero_slice(&c, &d2, 2, 1024, 9),
        Err(Stage4Error::Verify(_))
    ));
}

#[test]
fn equiv_detects_noncommuting_swap() {
    let a = vec![g(1, false, &[(0, true)]), g(2, false, &[(1, true)])];
    let b = vec![a[1].clone(), a[0].clone()];
    assert!(verify_equiv(&a, &a, 3, 77).is_ok());
    assert!(matches!(
        verify_equiv(&a, &b, 3, 77),
        Err(Stage4Error::Verify(_))
    ));
    // same-target writes commute
    let c = vec![g(2, false, &[(0, true)]), g(2, false, &[(1, true)])];
    let d = vec![c[1].clone(), c[0].clone()];
    assert!(verify_equiv(&c, &d, 3, 77).is_ok());
}

#[test]
fn window_check_is_exact_and_detects_rare_difference() {
    // a 12-input AND differs from its negation-free twin on 1 of 4096 inputs
    let ctrls: Vec<(u16, bool)> = (0..12).map(|w| (w, true)).collect();
    let mut ctrls2 = ctrls.clone();
    ctrls2[11].1 = false;
    let p = vec![g(12, false, &ctrls), g(13, false, &[(12, true)])];
    let q = vec![g(12, false, &ctrls2), g(13, false, &[(12, true)])];
    let d = deliver(&p, &[], 1, 14);
    let ok = check_windows_anf(&p, &[], &d, 16, 1).unwrap();
    assert_eq!((ok.windows, ok.exhaustive, ok.sampled), (1, 1, 0));
    // the deliverable of q checked against the gates of p
    let dq = deliver(&q, &[], 1, 14);
    assert!(check_windows_anf(&p, &[], &dq, 16, 1).is_err());
    // end-to-end eval check on 1024 random states would almost surely miss
    // nothing here either way; the exact window check is the guarantee.
    assert!(check_deliverable(&p, 14, &d, 1).is_ok());
}

#[test]
fn deliverable_check_rejects_a_wrong_body() {
    let c = parse_oracle_mpmct("mpmct1 6 4\n1 0 1 0 1\n2 1 1 1 0\n3 0 2 0 1 2 1\n5 0 0\n").unwrap();
    let d = deliver(&c.gates, &[2], 1, 6);
    assert!(check_deliverable(&c.gates, 6, &d, 1).is_ok());
    let mut other = c.gates.clone();
    other[3] = g(4, false, &[]);
    assert!(check_deliverable(&other, 6, &d, 1).is_err());
    assert!(check_windows_anf(&other, &[2], &d, 16, 1).is_err());
}
