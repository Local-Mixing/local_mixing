use super::*;
use crate::stages::packing::serialize::{Deliverable, Format, decode, encode_body};
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

#[test]
fn universe_is_support_and_changed_plus_the_lowest_pool_wires() {
    // window: x10 ^= x20 x30 ; x40 ^= 1  (P_10 = x10 + x20 x30, P_40 = x40 + 1:
    // the support holds every wire of some monomial, the changed wires included)
    let gates = vec![g(10, false, &[(20, true), (30, true)]), g(40, false, &[])];
    let d = deliver(&gates, &[], 64);
    let v = d.window(0);
    assert_eq!(v.support, &[10, 20, 30, 40]);
    let u = LocalUniverse::for_window(&v, 64, 4);
    assert_eq!(u.wires, vec![0, 1, 2, 3, 10, 20, 30, 40]);
    assert_eq!(u.local(20), 5);
    assert_eq!(u.global(7), 40);
    // pool larger than the state: every wire
    let u = LocalUniverse::for_window(&v, 8, 100);
    assert_eq!(
        u.wires,
        (0..8)
            .map(|w| w as u16)
            .chain([10, 20, 30, 40])
            .collect::<Vec<_>>()
    );
    // no pool
    let u0 = LocalUniverse::for_window(&v, 64, 0);
    assert_eq!(u0.wires, vec![10, 20, 30, 40]);
    assert_eq!(u0.len(), 4);
    assert!(!u0.is_empty());
}

#[test]
fn spec_and_stats_follow_the_decoded_window() {
    let gates = vec![
        g(10, false, &[(20, true), (30, true)]),
        g(10, true, &[(20, false)]),
        g(40, false, &[]),
    ];
    let d = deliver(&gates, &[], 64);
    let v = d.window(0);
    let u = LocalUniverse::for_window(&v, 64, 0);
    let s: WindowSpec<1> = build_spec(&v, &u);
    assert_eq!(s.num_wires, 4);
    // local ids: 10->0, 20->1, 30->2, 40->3
    // P_10 = x10 + x20 x30 + (1 + x20 + 1)... : x10 ^= x20 x30 then x10 ^= NOT(NOT x20) = x20
    // => P_10 = x10 + x20 + x20 x30 ; P_40 = x40 + 1
    assert_eq!(s.polys.len(), 2);
    assert_eq!(s.polys[0].0, 0);
    let mut p0 = s.polys[0].1.clone();
    p0.sort();
    let mut want = vec![Mono::<1>::var(0), Mono::var(1), Mono::from_vars(&[1, 2])];
    want.sort();
    assert_eq!(p0, want);
    assert_eq!(s.polys[1], (3, vec![Mono::ONE, Mono::var(3)]));
    let st = anf_stats(&v);
    assert_eq!(
        st,
        AnfStats {
            changed_wires: 2,
            input_vars: 4,
            monomials: 5,
            max_degree: 2
        }
    );
}

#[test]
fn gates_map_back_to_global_wires_with_sorted_controls() {
    let u = LocalUniverse {
        wires: vec![3, 7, 9],
    };
    let out = u.to_xgates(&[
        SGate::tof(0, 2, false, 1, true, true),
        SGate::cnot(1, 0, true, false),
        SGate::not(2),
    ]);
    assert_eq!(out[0], g(3, true, &[(7, true), (9, false)]));
    assert_eq!(out[1], g(7, false, &[(3, true)]));
    assert_eq!(out[2], XGate::x_gate(9));
}
