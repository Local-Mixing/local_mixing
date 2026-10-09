use super::*;
use crate::stages::packing::pyrand::PyRandom;
use smallvec::SmallVec;

fn g(t: u16, comp: bool, ctrls: &[(u16, bool)]) -> XGate {
    XGate {
        target: t,
        comp,
        ctrls: SmallVec::from_slice(ctrls),
    }
}

fn set_of(keys: &[u64]) -> Poly<1> {
    let mut s = PySet::new();
    for &k in keys {
        s.add(Mono([k]));
    }
    s
}

fn sorted(p: &Poly<1>) -> Vec<u64> {
    let mut v: Vec<u64> = p.iter().map(|m| m.0[0]).collect();
    v.sort_unstable();
    v
}

#[test]
fn worked_example_toffoli_and_cnot() {
    // t2 ^= x0 AND NOT x1 ; then t3 ^= t2
    let gates = [
        g(2, false, &[(0, true), (1, false)]),
        g(3, false, &[(2, true)]),
    ];
    let mut st = MulStats::default();
    let a = window_anf::<1>(&gates, 1 << 30, CapSemantics::CPython39, &mut st).unwrap();
    // x0(1+x1) = x0 + x0x1
    let x0 = 1u64;
    let x01 = 0b11;
    let w2: Vec<u64> = a.changed[0].1.iter().map(|m| m.0[0]).collect();
    let w3: Vec<u64> = a.changed[1].1.iter().map(|m| m.0[0]).collect();
    assert_eq!(a.changed[0].0, 2);
    assert_eq!(w2, vec![x0, x01, 1 << 2]);
    assert_eq!(a.changed[1].0, 3);
    assert_eq!(w3, vec![x0, x01, 1 << 2, 1 << 3]);
    assert_eq!(a.total, 7);
    assert_eq!(a.support(), vec![0, 1, 2, 3]);
}

#[test]
fn pair_toggle_not_dedup() {
    // (x0 + x1)(x0 + x1) = x0 + x1 (cross terms cancel pairwise)
    let p = set_of(&[1, 2]);
    let mut st = MulStats::default();
    let r = mul(&p, &p, 100, CapSemantics::CPython39, &mut st).unwrap();
    assert_eq!(sorted(&r), vec![1, 2]);
}

#[test]
fn swap_only_when_strictly_larger() {
    // outer = first operand on a tie; the result's table order shows it.
    let a = set_of(&[1, 2]);
    let b = set_of(&[4, 8]);
    let mut st = MulStats::default();
    let ab = mul(&a, &b, 100, CapSemantics::CPython39, &mut st).unwrap();
    let ba = mul(&b, &a, 100, CapSemantics::CPython39, &mut st).unwrap();
    let oab: Vec<u64> = ab.iter().map(|m| m.0[0]).collect();
    let oba: Vec<u64> = ba.iter().map(|m| m.0[0]).collect();
    assert_eq!(sorted(&ab), sorted(&ba));
    // insertion orders differ (5,9,6,10 vs 5,6,9,10); with 8 slots both land by hash
    assert_eq!(oab.len(), 4);
    assert_eq!(oba.len(), 4);
    // strictly larger first operand: swapped, so row count = |smaller|
    let big = set_of(&[1, 2, 4]);
    let small = set_of(&[8]);
    let mut st2 = MulStats::default();
    // cap 2: |small|=1 outer -> 1 row of 3 -> 3 > 2 aborts at the first row
    assert!(mul(&big, &small, 2, CapSemantics::CPython39, &mut st2).is_none());
    assert_eq!(st2.ambiguous, 0);
}

#[test]
fn running_sum_counts_identity_entries() {
    // three NOT-pairs leave every wire identity-valued but polys hold them
    let gates = [
        g(0, false, &[]),
        g(1, false, &[]),
        g(0, false, &[]),
        g(1, false, &[]),
    ];
    let mut st = MulStats::default();
    // after gate 2: polys {0: x0+1, 1: x1+1} sum 4 > 4*0? use budget 1: 4 > 4 false
    let r = window_anf::<1>(&gates, 1, CapSemantics::CPython39, &mut st).unwrap();
    assert_eq!(r.total, 0);
    assert!(r.changed.is_empty());
    let gates3 = [g(0, false, &[]), g(1, false, &[]), g(2, false, &[])];
    // sum after gate 3 = 6 > 4 -> RunningSum
    assert_eq!(
        window_anf::<1>(&gates3, 1, CapSemantics::CPython39, &mut st),
        Err(AnfOverflow::RunningSum)
    );
    // identity-valued entries count: x0 toggled twice is identity but kept in polys
    let gates4 = [
        g(0, false, &[]),
        g(0, false, &[]),
        g(1, false, &[]),
        g(2, false, &[]),
    ];
    // polys after gate 4: {0: x0 (1), 1: 2, 2: 2} = 5 > 4
    assert_eq!(
        window_anf::<1>(&gates4, 1, CapSemantics::CPython39, &mut st),
        Err(AnfOverflow::RunningSum)
    );
}

/// u = x_c + sum of k inputs on c1 and c2, then t ^= c1 c2: the product's
/// CPython prefix exceeds a cap the finished product (3k+1) fits.
fn orderdep_window(k: u16) -> Vec<XGate> {
    let (c1, c2, t) = (100u16, 101, 102);
    let mut v = Vec::new();
    for x in 0..k {
        v.push(g(c1, false, &[(x, true)]));
        v.push(g(c2, false, &[(x, true)]));
    }
    v.push(g(t, false, &[(c1, true), (c2, true)]));
    v
}

#[test]
fn prefix_abort_cpython_vs_canonical() {
    let w = orderdep_window(64);
    // finished product 3*64+1 = 193 monomials; prefix ~ 66^2/4 = 1089
    let mut found = false;
    for cap in (200..1100).step_by(25) {
        let mut sc = MulStats::default();
        let mut sk = MulStats::default();
        let mut polys = WindowState::<2>::new(cap);
        let mut canon = WindowState::<2>::new(cap);
        let mut ok_c = true;
        let mut ok_k = true;
        for gate in &w {
            ok_c &= polys.compose(gate, CapSemantics::CPython39, &mut sc);
            ok_k &= canon.compose(gate, CapSemantics::Canonical, &mut sk);
        }
        assert!(
            ok_k,
            "canonical never aborts a product that fits (cap {cap})"
        );
        if !ok_c {
            found = true;
            assert_eq!(sc.aborts, 1);
            assert_eq!(sc.aborts_order_dependent, 1);
            assert_eq!(sk.aborts, 0);
            assert_eq!(sk.aborts_order_dependent, 1);
        }
    }
    assert!(found, "no cap made the CPython prefix abort");
}

#[test]
fn compose_failure_mutates_nothing() {
    let w = orderdep_window(40);
    let mut st = MulStats::default();
    let mut ws = WindowState::<2>::new(150);
    for gate in &w[..w.len() - 1] {
        assert!(ws.compose(gate, CapSemantics::CPython39, &mut st));
    }
    let before = ws.monos();
    assert!(!ws.compose(&w[w.len() - 1], CapSemantics::CPython39, &mut st));
    assert_eq!(ws.monos(), before);
    ws.reset();
    assert_eq!(ws.monos(), 0);
}

#[test]
fn order_dependent_counter_zero_when_not_ambiguous() {
    let a = set_of(&[1, 2, 4]);
    let b = set_of(&[8, 16, 32]);
    let mut st = MulStats::default();
    // 9 distinct products; cap 5: (3-1)*3 = 6 > 5 ambiguous, full 9 > 5 -> not order-dependent
    assert!(mul(&a, &b, 5, CapSemantics::CPython39, &mut st).is_none());
    assert_eq!(
        (st.ambiguous, st.aborts, st.aborts_order_dependent),
        (1, 1, 0)
    );
    // cap 6: not ambiguous ((3-1)*3 = 6), aborts only at the last row
    let mut st2 = MulStats::default();
    assert!(mul(&a, &b, 6, CapSemantics::Canonical, &mut st2).is_none());
    assert_eq!(
        (st2.ambiguous, st2.aborts, st2.aborts_order_dependent),
        (0, 1, 0)
    );
}

/// The ANF by brute force: truth tables over the touched wires, Möbius transform.
#[allow(clippy::needless_range_loop)]
fn brute_anf(gates: &[XGate], nw: usize) -> Vec<(u16, Vec<u64>)> {
    let mut touched: Vec<usize> = gates
        .iter()
        .flat_map(|x| {
            x.ctrls
                .iter()
                .map(|c| c.0 as usize)
                .chain([x.target as usize])
        })
        .collect();
    touched.sort_unstable();
    touched.dedup();
    let f = touched.len();
    let mut tables: Vec<Vec<u8>> = vec![vec![0; 1 << f]; touched.len()];
    for a in 0..(1usize << f) {
        let mut st = vec![0u64; nw.div_ceil(64)];
        for (j, &w) in touched.iter().enumerate() {
            st[w >> 6] |= (((a >> j) & 1) as u64) << (w & 63);
        }
        for x in gates {
            x.apply_limbs(&mut st);
        }
        for (j, &w) in touched.iter().enumerate() {
            tables[j][a] = ((st[w >> 6] >> (w & 63)) & 1) as u8;
        }
    }
    let mut out = Vec::new();
    for (j, &w) in touched.iter().enumerate() {
        let mut t = tables[j].clone();
        for i in 0..f {
            for a in 0..(1usize << f) {
                if a >> i & 1 == 1 {
                    t[a] ^= t[a ^ (1 << i)];
                }
            }
        }
        let mut monos: Vec<u64> = (0..(1usize << f))
            .filter(|&a| t[a] == 1)
            .map(|a| {
                (0..f)
                    .filter(|&i| a >> i & 1 == 1)
                    .map(|i| 1u64 << touched[i])
                    .fold(0, |x, y| x | y)
            })
            .collect();
        monos.sort_unstable();
        if monos != vec![1u64 << w] {
            out.push((w as u16, monos));
        }
    }
    out
}

#[test]
fn window_anf_equals_moebius_of_truth_table() {
    let mut r = PyRandom::new(42);
    for trial in 0..300 {
        let nw = 10usize;
        let m = 1 + (r.genrand_u32() % 14) as usize;
        let gates: Vec<XGate> = (0..m)
            .map(|_| {
                let t = (r.genrand_u32() % nw as u32) as u16;
                let k = (r.genrand_u32() % 4) as usize;
                let mut ctrls: Vec<(u16, bool)> = Vec::new();
                while ctrls.len() < k {
                    let c = (r.genrand_u32() % nw as u32) as u16;
                    if c != t && !ctrls.iter().any(|x| x.0 == c) {
                        ctrls.push((c, r.genrand_u32() & 1 == 1));
                    }
                }
                ctrls.sort_unstable();
                g(t, r.genrand_u32() & 1 == 1, &ctrls)
            })
            .collect();
        let mut st = MulStats::default();
        let a = window_anf::<1>(&gates, 1 << 30, CapSemantics::CPython39, &mut st).unwrap();
        let got: Vec<(u16, Vec<u64>)> = a
            .changed
            .iter()
            .map(|(w, p)| (*w, p.iter().map(|m| m.0[0]).collect()))
            .collect();
        assert_eq!(got, brute_anf(&gates, nw), "trial {trial}");
    }
}
