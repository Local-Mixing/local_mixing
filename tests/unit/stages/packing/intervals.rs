use super::*;
use crate::stages::packing::fingerprint::Spans;
use rand::{Rng, SeedableRng, rngs::StdRng};

fn g(t: u16, ctrls: &[u16]) -> XGate {
    XGate::conj(t, ctrls.iter().map(|&c| (c, true))).unwrap()
}

fn conflict(x: &XGate, y: &XGate) -> bool {
    x.ctrls.iter().any(|&(c, _)| c == y.target) || y.ctrls.iter().any(|&(c, _)| c == x.target)
}

/// Gates of (b, e) reachable from b in the conflict order, then those of them
/// from which e is reachable through such gates; plus b and e (sorted).
fn brute(gates: &[XGate], b: usize, e: usize) -> Vec<u32> {
    let mut fwd = vec![false; gates.len()];
    fwd[b] = true;
    for i in b + 1..e {
        fwd[i] = (b..i).any(|j| fwd[j] && conflict(&gates[j], &gates[i]));
    }
    let mut bwd = vec![false; gates.len()];
    bwd[e] = true;
    for i in (b + 1..e).rev() {
        bwd[i] = fwd[i] && (i + 1..=e).any(|j| bwd[j] && conflict(&gates[i], &gates[j]));
    }
    let mut m: Vec<u32> = (b + 1..e).filter(|&i| bwd[i]).map(|i| i as u32).collect();
    m.push(b as u32);
    m.push(e as u32);
    m.sort_unstable();
    m
}

fn random_circuit(rng: &mut StdRng, n: u16, m: usize) -> Vec<XGate> {
    (0..m)
        .map(|_| {
            let t = rng.random_range(0..n);
            let k = rng.random_range(0..3usize);
            let mut cs: Vec<u16> = Vec::new();
            while cs.len() < k {
                let c = rng.random_range(0..n);
                if c != t && !cs.contains(&c) {
                    cs.push(c);
                }
            }
            g(t, &cs)
        })
        .collect()
}

#[test]
fn interval_set_matches_bruteforce_taint() {
    let mut rng = StdRng::seed_from_u64(7);
    for _ in 0..300 {
        let n = rng.random_range(3..12u16);
        let gates = random_circuit(&mut rng, n, 60);
        let b = rng.random_range(0..50);
        let e = rng.random_range(b + 1..60);
        let mut got = interval_set(&gates, b, e, n as usize, usize::MAX).unwrap();
        got.sort_unstable();
        assert_eq!(got, brute(&gates, b, e), "b={b} e={e}");
    }
}

#[test]
fn cap_returns_none_at_cap_minus_2() {
    // chain b -> g1 -> ... -> g5 -> e: 5 interior members, 7 in all
    let mut gates = vec![g(1, &[0])];
    for w in 2..7u16 {
        gates.push(g(w, &[w - 1]));
    }
    gates.push(g(1, &[6]));
    let e = gates.len() - 1;
    let m = interval_set(&gates, 0, e, 8, 7).unwrap();
    assert_eq!(m, vec![5, 4, 3, 2, 1, 0, 6]); // backward walk order, then b, e
    assert!(interval_set(&gates, 0, e, 8, 6).is_none());
}

#[test]
fn rescue_next_next_writer() {
    // b: w1 ^= w0; chain g1 (reads w1) -> ... -> e (writes w1, reads the
    // chain's end); e2 = NOT w1 depends only on g1 (it reads w1).
    let mut gates = vec![g(1, &[0]), g(2, &[1])];
    for w in 3..8u16 {
        gates.push(g(w, &[w - 1]));
    }
    gates.push(g(1, &[7])); // e
    let e = gates.len() - 1;
    gates.push(XGate::x_gate(1)); // e2
    let e2 = gates.len() - 1;
    let sp = Spans {
        spans: vec![(0, e as u32, 0)],
        forced: vec![],
        agree: 0,
    };
    let iv = build_intervals(&gates, &sp, 8, 3, 3000);
    assert_eq!(iv.stats.rescued, 1);
    assert_eq!(iv.stats.winnable, 1);
    assert_eq!(iv.ie, vec![e2 as u32]);
    assert_eq!(iv.members(0), &[0, 1, e2 as u32]);
    // over the span cap the rescue is not tried
    let iv = build_intervals(&gates, &sp, 8, 3, e2);
    assert_eq!((iv.stats.rescued, iv.stats.disq_big), (0, 1));
    // a long span disqualifies before any interval is built
    let iv = build_intervals(&gates, &sp, 8, 3, e - 1);
    assert_eq!((iv.stats.disq_long, iv.len()), (1, 0));
}

#[test]
fn iid_order_is_first_appearance() {
    let mut rng = StdRng::seed_from_u64(3);
    let gates = random_circuit(&mut rng, 6, 20);
    let sp = Spans {
        spans: vec![(0, 5, 7), (1, 6, 3), (2, 8, 7), (3, 9, 11)],
        forced: vec![11],
        agree: 0,
    };
    let iv = build_intervals(&gates, &sp, 6, 512, 3000);
    assert_eq!(iv.stats.by_val, 3);
    assert_eq!(iv.stats.winnable, 2); // 11 is output-forced: neither won nor disqualified
    assert_eq!(iv.val_a, vec![7, 3]);
    assert_eq!(iv.value_iids(0), 0..2);
    assert_eq!(iv.value_iids(1), 2..3);
    assert_eq!(
        (iv.ib.clone(), iv.ie.clone()),
        (vec![0, 2, 1], vec![5, 8, 6])
    );
    for iid in 0..iv.len() {
        assert!(iv.members(iid).windows(2).all(|w| w[0] < w[1]));
        for &m in iv.members(iid) {
            assert!(iv.member_of(m as usize).contains(&(iid as u32)));
        }
    }
}
