use super::*;
use rand::{SeedableRng, rngs::StdRng};

fn flat(n: usize) -> (Vec<[u16; 3]>, Vec<Tag>) {
    let gates: Vec<[u16; 3]> = (0..n)
        .map(|i| [i as u16, (i + 1) as u16, (i + 2) as u16])
        .collect();
    let tags: Vec<Tag> = (0..n).map(Tag::survivor).collect();
    (gates, tags)
}

#[test]
fn roundtrip_flat() {
    for n in [0usize, 1, 5, 1023, 1024, 1025, 5000] {
        let (g, t) = flat(n);
        let sc = SegCircuit::from_flat(&g, &t);
        assert_eq!(sc.len(), n);
        let (g2, t2) = sc.to_flat();
        assert_eq!(g, g2);
        assert_eq!(t, t2);
    }
}

#[test]
fn local_splice_matches_vec() {
    let (mut g, mut t) = flat(5000);
    let mut sc = SegCircuit::from_flat(&g, &t);
    // a local replacement well inside one chunk
    let idx = 1500;
    let new_g = vec![[9, 9, 9], [8, 8, 8], [7, 7, 7]];
    let new_t = vec![Tag(100), Tag(101), Tag(102)];
    sc.splice(idx, 2, &new_g, &new_t);
    g.splice(idx..idx + 2, new_g.iter().copied());
    t.splice(idx..idx + 2, new_t.iter().copied());
    let (g2, t2) = sc.to_flat();
    assert_eq!(g, g2);
    assert_eq!(t, t2);
    assert_eq!(sc.len(), g.len());
}

#[test]
fn cross_chunk_splice_matches_vec() {
    let (mut g, mut t) = flat(5000);
    let mut sc = SegCircuit::from_flat(&g, &t);
    // a removal spanning a chunk boundary (~1024)
    let idx = 1000;
    let rem = 100; // crosses into the next chunk
    let new_g = vec![[1, 2, 3]];
    let new_t = vec![Tag(42)];
    sc.splice(idx, rem, &new_g, &new_t);
    g.splice(idx..idx + rem, new_g.iter().copied());
    t.splice(idx..idx + rem, new_t.iter().copied());
    let (g2, t2) = sc.to_flat();
    assert_eq!(g, g2);
    assert_eq!(t, t2);
}

#[test]
fn big_insert_rebalances_and_matches() {
    let (mut g, mut t) = flat(100);
    let mut sc = SegCircuit::from_flat(&g, &t);
    let new_g: Vec<[u16; 3]> = (0..5000).map(|i| [i as u16, 0, 1]).collect();
    let new_t: Vec<Tag> = vec![Tag(7); 5000];
    sc.splice(50, 10, &new_g, &new_t);
    g.splice(50..60, new_g.iter().copied());
    t.splice(50..60, new_t.iter().copied());
    let (g2, t2) = sc.to_flat();
    assert_eq!(g, g2);
    assert_eq!(t, t2);
}

#[test]
fn min_gen_and_anchor() {
    // tags with a unique minimum
    let g: Vec<[u16; 3]> = (0..3000).map(|i| [i as u16, 0, 1]).collect();
    let mut t: Vec<Tag> = vec![Tag(5); 3000];
    t[1234] = Tag(2); // unique min
    let sc = SegCircuit::from_flat(&g, &t);
    assert_eq!(sc.min_gen(), 2);
    let mut rng = StdRng::seed_from_u64(1);
    for _ in 0..20 {
        assert_eq!(sc.random_min_gen_index(&mut rng), Some(1234));
    }
}

use crate::db_mixing::transpositions::Transpositions;

fn rand_transpositions(n: usize, k: usize, rng: &mut impl Rng) -> (Transpositions, Vec<u8>) {
    let mut ts = Vec::new();
    for _ in 0..k {
        let a = rng.random_range(0..n) as u16;
        let mut b = rng.random_range(0..n) as u16;
        while b == a {
            b = rng.random_range(0..n) as u16;
        }
        ts.push((a, b, rng.random_range(0..4u16)));
    }
    let mut neg = vec![0u8; n];
    for w in 0..n {
        neg[w] = rng.random_range(0..2u8);
    }
    (Transpositions { transpositions: ts }, neg)
}

#[test]
fn samftail_perm_matches_evaluate() {
    let n = 40;
    let mut rng = StdRng::seed_from_u64(0xa1);
    for _ in 0..50 {
        let (t, neg) = rand_transpositions(n, 30, &mut rng);
        let tail = SamfTail::from_transpositions(&t, &neg, n);
        for w in 0..n as u16 {
            assert_eq!(tail.perm[w as usize], t.evaluate(w));
        }
    }
}

// Combine two SAMF rounds the way the shooting driver does, returning the compact result.
fn combine_like_code(
    ta: &Transpositions,
    na: &[u8],
    tb: &Transpositions,
    nb: &[u8],
    n: usize,
) -> (Transpositions, Vec<u8>) {
    let t = ta.concat(tb);
    let mut neg = nb.to_vec();
    for w in 0..n {
        if na[w] == 1 {
            neg[tb.evaluate(w as u16) as usize] ^= 1;
        }
    }
    (t, neg)
}

#[test]
fn samftail_then_matches_concat() {
    let n = 40;
    let mut rng = StdRng::seed_from_u64(0xb2);
    for _ in 0..50 {
        let (ta, na) = rand_transpositions(n, 20, &mut rng);
        let (tb, nb) = rand_transpositions(n, 20, &mut rng);
        let a = SamfTail::from_transpositions(&ta, &na, n);
        let b = SamfTail::from_transpositions(&tb, &nb, n);
        let combined = a.then(&b);
        let (tc, nc) = combine_like_code(&ta, &na, &tb, &nb, n);
        let expect = SamfTail::from_transpositions(&tc, &nc, n);
        assert_eq!(combined, expect);
    }
}

fn rand_perm_tail(n: usize, rng: &mut impl Rng) -> SamfTail {
    let mut perm: Vec<u16> = (0..n as u16).collect();
    for i in (1..n).rev() {
        let j = rng.random_range(0..=i);
        perm.swap(i, j);
    }
    let neg: Vec<u8> = (0..n).map(|_| rng.random_range(0..2u8)).collect();
    SamfTail { perm, neg }
}

#[test]
fn ledger_materialize_matches_per_gate_inbound() {
    let n = 20;
    let mut rng = StdRng::seed_from_u64(0xd4);
    for _ in 0..30 {
        let m = 200;
        let gates: Vec<[u16; 3]> = (0..m)
            .map(|_| {
                [
                    rng.random_range(0..n) as u16,
                    rng.random_range(0..n) as u16,
                    rng.random_range(0..n) as u16,
                ]
            })
            .collect();
        // ledger: a handful of perm-only entries at distinct sorted positions
        let mut ledger = SamfLedger::new();
        let mut positions: Vec<usize> = (0..8).map(|_| rng.random_range(0..m)).collect();
        positions.sort_unstable();
        positions.dedup();
        for &p in &positions {
            let mut t = rand_perm_tail(n, &mut rng);
            t.neg = vec![0u8; n]; // perm-only for materialize_perm
            ledger.insert(p, t);
        }
        // eager materialization
        let mut eager = gates.clone();
        ledger.clone().materialize_perm(&mut eager);
        // per-gate: relabel by compose of all entries with pos <= x  (= forward_inbound(0,x))
        for x in 0..m {
            let inbound = ledger.forward_inbound(0, x, n);
            assert_eq!(eager[x], inbound.relabel(gates[x]), "x={x}");
        }
    }
}

#[test]
fn ledger_ignore_left_compose_crossed() {
    // T_start . forward_inbound(start, x) == forward_inbound(0, x): the uniform left factor
    // composes with the crossed factor to give the full inbound (the relabeling-invariance
    // that lets a pass ignore SAMFs left of its anchor).
    let n = 16;
    let mut rng = StdRng::seed_from_u64(0xe5);
    for _ in 0..30 {
        let m = 100;
        let mut ledger = SamfLedger::new();
        let mut positions: Vec<usize> = (0..10).map(|_| rng.random_range(0..m)).collect();
        positions.sort_unstable();
        positions.dedup();
        for &p in &positions {
            ledger.insert(p, rand_perm_tail(n, &mut rng));
        }
        for &start in &[0usize, 20, 50, 80] {
            let t_left = if start == 0 {
                SamfTail::identity(n)
            } else {
                ledger.forward_inbound(0, start - 1, n)
            };
            for x in start..m {
                let combined = t_left.then(&ledger.forward_inbound(start, x, n));
                assert_eq!(
                    combined,
                    ledger.forward_inbound(0, x, n),
                    "start={start} x={x}"
                );
            }
        }
    }
}

#[test]
fn ledger_insert_sorted_and_merges() {
    let n = 8;
    let mut rng = StdRng::seed_from_u64(0xf6);
    let mut ledger = SamfLedger::new();
    let a = rand_perm_tail(n, &mut rng);
    let b = rand_perm_tail(n, &mut rng);
    ledger.insert(5, a.clone());
    ledger.insert(2, b.clone());
    ledger.insert(5, b.clone()); // merge into existing pos-5 entry: a then b
    assert_eq!(ledger.len(), 2);
    // pos 2 = b, pos 5 = a.then(b)
    assert_eq!(ledger.forward_inbound(2, 2, n), b);
    assert_eq!(ledger.forward_inbound(5, 5, n), a.then(&b));
}

#[test]
fn perm_to_swaps_reproduces_perm() {
    let n = 40;
    let mut rng = StdRng::seed_from_u64(0x9a7);
    for _ in 0..100 {
        let tail = rand_perm_tail(n, &mut rng);
        let swaps = tail.perm_to_swaps();
        let t = Transpositions {
            transpositions: swaps,
        };
        for w in 0..n as u16 {
            assert_eq!(t.evaluate(w), tail.perm[w as usize]);
        }
    }
}

#[test]
fn samftail_invert_cancels() {
    let n = 40;
    let mut rng = StdRng::seed_from_u64(0xc3);
    for _ in 0..50 {
        let (t, neg) = rand_transpositions(n, 25, &mut rng);
        let tail = SamfTail::from_transpositions(&t, &neg, n);
        let id = SamfTail::identity(n);
        assert_eq!(tail.then(&tail.invert()), id);
        assert_eq!(tail.invert().then(&tail), id);
    }
}

#[test]
fn anchor_uniform_over_ties() {
    let g: Vec<[u16; 3]> = (0..1000).map(|i| [i as u16, 0, 1]).collect();
    let mut t: Vec<Tag> = vec![Tag(9); 1000];
    t[10] = Tag(1);
    t[500] = Tag(1);
    t[900] = Tag(1);
    let sc = SegCircuit::from_flat(&g, &t);
    let mut rng = StdRng::seed_from_u64(7);
    let mut counts = std::collections::HashMap::new();
    for _ in 0..3000 {
        let i = sc.random_min_gen_index(&mut rng).unwrap();
        assert!(i == 10 || i == 500 || i == 900);
        *counts.entry(i).or_insert(0) += 1;
    }
    // each tie should be hit a healthy fraction of the time
    for k in [10, 500, 900] {
        assert!(counts[&k] > 500, "tie {k} under-sampled: {:?}", counts);
    }
}
