use super::*;
use crate::stages::packing::certify::certify_cancel;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn cx(t: u16, c: u16) -> XGate {
    XGate::cnot(t, c)
}

/// A literal transcription of `order_repair.cancel_pass` (lists + bisect).
fn cancel_pass_py(gates: &[XGate]) -> (Vec<bool>, usize) {
    let m = gates.len();
    let mut reads: std::collections::HashMap<u16, Vec<usize>> = Default::default();
    let mut writes: std::collections::HashMap<u16, Vec<usize>> = Default::default();
    let mut same: std::collections::HashMap<&XGate, Vec<usize>> = Default::default();
    for (i, g) in gates.iter().enumerate() {
        writes.entry(g.target).or_default().push(i);
        for &(c, _) in &g.ctrls {
            reads.entry(c).or_default().push(i);
        }
        same.entry(g).or_default().push(i);
    }
    let nxt = |lst: Option<&Vec<usize>>, i: usize| -> usize {
        let Some(lst) = lst else { return m };
        let k = lst.partition_point(|&x| x <= i);
        lst.get(k).copied().unwrap_or(m)
    };
    let mut dead = vec![false; m];
    let mut pairs = 0;
    for (i, g) in gates.iter().enumerate() {
        if dead[i] {
            continue;
        }
        let mut blk = nxt(reads.get(&g.target), i);
        for &(c, _) in &g.ctrls {
            blk = blk.min(nxt(writes.get(&c), i));
        }
        let lst = &same[g];
        let mut k = lst.partition_point(|&x| x <= i);
        while k < lst.len() && lst[k] < blk && dead[lst[k]] {
            k += 1;
        }
        if k < lst.len() && lst[k] < blk {
            dead[i] = true;
            dead[lst[k]] = true;
            pairs += 1;
        }
    }
    (dead, pairs)
}

fn random_gate(rng: &mut StdRng, nw: u16) -> XGate {
    let t = rng.random_range(0..nw);
    let k = rng.random_range(0..3usize);
    let mut lits = Vec::new();
    for _ in 0..k {
        let c = rng.random_range(0..nw);
        if c != t && !lits.iter().any(|&(w, _)| w == c) {
            lits.push((c, rng.random_bool(0.7)));
        }
    }
    let mut g = XGate::conj(t, lits).unwrap();
    g.comp = rng.random_bool(0.2);
    g
}

/// Random circuits rich in identical pairs: a small gate alphabet.
fn random_circuit(rng: &mut StdRng, m: usize, nw: u16, alphabet: usize) -> Vec<XGate> {
    let alpha: Vec<XGate> = (0..alphabet).map(|_| random_gate(rng, nw)).collect();
    (0..m)
        .map(|_| alpha[rng.random_range(0..alphabet)].clone())
        .collect()
}

fn eval_all(gates: &[XGate], nw: usize) -> Vec<u64> {
    (0..1u64 << nw)
        .map(|x| crate::circuit::xgate::eval_u64(gates, x))
        .collect()
}

#[test]
fn cancel_pass_matches_python_transcription() {
    let mut rng = StdRng::seed_from_u64(7);
    for _ in 0..300 {
        let m = rng.random_range(0..60);
        let alpha = rng.random_range(1..8);
        let g = random_circuit(&mut rng, m, 6, alpha);
        let refs: Vec<&XGate> = g.iter().collect();
        let (dead, pairs) = cancel_pass(&refs);
        let (dead_py, pairs_py) = cancel_pass_py(&g);
        assert_eq!(dead, dead_py);
        assert_eq!(pairs.len(), pairs_py);
    }
}

#[test]
fn cancel_dead_gates_still_block() {
    // A B B A with A reading B's target: pass 1 kills only B B (the dead B's
    // still block A), pass 2 kills A A.
    let a = cx(1, 0);
    let b = cx(0, 2);
    let arena = vec![a.clone(), b.clone(), b, a];
    let refs: Vec<&XGate> = arena.iter().collect();
    let (dead, pairs) = cancel_pass(&refs);
    assert_eq!(dead, vec![false, true, true, false]);
    assert_eq!(pairs, vec![(1, 2)]);
    let mut seq: Vec<u32> = vec![0, 1, super::super::seq::MARKER, 2, 3];
    let passes = run_cancel(&mut seq, &arena);
    assert_eq!(passes, vec![vec![(1, 2)], vec![(0, 3)]]);
    assert_eq!(seq, vec![super::super::seq::MARKER]);
    certify_cancel(&arena, &passes).unwrap();
}

#[test]
fn cancel_reaches_fixed_point_and_preserves_function() {
    let mut rng = StdRng::seed_from_u64(11);
    let mut multi_pass = 0;
    for _ in 0..300 {
        let m = rng.random_range(0..80);
        let alpha = rng.random_range(1..6);
        let arena = random_circuit(&mut rng, m, 6, alpha);
        let mut seq: Vec<u32> = (0..m as u32).collect();
        let passes = run_cancel(&mut seq, &arena);
        multi_pass += (passes.len() >= 2) as usize;
        // fixed point: one more pass finds nothing
        let refs: Vec<&XGate> = seq.iter().map(|&o| &arena[o as usize]).collect();
        assert!(cancel_pass(&refs).1.is_empty());
        // function preserved (exhaustively on 6 wires) and certified
        let kept: Vec<XGate> = seq.iter().map(|&o| arena[o as usize].clone()).collect();
        assert_eq!(eval_all(&arena, 6), eval_all(&kept, 6));
        certify_cancel(&arena, &passes).unwrap();
    }
    assert!(
        multi_pass > 10,
        "the generator exercises multi-pass cancels"
    );
}
