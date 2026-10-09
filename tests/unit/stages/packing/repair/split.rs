use super::*;

fn cx(t: u16, c: u16) -> XGate {
    XGate::cnot(t, c)
}

/// `n` gates, each changing its own wire to `w ^ w0`: a window of k gates
/// has k changed wires of 2 monomials, so it fits a budget iff 2k <= budget.
fn ladder(n: usize) -> Vec<XGate> {
    (0..n).map(|i| cx(1 + i as u16, 0)).collect()
}

#[test]
fn split_candidates_and_smaller_tie() {
    assert_eq!(split_candidates(0, 10), (1..10).collect::<Vec<_>>());
    assert_eq!(split_candidates(5, 8), vec![6, 7]);
    let c = split_candidates(100, 1000);
    assert_eq!(c.len(), 32);
    assert_eq!(c[0], 100 + 900 / 33);
    assert_eq!(*c.last().unwrap(), 100 + 900 * 32 / 33);
    // nearest the floor middle; the smaller on a tie
    assert_eq!(pick_candidate(&[4, 6], 0, 10), 4);
    assert_eq!(pick_candidate(&[4, 6], 0, 11), 4); // mid 5
    assert_eq!(pick_candidate(&[3, 6], 0, 10), 6);
}

#[test]
fn split_never_splits_two_gate_windows() {
    let g = ladder(6);
    let mut fits = FitCache::new(&g, 8, 1, CapSemantics::CPython39);
    let mut calls = 0;
    // windows [0,2) [2,3) [3,5) [5,6): none has more than 2 gates
    let sp = split_pass(6, &[2, 3, 5], &mut fits, &mut |_| {
        calls += 1;
        FxHashSet::default()
    });
    assert!(sp.new.is_empty());
    assert_eq!(calls, 0);
    assert_eq!(sp.bounds_out, vec![2, 3, 5]);
}

#[test]
fn split_recursion_and_clean_candidates() {
    let g = ladder(12);
    let mut fits = FitCache::new(&g, 16, 6, CapSemantics::CPython39);
    let mut asked: Vec<Vec<usize>> = Vec::new();
    let sp = split_pass(12, &[], &mut fits, &mut |c| {
        asked.push(c.to_vec());
        // cut 6 leaks: the split must avoid it
        [6].into_iter().collect()
    });
    // budget 6 = 3 gates. [0,12): mid 6 leaks -> 5 (tie 5/7 -> smaller);
    // [0,5) -> 2; [0,2) and [2,5) are done; [5,12): mid 8 -> 8; [5,8) fits;
    // [8,12): mid 10 -> 10
    assert_eq!(sp.new, vec![5, 2, 8, 10]);
    assert!(sp.forced.is_empty());
    assert_eq!(sp.bounds_out, vec![2, 5, 8, 10]);
    assert_eq!(asked[0], (1..12).collect::<Vec<_>>());
}

#[test]
fn split_forced_when_no_clean_candidate() {
    let g = ladder(8);
    let mut fits = FitCache::new(&g, 16, 8, CapSemantics::CPython39);
    let sp = split_pass(8, &[8, 9], &mut fits, &mut |c| c.iter().copied().collect());
    assert_eq!(sp.new, vec![4]);
    assert_eq!(sp.forced, vec![(0, 8)]);
    // out-of-range input bounds are kept, as the oracle's set union keeps them
    assert_eq!(sp.bounds_in, 2);
    assert_eq!(sp.bounds_out, vec![4, 8, 9]);
}
