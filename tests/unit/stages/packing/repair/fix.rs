use super::*;

const M: u32 = MARKER;

fn cx(t: u16, c: u16) -> XGate {
    XGate::cnot(t, c)
}

fn row(c_gate: usize, cut: usize, support: &[u32]) -> FixRow {
    FixRow {
        c_gate,
        cut,
        support: support.to_vec(),
    }
}

#[test]
fn fix_breaker_located_before_completer_moves() {
    // S = {2, 3}, cut after one gate. 0: completer w2^=w0; 1: w6^=w5
    // (commutes); 2: breaker w3^=w2 (reads 2, so it blocks the completer).
    let arena = [cx(2, 0), cx(6, 5), cx(3, 2)];
    let mut seq = vec![0, M, 1, 2];
    let r = run_fix(
        &mut seq,
        &arena,
        &[row(7, 1, &[2, 3])],
        &FxHashSet::default(),
    );
    // the completer lands just before the breaker; the breaker (located
    // before the move) cannot pass it. Re-locating the breaker after the move
    // would find the completer itself (now past the cut) and slide it back.
    assert_eq!(seq, vec![M, 1, 0, 2]);
    assert_eq!(r, FixReport { rows: 1, moved: 1 });
}

#[test]
fn fix_drops_breaker_when_completer_passes_it() {
    // completer and breaker both write wire 2: they commute, so the
    // completer slides past the breaker and the breaker is not moved.
    let arena = [cx(2, 0), cx(2, 1), cx(5, 4)];
    let mut seq = vec![0, M, 1, 2];
    let r = run_fix(&mut seq, &arena, &[row(7, 1, &[2])], &FxHashSet::default());
    assert_eq!(seq, vec![M, 1, 2, 0]);
    assert_eq!(r.moved, 1);
}

#[test]
fn fix_uses_stale_gate_count_cut() {
    // 0: w2^=w0, 1: w4^=w1, 2: w5^=w2 (blocks 0)
    let arena = [cx(2, 0), cx(4, 1), cx(5, 2)];
    let mut seq = vec![0, 1, 2];
    let rows = [row(10, 1, &[2]), row(11, 1, &[4])];
    let r = run_fix(&mut seq, &arena, &rows, &FxHashSet::default());
    // row 1 moves gate 0 past gate 1, so for row 2 "the first gate" is gate 1:
    // it is row 2's completer and slides to the end.
    assert_eq!(seq, vec![0, 2, 1]);
    assert_eq!(r, FixReport { rows: 2, moved: 2 });
}

#[test]
fn fix_skips_outputs_and_empty_supports() {
    let arena = [cx(2, 0), cx(4, 1), cx(5, 2)];
    let mut seq = vec![0, 1, 2];
    let outs: FxHashSet<usize> = [10].into_iter().collect();
    let rows = [row(10, 1, &[2]), row(11, 1, &[])];
    let r = run_fix(&mut seq, &arena, &rows, &outs);
    assert_eq!(seq, vec![0, 1, 2]);
    assert_eq!(r, FixReport::default());
}
