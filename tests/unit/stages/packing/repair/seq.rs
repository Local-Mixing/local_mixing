use super::*;

fn cx(t: u16, c: u16) -> XGate {
    XGate::cnot(t, c)
}

const M: u32 = MARKER;

#[test]
fn to_from_seq_merges_and_drops_markers() {
    // duplicates, unsorted, 0, m and beyond are dropped by to_seq
    let seq = to_seq(&[10, 11, 12, 13], &[3, 1, 1, 0, 4, 7, -2]);
    assert_eq!(seq, vec![10, M, 11, 12, M, 13]);
    // from_seq: leading markers dropped, adjacent merged, trailing dropped
    let (g, b) = from_seq(&[M, M, 10, M, M, 11, 12, M, 13, M, M]);
    assert_eq!(g, vec![10, 11, 12, 13]);
    assert_eq!(b, vec![1, 3]);
    let (g, b) = from_seq(&[M]);
    assert!(g.is_empty() && b.is_empty());
    // round trip normalizes
    let s = [10, M, M, 11, M];
    let (g, b) = from_seq(&s);
    let bi: Vec<i64> = b.iter().map(|&x| x as i64).collect();
    assert_eq!(to_seq(&g, &bi), vec![10, M, 11]);
}

#[test]
fn commute_is_read_write_only() {
    // same target: commute (XOR), even with different controls
    assert!(commute(&cx(2, 0), &cx(2, 1)));
    // one reads the other's target
    assert!(!commute(&cx(2, 0), &cx(3, 2)));
    assert!(!commute(&cx(3, 2), &cx(2, 0)));
    // both read the same wire
    assert!(commute(&cx(2, 0), &cx(3, 0)));
    // polarity-separated conjunctions do NOT commute here (unlike `collides`)
    let a = XGate::conj(2, [(0, true), (1, true)]).unwrap();
    let b = XGate::conj(1, [(0, false), (3, true)]).unwrap();
    assert!(!XGate::collides(&a, &b));
    assert!(!commute(&a, &b));
}

#[test]
fn slide_no_move_when_only_markers_before_blocker() {
    // arena: 0: w2^=w0, 1: w3^=w2 (reads 2: blocks gate 0)
    let arena = [cx(2, 0), cx(3, 2)];
    let mut seq = vec![0, M, M, 1];
    assert_eq!(slide(&mut seq, &arena, 0, true), 0);
    assert_eq!(seq, vec![0, M, M, 1]);
    let mut seq = vec![0, M, 1];
    assert_eq!(slide(&mut seq, &arena, 2, false), 2);
    assert_eq!(seq, vec![0, M, 1]);
}

#[test]
fn slide_lands_before_markers_forward_after_backward() {
    // 0: w2^=w0, 1: w4^=w1 (commutes with 0), 2: w3^=w2 (blocks 0)
    let arena = [cx(2, 0), cx(4, 1), cx(3, 2)];
    // forward: past gate 1 and the markers, just before the blocker
    let mut seq = vec![0, M, 1, M, M, 2];
    assert_eq!(slide(&mut seq, &arena, 0, true), 4);
    assert_eq!(seq, vec![M, 1, M, M, 0, 2]);
    // backward: lands just after the blocker, before the markers
    let mut seq = vec![0, M, M, 1, M, 2];
    assert_eq!(slide(&mut seq, &arena, 5, false), 1);
    assert_eq!(seq, vec![0, 2, M, M, 1, M]);
    // no blocker: to the very end / the very front, past every marker
    let mut seq = vec![1, M, 0, M];
    assert_eq!(slide(&mut seq, &arena, 0, true), 3);
    assert_eq!(seq, vec![M, 0, M, 1]);
    let mut seq = vec![M, 0, M, 1];
    assert_eq!(slide(&mut seq, &arena, 3, false), 0);
    assert_eq!(seq, vec![1, M, 0, M]);
}
