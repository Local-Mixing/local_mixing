use super::*;

fn g(t: u16, ctrls: &[u16]) -> XGate {
    XGate::conj(t, ctrls.iter().map(|&c| (c, true))).unwrap()
}

#[test]
fn reorder_cert_accepts_commuting_swap() {
    let c = [g(1, &[0]), g(3, &[2]), g(4, &[0, 2])];
    certify_reorder(&c, &[0, 1, 2]).unwrap();
    certify_reorder(&c, &[1, 0, 2]).unwrap();
    certify_reorder(&c, &[2, 1, 0]).unwrap(); // reads of 0 and 2 commute
}

#[test]
fn same_target_writes_commute() {
    // both write wire 2, neither reads the other's target
    let c = [g(2, &[0]), g(2, &[1]), XGate::x_gate(2)];
    certify_reorder(&c, &[2, 1, 0]).unwrap();
}

#[test]
fn reorder_cert_rejects_conflicting_swap() {
    // g0 writes 1, g1 reads 1
    let c = [g(1, &[0]), g(2, &[1])];
    assert!(matches!(
        certify_reorder(&c, &[1, 0]),
        Err(Stage4Error::Verify(_))
    ));
    // differs on only a 2^-5 fraction of inputs (5-control Toffoli reading the
    // swapped write): still rejected, the certificate is structural
    let c = [g(1, &[0]), g(7, &[1, 2, 3, 4, 5])];
    assert!(certify_reorder(&c, &[1, 0]).is_err());
    // a write moved across a read of its target, off the zero slice only
    let c = [g(5, &[0]), g(6, &[5]), g(5, &[0])];
    assert!(certify_reorder(&c, &[0, 2, 1]).is_err());
    // runs W,R,W on one wire: the middle read cannot leave its run
    certify_reorder(&c, &[0, 1, 2]).unwrap();
}

#[test]
fn reorder_cert_rejects_non_permutations() {
    let c = [g(1, &[0]), g(3, &[2])];
    assert!(certify_reorder(&c, &[0]).is_err());
    assert!(certify_reorder(&c, &[0, 0]).is_err());
    assert!(certify_reorder(&c, &[0, 2]).is_err());
    certify_reorder(&[], &[]).unwrap();
}

#[test]
fn cancel_cert_rejects_blocked_pair() {
    // A B A with B writing A's control: A B A != B
    let a = g(1, &[0]);
    let c = [a.clone(), g(0, &[2]), a.clone()];
    assert!(matches!(
        certify_cancel(&c, &[vec![(0, 2)]]),
        Err(Stage4Error::Verify(_))
    ));
    // ... and with B reading A's target
    let c = [a.clone(), g(3, &[1]), a.clone()];
    assert!(certify_cancel(&c, &[vec![(0, 2)]]).is_err());
    // a commuting middle gate is fine
    let c = [a.clone(), g(3, &[0]), a.clone()];
    let live = certify_cancel(&c, &[vec![(0, 2)]]).unwrap();
    assert_eq!(live, vec![false, true, false]);
    // a blocker cancelled in an earlier pass no longer blocks
    let b = g(0, &[2]);
    let c = [a.clone(), b.clone(), b, a];
    certify_cancel(&c, &[vec![(1, 2)], vec![(0, 3)]]).unwrap();
    assert!(certify_cancel(&c, &[vec![(0, 3)]]).is_err());
}

#[test]
fn cancel_cert_rejects_nonidentical_pair() {
    let c = [g(1, &[0]), g(1, &[2])];
    assert!(certify_cancel(&c, &[vec![(0, 1)]]).is_err());
    // polarity and complement are part of the identity
    let mut x = g(1, &[0]);
    x.comp = true;
    let c = [g(1, &[0]), x];
    assert!(certify_cancel(&c, &[vec![(0, 1)]]).is_err());
    // reversed, repeated or out-of-range pairs
    let c = [g(1, &[0]), g(1, &[0])];
    assert!(certify_cancel(&c, &[vec![(1, 0)]]).is_err());
    assert!(certify_cancel(&c, &[vec![(0, 1)], vec![(0, 1)]]).is_err());
    assert!(certify_cancel(&c, &[vec![(0, 2)]]).is_err());
}

#[test]
fn repair_cert_combines_cancel_and_reorder() {
    // 0,2 cancel; 1 and 3 commute (reads of 0 and 2 only)
    let a = g(5, &[4]);
    let c = [a.clone(), g(1, &[0]), a, g(3, &[2])];
    certify_repair(&c, &[vec![(0, 2)]], &[3, 1]).unwrap();
    certify_repair(&c, &[vec![(0, 2)]], &[1, 3]).unwrap();
    // a cancelled gate may not reappear; a survivor may not vanish
    assert!(certify_repair(&c, &[vec![(0, 2)]], &[0, 1, 3]).is_err());
    assert!(certify_repair(&c, &[vec![(0, 2)]], &[1]).is_err());
    // the subset form rejects a conflicting swap
    let d = [g(1, &[0]), g(2, &[1])];
    certify_reorder_subset(&d, &[0, 1], &[0, 1]).unwrap();
    assert!(certify_reorder_subset(&d, &[0, 1], &[1, 0]).is_err());
}
