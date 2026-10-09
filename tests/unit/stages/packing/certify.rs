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
