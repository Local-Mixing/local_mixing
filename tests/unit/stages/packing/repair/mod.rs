use super::*;

fn cx(t: u16, c: u16) -> XGate {
    XGate::cnot(t, c)
}

fn ccx(t: u16, a: u16, b: u16) -> XGate {
    XGate::conj(t, [(a, true), (b, true)]).unwrap()
}

fn params(fix: usize, fallback: usize, budget: usize) -> RepairParams {
    RepairParams {
        fix_iterations: fix,
        fallback_iterations: fallback,
        split_budget: budget,
        ..Default::default()
    }
}

fn run(
    packed: &[XGate],
    bounds: &[i64],
    c: &[XGate],
    n: usize,
    p: &RepairParams,
) -> (Repaired, String) {
    let cv = CView::new(c, n, &p.cview);
    let mut log = Vec::new();
    let r = repair(packed, 8, bounds, &cv, p, "P", &mut log).unwrap();
    (r, String::from_utf8(log).unwrap())
}

/// n = 2, C = CNOT(1<-0), CNOT(0<-1), CNOT(1<-0): gate 0's value x0^x1 is
/// not an output, and every value is affine in the inputs.
fn linear_c() -> Vec<XGate> {
    vec![cx(1, 0), cx(0, 1), cx(1, 0)]
}

#[test]
fn fix_exhausted_without_rescan_then_merge() {
    // P = C with a boundary after gate 0: all three values are visible there
    // (v = 3 > n = 2); the leak of gate 0 cannot be fixed (gate 1 reads its
    // wire, gate 2 reads gate 1's), so both fix passes move nothing.
    let c = linear_c();
    let (r, log) = run(&c, &[1], &c, 2, &params(2, 4, 65536));
    let rp = &r.report;
    assert_eq!(rp.fix_trajectory, vec![3, 3]);
    assert_eq!(rp.fix_moved, vec![0, 0]);
    assert!(rp.fix_exhausted);
    // fallback 1: merge drops the only boundary; the single window fits;
    // fallback 2: no interior cut (visible 0), nothing to split: done.
    assert_eq!(rp.fallback_trajectory, vec![3, 0]);
    assert_eq!(rp.merged, vec![1]);
    assert_eq!(rp.splits, vec![0, 0]);
    assert_eq!(rp.single_window_scans, 1);
    assert!(r.bounds.is_empty());
    assert_eq!(rp.final_visible, 0);
    assert_eq!(rp.status(false, false).unwrap(), "ok");
    // exactly two scans in the fix loop: no rescan after the last fix
    assert_eq!(rp.scans, 4);
    let want = "cancelled 0 pairs (0 gates)\n\
                3 -> 3 gates; 1 -> 1 boundaries\n\
                P: 3 vs 3 gates; EQUIVALENT (1024 random states)\n\
                fix iter 1: C-view visible 3 (floor 2)\n\
                3 -> 3 gates; 1 -> 1 boundaries\n\
                fix iter 2: C-view visible 3 (floor 2)\n\
                3 -> 3 gates; 1 -> 1 boundaries\n\
                P: 3 vs 3 gates; EQUIVALENT (1024 random states)\n\
                fallback iter 1: C-view visible 3\n\
                merge: removed 1 of 1 boundaries\n  \
                P: 0 splits added (0 -> 0 boundaries)\n\
                fallback iter 2: C-view visible 0\n  \
                (single window: no interior cut, scan skipped)\n  \
                P: 0 splits added (0 -> 0 boundaries)\n";
    assert_eq!(log, want);
    // the last in-loop scan is the skipped one (zero cuts)
    assert_eq!(r.last_scan.cuts, 0);
}

#[test]
fn single_window_skips_scans() {
    let c = linear_c();
    let (r, _) = run(&c, &[], &c, 2, &params(4, 4, 65536));
    let rp = &r.report;
    assert_eq!(rp.fix_trajectory, vec![0]);
    assert_eq!(rp.fallback_trajectory, vec![0]);
    assert_eq!(rp.single_window_scans, 2);
    assert!(r.bounds.is_empty());
    assert_eq!(r.gates, c);
    assert_eq!(rp.status(false, false).unwrap(), "ok");
}

#[test]
fn oracle_loop_break_rules() {
    // n = 3, C = three Toffolis on distinct wires: every value is an output
    // and nonlinear. P = a ladder of copies of x0 (wires 3..10), then C.
    let c = vec![ccx(2, 0, 1), ccx(0, 1, 2), ccx(1, 0, 2)];
    let mut p: Vec<XGate> = (0..8).map(|i| cx(3 + i, 0)).collect();
    p.extend(c.iter().cloned());
    // budget 6: a window of more than 3 ladder gates does not fit
    let (r, log) = run(&p, &[], &c, 3, &params(4, 4, 6));
    let rp = &r.report;
    // fix: visible 0 <= 3 at once; fallback 1 splits (no break although
    // visible <= n); fallback 2 adds nothing: break
    assert_eq!(rp.fix_trajectory, vec![0]);
    assert_eq!(rp.fallback_trajectory.len(), 2);
    assert!(rp.splits[0] > 0);
    assert_eq!(rp.splits[1], 0);
    assert!(rp.merged.is_empty());
    assert!(log.contains("fallback iter 2"));
    assert!(!log.contains("fallback iter 3"));
    assert!(rp.final_nonoutput_visible.is_empty());
    // the final windows all fit, except possibly 1-2 gate windows
    for &(lo, hi) in &rp.final_over_budget {
        assert!(hi - lo <= 2);
    }
}

#[test]
fn cancel_then_certify_and_floor_status() {
    // a cancelling pair around a commuting gate, on top of C
    let c = linear_c();
    let mut p = vec![cx(5, 4), cx(6, 7), cx(5, 4)];
    p.extend(c.iter().cloned());
    let (r, log) = run(&p, &[3, 3, 9], &c, 2, &params(0, 0, 65536));
    assert_eq!(r.report.cancel_passes, vec![1]);
    assert_eq!(r.cancelled, vec![vec![(0, 2)]]);
    assert!(log.starts_with(
        "  cancel pass: 1 pairs\ncancelled 1 pairs (2 gates)\n6 -> 4 gates; 3 -> 1 boundaries\n"
    ));
    assert_eq!(r.origins, vec![1, 3, 4, 5]);
    assert_eq!(r.bounds, vec![1]);
    // no loop ran: the final scan at cut 1 (after the commuting gate) sees
    // only the inputs, i.e. C's output value x0 (gate 2) and x1 (gate 1)
    let rp = &r.report;
    assert!(rp.fix_trajectory.is_empty() && rp.fallback_trajectory.is_empty());
    assert_eq!(rp.final_visible, 3);
    // gate 0's value x0^x1 = w0 ^ w1 is visible and not an output
    assert_eq!(rp.final_nonoutput_visible, vec![0]);
    assert!(matches!(
        rp.status(false, false),
        Err(Stage4Error::Floor { .. })
    ));
    assert_eq!(rp.status(true, false).unwrap(), "floor_miss");
}
