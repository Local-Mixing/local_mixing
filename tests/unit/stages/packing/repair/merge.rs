use super::*;
use crate::stages::packing::cview::{BornRow, ExposureOptions};

fn report(allcuts: Vec<(u32, usize, u32)>) -> ExposureReport {
    // rows: gate 0 (wire 1), gate 1 (wire 0), gate 2 (wire 1)
    let rows = vec![
        BornRow {
            prefix: 1,
            gate: 0,
            wire: 1,
        },
        BornRow {
            prefix: 2,
            gate: 1,
            wire: 0,
        },
        BornRow {
            prefix: 3,
            gate: 2,
            wire: 1,
        },
    ];
    ExposureReport {
        best: vec![-1; rows.len()],
        witness: vec![None; rows.len()],
        rows,
        allcuts,
        approx: 0,
        options: ExposureOptions::historic(true, true),
    }
}

#[test]
fn merge_keeps_input_order() {
    let outs: FxHashSet<usize> = [1, 2].into_iter().collect();
    // row 0 (gate 0, not an output) is affine at cuts 30 and 10; outputs at 20
    let s = report(vec![(0, 30, 1), (1, 20, 1), (0, 10, 2), (2, 20, 1)]);
    let (kept, removed) = merge_bounds(&[40, 30, 20, 10, 30], &s, &outs);
    assert_eq!(kept, vec![40, 20]);
    // len(set(bounds) & drop): the duplicate 30 counts once
    assert_eq!(removed, 2);
    // nothing to drop
    let (kept, removed) = merge_bounds(&[5, 7], &s, &outs);
    assert_eq!((kept, removed), (vec![5, 7], 0));
}
