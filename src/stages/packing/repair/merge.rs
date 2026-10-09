//! `merge_bounds2.py`: delete every boundary at which an interior (non-output)
//! C value is exactly affine, from an all-cuts scan of the current lattice.
use crate::stages::packing::cview::ExposureReport;
use rustc_hash::FxHashSet;

/// Returns the kept bounds (input order kept) and the oracle's "removed"
/// count, `len(set(bounds) & drop)`.
pub fn merge_bounds(
    bounds: &[usize],
    scan: &ExposureReport,
    outs: &FxHashSet<usize>,
) -> (Vec<usize>, usize) {
    assert!(scan.options.collect_allcuts, "merge needs an all-cuts scan");
    let drop: FxHashSet<usize> = scan
        .allcuts
        .iter()
        .filter(|(r, _, _)| !outs.contains(&scan.rows[*r as usize].gate))
        .map(|&(_, cut, _)| cut)
        .collect();
    let kept: Vec<usize> = bounds
        .iter()
        .copied()
        .filter(|b| !drop.contains(b))
        .collect();
    let distinct: FxHashSet<usize> = bounds.iter().copied().collect();
    let removed = distinct.iter().filter(|b| drop.contains(b)).count();
    (kept, removed)
}

#[cfg(test)]
#[path = "../../../../tests/unit/stages/packing/repair/merge.rs"]
mod tests;
