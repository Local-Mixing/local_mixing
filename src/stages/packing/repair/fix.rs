//! The fix pass of `order_repair.py` (canonical version: the breaker is
//! located BEFORE the completer moves).
//!
//! For each interior C leak (a witness row: predictor support S at a cut),
//! the relation lives while all of S's current segments coexist: from the
//! last write to S before the cut (the COMPLETER) to the first write to S
//! after it (the BREAKER). The completer slides forward and the breaker
//! backward as far as commutation allows.
use super::seq::{MARKER, slide};
use crate::circuit::xgate::XGate;
use rustc_hash::FxHashSet;

/// One witness row (`wit.tsv`): C gate, cut position (a GATE COUNT in the
/// scanned circuit), predictor support (P wires).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixRow {
    pub c_gate: usize,
    pub cut: usize,
    pub support: Vec<u32>,
}

/// What one fix pass did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FixReport {
    /// Rows processed (non-output, nonempty support).
    pub rows: usize,
    /// Slides that changed a gate's position (`moved` in the oracle).
    pub moved: usize,
}

/// `run_fix(seq, wit, exposure)`. `rows` are the witness rows in file order
/// (ascending C gate); rows of output gates and rows with an empty support
/// are skipped. The cut is applied to the CURRENT sequence (the first `cut`
/// gates of it), which earlier rows of the same pass may have reordered:
/// the oracle's stale gate-count cut.
pub fn run_fix(
    seq: &mut Vec<u32>,
    arena: &[XGate],
    rows: &[FixRow],
    outs: &FxHashSet<usize>,
) -> FixReport {
    let mut rep = FixReport::default();
    for r in rows {
        if outs.contains(&r.c_gate) || r.support.is_empty() {
            continue;
        }
        rep.rows += 1;
        let in_s = |x: u32| {
            let t = arena[x as usize].target as u32;
            r.support.contains(&t)
        };
        // Completer: the last gate among the first `cut` gates writing S.
        // Breaker: the first gate from gate `cut` on writing S.
        let mut comp_i: Option<usize> = None;
        let mut brk_i: Option<usize> = None;
        let mut gcount = 0usize;
        for (i, &x) in seq.iter().enumerate() {
            if x == MARKER {
                continue;
            }
            if gcount < r.cut {
                if in_s(x) {
                    comp_i = Some(i);
                }
            } else if in_s(x) {
                brk_i = Some(i);
                break;
            }
            gcount += 1;
        }
        if let Some(ci) = comp_i {
            let new = slide(seq, arena, ci, true);
            rep.moved += (new != ci) as usize;
            // The completer moved from ci to new: for every index past `new`
            // the removal and the insertion cancel, so brk_i still points at
            // the breaker unless the completer passed it (same-target writes
            // commute), which already breaks the relation.
            if let Some(bi) = brk_i
                && new >= bi
            {
                brk_i = None;
            }
        }
        if let Some(bi) = brk_i {
            let new = slide(seq, arena, bi, false);
            rep.moved += (new != bi) as usize;
        }
    }
    rep
}

#[cfg(test)]
#[path = "../../../../tests/unit/stages/packing/repair/fix.rs"]
mod tests;
