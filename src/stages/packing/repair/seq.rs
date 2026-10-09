//! The gate/marker sequence of `order_repair.py`: gates interleaved with
//! window-boundary markers (Python `None`).
//!
//! A sequence item is a `u32`: the gate's ORIGIN (its index in the packed
//! circuit, which is the arena every gate is looked up in) or [`MARKER`].
//! Nothing in the oracle algorithm reads the origin; it lets the exact
//! certificates follow every gate from the packed circuit to the repaired one.
use crate::circuit::xgate::XGate;

/// A window-boundary marker.
pub const MARKER: u32 = u32::MAX;

/// `to_seq(gates, bounds)`: a marker before gate `i` for every distinct bound
/// `0 < i < len(gates)` (sorted, deduplicated; others are ignored).
pub fn to_seq(origins: &[u32], bounds: &[i64]) -> Vec<u32> {
    let m = origins.len();
    let mut bset: Vec<usize> = bounds
        .iter()
        .filter(|&&b| b > 0 && (b as u64) < m as u64)
        .map(|&b| b as usize)
        .collect();
    bset.sort_unstable();
    bset.dedup();
    let mut seq = Vec::with_capacity(m + bset.len());
    let mut k = 0;
    for (i, &o) in origins.iter().enumerate() {
        while k < bset.len() && bset[k] == i {
            seq.push(MARKER);
            k += 1;
        }
        seq.push(o);
    }
    seq
}

/// `from_seq(seq)`: the gates, and a bound at every marker that has a gate
/// before it, adjacent markers merged and a trailing marker dropped.
pub fn from_seq(seq: &[u32]) -> (Vec<u32>, Vec<usize>) {
    let mut gates = Vec::with_capacity(seq.len());
    let mut bounds: Vec<usize> = Vec::new();
    for &x in seq {
        if x == MARKER {
            if !gates.is_empty() && bounds.last() != Some(&gates.len()) {
                bounds.push(gates.len());
            }
        } else {
            gates.push(x);
        }
    }
    let m = gates.len();
    bounds.retain(|&b| b < m);
    (gates, bounds)
}

/// Gates commute iff neither's target is a control of the other. Two writes
/// of one target commute (XOR); this is NOT `XGate::collides`, which exempts
/// polarity-separated conjunctions.
#[inline]
pub fn commute(a: &XGate, b: &XGate) -> bool {
    !a.reads(b.target) && !b.reads(a.target)
}

/// `slide(seq, pos, direction)`: move the gate at `pos` as far as commutation
/// allows (past markers too) and return its new index.
///
/// The scan stops at the first non-commuting gate (the blocker) or the end.
/// If no GATE was passed, nothing moves, even when markers lie before the
/// blocker. Otherwise the gate lands just before the blocker (forward) or just
/// after it (backward), i.e. past every marker between the last passed gate
/// and the blocker; with no blocker it goes to the very end (forward) or the
/// very front (backward).
pub fn slide(seq: &mut Vec<u32>, arena: &[XGate], pos: usize, forward: bool) -> usize {
    let item = seq[pos];
    debug_assert_ne!(item, MARKER, "slide of a marker");
    let g = &arena[item as usize];
    let len = seq.len();
    let mut dest = pos;
    // `j` is the blocker, or None when the scan ran off the end.
    let mut blocker: Option<usize> = None;
    if forward {
        for (j, &x) in seq.iter().enumerate().skip(pos + 1) {
            if x != MARKER {
                if !commute(g, &arena[x as usize]) {
                    blocker = Some(j);
                    break;
                }
                dest = j;
            }
        }
    } else {
        for j in (0..pos).rev() {
            let x = seq[j];
            if x != MARKER {
                if !commute(g, &arena[x as usize]) {
                    blocker = Some(j);
                    break;
                }
                dest = j;
            }
        }
    }
    if dest == pos {
        return pos;
    }
    let land = match (blocker, forward) {
        (Some(j), true) => j - 1,
        (Some(j), false) => j + 1,
        (None, true) => len - 1,
        (None, false) => 0,
    };
    let it = seq.remove(pos);
    seq.insert(land, it);
    land
}

#[cfg(test)]
#[path = "../../../../tests/unit/stages/packing/repair/seq.rs"]
mod tests;
