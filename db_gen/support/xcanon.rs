//! XGate-level circuit canonicalization (S4 of the wide-gate design).
//!
//! Mirrors `CircuitSeq::canonicalize`'s insertion walk — pull each gate as
//! early as commutation allows when that lowers the lexicographic order —
//! generalized to the mixed alphabet via `XGate::collides` and a total order
//! that reproduces the legacy g57 order exactly on lifted g57 gates.
//!
//! The order must NOT be naive (target, comp, lits)-lex: legacy
//! `Gate::ordered_index` compares the raw `[a, x, y]` triple, i.e. the
//! NEGATIVE control wire before the positive one, while lit-lex compares the
//! smaller WIRE first — e.g. `[0,9,1]` vs `[0,2,8]` order differently under
//! the two schemes. Complemented width-2 gates therefore compare via their
//! reconstructed g57 triple; everything else (wide conjunctions, CNOTs,
//! X gates) uses (target, comp, lits)-lex in a separate, later class.

use crate::circuit::xgate::XGate;
use std::cmp::Ordering;

/// Sort key: g57-reconstructable gates order among themselves exactly like
/// the legacy triple; all other gates form a later class with lit-lex order.
fn order_key(g: &XGate) -> (u8, u16, bool, Vec<(u16, bool)>) {
    if g.comp && g.ctrls.len() == 2 {
        // from_g57: x carries polarity false (negative), y polarity true.
        let (w0, p0) = g.ctrls[0];
        let (w1, p1) = g.ctrls[1];
        if p0 != p1 {
            let (x, y) = if !p0 { (w0, w1) } else { (w1, w0) };
            // Legacy triple order = (target, x, y) lex.
            return (0, g.target, false, vec![(x, false), (y, false)]);
        }
    }
    (1, g.target, g.comp, g.ctrls.iter().copied().collect())
}

/// Total order over XGates; on lifted (non-degenerate) g57 gates it agrees
/// with `Gate::ordered_index`.
pub fn xgate_order(a: &XGate, b: &XGate) -> Ordering {
    order_key(a).cmp(&order_key(b))
}

/// True if two identical gates sit adjacent anywhere: every XGate is its own
/// inverse, so the pair is an exact no-op and the circuit is never a useful
/// DB candidate (mirrors `CircuitSeq::adjacent_id` for the g57 path).
pub fn xgate_adjacent_id(gates: &[XGate]) -> bool {
    gates.windows(2).any(|w| w[0] == w[1])
}

/// Canonicalize gate order in place: an exact port of
/// `CircuitSeq::canonicalize`'s loop with `XGate::collides` for commutation
/// and `xgate_order` for the target ordering. Function-preserving by
/// construction (only commuting swaps are performed).
pub fn xgate_canonicalize(gates: &mut Vec<XGate>) {
    for i in 1..gates.len() {
        let mut to_swap: Option<usize> = None;
        let mut j = i;
        while j > 0 {
            j -= 1;
            if XGate::collides(&gates[i], &gates[j]) {
                break;
            } else if xgate_order(&gates[j], &gates[i]) == Ordering::Greater {
                to_swap = Some(j);
            }
        }
        if let Some(pos) = to_swap {
            let g = gates.remove(i);
            gates.insert(pos, g);
        }
    }
}

#[cfg(test)]
#[path = "../../tests/db_gen/support/xcanon/tests.rs"]
mod tests;
