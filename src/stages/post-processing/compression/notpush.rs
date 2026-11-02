//! Exact NOT-pushing: sweep the circuit once, absorbing every interior bare
//! NOT (and dropping every never-firing gate) into a per-wire parity that is
//! folded into the control polarities of the gates downstream, then emit the
//! residual NOTs at the tail. Function-preserving by construction and
//! verified by a sampled equivalence check in the caller.
//!
//! Mechanism (from `notpush.py`, rust2 shortening 2026-10-31: the C-blind
//! toolchain leaves tens of thousands of bare NOTs and complemented gates;
//! collapsing them was -2.6% and reopened fcompress/ANF). Model: maintain
//! `p[w]`, the parity of NOTs pushed onto wire `w` so far, so the swept
//! state `y` relates to the original `x` by `y[w] = x[w] ^ p[w]`.
//!
//!   * a gate with no controls fires iff `!comp` (`XGate::x_gate` is the
//!     always-firing NOT, `comp = true` the never-firing no-op): a NOT flips
//!     `p[target]` and is dropped, a no-op is dropped;
//!   * any other gate keeps its target and `comp`, and each control literal
//!     `(w, pol)` becomes `(w, pol ^ p[w])` -- `lit(x) == lit'(y)`, so the
//!     fire value is unchanged and `y[target] ^= fire` matches
//!     `x[target] ^= fire` with `p[target]` untouched;
//!   * at the tail, one NOT per wire with `p[w]` set restores `x` from `y`.
//!
//! The pushed circuit computes exactly the same permutation. Interior NOTs
//! (one per wire at most survive, at the tail) and no-ops are gone; the
//! normalized polarities let the gather/reduce reducer and the ANF
//! synthesizer find more merges on the next pass.
use crate::circuit::xgate::XGate;

/// What a NOT-push removed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NotPushReport {
    /// Interior bare NOTs absorbed into the parity.
    pub nots_absorbed: u64,
    /// Never-firing gates (empty controls, `comp = true`) dropped.
    pub noops_dropped: u64,
    /// Control literals whose polarity flipped.
    pub lits_flipped: u64,
    /// Residual NOTs emitted at the tail (at most one per wire).
    pub trailing_nots: usize,
}

/// Push every interior NOT to the tail and drop no-ops, exactly preserving
/// the circuit's function on `width` wires. Returns the new gate list and a
/// report; the gate count drops by `nots_absorbed + noops_dropped -
/// trailing_nots`.
pub fn push_nots(gates: &[XGate], width: usize) -> (Vec<XGate>, NotPushReport) {
    let mut parity = vec![false; width];
    let mut out: Vec<XGate> = Vec::with_capacity(gates.len());
    let mut rep = NotPushReport::default();
    for g in gates {
        let t = g.target as usize;
        if g.ctrls.is_empty() {
            // fires iff !comp: `comp == false` is the NOT, `true` the no-op.
            if g.comp {
                rep.noops_dropped += 1;
            } else {
                parity[t] ^= true;
                rep.nots_absorbed += 1;
            }
            continue;
        }
        let mut ng = g.clone();
        for lit in ng.ctrls.iter_mut() {
            if parity[lit.0 as usize] {
                lit.1 = !lit.1;
                rep.lits_flipped += 1;
            }
        }
        out.push(ng);
    }
    for (w, &p) in parity.iter().enumerate() {
        if p {
            out.push(XGate::x_gate(w as u16));
            rep.trailing_nots += 1;
        }
    }
    (out, rep)
}

#[cfg(test)]
#[path = "../../../../tests/unit/stages/post-processing/compression/notpush_tests.rs"]
mod notpush_tests;
