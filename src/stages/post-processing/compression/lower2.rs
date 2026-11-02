//! Exact 2-control lowering of the wide gates left by stage 5. A gate with
//! more than `max_controls` controls is rewritten, in place, as a chain of
//! 2-control gates by the dirty-ancilla Barenco decomposition (Lemma 7.2:
//! `t ^= c_1 ... c_k` with `k - 2` borrowed helpers and `4(k - 2)`
//! positive Toffolis), with negative controls conjugated by a NOT on each
//! side and the complement folded into a trailing NOT. The helpers are
//! borrowed from other wires and restored, so the pass is exact and the
//! function is unchanged.
//!
//! Why (rust2 shortening 2026-11-01): the ANF synthesizer only emits
//! <= 2-control gates, but a block that does not synthesize keeps stage 5's
//! wide gates (up to 31 % of rust2's delivered circuit, controls up to 12).
//! Counting a k-control gate as one gate understates it (it is `4k - 8`
//! two-control gates), and the wide gates mark exactly the regions stage 6
//! did not touch. Forcing the whole dense block through resynthesis inflates
//! and leaks; lowering the wide gates ALONE is exact and lets the small-block
//! passes work the region afterwards in the deliverable's gate set
//! (<= 2 controls, like stage 2 and the synthesizer).
use crate::circuit::xgate::XGate;

/// What a lowering removed and added.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LowerReport {
    /// Gates with more than `max_controls` controls that were lowered.
    pub wide_in: u64,
    /// The widest control count seen (0 if none).
    pub max_controls_in: usize,
    /// Wide gates kept verbatim for want of enough spare helper wires (never
    /// happens at a stage-5 width, where `width >= 2k - 1`).
    pub wide_kept: u64,
    /// Net gate-count change (output minus input).
    pub gates_delta: i64,
}

/// The positive-control Barenco ladder (`synth::barenco`, ported to XGate):
/// `t ^= c_0 ... c_{k-1}` with dirty helpers `h_0 .. h_{k-3}`, all controls
/// positive, `4(k-2)` Toffolis, helpers restored.
fn barenco_xgate(t: u16, c: &[u16], h: &[u16], out: &mut Vec<XGate>) {
    let k = c.len();
    debug_assert!(k >= 3 && h.len() == k - 2);
    fn tof(t: u16, a: u16, b: u16, out: &mut Vec<XGate>) {
        out.push(XGate::conj(t, [(a, true), (b, true)]).expect("distinct positive controls"));
    }
    fn ladder(m: usize, c: &[u16], h: &[u16], out: &mut Vec<XGate>) {
        if m == 1 {
            tof(h[0], c[0], c[1], out);
        } else {
            tof(h[m - 1], c[m], h[m - 2], out);
            ladder(m - 1, c, h, out);
            tof(h[m - 1], c[m], h[m - 2], out);
        }
    }
    tof(t, c[k - 1], h[k - 3], out);
    ladder(k - 2, c, h, out);
    tof(t, c[k - 1], h[k - 3], out);
    ladder(k - 2, c, h, out);
}

/// Rewrite every gate with more than `max_controls` controls (which must be
/// at least 2) into 2-control gates, exactly. Other gates pass through
/// unchanged. Panics never: a wide gate always has enough spare wires on a
/// `width`-wire circuit (it needs `k - 2` helpers distinct from its target
/// and `k` controls, and `width >= 2k - 1` holds for every stage-5 width).
pub fn lower_wide_gates(
    gates: &[XGate],
    width: usize,
    max_controls: usize,
) -> (Vec<XGate>, LowerReport) {
    assert!(max_controls >= 2, "max_controls must be at least 2");
    let mut out: Vec<XGate> = Vec::with_capacity(gates.len());
    let mut rep = LowerReport::default();
    for g in gates {
        let k = g.ctrls.len();
        rep.max_controls_in = rep.max_controls_in.max(k);
        if k <= max_controls {
            out.push(g.clone());
            continue;
        }
        let t = g.target;
        // Positive control wires (after conjugating the negatives by NOTs).
        let ctrl_wires: Vec<u16> = g.ctrls.iter().map(|&(w, _)| w).collect();
        let neg: Vec<u16> = g
            .ctrls
            .iter()
            .filter(|&&(_, p)| !p)
            .map(|&(w, _)| w)
            .collect();
        // k - 2 dirty helpers: the lowest-indexed wires that are neither the
        // target nor a control.
        let mut used = vec![false; width];
        used[t as usize] = true;
        for &w in &ctrl_wires {
            used[w as usize] = true;
        }
        let helpers: Vec<u16> = (0..width as u16)
            .filter(|&w| !used[w as usize])
            .take(k - 2)
            .collect();
        if helpers.len() < k - 2 {
            // Not enough spare wires for the dirty-ancilla chain: keep the
            // wide gate as it is (exact, just not lowered).
            rep.wide_kept += 1;
            out.push(g.clone());
            continue;
        }
        rep.wide_in += 1;
        let before = out.len();
        // Conjugate negatives to positive: NOT each negative control wire.
        for &w in &neg {
            out.push(XGate::x_gate(w));
        }
        barenco_xgate(t, &ctrl_wires, &helpers, &mut out);
        for &w in &neg {
            out.push(XGate::x_gate(w));
        }
        // The complement: `t ^= 1`.
        if g.comp {
            out.push(XGate::x_gate(t));
        }
        rep.gates_delta += (out.len() - before) as i64 - 1;
    }
    (out, rep)
}

#[cfg(test)]
#[path = "../../../../tests/unit/stages/post-processing/compression/lower2_tests.rs"]
mod lower2_tests;
