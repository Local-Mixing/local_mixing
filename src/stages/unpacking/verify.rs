//! Stage-7 checks.
//!
//! * [`check_window_exact`]: the synthesized (or fallback) window's gates
//!   compose, through stage 4's audited `window_anf`, to EXACTLY the
//!   deliverable's window (blob equality, as `check_windows_anf` does).
//! * [`check_window_sampled`]: the same gates on 1024 random full states:
//!   every changed wire equals the window ANF evaluated directly from the
//!   decoded masks, every other wire is unchanged (so a borrowed ancilla
//!   must have been restored). Independent of the composer.
//! * [`check_equiv`]: the whole circuit against `Deliverable::eval_lanes16`
//!   on random all-wire states (several batches of 1024).
//! * [`check_zero_slice_gates`]: x on wires 0..n, every other wire 0; the
//!   circuit must leave C(x) on wires n..2n.
use super::types::Stage7Error;
use crate::circuit::xgate::XGate;
use crate::stages::packing::anf::{AnfOverflow, MulStats, window_anf};
use crate::stages::packing::mono::{MonoWidthFn, with_mono_width};
use crate::stages::packing::pyrand::PyRandom;
use crate::stages::packing::serialize::{Deliverable, WindowView, encode_window};
use crate::stages::packing::types::CapSemantics;
use crate::stages::packing::verify::{Lanes, eval_gates_lanes16, random_lanes, view_blob};

/// Why the exact check could not confirm a window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExactFailure {
    /// The composition exceeded `budget` (the gates may still be right).
    Overflow(AnfOverflow),
    /// The composed ANF differs from the deliverable's window.
    Mismatch,
}

struct Exact<'a> {
    gates: &'a [XGate],
    view: &'a WindowView<'a>,
    budget: usize,
}

impl MonoWidthFn<Result<(), ExactFailure>> for Exact<'_> {
    fn call<const W: usize>(self) -> Result<(), ExactFailure> {
        let mut stats = MulStats::default();
        let anf = window_anf::<W>(self.gates, self.budget, CapSemantics::Canonical, &mut stats)
            .map_err(ExactFailure::Overflow)?;
        let blob = encode_window(&anf).map_err(|_| ExactFailure::Mismatch)?;
        if blob == view_blob(self.view) {
            Ok(())
        } else {
            Err(ExactFailure::Mismatch)
        }
    }
}

/// Exact window check: `window_anf(gates, budget)` encoded == the decoded
/// window's blob. `state_wires` sets the monomial width (≤ 1024).
pub fn check_window_exact(
    gates: &[XGate],
    view: &WindowView<'_>,
    state_wires: usize,
    budget: usize,
) -> Result<(), ExactFailure> {
    with_mono_width(
        state_wires,
        Exact {
            gates,
            view,
            budget,
        },
    )
}

/// The window ANF of `view` evaluated on all-wire lanes `inp`: the new
/// values of its changed wires, in order.
pub fn eval_view_lanes(view: &WindowView<'_>, inp: &[Lanes]) -> Vec<(u16, Lanes)> {
    let snap: Vec<Lanes> = view.support.iter().map(|&w| inp[w as usize]).collect();
    let fb = view.mask_bytes;
    view.changed()
        .map(|(w, masks)| {
            let mut acc = [0u64; 16];
            for m in masks.chunks_exact(fb) {
                let mut term = [!0u64; 16];
                for (bi, &byte) in m.iter().enumerate() {
                    let mut b = byte;
                    while b != 0 {
                        let l = bi * 8 + b.trailing_zeros() as usize;
                        b &= b - 1;
                        let s = &snap[l];
                        for k in 0..16 {
                            term[k] &= s[k];
                        }
                    }
                }
                for k in 0..16 {
                    acc[k] ^= term[k];
                }
            }
            (w, acc)
        })
        .collect()
}

/// Sampled window check on 1024 random full states (`random.Random(seed)`
/// over all `state_wires`): changed wires == the ANF, every other wire
/// unchanged, every gate inside the state.
pub fn check_window_sampled(
    gates: &[XGate],
    view: &WindowView<'_>,
    state_wires: usize,
    seed: u64,
) -> Result<(), String> {
    if let Some(w) = gates.iter().map(|g| g.max_wire() as usize).max()
        && w >= state_wires
    {
        return Err(format!(
            "gate on wire {w}, beyond the {state_wires}-wire state"
        ));
    }
    let mut rng = PyRandom::new(seed);
    let inp = random_lanes(&mut rng, state_wires);
    let mut st = inp.clone();
    eval_gates_lanes16(gates, &mut st);
    let want = eval_view_lanes(view, &inp);
    let mut changed = vec![false; state_wires];
    for (w, v) in &want {
        changed[*w as usize] = true;
        if st[*w as usize] != *v {
            return Err(format!("wire {w}: gates != window ANF"));
        }
    }
    for w in 0..state_wires {
        if !changed[w] && st[w] != inp[w] {
            return Err(format!("wire {w}: changed by the gates, not by the ANF"));
        }
    }
    Ok(())
}

/// Whole-circuit sampled equivalence: `batches` × 1024 random all-wire
/// states through the gates and through the deliverable. Returns the
/// number of samples.
pub fn check_equiv(
    gates: &[XGate],
    d: &Deliverable,
    seed: u64,
    batches: usize,
    what: &str,
) -> Result<usize, Stage7Error> {
    let nw = d.state_wires as usize;
    if let Some(w) = gates.iter().map(|g| g.max_wire() as usize).max()
        && w >= nw
    {
        return Err(Stage7Error::Verify(format!(
            "{what}: gate on wire {w}, beyond the {nw}-wire state"
        )));
    }
    let mut rng = PyRandom::new(seed);
    for b in 0..batches.max(1) {
        let x = random_lanes(&mut rng, nw);
        let mut sg = x.clone();
        eval_gates_lanes16(gates, &mut sg);
        let mut sd = x;
        d.eval_lanes16(&mut sd);
        let bad = (0..nw).filter(|&w| sg[w] != sd[w]).count();
        if bad > 0 {
            return Err(Stage7Error::Verify(format!(
                "{what}: {} gates differ from the deliverable on {bad} wires (batch {b} of 1024 random states, seed {seed})",
                gates.len()
            )));
        }
    }
    Ok(batches.max(1) * 1024)
}

/// The honest-slice check on a gate list: x on wires 0..n, every other wire
/// 0; wires n..2n must hold C(x). `samples` is rounded up to 1024s.
pub fn check_zero_slice_gates(
    c: &[XGate],
    gates: &[XGate],
    n: usize,
    state_wires: usize,
    samples: usize,
    seed: u64,
) -> Result<usize, Stage7Error> {
    let cw = c
        .iter()
        .map(|g| g.max_wire() as usize + 1)
        .max()
        .unwrap_or(0);
    if n == 0 || 2 * n > state_wires || cw > n {
        return Err(Stage7Error::Usage(format!(
            "zero slice: C uses {cw} wires, n={n}, state {state_wires}"
        )));
    }
    let mut rng = PyRandom::new(seed);
    let rounds = samples.div_ceil(1024).max(1);
    for r in 0..rounds {
        let x = random_lanes(&mut rng, n);
        let mut want = x.clone();
        eval_gates_lanes16(c, &mut want);
        let mut st = x;
        st.resize(state_wires, [0; 16]);
        eval_gates_lanes16(gates, &mut st);
        if let Some(w) = (0..n).find(|&w| st[n + w] != want[w]) {
            return Err(Stage7Error::Verify(format!(
                "zero slice: wire {} != C(x) wire {w} (batch {r})",
                n + w
            )));
        }
    }
    Ok(rounds * 1024)
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/unpacking/verify.rs"]
mod tests;
