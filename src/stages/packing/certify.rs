//! Exact release-mode certificates (docs/STAGE4_RUST_DESIGN.md §4.11a).
//!
//! [`certify_reorder`]: `order` is a permutation of `0..|g|` and, on every
//! wire, the RUN-CANONICAL form of the access sequence is the same in both
//! orders. The access sequence of wire w lists the gates touching w, each
//! labelled W (writes w) or R (controls on w); its run-canonical form is the
//! sequence of maximal same-label runs, each as a set. Equal forms on every
//! wire iff every conflicting pair (one writes w, the other reads w) is
//! ordered identically, i.e. both orders are linear extensions of the same
//! dependency order; non-conflicting gates commute (writes on one wire are
//! XORs, reads never change state), so the circuits are equal as functions.
use super::types::Stage4Error;
use crate::circuit::xgate::XGate;

const NONE: u32 = u32::MAX;

/// Check `order` (indices into `g`) against `g`'s own order. Runs in
/// O(gate-wire incidences).
pub fn certify_reorder(g: &[XGate], order: &[u32]) -> Result<(), Stage4Error> {
    let m = g.len();
    let fail = |msg: String| Err(Stage4Error::Verify(format!("reorder certificate: {msg}")));
    if order.len() != m {
        return fail(format!("{} gates emitted, circuit has {m}", order.len()));
    }
    let mut seen = vec![false; m];
    for &i in order {
        let i = i as usize;
        if i >= m || seen[i] {
            return fail(format!("gate {i} out of range or emitted twice"));
        }
        seen[i] = true;
    }
    drop(seen);
    let nw = g
        .iter()
        .map(|x| x.max_wire() as usize + 1)
        .max()
        .unwrap_or(0);
    // Access k of gate i is at acc_off[i] + k: k = 0 the target (W), then
    // the controls (R) in order.
    let mut acc_off = Vec::with_capacity(m + 1);
    acc_off.push(0u32);
    for x in g {
        acc_off.push(acc_off.last().unwrap() + 1 + x.ctrls.len() as u32);
    }
    let mut acc_run = vec![0u32; *acc_off.last().unwrap() as usize];
    let mut run_len: Vec<u32> = Vec::new();
    let mut run_next: Vec<u32> = Vec::new();
    let mut head = vec![NONE; nw];
    let mut cur = vec![NONE; nw];
    let mut cur_w = vec![false; nw]; // label of the wire's current run (true = W)
    let mut assign = |w: usize, is_w: bool, run_len: &mut Vec<u32>, run_next: &mut Vec<u32>| {
        if cur[w] == NONE || cur_w[w] != is_w {
            let id = run_len.len() as u32;
            run_len.push(0);
            run_next.push(NONE);
            if cur[w] == NONE {
                head[w] = id;
            } else {
                run_next[cur[w] as usize] = id;
            }
            cur[w] = id;
            cur_w[w] = is_w;
        }
        run_len[cur[w] as usize] += 1;
        cur[w]
    };
    for (i, x) in g.iter().enumerate() {
        let o = acc_off[i] as usize;
        acc_run[o] = assign(x.target as usize, true, &mut run_len, &mut run_next);
        for (k, &(c, _)) in x.ctrls.iter().enumerate() {
            acc_run[o + 1 + k] = assign(c as usize, false, &mut run_len, &mut run_next);
        }
    }
    // Replay in the new order: each wire must consume its runs in order.
    let mut at = head;
    let mut used = vec![0u32; nw];
    for (pos, &i) in order.iter().enumerate() {
        let x = &g[i as usize];
        let o = acc_off[i as usize] as usize;
        let wires = std::iter::once(x.target).chain(x.ctrls.iter().map(|&(c, _)| c));
        for (k, w) in wires.enumerate() {
            let w = w as usize;
            while used[w] == run_len[at[w] as usize] {
                at[w] = run_next[at[w] as usize];
                used[w] = 0;
                if at[w] == NONE {
                    return fail(format!("wire {w} over-consumed at position {pos}"));
                }
            }
            if acc_run[o + k] != at[w] {
                return fail(format!(
                    "gate {i} at position {pos} crosses a conflicting access on wire {w}"
                ));
            }
            used[w] += 1;
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/packing/certify.rs"]
mod tests;
