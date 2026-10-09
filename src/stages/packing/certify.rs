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
    reorder_core(g.len(), |i| &g[i], order)
}

/// [`certify_reorder`] on a subsequence of an arena: `reference` lists arena
/// indices in the reference order, `order` the same indices in the new order.
pub fn certify_reorder_subset(
    arena: &[XGate],
    reference: &[u32],
    order: &[u32],
) -> Result<(), Stage4Error> {
    let fail = |msg: String| Err(Stage4Error::Verify(format!("reorder certificate: {msg}")));
    if order.len() != reference.len() {
        return fail(format!(
            "{} gates in the new order, reference has {}",
            order.len(),
            reference.len()
        ));
    }
    let mut at = vec![NONE; arena.len()];
    for (k, &o) in reference.iter().enumerate() {
        let o = o as usize;
        if o >= arena.len() || at[o] != NONE {
            return fail(format!("reference gate {o} out of range or repeated"));
        }
        at[o] = k as u32;
    }
    let mut pos = Vec::with_capacity(order.len());
    for &o in order {
        let k = at.get(o as usize).copied().unwrap_or(NONE);
        if k == NONE {
            return fail(format!("gate {o} is not in the reference"));
        }
        pos.push(k);
    }
    drop(at);
    reorder_core(reference.len(), |k| &arena[reference[k] as usize], &pos)
}

fn reorder_core<'a>(
    m: usize,
    gate: impl Fn(usize) -> &'a XGate,
    order: &[u32],
) -> Result<(), Stage4Error> {
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
    let nw = (0..m)
        .map(|i| gate(i).max_wire() as usize + 1)
        .max()
        .unwrap_or(0);
    // Access k of gate i is at acc_off[i] + k: k = 0 the target (W), then
    // the controls (R) in order.
    let mut acc_off = Vec::with_capacity(m + 1);
    acc_off.push(0u32);
    for i in 0..m {
        acc_off.push(acc_off.last().unwrap() + 1 + gate(i).ctrls.len() as u32);
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
    for (i, &off) in acc_off[..m].iter().enumerate() {
        let x = gate(i);
        let o = off as usize;
        acc_run[o] = assign(x.target as usize, true, &mut run_len, &mut run_next);
        for (k, &(c, _)) in x.ctrls.iter().enumerate() {
            acc_run[o + 1 + k] = assign(c as usize, false, &mut run_len, &mut run_next);
        }
    }
    // Replay in the new order: each wire must consume its runs in order.
    let mut at = head;
    let mut used = vec![0u32; nw];
    for (pos, &i) in order.iter().enumerate() {
        let x = gate(i as usize);
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

/// Per-wire position lists over the arena order with deletions: `pos` holds
/// each wire's positions ascending in `off[w]..off[w + 1]`; `next[k]` points
/// toward the smallest LIVE index at or after k (path-halving union-find,
/// sentinel `pos.len()`).
struct LiveLists {
    off: Vec<u32>,
    pos: Vec<u32>,
    next: Vec<u32>,
}

impl LiveLists {
    fn build(nw: usize, each: impl Fn(&mut dyn FnMut(usize, u32))) -> Self {
        let mut off = vec![0u32; nw + 1];
        each(&mut |w, _| off[w + 1] += 1);
        for w in 0..nw {
            off[w + 1] += off[w];
        }
        let total = off[nw] as usize;
        let mut fill = off.clone();
        let mut pos = vec![0u32; total];
        each(&mut |w, p| {
            pos[fill[w] as usize] = p;
            fill[w] += 1;
        });
        let next = (0..=total as u32).collect();
        LiveLists { off, pos, next }
    }

    fn find(&mut self, mut k: u32) -> u32 {
        while self.next[k as usize] != k {
            let up = self.next[self.next[k as usize] as usize];
            self.next[k as usize] = up;
            k = up;
        }
        k
    }

    /// Whether wire `w` has a live position strictly between `i` and `j`.
    fn any_between(&mut self, w: usize, i: u32, j: u32) -> bool {
        let (a, b) = (self.off[w] as usize, self.off[w + 1] as usize);
        let k = a + self.pos[a..b].partition_point(|&p| p <= i);
        let k = self.find(k as u32) as usize;
        k < b && self.pos[k] < j
    }

    fn delete(&mut self, w: usize, p: u32) {
        let (a, b) = (self.off[w] as usize, self.off[w + 1] as usize);
        let k = a + self.pos[a..b].partition_point(|&x| x < p);
        debug_assert!(k < b && self.pos[k] == p);
        self.next[k] = k as u32 + 1;
    }
}

/// Replay recorded cancellation passes on the arena order and return the live
/// flags. Each pair `(i, j)` (arena indices) must hold two identical live
/// gates, `i < j`, and every LIVE gate strictly between them must commute with
/// them (none reads their target, none writes one of their controls): then
/// `g X g = X`, and both are deleted. Runs in O(incidences · log).
pub fn certify_cancel(
    arena: &[XGate],
    passes: &[Vec<(u32, u32)>],
) -> Result<Vec<bool>, Stage4Error> {
    let fail = |msg: String| Err(Stage4Error::Verify(format!("cancel certificate: {msg}")));
    let m = arena.len();
    let nw = arena
        .iter()
        .map(|x| x.max_wire() as usize + 1)
        .max()
        .unwrap_or(0);
    let mut writes = LiveLists::build(nw, |f| {
        for (i, x) in arena.iter().enumerate() {
            f(x.target as usize, i as u32);
        }
    });
    let mut reads = LiveLists::build(nw, |f| {
        for (i, x) in arena.iter().enumerate() {
            for &(c, _) in &x.ctrls {
                f(c as usize, i as u32);
            }
        }
    });
    let mut live = vec![true; m];
    for (pi, pass) in passes.iter().enumerate() {
        for &(i, j) in pass {
            let (iu, ju) = (i as usize, j as usize);
            if i >= j || ju >= m || !live[iu] || !live[ju] {
                return fail(format!(
                    "pass {}: pair ({i}, {j}) not two live gates in order",
                    pi + 1
                ));
            }
            let g = &arena[iu];
            if *g != arena[ju] {
                return fail(format!("pass {}: gates {i} and {j} differ", pi + 1));
            }
            if reads.any_between(g.target as usize, i, j) {
                return fail(format!(
                    "pass {}: a live gate between {i} and {j} reads their target",
                    pi + 1
                ));
            }
            for &(c, _) in &g.ctrls {
                if writes.any_between(c as usize, i, j) {
                    return fail(format!(
                        "pass {}: a live gate between {i} and {j} writes their control {c}",
                        pi + 1
                    ));
                }
            }
            for p in [i, j] {
                live[p as usize] = false;
                writes.delete(g.target as usize, p);
                for &(c, _) in &g.ctrls {
                    reads.delete(c as usize, p);
                }
            }
        }
    }
    Ok(live)
}

/// The repair certificate: the cancellations replay ([`certify_cancel`]), and
/// the repaired order (`rep_origins`, arena indices) is a linear extension of
/// the dependency order of the surviving gates ([`certify_reorder_subset`]).
/// Fix slides are commutations, so they are covered.
pub fn certify_repair(
    arena: &[XGate],
    passes: &[Vec<(u32, u32)>],
    rep_origins: &[u32],
) -> Result<(), Stage4Error> {
    let live = certify_cancel(arena, passes)?;
    let survivors: Vec<u32> = (0..arena.len() as u32)
        .filter(|&i| live[i as usize])
        .collect();
    certify_reorder_subset(arena, &survivors, rep_origins)
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/packing/certify.rs"]
mod tests;
