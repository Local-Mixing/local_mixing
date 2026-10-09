//! The cancel pass of `order_repair.py`: delete pairs of IDENTICAL gates that
//! commutation can bring together (every gate between them commutes with
//! them), iterated to a fixed point.
use super::seq::MARKER;
use crate::circuit::xgate::XGate;
use rustc_hash::FxHashMap;

/// Positions per wire in compressed form: `pos[off[w]..off[w + 1]]`, ascending.
struct WireLists {
    off: Vec<u32>,
    pos: Vec<u32>,
}

impl WireLists {
    fn build(nw: usize, items: impl Fn(&mut dyn FnMut(usize, u32))) -> Self {
        let mut cnt = vec![0u32; nw + 1];
        items(&mut |w, _| cnt[w + 1] += 1);
        for w in 0..nw {
            cnt[w + 1] += cnt[w];
        }
        let mut fill = cnt.clone();
        let mut pos = vec![0u32; *cnt.last().unwrap() as usize];
        items(&mut |w, p| {
            pos[fill[w] as usize] = p;
            fill[w] += 1;
        });
        WireLists { off: cnt, pos }
    }

    /// The first position on wire `w` strictly after `i`, or `none`.
    #[inline]
    fn next_after(&self, w: usize, i: u32, none: u32) -> u32 {
        let l = &self.pos[self.off[w] as usize..self.off[w + 1] as usize];
        let k = l.partition_point(|&p| p <= i);
        l.get(k).copied().unwrap_or(none)
    }
}

/// One pass (`cancel_pass`): returns the dead flags and the cancelled pairs
/// `(i, j)` as positions in `gates`, in the order they were found.
///
/// For each live gate i in order: the blocker is the first later gate that
/// reads i's target or writes one of its controls; i cancels with the first
/// LIVE identical gate before the blocker. Gates killed earlier in the pass
/// still block (they are in the read/write lists).
pub fn cancel_pass(gates: &[&XGate]) -> (Vec<bool>, Vec<(u32, u32)>) {
    let m = gates.len();
    assert!(m < u32::MAX as usize, "circuit too long");
    let none = m as u32;
    let nw = gates
        .iter()
        .map(|g| g.max_wire() as usize + 1)
        .max()
        .unwrap_or(0);
    let reads = WireLists::build(nw, |f| {
        for (i, g) in gates.iter().enumerate() {
            for &(c, _) in &g.ctrls {
                f(c as usize, i as u32);
            }
        }
    });
    let writes = WireLists::build(nw, |f| {
        for (i, g) in gates.iter().enumerate() {
            f(g.target as usize, i as u32);
        }
    });
    // next_same[i]: the next position holding an identical gate.
    let mut next_same = vec![none; m];
    {
        let mut last: FxHashMap<&XGate, u32> = FxHashMap::default();
        for i in (0..m).rev() {
            if let Some(j) = last.insert(gates[i], i as u32) {
                next_same[i] = j;
            }
        }
    }
    let mut dead = vec![false; m];
    let mut pairs = Vec::new();
    for i in 0..m {
        if dead[i] {
            continue;
        }
        let g = gates[i];
        let iu = i as u32;
        let mut blk = reads.next_after(g.target as usize, iu, none);
        for &(c, _) in &g.ctrls {
            blk = blk.min(writes.next_after(c as usize, iu, none));
        }
        let mut j = next_same[i];
        while j < blk && dead[j as usize] {
            j = next_same[j as usize];
        }
        if j < blk {
            dead[i] = true;
            dead[j as usize] = true;
            pairs.push((iu, j));
        }
    }
    (dead, pairs)
}

/// `run_cancel(seq)`: cancel passes until one finds no pair. Returns the pairs
/// of every productive pass, as ORIGINS. Markers stay in place.
pub fn run_cancel(seq: &mut Vec<u32>, arena: &[XGate]) -> Vec<Vec<(u32, u32)>> {
    let mut passes = Vec::new();
    loop {
        let gidx: Vec<usize> = (0..seq.len()).filter(|&i| seq[i] != MARKER).collect();
        let gates: Vec<&XGate> = gidx.iter().map(|&i| &arena[seq[i] as usize]).collect();
        let (dead, pairs) = cancel_pass(&gates);
        drop(gates);
        if pairs.is_empty() {
            return passes;
        }
        passes.push(
            pairs
                .iter()
                .map(|&(i, j)| (seq[gidx[i as usize]], seq[gidx[j as usize]]))
                .collect(),
        );
        let mut drop_at = vec![false; seq.len()];
        for (k, &d) in dead.iter().enumerate() {
            if d {
                drop_at[gidx[k]] = true;
            }
        }
        let mut k = 0;
        seq.retain(|_| {
            let keep = !drop_at[k];
            k += 1;
            keep
        });
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/stages/packing/repair/cancel.rs"]
mod tests;
