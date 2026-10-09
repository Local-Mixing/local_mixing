//! Interval sets and winnable values (`window_pack3.py` 92-133, 175-226;
//! docs/STAGE4_RUST_DESIGN.md §4.8).
//!
//! A span `(b, e, a)` of B is HIDDEN when gates b..=e that depend on b and
//! that e depends on (the interval) all land in one window. A value `a` is
//! winnable when every one of its spans has an interval of at most `w_cap`
//! gates (with one rescue: a too-big interval may instead end at the
//! next-next writer of e's target). Intervals are numbered (iid) in
//! first-appearance order of the value over the spans, then span order.
//! The numeric value of `a` never orders anything.
use super::fingerprint::Spans;
use crate::circuit::xgate::XGate;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

/// Reusable per-thread buffers of [`interval_set`].
pub struct IntervalScratch {
    tw: Vec<u8>,
    tr: Vec<u8>,
    twb: Vec<u8>,
    trb: Vec<u8>,
    fwd: Vec<u8>,
}

impl IntervalScratch {
    pub fn new(n: usize) -> Self {
        IntervalScratch {
            tw: vec![0; n],
            tr: vec![0; n],
            twb: vec![0; n],
            trb: vec![0; n],
            fwd: Vec::new(),
        }
    }
}

/// `interval_set(gates, b, e, n, cap)`: the members in the oracle's order
/// (backward walk from e-1 down to b+1, then b, then e), or `None` when the
/// backward walk exceeds `cap - 2` members.
pub fn interval_set(gates: &[XGate], b: usize, e: usize, n: usize, cap: usize) -> Option<Vec<u32>> {
    let mut s = IntervalScratch::new(n);
    interval_set_with(gates, b, e, cap, &mut s)
}

#[allow(clippy::needless_range_loop)] // transcribes WP index for index
pub fn interval_set_with(
    gates: &[XGate],
    b: usize,
    e: usize,
    cap: usize,
    s: &mut IntervalScratch,
) -> Option<Vec<u32>> {
    s.tw.fill(0);
    s.tr.fill(0);
    s.twb.fill(0);
    s.trb.fill(0);
    let gb = &gates[b];
    s.tw[gb.target as usize] = 1;
    for &(c, _) in &gb.ctrls {
        s.tr[c as usize] = 1;
    }
    let len = e - b - 1;
    s.fwd.clear();
    s.fwd.resize(len, 0);
    for idx in b + 1..e {
        let g = &gates[idx];
        let tainted =
            s.tr[g.target as usize] != 0 || g.ctrls.iter().any(|&(c, _)| s.tw[c as usize] != 0);
        if tainted {
            s.fwd[idx - b - 1] = 1;
            s.tw[g.target as usize] = 1;
            for &(c, _) in &g.ctrls {
                s.tr[c as usize] = 1;
            }
        }
    }
    let ge = &gates[e];
    s.trb[ge.target as usize] = 1;
    for &(c, _) in &ge.ctrls {
        s.twb[c as usize] = 1;
    }
    let mut members: Vec<u32> = Vec::new();
    for idx in (b + 1..e).rev() {
        if s.fwd[idx - b - 1] == 0 {
            continue;
        }
        let g = &gates[idx];
        let tainted =
            s.twb[g.target as usize] != 0 || g.ctrls.iter().any(|&(c, _)| s.trb[c as usize] != 0);
        if tainted {
            members.push(idx as u32);
            if members.len() + 2 > cap {
                return None;
            }
            s.trb[g.target as usize] = 1;
            for &(c, _) in &g.ctrls {
                s.twb[c as usize] = 1;
            }
        }
    }
    members.push(b as u32);
    members.push(e as u32);
    Some(members)
}

/// Counters of the `winnable values` log line.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WinnableStats {
    /// Distinct values with at least one span (`len(by_val)`).
    pub by_val: usize,
    pub winnable: usize,
    pub disq_long: usize,
    pub disq_big: usize,
    pub rescued: usize,
}

/// Intervals by iid, in compressed-row form.
pub struct Intervals {
    /// Members of iid i (ascending): `mem[mem_off[i]..mem_off[i+1]]`.
    pub mem_off: Vec<u32>,
    pub mem: Vec<u32>,
    /// Dense winnable-value index of each iid.
    pub ival: Vec<u32>,
    /// Span start and (possibly rescued) end of each iid.
    pub ib: Vec<u32>,
    pub ie: Vec<u32>,
    /// Per dense value: the A id and its iid range `val_off[v]..val_off[v+1]`.
    pub val_a: Vec<u32>,
    pub val_off: Vec<u32>,
    /// Per gate, the iids containing it (ascending).
    pub member_off: Vec<u32>,
    pub member: Vec<u32>,
    pub stats: WinnableStats,
}

impl Intervals {
    pub fn len(&self) -> usize {
        self.ival.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ival.is_empty()
    }

    #[inline]
    pub fn members(&self, iid: usize) -> &[u32] {
        &self.mem[self.mem_off[iid] as usize..self.mem_off[iid + 1] as usize]
    }

    #[inline]
    pub fn size(&self, iid: usize) -> usize {
        (self.mem_off[iid + 1] - self.mem_off[iid]) as usize
    }

    #[inline]
    pub fn member_of(&self, g: usize) -> &[u32] {
        &self.member[self.member_off[g] as usize..self.member_off[g + 1] as usize]
    }

    #[inline]
    pub fn value_iids(&self, v: usize) -> std::ops::Range<usize> {
        self.val_off[v] as usize..self.val_off[v + 1] as usize
    }
}

enum Verdict {
    Forced,
    Long,
    /// Too-big interval; spans rescued before the failure.
    Big(usize),
    /// (sorted members, b, e) per span; spans rescued.
    Win(Vec<(Vec<u32>, u32, u32)>, usize),
}

/// `window_pack3.py` 175-226. `n` is the state width used by the oracle
/// (max of the two computed widths).
pub fn build_intervals(
    b_gates: &[XGate],
    spans: &Spans,
    n: usize,
    w_cap: usize,
    span_cap: usize,
) -> Intervals {
    // writer positions per wire
    let mut wr_pos: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (i, g) in b_gates.iter().enumerate() {
        wr_pos[g.target as usize].push(i as u32);
    }
    // by_val in first-appearance order
    let mut slot: FxHashMap<u32, u32> = FxHashMap::default();
    let mut by_val: Vec<(u32, Vec<(u32, u32)>)> = Vec::new();
    for &(b, e, a) in &spans.spans {
        let k = *slot.entry(a).or_insert_with(|| {
            by_val.push((a, Vec::new()));
            (by_val.len() - 1) as u32
        });
        by_val[k as usize].1.push((b, e));
    }
    let verdicts: Vec<Verdict> = by_val
        .par_iter()
        .map_init(
            || IntervalScratch::new(n),
            |s, (a, ss)| {
                if spans.is_forced(*a) {
                    return Verdict::Forced;
                }
                if ss.iter().any(|&(b, e)| (e - b + 1) as usize > span_cap) {
                    return Verdict::Long;
                }
                let mut mems = Vec::with_capacity(ss.len());
                let mut rescued = 0usize;
                for &(b, e) in ss {
                    let (b, mut e) = (b as usize, e as usize);
                    let mut mem = interval_set_with(b_gates, b, e, w_cap, s);
                    if mem.is_none() {
                        let ws = &wr_pos[b_gates[e].target as usize];
                        let k = ws.partition_point(|&x| (x as usize) < e);
                        // ws[k+1] - b + 1 <= span_cap
                        if k + 1 < ws.len() && (ws[k + 1] as usize - b) < span_cap {
                            let e2 = ws[k + 1] as usize;
                            mem = interval_set_with(b_gates, b, e2, w_cap, s);
                            if mem.is_some() {
                                rescued += 1;
                                e = e2;
                            }
                        }
                    }
                    match mem {
                        None => return Verdict::Big(rescued),
                        Some(mut m) => {
                            m.sort_unstable();
                            mems.push((m, b as u32, e as u32));
                        }
                    }
                }
                Verdict::Win(mems, rescued)
            },
        )
        .collect();

    let mut stats = WinnableStats {
        by_val: by_val.len(),
        ..Default::default()
    };
    let mut mem_off = vec![0u32];
    let mut mem = Vec::new();
    let (mut ival, mut ib, mut ie) = (Vec::new(), Vec::new(), Vec::new());
    let (mut val_a, mut val_off) = (Vec::new(), vec![0u32]);
    for ((a, _), v) in by_val.iter().zip(verdicts) {
        match v {
            Verdict::Forced => {}
            Verdict::Long => stats.disq_long += 1,
            Verdict::Big(r) => {
                stats.disq_big += 1;
                stats.rescued += r;
            }
            Verdict::Win(mems, r) => {
                stats.winnable += 1;
                stats.rescued += r;
                let vi = val_a.len() as u32;
                val_a.push(*a);
                for (m, b, e) in mems {
                    mem.extend_from_slice(&m);
                    mem_off.push(mem.len() as u32);
                    ival.push(vi);
                    ib.push(b);
                    ie.push(e);
                }
                val_off.push(ival.len() as u32);
            }
        }
    }
    // member_of (CSR, iid ascending per gate)
    let m = b_gates.len();
    let mut member_off = vec![0u32; m + 1];
    for &g in &mem {
        member_off[g as usize + 1] += 1;
    }
    for i in 0..m {
        member_off[i + 1] += member_off[i];
    }
    let mut fill = member_off.clone();
    let mut member = vec![0u32; mem.len()];
    for iid in 0..ival.len() {
        for &g in &mem[mem_off[iid] as usize..mem_off[iid + 1] as usize] {
            member[fill[g as usize] as usize] = iid as u32;
            fill[g as usize] += 1;
        }
    }
    Intervals {
        mem_off,
        mem,
        ival,
        ib,
        ie,
        val_a,
        val_off,
        member_off,
        member,
        stats,
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/packing/intervals.rs"]
mod tests;
