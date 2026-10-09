//! The joint window packer (`window_pack3.py` 152-505; docs/STAGE4_RUST_DESIGN.md
//! §4.9). A faithful transcription: the same fingerprints and value keys,
//! the same intervals and iid order, the same epoch readiness, heaps,
//! ordered `committed` map, stable sorts, window caps on the CPython 3.9 set
//! emulator, and the same quirks:
//!
//! * on a compose failure the window closes AFTER g is appended and BEFORE
//!   g's interval bookkeeping;
//! * the contagion snapshot goes stale within a pass;
//! * a burst continues through a mid-burst close;
//! * caps are tested before a gate is added, so windows overshoot;
//! * the pre-emptive close is unconditional on `win_load`, so it can emit a
//!   duplicate or zero boundary (and an empty window in the statistics);
//! * the final filter keeps `0 < b < m` without dedup.
//!
//! The oracle's unreachable paths (`burst stuck`, `scheduler stuck`, an
//! interval resumed in a later window) and the lemma "the lowest unscheduled
//! gate is the minimum unscheduled entry of `anyready`" are release-mode
//! checks returning `Stage4Error::Internal`.
use super::anf::{MulStats, WindowState};
use super::certify::certify_reorder;
use super::fingerprint::{Spans, index_a, inputs, spans_b};
use super::intervals::{Intervals, WinnableStats, build_intervals};
use super::mono::{MonoWidthFn, with_mono_width};
use super::oracle_io::OracleCircuit;
use super::types::{CapSemantics, PackParams, Stage4Error};
use crate::circuit::xgate::XGate;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::io::Write;

/// The packer's incremental window ANF behind a width-erased interface.
pub trait Composer {
    fn compose(&mut self, g: &XGate, sem: CapSemantics, stats: &mut MulStats) -> bool;
    fn monos(&self) -> usize;
    fn reset(&mut self);
}

impl<const W: usize> Composer for WindowState<W> {
    fn compose(&mut self, g: &XGate, sem: CapSemantics, stats: &mut MulStats) -> bool {
        WindowState::compose(self, g, sem, stats)
    }
    fn monos(&self) -> usize {
        WindowState::monos(self)
    }
    fn reset(&mut self) {
        WindowState::reset(self)
    }
}

struct MakeComposer(usize);

impl MonoWidthFn<Box<dyn Composer>> for MakeComposer {
    fn call<const W: usize>(self) -> Box<dyn Composer> {
        Box::new(WindowState::<W>::new(self.0))
    }
}

/// (median, p90, max) as the oracle reports them: `v[k//2]`, `v[9k//10]`, `v[-1]`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Quantiles {
    pub med: usize,
    pub p90: usize,
    pub max: usize,
}

fn quantiles(v: &[usize]) -> Quantiles {
    let mut s = v.to_vec();
    s.sort_unstable();
    let k = s.len();
    Quantiles {
        med: s[k / 2],
        p90: s[9 * k / 10],
        max: s[k - 1],
    }
}

/// Every number in the packer's log, plus the product counters.
#[derive(Clone, Debug, Default)]
pub struct PackReport {
    pub gates: usize,
    pub state_wires: usize,
    pub spans: usize,
    pub carried: usize,
    pub forced: usize,
    pub agree: usize,
    pub agree_of: usize,
    pub winnable: WinnableStats,
    pub intervals: usize,
    /// `close_window` calls (empty windows included), the oracle's `windows`.
    pub windows: usize,
    pub hidden_spans: usize,
    pub hidden_values: usize,
    /// Death causes in first-increment order (the oracle's dict order).
    pub deaths: Vec<(&'static str, u64)>,
    pub verify_agree: usize,
    pub carried_emitted: usize,
    pub exposed: usize,
    pub distinct_a: usize,
    /// (carried, exposed) for the classes data (< 64), mid (< 128), band.
    pub classes: [(usize, usize); 3],
    pub win_gates: Quantiles,
    pub win_monos: Quantiles,
    pub mul: MulStats,
    /// Gates whose compose aborted (each closes its window right after it).
    pub compose_aborts: usize,
    /// Boundaries kept after the `0 < b < m` filter, and how many of them
    /// repeat the previous one.
    pub bounds: usize,
    pub duplicate_bounds: usize,
}

impl PackReport {
    pub fn verify_ok(&self) -> bool {
        self.verify_agree == self.agree_of
    }

    pub fn death(&self, k: &str) -> u64 {
        self.deaths
            .iter()
            .find(|(n, _)| *n == k)
            .map_or(0, |(_, v)| *v)
    }

    /// `{'burst_close': 16, 'sibling': 332}`: Python's `dict` repr.
    pub fn deaths_repr(&self) -> String {
        let parts: Vec<String> = self
            .deaths
            .iter()
            .map(|(k, v)| format!("'{k}': {v}"))
            .collect();
        format!("{{{}}}", parts.join(", "))
    }
}

pub struct Packed {
    /// Output header width: B's computed width (WP writes `nb`).
    pub n_out: usize,
    /// `order[i]` = index into B of the i-th emitted gate.
    pub order: Vec<u32>,
    /// Window boundaries after the `0 < b < m` filter, in emission order.
    pub bounds: Vec<usize>,
    pub report: PackReport,
}

impl Packed {
    pub fn gates<'a>(&'a self, b: &'a [XGate]) -> impl Iterator<Item = &'a XGate> + 'a {
        self.order.iter().map(move |&i| &b[i as usize])
    }
}

/// `window_pack3.load`'s width: `max(header n, max wire + 1)`, where the
/// max wire of an empty circuit is 0.
pub fn wp_width(c: &OracleCircuit) -> usize {
    c.computed_n.max(1)
}

const NONE: u32 = u32::MAX;

/// The insertion-ordered `committed` dict (iid -> remaining members).
struct Committed {
    ent: Vec<(u32, u32)>,
    live: Vec<bool>,
    slot: Vec<u32>,
    nlive: usize,
}

impl Committed {
    fn new(n_iv: usize) -> Self {
        Committed {
            ent: Vec::new(),
            live: Vec::new(),
            slot: vec![NONE; n_iv],
            nlive: 0,
        }
    }

    fn is_empty(&self) -> bool {
        self.nlive == 0
    }

    fn get(&self, iid: u32) -> Option<u32> {
        match self.slot[iid as usize] {
            NONE => None,
            s => Some(self.ent[s as usize].1),
        }
    }

    fn insert(&mut self, iid: u32, rem: u32) {
        debug_assert_eq!(self.slot[iid as usize], NONE);
        self.slot[iid as usize] = self.ent.len() as u32;
        self.ent.push((iid, rem));
        self.live.push(true);
        self.nlive += 1;
    }

    fn set(&mut self, iid: u32, rem: u32) {
        let s = self.slot[iid as usize] as usize;
        self.ent[s].1 = rem;
    }

    fn remove(&mut self, iid: u32) {
        let s = self.slot[iid as usize] as usize;
        self.slot[iid as usize] = NONE;
        self.live[s] = false;
        self.nlive -= 1;
        if self.ent.len() > 64 && self.nlive * 2 < self.ent.len() {
            self.compact();
        }
    }

    fn compact(&mut self) {
        let mut k = 0;
        for i in 0..self.ent.len() {
            if self.live[i] {
                self.ent[k] = self.ent[i];
                self.slot[self.ent[k].0 as usize] = k as u32;
                k += 1;
            }
        }
        self.ent.truncate(k);
        self.live.clear();
        self.live.resize(k, true);
    }

    /// Live entries in insertion order.
    fn entries(&self) -> Vec<(u32, u32)> {
        self.ent
            .iter()
            .zip(&self.live)
            .filter(|(_, l)| **l)
            .map(|(e, _)| *e)
            .collect()
    }

    fn clear(&mut self) {
        for &(iid, _) in &self.ent {
            self.slot[iid as usize] = NONE;
        }
        self.ent.clear();
        self.live.clear();
        self.nlive = 0;
    }
}

/// Readiness buckets keyed (wire, count) as one dense id per wire event.
struct Buckets {
    /// `base[w] + k - 1` = id of (w, k).
    base: Vec<u32>,
    off: Vec<u32>,
    items: Vec<u32>,
}

impl Buckets {
    fn build(counts: &[u32], entries: &[(u16, u32, u32)]) -> Buckets {
        let mut base = Vec::with_capacity(counts.len());
        let mut acc = 0u32;
        for &c in counts {
            base.push(acc);
            acc += c;
        }
        let id = |w: u16, k: u32| (base[w as usize] + k - 1) as usize;
        let mut off = vec![0u32; acc as usize + 1];
        for &(w, k, _) in entries {
            off[id(w, k) + 1] += 1;
        }
        for i in 0..acc as usize {
            off[i + 1] += off[i];
        }
        let mut fill = off.clone();
        let mut items = vec![0u32; entries.len()];
        for &(w, k, g) in entries {
            let d = id(w, k);
            items[fill[d] as usize] = g;
            fill[d] += 1;
        }
        Buckets { base, off, items }
    }

    fn range(&self, w: usize, k: u32) -> std::ops::Range<usize> {
        let id = (self.base[w] + k - 1) as usize;
        self.off[id] as usize..self.off[id + 1] as usize
    }
}

struct Sched<'a> {
    gates: &'a [XGate],
    iv: &'a Intervals,
    g_cap: usize,
    b_mono: usize,
    pad: usize,
    cmax: usize,
    sem: CapSemantics,
    read_bucket: Buckets,
    write_bucket: Buckets,
    unmet: Vec<u32>,
    writes_done: Vec<u32>,
    readers_done: Vec<u32>,
    scheduled: Vec<bool>,
    isready: Vec<bool>,
    lowest: usize,
    order: Vec<u32>,
    boundaries: Vec<usize>,
    win_load: usize,
    win_id: usize,
    in_burst: bool,
    committed: Committed,
    dead: Vec<bool>,
    started: Vec<bool>,
    hidden: Vec<bool>,
    hidden_count: usize,
    prot: Vec<u32>,
    filler: BinaryHeap<Reverse<u32>>,
    anyready: BinaryHeap<Reverse<u32>>,
    deaths: Vec<(&'static str, u64)>,
    ws: Box<dyn Composer>,
    win_gates_hist: Vec<usize>,
    win_monos_hist: Vec<usize>,
    stats: MulStats,
    compose_aborts: usize,
}

impl<'a> Sched<'a> {
    fn new(gates: &'a [XGate], iv: &'a Intervals, n: usize, p: &PackParams) -> Sched<'a> {
        let m = gates.len();
        // epoch bookkeeping (WP 230-245)
        let mut writes_seen = vec![0u32; n];
        let mut readers_before = vec![0u32; n];
        let mut unmet = vec![0u32; m];
        // (wire, count, gate): gate waits until the count-th write (read) of wire
        let mut rd: Vec<(u16, u32, u32)> = Vec::new();
        let mut wr: Vec<(u16, u32, u32)> = Vec::new();
        for (i, g) in gates.iter().enumerate() {
            let mut cnt = 0;
            for &(c, _) in &g.ctrls {
                let k = writes_seen[c as usize];
                if k > 0 {
                    rd.push((c, k, i as u32));
                    cnt += 1;
                }
            }
            let t = g.target as usize;
            if readers_before[t] > 0 {
                wr.push((g.target, readers_before[t], i as u32));
                cnt += 1;
            }
            unmet[i] = cnt;
            for &(c, _) in &g.ctrls {
                readers_before[c as usize] += 1;
            }
            writes_seen[t] += 1;
        }
        let read_bucket = Buckets::build(&writes_seen, &rd);
        let write_bucket = Buckets::build(&readers_before, &wr);
        let niv = iv.len();
        let mut prot = vec![0u32; m];
        for &g in &iv.mem {
            prot[g as usize] += 1;
        }
        Sched {
            gates,
            iv,
            g_cap: p.g,
            b_mono: p.b_mono,
            pad: p.pad,
            cmax: p.cmax,
            sem: p.cap_semantics,
            read_bucket,
            write_bucket,
            unmet,
            writes_done: vec![0; n],
            readers_done: vec![0; n],
            scheduled: vec![false; m],
            isready: vec![false; m],
            lowest: 0,
            order: Vec::with_capacity(m),
            boundaries: Vec::new(),
            win_load: 0,
            win_id: 0,
            in_burst: false,
            committed: Committed::new(niv),
            dead: vec![false; niv],
            started: vec![false; niv],
            hidden: vec![false; niv],
            hidden_count: 0,
            prot,
            filler: BinaryHeap::new(),
            anyready: BinaryHeap::new(),
            deaths: Vec::new(),
            ws: with_mono_width(n, MakeComposer(p.mul_cap())),
            win_gates_hist: Vec::new(),
            win_monos_hist: Vec::new(),
            stats: MulStats::default(),
            compose_aborts: 0,
        }
    }

    fn death(&mut self, k: &'static str) {
        match self.deaths.iter_mut().find(|(n, _)| *n == k) {
            Some(e) => e.1 += 1,
            None => self.deaths.push((k, 1)),
        }
    }

    fn on_ready(&mut self, g: u32) {
        self.isready[g as usize] = true;
        self.anyready.push(Reverse(g));
        if self.prot[g as usize] == 0 {
            self.filler.push(Reverse(g));
        }
    }

    fn satisfy(&mut self, g: u32) {
        self.unmet[g as usize] -= 1;
        if self.unmet[g as usize] == 0 {
            self.on_ready(g);
        }
    }

    fn release(&mut self, iid: usize) {
        let iv = self.iv;
        for &h in iv.members(iid) {
            let h = h as usize;
            self.prot[h] -= 1;
            if self.prot[h] == 0 && self.isready[h] && !self.scheduled[h] {
                self.filler.push(Reverse(h as u32));
            }
        }
    }

    fn kill_value(&mut self, v: usize, trigger: Option<usize>) {
        for sib in self.iv.value_iids(v) {
            if !self.dead[sib] {
                self.dead[sib] = true;
                if Some(sib) != trigger {
                    self.death("sibling");
                }
                if !self.started[sib] {
                    self.release(sib);
                }
            }
        }
    }

    fn close_window(&mut self, reason: &'static str) {
        let mut doomed = Vec::new();
        for (iid, rem) in self.committed.entries() {
            if rem > 0 && !self.dead[iid as usize] {
                self.death(reason);
                doomed.push((self.iv.ival[iid as usize] as usize, iid as usize));
            }
        }
        self.committed.clear();
        for (v, iid) in doomed {
            self.dead[iid] = true;
            self.kill_value(v, Some(iid));
        }
        self.boundaries.push(self.order.len());
        self.win_gates_hist.push(self.win_load);
        self.win_monos_hist.push(self.ws.monos());
        self.win_load = 0;
        self.win_id += 1;
        self.ws.reset();
    }

    fn reason(&self) -> &'static str {
        if self.in_burst {
            "burst_close"
        } else {
            "drain_close"
        }
    }

    fn schedule(&mut self, g: u32) -> Result<(), Stage4Error> {
        let gi = g as usize;
        if self.win_load > 0 {
            let (mcap, gcap) = if self.in_burst {
                (2 * self.b_mono, 2 * self.g_cap)
            } else {
                (self.b_mono, self.g_cap)
            };
            if self.ws.monos() > mcap || self.win_load >= gcap {
                self.close_window(self.reason());
            }
        }
        self.scheduled[gi] = true;
        self.order.push(g);
        self.win_load += 1;
        while self.lowest < self.scheduled.len() && self.scheduled[self.lowest] {
            self.lowest += 1;
        }
        let gates = self.gates;
        let gate = &gates[gi];
        if !self.ws.compose(gate, self.sem, &mut self.stats) {
            // pathological product: isolate g by closing right after it
            self.compose_aborts += 1;
            self.close_window(self.reason());
        }
        let iv = self.iv;
        for &iid in iv.member_of(gi) {
            let iid = iid as usize;
            if self.dead[iid] {
                continue;
            }
            if !self.started[iid] {
                self.started[iid] = true;
                self.committed.insert(iid as u32, iv.size(iid) as u32);
                self.release(iid);
            }
            match self.committed.get(iid as u32) {
                Some(rem) => {
                    if rem == 1 {
                        self.hidden[iid] = true;
                        self.hidden_count += 1;
                        self.committed.remove(iid as u32);
                    } else {
                        self.committed.set(iid as u32, rem - 1);
                    }
                }
                None => {
                    return Err(Stage4Error::Internal(format!(
                        "packer: interval {iid} resumed in a later window (unreachable in window_pack3)"
                    )));
                }
            }
        }
        let t = gate.target as usize;
        self.writes_done[t] += 1;
        for j in self.read_bucket.range(t, self.writes_done[t]) {
            let h = self.read_bucket.items[j];
            self.satisfy(h);
        }
        for &(c, _) in &gate.ctrls {
            let c = c as usize;
            self.readers_done[c] += 1;
            for j in self.write_bucket.range(c, self.readers_done[c]) {
                let h = self.write_bucket.items[j];
                self.satisfy(h);
            }
        }
        Ok(())
    }

    /// Pop the lowest ready unscheduled gate (lazy deletion), checking the
    /// lemma that it is the lowest unscheduled gate overall.
    fn pop_lowest_ready(&mut self) -> Option<u32> {
        while let Some(&Reverse(h)) = self.anyready.peek() {
            if self.scheduled[h as usize] {
                self.anyready.pop();
            } else {
                break;
            }
        }
        self.anyready.pop().map(|Reverse(h)| h)
    }

    fn check_lowest(&self, h: u32) -> Result<(), Stage4Error> {
        if h as usize != self.lowest {
            return Err(Stage4Error::Internal(format!(
                "packer: lowest ready gate {h} is not the lowest unscheduled gate {}",
                self.lowest
            )));
        }
        Ok(())
    }

    fn burst(&mut self, iid: usize) -> Result<(), Stage4Error> {
        let iv = self.iv;
        for &g in iv.members(iid) {
            let gi = g as usize;
            if self.scheduled[gi] {
                continue;
            }
            while !self.isready[gi] {
                let h = self
                    .pop_lowest_ready()
                    .ok_or_else(|| Stage4Error::Internal("packer: burst stuck".into()))?;
                self.check_lowest(h)?;
                self.schedule(h)?;
            }
            if !self.scheduled[gi] {
                self.schedule(g)?;
            }
        }
        Ok(())
    }

    fn contagion(&mut self) -> Result<(), Stage4Error> {
        let mut changed = true;
        while changed && !self.committed.is_empty() {
            changed = false;
            let mut snap = self.committed.entries();
            snap.sort_by_key(|&(_, rem)| rem); // stable: ties keep insertion order
            for (iid2, _) in snap {
                if let Some(rem) = self.committed.get(iid2)
                    && !self.dead[iid2 as usize]
                    && rem as usize <= self.cmax
                {
                    let wid0 = self.win_id;
                    self.burst(iid2 as usize)?;
                    changed = true;
                    if self.win_id != wid0 {
                        return Ok(());
                    }
                }
            }
        }
        Ok(())
    }

    fn run(&mut self, log: &mut dyn Write) -> Result<(), Stage4Error> {
        let m = self.gates.len();
        for i in 0..m {
            if self.unmet[i] == 0 {
                self.on_ready(i as u32);
            }
        }
        let iv = self.iv;
        let niv = iv.len();
        let mut by_end: Vec<u32> = (0..niv as u32).collect();
        by_end.sort_by_key(|&i| iv.ie[i as usize]); // stable
        let progress = (niv / 10).max(1);
        for (nproc, &iid) in by_end.iter().enumerate() {
            let iid = iid as usize;
            if self.dead[iid] {
                continue;
            }
            if self.started[iid] {
                if self.committed.get(iid as u32).is_some() {
                    self.in_burst = true;
                    self.burst(iid)?;
                    self.contagion()?;
                    self.in_burst = false;
                }
                continue;
            }
            let e_x = iv.ie[iid];
            while let Some(&Reverse(g)) = self.filler.peek() {
                if self.scheduled[g as usize] || self.prot[g as usize] > 0 {
                    self.filler.pop();
                    continue;
                }
                if g > e_x {
                    break;
                }
                self.filler.pop();
                self.schedule(g)?;
            }
            if self.dead[iid] || self.started[iid] {
                continue;
            }
            if (self.g_cap as i64 - self.win_load as i64) < (iv.size(iid) + self.pad) as i64
                || self.ws.monos() > self.b_mono * 3 / 4
            {
                self.close_window("drain_close");
            }
            self.in_burst = true;
            self.burst(iid)?;
            self.contagion()?;
            self.in_burst = false;
            if (nproc + 1) % progress == 0 {
                writeln!(
                    log,
                    "  processed {}/{} intervals, {}/{} gates, windows={}, hidden spans={}",
                    nproc + 1,
                    niv,
                    self.order.len(),
                    m,
                    self.win_id,
                    self.hidden_count
                )
                .map_err(|e| Stage4Error::io("<pack log>", e))?;
                log.flush().map_err(|e| Stage4Error::io("<pack log>", e))?;
            }
        }
        while self.order.len() < m {
            let h = self.pop_lowest_ready().ok_or_else(|| {
                Stage4Error::Internal("packer: scheduler stuck: no ready gates".into())
            })?;
            self.check_lowest(h)?;
            self.schedule(h)?;
        }
        if self.win_load > 0 {
            self.close_window("final");
        }
        Ok(())
    }
}

/// Distinct ids among `spans` and `forced`, flagged in `seen` (sized |A|).
fn carried_flags(sp: &Spans, n_a: usize) -> (Vec<bool>, usize) {
    let mut seen = vec![false; n_a];
    let mut k = 0;
    for a in sp
        .spans
        .iter()
        .map(|s| s.2)
        .chain(sp.forced.iter().copied())
    {
        if !seen[a as usize] {
            seen[a as usize] = true;
            k += 1;
        }
    }
    (seen, k)
}

fn wire_class(w: u16) -> usize {
    if w < 64 {
        0
    } else if w < 128 {
        1
    } else {
        2
    }
}

fn wl(log: &mut dyn Write, s: std::fmt::Arguments) -> Result<(), Stage4Error> {
    log.write_fmt(s)
        .and_then(|_| log.write_all(b"\n"))
        .and_then(|_| log.flush())
        .map_err(|e| Stage4Error::io("<pack log>", e))
}

/// `window_pack3.py A B W_cap G B_mono span_cap label`: pack B (a reorder of
/// a circuit carrying A's values) into windows, writing the oracle's log
/// lines to `log`. Exact certificate: the emitted order is a linear
/// extension of B's dependency order (`certify_reorder`); the oracle's
/// sampled VERIFY is also required to pass.
pub fn pack(
    a: &OracleCircuit,
    b: &OracleCircuit,
    p: &PackParams,
    label: &str,
    log: &mut dyn Write,
) -> Result<Packed, Stage4Error> {
    if b.gates.is_empty() {
        return Err(Stage4Error::Input(
            "B has no gates (window_pack3.py fails on an empty circuit)".into(),
        ));
    }
    let na = wp_width(a);
    let nb = wp_width(b);
    let n = na.max(nb);
    if n > 1024 {
        return Err(Stage4Error::Usage(format!(
            "{n} state wires: stage 4 supports at most 1024"
        )));
    }
    let m = b.gates.len();
    let fp = inputs(n, p.fp_seed);
    let ai = index_a(&a.gates, &fp, na);
    let sp = spans_b(&b.gates, None, &fp, nb, na, &ai);
    let (_, carried) = carried_flags(&sp, a.gates.len());
    let agree_of = na.min(nb);
    wl(
        log,
        format_args!(
            "{label}: B={m} gates; {} interior spans, {carried} carried values ({} output-forced); \
             input function check {}/{agree_of}; W_cap={} G={} B_mono={}",
            sp.spans.len(),
            sp.forced.len(),
            sp.agree,
            p.w_cap,
            p.g,
            p.b_mono
        ),
    )?;
    let iv = build_intervals(&b.gates, &sp, n, p.w_cap, p.span_cap);
    let ws = iv.stats;
    wl(
        log,
        format_args!(
            "winnable values: {} of {} carried (disq: {} long-span, {} big-interval; \
             {} spans rescued by next-next writer)",
            ws.winnable, ws.by_val, ws.disq_long, ws.disq_big, ws.rescued
        ),
    )?;
    wl(log, format_args!("intervals to pack: {}", iv.len()))?;

    let mut s = Sched::new(&b.gates, &iv, n, p);
    s.run(log)?;
    let order = std::mem::take(&mut s.order);
    let bounds: Vec<usize> = s
        .boundaries
        .iter()
        .copied()
        .filter(|&x| 0 < x && x < m)
        .collect();
    let nh = (0..iv.val_a.len())
        .filter(|&v| iv.value_iids(v).all(|i| s.hidden[i]))
        .count();
    let mut r = PackReport {
        gates: m,
        state_wires: n,
        spans: sp.spans.len(),
        carried,
        forced: sp.forced.len(),
        agree: sp.agree,
        agree_of,
        winnable: ws,
        intervals: iv.len(),
        windows: s.win_id,
        hidden_spans: s.hidden_count,
        hidden_values: nh,
        deaths: s.deaths.clone(),
        distinct_a: ai.distinct(),
        win_gates: quantiles(&s.win_gates_hist),
        win_monos: quantiles(&s.win_monos_hist),
        mul: s.stats,
        compose_aborts: s.compose_aborts,
        bounds: bounds.len(),
        duplicate_bounds: bounds.windows(2).filter(|w| w[0] == w[1]).count(),
        ..Default::default()
    };
    drop(s);
    wl(
        log,
        format_args!(
            "scheduled: {} windows, {}/{} interval-spans hidden, {}/{} winnable VALUES fully hidden \
             by construction",
            r.windows, r.hidden_spans, r.intervals, nh, ws.winnable
        ),
    )?;
    wl(log, format_args!("death causes: {}", r.deaths_repr()))?;

    // --- verify (the oracle's sampled check) + exposure statistics
    let sp2 = spans_b(&b.gates, Some(&order), &fp, nb, na, &ai);
    r.verify_agree = sp2.agree;
    wl(
        log,
        format_args!(
            "VERIFY: emitted function check {}/{agree_of} wires ({})",
            sp2.agree,
            if r.verify_ok() { "OK" } else { "FAIL" }
        ),
    )?;
    let mut bset = bounds.clone();
    bset.sort_unstable();
    let (car2, carried2) = carried_flags(&sp2, a.gates.len());
    let mut exposed = vec![false; a.gates.len()];
    for &x in &sp2.forced {
        exposed[x as usize] = true;
    }
    for &(sb, se, sa) in &sp2.spans {
        let i = bset.partition_point(|&x| x < sb as usize + 1);
        if i < bset.len() && bset[i] <= se as usize {
            exposed[sa as usize] = true;
        }
    }
    let mut n_exp = 0;
    for (aid, &c) in car2.iter().enumerate() {
        if c {
            let cls = wire_class(ai.a_wire[aid]);
            r.classes[cls].0 += 1;
            if exposed[aid] {
                r.classes[cls].1 += 1;
            }
        }
        if exposed[aid] {
            n_exp += 1;
        }
    }
    r.carried_emitted = carried2;
    r.exposed = n_exp;
    wl(
        log,
        format_args!(
            "RESULT: carried in emitted order {carried2}; distinct A values exposed at window \
             boundaries: {n_exp} ({:.1}% of carried, {:.1}% of all distinct A)",
            (100 * n_exp) as f64 / carried2.max(1) as f64,
            (100 * n_exp) as f64 / r.distinct_a.max(1) as f64
        ),
    )?;
    for (cls, name) in ["data", "mid", "band"].iter().enumerate() {
        let (car, ex) = r.classes[cls];
        if car > 0 {
            wl(
                log,
                format_args!(
                    "  {name}: exposed {ex}/{car} ({:.1}%)",
                    (100 * ex) as f64 / car as f64
                ),
            )?;
        }
    }
    let (g, mo) = (r.win_gates, r.win_monos);
    wl(
        log,
        format_args!(
            "window gates: med={} p90={} max={}; window monomials: med={} p90={} max={}; windows={}",
            g.med, g.p90, g.max, mo.med, mo.p90, mo.max, r.windows
        ),
    )?;
    if !r.verify_ok() {
        return Err(Stage4Error::Verify(format!(
            "pack VERIFY: emitted function check {}/{agree_of} wires",
            sp2.agree
        )));
    }
    certify_reorder(&b.gates, &order)?;
    Ok(Packed {
        n_out: nb,
        order,
        bounds,
        report: r,
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/packing/scheduler.rs"]
mod tests;
