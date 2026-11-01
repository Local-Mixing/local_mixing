//! Configurable role banks; growth writes require a reference-free bank.
//!
//! Ported verbatim from the `stage2_compute_pilot` generator (`src/band.rs`).

use super::fire::Mask;
use super::{ThreeMaskParams, WIRE_COHORT};
use crate::circuit::XGate;
use rand::{Rng, seq::SliceRandom};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Serving,
    Retiring,
    Growing,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rotation {
    pub fire: usize,
    pub promoted: usize,
    pub retiring: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CohortPhase {
    Serving,
    Retiring,
    Done,
}

/// A promotion cohort of the wire-turnover mode: the wires promoted
/// together. Masks are drawn inside one serving cohort, the oldest serving
/// cohort retires as a unit, and its wires leave one by one as their last
/// reference is replaced.
#[derive(Clone, Debug)]
struct Cohort {
    phase: CohortPhase,
    promoted_at: usize,
    retired_at: Option<usize>,
    /// Band-relative wire indices still in the cohort.
    wires: Vec<usize>,
}

/// The wire-turnover state: `Band::phase`, `Band::updates` (growth
/// rewrites) and `cohort_of` are per wire, cohorts replace banks.
#[derive(Clone, Debug)]
struct WireState {
    cohorts: Vec<Cohort>,
    /// Per wire: its cohort while serving or retiring.
    cohort_of: Vec<Option<usize>>,
    /// Round-robin cursor over the band for growth targets.
    next_target: usize,
}

const NO_COHORT: usize = usize::MAX;

pub struct Band {
    pub np: usize,
    pub size: usize,
    /// Per bank, or per wire in the wire-turnover mode.
    pub phase: Vec<Phase>,
    pub refs: Vec<usize>,
    pub updates: Vec<usize>,
    cursors: Vec<usize>,
    next_bank: usize,
    service_age: Vec<usize>,
    retire_since: Vec<Option<usize>>,
    local_masks: bool,
    cross_bank_growth: bool,
    cross_bank_pairs: bool,
    bank_disjoint_masks: bool,
    prefer_fresh: bool,
    sweeps_per_visit: usize,
    max_retiring: usize,
    pub local_mask_fallbacks: usize,
    pub retirement_waits: Vec<usize>,
    pub promotion_stalls: usize,
    pub rotations: Vec<Rotation>,
    pub retired: usize,
    wire: Option<WireState>,
    /// Wire turnover: per drained wire, fires between its cohort's
    /// retirement and its own return to growth.
    pub wire_limbo_waits: Vec<usize>,
    /// Helper lanes (fire interleaving): reference-free wires that growth
    /// never writes, refreshed every fire. Bank mode: the drained wires of
    /// retiring banks; wire mode: the most rewritten growing wires, which
    /// growth then skips. Absolute wire ids, ascending.
    pub reserve: Vec<u16>,
    lanes: bool,
    reserved: Vec<bool>,
    /// The reserve's slots (band-relative wires): a wire keeps its slot,
    /// hence its lane, until it leaves the candidate set; only emptied
    /// slots are refilled, so lanes stay disjoint across consecutive fires.
    slots: Vec<Option<usize>>,
}

impl Band {
    pub fn new(np: usize, r: usize, cfg: &ThreeMaskParams) -> Self {
        if cfg.wire_turnover {
            return Self::new_wire_mode(np, r, cfg);
        }
        let banks = cfg.banks;
        let serving = banks / 2;
        Self {
            np,
            size: r / banks,
            phase: (0..banks)
                .map(|i| {
                    if i < serving {
                        Phase::Serving
                    } else {
                        Phase::Growing
                    }
                })
                .collect(),
            refs: vec![0; r],
            updates: vec![0; banks],
            cursors: vec![0; banks],
            next_bank: serving,
            service_age: (0..banks)
                .map(|i| if i < serving { i } else { 0 })
                .collect(),
            retire_since: vec![None; banks],
            local_masks: cfg.local_masks,
            cross_bank_growth: cfg.cross_bank_growth,
            cross_bank_pairs: cfg.cross_bank_pairs,
            bank_disjoint_masks: cfg.bank_disjoint_masks,
            prefer_fresh: cfg.prefer_fresh,
            sweeps_per_visit: cfg.growth_sweeps_per_visit,
            max_retiring: if cfg.max_retiring == 0 {
                serving - 1
            } else {
                cfg.max_retiring
            },
            local_mask_fallbacks: 0,
            retirement_waits: vec![],
            promotion_stalls: 0,
            rotations: vec![],
            retired: 0,
            wire: None,
            wire_limbo_waits: vec![],
            reserve: vec![],
            lanes: cfg.interleave_window > 0,
            reserved: vec![false; r],
            slots: vec![None; super::helper_reserve(r)],
        }
    }

    /// The helper reserve for the coming fire (see `reserve`).
    fn refresh_reserve(&mut self) {
        if !self.lanes {
            return;
        }
        let r = self.refs.len();
        let candidate = |band: &Self, w: usize| match &band.wire {
            None => band.phase_of((band.np + w) as u16) == Phase::Retiring && band.refs[w] == 0,
            Some(_) => band.phase[w] == Phase::Growing,
        };
        for k in 0..self.slots.len() {
            if let Some(w) = self.slots[k]
                && !candidate(self, w)
            {
                self.slots[k] = None;
            }
        }
        let held: Vec<usize> = self.slots.iter().flatten().copied().collect();
        let mut fresh: Vec<usize> = (0..r)
            .filter(|&w| candidate(self, w) && !held.contains(&w))
            .collect();
        // most rewritten first (wire mode: `updates` is per wire); bank mode
        // takes drained retiring wires by position (`updates` is per bank)
        let wire_mode = self.wire.is_some();
        fresh.sort_by_key(|&w| {
            (
                std::cmp::Reverse(if wire_mode { self.updates[w] } else { 0 }),
                w,
            )
        });
        let mut fresh = fresh.into_iter();
        for slot in self.slots.iter_mut() {
            if slot.is_none() {
                *slot = fresh.next();
            }
        }
        self.reserved.fill(false);
        for w in self.slots.iter().flatten() {
            self.reserved[*w] = true;
        }
        // in slot order (lane = slot mod HELPER_LANES); a partial reserve is
        // shorter and the compute falls back to the shuffled pool
        self.reserve = self
            .slots
            .iter()
            .flatten()
            .map(|&w| (self.np + w) as u16)
            .collect();
    }

    /// Wire mode: the wire of retiring cohort `c` with the fewest remaining
    /// references, so that wires leave the cohort one after another.
    pub fn drain_focus(&self, c: usize) -> Option<u16> {
        let ws = self.wire.as_ref()?;
        ws.cohorts[c]
            .wires
            .iter()
            .copied()
            .filter(|&w| self.refs[w] > 0)
            .min_by_key(|&w| (self.refs[w], w))
            .map(|w| (self.np + w) as u16)
    }

    /// Half the band serving in positional cohorts of [`WIRE_COHORT`]
    /// wires (aged like the bank mode's initial banks), the rest growing.
    fn new_wire_mode(np: usize, r: usize, cfg: &ThreeMaskParams) -> Self {
        let cohorts_total = r / WIRE_COHORT;
        let serving = cohorts_total / 2;
        let cohorts: Vec<Cohort> = (0..serving)
            .map(|k| Cohort {
                phase: CohortPhase::Serving,
                promoted_at: k,
                retired_at: None,
                wires: (k * WIRE_COHORT..(k + 1) * WIRE_COHORT).collect(),
            })
            .collect();
        let serving_wires = serving * WIRE_COHORT;
        Self {
            np,
            size: 1,
            phase: (0..r)
                .map(|w| {
                    if w < serving_wires {
                        Phase::Serving
                    } else {
                        Phase::Growing
                    }
                })
                .collect(),
            refs: vec![0; r],
            updates: vec![0; r],
            cursors: vec![],
            next_bank: 0,
            service_age: vec![],
            retire_since: vec![],
            local_masks: cfg.local_masks,
            cross_bank_growth: cfg.cross_bank_growth,
            cross_bank_pairs: false,
            bank_disjoint_masks: false,
            prefer_fresh: cfg.prefer_fresh,
            sweeps_per_visit: cfg.growth_sweeps_per_visit,
            max_retiring: if cfg.max_retiring == 0 {
                serving - 1
            } else {
                cfg.max_retiring
            },
            local_mask_fallbacks: 0,
            retirement_waits: vec![],
            promotion_stalls: 0,
            rotations: vec![],
            retired: 0,
            wire: Some(WireState {
                cohorts,
                cohort_of: (0..r)
                    .map(|w| (w < serving_wires).then_some(w / WIRE_COHORT))
                    .collect(),
                next_target: serving_wires,
            }),
            wire_limbo_waits: vec![],
            reserve: vec![],
            lanes: cfg.interleave_window > 0,
            reserved: vec![false; r],
            slots: vec![None; super::helper_reserve(r)],
        }
    }

    pub fn wire_mode(&self) -> bool {
        self.wire.is_some()
    }

    /// The bank of `w`; in the wire-turnover mode its cohort id, or a
    /// distinct id per growing wire (no cohort).
    pub fn bank(&self, w: u16) -> usize {
        match &self.wire {
            None => (w as usize - self.np) / self.size,
            Some(ws) => {
                ws.cohort_of[w as usize - self.np].unwrap_or(NO_COHORT - (w as usize - self.np))
            }
        }
    }

    /// The phase of wire `w` (its bank's, or its own in the wire mode).
    pub fn phase_of(&self, w: u16) -> Phase {
        if self.wire.is_some() {
            self.phase[w as usize - self.np]
        } else {
            self.phase[self.bank(w)]
        }
    }

    pub fn sample(&mut self, excluded: &[u16], rng: &mut impl Rng) -> Mask {
        self.sample_mode(excluded, false, rng)
    }

    pub fn sample_fresh(&mut self, excluded: &[u16], rng: &mut impl Rng) -> Mask {
        self.sample_mode(excluded, self.prefer_fresh, rng)
    }

    fn sample_mode(&mut self, excluded: &[u16], prefer_fresh: bool, rng: &mut impl Rng) -> Mask {
        let mut avail: Vec<_> = (self.np..self.np + self.refs.len())
            .map(|w| w as u16)
            .filter(|&w| self.phase_of(w) == Phase::Serving && !excluded.contains(&w))
            .collect();
        assert!(
            avail.len() >= 3,
            "insufficient serving wires; configuration/exclusion invariant violated"
        );
        avail.shuffle(rng);
        if let Some(ws) = &self.wire {
            // Cohort-local masks: the freshest (or a random) serving cohort
            // with three available wires of both parities; otherwise any.
            if self.local_masks {
                let eligible: Vec<usize> = (0..ws.cohorts.len())
                    .filter(|&c| {
                        ws.cohorts[c].phase == CohortPhase::Serving && {
                            let in_c: Vec<_> = avail
                                .iter()
                                .copied()
                                .filter(|&w| ws.cohort_of[w as usize - self.np] == Some(c))
                                .collect();
                            in_c.len() >= 3
                                && in_c
                                    .iter()
                                    .any(|&w| (w as usize - self.np).is_multiple_of(2))
                                && in_c.iter().any(|&w| (w as usize - self.np) % 2 == 1)
                        }
                    })
                    .collect();
                if eligible.is_empty() {
                    self.local_mask_fallbacks += 1;
                } else {
                    let c = if prefer_fresh && !self.rotations.is_empty() {
                        *eligible
                            .iter()
                            .max_by_key(|&&c| ws.cohorts[c].promoted_at)
                            .unwrap()
                    } else {
                        eligible[rng.random_range(0..eligible.len())]
                    };
                    avail.retain(|&w| ws.cohort_of[w as usize - self.np] == Some(c));
                }
            }
            let x = *avail
                .iter()
                .find(|&&w| (w as usize - self.np).is_multiple_of(2))
                .unwrap();
            let y = *avail
                .iter()
                .find(|&&w| (w as usize - self.np) % 2 == 1)
                .unwrap();
            let z = *avail.iter().find(|&&w| w != x && w != y).unwrap();
            let m = [x, y, z];
            for w in m {
                self.refs[w as usize - self.np] += 1;
            }
            return m;
        }
        if self.cross_bank_pairs {
            let banks = self.phase.len();
            let mut counts = vec![[0usize; 2]; banks];
            for &w in &avail {
                counts[self.bank(w)][(w as usize - self.np) % 2] += 1;
            }
            let mut pairs = Vec::new();
            for a in 0..banks {
                if counts[a][0] == 0 {
                    continue;
                }
                for b in 0..banks {
                    if a == b || counts[b][1] == 0 {
                        continue;
                    }
                    if counts[a].iter().sum::<usize>() + counts[b].iter().sum::<usize>() >= 3 {
                        pairs.push((a, b));
                    }
                }
            }
            if !pairs.is_empty() {
                let (a, b) = if prefer_fresh && !self.rotations.is_empty() {
                    *pairs
                        .iter()
                        .max_by_key(|&&(a, b)| {
                            (
                                self.service_age[a].min(self.service_age[b]),
                                self.service_age[a].max(self.service_age[b]),
                            )
                        })
                        .unwrap()
                } else {
                    pairs[rng.random_range(0..pairs.len())]
                };
                let x = *avail
                    .iter()
                    .find(|&&w| self.bank(w) == a && (w as usize - self.np).is_multiple_of(2))
                    .unwrap();
                let y = *avail
                    .iter()
                    .find(|&&w| self.bank(w) == b && (w as usize - self.np) % 2 == 1)
                    .unwrap();
                let z = *avail
                    .iter()
                    .find(|&&w| w != x && w != y && (self.bank(w) == a || self.bank(w) == b))
                    .unwrap();
                let m = [x, y, z];
                for w in m {
                    self.refs[w as usize - self.np] += 1;
                }
                return m;
            }
            self.local_mask_fallbacks += 1;
        }
        if self.local_masks {
            let eligible: Vec<_> = (0..self.phase.len())
                .filter(|&b| {
                    if self.bank_disjoint_masks && excluded.iter().any(|&w| self.bank(w) == b) {
                        return false;
                    }
                    let ws: Vec<_> = avail
                        .iter()
                        .copied()
                        .filter(|&w| self.bank(w) == b)
                        .collect();
                    ws.len() >= 3
                        && ws.iter().any(|&w| (w as usize - self.np).is_multiple_of(2))
                        && ws.iter().any(|&w| (w as usize - self.np) % 2 == 1)
                })
                .collect();
            if !eligible.is_empty() {
                let b = if prefer_fresh && !self.rotations.is_empty() {
                    *eligible
                        .iter()
                        .max_by_key(|&&b| self.service_age[b])
                        .unwrap()
                } else {
                    eligible[rng.random_range(0..eligible.len())]
                };
                avail.retain(|&w| self.bank(w) == b);
            } else {
                // Narrow configurations can exhaust each small bank separately.
                // A cross-bank mask remains correct and is explicitly counted.
                self.local_mask_fallbacks += 1;
            }
        }
        // Every live quadratic pair crosses the fixed parity partition.
        // Quartic helper products can then pair like colors, so no helper
        // update reproduces ANY live mask's quadratic pair. Among three
        // band literals there is likewise always a same-color pair.
        let x = *avail
            .iter()
            .find(|&&w| (w as usize - self.np).is_multiple_of(2))
            .unwrap();
        let y = *avail
            .iter()
            .find(|&&w| (w as usize - self.np) % 2 == 1)
            .unwrap();
        let z = *avail.iter().find(|&&w| w != x && w != y).unwrap();
        let m = [x, y, z];
        for w in m {
            self.refs[w as usize - self.np] += 1;
        }
        m
    }

    pub fn release(&mut self, m: Mask) {
        for w in m {
            self.refs[w as usize - self.np] =
                self.refs[w as usize - self.np].checked_sub(1).unwrap();
        }
    }

    pub fn borrow_helpers(&self) -> Vec<u16> {
        let out: Vec<_> = (self.np..self.np + self.refs.len())
            .map(|w| w as u16)
            .filter(|&w| self.phase_of(w) == Phase::Growing)
            .take(4)
            .collect();
        assert_eq!(out.len(), 4);
        out
    }

    pub fn borrow_helpers_pool(&self, rng: &mut impl Rng) -> Vec<u16> {
        let mut out: Vec<_> = (self.np..self.np + self.refs.len())
            .map(|w| w as u16)
            .filter(|&w| match self.phase_of(w) {
                Phase::Growing => true,
                Phase::Retiring => self.refs[w as usize - self.np] == 0,
                Phase::Serving => false,
            })
            .collect();
        out.shuffle(rng);
        assert!(out.len() >= 4);
        out
    }

    pub fn helpers(&self, rng: &mut impl Rng) -> [u16; 2] {
        let mut avail: Vec<_> = (self.np..self.np + self.refs.len())
            .map(|w| w as u16)
            .filter(|&w| self.phase_of(w) == Phase::Growing)
            .collect();
        avail.shuffle(rng);
        [avail[0], avail[1]]
    }

    pub fn retiring_mask(&self, m: Mask) -> bool {
        m.iter().any(|&w| self.phase_of(w) == Phase::Retiring)
    }

    /// The bank (or cohort) retiring longest, by id.
    pub fn oldest_retiring(&self) -> Option<usize> {
        if let Some(ws) = &self.wire {
            return (0..ws.cohorts.len())
                .filter(|&c| ws.cohorts[c].phase == CohortPhase::Retiring)
                .min_by_key(|&c| ws.cohorts[c].retired_at.unwrap());
        }
        (0..self.phase.len())
            .filter(|&b| self.phase[b] == Phase::Retiring)
            .min_by_key(|&b| self.retire_since[b].unwrap())
    }

    /// Wire mode: rewrite growing wires round-robin, each with one control
    /// among the other growing wires (a rewritten one when any exists) and
    /// one anywhere in the band, so a grown value composes fresh material
    /// with current serving coordinates. Every target is reference-free.
    fn growth_wire_mode(&mut self, count: usize, rng: &mut impl Rng) -> Vec<XGate> {
        let r = self.refs.len();
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let growing: Vec<usize> = (0..r)
                .filter(|&w| self.phase[w] == Phase::Growing && !self.reserved[w])
                .collect();
            if growing.len() < 2 {
                break;
            }
            let ws = self.wire.as_mut().expect("wire mode");
            let t = (0..r)
                .map(|d| (ws.next_target + d) % r)
                .find(|&w| self.phase[w] == Phase::Growing && !self.reserved[w])
                .unwrap();
            ws.next_target = (t + 1) % r;
            let rewritten: Vec<usize> = growing
                .iter()
                .copied()
                .filter(|&w| w != t && self.updates[w] > 0)
                .collect();
            let y = if rewritten.is_empty() {
                let others: Vec<usize> = growing.iter().copied().filter(|&w| w != t).collect();
                others[rng.random_range(0..others.len())]
            } else {
                rewritten[rng.random_range(0..rewritten.len())]
            };
            let x = loop {
                let w = rng.random_range(0..r);
                if w != t && w != y && !self.reserved[w] {
                    break w;
                }
            };
            assert_eq!(self.refs[t], 0);
            out.push(
                XGate::conj(
                    (self.np + t) as u16,
                    [
                        ((self.np + x) as u16, rng.random_bool(0.5)),
                        ((self.np + y) as u16, rng.random_bool(0.5)),
                    ],
                )
                .unwrap(),
            );
            self.updates[t] += 1;
        }
        out
    }

    /// Wire mode: drained retiring wires return to growth at once; on a
    /// rotation fire, while fewer cohorts than the cap retire, the most
    /// rewritten mature growing wires (half of each parity) form a new
    /// serving cohort and the oldest serving cohort retires.
    fn rotate_wire_mode(&mut self, fire: usize, interval: usize, maturity: usize) {
        let r = self.refs.len();
        let ws = self.wire.as_mut().expect("wire mode");
        for w in 0..r {
            if self.phase[w] == Phase::Retiring && self.refs[w] == 0 {
                let c = ws.cohort_of[w].take().expect("a retiring wire's cohort");
                let cohort = &mut ws.cohorts[c];
                cohort.wires.retain(|&v| v != w);
                self.phase[w] = Phase::Growing;
                self.updates[w] = 0;
                self.wire_limbo_waits
                    .push(fire - cohort.retired_at.expect("retired"));
                if cohort.wires.is_empty() {
                    cohort.phase = CohortPhase::Done;
                    self.retired += 1;
                    self.retirement_waits
                        .push(fire - cohort.retired_at.expect("retired"));
                }
            }
        }
        if interval == 0 || !fire.is_multiple_of(interval) {
            return;
        }
        let retiring = ws
            .cohorts
            .iter()
            .filter(|c| c.phase == CohortPhase::Retiring)
            .count();
        if retiring >= self.max_retiring {
            self.promotion_stalls += 1;
            return;
        }
        let mut mature: Vec<usize> = (0..r)
            .filter(|&w| self.phase[w] == Phase::Growing && self.updates[w] >= maturity)
            .collect();
        // most rewritten first, ties by position
        mature.sort_by_key(|&w| (std::cmp::Reverse(self.updates[w]), w));
        let half = WIRE_COHORT / 2;
        let even: Vec<usize> = mature
            .iter()
            .copied()
            .filter(|w| w.is_multiple_of(2))
            .take(half)
            .collect();
        let odd: Vec<usize> = mature
            .iter()
            .copied()
            .filter(|w| w % 2 == 1)
            .take(half)
            .collect();
        if even.len() < half || odd.len() < half {
            self.promotion_stalls += 1;
            return;
        }
        let old = (0..ws.cohorts.len())
            .filter(|&c| ws.cohorts[c].phase == CohortPhase::Serving)
            .min_by_key(|&c| ws.cohorts[c].promoted_at)
            .expect("a serving cohort");
        let new = ws.cohorts.len();
        let mut wires: Vec<usize> = even.into_iter().chain(odd).collect();
        wires.sort_unstable();
        for &w in &wires {
            self.phase[w] = Phase::Serving;
            ws.cohort_of[w] = Some(new);
        }
        ws.cohorts.push(Cohort {
            phase: CohortPhase::Serving,
            promoted_at: fire + r / (2 * WIRE_COHORT),
            retired_at: None,
            wires,
        });
        let cohort = &mut ws.cohorts[old];
        cohort.phase = CohortPhase::Retiring;
        cohort.retired_at = Some(fire);
        for &w in &cohort.wires {
            self.phase[w] = Phase::Retiring;
        }
        self.rotations.push(Rotation {
            fire,
            promoted: new,
            retiring: old,
        });
    }

    /// Pre-growth (RC 2026-11-01): `sweeps` rewrites of every serving wire,
    /// in band order, before any mask references the band, so the first
    /// serving coordinates are not the raw band seed. Each gate composes a
    /// growing wire with another band wire, random polarities. Bank and
    /// wire modes alike; growth counters are untouched (serving wires
    /// restart them when they return to growth).
    pub fn pregrow(&mut self, sweeps: usize, rng: &mut impl Rng) -> Vec<XGate> {
        let r = self.refs.len();
        let np = self.np;
        let serving: Vec<usize> = (0..r)
            .filter(|&w| self.phase_of((np + w) as u16) == Phase::Serving)
            .collect();
        let growing: Vec<usize> = (0..r)
            .filter(|&w| self.phase_of((np + w) as u16) == Phase::Growing)
            .collect();
        assert!(
            !growing.is_empty() && r >= 3,
            "pre-growth needs a growing wire"
        );
        let mut out = Vec::with_capacity(sweeps * serving.len());
        for _ in 0..sweeps {
            for &t in &serving {
                assert_eq!(self.refs[t], 0, "pre-growth before any reference");
                let y = growing[rng.random_range(0..growing.len())];
                let x = loop {
                    let w = rng.random_range(0..r);
                    if w != t && w != y {
                        break w;
                    }
                };
                out.push(
                    XGate::conj(
                        (np + t) as u16,
                        [
                            ((np + x) as u16, rng.random_bool(0.5)),
                            ((np + y) as u16, rng.random_bool(0.5)),
                        ],
                    )
                    .expect("distinct pre-growth wires"),
                );
            }
        }
        out
    }

    pub fn growth(&mut self, count: usize, rng: &mut impl Rng) -> Vec<XGate> {
        if self.wire.is_some() {
            return self.growth_wire_mode(count, rng);
        }
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let banks = self.phase.len();
            let b = (0..banks)
                .map(|d| (self.next_bank + d) % banks)
                .find(|&b| self.phase[b] == Phase::Growing)
                .unwrap();
            let half = self.size / 2;
            // A complete half-layer precedes use of its newly changed values
            // as controls. Consecutive gates write different targets.
            let c = self.cursors[b];
            let side = (c / half) % 2;
            let offset = c % half;
            let base = self.np + b * self.size;
            let t = base + side * half + offset;
            let ctrlbase = base + (1 - side) * half;
            let a = rng.random_range(0..half);
            let mut d = rng.random_range(0..half - 1);
            if d >= a {
                d += 1;
            }
            // Same random draw and same gate count as local growth. The ring
            // connects banks algebraically while only reference-free targets change.
            let x = if self.cross_bank_growth {
                self.np + ((b + 1) % banks) * self.size + (1 - side) * half + a
            } else {
                ctrlbase + a
            };
            let y = ctrlbase + d;
            assert_eq!(self.refs[t - self.np], 0);
            out.push(
                XGate::conj(
                    t as u16,
                    [
                        (x as u16, rng.random_bool(0.5)),
                        (y as u16, rng.random_bool(0.5)),
                    ],
                )
                .unwrap(),
            );
            self.cursors[b] += 1;
            self.updates[b] += 1;
            if self.cursors[b].is_multiple_of(self.size * self.sweeps_per_visit) {
                self.next_bank = (b + 1) % banks;
            } else {
                self.next_bank = b;
            }
        }
        out
    }

    pub fn rotate(&mut self, fire: usize, interval: usize, maturity: usize) {
        if self.wire.is_some() {
            self.rotate_wire_mode(fire, interval, maturity);
        } else {
            self.rotate_bank_mode(fire, interval, maturity);
        }
        self.refresh_reserve();
    }

    fn rotate_bank_mode(&mut self, fire: usize, interval: usize, maturity: usize) {
        let banks = self.phase.len();
        for b in 0..banks {
            if self.phase[b] == Phase::Retiring
                && self.refs[b * self.size..(b + 1) * self.size]
                    .iter()
                    .all(|&n| n == 0)
            {
                self.phase[b] = Phase::Growing;
                self.updates[b] = 0;
                self.cursors[b] = 0;
                self.retired += 1;
                self.retirement_waits
                    .push(fire - self.retire_since[b].take().unwrap());
            }
        }
        if interval == 0 || !fire.is_multiple_of(interval) {
            return;
        }
        if self.phase.iter().filter(|&&p| p == Phase::Retiring).count() >= self.max_retiring {
            self.promotion_stalls += 1;
            return;
        }
        if let Some(new) = (0..banks)
            .filter(|&b| self.phase[b] == Phase::Growing && self.updates[b] >= maturity * self.size)
            .max_by_key(|&b| self.updates[b])
        {
            let old = (0..banks)
                .filter(|&b| self.phase[b] == Phase::Serving)
                .min_by_key(|&b| self.service_age[b])
                .unwrap();
            self.phase[old] = Phase::Retiring;
            self.phase[new] = Phase::Serving;
            self.service_age[new] = fire + banks / 2;
            self.retire_since[old] = Some(fire);
            self.rotations.push(Rotation {
                fire,
                promoted: new,
                retiring: old,
            });
        }
    }

    pub fn audit(&self, masks: &[Vec<Mask>]) {
        let mut refs = vec![0; self.refs.len()];
        for ms in masks {
            assert!((1..=3).contains(&ms.len()));
            for m in ms {
                for &w in m {
                    refs[w as usize - self.np] += 1;
                }
            }
        }
        assert_eq!(refs, self.refs);
        for (i, &n) in refs.iter().enumerate() {
            if self.phase_of((self.np + i) as u16) == Phase::Growing {
                assert_eq!(n, 0);
            }
        }
        if let Some(ws) = &self.wire {
            for (w, c) in ws.cohort_of.iter().enumerate() {
                match c {
                    None => assert_eq!(self.phase[w], Phase::Growing),
                    Some(c) => {
                        let cohort = &ws.cohorts[*c];
                        assert!(cohort.wires.contains(&w));
                        assert_eq!(
                            self.phase[w],
                            if cohort.phase == CohortPhase::Serving {
                                Phase::Serving
                            } else {
                                Phase::Retiring
                            }
                        );
                    }
                }
            }
        }
    }
}
