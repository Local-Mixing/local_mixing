//! Configurable role banks; growth writes require a reference-free bank.
//!
//! Ported verbatim from the `stage2_compute_pilot` generator (`src/band.rs`).

use super::ThreeMaskParams;
use super::fire::Mask;
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

pub struct Band {
    pub np: usize,
    pub size: usize,
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
}

impl Band {
    pub fn new(np: usize, r: usize, cfg: &ThreeMaskParams) -> Self {
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
        }
    }

    pub fn bank(&self, w: u16) -> usize {
        (w as usize - self.np) / self.size
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
            .filter(|&w| self.phase[self.bank(w)] == Phase::Serving && !excluded.contains(&w))
            .collect();
        assert!(
            avail.len() >= 3,
            "insufficient serving wires; configuration/exclusion invariant violated"
        );
        avail.shuffle(rng);
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
            .filter(|&w| self.phase[self.bank(w)] == Phase::Growing)
            .take(4)
            .collect();
        assert_eq!(out.len(), 4);
        out
    }

    pub fn borrow_helpers_pool(&self, rng: &mut impl Rng) -> Vec<u16> {
        let mut out: Vec<_> = (self.np..self.np + self.refs.len())
            .map(|w| w as u16)
            .filter(|&w| match self.phase[self.bank(w)] {
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
            .filter(|&w| self.phase[self.bank(w)] == Phase::Growing)
            .collect();
        avail.shuffle(rng);
        [avail[0], avail[1]]
    }

    pub fn retiring_mask(&self, m: Mask) -> bool {
        m.iter()
            .any(|&w| self.phase[self.bank(w)] == Phase::Retiring)
    }

    pub fn oldest_retiring(&self) -> Option<usize> {
        (0..self.phase.len())
            .filter(|&b| self.phase[b] == Phase::Retiring)
            .min_by_key(|&b| self.retire_since[b].unwrap())
    }

    pub fn growth(&mut self, count: usize, rng: &mut impl Rng) -> Vec<XGate> {
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
            if self.phase[i / self.size] == Phase::Growing {
                assert_eq!(n, 0);
            }
        }
    }
}
