//! Sparse window ANF composition (docs/STAGE4_RUST_DESIGN.md §4.6).
//!
//! A polynomial is a set of monomials; a monomial is a bitmask over GLOBAL
//! wire ids (bit w = wire w's value at the window input); GF(2) addition is
//! symmetric difference. Two entry points reproduce the oracle:
//!
//! * [`window_anf`]: `anf_compose.window_anf` (AC:66-91), used by the
//!   serialization (`serialize::EMIT_BUDGET`), the split predicate and the
//!   size report: product cap 8·budget checked after every outer row of a
//!   product, running sum over ALL entries > 4·budget after each gate, final
//!   changed total > budget.
//! * [`WindowState`]: the packer's incremental compose (WP:324-344), product
//!   cap `mul_cap` only.
//!
//! Both run on the CPython 3.9 set emulator so that prefix aborts happen
//! exactly where the oracle's do (`CapSemantics::CPython39`).
use super::mono::Mono;
use super::pyset::PySet;
use super::types::CapSemantics;
use crate::circuit::xgate::XGate;
use rustc_hash::FxHashMap;

pub type Poly<const W: usize> = PySet<Mono<W>>;

/// Counters of the product routine (reported, never decisive).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MulStats {
    pub calls: u64,
    /// (|small| - 1)·|large| > cap: the only zone where the iteration order
    /// can decide an abort.
    pub ambiguous: u64,
    /// Products that aborted under the configured semantics.
    pub aborts: u64,
    /// Products whose decision depended on the order: the finished product
    /// fits, yet the CPython prefix exceeded the cap (an abort under
    /// `CPython39`, a non-abort under `Canonical`).
    pub aborts_order_dependent: u64,
}

impl MulStats {
    pub fn add(&mut self, o: &MulStats) {
        self.calls += o.calls;
        self.ambiguous += o.ambiguous;
        self.aborts += o.aborts;
        self.aborts_order_dependent += o.aborts_order_dependent;
    }
}

/// `mul(Ap, Bp, cap)` (AC:49-64 = WP:135-150): swap so the smaller operand is
/// outer (only when `len(a) > len(b)`), toggle `a | b` for every pair in
/// table order, and check `len(res) > cap` after each outer row.
pub fn mul<const W: usize>(
    a: &Poly<W>,
    b: &Poly<W>,
    cap: usize,
    sem: CapSemantics,
    stats: &mut MulStats,
) -> Option<Poly<W>> {
    stats.calls += 1;
    let (outer, inner) = if a.len() > b.len() { (b, a) } else { (a, b) };
    let ambiguous = outer.len().saturating_sub(1).saturating_mul(inner.len()) > cap;
    if ambiguous {
        stats.ambiguous += 1;
    }
    let inner_keys = inner.to_vec();
    let mut res: Poly<W> = PySet::new();
    let mut prefix_exceeded = false;
    for a in outer.iter() {
        for b in &inner_keys {
            res.toggle(a.or(b));
        }
        if res.len() > cap {
            prefix_exceeded = true;
            if sem == CapSemantics::CPython39 {
                if ambiguous && full_product_fits(outer, &inner_keys, cap) {
                    stats.aborts_order_dependent += 1;
                }
                stats.aborts += 1;
                return None;
            }
        }
    }
    // Canonical: the finished product decides.
    if res.len() > cap {
        stats.aborts += 1;
        return None;
    }
    if prefix_exceeded {
        stats.aborts_order_dependent += 1;
    }
    Some(res)
}

/// Whether the finished product has at most `cap` monomials (order-free; a
/// plain hash set suffices). Runs only after a CPython39 abort.
fn full_product_fits<const W: usize>(outer: &Poly<W>, inner: &[Mono<W>], cap: usize) -> bool {
    let mut res: rustc_hash::FxHashSet<Mono<W>> = Default::default();
    for a in outer.iter() {
        for b in inner {
            let v = a.or(b);
            if !res.remove(&v) {
                res.insert(v);
            }
        }
    }
    res.len() <= cap
}

/// Why a window's ANF was refused (all three are `None` in the oracle).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnfOverflow {
    /// A product exceeded 8·budget.
    Product,
    /// After a gate, the sum of all polynomial sizes exceeded 4·budget.
    RunningSum,
    /// The changed wires' total exceeded the budget.
    Final,
}

/// A window's ANF: changed wires ascending, each with its monomials ascending
/// (Python int order).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowAnf<const W: usize> {
    pub changed: Vec<(u16, Vec<Mono<W>>)>,
    pub total: usize,
}

impl<const W: usize> WindowAnf<W> {
    /// The window's support: every wire in some monomial, ascending.
    pub fn support(&self) -> Vec<u16> {
        let mut acc = Mono::<W>::ZERO;
        for (_, poly) in &self.changed {
            for m in poly {
                acc = acc.or(m);
            }
        }
        acc.bits().map(|b| b as u16).collect()
    }
}

/// One gate composed onto the window's polynomials (AC:68-84 = WP:324-341):
/// returns the new target polynomial, or `None` when a product aborts.
/// `polys` is not mutated.
#[inline]
fn compose_gate<const W: usize>(
    polys: &FxHashMap<u16, Poly<W>>,
    g: &XGate,
    cap: usize,
    sem: CapSemantics,
    stats: &mut MulStats,
) -> Option<Poly<W>> {
    let zero = || PySet::singleton(Mono::<W>::ZERO);
    let mut prod = zero();
    for &(c, positive) in &g.ctrls {
        let lit;
        let mut cp: &Poly<W> = match polys.get(&c) {
            Some(p) => p,
            None => {
                lit = PySet::singleton(Mono::bit(c as usize));
                &lit
            }
        };
        let neg;
        if !positive {
            neg = PySet::xor(cp, &zero());
            cp = &neg;
        }
        prod = mul(&prod, cp, cap, sem, stats)?;
    }
    if g.comp {
        prod.xor_assign(&zero());
    }
    let tgt_lit;
    let tgt: &Poly<W> = match polys.get(&g.target) {
        Some(p) => p,
        None => {
            tgt_lit = PySet::singleton(Mono::bit(g.target as usize));
            &tgt_lit
        }
    };
    Some(PySet::xor(tgt, &prod))
}

/// `anf_compose.window_anf(gates, lo, hi, budget)` on the window's gates.
pub fn window_anf<const W: usize>(
    gates: &[XGate],
    budget: usize,
    sem: CapSemantics,
    stats: &mut MulStats,
) -> Result<WindowAnf<W>, AnfOverflow> {
    let mut polys: FxHashMap<u16, Poly<W>> = FxHashMap::default();
    let mut running = 0usize;
    let cap = budget.saturating_mul(8);
    let sum_cap = budget.saturating_mul(4);
    for g in gates {
        let new = compose_gate(&polys, g, cap, sem, stats).ok_or(AnfOverflow::Product)?;
        let old = polys.get(&g.target).map_or(0, |p| p.len());
        running = running + new.len() - old;
        polys.insert(g.target, new);
        if running > sum_cap {
            return Err(AnfOverflow::RunningSum);
        }
    }
    let mut changed: Vec<(u16, Vec<Mono<W>>)> = Vec::new();
    let mut total = 0usize;
    for (&w, p) in &polys {
        let id = Mono::bit(w as usize);
        if p.len() == 1 && p.contains(&id) {
            continue;
        }
        let mut monos = p.to_vec();
        monos.sort_unstable();
        total += monos.len();
        changed.push((w, monos));
    }
    if total > budget {
        return Err(AnfOverflow::Final);
    }
    changed.sort_unstable_by_key(|(w, _)| *w);
    Ok(WindowAnf { changed, total })
}

/// The packer's incremental window ANF (WP compose). Monomials are over
/// global wire ids; `monos()` is the sum of all polynomial sizes, including
/// identity-valued ones (WP's `win_monos`).
pub struct WindowState<const W: usize> {
    polys: FxHashMap<u16, Poly<W>>,
    monos: usize,
    mul_cap: usize,
}

impl<const W: usize> WindowState<W> {
    pub fn new(mul_cap: usize) -> Self {
        WindowState {
            polys: FxHashMap::default(),
            monos: 0,
            mul_cap,
        }
    }

    /// Compose one gate. `false` = a product aborted; nothing was mutated.
    pub fn compose(&mut self, g: &XGate, sem: CapSemantics, stats: &mut MulStats) -> bool {
        match compose_gate(&self.polys, g, self.mul_cap, sem, stats) {
            None => false,
            Some(new) => {
                let old = self.polys.get(&g.target).map_or(0, |p| p.len());
                self.monos = self.monos + new.len() - old;
                self.polys.insert(g.target, new);
                true
            }
        }
    }

    pub fn monos(&self) -> usize {
        self.monos
    }

    pub fn reset(&mut self) {
        self.polys.clear();
        self.monos = 0;
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/packing/anf.rs"]
mod tests;
