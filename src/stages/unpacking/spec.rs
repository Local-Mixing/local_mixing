//! From a decoded deliverable window to the synthesizer's input, and back.
//!
//! The synthesizer works over a LOCAL variable universe: the window's
//! support and changed wires (ascending global order) plus the lowest
//! `pool` state wires outside them, which it may borrow as dirty ancillas
//! for helper macros (`anf_synth` picked the lowest-numbered clean wires of
//! the whole state; the pool reproduces that choice as long as a macro
//! needs no more than `pool` outside wires). Local ids follow ascending
//! global order, so every tie-break key the synthesizer sorts on is
//! order-preserved. The monomial width is chosen from the universe size
//! (`packing::mono::mono_words`).
use super::mono::Mono;
use super::synth::{SGate, WindowSpec};
use crate::circuit::xgate::{Lits, XGate};
use crate::stages::packing::serialize::WindowView;
use smallvec::SmallVec;

/// Local id -> global wire, ascending.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalUniverse {
    pub wires: Vec<u16>,
}

impl LocalUniverse {
    /// Support and changed wires of `view` plus the `pool` lowest other
    /// wires of a `state_wires`-wide state.
    pub fn for_window(view: &WindowView<'_>, state_wires: usize, pool: usize) -> LocalUniverse {
        let mut base: Vec<u16> = view.support.to_vec();
        base.extend(view.changed().map(|(w, _)| w));
        base.sort_unstable();
        base.dedup();
        let mut wires = base.clone();
        let mut extra = 0usize;
        let mut it = base.iter().peekable();
        for w in 0..state_wires {
            if extra >= pool {
                break;
            }
            let w = w as u16;
            if it.peek().is_some_and(|&&b| b == w) {
                it.next();
                continue;
            }
            wires.push(w);
            extra += 1;
        }
        wires.sort_unstable();
        LocalUniverse { wires }
    }

    pub fn len(&self) -> usize {
        self.wires.len()
    }

    pub fn is_empty(&self) -> bool {
        self.wires.is_empty()
    }

    #[inline]
    pub fn global(&self, local: u32) -> u16 {
        self.wires[local as usize]
    }

    /// Local id of a global wire (it must be in the universe).
    #[inline]
    pub fn local(&self, global: u16) -> u32 {
        self.wires
            .binary_search(&global)
            .expect("wire in the local universe") as u32
    }

    /// Synthesized gates mapped back to global wires.
    pub fn to_xgates(&self, gates: &[SGate]) -> Vec<XGate> {
        gates
            .iter()
            .map(|g| {
                let mut ctrls: Lits = SmallVec::new();
                for i in 0..g.k as usize {
                    ctrls.push((self.global(g.c[i]), g.p[i]));
                }
                crate::circuit::xgate::sort_lits(&mut ctrls);
                XGate {
                    target: self.global(g.t),
                    comp: g.comp,
                    ctrls,
                }
            })
            .collect()
    }
}

/// The spec of `view` over `uni` (its monomials are masks over the window's
/// support indices).
pub fn build_spec<const W: usize>(view: &WindowView<'_>, uni: &LocalUniverse) -> WindowSpec<W> {
    let sup_local: Vec<u32> = view.support.iter().map(|&w| uni.local(w)).collect();
    let fb = view.mask_bytes;
    let mut polys = Vec::with_capacity(view.changed_len());
    for (w, masks) in view.changed() {
        let mut p: Vec<Mono<W>> = Vec::with_capacity(masks.len() / fb);
        for m in masks.chunks_exact(fb) {
            let mut mono = Mono::<W>::ONE;
            for (bi, &byte) in m.iter().enumerate() {
                let mut b = byte;
                while b != 0 {
                    let l = bi * 8 + b.trailing_zeros() as usize;
                    b &= b - 1;
                    mono = mono.with(sup_local[l]);
                }
            }
            p.push(mono);
        }
        polys.push((uni.local(w), p));
    }
    WindowSpec {
        num_wires: uni.len() as u32,
        polys,
    }
}

/// The `windows.tsv` ANF columns of a window.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AnfStats {
    /// Wires whose exit polynomial differs from `x_w`.
    pub changed_wires: usize,
    /// Distinct variables over all monomials (the support size).
    pub input_vars: usize,
    /// Monomials over all changed wires (the constant counts as one).
    pub monomials: usize,
    pub max_degree: usize,
}

pub fn anf_stats(view: &WindowView<'_>) -> AnfStats {
    let fb = view.mask_bytes;
    let mut s = AnfStats {
        changed_wires: view.changed_len(),
        input_vars: view.support.len(),
        ..Default::default()
    };
    for (_, masks) in view.changed() {
        for m in masks.chunks_exact(fb) {
            s.monomials += 1;
            let d: u32 = m.iter().map(|b| b.count_ones()).sum();
            s.max_degree = s.max_degree.max(d as usize);
        }
    }
    s
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/unpacking/spec.rs"]
mod tests;
