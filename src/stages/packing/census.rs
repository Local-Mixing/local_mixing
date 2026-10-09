//! Report-only censuses (docs/STAGE4_RUST_DESIGN.md §4.14).
//!
//! [`anf_size_report`] reproduces `anf_compose.py`'s stdout exactly,
//! including its spot checks: windows with `wi % check_every == 0` that fit
//! the budget are re-evaluated on `random.Random(20261004)` 256-bit states
//! drawn in window order.
use super::anf::{AnfOverflow, MulStats, WindowAnf, window_anf};
use super::mono::{MonoWidthFn, with_mono_width};
use super::oracle_io::window_cuts;
use super::pyrand::PyRandom;
use super::types::CapSemantics;
use crate::circuit::xgate::XGate;
use rayon::prelude::*;
use std::fmt::Write as _;

/// `anf_compose.py`'s spot-check seed.
pub const AC_SEED: u64 = 20_261_004;

/// The numbers behind the report text.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AnfSizeReport {
    pub text: String,
    pub windows: usize,
    pub overflow: usize,
    pub overflow_product: usize,
    pub overflow_running_sum: usize,
    pub overflow_final: usize,
    pub total_monomials: u64,
    pub checked: usize,
    pub failed: usize,
    pub stats: MulStats,
}

/// A fitting window's numbers; its ANF is kept only for spot-checked windows.
struct Fit<const W: usize> {
    total: usize,
    f: usize,
    changed: usize,
    anf: Option<WindowAnf<W>>,
}

struct WinOut<const W: usize> {
    gate_bits: u64,
    anf: Result<Fit<W>, AnfOverflow>,
    stats: MulStats,
}

struct Report<'a> {
    gates: &'a [XGate],
    n: usize,
    cuts: &'a [usize],
    budget: usize,
    label: &'a str,
    check_every: usize,
    sem: CapSemantics,
}

/// AC `check_window`: 256 random states (4 lanes) over all n wires.
fn check_window<const W: usize>(
    gates: &[XGate],
    anf: &WindowAnf<W>,
    n: usize,
    rng: &mut PyRandom,
) -> bool {
    let xs: Vec<[u64; 4]> = (0..n).map(|_| rng.getrandbits_lanes::<4>()).collect();
    let mut st = xs.clone();
    for g in gates {
        g.apply_lanes4(&mut st);
    }
    let mut changed = vec![false; n];
    for (w, poly) in &anf.changed {
        changed[*w as usize] = true;
        let mut v = [0u64; 4];
        for m in poly {
            let mut term = [!0u64; 4];
            for b in m.bits() {
                for k in 0..4 {
                    term[k] &= xs[b][k];
                }
            }
            for k in 0..4 {
                v[k] ^= term[k];
            }
        }
        if v != st[*w as usize] {
            return false;
        }
    }
    (0..n).all(|w| changed[w] || st[w] == xs[w])
}

impl MonoWidthFn<AnfSizeReport> for Report<'_> {
    fn call<const W: usize>(self) -> AnfSizeReport {
        let nwin = self.cuts.len() - 1;
        let outs: Vec<WinOut<W>> = (0..nwin)
            .into_par_iter()
            .map(|wi| {
                let g = &self.gates[self.cuts[wi]..self.cuts[wi + 1]];
                let gate_bits = g.iter().map(|x| 8 + 1 + 4 + 9 * x.ctrls.len() as u64).sum();
                let mut stats = MulStats::default();
                let checked = wi % self.check_every == 0;
                let anf = window_anf::<W>(g, self.budget, self.sem, &mut stats).map(|a| Fit {
                    total: a.total,
                    f: a.support().len().max(1),
                    changed: a.changed.len(),
                    anf: checked.then_some(a),
                });
                WinOut {
                    gate_bits,
                    anf,
                    stats,
                }
            })
            .collect();
        let mut r = AnfSizeReport {
            windows: nwin,
            ..Default::default()
        };
        let mut rng = PyRandom::new(AC_SEED);
        let (mut tot_anf_bits, mut tot_gate_bits) = (0u64, 0u64);
        let mut per_win: Vec<usize> = Vec::new();
        for (wi, o) in outs.iter().enumerate() {
            r.stats.add(&o.stats);
            tot_gate_bits += o.gate_bits;
            match &o.anf {
                Err(kind) => {
                    r.overflow += 1;
                    match kind {
                        AnfOverflow::Product => r.overflow_product += 1,
                        AnfOverflow::RunningSum => r.overflow_running_sum += 1,
                        AnfOverflow::Final => r.overflow_final += 1,
                    }
                    tot_anf_bits += o.gate_bits;
                }
                Ok(fit) => {
                    let f = fit.f as u64;
                    let total = fit.total as u64;
                    tot_anf_bits += 8 + f * 8 + fit.changed as u64 * (8 + 16) + total * f;
                    r.total_monomials += total;
                    per_win.push(fit.total);
                    if let Some(anf) = &fit.anf {
                        r.checked += 1;
                        let (lo, hi) = (self.cuts[wi], self.cuts[wi + 1]);
                        if !check_window(&self.gates[lo..hi], anf, self.n, &mut rng) {
                            r.failed += 1;
                            let _ = writeln!(r.text, "  CHECK FAILED window {wi} [{lo},{hi})");
                        }
                    }
                }
            }
        }
        per_win.sort(); // stable sort by total, as `per_win.sort(key=lambda r: r[1])`
        let k = per_win.len();
        let _ = writeln!(
            r.text,
            "{}: {} windows, budget={} monomials; overflow {} ({:.2}%)",
            self.label,
            nwin,
            self.budget,
            r.overflow,
            100.0 * r.overflow as f64 / nwin as f64
        );
        if k > 0 {
            let _ = writeln!(
                r.text,
                "monomials/window: med={} p90={} p99={} max={}; total={}",
                per_win[k / 2],
                per_win[9 * k / 10],
                per_win[99 * k / 100],
                per_win[k - 1],
                r.total_monomials
            );
        }
        let _ = writeln!(
            r.text,
            "size: ANF {:.1} MB vs gate-list {:.1} MB (x{:.2})",
            tot_anf_bits as f64 / 8e6,
            tot_gate_bits as f64 / 8e6,
            tot_anf_bits as f64 / tot_gate_bits.max(1) as f64
        );
        let _ = writeln!(
            r.text,
            "spot checks: {} windows re-evaluated, {} failures{}",
            r.checked,
            r.failed,
            if r.failed > 0 {
                "  ** FUNCTIONAL MISMATCH **"
            } else {
                " (OK)"
            }
        );
        r
    }
}

/// `anf_compose.py C B budget label check_every` on a loaded circuit: `n` is
/// the oracle's computed width (`max(header n, maxwire + 1)`).
pub fn anf_size_report(
    gates: &[XGate],
    n: usize,
    bounds: &[i64],
    budget: usize,
    label: &str,
    check_every: usize,
    sem: CapSemantics,
) -> AnfSizeReport {
    let cuts = window_cuts(bounds, gates.len());
    with_mono_width(
        n,
        Report {
            gates,
            n,
            cuts: &cuts,
            budget,
            label,
            check_every: check_every.max(1),
            sem,
        },
    )
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/packing/census.rs"]
mod tests;
