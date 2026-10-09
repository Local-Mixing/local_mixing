//! Stage-4 checks (docs/STAGE4_RUST_DESIGN.md §4.11a, §4.13).
//!
//! * [`verify_equiv`]: `verify_equiv.py` (1024 random all-wire states,
//!   `random.Random(77)`).
//! * [`check_deliverable`]: `anf_dump_eval.py`'s `match=` check (decoded ANF
//!   vs gates on 1024 random all-wire states, `random.Random(1)`), end to end
//!   through the decoder.
//! * [`check_windows_anf`]: every decoded window equals `window_anf` EXACTLY
//!   (blob equality), and the ANF equals the window's gates: exhaustively
//!   for windows touching at most `exhaustive_max` wires, on 1024 random
//!   states otherwise. Replaces `anf_compose.py`'s spot checks (one window in
//!   500) on the deliverable path.
//! * [`check_zero_slice`]: x on wires 0..n, every other wire 0: the
//!   deliverable leaves C(x) on wires n..2n.
use super::anf::WindowAnf;
use super::mono::{MonoWidthFn, with_mono_width};
use super::oracle_io::window_cuts;
use super::pyrand::PyRandom;
use super::serialize::{Deliverable, WindowView, emit_anf, encode_window};
use super::types::Stage4Error;
use crate::circuit::xgate::XGate;
use rayon::prelude::*;

pub type Lanes = [u64; 16];

/// `[rng.getrandbits(1024) for _ in range(n)]` as bit-sliced lanes.
pub fn random_lanes(rng: &mut PyRandom, n: usize) -> Vec<Lanes> {
    (0..n).map(|_| rng.getrandbits_lanes::<16>()).collect()
}

pub fn eval_gates_lanes16(gates: &[XGate], state: &mut [Lanes]) {
    for g in gates {
        g.apply_lanes16(state);
    }
}

fn width_of(gates: &[XGate]) -> usize {
    gates
        .iter()
        .map(|g| g.max_wire() as usize + 1)
        .max()
        .unwrap_or(0)
}

/// `verify_equiv.py A B`: run both on the same 1024 random states over `n`
/// wires (`n` = max of the two header widths in the oracle).
pub fn verify_equiv(a: &[XGate], b: &[XGate], n: usize, seed: u64) -> Result<(), Stage4Error> {
    let n = n.max(width_of(a)).max(width_of(b));
    let mut rng = PyRandom::new(seed);
    let x = random_lanes(&mut rng, n);
    let mut sa = x.clone();
    let mut sb = x;
    eval_gates_lanes16(a, &mut sa);
    eval_gates_lanes16(b, &mut sb);
    let bad = sa.iter().zip(&sb).filter(|(u, v)| u != v).count();
    if bad == 0 {
        Ok(())
    } else {
        Err(Stage4Error::Verify(format!(
            "{} vs {} gates: MISMATCH on {bad} wires (1024 random states, seed {seed})",
            a.len(),
            b.len()
        )))
    }
}

/// `anf_dump_eval.py`'s evaluation check: decoded deliverable vs the gates on
/// `random.Random(seed)` 1024-bit states over `n` wires (DE uses the computed
/// width of the circuit and seed 1).
pub fn check_deliverable(
    gates: &[XGate],
    n: usize,
    d: &Deliverable,
    seed: u64,
) -> Result<(), Stage4Error> {
    let n = n.max(width_of(gates));
    if n > d.state_wires as usize {
        return Err(Stage4Error::Verify(format!(
            "circuit uses {n} wires, the deliverable's state has {}",
            d.state_wires
        )));
    }
    let mut rng = PyRandom::new(seed);
    let x = random_lanes(&mut rng, n);
    let mut sg = x.clone();
    eval_gates_lanes16(gates, &mut sg);
    let mut sd = x;
    sd.resize(d.state_wires as usize, [0; 16]);
    d.eval_lanes16(&mut sd);
    let bad = (0..n).filter(|&w| sg[w] != sd[w]).count();
    if bad == 0 {
        Ok(())
    } else {
        Err(Stage4Error::Verify(format!(
            "decoded deliverable differs from the gates on {bad} wires (1024 random states, seed {seed})"
        )))
    }
}

/// The blob of a decoded window, rebuilt byte for byte.
pub fn view_blob(v: &WindowView<'_>) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&(v.support.len() as u16).to_le_bytes());
    b.extend_from_slice(&(v.changed_len() as u16).to_le_bytes());
    for &w in v.support {
        b.extend_from_slice(&w.to_le_bytes());
    }
    for (w, masks) in v.changed() {
        b.extend_from_slice(&w.to_le_bytes());
        b.extend_from_slice(&((masks.len() / v.mask_bytes) as u32).to_le_bytes());
        b.extend_from_slice(masks);
    }
    b
}

/// Evaluate a window ANF (global monomials) on input lanes `inp` (indexed by
/// global wire): the new values of its changed wires, in order.
fn eval_anf<const W: usize>(anf: &WindowAnf<W>, inp: &[Lanes]) -> Vec<Lanes> {
    anf.changed
        .iter()
        .map(|(_, poly)| {
            let mut acc = [0u64; 16];
            for m in poly {
                let mut term = [!0u64; 16];
                for w in m.bits() {
                    let s = &inp[w];
                    for k in 0..16 {
                        term[k] &= s[k];
                    }
                }
                for k in 0..16 {
                    acc[k] ^= term[k];
                }
            }
            acc
        })
        .collect()
}

/// Check one window's ANF against its gates on `rounds` batches of 1024
/// states over the touched wires (exhaustive enumeration when `exhaustive`).
fn check_anf_against_gates<const W: usize>(
    gates: &[XGate],
    anf: &WindowAnf<W>,
    nw: usize,
    exhaustive_max: usize,
    seed: u64,
) -> Result<bool, String> {
    let mut touched: Vec<usize> = gates
        .iter()
        .flat_map(|g| {
            g.ctrls
                .iter()
                .map(|&(w, _)| w as usize)
                .chain([g.target as usize])
        })
        .collect();
    touched.sort_unstable();
    touched.dedup();
    let exhaustive = touched.len() <= exhaustive_max;
    let chunks = if exhaustive {
        (1usize << touched.len()).div_ceil(1024)
    } else {
        1
    };
    let mut rng = PyRandom::new(seed);
    let mut inp = vec![[0u64; 16]; nw];
    for chunk in 0..chunks {
        if exhaustive {
            let f = touched.len();
            for (j, &w) in touched.iter().enumerate() {
                let mut lanes = [0u64; 16];
                for (k, lane) in lanes.iter_mut().enumerate() {
                    for bit in 0..64 {
                        // global assignment index; wraps when 2^f < 1024
                        let g = (chunk * 1024 + k * 64 + bit) & ((1usize << f) - 1);
                        *lane |= (((g >> j) & 1) as u64) << bit;
                    }
                }
                inp[w] = lanes;
            }
        } else {
            for &w in &touched {
                inp[w] = rng.getrandbits_lanes::<16>();
            }
        }
        let mut st = inp.clone();
        eval_gates_lanes16(gates, &mut st);
        let got = eval_anf(anf, &inp);
        let mut ci = 0;
        for &w in &touched {
            let want = &st[w];
            if ci < anf.changed.len() && anf.changed[ci].0 as usize == w {
                if got[ci] != *want {
                    return Err(format!("wire {w}: ANF != gates"));
                }
                ci += 1;
            } else if *want != inp[w] {
                return Err(format!("wire {w}: gates change it, the ANF does not"));
            }
        }
        if ci != anf.changed.len() {
            return Err("the ANF changes a wire no gate touches".into());
        }
    }
    Ok(exhaustive)
}

/// Result counts of [`check_windows_anf`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WindowCheck {
    pub windows: usize,
    pub exhaustive: usize,
    pub sampled: usize,
}

struct CheckWindows<'a> {
    gates: &'a [XGate],
    cuts: &'a [usize],
    d: &'a Deliverable,
    nw: usize,
    exhaustive_max: usize,
    seed: u64,
}

impl MonoWidthFn<Result<WindowCheck, Stage4Error>> for CheckWindows<'_> {
    fn call<const W: usize>(self) -> Result<WindowCheck, Stage4Error> {
        let nwin = self.cuts.len() - 1;
        if self.d.window_count() != nwin {
            return Err(Stage4Error::Verify(format!(
                "deliverable has {} windows, the lattice {nwin}",
                self.d.window_count()
            )));
        }
        let results: Vec<Result<bool, Stage4Error>> = (0..nwin)
            .into_par_iter()
            .map(|wi| {
                let gates = &self.gates[self.cuts[wi]..self.cuts[wi + 1]];
                let anf = emit_anf::<W>(gates);
                let blob = encode_window(&anf)?;
                if view_blob(&self.d.window(wi)) != blob {
                    return Err(Stage4Error::Verify(format!(
                        "window {wi} [{}, {}): decoded window != window_anf",
                        self.cuts[wi],
                        self.cuts[wi + 1]
                    )));
                }
                check_anf_against_gates::<W>(
                    gates,
                    &anf,
                    self.nw,
                    self.exhaustive_max,
                    self.seed ^ (wi as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                )
                .map_err(|m| {
                    Stage4Error::Verify(format!(
                        "window {wi} [{}, {}): {m}",
                        self.cuts[wi],
                        self.cuts[wi + 1]
                    ))
                })
            })
            .collect();
        let mut c = WindowCheck {
            windows: nwin,
            ..Default::default()
        };
        for r in results {
            if r? {
                c.exhaustive += 1;
            } else {
                c.sampled += 1;
            }
        }
        Ok(c)
    }
}

/// Exact window check (§4.11a): decoded window == `window_anf` (blob
/// equality), and the ANF == the gates (exhaustive up to `exhaustive_max`
/// touched wires, else 1024 random states).
pub fn check_windows_anf(
    gates: &[XGate],
    bounds: &[i64],
    d: &Deliverable,
    exhaustive_max: usize,
    seed: u64,
) -> Result<WindowCheck, Stage4Error> {
    let nw = (d.state_wires as usize).max(width_of(gates));
    let cuts = window_cuts(bounds, gates.len());
    with_mono_width(
        nw,
        CheckWindows {
            gates,
            cuts: &cuts,
            d,
            nw,
            exhaustive_max,
            seed,
        },
    )
}

/// The honest-slice end-to-end check: x on wires 0..n, every other wire 0;
/// the deliverable must leave C(x) on wires n..2n. `samples` is rounded up
/// to a multiple of 1024.
pub fn check_zero_slice(
    c: &[XGate],
    d: &Deliverable,
    n: usize,
    samples: usize,
    seed: u64,
) -> Result<usize, Stage4Error> {
    if 2 * n > d.state_wires as usize || width_of(c) > n {
        return Err(Stage4Error::Usage(format!(
            "zero slice: C uses {} wires, n={n}, state {}",
            width_of(c),
            d.state_wires
        )));
    }
    let mut rng = PyRandom::new(seed);
    let rounds = samples.div_ceil(1024).max(1);
    for r in 0..rounds {
        let x = random_lanes(&mut rng, n);
        let mut want = x.clone();
        eval_gates_lanes16(c, &mut want);
        let mut st = x;
        st.resize(d.state_wires as usize, [0; 16]);
        d.eval_lanes16(&mut st);
        if let Some(w) = (0..n).find(|&w| st[n + w] != want[w]) {
            return Err(Stage4Error::Verify(format!(
                "zero slice: wire {} != C(x) wire {w} (batch {r})",
                n + w
            )));
        }
    }
    Ok(rounds * 1024)
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/packing/verify.rs"]
mod tests;
