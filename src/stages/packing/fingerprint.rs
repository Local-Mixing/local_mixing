//! Value fingerprints and proxy spans (`window_pack3.py` 52-90;
//! docs/STAGE4_RUST_DESIGN.md §4.7).
//!
//! Every wire carries 1024 random samples, bit-sliced as 16 u64 lanes
//! (lane i = bits 64i..64i+63 of the oracle's 1024-bit int). After each gate
//! the target's value is NORMALIZED (complemented when sample 0 is set) and,
//! unless constant, keyed by `blake2b(value.to_bytes(128,'little'), 16)`,
//! exactly the oracle's key.
//!
//! * [`index_a`]: the first A gate that births each distinct value (its id
//!   `a`) and the wire it was born on.
//! * [`spans_b`]: walking B (or B in a given order), a B gate that births a
//!   known A value opens a pending span on its wire; the next write to that
//!   wire (ANY write, including a constant or equal value) closes it as
//!   `(b, e, a)`. Spans still pending at the end are "output-forced".
use super::blake2b::blake2b16_lanes;
use super::pyrand::PyRandom;
use crate::circuit::xgate::XGate;
use rustc_hash::FxHashMap;

pub type Fp = [u64; 16];

/// `[rng.getrandbits(1024) for _ in range(n)]` with `rng = Random(seed)`.
pub fn inputs(n: usize, seed: u64) -> Vec<Fp> {
    let mut rng = PyRandom::new(seed);
    (0..n).map(|_| rng.getrandbits_lanes::<16>()).collect()
}

/// `v ^ MASK if v & 1 else v`; `None` when the normalized value is 0
/// (a constant wire).
#[inline]
pub fn normalize(v: &Fp) -> Option<Fp> {
    let mut out = *v;
    if v[0] & 1 == 1 {
        for x in out.iter_mut() {
            *x = !*x;
        }
    }
    if out.iter().all(|&x| x == 0) {
        None
    } else {
        Some(out)
    }
}

/// The oracle's value key.
#[inline]
pub fn value_key(v: &Fp) -> [u8; 16] {
    blake2b16_lanes(v)
}

/// A side of `proxy_spans`.
pub struct AIndex {
    /// key -> first A gate index birthing it.
    pub index: FxHashMap<[u8; 16], u32>,
    /// A gate index -> wire, for first-birth gates only (`u16::MAX` otherwise).
    pub a_wire: Vec<u16>,
    /// A's final state.
    pub final_a: Vec<Fp>,
}

impl AIndex {
    pub fn distinct(&self) -> usize {
        self.index.len()
    }
}

/// Initial state `inputs[:width] + [0] * (n - width)`.
fn initial_state(inputs: &[Fp], width: usize) -> Vec<Fp> {
    let mut st = inputs.to_vec();
    for w in st.iter_mut().skip(width) {
        *w = [0; 16];
    }
    st
}

pub fn index_a(a: &[XGate], inputs: &[Fp], na: usize) -> AIndex {
    let mut st = initial_state(inputs, na);
    let mut index: FxHashMap<[u8; 16], u32> = FxHashMap::default();
    index.reserve(a.len() / 2);
    let mut a_wire = vec![u16::MAX; a.len()];
    for (i, g) in a.iter().enumerate() {
        g.apply_lanes16(&mut st);
        let t = g.target;
        if let Some(fp) = normalize(&st[t as usize]) {
            let d = value_key(&fp);
            if let std::collections::hash_map::Entry::Vacant(e) = index.entry(d) {
                e.insert(i as u32);
                a_wire[i] = t;
            }
        }
    }
    AIndex {
        index,
        a_wire,
        final_a: st,
    }
}

/// B side of `proxy_spans`.
pub struct Spans {
    /// `(b, e, a)` in closing order (e ascending); positions are in the
    /// walked order.
    pub spans: Vec<(u32, u32, u32)>,
    /// A ids still pending at the end, ascending (a set in the oracle; only
    /// membership and counts are used).
    pub forced: Vec<u32>,
    /// Wires `w < min(na, nb)` whose final value equals A's.
    pub agree: usize,
}

impl Spans {
    pub fn is_forced(&self, a: u32) -> bool {
        self.forced.binary_search(&a).is_ok()
    }
}

/// Walk `order` (indices into `b`; `None` = B's own order).
pub fn spans_b(
    b: &[XGate],
    order: Option<&[u32]>,
    inputs: &[Fp],
    nb: usize,
    na: usize,
    ai: &AIndex,
) -> Spans {
    let mut st = initial_state(inputs, nb);
    let mut pending: Vec<Option<(u32, u32)>> = vec![None; inputs.len()];
    let mut spans = Vec::new();
    let m = order.map_or(b.len(), |o| o.len());
    for i in 0..m {
        let g = match order {
            None => &b[i],
            Some(o) => &b[o[i] as usize],
        };
        g.apply_lanes16(&mut st);
        let t = g.target as usize;
        if let Some((pb, pa)) = pending[t].take() {
            spans.push((pb, i as u32, pa));
        }
        if let Some(fp) = normalize(&st[t])
            && let Some(&a_id) = ai.index.get(&value_key(&fp))
        {
            pending[t] = Some((i as u32, a_id));
        }
    }
    let mut forced: Vec<u32> = pending.iter().flatten().map(|&(_, a)| a).collect();
    forced.sort_unstable();
    forced.dedup();
    let agree = (0..na.min(nb)).filter(|&w| st[w] == ai.final_a[w]).count();
    Spans {
        spans,
        forced,
        agree,
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/packing/fingerprint.rs"]
mod tests;
