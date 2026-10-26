//! The every-cut direct/affine exposure census of a circuit against the
//! source C, with the fresh-input audit of every interval-entry witness
//! (port of the server prototype's `proxy_intervals.rs` and
//! `audit_port.py`).
//!
//! Discovery: the circuit is simulated on the honest slice (`HonestSamples`,
//! 64 inputs per batch) while a dynamic GF(2) basis of the live wire values
//! plus the constant 1 is maintained incrementally. Linear gates (NOT, CNOT
//! of either polarity) update only basis coordinates; a nonlinear gate
//! removes its target from the basis and re-inserts the new value. A birth
//! `c_j(x)` of C is AFFINELY exposed at a cut when it lies in the span; its
//! coordinates at the entry cut give the interval's witness (`support` =
//! the locations of the wires with a nonzero coefficient, `constant` = the
//! coefficient of 1). A birth is DIRECTLY exposed on a wire when the wire's
//! value equals `c_j(x)` or its complement on every sample; one direct
//! interval per (birth, location), spanning the location's lifetime.
//!
//! The affine family is "all affine combinations of simultaneously live
//! physical wires"; exposure is exact on the discovery inputs, not a
//! Boolean proof. The audit re-tests every DISTINCT entry witness
//! (`source_gate, constant, support`) on [`AUDIT_RANDOM_INPUTS`] fresh
//! random inputs plus the structured inputs of the prototype (0, all ones,
//! unit vectors and their complements, all pairs, and per C gate eight
//! control patterns pulled back through the C prefix). A witness that fails
//! anywhere marks its intervals `validated = false`. Interval membership
//! between entries is the discovery census's (the coefficients may change
//! along an interval; only the entry witness is audited).
//!
//! Library code: no environment reads, nothing printed.
use super::types::{
    AuditSummary, Census, FamilyMetrics, Interval, LeakageMetrics, Location, Source,
};
use crate::circuit::xgate::{XGate, max_wire};
use crate::stages::packing::cview::{HonestSamples, born_rows};
use crate::stages::packing::gf2::{bit, first_set, xor_into};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

/// Discovery batches of the prototype (6144 inputs).
pub const DEFAULT_BATCHES: usize = 96;
/// The prototype's `HonestSamples` seed.
pub const DISCOVERY_SEED: u64 = 20_261_016;
/// Fresh random inputs of the audit.
pub const AUDIT_RANDOM_INPUTS: usize = 8192;
/// Seed of the audit's fresh inputs (the prototype's Python seed; the
/// generator here is `StdRng`, so the inputs differ from the prototype's
/// while staying fixed across runs).
pub const AUDIT_SEED: u64 = 202_610_190_913;
/// Peak bytes of location values held by one audit replay.
const AUDIT_CHUNK_BYTES: usize = 256 << 20;

const INTERVAL_HEADER: &str =
    "source_gate\tstart_cut\tend_cut\tconstant\tsupport\tis_output\tvalidated";

#[inline]
fn is_zero(v: &[u64]) -> bool {
    v.iter().all(|&x| x == 0)
}

/// The census: discovery on `batches` honest batches, then the audit.
/// `width` is the circuit's wire count (`>= source.n`, above every wire the
/// circuit uses). Requires `64 * batches > width + 128` so that the sample
/// space exceeds the number of wire columns (the prototype's guard).
pub fn census(source: &Source, circuit: &[XGate], width: usize, batches: usize) -> Census {
    let started = Instant::now();
    let mut c = discover(source, circuit, width, batches);
    c.audit = audit(source, circuit, width, &mut c.affine, &mut c.direct);
    c.seconds = started.elapsed().as_secs_f64();
    c
}

// ---------------------------------------------------------------------------
// discovery
// ---------------------------------------------------------------------------

struct Row {
    v: Vec<u64>,
    c: Vec<u64>,
    p: usize,
}

/// Dynamic GF(2) basis of the wire values (`w`, the last entry being the
/// constant 1) with coefficient tracking, and every target's residual `r`
/// and coefficients `c` against it. Invariants: every row's `v` is the XOR
/// of the wires in its `c`, which names basic wires only; pivot bits are set
/// in exactly one row and cleared in every residual, so a residual is zero
/// iff the target lies in the span.
struct DynamicBasis {
    w: Vec<Vec<u64>>,
    basic: Vec<bool>,
    rows: Vec<Row>,
    r: Vec<Vec<u64>>,
    c: Vec<Vec<u64>>,
    coef_words: usize,
    visible: Vec<bool>,
    /// Targets whose visibility toggled since the caller last drained this.
    touched: Vec<usize>,
}

impl DynamicBasis {
    fn new(w: Vec<Vec<u64>>, targets: Vec<Vec<u64>>) -> Self {
        let nw = w.len();
        let nr = targets.len();
        let coef_words = nw.div_ceil(64);
        let mut s = Self {
            w,
            basic: vec![false; nw],
            rows: Vec::new(),
            r: targets,
            c: vec![vec![0; coef_words]; nr],
            coef_words,
            visible: vec![false; nr],
            touched: Vec::new(),
        };
        for i in 0..nr {
            if is_zero(&s.r[i]) {
                s.visible[i] = true;
                s.touched.push(i);
            }
        }
        for w in 0..nw {
            s.insert(w);
        }
        s
    }

    fn insert(&mut self, w: usize) {
        debug_assert!(!self.basic[w]);
        let mut v = self.w[w].clone();
        let mut c = vec![0u64; self.coef_words];
        c[w / 64] ^= 1 << (w % 64);
        for row in &self.rows {
            if bit(&v, row.p) {
                xor_into(&mut v, &row.v);
                xor_into(&mut c, &row.c);
            }
        }
        let Some(p) = first_set(&v) else {
            return;
        };
        for row in &mut self.rows {
            if bit(&row.v, p) {
                xor_into(&mut row.v, &v);
                xor_into(&mut row.c, &c);
            }
        }
        for i in 0..self.r.len() {
            if bit(&self.r[i], p) {
                xor_into(&mut self.r[i], &v);
                xor_into(&mut self.c[i], &c);
                if is_zero(&self.r[i]) && !self.visible[i] {
                    self.visible[i] = true;
                    self.touched.push(i);
                }
            }
        }
        self.rows.push(Row { v, c, p });
        self.basic[w] = true;
    }

    fn remove(&mut self, w: usize) {
        if !self.basic[w] {
            return;
        }
        // A non-basic wire whose expression over the basic wires uses `w`
        // can replace it without changing the span.
        let replacement = (0..self.w.len()).find(|&i| {
            i != w
                && !self.basic[i]
                && self
                    .rows
                    .iter()
                    .fold(false, |a, r| a ^ (bit(&r.c, w) && bit(&self.w[i], r.p)))
        });
        if let Some(i) = replacement {
            let mut d = vec![0u64; self.coef_words];
            d[i / 64] ^= 1 << (i % 64);
            for row in &self.rows {
                if bit(&self.w[i], row.p) {
                    xor_into(&mut d, &row.c);
                }
            }
            debug_assert!(bit(&d, w));
            for row in &mut self.rows {
                if bit(&row.c, w) {
                    xor_into(&mut row.c, &d);
                }
            }
            for c in &mut self.c {
                if bit(c, w) {
                    xor_into(c, &d);
                }
            }
            self.basic[w] = false;
            self.basic[i] = true;
            return;
        }
        // Otherwise the span shrinks: drop the one row that uses `w`.
        let j = self
            .rows
            .iter()
            .position(|r| bit(&r.c, w))
            .expect("a basic wire appears in some row");
        let row = self.rows.swap_remove(j);
        for r in &mut self.rows {
            if bit(&r.c, w) {
                xor_into(&mut r.v, &row.v);
                xor_into(&mut r.c, &row.c);
            }
        }
        for i in 0..self.r.len() {
            if bit(&self.c[i], w) {
                xor_into(&mut self.r[i], &row.v);
                xor_into(&mut self.c[i], &row.c);
                if self.visible[i] {
                    self.visible[i] = false;
                    self.touched.push(i);
                }
            }
        }
        self.basic[w] = false;
    }

    /// `w[t] ^= w[c]` (a linear gate): the span is unchanged, coordinates
    /// that mention `t` are rewritten.
    fn linear_add(&mut self, t: usize, c: usize) {
        debug_assert_ne!(t, c);
        if self.basic[t] {
            let mut delta = vec![0u64; self.coef_words];
            if self.basic[c] {
                delta[c / 64] ^= 1 << (c % 64);
            } else {
                for r in &self.rows {
                    if bit(&self.w[c], r.p) {
                        xor_into(&mut delta, &r.c);
                    }
                }
            }
            if bit(&delta, t) {
                delta[c / 64] ^= 1 << (c % 64);
                self.basic[t] = false;
                self.basic[c] = true;
            }
            for r in &mut self.rows {
                if bit(&r.c, t) {
                    xor_into(&mut r.c, &delta);
                }
            }
            for coef in &mut self.c {
                if bit(coef, t) {
                    xor_into(coef, &delta);
                }
            }
        }
        let (lo, hi) = (t.min(c), t.max(c));
        let (a, b) = self.w.split_at_mut(hi);
        if t < c {
            xor_into(&mut a[lo], &b[0]);
        } else {
            xor_into(&mut b[0], &a[lo]);
        }
    }

    fn gate(&mut self, g: &XGate) {
        let t = g.target as usize;
        let one = self.w.len() - 1;
        match g.ctrls.len() {
            0 => {
                if !g.comp {
                    self.linear_add(t, one);
                }
            }
            1 => {
                let (c, pos) = g.ctrls[0];
                self.linear_add(t, c as usize);
                if (!pos) ^ g.comp {
                    self.linear_add(t, one);
                }
            }
            _ => {
                self.remove(t);
                let mut acc = vec![!0u64; self.w[t].len()];
                for &(w, pos) in &g.ctrls {
                    let m = (pos as u64).wrapping_sub(1);
                    for (a, &x) in acc.iter_mut().zip(&self.w[w as usize]) {
                        *a &= x ^ m;
                    }
                }
                let comp = 0u64.wrapping_sub(g.comp as u64);
                for (d, a) in self.w[t].iter_mut().zip(acc) {
                    *d ^= a ^ comp;
                }
                self.insert(t);
            }
        }
    }
}

#[inline]
fn direct_key(first: u64) -> u64 {
    if first & 1 == 1 { !first } else { first }
}

fn direct_index(targets: &[Vec<u64>]) -> FxHashMap<u64, Vec<usize>> {
    let mut index: FxHashMap<u64, Vec<usize>> = FxHashMap::default();
    for (i, v) in targets.iter().enumerate() {
        index.entry(direct_key(v[0])).or_default().push(i);
    }
    index
}

/// Targets equal to `v` (`constant = false`) or its complement (`true`).
fn match_direct(
    v: &[u64],
    targets: &[Vec<u64>],
    index: &FxHashMap<u64, Vec<usize>>,
) -> Vec<(usize, bool)> {
    let mut found = Vec::new();
    if let Some(ids) = index.get(&direct_key(v[0])) {
        for &i in ids {
            let flip = (v[0] ^ targets[i][0]) & 1 == 1;
            let mask = 0u64.wrapping_sub(flip as u64);
            if v.iter().zip(&targets[i]).all(|(a, b)| a ^ b == mask) {
                found.push((i, flip));
            }
        }
    }
    found
}

/// Discovery only: every interval with `validated = false`, no audit.
pub fn discover(source: &Source, circuit: &[XGate], width: usize, batches: usize) -> Census {
    let started = Instant::now();
    let n = source.n;
    let nw = width;
    let m = circuit.len();
    assert!(nw >= n, "circuit narrower than the source input width");
    assert!(
        circuit.is_empty() || (max_wire(circuit) as usize) < nw,
        "circuit uses a wire beyond width {nw}"
    );
    assert!(
        64 * batches > nw + 128,
        "need more than width + 128 discovery inputs"
    );
    let samples = HonestSamples::draw(n, batches, DISCOVERY_SEED);
    let prefixes: Vec<usize> = (1..=source.gates.len()).collect();
    let rows = born_rows(&source.gates, n, &prefixes, &samples);
    let targets: Vec<Vec<u64>> = (0..rows.len()).map(|i| rows.vec(i).to_vec()).collect();
    let nr = targets.len();
    let is_output = |ri: usize| source.outputs.contains(&rows.rows[ri].gate);

    let mut wire = vec![vec![0u64; batches]; nw + 1];
    for (b, w) in (0..batches).flat_map(|b| (0..n).map(move |w| (b, w))) {
        wire[w][b] = samples.batch(b)[w];
    }
    wire[nw].fill(!0);
    let index = direct_index(&targets);
    let mut s = DynamicBasis::new(wire, targets.clone());

    let mut live: Vec<Location> = (0..nw).collect();
    let mut birth_cut = vec![0usize; nw];
    let mut open: Vec<Option<(usize, bool, Vec<Location>)>> = vec![None; nr];
    let mut direct_live: Vec<Vec<(usize, bool)>> = (0..nw)
        .map(|w| match_direct(&s.w[w], &targets, &index))
        .collect();
    let mut affine = Vec::new();
    let mut direct = Vec::new();
    let mut ever_affine = vec![false; nr];
    let mut ever_direct = vec![false; nr];

    for cut in 0..=m {
        if cut > 0 {
            let g = &circuit[cut - 1];
            let t = g.target as usize;
            for (ri, constant) in std::mem::take(&mut direct_live[t]) {
                direct.push(Interval {
                    source_gate: rows.rows[ri].gate,
                    start_cut: birth_cut[t],
                    end_cut: cut,
                    constant,
                    support: vec![live[t]],
                    is_output: is_output(ri),
                    validated: false,
                });
                ever_direct[ri] = true;
            }
            s.gate(g);
            live[t] = nw + cut - 1;
            birth_cut[t] = cut;
            direct_live[t] = match_direct(&s.w[t], &targets, &index);
        }
        let mut changed = std::mem::take(&mut s.touched);
        changed.sort_unstable();
        changed.dedup();
        for i in changed {
            if s.visible[i] {
                if open[i].is_none() {
                    debug_assert!(is_zero(&s.r[i]));
                    let support: Vec<Location> = (0..nw)
                        .filter(|&w| bit(&s.c[i], w))
                        .map(|w| live[w])
                        .collect();
                    open[i] = Some((cut, bit(&s.c[i], nw), support));
                    ever_affine[i] = true;
                }
            } else if let Some((start, constant, support)) = open[i].take() {
                affine.push(Interval {
                    source_gate: rows.rows[i].gate,
                    start_cut: start,
                    end_cut: cut,
                    constant,
                    support,
                    is_output: is_output(i),
                    validated: false,
                });
            }
        }
    }
    for (i, slot) in open.iter_mut().enumerate() {
        if let Some((start, constant, support)) = slot.take() {
            affine.push(Interval {
                source_gate: rows.rows[i].gate,
                start_cut: start,
                end_cut: m + 1,
                constant,
                support,
                is_output: is_output(i),
                validated: false,
            });
        }
    }
    for w in 0..nw {
        for (ri, constant) in std::mem::take(&mut direct_live[w]) {
            direct.push(Interval {
                source_gate: rows.rows[ri].gate,
                start_cut: birth_cut[w],
                end_cut: m + 1,
                constant,
                support: vec![live[w]],
                is_output: is_output(ri),
                validated: false,
            });
            ever_direct[ri] = true;
        }
    }
    debug_assert!(
        (0..nr).all(|i| !ever_direct[i] || ever_affine[i]),
        "a direct proxy is an affine proxy"
    );
    Census {
        gates: m,
        width: nw,
        samples: 64 * batches,
        affine,
        direct,
        audit: AuditSummary::default(),
        seconds: started.elapsed().as_secs_f64(),
    }
}

// ---------------------------------------------------------------------------
// audit
// ---------------------------------------------------------------------------

/// The audit's inputs: `random` fresh inputs, then the structured ones, as
/// bit lanes `[wire < n][word]`.
struct AuditInputs {
    n: usize,
    words: usize,
    random: usize,
    structured: usize,
    lanes: Vec<Vec<u64>>,
}

impl AuditInputs {
    fn total(&self) -> usize {
        self.random + self.structured
    }

    /// Clear the bits beyond `random + structured`.
    fn mask_tail(&self, v: &mut [u64]) {
        let t = self.total();
        if !t.is_multiple_of(64) {
            v[t / 64] &= (1u64 << (t % 64)) - 1;
        }
    }
}

/// The structured inputs of the prototype's `inputs()`: 0, all ones, every
/// unit vector and its complement, every pair, and for every gate of C the
/// eight control patterns (0, ones, the target bit and its complement, the
/// positive controls with and without the target, all controls satisfied
/// with and without the target) pulled back through the C prefix ending at
/// that gate (gates are involutions, so the prefix is undone by applying it
/// in reverse). Deduplicated in order of first appearance.
fn structured_inputs(source: &Source) -> Vec<Vec<u64>> {
    let n = source.n;
    let limbs = n.div_ceil(64).max(1);
    let mut ones = vec![!0u64; limbs];
    if !n.is_multiple_of(64) {
        ones[n / 64] = (1u64 << (n % 64)) - 1;
    }
    let unit = |i: usize| {
        let mut v = vec![0u64; limbs];
        v[i / 64] |= 1 << (i % 64);
        v
    };
    let xor = |a: &[u64], b: &[u64]| {
        let mut v = a.to_vec();
        xor_into(&mut v, b);
        v
    };
    let mut values: Vec<Vec<u64>> = vec![vec![0u64; limbs], ones.clone()];
    for i in 0..n {
        values.push(unit(i));
        values.push(xor(&ones, &unit(i)));
        for j in 0..i {
            values.push(xor(&unit(i), &unit(j)));
        }
    }
    for (cut, g) in source.gates.iter().enumerate() {
        let t = unit(g.target as usize);
        let mut mask = vec![0u64; limbs];
        let mut pos = vec![0u64; limbs];
        for &(w, p) in &g.ctrls {
            mask[w as usize / 64] |= 1 << (w as usize % 64);
            if p {
                pos[w as usize / 64] |= 1 << (w as usize % 64);
            }
        }
        let satisfied = xor(&ones, &xor(&mask, &pos));
        let patterns = [
            vec![0u64; limbs],
            ones.clone(),
            t.clone(),
            xor(&ones, &t),
            pos.clone(),
            xor(&pos, &t),
            satisfied.clone(),
            xor(&satisfied, &t),
        ];
        for mut v in patterns {
            for h in source.gates[..=cut].iter().rev() {
                h.apply_limbs(&mut v);
            }
            values.push(v);
        }
    }
    let mut seen: FxHashSet<Vec<u64>> = FxHashSet::default();
    values.retain(|v| seen.insert(v.clone()));
    values
}

fn audit_inputs(source: &Source, random: usize) -> AuditInputs {
    let n = source.n;
    let structured = structured_inputs(source);
    let total = random + structured.len();
    let words = total.div_ceil(64).max(1);
    let mut rng = StdRng::seed_from_u64(AUDIT_SEED);
    let mut lanes = vec![vec![0u64; words]; n];
    for lane in lanes.iter_mut() {
        for word in lane.iter_mut().take(random.div_ceil(64)) {
            *word = rng.random::<u64>();
        }
        if !random.is_multiple_of(64) {
            lane[random / 64] &= (1u64 << (random % 64)) - 1;
        }
    }
    for (k, x) in structured.iter().enumerate() {
        let pos = random + k;
        for (w, lane) in lanes.iter_mut().enumerate() {
            if bit(x, w) {
                lane[pos / 64] |= 1 << (pos % 64);
            }
        }
    }
    AuditInputs {
        n,
        words,
        random,
        structured: structured.len(),
        lanes,
    }
}

/// Apply `g` to a `[wire][word]` lane state (`acc` is a `words`-long
/// scratch buffer); per word this is `XGate::apply_lanes`.
#[inline]
pub(crate) fn apply_words(g: &XGate, state: &mut [Vec<u64>], acc: &mut [u64]) {
    acc.fill(!0u64);
    for &(w, pos) in &g.ctrls {
        let m = (pos as u64).wrapping_sub(1);
        for (a, &x) in acc.iter_mut().zip(&state[w as usize]) {
            *a &= x ^ m;
        }
    }
    let c = 0u64.wrapping_sub(g.comp as u64);
    for (d, &a) in state[g.target as usize].iter_mut().zip(acc.iter()) {
        *d ^= a ^ c;
    }
}

/// The values of the `needed` locations (ascending) of `circuit` on the
/// audit inputs (zeros on wires `>= n`), in the same order.
fn replay(
    circuit: &[XGate],
    width: usize,
    inputs: &AuditInputs,
    needed: &[Location],
) -> Vec<Vec<u64>> {
    debug_assert!(needed.windows(2).all(|w| w[0] < w[1]));
    let words = inputs.words;
    let mut state: Vec<Vec<u64>> = (0..width)
        .map(|w| {
            if w < inputs.n {
                inputs.lanes[w].clone()
            } else {
                vec![0u64; words]
            }
        })
        .collect();
    let mut out = Vec::with_capacity(needed.len());
    let mut k = 0;
    while k < needed.len() && needed[k] < width {
        out.push(state[needed[k]].clone());
        k += 1;
    }
    let mut acc = vec![0u64; words];
    for (i, g) in circuit.iter().enumerate() {
        if k == needed.len() {
            break;
        }
        apply_words(g, &mut state, &mut acc);
        while k < needed.len() && needed[k] == width + i {
            out.push(state[g.target as usize].clone());
            k += 1;
        }
    }
    assert_eq!(k, needed.len(), "a location beyond the circuit");
    out
}

type Witness = (usize, bool, Vec<Location>);

fn witness(iv: &Interval) -> Witness {
    (iv.source_gate, iv.constant, iv.support.clone())
}

/// Re-test every distinct entry witness of `affine` and `direct` on the
/// audit inputs, set each interval's `validated`, and count the failed
/// distinct witnesses per family.
pub fn audit(
    source: &Source,
    circuit: &[XGate],
    width: usize,
    affine: &mut [Interval],
    direct: &mut [Interval],
) -> AuditSummary {
    let inputs = audit_inputs(source, AUDIT_RANDOM_INPUTS);
    let n = source.n;
    let mut unique: BTreeSet<Witness> = BTreeSet::new();
    for iv in affine.iter().chain(direct.iter()) {
        unique.insert(witness(iv));
    }
    let unique: Vec<Witness> = unique.into_iter().collect();
    let gids: BTreeSet<usize> = unique.iter().map(|w| w.0).collect();
    let truth_needed: Vec<Location> = gids.iter().map(|&g| n + g).collect();
    let truth_values = replay(&source.gates, n, &inputs, &truth_needed);
    let truth: FxHashMap<usize, &Vec<u64>> = gids.iter().copied().zip(&truth_values).collect();

    let mut passed: FxHashMap<&Witness, bool> = FxHashMap::default();
    let mut i = 0;
    while i < unique.len() {
        let start = i;
        let mut needed: BTreeSet<Location> = BTreeSet::new();
        while i < unique.len()
            && (i == start || needed.len() * inputs.words * 8 < AUDIT_CHUNK_BYTES)
        {
            needed.extend(unique[i].2.iter().copied());
            i += 1;
        }
        let needed: Vec<Location> = needed.into_iter().collect();
        let values = replay(circuit, width, &inputs, &needed);
        let at: FxHashMap<Location, usize> =
            needed.iter().enumerate().map(|(k, &l)| (l, k)).collect();
        for w in &unique[start..i] {
            let mut diff = truth[&w.0].clone();
            if w.1 {
                for x in diff.iter_mut() {
                    *x = !*x;
                }
            }
            for loc in &w.2 {
                xor_into(&mut diff, &values[at[loc]]);
            }
            inputs.mask_tail(&mut diff);
            passed.insert(w, is_zero(&diff));
        }
    }
    let mut failed = [FxHashSet::default(), FxHashSet::default()];
    for (fam, ivs) in [direct, affine].into_iter().enumerate() {
        for iv in ivs.iter_mut() {
            let w = witness(iv);
            iv.validated = passed[&w];
            if !iv.validated {
                failed[fam].insert(w);
            }
        }
    }
    AuditSummary {
        random_inputs: inputs.random,
        structured_inputs: inputs.structured,
        failed_direct: failed[0].len(),
        failed_affine: failed[1].len(),
    }
}

// ---------------------------------------------------------------------------
// derived quantities (`metrics`, `coverage`, `interior_sets` of the prototype)
// ---------------------------------------------------------------------------

impl Census {
    /// Per family: the internal births exposed somewhere (`validated_only`
    /// restricts to validated intervals), the deeper ones, and the union of
    /// exposure cuts over `|J| * (m + 1)`.
    pub fn metrics(&self, source: &Source, validated_only: bool) -> LeakageMetrics {
        let m = self.gates;
        let fam = |ivs: &[Interval]| {
            let mut by: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
            for iv in ivs {
                if iv.is_output
                    || !source.internal.contains(&iv.source_gate)
                    || (validated_only && !iv.validated)
                {
                    continue;
                }
                by.entry(iv.source_gate)
                    .or_default()
                    .push((iv.start_cut, iv.end_cut));
            }
            let mut union = 0usize;
            for spans in by.values_mut() {
                spans.sort_unstable();
                let mut cur: Option<(usize, usize)> = None;
                for &(s, e) in spans.iter() {
                    match cur {
                        Some((cs, ce)) if s <= ce => cur = Some((cs, ce.max(e))),
                        _ => {
                            if let Some((cs, ce)) = cur {
                                union += ce - cs;
                            }
                            cur = Some((s, e));
                        }
                    }
                }
                if let Some((cs, ce)) = cur {
                    union += ce - cs;
                }
            }
            let ids: BTreeSet<usize> = by.keys().copied().collect();
            FamilyMetrics {
                all_internal_births: ids.len(),
                deeper_births: ids.difference(&source.excluded).count(),
                cut_exposure_fraction: union as f64
                    / (source.internal.len() * (m + 1)).max(1) as f64,
                ids,
            }
        };
        LeakageMetrics {
            direct: fam(&self.direct),
            affine: fam(&self.affine),
        }
    }

    /// Per cut `0..=m+1`: the number of affine non-output intervals covering
    /// it (discovery intervals, all internal births).
    pub fn coverage(&self) -> Vec<i32> {
        let m = self.gates;
        let mut cov = vec![0i32; m + 2];
        for iv in &self.affine {
            if iv.is_output {
                continue;
            }
            cov[iv.start_cut] += 1;
            cov[iv.end_cut.min(m + 1)] -= 1;
        }
        let mut acc = 0i32;
        for c in cov.iter_mut() {
            acc += *c;
            *c = acc;
        }
        cov
    }

    /// Per block of `edges` (`[0] + bounds + [m]`): the deeper internal
    /// births affinely exposed at some cut STRICTLY inside the block.
    pub fn interior_sets(&self, edges: &[usize], source: &Source) -> Vec<BTreeSet<usize>> {
        assert!(edges.len() >= 2, "edges = [0] + bounds + [m]");
        let his = &edges[1..];
        let mut sets: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); his.len()];
        for iv in &self.affine {
            let gid = iv.source_gate;
            if iv.is_output || !source.internal.contains(&gid) || source.excluded.contains(&gid) {
                continue;
            }
            let (s, e) = (iv.start_cut, iv.end_cut);
            // The interval's cuts are s..e; block k's interior cuts are
            // edges[k]+1 ..= his[k]-1.
            let mut k = his.partition_point(|&hi| hi < s + 1);
            while k < his.len() && edges[k] + 2 <= e {
                // The block has an interior cut at all.
                if edges[k] + 2 <= his[k] {
                    sets[k].insert(gid);
                }
                k += 1;
            }
        }
        sets
    }
}

// ---------------------------------------------------------------------------
// files
// ---------------------------------------------------------------------------

fn write_intervals(w: &mut impl Write, ivs: &[Interval]) -> io::Result<()> {
    writeln!(w, "{INTERVAL_HEADER}")?;
    for iv in ivs {
        let support = iv
            .support
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join(";");
        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}",
            iv.source_gate,
            iv.start_cut,
            iv.end_cut,
            iv.constant as u8,
            support,
            iv.is_output,
            iv.validated
        )?;
    }
    Ok(())
}

pub(crate) fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

pub(crate) fn parse_bool(s: &str, what: &str) -> io::Result<bool> {
    match s {
        "true" | "1" => Ok(true),
        "false" | "0" => Ok(false),
        _ => Err(invalid(format!("{what}: not a boolean: {s:?}"))),
    }
}

pub(crate) fn parse_num<T: std::str::FromStr>(s: &str, what: &str) -> io::Result<T> {
    s.parse()
        .map_err(|_| invalid(format!("{what}: not a number: {s:?}")))
}

fn read_intervals(path: &Path) -> io::Result<Vec<Interval>> {
    let f = BufReader::new(fs::File::open(path)?);
    let mut out = Vec::new();
    for (k, line) in f.lines().enumerate() {
        let line = line?;
        if k == 0 {
            if line.trim_end() != INTERVAL_HEADER {
                return Err(invalid(format!("{}: unexpected header", path.display())));
            }
            continue;
        }
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 7 {
            return Err(invalid(format!(
                "{}: line {}: 7 columns expected",
                path.display(),
                k + 1
            )));
        }
        let support = f[4]
            .split(';')
            .filter(|s| !s.is_empty())
            .map(|s| parse_num(s, "support"))
            .collect::<io::Result<Vec<Location>>>()?;
        out.push(Interval {
            source_gate: parse_num(f[0], "source_gate")?,
            start_cut: parse_num(f[1], "start_cut")?,
            end_cut: parse_num(f[2], "end_cut")?,
            constant: parse_bool(f[3], "constant")?,
            support,
            is_output: parse_bool(f[5], "is_output")?,
            validated: parse_bool(f[6], "validated")?,
        });
    }
    Ok(out)
}

/// Parse `key=value` lines (blank lines and `#` comments ignored).
pub(crate) fn read_keyvals(path: &Path) -> io::Result<BTreeMap<String, String>> {
    let text = fs::read_to_string(path)?;
    let mut map = BTreeMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (k, v) = line
            .split_once('=')
            .ok_or_else(|| invalid(format!("{}: not key=value: {line:?}", path.display())))?;
        map.insert(k.trim().to_string(), v.trim().to_string());
    }
    Ok(map)
}

pub(crate) fn keyval<'a>(
    map: &'a BTreeMap<String, String>,
    key: &str,
    path: &Path,
) -> io::Result<&'a str> {
    map.get(key)
        .map(String::as_str)
        .ok_or_else(|| invalid(format!("{}: missing key {key}", path.display())))
}

/// `affine_intervals.tsv`, `direct_intervals.tsv` (the prototype's columns
/// plus `validated`), `summary.txt` and `audit.txt` (`key=value` lines).
pub fn write_dir(c: &Census, dir: &Path) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    for (name, ivs) in [("affine", &c.affine), ("direct", &c.direct)] {
        let mut w = BufWriter::new(fs::File::create(dir.join(format!("{name}_intervals.tsv")))?);
        write_intervals(&mut w, ivs)?;
        w.flush()?;
    }
    let mut s = BufWriter::new(fs::File::create(dir.join("summary.txt"))?);
    writeln!(s, "gates={}", c.gates)?;
    writeln!(s, "width={}", c.width)?;
    writeln!(s, "samples={}", c.samples)?;
    writeln!(s, "affine_intervals={}", c.affine.len())?;
    writeln!(s, "direct_intervals={}", c.direct.len())?;
    writeln!(s, "seconds={:?}", c.seconds)?;
    writeln!(s, "all_cuts=true")?;
    writeln!(s, "all_direct_locations=true")?;
    writeln!(s, "exact_boolean_proof=false")?;
    s.flush()?;
    let mut a = BufWriter::new(fs::File::create(dir.join("audit.txt"))?);
    writeln!(a, "random_inputs={}", c.audit.random_inputs)?;
    writeln!(a, "structured_inputs={}", c.audit.structured_inputs)?;
    writeln!(a, "failed_direct={}", c.audit.failed_direct)?;
    writeln!(a, "failed_affine={}", c.audit.failed_affine)?;
    writeln!(a, "all_passed={}", c.audit.all_passed())?;
    a.flush()
}

/// The exact inverse of [`write_dir`].
pub fn read_dir(dir: &Path) -> io::Result<Census> {
    let summary_path = dir.join("summary.txt");
    let summary = read_keyvals(&summary_path)?;
    let audit_path = dir.join("audit.txt");
    let audit = read_keyvals(&audit_path)?;
    let get = |k: &str| keyval(&summary, k, &summary_path);
    let geta = |k: &str| keyval(&audit, k, &audit_path);
    let affine = read_intervals(&dir.join("affine_intervals.tsv"))?;
    let direct = read_intervals(&dir.join("direct_intervals.tsv"))?;
    let c = Census {
        gates: parse_num(get("gates")?, "gates")?,
        width: parse_num(get("width")?, "width")?,
        samples: parse_num(get("samples")?, "samples")?,
        affine,
        direct,
        audit: AuditSummary {
            random_inputs: parse_num(geta("random_inputs")?, "random_inputs")?,
            structured_inputs: parse_num(geta("structured_inputs")?, "structured_inputs")?,
            failed_direct: parse_num(geta("failed_direct")?, "failed_direct")?,
            failed_affine: parse_num(geta("failed_affine")?, "failed_affine")?,
        },
        seconds: parse_num(get("seconds")?, "seconds")?,
    };
    let n_aff: usize = parse_num(get("affine_intervals")?, "affine_intervals")?;
    let n_dir: usize = parse_num(get("direct_intervals")?, "direct_intervals")?;
    if n_aff != c.affine.len() || n_dir != c.direct.len() {
        return Err(invalid(format!(
            "{}: interval counts disagree with the TSVs",
            summary_path.display()
        )));
    }
    Ok(c)
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/adaptive/census.rs"]
mod tests;
