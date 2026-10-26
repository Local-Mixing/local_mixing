//! RC's BALANCED source construction (stage 1, the `balanced` choice of
//! [`super::SourceConstruction`]): a g57 circuit in which every completed
//! internal value is read exactly twice as a control, no value — input,
//! internal or output — is read more than twice, and every gate targets a
//! wire whose write count is the current minimum (the write spread never
//! exceeds 1).
//!
//! Bit-exact port of the Python builder `balanced_c.py` (`generate`, `g57`
//! and `audit`) over CPython's `random.Random(seed)` ([`PyRandom`]): the same
//! draws in the same order, the same immediate-dead-end rejection, the same
//! 1000-attempt budget and the same error when it is exhausted, so a seed
//! reproduces the Python builder's circuit gate for gate. The campaign's
//! reference circuit (n=64, m=1000) is pinned by hash in the unit tests.

use crate::circuit::CircuitSeq;
use crate::stages::packing::pyrand::PyRandom;
use std::collections::HashSet;

/// One gate in the builder's own form: `t ^= 1 ^ (lit(a, pa) & lit(z, pz))`
/// with `a < z` and `pa + pz == 1`; `polarity` is `pa`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BuilderGate {
    t: usize,
    a: usize,
    z: usize,
    polarity: usize,
}

impl BuilderGate {
    /// The Python `g57` mapping: `t ^= x OR NOT y` with `x` the
    /// negative-polarity control and `y` the positive one, as the CircuitSeq
    /// triple `[t, x, y]` (target, positive g57 control, negative g57 control).
    fn to_g57(self) -> [u16; 3] {
        let (x, y) = if self.polarity == 0 {
            (self.a, self.z)
        } else {
            (self.z, self.a)
        };
        [self.t as u16, x as u16, y as u16]
    }
}

/// The topology census of a balanced circuit, as `audit` reports it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BalancedAudit {
    /// `n`.
    pub wires: usize,
    /// `m`, the gate count.
    pub gates: usize,
    /// Values born by a gate and later overwritten (not circuit outputs);
    /// each of them is read exactly twice.
    pub internal_births: usize,
    /// Largest number of control reads of any value (inputs and outputs
    /// included); at most 2.
    pub maximum_control_fanout: usize,
    /// Largest `max(writes) - min(writes)` over the gate sequence; at most 1.
    pub maximum_write_spread: usize,
    /// The distinct per-wire write counts at the end, ascending.
    pub final_writes: Vec<usize>,
}

/// Port of the Python `generate`: the gates of one balanced circuit, or the
/// builder's error once its 1000 attempts are exhausted.
fn generate(n: usize, count: usize, seed: u64) -> Result<Vec<BuilderGate>, String> {
    let mut rng = PyRandom::new(seed);
    for _attempt in 0..1000 {
        let mut reads = vec![0usize; n];
        let mut writes = vec![0usize; n];
        let mut gates: Vec<BuilderGate> = Vec::with_capacity(count);
        for index in 0..count {
            let minimum = writes.iter().copied().min().unwrap_or(0);
            let targets: Vec<usize> = (0..n)
                .filter(|&w| writes[w] == minimum && (writes[w] == 0 || reads[w] == 2))
                .collect();
            let mut chosen = None;
            for _ in 0..1000 {
                if targets.is_empty() {
                    break;
                }
                let t = rng.choice(&targets);
                let mut available: Vec<usize> =
                    (0..n).filter(|&w| w != t && reads[w] < 2).collect();
                if available.len() < 2 {
                    break;
                }
                let mut weights: Vec<u64> = available
                    .iter()
                    .map(|&w| {
                        if writes[w] == minimum && reads[w] == 1 {
                            100
                        } else if writes[w] == minimum {
                            20
                        } else {
                            1
                        }
                    })
                    .collect();
                let u = rng.choices_one(&available, &weights);
                let j = available
                    .iter()
                    .position(|&w| w == u)
                    .expect("u was drawn from available");
                available.remove(j);
                weights.remove(j);
                let v = rng.choices_one(&available, &weights);
                let mut rr = reads.clone();
                let mut ww = writes.clone();
                rr[u] += 1;
                rr[v] += 1;
                rr[t] = 0;
                ww[t] += 1;
                let floor = *ww.iter().min().expect("n >= 1");
                // An immediate dead end (no legal target for the next gate)
                // is rejected and the gate is redrawn; the polarity draw
                // happens only for an accepted gate.
                if index + 1 < count
                    && !(0..n).any(|w| ww[w] == floor && (ww[w] == 0 || rr[w] == 2))
                {
                    continue;
                }
                let (a, z) = if u < v { (u, v) } else { (v, u) };
                let polarity = rng.randrange(2);
                chosen = Some(BuilderGate { t, a, z, polarity });
                reads = rr;
                writes = ww;
                break;
            }
            match chosen {
                Some(gate) => gates.push(gate),
                None => break,
            }
        }
        if gates.len() == count {
            return Ok(gates);
        }
    }
    Err("constrained generation did not find a full circuit".into())
}

/// RC's balanced C: `m` g57 gates on wires `0..n` drawn from
/// `random.Random(seed)`, audited by [`audit_balanced`] before it is
/// returned (an audit failure is an internal error). Fails like the Python
/// builder when 1000 attempts do not complete a circuit (for instance
/// `n < 3`).
pub fn balanced_source(n: usize, m: usize, seed: u64) -> Result<CircuitSeq, String> {
    if n > u16::MAX as usize + 1 {
        return Err(format!("n={n} exceeds the u16 wire index range"));
    }
    let gates = generate(n, m, seed)?;
    let circuit = CircuitSeq {
        gates: gates.iter().map(|g| g.to_g57()).collect(),
    };
    audit_balanced(n, &circuit).map_err(|error| {
        format!("internal error: the balanced construction failed its own audit: {error}")
    })?;
    Ok(circuit)
}

/// Port of the Python `audit`: every value a gate overwrites (other than a
/// circuit input) was read exactly twice, every internal value (born by a
/// gate, not an output) is read exactly twice, no value is read more than
/// twice, and the write spread never exceeds 1. Value `w < n` is input wire
/// `w`; value `n + i` is the one gate `i` writes.
pub fn audit_balanced(n: usize, c: &CircuitSeq) -> Result<BalancedAudit, String> {
    if n == 0 {
        return Err("audit needs at least one wire".into());
    }
    let m = c.gates.len();
    let mut live: Vec<usize> = (0..n).collect();
    let mut consumers = vec![0usize; n + m];
    let mut writes = vec![0usize; n];
    let mut spread = 0usize;
    for (i, gate) in c.gates.iter().enumerate() {
        let [t, x, y] = gate.map(usize::from);
        if t == x || t == y || x == y {
            return Err(format!("gate {i}: wires {t} {x} {y} are not distinct"));
        }
        if t >= n || x >= n || y >= n {
            return Err(format!(
                "gate {i}: wires {t} {x} {y} are not all below n={n}"
            ));
        }
        consumers[live[x]] += 1;
        consumers[live[y]] += 1;
        let old = live[t];
        if old >= n && consumers[old] != 2 {
            return Err(format!(
                "gate {i} overwrites the value born by gate {} after {} control reads, not 2",
                old - n,
                consumers[old]
            ));
        }
        live[t] = n + i;
        writes[t] += 1;
        let (lo, hi) = (
            *writes.iter().min().expect("n >= 1"),
            *writes.iter().max().expect("n >= 1"),
        );
        spread = spread.max(hi - lo);
    }
    let output: HashSet<usize> = live.iter().copied().collect();
    let internal: Vec<usize> = (n..n + m).filter(|v| !output.contains(v)).collect();
    if let Some(&v) = internal.iter().find(|&&v| consumers[v] != 2) {
        return Err(format!(
            "internal value born by gate {} is read {} times, not 2",
            v - n,
            consumers[v]
        ));
    }
    if let Some(v) = consumers.iter().position(|&reads| reads > 2) {
        return Err(format!(
            "value {v} is read {} times (more than 2)",
            consumers[v]
        ));
    }
    if spread > 1 {
        return Err(format!("write spread {spread} exceeds 1"));
    }
    let mut final_writes: Vec<usize> = writes.clone();
    final_writes.sort_unstable();
    final_writes.dedup();
    Ok(BalancedAudit {
        wires: n,
        gates: m,
        internal_births: internal.len(),
        maximum_control_fanout: consumers.iter().copied().max().unwrap_or(0),
        maximum_write_spread: spread,
        final_writes,
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/sandwich/balanced.rs"]
mod tests;
