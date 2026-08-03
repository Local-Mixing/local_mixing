//! Polynomial and canonical-key support for heterogeneous [`XGate`] circuits.
//!
//! `CircuitSeq::to_polynomial` is intentionally specialized to legacy g57
//! triples.  Post-mix tapes contain arbitrary-width conjunction gates and
//! complemented conjunction gates, so their firing polynomial is instead
//!
//! ```text
//! comp XOR PRODUCT(literal),
//! literal(w, true)  = P[w],
//! literal(w, false) = 1 XOR P[w].
//! ```
//!
//! The canonicalization entry point below first remaps the wires touched by a
//! window to a dense space.  Consequently its output is byte-compatible with
//! the canonical polynomial keys produced by the legacy g57 path.

use std::sync::atomic::{AtomicU64, Ordering};
use super::xgate::XGate;
use crate::circuit::circuit::{
    Monomial, Permutation, Polynomial, canonicalize_polys_4, polynomial_from_terms,
};

/// Work limits for polynomial composition.
///
/// Wide mixed-polarity cubes can expand much faster than legacy two-control
/// g57 gates.  Hitting a limit returns [`XPolyError::BudgetExceeded`], allowing
/// a database-compression caller to skip that window without changing it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct XPolyBudget {
    /// Maximum raw monomial-pair products in any one polynomial multiply.
    pub max_mul_terms: usize,
    /// Maximum reduced monomials in any individual wire polynomial or
    /// intermediate literal product.
    pub max_poly_terms: usize,
    /// Maximum reduced monomials across all live wire polynomials.
    pub max_total_terms: usize,
}

impl Default for XPolyBudget {
    fn default() -> Self {
        Self {
            max_mul_terms: 1 << 20,
            max_poly_terms: 1 << 18,
            max_total_terms: 1 << 20,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum XPolyError {
    /// Monomials are `u64` variable masks, so a dense window may touch at most
    /// 64 wires.
    TooManyWires {
        wires: usize,
    },
    WireOutOfRange {
        wire: u16,
        num_wires: usize,
    },
    /// Valid XGates never read their own target.  Besides violating the type's
    /// invariant, such a gate need not be an involution, so reverse-key lookup
    /// would be unsound.
    TargetInControls {
        target: u16,
    },
    BudgetExceeded {
        stage: &'static str,
        attempted: usize,
        limit: usize,
    },
    CanonicalizationFailed,
    /// The window's exact ANF degree exceeds the store's maximum, so no stored
    /// circuit can compute this function. Read off the polynomial rather than
    /// probed: exact, and free, because canonicalization needs that polynomial
    /// anyway.
    DegreeExceeded {
        degree: usize,
        limit: usize,
    },
}

/// Canonical polynomials and the maps needed to return a frozen-store friend
/// to the original circuit's wire space.
#[derive(Clone, Debug)]
pub struct CanonicalXPolys {
    pub polys: Vec<Polynomial>,
    /// `order.data[canonical_wire] = dense_window_wire`.
    pub order: Permutation,
    /// `used_wires[dense_window_wire] = original_wire`.
    pub used_wires: Vec<u16>,
}

fn budget_error(stage: &'static str, attempted: usize, limit: usize) -> XPolyError {
    XPolyError::BudgetExceeded {
        stage,
        attempted,
        limit,
    }
}

fn check_poly_len(
    poly: &Polynomial,
    stage: &'static str,
    budget: XPolyBudget,
) -> Result<(), XPolyError> {
    if poly.len() > budget.max_poly_terms {
        Err(budget_error(stage, poly.len(), budget.max_poly_terms))
    } else {
        Ok(())
    }
}

fn check_total(polys: &[Polynomial], budget: XPolyBudget) -> Result<(), XPolyError> {
    let mut total = 0usize;
    for p in polys {
        total = total
            .checked_add(p.len())
            .ok_or_else(|| budget_error("total terms", usize::MAX, budget.max_total_terms))?;
        if total > budget.max_total_terms {
            return Err(budget_error("total terms", total, budget.max_total_terms));
        }
    }
    Ok(())
}

fn toggle_monomial(poly: &mut Polynomial, monomial: Monomial) {
    match poly.binary_search(&monomial) {
        Ok(i) => {
            poly.remove(i);
        }
        Err(i) => poly.insert(i, monomial),
    }
}

fn poly_not(poly: &Polynomial) -> Polynomial {
    let mut out = poly.clone();
    toggle_monomial(&mut out, 0);
    out
}

/// XOR two normalized polynomials while retaining sorted, duplicate-free form.
fn poly_xor_assign(left: &mut Polynomial, right: Polynomial) {
    let old = std::mem::take(left);
    let mut out = Vec::with_capacity(old.len().max(right.len()));
    let (mut i, mut j) = (0usize, 0usize);
    while i < old.len() && j < right.len() {
        match old[i].cmp(&right[j]) {
            std::cmp::Ordering::Less => {
                out.push(old[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(right[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&old[i..]);
    out.extend_from_slice(&right[j..]);
    *left = out;
}

fn poly_mul(
    left: &Polynomial,
    right: &Polynomial,
    budget: XPolyBudget,
) -> Result<Polynomial, XPolyError> {
    let attempted = left
        .len()
        .checked_mul(right.len())
        .ok_or_else(|| budget_error("polynomial multiply", usize::MAX, budget.max_mul_terms))?;
    if attempted > budget.max_mul_terms {
        return Err(budget_error(
            "polynomial multiply",
            attempted,
            budget.max_mul_terms,
        ));
    }
    let mut terms = Vec::with_capacity(attempted);
    for &a in left {
        for &b in right {
            // Boolean variables are idempotent, so monomial multiplication is
            // set union rather than integer addition of exponents.
            terms.push(a | b);
        }
    }
    let out = polynomial_from_terms(terms);
    check_poly_len(&out, "polynomial multiply result", budget)?;
    Ok(out)
}

/// Compose a heterogeneous gate sequence into per-wire output polynomials.
///
/// `num_wires` is the polynomial variable count for this sequence.  Callers
/// working in a large global circuit should dense-remap a small window first;
/// [`canonicalize_xgates_single`] does that automatically.
pub fn xgates_to_polynomial(
    gates: &[XGate],
    num_wires: usize,
    budget: XPolyBudget,
) -> Result<Vec<Polynomial>, XPolyError> {
    if num_wires > 64 {
        return Err(XPolyError::TooManyWires { wires: num_wires });
    }
    let mut polys: Vec<Polynomial> = (0..num_wires).map(|w| vec![1u64 << w]).collect();
    check_total(&polys, budget)?;

    for gate in gates {
        let target = gate.target as usize;
        if target >= num_wires {
            return Err(XPolyError::WireOutOfRange {
                wire: gate.target,
                num_wires,
            });
        }
        let mut product: Polynomial = vec![0]; // constant one
        for &(wire, positive) in &gate.ctrls {
            if wire == gate.target {
                return Err(XPolyError::TargetInControls {
                    target: gate.target,
                });
            }
            let wire_idx = wire as usize;
            if wire_idx >= num_wires {
                return Err(XPolyError::WireOutOfRange { wire, num_wires });
            }
            let literal = if positive {
                polys[wire_idx].clone()
            } else {
                poly_not(&polys[wire_idx])
            };
            product = poly_mul(&product, &literal, budget)?;
        }

        // P[target] ^= PRODUCT(literals) ^ comp.
        poly_xor_assign(&mut polys[target], product);
        if gate.comp {
            toggle_monomial(&mut polys[target], 0);
        }
        check_poly_len(&polys[target], "target polynomial", budget)?;
        check_total(&polys, budget)?;
    }
    Ok(polys)
}

/// Sorted set of every target and control wire touched by `gates`.
pub fn xgate_used_wires(gates: &[XGate]) -> Vec<u16> {
    let mut used = Vec::new();
    for gate in gates {
        used.push(gate.target);
        used.extend(gate.ctrls.iter().map(|&(wire, _)| wire));
    }
    used.sort_unstable();
    used.dedup();
    used
}

fn dense_remap(gates: &[XGate], used: &[u16]) -> Vec<XGate> {
    gates
        .iter()
        .map(|gate| {
            let map =
                |wire: u16| used.binary_search(&wire).expect("wire came from used set") as u16;
            XGate {
                target: map(gate.target),
                comp: gate.comp,
                ctrls: gate
                    .ctrls
                    .iter()
                    .map(|&(wire, polarity)| (map(wire), polarity))
                    .collect(),
            }
        })
        .collect()
}

/// Produce frozen-store-compatible canonical polynomials for one direction of
/// an XGate window.
///
/// With `reversed=true`, gate order is reversed before composition.  XGates
/// satisfying their invariant (the target is absent from the controls) are
/// involutions, so this is the inverse-circuit direction used by the legacy
/// compressor.
/// Stage timers for the canonicalization path (see the note inside
/// `canonicalize_xgates_single`). Relaxed atomics: this is instrumentation,
/// so a lost update is cheaper than a fence on the hot path.
pub static POLY_NS: AtomicU64 = AtomicU64::new(0);
pub static CANON_NS: AtomicU64 = AtomicU64::new(0);
pub static CANON_CALLS: AtomicU64 = AtomicU64::new(0);
/// Wall nanoseconds inside the DB splice's verification gate.
pub static VERIFY_NS: AtomicU64 = AtomicU64::new(0);
/// Wall nanoseconds inside the randomized degree PROBE (xgate_degree_exceeds).
/// Measured to decide whether the probe earns its place against simply
/// reading the exact degree off the polynomial canonicalization needs anyway.
pub static DEGREE_NS: AtomicU64 = AtomicU64::new(0);
pub static DEGREE_CALLS: AtomicU64 = AtomicU64::new(0);

/// Exact max ANF degree over a polynomial set: the largest number of variables
/// in any monomial. Monomials are `u64` variable masks, so this is a popcount.
pub fn polys_max_degree(polys: &[Polynomial]) -> usize {
    polys
        .iter()
        .flat_map(|p| p.iter().map(|m| m.count_ones() as usize))
        .max()
        .unwrap_or(0)
}

pub fn canonicalize_xgates_single(
    gates: &[XGate],
    reversed: bool,
    budget: XPolyBudget,
) -> Result<CanonicalXPolys, XPolyError> {
    canonicalize_xgates_single_capped(gates, reversed, budget, 0)
}

/// As [`canonicalize_xgates_single`], but with `max_degree > 0` the exact ANF
/// degree is checked BETWEEN the two stages: after composing the polynomial
/// (2.5us, measured) and before canonicalizing it (1.16ms, measured). A window
/// above the store's maximum degree cannot match anything, so canonicalizing it
/// is pure waste.
///
/// This replaces a randomized probe over `probes` affine subspaces of dimension
/// `max_degree + 1`. That probe cost 43us per call -- 17x the polynomial it was
/// protecting -- rebuilt a window-independent axis table on every call, was
/// one-sided (it could certify "over" but silently miss it), and threw away
/// work canonicalization then redid. It never once fired in production
/// (`dsk=0` across every run). `max_degree = 0` disables the check.
pub fn canonicalize_xgates_single_capped(
    gates: &[XGate],
    reversed: bool,
    budget: XPolyBudget,
    max_degree: usize,
) -> Result<CanonicalXPolys, XPolyError> {
    let used_wires = xgate_used_wires(gates);
    if used_wires.len() > 64 {
        return Err(XPolyError::TooManyWires {
            wires: used_wires.len(),
        });
    }
    if used_wires.is_empty() {
        return Ok(CanonicalXPolys {
            polys: Vec::new(),
            order: Permutation { data: Vec::new() },
            used_wires,
        });
    }

    let mut dense = dense_remap(gates, &used_wires);
    if reversed {
        dense.reverse();
    }
    // Stage split, measured. The descent recomputes BOTH stages from scratch
    // at every rung even though consecutive rungs differ by one gate at one
    // end. Composition is peelable (one gate touches <=3 wires); the canonical
    // wire ORDER is a global property of the whole polynomial set and is not.
    // Which stage dominates decides whether peeling is worth building.
    // Time both stages INCLUDING failures. A window that blows the monomial
    // budget does most of the work before erroring, and an early `?` would
    // have hidden exactly the expensive cases.
    let t0 = std::time::Instant::now();
    let poly_res = xgates_to_polynomial(&dense, used_wires.len(), budget);
    let t1 = std::time::Instant::now();
    POLY_NS.fetch_add((t1 - t0).as_nanos() as u64, Ordering::Relaxed);
    CANON_CALLS.fetch_add(1, Ordering::Relaxed);
    let polys = poly_res?;
    if max_degree > 0 {
        let deg = polys_max_degree(&polys);
        if deg > max_degree {
            return Err(XPolyError::DegreeExceeded {
                degree: deg,
                limit: max_degree,
            });
        }
    }
    let t2 = std::time::Instant::now();
    let canon_res = canonicalize_polys_4(polys, true);
    let t3 = std::time::Instant::now();
    CANON_NS.fetch_add((t3 - t2).as_nanos() as u64, Ordering::Relaxed);
    let (polys, order) = canon_res.map_err(|_| XPolyError::CanonicalizationFailed)?;
    Ok(CanonicalXPolys {
        polys,
        order,
        used_wires,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::circuit::CircuitSeq;
    use crate::postmix::xgate::eval_lanes;

    fn eval_polys(polys: &[Polynomial], input: u64) -> u64 {
        let mut out = 0u64;
        for (wire, poly) in polys.iter().enumerate() {
            let bit = poly.iter().fold(false, |acc, &m| acc ^ ((input & m) == m));
            if bit {
                out |= 1u64 << wire;
            }
        }
        out
    }

    fn apply_scalar(gates: &[XGate], input: u64, n: usize) -> u64 {
        let mut lanes = (0..n)
            .map(|wire| if input >> wire & 1 == 1 { 1 } else { 0 })
            .collect::<Vec<u64>>();
        eval_lanes(gates.iter(), &mut lanes);
        lanes
            .iter()
            .enumerate()
            .fold(0, |acc, (wire, &v)| acc | ((v & 1) << wire))
    }

    #[test]
    fn arbitrary_xgate_polynomials_match_evaluation() {
        let gates = vec![
            XGate::x_gate(0),
            XGate::conj(1, [(0, true), (2, false)]).unwrap(),
            XGate {
                target: 3,
                comp: true,
                ctrls: [(0, false), (1, true), (2, false)].into_iter().collect(),
            },
            // comp XOR empty product = 1 XOR 1 = 0: an exact no-op.
            XGate {
                target: 2,
                comp: true,
                ctrls: Default::default(),
            },
        ];
        let polys = xgates_to_polynomial(&gates, 4, XPolyBudget::default()).unwrap();
        for input in 0..16u64 {
            assert_eq!(eval_polys(&polys, input), apply_scalar(&gates, input, 4));
        }
    }

    #[test]
    fn g57_canonical_keys_match_legacy_in_both_directions() {
        // KEY-COMPATIBILITY INVARIANT for the frozen-DB move: a window's true
        // function polynomial (canonicalize_xgates_single) must canonicalize to
        // the same key the DB was built under (canonicalize_polys_single on the
        // g57 triples). from_g57 and to_polynomial share the g57 convention, so
        // a triple decodes with plain from_g57.
        let legacy = CircuitSeq {
            gates: vec![[7, 2, 11], [2, 7, 5], [11, 5, 2], [5, 11, 7]],
        };
        let gates: Vec<XGate> = legacy.gates.iter().copied().map(XGate::from_g57).collect();
        for reversed in [false, true] {
            let old = legacy.canonicalize_polys_single(reversed);
            let new = canonicalize_xgates_single(&gates, reversed, XPolyBudget::default()).unwrap();
            assert_eq!(new.polys, old.0);
            assert_eq!(new.order, old.1);
            assert_eq!(new.used_wires, old.2);
        }
    }

    #[test]
    fn budget_exhaustion_is_a_clean_error() {
        let gates = vec![XGate::from_g57([0, 1, 2])];
        let budget = XPolyBudget {
            max_mul_terms: 1,
            max_poly_terms: 32,
            max_total_terms: 64,
        };
        assert!(matches!(
            xgates_to_polynomial(&gates, 3, budget),
            Err(XPolyError::BudgetExceeded {
                stage: "polynomial multiply",
                ..
            })
        ));
    }

    #[test]
    fn target_in_controls_is_rejected_before_reverse_lookup() {
        let gates = vec![XGate {
            target: 0,
            comp: false,
            ctrls: [(0, true)].into_iter().collect(),
        }];
        assert_eq!(
            canonicalize_xgates_single(&gates, false, XPolyBudget::default()).unwrap_err(),
            XPolyError::TargetInControls { target: 0 }
        );
    }
}
