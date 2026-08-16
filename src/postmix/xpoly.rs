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
    /// Exact ANF degree exceeds the maximum represented by the store.
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
            // Positive literals multiply by the wire polynomial directly; only
            // negation needs a temporary.
            product = if positive {
                poly_mul(&product, &polys[wire_idx], budget)?
            } else {
                let literal = poly_not(&polys[wire_idx]);
                poly_mul(&product, &literal, budget)?
            };
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
pub fn canonicalize_xgates_single(
    gates: &[XGate],
    reversed: bool,
    budget: XPolyBudget,
) -> Result<CanonicalXPolys, XPolyError> {
    canonicalize_xgates_single_capped(gates, reversed, budget, 0)
}

/// Exact maximum ANF degree over a polynomial set.
pub fn polys_max_degree(polys: &[Polynomial]) -> usize {
    polys
        .iter()
        .flat_map(|polynomial| {
            polynomial
                .iter()
                .map(|monomial| monomial.count_ones() as usize)
        })
        .max()
        .unwrap_or(0)
}

/// Canonicalize while rejecting over-degree functions after polynomial
/// composition but before the much more expensive canonical wire ordering.
/// A zero cap preserves the historical behavior.
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
    // Everything below is a pure function of (dense, wire count, caps) — the
    // dense window is independent of the original wire ids — so windows repeat
    // heavily across the circuit and an exact process-wide cache applies
    // (mirrors the legacy g57 canon cache in circuit.rs). Both Ok results and
    // cap/error outcomes are cached; the caps are part of the key so differing
    // budgets never alias.
    let num_wires = used_wires.len();
    let Some(cache) = xpoly_canon_cache() else {
        let (polys, order) = compose_and_canonicalize(&dense, num_wires, budget, max_degree)?;
        return Ok(CanonicalXPolys {
            polys,
            order,
            used_wires,
        });
    };
    let key = xpoly_canon_cache_key(&dense, num_wires, budget, max_degree);
    XPOLY_CANON_CACHE_QUERIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if let Some(entry) = cache.get(&key) {
        XPOLY_CANON_CACHE_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return match entry.value().as_ref() {
            Ok((polys, order)) => Ok(CanonicalXPolys {
                polys: polys.clone(),
                order: order.clone(),
                used_wires,
            }),
            Err(e) => Err(e.clone()),
        };
    }
    let computed = compose_and_canonicalize(&dense, num_wires, budget, max_degree);
    let entry_bytes = key.len() as u64
        + 64
        + match &computed {
            Ok((polys, order)) => {
                (polys.iter().map(|p| p.len()).sum::<usize>() * 8 + order.data.len() * 8) as u64
            }
            Err(_) => 16,
        };
    if XPOLY_CANON_CACHE_BYTES.fetch_add(entry_bytes, std::sync::atomic::Ordering::Relaxed)
        + entry_bytes
        > xpoly_canon_cache_cap_bytes()
    {
        cache.clear();
        XPOLY_CANON_CACHE_BYTES.store(entry_bytes, std::sync::atomic::Ordering::Relaxed);
    }
    cache.insert(key, std::sync::Arc::new(computed.clone()));
    let (polys, order) = computed?;
    Ok(CanonicalXPolys {
        polys,
        order,
        used_wires,
    })
}

/// The cacheable core of [`canonicalize_xgates_single_capped`]: compose the
/// dense window, apply the degree cap, canonicalize.
fn compose_and_canonicalize(
    dense: &[XGate],
    num_wires: usize,
    budget: XPolyBudget,
    max_degree: usize,
) -> Result<(Vec<Polynomial>, Permutation), XPolyError> {
    // Time both stages INCLUDING failures (a window that blows the monomial
    // budget does most of the work before erroring). Cache hits skip this fn
    // entirely, so the accumulated times cover fresh computations only.
    let t0 = std::time::Instant::now();
    let poly_res = xgates_to_polynomial(dense, num_wires, budget);
    POLY_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    CANON_CALLS.fetch_add(1, Ordering::Relaxed);
    let polys = poly_res?;
    if max_degree > 0 {
        let degree = polys_max_degree(&polys);
        if degree > max_degree {
            return Err(XPolyError::DegreeExceeded {
                degree,
                limit: max_degree,
            });
        }
    }
    let t2 = std::time::Instant::now();
    let canon_res = canonicalize_polys_4(polys, true);
    CANON_NS.fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);
    canon_res.map_err(|_| XPolyError::CanonicalizationFailed)
}

use std::sync::atomic::{AtomicU64, Ordering};

// Fresh-computation stage timers read by the fmix report line (cache hits do
// not accumulate — they skip both stages).
pub static POLY_NS: AtomicU64 = AtomicU64::new(0);
pub static CANON_NS: AtomicU64 = AtomicU64::new(0);
pub static CANON_CALLS: AtomicU64 = AtomicU64::new(0);
// Accumulated from the mix loop's local-verify path.
pub static VERIFY_NS: AtomicU64 = AtomicU64::new(0);
pub static DEGREE_NS: AtomicU64 = AtomicU64::new(0);
pub static DEGREE_CALLS: AtomicU64 = AtomicU64::new(0);

type XPolyCanonResult = Result<(Vec<Polynomial>, Permutation), XPolyError>;
type XPolyCanonMap =
    dashmap::DashMap<Box<[u8]>, std::sync::Arc<XPolyCanonResult>, rustc_hash::FxBuildHasher>;

pub static XPOLY_CANON_CACHE_HITS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static XPOLY_CANON_CACHE_QUERIES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static XPOLY_CANON_CACHE_BYTES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// XPOLY_CANON_CACHE_MB caps the approximate cache footprint (default 1024;
/// 0 disables). Cap overflow clears the whole map (epoch reset), like the
/// legacy canon and lookup caches. The default was raised from 256 after a
/// profiled 200k-move DB run measured 3 epoch resets and a 35% hit rate at
/// 256MB — the working set of a production phase A does not fit.
fn xpoly_canon_cache_cap_bytes() -> u64 {
    static CAP: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("XPOLY_CANON_CACHE_MB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(1024)
            .saturating_mul(1024 * 1024)
    })
}

fn xpoly_canon_cache() -> Option<&'static XPolyCanonMap> {
    static CACHE: std::sync::OnceLock<Option<XPolyCanonMap>> = std::sync::OnceLock::new();
    CACHE
        .get_or_init(|| (xpoly_canon_cache_cap_bytes() > 0).then(XPolyCanonMap::default))
        .as_ref()
}

/// Canonical byte encoding of the cache key. Dense wires are < 64 and ctrls
/// are sorted by wire (XGate invariant), so the encoding is canonical.
fn xpoly_canon_cache_key(
    dense: &[XGate],
    num_wires: usize,
    budget: XPolyBudget,
    max_degree: usize,
) -> Box<[u8]> {
    let mut k = Vec::with_capacity(40 + dense.len() * 16);
    k.push(num_wires as u8);
    k.extend_from_slice(&(budget.max_mul_terms as u64).to_le_bytes());
    k.extend_from_slice(&(budget.max_poly_terms as u64).to_le_bytes());
    k.extend_from_slice(&(budget.max_total_terms as u64).to_le_bytes());
    k.extend_from_slice(&(max_degree as u64).to_le_bytes());
    for g in dense {
        k.push(g.target as u8);
        k.push(g.comp as u8);
        k.push(g.ctrls.len() as u8);
        for &(w, p) in &g.ctrls {
            k.push(w as u8);
            k.push(p as u8);
        }
    }
    k.into_boxed_slice()
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
    fn exact_degree_cap_rejects_high_degree_and_keeps_low_degree() {
        let high = vec![XGate::conj(0, [(1, true), (2, true), (3, true), (4, true)]).unwrap()];
        assert!(matches!(
            canonicalize_xgates_single_capped(&high, false, XPolyBudget::default(), 3),
            Err(XPolyError::DegreeExceeded {
                degree: 4,
                limit: 3
            })
        ));

        let low = vec![XGate::from_g57([0, 1, 2])];
        assert!(canonicalize_xgates_single_capped(&low, false, XPolyBudget::default(), 2).is_ok());
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
