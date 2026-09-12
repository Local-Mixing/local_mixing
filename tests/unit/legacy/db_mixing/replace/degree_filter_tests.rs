use super::*;
use crate::circuit::CircuitSeq;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// The flat bit-sliced witness must agree with the nested-Vec original on
// every (gates, dirs, base) input, including the m*nw layout mapping.
#[test]
fn opt_equiv_derivative_witness_flat_matches_nested() {
    let mut rng = StdRng::seed_from_u64(0xdeadbeef);
    for _ in 0..300 {
        let nw = rng.random_range(3..12usize);
        let m = rng.random_range(1..6usize);
        let n_gates = rng.random_range(0..25usize);
        let gates: Vec<[u16; 3]> = (0..n_gates)
            .map(|_| {
                [
                    rng.random_range(0..nw) as u16,
                    rng.random_range(0..nw) as u16,
                    rng.random_range(0..nw) as u16,
                ]
            })
            .collect();
        let dirs_nested: Vec<Vec<bool>> = (0..m)
            .map(|_| (0..nw).map(|_| rng.random_bool(0.5)).collect())
            .collect();
        let base: Vec<bool> = (0..nw).map(|_| rng.random_bool(0.5)).collect();
        let dirs_flat: Vec<bool> = dirs_nested.iter().flatten().copied().collect();
        let total = 1usize << m;
        let words = total.div_ceil(64);
        let mut col = vec![0u64; m * words];
        for i in 0..m {
            for p in 0..total {
                if (p >> i) & 1 == 1 {
                    col[i * words + p / 64] |= 1u64 << (p % 64);
                }
            }
        }
        let mut state = vec![0u64; nw * words];
        assert_eq!(
            derivative_witness_flat(&gates, nw, m, &dirs_flat, &base, &col, &mut state),
            derivative_witness(&gates, nw, &dirs_nested, &base),
            "nw={nw} m={m} gates={gates:?}"
        );
    }
}

// True max ANF degree of the window (both directions), computed exactly via to_polynomial.
fn true_max_degree(sub: &CircuitSeq) -> usize {
    let deg = |c: &CircuitSeq| -> usize {
        let n = c.max_wire() as usize + 1;
        c.to_polynomial(n, 0, c.gates.len())
            .iter()
            .flat_map(|p| p.iter().map(|m| m.count_ones() as usize))
            .max()
            .unwrap_or(0)
    };
    let rev = CircuitSeq {
        gates: sub.gates.iter().rev().copied().collect(),
    };
    deg(sub).max(deg(&rev))
}

#[test]
fn degree_filter_is_sound_and_effective() {
    let mut rng = StdRng::seed_from_u64(12345);
    let d = 5usize; // small threshold so tests exercise both sides quickly
    let mut discarded_ok = 0;
    let mut low_seen = 0;
    let mut high_seen = 0;
    let mut high_caught = 0;
    for _ in 0..4000 {
        let nw = rng.random_range(6..14u16);
        let ng = rng.random_range(4..16usize);
        let mut gates: Vec<[u16; 3]> = Vec::new();
        while gates.len() < ng {
            let a = rng.random_range(0..nw);
            let b = rng.random_range(0..nw);
            let c = rng.random_range(0..nw);
            if a != b && a != c && b != c {
                gates.push([a, b, c]);
            }
        }
        let sub = CircuitSeq { gates };
        let truth = true_max_degree(&sub);
        let discards = degree_filter_discards(&sub, d, 10, &mut rng);
        // SOUNDNESS: never discard a window whose min-direction degree could match (<= d).
        // degree_filter_discards requires BOTH directions > d, so if EITHER direction <= d it
        // must not discard. Reconstruct per-direction to assert precisely.
        let fwd_deg = {
            let n = sub.max_wire() as usize + 1;
            sub.to_polynomial(n, 0, sub.gates.len())
                .iter()
                .flat_map(|p| p.iter().map(|m| m.count_ones() as usize))
                .max()
                .unwrap_or(0)
        };
        let rev_c = CircuitSeq {
            gates: sub.gates.iter().rev().copied().collect(),
        };
        let rev_deg = {
            let n = rev_c.max_wire() as usize + 1;
            rev_c
                .to_polynomial(n, 0, rev_c.gates.len())
                .iter()
                .flat_map(|p| p.iter().map(|m| m.count_ones() as usize))
                .max()
                .unwrap_or(0)
        };
        if fwd_deg <= d || rev_deg <= d {
            assert!(
                !discards,
                "SOUNDNESS VIOLATION: discarded a matchable window (fwd_deg={}, rev_deg={}, d={})",
                fwd_deg, rev_deg, d
            );
            low_seen += 1;
        } else {
            high_seen += 1;
            if discards {
                high_caught += 1;
                discarded_ok += 1;
            }
        }
        let _ = truth;
    }
    println!(
        "low(unmatchable-guard held)={} high={} high_caught={} ({:.0}%)",
        low_seen,
        high_seen,
        high_caught,
        100.0 * high_caught as f64 / high_seen.max(1) as f64
    );
    assert!(discarded_ok > 0, "filter never fired — probes ineffective");
    let _ = high_seen;
}

// A balanced product tree: pairs of disjoint fresh inputs multiplied up `levels` deep, so
// the top monomial degree is 2^levels — the shape mixing manufactures and the pathology the
// filter exists for. Gate [a,b,c] gives a' = a ^ 1 ^ b ^ bc, top term b*c; feeding disjoint
// subtrees into b and c squares the degree each level.
fn product_tree(levels: u32) -> (CircuitSeq, usize) {
    let leaves = 1usize << levels;
    let mut next = leaves as u16; // fresh target wires start above the input leaves
    let mut gates: Vec<[u16; 3]> = Vec::new();
    // level 1: pair leaves 2i,2i+1 into fresh targets
    let mut layer: Vec<u16> = Vec::new();
    let mut i = 0u16;
    while (i as usize) < leaves {
        let t = next;
        next += 1;
        gates.push([t, i, i + 1]);
        layer.push(t);
        i += 2;
    }
    // higher levels: pair disjoint sub-results
    while layer.len() > 1 {
        let mut nl = Vec::new();
        let mut j = 0;
        while j + 1 < layer.len() {
            let t = next;
            next += 1;
            gates.push([t, layer[j], layer[j + 1]]);
            nl.push(t);
            j += 2;
        }
        layer = nl;
    }
    (CircuitSeq { gates }, 1usize << levels)
}

// Effectiveness on the windows that actually matter: constructed product trees whose degree
// (2^levels) is well above the threshold. These must be caught essentially always.
#[test]
fn degree_filter_catches_explosive_windows() {
    let mut rng = StdRng::seed_from_u64(999);
    let d = 5usize;
    let deg = |c: &CircuitSeq| -> usize {
        let n = c.max_wire() as usize + 1;
        c.to_polynomial(n, 0, c.gates.len())
            .iter()
            .flat_map(|p| p.iter().map(|m| m.count_ones() as usize))
            .max()
            .unwrap_or(0)
    };
    // The windows that actually STALL canonicalization are the DENSE high-degree ones
    // (monomial count is what canon4 cost scales with; sparse high-degree windows are cheap).
    // Deeper product trees are both higher-degree and denser (level 4 ~ 33k monomials), so
    // the per-probe hit rate climbs steeply — that is exactly the regime the filter targets.
    // A product tree's inverse is low-degree (uncompute), so degree_filter_discards (both
    // directions) must NOT fire: reverse-compressible windows are preserved.
    let mono = |c: &CircuitSeq| -> usize {
        let n = c.max_wire() as usize + 1;
        c.to_polynomial(n, 0, c.gates.len())
            .iter()
            .map(|p| p.len())
            .max()
            .unwrap_or(0)
    };
    // Cap at level 4 (degree 16, ~33k monomials): level 5 would be degree 32, whose ground-
    // truth to_polynomial here explodes to billions of monomials — the very pathology the
    // filter sidesteps but the exact-degree oracle cannot. Level 4 already exercises the
    // dense-high regime the filter targets.
    for levels in [3u32, 4] {
        let (sub, top_deg) = product_tree(levels);
        assert!(deg(&sub) > d, "forward tree not high-degree");
        let trials = 200;
        let mut fwd_caught = 0;
        let mut wrongly_discarded = 0;
        for _ in 0..trials {
            if degree_exceeds_dir(&sub, false, d, 8, &mut rng) {
                fwd_caught += 1;
            }
            if degree_filter_discards(&sub, d, 8, &mut rng) {
                wrongly_discarded += 1;
            }
        }
        println!(
            "tree deg {} ({} monomials): forward caught {}/{}, wrongly discarded {}",
            top_deg,
            mono(&sub),
            fwd_caught,
            trials,
            wrongly_discarded
        );
        let rev = CircuitSeq {
            gates: sub.gates.iter().rev().copied().collect(),
        };
        if deg(&rev) <= d {
            assert_eq!(
                wrongly_discarded, 0,
                "discarded a reverse-compressible window (deg {})",
                top_deg
            );
        }
        // The dense deep trees (the stall-causing regime) must be caught essentially always.
        if levels >= 4 {
            assert!(
                fwd_caught * 100 >= trials * 90,
                "dense deg-{} tree caught < 90%: {}/{}",
                top_deg,
                fwd_caught,
                trials
            );
        }
    }
}
