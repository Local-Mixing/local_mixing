use super::*;
use crate::circuit::random_circuit;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BTreeSet;

#[test]
fn polynomial_from_terms_sorts_and_cancels_pairs() {
    assert_eq!(polynomial_from_terms([5, 3, 5, 1, 3, 5, 7, 7]), vec![1, 5]);
}

// The byte parser replaced a char-by-char one. This is the old
// implementation verbatim, so any grammar drift shows up as a diff.
fn ref_from_string(s: &str) -> CircuitSeq {
    fn char_to_wire(c: char) -> u8 {
        match c {
            '0'..='9' => c as u8 - b'0',
            'a'..='z' => c as u8 - b'a' + 10,
            'A'..='Z' => c as u8 - b'A' + 36,
            '!' => 62,
            '@' => 63,
            '#' => 64,
            '$' => 65,
            '%' => 66,
            '^' => 67,
            '&' => 68,
            '*' => 69,
            '(' => 70,
            ')' => 71,
            '-' => 72,
            '_' => 73,
            '=' => 74,
            '+' => 75,
            '[' => 76,
            ']' => 77,
            '{' => 78,
            '}' => 79,
            '<' => 80,
            '>' => 81,
            '?' => 82,
            _ => panic!("Invalid wire char: {}", c),
        }
    }
    const BASE: u32 = 83;
    let gates: Vec<[u16; 3]> = s
        .trim()
        .split(';')
        .filter(|part| !part.is_empty())
        .map(|gate_str| {
            let mut chars = gate_str.chars().peekable();
            let mut wires = Vec::new();
            while chars.peek().is_some() {
                let mut overflow = 0;
                while chars.peek() == Some(&'~') {
                    overflow += 1;
                    chars.next();
                }
                let c = chars.next().expect("Expected wire character after ~");
                wires.push((char_to_wire(c) as u32 + overflow * BASE) as u16);
            }
            assert_eq!(wires.len(), 3, "Each gate must have exactly 3 wires");
            [wires[0], wires[1], wires[2]]
        })
        .collect();
    CircuitSeq { gates }
}

#[test]
fn opt_equiv_from_bytes_matches_char_parser_reference() {
    let mut rng = StdRng::seed_from_u64(0x5eed_2026);
    // Widths that exercise plain characters, one `~` and several `~`.
    for &wires in &[8usize, 83, 84, 166, 167, 300, 1024] {
        for _ in 0..40 {
            let m = rng.random_range(0..60usize);
            let gates: Vec<[u16; 3]> = (0..m)
                .map(|_| {
                    [
                        rng.random_range(0..wires) as u16,
                        rng.random_range(0..wires) as u16,
                        rng.random_range(0..wires) as u16,
                    ]
                })
                .collect();
            let text = CircuitSeq { gates }.repr();
            assert_eq!(
                CircuitSeq::from_string(&text).gates,
                ref_from_string(&text).gates,
                "parser drift on {text}"
            );
        }
    }
}

#[test]
fn opt_equiv_from_bytes_keeps_whitespace_and_empty_segment_handling() {
    // Every wire alphabet class, plus one- and two-`~` overflow prefixes
    // (wires 83 and 175).
    let body = "0az;A!?;~0~~9~1;))(;";
    let expect = ref_from_string(body).gates;
    for wrapped in [
        body.to_string(),
        format!("  {body}\n"),
        format!("\n\t{body}  \r\n"),
        // Empty segments between separators are skipped, not gates.
        body.replace(';', ";;"),
    ] {
        assert_eq!(
            CircuitSeq::from_string(&wrapped).gates,
            expect,
            "mismatch on {wrapped:?}"
        );
    }
    // Empty and separator-only input yield no gates, as before.
    assert!(CircuitSeq::from_string("").gates.is_empty());
    assert!(CircuitSeq::from_string("   \n").gates.is_empty());
    assert!(CircuitSeq::from_string(";;;").gates.is_empty());
}

#[test]
#[should_panic(expected = "Invalid wire char")]
fn from_bytes_rejects_a_non_alphabet_character() {
    CircuitSeq::from_string("01 2;");
}

#[test]
#[should_panic(expected = "exactly 3 wires")]
fn from_bytes_rejects_a_short_gate() {
    CircuitSeq::from_string("01;");
}

// The bit-sliced kernel must reproduce the scalar one lane for lane,
// including the X-gate spelling (pos == neg), which fires unconditionally.
#[test]
fn opt_equiv_lane_kernel_matches_scalar_evaluation() {
    let mut rng = StdRng::seed_from_u64(0xfeed_babe);
    for &wires in &[1usize, 2, 8, 64, 65, 128, 200, 256] {
        for trial in 0..25 {
            let m = rng.random_range(0..80usize);
            let gates: Vec<[u16; 3]> = (0..m)
                .map(|_| {
                    let t = rng.random_range(0..wires) as u16;
                    let x = rng.random_range(0..wires) as u16;
                    // Every third gate is an X gate: controls equal.
                    let y = if trial % 3 == 0 && rng.random_bool(0.5) {
                        x
                    } else {
                        rng.random_range(0..wires) as u16
                    };
                    [t, x, y]
                })
                .collect();

            let len = lane_state_len(wires);
            let mut lanes = vec![0u64; len];
            for lane in lanes[..wires].iter_mut() {
                *lane = rng.random();
            }
            let seed = lanes.clone();
            Gate::eval_lanes_index_list(&gates, &mut lanes);

            // Each of the 64 lanes must equal the scalar walk on the
            // sample that lane carries.
            for bit in 0..64 {
                let mut scalar = U1024::zero();
                for w in 0..wires {
                    if (seed[w] >> bit) & 1 == 1 {
                        scalar = scalar | (U1024::one() << w);
                    }
                }
                let out = Gate::evaluate_index_list_1024(scalar, &gates);
                for w in 0..wires {
                    let want = ((out >> w) & U1024::one()) != U1024::zero();
                    let got = (lanes[w] >> bit) & 1 == 1;
                    assert_eq!(want, got, "wire {w} lane {bit} wires={wires}");
                }
            }
        }
    }
}

#[test]
fn opt_equiv_evaluate_index_list_64_matches_wider_kernels() {
    let mut rng = StdRng::seed_from_u64(0xc0ffee);
    for _ in 0..200 {
        let m = rng.random_range(0..64usize);
        let gates: Vec<[u16; 3]> = (0..m)
            .map(|_| {
                let x = rng.random_range(0..64u16);
                [
                    rng.random_range(0..64u16),
                    x,
                    if rng.random_bool(0.15) {
                        x
                    } else {
                        rng.random_range(0..64u16)
                    },
                ]
            })
            .collect();
        let input: u64 = rng.random();
        let got = Gate::evaluate_index_list_64(input, &gates);
        let want = Gate::evaluate_index_list_128(input as u128, &gates);
        assert_eq!(got as u128, want);
        let mut v = u256::zero();
        v.0[0] = input;
        assert_eq!(Gate::evaluate_index_list_256(v, &gates).0[0], got);
    }
}

// Limb-indexed wide kernels must be bit-exact against the original
// full-width bignum shift formulation for every in-range wire, including
// limb boundaries.
#[test]
fn opt_equiv_limb_kernels_match_bignum_shifts() {
    fn ref_256(state: u256, gate: [u16; 3]) -> u256 {
        let one = u256::one();
        let c1 = (state >> gate[1]) & one;
        let c2 = (state >> gate[2]) & one;
        state ^ ((c1 | (one ^ c2)) << gate[0])
    }
    fn ref_512(state: u512, gate: [u16; 3]) -> u512 {
        let one = u512::one();
        let c1 = (state >> gate[1]) & one;
        let c2 = (state >> gate[2]) & one;
        state ^ ((c1 | (one ^ c2)) << gate[0])
    }
    fn ref_1024(state: U1024, gate: [u16; 3]) -> U1024 {
        let one = U1024::one();
        let c1 = (state >> gate[1]) & one;
        let c2 = (state >> gate[2]) & one;
        state ^ ((c1 | (one ^ c2)) << gate[0])
    }
    let mut state = 0x0dd_ba11_5eed_2026u64;
    let mut next = |m: u64| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) % m
    };
    let boundary = [0u64, 1, 63, 64, 65, 127, 128, 191, 255];
    for trial in 0..4000 {
        let pick = |next: &mut dyn FnMut(u64) -> u64, width: u64| -> u16 {
            if trial % 3 == 0 {
                boundary[next(boundary.len() as u64) as usize].min(width - 1) as u16
            } else {
                next(width) as u16
            }
        };
        let mut bytes = [0u8; 128];
        for b in bytes.iter_mut() {
            *b = next(256) as u8;
        }
        let g256 = [
            pick(&mut next, 256),
            pick(&mut next, 256),
            pick(&mut next, 256),
        ];
        let s256 = u256::from_little_endian(&bytes[..32]);
        assert_eq!(Gate::evaluate_index_256(s256, g256), ref_256(s256, g256));
        let g512 = [
            pick(&mut next, 512),
            pick(&mut next, 512),
            pick(&mut next, 512),
        ];
        let s512 = u512::from_little_endian(&bytes[..64]);
        assert_eq!(Gate::evaluate_index_512(s512, g512), ref_512(s512, g512));
        let g1024 = [
            pick(&mut next, 1024),
            pick(&mut next, 1024),
            pick(&mut next, 1024),
        ];
        let s1024 = U1024::from_little_endian(&bytes);
        assert_eq!(
            Gate::evaluate_index_1024(s1024, g1024),
            ref_1024(s1024, g1024)
        );
    }
}

// The u512 probably_equal arm (300 eval wires) must keep both verdicts.
#[test]
fn opt_equiv_probably_equal_512_arm_agrees() {
    let c = random_circuit(300, 400);
    // Equivalent pair: append a cancelling gate pair (target not among its
    // controls, so the two applications undo each other).
    let mut c2 = c.clone();
    c2.gates.push([5, 20, 30]);
    c2.gates.push([5, 20, 30]);
    assert!(c.probably_equal(&c2, 300, 64).is_ok());
    // Non-equivalent pair: [0,1,1] fires on every input (c1 | !c1), so
    // wire 0 differs on every drawn input.
    let mut c3 = c.clone();
    c3.gates.push([0, 1, 1]);
    assert!(c.probably_equal(&c3, 300, 64).is_err());
}

// The stack-style pass must reproduce the historical drain-with-backtrack
// cancellation exactly, including tag lockstep.
#[test]
fn opt_equiv_cancel_adjacent_duplicates_matches_drain_reference() {
    fn reference(gates: &mut Vec<[u16; 3]>, tags: Option<&mut Vec<u32>>) {
        let mut tags = tags;
        let mut j = 0;
        while j < gates.len().saturating_sub(1) {
            if gates[j] == gates[j + 1] {
                gates.drain(j..=j + 1);
                if let Some(tags) = tags.as_deref_mut() {
                    tags.drain(j..=j + 1);
                }
                j = j.saturating_sub(2);
            } else {
                j += 1;
            }
        }
    }
    let mut state = 0xdead_beef_cafe_1234u64;
    let mut next = |m: u64| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) % m
    };
    for len in [0usize, 1, 2, 3, 10, 200, 2000] {
        // A tiny alphabet makes adjacent duplicates and cascades common.
        let mut gates: Vec<[u16; 3]> = (0..len)
            .map(|_| [next(3) as u16, next(3) as u16, next(3) as u16])
            .collect();
        let mut tags: Vec<u32> = (0..len as u32).collect();
        let mut ref_gates = gates.clone();
        let mut ref_tags = tags.clone();
        cancel_adjacent_duplicates(&mut gates, Some(&mut tags));
        reference(&mut ref_gates, Some(&mut ref_tags));
        assert_eq!(gates, ref_gates, "len={len}");
        assert_eq!(tags, ref_tags, "len={len}");
    }
}

#[test]
fn polynomial_xor_assign_cancels_shared_terms() {
    let mut left = vec![1, 3, 8];
    poly_xor_assign(&mut left, vec![3, 5, 8]);
    assert_eq!(left, vec![1, 5]);
}

#[test]
fn to_polynomial_keeps_terms_sorted_and_cancelled() {
    let circuit = CircuitSeq {
        gates: vec![[0, 1, 2]],
    };
    let polys = circuit.to_polynomial(3, 0, 1);
    assert_eq!(polys[0], vec![0, 1, 4, 6]);
    assert_eq!(polys[1], vec![2]);
    assert_eq!(polys[2], vec![4]);
}

#[test]
fn evaluate_1024_handles_wires_above_512() {
    let circuit = CircuitSeq {
        gates: vec![[900, 901, 902]],
    };
    let one = U1024::one();

    let flipped = circuit.evaluate_1024(U1024::zero());
    assert_eq!((flipped >> 900) & one, one);

    let blocked = circuit.evaluate_1024(one << 902);
    assert_eq!((blocked >> 900) & one, U1024::zero());
}

#[test]
fn evaluate_128_matches_256_for_supported_wires() {
    let mut rng = fastrand::Rng::with_seed(0x6576_616c_3132_38);
    for _ in 0..500 {
        let n = rng.usize(3..=128);
        let m = rng.usize(0..=(3 * n));
        fastrand::seed(rng.u64(..));
        let circuit = random_circuit(n, m);
        let input = rng.u128(..);
        let mask = if n < 128 { (1u128 << n) - 1 } else { u128::MAX };

        let output_128 = circuit.evaluate_128(input) & mask;
        let output_256 = circuit.evaluate_256(u256::from(input)) & u256::from(mask);
        assert_eq!(u256::from(output_128), output_256, "n={n} m={m}");
    }
}

fn old_toggle(poly: &mut BTreeSet<Monomial>, m: Monomial) {
    if !poly.remove(&m) {
        poly.insert(m);
    }
}

fn old_xor(mut left: BTreeSet<Monomial>, right: BTreeSet<Monomial>) -> BTreeSet<Monomial> {
    for m in right {
        old_toggle(&mut left, m);
    }
    left
}

fn old_and(left: &BTreeSet<Monomial>, right: &BTreeSet<Monomial>) -> BTreeSet<Monomial> {
    let mut result = BTreeSet::new();
    for &m1 in left {
        for &m2 in right {
            old_toggle(&mut result, m1 | m2);
        }
    }
    result
}

fn old_not(poly: BTreeSet<Monomial>) -> BTreeSet<Monomial> {
    old_xor(BTreeSet::from([0u64]), poly)
}

fn old_hashset_style_to_polynomial(circuit: &CircuitSeq, n: usize) -> Vec<Polynomial> {
    let mut polys: Vec<BTreeSet<Monomial>> = (0..n).map(|i| BTreeSet::from([1u64 << i])).collect();

    for &[a, b, c] in &circuit.gates {
        let not_b = old_not(polys[b as usize].clone());
        let term = old_and(&polys[c as usize], &not_b);
        let mut new_a = old_xor(polys[a as usize].clone(), term);
        old_toggle(&mut new_a, 0u64);
        polys[a as usize] = new_a;
    }

    polys
        .into_iter()
        .map(|poly| poly.into_iter().collect())
        .collect()
}

#[test]
fn to_polynomial_matches_old_hashset_style_implementation() {
    let mut rng = fastrand::Rng::with_seed(0x706f_6c79_7665_6375);
    for _ in 0..200 {
        let n = rng.usize(3..=12);
        let m = rng.usize(0..=(3 * n));
        fastrand::seed(rng.u64(..));
        let circuit = random_circuit(n, m);

        assert_eq!(
            circuit.to_polynomial(n, 0, circuit.gates.len()),
            old_hashset_style_to_polynomial(&circuit, n)
        );
    }
}

#[test]
fn to_polynomial_capped_matches_unbounded_and_bails_cleanly() {
    let circuit = CircuitSeq {
        gates: vec![[0, 1, 2]],
    };
    let expected = circuit.to_polynomial(3, 0, circuit.gates.len());

    assert_eq!(
        circuit.to_polynomial_capped(3, 0, circuit.gates.len(), 4),
        Some(expected)
    );
    assert_eq!(
        circuit.to_polynomial_capped(3, 0, circuit.gates.len(), 3),
        None
    );
}

#[test]
fn canonicalize_skips_window_over_64_distinct_wires() {
    // 22 disjoint triples touch 66 distinct wires. Building u64
    // monomials for this window would alias variables above bit 63.
    let gates: Vec<[u16; 3]> = (0..22u16)
        .map(|gate| [3 * gate, 3 * gate + 1, 3 * gate + 2])
        .collect();
    let circuit = CircuitSeq { gates };
    assert_eq!(circuit.used_wires().len(), 66);

    let skips_before = OVERSIZED_CANON_SKIPS.load(Ordering::Relaxed);
    assert!(circuit.canonicalize_polys_single(false).0.is_empty());
    assert!(circuit.canonicalize_polys_single(true).0.is_empty());
    assert!(circuit.canonicalize_polys_single_neg(&[]).0.is_empty());
    assert!(OVERSIZED_CANON_SKIPS.load(Ordering::Relaxed) >= skips_before.saturating_add(3));

    let small = CircuitSeq {
        gates: vec![[0, 1, 2], [1, 2, 0]],
    };
    assert!(!small.canonicalize_polys_single(false).0.is_empty());
}

fn eval_poly(poly: &Polynomial, input: usize) -> usize {
    poly.iter().fold(0usize, |acc, &monomial| {
        let term = if monomial == 0 || ((input as u64) & monomial) == monomial {
            1
        } else {
            0
        };
        acc ^ term
    })
}

fn eval_polys(polys: &[Polynomial], input: usize) -> usize {
    polys.iter().enumerate().fold(0usize, |acc, (wire, poly)| {
        acc | (eval_poly(poly, input) << wire)
    })
}

#[test]
fn probably_equal_widens_to_actual_circuit_wires() {
    // Both circuits compute w2 ^= (w1 AND NOT w0) on the 200-wire compare
    // contract, via a zero-initialized aux wire. A's aux lives at wire
    // 300/301 — beyond u256 — so the old num_wires-based width dispatch
    // evaluated it in u256 where shifts >= 256 silently return 0,
    // corrupting the aux accesses and reporting a false "not equal".
    // Width must follow the circuits' actual max wire.
    let a = CircuitSeq {
        gates: vec![[300, 0, 1], [2, 301, 300]],
    };
    let b = CircuitSeq {
        gates: vec![[250, 0, 1], [2, 251, 250]],
    };
    assert!(
        a.probably_equal(&b, 200, 512).is_ok(),
        "equivalent circuits reported non-equal — width dispatch regressed"
    );
    assert!(b.probably_equal(&a, 200, 512).is_ok());

    // Sanity: genuinely different circuits are still detected at the
    // widened evaluation width.
    let c = CircuitSeq {
        gates: vec![[2, 1, 0]],
    };
    assert!(a.probably_equal(&c, 200, 512).is_err());
}

// The packed-prefix rank-key comparator must order and equate exactly like
// the full 65-byte lexicographic compare it replaces.
#[test]
fn opt_equiv_rank_key_prefix_cmp_matches_lexicographic() {
    let mut state = 0x7261_6e6b_6b65_7934u64;
    let mut next = |m: u64| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) % m
    };
    let mut keys: Vec<MonomialRankKey4> = Vec::new();
    // Raw random byte arrays with a tiny alphabet and frequent short
    // fills, so prefix ties (and ties resolved only past byte 16) are
    // common.
    for _ in 0..200 {
        let mut encoded_ranks = [0u8; MONOMIAL_RANK_KEY_LEN_4];
        let filled = if next(2) == 0 {
            next(6) as usize
        } else {
            next(1 + MONOMIAL_RANK_KEY_LEN_4 as u64) as usize
        };
        for slot in 0..filled {
            encoded_ranks[slot] = next(4) as u8;
        }
        let degree = encoded_ranks.iter().filter(|&&b| b != 0).count() as u8;
        let prefix = u128::from_be_bytes(
            encoded_ranks[..MONOMIAL_RANK_PREFIX_LEN_4]
                .try_into()
                .unwrap(),
        );
        keys.push(MonomialRankKey4 {
            degree,
            prefix,
            encoded_ranks,
        });
    }
    // Realistic keys from random monomials under random rank vectors.
    for _ in 0..200 {
        let n = 1 + next(16) as usize;
        let vr: Vec<usize> = (0..n).map(|_| next(n as u64) as usize).collect();
        let m: Monomial = next(1u64 << n);
        keys.push(monomial_rank_key_4(m, &vr, n));
    }
    for a in &keys {
        for b in &keys {
            assert_eq!(a.cmp(b), a.encoded_ranks.cmp(&b.encoded_ranks));
            assert_eq!(*a == *b, a.encoded_ranks == b.encoded_ranks);
        }
    }
}

// The stack-bitset used_wires fast path (and the len-only variant) must
// match the historical marking implementation, on either side of the
// 1024-wire fallback boundary.
#[test]
fn opt_equiv_used_wires_matches_marking_reference() {
    fn reference(gates: &[[u16; 3]]) -> Vec<u16> {
        let Some(max_wire) = gates.iter().flatten().copied().max() else {
            return Vec::new();
        };
        let mut used = vec![false; max_wire as usize + 1];
        for &[target, control_a, control_b] in gates {
            used[target as usize] = true;
            used[control_a as usize] = true;
            used[control_b as usize] = true;
        }
        used.into_iter()
            .enumerate()
            .filter_map(|(wire, is_used)| is_used.then_some(wire as u16))
            .collect()
    }
    let mut state = 0x7573_6564_7769_7265u64;
    let mut next = |m: u64| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) % m
    };
    for len in [0usize, 1, 5, 40, 300] {
        for &wire_span in &[3u64, 60, 1023, 1024, 1500, 65535] {
            let gates: Vec<[u16; 3]> = (0..len)
                .map(|_| {
                    [
                        next(wire_span + 1) as u16,
                        next(wire_span + 1) as u16,
                        next(wire_span + 1) as u16,
                    ]
                })
                .collect();
            let circuit = CircuitSeq { gates };
            let expected = reference(&circuit.gates);
            assert_eq!(circuit.used_wires(), expected, "len={len} span={wire_span}");
            assert_eq!(circuit.used_wires_len(), expected.len());
        }
    }
}

// Scratch-buffer to_polynomial/_capped must reproduce the historical
// allocate-per-gate pipeline exactly, including the capped budget checks.
fn reference_to_polynomial(
    circuit: &CircuitSeq,
    n: usize,
    start: usize,
    end: usize,
) -> Vec<Polynomial> {
    let gates = &circuit.gates[start..end];
    let mut polys: Vec<Polynomial> = (0..n).map(|i| vec![1u64 << i]).collect();
    for &[a, b, c] in gates {
        let term = poly_and_not(&polys[c as usize], &polys[b as usize]);
        poly_xor_assign(&mut polys[a as usize], term);
        toggle_monomial(&mut polys[a as usize], 0u64);
    }
    polys
}

fn reference_to_polynomial_capped(
    circuit: &CircuitSeq,
    n: usize,
    start: usize,
    end: usize,
    cap: usize,
) -> Option<Vec<Polynomial>> {
    let gates = &circuit.gates[start..end];
    let mut polys: Vec<Polynomial> = (0..n).map(|i| vec![1u64 << i]).collect();
    for &[a, b, c] in gates {
        if polys[b as usize]
            .len()
            .saturating_mul(polys[c as usize].len())
            > cap.saturating_mul(16)
        {
            return None;
        }
        let term = poly_and_not(&polys[c as usize], &polys[b as usize]);
        poly_xor_assign(&mut polys[a as usize], term);
        toggle_monomial(&mut polys[a as usize], 0u64);
        if polys[a as usize].len() > cap {
            return None;
        }
    }
    Some(polys)
}

#[test]
fn opt_equiv_to_polynomial_scratch_reuse_matches_reference() {
    let mut rng = fastrand::Rng::with_seed(0x7363_7261_7463_6821);
    for _ in 0..200 {
        let n = rng.usize(3..=12);
        let m = rng.usize(0..=(3 * n));
        fastrand::seed(rng.u64(..));
        let circuit = random_circuit(n, m);
        let start = rng.usize(0..=circuit.gates.len());
        let end = rng.usize(start..=circuit.gates.len());
        assert_eq!(
            circuit.to_polynomial(n, start, end),
            reference_to_polynomial(&circuit, n, start, end),
            "n={n} m={m} start={start} end={end}"
        );
        for cap in [1usize, 2, 4, 8, 64, usize::MAX] {
            assert_eq!(
                circuit.to_polynomial_capped(n, start, end, cap),
                reference_to_polynomial_capped(&circuit, n, start, end, cap),
                "n={n} m={m} start={start} end={end} cap={cap}"
            );
        }
    }
}

// Merge-based input negation must match the historical per-rest
// binary-search toggles on normalized polynomials.
#[test]
fn opt_equiv_substitute_input_negation_matches_toggle_reference() {
    fn reference(poly: &mut Polynomial, w: usize) {
        let bit = 1u64 << w;
        let rests: Vec<Monomial> = poly
            .iter()
            .filter(|&&m| m & bit != 0)
            .map(|&m| m & !bit)
            .collect();
        for rest in rests {
            toggle_monomial(poly, rest);
        }
    }
    let mut rng = fastrand::Rng::with_seed(0x6e65_6761_7465_7331);
    for _ in 0..500 {
        let k = rng.usize(1..=12);
        let terms: Vec<Monomial> = (0..rng.usize(0..=40))
            .map(|_| rng.u64(..) & ((1u64 << k) - 1))
            .collect();
        let poly = polynomial_from_terms(terms);
        let w = rng.usize(0..k);
        let mut new_poly = poly.clone();
        let mut old_poly = poly.clone();
        substitute_input_negation(&mut new_poly, w);
        reference(&mut old_poly, w);
        assert_eq!(new_poly, old_poly, "k={k} w={w} poly={poly:?}");
        // Involution sanity: substituting the same wire twice restores
        // the input.
        substitute_input_negation(&mut new_poly, w);
        assert_eq!(new_poly, poly, "k={k} w={w}");
    }
}

// The hashed single-direction variant must agree with the plain variant
// on order and used wires, and its key must be exactly
// xxh3_128(polys_repr_blob(plain polys)) in both fresh and cached paths.
#[test]
fn opt_equiv_canonicalize_polys_single_hashed_matches_plain() {
    use xxhash_rust::xxh3::xxh3_128;
    let mut rng = fastrand::Rng::with_seed(0x6861_7368_6564_2101);
    for trial in 0..60 {
        let n = rng.usize(3..=10);
        let m = rng.usize(1..=(3 * n));
        fastrand::seed(rng.u64(..));
        let circuit = random_circuit(n, m);
        for reversed in [false, true] {
            let (hash, h_order, h_used) = circuit.canonicalize_polys_single_hashed(reversed);
            let (polys, p_order, p_used) = circuit.canonicalize_polys_single(reversed);
            assert_eq!(h_order.data, p_order.data, "trial={trial} rev={reversed}");
            assert_eq!(h_used, p_used);
            assert_eq!(
                hash,
                Some(xxh3_128(&polys_repr_blob(&polys)).to_le_bytes()),
                "trial={trial} rev={reversed}"
            );
            // Second call exercises the cached path when the cache is on.
            let (hash2, order2, used2) = circuit.canonicalize_polys_single_hashed(reversed);
            assert_eq!(hash2, hash);
            assert_eq!(order2.data, p_order.data);
            assert_eq!(used2, p_used);
        }
    }
    // Oversized windows keep the skip contract: no key, empty order.
    let gates: Vec<[u16; 3]> = (0..22u16)
        .map(|gate| [3 * gate, 3 * gate + 1, 3 * gate + 2])
        .collect();
    let big = CircuitSeq { gates };
    let (hash, order, used) = big.canonicalize_polys_single_hashed(false);
    assert_eq!(hash, None);
    assert!(order.data.is_empty());
    assert_eq!(used.len(), 66);
}

// The cached neg variant must match the historical uncached pipeline
// (per-input substitution in caller order) on fresh and hit paths, and an
// empty negation set must coincide with the plain canonicalization.
#[test]
fn opt_equiv_canon_single_neg_cache_matches_uncached_reference() {
    fn reference_neg(
        circuit: &CircuitSeq,
        negated_inputs: &[u16],
    ) -> (Vec<Polynomial>, Permutation, Vec<u16>) {
        let used = circuit.used_wires();
        if used.len() > 64 {
            return (Vec::new(), Permutation { data: Vec::new() }, used);
        }
        let wire_map = dense_wire_map(&used);
        let mut c = CircuitSeq {
            gates: circuit
                .gates
                .iter()
                .map(|&[t, c1, c2]| {
                    [
                        wire_map[t as usize],
                        wire_map[c1 as usize],
                        wire_map[c2 as usize],
                    ]
                })
                .collect(),
        };
        c.canonicalize();
        let n = c.max_wire() as usize + 1;
        let mut polys = c.to_polynomial(n, 0, c.gates.len());
        for &w in negated_inputs {
            let mapped = match wire_map.get(w as usize) {
                Some(&mw) if mw != u16::MAX => mw as usize,
                _ => continue,
            };
            for p in polys.iter_mut() {
                // Sequential toggle substitution, as the historical code
                // did.
                let bit = 1u64 << mapped;
                let rests: Vec<Monomial> = p
                    .iter()
                    .filter(|&&mm| mm & bit != 0)
                    .map(|&mm| mm & !bit)
                    .collect();
                for rest in rests {
                    toggle_monomial(p, rest);
                }
            }
        }
        let canon = match canonicalize_polys_4(polys, true) {
            Ok(canon) => canon,
            Err(()) => return (Vec::new(), Permutation { data: Vec::new() }, used),
        };
        (canon.0, canon.1, used)
    }

    let mut rng = fastrand::Rng::with_seed(0x6e65_675f_6361_6368);
    for trial in 0..60 {
        let n = rng.usize(3..=10);
        let m = rng.usize(1..=(3 * n));
        fastrand::seed(rng.u64(..));
        let circuit = random_circuit(n, m);
        // Random negation list: in-range wires (used or not), a possible
        // duplicate (involution), and a possible out-of-range wire to
        // exercise the skip filter.
        let mut negs: Vec<u16> = (0..rng.usize(0..=4))
            .map(|_| rng.u16(0..n as u16))
            .collect();
        if rng.bool() && !negs.is_empty() {
            let dup = negs[0];
            negs.push(dup);
        }
        if rng.bool() {
            negs.push(500);
        }
        let expected = reference_neg(&circuit, &negs);
        let first = circuit.canonicalize_polys_single_neg(&negs);
        let second = circuit.canonicalize_polys_single_neg(&negs);
        assert_eq!(first.0, expected.0, "trial={trial} negs={negs:?}");
        assert_eq!(first.1.data, expected.1.data, "trial={trial} negs={negs:?}");
        assert_eq!(first.2, expected.2);
        assert_eq!(second.0, expected.0, "hit path, trial={trial}");
        assert_eq!(second.1.data, expected.1.data);
        assert_eq!(second.2, expected.2);
        // Empty negation set must coincide with plain canonicalization.
        let plain = circuit.canonicalize_polys_single(false);
        let neg_empty = circuit.canonicalize_polys_single_neg(&[]);
        assert_eq!(neg_empty.0, plain.0, "trial={trial}");
        assert_eq!(neg_empty.1.data, plain.1.data);
        assert_eq!(neg_empty.2, plain.2);
    }
}

#[test]
fn to_polynomial_matches_evaluate_control_semantics() {
    let cases = [
        CircuitSeq {
            gates: vec![[0, 1, 2]],
        },
        CircuitSeq {
            gates: vec![[3, 0, 1], [1, 2, 3], [2, 3, 0], [0, 1, 4]],
        },
    ];

    for circuit in cases {
        let n = 5;
        let mask = (1usize << n) - 1;
        let polys = circuit.to_polynomial(n, 0, circuit.gates.len());
        for input in 0..(1usize << n) {
            assert_eq!(
                eval_polys(&polys, input) & mask,
                circuit.evaluate(input) & mask,
                "input={input:#b} circuit={:?}",
                circuit.gates
            );
        }
    }
}

fn canon_hash(seed: u64, n_wires: u16, gates: usize) -> u128 {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut c = CircuitSeq { gates: Vec::new() };
    while c.gates.len() < gates {
        let a = rng.random_range(0..n_wires);
        let b = rng.random_range(0..n_wires);
        let d = rng.random_range(0..n_wires);
        if a != b && a != d && b != d {
            c.gates.push([a, b, d]);
        }
    }
    let (polys, _, _) = c.canonicalize_polys_single(seed % 2 == 0);
    xxhash_rust::xxh3::xxh3_128(&polys_repr_blob(&polys))
}

// Golden canonical-form hashes. The canonical form defines every curated-DB key, so any
// change to these values means the DB has been silently invalidated. Do not regenerate
// casually. Regenerated 2026-07-18 after fixing a swapped-argument bug in to_polynomial
// (the g57 monomial was b*NOT(c); the executor and from_g57 use NOT(b)*c) — this realigns
// our DB keys with the source/upstream convention. Regenerate via the #[ignore]d
// regenerate_canon_golden test only when the canonical form legitimately changes.
#[test]
fn canonical_form_golden() {
    const GOLDEN: &[(&str, &str)] = &[
        ("G0", "6fa0209f74c6aca5a629ea4de7b882dd"),
        ("H0", "f8decf2b941d92e0016521e783bea5a8"),
        ("G1", "541946580565039aca15fdbd19755175"),
        ("H1", "25fb31deb42fd599dc90d9dea6fba89d"),
        ("G2", "5d551f5eba5474d33a871ecb0a8e46b7"),
        ("H2", "36a43931107c7d80f1fffbc5527c313a"),
        ("G3", "fd20b272d990c65207f9ffa0b3582b31"),
        ("H3", "1502ecfb3f88e59418564f2185548dce"),
        ("G4", "b6d47169d8e88efc8d965ff190d6c21f"),
        ("H4", "56986adc1dfa776e9de5480a80e9ab46"),
        ("G5", "fdf065eb905e344adb801bf88319a929"),
        ("H5", "e924c80a183b499d7b576bde07129131"),
        ("G6", "fbda8128fafdf7924880c95813641b7a"),
        ("H6", "245dd8d448f23884fbbac2b87aacaa86"),
        ("G7", "f7594dc87bdccb66a5e77d816b05b06a"),
        ("H7", "e015d3a5ec42bab54220110ad22f2612"),
        ("G8", "c29cd6d9be717edd6850b647f5370c0c"),
        ("H8", "602c1fad413f6816a32e4d299d096879"),
        ("G9", "e36265eba0672f79bbd54ed94c388564"),
        ("H9", "3fd134840c7913907227cf9205bdac9e"),
        ("G10", "d19251954f40b19ccb15c700f246276b"),
        ("H10", "59e94f3dedd110d92c1dae1e0a74cade"),
        ("G11", "b3cb06c6bb4c1f8f3f8e612e2622f2bf"),
        ("H11", "5168e7504437bd05203e558be87d7a5a"),
        ("G12", "3a2eebdc712c2cb9f8896f11169e4026"),
        ("H12", "c6ad8d964f866af06894dae44cad7911"),
        ("G13", "586eb625594255c9621e813a91669339"),
        ("H13", "2bfc4e6ff9208214514c899ffc57bb62"),
        ("G14", "1192d8249e2773dd389c2e8611bf256c"),
        ("H14", "d28943f055eb4a63aa4e4f4c4670ba13"),
        ("G15", "bca4629cf33574b8bd9eec6527e0c95f"),
        ("H15", "3523ee73c966859cabd1a5e626cdc51f"),
        ("G16", "2f0196c07918909d574b35e423bc60c7"),
        ("H16", "b76ea7f0d46c738a83d5ccdfacec678c"),
        ("G17", "1f0f68761b56f660f88c9ebdc2177bb7"),
        ("H17", "bd6e51d525fbeab79ec34b154ee398b4"),
        ("G18", "1d12417189ed645c0d68e74c85c3e4bd"),
        ("H18", "360c53cc42d3591e7c3558384e5d7a7f"),
        ("G19", "ac596d2604e6963258e9de7746f1efd7"),
        ("H19", "c24ea23211b28ac22faac1480c51289b"),
        ("G20", "e020d8c2d1dd740c9ef7e71d0ae52ac9"),
        ("H20", "a5b37a68dddb773429da7672d6c4870d"),
        ("G21", "39b636d94d180700abfe4e0b299e62c8"),
        ("H21", "c7b6d35f0b75ee1a3a23910657d13935"),
        ("G22", "b0a48bb2e87a13245269b1e309380d07"),
        ("H22", "53c37dd76b64c030b4c50b3ffd68a2c9"),
        ("G23", "631171335b8ff30b1a4406c1913cfed8"),
        ("H23", "acf0954bed92446336cd7dd618c46a22"),
        ("G24", "6d1677f347a24adaddd438be1758575f"),
        ("H24", "227c2822c97ec6a2add6ef947e57557b"),
        ("G25", "00c835f56a23564e21bc44105f1169b2"),
        ("H25", "5fd407309e7f74231d8cc0890f7d63f6"),
        ("G26", "783cd181e4f96d3bf9656d5aa0737b98"),
        ("H26", "d8dc7d0de786e49c90e97f77cebc19bf"),
        ("G27", "5d739004d2e1c3232660d923c341950f"),
        ("H27", "e854979b9d3fe892e12223aa67f261a1"),
        ("G28", "bcceaba13911caa90caddbd5c9090af3"),
        ("H28", "7a70480cb8f8a667167669f7a1128d14"),
        ("G29", "39c9d5702ff15e41a0504d0e6cbefe39"),
        ("H29", "ece8b5e185ca0c4cb818fc3fea4cf033"),
    ];
    for (tag, want) in GOLDEN {
        let (series, seed_s) = tag.split_at(1);
        let seed: u64 = seed_s.parse().unwrap();
        let got = match series {
            "G" => canon_hash(seed, 10, 8),
            _ => canon_hash(seed.wrapping_mul(0x9e37), 14, 12),
        };
        assert_eq!(
            format!("{:032x}", got),
            *want,
            "canonical form changed for {}",
            tag
        );
    }
}

#[test]
#[ignore]
fn regenerate_canon_golden() {
    for i in 0..30u64 {
        let g = canon_hash(i, 10, 8);
        let h = canon_hash(i.wrapping_mul(0x9e37), 14, 12);
        println!("(\"G{i}\", \"{:032x}\"),", g);
        println!("(\"H{i}\", \"{:032x}\"),", h);
    }
}

#[test]
fn substitute_input_negation_flips_that_variable() {
    // eval a polynomial (XOR of monomials; monomial bits = AND of those input vars; 0 = const).
    fn eval(poly: &Polynomial, x: u64) -> u8 {
        let mut v = 0u8;
        for &m in poly {
            if x & m == m {
                v ^= 1;
            }
        }
        v
    }
    let n = 4;
    for _ in 0..40 {
        let c = random_circuit(n, 8);
        let polys = c.to_polynomial(n, 0, c.gates.len());
        for w in 0..n {
            let mut flipped = polys.clone();
            for p in flipped.iter_mut() {
                substitute_input_negation(p, w);
            }
            // substituting x_w -> x_w+1 means flipped(x) == original(x ^ (1<<w)) for every wire.
            for x in 0..(1u64 << n) {
                for i in 0..n {
                    assert_eq!(
                        eval(&flipped[i], x),
                        eval(&polys[i], x ^ (1u64 << w)),
                        "w={w} x={x} wire={i}"
                    );
                }
            }
        }
    }
}

// The three canon4 scan configurations must be interchangeable: the
// legacy per-level rank rescan (fat entries, groups=None), the tied-group
// precompute (fat entries, groups=Some), and the compact-entry scan
// (deg<=16). Same split verdict, same split-group mask, same vr after.
#[test]
fn opt_equiv_canon4_scan_paths_agree() {
    let mut rng = StdRng::seed_from_u64(0x5ca9_2026);
    for trial in 0..400 {
        // Wide trials force degrees > 16 to exercise the fat fallback
        // beside the group precompute; narrow trials cover compact too.
        let wide = trial % 4 == 3;
        let n = if wide {
            rng.random_range(17..=22usize)
        } else {
            rng.random_range(4..=12usize)
        };
        let ranks = rng.random_range(1..=n);
        let vr0: Vec<usize> = (0..n).map(|_| rng.random_range(0..ranks)).collect();
        let terms = rng.random_range(1..=12usize);
        let mut cp: Vec<(Monomial, usize)> = (0..terms)
            .map(|_| {
                let m = if n >= 64 {
                    rng.random_range(0..u64::MAX)
                } else {
                    rng.random_range(0..(1u64 << n))
                };
                (m, 1usize)
            })
            .collect();
        coalesce_class_poly(&mut cp);
        let tied_mask = tied_mask_4(&vr0);
        let mut groups_meta = Vec::new();
        let mut groups_members = Vec::new();
        tied_groups_4(&vr0, n, &mut groups_meta, &mut groups_members);
        let groups = Some((groups_meta.as_slice(), groups_members.as_slice()));

        let scratch = || (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut le, mut f, mut t, mut s, mut sr) = scratch();
        let mut vr_legacy = vr0.clone();
        let res_legacy = scan_class_poly_levels_4(
            &cp,
            &mut vr_legacy,
            n,
            tied_mask,
            &mut le,
            &mut f,
            &mut t,
            &mut s,
            &mut sr,
            None,
        );
        let (mut le, mut f, mut t, mut s, mut sr) = scratch();
        let mut vr_groups = vr0.clone();
        let res_groups = scan_class_poly_levels_4(
            &cp,
            &mut vr_groups,
            n,
            tied_mask,
            &mut le,
            &mut f,
            &mut t,
            &mut s,
            &mut sr,
            groups,
        );
        assert_eq!(res_legacy, res_groups, "trial={trial} n={n}");
        assert_eq!(vr_legacy, vr_groups, "trial={trial} n={n}");

        let compact_ok = cp.iter().all(|&(m, _)| m.count_ones() <= 16);
        if compact_ok {
            let mut ec = Vec::new();
            let (_, mut f, mut t, mut s, mut sr) = scratch();
            let mut vr_c = vr0.clone();
            let res_c = scan_class_poly_levels_c(
                &cp, &mut vr_c, n, tied_mask, &mut ec, &mut f, &mut t, &mut s, &mut sr, groups,
            );
            assert_eq!(res_legacy, res_c, "compact trial={trial} n={n}");
            assert_eq!(vr_legacy, vr_c, "compact trial={trial} n={n}");
        }
    }
}

// Compact tiebreak-#1 keys must reproduce the fat keyed comparison
// exactly (ordering AND equality classes) whenever all degrees <= 16.
#[test]
fn opt_equiv_canon4_poly_key_compact_matches_fat() {
    let mut rng = StdRng::seed_from_u64(0x9e37_2026);
    for trial in 0..2000 {
        let n = rng.random_range(2..=14usize);
        let ranks = rng.random_range(1..=n);
        let vr: Vec<usize> = (0..n).map(|_| rng.random_range(0..ranks)).collect();
        let mut poly = |force_share: Option<&Polynomial>| -> Polynomial {
            // Bias toward shared monomials so equality cases actually occur.
            let terms = rng.random_range(0..=6usize);
            (0..terms)
                .map(|_| match force_share {
                    Some(other) if !other.is_empty() && rng.random_range(0..2u8) == 0 => {
                        other[rng.random_range(0..other.len())]
                    }
                    _ => rng.random_range(0..(1u64 << n)),
                })
                .collect()
        };
        let a = poly(None);
        let b = poly(Some(&a));
        let fat_a = poly_key_4(&a, &vr, n);
        let fat_b = poly_key_4(&b, &vr, n);
        let c_a = poly_key_c(&a, &vr);
        let c_b = poly_key_c(&b, &vr);
        assert_eq!(
            cmp_poly_key_c(&c_a, &c_b),
            fat_a.cmp(&fat_b),
            "trial={trial} n={n}"
        );
        assert_eq!(c_a == c_b, fat_a == fat_b, "trial={trial} n={n}");
        // Term order must match one-to-one: (degree, prefix) of fat terms.
        assert_eq!(
            c_a,
            fat_a
                .iter()
                .map(|k| (k.degree, k.prefix))
                .collect::<Vec<_>>(),
            "trial={trial} n={n}"
        );
    }
}

// End-to-end: the shipped compact/split2/cleanskip canonicalizer must
// stay deterministic and produce a canonical form invariant under wire
// relabeling — the property every curated-DB key relies on. Narrow trials
// run the compact path, wide trials (a forced degree-17+ monomial) run
// the fat fallback through the same driver.
#[test]
fn opt_equiv_canon4_form_invariant_under_relabeling() {
    let mut rng = StdRng::seed_from_u64(0xfab1e_2026);
    for trial in 0..72 {
        let wide = trial % 6 == 5;
        let n = if wide {
            rng.random_range(17..=20usize)
        } else {
            rng.random_range(3..=9usize)
        };
        let mut polys: Vec<Polynomial> = (0..n)
            .map(|_| {
                let terms = rng.random_range(1..=5usize);
                (0..terms)
                    .map(|_| rng.random_range(0..(1u64 << n)))
                    .collect()
            })
            .collect();
        if wide {
            // Guarantee a degree > 16 monomial so compact_ok is false.
            let w = rng.random_range(0..n);
            polys[w].push((1u64 << n) - 1);
        }
        let (canon_a, _) = canonicalize_polys_4(polys.clone(), true).unwrap();
        let (canon_again, _) = canonicalize_polys_4(polys.clone(), true).unwrap();
        assert_eq!(canon_a, canon_again, "determinism trial={trial} n={n}");

        let mut sigma: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = rng.random_range(0..=i);
            sigma.swap(i, j);
        }
        let mut relabeled: Vec<Polynomial> = vec![Vec::new(); n];
        for w in 0..n {
            relabeled[sigma[w]] = polys[w]
                .iter()
                .map(|&m| {
                    let mut r = 0u64;
                    let mut mm = m;
                    while mm != 0 {
                        let v = mm.trailing_zeros() as usize;
                        r |= 1u64 << sigma[v];
                        mm &= mm - 1;
                    }
                    r
                })
                .collect();
        }
        let (canon_b, _) = canonicalize_polys_4(relabeled, true).unwrap();
        assert_eq!(
            canon_a, canon_b,
            "relabeling trial={trial} n={n} sigma={sigma:?}"
        );
    }
}
