//! Masking-safe native SAMF packets for heterogeneous circuits.
//!
//! A normal three-CNOT XOR swap temporarily writes `a XOR b` to one wire. If
//! `a` and `b` are the two carriers of one shared value, that intermediate is
//! the unmasked logical value. The packets here use a dedicated independent
//! random mask wire `r` and restore it exactly:
//!
//! ```text
//! a ^= r; a ^= b; b ^= a; a ^= b; b ^= r
//! ```
//!
//! The net action is `swap(a,b)`, while every prefix retains either the share
//! mask or `r`. Optional output negations give the four legacy signed-SAMF
//! types. Reversing a packet is its exact circuit inverse and is the native
//! unsamf operation.

use super::xgate::XGate;
use rand::Rng;

pub fn masked_swap_packet(a: u16, b: u16, random_mask: u16) -> Vec<XGate> {
    assert!(a != b && a != random_mask && b != random_mask);
    vec![
        XGate::cnot(a, random_mask),
        XGate::cnot(a, b),
        XGate::cnot(b, a),
        XGate::cnot(a, b),
        XGate::cnot(b, random_mask),
    ]
}

/// Signed swap using the same convention as legacy SAMFs after the swap:
/// 0=plain, 1=negate `a`, 2=negate `b`, 3=negate both.
pub fn signed_masked_swap_packet(
    a: u16,
    b: u16,
    random_mask: u16,
    negation_type: u16,
) -> Vec<XGate> {
    let mut packet = masked_swap_packet(a, b, random_mask);
    match negation_type {
        0 => {}
        1 => packet.push(XGate::x_gate(a)),
        2 => packet.push(XGate::x_gate(b)),
        3 => {
            packet.push(XGate::x_gate(a));
            packet.push(XGate::x_gate(b));
        }
        _ => panic!("invalid SAMF negation type {negation_type}"),
    }
    packet
}

pub fn inverse_packet(packet: &[XGate]) -> Vec<XGate> {
    packet.iter().rev().cloned().collect()
}

pub fn conjugate_gate_by_swap(gate: &XGate, a: u16, b: u16) -> XGate {
    let swap = |wire: u16| {
        if wire == a {
            b
        } else if wire == b {
            a
        } else {
            wire
        }
    };
    let mut controls = gate
        .ctrls
        .iter()
        .map(|&(wire, polarity)| (swap(wire), polarity))
        .collect::<super::xgate::Lits>();
    controls.sort_unstable();
    XGate {
        target: swap(gate.target),
        comp: gate.comp,
        ctrls: controls,
    }
}

/// Insert disjoint native SAMF/unsamf brackets after all heterogeneous
/// rewrites have finished. Interiors never touch `random_mask`, and processing
/// stops after insertion, so each opening packet starts with the independent
/// helper restored and each closing packet is its exact reverse.
pub fn insert_masked_swap_samfs(
    gates: &mut Vec<XGate>,
    data_wires: usize,
    random_mask: u16,
    requested: usize,
    rng: &mut impl Rng,
) -> usize {
    assert_eq!(random_mask as usize, data_wires);
    assert!(
        gates
            .iter()
            .all(|gate| gate.target != random_mask && !gate.reads(random_mask)),
        "dedicated SAMF mask wire is already in use"
    );
    if data_wires < 2 || gates.is_empty() || requested == 0 {
        return 0;
    }

    let original_len = gates.len();
    let count = requested.min(original_len);
    let chunk = original_len.div_ceil(count);
    let mut edits = Vec::with_capacity(count);
    for index in 0..count {
        let start = index * chunk;
        if start >= original_len {
            break;
        }
        let end = ((index + 1) * chunk).min(original_len);
        let touched: Vec<u16> = gates[start..end]
            .iter()
            .flat_map(|gate| {
                std::iter::once(gate.target).chain(gate.ctrls.iter().map(|&(wire, _)| wire))
            })
            .filter(|&wire| (wire as usize) < data_wires)
            .collect();
        if touched.is_empty() {
            continue;
        }
        let a = touched[rng.random_range(0..touched.len())];
        let b = loop {
            let candidate = rng.random_range(0..data_wires) as u16;
            if candidate != a {
                break candidate;
            }
        };
        edits.push((start, end, a, b));
    }

    for &(start, end, a, b) in edits.iter().rev() {
        for gate in &mut gates[start..end] {
            *gate = conjugate_gate_by_swap(gate, a, b);
        }
        let opening = masked_swap_packet(a, b, random_mask);
        let closing = inverse_packet(&opening);
        gates.splice(end..end, closing);
        gates.splice(start..start, opening);
    }
    edits.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postmix::xgate::eval_u64;
    use rand::SeedableRng;

    fn three_share_probe_leaks(gates: &[XGate]) -> (usize, usize, usize) {
        let evolutions: Vec<Vec<Vec<u64>>> = (0..2u64)
            .map(|secret| {
                (0..8u64)
                    .map(|randomness| {
                        let a = randomness & 1;
                        let b = (randomness >> 1) & 1;
                        let helper = (randomness >> 2) & 1;
                        let c = secret ^ a ^ b;
                        let mut state = a | (b << 1) | (c << 2) | (helper << 3);
                        let mut evolution = vec![state];
                        for gate in gates {
                            state = gate.apply_u64(state);
                            evolution.push(state);
                        }
                        evolution
                    })
                    .collect()
            })
            .collect();
        let wires = 4;
        let points = (gates.len() + 1) * wires;
        let observed = |evolution: &[u64], point: usize| -> usize {
            ((evolution[point / wires] >> (point % wires)) & 1) as usize
        };

        let singles = (0..points)
            .filter(|&point| {
                let count = |samples: &[Vec<u64>]| {
                    samples
                        .iter()
                        .map(|evolution| observed(evolution, point))
                        .sum::<usize>()
                };
                count(&evolutions[0]) != count(&evolutions[1])
            })
            .count();
        let mut same_prefix_pairs = 0;
        let mut all_space_time_pairs = 0;
        for left in 0..points {
            for right in left + 1..points {
                let histogram = |samples: &[Vec<u64>]| {
                    let mut counts = [0usize; 4];
                    for evolution in samples {
                        counts[observed(evolution, left) | (observed(evolution, right) << 1)] += 1;
                    }
                    counts
                };
                if histogram(&evolutions[0]) != histogram(&evolutions[1]) {
                    all_space_time_pairs += 1;
                    if left / wires == right / wires {
                        same_prefix_pairs += 1;
                    }
                }
            }
        }
        (singles, same_prefix_pairs, all_space_time_pairs)
    }

    #[test]
    fn masked_swap_and_every_signed_inverse_are_exact() {
        for negation_type in 0..=3 {
            let packet = signed_masked_swap_packet(0, 1, 2, negation_type);
            let inverse = inverse_packet(&packet);
            for input in 0..8u64 {
                let a = input & 1;
                let b = (input >> 1) & 1;
                let r = (input >> 2) & 1;
                let expected_a = b ^ ((negation_type & 1) != 0) as u64;
                let expected_b = a ^ ((negation_type & 2) != 0) as u64;
                let expected = expected_a | (expected_b << 1) | (r << 2);
                let swapped = eval_u64(&packet, input);
                assert_eq!(swapped, expected);
                assert_eq!(eval_u64(&inverse, swapped), input);
            }
        }
    }

    #[test]
    fn masked_swap_prefixes_never_expose_xor_of_complementary_shares() {
        for negation_type in 0..=3 {
            let packet = signed_masked_swap_packet(0, 1, 2, negation_type);
            for secret in 0..2u64 {
                for prefix in 0..=packet.len() {
                    let mut ones = [0usize; 3];
                    for share_mask in 0..2u64 {
                        for random_mask in 0..2u64 {
                            let input =
                                share_mask | ((share_mask ^ secret) << 1) | (random_mask << 2);
                            let output = eval_u64(&packet[..prefix], input);
                            for (wire, count) in ones.iter_mut().enumerate() {
                                *count += ((output >> wire) & 1) as usize;
                            }
                        }
                    }
                    assert_eq!(ones, [2; 3], "type={negation_type} prefix={prefix}");
                }
            }
        }
    }

    #[test]
    fn masked_swap_and_inverse_are_second_order_safe_for_three_shares() {
        for negation_type in 0..=3 {
            let packet = signed_masked_swap_packet(0, 1, 3, negation_type);
            assert_eq!(three_share_probe_leaks(&packet), (0, 0, 0));
            assert_eq!(three_share_probe_leaks(&inverse_packet(&packet)), (0, 0, 0));
        }
    }

    #[test]
    fn ordinary_three_cnot_swap_is_not_masking_safe() {
        let ordinary = [XGate::cnot(0, 1), XGate::cnot(1, 0), XGate::cnot(0, 1)];
        for secret in 0..2u64 {
            let values: Vec<u64> = (0..2u64)
                .map(|mask| eval_u64(&ordinary[..1], mask | ((mask ^ secret) << 1)) & 1)
                .collect();
            assert_eq!(values, vec![secret, secret]);
        }
    }

    #[test]
    fn disjoint_samf_brackets_preserve_an_arbitrary_fragment_circuit() {
        let original = vec![
            XGate::from_g57([0, 1, 2]),
            XGate::conj(2, [(0, false), (3, true)]).unwrap(),
            XGate::cnot(1, 3),
            XGate::x_gate(0),
            XGate::conj(3, [(1, true), (2, false)]).unwrap(),
        ];
        let mut transformed = original.clone();
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5a4f);
        let inserted = insert_masked_swap_samfs(&mut transformed, 4, 4, 3, &mut rng);
        assert_eq!(inserted, 3);
        assert_eq!(transformed.len(), original.len() + 10 * inserted);
        for input in 0..32u64 {
            assert_eq!(eval_u64(&transformed, input), eval_u64(&original, input));
        }
    }
}
