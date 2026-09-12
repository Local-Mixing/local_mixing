use super::*;
use crate::circuit::xgate::eval_u64;
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
                        let input = share_mask | ((share_mask ^ secret) << 1) | (random_mask << 2);
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
