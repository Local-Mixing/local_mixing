use super::*;
use crate::circuit::xgate::eval_u64;
use rand::{SeedableRng, rngs::StdRng};

fn canonical_state() -> FeistalState {
    FeistalState {
        sharing: GadgetState {
            n: 3,
            pairs: vec![(0, 1), (3, 4), (6, 7)],
        },
        free: vec![2, 5, 8],
        q: vec![0, 1, 2],
    }
}

fn encode_two_share(logical: u64, masks: u64, pairs: &[(usize, usize)]) -> u64 {
    let mut physical = 0u64;
    for (index, &(p0_wire, p1_wire)) in pairs.iter().enumerate() {
        let p0 = (masks >> index) & 1;
        let p1 = ((logical >> index) & 1) ^ p0;
        physical |= p0 << p0_wire;
        physical |= p1 << p1_wire;
    }
    physical
}

fn decode_two_share(physical: u64, pairs: &[(usize, usize)]) -> u64 {
    pairs
        .iter()
        .enumerate()
        .fold(0, |logical, (index, &(p0, p1))| {
            logical | ((((physical >> p0) ^ (physical >> p1)) & 1) << index)
        })
}

fn decode_three_share(state: &FeistalState, physical: u64) -> u64 {
    (0..state.sharing.n).fold(0, |logical, index| {
        let (p0, p1) = state.sharing.pairs[index];
        let value = ((physical >> p0) ^ (physical >> p1) ^ (physical >> state.free[index])) & 1;
        logical | (value << index)
    })
}

fn encode_three_share(state: &FeistalState, logical: u64, masks: u64) -> u64 {
    let mut physical = 0u64;
    for index in 0..state.sharing.n {
        let p0 = (masks >> (2 * index)) & 1;
        let p1 = (masks >> (2 * index + 1)) & 1;
        let free = ((logical >> index) & 1) ^ p0 ^ p1;
        let (p0_wire, p1_wire) = state.sharing.pairs[index];
        physical |= p0 << p0_wire;
        physical |= p1 << p1_wire;
        physical |= free << state.free[index];
    }
    physical
}

/// Count secret-dependent single probes, same-prefix probe pairs, and all
/// space-time probe pairs. Randomness is enumerated exactly for each fixed
/// secret; unequal observation histograms are counted as leakage.
fn probe_leak_counts(
    gates: &[XGate],
    wires: usize,
    secret_count: usize,
    randomness_count: usize,
    encode: impl Fn(usize, usize) -> u64,
) -> (usize, usize, usize) {
    let evolutions: Vec<Vec<Vec<u64>>> = (0..secret_count)
        .map(|secret| {
            (0..randomness_count)
                .map(|randomness| {
                    let mut state = encode(secret, randomness);
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
    let points = (gates.len() + 1) * wires;
    let observed = |evolution: &[u64], point: usize| -> usize {
        ((evolution[point / wires] >> (point % wires)) & 1) as usize
    };

    let mut singles = 0usize;
    for point in 0..points {
        let base: usize = evolutions[0]
            .iter()
            .map(|evolution| observed(evolution, point))
            .sum();
        if evolutions[1..].iter().any(|samples| {
            samples
                .iter()
                .map(|evolution| observed(evolution, point))
                .sum::<usize>()
                != base
        }) {
            singles += 1;
        }
    }

    let mut same_prefix_pairs = 0usize;
    let mut space_time_pairs = 0usize;
    for left in 0..points {
        for right in left + 1..points {
            let histogram = |samples: &[Vec<u64>]| {
                let mut counts = [0usize; 4];
                for evolution in samples {
                    let value = observed(evolution, left) | (observed(evolution, right) << 1);
                    counts[value] += 1;
                }
                counts
            };
            let base = histogram(&evolutions[0]);
            if evolutions[1..]
                .iter()
                .any(|samples| histogram(samples) != base)
            {
                space_time_pairs += 1;
                if left / wires == right / wires {
                    same_prefix_pairs += 1;
                }
            }
        }
    }
    (singles, same_prefix_pairs, space_time_pairs)
}

fn algebraic_output_degrees(gates: &[XGate], wires: usize) -> Vec<usize> {
    assert!(wires < usize::BITS as usize);
    (0..wires)
        .map(|output_wire| {
            let mut anf: Vec<u8> = (0..1usize << wires)
                .map(|input| ((eval_u64(gates, input as u64) >> output_wire) & 1) as u8)
                .collect();
            for variable in 0..wires {
                for monomial in 0..1usize << wires {
                    if monomial & (1 << variable) != 0 {
                        anf[monomial] ^= anf[monomial ^ (1 << variable)];
                    }
                }
            }
            anf.iter()
                .enumerate()
                .filter_map(|(monomial, &coefficient)| {
                    (coefficient != 0).then_some(monomial.count_ones() as usize)
                })
                .max()
                .unwrap_or(0)
        })
        .collect()
}

fn assert_cnot_network_is_gate_locally_noncomplete(
    gates: &[XGate],
    wires: usize,
    share_groups: &[(usize, usize)],
) {
    let mut dependencies: Vec<u64> = (0..wires).map(|wire| 1u64 << wire).collect();
    for (index, gate) in gates.iter().enumerate() {
        assert!(
            !gate.comp && gate.width() == 1,
            "gate {index} is not a CNOT"
        );
        let target = gate.target as usize;
        let control = gate.ctrls[0].0 as usize;
        let gate_dependencies = dependencies[target] | dependencies[control];
        for &(share0, share1) in share_groups {
            let complete = (1u64 << share0) | (1u64 << share1);
            assert_ne!(
                gate_dependencies & complete,
                complete,
                "gate {index} consumes a complete sharing ({share0},{share1})"
            );
        }
        dependencies[target] ^= dependencies[control];
    }
}

#[test]
fn w_i_cnot_has_the_required_linear_map() {
    let mut gates = Vec::new();
    emit_w_i_cnot(0, 1, 2, 3, &mut gates);
    assert_eq!(gates.len(), 7);
    assert!(gates.iter().all(|gate| !gate.comp && gate.width() == 1));
    for input in 0..16u64 {
        let q0 = input & 1;
        let q1 = (input >> 1) & 1;
        let q2 = (input >> 2) & 1;
        let q3 = (input >> 3) & 1;
        let expected = q2 | (q3 << 1) | ((q0 ^ q1) << 2) | (q1 << 3);
        assert_eq!(eval_u64(&gates, input), expected);
    }
}

#[test]
fn four_fragment_two_share_sg_is_correct_and_prefix_masked() {
    let state = GadgetState {
        n: 3,
        pairs: vec![(0, 1), (2, 3), (4, 5)],
    };
    let mut gates = Vec::new();
    emit_gadget_x(&state, [0, 1, 2], &mut gates);
    assert_eq!(gates.len(), 4);
    assert!(gates.iter().all(|gate| !gate.comp && gate.width() == 2));
    for logical in 0..8u64 {
        for prefix in 0..=gates.len() {
            let mut ones = [0usize; 6];
            for masks in 0..8u64 {
                let input = encode_two_share(logical, masks, &state.pairs);
                let output = eval_u64(&gates[..prefix], input);
                for (wire, count) in ones.iter_mut().enumerate() {
                    *count += ((output >> wire) & 1) as usize;
                }
                if prefix == gates.len() {
                    assert_eq!(
                        decode_two_share(output, &state.pairs),
                        XGate::from_g57([0, 1, 2]).apply_u64(logical)
                    );
                }
            }
            assert_eq!(ones, [4; 6]);
        }
    }
}

#[test]
fn legacy_three_share_sg_retains_stronger_two_probe_boundary_security() {
    let state = canonical_state();
    let mut legacy = Vec::new();
    emit_sg3_x(&state, [0, 1, 2], &mut legacy);
    assert_eq!(legacy.len(), 9);
    assert!(legacy.iter().all(|gate| gate.comp && gate.width() == 2));

    // Rejected eight-gate candidate: temporarily reduce each control from
    // three carriers to two, use the four-fragment SG, then restore it.
    let mut rejected = Vec::new();
    emit_transvection_cnot(3, 4, &mut rejected);
    emit_transvection_cnot(6, 7, &mut rejected);
    emit_shared_g57_frag2(2, 3, 5, 6, 8, &mut rejected);
    emit_transvection_cnot(6, 7, &mut rejected);
    emit_transvection_cnot(3, 4, &mut rejected);
    assert_eq!(rejected.len(), 8);

    let legacy_leaks = probe_leak_counts(&legacy, 9, 8, 64, |secret, randomness| {
        encode_three_share(&state, secret as u64, randomness as u64)
    });
    let rejected_leaks = probe_leak_counts(&rejected, 9, 8, 64, |secret, randomness| {
        encode_three_share(&state, secret as u64, randomness as u64)
    });
    println!("three-share SG probes: legacy={legacy_leaks:?} rejected={rejected_leaks:?}");
    assert_eq!(legacy_leaks, (0, 0, 26));
    assert_eq!(rejected_leaks, (0, 12, 123));

    for logical in 0..8u64 {
        for prefix in 0..=legacy.len() {
            let mut ones = [0usize; 9];
            for masks in 0..64u64 {
                let input = encode_three_share(&state, logical, masks);
                let output = eval_u64(&legacy[..prefix], input);
                for (wire, count) in ones.iter_mut().enumerate() {
                    *count += ((output >> wire) & 1) as usize;
                }
                if prefix == legacy.len() {
                    assert_eq!(
                        decode_three_share(&state, output),
                        XGate::from_g57([0, 1, 2]).apply_u64(logical)
                    );
                }
            }
            assert_eq!(ones, [32; 9]);
        }
    }
}

#[test]
fn two_share_sg_probe_comparison_favors_the_four_fragment_variant() {
    let state = GadgetState {
        n: 3,
        pairs: vec![(0, 1), (2, 3), (4, 5)],
    };
    let mut old_g57 = Vec::new();
    emit_gadget(&state, [0, 1, 2], &mut old_g57);
    let old_g57: Vec<XGate> = old_g57.into_iter().map(XGate::from_g57).collect();
    let mut fragments = Vec::new();
    emit_gadget_x(&state, [0, 1, 2], &mut fragments);
    let old_leaks = probe_leak_counts(&old_g57, 6, 8, 8, |secret, randomness| {
        encode_two_share(secret as u64, randomness as u64, &state.pairs)
    });
    let new_leaks = probe_leak_counts(&fragments, 6, 8, 8, |secret, randomness| {
        encode_two_share(secret as u64, randomness as u64, &state.pairs)
    });
    println!("two-share SG probes: legacy={old_leaks:?} fragments={new_leaks:?}");
    assert_eq!(old_leaks, (0, 19, 151));
    assert_eq!(new_leaks, (0, 14, 73));

    // The old SG has gates reading both B carriers simultaneously. Every
    // new fragment reads at most one carrier from each logical control.
    assert!(old_g57.iter().any(|gate| gate.reads(2) && gate.reads(3)));
    assert!(
        fragments
            .iter()
            .all(|gate| !(gate.reads(2) && gate.reads(3)))
    );
    assert!(
        fragments
            .iter()
            .all(|gate| !(gate.reads(4) && gate.reads(5)))
    );
}

#[test]
fn selected_cnot_rg_variants_preserve_values_masks_and_noncompleteness() {
    for variant in 1..=2 {
        let original_pairs = vec![(0, 1), (2, 3)];
        let mut state = GadgetState {
            n: 2,
            pairs: original_pairs.clone(),
        };
        let mut gates = Vec::new();
        if variant == 1 {
            emit_rg1_x(&mut state, 0, 1, &mut gates);
            assert_eq!(gates.len(), 6);
        } else {
            emit_rg2_x(&mut state, 0, 1, &mut gates);
            assert_eq!(gates.len(), 3);
        }
        assert_cnot_network_is_gate_locally_noncomplete(&gates, 4, &original_pairs);
        for logical in 0..4u64 {
            for prefix in 0..=gates.len() {
                let mut ones = [0usize; 4];
                for masks in 0..4u64 {
                    let input = encode_two_share(logical, masks, &original_pairs);
                    let output = eval_u64(&gates[..prefix], input);
                    for (wire, count) in ones.iter_mut().enumerate() {
                        *count += ((output >> wire) & 1) as usize;
                    }
                    if prefix == gates.len() {
                        assert_eq!(decode_two_share(output, &state.pairs), logical);
                    }
                }
                assert_eq!(ones, [2; 4], "RG{variant} prefix={prefix}");
            }
        }
    }

    let pairs = vec![(0, 1), (2, 3), (4, 5)];
    let state = GadgetState {
        n: 3,
        pairs: pairs.clone(),
    };
    let mut gates = Vec::new();
    emit_rg3_x(&state, 0, 2, &mut gates);
    assert_eq!(gates.len(), 2);
    assert_cnot_network_is_gate_locally_noncomplete(&gates, 6, &pairs);
    for logical in 0..8u64 {
        for prefix in 0..=gates.len() {
            let mut ones = [0usize; 6];
            for masks in 0..8u64 {
                let input = encode_two_share(logical, masks, &pairs);
                let output = eval_u64(&gates[..prefix], input);
                for (wire, count) in ones.iter_mut().enumerate() {
                    *count += ((output >> wire) & 1) as usize;
                }
                if prefix == gates.len() {
                    assert_eq!(decode_two_share(output, &pairs), logical);
                }
            }
            assert_eq!(ones, [4; 6], "RG3 prefix={prefix}");
        }
    }
}

#[test]
fn six_cnot_rg1_is_minimal_under_gate_local_noncompleteness() {
    let start = [1u8, 2, 4, 8];
    let goal = |rows: [u8; 4]| rows[3] ^ rows[2] == 3 && rows[0] ^ rows[1] == 12;
    let noncomplete = |left: u8, right: u8| {
        let dependencies = left | right;
        dependencies & 3 != 3 && dependencies & 12 != 12
    };
    let mut queue = VecDeque::from([(start, 0usize)]);
    let mut seen = std::collections::HashSet::from([start]);
    while let Some((rows, depth)) = queue.pop_front() {
        assert!(!(depth <= 5 && depth != 0 && goal(rows)));
        if depth == 5 {
            continue;
        }
        for target in 0..4 {
            for control in 0..4 {
                if target == control || !noncomplete(rows[target], rows[control]) {
                    continue;
                }
                let mut next = rows;
                next[target] ^= next[control];
                if seen.insert(next) {
                    queue.push_back((next, depth + 1));
                }
            }
        }
    }

    let mut state = GadgetState {
        n: 2,
        pairs: vec![(0, 1), (3, 2)],
    };
    let mut selected = Vec::new();
    emit_rg1_x(&mut state, 0, 1, &mut selected);
    assert_eq!(selected.len(), 6);
}

#[test]
fn old_and_new_rg_probe_comparison_supports_cnot_replacements() {
    let make_two_share = |variant: usize, mixed: bool| {
        let mut state = GadgetState {
            n: 2,
            pairs: vec![(0, 1), (3, 2)],
        };
        if mixed {
            let mut gates = Vec::new();
            if variant == 1 {
                emit_rg1_x(&mut state, 0, 1, &mut gates);
            } else {
                emit_rg2_x(&mut state, 0, 1, &mut gates);
            }
            gates
        } else {
            let mut gates = Vec::new();
            if variant == 1 {
                emit_rg1(&mut state, 0, 1, &mut gates);
            } else {
                emit_rg2(&mut state, 0, 1, &mut gates);
            }
            gates.into_iter().map(XGate::from_g57).collect()
        }
    };
    let pairs = vec![(0, 1), (3, 2)];
    for variant in 1..=2 {
        let old = make_two_share(variant, false);
        let new = make_two_share(variant, true);
        let old_leaks = probe_leak_counts(&old, 4, 4, 4, |secret, randomness| {
            encode_two_share(secret as u64, randomness as u64, &pairs)
        });
        let new_leaks = probe_leak_counts(&new, 4, 4, 4, |secret, randomness| {
            encode_two_share(secret as u64, randomness as u64, &pairs)
        });
        println!("two-share RG{variant} probes: legacy={old_leaks:?} selected={new_leaks:?}");
        // The first legacy G57 in both RG1 and RG2 reads both carriers of
        // logical j. The selected CNOT networks are checked separately
        // above with evolving symbolic dependencies.
        assert!(old[0].reads(2) && old[0].reads(3));
        if variant == 1 {
            assert_eq!(old_leaks, (0, 14, 191));
            assert_eq!(new_leaks, (0, 10, 68));
        } else {
            // Legacy RG2 has secret-dependent 1/4-vs-3/4 bias at four
            // individual space-time probe locations.
            assert_eq!(old_leaks, (4, 21, 205));
            assert_eq!(new_leaks, (0, 6, 24));
        }
    }

    // In the Feistel representation y and its pair mask are random while
    // x is fixed. New RG1/RG2 retain second-order masking of x. Legacy
    // RG2 does not: seven space-time pairs have x-dependent histograms.
    let feistal_encode = |secret: usize, randomness: usize| {
        let x0 = secret & 1;
        let x1 = (secret >> 1) & 1;
        let y0 = randomness & 1;
        let m0 = (randomness >> 1) & 1;
        let y1 = (randomness >> 2) & 1;
        let m1 = (randomness >> 3) & 1;
        (m0 | ((m0 ^ y0) << 1) | ((x0 ^ y0) << 2) | (m1 << 3) | ((m1 ^ y1) << 4) | ((x1 ^ y1) << 5))
            as u64
    };
    for variant in 1..=2 {
        let old = make_two_share(variant, false);
        let new = make_two_share(variant, true);
        let remap = |gates: Vec<XGate>| {
            gates
                .into_iter()
                .map(|gate| {
                    let map = [0u16, 1, 4, 3];
                    let mut ctrls: crate::circuit::xgate::Lits = gate
                        .ctrls
                        .into_iter()
                        .map(|(wire, polarity)| (map[wire as usize], polarity))
                        .collect();
                    ctrls.sort_unstable();
                    XGate {
                        target: map[gate.target as usize],
                        comp: gate.comp,
                        ctrls,
                    }
                })
                .collect::<Vec<_>>()
        };
        let old_leaks = probe_leak_counts(&remap(old), 6, 4, 16, feistal_encode);
        let new_leaks = probe_leak_counts(&remap(new), 6, 4, 16, feistal_encode);
        println!("Feistal RG{variant} probes: legacy={old_leaks:?} selected={new_leaks:?}");
        if variant == 1 {
            assert_eq!(old_leaks, (0, 0, 0));
            assert_eq!(new_leaks, (0, 0, 0));
        } else {
            assert_eq!(old_leaks, (0, 1, 7));
            assert_eq!(new_leaks, (0, 0, 0));
        }
    }
}

#[test]
fn rg3_g57_and_selected_cnot_probe_comparison() {
    let pairs = vec![(0, 1), (2, 3), (4, 5)];
    let state = GadgetState {
        n: 3,
        pairs: pairs.clone(),
    };
    let mut legacy_g57 = Vec::new();
    emit_rg3(&state, 0, 2, 4, &mut legacy_g57);
    let legacy: Vec<XGate> = legacy_g57.into_iter().map(XGate::from_g57).collect();
    let mut selected = Vec::new();
    emit_rg3_x(&state, 0, 2, &mut selected);
    let legacy_leaks = probe_leak_counts(&legacy, 6, 8, 8, |secret, randomness| {
        encode_two_share(secret as u64, randomness as u64, &pairs)
    });
    let selected_leaks = probe_leak_counts(&selected, 6, 8, 8, |secret, randomness| {
        encode_two_share(secret as u64, randomness as u64, &pairs)
    });
    println!("two-share RG3 probes: legacy={legacy_leaks:?} selected={selected_leaks:?}");
    assert_eq!(legacy_leaks, (0, 9, 27));
    assert_eq!(selected_leaks, (0, 8, 22));
    assert_eq!(legacy.len(), selected.len());
    assert_eq!(algebraic_output_degrees(&legacy, 6)[..2], [2, 2]);
    assert_eq!(algebraic_output_degrees(&selected, 6)[..2], [1, 1]);

    // Survey the raw ordered (r1,r2) placement space. Production now
    // excludes the third carrier; every remaining placement is clean.
    let feistal = canonical_state();
    let (p0, p1) = feistal.sharing.pairs[0];
    let mut legacy_max = (0, 0, 0);
    let mut linear_max = (0, 0, 0);
    let mut legacy_unsafe = Vec::new();
    let mut linear_unsafe = Vec::new();
    for r1 in 0..9 {
        if r1 == p0 || r1 == p1 {
            continue;
        }
        for r2 in 0..9 {
            if r2 == p0 || r2 == p1 || r2 == r1 {
                continue;
            }
            let mut old_g57 = Vec::new();
            emit_rg3(&feistal.sharing, 0, r1, r2, &mut old_g57);
            let old: Vec<XGate> = old_g57.into_iter().map(XGate::from_g57).collect();
            let mut cnot = Vec::new();
            emit_rg3_x(&feistal.sharing, 0, r1, &mut cnot);
            let encode = |secret: usize, randomness: usize| {
                encode_three_share(&feistal, secret as u64, randomness as u64)
            };
            let old_leaks = probe_leak_counts(&old, 9, 8, 64, encode);
            let cnot_leaks = probe_leak_counts(&cnot, 9, 8, 64, encode);
            if r1 != feistal.free[0] && r2 != feistal.free[0] {
                assert_eq!(old_leaks, (0, 0, 0));
            }
            if r1 != feistal.free[0] {
                assert_eq!(cnot_leaks, (0, 0, 0));
            }
            if old_leaks.1 != 0 {
                legacy_unsafe.push((r1, r2, old_leaks));
            }
            if cnot_leaks.1 != 0 {
                linear_unsafe.push((r1, r2, cnot_leaks));
            }
            legacy_max.0 = legacy_max.0.max(old_leaks.0);
            legacy_max.1 = legacy_max.1.max(old_leaks.1);
            legacy_max.2 = legacy_max.2.max(old_leaks.2);
            linear_max.0 = linear_max.0.max(cnot_leaks.0);
            linear_max.1 = linear_max.1.max(cnot_leaks.1);
            linear_max.2 = linear_max.2.max(cnot_leaks.2);
        }
    }
    println!("three-share RG3 placement maxima: legacy={legacy_max:?} selected={linear_max:?}");
    println!(
        "three-share RG3 unsafe placements: legacy={legacy_unsafe:?} selected={linear_unsafe:?}"
    );
    assert_eq!(legacy_max, (0, 1, 5));
    assert_eq!(linear_max, (0, 1, 5));
    assert_eq!(legacy_unsafe.len(), 12);
    assert_eq!(linear_unsafe.len(), 6);
}

#[test]
fn selected_rg_networks_are_deliberately_linear() {
    let make = |variant: usize| {
        let mut state = GadgetState {
            n: 2,
            pairs: vec![(0, 1), (3, 2)],
        };
        let mut gates = Vec::new();
        if variant == 1 {
            emit_rg1_x(&mut state, 0, 1, &mut gates);
        } else {
            emit_rg2_x(&mut state, 0, 1, &mut gates);
        }
        gates
    };
    assert_eq!(algebraic_output_degrees(&make(1), 4), vec![1; 4]);
    assert_eq!(algebraic_output_degrees(&make(2), 4), vec![1; 4]);
}

#[test]
fn feistal_cnot_rgs_preserve_overlapping_x_y_and_prefix_masking() {
    for variant in 1..=3 {
        let initial_pairs = vec![(0, 1), (3, 4), (6, 7)];
        let mut state = canonical_state();
        state.q = vec![0, 1, 2];
        let mut gates = Vec::new();
        match variant {
            1 => emit_rg1_x(&mut state.sharing, 0, 1, &mut gates),
            2 => emit_rg2_x(&mut state.sharing, 0, 1, &mut gates),
            _ => emit_rg3_x(&state.sharing, 0, state.free[1], &mut gates),
        }
        for x in 0..8u64 {
            for prefix in 0..=gates.len() {
                let mut ones = [0usize; 9];
                for y in 0..8u64 {
                    for masks in 0..8u64 {
                        let mut input = 0u64;
                        for index in 0..3 {
                            let p0 = (masks >> index) & 1;
                            let y_bit = (y >> index) & 1;
                            let free = ((x >> index) & 1) ^ y_bit;
                            input |= p0 << initial_pairs[index].0;
                            input |= (p0 ^ y_bit) << initial_pairs[index].1;
                            input |= free << state.free[index];
                        }
                        let output = eval_u64(&gates[..prefix], input);
                        for (wire, count) in ones.iter_mut().enumerate() {
                            *count += ((output >> wire) & 1) as usize;
                        }
                        if prefix == gates.len() {
                            assert_eq!(decode_three_share(&state, output), x);
                            let decoded_y = (0..3).fold(0u64, |value, host| {
                                let (p0, p1) = state.sharing.pairs[host];
                                value | ((((output >> p0) ^ (output >> p1)) & 1) << state.q[host])
                            });
                            assert_eq!(decoded_y, y);
                        }
                    }
                }
                assert_eq!(ones, [32; 9], "Feistal RG{variant} prefix={prefix}");
            }
        }
    }
}

#[test]
fn two_share_homomorphic_cnot_is_correct_and_prefix_masked() {
    let gates = homomorphic_cnot2((0, 1), (2, 3));
    assert_eq!(gates.len(), 2);
    assert_cnot_network_is_gate_locally_noncomplete(&gates, 4, &[(0, 1), (2, 3)]);
    for logical in 0..4u64 {
        let a = logical & 1;
        let b = (logical >> 1) & 1;
        for prefix in 0..=gates.len() {
            let mut ones = [0usize; 4];
            for masks in 0..4u64 {
                let a0 = masks & 1;
                let b0 = (masks >> 1) & 1;
                let input = a0 | ((a ^ a0) << 1) | (b0 << 2) | ((b ^ b0) << 3);
                let output = eval_u64(&gates[..prefix], input);
                for (wire, count) in ones.iter_mut().enumerate() {
                    *count += ((output >> wire) & 1) as usize;
                }
                if prefix == gates.len() {
                    assert_eq!(((output >> 0) ^ (output >> 1)) & 1, a ^ b);
                    assert_eq!(((output >> 2) ^ (output >> 3)) & 1, b);
                }
            }
            assert_eq!(ones, [2; 4]);
        }
    }
}

#[test]
fn three_share_homomorphic_cnot_is_correct_and_prefix_masked() {
    let state = canonical_state();
    let gates = homomorphic_cnot3((0, 1, 2), (3, 4, 5));
    assert_eq!(gates.len(), 3);
    for logical in 0..8u64 {
        for prefix in 0..=gates.len() {
            let mut ones = [0usize; 9];
            for masks in 0..64u64 {
                let input = encode_three_share(&state, logical, masks);
                let output = eval_u64(&gates[..prefix], input);
                for (wire, count) in ones.iter_mut().enumerate() {
                    *count += ((output >> wire) & 1) as usize;
                }
                if prefix == gates.len() {
                    let decoded = decode_three_share(&state, output);
                    let expected = logical ^ (((logical >> 1) & 1) << 0);
                    assert_eq!(decoded, expected);
                }
            }
            assert_eq!(ones, [32; 9]);
        }
    }
}

#[test]
fn shared_fragments_compute_without_unmasking_any_single_carrier() {
    let state = canonical_state();
    let logical_gates = [
        XGate::cnot(0, 1),
        XGate::conj(2, [(0, false)]).unwrap(),
        XGate::conj(1, [(0, true), (2, false)]).unwrap(),
        XGate::from_g57([0, 1, 2]),
    ];
    for logical_gate in logical_gates {
        let mut physical_gates = Vec::new();
        emit_shared_fragment3(&state, &logical_gate, &mut physical_gates);
        for logical in 0..8u64 {
            for masks in 0..64u64 {
                let encoded = encode_three_share(&state, logical, masks);
                let result = eval_u64(&physical_gates, encoded);
                assert_eq!(
                    decode_three_share(&state, result),
                    logical_gate.apply_u64(logical)
                );
            }

            // At every physical-gate prefix, every individual carrier is
            // exactly balanced over the masks for each fixed secret.
            for prefix in 0..=physical_gates.len() {
                let mut ones = [0usize; 9];
                for masks in 0..64u64 {
                    let encoded = encode_three_share(&state, logical, masks);
                    let result = eval_u64(&physical_gates[..prefix], encoded);
                    for (wire, count) in ones.iter_mut().enumerate() {
                        *count += ((result >> wire) & 1) as usize;
                    }
                }
                assert_eq!(ones, [32; 9]);
            }
        }
    }
}

#[test]
fn feistal_n_cnot_keeps_every_prefix_first_order_masked() {
    for q in [
        vec![0, 1, 2],
        vec![0, 2, 1],
        vec![1, 0, 2],
        vec![1, 2, 0],
        vec![2, 0, 1],
        vec![2, 1, 0],
    ] {
        let mut state = canonical_state();
        state.q = q;
        let mut gates = Vec::new();
        emit_feistal_n_cnot(&state, &mut gates);
        for logical_x in 0..8u64 {
            for prefix in 0..=gates.len() {
                let mut ones = [0usize; 9];
                for masks in 0..64u64 {
                    // Independent p0/p1 choices are equivalent to averaging
                    // over the Feistel y values and their random pair masks
                    // for this fixed original x.
                    let input = encode_three_share(&state, logical_x, masks);
                    let output = eval_u64(&gates[..prefix], input);
                    for (wire, count) in ones.iter_mut().enumerate() {
                        *count += ((output >> wire) & 1) as usize;
                    }
                }
                assert_eq!(ones, [32; 9], "q={:?} prefix={prefix}", state.q);
            }
        }
    }
}

#[test]
fn gadgetize_cnot_preserves_the_first_n_wires() {
    let n = 3;
    let main = CircuitSeq {
        gates: vec![[0, 1, 2], [2, 0, 1], [1, 2, 0]],
    };
    for seed in 0..8u64 {
        let mut rng = StdRng::seed_from_u64(0xc001_0000 + seed);
        let transformed = gadgetize_cnot(
            &main,
            n,
            2,
            &MaskConfig::off(),
            &ProdConfig::off(),
            &mut rng,
        );
        assert_eq!(transformed.num_wires, 2 * n);
        let mask = (1u64 << n) - 1;
        for input in 0..(1u64 << (2 * n)) {
            let expected = main.evaluate((input & mask) as usize) as u64 & mask;
            assert_eq!(eval_u64(&transformed.gates, input) & mask, expected);
        }
    }
}

/// An n=4 body (distinct operands per triple) — wide enough to leave a
/// live source pool under partial coverage, and long enough for the mask
/// top-up to run past its taper.
const MASKED_TEST_N: usize = 4;
fn masked_test_main() -> CircuitSeq {
    let mut gates = Vec::new();
    for _ in 0..4 {
        for &g in &[
            [0u16, 1, 2],
            [3, 2, 0],
            [1, 3, 2],
            [2, 0, 3],
            [0, 3, 1],
            [3, 1, 0],
        ] {
            gates.push(g);
        }
    }
    CircuitSeq { gates }
}

/// A random-ish g57 body on `n >= 4` wires, for tests that need a width
/// the 4-wire fixture cannot give (a wire census wants room for a band).
fn masked_test_main_wide(n: usize) -> CircuitSeq {
    let mut rng = StdRng::seed_from_u64(0x9e37_9b91);
    let mut gates = Vec::new();
    for _ in 0..4 * n {
        let a = rng.random_range(0..n) as u16;
        let b = loop {
            let w = rng.random_range(0..n) as u16;
            if w != a {
                break w;
            }
        };
        let c = loop {
            let w = rng.random_range(0..n) as u16;
            if w != a && w != b {
                break w;
            }
        };
        gates.push([a, b, c]);
    }
    CircuitSeq { gates }
}

fn masked_test_config() -> MaskConfig {
    // cov 0.75 keeps an unmasked source pool alive on this width.
    MaskConfig {
        cov: 0.75,
        k: 2,
        depth: 2,
        taper: Some(0),
    }
}

#[test]
fn emit_poly_add_realizes_random_polynomials() {
    let total = 8usize;
    let target = 3u16;
    let mut rng = StdRng::seed_from_u64(0x901f_0000);
    for _ in 0..200 {
        let mut poly = WirePoly::default();
        for _ in 0..rng.random_range(1..5usize) {
            let size = rng.random_range(0..=3usize);
            let mut m = Vec::new();
            while m.len() < size {
                let w = rng.random_range(0..total) as u16;
                if w != target && !m.contains(&w) {
                    m.push(w);
                }
            }
            poly.toggle(m);
        }
        let mut gates = Vec::new();
        emit_poly_add(target, &poly, total, &mut rng, &mut gates);
        assert!(
            gates.iter().all(|g| !g.ctrls.is_empty()),
            "emit_poly_add must never emit a bare X"
        );
        for input in 0..(1u64 << total) {
            let expected = input ^ ((poly.eval_u64(input) as u64) << target);
            assert_eq!(
                eval_u64(&gates, input),
                expected,
                "poly={poly:?} input={input:#x}"
            );
        }
    }
}

/// XOR of a value's two carrier bits in a packed state — the only
/// gadget-visible quantity; mask compensation preserves this, not the
/// individual carriers (a flush may target either carrier, dirtying both
/// by an equal amount like an RG3 refresh).
fn pair_xor(state: u64, pair: (usize, usize)) -> u64 {
    ((state >> pair.0) ^ (state >> pair.1)) & 1
}

#[test]
fn mask_inject_then_flush_preserves_every_value() {
    // Inject a stack per value, then flush_all: each value's pair-XOR must
    // be restored for every input and every realization draw. The
    // value-sourced cascade-free ledger keeps every source value at its
    // injection value, so the compensation is exact regardless of the
    // (either-carrier) flush order.
    let n = 4;
    let total = 2 * n;
    let pairs: Vec<(usize, usize)> = (0..n).map(|v| (v, v + n)).collect();
    // No value is ever a CG target here (pure ledger exercise), so every
    // value is an ideal (never-disturbed) source.
    let targets = vec![Vec::new(); n];
    for seed in 0..64u64 {
        let mut rng = StdRng::seed_from_u64(0x1f1a_0000 + seed);
        let state = GadgetState {
            n,
            pairs: pairs.clone(),
        };
        // cov 0.75 keeps a live source pool; k=2 stacks.
        let cfg = MaskConfig {
            cov: 0.75,
            k: 2,
            depth: 2,
            taper: Some(0),
        };
        let mut ledger = MaskLedger::new(n, &cfg, targets.clone(), &mut rng);
        let mut gates = Vec::new();
        for _round in 0..3 {
            for value in 0..n {
                ledger.inject(value, 0, &state, total, &mut rng, &mut gates);
            }
        }
        ledger.flush_all(&state, total, &mut rng, &mut gates);
        assert!(ledger.masks.is_empty());
        assert!(ledger.injected > 0, "seed={seed}: nothing injected");
        for input in 0..(1u64 << total) {
            let out = eval_u64(&gates, input);
            for &p in &pairs {
                assert_eq!(
                    pair_xor(out, p),
                    pair_xor(input, p),
                    "seed={seed} pair={p:?}"
                );
            }
        }
    }
}

#[test]
fn mask_peek_bracket_restores_the_masked_value() {
    // Un-mask (before_cg) then re-mask (after_cg) around a value read must
    // (a) leave the value's pair-XOR exactly masked again afterward, and
    // (b) expose the TRUE value in between (what the vanilla CG needs).
    let n = 4;
    let total = 2 * n;
    let pairs: Vec<(usize, usize)> = (0..n).map(|v| (v, v + n)).collect();
    let targets = vec![Vec::new(); n];
    for seed in 0..64u64 {
        let mut rng = StdRng::seed_from_u64(0x9ee0_0000 + seed);
        let state = GadgetState {
            n,
            pairs: pairs.clone(),
        };
        let cfg = MaskConfig {
            cov: 0.75,
            k: 1,
            depth: 2,
            taper: Some(0),
        };
        let mut ledger = MaskLedger::new(n, &cfg, targets.clone(), &mut rng);
        // Mask value 0 (sources chosen from the unmasked pool 1..4).
        let mut pre = Vec::new();
        if !ledger.inject(0, 0, &state, total, &mut rng, &mut pre) {
            continue;
        }
        // Peek value 0 as a read of a gate whose target is NOT a source of
        // value 0's mask — otherwise before_cg legitimately flushes it
        // (source recomputed) instead of peeking, a different path.
        let (s0, s1) = ledger.masks[0].sources;
        let target = (1..n).find(|t| *t != s0 && *t != s1).unwrap();
        let mut br_open = Vec::new();
        ledger.before_cg(&[0], target, &state, total, &mut rng, &mut br_open);
        let mut br_close = Vec::new();
        ledger.after_cg(&[0], &state, total, &mut rng, &mut br_close);
        for input in 0..(1u64 << total) {
            let after_inject = eval_u64(&pre, input);
            // Inside the bracket: value 0 reads TRUE (un-masked).
            let mut peeked = after_inject;
            peeked = eval_u64(&br_open, peeked);
            assert_eq!(
                pair_xor(peeked, pairs[0]),
                pair_xor(input, pairs[0]),
                "seed={seed}: value not reconstructed inside the peek"
            );
            // After re-mask: value 0 is masked again (== state after inject).
            let closed = eval_u64(&br_close, peeked);
            assert_eq!(
                pair_xor(closed, pairs[0]),
                pair_xor(after_inject, pairs[0]),
                "seed={seed}: peek not undone"
            );
        }
    }
}

#[test]
fn masked_gadgetize_cnot_preserves_the_first_n_wires() {
    let n = MASKED_TEST_N;
    let main = masked_test_main();
    let mask = (1u64 << n) - 1;
    for seed in 0..8u64 {
        let mut rng = StdRng::seed_from_u64(0x3a5c_0000 + seed);
        let masked = gadgetize_cnot(
            &main,
            n,
            2,
            &masked_test_config(),
            &ProdConfig::off(),
            &mut rng,
        );
        assert_eq!(masked.num_wires, 2 * n);
        assert!(
            masked.gates.iter().all(|g| !g.ctrls.is_empty()),
            "masked body must not contain a bare X"
        );
        for input in 0..(1u64 << (2 * n)) {
            let expected = main.evaluate((input & mask) as usize) as u64 & mask;
            assert_eq!(eval_u64(&masked.gates, input) & mask, expected);
        }
        // Same seed, masks off: the masked build must actually have paid
        // mask gates into the body.
        let mut rng = StdRng::seed_from_u64(0x3a5c_0000 + seed);
        let plain = gadgetize_cnot(
            &main,
            n,
            2,
            &MaskConfig::off(),
            &ProdConfig::off(),
            &mut rng,
        );
        assert!(
            masked.gates.len() > plain.gates.len(),
            "seed={seed}: masks enabled but no mask gates emitted"
        );
    }
}

#[test]
fn masked_slice_zero_gadgetize_matches_on_the_zero_slice() {
    let n = MASKED_TEST_N;
    let main = masked_test_main();
    let mask = (1u64 << n) - 1;
    for seed in 0..4u64 {
        let mut rng = StdRng::seed_from_u64(0x3a5d_0000 + seed);
        let transformed = gadgetize_with_slice_zero_ccnot(
            &main,
            n,
            2,
            6 * n,
            &masked_test_config(),
            &ProdConfig::off(),
            &mut rng,
        );
        for x in 0..=mask {
            let expected = main.evaluate(x as usize) as u64 & mask;
            assert_eq!(eval_u64(&transformed.gates, x) & mask, expected);
        }
    }
}

#[test]
fn masked_gadgetize_xgates_preserves_the_low_wires() {
    let n = MASKED_TEST_N;
    let mask = (1u64 << n) - 1;
    let source = vec![
        XGate::from_g57([0, 1, 2]),
        XGate::cnot(0, 3),
        XGate::conj(2, [(0u16, true), (1u16, true)]).unwrap(),
        XGate::conj(1, [(3u16, false)]).unwrap(),
        XGate::from_g57([3, 0, 1]),
        XGate::cnot(1, 0),
        XGate::from_g57([0, 2, 3]),
        XGate::conj(0, [(1u16, false), (2u16, true)]).unwrap(),
        XGate::from_g57([1, 3, 2]),
        XGate::cnot(2, 1),
    ];
    for seed in 0..8u64 {
        let mut rng = StdRng::seed_from_u64(0x3a5e_0000 + seed);
        let g = gadgetize_xgates(
            &source,
            n,
            2,
            &masked_test_config(),
            &ProdConfig::off(),
            &mut rng,
        );
        assert_eq!(g.num_wires, 2 * n);
        for input in 0..(1u64 << (2 * n)) {
            let expected = eval_u64(&source, input & mask) & mask;
            assert_eq!(
                eval_u64(&g.gates, input) & mask,
                expected,
                "input={input:#x}"
            );
        }
    }
}

fn prod_test_config() -> ProdConfig {
    ProdConfig {
        k: 2,
        deg: 2,
        k_hi: 0,
        deg_hi: 3,
        band: 6,
        rsrc: 1,
        ..ProdConfig::off()
    }
}

/// Test-side decode under the product-share ledger state: pair-XOR of the
/// value's carriers, XOR each registered slot's product PROD(w_j ^ a_j),
/// XOR c.
fn prod_decode(
    state: u64,
    value: usize,
    pairs: &[(usize, usize)],
    slots: &[Vec<ProdSlot>],
    consts: &[bool],
    loc: &[u16],
) -> u64 {
    // Single-carrier builds record `pairs[v] = (w, w)`, and XORing the wire
    // with itself would decode every value as 0 -- so read one carrier when
    // the pair collapses, both when it does not.
    let (c0, c1) = pairs[value];
    let mut v = if c0 == c1 {
        (state >> c0) & 1
    } else {
        ((state >> c0) ^ (state >> c1)) & 1
    };
    for slot in &slots[value] {
        // Factors name band VARIABLES; `loc` says where each one lives.
        let factor = slot
            .factors
            .iter()
            .all(|&(b, a)| ((state >> loc[b as usize]) & 1 != 0) ^ a);
        v ^= factor as u64;
    }
    v ^ consts[value] as u64
}

fn eval_anf_atoms(state: u64, atoms: &[Vec<(u16, bool)>]) -> u64 {
    atoms.iter().fold(0u64, |value, atom| {
        value
            ^ atom
                .iter()
                .all(|&(wire, polarity)| ((state >> wire) & 1 != 0) == polarity)
                as u64
    })
}

/// Truth table of every physical wire at every circuit prefix, packed one
/// column at a time.  Keeping the whole Boolean function (rather than a
/// sample or a correlation) lets the space-time checks below prove an
/// identity over every possible incoming dirty state.
fn prefix_wire_signatures(gates: &[XGate], wires: usize) -> Vec<Vec<u64>> {
    let rows = 1usize << wires;
    let words = (rows + 63) / 64;
    let mut columns = vec![vec![0u64; words]; (gates.len() + 1) * wires];
    for input in 0..rows {
        let mut physical = input as u64;
        for prefix in 0..=gates.len() {
            for wire in 0..wires {
                if (physical >> wire) & 1 != 0 {
                    columns[prefix * wires + wire][input / 64] |= 1u64 << (input % 64);
                }
            }
            if let Some(gate) = gates.get(prefix) {
                physical = gate.apply_u64(physical);
            }
        }
    }
    columns
}

fn prod_decode_signature(
    wires: usize,
    value: usize,
    pairs: &[(usize, usize)],
    slots: &[Vec<ProdSlot>],
    consts: &[bool],
    loc: &[u16],
) -> Vec<u64> {
    let rows = 1usize << wires;
    let mut signature = vec![0u64; (rows + 63) / 64];
    for input in 0..rows {
        if prod_decode(input as u64, value, pairs, slots, consts, loc) != 0 {
            signature[input / 64] |= 1u64 << (input % 64);
        }
    }
    signature
}

fn xor_signatures(left: &[u64], right: &[u64]) -> Vec<u64> {
    left.iter().zip(right).map(|(&a, &b)| a ^ b).collect()
}

fn signature_pivot(value: &[u64]) -> Option<usize> {
    value.iter().enumerate().rev().find_map(|(word, &bits)| {
        (bits != 0).then(|| word * 64 + (63 - bits.leading_zeros() as usize))
    })
}

/// Does `target` belong to the affine span of these simultaneous wires?
fn affine_span_contains(columns: &[Vec<u64>], target: &[u64]) -> bool {
    let bit_count = target.len() * 64;
    let mut basis: Vec<Option<Vec<u64>>> = vec![None; bit_count];
    let mut insert = |mut value: Vec<u64>| {
        while let Some(pivot) = signature_pivot(&value) {
            if let Some(row) = &basis[pivot] {
                for (word, &rhs) in value.iter_mut().zip(row) {
                    *word ^= rhs;
                }
            } else {
                basis[pivot] = Some(value);
                return;
            }
        }
    };
    for column in columns {
        insert(column.clone());
    }
    insert(vec![u64::MAX; target.len()]);

    let mut residual = target.to_vec();
    while let Some(pivot) = signature_pivot(&residual) {
        let Some(row) = &basis[pivot] else {
            return false;
        };
        for (word, &rhs) in residual.iter_mut().zip(row) {
            *word ^= rhs;
        }
    }
    true
}

/// Find the specific short space-time identity at issue:
///
/// logical operand = its entry carrier XOR wire@p XOR wire@q XOR constant.
///
/// This search is deliberately blind to which wires the fold borrowed and
/// where its gather/strip boundaries are.
fn same_wire_space_time_witness(
    trace: &[Vec<u64>],
    wires: usize,
    needed_delta: &[u64],
) -> Option<(usize, usize, usize, bool)> {
    let prefixes = trace.len() / wires;
    for wire in 0..wires {
        for left in 0..prefixes {
            for right in left + 1..prefixes {
                let delta =
                    xor_signatures(&trace[left * wires + wire], &trace[right * wires + wire]);
                if delta == needed_delta {
                    return Some((wire, left, right, false));
                }
                if delta
                    .iter()
                    .zip(needed_delta)
                    .all(|(&observed, &needed)| observed == !needed)
                {
                    return Some((wire, left, right, true));
                }
            }
        }
    }
    None
}

#[test]
fn prod_fold_cg_applies_the_virtual_gate_share_natively() {
    // Manually built ledger; fold one g57, one CNOT, one CCNOT-with-
    // polarity, one X: for every input, the target's decode transitions
    // by exactly the virtual gate while every other value is untouched —
    // and no emitted gate writes anything but the target's carriers.
    let n = 3;
    let carrier_total = 2 * n;
    let pairs = vec![(0usize, 1usize), (2, 3), (4, 5)];
    let band = 4usize; // wires 6..10
    let total = carrier_total + band;
    let cfg = ProdConfig {
        k: 1,
        deg: 2,
        k_hi: 0,
        deg_hi: 3,
        band,
        rsrc: 0,
        ..ProdConfig::off()
    };
    let sources: Vec<XGate> = vec![
        XGate::from_g57([0, 1, 2]),
        XGate::cnot(1, 2),
        XGate::conj(2, [(0u16, false), (1u16, true)]).unwrap(),
        XGate::x_gate(0),
    ];
    for seed in 0..32u64 {
        let mut rng = StdRng::seed_from_u64(0x9d0d_0000 + seed);
        let state = GadgetState {
            n,
            pairs: pairs.clone(),
        };
        let mut ledger = ProdLedger::new(n, &cfg, carrier_total, None);
        let mut ramp = Vec::new();
        ledger.inject_all(&state, &mut rng, &mut ramp);
        for gate in &sources {
            let slots_before = ledger.slots.clone();
            let consts_before = ledger.consts.clone();
            let mut fold = Vec::new();
            ledger.fold_cg(gate, &state, &mut rng, &mut fold);
            let t = gate.target as usize;
            for g in &fold {
                assert!(
                    g.target as usize == pairs[t].0 || g.target as usize == pairs[t].1,
                    "fold writes outside the target's carriers"
                );
                assert!(!g.ctrls.is_empty(), "fold emitted a bare X");
            }
            for input in 0..(1u64 << total) {
                let before: Vec<u64> = (0..n)
                    .map(|v| {
                        prod_decode(input, v, &pairs, &slots_before, &consts_before, &ledger.loc)
                    })
                    .collect();
                let out_state = eval_u64(&fold, input);
                let after: Vec<u64> = (0..n)
                    .map(|v| {
                        prod_decode(
                            out_state,
                            v,
                            &pairs,
                            &ledger.slots,
                            &ledger.consts,
                            &ledger.loc,
                        )
                    })
                    .collect();
                // The virtual gate on the decoded values.
                let fires = gate
                    .ctrls
                    .iter()
                    .all(|&(w, pol)| (before[w as usize] != 0) == pol)
                    ^ gate.comp;
                for v in 0..n {
                    let expected = before[v] ^ ((v == t && fires) as u64);
                    assert_eq!(after[v], expected, "seed={seed} gate={gate:?} value={v}");
                }
            }
            // ledger state changed only in consts (slots untouched by CGs)
            assert_eq!(slots_before, ledger.slots);
        }
    }
}

#[test]
fn prod_degree_three_masks_round_trip_and_widen_fragments() {
    // Tower level: deg=3 masks. Functionality must be exact, and the fold
    // must actually emit wider (>= width-3) fragments — the algebraic
    // signature of a degree-3 mask term.
    let n = MASKED_TEST_N;
    let main = masked_test_main();
    let mask = (1u64 << n) - 1;
    let cfg = ProdConfig {
        k: 2,
        deg: 3,
        k_hi: 0,
        deg_hi: 3,
        band: 8,
        rsrc: 1,
        ..ProdConfig::off()
    };
    for seed in 0..6u64 {
        let mut rng = StdRng::seed_from_u64(0x0e63_0000 + seed);
        let g = gadgetize_cnot(&main, n, 2, &MaskConfig::off(), &cfg, &mut rng);
        assert_eq!(g.num_wires, 2 * n + 8);
        assert!(
            g.gates.iter().all(|gate| !gate.ctrls.is_empty()),
            "no bare X"
        );
        let max_width = g.gates.iter().map(|gate| gate.width()).max().unwrap();
        assert!(
            max_width >= 3,
            "seed={seed}: deg-3 masks must widen fragments (got {max_width})"
        );
        for input in 0..(1u64 << g.num_wires) {
            let expected = main.evaluate((input & mask) as usize) as u64 & mask;
            assert_eq!(eval_u64(&g.gates, input) & mask, expected, "seed={seed}");
        }
    }
}

#[test]
fn emit_g57_form_realizes_the_exact_conjunction() {
    // All 1- and 2-literal polarity combos, both variant draws: the gate
    // run must add conj(lits) ^ konst to the target and nothing else.
    for seed in 0..64u64 {
        let mut rng = StdRng::seed_from_u64(0x657f_0000 + seed);
        for lits in [
            vec![(1u16, true)],
            vec![(1u16, false)],
            vec![(1u16, true), (2u16, true)],
            vec![(1u16, false), (2u16, true)],
            vec![(1u16, true), (2u16, false)],
            vec![(1u16, false), (2u16, false)],
        ] {
            let mut gates = Vec::new();
            let konst = emit_g57_form(0, &lits, &mut rng, &mut gates);
            for input in 0..8u64 {
                let expected_fire = lits.iter().all(|&(w, p)| ((input >> w) & 1 != 0) == p);
                let out_state = eval_u64(&gates, input);
                let expected = input ^ (expected_fire as u64 ^ konst as u64);
                assert_eq!(out_state, expected, "seed={seed} lits={lits:?}");
            }
        }
    }
}

#[test]
fn emit_narrow_fragment_ladders_exactly_over_dirty_borrows() {
    // Widths 3..=6 over mixed polarities, caps 2 and 3. The ladder must
    // add exactly conj(lits) ^ konst to the target and leave every other
    // wire — including the DIRTY borrowed carriers — untouched, for EVERY
    // input state (no clean-ancilla assumption anywhere), while staying
    // within the width cap.
    let total = 12usize; // literals on 1..=6, borrows drawn from 0..12
    for seed in 0..48u64 {
        let mut rng = StdRng::seed_from_u64(0x1add_0000 + seed);
        for cap in 2..=3usize {
            for width in 3..=6usize {
                let lits: Vec<(u16, bool)> = (1..=width as u16)
                    .map(|w| (w, rng.random::<bool>()))
                    .collect();
                let mut gates = Vec::new();
                // Role set = every wire: this unit test has no ledger, and
                // the point here is the ladder algebra, not the pool policy.
                let all: Vec<u16> = (0..total as u16).collect();
                let konst = emit_narrow_fragment(
                    0,
                    &lits,
                    cap,
                    &all,
                    total,
                    &[],
                    &[],
                    1,
                    &mut rng,
                    &mut gates,
                );
                assert!(
                    gates.iter().all(|g| g.width() <= cap),
                    "seed={seed} cap={cap} width={width}: ladder exceeded the cap"
                );
                for input in 0..(1u64 << total) {
                    let expected_fire = lits.iter().all(|&(w, p)| ((input >> w) & 1 != 0) == p);
                    let out_state = eval_u64(&gates, input);
                    let expected = input ^ (expected_fire as u64 ^ konst as u64);
                    assert_eq!(
                        out_state, expected,
                        "seed={seed} cap={cap} width={width} input={input:#x}"
                    );
                }
            }
        }
    }
}

#[test]
fn prod_narrow_fold_cg_is_share_native_and_two_control() {
    // The narrow fold: same share-native contract as the wide fold — the
    // target's decode transitions by exactly the virtual gate and every
    // other value is untouched, for EVERY input (dirty borrows, no clean
    // ancilla) — with every gate at most 2 controls.
    let n = 3;
    let carrier_total = 2 * n;
    let pairs = vec![(0usize, 1usize), (2, 3), (4, 5)];
    let band = 6usize; // wires 6..12; scratch 12..16
    let cfg = ProdConfig {
        k: 1,
        deg: 2,
        k_hi: 1,
        deg_hi: 3,
        band,
        rsrc: 0,
        max_width: 2,
        fill_nl: 0,
        roll: 0,
        src_dist: 0,
        src_horizon: 0,
        src_lo: 0,
        src_hi: 0,
        fill_pivots: 0,
        g57_narrow: 0,
        ladder_cap: 0,
        cg_jitter: 0,
        rung_menu: 0,

        epoch: 0,
        refill_data: 0,
        single: 0,
        gray_fold: 0,
        swap_refresh: 0,
        close_slice: 0,
    };
    let live = carrier_total + band; // the whole wire space: no pinned wires
    let sources: Vec<XGate> = vec![
        XGate::from_g57([0, 1, 2]),
        XGate::cnot(1, 2),
        XGate::conj(2, [(0u16, false), (1u16, true)]).unwrap(),
        XGate::x_gate(0),
    ];
    for seed in 0..16u64 {
        let mut rng = StdRng::seed_from_u64(0x9d0e_0000 + seed);
        let state = GadgetState {
            n,
            pairs: pairs.clone(),
        };
        let mut ledger = ProdLedger::new(n, &cfg, carrier_total, None);
        let mut ramp = Vec::new();
        ledger.inject_all(&state, &mut rng, &mut ramp);
        for gate in &sources {
            let slots_before = ledger.slots.clone();
            let consts_before = ledger.consts.clone();
            let mut fold = Vec::new();
            ledger.fold_cg(gate, &state, &mut rng, &mut fold);
            let t = gate.target as usize;
            for g in &fold {
                assert!(g.width() <= 2, "narrow fold emitted a wide gate");
            }
            // A ladder rung may borrow ANY wire, band included — the
            // double sweep restores it, which the next assertion checks
            // over the whole input domain.
            // Net effect must live ENTIRELY on the target's two carriers:
            // every dirty borrow is restored, for every input.
            let touched = (1u64 << pairs[t].0) | (1u64 << pairs[t].1);
            for input in 0..(1u64 << live) {
                let before: Vec<u64> = (0..n)
                    .map(|v| {
                        prod_decode(input, v, &pairs, &slots_before, &consts_before, &ledger.loc)
                    })
                    .collect();
                let out_state = eval_u64(&fold, input);
                assert_eq!(
                    (out_state ^ input) & !touched,
                    0,
                    "seed={seed}: a borrowed wire was not restored"
                );
                let after: Vec<u64> = (0..n)
                    .map(|v| {
                        prod_decode(
                            out_state,
                            v,
                            &pairs,
                            &ledger.slots,
                            &ledger.consts,
                            &ledger.loc,
                        )
                    })
                    .collect();
                let fires = gate
                    .ctrls
                    .iter()
                    .all(|&(w, pol)| (before[w as usize] != 0) == pol)
                    ^ gate.comp;
                for v in 0..n {
                    let expected = before[v] ^ ((v == t && fires) as u64);
                    assert_eq!(after[v], expected, "seed={seed} gate={gate:?} value={v}");
                }
            }
            assert_eq!(slots_before, ledger.slots);
        }
    }
}

#[test]
fn prod_micro_gray_quartic_gather_is_exact_over_two_dirty_helpers() {
    // acc=0, helpers=1,2, quartic inputs=3..6. Exercise every literal
    // polarity and every incoming state: only acc may change, by exactly
    // the requested quartic, and both arbitrary-dirty helpers come back.
    for polarity_mask in 0u8..16 {
        let atom: Vec<(u16, bool)> = (0..4)
            .map(|index| (3 + index, polarity_mask & (1 << index) != 0))
            .collect();
        let mut rng = StdRng::seed_from_u64(0x4d47_0000 + polarity_mask as u64);
        let mut seen = std::collections::HashMap::new();
        let mut gates = Vec::new();
        assert!(!emit_micro_atom_onto(
            0,
            &atom,
            MicroAtomPlan {
                helper0: 1,
                helper1: 2,
                pivot: 0,
            },
            1,
            &mut seen,
            &mut rng,
            &mut gates,
        ));
        assert_eq!(gates.len(), 8, "quartic micro ladder cost drifted");
        assert!(gates.iter().all(|gate| gate.width() <= 2));
        for input in 0u64..128 {
            let fire = atom
                .iter()
                .all(|&(wire, polarity)| ((input >> wire) & 1 != 0) == polarity);
            assert_eq!(
                eval_u64(&gates, input),
                input ^ fire as u64,
                "polarity={polarity_mask:#06b} input={input:#09b}"
            );
        }
    }
}

#[test]
fn prod_micro_gray_trace_needs_all_four_share_deltas() {
    // Four one-atom formal shares, each gathered and immediately stripped
    // from the same dirty accumulator. The trace contains each individual
    // row delta, but never the complete operand on one before/after pair.
    let cfg = ProdConfig::off();
    let ledger = ProdLedger::new(1, &cfg, 6, None);
    let atoms: Vec<Vec<(u16, bool)>> = (1..=4u16).map(|wire| vec![(wire, true)]).collect();
    let mut rng = StdRng::seed_from_u64(0x4d47_1001);
    let shares = micro_partition_atoms(&atoms, &mut rng).expect("four atoms make four rows");
    let plan = [MicroAtomPlan {
        helper0: 5,
        helper1: 5,
        pivot: 0,
    }];
    let mut seen = std::collections::HashMap::new();
    let mut gates = Vec::new();
    let mut intervals = Vec::new();
    for share in &shares {
        let before = gates.len();
        ledger.gather_micro_share_exact(0, share, &plan, 5, &mut seen, &mut rng, &mut gates);
        let after = gates.len();
        intervals.push((before, after));
        ledger.gather_micro_share_exact(0, share, &plan, 5, &mut seen, &mut rng, &mut gates);
    }
    let wires = 6;
    let trace = prefix_wire_signatures(&gates, wires);
    let mut operand = vec![0u64; 1];
    for wire in 1..=4 {
        operand = xor_signatures(&operand, &trace[wire]);
    }
    let deltas: Vec<Vec<u64>> = intervals
        .iter()
        .map(|&(before, after)| xor_signatures(&trace[before * wires], &trace[after * wires]))
        .collect();
    for subset in 1usize..16 {
        let mut sum = vec![0u64; operand.len()];
        for (row, delta) in deltas.iter().enumerate() {
            if subset & (1 << row) != 0 {
                sum = xor_signatures(&sum, delta);
            }
        }
        assert_eq!(
            sum == operand,
            subset == 15,
            "formal share subset {subset:#06b} changed rank"
        );
    }
    assert!(
        same_wire_space_time_witness(&trace, wires, &operand).is_none(),
        "a single accumulator interval gathered the complete operand"
    );
    for input in 0u64..(1 << wires) {
        assert_eq!(
            eval_u64(&gates, input),
            input,
            "share gathers did not strip back to arbitrary incoming junk"
        );
    }
}

#[test]
fn prod_micro_gray_generic_fold_is_exhaustive_and_two_control() {
    // Four atoms per operand, including a quartic mask, force the r=4
    // schedule and its two-helper path. Exhaust the complete 12-wire state
    // space so target semantics and restoration of every dirty borrow are
    // checked without a clean-wire assumption.
    let n = 7;
    let carrier_total = n;
    let pairs: Vec<(usize, usize)> = (0..n).map(|wire| (wire, wire)).collect();
    let band = 5;
    let live = carrier_total + band;
    let cfg = ProdConfig {
        k: 2,
        deg: 2,
        k_hi: 1,
        deg_hi: 4,
        band,
        rsrc: 0,
        single: 1,
        g57_narrow: 1,
        rung_menu: 1,
        gray_fold: 2,
        ..ProdConfig::off()
    };
    let gate = XGate::from_g57([0, 1, 2]);
    for seed in 0..3u64 {
        let state = GadgetState {
            n,
            pairs: pairs.clone(),
        };
        let mut rng = StdRng::seed_from_u64(0x4d47_2000 + seed);
        let mut ledger = ProdLedger::new(n, &cfg, carrier_total, None);
        let mut ramp = Vec::new();
        ledger.inject_all(&state, &mut rng, &mut ramp);
        let slots_before = ledger.slots.clone();
        let consts_before = ledger.consts.clone();
        let mut fold = Vec::new();
        ledger.fold_cg(&gate, &state, &mut rng, &mut fold);
        assert_eq!(ledger.cg_gray, 1, "seed={seed}: micro path declined");
        assert!(
            fold.iter().all(|emitted| emitted.width() <= 2),
            "seed={seed}: micro product emitted above the width cap"
        );
        let touched = 1u64 << gate.target;
        for input in 0u64..(1u64 << live) {
            let before: Vec<u64> = (0..n)
                .map(|value| {
                    prod_decode(
                        input,
                        value,
                        &pairs,
                        &slots_before,
                        &consts_before,
                        &ledger.loc,
                    )
                })
                .collect();
            let output = eval_u64(&fold, input);
            assert_eq!(
                (output ^ input) & !touched,
                0,
                "seed={seed} input={input:#x}: dirty borrow was not restored"
            );
            let after: Vec<u64> = (0..n)
                .map(|value| {
                    prod_decode(
                        output,
                        value,
                        &pairs,
                        &ledger.slots,
                        &ledger.consts,
                        &ledger.loc,
                    )
                })
                .collect();
            let fires = gate
                .ctrls
                .iter()
                .all(|&(wire, polarity)| (before[wire as usize] != 0) == polarity)
                ^ gate.comp;
            for value in 0..n {
                assert_eq!(
                    after[value],
                    before[value] ^ ((value == gate.target as usize && fires) as u64),
                    "seed={seed} input={input:#x} value={value}"
                );
            }
        }
    }
}

#[test]
fn prod_micro_gray_five_six_seven_width_and_restoration_census() {
    let gate = XGate::from_g57([0, 1, 2]);
    let check = |label: &str,
                 fold: &[XGate],
                 update_len: usize,
                 total: usize,
                 target_wires: &[usize],
                 before_atoms: &[Vec<Vec<(u16, bool)>>],
                 after_atoms: &[Vec<Vec<(u16, bool)>>]| {
        assert!(
            fold[update_len..]
                .iter()
                .all(|emitted| emitted.width() <= 2),
            "{label}: micro product/gather suffix exceeded two controls"
        );
        let live_mask = (1u64 << total) - 1;
        let target_mask = target_wires
            .iter()
            .fold(0u64, |mask, &wire| mask | (1u64 << wire));
        let mut samples = vec![0u64, live_mask];
        samples.extend((0..1024u64).map(|index| {
            index
                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .rotate_left((index as u32 * 11) & 63)
                & live_mask
        }));
        for input in samples {
            let before: Vec<u64> = before_atoms
                .iter()
                .map(|atoms| eval_anf_atoms(input, atoms))
                .collect();
            let output = eval_u64(fold, input);
            assert_eq!(
                (output ^ input) & (live_mask ^ target_mask),
                0,
                "{label}: accumulator/helper was not restored at {input:#x}"
            );
            let after: Vec<u64> = after_atoms
                .iter()
                .map(|atoms| eval_anf_atoms(output, atoms))
                .collect();
            let fires = gate
                .ctrls
                .iter()
                .all(|&(wire, polarity)| (before[wire as usize] != 0) == polarity)
                ^ gate.comp;
            for value in 0..before.len() {
                assert_eq!(
                    after[value],
                    before[value] ^ ((value == gate.target as usize && fires) as u64),
                    "{label}: logical mismatch at input={input:#x} value={value}"
                );
            }
        }
    };

    let n = 7;
    let mut cfg = ProdConfig::off();
    cfg.gray_fold = 2;
    cfg.rung_menu = 1;

    let five = FiveCarrierState::home(n);
    let mut five_ledger = ProdLedger::new(n, &cfg, 5 * n, None);
    let five_before: Vec<_> = (0..n)
        .map(|value| five_ledger.five_decode_atoms(value, true, &five))
        .collect();
    let mut rng = StdRng::seed_from_u64(0x4d47_5005);
    let mut five_fold = Vec::new();
    five_ledger.fold_five(&gate, &five, &mut rng, &mut five_fold);
    assert_eq!(five_ledger.cg_gray, 1, "five-carrier micro path declined");
    assert_eq!((five_ledger.cg_fragments, five_ledger.cg_narrow), (64, 64));
    let five_after: Vec<_> = (0..n)
        .map(|value| five_ledger.five_decode_atoms(value, true, &five))
        .collect();
    check(
        "five",
        &five_fold,
        FIVE_CARRIER_U0_GATES.len(),
        5 * n,
        &five.carriers[0],
        &five_before,
        &five_after,
    );

    let six = SixCarrierState::home(n);
    let mut six_ledger = ProdLedger::new(n, &cfg, 6 * n, None);
    let six_before: Vec<_> = (0..n)
        .map(|value| six_ledger.six_decode_atoms(value, true, &six))
        .collect();
    let mut rng = StdRng::seed_from_u64(0x4d47_6006);
    let mut six_fold = Vec::new();
    six_ledger.fold_six(&gate, &six, &mut rng, &mut six_fold);
    assert_eq!(six_ledger.cg_gray, 1, "six-carrier micro path declined");
    assert_eq!((six_ledger.cg_fragments, six_ledger.cg_narrow), (64, 64));
    let six_after: Vec<_> = (0..n)
        .map(|value| six_ledger.six_decode_atoms(value, true, &six))
        .collect();
    check(
        "six",
        &six_fold,
        SIX_CARRIER_U0_GATES.len(),
        6 * n,
        &six.carriers[0],
        &six_before,
        &six_after,
    );

    let seven = SevenCarrierState::home(n);
    let mut seven_ledger = ProdLedger::new(n, &cfg, 7 * n, None);
    let seven_before: Vec<_> = (0..n)
        .map(|value| seven_ledger.seven_decode_atoms(value, true, &seven))
        .collect();
    let mut rng = StdRng::seed_from_u64(0x4d47_7007);
    let mut seven_fold = Vec::new();
    seven_ledger.fold_seven(&gate, &seven, &mut rng, &mut seven_fold);
    assert_eq!(seven_ledger.cg_gray, 1, "seven-carrier micro path declined");
    assert_eq!(
        (seven_ledger.cg_fragments, seven_ledger.cg_narrow),
        (64, 64)
    );
    let seven_after: Vec<_> = (0..n)
        .map(|value| seven_ledger.seven_decode_atoms(value, true, &seven))
        .collect();
    check(
        "seven",
        &seven_fold,
        SEVEN_CARRIER_U0_GATES.len(),
        7 * n,
        &seven.carriers[0],
        &seven_before,
        &seven_after,
    );
}

#[test]
fn prod_sentinel_cross_ladders_are_exact_and_force_a_cross_rung() {
    // target=0, dirty helpers=1,2, blind=3, H literals=4..6. Every
    // polarity pattern and incoming state is covered. The gates writing
    // helper0 must read the blind plus exactly one H literal; choosing two
    // H literals here would expose the complete cubic H one rung later.
    for width in 3usize..=4 {
        for polarity_mask in 0usize..(1usize << width) {
            let lits: Vec<(u16, bool)> = (0..width)
                .map(|index| ((index + 3) as u16, polarity_mask & (1 << index) != 0))
                .collect();
            let mut gates = Vec::new();
            emit_exact_dirty_cap2(0, &lits, 1, 2, &mut gates);
            assert_eq!(gates.len(), if width == 3 { 4 } else { 8 });
            assert!(gates.iter().all(|gate| gate.width() <= 2));
            for gate in gates.iter().filter(|gate| gate.target == 1) {
                let controls: Vec<u16> = gate.ctrls.iter().map(|&(wire, _)| wire).collect();
                assert!(controls.contains(&3), "rung zero lost the blind factor");
                assert!(controls.contains(&4), "rung zero lost its first H factor");
                assert_eq!(controls.len(), 2);
            }
            let wires = width + 3;
            for input in 0u64..(1u64 << wires) {
                let fire = lits
                    .iter()
                    .all(|&(wire, polarity)| ((input >> wire) & 1 != 0) == polarity);
                assert_eq!(
                    eval_u64(&gates, input),
                    input ^ fire as u64,
                    "width={width} polarity={polarity_mask:#x} input={input:#x}"
                );
            }
        }
    }
}

#[test]
fn prod_sentinel_schedule_is_exhaustive_restores_junk_and_has_no_full_gather() {
    // Physical roles: target 0; live linear operands 1,2; unrelated dirty
    // u,z,h0,h1 = 3..6; shared mask variables 7..12. The Q/H factors
    // overlap deliberately, so this is not a disjoint-variable toy.
    let lists = [
        vec![
            vec![(1u16, false)],
            vec![(7, true), (8, false)],
            vec![(7, true), (8, false), (9, true)],
        ],
        vec![
            vec![(2u16, true)],
            vec![(8, true), (9, true)],
            vec![(10, false), (11, true), (12, false)],
        ],
    ];
    let parts = [
        partition_max_degree_sentinel(&lists[0]).unwrap(),
        partition_max_degree_sentinel(&lists[1]).unwrap(),
    ];
    let cfg = ProdConfig::off();
    let mut ledger = ProdLedger::new(7, &cfg, 7, None);
    let mut gates = Vec::new();
    ledger.emit_sentinel_schedule(0, 0, &parts, [3, 4, 5, 6], &mut gates);
    let census = |width| gates.iter().filter(|gate| gate.width() == width).count();
    assert_eq!(gates.len(), 62, "sentinel primitive cost drifted");
    assert_eq!(census(2), 61, "cap-two population drifted");
    assert_eq!(census(6), 1, "H*H sentinel fossil drifted");
    assert_eq!(ledger.cg_laddered, 6, "expected six cross-tail ladders");
    assert_eq!(ledger.cg_fossils, 1, "expected exactly one H*H fossil");

    let wires = 13;
    for input in 0u64..(1u64 << wires) {
        let left = eval_anf_atoms(input, &lists[0]);
        let right = eval_anf_atoms(input, &lists[1]);
        let output = eval_u64(&gates, input);
        assert_eq!(
            output,
            input ^ (left & right),
            "sentinel identity/restoration failed at {input:#x}"
        );
    }

    // Q is the intentional canary: an accumulator interval must reveal it.
    // Neither H nor a complete operand may appear on u/z/helper deltas.
    let trace = prefix_wire_signatures(&gates, wires);
    let signature = |atoms: &[Vec<(u16, bool)>]| -> Vec<u64> {
        let mut value = vec![0u64; (1usize << wires) / 64];
        for input in 0usize..(1usize << wires) {
            if eval_anf_atoms(input as u64, atoms) != 0 {
                value[input / 64] |= 1u64 << (input % 64);
            }
        }
        value
    };
    let q = [signature(&parts[0].gathered), signature(&parts[1].gathered)];
    let forbidden = [
        signature(&parts[0].high),
        signature(&parts[1].high),
        signature(&lists[0]),
        signature(&lists[1]),
    ];
    let prefixes = gates.len() + 1;
    let mut saw_q = [false; 2];
    for wire in [3usize, 4, 5, 6] {
        for before in 0..prefixes {
            for after in before + 1..prefixes {
                let delta =
                    xor_signatures(&trace[before * wires + wire], &trace[after * wires + wire]);
                for side in 0..2 {
                    saw_q[side] |= delta == q[side];
                }
                assert!(
                    forbidden.iter().all(|secret| delta != *secret),
                    "borrowed wire {wire} gathered H or a complete operand"
                );
            }
        }
    }
    assert_eq!(saw_q, [true, true], "Q canary was not trace-recoverable");
}

#[test]
fn prod_sentinel_production_single_is_exhaustive_with_signed_constants() {
    // Deterministic [2,2,2,3] production mask plan on two disjoint halves
    // of a six-wire band. This pins the nominal sentinel census exactly:
    // 69 cap-two primitives and one cubic*cubic width-six fossil.
    let n = 7;
    let carrier_total = n;
    let pairs: Vec<(usize, usize)> = (0..n).map(|wire| (wire, wire)).collect();
    let state = GadgetState {
        n,
        pairs: pairs.clone(),
    };
    let cfg = ProdConfig {
        k: 3,
        deg: 2,
        k_hi: 1,
        deg_hi: 3,
        band: 6,
        rsrc: 0,
        single: 1,
        gray_fold: 3,
        ..ProdConfig::off()
    };
    let source = XGate::from_g57([0, 1, 2]);
    let plans = [
        vec![
            ProdSlot {
                factors: vec![(0, false), (1, false)],
            },
            ProdSlot {
                factors: vec![(0, false), (2, false)],
            },
            ProdSlot {
                factors: vec![(1, false), (2, false)],
            },
            ProdSlot {
                factors: vec![(0, false), (1, false), (2, false)],
            },
        ],
        vec![
            ProdSlot {
                factors: vec![(3, false), (4, false)],
            },
            ProdSlot {
                factors: vec![(3, false), (5, false)],
            },
            ProdSlot {
                factors: vec![(4, false), (5, false)],
            },
            ProdSlot {
                factors: vec![(3, false), (4, false), (5, false)],
            },
        ],
    ];
    let live = carrier_total + 6;
    for constant_mask in 0usize..4 {
        let mut ledger = ProdLedger::new(n, &cfg, carrier_total, None);
        ledger.slots[1] = plans[0].clone();
        ledger.slots[2] = plans[1].clone();
        ledger.consts[1] = constant_mask & 1 != 0;
        ledger.consts[2] = constant_mask & 2 != 0;
        let slots_before = ledger.slots.clone();
        let consts_before = ledger.consts.clone();
        let mut rng = StdRng::seed_from_u64(0x5e17_0000 + constant_mask as u64);
        let mut fold = Vec::new();
        ledger.fold_cg(&source, &state, &mut rng, &mut fold);
        assert_eq!(ledger.cg_sentinel, 1, "sentinel path declined");
        assert_eq!(fold.len(), 70, "production-single cost drifted");
        assert_eq!(fold.iter().filter(|gate| gate.width() == 2).count(), 69);
        assert_eq!(fold.iter().filter(|gate| gate.width() == 6).count(), 1);
        let touched = 1u64 << source.target;
        for input in 0u64..(1u64 << live) {
            let before: Vec<u64> = (0..n)
                .map(|value| {
                    prod_decode(
                        input,
                        value,
                        &pairs,
                        &slots_before,
                        &consts_before,
                        &ledger.loc,
                    )
                })
                .collect();
            let output = eval_u64(&fold, input);
            assert_eq!(
                (output ^ input) & !touched,
                0,
                "constant_mask={constant_mask} input={input:#x}: borrow leaked"
            );
            let after: Vec<u64> = (0..n)
                .map(|value| {
                    prod_decode(
                        output,
                        value,
                        &pairs,
                        &ledger.slots,
                        &ledger.consts,
                        &ledger.loc,
                    )
                })
                .collect();
            let fires = source
                .ctrls
                .iter()
                .all(|&(wire, polarity)| (before[wire as usize] != 0) == polarity)
                ^ source.comp;
            for value in 0..n {
                assert_eq!(
                    after[value],
                    before[value] ^ ((value == 0 && fires) as u64),
                    "constant_mask={constant_mask} input={input:#x} value={value}"
                );
            }
        }
    }
}

#[test]
fn prod_sentinel_five_six_seven_correctness_and_width_census() {
    let n = 7usize;
    let band = 6usize;
    let source = XGate::from_g57([0, 1, 2]);
    let cfg = ProdConfig {
        k: 3,
        deg: 2,
        k_hi: 1,
        deg_hi: 3,
        band,
        rsrc: 0,
        single: 1,
        gray_fold: 3,
        ..ProdConfig::off()
    };
    let plans = [
        vec![
            ProdSlot {
                factors: vec![(0, false), (1, false)],
            },
            ProdSlot {
                factors: vec![(0, false), (2, false)],
            },
            ProdSlot {
                factors: vec![(1, false), (2, false)],
            },
            ProdSlot {
                factors: vec![(0, false), (1, false), (2, false)],
            },
        ],
        vec![
            ProdSlot {
                factors: vec![(3, false), (4, false)],
            },
            ProdSlot {
                factors: vec![(3, false), (5, false)],
            },
            ProdSlot {
                factors: vec![(4, false), (5, false)],
            },
            ProdSlot {
                factors: vec![(3, false), (4, false), (5, false)],
            },
        ],
    ];
    let census = |gates: &[XGate]| -> Vec<(usize, usize)> {
        let mut counts = std::collections::BTreeMap::new();
        for gate in gates {
            *counts.entry(gate.width()).or_insert(0usize) += 1;
        }
        counts.into_iter().collect()
    };
    let samples = |total: usize| -> Vec<u64> {
        let mask = (1u64 << total) - 1;
        std::iter::once(0)
            .chain(std::iter::once(mask))
            .chain((0..2048u64).map(|index| {
                index
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    .rotate_left((index as u32 * 7) & 63)
                    & mask
            }))
            .collect()
    };

    let five = FiveCarrierState::home(n);
    let mut five_ledger = ProdLedger::new(n, &cfg, 5 * n, None);
    five_ledger.slots[1] = plans[0].clone();
    five_ledger.slots[2] = plans[1].clone();
    let five_before: Vec<_> = (0..n)
        .map(|value| five_ledger.five_decode_atoms(value, true, &five))
        .collect();
    let mut five_fold = Vec::new();
    let mut rng = StdRng::seed_from_u64(0x5e17_5005);
    five_ledger.fold_five(&source, &five, &mut rng, &mut five_fold);
    let five_after: Vec<_> = (0..n)
        .map(|value| five_ledger.five_decode_atoms(value, true, &five))
        .collect();
    let five_suffix = &five_fold[FIVE_CARRIER_U0_GATES.len()..];
    assert_eq!(five_ledger.cg_sentinel, 1, "five sentinel path declined");
    assert_eq!(census(five_suffix), vec![(2, 85), (6, 1)]);

    let six = SixCarrierState::home(n);
    let mut six_ledger = ProdLedger::new(n, &cfg, 6 * n, None);
    six_ledger.slots[1] = plans[0].clone();
    six_ledger.slots[2] = plans[1].clone();
    let six_before: Vec<_> = (0..n)
        .map(|value| six_ledger.six_decode_atoms(value, true, &six))
        .collect();
    let mut six_fold = Vec::new();
    let mut rng = StdRng::seed_from_u64(0x5e17_6006);
    six_ledger.fold_six(&source, &six, &mut rng, &mut six_fold);
    let six_after: Vec<_> = (0..n)
        .map(|value| six_ledger.six_decode_atoms(value, true, &six))
        .collect();
    let six_suffix = &six_fold[SIX_CARRIER_U0_GATES.len()..];
    assert_eq!(six_ledger.cg_sentinel, 1, "six sentinel path declined");
    assert_eq!(census(six_suffix), vec![(2, 348), (6, 9)]);

    let seven = SevenCarrierState::home(n);
    let mut seven_ledger = ProdLedger::new(n, &cfg, 7 * n, None);
    seven_ledger.slots[1] = plans[0].clone();
    seven_ledger.slots[2] = plans[1].clone();
    let seven_before: Vec<_> = (0..n)
        .map(|value| seven_ledger.seven_decode_atoms(value, true, &seven))
        .collect();
    let mut seven_fold = Vec::new();
    let mut rng = StdRng::seed_from_u64(0x5e17_7007);
    seven_ledger.fold_seven(&source, &seven, &mut rng, &mut seven_fold);
    let seven_after: Vec<_> = (0..n)
        .map(|value| seven_ledger.seven_decode_atoms(value, true, &seven))
        .collect();
    let seven_suffix = &seven_fold[SEVEN_CARRIER_U0_GATES.len()..];
    assert_eq!(seven_ledger.cg_sentinel, 1, "seven sentinel path declined");
    // Quartic H makes H*accumulator width five; mode 3 intentionally
    // ladders only through width four, so those ten brackets remain as
    // fossils. The production cubic mask is below the representation's
    // max degree and is therefore gathered as Q, not treated as H.
    assert_eq!(census(seven_suffix), vec![(2, 61), (5, 10), (8, 1)]);

    let fixtures = [
        (
            "five",
            5 * n + band,
            &five_fold[..],
            &five.carriers[0][..],
            &five_before[..],
            &five_after[..],
        ),
        (
            "six",
            6 * n + band,
            &six_fold[..],
            &six.carriers[0][..],
            &six_before[..],
            &six_after[..],
        ),
        (
            "seven",
            7 * n + band,
            &seven_fold[..],
            &seven.carriers[0][..],
            &seven_before[..],
            &seven_after[..],
        ),
    ];
    for (label, total, fold, target_group, before_atoms, after_atoms) in fixtures {
        let target_mask = target_group
            .iter()
            .fold(0u64, |mask, &wire| mask | (1u64 << wire));
        let total_mask = (1u64 << total) - 1;
        for input in samples(total) {
            let before: Vec<u64> = before_atoms
                .iter()
                .map(|atoms| eval_anf_atoms(input, atoms))
                .collect();
            let output = eval_u64(fold, input);
            assert_eq!(
                (output ^ input) & (total_mask ^ target_mask),
                0,
                "{label}: unrelated dirty wire changed at {input:#x}"
            );
            let after: Vec<u64> = after_atoms
                .iter()
                .map(|atoms| eval_anf_atoms(output, atoms))
                .collect();
            let fires = source
                .ctrls
                .iter()
                .all(|&(wire, polarity)| (before[wire as usize] != 0) == polarity)
                ^ source.comp;
            for value in 0..n {
                assert_eq!(
                    after[value],
                    before[value] ^ ((value == 0 && fires) as u64),
                    "{label}: logical mismatch at {input:#x}, value={value}"
                );
            }
        }
    }
}

#[test]
fn prod_gray_fold_is_share_native_and_two_control() {
    // The Gray fold's contract, over the WHOLE input domain (dirty borrows,
    // no clean-ancilla assumption anywhere): the target value's decode
    // transitions by exactly the virtual gate, every other value is
    // untouched, every borrowed accumulator and sandwich helper is restored,
    // and no emitted gate has more than two controls.
    //
    // The residual-constant trap lives here: a gather lands `M + delta`, and
    // if the constant-atom absorption were wrong the fold would silently
    // compute `(M_b + delta)(M_c + eps)` -- a WRONG FUNCTION, not a leak.
    // Only a full-domain check over both operands' masks catches it, which
    // is why the source list below includes a mixed-polarity CCNOT.
    // n must leave carriers over to borrow: with n=3 every carrier is the
    // target or an operand, the fold declines, and the test would only be
    // re-checking the odometer (the `gray_blocks` assertion at the end).
    let n = 6;
    let carrier_total = n; // single-carrier: value v lives on wire v
    let pairs: Vec<(usize, usize)> = (0..n).map(|v| (v, v)).collect();
    let band = 5usize; // wires 6..11
    let live = carrier_total + band;
    let cfg = ProdConfig {
        k: 1,
        deg: 2,
        k_hi: 1,
        deg_hi: 3,
        band,
        rsrc: 0,
        single: 1,
        g57_narrow: 1,
        gray_fold: 1,
        ..ProdConfig::off()
    };
    let sources: Vec<XGate> = vec![
        XGate::from_g57([0, 1, 2]),
        XGate::conj(2, [(0u16, false), (1u16, true)]).unwrap(),
        XGate::conj(0, [(1u16, true), (2u16, true)]).unwrap(),
        XGate::conj(1, [(0u16, false), (2u16, false)]).unwrap(),
        XGate::cnot(1, 2),
        XGate::x_gate(0),
    ];
    let mut gray_blocks = 0u64;
    for seed in 0..24u64 {
        let mut rng = StdRng::seed_from_u64(0x67a4_0000 + seed);
        let state = GadgetState {
            n,
            pairs: pairs.clone(),
        };
        let mut ledger = ProdLedger::new(n, &cfg, carrier_total, None);
        let mut ramp = Vec::new();
        ledger.inject_all(&state, &mut rng, &mut ramp);
        for gate in &sources {
            let slots_before = ledger.slots.clone();
            let consts_before = ledger.consts.clone();
            let mut fold = Vec::new();
            ledger.fold_cg(gate, &state, &mut rng, &mut fold);
            let t = gate.target as usize;
            for g in &fold {
                assert!(
                    g.width() <= 2,
                    "seed={seed} gate={gate:?}: gray fold emitted a {}-control gate",
                    g.width()
                );
            }
            let touched = 1u64 << pairs[t].0;
            for input in 0..(1u64 << live) {
                let before: Vec<u64> = (0..n)
                    .map(|v| {
                        prod_decode(input, v, &pairs, &slots_before, &consts_before, &ledger.loc)
                    })
                    .collect();
                let out_state = eval_u64(&fold, input);
                // Every accumulator and every sandwich helper is restored:
                // the net effect lives entirely on the target's carrier.
                assert_eq!(
                    (out_state ^ input) & !touched,
                    0,
                    "seed={seed} gate={gate:?}: a borrowed wire was not restored"
                );
                let after: Vec<u64> = (0..n)
                    .map(|v| {
                        prod_decode(
                            out_state,
                            v,
                            &pairs,
                            &ledger.slots,
                            &ledger.consts,
                            &ledger.loc,
                        )
                    })
                    .collect();
                let fires = gate
                    .ctrls
                    .iter()
                    .all(|&(w, pol)| (before[w as usize] != 0) == pol)
                    ^ gate.comp;
                for v in 0..n {
                    let expected = before[v] ^ ((v == t && fires) as u64);
                    assert_eq!(
                        after[v], expected,
                        "seed={seed} gate={gate:?} value={v} input={input:#x}"
                    );
                }
            }
            assert_eq!(slots_before, ledger.slots, "the fold disturbed a slot");
        }
        gray_blocks += ledger.cg_gray;
    }
    // The arity-2 sources must actually take the Gray path, or the test
    // above is only re-checking the odometer.
    assert!(
        gray_blocks >= 24 * 4,
        "expected every arity-2 block to fold the Gray way, got {gray_blocks}"
    );
}

#[test]
fn prod_gray_fold_keeps_the_accumulators_dirty() {
    // The security invariant, structurally: no emitted gate may read an
    // accumulator while that wire holds a CLEAN mask sum. Equivalently --
    // and this is what is checkable without re-running the exposure audit --
    // the fold must never write a wire that is bare-zero-initialized, and
    // every wire it borrows must be one the block does not otherwise read.
    //
    // What is asserted here: the accumulators are drawn from the CARRIER
    // roles (not the band, not by index), they are distinct from the
    // target's carrier and from every literal the block reads, and each is
    // written by an even number of gates so its incoming junk survives to
    // cancel. A clean accumulator would show up as a wire whose first
    // touch is a write with no prior read -- covered by the restoration
    // assertion in the test above, which a clean-ancilla variant fails.
    let n = 6;
    let carrier_total = n;
    let pairs: Vec<(usize, usize)> = (0..n).map(|v| (v, v)).collect();
    let band = 8usize;
    let cfg = ProdConfig {
        k: 1,
        deg: 2,
        k_hi: 2,
        deg_hi: 3,
        band,
        rsrc: 0,
        single: 1,
        g57_narrow: 1,
        gray_fold: 1,
        ..ProdConfig::off()
    };
    for seed in 0..32u64 {
        let mut rng = StdRng::seed_from_u64(0x67a5_0000 + seed);
        let state = GadgetState {
            n,
            pairs: pairs.clone(),
        };
        let mut ledger = ProdLedger::new(n, &cfg, carrier_total, None);
        let mut ramp = Vec::new();
        ledger.inject_all(&state, &mut rng, &mut ramp);
        let gate = XGate::from_g57([0, 1, 2]);
        let mut fold = Vec::new();
        ledger.fold_cg(&gate, &state, &mut rng, &mut fold);
        assert_eq!(ledger.cg_gray, 1, "seed={seed}: not the gray path");
        // Wires the block reads as mask/carrier literals of its operands.
        let mut operand_wires: Vec<u16> = vec![0, 1, 2];
        for w in [1usize, 2] {
            for slot in &ledger.slots[w] {
                operand_wires.extend(slot.lits(&ledger.loc).iter().map(|&(x, _)| x));
            }
        }
        let mut writes: std::collections::HashMap<u16, usize> = std::collections::HashMap::new();
        for g in &fold {
            *writes.entry(g.target).or_default() += 1;
        }
        for (&w, &count) in &writes {
            if w == 0 {
                continue; // the target's carrier: written an odd number of times
            }
            assert_eq!(
                count % 2,
                0,
                "seed={seed}: borrowed wire {w} is written {count} times, so its \
                 incoming value does not cancel"
            );
            assert!(
                !operand_wires.contains(&w),
                "seed={seed}: wire {w} is both borrowed and read as an operand literal"
            );
        }
    }
}

#[test]
fn prod_gray_fold_has_an_exact_space_time_operand_recovery() {
    // Red-team the assumption made by the ordinary Gray audit: it checks
    // every prefix separately, while a trace adversary can combine two
    // prefixes.  Compare identical masks and RNG choices with and without
    // the aggregate Gray gather.
    let n = 6;
    let band = 5usize;
    let wires = n + band;
    let pairs: Vec<(usize, usize)> = (0..n).map(|value| (value, value)).collect();
    let state = GadgetState {
        n,
        pairs: pairs.clone(),
    };
    let source = XGate::from_g57([0, 1, 2]);

    let build = |gray: bool| {
        // Build the SAME masks and consume the SAME injection RNG in both
        // arms, then switch only the fold strategy.  Configuring max_width
        // differently before injection shifts the random mask stream and
        // invalidates the A/B comparison this test is meant to make.
        let cfg = ProdConfig {
            k: 3,
            deg: 2,
            k_hi: 1,
            deg_hi: 3,
            band,
            rsrc: 0,
            single: 1,
            max_width: 2,
            g57_narrow: 1,
            gray_fold: 0,
            ..ProdConfig::off()
        };
        let mut rng = StdRng::seed_from_u64(0x67a6_5ace);
        let mut ledger = ProdLedger::new(n, &cfg, n, None);
        let mut ramp = Vec::new();
        ledger.inject_all(&state, &mut rng, &mut ramp);
        let slots = ledger.slots.clone();
        let consts = ledger.consts.clone();
        let loc = ledger.loc.clone();
        ledger.gray_fold = gray;
        let mut fold = Vec::new();
        ledger.fold_cg(&source, &state, &mut rng, &mut fold);
        (fold, slots, consts, loc, ledger.cg_gray)
    };

    let (gray_fold, slots, consts, loc, gray_blocks) = build(true);
    let (expanded_fold, expanded_slots, expanded_consts, expanded_loc, expanded_gray) =
        build(false);
    assert_eq!(slots, expanded_slots, "A/B masks differ");
    assert_eq!(consts, expanded_consts, "A/B mask constants differ");
    assert_eq!(loc, expanded_loc, "A/B band locations differ");
    assert_eq!(gray_blocks, 1, "fixture did not exercise the Gray path");
    assert_eq!(expanded_gray, 0, "control unexpectedly used the Gray path");

    let gray_trace = prefix_wire_signatures(&gray_fold, wires);
    let expanded_trace = prefix_wire_signatures(&expanded_fold, wires);
    for operand in [1usize, 2] {
        let logical = prod_decode_signature(wires, operand, &pairs, &slots, &consts, &loc);
        let carrier_at_entry = &gray_trace[operand];
        let needed_delta = xor_signatures(&logical, carrier_at_entry);

        // This is the guarantee the current per-prefix audit actually
        // establishes: no simultaneous affine view recovers the operand.
        for (prefix, columns) in gray_trace.chunks_exact(wires).enumerate() {
            assert!(
                !affine_span_contains(columns, &logical),
                "operand {operand} is already affine at Gray prefix {prefix}"
            );
        }

        // Combining two times on one (unknown in advance) physical wire
        // changes the answer from impossible to exact over all 2^11 input
        // states.
        let witness = same_wire_space_time_witness(&gray_trace, wires, &needed_delta)
            .unwrap_or_else(|| panic!("no Gray space-time witness for operand {operand}"));
        eprintln!(
            "operand {operand} = entry carrier {operand} XOR wire {}@prefix {} XOR \
             wire {}@prefix {} XOR {} (all {} states)",
            witness.0,
            witness.1,
            witness.0,
            witness.2,
            witness.3 as u8,
            1usize << wires
        );

        // Proper per-monomial expansion never gathers the complete mask on
        // one borrowed wire, so the same short aggregate-mask identity must
        // be absent in the otherwise identical control.
        assert!(
            same_wire_space_time_witness(&expanded_trace, wires, &needed_delta).is_none(),
            "expanded control unexpectedly has the Gray aggregate-mask witness"
        );
    }
}

#[test]
fn prod_fold_cg_emits_its_fragments_out_of_odometer_order() {
    // The fold's fragments all XOR into the target value's two carriers
    // and read nothing else it writes, so their order is free — and the
    // deterministic odometer order is a static per-gate progress clock
    // (consecutive fragments share atom prefixes) readable with no
    // execution at all. Both fold paths must shuffle it away.
    let n = 3;
    let carrier_total = 2 * n;
    let pairs = vec![(0usize, 1usize), (2, 3), (4, 5)];
    let state = GadgetState { n, pairs };
    // 2 controls x (2 carriers + 2 mask atoms) = 16 fragments per fold.
    let gate = XGate::from_g57([0, 1, 2]);
    for cap in [0usize, 2] {
        let cfg = ProdConfig {
            k: 2,
            deg: 2,
            band: 6,
            rsrc: 0,
            max_width: cap,
            ..ProdConfig::off()
        };
        let mut shuffled = 0usize;
        for seed in 0..8u64 {
            let mut rng = StdRng::seed_from_u64(0x5017_0000 + seed);
            let mut ledger = ProdLedger::new(n, &cfg, carrier_total, None);
            let mut ramp = Vec::new();
            ledger.inject_all(&state, &mut rng, &mut ramp);
            let mut fold = Vec::new();
            ledger.fold_cg(&gate, &state, &mut rng, &mut fold);
            // 4 atoms per control gives 16 combinations, but a fold
            // fragment is DROPPED when two atoms meet on one wire with
            // opposite polarity (the conjunction is identically zero), and
            // at band 6 with two degree-2 masks per value such a collision
            // is ordinary. Assert only that the fold is wide enough for the
            // run-length test below to mean something -- pinning the exact
            // count makes this test a hostage to the RNG stream, which any
            // change in how earlier gates are spelled will shift.
            assert!(fold.len() >= 12, "expected a wide fold, got {}", fold.len());
            // Odometer order walks the first control's atoms fastest, so
            // it emits long runs that read the same second-control atom.
            // A shuffled order breaks those runs.
            let second: Vec<Vec<u16>> = fold
                .iter()
                .map(|g| {
                    let mut ws: Vec<u16> = g
                        .ctrls
                        .iter()
                        .map(|&(w, _)| w)
                        .filter(|&w| w >= 4)
                        .collect();
                    ws.sort_unstable();
                    ws
                })
                .collect();
            let runs = 1 + second.windows(2).filter(|w| w[0] != w[1]).count();
            if runs > second.len() / 2 {
                shuffled += 1;
            }
        }
        assert!(
            shuffled >= 7,
            "cap={cap}: fold fragments still come out in odometer order \
             ({shuffled}/8 seeds shuffled)"
        );
    }
}

#[test]
fn prod_band_roll_relocates_the_band_and_preserves_every_value() {
    // The roll is RG2's move applied to a band variable: the emitted
    // 3-CNOT swap must leave every logical value's decode unchanged under
    // the updated bookkeeping, for every input, and the band must
    // actually end up somewhere else — including inside the carrier
    // space, with the vacated wire becoming a carrier.
    let n = 3;
    let carrier_total = 2 * n;
    let band = 4usize;
    let total = carrier_total + band;
    let cfg = ProdConfig {
        k: 1,
        deg: 2,
        band,
        rsrc: 0,
        roll: 1,
        ..ProdConfig::off()
    };
    let mut left_home = 0usize;
    for seed in 0..24u64 {
        let mut rng = StdRng::seed_from_u64(0x0011_0000 + seed);
        let mut state = GadgetState {
            n,
            pairs: vec![(0usize, 1usize), (2, 3), (4, 5)],
        };
        let mut ledger = ProdLedger::new(n, &cfg, carrier_total, None);
        let mut ramp = Vec::new();
        ledger.inject_all(&state, &mut rng, &mut ramp);
        for _ in 0..6 {
            let pairs_before = state.pairs.clone();
            let slots_before = ledger.slots.clone();
            let consts_before = ledger.consts.clone();
            let loc_before = ledger.loc.clone();
            let mut moved = Vec::new();
            ledger.roll(&mut state, &mut rng, &mut moved);
            // RG2's three transvections, each either a plain CNOT or the
            // two-term form, so no wire is written only by width-1 gates.
            // The two-term form is spelled in the g57 vocabulary: the
            // same-polarity half costs g57+CNOT and the mixed-polarity
            // half one g57, so that branch is 3 gates and a roll spans
            // 3 (all CNOT) to 9 (all two-term) gates.
            assert!(
                (3..=9).contains(&moved.len()),
                "a roll is three transvections, got {}",
                moved.len()
            );
            // Width 1..=2 still holds, but comp=1 is now EXPECTED: the
            // two-term transvection is spelled in the g57 vocabulary, and
            // a g57 carries comp. What must hold is that the pair leaves no
            // NET constant -- a roll has no ledger to defer one to -- and
            // that is what the decode check below actually verifies.
            assert!(moved.iter().all(|g| (1..=2).contains(&g.width())));
            for input in 0..(1u64 << total) {
                let out_state = eval_u64(&moved, input);
                for v in 0..n {
                    let before = prod_decode(
                        input,
                        v,
                        &pairs_before,
                        &slots_before,
                        &consts_before,
                        &loc_before,
                    );
                    let after = prod_decode(
                        out_state,
                        v,
                        &state.pairs,
                        &ledger.slots,
                        &ledger.consts,
                        &ledger.loc,
                    );
                    assert_eq!(before, after, "seed={seed} value={v}: roll changed a value");
                }
            }
            // Carriers and band wires stay a partition of the wire space.
            let mut occupied: Vec<u16> = ledger.loc.clone();
            for &(s, p) in &state.pairs {
                occupied.push(s as u16);
                occupied.push(p as u16);
            }
            occupied.sort_unstable();
            occupied.dedup();
            assert_eq!(
                occupied.len(),
                total,
                "carriers and band overlap after a roll"
            );
        }
        if ledger.loc.iter().any(|&w| (w as usize) < carrier_total) {
            left_home += 1;
        }
    }
    assert!(
        left_home >= 20,
        "the band almost never leaves its home range ({left_home}/24)"
    );
}

#[test]
fn prod_rolling_band_gadget_is_correct_and_writes_the_band_in_the_body() {
    // End to end with --prod-roll: the endpoint contract must survive for
    // ARBITRARY junk on every non-data wire, and the roll must actually
    // change the emitted circuit's write profile on the band. ("Every wire
    // is written somewhere" would be vacuous — the two band fills already
    // write every band wire even at roll 0; the comparison below is
    // against the same seed with rolls off.)
    let n = MASKED_TEST_N;
    let main = masked_test_main();
    let mask = (1u64 << n) - 1;
    let band = 6usize;
    for (max_width, fill_nl) in [(0usize, 2usize), (2, 2)] {
        let cfg = |roll| ProdConfig {
            k: 1,
            deg: 2,
            k_hi: 1,
            deg_hi: 3,
            band,
            rsrc: 1,
            max_width,
            fill_nl,
            roll,
            src_dist: 0,
            src_horizon: 0,
            src_lo: 0,
            src_hi: 0,
            fill_pivots: 0,
            g57_narrow: 0,
            ladder_cap: 0,
            cg_jitter: 0,
            rung_menu: 0,

            epoch: 0,
            refill_data: 0,
            single: 0,
            gray_fold: 0,
            swap_refresh: 0,
            close_slice: 0,
        };
        let band_writes = |g: &CnotCircuit| -> usize {
            g.gates
                .iter()
                .filter(|gate| (gate.target as usize) >= 2 * n)
                .count()
        };
        for seed in 0..3u64 {
            let mut rng = StdRng::seed_from_u64(0x0b0d_0000 + seed);
            let rolled = gadgetize_cnot(&main, n, 2, &MaskConfig::off(), &cfg(1), &mut rng);
            assert_eq!(rolled.num_wires, 2 * n + band, "rolls cost no wires");
            for input in 0..(1u64 << rolled.num_wires) {
                let expected = main.evaluate((input & mask) as usize) as u64 & mask;
                assert_eq!(
                    eval_u64(&rolled.gates, input) & mask,
                    expected,
                    "max_width={max_width} seed={seed} input={input:#x}"
                );
            }
            let mut rng = StdRng::seed_from_u64(0x0b0d_0000 + seed);
            let still = gadgetize_cnot(&main, n, 2, &MaskConfig::off(), &cfg(0), &mut rng);
            assert!(
                band_writes(&rolled) > 2 * band_writes(&still),
                "max_width={max_width} seed={seed}: rolling barely touched the band \
                 ({} writes vs {} without rolls)",
                band_writes(&rolled),
                band_writes(&still)
            );
        }
    }
}

#[test]
fn prod_narrow_gadget_round_trips_in_the_g57_vocabulary() {
    // Full narrow gadget (mixed [2,3] plan + nonlinear cascaded band fill
    // + mirror): every gate is within the phase-A DB width, no wire is
    // added over the wide build, and the endpoint contract holds for
    // ARBITRARY junk on every non-data wire (dirty borrows, nothing
    // pinned). Also records the true-g57 share.
    let n = MASKED_TEST_N;
    let main = masked_test_main();
    let mask = (1u64 << n) - 1;
    let cfg = ProdConfig {
        k: 1,
        deg: 2,
        k_hi: 1,
        deg_hi: 3,
        band: 6,
        rsrc: 1,
        max_width: 2,
        fill_nl: 2,
        roll: 0,
        src_dist: 0,
        src_horizon: 0,
        src_lo: 0,
        src_hi: 0,
        fill_pivots: 0,
        g57_narrow: 0,
        ladder_cap: 0,
        cg_jitter: 0,
        rung_menu: 0,

        epoch: 0,
        refill_data: 0,
        single: 0,
        gray_fold: 0,
        swap_refresh: 0,
        close_slice: 0,
    };
    for seed in 0..3u64 {
        let mut rng = StdRng::seed_from_u64(0xa550_0000 + seed);
        let g = gadgetize_cnot(&main, n, 2, &MaskConfig::off(), &cfg, &mut rng);
        // No scratch region: narrow mode costs exactly zero extra wires.
        assert_eq!(g.num_wires, 2 * n + 6);
        let mut g57s = 0usize;
        for gate in &g.gates {
            assert!(!gate.ctrls.is_empty(), "bare X in narrow gadget");
            assert!(gate.width() <= 2, "wide gate in narrow gadget: {gate:?}");
            let mut pols: Vec<bool> = gate.ctrls.iter().map(|&(_, p)| p).collect();
            pols.sort_unstable();
            if gate.width() == 2 && gate.comp && pols == vec![false, true] {
                g57s += 1;
            }
        }
        // Ladder rungs must be EXACT, and an exact 2-control conjunction
        // is not a sum of g57s (each g57 carries a constant 1: an odd
        // count leaves a stray 1, an even count collapses the monomials
        // to a plain XOR). So rungs are comp=0 width-2 gates — still
        // inside the phase-A DB width, which filters on width, not comp.
        assert!(g57s > 0, "seed={seed}: no g57s at all");
        // Full domain: arbitrary junk on carriers and band alike.
        for input in 0..(1u64 << g.num_wires) {
            let expected = main.evaluate((input & mask) as usize) as u64 & mask;
            assert_eq!(eval_u64(&g.gates, input) & mask, expected, "seed={seed}");
        }
    }
}

/// Distributed sourcing: the encoding with NO band at all.
///
/// Exactness is the barrier's own test. A mask whose source wire is
/// written between injection and strip fails to cancel — the strip emits
/// the same conjunction over a bit that has since changed — so the value
/// decodes wrong and the endpoint moves. Running the full input domain
/// with RG traffic (rg_freq 2, so RG1/RG2 re-pair and RG3 refreshes fire
/// throughout) and re-source churn on top exercises every release path.
/// The gate-local non-completeness that no endpoint test can see is
/// asserted at emission time inside `debug_check_fragment`.
#[test]
fn prod_distributed_sourcing_is_exact_and_costs_no_wires() {
    // Width 6, not the 4-wire fixture: with one factor per owning value,
    // a degree-3 slot needs three values besides its own, so n = 4 leaves
    // the draw no freedom at all (and the dedup set nothing to draw from).
    let n = 6usize;
    let main = masked_test_main_wide(n);
    let mask = (1u64 << n) - 1;
    for max_width in [0usize, 2] {
        let cfg = ProdConfig {
            k: 1,
            deg: 2,
            k_hi: 1,
            deg_hi: 3,
            band: 0,
            rsrc: 1,
            max_width,
            fill_nl: 0,
            roll: 0,
            src_dist: 1,
            src_horizon: 0,
            src_lo: 0,
            src_hi: 0,
            fill_pivots: 0,
            g57_narrow: 0,
            ladder_cap: 0,
            cg_jitter: 0,
            rung_menu: 0,

            epoch: 0,
            refill_data: 0,
            single: 0,
            gray_fold: 0,
            swap_refresh: 0,
            close_slice: 0,
        };
        for seed in 0..2u64 {
            let mut rng = StdRng::seed_from_u64(0x0d15_0000 + seed);
            let g = gadgetize_cnot(&main, n, 2, &MaskConfig::off(), &cfg, &mut rng);
            assert_eq!(
                g.num_wires,
                2 * n,
                "distributed sourcing must not widen the gadget"
            );
            assert!(
                g.gates.iter().all(|gate| !gate.ctrls.is_empty()),
                "distributed build must not contain a bare X"
            );
            // Arbitrary junk on every non-data wire, exhaustively.
            for input in 0..(1u64 << g.num_wires) {
                let expected = main.evaluate((input & mask) as usize) as u64 & mask;
                assert_eq!(
                    eval_u64(&g.gates, input) & mask,
                    expected,
                    "max_width={max_width} seed={seed} input={input:#x}"
                );
            }
        }
    }
}

/// Single-carrier decode `v = c_v ^ masks ^ κ`: exact under arbitrary band
/// junk, on n carriers instead of 2n. Exactness is the real test here —
/// the strip only cancels if every mask term still denotes the bit it did
/// at injection, which is what makes the frozen band load-bearing, and the
/// relocations must leave value v back on wire v at the end.
#[test]
fn prod_single_carrier_is_exact_on_half_the_wires() {
    let n = 6usize;
    let main = masked_test_main_wide(n);
    let mask = (1u64 << n) - 1;
    // [1,2,3,3] and [1,2,2,3]: one linear term, the rest nonlinear.
    // Rolls on: a roll can leave a value sitting on a former band wire, so
    // the final routing has to be a full permutation, not a carrier-space
    // one. That is exactly what this exercises.
    for (k, deg, k_hi, deg_hi, roll) in [
        (1usize, 2usize, 2usize, 3usize, 0usize),
        (2, 2, 1, 3, 0),
        (1, 2, 2, 3, 1),
    ] {
        let cfg = ProdConfig {
            k,
            deg,
            k_hi,
            deg_hi,
            band: 8,
            rsrc: 1,
            max_width: 0,
            fill_nl: 2,
            roll,
            src_dist: 0,
            src_horizon: 0,
            src_lo: 0,
            src_hi: 0,
            fill_pivots: 0,
            g57_narrow: 0,
            ladder_cap: 0,
            cg_jitter: 0,
            rung_menu: 0,

            epoch: 0,
            refill_data: 0,
            single: 1,
            gray_fold: 0,
            swap_refresh: 0,
            close_slice: 0,
        };
        for seed in 0..3u64 {
            let mut rng = StdRng::seed_from_u64(0x51_0000 + seed);
            let g = gadgetize_cnot_single(&main, n, 2, &cfg, &mut rng);
            assert_eq!(g.num_wires, n + 8, "single carrier: n carriers, not 2n");
            assert!(
                g.gates.iter().all(|gate| !gate.ctrls.is_empty()),
                "single-carrier build must not contain a bare X"
            );
            for input in 0..(1u64 << g.num_wires) {
                let expected = main.evaluate((input & mask) as usize) as u64 & mask;
                assert_eq!(
                    eval_u64(&g.gates, input) & mask,
                    expected,
                    "plan [{deg}x{k},{deg_hi}x{k_hi}] seed={seed} input={input:#x}"
                );
            }
        }
    }
}

/// Selective laddering: fold fragments of width in (2, cap] are realized
/// over BORROWED DIRTY carriers instead of as one wide gate.
///
/// The borrows are the whole risk. A ladder parks partial products on
/// wires it does not own, so it is exact only if every borrow is visited
/// an even number of times and restored before anything else reads it --
/// and the borrow pool now has to dodge the target's sibling carrier and
/// every operand's sibling, or one gate ends up seeing both carriers of a
/// single value. Run the full input domain (band junk included, since the
/// high wires are unconstrained) at several ceilings, and check that the
/// fossil count actually falls -- an exactness test alone would pass on a
/// ladder_cap that silently did nothing.
#[test]
fn prod_laddering_is_exact_and_removes_wide_gates() {
    let n = 6usize;
    let main = masked_test_main_wide(n);
    let mask = (1u64 << n) - 1;
    let build = |ladder_cap: usize, seed: u64| {
        let cfg = ProdConfig {
            k: 1,
            deg: 2,
            k_hi: 2,
            deg_hi: 3,
            band: 8,
            rsrc: 1,
            max_width: 0,
            fill_nl: 2,
            roll: 1,
            src_dist: 0,
            src_horizon: 0,
            src_lo: 0,
            src_hi: 0,
            fill_pivots: 0,
            g57_narrow: 1,
            ladder_cap,
            cg_jitter: 0,
            rung_menu: 0,
            epoch: 0,
            refill_data: 0,
            single: 1,
            gray_fold: 0,
            swap_refresh: 0,
            close_slice: 0,
        };
        let mut rng = StdRng::seed_from_u64(0x1add_0000 + seed);
        gadgetize_cnot_single(&main, n, 2, &cfg, &mut rng)
    };
    let wide = |g: &CnotCircuit| g.gates.iter().filter(|x| x.ctrls.len() > 2).count();
    let base = wide(&build(0, 0));
    for cap in [3usize, 4, 6] {
        let g = build(cap, 0);
        assert!(
            wide(&g) < base,
            "ladder_cap {cap} removed no wide gates ({} vs baseline {base})",
            wide(&g)
        );
        for seed in 0..2u64 {
            let g = build(cap, seed);
            for input in 0..(1u64 << g.num_wires) {
                let expected = main.evaluate((input & mask) as usize) as u64 & mask;
                assert_eq!(
                    eval_u64(&g.gates, input) & mask,
                    expected,
                    "ladder_cap={cap} seed={seed} input={input:#x}"
                );
            }
        }
    }
    // A ceiling above every fragment width must leave nothing wide behind.
    assert_eq!(
        wide(&build(64, 0)),
        0,
        "an unbounded ceiling still left wide gates"
    );
}

/// The DEFAULT config is the hardened construction, not a bare encoding.
///
/// `production_single` spent a day as a free-standing constant with no
/// callers, so every lever it named was off in every circuit anyone built
/// while the docs described it as "the validated production setting". Both
/// entry points now build from it, and this pins the values so a revert to
/// the old all-zero defaults fails here rather than silently shipping a
/// materially weaker gadget.
#[test]
fn production_preset_is_the_hardened_construction() {
    let p = ProdConfig::production_single();
    assert!(p.enabled(), "the default must have the encoding ON");
    // [2,2,2,3] -- three degree-2 mask terms and one degree-3, replacing
    // [2,3,3]. A degree-2 atom is the stronger STATISTICAL masker (piling-up
    // factor 0.5 against 0.75) and the weaker ALGEBRAIC one (it sits inside
    // a degree-2 exact adversary's span), so the mix trades one against the
    // other: eps 0.09375 against 0.28125, measured leak 3.2x lower, at 14%
    // FEWER gates. The single surviving degree-3 atom is what keeps the plan
    // out of degree-2 exact reach; drop it and the value is recoverable
    // exactly, which is why deg_hi and k_hi >= 1 are pinned here.
    assert_eq!(
        (p.k, p.deg, p.k_hi, p.deg_hi),
        (3, 2, 1, 3),
        "plan is [2,2,2,3]"
    );
    assert!(
        p.k_hi >= 1 && p.deg_hi >= 3,
        "at least one degree-3 atom, or the value drops into degree-2 exact range"
    );
    assert_eq!(p.single, 1, "single-carrier decode");
    assert_eq!(p.band, 0, "band 0 == match the value count");
    assert_eq!(p.band_size(128), 128, "band 0 must resolve to n");
    assert!(
        p.rsrc >= 1,
        "single-carrier mode needs a representation refresh"
    );
    assert_eq!(p.fill_nl, 2, "nonlinear band fill");
    assert_eq!(
        p.roll, 1,
        "rolling band -- without it the write census separates"
    );
    assert_eq!(
        p.g57_narrow, 1,
        "narrow fragments in the store's vocabulary"
    );
    // Selective laddering at cap 3: the expanded fold's wide product
    // fragments must be re-spelled into the g57/CNOT vocabulary for
    // fmix (cap 4's extra narrowness arrives as store-weak plain conj-2
    // gates at twice the size), and the ladder's scratch is the band
    // pool under swap mode (live-carrier borrows exposed data states in
    // the chain deltas).
    assert_eq!(
        p.ladder_cap, 3,
        "expanded-fold production needs the selective ladder"
    );
    // The Gray fold is ON: the fold emits no wide fragment at all, and
    // store-reachability goes 31.55% -> 95.47% (97.53% at this mask plan).
    assert_eq!(p.gray_fold, 1, "the Gray-code fold is the default CG");
    assert_eq!(
        p.rung_menu, 1,
        "free spelling variability is on -- it costs nothing"
    );
    assert_eq!(p.cg_jitter, 50, "block-count entropy at its maximum");
    // A frozen band is recoverable by FUNCTION LIFETIME alone, so some
    // channel must turn the band's functions over. Since 2026-08-24 that
    // is the drain set rather than `epoch`: it steers retirements the fold
    // already makes instead of paying to release a live variable, so it
    // buys ~2x the turnovers for less than `epoch` cost. Exactly one of
    // the two should be running -- both is double payment, neither is a
    // frozen band.
    assert!(
        (p.epoch > 0) ^ (p.swap_refresh > 0 && drain_cap(&p, p.band_size(128)) > 0),
        "a frozen band is recoverable by function lifetime: run the drain set \
         (swap_refresh > 0) or the epoch channel, not both and not neither"
    );
    assert_eq!(p.fill_pivots, 0, "band = n leaves the pivot block no room");
    // The 2026-08-20 redesign: without the per-gate swap the masks cancel
    // in every fold's before/after XOR (carrier delta == source delta,
    // measured 100% on linear gates); without the closing block the
    // zero-slice phase exists at the input port only.
    assert_eq!(
        p.swap_refresh, 3,
        "per-gate mask swap-with-refresh is the default, at the 3 retirement \
         sides that buy 14.7 band turnovers for +0.3% gates (2 = the 2026-08-20 stream)"
    );
    assert_eq!(
        p.close_slice, 1,
        "the closing zero-slice block is the default"
    );
}

#[test]
fn no_gray_phase_a_preset_changes_only_the_fold_strategy() {
    let gray = ProdConfig::production_single();
    let safe = ProdConfig::production_single_no_gray_phase_a();
    assert_eq!(
        safe.gray_fold, 0,
        "must not gather an aggregate operand mask"
    );
    assert_eq!(safe.ladder_cap, 4, "measured selective-narrow ceiling");

    let mut expected = gray;
    expected.gray_fold = 0;
    expected.ladder_cap = 4;
    assert_eq!(
        safe, expected,
        "preset drifted beyond its two measured levers"
    );
}

/// `strip_all`'s constant discharge must not read the target's SIBLING.
///
/// It emits `target ^= !u` then `target ^= u` to realize a bare constant
/// without an X gate, and it drew `u` from the index range
/// `0..carrier_total` excluding only the target -- so on a two-carrier build
/// the target value's other carrier was a legal draw, and that first gate
/// then read one carrier of a value while writing the other: a whole
/// sharing inside one gate. It is the role-versus-index confusion this file
/// has now had three times, invisible under `--prod-single 1` because there
/// is no sibling to hit, which is why the suite passed with it present.
///
/// Tested on `strip_all` directly rather than on a finished gadget: the
/// sharing BOOKENDS legitimately touch both carriers (that is how the
/// sharing is created), so a whole-circuit scan cannot separate the two.
#[test]
fn prod_strip_constant_never_reads_the_targets_sibling() {
    let n = MASKED_TEST_N;
    let cfg = ProdConfig {
        k: 1,
        deg: 2,
        k_hi: 1,
        deg_hi: 3,
        band: 10,
        rsrc: 1,
        fill_nl: 2,
        single: 0, // two carriers: this is the only mode with a sibling
        ..ProdConfig::off()
    };
    let carrier_total = 2 * n;
    let pairs: Vec<(usize, usize)> = (0..n).map(|v| (2 * v, 2 * v + 1)).collect();
    let state = GadgetState { n, pairs };
    for seed in 0..64u64 {
        let mut rng = StdRng::seed_from_u64(0x5721_0000 + seed);
        let mut ledger = ProdLedger::new(n, &cfg, carrier_total, None);
        let mut ramp = Vec::new();
        ledger.inject_all(&state, &mut rng, &mut ramp);
        // Force every value to owe a constant, so the discharge path runs
        // for all of them rather than whichever parity happened to land.
        for v in 0..n {
            ledger.consts[v] = true;
        }
        let mut out = Vec::new();
        ledger.strip_all(&state, &mut rng, &mut out);
        for (i, gate) in out.iter().enumerate() {
            let mut touched: Vec<u16> = gate.ctrls.iter().map(|&(w, _)| w).collect();
            touched.push(gate.target);
            for v in 0..n {
                let (c0, c1) = (2 * v as u16, 2 * v as u16 + 1);
                assert!(
                    !(touched.contains(&c0) && touched.contains(&c1)),
                    "seed={seed} strip gate {i} ({gate:?}) holds both carriers \
                     {c0},{c1} of value {v}"
                );
            }
        }
    }
}

/// Retire-and-refill epochs: a band variable's VALUE changes mid-body.
///
/// This is the exactness test that matters for the mechanism: a refill
/// rewrites a wire that masks were reading a moment ago, so if the release
/// step ever misses a live slot, that slot's strip cancels the wrong
/// product and the endpoint moves. Run over the full input domain with
/// arbitrary band junk, at both refill compositions (band-internal and
/// carrier-injecting) and with rolling on, since a roll relocates the very
/// variable the next epoch retires.
#[test]
fn prod_retire_refill_is_exact_under_arbitrary_junk() {
    let n = MASKED_TEST_N;
    let main = masked_test_main();
    let mask = (1u64 << n) - 1;
    for (epoch, refill_data, roll) in [
        (1usize, 0usize, 0usize),
        (1, 100, 0),
        (2, 50, 1),
        (1, 50, 1),
    ] {
        let cfg = ProdConfig {
            k: 1,
            deg: 2,
            k_hi: 1,
            deg_hi: 3,
            band: 6,
            rsrc: 1,
            max_width: 0,
            fill_nl: 2,
            roll,
            src_dist: 0,
            src_horizon: 0,
            src_lo: 0,
            src_hi: 0,
            fill_pivots: 1,
            g57_narrow: 0,
            ladder_cap: 0,
            cg_jitter: 0,
            rung_menu: 0,
            epoch,
            refill_data,
            single: 0,
            gray_fold: 0,
            swap_refresh: 0,
            close_slice: 0,
        };
        for seed in 0..3u64 {
            let mut rng = StdRng::seed_from_u64(0xbeef_0000 + seed);
            let g = gadgetize_cnot(&main, n, 2, &MaskConfig::off(), &cfg, &mut rng);
            assert!(
                g.gates.iter().all(|gate| !gate.ctrls.is_empty()),
                "retire-refill must not emit a bare X"
            );
            for input in 0..(1u64 << g.num_wires) {
                let expected = main.evaluate((input & mask) as usize) as u64 & mask;
                assert_eq!(
                    eval_u64(&g.gates, input) & mask,
                    expected,
                    "epoch={epoch} refill_data={refill_data} roll={roll} seed={seed} input={input:#x}"
                );
            }
        }
    }
}

/// The point of the design: no wire is quiet, because no wire has the
/// dedicated source role. In the band build the band wires are written
/// only by the two fills — a census separates them from the carriers by a
/// single threshold. Here every wire must be written by BODY traffic.
#[test]
fn prod_distributed_sourcing_leaves_no_quiet_wire() {
    let n = 6usize;
    let main = masked_test_main_wide(n);
    let cfg = |src_dist| ProdConfig {
        k: 1,
        deg: 2,
        k_hi: 1,
        deg_hi: 3,
        band: 8,
        rsrc: 1,
        max_width: 0,
        fill_nl: 2,
        roll: 0,
        src_dist,
        src_horizon: 0,
        src_lo: 0,
        src_hi: 0,
        fill_pivots: 0,
        g57_narrow: 0,
        ladder_cap: 0,
        cg_jitter: 0,
        rung_menu: 0,

        epoch: 0,
        refill_data: 0,
        single: 0,
        gray_fold: 0,
        swap_refresh: 0,
        close_slice: 0,
    };
    let mut rng = StdRng::seed_from_u64(0x0d16_0001);
    let dist = gadgetize_cnot(&main, n, 2, &MaskConfig::off(), &cfg(1), &mut rng);
    let mut rng = StdRng::seed_from_u64(0x0d16_0001);
    let band = gadgetize_cnot(&main, n, 2, &MaskConfig::off(), &cfg(0), &mut rng);
    assert_eq!(dist.num_wires, 2 * n, "no band wires");
    assert_eq!(band.num_wires, 2 * n + 8, "band build keeps its band");

    // Body = everything strictly between the ports, so the input/output
    // fills (which write every band wire in the band build) do not mask
    // the distinction being measured.
    let writes = |g: &CnotCircuit, wires: usize| -> Vec<usize> {
        let lo = g.gates.len() / 4;
        let hi = g.gates.len() - g.gates.len() / 4;
        let mut w = vec![0usize; wires];
        for gate in &g.gates[lo..hi] {
            w[gate.target as usize] += 1;
        }
        w
    };
    let dist_w = writes(&dist, dist.num_wires);
    assert!(
        dist_w.iter().all(|&c| c > 0),
        "distributed build left an unwritten wire in the body: {dist_w:?}"
    );
    let band_w = writes(&band, band.num_wires);
    assert!(
        band_w[2 * n..].iter().all(|&c| c == 0),
        "band wires should be body-static in the band build (the weakness \
         distributed sourcing removes): {:?}",
        &band_w[2 * n..]
    );
}

/// The reserved pivot block makes the band JOINTLY uniform, not merely
/// balanced wire by wire.
///
/// This is the property every statistical claim about the encoding needs
/// and the one the marginal test cannot see: a mask multiplies three band
/// wires, so what matters is the joint law. Checking it means checking
/// EVERY nonempty subset XOR is balanced — a subset that is biased is a
/// direction in which the band is predictable.
///
/// The legacy draw fails this (pivots are drawn with replacement and only
/// a wire's OWN pivot is excluded from its own material), which is why the
/// comparison against it is part of the test rather than folklore.
#[test]
fn prod_reserved_pivots_make_the_band_jointly_uniform() {
    let n = 10usize;
    let b = 5usize;
    let band: Vec<u16> = (n as u16..(n + b) as u16).collect();
    let subset_bias = |gates: &[XGate]| -> f64 {
        let mut worst = 0f64;
        for mask in 1u32..(1 << b) {
            let ones = (0..(1u64 << n))
                .filter(|&x| {
                    let st = eval_u64(gates, x);
                    let mut parity = 0u64;
                    for (i, &w) in band.iter().enumerate() {
                        if mask >> i & 1 == 1 {
                            parity ^= (st >> w) & 1;
                        }
                    }
                    parity == 1
                })
                .count() as f64;
            let bias = (ones / (1u64 << n) as f64 - 0.5).abs();
            if bias > worst {
                worst = bias;
            }
        }
        worst
    };
    let (mut reserved_worst, mut legacy_worst) = (0f64, 0f64);
    for seed in 0..12u64 {
        let mut rng = StdRng::seed_from_u64(0x91_0000 + seed);
        let mut g = Vec::new();
        emit_band_fill_nl_pivots(n, &band, 2, true, &mut rng, &mut g);
        reserved_worst = reserved_worst.max(subset_bias(&g));

        let mut rng = StdRng::seed_from_u64(0x91_0000 + seed);
        let mut g = Vec::new();
        emit_band_fill_nl_pivots(n, &band, 2, false, &mut rng, &mut g);
        legacy_worst = legacy_worst.max(subset_bias(&g));
    }
    assert_eq!(
        reserved_worst, 0.0,
        "reserved pivots must make EVERY subset XOR exactly balanced; worst bias {reserved_worst}"
    );
    assert!(
        legacy_worst > 0.0,
        "the legacy draw is supposed to be jointly biased — if this fires, the \
         comparison has stopped being meaningful (worst bias {legacy_worst})"
    );
    println!(
        "[pivot-block] worst subset bias: reserved {reserved_worst:.4} vs legacy {legacy_worst:.4}"
    );
}

#[test]
fn prod_band_fill_nl_is_balanced_and_nonlinear() {
    // Every band wire's fill must be exactly balanced over uniform data
    // (the pivot guarantee), and the cascade must actually produce
    // nonlinearity in at least one band wire.
    let n = 10;
    // Deliberately NOT contiguous and not in wire order: after a roll the
    // fill's wire list is an arbitrary set (the mirror fill takes it as it
    // finds it), and the cascade's "earlier band wire" bookkeeping must
    // key on position in the list, not on wire index arithmetic.
    let band: Vec<u16> = vec![14, 10, 17, 11, 16, 12, 13, 15];
    let mut rng = StdRng::seed_from_u64(0xf111_0001);
    let mut gates = Vec::new();
    emit_band_fill_nl(n, &band, 2, &mut rng, &mut gates);
    let f = |x: u64| eval_u64(&gates, x);
    let mut any_nonlinear = false;
    for bw in band.iter().map(|&w| w as usize) {
        let bit = |x: u64| (f(x) >> bw) & 1;
        let ones: u64 = (0..(1u64 << n)).map(bit).sum();
        assert_eq!(ones, 1 << (n - 1), "band wire {bw} fill is biased");
        'nl: for i in 0..n {
            for j in (i + 1)..n {
                let (ei, ej) = (1u64 << i, 1u64 << j);
                if bit(ei ^ ej) ^ bit(ei) ^ bit(ej) ^ bit(0) != 0 {
                    any_nonlinear = true;
                    break 'nl;
                }
            }
        }
    }
    assert!(
        any_nonlinear,
        "cascaded fill produced no nonlinear band wire"
    );
}

#[test]
fn prod_gadgetize_cnot_preserves_the_first_n_wires() {
    let n = MASKED_TEST_N;
    let main = masked_test_main();
    let mask = (1u64 << n) - 1;
    for seed in 0..8u64 {
        let mut rng = StdRng::seed_from_u64(0x960d_0000 + seed);
        let prodded = gadgetize_cnot(
            &main,
            n,
            2,
            &MaskConfig::off(),
            &prod_test_config(),
            &mut rng,
        );
        assert_eq!(prodded.num_wires, 2 * n + 6);
        assert!(
            prodded.gates.iter().all(|g| !g.ctrls.is_empty()),
            "prod body must not contain a bare X"
        );
        for input in 0..(1u64 << prodded.num_wires) {
            let expected = main.evaluate((input & mask) as usize) as u64 & mask;
            assert_eq!(
                eval_u64(&prodded.gates, input) & mask,
                expected,
                "seed={seed}"
            );
        }
        // Same seed, prod off: the encoding must actually have paid gates.
        let mut rng = StdRng::seed_from_u64(0x960d_0000 + seed);
        let plain = gadgetize_cnot(
            &main,
            n,
            2,
            &MaskConfig::off(),
            &ProdConfig::off(),
            &mut rng,
        );
        assert!(prodded.gates.len() > plain.gates.len());
    }
}

#[test]
fn prod_gadgetize_xgates_preserves_the_low_wires() {
    let n = MASKED_TEST_N;
    let mask = (1u64 << n) - 1;
    let source = vec![
        XGate::from_g57([0, 1, 2]),
        XGate::cnot(0, 3),
        XGate::conj(2, [(0u16, true), (1u16, true)]).unwrap(),
        XGate::conj(1, [(3u16, false)]).unwrap(),
        XGate::from_g57([3, 0, 1]),
        XGate::x_gate(2),
        XGate::cnot(1, 0),
        XGate::from_g57([0, 2, 3]),
        XGate::conj(0, [(1u16, false), (2u16, true)]).unwrap(),
        XGate::from_g57([1, 3, 2]),
        XGate::cnot(2, 1),
    ];
    for seed in 0..8u64 {
        let mut rng = StdRng::seed_from_u64(0x960e_0000 + seed);
        let g = gadgetize_xgates(
            &source,
            n,
            2,
            &MaskConfig::off(),
            &prod_test_config(),
            &mut rng,
        );
        assert_eq!(g.num_wires, 2 * n + 6);
        for input in 0..(1u64 << g.num_wires) {
            let expected = eval_u64(&source, input & mask) & mask;
            assert_eq!(
                eval_u64(&g.gates, input) & mask,
                expected,
                "input={input:#x}"
            );
        }
    }
}

#[test]
fn prod_slice_zero_gadgetize_matches_on_the_zero_slice() {
    let n = MASKED_TEST_N;
    let main = masked_test_main();
    let mask = (1u64 << n) - 1;
    for seed in 0..4u64 {
        let mut rng = StdRng::seed_from_u64(0x960f_0000 + seed);
        let transformed = gadgetize_with_slice_zero_ccnot(
            &main,
            n,
            2,
            6 * n,
            &MaskConfig::off(),
            &prod_test_config(),
            &mut rng,
        );
        assert_eq!(transformed.num_wires, 2 * n + 6);
        for x in 0..=mask {
            let expected = main.evaluate(x as usize) as u64 & mask;
            assert_eq!(eval_u64(&transformed.gates, x) & mask, expected);
        }
    }
}

#[test]
fn feistalize_cnot_moves_functionality_to_the_middle_n_wires() {
    let n = 3;
    let main = CircuitSeq {
        gates: vec![[0, 1, 2], [2, 0, 1], [1, 2, 0]],
    };
    for seed in 0..8u64 {
        let mut rng = StdRng::seed_from_u64(0xfe15_0000 + seed);
        let transformed = feistalize_cnot(&main, n, 2, &mut rng);
        assert_eq!(transformed.num_wires, 3 * n);
        let mask = (1u64 << n) - 1;
        for input in 0..(1u64 << (3 * n)) {
            let x = input & mask;
            let y = (input >> n) & mask;
            let expected = y ^ (main.evaluate(x as usize) as u64 & mask);
            assert_eq!((eval_u64(&transformed.gates, input) >> n) & mask, expected);
        }
    }
}

#[test]
fn random_fragment_preblock_has_one_and_only_one_fixed_aux_slice() {
    let n = 3;
    let mask = (1u64 << n) - 1;
    for seed in 0..8u64 {
        let mut rng = StdRng::seed_from_u64(0x51ce_0000 + seed);
        let preblock = slice_zero_random_preblock_cnot(n, 96, &mut rng);
        assert_eq!(preblock.circuit.gates.len(), 96);
        let public_y = preblock.public_y[0] & mask;
        let public_z = preblock.public_z[0] & mask;
        for y in 0..=mask {
            for z in 0..=mask {
                for x in 0..=mask {
                    let input = x | (y << n) | (z << (2 * n));
                    let output = eval_u64(&preblock.circuit.gates, input);
                    assert_eq!((output >> n) & ((1u64 << (2 * n)) - 1), input >> n);
                    if y == public_y && z == public_z {
                        assert_eq!(output, input);
                    } else {
                        assert_ne!(output & mask, x);
                    }
                }
            }
        }
    }
}

#[test]
fn ccnot_preblock_fixes_exactly_the_zero_slice() {
    // "Only the all-zero slice is fixed" is now a THEOREM of the pinned
    // target/control split (see slice_zero_ccnot_preblock), not a
    // measured tendency: exhaustively over every slice — aux, BAND, and
    // mixed — and every seed, at both widths and with the band present.
    for (n, band) in [(3usize, 0usize), (4, 0), (4, 2), (5, 3)] {
        let mask = (1u64 << n) - 1;
        let slices = 1u64 << (n + band);
        for seed in 0..8u64 {
            let mut rng = StdRng::seed_from_u64(0xcc00_0000 + seed);
            let preblock = slice_zero_ccnot_preblock(n, band, 6 * n, &mut rng);
            assert_eq!(preblock.gates.len(), 6 * n);
            assert_eq!(preblock.num_wires, 2 * n + band);
            for s in 0..slices {
                let mut identity_on_slice = true;
                for x in 0..=mask {
                    let input = x | (s << n);
                    let output = eval_u64(&preblock.gates, input);
                    assert_eq!(output >> n, s, "non-data wires must pass through");
                    if s == 0 {
                        assert_eq!(output, input, "zero slice must be fixed");
                    } else if output != input {
                        identity_on_slice = false;
                    }
                }
                if s != 0 {
                    assert!(
                        !identity_on_slice,
                        "seed={seed:#x} n={n} band={band} slice s={s:#x} is also fixed"
                    );
                }
            }
        }
    }
}

/// Brute-force version of the exactness property, valid for any gate
/// degree: enumerate every slice and every input.
fn only_zero_slice_is_fixed(gates: &[XGate], n: usize, nondata: usize) -> bool {
    let mask = (1u64 << n) - 1;
    (1..(1u64 << nondata)).all(|s| (0..=mask).any(|x| eval_u64(gates, x | (s << n)) & mask != x))
}

#[test]
fn ccnot_preblock_is_quadratic_in_the_data_off_slice() {
    // With one data control per gate the block is AFFINE in x for every
    // fixed slice, whatever the gate count: each gate becomes a constant
    // flip or a transvection, and those compose to an affine map. The
    // three-control gates are there to break that, so at least one slice
    // must show a genuine second-order term:
    //   S(a^b) ^ S(a) ^ S(b) ^ S(0)  !=  0.
    let (n, band) = (8usize, 4usize);
    let mask = (1u64 << n) - 1;
    let mut nonlinear_slices = 0usize;
    for seed in 0..4u64 {
        let mut rng = StdRng::seed_from_u64(0xcc60_0000 + seed);
        let preblock = slice_zero_ccnot_preblock(n, band, 10 * n, &mut rng);
        let s = |x: u64, slice: u64| eval_u64(&preblock.gates, x | (slice << n)) & mask;
        for slice in 1..(1u64 << (n + band)) {
            let base = s(0, slice);
            let quadratic = (0..n).any(|i| {
                ((i + 1)..n).any(|j| {
                    let (a, b) = (1u64 << i, 1u64 << j);
                    s(a ^ b, slice) ^ s(a, slice) ^ s(b, slice) ^ base != 0
                })
            });
            if quadratic {
                nonlinear_slices += 1;
            }
        }
    }
    assert!(
        nonlinear_slices > 0,
        "the preblock is affine in x on every slice — the three-control \
         gates are not doing their job"
    );
}

#[test]
fn prod_slice_zero_gadget_carries_three_control_preblock_gates() {
    // End to end: the gadget must compute C on the zero slice, and the
    // preblock's three-control gates — the ones that keep the off-slice
    // disturbance from being affine in x — must survive into the emitted
    // circuit.
    let n = 6;
    let band = 6;
    let mask = (1u64 << n) - 1;
    let main = CircuitSeq {
        gates: vec![[0, 1, 2], [2, 0, 1], [1, 2, 0], [3, 4, 5], [5, 3, 4]],
    };
    let cfg = ProdConfig {
        k: 2,
        deg: 2,
        band,
        rsrc: 1,
        roll: 1,
        ..ProdConfig::off()
    };
    for seed in 0..3u64 {
        let mut rng = StdRng::seed_from_u64(0xcc70_0000 + seed);
        let g = gadgetize_with_slice_zero_ccnot(
            &main,
            n,
            2,
            10 * n,
            &MaskConfig::off(),
            &cfg,
            &mut rng,
        );
        assert_eq!(g.num_wires, 2 * n + band);
        for x in 0..=mask {
            let expected = main.evaluate(x as usize) as u64 & mask;
            assert_eq!(eval_u64(&g.gates, x) & mask, expected, "seed={seed} x={x}");
        }
        // A preblock three-control gate: target and two controls in the
        // data half, exactly one control in the slice half, all positive.
        let has_quad = g.gates.iter().any(|gate| {
            gate.ctrls.len() == 3
                && !gate.comp
                && (gate.target as usize) < n
                && gate.ctrls.iter().all(|&(_, p)| p)
                && gate
                    .ctrls
                    .iter()
                    .filter(|&&(w, _)| (w as usize) < n)
                    .count()
                    == 2
                && gate
                    .ctrls
                    .iter()
                    .filter(|&&(w, _)| (w as usize) >= n)
                    .count()
                    == 1
        });
        assert!(
            has_quad,
            "seed={seed}: no three-control preblock gate survived"
        );
    }
}

#[test]
fn ccnot_preblock_builds_across_the_supported_widths() {
    // The constructor rejects and redraws until no nonzero slice is
    // fixed, which can fail outright when the data half is too narrow to
    // disturb every slice, so it must be exercised across the widths the
    // gadget paths actually reach — at the default 10n budget and at the
    // bare minimum of one gate per non-data wire. Where the space is small
    // enough, exactness is re-checked here too.
    for n in 3..=10usize {
        for band in [0usize, 2, 5, 8] {
            // The bare-minimum budget (one gate per slice wire) is only
            // claimed where the data half is wide enough to disturb every
            // slice with that few gates; where it is not, the constructor
            // says so with a panic rather than emitting a weak block.
            let budgets: &[usize] = if n * n / 4 >= n + band {
                &[n + band, 10 * n]
            } else {
                &[10 * n]
            };
            for &gate_count in budgets {
                let mut rng = StdRng::seed_from_u64(0xcc50_0000 + (n * 32 + band) as u64);
                let preblock = slice_zero_ccnot_preblock(n, band, gate_count, &mut rng);
                assert_eq!(preblock.gates.len(), gate_count);
                assert_eq!(preblock.num_wires, 2 * n + band);
                // Exactness by brute force where the space is small — the
                // block is quadratic in x now, so the affine shortcut the
                // fallback uses does not apply here.
                if n <= 6 && n + band <= 10 {
                    assert!(
                        only_zero_slice_is_fixed(&preblock.gates, n, n + band),
                        "n={n} band={band} gates={gate_count}: some nonzero slice is fixed"
                    );
                }
            }
        }
    }
}

#[test]
fn ccnot_preblock_band_slices_are_disturbed_like_aux_slices() {
    // The point of putting the band in the slice: flipping ONE band wire
    // must junk the data exactly as flipping one aux wire does. (Before,
    // the band was outside the preblock entirely, so every band-only
    // slice was provably fixed — a one-query aux/band distinguisher.)
    let n = 8;
    let band = 5;
    let mask = (1u64 << n) - 1;
    for seed in 0..8u64 {
        let mut rng = StdRng::seed_from_u64(0xcc40_0000 + seed);
        let preblock = slice_zero_ccnot_preblock(n, band, 10 * n, &mut rng);
        for w in 0..(n + band) {
            let s = 1u64 << w;
            let disturbed = (0..=mask).any(|x| eval_u64(&preblock.gates, x | (s << n)) & mask != x);
            assert!(disturbed, "seed={seed:#x} single-wire slice {w} is fixed");
        }
    }
}

#[test]
fn ccnot_preblock_uses_only_the_agreed_gate_shapes() {
    let n = 6;
    let band = 3;
    let gate_count = 6 * n;
    let mut rng = StdRng::seed_from_u64(0xcc10_0000);
    let preblock = slice_zero_ccnot_preblock(n, band, gate_count, &mut rng);
    let mut cnots = 0usize;
    let mut ccnots = 0usize;
    let mut quads = 0usize;
    let mut slice_controls = std::collections::HashSet::new();
    let mut targets = std::collections::HashSet::new();
    let mut data_controls = std::collections::HashSet::new();
    for gate in &preblock.gates {
        assert!(!gate.comp, "no complemented gates");
        assert!((gate.target as usize) < n, "targets stay in the data half");
        assert!(gate.ctrls.iter().all(|&(_, positive)| positive));
        targets.insert(gate.target);
        // ctrls are sorted by wire, so data controls come before slice
        // controls, and there is exactly one slice control per gate.
        let data: Vec<u16> = gate
            .ctrls
            .iter()
            .map(|&(w, _)| w)
            .filter(|&w| (w as usize) < n)
            .collect();
        let slice: Vec<u16> = gate
            .ctrls
            .iter()
            .map(|&(w, _)| w)
            .filter(|&w| (w as usize) >= n)
            .collect();
        assert_eq!(slice.len(), 1, "every gate reads exactly one slice wire");
        slice_controls.insert(slice[0]);
        data_controls.extend(data.iter().copied());
        match data.len() {
            0 => cnots += 1,
            1 => ccnots += 1,
            2 => quads += 1,
            other => panic!("unexpected data-control count {other}"),
        }
    }
    assert_eq!(cnots, gate_count / 3);
    assert_eq!(ccnots + quads, gate_count - gate_count / 3);
    // Three-control gates are what make the disturbance quadratic in x.
    assert!(quads > 0, "no three-control gates emitted");
    // Deliberately UNSTRUCTURED: a data wire is free to be a target of one
    // gate and a control of another. A disjoint target/control split would
    // buy an exactness theorem, but it also exempts the control pool from
    // ever being disturbed and lets an adversary switch the nonlinearity
    // off by zeroing it.
    assert!(
        targets.intersection(&data_controls).count() > 0,
        "targets and data controls should overlap freely"
    );
    // Every non-data wire, band included, is read by the block: a wire
    // nothing reads could not be pinned.
    assert_eq!(slice_controls.len(), n + band);

    // Uniform order: the CNOTs must be interleaved with the wider gates,
    // not bunched into a contiguous run (deterministic under the fixed
    // seed; a uniform shuffle makes a contiguous run astronomically
    // unlikely).
    let kinds: Vec<usize> = preblock.gates.iter().map(|g| g.ctrls.len()).collect();
    let first_cnot = kinds.iter().position(|&k| k == 1).unwrap();
    let last_cnot = kinds.iter().rposition(|&k| k == 1).unwrap();
    assert!(
        kinds[first_cnot..=last_cnot].iter().any(|&k| k > 1),
        "CNOTs and wider gates should be interleaved"
    );
}

#[test]
fn slice_zero_ccnot_gadgetize_matches_only_on_the_zero_slice() {
    let n = 3;
    let mask = (1u64 << n) - 1;
    let main = CircuitSeq {
        gates: vec![[0, 1, 2], [2, 0, 1], [1, 2, 0]],
    };
    for seed in 0..8u64 {
        let mut rng = StdRng::seed_from_u64(0xcc20_0000 + seed);
        let transformed = gadgetize_with_slice_zero_ccnot(
            &main,
            n,
            2,
            6 * n,
            &MaskConfig::off(),
            &ProdConfig::off(),
            &mut rng,
        );
        assert_eq!(transformed.num_wires, 2 * n);
        for x in 0..=mask {
            let expected = main.evaluate(x as usize) as u64 & mask;
            assert_eq!(eval_u64(&transformed.gates, x) & mask, expected);
        }
        // The gadget contract holds for any second-half value and the
        // original is a permutation, so a wrong slice reproduces C at x
        // exactly when the preblock fixes (x, a) — and each nonzero
        // slice must disturb at least one x.
        for a in 1..=mask {
            let disturbed = (0..=mask).any(|x| {
                let input = x | (a << n);
                let expected = main.evaluate(x as usize) as u64 & mask;
                eval_u64(&transformed.gates, input) & mask != expected
            });
            assert!(disturbed, "seed={seed:#x} slice a={a:#x} still computes C");
        }
    }
}

/// A scaled-down production-shaped config for the swap-refresh tests:
/// single carrier, [2,2,2,3] plan, band = n, churn on, swap on. Gray is
/// left at the production default (1) deliberately — swap mode must
/// decline it and still verify.
fn swap_test_config() -> ProdConfig {
    let mut p = ProdConfig::production_single();
    p.cg_jitter = 0;
    // The production rate, so every exactness test in this group runs with
    // a LIVE DRAIN SET: band variables are rewritten mid-body while masks
    // are being drawn and retired around them, which is precisely where a
    // bookkeeping slip would corrupt the endpoint.
    p.swap_refresh = 3;
    p.close_slice = 1;
    // At toy n the auto band (= n) leaves a value's disjointness draw a
    // single free pair, and the per-gate refresh churn exhausts its four
    // polarity variants; production bands are orders of magnitude wider.
    p.band = 24;
    p
}

fn swap_test_source(n: u16, rng: &mut StdRng) -> Vec<XGate> {
    // A mix of every source shape the sandwich feeds the gadgetizer:
    // CNOT, NCNOT, g57, and 2-3-control conjunctions.
    fn distinct(n: u16, taken: &[u16], rng: &mut StdRng) -> u16 {
        loop {
            let w = rng.random_range(0..n);
            if !taken.contains(&w) {
                return w;
            }
        }
    }
    let mut gates = Vec::new();
    for _ in 0..60 {
        let t = rng.random_range(0..n);
        let a = distinct(n, &[t], rng);
        let b = distinct(n, &[t, a], rng);
        let gate = match rng.random_range(0..5) {
            0 => XGate::cnot(t, a),
            1 => XGate::conj(t, [(a, false)]).unwrap(),
            2 => XGate::from_g57([t, a, b]),
            3 => XGate::conj(t, [(a, true), (b, true)]).unwrap(),
            _ => {
                let c = distinct(n, &[t, a, b], rng);
                XGate::conj(t, [(a, true), (b, false), (c, true)]).unwrap()
            }
        };
        gates.push(gate);
    }
    gates
}

/// The drain set turns the band over by SCHEDULING rather than by luck,
/// and the turnover count scales with the retirement rate.
///
/// Exactness under a live drain set is covered by the zero-slice tests
/// below (`swap_test_config` runs at the production rate), so this one
/// checks the mechanism: variables actually reach zero references and get
/// rewritten, more retirement sides buy more turnovers, and the reference
/// bookkeeping the whole thing rests on agrees with the live slots at the
/// end. `rewrite_var` asserts unconditionally that nothing names a
/// variable it overwrites, so a passing build is also evidence for the one
/// invariant that would silently corrupt the decode.
#[test]
fn drain_set_turns_the_band_over_and_scales_with_the_rate() {
    let n = 16usize;
    let mut turnovers: Vec<u64> = Vec::new();
    for sides in [2usize, 4] {
        let mut prod = swap_test_config();
        prod.swap_refresh = sides;
        prod.band = 48;
        let band_len = prod.band_size(n);
        let mut rng = StdRng::seed_from_u64(0xd7a1_0000 + sides as u64);
        let source = swap_test_source(n as u16, &mut rng);
        let state = GadgetState {
            n,
            pairs: (0..n).map(|w| (w, w)).collect(),
        };
        let mut ledger = ProdLedger::new(n, &prod, n, None);
        let mut out: Vec<XGate> = Vec::new();
        ledger.inject_all(&state, &mut rng, &mut out);
        assert!(
            ledger.drain_cap > 0,
            "sides={sides} drain set is not running"
        );
        let plan_before: Vec<Vec<usize>> = (0..n)
            .map(|v| {
                let mut d: Vec<usize> = ledger.slots[v].iter().map(|s| s.factors.len()).collect();
                d.sort_unstable();
                d
            })
            .collect();
        for gate in &source {
            ledger.fold_cg(gate, &state, &mut rng, &mut out);
        }
        // Steering retires whatever degree it lands on, so the replacement
        // MUST be drawn at the retired slot's degree: the per-value degree
        // multiset is the mask plan, and the piling-up bound a build
        // commits to is read straight off it. Drift here would move the
        // security claim without moving anything that reports it.
        for value in 0..n {
            let mut after: Vec<usize> = ledger.slots[value]
                .iter()
                .map(|s| s.factors.len())
                .collect();
            after.sort_unstable();
            assert_eq!(
                after, plan_before[value],
                "sides={sides} value {value} mask plan drifted"
            );
        }
        assert!(
            ledger.drained > 0,
            "sides={sides} no band variable ever came free: steering is stalled"
        );
        // The counts the rewrite guard reads must match the live slots. A
        // stale ZERO here is the failure that matters -- it would let a
        // referenced variable be overwritten -- so recount from scratch
        // rather than trusting the incremental path that produced them.
        let mut recount = vec![0u32; band_len];
        for value in 0..n {
            for slot in &ledger.slots[value] {
                for &(b, _) in &slot.factors {
                    recount[b as usize] += 1;
                }
            }
        }
        assert_eq!(
            ledger.var_refs, recount,
            "sides={sides} var_refs drifted from the live slots"
        );
        ledger.strip_all(&state, &mut rng, &mut out);
        assert!(
            ledger.var_refs.iter().all(|&r| r == 0),
            "sides={sides} strip_all left live references behind"
        );
        turnovers.push(ledger.drained);
    }
    assert!(
        turnovers[1] > turnovers[0],
        "retirement rate is the turnover lever, but 4 sides bought {} against 2 sides' {}",
        turnovers[1],
        turnovers[0]
    );
}

/// The 2026-08-20 swap-refresh redesign preserves the function exactly:
/// on the zero band slice the single-carrier gadget still computes the
/// source, while every fold retires and refreshes one mask term on the
/// target and on one control.
#[test]
fn swap_refresh_single_carrier_matches_source_on_the_zero_slice() {
    let n = 8usize;
    for seed in 0..4u64 {
        let mut rng = StdRng::seed_from_u64(0x5a70_0000 + seed);
        let source = swap_test_source(n as u16, &mut rng);
        // close_slice off: the builder strips every value and the whole
        // data range must match the source exactly.
        let mut prod = swap_test_config();
        prod.close_slice = 0;
        let gadget = gadgetize_xgates_single(&source, n, 1, &prod, &mut rng);
        assert!(gadget.num_wires <= 64, "test sized for u64 evaluation");
        let mask = (1u64 << n) - 1;
        for x in 0..=mask {
            let expected = eval_u64(&source, x) & mask;
            assert_eq!(
                eval_u64(&gadget.gates, x) & mask,
                expected,
                "seed={seed} x={x:#x}"
            );
        }
    }
}

/// With close_slice on, the BUILDER (no wrapper guards) still matches
/// the source on ALL data wires: every value's registry is discharged —
/// an undischarged registry leaves the emission telescope open under
/// reverse evaluation and corrupts the reverse payload (see the comment
/// at the strip_all call site). The junk-half divergence of the
/// delivered composite comes only from the wrapper's closing guard.
#[test]
fn swap_refresh_builder_matches_source_fully_even_with_close_slice() {
    let n = 8usize;
    let mut rng = StdRng::seed_from_u64(0x5a70_0100);
    let source = swap_test_source(n as u16, &mut rng);
    let prod = swap_test_config();
    let gadget = gadgetize_xgates_single(&source, n, 1, &prod, &mut rng);
    let mask = (1u64 << n) - 1;
    for x in 0..=mask {
        let expected = eval_u64(&source, x) & mask;
        let got = eval_u64(&gadget.gates, x) & mask;
        assert_eq!(got, expected, "x={x:#x}");
    }
}

/// Symmetric ports: with both guards junk-half-only, the REVERSED gadget
/// on the reverse-honest slice (a on the low half, zero upper half, zero
/// band) reproduces the REVERSED source's upper half — the gadget-level
/// mirror of the sandwich's A^-1(a,0) = (junk, D^-1(a)) contract. Every
/// XGate is an involution, so the reversed gate list IS the inverse.
#[test]
fn symmetric_guards_make_reverse_evaluation_honest() {
    let n = 8usize;
    for seed in 0..3u64 {
        let mut rng = StdRng::seed_from_u64(0x4e7e_0000 + seed);
        let source = swap_test_source(n as u16, &mut rng);
        let prod = swap_test_config();
        let slice_gates = 4 * prod.band_size(n);
        let circuit = gadgetize_xgates_with_slice_zero_ccnot_single(
            &source,
            n,
            1,
            slice_gates,
            &prod,
            &mut rng,
        );
        let rev_gadget: Vec<XGate> = circuit.gates.iter().rev().cloned().collect();
        let rev_source: Vec<XGate> = source.iter().rev().cloned().collect();
        let mask = (1u64 << n) - 1;
        let low = (1u64 << (n / 2)) - 1;
        let upper = mask & !low;
        for a in 0..=low {
            let expected = eval_u64(&rev_source, a) & upper;
            let got = eval_u64(&rev_gadget, a) & upper;
            assert_eq!(got, expected, "seed={seed} a={a:#x}");
        }
    }
}

/// The slice-zero wrapper with the closing block preserves the source on
/// the UPPER half of the data wires (the sandwich payload) and fires the
/// closing guard into the junk half on the honest forward run.
#[test]
fn closing_slice_wrapper_preserves_the_upper_half() {
    let n = 8usize;
    let mut rng = StdRng::seed_from_u64(0xc105_c105);
    let source = swap_test_source(n as u16, &mut rng);
    let prod = swap_test_config();
    // Several gates per slice wire: at one gate per wire the closing
    // block's halved target range pigeonholes same-target pure-CNOT
    // pairs, which cancel exactly on weight-2 slices and starve the
    // acceptance draw (production runs ~10 gates per slice wire).
    let slice_gates = 4 * prod.band_size(n);
    let circuit =
        gadgetize_xgates_with_slice_zero_ccnot_single(&source, n, 1, slice_gates, &prod, &mut rng);
    assert!(circuit.num_wires <= 64, "test sized for u64 evaluation");
    let mask = (1u64 << n) - 1;
    let mut low_half_diverged = false;
    for x in 0..=mask {
        let expected = eval_u64(&source, x) & mask;
        let got = eval_u64(&circuit.gates, x) & mask;
        let upper = !((1u64 << (n / 2)) - 1) & mask;
        assert_eq!(got & upper, expected & upper, "payload half x={x:#x}");
        if got & !upper & mask != expected & !upper & mask {
            low_half_diverged = true;
        }
    }
    assert!(
        low_half_diverged,
        "closing guard never fired: the low half matches the source everywhere, \
         so the appended block is not doing its job"
    );
}

#[test]
fn slice_block_stops_the_inverse_from_revealing_c_inverse() {
    let n = 3;
    let mask = (1u64 << n) - 1;
    let main = CircuitSeq {
        gates: vec![[0, 1, 2], [2, 0, 1], [1, 2, 0]],
    };
    for seed in 0..8u64 {
        let mut rng = StdRng::seed_from_u64(0xcc30_0000 + seed);
        let transformed = gadgetize_with_slice_zero_ccnot(
            &main,
            n,
            2,
            18,
            &MaskConfig::off(),
            &ProdConfig::off(),
            &mut rng,
        );
        // Every XGate is an involution, so the reversed gate list is the
        // inverse circuit; the slice block runs LAST there and fires on
        // the gadget's mask residue, junking the low half. Without it a
        // bare gadget's inverse returns C^-1 on the low wires for ANY
        // junk input.
        let reversed: Vec<XGate> = transformed.gates.iter().rev().cloned().collect();
        let mut leaks = 0usize;
        for p in 0..=mask {
            let c_inv = (0..=mask)
                .find(|&x| main.evaluate(x as usize) as u64 & mask == p)
                .unwrap();
            if eval_u64(&reversed, p) & mask == c_inv {
                leaks += 1;
            }
        }
        assert!(
            leaks < (mask as usize + 1),
            "inverse hands out C^-1 verbatim (seed={seed:#x})"
        );
    }
}

#[test]
fn gadgetize_xgates_preserves_the_low_wires_for_heterogeneous_sources() {
    let n = 3;
    let mask = (1u64 << n) - 1;
    // A heterogeneous mpmct1 source: g57, CNOT, CCNOT, and a negated
    // fragment — everything emit_shared_xgate2 must handle.
    let source = vec![
        XGate::from_g57([0, 1, 2]),
        XGate::cnot(0, 1),
        XGate::conj(2, [(0u16, true), (1u16, true)]).unwrap(),
        XGate::conj(1, [(2u16, false)]).unwrap(),
    ];
    for seed in 0..8u64 {
        let mut rng = StdRng::seed_from_u64(0xa11d_0000 + seed);
        let g = gadgetize_xgates(
            &source,
            n,
            2,
            &MaskConfig::off(),
            &ProdConfig::off(),
            &mut rng,
        );
        assert_eq!(g.num_wires, 2 * n);
        // Low n output = source(low n input) for ANY aux value.
        for input in 0..(1u64 << (2 * n)) {
            let expected = eval_u64(&source, input & mask) & mask;
            assert_eq!(
                eval_u64(&g.gates, input) & mask,
                expected,
                "input={input:#x}"
            );
        }
    }
}

#[test]
fn commuting_shuffle_preserves_function_and_relocates_gates() {
    let n = 6usize;
    for seed in 0..8u64 {
        let mut rng = StdRng::seed_from_u64(0x5f1e_0000 + seed);
        let mut gates: Vec<XGate> = Vec::new();
        for _ in 0..200 {
            let a = rng.random_range(0..n as u16);
            let b = loop {
                let w = rng.random_range(0..n as u16);
                if w != a {
                    break w;
                }
            };
            let c = loop {
                let w = rng.random_range(0..n as u16);
                if w != a && w != b {
                    break w;
                }
            };
            gates.push(match rng.random_range(0..3u32) {
                0 => XGate::cnot(a, b),
                1 => XGate::conj(a, [(b, true), (c, rng.random_bool(0.5))]).unwrap(),
                _ => XGate::from_g57([a, b, c]),
            });
        }
        let before = gates.clone();
        commuting_shuffle(&mut gates, &mut rng);
        // Same multiset of gates, same function on every input, new order.
        let mut counts = std::collections::HashMap::new();
        for g in &before {
            *counts.entry(g.clone()).or_insert(0i64) += 1;
        }
        for g in &gates {
            *counts.entry(g.clone()).or_insert(0i64) -= 1;
        }
        assert!(counts.values().all(|&c| c == 0), "gate multiset changed");
        for input in 0..(1u64 << n) {
            assert_eq!(
                eval_u64(&gates, input),
                eval_u64(&before, input),
                "seed={seed:#x} input={input:#x}"
            );
        }
        assert_ne!(gates, before, "seed={seed:#x}: order untouched");
    }
}

#[test]
fn cg_menu_variants_are_correct_and_stay_in_vocabulary() {
    // Values a,b,c live on carrier pairs (0,1), (2,3), (4,5). Every
    // variant, under every random role assignment, must (a) apply
    // exactly A ^= B OR !C to the target value, (b) leave every wire
    // outside a's carriers unchanged (collapses restored at wire
    // level), and (c) emit only g57s / 1-2-control conjunctions —
    // never a bare X, whose census would count the source gates.
    let state = GadgetState {
        n: 3,
        pairs: vec![(0, 1), (2, 3), (4, 5)],
    };
    let bit = |v: u64, w: usize| (v >> w) & 1;
    for variant in 0..CG_VARIANTS {
        for role_seed in 0..8u64 {
            let mut rng = StdRng::seed_from_u64(0xc6_0000 + role_seed);
            let mut gates = Vec::new();
            emit_cg_variant(&state, [0, 1, 2], variant, &mut rng, &mut gates);
            for g in &gates {
                if g.comp {
                    assert_eq!(g.ctrls.len(), 2, "complemented gate must be a g57");
                } else {
                    assert!(
                        (1..=2).contains(&g.ctrls.len()),
                        "conjunction must have 1 or 2 controls (no bare X)"
                    );
                }
            }
            for input in 0..64u64 {
                let output = eval_u64(&gates, input);
                assert_eq!(
                    output & !0b11,
                    input & !0b11,
                    "variant {variant}: non-target wires disturbed"
                );
                let b_val = bit(input, 2) ^ bit(input, 3);
                let c_val = bit(input, 4) ^ bit(input, 5);
                let f = b_val | (1 ^ c_val);
                let a_old = bit(input, 0) ^ bit(input, 1);
                let a_new = bit(output, 0) ^ bit(output, 1);
                assert_eq!(
                    a_new,
                    a_old ^ f,
                    "variant {variant} input {input:#08b}: wrong update"
                );
            }
        }
    }
}

#[test]
fn commuting_shuffle_reorders_across_an_opposite_polarity_crossing() {
    // A writes wire 0, B reads wire 0 — a read/write crossing — but they
    // share control wire 3 with opposite polarities, so their firing
    // supports are disjoint and they commute (two conjunction gates
    // sharing an opposite-polarity control). The shuffle must treat the
    // pair as mobile, not pin it by the crossing alone.
    let a = XGate::conj(0, [(2u16, true), (3u16, true)]).unwrap();
    let b = XGate::conj(1, [(0u16, true), (3u16, false)]).unwrap();
    assert!(!XGate::collides(&a, &b));
    let before = vec![a.clone(), b.clone()];
    let mut seen_swapped = false;
    for seed in 0..32u64 {
        let mut rng = StdRng::seed_from_u64(0xccc0_0000 + seed);
        let mut gates = before.clone();
        commuting_shuffle(&mut gates, &mut rng);
        for input in 0..(1u64 << 4) {
            assert_eq!(eval_u64(&gates, input), eval_u64(&before, input));
        }
        if gates[0] == b {
            seen_swapped = true;
        }
    }
    assert!(seen_swapped, "the separation-exempt pair never reordered");
}

#[test]
fn gadget_body_carries_nonlinear_rg_material() {
    // Complemented (comp=1) gates come only from the Z bookends and the
    // reinstated nonlinear g57 RG networks; the preblock, W_i, SG
    // fragments, and any linear RG are pure conjunctions. So a count
    // well above the two bookends certifies the RG policy is nonlinear.
    let n = 6;
    let main = CircuitSeq {
        gates: (0..40)
            .map(|k| [(k % n) as u16, ((k + 1) % n) as u16, ((k + 2) % n) as u16])
            .collect(),
    };
    let bookend_size = (2 * n * (n as f64).ln() as usize).max(64);
    let mut rng = StdRng::seed_from_u64(0xda7a_0001);
    let g = gadgetize_cnot(
        &main,
        n,
        1,
        &MaskConfig::off(),
        &ProdConfig::off(),
        &mut rng,
    );
    let comp_gates = g.gates.iter().filter(|g| g.comp).count();
    assert!(
        comp_gates > 2 * bookend_size,
        "expected nonlinear RG g57s beyond the {} bookend gates, found {} comp gates",
        2 * bookend_size,
        comp_gates
    );
}

#[test]
fn compose_a_realizes_the_reference_map() {
    let n = 3;
    let mask = 1usize << n;
    let c = CircuitSeq {
        gates: vec![[0, 1, 2], [2, 0, 1], [1, 2, 0]],
    };
    let d = CircuitSeq {
        gates: vec![[1, 0, 2], [0, 2, 1]],
    };
    let a = compose_a(&c, &d, n);
    assert!(a.gates.iter().flatten().all(|&w| (w as usize) < 2 * n));
    for x in 0..mask {
        for z in 0..mask {
            let input = x | (z << n);
            let out = a.evaluate(input);
            let cx = c.evaluate(x);
            let expected_lo = d.evaluate(cx); // D(C(x))
            let expected_hi = z ^ cx; // z ^ C(x)
            assert_eq!(out & (mask - 1), expected_lo, "low x={x} z={z}");
            assert_eq!((out >> n) & (mask - 1), expected_hi, "high x={x} z={z}");
        }
    }
}

#[test]
fn cnot_transformations_are_leaner_than_legacy_on_representative_circuits() {
    let n = 8;
    let main = CircuitSeq {
        gates: (0..64)
            .map(|index| {
                [
                    (index % n) as u16,
                    ((index + 1) % n) as u16,
                    ((index + 2) % n) as u16,
                ]
            })
            .collect(),
    };
    let mut legacy_gadget_total = 0usize;
    let mut cnot_gadget_total = 0usize;
    let mut legacy_feistal_total = 0usize;
    let mut cnot_feistal_total = 0usize;
    for seed in 0..16u64 {
        let mut legacy_rng = StdRng::seed_from_u64(0x1ea0_0000 + seed);
        let mut cnot_rng = StdRng::seed_from_u64(0x1ea0_0000 + seed);
        // Matched RG rate: one nonlinear RG per SG on both paths (the
        // cnot path now draws the same {RG1,RG2,RG3} g57 networks, so the
        // lean margin comes from the 4-fragment SG and 7-CNOT W_i alone).
        let legacy_gadget = gadgetize(&main, n, 1, &mut legacy_rng).gates.len();
        let cnot_gadget = gadgetize_cnot(
            &main,
            n,
            1,
            &MaskConfig::off(),
            &ProdConfig::off(),
            &mut cnot_rng,
        )
        .gates
        .len();
        assert!(cnot_gadget < legacy_gadget, "gadget seed={seed}");
        legacy_gadget_total += legacy_gadget;
        cnot_gadget_total += cnot_gadget;

        let mut legacy_rng = StdRng::seed_from_u64(0xfe15_0000 + seed);
        let mut cnot_rng = StdRng::seed_from_u64(0xfe15_0000 + seed);
        let legacy_feistal = feistalize(&main, n, 2, &mut legacy_rng).gates.len();
        let cnot_feistal = feistalize_cnot(&main, n, 2, &mut cnot_rng).gates.len();
        assert!(cnot_feistal < legacy_feistal, "Feistal seed={seed}");
        legacy_feistal_total += legacy_feistal;
        cnot_feistal_total += cnot_feistal;
    }
    println!(
        "representative averages: gadget {} -> {}; Feistal {} -> {}",
        legacy_gadget_total / 16,
        cnot_gadget_total / 16,
        legacy_feistal_total / 16,
        cnot_feistal_total / 16,
    );
}

// Keep a test-side copy: comparing the circuit only to the implementation
// constant would let both drift together without pinning the supplied map.
const EXPECTED_FIVE_CARRIER_U0: [u8; 32] = [
    19, 27, 8, 28, 0, 25, 12, 22, 24, 18, 1, 30, 5, 7, 3, 11, 29, 21, 23, 15, 26, 6, 4, 31, 16, 14,
    13, 10, 9, 20, 2, 17,
];

#[test]
fn five_carrier_u0_realization_has_the_supplied_truth_table() {
    assert_eq!(
        FIVE_CARRIER_U0, EXPECTED_FIVE_CARRIER_U0,
        "the frozen U0 table drifted from the supplied permutation"
    );
    let carriers = [0usize, 1, 2, 3, 4];
    let mut gates = Vec::new();
    emit_five_carrier_update(&carriers, &mut gates);
    assert_eq!(gates.len(), 40, "the frozen U0 realization drifted");

    let mut seen_u0 = [false; 32];
    let mut seen_u1 = [false; 32];
    let mut classes = [0usize; 2];
    for input in 0u8..32 {
        let output = eval_u64(&gates, input as u64) as u8;
        assert_eq!(
            output, EXPECTED_FIVE_CARRIER_U0[input as usize],
            "U0 truth-table mismatch at {input:#07b}"
        );
        assert!(!seen_u0[output as usize], "U0 is not injective");
        seen_u0[output as usize] = true;
        assert_ne!(output, input, "U0 has a fixed point at {input:#07b}");
        assert_eq!(
            five_carrier_decode_word(output),
            five_carrier_decode_word(input),
            "U0 changed the decode class at {input:#07b}"
        );

        // U1 is U0 followed by the designated c0 flip.  It is another
        // fixed-point-free permutation and changes exactly the D class.
        let output_u1 = output ^ 1;
        assert!(!seen_u1[output_u1 as usize], "U1 is not injective");
        seen_u1[output_u1 as usize] = true;
        assert_ne!(output_u1, input, "U1 has a fixed point at {input:#07b}");
        assert_ne!(
            five_carrier_decode_word(output_u1),
            five_carrier_decode_word(input),
            "U1 did not change the decode class at {input:#07b}"
        );
        classes[five_carrier_decode_word(input) as usize] += 1;
    }
    assert!(seen_u0.into_iter().all(|seen| seen));
    assert!(seen_u1.into_iter().all(|seen| seen));
    assert_eq!(classes, [16, 16], "D must split the carrier space evenly");
    assert!(
        gates.iter().any(|gate| gate.width() == 4),
        "an ancilla-free realization of the odd U0 permutation needs a width-4 gate"
    );
}

#[test]
fn five_carrier_update_trace_has_no_weight_one_or_two_walsh_detector() {
    // A trace row consists of the five carrier bits immediately before and
    // after one update.  For firing bit f, the post-state is
    // U_f(x) = U0(x) XOR f*e0.  Its Walsh coefficient against a detector
    // is sum_{x,f} (-1)^(f XOR parity(detector & (x,U_f(x)))).
    // Thus a zero sum means exactly zero correlation with f.
    let walsh_sum = |detector: u16| -> i32 {
        let mut sum = 0i32;
        for input in 0u16..32 {
            for firing in 0u16..2 {
                let output = EXPECTED_FIVE_CARRIER_U0[input as usize] as u16 ^ firing;
                let trace = input | (output << 5);
                let prediction = (trace & detector).count_ones() as u16 & 1;
                sum += if prediction == firing { 1 } else { -1 };
            }
        }
        sum
    };

    for detector in 1u16..(1u16 << 10) {
        let weight = detector.count_ones();
        if weight <= 2 {
            assert_eq!(
                walsh_sum(detector),
                0,
                "weight-{weight} trace detector {detector:#012b} is correlated"
            );
        }
    }

    let max_weight_three = (1u16..(1u16 << 10))
        .filter(|detector| detector.count_ones() == 3)
        .map(|detector| walsh_sum(detector).unsigned_abs())
        .max()
        .unwrap();
    assert_eq!(
        max_weight_three, 32,
        "the first nonzero trace spectrum should have Walsh magnitude 1/2"
    );
    // There are 64 equally weighted (x,f) rows.  A Walsh magnitude of 32
    // is |Pr[predict=f] - 1/2| = 32/(2*64) = 1/4.
}

#[test]
fn five_carrier_endpoint_firing_bit_is_exactly_degree_two() {
    // Pin the algebraic boundary separately from the raw-parity spectrum.
    // The supplied decode is quadratic, so every transition satisfies
    //
    //   firing = D(carrier_before) XOR D(carrier_after).
    //
    // Zero correlation with all weight-one/two XOR detectors does not
    // imply immunity to Gaussian elimination on the second tensor: the
    // right-hand side below is an XOR of nine degree-one/two features.
    fn gf2_rank(signatures: impl IntoIterator<Item = u64>) -> usize {
        let mut basis = [0u64; 64];
        let mut rank = 0usize;
        for mut signature in signatures {
            while signature != 0 {
                let pivot = 63 - signature.leading_zeros() as usize;
                if basis[pivot] != 0 {
                    signature ^= basis[pivot];
                } else {
                    basis[pivot] = signature;
                    rank += 1;
                    break;
                }
            }
        }
        rank
    }

    let mut columns = [0u64; 10];
    let mut firing_signature = 0u64;
    let mut decode_delta_signature = 0u64;
    for input in 0usize..32 {
        for firing in 0usize..2 {
            let row = 2 * input + firing;
            let output = EXPECTED_FIVE_CARRIER_U0[input] as usize ^ firing;
            let trace = input | (output << 5);
            for (wire, column) in columns.iter_mut().enumerate() {
                if trace & (1usize << wire) != 0 {
                    *column |= 1u64 << row;
                }
            }
            if firing != 0 {
                firing_signature |= 1u64 << row;
            }
            if five_carrier_decode_word(input as u8) ^ five_carrier_decode_word(output as u8) {
                decode_delta_signature |= 1u64 << row;
            }
        }
    }
    assert_eq!(
        decode_delta_signature, firing_signature,
        "the supplied quadratic decode must recover every firing bit exactly"
    );

    let degree_one: Vec<u64> = std::iter::once(u64::MAX).chain(columns).collect();
    assert_eq!(gf2_rank(degree_one.iter().copied()), 11);
    assert_eq!(
        gf2_rank(
            degree_one
                .iter()
                .copied()
                .chain(std::iter::once(firing_signature))
        ),
        12,
        "degree-one endpoint features unexpectedly recovered the firing bit"
    );

    let mut degree_two = degree_one;
    for left in 0..10 {
        for right in left + 1..10 {
            degree_two.push(columns[left] & columns[right]);
        }
    }
    assert_eq!(gf2_rank(degree_two.iter().copied()), 42);
    assert_eq!(
        gf2_rank(
            degree_two
                .iter()
                .copied()
                .chain(std::iter::once(firing_signature))
        ),
        42,
        "the exact recovery boundary must be degree two"
    );
}

#[test]
fn five_carrier_endpoint_has_no_perfect_xor_detector_at_any_weight() {
    let walsh_sum = |detector: u16| -> i32 {
        let mut sum = 0i32;
        for input in 0u16..32 {
            for firing in 0u16..2 {
                let output = EXPECTED_FIVE_CARRIER_U0[input as usize] as u16 ^ firing;
                let trace = input | (output << 5);
                let prediction = (trace & detector).count_ones() as u16 & 1;
                sum += if prediction == firing { 1 } else { -1 };
            }
        }
        sum
    };

    let maximum = (1u16..(1u16 << 10))
        .map(|detector| walsh_sum(detector).unsigned_abs())
        .max();
    assert_eq!(
        maximum,
        Some(32),
        "the supplied map should have no perfect affine endpoint relation"
    );
}

#[test]
fn strong_five_carrier_u0_decode_and_degree_boundary() {
    let carriers = [0usize, 1, 2, 3, 4];
    let mut gates = Vec::new();
    emit_strong_five_carrier_update(&carriers, &mut gates);
    assert_eq!(gates.len(), 6, "strong-five U0 gate count drifted");

    let mut seen_u0 = [false; 32];
    let mut seen_u1 = [false; 32];
    for input in 0u8..32 {
        let output = eval_u64(&gates, input as u64) as u8;
        assert_eq!(output, STRONG_FIVE_CARRIER_U0[input as usize]);
        assert!(!seen_u0[output as usize]);
        seen_u0[output as usize] = true;
        assert_ne!(output, input, "strong-five U0 has a fixed point");
        assert_eq!(
            strong_five_carrier_decode_word(output),
            strong_five_carrier_decode_word(input),
            "strong-five U0 changed decode class"
        );

        let output_u1 = output ^ 1;
        assert!(!seen_u1[output_u1 as usize]);
        seen_u1[output_u1 as usize] = true;
        assert_ne!(output_u1, input, "strong-five U1 has a fixed point");
        assert_ne!(
            strong_five_carrier_decode_word(output_u1),
            strong_five_carrier_decode_word(input),
            "strong-five U1 did not flip decode class"
        );
    }
    assert!(seen_u0.into_iter().all(|seen| seen));
    assert!(seen_u1.into_iter().all(|seen| seen));

    fn gf2_rank(signatures: impl IntoIterator<Item = u64>) -> usize {
        let mut basis = [0u64; 64];
        let mut rank = 0usize;
        for mut signature in signatures {
            while signature != 0 {
                let pivot = 63 - signature.leading_zeros() as usize;
                if basis[pivot] != 0 {
                    signature ^= basis[pivot];
                } else {
                    basis[pivot] = signature;
                    rank += 1;
                    break;
                }
            }
        }
        rank
    }

    let mut columns = [0u64; 10];
    let mut firing_signature = 0u64;
    let mut decode_delta = 0u64;
    for input in 0usize..32 {
        for firing in 0usize..2 {
            let row = 2 * input + firing;
            let output = STRONG_FIVE_CARRIER_U0[input] as usize ^ firing;
            let trace = input | (output << 5);
            for (wire, column) in columns.iter_mut().enumerate() {
                if trace & (1usize << wire) != 0 {
                    *column |= 1u64 << row;
                }
            }
            if firing != 0 {
                firing_signature |= 1u64 << row;
            }
            if strong_five_carrier_decode_word(input as u8)
                ^ strong_five_carrier_decode_word(output as u8)
            {
                decode_delta |= 1u64 << row;
            }
        }
    }
    assert_eq!(decode_delta, firing_signature);

    let degree_one: Vec<u64> = std::iter::once(u64::MAX).chain(columns).collect();
    let mut degree_two = degree_one.clone();
    for a in 0..10 {
        for b in a + 1..10 {
            degree_two.push(columns[a] & columns[b]);
        }
    }
    let rank_two = gf2_rank(degree_two.iter().copied());
    assert_eq!(rank_two, 22);
    assert_eq!(
        gf2_rank(
            degree_two
                .iter()
                .copied()
                .chain(std::iter::once(firing_signature))
        ),
        rank_two + 1,
        "degree-two endpoint features recovered strong-five firing"
    );

    let mut degree_three = degree_two;
    for a in 0..10 {
        for b in a + 1..10 {
            for c in b + 1..10 {
                degree_three.push(columns[a] & columns[b] & columns[c]);
            }
        }
    }
    assert_eq!(gf2_rank(degree_three.iter().copied()), 42);
    assert_eq!(
        gf2_rank(
            degree_three
                .iter()
                .copied()
                .chain(std::iter::once(firing_signature))
        ),
        42,
        "strong-five exact boundary must first appear at degree three"
    );
}

#[test]
fn strong_five_carrier_endpoint_walsh_and_affine_structure() {
    let walsh_sum = |detector: u16| -> i32 {
        let mut sum = 0i32;
        for input in 0u16..32 {
            for firing in 0u16..2 {
                let output = STRONG_FIVE_CARRIER_U0[input as usize] as u16 ^ firing;
                let trace = input | (output << 5);
                let prediction = (trace & detector).count_ones() as u16 & 1;
                sum += if prediction == firing { 1 } else { -1 };
            }
        }
        sum
    };

    for detector in 1u16..(1u16 << 10) {
        if detector.count_ones() <= 2 {
            assert_eq!(walsh_sum(detector), 0);
        }
        assert_ne!(walsh_sum(detector).unsigned_abs(), 64);
    }
    let weight_three: Vec<u32> = (1u16..(1u16 << 10))
        .filter(|detector| detector.count_ones() == 3)
        .map(|detector| walsh_sum(detector).unsigned_abs())
        .collect();
    assert_eq!(weight_three.iter().copied().max(), Some(16));
    assert_eq!(
        weight_three
            .iter()
            .filter(|&&magnitude| magnitude != 0)
            .count(),
        2
    );

    // The compact affine tail deliberately leaves c1 and c2 fixed.  Pin
    // that structural tradeoff so it cannot be omitted from the mode's
    // documentation or accidentally mistaken for full affine rank.
    for input in 0u8..32 {
        let output = STRONG_FIVE_CARRIER_U0[input as usize];
        assert_eq!((input >> 1) & 1, (output >> 1) & 1);
        assert_eq!((input >> 2) & 1, (output >> 2) & 1);
    }
}

#[test]
fn strong_five_carrier_all_fold_modes_preserve_dirty_high_inputs() {
    let n = 6usize;
    let band = 6usize;
    let source = vec![
        XGate::from_g57([0, 1, 2]),
        XGate::cnot(3, 4),
        XGate::conj(5, [(0u16, true), (2u16, false)]).unwrap(),
    ];
    let low_mask = (1u64 << n) - 1;
    let total = 5 * n + band;
    let high_mask = ((1u64 << total) - 1) ^ low_mask;
    let junk_patterns = [
        0,
        high_mask,
        0xaaaa_aaaa_aaaa_aaaau64 & high_mask,
        0x36db_6db6_db6d_b6dbu64 & high_mask,
    ];

    for gray_fold in 0..=3usize {
        let mut prod = ProdConfig::production_five_carrier();
        prod.band = band;
        prod.gray_fold = gray_fold;
        prod.cg_jitter = 0;
        let mut rng = StdRng::seed_from_u64(0x5c0b_1c00 + gray_fold as u64);
        let gadget = gadgetize_xgates_strong_five_carrier(&source, n, 2, &prod, &mut rng);
        assert_eq!(gadget.num_wires, total);
        for low in 0..=low_mask {
            let expected = eval_u64(&source, low) & low_mask;
            for &junk in &junk_patterns {
                assert_eq!(
                    eval_u64(&gadget.gates, low | junk) & low_mask,
                    expected,
                    "strong-five mode={gray_fold} low={low:#x} junk={junk:#x}"
                );
            }
        }
    }
}

#[test]
fn five_carrier_gadget_preserves_low_wires_for_dirty_high_inputs() {
    let n = 6usize;
    let band = 6usize;
    let mut prod = ProdConfig::production_five_carrier();
    prod.band = band;

    // Certify that this fixture really reaches the optimized four-phase
    // g57 fold, rather than silently exercising only the general odometer.
    let state = FiveCarrierState::home(n);
    let mut audit_rng = StdRng::seed_from_u64(0x5ca1_67a7);
    let mut ledger = ProdLedger::new(n, &prod, 5 * n, None);
    let mut injection = Vec::new();
    ledger.inject_all(&state.c0_view(), &mut audit_rng, &mut injection);
    let mut optimized_fold = Vec::new();
    ledger.fold_five(
        &XGate::from_g57([0, 1, 2]),
        &state,
        &mut audit_rng,
        &mut optimized_fold,
    );
    assert_eq!(ledger.cg_gray, 1, "fixture missed the optimized g57 path");

    // Keep the optimized g57 and add native heterogeneous forms handled by
    // fold_five's general path: CNOT plus a mixed-polarity CCNOT.
    let source = vec![
        XGate::from_g57([0, 1, 2]),
        XGate::cnot(3, 4),
        XGate::conj(5, [(0u16, true), (2u16, false)]).unwrap(),
    ];
    let low_mask = (1u64 << n) - 1;
    let total = 5 * n + band;
    let high_mask = ((1u64 << total) - 1) ^ low_mask;
    let fixed_junk = [
        0,
        high_mask,
        0xaaaa_aaaa_aaaa_aaaau64 & high_mask,
        0x5555_5555_5555_5555u64 & high_mask,
        0x9249_2492_4924_9249u64 & high_mask,
        0x36db_6db6_db6d_b6dbu64 & high_mask,
    ];

    for seed in 0..3u64 {
        let mut rng = StdRng::seed_from_u64(0x5ca1_0000 + seed);
        let gadget = gadgetize_xgates_five_carrier(&source, n, 2, &prod, &mut rng);
        assert_eq!(gadget.num_wires, total, "expected 5*n + band wires");

        for low in 0..=low_mask {
            let expected = eval_u64(&source, low) & low_mask;
            for (junk_index, &junk) in fixed_junk.iter().enumerate() {
                let input = low | junk;
                assert_eq!(
                    eval_u64(&gadget.gates, input) & low_mask,
                    expected,
                    "seed={seed} low={low:#x} junk pattern {junk_index}"
                );
            }
            // One extra non-periodic high assignment per low word catches
            // accidental assumptions hidden by the structured patterns.
            let junk = (low
                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .rotate_left((low as u32) & 31))
                & high_mask;
            assert_eq!(
                eval_u64(&gadget.gates, low | junk) & low_mask,
                expected,
                "seed={seed} low={low:#x} hashed high junk"
            );
        }
    }
}

const EXPECTED_SIX_CARRIER_U0: [u8; 64] = [
    2, 3, 0, 1, 7, 6, 5, 4, 11, 10, 9, 8, 14, 15, 12, 13, 30, 31, 29, 28, 27, 26, 24, 25, 22, 23,
    21, 20, 19, 18, 16, 17, 42, 43, 41, 40, 46, 47, 45, 44, 35, 34, 32, 33, 39, 38, 36, 37, 54, 55,
    53, 52, 51, 50, 48, 49, 63, 62, 60, 61, 58, 59, 57, 56,
];

#[test]
fn six_carrier_u0_realization_has_the_supplied_truth_table() {
    assert_eq!(SIX_CARRIER_U0, EXPECTED_SIX_CARRIER_U0);
    let carriers = [0usize, 1, 2, 3, 4, 5];
    let mut gates = Vec::new();
    emit_six_carrier_update(&carriers, &mut gates);
    assert_eq!(gates.len(), 10, "the compact U0 realization drifted");

    let mut seen_u0 = [false; 64];
    let mut seen_u1 = [false; 64];
    let mut classes = [0usize; 2];
    for input in 0u8..64 {
        let output = eval_u64(&gates, input as u64) as u8;
        assert_eq!(
            output, EXPECTED_SIX_CARRIER_U0[input as usize],
            "U0 truth-table mismatch at {input:#08b}"
        );
        assert!(!seen_u0[output as usize], "U0 is not injective");
        seen_u0[output as usize] = true;
        assert_ne!(output, input, "U0 fixed point at {input:#08b}");
        assert_eq!(
            EXPECTED_SIX_CARRIER_U0[output as usize], input,
            "the supplied U0 must be an involution"
        );
        assert_eq!(
            six_carrier_decode_word(output),
            six_carrier_decode_word(input),
            "U0 changed the decode class at {input:#08b}"
        );

        let output_u1 = output ^ 1;
        assert!(!seen_u1[output_u1 as usize], "U1 is not injective");
        seen_u1[output_u1 as usize] = true;
        assert_ne!(output_u1, input, "U1 fixed point at {input:#08b}");
        assert_ne!(
            six_carrier_decode_word(output_u1),
            six_carrier_decode_word(input),
            "U1 did not change the decode class at {input:#08b}"
        );
        classes[six_carrier_decode_word(input) as usize] += 1;
    }
    assert!(seen_u0.into_iter().all(|seen| seen));
    assert!(seen_u1.into_iter().all(|seen| seen));
    assert_eq!(classes, [32, 32], "D must split the carrier space evenly");
    let widths: Vec<usize> = gates.iter().map(XGate::width).collect();
    assert_eq!(widths.iter().filter(|&&width| width == 0).count(), 1);
    assert_eq!(widths.iter().filter(|&&width| width == 1).count(), 5);
    assert_eq!(widths.iter().filter(|&&width| width == 2).count(), 1);
    assert_eq!(widths.iter().filter(|&&width| width == 3).count(), 3);
}

#[test]
fn six_carrier_decode_has_no_weight_one_or_two_static_parity_correlation() {
    let correlation_sum = |detector: u8| -> i32 {
        (0u8..64)
            .map(|carrier| {
                let detector_value = (carrier & detector).count_ones() & 1 != 0;
                if detector_value == six_carrier_decode_word(carrier) {
                    1
                } else {
                    -1
                }
            })
            .sum()
    };
    for detector in 1u8..64 {
        if detector.count_ones() <= 2 {
            assert_eq!(
                correlation_sum(detector),
                0,
                "low-weight static detector {detector:#08b} is correlated"
            );
        }
    }
    let max_weight_three = (1u8..64)
        .filter(|detector| detector.count_ones() == 3)
        .map(|detector| correlation_sum(detector).unsigned_abs())
        .max()
        .unwrap();
    assert_eq!(max_weight_three, 16, "expected normalized magnitude 1/4");
}

#[test]
fn six_carrier_update_trace_has_no_weight_one_two_or_three_walsh_detector() {
    let walsh_sum = |detector: u16| -> i32 {
        let mut sum = 0i32;
        for input in 0u16..64 {
            for firing in 0u16..2 {
                let output = EXPECTED_SIX_CARRIER_U0[input as usize] as u16 ^ firing;
                let trace = input | (output << 6);
                let prediction = (trace & detector).count_ones() as u16 & 1;
                sum += if prediction == firing { 1 } else { -1 };
            }
        }
        sum
    };

    for detector in 1u16..(1u16 << 12) {
        let weight = detector.count_ones();
        if weight <= 3 {
            assert_eq!(
                walsh_sum(detector),
                0,
                "weight-{weight} endpoint detector {detector:#014b} is correlated"
            );
        }
    }
    let max_weight_four = (1u16..(1u16 << 12))
        .filter(|detector| detector.count_ones() == 4)
        .map(|detector| walsh_sum(detector).unsigned_abs())
        .max()
        .unwrap();
    assert_eq!(
        max_weight_four, 32,
        "first normalized Walsh magnitude is 1/4"
    );
}

#[test]
fn six_carrier_endpoint_firing_bit_is_outside_the_degree_two_trace_span() {
    fn gf2_rank(signatures: impl IntoIterator<Item = u128>) -> usize {
        let mut basis = [0u128; 128];
        let mut rank = 0usize;
        for mut signature in signatures {
            while signature != 0 {
                let pivot = 127 - signature.leading_zeros() as usize;
                if basis[pivot] != 0 {
                    signature ^= basis[pivot];
                } else {
                    basis[pivot] = signature;
                    rank += 1;
                    break;
                }
            }
        }
        rank
    }

    let mut columns = [0u128; 12];
    let mut firing_signature = 0u128;
    for input in 0usize..64 {
        for firing in 0usize..2 {
            let row = 2 * input + firing;
            let output = EXPECTED_SIX_CARRIER_U0[input] as usize ^ firing;
            let trace = input | (output << 6);
            for (wire, column) in columns.iter_mut().enumerate() {
                if (trace >> wire) & 1 != 0 {
                    *column |= 1u128 << row;
                }
            }
            if firing != 0 {
                firing_signature |= 1u128 << row;
            }
        }
    }

    let mut degree_two = vec![u128::MAX];
    degree_two.extend(columns);
    for left in 0..12 {
        for right in left + 1..12 {
            degree_two.push(columns[left] & columns[right]);
        }
    }
    let rank_two = gf2_rank(degree_two.iter().copied());
    let rank_with_firing = gf2_rank(
        degree_two
            .iter()
            .copied()
            .chain(std::iter::once(firing_signature)),
    );
    assert_eq!(rank_two, 29, "degree-two endpoint feature rank drifted");
    assert_eq!(
        rank_with_firing,
        rank_two + 1,
        "degree-two Gaussian elimination recovered the firing bit"
    );

    // Pin the intended boundary: degree three does contain an exact
    // recovery, while degree two does not.
    let mut degree_three = degree_two;
    for a in 0..12 {
        for b in a + 1..12 {
            for c in b + 1..12 {
                degree_three.push(columns[a] & columns[b] & columns[c]);
            }
        }
    }
    assert_eq!(
        gf2_rank(degree_three.iter().copied()),
        gf2_rank(
            degree_three
                .iter()
                .copied()
                .chain(std::iter::once(firing_signature))
        ),
        "degree-three boundary unexpectedly moved"
    );
}

#[test]
fn six_carrier_gadget_preserves_low_wires_for_dirty_high_inputs() {
    let n = 6usize;
    let band = 6usize;
    let mut prod = ProdConfig::production_six_carrier();
    prod.band = band;

    let state = SixCarrierState::home(n);
    let mut audit_rng = StdRng::seed_from_u64(0x6ca1_67a7);
    let mut ledger = ProdLedger::new(n, &prod, 6 * n, None);
    let mut injection = Vec::new();
    ledger.inject_all(&state.c0_view(), &mut audit_rng, &mut injection);
    let mut optimized_fold = Vec::new();
    ledger.fold_six(
        &XGate::from_g57([0, 1, 2]),
        &state,
        &mut audit_rng,
        &mut optimized_fold,
    );
    assert_eq!(
        ledger.cg_gray, 1,
        "fixture missed the six-carrier Gray fold"
    );

    let source = vec![
        XGate::from_g57([0, 1, 2]),
        XGate::cnot(3, 4),
        XGate::conj(5, [(0u16, true), (2u16, false)]).unwrap(),
    ];
    let low_mask = (1u64 << n) - 1;
    let total = 6 * n + band;
    let high_mask = ((1u64 << total) - 1) ^ low_mask;
    let fixed_junk = [
        0,
        high_mask,
        0xaaaa_aaaa_aaaa_aaaau64 & high_mask,
        0x5555_5555_5555_5555u64 & high_mask,
        0x9249_2492_4924_9249u64 & high_mask,
        0x36db_6db6_db6d_b6dbu64 & high_mask,
    ];

    for seed in 0..3u64 {
        let mut rng = StdRng::seed_from_u64(0x6ca1_0000 + seed);
        let gadget = gadgetize_xgates_six_carrier(&source, n, 2, &prod, &mut rng);
        assert_eq!(gadget.num_wires, total, "expected 6*n + band wires");

        for low in 0..=low_mask {
            let expected = eval_u64(&source, low) & low_mask;
            for (junk_index, &junk) in fixed_junk.iter().enumerate() {
                assert_eq!(
                    eval_u64(&gadget.gates, low | junk) & low_mask,
                    expected,
                    "seed={seed} low={low:#x} junk pattern {junk_index}"
                );
            }
            let junk = (low
                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .rotate_left((low as u32) & 31))
                & high_mask;
            assert_eq!(
                eval_u64(&gadget.gates, low | junk) & low_mask,
                expected,
                "seed={seed} low={low:#x} hashed high junk"
            );
        }
    }
}

#[test]
fn six_carrier_public_cnot_and_slice_wrappers_have_the_expected_port() {
    let n = 6usize;
    let band = 6usize;
    let mut prod = ProdConfig::production_six_carrier();
    prod.band = band;
    let total = 6 * n + band;
    let low_mask = (1u64 << n) - 1;
    let high_mask = ((1u64 << total) - 1) ^ low_mask;
    let main = CircuitSeq {
        gates: vec![[0, 1, 2]],
    };
    let source = vec![XGate::from_g57([0, 1, 2])];

    let mut bare_rng = StdRng::seed_from_u64(0x6ca2_0001);
    let bare = gadgetize_cnot_six_carrier(&main, n, 1, &prod, &mut bare_rng);
    let mut slice_cnot_rng = StdRng::seed_from_u64(0x6ca2_0002);
    let slice_cnot = gadgetize_with_slice_zero_ccnot_six_carrier(
        &main,
        n,
        1,
        10 * n,
        &MaskConfig::off(),
        &prod,
        &mut slice_cnot_rng,
    );
    let mut slice_xgate_rng = StdRng::seed_from_u64(0x6ca2_0003);
    let slice_xgate = gadgetize_xgates_with_slice_zero_ccnot_six_carrier(
        &source,
        n,
        1,
        10 * n,
        &prod,
        &mut slice_xgate_rng,
    );
    assert_eq!(bare.num_wires, total);
    assert_eq!(slice_cnot.num_wires, total);
    assert_eq!(slice_xgate.num_wires, total);

    for low in 0..=low_mask {
        let expected = eval_u64(&source, low) & low_mask;
        assert_eq!(eval_u64(&bare.gates, low) & low_mask, expected);
        assert_eq!(eval_u64(&bare.gates, low | high_mask) & low_mask, expected);
        assert_eq!(eval_u64(&slice_cnot.gates, low) & low_mask, expected);
        assert_eq!(eval_u64(&slice_xgate.gates, low) & low_mask, expected);
    }
}

#[test]
fn six_carrier_empty_source_is_the_identity_for_arbitrary_high_junk() {
    let n = 3usize;
    let mut prod = ProdConfig::production_six_carrier();
    prod.band = 6;
    let total = 6 * n + prod.band_size(n);
    let low_mask = (1u64 << n) - 1;
    let high_mask = ((1u64 << total) - 1) ^ low_mask;
    let mut rng = StdRng::seed_from_u64(0x6ca3_0001);
    let gadget = gadgetize_xgates_six_carrier(&[], n, 1, &prod, &mut rng);
    assert_eq!(gadget.num_wires, total);

    for low in 0..=low_mask {
        for junk in [
            0,
            high_mask,
            0xaaaa_aaaa_aaaa_aaaau64 & high_mask,
            0x9249_2492_4924_9249u64 & high_mask,
        ] {
            assert_eq!(eval_u64(&gadget.gates, low | junk) & low_mask, low);
        }
    }
}

#[test]
fn strong_six_carrier_u0_truth_table_class_and_width_census() {
    let carriers = [0usize, 1, 2, 3, 4, 5];
    let mut gates = Vec::new();
    emit_strong_six_carrier_update(&carriers, &mut gates);
    assert_eq!(gates.len(), 21, "strong-six U0 gate count drifted");

    let mut seen_u0 = [false; 64];
    let mut seen_u1 = [false; 64];
    let mut classes = [0usize; 2];
    for input in 0u8..64 {
        let output = eval_u64(&gates, input as u64) as u8;
        assert_eq!(
            output, STRONG_SIX_CARRIER_U0[input as usize],
            "strong-six U0 truth-table mismatch at {input:#08b}"
        );
        assert!(!seen_u0[output as usize], "strong-six U0 is not injective");
        seen_u0[output as usize] = true;
        assert_ne!(output, input, "strong-six U0 fixed point at {input:#08b}");
        assert_eq!(
            six_carrier_decode_word(output),
            six_carrier_decode_word(input),
            "strong-six U0 changed the decode class at {input:#08b}"
        );

        let output_u1 = output ^ 1;
        assert!(
            !seen_u1[output_u1 as usize],
            "strong-six U1 is not injective"
        );
        seen_u1[output_u1 as usize] = true;
        assert_ne!(
            output_u1, input,
            "strong-six U1 fixed point at {input:#08b}"
        );
        assert_ne!(
            six_carrier_decode_word(output_u1),
            six_carrier_decode_word(input),
            "strong-six U1 did not change the decode class at {input:#08b}"
        );
        classes[six_carrier_decode_word(input) as usize] += 1;
    }
    assert!(seen_u0.into_iter().all(|seen| seen));
    assert!(seen_u1.into_iter().all(|seen| seen));
    assert_eq!(classes, [32, 32]);

    let widths: Vec<usize> = gates.iter().map(XGate::width).collect();
    for (width, expected) in [(0, 1), (1, 6), (2, 7), (3, 6), (4, 1)] {
        assert_eq!(
            widths.iter().filter(|&&actual| actual == width).count(),
            expected,
            "strong-six width-{width} gate count drifted"
        );
    }
}

#[test]
fn strong_six_carrier_endpoint_has_full_affine_rank_and_pinned_walsh_spectrum() {
    fn gf2_rank(signatures: impl IntoIterator<Item = u64>) -> usize {
        let mut basis = [0u64; 64];
        let mut rank = 0usize;
        for mut signature in signatures {
            while signature != 0 {
                let pivot = 63 - signature.leading_zeros() as usize;
                if basis[pivot] != 0 {
                    signature ^= basis[pivot];
                } else {
                    basis[pivot] = signature;
                    rank += 1;
                    break;
                }
            }
        }
        rank
    }

    // Constant plus all six before and six U0-after coordinates have the
    // maximum possible rank.  Pin the movement census too: unlike the
    // compact sibling, no carrier lane is frozen or an affine duplicate.
    let mut graph_columns = [0u64; 12];
    let mut movement = [0usize; 6];
    for input in 0usize..64 {
        let output = STRONG_SIX_CARRIER_U0[input] as usize;
        let trace = input | (output << 6);
        for (wire, column) in graph_columns.iter_mut().enumerate() {
            if (trace >> wire) & 1 != 0 {
                *column |= 1u64 << input;
            }
        }
        for (lane, count) in movement.iter_mut().enumerate() {
            *count += ((input ^ output) >> lane) & 1;
        }
    }
    assert_eq!(
        gf2_rank(std::iter::once(u64::MAX).chain(graph_columns)),
        13,
        "strong-six endpoint graph lost full affine rank"
    );
    assert_eq!(movement, [32, 48, 32, 32, 16, 16]);
    assert!(movement.into_iter().all(|count| count != 0));

    // Histogram bins are |W| = 16,32,48,64 over all 128 (input,firing)
    // rows.  No coefficient exists through detector weight three; the
    // first signal is rho=32/128=1/4 at weight four.  No parity detector
    // at any weight is perfect.
    let expected: [[usize; 4]; 13] = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [2, 15, 0, 0],
        [12, 53, 2, 2],
        [33, 66, 9, 7],
        [65, 71, 5, 5],
        [60, 52, 10, 2],
        [33, 33, 9, 1],
        [13, 10, 1, 1],
        [2, 3, 0, 0],
        [0, 1, 0, 0],
    ];
    let mut actual = [[0usize; 4]; 13];
    for detector in 1u16..(1u16 << 12) {
        let mut sum = 0i32;
        for input in 0u16..64 {
            for firing in 0u16..2 {
                let output = STRONG_SIX_CARRIER_U0[input as usize] as u16 ^ firing;
                let trace = input | (output << 6);
                let prediction = (trace & detector).count_ones() as u16 & 1;
                sum += if prediction == firing { 1 } else { -1 };
            }
        }
        let magnitude = sum.unsigned_abs() as usize;
        if magnitude != 0 {
            assert!(magnitude <= 64, "strong-six gained a stronger parity leak");
            assert_eq!(magnitude % 16, 0);
            actual[detector.count_ones() as usize][magnitude / 16 - 1] += 1;
        }
    }
    assert_eq!(
        actual, expected,
        "strong-six endpoint Walsh spectrum drifted"
    );
}

#[test]
fn strong_six_carrier_exact_firing_boundary_is_degree_three() {
    fn gf2_rank(signatures: impl IntoIterator<Item = u128>) -> usize {
        let mut basis = [0u128; 128];
        let mut rank = 0usize;
        for mut signature in signatures {
            while signature != 0 {
                let pivot = 127 - signature.leading_zeros() as usize;
                if basis[pivot] != 0 {
                    signature ^= basis[pivot];
                } else {
                    basis[pivot] = signature;
                    rank += 1;
                    break;
                }
            }
        }
        rank
    }

    let mut columns = [0u128; 12];
    let mut firing_signature = 0u128;
    for input in 0usize..64 {
        for firing in 0usize..2 {
            let row = 2 * input + firing;
            let output = STRONG_SIX_CARRIER_U0[input] as usize ^ firing;
            let trace = input | (output << 6);
            for (wire, column) in columns.iter_mut().enumerate() {
                if (trace >> wire) & 1 != 0 {
                    *column |= 1u128 << row;
                }
            }
            if firing != 0 {
                firing_signature |= 1u128 << row;
            }
        }
    }

    let mut features = vec![u128::MAX];
    features.extend(columns);
    assert_eq!(gf2_rank(features.iter().copied()), 13);
    assert_eq!(
        gf2_rank(
            features
                .iter()
                .copied()
                .chain(std::iter::once(firing_signature))
        ),
        14
    );
    for a in 0..12 {
        for b in a + 1..12 {
            features.push(columns[a] & columns[b]);
        }
    }
    assert_eq!(gf2_rank(features.iter().copied()), 53);
    assert_eq!(
        gf2_rank(
            features
                .iter()
                .copied()
                .chain(std::iter::once(firing_signature))
        ),
        54,
        "degree-two endpoint features recovered strong-six firing"
    );
    for a in 0..12 {
        for b in a + 1..12 {
            for c in b + 1..12 {
                features.push(columns[a] & columns[b] & columns[c]);
            }
        }
    }
    assert_eq!(gf2_rank(features.iter().copied()), 103);
    assert_eq!(
        gf2_rank(
            features
                .iter()
                .copied()
                .chain(std::iter::once(firing_signature))
        ),
        103,
        "strong-six exact firing boundary moved above degree three"
    );
}

#[test]
fn strong_six_carrier_all_fold_modes_preserve_dirty_high_inputs() {
    let n = 6usize;
    let band = 6usize;
    let source = vec![
        XGate::from_g57([0, 1, 2]),
        XGate::cnot(3, 4),
        XGate::conj(5, [(0u16, true), (2u16, false)]).unwrap(),
    ];
    let low_mask = (1u64 << n) - 1;
    let total = 6 * n + band;
    let high_mask = ((1u64 << total) - 1) ^ low_mask;
    let junk_patterns = [
        0,
        high_mask,
        0xaaaa_aaaa_aaaa_aaaau64 & high_mask,
        0x36db_6db6_db6d_b6dbu64 & high_mask,
    ];

    // Exercise state-dispatched updates directly as well as through every
    // folding implementation: expanded, aggregate Gray, micro Gray, and
    // sentinel Gray.
    let state = SixCarrierState::strong_home(n);
    let mut update = Vec::new();
    state.emit_update(0, &mut update);
    assert_eq!(update.len(), STRONG_SIX_CARRIER_U0_GATES.len());

    for gray_fold in 0..=3usize {
        let mut prod = ProdConfig::production_six_carrier();
        prod.band = band;
        prod.gray_fold = gray_fold;
        prod.cg_jitter = 0;
        let mut rng = StdRng::seed_from_u64(0x6c0b_1c00 + gray_fold as u64);
        let gadget = gadgetize_xgates_strong_six_carrier(&source, n, 2, &prod, &mut rng);
        assert_eq!(gadget.num_wires, total);
        for low in 0..=low_mask {
            let expected = eval_u64(&source, low) & low_mask;
            for &junk in &junk_patterns {
                assert_eq!(
                    eval_u64(&gadget.gates, low | junk) & low_mask,
                    expected,
                    "strong-six mode={gray_fold} low={low:#x} junk={junk:#x}"
                );
            }
        }
    }
}

#[test]
fn strong_six_carrier_public_wrappers_have_the_expected_port() {
    // The nonlinear slice preblock needs enough data coordinates to
    // disturb its 5*n+band nonzero slices; n=3 is intentionally rejected
    // by its exhaustive constructor for this width.
    let n = 6usize;
    let band = 6usize;
    let mut prod = ProdConfig::production_six_carrier();
    prod.band = band;
    let total = 6 * n + band;
    let low_mask = (1u64 << n) - 1;
    let main = CircuitSeq {
        gates: vec![[0, 1, 2]],
    };
    let source = vec![XGate::from_g57([0, 1, 2])];

    let mut bare_cnot_rng = StdRng::seed_from_u64(0x6c06_0001);
    let bare_cnot = gadgetize_cnot_strong_six_carrier(&main, n, 1, &prod, &mut bare_cnot_rng);
    let mut bare_xgate_rng = StdRng::seed_from_u64(0x6c06_0002);
    let bare_xgate = gadgetize_xgates_strong_six_carrier(&source, n, 1, &prod, &mut bare_xgate_rng);
    let mut slice_cnot_rng = StdRng::seed_from_u64(0x6c06_0003);
    let slice_cnot = gadgetize_with_slice_zero_ccnot_strong_six_carrier(
        &main,
        n,
        1,
        10 * n,
        &MaskConfig::off(),
        &prod,
        &mut slice_cnot_rng,
    );
    let mut slice_xgate_rng = StdRng::seed_from_u64(0x6c06_0004);
    let slice_xgate = gadgetize_xgates_with_slice_zero_ccnot_strong_six_carrier(
        &source,
        n,
        1,
        10 * n,
        &prod,
        &mut slice_xgate_rng,
    );
    for circuit in [&bare_cnot, &bare_xgate, &slice_cnot, &slice_xgate] {
        assert_eq!(circuit.num_wires, total);
        for low in 0..=low_mask {
            assert_eq!(
                eval_u64(&circuit.gates, low) & low_mask,
                eval_u64(&source, low) & low_mask
            );
        }
    }
}

#[test]
fn seven_carrier_quartic_dirty_gather_restores_every_borrow() {
    // Wires: acc, h, A, B, C, D. Exercise both dirty inputs explicitly;
    // the adjusted Gray fold may not assume either scratch bit is clean.
    let atom = [(2u16, true), (3, true), (4, true), (5, true)];
    let mut rng = StdRng::seed_from_u64(0x7ca0_0001);
    let mut seen = std::collections::HashMap::new();
    let mut gates = Vec::new();
    assert!(!emit_atom_onto(
        0, &atom, 1, 0, 1, &mut seen, &mut rng, &mut gates,
    ));
    assert_eq!(gates.len(), 4);
    assert!(gates.iter().all(|gate| gate.width() <= 3));
    for input in 0u64..64 {
        let product =
            ((input >> 2) & 1) & ((input >> 3) & 1) & ((input >> 4) & 1) & ((input >> 5) & 1);
        let expected = input ^ product;
        assert_eq!(
            eval_u64(&gates, input),
            expected,
            "dirty quartic gather failed at {input:#08b}"
        );
    }
}

#[test]
fn seven_carrier_u0_realization_has_the_selected_truth_table() {
    let carriers = [0usize, 1, 2, 3, 4, 5, 6];
    let mut gates = Vec::new();
    emit_seven_carrier_update(&carriers, &mut gates);
    assert_eq!(gates.len(), 26, "the selected nonlinear U0 drifted");

    let mut seen_u0 = [false; 128];
    let mut seen_u1 = [false; 128];
    let mut classes = [0usize; 2];
    for input in 0u8..128 {
        let output = eval_u64(&gates, input as u64) as u8;
        assert_eq!(
            output, SEVEN_CARRIER_U0[input as usize],
            "U0 truth-table mismatch at {input:#09b}"
        );
        assert!(!seen_u0[output as usize], "U0 is not injective");
        seen_u0[output as usize] = true;
        assert_ne!(output, input, "U0 fixed point at {input:#09b}");
        assert_eq!(
            seven_carrier_decode_word(output),
            seven_carrier_decode_word(input),
            "U0 changed the decode class at {input:#09b}"
        );

        let output_u1 = output ^ 1;
        assert!(!seen_u1[output_u1 as usize], "U1 is not injective");
        seen_u1[output_u1 as usize] = true;
        assert_ne!(output_u1, input, "U1 fixed point at {input:#09b}");
        assert_ne!(
            seven_carrier_decode_word(output_u1),
            seven_carrier_decode_word(input),
            "U1 did not change the decode class at {input:#09b}"
        );
        classes[seven_carrier_decode_word(input) as usize] += 1;
    }
    assert!(seen_u0.into_iter().all(|seen| seen));
    assert!(seen_u1.into_iter().all(|seen| seen));
    assert_eq!(classes, [64, 64], "D must split the carrier space evenly");

    let widths: Vec<usize> = gates.iter().map(XGate::width).collect();
    for (width, expected) in [(0, 2), (1, 7), (2, 9), (3, 4), (4, 4)] {
        assert_eq!(
            widths.iter().filter(|&&actual| actual == width).count(),
            expected,
            "width-{width} gate count drifted"
        );
    }
}

#[test]
fn seven_carrier_decode_has_no_weight_one_or_two_static_parity_correlation() {
    let correlation_sum = |detector: u8| -> i32 {
        (0u8..128)
            .map(|carrier| {
                let prediction = (carrier & detector).count_ones() & 1 != 0;
                if prediction == seven_carrier_decode_word(carrier) {
                    1
                } else {
                    -1
                }
            })
            .sum()
    };
    for detector in 1u8..128 {
        if detector.count_ones() <= 2 {
            assert_eq!(
                correlation_sum(detector),
                0,
                "low-weight static detector {detector:#09b} is correlated"
            );
        }
    }
    let max_weight_three = (1u8..128)
        .filter(|detector| detector.count_ones() == 3)
        .map(|detector| correlation_sum(detector).unsigned_abs())
        .max()
        .unwrap();
    assert_eq!(
        max_weight_three, 16,
        "first normalized static magnitude must be 1/8"
    );
}

#[test]
fn seven_carrier_update_trace_has_zero_walsh_through_weight_three() {
    let walsh_sum = |detector: u16| -> i32 {
        let mut sum = 0i32;
        for input in 0u16..128 {
            for firing in 0u16..2 {
                let output = SEVEN_CARRIER_U0[input as usize] as u16 ^ firing;
                let trace = input | (output << 7);
                let prediction = (trace & detector).count_ones() as u16 & 1;
                sum += if prediction == firing { 1 } else { -1 };
            }
        }
        sum
    };

    for detector in 1u16..(1u16 << 14) {
        let weight = detector.count_ones();
        if weight <= 3 {
            assert_eq!(
                walsh_sum(detector),
                0,
                "weight-{weight} endpoint detector {detector:#016b} is correlated"
            );
        }
    }
    let weight_four: Vec<u32> = (1u16..(1u16 << 14))
        .filter(|detector| detector.count_ones() == 4)
        .map(|detector| walsh_sum(detector).unsigned_abs())
        .collect();
    assert_eq!(weight_four.iter().copied().max(), Some(64));
    assert_eq!(
        weight_four
            .iter()
            .filter(|&&magnitude| magnitude == 64)
            .count(),
        3,
        "maximum-bias weight-four endpoint multiplicity drifted"
    );
    assert_eq!(
        weight_four
            .iter()
            .filter(|&&magnitude| magnitude != 0)
            .count(),
        21,
        "weight-four endpoint spectrum drifted"
    );
}

#[test]
fn seven_carrier_endpoint_affine_and_degree_boundary() {
    type Signature = [u64; 4];

    fn and(left: Signature, right: Signature) -> Signature {
        std::array::from_fn(|word| left[word] & right[word])
    }

    fn gf2_rank(signatures: impl IntoIterator<Item = Signature>) -> usize {
        let mut basis = [[0u64; 4]; 256];
        let mut rank = 0usize;
        for mut signature in signatures {
            loop {
                let Some(pivot) = (0..256)
                    .rev()
                    .find(|&bit| signature[bit / 64] & (1u64 << (bit % 64)) != 0)
                else {
                    break;
                };
                if basis[pivot] == [0; 4] {
                    basis[pivot] = signature;
                    rank += 1;
                    break;
                }
                for word in 0..4 {
                    signature[word] ^= basis[pivot][word];
                }
            }
        }
        rank
    }

    fn with_firing_rank(features: &[Signature], firing: Signature) -> usize {
        gf2_rank(features.iter().copied().chain(std::iter::once(firing)))
    }

    let mut columns = [[0u64; 4]; 14];
    let mut firing = [0u64; 4];
    for input in 0usize..128 {
        for fires in 0usize..2 {
            let row = 2 * input + fires;
            let output = SEVEN_CARRIER_U0[input] as usize ^ fires;
            let trace = input | (output << 7);
            for (wire, column) in columns.iter_mut().enumerate() {
                if trace & (1usize << wire) != 0 {
                    column[row / 64] |= 1u64 << (row % 64);
                }
            }
            if fires != 0 {
                firing[row / 64] |= 1u64 << (row % 64);
            }
        }
    }

    let degree_one: Vec<Signature> = std::iter::once([u64::MAX; 4]).chain(columns).collect();
    assert_eq!(gf2_rank(degree_one.iter().copied()), 15);
    assert_eq!(with_firing_rank(&degree_one, firing), 16);

    let mut degree_two = degree_one.clone();
    for a in 0..14 {
        for b in a + 1..14 {
            degree_two.push(and(columns[a], columns[b]));
        }
    }
    assert_eq!(gf2_rank(degree_two.iter().copied()), 71);
    assert_eq!(with_firing_rank(&degree_two, firing), 72);

    let mut degree_three = degree_two.clone();
    for a in 0..14 {
        for b in a + 1..14 {
            for c in b + 1..14 {
                degree_three.push(and(and(columns[a], columns[b]), columns[c]));
            }
        }
    }
    assert_eq!(gf2_rank(degree_three.iter().copied()), 160);
    assert_eq!(with_firing_rank(&degree_three, firing), 161);

    let mut degree_four = degree_three.clone();
    for a in 0..14 {
        for b in a + 1..14 {
            for c in b + 1..14 {
                for d in c + 1..14 {
                    degree_four.push(and(
                        and(columns[a], columns[b]),
                        and(columns[c], columns[d]),
                    ));
                }
            }
        }
    }
    assert_eq!(gf2_rank(degree_four.iter().copied()), 226);
    assert_eq!(
        with_firing_rank(&degree_four, firing),
        226,
        "the exact recovery boundary must first appear at degree four"
    );
}

#[test]
fn seven_carrier_role_automorphisms_preserve_every_decode_class() {
    for seed in 0..64u64 {
        let mut rng = StdRng::seed_from_u64(0x7d15_7000 + seed);
        let roles = seven_carrier_role_automorphism(&mut rng);
        let mut seen_roles = [false; 7];
        for &role in &roles {
            assert!(!seen_roles[role as usize]);
            seen_roles[role as usize] = true;
        }
        for input in 0u8..128 {
            let relabeled = (0..7).fold(0u8, |word, canonical| {
                word | (((input >> canonical) & 1) << roles[canonical as usize])
            });
            assert_eq!(
                seven_carrier_decode_word(relabeled),
                seven_carrier_decode_word(input),
                "seed {seed} changed D at {input:#09b}"
            );
        }
    }
}

#[test]
fn seven_carrier_distributed_fold_is_exact_on_arbitrary_representatives() {
    let n = 3usize;
    let state = SevenCarrierState::home(n);
    let cfg = ProdConfig::off();
    let mut ledger = ProdLedger::new(n, &cfg, 7 * n, None);
    let mut rng = StdRng::seed_from_u64(0x7d15_f01d);
    let mut fold = Vec::new();
    ledger.fold_seven_distributed(&XGate::cnot(0, 1), &state, 0, n, &mut rng, &mut fold);
    // Six decode atoms produce six source fragments.  Five boundaries,
    // each carrying a three-gate shear, give 6 + 5*3 gates and no U0.
    assert_eq!(fold.len(), 21);
    assert!(
        fold.iter().all(|gate| !gate.ctrls.is_empty()),
        "the refresh fold emitted an always-firing gate"
    );
    let conditional_targets: std::collections::HashSet<u16> =
        fold.iter().map(|gate| gate.target).collect();
    assert_eq!(conditional_targets.len(), 5);

    let pack = |value: usize, carrier: u8| -> u64 {
        (0..7).fold(0u64, |word, lane| {
            word | ((((carrier >> lane) & 1) as u64) << (lane * n + value))
        })
    };
    let unpack = |physical: u64, value: usize| -> u8 {
        (0..7).fold(0u8, |carrier, lane| {
            carrier | ((((physical >> (lane * n + value)) & 1) as u8) << lane)
        })
    };
    for target_carrier in 0u8..128 {
        for source_carrier in 0u8..128 {
            let input = pack(0, target_carrier) | pack(1, source_carrier);
            let output = eval_u64(&fold, input);
            assert_eq!(
                seven_carrier_decode_word(unpack(output, 0)),
                seven_carrier_decode_word(target_carrier)
                    ^ seven_carrier_decode_word(source_carrier),
                "target={target_carrier:#09b} source={source_carrier:#09b}"
            );
            assert_eq!(
                unpack(output, 1),
                source_carrier,
                "the source representative was modified"
            );
        }
    }
}

#[test]
fn seven_carrier_partitioned_fold_reaches_128_and_is_exact() {
    // A CNOT's six source fragments need five selector bits to clear the
    // floor.  The eligible prefix supplies exactly values 2..6 after the
    // target/control exclusion; values 7 and 8 deliberately sit outside
    // it, modeling the sliced sandwich's fixed upper half.
    let n = 9usize;
    let live_helper_prefix = 7usize;
    let state = SevenCarrierState::home(n);
    let cfg = ProdConfig::off();
    let mut ledger = ProdLedger::new(n, &cfg, 7 * n, None);
    let mut rng = StdRng::seed_from_u64(0x7d15_1280);
    let mut fold = Vec::new();
    ledger.fold_seven_distributed(
        &XGate::cnot(0, 1),
        &state,
        128,
        live_helper_prefix,
        &mut rng,
        &mut fold,
    );
    // Six original decode atoms need five polarity bits: 6*32=192
    // branches.  There is one three-gate shear at every boundary.
    assert_eq!(ledger.distributed_fold_original_fragments, vec![6]);
    assert_eq!(ledger.distributed_fold_fragments, vec![192]);
    assert_eq!(ledger.cg_fragments, 192);
    assert_eq!(fold.len(), 192 + 3 * 191);
    assert!(fold.iter().all(|gate| !gate.ctrls.is_empty()));
    let expected_helpers: std::collections::HashSet<u16> = (2..7).collect();
    for fragment in fold.iter().step_by(4) {
        let actual_helpers: std::collections::HashSet<u16> = fragment
            .ctrls
            .iter()
            .filter_map(|&(wire, _)| expected_helpers.contains(&wire).then_some(wire))
            .collect();
        assert_eq!(actual_helpers, expected_helpers);
        assert!(
            fragment
                .ctrls
                .iter()
                .all(|&(wire, _)| wire != 7 && wire != 8)
        );
    }

    let pack = |value: usize, carrier: u8| -> u64 {
        (0..7).fold(0u64, |word, lane| {
            word | ((((carrier >> lane) & 1) as u64) << (lane * n + value))
        })
    };
    let unpack = |physical: u64, value: usize| -> u8 {
        (0..7).fold(0u8, |carrier, lane| {
            carrier | ((((physical >> (lane * n + value)) & 1) as u8) << lane)
        })
    };
    for helper_seed in [0u8, 0x7f, 0x55, 0x2a] {
        for target_carrier in 0u8..128 {
            for source_carrier in 0u8..128 {
                let helpers: Vec<u8> = (2..n)
                    .map(|value| helper_seed.rotate_left((value - 2) as u32) & 0x7f)
                    .collect();
                let mut input = pack(0, target_carrier) | pack(1, source_carrier);
                for (value, &helper) in (2..n).zip(&helpers) {
                    input |= pack(value, helper);
                }
                let output = eval_u64(&fold, input);
                assert_eq!(
                    seven_carrier_decode_word(unpack(output, 0)),
                    seven_carrier_decode_word(target_carrier)
                        ^ seven_carrier_decode_word(source_carrier)
                );
                assert_eq!(unpack(output, 1), source_carrier);
                for (value, &helper) in (2..n).zip(&helpers) {
                    assert_eq!(unpack(output, value), helper);
                }
            }
        }
    }
}

#[test]
fn seven_carrier_partitioned_floor1024_is_exact_and_shuffles_each_cell_block() {
    // A two-control fold has 6*6=36 source fragments. Five independent
    // helper bits raise it to 36*32=1152 branches, just over floor 1024.
    let n = 8usize;
    let state = SevenCarrierState::home(n);
    let cfg = ProdConfig::off();
    let mut ledger = ProdLedger::new(n, &cfg, 7 * n, None);
    let mut rng = StdRng::seed_from_u64(0x7d15_1024);
    let gate = XGate::conj(0, [(1u16, true), (2u16, true)]).unwrap();
    let mut fold = Vec::new();
    ledger.fold_seven_distributed(&gate, &state, 1024, n, &mut rng, &mut fold);
    assert_eq!(ledger.distributed_fold_original_fragments, vec![36]);
    assert_eq!(ledger.distributed_fold_fragments, vec![1152]);
    assert_eq!(ledger.distributed_fold_floors, vec![1024]);
    assert_eq!(fold.len(), 4 * 1152 - 3);

    // Source branches occupy every fourth position. Every 32-branch group
    // must enumerate the full cell cube, and independent shuffles should
    // give consecutive original fragments different orders.
    let branches: Vec<&XGate> = fold.iter().step_by(4).collect();
    assert_eq!(branches.len(), 1152);
    let pattern = |fragment: &XGate| -> u8 {
        (3u16..8).enumerate().fold(0u8, |word, (bit, helper)| {
            let polarity = fragment
                .ctrls
                .iter()
                .find_map(|&(wire, polarity)| (wire == helper).then_some(polarity))
                .expect("every branch must carry every helper literal");
            word | ((polarity as u8) << bit)
        })
    };
    let first: Vec<u8> = branches[..32].iter().map(|gate| pattern(gate)).collect();
    let second: Vec<u8> = branches[32..64].iter().map(|gate| pattern(gate)).collect();
    assert_eq!(
        first
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        32
    );
    assert_eq!(
        second
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        32
    );
    assert_ne!(first, second, "cell order was reused across fragments");

    let pack = |value: usize, carrier: u8| -> u64 {
        (0..7).fold(0u64, |word, lane| {
            word | ((((carrier >> lane) & 1) as u64) << (lane * n + value))
        })
    };
    let unpack = |physical: u64, value: usize| -> u8 {
        (0..7).fold(0u8, |carrier, lane| {
            carrier | ((((physical >> (lane * n + value)) & 1) as u8) << lane)
        })
    };
    for _ in 0..2048 {
        let carriers: [u8; 8] = std::array::from_fn(|_| rng.random_range(0..128));
        let input = carriers
            .iter()
            .enumerate()
            .fold(0u64, |word, (value, &carrier)| word | pack(value, carrier));
        let output = eval_u64(&fold, input);
        assert_eq!(
            seven_carrier_decode_word(unpack(output, 0)),
            seven_carrier_decode_word(carriers[0])
                ^ (seven_carrier_decode_word(carriers[1]) & seven_carrier_decode_word(carriers[2]))
        );
        for (value, &carrier) in carriers.iter().enumerate().skip(1) {
            assert_eq!(unpack(output, value), carrier);
        }
    }
}

#[test]
fn seven_carrier_initial_boundary_partition_preserves_complemented_gates() {
    let band: Vec<u16> = (20..32).collect();
    let supports: Vec<Vec<u64>> = (0..band.len()).map(|bit| vec![1u64 << bit]).collect();
    let original = vec![
        XGate::conj(0, [(1u16, true), (2u16, false)]).unwrap(),
        XGate::from_g57([3, 4, 5]),
    ];
    let mut rng = StdRng::seed_from_u64(0xb0a1_0d10);
    let mut partitioned = Vec::new();
    let emitted = emit_partitioned_initial_injection(
        &original,
        &band,
        &supports,
        4,
        &mut rng,
        &mut partitioned,
    );
    // Pure conjunction: 16 cells. Complemented conjunction: cell-only
    // plus F-and-cell in every cell, for 32 more emissions.
    assert_eq!(emitted, 48);
    assert!(partitioned.iter().all(|gate| !gate.comp));
    for _ in 0..4096 {
        let input = rng.random::<u64>() & ((1u64 << 32) - 1);
        assert_eq!(eval_u64(&partitioned, input), eval_u64(&original, input));
    }
}

#[test]
fn seven_carrier_floor4096_boundary_r10_preserves_the_zero_slice() {
    let n = 12usize;
    let band = 32usize;
    let total = 7 * n + band;
    let mut prod = ProdConfig::production_seven_carrier();
    prod.gray_fold = 0;
    prod.fill_nl = 0;
    prod.band = band;
    prod.cg_jitter = 0;
    let mut rng = StdRng::seed_from_u64(0xb0a1_0d11);
    let source = [XGate::cnot(0, 1)];
    let circuit = gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed_partitioned_floor4096_boundary_r10_live_prefix_unshuffled(
        &source,
        n,
        n,
        1,
        total,
        &prod,
        &mut rng,
    );
    assert_eq!(circuit.num_wires, total);
    assert!(
        circuit
            .gates
            .iter()
            .rev()
            .take(5 * n)
            .all(|gate| (gate.target as usize) < n),
        "the boundary fixture unexpectedly retained the terminal high-wire fill"
    );

    let evaluate = |input: u16| -> u16 {
        let mut state = vec![false; total];
        for wire in 0..n {
            state[wire] = input & (1 << wire) != 0;
        }
        for gate in &circuit.gates {
            let firing = gate.comp
                ^ gate
                    .ctrls
                    .iter()
                    .all(|&(wire, polarity)| state[wire as usize] == polarity);
            state[gate.target as usize] ^= firing;
        }
        (0..n).fold(0u16, |word, wire| word | ((state[wire] as u16) << wire))
    };
    for input in [0u16, 1, 0x555, 0xaaa, 0xfff, 0x31c, 0x8e7] {
        let expected = input ^ (((input >> 1) & 1) << 0);
        assert_eq!(evaluate(input), expected);
    }
}

#[test]
fn seven_carrier_floor4096_terminal_fence_is_exact_and_adj32_stays_in_prefix() {
    let n = 12usize;
    let band = 32usize;
    let total = 7 * n + band;
    let mut prod = ProdConfig::production_seven_carrier();
    prod.gray_fold = 0;
    prod.fill_nl = 0;
    prod.band = band;
    prod.cg_jitter = 0;
    let source = [XGate::cnot(0, 1)];

    let mut ordered_rng = StdRng::seed_from_u64(0xb0a1_0d12);
    let (ordered, ordered_terminal_start) = gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed_partitioned_floor4096_boundary_r10_live_prefix_terminal_fenced_unshuffled(
        &source,
        n,
        n,
        1,
        total,
        &prod,
        &mut ordered_rng,
    );
    let mut shuffled_rng = StdRng::seed_from_u64(0xb0a1_0d12);
    let (shuffled, shuffled_terminal_start) = gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed_partitioned_floor4096_boundary_r10_live_prefix_terminal_fenced_adj32(
        &source,
        n,
        n,
        1,
        total,
        &prod,
        &mut shuffled_rng,
    );

    assert_eq!(ordered.num_wires, shuffled.num_wires);
    assert_eq!(ordered.gates.len(), shuffled.gates.len());
    assert_eq!(ordered_terminal_start, shuffled_terminal_start);
    assert!(ordered_terminal_start > 0);
    assert!(ordered_terminal_start < ordered.gates.len());
    assert_eq!(
        &ordered.gates[ordered_terminal_start..],
        &shuffled.gates[shuffled_terminal_start..],
        "a prefix-only reorder modified the protected terminal suffix"
    );
    assert_ne!(
        &ordered.gates[..ordered_terminal_start],
        &shuffled.gates[..shuffled_terminal_start],
        "the deterministic adjacent passes did not reorder the prefix"
    );

    let evaluate = |circuit: &CnotCircuit, input: u16| -> u16 {
        let mut state = vec![false; total];
        for wire in 0..n {
            state[wire] = input & (1 << wire) != 0;
        }
        for gate in &circuit.gates {
            let firing = gate.comp
                ^ gate
                    .ctrls
                    .iter()
                    .all(|&(wire, polarity)| state[wire as usize] == polarity);
            state[gate.target as usize] ^= firing;
        }
        (0..n).fold(0u16, |word, wire| word | ((state[wire] as u16) << wire))
    };
    for input in [0u16, 1, 0x555, 0xaaa, 0xfff, 0x31c, 0x8e7] {
        let expected = input ^ (((input >> 1) & 1) << 0);
        assert_eq!(evaluate(&ordered, input), expected);
        assert_eq!(evaluate(&shuffled, input), expected);
    }
}

#[test]
fn seven_carrier_distributed_public_paths_preserve_dirty_high_inputs() {
    // Six values leave four distinct helpers for the CNOT's floor-128
    // partition while keeping the complete public port inside u64.
    let n = 6usize;
    let band = 8usize;
    let total = 7 * n + band;
    let source = vec![XGate::from_g57([0, 1, 2]), XGate::cnot(3, 0)];
    let mut prod = ProdConfig::production_seven_carrier();
    prod.gray_fold = 0;
    prod.band = band;
    prod.cg_jitter = 0;

    let mut shuffled_rng = StdRng::seed_from_u64(0x7d15_e2e1);
    let shuffled =
        gadgetize_xgates_seven_carrier_distributed(&source, n, 1, &prod, &mut shuffled_rng);
    let mut ordered_rng = StdRng::seed_from_u64(0x7d15_e2e2);
    let ordered = gadgetize_xgates_seven_carrier_distributed_unshuffled(
        &source,
        n,
        1,
        &prod,
        &mut ordered_rng,
    );
    let mut partitioned_rng = StdRng::seed_from_u64(0x7d15_e2e3);
    let partitioned = gadgetize_xgates_seven_carrier_distributed_partitioned(
        &source,
        n,
        1,
        &prod,
        &mut partitioned_rng,
    );
    let mut partitioned_ordered_rng = StdRng::seed_from_u64(0x7d15_e2e4);
    let partitioned_ordered = gadgetize_xgates_seven_carrier_distributed_partitioned_unshuffled(
        &source,
        n,
        1,
        &prod,
        &mut partitioned_ordered_rng,
    );
    let low_mask = (1u64 << n) - 1;
    let high_mask = ((1u64 << total) - 1) ^ low_mask;
    let junk = [
        0,
        high_mask,
        0xaaaa_aaaa & high_mask,
        0x5555_5555 & high_mask,
        0x9249_2492 & high_mask,
    ];
    for circuit in [&shuffled, &ordered, &partitioned, &partitioned_ordered] {
        assert_eq!(circuit.num_wires, total);
        for low in 0..=low_mask {
            let expected = eval_u64(&source, low) & low_mask;
            for &high in &junk {
                assert_eq!(
                    eval_u64(&circuit.gates, low | high) & low_mask,
                    expected,
                    "low={low:#x} high={high:#x}"
                );
            }
        }
    }
}

#[test]
fn rejected_direct_switch_trace_has_a_measured_boundary() {
    // This models the earlier direct class-switch candidate, not the
    // shipped opt-in shear fold.  Keep its falsification pinned: it looked
    // good in isolation but one later U0 put the firing bit back in span,
    // which is why the implemented path removes every fixed U0 instead.
    type Signature = [u64; 4];

    fn rank(signatures: impl IntoIterator<Item = Signature>) -> usize {
        let mut basis = [[0u64; 4]; 256];
        let mut rank = 0usize;
        for mut signature in signatures {
            loop {
                let Some(pivot) = (0..256)
                    .rev()
                    .find(|&bit| signature[bit / 64] & (1u64 << (bit % 64)) != 0)
                else {
                    break;
                };
                if basis[pivot] == [0; 4] {
                    basis[pivot] = signature;
                    rank += 1;
                    break;
                }
                for word in 0..4 {
                    signature[word] ^= basis[pivot][word];
                }
            }
        }
        rank
    }

    fn contains(features: &[Signature], target: Signature) -> bool {
        rank(features.iter().copied())
            == rank(features.iter().copied().chain(std::iter::once(target)))
    }

    // A fixed member of the randomized family: R toggles c0,c1 by !c3;
    // the middle is the c3/c5-oriented class-switch core.
    let plan: Vec<(u8, Vec<(u8, bool)>)> = vec![
        (0, vec![(3, false)]),
        (1, vec![(3, false)]),
        (3, vec![(5, true)]),
        (1, vec![(4, false)]),
        (1, vec![(4, true), (5, false)]),
        (1, vec![(4, true), (5, true), (6, true)]),
        (1, vec![(3, false)]),
        (0, vec![(3, false)]),
    ];

    let trace = |suffix_updates: usize| {
        let mut columns: Vec<Signature> = Vec::new();
        let mut owners: Vec<u8> = Vec::new();
        let mut firing = [0u64; 4];
        for input in 0u64..128 {
            for fires in 0u64..2 {
                let row = (2 * input + fires) as usize;
                if fires != 0 {
                    firing[row / 64] |= 1u64 << (row % 64);
                }
                let mut state = input;
                // Build each row independently, then merge it into the
                // already allocated chronological columns.
                let mut column_index = 0usize;
                let mut apply_and_record =
                    |target: u8, selector: &[(u8, bool)], conditional: bool| {
                        if !conditional || fires != 0 {
                            let fire = selector
                                .iter()
                                .all(|&(wire, polarity)| ((state >> wire) & 1 != 0) == polarity);
                            if fire {
                                state ^= 1u64 << target;
                            }
                        }
                        if columns.len() == column_index {
                            columns.push([0; 4]);
                            owners.push(target);
                        }
                        debug_assert_eq!(owners[column_index], target);
                        if state & (1u64 << target) != 0 {
                            columns[column_index][row / 64] |= 1u64 << (row % 64);
                        }
                        column_index += 1;
                    };

                for &(target, controls) in SEVEN_CARRIER_U0_GATES {
                    apply_and_record(target, controls, false);
                }
                for (target, selector) in &plan {
                    apply_and_record(*target, selector, true);
                }
                for _ in 0..suffix_updates {
                    for &(target, controls) in SEVEN_CARRIER_U0_GATES {
                        apply_and_record(target, controls, false);
                    }
                }
            }
        }
        let mut affine = vec![[u64::MAX; 4]];
        affine.extend(columns.iter().copied());
        (affine, owners, firing)
    };

    let (isolated, _, firing) = trace(0);
    assert!(
        !contains(&isolated, firing),
        "the isolated distributed block unexpectedly recovered its firing bit"
    );

    // Exact limitation: this local construction raises checkpoint distance
    // but does not remove the firing bit forever.  One subsequent fixed U0
    // already puts it back in the *global* affine span.
    let (after_one, _, firing) = trace(1);
    assert!(contains(&after_one, firing));

    // The corresponding one-wire catalog lasts longer for this fixed
    // member, but it too eventually closes after repeated identical U0s.
    let (after_six, owners_six, firing) = trace(6);
    for wire in 0..7u8 {
        let mut features = vec![[u64::MAX; 4]];
        features.extend(
            after_six
                .iter()
                .skip(1)
                .zip(&owners_six)
                .filter_map(|(&signature, &owner)| (owner == wire).then_some(signature)),
        );
        assert!(!contains(&features, firing), "wire {wire} closed too early");
    }
    let (after_seven, owners_seven, firing) = trace(7);
    assert!((0..7u8).any(|wire| {
        let mut features = vec![[u64::MAX; 4]];
        features.extend(
            after_seven
                .iter()
                .skip(1)
                .zip(&owners_seven)
                .filter_map(|(&signature, &owner)| (owner == wire).then_some(signature)),
        );
        contains(&features, firing)
    }));
}

#[test]
fn seven_carrier_gadget_preserves_low_wires_for_dirty_high_inputs() {
    let n = 6usize;
    let band = 6usize;
    let mut prod = ProdConfig::production_seven_carrier();
    prod.band = band;

    // Pin the production [2,2,2,3] mask plan and certify that the quartic
    // carrier decode still takes the adjusted Gray path. Keep this local
    // audit deterministic, while the end-to-end runs below retain the
    // production preset's extra per-value mask-atom jitter.
    let mut audit_prod = prod;
    audit_prod.cg_jitter = 0;
    let state = SevenCarrierState::home(n);
    let mut audit_rng = StdRng::seed_from_u64(0x7ca1_67a7);
    let mut ledger = ProdLedger::new(n, &audit_prod, 7 * n, None);
    let mut injection = Vec::new();
    ledger.inject_all(&state.c0_view(), &mut audit_rng, &mut injection);
    for slots in &ledger.slots {
        assert_eq!(
            slots
                .iter()
                .map(|slot| slot.factors.len())
                .collect::<Vec<_>>(),
            vec![2, 2, 2, 3]
        );
    }
    let mut optimized_fold = Vec::new();
    ledger.fold_seven(
        &XGate::from_g57([0, 1, 2]),
        &state,
        &mut audit_rng,
        &mut optimized_fold,
    );
    assert_eq!(
        ledger.cg_gray, 1,
        "fixture missed the seven-carrier Gray fold"
    );
    assert!(
        optimized_fold.iter().all(|gate| gate.width() <= 4),
        "adjusted Gray fold emitted a wider-than-decode gate"
    );

    let source = vec![
        XGate::from_g57([0, 1, 2]),
        XGate::cnot(3, 4),
        XGate::conj(5, [(0u16, true), (2u16, false)]).unwrap(),
    ];
    let low_mask = (1u64 << n) - 1;
    let total = 7 * n + band;
    let high_mask = ((1u64 << total) - 1) ^ low_mask;
    let fixed_junk = [
        0,
        high_mask,
        0xaaaa_aaaa_aaaa_aaaau64 & high_mask,
        0x5555_5555_5555_5555u64 & high_mask,
        0x9249_2492_4924_9249u64 & high_mask,
        0x36db_6db6_db6d_b6dbu64 & high_mask,
    ];

    for seed in 0..3u64 {
        let mut rng = StdRng::seed_from_u64(0x7ca1_0000 + seed);
        let gadget = gadgetize_xgates_seven_carrier(&source, n, 2, &prod, &mut rng);
        assert_eq!(gadget.num_wires, total, "expected 7*n + band wires");

        for low in 0..=low_mask {
            let expected = eval_u64(&source, low) & low_mask;
            for (junk_index, &junk) in fixed_junk.iter().enumerate() {
                assert_eq!(
                    eval_u64(&gadget.gates, low | junk) & low_mask,
                    expected,
                    "seed={seed} low={low:#x} junk pattern {junk_index}"
                );
            }
            let junk = (low
                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .rotate_left((low as u32) & 31))
                & high_mask;
            assert_eq!(
                eval_u64(&gadget.gates, low | junk) & low_mask,
                expected,
                "seed={seed} low={low:#x} hashed high junk"
            );
        }
    }
}

#[test]
fn seven_carrier_public_cnot_and_slice_wrappers_have_the_expected_port() {
    let n = 6usize;
    let band = 6usize;
    let mut prod = ProdConfig::production_seven_carrier();
    prod.band = band;
    prod.cg_jitter = 0;
    let total = 7 * n + band;
    let low_mask = (1u64 << n) - 1;
    let high_mask = ((1u64 << total) - 1) ^ low_mask;
    let main = CircuitSeq {
        gates: vec![[0, 1, 2]],
    };
    let source = vec![XGate::from_g57([0, 1, 2])];

    let mut bare_rng = StdRng::seed_from_u64(0x7ca2_0001);
    let bare = gadgetize_cnot_seven_carrier(&main, n, 1, &prod, &mut bare_rng);
    let mut slice_cnot_rng = StdRng::seed_from_u64(0x7ca2_0002);
    let slice_cnot = gadgetize_with_slice_zero_ccnot_seven_carrier(
        &main,
        n,
        1,
        10 * n,
        &MaskConfig::off(),
        &prod,
        &mut slice_cnot_rng,
    );
    let mut slice_xgate_rng = StdRng::seed_from_u64(0x7ca2_0003);
    let slice_xgate = gadgetize_xgates_with_slice_zero_ccnot_seven_carrier(
        &source,
        n,
        1,
        10 * n,
        &prod,
        &mut slice_xgate_rng,
    );
    assert_eq!(bare.num_wires, total);
    assert_eq!(slice_cnot.num_wires, total);
    assert_eq!(slice_xgate.num_wires, total);

    for low in 0..=low_mask {
        let expected = eval_u64(&source, low) & low_mask;
        assert_eq!(eval_u64(&bare.gates, low) & low_mask, expected);
        assert_eq!(eval_u64(&bare.gates, low | high_mask) & low_mask, expected);
        assert_eq!(eval_u64(&slice_cnot.gates, low) & low_mask, expected);
        assert_eq!(eval_u64(&slice_xgate.gates, low) & low_mask, expected);
    }
}

#[test]
fn seven_carrier_empty_source_is_the_identity_for_arbitrary_high_junk() {
    let n = 3usize;
    let mut prod = ProdConfig::production_seven_carrier();
    prod.band = 6;
    prod.cg_jitter = 0;
    let total = 7 * n + prod.band_size(n);
    let low_mask = (1u64 << n) - 1;
    let high_mask = ((1u64 << total) - 1) ^ low_mask;
    let mut rng = StdRng::seed_from_u64(0x7ca3_0001);
    let gadget = gadgetize_xgates_seven_carrier(&[], n, 1, &prod, &mut rng);
    assert_eq!(gadget.num_wires, total);

    for low in 0..=low_mask {
        for junk in [
            0,
            high_mask,
            0xaaaa_aaaa_aaaa_aaaau64 & high_mask,
            0x9249_2492_4924_9249u64 & high_mask,
        ] {
            assert_eq!(eval_u64(&gadget.gates, low | junk) & low_mask, low);
        }
    }
}
