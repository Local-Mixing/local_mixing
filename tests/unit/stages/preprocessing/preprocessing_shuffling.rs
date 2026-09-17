use super::*;
use crate::circuit::xgate::{eval_lanes, eval_u64};

fn fixture() -> (Vec<XGate>, Vec<FireBox>) {
    let mut gates = vec![XGate::cnot(6, 1)];
    let mut boxes = Vec::new();
    for target in [0, 1] {
        let start = gates.len();
        let mut cuts = vec![start];
        let operand = 1 - target;
        for _ in 0..12 {
            // A complete dirty-helper bracket: helper 4 is restored, while
            // the target gains a degree-three monomial.
            gates.push(XGate::conj(4, [(operand, true), (2, false)]).unwrap());
            gates.push(XGate::conj(target, [(4, true), (5, true)]).unwrap());
            gates.push(XGate::conj(4, [(operand, true), (2, false)]).unwrap());
            gates.push(XGate::conj(target, [(4, true), (5, true)]).unwrap());
            cuts.push(gates.len());
            gates.push(XGate::from_g57([target, 7, 8]));
            gates.push(XGate::cnot(target, 9));
            cuts.push(gates.len());
        }
        boxes.push(FireBox {
            target,
            start,
            end: gates.len(),
            mask_support: vec![(start, vec![7, 8, 9]), (cuts[12], vec![8, 9, 10])],
            unit_cuts: cuts,
        });
        // Reads the just-updated target outside the box. These gates exercise
        // cumulative layout mapping and removal of every temporary mask.
        gates.push(XGate::cnot(11, target));
    }
    gates.push(XGate::cnot(2, 12));
    (gates, boxes)
}

fn mapped_independently(gate: &XGate, layout: &[u16]) -> XGate {
    let mut ctrls = gate.ctrls.clone();
    for (wire, _) in &mut ctrls {
        *wire = layout[*wire as usize];
    }
    ctrls.sort_unstable();
    XGate {
        target: layout[gate.target as usize],
        comp: gate.comp,
        ctrls,
    }
}

/// Reconstruct packets, base gates and mask removals using only the baseline,
/// exact emitter ledger and recorded transfers. Also audit partner exclusions.
fn replay(gates: &[XGate], boxes: &[FireBox], output: &ShufflingOutput, np: usize) {
    let mut layout: Vec<u16> = (0..output.final_layout.len() as u16).collect();
    let mut reconstructed = Vec::new();
    let mut source_cursor = 0;
    for (fire_box, stats) in boxes.iter().zip(&output.stats.boxes) {
        for gate in &gates[source_cursor..fire_box.start] {
            reconstructed.push(mapped_independently(gate, &layout));
        }
        assert_eq!(stats.output_start, reconstructed.len());
        let mut pending = Vec::new();
        let mut used = Vec::new();
        let mut events = stats.transfers.iter().peekable();
        for cut in fire_box.start..=fire_box.end {
            if events.peek().is_some_and(|event| event.cut == cut) {
                let event = events.next().unwrap();
                assert!(fire_box.unit_cuts.contains(&cut));
                assert_eq!(layout[fire_box.target as usize], event.from);
                assert_eq!(layout[event.partner as usize], event.to);
                if event.returning_home {
                    assert_eq!(cut, fire_box.end, "return must fence the last segment");
                    assert_eq!(pending.remove(0), event.partner);
                    assert_eq!(event.to, fire_box.target);
                } else {
                    assert!(event.partner as usize >= np);
                    assert!(!used.contains(&event.partner));
                    let support = &fire_box
                        .mask_support
                        .iter()
                        .rev()
                        .find(|(at, _)| *at <= cut)
                        .unwrap()
                        .1;
                    assert!(!support.contains(&event.partner));
                    for unit_cuts in fire_box.unit_cuts.windows(2) {
                        if unit_cuts[0] < cut {
                            continue;
                        }
                        let unit = &gates[unit_cuts[0]..unit_cuts[1]];
                        if unit.iter().any(|gate| gate.target == fire_box.target) {
                            assert!(unit.iter().all(|gate| !gate.reads(event.partner)));
                        }
                    }
                    used.push(event.partner);
                    pending.push(event.partner);
                }
                reconstructed.push(XGate::cnot(event.to, event.from));
                reconstructed.push(XGate::cnot(event.from, event.to));
                layout.swap(fire_box.target as usize, event.partner as usize);
            }
            if cut != fire_box.end {
                reconstructed.push(mapped_independently(&gates[cut], &layout));
            }
        }
        assert!(events.next().is_none());
        for partner in pending {
            reconstructed.push(XGate::cnot(
                layout[fire_box.target as usize],
                layout[partner as usize],
            ));
        }
        assert_eq!(stats.output_end, reconstructed.len());
        let mut counts = std::collections::BTreeMap::new();
        for gate in &reconstructed[stats.output_start..stats.output_end] {
            *counts.entry(gate.target).or_insert(0usize) += 1;
        }
        assert_eq!(
            stats.all_target_counts,
            counts.into_iter().collect::<Vec<_>>()
        );
        assert_eq!(
            stats
                .original_target_counts
                .iter()
                .map(|(_, count)| count)
                .sum::<usize>(),
            gates[fire_box.start..fire_box.end]
                .iter()
                .filter(|gate| gate.target == fire_box.target)
                .count()
        );
        source_cursor = fire_box.end;
    }
    for gate in &gates[source_cursor..] {
        reconstructed.push(mapped_independently(gate, &layout));
    }
    assert_eq!(reconstructed, output.gates);
    assert_eq!(layout, output.final_layout);
}

#[test]
fn transfers_replay_and_preserve_4096_dirty_states_and_inverses() {
    let (gates, boxes) = fixture();
    for return_home in [false, true] {
        for seed in [0, 9001, u64::MAX] {
            let output = apply_shuffling(&gates, &boxes, 3, 32, 8, return_home, seed);
            replay(&gates, &boxes, &output, 3);
            assert!(output.stats.transfer_count > 0);
            if return_home {
                assert_eq!(output.final_layout[..3], [0, 1, 2]);
                assert!(
                    output.stats.boxes.iter().all(|item| item
                        .transfers
                        .last()
                        .unwrap()
                        .returning_home)
                );
            }
            let mut rng = StdRng::seed_from_u64(0xd17_57a7e);
            for _ in 0..64 {
                let input: Vec<u64> = (0..32).map(|_| rng.random()).collect();
                let mut baseline = input.clone();
                let mut shuffled = input.clone();
                eval_lanes(&gates, &mut baseline);
                eval_lanes(&output.gates, &mut shuffled);
                for role in 0..32 {
                    assert_eq!(baseline[role], shuffled[output.final_layout[role] as usize]);
                }
                eval_lanes(output.gates.iter().rev(), &mut shuffled);
                assert_eq!(shuffled, input);
            }
            for input in [0, u32::MAX as u64]
                .into_iter()
                .chain((0..32).map(|wire| 1u64 << wire))
            {
                let baseline = eval_u64(&gates, input);
                let shuffled = eval_u64(&output.gates, input);
                for role in 0..32 {
                    assert_eq!(
                        (baseline >> role) & 1,
                        (shuffled >> output.final_layout[role]) & 1
                    );
                }
                assert_eq!(eval_u64(output.gates.iter().rev(), shuffled), input);
            }
        }
    }
}

#[test]
fn skipping_unavailable_partners_never_drops_source_gates() {
    let gates = vec![XGate::cnot(0, 1); 48];
    let boxes = vec![FireBox {
        target: 0,
        start: 0,
        end: gates.len(),
        unit_cuts: (0..=gates.len()).collect(),
        // No mask exclusion: the remaining target-write controls alone must
        // make the only band role unavailable at every selected cut.
        mask_support: vec![(0, vec![])],
    }];
    for return_home in [false, true] {
        let output = apply_shuffling(&gates, &boxes, 1, 2, 8, return_home, 9001);
        assert_eq!(output.gates, gates);
        assert_eq!(output.final_layout, [0, 1]);
        assert_eq!(output.stats.skipped_cut_count, 8);
        assert_eq!(output.stats.transfer_count, 0);
        replay(&gates, &boxes, &output, 1);
    }
}

#[test]
fn dirty_helper_controls_are_excluded_even_when_target_gates_do_not_read_them() {
    // Role 3 is read only by helper writes, but participates in the bracket's
    // net update target ^= data_1 * role_3 * role_4. It cannot be an escort.
    let bracket = [
        XGate::conj(2, [(1, true), (3, true)]).unwrap(),
        XGate::conj(0, [(2, true), (4, true)]).unwrap(),
        XGate::conj(2, [(1, true), (3, true)]).unwrap(),
        XGate::conj(0, [(2, true), (4, true)]).unwrap(),
    ];
    let gates: Vec<_> = bracket
        .iter()
        .cloned()
        .cycle()
        .take(9 * bracket.len())
        .collect();
    let boxes = vec![FireBox {
        target: 0,
        start: 0,
        end: gates.len(),
        unit_cuts: (0..=gates.len()).step_by(4).collect(),
        mask_support: vec![(0, vec![])],
    }];
    for return_home in [false, true] {
        // All three band roles occur somewhere in the target-writing unit.
        let blocked = apply_shuffling(&gates, &boxes, 2, 5, 8, return_home, 7);
        assert_eq!(blocked.gates, gates);
        assert_eq!(blocked.stats.transfer_count, 0);

        // With one independent band role, transfers still run and preserve all
        // dirty inputs, including the helper, after decoding the final layout.
        let shuffled = apply_shuffling(&gates, &boxes, 2, 6, 8, return_home, 7);
        assert!(shuffled.stats.transfer_count > 0);
        assert!(
            shuffled.stats.boxes[0]
                .transfers
                .iter()
                .all(|event| event.partner == 5)
        );
        replay(&gates, &boxes, &shuffled, 2);
        for input in 0..64 {
            let expected = eval_u64(&gates, input);
            let actual = eval_u64(&shuffled.gates, input);
            for role in 0..6 {
                assert_eq!(
                    (expected >> role) & 1,
                    (actual >> shuffled.final_layout[role]) & 1
                );
            }
            assert_eq!(eval_u64(shuffled.gates.iter().rev(), actual), input);
        }
    }
}

#[test]
fn return_home_still_runs_after_all_intermediate_cuts_are_skipped() {
    let gates = vec![XGate::x_gate(0); 16];
    let boxes = vec![FireBox {
        target: 0,
        start: 0,
        end: gates.len(),
        unit_cuts: (0..=gates.len()).collect(),
        mask_support: vec![(0, vec![])],
    }];
    let output = apply_shuffling(&gates, &boxes, 1, 2, 8, true, 123);
    assert_eq!(output.stats.transfer_count, 2);
    assert_eq!(output.stats.skipped_cut_count, 6);
    assert_eq!(output.stats.added_gates, 4);
    assert_eq!(output.final_layout, [0, 1]);
    replay(&gates, &boxes, &output, 1);
    for input in 0..4 {
        assert_eq!(eval_u64(&gates, input), eval_u64(&output.gates, input));
    }
}

#[test]
fn updated_mask_support_can_make_a_late_partner_available() {
    let gates = vec![XGate::x_gate(0); 16];
    let boxes = vec![FireBox {
        target: 0,
        start: 0,
        end: gates.len(),
        unit_cuts: (0..=gates.len()).collect(),
        mask_support: vec![(0, vec![1]), (12, vec![])],
    }];
    let output = apply_shuffling(&gates, &boxes, 1, 2, 8, true, 123);
    assert_eq!(output.stats.transfer_count, 2);
    assert_eq!(output.stats.skipped_cut_count, 6);
    assert_eq!(output.stats.boxes[0].transfers[0].cut, 12);
    assert_eq!(output.stats.boxes[0].transfers[1].cut, gates.len());
    assert_eq!(output.final_layout, [0, 1]);
    replay(&gates, &boxes, &output, 1);
    for input in 0..4 {
        assert_eq!(eval_u64(&gates, input), eval_u64(&output.gates, input));
    }
}

#[test]
fn return_home_removes_first_mask_and_closes_remaining_masks() {
    let gates = vec![XGate::x_gate(0); 16];
    let boxes = vec![FireBox {
        target: 0,
        start: 0,
        end: gates.len(),
        unit_cuts: (0..=gates.len()).collect(),
        mask_support: vec![(0, vec![])],
    }];
    let output = apply_shuffling(&gates, &boxes, 1, 10, 8, true, 123);
    let events = &output.stats.boxes[0].transfers;
    assert_eq!(events.len(), 8);
    assert_eq!(events[0].partner, events[7].partner);
    assert!(events[7].returning_home);
    assert_eq!(events[7].cut, gates.len());
    assert!(
        output.stats.boxes[0]
            .original_target_counts
            .iter()
            .all(|&(wire, _)| wire != 0)
    );
    assert_eq!(output.stats.added_gates, 22); // 8 two-CNOT packets + 6 closes.
    replay(&gates, &boxes, &output, 1);
    for input in 0..1 << 10 {
        let baseline = eval_u64(&gates, input);
        let shuffled = eval_u64(&output.gates, input);
        for role in 0..10 {
            assert_eq!(
                (baseline >> role) & 1,
                (shuffled >> output.final_layout[role]) & 1
            );
        }
    }
}

#[test]
fn one_available_cut_keeps_data_home() {
    let gates = vec![XGate::cnot(0, 1)];
    let fire_box = FireBox {
        target: 0,
        start: 0,
        end: 1,
        unit_cuts: vec![0, 1],
        mask_support: vec![(0, vec![])],
    };
    let output = apply_shuffling(&gates, &[fire_box.clone()], 3, 5, 8, true, 1);
    assert_eq!(output.gates, gates);
    assert_eq!(output.stats.skipped_cut_count, 1);
    assert_eq!(output.final_layout, [0, 1, 2, 3, 4]);
    // Zero-write boxes have no transfers and leave the gate stream intact.
    let mut no_target_writes = fire_box.clone();
    no_target_writes.target = 2;
    let output = apply_shuffling(&gates, &[no_target_writes], 3, 5, 8, true, 1);
    assert_eq!(output.gates, gates);
    assert_eq!(output.final_layout, [0, 1, 2, 3, 4]);
}

#[test]
fn mapping_keeps_polarities_and_sorts_controls() {
    let gate = XGate::from_g57([0, 1, 2]);
    let mapped = remap_gate(&gate, &[3, 2, 0, 1]);
    assert_eq!(mapped.target, 3);
    assert!(mapped.comp);
    assert_eq!(mapped.ctrls.as_slice(), &[(0, true), (2, false)]);
}

#[test]
fn bounded_quantiles_match_naive_selection_and_handle_maximum_segments() {
    for shift in 0..5 {
        let gates: Vec<_> = (0..37)
            .map(|index| XGate::x_gate(u16::from((index + shift) % 3 == 0)))
            .collect();
        let fire_box = FireBox {
            target: 0,
            start: 0,
            end: gates.len(),
            unit_cuts: (0..gates.len()).step_by(2).chain([gates.len()]).collect(),
            mask_support: vec![(0, vec![])],
        };
        let mut prefix = vec![0usize];
        for gate in &gates {
            prefix.push(prefix.last().unwrap() + usize::from(gate.target == 0));
        }
        for segments in 8..=100 {
            let mut naive = vec![0];
            for quantile in 1..segments {
                let cut = *fire_box
                    .unit_cuts
                    .iter()
                    .filter(|&&cut| cut > 0 && cut < gates.len())
                    .min_by_key(|&&cut| {
                        (
                            (prefix[cut] * segments).abs_diff(prefix[gates.len()] * quantile),
                            cut,
                        )
                    })
                    .unwrap();
                naive.push(cut);
            }
            naive.sort_unstable();
            naive.dedup();
            assert_eq!(selected_cuts(&gates, &fire_box, segments), naive);
        }
        let maximum = selected_cuts(&gates, &fire_box, usize::MAX);
        assert!(maximum.len() <= fire_box.unit_cuts.len() - 1);
        assert_eq!(maximum.first(), Some(&0));
        assert!(maximum.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(maximum.iter().all(|cut| fire_box.unit_cuts.contains(cut)));
        assert!(!maximum.contains(&fire_box.end));
    }
}

#[test]
fn disabled_shuffling_preserves_exact_stream_and_identity_layout() {
    let (gates, _) = fixture();
    let output = apply_shuffling(&gates, &[], 3, 32, 0, true, 777);
    assert_eq!(output.gates, gates);
    assert_eq!(output.final_layout, (0..32).collect::<Vec<u16>>());
    assert_eq!(output.stats.added_gates, 0);
}

#[test]
#[should_panic(expected = "never read inside its fire box")]
fn target_read_in_box_is_rejected() {
    let gates = vec![XGate::cnot(1, 0)];
    let boxes = vec![FireBox {
        target: 0,
        start: 0,
        end: 1,
        unit_cuts: vec![0, 1],
        mask_support: vec![(0, vec![])],
    }];
    apply_shuffling(&gates, &boxes, 1, 2, 8, true, 1);
}

#[test]
#[should_panic]
fn mask_snapshot_inside_atomic_unit_is_rejected() {
    let gates = vec![XGate::cnot(0, 1); 4];
    let boxes = vec![FireBox {
        target: 0,
        start: 0,
        end: 4,
        unit_cuts: vec![0, 4],
        mask_support: vec![(0, vec![]), (1, vec![1])],
    }];
    apply_shuffling(&gates, &boxes, 1, 3, 8, true, 1);
}
