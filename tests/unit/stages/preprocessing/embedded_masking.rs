use super::*;
use rand::Rng;

// A random g57 circuit on `n` wires (from_g57 triples with distinct wires).
fn random_a(n: u16, m: usize, rng: &mut StdRng) -> Vec<XGate> {
    let mut out = Vec::with_capacity(m);
    while out.len() < m {
        let a = rng.random_range(0..n);
        let x = rng.random_range(0..n);
        let y = rng.random_range(0..n);
        if a != x && a != y && x != y {
            out.push(XGate::from_g57([a, x, y]));
        }
    }
    out
}

// The gadget must compute A on the data wires for EVERY data input and
// EVERY band state (masks open and close symmetrically; band updates never
// straddle the masks that read them). Exhaustive over the data, sampled
// over the band, both read modes, several seeds.
#[test]
fn gadget_computes_a_exhaustively_in_both_read_modes() {
    let n: u16 = 6;
    let np = n as usize;
    for seed in 0..6u64 {
        let mut rng = StdRng::seed_from_u64(0xB5_0000 + seed);
        let a = random_a(n, 24, &mut rng);
        // (quad_fire, balanced, encoded_io, k, repair slots): both read modes,
        // both mask kinds, encoded I/O, odd K (K3 == K2 + the balancing wire),
        // K4 (two pairs per LGI) and REPAIR-kind rerand slots.
        for (quad_fire, balanced, encoded_io, k, repair) in [
            (true, false, false, 2, 0),
            (false, false, false, 2, 0),
            (true, true, false, 2, 0),
            (false, true, false, 2, 0),
            (true, false, true, 2, 0),
            (true, true, true, 2, 0),
            (true, true, false, 3, 0),
            (true, false, false, 3, 0),
            (true, true, false, 4, 2),
            (false, true, false, 4, 2),
            (true, true, true, 2, 2),
        ] {
            let p = EmbeddedMaskingParams {
                quad_fire,
                balanced,
                encoded_io,
                k,
                rerand_repair: repair,
                // exercise both the min-open rule (default 2) and its absence
                min_open: if repair > 0 { 1 } else { 2 },
                ..EmbeddedMaskingParams::production(100 + seed)
            };
            let out = preprocess_embedded_masking(&a, np, &p);
            assert_eq!(out.num_wires, 2 * np);
            // no clean ancillas: the fire borrows dirty band wires
            assert_eq!(out.scratch_wires, 0);
            assert_eq!(out.pre_gates.is_empty(), !encoded_io);
            assert_eq!(out.post_gates.is_empty(), !encoded_io);
            if encoded_io {
                // every data wire is masked at the start and at the end
                for w in 0..np as u16 {
                    assert!(out.pre_gates.iter().any(|g| g.target == w));
                    assert!(out.post_gates.iter().any(|g| g.target == w));
                }
            }
            for band in 0..8u64 {
                let band_bits = if band == 0 {
                    0
                } else {
                    rng.random::<u64>() >> (64 - np)
                };
                for data in 0..(1u64 << np) {
                    let mut expect = data;
                    for g in &a {
                        expect = g.apply_u64(expect);
                    }
                    let mut st = data | (band_bits << np);
                    for g in out
                        .pre_gates
                        .iter()
                        .chain(&out.gates)
                        .chain(&out.post_gates)
                    {
                        st = g.apply_u64(st);
                    }
                    assert_eq!(
                        st & ((1u64 << np) - 1),
                        expect,
                        "seed {seed} quad_fire={quad_fire} balanced={balanced} encoded_io={encoded_io} k={k} repair={repair} band {band:#x} data {data:#x}"
                    );
                }
            }
            if quad_fire {
                // the masked fire is two-control only (the DB is a g57 ball)
                assert!(out.gates.iter().all(|g| g.ctrls.len() <= 2));
                // and the scratch wires are clean again after every fire:
                // checked implicitly by the exhaustive equivalence above
            }
        }
    }
}

fn assert_shuffling_layout(output: &EmbeddedMaskingOutput, np: usize, return_home: bool) {
    assert_eq!(output.final_layout.len(), output.num_wires);
    let mut destinations = output.final_layout.clone();
    destinations.sort_unstable();
    assert_eq!(
        destinations,
        (0..output.num_wires as u16).collect::<Vec<_>>()
    );
    if return_home {
        assert_eq!(
            output.final_layout[..np],
            (0..np as u16).collect::<Vec<_>>()
        );
    }
    for gate in output
        .pre_gates
        .iter()
        .chain(&output.gates)
        .chain(&output.post_gates)
    {
        assert!((gate.target as usize) < output.num_wires);
        assert!(
            gate.ctrls
                .iter()
                .all(|&(wire, _)| { (wire as usize) < output.num_wires && wire != gate.target })
        );
        assert!(gate.ctrls.windows(2).all(|pair| pair[0].0 < pair[1].0));
    }
}

// Every lane is a separate full state. This lets the dirty-state checks cover
// thousands of assignments without traversing the circuit once per assignment.
fn shuffling_sample_lanes(samples: &[u64], width: usize) -> Vec<[u64; 4]> {
    assert!(samples.len() <= 256);
    let mut state = vec![[0u64; 4]; width];
    for (sample_index, &sample) in samples.iter().enumerate() {
        for (wire, lanes) in state.iter_mut().enumerate() {
            lanes[sample_index / 64] |= ((sample >> wire) & 1) << (sample_index % 64);
        }
    }
    state
}

fn assert_emitted_shuffling_replays(
    baseline: &EmbeddedMaskingOutput,
    shuffled: &EmbeddedMaskingOutput,
    np: usize,
) {
    // Rebuild the output using recorded events, independently of the routing
    // implementation. The emitter ledger, not inferred gate patterns, supplies
    // the complete-unit cuts and live mask support used for this audit.
    let map_gate = |gate: &XGate, layout: &[u16]| {
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
    };
    let mut layout: Vec<u16> = (0..baseline.num_wires as u16).collect();
    let mut rebuilt = Vec::new();
    let mut cursor = 0;
    assert_eq!(
        shuffled.shuffling_boxes.len(),
        shuffled.shuffling_stats.boxes.len()
    );
    for (fire_box, stats) in shuffled
        .shuffling_boxes
        .iter()
        .zip(&shuffled.shuffling_stats.boxes)
    {
        assert_eq!(fire_box.start, stats.source_start);
        assert_eq!(fire_box.end, stats.source_end);
        assert_eq!(fire_box.target, stats.target);
        for gate in &baseline.gates[cursor..fire_box.start] {
            rebuilt.push(map_gate(gate, &layout));
        }
        assert_eq!(rebuilt.len(), stats.output_start);
        let mut events = stats.transfers.iter().peekable();
        let mut used = BTreeSet::new();
        let mut pending = Vec::new();
        for cut in fire_box.start..=fire_box.end {
            if events.peek().is_some_and(|event| event.cut == cut) {
                let event = events.next().unwrap();
                assert!(fire_box.unit_cuts.contains(&cut));
                assert_eq!(event.from, layout[fire_box.target as usize]);
                assert_eq!(event.to, layout[event.partner as usize]);
                assert!(event.partner as usize >= np);
                if event.returning_home {
                    assert_eq!(cut, fire_box.end, "return must fence the final segment");
                    assert_eq!(pending.remove(0), event.partner);
                    assert_eq!(event.to, fire_box.target);
                } else {
                    assert!(
                        used.insert(event.partner),
                        "temporary masks must be distinct"
                    );
                    let support = &fire_box
                        .mask_support
                        .iter()
                        .rev()
                        .find(|(at, _)| *at <= cut)
                        .unwrap()
                        .1;
                    assert!(
                        !support.contains(&event.partner),
                        "partner is in a live target mask"
                    );
                    for unit_cuts in fire_box.unit_cuts.windows(2) {
                        if unit_cuts[0] < cut {
                            continue;
                        }
                        let unit = &baseline.gates[unit_cuts[0]..unit_cuts[1]];
                        if unit.iter().any(|gate| gate.target == fire_box.target) {
                            assert!(
                                unit.iter().all(|gate| !gate.reads(event.partner)),
                                "partner is read in a remaining target-writing unit"
                            );
                        }
                    }
                    pending.push(event.partner);
                }
                rebuilt.push(XGate::cnot(event.to, event.from));
                rebuilt.push(XGate::cnot(event.from, event.to));
                layout.swap(fire_box.target as usize, event.partner as usize);
            }
            if cut < fire_box.end {
                let gate = &baseline.gates[cut];
                assert!(
                    !gate.reads(fire_box.target),
                    "target read inside a fire box"
                );
                rebuilt.push(map_gate(gate, &layout));
            }
        }
        assert!(events.next().is_none());
        for partner in pending {
            rebuilt.push(XGate::cnot(
                layout[fire_box.target as usize],
                layout[partner as usize],
            ));
        }
        assert_eq!(rebuilt.len(), stats.output_end);
        cursor = fire_box.end;
    }
    for gate in &baseline.gates[cursor..] {
        rebuilt.push(map_gate(gate, &layout));
    }
    assert_eq!(
        rebuilt, shuffled.gates,
        "packets, remapped gates and closes must replay exactly"
    );
    assert_eq!(layout, shuffled.final_layout);
    assert_eq!(baseline.pre_gates, shuffled.pre_gates);
    assert_eq!(
        baseline
            .post_gates
            .iter()
            .map(|gate| map_gate(gate, &layout))
            .collect::<Vec<_>>(),
        shuffled.post_gates
    );
}

fn assert_shuffling_state_equivalence(
    source: &[XGate],
    baseline: &EmbeddedMaskingOutput,
    shuffled: &EmbeddedMaskingOutput,
    np: usize,
    samples: &[u64],
    check_raw_compute: bool,
    context: &str,
) {
    use crate::circuit::eval_lanes4;

    assert_emitted_shuffling_replays(baseline, shuffled, np);
    for inputs in samples.chunks(256) {
        let initial = shuffling_sample_lanes(inputs, baseline.num_wires);
        if check_raw_compute {
            // Compute equivalence must hold for arbitrary encoded or unencoded
            // states, independently of any relationship between band and data.
            let mut expected = initial.clone();
            let mut actual = initial.clone();
            eval_lanes4(&baseline.gates, &mut expected);
            eval_lanes4(&shuffled.gates, &mut actual);
            for (role, &physical) in shuffled.final_layout.iter().enumerate() {
                assert_eq!(
                    actual[physical as usize], expected[role],
                    "compute role={role}: {context}"
                );
            }
            eval_lanes4(shuffled.gates.iter().rev(), &mut actual);
            assert_eq!(actual, initial, "compute inverse: {context}");
        }

        let mut expected = initial.clone();
        let mut actual = initial.clone();
        eval_lanes4(
            baseline
                .pre_gates
                .iter()
                .chain(&baseline.gates)
                .chain(&baseline.post_gates),
            &mut expected,
        );
        eval_lanes4(
            shuffled
                .pre_gates
                .iter()
                .chain(&shuffled.gates)
                .chain(&shuffled.post_gates),
            &mut actual,
        );
        for (role, &physical) in shuffled.final_layout.iter().enumerate() {
            assert_eq!(
                actual[physical as usize], expected[role],
                "decoded role={role}: {context}"
            );
        }
        let mut source_result = initial.clone();
        eval_lanes4(source, &mut source_result);
        for (role, &value) in source_result.iter().enumerate().take(np) {
            assert_eq!(
                actual[shuffled.final_layout[role] as usize], value,
                "source role={role}: {context}"
            );
        }
        eval_lanes4(
            shuffled
                .post_gates
                .iter()
                .rev()
                .chain(shuffled.gates.iter().rev())
                .chain(shuffled.pre_gates.iter().rev()),
            &mut actual,
        );
        assert_eq!(actual, initial, "composed inverse: {context}");
    }
}

#[test]
fn preprocessing_shuffling_preserves_exhaustive_data_outputs_in_both_read_modes() {
    let np = 6;
    let band_width = 24;
    for seed in 0..3 {
        let mut rng = StdRng::seed_from_u64(0x510F_F1E0 + seed);
        let mut source = random_a(np as u16, 8, &mut rng);
        // Include mixed polarity and a single-control source operation, as well
        // as the g57 fixtures; all helpers and band wires remain dirty.
        source.push(XGate::conj(0, [(1, false)]).unwrap());
        source.push(XGate::conj(2, [(3, true), (4, false)]).unwrap());
        let band_mask = (1u64 << band_width) - 1;
        let bands = [
            0,
            band_mask,
            rng.random::<u64>() & band_mask,
            rng.random::<u64>() & band_mask,
        ];
        let samples: Vec<_> = bands
            .into_iter()
            .flat_map(|band| (0..1u64 << np).map(move |data| data | (band << np)))
            .collect();
        for (quad_fire, balanced, k, repair) in [
            (true, true, 2, 0),
            (true, false, 2, 0),
            (false, true, 2, 0),
            (false, false, 2, 0),
            (true, true, 4, 2),
            (false, true, 4, 2),
        ] {
            for encoded_io in [false, true] {
                let params = EmbeddedMaskingParams {
                    r: band_width,
                    quad_fire,
                    balanced,
                    encoded_io,
                    k,
                    rerand_repair: repair,
                    min_open: if repair > 0 { 1 } else { 2 },
                    ..EmbeddedMaskingParams::production(seed + 317)
                };
                let baseline = preprocess_embedded_masking(&source, np, &params);
                for return_home in [false, true] {
                    let shuffled = preprocess_embedded_masking(
                        &source,
                        np,
                        &EmbeddedMaskingParams {
                            shuffling_segments: 8,
                            shuffling_return_home: return_home,
                            ..params
                        },
                    );
                    let context = format!(
                        "seed={seed} quad={quad_fire} balanced={balanced} encoded={encoded_io} k={k} repair={repair} home={return_home}"
                    );
                    assert_shuffling_layout(&shuffled, np, return_home);
                    assert_eq!(shuffled.pre_gates, baseline.pre_gates, "encoder: {context}");
                    assert_eq!(shuffled.scratch_wires, 0);
                    assert!(
                        shuffled.shuffling_stats.transfer_count > 0,
                        "inactive fixture: {context}"
                    );
                    if quad_fire {
                        assert!(shuffled.gates.iter().all(|gate| gate.ctrls.len() <= 2));
                    }
                    assert_shuffling_state_equivalence(
                        &source, &baseline, &shuffled, np, &samples, false, &context,
                    );
                }
            }
        }
    }
}

#[test]
fn preprocessing_shuffling_matches_all_wires_on_4096_dirty_states_and_every_one_hot() {
    let np = 8;
    // Whole-unit exclusions also include dirty-helper operands. Keep enough
    // independent band roles to exercise transfers in every read-mode fixture.
    let band_width = 48;
    let width = np + band_width;
    let mut rng = StdRng::seed_from_u64(0xD177_5A7E);
    let source = random_a(np as u16, 12, &mut rng);
    let mask = (1u64 << width) - 1;
    let mut samples: Vec<_> = (0..4096).map(|_| rng.random::<u64>() & mask).collect();
    samples.extend([0, mask]);
    samples.extend((0..width).map(|wire| 1u64 << wire));
    for (index, (quad_fire, balanced, encoded_io, k, repair)) in [
        (true, true, false, 2, 0),
        (false, true, true, 2, 0),
        (true, false, true, 4, 2),
        (false, false, false, 4, 2),
    ]
    .into_iter()
    .enumerate()
    {
        let params = EmbeddedMaskingParams {
            r: band_width,
            quad_fire,
            balanced,
            encoded_io,
            k,
            rerand_repair: repair,
            min_open: if repair > 0 { 1 } else { 2 },
            ..EmbeddedMaskingParams::production(941 + index as u64)
        };
        let baseline = preprocess_embedded_masking(&source, np, &params);
        for return_home in [false, true] {
            let shuffled = preprocess_embedded_masking(
                &source,
                np,
                &EmbeddedMaskingParams {
                    shuffling_segments: 8,
                    shuffling_return_home: return_home,
                    ..params
                },
            );
            assert_shuffling_layout(&shuffled, np, return_home);
            assert!(
                shuffled.shuffling_stats.transfer_count > 0,
                "dirty fixture={index} home={return_home} must exercise transfers"
            );
            assert_shuffling_state_equivalence(
                &source,
                &baseline,
                &shuffled,
                np,
                &samples,
                true,
                &format!("dirty fixture={index} home={return_home}"),
            );
        }
    }
}

#[test]
fn preprocessing_shuffling_preserves_existing_zero_control_lowering() {
    use crate::circuit::eval_lanes4;

    // Quadratic-fire and linear-read emitters use different zero-control
    // complement conventions. This test checks that shuffling preserves each
    // emitter's output.
    let source = vec![
        XGate::x_gate(0),
        XGate {
            target: 1,
            comp: true,
            ctrls: Default::default(),
        },
        XGate::from_g57([2, 0, 1]),
    ];
    let mut rng = StdRng::seed_from_u64(0x0C07_7010);
    let samples: Vec<_> = (0..256)
        .map(|_| rng.random::<u64>() & ((1u64 << 30) - 1))
        .collect();
    for quad_fire in [false, true] {
        let params = EmbeddedMaskingParams {
            r: 24,
            quad_fire,
            encoded_io: true,
            ..EmbeddedMaskingParams::production(107)
        };
        let baseline = preprocess_embedded_masking(&source, 6, &params);
        for return_home in [false, true] {
            let shuffled = preprocess_embedded_masking(
                &source,
                6,
                &EmbeddedMaskingParams {
                    shuffling_segments: 8,
                    shuffling_return_home: return_home,
                    ..params
                },
            );
            assert_shuffling_layout(&shuffled, 6, return_home);
            let mut expected = shuffling_sample_lanes(&samples, baseline.num_wires);
            let mut actual = expected.clone();
            eval_lanes4(&baseline.gates, &mut expected);
            eval_lanes4(&shuffled.gates, &mut actual);
            for (role, &physical) in shuffled.final_layout.iter().enumerate() {
                assert_eq!(
                    actual[physical as usize], expected[role],
                    "quad={quad_fire} home={return_home} role={role}"
                );
            }
        }
    }
}

#[test]
fn preprocessing_shuffling_reports_write_distribution_and_bounded_overhead() {
    let np = 32;
    let source: Vec<_> = (0..64)
        .map(|i| {
            XGate::from_g57([
                (i % np) as u16,
                ((i + 3) % np) as u16,
                ((i + 5) % np) as u16,
            ])
        })
        .collect();
    for seed in 7064..=7069 {
        let params = EmbeddedMaskingParams {
            encoded_io: true,
            ..EmbeddedMaskingParams::production(seed)
        };
        let baseline = preprocess_embedded_masking(&source, np, &params);
        for return_home in [false, true] {
            let shuffled = preprocess_embedded_masking(
                &source,
                np,
                &EmbeddedMaskingParams {
                    shuffling_segments: 8,
                    shuffling_return_home: return_home,
                    ..params
                },
            );
            let stats = &shuffled.shuffling_stats;
            assert_emitted_shuffling_replays(&baseline, &shuffled, np);
            assert_eq!(stats.boxes.len(), source.len());
            assert_eq!(stats.original_gates, baseline.gates.len());
            assert_eq!(stats.shuffled_gates, shuffled.gates.len());
            assert_eq!(
                stats.added_gates,
                shuffled.gates.len() - baseline.gates.len()
            );
            let mut peaks = Vec::new();
            let mut baseline_peaks = Vec::new();
            let mut original_work_peaks = Vec::new();
            let mut original_work_carriers = 0usize;
            let mut literal_original_writes = 0usize;
            let mut all_box_writes = 0usize;
            for fire_box in &stats.boxes {
                let mut counts = vec![0usize; shuffled.num_wires];
                for gate in &shuffled.gates[fire_box.output_start..fire_box.output_end] {
                    counts[gate.target as usize] += 1;
                }
                let recorded: Vec<_> = counts
                    .iter()
                    .enumerate()
                    .filter_map(|(wire, &count)| (count > 0).then_some((wire as u16, count)))
                    .collect();
                assert_eq!(fire_box.all_target_counts, recorded);
                let total = fire_box.output_end - fire_box.output_start;
                assert_eq!(total, fire_box.original_gates + fire_box.added_gates);
                peaks.push(*counts.iter().max().unwrap() as f64 / total as f64);
                let mut baseline_counts = vec![0usize; shuffled.num_wires];
                for gate in &baseline.gates[fire_box.source_start..fire_box.source_end] {
                    baseline_counts[gate.target as usize] += 1;
                }
                baseline_peaks.push(
                    *baseline_counts.iter().max().unwrap() as f64 / fire_box.original_gates as f64,
                );
                literal_original_writes += counts[fire_box.target as usize];
                all_box_writes += total;
                // Count only descendant writes of the original active role to
                // rule out improvements caused solely by extra routing gates.
                let source_writes = baseline.gates[fire_box.source_start..fire_box.source_end]
                    .iter()
                    .filter(|gate| gate.target == fire_box.target)
                    .count();
                assert_eq!(
                    fire_box
                        .original_target_counts
                        .iter()
                        .map(|&(_, count)| count)
                        .sum::<usize>(),
                    source_writes
                );
                // A narrow or fully occupied band may legitimately have no
                // legal escort at any cut; do not require every box to move.
                assert!(!fire_box.original_target_counts.is_empty());
                original_work_peaks.push(
                    fire_box
                        .original_target_counts
                        .iter()
                        .map(|&(_, count)| count)
                        .max()
                        .unwrap() as f64
                        / source_writes as f64,
                );
                original_work_carriers += fire_box.original_target_counts.len();
            }
            let mean_peak = peaks.iter().sum::<f64>() / peaks.len() as f64;
            let baseline_mean_peak =
                baseline_peaks.iter().sum::<f64>() / baseline_peaks.len() as f64;
            let worst_peak = peaks.iter().copied().fold(0.0_f64, f64::max);
            let overhead = stats.added_gates as f64 / stats.original_gates as f64;
            let original_work_mean_peak =
                original_work_peaks.iter().sum::<f64>() / original_work_peaks.len() as f64;
            eprintln!(
                "preprocessing shuffling n32/K64 seed={seed} home={return_home}: mean_peak={:.4}% worst_peak={:.4}% literal_original_share={:.4}% original_work_mean_peak={:.4}% original_work_mean_carriers={:.4} overhead={:.4}% original_gates={} shuffled_gates={} transfers={} skipped={}",
                mean_peak * 100.0,
                worst_peak * 100.0,
                100.0 * literal_original_writes as f64 / all_box_writes as f64,
                original_work_mean_peak * 100.0,
                original_work_carriers as f64 / stats.boxes.len() as f64,
                overhead * 100.0,
                stats.original_gates,
                stats.shuffled_gates,
                stats.transfer_count,
                stats.skipped_cut_count
            );
            assert!(
                mean_peak < baseline_mean_peak,
                "mean active-write concentration={mean_peak} vs baseline={baseline_mean_peak}: seed={seed} home={return_home}"
            );
            assert!(original_work_carriers > stats.boxes.len());
            assert!(
                overhead <= 0.12,
                "gate overhead={overhead}: seed={seed} home={return_home}"
            );
        }
    }
}
