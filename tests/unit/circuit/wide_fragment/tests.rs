use super::super::xgate::eval_lanes;
use super::*;
use rand::RngCore;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn legacy_available_order(gate: &XGate, num_wires: usize, rng: &mut impl Rng) -> Vec<u16> {
    let mut unavailable = vec![gate.target as usize];
    unavailable.extend(gate.ctrls.iter().map(|&(wire, _)| wire as usize));
    let mut available: Vec<u16> = (0..num_wires)
        .filter(|wire| !unavailable.contains(wire))
        .map(|wire| wire as u16)
        .collect();
    available.shuffle(rng);
    available
}

#[test]
fn post_fragment_restores_arbitrary_dirty_helpers_and_matches_wide_gate() {
    for style in [FragmentStyle::Exact, FragmentStyle::NativeDeep] {
        for width in 3..=8usize {
            for comp in [false, true] {
                for polarity_seed in 0..16u64 {
                    let total = 2 * width + 3;
                    let target = 0u16;
                    let controls: Vec<(u16, bool)> = (0..width)
                        .map(|i| ((i + 1) as u16, (polarity_seed >> (i % 4)) & 1 != 0))
                        .collect();
                    let original = vec![XGate {
                        target,
                        comp,
                        ctrls: controls.iter().copied().collect(),
                    }];
                    let mut fragmented = original.clone();
                    let mut rng =
                        StdRng::seed_from_u64(0xF12A_6E17 ^ polarity_seed ^ ((width as u64) << 16));
                    let stats = fragment_wide_post_shuffle(&mut fragmented, total, style, &mut rng)
                        .expect("ample dirty helpers");
                    assert_eq!(stats.fragmented_gates, 1);
                    assert!(fragmented.iter().all(|gate| gate.width() <= 2));

                    for sample in 0..8u64 {
                        let mut lanes: Vec<u64> = (0..total).map(|_| rng.next_u64()).collect();
                        // Include structured all-zero/all-one lanes among
                        // the arbitrary dirty helper samples.
                        lanes[0] ^= sample.wrapping_mul(!0u64 / 7);
                        let mut want = lanes.clone();
                        let mut got = lanes.clone();
                        eval_lanes(original.iter(), &mut want);
                        eval_lanes(fragmented.iter(), &mut got);
                        assert_eq!(
                            got, want,
                            "style={style:?} width={width} comp={comp} polarity={polarity_seed}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn reused_membership_storage_preserves_legacy_helper_selection() {
    let gates = vec![
        XGate::conj(3, [(0, true), (2, false), (7, true), (11, false)]).unwrap(),
        XGate::conj(
            14,
            [(1, false), (5, true), (8, true), (13, false), (19, true)],
        )
        .unwrap(),
    ];
    let num_wires = 24;
    let seed = 0x6865_6c70_6572_7331;
    let mut legacy_rng = StdRng::seed_from_u64(seed);
    let legacy_orders: Vec<Vec<u16>> = gates
        .iter()
        .map(|gate| legacy_available_order(gate, num_wires, &mut legacy_rng))
        .collect();

    let mut unavailable = vec![false; num_wires];
    let mut available = Vec::with_capacity(num_wires);
    let mut reused_rng = StdRng::seed_from_u64(seed);
    for (gate, legacy) in gates.iter().zip(legacy_orders) {
        collect_available_wires(gate, num_wires, &mut unavailable, &mut available);
        available.shuffle(&mut reused_rng);
        assert_eq!(available, legacy);
        clear_unavailable_wires(gate, num_wires, &mut unavailable);
    }
    assert_eq!(reused_rng.next_u64(), legacy_rng.next_u64());
}
