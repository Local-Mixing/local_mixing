use super::*;
use crate::circuit::eval_lanes;
use rand::SeedableRng;
use rand::rngs::StdRng;

#[test]
fn commuting_shuffle_preserves_truth_table_and_sidecar_order() {
    let mixed = vec![
        XGate::from_g57([0, 1, 2]),
        XGate::conj(3, [(4, false), (5, true)]).unwrap(),
        XGate::conj(1, [(0, true)]).unwrap(),
        XGate::from_g57([5, 3, 4]),
        XGate::conj(2, []).unwrap(),
        XGate::from_g57([4, 0, 1]),
        XGate::conj(3, [(2, true), (5, false)]).unwrap(),
        XGate::from_g57([0, 1, 2]),
    ];
    let inputs: Vec<u64> = (0..6)
        .map(|wire| (0..64).fold(0, |column, lane| column | (((lane >> wire) & 1) << lane)))
        .collect();
    for source in [Vec::new(), vec![mixed[0].clone()], mixed] {
        let mut expected = inputs.clone();
        eval_lanes(&source, &mut expected);
        for seed in [0, 1, 42, 0x5eed, u64::MAX] {
            let mut shuffled = source.clone();
            let order = commuting_shuffle_order(&mut shuffled, &mut StdRng::seed_from_u64(seed));
            let mut sorted_order = order.clone();
            sorted_order.sort_unstable();
            assert_eq!(sorted_order, (0..source.len() as u32).collect::<Vec<_>>());
            for (gate, &source_index) in shuffled.iter().zip(&order) {
                assert_eq!(
                    gate, &source[source_index as usize],
                    "sidecar must follow its gate"
                );
            }
            let mut actual = inputs.clone();
            eval_lanes(&shuffled, &mut actual);
            assert_eq!(
                actual, expected,
                "shuffle changed a truth table at seed {seed}"
            );
        }
    }
}

fn bitmap_reference(n: usize, m: usize, rng: &mut fastrand::Rng) -> CircuitSeq {
    let mut circuit = Vec::with_capacity(m);
    for _ in 0..m {
        loop {
            let mut set = vec![false; n];
            let mut gate = [0u16; 3];
            for pin in &mut gate {
                loop {
                    let v = rng.usize(..n);
                    if !set[v] {
                        set[v] = true;
                        *pin = v as u16;
                        break;
                    }
                }
            }
            if circuit.last() != Some(&gate) {
                circuit.push(gate);
                break;
            }
        }
    }
    CircuitSeq { gates: circuit }
}

#[test]
fn pin_scan_preserves_bitmap_selection_and_rng_consumption() {
    for (n, m, seed) in [(3, 200, 1), (4, 1_000, 0x1234), (257, 500, u64::MAX)] {
        let mut reference_rng = fastrand::Rng::with_seed(seed);
        let expected = bitmap_reference(n, m, &mut reference_rng);

        let mut optimized_rng = fastrand::Rng::with_seed(seed);
        let actual = random_circuit_with_draw(n, m, |upper| optimized_rng.usize(..upper));

        assert_eq!(actual.gates, expected.gates, "n={n} m={m} seed={seed}");
        assert_eq!(
            optimized_rng.u64(..),
            reference_rng.u64(..),
            "optimized selection must consume exactly the same random draws"
        );
    }
}
