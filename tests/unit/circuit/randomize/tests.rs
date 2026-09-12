use super::*;

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
