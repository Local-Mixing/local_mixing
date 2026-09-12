use super::*;
use rand::{SeedableRng, rngs::StdRng};
use std::collections::HashSet;

#[test]
fn transformed_small_circuit_is_a_permutation_with_nonconstant_garbage() {
    let n = 3;
    let mask = (1usize << n) - 1;
    let main = CircuitSeq {
        gates: vec![[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 2, 1]],
    };
    for seed in 0x400u64..0x408 {
        let mut rng = StdRng::seed_from_u64(seed);
        let transformed = feistalize(&main, n, 1, &mut rng);
        let outputs: Vec<usize> = (0..512).map(|input| transformed.evaluate(input)).collect();
        assert_eq!(outputs.iter().copied().collect::<HashSet<_>>().len(), 512);
        assert!(
            outputs
                .iter()
                .map(|v| v & mask)
                .collect::<HashSet<_>>()
                .len()
                > 1
        );
        assert!(
            outputs
                .iter()
                .map(|v| (v >> (2 * n)) & mask)
                .collect::<HashSet<_>>()
                .len()
                > 1
        );
    }
}
