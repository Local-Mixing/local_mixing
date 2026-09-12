use super::*;
use crate::circuit::Gate;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::collections::HashSet;

fn canonical_state() -> FeistalState {
    FeistalState {
        sharing: GadgetState {
            n: 3,
            pairs: vec![(0, 1), (3, 4), (6, 7)],
        },
        free: vec![2, 5, 8],
        q: vec![1, 2, 0],
    }
}

fn virtual_values(state: &FeistalState, physical: usize) -> (usize, usize) {
    let mut x = 0usize;
    let mut y = 0usize;
    for i in 0..state.sharing.n {
        let (p0, p1) = state.sharing.pairs[i];
        let pair = ((physical >> p0) & 1) ^ ((physical >> p1) & 1);
        let x_bit = pair ^ ((physical >> state.free[i]) & 1);
        x |= x_bit << i;
        y |= pair << state.q[i];
    }
    (x, y)
}

fn evaluate_gates(input: usize, gates: &[[u16; 3]]) -> usize {
    Gate::evaluate_index_list(input, &gates.to_vec())
}

fn deterministic_circuit(n: usize, m: usize, seed: u64) -> CircuitSeq {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut gates = Vec::with_capacity(m);
    for _ in 0..m {
        let active = rng.random_range(0..n) as u16;
        let pos = loop {
            let wire = rng.random_range(0..n) as u16;
            if wire != active {
                break wire;
            }
        };
        let neg = loop {
            let wire = rng.random_range(0..n) as u16;
            if wire != active && wire != pos {
                break wire;
            }
        };
        gates.push([active, pos, neg]);
    }
    CircuitSeq { gates }
}

fn packed_words_to_usize(words: &[u64], n: usize) -> usize {
    let mut out = 0usize;
    for bit in 0..n {
        if packed_bit(words, bit) {
            out |= 1usize << bit;
        }
    }
    out
}

#[test]
fn each_rg_variant_preserves_overlapping_x_and_y_values() {
    for variant in 0..3 {
        let mut state = canonical_state();
        let mut gates = Vec::new();
        match variant {
            0 => emit_rg1(&mut state.sharing, 0, 1, &mut gates),
            1 => emit_rg2(&mut state.sharing, 0, 1, &mut gates),
            2 => emit_rg3(&state.sharing, 0, 3, 6, &mut gates),
            _ => unreachable!(),
        }
        for input in 0..512usize {
            let before = virtual_values(&canonical_state(), input);
            let output = evaluate_gates(input, &gates);
            assert_eq!(virtual_values(&state, output), before, "RG{}", variant + 1);
        }
    }
}

#[test]
fn n_tilde_updates_y_by_x_and_preserves_x() {
    let state = canonical_state();
    let mut gates = Vec::new();
    emit_feistal_n(&state, &mut gates);
    for input in 0..512usize {
        let (x, y) = virtual_values(&state, input);
        let output = evaluate_gates(input, &gates);
        assert_eq!(virtual_values(&state, output), (x, y ^ x));
    }
}

#[test]
fn sg3_preserves_all_y_values() {
    let state = canonical_state();
    for gate in [[0, 1, 2], [1, 2, 0], [2, 0, 1]] {
        let mut gates = Vec::new();
        emit_sg3(&state, gate, &mut gates);
        for input in 0..512usize {
            let (x, y) = virtual_values(&state, input);
            let output = evaluate_gates(input, &gates);
            let expected_x = Gate::evaluate_index(x, gate);
            assert_eq!(virtual_values(&state, output), (expected_x, y));
        }
    }
}

#[test]
fn feistalize_end_to_end_matrix() {
    let cases = [
        (3usize, 0usize, 0x100u64),
        (3, 1, 0x101),
        (3, 7, 0x102),
        (4, 3, 0x103),
        (4, 12, 0x104),
        (5, 20, 0x105),
    ];
    for (n, m, circuit_seed) in cases {
        let main = deterministic_circuit(n, m, circuit_seed);
        let mask = (1usize << n) - 1;
        for rg_freq in [1usize, 2, 3, m.max(1) + 1] {
            for layout_seed in [0x200u64, 0x201, 0x202, 0x203] {
                let mut rng = StdRng::seed_from_u64(layout_seed ^ circuit_seed);
                let transformed = feistalize(&main, n, rg_freq, &mut rng);
                assert!(
                    transformed
                        .gates
                        .iter()
                        .flatten()
                        .all(|&w| (w as usize) < 3 * n)
                );

                let inputs: Vec<usize> = if n == 3 {
                    (0..(1usize << (3 * n))).collect()
                } else {
                    let mut sample_rng =
                        StdRng::seed_from_u64(layout_seed ^ circuit_seed ^ rg_freq as u64);
                    let mut values = vec![0, mask, mask << n, mask << (2 * n)];
                    for x in 0..=mask {
                        values.push(x);
                        values.push(x | (mask << (2 * n)));
                    }
                    for _ in 0..256 {
                        values.push(sample_rng.random_range(0..(1usize << (3 * n))));
                    }
                    values
                };

                for input in inputs {
                    let x = input & mask;
                    let y = (input >> n) & mask;
                    let output = transformed.evaluate(input);
                    assert_eq!(
                        (output >> n) & mask,
                        y ^ main.evaluate(x),
                        "n={n} m={m} rg={rg_freq} seed={layout_seed:#x} input={input:#x}",
                    );
                }
            }
        }
    }
}

#[test]
fn zero_initialized_middle_block_is_exactly_cx() {
    let n = 5;
    let mask = (1usize << n) - 1;
    for circuit_seed in 0x300u64..0x308 {
        let main = deterministic_circuit(n, 15, circuit_seed);
        let mut rng = StdRng::seed_from_u64(circuit_seed ^ 0x57);
        let transformed = feistalize(&main, n, 2, &mut rng);
        for x in 0..=mask {
            for z in [0usize, 1, mask / 2, mask] {
                let input = x | (z << (2 * n));
                assert_eq!((transformed.evaluate(input) >> n) & mask, main.evaluate(x));
            }
        }
    }
}

#[test]
fn slice_zero_preblock_fixes_exactly_the_zero_slice() {
    let n = 3;
    let mask = (1usize << n) - 1;
    for seed in 0x5a10u64..0x5a18 {
        let mut rng = StdRng::seed_from_u64(seed);
        let block = slice_zero_preblock(n, &mut rng);
        assert!(block.gates.iter().flatten().all(|&w| (w as usize) < 3 * n));

        let mut outputs = HashSet::new();
        for input in 0..(1usize << (3 * n)) {
            let x = input & mask;
            let y = (input >> n) & mask;
            let z = (input >> (2 * n)) & mask;
            let output = block.evaluate(input);
            outputs.insert(output);

            assert_eq!((output >> n) & mask, y);
            assert_eq!((output >> (2 * n)) & mask, z);
            if y == 0 && z == 0 {
                assert_eq!(output & mask, x);
            } else {
                assert_ne!(output & mask, x, "seed={seed:#x} input={input:#x}");
            }
        }
        assert_eq!(outputs.len(), 1usize << (3 * n));
    }
}

#[test]
fn slice_zero_hardcoded_preblock_fixes_zero_slice() {
    let n = 4;
    let mask = (1usize << n) - 1;
    for seed in 0x5b10u64..0x5b18 {
        let mut rng = StdRng::seed_from_u64(seed);
        let block = slice_zero_hardcoded_preblock(n, 2, &mut rng);
        assert!(block.gates.iter().flatten().all(|&w| (w as usize) < 3 * n));
        assert!(!block.gates.is_empty());

        for x in 0..=mask {
            let output = block.evaluate(x);
            assert_eq!(output & mask, x, "seed={seed:#x} x={x:#x}");
            assert_eq!((output >> n) & mask, 0);
            assert_eq!((output >> (2 * n)) & mask, 0);
        }

        let mut moved = false;
        for y in 1..=mask {
            let output = block.evaluate(y << n);
            moved |= output != (y << n);
        }
        assert!(
            moved,
            "hardcoded M should change at least one off-slice input"
        );
    }
}

#[test]
fn slice_zero_feistalize_matches_original_only_on_zero_slice() {
    let n = 3;
    let mask = (1usize << n) - 1;
    let main = CircuitSeq {
        gates: vec![[0, 1, 2], [2, 0, 1], [1, 2, 0]],
    };

    for seed in 0x5f00u64..0x5f08 {
        let mut rng = StdRng::seed_from_u64(seed);
        let transformed = feistalize_with_slice_zero(&main, n, 2, &mut rng);
        for input in 0..(1usize << (3 * n)) {
            let x = input & mask;
            let y = (input >> n) & mask;
            let z = (input >> (2 * n)) & mask;
            let middle = (transformed.evaluate(input) >> n) & mask;
            let old_middle = y ^ main.evaluate(x);
            if y == 0 && z == 0 {
                assert_eq!(middle, main.evaluate(x));
            } else {
                assert_ne!(
                    middle, old_middle,
                    "seed={seed:#x} input={input:#x} x={x:#x} y={y:#x} z={z:#x}"
                );
            }
        }
    }
}

#[test]
fn slice_zero_hardcoded_feistalize_matches_original_on_zero_slice() {
    let n = 3;
    let mask = (1usize << n) - 1;
    let main = CircuitSeq {
        gates: vec![[0, 1, 2], [2, 0, 1], [1, 2, 0]],
    };

    for seed in 0x5c00u64..0x5c08 {
        let mut rng = StdRng::seed_from_u64(seed);
        let transformed = feistalize_with_slice_zero_hardcoded(&main, n, 2, 1, &mut rng);
        for x in 0..=mask {
            let middle = (transformed.evaluate(x) >> n) & mask;
            assert_eq!(middle, main.evaluate(x), "seed={seed:#x} x={x:#x}");
        }
    }
}

#[test]
fn slice_zero_random_preblock_fixes_public_slice() {
    let n = 4;
    let mask = (1usize << n) - 1;
    for seed in 0x6100u64..0x6108 {
        let mut rng = StdRng::seed_from_u64(seed);
        let block = slice_zero_random_preblock(n, 256, &mut rng);
        let public_y = packed_words_to_usize(&block.public_y, n);
        let public_z = packed_words_to_usize(&block.public_z, n);

        assert!(
            block
                .circuit
                .gates
                .iter()
                .flatten()
                .all(|&w| (w as usize) < 3 * n)
        );
        for x in 0..=mask {
            let input = x | (public_y << n) | (public_z << (2 * n));
            let output = block.circuit.evaluate(input);
            assert_eq!(output & mask, x, "seed={seed:#x} x={x:#x}");
            assert_eq!((output >> n) & mask, public_y);
            assert_eq!((output >> (2 * n)) & mask, public_z);
        }
    }
}

#[test]
fn slice_zero_random_feistalize_matches_original_on_public_slice() {
    let n = 3;
    let mask = (1usize << n) - 1;
    let main = CircuitSeq {
        gates: vec![[0, 1, 2], [2, 0, 1], [1, 2, 0]],
    };

    for seed in 0x6200u64..0x6208 {
        let mut rng = StdRng::seed_from_u64(seed);
        let transformed = feistalize_with_slice_zero_random(&main, n, 2, 128, &mut rng);
        let public_y = packed_words_to_usize(&transformed.public_y, n);
        let public_z = packed_words_to_usize(&transformed.public_z, n);

        for x in 0..=mask {
            let input = x | (public_y << n) | (public_z << (2 * n));
            let middle = (transformed.circuit.evaluate(input) >> n) & mask;
            assert_eq!(
                middle,
                public_y ^ main.evaluate(x),
                "seed={seed:#x} x={x:#x}"
            );
        }
    }
}
