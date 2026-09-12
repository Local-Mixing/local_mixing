use super::*;
use crate::circuit::Gate;
use primitive_types::U256;
use rand::{Rng, SeedableRng, rngs::StdRng};

fn circuit_32(seed: u64, gates: usize) -> CircuitSeq {
    let mut rng = StdRng::seed_from_u64(seed);
    random_circuit_with_rng(32, gates, &mut rng)
}

fn assert_middle_block(
    transformed: &CircuitSeq,
    original: &CircuitSeq,
    x: u32,
    y: u32,
    z: u32,
    context: &str,
) {
    let input = U256::from(x) | (U256::from(y) << 32) | (U256::from(z) << 64);
    let output = Gate::evaluate_index_list_256(input, &transformed.gates);
    let cx = Gate::evaluate_index_list_256(U256::from(x), &original.gates);
    let middle = ((output >> 32) & U256::from(u32::MAX)).low_u32();
    assert_eq!(
        middle,
        y ^ cx.low_u32(),
        "{context}: x={x:#010x} y={y:#010x} z={z:#010x}"
    );
}

fn middle_block(transformed: &CircuitSeq, x: u32, y: u32, z: u32) -> u32 {
    let input = U256::from(x) | (U256::from(y) << 32) | (U256::from(z) << 64);
    let output = Gate::evaluate_index_list_256(input, &transformed.gates);
    ((output >> 32) & U256::from(u32::MAX)).low_u32()
}

fn packed_words_to_u32(words: &[u64]) -> u32 {
    let mut out = 0u32;
    for bit in 0..32 {
        if packed_bit(words, bit) {
            out |= 1u32 << bit;
        }
    }
    out
}

fn eval_x_block(circuit: &CircuitSeq, x: u32, y: u32, z: u32) -> u32 {
    let input = U256::from(x) | (U256::from(y) << 32) | (U256::from(z) << 64);
    let output = Gate::evaluate_index_list_256(input, &circuit.gates);
    (output & U256::from(u32::MAX)).low_u32()
}

#[test]
fn thirty_two_wire_middle_block_is_y_plus_cx_for_many_inputs() {
    let patterns = [
        0u32,
        1,
        u32::MAX,
        0xaaaa_aaaa,
        0x5555_5555,
        0x8000_0000,
        0x7fff_ffff,
        0x0123_4567,
        0x89ab_cdef,
    ];

    for circuit_seed in [0x3200u64, 0x3201, 0x3202, 0x3203] {
        let original = circuit_32(circuit_seed, 96);
        for rg_freq in [1usize, 2, 5, 17] {
            let layout_seed = circuit_seed ^ ((rg_freq as u64) << 40) ^ 0xfe15_7a;
            let mut layout_rng = StdRng::seed_from_u64(layout_seed);
            let transformed = feistalize(&original, 32, rg_freq, &mut layout_rng);
            let context = format!("circuit_seed={circuit_seed:#x} rg_freq={rg_freq}");

            for &x in &patterns {
                for &y in &patterns {
                    for &z in &[0u32, u32::MAX, 0xa5a5_5a5a] {
                        assert_middle_block(&transformed, &original, x, y, z, &context);
                    }
                }
            }

            let mut input_rng = StdRng::seed_from_u64(layout_seed ^ 0x1a2b_3c4d);
            for _ in 0..512 {
                assert_middle_block(
                    &transformed,
                    &original,
                    input_rng.random::<u32>(),
                    input_rng.random::<u32>(),
                    input_rng.random::<u32>(),
                    &context,
                );
            }
        }
    }
}

#[test]
fn slice_zero_thirty_two_wire_zero_slice_matches_and_off_slice_changes() {
    let patterns = [
        0u32,
        1,
        u32::MAX,
        0xaaaa_aaaa,
        0x5555_5555,
        0x8000_0000,
        0x0123_4567,
        0x89ab_cdef,
    ];

    for circuit_seed in [0x4200u64, 0x4201] {
        let original = circuit_32(circuit_seed, 80);
        let mut layout_rng = StdRng::seed_from_u64(circuit_seed ^ 0x510c_e0);
        let transformed = feistalize_with_slice_zero(&original, 32, 3, &mut layout_rng);

        for &x in &patterns {
            let cx = Gate::evaluate_index_list_256(U256::from(x), &original.gates).low_u32();
            assert_eq!(middle_block(&transformed, x, 0, 0), cx);
            for &(y, z) in &[(1u32, 0u32), (0, 1), (0xa5a5_5a5a, 0), (0, 0x5a5a_a5a5)] {
                assert_ne!(middle_block(&transformed, x, y, z), y ^ cx);
            }
        }

        let mut input_rng = StdRng::seed_from_u64(circuit_seed ^ 0x7123);
        for _ in 0..128 {
            let x = input_rng.random::<u32>();
            let y = input_rng.random::<u32>();
            let z = input_rng.random::<u32>() | 1;
            let cx = Gate::evaluate_index_list_256(U256::from(x), &original.gates).low_u32();
            assert_ne!(middle_block(&transformed, x, y, z), y ^ cx);
        }
    }
}

#[test]
fn slice_zero_random_thirty_two_wire_public_slice_fixed_and_off_slice_moves_x() {
    for seed in [0x6300u64, 0x6301, 0x6302, 0x6303] {
        let mut rng = StdRng::seed_from_u64(seed);
        let block = slice_zero_random_preblock(32, SLICE_ZERO_RANDOM_GATES_PER_WIRE * 32, &mut rng);
        let public_y = packed_words_to_u32(&block.public_y);
        let public_z = packed_words_to_u32(&block.public_z);

        for &x in &[0u32, 1, u32::MAX, 0xaaaa_aaaa, 0x0123_4567] {
            assert_eq!(eval_x_block(&block.circuit, x, public_y, public_z), x);
        }

        for bit in 0..32 {
            let y_delta = eval_x_block(&block.circuit, 0, public_y ^ (1u32 << bit), public_z);
            let z_delta = eval_x_block(&block.circuit, 0, public_y, public_z ^ (1u32 << bit));
            assert!(
                y_delta.count_ones() >= 4,
                "seed={seed:#x} y bit={bit} delta={y_delta:#010x}"
            );
            assert!(
                z_delta.count_ones() >= 4,
                "seed={seed:#x} z bit={bit} delta={z_delta:#010x}"
            );
        }
    }
}
