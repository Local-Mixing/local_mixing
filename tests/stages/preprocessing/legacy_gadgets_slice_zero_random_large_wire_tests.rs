use super::*;
use crate::circuit::{Gate, U1024};
use primitive_types::U512;
use rand::{SeedableRng, rngs::StdRng};

fn packed_words_to_u512(words: &[u64], n: usize) -> U512 {
    let mut out = U512::zero();
    for bit in 0..n {
        if packed_bit(words, bit) {
            out |= U512::one() << bit;
        }
    }
    out
}

fn packed_words_to_u1024(words: &[u64], n: usize) -> U1024 {
    let mut out = U1024::zero();
    for bit in 0..n {
        if packed_bit(words, bit) {
            out = out | (U1024::one() << bit);
        }
    }
    out
}

fn pattern_u512(n: usize, mode: usize) -> U512 {
    let mut out = U512::zero();
    for bit in 0..n {
        let set = match mode {
            0 => false,
            1 => true,
            2 => bit % 2 == 0,
            _ => bit % 3 == 1,
        };
        if set {
            out |= U512::one() << bit;
        }
    }
    out
}

fn pattern_u1024(n: usize, mode: usize) -> U1024 {
    let mut out = U1024::zero();
    for bit in 0..n {
        let set = match mode {
            0 => false,
            1 => true,
            2 => bit % 2 == 0,
            _ => bit % 3 == 1,
        };
        if set {
            out = out | (U1024::one() << bit);
        }
    }
    out
}

fn low_weight_u512(value: U512, n: usize) -> u32 {
    let mut weight = 0;
    for bit in 0..n {
        if ((value >> bit) & U512::one()) == U512::one() {
            weight += 1;
        }
    }
    weight
}

fn low_weight_u1024(value: U1024, n: usize) -> u32 {
    let mut weight = 0;
    for bit in 0..n {
        if ((value >> bit) & U1024::one()) == U1024::one() {
            weight += 1;
        }
    }
    weight
}

#[test]
fn slice_zero_random_n128_default_32n_fixes_public_slice_and_moves_x() {
    let n = 128;
    let mut rng = StdRng::seed_from_u64(0x1280_32);
    let block = slice_zero_random_preblock(n, SLICE_ZERO_RANDOM_GATES_PER_WIRE * n, &mut rng);
    assert_eq!(block.circuit.gates.len(), 32 * n);

    let public_y = packed_words_to_u512(&block.public_y, n);
    let public_z = packed_words_to_u512(&block.public_z, n);
    let mask = (U512::one() << n) - U512::one();

    for mode in 0..4 {
        let x = pattern_u512(n, mode);
        let input = x | (public_y << n) | (public_z << (2 * n));
        let output = Gate::evaluate_index_list_512(input, &block.circuit.gates);
        assert_eq!(output & mask, x);
        assert_eq!((output >> n) & mask, public_y);
        assert_eq!((output >> (2 * n)) & mask, public_z);
    }

    for bit in 0..n {
        let y_input = (public_y ^ (U512::one() << bit)) << n;
        let z_input = (public_z ^ (U512::one() << bit)) << (2 * n);
        let y_delta =
            Gate::evaluate_index_list_512(y_input | (public_z << (2 * n)), &block.circuit.gates)
                & mask;
        let z_delta =
            Gate::evaluate_index_list_512((public_y << n) | z_input, &block.circuit.gates) & mask;
        assert!(
            low_weight_u512(y_delta, n) >= 4,
            "n=128 y bit={bit} delta_weight={}",
            low_weight_u512(y_delta, n)
        );
        assert!(
            low_weight_u512(z_delta, n) >= 4,
            "n=128 z bit={bit} delta_weight={}",
            low_weight_u512(z_delta, n)
        );
    }
}

#[test]
fn slice_zero_random_n256_default_32n_fixes_public_slice_and_moves_x() {
    let n = 256;
    let mut rng = StdRng::seed_from_u64(0x2560_32);
    let block = slice_zero_random_preblock(n, SLICE_ZERO_RANDOM_GATES_PER_WIRE * n, &mut rng);
    assert_eq!(block.circuit.gates.len(), 32 * n);

    let public_y = packed_words_to_u1024(&block.public_y, n);
    let public_z = packed_words_to_u1024(&block.public_z, n);
    let mask = (U1024::one() << n) - U1024::one();

    for mode in 0..4 {
        let x = pattern_u1024(n, mode);
        let input = x | (public_y << n) | (public_z << (2 * n));
        let output = Gate::evaluate_index_list_1024(input, &block.circuit.gates);
        assert_eq!(output & mask, x);
        assert_eq!((output >> n) & mask, public_y);
        assert_eq!((output >> (2 * n)) & mask, public_z);
    }

    for bit in 0..n {
        let y_input = (public_y ^ (U1024::one() << bit)) << n;
        let z_input = (public_z ^ (U1024::one() << bit)) << (2 * n);
        let y_delta =
            Gate::evaluate_index_list_1024(y_input | (public_z << (2 * n)), &block.circuit.gates)
                & mask;
        let z_delta =
            Gate::evaluate_index_list_1024((public_y << n) | z_input, &block.circuit.gates) & mask;
        assert!(
            low_weight_u1024(y_delta, n) >= 4,
            "n=256 y bit={bit} delta_weight={}",
            low_weight_u1024(y_delta, n)
        );
        assert!(
            low_weight_u1024(z_delta, n) >= 4,
            "n=256 z bit={bit} delta_weight={}",
            low_weight_u1024(z_delta, n)
        );
    }
}
