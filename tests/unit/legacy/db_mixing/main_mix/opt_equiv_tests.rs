use super::*;
use rand::{SeedableRng, rngs::StdRng};

fn random_circuit(num_wires: u16, num_gates: usize, rng: &mut StdRng) -> CircuitSeq {
    let mut gates = Vec::with_capacity(num_gates);
    while gates.len() < num_gates {
        let a = rng.random_range(0..num_wires);
        let b = rng.random_range(0..num_wires);
        let c = rng.random_range(0..num_wires);
        if a != b && a != c && b != c {
            gates.push([a, b, c]);
        }
    }
    CircuitSeq { gates }
}

// The width-dispatched middle-block probe must be bit-identical to the original U1024
// computation for every kernel width (128/256/512/1024), including the all-zero input.
#[test]
fn opt_equiv_feistal_middle_dispatch_matches_1024() {
    let mut rng = StdRng::seed_from_u64(0xfe15_7a1d);
    for &total_wires in &[24usize, 120, 240, 450, 1020] {
        let original_n = total_wires / 3;
        let original = random_circuit(original_n as u16, 40, &mut rng);
        let transformed = random_circuit(total_wires as u16, 150, &mut rng);
        let mask = (U1024::one() << original_n) - U1024::one();
        let eval_wires = total_wires
            .max(original.max_wire() + 1)
            .max(transformed.max_wire() + 1);
        for probe in 0..40 {
            let mut bytes = [0u8; 128];
            if probe > 0 {
                rng.fill_bytes(&mut bytes);
            }
            let random = U1024::from_little_endian(&bytes);
            let x = random & mask;
            let y = (random >> original_n) & mask;
            let z = (random >> (2 * original_n)) & mask;
            let extra = if total_wires > 3 * original_n {
                let extra_mask = (U1024::one() << (total_wires - 3 * original_n)) - U1024::one();
                (random >> (3 * original_n)) & extra_mask
            } else {
                U1024::zero()
            };
            let input =
                x | (y << original_n) | (z << (2 * original_n)) | (extra << (3 * original_n));
            assert_eq!(
                feistal_middle_matches_once(
                    &original,
                    &transformed,
                    original_n,
                    eval_wires,
                    x,
                    y,
                    input,
                    mask,
                ),
                feistal_middle_matches_once_1024(
                    &original,
                    &transformed,
                    original_n,
                    x,
                    y,
                    input,
                    mask,
                ),
                "width {} probe {}",
                total_wires,
                probe
            );
        }
    }
}
