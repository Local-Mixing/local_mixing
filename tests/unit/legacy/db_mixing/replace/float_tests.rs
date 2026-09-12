use super::*;
use crate::circuit::{CircuitSeq, Gate};
use rand::SeedableRng;
use rand::rngs::StdRng;

// float_all_gates must preserve the circuit's function and keep each gate's tag attached
// to that gate through every slide.
#[test]
fn float_preserves_function_and_tag_pairing() {
    for seed in 0..40u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let n = 12u16;
        let mut gates: Vec<[u16; 3]> = Vec::new();
        // Distinct tag per gate via a unique low bit pattern we can track: store index in Tag.0.
        for k in 0..60u64 {
            use rand::Rng;
            let a = rng.random_range(0..n);
            let b = rng.random_range(0..n);
            let c = rng.random_range(0..n);
            if a != b && a != c && b != c {
                gates.push([a, b, c]);
                let _ = k;
            }
        }
        let mut tags: Vec<Tag> = (0..gates.len() as u64).map(Tag).collect();
        let before = CircuitSeq {
            gates: gates.clone(),
        };
        // Map: which gate does each tag sit on (by gate content is ambiguous; track by
        // pairing invariant — tag i must always sit on the gate originally at index i).
        let orig: Vec<[u16; 3]> = gates.clone();
        float_all_gates(&mut gates, &mut tags, &mut rng);
        let after = CircuitSeq {
            gates: gates.clone(),
        };
        assert!(
            before.probably_equal(&after, n as usize, 400).is_ok(),
            "float changed function (seed {})",
            seed
        );
        // Tag/gate pairing: the gate now carrying tag t must equal orig[t].
        for (pos, t) in tags.iter().enumerate() {
            assert_eq!(
                gates[pos], orig[t.0 as usize],
                "tag/gate desync (seed {})",
                seed
            );
        }
        // No collision was crossed: relative order of any two colliding gates is preserved.
        for x in 0..orig.len() {
            for y in (x + 1)..orig.len() {
                if Gate::collides_index(&orig[x], &orig[y]) {
                    let px = tags.iter().position(|t| t.0 == x as u64).unwrap();
                    let py = tags.iter().position(|t| t.0 == y as u64).unwrap();
                    assert!(px < py, "colliding pair reordered (seed {})", seed);
                }
            }
        }
    }
}
