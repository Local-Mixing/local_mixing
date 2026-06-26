use local_mixing::circuit::circuit::{CircuitSeq, Permutation, Polynomial, canonicalize_polys_4};
use local_mixing::circuit::poly_canon_graph::{canonical_form, canonicalize_graph};
use local_mixing::random::random_data::random_circuit;
use std::collections::HashMap;

fn dense(circuit: &CircuitSeq) -> CircuitSeq {
    let used = circuit.used_wires();
    let wire_map: HashMap<_, _> = used
        .iter()
        .enumerate()
        .map(|(dense, &wire)| (wire, dense as u16))
        .collect();
    CircuitSeq {
        gates: circuit
            .gates
            .iter()
            .map(|&[target, control_a, control_b]| {
                [
                    wire_map[&target],
                    wire_map[&control_a],
                    wire_map[&control_b],
                ]
            })
            .collect(),
    }
}

fn sorted_form(polys: &[Polynomial]) -> Vec<Vec<u64>> {
    polys
        .iter()
        .map(|poly| {
            let mut monomials: Vec<_> = poly.iter().copied().collect();
            monomials.sort_unstable();
            monomials
        })
        .collect()
}

#[test]
fn polycanon_regression_case_21_is_rewire_invariant() {
    let n = 11;
    let seed = 179_275_629_541_752_411u64;
    let shuffle = Permutation::new(vec![4, 7, 3, 10, 6, 9, 0, 5, 8, 1, 2]);

    fastrand::seed(seed);
    let original = random_circuit(n, 2);
    let mut shuffled = original.clone();
    shuffled.rewire(&shuffle, n);

    let original_dense = dense(&original);
    let shuffled_dense = dense(&shuffled);
    let dense_n = original_dense.used_wires().len();
    assert_eq!(dense_n, shuffled_dense.used_wires().len());

    let original_polys = original_dense.to_polynomial(dense_n, 0, original_dense.gates.len());
    let shuffled_polys = shuffled_dense.to_polynomial(dense_n, 0, shuffled_dense.gates.len());
    let original_perm = canonicalize_graph(&original_polys, dense_n);
    let shuffled_perm = canonicalize_graph(&shuffled_polys, dense_n);
    let original_form = canonical_form(&original_polys, &original_perm);
    let shuffled_form = canonical_form(&shuffled_polys, &shuffled_perm);
    let (canon4_original_form, _) = canonicalize_polys_4(original_polys.clone(), true).unwrap();
    let (canon4_shuffled_form, _) = canonicalize_polys_4(shuffled_polys.clone(), true).unwrap();

    let mut original_canonical = original_dense.clone();
    let mut shuffled_canonical = shuffled_dense.clone();
    original_canonical.rewire(&original_perm, dense_n);
    shuffled_canonical.rewire(&shuffled_perm, dense_n);
    let functionally_equal = (0..(1usize << dense_n))
        .all(|input| original_canonical.evaluate(input) == shuffled_canonical.evaluate(input));

    println!("seed: {seed}");
    println!("shuffle: {:?}", shuffle.data);
    println!("original circuit: {:?}", original.gates);
    println!("shuffled circuit: {:?}", shuffled.gates);
    println!("dense original: {:?}", original_dense.gates);
    println!("dense shuffled: {:?}", shuffled_dense.gates);
    println!("polycanon original permutation: {:?}", original_perm.data);
    println!("polycanon shuffled permutation: {:?}", shuffled_perm.data);
    println!("polycanon original form: {:?}", sorted_form(&original_form));
    println!("polycanon shuffled form: {:?}", sorted_form(&shuffled_form));
    println!(
        "canon4 original form: {:?}",
        sorted_form(&canon4_original_form)
    );
    println!(
        "canon4 shuffled form: {:?}",
        sorted_form(&canon4_shuffled_form)
    );
    println!("polycanon original circuit: {:?}", original_canonical.gates);
    println!("polycanon shuffled circuit: {:?}", shuffled_canonical.gates);
    println!("forms equal: {}", original_form == shuffled_form);
    println!(
        "canon4 forms equal: {}",
        canon4_original_form == canon4_shuffled_form
    );
    println!("functionally equal: {functionally_equal}");

    assert_eq!(canon4_original_form, canon4_shuffled_form);
    assert_eq!(original_form, shuffled_form);
    assert!(functionally_equal);
}
