use local_mixing::circuit::circuit::{CircuitSeq, Permutation, Polynomial, canonicalize_polys_4};
use local_mixing::circuit::poly_canon_graph::{canonical_form, canonicalize_graph};

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

fn circuits_are_equivalent(left: &CircuitSeq, right: &CircuitSeq, n: usize) -> bool {
    (0..(1usize << n)).all(|input| left.evaluate(input) == right.evaluate(input))
}

#[test]
fn compare_polycanon_and_canon4_on_simple_rewiring() {
    let n = 9;
    let original = CircuitSeq {
        gates: vec![[0, 1, 2], [3, 4, 5], [6, 7, 8]],
    };
    let global_rewire = Permutation::new(vec![2, 3, 4, 5, 6, 7, 8, 0, 1]);
    let mut rewired = original.clone();
    rewired.rewire(&global_rewire, n);
    assert_eq!(rewired.gates, vec![[2, 3, 4], [5, 6, 7], [8, 0, 1]]);

    let original_polys = original.to_polynomial(n, 0, original.gates.len());
    let rewired_polys = rewired.to_polynomial(n, 0, rewired.gates.len());

    let poly_original_perm = canonicalize_graph(&original_polys, n);
    let poly_rewired_perm = canonicalize_graph(&rewired_polys, n);
    let poly_original_form = canonical_form(&original_polys, &poly_original_perm);
    let poly_rewired_form = canonical_form(&rewired_polys, &poly_rewired_perm);
    let mut poly_original_circuit = original.clone();
    let mut poly_rewired_circuit = rewired.clone();
    poly_original_circuit.rewire(&poly_original_perm, n);
    poly_rewired_circuit.rewire(&poly_rewired_perm, n);

    let (canon4_original_form, canon4_original_perm) =
        canonicalize_polys_4(original_polys, true).unwrap();
    let (canon4_rewired_form, canon4_rewired_perm) =
        canonicalize_polys_4(rewired_polys, true).unwrap();
    let mut canon4_original_circuit = original.clone();
    let mut canon4_rewired_circuit = rewired.clone();
    canon4_original_circuit.rewire(&canon4_original_perm.invert(), n);
    canon4_rewired_circuit.rewire(&canon4_rewired_perm.invert(), n);

    println!("input original circuit: {:?}", original.gates);
    println!("input rewired circuit:  {:?}", rewired.gates);
    println!();
    println!("polycanon original perm: {:?}", poly_original_perm.data);
    println!("polycanon rewired perm:  {:?}", poly_rewired_perm.data);
    println!(
        "polycanon original form: {:?}",
        sorted_form(&poly_original_form)
    );
    println!(
        "polycanon rewired form:  {:?}",
        sorted_form(&poly_rewired_form)
    );
    println!(
        "polycanon original circuit: {:?}",
        poly_original_circuit.gates
    );
    println!(
        "polycanon rewired circuit:  {:?}",
        poly_rewired_circuit.gates
    );
    println!(
        "polycanon polynomial forms equal: {}",
        poly_original_form == poly_rewired_form
    );
    println!(
        "polycanon rewired circuits equal: {}",
        poly_original_circuit == poly_rewired_circuit
    );
    println!(
        "polycanon circuits functionally equal: {}",
        circuits_are_equivalent(&poly_original_circuit, &poly_rewired_circuit, n)
    );
    println!();
    println!("canon4 original perm: {:?}", canon4_original_perm.data);
    println!("canon4 rewired perm:  {:?}", canon4_rewired_perm.data);
    println!(
        "canon4 original form: {:?}",
        sorted_form(&canon4_original_form)
    );
    println!(
        "canon4 rewired form:  {:?}",
        sorted_form(&canon4_rewired_form)
    );
    println!(
        "canon4 original circuit: {:?}",
        canon4_original_circuit.gates
    );
    println!(
        "canon4 rewired circuit:  {:?}",
        canon4_rewired_circuit.gates
    );
    println!(
        "canon4 polynomial forms equal: {}",
        canon4_original_form == canon4_rewired_form
    );
    println!(
        "canon4 rewired circuits equal: {}",
        canon4_original_circuit == canon4_rewired_circuit
    );
    println!(
        "canon4 circuits functionally equal: {}",
        circuits_are_equivalent(&canon4_original_circuit, &canon4_rewired_circuit, n)
    );

    assert_eq!(canon4_original_form, canon4_rewired_form);
    assert!(circuits_are_equivalent(
        &canon4_original_circuit,
        &canon4_rewired_circuit,
        n
    ));
}
