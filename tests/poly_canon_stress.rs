use local_mixing::circuit::circuit::{CircuitSeq, Permutation};
use local_mixing::circuit::poly_canon_graph::{canonical_form, canonicalize_graph};
use local_mixing::random::random_data::random_circuit;
use std::collections::HashMap;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn canonicalize_circuit(circuit: &CircuitSeq) -> (Vec<Vec<u64>>, CircuitSeq, usize) {
    let used = circuit.used_wires();
    let wire_map: HashMap<_, _> = used
        .iter()
        .enumerate()
        .map(|(dense, &wire)| (wire, dense as u16))
        .collect();
    let dense = CircuitSeq {
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
    };
    let n = used.len();
    let polys = dense.to_polynomial(n, 0, dense.gates.len());
    let perm = canonicalize_graph(&polys, n);
    let form = canonical_form(&polys, &perm)
        .into_iter()
        .map(|poly| {
            let mut monomials: Vec<_> = poly.into_iter().collect();
            monomials.sort_unstable();
            monomials
        })
        .collect();
    let mut canonical_circuit = dense;
    canonical_circuit.rewire(&perm, n);
    (form, canonical_circuit, n)
}

fn sampled_equivalent(
    left: &CircuitSeq,
    right: &CircuitSeq,
    n: usize,
    rng: &mut fastrand::Rng,
    samples: usize,
) -> bool {
    if n < usize::BITS as usize && (1usize << n) <= samples {
        return (0..(1usize << n)).all(|input| left.evaluate(input) == right.evaluate(input));
    }

    (0..samples).all(|_| {
        let input = if n == usize::BITS as usize {
            rng.usize(..)
        } else {
            rng.usize(0..(1usize << n))
        };
        left.evaluate(input) == right.evaluate(input)
    })
}

#[test]
fn random_rewirings_have_equal_polycanon_outputs() {
    let cases = env_usize("POLYCANON_STRESS_CASES", 1_000);
    let max_n = env_usize("POLYCANON_STRESS_MAX_N", 12).clamp(4, 63);
    let max_gate_factor = env_usize("POLYCANON_STRESS_GATE_FACTOR", 2).max(1);
    let samples = env_usize("POLYCANON_STRESS_SAMPLES", 256);
    let master_seed = 0x706f_6c79_6361_6e6fu64;
    let mut rng = fastrand::Rng::with_seed(master_seed);
    let mut form_failures = 0usize;
    let mut functional_failures = 0usize;
    let mut examples = Vec::new();

    for case in 0..cases {
        let n = rng.usize(4..=max_n);
        let m = rng.usize(1..=(max_gate_factor * n));
        let circuit_seed = rng.u64(..);
        fastrand::seed(circuit_seed);
        let original = random_circuit(n, m);

        let mut wire_order: Vec<_> = (0..n).collect();
        rng.shuffle(&mut wire_order);
        let shuffle = Permutation::new(wire_order);
        let mut rewired = original.clone();
        rewired.rewire(&shuffle, n);

        let (original_form, original_canonical, original_n) = canonicalize_circuit(&original);
        let (rewired_form, rewired_canonical, rewired_n) = canonicalize_circuit(&rewired);
        assert_eq!(original_n, rewired_n);
        let forms_equal = original_form == rewired_form;
        let functions_equal = sampled_equivalent(
            &original_canonical,
            &rewired_canonical,
            original_n,
            &mut rng,
            samples,
        );

        form_failures += usize::from(!forms_equal);
        functional_failures += usize::from(!functions_equal);
        if (!forms_equal || !functions_equal) && examples.len() < 10 {
            examples.push((
                case,
                circuit_seed,
                n,
                m,
                shuffle.data,
                forms_equal,
                functions_equal,
            ));
        }
    }

    println!("polycanon shuffle stress");
    println!("cases:                 {cases}");
    println!("master seed:           {master_seed}");
    println!("max wires:             {max_n}");
    println!("max gate factor:       {max_gate_factor}");
    println!("samples/case:          {samples}");
    println!("form failures:         {form_failures}");
    println!("functional failures:   {functional_failures}");
    for example in examples {
        println!(
            "failure case={} seed={} n={} m={} shuffle={:?} forms_equal={} functions_equal={}",
            example.0, example.1, example.2, example.3, example.4, example.5, example.6
        );
    }

    assert_eq!(form_failures, 0, "polycanon forms changed under rewiring");
    assert_eq!(
        functional_failures, 0,
        "polycanon canonical circuits changed function under rewiring"
    );
}
