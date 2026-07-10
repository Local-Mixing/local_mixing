use std::hint::black_box;

use local_mixing::{
    circuit::{
        Polynomial,
        circuit::{CircuitSeq, poly_to_compressed_str, polys_repr_blob},
    },
    random::random_data::random_circuit,
};

fn main() {
    let n: usize = 8;

    let m: usize = 256;

    let mut ckt = random_circuit(n, m);

    ckt.canonicalize();

    let start = std::time::Instant::now();
    let v1: Vec<Polynomial> = black_box(ckt.to_polynomial(n, 0, m / 3));
    let v2: Vec<Polynomial> = black_box(ckt.to_polynomial(n, m / 3, 2 * m / 3));
    let v3: Vec<Polynomial> = black_box(ckt.to_polynomial(n, 2 * m / 3, m));
    // let r: Vec<Polynomial> = v1
    //     .iter()
    //     .zip(v2.iter())
    //     .map(|(v1, v2)| CircuitSeq::compose_polys(&v1, &v2))
    //     .collect();
    let duration = start.elapsed();
    println!("Half Time: {} ms", duration.as_millis());

    let start = std::time::Instant::now();
    let v: Vec<Polynomial> = black_box(ckt.to_polynomial(n, 0, m));
    // assert!(r == v);
    let duration = start.elapsed();
    println!("Full Time: {} ms", duration.as_millis());

    // for p in v {
    //     println!("{:?}", poly_to_compressed_str(&p, n));
    // }
}
