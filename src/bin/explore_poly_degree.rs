use local_mixing::random::random_data::random_circuit;

fn main() {
    let n: usize = 16;

    let m: usize = 64;

    let mut ckt = random_circuit(n, m);

    let g = ckt.gates.clone();

    ckt.canonicalize();

    ckt.gates.extend(g.iter().rev());

    let v = ckt.poly_num_terms(n);
    println!("{:?}", v);
}