use local_mixing::random::random_data::random_circuit;

fn main() {
    let n: usize = 16;

    let m: usize = 128;

    let mut ckt = random_circuit(n, m);

    ckt.canonicalize();

    let v = ckt.poly_stats(n);
    println!("{:?}", v);
}