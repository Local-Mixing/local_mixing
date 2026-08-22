//! Generate a fresh random g57 circuit and write it as mpmct1.
//! Usage: gen_random_mpmct <out.mpmct1> <n> <m> <seed>
use local_mixing::engine::format::write_mpmct;
use local_mixing::circuit::xgate::XGate;
use local_mixing::circuit::random_circuit;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (out, n, m, seed) = (&a[1], a[2].parse().unwrap(), a[3].parse().unwrap(), a[4].parse().unwrap());
    fastrand::seed(seed);
    let c = random_circuit(n, m);
    let gates: Vec<XGate> = c.gates.iter().map(|g| XGate::from_g57(*g)).collect();
    write_mpmct(out, &gates, n).expect("write");
    println!("[genrand] wrote {} gates / {} wires to {}", gates.len(), n, out);
}
