//! Random float of gates within their commutation brackets: apply
//! commuting_shuffle to a whole mpmct1 circuit (order rerandomized, function
//! unchanged) and write the result.
//! Usage: shuffle_mpmct <in.mpmct1> <out.mpmct1> <seed>
//! (The optional litter-sidecar permutation went to the archive branch
//! ran/archive/curated-and-postmix with its producer and the mixer's
//! --litter-in.)
use local_mixing::circuit::formats::{read_mpmct, write_mpmct};
use local_mixing::circuit::randomize::commuting_shuffle_order;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (inp, outp, seed): (&str, &str, u64) = (&a[1], &a[2], a[3].parse().unwrap());
    let (mut gates, wires) = read_mpmct(inp).expect("read");
    let mut rng = StdRng::seed_from_u64(seed);
    let _order = commuting_shuffle_order(&mut gates, &mut rng);
    write_mpmct(outp, &gates, wires).expect("write");
    println!("[shuffle] {} gates floated, wrote {outp}", gates.len());
}
