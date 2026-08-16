//! Random float of gates within their commutation brackets: apply
//! commuting_shuffle to a whole mpmct1 circuit (order rerandomized, function
//! unchanged) and write the result. An optional litter sidecar is permuted
//! identically so litter identity survives the float.
//! Usage: shuffle_mpmct <in.mpmct1> <out.mpmct1> <seed> [litter_in litter_out]
use local_mixing::postmix::format::{read_mpmct, write_mpmct};
use local_mixing::replace::gadgets::commuting_shuffle_order;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::io::Write;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (inp, outp, seed): (&str, &str, u64) = (&a[1], &a[2], a[3].parse().unwrap());
    let (mut gates, wires) = read_mpmct(inp).expect("read");
    let mut rng = StdRng::seed_from_u64(seed);
    let order = commuting_shuffle_order(&mut gates, &mut rng);
    write_mpmct(outp, &gates, wires).expect("write");
    if a.len() >= 6 {
        let s = std::fs::read_to_string(&a[4]).expect("read litter sidecar");
        let ids: Vec<u64> = s
            .lines()
            .skip(1)
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.trim().parse().expect("bad litter id"))
            .collect();
        assert_eq!(ids.len(), gates.len(), "litter sidecar length mismatch");
        let mut f = std::fs::File::create(&a[5]).expect("litter out");
        writeln!(f, "litter1 {}", ids.len()).unwrap();
        for &i in &order {
            writeln!(f, "{}", ids[i as usize]).unwrap();
        }
    }
    println!("[shuffle] {} gates floated, wrote {outp}", gates.len());
}
