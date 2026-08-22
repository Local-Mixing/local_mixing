//! Probe a gadget's input convention: for x on 0..n-1, z=0 on n..2n-1, and aux
//! on 2n..4n-1 set to either 0 or random, does G's low-2n output equal the
//! sandwich A(x,0), and does C(x) appear on the high half?
//! Usage: check_output <C.g57> <G> <g57|mpmct1> <n> <A.g57 2n-wire> <aux:zero|rand>
use local_mixing::circuit::U1024;
use local_mixing::engine::format::{read_g57_file, read_mpmct};
use local_mixing::circuit::xgate::eval_u1024;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let c = read_g57_file(&a[1]).expect("read C g57");
    let g = if a[3] == "g57" {
        read_g57_file(&a[2]).expect("read G g57")
    } else {
        read_mpmct(&a[2]).expect("read G mpmct1").0
    };
    let n: usize = a[4].parse().unwrap();
    let asw = read_g57_file(&a[5]).expect("read A g57");
    let mode = a.get(6).map(|s| s.as_str()).unwrap_or("zero"); // zero | rand | full256
    let mask = (U1024::one() << n) - U1024::one();
    let mask2 = (U1024::one() << (2 * n)) - U1024::one();
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let trials = 20;
    let (mut low_a, mut high_c) = (0, 0);
    for _ in 0..trials {
        let mut b = [0u8; 128];
        rng.fill_bytes(&mut b);
        // sandwich input on 0..2n-1: full256 = (x,z) both random; else (x, z=0)
        let sand: U1024 = if mode == "full256" {
            let mut b2 = [0u8; 128];
            rng.fill_bytes(&mut b2);
            U1024::from_little_endian(&b2) & mask2
        } else {
            U1024::from_little_endian(&b) & mask // x on 0..n-1, z=0
        };
        let mut ginput = sand;
        if mode == "rand" {
            let mut ab = [0u8; 128];
            rng.fill_bytes(&mut ab);
            ginput = ginput | ((U1024::from_little_endian(&ab)) & (mask2 << (2 * n)));
        }
        let ax = eval_u1024(&asw, sand) & mask2; // A(sandwich input) on 2n wires
        let out = eval_u1024(&g, ginput);
        if (out & mask2) == ax { low_a += 1; }
        // C(x) on the high half only meaningful when z=0
        if mode != "full256" {
            let cx = eval_u1024(&c, sand & mask) & mask;
            if ((out >> n) & mask) == cx { high_c += 1; }
        }
    }
    println!(
        "mode={:8}  G low-2n == A(input): {}/{}   G high-half == C(x): {}/{}",
        mode, low_a, trials, high_c, trials
    );
}
