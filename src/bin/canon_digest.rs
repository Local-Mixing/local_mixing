// Cross-version canonical-form compatibility digest.
//
// Runs the exact entry point build_from_rocks uses per candidate
// (CircuitSeq::canonicalize_polys -> canonicalize_polys_4 both directions)
// over a deterministic corpus and digests the DB key material (canonical
// polys blob), the DB value material (canonical circuit blob), and the
// chosen direction. Two builds of this binary — e.g. before and after a
// canonicalization patch — must print the same digest, or the patch changes
// DB keys and must not be deployed mid-pipeline.
//
// Self-contained on purpose: inline xorshift RNG (no fastrand/rand),
// only stable lib APIs shared by both the local and server trees.
//
// Env knobs: CANON_DIGEST_CASES (default 60), CANON_DIGEST_SYM_MAX (default 7).

use local_mixing::circuit::circuit::{CircuitSeq, polys_repr_blob};
use std::time::Instant;
use xxhash_rust::xxh3::Xxh3;

struct XorShift64(u64);

impl XorShift64 {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn rand_gate(rng: &mut XorShift64, n: u16) -> [u16; 3] {
    let a = rng.below(n as u64) as u16;
    let mut b = rng.below(n as u64) as u16;
    while b == a {
        b = rng.below(n as u64) as u16;
    }
    let mut c = rng.below(n as u64) as u16;
    while c == a || c == b {
        c = rng.below(n as u64) as u16;
    }
    [a, b, c]
}

fn digest_circuit(c: &CircuitSeq, hasher: &mut Xxh3, count: &mut usize) {
    match c.canonicalize_polys(30, true) {
        Some(canon) => {
            hasher.update(&polys_repr_blob(&canon.0));
            hasher.update(&canon.1.repr_blob());
            hasher.update(&[canon.2 as u8]);
        }
        None => hasher.update(b"none"),
    }
    *count += 1;
}

fn main() {
    let cases = env_usize("CANON_DIGEST_CASES", 60);
    let sym_max = env_usize("CANON_DIGEST_SYM_MAX", 7);
    let mut hasher = Xxh3::new();
    let mut rng = XorShift64(0x00d1_6e57_c0ffee_11);
    let mut count = 0usize;
    let t0 = Instant::now();

    // A) m10-candidate shape: 10 random gates on 24..30 wires.
    for &n in &[24u16, 26, 28, 30] {
        let ts = Instant::now();
        for _ in 0..cases {
            let gates: Vec<[u16; 3]> = (0..10).map(|_| rand_gate(&mut rng, n)).collect();
            digest_circuit(&CircuitSeq { gates }, &mut hasher, &mut count);
        }
        println!(
            "section a n={n} done count={count} elapsed_ms={}",
            ts.elapsed().as_millis()
        );
    }

    // B) denser small circuits: 5..=12 gates on 12 wires.
    for g in 5usize..=12 {
        let ts = Instant::now();
        for _ in 0..cases {
            let gates: Vec<[u16; 3]> = (0..g).map(|_| rand_gate(&mut rng, 12)).collect();
            digest_circuit(&CircuitSeq { gates }, &mut hasher, &mut count);
        }
        println!(
            "section b g={g} done count={count} elapsed_ms={}",
            ts.elapsed().as_millis()
        );
    }

    // C) symmetric Rule L stress: k parallel gates / k parallel 2-gate motifs.
    for k in 2..=sym_max as u16 {
        let ts = Instant::now();
        let single = CircuitSeq {
            gates: (0..k).map(|i| [3 * i, 3 * i + 1, 3 * i + 2]).collect(),
        };
        digest_circuit(&single, &mut hasher, &mut count);
        let motif = CircuitSeq {
            gates: (0..k)
                .flat_map(|i| [[3 * i, 3 * i + 1, 3 * i + 2], [3 * i + 1, 3 * i + 2, 3 * i]])
                .collect(),
        };
        digest_circuit(&motif, &mut hasher, &mut count);
        println!(
            "section c k={k} done count={count} elapsed_ms={}",
            ts.elapsed().as_millis()
        );
    }

    println!("digest: {:032x}", hasher.digest128());
    println!("count: {count} total_elapsed_ms: {}", t0.elapsed().as_millis());
}
