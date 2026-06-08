use local_mixing::bench_support::{N_GRID, SEEDS, default_m, gen_polys};
use local_mixing::circuit::circuit::{Permutation, canonicalize_polys_4};
use local_mixing::random::random_data::random_circuit;
use std::hint::black_box;
use std::time::Instant;

const K: usize = 5;

fn is_valid(seed: u64, n: usize, m: usize, allow_rule_l: bool) -> bool {
    let original = gen_polys(seed, n, m);

    fastrand::seed(seed);
    let mut rewired = random_circuit(n, m);
    let mut wire_order: Vec<usize> = (0..n).collect();
    let mut rng = fastrand::Rng::with_seed(seed ^ 0x9e37_79b9_7f4a_7c15);
    rng.shuffle(&mut wire_order);
    rewired.rewire(&Permutation::new(wire_order), n);
    let rewired = rewired.to_polynomial(n, 0, m);

    match (
        canonicalize_polys_4(original, allow_rule_l),
        canonicalize_polys_4(rewired, allow_rule_l),
    ) {
        (Ok((original, _)), Ok((rewired, _))) => original == rewired,
        _ => false,
    }
}

fn benchmark(polys: Vec<local_mixing::circuit::Polynomial>, allow_rule_l: bool) -> (u128, bool) {
    let warmup_ok = black_box(canonicalize_polys_4(black_box(polys.clone()), allow_rule_l)).is_ok();
    let inputs: Vec<_> = (0..K).map(|_| polys.clone()).collect();
    let mut nanos = Vec::with_capacity(K);
    let mut all_ok = warmup_ok;

    for input in inputs {
        let start = Instant::now();
        let result = black_box(canonicalize_polys_4(
            black_box(input),
            black_box(allow_rule_l),
        ));
        nanos.push(start.elapsed().as_nanos());
        all_ok &= result.is_ok();
    }

    nanos.sort_unstable();
    (nanos[K / 2], all_ok)
}

fn main() {
    println!("algo,n,m,seed,variant,nanos,valid");

    for &n in N_GRID {
        let m = default_m(n);
        for &seed in SEEDS {
            for &(variant, allow_rule_l) in &[("rule_l_on", true), ("rule_l_off", false)] {
                let polys = gen_polys(seed, n, m);
                let (nanos, runs_ok) = benchmark(polys, allow_rule_l);
                let valid = runs_ok && is_valid(seed, n, m, allow_rule_l);
                println!(
                    "canon4,{n},{m},{seed},{variant},{nanos},{}",
                    u8::from(valid)
                );
            }
        }
    }
}
