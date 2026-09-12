use super::{Transpositions, apply_neg_to_mask, apply_unsamf};
use crate::circuit::{Gate, U1024};
use crate::db_mixing::frozen::FrozenDb;
use rand::{Rng, RngCore};

// The full SAMF-insert -> unsamf cycle must be the identity, at LARGE n (n=384 = feistelized
// n=128) where the production failures appear. Builds a random sequence of swap+negation
// gadgets (physical) while tracking (t_list perm, mask neg); apply_unsamf appends the inverse;
// circ ++ unsamf must compute identity on every wire. Disabled stores -> integrate appends
// raw, so the real apply_unsamf perm/neg logic is exercised without any DB.
fn run_cycle(n: usize, num_samfs: usize, trials: usize) {
    let db = FrozenDb::empty();
    let mut rng = rand::rng();
    let wmask = (U1024::one() << n) - U1024::one();
    for trial in 0..trials {
        let mut circ: Vec<[u16; 3]> = Vec::new();
        let mut t_list = Transpositions {
            transpositions: Vec::new(),
        };
        let mut mask = vec![0u8; n];
        for _ in 0..num_samfs {
            let lo = rng.random_range(0..n as u16);
            let hi = loop {
                let w = rng.random_range(0..n as u16);
                if w != lo {
                    break w;
                }
            };
            let neg_type = rng.random_range(0u16..=3);
            circ.extend_from_slice(&Transpositions::gen_gates_swap(n, (lo, hi, neg_type)));
            t_list.transpositions.push((lo, hi, neg_type));
            apply_neg_to_mask(&mut mask, lo as usize, hi as usize, neg_type);
        }
        let mut g = circ.clone();
        apply_unsamf(
            &mut g,
            &t_list,
            &mask,
            n,
            &db,
            false,
            false,
            &mut Vec::new(),
        );
        for _ in 0..100 {
            let mut bytes = [0u8; 128];
            rng.fill_bytes(&mut bytes);
            let x = U1024::from_little_endian(&bytes) & wmask;
            let out = Gate::evaluate_index_list_1024(x, &g);
            assert_eq!(out, x, "n={n} trial={trial}: circ ∘ unsamf != identity");
        }
    }
}

#[test]
fn unsamf_cycle_identity_n64() {
    run_cycle(64, 200, 20);
}

#[test]
fn unsamf_cycle_identity_n192() {
    run_cycle(192, 200, 20);
}

#[test]
fn unsamf_cycle_identity_n384() {
    run_cycle(384, 300, 30);
}
