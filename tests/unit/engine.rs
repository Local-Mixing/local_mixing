use super::arena::Arena;
use super::moves::rules;
use crate::circuit::Gate;
use crate::circuit::xgate::XGate;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn rng() -> StdRng {
    StdRng::seed_from_u64(0xf5197)
}

fn random_g57(n: u16, rng: &mut impl Rng) -> [u16; 3] {
    loop {
        let a = rng.random_range(0..n);
        let x = rng.random_range(0..n);
        let y = rng.random_range(0..n);
        if a != x && a != y && x != y {
            return [a, x, y];
        }
    }
}

fn random_circuit(n: u16, m: usize, rng: &mut impl Rng) -> Vec<XGate> {
    (0..m)
        .map(|_| XGate::from_g57(random_g57(n, rng)))
        .collect()
}

// The g57 firing convention must match the G57 evaluator bit for bit.
#[test]
fn xgate_eval_matches_g57() {
    let mut r = rng();
    for _ in 0..500 {
        let g = random_g57(20, &mut r);
        let xg = XGate::from_g57(g);
        for _ in 0..16 {
            let state: usize = r.random_range(0..(1usize << 20));
            let expect = Gate::evaluate_index(state, g);
            let mut lanes = vec![0u64; 20];
            for (i, l) in lanes.iter_mut().enumerate() {
                *l = ((state >> i) & 1) as u64;
            }
            xg.apply_lanes(&mut lanes);
            let mut got = 0usize;
            for (i, l) in lanes.iter().enumerate() {
                got |= ((l & 1) as usize) << i;
            }
            assert_eq!(got, expect, "g57 {g:?} state {state:b}");
        }
    }
}

// verify_rewrite must reject a wrong rewrite (negative control).
#[test]
fn verify_rejects_bad_rewrite() {
    let g = XGate::conj(0, [(1, true)]).unwrap();
    let h = XGate::conj(1, [(2, true)]).unwrap();
    // Naive swap of colliding gates is wrong.
    assert!(!rules::verify_rewrite(
        &[g.clone(), h.clone()],
        &[h.clone(), g.clone()]
    ));
    assert!(rules::verify_rewrite(&[g.clone(), h.clone()], &[g, h]));
}

#[test]
fn arena_basics() {
    let mut r = rng();
    let gates = random_circuit(8, 20, &mut r);
    let mut ar = Arena::from_gates(gates.clone());
    assert_eq!(ar.to_vec(), gates);
    // Unlink+relink a middle node to the front.
    let ids = ar.ids_in_order();
    ar.unlink(ids[5]);
    ar.link_after(ids[5], super::arena::NIL);
    let v = ar.to_vec();
    assert_eq!(v.len(), 20);
    assert_eq!(v[0], gates[5]);
    // Replace via insert/free.
    let id = ar.insert_after(ids[3], gates[0].clone());
    assert!(ar.is_linked(id));
    assert_eq!(ar.len(), 21);
}

#[test]
fn mpmct_roundtrip() {
    let mut r = rng();
    let mut gates = random_circuit(10, 15, &mut r);
    gates.push(XGate::conj(0, [(3, false), (7, true), (9, false)]).unwrap());
    gates.push(XGate::x_gate(4));
    let dir = std::env::temp_dir().join("mpmct_roundtrip_test.txt");
    let path = dir.to_str().unwrap();
    crate::circuit::formats::write_mpmct(path, &gates, 10).unwrap();
    let (back, n) = crate::circuit::formats::read_mpmct(path).unwrap();
    assert_eq!(n, 10);
    assert_eq!(back, gates);
    std::fs::remove_file(path).ok();
}
