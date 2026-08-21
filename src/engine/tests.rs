use super::arena::Arena;
use super::rules::{self, Outcome, Role};
use crate::circuit::Gate;
use crate::circuit::xgate::{XGate, eval_lanes};
use crate::experimental::split_engine::{Engine, Params};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn rng() -> StdRng {
    StdRng::seed_from_u64(0xf5197)
}

// Exhaustive functional equality of two gate sequences over wires 0..n (n <= 16).
fn eq_exhaustive(a: &[XGate], b: &[XGate], n: usize) -> bool {
    let total: u64 = 1 << n;
    let mut v = 0u64;
    while v < total {
        let mut st_a = vec![0u64; n];
        for (i, w) in st_a.iter_mut().enumerate() {
            let mut acc = 0u64;
            for l in 0..64u64 {
                if ((v + l) >> i) & 1 == 1 {
                    acc |= 1 << l;
                }
            }
            *w = acc;
        }
        let mut st_b = st_a.clone();
        eval_lanes(a, &mut st_a);
        eval_lanes(b, &mut st_b);
        let valid: u64 = if total - v >= 64 {
            !0
        } else {
            (1u64 << (total - v)) - 1
        };
        for i in 0..n {
            if (st_a[i] ^ st_b[i]) & valid != 0 {
                return false;
            }
        }
        v += 64;
    }
    true
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

// Random conjunction gate with `w` controls on wires 0..n, avoiding `target`.
fn random_conj(target: u16, w: usize, n: u16, rng: &mut impl Rng) -> XGate {
    let mut wires: Vec<u16> = (0..n).filter(|&x| x != target).collect();
    for i in 0..wires.len() {
        let j = rng.random_range(i..wires.len());
        wires.swap(i, j);
    }
    XGate::conj(
        target,
        wires[..w].iter().map(|&x| (x, rng.random_bool(0.5))),
    )
    .unwrap()
}

// The g57 firing convention must match the legacy evaluator bit for bit.
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

// Pre-split: function equality AND exclusivity (exactly one piece fires when the
// parent fires, none otherwise).
#[test]
fn presplit_exhaustive() {
    let mut r = rng();
    for _ in 0..200 {
        let g = XGate::from_g57(random_g57(6, &mut r));
        let pieces = rules::presplit(&g, &mut r);
        assert!(eq_exhaustive(&[g.clone()], &pieces, 6));
        for v in 0..(1u64 << 6) {
            let fires = |x: &XGate| -> bool {
                let mut acc = true;
                for &(w, p) in &x.ctrls {
                    acc &= ((v >> w) & 1 == 1) == p;
                }
                acc ^ x.comp
            };
            let parent = fires(&g);
            let count = pieces.iter().filter(|p| fires(p)).count();
            assert_eq!(count, usize::from(parent), "exclusivity broken at {v:b}");
        }
    }
}

// Random conjunction/g57 pairs through cross(): every Rewrite must be exactly
// functionally equal to [g, h] (verify_rewrite is itself exhaustive over the
// support, but check independently over all wires here).
#[test]
fn cross_random_pairs() {
    let n: u16 = 8;
    let mut r = rng();
    let mut seen = [0usize; 3];
    for i in 0..4000 {
        let a = r.random_range(0..n);
        let g = random_conj(a, r.random_range(1..=3), n, &mut r);
        let h = if i % 2 == 0 {
            let mut hb;
            loop {
                hb = random_g57(n, &mut r);
                if hb[0] != a {
                    break;
                }
            }
            XGate::from_g57(hb)
        } else {
            let mut b;
            loop {
                b = r.random_range(0..n);
                if b != a {
                    break;
                }
            }
            random_conj(b, r.random_range(1..=3), n, &mut r)
        };
        match rules::cross(&g, &h, 16, &mut r) {
            Outcome::Rewrite { seq, kind, .. } => {
                seen[match kind {
                    rules::RuleKind::R1 => 0,
                    rules::RuleKind::R2 => 1,
                    rules::RuleKind::R3 => 2,
                }] += 1;
                let after: Vec<XGate> = seq.iter().map(|(x, _)| x.clone()).collect();
                assert!(
                    eq_exhaustive(&[g.clone(), h.clone()], &after, n as usize),
                    "{kind:?} wrong: {g:?} x {h:?} -> {after:?}"
                );
                assert!(rules::verify_rewrite(&[g.clone(), h.clone()], &after));
                // Leftward form: [h, g] = reversed sequence.
                let rev: Vec<XGate> = seq.iter().rev().map(|(x, _)| x.clone()).collect();
                assert!(
                    eq_exhaustive(&[h.clone(), g.clone()], &rev, n as usize),
                    "{kind:?} leftward wrong: {h:?} then {g:?}"
                );
            }
            Outcome::PresplitColliding => {
                assert!(h.comp && h.reads(a) && !g.reads(h.target));
            }
            Outcome::R0Swap => {
                assert!(!XGate::collides(&g, &h));
            }
            Outcome::Blocked(_) => {}
        }
    }
    assert!(
        seen.iter().all(|&c| c > 100),
        "rule coverage too thin: {seen:?}"
    );
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

// The note's Fig. 12 identity: [a,b,q] x [b,a,v] (both g57) via presplit + R3.
// [a,b,q]·[b,a,v] = T(b,v→a)·T(¬b,¬q,v→a) · [b,a,v] · T(¬b,¬v→a)·T(b,¬q,¬v→a)
#[test]
fn fig12_regression() {
    let (a, b, q, v) = (0u16, 1u16, 2u16, 3u16);
    let g = XGate::from_g57([a, b, q]);
    let h = XGate::from_g57([b, a, v]);
    let expected = vec![
        XGate::conj(a, [(b, true), (v, true)]).unwrap(),
        XGate::conj(a, [(b, false), (q, false), (v, true)]).unwrap(),
        h.clone(),
        XGate::conj(a, [(b, false), (v, false)]).unwrap(),
        XGate::conj(a, [(b, true), (q, false), (v, false)]).unwrap(),
    ];
    assert!(eq_exhaustive(&[g.clone(), h.clone()], &expected, 4));

    // Cross the note's pre-split pieces (the presplit's ladder order is a free
    // randomized choice, so build the note's decomposition explicitly here;
    // presplit_exhaustive covers the presplit itself).
    let mut r = rng();
    let pieces = vec![
        XGate::conj(a, [(b, true)]).unwrap(),
        XGate::conj(a, [(b, false), (q, false)]).unwrap(),
    ];
    assert!(rules::verify_rewrite(std::slice::from_ref(&g), &pieces));
    let mut produced: Vec<XGate> = Vec::new();
    for p in &pieces {
        match rules::cross(p, &h, 16, &mut r) {
            Outcome::Rewrite { seq, kind, .. } => {
                assert_eq!(kind, rules::RuleKind::R3);
                for (x, role) in seq {
                    if role != Role::CollidingIntact {
                        produced.push(x);
                    }
                }
            }
            other => panic!("expected R3 rewrite, got {other:?}"),
        }
    }
    let mut want: Vec<XGate> = expected.iter().filter(|x| **x != h).cloned().collect();
    let key = |x: &XGate| format!("{x:?}");
    produced.sort_by_key(key);
    want.sort_by_key(key);
    assert_eq!(produced, want, "residue multiset differs from the note");
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
fn engine_small_circuit_exhaustive() {
    let n: u16 = 12;
    let mut r = rng();
    let gates = random_circuit(n, 80, &mut r);
    let mut e = Engine::new(
        gates.clone(),
        Params {
            k_max: 4,
            size_bound: 240,
            verify_every: 8,
            report_every: 1 << 30,
            seed: 42,
            ..Params::default()
        },
    );
    e.run();
    let out = e.arena.to_vec();
    assert!(out.len() >= 240, "size bound not reached: {}", out.len());
    assert!(
        eq_exhaustive(&gates, &out, n as usize),
        "engine broke the function"
    );
    // Some splitting actually happened.
    assert!(e.counters.splits_r1 + e.counters.splits_r2 + e.counters.splits_r3 > 0);
    // Cap respected.
    assert!(out.iter().all(|g| g.width() <= 4));

    // Final float preserves the function too.
    let (moved, _) = e.final_float();
    assert!(moved > 0);
    let floated = e.arena.to_vec();
    assert!(
        eq_exhaustive(&gates, &floated, n as usize),
        "final float broke the function"
    );
}

#[test]
fn engine_saturates_gracefully() {
    // Tiny wire count + tiny K: should stop by saturation, still correct.
    let n: u16 = 5;
    let mut r = rng();
    let gates = random_circuit(n, 30, &mut r);
    let mut e = Engine::new(
        gates.clone(),
        Params {
            k_max: 2,
            size_bound: 10_000,
            saturation_patience: 30,
            verify_every: 16,
            report_every: 1 << 30,
            seed: 7,
            ..Params::default()
        },
    );
    e.run();
    assert!(eq_exhaustive(&gates, &e.arena.to_vec(), n as usize));
}

// g57 x g57 targeting: with the window on, some episodes set up g57-g57
// collisions and the function is preserved; with the window at 0 it never does.
#[test]
fn engine_g57_g57_targeting() {
    let n: u16 = 12;
    let mut r = rng();
    let gates = random_circuit(n, 80, &mut r);

    let mut on = Engine::new(
        gates.clone(),
        Params {
            k_max: 4,
            g57_target_window: 64,
            size_bound: 220,
            verify_every: 8,
            report_every: 1 << 30,
            seed: 5,
            ..Params::default()
        },
    );
    on.run();
    assert!(
        eq_exhaustive(&gates, &on.arena.to_vec(), n as usize),
        "targeting broke the function"
    );
    assert!(
        on.counters.g57_g57_setups > 0,
        "no g57xg57 collisions were set up"
    );

    let mut off = Engine::new(
        gates.clone(),
        Params {
            k_max: 4,
            g57_target_window: 0,
            size_bound: 220,
            verify_every: 8,
            report_every: 1 << 30,
            seed: 5,
            ..Params::default()
        },
    );
    off.run();
    assert_eq!(
        off.counters.g57_g57_setups, 0,
        "window 0 must disable targeting"
    );
    assert!(eq_exhaustive(&gates, &off.arena.to_vec(), n as usize));
}

#[test]
fn mpmct_roundtrip() {
    let mut r = rng();
    let mut gates = random_circuit(10, 15, &mut r);
    gates.push(XGate::conj(0, [(3, false), (7, true), (9, false)]).unwrap());
    gates.push(XGate::x_gate(4));
    let dir = std::env::temp_dir().join("fsplit_roundtrip_test.txt");
    let path = dir.to_str().unwrap();
    super::format::write_mpmct(path, &gates, 10).unwrap();
    let (back, n) = super::format::read_mpmct(path).unwrap();
    assert_eq!(n, 10);
    assert_eq!(back, gates);
    std::fs::remove_file(path).ok();
}
