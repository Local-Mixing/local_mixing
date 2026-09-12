use super::{
    SWAP_N1_3W, SWAP_N1_4W, SWAP_N2_3W, SWAP_N2_4W, Transpositions, WireMap, neg_flips,
    random_neg_type, shoot_gate_to_first_collision, shoot_materialized_gate_to_first_collision,
};
use crate::circuit::CircuitSeq;
use std::collections::VecDeque;

#[test]
fn shot_gate_moves_to_its_first_collision() {
    let shot = [0, 1, 2];
    let pass_a = [3, 4, 5];
    let pass_b = [6, 7, 8];
    let collider = [9, 0, 10];
    let suffix = [11, 12, 13];
    let mut remaining = VecDeque::from([shot, pass_a, pass_b, collider, suffix]);
    let t = Transpositions {
        transpositions: Vec::new(),
    };
    let wm = WireMap::from_transpositions(&t, 17);

    let (actual_shot, passed, collided) =
        shoot_gate_to_first_collision(&mut remaining, &wm, &[0; 17]).unwrap();

    assert_eq!(actual_shot, shot);
    assert_eq!(passed, vec![pass_a, pass_b]);
    assert!(collided);
    assert_eq!(remaining, VecDeque::from([collider, suffix]));
}

#[test]
fn next_shot_continues_from_suffix_after_collision() {
    let collider = [9, 0, 10];
    let suffix_a = [11, 12, 13];
    let suffix_b = [14, 15, 16];
    let mut remaining = VecDeque::from([collider, suffix_a, suffix_b]);
    let t = Transpositions {
        transpositions: Vec::new(),
    };
    let wm = WireMap::from_transpositions(&t, 17);

    let (actual_shot, passed, collided) =
        shoot_gate_to_first_collision(&mut remaining, &wm, &[0; 17]).unwrap();

    assert_eq!(actual_shot, collider);
    assert_eq!(passed, vec![suffix_a, suffix_b]);
    assert!(!collided);
    assert!(remaining.is_empty());
}

#[test]
fn materialized_replacement_tail_is_not_relabelled_again() {
    let shot = [0, 1, 2];
    let commuting = [3, 4, 5];
    let collider_after_relabel = [6, 7, 8];
    let mut remaining = VecDeque::from([commuting, collider_after_relabel]);
    let t = Transpositions {
        transpositions: vec![(0, 7, 0)],
    };
    let wm = WireMap::from_transpositions(&t, 17);

    let (passed, collided) =
        shoot_materialized_gate_to_first_collision(shot, &mut remaining, &wm, &[0; 17]);

    assert_eq!(passed, vec![commuting]);
    assert!(collided);
    assert_eq!(remaining, VecDeque::from([collider_after_relabel]));
}

#[test]
fn materialized_shot_stops_before_dirty_control_correction() {
    let shot = [0, 1, 2];
    let dirty_control_gate = [3, 4, 5];
    let suffix = [6, 7, 8];
    let mut remaining = VecDeque::from([dirty_control_gate, suffix]);
    let t = Transpositions {
        transpositions: Vec::new(),
    };
    let wm = WireMap::from_transpositions(&t, 9);
    let mut negation_mask = [0; 9];
    negation_mask[4] = 1;

    let (passed, collided) =
        shoot_materialized_gate_to_first_collision(shot, &mut remaining, &wm, &negation_mask);

    assert!(passed.is_empty());
    assert!(!collided);
    assert_eq!(remaining, VecDeque::from([dirty_control_gate, suffix]));
}

#[test]
fn dirty_control_shot_stays_before_commuting_suffix() {
    let shot = [0, 1, 2];
    let commuting = [3, 4, 5];
    let mut remaining = VecDeque::from([shot, commuting]);
    let t = Transpositions {
        transpositions: Vec::new(),
    };
    let wm = WireMap::from_transpositions(&t, 6);
    let mut negation_mask = [0; 6];
    negation_mask[1] = 1;

    let (actual_shot, passed, collided) =
        shoot_gate_to_first_collision(&mut remaining, &wm, &negation_mask).unwrap();

    assert_eq!(actual_shot, shot);
    assert!(passed.is_empty());
    assert!(!collided);
    assert_eq!(remaining, VecDeque::from([commuting]));
}

// Structural canonical form of a gadget UP TO (a) wire relabeling and (b) reordering of
// commuting gates. We densify the used wires, then over every permutation of those wires
// run CircuitSeq::canonicalize() (which canonicalizes commuting-gate order) and keep the
// lexicographically-smallest gate sequence. Two gadgets share a key iff they are the same
// circuit up to relabeling wires and swapping adjacent commuting gates.
fn structural_key(gates: &[[u16; 3]]) -> Vec<[u16; 3]> {
    use itertools::Itertools;
    let c = CircuitSeq {
        gates: gates.to_vec(),
    };
    let used = c.used_wires(); // sorted, unique
    let k = used.len();
    let dense: std::collections::HashMap<u16, u16> = used
        .iter()
        .enumerate()
        .map(|(i, &w)| (w, i as u16))
        .collect();
    let base: Vec<[u16; 3]> = gates
        .iter()
        .map(|g| [dense[&g[0]], dense[&g[1]], dense[&g[2]]])
        .collect();
    let mut best: Option<Vec<[u16; 3]>> = None;
    for perm in (0..k as u16).permutations(k) {
        let relabeled: Vec<[u16; 3]> = base
            .iter()
            .map(|g| {
                [
                    perm[g[0] as usize],
                    perm[g[1] as usize],
                    perm[g[2] as usize],
                ]
            })
            .collect();
        let mut cc = CircuitSeq { gates: relabeled };
        cc.canonicalize();
        if best.as_ref().is_none_or(|b| &cc.gates < b) {
            best = Some(cc.gates);
        }
    }
    best.unwrap()
}

// The N1/N2 pools now also contain the (unique) negate-then-swap reversals of the opposite
// type. Since reverse(N2) computes the N1 permutation (and vice-versa), every reversed
// circuit of the opposite pool must already appear in the destination pool's structural
// forms (up to wire relabeling + commuting-gate order) — i.e. the pools are closed under
// reversal-of-the-opposite-type. This guards that the hardcoded reversals are complete.
#[test]
fn pools_closed_under_reversal() {
    use std::collections::HashSet;
    for (dst_pool, src_pool) in [
        (SWAP_N1_3W, SWAP_N2_3W),
        (SWAP_N1_4W, SWAP_N2_4W),
        (SWAP_N2_3W, SWAP_N1_3W),
        (SWAP_N2_4W, SWAP_N1_4W),
    ] {
        let dst: HashSet<Vec<[u16; 3]>> = dst_pool.iter().map(|c| structural_key(c)).collect();
        for c in src_pool.iter() {
            let mut g = c.to_vec();
            g.reverse();
            assert!(
                dst.contains(&structural_key(&g)),
                "a reversal of the opposite pool is missing from the destination pool"
            );
        }
    }
}

#[test]
fn random_neg_type_in_range() {
    let mut rng = rand::rng();
    let mut seen = [false; 4];
    for _ in 0..5000 {
        let t = random_neg_type(&mut rng);
        assert!(t <= 3, "neg type out of range: {}", t);
        seen[t as usize] = true;
    }
    assert!(seen.iter().all(|&s| s), "not all of 0..=3 were drawn");
}

// Logical 2-bit op of a swap gadget on wires (a, b), all other wires held at 0.
// Asserts every non-(a,b) wire is restored to 0 (ancilla clean).
fn logical_op(
    gates: &[[u16; 3]],
    n: usize,
    a: u16,
    b: u16,
    xa: usize,
    xb: usize,
) -> (usize, usize) {
    let input = (xa << a) | (xb << b);
    let c = CircuitSeq {
        gates: gates.to_vec(),
    };
    let out = c.evaluate(input);
    for w in 0..n {
        if w as u16 != a && w as u16 != b {
            assert_eq!((out >> w) & 1, 0, "ancilla wire {} not restored to 0", w);
        }
    }
    ((out >> a) & 1, (out >> b) & 1)
}

// The net op of any neg_type is "swap, then negate per neg_flips":
//   out_a = x_b ^ flip_lo,  out_b = x_a ^ flip_hi.
fn expected(neg: u16, xa: usize, xb: usize) -> (usize, usize) {
    let (flip_lo, flip_hi) = neg_flips(neg);
    (xb ^ flip_lo as usize, xa ^ flip_hi as usize)
}

#[test]
fn neg_flips_parity() {
    assert_eq!(neg_flips(0), (false, false));
    assert_eq!(neg_flips(1), (true, false));
    assert_eq!(neg_flips(2), (false, true));
    assert_eq!(neg_flips(3), (true, true));
}

#[test]
fn gen_gates_swap_logical_op_all_types() {
    // n=3 exercises only 3-wire pools; n=4 also exercises 4-wire pools. Many iterations
    // cover the random pool + ancilla choices, so every pool entry (including the
    // hardcoded negate-then-swap reversals) is checked to compute its type's permutation.
    for &(n, a, b) in &[(3usize, 1u16, 2u16), (4, 1, 3), (4, 0, 2)] {
        for neg in 0u16..=3 {
            for _ in 0..300 {
                let gates = Transpositions::gen_gates_swap(n, (a, b, neg));
                assert!(!gates.is_empty());
                for xa in 0..2 {
                    for xb in 0..2 {
                        assert_eq!(
                            logical_op(&gates, n, a, b, xa, xb),
                            expected(neg, xa, xb),
                            "neg={} n={} (a,b)=({},{}) input=({},{})",
                            neg,
                            n,
                            a,
                            b,
                            xa,
                            xb
                        );
                    }
                }
            }
        }
    }
}
