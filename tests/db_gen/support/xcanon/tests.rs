use super::*;
use crate::circuit::CircuitSeq;
use crate::engine::xpoly::{XPolyBudget, xgates_to_polynomial};

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn random_g57_circuit(rng: &mut Lcg, wires: u64, len: usize) -> Vec<[u16; 3]> {
    (0..len)
        .map(|_| {
            let a = rng.below(wires) as u16;
            let mut x = rng.below(wires) as u16;
            while x == a {
                x = rng.below(wires) as u16;
            }
            let mut y = rng.below(wires) as u16;
            while y == a || y == x {
                y = rng.below(wires) as u16;
            }
            [a, x, y]
        })
        .collect()
}

fn random_mixed_circuit(rng: &mut Lcg, wires: u64, len: usize) -> Vec<XGate> {
    (0..len)
        .map(|_| {
            if rng.below(2) == 0 {
                let g = random_g57_circuit(rng, wires, 1);
                XGate::from_g57(g[0])
            } else {
                let target = rng.below(wires) as u16;
                let k = 1 + rng.below(3) as usize;
                let mut lits: Vec<(u16, bool)> = Vec::new();
                while lits.len() < k {
                    let w = rng.below(wires) as u16;
                    if w != target && !lits.iter().any(|&(lw, _)| lw == w) {
                        lits.push((w, rng.below(2) == 1));
                    }
                }
                XGate::conj(target, lits).unwrap()
            }
        })
        .collect()
}

#[test]
fn pure_g57_canonicalization_agrees_with_legacy() {
    let mut rng = Lcg(0x1234_5678_9abc_def0);
    for case in 0..20_000 {
        let len = 3 + rng.below(6) as usize;
        let g57 = random_g57_circuit(&mut rng, 10, len);
        let mut legacy = CircuitSeq { gates: g57.clone() };
        legacy.canonicalize();
        let want: Vec<XGate> = legacy.gates.iter().copied().map(XGate::from_g57).collect();
        let mut got: Vec<XGate> = g57.iter().copied().map(XGate::from_g57).collect();
        xgate_canonicalize(&mut got);
        assert_eq!(got, want, "case {case}: {g57:?}");
    }
}

#[test]
fn canonicalization_is_idempotent_and_function_preserving() {
    let mut rng = Lcg(0xfeed_face_dead_beef);
    for case in 0..5_000 {
        let len = 3 + rng.below(6) as usize;
        let gates = random_mixed_circuit(&mut rng, 9, len);
        let before = xgates_to_polynomial(&gates, 9, XPolyBudget::default()).unwrap();
        let mut canon = gates.clone();
        xgate_canonicalize(&mut canon);
        let after = xgates_to_polynomial(&canon, 9, XPolyBudget::default()).unwrap();
        assert_eq!(before, after, "case {case}: function changed");
        let mut twice = canon.clone();
        xgate_canonicalize(&mut twice);
        assert_eq!(twice, canon, "case {case}: not idempotent");
    }
}

#[test]
fn adjacent_id_detects_cancelling_pairs() {
    let g = XGate::conj(0, [(1, true), (2, false), (3, true)]).unwrap();
    let h = XGate::from_g57([4, 1, 2]);
    assert!(xgate_adjacent_id(&[h.clone(), g.clone(), g.clone()]));
    assert!(!xgate_adjacent_id(&[g.clone(), h.clone(), g]));
}
