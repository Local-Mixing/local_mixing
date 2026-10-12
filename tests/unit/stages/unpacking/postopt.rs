use super::*;
use crate::stages::unpacking::mono::Rng;

type Lanes = [u64; 4];

fn simulate(gates: &[SGate], st: &mut [Lanes]) {
    for g in gates {
        let mut acc = [!0u64; 4];
        for i in 0..g.k as usize {
            let v = st[g.c[i] as usize];
            for j in 0..4 {
                acc[j] &= if g.p[i] { v[j] } else { !v[j] };
            }
        }
        if g.comp {
            for a in acc.iter_mut() {
                *a = !*a;
            }
        }
        for j in 0..4 {
            st[g.t as usize][j] ^= acc[j];
        }
    }
}

fn random_state(n: usize, rng: &mut Rng) -> Vec<Lanes> {
    (0..n)
        .map(|_| {
            let mut l = [0u64; 4];
            for x in l.iter_mut() {
                *x = rng.next_u64();
            }
            l
        })
        .collect()
}

#[test]
fn peephole_preserves_the_function_on_random_lists() {
    let n = 4;
    let m = 8;
    let mut rng = Rng::new(3);
    let mut removed = 0;
    for _ in 0..1500 {
        let mut gates: Vec<SGate> = Vec::new();
        for _ in 0..m {
            let t = rng.below(n as u64) as u32;
            let k = rng.below(3);
            let mut a = rng.below(n as u64) as u32;
            while a == t {
                a = rng.below(n as u64) as u32;
            }
            let mut b = rng.below(n as u64) as u32;
            while b == t || b == a {
                b = rng.below(n as u64) as u32;
            }
            let comp = rng.below(2) == 1;
            gates.push(match k {
                0 => SGate::not(t),
                1 => SGate::cnot(t, a, rng.below(2) == 1, comp),
                _ => SGate::tof(t, a, rng.below(2) == 1, b, rng.below(2) == 1, comp),
            });
        }
        let mut opt = gates.clone();
        removed += peephole(&mut opt);
        assert!(opt.len() <= gates.len());
        assert!(opt.iter().all(|g| !g.is_noop()));
        let st0 = random_state(n, &mut rng);
        let mut s1 = st0.clone();
        simulate(&gates, &mut s1);
        let mut s2 = st0;
        simulate(&opt, &mut s2);
        assert_eq!(s1, s2, "{gates:?} -> {opt:?}");
    }
    assert!(removed > 0, "the pass never fired");
}

#[test]
fn identical_gates_cancel_across_commuting_gates_and_merges_are_gates() {
    // two equal Toffolis separated by a gate on another target cancel
    let g = SGate::tof(0, 1, true, 2, false, false);
    let mut v = vec![g, SGate::cnot(3, 1, true, false), g];
    assert_eq!(peephole(&mut v), 2);
    assert_eq!(v, vec![SGate::cnot(3, 1, true, false)]);
    // CNOT(0;1) then NOT(0) merge into one complemented CNOT
    let mut v = vec![SGate::cnot(0, 1, true, false), SGate::not(0)];
    assert_eq!(peephole(&mut v), 1);
    assert_eq!(v, vec![SGate::cnot(0, 1, true, true)]);
    // a gate reading the target blocks the move
    let mut v = vec![g, SGate::cnot(3, 0, true, false), g];
    assert_eq!(peephole(&mut v), 0);
    // x_1 + x_2 is not a gate polynomial: no merge
    let mut v = vec![
        SGate::cnot(0, 1, true, false),
        SGate::cnot(0, 2, true, false),
    ];
    assert_eq!(peephole(&mut v), 0);
}
