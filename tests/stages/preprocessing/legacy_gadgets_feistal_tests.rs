use super::*;
use crate::circuit::Gate;
use rand::{SeedableRng, rngs::StdRng};

#[test]
fn middle_block_is_y_plus_cx() {
    let n = 3;
    let main = CircuitSeq {
        gates: vec![[0, 1, 2], [2, 0, 1]],
    };
    let mut rng = StdRng::seed_from_u64(0x57fe157a);
    let circuit = feistalize(&main, n, 1, &mut rng);
    let mask = (1usize << n) - 1;
    for input in 0..(1usize << (3 * n)) {
        let x = input & mask;
        let y = (input >> n) & mask;
        assert_eq!((circuit.evaluate(input) >> n) & mask, y ^ main.evaluate(x));
    }
}

fn middle_is_y_plus_cx(circuit: &CircuitSeq, main: &CircuitSeq, n: usize) -> bool {
    let mask = (1usize << n) - 1;
    (0..(1usize << (3 * n))).all(|input| {
        let x = input & mask;
        let y = (input >> n) & mask;
        (circuit.evaluate(input) >> n) & mask == y ^ main.evaluate(x)
    })
}

#[test]
fn symmetric_cd_middle_block_correct() {
    let n = 3;
    let main = CircuitSeq {
        gates: vec![[0, 1, 2], [2, 0, 1], [1, 2, 0]],
    };
    for seed in 0x7100u64..0x7106 {
        let mut rng = StdRng::seed_from_u64(seed);
        let circuit = feistalize_inner(&main, n, 1, &mut rng, true, false);
        assert!(
            middle_is_y_plus_cx(&circuit, &main, n),
            "sym_cd seed={seed:#x}"
        );
    }
}

#[test]
fn symmetric_g_middle_block_correct() {
    let n = 3;
    let main = CircuitSeq {
        gates: vec![[0, 1, 2], [2, 0, 1], [1, 2, 0]],
    };
    for seed in 0x7200u64..0x7206 {
        let mut rng = StdRng::seed_from_u64(seed);
        let circuit = feistalize_inner(&main, n, 1, &mut rng, true, true);
        assert!(
            middle_is_y_plus_cx(&circuit, &main, n),
            "sym_g seed={seed:#x}"
        );
    }
}

#[test]
fn sg3_realizes_g57() {
    let state = FeistalState {
        sharing: GadgetState {
            n: 3,
            pairs: vec![(0, 1), (3, 4), (6, 7)],
        },
        free: vec![2, 5, 8],
        q: vec![1, 2, 0],
    };
    let mut gates = Vec::new();
    emit_sg3(&state, [0, 1, 2], &mut gates);
    for input in 0..512usize {
        let decode = |v: usize| {
            (0..3).fold(0, |acc, i| {
                acc | ((((v >> state.sharing.pairs[i].0) & 1)
                    ^ ((v >> state.sharing.pairs[i].1) & 1)
                    ^ ((v >> state.free[i]) & 1))
                    << i)
            })
        };
        assert_eq!(
            decode(Gate::evaluate_index_list(input, &gates)),
            Gate::evaluate_index(decode(input), [0, 1, 2])
        );
    }
}
