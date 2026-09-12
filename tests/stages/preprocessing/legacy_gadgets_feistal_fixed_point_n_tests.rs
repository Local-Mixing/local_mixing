use super::*;
use crate::circuit::Gate;

fn decode(state: &FeistalState, physical: usize) -> (usize, usize) {
    let mut x = 0usize;
    let mut y = 0usize;
    for i in 0..state.sharing.n {
        let (p0, p1) = state.sharing.pairs[i];
        let pair = ((physical >> p0) & 1) ^ ((physical >> p1) & 1);
        x |= (pair ^ ((physical >> state.free[i]) & 1)) << i;
        y |= pair << state.q[i];
    }
    (x, y)
}

#[test]
fn n_tilde_supports_q_fixed_points_without_moving_carriers() {
    let state = FeistalState {
        sharing: GadgetState {
            n: 3,
            pairs: vec![(0, 1), (3, 4), (6, 7)],
        },
        free: vec![2, 5, 8],
        q: vec![0, 1, 2],
    };
    let original_pairs = state.sharing.pairs.clone();
    let original_free = state.free.clone();
    let mut gates = Vec::new();
    emit_feistal_n(&state, &mut gates);
    assert_eq!(state.sharing.pairs, original_pairs);
    assert_eq!(state.free, original_free);
    for input in 0..512usize {
        let (x, y) = decode(&state, input);
        let output = Gate::evaluate_index_list(input, &gates);
        assert_eq!(decode(&state, output), (x, y ^ x));
    }
}
