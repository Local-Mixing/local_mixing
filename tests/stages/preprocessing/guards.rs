use super::*;
use crate::circuit::xgate::eval_u64;
use rand::{SeedableRng, rngs::StdRng};

/// The closing zero-slice block has the opening block's specification —
/// identity exactly on the zero slice, every nonzero slice perturbs the
/// data — with its targets confined to the low (forward-junk) half.
#[test]
fn slice_zero_postblock_fixes_only_zero_slice_and_targets_the_low_half() {
    let n = 6usize;
    let nondata = 4usize;
    let mask = (1u64 << n) - 1;
    for seed in 0..8u64 {
        let mut rng = StdRng::seed_from_u64(0xc105_0000 + seed);
        let block = slice_zero_junk_guard_dims(n, nondata, 3 * nondata, &mut rng);
        for g in &block.gates {
            assert!(
                (g.target as usize) < n / 2,
                "closing-block target {} outside the junk half",
                g.target
            );
        }
        for x in 0..=mask {
            assert_eq!(
                eval_u64(&block.gates, x) & mask,
                x,
                "not identity on the zero slice (seed={seed})"
            );
        }
        for s in 1..(1u64 << nondata) {
            let disturbed = (0..=mask).any(|x| eval_u64(&block.gates, x | (s << n)) & mask != x);
            assert!(
                disturbed,
                "slice {s:#x} leaves the data fixed (seed={seed})"
            );
        }
    }
}

#[test]
fn nonlinear_guard_decomposition_preserves_every_slice_and_restores_scratch() {
    let (n, nondata, gates) = (6, 4, 40);
    for seed in 0..4 {
        let mut wide_rng = StdRng::seed_from_u64(seed);
        let mut narrow_rng = StdRng::seed_from_u64(seed);
        let wide =
            try_nonlinear_slice_zero_preblock_dims(n, nondata, gates, false, 6, 7, &mut wide_rng)
                .unwrap();
        let narrow =
            try_nonlinear_slice_zero_preblock_dims(n, nondata, gates, true, 6, 7, &mut narrow_rng)
                .unwrap();
        assert!(narrow.gates.iter().all(|gate| gate.ctrls.len() <= 2));
        for state in 0..1 << (n + nondata) {
            let actual = eval_u64(&narrow.gates, state);
            assert_eq!(actual, eval_u64(&wide.gates, state));
            assert_eq!(
                actual >> n,
                state >> n,
                "dirty scratch and all slice wires must be restored"
            );
        }
    }
}

#[test]
fn nonlinear_preblock_weight2_decomposition_is_exact_and_bounded() {
    let (n, nondata, logical_gates) = (8usize, 20usize, 200usize);
    let scratch = n as u16;
    let scratch2 = (n + 1) as u16;
    let mut wide_rng = StdRng::seed_from_u64(0xcc88_0001);
    let wide = try_nonlinear_slice_zero_preblock_dims(
        n,
        nondata,
        logical_gates,
        false,
        scratch,
        scratch2,
        &mut wide_rng,
    )
    .unwrap();
    let mut weight2_rng = StdRng::seed_from_u64(0xcc88_0001);
    let weight2 = try_nonlinear_slice_zero_preblock_dims(
        n,
        nondata,
        logical_gates,
        true,
        scratch,
        scratch2,
        &mut weight2_rng,
    )
    .unwrap();

    let quads = (logical_gates - logical_gates / 3) / 2;
    assert_eq!(wide.gates.len(), logical_gates);
    assert_eq!(weight2.gates.len(), logical_gates + 3 * quads);
    assert!(weight2.gates.iter().all(|gate| gate.ctrls.len() <= 2));
    assert_eq!(wide.num_wires, n + nondata);
    assert_eq!(weight2.num_wires, n + nondata);

    for input in 0..(1u64 << n) {
        assert_eq!(eval_u64(&wide.gates, input), input);
        assert_eq!(eval_u64(&weight2.gates, input), input);
    }

    let mut state_rng = StdRng::seed_from_u64(0xcc88_0002);
    let state_mask = (1u64 << (n + nondata)) - 1;
    let dirty_q_mask = (1u64 << scratch) | (1u64 << scratch2);
    for _ in 0..512 {
        let state = rand::RngCore::next_u64(&mut state_rng) & state_mask;
        let decomposed = eval_u64(&weight2.gates, state);
        assert_eq!(
            decomposed,
            eval_u64(&wide.gates, state),
            "dirty-q decomposition changed the preblock function"
        );
        assert_eq!(
            decomposed & dirty_q_mask,
            state & dirty_q_mask,
            "dirty-q decomposition did not restore its scratch wires"
        );
    }
}
