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
        let wide = try_slice_zero_block_dims(n, 0, n, nondata, gates, &mut wide_rng).unwrap();
        let narrow =
            try_nonlinear_slice_zero_preblock_dims(n, nondata, gates, 6, 7, &mut narrow_rng)
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
fn nonlinear_preblock_fan_in_two_preserves_zero_slice_and_dirty_auxiliaries() {
    let (n, nondata, logical_gates) = (8usize, 20usize, 200usize);
    let scratch = n as u16;
    let scratch2 = (n + 1) as u16;
    let mut weight2_rng = StdRng::seed_from_u64(0xcc88_0001);
    let weight2 = try_nonlinear_slice_zero_preblock_dims(
        n,
        nondata,
        logical_gates,
        scratch,
        scratch2,
        &mut weight2_rng,
    )
    .unwrap();

    let quads = (logical_gates - logical_gates / 3) / 2;
    assert_eq!(weight2.gates.len(), logical_gates + 3 * quads);
    assert!(weight2.gates.iter().all(|gate| gate.ctrls.len() <= 2));
    assert_eq!(weight2.num_wires, n + nondata);

    for input in 0..(1u64 << n) {
        assert_eq!(eval_u64(&weight2.gates, input), input);
    }

    let mut state_rng = StdRng::seed_from_u64(0xcc88_0002);
    let state_mask = (1u64 << (n + nondata)) - 1;
    let auxiliary_mask = state_mask ^ ((1u64 << n) - 1);
    for _ in 0..512 {
        let state = rand::RngCore::next_u64(&mut state_rng) & state_mask;
        let decomposed = eval_u64(&weight2.gates, state);
        assert_eq!(
            decomposed & auxiliary_mask,
            state & auxiliary_mask,
            "preblock did not restore its auxiliary wires"
        );
    }
}

#[test]
fn feistel_guard_is_dead_only_on_zero_slice_and_width_bounded() {
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    // Small enough for the exhaustive only-zero-slice-fixes check, wide
    // enough band (6 > acc_size 2) to exercise real rounds.
    let (n, nondata, gates) = (8, 6, 40);
    for seed in [1u64, 7, 99] {
        let mut rng = StdRng::seed_from_u64(seed);
        let block = try_slice_zero_feistel_dims(n, 0, n / 2, nondata, gates, &mut rng).unwrap();
        assert_eq!(block.gates.len(), gates);
        assert!(
            block.gates.iter().all(|g| g.ctrls.len() <= 2),
            "feistel guard must stay within two controls"
        );
        assert!(
            block
                .gates
                .iter()
                .all(|g| (g.target as usize) < n / 2 || (g.target as usize) >= n),
            "data writes must stay in the junk half"
        );
        assert!(slice_preblock_fixes_only_zero_slice(
            &block.gates,
            n,
            nondata
        ));
        // Zero slice: data AND clean band wires must pass through exactly
        // (accumulators were uncomputed; triggers never written).
        let mask = (1u64 << n) - 1;
        for x in [0u64, 0x5a, 0xff, 0x13] {
            let out = crate::circuit::xgate::eval_u64(&block.gates, x & mask);
            assert_eq!(out & mask, x & mask, "zero-slice data pass-through");
        }
    }
}

/// The two-layer opening guard (RC 2026-11-02), checked independently of its
/// own draw-time check, on every state at n = 3 and 4 (band 2n): the
/// identity on A and B exactly when B and the band are zero; every nonzero
/// (B, band) disturbs A for some A; every nonzero band disturbs B for some
/// (A, B); no gate wider than two controls; the size is two prologue gates
/// per helper wire, two gates per layer-1 and three per layer-2 three-control
/// gate.
#[test]
fn two_layer_guard_is_the_identity_exactly_on_the_honest_slice() {
    for n in [3usize, 4] {
        let nondata = 2 * n;
        let total = 2 * n + nondata;
        let a_mask = (1u64 << n) - 1;
        let data_mask = (1u64 << (2 * n)) - 1;
        for seed in 0..4u64 {
            let mut rng = StdRng::seed_from_u64(0x2_1a7e_0000 + seed);
            let guard = two_layer_opening_guard_dims(n, nondata, &mut rng);
            assert_eq!(guard.num_wires, total);
            assert!(guard.gates.iter().all(|g| g.ctrls.len() <= 2), "width <= 2");
            let helpers = (nondata / 16).clamp(1, n.min(nondata - 1));
            let per1 = two_layer_guard_gates_per_layer(n, nondata - helpers);
            let per2 = two_layer_guard_gates_per_layer(n, n);
            assert_eq!(guard.gates.len(), 2 * helpers + 2 * per1 + 3 * per2);
            let mut a_hit = vec![false; 1 << (n + nondata)];
            let mut b_hit = vec![false; 1 << nondata];
            for state in 0..(1u64 << total) {
                let out = eval_u64(&guard.gates, state);
                if state >> n == 0 {
                    assert_eq!(out & data_mask, state & data_mask, "slice moved at n={n}");
                }
                if (out ^ state) & a_mask != 0 {
                    a_hit[(state >> n) as usize] = true;
                }
                if (out ^ state) & data_mask & !a_mask != 0 {
                    b_hit[(state >> (2 * n)) as usize] = true;
                }
            }
            assert!(
                a_hit.iter().skip(1).all(|&h| h),
                "a nonzero key left A alone at n={n}"
            );
            assert!(
                b_hit.iter().skip(1).all(|&h| h),
                "a nonzero band left B alone at n={n}"
            );
        }
    }
}

/// At a production-like width (n = 64, band 128) the guard draws, passes its
/// spot check, and has n * ceil(log2 n) = 384 three-control gates per layer;
/// on the honest slice the key wires of the band stay zero and B comes out
/// zero (the layer-2 helpers are restored), while the band's helper wires
/// hold products of x (layer 1 does not restore them).
#[test]
fn two_layer_guard_at_n64_restores_b_and_leaves_products_on_band_helpers() {
    let (n, nondata) = (64usize, 128usize);
    let mut rng = StdRng::seed_from_u64(0x2_1a7e_0064);
    let guard = two_layer_opening_guard_dims(n, nondata, &mut rng);
    assert_eq!(two_layer_guard_gates_per_layer(n, nondata - 8), 384);
    assert_eq!(guard.gates.len(), 2 * 8 + 2 * 384 + 3 * 384);
    let mut state = vec![0u64; guard.num_wires];
    for lane in state.iter_mut().take(n) {
        *lane = rng.random();
    }
    let input = state.clone();
    for g in &guard.gates {
        g.apply_lanes(&mut state);
    }
    assert_eq!(
        state[..2 * n],
        input[..2 * n],
        "A and B unchanged on the slice"
    );
    let dirty = (2 * n..guard.num_wires).filter(|&w| state[w] != 0).count();
    assert!(
        (1..=8).contains(&dirty),
        "{dirty} band wires dirty (8 helpers)"
    );
}
