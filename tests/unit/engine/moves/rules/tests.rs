// The hoisted constant lane basis in verify_rewrite must match the
// original per-bit construction for every wire index and batch base.
#[test]
fn opt_equiv_lane_basis_matches_bitwise_construction() {
    const LANE: [u64; 6] = [
        0xAAAA_AAAA_AAAA_AAAA,
        0xCCCC_CCCC_CCCC_CCCC,
        0xF0F0_F0F0_F0F0_F0F0,
        0xFF00_FF00_FF00_FF00,
        0xFFFF_0000_FFFF_0000,
        0xFFFF_FFFF_0000_0000,
    ];
    for k in 1usize..=12 {
        let total = 1u64 << k;
        let mut v = 0u64;
        while v < total {
            for i in 0..k {
                let mut acc = 0u64;
                for l in 0..64u64 {
                    if ((v + l) >> i) & 1 == 1 {
                        acc |= 1 << l;
                    }
                }
                let fast = if i < 6 {
                    LANE[i]
                } else {
                    0u64.wrapping_sub((v >> i) & 1)
                };
                assert_eq!(acc, fast, "k={k} v={v} i={i}");
            }
            v += 64;
        }
    }
}
