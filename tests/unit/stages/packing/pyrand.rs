use super::*;

// Golden vectors recorded with CPython 3.9.6 (docs/STAGE4_RUST_DESIGN.md Appendix A).

fn low128(words: &[u32]) -> u128 {
    words[..4]
        .iter()
        .enumerate()
        .fold(0u128, |a, (i, &w)| a | ((w as u128) << (32 * i)))
}

#[test]
fn mt_first_words_seed_20261004() {
    let mut r = PyRandom::new(20_261_004);
    let got: Vec<u32> = (0..4).map(|_| r.genrand_u32()).collect();
    assert_eq!(
        got,
        vec![749_935_133, 232_581_420, 3_240_862_439, 1_300_416_022]
    );
}

#[test]
fn getrandbits_1024_lane_layout() {
    let mut r = PyRandom::new(20_261_004);
    let w0 = r.getrandbits_lanes::<16>();
    assert_eq!(w0[0], 0x0ddc_e92c_2cb3_1a1d);
    assert_eq!(w0[15], 0x3f68_8f70_a42f_573a);
    assert_eq!(
        (w0[0] as u128) | ((w0[1] as u128) << 64),
        0x4d82_c616_c12b_a2e7_0ddc_e92c_2cb3_1a1d
    );
    let w1 = r.getrandbits_lanes::<16>();
    assert_eq!(w1[0], 0x12e4_aa97_d52e_e91c);
    // the word API agrees with the lane API
    let mut r2 = PyRandom::new(20_261_004);
    let words = r2.getrandbits_words(1024);
    assert_eq!(low128(&words), (w0[0] as u128) | ((w0[1] as u128) << 64));
}

#[test]
fn seed_77_and_1_low128() {
    for (seed, want) in [
        (77u64, 0x3283_3106_536e_95df_40b2_d002_cc92_d33du128),
        (1, 0xcd61_3e30_d8f1_6adf_91b7_584a_2265_b1f5),
    ] {
        let mut r = PyRandom::new(seed);
        assert_eq!(low128(&r.getrandbits_words(1024)), want, "seed {seed}");
    }
}

#[test]
fn multiword_seeds_and_partial_words() {
    // (seed, getrandbits(32), getrandbits(37), getrandbits(1))
    for (seed, a, b, c) in [
        (0u64, 3_626_764_237u32, 104_733_831_102u64, true),
        ((1 << 40) + 5, 2_166_296_868, 36_579_899_196, true),
        ((1 << 63) - 17, 2_562_872_105, 95_216_842_677, true),
    ] {
        let mut r = PyRandom::new(seed);
        assert_eq!(r.getrandbits_words(32), vec![a], "seed {seed}");
        let w = r.getrandbits_words(37);
        assert_eq!((w[0] as u64) | ((w[1] as u64) << 32), b, "seed {seed}");
        assert_eq!(r.getrandbit(), c, "seed {seed}");
    }
}

#[test]
fn getrandbit_is_top_bit() {
    let mut r = PyRandom::new(5);
    let bits: Vec<bool> = (0..8).map(|_| r.getrandbit()).collect();
    assert_eq!(bits, [true, false, true, false, true, true, true, true]);
    let mut r = PyRandom::new(5);
    let tops: Vec<bool> = (0..8).map(|_| r.genrand_u32() >> 31 == 1).collect();
    assert_eq!(bits, tops);
}
