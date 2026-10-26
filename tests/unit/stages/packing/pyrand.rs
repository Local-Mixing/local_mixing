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

#[test]
fn cpython_float_and_choice_primitives_golden() {
    // random(), _randbelow(n), choice(range(64)), randrange(2) and
    // choices(population, weights=...) recorded with CPython 3.9.6 (the
    // draw sequences of these methods are unchanged through 3.12). Seed
    // 20261020091 keys the generator with two 32-bit words; the first
    // getrandbits(32) words pin that seeding.
    struct Golden {
        seed: u64,
        words: [u32; 4],
        random: [f64; 3],
        randbelow: [u64; 7],
        choice64: [usize; 6],
        randrange2: [usize; 16],
        choices: [u32; 12],
        uniform: [u32; 12],
    }
    const RANDBELOW_ARGS: [u64; 7] = [2, 7, 64, 1000, 100_000, 1, 4_294_967_297];
    for g in [
        Golden {
            seed: 5,
            words: [2_675_342_405, 1_097_127_993, 3_185_950_873, 1_539_898_300],
            random: [0.6229016948897019, 0.7417869892607294, 0.7951935655656966],
            randbelow: [1, 5, 45, 814, 90_498, 0, 3_609_267_711],
            choice64: [32, 45, 3, 59, 31, 6],
            randrange2: [1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
            choices: [10, 20, 20, 40, 20, 40, 10, 10, 40, 10, 40, 10],
            uniform: [1, 2, 2, 2, 2, 2, 0, 1, 2, 1, 2, 0],
        },
        Golden {
            seed: 20_261_020_091,
            words: [1_625_566_369, 1_683_346_776, 1_917_009_253, 3_057_331_068],
            random: [0.37848166668366423, 0.44633850240588446, 0.7409618758812494],
            randbelow: [1, 3, 57, 728, 97_119, 0, 2_912_959_170],
            choice64: [48, 50, 57, 37, 54, 59],
            randrange2: [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
            choices: [10, 10, 20, 10, 20, 10, 40, 10, 20, 10, 10, 20],
            uniform: [1, 1, 2, 0, 2, 1, 2, 0, 2, 1, 1, 2],
        },
    ] {
        let seed = g.seed;
        let mut r = PyRandom::new(seed);
        let words: Vec<u32> = (0..4).map(|_| r.getrandbits_u64(32) as u32).collect();
        assert_eq!(words, g.words, "seed {seed}: getrandbits(32)");

        let mut r = PyRandom::new(seed);
        let random: Vec<f64> = (0..3).map(|_| r.random()).collect();
        assert_eq!(random, g.random, "seed {seed}: random()");

        let mut r = PyRandom::new(seed);
        let randbelow: Vec<u64> = RANDBELOW_ARGS.iter().map(|&n| r.randbelow(n)).collect();
        assert_eq!(randbelow, g.randbelow, "seed {seed}: _randbelow");

        let mut r = PyRandom::new(seed);
        let population: Vec<usize> = (0..64).collect();
        let choice: Vec<usize> = (0..6).map(|_| r.choice(&population)).collect();
        assert_eq!(choice, g.choice64, "seed {seed}: choice");

        let mut r = PyRandom::new(seed);
        let randrange: Vec<usize> = (0..16).map(|_| r.randrange(2)).collect();
        assert_eq!(randrange, g.randrange2, "seed {seed}: randrange(2)");

        let mut r = PyRandom::new(seed);
        let choices: Vec<u32> = (0..12)
            .map(|_| r.choices_one(&[10, 20, 30, 40], &[100, 20, 1, 20]))
            .collect();
        assert_eq!(choices, g.choices, "seed {seed}: choices(weights)");

        let mut r = PyRandom::new(seed);
        let uniform: Vec<u32> = (0..12)
            .map(|_| r.choices_one(&[0, 1, 2], &[1, 1, 1]))
            .collect();
        assert_eq!(uniform, g.uniform, "seed {seed}: choices(uniform weights)");
    }
}
