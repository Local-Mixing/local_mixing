use super::*;

#[test]
fn int_hash_matches_cpython() {
    // hash(1 << b) on CPython 3.9.6 (64-bit)
    for (b, h) in [
        (0usize, 1u64),
        (60, 1 << 60),
        (61, 1),
        (62, 2),
        (63, 4),
        (64, 8),
        (122, 1),
        (127, 32),
        (128, 64),
        (255, 2048),
        (256, 4096),
        (511, 8_388_608),
        (1023, 140_737_488_355_328),
    ] {
        assert_eq!(Mono::<16>::bit(b).py_hash(), h, "bit {b}");
    }
    assert_eq!(Mono::<1>(([M61])).py_hash(), 0);
    assert_eq!(u64::MAX.py_hash(), 7);
    assert_eq!(Mono::<2>([u64::MAX, u64::MAX]).py_hash(), 63);
    // random keys (random.Random(9).getrandbits(k))
    for (hex, h) in [
        ("9cfbac6e7687a66e", 2_088_452_442_651_338_354u64),
        ("2fa73207237751aa4462ebfc5f915ef0", 116_104_259_548_474_446),
        (
            "e576b6745180b65386569c803601a5ba50ad38835eddd6ff55",
            1_133_105_483_693_653_597,
        ),
        (
            "b99de255f386825473b7a490f23b2cc4b4174a672b5ebaa061076dc3ba6ace6c0a78250fb339a4769ddcc6f8efb6fbfe8de4ab47558298e214b044d79acd8acd",
            316_245_690_575_230_236,
        ),
        (
            "4448a6a1c5c7d1861674518de3bb41b36bf82959cb01c357b9c7e435396bcb8fac9abb0c3478442b4a8aa593eb40a9b81a070205e323bb2abf00188dca22e4c76237dbe6b03da701c632976a10363c5f972651dafdb119a9ec801bdfdf2965b3819ad93b21e6a46f1c670ea90d243a163cee5e2c2b1e1885283b73a66c2ea417",
            272_599_758_720_603_747,
        ),
    ] {
        assert_eq!(Mono::<16>::from_hex(hex).unwrap().py_hash(), h, "{hex}");
    }
}

#[test]
fn mersenne_fold_equals_u128_mod() {
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    for _ in 0..2000 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let y = x.rotate_left(29) ^ 0x1234_5678;
        let v = ((y as u128) << 64) | x as u128;
        assert_eq!(py_int_hash(&[x, y]), (v % M61 as u128) as u64);
        assert_eq!(py_int_hash(&[x]), x % M61);
    }
}

#[test]
fn ord_matches_python_int_order_multi_limb() {
    let a = Mono::<2>([u64::MAX, 0]);
    let b = Mono::<2>([0, 1]);
    assert!(a < b, "limb 1 is more significant");
    let mut v = vec![b, a, Mono::ZERO, Mono::bit(64), Mono::bit(3)];
    v.sort();
    let hex: Vec<String> = v.iter().map(|m| format!("{m:x}")).collect();
    assert_eq!(
        hex,
        [
            "0",
            "8",
            "ffffffffffffffff",
            "10000000000000000",
            "10000000000000000"
        ]
    );
}

#[test]
fn hex_round_trip_and_bits() {
    let m = Mono::<4>::from_hex("8000000000000000000000000000000000000000000000001").unwrap();
    assert_eq!(m.bits().collect::<Vec<_>>(), vec![0, 195]);
    assert_eq!(
        format!("{m:x}"),
        "8000000000000000000000000000000000000000000000001"
    );
    assert!(Mono::<1>::from_hex("10000000000000000").is_none());
    assert_eq!(mono_words(32), Some(1));
    assert_eq!(mono_words(65), Some(2));
    assert_eq!(mono_words(256), Some(4));
    assert_eq!(mono_words(300), Some(8));
    assert_eq!(mono_words(1024), Some(16));
    assert_eq!(mono_words(1025), None);
}
