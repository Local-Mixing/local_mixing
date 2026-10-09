use super::*;
use crate::stages::packing::pyrand::PyRandom;

// Vectors recorded with CPython 3.9.6 hashlib.

#[test]
fn rfc7693_vectors() {
    assert_eq!(
        hex(&blake2b(b"abc", 64)),
        "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d1\
         7d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923"
    );
    assert_eq!(hex(&blake2b(b"", 16)), "cae66941d9efbd404e4d88758ea67670");
    assert_eq!(
        hex(&blake2b(b"abc", 16)),
        "cf4ab791c62b8d2b2109c90275287816"
    );
    let r: Vec<u8> = (0..=255u8).collect();
    let r3: Vec<u8> = r.iter().chain(&r).chain(&r).copied().collect();
    assert_eq!(hex(&blake2b(&r3, 16)), "e0892860f9fc0da273b9db57e857c19a");
    assert_eq!(
        hex(&blake2b(&r[..128], 16)),
        "a74787004ef589e31149183900d0294a"
    );
    assert_eq!(
        hex(&blake2b(&r[..129], 32)),
        "f7f3c46ba2564ff4c4c162da1f5b605f9f1c4aa6a20652a9f9a337c1a2f5b9c9"
    );
}

#[test]
fn incremental_equals_one_shot() {
    let data: Vec<u8> = (0..1000u32).map(|i| (i * 7 + 3) as u8).collect();
    for split in [0, 1, 127, 128, 129, 256, 999, 1000] {
        let mut h = Blake2b::new(16);
        h.update(&data[..split]);
        h.update(&data[split..]);
        assert_eq!(h.finalize(), blake2b(&data, 16), "split {split}");
    }
}

#[test]
fn wire0_value_key_matches_hashlib() {
    // Appendix A: blake2b(getrandbits(1024).to_bytes(128,'little'), digest_size=16)
    // for wire 0 of Random(20261004).
    let mut r = PyRandom::new(20_261_004);
    let w0 = r.getrandbits_lanes::<16>();
    assert_eq!(
        hex(&blake2b16_lanes(&w0)),
        "10443c04d722b9ea2873b4564780ebb1"
    );
    let mut bytes = Vec::new();
    for l in &w0 {
        bytes.extend_from_slice(&l.to_le_bytes());
    }
    assert_eq!(blake2b16_lanes(&w0).to_vec(), blake2b(&bytes, 16));
}
