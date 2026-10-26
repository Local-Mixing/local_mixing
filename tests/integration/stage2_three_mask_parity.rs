//! `stage2_three_mask_parity`: the ported three-mask stage 2 reproduces the
//! server generator (`stage2_compute_pilot`, built against this crate at
//! 4a85b09f) byte for byte. The generator ran n=32, |C|=|D|=128, seed 1,
//! slice s=160 passed both as the sandwich's slice size and as gen's
//! `slice_gates`, `--band-wires 128` and the "modified / v2" flags (the
//! THREE_MASK_* defaults); its `pilot_stage2.mpmct` and `sandwich.mpmct`
//! (written with this crate's `write_mpmct`) hashed as pinned below.
use std::fs;
use std::process::Command;

const STAGE2_SHA256: &str = "e9f296d0470ccca8c982720ca26a5351eac896f3b77d5178bdb6a17eb7580bc2";
const SANDWICH_SHA256: &str = "024fc1164bcc00465e879dd42de9344b3653c498d4e3944d07bfdbb7d882dee5";

/// SHA-256 (FIPS 180-4), self-contained: the dependency set has no SHA
/// implementation and the server digests are SHA-256.
fn sha256(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut message = data.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    for block in message.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (word, add) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *word = word.wrapping_add(add);
        }
    }
    h.iter().map(|word| format!("{word:08x}")).collect()
}

#[test]
fn sha256_matches_the_standard_vectors() {
    assert_eq!(
        sha256(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        sha256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}

#[test]
fn stage2_three_mask_matches_the_server_generator() {
    let dir = std::env::temp_dir().join(format!("stage2_three_mask_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let out = dir.join("tdp.mpmct1");
    let o = Command::new(env!("CARGO_BIN_EXE_gen_sandwich_gadget"))
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("THREE_MASK_BAND_WIRES", "128")
        .arg(&out)
        .args([
            "32",
            "128",
            "128",
            "160",
            "1",
            "160",
            "1",
            "1",
            "1",
            "three-mask",
        ])
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let stdout = String::from_utf8_lossy(&o.stdout);
    assert!(stdout.contains("gadgetization_mode=three-mask"), "{stdout}");
    // The generator's stats.json: compute_start 871 = 160 guard + 711 seed,
    // reseed 727, pilot_compute_gates 101847, band_ingress_gates 320,
    // seeded_band_wires 64.
    assert!(
        stdout
            .contains("[gen] three-mask: band=128 wires (seeded prefix 64), compute 101847 gates"),
        "{stdout}"
    );
    assert!(stdout.contains("+ 320 ingress)"), "{stdout}");
    assert!(
        stdout.contains("711 band-seed + 727 band-reseed + 160 slice-guard gates each side"),
        "{stdout}"
    );
    assert!(stdout.contains("[gen] verify PASSED"), "{stdout}");
    assert!(stdout.contains("[gen] reverse verify PASSED"), "{stdout}");
    let stage2 = fs::read(&out).unwrap();
    assert!(stage2.starts_with(b"mpmct1 192 103605\n"));
    assert_eq!(
        sha256(&stage2),
        STAGE2_SHA256,
        "stage-2 bytes differ from the server generator"
    );
    let mut sandwich_path = out.as_os_str().to_owned();
    sandwich_path.push(".sandwich.mpmct1");
    let sandwich = fs::read(&sandwich_path).unwrap();
    assert!(sandwich.starts_with(b"mpmct1 64 608\n"));
    assert_eq!(
        sha256(&sandwich),
        SANDWICH_SHA256,
        "sandwich bytes differ from the server run"
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn three_mask_rejects_outer_block_overrides_and_bad_knobs() {
    let dir = std::env::temp_dir().join(format!("stage2_three_mask_bad_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    for (key, value, expected) in [
        (
            "EMBEDDED_MASKING_PREMASK",
            "1",
            "three-mask fixes the outer blocks to preset L8; unset EMBEDDED_MASKING_PREMASK",
        ),
        (
            "SLICE_GUARD_LEGACY",
            "1",
            "three-mask fixes the outer blocks to preset L8; unset SLICE_GUARD_LEGACY",
        ),
        (
            "THREE_MASK_BANKS",
            "eight",
            "THREE_MASK_BANKS must be a non-negative integer",
        ),
        (
            "THREE_MASK_UNIQUE_COVER",
            "yes",
            "THREE_MASK_UNIQUE_COVER must be 0, 1, false or true",
        ),
        (
            "THREE_MASK_BAND_WIRE",
            "128",
            "unknown three-mask control(s): THREE_MASK_BAND_WIRE",
        ),
    ] {
        let out = dir.join(format!("{key}.mpmct1"));
        let o = Command::new(env!("CARGO_BIN_EXE_gen_sandwich_gadget"))
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env(key, value)
            .arg(&out)
            .args([
                "8",
                "16",
                "16",
                "24",
                "1",
                "160",
                "1",
                "1",
                "1",
                "three-mask",
            ])
            .output()
            .unwrap();
        assert_eq!(o.status.code(), Some(2), "{key}");
        let stderr = String::from_utf8_lossy(&o.stderr);
        assert!(stderr.contains(expected), "{key}: {stderr}");
        assert!(!out.exists(), "{key}: no artifact may be written");
    }
    // THREE_MASK_* under embedded-masking is refused the same way.
    let out = dir.join("foreign.mpmct1");
    let o = Command::new(env!("CARGO_BIN_EXE_gen_sandwich_gadget"))
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("THREE_MASK_BANKS", "8")
        .arg(&out)
        .args([
            "8",
            "16",
            "16",
            "24",
            "1",
            "160",
            "1",
            "1",
            "1",
            "embedded-masking",
        ])
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&o.stderr)
            .contains("THREE_MASK_* controls require the three-mask mode; unset THREE_MASK_BANKS")
    );
    let _ = fs::remove_dir_all(dir);
}
