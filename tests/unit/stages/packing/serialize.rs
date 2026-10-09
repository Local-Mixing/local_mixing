use super::*;
use crate::stages::packing::oracle_io::parse_oracle_mpmct;
use crate::stages::packing::pyrand::PyRandom;

fn body_of(text: &str, bounds: &[i64]) -> (Vec<u8>, EmitReport, Vec<XGate>, usize) {
    let c = parse_oracle_mpmct(text).unwrap();
    let mut out = Vec::new();
    let r = encode_body(&c.gates, bounds, c.header_m, c.computed_n, 3, &mut out).unwrap();
    (out, r, c.gates, c.computed_n)
}

#[test]
fn crc32_matches_zlib_vectors() {
    assert_eq!(crc32_ieee(b""), 0);
    assert_eq!(crc32_ieee(b"123456789"), 0xCBF4_3926);
    assert_eq!(
        crc32_ieee(b"The quick brown fox jumps over the lazy dog"),
        0x414F_A339
    );
    let mut c = Crc32::default();
    c.update(b"12345");
    c.update(b"6789");
    assert_eq!(c.finish(), 0xCBF4_3926);
}

#[test]
fn blob_small_example_and_constant_monomial_first() {
    // one window: w1 ^= NOT x0 (= x1 + x0 + 1)
    let (body, r, _, _) = body_of("mpmct1 2 1\n1 0 1 0 0\n", &[]);
    // <I 17> <HH f=2 c=1> <H 0><H 1> <HI 1 3> masks 0x00 0x01 0x02
    let want: Vec<u8> = [
        &17u32.to_le_bytes()[..],
        &2u16.to_le_bytes(),
        &1u16.to_le_bytes(),
        &0u16.to_le_bytes(),
        &1u16.to_le_bytes(),
        &1u16.to_le_bytes(),
        &3u32.to_le_bytes(),
        &[0x00, 0x01, 0x02],
    ]
    .concat();
    assert_eq!(body, want);
    assert_eq!((r.windows, r.monomials, r.body_bytes), (1, 3, 21));
    assert_eq!(r.body_crc32, crc32_ieee(&want));
}

#[test]
fn empty_and_duplicate_windows() {
    // bounds 1,1: windows [0,1) [1,1) [1,2); the second is empty (4-byte blob)
    let (body, r, _, _) = body_of("mpmct1 3 2\n1 0 1 0 1\n1 0 1 0 1\n", &[1, 1]);
    assert_eq!(r.windows, 3);
    assert_eq!(r.empty_windows, 1);
    let d = decode(
        &body,
        Format::Raw {
            source_wires: 1,
            state_wires: 3,
        },
    )
    .unwrap();
    assert_eq!(d.window_count(), 3);
    assert_eq!(d.window(1).support.len(), 0);
    assert_eq!(d.window(1).changed_len(), 0);
}

#[test]
fn container_header_fields_round_trip() {
    let h = ContainerHeader {
        source_wires: 64,
        state_wires: 256,
        io_layout: IO_LAYOUT_CLASSIC_ZERO_SLICE,
        window_count: 2776,
        body_bytes: 26_036_498,
        body_crc32: 0xDEAD_BEEF,
    };
    let b = h.to_bytes();
    assert_eq!(&b[0..8], b"LMTDPANF");
    assert_eq!(u16::from_le_bytes([b[8], b[9]]), 1);
    assert_eq!(u16::from_le_bytes([b[10], b[11]]), 48);
    assert_eq!(&b[12..16], &[0, 0, 0, 0]);
    assert_eq!(
        u32::from_le_bytes(b[44..48].try_into().unwrap()),
        crc32_ieee(&b[..44])
    );
    assert_eq!(ContainerHeader::parse(&b).unwrap(), h);
    for (off, why) in [
        (0usize, "magic"),
        (8, "version"),
        (12, "flags"),
        (20, "state"),
        (44, "hcrc"),
    ] {
        let mut x = b;
        x[off] ^= 0x40;
        assert!(ContainerHeader::parse(&x).is_err(), "{why}");
    }
}

#[test]
fn decode_requires_explicit_format() {
    let (body, r, _, _) = body_of("mpmct1 4 2\n2 0 1 0 1\n3 1 1 2 0\n", &[1]);
    let h = ContainerHeader {
        source_wires: 1,
        state_wires: 4,
        io_layout: IO_LAYOUT_CLASSIC_ZERO_SLICE,
        window_count: r.windows as u32,
        body_bytes: body.len() as u64,
        body_crc32: r.body_crc32,
    };
    let mut cont = h.to_bytes().to_vec();
    cont.extend_from_slice(&body);
    let raw = Format::Raw {
        source_wires: 1,
        state_wires: 4,
    };
    assert!(decode(&cont, Format::Container).is_ok());
    assert!(decode(&body, raw).is_ok());
    assert!(decode(&cont, raw).is_err(), "raw on a container");
    assert!(
        decode(&body, Format::Container).is_err(),
        "container on a raw body"
    );
    let mut bad = cont.clone();
    let last = bad.len() - 1;
    bad[last] ^= 1;
    assert!(decode(&bad, Format::Container).is_err(), "body CRC");
}

#[test]
fn decoder_rejects_each_invariant_violation() {
    let blob = |f: u16, sup: &[u16], ch: &[(u16, &[u8])], fb: usize| -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&f.to_le_bytes());
        b.extend_from_slice(&(ch.len() as u16).to_le_bytes());
        for w in sup {
            b.extend_from_slice(&w.to_le_bytes());
        }
        for (w, masks) in ch {
            b.extend_from_slice(&w.to_le_bytes());
            b.extend_from_slice(&((masks.len() / fb) as u32).to_le_bytes());
            b.extend_from_slice(masks);
        }
        let mut out = (b.len() as u32).to_le_bytes().to_vec();
        out.extend(b);
        out
    };
    let raw = Format::Raw {
        source_wires: 2,
        state_wires: 8,
    };
    assert!(decode(&blob(2, &[0, 1], &[(1, &[0, 1, 2])], 1), raw).is_ok());
    for (body, why) in [
        (blob(2, &[1, 0], &[(1, &[0, 1, 2])], 1), "support order"),
        (
            blob(2, &[0, 9], &[(1, &[0, 1, 2])], 1),
            "support beyond state",
        ),
        (blob(2, &[0, 1], &[(1, &[0, 2, 1])], 1), "mask order"),
        (blob(2, &[0, 1], &[(1, &[0, 1, 1])], 1), "duplicate mask"),
        (blob(2, &[0, 1], &[(1, &[0, 1, 4])], 1), "mask bit >= f"),
        (blob(2, &[0, 1], &[(1, &[0, 1])], 1), "support unused"),
        (
            blob(2, &[0, 1], &[(3, &[1, 2]), (1, &[1, 2])], 1),
            "changed order",
        ),
        (blob(0, &[], &[(1, &[0])], 1), "f = 0 with c > 0"),
        (blob(1, &[0], &[], 1), "c = 0 with f > 0"),
        (blob(2, &[0, 1], &[(9, &[1, 2])], 1), "changed beyond state"),
    ] {
        assert!(decode(&body, raw).is_err(), "{why}");
    }
    let mut trunc = blob(2, &[0, 1], &[(1, &[0, 1, 2])], 1);
    trunc.pop();
    assert!(decode(&trunc, raw).is_err(), "truncated");
    let mut extra = blob(2, &[0, 1], &[(1, &[0, 1, 2])], 1);
    extra[0] += 1;
    extra.push(0);
    assert!(decode(&extra, raw).is_err(), "trailing bytes in a blob");
}

#[test]
fn decode_eval_equals_gates() {
    let mut r = PyRandom::new(7);
    let nw = 40usize;
    let mut text = String::new();
    let m = 600;
    text.push_str(&format!("mpmct1 {nw} {m}\n"));
    for _ in 0..m {
        let base = (r.genrand_u32() % 30) as u16;
        let t = base + (r.genrand_u32() % 10) as u16;
        let mut ctrls: Vec<u16> = (0..(r.genrand_u32() % 4))
            .map(|_| base + (r.genrand_u32() % 10) as u16)
            .filter(|&c| c != t)
            .collect();
        ctrls.sort_unstable();
        ctrls.dedup();
        text.push_str(&format!("{t} {} {}", r.genrand_u32() & 1, ctrls.len()));
        for c in ctrls {
            text.push_str(&format!(" {c} {}", r.genrand_u32() & 1));
        }
        text.push('\n');
    }
    let bounds: Vec<i64> = (1..30).map(|i| i * 20).collect();
    let (body, _, gates, n) = body_of(&text, &bounds);
    let d = decode(
        &body,
        Format::Raw {
            source_wires: 1,
            state_wires: n as u32,
        },
    )
    .unwrap();
    let mut st: Vec<[u64; 16]> = (0..n).map(|_| r.getrandbits_lanes::<16>()).collect();
    let mut sg = st.clone();
    for g in &gates {
        g.apply_lanes16(&mut sg);
    }
    d.eval_lanes16(&mut st);
    assert_eq!(st, sg);
    // single-state evaluator agrees
    let mut limb = vec![r.genrand_u32() as u64 | ((r.genrand_u32() as u64) << 32)];
    let mut lg = limb.clone();
    for g in &gates {
        g.apply_limbs(&mut lg);
    }
    d.eval_limbs(&mut limb);
    assert_eq!(limb, lg);
}
