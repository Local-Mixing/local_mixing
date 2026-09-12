use super::*;

fn sample_circuits() -> Vec<Vec<XGate>> {
    vec![
        vec![
            XGate::from_g57([7, 2, 11]),
            XGate::conj(3, [(0, true), (5, false), (9, true)]).unwrap(),
            XGate::x_gate(4),
        ],
        vec![
            XGate::conj(1, [(2, false), (3, false), (6, true)]).unwrap(),
            XGate::cnot(0, 8),
        ],
    ]
}

#[test]
fn mpx1_round_trip_exact() {
    let circuits = sample_circuits();
    let untagged = encode_value(&circuits).unwrap();
    assert_eq!(decode_value(&untagged).unwrap(), circuits);
    let tagged = tag_value(&untagged);
    assert!(is_mpx1(&tagged));
    assert_eq!(decode_value(&tagged).unwrap(), circuits);
    // Merged form: two chains concatenated, tags mid-stream skipped.
    let mut merged = tagged.clone();
    merged.extend_from_slice(&tagged);
    let doubled = decode_value(&merged).unwrap();
    assert_eq!(doubled.len(), circuits.len() * 2);
}

#[test]
fn mpx1_round_trip_randomized() {
    let mut state = 0xdeadbeefcafef00du64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state >> 33
    };
    for case in 0..20_000 {
        let n = 1 + (next() % 12) as usize;
        let mut gates = Vec::with_capacity(n);
        for _ in 0..n {
            let target = (next() % 64) as u16;
            let k = (next() % 4) as usize;
            let mut lits: Vec<(u16, bool)> = Vec::new();
            for _ in 0..k {
                let mut w = (next() % 64) as u16;
                while w == target || lits.iter().any(|&(lw, _)| lw == w) {
                    w = (next() % 64) as u16;
                }
                lits.push((w, next() & 1 == 1));
            }
            if next() % 4 == 0 && !lits.is_empty() {
                // g57-style complemented gate on the same literal set
                let mut ctrls: Lits = lits.iter().copied().collect();
                sort_lits(&mut ctrls);
                gates.push(XGate {
                    target,
                    comp: true,
                    ctrls,
                });
            } else {
                gates.push(XGate::conj(target, lits).unwrap());
            }
        }
        let circuits = vec![gates];
        let enc = encode_value(&circuits).unwrap();
        assert_eq!(decode_value(&enc).unwrap(), circuits, "case {case}");
    }
}

#[test]
fn legacy_percent3_decoder_declines_tagged_values() {
    // The exact loop shape used by db_replace::decode_value,
    // regular::validate_value_chain and friends: read [len], require
    // len % 3 == 0, else break. A tagged MPX1 value must yield ZERO
    // circuits (first chunk len = 2), never a misparse.
    fn legacy_walk(value: &[u8]) -> usize {
        let mut count = 0usize;
        let mut i = 0usize;
        while i < value.len() {
            let len = value[i] as usize;
            if len == 0 || len % 3 != 0 || i + 1 + len > value.len() {
                break;
            }
            count += 1;
            i += 1 + len;
        }
        count
    }
    let tagged = tag_value(&encode_value(&sample_circuits()).unwrap());
    assert_eq!(legacy_walk(&tagged), 0);
    // And the tag byte choice matters: 0xFF's len (255) passes % 3 == 0.
    assert_eq!(255 % 3, 0);
    assert_ne!(MPX1_TAG[0] as usize % 3, 0);
}

#[test]
fn mpx1_rejects_corruption() {
    let good = encode_value(&sample_circuits()).unwrap();
    // Truncation
    assert!(decode_value(&good[..good.len() - 1]).is_err());
    // Duplicate control wire inside a record
    let mut dup = vec![0u8];
    let body = [0x02u8, 3, (5 << 1) | 1, (5 << 1) | 1];
    dup[0] = body.len() as u8;
    dup.extend_from_slice(&body);
    assert!(decode_value(&dup).is_err());
    // Control on its own target
    let mut selfc = vec![0u8];
    let body2 = [0x01u8, 5, (5 << 1) | 1];
    selfc[0] = body2.len() as u8;
    selfc.extend_from_slice(&body2);
    assert!(decode_value(&selfc).is_err());
}
