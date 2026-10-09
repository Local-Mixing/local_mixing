use super::*;

fn g(t: u16, ctrls: &[u16]) -> XGate {
    XGate::conj(t, ctrls.iter().map(|&c| (c, true))).unwrap()
}

#[test]
fn complement_normalization_and_constants() {
    assert!(normalize(&[0; 16]).is_none());
    assert!(normalize(&[!0; 16]).is_none()); // all ones complements to 0
    let mut v = [0u64; 16];
    v[3] = 5; // sample 0 clear: unchanged
    assert_eq!(normalize(&v), Some(v));
    let mut w = v;
    w[0] = 1; // sample 0 set: complemented
    let n = normalize(&w).unwrap();
    assert_eq!(n[0], !1);
    assert_eq!(n[3], !5);
    assert_eq!(n[0] & 1, 0);
    // a value and its complement share one key
    assert_eq!(value_key(&normalize(&w).unwrap()), value_key(&n));
}

#[test]
fn span_closes_on_every_write() {
    // A births w1 ^= w0. In B the same gate opens a span on wire 1; a NOT on
    // wire 1 (same normalized value) closes it and reopens it as pending.
    let fp = inputs(2, 20_261_004);
    let ai = index_a(&[g(1, &[0])], &fp, 2);
    assert_eq!(ai.distinct(), 1);
    let sp = spans_b(&[g(1, &[0]), XGate::x_gate(1)], None, &fp, 2, 2, &ai);
    assert_eq!(sp.spans, vec![(0, 1, 0)]);
    assert_eq!(sp.forced, vec![0]);
    assert_eq!(sp.agree, 1); // wire 0 agrees, wire 1 is complemented
    // a write of an unknown value closes the span and leaves nothing pending
    let sp = spans_b(&[g(1, &[0]), g(1, &[0])], None, &fp, 2, 2, &ai);
    assert_eq!(sp.spans, vec![(0, 1, 0)]);
    assert!(sp.forced.is_empty());
    assert_eq!(sp.agree, 1); // B undoes the CNOT on wire 1
    // walking an explicit order gives positions in that order
    let b = [XGate::x_gate(0), g(1, &[0])];
    let sp = spans_b(&b, Some(&[1, 0]), &fp, 2, 2, &ai);
    assert_eq!(sp.forced, vec![0]);
}

#[test]
fn first_birth_is_value_id() {
    let fp = inputs(3, 20_261_004);
    // gate 1 rebirths gate 0's value (complemented); gate 2 births a new one
    let a = [g(1, &[0]), XGate::x_gate(1), g(2, &[1])];
    let ai = index_a(&a, &fp, 3);
    assert_eq!(ai.distinct(), 2);
    assert_eq!(ai.a_wire, vec![1, u16::MAX, 2]);
    let k0 = value_key(
        &normalize(&{
            let mut st = fp.clone();
            a[0].apply_lanes16(&mut st);
            st[1]
        })
        .unwrap(),
    );
    assert_eq!(ai.index[&k0], 0);
}

#[test]
fn wires_beyond_width_start_at_zero() {
    // A is 1 wire wide: wire 1 starts at 0 in A, random in B (nb = 2)
    let fp = inputs(2, 20_261_004);
    let ai = index_a(&[g(1, &[0])], &fp, 1);
    let st1 = {
        let mut st = vec![fp[0], [0; 16]];
        g(1, &[0]).apply_lanes16(&mut st);
        st[1]
    };
    assert_eq!(st1, fp[0]); // w1 = 0 ^ w0
    let sp = spans_b(&[g(1, &[0])], None, &fp, 2, 1, &ai);
    assert!(sp.spans.is_empty());
    assert!(sp.forced.is_empty()); // B's w1 = r1 ^ w0, not A's value
    assert_eq!(sp.agree, 1);
}
