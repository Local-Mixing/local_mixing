use super::*;
use crate::circuit::xgate::XGate;
use crate::stages::adaptive::types::Source;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::time::Instant;

fn tof(t: u16, a: u16, b: u16) -> XGate {
    XGate::conj(t, [(a, true), (b, true)]).unwrap()
}

// C on 5 wires:
//   g0: w1 = x1 ^ x0                       linear in x;  w1 = y1 ^ y3 y4 in y
//   g1: w2 = x2 ^ x0 ^ x0 x1               quadratic;    w2 = y2 ^ y4
//   g2: w3 = x3 ^ x0 (1^x1) (1^x2)         cubic;        y3 ^ y1 y2 ^ y1 y4 ^ y2 y3 y4 ^ y3 y4 (cubic)
//   g3: w3 ^= w1 w2                        cubic;        = y3 (output of wire 3)
//   g4: w1 ^= w3 x4                        degree 4;     = y1 (output of wire 1)
//   g5: w2 ^= x4                           quadratic;    = y2 (output of wire 2)
fn hand_c() -> Vec<XGate> {
    vec![
        XGate::cnot(1, 0),
        tof(2, 0, 1),
        XGate::conj(3, [(0, true), (1, true), (2, true)]).unwrap(),
        tof(3, 1, 2),
        tof(1, 3, 4),
        XGate::cnot(2, 4),
    ]
}

#[test]
fn hand_built_births_classify_on_both_sides() {
    let c = hand_c();
    let b = classify(&c, 5, 1024);
    let rows: Vec<(usize, u16, bool, bool, bool)> = b
        .rows
        .iter()
        .map(|r| {
            (
                r.source_gate,
                r.source_wire,
                r.input_quadratic,
                r.output_quadratic,
                r.is_output,
            )
        })
        .collect();
    assert_eq!(
        rows,
        vec![
            (0, 1, true, true, false),
            (1, 2, true, true, false),
            (2, 3, false, false, false),
            (3, 3, false, true, true),
            (4, 1, false, true, true),
            (5, 2, true, true, true),
        ]
    );
    assert_eq!(b.excluded, [0, 1, 3, 4, 5].into_iter().collect());
    assert_eq!(b.probe_inputs, 1024);
    let s = Source::new(c, 5, b.excluded.clone());
    for r in &b.rows {
        assert_eq!(s.outputs.contains(&r.source_gate), r.is_output);
    }
    assert_eq!(s.deeper(), [2].into_iter().collect());
}

#[test]
fn boundary_file_round_trip() {
    let b = classify(&hand_c(), 5, 256);
    let path =
        std::env::temp_dir().join(format!("lm_adaptive_boundary_{}.tsv", std::process::id()));
    write_file(&b, &path).unwrap();
    let back = read_file(&path).unwrap();
    assert_eq!(back, b);
    std::fs::remove_file(&path).unwrap();
    let text = "# probe_inputs=7\nsource_gate\tsource_wire\tinput_quadratic\toutput_quadratic\tis_output\n3\t1\ttrue\tfalse\ttrue\n";
    std::fs::write(&path, text).unwrap();
    let parsed = read_file(&path).unwrap();
    assert_eq!(parsed.probe_inputs, 7);
    assert_eq!(parsed.rows.len(), 1);
    assert_eq!(parsed.excluded, [3].into_iter().collect());
    std::fs::remove_file(&path).unwrap();
}

/// A random n = 64 source of 2000 gates (1..3 controls); prints the time.
#[test]
fn random_n64_source_classifies_in_reasonable_time() {
    let mut rng = StdRng::seed_from_u64(64);
    let n = 64u16;
    let c: Vec<XGate> = (0..2000)
        .map(|_| {
            let t = rng.random_range(0..n);
            let k = rng.random_range(1..=3usize);
            let mut lits: Vec<(u16, bool)> = Vec::new();
            while lits.len() < k {
                let w = rng.random_range(0..n);
                if w != t && !lits.iter().any(|&(x, _)| x == w) {
                    lits.push((w, rng.random::<bool>()));
                }
            }
            let mut g = XGate::conj(t, lits).unwrap();
            g.comp = rng.random::<bool>();
            g
        })
        .collect();
    let t0 = Instant::now();
    let b = classify(&c, 64, DEFAULT_PROBE_INPUTS);
    let secs = t0.elapsed().as_secs_f64();
    eprintln!(
        "boundary n=64 |C|=2000 probes={DEFAULT_PROBE_INPUTS}: {secs:.3} s, excluded {}",
        b.excluded.len()
    );
    assert_eq!(b.rows.len(), 2000);
    // The first birth is x_t plus a product of its controls: quadratic iff
    // at most two controls. The last gate's birth is an output coordinate.
    assert_eq!(b.rows[0].input_quadratic, c[0].ctrls.len() <= 2);
    assert!(b.rows[1999].output_quadratic);
    assert!(
        b.rows
            .iter()
            .any(|r| !r.input_quadratic && !r.output_quadratic)
    );
    assert!(secs < 120.0, "classification took {secs} s");
}
