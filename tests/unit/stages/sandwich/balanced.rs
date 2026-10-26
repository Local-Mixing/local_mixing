use super::*;
use crate::stages::sandwich::{SourceConstruction, prepare_source};
use xxhash_rust::xxh3::xxh3_128;

/// The Python builder's output-file body: one `t x y` line per gate, no
/// header.
fn triples_text(c: &CircuitSeq) -> String {
    c.gates
        .iter()
        .map(|[t, x, y]| format!("{t} {x} {y}\n"))
        .collect()
}

#[test]
fn reference_balanced_c_is_reproduced_gate_for_gate() {
    // generate(64, 1000, 20261020091) mapped through g57 is the campaign's
    // reference balanced C. Its triples text has
    // sha256 2c7a9e0592ba2c75929b64c99f669f4b817b718b703e15e597d639c61148bd5c
    // (recorded with shasum); the crate carries no sha256, so the test pins
    // the xxh3-128 of the same text, taken from a build whose text matched
    // that sha256.
    let c = balanced_source(64, 1000, 20_261_020_091).unwrap();
    assert_eq!(c.gates.len(), 1000);
    let text = triples_text(&c);
    assert_eq!(
        xxh3_128(text.as_bytes()),
        0xc5b1_b466_7d52_7375_6dde_8be2_75ed_c138
    );
}

/// An independent recount of the control reads of every value (input wire
/// `w` is value `w`, gate `i` writes value `n + i`) and the values alive at
/// the end.
fn reads_per_value(n: usize, c: &CircuitSeq) -> (Vec<usize>, Vec<usize>) {
    let mut live: Vec<usize> = (0..n).collect();
    let mut reads = vec![0usize; n + c.gates.len()];
    for (i, [t, x, y]) in c.gates.iter().enumerate() {
        reads[live[*x as usize]] += 1;
        reads[live[*y as usize]] += 1;
        live[*t as usize] = n + i;
    }
    (reads, live)
}

#[test]
fn balanced_sources_pass_the_audit_across_sizes() {
    for (n, m, seed) in [
        (16usize, 64usize, 1u64),
        (8, 24, 3),
        (16, 64, 1 << 40),
        (64, 1000, 20_261_026),
        (128, 3000, 7),
    ] {
        let c = balanced_source(n, m, seed).unwrap_or_else(|e| panic!("n={n} m={m}: {e}"));
        assert_eq!(c.gates.len(), m);
        for g in &c.gates {
            assert!(
                g.iter().all(|&w| (w as usize) < n),
                "n={n}: wire outside 0..n"
            );
            assert!(
                g[0] != g[1] && g[0] != g[2] && g[1] != g[2],
                "n={n}: repeated wire"
            );
        }
        let a = audit_balanced(n, &c).unwrap_or_else(|e| panic!("n={n} m={m}: {e}"));
        assert_eq!((a.wires, a.gates), (n, m));
        assert!(a.maximum_control_fanout <= 2);
        assert!(a.maximum_write_spread <= 1);
        // Targets always carry the minimum write count: the final counts
        // take at most two adjacent values.
        match a.final_writes[..] {
            [_] => {}
            [lo, hi] => assert_eq!(hi, lo + 1, "n={n}: final writes {:?}", a.final_writes),
            _ => panic!("n={n}: final writes {:?}", a.final_writes),
        }
        // Every gate-born value is an output (one per written wire) or an
        // internal birth.
        let written: std::collections::HashSet<u16> = c.gates.iter().map(|g| g[0]).collect();
        assert_eq!(a.internal_births, m - written.len());
        // Independent recount: internal values are read exactly twice, no
        // value (inputs and outputs included) more than twice.
        let (reads, live) = reads_per_value(n, &c);
        assert!(
            reads.iter().all(|&r| r <= 2),
            "n={n}: a value read more than twice"
        );
        assert_eq!(reads.iter().max(), Some(&a.maximum_control_fanout));
        for (v, &r) in reads.iter().enumerate().skip(n) {
            if !live.contains(&v) {
                assert_eq!(r, 2, "n={n}: internal value {v} read {r} times");
            }
        }
    }
}

#[test]
fn balanced_source_is_deterministic_in_the_seed() {
    let a = balanced_source(16, 64, 9).unwrap();
    assert_eq!(a, balanced_source(16, 64, 9).unwrap());
    assert_ne!(a, balanced_source(16, 64, 10).unwrap());
    // The builder's edge cases: no gates is a complete (empty) circuit, and
    // fewer than three wires exhaust the attempt budget like the Python
    // builder does.
    assert_eq!(balanced_source(8, 0, 1).unwrap().gates.len(), 0);
    assert_eq!(
        balanced_source(2, 5, 1).unwrap_err(),
        "constrained generation did not find a full circuit"
    );
}

#[test]
fn prepare_source_dispatches_on_the_construction() {
    let (n, m, seed) = (16, 64, 41);
    // The random path is unchanged: C comes from the fastrand stream seeded
    // with `seed`, and it is the default construction.
    fastrand::seed(seed);
    let expected = crate::circuit::random_circuit(n, m);
    assert_eq!(
        prepare_source(n, m, seed, None, SourceConstruction::Random).unwrap(),
        expected
    );
    assert_eq!(
        prepare_source(n, m, seed, None, SourceConstruction::default()).unwrap(),
        expected
    );
    assert_eq!(
        prepare_source(n, m, seed, None, SourceConstruction::Balanced).unwrap(),
        balanced_source(n, m, seed).unwrap()
    );
    // A supplied source is exclusive with the balanced construction, even a
    // valid balanced one.
    let supplied = balanced_source(n, m, seed).unwrap();
    assert_eq!(
        prepare_source(
            n,
            m,
            seed,
            Some(supplied.clone()),
            SourceConstruction::Random
        )
        .unwrap(),
        supplied
    );
    assert_eq!(
        prepare_source(n, m, seed, Some(supplied), SourceConstruction::Balanced).unwrap_err(),
        "source.path and the balanced construction are exclusive"
    );
}

#[test]
fn source_construction_names_round_trip() {
    assert_eq!(SourceConstruction::default(), SourceConstruction::Random);
    for c in [SourceConstruction::Random, SourceConstruction::Balanced] {
        assert_eq!(SourceConstruction::parse(c.name()), Some(c));
    }
    assert!(SourceConstruction::Balanced.is_balanced());
    assert!(!SourceConstruction::Random.is_balanced());
    for bad in ["", "Random", "mirror", "classic", "uniform"] {
        assert_eq!(SourceConstruction::parse(bad), None, "{bad:?}");
    }
}

#[test]
fn audit_accepts_a_balanced_circuit_and_rejects_each_violation() {
    let seq = |gates: Vec<[u16; 3]>| CircuitSeq { gates };
    // Values: inputs 0..4; gate i writes value 4 + i. Value 4 (wire 0) is
    // read by gates 1 and 2, then stays an output; nothing is read thrice.
    let ok = audit_balanced(4, &seq(vec![[0, 1, 2], [3, 0, 1], [2, 0, 3]])).unwrap();
    assert_eq!(
        ok,
        BalancedAudit {
            wires: 4,
            gates: 3,
            internal_births: 0,
            maximum_control_fanout: 2,
            maximum_write_spread: 1,
            final_writes: vec![0, 1],
        }
    );
    let err = |n: usize, gates: Vec<[u16; 3]>| audit_balanced(n, &seq(gates)).unwrap_err();
    assert!(err(4, vec![[0, 0, 1]]).contains("not distinct"));
    assert!(err(3, vec![[0, 1, 3]]).contains("below n=3"));
    // Input 1 read three times.
    assert!(err(5, vec![[0, 1, 2], [3, 1, 2], [4, 1, 2]]).contains("more than 2"));
    // The value gate 0 wrote is overwritten after a single read.
    assert!(err(4, vec![[0, 1, 2], [3, 0, 1], [0, 1, 2]]).contains("after 1 control reads"));
    // Reads are fine but wire 0 is written twice while wire 1 is never
    // written: spread 2.
    assert!(err(4, vec![[0, 1, 2], [3, 0, 1], [2, 0, 3], [0, 3, 2]]).contains("write spread 2"));
    assert!(err(0, vec![]).contains("at least one wire"));
}
