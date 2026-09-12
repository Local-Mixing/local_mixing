use super::*;

fn word_perm(e: &SwapWordEngine, w: &[u8]) -> u64 {
    w.iter()
        .fold(IDENT, |p, &g| compose(p, e.gate_perm(g as usize)))
}

#[test]
fn engine_dist_s_ab_is_6() {
    let e = SwapWordEngine::new();
    let w = e.solve(e.s_ab, MAX_WORD).expect("swap solvable");
    assert_eq!(w.len(), 6);
    assert_eq!(word_perm(&e, &w), e.s_ab);
    // A context can realize S_ab exactly (e.g. a 3-CNOT swap), making the
    // seam target the identity: the empty word is the answer, and the
    // caller's net (0 - k) goes NEGATIVE — a shrinking seam, which the
    // selection arithmetic must represent (it once panicked on usize).
    assert_eq!(e.solve(IDENT, MAX_WORD).as_deref(), Some(&[][..]));
}

#[test]
fn engine_reproduces_hidden_swap_identity() {
    // [a,b,c] . swap(a,b) . [b,c,a] == some 4-gate all-g57 word (the
    // HIDDEN_SWAP_IDENTITY in mix.rs exhibits one; here the engine must
    // find one of the same length).
    let e = SwapWordEngine::new();
    let g1 = xgate_perm(0, &[(1, false), (2, true)], true); // [a,b,c]
    let g2 = xgate_perm(1, &[(2, false), (0, true)], true); // [b,c,a]
    let target = compose(compose(g1, e.s_ab), g2);
    let w = e.solve(target, MAX_WORD).expect("pair context solvable");
    assert_eq!(w.len(), 4, "same-3-wire pair must cost net +2");
    assert_eq!(word_perm(&e, &w), target);
}

#[test]
fn engine_solutions_verify_and_match_census() {
    // Cross-check against the exhaustive Python census (2026-07-29):
    // over ALL ordered pairs (h1, h2) of distinct 3-wire g57s on {a,b,c},
    // h1 . h2 . S_ab is solvable, and the length-4 count is 12 of 30
    // (fixed-op 2-prefix coverage 12/36 minus the 6 identical pairs).
    let e = SwapWordEngine::new();
    let mut on3: Vec<u64> = Vec::new();
    for i in 0..NGATES {
        let (t, n, p) = gate_pins(i);
        if t < 3 && n < 3 && p < 3 {
            on3.push(e.gate_perm(i));
        }
    }
    assert_eq!(on3.len(), 6);
    let (mut len4, mut seen) = (0, 0);
    for &h1 in &on3 {
        for &h2 in &on3 {
            if h1 == h2 {
                continue;
            }
            seen += 1;
            let target = compose(compose(h1, h2), e.s_ab);
            // dist(h1.h2.S) ranges over [4, 8]; 8 exceeds the 3+4 MITM
            // reach and solve correctly returns None there — the placer
            // then consumes fewer gates instead.
            let Some(w) = e.solve(target, MAX_WORD) else {
                continue;
            };
            assert_eq!(word_perm(&e, &w), target);
            assert!(w.len() >= 4 && w.len() <= 7);
            if w.len() == 4 {
                len4 += 1;
            }
        }
    }
    assert_eq!(seen, 30);
    assert_eq!(
        len4, 12,
        "census: 12/30 ordered pairs at length 4 for fixed S_ab"
    );
}

#[test]
fn engine_decode_roundtrip() {
    let e = SwapWordEngine::new();
    let w = e.solve(e.s_ab, MAX_WORD).unwrap();
    let gates = e.decode(&w, &[7, 3, 11, 0]);
    assert_eq!(gates.len(), 6);
    for g in &gates {
        assert!(g.comp && g.ctrls.len() == 2, "decoded gates are g57-shaped");
    }
}
