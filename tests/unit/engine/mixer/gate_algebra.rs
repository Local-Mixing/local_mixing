// The mask-algebra collides predicate (arena.rs GateMask/mask_collides,
// behind Arena::collides_ids) must equal XGate::collides on every pair.
// Wire universes rotate through dense-low (shared control wires and the
// opposite-polarity separation exemption fire constantly), mid, one-word
// boundary, and > 64 (exercises the second mask word). g57s (comp = true)
// come from rand_gate's allow_comp draw.
#[test]
fn mask_collides_matches_xgate_collides() {
    use super::super::arena::{Arena, GateMask, MASK_WORDS};
    let mut rng = StdRng::seed_from_u64(0xC0111DE);
    let mut collided = 0usize;
    let mut separated = 0usize;
    for i in 0..1_000_000usize {
        let wires: u16 = match i % 4 {
            0 => 5,
            1 => 16,
            2 => 64,
            _ => 127,
        };
        let g = rand_gate(&mut rng, wires, 4, true);
        let h = rand_gate(&mut rng, wires, 4, true);
        let mg = GateMask::of(&g).expect("wire < 64 * MASK_WORDS has a mask");
        let mh = GateMask::of(&h).expect("wire < 64 * MASK_WORDS has a mask");
        let want = XGate::collides(&g, &h);
        assert_eq!(
            Arena::mask_collides(&mg, &mh),
            want,
            "mask mismatch: {g:?} vs {h:?}"
        );
        if want {
            collided += 1;
        } else if g.reads(h.target) || h.reads(g.target) {
            separated += 1; // commuted only via the polarity exemption
        }
    }
    // The draw must actually exercise both hard branches.
    assert!(collided > 10_000, "too few colliding pairs: {collided}");
    assert!(
        separated > 1_000,
        "too few exemption-separated pairs: {separated}"
    );
    // Out-of-range wires have no mask: collides_ids falls back to
    // XGate::collides (arena poisons masks_ok on such an alloc).
    let lim = (64 * MASK_WORDS) as u16;
    assert!(GateMask::of(&XGate::cnot(0, lim)).is_none());
    assert!(GateMask::of(&XGate::cnot(lim, 0)).is_none());
    assert!(GateMask::of(&XGate::cnot(0, lim - 1)).is_some());
}

// Every merge the catalogue accepts is a verified identity with comp=0
// output; the presplit-pair rejoin (complemented result) is rejected.
#[test]
fn merge_catalogue_sound_and_comp_guarded() {
    let mut rng = StdRng::seed_from_u64(7);
    let mut accepted = 0usize;
    for _ in 0..20_000 {
        let g = rand_gate(&mut rng, 6, 3, true);
        let h = rand_gate(&mut rng, 6, 3, true);
        if let Some(m) = merge_result(&g, &h) {
            let out = m.gates();
            assert!(
                out.iter().all(|x| !x.comp),
                "merge emitted comp: {g:?}+{h:?}"
            );
            assert!(
                rules::verify_rewrite(&[g.clone(), h.clone()], &out),
                "unsound merge: {g:?} + {h:?} -> {out:?}"
            );
            accepted += 1;
        }
    }
    assert!(
        accepted > 50,
        "catalogue accepted too few pairs to be tested: {accepted}"
    );

    // The presplit pieces of a g57 (x and !x!y on the same target) XOR to
    // the complemented parent: must be rejected.
    let p0 = XGate::conj(0, [(1u16, true)]).unwrap();
    let p1 = XGate::conj(0, [(1u16, false), (2u16, false)]).unwrap();
    assert!(
        merge_result(&p0, &p1).is_none(),
        "presplit rejoin must be comp-guarded"
    );

    // Two g57s differing in one polarity fuse into a conjunction (fossil
    // erosion), and a g57 plus its own monomial fuse into a NOT gate.
    let g57a = XGate {
        target: 0,
        comp: true,
        ctrls: p1.ctrls.clone(),
    };
    let mut g57b = g57a.clone();
    g57b.ctrls[0].1 = true;
    match merge_result(&g57a, &g57b) {
        Some(Merge::DropLit(m)) => assert!(!m.comp && m.width() == 1),
        other => panic!(
            "comp-comp polarity pair should DropLit, got {:?}",
            other.map(|m| m.gates())
        ),
    }
    let mono = XGate {
        target: 0,
        comp: false,
        ctrls: g57a.ctrls.clone(),
    };
    match merge_result(&g57a, &mono) {
        Some(Merge::XFuse(m)) => assert!(!m.comp && m.width() == 0),
        other => panic!(
            "g57 + own monomial should XFuse, got {:?}",
            other.map(|m| m.gates())
        ),
    }
}

// fresh-wire split and unsubsume each round-trip through the catalogue back
// to the exact original gate.
#[test]
fn split_merge_roundtrips() {
    let mut rng = StdRng::seed_from_u64(11);
    for _ in 0..2_000 {
        let g = rand_gate(&mut rng, 8, 3, false);
        // fresh split on a wire the gate does not touch
        let x = (0..8u16).find(|&w| w != g.target && !g.reads(w)).unwrap();
        let a = XGate::conj(g.target, g.ctrls.iter().copied().chain([(x, true)])).unwrap();
        let b = XGate::conj(g.target, g.ctrls.iter().copied().chain([(x, false)])).unwrap();
        match merge_result(&a, &b) {
            Some(Merge::DropLit(m)) => assert_eq!(m, g),
            _ => panic!("fresh-split pieces must DropLit back to the parent"),
        }
        // unsubsume round-trip
        if g.width() > 0 {
            let (w, p) = g.ctrls[rng.random_range(0..g.ctrls.len())];
            let without = XGate::conj(g.target, g.ctrls_without(w)).unwrap();
            let flipped = XGate::conj(
                g.target,
                g.ctrls
                    .iter()
                    .map(|&(cw, cp)| if cw == w { (cw, !cp) } else { (cw, cp) }),
            )
            .unwrap();
            let _ = p;
            match merge_result(&without, &flipped) {
                Some(Merge::Subsume(m)) => assert_eq!(m, g),
                _ => panic!("unsubsume pieces must Subsume back to the parent"),
            }
        }
    }
}

// Soundness of the collision predicate: ANY pair it calls non-colliding —
// no read of the other's target, or separated by an opposite shared
// control literal — must actually commute. (The converse is not claimed:
// collides() may stay conservatively true on commuting pairs.)
#[test]
fn collides_separation_exemption_sound() {
    let mut rng = StdRng::seed_from_u64(23);
    let mut exempted = 0usize;
    for _ in 0..30_000 {
        let a = rand_gate(&mut rng, 6, 3, true);
        let b = rand_gate(&mut rng, 6, 3, true);
        let reads = a.reads(b.target) || b.reads(a.target);
        if !XGate::collides(&a, &b) {
            assert!(
                rules::verify_rewrite(&[a.clone(), b.clone()], &[b.clone(), a.clone()]),
                "non-colliding pair does not commute: {a:?} / {b:?}"
            );
            if reads {
                exempted += 1; // separated despite a read of a target
                assert!(!a.comp && !b.comp, "comp gate got the exemption");
            }
        }
    }
    assert!(
        exempted > 20,
        "exemption never fired in the sample: {exempted}"
    );
}

// The bridge wake algebra: for random carrier/interior-gate pairs, the
// claimed conjugate u·h·u = [h, corrections] must hold exactly — checked
// against exhaustive evaluation, with coverage over both collision modes
// (h reads the carrier's target / h writes a carrier control wire) and
// the commuting and contradictory (correction-vanishes) cases.
#[test]
fn conj_wake_is_the_exact_conjugate() {
    let n: u16 = 8;
    let mut rng = StdRng::seed_from_u64(0xb41d6e);
    let mut seen = [0usize; 3]; // commuting/vanished, mode-a, mode-b
    let mut refused = 0usize;
    for i in 0..6000 {
        let tu = rng.random_range(0..n);
        let mut xw = rng.random_range(0..n);
        let mut yw = rng.random_range(0..n);
        while xw == tu {
            xw = rng.random_range(0..n);
        }
        while yw == tu || yw == xw {
            yw = rng.random_range(0..n);
        }
        let u = XGate::conj(tu, [(xw, rng.random_bool(0.5)), (yw, rng.random_bool(0.5))]).unwrap();
        // Alternate g57 and conjunction interiors.
        let h = if i % 2 == 0 {
            let t = rng.random_range(0..n);
            let mut a = rng.random_range(0..n);
            let mut b = rng.random_range(0..n);
            while a == t {
                a = rng.random_range(0..n);
            }
            while b == t || b == a {
                b = rng.random_range(0..n);
            }
            XGate::from_g57([t, a, b])
        } else {
            let t = rng.random_range(0..n);
            let w = rng.random_range(1..=3);
            let mut wires: Vec<u16> = (0..n).filter(|&x| x != t).collect();
            for k in 0..wires.len() {
                let j = rng.random_range(k..wires.len());
                wires.swap(k, j);
            }
            XGate::conj(t, wires[..w].iter().map(|&x| (x, rng.random_bool(0.5)))).unwrap()
        };
        let Some(corrs) = conj_wake(&u, &h, 12) else {
            // Mode c (mutual collision) — must really be mutual.
            assert!(
                h.reads(tu) && u.reads(h.target),
                "spurious refusal: {u:?} x {h:?}"
            );
            refused += 1;
            continue;
        };
        let mut after = vec![h.clone()];
        after.extend(corrs.iter().cloned());
        assert!(
            rules::verify_rewrite(&[u.clone(), h.clone(), u.clone()], &after),
            "conjugate wrong: {u:?} x {h:?} -> {after:?}"
        );
        if corrs.is_empty() {
            seen[0] += 1;
        } else if h.reads(tu) {
            seen[1] += 1;
        } else {
            seen[2] += 1;
        }
    }
    assert!(
        seen.iter().all(|&c| c > 100) && refused > 0,
        "coverage too thin: {seen:?} refused={refused}"
    );
}
