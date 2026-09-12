use super::*;
use crate::circuit::xgate::{XGate, eval_lanes};

fn conj(t: u16, lits: &[(u16, bool)]) -> XGate {
    XGate::conj(t, lits.iter().copied()).unwrap()
}

fn equal_on(a: &[XGate], b: &[XGate], wires: usize, live: Option<&[bool]>) -> bool {
    let mut rng = StdRng::seed_from_u64(99);
    for _ in 0..64 {
        let mut sa: Vec<u64> = (0..wires).map(|_| rng.random::<u64>()).collect();
        let mut sb = sa.clone();
        eval_lanes(a.iter(), &mut sa);
        eval_lanes(b.iter(), &mut sb);
        for w in 0..wires {
            if live.is_none_or(|l| l[w]) && sa[w] != sb[w] {
                return false;
            }
        }
    }
    true
}

#[test]
fn opt_equiv_masked_parity_matches_reference() {
    let mut rng = StdRng::seed_from_u64(0x5EED_CAFE);
    for case in 0..240usize {
        // Wire pools spanning verify_group's exhaustive (<=16 support) and
        // sampled (>16) branches; the largest also forces the multi-word
        // mask path (support > 64 bits).
        let pool: u16 = [6, 20, 80][case % 3];
        let n_gates = rng.random_range(1..=10usize);
        let mut gates: Vec<XGate> = Vec::new();
        for _ in 0..n_gates {
            let width = rng.random_range(0..=6usize);
            let mut lits: Vec<(u16, bool)> = Vec::with_capacity(width);
            while lits.len() < width {
                let w = rng.random_range(0..pool);
                if lits.iter().all(|&(seen, _)| seen != w) {
                    lits.push((w, rng.random_bool(0.5)));
                }
            }
            let mut g = XGate::conj(pool, lits).expect("distinct literals");
            g.comp = rng.random_bool(0.5);
            gates.push(g);
        }
        if pool > 64 {
            // Singletons on every pool wire guarantee a two-word support.
            for w in 0..pool {
                let mut g = XGate::conj(pool, [(w, rng.random_bool(0.5))]).unwrap();
                g.comp = rng.random_bool(0.5);
                gates.push(g);
            }
        }
        let mut support: Vec<u16> = gates
            .iter()
            .flat_map(|g| g.ctrls.iter().map(|&(w, _)| w))
            .collect();
        support.sort_unstable();
        support.dedup();
        let words = support.len().div_ceil(64).max(1);
        let (comps, pos, neg) = cube_masks(&gates, &support, words);
        let compare = |bits: &[bool]| {
            let mut assign = vec![0u64; words];
            for (i, &b) in bits.iter().enumerate() {
                if b {
                    assign[i / 64] |= 1u64 << (i % 64);
                }
            }
            let reference = parity_of_reference(&gates, &|w| {
                bits[support.binary_search(&w).expect("wire in support")]
            });
            assert_eq!(
                masked_parity(&comps, &pos, &neg, words, &assign),
                reference,
                "case={case} bits={bits:?}"
            );
        };
        if support.len() <= 12 {
            for a in 0u32..(1u32 << support.len()) {
                let bits: Vec<bool> = (0..support.len()).map(|i| a >> i & 1 == 1).collect();
                compare(&bits);
            }
        }
        for _ in 0..64 {
            let bits: Vec<bool> = (0..support.len()).map(|_| rng.random_bool(0.5)).collect();
            compare(&bits);
        }
    }
}

#[test]
fn adjacent_identical_gates_cancel() {
    let g = conj(0, &[(1, true), (2, false)]);
    let gates = vec![g.clone(), conj(3, &[(1, true)]), g.clone()];
    let (out, rep) = compress(gates.clone(), 4, &CompressParams::default());
    assert_eq!(out.len(), 1, "pair cancels across a non-conflicting gate");
    assert!(equal_on(&gates, &out, 4, None));
    assert!(rep.catalogue_merges >= 1);
}

#[test]
fn reader_blocks_gather() {
    let g = conj(0, &[(1, true)]);
    // Gate 1 reads wire 0, pinning the two writes apart.
    let gates = vec![g.clone(), conj(2, &[(0, true)]), g.clone()];
    let (out, _) = compress(gates.clone(), 3, &CompressParams::default());
    assert_eq!(out.len(), 3, "reader of the target wire must block merging");
    assert!(equal_on(&gates, &out, 3, None));
}

#[test]
fn control_write_blocks_gather() {
    let g = conj(0, &[(1, true)]);
    // Gate 1 writes wire 1 (g's control): second copy may not join.
    let gates = vec![g.clone(), conj(1, &[(2, true)]), g.clone()];
    let (out, _) = compress(gates.clone(), 3, &CompressParams::default());
    assert_eq!(out.len(), 3);
    assert!(equal_on(&gates, &out, 3, None));
}

#[test]
fn comp_parity_folds() {
    // (1 XOR c) XOR c = 1: fossil + its own monomial fuse to an X gate.
    let mut fossil = conj(0, &[(1, false), (2, true)]);
    fossil.comp = true;
    let plain = conj(0, &[(1, false), (2, true)]);
    let gates = vec![fossil.clone(), plain];
    let (out, _) = compress(gates.clone(), 3, &CompressParams::default());
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].width(), 0);
    assert!(equal_on(&gates, &out, 3, None));
    // (1 XOR c) XOR (1 XOR c) = 0: two identical fossils vanish.
    let gates2 = vec![fossil.clone(), fossil];
    let (out2, _) = compress(gates2.clone(), 3, &CompressParams::default());
    assert!(out2.is_empty());
    assert!(equal_on(&gates2, &out2, 3, None));
}

#[test]
fn anf_collapses_beyond_pairwise() {
    // x AND y, x AND NOT y, x: pairwise DropLit then Cancel wipes it out;
    // add a 3-cube ANF-only case too and check function preservation.
    let gates = vec![
        conj(0, &[(1, true), (2, true)]),
        conj(0, &[(1, true), (2, false)]),
        conj(0, &[(1, true)]),
    ];
    let (out, _) = compress(gates.clone(), 3, &CompressParams::default());
    assert!(out.is_empty(), "xy + x!y + x = 0");
    let gates2 = vec![
        conj(0, &[(1, true), (2, true)]),
        conj(0, &[(2, true), (3, true)]),
        conj(0, &[(1, true), (3, true)]),
        conj(0, &[(1, true), (2, true), (3, true)]),
    ];
    let (out2, _) = compress(gates2.clone(), 4, &CompressParams::default());
    assert!(equal_on(&gates2, &out2, 4, None));
    assert!(out2.len() <= 4);
}

#[test]
fn ancestry_threads_through_compression() {
    // Two same-target gates that DropLit into one (t0 ^= x1  ⊕  t0 ^= ¬x1
    // → t0 ^= 1), carrying disjoint ancestor sets: the survivor must hold
    // the union. A bystander gate keeps its own set untouched.
    let g1 = conj(0, &[(1, true)]);
    let g2 = conj(0, &[(1, false)]);
    let by = conj(2, &[(3, true)]);
    let gates = vec![g1, g2, by];
    let anc = vec![vec![0b01u64], vec![0b10u64], vec![0b100u64]];
    let p = CompressParams::default();
    let (out, out_anc, rep) = compress_anc(gates.clone(), Some(anc), 4, &p);
    let out_anc = out_anc.expect("tags threaded");
    assert_eq!(
        out.len(),
        out_anc.len(),
        "tags must align with output gates"
    );
    assert!(rep.catalogue_merges >= 1, "the pair must merge");
    let mut found_union = false;
    for (g, a) in out.iter().zip(out_anc.iter()) {
        if g.target == 0 {
            assert_eq!(a, &vec![0b11u64], "survivor carries the members' union");
            found_union = true;
        } else {
            assert_eq!(a, &vec![0b100u64], "bystander keeps its own set");
        }
    }
    assert!(
        found_union,
        "a target-0 survivor must exist (parity X gate)"
    );
    // Function must be preserved with tags threaded (same pass, same rng).
    assert!(equal_on(&gates, &out, 4, None));
}

#[test]
fn two_member_complemented_pair_collapses() {
    // ab XOR !b = 1 XOR a!b: one comp'd cube. The pairwise catalogue must
    // refuse this (flipped shared polarity); the ANF path absorbs the
    // complement into the parity slot. Regression for the old >=3 gate.
    let gates = vec![conj(0, &[(1, true), (2, true)]), conj(0, &[(2, false)])];
    let (out, rep) = compress(gates.clone(), 3, &CompressParams::default());
    assert_eq!(out.len(), 1, "pair must collapse to one complemented cube");
    assert!(out[0].comp, "the complement must land in the comp bit");
    assert!(equal_on(&gates, &out, 3, None));
    assert!(rep.anf_wins >= 1);
}

#[test]
fn exact_table_beats_pairing() {
    // a XOR b XOR ab = 1 XOR !a!b: three cubes into one comp'd cube. The
    // catalogue gets stuck at two (a!b, b); the exact <=4-support table
    // finds the minimum witness.
    let gates = vec![
        conj(0, &[(1, true)]),
        conj(0, &[(2, true)]),
        conj(0, &[(1, true), (2, true)]),
    ];
    let (out, rep) = compress(gates.clone(), 3, &CompressParams::default());
    assert_eq!(out.len(), 1, "exact minimum is one complemented cube");
    assert_eq!(out[0].width(), 2);
    assert!(out[0].comp);
    assert!(equal_on(&gates, &out, 3, None));
    assert!(rep.exact_wins >= 1);
}

#[test]
fn random_groups_reduce_and_preserve_function() {
    // Soak the multi-strategy reducer: random same-target groups across
    // the exact (<=4), cover/matching, and large-support paths.
    let mut rng = StdRng::seed_from_u64(0xE50);
    for case in 0..200usize {
        let pool = [3u16, 4, 6, 10][case % 4];
        let n = rng.random_range(2..=10usize);
        let mut gates: Vec<XGate> = Vec::new();
        for _ in 0..n {
            let w = rng.random_range(0..=pool.min(5) as usize);
            let mut lits: Vec<(u16, bool)> = Vec::new();
            while lits.len() < w {
                let c = rng.random_range(1..=pool);
                if lits.iter().all(|&(seen, _)| seen != c) {
                    lits.push((c, rng.random_bool(0.5)));
                }
            }
            let mut g = XGate::conj(0, lits).expect("distinct literals");
            g.comp = rng.random_bool(0.3);
            gates.push(g);
        }
        let (out, _) = compress(gates.clone(), pool as usize + 1, &CompressParams::default());
        assert!(out.len() <= gates.len());
        assert!(
            equal_on(&gates, &out, pool as usize + 1, None),
            "case {case} changed the function"
        );
    }
}

#[test]
fn zero_specialization_folds_and_kills() {
    // zero_in = wire 0. Gate 0 can never fire (positive literal on a zero
    // wire); gate 1 folds its always-true literal away; the comp=1 gate
    // with a dead cube still applies t ^= 1 and must DEGRADE to an X, not
    // vanish, while the comp=1 gate folding to an empty cube is a no-op
    // and must vanish; the write to wire 0 ends its known-zero status, so
    // the last gate survives untouched. Equality is only promised on the
    // zero slice.
    let dead_comp = {
        let mut g = conj(5, &[(0, true), (2, true)]);
        g.comp = true;
        g
    };
    let noop_comp = {
        let mut g = conj(6, &[(0, false)]);
        g.comp = true;
        g
    };
    let gates = vec![
        conj(3, &[(0, true), (1, true)]),
        conj(3, &[(0, false), (1, true)]),
        dead_comp,
        noop_comp,
        conj(0, &[(1, true)]),
        conj(4, &[(0, true), (2, true)]),
    ];
    let mut zero = vec![false; 7];
    zero[0] = true;
    let p = CompressParams {
        zero_in: Some(zero.clone()),
        ..Default::default()
    };
    let (out, rep) = compress(gates.clone(), 7, &p);
    assert_eq!(rep.zero_killed, 2, "dead comp=0 gate and no-op comp gate");
    assert!(rep.zero_lits_dropped >= 1);
    assert_eq!(out.len(), 4);
    assert!(out.iter().any(|g| g.target == 3 && g.width() == 1));
    assert!(out.iter().any(|g| g.target == 4 && g.width() == 2));
    assert!(
        out.iter()
            .any(|g| g.target == 5 && g.width() == 0 && !g.comp),
        "dead comp=1 cube must leave a bare X behind"
    );
    assert!(out.iter().all(|g| g.target != 6));
    // Equal on the promised subspace: zero wires forced to 0 at entry.
    let mut rng = StdRng::seed_from_u64(7);
    for _ in 0..64 {
        let sa: Vec<u64> = (0..7)
            .map(|w| if zero[w] { 0 } else { rng.random::<u64>() })
            .collect();
        let mut sb = sa.clone();
        let mut sa = sa;
        eval_lanes(gates.iter(), &mut sa);
        eval_lanes(out.iter(), &mut sb);
        assert_eq!(sa, sb, "zero-slice equality violated");
    }
}

#[test]
fn downhill_collapses_crossing_ladder() {
    // t ^= bx; b ^= c; t ^= bx; t ^= cx computes just b ^= c: the trailing
    // pair is the case-split ladder of floating the first write across the
    // CNOT. Gathering alone is stuck (the CNOT pins both sides); the
    // interleaved downhill pass conjugates the ladder back and the next
    // iteration cancels everything.
    let gates = vec![
        conj(0, &[(1, true), (3, true)]),
        conj(1, &[(2, true)]),
        conj(0, &[(1, true), (3, true)]),
        conj(0, &[(2, true), (3, true)]),
    ];
    let (out, rep) = compress(gates.clone(), 4, &CompressParams::default());
    assert_eq!(out.len(), 1, "everything but the CNOT must vanish");
    assert_eq!(out[0], conj(1, &[(2, true)]));
    // With the defaults the reverse-gather transport folds the ladder
    // before downhill runs; either conjugation route counts.
    assert!(rep.downhill_swaps + rep.transports >= 1);
    assert!(equal_on(&gates, &out, 4, None));
    // With downhill AND in-gather transport disabled the ladder must
    // survive: the win comes from conjugation (either the adjacent
    // downhill pass or the reverse-gather transport), not from plain
    // gathering. Legacy gathering plus downhill alone still wins.
    let (out2, _) = compress(gates.clone(), 4, &legacy());
    assert_eq!(out2.len(), 4);
    let p = CompressParams {
        downhill: true,
        ..legacy()
    };
    let (out3, rep3) = compress(gates.clone(), 4, &p);
    assert_eq!(out3.len(), 1);
    assert!(rep3.downhill_swaps >= 1);
}

#[test]
fn liveness_prunes_dead_cones() {
    // Wire 0 live, wire 1 dead. Last write to 1 is deletable; the earlier
    // write to 1 feeds the live gate through 1 and must stay.
    let gates = vec![
        conj(1, &[(2, true)]), // stays: 1 read below by live gate
        conj(0, &[(1, true)]), // live
        conj(1, &[(0, true)]), // dead: nothing live reads 1 after
    ];
    let live = vec![true, false, true];
    let p = CompressParams {
        live_out: Some(live.clone()),
        ..Default::default()
    };
    let (out, rep) = compress(gates.clone(), 3, &p);
    assert_eq!(rep.liveness_dropped, 1);
    assert!(equal_on(&gates, &out, 3, Some(&live)));
}

#[test]
fn random_circuit_compresses_and_preserves_function() {
    let mut rng = StdRng::seed_from_u64(5);
    let wires = 8u16;
    let mut gates: Vec<XGate> = Vec::new();
    while gates.len() < 400 {
        let t = rng.random_range(0..wires);
        let w = rng.random_range(0..=3usize);
        let lits: Vec<(u16, bool)> = (0..w)
            .map(|_| {
                let mut c = rng.random_range(0..wires);
                while c == t {
                    c = rng.random_range(0..wires);
                }
                (c, rng.random_bool(0.5))
            })
            .collect();
        if let Some(mut g) = XGate::conj(t, lits) {
            g.comp = rng.random_bool(0.2);
            gates.push(g);
        }
    }
    let (out, rep) = compress(gates.clone(), wires as usize, &CompressParams::default());
    assert!(out.len() <= gates.len());
    assert!(rep.iters >= 1);
    assert!(equal_on(&gates, &out, wires as usize, None));
    // On a dense 8-wire circuit real reduction should happen.
    assert!(
        out.len() < gates.len(),
        "expected some compression on dense circuit"
    );
}

// Gather-only parameters: the legacy rule set (no transport, no
// separation passes, no reverse pass, no downhill).
fn legacy() -> CompressParams {
    CompressParams {
        downhill: false,
        transport: false,
        sep_reads: false,
        reverse_pass: false,
        ..Default::default()
    }
}

#[test]
fn toffoli_sliding_pair_cancels() {
    // The family-B triple from the K2 final (gates 837-839): a control
    // flipped under the other controls slides through with a polarity
    // change, so the two copies are one identity. Legacy gathering
    // cannot form the group (the flip writes a control); transport
    // conjugates the first copy across the flip and the pair cancels.
    let g = conj(3, &[(0, false), (1, true), (2, true)]);
    let flip = conj(1, &[(0, false), (2, true)]);
    let g2 = conj(3, &[(0, false), (1, false), (2, true)]);
    let gates = vec![g, flip.clone(), g2];
    let (out, rep) = compress(gates.clone(), 4, &CompressParams::default());
    assert_eq!(out, vec![flip], "only the flip survives");
    assert!(rep.transports >= 1);
    assert!(equal_on(&gates, &out, 4, None));
    let (legacy_out, _) = compress(gates.clone(), 4, &legacy());
    assert_eq!(
        legacy_out.len(),
        3,
        "legacy gathering is blocked by the flip"
    );
}

#[test]
fn toffoli_sliding_at_distance_and_through_reads() {
    // Same relation, copies seven gates apart, with the flipped control
    // READ in between (which closes the flip's own group but must not
    // close the transported group: it depends on the flip's group, and a
    // dependency emitted early is fine). Complemented flip too.
    let g = conj(3, &[(0, false), (1, true), (2, true)]);
    let mut flip = conj(1, &[(0, false), (2, true)]);
    flip.comp = true; // u ^= NOT(!w0 & w2): flips u where g's cube holds
    // g[u] under u <- u ^ 1 ^ (!w0&w2) = g with u kept (comp adds a flip
    // everywhere, the cube undoes it where g fires): copy reads u
    // positively... work it out: on g's cube the flip fires 0 -> u
    // unchanged. So the identical copy is the identity here.
    let g2 = conj(3, &[(0, false), (1, true), (2, true)]);
    // Fillers read u and each carries its own private literal, so they
    // neither merge with each other nor touch g.
    let filler: Vec<XGate> = (0..6)
        .map(|k| conj(4 + (k % 3), &[(1, k % 2 == 0), (8 + k as u16, true)]))
        .collect();
    let mut gates = vec![g, flip.clone()];
    gates.extend(filler.iter().cloned());
    gates.push(g2);
    let (out, rep) = compress(gates.clone(), 14, &CompressParams::default());
    assert_eq!(
        out.len(),
        7,
        "the pair cancels through six fillers: {out:?}"
    );
    // The complemented flip fires 0 on g's cube, so the conjugated ESOP
    // is g itself: a no-op transport (no frame dependency needed).
    assert!(rep.transports + rep.transport_noops >= 1);
    assert!(equal_on(&gates, &out, 14, None));
}

#[test]
fn transport_frame_order_respects_dependencies() {
    // g floats across the flip (dependency on the flip's group); a read
    // of t then closes g's group BEFORE the flip's group would close on
    // its own, and another writer of u arrives afterwards. The flip must
    // be emitted before the transported cube and the later writer must
    // not be gathered across it.
    let g = conj(3, &[(0, false), (1, true), (2, true)]);
    let flip = conj(1, &[(0, false), (2, true)]);
    let read_t = conj(5, &[(3, true)]);
    let later_u = conj(1, &[(6, true)]);
    let g2 = conj(3, &[(0, false), (1, false), (2, true)]);
    let gates = vec![g, flip, read_t, later_u, g2];
    let (out, _) = compress(gates.clone(), 8, &CompressParams::default());
    assert!(equal_on(&gates, &out, 8, None));
    // And the same with the flip's group forced closed by a read of u in
    // between, so the transported group outlives its dependency.
    let read_u = conj(5, &[(1, true)]);
    let g = conj(3, &[(0, false), (1, true), (2, true)]);
    let flip = conj(1, &[(0, false), (2, true)]);
    let g2 = conj(3, &[(0, false), (1, false), (2, true)]);
    let gates = vec![g, flip, read_u, g2];
    let (out, rep) = compress(gates.clone(), 8, &CompressParams::default());
    assert!(equal_on(&gates, &out, 8, None));
    assert_eq!(out.len(), 2, "pair cancels across the read of u: {out:?}");
    assert!(rep.transports >= 1);
}

#[test]
fn separated_reader_does_not_close_group() {
    // r reads t but is separated from both copies on wire 2 (opposite
    // polarity), so it commutes with them and the copies cancel.
    let g = conj(0, &[(1, true), (2, true)]);
    let r = conj(3, &[(0, true), (2, false)]);
    let gates = vec![g.clone(), r.clone(), g.clone()];
    let (out, rep) = compress(gates.clone(), 4, &CompressParams::default());
    assert_eq!(out, vec![r], "separated reader passes, pair cancels");
    assert!(rep.sep_passes >= 1);
    assert!(equal_on(&gates, &out, 4, None));
    let (legacy_out, _) = compress(gates.clone(), 4, &legacy());
    assert_eq!(legacy_out.len(), 3);
}

#[test]
fn reverse_pass_folds_leftward_ladder() {
    // h, then (three gates later) the case-split ladder {t^=L&u, t^=L&M}
    // left by crossing t^=L&u leftward over h (u ^= M). Downhill sees
    // only immediate neighbours and the forward gather floats the ladder
    // away from h; the reversed gather floats it back onto h and the
    // transport folds it to one cube.
    let h = conj(1, &[(4, true)]);
    let fill: Vec<XGate> = vec![
        conj(5, &[(6, true)]),
        conj(6, &[(7, false)]),
        conj(7, &[(5, true)]),
    ];
    let ladder = vec![
        conj(0, &[(1, true), (2, true)]),
        conj(0, &[(2, true), (4, true)]),
    ];
    let mut gates = vec![h];
    gates.extend(fill);
    gates.extend(ladder);
    let (out, rep) = compress(gates.clone(), 8, &CompressParams::default());
    assert_eq!(out.len(), 5, "ladder folds to one cube: {out:?}");
    assert!(rep.transports >= 1);
    assert!(equal_on(&gates, &out, 8, None));
    let p = CompressParams {
        reverse_pass: false,
        ..Default::default()
    };
    let (fwd_only, _) = compress(gates.clone(), 8, &p);
    assert_eq!(fwd_only.len(), 6, "forward-only gathering cannot reach h");
}

#[test]
fn dense_random_soak_all_rules() {
    // Dense circuits on few wires exercise transports, dependency
    // cascades, separated passes and the reverse pass together; the
    // pass must preserve the function and never grow the circuit.
    let mut rng = StdRng::seed_from_u64(0xD0_5EED);
    for case in 0..120usize {
        let wires: u16 = [4, 5, 6, 8][case % 4];
        let n = 40 + (case * 7) % 260;
        let mut gates: Vec<XGate> = Vec::new();
        while gates.len() < n {
            let t = rng.random_range(0..wires);
            let w = rng.random_range(0..=3usize);
            let mut lits: Vec<(u16, bool)> = Vec::new();
            while lits.len() < w {
                let c = rng.random_range(0..wires);
                if c != t && lits.iter().all(|&(seen, _)| seen != c) {
                    lits.push((c, rng.random_bool(0.5)));
                }
            }
            let mut g = XGate::conj(t, lits).expect("distinct literals");
            g.comp = rng.random_bool(0.25);
            gates.push(g);
        }
        for (label, p) in [
            ("all", CompressParams::default()),
            (
                "slack1",
                CompressParams {
                    transport_slack: 1,
                    ..Default::default()
                },
            ),
            ("legacy", legacy()),
        ] {
            let (out, _) = compress(gates.clone(), wires as usize, &p);
            assert!(
                out.len() <= gates.len(),
                "case {case} {label} grew the circuit"
            );
            assert!(
                equal_on(&gates, &out, wires as usize, None),
                "case {case} {label} changed the function"
            );
        }
    }
}

#[test]
fn pack_is_exact_canonical_and_unbounded() {
    use crate::circuit::formats::{expand_packed, read_anf1, read_mpmct, write_anf1};
    // Exactness on random compressed circuits, through the file format.
    let mut rng = StdRng::seed_from_u64(0x9AC7);
    for case in 0..40usize {
        let wires: u16 = [5, 8, 12][case % 3];
        let mut gates: Vec<XGate> = Vec::new();
        while gates.len() < 150 {
            let t = rng.random_range(0..wires);
            let w = rng.random_range(0..=3usize);
            let mut lits: Vec<(u16, bool)> = Vec::new();
            while lits.len() < w {
                let c = rng.random_range(0..wires);
                if c != t && lits.iter().all(|&(seen, _)| seen != c) {
                    lits.push((c, rng.random_bool(0.5)));
                }
            }
            let mut g = XGate::conj(t, lits).expect("distinct literals");
            g.comp = rng.random_bool(0.3);
            gates.push(g);
        }
        let (out, _) = compress(gates.clone(), wires as usize, &CompressParams::default());
        let packed = pack(&out);
        assert!(packed.len() <= out.len());
        assert!(
            equal_on(&gates, &expand_packed(&packed), wires as usize, None),
            "case {case}"
        );
        if case == 0 {
            let path = std::env::temp_dir()
                .join(format!("fcompress_pack_test_{}.anf1", std::process::id()));
            let path = path.to_str().unwrap().to_string();
            write_anf1(&path, &packed, wires as usize).unwrap();
            let (back, w) = read_anf1(&path).unwrap();
            assert_eq!(back, packed, "anf1 round trip");
            assert_eq!(w, wires as usize);
            let (via_mpmct, _) = read_mpmct(&path).unwrap();
            assert_eq!(
                via_mpmct,
                expand_packed(&packed),
                "read_mpmct dispatches on the anf1 header"
            );
            std::fs::remove_file(&path).ok();
        }
    }
    // Canonical: two spellings of one function pack identically, cubes
    // with negative literals expand, comp bits become the constant.
    let a = vec![
        conj(0, &[(1, true), (2, true)]),
        conj(0, &[(1, true), (2, false)]),
    ];
    let b = vec![conj(0, &[(1, true)])];
    assert_eq!(pack(&a), pack(&b));
    let mut c = conj(0, &[(1, false), (2, false)]);
    c.comp = true; // 1 ^ (1^a)(1^b) = a ^ b ^ ab
    let pc = pack(&[c]);
    assert_eq!(
        pc[0].terms,
        vec![
            vec![(1u16, true)],
            vec![(2u16, true)],
            vec![(1u16, true), (2u16, true)]
        ]
    );
    // Unbounded support: a run over 80 wires packs in one gate.
    let wide: Vec<XGate> = (1u16..=80)
        .map(|w| conj(0, &[(w, true), (w + 100, false)]))
        .collect();
    let pw = pack(&wide);
    assert_eq!(pw.len(), 1);
    assert_eq!(pw[0].terms.len(), 160);
    assert!(equal_on(&wide, &expand_packed(&pw), 181, None));
}

#[test]
fn compaction_is_exact_smaller_and_a_function_of_the_anf() {
    use crate::circuit::formats::{expand_packed, read_mpmct, write_esop1};
    // Two spellings of one function -> one ANF -> one compacted gate.
    let a = vec![conj(0, &[(1, false), (2, false), (3, true)])];
    let b: Vec<XGate> = a[0]
        .ctrls
        .iter()
        .map(|_| ())
        .take(0)
        .map(|_| a[0].clone())
        .collect::<Vec<_>>();
    drop(b);
    let pa = pack(&a);
    assert_eq!(pa[0].terms.len(), 4, "!x!y z = z + xz + yz + xyz");
    let ca = compact(&pa);
    assert_eq!(ca.len(), 1);
    assert!(ca[0].terms.len() <= pa[0].terms.len());
    assert!(equal_on(&a, &expand_packed(&ca), 4, None));
    // The same function spelled as its four monomials compacts identically.
    let mono: Vec<XGate> = vec![
        conj(0, &[(3, true)]),
        conj(0, &[(1, true), (3, true)]),
        conj(0, &[(2, true), (3, true)]),
        conj(0, &[(1, true), (2, true), (3, true)]),
    ];
    assert_eq!(compact(&pack(&mono)), ca);
    // Exactness + non-growth on random compressed circuits, through esop1.
    let mut rng = StdRng::seed_from_u64(0xE50_9);
    for case in 0..40usize {
        let wires: u16 = [5, 8, 12][case % 3];
        let mut gates: Vec<XGate> = Vec::new();
        while gates.len() < 150 {
            let t = rng.random_range(0..wires);
            let w = rng.random_range(0..=3usize);
            let mut lits: Vec<(u16, bool)> = Vec::new();
            while lits.len() < w {
                let c = rng.random_range(0..wires);
                if c != t && lits.iter().all(|&(seen, _)| seen != c) {
                    lits.push((c, rng.random_bool(0.5)));
                }
            }
            let mut g = XGate::conj(t, lits).expect("distinct literals");
            g.comp = rng.random_bool(0.3);
            gates.push(g);
        }
        let (out, _) = compress(gates.clone(), wires as usize, &CompressParams::default());
        let anf = pack(&out);
        let esop = compact(&anf);
        let (ta, te): (usize, usize) = (
            anf.iter().map(|g| g.terms.len()).sum(),
            esop.iter().map(|g| g.terms.len()).sum(),
        );
        assert!(te <= ta, "case {case}: compaction grew {ta} -> {te}");
        assert!(
            equal_on(&gates, &expand_packed(&esop), wires as usize, None),
            "case {case}"
        );
        if case == 1 {
            let path = std::env::temp_dir()
                .join(format!("fcompress_esop_test_{}.esop1", std::process::id()));
            let path = path.to_str().unwrap().to_string();
            write_esop1(&path, &esop, wires as usize).unwrap();
            let (via_mpmct, _) = read_mpmct(&path).unwrap();
            assert_eq!(via_mpmct, expand_packed(&esop));
            std::fs::remove_file(&path).ok();
        }
    }
}
