use super::*;

// apply_lanes4 backs Mixer::global_check, the whole-circuit equivalence
// check that guards every mixing stage. A divergence here would not fail
// loudly -- it would silently compare the wrong bits and let a broken
// circuit pass verification. Pin the fused form against four independent
// apply_lanes runs over the same samples.
#[test]
fn opt_equiv_apply_lanes4_matches_four_apply_lanes() {
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    const NW: usize = 24;
    for case in 0..400 {
        let gates: Vec<XGate> = (0..(next() % 8) + 1)
            .map(|_| {
                let target = (next() % NW as u64) as u16;
                let mut ctrls: Lits = SmallVec::new();
                for _ in 0..(next() % 6) {
                    let wire = (next() % NW as u64) as u16;
                    if wire != target && !ctrls.iter().any(|&(w, _)| w == wire) {
                        ctrls.push((wire, next() % 2 == 0));
                    }
                }
                ctrls.sort_unstable();
                XGate {
                    target,
                    comp: next() % 2 == 0,
                    ctrls,
                }
            })
            .collect();

        // Identical starting samples in both layouts.
        let mut fused = vec![[0u64; 4]; NW];
        let mut split: Vec<Vec<u64>> = (0..4).map(|_| vec![0u64; NW]).collect();
        for b in 0..4 {
            for w in 0..NW {
                let v = next();
                fused[w][b] = v;
                split[b][w] = v;
            }
        }

        eval_lanes4(&gates, &mut fused);
        for b in 0..4 {
            eval_lanes(gates.iter(), &mut split[b]);
        }

        for b in 0..4 {
            for w in 0..NW {
                assert_eq!(
                    fused[w][b],
                    split[b][w],
                    "case {case}: wire {w}, batch {b} diverged over {} gates",
                    gates.len()
                );
            }
        }
    }
}

// A gate with no controls fires unconditionally (acc stays all-ones), and
// comp inverts it: the two edge cases global_check relies on most.
#[test]
fn opt_equiv_apply_lanes4_handles_empty_controls_and_comp() {
    for comp in [false, true] {
        let g = XGate {
            target: 1,
            comp,
            ctrls: SmallVec::new(),
        };
        let mut fused = vec![[0u64; 4]; 4];
        let mut single = vec![0u64; 4];
        for (b, chunk) in fused.iter_mut().enumerate() {
            chunk[0] = b as u64;
        }
        for (w, v) in single.iter_mut().enumerate() {
            *v = fused[w][0];
        }
        g.apply_lanes4(&mut fused);
        g.apply_lanes(&mut single);
        for w in 0..4 {
            assert_eq!(fused[w][0], single[w], "comp={comp}, wire {w}");
        }
    }
}
