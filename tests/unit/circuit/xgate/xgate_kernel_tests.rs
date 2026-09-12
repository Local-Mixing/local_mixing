use super::*;

// Naive per-sample semantics, straight from the type's contract comment:
//   fires(x) = comp XOR AND_i lit_i(x)
// Every kernel below is a different packing of the same function, so they
// are all pinned against this rather than against each other.
fn ref_fires(g: &XGate, bit_at: impl Fn(u16) -> bool) -> bool {
    let mut f = true;
    for &(w, p) in &g.ctrls {
        f &= bit_at(w) == p;
    }
    f ^ g.comp
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 1
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

// A gate mix that covers what a post-split mpmct1 artifact contains:
// k=0 X gates, single-control CNOTs, complemented g57s, and wide
// mixed-polarity conjunctions.
fn random_gate(rng: &mut Lcg, wires: u16) -> XGate {
    let target = rng.below(wires as u64) as u16;
    let k = match rng.below(10) {
        0 => 0, // X gate
        1 | 2 => 1,
        3..=6 => 2,
        7 | 8 => 4,
        _ => 6,
    };
    let mut ctrls: Lits = SmallVec::new();
    for _ in 0..k {
        let w = rng.below(wires as u64) as u16;
        if w != target && !ctrls.iter().any(|&(cw, _)| cw == w) {
            ctrls.push((w, rng.below(2) == 0));
        }
    }
    ctrls.sort_unstable();
    XGate {
        target,
        comp: rng.below(4) == 0,
        ctrls,
    }
}

#[test]
fn opt_equiv_apply_u64_matches_boolean_reference() {
    let mut rng = Lcg(0x1234_5678_9abc_def0);
    for _ in 0..5000 {
        let g = random_gate(&mut rng, 64);
        let state = rng.next() ^ (rng.next() << 32);
        let fires = ref_fires(&g, |w| (state >> w) & 1 == 1);
        let want = if fires {
            state ^ (1u64 << g.target)
        } else {
            state
        };
        assert_eq!(g.apply_u64(state), want, "gate {g:?} state {state:x}");
    }
}

#[test]
fn opt_equiv_apply_limbs_matches_boolean_reference_at_every_width() {
    let mut rng = Lcg(0xdead_beef_0bad_f00d);
    for &wires in &[1u16, 2, 63, 64, 65, 127, 128, 200, 1023] {
        let limbs = (wires as usize).div_ceil(64);
        for _ in 0..600 {
            let g = random_gate(&mut rng, wires);
            let mut state: Vec<u64> = (0..limbs)
                .map(|_| rng.next() ^ (rng.next() << 32))
                .collect();
            let before = state.clone();
            let fires = ref_fires(&g, |w| (before[(w >> 6) as usize] >> (w & 63)) & 1 == 1);
            g.apply_limbs(&mut state);
            let mut want = before.clone();
            if fires {
                want[(g.target >> 6) as usize] ^= 1u64 << (g.target & 63);
            }
            assert_eq!(state, want, "gate {g:?} wires {wires}");
        }
    }
}

// apply_u1024 is the 1024-bit view of apply_limbs; pin the bignum wrapper
// itself so a limb-order mistake cannot hide behind the limb test.
#[test]
fn opt_equiv_apply_u1024_matches_boolean_reference() {
    use crate::circuit::U1024;
    let mut rng = Lcg(0x0bad_cafe_1234_9999);
    for _ in 0..2000 {
        let g = random_gate(&mut rng, 1024);
        let mut bytes = [0u8; 128];
        for b in bytes.iter_mut() {
            *b = rng.below(256) as u8;
        }
        let state = U1024::from_little_endian(&bytes);
        let one = U1024::one();
        let fires = ref_fires(&g, |w| ((state >> w as usize) & one) != U1024::zero());
        let want = if fires {
            state ^ (one << g.target as usize)
        } else {
            state
        };
        assert_eq!(g.apply_u1024(state), want, "gate {g:?}");
    }
}

#[test]
fn opt_equiv_lane_kernels_match_boolean_reference() {
    let mut rng = Lcg(0xabcd_0123_4567_89ab);
    const NW: u16 = 40;
    for _ in 0..800 {
        let g = random_gate(&mut rng, NW);
        let seed: Vec<u64> = (0..NW).map(|_| rng.next() ^ (rng.next() << 32)).collect();

        let mut lanes = seed.clone();
        g.apply_lanes(&mut lanes);

        let mut lanes4: Vec<[u64; 4]> = seed.iter().map(|&v| [v, !v, v ^ 0x5555, 0]).collect();
        let seed4 = lanes4.clone();
        g.apply_lanes4(&mut lanes4);

        for bit in 0..64 {
            // 64-lane form.
            let fires = ref_fires(&g, |w| (seed[w as usize] >> bit) & 1 == 1);
            for w in 0..NW as usize {
                let mut want = (seed[w] >> bit) & 1 == 1;
                if fires && w == g.target as usize {
                    want = !want;
                }
                assert_eq!((lanes[w] >> bit) & 1 == 1, want, "lanes wire {w} bit {bit}");
            }
            // 256-lane form: four independent batches in one walk.
            for b in 0..4 {
                let fires = ref_fires(&g, |w| (seed4[w as usize][b] >> bit) & 1 == 1);
                for w in 0..NW as usize {
                    let mut want = (seed4[w][b] >> bit) & 1 == 1;
                    if fires && w == g.target as usize {
                        want = !want;
                    }
                    assert_eq!(
                        (lanes4[w][b] >> bit) & 1 == 1,
                        want,
                        "lanes4 wire {w} batch {b} bit {bit}"
                    );
                }
            }
        }
    }
}

// A g57 lifted to an XGate must evaluate identically to the g57 kernel,
// X-gate spelling (x == y) included.
#[test]
fn lifted_g57_matches_the_g57_kernel_including_x_gates() {
    use crate::circuit::Gate;
    let mut rng = Lcg(0x7777_1111_2222_3333);
    for _ in 0..3000 {
        let a = rng.below(60) as u16;
        let x = rng.below(60) as u16;
        let y = if rng.below(4) == 0 {
            x // X gate
        } else {
            rng.below(60) as u16
        };
        if x == a || y == a {
            continue; // a control on its own target is not a valid XGate
        }
        let state = rng.next() ^ (rng.next() << 32);
        let want = Gate::evaluate_index_list_64(state, &[[a, x, y]]);
        assert_eq!(XGate::from_g57([a, x, y]).apply_u64(state), want);
    }
}
