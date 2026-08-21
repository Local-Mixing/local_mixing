// General circuit gate type: a single-target controlled XOR whose control is
// a mixed-polarity conjunction, optionally complemented.
//
//   fires(x) = comp XOR AND_i lit_i(x),   lit = wire (pos=true) or NOT wire
//   effect:  x[target] ^= fires(x)
//
// A g57 [a, x, y] (a ^= x OR NOT y) is comp=1 with monomial (NOT x AND y).
// All residues produced by the splitting rules are pure conjunctions (comp=0),
// so comp=1 marks an original, never-split g57.
use smallvec::SmallVec;

pub type Lits = SmallVec<[(u16, bool); 6]>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct XGate {
    pub target: u16,
    pub comp: bool,
    // Sorted by wire, at most one literal per wire, never contains `target`.
    pub ctrls: Lits,
}

impl XGate {
    // Conjunction gate (comp=0) from literals. Returns None when the literal set
    // is contradictory (two polarities on one wire): the gate never fires and
    // must be dropped by the caller. Duplicate literals merge.
    pub fn conj(target: u16, lits: impl IntoIterator<Item = (u16, bool)>) -> Option<XGate> {
        let mut v: Lits = SmallVec::new();
        for (w, p) in lits {
            assert_ne!(w, target, "control literal on the gate's own target");
            v.push((w, p));
        }
        sort_lits(&mut v);
        let mut out: Lits = SmallVec::new();
        for (w, p) in v {
            match out.last() {
                Some(&(lw, lp)) if lw == w => {
                    if lp != p {
                        return None; // w AND NOT w: never fires
                    }
                }
                _ => out.push((w, p)),
            }
        }
        Some(XGate {
            target,
            comp: false,
            ctrls: out,
        })
    }

    // Always-firing gate (X / NOT on `target`).
    pub fn x_gate(target: u16) -> XGate {
        XGate {
            target,
            comp: false,
            ctrls: SmallVec::new(),
        }
    }

    /// Positive-control CNOT: `target ^= control`.
    pub fn cnot(target: u16, control: u16) -> XGate {
        XGate::conj(target, [(control, true)]).expect("a CNOT has one valid control")
    }

    pub fn from_g57(g: [u16; 3]) -> XGate {
        let [a, x, y] = g;
        if x == y {
            // fires iff x OR NOT x == always
            return XGate::x_gate(a);
        }
        // monomial NOT x AND y, emitted in wire order directly: a generic
        // `sort_unstable` call on two elements dominated this conversion when
        // lifting a multi-million-gate g57 circuit.
        let mut ctrls: Lits = SmallVec::new();
        if x < y {
            ctrls.push((x, false));
            ctrls.push((y, true));
        } else {
            ctrls.push((y, true));
            ctrls.push((x, false));
        }
        XGate {
            target: a,
            comp: true,
            ctrls,
        }
    }

    pub fn width(&self) -> usize {
        self.ctrls.len()
    }

    pub fn reads(&self, w: u16) -> bool {
        self.ctrls.iter().any(|&(cw, _)| cw == w)
    }

    pub fn lit_on(&self, w: u16) -> Option<bool> {
        self.ctrls.iter().find(|&&(cw, _)| cw == w).map(|&(_, p)| p)
    }

    // Literal set minus the literal on wire `w`.
    pub fn ctrls_without(&self, w: u16) -> Lits {
        self.ctrls
            .iter()
            .copied()
            .filter(|&(cw, _)| cw != w)
            .collect()
    }

    // Two gates collide iff either one's target is read by the other AND no
    // shared control wire separates their firing supports. Equal targets alone
    // do NOT collide (toggles on one wire commute).
    //
    // The separation exemption: pure conjunctions fire only inside their
    // control subcube, so opposite polarities on a shared control wire w make
    // the supports disjoint. Neither gate writes w (a target is never among
    // its own controls), so on any input at most one can fire and the one
    // that fires cannot unlock the other: they commute regardless of the
    // read/write structure on all other wires. The more controls a
    // conjunction has, the smaller its subcube and the easier it separates —
    // width is INVERSELY related to blocking. Complemented gates (g57s) fire
    // on the COMPLEMENT of a subcube, which touches both halves of every
    // wire, so no single literal separates them: no exemption.
    pub fn collides(a: &XGate, b: &XGate) -> bool {
        if !(a.reads(b.target) || b.reads(a.target)) {
            return false;
        }
        if a.comp || b.comp {
            return true;
        }
        // ctrls are sorted by wire: linear scan for an opposite shared literal.
        let (mut i, mut j) = (0usize, 0usize);
        while i < a.ctrls.len() && j < b.ctrls.len() {
            let (wa, pa) = a.ctrls[i];
            let (wb, pb) = b.ctrls[j];
            if wa == wb {
                if pa != pb {
                    return false;
                }
                i += 1;
                j += 1;
            } else if wa < wb {
                i += 1;
            } else {
                j += 1;
            }
        }
        true
    }

    // 64-lane bit-sliced application: state[w] holds one bit per sample lane.
    //
    // The polarity and complement flags are folded into XOR masks rather than
    // branches: `v ^ pol_mask(p)` is `v` for a positive literal and `!v` for a
    // negative one, and `acc ^ comp_mask` complements the accumulator. The old
    // `if p { v } else { !v }` sat inside the control loop, where the branch
    // is decided by gate data and so is unpredictable.
    #[inline]
    pub fn apply_lanes(&self, state: &mut [u64]) {
        let mut acc = !0u64;
        for &(w, p) in &self.ctrls {
            acc &= state[w as usize] ^ pol_mask(p);
        }
        acc ^= comp_mask(self.comp);
        state[self.target as usize] ^= acc;
    }

    // 256-lane bit-sliced application: `state[w][b]` carries batch b's 64 lanes
    // for wire w. Per batch this is bit-for-bit `apply_lanes` on that batch
    // alone. Callers that want several independent lane batches over the same
    // circuit should use this instead of looping `apply_lanes`: the traversal,
    // the gate load and the `ctrls` walk are paid once rather than per batch,
    // and the extra work is 3 more register ANDs per control.
    #[inline]
    pub fn apply_lanes4(&self, state: &mut [[u64; 4]]) {
        let mut acc = [!0u64; 4];
        for &(w, p) in &self.ctrls {
            let v = state[w as usize];
            let m = pol_mask(p);
            for b in 0..4 {
                acc[b] &= v[b] ^ m;
            }
        }
        let c = comp_mask(self.comp);
        let t = &mut state[self.target as usize];
        for b in 0..4 {
            t[b] ^= acc[b] ^ c;
        }
    }

    pub fn max_wire(&self) -> u16 {
        self.ctrls
            .iter()
            .map(|&(w, _)| w)
            .chain([self.target])
            .max()
            .unwrap()
    }

    // Single-word application: one bit per wire (up to 64 wires).
    #[inline]
    pub fn apply_u64(&self, state: u64) -> u64 {
        debug_assert!(self.max_wire() < 64, "apply_u64 needs every wire below 64");
        let mut fires = 1u64;
        for &(wire, positive) in &self.ctrls {
            fires &= ((state >> wire) & 1) ^ (!positive as u64);
        }
        fires ^= self.comp as u64;
        // `fires` is 0 or 1, so this is the old `if fires { toggle } else
        // { state }` without the data-dependent branch.
        state ^ (fires << self.target)
    }

    /// Apply against a little-endian limb array holding one bit per wire.
    ///
    /// `state.len() * 64` must exceed every wire the gate touches; callers
    /// size the array from `max_wire`.
    ///
    /// This is the kernel every fixed-width scalar entry point routes through.
    /// Reading a control by (limb, bit) costs three u64 ops; the bignum
    /// `(state >> wire) & one` it replaces expanded to a full-width shift and
    /// a full-width compare *per control literal* — 16 limbs of work each at
    /// 1024 bits, which is where `apply_u1024`'s ~92 ns/gate went.
    #[inline]
    pub fn apply_limbs(&self, state: &mut [u64]) {
        debug_assert!(
            (self.max_wire() as usize) < state.len() * 64,
            "limb array too narrow for this gate"
        );
        let mut fires = 1u64;
        for &(wire, positive) in &self.ctrls {
            let v = (state[(wire >> 6) as usize] >> (wire & 63)) & 1;
            fires &= v ^ (!positive as u64);
        }
        fires ^= self.comp as u64;
        state[(self.target >> 6) as usize] ^= fires << (self.target & 63);
    }

    // 1024-bit application: one bit per wire (up to 1024 wires).
    #[inline]
    pub fn apply_u1024(&self, mut state: crate::circuit::U1024) -> crate::circuit::U1024 {
        self.apply_limbs(&mut state.0);
        state
    }
}

/// Sort a control-literal list by `(wire, polarity)`.
///
/// Insertion sort: `ctrls` holds at most a handful of literals and is usually
/// already ordered (both the mpmct1 reader and every internal producer emit
/// wire order), so this is a linear scan with no swaps in the common case.
/// `slice::sort_unstable`'s fixed setup cost showed up as a third of the
/// mpmct1 read time and of the g57 -> XGate lift.
#[inline]
pub fn sort_lits(v: &mut Lits) {
    for i in 1..v.len() {
        let cur = v[i];
        let mut j = i;
        while j > 0 && v[j - 1] > cur {
            v[j] = v[j - 1];
            j -= 1;
        }
        v[j] = cur;
    }
}

/// `0` for a positive literal and `!0` for a negative one, so `v ^ pol_mask(p)`
/// selects `v` or `!v` without branching.
#[inline(always)]
fn pol_mask(positive: bool) -> u64 {
    (positive as u64).wrapping_sub(1)
}

/// `!0` when the gate is complemented, `0` otherwise.
#[inline(always)]
fn comp_mask(comp: bool) -> u64 {
    0u64.wrapping_sub(comp as u64)
}

pub fn eval_lanes<'a>(gates: impl IntoIterator<Item = &'a XGate>, state: &mut [u64]) {
    for g in gates {
        g.apply_lanes(state);
    }
}

/// `eval_lanes` for four independent 64-lane batches carried together.
pub fn eval_lanes4<'a>(gates: impl IntoIterator<Item = &'a XGate>, state: &mut [[u64; 4]]) {
    for g in gates {
        g.apply_lanes4(state);
    }
}

pub fn eval_u64<'a>(gates: impl IntoIterator<Item = &'a XGate>, mut state: u64) -> u64 {
    for gate in gates {
        state = gate.apply_u64(state);
    }
    state
}

/// Walk a gate list against a limb array in place, one bit per wire.
///
/// The width-agnostic scalar evaluator: `state` may be any length, so this
/// also serves circuits wider than the fixed bignum types cover.
pub fn eval_limbs<'a>(gates: impl IntoIterator<Item = &'a XGate>, state: &mut [u64]) {
    for gate in gates {
        gate.apply_limbs(state);
    }
}

pub fn eval_u1024<'a>(
    gates: impl IntoIterator<Item = &'a XGate>,
    mut state: crate::circuit::U1024,
) -> crate::circuit::U1024 {
    // Threading the limb array through the walk keeps one 128-byte stack slot
    // live instead of copying the bignum in and out per gate.
    eval_limbs(gates, &mut state.0);
    state
}

pub fn max_wire<'a>(gates: impl IntoIterator<Item = &'a XGate>) -> u16 {
    gates.into_iter().map(|g| g.max_wire()).max().unwrap_or(0)
}

#[cfg(test)]
mod xgate_lane_tests {
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
}

#[cfg(test)]
mod xgate_kernel_tests {
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
}
