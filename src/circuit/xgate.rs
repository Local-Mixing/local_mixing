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
#[path = "../../tests/unit/circuit/xgate/xgate_lane_tests.rs"]
mod xgate_lane_tests;

#[cfg(test)]
#[path = "../../tests/unit/circuit/xgate/xgate_kernel_tests.rs"]
mod xgate_kernel_tests;
