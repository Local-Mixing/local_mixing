// Post-mixing gate type: a single-target controlled XOR whose control is a
// mixed-polarity conjunction, optionally complemented.
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
        v.sort_unstable();
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
        Some(XGate { target, comp: false, ctrls: out })
    }

    // Always-firing gate (X / NOT on `target`).
    pub fn x_gate(target: u16) -> XGate {
        XGate { target, comp: false, ctrls: SmallVec::new() }
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
        let mut ctrls: Lits = SmallVec::new();
        ctrls.push((x, false)); // monomial NOT x AND y
        ctrls.push((y, true));
        ctrls.sort_unstable();
        XGate { target: a, comp: true, ctrls }
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
        self.ctrls.iter().copied().filter(|&(cw, _)| cw != w).collect()
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
    #[inline]
    pub fn apply_lanes(&self, state: &mut [u64]) {
        let mut acc = !0u64;
        for &(w, p) in &self.ctrls {
            let v = state[w as usize];
            acc &= if p { v } else { !v };
        }
        if self.comp {
            acc = !acc;
        }
        state[self.target as usize] ^= acc;
    }

    pub fn max_wire(&self) -> u16 {
        self.ctrls.iter().map(|&(w, _)| w).chain([self.target]).max().unwrap()
    }

    // Single-word application: one bit per wire (up to 64 wires).
    pub fn apply_u64(&self, state: u64) -> u64 {
        let mut fires = true;
        for &(wire, positive) in &self.ctrls {
            let value = ((state >> wire) & 1) != 0;
            fires &= value == positive;
        }
        fires ^= self.comp;
        if fires {
            state ^ (1u64 << self.target)
        } else {
            state
        }
    }

    // 1024-bit application: one bit per wire (up to 1024 wires).
    pub fn apply_u1024(&self, state: crate::circuit::circuit::U1024) -> crate::circuit::circuit::U1024 {
        use crate::circuit::circuit::U1024;
        let one = U1024::one();
        let mut fires = true;
        for &(wire, positive) in &self.ctrls {
            let value = ((state >> wire as usize) & one) != U1024::zero();
            fires &= value == positive;
        }
        fires ^= self.comp;
        if fires {
            state ^ (one << self.target as usize)
        } else {
            state
        }
    }
}

pub fn eval_lanes<'a>(gates: impl IntoIterator<Item = &'a XGate>, state: &mut [u64]) {
    for g in gates {
        g.apply_lanes(state);
    }
}

pub fn eval_u64<'a>(gates: impl IntoIterator<Item = &'a XGate>, mut state: u64) -> u64 {
    for gate in gates {
        state = gate.apply_u64(state);
    }
    state
}

pub fn eval_u1024<'a>(
    gates: impl IntoIterator<Item = &'a XGate>,
    mut state: crate::circuit::circuit::U1024,
) -> crate::circuit::circuit::U1024 {
    for gate in gates {
        state = gate.apply_u1024(state);
    }
    state
}

pub fn max_wire<'a>(gates: impl IntoIterator<Item = &'a XGate>) -> u16 {
    gates.into_iter().map(|g| g.max_wire()).max().unwrap_or(0)
}
