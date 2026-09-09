//! Gate-57-only gadgets with **distinct pins only**.
//!
//! Found by exhaustive search over 3-wire circuits of length ≤ 8
//! (pool = 3! = 6 gates). Each gadget leaves the other two wires unchanged.
//!
//! Deg-3/4 products use **dirty-scratch native** schedules: scratch may start
//! at any value and is restored; pollution terms are canceled explicitly.

/// Local wire 0: `t ^= 1`; wires 1,2 unchanged. Length 8.
const NOT_LOCAL: [[u16; 3]; 8] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [0, 2, 1],
    [2, 0, 1],
    [1, 2, 0],
    [0, 2, 1],
    [2, 1, 0],
];

/// Local: wire0 ^= wire1; wire1, wire2 unchanged. Length 6.
const CNOT_LOCAL: [[u16; 3]; 6] = [
    [1, 0, 2],
    [2, 1, 0],
    [0, 1, 2],
    [1, 2, 0],
    [0, 2, 1],
    [2, 0, 1],
];

/// Local: wire0 ^= wire1 & wire2; wire1, wire2 unchanged. Length 7.
const AND_LOCAL: [[u16; 3]; 7] = [
    [0, 1, 2],
    [1, 2, 0],
    [0, 2, 1],
    [2, 0, 1],
    [0, 1, 2],
    [1, 0, 2],
    [2, 1, 0],
];

fn map_gate(g: [u16; 3], map: [u16; 3]) -> [u16; 3] {
    [map[g[0] as usize], map[g[1] as usize], map[g[2] as usize]]
}

fn emit_mapped(template: &[[u16; 3]], map: [u16; 3], out: &mut Vec<[u16; 3]>) {
    for &g in template {
        let e = map_gate(g, map);
        debug_assert!(e[0] != e[1] && e[0] != e[2] && e[1] != e[2]);
        out.push(e);
    }
}

/// `target ^= 1`. Needs two distinct helper wires (restored).
pub fn emit_not(target: u16, h1: u16, h2: u16, out: &mut Vec<[u16; 3]>) {
    debug_assert!(target != h1 && target != h2 && h1 != h2);
    emit_mapped(&NOT_LOCAL, [target, h1, h2], out);
}

/// `target ^= src`. Needs one helper (restored).
pub fn emit_cnot(target: u16, src: u16, helper: u16, out: &mut Vec<[u16; 3]>) {
    debug_assert!(target != src && target != helper && src != helper);
    emit_mapped(&CNOT_LOCAL, [target, src, helper], out);
}

/// `target ^= c1 & c2`. No extra helper beyond `{target,c1,c2}`; c1,c2 unchanged.
pub fn emit_and(target: u16, c1: u16, c2: u16, out: &mut Vec<[u16; 3]>) {
    debug_assert!(target != c1 && target != c2 && c1 != c2);
    emit_mapped(&AND_LOCAL, [target, c1, c2], out);
}

/// `target ^= a ∧ b ∧ c` with dirty scratch `s` (any start value; restored).
///
/// Schedule: s^=ab; t^=sc; s^=ab; t^=sc — last AND cancels pollution `s0∧c`.
pub fn emit_and3(target: u16, a: u16, b: u16, c: u16, s: u16, out: &mut Vec<[u16; 3]>) {
    debug_assert!(
        [target, a, b, c, s]
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            == 5
    );
    emit_and(s, a, b, out);
    emit_and(target, s, c, out);
    emit_and(s, a, b, out);
    emit_and(target, s, c, out);
}

/// `target ^= a ∧ b ∧ c ∧ d` with dirty scratches `s0,s1` (restored).
pub fn emit_and4(
    target: u16,
    a: u16,
    b: u16,
    c: u16,
    d: u16,
    s0: u16,
    s1: u16,
    out: &mut Vec<[u16; 3]>,
) {
    debug_assert!(
        [target, a, b, c, d, s0, s1]
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            == 7
    );
    // s0 ^= ab; s1 ^= cd; t ^= s0 s1
    emit_and(s0, a, b, out);
    emit_and(s1, c, d, out);
    emit_and(target, s0, s1, out);
    // restore scratches
    emit_and(s0, a, b, out);
    emit_and(s1, c, d, out);
    // cancel s0₀·s1₀, s0₀·cd, ab·s1₀
    emit_and(target, s0, s1, out);
    emit_and3(target, s0, c, d, s1, out);
    emit_and3(target, a, b, s1, s0, out);
}

/// XOR square-free monomial `wires` (deg 1..4) onto `target`.
///
/// Deg 3–4: dirty scratch OK (native product gadgets).
pub fn emit_monomial_xor(
    target: u16,
    wires: &[u16],
    scratches: &[u16],
    cnot_helper: u16,
    out: &mut Vec<[u16; 3]>,
) {
    match wires.len() {
        0 => panic!("deg0: call emit_not"),
        1 => emit_cnot(target, wires[0], cnot_helper, out),
        2 => emit_and(target, wires[0], wires[1], out),
        3 => {
            assert!(scratches.len() >= 1);
            emit_and3(
                target,
                wires[0],
                wires[1],
                wires[2],
                scratches[0],
                out,
            );
        }
        4 => {
            assert!(scratches.len() >= 2);
            emit_and4(
                target,
                wires[0],
                wires[1],
                wires[2],
                wires[3],
                scratches[0],
                scratches[1],
                out,
            );
        }
        d => panic!("monomial degree {d} not supported"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::circuit::Gate;

    fn eval(mut st: usize, gates: &[[u16; 3]]) -> usize {
        for g in gates {
            st = Gate::evaluate_index(st, *g);
            assert!(g[0] != g[1] && g[0] != g[2] && g[1] != g[2], "shared pins");
        }
        st
    }

    #[test]
    fn not_cnot_and_basic() {
        let mut g = Vec::new();
        emit_not(0, 1, 2, &mut g);
        for x in 0..8usize {
            assert_eq!(eval(x, &g) & 1, (x & 1) ^ 1);
        }
        g.clear();
        emit_cnot(0, 1, 2, &mut g);
        for x in 0..8usize {
            assert_eq!(eval(x, &g) & 1, (x & 1) ^ ((x >> 1) & 1));
        }
        g.clear();
        emit_and(0, 1, 2, &mut g);
        for x in 0..8usize {
            assert_eq!(
                eval(x, &g) & 1,
                (x & 1) ^ (((x >> 1) & 1) & ((x >> 2) & 1))
            );
        }
    }

    #[test]
    fn and3_dirty_scratch_exhaustive() {
        // wires: t=0,a=1,b=2,c=3,s=4 — full 5-bit space
        let mut g = Vec::new();
        emit_and3(0, 1, 2, 3, 4, &mut g);
        for x in 0..32usize {
            let y = eval(x, &g);
            let t = (x >> 0) & 1;
            let a = (x >> 1) & 1;
            let b = (x >> 2) & 1;
            let c = (x >> 3) & 1;
            let s = (x >> 4) & 1;
            let want_t = t ^ (a & b & c);
            assert_eq!(y & 1, want_t, "x={x:05b}");
            assert_eq!((y >> 1) & 1, a);
            assert_eq!((y >> 2) & 1, b);
            assert_eq!((y >> 3) & 1, c);
            assert_eq!((y >> 4) & 1, s, "scratch restored");
        }
    }

    #[test]
    fn and4_dirty_scratch_exhaustive() {
        // t=0,a=1,b=2,c=3,d=4,s0=5,s1=6 — 7 bits (128 states)
        let mut g = Vec::new();
        emit_and4(0, 1, 2, 3, 4, 5, 6, &mut g);
        for x in 0..128usize {
            let y = eval(x, &g);
            let t = (x >> 0) & 1;
            let a = (x >> 1) & 1;
            let b = (x >> 2) & 1;
            let c = (x >> 3) & 1;
            let d = (x >> 4) & 1;
            let s0 = (x >> 5) & 1;
            let s1 = (x >> 6) & 1;
            let want_t = t ^ (a & b & c & d);
            assert_eq!(y & 1, want_t, "x={x:07b}");
            assert_eq!((y >> 1) & 1, a);
            assert_eq!((y >> 2) & 1, b);
            assert_eq!((y >> 3) & 1, c);
            assert_eq!((y >> 4) & 1, d);
            assert_eq!((y >> 5) & 1, s0, "s0 restored");
            assert_eq!((y >> 6) & 1, s1, "s1 restored");
        }
    }
}
