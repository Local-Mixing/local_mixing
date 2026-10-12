//! Pairwise peephole post-optimisation on a synthesized gate list (ported
//! from `anf_synth`'s `postopt.rs`; it sees only the synthesized gates).
//! Two gates commute iff neither target is a control of the other. Two
//! gates on the same target whose control sets span <= 2 variables and that
//! can be brought adjacent by commutation are merged into one gate when the
//! XOR of their polynomials is again a gate polynomial (identical gates
//! cancel). The group-merge variant of `anf_synth` (opt-in, measured as a
//! zero gain) is not ported.
use super::synth::SGate;

/// Bits of `L` over the ordered variable pair `(a, b)`: 1 = const, 2 = `x_a`,
/// 4 = `x_b`, 8 = `x_a x_b`. Requires the gate's controls to be within `{a, b}`.
fn lbits(g: &SGate, a: u32, b: u32) -> Option<u8> {
    let mut bits: u8;
    match g.k {
        0 => {
            // empty AND = 1: L = 1 + comp
            bits = if g.comp { 0 } else { 1 };
        }
        1 => {
            let c = g.c[0];
            let vb = if c == a {
                2
            } else if c == b {
                4
            } else {
                return None;
            };
            bits = vb;
            if !g.p[0] ^ g.comp {
                bits |= 1;
            }
        }
        _ => {
            let (v0, v1) = if g.c[0] == a && g.c[1] == b {
                (2u8, 4u8)
            } else if g.c[0] == b && g.c[1] == a {
                (4u8, 2u8)
            } else {
                return None;
            };
            // (x0 + p0')(x1 + p1') + comp = x0x1 + p1' x0 + p0' x1 + p0'p1' + comp
            let p0_ = !g.p[0];
            let p1_ = !g.p[1];
            bits = 8;
            if p1_ {
                bits |= v0;
            }
            if p0_ {
                bits |= v1;
            }
            if (p0_ && p1_) ^ g.comp {
                bits |= 1;
            }
        }
    }
    Some(bits)
}

/// `Some(None)` = identity (delete), `Some(Some(g))` = merged gate, `None` =
/// not a gate polynomial (`x_a + x_b (+1)` needs two gates).
fn gate_from_bits(t: u32, a: u32, b: u32, bits: u8) -> Option<Option<SGate>> {
    if bits == 0 {
        return Some(None);
    }
    let gamma = bits & 1 != 0;
    if bits & 8 != 0 {
        let alpha = bits & 2 != 0;
        let beta = bits & 4 != 0;
        let pa = !beta;
        let pb = !alpha;
        let comp = gamma ^ (alpha && beta);
        return Some(Some(SGate::tof(t, a, pa, b, pb, comp)));
    }
    match bits & 6 {
        2 => Some(Some(SGate::cnot(t, a, true, gamma))),
        4 => Some(Some(SGate::cnot(t, b, true, gamma))),
        0 => Some(Some(SGate::not(t))),
        _ => None,
    }
}

fn var_pair(g: &SGate, h: &SGate) -> Option<(u32, u32)> {
    let mut vs: Vec<u32> = Vec::new();
    for &c in g.controls().iter().chain(h.controls().iter()) {
        if !vs.contains(&c) {
            vs.push(c);
        }
    }
    vs.sort_unstable();
    match vs.len() {
        0 => Some((u32::MAX - 1, u32::MAX)),
        1 => Some((vs[0], u32::MAX)),
        2 => Some((vs[0], vs[1])),
        _ => None,
    }
}

/// Returns the number of gates removed.
pub fn peephole(gates: &mut Vec<SGate>) -> usize {
    let n0 = gates.len();
    let mut changed = true;
    while changed {
        changed = false;
        let mut i = 0;
        'outer: while i < gates.len() {
            let g = gates[i];
            let mut j = i + 1;
            while j < gates.len() {
                let h = gates[j];
                if h.t == g.t {
                    if let Some((a, b)) = var_pair(&g, &h)
                        && let (Some(lg), Some(lh)) = (lbits(&g, a, b), lbits(&h, a, b))
                        && let Some(merged) = gate_from_bits(g.target(), a, b, lg ^ lh)
                    {
                        // g commutes with everything in (i, j): move it
                        // forward onto h's position and merge there.
                        match merged {
                            None => {
                                gates.remove(j);
                                gates.remove(i);
                            }
                            Some(m) => {
                                gates[j] = m;
                                gates.remove(i);
                            }
                        }
                        changed = true;
                        continue 'outer; // re-examine position i
                    }
                    j += 1;
                    continue;
                }
                if !g.commutes(&h) {
                    break;
                }
                j += 1;
            }
            i += 1;
        }
        gates.retain(|g| !g.is_noop());
    }
    n0 - gates.len()
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/unpacking/postopt.rs"]
mod tests;
