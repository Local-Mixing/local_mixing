//! S5 of the wide-gate design: candidate enumeration of 3-control conjunction
//! gates (8 polarity variants) for the append pass, with fresh-wire class
//! collapse mirroring `abstract_gates_for_circuit_filtered` (regular.rs).
//!
//! Controls are an UNORDERED set — enumerating ordered slots would emit each
//! class 3! = 6 times only for canonicalization to collapse them (pure wasted
//! compute; the design marks this collapse mandatory). Two further collapses
//! come from fresh-wire interchangeability: fresh wires are relabelable, so
//! (a) only one representative wire assignment per abstract class is emitted,
//! and (b) polarity patterns across MULTIPLE fresh control slots dedup to
//! polarity multisets (swapping two fresh control wires together with their
//! polarities is a relabeling).
//!
//! Accounting contract (mirrors the g57 enumerator): `emitted + skipped`
//! equals the full concrete single-gate universe restricted to this source,
//! i.e. `8 * n * C(n-1, 3)` summed over the fresh classes — pinned by test.

use crate::circuit::xgate::XGate;

fn c2(n: usize) -> usize {
    if n >= 2 { n * (n - 1) / 2 } else { 0 }
}
fn c3(n: usize) -> usize {
    if n >= 3 { n * (n - 1) * (n - 2) / 6 } else { 0 }
}

const POL1: [bool; 2] = [true, false];
// Polarity multisets over 2 and 3 interchangeable fresh slots.
const POL2_MULTI: [(bool, bool); 3] = [(true, true), (true, false), (false, false)];
const POL3_MULTI: [(bool, bool, bool); 4] = [
    (true, true, true),
    (true, true, false),
    (true, false, false),
    (false, false, false),
];

fn conj(t: u16, lits: [(u16, bool); 3]) -> XGate {
    XGate::conj(t, lits).expect("distinct wires never contradict")
}

/// Enumerate one representative per abstract wide-gate class for a source
/// circuit touching `touched` wires (sorted, deduped) in an `n`-wire
/// universe, skipping whole classes that cannot satisfy the band bounds.
/// Returns `(representatives, skipped_concrete_count)`.
pub fn wide_gates_for_circuit_filtered(
    touched: &[u16],
    n: usize,
    min_n: usize,
    max_n: usize,
) -> (Vec<XGate>, usize) {
    let untouched: Vec<u16> = (0..n as u16).filter(|w| !touched.contains(w)).collect();
    let u = touched.len();
    let f = untouched.len();
    let mut out = Vec::new();
    let mut skipped = 0usize;

    let allowed = |new_wires: usize| {
        let used = u + new_wires;
        used >= min_n && (max_n == 0 || used <= max_n)
    };

    // ---- j = 0: all four wires touched. Every concrete gate its own class.
    let concrete0 = 8 * u * c3(u.saturating_sub(1));
    if allowed(0) && u >= 4 {
        for &t in touched {
            for i in 0..u {
                for j in i + 1..u {
                    for k in j + 1..u {
                        let (b, c, d) = (touched[i], touched[j], touched[k]);
                        if b == t || c == t || d == t {
                            continue;
                        }
                        for &pb in &POL1 {
                            for &pc in &POL1 {
                                for &pd in &POL1 {
                                    out.push(conj(t, [(b, pb), (c, pc), (d, pd)]));
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        skipped += concrete0;
    }

    // ---- j = 1.
    if f >= 1 {
        let u0 = untouched[0];
        // (1a) fresh target, three touched controls.
        let concrete1a = 8 * c3(u) * f;
        if allowed(1) && u >= 3 {
            for i in 0..u {
                for j in i + 1..u {
                    for k in j + 1..u {
                        let (b, c, d) = (touched[i], touched[j], touched[k]);
                        for &pb in &POL1 {
                            for &pc in &POL1 {
                                for &pd in &POL1 {
                                    out.push(conj(u0, [(b, pb), (c, pc), (d, pd)]));
                                }
                            }
                        }
                    }
                }
            }
            skipped += concrete1a - 8 * c3(u);
        } else {
            skipped += concrete1a;
        }
        // (1b) touched target, one fresh control (its polarity is real: a
        // polarity flip is not a relabeling), two touched controls.
        let concrete1b = 8 * u * c2(u.saturating_sub(1)) * f;
        if allowed(1) && u >= 3 {
            for &t in touched {
                for i in 0..u {
                    for j in i + 1..u {
                        let (b, c) = (touched[i], touched[j]);
                        if b == t || c == t {
                            continue;
                        }
                        for &pf in &POL1 {
                            for &pb in &POL1 {
                                for &pc in &POL1 {
                                    out.push(conj(t, [(u0, pf), (b, pb), (c, pc)]));
                                }
                            }
                        }
                    }
                }
            }
            skipped += concrete1b - 8 * u * c2(u - 1);
        } else {
            skipped += concrete1b;
        }
    }

    // ---- j = 2.
    if f >= 2 {
        let (u0, u1) = (untouched[0], untouched[1]);
        // (2a) fresh target + one fresh control + two touched controls.
        let concrete2a = 8 * c2(u) * f * (f - 1);
        if allowed(2) && u >= 2 {
            for i in 0..u {
                for j in i + 1..u {
                    let (b, c) = (touched[i], touched[j]);
                    for &pf in &POL1 {
                        for &pb in &POL1 {
                            for &pc in &POL1 {
                                out.push(conj(u0, [(u1, pf), (b, pb), (c, pc)]));
                            }
                        }
                    }
                }
            }
            skipped += concrete2a - 8 * c2(u);
        } else {
            skipped += concrete2a;
        }
        // (2b) touched target, two fresh controls (polarity MULTISET) + one
        // touched control.
        let concrete2b = 8 * u * (u.saturating_sub(1)) * c2(f);
        if allowed(2) && u >= 2 {
            for &t in touched {
                for &b in touched {
                    if b == t {
                        continue;
                    }
                    for &(p0, p1) in &POL2_MULTI {
                        for &pb in &POL1 {
                            out.push(conj(t, [(u0, p0), (u1, p1), (b, pb)]));
                        }
                    }
                }
            }
            skipped += concrete2b - 6 * u * (u - 1);
        } else {
            skipped += concrete2b;
        }
    }

    // ---- j = 3.
    if f >= 3 {
        let (u0, u1, u2) = (untouched[0], untouched[1], untouched[2]);
        // (3a) fresh target + two fresh controls (multiset) + one touched.
        let concrete3a = 8 * u * f * c2(f - 1);
        if allowed(3) && u >= 1 {
            for &b in touched {
                for &(p0, p1) in &POL2_MULTI {
                    for &pb in &POL1 {
                        out.push(conj(u0, [(u1, p0), (u2, p1), (b, pb)]));
                    }
                }
            }
            skipped += concrete3a - 6 * u;
        } else {
            skipped += concrete3a;
        }
        // (3b) touched target + three fresh controls (polarity multiset).
        let concrete3b = 8 * u * c3(f);
        if allowed(3) && u >= 1 {
            for &t in touched {
                for &(p0, p1, p2) in &POL3_MULTI {
                    out.push(conj(t, [(u0, p0), (u1, p1), (u2, p2)]));
                }
            }
            skipped += concrete3b - 4 * u;
        } else {
            skipped += concrete3b;
        }
    }

    // ---- j = 4: everything fresh.
    if f >= 4 {
        let concrete4 = 8 * f * c3(f - 1);
        if allowed(4) {
            let (u0, u1, u2, u3) = (untouched[0], untouched[1], untouched[2], untouched[3]);
            for &(p0, p1, p2) in &POL3_MULTI {
                out.push(conj(u0, [(u1, p0), (u2, p1), (u3, p2)]));
            }
            skipped += concrete4 - 4;
        } else {
            skipped += concrete4;
        }
    }

    (out, skipped)
}

#[cfg(test)]
#[path = "../tests/db_gen/wide_gates/tests.rs"]
mod tests;
