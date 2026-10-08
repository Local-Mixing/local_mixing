// Exhaustive local equivalence check for rewrites. The crossing rules
// (R1/R2/R3, presplit) that used to live here belonged to the historic old
// stage 5 (crossing walk); see branch ran/archive/curated-and-postmix.
use crate::circuit::xgate::XGate;

// Exhaustive local equivalence check over the support wires of both sides.
// Support is small (a handful of narrow gates), so 2^support is cheap; run on
// every local rewrite in production (twists, bridges, DB splices). Returns
// true iff the two sequences compute the same function on all wires.
pub fn verify_rewrite(before: &[XGate], after: &[XGate]) -> bool {
    let mut wires: Vec<u16> = Vec::new();
    for g in before.iter().chain(after.iter()) {
        wires.push(g.target);
        wires.extend(g.ctrls.iter().map(|&(w, _)| w));
    }
    wires.sort_unstable();
    wires.dedup();
    let k = wires.len();
    assert!(k <= 24, "rewrite support unexpectedly large: {k} wires");
    let dense = |g: &XGate| {
        let map = |w: u16| wires.binary_search(&w).unwrap() as u16;
        XGate {
            target: map(g.target),
            comp: g.comp,
            ctrls: g.ctrls.iter().map(|&(w, p)| (map(w), p)).collect(),
        }
    };
    let bg: Vec<XGate> = before.iter().map(dense).collect();
    let ag: Vec<XGate> = after.iter().map(dense).collect();

    let total: u64 = 1u64 << k;
    // 64 assignments per batch: lane l is assignment v + l, i.e. bit l of wire
    // i's lane word is ((v+l)>>i)&1. With v a multiple of 64 the low six wires
    // are the constant patterns below and each higher wire is all-ones or
    // all-zeros according to bit i of v (no carry: l < 64).
    const LANE: [u64; 6] = [
        0xAAAA_AAAA_AAAA_AAAA,
        0xCCCC_CCCC_CCCC_CCCC,
        0xF0F0_F0F0_F0F0_F0F0,
        0xFF00_FF00_FF00_FF00,
        0xFFFF_0000_FFFF_0000,
        0xFFFF_FFFF_0000_0000,
    ];
    let mut st_b = vec![0u64; k];
    let mut st_a = vec![0u64; k];
    let mut v = 0u64;
    while v < total {
        for i in 0..k.min(6) {
            st_b[i] = LANE[i];
        }
        for i in 6..k {
            st_b[i] = 0u64.wrapping_sub((v >> i) & 1);
        }
        st_a.copy_from_slice(&st_b);
        crate::circuit::xgate::eval_lanes(&bg, &mut st_b);
        crate::circuit::xgate::eval_lanes(&ag, &mut st_a);
        // Ignore lanes beyond `total` when k < 6.
        let valid: u64 = if total - v >= 64 {
            !0
        } else {
            (1u64 << (total - v)) - 1
        };
        for i in 0..k {
            if (st_b[i] ^ st_a[i]) & valid != 0 {
                return false;
            }
        }
        v += 64;
    }
    true
}

#[cfg(test)]
#[path = "../../../tests/unit/engine/moves/rules/tests.rs"]
mod tests;
