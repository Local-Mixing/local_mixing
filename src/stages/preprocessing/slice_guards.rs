//! Zero-slice guards for the full TDP input and output ports.

use crate::circuit::Circuit as CnotCircuit;
use crate::circuit::randomize::random_wire_except;
use crate::circuit::xgate::XGate;
use rand::{Rng, prelude::SliceRandom};

/// The junk-half slice guard (used at both TDP ports): a slice
/// block — identity exactly on the zero slice, every nonzero slice perturbs
/// the data — whose targets are restricted to the LOW half of the data
/// wires (the sandwich's forward-junk half). Under the symmetric-ports
/// design this generator is drawn independently for BOTH ports:
///
/// - at the OUTPUT port the forward-honest run arrives with a junked band,
///   so the guard fires and must not touch the live payload on the upper
///   half;
/// - at the INPUT port the guard is dead on the honest forward slice
///   regardless of targets, but its INVERSE runs last under reverse
///   evaluation and fires on the then-junk band — an upper-half target
///   there would junk the reverse payload D^-1(a), which by the sandwich's
///   reverse contract A^-1(a,0) = (junk, D^-1(a)) has just emerged on the
///   upper half.
///
/// With both guards junk-half-only, the composite is REVERSE-HONEST: the
/// reversed gadget on (a, 0-upper, 0-band) reproduces the reversed source's
/// upper half (see `symmetric_guards_make_reverse_evaluation_honest`), so
/// the same artifact evaluates C(x) forward and D^-1(a) backward, each on
/// its own zero slice — the gadget-level mirror of the sandwich's symmetry.
pub fn slice_zero_junk_guard_dims(
    n: usize,
    nondata: usize,
    gate_count: usize,
    rng: &mut impl Rng,
) -> CnotCircuit {
    try_slice_zero_block_dims(n, 0, n / 2, nondata, gate_count, rng)
        .unwrap_or_else(|error| panic!("{error}"))
}

/// The balanced-sandwich counterpart of [`slice_zero_junk_guard_dims`]: an
/// otherwise-identical junk-half slice guard whose targets are restricted to the
/// HIGH data half (`n/2..n`) instead of the low half. The balanced sandwich
/// carries its forward payload C(x) on the LOW half (classic keeps it on the
/// upper), so at the OUTPUT port the CLOSING guard must junk the HIGH
/// (forward-junk) half, leaving the low-half payload intact. Band-gating (identity
/// iff the band is 0) is unchanged — only which data half is disturbed differs.
/// The OPENING guard stays low-half (its reverse still junks low), so this pairs
/// with an unmodified [`slice_zero_junk_guard_dims`] on the input port.
pub fn slice_zero_junk_guard_dims_high(
    n: usize,
    nondata: usize,
    gate_count: usize,
    rng: &mut impl Rng,
) -> CnotCircuit {
    try_slice_zero_block_dims(n, n / 2, n, nondata, gate_count, rng)
        .unwrap_or_else(|error| panic!("{error}"))
}

/// Shared generator for the opening/closing slice blocks: targets are drawn
/// from `target_lo..target_hi`, slice controls from the `nondata` wires above
/// `n`. Classic guards target the LOW data half (`0..n/2`); the balanced
/// closing guard targets the HIGH half (`n/2..n`) — see
/// [`slice_zero_junk_guard_dims`] and [`slice_zero_junk_guard_dims_high`].
pub(crate) fn try_slice_zero_block_dims(
    n: usize,
    target_lo: usize,
    target_hi: usize,
    nondata: usize,
    gate_count: usize,
    rng: &mut impl Rng,
) -> Result<CnotCircuit, String> {
    if n < 3 {
        return Err(format!(
            "slice_zero_ccnot_preblock requires n >= 3, got {n}"
        ));
    }
    if !(3..=n).contains(&target_hi) || target_lo >= target_hi {
        return Err(format!(
            "slice block targets need 3 <= target_hi <= {n} and target_lo < target_hi, \
             got [{target_lo}, {target_hi})"
        ));
    }
    let total = n
        .checked_add(nondata)
        .ok_or_else(|| "slice-zero preblock wire-count overflow".to_string())?;
    if total > u16::MAX as usize {
        return Err(format!(
            "slice-zero preblock needs {total} wires; capacity is {}",
            u16::MAX
        ));
    }
    let band = nondata.saturating_sub(n);
    if gate_count < nondata {
        return Err(format!(
            "every non-data wire must be read: needs at least {nondata} gates, got {gate_count}"
        ));
    }

    // Shape mix: a third CNOTs, the rest split between one and two data
    // controls. Two data controls need three distinct wires with the target.
    let cnots = gate_count / 3;
    let rest = gate_count - cnots;
    let quads = if n >= 3 { rest / 2 } else { 0 };
    let ccnots = rest - quads;

    for _ in 0..1000 {
        // Balanced slice-control assignment, then shuffled.
        let mut slice_ctrl: Vec<usize> = (0..gate_count).map(|i| n + i % nondata).collect();
        slice_ctrl.shuffle(rng);
        let mut gates: Vec<XGate> = Vec::with_capacity(gate_count);
        for (i, &w) in slice_ctrl.iter().enumerate() {
            let target = rng.random_range(target_lo..target_hi);
            let data_ctrls = if i < cnots {
                0
            } else if i < cnots + ccnots {
                1
            } else {
                2
            };
            let mut lits: Vec<(u16, bool)> = Vec::with_capacity(data_ctrls + 1);
            let mut taken = vec![target];
            for _ in 0..data_ctrls {
                let c = random_wire_except(n, &taken, rng);
                taken.push(c);
                lits.push((c as u16, true));
            }
            lits.push((w as u16, true));
            gates.push(XGate::conj(target as u16, lits).expect("preblock wires are distinct"));
        }
        gates.shuffle(rng);
        debug_assert_eq!(gates.len(), gate_count);
        let ok = if nondata + n <= 20 {
            slice_preblock_fixes_only_zero_slice(&gates, n, nondata)
        } else {
            slice_preblock_spot_check(&gates, n, total, rng)
        };
        if ok {
            return Ok(CnotCircuit {
                gates,
                num_wires: total,
            });
        }
    }
    Err(format!(
        "no slice preblock with every nonzero slice disturbed found at n={n} \
         band={band} gates={gate_count} in 1000 draws: {n} data wires may be too \
         few to disturb 2^{nondata} slices distinctly — raise the data width or reduce the auxiliary width"
    ))
}

/// Exhaustive check that only the all-zero slice leaves the data untouched:
/// every slice against every input. Affordable only while `2n + band` is
/// small, which is exactly the regime where wrong-slice fixes were ever
/// observed in the first place.
pub(crate) fn slice_preblock_fixes_only_zero_slice(
    gates: &[XGate],
    n: usize,
    nondata: usize,
) -> bool {
    let mask = (1u64 << n) - 1;
    (1..(1u64 << nondata)).all(|s| {
        (0..=mask).any(|x| crate::circuit::xgate::eval_u64(gates, x | (s << n)) & mask != x)
    })
}

/// Sampled version for widths the exhaustive check cannot reach: every
/// single-wire slice (the ones firing fewest gates, hence likeliest to
/// cancel), many weight-2 slices, and random slices, each against 64
/// bit-sliced random inputs at once. A spot check, not a proof.
fn slice_preblock_spot_check(gates: &[XGate], n: usize, total: usize, rng: &mut impl Rng) -> bool {
    let disturbs = |hot: &[usize], rng: &mut dyn rand::RngCore| {
        // Lane l = sample l: 64 random inputs at once, with the hot slice
        // wires held at 1 across every lane and the rest at 0.
        let mut state = vec![0u64; total];
        for lane in state.iter_mut().take(n) {
            *lane = rng.next_u64();
        }
        let input: Vec<u64> = state[..n].to_vec();
        for &w in hot {
            state[w] = !0u64;
        }
        for g in gates {
            g.apply_lanes(&mut state);
        }
        (0..n).any(|w| state[w] != input[w])
    };
    for w in n..total {
        if !disturbs(&[w], rng) {
            return false;
        }
    }
    for _ in 0..512 {
        let a = rng.random_range(n..total);
        let b = loop {
            let b = rng.random_range(n..total);
            if b != a {
                break b;
            }
        };
        if !disturbs(&[a, b], rng) {
            return false;
        }
    }
    for _ in 0..512 {
        let hot: Vec<usize> = (n..total).filter(|_| rng.random_bool(0.5)).collect();
        if !hot.is_empty() && !disturbs(&hot, rng) {
            return false;
        }
    }
    true
}

/// Punchlist #7 (RC 2026-10-06): the accumulate-and-inject slice guard.
/// Same zero-slice contract as [`try_slice_zero_block_dims`] — identity
/// exactly on the zero slice, every nonzero slice perturbs the data, targets
/// restricted to the junk half — with NO gate wider than two controls, so
/// the whole block lives in the vocabulary db_mixing digests.
///
/// Three gate populations:
/// - PROLOGUE (first ~third, at least one gate per band wire, balanced):
///   the original guard's width-<=2 shapes, `d ^= s` and `d ^= d_c & s`
///   with `s` a band slice control. Dead on the zero slice; gives every
///   single-hot band wire a direct data disturbance.
/// - ACCUMULATE / UNCOMPUTE: `t ^= d_i & d_j` with `t` the live round's
///   accumulator band wire and `d_i, d_j` drawn from the data wires the
///   guard never writes (the complement of the target range), so the
///   closing uncompute cancels exactly. Fires on every slice; only dirties
///   band junk, which the port contract treats as junk anyway.
/// - INJECTIONS: `d ^= t & s` with `t` a dirty accumulator and `s` a
///   TRIGGER — a band wire clean at that point (never written, or already
///   uncomputed). On the zero slice every trigger still holds its port 0,
///   so the data is untouched; on nonzero slices the injection deposits
///   degree-3 data-by-data products gated by the band — the same algebraic
///   degree the width-3 quads used to provide.
///
/// Round discipline: disjoint accumulator sets per round, each round fully
/// uncomputed before the next begins, triggers drawn outside the live
/// round's accumulators. Rounds need at least three band wires; narrower
/// bands fall back to an all-prologue block (still a valid guard).
pub(crate) fn try_slice_zero_feistel_dims(
    n: usize,
    target_lo: usize,
    target_hi: usize,
    nondata: usize,
    gate_count: usize,
    rng: &mut impl Rng,
) -> Result<CnotCircuit, String> {
    if n < 3 {
        return Err(format!("slice feistel guard requires n >= 3, got {n}"));
    }
    if !(3..=n).contains(&target_hi) || target_lo >= target_hi {
        return Err(format!(
            "slice guard targets need 3 <= target_hi <= {n} and target_lo < target_hi, \
             got [{target_lo}, {target_hi})"
        ));
    }
    let total = n
        .checked_add(nondata)
        .ok_or_else(|| "slice guard wire-count overflow".to_string())?;
    if total > u16::MAX as usize {
        return Err(format!(
            "slice guard needs {total} wires; capacity is {}",
            u16::MAX
        ));
    }
    if gate_count < nondata {
        return Err(format!(
            "every non-data wire must be read: needs at least {nondata} gates, got {gate_count}"
        ));
    }
    // Accumulator sources: data wires the guard never writes.
    let sources: Vec<u16> = (0..n)
        .filter(|w| !(target_lo..target_hi).contains(w))
        .map(|w| w as u16)
        .collect();
    let acc_size = (nondata / 8).clamp(2, 16);
    let rounds_possible = sources.len() >= 2 && nondata > acc_size;

    for _ in 0..1000 {
        // Budget: ~a third prologue (floored at one gate per band wire),
        // remainder in accumulate/inject/uncompute rounds; any unroundable
        // remainder tops up the prologue so the total is exact.
        let mut prologue = (gate_count / 3).max(nondata).min(gate_count);
        let mut rounds_budget = gate_count - prologue;
        if !rounds_possible {
            prologue = gate_count;
            rounds_budget = 0;
        }
        let round_floor = 2 * acc_size + 8;
        prologue += rounds_budget % round_floor.max(1);
        // (exact accounting happens below; the modulo just reduces leftovers)
        let mut gates: Vec<XGate> = Vec::with_capacity(gate_count);

        // --- rounds plan first so the prologue can absorb the remainder ---
        let mut planned: Vec<(Vec<u16>, usize)> = Vec::new(); // (acc set, injections)
        let mut left = gate_count - prologue;
        let mut band_order: Vec<u16> = ((n as u16)..(total as u16)).collect();
        while left >= round_floor {
            band_order.shuffle(rng);
            let acc: Vec<u16> = band_order[..acc_size].to_vec();
            let inj = (left - 2 * acc_size).min(6 * acc_size);
            planned.push((acc, inj));
            left -= 2 * acc_size + inj;
        }
        prologue += left;

        // --- prologue: balanced slice-control coverage, width <= 2 ---
        let mut slice_ctrl: Vec<usize> = (0..prologue).map(|i| n + i % nondata).collect();
        slice_ctrl.shuffle(rng);
        for (i, &w) in slice_ctrl.iter().enumerate() {
            let target = rng.random_range(target_lo..target_hi);
            if i % 3 == 0 {
                gates.push(XGate::cnot(target as u16, w as u16));
            } else {
                let c = random_wire_except(n, &[target], rng);
                gates.push(
                    XGate::conj(target as u16, [(c as u16, true), (w as u16, true)])
                        .expect("guard wires are distinct"),
                );
            }
        }

        // --- rounds ---
        for (acc, inj) in &planned {
            let acc_set: std::collections::HashSet<u16> = acc.iter().copied().collect();
            let mut acc_gates: Vec<XGate> = Vec::with_capacity(acc.len());
            for &t in acc {
                let i1 = sources[rng.random_range(0..sources.len())];
                let i2 = loop {
                    let c = sources[rng.random_range(0..sources.len())];
                    if c != i1 {
                        break c;
                    }
                };
                acc_gates
                    .push(XGate::conj(t, [(i1, true), (i2, true)]).expect("distinct acc wires"));
            }
            gates.extend(acc_gates.iter().cloned());
            for _ in 0..*inj {
                let t = acc[rng.random_range(0..acc.len())];
                let s = loop {
                    let w = rng.random_range(n..total) as u16;
                    if !acc_set.contains(&w) {
                        break w;
                    }
                };
                let target = rng.random_range(target_lo..target_hi);
                gates.push(
                    XGate::conj(target as u16, [(t, true), (s, true)])
                        .expect("injection wires are distinct"),
                );
            }
            // Uncompute: replay the same accumulator gates (sources are
            // never written by the guard, so the XORed terms cancel exactly
            // and the accumulators return to their port values).
            gates.extend(acc_gates.into_iter().rev());
        }
        debug_assert_eq!(gates.len(), gate_count);

        let ok = if nondata + n <= 20 {
            slice_preblock_fixes_only_zero_slice(&gates, n, nondata)
        } else {
            slice_preblock_spot_check(&gates, n, total, rng)
        };
        if ok {
            return Ok(CnotCircuit {
                gates,
                num_wires: total,
            });
        }
    }
    Err(format!(
        "no feistel slice guard with every nonzero slice disturbed found at n={n} \
         band={nondata} gates={gate_count} in 1000 draws"
    ))
}

/// Accumulate-and-inject counterpart of [`slice_zero_junk_guard_dims`].
pub fn slice_zero_feistel_guard_dims(
    n: usize,
    nondata: usize,
    gate_count: usize,
    rng: &mut impl Rng,
) -> CnotCircuit {
    try_slice_zero_feistel_dims(n, 0, n / 2, nondata, gate_count, rng)
        .unwrap_or_else(|error| panic!("{error}"))
}

/// Accumulate-and-inject counterpart of [`slice_zero_junk_guard_dims_high`].
pub fn slice_zero_feistel_guard_dims_high(
    n: usize,
    nondata: usize,
    gate_count: usize,
    rng: &mut impl Rng,
) -> CnotCircuit {
    try_slice_zero_feistel_dims(n, n / 2, n, nondata, gate_count, rng)
        .unwrap_or_else(|error| panic!("{error}"))
}

/// Number of logical slice probes per auxiliary wire in managed TDP.
pub const SLICE_ZERO_CCNOT_GATES_PER_WIRE: usize = 10;

/// Nonlinear-TDP counterpart of [`try_slice_zero_block_dims`].
///
/// Uses the same logical probes as the standard guard, decomposed to fan-in
/// two. Nonlinear layouts can have tens of thousands of slice wires, so an
/// indexed, batched checker validates their disturbance without replaying the
/// whole preblock once per singleton slice.
pub(crate) fn try_nonlinear_slice_zero_preblock_dims(
    n: usize,
    nondata: usize,
    gate_count: usize,
    scratch: u16,
    scratch2: u16,
    rng: &mut impl Rng,
) -> Result<CnotCircuit, String> {
    if n < 3 {
        return Err(format!(
            "nonlinear slice-zero preblock requires n >= 3, got {n}"
        ));
    }
    if nondata == 0 {
        return Err("nonlinear slice-zero preblock requires at least one slice wire".to_string());
    }
    let total = n
        .checked_add(nondata)
        .ok_or_else(|| "nonlinear slice-zero preblock wire-count overflow".to_string())?;
    if total > u16::MAX as usize {
        return Err(format!(
            "nonlinear slice-zero preblock needs {total} wires; capacity is {}",
            u16::MAX
        ));
    }
    let band = nondata.saturating_sub(n);
    if gate_count < nondata {
        return Err(format!(
            "every non-data wire must be read: needs at least {nondata} gates, got {gate_count}"
        ));
    }
    for (name, wire) in [("scratch", scratch), ("scratch2", scratch2)] {
        if !(n..total).contains(&(wire as usize)) {
            return Err(format!(
                "nonlinear fan-in-two preblock {name} wire {wire} must be a non-data wire in {n}..{total}"
            ));
        }
    }
    if scratch == scratch2 {
        return Err(format!(
            "nonlinear fan-in-two preblock scratch wires must be distinct, got {scratch} twice"
        ));
    }

    let cnots = gate_count / 3;
    let rest = gate_count - cnots;
    let quads = rest / 2;
    let ccnots = rest - quads;
    let emitted_count = gate_count
        .checked_add(
            quads
                .checked_mul(3)
                .ok_or_else(|| "nonlinear preblock decomposed gate-count overflow".to_string())?,
        )
        .ok_or_else(|| "nonlinear preblock emitted gate-count overflow".to_string())?;

    #[derive(Clone, Copy)]
    struct MacroSpec {
        target: u16,
        slice: u16,
        data: [u16; 2],
        data_len: u8,
    }

    for _ in 0..1000 {
        let mut slice_ctrl = Vec::new();
        slice_ctrl.try_reserve_exact(gate_count).map_err(|error| {
            format!("nonlinear preblock slice-control allocation failed: {error}")
        })?;
        slice_ctrl.extend((0..gate_count).map(|i| n + i % nondata));
        slice_ctrl.shuffle(rng);
        let mut macros = Vec::new();
        macros
            .try_reserve_exact(gate_count)
            .map_err(|error| format!("nonlinear preblock macro allocation failed: {error}"))?;
        for (i, &w) in slice_ctrl.iter().enumerate() {
            let target = rng.random_range(0..n);
            let data_ctrls = if i < cnots {
                0
            } else if i < cnots + ccnots {
                1
            } else {
                2
            };
            let mut data = [0u16; 2];
            if data_ctrls >= 1 {
                data[0] = random_wire_except(n, &[target], rng) as u16;
            }
            if data_ctrls == 2 {
                data[1] = random_wire_except(n, &[target, data[0] as usize], rng) as u16;
            }
            macros.push(MacroSpec {
                target: target as u16,
                slice: w as u16,
                data,
                data_len: data_ctrls as u8,
            });
        }
        macros.shuffle(rng);

        let mut gates = Vec::new();
        gates
            .try_reserve_exact(emitted_count)
            .map_err(|error| format!("nonlinear preblock gate allocation failed: {error}"))?;
        let bucket_capacity = gate_count / nondata + usize::from(gate_count % nondata != 0);
        let mut by_slice = Vec::new();
        by_slice
            .try_reserve_exact(nondata)
            .map_err(|error| format!("nonlinear preblock index allocation failed: {error}"))?;
        for _ in 0..nondata {
            let mut bucket = Vec::new();
            bucket.try_reserve_exact(bucket_capacity).map_err(|error| {
                format!("nonlinear preblock index-bucket allocation failed: {error}")
            })?;
            by_slice.push(bucket);
        }
        for spec in macros {
            let start = gates.len();
            match spec.data_len {
                0 => gates.push(
                    XGate::conj(spec.target, [(spec.slice, true)])
                        .expect("preblock target and slice control are distinct"),
                ),
                1 => gates.push(
                    XGate::conj(spec.target, [(spec.data[0], true), (spec.slice, true)])
                        .expect("preblock target and controls are distinct"),
                ),
                2 => {
                    // Exact dirty-q decomposition of t ^= a*b*c.  q may start
                    // arbitrarily and is restored by the contiguous macro:
                    // q^=ab; t^=qc; q^=ab; t^=qc.
                    let q = if spec.slice == scratch {
                        scratch2
                    } else {
                        scratch
                    };
                    debug_assert_ne!(q, spec.slice);
                    let build_q = XGate::conj(q, [(spec.data[0], true), (spec.data[1], true)])
                        .expect("dirty-q wire is non-data and distinct from data controls");
                    let use_q = XGate::conj(spec.target, [(q, true), (spec.slice, true)])
                        .expect("dirty-q and slice controls are distinct from the data target");
                    gates.push(build_q.clone());
                    gates.push(use_q.clone());
                    gates.push(build_q);
                    gates.push(use_q);
                }
                other => unreachable!("unsupported preblock data-control count {other}"),
            }
            by_slice[spec.slice as usize - n].push((start, gates.len()));
        }
        debug_assert_eq!(gates.len(), emitted_count);
        let ok = if total <= 20 {
            slice_preblock_fixes_only_zero_slice(&gates, n, nondata)
        } else {
            nonlinear_slice_preblock_spot_check(&gates, &by_slice, n, total, rng)
        };
        if ok {
            return Ok(CnotCircuit {
                gates,
                num_wires: total,
            });
        }
    }
    Err(format!(
        "no nonlinear slice preblock with every nonzero slice disturbed found at n={n} \
         band={band} gates={gate_count} in 1000 draws: {n} data wires may be too \
         few to disturb 2^{nondata} slices distinctly"
    ))
}

/// Scalable wide-slice checker used only by the nonlinear TDP adapter.
///
/// Each generated macro has one positive slice control and restores any dirty
/// decomposition scratch before the next macro. With a singleton or pair
/// slice, macros controlled by every other slice wire are therefore identities;
/// indexing the active macro ranges preserves their original order while
/// reducing each check to roughly ten or twenty logical macros.
///
/// The random phase still samples 512 slice values. It packs eight slice
/// nonzero values into disjoint eight-lane groups per `u64` traversal and gives
/// each slice eight independent data inputs, accepting it when at least one
/// lane in its group witnesses a disturbance. This retains the intended per-slice
/// existential test while bounding the random phase at 64 full traversals.
fn nonlinear_slice_preblock_spot_check(
    gates: &[XGate],
    by_slice: &[Vec<(usize, usize)>],
    n: usize,
    total: usize,
    rng: &mut impl Rng,
) -> bool {
    let nondata = total - n;
    if by_slice.len() != nondata
        || by_slice
            .iter()
            .flatten()
            .any(|&(start, end)| start >= end || end > gates.len())
    {
        return false;
    }

    let mut state = vec![0u64; total];
    let mut input = vec![0u64; n];
    let mut disturbs = |active: &[(usize, usize)], hot: &[usize], rng: &mut dyn rand::RngCore| {
        for wire in 0..n {
            state[wire] = rng.next_u64();
        }
        input.copy_from_slice(&state[..n]);
        for &wire in hot {
            state[wire] = !0u64;
        }
        for &(start, end) in active {
            for gate in &gates[start..end] {
                gate.apply_lanes(&mut state);
            }
        }
        let changed = (0..n).any(|wire| state[wire] != input[wire]);
        for &wire in hot {
            state[wire] = 0;
        }
        changed
    };

    for wire in n..total {
        if !disturbs(&by_slice[wire - n], &[wire], rng) {
            return false;
        }
    }

    let mut pair_active = Vec::new();
    if nondata >= 2 {
        for _ in 0..512 {
            let a = rng.random_range(n..total);
            let b = loop {
                let candidate = rng.random_range(n..total);
                if candidate != a {
                    break candidate;
                }
            };
            pair_active.clear();
            let (left, right) = (&by_slice[a - n], &by_slice[b - n]);
            let (mut i, mut j) = (0usize, 0usize);
            while i < left.len() || j < right.len() {
                if j == right.len() || (i < left.len() && left[i].0 < right[j].0) {
                    pair_active.push(left[i]);
                    i += 1;
                } else {
                    pair_active.push(right[j]);
                    j += 1;
                }
            }
            if !disturbs(&pair_active, &[a, b], rng) {
                return false;
            }
        }
    }
    drop(disturbs);

    const SLICES_PER_BATCH: usize = 8;
    const INPUTS_PER_SLICE: usize = 8;
    let mut batch_state = vec![0u64; total];
    let mut batch_input = vec![0u64; n];
    for _ in 0..(512 / SLICES_PER_BATCH) {
        for wire in 0..n {
            let value = rng.next_u64();
            batch_state[wire] = value;
            batch_input[wire] = value;
        }
        batch_state[n..].fill(0);
        let mut nonempty = [false; SLICES_PER_BATCH];
        for (group, is_nonempty) in nonempty.iter_mut().enumerate() {
            let group_mask =
                u64::MAX >> (u64::BITS as usize - INPUTS_PER_SLICE) << (group * INPUTS_PER_SLICE);
            while !*is_nonempty {
                for wire_state in batch_state.iter_mut().take(total).skip(n) {
                    if rng.random_bool(0.5) {
                        *wire_state |= group_mask;
                        *is_nonempty = true;
                    }
                }
            }
        }
        for gate in gates {
            gate.apply_lanes(&mut batch_state);
        }
        let changed = (0..n).fold(0u64, |mask, wire| {
            mask | (batch_state[wire] ^ batch_input[wire])
        });
        for (group, &is_nonempty) in nonempty.iter().enumerate() {
            debug_assert!(is_nonempty);
            let group_mask =
                u64::MAX >> (u64::BITS as usize - INPUTS_PER_SLICE) << (group * INPUTS_PER_SLICE);
            if changed & group_mask == 0 {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/preprocessing/slice_guards.rs"]
mod tests;
