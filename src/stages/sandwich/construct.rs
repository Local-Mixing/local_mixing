//! Sliced sandwich construction; Classic is the full GSS layout.

use crate::circuit::Circuit as CnotCircuit;
use crate::circuit::randomize::random_wire_except;
use crate::circuit::{CircuitSeq, xgate::XGate};
use rand::{Rng, prelude::SliceRandom};

/// Which sliced-sandwich construction to build: the choice bit selecting the
/// classic layout or its balanced mirror. See [`sliced_sandwich_cnot`] for the
/// full slice algebra of each.
///
/// * [`SandwichVariant::Classic`] — N copies UP (`y ^= x`), D shares the low
///   half with C, and both slice blocks target the low half. The high half is
///   a pure answer register, and BOTH directions read out there on the SAME
///   zero slice `y = 0`.
/// * [`SandwichVariant::Balanced`] — N copies DOWN (`x ^= y`) and D sits on
///   the high half, so each half hosts one computation. S2 mirrors with D
///   (targets the high half, reads the low half), which is what keeps a slice
///   contract at all. Forward reads out on the LOW half at `y = 0`; the
///   inverse reads out on the HIGH half at the MIRRORED slice `x = 0`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SandwichVariant {
    #[default]
    Classic,
    Balanced,
}

impl SandwichVariant {
    /// Parse the CLI / positional-argument spelling.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "classic" | "original" => Some(Self::Classic),
            "balanced" | "mirror" => Some(Self::Balanced),
            _ => None,
        }
    }

    /// The canonical spelling, for logs and provenance lines.
    pub fn name(self) -> &'static str {
        match self {
            Self::Classic => "classic",
            Self::Balanced => "balanced",
        }
    }

    pub fn is_balanced(self) -> bool {
        self == Self::Balanced
    }
}

/// Default slice-block size s = round(n * log2 n), floored at n.
pub fn sandwich_default_s(n: usize) -> usize {
    ((n as f64) * (n as f64).log2()).round().max(n as f64) as usize
}

/// Default D-computation size m = round(n * (log2 n)^2), floored at n.
pub fn sandwich_default_m(n: usize) -> usize {
    let l = (n as f64).log2();
    ((n as f64) * l * l).round().max(n as f64) as usize
}

/// Resolve a generated or supplied G57 source before building the sandwich.
/// Reseeding even for supplied sources preserves historical RNG behavior.
pub fn prepare_source(
    n: usize,
    gate_count: usize,
    seed: u64,
    supplied: Option<CircuitSeq>,
) -> Result<CircuitSeq, String> {
    fastrand::seed(seed);
    match supplied {
        Some(source) => {
            if source.gates.is_empty() {
                return Err("decoded to an empty circuit".into());
            }
            let width = source.max_wire() + 1;
            if width > n {
                return Err(format!("references wire {} but n={n}", width - 1));
            }
            if source.gates.len() != gate_count {
                return Err(format!(
                    "has {} gates but m_C={gate_count}; pass the matching m_C so the sandwich and the resource plan are sized correctly",
                    source.gates.len()
                ));
            }
            Ok(source)
        }
        None => Ok(crate::circuit::random_circuit(n, gate_count)),
    }
}

/// Construct using the dedicated sandwich random stream; the source's random
/// stream and the subsequent preprocessing stream stay independent.
pub fn construct_seeded_sandwich(
    source: &CircuitSeq,
    n: usize,
    companion_gates: usize,
    slice_gates: usize,
    variant: SandwichVariant,
    seed: u64,
) -> CnotCircuit {
    use rand::{SeedableRng, rngs::StdRng};
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5150_1CED);
    sliced_sandwich_cnot(source, n, companion_gates, slice_gates, variant, &mut rng)
}

/// One slice block for the sliced-sandwich construction: `s` gates whose
/// targets all lie in ONE half and which each read at least one wire of the
/// OTHER half with positive polarity, so the whole block is dead when that
/// other half is zero. ~1/3 are CNOTs `t_i ^= r_j` (control in the read
/// half); the rest are CCNOTs `t_i ^= t_j & r_k` (one control per half).
///
/// `target_second_half` picks the side. `false` is the classic block —
/// targets on 0..n, reads n..2n, dead when the second half is zero — used for
/// S1 in both variants and for S2 in the classic one. `true` is its mirror —
/// targets on n..2n, reads 0..n, dead when the FIRST half is zero — used for
/// S2 in the balanced variant, where D has moved to the high wires and the
/// block must guard the mirrored slice `x = 0` instead.
fn sandwich_slice_gates(
    n: usize,
    s: usize,
    target_second_half: bool,
    rng: &mut impl Rng,
) -> Vec<XGate> {
    let (target_base, read_base) = if target_second_half { (n, 0) } else { (0, n) };
    (0..s)
        .map(|_| {
            let target = target_base + rng.random_range(0..n);
            if rng.random_bool(1.0 / 3.0) {
                let control = read_base + rng.random_range(0..n);
                XGate::cnot(target as u16, control as u16)
            } else {
                let first_control =
                    target_base + random_wire_except(n, &[target - target_base], rng);
                let second_control = read_base + rng.random_range(0..n);
                XGate::conj(
                    target as u16,
                    [(first_control as u16, true), (second_control as u16, true)],
                )
                .expect("sandwich CCNOT controls are distinct")
            }
        })
        .collect()
}

/// `g` with every wire index raised by `offset`, used to place the balanced
/// variant's D block on the high half. A uniform shift preserves the sorted
/// control order, so the result is a well-formed [`XGate`].
fn shift_xgate_wires(g: &XGate, offset: usize) -> XGate {
    let offset = u16::try_from(offset).expect("wire offset fits a wire index");
    XGate {
        target: g.target + offset,
        comp: g.comp,
        ctrls: g.ctrls.iter().map(|&(w, p)| (w + offset, p)).collect(),
    }
}

/// `m` random g57 gates on wires 0..n, as XGates — the same design as the
/// C (source) block, used for the sandwich's random D computation.
fn random_g57_xgates(n: usize, m: usize, rng: &mut impl Rng) -> Vec<XGate> {
    assert!(n >= 3, "random g57 gates need n >= 3 wires");
    (0..m)
        .map(|_| {
            let a = rng.random_range(0..n);
            let x = random_wire_except(n, &[a], rng);
            let y = random_wire_except(n, &[a, x], rng);
            XGate::from_g57([a as u16, x as u16, y as u16])
        })
        .collect()
}

/// A uniformly random interleaving of `computation` and `slice` that
/// preserves the internal order of each. `computation`'s order is a hard
/// constraint (it must still compute its function); `slice`'s order is
/// immaterial on the zero slice (all its gates are dead there) but its
/// gates must not be reordered relative to the shuffle already applied.
fn random_interleave(computation: Vec<XGate>, slice: Vec<XGate>, rng: &mut impl Rng) -> Vec<XGate> {
    let mut out = Vec::with_capacity(computation.len() + slice.len());
    let (mut ci, mut si) = (0usize, 0usize);
    while ci < computation.len() || si < slice.len() {
        let rem_c = computation.len() - ci;
        let rem_s = slice.len() - si;
        let take_computation = si >= slice.len()
            || (ci < computation.len() && rng.random_range(0..rem_c + rem_s) < rem_c);
        if take_computation {
            out.push(computation[ci].clone());
            ci += 1;
        } else {
            out.push(slice[si].clone());
            si += 1;
        }
    }
    out
}

/// Slide the gate at `pos` in one direction via adjacent swaps as far as it
/// can go — until the neighbor truly collides per [`XGate::collides`]
/// (commute unless proven otherwise) or the circuit end. Returns the final
/// position. Function-preserving: every hop is an adjacent commuting swap.
fn float_extremal(gates: &mut [XGate], mut pos: usize, dir_left: bool) -> usize {
    if dir_left {
        while pos > 0 && !XGate::collides(&gates[pos], &gates[pos - 1]) {
            gates.swap(pos, pos - 1);
            pos -= 1;
        }
    } else {
        while pos + 1 < gates.len() && !XGate::collides(&gates[pos], &gates[pos + 1]) {
            gates.swap(pos, pos + 1);
            pos += 1;
        }
    }
    pos
}

/// The **sliced sandwich** construction on 2n wires (first half = x, second
/// half = y), in either of its two variants (the `variant` choice bit).
///
/// # Classic ([`SandwichVariant::Classic`])
///
///   A = [ C interleaved with S1 ] ; N(y ^= x) ; [ D interleaved with S2 ]
///
/// C is the source circuit and D a fresh random circuit of `m` g57 gates,
/// BOTH on wires 0..n; N copies the low half up (`y ^= x`, n CNOTs); S1 and
/// S2 are independent slice blocks of `s` gates each
/// ([`sandwich_slice_gates`], both targeting the low half and reading the
/// high one), randomly interleaved with C and D respectively.
///
/// On the zero slice the second half carries the answer:
///   A(x, 0) = (junk, C(x)),
/// because S1 is dead during C (second half still 0) and S2, though live
/// during D (second half already holds C(x)), only targets the junk first
/// half. Symmetrically the inverse gives A^-1(p, 0) = (junk, D^-1(p)): there
/// S2 is dead and S1 fires, so neither direction hands out its computation
/// on a wrong slice, and neither reveals the other's function in the clear.
/// Both directions are sliced at `y = 0` and both read out on the high half;
/// the low half is a pure workspace and the high half a pure answer register.
///
/// # Balanced ([`SandwichVariant::Balanced`])
///
///   A = [ C interleaved with S1 ] ; N(x ^= y) ; [ D' interleaved with S2' ]
///
/// The N column flips (`x ^= y`: the high half is XORed DOWN into the low
/// one) and D moves to the high half, so each half hosts one computation
/// instead of both sharing the low one. S2 mirrors with D — targets on
/// n..2n, reads 0..n — which is forced: with D on the high wires, a
/// low-targeting S2 would fire as soon as D makes the high half nonzero and
/// would overwrite the answer that now stays in the low half. Block 1 (C, S1)
/// and the interleaving and float stages are untouched.
///
/// The slice contract mirrors along with it. Forward, on `y = 0`:
///   A(x, 0) = (C(x), junk),
/// because S1 is dead through C (nothing writes the high half before N), N is
/// then a no-op, and block 2 writes only the high half — so C(x) simply stays
/// where block 1 left it. Backward, on the MIRRORED slice `x = 0`:
///   A^-1(0, q) = (junk, D^-1(q)),
/// because S2 is dead with the low half zero, D^-1 runs cleanly on the high
/// half, and the reversed block 1 (C^-1 with S1 now firing) junks only the
/// low half. Each direction has its own slice, its own computation lane, and
/// its own answer half.
///
/// Off-slice the answer half is masked exactly as in the classic variant:
/// the flipped N step XORs y into x, so the low output is C'(x, y) ^ y for
/// the S1-disturbed computation C', and no nonzero slice yields clean C(x).
///
/// A final float stage then slides each N CNOT in a random direction as far
/// as commutation allows, dissolving the middle column into a band (see the
/// stage comment in [`sliced_sandwich_with_d`]).
pub fn sliced_sandwich_cnot(
    main: &CircuitSeq,
    n: usize,
    m: usize,
    s: usize,
    variant: SandwichVariant,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let d_gates = random_g57_xgates(n, m, rng);
    sliced_sandwich_with_d(main, &d_gates, n, s, variant, rng)
}

/// Sliced sandwich with an explicit D block (given as XGates on wires 0..n,
/// shifted onto the high half internally when `variant` is balanced).
/// Used when C and D must be shared with another pipeline (e.g. an A/B against
/// the legacy compose_a sandwich on the same C, D). See
/// [`sliced_sandwich_cnot`] for the semantics.
pub fn sliced_sandwich_with_d(
    main: &CircuitSeq,
    d_gates: &[XGate],
    n: usize,
    s: usize,
    variant: SandwichVariant,
    rng: &mut impl Rng,
) -> CnotCircuit {
    sliced_sandwich_build(main, d_gates, n, s, variant, rng).0
}

/// Shared builder behind [`sliced_sandwich_cnot`] and
/// [`sliced_sandwich_with_d`], additionally returning the FINAL (post-float)
/// positions of the N column's CNOTs in ascending order. The column is not
/// recoverable from the finished gate list in the balanced variant — there
/// the N gates target the low half and read the high one, exactly the shape
/// of S1's CNOTs — so the band regression tests read the tracked positions
/// instead of filtering by target half.
fn sliced_sandwich_build(
    main: &CircuitSeq,
    d_gates: &[XGate],
    n: usize,
    s: usize,
    variant: SandwichVariant,
    rng: &mut impl Rng,
) -> (CnotCircuit, Vec<usize>) {
    assert!(n >= 3, "sliced_sandwich_with_d requires n >= 3");
    assert!(2 * n <= u16::MAX as usize, "too many wires");
    assert!(
        main.gates.iter().flatten().all(|&wire| (wire as usize) < n),
        "source wire outside 0..n"
    );
    assert!(
        d_gates
            .iter()
            .all(|g| { (g.target as usize) < n && g.ctrls.iter().all(|&(w, _)| (w as usize) < n) }),
        "D wire outside 0..n"
    );
    let total = 2 * n;
    let balanced = variant.is_balanced();

    // Block 1: C (the source) interleaved with S1. Identical in both
    // variants — C always computes on the low half, and S1 always targets
    // the low half and reads the high one, so it is dead on the forward
    // slice and fires under reverse evaluation.
    let c_gates: Vec<XGate> = main.gates.iter().map(|&g| XGate::from_g57(g)).collect();
    let s1 = sandwich_slice_gates(n, s, false, rng);
    let mut out = random_interleave(c_gates, s1, rng);

    // N step: y ^= x (classic, the answer is copied UP into the register) or
    // x ^= y (balanced, the register is XORed DOWN onto the answer, which is
    // already in place). Either way the answer half ends up masked by the
    // other half off-slice, and untouched on the slice.
    let n_start = out.len();
    for i in 0..n {
        out.push(if balanced {
            XGate::cnot(i as u16, (n + i) as u16)
        } else {
            XGate::cnot((n + i) as u16, i as u16)
        });
    }

    // Block 2: D interleaved with S2. The balanced variant lifts D onto the
    // high half and mirrors S2 with it (targets high, reads low), so block 2
    // writes only the high half and the forward answer in the low half is
    // frozen from N onwards.
    let s2 = sandwich_slice_gates(n, s, balanced, rng);
    let d_placed: Vec<XGate> = if balanced {
        d_gates.iter().map(|g| shift_xgate_wires(g, n)).collect()
    } else {
        d_gates.to_vec()
    };
    out.extend(random_interleave(d_placed, s2, rng));

    // Final float stage: the N column is the sandwich's most
    // structure-revealing part (the C|N|D seam). Each of its CNOTs is
    // ASSIGNED an independent random direction, registered up front, and
    // then floats in that direction as far as commutation allows — deep
    // into C/S1 or D/S2 wherever its wires stay cold — dissolving the
    // column into a wide band before gadgetizing. The registered direction
    // matters: float passes repeat until a fixpoint, and a gate always
    // continues in ITS direction, so gates never oscillate and any gate
    // unblocked by another's departure keeps drifting the same way. The N
    // gates mutually commute in both variants (distinct targets, and every
    // control sits in the opposite half from every target), so they pass
    // each other freely; every hop is a commuting swap, so A's function and
    // all slice/inverse guarantees are unchanged.
    // The column is addressed by its recorded insertion range rather than by
    // "targets the high half": that filter identifies exactly the N gates in
    // the classic variant only, and picks out D/S2 instead in the balanced
    // one. Over the classic layout the two agree gate for gate and draw for
    // draw, so classic seeds still reproduce their circuits bit for bit.
    let mut floaters: Vec<(usize, bool)> = (n_start..n_start + n)
        .map(|i| (i, rng.random_bool(0.5)))
        .collect();
    floaters.shuffle(rng);
    // One pass reaches the fixpoint of same-direction floating: the blockers
    // are static (only N gates move) and the floaters mutually commute, so
    // once each has floated to its extreme, further passes could only swap
    // commuting floaters among themselves — a functional no-op, not travel.
    for k in 0..floaters.len() {
        let (p, dir_left) = floaters[k];
        let q = float_extremal(&mut out, p, dir_left);
        floaters[k].0 = q;
        for (idx, (r, _)) in floaters.iter_mut().enumerate() {
            if idx == k {
                continue;
            }
            if q < p && *r >= q && *r < p {
                *r += 1;
            } else if q > p && *r > p && *r <= q {
                *r -= 1;
            }
        }
    }

    let mut positions: Vec<usize> = floaters.iter().map(|&(p, _)| p).collect();
    positions.sort_unstable();

    (
        CnotCircuit {
            gates: out,
            num_wires: total,
        },
        positions,
    )
}

#[cfg(test)]
#[path = "../../../tests/stages/sandwich/construct.rs"]
mod tests;
