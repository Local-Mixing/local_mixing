//! Sliced sandwich construction; Classic is the full TDP layout.

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

/// Default size t of the V gadget's random circuit: round(n * log2 n),
/// floored at n (the slice-block default).
pub fn sandwich_default_v(n: usize) -> usize {
    sandwich_default_s(n)
}

/// The stage-1 shape knobs (RC, 2026-11-01; docs/notes/stage2_interleaving_20261101.md):
/// whether the slice blocks are interleaved with C and D, how random
/// circuits are sampled, and the V gadget on the payload register.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SandwichShape {
    /// Interleave the slice blocks S1 (with C) and S2 (with D).
    pub slicing: bool,
    /// Gates per layer when sampling a random circuit (the random C, D and
    /// V): each layer is `layer` wire-disjoint gates on 3 * `layer` distinct
    /// wires. 1 is the fully random sequence; at most n / 3.
    pub layer: usize,
    /// Gates of the V gadget's random circuit; 0 = no gadget (the classic
    /// N column of n copy CNOTs). Nonzero: C's outputs are copied into the
    /// register by a nonlinear bijection on groups of three wires, V and V*
    /// run on the register beside D, and the register's own inverse of the
    /// bijection restores C(x) at the end ([`v_gadget_sandwich`]).
    pub v_gates: usize,
}

impl SandwichShape {
    /// The construction before 2026-11-01: slice blocks, fully random
    /// sequences and the N column of copy CNOTs. Reproduces old seeds.
    pub const LEGACY: SandwichShape = SandwichShape {
        slicing: true,
        layer: 1,
        v_gates: 0,
    };

    /// The defaults for `n` source wires: no slicing, fully random
    /// sequences (`layer` 1) and a V gadget of [`sandwich_default_v`] gates.
    pub fn default_for(n: usize) -> Self {
        SandwichShape {
            slicing: false,
            layer: 1,
            v_gates: sandwich_default_v(n),
        }
    }

    /// The constraints [`construct_shaped_sandwich`] relies on.
    pub fn validate(&self, n: usize, variant: SandwichVariant) -> Result<(), String> {
        if self.layer == 0 || 3 * self.layer > n {
            return Err(format!(
                "layer size {}: need 1 <= w <= n/3 = {} (each gate of a layer takes three distinct wires)",
                self.layer,
                n / 3
            ));
        }
        if self.v_gates > 0 && variant.is_balanced() {
            return Err(
                "the V gadget needs the classic sandwich (the balanced variant runs D on the register)"
                    .into(),
            );
        }
        Ok(())
    }
}

/// What a shaped sandwich is made of, for reports and tests.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SandwichLayout {
    pub c_gates: usize,
    pub s1_gates: usize,
    /// The N column (n CNOTs) or the V gadget's nonlinear copy.
    pub copy_gates: usize,
    /// |V| (V* has as many).
    pub v_gates: usize,
    /// The register's inverse of the copy bijection (V gadget only).
    pub recovery_gates: usize,
    pub d_gates: usize,
    pub s2_gates: usize,
    /// Copy groups of the V gadget (0 without it).
    pub copy_groups: usize,
    /// Final positions of the copy gates, ascending.
    pub copy_positions: Vec<usize>,
}

/// Layered sampling: `m` g57 triples `[target, x, y]` on wires `0..n`,
/// drawn `layer` wire-disjoint gates at a time (the 3 * `layer` wires of a
/// layer are a uniformly random sample, grouped in drawing order). A layer
/// repeating a gate of the previous layer (which would cancel it) is
/// redrawn. `draw(k)` returns a uniform integer in `0..k`.
fn layered_triples(
    n: usize,
    m: usize,
    layer: usize,
    mut draw: impl FnMut(usize) -> usize,
) -> Vec<[u16; 3]> {
    assert!(layer >= 1 && 3 * layer <= n, "layer size must be 1..=n/3");
    let mut wires: Vec<u16> = (0..n as u16).collect();
    let mut out: Vec<[u16; 3]> = Vec::with_capacity(m);
    let mut previous: Vec<[u16; 3]> = Vec::new();
    while out.len() < m {
        let size = layer.min(m - out.len());
        let current = loop {
            for i in 0..3 * size {
                let j = i + draw(n - i);
                wires.swap(i, j);
            }
            let current: Vec<[u16; 3]> = (0..size)
                .map(|k| [wires[3 * k], wires[3 * k + 1], wires[3 * k + 2]])
                .collect();
            if !current.iter().any(|g| previous.contains(g)) {
                break current;
            }
        };
        out.extend_from_slice(&current);
        previous = current;
    }
    out
}

/// `m` random g57 gates on wires `0..n` drawn from the source's fastrand
/// stream in layers of `layer` wire-disjoint gates (the layered random C).
pub fn layered_random_circuit(n: usize, m: usize, layer: usize) -> CircuitSeq {
    CircuitSeq {
        gates: layered_triples(n, m, layer, |k| fastrand::usize(..k)),
    }
}

/// `m` random gates on wires `offset..offset + n` from `rng`: g57 gates, or
/// positive two-control conjunctions when `zero_preserving` (a circuit of
/// those maps the zero state to itself). `layer` 1 draws the fully random
/// sequence ([`random_g57_xgates`]'s draws for g57 gates).
fn sampled_xgates(
    n: usize,
    m: usize,
    layer: usize,
    offset: usize,
    zero_preserving: bool,
    rng: &mut impl Rng,
) -> Vec<XGate> {
    let triples: Vec<[u16; 3]> = if layer == 1 {
        (0..m)
            .map(|_| {
                let a = rng.random_range(0..n);
                let x = random_wire_except(n, &[a], rng);
                let y = random_wire_except(n, &[a, x], rng);
                [a as u16, x as u16, y as u16]
            })
            .collect()
    } else {
        layered_triples(n, m, layer, |k| rng.random_range(0..k))
    };
    let o = u16::try_from(offset).expect("wire offset fits a wire index");
    triples
        .into_iter()
        .map(|[a, x, y]| {
            if zero_preserving {
                XGate::conj(a + o, [(x + o, true), (y + o, true)])
                    .expect("distinct conjunction wires")
            } else {
                XGate::from_g57([a + o, x + o, y + o])
            }
        })
        .collect()
}

/// How stage 1 draws the secret source circuit C when none is supplied.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SourceConstruction {
    /// `m` uniformly random g57 gates ([`crate::circuit::random_circuit`])
    /// from the fastrand stream seeded with the source seed.
    #[default]
    Random,
    /// RC's balanced C ([`super::balanced_source`]): every completed internal
    /// value is read exactly twice as a control, no value more than twice,
    /// and targets always have the minimum write count; drawn from a
    /// CPython-exact `random.Random(seed)` stream.
    Balanced,
}

impl SourceConstruction {
    /// Parse the configuration spelling.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "random" => Some(Self::Random),
            "balanced" => Some(Self::Balanced),
            _ => None,
        }
    }

    /// The canonical spelling, for logs and provenance lines.
    pub fn name(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::Balanced => "balanced",
        }
    }

    pub fn is_balanced(self) -> bool {
        self == Self::Balanced
    }
}

/// Resolve a generated or supplied G57 source before building the sandwich.
///
/// The fastrand stream is reseeded first on EVERY path (historical behavior,
/// kept so the random path stays byte-identical): the random construction
/// then draws C from it, while a supplied source and the balanced
/// construction leave it freshly seeded and unconsumed — the balanced
/// builder draws from its own CPython-exact stream keyed by the same `seed`.
/// A supplied source cannot be combined with the balanced construction.
pub fn prepare_source(
    n: usize,
    gate_count: usize,
    seed: u64,
    supplied: Option<CircuitSeq>,
    construction: SourceConstruction,
) -> Result<CircuitSeq, String> {
    prepare_source_layered(n, gate_count, seed, supplied, construction, 1)
}

/// [`prepare_source`] with the random construction drawn in layers of
/// `layer` wire-disjoint gates ([`layered_random_circuit`]; `layer` 1 is
/// the fully random sequence, byte-identical to [`prepare_source`]). A
/// supplied or balanced source is not sampled gate by gate, so `layer`
/// leaves it unchanged.
pub fn prepare_source_layered(
    n: usize,
    gate_count: usize,
    seed: u64,
    supplied: Option<CircuitSeq>,
    construction: SourceConstruction,
    layer: usize,
) -> Result<CircuitSeq, String> {
    fastrand::seed(seed);
    if layer > 1 && supplied.is_none() && construction == SourceConstruction::Random {
        if 3 * layer > n {
            return Err(format!(
                "layer size {layer}: need 1 <= w <= n/3 = {}",
                n / 3
            ));
        }
        return Ok(layered_random_circuit(n, gate_count, layer));
    }
    match (supplied, construction) {
        (Some(_), SourceConstruction::Balanced) => {
            Err("source.path and the balanced construction are exclusive".into())
        }
        (Some(source), SourceConstruction::Random) => {
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
        (None, SourceConstruction::Random) => Ok(crate::circuit::random_circuit(n, gate_count)),
        (None, SourceConstruction::Balanced) => super::balanced_source(n, gate_count, seed),
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

/// [`construct_seeded_sandwich`] with the stage-1 shape knobs, on the same
/// dedicated stream; [`SandwichShape::LEGACY`] reproduces it gate for gate.
/// D is drawn first (layered when `shape.layer > 1`), then the blocks: the
/// classic column ([`sliced_sandwich_cnot`], slice blocks optional) when
/// `shape.v_gates` is 0, the V gadget ([`v_gadget_sandwich`]) otherwise.
pub fn construct_shaped_sandwich(
    source: &CircuitSeq,
    n: usize,
    companion_gates: usize,
    slice_gates: usize,
    variant: SandwichVariant,
    seed: u64,
    shape: SandwichShape,
) -> Result<(CnotCircuit, SandwichLayout), String> {
    use rand::{SeedableRng, rngs::StdRng};
    shape.validate(n, variant)?;
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5150_1CED);
    Ok(shaped_sandwich(
        source,
        n,
        companion_gates,
        slice_gates,
        variant,
        shape,
        &mut rng,
    ))
}

/// The shaped sandwich on an explicit stream (see [`construct_shaped_sandwich`]).
pub fn shaped_sandwich(
    main: &CircuitSeq,
    n: usize,
    m: usize,
    s: usize,
    variant: SandwichVariant,
    shape: SandwichShape,
    rng: &mut impl Rng,
) -> (CnotCircuit, SandwichLayout) {
    let d_gates = if shape.layer == 1 {
        random_g57_xgates(n, m, rng)
    } else {
        sampled_xgates(n, m, shape.layer, 0, false, rng)
    };
    if shape.v_gates == 0 {
        let (circuit, positions) =
            sliced_sandwich_build_opt(main, &d_gates, n, s, variant, shape.slicing, rng);
        let slice = if shape.slicing { s } else { 0 };
        let layout = SandwichLayout {
            c_gates: main.gates.len(),
            s1_gates: slice,
            copy_gates: n,
            d_gates: d_gates.len(),
            s2_gates: slice,
            copy_positions: positions,
            ..SandwichLayout::default()
        };
        (circuit, layout)
    } else {
        v_gadget_sandwich(main, &d_gates, n, s, shape, rng)
    }
}

/// A random partition of the wires `0..n` into copy groups of three, the
/// last group taking the remainder (3, 4 or 5 wires); each group lists its
/// wires in a random order (the first is the group's selector).
fn copy_groups(n: usize, rng: &mut impl Rng) -> Vec<Vec<u16>> {
    assert!(n >= 3, "the V gadget needs n >= 3");
    let mut wires: Vec<u16> = (0..n as u16).collect();
    wires.shuffle(rng);
    let count = n / 3;
    (0..count)
        .map(|k| {
            let end = if k + 1 == count { n } else { 3 * k + 3 };
            wires[3 * k..end].to_vec()
        })
        .collect()
}

/// The V gadget's copy of one group and the register's inverse of it.
///
/// For the group's low wires (a; x_1..x_m), m >= 2, with register wires
/// H(v) = n + v, the copy XORs into the register the bijection
///
///   H(x_1) ^= mux(a; x_1^q1, x_2^p0)      H(a) ^= mux(a; x_1^q1, x_2^!p0)
///   H(x_i) ^= mux(a; x_i^qi, x_pi(i)^pi)  for i = 2..m,
///
/// mux(a; u, v) = (!a & u) ^ (a & v), each realized by two two-control
/// conjunctions (no CNOT, never a single-literal value on a wire), where
/// pi(2) = 1 for m = 2 and otherwise pi(2) = m, pi(3) = 1, pi(i) = i - 1:
/// every coordinate is a balanced degree-2 function of three of the group's
/// values. Superscripts are literal polarities, random unless
/// `zero_preserving` (then the a = 0 branches are positive, so the copy
/// writes zero when C's outputs are zero).
///
/// The recovery acts on the register alone and inverts the copy on the
/// zero slice: H(a) ^= H(x_1) leaves a there; per-wire CNOTs from H(a)
/// make both branches positive; controlled swaps on a put each x_i back on
/// H(x_i). Off the zero slice the register keeps nonlinear junk.
fn group_copy(
    group: &[u16],
    n: usize,
    zero_preserving: bool,
    rng: &mut impl Rng,
) -> (Vec<XGate>, Vec<XGate>) {
    let m = group.len() - 1;
    assert!(m >= 2, "copy groups have at least three wires");
    let off = n as u16;
    let a = group[0];
    let x = |i: usize| group[i]; // x_i, i = 1..=m
    let pi = |i: usize| -> usize {
        match i {
            2 if m == 2 => 1,
            2 => m,
            3 => 1,
            _ => i - 1,
        }
    };
    // branch-0 polarity: positive when the copy must preserve zero
    let polarity =
        |rng: &mut dyn rand::RngCore| -> bool { zero_preserving || rng.random_bool(0.5) };
    // (register wire, branch-0 literal, branch-1 literal) per coordinate
    type Literal = (u16, bool);
    let mut coords: Vec<(u16, Literal, Literal)> = Vec::with_capacity(m + 1);
    let q1 = polarity(rng);
    let p0 = rng.random_bool(0.5);
    coords.push((off + x(1), (x(1), q1), (x(2), p0)));
    coords.push((off + a, (x(1), q1), (x(2), !p0)));
    for i in 2..=m {
        let qi = polarity(rng);
        let p = rng.random_bool(0.5);
        coords.push((off + x(i), (x(i), qi), (x(pi(i)), p)));
    }
    let mut copy: Vec<XGate> = Vec::with_capacity(2 * coords.len());
    for &(h, l0, l1) in &coords {
        copy.push(XGate::conj(h, [(a, false), l0]).expect("distinct copy wires"));
        copy.push(XGate::conj(h, [(a, true), l1]).expect("distinct copy wires"));
    }
    copy.shuffle(rng);

    let ha = off + a;
    let mut recovery = vec![XGate::cnot(ha, off + x(1))];
    // after the first step, H(x_1) holds (q1 on a = 0, p0 on a = 1) and
    // H(x_i) holds (qi, pi); make both branches positive
    let mut branches: Vec<(u16, bool, bool)> = vec![(off + x(1), q1, p0)];
    for (k, &(h, l0, l1)) in coords.iter().enumerate().skip(2) {
        debug_assert_eq!(k, branches.len() + 1);
        branches.push((h, l0.1, l1.1));
    }
    for &(h, q, p) in &branches {
        // correction c(a) = [q negative] * !a  ^  [p negative] * a
        if !q {
            recovery.push(XGate::conj(h, [(ha, false)]).expect("distinct wires"));
        }
        if !p {
            recovery.push(XGate::cnot(h, ha));
        }
    }
    // on a = 1, register wire H(x_i) now holds x_rho(i): rho(1) = 2,
    // rho(i) = pi(i); controlled swaps sort the contents home
    let mut content: Vec<usize> = (0..=m).collect();
    content[1] = 2;
    for (i, slot) in content.iter_mut().enumerate().skip(2) {
        *slot = pi(i);
    }
    let reg = |i: usize| off + x(i);
    for i in 1..=m {
        while content[i] != i {
            let j = content[i];
            // controlled swap of H(x_i), H(x_j) on H(a)
            recovery.push(XGate::cnot(reg(j), reg(i)));
            recovery.push(XGate::conj(reg(i), [(ha, true), (reg(j), true)]).expect("distinct"));
            recovery.push(XGate::cnot(reg(j), reg(i)));
            content.swap(i, j);
        }
    }
    (copy, recovery)
}

/// The classic sandwich with the V gadget on the payload register:
///
///   A = [ C (with S1) ] ; copy ; [ D (with S2) beside V, V*, recovery ]
///
/// The copy writes a nonlinear bijection of C's outputs into the register
/// in groups of three ([`group_copy`]), V is `shape.v_gates` random gates on
/// the register and V* its inverse, and the recovery (the register's own
/// inverse of the copy) restores C(x) on the zero slice:
///   A(x, 0) = (D(C(x)) [junked by S2], C(x)).
/// V runs beside D's first half and V*, then the recovery, beside its
/// second half; V and D touch disjoint halves and interleave freely.
/// Reverse, A^-1(p, 0) leaves on the register the copy's bijection of
/// D^-1(p) (shifted by its value at zero). With S2 on, V must map zero to
/// zero (positive conjunctions) and so must the copy, or S2 would fire
/// under reverse evaluation; without slice blocks V is g57 gates like C, D.
///
/// The copy gates then float backward (into C, where their wires are
/// cold) as far as commutation allows; they cannot pass V.
pub fn v_gadget_sandwich(
    main: &CircuitSeq,
    d_gates: &[XGate],
    n: usize,
    s: usize,
    shape: SandwichShape,
    rng: &mut impl Rng,
) -> (CnotCircuit, SandwichLayout) {
    assert!(n >= 3, "the V gadget needs n >= 3");
    assert!(2 * n <= u16::MAX as usize, "too many wires");
    assert!(shape.v_gates > 0);
    assert!(
        main.gates.iter().flatten().all(|&wire| (wire as usize) < n),
        "source wire outside 0..n"
    );
    let zero_preserving = shape.slicing;
    let c_gates: Vec<XGate> = main.gates.iter().map(|&g| XGate::from_g57(g)).collect();
    let mut out = if shape.slicing {
        let s1 = sandwich_slice_gates(n, s, false, rng);
        random_interleave(c_gates, s1, rng)
    } else {
        c_gates
    };
    let groups = copy_groups(n, rng);
    let mut copy = Vec::new();
    let mut recovery = Vec::new();
    for group in &groups {
        let (c, r) = group_copy(group, n, zero_preserving, rng);
        copy.extend(c);
        recovery.extend(r);
    }
    let copy_start = out.len();
    let copy_count = copy.len();
    out.extend(copy);
    let v = sampled_xgates(n, shape.v_gates, shape.layer, n, zero_preserving, rng);
    let mut high_tail: Vec<XGate> = v.iter().rev().cloned().collect();
    let recovery_count = recovery.len();
    high_tail.extend(recovery);
    let low: Vec<XGate> = if shape.slicing {
        let s2 = sandwich_slice_gates(n, s, false, rng);
        random_interleave(d_gates.to_vec(), s2, rng)
    } else {
        d_gates.to_vec()
    };
    let half = low.len() / 2;
    out.extend(random_interleave(low[..half].to_vec(), v, rng));
    out.extend(random_interleave(low[half..].to_vec(), high_tail, rng));

    // Float the copy gates backward in a random order; positions of the
    // others shift as one passes them.
    let mut floaters: Vec<usize> = (copy_start..copy_start + copy_count).collect();
    floaters.shuffle(rng);
    for k in 0..floaters.len() {
        let p = floaters[k];
        let q = float_extremal(&mut out, p, true);
        floaters[k] = q;
        for (idx, r) in floaters.iter_mut().enumerate() {
            if idx != k && *r >= q && *r < p {
                *r += 1;
            }
        }
    }
    floaters.sort_unstable();
    let slice = if shape.slicing { s } else { 0 };
    let layout = SandwichLayout {
        c_gates: main.gates.len(),
        s1_gates: slice,
        copy_gates: copy_count,
        v_gates: shape.v_gates,
        recovery_gates: recovery_count,
        d_gates: d_gates.len(),
        s2_gates: slice,
        copy_groups: groups.len(),
        copy_positions: floaters,
    };
    (
        CnotCircuit {
            gates: out,
            num_wires: 2 * n,
        },
        layout,
    )
}

/// Longest chain of the dependency relation [`crate::stages::preprocessing::three_mask::interleave`]
/// orders by (a gate follows the latest writers of the wires it reads and
/// the latest readers of the wire it writes; XOR writes into one wire with
/// no read between them commute). gates / depth is the mean number of
/// gates free to run at a time (the circuit's dependency width).
pub fn dependency_depth(gates: &[XGate]) -> usize {
    let width = gates
        .iter()
        .map(|g| g.max_wire() as usize + 1)
        .max()
        .unwrap_or(0);
    // per wire: depth after the latest writer group, after the latest reader
    // group, and the depth floor a new group of the other kind starts from
    let mut writers_done = vec![0usize; width];
    let mut readers_done = vec![0usize; width];
    let mut last_was_writer = vec![false; width];
    let mut writer_floor = vec![0usize; width];
    let mut reader_floor = vec![0usize; width];
    let mut max_depth = 0;
    for g in gates {
        let t = g.target as usize;
        let mut d = 0usize;
        for &(c, _) in &g.ctrls {
            let c = c as usize;
            // a read follows the latest writer group
            d = d.max(if last_was_writer[c] {
                writers_done[c]
            } else {
                reader_floor[c]
            });
        }
        // a write follows the latest reader group
        d = d.max(if last_was_writer[t] {
            writer_floor[t]
        } else {
            readers_done[t]
        });
        let depth = d + 1;
        max_depth = max_depth.max(depth);
        for &(c, _) in &g.ctrls {
            let c = c as usize;
            if last_was_writer[c] {
                // a new reader group starts after the writer group
                reader_floor[c] = writers_done[c];
                readers_done[c] = 0;
                last_was_writer[c] = false;
            }
            readers_done[c] = readers_done[c].max(depth);
        }
        if !last_was_writer[t] {
            writer_floor[t] = readers_done[t];
            writers_done[t] = 0;
            last_was_writer[t] = true;
        }
        writers_done[t] = writers_done[t].max(depth);
    }
    max_depth
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
    sliced_sandwich_build_opt(main, d_gates, n, s, variant, true, rng)
}

/// [`sliced_sandwich_build`] with the slice blocks optional: `slicing`
/// false leaves out S1 and S2 (C ; N ; D) and draws nothing for them, so
/// `slicing` true keeps every draw, hence every old seed's circuit.
fn sliced_sandwich_build_opt(
    main: &CircuitSeq,
    d_gates: &[XGate],
    n: usize,
    s: usize,
    variant: SandwichVariant,
    slicing: bool,
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
    let mut out = if slicing {
        let s1 = sandwich_slice_gates(n, s, false, rng);
        random_interleave(c_gates, s1, rng)
    } else {
        c_gates
    };

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
    let s2 = if slicing {
        sandwich_slice_gates(n, s, balanced, rng)
    } else {
        Vec::new()
    };
    let d_placed: Vec<XGate> = if balanced {
        d_gates.iter().map(|g| shift_xgate_wires(g, n)).collect()
    } else {
        d_gates.to_vec()
    };
    if slicing {
        out.extend(random_interleave(d_placed, s2, rng));
    } else {
        out.extend(d_placed);
    }

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
#[path = "../../../tests/unit/stages/sandwich/construct.rs"]
mod tests;
