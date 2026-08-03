use crate::circuit::circuit::CircuitSeq;
use crate::postmix::xgate::XGate;
use crate::random::random_data::shoot_random_gate_with_rng;
use rand::{Rng, RngCore, prelude::SliceRandom};
use std::collections::VecDeque;

/// 6-gate homomorphic gadget for secret-shared g57.
/// Local wire map: 0=share_a, 3=share_c, 4=pad_c, 5=share_b, 6=pad_b.
const GADGET: [[u16; 3]; 6] = [
    [4, 5, 6],
    [0, 4, 6],
    [0, 5, 4],
    [4, 5, 6],
    [0, 6, 3],
    [0, 3, 5],
];

/// RG1: swap virtual values between two pairs.
/// Wire map: 0=share_i, 1=pad_i, 2=pad_j, 3=share_j.
const RG1: [[u16; 3]; 6] = [
    [1, 2, 3],
    [0, 3, 2],
    [3, 1, 0],
    [2, 0, 1],
    [0, 3, 2],
    [1, 2, 3],
];

/// RG2: re-pair two pairs (break the pairings) while keeping virtual values intact.
/// Wire map: 0=share_i, 1=pad_i, 2=pad_j, 3=share_j.
/// Under r57 this yields (degrees 2/3, found by search):
///   w0' = 1+w0+w2+w2w3
///   w1' = 1+w0+w1+w2+w3+w0w2+w0w3+w2w3
///   w2' = 1+w0+w3+w2w3
///   w3' = 1+w2+w3+w0w2+w0w3+w2w3
/// so w0'+w2' = w2+w3 = s_j  and  w1'+w3' = w0+w1 = s_i, i.e. virtual_i moves to
/// wires (pad_i, share_j) and virtual_j moves to wires (share_i, pad_j).
const RG2: [[u16; 3]; 6] = [
    [0, 3, 2],
    [1, 0, 2],
    [2, 0, 3],
    [2, 3, 0],
    [1, 3, 2],
    [3, 0, 2],
];

/// 11-gate W_i sequence for the r57 gate (a ^= pos | !neg) used in this codebase.
/// Local wires: 0=w0(i), 1=w1(n+i), 2=w2(p(i)), 3=w3(p(n+i)).
/// Effect: (w0,w1,w2,w3) -> (w2, w3, w0 XOR w1, w1).
/// All pins distinct; reversed sequence gives W_i^{-1} (each r57 gate is self-inverse).
/// Found by meet-in-the-middle search over r57 gates.
const W_I_GATES: [[u16; 3]; 11] = [
    [0, 3, 2],
    [3, 2, 1],
    [1, 3, 2],
    [2, 0, 1],
    [2, 1, 0],
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 3],
    [3, 0, 1],
    [3, 1, 0],
    [1, 3, 0],
];

/// Fixed 4-wire NOT gadget from the SAMF templates. Local wire 1 is flipped;
/// local wires 0, 2, and 3 are borrowed and restored.
const NOT_4W_GATES: [[u16; 3]; 7] = [
    [0, 2, 3],
    [1, 0, 2],
    [1, 0, 3],
    [0, 2, 3],
    [1, 2, 0],
    [1, 2, 3],
    [1, 3, 0],
];

pub const SLICE_ZERO_RANDOM_GATES_PER_WIRE: usize = 32;
pub const SLICE_ZERO_HARDCODED_DEFAULT_ROUNDS: usize = 1;

/// Secret-sharing state: pairs[v] = (share_wire, pad_wire) for virtual value v.
pub struct GadgetState {
    pub n: usize,
    pub pairs: Vec<(usize, usize)>,
}

#[derive(Clone, Debug)]
pub struct SliceZeroRandomCircuit {
    pub circuit: CircuitSeq,
    pub public_y: Vec<u64>,
    pub public_z: Vec<u64>,
}

/// Heterogeneous circuit emitted by the `sss --cnot` gadget path.
#[derive(Clone, Debug)]
pub struct CnotCircuit {
    pub gates: Vec<XGate>,
    pub num_wires: usize,
}

#[derive(Clone, Debug)]
pub struct CnotSliceZeroRandomCircuit {
    pub circuit: CnotCircuit,
    pub public_y: Vec<u64>,
    pub public_z: Vec<u64>,
}

/// What a physical wire currently holds, used by the bookends to track live
/// locations of every value as W_i gadgets relocate wire contents.
#[derive(Clone, Copy, PartialEq)]
enum Slot {
    Data(usize),   // raw data value v (left bookend, not yet shared)
    Aux(usize),    // raw aux value v
    Pair(usize),   // a share/pad of virtual value v
    Output(usize), // decoded output (right bookend), or sentinel
}

/// A value moved from wire `frm` to wire `to`; update whichever tracker owns it.
fn reloc(
    slot: Slot,
    frm: usize,
    to: usize,
    dloc: &mut [usize],
    aloc: &mut [usize],
    pairs: &mut [(usize, usize)],
) {
    match slot {
        Slot::Data(v) => dloc[v] = to,
        Slot::Aux(v) => aloc[v] = to,
        Slot::Pair(u) => {
            let (sw, pw) = pairs[u];
            pairs[u] = (
                if sw == frm { to } else { sw },
                if pw == frm { to } else { pw },
            );
        }
        Slot::Output(_) => {}
    }
}

/// Two distinct wires not in `exclude`, used as read-only scratch for transvections.
fn pick_two_helpers(total: usize, exclude: &[usize]) -> (usize, usize) {
    let mut it = (0..total).filter(|w| !exclude.contains(w));
    (it.next().unwrap(), it.next().unwrap())
}

fn pick_three_helpers(total: usize, exclude: &[usize]) -> (usize, usize, usize) {
    let mut it = (0..total).filter(|w| !exclude.contains(w));
    (it.next().unwrap(), it.next().unwrap(), it.next().unwrap())
}

/// Emit r57 gates computing `wire a ^= wire s`, leaving s, h1, h2 unchanged.
/// (Single transvection; needs two read-only helper wires under the r57 gate.)
fn emit_transvection(a: usize, s: usize, h1: usize, h2: usize, out: &mut Vec<[u16; 3]>) {
    let (a, s, h1, h2) = (a as u16, s as u16, h1 as u16, h2 as u16);
    out.push([s, h1, h2]);
    out.push([a, s, h1]);
    out.push([s, h1, h2]);
    out.push([a, h1, s]);
}

fn emit_not(wire: usize, total: usize, out: &mut Vec<[u16; 3]>) {
    let (h0, h1, h2) = pick_three_helpers(total, &[wire]);
    let map = [h0 as u16, wire as u16, h1 as u16, h2 as u16];
    for &[a, b, c] in &NOT_4W_GATES {
        out.push([map[a as usize], map[b as usize], map[c as usize]]);
    }
}

fn emit_w_i(w0: usize, w1: usize, w2: usize, w3: usize, out: &mut Vec<[u16; 3]>) {
    let map = [w0 as u16, w1 as u16, w2 as u16, w3 as u16];
    for &[a, b, c] in &W_I_GATES {
        out.push([map[a as usize], map[b as usize], map[c as usize]]);
    }
}

fn emit_w_i_inv(w0: usize, w1: usize, w2: usize, w3: usize, out: &mut Vec<[u16; 3]>) {
    let map = [w0 as u16, w1 as u16, w2 as u16, w3 as u16];
    for &[a, b, c] in W_I_GATES.iter().rev() {
        out.push([map[a as usize], map[b as usize], map[c as usize]]);
    }
}

/// Balanced random gates on aux wires (n..2n), controls from all 2n wires.
fn rand_z_gates(n: usize, m: usize, rng: &mut impl Rng) -> Vec<[u16; 3]> {
    let total = 2 * n;
    let mut gates = Vec::with_capacity(m);
    let mut round: Vec<usize> = (n..2 * n).collect();
    let mut pos = round.len();
    for _ in 0..m {
        if pos >= round.len() {
            round.shuffle(rng);
            pos = 0;
        }
        let active = round[pos] as u16;
        pos += 1;
        let ctrl1 = loop {
            let w = rng.random_range(0..total) as u16;
            if w != active {
                break w;
            }
        };
        let ctrl2 = loop {
            let w = rng.random_range(0..total) as u16;
            if w != active && w != ctrl1 {
                break w;
            }
        };
        gates.push([active, ctrl1, ctrl2]);
    }
    gates
}

pub fn emit_gadget(state: &GadgetState, gate: [u16; 3], out: &mut Vec<[u16; 3]>) {
    let a = gate[0] as usize;
    let b = gate[1] as usize;
    let c = gate[2] as usize;
    let map: [u16; 7] = [
        state.pairs[a].0 as u16,
        0,
        0,
        state.pairs[c].0 as u16,
        state.pairs[c].1 as u16,
        state.pairs[b].0 as u16,
        state.pairs[b].1 as u16,
    ];
    for &[ga, gb, gc] in &GADGET {
        out.push([map[ga as usize], map[gb as usize], map[gc as usize]]);
    }
}

pub fn emit_rg1(state: &mut GadgetState, i: usize, j: usize, out: &mut Vec<[u16; 3]>) {
    let map = [
        state.pairs[i].0 as u16,
        state.pairs[i].1 as u16,
        state.pairs[j].1 as u16,
        state.pairs[j].0 as u16,
    ];
    for &[a, b, c] in &RG1 {
        out.push([map[a as usize], map[b as usize], map[c as usize]]);
    }
    state.pairs.swap(i, j);
}

pub fn emit_rg2(state: &mut GadgetState, i: usize, j: usize, out: &mut Vec<[u16; 3]>) {
    let map = [
        state.pairs[i].0 as u16,
        state.pairs[i].1 as u16,
        state.pairs[j].1 as u16,
        state.pairs[j].0 as u16,
    ];
    for &[a, b, c] in &RG2 {
        out.push([map[a as usize], map[b as usize], map[c as usize]]);
    }
    // After RG2: virtual_i now lives on (pad_i, share_j); virtual_j on (share_i, pad_j).
    // (SG is symmetric in a pair's two shares, so .0/.1 order is free.)
    let new_i = (state.pairs[i].1, state.pairs[j].0);
    let new_j = (state.pairs[i].0, state.pairs[j].1);
    state.pairs[i] = new_i;
    state.pairs[j] = new_j;
}

pub fn emit_rg3(state: &GadgetState, i: usize, r1: usize, r2: usize, out: &mut Vec<[u16; 3]>) {
    out.push([state.pairs[i].0 as u16, r1 as u16, r2 as u16]);
    out.push([state.pairs[i].1 as u16, r1 as u16, r2 as u16]);
}

fn next_pair(queue: &mut VecDeque<(usize, usize)>, n: usize, rng: &mut impl Rng) -> (usize, usize) {
    if queue.is_empty() {
        let mut pairs: Vec<(usize, usize)> = (0..n)
            .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
            .collect();
        pairs.shuffle(rng);
        queue.extend(pairs);
    }
    queue.pop_front().unwrap()
}

fn next_single(queue: &mut VecDeque<usize>, n: usize, rng: &mut impl Rng) -> usize {
    if queue.is_empty() {
        let mut wires: Vec<usize> = (0..n).collect();
        wires.shuffle(rng);
        queue.extend(wires);
    }
    queue.pop_front().unwrap()
}

/// Gadgetize a circuit: add n aux wires (total 2n), secret-share all values via
/// Latin-square Z + matching permutation M_p, process gates as SG gadgets
/// interleaved with RG gadgets, then restore via M_p^{-1} + Z.
pub fn gadgetize(main: &CircuitSeq, n: usize, rg_freq: usize, rng: &mut impl Rng) -> CircuitSeq {
    // Start by randomizing the gate order of the input circuit (functionality-preserving:
    // shoot_random_gate only slides gates past non-colliding neighbors).
    let mut main = main.clone();
    let rounds = main.gates.len();
    shoot_random_gate_with_rng(&mut main, rounds, rng);
    let main = &main;

    let bookend_size = (2 * n * (n as f64).ln() as usize).max(64);
    let mut out: Vec<[u16; 3]> = Vec::new();

    // Left bookend: Z — randomize aux wires n..2n
    out.extend(rand_z_gates(n, bookend_size, rng));

    // Left bookend: M_p via live-tracking W_i. Each data value is secret-shared and the
    // shares placed at randomly chosen wires. We track the live location of every value
    // (data, aux, finished pair) so that the wire relocations done by one W_i never
    // corrupt a wire a later W_i needs — fixing the fixed-permutation interference hazard.
    let total = 2 * n;
    let mut dloc: Vec<usize> = (0..n).collect();
    let mut aloc: Vec<usize> = (n..2 * n).collect();
    let mut on: Vec<Slot> = (0..total)
        .map(|w| {
            if w < n {
                Slot::Data(w)
            } else {
                Slot::Aux(w - n)
            }
        })
        .collect();
    let mut pairs = vec![(0usize, 0usize); n];

    for v in 0..n {
        let d = dloc[v];
        let a = aloc[v];
        // Targets s (share), t (pad): any two distinct wires other than d, a.
        let s = loop {
            let w = rng.random_range(0..total);
            if w != d && w != a {
                break w;
            }
        };
        let t = loop {
            let w = rng.random_range(0..total);
            if w != d && w != a && w != s {
                break w;
            }
        };
        emit_w_i(d, a, s, t, &mut out);
        let moved_s = on[s];
        let moved_t = on[t];
        on[s] = Slot::Pair(v);
        on[t] = Slot::Pair(v);
        pairs[v] = (s, t);
        // W_i relocates old(s) -> d and old(t) -> a; keep all trackers consistent.
        reloc(moved_s, s, d, &mut dloc, &mut aloc, &mut pairs);
        reloc(moved_t, t, a, &mut dloc, &mut aloc, &mut pairs);
        on[d] = moved_s;
        on[a] = moved_t;
    }

    let mut state = GadgetState { n, pairs };
    let mut rg_pair_queue: VecDeque<(usize, usize)> = VecDeque::new();
    let mut rg3_queue: VecDeque<usize> = VecDeque::new();

    for (idx, &gate) in main.gates.iter().enumerate() {
        emit_gadget(&state, gate, &mut out);

        if (idx + 1) % rg_freq == 0 {
            match rng.random_range(0..3u32) {
                0 => {
                    let (i, j) = next_pair(&mut rg_pair_queue, n, rng);
                    emit_rg1(&mut state, i, j, &mut out);
                }
                1 => {
                    let (i, j) = next_pair(&mut rg_pair_queue, n, rng);
                    emit_rg2(&mut state, i, j, &mut out);
                }
                _ => {
                    let i = next_single(&mut rg3_queue, n, rng);
                    // Draw one carrier from each of two other logical values.
                    // This keeps the legacy G57 helper gate non-complete too.
                    let j = random_wire_except(n, &[i], rng);
                    let k = random_wire_except(n, &[i, j], rng);
                    let j_pair = state.pairs[j];
                    let k_pair = state.pairs[k];
                    let r1 = if rng.random_bool(0.5) {
                        j_pair.0
                    } else {
                        j_pair.1
                    };
                    let r2 = if rng.random_bool(0.5) {
                        k_pair.0
                    } else {
                        k_pair.1
                    };
                    emit_rg3(&state, i, r1, r2, &mut out);
                }
            }
        }
    }

    // Right bookend: live-tracking decode. Only wires 0..n must end correct (the upper
    // n are allowed to be random). For each virtual value v we place its decoded value
    // (share ^ pad) onto wire v, relocating any displaced value so nothing is lost.
    // Processing v in increasing order guarantees pair v's wires are never an already
    // decoded wire (< v), so finished outputs are never clobbered.
    for w in 0..total {
        on[w] = Slot::Output(usize::MAX);
    }
    for v in 0..n {
        on[state.pairs[v].0] = Slot::Pair(v);
        on[state.pairs[v].1] = Slot::Pair(v);
    }
    let mut finalized = vec![false; total];
    for v in 0..n {
        let (sw, pw) = state.pairs[v];
        if sw == v {
            // share already on wire v: wire v ^= wire pw  ->  v = share ^ pad
            let (h1, h2) = pick_two_helpers(total, &[v, pw]);
            emit_transvection(v, pw, h1, h2, &mut out);
        } else if pw == v {
            let (h1, h2) = pick_two_helpers(total, &[v, sw]);
            emit_transvection(v, sw, h1, h2, &mut out);
        } else {
            // W_i^{-1}(v, b, sw, pw): wire v <- share^pad; relocates old(v)->sw, old(b)->pw.
            let b = (0..total)
                .find(|&w| !finalized[w] && w != v && w != sw && w != pw)
                .unwrap();
            let moved_v = on[v];
            let moved_b = on[b];
            emit_w_i_inv(v, b, sw, pw, &mut out);
            reloc(moved_v, v, sw, &mut dloc, &mut aloc, &mut state.pairs);
            reloc(moved_b, b, pw, &mut dloc, &mut aloc, &mut state.pairs);
            on[sw] = moved_v;
            on[pw] = moved_b;
        }
        finalized[v] = true;
        on[v] = Slot::Output(v);
    }

    // Right bookend: Z — randomize aux wires
    out.extend(rand_z_gates(n, bookend_size, rng));

    CircuitSeq { gates: out }
}

/// Three-share state: x_i = pair_0 ^ pair_1 ^ free, while
/// y_{q[i]} = pair_0 ^ pair_1.
struct FeistalState {
    sharing: GadgetState,
    free: Vec<usize>,
    q: Vec<usize>,
}

#[derive(Clone, Copy)]
enum FeistalSlot {
    RawX(usize),
    RawY(usize),
    RawZ(usize),
    Pair0(usize),
    Pair1(usize),
    Free(usize),
}

fn reloc_feistal(
    slot: FeistalSlot,
    frm: usize,
    to: usize,
    xloc: &mut [usize],
    yloc: &mut [usize],
    zloc: &mut [usize],
    pairs: &mut [(usize, usize)],
    free: &mut [usize],
) {
    match slot {
        FeistalSlot::RawX(v) => xloc[v] = to,
        FeistalSlot::RawY(v) => yloc[v] = to,
        FeistalSlot::RawZ(v) => zloc[v] = to,
        FeistalSlot::Pair0(v) => {
            debug_assert_eq!(pairs[v].0, frm);
            pairs[v].0 = to;
        }
        FeistalSlot::Pair1(v) => {
            debug_assert_eq!(pairs[v].1, frm);
            pairs[v].1 = to;
        }
        FeistalSlot::Free(v) => {
            debug_assert_eq!(free[v], frm);
            free[v] = to;
        }
    }
}

fn random_permutation(n: usize, rng: &mut impl Rng) -> Vec<usize> {
    let mut q: Vec<usize> = (0..n).collect();
    q.shuffle(rng);
    q
}

fn random_circuit_with_rng(n: usize, m: usize, rng: &mut impl Rng) -> CircuitSeq {
    let mut gates = Vec::with_capacity(m);
    for _ in 0..m {
        let active = rng.random_range(0..n) as u16;
        let pos = loop {
            let w = rng.random_range(0..n) as u16;
            if w != active {
                break w;
            }
        };
        let neg = loop {
            let w = rng.random_range(0..n) as u16;
            if w != active && w != pos {
                break w;
            }
        };
        gates.push([active, pos, neg]);
    }
    CircuitSeq { gates }
}

fn random_invertible_matrix_rows(n: usize, rng: &mut impl Rng) -> Vec<Vec<u64>> {
    let words = n.div_ceil(64);
    let mut rows = vec![vec![0u64; words]; n];
    for i in 0..n {
        rows[i][i / 64] |= 1u64 << (i % 64);
    }

    let rounds = (2 * n).max(64);
    for _ in 0..rounds {
        let a = rng.random_range(0..n);
        let b = loop {
            let b = rng.random_range(0..n);
            if b != a {
                break b;
            }
        };
        if rng.random_range(0..4usize) == 0 {
            rows.swap(a, b);
        } else if a < b {
            let (left, right) = rows.split_at_mut(b);
            for (dst, &src) in left[a].iter_mut().zip(&right[0]) {
                *dst ^= src;
            }
        } else {
            let (left, right) = rows.split_at_mut(a);
            for (dst, &src) in right[0].iter_mut().zip(&left[b]) {
                *dst ^= src;
            }
        }
    }

    rows
}

fn matrix_bit(rows: &[Vec<u64>], row: usize, col: usize) -> bool {
    rows[row][col / 64] & (1u64 << (col % 64)) != 0
}

pub fn packed_bit(words: &[u64], bit: usize) -> bool {
    words[bit / 64] & (1u64 << (bit % 64)) != 0
}

fn set_packed_bit(words: &mut [u64], bit: usize) {
    words[bit / 64] |= 1u64 << (bit % 64);
}

fn random_public_slice(n: usize, rng: &mut impl Rng) -> (Vec<u64>, Vec<u64>) {
    let words = n.div_ceil(64);
    loop {
        let mut public_y = vec![0u64; words];
        let mut public_z = vec![0u64; words];
        let mut ones = 0usize;

        for bit in 0..n {
            if rng.random::<bool>() {
                set_packed_bit(&mut public_y, bit);
                ones += 1;
            }
            if rng.random::<bool>() {
                set_packed_bit(&mut public_z, bit);
                ones += 1;
            }
        }

        if ones > 0 && ones < 2 * n {
            return (public_y, public_z);
        }
    }
}

/// Build M on raw feistal wires:
///   M(x,y,z) = (x ^ A*(y OR z), y, z)
/// where A is a random invertible binary matrix. Thus the zero slice
/// (y,z)=(0,0) is fixed exactly, while every nonzero slice changes x.
pub fn slice_zero_preblock(n: usize, rng: &mut impl Rng) -> CircuitSeq {
    assert!(n >= 3, "slice_zero_preblock requires n >= 3");
    assert!(3 * n <= u16::MAX as usize, "too many wires");

    let total = 3 * n;
    let matrix = random_invertible_matrix_rows(n, rng);
    let mut out = Vec::new();

    for col in 0..n {
        let y = n + col;
        let z = 2 * n + col;
        emit_not(z, total, &mut out);
        for row in 0..n {
            if matrix_bit(&matrix, row, col) {
                out.push([row as u16, y as u16, z as u16]);
            }
        }
        emit_not(z, total, &mut out);
    }

    CircuitSeq { gates: out }
}

fn random_wire_except(total: usize, excluded: &[usize], rng: &mut impl Rng) -> usize {
    loop {
        let w = rng.random_range(0..total);
        if !excluded.contains(&w) {
            return w;
        }
    }
}

fn emit_and_update(
    target: usize,
    aux_factor: usize,
    other_factor: usize,
    total: usize,
    out: &mut Vec<[u16; 3]>,
) {
    debug_assert_ne!(target, aux_factor);
    debug_assert_ne!(target, other_factor);
    debug_assert_ne!(aux_factor, other_factor);

    out.push([target as u16, aux_factor as u16, other_factor as u16]);
    emit_not(target, total, out);
    let (h1, h2) = pick_two_helpers(total, &[target, aux_factor, other_factor]);
    emit_transvection(target, other_factor, h1, h2, out);
}

/// Build M on raw feistal wires with the zero slice hardcoded:
///   M(x,0,0) = (x,0,0).
/// Off the zero slice, each update applies target ^= aux * other.
pub fn slice_zero_hardcoded_preblock(n: usize, rounds: usize, rng: &mut impl Rng) -> CircuitSeq {
    assert!(n >= 3, "slice_zero_hardcoded_preblock requires n >= 3");
    assert!(3 * n <= u16::MAX as usize, "too many wires");

    let total = 3 * n;
    let mut out = Vec::new();
    let mut order: Vec<usize> = (0..n).collect();

    for _ in 0..rounds {
        order.shuffle(rng);
        for &i in &order {
            let target = i;
            let aux = if rng.random_bool(0.5) {
                n + rng.random_range(0..n)
            } else {
                2 * n + rng.random_range(0..n)
            };
            let other = random_wire_except(total, &[target, aux], rng);
            emit_and_update(target, aux, other, total, &mut out);
        }

        order.shuffle(rng);
        for &i in &order {
            let target = n + i;
            let aux = 2 * n + rng.random_range(0..n);
            let other = random_wire_except(total, &[target, aux], rng);
            emit_and_update(target, aux, other, total, &mut out);
        }

        order.shuffle(rng);
        for &i in &order {
            let target = 2 * n + i;
            let aux = n + rng.random_range(0..n);
            let other = random_wire_except(total, &[target, aux], rng);
            emit_and_update(target, aux, other, total, &mut out);
        }
    }

    CircuitSeq { gates: out }
}

/// Build M on raw feistal wires for a random public slice (Y,Z):
///   M(x,Y,Z) = (x,Y,Z)
/// and random r57 gates disturb x away from that public slice. Each gate uses
/// a public-0 bit as its positive control and a public-1 bit as its negative
/// control, so the gate condition is false exactly on the public slice.
pub fn slice_zero_random_preblock(
    n: usize,
    gate_count: usize,
    rng: &mut impl Rng,
) -> SliceZeroRandomCircuit {
    assert!(n >= 3, "slice_zero_random_preblock requires n >= 3");
    assert!(3 * n <= u16::MAX as usize, "too many wires");

    let (public_y, public_z) = random_public_slice(n, rng);
    let mut zero_controls = Vec::new();
    let mut one_controls = Vec::new();

    for bit in 0..n {
        if packed_bit(&public_y, bit) {
            one_controls.push(n + bit);
        } else {
            zero_controls.push(n + bit);
        }

        if packed_bit(&public_z, bit) {
            one_controls.push(2 * n + bit);
        } else {
            zero_controls.push(2 * n + bit);
        }
    }

    debug_assert!(!zero_controls.is_empty());
    debug_assert!(!one_controls.is_empty());

    let mut gates = Vec::with_capacity(gate_count);
    for _ in 0..gate_count {
        let active = rng.random_range(0..n) as u16;
        let pos = zero_controls[rng.random_range(0..zero_controls.len())] as u16;
        let neg = one_controls[rng.random_range(0..one_controls.len())] as u16;
        gates.push([active, pos, neg]);
    }

    SliceZeroRandomCircuit {
        circuit: CircuitSeq { gates },
        public_y,
        public_z,
    }
}

fn rand_feistal_z_gates(n: usize, m: usize, rng: &mut impl Rng) -> Vec<[u16; 3]> {
    let total = 3 * n;
    let mut gates = Vec::with_capacity(m);
    let mut round: Vec<usize> = (2 * n..total).collect();
    let mut pos = round.len();
    for _ in 0..m {
        if pos >= round.len() {
            round.shuffle(rng);
            pos = 0;
        }
        let active = round[pos] as u16;
        pos += 1;
        let c1 = loop {
            let w = rng.random_range(0..total) as u16;
            if w != active {
                break w;
            }
        };
        let c2 = loop {
            let w = rng.random_range(0..total) as u16;
            if w != active && w != c1 {
                break w;
            }
        };
        gates.push([active, c1, c2]);
    }
    gates
}

// 1-out-of-3 shared g57. All nine gates update the third x carrier;
// updating either paired carrier would also change the overlapping y value.
fn emit_sg3(state: &FeistalState, gate: [u16; 3], out: &mut Vec<[u16; 3]>) {
    let a = gate[0] as usize;
    let b = gate[1] as usize;
    let c = gate[2] as usize;
    let bs = [
        state.sharing.pairs[b].0,
        state.sharing.pairs[b].1,
        state.free[b],
    ];
    let cs = [
        state.sharing.pairs[c].0,
        state.sharing.pairs[c].1,
        state.free[c],
    ];
    for &bw in &bs {
        for &cw in &cs {
            out.push([state.free[a] as u16, bw as u16, cw as u16]);
        }
    }
}

fn emit_feistal_rg(
    state: &mut FeistalState,
    pq: &mut VecDeque<(usize, usize)>,
    sq: &mut VecDeque<usize>,
    rng: &mut impl Rng,
    out: &mut Vec<[u16; 3]>,
) {
    let n = state.sharing.n;
    match rng.random_range(0..3u32) {
        0 => {
            let (i, j) = next_pair(pq, n, rng);
            emit_rg1(&mut state.sharing, i, j, out);
        }
        1 => {
            let (i, j) = next_pair(pq, n, rng);
            emit_rg2(&mut state.sharing, i, j, out);
        }
        _ => {
            let i = next_single(sq, n, rng);
            // Draw helpers from two distinct other shared values. Besides
            // avoiding the refreshed value's third carrier, this prevents a
            // G57 helper gate from consuming multiple carriers of one value.
            let j = random_wire_except(n, &[i], rng);
            let k = random_wire_except(n, &[i, j], rng);
            let carrier = |logical: usize, choice: usize| match choice {
                0 => state.sharing.pairs[logical].0,
                1 => state.sharing.pairs[logical].1,
                _ => state.free[logical],
            };
            let r1 = carrier(j, rng.random_range(0..3));
            let r2 = carrier(k, rng.random_range(0..3));
            emit_rg3(&state.sharing, i, r1, r2, out);
        }
    }
}

fn emit_sg3_rg_block(
    c: &CircuitSeq,
    state: &mut FeistalState,
    rg_freq: usize,
    rng: &mut impl Rng,
    out: &mut Vec<[u16; 3]>,
) {
    let mut pq = VecDeque::new();
    let mut sq = VecDeque::new();
    for (idx, &gate) in c.gates.iter().enumerate() {
        emit_sg3(state, gate, out);
        if (idx + 1) % rg_freq == 0 {
            emit_feistal_rg(state, &mut pq, &mut sq, rng, out);
        }
    }
}

fn emit_feistal_n(state: &FeistalState, out: &mut Vec<[u16; 3]>) {
    let n = state.sharing.n;
    let mut host_of_y = vec![0usize; n];
    for (host, &y) in state.q.iter().enumerate() {
        host_of_y[y] = host;
    }
    for x in 0..n {
        let host = host_of_y[x];
        let (yc, other_y_carrier) = state.sharing.pairs[host];
        let hf = state.free[host];
        if host == x {
            // For (a,b,c), y=a+b and x=a+b+c. Map to (a,a+c,a+b),
            // so the pair becomes y+x=c while the triple remains x.
            for (dst, src) in [
                (other_y_carrier, yc),
                (hf, yc),
                (other_y_carrier, hf),
                (hf, other_y_carrier),
                (other_y_carrier, hf),
            ] {
                let (h1, h2) = pick_two_helpers(3 * n, &[dst, src]);
                emit_transvection(dst, src, h1, h2, out);
            }
            continue;
        }
        for source in [
            state.sharing.pairs[x].0,
            state.sharing.pairs[x].1,
            state.free[x],
        ] {
            let (h1, h2) = pick_two_helpers(3 * n, &[yc, hf, source]);
            emit_transvection(yc, source, h1, h2, out);
            emit_transvection(hf, source, h1, h2, out);
        }
    }
}

fn feistal_bit(row: &[u64], bit: usize) -> bool {
    row[bit / 64] & (1u64 << (bit % 64)) != 0
}
fn feistal_xor_row(rows: &mut [Vec<u64>], dst: usize, src: usize) {
    if dst < src {
        let (a, b) = rows.split_at_mut(src);
        for (d, &s) in a[dst].iter_mut().zip(&b[0]) {
            *d ^= s;
        }
    } else {
        let (a, b) = rows.split_at_mut(dst);
        for (d, &s) in b[0].iter_mut().zip(&a[src]) {
            *d ^= s;
        }
    }
}

fn emit_feistal_decode(state: &FeistalState, out: &mut Vec<[u16; 3]>) {
    let n = state.sharing.n;
    let total = 3 * n;
    let mut rows = vec![vec![0u64; total.div_ceil(64)]; total];
    for i in 0..n {
        let (p0, p1) = state.sharing.pairs[i];
        for bit in [p0, p1, state.free[i]] {
            rows[i][bit / 64] |= 1 << (bit % 64);
        }
        for bit in [p0, p1] {
            rows[n + state.q[i]][bit / 64] |= 1 << (bit % 64);
        }
        rows[2 * n + i][p0 / 64] |= 1 << (p0 % 64);
    }
    let mut ops = Vec::new();
    for col in 0..total {
        let pivot = (col..total)
            .find(|&r| feistal_bit(&rows[r], col))
            .expect("invertible decode");
        if pivot != col {
            feistal_xor_row(&mut rows, col, pivot);
            ops.push((col, pivot));
            feistal_xor_row(&mut rows, pivot, col);
            ops.push((pivot, col));
            feistal_xor_row(&mut rows, col, pivot);
            ops.push((col, pivot));
        }
        for row in 0..total {
            if row != col && feistal_bit(&rows[row], col) {
                feistal_xor_row(&mut rows, row, col);
                ops.push((row, col));
            }
        }
    }
    for &(dst, src) in ops.iter().rev() {
        let (h1, h2) = pick_two_helpers(total, &[dst, src]);
        emit_transvection(dst, src, h1, h2, out);
    }
}

/// Build the one-way layout on 3n wires. For input (x,y,z), the middle block
/// outputs y ^ C(x), the first block outputs D(C(x)) for a random same-size D,
/// and the final block is auxiliary output.
pub fn feistalize(main: &CircuitSeq, n: usize, rg_freq: usize, rng: &mut impl Rng) -> CircuitSeq {
    assert!(n >= 3, "feistalize requires n >= 3");
    assert!(3 * n <= u16::MAX as usize, "too many wires");
    assert!(rg_freq > 0, "rg_freq must be nonzero");
    assert!(
        main.gates.iter().flatten().all(|&w| (w as usize) < n),
        "input wire outside 0..n"
    );
    let total = 3 * n;
    let bookend = (((3 * n) as f64 * (n as f64).ln()).ceil() as usize).max(64);
    let mut out = rand_feistal_z_gates(n, bookend, rng);
    let q = random_permutation(n, rng);
    let mut xloc: Vec<usize> = (0..n).collect();
    let mut yloc: Vec<usize> = (n..2 * n).collect();
    let mut zloc: Vec<usize> = (2 * n..total).collect();
    let mut on: Vec<FeistalSlot> = (0..total)
        .map(|w| {
            if w < n {
                FeistalSlot::RawX(w)
            } else if w < 2 * n {
                FeistalSlot::RawY(w - n)
            } else {
                FeistalSlot::RawZ(w - 2 * n)
            }
        })
        .collect();
    let mut pairs = vec![(usize::MAX, usize::MAX); n];
    let mut free = vec![usize::MAX; n];
    for i in 0..n {
        let (x, y, z) = (xloc[i], yloc[q[i]], zloc[i]);
        let s = loop {
            let w = rng.random_range(0..total);
            if w != x && w != y && w != z {
                break w;
            }
        };
        let t = loop {
            let w = rng.random_range(0..total);
            if w != x && w != y && w != z && w != s {
                break w;
            }
        };
        let (moved_s, moved_t) = (on[s], on[t]);
        emit_w_i(x, z, s, t, &mut out);
        reloc_feistal(
            moved_s, s, x, &mut xloc, &mut yloc, &mut zloc, &mut pairs, &mut free,
        );
        reloc_feistal(
            moved_t, t, z, &mut xloc, &mut yloc, &mut zloc, &mut pairs, &mut free,
        );
        on[x] = moved_s;
        on[z] = moved_t;
        let (h1, h2) = pick_two_helpers(total, &[y, s]);
        emit_transvection(y, s, h1, h2, &mut out);
        let (h1, h2) = pick_two_helpers(total, &[t, y]);
        emit_transvection(t, y, h1, h2, &mut out);
        pairs[i] = (s, y);
        free[i] = t;
        on[s] = FeistalSlot::Pair0(i);
        on[y] = FeistalSlot::Pair1(i);
        on[t] = FeistalSlot::Free(i);
    }
    let mut state = FeistalState {
        sharing: GadgetState { n, pairs },
        free,
        q,
    };
    let mut c = main.clone();
    let rounds = c.gates.len();
    shoot_random_gate_with_rng(&mut c, rounds, rng);
    emit_sg3_rg_block(&c, &mut state, rg_freq, rng, &mut out);
    emit_feistal_n(&state, &mut out);
    let d = random_circuit_with_rng(n, main.gates.len(), rng);
    emit_sg3_rg_block(&d, &mut state, rg_freq, rng, &mut out);
    emit_feistal_decode(&state, &mut out);
    out.extend(rand_feistal_z_gates(n, bookend, rng));
    CircuitSeq { gates: out }
}

pub fn feistalize_with_slice_zero(
    main: &CircuitSeq,
    n: usize,
    rg_freq: usize,
    rng: &mut impl Rng,
) -> CircuitSeq {
    let mut gates = slice_zero_preblock(n, rng).gates;
    gates.extend(feistalize(main, n, rg_freq, rng).gates);
    CircuitSeq { gates }
}

pub fn feistalize_with_slice_zero_hardcoded(
    main: &CircuitSeq,
    n: usize,
    rg_freq: usize,
    rounds: usize,
    rng: &mut impl Rng,
) -> CircuitSeq {
    let mut gates = slice_zero_hardcoded_preblock(n, rounds, rng).gates;
    gates.extend(feistalize(main, n, rg_freq, rng).gates);
    CircuitSeq { gates }
}

pub fn feistalize_with_slice_zero_random(
    main: &CircuitSeq,
    n: usize,
    rg_freq: usize,
    gate_count: usize,
    rng: &mut impl Rng,
) -> SliceZeroRandomCircuit {
    let mut preblock = slice_zero_random_preblock(n, gate_count, rng);
    preblock
        .circuit
        .gates
        .extend(feistalize(main, n, rg_freq, rng).gates);
    preblock
}

// ---- Heterogeneous CNOT/fragment gadget path --------------------------------------------

fn emit_transvection_cnot(target: usize, source: usize, out: &mut Vec<XGate>) {
    out.push(XGate::cnot(target as u16, source as u16));
}

/// Masking-safe homomorphic CNOT for a two-share value. If
/// `a = a0 XOR a1` and `b = b0 XOR b1`, update the shares component-wise.
/// Each physical CNOT consumes one carrier from each logical value.
pub fn homomorphic_cnot2(target: (u16, u16), control: (u16, u16)) -> Vec<XGate> {
    assert!(
        target.0 != target.1
            && control.0 != control.1
            && ![control.0, control.1].contains(&target.0)
            && ![control.0, control.1].contains(&target.1),
        "homomorphic CNOT values must use disjoint carriers"
    );
    vec![
        XGate::cnot(target.0, control.0),
        XGate::cnot(target.1, control.1),
    ]
}

/// Masking-safe homomorphic CNOT for a three-share value. Only the free
/// target carrier is updated; the paired carriers (and therefore the
/// overlapping Feistel y value) remain untouched.
pub fn homomorphic_cnot3(target: (u16, u16, u16), control: (u16, u16, u16)) -> Vec<XGate> {
    assert!(
        target.0 != target.1
            && target.0 != target.2
            && target.1 != target.2
            && control.0 != control.1
            && control.0 != control.2
            && control.1 != control.2
            && ![control.0, control.1, control.2].contains(&target.0)
            && ![control.0, control.1, control.2].contains(&target.1)
            && ![control.0, control.1, control.2].contains(&target.2),
        "homomorphic CNOT values must use disjoint carriers"
    );
    vec![
        XGate::cnot(target.2, control.0),
        XGate::cnot(target.2, control.1),
        XGate::cnot(target.2, control.2),
    ]
}

/// Seven-CNOT realization of
/// `(q0,q1,q2,q3) -> (q2,q3,q0 XOR q1,q1)`.
fn emit_w_i_cnot(q0: usize, q1: usize, q2: usize, q3: usize, out: &mut Vec<XGate>) {
    for (target, control) in [
        (q0, q1),
        (q0, q2),
        (q1, q3),
        (q2, q0),
        (q0, q2),
        (q3, q1),
        (q1, q3),
    ] {
        emit_transvection_cnot(target, control, out);
    }
}

fn emit_w_i_inv_cnot(q0: usize, q1: usize, q2: usize, q3: usize, out: &mut Vec<XGate>) {
    let mut forward = Vec::with_capacity(7);
    emit_w_i_cnot(q0, q1, q2, q3, &mut forward);
    forward.reverse();
    out.extend(forward);
}

/// Four-cube ESOP for a G57 over two-share controls. For `B=b0+b1` and
/// `C=c0+c1`, these four disjoint/mixed-polarity fragments XOR to
/// `B OR !C`. Only `target` is written, so its untouched mate remains a mask
/// at every physical-gate prefix.
fn emit_shared_g57_frag2(
    target: usize,
    b0: usize,
    b1: usize,
    c0: usize,
    c1: usize,
    out: &mut Vec<XGate>,
) {
    for controls in [
        [(c1 as u16, false), (b0 as u16, true)],
        [(c1 as u16, true), (b1 as u16, false)],
        [(c0 as u16, true), (b1 as u16, true)],
        [(c0 as u16, false), (b0 as u16, false)],
    ] {
        out.push(XGate::conj(target as u16, controls).expect("distinct SG carriers"));
    }
}

#[cfg(test)]
fn emit_gadget_x(state: &GadgetState, gate: [u16; 3], out: &mut Vec<XGate>) {
    let [a, b, c] = gate.map(|wire| wire as usize);
    let (b0, b1) = state.pairs[b];
    let (c0, c1) = state.pairs[c];
    emit_shared_g57_frag2(state.pairs[a].0, b0, b1, c0, c1, out);
}

/// Homomorphically apply one heterogeneous logical gate to a two-share
/// representation.  G57 and CNOT gates retain their smaller dedicated
/// gadgets; the remaining conjunction fragments are expanded over the two
/// carriers of each logical control and update only one target carrier.
fn emit_shared_fragment2(state: &GadgetState, gate: &XGate, out: &mut Vec<XGate>) {
    let logical_target = gate.target as usize;
    debug_assert!(logical_target < state.n);

    // Preserve the existing four-fragment G57 gadget.  XGate stores a G57's
    // positive-OR input as a negative conjunction literal and its negative-OR
    // input as a positive conjunction literal.
    if gate.comp && gate.ctrls.len() == 2 {
        let negative = gate.ctrls.iter().find(|&&(_, polarity)| !polarity);
        let positive = gate.ctrls.iter().find(|&&(_, polarity)| polarity);
        if let (Some(&(b, false)), Some(&(c, true))) = (negative, positive) {
            let (b0, b1) = state.pairs[b as usize];
            let (c0, c1) = state.pairs[c as usize];
            emit_shared_g57_frag2(state.pairs[logical_target].0, b0, b1, c0, c1, out);
            return;
        }
    }

    if !gate.comp && gate.ctrls.len() == 1 && gate.ctrls[0].1 {
        let logical_control = gate.ctrls[0].0 as usize;
        out.extend(homomorphic_cnot2(
            (
                state.pairs[logical_target].0 as u16,
                state.pairs[logical_target].1 as u16,
            ),
            (
                state.pairs[logical_control].0 as u16,
                state.pairs[logical_control].1 as u16,
            ),
        ));
        return;
    }

    // Expand each logical literal in ANF: b = b0+b1 and !b = 1+b0+b1.
    // Duplicate physical monomials cancel over GF(2).
    let mut terms: Vec<Vec<(u16, bool)>> = vec![Vec::new()];
    for &(logical_control, positive) in &gate.ctrls {
        let logical_control = logical_control as usize;
        debug_assert!(logical_control < state.n);
        debug_assert_ne!(logical_control, logical_target);
        let carriers = state.pairs[logical_control];
        let previous = std::mem::take(&mut terms);
        for term in previous {
            if !positive {
                toggle_anf_term(&mut terms, term.clone());
            }
            for carrier in [carriers.0, carriers.1] {
                let mut next = term.clone();
                next.push((carrier as u16, true));
                next.sort_unstable();
                toggle_anf_term(&mut terms, next);
            }
        }
    }
    if gate.comp {
        toggle_anf_term(&mut terms, Vec::new());
    }
    terms.sort_unstable();
    for term in terms {
        if let Some(fragment) = XGate::conj(state.pairs[logical_target].0 as u16, term) {
            out.push(fragment);
        }
    }
}

fn emit_rg1_x(state: &mut GadgetState, i: usize, j: usize, out: &mut Vec<XGate>) {
    // Six-CNOT RG1. Unlike the legacy nonlinear network, no gate consumes both
    // carriers of either logical value. Exhaustive bounded search (regressed
    // below) finds no five-CNOT realization with this non-completeness property.
    let map = [
        state.pairs[i].0,
        state.pairs[i].1,
        state.pairs[j].1,
        state.pairs[j].0,
    ];
    for (target, control) in [(0, 2), (1, 3), (2, 0), (0, 2), (3, 1), (0, 3)] {
        emit_transvection_cnot(map[target], map[control], out);
    }
    state.pairs.swap(i, j);
}

fn emit_rg2_x(state: &mut GadgetState, i: usize, j: usize, out: &mut Vec<XGate>) {
    // Local map: 0=i0, 1=i1, 2=j1, 3=j0. Swapping i0 and j0 with three CNOTs
    // realizes the required re-pairing. The swapped carriers belong to
    // different logical values, so this is unlike an unsafe swap of the two
    // complementary carriers of one value: no gate sees a complete sharing.
    let map = [
        state.pairs[i].0,
        state.pairs[i].1,
        state.pairs[j].1,
        state.pairs[j].0,
    ];
    for (target, control) in [(0, 3), (3, 0), (0, 3)] {
        emit_transvection_cnot(map[target], map[control], out);
    }
    let new_i = (state.pairs[i].1, state.pairs[j].0);
    let new_j = (state.pairs[i].0, state.pairs[j].1);
    state.pairs[i] = new_i;
    state.pairs[j] = new_j;
}

fn emit_rg3_x(state: &GadgetState, i: usize, random_carrier: usize, out: &mut Vec<XGate>) {
    // Refresh both carriers with one carrier from a different shared value.
    // This keeps the logical XOR unchanged and never brings both carriers of
    // either value into one physical CNOT.
    emit_transvection_cnot(state.pairs[i].0, random_carrier, out);
    emit_transvection_cnot(state.pairs[i].1, random_carrier, out);
}

fn rand_z_xgates(n: usize, m: usize, rng: &mut impl Rng) -> Vec<XGate> {
    rand_z_gates(n, m, rng)
        .into_iter()
        .map(XGate::from_g57)
        .collect()
}

// ---- Nonlinear product-share encoding -----------------------------------------------

/// Tuning for the product-share encoding used by nonlinear gadgetization.
///
/// A logical value is decoded as one carrier plus permanent
/// multiplicative terms over frozen band values whose physical locations may
/// roll. The only production plan is `[2,2,2,3]`, using a single carrier and
/// the Gray-code fold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ProdConfig {
    pub(crate) k: usize,
    pub(crate) deg: usize,
    pub(crate) k_hi: usize,
    pub(crate) deg_hi: usize,
    /// Extra source-band wires (0 selects the automatic size).
    pub(crate) band: usize,
    /// Re-source moves per gap between logical source gates.
    pub(crate) rsrc: usize,
    /// Maximum emitted control width (0 keeps the native wide fragments).
    pub(crate) max_width: usize,
    /// Cascaded nonlinear products used to fill each band wire.
    pub(crate) fill_nl: usize,
    /// Band-variable relocations per gap between logical source gates.
    pub(crate) roll: usize,
    /// Distributed sourcing (0 = the dedicated band; 1 = no band at all).
    ///
    /// DECLARED BUT INERT in this tree: the reference implementation's
    /// band-free sourcing (`refs`/`owner`/write-barrier machinery) was not
    /// ported. Kept so the production preset can be transcribed verbatim and so
    /// a later port has the knob to fill in.
    src_dist: usize,
    /// Lookahead horizon for distributed sourcing, in source-gate positions
    /// (0 = auto, n/2). DECLARED BUT INERT: see `src_dist`.
    src_horizon: usize,
    /// Restrict distributed sources to the value range `[src_lo, src_hi)`
    /// (`src_hi = 0` means "up to n"). DECLARED BUT INERT: see `src_dist`.
    src_lo: usize,
    src_hi: usize,
    /// Realize DB-eligible fold fragments in the g57/CNOT vocabulary (0 = off).
    ///
    /// Clearing the width cap is necessary but not sufficient for the frozen
    /// store: a comp=0 width-2 conjunction is not in the X-free g57 span, so a
    /// narrow fold fragment can still be undigestible. This re-realizes exactly
    /// the fragments that already pass the cap, at 1-2 gates instead of 1.
    ///
    /// DECLARED BUT INERT on the ODOMETER path in this tree — turning it on
    /// there would change the RNG stream of every already-shipped preset. The
    /// Gray fold emits in the g57/CNOT vocabulary unconditionally (every
    /// emission goes through `spellings_at`), so `gray_fold = 1` already gets
    /// what this buys.
    g57_narrow: usize,
    /// Ladder fold fragments of width in (2, ladder_cap] down to <= 2 controls
    /// with the borrowed-carrier double sweep; 0 disables. DECLARED BUT INERT:
    /// nh's laddering is selected by `max_width` instead. The production preset
    /// sets it to 0, so nothing is lost by not honoring it.
    ladder_cap: usize,
    /// Percent of values given one EXTRA high-degree mask term, so a CG block's
    /// fragment count stops being the fixed (1+k_total)^arity. DECLARED BUT
    /// INERT: `plan_extra` and the jitter block in `inject_all` were not ported.
    cg_jitter: usize,
    /// Draw each emission's spelling from the equivalent menu at this level,
    /// forcing two copies of one function to differ (0 = one fixed spelling).
    ///
    /// HONORED, by the Gray fold's emissions (`spellings_at`). Over the
    /// generators {cnot(h,x), cnot(h,y), g57(h;x,y), g57(h;y,x)} there is one
    /// relation, so every reachable function has exactly two subset spellings:
    /// level 1 takes only the equal-size ones (free), level 2 takes all of them
    /// (+2 gates on a mixed-polarity emission).
    rung_menu: usize,
    /// Retire-and-refill epochs: inter-SG gaps between two events (0 = off).
    /// DECLARED BUT INERT: `retire_refill`/`inject_avoiding_var` were not
    /// ported.
    epoch: usize,
    /// Percentage of refill sources drawn from CARRIERS rather than from other
    /// band wires. DECLARED BUT INERT: see `epoch`.
    refill_data: usize,
    /// Reserved pivot block in the nonlinear band fill (0 = legacy draw).
    /// DECLARED BUT INERT: the reserved-pivot draw was not ported. The
    /// production preset sets it to 0 (band = n leaves the pivot block no
    /// room), so nothing is lost by not honoring it.
    fill_pivots: usize,
    /// Single-carrier decode: one linear carrier plus the four nonlinear
    /// mask terms. Production always enables this.
    single: usize,
    /// Gray-code fold: gather each operand's mask sum onto a dirty accumulator
    /// once and read it back four times, instead of expanding the cartesian
    /// product into fragments of width up to `arity * max_deg` (0 = off).
    ///
    /// Every emitted gate is then at most two controls WITHOUT laddering
    /// anything: the fold stops producing the width-3..6 material the frozen
    /// store cannot digest (0.41% hit rate at width 3, absent above), at ~3x the
    /// block's fragment count rather than full narrow mode's ~6.2x, because the
    /// mask products are derived once per block instead of once per fragment.
    /// Blocks it cannot amortize (arity != 2, an operand with no mask terms)
    /// fall back to the odometer.
    ///
    /// See [`ProdLedger::fold_cg_gray`].
    gray_fold: usize,
}

impl ProdConfig {
    const fn off() -> ProdConfig {
        ProdConfig {
            k: 0,
            deg: 2,
            k_hi: 0,
            deg_hi: 3,
            band: 0,
            rsrc: 1,
            max_width: 0,
            fill_nl: 0,
            roll: 0,
            src_dist: 0,
            src_horizon: 0,
            src_lo: 0,
            src_hi: 0,
            g57_narrow: 0,
            ladder_cap: 0,
            cg_jitter: 0,
            rung_menu: 0,
            epoch: 0,
            refill_data: 0,
            fill_pivots: 0,
            single: 0,
            gray_fold: 0,
        }
    }

    /// The sole supported nonlinear production construction: three degree-2
    /// mask terms and one degree-3 term, a single carrier, nonlinear band fill,
    /// rolling, and the Gray-code fold.
    pub(crate) const fn production() -> ProdConfig {
        ProdConfig {
            k: 3,
            deg: 2,
            k_hi: 1,
            deg_hi: 3,
            band: 0,
            rsrc: 1,
            max_width: 0,
            fill_nl: 2,
            roll: 1,
            src_dist: 0,
            src_horizon: 0,
            src_lo: 0,
            src_hi: 0,
            g57_narrow: 0,
            ladder_cap: 0,
            cg_jitter: 0,
            rung_menu: 1,
            epoch: 0,
            refill_data: 0,
            fill_pivots: 0,
            single: 1,
            gray_fold: 1,
        }
    }

    fn enabled(&self) -> bool {
        self.k > 0 || self.k_hi > 0
    }

    /// Single-carrier decode: one linear term per value instead of two.
    fn single_carrier(&self) -> bool {
        self.enabled() && self.single > 0
    }

    /// The Gray-code fold handles the arity-2 CG blocks (see
    /// [`ProdLedger::fold_cg_gray`]); everything else falls back to the
    /// odometer.
    fn gray(&self) -> bool {
        self.enabled() && self.gray_fold > 0
    }

    fn narrow(&self) -> bool {
        self.enabled() && self.max_width >= 2
    }

    fn k_total(&self) -> usize {
        self.k + self.k_hi
    }

    fn max_deg(&self) -> usize {
        let mut degree = 2;
        if self.k > 0 {
            degree = degree.max(self.deg);
        }
        if self.k_hi > 0 {
            degree = degree.max(self.deg_hi);
        }
        degree
    }

    fn band_size(&self, n: usize) -> usize {
        if !self.enabled() {
            return 0;
        }
        if self.band > 0 {
            return self.band;
        }
        // Production uses a 1:1 carrier/band split at real workloads. Tiny
        // fixtures still need enough distinct variables for the degree-3 term
        // and re-source churn; this is the reference production floor.
        n.max(6).max(self.max_deg() + 3)
    }
}

/// One permanent multiplicative mask term over distinct band variables.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ProdSlot {
    /// `(band-variable id, a)` represents the factor `variable XOR a`.
    // Production degrees are 2 and 3, so the complete tuple stays inline.
    factors: smallvec::SmallVec<[(u16, bool); 4]>,
}

impl ProdSlot {
    fn lits(&self, locations: &[u16]) -> smallvec::SmallVec<[(u16, bool); 4]> {
        self.factors
            .iter()
            .map(|&(variable, a)| (locations[variable as usize], !a))
            .collect()
    }
}

/// Emit `target ^= AND(lits)` for one or two literals using CNOT/g57
/// vocabulary. The return value is the extra constant left by that
/// realization; callers absorb it into their ledger (or free band junk).
fn emit_g57_form(
    target: u16,
    lits: &[(u16, bool)],
    rng: &mut impl Rng,
    out: &mut Vec<XGate>,
) -> bool {
    match *lits {
        [(wire, true)] => {
            out.push(XGate::cnot(target, wire));
            false
        }
        [(wire, false)] => {
            out.push(XGate::cnot(target, wire));
            true
        }
        [a, b] => {
            let ((xw, xp), (yw, yp)) = if rng.random_bool(0.5) { (a, b) } else { (b, a) };
            match (xp, yp) {
                (false, true) => {
                    out.push(XGate::from_g57([target, xw, yw]));
                    true
                }
                (true, false) => {
                    out.push(XGate::from_g57([target, yw, xw]));
                    true
                }
                (true, true) => {
                    out.push(XGate::from_g57([target, xw, yw]));
                    out.push(XGate::cnot(target, yw));
                    true
                }
                (false, false) => {
                    out.push(XGate::from_g57([target, xw, yw]));
                    out.push(XGate::cnot(target, xw));
                    false
                }
            }
        }
        _ => unreachable!("emit_g57_form takes one or two literals"),
    }
}

// ---- Gray-code fold: spelling menu and dirty-accumulator gathers ------------

/// One spelling is at most three gates and every menu has at most four entries.
/// Keep both levels inline: a Gray block builds dozens of these tiny menus.
type GateSpelling = smallvec::SmallVec<[XGate; 3]>;
type SpellingMenu = smallvec::SmallVec<[GateSpelling; 4]>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct SpellingKey {
    target: u16,
    len: u8,
    lits: [(u16, bool); 2],
}

impl SpellingKey {
    fn new(target: u16, lits: &[(u16, bool)]) -> Self {
        debug_assert!((1..=2).contains(&lits.len()));
        let mut key = Self {
            target,
            len: lits.len() as u8,
            lits: [(0, false); 2],
        };
        key.lits[..lits.len()].copy_from_slice(lits);
        key
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct SpellingUse {
    used: u8,
    last: Option<u8>,
}

/// Spellings already used for each (target, literal list) within one CG block.
///
/// The Gray fold emits the SAME function several times by construction -- four
/// times for `A_u * A_z` and for each accumulator gate across the gather and the
/// strip, twice for each `L x A` pair -- so avoiding only the IMMEDIATELY
/// PRECEDING spelling is not enough; the emissions are interleaved with other
/// gates and the repeats land far apart. Keeping the used set per block lets a
/// same-polarity function, which has exactly four equal-size spellings, use a
/// different one every time.
type SpellingLog = std::collections::HashMap<SpellingKey, SpellingUse>;

/// Pick a spelling not yet used for this (target, literals) in this block,
/// falling back to a fresh uniform draw once every spelling has been used.
fn pick_spelling(
    target: u16,
    lits: &[(u16, bool)],
    len: usize,
    seen: &mut SpellingLog,
    rng: &mut impl Rng,
) -> usize {
    if len < 2 {
        return 0;
    }
    debug_assert!(len <= u8::BITS as usize);
    let used = seen.entry(SpellingKey::new(target, lits)).or_default();
    let fresh_count = (0..len)
        .filter(|&index| used.used & (1u8 << index) == 0)
        .count();
    let pick = if fresh_count == 0 {
        // Every spelling is spent; restart the cycle rather than repeat the
        // most recent one.
        let last = used.last.expect("a spent spelling cycle has a last entry") as usize;
        let draw = rng.random_range(0..len - 1);
        used.used = 0;
        if draw < last { draw } else { draw + 1 }
    } else {
        let draw = rng.random_range(0..fresh_count);
        (0..len)
            .filter(|&index| used.used & (1u8 << index) == 0)
            .nth(draw)
            .expect("fresh spelling count and selection agree")
    };
    used.used |= 1u8 << pick;
    used.last = Some(pick as u8);
    pick
}

/// Every ordered gate sequence that contributes the SAME function to `h`, so a
/// rung's two emissions can be spelled differently while still cancelling.
///
/// Why a menu is needed at all: the double sweep emits its rung twice, and
/// cloning one prebuilt gate plants a pair of BYTE-IDENTICAL gates. On the
/// shipped n=128 build 143,100 of 184,898 comp=0 width-2 instances (77.4%) sat
/// in duplicate groups and 50.4% of the whole circuit sat in exact pairs, so
/// `sort | uniq -c` -- no execution, no algebra -- located every ladder, its
/// borrowed wire and two of its three literals. Routing the rung through
/// `emit_g57_form`'s coin does NOT fix it (63.2% -> 69.6%), and the enumeration
/// below says why: for a MIXED-polarity pair that coin emits the same gate
/// either way, so those pairs always matched.
///
/// The menu is exact and complete. Over generators {cnot(h,x), cnot(h,y),
/// g57(h;x,y), g57(h;y,x)} the reachable functions span dimension 3 with the
/// single relation g1^g2^g3^g4 = 0, so every achievable function has exactly
/// TWO subset spellings; and since each gate is `h ^= f(x,y)` and none READS h,
/// they all commute, so every ordering of a subset is equally valid. Same
/// polarity gives two 2-gate spellings -- differing is FREE. Mixed polarity
/// gives one 1-gate and one 3-gate spelling, so differing costs +2 there.
///
/// All spellings of one function share the same residual constant by
/// construction, which is what keeps the borrow restored when the two
/// emissions differ.
fn gate_spelling<const N: usize>(gates: [XGate; N]) -> GateSpelling {
    gates.into_iter().collect()
}

fn spelling_menu<const N: usize>(spellings: [GateSpelling; N]) -> SpellingMenu {
    spellings.into_iter().collect()
}

#[inline]
fn rung_residual(lits: &[(u16, bool)]) -> bool {
    match *lits {
        [(_, positive)] => !positive,
        [(_, xp), (_, yp)] => xp || yp,
        _ => unreachable!("rung residual takes one or two literals"),
    }
}

fn rung_spellings(h: u16, lits: &[(u16, bool)]) -> (SpellingMenu, bool) {
    let g57 = |p: u16, q: u16| XGate::from_g57([h, p, q]);
    match *lits {
        [(w, p)] => (spelling_menu([gate_spelling([XGate::cnot(h, w)])]), !p),
        [(xw, xp), (yw, yp)] => {
            let cx = || XGate::cnot(h, xw);
            let cy = || XGate::cnot(h, yw);
            let menu = match (xp, yp) {
                // ~x&y : short g57(h;x,y); long cnot_x + cnot_y + g57(h;y,x)
                (false, true) => spelling_menu([
                    gate_spelling([g57(xw, yw)]),
                    gate_spelling([cx(), cy(), g57(yw, xw)]),
                    gate_spelling([cy(), g57(yw, xw), cx()]),
                    gate_spelling([g57(yw, xw), cx(), cy()]),
                ]),
                // x&~y : the same with the roles of x and y exchanged
                (true, false) => spelling_menu([
                    gate_spelling([g57(yw, xw)]),
                    gate_spelling([cy(), cx(), g57(xw, yw)]),
                    gate_spelling([cx(), g57(xw, yw), cy()]),
                    gate_spelling([g57(xw, yw), cy(), cx()]),
                ]),
                // x&y and ~x&~y : two 2-gate spellings, both orderings of each
                (true, true) => spelling_menu([
                    gate_spelling([g57(xw, yw), cy()]),
                    gate_spelling([cy(), g57(xw, yw)]),
                    gate_spelling([g57(yw, xw), cx()]),
                    gate_spelling([cx(), g57(yw, xw)]),
                ]),
                (false, false) => spelling_menu([
                    gate_spelling([g57(xw, yw), cx()]),
                    gate_spelling([cx(), g57(xw, yw)]),
                    gate_spelling([g57(yw, xw), cy()]),
                    gate_spelling([cy(), g57(yw, xw)]),
                ]),
            };
            // Residual constant: only the both-negative pattern realizes its
            // conjunction outright; the other three carry a 1.
            (menu, rung_residual(lits))
        }
        _ => unreachable!("rung_spellings takes one or two literals"),
    }
}

/// The spelling menu at a given variability LEVEL.
///
/// Over the generators {cnot(h,x), cnot(h,y), g57(h;x,y), g57(h;y,x)} there is
/// exactly one relation, g1^g2^g3^g4 = 0, so every reachable function has
/// exactly two subset spellings -- a set and its complement. That makes the
/// sizes 2 and 2 for a SAME-polarity conjunction (the function is a 2-subset)
/// but 1 and 3 for a MIXED-polarity one (the function IS a generator). So
/// varying the spelling is free on same-polarity emissions and costs +2 gates
/// on mixed ones, and the level chooses which of those to buy:
///
///   0  the single canonical spelling -- no variability, no cost.
///   1  only the equal-size spellings: four for same polarity, one for mixed.
///      Diversity exactly where it is free. Measured n=16 at ladder_cap 3:
///      identical gate count to level 0, width-2 duplicate groups 60.7% -> 52.6%.
///   2  every spelling, longer ones included. Reaches 41.0% but at +18.8%
///      gates, because each mixed-polarity emission then pays 1+3 instead of
///      1+1 to differ.
///
/// All spellings of one function share the same residual constant by
/// construction, which is what keeps the borrow restored when two emissions of
/// a rung differ.
fn spellings_at(h: u16, lits: &[(u16, bool)], level: usize) -> (SpellingMenu, bool) {
    let (full, konst) = rung_spellings(h, lits);
    if level >= 2 {
        return (full, konst);
    }
    let min_len = full.iter().map(|m| m.len()).min().unwrap_or(0);
    let mut free: SpellingMenu = full.into_iter().filter(|m| m.len() == min_len).collect();
    if level == 0 {
        free.truncate(1);
    }
    (free, konst)
}

/// Which literal of a degree-3 atom to pair with the helper. A rung of two
/// NEGATIVE literals is the one polarity pattern that realizes its conjunction
/// outright, so choosing it saves the correction gate in [`emit_atom_onto`];
/// among the choices that do, and otherwise among all of them, the draw is
/// uniform.
///
/// Steering instead toward a SAME-polarity rung (always available among three
/// literals, by pigeonhole) would give the rung four equal-size spellings
/// rather than one. Measured and rejected: those four spellings are two gate
/// multisets reordered, so an order-blind duplicate census does not see them,
/// and forcing the choice costs gates.
fn choose_pivot(atom: &[(u16, bool)], rng: &mut impl Rng) -> usize {
    debug_assert!(atom.len() <= 3);
    let mut cheap = [0usize; 3];
    let mut cheap_len = 0usize;
    for index in 0..atom.len() {
        if (0..atom.len())
            .filter(|&other| other != index)
            .all(|other| !atom[other].1)
        {
            cheap[cheap_len] = index;
            cheap_len += 1;
        }
    }
    if cheap_len == 0 {
        rng.random_range(0..atom.len())
    } else {
        cheap[rng.random_range(0..cheap_len)]
    }
}

/// Add one mask atom to a DIRTY accumulator, every emitted gate at most two
/// controls, over a dirty borrowed helper. Returns the residual constant.
///
/// `pivot` fixes which literal pairs with the helper on the accumulator's own
/// gates, and the caller draws it ONCE for both the gather and the strip. That
/// is not cosmetic. The residual depends on the rung's POLARITY PATTERN, so
/// routing a gather and its strip through independent pivot draws leaves
/// DIFFERENT residuals, and the accumulator comes back off by one. The borrow
/// is then not restored at all, which is a corrupted wire rather than a wrong
/// constant. (Caught by `prod_gray_fold_keeps_the_accumulators_dirty`.)
///
/// Spellings still differ between the two passes: every spelling of one
/// function carries the same constant, which is exactly what makes varying them
/// safe here.
fn emit_atom_onto(
    acc: u16,
    atom: &[(u16, bool)],
    helper: u16,
    pivot: usize,
    menu: usize,
    seen: &mut SpellingLog,
    rng: &mut impl Rng,
    out: &mut Vec<XGate>,
) -> bool {
    if atom.len() <= 2 {
        let (mut spellings, konst) = spellings_at(acc, atom, menu);
        let pick = pick_spelling(acc, atom, spellings.len(), seen, rng);
        out.extend(spellings.swap_remove(pick));
        return konst;
    }
    debug_assert_eq!(atom.len(), 3, "only degree <= 3 masks are gathered");
    let lam = atom[pivot];
    let rung = match pivot {
        0 => [atom[1], atom[2]],
        1 => [atom[0], atom[2]],
        2 => [atom[0], atom[1]],
        _ => unreachable!("degree-3 pivot is in range"),
    };
    // The helper literal's polarity is FREE -- the sweep contributes
    // `lam * (f(h0) + f(h0 + R))` for whichever literal `f` of the helper is
    // used, and negating it adds 1 to both readings, which cancels. Matching it
    // to `lam` would make this gate same-polarity and so give it four
    // equal-size spellings instead of one. MEASURED and REJECTED: it costs
    // +5.7% gates (a mixed-polarity conjunction is a 1-gate emission, a
    // same-polarity one is 2) and bought nothing, because the four spellings of
    // a same-polarity function are only TWO distinct gate multisets reordered,
    // and both the duplicate census and `commuting_shuffle` are order-blind.
    // Keep the cheap spelling.
    let t_lits = [lam, (helper, true)];
    let rung_konst = rung_residual(&rung);
    // Accumulator gate, then rung, twice: the helper is visited an even number
    // of times, so its unknown incoming value cancels and it is restored. The
    // accumulator gate's own complement cancels between its two emissions.
    for _ in 0..2 {
        let (mut sp, _) = spellings_at(acc, &t_lits, menu);
        let pick = pick_spelling(acc, &t_lits, sp.len(), seen, rng);
        out.extend(sp.swap_remove(pick));
        let (mut sp, _) = spellings_at(helper, &rung, menu);
        let pick = pick_spelling(helper, &rung, sp.len(), seen, rng);
        out.extend(sp.swap_remove(pick));
    }
    // A rung spelled with a complement leaves `kappa * lam` on the accumulator
    // -- a LITERAL, not a constant, because it multiplies the accumulator
    // gate's other literal. One more emission of `lam` removes it; that
    // emission's own constant is what the caller absorbs.
    if rung_konst {
        return emit_g57_form(acc, &[lam], rng, out);
    }
    false
}

/// Realize a wide conjunction using dirty borrowed carrier wires and restore
/// every borrow exactly. This is the generalized Barenco double sweep used by
/// the reference implementation's optional narrow mode.
fn emit_narrow_fragment(
    target: u16,
    lits: &[(u16, bool)],
    cap: usize,
    carrier_total: usize,
    borrow_total: usize,
    rng: &mut impl Rng,
    out: &mut Vec<XGate>,
) -> bool {
    debug_assert!(cap >= 2);
    debug_assert!(!lits.is_empty());
    if lits.len() <= cap {
        if lits.len() <= 2 {
            return emit_g57_form(target, lits, rng, out);
        }
        out.push(XGate::conj(target, lits.iter().copied()).expect("normalized literals"));
        return false;
    }

    let chunk = cap - 1;
    let rung_count = (lits.len() - cap).div_ceil(chunk);
    let mut taken = vec![target as usize];
    taken.extend(lits.iter().map(|&(wire, _)| wire as usize));
    let free_in = |pool: usize| pool - taken.iter().filter(|&&wire| wire < pool).count();
    let borrow_pool = if free_in(carrier_total) >= rung_count {
        carrier_total
    } else {
        borrow_total
    };
    assert!(
        free_in(borrow_pool) >= rung_count,
        "narrow fragment needs {rung_count} borrows but only {} of {borrow_pool} wires are free",
        free_in(borrow_pool)
    );
    let mut borrowed = Vec::with_capacity(rung_count);
    for _ in 0..rung_count {
        let helper = random_wire_except(borrow_pool, &taken, rng) as u16;
        taken.push(helper as usize);
        borrowed.push(helper);
    }

    let mut rungs: Vec<XGate> = Vec::with_capacity(rung_count);
    let mut consumed = 0;
    for (index, &helper) in borrowed.iter().enumerate() {
        let mut rung_lits = Vec::with_capacity(cap);
        if index == 0 {
            rung_lits.extend_from_slice(&lits[..cap]);
            consumed = cap;
        } else {
            rung_lits.push((borrowed[index - 1], true));
            let upto = (consumed + chunk).min(lits.len());
            rung_lits.extend_from_slice(&lits[consumed..upto]);
            consumed = upto;
        }
        rungs.push(
            XGate::conj(helper, rung_lits)
                .expect("borrowed wires are distinct from fragment literals"),
        );
    }

    let mut target_lits = vec![(*borrowed.last().unwrap(), true)];
    target_lits.extend_from_slice(&lits[consumed..]);
    debug_assert!(target_lits.len() <= cap);

    for _ in 0..2 {
        if target_lits.len() <= 2 {
            let _ = emit_g57_form(target, &target_lits, rng, out);
        } else {
            out.push(
                XGate::conj(target, target_lits.iter().copied())
                    .expect("borrowed target fragment is valid"),
            );
        }
        for rung in rungs.iter().rev() {
            out.push(rung.clone());
        }
        for rung in rungs.iter().skip(1) {
            out.push(rung.clone());
        }
    }
    false
}

/// Dedupe an interleaved literal list without sorting it. Opposite
/// polarities on one wire make the conjunction identically zero.
fn normalize_prod_lits(lits: &mut Vec<(u16, bool)>) -> Option<()> {
    // Detect contradictions before mutating, retaining the old all-or-nothing
    // failure behavior and first-occurrence ordering.
    for first in 0..lits.len() {
        for second in first + 1..lits.len() {
            if lits[first].0 == lits[second].0 && lits[first].1 != lits[second].1 {
                return None;
            }
        }
    }
    let mut write = 0usize;
    for read in 0..lits.len() {
        let literal = lits[read];
        if !lits[..write].iter().any(|&(wire, _)| wire == literal.0) {
            lits[write] = literal;
            write += 1;
        }
    }
    lits.truncate(write);
    Some(())
}

/// Ledger for the nonlinear decode. Product sources are frozen band values,
/// whose physical locations may roll through the carrier/band wire space.
struct ProdLedger {
    plan: Vec<usize>,
    carrier_total: usize,
    /// Physical wire currently holding each band variable.
    loc: Vec<u16>,
    cap: usize,
    slots: Vec<Vec<ProdSlot>>,
    consts: Vec<bool>,
    used: std::collections::HashSet<ProdSlot>,
    /// One linear carrier per value instead of a pair (`ProdConfig::single`).
    single: bool,
    /// Gather each operand's mask sum onto a dirty accumulator (see
    /// [`ProdLedger::fold_cg_gray`]).
    gray_fold: bool,
    /// Spelling-menu level for the Gray fold's emissions (`spellings_at`).
    rung_menu: usize,
    injected: u64,
    resourced: u64,
    rolled: u64,
    cg_fragments: u64,
    /// Fold fragments narrow enough for the frozen-DB channel (<= 2 controls
    /// at the production --db-ctrl-cap). The fold is the material that carries
    /// the computation, so this is the fraction of the CORE that phase A's
    /// re-encoding can ever reach.
    cg_narrow: u64,
    /// CG blocks emitted by the Gray fold rather than the odometer.
    cg_gray: u64,
    ledger_consts: u64,
}

impl ProdLedger {
    fn new(n: usize, config: &ProdConfig, carrier_total: usize) -> ProdLedger {
        let band_len = config.band_size(n);
        let mut plan = Vec::new();
        plan.extend(std::iter::repeat_n(config.deg.max(2), config.k));
        plan.extend(std::iter::repeat_n(config.deg_hi.max(2), config.k_hi));

        if config.enabled() {
            let max_degree = config.max_deg();
            assert!(
                band_len > max_degree,
                "nonlinear gadget source band needs at least max_degree+1 wires"
            );
            let mut tuple_space = 1u128;
            for index in 0..max_degree {
                tuple_space = tuple_space * (band_len - index) as u128 / (index as u128 + 1);
            }
            tuple_space = tuple_space.saturating_mul(1u128 << max_degree);
            assert!(
                tuple_space >= (2 * n * config.k_total()) as u128,
                "nonlinear gadget source band is too small: {band_len} wires provide \
                 {tuple_space} degree-{max_degree} slots for {} live terms",
                n * config.k_total()
            );
        }

        ProdLedger {
            plan,
            carrier_total,
            loc: (carrier_total as u16..(carrier_total + band_len) as u16).collect(),
            cap: if config.narrow() { config.max_width } else { 0 },
            slots: vec![Vec::new(); n],
            consts: vec![false; n],
            used: std::collections::HashSet::new(),
            single: config.single_carrier(),
            gray_fold: config.gray(),
            rung_menu: config.rung_menu,
            injected: 0,
            resourced: 0,
            rolled: 0,
            cg_fragments: 0,
            cg_narrow: 0,
            cg_gray: 0,
            ledger_consts: 0,
        }
    }

    fn enabled(&self) -> bool {
        !self.plan.is_empty()
    }

    fn borrow_total(&self) -> usize {
        self.carrier_total + self.loc.len()
    }

    fn draw_slot(&mut self, degree: usize, rng: &mut impl Rng) -> ProdSlot {
        let band_len = self.loc.len() as u16;
        for _ in 0..100_000 {
            let mut variables: smallvec::SmallVec<[u16; 4]> =
                smallvec::SmallVec::with_capacity(degree);
            while variables.len() < degree {
                let variable = rng.random_range(0..band_len);
                if !variables.contains(&variable) {
                    variables.push(variable);
                }
            }
            variables.sort_unstable();
            let slot = ProdSlot {
                factors: variables
                    .into_iter()
                    .map(|variable| (variable, rng.random::<bool>()))
                    .collect(),
            };
            if self.used.insert(slot.clone()) {
                return slot;
            }
        }
        panic!("nonlinear gadget source band exhausted");
    }

    fn emit_slot(
        &self,
        value: usize,
        slot: &ProdSlot,
        state: &GadgetState,
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) -> bool {
        let (carrier0, carrier1) = state.pairs[value];
        let target = if rng.random_bool(0.5) {
            carrier0
        } else {
            carrier1
        } as u16;
        let lits = slot.lits(&self.loc);
        if self.cap >= 2 {
            emit_narrow_fragment(
                target,
                &lits,
                self.cap,
                self.carrier_total,
                self.borrow_total(),
                rng,
                out,
            )
        } else {
            out.push(XGate::conj(target, lits).expect("product sources are distinct band values"));
            false
        }
    }

    fn inject(
        &mut self,
        value: usize,
        degree: usize,
        state: &GadgetState,
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) {
        let slot = self.draw_slot(degree, rng);
        self.consts[value] ^= self.emit_slot(value, &slot, state, rng, out);
        self.slots[value].push(slot);
        self.injected += 1;
    }

    fn inject_all(&mut self, state: &GadgetState, rng: &mut impl Rng, out: &mut Vec<XGate>) {
        if !self.enabled() {
            return;
        }
        for value in 0..state.n {
            for index in 0..self.plan.len() {
                self.inject(value, self.plan[index], state, rng, out);
            }
        }
    }

    /// Replace one live term, injecting the replacement before stripping the
    /// old term so the value is never momentarily left with fewer masks.
    fn resource(&mut self, state: &GadgetState, rng: &mut impl Rng, out: &mut Vec<XGate>) {
        if !self.enabled() {
            return;
        }
        let value = rng.random_range(0..state.n);
        let old_index = rng.random_range(0..self.slots[value].len());
        let degree = self.slots[value][old_index].factors.len();
        self.inject(value, degree, state, rng, out);
        let old = self.slots[value].remove(old_index);
        self.consts[value] ^= self.emit_slot(value, &old, state, rng, out);
        self.used.remove(&old);
        self.resourced += 1;
    }

    /// Relocate one band value by swapping its physical wire with another
    /// band wire or carrier. Slots continue naming values and resolve through
    /// `loc`, while carrier ownership is re-pointed in `state`.
    fn roll(&mut self, state: &mut GadgetState, rng: &mut impl Rng, out: &mut Vec<XGate>) {
        if !self.enabled() || self.loc.is_empty() {
            return;
        }
        let total = self.borrow_total();
        let variable = rng.random_range(0..self.loc.len());
        let from = self.loc[variable];
        let to = loop {
            let wire = rng.random_range(0..total) as u16;
            if wire != from {
                break wire;
            }
        };

        if let Some(other_variable) = self.loc.iter().position(|&wire| wire == to) {
            self.loc[other_variable] = from;
        } else {
            let mut found = false;
            for value in 0..state.n {
                let (carrier0, carrier1) = state.pairs[value];
                if self.single {
                    // One carrier: both entries name the SAME wire, so both
                    // must follow it across the swap. Re-pointing only `.0`
                    // (the two-carrier branch below) would leave the pair
                    // half-moved, and `fold_cg`'s collapse assertion — plus
                    // every decode — would then be reading a wire the value no
                    // longer lives on.
                    if carrier0 == to as usize {
                        state.pairs[value] = (from as usize, from as usize);
                        found = true;
                        break;
                    }
                    continue;
                }
                if carrier0 == to as usize {
                    state.pairs[value].0 = from as usize;
                    found = true;
                    break;
                }
                if carrier1 == to as usize {
                    state.pairs[value].1 = from as usize;
                    found = true;
                    break;
                }
            }
            debug_assert!(
                found,
                "roll partner {to} is neither a band wire nor a carrier"
            );
        }
        self.loc[variable] = to;

        let (first, second) = if rng.random_bool(0.5) {
            (from, to)
        } else {
            (to, from)
        };
        for (target, source) in [(first, second), (second, first), (first, second)] {
            emit_transvection_mixed(target, source, total, rng, out);
        }
        self.rolled += 1;
    }

    /// Apply one logical gate without reconstructing any operand. Each
    /// operand literal is expanded into its full nonlinear decode, and the
    /// Cartesian products are emitted directly as target-carrier fragments.
    fn fold_cg(
        &mut self,
        gate: &XGate,
        state: &GadgetState,
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) {
        let target_value = gate.target as usize;
        debug_assert!(target_value < state.n);

        let mut lists: Vec<Vec<Vec<(u16, bool)>>> = Vec::with_capacity(gate.ctrls.len());
        for &(wire, positive) in &gate.ctrls {
            let value = wire as usize;
            debug_assert_ne!(value, target_value);
            let (carrier0, carrier1) = state.pairs[value];
            // Single-carrier decode contributes ONE linear atom, not two: the
            // pair collapses to a single wire, and emitting it twice would
            // cancel. Every other atom (the mask terms, the constant) is
            // unchanged, so the fold's fragment count drops from
            // (2 + k)^arity to (1 + k)^arity at equal mask strength.
            let mut atoms = if self.single {
                assert_eq!(
                    carrier0, carrier1,
                    "single-carrier decode on a two-carrier pairing: no entry point in this \
                     tree lays out pairs[v] = (w, w), so ProdConfig::single is not usable here"
                );
                vec![vec![(carrier0 as u16, true)]]
            } else {
                vec![vec![(carrier0 as u16, true)], vec![(carrier1 as u16, true)]]
            };
            atoms.extend(
                self.slots[value]
                    .iter()
                    .map(|slot| slot.lits(&self.loc).into_vec()),
            );
            if self.consts[value] ^ !positive {
                atoms.push(Vec::new());
            }
            lists.push(atoms);
        }

        if gate.comp {
            self.consts[target_value] ^= true;
            self.ledger_consts += 1;
        }

        // The Gray fold handles the arity-2 blocks -- which is every source
        // gate in the g57 body -- with no wide fragment at all. It declines the
        // shapes it cannot amortize (arity != 2, an operand with no mask terms,
        // a degree-4+ mask, no room to borrow); those fall through to the
        // odometer below, exactly as before.
        if self.gray_fold && self.fold_cg_gray(target_value, &lists, state, rng, out) {
            return;
        }

        let mut fragments = Vec::new();
        let mut combination = vec![0usize; lists.len()];
        'odometer: loop {
            let picked: Vec<&Vec<(u16, bool)>> = lists
                .iter()
                .zip(&combination)
                .map(|(list, &index)| &list[index])
                .collect();
            let width = picked.iter().map(|atom| atom.len()).max().unwrap_or(0);
            let mut lits = Vec::new();
            // Interleave atom literals so optional narrow ladders never
            // materialize one complete logical operand on a helper wire.
            for index in 0..width {
                for atom in &picked {
                    if let Some(&lit) = atom.get(index) {
                        lits.push(lit);
                    }
                }
            }

            if lits.is_empty() {
                self.consts[target_value] ^= true;
                self.ledger_consts += 1;
            } else if normalize_prod_lits(&mut lits).is_some() {
                fragments.push(lits);
            }

            let mut axis = 0;
            loop {
                if axis == combination.len() {
                    break 'odometer;
                }
                combination[axis] += 1;
                if combination[axis] < lists[axis].len() {
                    break;
                }
                combination[axis] = 0;
                axis += 1;
            }
        }

        // With the Gray fold on, the blocks it DECLINED are laddered rather
        // than left wide: a single surviving 3-control gate would undo the
        // point of the exercise. (The reference forces `ladder_cap` to
        // `usize::MAX` at this same seam.) Off, this is exactly `self.cap`.
        let cap = if self.gray_fold {
            self.cap.max(2)
        } else {
            self.cap
        };
        fragments.shuffle(rng);
        for lits in &fragments {
            let (carrier0, carrier1) = state.pairs[target_value];
            let target = if rng.random_bool(0.5) {
                carrier0
            } else {
                carrier1
            } as u16;
            let residual = if cap >= 2 {
                emit_narrow_fragment(
                    target,
                    lits,
                    cap,
                    self.carrier_total,
                    self.borrow_total(),
                    rng,
                    out,
                )
            } else {
                out.push(
                    XGate::conj(target, lits.iter().copied())
                        .expect("normalized product-fold fragment"),
                );
                false
            };
            self.consts[target_value] ^= residual;
            self.cg_fragments += 1;
        }
    }

    /// The carrier of `value` the encoding may write. Both are free under the
    /// band build, so the choice stays random.
    fn free_carrier(&self, value: usize, state: &GadgetState, rng: &mut impl Rng) -> u16 {
        let (carrier0, carrier1) = state.pairs[value];
        if rng.random_bool(0.5) {
            carrier0 as u16
        } else {
            carrier1 as u16
        }
    }

    /// The wires CURRENTLY holding carriers, resolved by role rather than by
    /// index. Rolling exchanges a band variable with an arbitrary wire, so the
    /// home index range stops describing the carrier set as soon as
    /// `roll` is on, and anything that borrows by index leaves a static
    /// index-shaped trace that rolling cannot average out.
    fn carrier_wires(&self, state: &GadgetState) -> Vec<u16> {
        let mut wires: Vec<u16> = Vec::with_capacity(2 * state.n);
        for value in 0..state.n {
            let (carrier0, carrier1) = state.pairs[value];
            wires.push(carrier0 as u16);
            if carrier1 != carrier0 {
                wires.push(carrier1 as u16);
            }
        }
        wires.sort_unstable();
        wires.dedup();
        wires
    }

    /// wire -> the other carrier of the same value (identity on the band).
    /// Built over the WHOLE wire space: a roll can put a value's carrier on a
    /// former band wire, so indexing by `carrier_total` alone goes out of
    /// bounds the moment rolling is on.
    fn sibling_map(&self, state: &GadgetState) -> Vec<u16> {
        let mut sibling: Vec<u16> = (0..self.borrow_total() as u16).collect();
        for value in 0..state.n {
            let (carrier0, carrier1) = state.pairs[value];
            sibling[carrier0] = carrier1 as u16;
            sibling[carrier1] = carrier0 as u16;
        }
        sibling
    }

    /// Two dirty accumulator wires for the Gray fold, drawn by ROLE from the
    /// carriers and excluding everything the block reads or writes. `None` when
    /// the pool is too small, which makes the caller fall back to the odometer.
    fn pick_accumulators(
        &self,
        forbidden: &[u16],
        sibling: &[u16],
        carrier_wires: &[u16],
        rng: &mut impl Rng,
    ) -> Option<(u16, u16)> {
        debug_assert!(forbidden.is_sorted());
        let free: Vec<u16> = carrier_wires
            .iter()
            .copied()
            .filter(|wire| forbidden.binary_search(wire).is_err())
            .collect();
        use rand::seq::IndexedRandom;
        let u = *free.choose(rng)?;
        // The second accumulator must not be the first's sibling: the four
        // A_u * A_z gates read BOTH, and one gate seeing both carriers of one
        // value is exactly the gate-local completeness the construction bans.
        let free2: Vec<u16> = free
            .into_iter()
            .filter(|&wire| wire != u && wire != sibling[u as usize])
            .collect();
        let z = *free2.choose(rng)?;
        Some((u, z))
    }

    /// Add every mask term in `atoms` to the dirty accumulator `acc`, following
    /// the per-atom `plan` of (helper wire, pivot) the caller drew ONCE. Returns
    /// the residual constant parity the realization leaves on `acc`.
    ///
    /// Reusing the plan is what makes the gather and the strip leave the SAME
    /// residual, which is what restores the accumulator: see [`emit_atom_onto`].
    fn gather_atoms(
        &self,
        acc: u16,
        atoms: &[Vec<(u16, bool)>],
        plan: &[(u16, usize)],
        seen: &mut SpellingLog,
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) -> bool {
        let mut konst = false;
        for (atom, &(helper, pivot)) in atoms.iter().zip(plan) {
            konst ^= emit_atom_onto(acc, atom, helper, pivot, self.rung_menu, seen, rng, out);
        }
        konst
    }

    /// GRAY-CODE FOLD: the arity-2 CG with every emitted gate at most TWO
    /// controls, without laddering a single fragment.
    ///
    /// The wide fold expands `PROD_w (carriers_w + masks_w)` into one gate per
    /// term of the cartesian product, so a `[1,1,2,3]` share gives 25 fragments
    /// of width up to `arity * max_deg` = 6. Everything above width 2 is
    /// invisible to the frozen store (width-3 gates hit at 0.41% against ~99%
    /// for narrow material) and is 56% of the shipped gadget. Laddering each
    /// wide fragment individually is full narrow mode's ~6.2x, because every
    /// fragment re-derives the same mask products from scratch.
    ///
    /// This gathers each operand's mask sum ONCE onto a borrowed wire and reads
    /// it back four times. Write `S_w = L_w + M_w` for operand `w`, with `L_w`
    /// the width-<=1 atoms (the carrier literals and the ledger's constant atom)
    /// and `M_w` the mask terms. Two dirty accumulators `u`, `z` are toggled
    /// around the Gray cycle over `(u holds M_b, z holds M_c)`:
    ///
    /// ```text
    ///   A=(0,0) --gather b--> B=(1,0) --gather c--> C=(1,1)
    ///                                                  |
    ///           A=(0,0) <--strip c-- D=(0,1) <--strip b--
    /// ```
    ///
    /// Reading `u` at one phase from each column and `z` at one from each row
    /// gives `A_u = U_1 + U_2 = M_b` and `A_z = M_c` -- the borrows' unknown
    /// incoming values cancel between the two readings, exactly as the ladder's
    /// double sweep cancels its own. Emitting `L_b x L_c` once, `L_b x A_z` and
    /// `A_u x L_c` once per column/row, and `A_u x A_z` once at every phase sums
    /// to `(L_b + A_u)(L_c + A_z) = S_b S_c`, and every one of those gates has at
    /// most two controls by construction.
    ///
    /// WHY THE ACCUMULATORS MUST BE DIRTY. `carrier_b + M_b` IS the operand's
    /// value: a clean accumulator would put `b` one XOR away from a wire pair,
    /// which is the same use-point re-exposure that sank the deferred-mask peek.
    /// Borrowed, the wire holds `u_0 + M_b`, and `u_0` is off the wire set for
    /// the whole dirty window. MEASURED over all 46 prefixes of the [1,2,3,3]
    /// block, exactly (Walsh transform per residual component, masks restricted
    /// to variables still live): the best affine predictor of any of `a, b, c,
    /// a'` peaks at 0.28125 = (1/2)(3/4)^2, which is the encoding's own
    /// steady-state piling-up bound -- the block's interior gives an affine
    /// adversary nothing the endpoints do not. The same audit run against a
    /// CLEAN-accumulator variant recovers `b` at correlation 1.0.
    ///
    /// THE RESIDUAL-CONSTANT TRAP. `emit_g57_form` leaves a complement on its
    /// target, so a gather actually lands `M_b + delta`. Left alone that is not
    /// a leak but a WRONG FUNCTION: the four-phase sum becomes
    /// `(M_b + delta)(M_c + eps)`, i.e. the block silently acquires
    /// `delta*M_c + eps*M_b`. It is absorbed for free by toggling the operand's
    /// CONSTANT ATOM -- `L'_w = L_w + delta` restores `L'_w + A_w = S_w` -- which
    /// is the same ledger mechanism the wide fold already uses for a negative
    /// control, and costs at most one extra width-<=1 fragment per operand.
    ///
    /// Returns false when the block does not fit the shape (arity != 2, or an
    /// operand with no mask terms to gather, or no room to borrow), leaving the
    /// caller to emit it the ordinary way.
    fn fold_cg_gray(
        &mut self,
        t: usize,
        lists: &[Vec<Vec<(u16, bool)>>],
        state: &GadgetState,
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) -> bool {
        if lists.len() != 2 {
            return false;
        }
        // The width-<=1 atoms stay in place; the mask terms are what an
        // accumulator gathers. Gathering a carrier would be pointless (it is
        // already one literal) and actively harmful: `u_0 + carrier_b` next to
        // the mask terms is the clean-accumulator failure in another spelling.
        let mut simple: Vec<Vec<Vec<(u16, bool)>>> = Vec::with_capacity(2);
        let mut masks: Vec<Vec<Vec<(u16, bool)>>> = Vec::with_capacity(2);
        for list in lists {
            simple.push(list.iter().filter(|a| a.len() <= 1).cloned().collect());
            masks.push(list.iter().filter(|a| a.len() >= 2).cloned().collect());
        }
        if masks[0].is_empty() || masks[1].is_empty() {
            // Nothing to amortize on one side: the plain expansion is at most
            // `1 + max_deg` wide and the accumulator would cost more than the
            // laddering it saves.
            return false;
        }
        if masks.iter().flatten().any(|atom| atom.len() > 3) {
            // A degree-4+ term needs a helper chain rather than one sandwich;
            // the production plan is [2,2,2,3] and the generalization is not
            // verified, so decline rather than emit something unaudited.
            return false;
        }
        // Borrow nothing the block reads or writes, nor any sibling of those:
        // a borrow that is the sibling of a literal would put both carriers of
        // one value into one gate across a read and a write.
        let sibling = self.sibling_map(state);
        let mut forbid: Vec<u16> = vec![state.pairs[t].0 as u16, state.pairs[t].1 as u16];
        for list in lists {
            for atom in list {
                for &(wire, _) in atom {
                    forbid.push(wire);
                    if (wire as usize) < sibling.len() {
                        forbid.push(sibling[wire as usize]);
                    }
                }
            }
        }
        for wire in [state.pairs[t].0, state.pairs[t].1] {
            if wire < sibling.len() {
                forbid.push(sibling[wire]);
            }
        }
        forbid.sort_unstable();
        forbid.dedup();
        let carrier_wires = self.carrier_wires(state);
        let Some((u, z)) = self.pick_accumulators(&forbid, &sibling, &carrier_wires, rng) else {
            return false;
        };
        // The sandwich helper must avoid the OTHER accumulator: it is restored
        // either way, but a gate reading `z` while writing `u` would mix the two
        // operands' mask material on one wire for no reason.
        let helper_pool: Vec<u16> = carrier_wires
            .iter()
            .copied()
            .filter(|&wire| wire != u && wire != z && forbid.binary_search(&wire).is_err())
            .collect();
        if helper_pool.is_empty() {
            return false;
        }
        // Per mask atom: the helper it sandwiches over and which literal pairs
        // with that helper. Drawn ONCE and reused by both the gather and the
        // strip, because a re-draw would change the residual and leave the
        // accumulator unrestored.
        let mut plans: Vec<Vec<(u16, usize)>> = Vec::with_capacity(2);
        for side in 0..2 {
            plans.push(
                masks[side]
                    .iter()
                    .map(|atom| {
                        (
                            helper_pool[rng.random_range(0..helper_pool.len())],
                            choose_pivot(atom, rng),
                        )
                    })
                    .collect(),
            );
        }

        // Emit the four toggles into buffers FIRST, so the residual parities are
        // known before the product terms are planned. A strip must return the
        // same parity as its gather -- that is what restores the borrow -- so
        // assert it rather than trust it.
        let mut toggles: Vec<Vec<XGate>> = Vec::with_capacity(4);
        let mut konst = [false; 2];
        // One spelling log for the whole block: a function emitted in the
        // gather and again in the strip must not be spelled the same way twice.
        let mut seen: SpellingLog = std::collections::HashMap::with_capacity(32);
        for (index, (acc, side)) in [(u, 0usize), (z, 1usize), (u, 0), (z, 1)]
            .into_iter()
            .enumerate()
        {
            let mut buf = Vec::with_capacity(masks[side].len() * 6);
            let parity =
                self.gather_atoms(acc, &masks[side], &plans[side], &mut seen, rng, &mut buf);
            if index < 2 {
                konst[side] = parity;
            } else {
                assert_eq!(
                    parity, konst[side],
                    "gray fold: strip parity differs from its gather, so the \
                     accumulator would not be restored"
                );
            }
            toggles.push(buf);
        }
        // Absorb the gathers' residual into the operand's constant atom.
        for side in 0..2 {
            if konst[side] {
                match simple[side].iter().position(|atom| atom.is_empty()) {
                    Some(index) => {
                        simple[side].remove(index);
                    }
                    None => simple[side].push(Vec::new()),
                }
            }
        }

        // Phase (u holds M_b, z holds M_c): A=(0,0) B=(1,0) C=(1,1) D=(0,1).
        const PA: usize = 0;
        const PB: usize = 1;
        const PC: usize = 2;
        const PD: usize = 3;
        let mut phases: Vec<Vec<Vec<(u16, bool)>>> = vec![Vec::new(); 4];
        // L_b x L_c: both factors are phase-independent, so these may sit
        // anywhere -- and they are spread at random rather than parked at one
        // phase, which would make the block's opening a fixed signature.
        for a in &simple[0] {
            for b in &simple[1] {
                let mut lits = a.clone();
                lits.extend_from_slice(b);
                phases[rng.random_range(0..4)].push(lits);
            }
        }
        // L_b x A_z: once where z is bare, once where it carries M_c.
        for a in &simple[0] {
            let mut lits = a.clone();
            lits.push((z, true));
            phases[[PA, PB][rng.random_range(0..2)]].push(lits.clone());
            phases[[PC, PD][rng.random_range(0..2)]].push(lits);
        }
        // A_u x L_c: once where u is bare, once where it carries M_b.
        for b in &simple[1] {
            let mut lits = vec![(u, true)];
            lits.extend_from_slice(b);
            phases[[PA, PD][rng.random_range(0..2)]].push(lits.clone());
            phases[[PB, PC][rng.random_range(0..2)]].push(lits);
        }
        // A_u x A_z: once at every phase, which is the inclusion-exclusion.
        for phase in phases.iter_mut() {
            phase.push(vec![(u, true), (z, true)]);
        }

        // The Gray structure emits the SAME literal list more than once by
        // construction -- `A_u * A_z` at all four phases, each `L x A` pair
        // twice -- so leaving each emission to its own coin plants exact gate
        // groups that `sort | uniq -c` finds without executing anything. The
        // spelling log picks a different spelling each time; the product
        // fragments are mostly all-positive (carrier and accumulator literals),
        // which is precisely the case with four equal-size spellings, so this
        // costs nothing.
        for (index, &phase) in [PA, PB, PC, PD].iter().enumerate() {
            let mut frags = std::mem::take(&mut phases[phase]);
            // Within a phase the fragments commute (they all XOR into the
            // target's carriers and read nothing the block writes), so the
            // order is free -- and a fixed one would be a per-block clock in
            // the same way the odometer was.
            frags.shuffle(rng);
            for mut lits in frags {
                if normalize_prod_lits(&mut lits).is_none() {
                    continue; // contradictory literals: the term is 0
                }
                if lits.is_empty() {
                    self.consts[t] ^= true;
                    self.ledger_consts += 1;
                    continue;
                }
                let target = self.free_carrier(t, state, rng);
                let (mut spellings, residual) = spellings_at(target, &lits, self.rung_menu);
                let pick = pick_spelling(target, &lits, spellings.len(), &mut seen, rng);
                out.extend(spellings.swap_remove(pick));
                self.consts[t] ^= residual;
                self.cg_fragments += 1;
                self.cg_narrow += 1;
            }
            out.append(&mut toggles[index]);
        }
        self.cg_gray += 1;
        true
    }

    fn strip_all(&mut self, state: &GadgetState, rng: &mut impl Rng, out: &mut Vec<XGate>) {
        for value in 0..state.n {
            while !self.slots[value].is_empty() {
                let slot = self.slots[value].remove(0);
                self.consts[value] ^= self.emit_slot(value, &slot, state, rng, out);
                self.used.remove(&slot);
            }
            if self.consts[value] {
                let (carrier0, carrier1) = state.pairs[value];
                let target = if rng.random_bool(0.5) {
                    carrier0
                } else {
                    carrier1
                };
                let helper = random_wire_except(self.carrier_total, &[target], rng) as u16;
                out.push(
                    XGate::conj(target as u16, [(helper, false)])
                        .expect("helper differs from target"),
                );
                out.push(XGate::cnot(target as u16, helper));
                self.consts[value] = false;
            }
        }
    }

    fn report(&self) {
        if self.enabled() {
            println!(
                "[nonlinear-gadgetize] plan={:?} band={} injected={} resourced={} \
                 rolled={} cg_fragments={} cg_narrow={} gray_blocks={} ledger_consts={}",
                self.plan,
                self.loc.len(),
                self.injected,
                self.resourced,
                self.rolled,
                self.cg_fragments,
                self.cg_narrow,
                self.cg_gray,
                self.ledger_consts
            );
        }
    }
}

/// Realize `target ^= source` either as a CNOT or as two complementary
/// width-2 conjunctions. Both forms are exact, while the mixed shapes avoid
/// making rolled band locations identifiable by width-1 writes alone.
fn emit_transvection_mixed(
    target: u16,
    source: u16,
    total: usize,
    rng: &mut impl Rng,
    out: &mut Vec<XGate>,
) {
    if rng.random_bool(0.5) {
        out.push(XGate::cnot(target, source));
        return;
    }
    let helper = random_wire_except(total, &[target as usize, source as usize], rng) as u16;
    for polarity in [true, false] {
        out.push(
            XGate::conj(target, [(source, true), (helper, polarity)])
                .expect("mixed transvection wires are distinct"),
        );
    }
}

/// Fill the band's current physical wires with balanced affine functions of
/// the input.
fn emit_band_fill(n: usize, band: &[u16], rng: &mut impl Rng, out: &mut Vec<XGate>) {
    for &band_wire in band {
        loop {
            let subset: Vec<usize> = (0..n).filter(|_| rng.random_bool(0.5)).collect();
            if subset.len() < 2 {
                continue;
            }
            for data_wire in subset {
                out.push(XGate::cnot(band_wire, data_wire as u16));
            }
            break;
        }
    }
}

/// Fill the frozen band with balanced nonlinear cascades. A fresh pivot that
/// does not occur in the remainder of the wire's transitive support preserves
/// exact balance, while products involving earlier band wires raise degree.
fn emit_band_fill_nl(
    n: usize,
    band: &[u16],
    nonlinear_terms: usize,
    rng: &mut impl Rng,
    out: &mut Vec<XGate>,
) {
    let mut supports: Vec<std::collections::HashSet<usize>> = Vec::new();

    for (band_index, &band_wire) in band.iter().enumerate() {
        let pivot = rng.random_range(0..n);
        let mut support: std::collections::HashSet<usize> = std::iter::once(pivot).collect();
        out.push(XGate::cnot(band_wire, pivot as u16));

        let linear_max = (n - 1).min(7);
        let linear_weight = 1 + rng.random_range(0..linear_max);
        let mut pool: Vec<usize> = (0..n).filter(|&wire| wire != pivot).collect();
        for _ in 0..linear_weight {
            let index = rng.random_range(0..pool.len());
            let wire = pool.swap_remove(index);
            support.insert(wire);
            out.push(XGate::cnot(band_wire as u16, wire as u16));
        }

        let eligible_band: Vec<usize> = (0..supports.len())
            .filter(|&index| !supports[index].contains(&pivot))
            .collect();
        for _ in 0..nonlinear_terms {
            let mut draw = |exclude: Option<u16>| loop {
                let wire = if !eligible_band.is_empty() && rng.random_bool(0.5) {
                    let index = eligible_band[rng.random_range(0..eligible_band.len())];
                    band[index]
                } else {
                    loop {
                        let wire = rng.random_range(0..n);
                        if wire != pivot {
                            break wire as u16;
                        }
                    }
                };
                if Some(wire) != exclude {
                    break wire;
                }
            };
            let source1 = draw(None);
            let source2 = draw(Some(source1));
            let lits = [
                (source1, rng.random::<bool>()),
                (source2, rng.random::<bool>()),
            ];
            let _ = emit_g57_form(band_wire, &lits, rng, out);
            for source in [source1, source2] {
                match band[..band_index].iter().position(|&wire| wire == source) {
                    Some(earlier) => support.extend(supports[earlier].iter().copied()),
                    None => {
                        support.insert(source as usize);
                    }
                }
            }
        }
        supports.push(support);
    }
}

/// Uniform nonlinear RG draw from the legacy g57 RG1/RG2/RG3 networks.
fn emit_nonlinear_rg(
    state: &mut GadgetState,
    pair_queue: &mut VecDeque<(usize, usize)>,
    single_queue: &mut VecDeque<usize>,
    out: &mut Vec<XGate>,
    rng: &mut impl Rng,
) {
    let n = state.n;
    let total = 2 * n;
    let mut gates = Vec::new();
    match rng.random_range(0..3u32) {
        0 => {
            let (i, j) = next_pair(pair_queue, n, rng);
            emit_rg1(state, i, j, &mut gates);
        }
        1 => {
            let (i, j) = next_pair(pair_queue, n, rng);
            emit_rg2(state, i, j, &mut gates);
        }
        _ => {
            let i = next_single(single_queue, n, rng);
            let (carrier0, carrier1) = state.pairs[i];
            let source1 = random_wire_except(total, &[carrier0, carrier1], rng);
            let source2 = random_wire_except(total, &[carrier0, carrier1, source1], rng);
            emit_rg3(state, i, source1, source2, &mut gates);
        }
    }
    out.extend(gates.into_iter().map(XGate::from_g57));
}

fn commuting_insertion_pass(order: &mut Vec<u32>, gates: &[XGate], rng: &mut impl Rng) {
    let mut output = Vec::with_capacity(order.len());
    for &gate_index in order.iter() {
        let gate = &gates[gate_index as usize];
        let mut span = 0;
        while span < output.len()
            && !XGate::collides(gate, &gates[output[output.len() - 1 - span] as usize])
        {
            span += 1;
        }
        let position = output.len() - rng.random_range(0..=span);
        output.insert(position, gate_index);
    }
    *order = output;
}

/// Redraw a random linear extension of the gate-dependency order. Every move
/// is an adjacent swap of commuting gates, so functionality is unchanged.
fn commuting_shuffle(gates: &mut Vec<XGate>, rng: &mut impl Rng) {
    if gates.len() < 2 {
        return;
    }
    let mut order: Vec<u32> = (0..gates.len() as u32).collect();
    const PASSES: usize = 3;
    for _ in 0..PASSES {
        commuting_insertion_pass(&mut order, gates, rng);
        order.reverse();
    }
    if PASSES % 2 == 1 {
        order.reverse();
    }
    let reordered = order
        .into_iter()
        .map(|index| gates[index as usize].clone())
        .collect();
    *gates = reordered;
}

/// Two-share gadgetization of a heterogeneous logical circuit with native
/// CNOT linear bookends, masking-safe SGs and gate-locally non-complete
/// CNOT RG1/RG2/RG3 blocks.
fn gadgetize_xgates(main: &[XGate], n: usize, rg_freq: usize, rng: &mut impl Rng) -> CnotCircuit {
    gadgetize_xgates_with_prod(main, n, rg_freq, &ProdConfig::off(), rng)
}

/// Shared implementation for the ordinary two-share path and the nonlinear
/// product-share path. `ProdConfig::off()` preserves the ordinary path:
/// 2n wires, linear RG cadence, no source band, and no final commuting
/// shuffle.
fn gadgetize_xgates_with_prod(
    main: &[XGate],
    n: usize,
    rg_freq: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    gadgetize_xgates_with_prod_mode(main, n, rg_freq, prod, rng)
}

/// Internal two-carrier implementation retained for the ordinary gadgetizer
/// and private primitive regression tests.
fn gadgetize_xgates_with_prod_mode(
    main: &[XGate],
    n: usize,
    rg_freq: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    assert!(n >= 3, "gadgetize_xgates requires n >= 3");
    assert!(rg_freq > 0, "rg_freq must be nonzero");
    // This core lays out `pairs[v] = (share, pad)`. A single-carrier config
    // would reach `fold_cg`'s collapse assertion several thousand gates later
    // with nothing to say about how it got there; name the wrong door here.
    assert!(
        !prod.single_carrier(),
        "ProdConfig::single is the single-carrier decode: use \
         gadgetize_xgates_single / gadgetize_cnot_single, which lay out \
         pairs[v] = (w, w). This two-carrier core cannot honor it"
    );
    assert!(
        main.iter().all(|gate| {
            (gate.target as usize) < n && gate.ctrls.iter().all(|&(wire, _)| (wire as usize) < n)
        }),
        "logical gate wire outside 0..n"
    );

    let bookend_size = (2 * n * (n as f64).ln() as usize).max(64);
    let carrier_total = 2 * n;
    let band_range = carrier_total..carrier_total + prod.band_size(n);
    let total = band_range.end;
    assert!(total <= u16::MAX as usize, "too many wires");
    let mut out = rand_z_xgates(n, bookend_size, rng);
    let band_home: Vec<u16> = band_range.clone().map(|wire| wire as u16).collect();
    if prod.enabled() {
        if prod.fill_nl > 0 {
            emit_band_fill_nl(n, &band_home, prod.fill_nl, rng, &mut out);
        } else {
            emit_band_fill(n, &band_home, rng, &mut out);
        }
    }
    let mut dloc: Vec<usize> = (0..n).collect();
    let mut aloc: Vec<usize> = (n..2 * n).collect();
    let mut on: Vec<Slot> = (0..total)
        .map(|wire| {
            if wire < n {
                Slot::Data(wire)
            } else if wire < carrier_total {
                Slot::Aux(wire - n)
            } else {
                // Band values start on these home wires. Rolling may later
                // exchange their physical role with a carrier.
                Slot::Output(usize::MAX)
            }
        })
        .collect();
    let mut pairs = vec![(0usize, 0usize); n];

    for value in 0..n {
        let data = dloc[value];
        let aux = aloc[value];
        let share = loop {
            let wire = rng.random_range(0..carrier_total);
            if wire != data && wire != aux {
                break wire;
            }
        };
        let pad = loop {
            let wire = rng.random_range(0..carrier_total);
            if wire != data && wire != aux && wire != share {
                break wire;
            }
        };
        emit_w_i_cnot(data, aux, share, pad, &mut out);
        let moved_share = on[share];
        let moved_pad = on[pad];
        on[share] = Slot::Pair(value);
        on[pad] = Slot::Pair(value);
        pairs[value] = (share, pad);
        reloc(moved_share, share, data, &mut dloc, &mut aloc, &mut pairs);
        reloc(moved_pad, pad, aux, &mut dloc, &mut aloc, &mut pairs);
        on[data] = moved_share;
        on[aux] = moved_pad;
    }

    let mut state = GadgetState { n, pairs };
    let mut pair_queue = VecDeque::new();
    let mut single_queue = VecDeque::new();
    let mut prod_ledger = ProdLedger::new(n, prod, carrier_total);
    prod_ledger.inject_all(&state, rng, &mut out);

    for (index, gate) in main.iter().enumerate() {
        if prod.enabled() {
            prod_ledger.fold_cg(gate, &state, rng, &mut out);
        } else {
            emit_shared_fragment2(&state, gate, &mut out);
        }

        if prod.enabled() {
            if index + 1 == main.len() {
                continue;
            }
            // On nonlinear gadgetization `rg_freq` is the number of
            // nonlinear RG draws per inter-source-gate gap.
            for _ in 0..rg_freq {
                emit_nonlinear_rg(
                    &mut state,
                    &mut pair_queue,
                    &mut single_queue,
                    &mut out,
                    rng,
                );
            }
            for _ in 0..prod.rsrc {
                prod_ledger.resource(&state, rng, &mut out);
            }
            for _ in 0..prod.roll {
                prod_ledger.roll(&mut state, rng, &mut out);
            }
        } else if (index + 1) % rg_freq == 0 {
            // Historical plain-CNOT semantics: one linear RG after every
            // `rg_freq` source gadgets.
            match rng.random_range(0..3u32) {
                0 => {
                    let (i, j) = next_pair(&mut pair_queue, n, rng);
                    emit_rg1_x(&mut state, i, j, &mut out);
                }
                1 => {
                    let (i, j) = next_pair(&mut pair_queue, n, rng);
                    emit_rg2_x(&mut state, i, j, &mut out);
                }
                _ => {
                    let i = next_single(&mut single_queue, n, rng);
                    let (s, p) = state.pairs[i];
                    let random_carrier = random_wire_except(carrier_total, &[s, p], rng);
                    emit_rg3_x(&state, i, random_carrier, &mut out);
                }
            }
        }
    }
    prod_ledger.strip_all(&state, rng, &mut out);
    prod_ledger.report();
    // Filling only the band's final locations would reveal where rolling left
    // it. Every non-output wire is junk after decode and is therefore a safe,
    // deliberately uninformative mirror-fill target.
    let band_final: Vec<u16> = (n..total).map(|wire| wire as u16).collect();

    for wire in 0..total {
        on[wire] = Slot::Output(usize::MAX);
    }
    for value in 0..n {
        on[state.pairs[value].0] = Slot::Pair(value);
        on[state.pairs[value].1] = Slot::Pair(value);
    }
    let mut finalized = vec![false; total];
    for value in 0..n {
        let (share, pad) = state.pairs[value];
        if share == value {
            emit_transvection_cnot(value, pad, &mut out);
        } else if pad == value {
            emit_transvection_cnot(value, share, &mut out);
        } else {
            let borrowed = (0..carrier_total)
                .find(|&wire| !finalized[wire] && wire != value && wire != share && wire != pad)
                .unwrap();
            let moved_value = on[value];
            let moved_borrowed = on[borrowed];
            emit_w_i_inv_cnot(value, borrowed, share, pad, &mut out);
            reloc(
                moved_value,
                value,
                share,
                &mut dloc,
                &mut aloc,
                &mut state.pairs,
            );
            reloc(
                moved_borrowed,
                borrowed,
                pad,
                &mut dloc,
                &mut aloc,
                &mut state.pairs,
            );
            on[share] = moved_value;
            on[pad] = moved_borrowed;
        }
        finalized[value] = true;
        on[value] = Slot::Output(value);
    }
    out.extend(rand_z_xgates(n, bookend_size, rng));
    if prod.enabled() {
        if prod.fill_nl > 0 {
            emit_band_fill_nl(n, &band_final, prod.fill_nl, rng, &mut out);
        } else {
            emit_band_fill(n, &band_final, rng, &mut out);
        }
    }
    if prod.enabled() {
        commuting_shuffle(&mut out, rng);
    }
    CnotCircuit {
        gates: out,
        num_wires: total,
    }
}

/// Default production budget for the zero-slice preblock.
pub const SLICE_ZERO_CCNOT_GATES_PER_WIRE: usize = 10;

/// Positive-polarity zero-slice preblock over
/// `data 0..n | aux n..2n | band 2n..2n+band`.
///
/// Every gate targets data and reads exactly one non-data wire. Consequently
/// the block is gate-by-gate inert only when every aux and band input is zero.
/// Draws are rejected when an exhaustive (small widths) or sampled (large
/// widths) check finds a nonzero slice on which the data map is the identity.
pub fn slice_zero_ccnot_preblock(
    n: usize,
    band: usize,
    gate_count: usize,
    rng: &mut impl Rng,
) -> CnotCircuit {
    // Two-carrier build: the slice is the aux half plus the band.
    slice_zero_preblock_dims(n, n + band, gate_count, rng)
}

/// The same preblock, parameterised by how wide the slice actually IS.
///
/// A single-carrier gadget has no aux half, so its slice is the band alone and
/// its total is `n + band` rather than `2n + band`. Everything below was
/// already written in terms of `(data n, nondata, total)`; only the dimensions
/// differ, so this is a pure extraction — [`slice_zero_ccnot_preblock`] passes
/// `nondata = n + band` and draws exactly the stream it drew before.
fn slice_zero_preblock_dims(
    n: usize,
    nondata: usize,
    gate_count: usize,
    rng: &mut impl Rng,
) -> CnotCircuit {
    assert!(n >= 3, "slice_zero_ccnot_preblock requires n >= 3");
    let total = n + nondata;
    assert!(total <= u16::MAX as usize, "too many wires");
    let band = nondata.saturating_sub(n);
    assert!(
        gate_count >= nondata,
        "every non-data wire must be read: needs at least {nondata} gates"
    );

    // One third CNOTs; split the remainder between one- and two-data-control
    // gates so off-slice behavior is nonlinear in the data.
    let cnots = gate_count / 3;
    let remainder = gate_count - cnots;
    let three_control = remainder / 2;
    let two_control = remainder - three_control;

    for _ in 0..1_000 {
        // Balanced coverage prevents an unread slice wire from being a
        // trivially fixed nonzero slice.
        let mut slice_controls: Vec<usize> =
            (0..gate_count).map(|index| n + index % nondata).collect();
        slice_controls.shuffle(rng);
        let mut gates = Vec::with_capacity(gate_count);
        for (index, &slice_wire) in slice_controls.iter().enumerate() {
            let target = rng.random_range(0..n);
            let data_controls = if index < cnots {
                0
            } else if index < cnots + two_control {
                1
            } else {
                2
            };
            let mut literals = Vec::with_capacity(data_controls + 1);
            let mut taken = vec![target];
            for _ in 0..data_controls {
                let control = random_wire_except(n, &taken, rng);
                taken.push(control);
                literals.push((control as u16, true));
            }
            literals.push((slice_wire as u16, true));
            gates.push(
                XGate::conj(target as u16, literals).expect("slice preblock wires are distinct"),
            );
        }
        gates.shuffle(rng);
        let valid = if total <= 20 {
            slice_preblock_fixes_only_zero_slice(&gates, n, nondata)
        } else {
            slice_preblock_spot_check(&gates, n, total, rng)
        };
        if valid {
            return CnotCircuit {
                gates,
                num_wires: total,
            };
        }
    }
    panic!(
        "no slice preblock with every checked nonzero slice disturbed found at \
         n={n} band={band} gates={gate_count} in 1000 draws"
    );
}

fn slice_preblock_fixes_only_zero_slice(gates: &[XGate], n: usize, nondata: usize) -> bool {
    let data_mask = (1u64 << n) - 1;
    (1..(1u64 << nondata)).all(|slice| {
        (0..=data_mask).any(|data| {
            crate::postmix::xgate::eval_u64(gates, data | (slice << n)) & data_mask != data
        })
    })
}

fn slice_preblock_spot_check(gates: &[XGate], n: usize, total: usize, rng: &mut impl Rng) -> bool {
    let disturbs = |hot: &[usize], rng: &mut dyn RngCore| {
        let mut state = vec![0u64; total];
        for data in state.iter_mut().take(n) {
            *data = rng.next_u64();
        }
        let input = state[..n].to_vec();
        for &wire in hot {
            state[wire] = !0u64;
        }
        for gate in gates {
            gate.apply_lanes(&mut state);
        }
        (0..n).any(|wire| state[wire] != input[wire])
    };

    for wire in n..total {
        if !disturbs(&[wire], rng) {
            return false;
        }
    }
    for _ in 0..512 {
        let first = rng.random_range(n..total);
        let second = loop {
            let wire = rng.random_range(n..total);
            if wire != first {
                break wire;
            }
        };
        if !disturbs(&[first, second], rng) {
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

/// Two-share gadgetization of a legacy G57 source.  This public entry point
/// keeps its historical API while delegating to the heterogeneous core used
/// by the 4n TDP construction.
pub fn gadgetize_cnot(
    main: &CircuitSeq,
    n: usize,
    rg_freq: usize,
    rng: &mut impl Rng,
) -> CnotCircuit {
    assert!(n >= 3, "gadgetize_cnot requires n >= 3");
    let mut source = main.clone();
    let rounds = source.gates.len();
    shoot_random_gate_with_rng(&mut source, rounds, rng);
    let gates: Vec<XGate> = source.gates.iter().copied().map(XGate::from_g57).collect();
    gadgetize_xgates(&gates, n, rg_freq, rng)
}

/// Swap the contents of two wires with the three-CNOT network.
fn emit_wire_swap(a: usize, b: usize, out: &mut Vec<XGate>) {
    for (target, source) in [(a, b), (b, a), (a, b)] {
        out.push(XGate::cnot(target as u16, source as u16));
    }
}

/// Relocate one value to a different wire — the single-carrier RG.
///
/// It is RG1's move without the pair: a content swap plus a re-point. With one
/// carrier per value there is no RG2 (re-pair) and no RG3 (refresh both
/// carriers) to fall back on, so this plus mask re-sourcing is the whole
/// re-randomisation layer, which is why the entry points below insist on
/// `rsrc >= 1`.
fn emit_value_relocation(
    state: &mut GadgetState,
    carrier_total: usize,
    out: &mut Vec<XGate>,
    rng: &mut impl Rng,
) {
    let i = rng.random_range(0..state.n);
    let j = loop {
        let j = rng.random_range(0..state.n);
        if j != i {
            break j;
        }
    };
    // No carrier-space assumption: once the band rolls, a value can sit on a
    // former band wire, and relocation has to move it from wherever it is.
    let (wi, wj) = (state.pairs[i].0, state.pairs[j].0);
    let _ = carrier_total;
    emit_wire_swap(wi, wj, out);
    state.pairs[i] = (wj, wj);
    state.pairs[j] = (wi, wi);
}

/// Shared guard for the single-carrier entry points.
///
/// `epoch` and `fill_pivots` are DECLARED BUT INERT in this tree (see their
/// `ProdConfig` field docs). The reference's single-carrier builders honor
/// both, so accepting them here and silently doing nothing would ship a gadget
/// materially different from the one the config names. Refuse instead — the
/// presets this tree drives from set both to 0.
fn assert_single_carrier_config(prod: &ProdConfig, who: &str) {
    assert!(
        prod.single_carrier(),
        "{who} needs --prod-single with a nonempty mask plan"
    );
    assert!(
        prod.src_dist == 0,
        "distributed sourcing and single-carrier are not combined yet"
    );
    // With one carrier there is no RG3 to refresh the representation: a
    // re-source is the ONLY move that changes a carrier's bit. rsrc = 0 leaves
    // relocation as the sole churn, which moves values without re-randomising
    // them — documented as load-bearing, so enforce it rather than trust it.
    assert!(
        prod.rsrc >= 1,
        "--prod-single needs --prod-rsrc >= 1: with a single carrier, mask \
         re-sourcing is the only representation refresh (R2/R3 have no analogue)"
    );
    assert!(
        prod.epoch == 0,
        "--prod-epoch (retire-and-refill) has no honoring code in this tree; \
         the reference's single-carrier builder fires it, so accepting it here \
         would claim a band turnover that never happens"
    );
    assert!(
        prod.fill_pivots == 0,
        "--prod-fill-pivots (reserved pivot block) has no honoring code in this \
         tree; the reference's single-carrier builder passes it to the band fill"
    );
}

/// Fill the source band, then hand back its home wires. Shared by the two
/// single-carrier builders and their mirror fill.
fn emit_single_band_fill(
    n: usize,
    band: &[u16],
    prod: &ProdConfig,
    rng: &mut impl Rng,
    out: &mut Vec<XGate>,
) {
    if prod.fill_nl > 0 {
        emit_band_fill_nl(n, band, prod.fill_nl, rng, out);
    } else {
        emit_band_fill(n, band, rng, out);
    }
}

/// Route every value back to its own index.
///
/// Relocations and rolls permute the values across the WHOLE wire space — after
/// a roll a value can sit on a former band wire — so this is a full cycle
/// resolution, not a carrier-space one. Afterwards wires `0..n` hold the values
/// and `n..total` hold band junk, which is what makes the mirror fill safe.
fn route_single_carriers_home(state: &mut GadgetState, total: usize, out: &mut Vec<XGate>) {
    let n = state.n;
    let mut owner: Vec<Option<usize>> = vec![None; total];
    for value in 0..n {
        owner[state.pairs[value].0] = Some(value);
    }
    for value in 0..n {
        let cur = state.pairs[value].0;
        if cur == value {
            continue;
        }
        emit_wire_swap(cur, value, out);
        let displaced = owner[value];
        owner[cur] = displaced;
        if let Some(u) = displaced {
            state.pairs[u] = (cur, cur);
        }
        owner[value] = Some(value);
        state.pairs[value] = (value, value);
    }
}

/// SINGLE-CARRIER gadgetization of a legacy g57 source: `n` carriers plus the
/// band, against the two-carrier build's `2n` plus the band.
///
/// Production uses one carrier and four nonlinear masks with degrees
/// [2,2,2,3]. This halves the carrier count, while the Gray fold keeps the
/// additional mask term affordable and emits only narrow fragments.
/// Ported from the reference implementation's `gadgetize_cnot_single`
/// (`f8afe640`, `src/replace/gadgets.rs:6283`). Divergences, both forced by
/// this tree: the source reorder uses `shoot_random_gate_with_rng` (the
/// reference's global-`fastrand` `shoot_random_gate` would make the build
/// non-reproducible under a caller-supplied seed, and every other nh entry
/// point already uses the rng-threaded form), and the band fill is
/// `emit_band_fill_nl` because the reserved-pivot draw was not ported.
#[cfg(test)]
fn gadgetize_cnot_single(
    main: &CircuitSeq,
    n: usize,
    rg_freq: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    assert!(n >= 3, "gadgetize_cnot_single requires n >= 3");
    assert_single_carrier_config(prod, "gadgetize_cnot_single");

    let mut main = main.clone();
    let rounds = main.gates.len();
    shoot_random_gate_with_rng(&mut main, rounds, rng);
    let gates: Vec<XGate> = main.gates.iter().copied().map(XGate::from_g57).collect();
    gadgetize_xgates_single(&gates, n, rg_freq, prod, rng)
}

/// SINGLE-CARRIER gadgetization of an already-heterogeneous mpmct1 source.
///
/// Ported from the reference implementation's `gadgetize_xgates_single`
/// (`f8afe640`, `src/replace/gadgets.rs:7359`).
fn gadgetize_xgates_single(
    source: &[XGate],
    n: usize,
    rg_freq: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    assert!(n >= 3, "gadgetize_xgates_single requires n >= 3");
    assert_single_carrier_config(prod, "gadgetize_xgates_single");
    assert!(
        source.iter().all(|gate| {
            (gate.target as usize) < n && gate.ctrls.iter().all(|&(wire, _)| (wire as usize) < n)
        }),
        "source wire outside 0..n"
    );

    let carrier_total = n;
    let band_len = prod.band_size(n);
    let total = carrier_total + band_len;
    assert!(total <= u16::MAX as usize, "too many wires");

    // The band is filled from the data wires while they still hold x, and is
    // not written again by the ordinary body path — the same source contract as
    // the two-carrier build, and the reason the strip cancels exactly under
    // arbitrary junk.
    let mut out: Vec<XGate> = Vec::new();
    let band_home: Vec<u16> = (carrier_total..total).map(|wire| wire as u16).collect();
    emit_single_band_fill(n, &band_home, prod, rng, &mut out);

    // Value v lives on wire v; `pairs` records only WHERE, with both entries
    // equal so every carrier lookup returns the one wire.
    let mut state = GadgetState {
        n,
        pairs: (0..n).map(|wire| (wire, wire)).collect(),
    };
    let mut prod_ledger = ProdLedger::new(n, prod, carrier_total);
    prod_ledger.inject_all(&state, rng, &mut out);

    for (index, gate) in source.iter().enumerate() {
        prod_ledger.fold_cg(gate, &state, rng, &mut out);
        if index + 1 == source.len() {
            break;
        }
        // The only surviving RG: relocate a value to another wire.
        for _ in 0..rg_freq {
            emit_value_relocation(&mut state, carrier_total, &mut out, rng);
        }
        for _ in 0..prod.rsrc {
            prod_ledger.resource(&state, rng, &mut out);
        }
        // Rolling the band across the carrier/band boundary matters MORE here
        // than in the paired build, not less: the band is a larger share of a
        // narrower gadget, and with one carrier per value each carrier absorbs
        // every fold write, so an unwritten band wire stands out against a
        // sharper contrast.
        for _ in 0..prod.roll {
            prod_ledger.roll(&mut state, rng, &mut out);
        }
    }
    prod_ledger.strip_all(&state, rng, &mut out);
    prod_ledger.report();

    route_single_carriers_home(&mut state, total, &mut out);

    // Mirror fill so the band is junk at both ports, as in the paired build.
    let band_final: Vec<u16> = (carrier_total..total).map(|wire| wire as u16).collect();
    emit_single_band_fill(n, &band_final, prod, rng, &mut out);

    commuting_shuffle(&mut out, rng);
    CnotCircuit {
        gates: out,
        num_wires: total,
    }
}

/// Keep a single-carrier gadget's product band above a reserved helper block.
/// Carrier wires stay in `0..carrier_wires`; every band wire is shifted by the
/// same amount, so gate semantics and control ordering are unchanged.
fn reserve_before_single_band(
    mut circuit: CnotCircuit,
    carrier_wires: usize,
    reserved_wires: usize,
) -> CnotCircuit {
    assert!(carrier_wires <= circuit.num_wires);
    let shifted_total = circuit
        .num_wires
        .checked_add(reserved_wires)
        .expect("padded single-carrier wire count overflow");
    assert!(shifted_total <= u16::MAX as usize, "too many padded wires");

    let shift = |wire: &mut u16| {
        if (*wire as usize) >= carrier_wires {
            *wire = (*wire as usize + reserved_wires) as u16;
        }
    };
    for gate in &mut circuit.gates {
        shift(&mut gate.target);
        for (wire, _) in &mut gate.ctrls {
            shift(wire);
        }
    }
    circuit.num_wires = shifted_total;
    circuit
}

/// Single-carrier gadget behind the zero-slice preblock. The slice is the band
/// ALONE — there is no aux half to pin — so the preblock is built over
/// `(data n, slice band)` instead of `(data n, slice n+band)`.
///
/// Ported from the reference implementation's
/// `gadgetize_xgates_with_slice_zero_ccnot_single` (`f8afe640`,
/// `src/replace/gadgets.rs:7487`).
fn gadgetize_xgates_with_slice_zero_ccnot_single(
    source: &[XGate],
    n: usize,
    rg_freq: usize,
    gate_count: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let band = prod.band_size(n);
    let mut circuit = slice_zero_preblock_dims(n, band, gate_count.max(band), rng);
    let gadget = gadgetize_xgates_single(source, n, rg_freq, prod, rng);
    circuit.num_wires = circuit.num_wires.max(gadget.num_wires);
    circuit.gates.extend(gadget.gates);
    commuting_shuffle(&mut circuit.gates, rng);
    circuit
}

/// Bare product-share nonlinear gadgetization using the sole production
/// `[2,2,2,3]` single-carrier Gray plan. `rg_draws` is the number of nonlinear RG draws
/// between consecutive logical source gates.
///
/// This core retains the stronger historical endpoint contract: low outputs
/// are correct for arbitrary non-data junk. The SSS production shortcut uses
/// [`nonlinear_gadgetize_with_slice_zero_cnot`] instead.
pub fn nonlinear_gadgetize_cnot(
    main: &CircuitSeq,
    n: usize,
    rg_draws: usize,
    rng: &mut impl Rng,
) -> CnotCircuit {
    nonlinear_gadgetize_cnot_with_config(main, n, rg_draws, &ProdConfig::production(), rng)
}

/// Configurable form used by focused tests and experiment code. SSS exposes
/// the production preset through `--nonlinear_gadgetize`.
fn nonlinear_gadgetize_cnot_with_config(
    main: &CircuitSeq,
    n: usize,
    rg_draws: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    assert!(
        prod.enabled(),
        "nonlinear gadgetization requires at least one product mask term"
    );
    let mut source = main.clone();
    let rounds = source.gates.len();
    shoot_random_gate_with_rng(&mut source, rounds, rng);
    let gates: Vec<XGate> = source.gates.iter().copied().map(XGate::from_g57).collect();
    if prod.single_carrier() {
        gadgetize_xgates_single(&gates, n, rg_draws, prod, rng)
    } else {
        gadgetize_xgates_with_prod(&gates, n, rg_draws, prod, rng)
    }
}

/// Production `[2,2,2,3]` single-carrier Gray gadgetization with nonlinear
/// fill and a rolling band, behind a band-only zero-slice preblock.
pub fn nonlinear_gadgetize_with_slice_zero_cnot(
    main: &CircuitSeq,
    n: usize,
    rg_draws: usize,
    rng: &mut impl Rng,
) -> CnotCircuit {
    nonlinear_gadgetize_with_slice_zero_cnot_with_config(
        main,
        n,
        rg_draws,
        &ProdConfig::production(),
        rng,
    )
}

/// Private configurable form retained for primitive regression tests.
/// Public callers always receive the canonical production configuration.
fn nonlinear_gadgetize_with_slice_zero_cnot_with_config(
    main: &CircuitSeq,
    n: usize,
    rg_draws: usize,
    config: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    assert!(
        config.enabled(),
        "nonlinear gadgetization requires at least one product mask term"
    );
    let gate_count = SLICE_ZERO_CCNOT_GATES_PER_WIRE * n;
    if config.single_carrier() {
        let mut source = main.clone();
        let rounds = source.gates.len();
        shoot_random_gate_with_rng(&mut source, rounds, rng);
        let gates: Vec<XGate> = source.gates.iter().copied().map(XGate::from_g57).collect();
        return gadgetize_xgates_with_slice_zero_ccnot_single(
            &gates, n, rg_draws, gate_count, config, rng,
        );
    }
    let mut circuit = slice_zero_ccnot_preblock(n, config.band_size(n), gate_count, rng);
    let gadget = nonlinear_gadgetize_cnot_with_config(main, n, rg_draws, config, rng);
    circuit.num_wires = circuit.num_wires.max(gadget.num_wires);
    circuit.gates.extend(gadget.gates);
    commuting_shuffle(&mut circuit.gates, rng);
    circuit
}

/// Gadgetize an already-heterogeneous logical circuit with the sole production
/// `[2,2,2,3]` single-carrier Gray construction behind a band-only zero slice.
pub fn gadgetize_xgates_with_slice_zero_ccnot(
    source: &[XGate],
    n: usize,
    rg_freq: usize,
    gate_count: usize,
    rng: &mut impl Rng,
) -> CnotCircuit {
    gadgetize_xgates_with_slice_zero_ccnot_with_config(
        source,
        n,
        rg_freq,
        gate_count,
        &ProdConfig::production(),
        rng,
    )
}

fn gadgetize_xgates_with_slice_zero_ccnot_with_config(
    source: &[XGate],
    n: usize,
    rg_freq: usize,
    gate_count: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    if prod.single_carrier() {
        return gadgetize_xgates_with_slice_zero_ccnot_single(
            source, n, rg_freq, gate_count, prod, rng,
        );
    }
    let mut circuit = slice_zero_ccnot_preblock(n, prod.band_size(n), gate_count, rng);
    let gadget = gadgetize_xgates_with_prod(source, n, rg_freq, prod, rng);
    circuit.num_wires = circuit.num_wires.max(gadget.num_wires);
    circuit.gates.extend(gadget.gates);
    // Shuffle across the preblock/gadget seam so the outer slice block does
    // not remain a contiguous prefix.
    commuting_shuffle(&mut circuit.gates, rng);
    circuit
}

fn slice_zero_preblock_cnot(n: usize, rng: &mut impl Rng) -> CnotCircuit {
    let total = 3 * n;
    let matrix = random_invertible_matrix_rows(n, rng);
    let mut gates = Vec::new();
    for col in 0..n {
        let y = n + col;
        let z = 2 * n + col;
        gates.push(XGate::x_gate(z as u16));
        for row in 0..n {
            if matrix_bit(&matrix, row, col) {
                gates.push(XGate::from_g57([row as u16, y as u16, z as u16]));
            }
        }
        gates.push(XGate::x_gate(z as u16));
    }
    CnotCircuit {
        gates,
        num_wires: total,
    }
}

fn slice_zero_hardcoded_preblock_cnot(n: usize, rounds: usize, rng: &mut impl Rng) -> CnotCircuit {
    let total = 3 * n;
    let mut gates = Vec::new();
    let mut order: Vec<usize> = (0..n).collect();
    let mut emit = |target: usize, aux: usize, other: usize| {
        gates.push(
            XGate::conj(target as u16, [(aux as u16, true), (other as u16, true)])
                .expect("hardcoded slice fragment"),
        );
    };
    for _ in 0..rounds {
        order.shuffle(rng);
        for &i in &order {
            let aux = if rng.random_bool(0.5) {
                n + rng.random_range(0..n)
            } else {
                2 * n + rng.random_range(0..n)
            };
            let other = random_wire_except(total, &[i, aux], rng);
            emit(i, aux, other);
        }
        order.shuffle(rng);
        for &i in &order {
            let target = n + i;
            let aux = 2 * n + rng.random_range(0..n);
            let other = random_wire_except(total, &[target, aux], rng);
            emit(target, aux, other);
        }
        order.shuffle(rng);
        for &i in &order {
            let target = 2 * n + i;
            let aux = n + rng.random_range(0..n);
            let other = random_wire_except(total, &[target, aux], rng);
            emit(target, aux, other);
        }
    }
    CnotCircuit {
        gates,
        num_wires: total,
    }
}

fn random_fragment_on_x(n: usize, rng: &mut impl Rng) -> XGate {
    let total = 3 * n;
    let target = rng.random_range(0..n) as u16;
    let width = match rng.random_range(0..10u32) {
        0 => 0,
        1..=5 => 1,
        6..=8 => 2,
        _ => 3,
    };
    let mut controls = Vec::with_capacity(width);
    while controls.len() < width {
        let wire = rng.random_range(0..total) as u16;
        if wire != target && !controls.iter().any(|&(w, _)| w == wire) {
            controls.push((wire, rng.random::<bool>()));
        }
    }
    XGate::conj(target, controls).expect("random fragment controls are distinct")
}

/// Public-slice preblock `R^-1 B R`. `B` pairs all 2n public-slice bits and
/// toggles one distinct x bit by the OR of the pair's two deviation bits. The
/// OR is one complemented conjunction fragment per pair (one pair is written
/// as two pure fragments when needed to hit an odd requested gate count).
/// Thus B, and therefore its conjugate, is the identity exactly on one slice.
pub fn slice_zero_random_preblock_cnot(
    n: usize,
    gate_count: usize,
    rng: &mut impl Rng,
) -> CnotSliceZeroRandomCircuit {
    assert!(n >= 3, "slice_zero_random_preblock_cnot requires n >= 3");
    assert!(
        gate_count >= n,
        "a uniquely fixed public slice needs at least n fragment gates"
    );
    let total = 3 * n;
    let (public_y, public_z) = random_public_slice(n, rng);
    let extra = gate_count - n;
    let split_one_or = extra % 2;
    let flank_len = (extra - split_one_or) / 2;
    let flank: Vec<XGate> = (0..flank_len)
        .map(|_| random_fragment_on_x(n, rng))
        .collect();
    let mut gates = flank.clone();

    let mut aux_wires: Vec<usize> = (n..total).collect();
    aux_wires.shuffle(rng);
    let mut targets: Vec<usize> = (0..n).collect();
    targets.shuffle(rng);
    let public_value = |wire: usize| {
        if wire < 2 * n {
            packed_bit(&public_y, wire - n)
        } else {
            packed_bit(&public_z, wire - 2 * n)
        }
    };
    for (index, (target, pair)) in targets
        .into_iter()
        .zip(aux_wires.chunks_exact(2))
        .enumerate()
    {
        let a = pair[0];
        let b = pair[1];
        // A deviation literal is true iff the public-slice bit is violated.
        let da = (a as u16, !public_value(a));
        let db = (b as u16, !public_value(b));
        if index == 0 && split_one_or == 1 {
            // da OR db = da XOR (!da AND db).
            gates.push(XGate::conj(target as u16, [da]).unwrap());
            gates.push(XGate::conj(target as u16, [(a as u16, public_value(a)), db]).unwrap());
        } else {
            // da OR db = 1 XOR (!da AND !db): fire unless both bits equal
            // their public values.
            let mut or_gate = XGate::conj(
                target as u16,
                [(a as u16, public_value(a)), (b as u16, public_value(b))],
            )
            .unwrap();
            or_gate.comp = true;
            gates.push(or_gate);
        }
    }
    gates.extend(flank.into_iter().rev());
    debug_assert_eq!(gates.len(), gate_count);
    CnotSliceZeroRandomCircuit {
        circuit: CnotCircuit {
            gates,
            num_wires: total,
        },
        public_y,
        public_z,
    }
}

fn rand_feistal_z_xgates(n: usize, m: usize, rng: &mut impl Rng) -> Vec<XGate> {
    rand_feistal_z_gates(n, m, rng)
        .into_iter()
        .map(XGate::from_g57)
        .collect()
}

fn emit_sg3_x(state: &FeistalState, gate: [u16; 3], out: &mut Vec<XGate>) {
    // Retain the nine-G57 3-share SG. A smaller mixed candidate is first-order
    // masked but temporarily reduces a control to two carriers, introducing
    // same-prefix two-probe leakage that this target-only legacy SG avoids.
    let mut legacy = Vec::with_capacity(9);
    emit_sg3(state, gate, &mut legacy);
    out.extend(legacy.into_iter().map(XGate::from_g57));
}

fn emit_feistal_rg_x(
    state: &mut FeistalState,
    pq: &mut VecDeque<(usize, usize)>,
    sq: &mut VecDeque<usize>,
    rng: &mut impl Rng,
    out: &mut Vec<XGate>,
) {
    let n = state.sharing.n;
    match rng.random_range(0..3u32) {
        0 => {
            let (i, j) = next_pair(pq, n, rng);
            emit_rg1_x(&mut state.sharing, i, j, out);
        }
        1 => {
            let (i, j) = next_pair(pq, n, rng);
            emit_rg2_x(&mut state.sharing, i, j, out);
        }
        _ => {
            let i = next_single(sq, n, rng);
            let (p0, p1) = state.sharing.pairs[i];
            // Using the third carrier of this same logical value as the
            // refresh carrier creates a same-prefix two-probe leak in RG3.
            // Keep it disjoint from all three carriers.
            let free = state.free[i];
            let random_carrier = random_wire_except(3 * n, &[p0, p1, free], rng);
            emit_rg3_x(&state.sharing, i, random_carrier, out);
        }
    }
}

fn emit_sg3_rg_block_x(
    circuit: &CircuitSeq,
    state: &mut FeistalState,
    rg_freq: usize,
    rng: &mut impl Rng,
    out: &mut Vec<XGate>,
) {
    let mut pq = VecDeque::new();
    let mut sq = VecDeque::new();
    for (index, &gate) in circuit.gates.iter().enumerate() {
        emit_sg3_x(state, gate, out);
        if (index + 1) % rg_freq == 0 {
            emit_feistal_rg_x(state, &mut pq, &mut sq, rng, out);
        }
    }
}

/// Random logical computation used for Feistel D. Most gates are CNOTs; the
/// remaining gates are small positive/mixed-polarity conjunction fragments.
fn random_feistal_d_xgates(n: usize, m: usize, rng: &mut impl Rng) -> Vec<XGate> {
    let mut gates = Vec::with_capacity(m);
    for _ in 0..m {
        let target = rng.random_range(0..n) as u16;
        let width = match rng.random_range(0..20u32) {
            0..=13 => 1,
            14..=18 => 2,
            _ => 3,
        }
        .min(n - 1);
        let mut controls = Vec::with_capacity(width);
        while controls.len() < width {
            let wire = rng.random_range(0..n) as u16;
            if wire != target && !controls.iter().any(|&(w, _)| w == wire) {
                // Positive controls keep the shared expansion especially lean,
                // while occasional negative literals retain fragment variety.
                controls.push((wire, !rng.random_bool(0.25)));
            }
        }
        gates.push(XGate::conj(target, controls).expect("valid random D fragment"));
    }
    gates
}

pub const SANDWICH_SLICE_GATES_PER_WIRE_LOG: f64 = 1.0;
pub const SANDWICH_D_GATES_PER_WIRE_LOG2: f64 = 1.0;

/// Default sliced-sandwich block size `s = round(n log2(n))`, floored at n.
pub fn sandwich_default_s(n: usize) -> usize {
    ((n as f64) * (n as f64).log2()).round().max(n as f64) as usize
}

/// Default random-D size `m = round(n log2(n)^2)`, floored at n.
pub fn sandwich_default_m(n: usize) -> usize {
    let log_n = (n as f64).log2();
    ((n as f64) * log_n * log_n).round().max(n as f64) as usize
}

/// One sliced-sandwich guard block. Every gate targets the first n wires and
/// reads at least one positive-polarity wire from the second half, so the
/// complete block is gate-by-gate dead when the second half is zero.
fn sandwich_slice_gates(n: usize, s: usize, rng: &mut impl Rng) -> Vec<XGate> {
    (0..s)
        .map(|_| {
            let target = rng.random_range(0..n);
            if rng.random_bool(1.0 / 3.0) {
                let control = n + rng.random_range(0..n);
                XGate::cnot(target as u16, control as u16)
            } else {
                let first_control = random_wire_except(n, &[target], rng);
                let second_control = n + rng.random_range(0..n);
                XGate::conj(
                    target as u16,
                    [(first_control as u16, true), (second_control as u16, true)],
                )
                .expect("sandwich CCNOT controls are distinct")
            }
        })
        .collect()
}

/// `m` random G57 gates over wires 0..n, represented as heterogeneous gates.
fn random_g57_xgates(n: usize, m: usize, rng: &mut impl Rng) -> Vec<XGate> {
    assert!(n >= 3, "random G57 gates need n >= 3 wires");
    (0..m)
        .map(|_| {
            let target = rng.random_range(0..n);
            let positive = random_wire_except(n, &[target], rng);
            let negative = random_wire_except(n, &[target, positive], rng);
            XGate::from_g57([target as u16, positive as u16, negative as u16])
        })
        .collect()
}

/// Uniformly interleave two ordered gate streams while preserving each
/// stream's internal order.
fn random_interleave(computation: Vec<XGate>, slice: Vec<XGate>, rng: &mut impl Rng) -> Vec<XGate> {
    let mut output = Vec::with_capacity(computation.len() + slice.len());
    let (mut computation_index, mut slice_index) = (0usize, 0usize);
    while computation_index < computation.len() || slice_index < slice.len() {
        let remaining_computation = computation.len() - computation_index;
        let remaining_slice = slice.len() - slice_index;
        let take_computation = slice_index >= slice.len()
            || (computation_index < computation.len()
                && rng.random_range(0..remaining_computation + remaining_slice)
                    < remaining_computation);
        if take_computation {
            output.push(computation[computation_index].clone());
            computation_index += 1;
        } else {
            output.push(slice[slice_index].clone());
            slice_index += 1;
        }
    }
    output
}

/// Float one gate in a registered direction across adjacent commuting gates.
fn float_extremal(gates: &mut [XGate], mut position: usize, left: bool) -> usize {
    if left {
        while position > 0 && !XGate::collides(&gates[position], &gates[position - 1]) {
            gates.swap(position, position - 1);
            position -= 1;
        }
    } else {
        while position + 1 < gates.len() && !XGate::collides(&gates[position], &gates[position + 1])
        {
            gates.swap(position, position + 1);
            position += 1;
        }
    }
    position
}

/// Build the sliced sandwich on 2n wires:
///
/// `[C interleaved with S1] ; N ; [D interleaved with S2]`.
///
/// The N step copies `x` into the second half (`y ^= x`). S1 and S2 are
/// independently sampled guard blocks, and the N gates are floated in fixed
/// random directions to dissolve the central column.
///
/// On the inner zero slice:
///
/// `A(x, 0) = (junk, C(x))`.
pub fn sliced_sandwich_cnot(
    main: &CircuitSeq,
    n: usize,
    m: usize,
    s: usize,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let d_gates = random_g57_xgates(n, m, rng);
    sliced_sandwich_with_d(main, &d_gates, n, s, rng)
}

/// Sliced sandwich with an explicit random-D stream.
pub fn sliced_sandwich_with_d(
    main: &CircuitSeq,
    d_gates: &[XGate],
    n: usize,
    s: usize,
    rng: &mut impl Rng,
) -> CnotCircuit {
    assert!(n >= 3, "sliced_sandwich_with_d requires n >= 3");
    assert!(2 * n <= u16::MAX as usize, "too many wires");
    assert!(
        main.gates.iter().flatten().all(|&wire| (wire as usize) < n),
        "source wire outside 0..n"
    );
    assert!(
        d_gates.iter().all(|gate| {
            (gate.target as usize) < n && gate.ctrls.iter().all(|&(wire, _)| (wire as usize) < n)
        }),
        "D wire outside 0..n"
    );

    let c_gates: Vec<XGate> = main.gates.iter().copied().map(XGate::from_g57).collect();
    let s1 = sandwich_slice_gates(n, s, rng);
    let mut output = random_interleave(c_gates, s1, rng);

    for wire in 0..n {
        output.push(XGate::cnot((n + wire) as u16, wire as u16));
    }

    let s2 = sandwich_slice_gates(n, s, rng);
    output.extend(random_interleave(d_gates.to_vec(), s2, rng));

    // The only gates targeting the second half are the n mutually commuting
    // N gates. Register one direction per gate, then float each to its
    // commutation extreme while maintaining the other registered positions.
    let mut floaters: Vec<(usize, bool)> = (0..output.len())
        .filter(|&index| (output[index].target as usize) >= n)
        .map(|index| (index, rng.random_bool(0.5)))
        .collect();
    floaters.shuffle(rng);
    for current in 0..floaters.len() {
        let (before, left) = floaters[current];
        let after = float_extremal(&mut output, before, left);
        floaters[current].0 = after;
        for (index, (position, _)) in floaters.iter_mut().enumerate() {
            if index == current {
                continue;
            }
            if after < before && *position >= after && *position < before {
                *position += 1;
            } else if after > before && *position > before && *position <= after {
                *position -= 1;
            }
        }
    }

    CnotCircuit {
        gates: output,
        num_wires: 2 * n,
    }
}

/// Build the ordinary 2n-wire TDP computation before masking:
///
///   (x, y) --C--> (C(x), y)
///          --N--> (C(x), y XOR C(x))
///          --D--> (D(C(x)), y XOR C(x)).
///
/// N is the bank of native CNOTs from wire i to wire n+i, and D is the same
/// random fragment computation used by the CNOT Feistal construction.
fn tdp2n_xgates(main: &CircuitSeq, n: usize, rng: &mut impl Rng) -> Vec<XGate> {
    let mut source = main.clone();
    let rounds = source.gates.len();
    shoot_random_gate_with_rng(&mut source, rounds, rng);

    let mut gates = Vec::with_capacity(2 * main.gates.len() + n);
    gates.extend(source.gates.into_iter().map(XGate::from_g57));
    gates.extend((0..n).map(|wire| XGate::cnot((n + wire) as u16, wire as u16)));
    gates.extend(random_feistal_d_xgates(n, main.gates.len(), rng));
    gates
}

/// Construct the ordinary TDP layout on 2n logical wires and then apply the
/// native two-share gadgetizer to all 2n values.  Physical input blocks are
/// X,Y,Z,W (n wires each); the decoded low blocks are
/// (D(C(x)), y XOR C(x)) and Z,W are randomized auxiliary outputs.
pub fn tdp4n_cnot(main: &CircuitSeq, n: usize, rg_freq: usize, rng: &mut impl Rng) -> CnotCircuit {
    assert!(n >= 3, "tdp4n_cnot requires n >= 3");
    assert!(4 * n <= u16::MAX as usize, "too many wires");
    assert!(rg_freq > 0, "rg_freq must be nonzero");
    assert!(
        main.gates.iter().flatten().all(|&wire| (wire as usize) < n),
        "input wire outside 0..n"
    );

    let logical = tdp2n_xgates(main, n, rng);
    gadgetize_xgates(&logical, 2 * n, rg_freq, rng)
}

/// Construct the ordinary 2n-logical-wire TDP and encode all 2n values with
/// the production nonlinear product-share gadgetizer.  The first 4n physical
/// wires retain the X,Y,Z,W carrier layout; a rolling nonlinear source band
/// is appended after W.
pub fn tdp4n_nonlinear_cnot(
    main: &CircuitSeq,
    n: usize,
    rg_draws: usize,
    rng: &mut impl Rng,
) -> CnotCircuit {
    tdp4n_nonlinear_cnot_with_config(main, n, rg_draws, &ProdConfig::production(), rng)
}

/// Configurable nonlinear TDP constructor used by focused research
/// experiments. Production callers should continue to use
/// [`tdp4n_nonlinear_cnot`].
fn tdp4n_nonlinear_cnot_with_config(
    main: &CircuitSeq,
    n: usize,
    rg_draws: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    assert!(n >= 3, "tdp4n_nonlinear_cnot requires n >= 3");
    assert!(rg_draws > 0, "rg_draws must be nonzero");
    assert!(
        prod.enabled(),
        "nonlinear TDP requires at least one product mask term"
    );
    assert!(
        main.gates.iter().flatten().all(|&wire| (wire as usize) < n),
        "input wire outside 0..n"
    );

    let logical = tdp2n_xgates(main, n, rng);
    if prod.single_carrier() {
        // Preserve the public X/Y/Z/W namespace used by the TDP fixed-slice
        // contract. X/Y are the 2n single carriers, Z/W are reserved helpers,
        // and the product band starts after 4n.
        let single = gadgetize_xgates_single(&logical, 2 * n, rg_draws, prod, rng);
        reserve_before_single_band(single, 2 * n, 2 * n)
    } else {
        gadgetize_xgates_with_prod(&logical, 2 * n, rg_draws, prod, rng)
    }
}

/// Fixed-public-slice 4n TDP construction.  The existing M preblock acts on
/// physical X,Y,Z and leaves W free.  At its public (Y,Z) slice M is exactly
/// the identity for every X and W, so the middle decoded block remains
/// y XOR C(x); away from that slice M disturbs X before the TDP computation.
pub fn tdp4n_with_slice_zero_random_cnot(
    main: &CircuitSeq,
    n: usize,
    rg_freq: usize,
    gate_count: usize,
    rng: &mut impl Rng,
) -> CnotSliceZeroRandomCircuit {
    let mut preblock = slice_zero_random_preblock_cnot(n, gate_count, rng);
    let tdp = tdp4n_cnot(main, n, rg_freq, rng);
    preblock.circuit.gates.extend(tdp.gates);
    preblock.circuit.num_wires = tdp.num_wires;
    preblock
}

/// Fixed-public-slice TDP with production nonlinear product-share
/// gadgetization.  Y and Z remain the fixed public slice blocks, W remains a
/// free n-wire helper block, and the appended product-source band is also
/// arbitrary helper input.
pub fn tdp4n_nonlinear_with_slice_zero_random_cnot(
    main: &CircuitSeq,
    n: usize,
    rg_draws: usize,
    gate_count: usize,
    rng: &mut impl Rng,
) -> CnotSliceZeroRandomCircuit {
    tdp4n_nonlinear_with_slice_zero_random_cnot_with_config(
        main,
        n,
        rg_draws,
        gate_count,
        &ProdConfig::production(),
        rng,
    )
}

/// Configurable fixed-public-slice nonlinear TDP constructor used by focused
/// research experiments. Production callers should continue to use
/// [`tdp4n_nonlinear_with_slice_zero_random_cnot`].
fn tdp4n_nonlinear_with_slice_zero_random_cnot_with_config(
    main: &CircuitSeq,
    n: usize,
    rg_draws: usize,
    gate_count: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotSliceZeroRandomCircuit {
    let mut preblock = slice_zero_random_preblock_cnot(n, gate_count, rng);
    let tdp = tdp4n_nonlinear_cnot_with_config(main, n, rg_draws, prod, rng);
    preblock.circuit.gates.extend(tdp.gates);
    preblock.circuit.num_wires = tdp.num_wires;
    preblock
}

fn toggle_anf_term(terms: &mut Vec<Vec<(u16, bool)>>, term: Vec<(u16, bool)>) {
    if let Some(index) = terms.iter().position(|present| *present == term) {
        terms.swap_remove(index);
    } else {
        terms.push(term);
    }
}

/// Homomorphically apply one logical fragment to the three-share Feistel
/// representation. A positive logical literal expands as p0+p1+free and a
/// negative one as 1+p0+p1+free. Every emitted physical gate still targets
/// only the free carrier, so no prefix reconstructs the logical target.
fn emit_shared_fragment3(state: &FeistalState, gate: &XGate, out: &mut Vec<XGate>) {
    let logical_target = gate.target as usize;
    debug_assert!(logical_target < state.sharing.n);

    if !gate.comp && gate.ctrls.len() == 1 && gate.ctrls[0].1 {
        let logical_control = gate.ctrls[0].0 as usize;
        let (target_p0, target_p1) = state.sharing.pairs[logical_target];
        let (control_p0, control_p1) = state.sharing.pairs[logical_control];
        out.extend(homomorphic_cnot3(
            (
                target_p0 as u16,
                target_p1 as u16,
                state.free[logical_target] as u16,
            ),
            (
                control_p0 as u16,
                control_p1 as u16,
                state.free[logical_control] as u16,
            ),
        ));
        return;
    }

    let mut terms: Vec<Vec<(u16, bool)>> = vec![Vec::new()];
    for &(logical_control, positive) in &gate.ctrls {
        let logical_control = logical_control as usize;
        debug_assert!(logical_control < state.sharing.n);
        debug_assert_ne!(logical_control, logical_target);
        let (p0, p1) = state.sharing.pairs[logical_control];
        let carriers = [p0, p1, state.free[logical_control]];
        let previous = std::mem::take(&mut terms);
        for term in previous {
            if !positive {
                toggle_anf_term(&mut terms, term.clone());
            }
            for carrier in carriers {
                let mut next = term.clone();
                next.push((carrier as u16, true));
                next.sort_unstable();
                toggle_anf_term(&mut terms, next);
            }
        }
    }
    if gate.comp {
        toggle_anf_term(&mut terms, Vec::new());
    }
    terms.sort_unstable();
    for term in terms {
        if let Some(fragment) = XGate::conj(state.free[logical_target] as u16, term) {
            out.push(fragment);
        }
    }
}

fn emit_fragment3_rg_block_x(
    circuit: &[XGate],
    state: &mut FeistalState,
    rg_freq: usize,
    rng: &mut impl Rng,
    out: &mut Vec<XGate>,
) {
    let mut pq = VecDeque::new();
    let mut sq = VecDeque::new();
    for (index, gate) in circuit.iter().enumerate() {
        emit_shared_fragment3(state, gate, out);
        if (index + 1) % rg_freq == 0 {
            emit_feistal_rg_x(state, &mut pq, &mut sq, rng, out);
        }
    }
}

fn emit_feistal_n_cnot(state: &FeistalState, out: &mut Vec<XGate>) {
    let n = state.sharing.n;
    let mut host_of_y = vec![0usize; n];
    for (host, &y) in state.q.iter().enumerate() {
        host_of_y[y] = host;
    }
    for x in 0..n {
        let host = host_of_y[x];
        let (yc, other_y_carrier) = state.sharing.pairs[host];
        let hf = state.free[host];
        if host == x {
            for (dst, src) in [
                (other_y_carrier, yc),
                (hf, yc),
                (other_y_carrier, hf),
                (hf, other_y_carrier),
                (other_y_carrier, hf),
            ] {
                emit_transvection_cnot(dst, src, out);
            }
            continue;
        }
        for source in [
            state.sharing.pairs[x].0,
            state.sharing.pairs[x].1,
            state.free[x],
        ] {
            emit_transvection_cnot(yc, source, out);
            emit_transvection_cnot(hf, source, out);
        }
    }
}

fn emit_feistal_decode_cnot(state: &FeistalState, out: &mut Vec<XGate>) {
    let n = state.sharing.n;
    let total = 3 * n;
    let mut rows = vec![vec![0u64; total.div_ceil(64)]; total];
    for i in 0..n {
        let (p0, p1) = state.sharing.pairs[i];
        for bit in [p0, p1, state.free[i]] {
            rows[i][bit / 64] |= 1 << (bit % 64);
        }
        for bit in [p0, p1] {
            rows[n + state.q[i]][bit / 64] |= 1 << (bit % 64);
        }
        rows[2 * n + i][p0 / 64] |= 1 << (p0 % 64);
    }
    let mut ops = Vec::new();
    for col in 0..total {
        let pivot = (col..total)
            .find(|&row| feistal_bit(&rows[row], col))
            .expect("invertible decode");
        if pivot != col {
            feistal_xor_row(&mut rows, col, pivot);
            ops.push((col, pivot));
            feistal_xor_row(&mut rows, pivot, col);
            ops.push((pivot, col));
            feistal_xor_row(&mut rows, col, pivot);
            ops.push((col, pivot));
        }
        for row in 0..total {
            if row != col && feistal_bit(&rows[row], col) {
                feistal_xor_row(&mut rows, row, col);
                ops.push((row, col));
            }
        }
    }
    for &(dst, src) in ops.iter().rev() {
        emit_transvection_cnot(dst, src, out);
    }
}

/// Feistelization with native CNOT linear layers, the stronger legacy
/// nine-G57 three-share SG, non-complete CNOT RG1/RG2/RG3 blocks,
/// and a fragment-based random D computation.
pub fn feistalize_cnot(
    main: &CircuitSeq,
    n: usize,
    rg_freq: usize,
    rng: &mut impl Rng,
) -> CnotCircuit {
    assert!(n >= 3, "feistalize_cnot requires n >= 3");
    assert!(3 * n <= u16::MAX as usize, "too many wires");
    assert!(rg_freq > 0, "rg_freq must be nonzero");
    assert!(
        main.gates.iter().flatten().all(|&wire| (wire as usize) < n),
        "input wire outside 0..n"
    );

    let total = 3 * n;
    let bookend = (((3 * n) as f64 * (n as f64).ln()).ceil() as usize).max(64);
    let mut out = rand_feistal_z_xgates(n, bookend, rng);
    let q = random_permutation(n, rng);
    let mut xloc: Vec<usize> = (0..n).collect();
    let mut yloc: Vec<usize> = (n..2 * n).collect();
    let mut zloc: Vec<usize> = (2 * n..total).collect();
    let mut on: Vec<FeistalSlot> = (0..total)
        .map(|wire| {
            if wire < n {
                FeistalSlot::RawX(wire)
            } else if wire < 2 * n {
                FeistalSlot::RawY(wire - n)
            } else {
                FeistalSlot::RawZ(wire - 2 * n)
            }
        })
        .collect();
    let mut pairs = vec![(usize::MAX, usize::MAX); n];
    let mut free = vec![usize::MAX; n];
    for i in 0..n {
        let (x, y, z) = (xloc[i], yloc[q[i]], zloc[i]);
        let share = random_wire_except(total, &[x, y, z], rng);
        let free_wire = random_wire_except(total, &[x, y, z, share], rng);
        let (moved_share, moved_free) = (on[share], on[free_wire]);
        emit_w_i_cnot(x, z, share, free_wire, &mut out);
        reloc_feistal(
            moved_share,
            share,
            x,
            &mut xloc,
            &mut yloc,
            &mut zloc,
            &mut pairs,
            &mut free,
        );
        reloc_feistal(
            moved_free, free_wire, z, &mut xloc, &mut yloc, &mut zloc, &mut pairs, &mut free,
        );
        on[x] = moved_share;
        on[z] = moved_free;
        emit_transvection_cnot(y, share, &mut out);
        emit_transvection_cnot(free_wire, y, &mut out);
        pairs[i] = (share, y);
        free[i] = free_wire;
        on[share] = FeistalSlot::Pair0(i);
        on[y] = FeistalSlot::Pair1(i);
        on[free_wire] = FeistalSlot::Free(i);
    }

    let mut state = FeistalState {
        sharing: GadgetState { n, pairs },
        free,
        q,
    };
    let mut source = main.clone();
    let source_rounds = source.gates.len();
    shoot_random_gate_with_rng(&mut source, source_rounds, rng);
    emit_sg3_rg_block_x(&source, &mut state, rg_freq, rng, &mut out);
    emit_feistal_n_cnot(&state, &mut out);
    let d = random_feistal_d_xgates(n, main.gates.len(), rng);
    emit_fragment3_rg_block_x(&d, &mut state, rg_freq, rng, &mut out);
    emit_feistal_decode_cnot(&state, &mut out);
    out.extend(rand_feistal_z_xgates(n, bookend, rng));

    CnotCircuit {
        gates: out,
        num_wires: total,
    }
}

pub fn feistalize_with_slice_zero_cnot(
    main: &CircuitSeq,
    n: usize,
    rg_freq: usize,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let mut preblock = slice_zero_preblock_cnot(n, rng);
    preblock
        .gates
        .extend(feistalize_cnot(main, n, rg_freq, rng).gates);
    preblock
}

pub fn feistalize_with_slice_zero_hardcoded_cnot(
    main: &CircuitSeq,
    n: usize,
    rg_freq: usize,
    rounds: usize,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let mut preblock = slice_zero_hardcoded_preblock_cnot(n, rounds, rng);
    preblock
        .gates
        .extend(feistalize_cnot(main, n, rg_freq, rng).gates);
    preblock
}

pub fn feistalize_with_slice_zero_random_cnot(
    main: &CircuitSeq,
    n: usize,
    rg_freq: usize,
    gate_count: usize,
    rng: &mut impl Rng,
) -> CnotSliceZeroRandomCircuit {
    let mut preblock = slice_zero_random_preblock_cnot(n, gate_count, rng);
    preblock
        .circuit
        .gates
        .extend(feistalize_cnot(main, n, rg_freq, rng).gates);
    preblock
}

#[cfg(test)]
mod cnot_gadget_tests {
    use super::*;
    use crate::postmix::xgate::eval_u64;
    use rand::{SeedableRng, rngs::StdRng};

    fn canonical_state() -> FeistalState {
        FeistalState {
            sharing: GadgetState {
                n: 3,
                pairs: vec![(0, 1), (3, 4), (6, 7)],
            },
            free: vec![2, 5, 8],
            q: vec![0, 1, 2],
        }
    }

    fn encode_two_share(logical: u64, masks: u64, pairs: &[(usize, usize)]) -> u64 {
        let mut physical = 0u64;
        for (index, &(p0_wire, p1_wire)) in pairs.iter().enumerate() {
            let p0 = (masks >> index) & 1;
            let p1 = ((logical >> index) & 1) ^ p0;
            physical |= p0 << p0_wire;
            physical |= p1 << p1_wire;
        }
        physical
    }

    fn decode_two_share(physical: u64, pairs: &[(usize, usize)]) -> u64 {
        pairs
            .iter()
            .enumerate()
            .fold(0, |logical, (index, &(p0, p1))| {
                logical | ((((physical >> p0) ^ (physical >> p1)) & 1) << index)
            })
    }

    fn prod_decode(
        physical: u64,
        value: usize,
        pairs: &[(usize, usize)],
        slots: &[Vec<ProdSlot>],
        consts: &[bool],
        locations: &[u16],
    ) -> u64 {
        // Single-carrier builds record `pairs[v] = (w, w)`, and XORing the wire
        // with itself would decode every value as 0 — so read one carrier when
        // the pair collapses, both when it does not.
        let (carrier0, carrier1) = pairs[value];
        let mut decoded = if carrier0 == carrier1 {
            (physical >> carrier0) & 1
        } else {
            ((physical >> carrier0) ^ (physical >> carrier1)) & 1
        };
        for slot in &slots[value] {
            let factor = slot
                .factors
                .iter()
                .all(|&(variable, a)| (((physical >> locations[variable as usize]) & 1) != 0) ^ a);
            decoded ^= factor as u64;
        }
        decoded ^ consts[value] as u64
    }

    fn decode_three_share(state: &FeistalState, physical: u64) -> u64 {
        (0..state.sharing.n).fold(0, |logical, index| {
            let (p0, p1) = state.sharing.pairs[index];
            let value = ((physical >> p0) ^ (physical >> p1) ^ (physical >> state.free[index])) & 1;
            logical | (value << index)
        })
    }

    fn encode_three_share(state: &FeistalState, logical: u64, masks: u64) -> u64 {
        let mut physical = 0u64;
        for index in 0..state.sharing.n {
            let p0 = (masks >> (2 * index)) & 1;
            let p1 = (masks >> (2 * index + 1)) & 1;
            let free = ((logical >> index) & 1) ^ p0 ^ p1;
            let (p0_wire, p1_wire) = state.sharing.pairs[index];
            physical |= p0 << p0_wire;
            physical |= p1 << p1_wire;
            physical |= free << state.free[index];
        }
        physical
    }

    /// Count secret-dependent single probes, same-prefix probe pairs, and all
    /// space-time probe pairs. Randomness is enumerated exactly for each fixed
    /// secret; unequal observation histograms are counted as leakage.
    fn probe_leak_counts(
        gates: &[XGate],
        wires: usize,
        secret_count: usize,
        randomness_count: usize,
        encode: impl Fn(usize, usize) -> u64,
    ) -> (usize, usize, usize) {
        let evolutions: Vec<Vec<Vec<u64>>> = (0..secret_count)
            .map(|secret| {
                (0..randomness_count)
                    .map(|randomness| {
                        let mut state = encode(secret, randomness);
                        let mut evolution = vec![state];
                        for gate in gates {
                            state = gate.apply_u64(state);
                            evolution.push(state);
                        }
                        evolution
                    })
                    .collect()
            })
            .collect();
        let points = (gates.len() + 1) * wires;
        let observed = |evolution: &[u64], point: usize| -> usize {
            ((evolution[point / wires] >> (point % wires)) & 1) as usize
        };

        let mut singles = 0usize;
        for point in 0..points {
            let base: usize = evolutions[0]
                .iter()
                .map(|evolution| observed(evolution, point))
                .sum();
            if evolutions[1..].iter().any(|samples| {
                samples
                    .iter()
                    .map(|evolution| observed(evolution, point))
                    .sum::<usize>()
                    != base
            }) {
                singles += 1;
            }
        }

        let mut same_prefix_pairs = 0usize;
        let mut space_time_pairs = 0usize;
        for left in 0..points {
            for right in left + 1..points {
                let histogram = |samples: &[Vec<u64>]| {
                    let mut counts = [0usize; 4];
                    for evolution in samples {
                        let value = observed(evolution, left) | (observed(evolution, right) << 1);
                        counts[value] += 1;
                    }
                    counts
                };
                let base = histogram(&evolutions[0]);
                if evolutions[1..]
                    .iter()
                    .any(|samples| histogram(samples) != base)
                {
                    space_time_pairs += 1;
                    if left / wires == right / wires {
                        same_prefix_pairs += 1;
                    }
                }
            }
        }
        (singles, same_prefix_pairs, space_time_pairs)
    }

    fn algebraic_output_degrees(gates: &[XGate], wires: usize) -> Vec<usize> {
        assert!(wires < usize::BITS as usize);
        (0..wires)
            .map(|output_wire| {
                let mut anf: Vec<u8> = (0..1usize << wires)
                    .map(|input| ((eval_u64(gates, input as u64) >> output_wire) & 1) as u8)
                    .collect();
                for variable in 0..wires {
                    for monomial in 0..1usize << wires {
                        if monomial & (1 << variable) != 0 {
                            anf[monomial] ^= anf[monomial ^ (1 << variable)];
                        }
                    }
                }
                anf.iter()
                    .enumerate()
                    .filter_map(|(monomial, &coefficient)| {
                        (coefficient != 0).then_some(monomial.count_ones() as usize)
                    })
                    .max()
                    .unwrap_or(0)
            })
            .collect()
    }

    fn assert_cnot_network_is_gate_locally_noncomplete(
        gates: &[XGate],
        wires: usize,
        share_groups: &[(usize, usize)],
    ) {
        let mut dependencies: Vec<u64> = (0..wires).map(|wire| 1u64 << wire).collect();
        for (index, gate) in gates.iter().enumerate() {
            assert!(
                !gate.comp && gate.width() == 1,
                "gate {index} is not a CNOT"
            );
            let target = gate.target as usize;
            let control = gate.ctrls[0].0 as usize;
            let gate_dependencies = dependencies[target] | dependencies[control];
            for &(share0, share1) in share_groups {
                let complete = (1u64 << share0) | (1u64 << share1);
                assert_ne!(
                    gate_dependencies & complete,
                    complete,
                    "gate {index} consumes a complete sharing ({share0},{share1})"
                );
            }
            dependencies[target] ^= dependencies[control];
        }
    }

    #[test]
    fn nonlinear_tdp_construction_seed_replays_and_diversifies() {
        let n = 6;
        let source = CircuitSeq {
            gates: (0..36)
                .map(|index| {
                    [
                        (index % n) as u16,
                        ((index + 1) % n) as u16,
                        ((index + 3) % n) as u16,
                    ]
                })
                .collect(),
        };
        let construct = |seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            tdp4n_nonlinear_with_slice_zero_random_cnot(&source, n, 1, 192, &mut rng)
        };

        let first = construct(0x5eed_2026_0727);
        let replay = construct(0x5eed_2026_0727);
        assert_eq!(first.public_y, replay.public_y);
        assert_eq!(first.public_z, replay.public_z);
        assert_eq!(first.circuit.num_wires, replay.circuit.num_wires);
        assert_eq!(first.circuit.gates, replay.circuit.gates);

        let different = construct(0x5eed_2026_0728);
        assert!(
            first.public_y != different.public_y
                || first.public_z != different.public_z
                || first.circuit.gates != different.circuit.gates,
            "distinct construction seeds produced identical artifacts"
        );
    }

    #[test]
    fn production_nonlinear_plan_is_only_2223_single_gray() {
        let config = ProdConfig::production();
        assert_eq!(
            (
                config.k,
                config.deg,
                config.k_hi,
                config.deg_hi,
                config.rsrc,
                config.max_width,
                config.fill_nl,
                config.roll,
                config.rung_menu,
                config.single,
                config.gray_fold,
            ),
            (3, 2, 1, 3, 1, 0, 2, 1, 1, 1, 1)
        );
        assert!(config.enabled());
        assert!(config.single_carrier());
        assert!(config.gray());
        assert_eq!(config.band_size(5), 6);
        assert_eq!(config.band_size(128), 128);
        assert_eq!(
            (
                config.g57_narrow,
                config.ladder_cap,
                config.cg_jitter,
                config.epoch,
                config.refill_data,
                config.fill_pivots,
            ),
            (0, 0, 0, 0, 0, 0),
            "the shuffletests port must not claim reference-only inactive levers"
        );

        let source = CircuitSeq {
            gates: vec![[0, 1, 2]],
        };
        let mut rng = StdRng::seed_from_u64(0x2223_600d);
        let transformed = nonlinear_gadgetize_cnot(&source, 5, 1, &mut rng);
        assert_eq!(
            transformed.num_wires, 11,
            "n carriers plus the production band"
        );
    }

    /// Locks the generated artifact and the construction RNG position while
    /// allocation-only Gray-fold changes are made. The hash is deliberately
    /// local and stable (rather than `DefaultHasher`, whose implementation is
    /// not a compatibility contract).
    #[test]
    fn production_2223_gray_fixed_seed_fingerprint() {
        let n = 8;
        let source = CircuitSeq {
            gates: vec![[0, 1, 2], [3, 4, 5], [6, 0, 3], [2, 7, 1]],
        };
        let mut rng = StdRng::seed_from_u64(0xa110_c222_3000_0001);
        let transformed = nonlinear_gadgetize_cnot(&source, n, 1, &mut rng);

        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        let mut absorb = |word: u64| {
            hash ^= word;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        };
        absorb(transformed.num_wires as u64);
        absorb(transformed.gates.len() as u64);
        for gate in &transformed.gates {
            absorb(gate.target as u64);
            absorb(gate.comp as u64);
            absorb(gate.ctrls.len() as u64);
            for &(wire, positive) in &gate.ctrls {
                absorb(wire as u64);
                absorb(positive as u64);
            }
        }
        let rng_tail = rng.next_u64();
        assert_eq!(transformed.gates.len(), 505);
        assert_eq!(
            hash, 0x39e8_05b5_28b6_619c,
            "fixed-seed 2223+Gray artifact changed"
        );
        assert_eq!(
            rng_tail, 0x25c8_4829_588e_6cbb,
            "fixed-seed 2223+Gray RNG consumption changed"
        );
    }

    #[test]
    fn prod_gray_fold_is_share_native_and_two_control() {
        // The Gray fold's contract, over the WHOLE input domain (dirty borrows,
        // no clean-ancilla assumption anywhere): the target value's decode
        // transitions by exactly the virtual gate, every other value is
        // untouched, every borrowed accumulator and sandwich helper is restored,
        // and no emitted gate has more than two controls.
        //
        // The residual-constant trap lives here: a gather lands `M + delta`, and
        // if the constant-atom absorption were wrong the fold would silently
        // compute `(M_b + delta)(M_c + eps)` -- a WRONG FUNCTION, not a leak.
        // Only a full-domain check over both operands' masks catches it, which
        // is why the source list below includes mixed-polarity CCNOTs.
        //
        // n must leave carriers over to borrow: values 0..3 are the operands,
        // and values 3 and 4 exist only so their four carriers can be borrowed.
        let n = 5;
        let carrier_total = 2 * n;
        let pairs: Vec<(usize, usize)> = (0..n).map(|v| (2 * v, 2 * v + 1)).collect();
        let band = 4usize; // wires 10..14
        let live = carrier_total + band;
        let config = ProdConfig {
            k: 1,
            deg: 2,
            k_hi: 1,
            deg_hi: 3,
            band,
            rsrc: 0,
            rung_menu: 1,
            gray_fold: 1,
            ..ProdConfig::off()
        };
        let source_gates: Vec<XGate> = vec![
            XGate::from_g57([0, 1, 2]),
            XGate::conj(2, [(0u16, false), (1u16, true)]).unwrap(),
            XGate::conj(0, [(1u16, true), (2u16, true)]).unwrap(),
            XGate::conj(1, [(0u16, false), (2u16, false)]).unwrap(),
            XGate::cnot(1, 2),
            XGate::x_gate(0),
        ];
        let mut gray_blocks = 0u64;
        for seed in 0..16u64 {
            let mut rng = StdRng::seed_from_u64(0x67a4_0000 + seed);
            let state = GadgetState {
                n,
                pairs: pairs.clone(),
            };
            let mut ledger = ProdLedger::new(n, &config, carrier_total);
            ledger.inject_all(&state, &mut rng, &mut Vec::new());
            for source_gate in &source_gates {
                let slots_before = ledger.slots.clone();
                let consts_before = ledger.consts.clone();
                let gray_before = ledger.cg_gray;
                let mut fold = Vec::new();
                ledger.fold_cg(source_gate, &state, &mut rng, &mut fold);
                let target_value = source_gate.target as usize;
                for gate in &fold {
                    assert!(
                        gate.width() <= 2,
                        "seed={seed} gate={source_gate:?}: emitted a {}-control gate",
                        gate.width()
                    );
                }
                if ledger.cg_gray > gray_before {
                    // Blocks the Gray fold declined are laddered by the
                    // odometer fallback instead; only count the Gray ones.
                    gray_blocks += 1;
                }
                let touched = (1u64 << pairs[target_value].0) | (1u64 << pairs[target_value].1);
                for input in 0..(1u64 << live) {
                    let before: Vec<u64> = (0..n)
                        .map(|value| {
                            prod_decode(
                                input,
                                value,
                                &pairs,
                                &slots_before,
                                &consts_before,
                                &ledger.loc,
                            )
                        })
                        .collect();
                    let out_state = eval_u64(&fold, input);
                    // Every accumulator and every sandwich helper is restored:
                    // the net effect lives entirely on the target's carriers.
                    assert_eq!(
                        (out_state ^ input) & !touched,
                        0,
                        "seed={seed} gate={source_gate:?}: a borrowed wire was not restored"
                    );
                    let after: Vec<u64> = (0..n)
                        .map(|value| {
                            prod_decode(
                                out_state,
                                value,
                                &pairs,
                                &ledger.slots,
                                &ledger.consts,
                                &ledger.loc,
                            )
                        })
                        .collect();
                    let fires = source_gate
                        .ctrls
                        .iter()
                        .all(|&(wire, polarity)| (before[wire as usize] != 0) == polarity)
                        ^ source_gate.comp;
                    for value in 0..n {
                        let expected = before[value] ^ ((value == target_value && fires) as u64);
                        assert_eq!(
                            after[value], expected,
                            "seed={seed} gate={source_gate:?} value={value} input={input:#x}"
                        );
                    }
                }
                assert_eq!(slots_before, ledger.slots, "the fold disturbed a slot");
            }
        }
        // The arity-2 sources must actually take the Gray path, or the test
        // above is only re-checking the odometer.
        assert_eq!(
            gray_blocks,
            16 * 4,
            "expected every arity-2 block to fold the Gray way, got {gray_blocks}"
        );
    }

    #[test]
    fn prod_gray_fold_keeps_the_accumulators_dirty() {
        // The security invariant, structurally: no emitted gate may read an
        // accumulator while that wire holds a CLEAN mask sum. Equivalently --
        // and this is what is checkable without re-running the exposure audit --
        // the fold must never write a wire that is bare-zero-initialized, and
        // every wire it borrows must be one the block does not otherwise read.
        //
        // What is asserted here: the accumulators are drawn from the CARRIER
        // roles (not the band, not by index), they are distinct from the
        // target's carriers and from every literal the block reads, and each is
        // written by an even number of gates so its incoming junk survives to
        // cancel. A clean accumulator would show up as a wire whose first
        // touch is a write with no prior read -- covered by the restoration
        // assertion in the test above, which a clean-ancilla variant fails.
        let n = 5;
        let carrier_total = 2 * n;
        let pairs: Vec<(usize, usize)> = (0..n).map(|v| (2 * v, 2 * v + 1)).collect();
        let band = 8usize;
        let config = ProdConfig {
            k: 1,
            deg: 2,
            k_hi: 2,
            deg_hi: 3,
            band,
            rsrc: 0,
            rung_menu: 1,
            gray_fold: 1,
            ..ProdConfig::off()
        };
        for seed in 0..32u64 {
            let mut rng = StdRng::seed_from_u64(0x67a5_0000 + seed);
            let state = GadgetState {
                n,
                pairs: pairs.clone(),
            };
            let mut ledger = ProdLedger::new(n, &config, carrier_total);
            ledger.inject_all(&state, &mut rng, &mut Vec::new());
            let source_gate = XGate::from_g57([0, 1, 2]);
            let mut fold = Vec::new();
            ledger.fold_cg(&source_gate, &state, &mut rng, &mut fold);
            assert_eq!(ledger.cg_gray, 1, "seed={seed}: not the gray path");
            // Wires the block reads as mask/carrier literals of its operands.
            let mut operand_wires: Vec<u16> = Vec::new();
            for value in 0..3usize {
                operand_wires.push(pairs[value].0 as u16);
                operand_wires.push(pairs[value].1 as u16);
            }
            for value in [1usize, 2] {
                for slot in &ledger.slots[value] {
                    operand_wires.extend(slot.lits(&ledger.loc).iter().map(|&(w, _)| w));
                }
            }
            let mut writes: std::collections::HashMap<u16, usize> =
                std::collections::HashMap::new();
            for gate in &fold {
                *writes.entry(gate.target).or_default() += 1;
            }
            for (&wire, &count) in &writes {
                if wire as usize == pairs[0].0 || wire as usize == pairs[0].1 {
                    continue; // the target's carriers: written any number of times
                }
                assert_eq!(
                    count % 2,
                    0,
                    "seed={seed}: borrowed wire {wire} is written {count} times, so its \
                     incoming value does not cancel"
                );
                assert!(
                    !operand_wires.contains(&wire),
                    "seed={seed}: wire {wire} is both borrowed and read as an operand literal"
                );
            }
        }
    }

    #[test]
    fn gray_fold_narrows_a_whole_gadget_and_still_tolerates_junk() {
        // End to end, against the wide fold on the same seed: the Gray fold has
        // to compose with the RG layer, the nonlinear band fill, re-sourcing
        // and the rolling band, not just work in isolation.
        //
        // The discriminator is the widest emitted gate. At [2,2,2,3] an arity-2
        // block's wide expansion reaches `arity * max_deg` = 6 controls; the
        // Gray fold's own emissions are all <= 2, so the only wide gates left
        // are the degree-3 SLOT emissions (inject / re-source / strip), which
        // stop at 3. Narrowing those is the `ladder_cap` lever, which is 0 in
        // the production preset here exactly as in the reference -- so 3 is the
        // correct ceiling to pin, not 2.
        let n = 5; // values 3 and 4 are idle, so their carriers can be borrowed
        let source = CircuitSeq {
            gates: vec![[0, 1, 2], [2, 0, 1], [1, 2, 0]],
        };
        // Pinned to the TWO-CARRIER decode on purpose. The subject here is the
        // fold, and the borrow pool is what the fixture is sized for: at n = 5
        // single-carrier there are five carrier wires total, and after
        // forbidding the target's and both operands' the accumulator pair
        // exhausts them, leaving `helper_pool` empty and the Gray fold
        // declining every block. The single-carrier end-to-end contract is
        // `prod_single_carrier_is_exact_on_half_the_wires` instead.
        let gray = ProdConfig {
            band: 6,
            single: 0,
            ..ProdConfig::production()
        };
        let wide = ProdConfig {
            gray_fold: 0,
            ..gray
        };
        assert_eq!((gray.k, gray.deg, gray.k_hi, gray.deg_hi), (3, 2, 1, 3));

        let build = |config: &ProdConfig| {
            let mut rng = StdRng::seed_from_u64(0x9c2a_0007);
            nonlinear_gadgetize_cnot_with_config(&source, n, 1, config, &mut rng)
        };
        let narrow = build(&gray);
        let legacy = build(&wide);
        let widest = |circuit: &CnotCircuit| {
            circuit
                .gates
                .iter()
                .map(|gate| gate.width())
                .max()
                .unwrap_or(0)
        };
        assert!(
            widest(&legacy) >= 4,
            "the wide fold should still emit the cartesian product's wide fragments, got {}",
            widest(&legacy)
        );
        assert!(
            widest(&narrow) <= 3,
            "the Gray fold left a {}-control gate; only degree-3 slot emissions may exceed 2",
            widest(&narrow)
        );
        // Which wires those surviving width-3 gates name cannot be read off
        // their indices -- a roll puts band variables on carrier-index wires
        // and carriers on band-index ones, which is the whole reason
        // `carrier_wires` resolves the pool by role. What is checkable from
        // outside is that the population shrank: the fold's contribution to the
        // wide census is gone and only the slot emissions are left.
        let wide_count =
            |circuit: &CnotCircuit| circuit.gates.iter().filter(|gate| gate.width() > 2).count();
        assert!(
            wide_count(&narrow) < wide_count(&legacy),
            "the Gray fold removed no wide gates: {} against {}",
            wide_count(&narrow),
            wide_count(&legacy)
        );

        // Correct for ARBITRARY junk on the aux and band wires, over the whole
        // physical domain -- the unconditional-endpoint contract the encoding
        // lives under, now with the accumulators borrowed dirty from it.
        assert_eq!(narrow.num_wires, 2 * n + 6);
        let low_mask = (1u64 << n) - 1;
        for input in 0..1u64 << narrow.num_wires {
            let expected = source.evaluate((input & low_mask) as usize) as u64 & low_mask;
            assert_eq!(
                eval_u64(&narrow.gates, input) & low_mask,
                expected,
                "input={input:#x}"
            );
        }
    }

    /// A random-ish g57 body on `n >= 4` wires, for tests that need a width
    /// the small fixtures cannot give (a wire census wants room for a band).
    fn single_test_main_wide(n: usize) -> CircuitSeq {
        let mut rng = StdRng::seed_from_u64(0x9e37_9b91);
        let mut gates = Vec::new();
        for _ in 0..4 * n {
            let a = rng.random_range(0..n) as u16;
            let b = loop {
                let wire = rng.random_range(0..n) as u16;
                if wire != a {
                    break wire;
                }
            };
            let c = loop {
                let wire = rng.random_range(0..n) as u16;
                if wire != a && wire != b {
                    break wire;
                }
            };
            gates.push([a, b, c]);
        }
        CircuitSeq { gates }
    }

    /// The single-carrier decode end to end: `n` carriers, not `2n`, and exact
    /// on the WHOLE physical domain (band junk included).
    ///
    /// Ported from the reference implementation's
    /// `prod_single_carrier_is_exact_on_half_the_wires` (`f8afe640`,
    /// `src/replace/gadgets.rs:10187`), with a fourth case added for the
    /// combination this tree ships: the Gray fold on a single-carrier build.
    #[test]
    fn prod_single_carrier_is_exact_on_half_the_wires() {
        let n = 6usize;
        let main = single_test_main_wide(n);
        let mask = (1u64 << n) - 1;
        // [1,2,3,3] and [1,2,2,3]: one linear term, the rest nonlinear.
        // Rolls on: a roll can leave a value sitting on a former band wire, so
        // the final routing has to be a full permutation, not a carrier-space
        // one. That is exactly what this exercises.
        for (k, deg, k_hi, deg_hi, roll, gray_fold) in [
            (1usize, 2usize, 2usize, 3usize, 0usize, 0usize),
            (2, 2, 1, 3, 0, 0),
            (1, 2, 2, 3, 1, 0),
            (3, 2, 1, 3, 1, 1),
        ] {
            let cfg = ProdConfig {
                k,
                deg,
                k_hi,
                deg_hi,
                band: 8,
                rsrc: 1,
                fill_nl: 2,
                roll,
                rung_menu: gray_fold,
                single: 1,
                gray_fold,
                ..ProdConfig::off()
            };
            assert!(cfg.single_carrier());
            for seed in 0..3u64 {
                let mut rng = StdRng::seed_from_u64(0x51_0000 + seed);
                let gadget = gadgetize_cnot_single(&main, n, 2, &cfg, &mut rng);
                assert_eq!(
                    gadget.num_wires,
                    n + 8,
                    "single carrier: n carriers, not 2n"
                );
                assert!(
                    gadget.gates.iter().all(|gate| !gate.ctrls.is_empty()),
                    "single-carrier build must not contain a bare X"
                );
                for input in 0..(1u64 << gadget.num_wires) {
                    let expected = main.evaluate((input & mask) as usize) as u64 & mask;
                    assert_eq!(
                        eval_u64(&gadget.gates, input) & mask,
                        expected,
                        "plan [{deg}x{k},{deg_hi}x{k_hi}] gray={gray_fold} seed={seed} \
                         input={input:#x}"
                    );
                }
            }
        }
    }

    /// The shipped preset, driven through the entry point the generator uses:
    /// `single = 1` has to come out as `n + band` wires, and the low `n`
    /// outputs have to be the source on the preblock's zero slice.
    #[test]
    fn production_gray_preset_builds_a_single_carrier_gadget() {
        let n = 6usize;
        let source: Vec<XGate> = single_test_main_wide(n)
            .gates
            .iter()
            .copied()
            .map(XGate::from_g57)
            .collect();
        let cfg = ProdConfig {
            band: 6,
            ..ProdConfig::production()
        };
        let mut rng = StdRng::seed_from_u64(0x2223_0001);
        let gadget = gadgetize_xgates_with_slice_zero_ccnot_with_config(
            &source,
            n,
            1,
            10 * n,
            &cfg,
            &mut rng,
        );
        // n carriers + band, against the two-carrier build's 2n + band.
        assert_eq!(gadget.num_wires, n + 6);
        let mask = (1u64 << n) - 1;
        let logical = single_test_main_wide(n);
        for x in 0..=mask {
            let expected = logical.evaluate(x as usize) as u64 & mask;
            assert_eq!(
                eval_u64(&gadget.gates, x) & mask,
                expected,
                "zero slice x={x:#x}"
            );
        }
    }

    #[test]
    fn product_share_fold_applies_virtual_gates_without_reconstructing_operands() {
        let n = 3;
        let carrier_total = 2 * n;
        let pairs = vec![(0usize, 1usize), (2, 3), (4, 5)];
        let config = ProdConfig {
            k: 1,
            deg: 2,
            k_hi: 1,
            deg_hi: 3,
            band: 6,
            rsrc: 0,
            max_width: 0,
            fill_nl: 0,
            roll: 0,
            ..ProdConfig::off()
        };
        let total = carrier_total + config.band_size(n);
        let source_gates = [
            XGate::from_g57([0, 1, 2]),
            XGate::cnot(1, 2),
            XGate::conj(2, [(0, false), (1, true)]).unwrap(),
            XGate::x_gate(0),
        ];

        for seed in 0..4u64 {
            let mut rng = StdRng::seed_from_u64(0x9d0d_0000 + seed);
            let state = GadgetState {
                n,
                pairs: pairs.clone(),
            };
            let mut ledger = ProdLedger::new(n, &config, carrier_total);
            ledger.inject_all(&state, &mut rng, &mut Vec::new());

            for source_gate in &source_gates {
                let slots_before = ledger.slots.clone();
                let consts_before = ledger.consts.clone();
                let mut fold = Vec::new();
                ledger.fold_cg(source_gate, &state, &mut rng, &mut fold);
                let target = source_gate.target as usize;
                assert!(fold.iter().all(|gate| {
                    !gate.ctrls.is_empty()
                        && [pairs[target].0, pairs[target].1].contains(&(gate.target as usize))
                }));

                for input in 0..(1u64 << total) {
                    let before: Vec<u64> = (0..n)
                        .map(|value| {
                            prod_decode(
                                input,
                                value,
                                &pairs,
                                &slots_before,
                                &consts_before,
                                &ledger.loc,
                            )
                        })
                        .collect();
                    let output = eval_u64(&fold, input);
                    let after: Vec<u64> = (0..n)
                        .map(|value| {
                            prod_decode(
                                output,
                                value,
                                &pairs,
                                &ledger.slots,
                                &ledger.consts,
                                &ledger.loc,
                            )
                        })
                        .collect();
                    let fires = source_gate
                        .ctrls
                        .iter()
                        .all(|&(wire, polarity)| (before[wire as usize] != 0) == polarity)
                        ^ source_gate.comp;
                    for value in 0..n {
                        assert_eq!(
                            after[value],
                            before[value] ^ ((value == target && fires) as u64),
                            "seed={seed} gate={source_gate:?} value={value}"
                        );
                    }
                }
                assert_eq!(slots_before, ledger.slots);
            }
        }
    }

    #[test]
    fn product_band_roll_relocates_sources_without_changing_any_decode() {
        let n = 3;
        let carrier_total = 2 * n;
        let band = 4;
        let total = carrier_total + band;
        let config = ProdConfig {
            k: 1,
            deg: 2,
            band,
            rsrc: 0,
            roll: 1,
            ..ProdConfig::off()
        };
        let mut left_home = 0;

        for seed in 0..12u64 {
            let mut rng = StdRng::seed_from_u64(0x0011_0000 + seed);
            let mut state = GadgetState {
                n,
                pairs: vec![(0, 1), (2, 3), (4, 5)],
            };
            let mut ledger = ProdLedger::new(n, &config, carrier_total);
            ledger.inject_all(&state, &mut rng, &mut Vec::new());

            for _ in 0..4 {
                let pairs_before = state.pairs.clone();
                let slots_before = ledger.slots.clone();
                let consts_before = ledger.consts.clone();
                let locations_before = ledger.loc.clone();
                let mut emitted = Vec::new();
                ledger.roll(&mut state, &mut rng, &mut emitted);
                assert!((3..=6).contains(&emitted.len()));
                assert!(
                    emitted
                        .iter()
                        .all(|gate| !gate.comp && (1..=2).contains(&gate.width()))
                );

                for input in 0..(1u64 << total) {
                    let output = eval_u64(&emitted, input);
                    for value in 0..n {
                        let before = prod_decode(
                            input,
                            value,
                            &pairs_before,
                            &slots_before,
                            &consts_before,
                            &locations_before,
                        );
                        let after = prod_decode(
                            output,
                            value,
                            &state.pairs,
                            &ledger.slots,
                            &ledger.consts,
                            &ledger.loc,
                        );
                        assert_eq!(
                            after, before,
                            "seed={seed} value={value}: rolling changed the decode"
                        );
                    }
                }

                let mut occupied = ledger.loc.clone();
                for &(carrier0, carrier1) in &state.pairs {
                    occupied.extend([carrier0 as u16, carrier1 as u16]);
                }
                occupied.sort_unstable();
                assert_eq!(occupied, (0..total as u16).collect::<Vec<_>>());
            }
            if ledger
                .loc
                .iter()
                .any(|&wire| (wire as usize) < carrier_total)
            {
                left_home += 1;
            }
        }
        assert!(
            left_home >= 9,
            "the rolling band rarely left its home range ({left_home}/12)"
        );
    }

    #[test]
    fn wide_product_fold_does_not_emit_odometer_order() {
        let n = 3;
        let carrier_total = 2 * n;
        let state = GadgetState {
            n,
            pairs: vec![(0, 1), (2, 3), (4, 5)],
        };
        let config = ProdConfig {
            k: 2,
            deg: 2,
            band: 6,
            rsrc: 0,
            ..ProdConfig::off()
        };
        let gate = XGate::from_g57([0, 1, 2]);

        for seed in 0..4u64 {
            let mut rng = StdRng::seed_from_u64(0x5017_0000 + seed);
            let mut ledger = ProdLedger::new(n, &config, carrier_total);
            ledger.inject_all(&state, &mut rng, &mut Vec::new());

            let mut lists = Vec::new();
            for &(wire, positive) in &gate.ctrls {
                let value = wire as usize;
                let (carrier0, carrier1) = state.pairs[value];
                let mut atoms = vec![vec![(carrier0 as u16, true)], vec![(carrier1 as u16, true)]];
                atoms.extend(
                    ledger.slots[value]
                        .iter()
                        .map(|slot| slot.lits(&ledger.loc).into_vec()),
                );
                if ledger.consts[value] ^ !positive {
                    atoms.push(Vec::new());
                }
                lists.push(atoms);
            }
            let mut odometer = Vec::new();
            let mut combination = vec![0usize; lists.len()];
            'enumerate: loop {
                let picked: Vec<&Vec<(u16, bool)>> = lists
                    .iter()
                    .zip(&combination)
                    .map(|(list, &index)| &list[index])
                    .collect();
                let width = picked.iter().map(|atom| atom.len()).max().unwrap_or(0);
                let mut literals = Vec::new();
                for index in 0..width {
                    for atom in &picked {
                        if let Some(&literal) = atom.get(index) {
                            literals.push(literal);
                        }
                    }
                }
                if !literals.is_empty() && normalize_prod_lits(&mut literals).is_some() {
                    let fragment = XGate::conj(0, literals).unwrap();
                    odometer.push(fragment.ctrls);
                }
                let mut axis = 0;
                loop {
                    if axis == combination.len() {
                        break 'enumerate;
                    }
                    combination[axis] += 1;
                    if combination[axis] < lists[axis].len() {
                        break;
                    }
                    combination[axis] = 0;
                    axis += 1;
                }
            }

            let mut emitted = Vec::new();
            ledger.fold_cg(&gate, &state, &mut rng, &mut emitted);
            let actual: Vec<_> = emitted
                .iter()
                .map(|fragment| fragment.ctrls.clone())
                .collect();
            assert_eq!(actual.len(), odometer.len());
            let mut actual_multiset = actual.clone();
            let mut expected_multiset = odometer.clone();
            actual_multiset.sort();
            expected_multiset.sort();
            assert_eq!(actual_multiset, expected_multiset);
            assert_ne!(
                actual, odometer,
                "seed={seed}: wide fold retained its static odometer clock"
            );
        }
    }

    #[test]
    fn narrow_rolling_product_gadget_preserves_arbitrary_junk() {
        let n = 3;
        let source = CircuitSeq {
            gates: vec![[0, 1, 2], [2, 0, 1], [1, 2, 0]],
        };
        let config = ProdConfig {
            k: 1,
            deg: 2,
            k_hi: 1,
            deg_hi: 3,
            band: 6,
            rsrc: 1,
            max_width: 2,
            fill_nl: 2,
            roll: 1,
            ..ProdConfig::off()
        };
        let mut rng = StdRng::seed_from_u64(0x0b0d_0001);
        let transformed = nonlinear_gadgetize_cnot_with_config(&source, n, 1, &config, &mut rng);
        assert_eq!(transformed.num_wires, 12);
        assert!(
            transformed
                .gates
                .iter()
                .all(|gate| !gate.ctrls.is_empty() && gate.width() <= 2)
        );
        let low_mask = (1u64 << n) - 1;
        for input in 0..1u64 << transformed.num_wires {
            let expected = source.evaluate((input & low_mask) as usize) as u64 & low_mask;
            assert_eq!(
                eval_u64(&transformed.gates, input) & low_mask,
                expected,
                "input={input:#x}"
            );
        }
    }

    #[test]
    fn nonlinear_band_fill_is_balanced_and_nonlinear() {
        let n = 4;
        let band = [9u16, 5, 11, 4, 8, 6];
        let mut rng = StdRng::seed_from_u64(0xbadd_f111);
        let mut gates = Vec::new();
        emit_band_fill_nl(n, &band, 2, &mut rng, &mut gates);

        let tables: Vec<Vec<bool>> = band
            .iter()
            .map(|&wire| {
                (0..1u64 << n)
                    .map(|input| ((eval_u64(&gates, input) >> wire) & 1) != 0)
                    .collect()
            })
            .collect();
        for (index, table) in tables.iter().enumerate() {
            assert_eq!(
                table.iter().filter(|&&value| value).count(),
                1 << (n - 1),
                "band wire {} is not balanced",
                band[index]
            );
        }
        let any_nonlinear = tables.iter().any(|table| {
            (0..1usize << n).any(|left| {
                (0..1usize << n)
                    .any(|right| table[0] ^ table[left] ^ table[right] ^ table[left ^ right])
            })
        });
        assert!(
            any_nonlinear,
            "cascaded fill produced only affine functions"
        );
    }

    #[test]
    fn production_slice_preblock_covers_aux_and_band_with_nonlinear_shapes() {
        let (n, band, gate_count) = (5usize, 2usize, 50usize);
        let mut rng = StdRng::seed_from_u64(0xcc60_0000);
        let preblock = slice_zero_ccnot_preblock(n, band, gate_count, &mut rng);
        assert_eq!(preblock.num_wires, 2 * n + band);
        assert_eq!(preblock.gates.len(), gate_count);
        assert!(slice_preblock_fixes_only_zero_slice(
            &preblock.gates,
            n,
            n + band
        ));

        let mut covered = std::collections::HashSet::new();
        let mut widths = [0usize; 4];
        for gate in &preblock.gates {
            assert!(!gate.comp);
            assert!((gate.target as usize) < n);
            assert!(gate.ctrls.iter().all(|&(_, positive)| positive));
            let slice_controls: Vec<u16> = gate
                .ctrls
                .iter()
                .map(|&(wire, _)| wire)
                .filter(|&wire| wire as usize >= n)
                .collect();
            assert_eq!(slice_controls.len(), 1);
            covered.insert(slice_controls[0]);
            widths[gate.width()] += 1;
        }
        assert_eq!(covered.len(), n + band);
        assert!(widths[1] > 0 && widths[2] > 0 && widths[3] > 0);

        let data_mask = (1u64 << n) - 1;
        let nonlinear = (1..1u64 << (n + band)).any(|slice| {
            let evaluate = |data| eval_u64(&preblock.gates, data | (slice << n)) & data_mask;
            let base = evaluate(0);
            (0..n).any(|left| {
                ((left + 1)..n).any(|right| {
                    let a = 1u64 << left;
                    let b = 1u64 << right;
                    evaluate(a ^ b) ^ evaluate(a) ^ evaluate(b) ^ base != 0
                })
            })
        });
        assert!(
            nonlinear,
            "three-control preblock shapes produced no observed quadratic slice"
        );
    }

    #[test]
    fn production_nonlinear_wrapper_is_correct_on_the_aux_and_band_zero_slice() {
        let n = 3;
        let source = CircuitSeq {
            gates: vec![[0, 1, 2], [2, 0, 1], [1, 2, 0]],
        };
        let mut rng = StdRng::seed_from_u64(0xcc70_0000);
        let transformed = nonlinear_gadgetize_with_slice_zero_cnot(&source, n, 1, &mut rng);
        assert_eq!(
            transformed.num_wires,
            n + ProdConfig::production().band_size(n),
            "production must use one carrier per value plus the product band"
        );
        let low_mask = (1u64 << n) - 1;
        for data in 0..=low_mask {
            let expected = source.evaluate(data as usize) as u64 & low_mask;
            assert_eq!(
                eval_u64(&transformed.gates, data) & low_mask,
                expected,
                "data={data:#x}"
            );
        }
        let has_three_control_slice_gate = transformed.gates.iter().any(|gate| {
            gate.width() == 3
                && !gate.comp
                && (gate.target as usize) < n
                && gate.ctrls.iter().all(|&(_, positive)| positive)
                && gate
                    .ctrls
                    .iter()
                    .filter(|&&(wire, _)| (wire as usize) < n)
                    .count()
                    == 2
                && gate
                    .ctrls
                    .iter()
                    .filter(|&&(wire, _)| (wire as usize) >= n)
                    .count()
                    == 1
        });
        assert!(has_three_control_slice_gate);

        // Every individual non-data wire, including every band home wire,
        // must affect the exposed low function for at least one data input.
        for slice_wire in n..transformed.num_wires {
            let slice = 1u64 << slice_wire;
            let disturbed = (0..=low_mask).any(|data| {
                let expected = source.evaluate(data as usize) as u64 & low_mask;
                eval_u64(&transformed.gates, data | slice) & low_mask != expected
            });
            assert!(disturbed, "slice wire {slice_wire} did not affect the data");
        }
    }

    #[test]
    fn nonlinear_gadgetization_preserves_low_outputs_for_arbitrary_junk() {
        let n = 3;
        let source = CircuitSeq {
            gates: vec![[0, 1, 2], [2, 0, 1], [1, 2, 0]],
        };
        let mut rng = StdRng::seed_from_u64(0x2330_c001);
        let transformed = nonlinear_gadgetize_cnot(&source, n, 1, &mut rng);
        assert_eq!(
            transformed.num_wires,
            n + ProdConfig::production().band_size(n),
            "production must use one carrier per value plus the product band"
        );
        assert!(transformed.gates.iter().all(|gate| !gate.ctrls.is_empty()));
        assert!(
            transformed
                .gates
                .iter()
                .map(XGate::width)
                .max()
                .unwrap_or(0)
                >= 3,
            "the degree-3 tower did not produce wide fragments"
        );

        let low_mask = (1u64 << n) - 1;
        for input in 0..1u64 << transformed.num_wires {
            let expected = source.evaluate((input & low_mask) as usize) as u64 & low_mask;
            assert_eq!(
                eval_u64(&transformed.gates, input) & low_mask,
                expected,
                "input={input:#x}"
            );
        }
    }

    #[test]
    fn w_i_cnot_has_the_required_linear_map() {
        let mut gates = Vec::new();
        emit_w_i_cnot(0, 1, 2, 3, &mut gates);
        assert_eq!(gates.len(), 7);
        assert!(gates.iter().all(|gate| !gate.comp && gate.width() == 1));
        for input in 0..16u64 {
            let q0 = input & 1;
            let q1 = (input >> 1) & 1;
            let q2 = (input >> 2) & 1;
            let q3 = (input >> 3) & 1;
            let expected = q2 | (q3 << 1) | ((q0 ^ q1) << 2) | (q1 << 3);
            assert_eq!(eval_u64(&gates, input), expected);
        }
    }

    #[test]
    fn four_fragment_two_share_sg_is_correct_and_prefix_masked() {
        let state = GadgetState {
            n: 3,
            pairs: vec![(0, 1), (2, 3), (4, 5)],
        };
        let mut gates = Vec::new();
        emit_gadget_x(&state, [0, 1, 2], &mut gates);
        assert_eq!(gates.len(), 4);
        assert!(gates.iter().all(|gate| !gate.comp && gate.width() == 2));
        for logical in 0..8u64 {
            for prefix in 0..=gates.len() {
                let mut ones = [0usize; 6];
                for masks in 0..8u64 {
                    let input = encode_two_share(logical, masks, &state.pairs);
                    let output = eval_u64(&gates[..prefix], input);
                    for (wire, count) in ones.iter_mut().enumerate() {
                        *count += ((output >> wire) & 1) as usize;
                    }
                    if prefix == gates.len() {
                        assert_eq!(
                            decode_two_share(output, &state.pairs),
                            XGate::from_g57([0, 1, 2]).apply_u64(logical)
                        );
                    }
                }
                assert_eq!(ones, [4; 6]);
            }
        }
    }

    #[test]
    fn legacy_three_share_sg_retains_stronger_two_probe_boundary_security() {
        let state = canonical_state();
        let mut legacy = Vec::new();
        emit_sg3_x(&state, [0, 1, 2], &mut legacy);
        assert_eq!(legacy.len(), 9);
        assert!(legacy.iter().all(|gate| gate.comp && gate.width() == 2));

        // Rejected eight-gate candidate: temporarily reduce each control from
        // three carriers to two, use the four-fragment SG, then restore it.
        let mut rejected = Vec::new();
        emit_transvection_cnot(3, 4, &mut rejected);
        emit_transvection_cnot(6, 7, &mut rejected);
        emit_shared_g57_frag2(2, 3, 5, 6, 8, &mut rejected);
        emit_transvection_cnot(6, 7, &mut rejected);
        emit_transvection_cnot(3, 4, &mut rejected);
        assert_eq!(rejected.len(), 8);

        let legacy_leaks = probe_leak_counts(&legacy, 9, 8, 64, |secret, randomness| {
            encode_three_share(&state, secret as u64, randomness as u64)
        });
        let rejected_leaks = probe_leak_counts(&rejected, 9, 8, 64, |secret, randomness| {
            encode_three_share(&state, secret as u64, randomness as u64)
        });
        println!("three-share SG probes: legacy={legacy_leaks:?} rejected={rejected_leaks:?}");
        assert_eq!(legacy_leaks, (0, 0, 26));
        assert_eq!(rejected_leaks, (0, 12, 123));

        for logical in 0..8u64 {
            for prefix in 0..=legacy.len() {
                let mut ones = [0usize; 9];
                for masks in 0..64u64 {
                    let input = encode_three_share(&state, logical, masks);
                    let output = eval_u64(&legacy[..prefix], input);
                    for (wire, count) in ones.iter_mut().enumerate() {
                        *count += ((output >> wire) & 1) as usize;
                    }
                    if prefix == legacy.len() {
                        assert_eq!(
                            decode_three_share(&state, output),
                            XGate::from_g57([0, 1, 2]).apply_u64(logical)
                        );
                    }
                }
                assert_eq!(ones, [32; 9]);
            }
        }
    }

    #[test]
    fn two_share_sg_probe_comparison_favors_the_four_fragment_variant() {
        let state = GadgetState {
            n: 3,
            pairs: vec![(0, 1), (2, 3), (4, 5)],
        };
        let mut old_g57 = Vec::new();
        emit_gadget(&state, [0, 1, 2], &mut old_g57);
        let old_g57: Vec<XGate> = old_g57.into_iter().map(XGate::from_g57).collect();
        let mut fragments = Vec::new();
        emit_gadget_x(&state, [0, 1, 2], &mut fragments);
        let old_leaks = probe_leak_counts(&old_g57, 6, 8, 8, |secret, randomness| {
            encode_two_share(secret as u64, randomness as u64, &state.pairs)
        });
        let new_leaks = probe_leak_counts(&fragments, 6, 8, 8, |secret, randomness| {
            encode_two_share(secret as u64, randomness as u64, &state.pairs)
        });
        println!("two-share SG probes: legacy={old_leaks:?} fragments={new_leaks:?}");
        assert_eq!(old_leaks, (0, 19, 151));
        assert_eq!(new_leaks, (0, 14, 73));

        // The old SG has gates reading both B carriers simultaneously. Every
        // new fragment reads at most one carrier from each logical control.
        assert!(old_g57.iter().any(|gate| gate.reads(2) && gate.reads(3)));
        assert!(
            fragments
                .iter()
                .all(|gate| !(gate.reads(2) && gate.reads(3)))
        );
        assert!(
            fragments
                .iter()
                .all(|gate| !(gate.reads(4) && gate.reads(5)))
        );
    }

    #[test]
    fn selected_cnot_rg_variants_preserve_values_masks_and_noncompleteness() {
        for variant in 1..=2 {
            let original_pairs = vec![(0, 1), (2, 3)];
            let mut state = GadgetState {
                n: 2,
                pairs: original_pairs.clone(),
            };
            let mut gates = Vec::new();
            if variant == 1 {
                emit_rg1_x(&mut state, 0, 1, &mut gates);
                assert_eq!(gates.len(), 6);
            } else {
                emit_rg2_x(&mut state, 0, 1, &mut gates);
                assert_eq!(gates.len(), 3);
            }
            assert_cnot_network_is_gate_locally_noncomplete(&gates, 4, &original_pairs);
            for logical in 0..4u64 {
                for prefix in 0..=gates.len() {
                    let mut ones = [0usize; 4];
                    for masks in 0..4u64 {
                        let input = encode_two_share(logical, masks, &original_pairs);
                        let output = eval_u64(&gates[..prefix], input);
                        for (wire, count) in ones.iter_mut().enumerate() {
                            *count += ((output >> wire) & 1) as usize;
                        }
                        if prefix == gates.len() {
                            assert_eq!(decode_two_share(output, &state.pairs), logical);
                        }
                    }
                    assert_eq!(ones, [2; 4], "RG{variant} prefix={prefix}");
                }
            }
        }

        let pairs = vec![(0, 1), (2, 3), (4, 5)];
        let state = GadgetState {
            n: 3,
            pairs: pairs.clone(),
        };
        let mut gates = Vec::new();
        emit_rg3_x(&state, 0, 2, &mut gates);
        assert_eq!(gates.len(), 2);
        assert_cnot_network_is_gate_locally_noncomplete(&gates, 6, &pairs);
        for logical in 0..8u64 {
            for prefix in 0..=gates.len() {
                let mut ones = [0usize; 6];
                for masks in 0..8u64 {
                    let input = encode_two_share(logical, masks, &pairs);
                    let output = eval_u64(&gates[..prefix], input);
                    for (wire, count) in ones.iter_mut().enumerate() {
                        *count += ((output >> wire) & 1) as usize;
                    }
                    if prefix == gates.len() {
                        assert_eq!(decode_two_share(output, &pairs), logical);
                    }
                }
                assert_eq!(ones, [4; 6], "RG3 prefix={prefix}");
            }
        }
    }

    #[test]
    fn six_cnot_rg1_is_minimal_under_gate_local_noncompleteness() {
        let start = [1u8, 2, 4, 8];
        let goal = |rows: [u8; 4]| rows[3] ^ rows[2] == 3 && rows[0] ^ rows[1] == 12;
        let noncomplete = |left: u8, right: u8| {
            let dependencies = left | right;
            dependencies & 3 != 3 && dependencies & 12 != 12
        };
        let mut queue = VecDeque::from([(start, 0usize)]);
        let mut seen = std::collections::HashSet::from([start]);
        while let Some((rows, depth)) = queue.pop_front() {
            assert!(!(depth <= 5 && depth != 0 && goal(rows)));
            if depth == 5 {
                continue;
            }
            for target in 0..4 {
                for control in 0..4 {
                    if target == control || !noncomplete(rows[target], rows[control]) {
                        continue;
                    }
                    let mut next = rows;
                    next[target] ^= next[control];
                    if seen.insert(next) {
                        queue.push_back((next, depth + 1));
                    }
                }
            }
        }

        let mut state = GadgetState {
            n: 2,
            pairs: vec![(0, 1), (3, 2)],
        };
        let mut selected = Vec::new();
        emit_rg1_x(&mut state, 0, 1, &mut selected);
        assert_eq!(selected.len(), 6);
    }

    #[test]
    fn old_and_new_rg_probe_comparison_supports_cnot_replacements() {
        let make_two_share = |variant: usize, mixed: bool| {
            let mut state = GadgetState {
                n: 2,
                pairs: vec![(0, 1), (3, 2)],
            };
            if mixed {
                let mut gates = Vec::new();
                if variant == 1 {
                    emit_rg1_x(&mut state, 0, 1, &mut gates);
                } else {
                    emit_rg2_x(&mut state, 0, 1, &mut gates);
                }
                gates
            } else {
                let mut gates = Vec::new();
                if variant == 1 {
                    emit_rg1(&mut state, 0, 1, &mut gates);
                } else {
                    emit_rg2(&mut state, 0, 1, &mut gates);
                }
                gates.into_iter().map(XGate::from_g57).collect()
            }
        };
        let pairs = vec![(0, 1), (3, 2)];
        for variant in 1..=2 {
            let old = make_two_share(variant, false);
            let new = make_two_share(variant, true);
            let old_leaks = probe_leak_counts(&old, 4, 4, 4, |secret, randomness| {
                encode_two_share(secret as u64, randomness as u64, &pairs)
            });
            let new_leaks = probe_leak_counts(&new, 4, 4, 4, |secret, randomness| {
                encode_two_share(secret as u64, randomness as u64, &pairs)
            });
            println!("two-share RG{variant} probes: legacy={old_leaks:?} selected={new_leaks:?}");
            // The first legacy G57 in both RG1 and RG2 reads both carriers of
            // logical j. The selected CNOT networks are checked separately
            // above with evolving symbolic dependencies.
            assert!(old[0].reads(2) && old[0].reads(3));
            if variant == 1 {
                assert_eq!(old_leaks, (0, 14, 191));
                assert_eq!(new_leaks, (0, 10, 68));
            } else {
                // Legacy RG2 has secret-dependent 1/4-vs-3/4 bias at four
                // individual space-time probe locations.
                assert_eq!(old_leaks, (4, 21, 205));
                assert_eq!(new_leaks, (0, 6, 24));
            }
        }

        // In the Feistel representation y and its pair mask are random while
        // x is fixed. New RG1/RG2 retain second-order masking of x. Legacy
        // RG2 does not: seven space-time pairs have x-dependent histograms.
        let feistal_encode = |secret: usize, randomness: usize| {
            let x0 = secret & 1;
            let x1 = (secret >> 1) & 1;
            let y0 = randomness & 1;
            let m0 = (randomness >> 1) & 1;
            let y1 = (randomness >> 2) & 1;
            let m1 = (randomness >> 3) & 1;
            (m0 | ((m0 ^ y0) << 1)
                | ((x0 ^ y0) << 2)
                | (m1 << 3)
                | ((m1 ^ y1) << 4)
                | ((x1 ^ y1) << 5)) as u64
        };
        for variant in 1..=2 {
            let old = make_two_share(variant, false);
            let new = make_two_share(variant, true);
            let remap = |gates: Vec<XGate>| {
                gates
                    .into_iter()
                    .map(|gate| {
                        let map = [0u16, 1, 4, 3];
                        let mut ctrls: crate::postmix::xgate::Lits = gate
                            .ctrls
                            .into_iter()
                            .map(|(wire, polarity)| (map[wire as usize], polarity))
                            .collect();
                        ctrls.sort_unstable();
                        XGate {
                            target: map[gate.target as usize],
                            comp: gate.comp,
                            ctrls,
                        }
                    })
                    .collect::<Vec<_>>()
            };
            let old_leaks = probe_leak_counts(&remap(old), 6, 4, 16, feistal_encode);
            let new_leaks = probe_leak_counts(&remap(new), 6, 4, 16, feistal_encode);
            println!("Feistal RG{variant} probes: legacy={old_leaks:?} selected={new_leaks:?}");
            if variant == 1 {
                assert_eq!(old_leaks, (0, 0, 0));
                assert_eq!(new_leaks, (0, 0, 0));
            } else {
                assert_eq!(old_leaks, (0, 1, 7));
                assert_eq!(new_leaks, (0, 0, 0));
            }
        }
    }

    #[test]
    fn rg3_g57_and_selected_cnot_probe_comparison() {
        let pairs = vec![(0, 1), (2, 3), (4, 5)];
        let state = GadgetState {
            n: 3,
            pairs: pairs.clone(),
        };
        let mut legacy_g57 = Vec::new();
        emit_rg3(&state, 0, 2, 4, &mut legacy_g57);
        let legacy: Vec<XGate> = legacy_g57.into_iter().map(XGate::from_g57).collect();
        let mut selected = Vec::new();
        emit_rg3_x(&state, 0, 2, &mut selected);
        let legacy_leaks = probe_leak_counts(&legacy, 6, 8, 8, |secret, randomness| {
            encode_two_share(secret as u64, randomness as u64, &pairs)
        });
        let selected_leaks = probe_leak_counts(&selected, 6, 8, 8, |secret, randomness| {
            encode_two_share(secret as u64, randomness as u64, &pairs)
        });
        println!("two-share RG3 probes: legacy={legacy_leaks:?} selected={selected_leaks:?}");
        assert_eq!(legacy_leaks, (0, 9, 27));
        assert_eq!(selected_leaks, (0, 8, 22));
        assert_eq!(legacy.len(), selected.len());
        assert_eq!(algebraic_output_degrees(&legacy, 6)[..2], [2, 2]);
        assert_eq!(algebraic_output_degrees(&selected, 6)[..2], [1, 1]);

        // Survey the raw ordered (r1,r2) placement space. Production now
        // excludes the third carrier; every remaining placement is clean.
        let feistal = canonical_state();
        let (p0, p1) = feistal.sharing.pairs[0];
        let mut legacy_max = (0, 0, 0);
        let mut linear_max = (0, 0, 0);
        let mut legacy_unsafe = Vec::new();
        let mut linear_unsafe = Vec::new();
        for r1 in 0..9 {
            if r1 == p0 || r1 == p1 {
                continue;
            }
            for r2 in 0..9 {
                if r2 == p0 || r2 == p1 || r2 == r1 {
                    continue;
                }
                let mut old_g57 = Vec::new();
                emit_rg3(&feistal.sharing, 0, r1, r2, &mut old_g57);
                let old: Vec<XGate> = old_g57.into_iter().map(XGate::from_g57).collect();
                let mut cnot = Vec::new();
                emit_rg3_x(&feistal.sharing, 0, r1, &mut cnot);
                let encode = |secret: usize, randomness: usize| {
                    encode_three_share(&feistal, secret as u64, randomness as u64)
                };
                let old_leaks = probe_leak_counts(&old, 9, 8, 64, encode);
                let cnot_leaks = probe_leak_counts(&cnot, 9, 8, 64, encode);
                if r1 != feistal.free[0] && r2 != feistal.free[0] {
                    assert_eq!(old_leaks, (0, 0, 0));
                }
                if r1 != feistal.free[0] {
                    assert_eq!(cnot_leaks, (0, 0, 0));
                }
                if old_leaks.1 != 0 {
                    legacy_unsafe.push((r1, r2, old_leaks));
                }
                if cnot_leaks.1 != 0 {
                    linear_unsafe.push((r1, r2, cnot_leaks));
                }
                legacy_max.0 = legacy_max.0.max(old_leaks.0);
                legacy_max.1 = legacy_max.1.max(old_leaks.1);
                legacy_max.2 = legacy_max.2.max(old_leaks.2);
                linear_max.0 = linear_max.0.max(cnot_leaks.0);
                linear_max.1 = linear_max.1.max(cnot_leaks.1);
                linear_max.2 = linear_max.2.max(cnot_leaks.2);
            }
        }
        println!("three-share RG3 placement maxima: legacy={legacy_max:?} selected={linear_max:?}");
        println!(
            "three-share RG3 unsafe placements: legacy={legacy_unsafe:?} selected={linear_unsafe:?}"
        );
        assert_eq!(legacy_max, (0, 1, 5));
        assert_eq!(linear_max, (0, 1, 5));
        assert_eq!(legacy_unsafe.len(), 12);
        assert_eq!(linear_unsafe.len(), 6);
    }

    #[test]
    fn selected_rg_networks_are_deliberately_linear() {
        let make = |variant: usize| {
            let mut state = GadgetState {
                n: 2,
                pairs: vec![(0, 1), (3, 2)],
            };
            let mut gates = Vec::new();
            if variant == 1 {
                emit_rg1_x(&mut state, 0, 1, &mut gates);
            } else {
                emit_rg2_x(&mut state, 0, 1, &mut gates);
            }
            gates
        };
        assert_eq!(algebraic_output_degrees(&make(1), 4), vec![1; 4]);
        assert_eq!(algebraic_output_degrees(&make(2), 4), vec![1; 4]);
    }

    #[test]
    fn feistal_cnot_rgs_preserve_overlapping_x_y_and_prefix_masking() {
        for variant in 1..=3 {
            let initial_pairs = vec![(0, 1), (3, 4), (6, 7)];
            let mut state = canonical_state();
            state.q = vec![0, 1, 2];
            let mut gates = Vec::new();
            match variant {
                1 => emit_rg1_x(&mut state.sharing, 0, 1, &mut gates),
                2 => emit_rg2_x(&mut state.sharing, 0, 1, &mut gates),
                _ => emit_rg3_x(&state.sharing, 0, state.free[1], &mut gates),
            }
            for x in 0..8u64 {
                for prefix in 0..=gates.len() {
                    let mut ones = [0usize; 9];
                    for y in 0..8u64 {
                        for masks in 0..8u64 {
                            let mut input = 0u64;
                            for index in 0..3 {
                                let p0 = (masks >> index) & 1;
                                let y_bit = (y >> index) & 1;
                                let free = ((x >> index) & 1) ^ y_bit;
                                input |= p0 << initial_pairs[index].0;
                                input |= (p0 ^ y_bit) << initial_pairs[index].1;
                                input |= free << state.free[index];
                            }
                            let output = eval_u64(&gates[..prefix], input);
                            for (wire, count) in ones.iter_mut().enumerate() {
                                *count += ((output >> wire) & 1) as usize;
                            }
                            if prefix == gates.len() {
                                assert_eq!(decode_three_share(&state, output), x);
                                let decoded_y = (0..3).fold(0u64, |value, host| {
                                    let (p0, p1) = state.sharing.pairs[host];
                                    value
                                        | ((((output >> p0) ^ (output >> p1)) & 1) << state.q[host])
                                });
                                assert_eq!(decoded_y, y);
                            }
                        }
                    }
                    assert_eq!(ones, [32; 9], "Feistal RG{variant} prefix={prefix}");
                }
            }
        }
    }

    #[test]
    fn two_share_homomorphic_cnot_is_correct_and_prefix_masked() {
        let gates = homomorphic_cnot2((0, 1), (2, 3));
        assert_eq!(gates.len(), 2);
        assert_cnot_network_is_gate_locally_noncomplete(&gates, 4, &[(0, 1), (2, 3)]);
        for logical in 0..4u64 {
            let a = logical & 1;
            let b = (logical >> 1) & 1;
            for prefix in 0..=gates.len() {
                let mut ones = [0usize; 4];
                for masks in 0..4u64 {
                    let a0 = masks & 1;
                    let b0 = (masks >> 1) & 1;
                    let input = a0 | ((a ^ a0) << 1) | (b0 << 2) | ((b ^ b0) << 3);
                    let output = eval_u64(&gates[..prefix], input);
                    for (wire, count) in ones.iter_mut().enumerate() {
                        *count += ((output >> wire) & 1) as usize;
                    }
                    if prefix == gates.len() {
                        assert_eq!(((output >> 0) ^ (output >> 1)) & 1, a ^ b);
                        assert_eq!(((output >> 2) ^ (output >> 3)) & 1, b);
                    }
                }
                assert_eq!(ones, [2; 4]);
            }
        }
    }

    #[test]
    fn three_share_homomorphic_cnot_is_correct_and_prefix_masked() {
        let state = canonical_state();
        let gates = homomorphic_cnot3((0, 1, 2), (3, 4, 5));
        assert_eq!(gates.len(), 3);
        for logical in 0..8u64 {
            for prefix in 0..=gates.len() {
                let mut ones = [0usize; 9];
                for masks in 0..64u64 {
                    let input = encode_three_share(&state, logical, masks);
                    let output = eval_u64(&gates[..prefix], input);
                    for (wire, count) in ones.iter_mut().enumerate() {
                        *count += ((output >> wire) & 1) as usize;
                    }
                    if prefix == gates.len() {
                        let decoded = decode_three_share(&state, output);
                        let expected = logical ^ (((logical >> 1) & 1) << 0);
                        assert_eq!(decoded, expected);
                    }
                }
                assert_eq!(ones, [32; 9]);
            }
        }
    }

    #[test]
    fn shared_fragments_compute_without_unmasking_any_single_carrier() {
        let state = canonical_state();
        let logical_gates = [
            XGate::cnot(0, 1),
            XGate::conj(2, [(0, false)]).unwrap(),
            XGate::conj(1, [(0, true), (2, false)]).unwrap(),
            XGate::from_g57([0, 1, 2]),
        ];
        for logical_gate in logical_gates {
            let mut physical_gates = Vec::new();
            emit_shared_fragment3(&state, &logical_gate, &mut physical_gates);
            for logical in 0..8u64 {
                for masks in 0..64u64 {
                    let encoded = encode_three_share(&state, logical, masks);
                    let result = eval_u64(&physical_gates, encoded);
                    assert_eq!(
                        decode_three_share(&state, result),
                        logical_gate.apply_u64(logical)
                    );
                }

                // At every physical-gate prefix, every individual carrier is
                // exactly balanced over the masks for each fixed secret.
                for prefix in 0..=physical_gates.len() {
                    let mut ones = [0usize; 9];
                    for masks in 0..64u64 {
                        let encoded = encode_three_share(&state, logical, masks);
                        let result = eval_u64(&physical_gates[..prefix], encoded);
                        for (wire, count) in ones.iter_mut().enumerate() {
                            *count += ((result >> wire) & 1) as usize;
                        }
                    }
                    assert_eq!(ones, [32; 9]);
                }
            }
        }
    }

    #[test]
    fn two_share_fragments_compute_without_unmasking_any_single_carrier() {
        let state = GadgetState {
            n: 4,
            pairs: vec![(0, 1), (2, 3), (4, 5), (6, 7)],
        };
        let logical_gates = [
            XGate::cnot(0, 1),
            XGate::conj(2, [(0, false)]).unwrap(),
            XGate::conj(1, [(0, true), (2, false)]).unwrap(),
            XGate::from_g57([0, 1, 2]),
            XGate::x_gate(3),
        ];
        for logical_gate in logical_gates {
            let mut physical_gates = Vec::new();
            emit_shared_fragment2(&state, &logical_gate, &mut physical_gates);
            for logical in 0..16u64 {
                for masks in 0..16u64 {
                    let encoded = encode_two_share(logical, masks, &state.pairs);
                    let result = eval_u64(&physical_gates, encoded);
                    assert_eq!(
                        decode_two_share(result, &state.pairs),
                        logical_gate.apply_u64(logical)
                    );
                }

                for prefix in 0..=physical_gates.len() {
                    let mut ones = [0usize; 8];
                    for masks in 0..16u64 {
                        let encoded = encode_two_share(logical, masks, &state.pairs);
                        let result = eval_u64(&physical_gates[..prefix], encoded);
                        for (wire, count) in ones.iter_mut().enumerate() {
                            *count += ((result >> wire) & 1) as usize;
                        }
                    }
                    assert_eq!(ones, [8; 8]);
                }
            }
        }
    }

    #[test]
    fn feistal_n_cnot_keeps_every_prefix_first_order_masked() {
        for q in [
            vec![0, 1, 2],
            vec![0, 2, 1],
            vec![1, 0, 2],
            vec![1, 2, 0],
            vec![2, 0, 1],
            vec![2, 1, 0],
        ] {
            let mut state = canonical_state();
            state.q = q;
            let mut gates = Vec::new();
            emit_feistal_n_cnot(&state, &mut gates);
            for logical_x in 0..8u64 {
                for prefix in 0..=gates.len() {
                    let mut ones = [0usize; 9];
                    for masks in 0..64u64 {
                        // Independent p0/p1 choices are equivalent to averaging
                        // over the Feistel y values and their random pair masks
                        // for this fixed original x.
                        let input = encode_three_share(&state, logical_x, masks);
                        let output = eval_u64(&gates[..prefix], input);
                        for (wire, count) in ones.iter_mut().enumerate() {
                            *count += ((output >> wire) & 1) as usize;
                        }
                    }
                    assert_eq!(ones, [32; 9], "q={:?} prefix={prefix}", state.q);
                }
            }
        }
    }

    #[test]
    fn gadgetize_cnot_preserves_the_first_n_wires() {
        let n = 3;
        let main = CircuitSeq {
            gates: vec![[0, 1, 2], [2, 0, 1], [1, 2, 0]],
        };
        for seed in 0..8u64 {
            let mut rng = StdRng::seed_from_u64(0xc001_0000 + seed);
            let transformed = gadgetize_cnot(&main, n, 2, &mut rng);
            assert_eq!(transformed.num_wires, 2 * n);
            let mask = (1u64 << n) - 1;
            for input in 0..(1u64 << (2 * n)) {
                let expected = main.evaluate((input & mask) as usize) as u64 & mask;
                assert_eq!(eval_u64(&transformed.gates, input) & mask, expected);
            }
        }
    }

    #[test]
    fn feistalize_cnot_moves_functionality_to_the_middle_n_wires() {
        let n = 3;
        let main = CircuitSeq {
            gates: vec![[0, 1, 2], [2, 0, 1], [1, 2, 0]],
        };
        for seed in 0..8u64 {
            let mut rng = StdRng::seed_from_u64(0xfe15_0000 + seed);
            let transformed = feistalize_cnot(&main, n, 2, &mut rng);
            assert_eq!(transformed.num_wires, 3 * n);
            let mask = (1u64 << n) - 1;
            for input in 0..(1u64 << (3 * n)) {
                let x = input & mask;
                let y = (input >> n) & mask;
                let expected = y ^ (main.evaluate(x as usize) as u64 & mask);
                assert_eq!((eval_u64(&transformed.gates, input) >> n) & mask, expected);
            }
        }
    }

    #[test]
    fn tdp4n_cnot_exhaustively_keeps_y_xor_cx_in_the_middle_block() {
        let n = 3;
        let main = CircuitSeq {
            gates: vec![[0, 1, 2], [2, 0, 1], [1, 2, 0]],
        };
        let mut rng = StdRng::seed_from_u64(0x4d54_4450);
        let transformed = tdp4n_cnot(&main, n, 2, &mut rng);
        assert_eq!(transformed.num_wires, 4 * n);
        let mask = (1u64 << n) - 1;
        for input in 0..(1u64 << (4 * n)) {
            let x = input & mask;
            let y = (input >> n) & mask;
            let expected = y ^ (main.evaluate(x as usize) as u64 & mask);
            assert_eq!(
                (eval_u64(&transformed.gates, input) >> n) & mask,
                expected,
                "input={input:#x}"
            );
        }
    }

    #[test]
    fn tdp4n_random_slice_fixes_yz_and_leaves_w_free() {
        let n = 3;
        let main = CircuitSeq {
            gates: vec![[0, 1, 2], [2, 0, 1], [1, 2, 0]],
        };
        let mut rng = StdRng::seed_from_u64(0x51ce_4d54);
        let transformed = tdp4n_with_slice_zero_random_cnot(&main, n, 2, 96, &mut rng);
        assert_eq!(transformed.circuit.num_wires, 4 * n);
        let mask = (1u64 << n) - 1;
        let public_y = transformed.public_y[0] & mask;
        let public_z = transformed.public_z[0] & mask;
        for x in 0..=mask {
            for w in 0..=mask {
                let input = x | (public_y << n) | (public_z << (2 * n)) | (w << (3 * n));
                let output = eval_u64(&transformed.circuit.gates, input);
                let expected = public_y ^ (main.evaluate(x as usize) as u64 & mask);
                assert_eq!((output >> n) & mask, expected, "x={x:#x} w={w:#x}");
            }
        }
    }

    #[test]
    fn nonlinear_tdp_random_slice_fixes_yz_and_leaves_w_and_band_free() {
        let n = 3;
        let main = CircuitSeq {
            gates: vec![[0, 1, 2], [2, 0, 1], [1, 2, 0]],
        };
        let mut rng = StdRng::seed_from_u64(0x51ce_2330);
        let transformed = tdp4n_nonlinear_with_slice_zero_random_cnot(&main, n, 1, 96, &mut rng);
        let band_width = ProdConfig::production().band_size(2 * n);
        assert_eq!(transformed.circuit.num_wires, 4 * n + band_width);
        let mask = (1u64 << n) - 1;
        let public_y = transformed.public_y[0] & mask;
        let public_z = transformed.public_z[0] & mask;
        let mut helpers = StdRng::seed_from_u64(0x51ce_badd);
        for x in 0..=mask {
            for _ in 0..32 {
                let w = helpers.random::<u64>() & mask;
                let band = helpers.random::<u64>() & ((1u64 << band_width) - 1);
                let input = x
                    | (public_y << n)
                    | (public_z << (2 * n))
                    | (w << (3 * n))
                    | (band << (4 * n));
                let output = eval_u64(&transformed.circuit.gates, input);
                let expected = public_y ^ (main.evaluate(x as usize) as u64 & mask);
                assert_eq!(
                    (output >> n) & mask,
                    expected,
                    "x={x:#x} w={w:#x} band={band:#x}"
                );
            }
        }
    }

    #[test]
    fn sandwich_slice_gates_are_dead_on_the_zero_slice() {
        let n = 4;
        let mask = (1u64 << n) - 1;
        for seed in 0..8u64 {
            let mut rng = StdRng::seed_from_u64(0x5a2d_0000 + seed);
            let block = sandwich_slice_gates(n, 5 * n, &mut rng);
            for gate in &block {
                assert!(!gate.comp);
                assert!((gate.target as usize) < n);
                assert!(gate.ctrls.iter().all(|&(_, positive)| positive));
                assert!(gate.ctrls.iter().any(|&(wire, _)| (wire as usize) >= n));
            }
            for x in 0..=mask {
                assert_eq!(eval_u64(&block, x), x);
            }
        }
    }

    #[test]
    fn sliced_sandwich_computes_c_on_the_second_half_on_the_zero_slice() {
        let n = 3;
        let mask = (1u64 << n) - 1;
        let main = CircuitSeq {
            gates: vec![[0, 1, 2], [2, 0, 1], [1, 2, 0], [0, 2, 1]],
        };
        for seed in 0..8u64 {
            let mut rng = StdRng::seed_from_u64(0x5a4d_0000 + seed);
            let sandwich = sliced_sandwich_cnot(&main, n, 12, 4 * n, &mut rng);
            assert_eq!(sandwich.num_wires, 2 * n);
            for x in 0..=mask {
                let expected = main.evaluate(x as usize) as u64 & mask;
                let actual = (eval_u64(&sandwich.gates, x) >> n) & mask;
                assert_eq!(actual, expected, "seed={seed} x={x:#x}");
            }
        }
    }

    #[test]
    fn sliced_sandwich_floats_the_copy_column_into_a_band() {
        let n = 6;
        let main = CircuitSeq {
            gates: (0..24)
                .map(|index| {
                    [
                        (index % n) as u16,
                        ((index + 1) % n) as u16,
                        ((index + 2) % n) as u16,
                    ]
                })
                .collect(),
        };
        for seed in 0..8u64 {
            let mut rng = StdRng::seed_from_u64(0x5a6d_0000 + seed);
            let sandwich = sliced_sandwich_cnot(&main, n, 20, 4 * n, &mut rng);
            let positions: Vec<usize> = sandwich
                .gates
                .iter()
                .enumerate()
                .filter_map(|(index, gate)| ((gate.target as usize) >= n).then_some(index))
                .collect();
            assert_eq!(positions.len(), n);
            assert!(
                positions.last().unwrap() - positions.first().unwrap() > n,
                "seed={seed}: copy column remained contiguous"
            );
        }
    }

    #[test]
    fn sliced_sandwich_2223_gray_gadget_preserves_the_nested_zero_slice() {
        let n = 3;
        let sandwich_n = 2 * n;
        let mask = (1u64 << n) - 1;
        let main = CircuitSeq {
            gates: vec![[0, 1, 2], [2, 0, 1], [1, 2, 0], [0, 2, 1]],
        };
        let mut sandwich_rng = StdRng::seed_from_u64(0x5a4d_2223);
        let sandwich = sliced_sandwich_cnot(&main, n, 12, 4 * n, &mut sandwich_rng);
        let mut gadget_rng = StdRng::seed_from_u64(0x6ad6_2223);
        let gadget = gadgetize_xgates_with_slice_zero_ccnot(
            &sandwich.gates,
            sandwich_n,
            1,
            10 * sandwich_n,
            &mut gadget_rng,
        );
        assert_eq!(gadget.num_wires, 2 * sandwich_n);
        for x in 0..=mask {
            let expected = main.evaluate(x as usize) as u64 & mask;
            let actual = (eval_u64(&gadget.gates, x) >> n) & mask;
            assert_eq!(actual, expected, "x={x:#x}");
        }
    }

    #[test]
    fn tdp4n_representative_n8_circuit_preserves_the_middle_view() {
        let n = 8;
        let main = CircuitSeq {
            gates: (0..64)
                .map(|index| {
                    [
                        (index % n) as u16,
                        ((index + 1) % n) as u16,
                        ((index + 3) % n) as u16,
                    ]
                })
                .collect(),
        };
        let mut rng = StdRng::seed_from_u64(0x4d54_0008);
        let transformed = tdp4n_with_slice_zero_random_cnot(&main, n, 2, 256, &mut rng);
        assert_eq!(transformed.circuit.num_wires, 32);
        assert!(!transformed.circuit.gates.is_empty());
        let mask = (1u64 << n) - 1;
        let public_y = transformed.public_y[0] & mask;
        let public_z = transformed.public_z[0] & mask;
        let mut samples = StdRng::seed_from_u64(0x4d54_5a4d);
        for _ in 0..128 {
            let x = samples.random::<u64>() & mask;
            let w = samples.random::<u64>() & mask;
            let input = x | (public_y << n) | (public_z << (2 * n)) | (w << (3 * n));
            let output = eval_u64(&transformed.circuit.gates, input);
            let expected = public_y ^ (main.evaluate(x as usize) as u64 & mask);
            assert_eq!((output >> n) & mask, expected);
        }
    }

    #[test]
    fn random_fragment_preblock_has_one_and_only_one_fixed_aux_slice() {
        let n = 3;
        let mask = (1u64 << n) - 1;
        for seed in 0..8u64 {
            let mut rng = StdRng::seed_from_u64(0x51ce_0000 + seed);
            let preblock = slice_zero_random_preblock_cnot(n, 96, &mut rng);
            assert_eq!(preblock.circuit.gates.len(), 96);
            let public_y = preblock.public_y[0] & mask;
            let public_z = preblock.public_z[0] & mask;
            for y in 0..=mask {
                for z in 0..=mask {
                    for x in 0..=mask {
                        let input = x | (y << n) | (z << (2 * n));
                        let output = eval_u64(&preblock.circuit.gates, input);
                        assert_eq!((output >> n) & ((1u64 << (2 * n)) - 1), input >> n);
                        if y == public_y && z == public_z {
                            assert_eq!(output, input);
                        } else {
                            assert_ne!(output & mask, x);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn cnot_transformations_are_leaner_than_legacy_on_representative_circuits() {
        let n = 8;
        let main = CircuitSeq {
            gates: (0..64)
                .map(|index| {
                    [
                        (index % n) as u16,
                        ((index + 1) % n) as u16,
                        ((index + 2) % n) as u16,
                    ]
                })
                .collect(),
        };
        let mut legacy_gadget_total = 0usize;
        let mut cnot_gadget_total = 0usize;
        let mut legacy_feistal_total = 0usize;
        let mut cnot_feistal_total = 0usize;
        for seed in 0..16u64 {
            let mut legacy_rng = StdRng::seed_from_u64(0x1ea0_0000 + seed);
            let mut cnot_rng = StdRng::seed_from_u64(0x1ea0_0000 + seed);
            let legacy_gadget = gadgetize(&main, n, 2, &mut legacy_rng).gates.len();
            let cnot_gadget = gadgetize_cnot(&main, n, 2, &mut cnot_rng).gates.len();
            assert!(cnot_gadget < legacy_gadget, "gadget seed={seed}");
            legacy_gadget_total += legacy_gadget;
            cnot_gadget_total += cnot_gadget;

            let mut legacy_rng = StdRng::seed_from_u64(0xfe15_0000 + seed);
            let mut cnot_rng = StdRng::seed_from_u64(0xfe15_0000 + seed);
            let legacy_feistal = feistalize(&main, n, 2, &mut legacy_rng).gates.len();
            let cnot_feistal = feistalize_cnot(&main, n, 2, &mut cnot_rng).gates.len();
            assert!(cnot_feistal < legacy_feistal, "Feistal seed={seed}");
            legacy_feistal_total += legacy_feistal;
            cnot_feistal_total += cnot_feistal;
        }
        println!(
            "representative averages: gadget {} -> {}; Feistal {} -> {}",
            legacy_gadget_total / 16,
            cnot_gadget_total / 16,
            legacy_feistal_total / 16,
            cnot_feistal_total / 16,
        );
    }
}

#[cfg(test)]
mod feistal_tests {
    use super::*;
    use crate::circuit::circuit::Gate;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn middle_block_is_y_plus_cx() {
        let n = 3;
        let main = CircuitSeq {
            gates: vec![[0, 1, 2], [2, 0, 1]],
        };
        let mut rng = StdRng::seed_from_u64(0x57fe157a);
        let circuit = feistalize(&main, n, 1, &mut rng);
        let mask = (1usize << n) - 1;
        for input in 0..(1usize << (3 * n)) {
            let x = input & mask;
            let y = (input >> n) & mask;
            assert_eq!((circuit.evaluate(input) >> n) & mask, y ^ main.evaluate(x));
        }
    }

    #[test]
    fn sg3_realizes_g57() {
        let state = FeistalState {
            sharing: GadgetState {
                n: 3,
                pairs: vec![(0, 1), (3, 4), (6, 7)],
            },
            free: vec![2, 5, 8],
            q: vec![1, 2, 0],
        };
        let mut gates = Vec::new();
        emit_sg3(&state, [0, 1, 2], &mut gates);
        for input in 0..512usize {
            let decode = |v: usize| {
                (0..3).fold(0, |acc, i| {
                    acc | ((((v >> state.sharing.pairs[i].0) & 1)
                        ^ ((v >> state.sharing.pairs[i].1) & 1)
                        ^ ((v >> state.free[i]) & 1))
                        << i)
                })
            };
            assert_eq!(
                decode(Gate::evaluate_index_list(input, &gates)),
                Gate::evaluate_index(decode(input), [0, 1, 2])
            );
        }
    }
}

#[cfg(test)]
mod feistal_property_tests {
    use super::*;
    use crate::circuit::circuit::Gate;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use std::collections::HashSet;

    fn canonical_state() -> FeistalState {
        FeistalState {
            sharing: GadgetState {
                n: 3,
                pairs: vec![(0, 1), (3, 4), (6, 7)],
            },
            free: vec![2, 5, 8],
            q: vec![1, 2, 0],
        }
    }

    fn virtual_values(state: &FeistalState, physical: usize) -> (usize, usize) {
        let mut x = 0usize;
        let mut y = 0usize;
        for i in 0..state.sharing.n {
            let (p0, p1) = state.sharing.pairs[i];
            let pair = ((physical >> p0) & 1) ^ ((physical >> p1) & 1);
            let x_bit = pair ^ ((physical >> state.free[i]) & 1);
            x |= x_bit << i;
            y |= pair << state.q[i];
        }
        (x, y)
    }

    fn evaluate_gates(input: usize, gates: &[[u16; 3]]) -> usize {
        Gate::evaluate_index_list(input, &gates.to_vec())
    }

    fn deterministic_circuit(n: usize, m: usize, seed: u64) -> CircuitSeq {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut gates = Vec::with_capacity(m);
        for _ in 0..m {
            let active = rng.random_range(0..n) as u16;
            let pos = loop {
                let wire = rng.random_range(0..n) as u16;
                if wire != active {
                    break wire;
                }
            };
            let neg = loop {
                let wire = rng.random_range(0..n) as u16;
                if wire != active && wire != pos {
                    break wire;
                }
            };
            gates.push([active, pos, neg]);
        }
        CircuitSeq { gates }
    }

    fn packed_words_to_usize(words: &[u64], n: usize) -> usize {
        let mut out = 0usize;
        for bit in 0..n {
            if packed_bit(words, bit) {
                out |= 1usize << bit;
            }
        }
        out
    }

    #[test]
    fn each_rg_variant_preserves_overlapping_x_and_y_values() {
        for variant in 0..3 {
            let mut state = canonical_state();
            let mut gates = Vec::new();
            match variant {
                0 => emit_rg1(&mut state.sharing, 0, 1, &mut gates),
                1 => emit_rg2(&mut state.sharing, 0, 1, &mut gates),
                2 => emit_rg3(&state.sharing, 0, 3, 6, &mut gates),
                _ => unreachable!(),
            }
            for input in 0..512usize {
                let before = virtual_values(&canonical_state(), input);
                let output = evaluate_gates(input, &gates);
                assert_eq!(virtual_values(&state, output), before, "RG{}", variant + 1);
            }
        }
    }

    #[test]
    fn n_tilde_updates_y_by_x_and_preserves_x() {
        let state = canonical_state();
        let mut gates = Vec::new();
        emit_feistal_n(&state, &mut gates);
        for input in 0..512usize {
            let (x, y) = virtual_values(&state, input);
            let output = evaluate_gates(input, &gates);
            assert_eq!(virtual_values(&state, output), (x, y ^ x));
        }
    }

    #[test]
    fn sg3_preserves_all_y_values() {
        let state = canonical_state();
        for gate in [[0, 1, 2], [1, 2, 0], [2, 0, 1]] {
            let mut gates = Vec::new();
            emit_sg3(&state, gate, &mut gates);
            for input in 0..512usize {
                let (x, y) = virtual_values(&state, input);
                let output = evaluate_gates(input, &gates);
                let expected_x = Gate::evaluate_index(x, gate);
                assert_eq!(virtual_values(&state, output), (expected_x, y));
            }
        }
    }

    #[test]
    fn feistalize_end_to_end_matrix() {
        let cases = [
            (3usize, 0usize, 0x100u64),
            (3, 1, 0x101),
            (3, 7, 0x102),
            (4, 3, 0x103),
            (4, 12, 0x104),
            (5, 20, 0x105),
        ];
        for (n, m, circuit_seed) in cases {
            let main = deterministic_circuit(n, m, circuit_seed);
            let mask = (1usize << n) - 1;
            for rg_freq in [1usize, 2, 3, m.max(1) + 1] {
                for layout_seed in [0x200u64, 0x201, 0x202, 0x203] {
                    let mut rng = StdRng::seed_from_u64(layout_seed ^ circuit_seed);
                    let transformed = feistalize(&main, n, rg_freq, &mut rng);
                    assert!(
                        transformed
                            .gates
                            .iter()
                            .flatten()
                            .all(|&w| (w as usize) < 3 * n)
                    );

                    let inputs: Vec<usize> = if n == 3 {
                        (0..(1usize << (3 * n))).collect()
                    } else {
                        let mut sample_rng =
                            StdRng::seed_from_u64(layout_seed ^ circuit_seed ^ rg_freq as u64);
                        let mut values = vec![0, mask, mask << n, mask << (2 * n)];
                        for x in 0..=mask {
                            values.push(x);
                            values.push(x | (mask << (2 * n)));
                        }
                        for _ in 0..256 {
                            values.push(sample_rng.random_range(0..(1usize << (3 * n))));
                        }
                        values
                    };

                    for input in inputs {
                        let x = input & mask;
                        let y = (input >> n) & mask;
                        let output = transformed.evaluate(input);
                        assert_eq!(
                            (output >> n) & mask,
                            y ^ main.evaluate(x),
                            "n={n} m={m} rg={rg_freq} seed={layout_seed:#x} input={input:#x}",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn zero_initialized_middle_block_is_exactly_cx() {
        let n = 5;
        let mask = (1usize << n) - 1;
        for circuit_seed in 0x300u64..0x308 {
            let main = deterministic_circuit(n, 15, circuit_seed);
            let mut rng = StdRng::seed_from_u64(circuit_seed ^ 0x57);
            let transformed = feistalize(&main, n, 2, &mut rng);
            for x in 0..=mask {
                for z in [0usize, 1, mask / 2, mask] {
                    let input = x | (z << (2 * n));
                    assert_eq!((transformed.evaluate(input) >> n) & mask, main.evaluate(x));
                }
            }
        }
    }

    #[test]
    fn slice_zero_preblock_fixes_exactly_the_zero_slice() {
        let n = 3;
        let mask = (1usize << n) - 1;
        for seed in 0x5a10u64..0x5a18 {
            let mut rng = StdRng::seed_from_u64(seed);
            let block = slice_zero_preblock(n, &mut rng);
            assert!(block.gates.iter().flatten().all(|&w| (w as usize) < 3 * n));

            let mut outputs = HashSet::new();
            for input in 0..(1usize << (3 * n)) {
                let x = input & mask;
                let y = (input >> n) & mask;
                let z = (input >> (2 * n)) & mask;
                let output = block.evaluate(input);
                outputs.insert(output);

                assert_eq!((output >> n) & mask, y);
                assert_eq!((output >> (2 * n)) & mask, z);
                if y == 0 && z == 0 {
                    assert_eq!(output & mask, x);
                } else {
                    assert_ne!(output & mask, x, "seed={seed:#x} input={input:#x}");
                }
            }
            assert_eq!(outputs.len(), 1usize << (3 * n));
        }
    }

    #[test]
    fn slice_zero_hardcoded_preblock_fixes_zero_slice() {
        let n = 4;
        let mask = (1usize << n) - 1;
        for seed in 0x5b10u64..0x5b18 {
            let mut rng = StdRng::seed_from_u64(seed);
            let block = slice_zero_hardcoded_preblock(n, 2, &mut rng);
            assert!(block.gates.iter().flatten().all(|&w| (w as usize) < 3 * n));
            assert!(!block.gates.is_empty());

            for x in 0..=mask {
                let output = block.evaluate(x);
                assert_eq!(output & mask, x, "seed={seed:#x} x={x:#x}");
                assert_eq!((output >> n) & mask, 0);
                assert_eq!((output >> (2 * n)) & mask, 0);
            }

            let mut moved = false;
            for y in 1..=mask {
                let output = block.evaluate(y << n);
                moved |= output != (y << n);
            }
            assert!(
                moved,
                "hardcoded M should change at least one off-slice input"
            );
        }
    }

    #[test]
    fn slice_zero_feistalize_matches_original_only_on_zero_slice() {
        let n = 3;
        let mask = (1usize << n) - 1;
        let main = CircuitSeq {
            gates: vec![[0, 1, 2], [2, 0, 1], [1, 2, 0]],
        };

        for seed in 0x5f00u64..0x5f08 {
            let mut rng = StdRng::seed_from_u64(seed);
            let transformed = feistalize_with_slice_zero(&main, n, 2, &mut rng);
            for input in 0..(1usize << (3 * n)) {
                let x = input & mask;
                let y = (input >> n) & mask;
                let z = (input >> (2 * n)) & mask;
                let middle = (transformed.evaluate(input) >> n) & mask;
                let old_middle = y ^ main.evaluate(x);
                if y == 0 && z == 0 {
                    assert_eq!(middle, main.evaluate(x));
                } else {
                    assert_ne!(
                        middle, old_middle,
                        "seed={seed:#x} input={input:#x} x={x:#x} y={y:#x} z={z:#x}"
                    );
                }
            }
        }
    }

    #[test]
    fn slice_zero_hardcoded_feistalize_matches_original_on_zero_slice() {
        let n = 3;
        let mask = (1usize << n) - 1;
        let main = CircuitSeq {
            gates: vec![[0, 1, 2], [2, 0, 1], [1, 2, 0]],
        };

        for seed in 0x5c00u64..0x5c08 {
            let mut rng = StdRng::seed_from_u64(seed);
            let transformed = feistalize_with_slice_zero_hardcoded(&main, n, 2, 1, &mut rng);
            for x in 0..=mask {
                let middle = (transformed.evaluate(x) >> n) & mask;
                assert_eq!(middle, main.evaluate(x), "seed={seed:#x} x={x:#x}");
            }
        }
    }

    #[test]
    fn slice_zero_random_preblock_fixes_public_slice() {
        let n = 4;
        let mask = (1usize << n) - 1;
        for seed in 0x6100u64..0x6108 {
            let mut rng = StdRng::seed_from_u64(seed);
            let block = slice_zero_random_preblock(n, 256, &mut rng);
            let public_y = packed_words_to_usize(&block.public_y, n);
            let public_z = packed_words_to_usize(&block.public_z, n);

            assert!(
                block
                    .circuit
                    .gates
                    .iter()
                    .flatten()
                    .all(|&w| (w as usize) < 3 * n)
            );
            for x in 0..=mask {
                let input = x | (public_y << n) | (public_z << (2 * n));
                let output = block.circuit.evaluate(input);
                assert_eq!(output & mask, x, "seed={seed:#x} x={x:#x}");
                assert_eq!((output >> n) & mask, public_y);
                assert_eq!((output >> (2 * n)) & mask, public_z);
            }
        }
    }

    #[test]
    fn slice_zero_random_feistalize_matches_original_on_public_slice() {
        let n = 3;
        let mask = (1usize << n) - 1;
        let main = CircuitSeq {
            gates: vec![[0, 1, 2], [2, 0, 1], [1, 2, 0]],
        };

        for seed in 0x6200u64..0x6208 {
            let mut rng = StdRng::seed_from_u64(seed);
            let transformed = feistalize_with_slice_zero_random(&main, n, 2, 128, &mut rng);
            let public_y = packed_words_to_usize(&transformed.public_y, n);
            let public_z = packed_words_to_usize(&transformed.public_z, n);

            for x in 0..=mask {
                let input = x | (public_y << n) | (public_z << (2 * n));
                let middle = (transformed.circuit.evaluate(input) >> n) & mask;
                assert_eq!(
                    middle,
                    public_y ^ main.evaluate(x),
                    "seed={seed:#x} x={x:#x}"
                );
            }
        }
    }
}

#[cfg(test)]
mod feistal_structural_tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};
    use std::collections::HashSet;

    #[test]
    fn transformed_small_circuit_is_a_permutation_with_nonconstant_garbage() {
        let n = 3;
        let mask = (1usize << n) - 1;
        let main = CircuitSeq {
            gates: vec![[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 2, 1]],
        };
        for seed in 0x400u64..0x408 {
            let mut rng = StdRng::seed_from_u64(seed);
            let transformed = feistalize(&main, n, 1, &mut rng);
            let outputs: Vec<usize> = (0..512).map(|input| transformed.evaluate(input)).collect();
            assert_eq!(outputs.iter().copied().collect::<HashSet<_>>().len(), 512);
            assert!(
                outputs
                    .iter()
                    .map(|v| v & mask)
                    .collect::<HashSet<_>>()
                    .len()
                    > 1
            );
            assert!(
                outputs
                    .iter()
                    .map(|v| (v >> (2 * n)) & mask)
                    .collect::<HashSet<_>>()
                    .len()
                    > 1
            );
        }
    }
}

#[cfg(test)]
mod feistal_32_wire_tests {
    use super::*;
    use crate::circuit::circuit::Gate;
    use primitive_types::U256;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    fn circuit_32(seed: u64, gates: usize) -> CircuitSeq {
        let mut rng = StdRng::seed_from_u64(seed);
        random_circuit_with_rng(32, gates, &mut rng)
    }

    fn assert_middle_block(
        transformed: &CircuitSeq,
        original: &CircuitSeq,
        x: u32,
        y: u32,
        z: u32,
        context: &str,
    ) {
        let input = U256::from(x) | (U256::from(y) << 32) | (U256::from(z) << 64);
        let output = Gate::evaluate_index_list_256(input, &transformed.gates);
        let cx = Gate::evaluate_index_list_256(U256::from(x), &original.gates);
        let middle = ((output >> 32) & U256::from(u32::MAX)).low_u32();
        assert_eq!(
            middle,
            y ^ cx.low_u32(),
            "{context}: x={x:#010x} y={y:#010x} z={z:#010x}"
        );
    }

    fn middle_block(transformed: &CircuitSeq, x: u32, y: u32, z: u32) -> u32 {
        let input = U256::from(x) | (U256::from(y) << 32) | (U256::from(z) << 64);
        let output = Gate::evaluate_index_list_256(input, &transformed.gates);
        ((output >> 32) & U256::from(u32::MAX)).low_u32()
    }

    fn packed_words_to_u32(words: &[u64]) -> u32 {
        let mut out = 0u32;
        for bit in 0..32 {
            if packed_bit(words, bit) {
                out |= 1u32 << bit;
            }
        }
        out
    }

    fn eval_x_block(circuit: &CircuitSeq, x: u32, y: u32, z: u32) -> u32 {
        let input = U256::from(x) | (U256::from(y) << 32) | (U256::from(z) << 64);
        let output = Gate::evaluate_index_list_256(input, &circuit.gates);
        (output & U256::from(u32::MAX)).low_u32()
    }

    #[test]
    fn thirty_two_wire_middle_block_is_y_plus_cx_for_many_inputs() {
        let patterns = [
            0u32,
            1,
            u32::MAX,
            0xaaaa_aaaa,
            0x5555_5555,
            0x8000_0000,
            0x7fff_ffff,
            0x0123_4567,
            0x89ab_cdef,
        ];

        for circuit_seed in [0x3200u64, 0x3201, 0x3202, 0x3203] {
            let original = circuit_32(circuit_seed, 96);
            for rg_freq in [1usize, 2, 5, 17] {
                let layout_seed = circuit_seed ^ ((rg_freq as u64) << 40) ^ 0xfe15_7a;
                let mut layout_rng = StdRng::seed_from_u64(layout_seed);
                let transformed = feistalize(&original, 32, rg_freq, &mut layout_rng);
                let context = format!("circuit_seed={circuit_seed:#x} rg_freq={rg_freq}");

                for &x in &patterns {
                    for &y in &patterns {
                        for &z in &[0u32, u32::MAX, 0xa5a5_5a5a] {
                            assert_middle_block(&transformed, &original, x, y, z, &context);
                        }
                    }
                }

                let mut input_rng = StdRng::seed_from_u64(layout_seed ^ 0x1a2b_3c4d);
                for _ in 0..512 {
                    assert_middle_block(
                        &transformed,
                        &original,
                        input_rng.random::<u32>(),
                        input_rng.random::<u32>(),
                        input_rng.random::<u32>(),
                        &context,
                    );
                }
            }
        }
    }

    #[test]
    fn slice_zero_thirty_two_wire_zero_slice_matches_and_off_slice_changes() {
        let patterns = [
            0u32,
            1,
            u32::MAX,
            0xaaaa_aaaa,
            0x5555_5555,
            0x8000_0000,
            0x0123_4567,
            0x89ab_cdef,
        ];

        for circuit_seed in [0x4200u64, 0x4201] {
            let original = circuit_32(circuit_seed, 80);
            let mut layout_rng = StdRng::seed_from_u64(circuit_seed ^ 0x510c_e0);
            let transformed = feistalize_with_slice_zero(&original, 32, 3, &mut layout_rng);

            for &x in &patterns {
                let cx = Gate::evaluate_index_list_256(U256::from(x), &original.gates).low_u32();
                assert_eq!(middle_block(&transformed, x, 0, 0), cx);
                for &(y, z) in &[(1u32, 0u32), (0, 1), (0xa5a5_5a5a, 0), (0, 0x5a5a_a5a5)] {
                    assert_ne!(middle_block(&transformed, x, y, z), y ^ cx);
                }
            }

            let mut input_rng = StdRng::seed_from_u64(circuit_seed ^ 0x7123);
            for _ in 0..128 {
                let x = input_rng.random::<u32>();
                let y = input_rng.random::<u32>();
                let z = input_rng.random::<u32>() | 1;
                let cx = Gate::evaluate_index_list_256(U256::from(x), &original.gates).low_u32();
                assert_ne!(middle_block(&transformed, x, y, z), y ^ cx);
            }
        }
    }

    #[test]
    fn slice_zero_random_thirty_two_wire_public_slice_fixed_and_off_slice_moves_x() {
        for seed in [0x6300u64, 0x6301, 0x6302, 0x6303] {
            let mut rng = StdRng::seed_from_u64(seed);
            let block =
                slice_zero_random_preblock(32, SLICE_ZERO_RANDOM_GATES_PER_WIRE * 32, &mut rng);
            let public_y = packed_words_to_u32(&block.public_y);
            let public_z = packed_words_to_u32(&block.public_z);

            for &x in &[0u32, 1, u32::MAX, 0xaaaa_aaaa, 0x0123_4567] {
                assert_eq!(eval_x_block(&block.circuit, x, public_y, public_z), x);
            }

            for bit in 0..32 {
                let y_delta = eval_x_block(&block.circuit, 0, public_y ^ (1u32 << bit), public_z);
                let z_delta = eval_x_block(&block.circuit, 0, public_y, public_z ^ (1u32 << bit));
                assert!(
                    y_delta.count_ones() >= 4,
                    "seed={seed:#x} y bit={bit} delta={y_delta:#010x}"
                );
                assert!(
                    z_delta.count_ones() >= 4,
                    "seed={seed:#x} z bit={bit} delta={z_delta:#010x}"
                );
            }
        }
    }
}

#[cfg(test)]
mod slice_zero_random_large_wire_tests {
    use super::*;
    use crate::circuit::circuit::{Gate, U1024};
    use primitive_types::U512;
    use rand::{SeedableRng, rngs::StdRng};

    fn packed_words_to_u512(words: &[u64], n: usize) -> U512 {
        let mut out = U512::zero();
        for bit in 0..n {
            if packed_bit(words, bit) {
                out |= U512::one() << bit;
            }
        }
        out
    }

    fn packed_words_to_u1024(words: &[u64], n: usize) -> U1024 {
        let mut out = U1024::zero();
        for bit in 0..n {
            if packed_bit(words, bit) {
                out = out | (U1024::one() << bit);
            }
        }
        out
    }

    fn pattern_u512(n: usize, mode: usize) -> U512 {
        let mut out = U512::zero();
        for bit in 0..n {
            let set = match mode {
                0 => false,
                1 => true,
                2 => bit % 2 == 0,
                _ => bit % 3 == 1,
            };
            if set {
                out |= U512::one() << bit;
            }
        }
        out
    }

    fn pattern_u1024(n: usize, mode: usize) -> U1024 {
        let mut out = U1024::zero();
        for bit in 0..n {
            let set = match mode {
                0 => false,
                1 => true,
                2 => bit % 2 == 0,
                _ => bit % 3 == 1,
            };
            if set {
                out = out | (U1024::one() << bit);
            }
        }
        out
    }

    fn low_weight_u512(value: U512, n: usize) -> u32 {
        let mut weight = 0;
        for bit in 0..n {
            if ((value >> bit) & U512::one()) == U512::one() {
                weight += 1;
            }
        }
        weight
    }

    fn low_weight_u1024(value: U1024, n: usize) -> u32 {
        let mut weight = 0;
        for bit in 0..n {
            if ((value >> bit) & U1024::one()) == U1024::one() {
                weight += 1;
            }
        }
        weight
    }

    #[test]
    fn slice_zero_random_n128_default_32n_fixes_public_slice_and_moves_x() {
        let n = 128;
        let mut rng = StdRng::seed_from_u64(0x1280_32);
        let block = slice_zero_random_preblock(n, SLICE_ZERO_RANDOM_GATES_PER_WIRE * n, &mut rng);
        assert_eq!(
            block.circuit.gates.len(),
            SLICE_ZERO_RANDOM_GATES_PER_WIRE * n
        );

        let public_y = packed_words_to_u512(&block.public_y, n);
        let public_z = packed_words_to_u512(&block.public_z, n);
        let mask = (U512::one() << n) - U512::one();

        for mode in 0..4 {
            let x = pattern_u512(n, mode);
            let input = x | (public_y << n) | (public_z << (2 * n));
            let output = Gate::evaluate_index_list_512(input, &block.circuit.gates);
            assert_eq!(output & mask, x);
            assert_eq!((output >> n) & mask, public_y);
            assert_eq!((output >> (2 * n)) & mask, public_z);
        }

        for bit in 0..n {
            let y_input = (public_y ^ (U512::one() << bit)) << n;
            let z_input = (public_z ^ (U512::one() << bit)) << (2 * n);
            let y_delta = Gate::evaluate_index_list_512(
                y_input | (public_z << (2 * n)),
                &block.circuit.gates,
            ) & mask;
            let z_delta =
                Gate::evaluate_index_list_512((public_y << n) | z_input, &block.circuit.gates)
                    & mask;
            assert!(
                low_weight_u512(y_delta, n) >= 4,
                "n=128 y bit={bit} delta_weight={}",
                low_weight_u512(y_delta, n)
            );
            assert!(
                low_weight_u512(z_delta, n) >= 4,
                "n=128 z bit={bit} delta_weight={}",
                low_weight_u512(z_delta, n)
            );
        }
    }

    #[test]
    fn slice_zero_random_n256_default_32n_fixes_public_slice_and_moves_x() {
        let n = 256;
        let mut rng = StdRng::seed_from_u64(0x2560_32);
        let block = slice_zero_random_preblock(n, SLICE_ZERO_RANDOM_GATES_PER_WIRE * n, &mut rng);
        assert_eq!(
            block.circuit.gates.len(),
            SLICE_ZERO_RANDOM_GATES_PER_WIRE * n
        );

        let public_y = packed_words_to_u1024(&block.public_y, n);
        let public_z = packed_words_to_u1024(&block.public_z, n);
        let mask = (U1024::one() << n) - U1024::one();

        for mode in 0..4 {
            let x = pattern_u1024(n, mode);
            let input = x | (public_y << n) | (public_z << (2 * n));
            let output = Gate::evaluate_index_list_1024(input, &block.circuit.gates);
            assert_eq!(output & mask, x);
            assert_eq!((output >> n) & mask, public_y);
            assert_eq!((output >> (2 * n)) & mask, public_z);
        }

        for bit in 0..n {
            let y_input = (public_y ^ (U1024::one() << bit)) << n;
            let z_input = (public_z ^ (U1024::one() << bit)) << (2 * n);
            let y_delta = Gate::evaluate_index_list_1024(
                y_input | (public_z << (2 * n)),
                &block.circuit.gates,
            ) & mask;
            let z_delta =
                Gate::evaluate_index_list_1024((public_y << n) | z_input, &block.circuit.gates)
                    & mask;
            assert!(
                low_weight_u1024(y_delta, n) >= 4,
                "n=256 y bit={bit} delta_weight={}",
                low_weight_u1024(y_delta, n)
            );
            assert!(
                low_weight_u1024(z_delta, n) >= 4,
                "n=256 z bit={bit} delta_weight={}",
                low_weight_u1024(z_delta, n)
            );
        }
    }
}

#[cfg(test)]
mod feistal_fixed_point_n_tests {
    use super::*;
    use crate::circuit::circuit::Gate;

    fn decode(state: &FeistalState, physical: usize) -> (usize, usize) {
        let mut x = 0usize;
        let mut y = 0usize;
        for i in 0..state.sharing.n {
            let (p0, p1) = state.sharing.pairs[i];
            let pair = ((physical >> p0) & 1) ^ ((physical >> p1) & 1);
            x |= (pair ^ ((physical >> state.free[i]) & 1)) << i;
            y |= pair << state.q[i];
        }
        (x, y)
    }

    #[test]
    fn n_tilde_supports_q_fixed_points_without_moving_carriers() {
        let state = FeistalState {
            sharing: GadgetState {
                n: 3,
                pairs: vec![(0, 1), (3, 4), (6, 7)],
            },
            free: vec![2, 5, 8],
            q: vec![0, 1, 2],
        };
        let original_pairs = state.sharing.pairs.clone();
        let original_free = state.free.clone();
        let mut gates = Vec::new();
        emit_feistal_n(&state, &mut gates);
        assert_eq!(state.sharing.pairs, original_pairs);
        assert_eq!(state.free, original_free);
        for input in 0..512usize {
            let (x, y) = decode(&state, input);
            let output = Gate::evaluate_index_list(input, &gates);
            assert_eq!(decode(&state, output), (x, y ^ x));
        }
    }
}
