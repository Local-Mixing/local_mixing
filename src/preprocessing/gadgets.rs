use crate::circuit::CircuitSeq;
use crate::circuit::shoot_random_gate;
use crate::circuit::xgate::XGate;
use itertools::Itertools;
use rand::{Rng, prelude::SliceRandom};
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
pub const SLICE_ZERO_CCNOT_GATES_PER_WIRE: usize = 10;

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
// Live (pending) fanout of every wire given the gates emitted so far: reads of a wire since it
// was last written. A gate [t, b, c] reads b and c (+1 each) and rewrites t (reset to 0).
fn live_fanouts(total: usize, out: &[[u16; 3]]) -> Vec<u32> {
    let mut lf = vec![0u32; total];
    for g in out {
        lf[g[1] as usize] += 1;
        lf[g[2] as usize] += 1;
        lf[g[0] as usize] = 0;
    }
    lf
}

// Available wires sorted by current live fanout ascending, random tiebreak. Helper wires are
// read-only scratch (the transvection/NOT gadgets restore them), so any wires are correct; we
// prefer the lowest-fanout ones to spread read-load and avoid concentrating fanout (#3).
fn helpers_by_low_fanout(total: usize, exclude: &[usize], out: &[[u16; 3]]) -> Vec<usize> {
    let lf = live_fanouts(total, out);
    let mut avail: Vec<usize> = (0..total).filter(|w| !exclude.contains(w)).collect();
    avail.shuffle(&mut rand::rng());
    avail.sort_by_key(|&w| lf[w]);
    avail
}

fn pick_two_helpers(total: usize, exclude: &[usize], out: &[[u16; 3]]) -> (usize, usize) {
    let a = helpers_by_low_fanout(total, exclude, out);
    (a[0], a[1])
}

fn pick_three_helpers(total: usize, exclude: &[usize], out: &[[u16; 3]]) -> (usize, usize, usize) {
    let a = helpers_by_low_fanout(total, exclude, out);
    (a[0], a[1], a[2])
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
    let (h0, h1, h2) = pick_three_helpers(total, &[wire], out);
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
    shoot_random_gate(&mut main, rounds);
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
                    let s = state.pairs[i].0;
                    let p = state.pairs[i].1;
                    let r1 = loop {
                        let w = rng.random_range(0..total);
                        if w != s && w != p {
                            break w;
                        }
                    };
                    let r2 = loop {
                        let w = rng.random_range(0..total);
                        if w != s && w != p && w != r1 {
                            break w;
                        }
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
            let (h1, h2) = pick_two_helpers(total, &[v, pw], &out);
            emit_transvection(v, pw, h1, h2, &mut out);
        } else if pw == v {
            let (h1, h2) = pick_two_helpers(total, &[v, sw], &out);
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
    let (h1, h2) = pick_two_helpers(total, &[target, aux_factor, other_factor], out);
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

    // Split the X block into disjoint target / control pools. X-control pairs target the target
    // pool and read the control pool; because no pair's target is any pair's X-control, the
    // on-slice operations (each a ^= xc) all COMMUTE, so the pairs (each present twice) cancel to
    // identity regardless of where G and G' land after shuffling.
    let mut xs: Vec<usize> = (0..n).collect();
    xs.shuffle(rng);
    let split = (n / 2).clamp(1, n - 1);
    let target_pool: Vec<usize> = xs[..split].to_vec();
    let control_pool: Vec<usize> = xs[split..].to_vec();

    // The block has exactly `gate_count` gates: a fraction are X-control pairs (for X fanout), the
    // rest are both-controls-in-Y/Z scrambling gates. Pairs are 2 gates each, so we emit
    // `num_pairs` pairs + `gate_count - 2*num_pairs` singles.
    let pairs_possible = !control_pool.is_empty() && one_controls.len() >= 2;
    let num_pairs = if pairs_possible { gate_count / 8 } else { 0 };
    let num_singles = gate_count - 2 * num_pairs;

    let mut gates: Vec<[u16; 3]> = Vec::with_capacity(gate_count);
    // Both controls in Y/Z, polarity opposite the fixed value: active ^= (zero OR NOT one) =
    // (0 OR NOT 1) = 0 on the public slice (no effect); scrambles x off-slice.
    for _ in 0..num_singles {
        let active = rng.random_range(0..n) as u16;
        let pos = zero_controls[rng.random_range(0..zero_controls.len())] as u16;
        let neg = one_controls[rng.random_range(0..one_controls.len())] as u16;
        gates.push([active, pos, neg]);
    }
    // X-control pairs G, G': same target a (target pool) and same X control xc (control pool,
    // positive), with two DIFFERENT value-1 second controls (negated -> 0 on slice). On the slice
    // each is `a ^= (xc OR 0) = a ^= xc`, so the pair cancels; off-slice the differing second
    // controls make the two gates differ, scrambling x while both read an X wire (the X fanout).
    for _ in 0..num_pairs {
        let a = target_pool[rng.random_range(0..target_pool.len())] as u16;
        let xc = control_pool[rng.random_range(0..control_pool.len())] as u16;
        let w1 = one_controls[rng.random_range(0..one_controls.len())] as u16;
        let w2 = loop {
            let w = one_controls[rng.random_range(0..one_controls.len())] as u16;
            if w != w1 {
                break w;
            }
        };
        gates.push([a, xc, w1]);
        gates.push([a, xc, w2]);
    }
    // Random order among all gates. Both-Y/Z gates are identity on the slice for any position, and
    // the X-control pairs commute (disjoint pools), so the shuffled block is still identity there.
    gates.shuffle(rng);

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
        // One control in X or Y (0..2n), the other in Z (2n..3n), both distinct from the target.
        // The target stays in Z, so these gates never write X or Y and cannot affect the middle
        // (Y) block; the controls are read-only, so their placement is free.
        let c_xy = rng.random_range(0..2 * n) as u16;
        let c_z = loop {
            let w = rng.random_range(2 * n..total) as u16;
            if w != active {
                break w;
            }
        };
        let (c1, c2) = if rng.random_bool(0.5) {
            (c_xy, c_z)
        } else {
            (c_z, c_xy)
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
            let (p0, p1) = state.sharing.pairs[i];
            let r1 = loop {
                let w = rng.random_range(0..3 * n);
                if w != p0 && w != p1 {
                    break w;
                }
            };
            let r2 = loop {
                let w = rng.random_range(0..3 * n);
                if w != p0 && w != p1 && w != r1 {
                    break w;
                }
            };
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
                let (h1, h2) = pick_two_helpers(3 * n, &[dst, src], out);
                emit_transvection(dst, src, h1, h2, out);
            }
            continue;
        }
        for source in [
            state.sharing.pairs[x].0,
            state.sharing.pairs[x].1,
            state.free[x],
        ] {
            let (h1, h2) = pick_two_helpers(3 * n, &[yc, hf, source], out);
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
        let (h1, h2) = pick_two_helpers(total, &[dst, src], out);
        emit_transvection(dst, src, h1, h2, out);
    }
}

/// Build the one-way layout on 3n wires. For input (x,y,z), the middle block
/// outputs y ^ C(x), the first block outputs D(C(x)) for a random same-size D,
/// and the final block is auxiliary output.
pub fn feistalize(main: &CircuitSeq, n: usize, rg_freq: usize, rng: &mut impl Rng) -> CircuitSeq {
    // SymmetricCD: make circuit D equal to C reversed. SymmetricG (requires SymmetricCD): also
    // make the right SG/RG gadget block the mirror image of the left. The two Z bookends stay
    // independently random in both cases.
    let sym_cd = std::env::var("SymmetricCD").is_ok();
    let sym_g = sym_cd && std::env::var("SymmetricG").is_ok();
    feistalize_inner(main, n, rg_freq, rng, sym_cd, sym_g)
}

fn feistalize_inner(
    main: &CircuitSeq,
    n: usize,
    rg_freq: usize,
    rng: &mut impl Rng,
    sym_cd: bool,
    sym_g: bool,
) -> CircuitSeq {
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
        let (h1, h2) = pick_two_helpers(total, &[y, s], &out);
        emit_transvection(y, s, h1, h2, &mut out);
        let (h1, h2) = pick_two_helpers(total, &[t, y], &out);
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
    shoot_random_gate(&mut c, rounds);
    // Sharing-wire layout before SG(c); the only part SG3+RG mutates is `pairs` (RG swaps).
    let pairs_pre = state.sharing.pairs.clone();
    let sgc_start = out.len();
    emit_sg3_rg_block(&c, &mut state, rg_freq, rng, &mut out);
    let sgc_end = out.len();
    emit_feistal_n(&state, &mut out);
    if sym_g {
        // Right SG/RG block = mirror image (reversed gate order) of the left block. Reversing the
        // gates inverts SG(c)'s value transformations and RG permutations, so the sharing layout
        // returns to its pre-SG(c) state; restore `state.pairs` accordingly so `decode` matches.
        let mut mirror: Vec<[u16; 3]> = out[sgc_start..sgc_end].to_vec();
        mirror.reverse();
        out.extend(mirror);
        state.sharing.pairs = pairs_pre;
    } else {
        // SymmetricCD: D = C reversed; otherwise an independent random D. D never affects the
        // middle block (it acts on the x-shares after feistal_n has already written y).
        let d = if sym_cd {
            CircuitSeq {
                gates: c.gates.iter().rev().copied().collect(),
            }
        } else {
            random_circuit_with_rng(n, main.gates.len(), rng)
        };
        emit_sg3_rg_block(&d, &mut state, rg_freq, rng, &mut out);
    }
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

#[cfg(test)]
mod feistal_tests {
    use super::*;
    use crate::circuit::Gate;
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

    fn middle_is_y_plus_cx(circuit: &CircuitSeq, main: &CircuitSeq, n: usize) -> bool {
        let mask = (1usize << n) - 1;
        (0..(1usize << (3 * n))).all(|input| {
            let x = input & mask;
            let y = (input >> n) & mask;
            (circuit.evaluate(input) >> n) & mask == y ^ main.evaluate(x)
        })
    }

    #[test]
    fn symmetric_cd_middle_block_correct() {
        let n = 3;
        let main = CircuitSeq {
            gates: vec![[0, 1, 2], [2, 0, 1], [1, 2, 0]],
        };
        for seed in 0x7100u64..0x7106 {
            let mut rng = StdRng::seed_from_u64(seed);
            let circuit = feistalize_inner(&main, n, 1, &mut rng, true, false);
            assert!(
                middle_is_y_plus_cx(&circuit, &main, n),
                "sym_cd seed={seed:#x}"
            );
        }
    }

    #[test]
    fn symmetric_g_middle_block_correct() {
        let n = 3;
        let main = CircuitSeq {
            gates: vec![[0, 1, 2], [2, 0, 1], [1, 2, 0]],
        };
        for seed in 0x7200u64..0x7206 {
            let mut rng = StdRng::seed_from_u64(seed);
            let circuit = feistalize_inner(&main, n, 1, &mut rng, true, true);
            assert!(
                middle_is_y_plus_cx(&circuit, &main, n),
                "sym_g seed={seed:#x}"
            );
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
    use crate::circuit::Gate;
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
    use crate::circuit::Gate;
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
    use crate::circuit::{Gate, U1024};
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
        assert_eq!(block.circuit.gates.len(), 32 * n);

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
        assert_eq!(block.circuit.gates.len(), 32 * n);

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
    use crate::circuit::Gate;

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

fn emit_gadget_x(state: &GadgetState, gate: [u16; 3], out: &mut Vec<XGate>) {
    let [a, b, c] = gate.map(|wire| wire as usize);
    let (b0, b1) = state.pairs[b];
    let (c0, c1) = state.pairs[c];
    emit_shared_g57_frag2(state.pairs[a].0, b0, b1, c0, c1, out);
}

/// Homomorphically share ONE arbitrary XGate into the two-share gadget,
/// targeting the target value's `.0` (share) carrier and leaving `.1` (pad)
/// as a mask. A positive control literal for value v expands as (v0 + v1), a
/// negative one as (1 + v0 + v1); the gate's conjunction becomes an ANF over
/// the carriers and each monomial is emitted as one fragment. Distinct
/// control values have disjoint carriers, so no fragment ever reads both
/// carriers of one value and every prefix stays first-order masked. Unlike
/// `emit_gadget_x` (g57-only, four fixed fragments) this handles g57, CNOT,
/// CCNOT and any conjunction fragment uniformly — the mpmct1-ingest path.
fn emit_shared_xgate2(state: &GadgetState, gate: &XGate, out: &mut Vec<XGate>) {
    let logical_target = gate.target as usize;
    debug_assert!(logical_target < state.n);
    let target_carrier = state.pairs[logical_target].0 as u16;

    let mut terms: Vec<Vec<(u16, bool)>> = vec![Vec::new()];
    for &(logical_control, positive) in &gate.ctrls {
        let logical_control = logical_control as usize;
        debug_assert_ne!(logical_control, logical_target);
        let (c0, c1) = state.pairs[logical_control];
        let carriers = [c0 as u16, c1 as u16];
        let previous = std::mem::take(&mut terms);
        for term in previous {
            if !positive {
                toggle_anf_term(&mut terms, term.clone());
            }
            for carrier in carriers {
                let mut next = term.clone();
                next.push((carrier, true));
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
        // Empty term => constant 1 => X gate (XGate::conj returns that).
        if let Some(fragment) = XGate::conj(target_carrier, term) {
            out.push(fragment);
        }
    }
}

/// Number of CG realizations in the menu drawn from by [`emit_cg_menu`].
const CG_VARIANTS: u32 = 7;

/// Emit ONE realization of the shared g57 update `A ^= B OR !C` (values
/// two-share: X = x0 XOR x1), chosen by `variant`, with the carrier roles
/// (which carrier of A is targeted, share/pad order of B and of C)
/// randomized per call. Every variant stays inside the agreed vocabulary —
/// g57s and pure conjunction fragments with one or two controls of any
/// polarity, never a bare X (an X census would count the source gates) —
/// leaves all non-target carriers wire-level unchanged, and never mutates
/// the sharing state. Writing f = B OR !C = 1 + C + B*C over GF(2):
///   0 "collapse both": 4 CNOT + 1 g57 — b0^=b1 and c0^=c1 put B and C on
///     single wires, one g57 fires f, both collapses restored.
///   1 "collapse c": 3 CNOT-kind + 2 g57 — with c0=C the pair
///     [a,b0,c0]+[a,b1,c0] sums to B*C and a^=!c0 adds 1+C.
///   2 "collapse b": 2 CNOT + 1 g57 + 1 fragment — with b0=B, [a,b0,c0]
///     gives 1+c0+B*c0 and the fragment !b0&c1 gives c1+B*c1.
///   3 "linear tail": !CNOT + CNOT + 4 g57 — a^=!c0 and a^=c1 give 1+C,
///     the quad [a,bi,cj] sums to B*C; no collapse, and no gate reads both
///     carriers of one value (masking-safe, like the ESOP).
///   4 legacy GADGET: 6 g57 — the classic nonlinear network, C's second
///     carrier borrowed as restored scratch.
///   5 "quad+pair": 5 g57 + 1 fragment — the quad gives B*C, then
///     (!c0&c1) + [a,c1,c0] = 1 + C (a g57 pair with one complement folded
///     into a fragment). All-g57 apart from one fragment, no collapse.
///   6 four-cube ESOP: 4 fragments (the previous fixed SG; masking-safe).
/// A per-gate uniform draw over these breaks the fixed-period SG
/// fingerprint and varies local gate-type statistics while keeping the
/// body g57-rich (DB-warm) with only sprinkled CNOTs.
fn emit_cg_variant(
    state: &GadgetState,
    gate: [u16; 3],
    variant: u32,
    rng: &mut impl Rng,
    out: &mut Vec<XGate>,
) {
    let [a, b, c] = gate.map(|w| w as usize);
    assert!(
        a != b && a != c && b != c,
        "CG operands must be distinct values"
    );
    let (pa, pb, pc) = (state.pairs[a], state.pairs[b], state.pairs[c]);
    let at = (if rng.random_bool(0.5) { pa.0 } else { pa.1 }) as u16;
    let (b0, b1) = if rng.random_bool(0.5) {
        (pb.0, pb.1)
    } else {
        (pb.1, pb.0)
    };
    let (c0, c1) = if rng.random_bool(0.5) {
        (pc.0, pc.1)
    } else {
        (pc.1, pc.0)
    };
    let (b0, b1, c0, c1) = (b0 as u16, b1 as u16, c0 as u16, c1 as u16);
    let g57 = |t: u16, x: u16, y: u16| XGate::from_g57([t, x, y]);
    let ncnot = |t: u16, s: u16| XGate::conj(t, [(s, false)]).expect("distinct carriers");
    let frag = |t: u16, l: [(u16, bool); 2]| XGate::conj(t, l).expect("distinct carriers");
    match variant {
        0 => {
            out.push(XGate::cnot(b0, b1));
            out.push(XGate::cnot(c0, c1));
            out.push(g57(at, b0, c0));
            out.push(XGate::cnot(c0, c1));
            out.push(XGate::cnot(b0, b1));
        }
        1 => {
            out.push(XGate::cnot(c0, c1));
            out.push(g57(at, b0, c0));
            out.push(g57(at, b1, c0));
            out.push(ncnot(at, c0));
            out.push(XGate::cnot(c0, c1));
        }
        2 => {
            out.push(XGate::cnot(b0, b1));
            out.push(g57(at, b0, c0));
            out.push(frag(at, [(b0, false), (c1, true)]));
            out.push(XGate::cnot(b0, b1));
        }
        3 => {
            out.push(ncnot(at, c0));
            out.push(XGate::cnot(at, c1));
            out.push(g57(at, b0, c0));
            out.push(g57(at, b1, c0));
            out.push(g57(at, b0, c1));
            out.push(g57(at, b1, c1));
        }
        4 => {
            let map = [at, 0, 0, c0, c1, b0, b1];
            for &[ga, gb, gc] in &GADGET {
                out.push(g57(map[ga as usize], map[gb as usize], map[gc as usize]));
            }
        }
        5 => {
            out.push(g57(at, b0, c0));
            out.push(g57(at, b1, c0));
            out.push(g57(at, b0, c1));
            out.push(g57(at, b1, c1));
            out.push(frag(at, [(c0, false), (c1, true)]));
            out.push(g57(at, c1, c0));
        }
        _ => emit_shared_g57_frag2(
            at as usize,
            b0 as usize,
            b1 as usize,
            c0 as usize,
            c1 as usize,
            out,
        ),
    }
}

/// One CG, drawn uniformly from the menu (see [`emit_cg_variant`]).
fn emit_cg_menu(state: &GadgetState, gate: [u16; 3], rng: &mut impl Rng, out: &mut Vec<XGate>) {
    let variant = rng.random_range(0..CG_VARIANTS);
    emit_cg_variant(state, gate, variant, rng, out);
}

/// Recognize an XGate that IS a g57 — complemented with exactly one negative
/// and one positive control — returning [target, x, y] with fires = x OR !y.
fn as_g57_triple(g: &XGate) -> Option<[u16; 3]> {
    if !g.comp || g.ctrls.len() != 2 {
        return None;
    }
    let (w0, p0) = g.ctrls[0];
    let (w1, p1) = g.ctrls[1];
    match (p0, p1) {
        (false, true) => Some([g.target, w0, w1]),
        (true, false) => Some([g.target, w1, w0]),
        _ => None,
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

// ---------------------------------------------------------------------------
// Deferred-mask encoding (RG4 / "split-RG3") — docs/NONLINEAR_SHARE_ENCODING.md
//
// The XOR pair makes every logical value a degree-1 function of the wires at
// every snapshot; that is the root of the progress diagonal that survives
// mixing. RG4 splits the RG3 compensated pair-write in time: a polynomial
// with quadratic content u*w is XORed into ONE carrier now and compensated
// only later (on either carrier of the value's then-current pair), so that
// between the two events the value reads at degree >= 2 with random support.
// The ledger below tracks pending polynomials, intercepts writes to their
// sources (flush, or extend into a tower when the write is registered), and
// keeps the invariant  v_i = c_i0 ^ c_i1 ^ XOR_t phi_t(current wires)  true
// at every emitted-unit boundary.
// ---------------------------------------------------------------------------

/// GF(2) multilinear polynomial over physical wire contents: an XOR of
/// monomials, each a sorted wire set; the empty monomial is the constant 1.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WirePoly {
    monomials: Vec<Vec<u16>>,
}

impl WirePoly {
    pub fn from_monomials(monomials: impl IntoIterator<Item = Vec<u16>>) -> WirePoly {
        let mut p = WirePoly::default();
        for m in monomials {
            p.toggle(m);
        }
        p
    }

    /// XOR one monomial in (x*x = x; duplicate monomials cancel).
    fn toggle(&mut self, mut m: Vec<u16>) {
        m.sort_unstable();
        m.dedup();
        if let Some(pos) = self.monomials.iter().position(|e| *e == m) {
            self.monomials.swap_remove(pos);
        } else {
            self.monomials.push(m);
        }
    }

    fn contains_wire(&self, w: u16) -> bool {
        self.monomials.iter().any(|m| m.contains(&w))
    }

    // The next three are the primitives for in-flight mask TOWERS (degree-3+
    // deferred masks by compensated source rewrites). v1 uses the cascade-free
    // ledger, which keeps masks at degree 2 and does not extend across source
    // writes, so they are unused for now; they are retained (and exercised via
    // `emit_poly_add`'s cubic path) for the deferred tower thread in
    // docs/NONLINEAR_SHARE_ENCODING.md.
    #[allow(dead_code)]
    fn support(&self) -> Vec<u16> {
        let mut s: Vec<u16> = self.monomials.iter().flatten().copied().collect();
        s.sort_unstable();
        s.dedup();
        s
    }

    #[allow(dead_code)]
    fn max_monomial_len(&self) -> usize {
        self.monomials.iter().map(|m| m.len()).max().unwrap_or(0)
    }

    /// Re-express the polynomial across a registered write `wire ^= psi`:
    /// the pre-write value is (post-write wire) ^ psi, so every monomial
    /// containing `wire` keeps its literal form (now reading the post-write
    /// wire) and XORs in psi * (monomial \ wire).
    #[allow(dead_code)]
    fn substitute(&mut self, wire: u16, psi: &WirePoly) {
        debug_assert!(
            !psi.contains_wire(wire),
            "a registered write must not read its own target"
        );
        let affected: Vec<Vec<u16>> = self
            .monomials
            .iter()
            .filter(|m| m.contains(&wire))
            .cloned()
            .collect();
        for m in affected {
            let rest: Vec<u16> = m.iter().copied().filter(|&w| w != wire).collect();
            for pm in &psi.monomials {
                let mut prod = rest.clone();
                prod.extend_from_slice(pm);
                self.toggle(prod);
            }
        }
    }

    /// Evaluate on a packed wire state (wire w = bit w), for tests/debug.
    pub fn eval_u64(&self, state: u64) -> bool {
        self.monomials
            .iter()
            .filter(|m| m.iter().all(|&w| (state >> w) & 1 != 0))
            .count()
            % 2
            == 1
    }
}

/// Emit vocabulary-legal gates adding `poly` (over current wire contents) to
/// `target`: g57s, 1–2-control conjunction fragments of any polarity, and
/// ±CNOTs; cubic monomials via a borrowed-and-restored scratch wire; never a
/// bare X (constants fold into complemented gates). The realization is drawn
/// fresh per call, so the inject and flush of one pending polynomial share no
/// syntactic signature. Largest monomials go first and every emitted gate
/// clears its top monomial while toggling only strictly smaller ones, so the
/// loop terminates.
fn emit_poly_add(
    target: u16,
    poly: &WirePoly,
    total: usize,
    rng: &mut impl Rng,
    out: &mut Vec<XGate>,
) {
    debug_assert!(
        !poly.contains_wire(target),
        "mask polynomial reads its own target"
    );
    let frag = |t: u16, u: u16, pu: bool, w: u16, pw: bool| {
        XGate::conj(t, [(u, pu), (w, pw)]).expect("distinct mask wires")
    };
    let mut p = poly.clone();
    while let Some(m) = p.monomials.iter().max_by_key(|m| m.len()).cloned() {
        match m.len() {
            0 => {
                // Constant 1 alone: fold through a random wire, no bare X.
                let u = random_wire_except(total, &[target as usize], rng) as u16;
                out.push(XGate::conj(target, [(u, false)]).expect("distinct wires"));
                out.push(XGate::cnot(target, u));
                p.toggle(vec![]);
            }
            1 => {
                let u = m[0];
                if p.monomials.iter().any(|e| e.is_empty()) {
                    out.push(XGate::conj(target, [(u, false)]).expect("distinct wires"));
                    p.toggle(vec![]);
                } else {
                    out.push(XGate::cnot(target, u));
                }
                p.toggle(m);
            }
            2 => {
                let (u, w) = (m[0], m[1]);
                match rng.random_range(0..6u32) {
                    0 => out.push(frag(target, u, true, w, true)),
                    1 => {
                        // g57: adds 1 + w + u*w
                        out.push(XGate::from_g57([target, u, w]));
                        p.toggle(vec![]);
                        p.toggle(vec![w]);
                    }
                    2 => {
                        out.push(XGate::from_g57([target, w, u]));
                        p.toggle(vec![]);
                        p.toggle(vec![u]);
                    }
                    3 => {
                        out.push(frag(target, u, false, w, true));
                        p.toggle(vec![w]);
                    }
                    4 => {
                        out.push(frag(target, u, true, w, false));
                        p.toggle(vec![u]);
                    }
                    _ => {
                        out.push(frag(target, u, false, w, false));
                        p.toggle(vec![]);
                        p.toggle(vec![u]);
                        p.toggle(vec![w]);
                    }
                }
                p.toggle(m);
            }
            3 => {
                // Borrowed-and-restored scratch h: adds exactly u*v*w
                // ((h ^ uv)w ^ hw = uvw), h untouched at the unit boundary.
                let (u, v, w) = (m[0], m[1], m[2]);
                let h = random_wire_except(
                    total,
                    &[target as usize, u as usize, v as usize, w as usize],
                    rng,
                ) as u16;
                out.push(frag(h, u, true, v, true));
                out.push(frag(target, h, true, w, true));
                out.push(frag(h, u, true, v, true));
                out.push(frag(target, h, true, w, true));
                p.toggle(m);
            }
            d => unreachable!("mask monomial degree {d} exceeds the supported tower cap"),
        }
    }
}

/// Tuning for the deferred-mask encoding. `off()` (coverage 0) reproduces the
/// unmasked gadget exactly. The v1 (cascade-free) ledger keeps masks at
/// degree 2; a workable validated setting is `cov: ~0.75, k: 1..3` — near-full
/// coverage self-limits because each mask needs two unmasked source carriers.
#[derive(Clone, Copy, Debug)]
pub struct MaskConfig {
    /// Target fraction of logical values carrying masks (0 disables). Actual
    /// coverage self-limits below 1: masks are sourced on unmasked values'
    /// carriers, so a pool must remain unmasked.
    pub cov: f64,
    /// Pending terms per masked value (piling-up: k stacked degree-2 masks
    /// push the best affine readout error toward 1/2).
    pub k: usize,
    /// Reserved for the deferred tower thread (degree-3+ masks); a no-op in
    /// the v1 cascade-free ledger, which keeps masks at degree 2.
    pub depth: usize,
    /// Gaps before the body's end to stop re-injecting (None = max(4, n/5)).
    pub taper: Option<usize>,
}

impl MaskConfig {
    pub fn off() -> MaskConfig {
        MaskConfig {
            cov: 0.0,
            k: 0,
            depth: 2,
            taper: None,
        }
    }
}

const MASK_INJECTS_PER_GAP: usize = 4;

/// The nonlinear RG3 randomizer over two logical SOURCE values, realized on
/// their current carriers: `v_{s0} \/ !v_{s1}` where `v = pair.0 ^ pair.1`.
/// Expanding `X \/ !Y = 1 ^ Y ^ X*Y` with `X = a0^a1`, `Y = b0^b1` gives a
/// degree-2 polynomial in the four carrier wires — the quadratic `X*Y` term
/// is exactly what makes the masked value degree-2 (a single linear source
/// would leave it degree-1 and transparent). Rebuilt from `state.pairs` at
/// every emit, so RG relocations of the sources are tracked automatically.
fn mask_term_poly(s0: usize, s1: usize, state: &GadgetState) -> WirePoly {
    let (a0, a1) = state.pairs[s0];
    let (b0, b1) = state.pairs[s1];
    let (a0, a1, b0, b1) = (a0 as u16, a1 as u16, b0 as u16, b1 as u16);
    // 1 ^ Y ^ X*Y  =  1 ^ b0 ^ b1 ^ (a0^a1)(b0^b1)
    WirePoly::from_monomials([
        vec![],
        vec![b0],
        vec![b1],
        vec![a0, b0],
        vec![a0, b1],
        vec![a1, b0],
        vec![a1, b1],
    ])
}

struct MaskEntry {
    id: u64,
    /// The masked logical value.
    value: usize,
    /// The two logical source values of the RG3 randomizer term.
    sources: (usize, usize),
}

/// Pending-mask bookkeeping for one gadgetize run (deferred-mask / RG4 =
/// "split RG3"). A mask is the nonlinear RG3 term `v_{s0} \/ !v_{s1}` XORed
/// into one carrier of a value `v` at inject and compensated later; between,
/// `v` reads at degree 2. Everything is keyed by LOGICAL VALUE:
///
/// * **Sources are logical values**, not physical wires. RG1/RG2/RG3 preserve
///   every logical value (they only relocate/refresh carriers), so the mask
///   term is invariant under all RG churn — the dominant write traffic never
///   disturbs a mask. Only a CG that *recomputes* a source value does, and
///   that schedule is the input gate list's target column, known in advance.
/// * **Lookahead source selection**: sources are the two unmasked values
///   whose next recomputation (`next_target`) is farthest out (finished
///   outputs are ideal), so inject and its compensation are separated by a
///   long, controlled stretch rather than collapsing back into a plain RG3.
/// * **Reads are peeked, not flushed**: when the virtual circuit uses `v` as
///   a control, the CG must see the true `v`, so the mask is momentarily
///   removed (un-mask), the vanilla CG runs, and the same term is re-applied
///   (re-mask) — the mask keeps its identity and lifetime, and `v` is clean
///   only for that one step.
///
/// **Cascade-free invariant**: a mask is sourced only on currently-unmasked
/// values. Maintained by drawing sources from unmasked values and flushing
/// any mask sourced on `v` before `v` is masked. Then recomputing/flushing a
/// masked value's carriers (every inject, flush, and peek does this) never
/// disturbs another mask's source, so flushes never cascade. In-flight
/// degree-3 towers are deferred (masks stay degree 2), and coverage
/// self-limits below 1 (a source pool must stay unmasked). See
/// docs/NONLINEAR_SHARE_ENCODING.md.
struct MaskLedger {
    k: usize,
    taper: usize,
    /// Values allowed to carry masks (coverage target); the complement is a
    /// permanent source pool.
    eligible: Vec<bool>,
    /// Pending-mask count per value (0 ⇒ unmasked ⇒ usable as a source).
    masked: Vec<usize>,
    /// Sorted target positions per value (the CG-recompute schedule).
    targets: Vec<Vec<usize>>,
    masks: Vec<MaskEntry>,
    next_id: u64,
    injected: u64,
    flushed: u64,
    peeked: u64,
    skipped: u64,
    mask_gates: u64,
    peak_cov: f64,
}

impl MaskLedger {
    fn new(n: usize, cfg: &MaskConfig, targets: Vec<Vec<usize>>, rng: &mut impl Rng) -> MaskLedger {
        let want = if cfg.k == 0 {
            0
        } else {
            ((cfg.cov.clamp(0.0, 1.0) * n as f64).round() as usize).min(n)
        };
        let mut eligible = vec![false; n];
        if want > 0 {
            let mut order: Vec<usize> = (0..n).collect();
            order.shuffle(rng);
            for &v in order.iter().take(want) {
                eligible[v] = true;
            }
        }
        MaskLedger {
            k: if want == 0 { 0 } else { cfg.k },
            taper: cfg.taper.unwrap_or_else(|| (n / 5).max(4)),
            eligible,
            masked: vec![0; n],
            targets,
            masks: Vec::new(),
            next_id: 0,
            injected: 0,
            flushed: 0,
            peeked: 0,
            skipped: 0,
            mask_gates: 0,
            peak_cov: 0.0,
        }
    }

    fn enabled(&self) -> bool {
        self.k > 0
    }

    /// Position of the next CG that recomputes `value` strictly after `pos`
    /// (usize::MAX ⇒ never again — a finished output, the ideal source).
    fn next_target(&self, value: usize, pos: usize) -> usize {
        let ts = &self.targets[value];
        let i = ts.partition_point(|&p| p <= pos);
        ts.get(i).copied().unwrap_or(usize::MAX)
    }

    /// Lookahead source selection: the two currently-unmasked values (≠ `value`)
    /// whose next recomputation is farthest after `pos`. `None` when fewer than
    /// two unmasked values remain (near-full coverage self-limits here).
    fn pick_sources(&self, value: usize, pos: usize) -> Option<(usize, usize)> {
        let mut pool: Vec<(usize, usize)> = (0..self.masked.len())
            .filter(|&y| y != value && self.masked[y] == 0)
            .map(|y| (self.next_target(y, pos), y))
            .collect();
        if pool.len() < 2 {
            return None;
        }
        // Farthest next-target first (usize::MAX = finished output = best).
        pool.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        Some((pool[0].1, pool[1].1))
    }

    /// Emit the mask term for value `v` (sources `s`) onto one of `v`'s current
    /// carriers, chosen at random; used identically for inject, flush, un-mask,
    /// and re-mask (the move is its own inverse over GF(2)).
    fn emit_term(
        &mut self,
        value: usize,
        sources: (usize, usize),
        state: &GadgetState,
        total: usize,
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) {
        let poly = mask_term_poly(sources.0, sources.1, state);
        let (c0, c1) = state.pairs[value];
        let target = if rng.random_bool(0.5) { c0 } else { c1 } as u16;
        let before = out.len();
        emit_poly_add(target, &poly, total, rng, out);
        self.mask_gates += (out.len() - before) as u64;
    }

    /// RG4-inject: one fresh mask on `value`, sources chosen by lookahead from
    /// unmasked values. Returns whether a mask was actually injected.
    fn inject(
        &mut self,
        value: usize,
        pos: usize,
        state: &GadgetState,
        total: usize,
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) -> bool {
        // `value` is about to (remain) masked: nothing may be sourced on it.
        self.flush_sourced_on(value, state, total, rng, out);
        let Some(sources) = self.pick_sources(value, pos) else {
            self.skipped += 1;
            return false;
        };
        self.emit_term(value, sources, state, total, rng, out);
        self.masks.push(MaskEntry {
            id: self.next_id,
            value,
            sources,
        });
        self.next_id += 1;
        self.injected += 1;
        self.masked[value] += 1;
        true
    }

    /// Flush one mask by id (no-op if gone). Cascade-free by the invariant —
    /// the write lands on a masked value's carrier, on which nothing is sourced.
    fn flush_id(
        &mut self,
        id: u64,
        state: &GadgetState,
        total: usize,
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) {
        let Some(pos) = self.masks.iter().position(|e| e.id == id) else {
            return;
        };
        let entry = self.masks.swap_remove(pos);
        self.emit_term(entry.value, entry.sources, state, total, rng, out);
        self.masked[entry.value] -= 1;
        self.flushed += 1;
    }

    /// Flush every pending mask whose source set includes `value` — used
    /// before a CG recomputes `value` (its logical value, hence every mask
    /// term reading it, is about to change).
    fn flush_sourced_on(
        &mut self,
        value: usize,
        state: &GadgetState,
        total: usize,
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) {
        let ids: Vec<u64> = self
            .masks
            .iter()
            .filter(|e| e.sources.0 == value || e.sources.1 == value)
            .map(|e| e.id)
            .collect();
        for id in ids {
            self.flush_id(id, state, total, rng, out);
        }
    }

    /// Before a CG on `gate` (reads `reads`, targets `target_value`): flush
    /// masks sourced on the target (its value is about to change), then
    /// UN-MASK every mask on a read value so the vanilla CG sees the true
    /// value. Pair with [`after_cg`], which re-masks them.
    fn before_cg(
        &mut self,
        reads: &[usize],
        target_value: usize,
        state: &GadgetState,
        total: usize,
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) {
        if !self.enabled() {
            return;
        }
        self.flush_sourced_on(target_value, state, total, rng, out);
        let peek: Vec<(usize, (usize, usize))> = self
            .masks
            .iter()
            .filter(|e| reads.contains(&e.value))
            .map(|e| (e.value, e.sources))
            .collect();
        for (value, sources) in peek {
            self.emit_term(value, sources, state, total, rng, out);
            self.peeked += 1;
        }
    }

    /// Re-mask (undo the peek) every mask on a read value, restoring the
    /// deferred masking the instant the CG has consumed the true value.
    fn after_cg(
        &mut self,
        reads: &[usize],
        state: &GadgetState,
        total: usize,
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) {
        if !self.enabled() {
            return;
        }
        let redo: Vec<(usize, (usize, usize))> = self
            .masks
            .iter()
            .filter(|e| reads.contains(&e.value))
            .map(|e| (e.value, e.sources))
            .collect();
        for (value, sources) in redo {
            self.emit_term(value, sources, state, total, rng, out);
        }
    }

    /// Coverage maintenance: inject toward k terms on every eligible value,
    /// a few per gap, tapering off near the body's end so the final flush
    /// thins out instead of bursting at the decode seam.
    fn top_up(
        &mut self,
        pos: usize,
        remaining_gaps: usize,
        state: &GadgetState,
        total: usize,
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) {
        if !self.enabled() || remaining_gaps <= self.taper {
            return;
        }
        let mut order: Vec<usize> = (0..state.n).collect();
        order.shuffle(rng);
        let mut budget = MASK_INJECTS_PER_GAP;
        for v in order {
            if budget == 0 {
                break;
            }
            if self.eligible[v] && self.masked[v] < self.k {
                if self.inject(v, pos, state, total, rng, out) {
                    budget -= 1;
                }
            }
        }
        self.peak_cov = self.peak_cov.max(self.coverage());
    }

    fn flush_all(
        &mut self,
        state: &GadgetState,
        total: usize,
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) {
        while let Some(id) = self.masks.first().map(|e| e.id) {
            self.flush_id(id, state, total, rng, out);
        }
    }

    /// Fraction of eligible values currently masked (coverage meter).
    fn coverage(&self) -> f64 {
        let elig = self.eligible.iter().filter(|&&e| e).count();
        if elig == 0 {
            return 0.0;
        }
        let covered = (0..self.masked.len())
            .filter(|&v| self.eligible[v] && self.masked[v] > 0)
            .count();
        covered as f64 / elig as f64
    }

    fn report(&self) {
        if self.enabled() {
            println!(
                "[mask] injected={} flushed={} peeked={} skipped={} mask_gates={} peak_cov={:.2}",
                self.injected,
                self.flushed,
                self.peeked,
                self.skipped,
                self.mask_gates,
                self.peak_cov
            );
        }
    }
}

/// Per-value sorted list of positions at which each of the `n` values is the
/// CG target (its logical value is recomputed) — the mask disturbance
/// schedule for lookahead source selection. `target_at(pos)` gives the target
/// value of the CG at each body position `0..count`.
fn target_schedule(count: usize, n: usize, target_at: impl Fn(usize) -> usize) -> Vec<Vec<usize>> {
    let mut sched = vec![Vec::new(); n];
    for pos in 0..count {
        sched[target_at(pos)].push(pos);
    }
    sched
}

// ---------------------------------------------------------------------------
// Product-share encoding ("prod") — the balanced nonlinear share encoding.
//
// Every logical value carries, on top of its XOR pair, k permanent
// ledger-registered multiplicative mask terms sourced on a frozen band of
// extra read-only wires:
//
//   v_i = w_{u(i)} ^ w_{u'(i)} ^ XOR_j (w_{p_j} ^ a_j)(w_{q_j} ^ b_j) ^ c_i
//
// The linear pair is forced by the balance obstruction: a pure 2-wire product
// decode has representation classes of sizes 3 and 1, and no exact reversible
// gadget can conditionally flip between unequal classes (a bijection cannot
// map a 3-class onto a 1-class), with any ancillae, garbage, or re-encoding.
// The product terms are what remove the degree-1 snapshot readability: the
// best affine approximation of one product term errs 1/4; k independent terms
// pile up to 1/2 - 2^-(k+1).
//
// Unlike the deferred-mask (RG4) encoding above, the CG never reconstructs an
// operand ("peek") — the use-point re-exposure that carried the progress
// diagonal. Instead the CG folds the gate's ANF over the operands' FULL
// decodes into the target's carriers: single-value products appear only as
// control conjunctions of emitted fragments, never on a wire. Source-band
// wires are written once at the input port (band fill) and only read after,
// so mask terms are invariant under all RG churn and CG traffic — nothing is
// ever flushed except the final unshare strip. X/NOT and all CG constants are
// ledger-only (c_i), so the encoding emits no bare X anywhere.
// ---------------------------------------------------------------------------

/// Tuning for the product-share encoding. `off()` (k = 0) reproduces the
/// plain gadget bit-identically.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProdConfig {
    /// Base product mask terms per value (0 disables base). Against a
    /// statistical degree-`<deg` adversary the terms pile up (k=1 -> error
    /// 0.25 at deg 2); against the exact-span reconstruction measure a single
    /// term of degree `deg` already hides from every adversary of degree
    /// `< deg`.
    pub k: usize,
    /// Base mask degree = literals per base term (min 2). Degree 2 is the
    /// balanced encoding; `deg = d` hides the value from any reconstruction
    /// adversary of degree `< d`, at the cost of fold fragments up to width
    /// `d * (source-gate arity)`.
    pub deg: usize,
    /// Additional higher-degree ("tower") mask terms per value (0 = none). A
    /// mixed design k deg-`deg` + k_hi deg-`deg_hi` gets statistical strength
    /// from the many low-degree terms AND algebraic hiding up to `deg_hi - 1`
    /// from the tower terms.
    pub k_hi: usize,
    /// Degree of the tower terms (used only when k_hi > 0; default 3).
    pub deg_hi: usize,
    /// Source-band width in wires (0 = auto: ~max(sqrt(4n*ktot), maxdeg+3)).
    /// The band is appended after the 2n carrier wires and is read-only past
    /// the input-port fill.
    pub band: usize,
    /// Re-source moves per inter-SG gap (mask churn; 0 disables).
    pub rsrc: usize,
    /// Maximum control width of the encoding's own emissions (fold fragments,
    /// slot injections). 0 = legacy wide fragments. >= 2 = ladder wider
    /// conjunctions down to this cap over the dedicated zero scratch wires;
    /// at 2 every emitted gate is a g57 or CNOT (the phase-A DB vocabulary,
    /// fully DB-eligible). Narrow mode is exact on the pinned zero-aux slice
    /// (scratch wires enter 0) — the slice-zero contract the new gadget
    /// already lives under.
    pub max_width: usize,
    /// Nonlinear band fill: product terms per band wire (0 = legacy linear
    /// fill). Each band wire gets junk ^ pivot ^ small linear part ^ fill_nl
    /// two-source products whose sources are data wires AND earlier band
    /// wires — the cascade multiplies input-degree up the band while every
    /// fill gate stays a CNOT/g57. The fresh pivot (excluded from the rest of
    /// the wire's transitive support) keeps every band bit exactly balanced.
    /// Enabling prod also emits a mirror fill on the output side.
    pub fill_nl: usize,
    /// Rolling band: band-variable relocations per inter-SG gap (0 = the band
    /// stays on its home wires for the whole body). One roll swaps the
    /// contents of the wire currently holding a random band variable with a
    /// uniformly random other wire — a carrier (RG2's move, extended to the
    /// band) or another band wire — and re-points the ledger (and, for a
    /// carrier partner, `GadgetState::pairs`) at the new locations. Values and
    /// mask terms are untouched: only where they live changes. Without it the
    /// band wires are body-static and so statically identifiable, which is
    /// exactly what a restriction adversary needs.
    pub roll: usize,
    /// Distributed sourcing (0 = the dedicated band; 1 = no band at all).
    ///
    /// The band exists only to guarantee that a registered product term
    /// denotes the SAME runtime bit from injection to strip, and freezing a
    /// dedicated wire set is the cheapest way to get that. It is not the only
    /// way: with the whole gate list in hand the gadgetizer knows every future
    /// write, so a mask can instead be sourced on an ORDINARY CARRIER of some
    /// other value and migrated (the re-source move) before anything writes
    /// it. Then there are no extra wires, no globally quiet wires, and the
    /// mask sources roll with the computation.
    ///
    /// Invariants that make this sound, all asserted in tests:
    /// * no slot names a carrier of its own value (self-reference would put a
    ///   value's own mask literal in its own fold's firing condition);
    /// * at most ONE carrier of any value is named by live slots ("invariant
    ///   S"), so every value always has a free carrier to be written on and
    ///   no fragment can see both carriers of a third value;
    /// * every emitted write to a named wire is preceded by a release (each
    ///   slot naming it is re-sourced first).
    pub src_dist: usize,
    /// Lookahead horizon for distributed sourcing, in SOURCE-GATE positions
    /// (0 = auto, n/2). A candidate source is preferred when its owning value
    /// is not targeted within the next `src_horizon` source gates, so masks
    /// usually die of ordinary re-source churn rather than of a forced
    /// migration. Relaxed automatically when no candidate qualifies.
    pub src_horizon: usize,
    /// Restrict distributed sources to the value range `[src_lo, src_hi)`
    /// (`src_hi = 0` means "up to n"). Default `0..0` = every value.
    ///
    /// Sourcing a mask on a carrier of an ACTIVELY COMPUTING value makes the
    /// mask move with the computation, which is measurably a progress leak:
    /// on the real n=128 sandwich, unrestricted distributed sourcing leaves a
    /// faint but perfectly monotone ridge (rho 0.995) where the frozen band
    /// leaves none. Restricting the pool to values that carry keyed junk
    /// rather than computation restores a progress-independent source without
    /// spending a wire — on the sliced sandwich the upper half `[n/2, n)` of
    /// the gadget's values is exactly that: each is targeted once, by the
    /// N-column CNOT, and otherwise carries slicing junk.
    pub src_lo: usize,
    pub src_hi: usize,
    /// Realize DB-eligible fold fragments in the g57/CNOT vocabulary (0 = off).
    ///
    /// Clearing the width cap is necessary but not sufficient for the frozen
    /// store: a comp=0 width-2 conjunction is not in the X-free g57 span, so a
    /// narrow fold fragment can still be undigestible. This re-realizes exactly
    /// the fragments that already pass the cap, at 1-2 gates instead of 1, and
    /// removes NO wide gates -- it improves the QUALITY of the reachable
    /// material, not its extent. Fossil count is untouched; see --prod-ladder.
    pub g57_narrow: usize,
    /// Ladder fold fragments of width in (2, ladder_cap] down to <=2 controls
    /// with the borrowed-carrier double sweep; 0 disables. Wider fragments are
    /// left as single wide gates -- laddering those is what full narrow mode's
    /// ~15x cost is made of.
    pub ladder_cap: usize,
    /// Percent of values given one EXTRA high-degree mask term, so a CG block's
    /// fragment count stops being the fixed (1+k_total)^arity.
    pub cg_jitter: usize,
    /// Draw each ladder emission's spelling from the full equivalent menu,
    /// forcing a rung's two copies to differ (0 = one fixed spelling).
    pub rung_menu: usize,
    /// Retire-and-refill epochs: inter-SG gaps between two events (0 = off).
    ///
    /// A band value is a FROZEN FUNCTION OF THE INPUT, and that is what a
    /// checkpoint fingerprint detects: rolling relocates the variable, but the
    /// wire's Boolean function is the same function, so the band population
    /// separates from the carriers by LIFETIME alone, without guessing a wire
    /// set. Measured on shipped builds with rolling on, the elbow lands on the
    /// band size (46 -> 45 recovered; 256 -> 254). Band width is no defense
    /// here, because the attack recovers the population rather than
    /// enumerating candidate subsets.
    ///
    /// One event retires a band variable (re-sourcing every live mask that
    /// names it, so nothing reads it) and then REWRITES it, changing the
    /// function. Round-robin over the variables, so the whole population turns
    /// over every `epoch * band` gaps.
    pub epoch: usize,
    /// Percentage of refill sources drawn from CARRIERS rather than from other
    /// band wires (0 = band-internal only, 100 = carriers only).
    ///
    /// ⚠️ This interacts with `fill_pivots`. A band-only refill is an
    /// INVERTIBLE SHEAR of the band vector -- `b_v ^= f(other band values)` is
    /// unit-triangular in `b_v`, so it maps a uniform band to a uniform band
    /// and the port-side joint-uniformity theorem survives every refill. A
    /// CARRIER-sourced refill has no such guarantee: a carrier is a function of
    /// the logical computation, which may itself depend on the retired
    /// variable's pivot, so the refilled value can correlate with the rest of
    /// the band. `fill_pivots` therefore proves uniformity at the PORT only,
    /// and `refill_data > 0` forfeits it for the body. Choosing both is a
    /// deliberate trade of a theorem for unpredictability, not an oversight.
    ///
    /// Band-internal refills are cosmetic against a semantic attacker: every
    /// future band value would stay inside the algebra generated by the
    /// initial band, with the mixing coefficients readable off the emitted gate
    /// list, so recovering the basis once propagates forward. Worse, the b
    /// values can drift into linear dependence, which quietly breaks the very
    /// joint-uniformity the mask statistics assume.
    ///
    /// Carrier sources inject genuinely new material, at the risk of
    /// correlating the band with computational progress -- the failure mode
    /// that made distributed sourcing leak. The two differ structurally: there,
    /// fold fragments READ carrier literals, coupling carriers of different
    /// values inside the fold's own algebra; here the carrier/band partition
    /// survives and the band merely DERIVES from carriers at discrete,
    /// ledger-known points. Which is why this is a rate to be measured rather
    /// than a switch to be feared.
    pub refill_data: usize,
    /// Reserved pivot block in the nonlinear band fill (0 = legacy draw).
    ///
    /// The legacy fill draws each wire's pivot independently (so pivots
    /// collide) and excludes only a wire's OWN pivot from its own material, so
    /// one wire's pivot can re-enter another's linear part. That delivers
    /// MARGINAL balance per band wire — which is all the fill's test checks —
    /// but a mask multiplies THREE band wires, and the statistics the encoding
    /// claims need the band to be JOINTLY uniform.
    ///
    /// With this set, pivots are drawn without replacement and the whole pivot
    /// set is excluded from every wire's non-pivot material, making the
    /// pivot-to-band map unit lower-triangular and hence the band exactly
    /// uniform on `{0,1}^b` for ANY nonlinear part. Measured: worst
    /// subset-XOR bias 0.0000 against the legacy draw's 0.0625
    /// (`prod_reserved_pivots_make_the_band_jointly_uniform`). Costs no gates.
    ///
    /// Requires `band <= n`; the output-side mirror fill targets more wires
    /// than there are data wires and falls back to the legacy draw.
    pub fill_pivots: usize,
    /// Single-carrier decode: one linear term instead of two.
    ///
    /// The shipped `[3,3]` decode is really `[1,1,3,3]` — two degree-1 carrier
    /// literals plus two degree-3 products. Only ONE linear term is forced:
    /// the balance obstruction rules out a decode with no linear part (a pure
    /// product is 3:1 and ungateable), but `D = c ^ g(sources)` is exactly
    /// balanced for any `g`, since flipping `c` flips `D`. The second carrier
    /// is free to an affine adversary — it just XORs both in — so it
    /// contributes nothing to the piling-up product, which runs over the
    /// NONLINEAR terms alone.
    ///
    /// Fold cost is the product over operands of their atom counts, so
    /// swapping the redundant carrier for a degree-2 mask is cost-neutral in
    /// fragments and strictly better statistically: `[1,2,3,3]` is 5 atoms and
    /// 0.641 agreement against `[1,1,3,3]`'s 5 atoms and 0.781. It also halves
    /// the carriers, n instead of 2n.
    ///
    /// What it costs: (a) one probe now sees `v ^ masks ^ κ`, correlated with
    /// the value at the piling-up rate, where a two-carrier share is uniform
    /// AND independent; (b) RG2 (re-pair) and RG3 (refresh both carriers) have
    /// no single-carrier analogue — only relocation and mask re-sourcing
    /// survive, so the re-randomisation layer is thinner.
    pub single: usize,
    /// Product-fold mode: 0 expands the cartesian product, 1 is the original
    /// aggregate Gray fold, 2 is the four-share micro-Gray fold, and 3 is the
    /// sentinel Gray fold that gathers only quadratic atoms.
    ///
    /// Mode 1 gathers each operand's complete mask sum onto a dirty
    /// accumulator once and reads it back four times. Mode 2 instead partitions
    /// every operand into four formal ANF shares and evaluates all sixteen
    /// share rectangles. No accumulator transition in mode 2 contains more
    /// than one share; the four share deltas still reconstruct the operand, so
    /// this is a narrower trace claim, not protection against a structural
    /// attacker that can identify and combine the whole block.
    ///
    /// In aggregate and micro mode the product reads are at most two controls;
    /// a degree-three gather uses a restored dirty helper and a quartic micro
    /// gather uses two. Sentinel mode instead ladders cross tails through
    /// width four and deliberately leaves wider high--high products as fossils.
    /// Blocks a mode cannot amortize fall back to the odometer.
    ///
    /// SECURITY LIMIT: the old audit considered each prefix independently.
    /// A trace adversary can subtract the accumulator before and after the
    /// gather and recover the complete operand mask sum exactly.  See
    /// `prod_gray_fold_has_an_exact_space_time_operand_recovery`; deployments
    /// that reject this witness should use
    /// [`ProdConfig::production_single_no_gray_phase_a`].
    pub gray_fold: usize,
    /// Per-gate mask swap-with-refresh: RETIREMENT SIDES PER FOLD (0 = off).
    /// Side 1 is the target value, whose emissions are interleaved INTERIOR to
    /// the fold's own fragment stream; sides 2.. are additional values (a
    /// control, or a value steered by the drain set) whose emissions follow the
    /// whole fold. Each side retires one mask term and gains a freshly drawn
    /// one of the same degree.
    ///
    /// ⚠️ SEMANTICS CHANGED 2026-08-24: this was previously a flag, and the
    /// fold always did exactly target + one control. `PROD_SWAP=2` reproduces
    /// that stream; `PROD_SWAP=0` still restores the Gray stream byte-for-byte.
    /// The rate is now a knob because it is the BANDWIDTH the drain set spends
    /// (see [`ProdLedger::drain_rotate`]): band turnovers over a circuit are
    /// `sides * folds / (values * factors-per-value)`. Measured at n=128
    /// against the 2026-08-20 stream's 884,408 gates / 6.32 turnovers, same
    /// seed, all arms forward+reverse verified:
    ///
    /// | sides | turnovers | gates    | vs baseline |
    /// |-------|-----------|----------|-------------|
    /// | 2     | 10.06     |  825,472 | -6.7%       |
    /// | 3     | 14.72     |  886,695 | +0.3%       |
    /// | 4     | 18.62     |  940,240 | +6.3%       |
    ///
    /// ⚠️ Cost per side is NOT flat, so do not extrapolate the table: a
    /// base-degree retirement is one g57, a degree-3 tower term is a 3-control
    /// conjunction the ladder re-spells into ~4 gates. Steering prefers the
    /// cheaper slot (see `drain_pick_slot`), so raising the rate pushes an
    /// increasing share of the work onto the expensive tier.
    ///
    /// WHY: with a time-invariant mask set, the masks cancel in every
    /// before/after XOR across a fold, leaving the exact GF(2) identity
    /// `carrier(post) ^ carrier(pre) = src(post) ^ src(pre)` (measured 100%
    /// on linear source gates by flip_match). The refresh puts one
    /// non-cancelling degree->=2 band monomial on each side of the gate.
    /// A VERBATIM move of the same monomial between the two values would not
    /// do: over GF(2) the moved term is emitted once on each wire it changes,
    /// so the two touched carriers' deltas XOR back to the source delta
    /// (conservation). Fresh draws are what break that parity.
    ///
    /// Swap mode forces the EXPANDED fold: the aggregate/micro/sentinel Gray
    /// gathers materialize an operand's complete mask sum on one accumulator
    /// wire as a segment pair, which reconstructs the operand linearly no
    /// matter how mask ownership is shuffled (see the SECURITY LIMIT above),
    /// and the gather/strip snapshot cannot tolerate a mid-block registry
    /// change. It also switches the emitting constructors to the
    /// target-stable commuting shuffle, so the interior interleave survives
    /// reordering (same-target XOR writes commute and would otherwise drift).
    pub swap_refresh: usize,
    /// Closing zero-slice block (0 = off). The slice-zero wrappers append a
    /// second, independently drawn slice-guard block at the END of the
    /// circuit with the same specification as the opening one — identity
    /// exactly on the zero band slice, every nonzero band slice perturbs the
    /// data — so a reverse evaluator meets the same structure at their entry
    /// as a forward evaluator does. Its targets are restricted to the low
    /// (forward-junk) half of the data wires: the forward-honest run reaches
    /// it with a junked band, so it FIRES on the honest slice and must only
    /// ever perturb wires whose contents are already junk.
    pub close_slice: usize,
}

impl ProdConfig {
    pub fn off() -> ProdConfig {
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
            fill_pivots: 0,
            g57_narrow: 0,
            ladder_cap: 0,
            cg_jitter: 0,
            rung_menu: 0,

            epoch: 0,
            refill_data: 0,
            single: 0,
            gray_fold: 0,
            swap_refresh: 0,
            close_slice: 0,
        }
    }

    /// The value range distributed sourcing may draw from.
    pub fn src_range(&self, n: usize) -> (usize, usize) {
        let hi = if self.src_hi == 0 {
            n
        } else {
            self.src_hi.min(n)
        };
        (self.src_lo.min(hi), hi)
    }

    /// The validated production setting, as measured, in one place.
    ///
    /// Every hardening lever here defaults OFF, which is right for
    /// reproducibility and wrong for anyone generating a circuit: the defaults
    /// (auto band, no nonlinear fill, no rolling, no epochs) produce a
    /// materially weaker gadget than anything the measurements describe. This
    /// is the setting the numbers in docs/SINGLE_CARRIER_CONSTRUCTION refer
    /// to. `n` is the ENTRY POINT's value count -- the sandwich width, not the
    /// source circuit's.
    ///
    /// Note band = n: that buys the 1:1 carrier/band split (no write-count
    /// threshold separates the populations) and rules out `fill_pivots`, which
    /// needs room for non-pivot data wires. The other way round -- band ~ 3n/4
    /// with `fill_pivots` -- trades homogeneity for provable joint uniformity.
    pub fn production_single() -> ProdConfig {
        ProdConfig {
            // [2,2,2,3]: three degree-2 mask terms and one degree-3, against
            // the [2,3,3] this replaces.
            //
            // A degree-`d` atom contributes `1 - 2^(1-d)` to the piling-up
            // product, so a degree-2 term is the STRONGER statistical masker
            // (0.5 against a degree-3's 0.75) while a degree-3 term is the
            // stronger ALGEBRAIC one -- a degree-2 atom sits inside a degree-2
            // exact adversary's span and a degree-3 atom does not. Low degree
            // buys statistics, high degree buys algebra; the plan is the mix.
            // [2,2,2,3] is 0.09375 against [2,3,3]'s 0.28125.
            //
            // MEASURED at n=128 (docs/CORRELATING_TWO_COMPUTATIONS), same C and
            // sandwich, every arm verified: the statistical leak is LINEAR in
            // the piling-up product (F1_raw = 0.262*eps + 0.007, R^2 = 0.996
            // over five plans spanning 4x in eps), so this is a 3.2x reduction
            // -- 0.0318 against 0.0817 -- and the stress battery drops from
            // ALIGNED-LEAK on both probes to flat on both. It also costs LESS:
            // 692,653 gates against 808,618, and store-reachability rises to
            // 97.53% from 95.47%. Cheaper, more digestible, lower leak.
            //
            // It is only affordable because of the Gray fold. Under the wide
            // fold a block emits (1+k)^arity fragments, so a fourth mask term
            // is MULTIPLICATIVE (+56%); the Gray fold's product part is a fixed
            // ~9 gates whatever k is, and a term costs only its own gather --
            // ~1 gate at degree 2 against ~4 at degree 3. Mask-plan cost is
            // additive here, which is why trading a degree-3 atom for degree-2
            // atoms makes the circuit smaller.
            //
            // What it gives up is REDUNDANCY, not threshold. Exact degree-D
            // recovery needs D >= max atom degree, so [2,2,2,3] and [2,3,3] are
            // equally out of a degree-2 adversary's reach -- both measured dead
            // at degree 2, zero interior rows -- but this plan holds ONE
            // degree-3 atom where the old one held two. If that atom is ever
            // compromised the value falls into degree-2 range with nothing
            // behind it. The natural step-up is [2,2,2,3,3] (`--prod-k-hi 2`):
            // eps 0.0703, F1 raw 0.0258, both degree-3 atoms back, for 924,284
            // gates (+14% over the old default rather than -14%).
            k: 3,
            deg: 2,
            k_hi: 1,
            deg_hi: 3,
            // 0 = match the value count; see band_size.
            band: 0,
            rsrc: 1,
            max_width: 0,
            fill_nl: 2,
            roll: 1,
            src_dist: 0,
            src_horizon: 0,
            src_lo: 0,
            src_hi: 0,
            fill_pivots: 0,
            g57_narrow: 1,
            // Selective laddering at the measured reachability ceiling
            // (cap 4). Under swap_refresh the expanded fold would otherwise
            // leave ~48% of the circuit as wide (3-6 control) fossils the
            // g57/CNOT store can never re-encode — fmix needs a
            // predominantly-g57 stream. The earlier cap-4 leak (9/84 linear
            // flip_match matches at n=32) was the ladder borrowing LIVE
            // CARRIERS as scratch, exposing data states in its chain deltas;
            // under swap mode the scratch pool is now the live band
            // variables (see ladder_borrow_pool), whose values are fill
            // junk. Cap 3 rather than the historical cap-4 ceiling: under the
            // expanded fold, cap 4's extra narrowness arrives as plain
            // 2-control conjunctions (~35% of the circuit — the weak store
            // shape g57_narrow exists to avoid) at twice cap 3's size, while
            // cap 3 lands 82.5% of the gates in the pure g57+CNOT vocabulary
            // (measured n=64: 45.6% g57 + 36.9% CNOT, 0.3% conj-2, 17.2%
            // wide). The campaign A/B lever is PROD_LADDER_CAP.
            ladder_cap: 3,
            cg_jitter: 50,
            // Spelling variability ON: it is now restricted to the emissions
            // where the equivalent spellings are the SAME SIZE, so it costs
            // nothing. (Applied uniformly it cost +54% gates, which is why it
            // was briefly off.) Moot unless --prod-ladder-cap is set, since the
            // pair it breaks is the double sweep's.
            rung_menu: 1,

            // OFF: the drain set (swap_refresh below) supersedes this channel.
            // Both do the same job -- turn a band variable's Boolean function
            // over so it stops being a lifetime signature -- but this one PAYS
            // to release the variable (~10 live references x 2 emissions =
            // ~23 gates per event, ~2.1% of the circuit for ~6 turnovers),
            // while the drain set steers retirements the fold is making anyway
            // and pays only the rewrite. `PROD_EPOCH=5` restores it for the
            // A/B; running both double-pays and inflates the turnover count.
            epoch: 0,
            refill_data: 50,
            single: 1,
            // ON. The fold no longer emits a single wide fragment, and
            // store-reachability goes 31.55% -> 95.47% at 2.38x the gates on
            // the [2,3,3] A/B -- more than half the old circuit was material
            // phase A could never re-encode. It is also what makes the mask
            // plan above affordable; see docs/GRAY_FOLD_CG.
            //
            // NOTE: with swap_refresh ON (below), fold_cg declines every Gray
            // mode and takes the expanded path regardless of this field --
            // the aggregate gather's operand-recovery witness is exactly the
            // kind of cross-circuit linear equation the swap exists to kill.
            // PROD_SWAP=0 restores the measured Gray stream byte-for-byte.
            gray_fold: 1,
            // The 2026-08-20 redesign: per-gate mask swap-with-refresh and the
            // closing zero-slice block. Together they remove every known GF(2)
            // linear relation between source wire-segment XORs and gadget
            // wire-segment XORs, and give reverse evaluation the same
            // structural entry as forward evaluation.
            //
            // 3 sides, not the redesign's 2: the extra one is what the drain
            // set spends to reach 14.7 band turnovers instead of the epoch
            // channel's 6.3 — 2.3x the turnover for +0.3% gates, because
            // dropping `epoch` above pays for it almost exactly. 4 sides
            // reaches 18.6 but costs +6.3%, which is the wrong side of the
            // knee; see the measured table on ProdConfig::swap_refresh.
            swap_refresh: 3,
            close_slice: 1,
        }
    }

    /// Production product-mask settings for the opt-in five-carrier
    /// representation.
    ///
    /// The mask plan and churn policy intentionally match
    /// [`Self::production_single`].  `single` remains set internally because
    /// the shared product ledger injects every mask through c0; representation
    /// selection itself is carried by the public five-carrier entry point, not
    /// by this legacy ledger bit.
    pub fn production_five_carrier() -> ProdConfig {
        Self::production_single()
    }

    /// Product-mask settings for the opt-in six-carrier representation.
    /// Representation selection remains the caller's responsibility; the
    /// shared ledger still injects masks through lane c0.
    pub fn production_six_carrier() -> ProdConfig {
        Self::production_single()
    }

    /// Product-mask settings for the opt-in seven-carrier representation.
    /// Representation selection remains the caller's responsibility; the
    /// shared ledger still injects masks through lane c0.
    pub fn production_seven_carrier() -> ProdConfig {
        Self::production_single()
    }

    /// Phase-A-friendly product sharing without the aggregate Gray gather.
    ///
    /// This expands source products atom-by-atom and narrows fragments through
    /// width four while the ledger still knows their operand boundaries.  It
    /// deliberately leaves the rarer width-five/six fragments intact: the
    /// frozen-store census and live GSS samplers both found full laddering less
    /// usable despite its slightly higher structural width eligibility.
    pub fn production_single_no_gray_phase_a() -> ProdConfig {
        let mut p = Self::production_single();
        p.gray_fold = 0;
        p.ladder_cap = 4;
        p
    }

    /// Distributed (band-free) sourcing: mask literals live on ordinary
    /// carriers, protected by the write barrier instead of by a freeze.
    pub fn dist(&self) -> bool {
        self.enabled() && self.src_dist > 0
    }

    /// Single-carrier decode: one linear term per value instead of two.
    pub fn single_carrier(&self) -> bool {
        self.enabled() && self.single > 0
    }

    /// Lookahead horizon in source-gate positions. Auto is `n/2`, which at the
    /// production sandwich sits just past the measured median inter-targeting
    /// gap (89 gates at n=128, 2n=256 values), so a typical draw survives to
    /// its natural re-source rather than to a forced migration.
    pub fn horizon(&self, n: usize) -> usize {
        if self.src_horizon > 0 {
            self.src_horizon
        } else {
            (n / 2).max(8)
        }
    }

    /// Narrow (phase-A vocabulary) mode: ladder emissions to <= max_width.
    /// Ladders borrow dirty carriers and restore them exactly, so narrow mode
    /// costs no extra wires and keeps the unconditional endpoint contract.
    pub fn narrow(&self) -> bool {
        self.enabled() && self.max_width >= 2
    }

    pub fn enabled(&self) -> bool {
        self.k > 0 || self.k_hi > 0
    }

    /// Total mask terms per value across both tiers.
    pub fn k_total(&self) -> usize {
        self.k + self.k_hi
    }

    /// Highest mask degree in use.
    pub fn max_deg(&self) -> usize {
        let mut d = 2;
        if self.k > 0 {
            d = d.max(self.deg);
        }
        if self.k_hi > 0 {
            d = d.max(self.deg_hi);
        }
        d
    }

    /// Band width: the live-slot demand is n*k_total distinct tuples of degree
    /// up to `max_deg`; auto sizing keeps ~4x headroom for re-source churn. A
    /// wider mask degree needs more band wires per slot but collides far less,
    /// so the linear bound (>= maxdeg + a small pad) plus the sqrt term covers
    /// both.
    pub fn band_size(&self, n: usize) -> usize {
        if !self.enabled() || self.dist() {
            // Distributed sourcing has no band: the source "role" moves over
            // the 2n carriers and costs no wire of its own.
            return 0;
        }
        if self.band > 0 {
            return self.band;
        }
        // 0 = "match the value count". A 1:1 carrier/band split is what makes
        // the write census fail to separate the two populations: at band = n
        // the two write distributions sit on top of each other (185/452/847
        // against 180/428/848 at n=128), while a narrow band is a minority a
        // windowed census can still isolate. It also rules out the reserved
        // pivot block, which needs room for non-pivot data wires -- that is
        // the homogeneity-versus-provable-uniformity trade, resolved here in
        // favour of homogeneity because the pivot theorem is forfeited by
        // carrier-sourced refills anyway.
        //
        // This REPLACED an auto rule of ceil(sqrt(4*n*k_total)) (56 at n=256),
        // which predates the homogeneity measurement. Pass --prod-band
        // explicitly for any other sizing.
        n.max(6).max(self.max_deg() + 3)
    }
}

/// One registered product mask term `PROD_j (b_j ^ a_j)` over `deg` distinct
/// band VARIABLES, stored sorted by variable id as (id, a) factors. Degree 2
/// is the balanced base encoding; higher degree is a tower level that hides
/// the value from any reconstruction adversary of degree < deg (the term is
/// not in the lower-degree GF(2) span), at the cost of wider fold fragments.
///
/// Factors name band variables, not physical wires: a rolling band relocates
/// variables between wires mid-body ([`ProdLedger::roll`]), and a slot's
/// meaning — hence the `used` dedup set and every strip — must be invariant
/// under that. Physical literals are resolved at emission time through the
/// ledger's `loc` map.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct ProdSlot {
    /// (band variable id, a): the factor is `b ^ a`; the emitted control
    /// literal is `(loc[b], !a)` — it fires exactly when `b ^ a == 1`.
    factors: Vec<(u16, bool)>,
}

impl ProdSlot {
    /// Control literals realizing the product as one conjunction, resolved
    /// against the band's current physical placement. Under distributed
    /// sourcing there is no band and no placement map: factors name physical
    /// carriers outright, and `loc` is empty (the identity).
    fn lits(&self, loc: &[u16]) -> Vec<(u16, bool)> {
        self.factors
            .iter()
            .map(|&(b, a)| {
                let wire = if loc.is_empty() { b } else { loc[b as usize] };
                (wire, !a)
            })
            .collect()
    }

    /// The wires this slot names (physical under distributed sourcing).
    fn wires(&self) -> impl Iterator<Item = usize> + '_ {
        self.factors.iter().map(|&(b, _)| b as usize)
    }
}

// ---------------------------------------------------------------------------
// Five-carrier nonlinear representation.
//
// A logical bit carried by the five physical wires `c[0..5]` is
//
//   D(c) = c0 + c1*c2 + c1*c3 + c2*c3 + c1*c4.
//
// Product masks and the ledger constant are added outside D exactly as in the
// existing product-share construction.  The update map below is the supplied
// fixed-point-free, class-preserving permutation U0; U1 is U0 followed by a
// flip of c0.  Since c0 occurs linearly in D, U1 flips the logical class.
// ---------------------------------------------------------------------------

#[cfg(test)]
const FIVE_CARRIER_U0: [u8; 32] = [
    19, 27, 8, 28, 0, 25, 12, 22, 24, 18, 1, 30, 5, 7, 3, 11, 29, 21, 23, 15, 26, 6, 4, 31, 16, 14,
    13, 10, 9, 20, 2, 17,
];

/// Exact 40-gate realization of the supplied U0 truth table on local wires
/// 0..4.
/// Each tuple is `(target, controls)`, with controls encoded as `(wire,
/// polarity)`.  The three width-4 gates are unavoidable in any no-helper
/// realization: U0 is an odd permutation, while every gate with at most three
/// controls induces an even permutation on five bits.
const FIVE_CARRIER_U0_GATES: &[(u8, &[(u8, bool)])] = &[
    (2, &[]),
    (1, &[(2, true), (3, true), (4, false)]),
    (2, &[(3, true)]),
    (0, &[(3, true)]),
    (3, &[(0, true)]),
    (0, &[(2, true)]),
    (4, &[(0, false), (1, true), (3, false)]),
    (0, &[(4, true)]),
    (2, &[(1, true), (3, false)]),
    (4, &[(0, true), (1, true)]),
    (1, &[(2, true)]),
    (0, &[(1, true), (2, true)]),
    (1, &[(0, true), (2, true), (4, false)]),
    (1, &[(4, true)]),
    (3, &[(4, true)]),
    (2, &[(1, true), (4, true)]),
    (4, &[(0, false), (1, true), (2, true)]),
    (4, &[(3, true)]),
    (0, &[(1, true), (3, true)]),
    (3, &[(0, true), (1, true), (2, true), (4, false)]),
    (0, &[(1, false), (2, false), (3, true)]),
    (0, &[(2, true), (3, true), (4, false)]),
    (2, &[(0, true), (3, true)]),
    (4, &[(1, true), (3, true)]),
    (3, &[(4, true)]),
    (4, &[(0, true), (1, true), (3, true)]),
    (1, &[(4, true)]),
    (3, &[(4, true)]),
    (4, &[(2, true), (3, true)]),
    (0, &[(4, true)]),
    (4, &[(1, true), (2, true), (3, true)]),
    (1, &[(2, true), (4, true)]),
    (2, &[(1, true), (3, false), (4, true)]),
    (0, &[(2, true), (3, true), (4, true)]),
    (3, &[(0, true), (1, true), (2, true), (4, true)]),
    (0, &[(2, false), (3, true), (4, true)]),
    (0, &[(1, true), (3, true), (4, true)]),
    (1, &[(0, true), (3, true), (4, true)]),
    (0, &[(1, true), (3, true), (4, true)]),
    (1, &[(0, false), (2, true), (3, true), (4, true)]),
];

// ---------------------------------------------------------------------------
// Strong five-carrier sibling.
//
// The screenshot-supplied five-carrier map above is quadratic and therefore
// has an exact degree-two endpoint recovery.  This opt-in sibling replaces it
// with the cubic decode
//
//   D(c) = c0 + c2 + c2*c3 + c1*c2*c3 + c1*c2*c4 + c3*c4.
//
// Its class-preserving update has the affine tail
//   (c1,c2,c3,c4) -> (c1,c2,c3+1,c4+c3)
// and compensates c0 by D's coboundary.  The six gates below are in execution
// order.  Exhaustive tests pin the truth table, class transition, raw Walsh
// spectrum, and exact degree-three boundary.  Two tail lanes remain fixed, so
// this is explicitly an experimental algebraic upgrade, not a claim that the
// whole carrier tuple is structurally hidden.
// ---------------------------------------------------------------------------

#[cfg(test)]
const STRONG_FIVE_CARRIER_U0: [u8; 32] = [
    8, 9, 10, 11, 13, 12, 14, 15, 16, 17, 18, 19, 21, 20, 23, 22, 25, 24, 27, 26, 28, 29, 31, 30,
    1, 0, 3, 2, 4, 5, 6, 7,
];

const STRONG_FIVE_CARRIER_U0_GATES: &[(u8, &[(u8, bool)])] = &[
    (0, &[(2, true)]),
    (0, &[(1, true), (2, true)]),
    (0, &[(1, true), (2, true), (3, true)]),
    (0, &[(4, true)]),
    (4, &[(3, true)]),
    (3, &[]),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FiveCarrierFlavor {
    SuppliedQuadratic,
    StrongCubic,
}

#[inline]
#[cfg(test)]
fn five_carrier_decode_word(x: u8) -> bool {
    let c = |i: usize| ((x >> i) & 1) != 0;
    c(0) ^ (c(1) & c(2)) ^ (c(1) & c(3)) ^ (c(2) & c(3)) ^ (c(1) & c(4))
}

#[inline]
#[cfg(test)]
fn strong_five_carrier_decode_word(x: u8) -> bool {
    let c = |i: usize| ((x >> i) & 1) != 0;
    c(0) ^ c(2) ^ (c(2) & c(3)) ^ (c(1) & c(2) & c(3)) ^ (c(1) & c(2) & c(4)) ^ (c(3) & c(4))
}

fn emit_five_carrier_update(carriers: &[usize; 5], out: &mut Vec<XGate>) {
    for &(target_local, controls) in FIVE_CARRIER_U0_GATES {
        let target = carriers[target_local as usize] as u16;
        let lits = controls
            .iter()
            .map(|&(wire, polarity)| (carriers[wire as usize] as u16, polarity));
        out.push(XGate::conj(target, lits).expect("U0 controls exclude their target"));
    }
}

fn emit_strong_five_carrier_update(carriers: &[usize; 5], out: &mut Vec<XGate>) {
    for &(target_local, controls) in STRONG_FIVE_CARRIER_U0_GATES {
        let target = carriers[target_local as usize] as u16;
        let lits = controls
            .iter()
            .map(|&(wire, polarity)| (carriers[wire as usize] as u16, polarity));
        out.push(XGate::conj(target, lits).expect("strong-five U0 controls exclude their target"));
    }
}

#[derive(Clone, Debug)]
struct FiveCarrierState {
    n: usize,
    flavor: FiveCarrierFlavor,
    /// Logical value -> its five current physical carrier locations.  Rolls
    /// may exchange any lane with a band variable, so this is a role map, not
    /// an arithmetic layout past the input port.
    carriers: Vec<[usize; 5]>,
}

impl FiveCarrierState {
    #[cfg(test)]
    fn home(n: usize) -> FiveCarrierState {
        Self::home_with_flavor(n, FiveCarrierFlavor::SuppliedQuadratic)
    }

    fn home_with_flavor(n: usize, flavor: FiveCarrierFlavor) -> FiveCarrierState {
        FiveCarrierState {
            n,
            flavor,
            carriers: (0..n)
                .map(|value| std::array::from_fn(|lane| lane * n + value))
                .collect(),
        }
    }

    fn emit_update(&self, value: usize, out: &mut Vec<XGate>) {
        match self.flavor {
            FiveCarrierFlavor::SuppliedQuadratic => {
                emit_five_carrier_update(&self.carriers[value], out)
            }
            FiveCarrierFlavor::StrongCubic => {
                emit_strong_five_carrier_update(&self.carriers[value], out)
            }
        }
    }

    /// The product ledger is deliberately reused for mask allocation and
    /// churn.  Its single-carrier view is exactly lane c0.
    fn c0_view(&self) -> GadgetState {
        GadgetState {
            n: self.n,
            pairs: self.carriers.iter().map(|c| (c[0], c[0])).collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// Six-carrier nonlinear representation.
//
//   D(c) = c0 + c1*c2 + c3 + c1*c3 + c4 + c1*c3*c4
//          + c5 + c2*c5 + c2*c3*c5.
//
// The fixed update U0 below preserves D and has no fixed points. U1 is U0
// followed by c0 ^= 1, which changes the decoded class because c0 is the only
// occurrence of lane zero in D. Product masks and ledger constants remain
// outside D exactly as in the five-carrier sibling.
// ---------------------------------------------------------------------------

const SIX_CARRIER_D_ATOMS: [&[u8]; 9] = [
    &[0],
    &[1, 2],
    &[3],
    &[1, 3],
    &[4],
    &[1, 3, 4],
    &[5],
    &[2, 5],
    &[2, 3, 5],
];

#[cfg(test)]
const SIX_CARRIER_U0: [u8; 64] = [
    2, 3, 0, 1, 7, 6, 5, 4, 11, 10, 9, 8, 14, 15, 12, 13, 30, 31, 29, 28, 27, 26, 24, 25, 22, 23,
    21, 20, 19, 18, 16, 17, 42, 43, 41, 40, 46, 47, 45, 44, 35, 34, 32, 33, 39, 38, 36, 37, 54, 55,
    53, 52, 51, 50, 48, 49, 63, 62, 60, 61, 58, 59, 57, 56,
];

/// Exact ten-gate ancilla-free realization of the supplied U0 truth table. The
/// first six gates update c0 from the original controls; keep them before the
/// affine four-gate tail. Controls are `(local_wire, positive_polarity)`.
const SIX_CARRIER_U0_GATES: &[(u8, &[(u8, bool)])] = &[
    (0, &[(3, true), (4, true), (5, false)]),
    (0, &[(2, true), (4, false), (5, true)]),
    (0, &[(1, true), (4, false), (5, true)]),
    (0, &[(1, true), (4, true)]),
    (0, &[(3, true)]),
    (0, &[(2, true)]),
    (1, &[]),
    (2, &[(4, true)]),
    (3, &[(4, true)]),
    (3, &[(5, true)]),
];

// Experimental structural six-carrier sibling.  It deliberately keeps the
// established cubic decode (and therefore its static Walsh properties and
// exact degree-three firing boundary), but replaces the compact update's
// affine/frozen tail with a nonlinear full-affine-rank endpoint graph.
//
// The first eleven gates compensate c0 by the coboundary
// D_tail(t) + D_tail(T(t)) while t is still unmodified.  The remaining ten
// gates realize T on c1..c5.  U1 is again U0 followed by c0 ^= 1.
#[cfg(test)]
const STRONG_SIX_CARRIER_U0: [u8; 64] = [
    2, 3, 9, 8, 7, 6, 5, 4, 27, 26, 0, 1, 14, 15, 12, 13, 60, 61, 63, 62, 25, 24, 10, 11, 54, 55,
    53, 52, 19, 18, 16, 17, 58, 59, 32, 33, 46, 47, 45, 44, 39, 38, 41, 40, 22, 23, 21, 20, 35, 34,
    36, 37, 51, 50, 48, 49, 29, 28, 30, 31, 56, 57, 43, 42,
];

const STRONG_SIX_CARRIER_U0_GATES: &[(u8, &[(u8, bool)])] = &[
    // c0 coboundary, evaluated on the original c1..c5 tail.
    (0, &[(1, true)]),
    (0, &[(2, true)]),
    (0, &[(1, true), (2, true)]),
    (0, &[(3, true)]),
    (0, &[(1, true), (2, true), (4, true)]),
    (0, &[(3, true), (4, true)]),
    (0, &[(1, true), (5, true)]),
    (0, &[(2, true), (5, true)]),
    (0, &[(2, true), (3, true), (5, true)]),
    (0, &[(4, true), (5, true)]),
    (0, &[(1, true), (4, true), (5, true)]),
    // Full-rank nonlinear tail permutation T.
    (1, &[]),
    (2, &[(4, true)]),
    (3, &[(4, true)]),
    (3, &[(5, true)]),
    (4, &[(2, true), (3, false), (5, true)]),
    (5, &[(2, true), (4, true)]),
    (1, &[(3, true), (4, true)]),
    (2, &[(1, true), (3, false), (4, false), (5, true)]),
    (3, &[(1, false), (2, false), (4, false)]),
    (4, &[(1, true), (2, false), (3, true)]),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SixCarrierFlavor {
    SuppliedCompact,
    StrongFullRank,
}

#[inline]
#[cfg(test)]
fn six_carrier_decode_word(x: u8) -> bool {
    SIX_CARRIER_D_ATOMS.iter().fold(false, |value, atom| {
        value ^ atom.iter().all(|&lane| ((x >> lane) & 1) != 0)
    })
}

fn emit_six_carrier_update(carriers: &[usize; 6], out: &mut Vec<XGate>) {
    for &(target_local, controls) in SIX_CARRIER_U0_GATES {
        let target = carriers[target_local as usize] as u16;
        let lits = controls
            .iter()
            .map(|&(wire, polarity)| (carriers[wire as usize] as u16, polarity));
        out.push(XGate::conj(target, lits).expect("U0 controls exclude their target"));
    }
}

fn emit_strong_six_carrier_update(carriers: &[usize; 6], out: &mut Vec<XGate>) {
    for &(target_local, controls) in STRONG_SIX_CARRIER_U0_GATES {
        let target = carriers[target_local as usize] as u16;
        let lits = controls
            .iter()
            .map(|&(wire, polarity)| (carriers[wire as usize] as u16, polarity));
        out.push(XGate::conj(target, lits).expect("strong-six U0 controls exclude their target"));
    }
}

#[derive(Clone, Debug)]
struct SixCarrierState {
    n: usize,
    flavor: SixCarrierFlavor,
    /// Logical value -> six current physical carrier locations. Band rolling
    /// changes this role map while the public ports remain lane-major.
    carriers: Vec<[usize; 6]>,
}

impl SixCarrierState {
    #[cfg(test)]
    fn home(n: usize) -> SixCarrierState {
        Self::home_with_flavor(n, SixCarrierFlavor::SuppliedCompact)
    }

    #[cfg(test)]
    fn strong_home(n: usize) -> SixCarrierState {
        Self::home_with_flavor(n, SixCarrierFlavor::StrongFullRank)
    }

    fn home_with_flavor(n: usize, flavor: SixCarrierFlavor) -> SixCarrierState {
        SixCarrierState {
            n,
            flavor,
            carriers: (0..n)
                .map(|value| std::array::from_fn(|lane| lane * n + value))
                .collect(),
        }
    }

    fn emit_update(&self, value: usize, out: &mut Vec<XGate>) {
        match self.flavor {
            SixCarrierFlavor::SuppliedCompact => {
                emit_six_carrier_update(&self.carriers[value], out)
            }
            SixCarrierFlavor::StrongFullRank => {
                emit_strong_six_carrier_update(&self.carriers[value], out)
            }
        }
    }

    fn c0_view(&self) -> GadgetState {
        GadgetState {
            n: self.n,
            pairs: self
                .carriers
                .iter()
                .map(|carriers| (carriers[0], carriers[0]))
                .collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// Seven-carrier nonlinear representation.
//
//   D(c) = c0 + c1 + c2 + c3*c4 + c5*c6 + c3*c4*c5*c6.
//
// The fixed update U0 below preserves D and has no fixed points.  The legacy
// fold obtains U1 by following U0 with c0 ^= 1.  The opt-in distributed-refresh
// fold deliberately emits no U0: it places the source fragments on a randomly
// relabeled linear lane, then separates consecutive fragments with nonlinear
// D-preserving shears. Product masks and ledger constants remain outside D
// exactly as in the five- and six-carrier siblings.
// ---------------------------------------------------------------------------

const SEVEN_CARRIER_D_ATOMS: [&[u8]; 6] = [&[0], &[1], &[2], &[3, 4], &[5, 6], &[3, 4, 5, 6]];

#[cfg(test)]
const SEVEN_CARRIER_U0: [u8; 128] = [
    80, 81, 44, 45, 47, 46, 83, 82, 119, 118, 94, 95, 107, 106, 90, 91, 60, 61, 66, 67, 65, 64, 63,
    62, 101, 100, 121, 120, 72, 73, 76, 77, 113, 112, 87, 86, 84, 85, 114, 115, 89, 88, 104, 105,
    93, 92, 116, 117, 69, 68, 99, 98, 96, 97, 70, 71, 102, 103, 122, 123, 75, 74, 79, 78, 16, 17,
    55, 54, 52, 53, 19, 18, 25, 24, 42, 43, 41, 40, 26, 27, 37, 36, 2, 3, 1, 0, 38, 39, 126, 127,
    11, 10, 8, 9, 125, 124, 49, 48, 22, 23, 21, 20, 50, 51, 110, 111, 31, 30, 28, 29, 109, 108, 4,
    5, 35, 34, 32, 33, 7, 6, 12, 13, 57, 56, 58, 59, 15, 14,
];

/// Exact 26-gate ancilla-free realization of the supplied U0 truth table.
/// Controls are `(local_wire, positive_polarity)` and must be applied in the
/// listed order. The two four-control decode gates are intentional.
const SEVEN_CARRIER_U0_GATES: &[(u8, &[(u8, bool)])] = &[
    (0, &[(2, true)]),
    (0, &[(3, true), (4, true)]),
    (0, &[(5, true), (6, true)]),
    (0, &[(3, true), (4, true), (5, true), (6, true)]),
    (2, &[(1, true)]),
    (2, &[(4, true)]),
    (5, &[(2, true)]),
    (4, &[]),
    (6, &[]),
    (0, &[(2, true)]),
    (0, &[(3, true), (4, true)]),
    (0, &[(5, true), (6, true)]),
    (0, &[(3, true), (4, true), (5, true), (6, true)]),
    (1, &[(2, true)]),
    (1, &[(3, true), (4, true)]),
    (1, &[(5, true), (6, true)]),
    (1, &[(3, true), (4, true), (5, true), (6, true)]),
    (5, &[(1, true), (3, true), (6, true)]),
    (2, &[(3, true), (4, true), (5, true)]),
    (3, &[(2, true), (5, true), (6, true)]),
    (4, &[(3, true), (5, true)]),
    (6, &[(2, true), (3, true), (5, true)]),
    (1, &[(2, true)]),
    (1, &[(3, true), (4, true)]),
    (1, &[(5, true), (6, true)]),
    (1, &[(3, true), (4, true), (5, true), (6, true)]),
];

/// A random automorphism of D: arbitrary S3 on the linear roles, independent
/// swaps inside the two nonlinear pairs, and an optional pair swap.  `role[i]`
/// is the physical carrier-lane role used for canonical coordinate i.
fn seven_carrier_role_automorphism(rng: &mut impl Rng) -> [u8; 7] {
    let mut linear = [0u8, 1, 2];
    linear.shuffle(rng);
    let mut left = [3u8, 4];
    let mut right = [5u8, 6];
    if rng.random_bool(0.5) {
        left.swap(0, 1);
    }
    if rng.random_bool(0.5) {
        right.swap(0, 1);
    }
    if rng.random_bool(0.5) {
        std::mem::swap(&mut left, &mut right);
    }
    [
        linear[0], linear[1], linear[2], left[0], left[1], right[0], right[1],
    ]
}

#[inline]
#[cfg(test)]
fn seven_carrier_decode_word(x: u8) -> bool {
    SEVEN_CARRIER_D_ATOMS.iter().fold(false, |value, atom| {
        value ^ atom.iter().all(|&lane| ((x >> lane) & 1) != 0)
    })
}

fn emit_seven_carrier_update(carriers: &[usize; 7], out: &mut Vec<XGate>) {
    for &(target_local, controls) in SEVEN_CARRIER_U0_GATES {
        let target = carriers[target_local as usize] as u16;
        let lits = controls
            .iter()
            .map(|&(wire, polarity)| (carriers[wire as usize] as u16, polarity));
        out.push(XGate::conj(target, lits).expect("U0 controls exclude their target"));
    }
}

#[derive(Clone, Debug)]
struct SevenCarrierState {
    n: usize,
    /// Logical value -> seven current physical carrier locations. Band rolling
    /// changes this role map while the public ports remain lane-major.
    carriers: Vec<[usize; 7]>,
}

impl SevenCarrierState {
    fn home(n: usize) -> SevenCarrierState {
        SevenCarrierState {
            n,
            carriers: (0..n)
                .map(|value| std::array::from_fn(|lane| lane * n + value))
                .collect(),
        }
    }

    fn c0_view(&self) -> GadgetState {
        GadgetState {
            n: self.n,
            pairs: self
                .carriers
                .iter()
                .map(|carriers| (carriers[0], carriers[0]))
                .collect(),
        }
    }
}

type SevenCarrierShearKey = [(u16, bool); 2];

/// Draw a two-literal selector from carrier coordinates of two distinct,
/// non-target logical values.  No target carrier is read, so the same selector
/// remains stable across all three gates of a shear.
fn draw_seven_carrier_shear_selector(
    state: &SevenCarrierState,
    target_value: usize,
    used: &mut std::collections::HashSet<SevenCarrierShearKey>,
    rng: &mut impl Rng,
) -> SevenCarrierShearKey {
    let partners: Vec<usize> = (0..state.n)
        .filter(|&value| value != target_value)
        .collect();
    assert!(
        partners.len() >= 2,
        "a seven-carrier shear needs two partner values"
    );

    // At n=3 there are 49 lane pairs * 4 polarity pairs = 196 selectors,
    // enough for the production two-control folds.  Wider heterogeneous gates
    // can exceed that finite set; start a fresh no-repeat epoch if necessary.
    let capacity = partners.len() * (partners.len() - 1) / 2 * 7 * 7 * 4;
    if used.len() == capacity {
        used.clear();
    }
    loop {
        let first_index = rng.random_range(0..partners.len());
        let second_index = loop {
            let index = rng.random_range(0..partners.len());
            if index != first_index {
                break index;
            }
        };
        let first_value = partners[first_index];
        let second_value = partners[second_index];
        let mut selector = [
            (
                state.carriers[first_value][rng.random_range(0..7)] as u16,
                rng.random_bool(0.5),
            ),
            (
                state.carriers[second_value][rng.random_range(0..7)] as u16,
                rng.random_bool(0.5),
            ),
        ];
        if selector[1] < selector[0] {
            selector.swap(0, 1);
        }
        if used.insert(selector) {
            return selector;
        }
    }
}

/// Emit the three-gate nonlinear shear associated with canonical nonlinear
/// coordinate `x` (3..=6), interpreted through a random D automorphism.
///
/// If `m` is x's pair-mate and `(p,q)` is the other nonlinear pair, changing
/// x by h changes D by `h*m*(1+p*q)`.  The first two gates add that exact
/// quantity to one linear coordinate before the third gate changes x, so the
/// complete shear preserves D for every carrier tuple and every dirty h.
fn emit_seven_carrier_preserving_shear(
    state: &SevenCarrierState,
    target_value: usize,
    roles: &[u8; 7],
    x: u8,
    selector: SevenCarrierShearKey,
    out: &mut Vec<XGate>,
) {
    let (mate, other_left, other_right) = match x {
        3 => (4, 5, 6),
        4 => (3, 5, 6),
        5 => (6, 3, 4),
        6 => (5, 3, 4),
        _ => panic!("seven-carrier shear coordinate must be in 3..=6"),
    };
    let carrier =
        |canonical: u8| state.carriers[target_value][roles[canonical as usize] as usize] as u16;
    let linear_target = carrier(0);
    let mut first = selector.to_vec();
    first.push((carrier(mate), true));
    out.push(
        XGate::conj(linear_target, first)
            .expect("external shear selector and target carriers are distinct"),
    );
    let mut second = selector.to_vec();
    second.extend([
        (carrier(mate), true),
        (carrier(other_left), true),
        (carrier(other_right), true),
    ]);
    out.push(
        XGate::conj(linear_target, second)
            .expect("external shear selector and target carriers are distinct"),
    );
    out.push(
        XGate::conj(carrier(x), selector)
            .expect("external shear selector and target carriers are distinct"),
    );
}

/// Emit `target ^= conj(lits)` (1 or 2 literals) in the phase-A g57/CNOT
/// vocabulary, returning the extra constant the realization leaves on the
/// wire (a g57's complement). The caller compensates it: ledger constant for
/// fold/inject targets, verbatim-replay cancellation for scratch rungs, or —
/// for band fills — absorption into F.
fn emit_g57_form(
    target: u16,
    lits: &[(u16, bool)],
    rng: &mut impl Rng,
    out: &mut Vec<XGate>,
) -> bool {
    match *lits {
        [(w, true)] => {
            out.push(XGate::cnot(target, w));
            false
        }
        [(w, false)] => {
            // ¬w = w ^ 1.
            out.push(XGate::cnot(target, w));
            true
        }
        [a, b] => {
            let ((xw, xp), (yw, yp)) = if rng.random_bool(0.5) { (a, b) } else { (b, a) };
            match (xp, yp) {
                // 1 ^ ¬x·y is exactly the g57 monomial.
                (false, true) => {
                    out.push(XGate::from_g57([target, xw, yw]));
                    true
                }
                (true, false) => {
                    out.push(XGate::from_g57([target, yw, xw]));
                    true
                }
                // x·y = (1 ^ ¬x·y) ^ y ^ 1.
                (true, true) => {
                    out.push(XGate::from_g57([target, xw, yw]));
                    out.push(XGate::cnot(target, yw));
                    true
                }
                // ¬x·¬y = (1 ^ ¬x·y) ^ x — no residual constant.
                (false, false) => {
                    out.push(XGate::from_g57([target, xw, yw]));
                    out.push(XGate::cnot(target, xw));
                    false
                }
            }
        }
        _ => unreachable!("emit_g57_form takes 1..=2 literals"),
    }
}

/// Narrow-mode realization of `target ^= conj(lits)` with every emitted gate
/// at most `cap` controls, using **dirty borrowed carriers** — no clean
/// ancilla, no dedicated wire, no assumption whatever about the borrowed
/// wires' contents (the generalized Barenco double sweep; the codebase's
/// `emit_poly_add` cubic case is its w=3 instance). Each borrowed wire is
/// visited an even number of times and its dirty value cancels between the
/// two readings, so the identity is exact for arbitrary inputs and every
/// borrow is restored — the gadget keeps the "correct under any junk on the
/// non-data wires" contract, and no wire is left sitting at a constant for a
/// snapshot adversary to pick out.
///
/// Rungs must be EXACT: a g57's complement on a borrowed wire does not cancel
/// (it leaks a spurious `c·κ` into the target), so rungs are plain width-<=cap
/// conjunctions. The target gates carry their complement freely — each appears
/// twice with the same polarities, so κ cancels there — and get the randomized
/// g57 realization.
///
/// `lits` must be contradiction-free, deduped, and atom-interleaved by the
/// caller, so no borrowed wire ever holds one value's whole mask term.
/// Returns the residual constant parity (nonzero only on the direct path).
///
/// Borrows come from `pool`, the wires CURRENTLY holding carriers, bar the
/// target and the fragment's own literals; it widens to the whole wire space
/// when that leaves too few, which a fragment covering most of the carriers
/// otherwise would. Any wire is sound (the double sweep restores it before
/// anything else can read it); carriers are merely preferred, since a band
/// variable's value is constant across the body and a partial product parked
/// on one is a longer-lived difference than the same product on a churning
/// carrier.
///
/// The pool must be resolved by ROLE, not by wire index. An earlier version
/// used `0..carrier_total` as the pool, which is the carriers' HOME index
/// range -- and `--prod-roll` moves variables between wires without moving
/// indices, so index and role come apart the moment rolling is on. That
/// version therefore missed its own stated goal (band variables sitting at low
/// indices were borrowed anyway) AND pinned every ladder rung's write traffic
/// to the low half of the wire space, which is a static separator that no
/// amount of rolling can average away: measured write-count AUC between the
/// two home halves went 0.518 (no laddering) to 0.875 (`ladder_cap` 3) at
/// n=16, band 32, roll 1.
fn emit_narrow_fragment(
    target: u16,
    lits: &[(u16, bool)],
    cap: usize,
    carrier_wires: &[u16],
    borrow_total: usize,
    forbidden: &[u16],
    atoms: &[Vec<(u16, bool)>],
    menu: usize,
    rng: &mut impl Rng,
    out: &mut Vec<XGate>,
) -> bool {
    debug_assert!(cap >= 2);
    debug_assert!(!lits.is_empty());
    if lits.len() <= cap {
        if lits.len() <= 2 {
            return emit_g57_form(target, lits, rng, out);
        }
        out.push(XGate::conj(target, lits.iter().copied()).expect("caller pre-normalizes"));
        return false;
    }
    // Borrow w - cap rungs' worth of wires, avoiding the target and every
    // wire the fragment reads. Any content is fine; they are restored exactly.
    let chunk = cap - 1;
    let rung_count = (lits.len() - cap).div_ceil(chunk);
    let mut taken: Vec<usize> = vec![target as usize];
    taken.extend(lits.iter().map(|&(w, _)| w as usize));
    // `forbidden` carries the SIBLING carriers of the target and of every
    // operand. Excluding only the target and the literals is not enough: in a
    // paired build the target value's OTHER carrier is neither, so it could be
    // borrowed -- and the target gate would then read c1(t) while writing
    // c0(t), one gate seeing a whole sharing. Dormant while narrow mode was
    // shelved; live the moment any laddering is switched on.
    taken.extend(forbidden.iter().map(|&w| w as usize));
    // A sibling may coincide with a literal or the target, and a duplicated
    // entry would overcount the exclusions and underflow the free count.
    taken.sort_unstable();
    taken.dedup();
    let free_roles: Vec<u16> = carrier_wires
        .iter()
        .copied()
        .filter(|&w| !taken.contains(&(w as usize)))
        .collect();
    let mut borrowed: Vec<u16> = Vec::with_capacity(rung_count);
    if free_roles.len() >= rung_count {
        // Preferred: wires currently PLAYING the carrier role, wherever they
        // sit. Because rolling keeps exchanging roles, this set sweeps the
        // whole wire space over the body and leaves no index-shaped residue.
        use rand::seq::IndexedRandom;
        for &h in free_roles.choose_multiple(rng, rung_count) {
            taken.push(h as usize);
            borrowed.push(h);
        }
    } else {
        let free_in =
            |pool: usize| pool.saturating_sub(taken.iter().filter(|&&w| w < pool).count());
        assert!(
            free_in(borrow_total) >= rung_count,
            "narrow fragment needs {rung_count} borrows but only {} of {borrow_total} wires are free",
            free_in(borrow_total)
        );
        for _ in 0..rung_count {
            let h = random_wire_except(borrow_total, &taken, rng) as u16;
            taken.push(h as usize);
            borrowed.push(h);
        }
    }
    // Rung i computes borrowed[i] ^= borrowed[i-1] & (next `chunk` literals);
    // rung 0 takes the first `cap` literals outright. Kept as LITERAL LISTS,
    // not prebuilt gates, so each of a rung's emissions can pick its own
    // spelling -- see the duplicate-pair note below.
    // PIVOT SELECTION. Which literals form rung 0 is free, and the positional
    // `lits[..cap]` choice spends that freedom on nothing. Two things want it:
    //
    // ADMISSIBILITY (a correctness-adjacent property, and currently VIOLATED).
    // A borrowed wire must never park a literal set equal to one value's whole
    // mask term -- that is what interleave_atoms exists to prevent. It does not
    // always succeed: instrumenting the shipped fold found 95 of 38,318 width-3
    // fragments where lits[..2] IS exactly one atom. Testing it here makes it
    // exact rather than something the interleave has to achieve structurally.
    //
    // COST. A same-polarity rung has two 2-gate spellings, so its two emissions
    // can differ for FREE; a mixed-polarity rung pays +2 gates to differ. That
    // difference is worth 29% of the circuit -- forcing spelling diversity
    // without steering the pivot cost exactly that at n=128 -- so among
    // admissible choices, prefer a same-polarity pair.
    let mut rung_lits_all: Vec<Vec<(u16, bool)>> = Vec::with_capacity(rung_count);
    let mut lits: Vec<(u16, bool)> = lits.to_vec();
    if rung_count >= 1 && lits.len() > cap {
        let is_whole_atom = |cand: &[(u16, bool)]| -> bool {
            atoms
                .iter()
                .any(|a| a.len() == cand.len() && a.iter().all(|l| cand.contains(l)))
        };
        let mut best: Option<(u8, Vec<usize>)> = None;
        for combo in (0..lits.len()).combinations(cap) {
            let cand: Vec<(u16, bool)> = combo.iter().map(|&i| lits[i]).collect();
            if is_whole_atom(&cand) {
                continue; // inadmissible: parks a whole mask term on the borrow
            }
            // Steer BOTH halves of the sweep's bracket into the free class.
            // A same-polarity pair has two equal-size spellings, so its two
            // emissions can differ at no cost; a mixed pair would have to pay
            // +2. The rung is `cand`; the target gate is [(borrow, true)] plus
            // whatever literals are left, so it is same-polarity exactly when
            // those leftovers are all positive. Score 0 = both free.
            let rung_mixed = !cand.iter().all(|l| l.1 == cand[0].1);
            let rest_all_positive = (0..lits.len())
                .filter(|i| !combo.contains(i))
                .all(|i| lits[i].1);
            let score = u8::from(rung_mixed) + u8::from(!rest_all_positive);
            let better = match &best {
                None => true,
                Some((s, _)) => score < *s || (score == *s && rng.random_bool(0.5)),
            };
            if better {
                best = Some((score, combo));
            }
        }
        if let Some((_, combo)) = best {
            // Reorder so the chosen pivot is the prefix; the rest keeps its
            // interleaved order, which is what keeps laddered and unladdered
            // fragments indistinguishable by control order.
            let mut chosen: Vec<(u16, bool)> = combo.iter().map(|&i| lits[i]).collect();
            let rest: Vec<(u16, bool)> = (0..lits.len())
                .filter(|i| !combo.contains(i))
                .map(|i| lits[i])
                .collect();
            chosen.extend(rest);
            lits = chosen;
        }
    }
    let lits = &lits[..];
    let mut consumed = 0;
    for i in 0..borrowed.len() {
        let mut rung_lits: Vec<(u16, bool)> = Vec::with_capacity(cap);
        if i == 0 {
            rung_lits.extend_from_slice(&lits[..cap]);
            consumed = cap;
        } else {
            rung_lits.push((borrowed[i - 1], true));
            let upto = (consumed + chunk).min(lits.len());
            rung_lits.extend_from_slice(&lits[consumed..upto]);
            consumed = upto;
        }
        rung_lits_all.push(rung_lits);
    }
    // Target gate: last borrow plus whatever literals remain.
    let mut t_lits: Vec<(u16, bool)> = vec![(*borrowed.last().unwrap(), true)];
    t_lits.extend_from_slice(&lits[consumed..]);
    debug_assert!(t_lits.len() <= cap);

    // A g57-spelled rung, and why it is allowed to vary between emissions.
    //
    // The sweep gives `t ^= R * lambda` for ANY rung function R, where lambda
    // is the conjunction of the target gate's non-borrow literals. Spelling the
    // rung through emit_g57_form makes R = conj(rung_lits) ^ kappa, so the
    // target picks up a spurious `kappa * lambda` -- a LITERAL, not a constant,
    // which is why the old comment here said rungs must be exact. It is still
    // cancellable: emitting lambda once more on the target removes it, and that
    // emission is itself a g57 form whose own residual goes to the ledger.
    //
    // kappa depends only on the POLARITY PATTERN of rung_lits, never on
    // emit_g57_form's internal coin, so both emissions of a rung contribute the
    // same R and the borrow is still restored exactly -- while the two may be
    // spelled differently. That matters: the previous code cloned one prebuilt
    // gate, so every laddered fragment planted a BYTE-IDENTICAL gate pair. On
    // the shipped n=128 build 143,100 of 184,898 comp=0 width-2 instances
    // (77.4%) sat in duplicate groups and 50.4% of the whole circuit sat in
    // exact pairs, so `sort | uniq -c` -- no execution, no algebra -- located
    // every ladder, its borrowed wire and two of its three literals.
    //
    // Restricted to the single-rung case, which is every ladder in production
    // (ladder_cap 3 means width 3, cap 2, chunk 1, rung_count 1). With several
    // rungs each kappa_i multiplies a DIFFERENT partial lambda and the
    // corrections compound; that generalization is not verified, so it keeps
    // the exact spelling.
    let g57_rungs = rung_count == 1 && !rung_lits_all[0].is_empty();
    let mut rung_konst = false;
    // Last spelling used per rung, so the next emission can avoid repeating it.
    let mut last_spelling: Vec<Option<usize>> = vec![None; rung_lits_all.len()];
    // The TARGET gate is emitted twice as well, and it needs the same treatment
    // for the same reason: the sweep's bracket is a pair, so leaving this half
    // on emit_g57_form's coin left half the planted pairs intact (measured at
    // n=128: fixing only the rung took width-2 duplicate groups 64.6% -> 51.1%,
    // not to the coincidence floor). Its two emissions contribute the same
    // function whichever spelling is drawn, so the g57 complements still cancel
    // and no ledger constant is owed -- exactly as before.
    let mut last_target: Option<usize> = None;
    for _ in 0..2 {
        // Target gate first, then down the rungs and back up (rung 0 once per
        // block, the rest twice), so every borrow is visited an even number of
        // times and its dirty value cancels.
        if t_lits.len() <= 2 {
            let (spellings, _) = spellings_at(target, &t_lits, menu);
            let prev = if menu > 0 { last_target } else { None };
            let pick = if spellings.len() < 2 {
                0
            } else {
                let choices: Vec<usize> =
                    (0..spellings.len()).filter(|i| Some(*i) != prev).collect();
                choices[rng.random_range(0..choices.len())]
            };
            out.extend(spellings[pick].iter().cloned());
            last_target = if menu > 0 { Some(pick) } else { None };
        } else {
            out.push(XGate::conj(target, t_lits.iter().copied()).expect("distinct wires"));
        }
        for i in (0..rung_lits_all.len()).rev() {
            let prev = if menu > 0 { last_spelling[i] } else { None };
            let (pick, k) = emit_rung(
                borrowed[i],
                &rung_lits_all[i],
                g57_rungs,
                menu,
                prev,
                rng,
                out,
            );
            last_spelling[i] = if menu > 0 { pick } else { None };
            rung_konst = k;
        }
        for i in 1..rung_lits_all.len() {
            let prev = if menu > 0 { last_spelling[i] } else { None };
            let (pick, k) = emit_rung(
                borrowed[i],
                &rung_lits_all[i],
                g57_rungs,
                menu,
                prev,
                rng,
                out,
            );
            last_spelling[i] = if menu > 0 { pick } else { None };
            rung_konst = k;
        }
    }
    // Cancel the kappa * lambda the g57 rung left on the target. lambda is the
    // target gate's literals minus the borrow; with one rung that is
    // `lits[consumed..]`, at most cap-1 literals.
    if g57_rungs && rung_konst {
        let lambda = &lits[consumed..];
        if lambda.is_empty() {
            // lambda == 1: the residual is the bare constant, which the caller
            // XORs into the ledger.
            return true;
        }
        return emit_g57_form(target, lambda, rng, out);
    }
    false
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
type SpellingLog = std::collections::HashMap<(u16, Vec<(u16, bool)>), Vec<usize>>;

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
    let key = (target, lits.to_vec());
    let used = seen.entry(key).or_default();
    let fresh: Vec<usize> = (0..len).filter(|i| !used.contains(i)).collect();
    let pick = if fresh.is_empty() {
        // Every spelling is spent; restart the cycle rather than repeat the
        // most recent one.
        let last = used.last().copied();
        let choices: Vec<usize> = (0..len).filter(|i| Some(*i) != last).collect();
        used.clear();
        choices[rng.random_range(0..choices.len())]
    } else {
        fresh[rng.random_range(0..fresh.len())]
    };
    used.push(pick);
    pick
}

/// Add one mask atom to a DIRTY accumulator, every emitted gate at most two
/// controls, over a dirty borrowed helper. Returns the residual constant.
///
/// `pivot` fixes which literal pairs with the helper on the accumulator's own
/// gates, and the caller draws it ONCE for both the gather and the strip. That
/// is not cosmetic. The residual depends on the rung's POLARITY PATTERN, and
/// [`emit_narrow_fragment`] chooses its pivot at random among equally-scored
/// candidates -- so routing a gather and its strip through it independently
/// leaves DIFFERENT residuals, and the accumulator comes back off by one. The
/// borrow is then not restored at all, which is a corrupted wire rather than a
/// wrong constant. (Caught by `prod_gray_fold_keeps_the_accumulators_dirty`.)
///
/// Spellings still differ between the two passes: every spelling of one
/// function carries the same constant, which is exactly what makes varying them
/// safe here and is why the ladder's own double sweep may vary them too.
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
        let (spellings, konst) = spellings_at(acc, atom, menu);
        let pick = pick_spelling(acc, atom, spellings.len(), seen, rng);
        out.extend(spellings[pick].iter().cloned());
        return konst;
    }
    debug_assert!(
        atom.len() == 3 || atom.len() == 4,
        "only degree <= 4 atoms are gathered"
    );
    if atom.len() == 4 {
        // Exact one-rung dirty double sweep for A*B*C*D. With dirty h,
        //   acc ^= h*C*D; h ^= A*B; acc ^= h*C*D; h ^= A*B
        // adds A*B*C*D and restores h. Every emitted gate has at most three
        // controls and there is no residual constant to ledger.
        let rung = [atom[0], atom[1]];
        let tail = [(helper, true), atom[2], atom[3]];
        for _ in 0..2 {
            out.push(XGate::conj(acc, tail).expect("quartic accumulator controls are distinct"));
            out.push(XGate::conj(helper, rung).expect("quartic atom literals are distinct"));
        }
        return false;
    }
    let lam = atom[pivot];
    let rung: Vec<(u16, bool)> = (0..atom.len())
        .filter(|&i| i != pivot)
        .map(|i| atom[i])
        .collect();
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
    let (_, rung_konst) = rung_spellings(helper, &rung);
    // Accumulator gate, then rung, twice: the helper is visited an even number
    // of times, so its unknown incoming value cancels and it is restored. The
    // accumulator gate's own complement cancels between its two emissions.
    for _ in 0..2 {
        let (sp, _) = spellings_at(acc, &t_lits, menu);
        let pick = pick_spelling(acc, &t_lits, sp.len(), seen, rng);
        out.extend(sp[pick].iter().cloned());
        let (sp, _) = spellings_at(helper, &rung, menu);
        let pick = pick_spelling(helper, &rung, sp.len(), seen, rng);
        out.extend(sp[pick].iter().cloned());
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

/// Which literal of a degree-3 atom to pair with the helper. A rung of two
/// NEGATIVE literals is the one polarity pattern that realizes its conjunction
/// outright, so choosing it saves the correction gate in [`emit_atom_onto`];
/// among the choices that do, and otherwise among all of them, the draw is
/// uniform.
///
/// Steering instead toward a SAME-polarity rung (always available among three
/// literals, by pigeonhole) would give the rung four equal-size spellings
/// rather than one. Measured and rejected for the same reason as the helper
/// polarity above: those four spellings are two gate multisets reordered, so an
/// order-blind duplicate census does not see them, and forcing the choice
/// costs gates.
fn choose_pivot(atom: &[(u16, bool)], rng: &mut impl Rng) -> usize {
    let rung_pols = |i: usize| -> Vec<bool> {
        (0..atom.len())
            .filter(|&j| j != i)
            .map(|j| atom[j].1)
            .collect()
    };
    let cheap: Vec<usize> = (0..atom.len())
        .filter(|&i| rung_pols(i).iter().all(|&x| !x))
        .collect();
    if cheap.is_empty() {
        rng.random_range(0..atom.len())
    } else {
        cheap[rng.random_range(0..cheap.len())]
    }
}

/// Fixed choices needed to add one formal ANF atom to a dirty accumulator and
/// later remove that same atom with exactly the same residual convention.
#[derive(Clone, Copy, Debug)]
struct MicroAtomPlan {
    helper0: u16,
    helper1: u16,
    pivot: usize,
}

/// Add one atom to a dirty accumulator with a primitive width cap of two.
///
/// Degrees through three use the established one-helper dirty sweep. A
/// quartic uses two arbitrary-dirty helpers:
///
///   a ^= h1*d; h1 ^= h0*c; h0 ^= a*b; h1 ^= h0*c
///
/// repeated twice. The inner palindrome changes `h1` by `a*b*c` while
/// restoring `h0`; the two outer reads therefore add `a*b*c*d` and restore
/// both helpers. No clean-ancilla assumption is made.
fn emit_micro_atom_onto(
    acc: u16,
    atom: &[(u16, bool)],
    plan: MicroAtomPlan,
    menu: usize,
    seen: &mut SpellingLog,
    rng: &mut impl Rng,
    out: &mut Vec<XGate>,
) -> bool {
    if atom.len() <= 3 {
        return emit_atom_onto(acc, atom, plan.helper0, plan.pivot, menu, seen, rng, out);
    }
    debug_assert_eq!(atom.len(), 4, "micro-Gray supports degree <= 4 atoms");
    let ab = [atom[0], atom[1]];
    let h0c = [(plan.helper0, true), atom[2]];
    let h1d = [(plan.helper1, true), atom[3]];
    for _ in 0..2 {
        out.push(XGate::conj(acc, h1d).expect("micro target controls are distinct"));
        out.push(XGate::conj(plan.helper1, h0c).expect("micro helper controls are distinct"));
        out.push(XGate::conj(plan.helper0, ab).expect("micro rung literals are distinct"));
        out.push(XGate::conj(plan.helper1, h0c).expect("micro helper controls are distinct"));
    }
    false
}

/// Random full-rank four-share schedule over an operand's formal ANF atoms.
/// The rows are a disjoint, nonempty partition: their XOR is the operand and
/// no proper subset of row deltas is. This is deliberately a statement about
/// the formal atom basis; cancellations caused by a particular runtime input
/// do not change the schedule's rank.
fn micro_partition_atoms(
    atoms: &[Vec<(u16, bool)>],
    rng: &mut impl Rng,
) -> Option<[Vec<Vec<(u16, bool)>>; 4]> {
    if atoms.len() < 4 {
        return None;
    }
    let mut order: Vec<usize> = (0..atoms.len()).collect();
    order.shuffle(rng);
    let mut shares: [Vec<Vec<(u16, bool)>>; 4] = std::array::from_fn(|_| Vec::new());
    for (position, atom) in order.into_iter().enumerate() {
        shares[position % 4].push(atoms[atom].clone());
    }
    let mut row_order = [0usize, 1, 2, 3];
    row_order.shuffle(rng);
    Some(std::array::from_fn(|i| shares[row_order[i]].clone()))
}

/// Remove constant atoms without ever materializing `1 * H`: an odd empty-atom
/// parity is absorbed by complementing one live linear literal. Complementing
/// a one-literal conjunction adds exactly one to its ANF function.
fn absorb_constants_into_linear_atom(atoms: &mut Vec<Vec<(u16, bool)>>) -> bool {
    let odd = atoms.iter().filter(|atom| atom.is_empty()).count() % 2 != 0;
    atoms.retain(|atom| !atom.is_empty());
    if !odd {
        return true;
    }
    let Some(linear) = atoms.iter_mut().find(|atom| atom.len() == 1) else {
        return false;
    };
    linear[0].1 = !linear[0].1;
    true
}

#[derive(Clone, Debug)]
struct SentinelParts {
    /// Nonempty degree-one atoms, including the live literal into which any
    /// source/ledger constant was absorbed.
    low: Vec<Vec<(u16, bool)>>,
    /// Degrees 2..max-1: the only material allowed onto an accumulator.
    gathered: Vec<Vec<(u16, bool)>>,
    /// Every maximum-degree atom. These remain explicit in all transitions.
    high: Vec<Vec<(u16, bool)>>,
}

fn partition_max_degree_sentinel(atoms: &[Vec<(u16, bool)>]) -> Option<SentinelParts> {
    if atoms.iter().any(|atom| atom.is_empty()) {
        return None;
    }
    let max_degree = atoms.iter().map(Vec::len).max()?;
    if max_degree < 3 {
        return None;
    }
    let low: Vec<_> = atoms
        .iter()
        .filter(|atom| atom.len() == 1)
        .cloned()
        .collect();
    let gathered: Vec<_> = atoms
        .iter()
        .filter(|atom| (2..max_degree).contains(&atom.len()))
        .cloned()
        .collect();
    let high: Vec<_> = atoms
        .iter()
        .filter(|atom| atom.len() == max_degree)
        .cloned()
        .collect();
    if low.is_empty()
        || gathered.is_empty()
        || high.is_empty()
        || gathered.iter().any(|atom| atom.len() > 4)
    {
        return None;
    }
    Some(SentinelParts {
        low,
        gathered,
        high,
    })
}

/// Exact cap-two dirty ladder for a normalized width-three/four conjunction.
/// The caller controls the literal order. For a sentinel cross tail it passes
/// `[blind, high[0], high[1], ...]`, making rung zero cross-factor by
/// construction; neither helper transition can then equal the whole H atom.
fn emit_exact_dirty_cap2(
    target: u16,
    lits: &[(u16, bool)],
    helper0: u16,
    helper1: u16,
    out: &mut Vec<XGate>,
) {
    debug_assert!((3..=4).contains(&lits.len()));
    let rung0 = [lits[0], lits[1]];
    if lits.len() == 3 {
        let tail = [(helper0, true), lits[2]];
        for _ in 0..2 {
            out.push(XGate::conj(target, tail).expect("sentinel target controls distinct"));
            out.push(XGate::conj(helper0, rung0).expect("sentinel rung-zero controls distinct"));
        }
        return;
    }
    let rung1 = [(helper0, true), lits[2]];
    let tail = [(helper1, true), lits[3]];
    for _ in 0..2 {
        out.push(XGate::conj(target, tail).expect("sentinel target controls distinct"));
        out.push(XGate::conj(helper1, rung1).expect("sentinel rung-one controls distinct"));
        out.push(XGate::conj(helper0, rung0).expect("sentinel rung-zero controls distinct"));
        out.push(XGate::conj(helper1, rung1).expect("sentinel rung-one controls distinct"));
    }
}

/// Every ordered gate sequence that contributes the SAME function to `h`, so a
/// rung's two emissions can be spelled differently while still cancelling.
///
/// Why a menu is needed at all: the double sweep emits its rung twice, and the
/// old code cloned one prebuilt gate, so every laddered fragment planted a pair
/// of BYTE-IDENTICAL gates. On the shipped n=128 build 143,100 of 184,898
/// comp=0 width-2 instances (77.4%) sat in duplicate groups and 50.4% of the
/// whole circuit sat in exact pairs, so `sort | uniq -c` -- no execution, no
/// algebra -- located every ladder, its borrowed wire and two of its three
/// literals. Routing the rung through `emit_g57_form`'s coin did NOT fix it
/// (63.2% -> 69.6%), and the enumeration below says why: for a MIXED-polarity
/// pair that coin emits the same gate either way, so those pairs always matched.
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
fn rung_spellings(h: u16, lits: &[(u16, bool)]) -> (Vec<Vec<XGate>>, bool) {
    let g57 = |p: u16, q: u16| XGate::from_g57([h, p, q]);
    match *lits {
        [(w, p)] => (vec![vec![XGate::cnot(h, w)]], !p),
        [(xw, xp), (yw, yp)] => {
            let cx = || XGate::cnot(h, xw);
            let cy = || XGate::cnot(h, yw);
            let menu = match (xp, yp) {
                // ~x&y : short g57(h;x,y); long cnot_x + cnot_y + g57(h;y,x)
                (false, true) => vec![
                    vec![g57(xw, yw)],
                    vec![cx(), cy(), g57(yw, xw)],
                    vec![cy(), g57(yw, xw), cx()],
                    vec![g57(yw, xw), cx(), cy()],
                ],
                // x&~y : the same with the roles of x and y exchanged
                (true, false) => vec![
                    vec![g57(yw, xw)],
                    vec![cy(), cx(), g57(xw, yw)],
                    vec![cx(), g57(xw, yw), cy()],
                    vec![g57(xw, yw), cy(), cx()],
                ],
                // x&y and ~x&~y : two 2-gate spellings, both orderings of each
                (true, true) => vec![
                    vec![g57(xw, yw), cy()],
                    vec![cy(), g57(xw, yw)],
                    vec![g57(yw, xw), cx()],
                    vec![cx(), g57(yw, xw)],
                ],
                (false, false) => vec![
                    vec![g57(xw, yw), cx()],
                    vec![cx(), g57(xw, yw)],
                    vec![g57(yw, xw), cy()],
                    vec![cy(), g57(yw, xw)],
                ],
            };
            // Residual constant: only the both-negative pattern realizes its
            // conjunction outright; the other three carry a 1.
            (menu, xp || yp)
        }
        _ => unreachable!("rung_spellings takes 1..=2 literals"),
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
fn spellings_at(h: u16, lits: &[(u16, bool)], level: usize) -> (Vec<Vec<XGate>>, bool) {
    let (full, konst) = rung_spellings(h, lits);
    if level >= 2 {
        return (full, konst);
    }
    let min_len = full.iter().map(|m| m.len()).min().unwrap_or(0);
    let mut free: Vec<Vec<XGate>> = full.into_iter().filter(|m| m.len() == min_len).collect();
    if level == 0 {
        free.truncate(1);
    }
    (free, konst)
}

/// One emission of a ladder rung, spelled DIFFERENTLY from the previous one.
///
/// `prev` is the menu index the rung's last emission used; this picks uniformly
/// from everything else, so a rung's two copies are never byte-identical and the
/// duplicate-pair census that located every ladder finds nothing. All spellings
/// of a rung contribute the same function, so the borrow is still restored
/// exactly whichever pair is drawn. Returns the index used and the residual.
fn emit_rung(
    h: u16,
    rung_lits: &[(u16, bool)],
    g57: bool,
    level: usize,
    prev: Option<usize>,
    rng: &mut impl Rng,
    out: &mut Vec<XGate>,
) -> (Option<usize>, bool) {
    if !g57 || rung_lits.len() > 2 {
        out.push(
            XGate::conj(h, rung_lits.iter().copied())
                .expect("borrowed wires are distinct from the literals"),
        );
        return (None, false);
    }
    let (menu, konst) = spellings_at(h, rung_lits, level);
    let pick = if menu.len() < 2 {
        0
    } else {
        let choices: Vec<usize> = (0..menu.len()).filter(|i| Some(*i) != prev).collect();
        choices[rng.random_range(0..choices.len())]
    };
    out.extend(menu[pick].iter().cloned());
    (Some(pick), konst)
}

/// Dedupe an interleaved literal list in place, preserving order. Returns
/// None when two polarities meet on one wire (the conjunction is identically
/// zero and the fragment must be dropped).
/// Flatten a fragment's per-control atoms round-robin rather than end to end.
///
/// This matters only when the fragment is going to be LADDERED, but it is
/// applied unconditionally. A ladder parks partial products of a literal
/// PREFIX on borrowed wires; concatenated order makes some prefix exactly one
/// value's whole mask term, which would sit unmasked on a borrowed wire for
/// the length of the sweep. Interleaved, no prefix ever completes a single
/// value's term. A plain conjunction is order-insensitive, so applying it
/// everywhere costs nothing and keeps laddered and unladdered fragments from
/// being told apart by control order alone.
fn interleave_atoms(atoms: &[Vec<(u16, bool)>]) -> Vec<(u16, bool)> {
    let width = atoms.iter().map(|a| a.len()).max().unwrap_or(0);
    let mut lits = Vec::new();
    for i in 0..width {
        for atom in atoms {
            if let Some(&lit) = atom.get(i) {
                lits.push(lit);
            }
        }
    }
    lits
}

fn normalize_lits(lits: &mut Vec<(u16, bool)>) -> Option<()> {
    let mut seen: Vec<(u16, bool)> = Vec::with_capacity(lits.len());
    for &(w, p) in lits.iter() {
        match seen.iter().find(|&&(sw, _)| sw == w) {
            Some(&(_, sp)) if sp != p => return None,
            Some(_) => {}
            None => seen.push((w, p)),
        }
    }
    *lits = seen;
    Some(())
}

/// A fragment's atoms contain some wire with both polarities: the product is
/// identically 0 and would emit nothing.
fn atoms_contradict(atoms: &[Vec<(u16, bool)>]) -> bool {
    let mut lits = interleave_atoms(atoms);
    normalize_lits(&mut lits).is_none()
}

/// Ledger for the product-share encoding: per-value slot lists + constants.
/// Sources are frozen band VALUES, so slots are never disturbed by RGs or
/// CGs — the only emissions are the inject ramp, optional re-source churn,
/// optional band rolls (which move a band value to another wire without
/// changing it), the per-CG ANF folds, and the final strip.
struct ProdLedger {
    /// Per-value injection plan: the multiset of mask degrees each value
    /// carries (k copies of `deg` then k_hi copies of `deg_hi`).
    plan: Vec<usize>,
    /// Per-value count of EXTRA high-degree mask terms beyond `plan`. Zero
    /// everywhere unless `--prod-cg-jitter` is set.
    plan_extra: Vec<usize>,
    /// 2n — random helper draws (constant folds) stay off the band.
    carrier_total: usize,
    /// Physical wire currently holding each band variable. Starts as the
    /// contiguous home range `carrier_total..carrier_total+band_len` and is
    /// permuted by [`ProdLedger::roll`]; a rolled variable may sit anywhere,
    /// including inside the carrier space (its former wire becomes a carrier).
    loc: Vec<u16>,
    /// Narrow mode: emission width cap (0 = legacy wide fragments). Ladders
    /// borrow dirty carriers, so there is no scratch region to track.
    cap: usize,
    slots: Vec<Vec<ProdSlot>>,
    consts: Vec<bool>,
    used: std::collections::HashSet<ProdSlot>,
    // ---- distributed sourcing (src_dist) ----
    /// Sources are ordinary carriers rather than band variables.
    dist: bool,
    /// One linear carrier per value instead of a pair.
    single: bool,
    /// Gather each operand's mask sum onto a dirty accumulator (see
    /// [`ProdLedger::fold_cg_gray`]).
    gray_fold: bool,
    /// Four-share micro-Gray mode. Kept separate from `gray_fold` so mode 1's
    /// established gate stream and defaults remain byte-for-byte unchanged.
    micro_gray: bool,
    /// Sentinel Gray mode. Quadratic atoms alone visit dirty accumulators;
    /// degree-three-plus atoms remain explicit sentinels in every transition.
    sentinel_gray: bool,
    /// Per-gate mask swap-with-refresh (see [`ProdConfig::swap_refresh`]).
    /// Forces the expanded fold: every Gray mode is declined while set.
    swap_refresh: bool,
    /// Under the closing-slice design the strip's constant-discharge helper
    /// is drawn from the band space only: the strip bares the data wires
    /// progressively, and a helper read from a bared wire writes that value
    /// out as a local segment delta (the payload class flip_match caught).
    tail_band_helpers: bool,
    /// Emit DB-eligible fold fragments in the g57/CNOT vocabulary.
    g57_narrow: bool,
    ladder_cap: usize,
    cg_jitter: usize,
    rung_menu: usize,
    /// refs[wire] = live slot factors naming that wire. A wire with refs > 0
    /// is "named": nothing may write it until every naming slot is released.
    /// Distributed sourcing only — the banded build's analogue is `var_refs`,
    /// which counts band VARIABLES (slots name variables, and rolling moves
    /// variables between wires, so a wire-indexed count would be wrong the
    /// moment `--prod-roll` fires).
    refs: Vec<u32>,
    /// var_refs[var] = live slot factors naming that band variable. Zero means
    /// the variable's value can be rewritten with no bookkeeping at all: no
    /// mask reads it, so no carrier has to be patched. That is the whole
    /// economics of the drain set.
    var_refs: Vec<u32>,
    /// var_holders[var] = the values holding a live slot that names `var`, as
    /// a multiset (a value appears once per naming slot). The reverse index
    /// that makes retirement steering O(1) instead of a scan over every value.
    var_holders: Vec<Vec<u32>>,
    /// The DRAIN SET: band variables excluded from fresh slot draws, so their
    /// reference counts only fall. Retirement steering prefers slots naming a
    /// member; a member that reaches zero references is rewritten for a couple
    /// of gates and swapped out for a fresh variable. See `drain_rotate`.
    drain: Vec<u16>,
    /// Target drain-set size (0 = off). Sized off the retirement rate: the set
    /// has to be big enough that a fold's values usually hold a member's
    /// reference, or steering stalls and the rate is wasted.
    drain_cap: usize,
    /// Retirement sides per fold (see [`ProdConfig::swap_refresh`]).
    swap_sides: usize,
    /// Refill channel config, kept here because the drain rewrite shares
    /// `retire_refill`'s emission shape (see `rewrite_var`).
    refill_data: usize,
    fill_nl: usize,
    /// owner[wire] = the value currently carrying on that wire, refreshed
    /// from `GadgetState::pairs` whenever the pairing changes.
    owner: Vec<u32>,
    /// Per-value source-gate positions at which it is a target, with a cursor
    /// advanced as the body is emitted: the lookahead that keeps a fresh
    /// source from being written soon after it is drawn.
    hits: Vec<Vec<usize>>,
    cursor: Vec<usize>,
    pos: usize,
    horizon: usize,
    /// Value range distributed sourcing draws from (see `ProdConfig::src_lo`).
    src_lo: usize,
    src_hi: usize,
    /// A 64-sample bit-sliced simulation of the circuit emitted so far, under
    /// the zero-slice convention, advanced lazily.
    ///
    /// Band wires are generic bits by construction; CARRIERS ARE NOT. A value
    /// that is identically 0 on the zero slice has two EQUAL shares, so the
    /// carrier space contains exact linear degeneracies — measured: 23 exact
    /// wire-pair relations at 25% depth in a distributed build, against 1 in a
    /// banded one. A mask drawn across such a pair collapses: with opposite
    /// offsets the product is identically zero, with equal offsets it drops a
    /// degree. Either way the value is silently short a term, and the
    /// gadgetizer cannot see it, because it picks factors by wire id and never
    /// looks at what they carry. This simulation is what lets the draw look.
    sim: Vec<u64>,
    sim_cursor: usize,
    degenerate_rejects: u64,
    injected: u64,
    resourced: u64,
    /// Fold-coupled swap-with-refresh operations (one per side per gate).
    swapped: u64,
    rolled: u64,
    migrated: u64,
    retired: u64,
    /// Drain-set refreshes completed, and retirements that steering placed on
    /// a drain member (the numerator and the bandwidth behind the turnover
    /// count `drained / band`).
    drained: u64,
    drain_steered: u64,
    next_retire: usize,
    /// Shuffled sweep order for retirement (see `retire_refill`).
    retire_queue: Vec<u16>,
    /// Whether any refill drew a carrier source (see `ProdConfig::refill_data`).
    refill_used_carriers: bool,
    cg_fragments: u64,
    /// Fold fragments narrow enough for the frozen-DB channel (<= 2 controls
    /// at the production --db-ctrl-cap). The fold is the material that carries
    /// the computation, so this is the fraction of the CORE that phase A's
    /// re-encoding can ever reach.
    cg_narrow: u64,
    cg_laddered: u64,
    cg_fossils: u64,
    /// CG blocks emitted by the Gray fold rather than the odometer.
    cg_gray: u64,
    /// Mode-3 blocks emitted by the max-degree sentinel schedule.
    cg_sentinel: u64,
    /// Original/emitted fragment counts and requested branch floors in each
    /// opt-in distributed seven-carrier fold.
    distributed_fold_original_fragments: Vec<usize>,
    distributed_fold_fragments: Vec<usize>,
    distributed_fold_floors: Vec<usize>,
    ledger_consts: u64,
}

/// Target drain-set size for a config, or 0 when the mechanism is off.
///
/// Two competing pressures. Too SMALL and steering stalls: a retirement can
/// only touch a slot of the value the fold happens to be folding, so the set
/// has to be broad enough that those values usually hold a member's reference
/// — with `|D|` members at ~`f` references each spread over `n` values, a
/// value holds one with probability `1 - exp(-|D|*f/n)`, which wants `|D|` on
/// the order of the retirement rate times a constant. Too LARGE and the draw
/// pool shrinks: every member is excluded from `draw_slot`, and the band is
/// also carrying the disjointness and no-repeat constraints.
///
/// `PROD_DRAIN` overrides for the A/B (0 disables the drain set entirely,
/// leaving the raised retirement rate as an isolated arm).
fn drain_cap(cfg: &ProdConfig, band_len: usize) -> usize {
    if !cfg.enabled() || cfg.dist() || cfg.swap_refresh == 0 || band_len == 0 {
        return 0;
    }
    if let Ok(v) = std::env::var("PROD_DRAIN") {
        if let Ok(v) = v.parse::<usize>() {
            return v.min(band_len.saturating_sub(cfg.max_deg() + 1));
        }
    }
    // band/6 is the ceiling that keeps the draw pool comfortable at every
    // sizing that ships (42 of 256 at n=128); 12 per side is what availability
    // wants at the production reference count.
    (12 * cfg.swap_refresh).min(band_len / 6)
}

impl ProdLedger {
    fn new(
        n: usize,
        cfg: &ProdConfig,
        carrier_total: usize,
        sched: Option<Vec<Vec<usize>>>,
    ) -> ProdLedger {
        assert!(
            cfg.gray_fold <= 3,
            "prod gray_fold mode must be 0 (expanded), 1 (aggregate), 2 (micro), or 3 (sentinel)"
        );
        let band_len = cfg.band_size(n);
        let mut plan: Vec<usize> = Vec::new();
        plan.extend(std::iter::repeat(cfg.deg.max(2)).take(cfg.k));
        plan.extend(std::iter::repeat(cfg.deg_hi.max(2)).take(cfg.k_hi));
        if cfg.dist() {
            // Slot space is the carriers themselves, minus the owner's own two
            // and the sibling of every named wire; C(2n, deg) dwarfs the live
            // demand at every production sizing, but keep the guard honest.
            assert!(
                carrier_total >= 2 * cfg.max_deg() + 4,
                "distributed sourcing needs more carriers than one slot's factors"
            );
            assert!(
                cfg.roll == 0,
                "--prod-roll relocates band variables; distributed sourcing has no band"
            );
        }
        if cfg.enabled() && !cfg.dist() {
            let max_deg = cfg.max_deg();
            assert!(
                band_len >= max_deg + 1,
                "prod band needs >= max_deg+1 wires"
            );
            // Distinct degree-`d` factor tuples for the widest tier: C(band,d)*2^d.
            let mut tuple_space: u128 = 1;
            for i in 0..max_deg {
                tuple_space = tuple_space * (band_len - i) as u128 / (i as u128 + 1);
            }
            tuple_space = tuple_space.saturating_mul(1u128 << max_deg);
            assert!(
                tuple_space >= (2 * n * cfg.k_total()) as u128,
                "prod source band too small: {band_len} wires give {tuple_space} degree-{max_deg} \
                 slot tuples for {} live slots",
                n * cfg.k_total()
            );
        }
        let hits = sched.unwrap_or_else(|| vec![Vec::new(); n]);
        ProdLedger {
            plan,
            plan_extra: vec![0; n],
            carrier_total,
            loc: (carrier_total as u16..(carrier_total + band_len) as u16).collect(),
            cap: if cfg.narrow() { cfg.max_width } else { 0 },
            slots: vec![Vec::new(); n],
            consts: vec![false; n],
            used: std::collections::HashSet::new(),
            dist: cfg.dist(),
            single: cfg.single_carrier(),
            gray_fold: cfg.gray_fold == 1,
            micro_gray: cfg.gray_fold == 2,
            sentinel_gray: cfg.gray_fold == 3,
            swap_refresh: cfg.swap_refresh > 0,
            tail_band_helpers: cfg.close_slice > 0,
            g57_narrow: cfg.g57_narrow > 0,
            ladder_cap: cfg.ladder_cap,
            cg_jitter: cfg.cg_jitter,
            rung_menu: cfg.rung_menu,
            refs: vec![0; carrier_total],
            var_refs: vec![0; band_len],
            var_holders: vec![Vec::new(); band_len],
            drain: Vec::new(),
            drain_cap: drain_cap(cfg, band_len),
            swap_sides: cfg.swap_refresh,
            refill_data: cfg.refill_data,
            fill_nl: cfg.fill_nl,
            owner: vec![u32::MAX; carrier_total],
            hits,
            cursor: vec![0; n],
            pos: 0,
            horizon: cfg.horizon(n),
            src_lo: cfg.src_range(n).0,
            src_hi: cfg.src_range(n).1,
            sim: Vec::new(),
            sim_cursor: 0,
            degenerate_rejects: 0,
            injected: 0,
            resourced: 0,
            swapped: 0,
            rolled: 0,
            migrated: 0,
            retired: 0,
            drained: 0,
            drain_steered: 0,
            next_retire: 0,
            retire_queue: Vec::new(),
            refill_used_carriers: false,
            cg_fragments: 0,
            cg_narrow: 0,
            cg_laddered: 0,
            cg_fossils: 0,
            cg_gray: 0,
            cg_sentinel: 0,
            distributed_fold_original_fragments: Vec::new(),
            distributed_fold_fragments: Vec::new(),
            distributed_fold_floors: Vec::new(),
            ledger_consts: 0,
        }
    }

    /// Refresh the wire -> owning-value map. The pairing changes under RG1/RG2
    /// (and the W_i ramp), so every barrier point re-reads it; it is O(n) on a
    /// per-source-gate cadence, not per emitted gate.
    fn sync(&mut self, state: &GadgetState) {
        if !self.dist {
            return;
        }
        for w in self.owner.iter_mut() {
            *w = u32::MAX;
        }
        for value in 0..state.n {
            let (c0, c1) = state.pairs[value];
            self.owner[c0] = value as u32;
            self.owner[c1] = value as u32;
        }
    }

    /// The carrier of `value` that no live slot names — always one of the two
    /// under invariant S, and the only wire the encoding may write. When both
    /// are free the choice stays random, as in the band build.
    fn free_carrier(&self, value: usize, state: &GadgetState, rng: &mut impl Rng) -> u16 {
        let (c0, c1) = state.pairs[value];
        let mut coin = || if rng.random_bool(0.5) { c0 } else { c1 } as u16;
        if !self.dist {
            return coin();
        }
        match (self.refs[c0], self.refs[c1]) {
            (0, 0) => coin(),
            (0, _) => c0 as u16,
            (_, 0) => c1 as u16,
            _ => unreachable!("invariant S: a value never has both carriers named"),
        }
    }

    /// Position of the next source gate targeting `value`, or `usize::MAX`.
    /// `self.pos` only advances, so the per-value cursor is amortized O(1).
    fn next_hit(&mut self, value: usize) -> usize {
        let hits = &self.hits[value];
        let cur = &mut self.cursor[value];
        while *cur < hits.len() && hits[*cur] < self.pos {
            *cur += 1;
        }
        hits.get(*cur).copied().unwrap_or(usize::MAX)
    }

    /// Register a slot's factors. `value` is the holder, which the banded
    /// build needs for the reverse index (`var_holders`) that retirement
    /// steering reads; distributed sourcing keeps its wire-indexed count.
    ///
    /// ⚠️ Every `slots[value].push` must be paired with this and every
    /// `slots[value].remove` with `drop_refs`, or `var_refs` goes stale — and
    /// a stale zero is what would let `rewrite_var` overwrite a LIVE band
    /// variable and silently break the decode. `emit_slot` debug-asserts the
    /// invariant from the other side.
    fn add_refs(&mut self, value: usize, slot: &ProdSlot) {
        if self.dist {
            for w in slot.wires() {
                self.refs[w] += 1;
            }
            return;
        }
        for &(b, _) in &slot.factors {
            self.var_refs[b as usize] += 1;
            self.var_holders[b as usize].push(value as u32);
        }
    }

    fn drop_refs(&mut self, value: usize, slot: &ProdSlot) {
        if self.dist {
            for w in slot.wires() {
                self.refs[w] -= 1;
            }
            return;
        }
        for &(b, _) in &slot.factors {
            self.var_refs[b as usize] -= 1;
            let holders = &mut self.var_holders[b as usize];
            if let Some(p) = holders.iter().position(|&v| v == value as u32) {
                holders.swap_remove(p);
            } else {
                debug_assert!(false, "var_holders[{b}] lost value {value}");
            }
        }
    }

    /// Draw a fresh degree-`deg` slot for `value` over ordinary carriers.
    ///
    /// A candidate wire `w` is legal when (a) it is not a carrier of `value`
    /// itself, (b) it is not in `forbidden` (the wires this emission is about
    /// to write or read completely), and (c) its sibling carrier is unnamed —
    /// which is what maintains invariant S, and hence what guarantees every
    /// value keeps a writable carrier and no fragment ever sees both carriers
    /// of a third value. Preference, not a requirement, is given to wires
    /// whose owning value is not targeted within the lookahead horizon.
    fn draw_slot_dist(
        &mut self,
        value: usize,
        deg: usize,
        forbidden: &[usize],
        state: &GadgetState,
        rng: &mut impl Rng,
    ) -> ProdSlot {
        let (own0, own1) = state.pairs[value];
        // Enumerate the legal candidates once (O(2n)) rather than rejection-
        // sampling them: at small n the legal tuple space is tight enough that
        // sampling-with-rejection spins essentially forever, and at production
        // n the enumeration is still one cheap pass per draw.
        //
        // Candidates are grouped by OWNING VALUE, and at most one is taken per
        // group — a value with both carriers free offers two, and taking both
        // would put its whole sharing into one fragment.
        let mut soon: Vec<(u32, [Option<u16>; 2])> = Vec::new();
        let mut later: Vec<(u32, [Option<u16>; 2])> = Vec::new();
        for owner in self.src_lo..self.src_hi {
            let (s0, s1) = state.pairs[owner];
            if s0 == own0 || s0 == own1 || s1 == own0 || s1 == own1 {
                continue; // the owner IS this value
            }
            let legal = |w: usize, sibling: usize| -> Option<u16> {
                // Invariant S: name a wire only while its sibling is free.
                (!forbidden.contains(&w) && self.refs[sibling] == 0).then_some(w as u16)
            };
            let pick = [legal(s0, s1), legal(s1, s0)];
            if pick[0].is_none() && pick[1].is_none() {
                continue;
            }
            let due = self.next_hit(owner);
            if due != usize::MAX && due < self.pos + self.horizon {
                soon.push((owner as u32, pick));
            } else {
                later.push((owner as u32, pick));
            }
        }
        // Lookahead is a preference: prefer groups that are not written soon,
        // and fall back to the rest rather than fail.
        use rand::seq::SliceRandom;
        later.shuffle(rng);
        soon.shuffle(rng);
        let mut pool: Vec<(u32, [Option<u16>; 2])> = later.into_iter().chain(soon).collect();
        assert!(
            pool.len() >= deg,
            "distributed sourcing found only {} legal source values for a degree-{deg} slot \
             (need {deg}); the gadget is too narrow for --prod-deg/--prod-deg-hi",
            pool.len()
        );
        // A handful of attempts to land a tuple no live slot already carries;
        // duplicates are a diversity nit, not a correctness problem (the dedup
        // set only steers the draw), so accept one rather than spin.
        let mut fallback: Option<Vec<u16>> = None;
        for attempt in 0..64 {
            if attempt > 0 {
                // Re-roll which groups are in front, so a collision retries a
                // different tuple rather than the same wires with new signs.
                pool.shuffle(rng);
            }
            let mut wires: Vec<u16> = Vec::with_capacity(deg);
            for (_, pick) in pool.iter().take(deg) {
                let w = match (pick[0], pick[1]) {
                    (Some(a), Some(b)) => {
                        if rng.random_bool(0.5) {
                            a
                        } else {
                            b
                        }
                    }
                    (Some(a), None) => a,
                    (None, Some(b)) => b,
                    (None, None) => unreachable!("empty groups are skipped"),
                };
                wires.push(w);
            }
            wires.sort_unstable();
            // Reject a factor set that is degenerate on the live simulation:
            // carriers, unlike band wires, contain exact linear relations.
            // This is a QUALITY filter, never a correctness one — on a narrow
            // gadget every legal tuple can be degenerate, and refusing to
            // return one would abort a build that is perfectly correct. So the
            // first rejected candidate is kept as a fallback.
            if self.tuple_is_degenerate(&wires) || self.duplicates_live_term(value, &wires) {
                self.degenerate_rejects += 1;
                if fallback.is_none() {
                    fallback = Some(wires);
                }
                continue;
            }
            // Offsets are all-positive here, unlike the band draw. Under
            // uniform sources the offsets buy nothing statistically (the term
            // fires at 2^-deg either way), but carriers are NOT uniform: they
            // contain exactly-equal pairs (a value that is identically 0 on
            // the zero slice has c0 == c1), and a term drawn over two equal
            // wires with OPPOSITE offsets is identically zero — a dead mask,
            // silently leaving that value short a term. Measured: dead terms
            // occur only in distributed builds, concentrated early
            // (docs/DISTRIBUTED_SOURCE_ENCODING.md).
            let factors: Vec<(u16, bool)> = wires.into_iter().map(|w| (w, false)).collect();
            let slot = ProdSlot { factors };
            if !self.used.contains(&slot) || attempt == 63 {
                self.used.insert(slot.clone());
                return slot;
            }
        }
        let wires = fallback.expect("at least one candidate tuple was formed");
        let slot = ProdSlot {
            factors: wires.into_iter().map(|w| (w, false)).collect(),
        };
        self.used.insert(slot.clone());
        slot
    }

    fn enabled(&self) -> bool {
        !self.plan.is_empty()
    }

    /// Wire space the narrow ladder may borrow from: everything. Borrows are
    /// restored within the fragment, so a band wire is as safe as a carrier —
    /// but the ladder prefers carriers and reaches past `carrier_total` only
    /// when a fragment's literals have eaten the carrier pool.
    fn borrow_total(&self) -> usize {
        self.carrier_total + self.loc.len()
    }

    /// A fresh, currently-unused degree-`deg` slot over the band variables,
    /// whose factors are DISJOINT from every variable this value's other live
    /// slots already name.
    ///
    /// Why disjointness matters (measured). The statistical strength of a
    /// value's mask is the piling-up product over its terms,
    /// `1/2 + (1/2) PROD_j (1 - 2^(1-d_j))`, and that formula assumes the
    /// terms are INDEPENDENT. Two terms sharing a source variable are not:
    /// `w_a w_b XOR w_a w_c = w_a (w_b XOR w_c)` has the strength of a SINGLE
    /// degree-2 term, not two. Drawing each term independently over the whole
    /// band (the previous behaviour) collides often — 14 draws over a 56-wire
    /// band expect ~1.6 colliding pairs — so added terms paid full fold cost
    /// and returned less than full hiding, and the shortfall GREW with `k`.
    /// Measured at n=64, plan `[2,2,3,3]`: predicted agreement 0.5703 for
    /// disjoint terms, 0.5874 as previously drawn, 0.5873 observed — the gap
    /// was entirely this. Excluding the value's own live variables is free
    /// (the band has ample slot space) and makes the piling-up figure the
    /// quantity the design actually gets.
    ///
    /// Only the value's OWN terms must be disjoint; sharing sources ACROSS
    /// values is both harmless to each value's own statistics and necessary,
    /// since the band is far smaller than the value count.
    fn draw_slot(&mut self, value: usize, deg: usize, rng: &mut impl Rng) -> ProdSlot {
        let band_len = self.loc.len() as u16;
        // Variables this value's live slots already name.
        let mut taken: Vec<u16> = Vec::new();
        for slot in &self.slots[value] {
            for &(b, _) in &slot.factors {
                if !taken.contains(&b) {
                    taken.push(b);
                }
            }
        }
        // Enforce disjointness only when the band can still afford it; a band
        // sized near the correctness minimum falls back to the old best-effort
        // draw rather than panicking. `PROD_DISJOINT=0` restores the legacy
        // independent-per-term draw, for A/B measurement of the effect above.
        let enforce = (band_len as usize).saturating_sub(taken.len()) >= deg
            && std::env::var("PROD_DISJOINT")
                .map(|v| v != "0")
                .unwrap_or(true);
        // Drain-set members take no new references — that is what lets their
        // counts fall to zero and makes the rewrite free. Defensive width
        // check for the same reason `enforce` has one: excluding more of the
        // band than it can spare would spin the draw loop below forever.
        let hold_drain = (band_len as usize).saturating_sub(self.drain.len()) >= deg + 1;
        for _ in 0..100_000 {
            let mut vars: Vec<u16> = Vec::with_capacity(deg);
            while vars.len() < deg {
                let b = rng.random_range(0..band_len);
                if hold_drain && self.drain.contains(&b) {
                    continue;
                }
                if !vars.contains(&b) && !(enforce && taken.contains(&b)) {
                    vars.push(b);
                }
            }
            vars.sort_unstable();
            let factors: Vec<(u16, bool)> = vars
                .into_iter()
                .map(|b| (b, rng.random::<bool>()))
                .collect();
            let slot = ProdSlot { factors };
            if !self.used.contains(&slot) {
                self.used.insert(slot.clone());
                return slot;
            }
        }
        panic!("prod source band exhausted; raise --prod-band");
    }

    /// Emit the slot's fragment onto a random carrier of `value`: one
    /// conjunction over the product's factor literals — the product is
    /// computed only inside the gate's firing condition, never onto a wire
    /// (narrow mode ladders it; the chain carries only partial prefixes,
    /// never the whole term). Self-inverse over GF(2) up to the returned
    /// constant parity, which the caller folds into the value's ledger
    /// constant. Used identically for inject and strip.
    fn emit_slot(
        &self,
        value: usize,
        slot: &ProdSlot,
        state: &GadgetState,
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) -> bool {
        // Distributed sourcing writes the value's unnamed carrier; the band
        // build has no named carriers at all and keeps the coin flip.
        let target = self.free_carrier(value, state, rng);
        let lits = slot.lits(&self.loc);
        self.debug_check_fragment(target, &lits, state);
        if self.cap >= 2 {
            let (s0, s1) = state.pairs[value];
            let sib = [if s0 as u16 == target {
                s1 as u16
            } else {
                s0 as u16
            }];
            emit_narrow_fragment(
                target,
                &lits,
                self.cap,
                &self.ladder_borrow_pool(state),
                self.borrow_total(),
                &sib,
                // A mask slot is ONE atom, so no 2-subset of a degree-3 term
                // can equal it and the admissibility test never binds here --
                // pass it anyway so the rule lives in one place.
                std::slice::from_ref(&lits),
                self.rung_menu,
                rng,
                out,
            )
        } else if lits.len() > 2 && lits.len() <= self.ladder_cap {
            // The fold is where most of the wide gates are, but it is not the
            // only place: a degree-3 mask term is a 3-control conjunction every
            // time it is injected, re-sourced or stripped. Leaving those out
            // would cap the ceiling's reach at the fold's share of the fossils
            // and leave a residue that is, worse, ATTRIBUTABLE -- the surviving
            // wide gates would be exactly the slot emissions, which name a
            // value's mask sources directly.
            let sibling = self.sibling_map(state);
            let mut forbidden: Vec<u16> = vec![sibling[target as usize % sibling.len()]];
            forbidden.extend(
                lits.iter()
                    .map(|&(w, _)| w)
                    .filter(|&w| (w as usize) < sibling.len())
                    .map(|w| sibling[w as usize]),
            );
            emit_narrow_fragment(
                target,
                &lits,
                2,
                &self.ladder_borrow_pool(state),
                self.borrow_total(),
                &forbidden,
                std::slice::from_ref(&lits),
                self.rung_menu,
                rng,
                out,
            )
        } else if lits.len() <= 2 {
            // Same treatment fold_cg gives its narrow fragments. This was a
            // bare conjunction purely by omission: the caller already XORs the
            // returned residual into the ledger, so the g57 spelling costs
            // nothing extra in bookkeeping and one gate at most in size, and a
            // mask term is emitted on every inject, re-source and strip.
            emit_g57_form(target, &lits, rng, out)
        } else {
            out.push(XGate::conj(target, lits).expect("band sources are distinct wires"));
            false
        }
    }

    fn inject(
        &mut self,
        value: usize,
        deg: usize,
        state: &GadgetState,
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) {
        self.inject_avoiding(value, deg, &[], state, rng, out);
    }

    /// `inject`, with wires this emission must not name (the wire being
    /// released, and the carriers a fold is about to read or write).
    fn inject_avoiding(
        &mut self,
        value: usize,
        deg: usize,
        forbidden: &[usize],
        state: &GadgetState,
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) {
        let slot = if self.dist {
            self.sync(state);
            self.advance_sim(out, rng);
            self.draw_slot_dist(value, deg, forbidden, state, rng)
        } else {
            self.draw_slot(value, deg, rng)
        };
        let konst = self.emit_slot(value, &slot, state, rng, out);
        self.consts[value] ^= konst;
        self.add_refs(value, &slot);
        self.slots[value].push(slot);
        self.injected += 1;
    }

    /// Release `wire` so it can be written: re-source every live slot that
    /// names it (fresh term first, old term stripped after, exactly as
    /// `resource` does, so no value is ever momentarily bare). The fresh draw
    /// avoids `wire` itself, and each emission writes its owner's free
    /// carrier, so releasing one wire never dirties another named one.
    fn release_wire(
        &mut self,
        wire: usize,
        barrier: &[usize],
        state: &GadgetState,
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) {
        if !self.dist || self.refs[wire] == 0 {
            return;
        }
        while self.refs[wire] > 0 {
            let found = (0..state.n).find_map(|value| {
                self.slots[value]
                    .iter()
                    .position(|s| s.wires().any(|w| w == wire))
                    .map(|idx| (value, idx))
            });
            let Some((value, idx)) = found else {
                debug_assert!(false, "refs[{wire}] > 0 with no slot naming it");
                break;
            };
            let deg = self.slots[value][idx].factors.len();
            // Forbid the WHOLE barrier, not just this wire: a fresh draw that
            // landed on another wire this same emission is about to overwrite
            // would be released again (or, worse, after that wire had already
            // been cleared) — the one way the discipline could still let a
            // live mask meet a write.
            self.inject_avoiding(value, deg, barrier, state, rng, out);
            let old = self.slots[value].remove(idx);
            let konst = self.emit_slot(value, &old, state, rng, out);
            self.consts[value] ^= konst;
            self.drop_refs(value, &old);
            self.used.remove(&old);
            self.migrated += 1;
        }
    }

    /// Release every wire in `wires` (a write barrier for one emission).
    fn release(
        &mut self,
        wires: &[usize],
        state: &GadgetState,
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) {
        if !self.dist {
            return;
        }
        self.sync(state);
        for &w in wires {
            self.release_wire(w, wires, state, rng, out);
        }
        debug_assert!(
            wires.iter().all(|&w| self.refs[w] == 0),
            "write barrier left a named wire behind"
        );
    }

    /// Advance the lookahead clock to source-gate position `pos`, and (when
    /// PROD_BARE_CENSUS is set) sample how many values are currently bare.
    fn set_pos(&mut self, pos: usize) {
        self.pos = pos;
    }

    /// Census hook: called per source gate when the env var is set.
    fn bare_census(&mut self, state: &GadgetState, out: &[XGate], rng: &mut impl Rng) {
        if !self.enabled() || std::env::var("PROD_BARE_CENSUS").is_err() {
            return;
        }
        if self.pos % 200 != 0 {
            return;
        }
        self.advance_sim(out, rng);
        let bare = self.bare_values(state);
        println!(
            "[bare] pos={} bare_values={}/{} gates={}",
            self.pos,
            bare,
            state.n,
            out.len()
        );
        if let Ok(spec) = std::env::var("PROD_DUMP_PAIRS") {
            // Is a given wire pair the two carriers of ONE value? That is the
            // difference between "a value lost its mask" and "the leak runs
            // across values", and the two have completely different fixes.
            for want in spec.split(',') {
                let Some((a, b)) = want.split_once(':') else {
                    continue;
                };
                let (a, b): (usize, usize) = (a.parse().unwrap(), b.parse().unwrap());
                let same = (0..state.n).find(|&v| {
                    let (c0, c1) = state.pairs[v];
                    (c0 == a && c1 == b) || (c0 == b && c1 == a)
                });
                match same {
                    Some(v) => println!("  [pairs] ({a},{b}) ARE both carriers of value {v}"),
                    None => {
                        let ov = |w: usize| {
                            (0..state.n).find(|&v| {
                                let (c0, c1) = state.pairs[v];
                                c0 == w || c1 == w
                            })
                        };
                        println!(
                            "  [pairs] ({a},{b}) are carriers of DIFFERENT values {:?} and {:?}",
                            ov(a),
                            ov(b)
                        );
                    }
                }
            }
        }
    }

    /// Bring the 64-sample simulation up to the end of `out`. Seeded on first
    /// use with the zero-slice convention (random data on the low half of the
    /// values' wires, zeros elsewhere), which is the distribution every
    /// measurement in the project reads the gadget under.
    fn advance_sim(&mut self, out: &[XGate], rng: &mut impl Rng) {
        if !self.enabled() {
            return;
        }
        if self.sim.is_empty() {
            // The full wire space, not just the carriers: in band mode the
            // replayed gates target band wires above `carrier_total` (the fill)
            // and `slot_product` indexes through `loc`, which points there too.
            self.sim = vec![0u64; self.borrow_total()];
            for w in 0..self.carrier_total / 4 {
                self.sim[w] = rng.random::<u64>();
            }
        }
        for gate in &out[self.sim_cursor..] {
            gate.apply_lanes(&mut self.sim);
        }
        self.sim_cursor = out.len();
    }

    /// How many values are BARE right now: their live mask terms XOR to zero
    /// on every sample, so the value decodes as a plain `c0 ^ c1` and is an
    /// exactly affine function of two wires. This is the leak the heatmap
    /// actually finds — every predictive relation measured had support 2 — so
    /// counting it directly is the difference between knowing the mechanism
    /// and guessing at it. Works in band mode too, for the control.
    fn bare_values(&self, state: &GadgetState) -> usize {
        if self.sim.is_empty() {
            return 0;
        }
        (0..state.n)
            .filter(|&v| {
                !self.slots[v].is_empty()
                    && self.slots[v]
                        .iter()
                        .fold(0u64, |acc, s| acc ^ self.slot_product(s))
                        == 0
            })
            .count()
    }

    /// The product a slot's factors evaluate to, on the 64-sample simulation.
    fn slot_product(&self, slot: &ProdSlot) -> u64 {
        let mut prod = !0u64;
        for (w, a) in slot.lits(&self.loc) {
            let v = self.sim[w as usize];
            prod &= if a { v } else { !v };
        }
        prod
    }

    /// Would this factor set duplicate a term the value already carries?
    ///
    /// The dedup set compares slots by WIRE TUPLE, which is the right notion
    /// when sources are band variables — distinct band wires are distinct
    /// bits. Over carriers it is not: the carrier space contains exactly-equal
    /// pairs, so two slots with completely different wire tuples can be the
    /// SAME FUNCTION. When that happens to the two terms of one value they
    /// cancel, `m1 ^ m2 = 0`, and the value decodes as a bare `c0 ^ c1` — an
    /// exactly affine value, which is precisely the leak the heatmap finds
    /// (every leaking relation measured had support exactly 2). The tuple
    /// dedup cannot see it; only evaluating can.
    fn duplicates_live_term(&self, value: usize, wires: &[u16]) -> bool {
        if self.sim.is_empty() {
            return false;
        }
        let mut prod = !0u64;
        for &w in wires {
            prod &= self.sim[w as usize];
        }
        self.slots[value]
            .iter()
            .any(|s| self.slot_product(s) == prod)
    }

    /// Would this factor set collapse? Rejects a tuple whose wires are not
    /// distinct as BITS (equal or complementary on every sample), and one
    /// whose product is constant. Both make the term degenerate: it either
    /// vanishes or loses a degree, and the value silently carries less mask
    /// than its plan says. 64 samples is a cheap filter, not a proof — a pair
    /// that agrees on all 64 is overwhelmingly an exact relation, and one that
    /// does not is certainly fine.
    fn tuple_is_degenerate(&self, wires: &[u16]) -> bool {
        if self.sim.is_empty() {
            return false;
        }
        for (i, &a) in wires.iter().enumerate() {
            for &b in &wires[i + 1..] {
                let (x, y) = (self.sim[a as usize], self.sim[b as usize]);
                if x == y || x == !y {
                    return true;
                }
            }
        }
        let mut prod = !0u64;
        for &w in wires {
            prod &= self.sim[w as usize];
        }
        prod == 0
    }

    /// Gate-local non-completeness: one emitted fragment must never touch both
    /// carriers of any value (as literals, or as one literal and the target).
    /// That is the property the share-native fold is FOR — the legacy g57
    /// gadget hides at degree 2 precisely because no gate reconstructs an
    /// operand — and it is invisible to any endpoint test, so it is asserted
    /// at the point of emission. Under distributed sourcing it is also the
    /// property most at risk, since mask literals now live on carriers.
    fn debug_check_fragment(&self, target: u16, lits: &[(u16, bool)], state: &GadgetState) {
        if !self.dist {
            return;
        }
        // Repeated literals on ONE wire are fine and common — two operands'
        // masks may share a source, and `XGate::conj` folds equal-polarity
        // duplicates (opposite polarities make the fragment identically zero
        // and it is dropped). What must never happen is two DISTINCT wires
        // that are the two carriers of one value.
        let mut seen_wires: Vec<u16> = Vec::with_capacity(lits.len() + 1);
        let mut seen_values: Vec<u32> = Vec::with_capacity(lits.len() + 1);
        for wire in std::iter::once(target).chain(lits.iter().map(|&(w, _)| w)) {
            if seen_wires.contains(&wire) {
                continue;
            }
            seen_wires.push(wire);
            let v = self.owner[wire as usize];
            debug_assert_ne!(v, u32::MAX, "fragment touches a wire with no owning value");
            debug_assert!(
                !seen_values.contains(&v),
                "fragment sees both carriers of value {v} (target {target}, wires {:?}, pair {:?})",
                lits.iter().map(|&(w, _)| w).collect::<Vec<_>>(),
                state.pairs[v as usize]
            );
            seen_values.push(v);
        }
    }

    /// W1 ramp: the planned mask multiset per value (k deg-`deg` + k_hi
    /// deg-`deg_hi`), right after the sharing bookend, so every value is
    /// product-masked before its first body use.
    fn inject_all(&mut self, state: &GadgetState, rng: &mut impl Rng, out: &mut Vec<XGate>) {
        if !self.enabled() {
            return;
        }
        if self.cg_jitter > 0 {
            let p = (self.cg_jitter.min(100) as f64) / 100.0;
            for value in 0..state.n {
                self.plan_extra[value] = usize::from(rng.random_bool(p));
            }
        }
        for value in 0..state.n {
            for i in 0..self.plan.len() {
                let deg = self.plan[i];
                self.inject(value, deg, state, rng, out);
            }
            // Jitter is EXTRA terms only, never fewer. The operating point a
            // build commits to is the weakest value in it, so removing a term
            // anywhere would move the commitment; adding one can only raise a
            // value above the floor. What it buys is that arity-2 CG blocks
            // no longer all emit (1+k_total)^2 fragments -- the count varies
            // with which values the source gate happens to read, and a block
            // boundary stops being findable by counting to 16.
            // The extra term is the LOW-degree one, not a copy of the high
            // one. Both break the count identically -- k_total goes 3 -> 4 and
            // an arity-2 block emits 20 or 25 fragments instead of 16 -- but a
            // degree-3 extra widens the fold's fragments, and measurement says
            // that is expensive in exactly the currency the width ceiling is
            // spending: at n=128 a high-degree jitter of 50% cost +14% gates
            // and +32% MORE wide gates, for no change in store reach.
            let deg = self.plan.first().copied().unwrap_or(2);
            for _ in 0..self.plan_extra[value] {
                self.inject(value, deg, state, rng, out);
            }
        }
    }

    /// RG3': re-randomize one slot of a random value — inject the fresh term
    /// FIRST (same degree as the one it replaces), then strip the old one, so
    /// the value is never momentarily bare.
    fn resource(&mut self, state: &GadgetState, rng: &mut impl Rng, out: &mut Vec<XGate>) {
        if !self.enabled() {
            return;
        }
        // Steered at the drain set when one is running: this channel can reach
        // ANY value, which is what makes it the drain's cover for the tiers
        // `swap_refresh_side` cannot touch (it retires base-degree slots only,
        // so a drain member held by the deg-3 tower term would otherwise never
        // come free and would cap the turnover rate on its own).
        let steered = self.drain_pick(rng);
        let (value, old_index) = match steered {
            Some(vs) => {
                self.drain_steered += 1;
                vs
            }
            None => {
                let value = rng.random_range(0..state.n);
                if self.slots[value].is_empty() {
                    return;
                }
                (value, rng.random_range(0..self.slots[value].len()))
            }
        };
        let deg = self.slots[value][old_index].factors.len();
        self.inject(value, deg, state, rng, out);
        let old = self.slots[value].remove(old_index);
        let konst = self.emit_slot(value, &old, state, rng, out);
        self.consts[value] ^= konst;
        self.drop_refs(value, &old);
        self.used.remove(&old);
        self.resourced += 1;
    }

    /// One side of the fold-coupled swap-with-refresh
    /// (see [`ProdConfig::swap_refresh`]): retire one base-degree slot of
    /// `value` and inject a freshly drawn same-degree replacement, emitting
    /// the two monomials into separate buffers so the caller can place them
    /// in the fold's fragment stream (inject strictly interior, strip at or
    /// after it — the inject-first order is `resource`'s never-momentarily-
    /// bare rule). Returns false, emitting nothing, when the value holds no
    /// slot to retire.
    ///
    /// WHICH slot is retired is where the drain set gets its bandwidth: a slot
    /// naming a draining variable is preferred over the ordinary base-degree
    /// pick, and the degree filter is dropped for it (the tower term holds
    /// references too, and leaving them to `resource` alone is what would make
    /// the drain tail the binding constraint). The replacement is drawn at the
    /// RETIRED slot's degree either way, so the per-value degree multiset --
    /// the mask plan a build commits to -- is preserved exactly.
    ///
    /// The fresh draw takes new band positions, never a polarity re-roll of
    /// the retired slot: the XOR of two polarity variants of one product is
    /// degree <= deg-1 in the band wires, whose values are themselves wire
    /// segments, so a re-roll's delta would sit back inside a linear
    /// adversary's span.
    fn swap_refresh_side(
        &mut self,
        value: usize,
        state: &GadgetState,
        rng: &mut impl Rng,
        inject_buf: &mut Vec<XGate>,
        strip_buf: &mut Vec<XGate>,
    ) -> bool {
        if self.dist {
            // Distributed sourcing draws need the fold's write barrier as a
            // forbidden set; the swap is a banded-production mechanism and
            // stays off rather than emit an unaudited dist draw.
            return false;
        }
        let base = self.plan.first().copied().unwrap_or(2);
        let old_index = match self.drain_pick_slot(value, rng) {
            Some(idx) => {
                self.drain_steered += 1;
                idx
            }
            None => {
                let cands: Vec<usize> = self.slots[value]
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| s.factors.len() == base)
                    .map(|(i, _)| i)
                    .collect();
                if cands.is_empty() {
                    return false;
                }
                cands[rng.random_range(0..cands.len())]
            }
        };
        let deg = self.slots[value][old_index].factors.len();
        let slot = self.draw_slot(value, deg, rng);
        let konst = self.emit_slot(value, &slot, state, rng, inject_buf);
        self.consts[value] ^= konst;
        self.add_refs(value, &slot);
        self.slots[value].push(slot);
        let old = self.slots[value].remove(old_index);
        let konst = self.emit_slot(value, &old, state, rng, strip_buf);
        self.consts[value] ^= konst;
        self.drop_refs(value, &old);
        self.used.remove(&old);
        self.swapped += 1;
        true
    }

    // ---- the drain set -------------------------------------------------
    //
    // A band variable is cheap to refresh exactly when nothing reads it: the
    // wire is overwritten and no carrier has to be patched. Referenced, it
    // costs ~2 emissions per naming slot to release first (`retire_refill`).
    // At production the reference count is the factors-per-value figure --
    // band = value count, so R = sum(plan) ~ 10 -- and an unreferenced
    // variable essentially never occurs by chance (e^-10).
    //
    // So it is arranged instead of waited for. A rolling set D of variables is
    // excluded from every fresh draw, so its counts only fall; retirement
    // steering aims the churn `swap_refresh` and `resource` are performing
    // ANYWAY at slots naming a member; a member that reaches zero is rewritten
    // and swapped out for a fresh variable. The retirements are not extra work
    // -- only their TARGETS changed -- so turnover costs the rewrite alone.
    //
    // Turnovers over a circuit = sides * folds / (values * factors-per-value).
    // The rate is `swap_refresh`; at n=128 two sides buy ~6 turnovers and four
    // buy ~12, which is why the production rate moved to 4.
    //
    // ⚠️ The schedule must not become legible. This is a ROLLING set with
    // random membership and one-at-a-time replacement, not a fixed partition
    // swept in waves: disjoint waves would print "these |D| wires stopped
    // being read together, then were all written", the same signal the
    // shuffled `retire_queue` exists to avoid on the epoch path.

    /// Seed the drain set (lazily, at the first rotation — `new` has no rng).
    fn drain_init(&mut self, rng: &mut impl Rng) {
        while self.drain.len() < self.drain_cap {
            let Some(var) = self.drain_admit(rng) else {
                break;
            };
            self.drain.push(var);
        }
    }

    /// A uniformly drawn variable not already draining, or None if the band
    /// has none to spare.
    fn drain_admit(&self, rng: &mut impl Rng) -> Option<u16> {
        let band_len = self.loc.len();
        if band_len <= self.drain.len() {
            return None;
        }
        for _ in 0..64 {
            let var = rng.random_range(0..band_len as u16);
            if !self.drain.contains(&var) {
                return Some(var);
            }
        }
        (0..band_len as u16).find(|v| !self.drain.contains(v))
    }

    /// Index of a slot of `value` naming a draining variable, if any.
    ///
    /// Among the candidates the CHEAPEST degree wins, because retirement cost
    /// is not flat: a base-degree slot emits one g57, while a tower term is a
    /// 3-control conjunction the ladder re-spells into several gates (measured
    /// at n=128: steering blind to degree cost +7.7% gates, and dropping the
    /// tower terms to last-resort recovers most of it). This never blocks a
    /// tower reference — when a variable's remaining references are all in
    /// tower terms, those are the only candidates and get picked — it just
    /// stops paying tower prices for work a base term could do.
    fn drain_pick_slot(&self, value: usize, rng: &mut impl Rng) -> Option<usize> {
        if self.drain.is_empty() {
            return None;
        }
        let cheapest = self.slots[value]
            .iter()
            .filter(|s| s.factors.iter().any(|&(b, _)| self.drain.contains(&b)))
            .map(|s| s.factors.len())
            .min()?;
        let cands: Vec<usize> = self.slots[value]
            .iter()
            .enumerate()
            .filter(|(_, s)| {
                s.factors.len() == cheapest
                    && s.factors.iter().any(|&(b, _)| self.drain.contains(&b))
            })
            .map(|(i, _)| i)
            .collect();
        Some(cands[rng.random_range(0..cands.len())])
    }

    /// A (value, slot index) pair holding a live reference to a draining
    /// variable, found through the reverse index rather than by scanning every
    /// value. Members are visited in random order so a stalled member does not
    /// monopolize the channel.
    fn drain_pick(&self, rng: &mut impl Rng) -> Option<(usize, usize)> {
        if self.drain.is_empty() {
            return None;
        }
        let start = rng.random_range(0..self.drain.len());
        for offset in 0..self.drain.len() {
            let var = self.drain[(start + offset) % self.drain.len()];
            let holders = &self.var_holders[var as usize];
            if holders.is_empty() {
                continue;
            }
            let value = holders[rng.random_range(0..holders.len())] as usize;
            // Cheapest naming slot, for the reason in `drain_pick_slot`.
            if let Some(idx) = self.slots[value]
                .iter()
                .enumerate()
                .filter(|(_, s)| s.factors.iter().any(|&(b, _)| b == var))
                .min_by_key(|(_, s)| s.factors.len())
                .map(|(i, _)| i)
            {
                return Some((value, idx));
            }
            debug_assert!(false, "var_holders[{var}] names value {value} with no slot");
        }
        None
    }

    /// Refresh every drained member and admit replacements.
    ///
    /// Emitted at a fold boundary, never inside one: the ladder borrows band
    /// wires as scratch mid-chain and restores them at the end of a fragment,
    /// so a rewrite landing between a borrow and its restore would corrupt the
    /// chain. A drained member is unreferenced by construction, so no live
    /// mask and therefore no fold fragment reads it.
    fn drain_rotate(&mut self, state: &GadgetState, rng: &mut impl Rng, out: &mut Vec<XGate>) {
        if self.drain_cap == 0 || !self.enabled() || self.dist || self.loc.is_empty() {
            return;
        }
        if self.drain.is_empty() {
            self.drain_init(rng);
            return;
        }
        // Snapshot the due members first. Replacements are admitted as we go,
        // and a replacement can itself land on an unreferenced variable —
        // iterating the live vector would then be unbounded. Anything admitted
        // here that is already free is simply picked up by the next rotation.
        let due: Vec<u16> = self
            .drain
            .iter()
            .copied()
            .filter(|&v| self.var_refs[v as usize] == 0)
            .collect();
        for var in due {
            self.rewrite_var(var, state, self.refill_data, self.fill_nl, rng, out);
            self.drained += 1;
            // Retire the member BEFORE drawing its replacement, so the draw
            // cannot hand back the variable just refreshed.
            if let Some(p) = self.drain.iter().position(|&v| v == var) {
                self.drain.swap_remove(p);
            }
            if let Some(fresh) = self.drain_admit(rng) {
                self.drain.push(fresh);
            }
        }
    }

    /// Retire one band variable and rewrite it, so its Boolean function stops
    /// being a lifetime signature.
    ///
    /// Order matters and is the whole correctness argument: every live mask
    /// naming the variable is re-sourced FIRST (inject-then-strip, so no value
    /// is ever momentarily bare), which leaves the variable unreferenced; only
    /// then is its wire rewritten. After the rewrite the variable is a
    /// different function, and any mask drawn later that names it means the new
    /// one -- the ledger never holds a stale reading, because it never holds a
    /// reference across the rewrite.
    fn retire_refill(
        &mut self,
        state: &GadgetState,
        refill_data: usize,
        fill_nl: usize,
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) {
        if !self.enabled() || self.loc.is_empty() || self.dist {
            return;
        }
        // Strict round-robin makes the refresh SCHEDULE itself a legible
        // object: each variable's next retirement is predictable, so the
        // epochs it creates are as enumerable as the frozen functions it
        // replaced. Draw from a shuffled queue instead -- same turnover
        // guarantee (every variable once per sweep), no fixed order.
        if self.retire_queue.is_empty() {
            self.retire_queue = (0..self.loc.len() as u16).collect();
            use rand::seq::SliceRandom;
            self.retire_queue.shuffle(rng);
        }
        let var = self.retire_queue.pop().expect("queue refilled above");

        // 1. Release the variable: re-source every live slot naming it.
        loop {
            let found = (0..state.n).find_map(|value| {
                self.slots[value]
                    .iter()
                    .position(|s| s.factors.iter().any(|&(b, _)| b == var))
                    .map(|idx| (value, idx))
            });
            let Some((value, idx)) = found else { break };
            let deg = self.slots[value][idx].factors.len();
            self.inject_avoiding_var(value, deg, var, state, rng, out);
            let old = self.slots[value].remove(idx);
            let konst = self.emit_slot(value, &old, state, rng, out);
            self.consts[value] ^= konst;
            // Release the retired slot's wires. Without this refs[] only ever
            // grows on the epoch path, so wires stay permanently "named" and
            // the distributed draw's legal pool shrinks toward empty.
            self.drop_refs(value, &old);
            self.used.remove(&old);
            self.migrated += 1;
        }

        // 2. Rewrite the wire.
        self.rewrite_var(var, state, refill_data, fill_nl, rng, out);
        self.retired += 1;
    }

    /// Rewrite one UNREFERENCED band variable's value: a band-sourced pivot
    /// CNOT plus `fill_nl` product terms that readmit carriers at
    /// `refill_data` percent. Shared by the epoch path (`retire_refill`, after
    /// it has paid to release the variable) and the drain set (`drain_rotate`,
    /// which waits until the release is free), so the two channels put
    /// statistically identical material into the band and an A/B between them
    /// measures the SCHEDULE, not the algebra.
    ///
    /// Sources are chosen BY ROLE, not by wire number. Rolling exchanges the
    /// carrier and band roles, so a numeric range over `0..carrier_total`
    /// does not mean "a carrier" -- it can land on a band variable, or on
    /// the target wire itself, and then `refill_data` percent is not the
    /// rate it claims to be. Carriers are reached through `state.pairs` and
    /// band variables through `loc`, which are the two role maps.
    /// Refill sourcing (symmetric-ports revision): the channels differ.
    ///
    /// The LINEAR skeleton — the pivot CNOT — is band-sourced ONLY. A
    /// carrier-sourced pivot is a verbatim linear copy of a masked data
    /// state into the band, and the low half is the payload's birthplace
    /// mid-circuit (it holds C(x) from mid-C until the D block junks it);
    /// the strip tail's mid-group windows cancelled the accompanying
    /// masks and read a payload bit back out of exactly such a copy
    /// (measured: exact all-band-reads windows at n=32 and n=128).
    /// Band-sourcing the linear part also keeps it an invertible shear.
    ///
    /// The PRODUCT terms readmit carriers at `refill_data` percent, from
    /// the full value range: a product of two masked carriers is degree-2
    /// in the logical values — nothing a linear adversary can peel — and
    /// carrier products are what keep the band honest three ways at once:
    /// they re-couple its functions to computational progress (band-only
    /// refills leave every future band value inside the algebra generated
    /// by the initial band, with coefficients readable off the gate list),
    /// they inject rank-independent material (band-only mixing can drift
    /// into linear dependence and silently break joint uniformity), and
    /// they make every refill cluster read across the band/carrier partition
    /// (an all-band-reads refill is a transitive band-labeling channel for a
    /// structural adversary).
    ///
    /// The rewrite is a SHEAR — `b ^= delta`, not `b := delta` — so the
    /// variable's new function is its old one plus fresh material. Balance
    /// carries over from the old value for free, and each turnover raises the
    /// degree rather than resetting it.
    fn rewrite_var(
        &mut self,
        var: u16,
        state: &GadgetState,
        refill_data: usize,
        fill_nl: usize,
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) {
        assert_eq!(
            self.var_refs[var as usize], 0,
            "rewrite of band variable {var} while {} live mask term(s) still name it: \
             every naming slot must be released (epoch path) or drained (drain set) first",
            self.var_refs[var as usize]
        );
        let wire = self.loc[var as usize];
        let mut draw = |rng: &mut dyn rand::RngCore, allow_carrier: bool| -> u16 {
            let use_carrier = allow_carrier
                && state.n > 0
                && (rng.next_u32() as usize % 100) < refill_data;
            if use_carrier {
                for _ in 0..64 {
                    let v = rng.next_u32() as usize % state.n;
                    let (c0, c1) = state.pairs[v];
                    let c = if rng.next_u32() & 1 == 0 { c0 } else { c1 } as u16;
                    if c != wire {
                        return c;
                    }
                }
            }
            loop {
                let other = rng.next_u32() as usize % self.loc.len();
                // `loc` is injective, so any other variable's wire differs from
                // the target's by construction.
                if other as u16 != var {
                    return self.loc[other];
                }
            }
        };
        // A pivot-shaped term first (one source, appearing once), then a small
        // nonlinear mix. Balance is only approximate here: mid-body there is no
        // pristine input bit to serve as the fill's fresh pivot, so this is a
        // measured property, not the theorem the port-side fill has.
        let p = draw(rng, false);
        if p != wire {
            out.push(XGate::cnot(wire, p));
        }
        for _ in 0..fill_nl.max(1) {
            let s1 = draw(rng, true);
            let s2 = draw(rng, true);
            if s1 == s2 || s1 == wire || s2 == wire {
                continue;
            }
            let lits = [(s1, rng.random::<bool>()), (s2, rng.random::<bool>())];
            emit_g57_form(wire, &lits, rng, out);
        }
        if refill_data > 0 {
            self.refill_used_carriers = true;
        }
    }

    /// `inject_avoiding_var`: draw a replacement slot that does NOT name the
    /// band variable being retired.
    fn inject_avoiding_var(
        &mut self,
        value: usize,
        deg: usize,
        avoid: u16,
        state: &GadgetState,
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) {
        let band_len = self.loc.len() as u16;
        assert!(
            (band_len as usize) > deg,
            "retire-refill needs a band wider than one slot's degree"
        );
        // Avoid the retired variable AND every variable this value's other live
        // slots already name. Excluding only the retired one is not enough: the
        // piling-up value assumes a value's terms are variable-DISJOINT, and
        // `draw_slot` enforces that on the ordinary path -- so a re-source that
        // ignores it lets an epoch quietly undo the guarantee the ordinary draw
        // maintains, on a build (epoch 5) where migrations are a large fraction
        // of all injections.
        let mut busy: Vec<u16> = vec![avoid];
        for slot in &self.slots[value] {
            for &(b, _) in &slot.factors {
                busy.push(b);
            }
        }
        busy.sort_unstable();
        busy.dedup();
        // Fall back to avoiding only the retired variable if the band is too
        // narrow to honour full disjointness -- correctness never depends on it.
        // `PROD_DISJOINT=0` relaxes here too: the ordinary `draw_slot` path
        // honours that switch, and an A/B that silently kept enforcing it on
        // the epoch path would measure a mixture of the two policies on any
        // build with `--prod-epoch` on.
        let relax = (band_len as usize) < busy.len() + deg
            || std::env::var("PROD_DISJOINT")
                .map(|v| v == "0")
                .unwrap_or(false);
        // Same drain-set exclusion as `draw_slot` — this is a slot draw like
        // any other, and a replacement that named a draining variable would
        // push its count back up and stall the rotation.
        let hold_drain = (band_len as usize).saturating_sub(self.drain.len()) >= deg + 1;
        for _ in 0..100_000 {
            let mut vars: Vec<u16> = Vec::with_capacity(deg);
            while vars.len() < deg {
                let b = rng.random_range(0..band_len);
                if hold_drain && self.drain.contains(&b) {
                    continue;
                }
                let blocked = if relax { b == avoid } else { busy.contains(&b) };
                if !blocked && !vars.contains(&b) {
                    vars.push(b);
                }
            }
            vars.sort_unstable();
            let factors: Vec<(u16, bool)> = vars
                .into_iter()
                .map(|b| (b, rng.random::<bool>()))
                .collect();
            let slot = ProdSlot { factors };
            if !self.used.contains(&slot) {
                self.used.insert(slot.clone());
                let konst = self.emit_slot(value, &slot, state, rng, out);
                self.consts[value] ^= konst;
                // Register the new slot, exactly as the ordinary inject does.
                // Omitting this under-counts a live slot, which is what
                // invariant S is enforced with in the distributed build and
                // what the drain set's zero-test reads in the banded one — a
                // variable could be rewritten while a mask still names it.
                self.add_refs(value, &slot);
                self.slots[value].push(slot);
                self.injected += 1;
                return;
            }
        }
        panic!("prod source band exhausted during retire-refill; raise --prod-band");
    }

    /// RG2', the band roll: relocate one band variable. A uniformly chosen
    /// band variable trades wires with a uniformly chosen other wire — a
    /// carrier of some value (RG2's own move, which swaps carriers between
    /// values, extended across the carrier/band boundary) or another band
    /// wire. The emitted network is the 3-CNOT swap RG2 already uses, so the
    /// move adds no new gate shape.
    ///
    /// Nothing about the encoding changes: the band variable keeps its value
    /// (slots name variables, not wires, and resolve through `loc`), and the
    /// carrier keeps its value (`state.pairs` is re-pointed), so every
    /// value's decode is invariant. What changes is WHERE the frozen band
    /// lives — after a few rolls the band is not a fixed wire range, and no
    /// wire is visibly write-free for the whole body.
    ///
    /// Invariant: carriers and band wires stay disjoint (a swap exchanges the
    /// two roles), so a fold/inject target is never also a mask literal.
    fn roll(&mut self, state: &mut GadgetState, rng: &mut impl Rng, out: &mut Vec<XGate>) {
        if !self.enabled() || self.loc.is_empty() {
            return;
        }
        let total = self.carrier_total + self.loc.len();
        let var = rng.random_range(0..self.loc.len());
        let from = self.loc[var];
        let to = loop {
            let w = rng.random_range(0..total) as u16;
            if w != from {
                break w;
            }
        } as u16;
        // Re-point whichever bookkeeping owns the partner wire.
        if let Some(other) = self.loc.iter().position(|&w| w == to) {
            self.loc[other] = from;
        } else {
            let mut found = false;
            for value in 0..state.n {
                let (s, p) = state.pairs[value];
                if self.single {
                    // One carrier: both entries name the same wire, so both
                    // must follow it across the swap.
                    if s == to as usize {
                        state.pairs[value] = (from as usize, from as usize);
                        found = true;
                        break;
                    }
                    continue;
                }
                if s == to as usize {
                    state.pairs[value].0 = from as usize;
                    found = true;
                    break;
                }
                if p == to as usize {
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
        self.loc[var] = to;
        // Content swap: three transvections, in either of the two orders,
        // each drawn from the mixed vocabulary so the wire that now holds a
        // band variable is not written exclusively by width-1 CNOTs (which
        // would be a gate-shape signature even once the write COUNTS match).
        let (a, b) = if rng.random_bool(0.5) {
            (from, to)
        } else {
            (to, from)
        };
        for (target, source) in [(a, b), (b, a), (a, b)] {
            emit_transvection_mixed(target, source, total, rng, out);
        }
        self.rolled += 1;
    }

    /// The share-native CG: fold `v_t ^= f(controls)` by expanding f's ANF
    /// over the operands' full decodes. Each control value contributes its
    /// summand atoms (carrier literals and k product-literal pairs); every
    /// cross-product of one atom per control is one conjunction fragment
    /// folded into either carrier of the target. Constant terms (and the
    /// gate's own complement) go to the target's ledger constant — no bare X.
    /// No wire is ever written except the target's carriers, and no fragment
    /// ever computes a single value's own two-carrier XOR or its product term
    /// onto a wire: operands stay masked through the gate.
    /// Distributed-sourcing barrier for one fold.
    ///
    /// The fold writes the target's free carrier and reads, per fragment, one
    /// atom of each operand — carrier literals plus mask literals. Two hazards
    /// follow, both removed by re-sourcing the offending slots first:
    ///
    /// * a mask of one operand naming a carrier of the TARGET would be read by
    ///   a fragment that also writes the target's other carrier — one gate
    ///   seeing both carriers of a value, the gate-local completeness the
    ///   share-native fold exists to keep;
    /// * a mask of one operand naming a carrier of ANOTHER OPERAND is the same
    ///   violation across the two read atoms.
    ///
    /// Invariant S already rules out the third case (two different slots
    /// naming the two different carriers of a third value), because only one
    /// carrier per value is ever named.
    fn guard_fold(
        &mut self,
        gate: &XGate,
        state: &GadgetState,
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) {
        if !self.dist {
            return;
        }
        self.sync(state);
        let t = gate.target as usize;
        let mut hot: Vec<usize> = vec![state.pairs[t].0, state.pairs[t].1];
        for &(w, _) in &gate.ctrls {
            let (c0, c1) = state.pairs[w as usize];
            hot.push(c0);
            hot.push(c1);
        }
        for &(w, _) in &gate.ctrls {
            let operand = w as usize;
            loop {
                let Some(idx) = self.slots[operand]
                    .iter()
                    .position(|s| s.wires().any(|x| hot.contains(&x)))
                else {
                    break;
                };
                let deg = self.slots[operand][idx].factors.len();
                self.inject_avoiding(operand, deg, &hot, state, rng, out);
                let old = self.slots[operand].remove(idx);
                let konst = self.emit_slot(operand, &old, state, rng, out);
                self.consts[operand] ^= konst;
                self.drop_refs(operand, &old);
                self.used.remove(&old);
                self.migrated += 1;
            }
        }
    }

    fn fold_cg(
        &mut self,
        gate: &XGate,
        state: &GadgetState,
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) {
        self.guard_fold(gate, state, rng, out);
        let t = gate.target as usize;
        debug_assert!(t < state.n);
        // Fold-coupled swap-with-refresh (see ProdConfig::swap_refresh). While
        // active, every Gray mode below is declined and the expanded fold's
        // atom spelling is used.
        let swapping = self.swap_refresh && !self.dist;
        let mut lists: Vec<Vec<Vec<(u16, bool)>>> = Vec::with_capacity(gate.ctrls.len());
        for &(w, positive) in &gate.ctrls {
            let w = w as usize;
            debug_assert_ne!(w, t);
            let (c0, c1) = state.pairs[w];
            // A control's decode is C + masks + delta, where delta combines
            // the ledger constant and the source-control polarity.  In the
            // expanded (non-Gray) construction, absorb delta into the polarity
            // of one carrier literal: !C is C+1.  This removes the empty atom.
            // Consequently an arity-2 fold has exactly one NONEMPTY atom from
            // each operand in every product fragment; the dirty ladder never
            // has to realize a constant times one operand's mask in isolation.
            //
            // Keep the aggregate Gray representation unchanged.  Besides
            // preserving its established spelling/measurements, its gather
            // residual is currently expressed by toggling an empty simple
            // atom inside fold_cg_gray.
            let delta = self.consts[w] ^ !positive;
            let aggregate = (self.gray_fold || self.micro_gray) && !swapping;
            // Sentinel mode follows the expanded encoding here: absorb the
            // constant into a live linear literal. An empty atom times H would
            // expose the sentinel itself on the target trace.
            let carrier_polarity = if aggregate { true } else { !delta };
            // Single-carrier decode contributes ONE linear atom, not two: the
            // pair collapses to a single wire, and emitting it twice would
            // cancel. Every other atom (the mask terms, the constant) is
            // unchanged, so the fold's fragment count drops from
            // (2 + k)^arity to (1 + k)^arity at equal mask strength.
            let mut atoms: Vec<Vec<(u16, bool)>> = if self.single {
                vec![vec![(c0 as u16, carrier_polarity)]]
            } else {
                vec![vec![(c0 as u16, carrier_polarity)], vec![(c1 as u16, true)]]
            };
            for slot in &self.slots[w] {
                atoms.push(slot.lits(&self.loc));
            }
            if aggregate && delta {
                atoms.push(Vec::new());
            }
            lists.push(atoms);
        }
        if gate.comp {
            self.consts[t] ^= true;
            self.ledger_consts += 1;
        }
        // The swap's emissions, buffered so their placement in the fragment
        // stream is chosen deliberately. The target-side pair writes the
        // target's carrier and reads only the band, so it interleaves freely
        // among the fragments; the control-side pair writes the chosen
        // control's carrier — which every fragment holding that carrier atom
        // READS — so it must follow the whole fold, and the read/write
        // collision keeps that order under any correct reordering.
        let mut swap_t_inject: Vec<XGate> = Vec::new();
        let mut swap_t_strip: Vec<XGate> = Vec::new();
        let mut swap_c: Vec<XGate> = Vec::new();
        if swapping {
            // Side 1 is the target, interleaved interior to the stream.
            self.swap_refresh_side(t, state, rng, &mut swap_t_inject, &mut swap_t_strip);
            // Sides 2.. follow the whole fold. Each prefers a value holding a
            // draining reference and falls back to a random control, so the
            // extra bandwidth goes where the drain set needs it; a side that
            // lands on a value with nothing to retire simply emits nothing.
            // Any value is a legal target here — these emissions write that
            // value's own carrier and read only the band, exactly as `resource`
            // does between folds.
            for _ in 1..self.swap_sides {
                let value = match self.drain_pick(rng) {
                    Some((value, _)) => value,
                    None if gate.ctrls.is_empty() => continue,
                    None => gate.ctrls[rng.random_range(0..gate.ctrls.len())].0 as usize,
                };
                let mut inj = Vec::new();
                let mut strip = Vec::new();
                if self.swap_refresh_side(value, state, rng, &mut inj, &mut strip) {
                    swap_c.extend(inj);
                    swap_c.extend(strip);
                }
            }
        }
        // The Gray fold handles the arity-2 blocks -- which is every source
        // gate in the g57 body -- with no wide fragment at all. It declines the
        // shapes it cannot amortize (arity != 2, an operand with no mask terms,
        // no room to borrow); those fall through, and are laddered below rather
        // than left wide, since a single surviving 3-control gate would undo
        // the point of the exercise. Swap mode declines every Gray mode: the
        // gathers materialize an operand's whole mask sum on one accumulator
        // segment pair, a linear operand recovery no mask shuffle removes.
        if !swapping && self.gray_fold && self.fold_cg_gray(t, &lists, state, rng, out) {
            return;
        }
        if !swapping && self.micro_gray {
            let target = self.free_carrier(t, state, rng);
            let (t0, t1) = state.pairs[t];
            let protected = [t0 as u16, t1 as u16];
            let carrier_groups: Vec<Vec<u16>> = state
                .pairs
                .iter()
                .map(|&(a, b)| {
                    if a == b {
                        vec![a as u16]
                    } else {
                        vec![a as u16, b as u16]
                    }
                })
                .collect();
            if self.fold_micro_product(target, &lists, &protected, &carrier_groups, rng, out) {
                return;
            }
        }
        if !swapping && self.sentinel_gray {
            let target = self.free_carrier(t, state, rng);
            let (t0, t1) = state.pairs[t];
            let protected = [t0 as u16, t1 as u16];
            let carrier_groups: Vec<Vec<u16>> = state
                .pairs
                .iter()
                .map(|&(a, b)| {
                    if a == b {
                        vec![a as u16]
                    } else {
                        vec![a as u16, b as u16]
                    }
                })
                .collect();
            if self.fold_sentinel_product(t, target, &lists, &protected, &carrier_groups, rng, out)
            {
                return;
            }
        }
        let ladder_cap = if (self.gray_fold || self.micro_gray) && !swapping {
            usize::MAX
        } else {
            self.ladder_cap
        };
        if self.cap >= 2 {
            // Narrow mode has no fragment stream to interleave into; the swap
            // pair brackets the block instead (interior pinning is weaker, and
            // narrow mode is not the production path).
            out.extend(swap_t_inject.drain(..));
            self.fold_cg_narrow(t, &lists, state, rng, out);
            out.extend(swap_t_strip.drain(..));
            out.extend(swap_c.drain(..));
            self.drain_rotate(state, rng, out);
            return;
        }
        // Odometer over the cartesian product (an empty `lists` — an X/NOT
        // source — contributes exactly the single constant-1 term). Fragments
        // are MATERIALIZED first and emitted in a shuffled order: the odometer
        // order is a static per-gate progress clock (consecutive fragments
        // share atom prefixes), readable with no execution at all.
        //
        // A fragment is kept as its LIST OF ATOMS, not as a flat literal list.
        // Laddering a fragment parks partial products on borrowed wires, and a
        // borrowed wire must never hold one value's whole mask term — so the
        // ladder path needs the atom boundaries to interleave across, which a
        // flattened list has already thrown away.
        let mut frags: Vec<Vec<Vec<(u16, bool)>>> = Vec::new();
        let mut combo = vec![0usize; lists.len()];
        'odometer: loop {
            let picked: Vec<Vec<(u16, bool)>> = lists
                .iter()
                .zip(&combo)
                .map(|(list, &pick)| list[pick].clone())
                .collect();
            if picked.iter().all(|a| a.is_empty()) {
                self.consts[t] ^= true;
                self.ledger_consts += 1;
            } else if atoms_contradict(&picked) {
                // The product contains w AND !w across its atoms: the term is
                // identically 0 and emits nothing. Dropping it HERE (rather
                // than at emission) keeps the swap's interior placement over
                // the actually-emitted fragment stream exact.
            } else {
                frags.push(picked);
            }
            let mut axis = 0;
            loop {
                if axis == combo.len() {
                    break 'odometer;
                }
                combo[axis] += 1;
                if combo[axis] < lists[axis].len() {
                    break;
                }
                combo[axis] = 0;
                axis += 1;
            }
        }
        let sibling = self.sibling_map(state);
        // All fragments XOR into the same value's two carriers and read only
        // other values' carriers and the band, so they commute freely.
        //
        use rand::seq::SliceRandom;
        frags.shuffle(rng);
        // Interior interleave for the target-side swap: the inject goes
        // STRICTLY interior (at least one fragment on each side), so every
        // contiguous window of the target carrier's writes that covers the
        // whole fold — the only subset whose XOR is the clean operand decode —
        // necessarily picks up a fresh non-cancelling monomial; the strip may
        // land anywhere at or after it. Under the target-stable shuffle the
        // per-wire write order is the emission order, so the pinning is exact
        // rather than probabilistic.
        let (inj_pos, strip_pos) = if frags.len() >= 2 {
            let i = rng.random_range(1..frags.len());
            (i, rng.random_range(i..=frags.len()))
        } else {
            (frags.len(), frags.len())
        };
        for (fi, atoms) in frags.iter().enumerate() {
            if fi == inj_pos {
                out.extend(swap_t_inject.drain(..));
            }
            if fi == strip_pos {
                out.extend(swap_t_strip.drain(..));
            }
            let target = self.free_carrier(t, state, rng);
            // Interleaving is only REQUIRED on the ladder path, but a fragment's
            // width is not known until its literals are normalized, and using a
            // different literal ORDER for laddered and unladdered fragments
            // would make the two populations distinguishable by control order
            // alone. Interleave uniformly; a conjunction does not care.
            let mut lits = interleave_atoms(atoms);
            if normalize_lits(&mut lits).is_none() {
                // Contradictory literals (w AND !w): the term is 0.
                continue;
            }
            self.debug_check_fragment(target, &lits, state);
            let width = lits.len();
            if width <= 2 {
                self.cg_narrow += 1;
                // Realize the DB-eligible fragments in the g57/CNOT
                // vocabulary rather than as exact conjunctions. MEASURED: at
                // n=128 band 256 this raises the frozen store's match rate
                // from 0.3506 to 0.4151 for +4.5% gates (pure sampler,
                // --db-dry-run --p-db 1.0, every other weight 0).
                //
                // ⚠ The MECHANISM is not established. An earlier version of
                // this comment argued that a comp=0 width-2 conjunction `xy`
                // is outside the X-free g57 span over {h,x,y} -- which is
                // <x, y, 1^xy>, i.e. exactly the f with const(f) = coeff_xy(f)
                // -- and concluded such a gate is invisible to a g57-built
                // store. The span identity is real, and it is why three of the
                // four polarity patterns below owe a ledger constant while
                // `~x~y` owes none. The CONCLUSION was wrong: the span only
                // describes circuits that never write x or y, and g57+CNOT
                // over three wires generates all of S8 -- `h ^= xy` is
                // reachable in 5 gates and even a bare NOT in 5. fmix's
                // replacement windows are [2,5] gates besides, so a lone gate
                // is never matched by itself. Keep the lever for the number,
                // not for the story.
                //
                // Costs 1-2 gates instead of 1; the residual goes to the
                // ledger exactly as the narrow path already does.
                if self.g57_narrow {
                    let konst = emit_g57_form(target, &lits, rng, out);
                    self.consts[t] ^= konst;
                    self.cg_fragments += 1;
                    continue;
                }
            } else if width <= ladder_cap {
                // SELECTIVE LADDERING. Full narrow mode ladders every fragment
                // and costs roughly 15x the fold; the fold's width profile is
                // heavily bottom-weighted, so a ceiling buys most of the fossil
                // reduction for a small fraction of that. Fragments wider than
                // the ceiling stay as single wide gates: laddering them is what
                // the 15x is made of, and they are the minority.
                let mut forbidden: Vec<u16> = vec![sibling[target as usize % sibling.len()]];
                forbidden.extend(
                    lits.iter()
                        .map(|&(w, _)| w)
                        .filter(|&w| (w as usize) < sibling.len())
                        .map(|w| sibling[w as usize]),
                );
                let konst = emit_narrow_fragment(
                    target,
                    &lits,
                    2,
                    &self.ladder_borrow_pool(state),
                    self.borrow_total(),
                    &forbidden,
                    atoms,
                    self.rung_menu,
                    rng,
                    out,
                );
                self.consts[t] ^= konst;
                self.cg_fragments += 1;
                self.cg_laddered += 1;
                continue;
            }
            if let Some(fragment) = XGate::conj(target, lits) {
                if fragment.ctrls.len() > 2 {
                    self.cg_fossils += 1;
                }
                out.push(fragment);
                self.cg_fragments += 1;
            }
        }
        // Whatever was not placed inside the stream (strip at the end
        // position, or a degenerate fold with fewer than two fragments), plus
        // the control-side pair, which must follow every fragment that reads
        // the swapped control's carrier.
        out.extend(swap_t_inject.drain(..));
        out.extend(swap_t_strip.drain(..));
        out.extend(swap_c.drain(..));
        // At the fold boundary, with no ladder chain open and every fragment
        // placed. The Gray early-returns above cannot reach here, but they are
        // all guarded by `!swapping` and the drain set requires swap mode.
        self.drain_rotate(state, rng, out);
    }

    /// Two dirty accumulator wires for the Gray fold, drawn by ROLE from the
    /// carriers and excluding everything the block reads or writes. `None` when
    /// the pool is too small, which makes the caller fall back to the odometer.
    fn pick_accumulators(
        &self,
        forbidden: &[u16],
        state: &GadgetState,
        rng: &mut impl Rng,
    ) -> Option<(u16, u16)> {
        let sibling = self.sibling_map(state);
        let free: Vec<u16> = self
            .carrier_wires(state)
            .into_iter()
            .filter(|w| !forbidden.contains(w))
            .collect();
        use rand::seq::IndexedRandom;
        let u = *free.choose(rng)?;
        // The second accumulator must not be the first's sibling: the four
        // A_u * A_z gates read BOTH, and one gate seeing both carriers of one
        // value is exactly the gate-local completeness the construction bans.
        let free2: Vec<u16> = free
            .into_iter()
            .filter(|&w| w != u && w != sibling[u as usize])
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
    /// The wide fold expands `PROD_w (carrier_w + masks_w)` into one gate per
    /// term of the cartesian product, so a `[1,2,3,3]` share gives 16 fragments
    /// of width up to `arity * max_deg` = 6. Everything above width 2 is
    /// invisible to the frozen store (width-3 gates hit at 0.41% against ~99%
    /// for narrow material) and is 56% of the shipped gadget. Laddering each
    /// wide fragment individually is full narrow mode's ~6.2x, because every
    /// fragment re-derives the same mask products from scratch.
    ///
    /// This gathers each operand's mask sum ONCE onto a borrowed wire and reads
    /// it back four times. Write `S_w = L_w + M_w` for operand `w`, with `L_w`
    /// the width-<=1 atoms (the carrier literal and the ledger's constant atom)
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
    /// adversary nothing the endpoints do not. No secret enters the span of the
    /// wires and their pairwise products at any prefix, and a quadratic mirror
    /// search peaks at the (3/4)^2 design bound. The same audit run against a
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
    /// Gather one micro share exactly onto an arbitrary-dirty accumulator.
    /// `plans` is reused for the matching strip so every spelling residual
    /// follows the same convention on both passes.
    fn gather_micro_share_exact(
        &self,
        acc: u16,
        atoms: &[Vec<(u16, bool)>],
        plans: &[MicroAtomPlan],
        constant_helper: u16,
        seen: &mut SpellingLog,
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) {
        debug_assert_eq!(atoms.len(), plans.len());
        let mut correction = false;
        for (atom, &plan) in atoms.iter().zip(plans) {
            if atom.is_empty() {
                correction ^= true;
            } else {
                correction ^= emit_micro_atom_onto(acc, atom, plan, self.rung_menu, seen, rng, out);
            }
        }
        if correction {
            // (!h) + h = 1 without a clean constant wire.
            out.push(
                XGate::conj(acc, [(constant_helper, false)])
                    .expect("micro constant helper differs from accumulator"),
            );
            out.push(XGate::cnot(acc, constant_helper));
        }
    }

    /// Emit one product of two nonempty formal atoms. Literal order starts
    /// with one factor from each atom, so every width-three/four ladder has a
    /// cross-factor rung zero. This is load-bearing for `blind * H`: starting
    /// with two H literals would make the whole cubic H appear as a helper
    /// before/after delta one rung later.
    fn emit_sentinel_atom_product(
        &mut self,
        ledger_target: usize,
        target: u16,
        left: &[(u16, bool)],
        right: &[(u16, bool)],
        helper0: u16,
        helper1: u16,
        out: &mut Vec<XGate>,
    ) {
        debug_assert!(!left.is_empty() && !right.is_empty());
        let mut lits = Vec::with_capacity(left.len() + right.len());
        lits.push(left[0]);
        lits.push(right[0]);
        lits.extend_from_slice(&left[1..]);
        lits.extend_from_slice(&right[1..]);
        if normalize_lits(&mut lits).is_none() {
            return;
        }
        if lits.is_empty() {
            self.consts[ledger_target] ^= true;
            self.ledger_consts += 1;
            return;
        }
        match lits.len() {
            1 | 2 => {
                out.push(XGate::conj(target, lits).expect("sentinel term excludes target"));
                self.cg_narrow += 1;
            }
            3 | 4 => {
                emit_exact_dirty_cap2(target, &lits, helper0, helper1, out);
                self.cg_laddered += 1;
            }
            _ => {
                out.push(XGate::conj(target, lits).expect("sentinel term excludes target"));
                self.cg_fossils += 1;
            }
        }
        self.cg_fragments += 1;
    }

    /// Exact Q toggle. Sentinel mode deliberately allows Q to be trace-affine,
    /// but never permits a maximum-degree H atom onto this accumulator. The
    /// raw dirty ladder has no spelling residual, so there is no hidden kappa
    /// that would also have to be propagated through every H tail.
    fn toggle_sentinel_q(
        &self,
        acc: u16,
        atoms: &[Vec<(u16, bool)>],
        helper0: u16,
        helper1: u16,
        out: &mut Vec<XGate>,
    ) {
        for atom in atoms {
            debug_assert!((2..=4).contains(&atom.len()));
            if atom.len() <= 2 {
                out.push(XGate::conj(acc, atom.iter().copied()).expect("Q excludes accumulator"));
            } else {
                emit_exact_dirty_cap2(acc, atom, helper0, helper1, out);
            }
        }
    }

    /// Emit the audited four-phase sentinel identity for fixed dirty borrows.
    /// With S=L+Q+H and T=R+Q'+K, the phase brackets contribute QR, QK, LQ',
    /// HQ', and QQ'; the direct tails contribute LR, HR, LK, HK. Incoming
    /// accumulator/helper junk cancels and all four borrowed wires restore.
    fn emit_sentinel_schedule(
        &mut self,
        ledger_target: usize,
        target: u16,
        parts: &[SentinelParts; 2],
        borrowed: [u16; 4],
        out: &mut Vec<XGate>,
    ) {
        let [u, z, helper0, helper1] = borrowed;
        let u_atom = [(u, true)];
        let z_atom = [(z, true)];
        let (left, right) = (&parts[0], &parts[1]);

        // Direct tails: LR + HR + LK + HK.
        for l in &left.low {
            for r in &right.low {
                self.emit_sentinel_atom_product(ledger_target, target, l, r, helper0, helper1, out);
            }
        }
        for h in &left.high {
            for r in &right.low {
                self.emit_sentinel_atom_product(ledger_target, target, h, r, helper0, helper1, out);
            }
        }
        for l in &left.low {
            for h in &right.high {
                self.emit_sentinel_atom_product(ledger_target, target, l, h, helper0, helper1, out);
            }
        }
        for h0 in &left.high {
            for h1 in &right.high {
                self.emit_sentinel_atom_product(
                    ledger_target,
                    target,
                    h0,
                    h1,
                    helper0,
                    helper1,
                    out,
                );
            }
        }

        // A: uR + uK + uz.
        for r in &right.low {
            self.emit_sentinel_atom_product(
                ledger_target,
                target,
                &u_atom,
                r,
                helper0,
                helper1,
                out,
            );
        }
        for h in &right.high {
            self.emit_sentinel_atom_product(
                ledger_target,
                target,
                &u_atom,
                h,
                helper0,
                helper1,
                out,
            );
        }
        self.emit_sentinel_atom_product(
            ledger_target,
            target,
            &u_atom,
            &z_atom,
            helper0,
            helper1,
            out,
        );
        self.toggle_sentinel_q(u, &left.gathered, helper0, helper1, out);

        // B: uR + uK + Lz + Hz + uz.
        for r in &right.low {
            self.emit_sentinel_atom_product(
                ledger_target,
                target,
                &u_atom,
                r,
                helper0,
                helper1,
                out,
            );
        }
        for h in &right.high {
            self.emit_sentinel_atom_product(
                ledger_target,
                target,
                &u_atom,
                h,
                helper0,
                helper1,
                out,
            );
        }
        for l in &left.low {
            self.emit_sentinel_atom_product(
                ledger_target,
                target,
                l,
                &z_atom,
                helper0,
                helper1,
                out,
            );
        }
        for h in &left.high {
            self.emit_sentinel_atom_product(
                ledger_target,
                target,
                h,
                &z_atom,
                helper0,
                helper1,
                out,
            );
        }
        self.emit_sentinel_atom_product(
            ledger_target,
            target,
            &u_atom,
            &z_atom,
            helper0,
            helper1,
            out,
        );
        self.toggle_sentinel_q(z, &right.gathered, helper0, helper1, out);

        // C: Lz + Hz + uz.
        for l in &left.low {
            self.emit_sentinel_atom_product(
                ledger_target,
                target,
                l,
                &z_atom,
                helper0,
                helper1,
                out,
            );
        }
        for h in &left.high {
            self.emit_sentinel_atom_product(
                ledger_target,
                target,
                h,
                &z_atom,
                helper0,
                helper1,
                out,
            );
        }
        self.emit_sentinel_atom_product(
            ledger_target,
            target,
            &u_atom,
            &z_atom,
            helper0,
            helper1,
            out,
        );
        self.toggle_sentinel_q(u, &left.gathered, helper0, helper1, out);

        // D: uz, then restore z.
        self.emit_sentinel_atom_product(
            ledger_target,
            target,
            &u_atom,
            &z_atom,
            helper0,
            helper1,
            out,
        );
        self.toggle_sentinel_q(z, &right.gathered, helper0, helper1, out);
    }

    fn fold_sentinel_product(
        &mut self,
        ledger_target: usize,
        target: u16,
        lists: &[Vec<Vec<(u16, bool)>>],
        protected: &[u16],
        carrier_groups: &[Vec<u16>],
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) -> bool {
        if lists.len() != 2 {
            return false;
        }
        let Some(left) = partition_max_degree_sentinel(&lists[0]) else {
            return false;
        };
        let Some(right) = partition_max_degree_sentinel(&lists[1]) else {
            return false;
        };

        let mut forbidden = protected.to_vec();
        forbidden.push(target);
        forbidden.extend(lists.iter().flatten().flatten().map(|&(wire, _)| wire));
        forbidden.sort_unstable();
        forbidden.dedup();
        let mut pool: Vec<&Vec<u16>> = carrier_groups
            .iter()
            .filter(|group| !group.is_empty() && group.iter().all(|wire| !forbidden.contains(wire)))
            .collect();
        pool.shuffle(rng);
        if pool.len() < 4 {
            return false;
        }
        let borrowed: [u16; 4] = std::array::from_fn(|index| {
            let group = pool[index];
            group[rng.random_range(0..group.len())]
        });
        self.emit_sentinel_schedule(ledger_target, target, &[left, right], borrowed, out);
        self.cg_gray += 1;
        self.cg_sentinel += 1;
        true
    }

    /// Four-share micro-Gray product. Every operand is partitioned into four
    /// formal ANF rows, and every one of the sixteen row rectangles is emitted
    /// as an independently restored dirty-accumulator inclusion/exclusion:
    ///
    ///   P; u^=q; P; z^=r; P; u^=q; P; z^=r
    ///
    /// where P toggles the target by `u*z`. Thus a rectangle contributes
    /// exactly `q*r`, while `u`, `z`, and all ladder helpers return to their
    /// arbitrary incoming values. No transition gathers the complete operand.
    fn fold_micro_product(
        &mut self,
        target: u16,
        lists: &[Vec<Vec<(u16, bool)>>],
        protected: &[u16],
        carrier_groups: &[Vec<u16>],
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) -> bool {
        if lists.len() != 2
            || lists.iter().any(|list| list.len() < 4)
            || lists.iter().flatten().any(|atom| atom.len() > 4)
        {
            return false;
        }
        let Some(left) = micro_partition_atoms(&lists[0], rng) else {
            return false;
        };
        let Some(right) = micro_partition_atoms(&lists[1], rng) else {
            return false;
        };
        let shares = [left, right];
        let quartic = lists.iter().flatten().any(|atom| atom.len() == 4);

        // Borrow by logical role, never merely by numeric wire range. A whole
        // group is rejected when any lane is read or protected, so each
        // accumulator/helper comes from a different unrelated representation.
        let mut forbidden = protected.to_vec();
        forbidden.push(target);
        for &(wire, _) in lists.iter().flatten().flatten() {
            forbidden.push(wire);
        }
        forbidden.sort_unstable();
        forbidden.dedup();
        let needed = if quartic { 4 } else { 3 };
        let mut pool: Vec<&Vec<u16>> = carrier_groups
            .iter()
            .filter(|group| !group.is_empty() && group.iter().all(|wire| !forbidden.contains(wire)))
            .collect();
        pool.shuffle(rng);
        if pool.len() < needed {
            return false;
        }
        let borrowed: Vec<u16> = pool
            .into_iter()
            .take(needed)
            .map(|group| group[rng.random_range(0..group.len())])
            .collect();
        let (u, z, helper0) = (borrowed[0], borrowed[1], borrowed[2]);
        let helper1 = if quartic { borrowed[3] } else { helper0 };

        let plans: Vec<Vec<Vec<MicroAtomPlan>>> = shares
            .iter()
            .map(|side| {
                side.iter()
                    .map(|share| {
                        share
                            .iter()
                            .map(|atom| MicroAtomPlan {
                                helper0,
                                helper1,
                                pivot: if atom.len() == 3 {
                                    choose_pivot(atom, rng)
                                } else {
                                    0
                                },
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let mut rectangles: Vec<(usize, usize)> = (0..4)
            .flat_map(|left| (0..4).map(move |right| (left, right)))
            .collect();
        rectangles.shuffle(rng);
        let mut seen: SpellingLog = std::collections::HashMap::new();
        let product_lits = [(u, true), (z, true)];
        for (li, ri) in rectangles {
            // Every spelling of u*z has the same residual constant. There are
            // four reads per rectangle, so that residual cancels exactly.
            let emit_product = |seen: &mut SpellingLog, rng: &mut _, out: &mut Vec<XGate>| {
                let (spellings, _) = spellings_at(target, &product_lits, self.rung_menu);
                let pick = pick_spelling(target, &product_lits, spellings.len(), seen, rng);
                out.extend(spellings[pick].iter().cloned());
            };
            emit_product(&mut seen, rng, out);
            self.gather_micro_share_exact(
                u,
                &shares[0][li],
                &plans[0][li],
                helper0,
                &mut seen,
                rng,
                out,
            );
            emit_product(&mut seen, rng, out);
            self.gather_micro_share_exact(
                z,
                &shares[1][ri],
                &plans[1][ri],
                helper0,
                &mut seen,
                rng,
                out,
            );
            emit_product(&mut seen, rng, out);
            self.gather_micro_share_exact(
                u,
                &shares[0][li],
                &plans[0][li],
                helper0,
                &mut seen,
                rng,
                out,
            );
            emit_product(&mut seen, rng, out);
            self.gather_micro_share_exact(
                z,
                &shares[1][ri],
                &plans[1][ri],
                helper0,
                &mut seen,
                rng,
                out,
            );
        }
        self.cg_fragments += 64;
        self.cg_narrow += 64;
        self.cg_gray += 1;
        true
    }

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
        if masks.iter().flatten().any(|a| a.len() > 3) {
            // A degree-4+ term needs a helper chain rather than one sandwich;
            // the production plan is [1,2,3,3] and the generalization is not
            // verified, so decline rather than emit something unaudited.
            return false;
        }

        // Borrow nothing the block reads or writes, nor any sibling of those:
        // a borrow that is the sibling of a literal would put both carriers of
        // one value into one gate across a read and a write.
        let sibling = self.sibling_map(state);
        let mut forbid: Vec<u16> = vec![state.pairs[t].0 as u16, state.pairs[t].1 as u16];
        for side in 0..2 {
            for atom in simple[side].iter().chain(&masks[side]) {
                for &(w, _) in atom {
                    forbid.push(w);
                    if (w as usize) < sibling.len() {
                        forbid.push(sibling[w as usize]);
                    }
                }
            }
        }
        for w in [state.pairs[t].0, state.pairs[t].1] {
            if w < sibling.len() {
                forbid.push(sibling[w]);
            }
        }
        forbid.sort_unstable();
        forbid.dedup();
        let Some((u, z)) = self.pick_accumulators(&forbid, state, rng) else {
            return false;
        };
        // The sandwich helper must avoid the OTHER accumulator: it is restored
        // either way, but a gate reading `z` while writing `u` would mix the two
        // operands' mask material on one wire for no reason.
        let mut helper_forbid = forbid.clone();
        helper_forbid.push(u);
        helper_forbid.push(z);
        helper_forbid.sort_unstable();
        helper_forbid.dedup();
        let helper_pool: Vec<u16> = self
            .carrier_wires(state)
            .into_iter()
            .filter(|w| !helper_forbid.contains(w))
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
        let mut tog: Vec<Vec<XGate>> = Vec::with_capacity(4);
        let mut konst = [false; 2];
        // One spelling log for the whole block: a function emitted in the
        // gather and again in the strip must not be spelled the same way twice.
        let mut seen: SpellingLog = std::collections::HashMap::new();
        for (i, (acc, side)) in [(u, 0usize), (z, 1usize), (u, 0), (z, 1)]
            .into_iter()
            .enumerate()
        {
            let mut buf = Vec::new();
            let k = self.gather_atoms(acc, &masks[side], &plans[side], &mut seen, rng, &mut buf);
            if i < 2 {
                konst[side] = k;
            } else {
                assert_eq!(
                    k, konst[side],
                    "gray fold: strip parity differs from its gather, so the \
                     accumulator would not be restored"
                );
            }
            tog.push(buf);
        }
        // Absorb the gathers' residual into the operand's constant atom.
        for side in 0..2 {
            if konst[side] {
                match simple[side].iter().position(|a| a.is_empty()) {
                    Some(i) => {
                        simple[side].remove(i);
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

        use rand::seq::SliceRandom;
        // The Gray structure emits the SAME literal list more than once by
        // construction -- `A_u * A_z` at all four phases, each `L x A` pair
        // twice -- so leaving each emission to its own coin plants exact gate
        // groups that `sort | uniq -c` finds without executing anything. Track
        // the last spelling per literal list and pick a different one; the
        // product fragments are mostly all-positive (carrier and accumulator
        // literals), which is precisely the case with four equal-size
        // spellings, so this costs nothing.
        for (i, &p) in [PA, PB, PC, PD].iter().enumerate() {
            let mut frags = std::mem::take(&mut phases[p]);
            // Within a phase the fragments commute (they all XOR into the
            // target's carriers and read nothing the block writes), so the
            // order is free -- and a fixed one would be a per-block clock in
            // the same way the odometer was.
            frags.shuffle(rng);
            for mut lits in frags {
                if normalize_lits(&mut lits).is_none() {
                    continue; // contradictory literals: the term is 0
                }
                if lits.is_empty() {
                    self.consts[t] ^= true;
                    self.ledger_consts += 1;
                    continue;
                }
                let target = self.free_carrier(t, state, rng);
                self.debug_check_fragment(target, &lits, state);
                let (spellings, k) = spellings_at(target, &lits, self.rung_menu);
                let pick = pick_spelling(target, &lits, spellings.len(), &mut seen, rng);
                out.extend(spellings[pick].iter().cloned());
                self.consts[t] ^= k;
                self.cg_fragments += 1;
                self.cg_narrow += 1;
            }
            out.extend(tog[i].iter().cloned());
        }
        self.cg_gray += 1;
        true
    }

    /// The ladder's scratch pool. Under swap mode this is the LIVE BAND
    /// VARIABLES (via `loc`), not the carriers: a ladder rung toggles its
    /// borrowed wire mid-chain, and the chain's target writes read it there,
    /// so a live-carrier borrow exposes data states in the chain's segment
    /// deltas (measured: 9/84 linear flip_match matches at n=32 under
    /// `--prod-ladder-cap 4` with carrier borrows). A band variable's value
    /// is fill junk by construction, wherever rolling has parked it, and the
    /// borrow/restore pair leaves every straddling mask-emission pair intact
    /// (read/write collisions keep other readers outside the chain).
    fn ladder_borrow_pool(&self, state: &GadgetState) -> Vec<u16> {
        if self.swap_refresh && !self.loc.is_empty() {
            self.loc.clone()
        } else {
            self.carrier_wires(state)
        }
    }

    /// The wires CURRENTLY holding carriers, resolved by role rather than by
    /// index. Rolling exchanges a band variable with an arbitrary wire, so the
    /// home index range stops describing the carrier set as soon as
    /// `--prod-roll` is on, and anything that borrows by index leaves a static
    /// index-shaped trace that rolling cannot average out.
    fn carrier_wires(&self, state: &GadgetState) -> Vec<u16> {
        let mut ws: Vec<u16> = Vec::with_capacity(2 * state.n);
        for v in 0..state.n {
            let (a, b) = state.pairs[v];
            ws.push(a as u16);
            if b != a {
                ws.push(b as u16);
            }
        }
        ws.sort_unstable();
        ws.dedup();
        ws
    }

    /// wire -> the other carrier of the same value (identity on the band).
    /// Built over the WHOLE wire space: a roll can put a value's carrier on a
    /// former band wire, so indexing by `carrier_total` alone goes out of
    /// bounds the moment `--prod-roll` is on.
    fn sibling_map(&self, state: &GadgetState) -> Vec<u16> {
        let mut sibling: Vec<u16> = (0..self.borrow_total() as u16).collect();
        for v in 0..state.n {
            let (a, b) = state.pairs[v];
            sibling[a] = b as u16;
            sibling[b] = a as u16;
        }
        sibling
    }

    /// Narrow-mode fold body: materialize the fragment list (one atom pick
    /// per control), shuffle it, then realize each fragment as a g57/CNOT
    /// ladder. Literals are interleaved round-robin across the controls'
    /// atoms so no ladder-chain prefix ever equals a single value's whole
    /// mask term — operands stay masked through the gate even against an
    /// adversary reading the scratch wires.
    fn fold_cg_narrow(
        &mut self,
        t: usize,
        lists: &[Vec<Vec<(u16, bool)>>],
        state: &GadgetState,
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) {
        // Keep the selected atoms beside the flattened conjunction.  The
        // narrowing primitive uses those boundaries when choosing its first
        // dirty rung; dropping them here silently disabled the guard that
        // prevents a borrowed wire from holding a whole operand atom.
        let mut frags: Vec<(Vec<(u16, bool)>, Vec<Vec<(u16, bool)>>)> = Vec::new();
        let mut combo = vec![0usize; lists.len()];
        'odometer: loop {
            let picked: Vec<&Vec<(u16, bool)>> = lists
                .iter()
                .zip(&combo)
                .map(|(list, &pick)| &list[pick])
                .collect();
            let picked: Vec<Vec<(u16, bool)>> = picked.into_iter().cloned().collect();
            let mut lits = interleave_atoms(&picked);
            if lits.is_empty() {
                self.consts[t] ^= true;
                self.ledger_consts += 1;
            } else if normalize_lits(&mut lits).is_some() {
                frags.push((lits, picked));
            }
            let mut axis = 0;
            loop {
                if axis == combo.len() {
                    break 'odometer;
                }
                combo[axis] += 1;
                if combo[axis] < lists[axis].len() {
                    break;
                }
                combo[axis] = 0;
                axis += 1;
            }
        }
        use rand::seq::SliceRandom;
        frags.shuffle(rng);
        // Borrowing the sibling of any wire this fragment READS would let one
        // gate see both carriers of a value; borrowing the target's sibling
        // would do the same across a read and a write. Build the wire->sibling
        // map once, then forbid only what each fragment actually needs -- a
        // blanket carrier exclusion would push every borrow onto band wires,
        // which is both needlessly tight and a signature of its own.
        // Over the WHOLE wire space: a roll can put a value's carrier on a
        // former band wire, so indexing by carrier_total alone goes out of
        // bounds the moment --prod-roll is on.
        let sibling = self.sibling_map(state);
        for (lits, atoms) in &frags {
            let target = self.free_carrier(t, state, rng);
            self.debug_check_fragment(target, lits, state);
            let mut forbidden: Vec<u16> = vec![sibling[target as usize % sibling.len()]];
            forbidden.extend(
                lits.iter()
                    .map(|&(w, _)| w)
                    .filter(|&w| (w as usize) < sibling.len())
                    .map(|w| sibling[w as usize]),
            );
            let konst = emit_narrow_fragment(
                target,
                lits,
                self.cap,
                &self.ladder_borrow_pool(state),
                self.borrow_total(),
                &forbidden,
                atoms,
                self.rung_menu,
                rng,
                out,
            );
            self.consts[t] ^= konst;
            self.cg_fragments += 1;
        }
    }

    /// Unshare: strip every slot and emit every pending ledger constant (as a
    /// !u/u fragment pair — no bare X), restoring the plain pair-XOR decode
    /// for the standard bookend.
    fn strip_all(&mut self, state: &GadgetState, rng: &mut impl Rng, out: &mut Vec<XGate>) {
        self.strip_from(0, state, rng, out);
    }

    /// `strip_all` for values `lo..n` only. Values below `lo` keep their mask
    /// terms and pending constants FOREVER — their carriers are never bared.
    /// Used by the closing-slice design for the sandwich's forward-junk half:
    /// stripping a junk value creates a stretch of bare junk segments at the
    /// output port whose local pair-XORs equal the source circuit's own
    /// wire-segment XORs (measured: the S2 slice gates matched at ~31% even
    /// with the per-gate swap on, all in the last 10% of the circuit), and
    /// nothing downstream ever needs to decode a junk value. The payload half
    /// must still be stripped — the circuit's contract is to output C(x) bare.
    fn strip_from(
        &mut self,
        lo: usize,
        state: &GadgetState,
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) {
        // The constant discharge below reads a helper wire twice, and its two
        // gates' write deltas are that helper's VALUE (and complement). The
        // strip bares the data wires progressively, and a BARE helper hands
        // a payload bit out as a local segment delta — measured as a
        // surviving flip_match class — so under the closing-slice design
        // (tail_band_helpers; strip runs after route-home) the helper pool
        // is the band space only, which is never bared.
        for value in lo..state.n {
            // Strip in plan order (base terms first): the highest-degree
            // tower term covers the value longest at the tail boundary.
            while !self.slots[value].is_empty() {
                let slot = self.slots[value].remove(0);
                let konst = self.emit_slot(value, &slot, state, rng, out);
                self.consts[value] ^= konst;
                self.drop_refs(value, &slot);
                self.used.remove(&slot);
            }
            if self.consts[value] {
                let target = self.free_carrier(value, state, rng) as usize;
                // The helper must not be the target value's OTHER carrier: this
                // gate reads u and writes target, so a sibling here would put
                // both carriers of one value into a single gate -- the
                // gate-local non-completeness violation the whole construction
                // is built to avoid. Drawing from the index range 0..carrier_total
                // and excluding only the target does not prevent it; that is the
                // role-versus-index confusion already fixed twice in this file,
                // and it is live on any two-carrier build (--prod-single 0).
                let (s0, s1) = state.pairs[value];
                let sibling = if s0 == target { s1 } else { s0 };
                let u = if self.tail_band_helpers && !self.loc.is_empty() {
                    // Live band VARIABLES only, resolved through `loc`: the
                    // raw band WIRE range also holds displaced carrier
                    // states after rolls and route-home, and a helper read
                    // from one of those re-injects an old data state as a
                    // tail segment delta (measured: three exact payload
                    // pairs at n=128 whose windows closed through a stale
                    // helper). A band variable's value is fill junk by
                    // construction, wherever rolling has parked it.
                    loop {
                        let w =
                            self.loc[rng.random_range(0..self.loc.len())] as usize;
                        if w == target || w == sibling {
                            continue;
                        }
                        break w as u16;
                    }
                } else {
                    random_wire_except(self.borrow_total(), &[target, sibling], rng) as u16
                };
                out.push(XGate::conj(target as u16, [(u, false)]).expect("distinct wires"));
                out.push(XGate::cnot(target as u16, u));
                self.consts[value] = false;
            }
        }
    }

    fn report(&self) {
        if self.enabled() {
            println!(
                "[prod] plan={:?} band={} src={} injected={} resourced={} swapped={} rolled={} migrated={} retired={} \
                 drained={} turnovers={:.2} steered={} \
                 degen_rejects={} cg_fragments={} cg_narrow={} laddered={} gray_blocks={} fossils={} \
                 ledger_consts={}{}",
                self.plan,
                self.loc.len(),
                if self.dist { "distributed" } else { "band" },
                self.injected,
                self.resourced,
                self.swapped,
                self.rolled,
                self.migrated,
                self.retired,
                self.drained,
                // Band turnovers actually achieved: the number the drain-set
                // rate was chosen for. Read this, not the configured rate.
                if self.loc.is_empty() {
                    0.0
                } else {
                    self.drained as f64 / self.loc.len() as f64
                },
                self.drain_steered,
                self.degenerate_rejects,
                self.cg_fragments,
                self.cg_narrow,
                self.cg_laddered,
                self.cg_gray,
                self.cg_fossils,
                self.ledger_consts,
                // Keyed on the flag alone, not on the epoch counter: the drain
                // set rewrites through the same carrier-sourced product
                // channel, so gating this on `retired` would silently drop the
                // caveat the moment `epoch` went to 0.
                if self.refill_used_carriers {
                    "  [port-uniformity forfeited: carrier-sourced refills]"
                } else {
                    ""
                }
            );
            if !self.distributed_fold_fragments.is_empty() {
                let mut original = self.distributed_fold_original_fragments.clone();
                let mut counts = self.distributed_fold_fragments.clone();
                let mut floors = self.distributed_fold_floors.clone();
                let below_floor = self
                    .distributed_fold_original_fragments
                    .iter()
                    .zip(&self.distributed_fold_fragments)
                    .zip(&self.distributed_fold_floors)
                    .filter(|&((original, count), floor)| {
                        *original > 0 && *floor > 0 && count < floor
                    })
                    .count();
                original.sort_unstable();
                counts.sort_unstable();
                floors.sort_unstable();
                println!(
                    "[prod] distributed-fold-fragments blocks={} original={}/{}/{} emitted={}/{}/{} floor={}/{}/{} below_floor={}",
                    counts.len(),
                    original[0],
                    original[original.len() / 2],
                    original[original.len() - 1],
                    counts[0],
                    counts[counts.len() / 2],
                    counts[counts.len() - 1],
                    floors[0],
                    floors[floors.len() / 2],
                    floors[floors.len() - 1],
                    below_floor,
                );
            }
        }
    }
}

impl ProdLedger {
    /// ANF atoms of one five-carrier logical literal.  An empty atom denotes
    /// the constant one, so the ordinary cartesian fold can treat constants
    /// exactly like every other summand.
    fn five_decode_atoms(
        &self,
        value: usize,
        positive: bool,
        state: &FiveCarrierState,
    ) -> Vec<Vec<(u16, bool)>> {
        let c = state.carriers[value];
        let mut atoms = match state.flavor {
            FiveCarrierFlavor::SuppliedQuadratic => vec![
                vec![(c[0] as u16, true)],
                vec![(c[1] as u16, true), (c[2] as u16, true)],
                vec![(c[1] as u16, true), (c[3] as u16, true)],
                vec![(c[2] as u16, true), (c[3] as u16, true)],
                vec![(c[1] as u16, true), (c[4] as u16, true)],
            ],
            FiveCarrierFlavor::StrongCubic => vec![
                vec![(c[0] as u16, true)],
                vec![(c[2] as u16, true)],
                vec![(c[2] as u16, true), (c[3] as u16, true)],
                vec![
                    (c[1] as u16, true),
                    (c[2] as u16, true),
                    (c[3] as u16, true),
                ],
                vec![
                    (c[1] as u16, true),
                    (c[2] as u16, true),
                    (c[4] as u16, true),
                ],
                vec![(c[3] as u16, true), (c[4] as u16, true)],
            ],
        };
        atoms.extend(self.slots[value].iter().map(|slot| slot.lits(&self.loc)));
        if self.consts[value] ^ !positive {
            atoms.push(Vec::new());
        }
        atoms
    }

    /// Add the supplied ANF atoms to a dirty accumulator exactly.  The phase-A
    /// spellings can leave a constant residual; a complementary one-control
    /// pair removes it without assuming a clean wire or changing the helper.
    fn gather_five_atoms_exact(
        &self,
        acc: u16,
        atoms: &[Vec<(u16, bool)>],
        plan: &[(u16, usize)],
        constant_helper: u16,
        seen: &mut SpellingLog,
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) {
        let mut correction = false;
        for (atom, &(helper, pivot)) in atoms.iter().zip(plan) {
            if atom.is_empty() {
                correction ^= true;
            } else {
                correction ^=
                    emit_atom_onto(acc, atom, helper, pivot, self.rung_menu, seen, rng, out);
            }
        }
        if correction {
            // (!h) + h = 1 for arbitrary dirty h.
            out.push(
                XGate::conj(acc, [(constant_helper, false)])
                    .expect("constant helper differs from accumulator"),
            );
            out.push(XGate::cnot(acc, constant_helper));
        }
    }

    /// The supplied four-phase g57 construction.  Both complete source
    /// decodes (five D atoms, every product-mask atom, and the ledger
    /// constant) are gathered, so the four `u*z` reads telescope to S_b*S_c.
    /// The remaining S_c term is emitted directly into target c0.
    ///
    /// SECURITY SCOPE: the tested zero weight-1/2 correlation is the ten-wire
    /// `(carrier_before, carrier_after)` trace of U0/U1.  As in the older Gray
    /// fold, a stronger space-time observer can XOR one accumulator immediately
    /// before and after its gather and recover S_b or S_c exactly.  Keeping the
    /// gather here is intentional fidelity to the supplied four-phase circuit,
    /// not a claim that this separate aggregate-accumulator witness vanished.
    fn fold_five_g57_gray(
        &mut self,
        gate: &XGate,
        state: &FiveCarrierState,
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) -> bool {
        if !gate.comp || gate.ctrls.len() != 2 {
            return false;
        }
        let Some(&(b_wire, false)) = gate.ctrls.iter().find(|&&(_, p)| !p) else {
            return false;
        };
        let Some(&(c_wire, true)) = gate.ctrls.iter().find(|&&(_, p)| p) else {
            return false;
        };
        let (t, b, c) = (gate.target as usize, b_wire as usize, c_wire as usize);
        let b_atoms = self.five_decode_atoms(b, true, state);
        let c_atoms = self.five_decode_atoms(c, true, state);
        if b_atoms.iter().chain(&c_atoms).any(|atom| atom.len() > 3) {
            return false;
        }

        // Borrow u, z, and h from three OTHER logical values.  Keeping their
        // five-tuples distinct ensures no local gate reads/writes two carriers
        // of one borrowed representation.  In a circuit too small to offer
        // three such values, the exact cartesian fold below is the safe
        // fallback.
        let mut borrow_values: Vec<usize> = (0..state.n)
            .filter(|value| *value != t && *value != b && *value != c)
            .collect();
        use rand::seq::SliceRandom;
        borrow_values.shuffle(rng);
        if borrow_values.len() < 3 {
            return false;
        }
        let u = state.carriers[borrow_values[0]][rng.random_range(0..5)] as u16;
        let z = state.carriers[borrow_values[1]][rng.random_range(0..5)] as u16;
        let helper = state.carriers[borrow_values[2]][rng.random_range(0..5)] as u16;
        let b_plan: Vec<(u16, usize)> = b_atoms
            .iter()
            .map(|atom| {
                let pivot = if atom.len() == 3 {
                    choose_pivot(atom, rng)
                } else {
                    0
                };
                (helper, pivot)
            })
            .collect();
        let c_plan: Vec<(u16, usize)> = c_atoms
            .iter()
            .map(|atom| {
                let pivot = if atom.len() == 3 {
                    choose_pivot(atom, rng)
                } else {
                    0
                };
                (helper, pivot)
            })
            .collect();
        let constant_helper = helper;
        let product = |out: &mut Vec<XGate>| {
            out.push(
                XGate::conj(state.carriers[t][0] as u16, [(u, true), (z, true)])
                    .expect("accumulators differ from target c0"),
            );
        };
        let mut seen: SpellingLog = std::collections::HashMap::new();

        // A(raw,raw), B(+S_b,raw), C(+S_b,+S_c), D(raw,+S_c).
        product(out);
        self.gather_five_atoms_exact(u, &b_atoms, &b_plan, constant_helper, &mut seen, rng, out);
        product(out);
        self.gather_five_atoms_exact(z, &c_atoms, &c_plan, constant_helper, &mut seen, rng, out);
        product(out);
        self.gather_five_atoms_exact(u, &b_atoms, &b_plan, constant_helper, &mut seen, rng, out);
        product(out);
        self.gather_five_atoms_exact(z, &c_atoms, &c_plan, constant_helper, &mut seen, rng, out);

        // g57 fires as 1 + S_c + S_b*S_c.  The caller already put the leading
        // one into kappa_t; emit the remaining S_c term atom-by-atom.
        for (atom, &(helper, pivot)) in c_atoms.iter().zip(&c_plan) {
            if atom.is_empty() {
                self.consts[t] ^= true;
                self.ledger_consts += 1;
                continue;
            }
            let residual = emit_atom_onto(
                state.carriers[t][0] as u16,
                atom,
                helper,
                pivot,
                self.rung_menu,
                &mut seen,
                rng,
                out,
            );
            if residual {
                self.consts[t] ^= true;
                self.ledger_consts += 1;
            }
        }
        self.cg_gray += 1;
        true
    }

    /// Apply one source gate to the five-carrier encoding.  U0 is always
    /// applied to the target tuple; c0 is then toggled by the source gate's
    /// logical firing function.  This realizes U0 on output zero and U1 on
    /// output one while preserving the product masks and ledger convention.
    fn fold_five(
        &mut self,
        gate: &XGate,
        state: &FiveCarrierState,
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) {
        let t = gate.target as usize;
        state.emit_update(t, out);
        if gate.comp {
            self.consts[t] ^= true;
            self.ledger_consts += 1;
        }
        if self.gray_fold && self.fold_five_g57_gray(gate, state, rng, out) {
            return;
        }

        let lists: Vec<Vec<Vec<(u16, bool)>>> = gate
            .ctrls
            .iter()
            .map(|&(wire, positive)| self.five_decode_atoms(wire as usize, positive, state))
            .collect();
        if self.micro_gray {
            let protected: Vec<u16> = state.carriers[t].iter().map(|&wire| wire as u16).collect();
            let carrier_groups: Vec<Vec<u16>> = state
                .carriers
                .iter()
                .map(|group| group.iter().map(|&wire| wire as u16).collect())
                .collect();
            if self.fold_micro_product(
                state.carriers[t][0] as u16,
                &lists,
                &protected,
                &carrier_groups,
                rng,
                out,
            ) {
                return;
            }
        }
        if self.sentinel_gray {
            let mut sentinel_lists = lists.clone();
            if sentinel_lists
                .iter_mut()
                .all(absorb_constants_into_linear_atom)
            {
                let protected: Vec<u16> =
                    state.carriers[t].iter().map(|&wire| wire as u16).collect();
                let carrier_groups: Vec<Vec<u16>> = state
                    .carriers
                    .iter()
                    .map(|group| group.iter().map(|&wire| wire as u16).collect())
                    .collect();
                if self.fold_sentinel_product(
                    t,
                    state.carriers[t][0] as u16,
                    &sentinel_lists,
                    &protected,
                    &carrier_groups,
                    rng,
                    out,
                ) {
                    return;
                }
            }
        }
        if lists.is_empty() {
            // AND over no controls is one.
            self.consts[t] ^= true;
            self.ledger_consts += 1;
            return;
        }

        let mut combo = vec![0usize; lists.len()];
        'odometer: loop {
            let atoms: Vec<Vec<(u16, bool)>> = lists
                .iter()
                .zip(&combo)
                .map(|(list, &index)| list[index].clone())
                .collect();
            let mut lits = interleave_atoms(&atoms);
            if lits.is_empty() {
                self.consts[t] ^= true;
                self.ledger_consts += 1;
            } else if normalize_lits(&mut lits).is_some() {
                if let Some(fragment) = XGate::conj(state.carriers[t][0] as u16, lits) {
                    if fragment.width() <= 2 {
                        self.cg_narrow += 1;
                    } else {
                        self.cg_fossils += 1;
                    }
                    out.push(fragment);
                    self.cg_fragments += 1;
                }
            }
            let mut axis = 0;
            loop {
                combo[axis] += 1;
                if combo[axis] < lists[axis].len() {
                    break;
                }
                combo[axis] = 0;
                axis += 1;
                if axis == combo.len() {
                    break 'odometer;
                }
            }
        }
    }

    /// Five-carrier variant of the rolling-band role swap.
    fn roll_five(
        &mut self,
        state: &mut FiveCarrierState,
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) {
        if !self.enabled() || self.loc.is_empty() {
            return;
        }
        let total = self.carrier_total + self.loc.len();
        let var = rng.random_range(0..self.loc.len());
        let from = self.loc[var];
        let to = loop {
            let wire = rng.random_range(0..total) as u16;
            if wire != from {
                break wire;
            }
        };
        if let Some(other) = self.loc.iter().position(|&wire| wire == to) {
            self.loc[other] = from;
        } else {
            let mut found = false;
            'values: for carriers in &mut state.carriers {
                for wire in carriers {
                    if *wire == to as usize {
                        *wire = from as usize;
                        found = true;
                        break 'values;
                    }
                }
            }
            assert!(found, "five-carrier roll partner has no role");
        }
        self.loc[var] = to;
        let (a, b) = if rng.random_bool(0.5) {
            (from, to)
        } else {
            (to, from)
        };
        for (target, source) in [(a, b), (b, a), (a, b)] {
            emit_transvection_mixed(target, source, total, rng, out);
        }
        self.rolled += 1;
    }
}

impl ProdLedger {
    /// Full ANF decode of one six-carrier logical literal. An empty atom is the
    /// constant one contributed by the ledger or a negative source literal.
    fn six_decode_atoms(
        &self,
        value: usize,
        positive: bool,
        state: &SixCarrierState,
    ) -> Vec<Vec<(u16, bool)>> {
        let carriers = state.carriers[value];
        let mut atoms: Vec<Vec<(u16, bool)>> = SIX_CARRIER_D_ATOMS
            .iter()
            .map(|atom| {
                atom.iter()
                    .map(|&lane| (carriers[lane as usize] as u16, true))
                    .collect()
            })
            .collect();
        atoms.extend(self.slots[value].iter().map(|slot| slot.lits(&self.loc)));
        if self.consts[value] ^ !positive {
            atoms.push(Vec::new());
        }
        atoms
    }

    /// Four-phase dirty-accumulator fold for a six-carrier g57. It gathers the
    /// complete D+mask+constant decode of each operand, so the four product
    /// reads telescope to S_b*S_c; the direct final pass contributes S_c.
    fn fold_six_g57_gray(
        &mut self,
        gate: &XGate,
        state: &SixCarrierState,
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) -> bool {
        if !gate.comp || gate.ctrls.len() != 2 {
            return false;
        }
        let Some(&(b_wire, false)) = gate.ctrls.iter().find(|&&(_, polarity)| !polarity) else {
            return false;
        };
        let Some(&(c_wire, true)) = gate.ctrls.iter().find(|&&(_, polarity)| polarity) else {
            return false;
        };
        let (target, b, c) = (gate.target as usize, b_wire as usize, c_wire as usize);
        let b_atoms = self.six_decode_atoms(b, true, state);
        let c_atoms = self.six_decode_atoms(c, true, state);
        if b_atoms.iter().chain(&c_atoms).any(|atom| atom.len() > 3) {
            return false;
        }

        // Use three distinct unrelated logical values for u, z, and the dirty
        // helper. Small source widths safely fall back to the exact cartesian
        // fold when this pool does not exist.
        let mut borrow_values: Vec<usize> = (0..state.n)
            .filter(|value| *value != target && *value != b && *value != c)
            .collect();
        use rand::seq::SliceRandom;
        borrow_values.shuffle(rng);
        if borrow_values.len() < 3 {
            return false;
        }
        let u = state.carriers[borrow_values[0]][rng.random_range(0..6)] as u16;
        let z = state.carriers[borrow_values[1]][rng.random_range(0..6)] as u16;
        let helper = state.carriers[borrow_values[2]][rng.random_range(0..6)] as u16;
        let b_plan: Vec<(u16, usize)> = b_atoms
            .iter()
            .map(|atom| {
                let pivot = if atom.len() == 3 {
                    choose_pivot(atom, rng)
                } else {
                    0
                };
                (helper, pivot)
            })
            .collect();
        let c_plan: Vec<(u16, usize)> = c_atoms
            .iter()
            .map(|atom| {
                let pivot = if atom.len() == 3 {
                    choose_pivot(atom, rng)
                } else {
                    0
                };
                (helper, pivot)
            })
            .collect();
        let product = |out: &mut Vec<XGate>| {
            out.push(
                XGate::conj(state.carriers[target][0] as u16, [(u, true), (z, true)])
                    .expect("accumulators differ from target c0"),
            );
        };
        let mut seen: SpellingLog = std::collections::HashMap::new();

        product(out);
        self.gather_five_atoms_exact(u, &b_atoms, &b_plan, helper, &mut seen, rng, out);
        product(out);
        self.gather_five_atoms_exact(z, &c_atoms, &c_plan, helper, &mut seen, rng, out);
        product(out);
        self.gather_five_atoms_exact(u, &b_atoms, &b_plan, helper, &mut seen, rng, out);
        product(out);
        self.gather_five_atoms_exact(z, &c_atoms, &c_plan, helper, &mut seen, rng, out);

        for (atom, &(atom_helper, pivot)) in c_atoms.iter().zip(&c_plan) {
            if atom.is_empty() {
                self.consts[target] ^= true;
                self.ledger_consts += 1;
                continue;
            }
            let residual = emit_atom_onto(
                state.carriers[target][0] as u16,
                atom,
                atom_helper,
                pivot,
                self.rung_menu,
                &mut seen,
                rng,
                out,
            );
            if residual {
                self.consts[target] ^= true;
                self.ledger_consts += 1;
            }
        }
        self.cg_gray += 1;
        true
    }

    /// Apply one heterogeneous source gate under the six-carrier decode.
    fn fold_six(
        &mut self,
        gate: &XGate,
        state: &SixCarrierState,
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) {
        let target = gate.target as usize;
        state.emit_update(target, out);
        if gate.comp {
            self.consts[target] ^= true;
            self.ledger_consts += 1;
        }
        if self.gray_fold && self.fold_six_g57_gray(gate, state, rng, out) {
            return;
        }

        let lists: Vec<Vec<Vec<(u16, bool)>>> = gate
            .ctrls
            .iter()
            .map(|&(wire, positive)| self.six_decode_atoms(wire as usize, positive, state))
            .collect();
        if self.micro_gray {
            let protected: Vec<u16> = state.carriers[target]
                .iter()
                .map(|&wire| wire as u16)
                .collect();
            let carrier_groups: Vec<Vec<u16>> = state
                .carriers
                .iter()
                .map(|group| group.iter().map(|&wire| wire as u16).collect())
                .collect();
            if self.fold_micro_product(
                state.carriers[target][0] as u16,
                &lists,
                &protected,
                &carrier_groups,
                rng,
                out,
            ) {
                return;
            }
        }
        if self.sentinel_gray {
            let mut sentinel_lists = lists.clone();
            if sentinel_lists
                .iter_mut()
                .all(absorb_constants_into_linear_atom)
            {
                let protected: Vec<u16> = state.carriers[target]
                    .iter()
                    .map(|&wire| wire as u16)
                    .collect();
                let carrier_groups: Vec<Vec<u16>> = state
                    .carriers
                    .iter()
                    .map(|group| group.iter().map(|&wire| wire as u16).collect())
                    .collect();
                if self.fold_sentinel_product(
                    target,
                    state.carriers[target][0] as u16,
                    &sentinel_lists,
                    &protected,
                    &carrier_groups,
                    rng,
                    out,
                ) {
                    return;
                }
            }
        }
        if lists.is_empty() {
            self.consts[target] ^= true;
            self.ledger_consts += 1;
            return;
        }

        let mut combo = vec![0usize; lists.len()];
        'odometer: loop {
            let atoms: Vec<Vec<(u16, bool)>> = lists
                .iter()
                .zip(&combo)
                .map(|(list, &index)| list[index].clone())
                .collect();
            let mut lits = interleave_atoms(&atoms);
            if lits.is_empty() {
                self.consts[target] ^= true;
                self.ledger_consts += 1;
            } else if normalize_lits(&mut lits).is_some() {
                if let Some(fragment) = XGate::conj(state.carriers[target][0] as u16, lits) {
                    if fragment.width() <= 2 {
                        self.cg_narrow += 1;
                    } else {
                        self.cg_fossils += 1;
                    }
                    out.push(fragment);
                    self.cg_fragments += 1;
                }
            }
            let mut axis = 0;
            loop {
                combo[axis] += 1;
                if combo[axis] < lists[axis].len() {
                    break;
                }
                combo[axis] = 0;
                axis += 1;
                if axis == combo.len() {
                    break 'odometer;
                }
            }
        }
    }

    fn roll_six(&mut self, state: &mut SixCarrierState, rng: &mut impl Rng, out: &mut Vec<XGate>) {
        if !self.enabled() || self.loc.is_empty() {
            return;
        }
        let total = self.carrier_total + self.loc.len();
        let var = rng.random_range(0..self.loc.len());
        let from = self.loc[var];
        let to = loop {
            let wire = rng.random_range(0..total) as u16;
            if wire != from {
                break wire;
            }
        };
        if let Some(other) = self.loc.iter().position(|&wire| wire == to) {
            self.loc[other] = from;
        } else {
            let mut found = false;
            'values: for carriers in &mut state.carriers {
                for wire in carriers {
                    if *wire == to as usize {
                        *wire = from as usize;
                        found = true;
                        break 'values;
                    }
                }
            }
            assert!(found, "six-carrier roll partner has no role");
        }
        self.loc[var] = to;
        let (a, b) = if rng.random_bool(0.5) {
            (from, to)
        } else {
            (to, from)
        };
        for (target, source) in [(a, b), (b, a), (a, b)] {
            emit_transvection_mixed(target, source, total, rng, out);
        }
        self.rolled += 1;
    }
}

impl ProdLedger {
    /// Full ANF decode of one seven-carrier logical literal. An empty atom is
    /// the constant one contributed by the ledger or a negative source
    /// literal. The quartic decode atom is gathered by [`emit_atom_onto`]'s
    /// dirty-helper double sweep.
    fn seven_decode_atoms(
        &self,
        value: usize,
        positive: bool,
        state: &SevenCarrierState,
    ) -> Vec<Vec<(u16, bool)>> {
        let carriers = state.carriers[value];
        let mut atoms: Vec<Vec<(u16, bool)>> = SEVEN_CARRIER_D_ATOMS
            .iter()
            .map(|atom| {
                atom.iter()
                    .map(|&lane| (carriers[lane as usize] as u16, true))
                    .collect()
            })
            .collect();
        atoms.extend(self.slots[value].iter().map(|slot| slot.lits(&self.loc)));
        if self.consts[value] ^ !positive {
            atoms.push(Vec::new());
        }
        atoms
    }

    /// Four-phase dirty-accumulator fold for a seven-carrier g57. It gathers
    /// the complete decode, including the quartic c3*c4*c5*c6 atom, so the
    /// four product reads telescope to S_b*S_c and the direct pass supplies
    /// S_c. Every gathered atom uses at most three controls after adjustment.
    fn fold_seven_g57_gray(
        &mut self,
        gate: &XGate,
        state: &SevenCarrierState,
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) -> bool {
        if !gate.comp || gate.ctrls.len() != 2 {
            return false;
        }
        let Some(&(b_wire, false)) = gate.ctrls.iter().find(|&&(_, polarity)| !polarity) else {
            return false;
        };
        let Some(&(c_wire, true)) = gate.ctrls.iter().find(|&&(_, polarity)| polarity) else {
            return false;
        };
        let (target, b, c) = (gate.target as usize, b_wire as usize, c_wire as usize);
        let b_atoms = self.seven_decode_atoms(b, true, state);
        let c_atoms = self.seven_decode_atoms(c, true, state);
        if b_atoms.iter().chain(&c_atoms).any(|atom| atom.len() > 4) {
            return false;
        }

        // u, z, and h come from three distinct unrelated logical values. This
        // keeps all borrowed wires distinct even after role rolling. For small
        // sources the ordinary exact cartesian fold remains the safe fallback.
        let mut borrow_values: Vec<usize> = (0..state.n)
            .filter(|value| *value != target && *value != b && *value != c)
            .collect();
        use rand::seq::SliceRandom;
        borrow_values.shuffle(rng);
        if borrow_values.len() < 3 {
            return false;
        }
        let u = state.carriers[borrow_values[0]][rng.random_range(0..7)] as u16;
        let z = state.carriers[borrow_values[1]][rng.random_range(0..7)] as u16;
        let helper = state.carriers[borrow_values[2]][rng.random_range(0..7)] as u16;
        let b_plan: Vec<(u16, usize)> = b_atoms
            .iter()
            .map(|atom| {
                let pivot = if atom.len() == 3 {
                    choose_pivot(atom, rng)
                } else {
                    0
                };
                (helper, pivot)
            })
            .collect();
        let c_plan: Vec<(u16, usize)> = c_atoms
            .iter()
            .map(|atom| {
                let pivot = if atom.len() == 3 {
                    choose_pivot(atom, rng)
                } else {
                    0
                };
                (helper, pivot)
            })
            .collect();
        let product = |out: &mut Vec<XGate>| {
            out.push(
                XGate::conj(state.carriers[target][0] as u16, [(u, true), (z, true)])
                    .expect("accumulators differ from target c0"),
            );
        };
        let mut seen: SpellingLog = std::collections::HashMap::new();

        product(out);
        self.gather_five_atoms_exact(u, &b_atoms, &b_plan, helper, &mut seen, rng, out);
        product(out);
        self.gather_five_atoms_exact(z, &c_atoms, &c_plan, helper, &mut seen, rng, out);
        product(out);
        self.gather_five_atoms_exact(u, &b_atoms, &b_plan, helper, &mut seen, rng, out);
        product(out);
        self.gather_five_atoms_exact(z, &c_atoms, &c_plan, helper, &mut seen, rng, out);

        for (atom, &(atom_helper, pivot)) in c_atoms.iter().zip(&c_plan) {
            if atom.is_empty() {
                self.consts[target] ^= true;
                self.ledger_consts += 1;
                continue;
            }
            let residual = emit_atom_onto(
                state.carriers[target][0] as u16,
                atom,
                atom_helper,
                pivot,
                self.rung_menu,
                &mut seen,
                rng,
                out,
            );
            if residual {
                self.consts[target] ^= true;
                self.ledger_consts += 1;
            }
        }
        self.cg_gray += 1;
        true
    }

    fn make_seven_distributed_fragment(
        &mut self,
        target: u16,
        mut lits: Vec<(u16, bool)>,
    ) -> Option<XGate> {
        if normalize_lits(&mut lits).is_none() {
            return None;
        }
        let Some(fragment) = XGate::conj(target, lits) else {
            return None;
        };
        Some(fragment)
    }

    fn record_seven_distributed_fragment(&mut self, fragment: &XGate) {
        if fragment.width() <= 2 {
            self.cg_narrow += 1;
        } else {
            self.cg_fossils += 1;
        }
        self.cg_fragments += 1;
    }

    /// Opt-in expanded seven-carrier fold with nonlinear D-preserving refreshes
    /// between every pair of consecutive emitted product fragments.
    ///
    /// A random automorphism first chooses which physical linear lane receives
    /// this occurrence's ANF fragments.  Each fragment still changes D by its
    /// firing value, while every inserted three-gate shear preserves D exactly.
    /// There is deliberately no fixed U0 bookend: exhaustive local trace tests
    /// found that even one such update collapses the intended distance gain.
    fn fold_seven_distributed(
        &mut self,
        gate: &XGate,
        state: &SevenCarrierState,
        partition_floor: usize,
        partition_helper_limit: usize,
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) {
        assert!(
            !self.gray_fold && !self.micro_gray && !self.sentinel_gray,
            "distributed seven-carrier switching currently requires expanded/no-Gray folding"
        );

        let target_value = gate.target as usize;
        let roles = seven_carrier_role_automorphism(rng);
        let physical_target = state.carriers[target_value][roles[0] as usize] as u16;
        if gate.comp {
            self.consts[target_value] ^= true;
            self.ledger_consts += 1;
        }
        let lists: Vec<Vec<Vec<(u16, bool)>>> = gate
            .ctrls
            .iter()
            .map(|&(wire, positive)| self.seven_decode_atoms(wire as usize, positive, state))
            .collect();

        let mut fragments = Vec::new();
        let mut physical_constant = false;
        if lists.is_empty() {
            if partition_floor > 0 {
                physical_constant = true;
            } else {
                self.consts[target_value] ^= true;
                self.ledger_consts += 1;
                self.distributed_fold_original_fragments.push(0);
                self.distributed_fold_fragments.push(0);
                self.distributed_fold_floors.push(0);
                return;
            }
        } else {
            let mut combo = vec![0usize; lists.len()];
            'odometer: loop {
                let atoms: Vec<Vec<(u16, bool)>> = lists
                    .iter()
                    .zip(&combo)
                    .map(|(list, &index)| list[index].clone())
                    .collect();
                let lits = interleave_atoms(&atoms);
                if lits.is_empty() {
                    if partition_floor > 0 {
                        physical_constant ^= true;
                    } else {
                        self.consts[target_value] ^= true;
                        self.ledger_consts += 1;
                    }
                } else if let Some(fragment) =
                    self.make_seven_distributed_fragment(physical_target, lits)
                {
                    fragments.push(fragment);
                }

                let mut axis = 0;
                loop {
                    combo[axis] += 1;
                    if combo[axis] < lists[axis].len() {
                        break;
                    }
                    combo[axis] = 0;
                    axis += 1;
                    if axis == combo.len() {
                        break 'odometer;
                    }
                }
            }
        }
        if physical_constant {
            fragments.push(XGate::x_gate(physical_target));
        }

        let original_fragment_count = fragments.len();
        if partition_floor > 0 && !fragments.is_empty() && fragments.len() < partition_floor {
            let controls: std::collections::HashSet<usize> =
                gate.ctrls.iter().map(|&(wire, _)| wire as usize).collect();
            assert!(partition_helper_limit <= state.n);
            let mut helper_values: Vec<usize> = (0..partition_helper_limit)
                .filter(|&value| value != target_value && !controls.contains(&value))
                .collect();
            let mut split_bits = 0usize;
            let cell_count = loop {
                let cells = 1usize
                    .checked_shl(split_bits as u32)
                    .expect("partition floor needs too many selector bits");
                if fragments
                    .len()
                    .checked_mul(cells)
                    .expect("partition branch count overflow")
                    >= partition_floor
                {
                    break cells;
                }
                split_bits += 1;
            };
            assert!(
                helper_values.len() >= split_bits,
                "partitioned seven-carrier folds need one distinct logical value outside the target and every gate control per split bit"
            );
            helper_values.shuffle(rng);
            // Use canonical c0 from distinct logical values in the caller's
            // eligible prefix.  A sliced-sandwich caller can restrict that
            // prefix to its known-live data half; this avoids selecting fixed
            // upper-half coordinates.  Several lanes of one freshly
            // initialized representative could likewise all be fixed zero.
            // Rolls change physical locations, not roles, so this remains c0.
            let helpers: Vec<u16> = helper_values
                .into_iter()
                .take(split_bits)
                .map(|value| state.carriers[value][0] as u16)
                .collect();
            let mut partitioned = Vec::with_capacity(fragments.len() * cell_count);
            for fragment in fragments {
                // Independent cell order per original fragment prevents the
                // same physical lane from walking through identical polarity
                // transitions at every product occurrence.
                let mut cells: Vec<usize> = (0..cell_count).collect();
                cells.shuffle(rng);
                for cell in cells {
                    let mut lits: Vec<(u16, bool)> = fragment.ctrls.iter().copied().collect();
                    lits.extend(
                        helpers
                            .iter()
                            .enumerate()
                            .map(|(bit, &wire)| (wire, cell & (1 << bit) != 0)),
                    );
                    partitioned.push(
                        XGate::conj(fragment.target, lits)
                            .expect("unrelated helper value cannot contradict a source fragment"),
                    );
                }
            }
            fragments = partitioned;
        }

        let mut used_selectors = std::collections::HashSet::new();
        let rotation = rng.random_range(0..4usize);
        let fragment_count = fragments.len();
        self.distributed_fold_original_fragments
            .push(original_fragment_count);
        self.distributed_fold_fragments.push(fragment_count);
        self.distributed_fold_floors.push(partition_floor);
        for (index, fragment) in fragments.into_iter().enumerate() {
            self.record_seven_distributed_fragment(&fragment);
            out.push(fragment);
            if index + 1 == fragment_count {
                continue;
            }
            let selector =
                draw_seven_carrier_shear_selector(state, target_value, &mut used_selectors, rng);
            let x = 3 + ((rotation + index) % 4) as u8;
            emit_seven_carrier_preserving_shear(state, target_value, &roles, x, selector, out);
        }
    }

    /// Apply one heterogeneous source gate under the legacy seven-carrier
    /// decode.  This retains the distinguished-c0 switch for byte-for-byte
    /// compatibility; the distributed sibling is explicitly selected by its
    /// own public gadgetizer entry point.
    fn fold_seven(
        &mut self,
        gate: &XGate,
        state: &SevenCarrierState,
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) {
        let target = gate.target as usize;
        emit_seven_carrier_update(&state.carriers[target], out);
        if gate.comp {
            self.consts[target] ^= true;
            self.ledger_consts += 1;
        }
        if self.gray_fold && self.fold_seven_g57_gray(gate, state, rng, out) {
            return;
        }

        let lists: Vec<Vec<Vec<(u16, bool)>>> = gate
            .ctrls
            .iter()
            .map(|&(wire, positive)| self.seven_decode_atoms(wire as usize, positive, state))
            .collect();
        if self.micro_gray {
            let protected: Vec<u16> = state.carriers[target]
                .iter()
                .map(|&wire| wire as u16)
                .collect();
            let carrier_groups: Vec<Vec<u16>> = state
                .carriers
                .iter()
                .map(|group| group.iter().map(|&wire| wire as u16).collect())
                .collect();
            if self.fold_micro_product(
                state.carriers[target][0] as u16,
                &lists,
                &protected,
                &carrier_groups,
                rng,
                out,
            ) {
                return;
            }
        }
        if self.sentinel_gray {
            let mut sentinel_lists = lists.clone();
            if sentinel_lists
                .iter_mut()
                .all(absorb_constants_into_linear_atom)
            {
                let protected: Vec<u16> = state.carriers[target]
                    .iter()
                    .map(|&wire| wire as u16)
                    .collect();
                let carrier_groups: Vec<Vec<u16>> = state
                    .carriers
                    .iter()
                    .map(|group| group.iter().map(|&wire| wire as u16).collect())
                    .collect();
                if self.fold_sentinel_product(
                    target,
                    state.carriers[target][0] as u16,
                    &sentinel_lists,
                    &protected,
                    &carrier_groups,
                    rng,
                    out,
                ) {
                    return;
                }
            }
        }
        if lists.is_empty() {
            self.consts[target] ^= true;
            self.ledger_consts += 1;
            return;
        }

        let mut combo = vec![0usize; lists.len()];
        'odometer: loop {
            let atoms: Vec<Vec<(u16, bool)>> = lists
                .iter()
                .zip(&combo)
                .map(|(list, &index)| list[index].clone())
                .collect();
            let mut lits = interleave_atoms(&atoms);
            if lits.is_empty() {
                self.consts[target] ^= true;
                self.ledger_consts += 1;
            } else if normalize_lits(&mut lits).is_some() {
                if let Some(fragment) = XGate::conj(state.carriers[target][0] as u16, lits) {
                    if fragment.width() <= 2 {
                        self.cg_narrow += 1;
                    } else {
                        self.cg_fossils += 1;
                    }
                    out.push(fragment);
                    self.cg_fragments += 1;
                }
            }
            let mut axis = 0;
            loop {
                combo[axis] += 1;
                if combo[axis] < lists[axis].len() {
                    break;
                }
                combo[axis] = 0;
                axis += 1;
                if axis == combo.len() {
                    break 'odometer;
                }
            }
        }
    }

    fn roll_seven(
        &mut self,
        state: &mut SevenCarrierState,
        rng: &mut impl Rng,
        out: &mut Vec<XGate>,
    ) {
        if !self.enabled() || self.loc.is_empty() {
            return;
        }
        let total = self.carrier_total + self.loc.len();
        let var = rng.random_range(0..self.loc.len());
        let from = self.loc[var];
        let to = loop {
            let wire = rng.random_range(0..total) as u16;
            if wire != from {
                break wire;
            }
        };
        if let Some(other) = self.loc.iter().position(|&wire| wire == to) {
            self.loc[other] = from;
        } else {
            let mut found = false;
            'values: for carriers in &mut state.carriers {
                for wire in carriers {
                    if *wire == to as usize {
                        *wire = from as usize;
                        found = true;
                        break 'values;
                    }
                }
            }
            assert!(found, "seven-carrier roll partner has no role");
        }
        self.loc[var] = to;
        let (a, b) = if rng.random_bool(0.5) {
            (from, to)
        } else {
            (to, from)
        };
        for (target, source) in [(a, b), (b, a), (a, b)] {
            emit_transvection_mixed(target, source, total, rng, out);
        }
        self.rolled += 1;
    }
}

/// `target ^= source`, realized either as the plain CNOT or as the two
/// width-2 conjunctions `source AND u` and `source AND NOT u` over a random
/// helper `u` — which sum to exactly `source`, add no constant, and (carrying
/// opposite literals on `u`) do not collide, so the commuting shuffle is free
/// to drive them apart. The point is the gate SHAPE: a wire written only by
/// width-1 CNOTs stands out from one written by the body's conjunctions, and
/// a band variable's wire is written by nothing but its rolls.
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
    let u = random_wire_except(total, &[target as usize, source as usize], rng) as u16;
    // The two halves sum to `source`. Spelling each in the g57 form keeps the
    // roll inside the store's vocabulary; it is sound WITHOUT a ledger because
    // emit_g57_form returns konst=true for both the (true,true) and the
    // (true,false) polarity case, so the two residual constants cancel each
    // other and a roll -- which has no ledger to defer a constant to -- stays
    // exact. Asserted rather than assumed.
    let mut konst = false;
    for polarity in [true, false] {
        konst ^= emit_g57_form(target, &[(source, true), (u, polarity)], rng, out);
    }
    debug_assert!(
        !konst,
        "transvection halves must leave no net constant: a roll has no ledger"
    );
}

/// W0: fill each band wire with an unbiased data-dependent bit — its input
/// junk XOR a random (weight >= 2) subset of the data wires, emitted at the
/// input port while the data wires still hold x. Over uniform x every fill is
/// exactly unbiased; under the hmap / zero-slice conventions (non-data inputs
/// pinned to 0) the band reads exactly <alpha, x>. The band is never written
/// again, so every registered product term is time-invariant.
fn emit_band_fill(n: usize, band: &[u16], rng: &mut impl Rng, out: &mut Vec<XGate>) {
    emit_band_fill_src(n, band, rng, out)
}

/// [`emit_band_fill`] with the fill sources restricted to data wires below
/// `src_hi`. The closing-slice design sources BOTH fills from the low data
/// half: at the input port that is where x lives (the high half is zero, so
/// nothing is lost), and at the output port the low half is still MASKED —
/// a fill CNOT reading a bare payload wire writes that payload bit out as a
/// local segment delta, which is exactly the boundary flip-match class the
/// redesign is eliminating.
fn emit_band_fill_src(src_hi: usize, band: &[u16], rng: &mut impl Rng, out: &mut Vec<XGate>) {
    for &band_wire in band {
        loop {
            let subset: Vec<usize> = (0..src_hi).filter(|_| rng.random_bool(0.5)).collect();
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

/// Linear band fill used by the opt-in boundary-partition experiment, with
/// the exact affine support of every band wire on a known-live input prefix.
/// Its draw/emission order deliberately matches [`emit_band_fill`].
fn emit_band_fill_with_live_supports(
    n: usize,
    band: &[u16],
    live_prefix: usize,
    rng: &mut impl Rng,
    out: &mut Vec<XGate>,
) -> Vec<Vec<u64>> {
    assert!(live_prefix <= n);
    let words = live_prefix.div_ceil(64);
    let mut supports = Vec::with_capacity(band.len());
    for &band_wire in band {
        loop {
            let subset: Vec<usize> = (0..n).filter(|_| rng.random_bool(0.5)).collect();
            if subset.len() < 2 {
                continue;
            }
            let mut support = vec![0u64; words];
            for data_wire in subset {
                if data_wire < live_prefix {
                    support[data_wire / 64] ^= 1u64 << (data_wire % 64);
                }
                out.push(XGate::cnot(band_wire, data_wire as u16));
            }
            supports.push(support);
            break;
        }
    }
    supports
}

/// W0 (nonlinear cascade): fill each band wire with
///   junk ^ pivot ^ (small linear part) ^ XOR of `fill_nl` two-source
///   products,
/// where a product's sources are data wires AND already-filled band wires —
/// the cascade multiplies input-degree up the band while every emitted gate
/// stays a CNOT or g57 (no wide gates, no cheap affine invariants for a
/// forward-learning SAT attacker). The fresh linear pivot is excluded from
/// the rest of the wire's transitive data support, so every band bit is
/// exactly balanced over uniform x for any junk, and distinct pivot draws
/// keep the fill's linear parts non-degenerate. Also used as the mirror fill
/// on the output side (the data wires then hold the output port).
fn emit_band_fill_nl(
    n: usize,
    band: &[u16],
    fill_nl: usize,
    rng: &mut impl Rng,
    out: &mut Vec<XGate>,
) {
    emit_band_fill_nl_pivots(n, band, fill_nl, false, rng, out)
}

/// The cascade fill, optionally with a RESERVED PIVOT BLOCK.
///
/// Per-wire balance only needs each wire's own pivot excluded from its own
/// support, which is what the legacy path does. But a mask multiplies THREE
/// band wires together, and marginal balance of each says nothing about the
/// product: the statistics the encoding claims (a degree-`d` term firing at
/// exactly `2^-d`, terms piling up independently) need the band bits to be
/// JOINTLY uniform.
///
/// With `reserve` set, the pivots are drawn WITHOUT replacement and the whole
/// pivot set is excluded from every wire's non-pivot material. Then, for any
/// fixing of the non-pivot inputs, the map
///     (x_{p_1}, .., x_{p_b}) -> (band_1, .., band_b)
/// is unit lower-triangular — band_j is x_{p_j} XOR a function of the
/// non-pivot inputs and of EARLIER band wires — hence a bijection. So the band
/// is exactly uniform on {0,1}^b and independent of the rest, for ANY choice of
/// the nonlinear part. The cascade is what triangularity absorbs, so it
/// survives untouched, and the cost is zero gates.
///
/// Requires `b <= n` reservable pivots; falls back to the legacy draw
/// otherwise (the output-side mirror fill targets more wires than there are
/// data wires, so it cannot carry the guarantee).
fn emit_band_fill_nl_pivots(
    n: usize,
    band: &[u16],
    fill_nl: usize,
    reserve: bool,
    rng: &mut impl Rng,
    out: &mut Vec<XGate>,
) {
    emit_band_fill_nl_pivots_src(n, band, fill_nl, reserve, rng, out)
}

/// [`emit_band_fill_nl_pivots`] with data sources restricted to wires below
/// `src_hi` (see [`emit_band_fill_src`] for why the closing-slice design
/// sources both fills from the low data half).
fn emit_band_fill_nl_pivots_src(
    n: usize,
    band: &[u16],
    fill_nl: usize,
    reserve: bool,
    rng: &mut impl Rng,
    out: &mut Vec<XGate>,
) {
    // Reserving needs room to spare, not just b <= n: every reserved pivot is
    // removed from the linear pool AND from the product-source draw, so at
    // b == n there is no legal non-pivot data wire left and the draw cannot
    // terminate. Leave at least a quarter of the data wires unreserved.
    let reserve = reserve && band.len() + band.len() / 3 + 2 <= n;
    // One private pivot per band wire, drawn without replacement.
    let pivots: Vec<usize> = if reserve {
        let mut all: Vec<usize> = (0..n).collect();
        for i in 0..band.len() {
            let j = i + rng.random_range(0..(n - i));
            all.swap(i, j);
        }
        all.truncate(band.len());
        all
    } else {
        Vec::new()
    };
    let reserved: std::collections::HashSet<usize> = pivots.iter().copied().collect();
    // Transitive data support per already-filled band wire (pivot included).
    let mut supports: Vec<std::collections::HashSet<usize>> = Vec::new();
    for (index, &band_wire) in band.iter().enumerate() {
        let pivot = if reserve {
            pivots[index]
        } else {
            rng.random_range(0..n)
        };
        let mut support: std::collections::HashSet<usize> = std::iter::once(pivot).collect();
        out.push(XGate::cnot(band_wire, pivot as u16));
        // Small linear part: 1..=min(7, n-1) extra data wires besides the pivot
        // (and, when reserved, besides every OTHER wire's pivot — that is the
        // whole difference, and it is what makes the map triangular).
        let lin_max = (n - 1).min(7);
        let lin_w = 1 + rng.random_range(0..lin_max);
        let mut pool: Vec<usize> = (0..n)
            .filter(|&w| w != pivot && !(reserve && reserved.contains(&w)))
            .collect();
        let lin_w = lin_w.min(pool.len());
        for _ in 0..lin_w {
            let i = rng.random_range(0..pool.len());
            let w = pool.swap_remove(i);
            support.insert(w);
            out.push(XGate::cnot(band_wire, w as u16));
        }
        // Nonlinear products, cascading over pivot-free earlier band wires.
        let eligible_band: Vec<usize> = (0..index)
            .filter(|&i| !supports[i].contains(&pivot))
            .collect();
        for _ in 0..fill_nl {
            let mut draw = |exclude: Option<u16>| loop {
                let wire = if !eligible_band.is_empty() && rng.random_bool(0.5) {
                    band[eligible_band[rng.random_range(0..eligible_band.len())]]
                } else {
                    loop {
                        let w = rng.random_range(0..n);
                        if w != pivot && !(reserve && reserved.contains(&w)) {
                            break w as u16;
                        }
                    }
                };
                if Some(wire) != exclude {
                    break wire;
                }
            };
            let s1 = draw(None);
            let s2 = draw(Some(s1));
            let lits = [(s1, rng.random::<bool>()), (s2, rng.random::<bool>())];
            // Residual constants just complement F — balance is unaffected.
            emit_g57_form(band_wire, &lits, rng, out);
            for s in [s1, s2] {
                match band[..index].iter().position(|&w| w == s) {
                    Some(earlier) => support.extend(supports[earlier].iter().copied()),
                    None => {
                        support.insert(s as usize);
                    }
                }
            }
        }
        supports.push(support);
    }
}

/// One uniform draw among the legacy nonlinear g57 RG networks — RG1
/// value-swap (deg 3), RG2 re-pair (deg 2), RG3 cross-value mask refresh
/// (deg 2) — emitted as XGates. The sharing-state bookkeeping is identical to
/// the linear `emit_rg1_x`/`emit_rg2_x` variants, so the decode bookend is
/// unaffected by which family produced the final pairing. Every RG preserves
/// all logical values (it only relocates/refreshes carriers), so the
/// value-sourced deferred masks are invariant under it and need no
/// interception here.
fn emit_nonlinear_rg(
    state: &mut GadgetState,
    pair_queue: &mut VecDeque<(usize, usize)>,
    single_queue: &mut VecDeque<usize>,
    prod: &mut ProdLedger,
    out: &mut Vec<XGate>,
    rng: &mut impl Rng,
) {
    let n = state.n;
    let total = 2 * n;
    let mut buf: Vec<[u16; 3]> = Vec::new();
    // Every RG network overwrites the carriers it touches (RG1 and RG2 target
    // all four wires of the two pairs, RG3 both carriers of one value), so
    // under distributed sourcing each of those wires is released first: any
    // mask naming one is re-sourced while its bit still means what the ledger
    // says. RG1/RG2 additionally re-pair, and releasing all four wires is what
    // keeps invariant S true across the re-pairing.
    match rng.random_range(0..3u32) {
        0 => {
            let (i, j) = next_pair(pair_queue, n, rng);
            prod.release(&rg_pair_wires(state, i, j), state, rng, out);
            emit_rg1(state, i, j, &mut buf);
        }
        1 => {
            let (i, j) = next_pair(pair_queue, n, rng);
            prod.release(&rg_pair_wires(state, i, j), state, rng, out);
            emit_rg2(state, i, j, &mut buf);
        }
        _ => {
            let i = next_single(single_queue, n, rng);
            let (s, p) = state.pairs[i];
            prod.release(&[s, p], state, rng, out);
            let (s, p) = state.pairs[i];
            let r1 = random_wire_except(total, &[s, p], rng);
            let r2 = random_wire_except(total, &[s, p, r1], rng);
            emit_rg3(state, i, r1, r2, &mut buf);
        }
    }
    out.extend(buf.into_iter().map(XGate::from_g57));
}

/// The four carriers an RG1/RG2 network rewrites.
fn rg_pair_wires(state: &GadgetState, i: usize, j: usize) -> [usize; 4] {
    let (a, b) = state.pairs[i];
    let (c, d) = state.pairs[j];
    [a, b, c, d]
}

/// One randomized-insertion pass over `order`. Each gate, taken in the
/// current order, is inserted uniformly at random among the legal slots of
/// the output built so far: it may hop left over exactly the maximal suffix
/// of gates it commutes with, where commutation is decided by
/// [`XGate::collides`] — gates commute unless proven otherwise, including
/// across a read/write crossing when a shared control with opposite
/// polarities makes the firing supports disjoint. Every hop is an adjacent
/// commuting swap, so the pass preserves the function exactly.
fn insertion_pass(order: &mut Vec<u32>, gates: &[XGate], rng: &mut impl Rng) {
    insertion_pass_by(order, gates, rng, |a, b| XGate::collides(a, b));
}

fn insertion_pass_by(
    order: &mut Vec<u32>,
    gates: &[XGate],
    rng: &mut impl Rng,
    collide: impl Fn(&XGate, &XGate) -> bool,
) {
    let mut out: Vec<u32> = Vec::with_capacity(order.len());
    for &gi in order.iter() {
        let g = &gates[gi as usize];
        let mut span = 0usize;
        while span < out.len() && !collide(g, &gates[out[out.len() - 1 - span] as usize]) {
            span += 1;
        }
        let pos = out.len() - rng.random_range(0..=span);
        out.insert(pos, gi);
    }
    *order = out;
}

/// Rerandomize the order of commuting gates, preserving the function
/// exactly. The only constraint kept is the relative order of every pair
/// that actually collides per [`XGate::collides`]; everything else is fair
/// game — equal-target XOR toggles, disjoint gates, shared reads, and
/// crossing pairs separated by an opposite-polarity shared control all
/// reorder freely. Unlike adjacent-swap churn the relocation is global: a
/// gate can land anywhere between its nearest colliding predecessor and
/// successor, so bookend, W_i, and slice-block material migrates deep into
/// the body, dissolving the construction-time block layout.
///
/// Implementation: alternating-direction randomized insertion passes (a
/// leftward pass gives every gate its full backward reach, the reversed
/// pass the forward reach; `collides` is symmetric and only the relative
/// order of colliding pairs matters, so working on the reversed index list
/// is sound).
pub fn commuting_shuffle(gates: &mut Vec<XGate>, rng: &mut impl Rng) {
    commuting_shuffle_order(gates, rng);
}

/// [`commuting_shuffle`] with one extra constraint: two writes to the SAME
/// wire keep their emission order (they commute as XOR updates, so the
/// standard shuffle scatters them freely). The swap-refresh redesign leans on
/// per-wire write order — the fresh mask monomial is placed strictly interior
/// to its fold's fragment stream, so no contiguous window of one carrier's
/// writes XORs to a clean operand decode — and this variant is what makes
/// that placement survive the reorder exactly instead of probabilistically.
/// Cross-wire mobility (what dissolves the construction-time block layout) is
/// untouched.
/// Peephole cleanup: cancel identical gate pairs that can be commuted
/// adjacent (no gate in between writes a wire the pair reads, or reads or
/// writes its target). Such pairs are pure emission waste — coinciding
/// residue CNOTs from independent spelling draws — measured at ~2.1% of a
/// production build, all single-control. The construction's DELIBERATE
/// redundancy (mask inject/strip pairs) is immune: its two halves always
/// have colliding readers between them — the folds that compensate the
/// mask — so a function-preserving pass cannot touch it. Iterates to a
/// fixpoint; function-preservation is exact (only provably commutable
/// identical pairs are removed).
pub fn cancel_identical_pairs(gates: &mut Vec<XGate>) -> usize {
    const WINDOW: usize = 60;
    let mut removed_total = 0;
    loop {
        let n = gates.len();
        let mut drop = vec![false; n];
        for i in 0..n {
            if drop[i] {
                continue;
            }
            let gi = gates[i].clone();
            for j in (i + 1)..(i + 1 + WINDOW).min(n) {
                if !drop[j] && gates[j] == gi {
                    let clear = (i + 1..j).all(|m| {
                        drop[m]
                            || !(gates[m].target == gi.target
                                || gates[m].reads(gi.target)
                                || gi.reads(gates[m].target))
                    });
                    if clear {
                        drop[i] = true;
                        drop[j] = true;
                    }
                    break;
                }
                let gj = &gates[j];
                if !drop[j]
                    && (gj.target == gi.target
                        || gj.reads(gi.target)
                        || gi.reads(gj.target))
                {
                    break;
                }
            }
        }
        let removed = drop.iter().filter(|&&d| d).count();
        if removed == 0 {
            return removed_total;
        }
        removed_total += removed;
        let mut kept = Vec::with_capacity(n - removed);
        for (g, d) in gates.drain(..).zip(drop) {
            if !d {
                kept.push(g);
            }
        }
        *gates = kept;
    }
}

pub fn commuting_shuffle_stable_targets(gates: &mut Vec<XGate>, rng: &mut impl Rng) {
    let m = gates.len();
    if m < 2 {
        return;
    }
    let collide = |a: &XGate, b: &XGate| a.target == b.target || XGate::collides(a, b);
    let mut order: Vec<u32> = (0..m as u32).collect();
    const PASSES: usize = 3;
    for _ in 0..PASSES {
        insertion_pass_by(&mut order, gates, rng, collide);
        order.reverse();
    }
    if PASSES % 2 == 1 {
        order.reverse();
    }
    let mut reordered = Vec::with_capacity(m);
    for &i in &order {
        reordered.push(gates[i as usize].clone());
    }
    *gates = reordered;
}

/// Like [`commuting_shuffle`], but returns the applied order (new position i
/// held old index order[i]) so callers can permute per-gate sidecars (litter
/// ids, origins) identically.
pub fn commuting_shuffle_order(gates: &mut Vec<XGate>, rng: &mut impl Rng) -> Vec<u32> {
    let m = gates.len();
    if m < 2 {
        return (0..m as u32).collect();
    }
    let mut order: Vec<u32> = (0..m as u32).collect();
    const PASSES: usize = 3;
    for _ in 0..PASSES {
        insertion_pass(&mut order, gates, rng);
        order.reverse();
    }
    if PASSES % 2 == 1 {
        order.reverse();
    }
    let mut reordered = Vec::with_capacity(m);
    for &i in &order {
        reordered.push(gates[i as usize].clone());
    }
    *gates = reordered;
    order
}

/// Bounded linear-time reorder used by the terminal-fence measurement
/// fixture. Each pass considers one parity of disjoint adjacent pairs and
/// swaps a fair random subset of the pairs that commute. Applying this to a
/// prefix slice makes crossing the slice boundary impossible by construction.
fn adjacent_commuting_swap_passes(gates: &mut [XGate], passes: usize, rng: &mut impl Rng) {
    for _ in 0..passes {
        let mut index = rng.random_range(0..2);
        while index + 1 < gates.len() {
            if rng.random_bool(0.5) && !XGate::collides(&gates[index], &gates[index + 1]) {
                gates.swap(index, index + 1);
            }
            index += 2;
        }
    }
}

/// Two-share gadgetization with native CNOT linear bookends, a per-gate
/// uniform draw from the seven-variant CG menu ([`emit_cg_variant`]), and
/// the legacy NONLINEAR g57 RGs drawn uniformly from
/// {RG1 value-swap, RG2 re-pair, RG3 mask-refresh} — `rg_freq` of them
/// (default 1) between consecutive SGs — plus the deferred-mask encoding
/// (RG4, per `masks`; [`MaskConfig::off`] reproduces the unmasked gadget).
/// The whole output is finished with a [`commuting_shuffle`] so W_i,
/// bookend, and body gates interleave wherever wire dependencies allow.
/// Gadgetize with a SINGLE-CARRIER decode: `v = c_v ^ (mask terms) ^ κ`.
///
/// The two-carrier build spends a wire per value on a linear share that an
/// affine adversary gets for free — it simply XORs both carriers in, so the
/// second one contributes nothing to the piling-up product, which runs over
/// the nonlinear terms alone. One linear term is all the balance obstruction
/// requires: flipping `c_v` flips the decode, so the representation classes
/// are equal for any masks.
///
/// Dropping it makes the construction strictly smaller in three ways: `n`
/// carriers instead of `2n`, `(1 + k)^arity` fold fragments instead of
/// `(2 + k)^arity`, and no `W_i` sharing ramp at all — a value simply sits on
/// its wire and is masked in place. Spending the saved atom on a degree-2 mask
/// (`[1,2,3,3]`) is cost-neutral against the shipped `[1,1,3,3]` and hides
/// strictly better (0.641 vs 0.781 best-predictor agreement).
///
/// What changes elsewhere, and is NOT free:
/// * one probe now reads `v ^ masks ^ κ`, correlated with the value at the
///   piling-up rate, where a two-carrier share is uniform AND independent;
/// * RG2 (re-pair) and RG3 (refresh both carriers) have no analogue here —
///   there is no pair to re-pair, and XORing into a lone carrier changes the
///   value. Only RG1's relocation survives, so churn is relocation plus mask
///   re-sourcing.
pub fn gadgetize_cnot_single(
    main: &CircuitSeq,
    n: usize,
    rg_freq: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    assert!(n >= 3, "gadgetize_cnot_single requires n >= 3");
    assert!(
        prod.single_carrier(),
        "gadgetize_cnot_single needs --prod-single with a nonempty mask plan"
    );
    assert!(
        !prod.dist(),
        "distributed sourcing and single-carrier are not combined yet"
    );
    // With one carrier there is no RG3 to refresh the representation: a
    // re-source is the ONLY move that changes a carrier's bit. rsrc = 0 leaves
    // relocation as the sole churn, which moves values without re-randomising
    // them -- documented as load-bearing, so enforce it rather than trust it.
    assert!(
        prod.rsrc >= 1,
        "--prod-single needs --prod-rsrc >= 1: with a single carrier, mask \
         re-sourcing is the only representation refresh (R2/R3 have no analogue)"
    );

    let mut main = main.clone();
    let rounds = main.gates.len();
    shoot_random_gate(&mut main, rounds);

    let carrier_total = n;
    let band_len = prod.band_size(n);
    let total = carrier_total + band_len;
    assert!(total <= u16::MAX as usize, "too many wires");

    // The band is filled from the data wires while they still hold x, and is
    // not written again by the ordinary body path — the same source contract as
    // the two-carrier build. (With --prod-epoch > 0 a retire-and-refill DOES
    // rewrite a band wire mid-body; that is the point of epochs, and the
    // release-then-rewrite order is what keeps it exact.)
    // build, and the reason the strip cancels exactly under arbitrary junk.
    let mut out: Vec<XGate> = Vec::new();
    let band_home: Vec<u16> = (carrier_total..total).map(|w| w as u16).collect();
    if prod.fill_nl > 0 {
        emit_band_fill_nl_pivots(
            n,
            &band_home,
            prod.fill_nl,
            prod.fill_pivots > 0,
            rng,
            &mut out,
        );
    } else {
        emit_band_fill(n, &band_home, rng, &mut out);
    }

    // Value v lives on wire v; `pairs` records only where, with both entries
    // equal so every carrier lookup returns the one wire.
    let mut state = GadgetState {
        n,
        pairs: (0..n).map(|w| (w, w)).collect(),
    };
    let mut prod_ledger = ProdLedger::new(n, prod, carrier_total, None);
    prod_ledger.inject_all(&state, rng, &mut out);

    for (index, &gate) in main.gates.iter().enumerate() {
        prod_ledger.set_pos(index);
        prod_ledger.fold_cg(&XGate::from_g57(gate), &state, rng, &mut out);
        if index + 1 == main.gates.len() {
            break;
        }
        // The only surviving RG: relocate a value to another wire. It is
        // RG1's move without the pair — a content swap plus a re-point — and
        // it keeps every decode invariant because the masks name band
        // variables, not carriers.
        for _ in 0..rg_freq {
            emit_value_relocation(&mut state, carrier_total, &mut out, rng);
        }
        for _ in 0..prod.rsrc {
            prod_ledger.resource(&state, rng, &mut out);
        }
        // Roll the band across the carrier/band boundary. This matters MORE
        // here than in the paired build, not less: the band is a larger share
        // of a narrower gadget, and with one carrier per value each carrier
        // absorbs every fold write, so an unwritten band wire stands out
        // against a sharper contrast.
        for _ in 0..prod.roll {
            prod_ledger.roll(&mut state, rng, &mut out);
        }
        // Fire with probability 1/epoch rather than on every epoch-th gate:
        // the same expected rate, without a period an attacker can lock onto.
        if prod.epoch > 0 && rng.random_range(0..prod.epoch) == 0 {
            prod_ledger.retire_refill(&state, prod.refill_data, prod.fill_nl, rng, &mut out);
        }
    }
    prod_ledger.strip_all(&state, rng, &mut out);
    prod_ledger.report();

    // Relocations and rolls permuted the values across the WHOLE wire space —
    // after a roll a value can sit on a former band wire. Route every value
    // home by cycle resolution, so wires 0..n hold the values and n..total
    // hold band junk, which is what makes the mirror fill below safe.
    let mut owner: Vec<Option<usize>> = vec![None; total];
    for value in 0..n {
        owner[state.pairs[value].0] = Some(value);
    }
    for value in 0..n {
        let cur = state.pairs[value].0;
        if cur == value {
            continue;
        }
        emit_wire_swap(cur, value, &mut out);
        let displaced = owner[value];
        owner[cur] = displaced;
        if let Some(u) = displaced {
            state.pairs[u] = (cur, cur);
        }
        owner[value] = Some(value);
        state.pairs[value] = (value, value);
    }

    // Mirror fill so the band is junk at both ports, as in the paired build.
    let band_final: Vec<u16> = (carrier_total..total).map(|w| w as u16).collect();
    if prod.fill_nl > 0 {
        emit_band_fill_nl_pivots(
            n,
            &band_final,
            prod.fill_nl,
            prod.fill_pivots > 0,
            rng,
            &mut out,
        );
    } else {
        emit_band_fill(n, &band_final, rng, &mut out);
    }

    commuting_shuffle(&mut out, rng);
    CnotCircuit {
        gates: out,
        num_wires: total,
    }
}

/// Swap the contents of two wires with the three-CNOT network.
fn emit_wire_swap(a: usize, b: usize, out: &mut Vec<XGate>) {
    for (t, c) in [(a, b), (b, a), (a, b)] {
        out.push(XGate::cnot(t as u16, c as u16));
    }
}

/// Aux-controlled swap ("switch") of two data wires: swaps the contents of
/// `a` and `b` iff the control literal fires, else identity. The three-gate
/// Fredkin network
///     b ^= a ;  a ^= (ctrl AND b) ;  b ^= a
/// only READS `ctrl`, and conserves the pair-sum `a ^ b` for every control
/// value -- that conserved sum is a value-agnostic fabric invariant (it says
/// nothing about which value sits on which wire), not a leak. `ctrl_pol`
/// selects a positive (true) or negated (false) control literal.
///
/// A value carried through L successive switches on DISTINCT pairs becomes a
/// depth-L decision-tree leaf, so recovering it costs ANF degree L+1; only a
/// degenerate schedule that keeps re-swapping the SAME pair collapses to
/// degree 2. Callers must therefore route each value across fresh pairs.
#[allow(dead_code)]
fn emit_cswap(ctrl: usize, ctrl_pol: bool, a: usize, b: usize, out: &mut Vec<XGate>) {
    debug_assert!(
        a != b && ctrl != a && ctrl != b,
        "a switch needs three distinct wires (two data operands + one aux control)"
    );
    out.push(XGate::cnot(b as u16, a as u16));
    out.push(
        XGate::conj(a as u16, [(b as u16, true), (ctrl as u16, ctrl_pol)])
            .expect("switch middle-gate literals are distinct"),
    );
    out.push(XGate::cnot(b as u16, a as u16));
}

/// Signed switch: `emit_cswap` followed by an aux-controlled NOT on wire `a`
/// (`a ^= sign` when the sign literal fires), realizing a hyperoctahedral
/// relocate-and-maybe-complement move rather than a pure wire permutation. The
/// extra term is degree-1, so it does NOT raise the routing recovery degree;
/// its only job is to erase the value-agnostic couple-sum / component-parity
/// invariants a pure-swap network leaves behind. Values then decode only up to
/// the known aux-dependent sign, so callers must track the accumulated mask.
#[allow(dead_code)]
fn emit_signed_cswap(
    ctrl: usize,
    ctrl_pol: bool,
    a: usize,
    b: usize,
    sign: usize,
    sign_pol: bool,
    out: &mut Vec<XGate>,
) {
    emit_cswap(ctrl, ctrl_pol, a, b, out);
    out.push(
        XGate::conj(a as u16, [(sign as u16, sign_pol)])
            .expect("sign literal is distinct from data wire a"),
    );
}

// ---------------------------------------------------------------------------
// Drip discipline: plain compute + bounded-depth aux-controlled routing.
//
// Replaces the masked compute phase. Each original gate is fired on operands
// first routed to depth `k` by aux-controlled switches (so every operand is
// recoverable only at ANF degree k+1 at its firing cut), then routed back. A
// value at depth d occupies 2^d wires, and 2n wires cannot hold all n values at
// depth >= 1 at once (n*2^d > 2n for d >= 1), so this variant deepens only the
// current gate's operands transiently and leaves values on their home wires
// between firings. Its gate count (~2^(arity*k) per gate) is the size the
// experiment asks for; between-firing secrecy is what masks would supply.
// ---------------------------------------------------------------------------

/// Raise the value on `home` to routing depth `k`: a genuine decision tree with
/// one shared control wire per level and a fresh partner wire per leaf. Emits
/// the switch gates, logs each for the reverse-replay lower, and returns the
/// leaf set `(wire, guard)` where `guard` is the routing literals selecting it.
fn drip_raise(
    home: u16,
    k: usize,
    cs: &[u16],
    partners: &[u16],
    out: &mut Vec<XGate>,
    log: &mut Vec<(u16, u16, u16)>,
) -> Vec<(u16, Vec<(u16, bool)>)> {
    let mut leaves: Vec<(u16, Vec<(u16, bool)>)> = vec![(home, Vec::new())];
    let mut part_idx = 0usize;
    for &ctrl in cs.iter().take(k) {
        let mut next: Vec<(u16, Vec<(u16, bool)>)> = Vec::with_capacity(leaves.len() * 2);
        for (w, guard) in &leaves {
            let part = partners[part_idx];
            part_idx += 1;
            emit_cswap(ctrl as usize, true, *w as usize, part as usize, out);
            log.push((ctrl, *w, part));
            let mut g_idle = guard.clone();
            g_idle.push((ctrl, false)); // idle branch: control did NOT fire
            next.push((*w, g_idle));
            let mut g_fired = guard.clone();
            g_fired.push((ctrl, true)); // fired branch: value moved to partner
            next.push((part, g_fired));
        }
        leaves = next;
    }
    leaves
}

/// Fire one original gate on operands routed to depth `k`, then route them back.
fn drip_fire_gate(gate: &XGate, n: usize, total: usize, k: usize, out: &mut Vec<XGate>) {
    let per_cs = k;
    let per_part = (1usize << k) - 1;
    let num_ops = 1 + gate.ctrls.len();
    debug_assert!(
        (total - n) >= (per_cs + per_part) * num_ops,
        "band too small for drip depth k"
    );

    let grab = |count: usize, nb: &mut u16| -> Vec<u16> {
        let v: Vec<u16> = (0..count).map(|i| *nb + i as u16).collect();
        *nb += count as u16;
        v
    };

    let mut nb = n as u16;
    let mut log: Vec<(u16, u16, u16)> = Vec::new();

    // Raise the target.
    let t_cs = grab(per_cs, &mut nb);
    let t_part = grab(per_part, &mut nb);
    let t_leaves = drip_raise(gate.target, k, &t_cs, &t_part, out, &mut log);

    // Raise each control (distinct scratch wires => guards never contradict).
    let mut c_leaves: Vec<(Vec<(u16, Vec<(u16, bool)>)>, bool)> = Vec::new();
    for &(cw, pol) in gate.ctrls.iter() {
        let cs = grab(per_cs, &mut nb);
        let part = grab(per_part, &mut nb);
        let leaves = drip_raise(cw, k, &cs, &part, out, &mut log);
        c_leaves.push((leaves, pol));
    }

    // Multiplex-fire: one guarded copy per (target leaf) x (each control leaf).
    let sizes: Vec<usize> = c_leaves.iter().map(|(l, _)| l.len()).collect();
    let combos: usize = sizes.iter().product::<usize>().max(1);
    for (t_wire, t_guard) in &t_leaves {
        for combo in 0..combos {
            let mut guard = t_guard.clone();
            let mut mono: Vec<(u16, bool)> = Vec::new();
            let mut rem = combo;
            for (i, (leaves_i, pol_i)) in c_leaves.iter().enumerate() {
                let idx = rem % sizes[i];
                rem /= sizes[i];
                let (ci_wire, ci_guard) = &leaves_i[idx];
                guard.extend_from_slice(ci_guard);
                mono.push((*ci_wire, *pol_i));
            }
            if gate.comp {
                // g57 form: t ^= guard ^ (guard AND mono). Two conjunctions.
                if let Some(g) = XGate::conj(*t_wire, guard.iter().copied()) {
                    out.push(g);
                }
                let both: Vec<(u16, bool)> = guard.iter().chain(mono.iter()).copied().collect();
                if let Some(g) = XGate::conj(*t_wire, both) {
                    out.push(g);
                }
            } else {
                let both: Vec<(u16, bool)> = guard.iter().chain(mono.iter()).copied().collect();
                if let Some(g) = XGate::conj(*t_wire, both) {
                    out.push(g);
                }
            }
        }
    }

    // Lower: replay every switch in reverse (self-inverse; controls unchanged).
    for &(ctrl, w, part) in log.iter().rev() {
        emit_cswap(ctrl as usize, true, w as usize, part as usize, out);
    }
}

/// Regional bounded-depth drip: gates whose index falls in `sensitive` are
/// routed to depth `k_hi`, the rest to `k_lo`. Because each gate independently
/// raises, fires, and lowers its own operands, the depth can vary gate-by-gate
/// with no cross-gate propagation. Use it to protect a sensitive band (e.g. the
/// sandwich's middle N column) at a higher degree floor while keeping the bulk
/// cheap. Data values live on `0..n`, band on `n..2n`.
pub fn gadgetize_drip_regional(
    source: &[XGate],
    n: usize,
    k_lo: usize,
    k_hi: usize,
    sensitive: std::ops::Range<usize>,
    rng: &mut impl Rng,
) -> CnotCircuit {
    assert!(k_lo >= 1 && k_hi >= 1, "drip depth must be >= 1");
    let total = 2 * n;
    assert!(total <= u16::MAX as usize, "too many wires");
    let k_max = k_lo.max(k_hi);
    let per_op = k_max + ((1usize << k_max) - 1);
    let max_ops = 1 + source.iter().map(|g| g.ctrls.len()).max().unwrap_or(0);
    assert!(
        n >= per_op * max_ops,
        "band ({n}) too small for drip k={k_max}: need {} scratch wires",
        per_op * max_ops
    );

    let mut out: Vec<XGate> = Vec::new();
    let band_home: Vec<u16> = (n..total).map(|w| w as u16).collect();
    emit_band_fill_src(n, &band_home, rng, &mut out);

    for (i, gate) in source.iter().enumerate() {
        let k = if sensitive.contains(&i) { k_hi } else { k_lo };
        drip_fire_gate(gate, n, total, k, &mut out);
    }

    CnotCircuit {
        gates: out,
        num_wires: total,
    }
}

/// Uniform-depth drip gadgetizer (degree floor `k+1` everywhere).
pub fn gadgetize_drip_single(
    source: &[XGate],
    n: usize,
    k: usize,
    rng: &mut impl Rng,
) -> CnotCircuit {
    gadgetize_drip_regional(source, n, k, k, 0..0, rng)
}

// ---------------------------------------------------------------------------
// Layered flat-drip (route-then-fire) — the g57-native part-3 discipline.
//
// Bunch the source into wire-disjoint layers; within each layer route each
// gate TYPE-class among its own slots via the in-alphabet NSWITCH, fire the
// identical class gate positionally (no multiplex — the whole 2^(a*k) blowup is
// gone), then route back (Home mode: pi^-1 . FIRE . pi = FIRE, correct for all
// Z). Everything emitted is native g57/CNOT. Aux rerandomization (data-ctrl +
// aux-ctrl -> aux-target g57s) is sprinkled between units, count = a parameter.
// See docs/FLAT_DRIP_RESHUFFLE.md and [[flat-drip-reshuffle-20260829]].
// ---------------------------------------------------------------------------

/// In-alphabet 3-gate controlled swap (NSWITCH): swaps wires p,q iff z=0, with
/// an UNCONDITIONAL complement of both (constant sign -> cancels over even
/// matching layers). Reads aux z, never writes it. Self-inverse for fixed z.
fn emit_nswitch(p: usize, z: usize, q: usize, out: &mut Vec<XGate>) {
    debug_assert!(p != q && z != p && z != q, "NSWITCH needs three distinct wires");
    out.push(XGate::cnot(p as u16, q as u16)); // G1: p ^= q
    // G2: q ^= 1 ^ (p AND NOT z) = q ^= z OR NOT p
    out.push(XGate::from_g57([q as u16, z as u16, p as u16]));
    out.push(XGate::cnot(p as u16, q as u16)); // G3: p ^= q
}

/// Type-class signature: gates with the same (comp, sorted control polarities)
/// are identical-form and may be routed among one another.
fn class_signature(g: &XGate) -> (bool, Vec<bool>) {
    let mut pol: Vec<bool> = g.ctrls.iter().map(|&(_, p)| p).collect();
    pol.sort_unstable();
    (g.comp, pol)
}

/// Canonical rail vector `[target, controls ordered by (polarity, wire)]`. Rail
/// role r is the same across every gate of a class, so an NSWITCH on rail r of
/// two buses moves the operands with roles intact.
fn rail_vector(g: &XGate) -> Vec<u16> {
    let mut c: Vec<(u16, bool)> = g.ctrls.iter().copied().collect();
    c.sort_by_key(|&(w, p)| (p, w));
    let mut v = vec![g.target];
    v.extend(c.iter().map(|&(w, _)| w));
    v
}

/// Emit the class gate on the given rails (positional FIRE).
fn emit_fire(rails: &[u16], comp: bool, pols: &[bool], out: &mut Vec<XGate>) {
    let lits = rails[1..].iter().zip(pols.iter()).map(|(&w, &p)| (w, p));
    let mut g = XGate::conj(rails[0], lits).expect("fire literals are consistent");
    g.comp = comp;
    out.push(g);
}

/// Route (2 matchings) -> FIRE positionally -> route back (reverse). Even
/// matchings cancel the NSWITCH complement; reverse-replay inverts the routing,
/// so the net data map is exactly FIRE-in-place regardless of the aux controls.
/// `buses` must have even length >= 2.
fn route_fire_layer_class(
    buses: &[Vec<u16>],
    comp: bool,
    pols: &[bool],
    bits: &[u16],
    out: &mut Vec<XGate>,
) {
    let k = buses.len();
    let r = buses[0].len();
    let m1: Vec<(usize, usize)> = (0..k / 2).map(|i| (2 * i, 2 * i + 1)).collect();
    let m2: Vec<(usize, usize)> = (0..k / 2).map(|i| (2 * i + 1, (2 * i + 2) % k)).collect();
    let mut fwd: Vec<(usize, usize, usize)> = Vec::new();
    let mut bi = 0usize;
    for matching in [&m1, &m2] {
        for &(i, j) in matching {
            let z = bits[bi % bits.len()] as usize;
            bi += 1;
            for rr in 0..r {
                let (p, q) = (buses[i][rr] as usize, buses[j][rr] as usize);
                emit_nswitch(p, z, q, out);
                fwd.push((p, z, q));
            }
        }
    }
    for bus in buses {
        emit_fire(bus, comp, pols, out);
    }
    for &(p, z, q) in fwd.iter().rev() {
        emit_nswitch(p, z, q, out);
    }
}

/// Greedy wire-disjoint, dependency-respecting layering (ASAP list schedule);
/// maximizes gates per layer. Returns index-lists into `source`.
fn layer_wire_disjoint(source: &[XGate], total_wires: usize) -> Vec<Vec<usize>> {
    let wires_of = |g: &XGate| -> Vec<u16> {
        let mut v: Vec<u16> = g.ctrls.iter().map(|&(w, _)| w).collect();
        v.push(g.target);
        v
    };
    let mut last_touch = vec![usize::MAX; total_wires];
    let mut occupied: Vec<Vec<bool>> = Vec::new();
    let mut layers: Vec<Vec<usize>> = Vec::new();
    for (gi, g) in source.iter().enumerate() {
        let ws = wires_of(g);
        let mut lo = 0usize;
        for &w in &ws {
            let t = last_touch[w as usize];
            if t != usize::MAX {
                lo = lo.max(t + 1);
            }
        }
        let mut placed = None;
        for l in lo..layers.len() {
            if ws.iter().all(|&w| !occupied[l][w as usize]) {
                placed = Some(l);
                break;
            }
        }
        let l = placed.unwrap_or_else(|| {
            occupied.push(vec![false; total_wires]);
            layers.push(Vec::new());
            layers.len() - 1
        });
        for &w in &ws {
            occupied[l][w as usize] = true;
            last_touch[w as usize] = l;
        }
        layers[l].push(gi);
    }
    layers
}

fn gate_wires(g: &XGate) -> Vec<u16> {
    let mut v: Vec<u16> = g.ctrls.iter().map(|&(w, _)| w).collect();
    v.push(g.target);
    v
}

/// N-column bridge gate signature: a positive CNOT `y_i ^= x_i` with
/// `target = control + half` (target in the high half, control in the low half).
fn is_n_column(g: &XGate, half: usize) -> bool {
    !g.comp
        && g.ctrls.len() == 1
        && g.ctrls[0].1
        && (g.target as usize) >= half
        && (g.ctrls[0].0 as usize) + half == g.target as usize
}

/// Wire-disjoint dependency-respecting layering that CONTRACTS all `in_group`
/// gates into one scheduling node, forcing them into a single layer (they must
/// be mutually wire-disjoint and form a convex set of the dependency order — the
/// N-column does). Two passes: pass 1 finds the group's common layer `gnl`; pass
/// 2 pins the group there so its consumers land strictly after it.
fn layer_wire_disjoint_grouped(
    source: &[XGate],
    total: usize,
    in_group: &[bool],
) -> Vec<Vec<usize>> {
    use std::collections::HashMap;
    const GNODE: usize = usize::MAX - 1;
    let node_of = |gi: usize| if in_group[gi] { GNODE } else { gi };

    // Pass 1: compute the group node's final layer.
    let mut node_layer: HashMap<usize, usize> = HashMap::new();
    let mut last_node = vec![usize::MAX; total];
    for (gi, g) in source.iter().enumerate() {
        let node = node_of(gi);
        let mut floor = 0usize;
        for &w in gate_wires(g).iter() {
            let ln = last_node[w as usize];
            if ln != usize::MAX && ln != node {
                floor = floor.max(node_layer.get(&ln).copied().unwrap_or(0) + 1);
            }
        }
        let e = node_layer.entry(node).or_insert(0);
        *e = (*e).max(floor);
        for &w in gate_wires(g).iter() {
            last_node[w as usize] = node;
        }
    }
    let gnl = node_layer.get(&GNODE).copied().unwrap_or(0);

    // Pass 2: pin GNODE at gnl; compute each gate's layer.
    node_layer.clear();
    node_layer.insert(GNODE, gnl);
    for v in last_node.iter_mut() {
        *v = usize::MAX;
    }
    let mut gate_layer = vec![0usize; source.len()];
    for (gi, g) in source.iter().enumerate() {
        let node = node_of(gi);
        let mut floor = 0usize;
        for &w in gate_wires(g).iter() {
            let ln = last_node[w as usize];
            if ln != usize::MAX && ln != node {
                floor = floor.max(node_layer.get(&ln).copied().unwrap_or(0) + 1);
            }
        }
        let layer = if node == GNODE {
            gnl
        } else {
            let e = node_layer.entry(node).or_insert(0);
            *e = (*e).max(floor);
            *e
        };
        gate_layer[gi] = layer;
        for &w in gate_wires(g).iter() {
            last_node[w as usize] = node;
        }
    }

    let maxl = *gate_layer.iter().max().unwrap_or(&0);
    let mut layers = vec![Vec::new(); maxl + 1];
    for gi in 0..source.len() {
        layers[gate_layer[gi]].push(gi);
    }
    layers.retain(|l| !l.is_empty());
    layers
}

/// g57-native layered route-then-fire gadgetizer. Data `0..n`, aux `n..2n`.
/// `n_rerand` aux-rerandomization g57s (data-ctrl + aux-ctrl -> aux-target) are
/// sprinkled after each layer.
pub fn gadgetize_drip_layered(
    source: &[XGate],
    n: usize,
    n_rerand: usize,
    rng: &mut impl Rng,
) -> CnotCircuit {
    use std::collections::HashMap;
    let total = 2 * n;
    assert!(total <= u16::MAX as usize, "too many wires");
    let mut out: Vec<XGate> = Vec::new();

    let band_home: Vec<u16> = (n..total).map(|w| w as u16).collect();
    emit_band_fill_src(n, &band_home, rng, &mut out);

    // Group the N-column (the wire-disjoint y_i ^= x_i bridge, half = n/2) into
    // one class: it is the one big persistent-hiding orbit and packs 128 gates.
    // Greedy wire-disjoint filter so a false-positive signature match can never
    // put two wire-sharing gates in the forced common layer.
    let half = n / 2;
    let mut in_group = vec![false; source.len()];
    {
        let mut used = vec![false; total];
        for (gi, g) in source.iter().enumerate() {
            if is_n_column(g, half) {
                let ws = gate_wires(g);
                if ws.iter().all(|&w| !used[w as usize]) {
                    in_group[gi] = true;
                    for &w in &ws {
                        used[w as usize] = true;
                    }
                }
            }
        }
    }
    // NOTE: N-column grouping (layer_wire_disjoint_grouped) is UNSOUND here — the
    // N-column is not convex in the dependency order (slice gates read high wires
    // and write low wires, coupling N-inputs to N-outputs), so contracting all N
    // into one layer violates dependencies. Use the correct plain ASAP layering;
    // a convexity-safe N-grouping is a future optimization.
    let _ = &in_group;
    let layers = layer_wire_disjoint(source, total);
    let aux_pool: Vec<u16> = (n..total).map(|w| w as u16).collect();
    let mut aux_ptr = 0usize;

    for layer in &layers {
        let mut classes: HashMap<(bool, Vec<bool>), Vec<usize>> = HashMap::new();
        for &gi in layer {
            classes
                .entry(class_signature(&source[gi]))
                .or_default()
                .push(gi);
        }
        let mut keys: Vec<_> = classes.keys().cloned().collect();
        keys.sort();
        for key in keys {
            let (comp, pols) = &key;
            let mut idxs = classes[&key].clone();
            // Even count for the matching parity; odd -> fire one in place.
            let singleton = if idxs.len() % 2 == 1 { idxs.pop() } else { None };
            if idxs.len() >= 2 {
                let buses: Vec<Vec<u16>> = idxs.iter().map(|&gi| rail_vector(&source[gi])).collect();
                let k = buses.len();
                let bits: Vec<u16> = (0..k)
                    .map(|_| {
                        let b = aux_pool[aux_ptr % aux_pool.len()];
                        aux_ptr += 1;
                        b
                    })
                    .collect();
                route_fire_layer_class(&buses, *comp, pols, &bits, &mut out);
            }
            if let Some(gi) = singleton {
                out.push(source[gi].clone());
            }
        }
        for _ in 0..n_rerand {
            let d = rng.random_range(0..n) as u16;
            let ac = (n + rng.random_range(0..n)) as u16;
            let at = (n + rng.random_range(0..n)) as u16;
            if at != ac && at as usize != d as usize && ac != d {
                out.push(XGate::from_g57([at, d, ac]));
            }
        }
    }

    CnotCircuit {
        gates: out,
        num_wires: total,
    }
}

/// Persist-mode drip: each value is kept DISPLACED onto an aux scratch wire
/// (aux-controlled NSWITCH) between its uses, and fetched home only for the
/// instant its own gate fires. No global route-back — values sit on
/// aux-dependent wires (degree-1-hidden) everywhere except their own use sites.
/// Displace+fetch is a matched NSWITCH pair, so the complement cancels. Data
/// `0..n`, aux `n..2n` (controls in `n..n+n/2`, scratch in `n+n/2..2n`).
pub fn gadgetize_drip_persist(source: &[XGate], n: usize, rng: &mut impl Rng) -> CnotCircuit {
    let total = 2 * n;
    assert!(n >= 4, "persist needs n >= 4");
    let mut out: Vec<XGate> = Vec::new();
    let band_home: Vec<u16> = (n..total).map(|w| w as u16).collect();
    emit_band_fill_src(n, &band_home, rng, &mut out);

    let (ctrl_lo, ctrl_hi) = (n, n + n / 8); // few stable controls
    let mut free_scratch: Vec<usize> = (n + n / 8..total).collect(); // most of aux = scratch
    let mut disp: Vec<Option<(usize, usize)>> = vec![None; n]; // value -> (scratch, ctrl)
    let mut cptr = 0usize;

    for gate in source {
        let mut ops: Vec<usize> = gate.ctrls.iter().map(|&(c, _)| c as usize).collect();
        ops.push(gate.target as usize);
        // fetch operands home
        for &w in &ops {
            if let Some((s, z)) = disp[w].take() {
                emit_nswitch(w, z, s, &mut out);
                free_scratch.push(s);
            }
        }
        // fire on home wires
        out.push(gate.clone());
        // re-displace operands to hide them again
        for &w in &ops {
            if disp[w].is_none() {
                if let Some(s) = free_scratch.pop() {
                    let z = ctrl_lo + cptr % (ctrl_hi - ctrl_lo);
                    cptr += 1;
                    emit_nswitch(w, z, s, &mut out);
                    disp[w] = Some((s, z));
                }
            }
        }
    }
    // canonicalize output: fetch everything still displaced
    for w in 0..n {
        if let Some((s, z)) = disp[w].take() {
            emit_nswitch(w, z, s, &mut out);
            free_scratch.push(s);
        }
    }

    CnotCircuit {
        gates: out,
        num_wires: total,
    }
}

/// Full 5-part deliverable: opening zero-slice guard (part 1), the drip body
/// (part 2 aux-fill + part 3 compute + junk aux = part 4), and a junk-half
/// closing guard (part 5). Dead at aux input = 0; payload is the sandwich answer
/// on the UPPER data half `n/2..n` (the junk-half closing guard corrupts only
/// `0..n/2`). Part 4 is NOT an inverse of part 2 — the band is junk at both
/// ports, per the production contract. Data `0..n`, aux `n..2n`.
pub fn wrap_drip_delivery(
    source: &[XGate],
    n: usize,
    n_rerand: usize,
    slice_gates: usize,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let nondata = n;
    let gc = slice_gates.max(nondata);
    let mut circuit = slice_zero_junk_guard_dims(n, nondata, gc, rng); // part 1
    let body = gadgetize_drip_layered(source, n, n_rerand, rng); // parts 2+3+4
    circuit.num_wires = circuit.num_wires.max(body.num_wires);
    circuit.gates.extend(body.gates);
    let close = slice_zero_junk_guard_dims(n, nondata, gc, rng); // part 5
    circuit.gates.extend(close.gates);
    circuit
}

#[cfg(test)]
mod drip_tests {
    use super::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    fn apply(gates: &[XGate], state: &mut [bool]) {
        for g in gates {
            let conj = g.ctrls.iter().all(|&(w, pol)| state[w as usize] == pol);
            if g.comp ^ conj {
                state[g.target as usize] ^= true;
            }
        }
    }

    fn random_source(n: usize, m: usize, rng: &mut impl Rng) -> Vec<XGate> {
        (0..m)
            .map(|_| {
                let target = rng.random_range(0..n) as u16;
                let width = rng.random_range(0..3usize); // 0, 1, or 2 controls
                let mut ctrls: Vec<(u16, bool)> = Vec::new();
                let mut used = vec![target];
                for _ in 0..width {
                    loop {
                        let w = rng.random_range(0..n) as u16;
                        if !used.contains(&w) {
                            used.push(w);
                            ctrls.push((w, rng.random_range(0..2) == 1));
                            break;
                        }
                    }
                }
                let comp = rng.random_range(0..2) == 1;
                XGate::conj(target, ctrls.iter().copied())
                    .map(|mut g| {
                        g.comp = comp;
                        g
                    })
                    .unwrap_or_else(|| XGate::x_gate(target))
            })
            .collect()
    }

    #[test]
    fn drip_gadget_computes_c_and_reports_size() {
        for &k in &[1usize, 2] {
            let mut rng = StdRng::seed_from_u64(0xd21b_9c4e ^ k as u64);
            let n = 24;
            let m = 60;
            let source = random_source(n, m, &mut rng);
            let gadget = gadgetize_drip_single(&source, n, k, &mut rng);
            let total = gadget.num_wires;
            for _ in 0..64 {
                let x: Vec<bool> = (0..n).map(|_| rng.random_range(0..2) == 1).collect();
                let mut refs = vec![false; n];
                refs.copy_from_slice(&x);
                apply(&source, &mut refs);
                let mut gs = vec![false; total];
                gs[..n].copy_from_slice(&x);
                apply(&gadget.gates, &mut gs);
                assert_eq!(&gs[..n], &refs[..], "drip k={k} output mismatch");
            }
            eprintln!(
                "[drip] k={k} n={n} |source|={m} -> gadget={} gates ({:.1}x/gate), wires={total}",
                gadget.gates.len(),
                gadget.gates.len() as f64 / m as f64
            );
        }

        // Regional (mixed depth: k=1 baseline, k=2 on a middle window) must
        // also compute C exactly -- each gate raises/fires/lowers on its own.
        let mut rng = StdRng::seed_from_u64(0xbeef_1234);
        let n = 24;
        let m = 60;
        let source = random_source(n, m, &mut rng);
        let gadget = gadgetize_drip_regional(&source, n, 1, 2, 20..40, &mut rng);
        let total = gadget.num_wires;
        for _ in 0..64 {
            let x: Vec<bool> = (0..n).map(|_| rng.random_range(0..2) == 1).collect();
            let mut refs = vec![false; n];
            refs.copy_from_slice(&x);
            apply(&source, &mut refs);
            let mut gs = vec![false; total];
            gs[..n].copy_from_slice(&x);
            apply(&gadget.gates, &mut gs);
            assert_eq!(&gs[..n], &refs[..], "regional drip output mismatch");
        }
    }

    #[test]
    fn drip_layered_computes_c_and_reports_size() {
        for &nr in &[0usize, 4] {
            let mut rng = StdRng::seed_from_u64(0x1a4e_2200 ^ nr as u64);
            let n = 32;
            let m = 150;
            let source = random_source(n, m, &mut rng);
            let gadget = gadgetize_drip_layered(&source, n, nr, &mut rng);
            let total = gadget.num_wires;
            let mut g57 = 0usize;
            for g in &gadget.gates {
                if g.comp && g.ctrls.len() == 2 {
                    g57 += 1;
                }
            }
            for _ in 0..64 {
                let x: Vec<bool> = (0..n).map(|_| rng.random_range(0..2) == 1).collect();
                let mut refs = vec![false; n];
                refs.copy_from_slice(&x);
                apply(&source, &mut refs);
                let mut gs = vec![false; total];
                gs[..n].copy_from_slice(&x);
                apply(&gadget.gates, &mut gs);
                assert_eq!(&gs[..n], &refs[..], "layered drip n_rerand={nr} mismatch");
            }
            let maxw = gadget.gates.iter().map(|g| g.ctrls.len()).max().unwrap_or(0);
            eprintln!(
                "[drip-layered] n={n} |src|={m} n_rerand={nr} -> {} gates ({:.1}x), g57={:.0}%, maxw={maxw}",
                gadget.gates.len(),
                gadget.gates.len() as f64 / m as f64,
                100.0 * g57 as f64 / gadget.gates.len() as f64
            );
        }
    }

    // Two-layer seam prototype: molecules (self-contained 2-wire sub-circuits)
    // routed among isomorphic siblings, fired position-agnostically, with the
    // permutation kept LIVE across the seam (no route-back). Compares against a
    // route-back variant. Exhaustive over all data AND all routing-aux settings.
    #[test]
    fn seam_molecule_prototype() {
        const NMOL: usize = 4;
        const NDATA: usize = 2 * NMOL; // 8
        const ZERO: usize = NDATA; // wire 8 = constant 0 (never set)
        const ABASE: usize = NDATA + 1; // routing bits: wires 9..15
        const NBITS: usize = 6;
        const NTOT: usize = ABASE + NBITS; // 15
        const NVAR: usize = NDATA + NBITS; // 14 varying input bits
        const NPT: usize = 1 << NVAR; // 16384
        const WORDS: usize = NPT / 64;

        // Clean aux-controlled swap of wires p,q iff bit z=1 (CSWITCH4, z2=ZERO).
        fn cswap(p: usize, q: usize, z: usize, out: &mut Vec<XGate>) {
            out.push(XGate::cnot(p as u16, q as u16));
            out.push(XGate::from_g57([q as u16, z as u16, p as u16]));
            out.push(XGate::from_g57([q as u16, ZERO as u16, p as u16]));
            out.push(XGate::cnot(p as u16, q as u16));
        }
        // Swap whole molecules i,j (both rails share the same control bit z).
        fn mol_swap(i: usize, j: usize, z: usize, out: &mut Vec<XGate>) {
            cswap(2 * i, 2 * j, z, out);
            cswap(2 * i + 1, 2 * j + 1, z, out);
        }
        // A connected route stage over 4 molecules using 3 control bits.
        fn route(bits: [usize; 3], out: &mut Vec<XGate>) {
            let swaps = [(0usize, 1usize, bits[0]), (2, 3, bits[1]), (1, 2, bits[2])];
            for &(i, j, z) in &swaps {
                mol_swap(i, j, z, out);
            }
        }
        fn unroute(bits: [usize; 3], out: &mut Vec<XGate>) {
            let swaps = [(1usize, 2usize, bits[2]), (2, 3, bits[1]), (0, 1, bits[0])];
            for &(i, j, z) in &swaps {
                mol_swap(i, j, z, out);
            }
        }
        fn layer1(out: &mut Vec<XGate>) {
            for i in 0..NMOL {
                out.push(XGate::cnot((2 * i) as u16, (2 * i + 1) as u16)); // a ^= b
            }
        }
        fn layer2(out: &mut Vec<XGate>) {
            for i in 0..NMOL {
                out.push(XGate::cnot((2 * i + 1) as u16, (2 * i) as u16)); // b ^= a
            }
        }

        let a = [9usize, 10, 11];
        let b = [12usize, 13, 14];

        // Ideal C (no aux): layer1 then layer2.
        let mut ideal = Vec::new();
        layer1(&mut ideal);
        let seam_ideal = ideal.len();
        layer2(&mut ideal);

        // PERSIST: route A; fire L1; [seam]; route B; fire L2; then un-route to
        // canonical output. Permutation stays live across the seam.
        let mut persist = Vec::new();
        route(a, &mut persist);
        layer1(&mut persist);
        let seam_persist = persist.len();
        route(b, &mut persist);
        layer2(&mut persist);
        unroute(b, &mut persist);
        unroute(a, &mut persist);

        // ROUTEBACK: route A; fire L1; un-route A; [seam canonical]; route B; L2; un-route B.
        let mut routeback = Vec::new();
        route(a, &mut routeback);
        layer1(&mut routeback);
        unroute(a, &mut routeback);
        let seam_rb = routeback.len();
        route(b, &mut routeback);
        layer2(&mut routeback);
        unroute(b, &mut routeback);

        // ---- exhaustive correctness: for every (data, routing bits), both
        // gadgets must compute ideal C on the 8 data wires (wire8 pinned 0) ----
        let set_state = |asg: usize| -> Vec<bool> {
            let mut s = vec![false; NTOT];
            for w in 0..NDATA {
                s[w] = (asg >> w) & 1 == 1;
            }
            for k in 0..NBITS {
                s[ABASE + k] = (asg >> (NDATA + k)) & 1 == 1;
            }
            s // wire ZERO stays false
        };
        for asg in 0..NPT {
            let base = set_state(asg);
            let mut want = base.clone();
            apply(&ideal, &mut want);
            for (name, g) in [("persist", &persist), ("routeback", &routeback)] {
                let mut got = base.clone();
                apply(g, &mut got);
                assert_eq!(&got[..NDATA], &want[..NDATA], "{name} wrong C at asg {asg}");
            }
        }

        // ---- exposure at the seam: how many of ideal's layer-1 output
        // functions lie in the GF(2) affine span of the gadget's data-wire
        // functions at its seam cut (exhaustive over all NPT inputs) ----
        let func_at = |gates: &[XGate], cut: usize, wire: usize| -> Vec<u64> {
            let mut v = vec![0u64; WORDS];
            for asg in 0..NPT {
                let mut s = set_state(asg);
                for g in &gates[..cut] {
                    let conj = g.ctrls.iter().all(|&(w, p)| s[w as usize] == p);
                    if g.comp ^ conj {
                        s[g.target as usize] ^= true;
                    }
                }
                if s[wire] {
                    v[asg >> 6] |= 1u64 << (asg & 63);
                }
            }
            v
        };
        let xor = |a: &mut Vec<u64>, b: &[u64]| {
            for i in 0..WORDS {
                a[i] ^= b[i];
            }
        };
        // Build a reduced basis of the predictor span (+constant 1), return #targets in span.
        let count_in_span = |preds: &[Vec<u64>], targets: &[Vec<u64>]| -> usize {
            let mut ones = vec![0u64; WORDS];
            for w in ones.iter_mut() {
                *w = !0u64;
            }
            let mut basis: Vec<Vec<u64>> = vec![ones];
            for p in preds {
                let mut r = p.clone();
                for bvec in &basis {
                    let piv = bvec.iter().position(|&x| x != 0);
                    if let Some(pi) = piv {
                        let bit = bvec[pi].trailing_zeros();
                        if (r[pi] >> bit) & 1 == 1 {
                            xor(&mut r, bvec);
                        }
                    }
                }
                if r.iter().any(|&x| x != 0) {
                    basis.push(r);
                }
            }
            let reduce = |mut r: Vec<u64>| -> bool {
                for bvec in &basis {
                    let pi = bvec.iter().position(|&x| x != 0).unwrap();
                    let bit = bvec[pi].trailing_zeros();
                    if (r[pi] >> bit) & 1 == 1 {
                        xor(&mut r, bvec);
                    }
                }
                r.iter().all(|&x| x == 0)
            };
            targets.iter().filter(|t| reduce((*t).clone())).count()
        };

        let targets: Vec<Vec<u64>> = (0..NDATA).map(|w| func_at(&ideal, seam_ideal, w)).collect();
        let preds_p: Vec<Vec<u64>> = (0..NDATA).map(|w| func_at(&persist, seam_persist, w)).collect();
        let preds_r: Vec<Vec<u64>> = (0..NDATA).map(|w| func_at(&routeback, seam_rb, w)).collect();

        let exp_p = count_in_span(&preds_p, &targets);
        let exp_r = count_in_span(&preds_r, &targets);
        eprintln!(
            "[seam] correctness OK (all {NPT} inputs x aux). Layer-1 segments affinely exposed at seam:  PERSIST {exp_p}/{NDATA}   ROUTEBACK {exp_r}/{NDATA}"
        );
        assert!(exp_p < exp_r, "persist should expose fewer seam segments than route-back");
    }

    #[test]
    fn drip_layered_on_sandwich_source_computes_c() {
        use crate::circuit::random_circuit;
        for &nr in &[0usize, 4] {
            let mut rng = StdRng::seed_from_u64(0x5a2d_0000 ^ nr as u64);
            let n_c = 16;
            let m = 100;
            let c = random_circuit(n_c, m);
            let s = sandwich_default_s(n_c);
            let sandwich = sliced_sandwich_cnot(&c, n_c, m, s, &mut rng);
            let sn = sandwich.num_wires;
            let g = gadgetize_drip_layered(&sandwich.gates, sn, nr, &mut rng);
            let total = g.num_wires;
            for _ in 0..48 {
                let x: Vec<bool> = (0..sn).map(|_| rng.random_range(0..2) == 1).collect();
                let mut refs = vec![false; sn];
                refs.copy_from_slice(&x);
                apply(&sandwich.gates, &mut refs);
                let mut gs = vec![false; total];
                gs[..sn].copy_from_slice(&x);
                apply(&g.gates, &mut gs);
                assert_eq!(&gs[..sn], &refs[..], "drip on SANDWICH source mismatch nr={nr}");
            }
        }
    }

    #[test]
    fn drip_persist_on_sandwich_source_computes_c() {
        use crate::circuit::random_circuit;
        let mut rng = StdRng::seed_from_u64(0x9e5a_11cc);
        let n_c = 16;
        let m = 100;
        let c = random_circuit(n_c, m);
        let s = sandwich_default_s(n_c);
        let sandwich = sliced_sandwich_cnot(&c, n_c, m, s, &mut rng);
        let sn = sandwich.num_wires;
        let g = gadgetize_drip_persist(&sandwich.gates, sn, &mut rng);
        let total = g.num_wires;
        for _ in 0..48 {
            let x: Vec<bool> = (0..sn).map(|_| rng.random_range(0..2) == 1).collect();
            let mut refs = vec![false; sn];
            refs.copy_from_slice(&x);
            apply(&sandwich.gates, &mut refs);
            let mut gs = vec![false; total];
            gs[..sn].copy_from_slice(&x);
            apply(&g.gates, &mut gs);
            assert_eq!(&gs[..sn], &refs[..], "persist on SANDWICH source mismatch");
        }
    }

    #[test]
    fn wrap_drip_computes_c_on_upper_half() {
        use crate::circuit::random_circuit;
        let mut rng = StdRng::seed_from_u64(0x11a2_5e6d);
        let n_c = 16;
        let m = 120;
        let c = random_circuit(n_c, m);
        let s = sandwich_default_s(n_c);
        let sandwich = sliced_sandwich_cnot(&c, n_c, m, s, &mut rng);
        let sn = sandwich.num_wires;
        let wrapped = wrap_drip_delivery(&sandwich.gates, sn, 4, 4 * sn, &mut rng);
        let total = wrapped.num_wires;
        let half = sn / 2;
        for _ in 0..48 {
            let x: Vec<bool> = (0..n_c).map(|_| rng.random_range(0..2) == 1).collect();
            let mut sin = vec![false; sn];
            sin[..n_c].copy_from_slice(&x);
            apply(&sandwich.gates, &mut sin);
            let mut ws = vec![false; total];
            ws[..n_c].copy_from_slice(&x);
            apply(&wrapped.gates, &mut ws);
            assert_eq!(&ws[half..sn], &sin[half..sn], "wrap upper-half payload mismatch");
        }
        eprintln!(
            "[wrap] n_c={n_c} sandwich={}g -> wrapped={}g/{total}w (upper-half payload verified)",
            sandwich.gates.len(),
            wrapped.gates.len()
        );
    }

    #[test]
    #[ignore = "emits mpmct1 files for segment_deduce measurement"]
    fn drip_emit_for_segment_deduce() {
        use crate::circuit::random_circuit;
        use crate::engine::format::write_mpmct;
        let dir = "/private/tmp/claude-501/-Users-rancanetti-Documents-local-mixing/659c6d96-f555-40bf-8d53-8d6a9dad9abf/scratchpad";
        let mut rng = StdRng::seed_from_u64(0x5e6d_2026);
        let n_c = 32;
        let m = 400;
        let c = random_circuit(n_c, m);
        let s = sandwich_default_s(n_c);
        let sandwich = sliced_sandwich_cnot(&c, n_c, m, s, &mut rng);
        let sn = sandwich.num_wires; // 64
        write_mpmct(&format!("{dir}/sd_sandwich.mpmct1"), &sandwich.gates, sn).unwrap();
        // Drip gadget for several aux-rerandomization densities.
        for &nr in &[0usize, 4, 16, 64] {
            let g = gadgetize_drip_layered(&sandwich.gates, sn, nr, &mut rng);
            write_mpmct(&format!("{dir}/sd_gadget_nr{nr}.mpmct1"), &g.gates, g.num_wires).unwrap();
            eprintln!("[emit] n_rerand={nr}: gadget={}g/{}w", g.gates.len(), g.num_wires);
        }
        // Persist-mode gadget (no route-back).
        let gp = gadgetize_drip_persist(&sandwich.gates, sn, &mut rng);
        write_mpmct(&format!("{dir}/sd_persist.mpmct1"), &gp.gates, gp.num_wires).unwrap();
        eprintln!("[emit] persist: gadget={}g/{}w", gp.gates.len(), gp.num_wires);
        // No-reshuffle positive control: sandwich fired in place + band fill on the 2n frame.
        let mut ctrl = Vec::new();
        let band: Vec<u16> = (sn..2 * sn).map(|w| w as u16).collect();
        emit_band_fill_src(sn, &band, &mut rng, &mut ctrl);
        ctrl.extend(sandwich.gates.iter().cloned());
        write_mpmct(&format!("{dir}/sd_control.mpmct1"), &ctrl, 2 * sn).unwrap();
        eprintln!(
            "[emit] n_c={n_c} blk={n_c} sandwich={}g/{sn}w control={}g -> {dir}/sd_*.mpmct1",
            sandwich.gates.len(),
            ctrl.len()
        );
    }

    #[test]
    #[ignore = "heavy: layered drip on a real |C|=|D|=3000 sandwich"]
    fn drip_layered_size_on_real_sandwich() {
        use crate::circuit::random_circuit;
        let mut rng = StdRng::seed_from_u64(0x1a4e_c0de);
        let n_c = 128;
        let m = 3000;
        let c = random_circuit(n_c, m);
        let s = sandwich_default_s(n_c);
        let sandwich = sliced_sandwich_cnot(&c, n_c, m, s, &mut rng);
        let sn = sandwich.num_wires;
        let raw = sandwich.gates.len();
        let half = sn / 2;
        let in_group: Vec<bool> = sandwich.gates.iter().map(|g| is_n_column(g, half)).collect();
        let layers = layer_wire_disjoint_grouped(&sandwich.gates, 2 * sn, &in_group);
        let maxlayer = layers.iter().map(|l| l.len()).max().unwrap_or(0);
        let avglayer = raw as f64 / layers.len() as f64;
        // max type-class size within any layer (the routable-class size)
        let mut maxclass = 0usize;
        for l in &layers {
            let mut cnt: std::collections::HashMap<(bool, Vec<bool>), usize> =
                std::collections::HashMap::new();
            for &gi in l {
                *cnt.entry(class_signature(&sandwich.gates[gi])).or_default() += 1;
            }
            maxclass = maxclass.max(cnt.values().copied().max().unwrap_or(0));
        }
        eprintln!("[drip-layered] N-column gates grouped; max type-class in a layer = {maxclass}");
        for &nr in &[0usize, 8] {
            let g = gadgetize_drip_layered(&sandwich.gates, sn, nr, &mut rng);
            let (mut g57, mut wide, mut maxw) = (0usize, 0usize, 0usize);
            for gate in &g.gates {
                let w = gate.ctrls.len();
                maxw = maxw.max(w);
                if gate.comp && w == 2 {
                    g57 += 1;
                }
                if w > 2 {
                    wide += 1;
                }
            }
            eprintln!(
                "[drip-layered] raw={raw}/{sn}w, {} layers (max {maxlayer}, avg {avglayer:.1}) | n_rerand={nr}: {} gates ({:.0}x), g57={:.0}%, wide={wide}, maxw={maxw}, wires={}",
                layers.len(),
                g.gates.len(),
                g.gates.len() as f64 / raw as f64,
                100.0 * g57 as f64 / g.gates.len() as f64,
                g.num_wires
            );
        }
    }

    #[test]
    #[ignore = "heavy: builds a real |C|=|D|=3000 sandwich and drips it"]
    fn drip_size_on_real_sliced_sandwich() {
        use crate::circuit::random_circuit;
        let mut rng = StdRng::seed_from_u64(0x5117_ced5);
        let n_c = 128;
        let m = 3000;
        let c = random_circuit(n_c, m);
        let s = sandwich_default_s(n_c);
        let sandwich = sliced_sandwich_cnot(&c, n_c, m, s, &mut rng);
        let sn = sandwich.num_wires; // 256
        let raw = sandwich.gates.len();

        // Locate the N-column bridge gates: CNOT high<-low, i.e. y_i ^= x_i.
        let n_pos: Vec<usize> = sandwich
            .gates
            .iter()
            .enumerate()
            .filter(|(_, g)| {
                g.ctrls.len() == 1
                    && !g.comp
                    && (g.target as usize) >= n_c
                    && (g.ctrls[0].0 as usize) < n_c
                    && g.ctrls[0].1
            })
            .map(|(i, _)| i)
            .collect();
        let n_lo = n_pos.iter().copied().min();
        let n_hi = n_pos.iter().copied().max();
        eprintln!("[drip-size] raw sliced sandwich = {raw} gates / {sn} wires (|C|=|D|={m})");
        eprintln!(
            "[drip-size] N-form bridge gates: count={}, span={:?}..{:?}",
            n_pos.len(),
            n_lo,
            n_hi
        );

        // Sensitive window = N span padded by n_c each side (fallback: middle).
        let margin = n_c;
        let sensitive = match (n_lo, n_hi) {
            (Some(a), Some(b)) => a.saturating_sub(margin)..(b + margin + 1).min(raw),
            _ => (raw / 2).saturating_sub(2 * n_c)..(raw / 2 + 2 * n_c).min(raw),
        };
        let win = sensitive.end.saturating_sub(sensitive.start);

        // Report a build: pre/post-frag size and the gate-type histogram
        // (indexed by number of controls; 0=NOT, 1=CNOT, 2=Toffoli/g57-form,
        // >=3 = wide multiplex copies). Everything the drip emits is comp=0.
        let report = |label: &str, g: &CnotCircuit| {
            let (mut frag, mut maxw, mut comp) = (0usize, 0usize, 0usize);
            let mut hist = [0usize; 12];
            for gate in &g.gates {
                let w = gate.ctrls.len();
                maxw = maxw.max(w);
                hist[w.min(11)] += 1;
                if gate.comp {
                    comp += 1;
                }
                frag += if w > 2 { 2 * (w - 2) + 1 } else { 1 };
            }
            let total = g.gates.len();
            eprintln!(
                "[drip-size] {label}: pre={total} ({:.0}x), post-frag~{frag} ({:.0}x), maxw={maxw}, comp=1 gates={comp}",
                total as f64 / raw as f64,
                frag as f64 / raw as f64
            );
            let pct = |c: usize| 100.0 * c as f64 / total as f64;
            eprintln!(
                "[drip-size]   makeup: NOT(0c)={} CNOT(1c)={} ({:.0}%) | 2c={} ({:.0}%) | 3c={} 4c={} 5c={} 6c={} 7c={} 8c={} | wide(>2c)={} ({:.0}%)",
                hist[0], hist[1], pct(hist[1]), hist[2], pct(hist[2]),
                hist[3], hist[4], hist[5], hist[6], hist[7], hist[8],
                hist[3..].iter().sum::<usize>(), pct(hist[3..].iter().sum())
            );
        };

        report(
            "BUILD A (k=1 all)          ",
            &gadgetize_drip_regional(&sandwich.gates, sn, 1, 1, 0..0, &mut rng),
        );
        report(
            &format!("BUILD B (k=1,k=2 mid {:.0}%)  ", 100.0 * win as f64 / raw as f64),
            &gadgetize_drip_regional(&sandwich.gates, sn, 1, 2, sensitive.clone(), &mut rng),
        );
        report(
            "ref     (k=2 all)          ",
            &gadgetize_drip_regional(&sandwich.gates, sn, 2, 2, 0..0, &mut rng),
        );
    }
}

/// Relocate one value to a different carrier wire (the single-carrier RG).
fn emit_value_relocation(
    state: &mut GadgetState,
    carrier_total: usize,
    out: &mut Vec<XGate>,
    rng: &mut impl Rng,
) -> (usize, usize) {
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
    (i, j)
}

/// Toggle c0 by the nonlinear part of D.  At the input port this encodes a
/// plain bit x as D(c)=x for arbitrary incoming c1..c4; at the output port the
/// identical self-inverse block decodes D back onto c0.
fn emit_five_carrier_decode_toggle(carriers: &[usize; 5], out: &mut Vec<XGate>) {
    for (a, b) in [(1usize, 2usize), (1, 3), (2, 3), (1, 4)] {
        out.push(
            XGate::conj(
                carriers[0] as u16,
                [(carriers[a] as u16, true), (carriers[b] as u16, true)],
            )
            .expect("five carrier lanes are distinct"),
        );
    }
}

/// Public-port encoder/decoder for the strong cubic five-carrier decode.
/// Every atom excludes c0, so the block is self-inverse for arbitrary junk in
/// c1..c4.
fn emit_strong_five_carrier_decode_toggle(carriers: &[usize; 5], out: &mut Vec<XGate>) {
    const ATOMS: &[&[usize]] = &[&[2], &[2, 3], &[1, 2, 3], &[1, 2, 4], &[3, 4]];
    for atom in ATOMS {
        let controls = atom.iter().map(|&lane| (carriers[lane] as u16, true));
        out.push(
            XGate::conj(carriers[0] as u16, controls).expect("strong-five decode atoms exclude c0"),
        );
    }
}

fn emit_five_carrier_port_toggle(
    flavor: FiveCarrierFlavor,
    carriers: &[usize; 5],
    out: &mut Vec<XGate>,
) {
    match flavor {
        FiveCarrierFlavor::SuppliedQuadratic => emit_five_carrier_decode_toggle(carriers, out),
        FiveCarrierFlavor::StrongCubic => emit_strong_five_carrier_decode_toggle(carriers, out),
    }
}

/// Resolve the joint carrier/band role permutation produced by rolling.  Every
/// carrier lane returns to `lane*n + value`; band roles absorb the displaced
/// junk.  This exposes decoded values only at the final port.
fn route_five_carriers_home(
    state: &mut FiveCarrierState,
    ledger: &mut ProdLedger,
    out: &mut Vec<XGate>,
) {
    for value in 0..state.n {
        for lane in 0..5 {
            let home = lane * state.n + value;
            let current = state.carriers[value][lane];
            if current == home {
                continue;
            }
            emit_wire_swap(current, home, out);

            if let Some(var) = ledger.loc.iter().position(|&wire| wire as usize == home) {
                ledger.loc[var] = current as u16;
            } else {
                let mut displaced = None;
                'find: for other_value in 0..state.n {
                    for other_lane in 0..5 {
                        if state.carriers[other_value][other_lane] == home {
                            displaced = Some((other_value, other_lane));
                            break 'find;
                        }
                    }
                }
                let (other_value, other_lane) =
                    displaced.expect("every non-band wire has one carrier role");
                state.carriers[other_value][other_lane] = current;
            }
            state.carriers[value][lane] = home;
        }
    }
}

fn gadgetize_five_carrier_source(
    source: &[XGate],
    n: usize,
    rg_freq: usize,
    prod: &ProdConfig,
    flavor: FiveCarrierFlavor,
    rng: &mut impl Rng,
) -> CnotCircuit {
    assert!(n >= 3, "five-carrier gadgetization requires n >= 3");
    assert!(
        prod.enabled(),
        "five-carrier gadgetization needs product masks"
    );
    assert!(
        !prod.dist(),
        "distributed mask sourcing is not supported by five-carrier mode"
    );
    assert!(
        source.iter().all(|gate| {
            (gate.target as usize) < n && gate.ctrls.iter().all(|&(wire, _)| (wire as usize) < n)
        }),
        "source wire outside 0..n"
    );

    let carrier_total = 5 * n;
    let band_len = prod.band_size(n);
    let total = carrier_total + band_len;
    assert!(total <= u16::MAX as usize, "too many wires");

    let mut out = Vec::new();
    let band_home: Vec<u16> = (carrier_total..total).map(|wire| wire as u16).collect();
    if prod.fill_nl > 0 {
        emit_band_fill_nl_pivots(
            n,
            &band_home,
            prod.fill_nl,
            prod.fill_pivots > 0,
            rng,
            &mut out,
        );
    } else {
        emit_band_fill(n, &band_home, rng, &mut out);
    }

    let mut state = FiveCarrierState::home_with_flavor(n, flavor);
    for carriers in &state.carriers {
        emit_five_carrier_port_toggle(flavor, carriers, &mut out);
    }
    let mut ledger = ProdLedger::new(n, prod, carrier_total, None);
    ledger.inject_all(&state.c0_view(), rng, &mut out);

    for (index, gate) in source.iter().enumerate() {
        ledger.set_pos(index);
        ledger.fold_five(gate, &state, rng, &mut out);
        if index + 1 == source.len() {
            break;
        }
        // U0 itself is a fixed-point-free, class-preserving representation
        // refresh, so it is the natural five-carrier RG between source gates.
        for _ in 0..rg_freq {
            let value = rng.random_range(0..n);
            state.emit_update(value, &mut out);
        }
        for _ in 0..prod.rsrc {
            ledger.resource(&state.c0_view(), rng, &mut out);
        }
        for _ in 0..prod.roll {
            ledger.roll_five(&mut state, rng, &mut out);
        }
        if prod.epoch > 0 && rng.random_range(0..prod.epoch) == 0 {
            ledger.retire_refill(
                &state.c0_view(),
                prod.refill_data,
                prod.fill_nl,
                rng,
                &mut out,
            );
        }
    }

    ledger.strip_all(&state.c0_view(), rng, &mut out);
    ledger.report();
    route_five_carriers_home(&mut state, &mut ledger, &mut out);
    for carriers in &state.carriers {
        emit_five_carrier_port_toggle(flavor, carriers, &mut out);
    }

    // Every high wire is junk at the public port: four carrier lanes plus the
    // mask band.  Filling the whole region avoids publishing which leftovers
    // are band roles after the rolling body.
    let nondata: Vec<u16> = (n..total).map(|wire| wire as u16).collect();
    if prod.fill_nl > 0 {
        emit_band_fill_nl_pivots(
            n,
            &nondata,
            prod.fill_nl,
            prod.fill_pivots > 0,
            rng,
            &mut out,
        );
    } else {
        emit_band_fill(n, &nondata, rng, &mut out);
    }

    commuting_shuffle(&mut out, rng);
    CnotCircuit {
        gates: out,
        num_wires: total,
    }
}

/// Five-carrier nonlinear gadgetization of a g57 source circuit.
pub fn gadgetize_cnot_five_carrier(
    main: &CircuitSeq,
    n: usize,
    rg_freq: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let mut main = main.clone();
    let rounds = main.gates.len();
    shoot_random_gate(&mut main, rounds);
    let source: Vec<XGate> = main.gates.iter().copied().map(XGate::from_g57).collect();
    gadgetize_five_carrier_source(
        &source,
        n,
        rg_freq,
        prod,
        FiveCarrierFlavor::SuppliedQuadratic,
        rng,
    )
}

/// Five-carrier nonlinear gadgetization of a heterogeneous XGate source.
pub fn gadgetize_xgates_five_carrier(
    source: &[XGate],
    n: usize,
    rg_freq: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    gadgetize_five_carrier_source(
        source,
        n,
        rg_freq,
        prod,
        FiveCarrierFlavor::SuppliedQuadratic,
        rng,
    )
}

/// Experimental strong five-carrier gadgetization.  This uses a cubic decode
/// whose firing bit is outside the complete degree-two endpoint span.
pub fn gadgetize_cnot_strong_five_carrier(
    main: &CircuitSeq,
    n: usize,
    rg_freq: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let mut main = main.clone();
    let rounds = main.gates.len();
    shoot_random_gate(&mut main, rounds);
    let source: Vec<XGate> = main.gates.iter().copied().map(XGate::from_g57).collect();
    gadgetize_five_carrier_source(
        &source,
        n,
        rg_freq,
        prod,
        FiveCarrierFlavor::StrongCubic,
        rng,
    )
}

/// Heterogeneous-source counterpart of
/// [`gadgetize_cnot_strong_five_carrier`].
pub fn gadgetize_xgates_strong_five_carrier(
    source: &[XGate],
    n: usize,
    rg_freq: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    gadgetize_five_carrier_source(
        source,
        n,
        rg_freq,
        prod,
        FiveCarrierFlavor::StrongCubic,
        rng,
    )
}

/// Toggle c0 by all non-c0 atoms of the six-carrier decode. This is a
/// self-inverse public-port encoder/decoder for arbitrary incoming c1..c5.
fn emit_six_carrier_decode_toggle(carriers: &[usize; 6], out: &mut Vec<XGate>) {
    for atom in SIX_CARRIER_D_ATOMS.iter().skip(1) {
        let controls = atom
            .iter()
            .map(|&lane| (carriers[lane as usize] as u16, true));
        out.push(
            XGate::conj(carriers[0] as u16, controls).expect("six-carrier decode atoms exclude c0"),
        );
    }
}

fn route_six_carriers_home(
    state: &mut SixCarrierState,
    ledger: &mut ProdLedger,
    out: &mut Vec<XGate>,
) {
    for value in 0..state.n {
        for lane in 0..6 {
            let home = lane * state.n + value;
            let current = state.carriers[value][lane];
            if current == home {
                continue;
            }
            emit_wire_swap(current, home, out);

            if let Some(var) = ledger.loc.iter().position(|&wire| wire as usize == home) {
                ledger.loc[var] = current as u16;
            } else {
                let mut displaced = None;
                'find: for other_value in 0..state.n {
                    for other_lane in 0..6 {
                        if state.carriers[other_value][other_lane] == home {
                            displaced = Some((other_value, other_lane));
                            break 'find;
                        }
                    }
                }
                let (other_value, other_lane) =
                    displaced.expect("every non-band wire has one six-carrier role");
                state.carriers[other_value][other_lane] = current;
            }
            state.carriers[value][lane] = home;
        }
    }
}

fn gadgetize_six_carrier_source(
    source: &[XGate],
    n: usize,
    rg_freq: usize,
    prod: &ProdConfig,
    flavor: SixCarrierFlavor,
    rng: &mut impl Rng,
) -> CnotCircuit {
    assert!(n >= 3, "six-carrier gadgetization requires n >= 3");
    assert!(
        prod.enabled(),
        "six-carrier gadgetization needs product masks"
    );
    assert!(
        !prod.dist(),
        "distributed mask sourcing is not supported by six-carrier mode"
    );
    assert!(
        source.iter().all(|gate| {
            (gate.target as usize) < n && gate.ctrls.iter().all(|&(wire, _)| (wire as usize) < n)
        }),
        "source wire outside 0..n"
    );

    let carrier_total = 6 * n;
    let band_len = prod.band_size(n);
    let total = carrier_total + band_len;
    assert!(total <= u16::MAX as usize, "too many wires");

    let mut out = Vec::new();
    let band_home: Vec<u16> = (carrier_total..total).map(|wire| wire as u16).collect();
    if prod.fill_nl > 0 {
        emit_band_fill_nl_pivots(
            n,
            &band_home,
            prod.fill_nl,
            prod.fill_pivots > 0,
            rng,
            &mut out,
        );
    } else {
        emit_band_fill(n, &band_home, rng, &mut out);
    }

    let mut state = SixCarrierState::home_with_flavor(n, flavor);
    for carriers in &state.carriers {
        emit_six_carrier_decode_toggle(carriers, &mut out);
    }
    let mut ledger = ProdLedger::new(n, prod, carrier_total, None);
    ledger.inject_all(&state.c0_view(), rng, &mut out);

    for (index, gate) in source.iter().enumerate() {
        ledger.set_pos(index);
        ledger.fold_six(gate, &state, rng, &mut out);
        if index + 1 == source.len() {
            break;
        }
        for _ in 0..rg_freq {
            let value = rng.random_range(0..n);
            state.emit_update(value, &mut out);
        }
        for _ in 0..prod.rsrc {
            ledger.resource(&state.c0_view(), rng, &mut out);
        }
        for _ in 0..prod.roll {
            ledger.roll_six(&mut state, rng, &mut out);
        }
        if prod.epoch > 0 && rng.random_range(0..prod.epoch) == 0 {
            ledger.retire_refill(
                &state.c0_view(),
                prod.refill_data,
                prod.fill_nl,
                rng,
                &mut out,
            );
        }
    }

    ledger.strip_all(&state.c0_view(), rng, &mut out);
    ledger.report();
    route_six_carriers_home(&mut state, &mut ledger, &mut out);
    for carriers in &state.carriers {
        emit_six_carrier_decode_toggle(carriers, &mut out);
    }

    // The five extra carrier lanes and the product band are all arbitrary
    // junk at the public port. Fill the entire high region uniformly.
    let nondata: Vec<u16> = (n..total).map(|wire| wire as u16).collect();
    if prod.fill_nl > 0 {
        emit_band_fill_nl_pivots(
            n,
            &nondata,
            prod.fill_nl,
            prod.fill_pivots > 0,
            rng,
            &mut out,
        );
    } else {
        emit_band_fill(n, &nondata, rng, &mut out);
    }

    commuting_shuffle(&mut out, rng);
    CnotCircuit {
        gates: out,
        num_wires: total,
    }
}

/// Six-carrier nonlinear gadgetization of a g57 source circuit.
pub fn gadgetize_cnot_six_carrier(
    main: &CircuitSeq,
    n: usize,
    rg_freq: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let mut main = main.clone();
    let rounds = main.gates.len();
    shoot_random_gate(&mut main, rounds);
    let source: Vec<XGate> = main.gates.iter().copied().map(XGate::from_g57).collect();
    gadgetize_six_carrier_source(
        &source,
        n,
        rg_freq,
        prod,
        SixCarrierFlavor::SuppliedCompact,
        rng,
    )
}

/// Six-carrier nonlinear gadgetization of a heterogeneous XGate source.
pub fn gadgetize_xgates_six_carrier(
    source: &[XGate],
    n: usize,
    rg_freq: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    gadgetize_six_carrier_source(
        source,
        n,
        rg_freq,
        prod,
        SixCarrierFlavor::SuppliedCompact,
        rng,
    )
}

/// Experimental structural six-carrier gadgetization.  It retains the
/// established cubic decode and degree-three firing boundary, while using a
/// fixed-point-free update whose endpoint graph has full affine rank and no
/// frozen carrier lane.
pub fn gadgetize_cnot_strong_six_carrier(
    main: &CircuitSeq,
    n: usize,
    rg_freq: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let mut main = main.clone();
    let rounds = main.gates.len();
    shoot_random_gate(&mut main, rounds);
    let source: Vec<XGate> = main.gates.iter().copied().map(XGate::from_g57).collect();
    gadgetize_six_carrier_source(
        &source,
        n,
        rg_freq,
        prod,
        SixCarrierFlavor::StrongFullRank,
        rng,
    )
}

/// Heterogeneous-source counterpart of
/// [`gadgetize_cnot_strong_six_carrier`].
pub fn gadgetize_xgates_strong_six_carrier(
    source: &[XGate],
    n: usize,
    rg_freq: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    gadgetize_six_carrier_source(
        source,
        n,
        rg_freq,
        prod,
        SixCarrierFlavor::StrongFullRank,
        rng,
    )
}

/// Toggle c0 by all non-c0 atoms of the seven-carrier decode. This is a
/// self-inverse public-port encoder/decoder for arbitrary incoming c1..c6.
fn emit_seven_carrier_decode_toggle(carriers: &[usize; 7], out: &mut Vec<XGate>) {
    for atom in SEVEN_CARRIER_D_ATOMS.iter().skip(1) {
        let controls = atom
            .iter()
            .map(|&lane| (carriers[lane as usize] as u16, true));
        out.push(
            XGate::conj(carriers[0] as u16, controls)
                .expect("seven-carrier decode atoms exclude c0"),
        );
    }
}

fn route_seven_carriers_home(
    state: &mut SevenCarrierState,
    ledger: &mut ProdLedger,
    out: &mut Vec<XGate>,
) {
    for value in 0..state.n {
        for lane in 0..7 {
            let home = lane * state.n + value;
            let current = state.carriers[value][lane];
            if current == home {
                continue;
            }
            emit_wire_swap(current, home, out);

            if let Some(var) = ledger.loc.iter().position(|&wire| wire as usize == home) {
                ledger.loc[var] = current as u16;
            } else {
                let mut displaced = None;
                'find: for other_value in 0..state.n {
                    for other_lane in 0..7 {
                        if state.carriers[other_value][other_lane] == home {
                            displaced = Some((other_value, other_lane));
                            break 'find;
                        }
                    }
                }
                let (other_value, other_lane) =
                    displaced.expect("every non-band wire has one seven-carrier role");
                state.carriers[other_value][other_lane] = current;
            }
            state.carriers[value][lane] = home;
        }
    }
}

fn live_support_pivot(words: &[u64]) -> Option<usize> {
    words.iter().enumerate().rev().find_map(|(index, &word)| {
        (word != 0).then(|| index * 64 + (63 - word.leading_zeros() as usize))
    })
}

/// Pick band wires whose affine functions on the live input prefix are
/// linearly independent. Candidate order is independently randomized for each
/// captured injection gate, matching the post-hoc boundary experiment.
fn select_independent_band_helpers(
    gate: &XGate,
    band: &[u16],
    supports: &[Vec<u64>],
    bits: usize,
    rng: &mut impl Rng,
) -> Vec<u16> {
    assert_eq!(band.len(), supports.len());
    let occupied: std::collections::HashSet<u16> = std::iter::once(gate.target)
        .chain(gate.ctrls.iter().map(|&(wire, _)| wire))
        .collect();
    let mut candidates: Vec<usize> = band
        .iter()
        .enumerate()
        .filter_map(|(index, &wire)| (!occupied.contains(&wire)).then_some(index))
        .collect();
    candidates.shuffle(rng);

    let basis_len = supports.first().map_or(0, |support| support.len() * 64);
    let mut basis: Vec<Option<Vec<u64>>> = vec![None; basis_len];
    let mut helpers = Vec::with_capacity(bits);
    for index in candidates {
        let mut residual = supports[index].clone();
        loop {
            let Some(pivot) = live_support_pivot(&residual) else {
                break;
            };
            if let Some(row) = &basis[pivot] {
                for (word, &basis_word) in residual.iter_mut().zip(row) {
                    *word ^= basis_word;
                }
                continue;
            }
            basis[pivot] = Some(residual);
            helpers.push(band[index]);
            break;
        }
        if helpers.len() == bits {
            return helpers;
        }
    }
    panic!(
        "initial boundary partition needs {bits} affine-independent band helpers, found {}",
        helpers.len()
    );
}

/// Replace every gate emitted by the one initial `inject_all` call with its
/// exhaustive polarity partition. Complemented conjunctions use
/// `!F = 1 XOR F`, hence the cell-only emission followed by the F-and-cell
/// emission. No shears are inserted in this boundary-specific experiment.
fn emit_partitioned_initial_injection(
    original: &[XGate],
    band: &[u16],
    supports: &[Vec<u64>],
    bits: usize,
    rng: &mut impl Rng,
    out: &mut Vec<XGate>,
) -> usize {
    let cell_count = 1usize
        .checked_shl(bits as u32)
        .expect("initial boundary partition needs too many helper bits");
    let before = out.len();
    for gate in original {
        let helpers = select_independent_band_helpers(gate, band, supports, bits, rng);
        let mut cells: Vec<usize> = (0..cell_count).collect();
        cells.shuffle(rng);
        for cell in cells {
            let cell_lits: Vec<(u16, bool)> = helpers
                .iter()
                .enumerate()
                .map(|(bit, &wire)| (wire, cell & (1 << bit) != 0))
                .collect();
            if gate.comp {
                out.push(
                    XGate::conj(gate.target, cell_lits.iter().copied())
                        .expect("independent helpers exclude the injection target"),
                );
            }
            let mut lits: Vec<(u16, bool)> = gate.ctrls.iter().copied().collect();
            lits.extend(cell_lits);
            out.push(
                XGate::conj(gate.target, lits)
                    .expect("boundary helpers exclude the injection controls"),
            );
        }
    }
    out.len() - before
}

fn gadgetize_seven_carrier_source_with_terminal_start(
    source: &[XGate],
    n: usize,
    rg_freq: usize,
    prod: &ProdConfig,
    distributed_switch: bool,
    partition_floor: usize,
    partition_helper_limit: usize,
    initial_boundary_partition_bits: usize,
    shuffle_output: bool,
    rng: &mut impl Rng,
) -> (CnotCircuit, usize) {
    assert!(n >= 3, "seven-carrier gadgetization requires n >= 3");
    assert!(
        prod.enabled(),
        "seven-carrier gadgetization needs product masks"
    );
    assert!(
        !prod.dist(),
        "distributed mask sourcing is not supported by seven-carrier mode"
    );
    if distributed_switch {
        assert_eq!(
            prod.gray_fold, 0,
            "distributed seven-carrier switching currently requires --prod-gray-fold 0"
        );
    }
    assert!(
        partition_floor == 0 || distributed_switch,
        "fragment partitioning belongs to the distributed seven-carrier path"
    );
    assert!(
        partition_helper_limit <= n,
        "partition helper prefix exceeds the logical wire count"
    );
    let band_len = prod.band_size(n);
    if initial_boundary_partition_bits > 0 {
        assert!(
            partition_floor >= 1024,
            "the boundary-r10 experiment needs a body floor of at least 1024"
        );
        assert_eq!(
            prod.fill_nl, 0,
            "the boundary-r10 experiment requires the linear band fill"
        );
        assert!(
            band_len >= initial_boundary_partition_bits,
            "not enough band wires for the initial boundary partition"
        );
    }
    assert!(
        source.iter().all(|gate| {
            (gate.target as usize) < n && gate.ctrls.iter().all(|&(wire, _)| (wire as usize) < n)
        }),
        "source wire outside 0..n"
    );

    let carrier_total = 7 * n;
    let total = carrier_total + band_len;
    assert!(total <= u16::MAX as usize, "too many wires");

    let mut out = Vec::new();
    let band_home: Vec<u16> = (carrier_total..total).map(|wire| wire as u16).collect();
    let live_band_supports = if initial_boundary_partition_bits > 0 {
        Some(emit_band_fill_with_live_supports(
            n,
            &band_home,
            partition_helper_limit,
            rng,
            &mut out,
        ))
    } else if prod.fill_nl > 0 {
        emit_band_fill_nl_pivots(
            n,
            &band_home,
            prod.fill_nl,
            prod.fill_pivots > 0,
            rng,
            &mut out,
        );
        None
    } else {
        emit_band_fill(n, &band_home, rng, &mut out);
        None
    };

    let mut state = SevenCarrierState::home(n);
    for carriers in &state.carriers {
        emit_seven_carrier_decode_toggle(carriers, &mut out);
    }
    let mut ledger = ProdLedger::new(n, prod, carrier_total, None);
    if initial_boundary_partition_bits > 0 {
        let mut original_injection = Vec::new();
        ledger.inject_all(&state.c0_view(), rng, &mut original_injection);
        let mut boundary_rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(734_001);
        let emitted = emit_partitioned_initial_injection(
            &original_injection,
            &band_home,
            live_band_supports
                .as_ref()
                .expect("boundary partition requires tracked band supports"),
            initial_boundary_partition_bits,
            &mut boundary_rng,
            &mut out,
        );
        println!(
            "[prod] initial-boundary-partition original={} emitted={} bits={}",
            original_injection.len(),
            emitted,
            initial_boundary_partition_bits
        );
    } else {
        ledger.inject_all(&state.c0_view(), rng, &mut out);
    }

    for (index, gate) in source.iter().enumerate() {
        ledger.set_pos(index);
        if distributed_switch {
            ledger.fold_seven_distributed(
                gate,
                &state,
                partition_floor,
                partition_helper_limit,
                rng,
                &mut out,
            );
        } else {
            ledger.fold_seven(gate, &state, rng, &mut out);
        }
        if index + 1 == source.len() {
            break;
        }
        for _ in 0..rg_freq {
            let value = rng.random_range(0..n);
            if distributed_switch {
                let roles = seven_carrier_role_automorphism(rng);
                let mut used = std::collections::HashSet::new();
                let selector = draw_seven_carrier_shear_selector(&state, value, &mut used, rng);
                let x = rng.random_range(3..7) as u8;
                emit_seven_carrier_preserving_shear(&state, value, &roles, x, selector, &mut out);
            } else {
                emit_seven_carrier_update(&state.carriers[value], &mut out);
            }
        }
        for _ in 0..prod.rsrc {
            ledger.resource(&state.c0_view(), rng, &mut out);
        }
        for _ in 0..prod.roll {
            ledger.roll_seven(&mut state, rng, &mut out);
        }
        if prod.epoch > 0 && rng.random_range(0..prod.epoch) == 0 {
            ledger.retire_refill(
                &state.c0_view(),
                prod.refill_data,
                prod.fill_nl,
                rng,
                &mut out,
            );
        }
    }

    // Everything from here onward is the terminal strip/route/decode suffix.
    // Experimental callers that return this boundary require every later
    // ordering transform to keep the suffix intact and forbid crossings at
    // this exact position.
    let terminal_start = out.len();
    ledger.strip_all(&state.c0_view(), rng, &mut out);
    ledger.report();
    route_seven_carriers_home(&mut state, &mut ledger, &mut out);
    for carriers in &state.carriers {
        emit_seven_carrier_decode_toggle(carriers, &mut out);
    }

    if initial_boundary_partition_bits == 0 {
        // The six extra carrier lanes and the product band are arbitrary junk
        // at the public port. Legacy and ordinary experimental paths retain
        // their output-side mirror fill; the boundary-r10 fixture omits it.
        let nondata: Vec<u16> = (n..total).map(|wire| wire as u16).collect();
        if prod.fill_nl > 0 {
            emit_band_fill_nl_pivots(
                n,
                &nondata,
                prod.fill_nl,
                prod.fill_pivots > 0,
                rng,
                &mut out,
            );
        } else {
            emit_band_fill(n, &nondata, rng, &mut out);
        }
    }

    if shuffle_output {
        commuting_shuffle(&mut out, rng);
    }
    (
        CnotCircuit {
            gates: out,
            num_wires: total,
        },
        terminal_start,
    )
}

fn gadgetize_seven_carrier_source(
    source: &[XGate],
    n: usize,
    rg_freq: usize,
    prod: &ProdConfig,
    distributed_switch: bool,
    partition_floor: usize,
    partition_helper_limit: usize,
    initial_boundary_partition_bits: usize,
    shuffle_output: bool,
    rng: &mut impl Rng,
) -> CnotCircuit {
    gadgetize_seven_carrier_source_with_terminal_start(
        source,
        n,
        rg_freq,
        prod,
        distributed_switch,
        partition_floor,
        partition_helper_limit,
        initial_boundary_partition_bits,
        shuffle_output,
        rng,
    )
    .0
}

/// Seven-carrier nonlinear gadgetization of a g57 source circuit.
pub fn gadgetize_cnot_seven_carrier(
    main: &CircuitSeq,
    n: usize,
    rg_freq: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let mut main = main.clone();
    let rounds = main.gates.len();
    shoot_random_gate(&mut main, rounds);
    let source: Vec<XGate> = main.gates.iter().copied().map(XGate::from_g57).collect();
    gadgetize_seven_carrier_source(&source, n, rg_freq, prod, false, 0, n, 0, true, rng)
}

/// Seven-carrier nonlinear gadgetization of a heterogeneous XGate source.
pub fn gadgetize_xgates_seven_carrier(
    source: &[XGate],
    n: usize,
    rg_freq: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    gadgetize_seven_carrier_source(source, n, rg_freq, prod, false, 0, n, 0, true, rng)
}

/// Experimental seven-carrier gadgetization with a randomized distributed
/// refresh schedule.  Each fold relabels the decode roles, emits its source
/// fragments on that occurrence's linear lane, and puts a nonlinear
/// D-preserving shear between consecutive fragments.  This currently supports
/// only expanded/no-Gray product folding (`prod.gray_fold == 0`).
pub fn gadgetize_cnot_seven_carrier_distributed(
    main: &CircuitSeq,
    n: usize,
    rg_freq: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let mut main = main.clone();
    let rounds = main.gates.len();
    shoot_random_gate(&mut main, rounds);
    let source: Vec<XGate> = main.gates.iter().copied().map(XGate::from_g57).collect();
    gadgetize_seven_carrier_source(&source, n, rg_freq, prod, true, 0, n, 0, true, rng)
}

/// Heterogeneous-source counterpart of
/// [`gadgetize_cnot_seven_carrier_distributed`].
pub fn gadgetize_xgates_seven_carrier_distributed(
    source: &[XGate],
    n: usize,
    rg_freq: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    gadgetize_seven_carrier_source(source, n, rg_freq, prod, true, 0, n, 0, true, rng)
}

/// Unshuffled A/B fixture for [`gadgetize_cnot_seven_carrier_distributed`].
/// It exists so trace measurements can distinguish the refresh schedule from
/// the final commuting shuffle; it is not the production-facing variant.
pub fn gadgetize_cnot_seven_carrier_distributed_unshuffled(
    main: &CircuitSeq,
    n: usize,
    rg_freq: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let mut main = main.clone();
    let rounds = main.gates.len();
    shoot_random_gate(&mut main, rounds);
    let source: Vec<XGate> = main.gates.iter().copied().map(XGate::from_g57).collect();
    gadgetize_seven_carrier_source(&source, n, rg_freq, prod, true, 0, n, 0, false, rng)
}

/// Heterogeneous-source counterpart of
/// [`gadgetize_cnot_seven_carrier_distributed_unshuffled`].
pub fn gadgetize_xgates_seven_carrier_distributed_unshuffled(
    source: &[XGate],
    n: usize,
    rg_freq: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    gadgetize_seven_carrier_source(source, n, rg_freq, prod, true, 0, n, 0, false, rng)
}

/// Second experimental distributed-refresh variant.  Every nonempty source
/// fold is polarity-partitioned to at least 128 emitted branches before the
/// preserving shears are inserted.  This removes the direct `2*m <= 100`
/// checkpoint upper bound of a low-fragment fold, but makes no claim about
/// unrelated whole-trace relations.  Every selector bit comes from canonical
/// c0 of a different logical value outside the fold's target and controls.
pub fn gadgetize_cnot_seven_carrier_distributed_partitioned(
    main: &CircuitSeq,
    n: usize,
    rg_freq: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let mut main = main.clone();
    let rounds = main.gates.len();
    shoot_random_gate(&mut main, rounds);
    let source: Vec<XGate> = main.gates.iter().copied().map(XGate::from_g57).collect();
    gadgetize_seven_carrier_source(&source, n, rg_freq, prod, true, 128, n, 0, true, rng)
}

/// Heterogeneous-source counterpart of
/// [`gadgetize_cnot_seven_carrier_distributed_partitioned`].
pub fn gadgetize_xgates_seven_carrier_distributed_partitioned(
    source: &[XGate],
    n: usize,
    rg_freq: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    gadgetize_seven_carrier_source(source, n, rg_freq, prod, true, 128, n, 0, true, rng)
}

/// Unshuffled A/B fixture for
/// [`gadgetize_cnot_seven_carrier_distributed_partitioned`].
pub fn gadgetize_cnot_seven_carrier_distributed_partitioned_unshuffled(
    main: &CircuitSeq,
    n: usize,
    rg_freq: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let mut main = main.clone();
    let rounds = main.gates.len();
    shoot_random_gate(&mut main, rounds);
    let source: Vec<XGate> = main.gates.iter().copied().map(XGate::from_g57).collect();
    gadgetize_seven_carrier_source(&source, n, rg_freq, prod, true, 128, n, 0, false, rng)
}

/// Heterogeneous-source counterpart of
/// [`gadgetize_cnot_seven_carrier_distributed_partitioned_unshuffled`].
pub fn gadgetize_xgates_seven_carrier_distributed_partitioned_unshuffled(
    source: &[XGate],
    n: usize,
    rg_freq: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    gadgetize_seven_carrier_source(source, n, rg_freq, prod, true, 128, n, 0, false, rng)
}

pub fn gadgetize_cnot(
    main: &CircuitSeq,
    n: usize,
    rg_freq: usize,
    masks: &MaskConfig,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    assert!(n >= 3, "gadgetize_cnot requires n >= 3");
    assert!(rg_freq > 0, "rg_freq must be nonzero");
    assert!(
        !(prod.enabled() && masks.cov > 0.0 && masks.k > 0),
        "product-share encoding and deferred masks (RG4) are mutually exclusive"
    );

    let mut main = main.clone();
    let rounds = main.gates.len();
    shoot_random_gate(&mut main, rounds);

    let bookend_size = (2 * n * (n as f64).ln() as usize).max(64);
    let carrier_total = 2 * n;
    let band_range = carrier_total..carrier_total + prod.band_size(n);
    let total = band_range.end;
    assert!(total <= u16::MAX as usize, "too many wires");
    let mut out = rand_z_xgates(n, bookend_size, rng);
    let band_home: Vec<u16> = band_range.map(|w| w as u16).collect();
    if prod.enabled() {
        if prod.fill_nl > 0 {
            emit_band_fill_nl_pivots(
                n,
                &band_home,
                prod.fill_nl,
                prod.fill_pivots > 0,
                rng,
                &mut out,
            );
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
                // Band wire: written by the fill and (under --prod-roll) by
                // band rolls, never relocated by the bookends.
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
    let targets = target_schedule(main.gates.len(), n, |pos| main.gates[pos][0] as usize);
    let prod_sched = prod
        .dist()
        .then(|| target_schedule(main.gates.len(), n, |pos| main.gates[pos][0] as usize));
    let mut ledger = MaskLedger::new(n, masks, targets, rng);
    let mut prod_ledger = ProdLedger::new(n, prod, carrier_total, prod_sched);
    prod_ledger.inject_all(&state, rng, &mut out);
    // RG policy: the legacy NONLINEAR g57 networks {RG1 value-swap (deg 3),
    // RG2 re-pair (deg 2), RG3 cross-value mask refresh (deg 2)}, `rg_freq`
    // uniform draws (default 1 on this path) between consecutive SG gadgets.
    // Reinstated over the linear CNOT {RG2_x, RG3_x}: the affine RGs left the
    // body's re-randomization transparent to degree-1/2 reconstruction. The
    // trade is deliberate — RG1/RG2 gates read both carriers of a value, so
    // gate-local non-completeness is given up for low-degree opacity.
    // Value-sourced deferred masks (RG4) interleave: masks sourced on the CG
    // target flush, masks on read operands are peeked (un-masked for the
    // vanilla CG, then re-masked); RGs preserve values and never disturb them.
    // With the product-share encoding the CG menu is replaced wholesale by the
    // share-native fold (the menu's collapse variants reconstruct operands,
    // which is exactly the use-point exposure the encoding exists to remove).
    for (index, &gate) in main.gates.iter().enumerate() {
        prod_ledger.set_pos(index);
        prod_ledger.bare_census(&state, &out, rng);
        if prod_ledger.enabled() {
            prod_ledger.fold_cg(&XGate::from_g57(gate), &state, rng, &mut out);
        } else {
            let [a, b, c] = gate.map(|w| w as usize);
            ledger.before_cg(&[b, c], a, &state, carrier_total, rng, &mut out);
            emit_cg_menu(&state, gate, rng, &mut out);
            ledger.after_cg(&[b, c], &state, carrier_total, rng, &mut out);
        }
        if index + 1 == main.gates.len() {
            break;
        }
        for _ in 0..rg_freq {
            emit_nonlinear_rg(
                &mut state,
                &mut pair_queue,
                &mut single_queue,
                &mut prod_ledger,
                &mut out,
                rng,
            );
        }
        ledger.top_up(
            index + 1,
            main.gates.len() - 1 - index,
            &state,
            carrier_total,
            rng,
            &mut out,
        );
        for _ in 0..prod.rsrc {
            prod_ledger.resource(&state, rng, &mut out);
        }
        // Rolling band: relocate band variables between physical wires so the
        // band is not a body-static, statically identifiable wire set.
        for _ in 0..prod.roll {
            prod_ledger.roll(&mut state, rng, &mut out);
        }
        // Fire with probability 1/epoch rather than on every epoch-th gate:
        // the same expected rate, without a period an attacker can lock onto.
        if prod.epoch > 0 && rng.random_range(0..prod.epoch) == 0 {
            prod_ledger.retire_refill(&state, prod.refill_data, prod.fill_nl, rng, &mut out);
        }
    }
    ledger.flush_all(&state, carrier_total, rng, &mut out);
    ledger.report();
    prod_ledger.strip_all(&state, rng, &mut out);
    prod_ledger.report();
    // The mirror fill covers EVERY non-output wire, not just the band's. Its
    // target set is part of the emitted gate list, so filling only the band's
    // final wires would publish where the rolls left it — the one fact the
    // roll exists to hide. Filling all of them is uninformative and cheap
    // (~10 gates per wire), and the non-band wires are junk at this point.
    let band_final: Vec<u16> = (n..total).map(|w| w as u16).collect();

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
            // Borrow among the carriers only: the band stays read-only.
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
    // Mirror fill F' on the output side: the band is junk at both ports, so
    // neither direction of a two-sided composition sees it anchored only at
    // its far end. (All slots are stripped by now; the content is free.)
    if prod.enabled() {
        if prod.fill_nl > 0 {
            emit_band_fill_nl_pivots(
                n,
                &band_final,
                prod.fill_nl,
                prod.fill_pivots > 0,
                rng,
                &mut out,
            );
        } else {
            emit_band_fill(n, &band_final, rng, &mut out);
        }
    }
    // Final rerandomization: the construction-time layout (Z | W | body | W^-1
    // | Z) is a legibility artifact; a commuting shuffle interleaves whatever
    // the wire dependencies do not pin down.
    commuting_shuffle(&mut out, rng);
    CnotCircuit {
        gates: out,
        num_wires: total,
    }
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

/// Zero-slice preblock for the gadget path, built purely from
/// positive-polarity CNOTs, CCNOTs and three-control gates, so the block is
/// drawn from the same vocabulary as ordinary mixed-circuit material (no
/// complemented gates, no polarity pattern encoding a slice).
///
/// The wire space is `data 0..n | aux n..2n | band 2n..2n+band`, and the
/// SLICE IS EVERY NON-DATA WIRE — aux and band alike. The product-share band
/// is a live part of the input condition: a nonzero band junks the data
/// exactly like a nonzero aux does. (Without this the band would be provably
/// irrelevant at the input port, which is itself a distinguisher: an
/// adversary can compare `circuit(x, s)` with `circuit(x, 0)` and read off
/// which non-data wires "matter", separating band from aux in a handful of
/// queries.)
///
/// Every gate targets a data wire and reads exactly one slice wire:
/// `x_t ^= s_w`, `x_t ^= x_a & s_w`, or `x_t ^= x_a & x_b & s_w`, with the
/// target and the data controls drawn freely over all data wires. On the
/// all-zero slice the slice control kills every gate individually, so the
/// block is the identity there with no ordering or pairing argument, and the
/// emitted order is one uniform shuffle.
///
/// The three-control shape is what keeps the block from being AFFINE in `x`.
/// With one data control per gate, fixing any slice turns every gate into a
/// constant flip or a transvection, and those compose to an affine map on `x`
/// — degree 1, pinned by `n+1` queries — so the "looks like a random function
/// of both arguments" intent would hold in the slice direction only.
///
/// ## What is guaranteed, and what is checked
/// Off-slice behaviour is deliberately UNSTRUCTURED: any data wire may be a
/// target of one gate and a control of another, so no wire set is exempt from
/// disturbance and no input subcube switches the nonlinearity off. The price
/// is that "no nonzero slice is fixed" is not a theorem, only a property of
/// the draw, so it is CHECKED and the draw repeated until it holds:
///
/// * where the whole space fits (`n + band <= 20`), exhaustively over every
///   slice and every input — that is the regime where wrong-slice fixes were
///   ever observed;
/// * otherwise by sampling: every single-wire slice, many weight-2 slices,
///   and random slices, each against 64 bit-sliced random inputs. This is a
///   spot check, NOT a proof. The honest statement at production width is the
///   old one: a slice is fixed only if its fired subsequence composes to the
///   identity, measured at ~4e-3 of draws at n=3 and 0/50k by n=8, decaying
///   fast in n — and the composite here is a product of hundreds of gates
///   over hundreds of wires.
///
/// Slice wires are covered by a balanced round-robin rather than left to
/// chance: a slice wire that no gate reads is a provably fixed slice, and free
/// draws leave one uncovered a few percent of the time at these budgets. That
/// balance is the only regularity imposed on the block.
///
/// ### An exactly-pinning variant, and why it is not used
/// Splitting the data wires into disjoint pools — targets `T`, data controls
/// `R` — makes every fired gate add `e_t * m(x_R)` for a monomial `m`, so no
/// gate can change another's monomial, the composite is order-free, and
/// "identity on slice s" becomes a LINEAR system in `s` that a rank check can
/// certify outright. One layer of that is exactly pinning but leaves `R` never
/// disturbed and, worse, makes the whole disturbance a function of `x_R`
/// alone, so an adversary who sets `x_R = 0` collapses the block to a
/// translation. Two layers with swapped pools (`T2 = R1`, `R2 = T1`) repair
/// both — the composite is the identity iff each layer's disturbance vanishes
/// identically, which decouples because layer 2 never writes `T1` — and cost
/// no extra gates.
///
/// It is still not used, because layer-2 gates read precisely what layer-1
/// gates write: the two halves cannot commute past each other, so the block
/// carries a MIXING BARRIER that no commuting shuffle and no downstream mixing
/// move can dissolve. That is a permanent two-phase signature bought to
/// exclude a failure — a wrong slice that also computes `C` — whose payoff to
/// an adversary is unclear, since `C` is computable on the honest slice
/// anyway. Structure that mixing cannot erase is the more expensive side of
/// that trade.
pub fn slice_zero_ccnot_preblock(
    n: usize,
    band: usize,
    gate_count: usize,
    rng: &mut impl Rng,
) -> CnotCircuit {
    // Paired build: the slice is the aux half plus the band.
    slice_zero_preblock_dims(n, n + band, gate_count, rng)
}

/// The preblock, parameterised by how wide the slice actually is. A
/// single-carrier gadget has no aux half, so its slice is the band alone;
/// everything below already works in terms of (data, slice, total) and only
/// the dimensions differ.
fn slice_zero_preblock_dims(
    n: usize,
    nondata: usize,
    gate_count: usize,
    rng: &mut impl Rng,
) -> CnotCircuit {
    try_slice_zero_preblock_dims(n, nondata, gate_count, rng)
        .unwrap_or_else(|error| panic!("{error}"))
}

pub(crate) fn try_slice_zero_preblock_dims(
    n: usize,
    nondata: usize,
    gate_count: usize,
    rng: &mut impl Rng,
) -> Result<CnotCircuit, String> {
    try_slice_zero_block_dims(n, n, nondata, gate_count, rng)
}

/// The junk-half slice guard (see [`ProdConfig::close_slice`]): a slice
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
    try_slice_zero_block_dims(n, n / 2, nondata, gate_count, rng)
        .unwrap_or_else(|error| panic!("{error}"))
}

/// Shared generator for the opening/closing slice blocks: targets are drawn
/// from `0..target_hi`, slice controls from the `nondata` wires above `n`.
fn try_slice_zero_block_dims(
    n: usize,
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
    if !(3..=n).contains(&target_hi) {
        return Err(format!(
            "slice block targets need 3 <= target_hi <= {n}, got {target_hi}"
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
            let target = rng.random_range(0..target_hi);
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
         few to disturb 2^{nondata} slices distinctly — raise n or lower --prod-band"
    ))
}

/// Nonlinear-GSS counterpart of [`try_slice_zero_preblock_dims`].
///
/// The draw and gate-shape policy deliberately matches the established
/// product-family constructor, but wide-slice validation uses an indexed,
/// batched checker.  Nonlinear layouts can have tens of thousands of slice
/// wires, where replaying the whole preblock once per singleton slice is
/// quadratic and makes an otherwise admissible layout impractical to build.
/// Keeping this as a separate entry point preserves the product constructor's
/// byte-for-byte RNG stream and artifacts.
pub(crate) fn try_nonlinear_slice_zero_preblock_dims(
    n: usize,
    nondata: usize,
    gate_count: usize,
    fanin_two: bool,
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
    if fanin_two {
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
    }

    let cnots = gate_count / 3;
    let rest = gate_count - cnots;
    let quads = rest / 2;
    let ccnots = rest - quads;
    let emitted_count =
        if fanin_two {
            gate_count
                .checked_add(quads.checked_mul(3).ok_or_else(|| {
                    "nonlinear preblock decomposed gate-count overflow".to_string()
                })?)
                .ok_or_else(|| "nonlinear preblock emitted gate-count overflow".to_string())?
        } else {
            gate_count
        };

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
            match (fanin_two, spec.data_len) {
                (_, 0) => gates.push(
                    XGate::conj(spec.target, [(spec.slice, true)])
                        .expect("preblock target and slice control are distinct"),
                ),
                (_, 1) => gates.push(
                    XGate::conj(spec.target, [(spec.data[0], true), (spec.slice, true)])
                        .expect("preblock target and controls are distinct"),
                ),
                (false, 2) => gates.push(
                    XGate::conj(
                        spec.target,
                        [
                            (spec.data[0], true),
                            (spec.data[1], true),
                            (spec.slice, true),
                        ],
                    )
                    .expect("preblock target and controls are distinct"),
                ),
                (true, 2) => {
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
                (_, other) => unreachable!("unsupported preblock data-control count {other}"),
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

/// Exhaustive check that only the all-zero slice leaves the data untouched:
/// every slice against every input. Affordable only while `2n + band` is
/// small, which is exactly the regime where wrong-slice fixes were ever
/// observed in the first place.
fn slice_preblock_fixes_only_zero_slice(gates: &[XGate], n: usize, nondata: usize) -> bool {
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

/// Scalable wide-slice checker used only by the nonlinear GSS adapter.
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
/// Gadgetization with the CNOT/CCNOT zero-slice preblock prepended: the
/// composite computes `main` on the low n wires exactly when every non-data
/// wire is zero, and `main` of a disturbed input on every other slice (the
/// disturbance is quadratic in x, not affine — see
/// [`slice_zero_ccnot_preblock`]). The slice block sits at the input port only.
///
/// Its inverse also guards the inverse circuit: A^-1 = G^-1 ; S1^-1, and
/// S1^-1 fires on the gadget's generically nonzero mask residue, junking
/// the low half — so the inverse does not surface C^-1 (a bare gadget
/// returns C^-1 on its low wires for ANY junk input).
pub fn gadgetize_with_slice_zero_ccnot(
    main: &CircuitSeq,
    n: usize,
    rg_freq: usize,
    gate_count: usize,
    masks: &MaskConfig,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    if prod.single_carrier() {
        // No aux half to pin: the slice is the band alone.
        let band = prod.band_size(n);
        let mut circuit = slice_zero_preblock_dims(n, band, gate_count.max(band), rng);
        let gadget = gadgetize_cnot_single(main, n, rg_freq, prod, rng);
        circuit.num_wires = circuit.num_wires.max(gadget.num_wires);
        circuit.gates.extend(gadget.gates);
        commuting_shuffle(&mut circuit.gates, rng);
        return circuit;
    }
    // The band is part of the slice: the preblock must see its width.
    let mut circuit = slice_zero_ccnot_preblock(n, prod.band_size(n), gate_count, rng);
    let gadget = gadgetize_cnot(main, n, rg_freq, masks, prod, rng);
    circuit.num_wires = circuit.num_wires.max(gadget.num_wires);
    circuit.gates.extend(gadget.gates);
    // Shuffle across the preblock/gadget seam too, so the slice block does
    // not sit as a contiguous prefix.
    commuting_shuffle(&mut circuit.gates, rng);
    circuit
}

/// Five-carrier gadgetization behind the nonlinear zero-slice preblock.  The
/// protected slice contains four extra carrier lanes per value plus the mask
/// band, so its width is `4*n + band`.
pub fn gadgetize_with_slice_zero_ccnot_five_carrier(
    main: &CircuitSeq,
    n: usize,
    rg_freq: usize,
    gate_count: usize,
    _masks: &MaskConfig,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let nondata = 4 * n + prod.band_size(n);
    let mut circuit = slice_zero_preblock_dims(n, nondata, gate_count.max(nondata), rng);
    let gadget = gadgetize_cnot_five_carrier(main, n, rg_freq, prod, rng);
    circuit.num_wires = circuit.num_wires.max(gadget.num_wires);
    circuit.gates.extend(gadget.gates);
    commuting_shuffle(&mut circuit.gates, rng);
    circuit
}

/// Strong cubic five-carrier counterpart of
/// [`gadgetize_with_slice_zero_ccnot_five_carrier`].
pub fn gadgetize_with_slice_zero_ccnot_strong_five_carrier(
    main: &CircuitSeq,
    n: usize,
    rg_freq: usize,
    gate_count: usize,
    _masks: &MaskConfig,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let nondata = 4 * n + prod.band_size(n);
    let mut circuit = slice_zero_preblock_dims(n, nondata, gate_count.max(nondata), rng);
    let gadget = gadgetize_cnot_strong_five_carrier(main, n, rg_freq, prod, rng);
    circuit.num_wires = circuit.num_wires.max(gadget.num_wires);
    circuit.gates.extend(gadget.gates);
    commuting_shuffle(&mut circuit.gates, rng);
    circuit
}

/// Six-carrier gadgetization behind the nonlinear zero-slice preblock. The
/// protected slice is five extra carrier lanes per value plus the mask band.
pub fn gadgetize_with_slice_zero_ccnot_six_carrier(
    main: &CircuitSeq,
    n: usize,
    rg_freq: usize,
    gate_count: usize,
    _masks: &MaskConfig,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let nondata = 5 * n + prod.band_size(n);
    let mut circuit = slice_zero_preblock_dims(n, nondata, gate_count.max(nondata), rng);
    let gadget = gadgetize_cnot_six_carrier(main, n, rg_freq, prod, rng);
    circuit.num_wires = circuit.num_wires.max(gadget.num_wires);
    circuit.gates.extend(gadget.gates);
    commuting_shuffle(&mut circuit.gates, rng);
    circuit
}

/// Full-affine-rank strong-six counterpart of
/// [`gadgetize_with_slice_zero_ccnot_six_carrier`].
pub fn gadgetize_with_slice_zero_ccnot_strong_six_carrier(
    main: &CircuitSeq,
    n: usize,
    rg_freq: usize,
    gate_count: usize,
    _masks: &MaskConfig,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let nondata = 5 * n + prod.band_size(n);
    let mut circuit = slice_zero_preblock_dims(n, nondata, gate_count.max(nondata), rng);
    let gadget = gadgetize_cnot_strong_six_carrier(main, n, rg_freq, prod, rng);
    circuit.num_wires = circuit.num_wires.max(gadget.num_wires);
    circuit.gates.extend(gadget.gates);
    commuting_shuffle(&mut circuit.gates, rng);
    circuit
}

/// Seven-carrier gadgetization behind the nonlinear zero-slice preblock. The
/// protected slice is six extra carrier lanes per value plus the mask band.
pub fn gadgetize_with_slice_zero_ccnot_seven_carrier(
    main: &CircuitSeq,
    n: usize,
    rg_freq: usize,
    gate_count: usize,
    _masks: &MaskConfig,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let nondata = 6 * n + prod.band_size(n);
    let mut circuit = slice_zero_preblock_dims(n, nondata, gate_count.max(nondata), rng);
    let gadget = gadgetize_cnot_seven_carrier(main, n, rg_freq, prod, rng);
    circuit.num_wires = circuit.num_wires.max(gadget.num_wires);
    circuit.gates.extend(gadget.gates);
    commuting_shuffle(&mut circuit.gates, rng);
    circuit
}

/// Two-share gadgetization of a heterogeneous mpmct1 `source` (CNOT/CCNOT/
/// g57/fragments) on `n` wires, producing a 2n-wire circuit whose low n
/// output wires equal `source(x)` for any aux input. Identical scaffolding to
/// [`gadgetize_cnot`] (bookends, W_i encode/decode, nonlinear {RG1,RG2,RG3}
/// policy, final [`commuting_shuffle`]) but each
/// source gate is shared by the general `emit_shared_xgate2` rather than the
/// g57-only four-fragment SG, so any mpmct1 circuit can be ingested. Unlike
/// the g57 path there is no `shoot_random_gate` reorder (the source is a fixed
/// heterogeneous list, not a g57 CircuitSeq).
pub fn gadgetize_xgates(
    source: &[XGate],
    n: usize,
    rg_freq: usize,
    masks: &MaskConfig,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    assert!(n >= 3, "gadgetize_xgates requires n >= 3");
    assert!(rg_freq > 0, "rg_freq must be nonzero");
    assert!(
        !(prod.enabled() && masks.cov > 0.0 && masks.k > 0),
        "product-share encoding and deferred masks (RG4) are mutually exclusive"
    );
    assert!(
        source
            .iter()
            .all(|g| { (g.target as usize) < n && g.ctrls.iter().all(|&(w, _)| (w as usize) < n) }),
        "source wire outside 0..n"
    );

    let bookend_size = (2 * n * (n as f64).ln() as usize).max(64);
    let carrier_total = 2 * n;
    let band_range = carrier_total..carrier_total + prod.band_size(n);
    let total = band_range.end;
    assert!(total <= u16::MAX as usize, "too many wires");
    let mut out = rand_z_xgates(n, bookend_size, rng);
    let band_home: Vec<u16> = band_range.map(|w| w as u16).collect();
    if prod.enabled() {
        if prod.fill_nl > 0 {
            emit_band_fill_nl_pivots(
                n,
                &band_home,
                prod.fill_nl,
                prod.fill_pivots > 0,
                rng,
                &mut out,
            );
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
                // Band wire: written by the fill and (under --prod-roll) by
                // band rolls, never relocated by the bookends.
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
    let targets = target_schedule(source.len(), n, |pos| source[pos].target as usize);
    let prod_sched = prod
        .dist()
        .then(|| target_schedule(source.len(), n, |pos| source[pos].target as usize));
    let mut ledger = MaskLedger::new(n, masks, targets, rng);
    let mut prod_ledger = ProdLedger::new(n, prod, carrier_total, prod_sched);
    prod_ledger.inject_all(&state, rng, &mut out);
    // Same nonlinear {RG1, RG2, RG3} policy, CG menu, and value-sourced
    // deferred-mask (RG4) handling as `gadgetize_cnot` (flush masks sourced
    // on the target, peek masks on the read operands); source gates that are
    // not g57-shaped keep the general ANF sharing. With the product-share
    // encoding every source gate goes through the share-native fold instead.
    for (index, gate) in source.iter().enumerate() {
        prod_ledger.set_pos(index);
        prod_ledger.bare_census(&state, &out, rng);
        if prod_ledger.enabled() {
            prod_ledger.fold_cg(gate, &state, rng, &mut out);
        } else {
            let reads: Vec<usize> = gate.ctrls.iter().map(|&(w, _)| w as usize).collect();
            ledger.before_cg(
                &reads,
                gate.target as usize,
                &state,
                carrier_total,
                rng,
                &mut out,
            );
            if let Some(triple) = as_g57_triple(gate) {
                emit_cg_menu(&state, triple, rng, &mut out);
            } else {
                emit_shared_xgate2(&state, gate, &mut out);
            }
            ledger.after_cg(&reads, &state, carrier_total, rng, &mut out);
        }
        if index + 1 == source.len() {
            break;
        }
        for _ in 0..rg_freq {
            emit_nonlinear_rg(
                &mut state,
                &mut pair_queue,
                &mut single_queue,
                &mut prod_ledger,
                &mut out,
                rng,
            );
        }
        ledger.top_up(
            index + 1,
            source.len() - 1 - index,
            &state,
            carrier_total,
            rng,
            &mut out,
        );
        for _ in 0..prod.rsrc {
            prod_ledger.resource(&state, rng, &mut out);
        }
        // Rolling band: relocate band variables between physical wires so the
        // band is not a body-static, statically identifiable wire set.
        for _ in 0..prod.roll {
            prod_ledger.roll(&mut state, rng, &mut out);
        }
        // Fire with probability 1/epoch rather than on every epoch-th gate:
        // the same expected rate, without a period an attacker can lock onto.
        if prod.epoch > 0 && rng.random_range(0..prod.epoch) == 0 {
            prod_ledger.retire_refill(&state, prod.refill_data, prod.fill_nl, rng, &mut out);
        }
    }
    ledger.flush_all(&state, carrier_total, rng, &mut out);
    ledger.report();
    prod_ledger.strip_all(&state, rng, &mut out);
    prod_ledger.report();
    // The mirror fill covers EVERY non-output wire, not just the band's. Its
    // target set is part of the emitted gate list, so filling only the band's
    // final wires would publish where the rolls left it — the one fact the
    // roll exists to hide. Filling all of them is uninformative and cheap
    // (~10 gates per wire), and the non-band wires are junk at this point.
    let band_final: Vec<u16> = (n..total).map(|w| w as u16).collect();

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
            // Borrow among the carriers only: the band stays read-only.
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
    // Mirror fill F' on the output side: the band is junk at both ports, so
    // neither direction of a two-sided composition sees it anchored only at
    // its far end. (All slots are stripped by now; the content is free.)
    if prod.enabled() {
        if prod.fill_nl > 0 {
            emit_band_fill_nl_pivots(
                n,
                &band_final,
                prod.fill_nl,
                prod.fill_pivots > 0,
                rng,
                &mut out,
            );
        } else {
            emit_band_fill(n, &band_final, rng, &mut out);
        }
    }
    // Final rerandomization: the construction-time layout (Z | W | body | W^-1
    // | Z) is a legibility artifact; a commuting shuffle interleaves whatever
    // the wire dependencies do not pin down.
    commuting_shuffle(&mut out, rng);
    CnotCircuit {
        gates: out,
        num_wires: total,
    }
}

/// New-gadgetize an mpmct1 `source`: the slice-zero-ccnot preblock prepended
/// to `gadgetize_xgates`. Used to gadgetize the sliced sandwich (a second,
/// independent zero-slice on top of the sandwich's own).
/// Single-carrier gadgetization of an XGate source (the sliced sandwich path).
/// Same decode as [`gadgetize_cnot_single`] — `v = c_v ^ masks ^ κ` on `n`
/// carriers rather than `2n` — for the production pipeline's input format.
///
/// Re-randomisation is R1 plus mask re-sourcing, which together cover what the
/// paired build got from R1/R2/R3: relocation moves a value to another wire
/// (position), and a re-source XORs a fresh product in and the old one out, so
/// the carrier's BIT changes (representation). R2 and R3 have no analogue and
/// are not needed.
pub fn gadgetize_xgates_single(
    source: &[XGate],
    n: usize,
    rg_freq: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    assert!(n >= 3, "gadgetize_xgates_single requires n >= 3");
    assert!(
        prod.single_carrier(),
        "gadgetize_xgates_single needs --prod-single with a nonempty mask plan"
    );
    assert!(
        !prod.dist(),
        "distributed sourcing and single-carrier are not combined yet"
    );
    // With one carrier there is no RG3 to refresh the representation: a
    // re-source is the ONLY move that changes a carrier's bit. rsrc = 0 leaves
    // relocation as the sole churn, which moves values without re-randomising
    // them -- documented as load-bearing, so enforce it rather than trust it.
    assert!(
        prod.rsrc >= 1,
        "--prod-single needs --prod-rsrc >= 1: with a single carrier, mask \
         re-sourcing is the only representation refresh (R2/R3 have no analogue)"
    );
    assert!(
        source
            .iter()
            .all(|g| { (g.target as usize) < n && g.ctrls.iter().all(|&(w, _)| (w as usize) < n) }),
        "source wire outside 0..n"
    );

    let carrier_total = n;
    let band_len = prod.band_size(n);
    let total = carrier_total + band_len;
    assert!(total <= u16::MAX as usize, "too many wires");

    let mut out: Vec<XGate> = Vec::new();
    // Under the closing-slice design both fills source from the LOW data
    // half: at the input port that is where x lives (the high half is zero),
    // and at the output port the low half is still masked — a fill CNOT
    // reading a bare payload wire writes that payload bit out as a local
    // segment delta (the boundary flip-match class the redesign eliminates).
    let fill_src_hi = if prod.close_slice > 0 { n / 2 } else { n };
    let band_home: Vec<u16> = (carrier_total..total).map(|w| w as u16).collect();
    if prod.fill_nl > 0 {
        emit_band_fill_nl_pivots_src(
            fill_src_hi,
            &band_home,
            prod.fill_nl,
            prod.fill_pivots > 0,
            rng,
            &mut out,
        );
    } else {
        emit_band_fill_src(fill_src_hi, &band_home, rng, &mut out);
    }

    let mut state = GadgetState {
        n,
        pairs: (0..n).map(|w| (w, w)).collect(),
    };
    let mut prod_ledger = ProdLedger::new(n, prod, carrier_total, None);
    prod_ledger.inject_all(&state, rng, &mut out);

    for (index, gate) in source.iter().enumerate() {
        prod_ledger.set_pos(index);
        prod_ledger.fold_cg(gate, &state, rng, &mut out);
        if index + 1 == source.len() {
            break;
        }
        for _ in 0..rg_freq {
            let (i, j) = emit_value_relocation(&mut state, carrier_total, &mut out, rng);
            // Relocation-coupled refresh: a relocation moves a value's mask
            // content WHOLESALE, so a rarely-written value (a payload value
            // is a fold target exactly once, at its N gate) re-exhibits the
            // same mask function at every stop, and segment pairs cutting
            // through two relocations of matching content recover its single
            // value transition exactly — measured as the last flip_match
            // residue. Refreshing one monomial of each moved value at every
            // relocation makes every representation event a fresh function.
            if prod.swap_refresh > 0 {
                for value in [i, j] {
                    let mut inj = Vec::new();
                    let mut strip = Vec::new();
                    if prod_ledger.swap_refresh_side(value, &state, rng, &mut inj, &mut strip)
                    {
                        out.extend(inj);
                        out.extend(strip);
                    }
                }
            }
        }
        for _ in 0..prod.rsrc {
            prod_ledger.resource(&state, rng, &mut out);
        }
        for _ in 0..prod.roll {
            prod_ledger.roll(&mut state, rng, &mut out);
        }
        // Fire with probability 1/epoch rather than on every epoch-th gate:
        // the same expected rate, without a period an attacker can lock onto.
        if prod.epoch > 0 && rng.random_range(0..prod.epoch) == 0 {
            prod_ledger.retire_refill(&state, prod.refill_data, prod.fill_nl, rng, &mut out);
        }
    }
    // Route every value home BEFORE stripping (rolls can leave one on a
    // former band wire), so wires 0..n hold the values and n..total hold band
    // junk. Routing bare values would be a leak in its own right: a wire swap
    // of two stripped carriers writes their bare values as local segment
    // deltas at the output port, and a bare payload delta is exactly a source
    // wire-segment XOR (measured: the N column and the CNOT-shaped S2 gates
    // matched at 100% through the swap redesign until this was reordered).
    // Swapping MASKED carriers leaves only mask-polluted deltas behind; the
    // slots travel with their values (they name band variables, not carrier
    // wires), so the strip below lands on the home wires unchanged.
    let mut owner: Vec<Option<usize>> = vec![None; total];
    for value in 0..n {
        owner[state.pairs[value].0] = Some(value);
    }
    // Wire -> band variable currently living there. Rolls can park a band
    // variable anywhere, including the carrier space, and the strip below
    // resolves slot literals through `loc` — so the swaps must carry the
    // band placement along with the values or the strip reads dead wires.
    let mut band_at: Vec<Option<usize>> = vec![None; total];
    for (b, &w) in prod_ledger.loc.iter().enumerate() {
        band_at[w as usize] = Some(b);
    }
    for value in 0..n {
        let cur = state.pairs[value].0;
        if cur == value {
            continue;
        }
        emit_wire_swap(cur, value, &mut out);
        let displaced = owner[value];
        owner[cur] = displaced;
        if let Some(u) = displaced {
            state.pairs[u] = (cur, cur);
        }
        owner[value] = Some(value);
        state.pairs[value] = (value, value);
        if let Some(b) = band_at[cur] {
            prod_ledger.loc[b] = value as u16;
        }
        if let Some(b) = band_at[value] {
            prod_ledger.loc[b] = cur as u16;
        }
        band_at.swap(cur, value);
    }

    // EVERY value's registry is discharged — including the junk half. An
    // undischarged registry leaves the emission telescope open under REVERSE
    // evaluation: a fold reading value v compensates v's mask fragments
    // against the carrier's reverse-time content, which is the XOR of the
    // emissions AFTER the fold; that set telescopes to v's fold-time slots
    // exactly when the final registry is stripped, and is off by the final
    // slots otherwise. The N column reads low values into the upper half, so
    // keeping the junk half masked corrupted the reverse payload D^-1(a).
    // The bare-junk tail leaks that once motivated masking are closed by the
    // rest of the tail hygiene: route-home precedes the strip (no bare-value
    // swaps), the discharge helpers come from the band-only pool, and the
    // closing guard (wrapper level) re-junks the low half before delivery.
    prod_ledger.strip_all(&state, rng, &mut out);
    prod_ledger.report();

    let band_final: Vec<u16> = (carrier_total..total).map(|w| w as u16).collect();
    if prod.fill_nl > 0 {
        emit_band_fill_nl_pivots_src(
            fill_src_hi,
            &band_final,
            prod.fill_nl,
            prod.fill_pivots > 0,
            rng,
            &mut out,
        );
    } else {
        emit_band_fill_src(fill_src_hi, &band_final, rng, &mut out);
    }

    if prod.swap_refresh > 0 {
        commuting_shuffle_stable_targets(&mut out, rng);
        cancel_identical_pairs(&mut out);
    } else {
        commuting_shuffle(&mut out, rng);
    }
    CnotCircuit {
        gates: out,
        num_wires: total,
    }
}

/// Single-carrier gadget behind the zero-slice preblock. The slice is the band
/// alone — there is no aux half to pin — so the preblock is built over
/// `(data n, slice band)` instead of `(data n, slice n+band)`.
pub fn gadgetize_xgates_with_slice_zero_ccnot_single(
    source: &[XGate],
    n: usize,
    rg_freq: usize,
    gate_count: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let band = prod.band_size(n);
    // Symmetric ports (close_slice): BOTH guards are independent draws of
    // the junk-half generator. The opening guard must avoid the upper half
    // for reverse honesty — its inverse runs last under reverse evaluation
    // and fires on the then-junk band, where an upper-half target would
    // junk the just-emerged reverse payload D^-1(a). See
    // slice_zero_junk_guard_dims for the full port algebra.
    let mut circuit = if prod.close_slice > 0 {
        slice_zero_junk_guard_dims(n, band, gate_count.max(band), rng)
    } else {
        slice_zero_preblock_dims(n, band, gate_count.max(band), rng)
    };
    let gadget = gadgetize_xgates_single(source, n, rg_freq, prod, rng);
    circuit.num_wires = circuit.num_wires.max(gadget.num_wires);
    circuit.gates.extend(gadget.gates);
    if prod.close_slice > 0 {
        // The closing guard: on the honest forward slice it fires against
        // the mirror fill's band junk and perturbs only the low (junk)
        // half, so the composite preserves main's output on the UPPER half
        // of the data wires; a reverse evaluator entering on a zero band
        // meets it as a dead guard, exactly as a forward evaluator meets
        // the opening one.
        let post = slice_zero_junk_guard_dims(n, band, gate_count.max(band), rng);
        circuit.gates.extend(post.gates);
    }
    if prod.swap_refresh > 0 {
        commuting_shuffle_stable_targets(&mut circuit.gates, rng);
        // Emission-waste cleanup (see cancel_identical_pairs): ~2% of the
        // build is coinciding residue CNOT pairs; the deliberate mask
        // redundancy is collision-guarded and untouchable by this pass.
        cancel_identical_pairs(&mut circuit.gates);
    } else {
        commuting_shuffle(&mut circuit.gates, rng);
    }
    circuit
}

/// XGate-source counterpart of
/// [`gadgetize_with_slice_zero_ccnot_five_carrier`].
pub fn gadgetize_xgates_with_slice_zero_ccnot_five_carrier(
    source: &[XGate],
    n: usize,
    rg_freq: usize,
    gate_count: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let nondata = 4 * n + prod.band_size(n);
    let mut circuit = slice_zero_preblock_dims(n, nondata, gate_count.max(nondata), rng);
    let gadget = gadgetize_xgates_five_carrier(source, n, rg_freq, prod, rng);
    circuit.num_wires = circuit.num_wires.max(gadget.num_wires);
    circuit.gates.extend(gadget.gates);
    commuting_shuffle(&mut circuit.gates, rng);
    circuit
}

/// XGate-source counterpart of
/// [`gadgetize_with_slice_zero_ccnot_strong_five_carrier`].
pub fn gadgetize_xgates_with_slice_zero_ccnot_strong_five_carrier(
    source: &[XGate],
    n: usize,
    rg_freq: usize,
    gate_count: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let nondata = 4 * n + prod.band_size(n);
    let mut circuit = slice_zero_preblock_dims(n, nondata, gate_count.max(nondata), rng);
    let gadget = gadgetize_xgates_strong_five_carrier(source, n, rg_freq, prod, rng);
    circuit.num_wires = circuit.num_wires.max(gadget.num_wires);
    circuit.gates.extend(gadget.gates);
    commuting_shuffle(&mut circuit.gates, rng);
    circuit
}

/// XGate-source counterpart of
/// [`gadgetize_with_slice_zero_ccnot_six_carrier`].
pub fn gadgetize_xgates_with_slice_zero_ccnot_six_carrier(
    source: &[XGate],
    n: usize,
    rg_freq: usize,
    gate_count: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let nondata = 5 * n + prod.band_size(n);
    let mut circuit = slice_zero_preblock_dims(n, nondata, gate_count.max(nondata), rng);
    let gadget = gadgetize_xgates_six_carrier(source, n, rg_freq, prod, rng);
    circuit.num_wires = circuit.num_wires.max(gadget.num_wires);
    circuit.gates.extend(gadget.gates);
    commuting_shuffle(&mut circuit.gates, rng);
    circuit
}

/// XGate-source counterpart of
/// [`gadgetize_with_slice_zero_ccnot_strong_six_carrier`].
pub fn gadgetize_xgates_with_slice_zero_ccnot_strong_six_carrier(
    source: &[XGate],
    n: usize,
    rg_freq: usize,
    gate_count: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let nondata = 5 * n + prod.band_size(n);
    let mut circuit = slice_zero_preblock_dims(n, nondata, gate_count.max(nondata), rng);
    let gadget = gadgetize_xgates_strong_six_carrier(source, n, rg_freq, prod, rng);
    circuit.num_wires = circuit.num_wires.max(gadget.num_wires);
    circuit.gates.extend(gadget.gates);
    commuting_shuffle(&mut circuit.gates, rng);
    circuit
}

/// XGate-source counterpart of
/// [`gadgetize_with_slice_zero_ccnot_seven_carrier`].
pub fn gadgetize_xgates_with_slice_zero_ccnot_seven_carrier(
    source: &[XGate],
    n: usize,
    rg_freq: usize,
    gate_count: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let nondata = 6 * n + prod.band_size(n);
    let mut circuit = slice_zero_preblock_dims(n, nondata, gate_count.max(nondata), rng);
    let gadget = gadgetize_xgates_seven_carrier(source, n, rg_freq, prod, rng);
    circuit.num_wires = circuit.num_wires.max(gadget.num_wires);
    circuit.gates.extend(gadget.gates);
    commuting_shuffle(&mut circuit.gates, rng);
    circuit
}

/// XGate-source counterpart of the (removed, dead) CNOT-source
/// `gadgetize_with_slice_zero_ccnot_seven_carrier_distributed` wrapper.
pub fn gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed(
    source: &[XGate],
    n: usize,
    rg_freq: usize,
    gate_count: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let nondata = 6 * n + prod.band_size(n);
    let mut circuit = slice_zero_preblock_dims(n, nondata, gate_count.max(nondata), rng);
    let gadget = gadgetize_xgates_seven_carrier_distributed(source, n, rg_freq, prod, rng);
    circuit.num_wires = circuit.num_wires.max(gadget.num_wires);
    circuit.gates.extend(gadget.gates);
    commuting_shuffle(&mut circuit.gates, rng);
    circuit
}

/// Unshuffled A/B fixture for
/// [`gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed`].
pub fn gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed_unshuffled(
    source: &[XGate],
    n: usize,
    rg_freq: usize,
    gate_count: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let nondata = 6 * n + prod.band_size(n);
    let mut circuit = slice_zero_preblock_dims(n, nondata, gate_count.max(nondata), rng);
    let gadget =
        gadgetize_xgates_seven_carrier_distributed_unshuffled(source, n, rg_freq, prod, rng);
    circuit.num_wires = circuit.num_wires.max(gadget.num_wires);
    circuit.gates.extend(gadget.gates);
    circuit
}

/// Partitioned-128 XGate-source counterpart of the (removed, dead) CNOT-source
/// `gadgetize_with_slice_zero_ccnot_seven_carrier_distributed_partitioned` wrapper.
pub fn gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed_partitioned(
    source: &[XGate],
    n: usize,
    rg_freq: usize,
    gate_count: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let nondata = 6 * n + prod.band_size(n);
    let mut circuit = slice_zero_preblock_dims(n, nondata, gate_count.max(nondata), rng);
    let gadget =
        gadgetize_xgates_seven_carrier_distributed_partitioned(source, n, rg_freq, prod, rng);
    circuit.num_wires = circuit.num_wires.max(gadget.num_wires);
    circuit.gates.extend(gadget.gates);
    commuting_shuffle(&mut circuit.gates, rng);
    circuit
}

/// Unshuffled A/B fixture for
/// [`gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed_partitioned`].
pub fn gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed_partitioned_unshuffled(
    source: &[XGate],
    n: usize,
    rg_freq: usize,
    gate_count: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let nondata = 6 * n + prod.band_size(n);
    let mut circuit = slice_zero_preblock_dims(n, nondata, gate_count.max(nondata), rng);
    let gadget = gadgetize_xgates_seven_carrier_distributed_partitioned_unshuffled(
        source, n, rg_freq, prod, rng,
    );
    circuit.num_wires = circuit.num_wires.max(gadget.num_wires);
    circuit.gates.extend(gadget.gates);
    circuit
}

/// Partitioned-128 sliced-sandwich fixture with an explicit prefix of logical
/// values eligible as selector helpers.  For a `2*original_n` sliced sandwich,
/// pass `live_helper_prefix = original_n`: the upper half is fixed zero on the
/// intended input slice and must not be used to create nominal-only cells.
pub fn gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed_partitioned_live_prefix(
    source: &[XGate],
    n: usize,
    live_helper_prefix: usize,
    rg_freq: usize,
    gate_count: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let nondata = 6 * n + prod.band_size(n);
    let mut circuit = slice_zero_preblock_dims(n, nondata, gate_count.max(nondata), rng);
    let gadget = gadgetize_seven_carrier_source(
        source,
        n,
        rg_freq,
        prod,
        true,
        128,
        live_helper_prefix,
        0,
        true,
        rng,
    );
    circuit.num_wires = circuit.num_wires.max(gadget.num_wires);
    circuit.gates.extend(gadget.gates);
    commuting_shuffle(&mut circuit.gates, rng);
    circuit
}

/// Unshuffled A/B counterpart of
/// [`gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed_partitioned_live_prefix`].
pub fn gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed_partitioned_live_prefix_unshuffled(
    source: &[XGate],
    n: usize,
    live_helper_prefix: usize,
    rg_freq: usize,
    gate_count: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let nondata = 6 * n + prod.band_size(n);
    let mut circuit = slice_zero_preblock_dims(n, nondata, gate_count.max(nondata), rng);
    let gadget = gadgetize_seven_carrier_source(
        source,
        n,
        rg_freq,
        prod,
        true,
        128,
        live_helper_prefix,
        0,
        false,
        rng,
    );
    circuit.num_wires = circuit.num_wires.max(gadget.num_wires);
    circuit.gates.extend(gadget.gates);
    circuit
}

/// Floor-1024 refinement of the live-prefix partitioned experiment.  This is
/// intentionally a separate opt-in API so neither the legacy path nor the
/// floor-128 fixture changes cost unexpectedly.
pub fn gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed_partitioned_floor1024_live_prefix(
    source: &[XGate],
    n: usize,
    live_helper_prefix: usize,
    rg_freq: usize,
    gate_count: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let nondata = 6 * n + prod.band_size(n);
    let mut circuit = slice_zero_preblock_dims(n, nondata, gate_count.max(nondata), rng);
    let gadget = gadgetize_seven_carrier_source(
        source,
        n,
        rg_freq,
        prod,
        true,
        1024,
        live_helper_prefix,
        0,
        true,
        rng,
    );
    circuit.num_wires = circuit.num_wires.max(gadget.num_wires);
    circuit.gates.extend(gadget.gates);
    commuting_shuffle(&mut circuit.gates, rng);
    circuit
}

/// Unshuffled A/B counterpart of the floor-1024 live-prefix fixture.
pub fn gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed_partitioned_floor1024_live_prefix_unshuffled(
    source: &[XGate],
    n: usize,
    live_helper_prefix: usize,
    rg_freq: usize,
    gate_count: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let nondata = 6 * n + prod.band_size(n);
    let mut circuit = slice_zero_preblock_dims(n, nondata, gate_count.max(nondata), rng);
    let gadget = gadgetize_seven_carrier_source(
        source,
        n,
        rg_freq,
        prod,
        true,
        1024,
        live_helper_prefix,
        0,
        false,
        rng,
    );
    circuit.num_wires = circuit.num_wires.max(gadget.num_wires);
    circuit.gates.extend(gadget.gates);
    circuit
}

/// Source-integrated boundary-r10 counterpart of the floor-1024 experiment.
/// It requires the linear input band fill, partitions only the gates captured
/// from the single initial `inject_all` call over ten affine-independent band
/// helpers, and omits the terminal nondata mirror fill. The slice preblock is
/// unchanged. This remains an opt-in measurement fixture.
pub fn gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed_partitioned_floor1024_boundary_r10_live_prefix(
    source: &[XGate],
    n: usize,
    live_helper_prefix: usize,
    rg_freq: usize,
    gate_count: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let nondata = 6 * n + prod.band_size(n);
    let mut circuit = slice_zero_preblock_dims(n, nondata, gate_count.max(nondata), rng);
    let gadget = gadgetize_seven_carrier_source(
        source,
        n,
        rg_freq,
        prod,
        true,
        1024,
        live_helper_prefix,
        10,
        true,
        rng,
    );
    circuit.num_wires = circuit.num_wires.max(gadget.num_wires);
    circuit.gates.extend(gadget.gates);
    commuting_shuffle(&mut circuit.gates, rng);
    circuit
}

/// Unshuffled A/B counterpart of the floor-1024 boundary-r10 fixture.
pub fn gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed_partitioned_floor1024_boundary_r10_live_prefix_unshuffled(
    source: &[XGate],
    n: usize,
    live_helper_prefix: usize,
    rg_freq: usize,
    gate_count: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let nondata = 6 * n + prod.band_size(n);
    let mut circuit = slice_zero_preblock_dims(n, nondata, gate_count.max(nondata), rng);
    let gadget = gadgetize_seven_carrier_source(
        source,
        n,
        rg_freq,
        prod,
        true,
        1024,
        live_helper_prefix,
        10,
        false,
        rng,
    );
    circuit.num_wires = circuit.num_wires.max(gadget.num_wires);
    circuit.gates.extend(gadget.gates);
    circuit
}

/// Floor-4096 live-prefix body experiment, without boundary-r10 changes.
pub fn gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed_partitioned_floor4096_live_prefix(
    source: &[XGate],
    n: usize,
    live_helper_prefix: usize,
    rg_freq: usize,
    gate_count: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let nondata = 6 * n + prod.band_size(n);
    let mut circuit = slice_zero_preblock_dims(n, nondata, gate_count.max(nondata), rng);
    let gadget = gadgetize_seven_carrier_source(
        source,
        n,
        rg_freq,
        prod,
        true,
        4096,
        live_helper_prefix,
        0,
        true,
        rng,
    );
    circuit.num_wires = circuit.num_wires.max(gadget.num_wires);
    circuit.gates.extend(gadget.gates);
    commuting_shuffle(&mut circuit.gates, rng);
    circuit
}

/// Unshuffled A/B counterpart of the floor-4096 live-prefix body fixture.
pub fn gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed_partitioned_floor4096_live_prefix_unshuffled(
    source: &[XGate],
    n: usize,
    live_helper_prefix: usize,
    rg_freq: usize,
    gate_count: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let nondata = 6 * n + prod.band_size(n);
    let mut circuit = slice_zero_preblock_dims(n, nondata, gate_count.max(nondata), rng);
    let gadget = gadgetize_seven_carrier_source(
        source,
        n,
        rg_freq,
        prod,
        true,
        4096,
        live_helper_prefix,
        0,
        false,
        rng,
    );
    circuit.num_wires = circuit.num_wires.max(gadget.num_wires);
    circuit.gates.extend(gadget.gates);
    circuit
}

/// Combined floor-4096 body and source-integrated boundary-r10 experiment.
pub fn gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed_partitioned_floor4096_boundary_r10_live_prefix(
    source: &[XGate],
    n: usize,
    live_helper_prefix: usize,
    rg_freq: usize,
    gate_count: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let nondata = 6 * n + prod.band_size(n);
    let mut circuit = slice_zero_preblock_dims(n, nondata, gate_count.max(nondata), rng);
    let gadget = gadgetize_seven_carrier_source(
        source,
        n,
        rg_freq,
        prod,
        true,
        4096,
        live_helper_prefix,
        10,
        true,
        rng,
    );
    circuit.num_wires = circuit.num_wires.max(gadget.num_wires);
    circuit.gates.extend(gadget.gates);
    commuting_shuffle(&mut circuit.gates, rng);
    circuit
}

/// Unshuffled A/B counterpart of the combined floor-4096 boundary-r10 fixture.
pub fn gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed_partitioned_floor4096_boundary_r10_live_prefix_unshuffled(
    source: &[XGate],
    n: usize,
    live_helper_prefix: usize,
    rg_freq: usize,
    gate_count: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    let nondata = 6 * n + prod.band_size(n);
    let mut circuit = slice_zero_preblock_dims(n, nondata, gate_count.max(nondata), rng);
    let gadget = gadgetize_seven_carrier_source(
        source,
        n,
        rg_freq,
        prod,
        true,
        4096,
        live_helper_prefix,
        10,
        false,
        rng,
    );
    circuit.num_wires = circuit.num_wires.max(gadget.num_wires);
    circuit.gates.extend(gadget.gates);
    circuit
}

/// Unshuffled floor-4096/boundary-r10 fixture with an explicit hard terminal
/// boundary. The returned index is the first gate of `strip_all`, followed by
/// carrier routing and final decode. Every downstream ordering transform must
/// operate independently on `gates[..terminal_start]` and
/// `gates[terminal_start..]`; allowing a gate to cross the boundary invalidates
/// the measured construction.
pub fn gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed_partitioned_floor4096_boundary_r10_live_prefix_terminal_fenced_unshuffled(
    source: &[XGate],
    n: usize,
    live_helper_prefix: usize,
    rg_freq: usize,
    gate_count: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> (CnotCircuit, usize) {
    let nondata = 6 * n + prod.band_size(n);
    let mut circuit = slice_zero_preblock_dims(n, nondata, gate_count.max(nondata), rng);
    let preblock_len = circuit.gates.len();
    let (gadget, gadget_terminal_start) = gadgetize_seven_carrier_source_with_terminal_start(
        source,
        n,
        rg_freq,
        prod,
        true,
        4096,
        live_helper_prefix,
        10,
        false,
        rng,
    );
    circuit.num_wires = circuit.num_wires.max(gadget.num_wires);
    circuit.gates.extend(gadget.gates);
    let terminal_start = preblock_len + gadget_terminal_start;
    assert!(terminal_start < circuit.gates.len());
    (circuit, terminal_start)
}

/// Deterministic A/B artifact for the hard-terminal-fence experiment. It uses
/// 32 linear-time adjacent-commuting passes on the prefix only and returns the
/// same exact boundary so later stages can preserve it. The dedicated seed is
/// fixed for reproducibility and does not perturb gadget generation.
pub fn gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed_partitioned_floor4096_boundary_r10_live_prefix_terminal_fenced_adj32(
    source: &[XGate],
    n: usize,
    live_helper_prefix: usize,
    rg_freq: usize,
    gate_count: usize,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> (CnotCircuit, usize) {
    let (mut circuit, terminal_start) = gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed_partitioned_floor4096_boundary_r10_live_prefix_terminal_fenced_unshuffled(
        source,
        n,
        live_helper_prefix,
        rg_freq,
        gate_count,
        prod,
        rng,
    );
    let mut shuffle_rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(880_033);
    adjacent_commuting_swap_passes(&mut circuit.gates[..terminal_start], 32, &mut shuffle_rng);
    (circuit, terminal_start)
}

pub fn gadgetize_xgates_with_slice_zero_ccnot(
    source: &[XGate],
    n: usize,
    rg_freq: usize,
    gate_count: usize,
    masks: &MaskConfig,
    prod: &ProdConfig,
    rng: &mut impl Rng,
) -> CnotCircuit {
    // The band is part of the slice: the preblock must see its width.
    let mut circuit = slice_zero_ccnot_preblock(n, prod.band_size(n), gate_count, rng);
    let gadget = gadgetize_xgates(source, n, rg_freq, masks, prod, rng);
    circuit.num_wires = circuit.num_wires.max(gadget.num_wires);
    circuit.gates.extend(gadget.gates);
    // Shuffle across the preblock/gadget seam too, so the slice block does
    // not sit as a contiguous prefix.
    commuting_shuffle(&mut circuit.gates, rng);
    circuit
}

/// The legacy "compose_A" sandwich (pure g57, 2n wires):
///   A = [ C on 0..n-1 ] ++ [ n six-g57-gate CNOT copies w_{n+i} ^= w_i ]
///       ++ [ D on 0..n-1 ],
/// realizing (x, z) -> (D(C(x)), z ^ C(x)). Each copy block is the minimal
/// six-g57 CNOT with helper wire (i+1) mod n (identity on control and helper).
/// This is the OLD sandwiching mechanism, gadgetized by the legacy `gadgetize`.
pub fn compose_a(c: &CircuitSeq, d: &CircuitSeq, n: usize) -> CircuitSeq {
    assert!(n >= 2, "compose_a requires n >= 2");
    assert!(2 * n <= u16::MAX as usize, "too many wires");
    // Six-gate g57 CNOT, local wires 0=helper, 1=control, 2=target.
    const CNOT6: [[usize; 3]; 6] = [
        [0, 2, 1],
        [2, 0, 1],
        [1, 0, 2],
        [2, 1, 0],
        [0, 1, 2],
        [1, 2, 0],
    ];
    let mut gates: Vec<[u16; 3]> = c.gates.clone();
    for i in 0..n {
        let map = [(i + 1) % n, i, n + i]; // helper, control, target
        for g in CNOT6 {
            gates.push([map[g[0]] as u16, map[g[1]] as u16, map[g[2]] as u16]);
        }
    }
    gates.extend(d.gates.iter().copied());
    CircuitSeq { gates }
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

/// One slice block for the sliced-sandwich construction: `s` gates whose
/// targets are all in the first half (wires 0..n), each reading at least one
/// second-half wire (n..2n) with positive polarity, so the whole block is
/// dead when the second half is zero. ~1/3 are CNOTs `x_i ^= a_j` (control
/// in the second half); the rest are CCNOTs `x_i ^= x_j & a_k` (one control
/// per half).
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
/// half = y):
///
///   A = [ C interleaved with S1 ] ; N ; [ D interleaved with S2 ]
///
/// where C is the source circuit, D is a fresh random circuit of `m` g57
/// gates on wires 0..n (same design as the C block), N copies the first
/// half into the second (`y ^= x`, n CNOTs),
/// and S1, S2 are independent slice blocks of `s` gates each
/// (`sandwich_slice_gates`), randomly interleaved with C and D respectively.
/// A final float stage then slides each N CNOT in a random direction as far
/// as commutation allows, dissolving the middle column into a band (see the
/// stage comment in [`sliced_sandwich_with_d`]).
///
/// On the zero slice the second half carries the answer:
///   A(x, 0) = (junk, C(x)),
/// because S1 is dead during C (second half still 0) and S2, though live
/// during D (second half already holds C(x)), only targets the junk first
/// half. Symmetrically the inverse gives A^-1(p, 0) = (junk, D^-1(p)): there
/// S2 is dead and S1 fires, so neither direction hands out its computation
/// on a wrong slice, and neither reveals the other's function in the clear.
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

/// Sliced sandwich with an explicit D block (given as XGates on wires 0..n).
/// Used when C and D must be shared with another pipeline (e.g. an A/B against
/// the legacy `compose_a` sandwich on the same C, D). See
/// [`sliced_sandwich_cnot`] for the semantics.
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
        d_gates
            .iter()
            .all(|g| { (g.target as usize) < n && g.ctrls.iter().all(|&(w, _)| (w as usize) < n) }),
        "D wire outside 0..n"
    );
    let total = 2 * n;

    // Block 1: C (the source) interleaved with S1.
    let c_gates: Vec<XGate> = main.gates.iter().map(|&g| XGate::from_g57(g)).collect();
    let s1 = sandwich_slice_gates(n, s, rng);
    let mut out = random_interleave(c_gates, s1, rng);

    // N step: y ^= x.
    for i in 0..n {
        out.push(XGate::cnot((n + i) as u16, i as u16));
    }

    // Block 2: D interleaved with S2.
    let s2 = sandwich_slice_gates(n, s, rng);
    out.extend(random_interleave(d_gates.to_vec(), s2, rng));

    // Final float stage: the N column is the sandwich's most
    // structure-revealing part (the C|N|D seam). Each of its CNOTs is
    // ASSIGNED an independent random direction, registered up front, and
    // then floats in that direction as far as commutation allows — deep
    // into C/S1 or D/S2 wherever its wires stay cold — dissolving the
    // column into a wide band before gadgetizing. The registered direction
    // matters: float passes repeat until a fixpoint, and a gate always
    // continues in ITS direction, so gates never oscillate and any gate
    // unblocked by another's departure keeps drifting the same way. The N
    // gates are exactly the gates targeting the second half (C, D, S1, S2
    // all target 0..n) and mutually commute (they pass each other freely);
    // every hop is a commuting swap, so A's function and all slice/inverse
    // guarantees are unchanged.
    let mut floaters: Vec<(usize, bool)> = (0..out.len())
        .filter(|&i| (out[i].target as usize) >= n)
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

    CnotCircuit {
        gates: out,
        num_wires: total,
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
    shoot_random_gate(&mut source, source_rounds);
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
    use crate::circuit::xgate::eval_u64;
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
                        let mut ctrls: crate::circuit::xgate::Lits = gate
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
            let transformed = gadgetize_cnot(
                &main,
                n,
                2,
                &MaskConfig::off(),
                &ProdConfig::off(),
                &mut rng,
            );
            assert_eq!(transformed.num_wires, 2 * n);
            let mask = (1u64 << n) - 1;
            for input in 0..(1u64 << (2 * n)) {
                let expected = main.evaluate((input & mask) as usize) as u64 & mask;
                assert_eq!(eval_u64(&transformed.gates, input) & mask, expected);
            }
        }
    }

    /// An n=4 body (distinct operands per triple) — wide enough to leave a
    /// live source pool under partial coverage, and long enough for the mask
    /// top-up to run past its taper.
    const MASKED_TEST_N: usize = 4;
    fn masked_test_main() -> CircuitSeq {
        let mut gates = Vec::new();
        for _ in 0..4 {
            for &g in &[
                [0u16, 1, 2],
                [3, 2, 0],
                [1, 3, 2],
                [2, 0, 3],
                [0, 3, 1],
                [3, 1, 0],
            ] {
                gates.push(g);
            }
        }
        CircuitSeq { gates }
    }

    /// A random-ish g57 body on `n >= 4` wires, for tests that need a width
    /// the 4-wire fixture cannot give (a wire census wants room for a band).
    fn masked_test_main_wide(n: usize) -> CircuitSeq {
        let mut rng = StdRng::seed_from_u64(0x9e37_9b91);
        let mut gates = Vec::new();
        for _ in 0..4 * n {
            let a = rng.random_range(0..n) as u16;
            let b = loop {
                let w = rng.random_range(0..n) as u16;
                if w != a {
                    break w;
                }
            };
            let c = loop {
                let w = rng.random_range(0..n) as u16;
                if w != a && w != b {
                    break w;
                }
            };
            gates.push([a, b, c]);
        }
        CircuitSeq { gates }
    }

    fn masked_test_config() -> MaskConfig {
        // cov 0.75 keeps an unmasked source pool alive on this width.
        MaskConfig {
            cov: 0.75,
            k: 2,
            depth: 2,
            taper: Some(0),
        }
    }

    #[test]
    fn emit_poly_add_realizes_random_polynomials() {
        let total = 8usize;
        let target = 3u16;
        let mut rng = StdRng::seed_from_u64(0x901f_0000);
        for _ in 0..200 {
            let mut poly = WirePoly::default();
            for _ in 0..rng.random_range(1..5usize) {
                let size = rng.random_range(0..=3usize);
                let mut m = Vec::new();
                while m.len() < size {
                    let w = rng.random_range(0..total) as u16;
                    if w != target && !m.contains(&w) {
                        m.push(w);
                    }
                }
                poly.toggle(m);
            }
            let mut gates = Vec::new();
            emit_poly_add(target, &poly, total, &mut rng, &mut gates);
            assert!(
                gates.iter().all(|g| !g.ctrls.is_empty()),
                "emit_poly_add must never emit a bare X"
            );
            for input in 0..(1u64 << total) {
                let expected = input ^ ((poly.eval_u64(input) as u64) << target);
                assert_eq!(
                    eval_u64(&gates, input),
                    expected,
                    "poly={poly:?} input={input:#x}"
                );
            }
        }
    }

    /// XOR of a value's two carrier bits in a packed state — the only
    /// gadget-visible quantity; mask compensation preserves this, not the
    /// individual carriers (a flush may target either carrier, dirtying both
    /// by an equal amount like an RG3 refresh).
    fn pair_xor(state: u64, pair: (usize, usize)) -> u64 {
        ((state >> pair.0) ^ (state >> pair.1)) & 1
    }

    #[test]
    fn mask_inject_then_flush_preserves_every_value() {
        // Inject a stack per value, then flush_all: each value's pair-XOR must
        // be restored for every input and every realization draw. The
        // value-sourced cascade-free ledger keeps every source value at its
        // injection value, so the compensation is exact regardless of the
        // (either-carrier) flush order.
        let n = 4;
        let total = 2 * n;
        let pairs: Vec<(usize, usize)> = (0..n).map(|v| (v, v + n)).collect();
        // No value is ever a CG target here (pure ledger exercise), so every
        // value is an ideal (never-disturbed) source.
        let targets = vec![Vec::new(); n];
        for seed in 0..64u64 {
            let mut rng = StdRng::seed_from_u64(0x1f1a_0000 + seed);
            let state = GadgetState {
                n,
                pairs: pairs.clone(),
            };
            // cov 0.75 keeps a live source pool; k=2 stacks.
            let cfg = MaskConfig {
                cov: 0.75,
                k: 2,
                depth: 2,
                taper: Some(0),
            };
            let mut ledger = MaskLedger::new(n, &cfg, targets.clone(), &mut rng);
            let mut gates = Vec::new();
            for _round in 0..3 {
                for value in 0..n {
                    ledger.inject(value, 0, &state, total, &mut rng, &mut gates);
                }
            }
            ledger.flush_all(&state, total, &mut rng, &mut gates);
            assert!(ledger.masks.is_empty());
            assert!(ledger.injected > 0, "seed={seed}: nothing injected");
            for input in 0..(1u64 << total) {
                let out = eval_u64(&gates, input);
                for &p in &pairs {
                    assert_eq!(
                        pair_xor(out, p),
                        pair_xor(input, p),
                        "seed={seed} pair={p:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn mask_peek_bracket_restores_the_masked_value() {
        // Un-mask (before_cg) then re-mask (after_cg) around a value read must
        // (a) leave the value's pair-XOR exactly masked again afterward, and
        // (b) expose the TRUE value in between (what the vanilla CG needs).
        let n = 4;
        let total = 2 * n;
        let pairs: Vec<(usize, usize)> = (0..n).map(|v| (v, v + n)).collect();
        let targets = vec![Vec::new(); n];
        for seed in 0..64u64 {
            let mut rng = StdRng::seed_from_u64(0x9ee0_0000 + seed);
            let state = GadgetState {
                n,
                pairs: pairs.clone(),
            };
            let cfg = MaskConfig {
                cov: 0.75,
                k: 1,
                depth: 2,
                taper: Some(0),
            };
            let mut ledger = MaskLedger::new(n, &cfg, targets.clone(), &mut rng);
            // Mask value 0 (sources chosen from the unmasked pool 1..4).
            let mut pre = Vec::new();
            if !ledger.inject(0, 0, &state, total, &mut rng, &mut pre) {
                continue;
            }
            // Peek value 0 as a read of a gate whose target is NOT a source of
            // value 0's mask — otherwise before_cg legitimately flushes it
            // (source recomputed) instead of peeking, a different path.
            let (s0, s1) = ledger.masks[0].sources;
            let target = (1..n).find(|t| *t != s0 && *t != s1).unwrap();
            let mut br_open = Vec::new();
            ledger.before_cg(&[0], target, &state, total, &mut rng, &mut br_open);
            let mut br_close = Vec::new();
            ledger.after_cg(&[0], &state, total, &mut rng, &mut br_close);
            for input in 0..(1u64 << total) {
                let after_inject = eval_u64(&pre, input);
                // Inside the bracket: value 0 reads TRUE (un-masked).
                let mut peeked = after_inject;
                peeked = eval_u64(&br_open, peeked);
                assert_eq!(
                    pair_xor(peeked, pairs[0]),
                    pair_xor(input, pairs[0]),
                    "seed={seed}: value not reconstructed inside the peek"
                );
                // After re-mask: value 0 is masked again (== state after inject).
                let closed = eval_u64(&br_close, peeked);
                assert_eq!(
                    pair_xor(closed, pairs[0]),
                    pair_xor(after_inject, pairs[0]),
                    "seed={seed}: peek not undone"
                );
            }
        }
    }

    #[test]
    fn masked_gadgetize_cnot_preserves_the_first_n_wires() {
        let n = MASKED_TEST_N;
        let main = masked_test_main();
        let mask = (1u64 << n) - 1;
        for seed in 0..8u64 {
            let mut rng = StdRng::seed_from_u64(0x3a5c_0000 + seed);
            let masked = gadgetize_cnot(
                &main,
                n,
                2,
                &masked_test_config(),
                &ProdConfig::off(),
                &mut rng,
            );
            assert_eq!(masked.num_wires, 2 * n);
            assert!(
                masked.gates.iter().all(|g| !g.ctrls.is_empty()),
                "masked body must not contain a bare X"
            );
            for input in 0..(1u64 << (2 * n)) {
                let expected = main.evaluate((input & mask) as usize) as u64 & mask;
                assert_eq!(eval_u64(&masked.gates, input) & mask, expected);
            }
            // Same seed, masks off: the masked build must actually have paid
            // mask gates into the body.
            let mut rng = StdRng::seed_from_u64(0x3a5c_0000 + seed);
            let plain = gadgetize_cnot(
                &main,
                n,
                2,
                &MaskConfig::off(),
                &ProdConfig::off(),
                &mut rng,
            );
            assert!(
                masked.gates.len() > plain.gates.len(),
                "seed={seed}: masks enabled but no mask gates emitted"
            );
        }
    }

    #[test]
    fn masked_slice_zero_gadgetize_matches_on_the_zero_slice() {
        let n = MASKED_TEST_N;
        let main = masked_test_main();
        let mask = (1u64 << n) - 1;
        for seed in 0..4u64 {
            let mut rng = StdRng::seed_from_u64(0x3a5d_0000 + seed);
            let transformed = gadgetize_with_slice_zero_ccnot(
                &main,
                n,
                2,
                6 * n,
                &masked_test_config(),
                &ProdConfig::off(),
                &mut rng,
            );
            for x in 0..=mask {
                let expected = main.evaluate(x as usize) as u64 & mask;
                assert_eq!(eval_u64(&transformed.gates, x) & mask, expected);
            }
        }
    }

    #[test]
    fn masked_gadgetize_xgates_preserves_the_low_wires() {
        let n = MASKED_TEST_N;
        let mask = (1u64 << n) - 1;
        let source = vec![
            XGate::from_g57([0, 1, 2]),
            XGate::cnot(0, 3),
            XGate::conj(2, [(0u16, true), (1u16, true)]).unwrap(),
            XGate::conj(1, [(3u16, false)]).unwrap(),
            XGate::from_g57([3, 0, 1]),
            XGate::cnot(1, 0),
            XGate::from_g57([0, 2, 3]),
            XGate::conj(0, [(1u16, false), (2u16, true)]).unwrap(),
            XGate::from_g57([1, 3, 2]),
            XGate::cnot(2, 1),
        ];
        for seed in 0..8u64 {
            let mut rng = StdRng::seed_from_u64(0x3a5e_0000 + seed);
            let g = gadgetize_xgates(
                &source,
                n,
                2,
                &masked_test_config(),
                &ProdConfig::off(),
                &mut rng,
            );
            assert_eq!(g.num_wires, 2 * n);
            for input in 0..(1u64 << (2 * n)) {
                let expected = eval_u64(&source, input & mask) & mask;
                assert_eq!(
                    eval_u64(&g.gates, input) & mask,
                    expected,
                    "input={input:#x}"
                );
            }
        }
    }

    fn prod_test_config() -> ProdConfig {
        ProdConfig {
            k: 2,
            deg: 2,
            k_hi: 0,
            deg_hi: 3,
            band: 6,
            rsrc: 1,
            ..ProdConfig::off()
        }
    }

    /// Test-side decode under the product-share ledger state: pair-XOR of the
    /// value's carriers, XOR each registered slot's product PROD(w_j ^ a_j),
    /// XOR c.
    fn prod_decode(
        state: u64,
        value: usize,
        pairs: &[(usize, usize)],
        slots: &[Vec<ProdSlot>],
        consts: &[bool],
        loc: &[u16],
    ) -> u64 {
        // Single-carrier builds record `pairs[v] = (w, w)`, and XORing the wire
        // with itself would decode every value as 0 -- so read one carrier when
        // the pair collapses, both when it does not.
        let (c0, c1) = pairs[value];
        let mut v = if c0 == c1 {
            (state >> c0) & 1
        } else {
            ((state >> c0) ^ (state >> c1)) & 1
        };
        for slot in &slots[value] {
            // Factors name band VARIABLES; `loc` says where each one lives.
            let factor = slot
                .factors
                .iter()
                .all(|&(b, a)| ((state >> loc[b as usize]) & 1 != 0) ^ a);
            v ^= factor as u64;
        }
        v ^ consts[value] as u64
    }

    fn eval_anf_atoms(state: u64, atoms: &[Vec<(u16, bool)>]) -> u64 {
        atoms.iter().fold(0u64, |value, atom| {
            value
                ^ atom
                    .iter()
                    .all(|&(wire, polarity)| ((state >> wire) & 1 != 0) == polarity)
                    as u64
        })
    }

    /// Truth table of every physical wire at every circuit prefix, packed one
    /// column at a time.  Keeping the whole Boolean function (rather than a
    /// sample or a correlation) lets the space-time checks below prove an
    /// identity over every possible incoming dirty state.
    fn prefix_wire_signatures(gates: &[XGate], wires: usize) -> Vec<Vec<u64>> {
        let rows = 1usize << wires;
        let words = (rows + 63) / 64;
        let mut columns = vec![vec![0u64; words]; (gates.len() + 1) * wires];
        for input in 0..rows {
            let mut physical = input as u64;
            for prefix in 0..=gates.len() {
                for wire in 0..wires {
                    if (physical >> wire) & 1 != 0 {
                        columns[prefix * wires + wire][input / 64] |= 1u64 << (input % 64);
                    }
                }
                if let Some(gate) = gates.get(prefix) {
                    physical = gate.apply_u64(physical);
                }
            }
        }
        columns
    }

    fn prod_decode_signature(
        wires: usize,
        value: usize,
        pairs: &[(usize, usize)],
        slots: &[Vec<ProdSlot>],
        consts: &[bool],
        loc: &[u16],
    ) -> Vec<u64> {
        let rows = 1usize << wires;
        let mut signature = vec![0u64; (rows + 63) / 64];
        for input in 0..rows {
            if prod_decode(input as u64, value, pairs, slots, consts, loc) != 0 {
                signature[input / 64] |= 1u64 << (input % 64);
            }
        }
        signature
    }

    fn xor_signatures(left: &[u64], right: &[u64]) -> Vec<u64> {
        left.iter().zip(right).map(|(&a, &b)| a ^ b).collect()
    }

    fn signature_pivot(value: &[u64]) -> Option<usize> {
        value.iter().enumerate().rev().find_map(|(word, &bits)| {
            (bits != 0).then(|| word * 64 + (63 - bits.leading_zeros() as usize))
        })
    }

    /// Does `target` belong to the affine span of these simultaneous wires?
    fn affine_span_contains(columns: &[Vec<u64>], target: &[u64]) -> bool {
        let bit_count = target.len() * 64;
        let mut basis: Vec<Option<Vec<u64>>> = vec![None; bit_count];
        let mut insert = |mut value: Vec<u64>| {
            while let Some(pivot) = signature_pivot(&value) {
                if let Some(row) = &basis[pivot] {
                    for (word, &rhs) in value.iter_mut().zip(row) {
                        *word ^= rhs;
                    }
                } else {
                    basis[pivot] = Some(value);
                    return;
                }
            }
        };
        for column in columns {
            insert(column.clone());
        }
        insert(vec![u64::MAX; target.len()]);

        let mut residual = target.to_vec();
        while let Some(pivot) = signature_pivot(&residual) {
            let Some(row) = &basis[pivot] else {
                return false;
            };
            for (word, &rhs) in residual.iter_mut().zip(row) {
                *word ^= rhs;
            }
        }
        true
    }

    /// Find the specific short space-time identity at issue:
    ///
    /// logical operand = its entry carrier XOR wire@p XOR wire@q XOR constant.
    ///
    /// This search is deliberately blind to which wires the fold borrowed and
    /// where its gather/strip boundaries are.
    fn same_wire_space_time_witness(
        trace: &[Vec<u64>],
        wires: usize,
        needed_delta: &[u64],
    ) -> Option<(usize, usize, usize, bool)> {
        let prefixes = trace.len() / wires;
        for wire in 0..wires {
            for left in 0..prefixes {
                for right in left + 1..prefixes {
                    let delta =
                        xor_signatures(&trace[left * wires + wire], &trace[right * wires + wire]);
                    if delta == needed_delta {
                        return Some((wire, left, right, false));
                    }
                    if delta
                        .iter()
                        .zip(needed_delta)
                        .all(|(&observed, &needed)| observed == !needed)
                    {
                        return Some((wire, left, right, true));
                    }
                }
            }
        }
        None
    }

    #[test]
    fn prod_fold_cg_applies_the_virtual_gate_share_natively() {
        // Manually built ledger; fold one g57, one CNOT, one CCNOT-with-
        // polarity, one X: for every input, the target's decode transitions
        // by exactly the virtual gate while every other value is untouched —
        // and no emitted gate writes anything but the target's carriers.
        let n = 3;
        let carrier_total = 2 * n;
        let pairs = vec![(0usize, 1usize), (2, 3), (4, 5)];
        let band = 4usize; // wires 6..10
        let total = carrier_total + band;
        let cfg = ProdConfig {
            k: 1,
            deg: 2,
            k_hi: 0,
            deg_hi: 3,
            band,
            rsrc: 0,
            ..ProdConfig::off()
        };
        let sources: Vec<XGate> = vec![
            XGate::from_g57([0, 1, 2]),
            XGate::cnot(1, 2),
            XGate::conj(2, [(0u16, false), (1u16, true)]).unwrap(),
            XGate::x_gate(0),
        ];
        for seed in 0..32u64 {
            let mut rng = StdRng::seed_from_u64(0x9d0d_0000 + seed);
            let state = GadgetState {
                n,
                pairs: pairs.clone(),
            };
            let mut ledger = ProdLedger::new(n, &cfg, carrier_total, None);
            let mut ramp = Vec::new();
            ledger.inject_all(&state, &mut rng, &mut ramp);
            for gate in &sources {
                let slots_before = ledger.slots.clone();
                let consts_before = ledger.consts.clone();
                let mut fold = Vec::new();
                ledger.fold_cg(gate, &state, &mut rng, &mut fold);
                let t = gate.target as usize;
                for g in &fold {
                    assert!(
                        g.target as usize == pairs[t].0 || g.target as usize == pairs[t].1,
                        "fold writes outside the target's carriers"
                    );
                    assert!(!g.ctrls.is_empty(), "fold emitted a bare X");
                }
                for input in 0..(1u64 << total) {
                    let before: Vec<u64> = (0..n)
                        .map(|v| {
                            prod_decode(
                                input,
                                v,
                                &pairs,
                                &slots_before,
                                &consts_before,
                                &ledger.loc,
                            )
                        })
                        .collect();
                    let out_state = eval_u64(&fold, input);
                    let after: Vec<u64> = (0..n)
                        .map(|v| {
                            prod_decode(
                                out_state,
                                v,
                                &pairs,
                                &ledger.slots,
                                &ledger.consts,
                                &ledger.loc,
                            )
                        })
                        .collect();
                    // The virtual gate on the decoded values.
                    let fires = gate
                        .ctrls
                        .iter()
                        .all(|&(w, pol)| (before[w as usize] != 0) == pol)
                        ^ gate.comp;
                    for v in 0..n {
                        let expected = before[v] ^ ((v == t && fires) as u64);
                        assert_eq!(after[v], expected, "seed={seed} gate={gate:?} value={v}");
                    }
                }
                // ledger state changed only in consts (slots untouched by CGs)
                assert_eq!(slots_before, ledger.slots);
            }
        }
    }

    #[test]
    fn prod_degree_three_masks_round_trip_and_widen_fragments() {
        // Tower level: deg=3 masks. Functionality must be exact, and the fold
        // must actually emit wider (>= width-3) fragments — the algebraic
        // signature of a degree-3 mask term.
        let n = MASKED_TEST_N;
        let main = masked_test_main();
        let mask = (1u64 << n) - 1;
        let cfg = ProdConfig {
            k: 2,
            deg: 3,
            k_hi: 0,
            deg_hi: 3,
            band: 8,
            rsrc: 1,
            ..ProdConfig::off()
        };
        for seed in 0..6u64 {
            let mut rng = StdRng::seed_from_u64(0x0e63_0000 + seed);
            let g = gadgetize_cnot(&main, n, 2, &MaskConfig::off(), &cfg, &mut rng);
            assert_eq!(g.num_wires, 2 * n + 8);
            assert!(
                g.gates.iter().all(|gate| !gate.ctrls.is_empty()),
                "no bare X"
            );
            let max_width = g.gates.iter().map(|gate| gate.width()).max().unwrap();
            assert!(
                max_width >= 3,
                "seed={seed}: deg-3 masks must widen fragments (got {max_width})"
            );
            for input in 0..(1u64 << g.num_wires) {
                let expected = main.evaluate((input & mask) as usize) as u64 & mask;
                assert_eq!(eval_u64(&g.gates, input) & mask, expected, "seed={seed}");
            }
        }
    }

    #[test]
    fn emit_g57_form_realizes_the_exact_conjunction() {
        // All 1- and 2-literal polarity combos, both variant draws: the gate
        // run must add conj(lits) ^ konst to the target and nothing else.
        for seed in 0..64u64 {
            let mut rng = StdRng::seed_from_u64(0x657f_0000 + seed);
            for lits in [
                vec![(1u16, true)],
                vec![(1u16, false)],
                vec![(1u16, true), (2u16, true)],
                vec![(1u16, false), (2u16, true)],
                vec![(1u16, true), (2u16, false)],
                vec![(1u16, false), (2u16, false)],
            ] {
                let mut gates = Vec::new();
                let konst = emit_g57_form(0, &lits, &mut rng, &mut gates);
                for input in 0..8u64 {
                    let expected_fire = lits.iter().all(|&(w, p)| ((input >> w) & 1 != 0) == p);
                    let out_state = eval_u64(&gates, input);
                    let expected = input ^ (expected_fire as u64 ^ konst as u64);
                    assert_eq!(out_state, expected, "seed={seed} lits={lits:?}");
                }
            }
        }
    }

    #[test]
    fn emit_narrow_fragment_ladders_exactly_over_dirty_borrows() {
        // Widths 3..=6 over mixed polarities, caps 2 and 3. The ladder must
        // add exactly conj(lits) ^ konst to the target and leave every other
        // wire — including the DIRTY borrowed carriers — untouched, for EVERY
        // input state (no clean-ancilla assumption anywhere), while staying
        // within the width cap.
        let total = 12usize; // literals on 1..=6, borrows drawn from 0..12
        for seed in 0..48u64 {
            let mut rng = StdRng::seed_from_u64(0x1add_0000 + seed);
            for cap in 2..=3usize {
                for width in 3..=6usize {
                    let lits: Vec<(u16, bool)> = (1..=width as u16)
                        .map(|w| (w, rng.random::<bool>()))
                        .collect();
                    let mut gates = Vec::new();
                    // Role set = every wire: this unit test has no ledger, and
                    // the point here is the ladder algebra, not the pool policy.
                    let all: Vec<u16> = (0..total as u16).collect();
                    let konst = emit_narrow_fragment(
                        0,
                        &lits,
                        cap,
                        &all,
                        total,
                        &[],
                        &[],
                        1,
                        &mut rng,
                        &mut gates,
                    );
                    assert!(
                        gates.iter().all(|g| g.width() <= cap),
                        "seed={seed} cap={cap} width={width}: ladder exceeded the cap"
                    );
                    for input in 0..(1u64 << total) {
                        let expected_fire = lits.iter().all(|&(w, p)| ((input >> w) & 1 != 0) == p);
                        let out_state = eval_u64(&gates, input);
                        let expected = input ^ (expected_fire as u64 ^ konst as u64);
                        assert_eq!(
                            out_state, expected,
                            "seed={seed} cap={cap} width={width} input={input:#x}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn prod_narrow_fold_cg_is_share_native_and_two_control() {
        // The narrow fold: same share-native contract as the wide fold — the
        // target's decode transitions by exactly the virtual gate and every
        // other value is untouched, for EVERY input (dirty borrows, no clean
        // ancilla) — with every gate at most 2 controls.
        let n = 3;
        let carrier_total = 2 * n;
        let pairs = vec![(0usize, 1usize), (2, 3), (4, 5)];
        let band = 6usize; // wires 6..12; scratch 12..16
        let cfg = ProdConfig {
            k: 1,
            deg: 2,
            k_hi: 1,
            deg_hi: 3,
            band,
            rsrc: 0,
            max_width: 2,
            fill_nl: 0,
            roll: 0,
            src_dist: 0,
            src_horizon: 0,
            src_lo: 0,
            src_hi: 0,
            fill_pivots: 0,
            g57_narrow: 0,
            ladder_cap: 0,
            cg_jitter: 0,
            rung_menu: 0,

            epoch: 0,
            refill_data: 0,
            single: 0,
            gray_fold: 0,
            swap_refresh: 0,
            close_slice: 0,
        };
        let live = carrier_total + band; // the whole wire space: no pinned wires
        let sources: Vec<XGate> = vec![
            XGate::from_g57([0, 1, 2]),
            XGate::cnot(1, 2),
            XGate::conj(2, [(0u16, false), (1u16, true)]).unwrap(),
            XGate::x_gate(0),
        ];
        for seed in 0..16u64 {
            let mut rng = StdRng::seed_from_u64(0x9d0e_0000 + seed);
            let state = GadgetState {
                n,
                pairs: pairs.clone(),
            };
            let mut ledger = ProdLedger::new(n, &cfg, carrier_total, None);
            let mut ramp = Vec::new();
            ledger.inject_all(&state, &mut rng, &mut ramp);
            for gate in &sources {
                let slots_before = ledger.slots.clone();
                let consts_before = ledger.consts.clone();
                let mut fold = Vec::new();
                ledger.fold_cg(gate, &state, &mut rng, &mut fold);
                let t = gate.target as usize;
                for g in &fold {
                    assert!(g.width() <= 2, "narrow fold emitted a wide gate");
                }
                // A ladder rung may borrow ANY wire, band included — the
                // double sweep restores it, which the next assertion checks
                // over the whole input domain.
                // Net effect must live ENTIRELY on the target's two carriers:
                // every dirty borrow is restored, for every input.
                let touched = (1u64 << pairs[t].0) | (1u64 << pairs[t].1);
                for input in 0..(1u64 << live) {
                    let before: Vec<u64> = (0..n)
                        .map(|v| {
                            prod_decode(
                                input,
                                v,
                                &pairs,
                                &slots_before,
                                &consts_before,
                                &ledger.loc,
                            )
                        })
                        .collect();
                    let out_state = eval_u64(&fold, input);
                    assert_eq!(
                        (out_state ^ input) & !touched,
                        0,
                        "seed={seed}: a borrowed wire was not restored"
                    );
                    let after: Vec<u64> = (0..n)
                        .map(|v| {
                            prod_decode(
                                out_state,
                                v,
                                &pairs,
                                &ledger.slots,
                                &ledger.consts,
                                &ledger.loc,
                            )
                        })
                        .collect();
                    let fires = gate
                        .ctrls
                        .iter()
                        .all(|&(w, pol)| (before[w as usize] != 0) == pol)
                        ^ gate.comp;
                    for v in 0..n {
                        let expected = before[v] ^ ((v == t && fires) as u64);
                        assert_eq!(after[v], expected, "seed={seed} gate={gate:?} value={v}");
                    }
                }
                assert_eq!(slots_before, ledger.slots);
            }
        }
    }

    #[test]
    fn prod_micro_gray_quartic_gather_is_exact_over_two_dirty_helpers() {
        // acc=0, helpers=1,2, quartic inputs=3..6. Exercise every literal
        // polarity and every incoming state: only acc may change, by exactly
        // the requested quartic, and both arbitrary-dirty helpers come back.
        for polarity_mask in 0u8..16 {
            let atom: Vec<(u16, bool)> = (0..4)
                .map(|index| (3 + index, polarity_mask & (1 << index) != 0))
                .collect();
            let mut rng = StdRng::seed_from_u64(0x4d47_0000 + polarity_mask as u64);
            let mut seen = std::collections::HashMap::new();
            let mut gates = Vec::new();
            assert!(!emit_micro_atom_onto(
                0,
                &atom,
                MicroAtomPlan {
                    helper0: 1,
                    helper1: 2,
                    pivot: 0,
                },
                1,
                &mut seen,
                &mut rng,
                &mut gates,
            ));
            assert_eq!(gates.len(), 8, "quartic micro ladder cost drifted");
            assert!(gates.iter().all(|gate| gate.width() <= 2));
            for input in 0u64..128 {
                let fire = atom
                    .iter()
                    .all(|&(wire, polarity)| ((input >> wire) & 1 != 0) == polarity);
                assert_eq!(
                    eval_u64(&gates, input),
                    input ^ fire as u64,
                    "polarity={polarity_mask:#06b} input={input:#09b}"
                );
            }
        }
    }

    #[test]
    fn prod_micro_gray_trace_needs_all_four_share_deltas() {
        // Four one-atom formal shares, each gathered and immediately stripped
        // from the same dirty accumulator. The trace contains each individual
        // row delta, but never the complete operand on one before/after pair.
        let cfg = ProdConfig::off();
        let ledger = ProdLedger::new(1, &cfg, 6, None);
        let atoms: Vec<Vec<(u16, bool)>> = (1..=4u16).map(|wire| vec![(wire, true)]).collect();
        let mut rng = StdRng::seed_from_u64(0x4d47_1001);
        let shares = micro_partition_atoms(&atoms, &mut rng).expect("four atoms make four rows");
        let plan = [MicroAtomPlan {
            helper0: 5,
            helper1: 5,
            pivot: 0,
        }];
        let mut seen = std::collections::HashMap::new();
        let mut gates = Vec::new();
        let mut intervals = Vec::new();
        for share in &shares {
            let before = gates.len();
            ledger.gather_micro_share_exact(0, share, &plan, 5, &mut seen, &mut rng, &mut gates);
            let after = gates.len();
            intervals.push((before, after));
            ledger.gather_micro_share_exact(0, share, &plan, 5, &mut seen, &mut rng, &mut gates);
        }
        let wires = 6;
        let trace = prefix_wire_signatures(&gates, wires);
        let mut operand = vec![0u64; 1];
        for wire in 1..=4 {
            operand = xor_signatures(&operand, &trace[wire]);
        }
        let deltas: Vec<Vec<u64>> = intervals
            .iter()
            .map(|&(before, after)| xor_signatures(&trace[before * wires], &trace[after * wires]))
            .collect();
        for subset in 1usize..16 {
            let mut sum = vec![0u64; operand.len()];
            for (row, delta) in deltas.iter().enumerate() {
                if subset & (1 << row) != 0 {
                    sum = xor_signatures(&sum, delta);
                }
            }
            assert_eq!(
                sum == operand,
                subset == 15,
                "formal share subset {subset:#06b} changed rank"
            );
        }
        assert!(
            same_wire_space_time_witness(&trace, wires, &operand).is_none(),
            "a single accumulator interval gathered the complete operand"
        );
        for input in 0u64..(1 << wires) {
            assert_eq!(
                eval_u64(&gates, input),
                input,
                "share gathers did not strip back to arbitrary incoming junk"
            );
        }
    }

    #[test]
    fn prod_micro_gray_generic_fold_is_exhaustive_and_two_control() {
        // Four atoms per operand, including a quartic mask, force the r=4
        // schedule and its two-helper path. Exhaust the complete 12-wire state
        // space so target semantics and restoration of every dirty borrow are
        // checked without a clean-wire assumption.
        let n = 7;
        let carrier_total = n;
        let pairs: Vec<(usize, usize)> = (0..n).map(|wire| (wire, wire)).collect();
        let band = 5;
        let live = carrier_total + band;
        let cfg = ProdConfig {
            k: 2,
            deg: 2,
            k_hi: 1,
            deg_hi: 4,
            band,
            rsrc: 0,
            single: 1,
            g57_narrow: 1,
            rung_menu: 1,
            gray_fold: 2,
            ..ProdConfig::off()
        };
        let gate = XGate::from_g57([0, 1, 2]);
        for seed in 0..3u64 {
            let state = GadgetState {
                n,
                pairs: pairs.clone(),
            };
            let mut rng = StdRng::seed_from_u64(0x4d47_2000 + seed);
            let mut ledger = ProdLedger::new(n, &cfg, carrier_total, None);
            let mut ramp = Vec::new();
            ledger.inject_all(&state, &mut rng, &mut ramp);
            let slots_before = ledger.slots.clone();
            let consts_before = ledger.consts.clone();
            let mut fold = Vec::new();
            ledger.fold_cg(&gate, &state, &mut rng, &mut fold);
            assert_eq!(ledger.cg_gray, 1, "seed={seed}: micro path declined");
            assert!(
                fold.iter().all(|emitted| emitted.width() <= 2),
                "seed={seed}: micro product emitted above the width cap"
            );
            let touched = 1u64 << gate.target;
            for input in 0u64..(1u64 << live) {
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
                assert_eq!(
                    (output ^ input) & !touched,
                    0,
                    "seed={seed} input={input:#x}: dirty borrow was not restored"
                );
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
                let fires = gate
                    .ctrls
                    .iter()
                    .all(|&(wire, polarity)| (before[wire as usize] != 0) == polarity)
                    ^ gate.comp;
                for value in 0..n {
                    assert_eq!(
                        after[value],
                        before[value] ^ ((value == gate.target as usize && fires) as u64),
                        "seed={seed} input={input:#x} value={value}"
                    );
                }
            }
        }
    }

    #[test]
    fn prod_micro_gray_five_six_seven_width_and_restoration_census() {
        let gate = XGate::from_g57([0, 1, 2]);
        let check = |label: &str,
                     fold: &[XGate],
                     update_len: usize,
                     total: usize,
                     target_wires: &[usize],
                     before_atoms: &[Vec<Vec<(u16, bool)>>],
                     after_atoms: &[Vec<Vec<(u16, bool)>>]| {
            assert!(
                fold[update_len..]
                    .iter()
                    .all(|emitted| emitted.width() <= 2),
                "{label}: micro product/gather suffix exceeded two controls"
            );
            let live_mask = (1u64 << total) - 1;
            let target_mask = target_wires
                .iter()
                .fold(0u64, |mask, &wire| mask | (1u64 << wire));
            let mut samples = vec![0u64, live_mask];
            samples.extend((0..1024u64).map(|index| {
                index
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    .rotate_left((index as u32 * 11) & 63)
                    & live_mask
            }));
            for input in samples {
                let before: Vec<u64> = before_atoms
                    .iter()
                    .map(|atoms| eval_anf_atoms(input, atoms))
                    .collect();
                let output = eval_u64(fold, input);
                assert_eq!(
                    (output ^ input) & (live_mask ^ target_mask),
                    0,
                    "{label}: accumulator/helper was not restored at {input:#x}"
                );
                let after: Vec<u64> = after_atoms
                    .iter()
                    .map(|atoms| eval_anf_atoms(output, atoms))
                    .collect();
                let fires = gate
                    .ctrls
                    .iter()
                    .all(|&(wire, polarity)| (before[wire as usize] != 0) == polarity)
                    ^ gate.comp;
                for value in 0..before.len() {
                    assert_eq!(
                        after[value],
                        before[value] ^ ((value == gate.target as usize && fires) as u64),
                        "{label}: logical mismatch at input={input:#x} value={value}"
                    );
                }
            }
        };

        let n = 7;
        let mut cfg = ProdConfig::off();
        cfg.gray_fold = 2;
        cfg.rung_menu = 1;

        let five = FiveCarrierState::home(n);
        let mut five_ledger = ProdLedger::new(n, &cfg, 5 * n, None);
        let five_before: Vec<_> = (0..n)
            .map(|value| five_ledger.five_decode_atoms(value, true, &five))
            .collect();
        let mut rng = StdRng::seed_from_u64(0x4d47_5005);
        let mut five_fold = Vec::new();
        five_ledger.fold_five(&gate, &five, &mut rng, &mut five_fold);
        assert_eq!(five_ledger.cg_gray, 1, "five-carrier micro path declined");
        assert_eq!((five_ledger.cg_fragments, five_ledger.cg_narrow), (64, 64));
        let five_after: Vec<_> = (0..n)
            .map(|value| five_ledger.five_decode_atoms(value, true, &five))
            .collect();
        check(
            "five",
            &five_fold,
            FIVE_CARRIER_U0_GATES.len(),
            5 * n,
            &five.carriers[0],
            &five_before,
            &five_after,
        );

        let six = SixCarrierState::home(n);
        let mut six_ledger = ProdLedger::new(n, &cfg, 6 * n, None);
        let six_before: Vec<_> = (0..n)
            .map(|value| six_ledger.six_decode_atoms(value, true, &six))
            .collect();
        let mut rng = StdRng::seed_from_u64(0x4d47_6006);
        let mut six_fold = Vec::new();
        six_ledger.fold_six(&gate, &six, &mut rng, &mut six_fold);
        assert_eq!(six_ledger.cg_gray, 1, "six-carrier micro path declined");
        assert_eq!((six_ledger.cg_fragments, six_ledger.cg_narrow), (64, 64));
        let six_after: Vec<_> = (0..n)
            .map(|value| six_ledger.six_decode_atoms(value, true, &six))
            .collect();
        check(
            "six",
            &six_fold,
            SIX_CARRIER_U0_GATES.len(),
            6 * n,
            &six.carriers[0],
            &six_before,
            &six_after,
        );

        let seven = SevenCarrierState::home(n);
        let mut seven_ledger = ProdLedger::new(n, &cfg, 7 * n, None);
        let seven_before: Vec<_> = (0..n)
            .map(|value| seven_ledger.seven_decode_atoms(value, true, &seven))
            .collect();
        let mut rng = StdRng::seed_from_u64(0x4d47_7007);
        let mut seven_fold = Vec::new();
        seven_ledger.fold_seven(&gate, &seven, &mut rng, &mut seven_fold);
        assert_eq!(seven_ledger.cg_gray, 1, "seven-carrier micro path declined");
        assert_eq!(
            (seven_ledger.cg_fragments, seven_ledger.cg_narrow),
            (64, 64)
        );
        let seven_after: Vec<_> = (0..n)
            .map(|value| seven_ledger.seven_decode_atoms(value, true, &seven))
            .collect();
        check(
            "seven",
            &seven_fold,
            SEVEN_CARRIER_U0_GATES.len(),
            7 * n,
            &seven.carriers[0],
            &seven_before,
            &seven_after,
        );
    }

    #[test]
    fn prod_sentinel_cross_ladders_are_exact_and_force_a_cross_rung() {
        // target=0, dirty helpers=1,2, blind=3, H literals=4..6. Every
        // polarity pattern and incoming state is covered. The gates writing
        // helper0 must read the blind plus exactly one H literal; choosing two
        // H literals here would expose the complete cubic H one rung later.
        for width in 3usize..=4 {
            for polarity_mask in 0usize..(1usize << width) {
                let lits: Vec<(u16, bool)> = (0..width)
                    .map(|index| ((index + 3) as u16, polarity_mask & (1 << index) != 0))
                    .collect();
                let mut gates = Vec::new();
                emit_exact_dirty_cap2(0, &lits, 1, 2, &mut gates);
                assert_eq!(gates.len(), if width == 3 { 4 } else { 8 });
                assert!(gates.iter().all(|gate| gate.width() <= 2));
                for gate in gates.iter().filter(|gate| gate.target == 1) {
                    let controls: Vec<u16> = gate.ctrls.iter().map(|&(wire, _)| wire).collect();
                    assert!(controls.contains(&3), "rung zero lost the blind factor");
                    assert!(controls.contains(&4), "rung zero lost its first H factor");
                    assert_eq!(controls.len(), 2);
                }
                let wires = width + 3;
                for input in 0u64..(1u64 << wires) {
                    let fire = lits
                        .iter()
                        .all(|&(wire, polarity)| ((input >> wire) & 1 != 0) == polarity);
                    assert_eq!(
                        eval_u64(&gates, input),
                        input ^ fire as u64,
                        "width={width} polarity={polarity_mask:#x} input={input:#x}"
                    );
                }
            }
        }
    }

    #[test]
    fn prod_sentinel_schedule_is_exhaustive_restores_junk_and_has_no_full_gather() {
        // Physical roles: target 0; live linear operands 1,2; unrelated dirty
        // u,z,h0,h1 = 3..6; shared mask variables 7..12. The Q/H factors
        // overlap deliberately, so this is not a disjoint-variable toy.
        let lists = [
            vec![
                vec![(1u16, false)],
                vec![(7, true), (8, false)],
                vec![(7, true), (8, false), (9, true)],
            ],
            vec![
                vec![(2u16, true)],
                vec![(8, true), (9, true)],
                vec![(10, false), (11, true), (12, false)],
            ],
        ];
        let parts = [
            partition_max_degree_sentinel(&lists[0]).unwrap(),
            partition_max_degree_sentinel(&lists[1]).unwrap(),
        ];
        let cfg = ProdConfig::off();
        let mut ledger = ProdLedger::new(7, &cfg, 7, None);
        let mut gates = Vec::new();
        ledger.emit_sentinel_schedule(0, 0, &parts, [3, 4, 5, 6], &mut gates);
        let census = |width| gates.iter().filter(|gate| gate.width() == width).count();
        assert_eq!(gates.len(), 62, "sentinel primitive cost drifted");
        assert_eq!(census(2), 61, "cap-two population drifted");
        assert_eq!(census(6), 1, "H*H sentinel fossil drifted");
        assert_eq!(ledger.cg_laddered, 6, "expected six cross-tail ladders");
        assert_eq!(ledger.cg_fossils, 1, "expected exactly one H*H fossil");

        let wires = 13;
        for input in 0u64..(1u64 << wires) {
            let left = eval_anf_atoms(input, &lists[0]);
            let right = eval_anf_atoms(input, &lists[1]);
            let output = eval_u64(&gates, input);
            assert_eq!(
                output,
                input ^ (left & right),
                "sentinel identity/restoration failed at {input:#x}"
            );
        }

        // Q is the intentional canary: an accumulator interval must reveal it.
        // Neither H nor a complete operand may appear on u/z/helper deltas.
        let trace = prefix_wire_signatures(&gates, wires);
        let signature = |atoms: &[Vec<(u16, bool)>]| -> Vec<u64> {
            let mut value = vec![0u64; (1usize << wires) / 64];
            for input in 0usize..(1usize << wires) {
                if eval_anf_atoms(input as u64, atoms) != 0 {
                    value[input / 64] |= 1u64 << (input % 64);
                }
            }
            value
        };
        let q = [signature(&parts[0].gathered), signature(&parts[1].gathered)];
        let forbidden = [
            signature(&parts[0].high),
            signature(&parts[1].high),
            signature(&lists[0]),
            signature(&lists[1]),
        ];
        let prefixes = gates.len() + 1;
        let mut saw_q = [false; 2];
        for wire in [3usize, 4, 5, 6] {
            for before in 0..prefixes {
                for after in before + 1..prefixes {
                    let delta =
                        xor_signatures(&trace[before * wires + wire], &trace[after * wires + wire]);
                    for side in 0..2 {
                        saw_q[side] |= delta == q[side];
                    }
                    assert!(
                        forbidden.iter().all(|secret| delta != *secret),
                        "borrowed wire {wire} gathered H or a complete operand"
                    );
                }
            }
        }
        assert_eq!(saw_q, [true, true], "Q canary was not trace-recoverable");
    }

    #[test]
    fn prod_sentinel_production_single_is_exhaustive_with_signed_constants() {
        // Deterministic [2,2,2,3] production mask plan on two disjoint halves
        // of a six-wire band. This pins the nominal sentinel census exactly:
        // 69 cap-two primitives and one cubic*cubic width-six fossil.
        let n = 7;
        let carrier_total = n;
        let pairs: Vec<(usize, usize)> = (0..n).map(|wire| (wire, wire)).collect();
        let state = GadgetState {
            n,
            pairs: pairs.clone(),
        };
        let cfg = ProdConfig {
            k: 3,
            deg: 2,
            k_hi: 1,
            deg_hi: 3,
            band: 6,
            rsrc: 0,
            single: 1,
            gray_fold: 3,
            ..ProdConfig::off()
        };
        let source = XGate::from_g57([0, 1, 2]);
        let plans = [
            vec![
                ProdSlot {
                    factors: vec![(0, false), (1, false)],
                },
                ProdSlot {
                    factors: vec![(0, false), (2, false)],
                },
                ProdSlot {
                    factors: vec![(1, false), (2, false)],
                },
                ProdSlot {
                    factors: vec![(0, false), (1, false), (2, false)],
                },
            ],
            vec![
                ProdSlot {
                    factors: vec![(3, false), (4, false)],
                },
                ProdSlot {
                    factors: vec![(3, false), (5, false)],
                },
                ProdSlot {
                    factors: vec![(4, false), (5, false)],
                },
                ProdSlot {
                    factors: vec![(3, false), (4, false), (5, false)],
                },
            ],
        ];
        let live = carrier_total + 6;
        for constant_mask in 0usize..4 {
            let mut ledger = ProdLedger::new(n, &cfg, carrier_total, None);
            ledger.slots[1] = plans[0].clone();
            ledger.slots[2] = plans[1].clone();
            ledger.consts[1] = constant_mask & 1 != 0;
            ledger.consts[2] = constant_mask & 2 != 0;
            let slots_before = ledger.slots.clone();
            let consts_before = ledger.consts.clone();
            let mut rng = StdRng::seed_from_u64(0x5e17_0000 + constant_mask as u64);
            let mut fold = Vec::new();
            ledger.fold_cg(&source, &state, &mut rng, &mut fold);
            assert_eq!(ledger.cg_sentinel, 1, "sentinel path declined");
            assert_eq!(fold.len(), 70, "production-single cost drifted");
            assert_eq!(fold.iter().filter(|gate| gate.width() == 2).count(), 69);
            assert_eq!(fold.iter().filter(|gate| gate.width() == 6).count(), 1);
            let touched = 1u64 << source.target;
            for input in 0u64..(1u64 << live) {
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
                assert_eq!(
                    (output ^ input) & !touched,
                    0,
                    "constant_mask={constant_mask} input={input:#x}: borrow leaked"
                );
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
                let fires = source
                    .ctrls
                    .iter()
                    .all(|&(wire, polarity)| (before[wire as usize] != 0) == polarity)
                    ^ source.comp;
                for value in 0..n {
                    assert_eq!(
                        after[value],
                        before[value] ^ ((value == 0 && fires) as u64),
                        "constant_mask={constant_mask} input={input:#x} value={value}"
                    );
                }
            }
        }
    }

    #[test]
    fn prod_sentinel_five_six_seven_correctness_and_width_census() {
        let n = 7usize;
        let band = 6usize;
        let source = XGate::from_g57([0, 1, 2]);
        let cfg = ProdConfig {
            k: 3,
            deg: 2,
            k_hi: 1,
            deg_hi: 3,
            band,
            rsrc: 0,
            single: 1,
            gray_fold: 3,
            ..ProdConfig::off()
        };
        let plans = [
            vec![
                ProdSlot {
                    factors: vec![(0, false), (1, false)],
                },
                ProdSlot {
                    factors: vec![(0, false), (2, false)],
                },
                ProdSlot {
                    factors: vec![(1, false), (2, false)],
                },
                ProdSlot {
                    factors: vec![(0, false), (1, false), (2, false)],
                },
            ],
            vec![
                ProdSlot {
                    factors: vec![(3, false), (4, false)],
                },
                ProdSlot {
                    factors: vec![(3, false), (5, false)],
                },
                ProdSlot {
                    factors: vec![(4, false), (5, false)],
                },
                ProdSlot {
                    factors: vec![(3, false), (4, false), (5, false)],
                },
            ],
        ];
        let census = |gates: &[XGate]| -> Vec<(usize, usize)> {
            let mut counts = std::collections::BTreeMap::new();
            for gate in gates {
                *counts.entry(gate.width()).or_insert(0usize) += 1;
            }
            counts.into_iter().collect()
        };
        let samples = |total: usize| -> Vec<u64> {
            let mask = (1u64 << total) - 1;
            std::iter::once(0)
                .chain(std::iter::once(mask))
                .chain((0..2048u64).map(|index| {
                    index
                        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                        .rotate_left((index as u32 * 7) & 63)
                        & mask
                }))
                .collect()
        };

        let five = FiveCarrierState::home(n);
        let mut five_ledger = ProdLedger::new(n, &cfg, 5 * n, None);
        five_ledger.slots[1] = plans[0].clone();
        five_ledger.slots[2] = plans[1].clone();
        let five_before: Vec<_> = (0..n)
            .map(|value| five_ledger.five_decode_atoms(value, true, &five))
            .collect();
        let mut five_fold = Vec::new();
        let mut rng = StdRng::seed_from_u64(0x5e17_5005);
        five_ledger.fold_five(&source, &five, &mut rng, &mut five_fold);
        let five_after: Vec<_> = (0..n)
            .map(|value| five_ledger.five_decode_atoms(value, true, &five))
            .collect();
        let five_suffix = &five_fold[FIVE_CARRIER_U0_GATES.len()..];
        assert_eq!(five_ledger.cg_sentinel, 1, "five sentinel path declined");
        assert_eq!(census(five_suffix), vec![(2, 85), (6, 1)]);

        let six = SixCarrierState::home(n);
        let mut six_ledger = ProdLedger::new(n, &cfg, 6 * n, None);
        six_ledger.slots[1] = plans[0].clone();
        six_ledger.slots[2] = plans[1].clone();
        let six_before: Vec<_> = (0..n)
            .map(|value| six_ledger.six_decode_atoms(value, true, &six))
            .collect();
        let mut six_fold = Vec::new();
        let mut rng = StdRng::seed_from_u64(0x5e17_6006);
        six_ledger.fold_six(&source, &six, &mut rng, &mut six_fold);
        let six_after: Vec<_> = (0..n)
            .map(|value| six_ledger.six_decode_atoms(value, true, &six))
            .collect();
        let six_suffix = &six_fold[SIX_CARRIER_U0_GATES.len()..];
        assert_eq!(six_ledger.cg_sentinel, 1, "six sentinel path declined");
        assert_eq!(census(six_suffix), vec![(2, 348), (6, 9)]);

        let seven = SevenCarrierState::home(n);
        let mut seven_ledger = ProdLedger::new(n, &cfg, 7 * n, None);
        seven_ledger.slots[1] = plans[0].clone();
        seven_ledger.slots[2] = plans[1].clone();
        let seven_before: Vec<_> = (0..n)
            .map(|value| seven_ledger.seven_decode_atoms(value, true, &seven))
            .collect();
        let mut seven_fold = Vec::new();
        let mut rng = StdRng::seed_from_u64(0x5e17_7007);
        seven_ledger.fold_seven(&source, &seven, &mut rng, &mut seven_fold);
        let seven_after: Vec<_> = (0..n)
            .map(|value| seven_ledger.seven_decode_atoms(value, true, &seven))
            .collect();
        let seven_suffix = &seven_fold[SEVEN_CARRIER_U0_GATES.len()..];
        assert_eq!(seven_ledger.cg_sentinel, 1, "seven sentinel path declined");
        // Quartic H makes H*accumulator width five; mode 3 intentionally
        // ladders only through width four, so those ten brackets remain as
        // fossils. The production cubic mask is below the representation's
        // max degree and is therefore gathered as Q, not treated as H.
        assert_eq!(census(seven_suffix), vec![(2, 61), (5, 10), (8, 1)]);

        let fixtures = [
            (
                "five",
                5 * n + band,
                &five_fold[..],
                &five.carriers[0][..],
                &five_before[..],
                &five_after[..],
            ),
            (
                "six",
                6 * n + band,
                &six_fold[..],
                &six.carriers[0][..],
                &six_before[..],
                &six_after[..],
            ),
            (
                "seven",
                7 * n + band,
                &seven_fold[..],
                &seven.carriers[0][..],
                &seven_before[..],
                &seven_after[..],
            ),
        ];
        for (label, total, fold, target_group, before_atoms, after_atoms) in fixtures {
            let target_mask = target_group
                .iter()
                .fold(0u64, |mask, &wire| mask | (1u64 << wire));
            let total_mask = (1u64 << total) - 1;
            for input in samples(total) {
                let before: Vec<u64> = before_atoms
                    .iter()
                    .map(|atoms| eval_anf_atoms(input, atoms))
                    .collect();
                let output = eval_u64(fold, input);
                assert_eq!(
                    (output ^ input) & (total_mask ^ target_mask),
                    0,
                    "{label}: unrelated dirty wire changed at {input:#x}"
                );
                let after: Vec<u64> = after_atoms
                    .iter()
                    .map(|atoms| eval_anf_atoms(output, atoms))
                    .collect();
                let fires = source
                    .ctrls
                    .iter()
                    .all(|&(wire, polarity)| (before[wire as usize] != 0) == polarity)
                    ^ source.comp;
                for value in 0..n {
                    assert_eq!(
                        after[value],
                        before[value] ^ ((value == 0 && fires) as u64),
                        "{label}: logical mismatch at {input:#x}, value={value}"
                    );
                }
            }
        }
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
        // is why the source list below includes a mixed-polarity CCNOT.
        // n must leave carriers over to borrow: with n=3 every carrier is the
        // target or an operand, the fold declines, and the test would only be
        // re-checking the odometer (the `gray_blocks` assertion at the end).
        let n = 6;
        let carrier_total = n; // single-carrier: value v lives on wire v
        let pairs: Vec<(usize, usize)> = (0..n).map(|v| (v, v)).collect();
        let band = 5usize; // wires 6..11
        let live = carrier_total + band;
        let cfg = ProdConfig {
            k: 1,
            deg: 2,
            k_hi: 1,
            deg_hi: 3,
            band,
            rsrc: 0,
            single: 1,
            g57_narrow: 1,
            gray_fold: 1,
            ..ProdConfig::off()
        };
        let sources: Vec<XGate> = vec![
            XGate::from_g57([0, 1, 2]),
            XGate::conj(2, [(0u16, false), (1u16, true)]).unwrap(),
            XGate::conj(0, [(1u16, true), (2u16, true)]).unwrap(),
            XGate::conj(1, [(0u16, false), (2u16, false)]).unwrap(),
            XGate::cnot(1, 2),
            XGate::x_gate(0),
        ];
        let mut gray_blocks = 0u64;
        for seed in 0..24u64 {
            let mut rng = StdRng::seed_from_u64(0x67a4_0000 + seed);
            let state = GadgetState {
                n,
                pairs: pairs.clone(),
            };
            let mut ledger = ProdLedger::new(n, &cfg, carrier_total, None);
            let mut ramp = Vec::new();
            ledger.inject_all(&state, &mut rng, &mut ramp);
            for gate in &sources {
                let slots_before = ledger.slots.clone();
                let consts_before = ledger.consts.clone();
                let mut fold = Vec::new();
                ledger.fold_cg(gate, &state, &mut rng, &mut fold);
                let t = gate.target as usize;
                for g in &fold {
                    assert!(
                        g.width() <= 2,
                        "seed={seed} gate={gate:?}: gray fold emitted a {}-control gate",
                        g.width()
                    );
                }
                let touched = 1u64 << pairs[t].0;
                for input in 0..(1u64 << live) {
                    let before: Vec<u64> = (0..n)
                        .map(|v| {
                            prod_decode(
                                input,
                                v,
                                &pairs,
                                &slots_before,
                                &consts_before,
                                &ledger.loc,
                            )
                        })
                        .collect();
                    let out_state = eval_u64(&fold, input);
                    // Every accumulator and every sandwich helper is restored:
                    // the net effect lives entirely on the target's carrier.
                    assert_eq!(
                        (out_state ^ input) & !touched,
                        0,
                        "seed={seed} gate={gate:?}: a borrowed wire was not restored"
                    );
                    let after: Vec<u64> = (0..n)
                        .map(|v| {
                            prod_decode(
                                out_state,
                                v,
                                &pairs,
                                &ledger.slots,
                                &ledger.consts,
                                &ledger.loc,
                            )
                        })
                        .collect();
                    let fires = gate
                        .ctrls
                        .iter()
                        .all(|&(w, pol)| (before[w as usize] != 0) == pol)
                        ^ gate.comp;
                    for v in 0..n {
                        let expected = before[v] ^ ((v == t && fires) as u64);
                        assert_eq!(
                            after[v], expected,
                            "seed={seed} gate={gate:?} value={v} input={input:#x}"
                        );
                    }
                }
                assert_eq!(slots_before, ledger.slots, "the fold disturbed a slot");
            }
            gray_blocks += ledger.cg_gray;
        }
        // The arity-2 sources must actually take the Gray path, or the test
        // above is only re-checking the odometer.
        assert!(
            gray_blocks >= 24 * 4,
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
        // target's carrier and from every literal the block reads, and each is
        // written by an even number of gates so its incoming junk survives to
        // cancel. A clean accumulator would show up as a wire whose first
        // touch is a write with no prior read -- covered by the restoration
        // assertion in the test above, which a clean-ancilla variant fails.
        let n = 6;
        let carrier_total = n;
        let pairs: Vec<(usize, usize)> = (0..n).map(|v| (v, v)).collect();
        let band = 8usize;
        let cfg = ProdConfig {
            k: 1,
            deg: 2,
            k_hi: 2,
            deg_hi: 3,
            band,
            rsrc: 0,
            single: 1,
            g57_narrow: 1,
            gray_fold: 1,
            ..ProdConfig::off()
        };
        for seed in 0..32u64 {
            let mut rng = StdRng::seed_from_u64(0x67a5_0000 + seed);
            let state = GadgetState {
                n,
                pairs: pairs.clone(),
            };
            let mut ledger = ProdLedger::new(n, &cfg, carrier_total, None);
            let mut ramp = Vec::new();
            ledger.inject_all(&state, &mut rng, &mut ramp);
            let gate = XGate::from_g57([0, 1, 2]);
            let mut fold = Vec::new();
            ledger.fold_cg(&gate, &state, &mut rng, &mut fold);
            assert_eq!(ledger.cg_gray, 1, "seed={seed}: not the gray path");
            // Wires the block reads as mask/carrier literals of its operands.
            let mut operand_wires: Vec<u16> = vec![0, 1, 2];
            for w in [1usize, 2] {
                for slot in &ledger.slots[w] {
                    operand_wires.extend(slot.lits(&ledger.loc).iter().map(|&(x, _)| x));
                }
            }
            let mut writes: std::collections::HashMap<u16, usize> =
                std::collections::HashMap::new();
            for g in &fold {
                *writes.entry(g.target).or_default() += 1;
            }
            for (&w, &count) in &writes {
                if w == 0 {
                    continue; // the target's carrier: written an odd number of times
                }
                assert_eq!(
                    count % 2,
                    0,
                    "seed={seed}: borrowed wire {w} is written {count} times, so its \
                     incoming value does not cancel"
                );
                assert!(
                    !operand_wires.contains(&w),
                    "seed={seed}: wire {w} is both borrowed and read as an operand literal"
                );
            }
        }
    }

    #[test]
    fn prod_gray_fold_has_an_exact_space_time_operand_recovery() {
        // Red-team the assumption made by the ordinary Gray audit: it checks
        // every prefix separately, while a trace adversary can combine two
        // prefixes.  Compare identical masks and RNG choices with and without
        // the aggregate Gray gather.
        let n = 6;
        let band = 5usize;
        let wires = n + band;
        let pairs: Vec<(usize, usize)> = (0..n).map(|value| (value, value)).collect();
        let state = GadgetState {
            n,
            pairs: pairs.clone(),
        };
        let source = XGate::from_g57([0, 1, 2]);

        let build = |gray: bool| {
            // Build the SAME masks and consume the SAME injection RNG in both
            // arms, then switch only the fold strategy.  Configuring max_width
            // differently before injection shifts the random mask stream and
            // invalidates the A/B comparison this test is meant to make.
            let cfg = ProdConfig {
                k: 3,
                deg: 2,
                k_hi: 1,
                deg_hi: 3,
                band,
                rsrc: 0,
                single: 1,
                max_width: 2,
                g57_narrow: 1,
                gray_fold: 0,
                ..ProdConfig::off()
            };
            let mut rng = StdRng::seed_from_u64(0x67a6_5ace);
            let mut ledger = ProdLedger::new(n, &cfg, n, None);
            let mut ramp = Vec::new();
            ledger.inject_all(&state, &mut rng, &mut ramp);
            let slots = ledger.slots.clone();
            let consts = ledger.consts.clone();
            let loc = ledger.loc.clone();
            ledger.gray_fold = gray;
            let mut fold = Vec::new();
            ledger.fold_cg(&source, &state, &mut rng, &mut fold);
            (fold, slots, consts, loc, ledger.cg_gray)
        };

        let (gray_fold, slots, consts, loc, gray_blocks) = build(true);
        let (expanded_fold, expanded_slots, expanded_consts, expanded_loc, expanded_gray) =
            build(false);
        assert_eq!(slots, expanded_slots, "A/B masks differ");
        assert_eq!(consts, expanded_consts, "A/B mask constants differ");
        assert_eq!(loc, expanded_loc, "A/B band locations differ");
        assert_eq!(gray_blocks, 1, "fixture did not exercise the Gray path");
        assert_eq!(expanded_gray, 0, "control unexpectedly used the Gray path");

        let gray_trace = prefix_wire_signatures(&gray_fold, wires);
        let expanded_trace = prefix_wire_signatures(&expanded_fold, wires);
        for operand in [1usize, 2] {
            let logical = prod_decode_signature(wires, operand, &pairs, &slots, &consts, &loc);
            let carrier_at_entry = &gray_trace[operand];
            let needed_delta = xor_signatures(&logical, carrier_at_entry);

            // This is the guarantee the current per-prefix audit actually
            // establishes: no simultaneous affine view recovers the operand.
            for (prefix, columns) in gray_trace.chunks_exact(wires).enumerate() {
                assert!(
                    !affine_span_contains(columns, &logical),
                    "operand {operand} is already affine at Gray prefix {prefix}"
                );
            }

            // Combining two times on one (unknown in advance) physical wire
            // changes the answer from impossible to exact over all 2^11 input
            // states.
            let witness = same_wire_space_time_witness(&gray_trace, wires, &needed_delta)
                .unwrap_or_else(|| panic!("no Gray space-time witness for operand {operand}"));
            eprintln!(
                "operand {operand} = entry carrier {operand} XOR wire {}@prefix {} XOR \
                 wire {}@prefix {} XOR {} (all {} states)",
                witness.0,
                witness.1,
                witness.0,
                witness.2,
                witness.3 as u8,
                1usize << wires
            );

            // Proper per-monomial expansion never gathers the complete mask on
            // one borrowed wire, so the same short aggregate-mask identity must
            // be absent in the otherwise identical control.
            assert!(
                same_wire_space_time_witness(&expanded_trace, wires, &needed_delta).is_none(),
                "expanded control unexpectedly has the Gray aggregate-mask witness"
            );
        }
    }

    #[test]
    fn prod_fold_cg_emits_its_fragments_out_of_odometer_order() {
        // The fold's fragments all XOR into the target value's two carriers
        // and read nothing else it writes, so their order is free — and the
        // deterministic odometer order is a static per-gate progress clock
        // (consecutive fragments share atom prefixes) readable with no
        // execution at all. Both fold paths must shuffle it away.
        let n = 3;
        let carrier_total = 2 * n;
        let pairs = vec![(0usize, 1usize), (2, 3), (4, 5)];
        let state = GadgetState { n, pairs };
        // 2 controls x (2 carriers + 2 mask atoms) = 16 fragments per fold.
        let gate = XGate::from_g57([0, 1, 2]);
        for cap in [0usize, 2] {
            let cfg = ProdConfig {
                k: 2,
                deg: 2,
                band: 6,
                rsrc: 0,
                max_width: cap,
                ..ProdConfig::off()
            };
            let mut shuffled = 0usize;
            for seed in 0..8u64 {
                let mut rng = StdRng::seed_from_u64(0x5017_0000 + seed);
                let mut ledger = ProdLedger::new(n, &cfg, carrier_total, None);
                let mut ramp = Vec::new();
                ledger.inject_all(&state, &mut rng, &mut ramp);
                let mut fold = Vec::new();
                ledger.fold_cg(&gate, &state, &mut rng, &mut fold);
                // 4 atoms per control gives 16 combinations, but a fold
                // fragment is DROPPED when two atoms meet on one wire with
                // opposite polarity (the conjunction is identically zero), and
                // at band 6 with two degree-2 masks per value such a collision
                // is ordinary. Assert only that the fold is wide enough for the
                // run-length test below to mean something -- pinning the exact
                // count makes this test a hostage to the RNG stream, which any
                // change in how earlier gates are spelled will shift.
                assert!(fold.len() >= 12, "expected a wide fold, got {}", fold.len());
                // Odometer order walks the first control's atoms fastest, so
                // it emits long runs that read the same second-control atom.
                // A shuffled order breaks those runs.
                let second: Vec<Vec<u16>> = fold
                    .iter()
                    .map(|g| {
                        let mut ws: Vec<u16> = g
                            .ctrls
                            .iter()
                            .map(|&(w, _)| w)
                            .filter(|&w| w >= 4)
                            .collect();
                        ws.sort_unstable();
                        ws
                    })
                    .collect();
                let runs = 1 + second.windows(2).filter(|w| w[0] != w[1]).count();
                if runs > second.len() / 2 {
                    shuffled += 1;
                }
            }
            assert!(
                shuffled >= 7,
                "cap={cap}: fold fragments still come out in odometer order \
                 ({shuffled}/8 seeds shuffled)"
            );
        }
    }

    #[test]
    fn prod_band_roll_relocates_the_band_and_preserves_every_value() {
        // The roll is RG2's move applied to a band variable: the emitted
        // 3-CNOT swap must leave every logical value's decode unchanged under
        // the updated bookkeeping, for every input, and the band must
        // actually end up somewhere else — including inside the carrier
        // space, with the vacated wire becoming a carrier.
        let n = 3;
        let carrier_total = 2 * n;
        let band = 4usize;
        let total = carrier_total + band;
        let cfg = ProdConfig {
            k: 1,
            deg: 2,
            band,
            rsrc: 0,
            roll: 1,
            ..ProdConfig::off()
        };
        let mut left_home = 0usize;
        for seed in 0..24u64 {
            let mut rng = StdRng::seed_from_u64(0x0011_0000 + seed);
            let mut state = GadgetState {
                n,
                pairs: vec![(0usize, 1usize), (2, 3), (4, 5)],
            };
            let mut ledger = ProdLedger::new(n, &cfg, carrier_total, None);
            let mut ramp = Vec::new();
            ledger.inject_all(&state, &mut rng, &mut ramp);
            for _ in 0..6 {
                let pairs_before = state.pairs.clone();
                let slots_before = ledger.slots.clone();
                let consts_before = ledger.consts.clone();
                let loc_before = ledger.loc.clone();
                let mut moved = Vec::new();
                ledger.roll(&mut state, &mut rng, &mut moved);
                // RG2's three transvections, each either a plain CNOT or the
                // two-term form, so no wire is written only by width-1 gates.
                // The two-term form is spelled in the g57 vocabulary: the
                // same-polarity half costs g57+CNOT and the mixed-polarity
                // half one g57, so that branch is 3 gates and a roll spans
                // 3 (all CNOT) to 9 (all two-term) gates.
                assert!(
                    (3..=9).contains(&moved.len()),
                    "a roll is three transvections, got {}",
                    moved.len()
                );
                // Width 1..=2 still holds, but comp=1 is now EXPECTED: the
                // two-term transvection is spelled in the g57 vocabulary, and
                // a g57 carries comp. What must hold is that the pair leaves no
                // NET constant -- a roll has no ledger to defer one to -- and
                // that is what the decode check below actually verifies.
                assert!(moved.iter().all(|g| (1..=2).contains(&g.width())));
                for input in 0..(1u64 << total) {
                    let out_state = eval_u64(&moved, input);
                    for v in 0..n {
                        let before = prod_decode(
                            input,
                            v,
                            &pairs_before,
                            &slots_before,
                            &consts_before,
                            &loc_before,
                        );
                        let after = prod_decode(
                            out_state,
                            v,
                            &state.pairs,
                            &ledger.slots,
                            &ledger.consts,
                            &ledger.loc,
                        );
                        assert_eq!(before, after, "seed={seed} value={v}: roll changed a value");
                    }
                }
                // Carriers and band wires stay a partition of the wire space.
                let mut occupied: Vec<u16> = ledger.loc.clone();
                for &(s, p) in &state.pairs {
                    occupied.push(s as u16);
                    occupied.push(p as u16);
                }
                occupied.sort_unstable();
                occupied.dedup();
                assert_eq!(
                    occupied.len(),
                    total,
                    "carriers and band overlap after a roll"
                );
            }
            if ledger.loc.iter().any(|&w| (w as usize) < carrier_total) {
                left_home += 1;
            }
        }
        assert!(
            left_home >= 20,
            "the band almost never leaves its home range ({left_home}/24)"
        );
    }

    #[test]
    fn prod_rolling_band_gadget_is_correct_and_writes_the_band_in_the_body() {
        // End to end with --prod-roll: the endpoint contract must survive for
        // ARBITRARY junk on every non-data wire, and the roll must actually
        // change the emitted circuit's write profile on the band. ("Every wire
        // is written somewhere" would be vacuous — the two band fills already
        // write every band wire even at roll 0; the comparison below is
        // against the same seed with rolls off.)
        let n = MASKED_TEST_N;
        let main = masked_test_main();
        let mask = (1u64 << n) - 1;
        let band = 6usize;
        for (max_width, fill_nl) in [(0usize, 2usize), (2, 2)] {
            let cfg = |roll| ProdConfig {
                k: 1,
                deg: 2,
                k_hi: 1,
                deg_hi: 3,
                band,
                rsrc: 1,
                max_width,
                fill_nl,
                roll,
                src_dist: 0,
                src_horizon: 0,
                src_lo: 0,
                src_hi: 0,
                fill_pivots: 0,
                g57_narrow: 0,
                ladder_cap: 0,
                cg_jitter: 0,
                rung_menu: 0,

                epoch: 0,
                refill_data: 0,
                single: 0,
                gray_fold: 0,
                swap_refresh: 0,
                close_slice: 0,
            };
            let band_writes = |g: &CnotCircuit| -> usize {
                g.gates
                    .iter()
                    .filter(|gate| (gate.target as usize) >= 2 * n)
                    .count()
            };
            for seed in 0..3u64 {
                let mut rng = StdRng::seed_from_u64(0x0b0d_0000 + seed);
                let rolled = gadgetize_cnot(&main, n, 2, &MaskConfig::off(), &cfg(1), &mut rng);
                assert_eq!(rolled.num_wires, 2 * n + band, "rolls cost no wires");
                for input in 0..(1u64 << rolled.num_wires) {
                    let expected = main.evaluate((input & mask) as usize) as u64 & mask;
                    assert_eq!(
                        eval_u64(&rolled.gates, input) & mask,
                        expected,
                        "max_width={max_width} seed={seed} input={input:#x}"
                    );
                }
                let mut rng = StdRng::seed_from_u64(0x0b0d_0000 + seed);
                let still = gadgetize_cnot(&main, n, 2, &MaskConfig::off(), &cfg(0), &mut rng);
                assert!(
                    band_writes(&rolled) > 2 * band_writes(&still),
                    "max_width={max_width} seed={seed}: rolling barely touched the band \
                     ({} writes vs {} without rolls)",
                    band_writes(&rolled),
                    band_writes(&still)
                );
            }
        }
    }

    #[test]
    fn prod_narrow_gadget_round_trips_in_the_g57_vocabulary() {
        // Full narrow gadget (mixed [2,3] plan + nonlinear cascaded band fill
        // + mirror): every gate is within the phase-A DB width, no wire is
        // added over the wide build, and the endpoint contract holds for
        // ARBITRARY junk on every non-data wire (dirty borrows, nothing
        // pinned). Also records the true-g57 share.
        let n = MASKED_TEST_N;
        let main = masked_test_main();
        let mask = (1u64 << n) - 1;
        let cfg = ProdConfig {
            k: 1,
            deg: 2,
            k_hi: 1,
            deg_hi: 3,
            band: 6,
            rsrc: 1,
            max_width: 2,
            fill_nl: 2,
            roll: 0,
            src_dist: 0,
            src_horizon: 0,
            src_lo: 0,
            src_hi: 0,
            fill_pivots: 0,
            g57_narrow: 0,
            ladder_cap: 0,
            cg_jitter: 0,
            rung_menu: 0,

            epoch: 0,
            refill_data: 0,
            single: 0,
            gray_fold: 0,
            swap_refresh: 0,
            close_slice: 0,
        };
        for seed in 0..3u64 {
            let mut rng = StdRng::seed_from_u64(0xa550_0000 + seed);
            let g = gadgetize_cnot(&main, n, 2, &MaskConfig::off(), &cfg, &mut rng);
            // No scratch region: narrow mode costs exactly zero extra wires.
            assert_eq!(g.num_wires, 2 * n + 6);
            let mut g57s = 0usize;
            for gate in &g.gates {
                assert!(!gate.ctrls.is_empty(), "bare X in narrow gadget");
                assert!(gate.width() <= 2, "wide gate in narrow gadget: {gate:?}");
                let mut pols: Vec<bool> = gate.ctrls.iter().map(|&(_, p)| p).collect();
                pols.sort_unstable();
                if gate.width() == 2 && gate.comp && pols == vec![false, true] {
                    g57s += 1;
                }
            }
            // Ladder rungs must be EXACT, and an exact 2-control conjunction
            // is not a sum of g57s (each g57 carries a constant 1: an odd
            // count leaves a stray 1, an even count collapses the monomials
            // to a plain XOR). So rungs are comp=0 width-2 gates — still
            // inside the phase-A DB width, which filters on width, not comp.
            assert!(g57s > 0, "seed={seed}: no g57s at all");
            // Full domain: arbitrary junk on carriers and band alike.
            for input in 0..(1u64 << g.num_wires) {
                let expected = main.evaluate((input & mask) as usize) as u64 & mask;
                assert_eq!(eval_u64(&g.gates, input) & mask, expected, "seed={seed}");
            }
        }
    }

    /// Distributed sourcing: the encoding with NO band at all.
    ///
    /// Exactness is the barrier's own test. A mask whose source wire is
    /// written between injection and strip fails to cancel — the strip emits
    /// the same conjunction over a bit that has since changed — so the value
    /// decodes wrong and the endpoint moves. Running the full input domain
    /// with RG traffic (rg_freq 2, so RG1/RG2 re-pair and RG3 refreshes fire
    /// throughout) and re-source churn on top exercises every release path.
    /// The gate-local non-completeness that no endpoint test can see is
    /// asserted at emission time inside `debug_check_fragment`.
    #[test]
    fn prod_distributed_sourcing_is_exact_and_costs_no_wires() {
        // Width 6, not the 4-wire fixture: with one factor per owning value,
        // a degree-3 slot needs three values besides its own, so n = 4 leaves
        // the draw no freedom at all (and the dedup set nothing to draw from).
        let n = 6usize;
        let main = masked_test_main_wide(n);
        let mask = (1u64 << n) - 1;
        for max_width in [0usize, 2] {
            let cfg = ProdConfig {
                k: 1,
                deg: 2,
                k_hi: 1,
                deg_hi: 3,
                band: 0,
                rsrc: 1,
                max_width,
                fill_nl: 0,
                roll: 0,
                src_dist: 1,
                src_horizon: 0,
                src_lo: 0,
                src_hi: 0,
                fill_pivots: 0,
                g57_narrow: 0,
                ladder_cap: 0,
                cg_jitter: 0,
                rung_menu: 0,

                epoch: 0,
                refill_data: 0,
                single: 0,
                gray_fold: 0,
                swap_refresh: 0,
                close_slice: 0,
            };
            for seed in 0..2u64 {
                let mut rng = StdRng::seed_from_u64(0x0d15_0000 + seed);
                let g = gadgetize_cnot(&main, n, 2, &MaskConfig::off(), &cfg, &mut rng);
                assert_eq!(
                    g.num_wires,
                    2 * n,
                    "distributed sourcing must not widen the gadget"
                );
                assert!(
                    g.gates.iter().all(|gate| !gate.ctrls.is_empty()),
                    "distributed build must not contain a bare X"
                );
                // Arbitrary junk on every non-data wire, exhaustively.
                for input in 0..(1u64 << g.num_wires) {
                    let expected = main.evaluate((input & mask) as usize) as u64 & mask;
                    assert_eq!(
                        eval_u64(&g.gates, input) & mask,
                        expected,
                        "max_width={max_width} seed={seed} input={input:#x}"
                    );
                }
            }
        }
    }

    /// Single-carrier decode `v = c_v ^ masks ^ κ`: exact under arbitrary band
    /// junk, on n carriers instead of 2n. Exactness is the real test here —
    /// the strip only cancels if every mask term still denotes the bit it did
    /// at injection, which is what makes the frozen band load-bearing, and the
    /// relocations must leave value v back on wire v at the end.
    #[test]
    fn prod_single_carrier_is_exact_on_half_the_wires() {
        let n = 6usize;
        let main = masked_test_main_wide(n);
        let mask = (1u64 << n) - 1;
        // [1,2,3,3] and [1,2,2,3]: one linear term, the rest nonlinear.
        // Rolls on: a roll can leave a value sitting on a former band wire, so
        // the final routing has to be a full permutation, not a carrier-space
        // one. That is exactly what this exercises.
        for (k, deg, k_hi, deg_hi, roll) in [
            (1usize, 2usize, 2usize, 3usize, 0usize),
            (2, 2, 1, 3, 0),
            (1, 2, 2, 3, 1),
        ] {
            let cfg = ProdConfig {
                k,
                deg,
                k_hi,
                deg_hi,
                band: 8,
                rsrc: 1,
                max_width: 0,
                fill_nl: 2,
                roll,
                src_dist: 0,
                src_horizon: 0,
                src_lo: 0,
                src_hi: 0,
                fill_pivots: 0,
                g57_narrow: 0,
                ladder_cap: 0,
                cg_jitter: 0,
                rung_menu: 0,

                epoch: 0,
                refill_data: 0,
                single: 1,
                gray_fold: 0,
                swap_refresh: 0,
                close_slice: 0,
            };
            for seed in 0..3u64 {
                let mut rng = StdRng::seed_from_u64(0x51_0000 + seed);
                let g = gadgetize_cnot_single(&main, n, 2, &cfg, &mut rng);
                assert_eq!(g.num_wires, n + 8, "single carrier: n carriers, not 2n");
                assert!(
                    g.gates.iter().all(|gate| !gate.ctrls.is_empty()),
                    "single-carrier build must not contain a bare X"
                );
                for input in 0..(1u64 << g.num_wires) {
                    let expected = main.evaluate((input & mask) as usize) as u64 & mask;
                    assert_eq!(
                        eval_u64(&g.gates, input) & mask,
                        expected,
                        "plan [{deg}x{k},{deg_hi}x{k_hi}] seed={seed} input={input:#x}"
                    );
                }
            }
        }
    }

    /// Selective laddering: fold fragments of width in (2, cap] are realized
    /// over BORROWED DIRTY carriers instead of as one wide gate.
    ///
    /// The borrows are the whole risk. A ladder parks partial products on
    /// wires it does not own, so it is exact only if every borrow is visited
    /// an even number of times and restored before anything else reads it --
    /// and the borrow pool now has to dodge the target's sibling carrier and
    /// every operand's sibling, or one gate ends up seeing both carriers of a
    /// single value. Run the full input domain (band junk included, since the
    /// high wires are unconstrained) at several ceilings, and check that the
    /// fossil count actually falls -- an exactness test alone would pass on a
    /// ladder_cap that silently did nothing.
    #[test]
    fn prod_laddering_is_exact_and_removes_wide_gates() {
        let n = 6usize;
        let main = masked_test_main_wide(n);
        let mask = (1u64 << n) - 1;
        let build = |ladder_cap: usize, seed: u64| {
            let cfg = ProdConfig {
                k: 1,
                deg: 2,
                k_hi: 2,
                deg_hi: 3,
                band: 8,
                rsrc: 1,
                max_width: 0,
                fill_nl: 2,
                roll: 1,
                src_dist: 0,
                src_horizon: 0,
                src_lo: 0,
                src_hi: 0,
                fill_pivots: 0,
                g57_narrow: 1,
                ladder_cap,
                cg_jitter: 0,
                rung_menu: 0,
                epoch: 0,
                refill_data: 0,
                single: 1,
                gray_fold: 0,
                swap_refresh: 0,
                close_slice: 0,
            };
            let mut rng = StdRng::seed_from_u64(0x1add_0000 + seed);
            gadgetize_cnot_single(&main, n, 2, &cfg, &mut rng)
        };
        let wide = |g: &CnotCircuit| g.gates.iter().filter(|x| x.ctrls.len() > 2).count();
        let base = wide(&build(0, 0));
        for cap in [3usize, 4, 6] {
            let g = build(cap, 0);
            assert!(
                wide(&g) < base,
                "ladder_cap {cap} removed no wide gates ({} vs baseline {base})",
                wide(&g)
            );
            for seed in 0..2u64 {
                let g = build(cap, seed);
                for input in 0..(1u64 << g.num_wires) {
                    let expected = main.evaluate((input & mask) as usize) as u64 & mask;
                    assert_eq!(
                        eval_u64(&g.gates, input) & mask,
                        expected,
                        "ladder_cap={cap} seed={seed} input={input:#x}"
                    );
                }
            }
        }
        // A ceiling above every fragment width must leave nothing wide behind.
        assert_eq!(
            wide(&build(64, 0)),
            0,
            "an unbounded ceiling still left wide gates"
        );
    }

    /// The DEFAULT config is the hardened construction, not a bare encoding.
    ///
    /// `production_single` spent a day as a free-standing constant with no
    /// callers, so every lever it named was off in every circuit anyone built
    /// while the docs described it as "the validated production setting". Both
    /// entry points now build from it, and this pins the values so a revert to
    /// the old all-zero defaults fails here rather than silently shipping a
    /// materially weaker gadget.
    #[test]
    fn production_preset_is_the_hardened_construction() {
        let p = ProdConfig::production_single();
        assert!(p.enabled(), "the default must have the encoding ON");
        // [2,2,2,3] -- three degree-2 mask terms and one degree-3, replacing
        // [2,3,3]. A degree-2 atom is the stronger STATISTICAL masker (piling-up
        // factor 0.5 against 0.75) and the weaker ALGEBRAIC one (it sits inside
        // a degree-2 exact adversary's span), so the mix trades one against the
        // other: eps 0.09375 against 0.28125, measured leak 3.2x lower, at 14%
        // FEWER gates. The single surviving degree-3 atom is what keeps the plan
        // out of degree-2 exact reach; drop it and the value is recoverable
        // exactly, which is why deg_hi and k_hi >= 1 are pinned here.
        assert_eq!(
            (p.k, p.deg, p.k_hi, p.deg_hi),
            (3, 2, 1, 3),
            "plan is [2,2,2,3]"
        );
        assert!(
            p.k_hi >= 1 && p.deg_hi >= 3,
            "at least one degree-3 atom, or the value drops into degree-2 exact range"
        );
        assert_eq!(p.single, 1, "single-carrier decode");
        assert_eq!(p.band, 0, "band 0 == match the value count");
        assert_eq!(p.band_size(128), 128, "band 0 must resolve to n");
        assert!(
            p.rsrc >= 1,
            "single-carrier mode needs a representation refresh"
        );
        assert_eq!(p.fill_nl, 2, "nonlinear band fill");
        assert_eq!(
            p.roll, 1,
            "rolling band -- without it the write census separates"
        );
        assert_eq!(
            p.g57_narrow, 1,
            "narrow fragments in the store's vocabulary"
        );
        // Selective laddering at cap 3: the expanded fold's wide product
        // fragments must be re-spelled into the g57/CNOT vocabulary for
        // fmix (cap 4's extra narrowness arrives as store-weak plain conj-2
        // gates at twice the size), and the ladder's scratch is the band
        // pool under swap mode (live-carrier borrows exposed data states in
        // the chain deltas).
        assert_eq!(p.ladder_cap, 3, "expanded-fold production needs the selective ladder");
        // The Gray fold is ON: the fold emits no wide fragment at all, and
        // store-reachability goes 31.55% -> 95.47% (97.53% at this mask plan).
        assert_eq!(p.gray_fold, 1, "the Gray-code fold is the default CG");
        assert_eq!(
            p.rung_menu, 1,
            "free spelling variability is on -- it costs nothing"
        );
        assert_eq!(p.cg_jitter, 50, "block-count entropy at its maximum");
        // A frozen band is recoverable by FUNCTION LIFETIME alone, so some
        // channel must turn the band's functions over. Since 2026-08-24 that
        // is the drain set rather than `epoch`: it steers retirements the fold
        // already makes instead of paying to release a live variable, so it
        // buys ~2x the turnovers for less than `epoch` cost. Exactly one of
        // the two should be running -- both is double payment, neither is a
        // frozen band.
        assert!(
            (p.epoch > 0) ^ (p.swap_refresh > 0 && drain_cap(&p, p.band_size(128)) > 0),
            "a frozen band is recoverable by function lifetime: run the drain set \
             (swap_refresh > 0) or the epoch channel, not both and not neither"
        );
        assert_eq!(p.fill_pivots, 0, "band = n leaves the pivot block no room");
        // The 2026-08-20 redesign: without the per-gate swap the masks cancel
        // in every fold's before/after XOR (carrier delta == source delta,
        // measured 100% on linear gates); without the closing block the
        // zero-slice phase exists at the input port only.
        assert_eq!(
            p.swap_refresh, 3,
            "per-gate mask swap-with-refresh is the default, at the 3 retirement \
             sides that buy 14.7 band turnovers for +0.3% gates (2 = the 2026-08-20 stream)"
        );
        assert_eq!(p.close_slice, 1, "the closing zero-slice block is the default");
    }

    #[test]
    fn no_gray_phase_a_preset_changes_only_the_fold_strategy() {
        let gray = ProdConfig::production_single();
        let safe = ProdConfig::production_single_no_gray_phase_a();
        assert_eq!(
            safe.gray_fold, 0,
            "must not gather an aggregate operand mask"
        );
        assert_eq!(safe.ladder_cap, 4, "measured selective-narrow ceiling");

        let mut expected = gray;
        expected.gray_fold = 0;
        expected.ladder_cap = 4;
        assert_eq!(
            safe, expected,
            "preset drifted beyond its two measured levers"
        );
    }

    /// `strip_all`'s constant discharge must not read the target's SIBLING.
    ///
    /// It emits `target ^= !u` then `target ^= u` to realize a bare constant
    /// without an X gate, and it drew `u` from the index range
    /// `0..carrier_total` excluding only the target -- so on a two-carrier build
    /// the target value's other carrier was a legal draw, and that first gate
    /// then read one carrier of a value while writing the other: a whole
    /// sharing inside one gate. It is the role-versus-index confusion this file
    /// has now had three times, invisible under `--prod-single 1` because there
    /// is no sibling to hit, which is why the suite passed with it present.
    ///
    /// Tested on `strip_all` directly rather than on a finished gadget: the
    /// sharing BOOKENDS legitimately touch both carriers (that is how the
    /// sharing is created), so a whole-circuit scan cannot separate the two.
    #[test]
    fn prod_strip_constant_never_reads_the_targets_sibling() {
        let n = MASKED_TEST_N;
        let cfg = ProdConfig {
            k: 1,
            deg: 2,
            k_hi: 1,
            deg_hi: 3,
            band: 10,
            rsrc: 1,
            fill_nl: 2,
            single: 0, // two carriers: this is the only mode with a sibling
            ..ProdConfig::off()
        };
        let carrier_total = 2 * n;
        let pairs: Vec<(usize, usize)> = (0..n).map(|v| (2 * v, 2 * v + 1)).collect();
        let state = GadgetState { n, pairs };
        for seed in 0..64u64 {
            let mut rng = StdRng::seed_from_u64(0x5721_0000 + seed);
            let mut ledger = ProdLedger::new(n, &cfg, carrier_total, None);
            let mut ramp = Vec::new();
            ledger.inject_all(&state, &mut rng, &mut ramp);
            // Force every value to owe a constant, so the discharge path runs
            // for all of them rather than whichever parity happened to land.
            for v in 0..n {
                ledger.consts[v] = true;
            }
            let mut out = Vec::new();
            ledger.strip_all(&state, &mut rng, &mut out);
            for (i, gate) in out.iter().enumerate() {
                let mut touched: Vec<u16> = gate.ctrls.iter().map(|&(w, _)| w).collect();
                touched.push(gate.target);
                for v in 0..n {
                    let (c0, c1) = (2 * v as u16, 2 * v as u16 + 1);
                    assert!(
                        !(touched.contains(&c0) && touched.contains(&c1)),
                        "seed={seed} strip gate {i} ({gate:?}) holds both carriers \
                         {c0},{c1} of value {v}"
                    );
                }
            }
        }
    }

    /// Retire-and-refill epochs: a band variable's VALUE changes mid-body.
    ///
    /// This is the exactness test that matters for the mechanism: a refill
    /// rewrites a wire that masks were reading a moment ago, so if the release
    /// step ever misses a live slot, that slot's strip cancels the wrong
    /// product and the endpoint moves. Run over the full input domain with
    /// arbitrary band junk, at both refill compositions (band-internal and
    /// carrier-injecting) and with rolling on, since a roll relocates the very
    /// variable the next epoch retires.
    #[test]
    fn prod_retire_refill_is_exact_under_arbitrary_junk() {
        let n = MASKED_TEST_N;
        let main = masked_test_main();
        let mask = (1u64 << n) - 1;
        for (epoch, refill_data, roll) in [
            (1usize, 0usize, 0usize),
            (1, 100, 0),
            (2, 50, 1),
            (1, 50, 1),
        ] {
            let cfg = ProdConfig {
                k: 1,
                deg: 2,
                k_hi: 1,
                deg_hi: 3,
                band: 6,
                rsrc: 1,
                max_width: 0,
                fill_nl: 2,
                roll,
                src_dist: 0,
                src_horizon: 0,
                src_lo: 0,
                src_hi: 0,
                fill_pivots: 1,
                g57_narrow: 0,
                ladder_cap: 0,
                cg_jitter: 0,
                rung_menu: 0,
                epoch,
                refill_data,
                single: 0,
                gray_fold: 0,
                swap_refresh: 0,
                close_slice: 0,
            };
            for seed in 0..3u64 {
                let mut rng = StdRng::seed_from_u64(0xbeef_0000 + seed);
                let g = gadgetize_cnot(&main, n, 2, &MaskConfig::off(), &cfg, &mut rng);
                assert!(
                    g.gates.iter().all(|gate| !gate.ctrls.is_empty()),
                    "retire-refill must not emit a bare X"
                );
                for input in 0..(1u64 << g.num_wires) {
                    let expected = main.evaluate((input & mask) as usize) as u64 & mask;
                    assert_eq!(
                        eval_u64(&g.gates, input) & mask,
                        expected,
                        "epoch={epoch} refill_data={refill_data} roll={roll} seed={seed} input={input:#x}"
                    );
                }
            }
        }
    }

    /// The point of the design: no wire is quiet, because no wire has the
    /// dedicated source role. In the band build the band wires are written
    /// only by the two fills — a census separates them from the carriers by a
    /// single threshold. Here every wire must be written by BODY traffic.
    #[test]
    fn prod_distributed_sourcing_leaves_no_quiet_wire() {
        let n = 6usize;
        let main = masked_test_main_wide(n);
        let cfg = |src_dist| ProdConfig {
            k: 1,
            deg: 2,
            k_hi: 1,
            deg_hi: 3,
            band: 8,
            rsrc: 1,
            max_width: 0,
            fill_nl: 2,
            roll: 0,
            src_dist,
            src_horizon: 0,
            src_lo: 0,
            src_hi: 0,
            fill_pivots: 0,
            g57_narrow: 0,
            ladder_cap: 0,
            cg_jitter: 0,
            rung_menu: 0,

            epoch: 0,
            refill_data: 0,
            single: 0,
            gray_fold: 0,
            swap_refresh: 0,
            close_slice: 0,
        };
        let mut rng = StdRng::seed_from_u64(0x0d16_0001);
        let dist = gadgetize_cnot(&main, n, 2, &MaskConfig::off(), &cfg(1), &mut rng);
        let mut rng = StdRng::seed_from_u64(0x0d16_0001);
        let band = gadgetize_cnot(&main, n, 2, &MaskConfig::off(), &cfg(0), &mut rng);
        assert_eq!(dist.num_wires, 2 * n, "no band wires");
        assert_eq!(band.num_wires, 2 * n + 8, "band build keeps its band");

        // Body = everything strictly between the ports, so the input/output
        // fills (which write every band wire in the band build) do not mask
        // the distinction being measured.
        let writes = |g: &CnotCircuit, wires: usize| -> Vec<usize> {
            let lo = g.gates.len() / 4;
            let hi = g.gates.len() - g.gates.len() / 4;
            let mut w = vec![0usize; wires];
            for gate in &g.gates[lo..hi] {
                w[gate.target as usize] += 1;
            }
            w
        };
        let dist_w = writes(&dist, dist.num_wires);
        assert!(
            dist_w.iter().all(|&c| c > 0),
            "distributed build left an unwritten wire in the body: {dist_w:?}"
        );
        let band_w = writes(&band, band.num_wires);
        assert!(
            band_w[2 * n..].iter().all(|&c| c == 0),
            "band wires should be body-static in the band build (the weakness \
             distributed sourcing removes): {:?}",
            &band_w[2 * n..]
        );
    }

    /// The reserved pivot block makes the band JOINTLY uniform, not merely
    /// balanced wire by wire.
    ///
    /// This is the property every statistical claim about the encoding needs
    /// and the one the marginal test cannot see: a mask multiplies three band
    /// wires, so what matters is the joint law. Checking it means checking
    /// EVERY nonempty subset XOR is balanced — a subset that is biased is a
    /// direction in which the band is predictable.
    ///
    /// The legacy draw fails this (pivots are drawn with replacement and only
    /// a wire's OWN pivot is excluded from its own material), which is why the
    /// comparison against it is part of the test rather than folklore.
    #[test]
    fn prod_reserved_pivots_make_the_band_jointly_uniform() {
        let n = 10usize;
        let b = 5usize;
        let band: Vec<u16> = (n as u16..(n + b) as u16).collect();
        let subset_bias = |gates: &[XGate]| -> f64 {
            let mut worst = 0f64;
            for mask in 1u32..(1 << b) {
                let ones = (0..(1u64 << n))
                    .filter(|&x| {
                        let st = eval_u64(gates, x);
                        let mut parity = 0u64;
                        for (i, &w) in band.iter().enumerate() {
                            if mask >> i & 1 == 1 {
                                parity ^= (st >> w) & 1;
                            }
                        }
                        parity == 1
                    })
                    .count() as f64;
                let bias = (ones / (1u64 << n) as f64 - 0.5).abs();
                if bias > worst {
                    worst = bias;
                }
            }
            worst
        };
        let (mut reserved_worst, mut legacy_worst) = (0f64, 0f64);
        for seed in 0..12u64 {
            let mut rng = StdRng::seed_from_u64(0x91_0000 + seed);
            let mut g = Vec::new();
            emit_band_fill_nl_pivots(n, &band, 2, true, &mut rng, &mut g);
            reserved_worst = reserved_worst.max(subset_bias(&g));

            let mut rng = StdRng::seed_from_u64(0x91_0000 + seed);
            let mut g = Vec::new();
            emit_band_fill_nl_pivots(n, &band, 2, false, &mut rng, &mut g);
            legacy_worst = legacy_worst.max(subset_bias(&g));
        }
        assert_eq!(
            reserved_worst, 0.0,
            "reserved pivots must make EVERY subset XOR exactly balanced; worst bias {reserved_worst}"
        );
        assert!(
            legacy_worst > 0.0,
            "the legacy draw is supposed to be jointly biased — if this fires, the \
             comparison has stopped being meaningful (worst bias {legacy_worst})"
        );
        println!(
            "[pivot-block] worst subset bias: reserved {reserved_worst:.4} vs legacy {legacy_worst:.4}"
        );
    }

    #[test]
    fn prod_band_fill_nl_is_balanced_and_nonlinear() {
        // Every band wire's fill must be exactly balanced over uniform data
        // (the pivot guarantee), and the cascade must actually produce
        // nonlinearity in at least one band wire.
        let n = 10;
        // Deliberately NOT contiguous and not in wire order: after a roll the
        // fill's wire list is an arbitrary set (the mirror fill takes it as it
        // finds it), and the cascade's "earlier band wire" bookkeeping must
        // key on position in the list, not on wire index arithmetic.
        let band: Vec<u16> = vec![14, 10, 17, 11, 16, 12, 13, 15];
        let mut rng = StdRng::seed_from_u64(0xf111_0001);
        let mut gates = Vec::new();
        emit_band_fill_nl(n, &band, 2, &mut rng, &mut gates);
        let f = |x: u64| eval_u64(&gates, x);
        let mut any_nonlinear = false;
        for bw in band.iter().map(|&w| w as usize) {
            let bit = |x: u64| (f(x) >> bw) & 1;
            let ones: u64 = (0..(1u64 << n)).map(bit).sum();
            assert_eq!(ones, 1 << (n - 1), "band wire {bw} fill is biased");
            'nl: for i in 0..n {
                for j in (i + 1)..n {
                    let (ei, ej) = (1u64 << i, 1u64 << j);
                    if bit(ei ^ ej) ^ bit(ei) ^ bit(ej) ^ bit(0) != 0 {
                        any_nonlinear = true;
                        break 'nl;
                    }
                }
            }
        }
        assert!(
            any_nonlinear,
            "cascaded fill produced no nonlinear band wire"
        );
    }

    #[test]
    fn prod_gadgetize_cnot_preserves_the_first_n_wires() {
        let n = MASKED_TEST_N;
        let main = masked_test_main();
        let mask = (1u64 << n) - 1;
        for seed in 0..8u64 {
            let mut rng = StdRng::seed_from_u64(0x960d_0000 + seed);
            let prodded = gadgetize_cnot(
                &main,
                n,
                2,
                &MaskConfig::off(),
                &prod_test_config(),
                &mut rng,
            );
            assert_eq!(prodded.num_wires, 2 * n + 6);
            assert!(
                prodded.gates.iter().all(|g| !g.ctrls.is_empty()),
                "prod body must not contain a bare X"
            );
            for input in 0..(1u64 << prodded.num_wires) {
                let expected = main.evaluate((input & mask) as usize) as u64 & mask;
                assert_eq!(
                    eval_u64(&prodded.gates, input) & mask,
                    expected,
                    "seed={seed}"
                );
            }
            // Same seed, prod off: the encoding must actually have paid gates.
            let mut rng = StdRng::seed_from_u64(0x960d_0000 + seed);
            let plain = gadgetize_cnot(
                &main,
                n,
                2,
                &MaskConfig::off(),
                &ProdConfig::off(),
                &mut rng,
            );
            assert!(prodded.gates.len() > plain.gates.len());
        }
    }

    #[test]
    fn prod_gadgetize_xgates_preserves_the_low_wires() {
        let n = MASKED_TEST_N;
        let mask = (1u64 << n) - 1;
        let source = vec![
            XGate::from_g57([0, 1, 2]),
            XGate::cnot(0, 3),
            XGate::conj(2, [(0u16, true), (1u16, true)]).unwrap(),
            XGate::conj(1, [(3u16, false)]).unwrap(),
            XGate::from_g57([3, 0, 1]),
            XGate::x_gate(2),
            XGate::cnot(1, 0),
            XGate::from_g57([0, 2, 3]),
            XGate::conj(0, [(1u16, false), (2u16, true)]).unwrap(),
            XGate::from_g57([1, 3, 2]),
            XGate::cnot(2, 1),
        ];
        for seed in 0..8u64 {
            let mut rng = StdRng::seed_from_u64(0x960e_0000 + seed);
            let g = gadgetize_xgates(
                &source,
                n,
                2,
                &MaskConfig::off(),
                &prod_test_config(),
                &mut rng,
            );
            assert_eq!(g.num_wires, 2 * n + 6);
            for input in 0..(1u64 << g.num_wires) {
                let expected = eval_u64(&source, input & mask) & mask;
                assert_eq!(
                    eval_u64(&g.gates, input) & mask,
                    expected,
                    "input={input:#x}"
                );
            }
        }
    }

    #[test]
    fn prod_slice_zero_gadgetize_matches_on_the_zero_slice() {
        let n = MASKED_TEST_N;
        let main = masked_test_main();
        let mask = (1u64 << n) - 1;
        for seed in 0..4u64 {
            let mut rng = StdRng::seed_from_u64(0x960f_0000 + seed);
            let transformed = gadgetize_with_slice_zero_ccnot(
                &main,
                n,
                2,
                6 * n,
                &MaskConfig::off(),
                &prod_test_config(),
                &mut rng,
            );
            assert_eq!(transformed.num_wires, 2 * n + 6);
            for x in 0..=mask {
                let expected = main.evaluate(x as usize) as u64 & mask;
                assert_eq!(eval_u64(&transformed.gates, x) & mask, expected);
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
    fn ccnot_preblock_fixes_exactly_the_zero_slice() {
        // "Only the all-zero slice is fixed" is now a THEOREM of the pinned
        // target/control split (see slice_zero_ccnot_preblock), not a
        // measured tendency: exhaustively over every slice — aux, BAND, and
        // mixed — and every seed, at both widths and with the band present.
        for (n, band) in [(3usize, 0usize), (4, 0), (4, 2), (5, 3)] {
            let mask = (1u64 << n) - 1;
            let slices = 1u64 << (n + band);
            for seed in 0..8u64 {
                let mut rng = StdRng::seed_from_u64(0xcc00_0000 + seed);
                let preblock = slice_zero_ccnot_preblock(n, band, 6 * n, &mut rng);
                assert_eq!(preblock.gates.len(), 6 * n);
                assert_eq!(preblock.num_wires, 2 * n + band);
                for s in 0..slices {
                    let mut identity_on_slice = true;
                    for x in 0..=mask {
                        let input = x | (s << n);
                        let output = eval_u64(&preblock.gates, input);
                        assert_eq!(output >> n, s, "non-data wires must pass through");
                        if s == 0 {
                            assert_eq!(output, input, "zero slice must be fixed");
                        } else if output != input {
                            identity_on_slice = false;
                        }
                    }
                    if s != 0 {
                        assert!(
                            !identity_on_slice,
                            "seed={seed:#x} n={n} band={band} slice s={s:#x} is also fixed"
                        );
                    }
                }
            }
        }
    }

    /// Brute-force version of the exactness property, valid for any gate
    /// degree: enumerate every slice and every input.
    fn only_zero_slice_is_fixed(gates: &[XGate], n: usize, nondata: usize) -> bool {
        let mask = (1u64 << n) - 1;
        (1..(1u64 << nondata))
            .all(|s| (0..=mask).any(|x| eval_u64(gates, x | (s << n)) & mask != x))
    }

    #[test]
    fn nonlinear_preblock_weight2_decomposition_is_exact_and_bounded() {
        let (n, nondata, logical_gates) = (8usize, 20usize, 200usize);
        let scratch = n as u16;
        let scratch2 = (n + 1) as u16;
        let mut wide_rng = StdRng::seed_from_u64(0xcc88_0001);
        let wide = try_nonlinear_slice_zero_preblock_dims(
            n,
            nondata,
            logical_gates,
            false,
            scratch,
            scratch2,
            &mut wide_rng,
        )
        .unwrap();
        let mut weight2_rng = StdRng::seed_from_u64(0xcc88_0001);
        let weight2 = try_nonlinear_slice_zero_preblock_dims(
            n,
            nondata,
            logical_gates,
            true,
            scratch,
            scratch2,
            &mut weight2_rng,
        )
        .unwrap();

        let quads = (logical_gates - logical_gates / 3) / 2;
        assert_eq!(wide.gates.len(), logical_gates);
        assert_eq!(weight2.gates.len(), logical_gates + 3 * quads);
        assert!(weight2.gates.iter().all(|gate| gate.ctrls.len() <= 2));
        assert_eq!(wide.num_wires, n + nondata);
        assert_eq!(weight2.num_wires, n + nondata);

        for input in 0..(1u64 << n) {
            assert_eq!(eval_u64(&wide.gates, input), input);
            assert_eq!(eval_u64(&weight2.gates, input), input);
        }

        let mut state_rng = StdRng::seed_from_u64(0xcc88_0002);
        let state_mask = (1u64 << (n + nondata)) - 1;
        let dirty_q_mask = (1u64 << scratch) | (1u64 << scratch2);
        for _ in 0..512 {
            let state = rand::RngCore::next_u64(&mut state_rng) & state_mask;
            let decomposed = eval_u64(&weight2.gates, state);
            assert_eq!(
                decomposed,
                eval_u64(&wide.gates, state),
                "dirty-q decomposition changed the preblock function"
            );
            assert_eq!(
                decomposed & dirty_q_mask,
                state & dirty_q_mask,
                "dirty-q decomposition did not restore its scratch wires"
            );
        }
    }

    #[test]
    fn ccnot_preblock_is_quadratic_in_the_data_off_slice() {
        // With one data control per gate the block is AFFINE in x for every
        // fixed slice, whatever the gate count: each gate becomes a constant
        // flip or a transvection, and those compose to an affine map. The
        // three-control gates are there to break that, so at least one slice
        // must show a genuine second-order term:
        //   S(a^b) ^ S(a) ^ S(b) ^ S(0)  !=  0.
        let (n, band) = (8usize, 4usize);
        let mask = (1u64 << n) - 1;
        let mut nonlinear_slices = 0usize;
        for seed in 0..4u64 {
            let mut rng = StdRng::seed_from_u64(0xcc60_0000 + seed);
            let preblock = slice_zero_ccnot_preblock(n, band, 10 * n, &mut rng);
            let s = |x: u64, slice: u64| eval_u64(&preblock.gates, x | (slice << n)) & mask;
            for slice in 1..(1u64 << (n + band)) {
                let base = s(0, slice);
                let quadratic = (0..n).any(|i| {
                    ((i + 1)..n).any(|j| {
                        let (a, b) = (1u64 << i, 1u64 << j);
                        s(a ^ b, slice) ^ s(a, slice) ^ s(b, slice) ^ base != 0
                    })
                });
                if quadratic {
                    nonlinear_slices += 1;
                }
            }
        }
        assert!(
            nonlinear_slices > 0,
            "the preblock is affine in x on every slice — the three-control \
             gates are not doing their job"
        );
    }

    #[test]
    fn prod_slice_zero_gadget_carries_three_control_preblock_gates() {
        // End to end: the gadget must compute C on the zero slice, and the
        // preblock's three-control gates — the ones that keep the off-slice
        // disturbance from being affine in x — must survive into the emitted
        // circuit.
        let n = 6;
        let band = 6;
        let mask = (1u64 << n) - 1;
        let main = CircuitSeq {
            gates: vec![[0, 1, 2], [2, 0, 1], [1, 2, 0], [3, 4, 5], [5, 3, 4]],
        };
        let cfg = ProdConfig {
            k: 2,
            deg: 2,
            band,
            rsrc: 1,
            roll: 1,
            ..ProdConfig::off()
        };
        for seed in 0..3u64 {
            let mut rng = StdRng::seed_from_u64(0xcc70_0000 + seed);
            let g = gadgetize_with_slice_zero_ccnot(
                &main,
                n,
                2,
                10 * n,
                &MaskConfig::off(),
                &cfg,
                &mut rng,
            );
            assert_eq!(g.num_wires, 2 * n + band);
            for x in 0..=mask {
                let expected = main.evaluate(x as usize) as u64 & mask;
                assert_eq!(eval_u64(&g.gates, x) & mask, expected, "seed={seed} x={x}");
            }
            // A preblock three-control gate: target and two controls in the
            // data half, exactly one control in the slice half, all positive.
            let has_quad = g.gates.iter().any(|gate| {
                gate.ctrls.len() == 3
                    && !gate.comp
                    && (gate.target as usize) < n
                    && gate.ctrls.iter().all(|&(_, p)| p)
                    && gate
                        .ctrls
                        .iter()
                        .filter(|&&(w, _)| (w as usize) < n)
                        .count()
                        == 2
                    && gate
                        .ctrls
                        .iter()
                        .filter(|&&(w, _)| (w as usize) >= n)
                        .count()
                        == 1
            });
            assert!(
                has_quad,
                "seed={seed}: no three-control preblock gate survived"
            );
        }
    }

    #[test]
    fn ccnot_preblock_builds_across_the_supported_widths() {
        // The constructor rejects and redraws until no nonzero slice is
        // fixed, which can fail outright when the data half is too narrow to
        // disturb every slice, so it must be exercised across the widths the
        // gadget paths actually reach — at the default 10n budget and at the
        // bare minimum of one gate per non-data wire. Where the space is small
        // enough, exactness is re-checked here too.
        for n in 3..=10usize {
            for band in [0usize, 2, 5, 8] {
                // The bare-minimum budget (one gate per slice wire) is only
                // claimed where the data half is wide enough to disturb every
                // slice with that few gates; where it is not, the constructor
                // says so with a panic rather than emitting a weak block.
                let budgets: &[usize] = if n * n / 4 >= n + band {
                    &[n + band, 10 * n]
                } else {
                    &[10 * n]
                };
                for &gate_count in budgets {
                    let mut rng = StdRng::seed_from_u64(0xcc50_0000 + (n * 32 + band) as u64);
                    let preblock = slice_zero_ccnot_preblock(n, band, gate_count, &mut rng);
                    assert_eq!(preblock.gates.len(), gate_count);
                    assert_eq!(preblock.num_wires, 2 * n + band);
                    // Exactness by brute force where the space is small — the
                    // block is quadratic in x now, so the affine shortcut the
                    // fallback uses does not apply here.
                    if n <= 6 && n + band <= 10 {
                        assert!(
                            only_zero_slice_is_fixed(&preblock.gates, n, n + band),
                            "n={n} band={band} gates={gate_count}: some nonzero slice is fixed"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn ccnot_preblock_band_slices_are_disturbed_like_aux_slices() {
        // The point of putting the band in the slice: flipping ONE band wire
        // must junk the data exactly as flipping one aux wire does. (Before,
        // the band was outside the preblock entirely, so every band-only
        // slice was provably fixed — a one-query aux/band distinguisher.)
        let n = 8;
        let band = 5;
        let mask = (1u64 << n) - 1;
        for seed in 0..8u64 {
            let mut rng = StdRng::seed_from_u64(0xcc40_0000 + seed);
            let preblock = slice_zero_ccnot_preblock(n, band, 10 * n, &mut rng);
            for w in 0..(n + band) {
                let s = 1u64 << w;
                let disturbed =
                    (0..=mask).any(|x| eval_u64(&preblock.gates, x | (s << n)) & mask != x);
                assert!(disturbed, "seed={seed:#x} single-wire slice {w} is fixed");
            }
        }
    }

    #[test]
    fn ccnot_preblock_uses_only_the_agreed_gate_shapes() {
        let n = 6;
        let band = 3;
        let gate_count = 6 * n;
        let mut rng = StdRng::seed_from_u64(0xcc10_0000);
        let preblock = slice_zero_ccnot_preblock(n, band, gate_count, &mut rng);
        let mut cnots = 0usize;
        let mut ccnots = 0usize;
        let mut quads = 0usize;
        let mut slice_controls = std::collections::HashSet::new();
        let mut targets = std::collections::HashSet::new();
        let mut data_controls = std::collections::HashSet::new();
        for gate in &preblock.gates {
            assert!(!gate.comp, "no complemented gates");
            assert!((gate.target as usize) < n, "targets stay in the data half");
            assert!(gate.ctrls.iter().all(|&(_, positive)| positive));
            targets.insert(gate.target);
            // ctrls are sorted by wire, so data controls come before slice
            // controls, and there is exactly one slice control per gate.
            let data: Vec<u16> = gate
                .ctrls
                .iter()
                .map(|&(w, _)| w)
                .filter(|&w| (w as usize) < n)
                .collect();
            let slice: Vec<u16> = gate
                .ctrls
                .iter()
                .map(|&(w, _)| w)
                .filter(|&w| (w as usize) >= n)
                .collect();
            assert_eq!(slice.len(), 1, "every gate reads exactly one slice wire");
            slice_controls.insert(slice[0]);
            data_controls.extend(data.iter().copied());
            match data.len() {
                0 => cnots += 1,
                1 => ccnots += 1,
                2 => quads += 1,
                other => panic!("unexpected data-control count {other}"),
            }
        }
        assert_eq!(cnots, gate_count / 3);
        assert_eq!(ccnots + quads, gate_count - gate_count / 3);
        // Three-control gates are what make the disturbance quadratic in x.
        assert!(quads > 0, "no three-control gates emitted");
        // Deliberately UNSTRUCTURED: a data wire is free to be a target of one
        // gate and a control of another. A disjoint target/control split would
        // buy an exactness theorem, but it also exempts the control pool from
        // ever being disturbed and lets an adversary switch the nonlinearity
        // off by zeroing it.
        assert!(
            targets.intersection(&data_controls).count() > 0,
            "targets and data controls should overlap freely"
        );
        // Every non-data wire, band included, is read by the block: a wire
        // nothing reads could not be pinned.
        assert_eq!(slice_controls.len(), n + band);

        // Uniform order: the CNOTs must be interleaved with the wider gates,
        // not bunched into a contiguous run (deterministic under the fixed
        // seed; a uniform shuffle makes a contiguous run astronomically
        // unlikely).
        let kinds: Vec<usize> = preblock.gates.iter().map(|g| g.ctrls.len()).collect();
        let first_cnot = kinds.iter().position(|&k| k == 1).unwrap();
        let last_cnot = kinds.iter().rposition(|&k| k == 1).unwrap();
        assert!(
            kinds[first_cnot..=last_cnot].iter().any(|&k| k > 1),
            "CNOTs and wider gates should be interleaved"
        );
    }

    #[test]
    fn slice_zero_ccnot_gadgetize_matches_only_on_the_zero_slice() {
        let n = 3;
        let mask = (1u64 << n) - 1;
        let main = CircuitSeq {
            gates: vec![[0, 1, 2], [2, 0, 1], [1, 2, 0]],
        };
        for seed in 0..8u64 {
            let mut rng = StdRng::seed_from_u64(0xcc20_0000 + seed);
            let transformed = gadgetize_with_slice_zero_ccnot(
                &main,
                n,
                2,
                6 * n,
                &MaskConfig::off(),
                &ProdConfig::off(),
                &mut rng,
            );
            assert_eq!(transformed.num_wires, 2 * n);
            for x in 0..=mask {
                let expected = main.evaluate(x as usize) as u64 & mask;
                assert_eq!(eval_u64(&transformed.gates, x) & mask, expected);
            }
            // The gadget contract holds for any second-half value and the
            // original is a permutation, so a wrong slice reproduces C at x
            // exactly when the preblock fixes (x, a) — and each nonzero
            // slice must disturb at least one x.
            for a in 1..=mask {
                let disturbed = (0..=mask).any(|x| {
                    let input = x | (a << n);
                    let expected = main.evaluate(x as usize) as u64 & mask;
                    eval_u64(&transformed.gates, input) & mask != expected
                });
                assert!(disturbed, "seed={seed:#x} slice a={a:#x} still computes C");
            }
        }
    }

    /// A scaled-down production-shaped config for the swap-refresh tests:
    /// single carrier, [2,2,2,3] plan, band = n, churn on, swap on. Gray is
    /// left at the production default (1) deliberately — swap mode must
    /// decline it and still verify.
    fn swap_test_config() -> ProdConfig {
        let mut p = ProdConfig::production_single();
        p.cg_jitter = 0;
        // The production rate, so every exactness test in this group runs with
        // a LIVE DRAIN SET: band variables are rewritten mid-body while masks
        // are being drawn and retired around them, which is precisely where a
        // bookkeeping slip would corrupt the endpoint.
        p.swap_refresh = 3;
        p.close_slice = 1;
        // At toy n the auto band (= n) leaves a value's disjointness draw a
        // single free pair, and the per-gate refresh churn exhausts its four
        // polarity variants; production bands are orders of magnitude wider.
        p.band = 24;
        p
    }

    fn swap_test_source(n: u16, rng: &mut StdRng) -> Vec<XGate> {
        // A mix of every source shape the sandwich feeds the gadgetizer:
        // CNOT, NCNOT, g57, and 2-3-control conjunctions.
        fn distinct(n: u16, taken: &[u16], rng: &mut StdRng) -> u16 {
            loop {
                let w = rng.random_range(0..n);
                if !taken.contains(&w) {
                    return w;
                }
            }
        }
        let mut gates = Vec::new();
        for _ in 0..60 {
            let t = rng.random_range(0..n);
            let a = distinct(n, &[t], rng);
            let b = distinct(n, &[t, a], rng);
            let gate = match rng.random_range(0..5) {
                0 => XGate::cnot(t, a),
                1 => XGate::conj(t, [(a, false)]).unwrap(),
                2 => XGate::from_g57([t, a, b]),
                3 => XGate::conj(t, [(a, true), (b, true)]).unwrap(),
                _ => {
                    let c = distinct(n, &[t, a, b], rng);
                    XGate::conj(t, [(a, true), (b, false), (c, true)]).unwrap()
                }
            };
            gates.push(gate);
        }
        gates
    }

    /// The drain set turns the band over by SCHEDULING rather than by luck,
    /// and the turnover count scales with the retirement rate.
    ///
    /// Exactness under a live drain set is covered by the zero-slice tests
    /// below (`swap_test_config` runs at the production rate), so this one
    /// checks the mechanism: variables actually reach zero references and get
    /// rewritten, more retirement sides buy more turnovers, and the reference
    /// bookkeeping the whole thing rests on agrees with the live slots at the
    /// end. `rewrite_var` asserts unconditionally that nothing names a
    /// variable it overwrites, so a passing build is also evidence for the one
    /// invariant that would silently corrupt the decode.
    #[test]
    fn drain_set_turns_the_band_over_and_scales_with_the_rate() {
        let n = 16usize;
        let mut turnovers: Vec<u64> = Vec::new();
        for sides in [2usize, 4] {
            let mut prod = swap_test_config();
            prod.swap_refresh = sides;
            prod.band = 48;
            let band_len = prod.band_size(n);
            let mut rng = StdRng::seed_from_u64(0xd7a1_0000 + sides as u64);
            let source = swap_test_source(n as u16, &mut rng);
            let state = GadgetState {
                n,
                pairs: (0..n).map(|w| (w, w)).collect(),
            };
            let mut ledger = ProdLedger::new(n, &prod, n, None);
            let mut out: Vec<XGate> = Vec::new();
            ledger.inject_all(&state, &mut rng, &mut out);
            assert!(ledger.drain_cap > 0, "sides={sides} drain set is not running");
            let plan_before: Vec<Vec<usize>> = (0..n)
                .map(|v| {
                    let mut d: Vec<usize> =
                        ledger.slots[v].iter().map(|s| s.factors.len()).collect();
                    d.sort_unstable();
                    d
                })
                .collect();
            for gate in &source {
                ledger.fold_cg(gate, &state, &mut rng, &mut out);
            }
            // Steering retires whatever degree it lands on, so the replacement
            // MUST be drawn at the retired slot's degree: the per-value degree
            // multiset is the mask plan, and the piling-up bound a build
            // commits to is read straight off it. Drift here would move the
            // security claim without moving anything that reports it.
            for value in 0..n {
                let mut after: Vec<usize> = ledger.slots[value]
                    .iter()
                    .map(|s| s.factors.len())
                    .collect();
                after.sort_unstable();
                assert_eq!(
                    after, plan_before[value],
                    "sides={sides} value {value} mask plan drifted"
                );
            }
            assert!(
                ledger.drained > 0,
                "sides={sides} no band variable ever came free: steering is stalled"
            );
            // The counts the rewrite guard reads must match the live slots. A
            // stale ZERO here is the failure that matters -- it would let a
            // referenced variable be overwritten -- so recount from scratch
            // rather than trusting the incremental path that produced them.
            let mut recount = vec![0u32; band_len];
            for value in 0..n {
                for slot in &ledger.slots[value] {
                    for &(b, _) in &slot.factors {
                        recount[b as usize] += 1;
                    }
                }
            }
            assert_eq!(
                ledger.var_refs, recount,
                "sides={sides} var_refs drifted from the live slots"
            );
            ledger.strip_all(&state, &mut rng, &mut out);
            assert!(
                ledger.var_refs.iter().all(|&r| r == 0),
                "sides={sides} strip_all left live references behind"
            );
            turnovers.push(ledger.drained);
        }
        assert!(
            turnovers[1] > turnovers[0],
            "retirement rate is the turnover lever, but 4 sides bought {} against 2 sides' {}",
            turnovers[1],
            turnovers[0]
        );
    }

    /// The 2026-08-20 swap-refresh redesign preserves the function exactly:
    /// on the zero band slice the single-carrier gadget still computes the
    /// source, while every fold retires and refreshes one mask term on the
    /// target and on one control.
    #[test]
    fn swap_refresh_single_carrier_matches_source_on_the_zero_slice() {
        let n = 8usize;
        for seed in 0..4u64 {
            let mut rng = StdRng::seed_from_u64(0x5a70_0000 + seed);
            let source = swap_test_source(n as u16, &mut rng);
            // close_slice off: the builder strips every value and the whole
            // data range must match the source exactly.
            let mut prod = swap_test_config();
            prod.close_slice = 0;
            let gadget = gadgetize_xgates_single(&source, n, 1, &prod, &mut rng);
            assert!(gadget.num_wires <= 64, "test sized for u64 evaluation");
            let mask = (1u64 << n) - 1;
            for x in 0..=mask {
                let expected = eval_u64(&source, x) & mask;
                assert_eq!(
                    eval_u64(&gadget.gates, x) & mask,
                    expected,
                    "seed={seed} x={x:#x}"
                );
            }
        }
    }

    /// With close_slice on, the BUILDER (no wrapper guards) still matches
    /// the source on ALL data wires: every value's registry is discharged —
    /// an undischarged registry leaves the emission telescope open under
    /// reverse evaluation and corrupts the reverse payload (see the comment
    /// at the strip_all call site). The junk-half divergence of the
    /// delivered composite comes only from the wrapper's closing guard.
    #[test]
    fn swap_refresh_builder_matches_source_fully_even_with_close_slice() {
        let n = 8usize;
        let mut rng = StdRng::seed_from_u64(0x5a70_0100);
        let source = swap_test_source(n as u16, &mut rng);
        let prod = swap_test_config();
        let gadget = gadgetize_xgates_single(&source, n, 1, &prod, &mut rng);
        let mask = (1u64 << n) - 1;
        for x in 0..=mask {
            let expected = eval_u64(&source, x) & mask;
            let got = eval_u64(&gadget.gates, x) & mask;
            assert_eq!(got, expected, "x={x:#x}");
        }
    }

    /// The closing zero-slice block has the opening block's specification —
    /// identity exactly on the zero slice, every nonzero slice perturbs the
    /// data — with its targets confined to the low (forward-junk) half.
    #[test]
    fn slice_zero_postblock_fixes_only_zero_slice_and_targets_the_low_half() {
        let n = 6usize;
        let nondata = 4usize;
        let mask = (1u64 << n) - 1;
        for seed in 0..8u64 {
            let mut rng = StdRng::seed_from_u64(0xc105_0000 + seed);
            let block = slice_zero_junk_guard_dims(n, nondata, 3 * nondata, &mut rng);
            for g in &block.gates {
                assert!(
                    (g.target as usize) < n / 2,
                    "closing-block target {} outside the junk half",
                    g.target
                );
            }
            for x in 0..=mask {
                assert_eq!(
                    eval_u64(&block.gates, x) & mask,
                    x,
                    "not identity on the zero slice (seed={seed})"
                );
            }
            for s in 1..(1u64 << nondata) {
                let disturbed =
                    (0..=mask).any(|x| eval_u64(&block.gates, x | (s << n)) & mask != x);
                assert!(disturbed, "slice {s:#x} leaves the data fixed (seed={seed})");
            }
        }
    }

    /// Symmetric ports: with both guards junk-half-only, the REVERSED gadget
    /// on the reverse-honest slice (a on the low half, zero upper half, zero
    /// band) reproduces the REVERSED source's upper half — the gadget-level
    /// mirror of the sandwich's A^-1(a,0) = (junk, D^-1(a)) contract. Every
    /// XGate is an involution, so the reversed gate list IS the inverse.
    #[test]
    fn symmetric_guards_make_reverse_evaluation_honest() {
        let n = 8usize;
        for seed in 0..3u64 {
            let mut rng = StdRng::seed_from_u64(0x4e7e_0000 + seed);
            let source = swap_test_source(n as u16, &mut rng);
            let prod = swap_test_config();
            let slice_gates = 4 * prod.band_size(n);
            let circuit = gadgetize_xgates_with_slice_zero_ccnot_single(
                &source,
                n,
                1,
                slice_gates,
                &prod,
                &mut rng,
            );
            let rev_gadget: Vec<XGate> = circuit.gates.iter().rev().cloned().collect();
            let rev_source: Vec<XGate> = source.iter().rev().cloned().collect();
            let mask = (1u64 << n) - 1;
            let low = (1u64 << (n / 2)) - 1;
            let upper = mask & !low;
            for a in 0..=low {
                let expected = eval_u64(&rev_source, a) & upper;
                let got = eval_u64(&rev_gadget, a) & upper;
                assert_eq!(got, expected, "seed={seed} a={a:#x}");
            }
        }
    }

    /// The slice-zero wrapper with the closing block preserves the source on
    /// the UPPER half of the data wires (the sandwich payload) and fires the
    /// closing guard into the junk half on the honest forward run.
    #[test]
    fn closing_slice_wrapper_preserves_the_upper_half() {
        let n = 8usize;
        let mut rng = StdRng::seed_from_u64(0xc105_c105);
        let source = swap_test_source(n as u16, &mut rng);
        let prod = swap_test_config();
        // Several gates per slice wire: at one gate per wire the closing
        // block's halved target range pigeonholes same-target pure-CNOT
        // pairs, which cancel exactly on weight-2 slices and starve the
        // acceptance draw (production runs ~10 gates per slice wire).
        let slice_gates = 4 * prod.band_size(n);
        let circuit = gadgetize_xgates_with_slice_zero_ccnot_single(
            &source,
            n,
            1,
            slice_gates,
            &prod,
            &mut rng,
        );
        assert!(circuit.num_wires <= 64, "test sized for u64 evaluation");
        let mask = (1u64 << n) - 1;
        let mut low_half_diverged = false;
        for x in 0..=mask {
            let expected = eval_u64(&source, x) & mask;
            let got = eval_u64(&circuit.gates, x) & mask;
            let upper = !((1u64 << (n / 2)) - 1) & mask;
            assert_eq!(got & upper, expected & upper, "payload half x={x:#x}");
            if got & !upper & mask != expected & !upper & mask {
                low_half_diverged = true;
            }
        }
        assert!(
            low_half_diverged,
            "closing guard never fired: the low half matches the source everywhere, \
             so the appended block is not doing its job"
        );
    }

    #[test]
    fn slice_block_stops_the_inverse_from_revealing_c_inverse() {
        let n = 3;
        let mask = (1u64 << n) - 1;
        let main = CircuitSeq {
            gates: vec![[0, 1, 2], [2, 0, 1], [1, 2, 0]],
        };
        for seed in 0..8u64 {
            let mut rng = StdRng::seed_from_u64(0xcc30_0000 + seed);
            let transformed = gadgetize_with_slice_zero_ccnot(
                &main,
                n,
                2,
                18,
                &MaskConfig::off(),
                &ProdConfig::off(),
                &mut rng,
            );
            // Every XGate is an involution, so the reversed gate list is the
            // inverse circuit; the slice block runs LAST there and fires on
            // the gadget's mask residue, junking the low half. Without it a
            // bare gadget's inverse returns C^-1 on the low wires for ANY
            // junk input.
            let reversed: Vec<XGate> = transformed.gates.iter().rev().cloned().collect();
            let mut leaks = 0usize;
            for p in 0..=mask {
                let c_inv = (0..=mask)
                    .find(|&x| main.evaluate(x as usize) as u64 & mask == p)
                    .unwrap();
                if eval_u64(&reversed, p) & mask == c_inv {
                    leaks += 1;
                }
            }
            assert!(
                leaks < (mask as usize + 1),
                "inverse hands out C^-1 verbatim (seed={seed:#x})"
            );
        }
    }

    #[test]
    fn gadgetize_xgates_preserves_the_low_wires_for_heterogeneous_sources() {
        let n = 3;
        let mask = (1u64 << n) - 1;
        // A heterogeneous mpmct1 source: g57, CNOT, CCNOT, and a negated
        // fragment — everything emit_shared_xgate2 must handle.
        let source = vec![
            XGate::from_g57([0, 1, 2]),
            XGate::cnot(0, 1),
            XGate::conj(2, [(0u16, true), (1u16, true)]).unwrap(),
            XGate::conj(1, [(2u16, false)]).unwrap(),
        ];
        for seed in 0..8u64 {
            let mut rng = StdRng::seed_from_u64(0xa11d_0000 + seed);
            let g = gadgetize_xgates(
                &source,
                n,
                2,
                &MaskConfig::off(),
                &ProdConfig::off(),
                &mut rng,
            );
            assert_eq!(g.num_wires, 2 * n);
            // Low n output = source(low n input) for ANY aux value.
            for input in 0..(1u64 << (2 * n)) {
                let expected = eval_u64(&source, input & mask) & mask;
                assert_eq!(
                    eval_u64(&g.gates, input) & mask,
                    expected,
                    "input={input:#x}"
                );
            }
        }
    }

    #[test]
    fn commuting_shuffle_preserves_function_and_relocates_gates() {
        let n = 6usize;
        for seed in 0..8u64 {
            let mut rng = StdRng::seed_from_u64(0x5f1e_0000 + seed);
            let mut gates: Vec<XGate> = Vec::new();
            for _ in 0..200 {
                let a = rng.random_range(0..n as u16);
                let b = loop {
                    let w = rng.random_range(0..n as u16);
                    if w != a {
                        break w;
                    }
                };
                let c = loop {
                    let w = rng.random_range(0..n as u16);
                    if w != a && w != b {
                        break w;
                    }
                };
                gates.push(match rng.random_range(0..3u32) {
                    0 => XGate::cnot(a, b),
                    1 => XGate::conj(a, [(b, true), (c, rng.random_bool(0.5))]).unwrap(),
                    _ => XGate::from_g57([a, b, c]),
                });
            }
            let before = gates.clone();
            commuting_shuffle(&mut gates, &mut rng);
            // Same multiset of gates, same function on every input, new order.
            let mut counts = std::collections::HashMap::new();
            for g in &before {
                *counts.entry(g.clone()).or_insert(0i64) += 1;
            }
            for g in &gates {
                *counts.entry(g.clone()).or_insert(0i64) -= 1;
            }
            assert!(counts.values().all(|&c| c == 0), "gate multiset changed");
            for input in 0..(1u64 << n) {
                assert_eq!(
                    eval_u64(&gates, input),
                    eval_u64(&before, input),
                    "seed={seed:#x} input={input:#x}"
                );
            }
            assert_ne!(gates, before, "seed={seed:#x}: order untouched");
        }
    }

    #[test]
    fn cg_menu_variants_are_correct_and_stay_in_vocabulary() {
        // Values a,b,c live on carrier pairs (0,1), (2,3), (4,5). Every
        // variant, under every random role assignment, must (a) apply
        // exactly A ^= B OR !C to the target value, (b) leave every wire
        // outside a's carriers unchanged (collapses restored at wire
        // level), and (c) emit only g57s / 1-2-control conjunctions —
        // never a bare X, whose census would count the source gates.
        let state = GadgetState {
            n: 3,
            pairs: vec![(0, 1), (2, 3), (4, 5)],
        };
        let bit = |v: u64, w: usize| (v >> w) & 1;
        for variant in 0..CG_VARIANTS {
            for role_seed in 0..8u64 {
                let mut rng = StdRng::seed_from_u64(0xc6_0000 + role_seed);
                let mut gates = Vec::new();
                emit_cg_variant(&state, [0, 1, 2], variant, &mut rng, &mut gates);
                for g in &gates {
                    if g.comp {
                        assert_eq!(g.ctrls.len(), 2, "complemented gate must be a g57");
                    } else {
                        assert!(
                            (1..=2).contains(&g.ctrls.len()),
                            "conjunction must have 1 or 2 controls (no bare X)"
                        );
                    }
                }
                for input in 0..64u64 {
                    let output = eval_u64(&gates, input);
                    assert_eq!(
                        output & !0b11,
                        input & !0b11,
                        "variant {variant}: non-target wires disturbed"
                    );
                    let b_val = bit(input, 2) ^ bit(input, 3);
                    let c_val = bit(input, 4) ^ bit(input, 5);
                    let f = b_val | (1 ^ c_val);
                    let a_old = bit(input, 0) ^ bit(input, 1);
                    let a_new = bit(output, 0) ^ bit(output, 1);
                    assert_eq!(
                        a_new,
                        a_old ^ f,
                        "variant {variant} input {input:#08b}: wrong update"
                    );
                }
            }
        }
    }

    #[test]
    fn commuting_shuffle_reorders_across_an_opposite_polarity_crossing() {
        // A writes wire 0, B reads wire 0 — a read/write crossing — but they
        // share control wire 3 with opposite polarities, so their firing
        // supports are disjoint and they commute (two conjunction gates
        // sharing an opposite-polarity control). The shuffle must treat the
        // pair as mobile, not pin it by the crossing alone.
        let a = XGate::conj(0, [(2u16, true), (3u16, true)]).unwrap();
        let b = XGate::conj(1, [(0u16, true), (3u16, false)]).unwrap();
        assert!(!XGate::collides(&a, &b));
        let before = vec![a.clone(), b.clone()];
        let mut seen_swapped = false;
        for seed in 0..32u64 {
            let mut rng = StdRng::seed_from_u64(0xccc0_0000 + seed);
            let mut gates = before.clone();
            commuting_shuffle(&mut gates, &mut rng);
            for input in 0..(1u64 << 4) {
                assert_eq!(eval_u64(&gates, input), eval_u64(&before, input));
            }
            if gates[0] == b {
                seen_swapped = true;
            }
        }
        assert!(seen_swapped, "the separation-exempt pair never reordered");
    }

    #[test]
    fn gadget_body_carries_nonlinear_rg_material() {
        // Complemented (comp=1) gates come only from the Z bookends and the
        // reinstated nonlinear g57 RG networks; the preblock, W_i, SG
        // fragments, and any linear RG are pure conjunctions. So a count
        // well above the two bookends certifies the RG policy is nonlinear.
        let n = 6;
        let main = CircuitSeq {
            gates: (0..40)
                .map(|k| [(k % n) as u16, ((k + 1) % n) as u16, ((k + 2) % n) as u16])
                .collect(),
        };
        let bookend_size = (2 * n * (n as f64).ln() as usize).max(64);
        let mut rng = StdRng::seed_from_u64(0xda7a_0001);
        let g = gadgetize_cnot(
            &main,
            n,
            1,
            &MaskConfig::off(),
            &ProdConfig::off(),
            &mut rng,
        );
        let comp_gates = g.gates.iter().filter(|g| g.comp).count();
        assert!(
            comp_gates > 2 * bookend_size,
            "expected nonlinear RG g57s beyond the {} bookend gates, found {} comp gates",
            2 * bookend_size,
            comp_gates
        );
    }

    #[test]
    fn compose_a_realizes_the_reference_map() {
        let n = 3;
        let mask = 1usize << n;
        let c = CircuitSeq {
            gates: vec![[0, 1, 2], [2, 0, 1], [1, 2, 0]],
        };
        let d = CircuitSeq {
            gates: vec![[1, 0, 2], [0, 2, 1]],
        };
        let a = compose_a(&c, &d, n);
        assert!(a.gates.iter().flatten().all(|&w| (w as usize) < 2 * n));
        for x in 0..mask {
            for z in 0..mask {
                let input = x | (z << n);
                let out = a.evaluate(input);
                let cx = c.evaluate(x);
                let expected_lo = d.evaluate(cx); // D(C(x))
                let expected_hi = z ^ cx; // z ^ C(x)
                assert_eq!(out & (mask - 1), expected_lo, "low x={x} z={z}");
                assert_eq!((out >> n) & (mask - 1), expected_hi, "high x={x} z={z}");
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
            for g in &block {
                assert!(!g.comp);
                assert!((g.target as usize) < n, "targets in the first half");
                assert!(g.ctrls.iter().all(|&(_, p)| p), "positive controls");
                assert!(
                    g.ctrls.iter().any(|&(w, _)| (w as usize) >= n),
                    "every gate reads a second-half wire"
                );
            }
            // Second half zero => identity on any first-half value.
            for x in 0..=mask {
                assert_eq!(eval_u64(&block, x), x, "dead on the zero slice");
            }
        }
    }

    #[test]
    fn sliced_sandwich_computes_c_on_the_second_half_on_the_zero_slice() {
        let n = 3;
        let mask = (1u64 << n) - 1;
        let full = (1u64 << (2 * n)) - 1;
        let main = CircuitSeq {
            gates: vec![[0, 1, 2], [2, 0, 1], [1, 2, 0], [0, 2, 1]],
        };
        for seed in 0..8u64 {
            let mut rng = StdRng::seed_from_u64(0x5a4d_0000 + seed);
            let a = sliced_sandwich_cnot(&main, n, 12, 4 * n, &mut rng);
            assert_eq!(a.num_wires, 2 * n);

            // Zero slice: the second half carries C(x).
            for x in 0..=mask {
                let expected = main.evaluate(x as usize) as u64 & mask;
                assert_eq!((eval_u64(&a.gates, x) >> n) & mask, expected, "A(x,0)");
            }
            // A is a permutation of the whole 2n-bit space.
            let mut seen = std::collections::HashSet::new();
            for input in 0..=full {
                assert!(seen.insert(eval_u64(&a.gates, input)), "A not injective");
            }
            // Off-slice, the second-half output is y-masked and differs from
            // C(x) for at least some inputs on some nonzero slice.
            let differs = (1..=mask).any(|y| {
                (0..=mask).any(|x| {
                    let input = x | (y << n);
                    let expected = main.evaluate(x as usize) as u64 & mask;
                    (eval_u64(&a.gates, input) >> n) & mask != expected
                })
            });
            assert!(differs, "seed={seed:#x}: no off-slice disturbance");
        }
    }

    #[test]
    fn sliced_sandwich_floats_the_middle_column_into_a_band() {
        // After the float stage the N CNOTs (the only gates targeting the
        // second half) must no longer sit as one contiguous column: their
        // positions should straddle other material on both sides for at
        // least some gates, under every seed.
        let n = 6;
        let main = CircuitSeq {
            gates: (0..24)
                .map(|k| [(k % n) as u16, ((k + 1) % n) as u16, ((k + 2) % n) as u16])
                .collect(),
        };
        for seed in 0..8u64 {
            let mut rng = StdRng::seed_from_u64(0x5a6d_0000 + seed);
            let a = sliced_sandwich_cnot(&main, n, 20, 4 * n, &mut rng);
            let positions: Vec<usize> = (0..a.gates.len())
                .filter(|&i| (a.gates[i].target as usize) >= n)
                .collect();
            assert_eq!(positions.len(), n, "exactly the n column CNOTs");
            let span = positions.last().unwrap() - positions.first().unwrap();
            assert!(
                span > n,
                "seed={seed:#x}: column still contiguous (span {span})"
            );
        }
    }

    #[test]
    fn sliced_sandwich_inverse_is_dead_slice_and_reveals_d_inverse() {
        let n = 3;
        let mask = (1u64 << n) - 1;
        let main = CircuitSeq {
            gates: vec![[0, 1, 2], [2, 0, 1]],
        };
        for seed in 0..8u64 {
            let mut rng = StdRng::seed_from_u64(0x5a5d_0000 + seed);
            let a = sliced_sandwich_cnot(&main, n, 10, 4 * n, &mut rng);
            let inverse: Vec<XGate> = a.gates.iter().rev().cloned().collect();
            // The inverse computes some permutation on the second half on the
            // zero slice (D^-1 up to the dead S2); check it is a bijection x
            // -> second-half output, i.e. the slice really carries a function.
            let mut outs = std::collections::HashSet::new();
            for p in 0..=mask {
                outs.insert((eval_u64(&inverse, p) >> n) & mask);
            }
            assert_eq!(
                outs.len() as u64,
                mask + 1,
                "inverse second-half map is a bijection on the zero slice"
            );
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
            // Matched RG rate: one nonlinear RG per SG on both paths (the
            // cnot path now draws the same {RG1,RG2,RG3} g57 networks, so the
            // lean margin comes from the 4-fragment SG and 7-CNOT W_i alone).
            let legacy_gadget = gadgetize(&main, n, 1, &mut legacy_rng).gates.len();
            let cnot_gadget = gadgetize_cnot(
                &main,
                n,
                1,
                &MaskConfig::off(),
                &ProdConfig::off(),
                &mut cnot_rng,
            )
            .gates
            .len();
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

    // Keep a test-side copy: comparing the circuit only to the implementation
    // constant would let both drift together without pinning the supplied map.
    const EXPECTED_FIVE_CARRIER_U0: [u8; 32] = [
        19, 27, 8, 28, 0, 25, 12, 22, 24, 18, 1, 30, 5, 7, 3, 11, 29, 21, 23, 15, 26, 6, 4, 31, 16,
        14, 13, 10, 9, 20, 2, 17,
    ];

    #[test]
    fn five_carrier_u0_realization_has_the_supplied_truth_table() {
        assert_eq!(
            FIVE_CARRIER_U0, EXPECTED_FIVE_CARRIER_U0,
            "the frozen U0 table drifted from the supplied permutation"
        );
        let carriers = [0usize, 1, 2, 3, 4];
        let mut gates = Vec::new();
        emit_five_carrier_update(&carriers, &mut gates);
        assert_eq!(gates.len(), 40, "the frozen U0 realization drifted");

        let mut seen_u0 = [false; 32];
        let mut seen_u1 = [false; 32];
        let mut classes = [0usize; 2];
        for input in 0u8..32 {
            let output = eval_u64(&gates, input as u64) as u8;
            assert_eq!(
                output, EXPECTED_FIVE_CARRIER_U0[input as usize],
                "U0 truth-table mismatch at {input:#07b}"
            );
            assert!(!seen_u0[output as usize], "U0 is not injective");
            seen_u0[output as usize] = true;
            assert_ne!(output, input, "U0 has a fixed point at {input:#07b}");
            assert_eq!(
                five_carrier_decode_word(output),
                five_carrier_decode_word(input),
                "U0 changed the decode class at {input:#07b}"
            );

            // U1 is U0 followed by the designated c0 flip.  It is another
            // fixed-point-free permutation and changes exactly the D class.
            let output_u1 = output ^ 1;
            assert!(!seen_u1[output_u1 as usize], "U1 is not injective");
            seen_u1[output_u1 as usize] = true;
            assert_ne!(output_u1, input, "U1 has a fixed point at {input:#07b}");
            assert_ne!(
                five_carrier_decode_word(output_u1),
                five_carrier_decode_word(input),
                "U1 did not change the decode class at {input:#07b}"
            );
            classes[five_carrier_decode_word(input) as usize] += 1;
        }
        assert!(seen_u0.into_iter().all(|seen| seen));
        assert!(seen_u1.into_iter().all(|seen| seen));
        assert_eq!(classes, [16, 16], "D must split the carrier space evenly");
        assert!(
            gates.iter().any(|gate| gate.width() == 4),
            "an ancilla-free realization of the odd U0 permutation needs a width-4 gate"
        );
    }

    #[test]
    fn five_carrier_update_trace_has_no_weight_one_or_two_walsh_detector() {
        // A trace row consists of the five carrier bits immediately before and
        // after one update.  For firing bit f, the post-state is
        // U_f(x) = U0(x) XOR f*e0.  Its Walsh coefficient against a detector
        // is sum_{x,f} (-1)^(f XOR parity(detector & (x,U_f(x)))).
        // Thus a zero sum means exactly zero correlation with f.
        let walsh_sum = |detector: u16| -> i32 {
            let mut sum = 0i32;
            for input in 0u16..32 {
                for firing in 0u16..2 {
                    let output = EXPECTED_FIVE_CARRIER_U0[input as usize] as u16 ^ firing;
                    let trace = input | (output << 5);
                    let prediction = (trace & detector).count_ones() as u16 & 1;
                    sum += if prediction == firing { 1 } else { -1 };
                }
            }
            sum
        };

        for detector in 1u16..(1u16 << 10) {
            let weight = detector.count_ones();
            if weight <= 2 {
                assert_eq!(
                    walsh_sum(detector),
                    0,
                    "weight-{weight} trace detector {detector:#012b} is correlated"
                );
            }
        }

        let max_weight_three = (1u16..(1u16 << 10))
            .filter(|detector| detector.count_ones() == 3)
            .map(|detector| walsh_sum(detector).unsigned_abs())
            .max()
            .unwrap();
        assert_eq!(
            max_weight_three, 32,
            "the first nonzero trace spectrum should have Walsh magnitude 1/2"
        );
        // There are 64 equally weighted (x,f) rows.  A Walsh magnitude of 32
        // is |Pr[predict=f] - 1/2| = 32/(2*64) = 1/4.
    }

    #[test]
    fn five_carrier_endpoint_firing_bit_is_exactly_degree_two() {
        // Pin the algebraic boundary separately from the raw-parity spectrum.
        // The supplied decode is quadratic, so every transition satisfies
        //
        //   firing = D(carrier_before) XOR D(carrier_after).
        //
        // Zero correlation with all weight-one/two XOR detectors does not
        // imply immunity to Gaussian elimination on the second tensor: the
        // right-hand side below is an XOR of nine degree-one/two features.
        fn gf2_rank(signatures: impl IntoIterator<Item = u64>) -> usize {
            let mut basis = [0u64; 64];
            let mut rank = 0usize;
            for mut signature in signatures {
                while signature != 0 {
                    let pivot = 63 - signature.leading_zeros() as usize;
                    if basis[pivot] != 0 {
                        signature ^= basis[pivot];
                    } else {
                        basis[pivot] = signature;
                        rank += 1;
                        break;
                    }
                }
            }
            rank
        }

        let mut columns = [0u64; 10];
        let mut firing_signature = 0u64;
        let mut decode_delta_signature = 0u64;
        for input in 0usize..32 {
            for firing in 0usize..2 {
                let row = 2 * input + firing;
                let output = EXPECTED_FIVE_CARRIER_U0[input] as usize ^ firing;
                let trace = input | (output << 5);
                for (wire, column) in columns.iter_mut().enumerate() {
                    if trace & (1usize << wire) != 0 {
                        *column |= 1u64 << row;
                    }
                }
                if firing != 0 {
                    firing_signature |= 1u64 << row;
                }
                if five_carrier_decode_word(input as u8) ^ five_carrier_decode_word(output as u8) {
                    decode_delta_signature |= 1u64 << row;
                }
            }
        }
        assert_eq!(
            decode_delta_signature, firing_signature,
            "the supplied quadratic decode must recover every firing bit exactly"
        );

        let degree_one: Vec<u64> = std::iter::once(u64::MAX).chain(columns).collect();
        assert_eq!(gf2_rank(degree_one.iter().copied()), 11);
        assert_eq!(
            gf2_rank(
                degree_one
                    .iter()
                    .copied()
                    .chain(std::iter::once(firing_signature))
            ),
            12,
            "degree-one endpoint features unexpectedly recovered the firing bit"
        );

        let mut degree_two = degree_one;
        for left in 0..10 {
            for right in left + 1..10 {
                degree_two.push(columns[left] & columns[right]);
            }
        }
        assert_eq!(gf2_rank(degree_two.iter().copied()), 42);
        assert_eq!(
            gf2_rank(
                degree_two
                    .iter()
                    .copied()
                    .chain(std::iter::once(firing_signature))
            ),
            42,
            "the exact recovery boundary must be degree two"
        );
    }

    #[test]
    fn five_carrier_endpoint_has_no_perfect_xor_detector_at_any_weight() {
        let walsh_sum = |detector: u16| -> i32 {
            let mut sum = 0i32;
            for input in 0u16..32 {
                for firing in 0u16..2 {
                    let output = EXPECTED_FIVE_CARRIER_U0[input as usize] as u16 ^ firing;
                    let trace = input | (output << 5);
                    let prediction = (trace & detector).count_ones() as u16 & 1;
                    sum += if prediction == firing { 1 } else { -1 };
                }
            }
            sum
        };

        let maximum = (1u16..(1u16 << 10))
            .map(|detector| walsh_sum(detector).unsigned_abs())
            .max();
        assert_eq!(
            maximum,
            Some(32),
            "the supplied map should have no perfect affine endpoint relation"
        );
    }

    #[test]
    fn strong_five_carrier_u0_decode_and_degree_boundary() {
        let carriers = [0usize, 1, 2, 3, 4];
        let mut gates = Vec::new();
        emit_strong_five_carrier_update(&carriers, &mut gates);
        assert_eq!(gates.len(), 6, "strong-five U0 gate count drifted");

        let mut seen_u0 = [false; 32];
        let mut seen_u1 = [false; 32];
        for input in 0u8..32 {
            let output = eval_u64(&gates, input as u64) as u8;
            assert_eq!(output, STRONG_FIVE_CARRIER_U0[input as usize]);
            assert!(!seen_u0[output as usize]);
            seen_u0[output as usize] = true;
            assert_ne!(output, input, "strong-five U0 has a fixed point");
            assert_eq!(
                strong_five_carrier_decode_word(output),
                strong_five_carrier_decode_word(input),
                "strong-five U0 changed decode class"
            );

            let output_u1 = output ^ 1;
            assert!(!seen_u1[output_u1 as usize]);
            seen_u1[output_u1 as usize] = true;
            assert_ne!(output_u1, input, "strong-five U1 has a fixed point");
            assert_ne!(
                strong_five_carrier_decode_word(output_u1),
                strong_five_carrier_decode_word(input),
                "strong-five U1 did not flip decode class"
            );
        }
        assert!(seen_u0.into_iter().all(|seen| seen));
        assert!(seen_u1.into_iter().all(|seen| seen));

        fn gf2_rank(signatures: impl IntoIterator<Item = u64>) -> usize {
            let mut basis = [0u64; 64];
            let mut rank = 0usize;
            for mut signature in signatures {
                while signature != 0 {
                    let pivot = 63 - signature.leading_zeros() as usize;
                    if basis[pivot] != 0 {
                        signature ^= basis[pivot];
                    } else {
                        basis[pivot] = signature;
                        rank += 1;
                        break;
                    }
                }
            }
            rank
        }

        let mut columns = [0u64; 10];
        let mut firing_signature = 0u64;
        let mut decode_delta = 0u64;
        for input in 0usize..32 {
            for firing in 0usize..2 {
                let row = 2 * input + firing;
                let output = STRONG_FIVE_CARRIER_U0[input] as usize ^ firing;
                let trace = input | (output << 5);
                for (wire, column) in columns.iter_mut().enumerate() {
                    if trace & (1usize << wire) != 0 {
                        *column |= 1u64 << row;
                    }
                }
                if firing != 0 {
                    firing_signature |= 1u64 << row;
                }
                if strong_five_carrier_decode_word(input as u8)
                    ^ strong_five_carrier_decode_word(output as u8)
                {
                    decode_delta |= 1u64 << row;
                }
            }
        }
        assert_eq!(decode_delta, firing_signature);

        let degree_one: Vec<u64> = std::iter::once(u64::MAX).chain(columns).collect();
        let mut degree_two = degree_one.clone();
        for a in 0..10 {
            for b in a + 1..10 {
                degree_two.push(columns[a] & columns[b]);
            }
        }
        let rank_two = gf2_rank(degree_two.iter().copied());
        assert_eq!(rank_two, 22);
        assert_eq!(
            gf2_rank(
                degree_two
                    .iter()
                    .copied()
                    .chain(std::iter::once(firing_signature))
            ),
            rank_two + 1,
            "degree-two endpoint features recovered strong-five firing"
        );

        let mut degree_three = degree_two;
        for a in 0..10 {
            for b in a + 1..10 {
                for c in b + 1..10 {
                    degree_three.push(columns[a] & columns[b] & columns[c]);
                }
            }
        }
        assert_eq!(gf2_rank(degree_three.iter().copied()), 42);
        assert_eq!(
            gf2_rank(
                degree_three
                    .iter()
                    .copied()
                    .chain(std::iter::once(firing_signature))
            ),
            42,
            "strong-five exact boundary must first appear at degree three"
        );
    }

    #[test]
    fn strong_five_carrier_endpoint_walsh_and_affine_structure() {
        let walsh_sum = |detector: u16| -> i32 {
            let mut sum = 0i32;
            for input in 0u16..32 {
                for firing in 0u16..2 {
                    let output = STRONG_FIVE_CARRIER_U0[input as usize] as u16 ^ firing;
                    let trace = input | (output << 5);
                    let prediction = (trace & detector).count_ones() as u16 & 1;
                    sum += if prediction == firing { 1 } else { -1 };
                }
            }
            sum
        };

        for detector in 1u16..(1u16 << 10) {
            if detector.count_ones() <= 2 {
                assert_eq!(walsh_sum(detector), 0);
            }
            assert_ne!(walsh_sum(detector).unsigned_abs(), 64);
        }
        let weight_three: Vec<u32> = (1u16..(1u16 << 10))
            .filter(|detector| detector.count_ones() == 3)
            .map(|detector| walsh_sum(detector).unsigned_abs())
            .collect();
        assert_eq!(weight_three.iter().copied().max(), Some(16));
        assert_eq!(
            weight_three
                .iter()
                .filter(|&&magnitude| magnitude != 0)
                .count(),
            2
        );

        // The compact affine tail deliberately leaves c1 and c2 fixed.  Pin
        // that structural tradeoff so it cannot be omitted from the mode's
        // documentation or accidentally mistaken for full affine rank.
        for input in 0u8..32 {
            let output = STRONG_FIVE_CARRIER_U0[input as usize];
            assert_eq!((input >> 1) & 1, (output >> 1) & 1);
            assert_eq!((input >> 2) & 1, (output >> 2) & 1);
        }
    }

    #[test]
    fn strong_five_carrier_all_fold_modes_preserve_dirty_high_inputs() {
        let n = 6usize;
        let band = 6usize;
        let source = vec![
            XGate::from_g57([0, 1, 2]),
            XGate::cnot(3, 4),
            XGate::conj(5, [(0u16, true), (2u16, false)]).unwrap(),
        ];
        let low_mask = (1u64 << n) - 1;
        let total = 5 * n + band;
        let high_mask = ((1u64 << total) - 1) ^ low_mask;
        let junk_patterns = [
            0,
            high_mask,
            0xaaaa_aaaa_aaaa_aaaau64 & high_mask,
            0x36db_6db6_db6d_b6dbu64 & high_mask,
        ];

        for gray_fold in 0..=3usize {
            let mut prod = ProdConfig::production_five_carrier();
            prod.band = band;
            prod.gray_fold = gray_fold;
            prod.cg_jitter = 0;
            let mut rng = StdRng::seed_from_u64(0x5c0b_1c00 + gray_fold as u64);
            let gadget = gadgetize_xgates_strong_five_carrier(&source, n, 2, &prod, &mut rng);
            assert_eq!(gadget.num_wires, total);
            for low in 0..=low_mask {
                let expected = eval_u64(&source, low) & low_mask;
                for &junk in &junk_patterns {
                    assert_eq!(
                        eval_u64(&gadget.gates, low | junk) & low_mask,
                        expected,
                        "strong-five mode={gray_fold} low={low:#x} junk={junk:#x}"
                    );
                }
            }
        }
    }

    #[test]
    fn five_carrier_gadget_preserves_low_wires_for_dirty_high_inputs() {
        let n = 6usize;
        let band = 6usize;
        let mut prod = ProdConfig::production_five_carrier();
        prod.band = band;

        // Certify that this fixture really reaches the optimized four-phase
        // g57 fold, rather than silently exercising only the general odometer.
        let state = FiveCarrierState::home(n);
        let mut audit_rng = StdRng::seed_from_u64(0x5ca1_67a7);
        let mut ledger = ProdLedger::new(n, &prod, 5 * n, None);
        let mut injection = Vec::new();
        ledger.inject_all(&state.c0_view(), &mut audit_rng, &mut injection);
        let mut optimized_fold = Vec::new();
        ledger.fold_five(
            &XGate::from_g57([0, 1, 2]),
            &state,
            &mut audit_rng,
            &mut optimized_fold,
        );
        assert_eq!(ledger.cg_gray, 1, "fixture missed the optimized g57 path");

        // Keep the optimized g57 and add native heterogeneous forms handled by
        // fold_five's general path: CNOT plus a mixed-polarity CCNOT.
        let source = vec![
            XGate::from_g57([0, 1, 2]),
            XGate::cnot(3, 4),
            XGate::conj(5, [(0u16, true), (2u16, false)]).unwrap(),
        ];
        let low_mask = (1u64 << n) - 1;
        let total = 5 * n + band;
        let high_mask = ((1u64 << total) - 1) ^ low_mask;
        let fixed_junk = [
            0,
            high_mask,
            0xaaaa_aaaa_aaaa_aaaau64 & high_mask,
            0x5555_5555_5555_5555u64 & high_mask,
            0x9249_2492_4924_9249u64 & high_mask,
            0x36db_6db6_db6d_b6dbu64 & high_mask,
        ];

        for seed in 0..3u64 {
            let mut rng = StdRng::seed_from_u64(0x5ca1_0000 + seed);
            let gadget = gadgetize_xgates_five_carrier(&source, n, 2, &prod, &mut rng);
            assert_eq!(gadget.num_wires, total, "expected 5*n + band wires");

            for low in 0..=low_mask {
                let expected = eval_u64(&source, low) & low_mask;
                for (junk_index, &junk) in fixed_junk.iter().enumerate() {
                    let input = low | junk;
                    assert_eq!(
                        eval_u64(&gadget.gates, input) & low_mask,
                        expected,
                        "seed={seed} low={low:#x} junk pattern {junk_index}"
                    );
                }
                // One extra non-periodic high assignment per low word catches
                // accidental assumptions hidden by the structured patterns.
                let junk = (low
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    .rotate_left((low as u32) & 31))
                    & high_mask;
                assert_eq!(
                    eval_u64(&gadget.gates, low | junk) & low_mask,
                    expected,
                    "seed={seed} low={low:#x} hashed high junk"
                );
            }
        }
    }

    const EXPECTED_SIX_CARRIER_U0: [u8; 64] = [
        2, 3, 0, 1, 7, 6, 5, 4, 11, 10, 9, 8, 14, 15, 12, 13, 30, 31, 29, 28, 27, 26, 24, 25, 22,
        23, 21, 20, 19, 18, 16, 17, 42, 43, 41, 40, 46, 47, 45, 44, 35, 34, 32, 33, 39, 38, 36, 37,
        54, 55, 53, 52, 51, 50, 48, 49, 63, 62, 60, 61, 58, 59, 57, 56,
    ];

    #[test]
    fn six_carrier_u0_realization_has_the_supplied_truth_table() {
        assert_eq!(SIX_CARRIER_U0, EXPECTED_SIX_CARRIER_U0);
        let carriers = [0usize, 1, 2, 3, 4, 5];
        let mut gates = Vec::new();
        emit_six_carrier_update(&carriers, &mut gates);
        assert_eq!(gates.len(), 10, "the compact U0 realization drifted");

        let mut seen_u0 = [false; 64];
        let mut seen_u1 = [false; 64];
        let mut classes = [0usize; 2];
        for input in 0u8..64 {
            let output = eval_u64(&gates, input as u64) as u8;
            assert_eq!(
                output, EXPECTED_SIX_CARRIER_U0[input as usize],
                "U0 truth-table mismatch at {input:#08b}"
            );
            assert!(!seen_u0[output as usize], "U0 is not injective");
            seen_u0[output as usize] = true;
            assert_ne!(output, input, "U0 fixed point at {input:#08b}");
            assert_eq!(
                EXPECTED_SIX_CARRIER_U0[output as usize], input,
                "the supplied U0 must be an involution"
            );
            assert_eq!(
                six_carrier_decode_word(output),
                six_carrier_decode_word(input),
                "U0 changed the decode class at {input:#08b}"
            );

            let output_u1 = output ^ 1;
            assert!(!seen_u1[output_u1 as usize], "U1 is not injective");
            seen_u1[output_u1 as usize] = true;
            assert_ne!(output_u1, input, "U1 fixed point at {input:#08b}");
            assert_ne!(
                six_carrier_decode_word(output_u1),
                six_carrier_decode_word(input),
                "U1 did not change the decode class at {input:#08b}"
            );
            classes[six_carrier_decode_word(input) as usize] += 1;
        }
        assert!(seen_u0.into_iter().all(|seen| seen));
        assert!(seen_u1.into_iter().all(|seen| seen));
        assert_eq!(classes, [32, 32], "D must split the carrier space evenly");
        let widths: Vec<usize> = gates.iter().map(XGate::width).collect();
        assert_eq!(widths.iter().filter(|&&width| width == 0).count(), 1);
        assert_eq!(widths.iter().filter(|&&width| width == 1).count(), 5);
        assert_eq!(widths.iter().filter(|&&width| width == 2).count(), 1);
        assert_eq!(widths.iter().filter(|&&width| width == 3).count(), 3);
    }

    #[test]
    fn six_carrier_decode_has_no_weight_one_or_two_static_parity_correlation() {
        let correlation_sum = |detector: u8| -> i32 {
            (0u8..64)
                .map(|carrier| {
                    let detector_value = (carrier & detector).count_ones() & 1 != 0;
                    if detector_value == six_carrier_decode_word(carrier) {
                        1
                    } else {
                        -1
                    }
                })
                .sum()
        };
        for detector in 1u8..64 {
            if detector.count_ones() <= 2 {
                assert_eq!(
                    correlation_sum(detector),
                    0,
                    "low-weight static detector {detector:#08b} is correlated"
                );
            }
        }
        let max_weight_three = (1u8..64)
            .filter(|detector| detector.count_ones() == 3)
            .map(|detector| correlation_sum(detector).unsigned_abs())
            .max()
            .unwrap();
        assert_eq!(max_weight_three, 16, "expected normalized magnitude 1/4");
    }

    #[test]
    fn six_carrier_update_trace_has_no_weight_one_two_or_three_walsh_detector() {
        let walsh_sum = |detector: u16| -> i32 {
            let mut sum = 0i32;
            for input in 0u16..64 {
                for firing in 0u16..2 {
                    let output = EXPECTED_SIX_CARRIER_U0[input as usize] as u16 ^ firing;
                    let trace = input | (output << 6);
                    let prediction = (trace & detector).count_ones() as u16 & 1;
                    sum += if prediction == firing { 1 } else { -1 };
                }
            }
            sum
        };

        for detector in 1u16..(1u16 << 12) {
            let weight = detector.count_ones();
            if weight <= 3 {
                assert_eq!(
                    walsh_sum(detector),
                    0,
                    "weight-{weight} endpoint detector {detector:#014b} is correlated"
                );
            }
        }
        let max_weight_four = (1u16..(1u16 << 12))
            .filter(|detector| detector.count_ones() == 4)
            .map(|detector| walsh_sum(detector).unsigned_abs())
            .max()
            .unwrap();
        assert_eq!(
            max_weight_four, 32,
            "first normalized Walsh magnitude is 1/4"
        );
    }

    #[test]
    fn six_carrier_endpoint_firing_bit_is_outside_the_degree_two_trace_span() {
        fn gf2_rank(signatures: impl IntoIterator<Item = u128>) -> usize {
            let mut basis = [0u128; 128];
            let mut rank = 0usize;
            for mut signature in signatures {
                while signature != 0 {
                    let pivot = 127 - signature.leading_zeros() as usize;
                    if basis[pivot] != 0 {
                        signature ^= basis[pivot];
                    } else {
                        basis[pivot] = signature;
                        rank += 1;
                        break;
                    }
                }
            }
            rank
        }

        let mut columns = [0u128; 12];
        let mut firing_signature = 0u128;
        for input in 0usize..64 {
            for firing in 0usize..2 {
                let row = 2 * input + firing;
                let output = EXPECTED_SIX_CARRIER_U0[input] as usize ^ firing;
                let trace = input | (output << 6);
                for (wire, column) in columns.iter_mut().enumerate() {
                    if (trace >> wire) & 1 != 0 {
                        *column |= 1u128 << row;
                    }
                }
                if firing != 0 {
                    firing_signature |= 1u128 << row;
                }
            }
        }

        let mut degree_two = vec![u128::MAX];
        degree_two.extend(columns);
        for left in 0..12 {
            for right in left + 1..12 {
                degree_two.push(columns[left] & columns[right]);
            }
        }
        let rank_two = gf2_rank(degree_two.iter().copied());
        let rank_with_firing = gf2_rank(
            degree_two
                .iter()
                .copied()
                .chain(std::iter::once(firing_signature)),
        );
        assert_eq!(rank_two, 29, "degree-two endpoint feature rank drifted");
        assert_eq!(
            rank_with_firing,
            rank_two + 1,
            "degree-two Gaussian elimination recovered the firing bit"
        );

        // Pin the intended boundary: degree three does contain an exact
        // recovery, while degree two does not.
        let mut degree_three = degree_two;
        for a in 0..12 {
            for b in a + 1..12 {
                for c in b + 1..12 {
                    degree_three.push(columns[a] & columns[b] & columns[c]);
                }
            }
        }
        assert_eq!(
            gf2_rank(degree_three.iter().copied()),
            gf2_rank(
                degree_three
                    .iter()
                    .copied()
                    .chain(std::iter::once(firing_signature))
            ),
            "degree-three boundary unexpectedly moved"
        );
    }

    #[test]
    fn six_carrier_gadget_preserves_low_wires_for_dirty_high_inputs() {
        let n = 6usize;
        let band = 6usize;
        let mut prod = ProdConfig::production_six_carrier();
        prod.band = band;

        let state = SixCarrierState::home(n);
        let mut audit_rng = StdRng::seed_from_u64(0x6ca1_67a7);
        let mut ledger = ProdLedger::new(n, &prod, 6 * n, None);
        let mut injection = Vec::new();
        ledger.inject_all(&state.c0_view(), &mut audit_rng, &mut injection);
        let mut optimized_fold = Vec::new();
        ledger.fold_six(
            &XGate::from_g57([0, 1, 2]),
            &state,
            &mut audit_rng,
            &mut optimized_fold,
        );
        assert_eq!(
            ledger.cg_gray, 1,
            "fixture missed the six-carrier Gray fold"
        );

        let source = vec![
            XGate::from_g57([0, 1, 2]),
            XGate::cnot(3, 4),
            XGate::conj(5, [(0u16, true), (2u16, false)]).unwrap(),
        ];
        let low_mask = (1u64 << n) - 1;
        let total = 6 * n + band;
        let high_mask = ((1u64 << total) - 1) ^ low_mask;
        let fixed_junk = [
            0,
            high_mask,
            0xaaaa_aaaa_aaaa_aaaau64 & high_mask,
            0x5555_5555_5555_5555u64 & high_mask,
            0x9249_2492_4924_9249u64 & high_mask,
            0x36db_6db6_db6d_b6dbu64 & high_mask,
        ];

        for seed in 0..3u64 {
            let mut rng = StdRng::seed_from_u64(0x6ca1_0000 + seed);
            let gadget = gadgetize_xgates_six_carrier(&source, n, 2, &prod, &mut rng);
            assert_eq!(gadget.num_wires, total, "expected 6*n + band wires");

            for low in 0..=low_mask {
                let expected = eval_u64(&source, low) & low_mask;
                for (junk_index, &junk) in fixed_junk.iter().enumerate() {
                    assert_eq!(
                        eval_u64(&gadget.gates, low | junk) & low_mask,
                        expected,
                        "seed={seed} low={low:#x} junk pattern {junk_index}"
                    );
                }
                let junk = (low
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    .rotate_left((low as u32) & 31))
                    & high_mask;
                assert_eq!(
                    eval_u64(&gadget.gates, low | junk) & low_mask,
                    expected,
                    "seed={seed} low={low:#x} hashed high junk"
                );
            }
        }
    }

    #[test]
    fn six_carrier_public_cnot_and_slice_wrappers_have_the_expected_port() {
        let n = 6usize;
        let band = 6usize;
        let mut prod = ProdConfig::production_six_carrier();
        prod.band = band;
        let total = 6 * n + band;
        let low_mask = (1u64 << n) - 1;
        let high_mask = ((1u64 << total) - 1) ^ low_mask;
        let main = CircuitSeq {
            gates: vec![[0, 1, 2]],
        };
        let source = vec![XGate::from_g57([0, 1, 2])];

        let mut bare_rng = StdRng::seed_from_u64(0x6ca2_0001);
        let bare = gadgetize_cnot_six_carrier(&main, n, 1, &prod, &mut bare_rng);
        let mut slice_cnot_rng = StdRng::seed_from_u64(0x6ca2_0002);
        let slice_cnot = gadgetize_with_slice_zero_ccnot_six_carrier(
            &main,
            n,
            1,
            10 * n,
            &MaskConfig::off(),
            &prod,
            &mut slice_cnot_rng,
        );
        let mut slice_xgate_rng = StdRng::seed_from_u64(0x6ca2_0003);
        let slice_xgate = gadgetize_xgates_with_slice_zero_ccnot_six_carrier(
            &source,
            n,
            1,
            10 * n,
            &prod,
            &mut slice_xgate_rng,
        );
        assert_eq!(bare.num_wires, total);
        assert_eq!(slice_cnot.num_wires, total);
        assert_eq!(slice_xgate.num_wires, total);

        for low in 0..=low_mask {
            let expected = eval_u64(&source, low) & low_mask;
            assert_eq!(eval_u64(&bare.gates, low) & low_mask, expected);
            assert_eq!(eval_u64(&bare.gates, low | high_mask) & low_mask, expected);
            assert_eq!(eval_u64(&slice_cnot.gates, low) & low_mask, expected);
            assert_eq!(eval_u64(&slice_xgate.gates, low) & low_mask, expected);
        }
    }

    #[test]
    fn six_carrier_empty_source_is_the_identity_for_arbitrary_high_junk() {
        let n = 3usize;
        let mut prod = ProdConfig::production_six_carrier();
        prod.band = 6;
        let total = 6 * n + prod.band_size(n);
        let low_mask = (1u64 << n) - 1;
        let high_mask = ((1u64 << total) - 1) ^ low_mask;
        let mut rng = StdRng::seed_from_u64(0x6ca3_0001);
        let gadget = gadgetize_xgates_six_carrier(&[], n, 1, &prod, &mut rng);
        assert_eq!(gadget.num_wires, total);

        for low in 0..=low_mask {
            for junk in [
                0,
                high_mask,
                0xaaaa_aaaa_aaaa_aaaau64 & high_mask,
                0x9249_2492_4924_9249u64 & high_mask,
            ] {
                assert_eq!(eval_u64(&gadget.gates, low | junk) & low_mask, low);
            }
        }
    }

    #[test]
    fn strong_six_carrier_u0_truth_table_class_and_width_census() {
        let carriers = [0usize, 1, 2, 3, 4, 5];
        let mut gates = Vec::new();
        emit_strong_six_carrier_update(&carriers, &mut gates);
        assert_eq!(gates.len(), 21, "strong-six U0 gate count drifted");

        let mut seen_u0 = [false; 64];
        let mut seen_u1 = [false; 64];
        let mut classes = [0usize; 2];
        for input in 0u8..64 {
            let output = eval_u64(&gates, input as u64) as u8;
            assert_eq!(
                output, STRONG_SIX_CARRIER_U0[input as usize],
                "strong-six U0 truth-table mismatch at {input:#08b}"
            );
            assert!(!seen_u0[output as usize], "strong-six U0 is not injective");
            seen_u0[output as usize] = true;
            assert_ne!(output, input, "strong-six U0 fixed point at {input:#08b}");
            assert_eq!(
                six_carrier_decode_word(output),
                six_carrier_decode_word(input),
                "strong-six U0 changed the decode class at {input:#08b}"
            );

            let output_u1 = output ^ 1;
            assert!(
                !seen_u1[output_u1 as usize],
                "strong-six U1 is not injective"
            );
            seen_u1[output_u1 as usize] = true;
            assert_ne!(
                output_u1, input,
                "strong-six U1 fixed point at {input:#08b}"
            );
            assert_ne!(
                six_carrier_decode_word(output_u1),
                six_carrier_decode_word(input),
                "strong-six U1 did not change the decode class at {input:#08b}"
            );
            classes[six_carrier_decode_word(input) as usize] += 1;
        }
        assert!(seen_u0.into_iter().all(|seen| seen));
        assert!(seen_u1.into_iter().all(|seen| seen));
        assert_eq!(classes, [32, 32]);

        let widths: Vec<usize> = gates.iter().map(XGate::width).collect();
        for (width, expected) in [(0, 1), (1, 6), (2, 7), (3, 6), (4, 1)] {
            assert_eq!(
                widths.iter().filter(|&&actual| actual == width).count(),
                expected,
                "strong-six width-{width} gate count drifted"
            );
        }
    }

    #[test]
    fn strong_six_carrier_endpoint_has_full_affine_rank_and_pinned_walsh_spectrum() {
        fn gf2_rank(signatures: impl IntoIterator<Item = u64>) -> usize {
            let mut basis = [0u64; 64];
            let mut rank = 0usize;
            for mut signature in signatures {
                while signature != 0 {
                    let pivot = 63 - signature.leading_zeros() as usize;
                    if basis[pivot] != 0 {
                        signature ^= basis[pivot];
                    } else {
                        basis[pivot] = signature;
                        rank += 1;
                        break;
                    }
                }
            }
            rank
        }

        // Constant plus all six before and six U0-after coordinates have the
        // maximum possible rank.  Pin the movement census too: unlike the
        // compact sibling, no carrier lane is frozen or an affine duplicate.
        let mut graph_columns = [0u64; 12];
        let mut movement = [0usize; 6];
        for input in 0usize..64 {
            let output = STRONG_SIX_CARRIER_U0[input] as usize;
            let trace = input | (output << 6);
            for (wire, column) in graph_columns.iter_mut().enumerate() {
                if (trace >> wire) & 1 != 0 {
                    *column |= 1u64 << input;
                }
            }
            for (lane, count) in movement.iter_mut().enumerate() {
                *count += ((input ^ output) >> lane) & 1;
            }
        }
        assert_eq!(
            gf2_rank(std::iter::once(u64::MAX).chain(graph_columns)),
            13,
            "strong-six endpoint graph lost full affine rank"
        );
        assert_eq!(movement, [32, 48, 32, 32, 16, 16]);
        assert!(movement.into_iter().all(|count| count != 0));

        // Histogram bins are |W| = 16,32,48,64 over all 128 (input,firing)
        // rows.  No coefficient exists through detector weight three; the
        // first signal is rho=32/128=1/4 at weight four.  No parity detector
        // at any weight is perfect.
        let expected: [[usize; 4]; 13] = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [2, 15, 0, 0],
            [12, 53, 2, 2],
            [33, 66, 9, 7],
            [65, 71, 5, 5],
            [60, 52, 10, 2],
            [33, 33, 9, 1],
            [13, 10, 1, 1],
            [2, 3, 0, 0],
            [0, 1, 0, 0],
        ];
        let mut actual = [[0usize; 4]; 13];
        for detector in 1u16..(1u16 << 12) {
            let mut sum = 0i32;
            for input in 0u16..64 {
                for firing in 0u16..2 {
                    let output = STRONG_SIX_CARRIER_U0[input as usize] as u16 ^ firing;
                    let trace = input | (output << 6);
                    let prediction = (trace & detector).count_ones() as u16 & 1;
                    sum += if prediction == firing { 1 } else { -1 };
                }
            }
            let magnitude = sum.unsigned_abs() as usize;
            if magnitude != 0 {
                assert!(magnitude <= 64, "strong-six gained a stronger parity leak");
                assert_eq!(magnitude % 16, 0);
                actual[detector.count_ones() as usize][magnitude / 16 - 1] += 1;
            }
        }
        assert_eq!(
            actual, expected,
            "strong-six endpoint Walsh spectrum drifted"
        );
    }

    #[test]
    fn strong_six_carrier_exact_firing_boundary_is_degree_three() {
        fn gf2_rank(signatures: impl IntoIterator<Item = u128>) -> usize {
            let mut basis = [0u128; 128];
            let mut rank = 0usize;
            for mut signature in signatures {
                while signature != 0 {
                    let pivot = 127 - signature.leading_zeros() as usize;
                    if basis[pivot] != 0 {
                        signature ^= basis[pivot];
                    } else {
                        basis[pivot] = signature;
                        rank += 1;
                        break;
                    }
                }
            }
            rank
        }

        let mut columns = [0u128; 12];
        let mut firing_signature = 0u128;
        for input in 0usize..64 {
            for firing in 0usize..2 {
                let row = 2 * input + firing;
                let output = STRONG_SIX_CARRIER_U0[input] as usize ^ firing;
                let trace = input | (output << 6);
                for (wire, column) in columns.iter_mut().enumerate() {
                    if (trace >> wire) & 1 != 0 {
                        *column |= 1u128 << row;
                    }
                }
                if firing != 0 {
                    firing_signature |= 1u128 << row;
                }
            }
        }

        let mut features = vec![u128::MAX];
        features.extend(columns);
        assert_eq!(gf2_rank(features.iter().copied()), 13);
        assert_eq!(
            gf2_rank(
                features
                    .iter()
                    .copied()
                    .chain(std::iter::once(firing_signature))
            ),
            14
        );
        for a in 0..12 {
            for b in a + 1..12 {
                features.push(columns[a] & columns[b]);
            }
        }
        assert_eq!(gf2_rank(features.iter().copied()), 53);
        assert_eq!(
            gf2_rank(
                features
                    .iter()
                    .copied()
                    .chain(std::iter::once(firing_signature))
            ),
            54,
            "degree-two endpoint features recovered strong-six firing"
        );
        for a in 0..12 {
            for b in a + 1..12 {
                for c in b + 1..12 {
                    features.push(columns[a] & columns[b] & columns[c]);
                }
            }
        }
        assert_eq!(gf2_rank(features.iter().copied()), 103);
        assert_eq!(
            gf2_rank(
                features
                    .iter()
                    .copied()
                    .chain(std::iter::once(firing_signature))
            ),
            103,
            "strong-six exact firing boundary moved above degree three"
        );
    }

    #[test]
    fn strong_six_carrier_all_fold_modes_preserve_dirty_high_inputs() {
        let n = 6usize;
        let band = 6usize;
        let source = vec![
            XGate::from_g57([0, 1, 2]),
            XGate::cnot(3, 4),
            XGate::conj(5, [(0u16, true), (2u16, false)]).unwrap(),
        ];
        let low_mask = (1u64 << n) - 1;
        let total = 6 * n + band;
        let high_mask = ((1u64 << total) - 1) ^ low_mask;
        let junk_patterns = [
            0,
            high_mask,
            0xaaaa_aaaa_aaaa_aaaau64 & high_mask,
            0x36db_6db6_db6d_b6dbu64 & high_mask,
        ];

        // Exercise state-dispatched updates directly as well as through every
        // folding implementation: expanded, aggregate Gray, micro Gray, and
        // sentinel Gray.
        let state = SixCarrierState::strong_home(n);
        let mut update = Vec::new();
        state.emit_update(0, &mut update);
        assert_eq!(update.len(), STRONG_SIX_CARRIER_U0_GATES.len());

        for gray_fold in 0..=3usize {
            let mut prod = ProdConfig::production_six_carrier();
            prod.band = band;
            prod.gray_fold = gray_fold;
            prod.cg_jitter = 0;
            let mut rng = StdRng::seed_from_u64(0x6c0b_1c00 + gray_fold as u64);
            let gadget = gadgetize_xgates_strong_six_carrier(&source, n, 2, &prod, &mut rng);
            assert_eq!(gadget.num_wires, total);
            for low in 0..=low_mask {
                let expected = eval_u64(&source, low) & low_mask;
                for &junk in &junk_patterns {
                    assert_eq!(
                        eval_u64(&gadget.gates, low | junk) & low_mask,
                        expected,
                        "strong-six mode={gray_fold} low={low:#x} junk={junk:#x}"
                    );
                }
            }
        }
    }

    #[test]
    fn strong_six_carrier_public_wrappers_have_the_expected_port() {
        // The nonlinear slice preblock needs enough data coordinates to
        // disturb its 5*n+band nonzero slices; n=3 is intentionally rejected
        // by its exhaustive constructor for this width.
        let n = 6usize;
        let band = 6usize;
        let mut prod = ProdConfig::production_six_carrier();
        prod.band = band;
        let total = 6 * n + band;
        let low_mask = (1u64 << n) - 1;
        let main = CircuitSeq {
            gates: vec![[0, 1, 2]],
        };
        let source = vec![XGate::from_g57([0, 1, 2])];

        let mut bare_cnot_rng = StdRng::seed_from_u64(0x6c06_0001);
        let bare_cnot = gadgetize_cnot_strong_six_carrier(&main, n, 1, &prod, &mut bare_cnot_rng);
        let mut bare_xgate_rng = StdRng::seed_from_u64(0x6c06_0002);
        let bare_xgate =
            gadgetize_xgates_strong_six_carrier(&source, n, 1, &prod, &mut bare_xgate_rng);
        let mut slice_cnot_rng = StdRng::seed_from_u64(0x6c06_0003);
        let slice_cnot = gadgetize_with_slice_zero_ccnot_strong_six_carrier(
            &main,
            n,
            1,
            10 * n,
            &MaskConfig::off(),
            &prod,
            &mut slice_cnot_rng,
        );
        let mut slice_xgate_rng = StdRng::seed_from_u64(0x6c06_0004);
        let slice_xgate = gadgetize_xgates_with_slice_zero_ccnot_strong_six_carrier(
            &source,
            n,
            1,
            10 * n,
            &prod,
            &mut slice_xgate_rng,
        );
        for circuit in [&bare_cnot, &bare_xgate, &slice_cnot, &slice_xgate] {
            assert_eq!(circuit.num_wires, total);
            for low in 0..=low_mask {
                assert_eq!(
                    eval_u64(&circuit.gates, low) & low_mask,
                    eval_u64(&source, low) & low_mask
                );
            }
        }
    }

    #[test]
    fn seven_carrier_quartic_dirty_gather_restores_every_borrow() {
        // Wires: acc, h, A, B, C, D. Exercise both dirty inputs explicitly;
        // the adjusted Gray fold may not assume either scratch bit is clean.
        let atom = [(2u16, true), (3, true), (4, true), (5, true)];
        let mut rng = StdRng::seed_from_u64(0x7ca0_0001);
        let mut seen = std::collections::HashMap::new();
        let mut gates = Vec::new();
        assert!(!emit_atom_onto(
            0, &atom, 1, 0, 1, &mut seen, &mut rng, &mut gates,
        ));
        assert_eq!(gates.len(), 4);
        assert!(gates.iter().all(|gate| gate.width() <= 3));
        for input in 0u64..64 {
            let product =
                ((input >> 2) & 1) & ((input >> 3) & 1) & ((input >> 4) & 1) & ((input >> 5) & 1);
            let expected = input ^ product;
            assert_eq!(
                eval_u64(&gates, input),
                expected,
                "dirty quartic gather failed at {input:#08b}"
            );
        }
    }

    #[test]
    fn seven_carrier_u0_realization_has_the_selected_truth_table() {
        let carriers = [0usize, 1, 2, 3, 4, 5, 6];
        let mut gates = Vec::new();
        emit_seven_carrier_update(&carriers, &mut gates);
        assert_eq!(gates.len(), 26, "the selected nonlinear U0 drifted");

        let mut seen_u0 = [false; 128];
        let mut seen_u1 = [false; 128];
        let mut classes = [0usize; 2];
        for input in 0u8..128 {
            let output = eval_u64(&gates, input as u64) as u8;
            assert_eq!(
                output, SEVEN_CARRIER_U0[input as usize],
                "U0 truth-table mismatch at {input:#09b}"
            );
            assert!(!seen_u0[output as usize], "U0 is not injective");
            seen_u0[output as usize] = true;
            assert_ne!(output, input, "U0 fixed point at {input:#09b}");
            assert_eq!(
                seven_carrier_decode_word(output),
                seven_carrier_decode_word(input),
                "U0 changed the decode class at {input:#09b}"
            );

            let output_u1 = output ^ 1;
            assert!(!seen_u1[output_u1 as usize], "U1 is not injective");
            seen_u1[output_u1 as usize] = true;
            assert_ne!(output_u1, input, "U1 fixed point at {input:#09b}");
            assert_ne!(
                seven_carrier_decode_word(output_u1),
                seven_carrier_decode_word(input),
                "U1 did not change the decode class at {input:#09b}"
            );
            classes[seven_carrier_decode_word(input) as usize] += 1;
        }
        assert!(seen_u0.into_iter().all(|seen| seen));
        assert!(seen_u1.into_iter().all(|seen| seen));
        assert_eq!(classes, [64, 64], "D must split the carrier space evenly");

        let widths: Vec<usize> = gates.iter().map(XGate::width).collect();
        for (width, expected) in [(0, 2), (1, 7), (2, 9), (3, 4), (4, 4)] {
            assert_eq!(
                widths.iter().filter(|&&actual| actual == width).count(),
                expected,
                "width-{width} gate count drifted"
            );
        }
    }

    #[test]
    fn seven_carrier_decode_has_no_weight_one_or_two_static_parity_correlation() {
        let correlation_sum = |detector: u8| -> i32 {
            (0u8..128)
                .map(|carrier| {
                    let prediction = (carrier & detector).count_ones() & 1 != 0;
                    if prediction == seven_carrier_decode_word(carrier) {
                        1
                    } else {
                        -1
                    }
                })
                .sum()
        };
        for detector in 1u8..128 {
            if detector.count_ones() <= 2 {
                assert_eq!(
                    correlation_sum(detector),
                    0,
                    "low-weight static detector {detector:#09b} is correlated"
                );
            }
        }
        let max_weight_three = (1u8..128)
            .filter(|detector| detector.count_ones() == 3)
            .map(|detector| correlation_sum(detector).unsigned_abs())
            .max()
            .unwrap();
        assert_eq!(
            max_weight_three, 16,
            "first normalized static magnitude must be 1/8"
        );
    }

    #[test]
    fn seven_carrier_update_trace_has_zero_walsh_through_weight_three() {
        let walsh_sum = |detector: u16| -> i32 {
            let mut sum = 0i32;
            for input in 0u16..128 {
                for firing in 0u16..2 {
                    let output = SEVEN_CARRIER_U0[input as usize] as u16 ^ firing;
                    let trace = input | (output << 7);
                    let prediction = (trace & detector).count_ones() as u16 & 1;
                    sum += if prediction == firing { 1 } else { -1 };
                }
            }
            sum
        };

        for detector in 1u16..(1u16 << 14) {
            let weight = detector.count_ones();
            if weight <= 3 {
                assert_eq!(
                    walsh_sum(detector),
                    0,
                    "weight-{weight} endpoint detector {detector:#016b} is correlated"
                );
            }
        }
        let weight_four: Vec<u32> = (1u16..(1u16 << 14))
            .filter(|detector| detector.count_ones() == 4)
            .map(|detector| walsh_sum(detector).unsigned_abs())
            .collect();
        assert_eq!(weight_four.iter().copied().max(), Some(64));
        assert_eq!(
            weight_four
                .iter()
                .filter(|&&magnitude| magnitude == 64)
                .count(),
            3,
            "maximum-bias weight-four endpoint multiplicity drifted"
        );
        assert_eq!(
            weight_four
                .iter()
                .filter(|&&magnitude| magnitude != 0)
                .count(),
            21,
            "weight-four endpoint spectrum drifted"
        );
    }

    #[test]
    fn seven_carrier_endpoint_affine_and_degree_boundary() {
        type Signature = [u64; 4];

        fn and(left: Signature, right: Signature) -> Signature {
            std::array::from_fn(|word| left[word] & right[word])
        }

        fn gf2_rank(signatures: impl IntoIterator<Item = Signature>) -> usize {
            let mut basis = [[0u64; 4]; 256];
            let mut rank = 0usize;
            for mut signature in signatures {
                loop {
                    let Some(pivot) = (0..256)
                        .rev()
                        .find(|&bit| signature[bit / 64] & (1u64 << (bit % 64)) != 0)
                    else {
                        break;
                    };
                    if basis[pivot] == [0; 4] {
                        basis[pivot] = signature;
                        rank += 1;
                        break;
                    }
                    for word in 0..4 {
                        signature[word] ^= basis[pivot][word];
                    }
                }
            }
            rank
        }

        fn with_firing_rank(features: &[Signature], firing: Signature) -> usize {
            gf2_rank(features.iter().copied().chain(std::iter::once(firing)))
        }

        let mut columns = [[0u64; 4]; 14];
        let mut firing = [0u64; 4];
        for input in 0usize..128 {
            for fires in 0usize..2 {
                let row = 2 * input + fires;
                let output = SEVEN_CARRIER_U0[input] as usize ^ fires;
                let trace = input | (output << 7);
                for (wire, column) in columns.iter_mut().enumerate() {
                    if trace & (1usize << wire) != 0 {
                        column[row / 64] |= 1u64 << (row % 64);
                    }
                }
                if fires != 0 {
                    firing[row / 64] |= 1u64 << (row % 64);
                }
            }
        }

        let degree_one: Vec<Signature> = std::iter::once([u64::MAX; 4]).chain(columns).collect();
        assert_eq!(gf2_rank(degree_one.iter().copied()), 15);
        assert_eq!(with_firing_rank(&degree_one, firing), 16);

        let mut degree_two = degree_one.clone();
        for a in 0..14 {
            for b in a + 1..14 {
                degree_two.push(and(columns[a], columns[b]));
            }
        }
        assert_eq!(gf2_rank(degree_two.iter().copied()), 71);
        assert_eq!(with_firing_rank(&degree_two, firing), 72);

        let mut degree_three = degree_two.clone();
        for a in 0..14 {
            for b in a + 1..14 {
                for c in b + 1..14 {
                    degree_three.push(and(and(columns[a], columns[b]), columns[c]));
                }
            }
        }
        assert_eq!(gf2_rank(degree_three.iter().copied()), 160);
        assert_eq!(with_firing_rank(&degree_three, firing), 161);

        let mut degree_four = degree_three.clone();
        for a in 0..14 {
            for b in a + 1..14 {
                for c in b + 1..14 {
                    for d in c + 1..14 {
                        degree_four.push(and(
                            and(columns[a], columns[b]),
                            and(columns[c], columns[d]),
                        ));
                    }
                }
            }
        }
        assert_eq!(gf2_rank(degree_four.iter().copied()), 226);
        assert_eq!(
            with_firing_rank(&degree_four, firing),
            226,
            "the exact recovery boundary must first appear at degree four"
        );
    }

    #[test]
    fn seven_carrier_role_automorphisms_preserve_every_decode_class() {
        for seed in 0..64u64 {
            let mut rng = StdRng::seed_from_u64(0x7d15_7000 + seed);
            let roles = seven_carrier_role_automorphism(&mut rng);
            let mut seen_roles = [false; 7];
            for &role in &roles {
                assert!(!seen_roles[role as usize]);
                seen_roles[role as usize] = true;
            }
            for input in 0u8..128 {
                let relabeled = (0..7).fold(0u8, |word, canonical| {
                    word | (((input >> canonical) & 1) << roles[canonical as usize])
                });
                assert_eq!(
                    seven_carrier_decode_word(relabeled),
                    seven_carrier_decode_word(input),
                    "seed {seed} changed D at {input:#09b}"
                );
            }
        }
    }

    #[test]
    fn seven_carrier_distributed_fold_is_exact_on_arbitrary_representatives() {
        let n = 3usize;
        let state = SevenCarrierState::home(n);
        let cfg = ProdConfig::off();
        let mut ledger = ProdLedger::new(n, &cfg, 7 * n, None);
        let mut rng = StdRng::seed_from_u64(0x7d15_f01d);
        let mut fold = Vec::new();
        ledger.fold_seven_distributed(&XGate::cnot(0, 1), &state, 0, n, &mut rng, &mut fold);
        // Six decode atoms produce six source fragments.  Five boundaries,
        // each carrying a three-gate shear, give 6 + 5*3 gates and no U0.
        assert_eq!(fold.len(), 21);
        assert!(
            fold.iter().all(|gate| !gate.ctrls.is_empty()),
            "the refresh fold emitted an always-firing gate"
        );
        let conditional_targets: std::collections::HashSet<u16> =
            fold.iter().map(|gate| gate.target).collect();
        assert_eq!(conditional_targets.len(), 5);

        let pack = |value: usize, carrier: u8| -> u64 {
            (0..7).fold(0u64, |word, lane| {
                word | ((((carrier >> lane) & 1) as u64) << (lane * n + value))
            })
        };
        let unpack = |physical: u64, value: usize| -> u8 {
            (0..7).fold(0u8, |carrier, lane| {
                carrier | ((((physical >> (lane * n + value)) & 1) as u8) << lane)
            })
        };
        for target_carrier in 0u8..128 {
            for source_carrier in 0u8..128 {
                let input = pack(0, target_carrier) | pack(1, source_carrier);
                let output = eval_u64(&fold, input);
                assert_eq!(
                    seven_carrier_decode_word(unpack(output, 0)),
                    seven_carrier_decode_word(target_carrier)
                        ^ seven_carrier_decode_word(source_carrier),
                    "target={target_carrier:#09b} source={source_carrier:#09b}"
                );
                assert_eq!(
                    unpack(output, 1),
                    source_carrier,
                    "the source representative was modified"
                );
            }
        }
    }

    #[test]
    fn seven_carrier_partitioned_fold_reaches_128_and_is_exact() {
        // A CNOT's six source fragments need five selector bits to clear the
        // floor.  The eligible prefix supplies exactly values 2..6 after the
        // target/control exclusion; values 7 and 8 deliberately sit outside
        // it, modeling the sliced sandwich's fixed upper half.
        let n = 9usize;
        let live_helper_prefix = 7usize;
        let state = SevenCarrierState::home(n);
        let cfg = ProdConfig::off();
        let mut ledger = ProdLedger::new(n, &cfg, 7 * n, None);
        let mut rng = StdRng::seed_from_u64(0x7d15_1280);
        let mut fold = Vec::new();
        ledger.fold_seven_distributed(
            &XGate::cnot(0, 1),
            &state,
            128,
            live_helper_prefix,
            &mut rng,
            &mut fold,
        );
        // Six original decode atoms need five polarity bits: 6*32=192
        // branches.  There is one three-gate shear at every boundary.
        assert_eq!(ledger.distributed_fold_original_fragments, vec![6]);
        assert_eq!(ledger.distributed_fold_fragments, vec![192]);
        assert_eq!(ledger.cg_fragments, 192);
        assert_eq!(fold.len(), 192 + 3 * 191);
        assert!(fold.iter().all(|gate| !gate.ctrls.is_empty()));
        let expected_helpers: std::collections::HashSet<u16> = (2..7).collect();
        for fragment in fold.iter().step_by(4) {
            let actual_helpers: std::collections::HashSet<u16> = fragment
                .ctrls
                .iter()
                .filter_map(|&(wire, _)| expected_helpers.contains(&wire).then_some(wire))
                .collect();
            assert_eq!(actual_helpers, expected_helpers);
            assert!(
                fragment
                    .ctrls
                    .iter()
                    .all(|&(wire, _)| wire != 7 && wire != 8)
            );
        }

        let pack = |value: usize, carrier: u8| -> u64 {
            (0..7).fold(0u64, |word, lane| {
                word | ((((carrier >> lane) & 1) as u64) << (lane * n + value))
            })
        };
        let unpack = |physical: u64, value: usize| -> u8 {
            (0..7).fold(0u8, |carrier, lane| {
                carrier | ((((physical >> (lane * n + value)) & 1) as u8) << lane)
            })
        };
        for helper_seed in [0u8, 0x7f, 0x55, 0x2a] {
            for target_carrier in 0u8..128 {
                for source_carrier in 0u8..128 {
                    let helpers: Vec<u8> = (2..n)
                        .map(|value| helper_seed.rotate_left((value - 2) as u32) & 0x7f)
                        .collect();
                    let mut input = pack(0, target_carrier) | pack(1, source_carrier);
                    for (value, &helper) in (2..n).zip(&helpers) {
                        input |= pack(value, helper);
                    }
                    let output = eval_u64(&fold, input);
                    assert_eq!(
                        seven_carrier_decode_word(unpack(output, 0)),
                        seven_carrier_decode_word(target_carrier)
                            ^ seven_carrier_decode_word(source_carrier)
                    );
                    assert_eq!(unpack(output, 1), source_carrier);
                    for (value, &helper) in (2..n).zip(&helpers) {
                        assert_eq!(unpack(output, value), helper);
                    }
                }
            }
        }
    }

    #[test]
    fn seven_carrier_partitioned_floor1024_is_exact_and_shuffles_each_cell_block() {
        // A two-control fold has 6*6=36 source fragments. Five independent
        // helper bits raise it to 36*32=1152 branches, just over floor 1024.
        let n = 8usize;
        let state = SevenCarrierState::home(n);
        let cfg = ProdConfig::off();
        let mut ledger = ProdLedger::new(n, &cfg, 7 * n, None);
        let mut rng = StdRng::seed_from_u64(0x7d15_1024);
        let gate = XGate::conj(0, [(1u16, true), (2u16, true)]).unwrap();
        let mut fold = Vec::new();
        ledger.fold_seven_distributed(&gate, &state, 1024, n, &mut rng, &mut fold);
        assert_eq!(ledger.distributed_fold_original_fragments, vec![36]);
        assert_eq!(ledger.distributed_fold_fragments, vec![1152]);
        assert_eq!(ledger.distributed_fold_floors, vec![1024]);
        assert_eq!(fold.len(), 4 * 1152 - 3);

        // Source branches occupy every fourth position. Every 32-branch group
        // must enumerate the full cell cube, and independent shuffles should
        // give consecutive original fragments different orders.
        let branches: Vec<&XGate> = fold.iter().step_by(4).collect();
        assert_eq!(branches.len(), 1152);
        let pattern = |fragment: &XGate| -> u8 {
            (3u16..8).enumerate().fold(0u8, |word, (bit, helper)| {
                let polarity = fragment
                    .ctrls
                    .iter()
                    .find_map(|&(wire, polarity)| (wire == helper).then_some(polarity))
                    .expect("every branch must carry every helper literal");
                word | ((polarity as u8) << bit)
            })
        };
        let first: Vec<u8> = branches[..32].iter().map(|gate| pattern(gate)).collect();
        let second: Vec<u8> = branches[32..64].iter().map(|gate| pattern(gate)).collect();
        assert_eq!(
            first
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            32
        );
        assert_eq!(
            second
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            32
        );
        assert_ne!(first, second, "cell order was reused across fragments");

        let pack = |value: usize, carrier: u8| -> u64 {
            (0..7).fold(0u64, |word, lane| {
                word | ((((carrier >> lane) & 1) as u64) << (lane * n + value))
            })
        };
        let unpack = |physical: u64, value: usize| -> u8 {
            (0..7).fold(0u8, |carrier, lane| {
                carrier | ((((physical >> (lane * n + value)) & 1) as u8) << lane)
            })
        };
        for _ in 0..2048 {
            let carriers: [u8; 8] = std::array::from_fn(|_| rng.random_range(0..128));
            let input = carriers
                .iter()
                .enumerate()
                .fold(0u64, |word, (value, &carrier)| word | pack(value, carrier));
            let output = eval_u64(&fold, input);
            assert_eq!(
                seven_carrier_decode_word(unpack(output, 0)),
                seven_carrier_decode_word(carriers[0])
                    ^ (seven_carrier_decode_word(carriers[1])
                        & seven_carrier_decode_word(carriers[2]))
            );
            for (value, &carrier) in carriers.iter().enumerate().skip(1) {
                assert_eq!(unpack(output, value), carrier);
            }
        }
    }

    #[test]
    fn seven_carrier_initial_boundary_partition_preserves_complemented_gates() {
        let band: Vec<u16> = (20..32).collect();
        let supports: Vec<Vec<u64>> = (0..band.len()).map(|bit| vec![1u64 << bit]).collect();
        let original = vec![
            XGate::conj(0, [(1u16, true), (2u16, false)]).unwrap(),
            XGate::from_g57([3, 4, 5]),
        ];
        let mut rng = StdRng::seed_from_u64(0xb0a1_0d10);
        let mut partitioned = Vec::new();
        let emitted = emit_partitioned_initial_injection(
            &original,
            &band,
            &supports,
            4,
            &mut rng,
            &mut partitioned,
        );
        // Pure conjunction: 16 cells. Complemented conjunction: cell-only
        // plus F-and-cell in every cell, for 32 more emissions.
        assert_eq!(emitted, 48);
        assert!(partitioned.iter().all(|gate| !gate.comp));
        for _ in 0..4096 {
            let input = rng.random::<u64>() & ((1u64 << 32) - 1);
            assert_eq!(eval_u64(&partitioned, input), eval_u64(&original, input));
        }
    }

    #[test]
    fn seven_carrier_floor4096_boundary_r10_preserves_the_zero_slice() {
        let n = 12usize;
        let band = 32usize;
        let total = 7 * n + band;
        let mut prod = ProdConfig::production_seven_carrier();
        prod.gray_fold = 0;
        prod.fill_nl = 0;
        prod.band = band;
        prod.cg_jitter = 0;
        let mut rng = StdRng::seed_from_u64(0xb0a1_0d11);
        let source = [XGate::cnot(0, 1)];
        let circuit = gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed_partitioned_floor4096_boundary_r10_live_prefix_unshuffled(
            &source,
            n,
            n,
            1,
            total,
            &prod,
            &mut rng,
        );
        assert_eq!(circuit.num_wires, total);
        assert!(
            circuit
                .gates
                .iter()
                .rev()
                .take(5 * n)
                .all(|gate| (gate.target as usize) < n),
            "the boundary fixture unexpectedly retained the terminal high-wire fill"
        );

        let evaluate = |input: u16| -> u16 {
            let mut state = vec![false; total];
            for wire in 0..n {
                state[wire] = input & (1 << wire) != 0;
            }
            for gate in &circuit.gates {
                let firing = gate.comp
                    ^ gate
                        .ctrls
                        .iter()
                        .all(|&(wire, polarity)| state[wire as usize] == polarity);
                state[gate.target as usize] ^= firing;
            }
            (0..n).fold(0u16, |word, wire| word | ((state[wire] as u16) << wire))
        };
        for input in [0u16, 1, 0x555, 0xaaa, 0xfff, 0x31c, 0x8e7] {
            let expected = input ^ (((input >> 1) & 1) << 0);
            assert_eq!(evaluate(input), expected);
        }
    }

    #[test]
    fn seven_carrier_floor4096_terminal_fence_is_exact_and_adj32_stays_in_prefix() {
        let n = 12usize;
        let band = 32usize;
        let total = 7 * n + band;
        let mut prod = ProdConfig::production_seven_carrier();
        prod.gray_fold = 0;
        prod.fill_nl = 0;
        prod.band = band;
        prod.cg_jitter = 0;
        let source = [XGate::cnot(0, 1)];

        let mut ordered_rng = StdRng::seed_from_u64(0xb0a1_0d12);
        let (ordered, ordered_terminal_start) = gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed_partitioned_floor4096_boundary_r10_live_prefix_terminal_fenced_unshuffled(
            &source,
            n,
            n,
            1,
            total,
            &prod,
            &mut ordered_rng,
        );
        let mut shuffled_rng = StdRng::seed_from_u64(0xb0a1_0d12);
        let (shuffled, shuffled_terminal_start) = gadgetize_xgates_with_slice_zero_ccnot_seven_carrier_distributed_partitioned_floor4096_boundary_r10_live_prefix_terminal_fenced_adj32(
            &source,
            n,
            n,
            1,
            total,
            &prod,
            &mut shuffled_rng,
        );

        assert_eq!(ordered.num_wires, shuffled.num_wires);
        assert_eq!(ordered.gates.len(), shuffled.gates.len());
        assert_eq!(ordered_terminal_start, shuffled_terminal_start);
        assert!(ordered_terminal_start > 0);
        assert!(ordered_terminal_start < ordered.gates.len());
        assert_eq!(
            &ordered.gates[ordered_terminal_start..],
            &shuffled.gates[shuffled_terminal_start..],
            "a prefix-only reorder modified the protected terminal suffix"
        );
        assert_ne!(
            &ordered.gates[..ordered_terminal_start],
            &shuffled.gates[..shuffled_terminal_start],
            "the deterministic adjacent passes did not reorder the prefix"
        );

        let evaluate = |circuit: &CnotCircuit, input: u16| -> u16 {
            let mut state = vec![false; total];
            for wire in 0..n {
                state[wire] = input & (1 << wire) != 0;
            }
            for gate in &circuit.gates {
                let firing = gate.comp
                    ^ gate
                        .ctrls
                        .iter()
                        .all(|&(wire, polarity)| state[wire as usize] == polarity);
                state[gate.target as usize] ^= firing;
            }
            (0..n).fold(0u16, |word, wire| word | ((state[wire] as u16) << wire))
        };
        for input in [0u16, 1, 0x555, 0xaaa, 0xfff, 0x31c, 0x8e7] {
            let expected = input ^ (((input >> 1) & 1) << 0);
            assert_eq!(evaluate(&ordered, input), expected);
            assert_eq!(evaluate(&shuffled, input), expected);
        }
    }

    #[test]
    fn seven_carrier_distributed_public_paths_preserve_dirty_high_inputs() {
        // Six values leave four distinct helpers for the CNOT's floor-128
        // partition while keeping the complete public port inside u64.
        let n = 6usize;
        let band = 8usize;
        let total = 7 * n + band;
        let source = vec![XGate::from_g57([0, 1, 2]), XGate::cnot(3, 0)];
        let mut prod = ProdConfig::production_seven_carrier();
        prod.gray_fold = 0;
        prod.band = band;
        prod.cg_jitter = 0;

        let mut shuffled_rng = StdRng::seed_from_u64(0x7d15_e2e1);
        let shuffled =
            gadgetize_xgates_seven_carrier_distributed(&source, n, 1, &prod, &mut shuffled_rng);
        let mut ordered_rng = StdRng::seed_from_u64(0x7d15_e2e2);
        let ordered = gadgetize_xgates_seven_carrier_distributed_unshuffled(
            &source,
            n,
            1,
            &prod,
            &mut ordered_rng,
        );
        let mut partitioned_rng = StdRng::seed_from_u64(0x7d15_e2e3);
        let partitioned = gadgetize_xgates_seven_carrier_distributed_partitioned(
            &source,
            n,
            1,
            &prod,
            &mut partitioned_rng,
        );
        let mut partitioned_ordered_rng = StdRng::seed_from_u64(0x7d15_e2e4);
        let partitioned_ordered = gadgetize_xgates_seven_carrier_distributed_partitioned_unshuffled(
            &source,
            n,
            1,
            &prod,
            &mut partitioned_ordered_rng,
        );
        let low_mask = (1u64 << n) - 1;
        let high_mask = ((1u64 << total) - 1) ^ low_mask;
        let junk = [
            0,
            high_mask,
            0xaaaa_aaaa & high_mask,
            0x5555_5555 & high_mask,
            0x9249_2492 & high_mask,
        ];
        for circuit in [&shuffled, &ordered, &partitioned, &partitioned_ordered] {
            assert_eq!(circuit.num_wires, total);
            for low in 0..=low_mask {
                let expected = eval_u64(&source, low) & low_mask;
                for &high in &junk {
                    assert_eq!(
                        eval_u64(&circuit.gates, low | high) & low_mask,
                        expected,
                        "low={low:#x} high={high:#x}"
                    );
                }
            }
        }
    }

    #[test]
    fn rejected_direct_switch_trace_has_a_measured_boundary() {
        // This models the earlier direct class-switch candidate, not the
        // shipped opt-in shear fold.  Keep its falsification pinned: it looked
        // good in isolation but one later U0 put the firing bit back in span,
        // which is why the implemented path removes every fixed U0 instead.
        type Signature = [u64; 4];

        fn rank(signatures: impl IntoIterator<Item = Signature>) -> usize {
            let mut basis = [[0u64; 4]; 256];
            let mut rank = 0usize;
            for mut signature in signatures {
                loop {
                    let Some(pivot) = (0..256)
                        .rev()
                        .find(|&bit| signature[bit / 64] & (1u64 << (bit % 64)) != 0)
                    else {
                        break;
                    };
                    if basis[pivot] == [0; 4] {
                        basis[pivot] = signature;
                        rank += 1;
                        break;
                    }
                    for word in 0..4 {
                        signature[word] ^= basis[pivot][word];
                    }
                }
            }
            rank
        }

        fn contains(features: &[Signature], target: Signature) -> bool {
            rank(features.iter().copied())
                == rank(features.iter().copied().chain(std::iter::once(target)))
        }

        // A fixed member of the randomized family: R toggles c0,c1 by !c3;
        // the middle is the c3/c5-oriented class-switch core.
        let plan: Vec<(u8, Vec<(u8, bool)>)> = vec![
            (0, vec![(3, false)]),
            (1, vec![(3, false)]),
            (3, vec![(5, true)]),
            (1, vec![(4, false)]),
            (1, vec![(4, true), (5, false)]),
            (1, vec![(4, true), (5, true), (6, true)]),
            (1, vec![(3, false)]),
            (0, vec![(3, false)]),
        ];

        let trace = |suffix_updates: usize| {
            let mut columns: Vec<Signature> = Vec::new();
            let mut owners: Vec<u8> = Vec::new();
            let mut firing = [0u64; 4];
            for input in 0u64..128 {
                for fires in 0u64..2 {
                    let row = (2 * input + fires) as usize;
                    if fires != 0 {
                        firing[row / 64] |= 1u64 << (row % 64);
                    }
                    let mut state = input;
                    // Build each row independently, then merge it into the
                    // already allocated chronological columns.
                    let mut column_index = 0usize;
                    let mut apply_and_record =
                        |target: u8, selector: &[(u8, bool)], conditional: bool| {
                            if !conditional || fires != 0 {
                                let fire = selector.iter().all(|&(wire, polarity)| {
                                    ((state >> wire) & 1 != 0) == polarity
                                });
                                if fire {
                                    state ^= 1u64 << target;
                                }
                            }
                            if columns.len() == column_index {
                                columns.push([0; 4]);
                                owners.push(target);
                            }
                            debug_assert_eq!(owners[column_index], target);
                            if state & (1u64 << target) != 0 {
                                columns[column_index][row / 64] |= 1u64 << (row % 64);
                            }
                            column_index += 1;
                        };

                    for &(target, controls) in SEVEN_CARRIER_U0_GATES {
                        apply_and_record(target, controls, false);
                    }
                    for (target, selector) in &plan {
                        apply_and_record(*target, selector, true);
                    }
                    for _ in 0..suffix_updates {
                        for &(target, controls) in SEVEN_CARRIER_U0_GATES {
                            apply_and_record(target, controls, false);
                        }
                    }
                }
            }
            let mut affine = vec![[u64::MAX; 4]];
            affine.extend(columns.iter().copied());
            (affine, owners, firing)
        };

        let (isolated, _, firing) = trace(0);
        assert!(
            !contains(&isolated, firing),
            "the isolated distributed block unexpectedly recovered its firing bit"
        );

        // Exact limitation: this local construction raises checkpoint distance
        // but does not remove the firing bit forever.  One subsequent fixed U0
        // already puts it back in the *global* affine span.
        let (after_one, _, firing) = trace(1);
        assert!(contains(&after_one, firing));

        // The corresponding one-wire catalog lasts longer for this fixed
        // member, but it too eventually closes after repeated identical U0s.
        let (after_six, owners_six, firing) = trace(6);
        for wire in 0..7u8 {
            let mut features = vec![[u64::MAX; 4]];
            features.extend(
                after_six
                    .iter()
                    .skip(1)
                    .zip(&owners_six)
                    .filter_map(|(&signature, &owner)| (owner == wire).then_some(signature)),
            );
            assert!(!contains(&features, firing), "wire {wire} closed too early");
        }
        let (after_seven, owners_seven, firing) = trace(7);
        assert!((0..7u8).any(|wire| {
            let mut features = vec![[u64::MAX; 4]];
            features.extend(
                after_seven
                    .iter()
                    .skip(1)
                    .zip(&owners_seven)
                    .filter_map(|(&signature, &owner)| (owner == wire).then_some(signature)),
            );
            contains(&features, firing)
        }));
    }

    #[test]
    fn seven_carrier_gadget_preserves_low_wires_for_dirty_high_inputs() {
        let n = 6usize;
        let band = 6usize;
        let mut prod = ProdConfig::production_seven_carrier();
        prod.band = band;

        // Pin the production [2,2,2,3] mask plan and certify that the quartic
        // carrier decode still takes the adjusted Gray path. Keep this local
        // audit deterministic, while the end-to-end runs below retain the
        // production preset's extra per-value mask-atom jitter.
        let mut audit_prod = prod;
        audit_prod.cg_jitter = 0;
        let state = SevenCarrierState::home(n);
        let mut audit_rng = StdRng::seed_from_u64(0x7ca1_67a7);
        let mut ledger = ProdLedger::new(n, &audit_prod, 7 * n, None);
        let mut injection = Vec::new();
        ledger.inject_all(&state.c0_view(), &mut audit_rng, &mut injection);
        for slots in &ledger.slots {
            assert_eq!(
                slots
                    .iter()
                    .map(|slot| slot.factors.len())
                    .collect::<Vec<_>>(),
                vec![2, 2, 2, 3]
            );
        }
        let mut optimized_fold = Vec::new();
        ledger.fold_seven(
            &XGate::from_g57([0, 1, 2]),
            &state,
            &mut audit_rng,
            &mut optimized_fold,
        );
        assert_eq!(
            ledger.cg_gray, 1,
            "fixture missed the seven-carrier Gray fold"
        );
        assert!(
            optimized_fold.iter().all(|gate| gate.width() <= 4),
            "adjusted Gray fold emitted a wider-than-decode gate"
        );

        let source = vec![
            XGate::from_g57([0, 1, 2]),
            XGate::cnot(3, 4),
            XGate::conj(5, [(0u16, true), (2u16, false)]).unwrap(),
        ];
        let low_mask = (1u64 << n) - 1;
        let total = 7 * n + band;
        let high_mask = ((1u64 << total) - 1) ^ low_mask;
        let fixed_junk = [
            0,
            high_mask,
            0xaaaa_aaaa_aaaa_aaaau64 & high_mask,
            0x5555_5555_5555_5555u64 & high_mask,
            0x9249_2492_4924_9249u64 & high_mask,
            0x36db_6db6_db6d_b6dbu64 & high_mask,
        ];

        for seed in 0..3u64 {
            let mut rng = StdRng::seed_from_u64(0x7ca1_0000 + seed);
            let gadget = gadgetize_xgates_seven_carrier(&source, n, 2, &prod, &mut rng);
            assert_eq!(gadget.num_wires, total, "expected 7*n + band wires");

            for low in 0..=low_mask {
                let expected = eval_u64(&source, low) & low_mask;
                for (junk_index, &junk) in fixed_junk.iter().enumerate() {
                    assert_eq!(
                        eval_u64(&gadget.gates, low | junk) & low_mask,
                        expected,
                        "seed={seed} low={low:#x} junk pattern {junk_index}"
                    );
                }
                let junk = (low
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    .rotate_left((low as u32) & 31))
                    & high_mask;
                assert_eq!(
                    eval_u64(&gadget.gates, low | junk) & low_mask,
                    expected,
                    "seed={seed} low={low:#x} hashed high junk"
                );
            }
        }
    }

    #[test]
    fn seven_carrier_public_cnot_and_slice_wrappers_have_the_expected_port() {
        let n = 6usize;
        let band = 6usize;
        let mut prod = ProdConfig::production_seven_carrier();
        prod.band = band;
        prod.cg_jitter = 0;
        let total = 7 * n + band;
        let low_mask = (1u64 << n) - 1;
        let high_mask = ((1u64 << total) - 1) ^ low_mask;
        let main = CircuitSeq {
            gates: vec![[0, 1, 2]],
        };
        let source = vec![XGate::from_g57([0, 1, 2])];

        let mut bare_rng = StdRng::seed_from_u64(0x7ca2_0001);
        let bare = gadgetize_cnot_seven_carrier(&main, n, 1, &prod, &mut bare_rng);
        let mut slice_cnot_rng = StdRng::seed_from_u64(0x7ca2_0002);
        let slice_cnot = gadgetize_with_slice_zero_ccnot_seven_carrier(
            &main,
            n,
            1,
            10 * n,
            &MaskConfig::off(),
            &prod,
            &mut slice_cnot_rng,
        );
        let mut slice_xgate_rng = StdRng::seed_from_u64(0x7ca2_0003);
        let slice_xgate = gadgetize_xgates_with_slice_zero_ccnot_seven_carrier(
            &source,
            n,
            1,
            10 * n,
            &prod,
            &mut slice_xgate_rng,
        );
        assert_eq!(bare.num_wires, total);
        assert_eq!(slice_cnot.num_wires, total);
        assert_eq!(slice_xgate.num_wires, total);

        for low in 0..=low_mask {
            let expected = eval_u64(&source, low) & low_mask;
            assert_eq!(eval_u64(&bare.gates, low) & low_mask, expected);
            assert_eq!(eval_u64(&bare.gates, low | high_mask) & low_mask, expected);
            assert_eq!(eval_u64(&slice_cnot.gates, low) & low_mask, expected);
            assert_eq!(eval_u64(&slice_xgate.gates, low) & low_mask, expected);
        }
    }

    #[test]
    fn seven_carrier_empty_source_is_the_identity_for_arbitrary_high_junk() {
        let n = 3usize;
        let mut prod = ProdConfig::production_seven_carrier();
        prod.band = 6;
        prod.cg_jitter = 0;
        let total = 7 * n + prod.band_size(n);
        let low_mask = (1u64 << n) - 1;
        let high_mask = ((1u64 << total) - 1) ^ low_mask;
        let mut rng = StdRng::seed_from_u64(0x7ca3_0001);
        let gadget = gadgetize_xgates_seven_carrier(&[], n, 1, &prod, &mut rng);
        assert_eq!(gadget.num_wires, total);

        for low in 0..=low_mask {
            for junk in [
                0,
                high_mask,
                0xaaaa_aaaa_aaaa_aaaau64 & high_mask,
                0x9249_2492_4924_9249u64 & high_mask,
            ] {
                assert_eq!(eval_u64(&gadget.gates, low | junk) & low_mask, low);
            }
        }
    }
}
