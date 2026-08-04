use crate::circuit::circuit::CircuitSeq;
use crate::postmix::xgate::XGate;
use crate::random::random_data::shoot_random_gate;
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
