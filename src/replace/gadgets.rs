use crate::circuit::circuit::CircuitSeq;
use crate::postmix::xgate::XGate;
use crate::random::random_data::shoot_random_gate;
use rand::{prelude::SliceRandom, Rng};
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
    use rand::{rngs::StdRng, SeedableRng};

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
    use crate::circuit::circuit::Gate;
    use rand::{rngs::StdRng, Rng, SeedableRng};
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
                    assert!(transformed
                        .gates
                        .iter()
                        .flatten()
                        .all(|&w| (w as usize) < 3 * n));

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

            assert!(block
                .circuit
                .gates
                .iter()
                .flatten()
                .all(|&w| (w as usize) < 3 * n));
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
    use rand::{rngs::StdRng, SeedableRng};
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
    use rand::{rngs::StdRng, Rng, SeedableRng};

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
    use rand::{rngs::StdRng, SeedableRng};

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
#[derive(Clone, Copy, Debug)]
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

            epoch: 0,
            refill_data: 0,
            single: 0,
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
    pub fn production_single(n: usize) -> ProdConfig {
        ProdConfig {
            k: 1,
            deg: 2,
            k_hi: 2,
            deg_hi: 3,
            band: n,
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
            ladder_cap: 3,
            cg_jitter: 50,

            epoch: 5,
            refill_data: 50,
            single: 1,
        }
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
        (((4 * n * self.k_total()) as f64).sqrt().ceil() as usize)
            .max(6)
            .max(self.max_deg() + 3)
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
/// Borrows come from `0..carrier_total` bar the target and the fragment's own
/// literals, and only widen to `0..borrow_total` (band included) when that
/// leaves too few — which a fragment whose literals cover most of the carriers
/// otherwise would. Any wire is sound (the double sweep restores it before
/// anything else can read it); carriers are merely preferred, since a band
/// wire's content is constant across the body and a partial product parked on
/// one is a longer-lived difference than the same product on a churning
/// carrier.
fn emit_narrow_fragment(
    target: u16,
    lits: &[(u16, bool)],
    cap: usize,
    carrier_total: usize,
    borrow_total: usize,
    forbidden: &[u16],
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
    let free_in = |pool: usize| pool.saturating_sub(taken.iter().filter(|&&w| w < pool).count());
    let pool = if free_in(carrier_total) >= rung_count {
        carrier_total
    } else {
        borrow_total
    };
    assert!(
        free_in(pool) >= rung_count,
        "narrow fragment needs {rung_count} borrows but only {} of {pool} wires are free",
        free_in(pool)
    );
    let mut borrowed: Vec<u16> = Vec::with_capacity(rung_count);
    for _ in 0..rung_count {
        let h = random_wire_except(pool, &taken, rng) as u16;
        taken.push(h as usize);
        borrowed.push(h);
    }
    // Rung i computes borrowed[i] ^= borrowed[i-1] & (next `chunk` literals);
    // rung 0 takes the first `cap` literals outright.
    let mut rungs: Vec<Vec<XGate>> = Vec::with_capacity(rung_count);
    let mut consumed = 0;
    for (i, &h) in borrowed.iter().enumerate() {
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
        rungs.push(vec![
            XGate::conj(h, rung_lits).expect("borrowed wires are distinct from the literals")
        ]);
    }
    // Target gate: last borrow plus whatever literals remain.
    let mut t_lits: Vec<(u16, bool)> = vec![(*borrowed.last().unwrap(), true)];
    t_lits.extend_from_slice(&lits[consumed..]);
    debug_assert!(t_lits.len() <= cap);
    let mut emit_target = |rng: &mut _, out: &mut Vec<XGate>| {
        if t_lits.len() <= 2 {
            emit_g57_form(target, &t_lits, rng, out);
        } else {
            out.push(XGate::conj(target, t_lits.iter().copied()).expect("distinct wires"));
        }
    };
    // One sweep block: target gate, down the rungs, back up (excluding rung 0).
    // Emitted twice; every rung appears an even number of times, so all dirty
    // contributions and all g57 complements cancel exactly.
    for _ in 0..2 {
        emit_target(rng, out);
        for rung in rungs.iter().rev() {
            out.extend(rung.iter().cloned());
        }
        for rung in rungs.iter().skip(1) {
            out.extend(rung.iter().cloned());
        }
    }
    false
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
    /// Emit DB-eligible fold fragments in the g57/CNOT vocabulary.
    g57_narrow: bool,
    ladder_cap: usize,
    cg_jitter: usize,
    /// refs[wire] = live slot factors naming that wire. A wire with refs > 0
    /// is "named": nothing may write it until every naming slot is released.
    refs: Vec<u32>,
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
    rolled: u64,
    migrated: u64,
    retired: u64,
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
    ledger_consts: u64,
}

impl ProdLedger {
    fn new(
        n: usize,
        cfg: &ProdConfig,
        carrier_total: usize,
        sched: Option<Vec<Vec<usize>>>,
    ) -> ProdLedger {
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
            g57_narrow: cfg.g57_narrow > 0,
            ladder_cap: cfg.ladder_cap,
            cg_jitter: cfg.cg_jitter,
            refs: vec![0; carrier_total],
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
            rolled: 0,
            migrated: 0,
            retired: 0,
            next_retire: 0,
            retire_queue: Vec::new(),
            refill_used_carriers: false,
            cg_fragments: 0,
            cg_narrow: 0,
            cg_laddered: 0,
            cg_fossils: 0,
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

    fn add_refs(&mut self, slot: &ProdSlot) {
        if self.dist {
            for w in slot.wires() {
                self.refs[w] += 1;
            }
        }
    }

    fn drop_refs(&mut self, slot: &ProdSlot) {
        if self.dist {
            for w in slot.wires() {
                self.refs[w] -= 1;
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

    /// A fresh, currently-unused degree-`deg` slot over the band variables.
    fn draw_slot(&mut self, deg: usize, rng: &mut impl Rng) -> ProdSlot {
        let band_len = self.loc.len() as u16;
        for _ in 0..100_000 {
            let mut vars: Vec<u16> = Vec::with_capacity(deg);
            while vars.len() < deg {
                let b = rng.random_range(0..band_len);
                if !vars.contains(&b) {
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
                self.carrier_total,
                self.borrow_total(),
                &sib,
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
                self.carrier_total,
                self.borrow_total(),
                &forbidden,
                rng,
                out,
            )
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
            self.draw_slot(deg, rng)
        };
        let konst = self.emit_slot(value, &slot, state, rng, out);
        self.consts[value] ^= konst;
        self.add_refs(&slot);
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
            self.drop_refs(&old);
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
        let value = rng.random_range(0..state.n);
        if self.slots[value].is_empty() {
            return;
        }
        let old_index = rng.random_range(0..self.slots[value].len());
        let deg = self.slots[value][old_index].factors.len();
        self.inject(value, deg, state, rng, out);
        let old = self.slots[value].remove(old_index);
        let konst = self.emit_slot(value, &old, state, rng, out);
        self.consts[value] ^= konst;
        self.drop_refs(&old);
        self.used.remove(&old);
        self.resourced += 1;
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
            self.used.remove(&old);
            self.migrated += 1;
        }

        // 2. Rewrite the wire. Sources are other band wires, mixed with
        //    carriers at `refill_data` percent -- see ProdConfig::refill_data
        //    for why neither extreme is right.
        let wire = self.loc[var as usize];
        // Sources are chosen BY ROLE, not by wire number. Rolling exchanges the
        // carrier and band roles, so a numeric range over `0..carrier_total`
        // does not mean "a carrier" -- it can land on a band variable, or on
        // the target wire itself, and then `refill_data` percent is not the
        // rate it claims to be. Carriers are reached through `state.pairs` and
        // band variables through `loc`, which are the two role maps.
        let mut draw = |rng: &mut dyn rand::RngCore| -> u16 {
            let use_carrier = state.n > 0 && (rng.next_u32() as usize % 100) < refill_data;
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
        let p = draw(rng);
        if p != wire {
            out.push(XGate::cnot(wire, p));
        }
        for _ in 0..fill_nl.max(1) {
            let s1 = draw(rng);
            let s2 = draw(rng);
            if s1 == s2 || s1 == wire || s2 == wire {
                continue;
            }
            let lits = [(s1, rng.random::<bool>()), (s2, rng.random::<bool>())];
            emit_g57_form(wire, &lits, rng, out);
        }
        if refill_data > 0 {
            self.refill_used_carriers = true;
        }
        self.retired += 1;
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
        for _ in 0..100_000 {
            let mut vars: Vec<u16> = Vec::with_capacity(deg);
            while vars.len() < deg {
                let b = rng.random_range(0..band_len);
                if b != avoid && !vars.contains(&b) {
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
    /// summand atoms (two carrier literals, k product-literal pairs, and a
    /// constant when c_w and/or a negative control polarity applies); every
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
                self.drop_refs(&old);
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
        let mut lists: Vec<Vec<Vec<(u16, bool)>>> = Vec::with_capacity(gate.ctrls.len());
        for &(w, positive) in &gate.ctrls {
            let w = w as usize;
            debug_assert_ne!(w, t);
            let (c0, c1) = state.pairs[w];
            // Single-carrier decode contributes ONE linear atom, not two: the
            // pair collapses to a single wire, and emitting it twice would
            // cancel. Every other atom (the mask terms, the constant) is
            // unchanged, so the fold's fragment count drops from
            // (2 + k)^arity to (1 + k)^arity at equal mask strength.
            let mut atoms: Vec<Vec<(u16, bool)>> = if self.single {
                vec![vec![(c0 as u16, true)]]
            } else {
                vec![vec![(c0 as u16, true)], vec![(c1 as u16, true)]]
            };
            for slot in &self.slots[w] {
                atoms.push(slot.lits(&self.loc));
            }
            // D'_w = D_w (+1 for a negative literal); with c_w set the two
            // constants cancel by parity.
            if self.consts[w] ^ !positive {
                atoms.push(Vec::new());
            }
            lists.push(atoms);
        }
        if gate.comp {
            self.consts[t] ^= true;
            self.ledger_consts += 1;
        }
        if self.cap >= 2 {
            self.fold_cg_narrow(t, &lists, state, rng, out);
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
        for atoms in frags {
            let target = self.free_carrier(t, state, rng);
            // Interleaving is only REQUIRED on the ladder path, but a fragment's
            // width is not known until its literals are normalized, and using a
            // different literal ORDER for laddered and unladdered fragments
            // would make the two populations distinguishable by control order
            // alone. Interleave uniformly; a conjunction does not care.
            let mut lits = interleave_atoms(&atoms);
            if normalize_lits(&mut lits).is_none() {
                // Contradictory literals (w AND !w): the term is 0.
                continue;
            }
            self.debug_check_fragment(target, &lits, state);
            let width = lits.len();
            if width <= 2 {
                self.cg_narrow += 1;
                // Realize the DB-eligible fragments in the g57/CNOT
                // vocabulary rather than as exact conjunctions. Passing the
                // width cap is not the same as being digestible: a comp=0
                // width-2 conjunction like `xy` is NOT in the X-free g57
                // span (which over {h,x,y} is <x, y, 1^xy>), so as a single
                // gate it is invisible to a store built from g57 circuits,
                // however narrow it looks. `~x~y` already is in the span,
                // which is why the gain is a fraction rather than all of it.
                // Costs 1-2 gates instead of 1; the residual goes to the
                // ledger exactly as the narrow path already does.
                if self.g57_narrow {
                    let konst = emit_g57_form(target, &lits, rng, out);
                    self.consts[t] ^= konst;
                    self.cg_fragments += 1;
                    continue;
                }
            } else if width <= self.ladder_cap {
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
                    self.carrier_total,
                    self.borrow_total(),
                    &forbidden,
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
        let mut frags: Vec<Vec<(u16, bool)>> = Vec::new();
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
                frags.push(lits);
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
        for lits in &frags {
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
                self.carrier_total,
                self.borrow_total(),
                &forbidden,
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
        for value in 0..state.n {
            // Strip in plan order (base terms first): the highest-degree
            // tower term covers the value longest at the tail boundary.
            while !self.slots[value].is_empty() {
                let slot = self.slots[value].remove(0);
                let konst = self.emit_slot(value, &slot, state, rng, out);
                self.consts[value] ^= konst;
                self.drop_refs(&slot);
                self.used.remove(&slot);
            }
            if self.consts[value] {
                let target = self.free_carrier(value, state, rng) as usize;
                let u = random_wire_except(self.carrier_total, &[target], rng) as u16;
                out.push(XGate::conj(target as u16, [(u, false)]).expect("distinct wires"));
                out.push(XGate::cnot(target as u16, u));
                self.consts[value] = false;
            }
        }
    }

    fn report(&self) {
        if self.enabled() {
            println!(
                "[prod] plan={:?} band={} src={} injected={} resourced={} rolled={} migrated={} retired={} \
                 degen_rejects={} cg_fragments={} cg_narrow={} laddered={} fossils={} ledger_consts={}{}",
                self.plan,
                self.loc.len(),
                if self.dist { "distributed" } else { "band" },
                self.injected,
                self.resourced,
                self.rolled,
                self.migrated,
                self.retired,
                self.degenerate_rejects,
                self.cg_fragments,
                self.cg_narrow,
                self.cg_laddered,
                self.cg_fossils,
                self.ledger_consts,
                if self.retired > 0 && self.refill_used_carriers {
                    "  [port-uniformity forfeited: carrier-sourced refills]"
                } else {
                    ""
                }
            );
        }
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
    for polarity in [true, false] {
        out.push(XGate::conj(target, [(source, true), (u, polarity)]).expect("distinct wires"));
    }
}

/// W0: fill each band wire with an unbiased data-dependent bit — its input
/// junk XOR a random (weight >= 2) subset of the data wires, emitted at the
/// input port while the data wires still hold x. Over uniform x every fill is
/// exactly unbiased; under the hmap / zero-slice conventions (non-data inputs
/// pinned to 0) the band reads exactly <alpha, x>. The band is never written
/// again, so every registered product term is time-invariant.
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
    let mut out: Vec<u32> = Vec::with_capacity(order.len());
    for &gi in order.iter() {
        let g = &gates[gi as usize];
        let mut span = 0usize;
        while span < out.len() && !XGate::collides(g, &gates[out[out.len() - 1 - span] as usize]) {
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
    let m = gates.len();
    if m < 2 {
        return;
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
    // never written again — the same frozen-source contract as the two-carrier
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

/// Relocate one value to a different carrier wire (the single-carrier RG).
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
    assert!(n >= 3, "slice_zero_ccnot_preblock requires n >= 3");
    let total = n + nondata;
    assert!(total <= u16::MAX as usize, "too many wires");
    let band = nondata.saturating_sub(n);
    assert!(
        gate_count >= nondata,
        "every non-data wire must be read: needs at least {nondata} gates"
    );

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
            let target = rng.random_range(0..n);
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
            return CnotCircuit {
                gates,
                num_wires: total,
            };
        }
    }
    panic!(
        "no slice preblock with every nonzero slice disturbed found at n={n} \
         band={band} gates={gate_count} in 1000 draws: {n} data wires may be too \
         few to disturb 2^{nondata} slices distinctly — raise n or lower --prod-band"
    );
}

/// Exhaustive check that only the all-zero slice leaves the data untouched:
/// every slice against every input. Affordable only while `2n + band` is
/// small, which is exactly the regime where wrong-slice fixes were ever
/// observed in the first place.
fn slice_preblock_fixes_only_zero_slice(gates: &[XGate], n: usize, nondata: usize) -> bool {
    let mask = (1u64 << n) - 1;
    (1..(1u64 << nondata)).all(|s| {
        (0..=mask).any(|x| crate::postmix::xgate::eval_u64(gates, x | (s << n)) & mask != x)
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
            emit_value_relocation(&mut state, carrier_total, &mut out, rng);
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
    prod_ledger.strip_all(&state, rng, &mut out);
    prod_ledger.report();

    // Route every value home (rolls can leave one on a former band wire), so
    // wires 0..n hold the values and n..total hold band junk.
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
    let mut circuit = slice_zero_preblock_dims(n, band, gate_count.max(band), rng);
    let gadget = gadgetize_xgates_single(source, n, rg_freq, prod, rng);
    circuit.num_wires = circuit.num_wires.max(gadget.num_wires);
    circuit.gates.extend(gadget.gates);
    commuting_shuffle(&mut circuit.gates, rng);
    circuit
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

pub const SANDWICH_SLICE_GATES_PER_WIRE_LOG: f64 = 1.0; // s = n log n
pub const SANDWICH_D_GATES_PER_WIRE_LOG2: f64 = 1.0; // m = n (log n)^2

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
    use crate::postmix::xgate::eval_u64;
    use rand::{rngs::StdRng, SeedableRng};

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
        assert!(fragments
            .iter()
            .all(|gate| !(gate.reads(2) && gate.reads(3))));
        assert!(fragments
            .iter()
            .all(|gate| !(gate.reads(4) && gate.reads(5))));
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
        let mut v = ((state >> pairs[value].0) ^ (state >> pairs[value].1)) & 1;
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
                    let konst = emit_narrow_fragment(
                        0,
                        &lits,
                        cap,
                        total,
                        total,
                        &[],
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

            epoch: 0,
            refill_data: 0,
            single: 0,
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
                assert!(fold.len() >= 16, "expected a wide fold, got {}", fold.len());
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
                // RG2's three transvections, each either a CNOT or its
                // two-conjunction form, so no wire is written only by
                // width-1 gates.
                assert!(
                    (3..=6).contains(&moved.len()),
                    "a roll is three transvections, got {}",
                    moved.len()
                );
                assert!(moved
                    .iter()
                    .all(|g| !g.comp && (1..=2).contains(&g.width())));
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

                epoch: 0,
                refill_data: 0,
                single: 0,
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

            epoch: 0,
            refill_data: 0,
            single: 0,
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

                epoch: 0,
                refill_data: 0,
                single: 0,
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

                epoch: 0,
                refill_data: 0,
                single: 1,
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
                epoch: 0,
                refill_data: 0,
                single: 1,
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
                epoch,
                refill_data,
                single: 0,
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

            epoch: 0,
            refill_data: 0,
            single: 0,
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
}
