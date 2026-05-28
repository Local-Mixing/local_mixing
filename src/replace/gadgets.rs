use rand::{Rng, prelude::SliceRandom};
use std::collections::VecDeque;
use crate::circuit::circuit::CircuitSeq;

/// 6-gate homomorphic gadget for secret-shared g57.
/// Local wire map: 0=share_a, 3=share_c, 4=pad_c, 5=share_b, 6=pad_b.
const GADGET: [[u16; 3]; 6] = [
    [4,5,6],[0,4,6],[0,5,4],[4,5,6],[0,6,3],[0,3,5],
];

/// RG1: swap virtual values between two pairs.
/// Wire map: 0=share_i, 1=pad_i, 2=pad_j, 3=share_j.
const RG1: [[u16; 3]; 6] = [
    [1,2,3],[0,3,2],[3,1,0],[2,0,1],[0,3,2],[1,2,3],
];

/// RG2: swap pad wires between two pairs.
/// Wire map: 0=share_i, 1=pad_i, 2=pad_j, 3=share_j.
const RG2: [[u16; 3]; 6] = [
    [0,2,3],[1,0,2],[2,0,3],[2,3,0],[1,3,2],[3,2,0],
];

/// 20-gate W_i sequence (Ran Canetti's design, g57 gates only).
/// Local wires: 0=w0(i), 1=w1(n+i), 2=w2(p(i)), 3=w3(p(n+i)).
/// Effect: (w0,w1,w2,w3) -> (w2, w3, w0 XOR w1, w1).
/// Reversed sequence gives W_i^{-1} (each g57 gate is self-inverse).
const W_I_GATES: [[u16; 3]; 20] = [
    [0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,1,0],[2,0,1],
    [0,2,1],[0,1,2],[1,0,3],[1,3,0],[3,0,1],[3,1,0],
    [1,3,0],[1,0,3],[3,2,1],[2,1,3],[1,3,2],[2,3,1],
    [1,2,3],[3,1,2],
];

/// Secret-sharing state: pairs[v] = (share_wire, pad_wire) for virtual value v.
pub struct GadgetState {
    pub n: usize,
    pub pairs: Vec<(usize, usize)>,
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
            if w != active { break w; }
        };
        let ctrl2 = loop {
            let w = rng.random_range(0..total) as u16;
            if w != active { break w; }
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
    let tmp = state.pairs[i].1;
    state.pairs[i].1 = state.pairs[j].1;
    state.pairs[j].1 = tmp;
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
    let bookend_size = (2 * n * (n as f64).ln() as usize).max(64);
    let mut out: Vec<[u16; 3]> = Vec::new();

    // Left bookend: Z — randomize aux wires n..2n
    out.extend(rand_z_gates(n, bookend_size, rng));

    // Left bookend: M_p — W_i for i=0..n-1 with random permutation p on [2n]
    let mut perm: Vec<usize> = (0..2 * n).collect();
    perm.shuffle(rng);

    let mut pairs = vec![(0usize, 0usize); n];
    for i in 0..n {
        emit_w_i(i, n + i, perm[i], perm[n + i], &mut out);
        pairs[i] = (perm[i], perm[n + i]);
    }

    let mut state = GadgetState { n, pairs };
    let mut rg_pair_queue: VecDeque<(usize, usize)> = VecDeque::new();
    let mut rg3_queue: VecDeque<usize> = VecDeque::new();
    let mut rg_type_counter = 0usize;

    for (idx, &gate) in main.gates.iter().enumerate() {
        emit_gadget(&state, gate, &mut out);

        if (idx + 1) % rg_freq == 0 {
            let rg_type = rg_type_counter % 3;
            rg_type_counter += 1;
            match rg_type {
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
                    let total = 2 * n;
                    let r1 = loop {
                        let w = rng.random_range(0..total);
                        if w != s && w != p { break w; }
                    };
                    let r2 = loop {
                        let w = rng.random_range(0..total);
                        if w != s && w != p { break w; }
                    };
                    emit_rg3(&state, i, r1, r2, &mut out);
                }
            }
        }
    }

    // Right bookend: M_p^{-1} — W_i^{-1} for i=n-1..0
    // W_i^{-1} on (i, n+i, pairs[i].1, pairs[i].0) decodes x'_i onto wire i
    for i in (0..n).rev() {
        emit_w_i_inv(i, n + i, state.pairs[i].1, state.pairs[i].0, &mut out);
    }

    // Right bookend: Z — randomize aux wires
    out.extend(rand_z_gates(n, bookend_size, rng));

    CircuitSeq { gates: out }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_rng() -> impl Rng { rand::rng() }

    #[test]
    fn verify_w_i_mapping() {
        let mut out = Vec::new();
        emit_w_i(0, 1, 2, 3, &mut out);
        let circ = CircuitSeq { gates: out };
        for s in 0usize..(1 << 4) {
            let w0 = (s >> 0) & 1;
            let w1 = (s >> 1) & 1;
            let w2 = (s >> 2) & 1;
            let w3 = (s >> 3) & 1;
            let out_state = circ.evaluate(s);
            assert_eq!((out_state >> 0) & 1, w2,      "w0 wrong for s={}", s);
            assert_eq!((out_state >> 1) & 1, w3,      "w1 wrong for s={}", s);
            assert_eq!((out_state >> 2) & 1, w0 ^ w1, "w2 wrong for s={}", s);
            assert_eq!((out_state >> 3) & 1, w1,      "w3 wrong for s={}", s);
        }
    }

    #[test]
    fn verify_w_i_inv_is_inverse() {
        let mut out = Vec::new();
        emit_w_i(0, 1, 2, 3, &mut out);
        emit_w_i_inv(0, 1, 2, 3, &mut out);
        let circ = CircuitSeq { gates: out };
        for s in 0usize..(1 << 4) {
            assert_eq!(circ.evaluate(s), s, "W_i composed W_i_inv != identity for s={}", s);
        }
    }

    #[test]
    fn rg1_swaps_pairs() {
        let mut state = GadgetState { n: 4, pairs: vec![(0,4),(1,5),(2,6),(3,7)] };
        let mut out = Vec::new();
        emit_rg1(&mut state, 0, 1, &mut out);
        assert_eq!(state.pairs[0], (1, 5));
        assert_eq!(state.pairs[1], (0, 4));
    }

    #[test]
    fn rg2_swaps_pads() {
        let mut state = GadgetState { n: 4, pairs: vec![(0,4),(1,5),(2,6),(3,7)] };
        let mut out = Vec::new();
        emit_rg2(&mut state, 0, 1, &mut out);
        assert_eq!(state.pairs[0], (0, 5));
        assert_eq!(state.pairs[1], (1, 4));
    }

    #[test]
    fn verify_6gate_gadget_semantics() {
        // pairs: share_a=0 pad_a=3  share_b=1 pad_b=4  share_c=2 pad_c=5
        let state = GadgetState { n: 3, pairs: vec![(0,3),(1,4),(2,5)] };
        let mut out = Vec::new();
        emit_gadget(&state, [0, 1, 2], &mut out);
        let gadget_circ = CircuitSeq { gates: out };

        for s in 0usize..(1 << 6) {
            let x_a = (s >> 0) & 1;
            let x_b = (s >> 1) & 1;
            let x_c = (s >> 2) & 1;
            let r_a = (s >> 3) & 1;
            let r_b = (s >> 4) & 1;
            let r_c = (s >> 5) & 1;
            let s_a = x_a ^ r_a;
            let s_b = x_b ^ r_b;
            let s_c = x_c ^ r_c;

            let out_state = gadget_circ.evaluate(s);

            for wire in [1usize, 2, 3, 4, 5] {
                assert_eq!((s >> wire) & 1, (out_state >> wire) & 1,
                    "wire {} changed for input {:#08b}", wire, s);
            }
            let new_x_a = (out_state >> 0) & 1;
            let expected = 1 ^ s_a ^ (s_c & (1 ^ s_b));
            assert_eq!(new_x_a ^ r_a, expected,
                "gadget invariant failed: s={:#08b}", s);
        }
    }

    #[test]
    fn verify_rg1_semantics() {
        let mut state = GadgetState { n: 4, pairs: vec![(0,4),(1,5),(2,6),(3,7)] };
        let mut out = Vec::new();
        emit_rg1(&mut state, 0, 1, &mut out);
        let circ = CircuitSeq { gates: out };
        for s in 0usize..(1 << 8) {
            let w0 = (s >> 0) & 1; // share_i=wire 0
            let w1 = (s >> 4) & 1; // pad_i=wire 4
            let w2 = (s >> 5) & 1; // pad_j=wire 5
            let w3 = (s >> 1) & 1; // share_j=wire 1
            let s_i = w0 ^ w1;
            let s_j = w2 ^ w3;
            let out_state = circ.evaluate(s);
            let nw0 = (out_state >> 0) & 1;
            let nw1 = (out_state >> 4) & 1;
            let nw2 = (out_state >> 5) & 1;
            let nw3 = (out_state >> 1) & 1;
            assert_eq!(nw0 ^ nw1, s_j, "pair i should carry s_j after RG1, s={:#010b}", s);
            assert_eq!(nw2 ^ nw3, s_i, "pair j should carry s_i after RG1, s={:#010b}", s);
        }
    }

    #[test]
    fn verify_rg2_semantics() {
        let mut state = GadgetState { n: 4, pairs: vec![(0,4),(1,5),(2,6),(3,7)] };
        let mut out = Vec::new();
        emit_rg2(&mut state, 0, 1, &mut out);
        let circ = CircuitSeq { gates: out };
        assert_eq!(state.pairs[0], (0, 5));
        assert_eq!(state.pairs[1], (1, 4));
        for s in 0usize..(1 << 8) {
            let w0 = (s >> 0) & 1; // share_i=wire 0
            let w1 = (s >> 4) & 1; // pad_i=wire 4
            let w2 = (s >> 5) & 1; // pad_j=wire 5
            let w3 = (s >> 1) & 1; // share_j=wire 1
            let s_i = w0 ^ w1;
            let s_j = w2 ^ w3;
            let out_state = circ.evaluate(s);
            let nw0 = (out_state >> 0) & 1; // share_i unchanged
            let nw2 = (out_state >> 5) & 1; // new pad_i (was pad_j)
            let nw1 = (out_state >> 4) & 1; // new pad_j (was pad_i)
            let nw3 = (out_state >> 1) & 1; // share_j unchanged
            assert_eq!(nw0 ^ nw2, s_i,
                "new pair i (share_i, old_pad_j) should carry s_i, s={:#010b}", s);
            assert_eq!(nw1 ^ nw3, s_j,
                "new pair j (share_j, old_pad_i) should carry s_j, s={:#010b}", s);
        }
    }

    #[test]
    fn verify_rg3_semantics() {
        let state = GadgetState { n: 4, pairs: vec![(0,4),(1,5),(2,6),(3,7)] };
        let mut out = Vec::new();
        emit_rg3(&state, 0, 2, 3, &mut out);
        let circ = CircuitSeq { gates: out };
        for s in 0usize..(1 << 8) {
            let w0 = (s >> 0) & 1;
            let w1 = (s >> 4) & 1;
            let s_val = w0 ^ w1;
            let out_state = circ.evaluate(s);
            let nw0 = (out_state >> 0) & 1;
            let nw1 = (out_state >> 4) & 1;
            assert_eq!(nw0 ^ nw1, s_val, "RG3 must preserve virtual value, s={:#010b}", s);
        }
    }

    #[test]
    fn gadgetize_preserves_functionality_on_main_wires() {
        let n = 3;
        let main = CircuitSeq { gates: vec![[0,1,2],[1,2,0],[2,0,1],[0,2,1]] };
        let mut rng = rand::rng();
        let gadgetized = gadgetize(&main, n, 3, &mut rng);
        let mask = (1usize << n) - 1;
        for input in 0usize..(1 << n) {
            let expected = main.evaluate(input) & mask;
            let actual = gadgetized.evaluate(input) & mask;
            assert_eq!(actual, expected,
                "input {:#05b}: expected {:#05b}, got {:#05b}", input, expected, actual);
        }
    }

    #[test]
    fn all_wires_in_range() {
        let n = 4;
        let main = CircuitSeq { gates: vec![[0,1,2],[1,0,3],[2,3,0],[3,1,2]] };
        let mut rng = rand::rng();
        let result = gadgetize(&main, n, 3, &mut rng);
        let total = 2 * n;
        for gate in &result.gates {
            for &w in gate {
                assert!((w as usize) < total, "wire {} out of range 0..{}", w, total);
            }
        }
    }

    #[test]
    fn gadgetize_32wire_probably_equal() {
        let n = 32;
        let main = crate::random::random_data::random_circuit(n, 200);
        let mut rng = rand::rng();
        let gadgetized = gadgetize(&main, n, 3, &mut rng);
        main.probably_equal(&gadgetized, n, 10_000)
            .expect("gadgetized circuit changed functionality on first 32 wires");
    }

    #[test]
    #[ignore]
    fn gadgetize_8wire_degree_checkpoints() {
        use crate::circuit::circuit::poly_degree;
        let n = 8usize;
        let total = 2 * n;
        let main = crate::random::random_data::random_circuit(n, 100);
        let mut rng = rand::rng();
        let gadgetized = gadgetize(&main, n, 3, &mut rng);

        let g = &gadgetized.gates;
        let total_gates = g.len();
        println!("\nn={n}, total wires={total}, gadgetized gates={total_gates}");

        let checkpoints = [20, 40, 60, 80, 100, total_gates / 4, total_gates / 2, total_gates];
        for &cp in &checkpoints {
            let cp = cp.min(total_gates);
            let circ = CircuitSeq { gates: g[..cp].to_vec() };
            let polys = circ.to_polynomial(total, 0, cp);
            let comp_degrees: Vec<u32> = (0..n).map(|w| poly_degree(&polys[w])).collect();
            let aux_degrees: Vec<u32> = (n..total).map(|w| poly_degree(&polys[w])).collect();
            let min_comp = *comp_degrees.iter().min().unwrap();
            println!("gate {:4}: comp min={min_comp} {:?}", cp, comp_degrees);
            println!("          aux       {:?}", aux_degrees);
        }

        let final_polys = CircuitSeq { gates: g.to_vec() }.to_polynomial(total, 0, total_gates);
        let min_comp_deg = (0..n).map(|w| poly_degree(&final_polys[w])).min().unwrap();
        assert!(min_comp_deg >= 6,
            "expected all computation wires to have degree >= 6, got min={min_comp_deg}");
    }
}
