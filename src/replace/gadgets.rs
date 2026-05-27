use std::collections::VecDeque;
use rand::{Rng, prelude::SliceRandom};
use crate::circuit::circuit::CircuitSeq;

/// 6-gate homomorphic gadget for secret-shared g57.
/// Gate [a, pos_ctrl, neg_ctrl] flips wire a UNLESS neg_ctrl=1 AND NOT pos_ctrl
/// ("flip a UNLESS b AND NOT c" where b=gate[2]=neg_ctrl, c=gate[1]=pos_ctrl).
/// Equivalently: flips when pos_ctrl=1 OR neg_ctrl=0.
/// Local wire mapping: 0=x_a, 3=x_c, 4=r_c, 5=x_b, 6=r_b (slots 1,2 unused).
const GADGET: [[u8; 3]; 6] = [
    [4,5,6],[0,4,6],[0,5,4],[4,5,6],[0,6,3],[0,3,5],
];

/// RG1: 6-gate virtual value swap between two pairs.
/// Wire map: 0=comp_i, 1=aux_i, 2=aux_j, 3=comp_j.
/// After: (w0'^w1')=s_j, (w2'^w3')=s_i — pairings unchanged, values swapped.
const RG1: [[u8; 3]; 6] = [
    [1,2,3],[0,3,2],[3,1,0],[2,0,1],[0,3,2],[1,2,3],
];

/// RG2: 6-gate re-pairing while preserving virtual values.
/// Wire map: 0=comp_i, 1=aux_i, 2=aux_j, 3=comp_j.
/// After: (w0'^w2')=s_j, (w1'^w3')=s_i — comp_i now paired with aux_j.
const RG2: [[u8; 3]; 6] = [
    [0,2,3],[1,0,2],[2,0,3],[2,3,0],[1,3,2],[3,2,0],
];

/// Manages 2n-wire secret-sharing state.
/// Wire layout: comp wires 0..n, aux wires n..2n.
pub struct GadgetScheduler {
    pub n: usize,
    pub pairing: Vec<usize>,                     // pairing[i] = aux wire for comp wire i
    pub rg_pair_queue: VecDeque<(usize, usize)>,  // shuffled pairs of comp indices
    pub rg3_queue: VecDeque<usize>,               // shuffled comp indices for RG3
    pub rg_type_counter: usize,                   // cycles 0=RG1, 1=RG2, 2=RG3
}

impl GadgetScheduler {
    pub fn new_random(n: usize, rng: &mut impl Rng) -> Self {
        assert!(n >= 2);
        let mut aux: Vec<usize> = (n..2 * n).collect();
        aux.shuffle(rng);
        GadgetScheduler {
            n,
            pairing: aux,
            rg_pair_queue: VecDeque::new(),
            rg3_queue: VecDeque::new(),
            rg_type_counter: 0,
        }
    }

    pub fn total_wires(&self) -> usize { 2 * self.n }

    pub fn current_aux(&self, i: usize) -> usize { self.pairing[i] }

    pub fn next_rg_type(&mut self) -> usize {
        let t = self.rg_type_counter % 3;
        self.rg_type_counter += 1;
        t
    }

    pub fn apply_rg2(&mut self, i: usize, j: usize) {
        self.pairing.swap(i, j);
    }

    pub fn next_rg_pair(&mut self, rng: &mut impl Rng) -> (usize, usize) {
        if self.rg_pair_queue.is_empty() {
            let mut pairs: Vec<(usize, usize)> = (0..self.n)
                .flat_map(|i| (i + 1..self.n).map(move |j| (i, j)))
                .collect();
            pairs.shuffle(rng);
            self.rg_pair_queue.extend(pairs);
        }
        self.rg_pair_queue.pop_front().unwrap()
    }

    pub fn next_rg3_wire(&mut self, rng: &mut impl Rng) -> usize {
        if self.rg3_queue.is_empty() {
            let mut wires: Vec<usize> = (0..self.n).collect();
            wires.shuffle(rng);
            self.rg3_queue.extend(wires);
        }
        self.rg3_queue.pop_front().unwrap()
    }
}

// CNOT(a <- b) with helper h; works for any h value, h is always restored.
fn cnot_gates(a: u8, b: u8, h: u8) -> [[u8; 3]; 6] {
    [[b,a,h],[h,b,a],[a,b,h],[b,h,a],[a,h,b],[h,a,b]]
}

// m balanced random gates on aux wires (n..2n), controls from all 2n wires.
fn rand_z_gates(n: usize, m: usize, rng: &mut impl Rng) -> Vec<[u8; 3]> {
    let total = 2 * n;
    let mut gates = Vec::with_capacity(m);
    let mut round: Vec<usize> = (n..2 * n).collect();
    let mut pos = round.len();
    for _ in 0..m {
        if pos >= round.len() {
            round.shuffle(rng);
            pos = 0;
        }
        let active = round[pos] as u8;
        pos += 1;
        let ctrl1 = loop {
            let w = rng.random_range(0..total) as u8;
            if w != active { break w; }
        };
        let ctrl2 = loop {
            let w = rng.random_range(0..total) as u8;
            if w != active { break w; }
        };
        gates.push([active, ctrl1, ctrl2]);
    }
    gates
}

// m random aux gates + XOR masking x_i ^= z_{perm[i]}.
// Returns (gates, pairing) where pairing[i] = aux wire paired with comp wire i.
fn bookend(n: usize, m: usize, rng: &mut impl Rng) -> (Vec<[u8; 3]>, Vec<usize>) {
    let mut gates = rand_z_gates(n, m, rng);
    let mut perm: Vec<usize> = (n..2 * n).collect();
    perm.shuffle(rng);
    for i in 0..n {
        let z_j = perm[i] as u8;
        let helper = perm[(i + 1) % n] as u8;
        for &g in &cnot_gates(i as u8, z_j, helper) {
            gates.push(g);
        }
    }
    (gates, perm)
}

pub fn emit_gadget(sched: &GadgetScheduler, gate: [u8; 3], out: &mut Vec<[u8; 3]>) {
    let a = gate[0] as usize;
    let b = gate[1] as usize;
    let c = gate[2] as usize;
    let r_b = sched.current_aux(b);
    let r_c = sched.current_aux(c);
    let map: [u8; 7] = [a as u8, 0, 0, c as u8, r_c as u8, b as u8, r_b as u8];
    for &[ga, gb, gc] in &GADGET {
        out.push([map[ga as usize], map[gb as usize], map[gc as usize]]);
    }
}

pub fn emit_rg1(sched: &GadgetScheduler, i: usize, j: usize, out: &mut Vec<[u8; 3]>) {
    let map = [i as u8, sched.pairing[i] as u8, sched.pairing[j] as u8, j as u8];
    for &[a, b, c] in &RG1 {
        out.push([map[a as usize], map[b as usize], map[c as usize]]);
    }
}

pub fn emit_rg2(sched: &mut GadgetScheduler, i: usize, j: usize, out: &mut Vec<[u8; 3]>) {
    let map = [i as u8, sched.pairing[i] as u8, sched.pairing[j] as u8, j as u8];
    for &[a, b, c] in &RG2 {
        out.push([map[a as usize], map[b as usize], map[c as usize]]);
    }
    sched.apply_rg2(i, j);
}

pub fn emit_rg3(sched: &GadgetScheduler, i: usize, r1: usize, r2: usize, out: &mut Vec<[u8; 3]>) {
    out.push([i as u8, r1 as u8, r2 as u8]);
    out.push([sched.pairing[i] as u8, r1 as u8, r2 as u8]);
}


/// Gadgetize a circuit: add n aux wires (total 2n), secret-share each wire,
/// process original gates as 6-gate SG gadgets interleaved with RG gadgets.
pub fn gadgetize(main: &CircuitSeq, n: usize, rg_freq: usize, rng: &mut impl Rng) -> CircuitSeq {
    const BOOKEND: usize = 640;
    let mut out = Vec::new();

    let (begin_gates, init_pairing) = bookend(n, BOOKEND, rng);
    out.extend(begin_gates);

    let mut sched = GadgetScheduler {
        n,
        pairing: init_pairing,
        rg_pair_queue: VecDeque::new(),
        rg3_queue: VecDeque::new(),
        rg_type_counter: 0,
    };

    // phys[v] = physical comp wire currently holding virtual wire v's value; starts as identity.
    let mut phys: Vec<usize> = (0..n).collect();

    for (idx, &gate) in main.gates.iter().enumerate() {
        let pa = phys[gate[0] as usize];
        let pb = phys[gate[1] as usize];
        let pc = phys[gate[2] as usize];
        emit_gadget(&sched, [pa as u8, pb as u8, pc as u8], &mut out);

        if (idx + 1) % rg_freq == 0 {
            let rg_type = sched.next_rg_type();
            match rg_type {
                0 => {
                    let (i, j) = sched.next_rg_pair(rng);
                    emit_rg1(&sched, i, j, &mut out);
                    let vi = (0..n).find(|&v| phys[v] == i).unwrap();
                    let vj = (0..n).find(|&v| phys[v] == j).unwrap();
                    phys[vi] = j;
                    phys[vj] = i;
                }
                1 => {
                    let (i, j) = sched.next_rg_pair(rng);
                    emit_rg2(&mut sched, i, j, &mut out);
                    let vi = (0..n).find(|&v| phys[v] == i).unwrap();
                    let vj = (0..n).find(|&v| phys[v] == j).unwrap();
                    phys[vi] = j;
                    phys[vj] = i;
                }
                _ => {
                    let i = sched.next_rg3_wire(rng);
                    let total = 2 * sched.n;
                    let comp_i = i;
                    let aux_i = sched.pairing[i];
                    let r1 = loop {
                        let w = rng.random_range(0..total);
                        if w != comp_i && w != aux_i { break w; }
                    };
                    let r2 = loop {
                        let w = rng.random_range(0..total);
                        if w != comp_i && w != aux_i { break w; }
                    };
                    emit_rg3(&sched, i, r1, r2, &mut out);
                }
            }
        }
    }

    // Restore identity permutation: bring each virtual wire v back to physical position v.
    // Each iteration does one RG1 swap and places exactly one virtual wire correctly.
    for v in 0..n {
        if phys[v] != v {
            let p = phys[v];
            emit_rg1(&sched, v, p, &mut out);
            let w = (0..n).find(|&x| phys[x] == v).unwrap();
            phys[w] = p;
            phys[v] = v;
        }
    }

    // End XOR unmasking: x_i ^= z_{pairing[i]}.
    for i in 0..n {
        let aux_i = sched.pairing[i] as u8;
        let helper = sched.pairing[(i + 1) % n] as u8;
        for &g in &cnot_gates(i as u8, aux_i, helper) {
            out.push(g);
        }
    }

    out.extend(rand_z_gates(n, BOOKEND, rng));

    CircuitSeq { gates: out }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_rng() -> impl Rng { rand::rng() }

    #[test]
    fn scheduler_initial_state() {
        let mut rng = test_rng();
        let sched = GadgetScheduler::new_random(4, &mut rng);
        assert_eq!(sched.total_wires(), 8);
        let paired: std::collections::HashSet<usize> = (0..4).map(|i| sched.current_aux(i)).collect();
        assert_eq!(paired.len(), 4);
        assert!(paired.iter().all(|&w| w >= 4 && w < 8));
    }

    #[test]
    fn rg_pair_queue_covers_all_pairs() {
        let mut rng = test_rng();
        let mut sched = GadgetScheduler::new_random(4, &mut rng);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..6 {
            seen.insert(sched.next_rg_pair(&mut rng));
        }
        for i in 0..4usize {
            for j in i+1..4 {
                assert!(seen.contains(&(i, j)), "missing pair ({},{})", i, j);
            }
        }
    }

    #[test]
    fn rg2_updates_pairing() {
        let mut rng = test_rng();
        let mut sched = GadgetScheduler::new_random(4, &mut rng);
        let orig_aux_0 = sched.current_aux(0);
        let orig_aux_1 = sched.current_aux(1);
        let mut out = Vec::new();
        emit_rg2(&mut sched, 0, 1, &mut out);
        assert_eq!(sched.current_aux(0), orig_aux_1);
        assert_eq!(sched.current_aux(1), orig_aux_0);
    }

    #[test]
    fn verify_6gate_gadget_semantics() {
        let n = 3;
        let sched = GadgetScheduler {
            n, pairing: vec![3, 4, 5],
            rg_pair_queue: VecDeque::new(), rg3_queue: VecDeque::new(), rg_type_counter: 0,
        };
        let mut out = Vec::new();
        emit_gadget(&sched, [0, 1, 2], &mut out);
        let gadget_circ = CircuitSeq { gates: out };

        for s in 0usize..(1 << 7) {
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
                    "wire {} changed for input {:#09b}", wire, s);
            }
            let new_x_a = (out_state >> 0) & 1;
            let expected = 1 ^ s_a ^ (s_c & (1 ^ s_b));
            assert_eq!(new_x_a ^ r_a, expected,
                "gadget invariant failed: s={:#09b}", s);
        }
    }

    #[test]
    fn verify_rg1_semantics() {
        let n = 4;
        let sched = GadgetScheduler {
            n, pairing: vec![4, 5, 6, 7],
            rg_pair_queue: VecDeque::new(), rg3_queue: VecDeque::new(), rg_type_counter: 0,
        };
        let mut out = Vec::new();
        emit_rg1(&sched, 0, 1, &mut out);
        let circ = CircuitSeq { gates: out };
        for s in 0usize..(1 << 8) {
            let w0 = (s >> 0) & 1; // comp_0
            let w1 = (s >> 4) & 1; // aux_0
            let w2 = (s >> 5) & 1; // aux_1
            let w3 = (s >> 1) & 1; // comp_1
            let s1 = w0 ^ w1;
            let s2 = w2 ^ w3;
            let out_state = circ.evaluate(s);
            let nw0 = (out_state >> 0) & 1;
            let nw1 = (out_state >> 4) & 1;
            let nw2 = (out_state >> 5) & 1;
            let nw3 = (out_state >> 1) & 1;
            assert_eq!(nw0 ^ nw1, s2,
                "pair (comp_0,aux_0) should carry s2 after RG1, s={:#010b}", s);
            assert_eq!(nw2 ^ nw3, s1,
                "pair (comp_1,aux_1) should carry s1 after RG1, s={:#010b}", s);
        }
    }

    #[test]
    fn verify_rg2_semantics() {
        let n = 4;
        let mut sched = GadgetScheduler {
            n, pairing: vec![4, 5, 6, 7],
            rg_pair_queue: VecDeque::new(), rg3_queue: VecDeque::new(), rg_type_counter: 0,
        };
        let mut out = Vec::new();
        emit_rg2(&mut sched, 0, 1, &mut out);
        let circ = CircuitSeq { gates: out };
        assert_eq!(sched.current_aux(0), 5, "comp_0 should now be paired with aux_1");
        assert_eq!(sched.current_aux(1), 4, "comp_1 should now be paired with aux_0");
        for s in 0usize..(1 << 8) {
            let w0 = (s >> 0) & 1; // comp_0
            let w1 = (s >> 4) & 1; // aux_0
            let w2 = (s >> 5) & 1; // aux_1
            let w3 = (s >> 1) & 1; // comp_1
            let s1 = w0 ^ w1;
            let s2 = w2 ^ w3;
            let out_state = circ.evaluate(s);
            let nw0 = (out_state >> 0) & 1;
            let nw2 = (out_state >> 5) & 1;
            let nw1 = (out_state >> 4) & 1;
            let nw3 = (out_state >> 1) & 1;
            assert_eq!(nw0 ^ nw2, s2,
                "new pair (comp_0,aux_1) should carry s2, s={:#010b}", s);
            assert_eq!(nw1 ^ nw3, s1,
                "new pair (comp_1,aux_0) should carry s1, s={:#010b}", s);
        }
    }

    #[test]
    fn verify_rg3_semantics() {
        let n = 4;
        let sched = GadgetScheduler {
            n, pairing: vec![4, 5, 6, 7],
            rg_pair_queue: VecDeque::new(), rg3_queue: VecDeque::new(), rg_type_counter: 0,
        };
        let mut out = Vec::new();
        emit_rg3(&sched, 0, 2, 3, &mut out);
        let circ = CircuitSeq { gates: out };
        for s in 0usize..(1 << 8) {
            let w0 = (s >> 0) & 1;
            let w1 = (s >> 4) & 1;
            let s1 = w0 ^ w1;
            let out_state = circ.evaluate(s);
            let nw0 = (out_state >> 0) & 1;
            let nw1 = (out_state >> 4) & 1;
            assert_eq!(nw0 ^ nw1, s1,
                "RG3 must preserve virtual value, s={:#010b}", s);
        }
    }

    #[test]
    fn gadgetize_preserves_functionality_on_main_wires() {
        let n = 3;
        let main = CircuitSeq { gates: vec![[0,1,2],[1,2,0],[2,0,1],[0,2,1]] };
        let mut rng = rand::rng();
        let gadgetized = gadgetize(&main, n, 3,&mut rng);
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
        let gadgetized = gadgetize(&main, n, 3,&mut rng);
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
        let gadgetized = gadgetize(&main, n, 3,&mut rng);

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
