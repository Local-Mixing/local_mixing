use std::collections::VecDeque;
use rand::{Rng, prelude::SliceRandom};
use crate::circuit::circuit::CircuitSeq;

/// 6-gate homomorphic gadget (local wire indices 0–6, slots 1–2 unused).
///
/// Local wire mapping (swapped convention: c in slot 3, b in slot 5):
///   0 = x_a    active main wire
///   3 = x_c    negative control (c in gate [a,b,c])
///   4 = r_c    aux paired with x_c
///   5 = x_b    positive control (b in gate [a,b,c])
///   6 = r_b    aux paired with x_b
///
/// Gate semantics: [a,b,c] flips wire a when wire[b]=1 OR wire[c]=0.
/// After the gadget: new_x_a XOR r_a = 1^s_a^(s_c&!s_b) = 1^s_a^s_c^(s_b&s_c)
///   = s_a ^ (s_b | !s_c) as required by gate [a,b,c].
/// Wire r_c is temporarily modified but restored. Wires 3,5,6 and r_a unchanged.
const GADGET: [[u8; 3]; 6] = [
    [4,5,6],[0,4,6],[0,5,4],[4,5,6],[0,6,3],[0,3,5],
];

/// Manages pairing between computation wires and auxiliary wires.
///
/// Wire layout for an n-wire original circuit (total = 2n+1):
///   0    .. n-1    computation wires
///   n    .. 2n     auxiliary wires  (n+1 of them; n paired, 1 free)
///
/// The scheduler tracks:
///   - which aux wire is paired to each computation wire
///   - which single aux wire is currently free (cycles via carrier switches)
///   - a shuffled queue for which computation wire to switch next,
///     reshuffled each time all n wires have been visited once
pub struct GadgetScheduler {
    n: usize,
    pairing: Vec<usize>,          // pairing[i] = aux wire currently paired with computation wire i
    free: usize,                  // the one aux wire not currently paired
    switch_queue: VecDeque<usize>, // shuffled order for carrier switches
}

impl GadgetScheduler {
    /// Create scheduler with a random initial pairing.
    /// Aux wires are n..=2n (n+1 total). One is chosen at random to be free.
    pub fn new_random(n: usize, rng: &mut impl Rng) -> Self {
        assert!(n >= 2, "need at least 2 wires");
        let mut aux: Vec<usize> = (n..=2 * n).collect(); // n+1 aux wires
        aux.shuffle(rng);
        let free = aux[n]; // last after shuffle = free wire
        let pairing = aux[..n].to_vec();
        GadgetScheduler {
            n,
            pairing,
            free,
            switch_queue: VecDeque::new(),
        }
    }

    /// Total wires in the gadgetized circuit (always 2n+1).
    pub fn total_wires(&self) -> usize {
        2 * self.n + 1
    }

    /// Aux wire currently paired with computation wire i.
    pub fn current_aux(&self, i: usize) -> usize {
        self.pairing[i]
    }

    /// The one currently-free aux wire.
    pub fn free_wire(&self) -> usize {
        self.free
    }

    /// Return the next computation wire to carrier-switch, using a shuffled
    /// schedule. Reshuffles and restarts when all n wires have been visited.
    pub fn next_switch_wire(&mut self, rng: &mut impl Rng) -> usize {
        if self.switch_queue.is_empty() {
            let mut order: Vec<usize> = (0..self.n).collect();
            order.shuffle(rng);
            self.switch_queue.extend(order);
        }
        self.switch_queue.pop_front().unwrap()
    }

    /// Emit the 3-gate carrier switch for computation wire `main_wire`.
    ///   Gate 1: [w_j, w_k, w_l]   — begin secret-share transfer
    ///   Gate 2: [w_j, w_l, w_k]   — complete transfer (secret now on w_j XOR w_l)
    ///   Gate 3: [w_k, last_active, w_l] — raise degree of freed aux
    ///
    /// After: pairing[main_wire] = w_l (new aux), w_k becomes free.
    pub fn carrier_switch(&mut self, main_wire: usize, last_active: usize) -> [[u8; 3]; 3] {
        let wj = main_wire;
        let wk = self.pairing[main_wire];
        let wl = self.free;
        self.pairing[main_wire] = wl;
        self.free = wk;
        [
            [wj as u8, wk as u8, wl as u8],
            [wj as u8, wl as u8, wk as u8],
            [wk as u8, last_active as u8, wl as u8],
        ]
    }
}

/// Emit the 6-gate gadget for main gate `gate` into `out`.
/// The pairing is read-only between carrier switches.
pub fn emit_gadget(sched: &GadgetScheduler, gate: [u8; 3], out: &mut Vec<[u8; 3]>) {
    let a = gate[0] as usize;
    let b = gate[1] as usize;
    let c = gate[2] as usize;

    let r_b = sched.current_aux(b);
    let r_c = sched.current_aux(c);

    // GADGET computes 1^s_a^(s_{slot3}&!s_{slot5}).
    // Gate [a,b,c] needs 1^s_a^s_c^(s_b&s_c); c(neg) in slot3, b(pos) in slot5 gives ✓.
    // Slot map: 0=x_a, 3=x_c, 4=r_c, 5=x_b, 6=r_b (slots 1,2 unused).
    let map: [u8; 7] = [
        a as u8, 0, 0,
        c as u8, r_c as u8,
        b as u8, r_b as u8,
    ];
    for &[ga, gb, gc] in &GADGET {
        out.push([map[ga as usize], map[gb as usize], map[gc as usize]]);
    }
}

// CNOT(a ← b) with helper h in the rule-57 gate basis.
// Works for any initial value of h; h is always restored to its original value.
fn cnot_gates(a: u8, b: u8, h: u8) -> [[u8; 3]; 6] {
    [[b,a,h], [h,b,a], [a,b,h], [b,h,a], [a,h,b], [h,a,b]]
}

/// Gadgetize a circuit:
///   1. Add n+1 aux wires → total 2n+1 wires.
///   2. Run 2 full degree-chain cycles on all n+1 aux wires.
///   3. Random initial pairing (n paired, 1 free).
///   4. Begin-XOR masking.
///   5. For each original gate: 6-gate gadget; every 2 gadgets → 3-gate carrier
///      switch, visiting all n computation wires in a shuffled order before
///      reshuffling and repeating.
///   6. End-XOR unmasking.
pub fn gadgetize(main: &CircuitSeq, n: usize, rng: &mut impl Rng) -> CircuitSeq {
    let num_aux = n + 1;           // n+1 aux wires
    let m = main.gates.len();
    let switches = m / 2;
    let mut out = Vec::with_capacity(2 * num_aux + 6 * n + m * 6 + switches * 3 + 6 * n);

    // Step 1: 2 full degree-chain cycles on all n+1 aux wires.
    // Aux wires are indices n..2n (n+1 wires); we don't know the final pairing
    // yet, so we run the chain on aux wire slots 0..num_aux, mapped to n..2n.
    for k in 1..=(2 * num_aux) {
        let active = (n + k % num_aux) as u8;
        let ctrl1  = (n + (k + 1) % num_aux) as u8;
        let ctrl2  = (n + (k + num_aux - 1) % num_aux) as u8;
        out.push([active, ctrl1, ctrl2]);
    }

    // Step 2: Random initial pairing.
    let mut sched = GadgetScheduler::new_random(n, rng);

    // Step 3: Begin-XOR masking — main_i ^= paired_aux_i.
    // Aux wires start at 0, so this is a no-op functionally but symmetric with end-XOR.
    let h = sched.free_wire() as u8;
    for i in 0..n {
        let aux_i = sched.current_aux(i) as u8;
        for &g in &cnot_gates(i as u8, aux_i, h) {
            out.push(g);
        }
    }

    // Step 4: Gadgets interleaved with carrier switches every 2 gates.
    for (i, &gate) in main.gates.iter().enumerate() {
        emit_gadget(&sched, gate, &mut out);

        if (i + 1) % 2 == 0 {
            let wire_to_switch = sched.next_switch_wire(rng);
            let switch_gates = sched.carrier_switch(wire_to_switch, gate[0] as usize);
            for g in switch_gates {
                out.push(g);
            }
        }
    }

    // Step 5: End-XOR unmasking — main_i ^= current_paired_aux_i.
    let h = sched.free_wire() as u8;
    for i in 0..n {
        let aux_i = sched.current_aux(i) as u8;
        for &g in &cnot_gates(i as u8, aux_i, h) {
            out.push(g);
        }
    }

    CircuitSeq { gates: out }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_rng() -> impl Rng {
        rand::rng()
    }

    #[test]
    fn scheduler_initial_state() {
        let mut rng = test_rng();
        let sched = GadgetScheduler::new_random(3, &mut rng);
        assert_eq!(sched.total_wires(), 7); // 2*3+1
        // Pairing covers 3 of wires 3..=6; free is the remaining one
        let paired: std::collections::HashSet<usize> = (0..3).map(|i| sched.current_aux(i)).collect();
        assert_eq!(paired.len(), 3);
        assert!(!paired.contains(&sched.free_wire()));
        let all_aux: std::collections::HashSet<usize> = (3..=6).collect();
        assert_eq!(paired.union(&std::collections::HashSet::from([sched.free_wire()])).count(), 4);
        assert!(all_aux.contains(&sched.free_wire()));
    }

    #[test]
    fn carrier_switch_updates_pairing() {
        let mut rng = test_rng();
        let mut sched = GadgetScheduler::new_random(3, &mut rng);
        let orig_free = sched.free_wire();
        let wire0_orig_aux = sched.current_aux(0);

        let gates = sched.carrier_switch(0, 0);

        // Gates: [0, old_aux, free], [0, free, old_aux], [old_aux, 0, free]
        assert_eq!(gates[0], [0, wire0_orig_aux as u8, orig_free as u8]);
        assert_eq!(gates[1], [0, orig_free as u8, wire0_orig_aux as u8]);
        assert_eq!(gates[2][0], wire0_orig_aux as u8); // freed wire is active in gate 3
        assert_eq!(gates[2][1], 0);                    // last_active = 0

        // After: wire 0 paired to orig_free; wire0_orig_aux is now free
        assert_eq!(sched.current_aux(0), orig_free);
        assert_eq!(sched.free_wire(), wire0_orig_aux);
    }

    #[test]
    fn switch_queue_covers_all_wires() {
        let mut rng = test_rng();
        let mut sched = GadgetScheduler::new_random(4, &mut rng);
        let mut seen = vec![0usize; 4];
        // Pull 2 full cycles; each should visit every wire exactly once
        for _ in 0..8 {
            let w = sched.next_switch_wire(&mut rng);
            seen[w] += 1;
        }
        assert!(seen.iter().all(|&c| c == 2), "each wire should appear exactly twice: {:?}", seen);
    }

    #[test]
    fn verify_6gate_gadget_semantics() {
        let rng = test_rng();
        // Fixed pairing for determinism: wire i paired to n+i, free = 2n
        let n = 3;
        let sched = GadgetScheduler {
            n,
            pairing: vec![3, 4, 5],
            free: 6,
            switch_queue: VecDeque::new(),
        };
        let mut out = Vec::new();
        emit_gadget(&sched, [0, 1, 2], &mut out);
        let gadget_circ = CircuitSeq { gates: out };

        // 7 wires: 0=x_a,1=x_b,2=x_c, 3=r_a,4=r_b,5=r_c, 6=free
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

            // Unchanged: r_a(3), x_b(1), x_c(2), r_b(4)
            for wire in [1usize, 2, 3, 4] {
                assert_eq!((s >> wire) & 1, (out_state >> wire) & 1,
                    "wire {} changed for input {:#09b}", wire, s);
            }
            // r_c (wire 5) is temporarily modified but must be restored
            assert_eq!((s >> 5) & 1, (out_state >> 5) & 1,
                "r_c changed for input {:#09b}", s);

            let new_x_a = (out_state >> 0) & 1;
            // g57 [a,b,c]: new_s_a = 1^s_a^(s_c&!s_b)
            let expected = 1 ^ s_a ^ (s_c & (1 ^ s_b));
            assert_eq!(new_x_a ^ r_a, expected,
                "invariant failed: s={:#09b} s_a={} s_b={} s_c={} new_x_a={} r_a={} expected={}",
                s, s_a, s_b, s_c, new_x_a, r_a, expected);
        }
        println!("6-gate gadget semantics verified for all 128 inputs");
        let _ = rng; // suppress unused warning
    }

    #[test]
    fn gadgetize_preserves_functionality_on_main_wires() {
        let n = 3;
        let main = CircuitSeq { gates: vec![[0,1,2],[1,2,0],[2,0,1],[0,2,1]] };
        let mut rng = rand::rng();
        let gadgetized = gadgetize(&main, n, &mut rng);

        let mask = (1usize << n) - 1;
        for input in 0usize..(1 << n) {
            let expected = main.evaluate(input) & mask;
            let actual   = gadgetized.evaluate(input) & mask;
            assert_eq!(actual, expected,
                "input {:#05b}: expected main wires {:#05b}, got {:#05b}",
                input, expected, actual);
        }
    }

    #[test]
    fn all_wires_in_range() {
        let n = 4;
        let main = CircuitSeq { gates: vec![[0,1,2],[1,0,3],[2,3,0],[3,1,2]] };
        let mut rng = rand::rng();
        let result = gadgetize(&main, n, &mut rng);
        let total = 2 * n + 1;
        for gate in &result.gates {
            for &w in gate {
                assert!((w as usize) < total, "wire {} out of range 0..{}", w, total);
            }
        }
    }

    #[test]
    fn gadgetize_8wire_degree_checkpoints() {
        use crate::circuit::circuit::poly_degree;
        let n = 8usize;
        let total = 2 * n + 1; // 17 wires
        let main = crate::random::random_data::random_circuit(n, 100);
        let mut rng = rand::rng();
        let gadgetized = gadgetize(&main, n, &mut rng);

        let g = &gadgetized.gates;
        let total_gates = g.len();
        println!("\nn={n}, total wires={total}, gadgetized gates={total_gates}");

        // Check degrees at several points through the gadgetized circuit
        let checkpoints = [20, 40, 60, 80, 100, total_gates / 4, total_gates / 2, total_gates];
        for &cp in &checkpoints {
            let cp = cp.min(total_gates);
            let circ = CircuitSeq { gates: g[..cp].to_vec() };
            let polys = circ.to_polynomial(total, 0, cp);
            let comp_degrees: Vec<u32> = (0..n).map(|w| poly_degree(&polys[w])).collect();
            let aux_degrees:  Vec<u32> = (n..total).map(|w| poly_degree(&polys[w])).collect();
            let min_comp = *comp_degrees.iter().min().unwrap();
            println!("gate {:4}: comp min={min_comp} {:?}", cp, comp_degrees);
            println!("          aux       {:?}", aux_degrees);
        }

        // At the end, computation wires should all have high degree
        let final_circ = CircuitSeq { gates: g.to_vec() };
        let final_polys = final_circ.to_polynomial(total, 0, total_gates);
        let min_comp_deg = (0..n).map(|w| poly_degree(&final_polys[w])).min().unwrap();
        assert!(min_comp_deg >= 6,
            "expected all computation wires to have degree ≥ 6, got min={min_comp_deg}");
    }

    #[test]
    fn gadgetize_32wire_probably_equal() {
        let n = 32;
        let main = crate::random::random_data::random_circuit(n, 200);
        let mut rng = rand::rng();
        let gadgetized = gadgetize(&main, n, &mut rng);
        main.probably_equal(&gadgetized, n, 10_000)
            .expect("gadgetized circuit changed functionality on first 32 wires");
        println!("32-wire gadgetize probably_equal: PASS");
    }

    #[test]
    fn degree_before_after() {
        use crate::circuit::circuit::poly_degree;

        let n = 5usize;
        let main = crate::random::random_data::random_circuit(n, 20);
        let mut rng = rand::rng();

        println!("\n=== Algebraic Degree Before/After Gadgetize (n={}) ===", n);
        println!("\nBEFORE ({} gates, {} wires):", main.gates.len(), n);
        let polys_before = main.to_polynomial(n, 0, main.gates.len());
        for (wire, poly) in polys_before.iter().enumerate() {
            println!("  wire {:2}: degree {}", wire, poly_degree(poly));
        }

        let gadgetized = gadgetize(&main, n, &mut rng);
        let total = 2 * n + 1;

        let print_snapshot = |gates: &[[u8; 3]], label: &str| {
            let circ = CircuitSeq { gates: gates.to_vec() };
            let polys = circ.to_polynomial(total, 0, gates.len());
            println!("\n{} ({} gates):", label, gates.len());
            for wire in 0..total {
                let cat = if wire < n { "comp" } else { "aux " };
                println!("  wire {:2} [{}]: degree {}", wire, cat, poly_degree(&polys[wire]));
            }
        };

        let g = &gadgetized.gates;
        let total_gates = g.len();
        print_snapshot(&g[..50.min(total_gates)],          "SNAPSHOT gate 50");
        print_snapshot(&g[..total_gates / 2], "SNAPSHOT half");
        print_snapshot(g, "AFTER");
    }
}
