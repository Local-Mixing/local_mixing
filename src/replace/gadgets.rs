use std::collections::VecDeque;
use crate::circuit::circuit::CircuitSeq;

/// 9-gate homomorphic gadget (local wire indices 0–6).
///
/// Local wire mapping:
///   0 = x_a    active main wire
///   1 = r1     current aux paired with x_a (freed after gadget)
///   2 = r2     incoming free aux wire      (becomes new pair for x_a)
///   3 = x_b    control wire 1
///   4 = r_b    aux currently paired with x_b (read-only)
///   5 = x_c    control wire 2
///   6 = r_c    aux currently paired with x_c (read-only)
///
/// Gate semantics: [a,b,c] flips wire a when wire[b]=1 OR wire[c]=0.
/// After the gadget: new_secret_a = 1 ^ s_a ^ (s_b & !s_c)
/// where s_i = main_i ^ aux_i is the unmasked secret value.
const GADGET: [[u8; 3]; 9] = [
    [0,3,5],[0,3,6],[0,4,5],[0,4,6],
    [0,1,2],[0,2,1],[0,1,1],
    [0,3,4],[0,4,3],
];

/// Manages which aux wire is paired to each main wire, and which are free.
///
/// Wire layout for an n-wire original circuit (total = 3n, constant):
///   0    .. n-1    main wires
///   n    .. 2n-1   paired aux      (pairing[i] = n+i initially)
///   2n   .. 3n-1   free pool       (n wires, giving max degree n in the chain)
///
/// The cycle for an aux wire:
///   free pool (front) → r2 in gadget → paired to a main wire
///   → r1 in next gadget for that wire → free pool (back)
///   → processed by degree chain → ... → free pool (front) again
///
/// With n free wires, each freed r1 waits ~n gadgets before becoming r2,
/// accumulating n chain steps and reaching degree n. The chain may ONLY
/// target free wires; paired wires must not be modified outside their gadget.
pub struct GadgetScheduler {
    n: usize,
    pairing: Vec<usize>,         // pairing[i] = aux wire currently paired with main wire i
    free_pool: VecDeque<usize>,  // front = next r2; back = most recently freed
}

impl GadgetScheduler {
    /// Create a scheduler for an n-wire circuit.
    /// Total wires: 3n (n main + n paired aux + n free aux).
    pub fn new(n: usize) -> Self {
        assert!(n >= 2, "need at least 2 wires for the degree chain");
        GadgetScheduler {
            n,
            pairing:   (n..2 * n).collect(),
            free_pool: (2 * n..3 * n).collect(),
        }
    }

    /// Total wires in the gadgetized circuit (always 3n).
    pub fn total_wires(&self) -> usize {
        3 * self.n
    }

    /// Aux wire currently paired with main wire i.
    pub fn current_aux(&self, i: usize) -> usize {
        self.pairing[i]
    }

    /// The wire at the front of the free pool — the next r2.
    pub fn next_r2(&self) -> usize {
        *self.free_pool.front().expect("free pool empty")
    }

    /// Current free pool in order (front = next r2, back = most recently freed).
    pub fn free_wires(&self) -> Vec<usize> {
        self.free_pool.iter().copied().collect()
    }

    /// Apply the pairing update for a gadget on active wire `a`:
    ///   - pops r2 from front of free pool (r2 becomes paired to a)
    ///   - pushes r1 (old pair of a) to back of free pool (r1 is now free)
    /// Returns (r1, r2).
    pub fn consume(&mut self, a: usize) -> (usize, usize) {
        let r2 = self.free_pool.pop_front().expect("free pool empty");
        let r1 = std::mem::replace(&mut self.pairing[a], r2);
        self.free_pool.push_back(r1);
        (r1, r2)
    }

    /// Remap a degree-chain gate from template indices 0..n-1 to the actual
    /// wire indices of the current free pool. Template index j → free_pool[j].
    pub fn remap_chain_gate(&self, gate: [u8; 3]) -> [u8; 3] {
        let free = self.free_wires();
        [
            free[gate[0] as usize] as u8,
            free[gate[1] as usize] as u8,
            free[gate[2] as usize] as u8,
        ]
    }
}

/// Emit the 9-gate gadget for main gate `gate` into `out`.
/// Updates the scheduler: r2 consumed from pool, r1 pushed to pool.
pub fn emit_gadget(sched: &mut GadgetScheduler, gate: [u8; 3], out: &mut Vec<[u8; 3]>) {
    let a = gate[0] as usize;
    let b = gate[1] as usize;
    let c = gate[2] as usize;

    let (r1, r2) = sched.consume(a);
    let r_b = sched.current_aux(b);
    let r_c = sched.current_aux(c);

    let map: [u8; 7] = [
        a as u8, r1 as u8, r2 as u8,
        b as u8, r_b as u8,
        c as u8, r_c as u8,
    ];
    for &[ga, gb, gc] in &GADGET {
        out.push([map[ga as usize], map[gb as usize], map[gc as usize]]);
    }
}

/// Generate the degree-raising chain circuit on n template wires (indices 0..n-1).
///
/// Pass to `gadgetize` as the `aux` argument. The scheduler remaps template
/// indices to the actual current free pool wires before emitting each gate,
/// so the chain always targets free wires regardless of pool churn.
///
/// Gate at step k: [k%n, (k+1)%n, (k-1+n)%n]
/// Semantics: new_x_active = x_active + 1 + x_{ctrl2} + x_{ctrl1}*x_{ctrl2}  (GF2)
/// After n sequential steps on n independent wires, max algebraic degree = n.
pub fn degree_chain_circuit(n: usize, steps: usize) -> CircuitSeq {
    assert!(n >= 2, "need at least 2 wires for the degree chain");
    let gates = (1..=steps)
        .map(|k| {
            let active = (k % n) as u8;
            let ctrl1  = ((k + 1) % n) as u8;
            let ctrl2  = ((k - 1 + n) % n) as u8;
            [active, ctrl1, ctrl2]
        })
        .collect();
    CircuitSeq { gates }
}

/// Wrap `main` with homomorphic gadgets interleaved with `aux` degree-chain gates.
///
/// `main` — original n-wire circuit.
/// `aux`  — degree-chain circuit on template indices 0..n-1,
///           generated with `degree_chain_circuit(n, steps)`.
/// `n`    — number of wires in the original circuit.
///
/// Aux gates are distributed evenly between gadgets so each freed r1 wire
/// accumulates ~n chain steps (degree n) before cycling back as r2.
/// Returns a circuit on 3n wires.
pub fn gadgetize(main: &CircuitSeq, aux: &CircuitSeq, n: usize) -> CircuitSeq {
    let mut sched = GadgetScheduler::new(n);
    let mut out = Vec::with_capacity(main.gates.len() * 9 + aux.gates.len());

    let m = main.gates.len();
    let a = aux.gates.len();
    let mut aux_cursor = 0usize;

    for (i, &gate) in main.gates.iter().enumerate() {
        let aux_target = if m > 0 { a * i / m } else { 0 };
        while aux_cursor < aux_target {
            out.push(sched.remap_chain_gate(aux.gates[aux_cursor]));
            aux_cursor += 1;
        }
        emit_gadget(&mut sched, gate, &mut out);
    }

    while aux_cursor < a {
        out.push(sched.remap_chain_gate(aux.gates[aux_cursor]));
        aux_cursor += 1;
    }

    CircuitSeq { gates: out }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduler_initial_state() {
        let sched = GadgetScheduler::new(3);
        // 3 main + 3 paired + 3 free = 9
        assert_eq!(sched.total_wires(), 9);
        assert_eq!(sched.current_aux(0), 3);
        assert_eq!(sched.current_aux(2), 5);
        assert_eq!(sched.next_r2(), 6);
        assert_eq!(sched.free_wires(), vec![6, 7, 8]);
    }

    #[test]
    fn scheduler_consume_cycles_correctly() {
        let mut sched = GadgetScheduler::new(3);
        // free pool = [6, 7, 8]; wire 0 paired to 3
        let (r1, r2) = sched.consume(0);
        assert_eq!(r1, 3);
        assert_eq!(r2, 6);
        assert_eq!(sched.current_aux(0), 6);
        assert_eq!(sched.free_wires(), vec![7, 8, 3]);

        let (r1b, r2b) = sched.consume(1);
        assert_eq!(r1b, 4);
        assert_eq!(r2b, 7);
        assert_eq!(sched.free_wires(), vec![8, 3, 4]);
    }

    #[test]
    fn remap_tracks_pool_changes() {
        let mut sched = GadgetScheduler::new(3);
        // free pool = [6, 7, 8]; template [0,1,2] → [6,7,8]
        assert_eq!(sched.remap_chain_gate([0, 1, 2]), [6, 7, 8]);
        sched.consume(0); // pool → [7, 8, 3]
        assert_eq!(sched.remap_chain_gate([0, 1, 2]), [7, 8, 3]);
    }

    #[test]
    fn gadgetize_gate_count() {
        let n = 4;
        let main = CircuitSeq { gates: vec![[0,1,2],[1,0,3],[2,3,0],[3,1,2]] };
        let aux  = degree_chain_circuit(n, 40);
        let result = gadgetize(&main, &aux, n);
        assert_eq!(result.gates.len(), 4 * 9 + 40);
    }

    #[test]
    fn all_wires_in_range() {
        let n = 4;
        let main = CircuitSeq { gates: vec![[0,1,2],[1,0,3],[2,3,0],[3,1,2]] };
        let aux  = degree_chain_circuit(n, 40);
        let result = gadgetize(&main, &aux, n);
        for gate in &result.gates {
            for &w in gate {
                assert!((w as usize) < 3 * n, "wire {} out of range 0..{}", w, 3 * n);
            }
        }
    }

    #[test]
    fn degree_before_after() {
        use crate::circuit::circuit::{poly_degree, poly_to_str};

        let n = 8usize;
        let main = crate::random::random_data::random_circuit(n, 20);
        assert_eq!(main.gates.len(), 20);

        println!("\n=== Algebraic Degree Before/After Gadgetize (n={}) ===", n);

        // Before: symbolic polynomials over n variables
        println!("\nBEFORE ({} gates, {} wires):", main.gates.len(), n);
        let polys_before = main.to_polynomial(n, 0, main.gates.len());
        for (wire, poly) in polys_before.iter().enumerate() {
            println!("  wire {:2}: degree {}", wire, poly_degree(poly));
        }

        // Windup: 2 full cycles on ALL 2n aux wires (paired n..2n-1 and free 2n..3n-1).
        // Template index j maps to wire n+j.
        let windup_circ = degree_chain_circuit(2 * n, 2 * (2 * n));
        let mut gadgetized_gates: Vec<[u8; 3]> = windup_circ.gates.iter()
            .map(|&[a, b, c]| [a + n as u8, b + n as u8, c + n as u8])
            .collect();

        // Body: interleave gadgets with free-pool degree chain.
        let steps = n * n;
        let body_aux = degree_chain_circuit(n, steps);
        let mut sched = GadgetScheduler::new(n);
        let m = main.gates.len();
        let mut aux_cursor = 0usize;
        for (i, &gate) in main.gates.iter().enumerate() {
            let aux_target = if m > 0 { steps * i / m } else { 0 };
            while aux_cursor < aux_target {
                gadgetized_gates.push(sched.remap_chain_gate(body_aux.gates[aux_cursor]));
                aux_cursor += 1;
            }
            emit_gadget(&mut sched, gate, &mut gadgetized_gates);
        }
        while aux_cursor < steps {
            gadgetized_gates.push(sched.remap_chain_gate(body_aux.gates[aux_cursor]));
            aux_cursor += 1;
        }
        let gadgetized = CircuitSeq { gates: gadgetized_gates };
        let total = 3 * n;

        let print_snapshot = |gates: &[[u8; 3]], label: &str| {
            let circ = CircuitSeq { gates: gates.to_vec() };
            let polys = circ.to_polynomial(total, 0, gates.len());
            println!("\n{} ({} gates):", label, gates.len());
            for wire in 0..total {
                let cat = if wire < n { "main  " } else if wire < 2*n { "paired" } else { "free  " };
                println!("  wire {:2} [{}]: degree {}", wire, cat, poly_degree(&polys[wire]));
            }
        };

        let g = &gadgetized.gates;
        let total_gates = g.len();
        print_snapshot(&g[..50.min(total_gates)],          "SNAPSHOT gate 50");
        print_snapshot(&g[..total_gates * 2 / 5], "SNAPSHOT 2/5");
        print_snapshot(&g[..total_gates * 3 / 5], "SNAPSHOT 3/5");
        print_snapshot(&g[..total_gates * 4 / 5], "SNAPSHOT 4/5");
        print_snapshot(g, "AFTER");
    }
}
