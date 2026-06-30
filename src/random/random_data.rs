// Updated canonicalization methods and new randomizations

use crate::{
    circuit::{CircuitSeq, Gate, Permutation},
    rainbow::canonical::{self, CandSet, Canonicalization},
};

use rand::{Rng, RngCore, prelude::IndexedRandom};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
// use duckdb::{Connection, AccessMode, Config};

// Used to keep track of "Indirect" collisions when searching for convex subcircuits
pub struct PathConnectedWires {
    wires: Vec<bool>,
    count: usize,
}

#[derive(Clone)]
struct DenseWireSet {
    wires: Vec<bool>,
    count: usize,
}

impl DenseWireSet {
    fn new(num_wires: usize) -> Self {
        Self {
            wires: vec![false; num_wires],
            count: 0,
        }
    }

    fn contains(&self, wire: &u16) -> bool {
        self.wires.get(*wire as usize).copied().unwrap_or(false)
    }

    fn insert(&mut self, wire: u16) {
        let idx = wire as usize;
        if idx >= self.wires.len() {
            self.wires.resize(idx + 1, false);
        }
        if !self.wires[idx] {
            self.wires[idx] = true;
            self.count += 1;
        }
    }

    fn extend_gate(&mut self, gate: [u16; 3]) {
        self.insert(gate[0]);
        self.insert(gate[1]);
        self.insert(gate[2]);
    }

    fn len_after_extending_gate(&self, gate: [u16; 3]) -> usize {
        let [a, b, c] = gate;
        let mut extra = 0;
        if !self.contains(&a) {
            extra += 1;
        }
        if b != a && !self.contains(&b) {
            extra += 1;
        }
        if c != a && c != b && !self.contains(&c) {
            extra += 1;
        }
        self.count + extra
    }

    fn len(&self) -> usize {
        self.count
    }
}

impl PathConnectedWires {
    pub fn new(num_wires: usize) -> Self {
        Self {
            wires: vec![false; num_wires],
            count: 0,
        }
    }

    pub fn all_wires_hit(&self) -> bool {
        self.count == self.wires.len()
    }

    pub fn wire_hit(&self, wire: usize) -> bool {
        self.wires[wire]
    }

    pub fn add_wire(&mut self, wire: usize) {
        if !self.wires[wire] {
            self.count += 1;
        }
        self.wires[wire] = true;
    }

    pub fn count(&self) -> usize {
        self.count
    }
}

// Computes a completely random circuit on n wires and m gates
pub fn random_circuit(n: usize, m: usize) -> CircuitSeq {
    assert!(n >= 3, "random circuits need at least 3 wires");
    assert!(
        n <= u16::MAX as usize + 1,
        "random circuit wire count exceeds u16 wire indices"
    );
    let mut circuit = Vec::with_capacity(m);

    for _ in 0..m {
        loop {
            // pick 3 distinct pins
            let mut gate = [0u16; 3];
            for j in 0..3 {
                loop {
                    let v = fastrand::usize(..n);
                    if !gate[..j].iter().any(|&pin| pin == v as u16) {
                        gate[j] = v as u16;
                        break;
                    }
                }
            }

            // check against last gate to avoid duplicates
            if circuit.last() == Some(&gate) {
                continue;
            } else {
                circuit.push(gate);
                break;
            }
        }
    }

    CircuitSeq { gates: circuit }
}

// Checks if a subcircuit (stored as indicies) is convex
pub fn is_convex(num_wires: usize, circuit: &CircuitSeq, convex_gate_ids: &[usize]) -> bool {
    // early exit for too few gates
    if convex_gate_ids.len() < 2 {
        return false;
    }

    // track gates outside the convex set that interfere with its paths
    let mut colliding_set = vec![];
    let mut path_colliding_targets = vec![false; num_wires];
    let mut path_colliding_controls = vec![false; num_wires];
    let mut selected_pos = 0;

    // iterate through gates between first and last of convex set
    for i in convex_gate_ids[0]..=*convex_gate_ids.last().unwrap() {
        if selected_pos < convex_gate_ids.len() && i == convex_gate_ids[selected_pos] {
            selected_pos += 1;
            // gate is inside convex set
            let selected_gate = circuit.gates[i];

            // check no collision with colliding_set
            for c_gate in colliding_set.iter() {
                if Gate::collides_index(&selected_gate, &c_gate) {
                    return false;
                }
            }

            let [t, c0, c1] = selected_gate;
            path_colliding_targets[t as usize] = true;
            path_colliding_controls[c0 as usize] = true;
            path_colliding_controls[c1 as usize] = true;
        } else {
            // gate outside convex set
            let g = circuit.gates[i];
            let [t, c0, c1] = g;

            if path_colliding_targets[c0 as usize]
                || path_colliding_targets[c1 as usize]
                || path_colliding_controls[t as usize]
            {
                colliding_set.push(g);
                path_colliding_targets[t as usize] = true;
                path_colliding_controls[c0 as usize] = true;
                path_colliding_controls[c1 as usize] = true;
            }
        }
    }

    true
}

// More complex method to find a convex subcircuit of up to max_wires. Starts with a random gate and then expands left and right, adding gates that collide with the current set until we can’t add more without exceeding max_wires. Then checks if the resulting set is convex; if not, retries.
// Use max_wires version or max_gates version for the maximal versions
// Use simple version for average case

// Like find_convex_subcircuit but greedily picks the candidate that adds the MOST new wires.
// At each step, scans all candidates and takes the first one adding 3 new wires; falls back
// to 2, then 1, then 0. This maximises wire diversity in the subcircuit.
pub fn find_convex_subcircuit_max_wires<R: RngCore>(
    max_wires: usize,
    num_wires: usize,
    circuit: &CircuitSeq,
    rng: &mut R,
) -> (Vec<usize>, usize) {
    let num_gates = circuit.gates.len();
    let mut search_attempts = 0;
    let max_attempts = 3;
    let window = 50;
    loop {
        search_attempts += 1;
        if search_attempts > max_attempts {
            return (vec![], search_attempts);
        }

        let len = circuit.gates.len();
        let mut selected_gate_idx = vec![0; len];
        selected_gate_idx[0] = rng.random_range(0..num_gates);
        let mut selected_gate_ctr = 1;

        let mut curr_wires = DenseWireSet::new(num_wires);
        curr_wires.extend_gate(circuit.gates[selected_gate_idx[0]]);

        while selected_gate_ctr < len {
            let mut candidates: Vec<(usize, usize)> = Vec::with_capacity(window * 2);

            // Left-most gate, go right
            let mut path_connected_target_wires = PathConnectedWires::new(num_wires);
            let mut path_connected_control_wires = PathConnectedWires::new(num_wires);
            let mut selected_gates_seen = 1;

            if selected_gate_idx[0] != num_gates - 1 {
                let right_bound = (selected_gate_idx[0] + window).min(num_gates - 1);
                for curr_idx in selected_gate_idx[0] + 1..right_bound {
                    if path_connected_target_wires.all_wires_hit()
                        || path_connected_control_wires.all_wires_hit()
                    {
                        break;
                    }
                    if selected_gates_seen < selected_gate_ctr
                        && curr_idx == selected_gate_idx[selected_gates_seen]
                    {
                        selected_gates_seen += 1;
                    } else {
                        let curr_gate = circuit.gates[curr_idx];
                        let mut collides = false;
                        let mut repeat = false;
                        for i in 0..selected_gates_seen {
                            if Gate::collides_index(
                                &curr_gate,
                                &circuit.gates[selected_gate_idx[i]],
                            ) {
                                collides = true;
                                break;
                            }
                        }
                        for i in 0..selected_gate_ctr {
                            if curr_gate == circuit.gates[selected_gate_idx[i]] {
                                repeat = true;
                                break;
                            }
                        }
                        let [t, c1, c2] = curr_gate;
                        let indirect = path_connected_control_wires.wire_hit(t as usize)
                            || path_connected_target_wires.wire_hit(c1 as usize)
                            || path_connected_target_wires.wire_hit(c2 as usize);
                        if collides || indirect {
                            path_connected_target_wires.add_wire(t as usize);
                            path_connected_control_wires.add_wire(c1 as usize);
                            path_connected_control_wires.add_wire(c2 as usize);
                            let new_w = curr_gate
                                .iter()
                                .filter(|&w| !curr_wires.contains(w))
                                .count();
                            if !indirect && !repeat {
                                candidates.push((curr_idx, new_w));
                            }
                        }
                    }
                }
            }

            // Right-most gate, go left
            let mut path_connected_target_wires = PathConnectedWires::new(num_wires);
            let mut path_connected_control_wires = PathConnectedWires::new(num_wires);
            let mut selected_gates_seen = 1;

            if selected_gate_idx[selected_gate_ctr - 1] != 0 {
                let left_bound = selected_gate_idx[selected_gate_ctr - 1].saturating_sub(window);
                for curr_idx in (left_bound..=selected_gate_idx[selected_gate_ctr - 1] - 1).rev() {
                    if path_connected_target_wires.all_wires_hit()
                        || path_connected_control_wires.all_wires_hit()
                    {
                        break;
                    }
                    if selected_gates_seen < selected_gate_ctr
                        && curr_idx
                            == selected_gate_idx[selected_gate_ctr - 1 - selected_gates_seen]
                    {
                        selected_gates_seen += 1;
                    } else {
                        let curr_gate = circuit.gates[curr_idx];
                        let mut collides = false;
                        let mut repeat = false;
                        for i in 0..selected_gates_seen {
                            if Gate::collides_index(
                                &curr_gate,
                                &circuit.gates[selected_gate_idx[selected_gate_ctr - 1 - i]],
                            ) {
                                collides = true;
                                break;
                            }
                        }
                        for i in 0..selected_gate_ctr {
                            if curr_gate == circuit.gates[selected_gate_idx[i]] {
                                repeat = true;
                                break;
                            }
                        }
                        let [t, c1, c2] = curr_gate;
                        let indirect = path_connected_control_wires.wire_hit(t as usize)
                            || path_connected_target_wires.wire_hit(c1 as usize)
                            || path_connected_target_wires.wire_hit(c2 as usize);
                        if collides || indirect {
                            path_connected_target_wires.add_wire(t as usize);
                            path_connected_control_wires.add_wire(c1 as usize);
                            path_connected_control_wires.add_wire(c2 as usize);
                            let new_w = curr_gate
                                .iter()
                                .filter(|&w| !curr_wires.contains(w))
                                .count();
                            if !indirect && !repeat {
                                candidates.push((curr_idx, new_w));
                            }
                        }
                    }
                }
            }

            if candidates.is_empty() {
                break;
            }

            // Pick candidate with the most new wires (prefer 3, then 2, then 1, then 0)
            let next_candidate = (0..=3).rev().find_map(|target| {
                candidates
                    .iter()
                    .find(|(_, nw)| *nw == target)
                    .map(|(idx, _)| *idx)
            });
            let next_candidate = match next_candidate {
                Some(x) => x,
                None => break,
            };

            let mut insert_pos = selected_gate_ctr;
            while insert_pos > 0 && selected_gate_idx[insert_pos - 1] > next_candidate {
                selected_gate_idx[insert_pos] = selected_gate_idx[insert_pos - 1];
                insert_pos -= 1;
            }
            selected_gate_idx[insert_pos] = next_candidate;
            selected_gate_ctr += 1;
            curr_wires.extend_gate(circuit.gates[next_candidate]);
            if curr_wires.len() >= max_wires {
                break;
            }
        }

        if selected_gate_ctr < 3 {
            continue;
        }
        if !is_convex(num_wires, circuit, &selected_gate_idx[..selected_gate_ctr]) {
            continue;
        }
        return (
            selected_gate_idx[..selected_gate_ctr].to_vec(),
            search_attempts,
        );
    }
}

// Like find_convex_subcircuit but greedily picks the candidate that adds the FEWEST new wires.
// At each step takes the first candidate adding 0 new wires; falls back to 1, 2, 3.
// This maximises gate count while reusing existing wires.
pub fn find_convex_subcircuit_max_gates<R: RngCore>(
    max_wires: usize,
    num_wires: usize,
    circuit: &CircuitSeq,
    rng: &mut R,
) -> (Vec<usize>, usize) {
    let num_gates = circuit.gates.len();
    let mut search_attempts = 0;
    let max_attempts = 3;
    let window = 50;
    loop {
        search_attempts += 1;
        if search_attempts > max_attempts {
            return (vec![], search_attempts);
        }

        let len = circuit.gates.len();
        let mut selected_gate_idx = vec![0; len];
        selected_gate_idx[0] = rng.random_range(0..num_gates);
        let mut selected_gate_ctr = 1;

        let mut curr_wires = DenseWireSet::new(num_wires);
        curr_wires.extend_gate(circuit.gates[selected_gate_idx[0]]);

        while selected_gate_ctr < len {
            let mut candidates: Vec<(usize, usize)> = Vec::with_capacity(window * 2);

            // Left-most gate, go right
            let mut path_connected_target_wires = PathConnectedWires::new(num_wires);
            let mut path_connected_control_wires = PathConnectedWires::new(num_wires);
            let mut selected_gates_seen = 1;

            if selected_gate_idx[0] != num_gates - 1 {
                let right_bound = (selected_gate_idx[0] + window).min(num_gates - 1);
                for curr_idx in selected_gate_idx[0] + 1..right_bound {
                    if path_connected_target_wires.all_wires_hit()
                        || path_connected_control_wires.all_wires_hit()
                    {
                        break;
                    }
                    if selected_gates_seen < selected_gate_ctr
                        && curr_idx == selected_gate_idx[selected_gates_seen]
                    {
                        selected_gates_seen += 1;
                    } else {
                        let curr_gate = circuit.gates[curr_idx];
                        let mut collides = false;
                        let mut repeat = false;
                        for i in 0..selected_gates_seen {
                            if Gate::collides_index(
                                &curr_gate,
                                &circuit.gates[selected_gate_idx[i]],
                            ) {
                                collides = true;
                                break;
                            }
                        }
                        for i in 0..selected_gate_ctr {
                            if curr_gate == circuit.gates[selected_gate_idx[i]] {
                                repeat = true;
                                break;
                            }
                        }
                        let [t, c1, c2] = curr_gate;
                        let indirect = path_connected_control_wires.wire_hit(t as usize)
                            || path_connected_target_wires.wire_hit(c1 as usize)
                            || path_connected_target_wires.wire_hit(c2 as usize);
                        if collides || indirect {
                            path_connected_target_wires.add_wire(t as usize);
                            path_connected_control_wires.add_wire(c1 as usize);
                            path_connected_control_wires.add_wire(c2 as usize);
                            let new_w = curr_gate
                                .iter()
                                .filter(|&w| !curr_wires.contains(w))
                                .count();
                            if !indirect && !repeat {
                                candidates.push((curr_idx, new_w));
                            }
                        }
                    }
                }
            }

            // Right-most gate, go left
            let mut path_connected_target_wires = PathConnectedWires::new(num_wires);
            let mut path_connected_control_wires = PathConnectedWires::new(num_wires);
            let mut selected_gates_seen = 1;

            if selected_gate_idx[selected_gate_ctr - 1] != 0 {
                let left_bound = selected_gate_idx[selected_gate_ctr - 1].saturating_sub(window);
                for curr_idx in (left_bound..=selected_gate_idx[selected_gate_ctr - 1] - 1).rev() {
                    if path_connected_target_wires.all_wires_hit()
                        || path_connected_control_wires.all_wires_hit()
                    {
                        break;
                    }
                    if selected_gates_seen < selected_gate_ctr
                        && curr_idx
                            == selected_gate_idx[selected_gate_ctr - 1 - selected_gates_seen]
                    {
                        selected_gates_seen += 1;
                    } else {
                        let curr_gate = circuit.gates[curr_idx];
                        let mut collides = false;
                        let mut repeat = false;
                        for i in 0..selected_gates_seen {
                            if Gate::collides_index(
                                &curr_gate,
                                &circuit.gates[selected_gate_idx[selected_gate_ctr - 1 - i]],
                            ) {
                                collides = true;
                                break;
                            }
                        }
                        for i in 0..selected_gate_ctr {
                            if curr_gate == circuit.gates[selected_gate_idx[i]] {
                                repeat = true;
                                break;
                            }
                        }
                        let [t, c1, c2] = curr_gate;
                        let indirect = path_connected_control_wires.wire_hit(t as usize)
                            || path_connected_target_wires.wire_hit(c1 as usize)
                            || path_connected_target_wires.wire_hit(c2 as usize);
                        if collides || indirect {
                            path_connected_target_wires.add_wire(t as usize);
                            path_connected_control_wires.add_wire(c1 as usize);
                            path_connected_control_wires.add_wire(c2 as usize);
                            let new_w = curr_gate
                                .iter()
                                .filter(|&w| !curr_wires.contains(w))
                                .count();
                            if !indirect && !repeat {
                                candidates.push((curr_idx, new_w));
                            }
                        }
                    }
                }
            }

            if candidates.is_empty() {
                break;
            }

            // Pick candidate with the fewest new wires (prefer 0, then 1, then 2, then 3)
            let next_candidate = (0..=3).find_map(|target| {
                candidates
                    .iter()
                    .find(|(_, nw)| *nw == target)
                    .map(|(idx, _)| *idx)
            });
            let next_candidate = match next_candidate {
                Some(x) => x,
                None => break,
            };

            let mut insert_pos = selected_gate_ctr;
            while insert_pos > 0 && selected_gate_idx[insert_pos - 1] > next_candidate {
                selected_gate_idx[insert_pos] = selected_gate_idx[insert_pos - 1];
                insert_pos -= 1;
            }
            selected_gate_idx[insert_pos] = next_candidate;
            selected_gate_ctr += 1;
            curr_wires.extend_gate(circuit.gates[next_candidate]);
            if curr_wires.len() >= max_wires {
                break;
            }
        }

        if selected_gate_ctr < 3 {
            continue;
        }
        if !is_convex(num_wires, circuit, &selected_gate_idx[..selected_gate_ctr]) {
            continue;
        }
        return (
            selected_gate_idx[..selected_gate_ctr].to_vec(),
            search_attempts,
        );
    }
}

// Same as above but instead of scanning an entire candidate list, just take the first candidate from the left and right and choose randomly from those two
pub fn simple_find_convex_subcircuit<R: RngCore>(
    num_wires: usize,
    circuit: &CircuitSeq,
    rng: &mut R,
) -> (Vec<usize>, usize) {
    let num_gates = circuit.gates.len();
    let mut search_attempts = 0;
    let max_attempts = 3;
    loop {
        search_attempts += 1;
        if search_attempts > max_attempts {
            // eprintln!(
            //     "No convex subcircuit found after {} attempts (set_size={}, max_wires={})",
            //     search_attempts, set_size, max_wires
            // );
            return (vec![], search_attempts);
        }

        // Start with one random gate
        let len = circuit.gates.len();
        let mut selected_gate_idx = vec![0; len];
        selected_gate_idx[0] = rng.random_range(0..num_gates);
        let mut selected_gate_ctr = 1;

        // Initialize wire set
        let mut curr_wires = DenseWireSet::new(num_wires);
        curr_wires.extend_gate(circuit.gates[selected_gate_idx[0]]);

        while selected_gate_ctr < len {
            if selected_gate_ctr >= 30 {
                break;
            }
            let mut candidates: Vec<usize> = Vec::with_capacity(2);

            // Left-most gate, go right
            let mut path_connected_target_wires = PathConnectedWires::new(num_wires);
            let mut path_connected_control_wires = PathConnectedWires::new(num_wires);
            let mut selected_gates_seen = 1;

            if selected_gate_idx[0] != num_gates - 1 {
                let right_bound = num_gates - 1;
                for curr_idx in selected_gate_idx[0] + 1..right_bound {
                    if path_connected_target_wires.all_wires_hit()
                        || path_connected_control_wires.all_wires_hit()
                    {
                        break;
                    }

                    if curr_idx == selected_gate_idx[selected_gates_seen] {
                        selected_gates_seen += 1;
                    } else {
                        let curr_gate = circuit.gates[curr_idx];
                        let mut collides_with_prev_selected = false;
                        let repeat_wires = false;

                        for i in 0..selected_gates_seen {
                            if Gate::collides_index(
                                &curr_gate,
                                &circuit.gates[selected_gate_idx[i]],
                            ) {
                                collides_with_prev_selected = true;
                                break;
                            }
                        }
                        //
                        // for i in 0..selected_gate_ctr {
                        //     if curr_gate == circuit.gates[selected_gate_idx[i]] {
                        //         repeat_wires = true;
                        //         break;
                        //     }
                        // }

                        let [t, c1, c2] = curr_gate;
                        let indirect_path_connected = path_connected_control_wires
                            .wire_hit(t as usize)
                            || path_connected_target_wires.wire_hit(c1 as usize)
                            || path_connected_target_wires.wire_hit(c2 as usize);

                        if collides_with_prev_selected || indirect_path_connected {
                            path_connected_target_wires.add_wire(t as usize);
                            path_connected_control_wires.add_wire(c1 as usize);
                            path_connected_control_wires.add_wire(c2 as usize);

                            if !indirect_path_connected && !repeat_wires {
                                candidates.push(curr_idx);
                                break;
                            }
                        }
                    }
                }
            }

            // Right-most gate, go left
            let mut path_connected_target_wires = PathConnectedWires::new(num_wires);
            let mut path_connected_control_wires = PathConnectedWires::new(num_wires);
            let mut selected_gates_seen = 1;

            if selected_gate_idx[selected_gate_ctr - 1] != 0 {
                let left_bound = 0;
                for curr_idx in (left_bound..=selected_gate_idx[selected_gate_ctr - 1] - 1).rev() {
                    if path_connected_target_wires.all_wires_hit()
                        || path_connected_control_wires.all_wires_hit()
                    {
                        break;
                    }

                    if selected_gates_seen < selected_gate_ctr
                        && curr_idx
                            == selected_gate_idx[selected_gate_ctr - 1 - selected_gates_seen]
                    {
                        selected_gates_seen += 1;
                    } else {
                        let curr_gate = circuit.gates[curr_idx];
                        let mut collides_with_prev_selected = false;
                        let repeat_wires = false;

                        for i in 0..selected_gates_seen {
                            if Gate::collides_index(
                                &curr_gate,
                                &circuit.gates[selected_gate_idx[selected_gate_ctr - 1 - i]],
                            ) {
                                collides_with_prev_selected = true;
                                break;
                            }
                        }
                        // for i in 0..selected_gate_ctr {
                        //     if curr_gate == circuit.gates[selected_gate_idx[i]] {
                        //         repeat_wires = true;
                        //         break;
                        //     }
                        // }

                        let [t, c1, c2] = curr_gate;
                        let indirect_path_connected = path_connected_control_wires
                            .wire_hit(t as usize)
                            || path_connected_target_wires.wire_hit(c1 as usize)
                            || path_connected_target_wires.wire_hit(c2 as usize);

                        if collides_with_prev_selected || indirect_path_connected {
                            path_connected_target_wires.add_wire(t as usize);
                            path_connected_control_wires.add_wire(c1 as usize);
                            path_connected_control_wires.add_wire(c2 as usize);

                            if !indirect_path_connected && !repeat_wires {
                                candidates.push(curr_idx);
                                break;
                            }
                        }
                    }
                }
            }

            // Stop expanding if no valid candidates
            if candidates.is_empty() {
                break;
            }

            // Pick a random next gate that hasn’t been used
            let mut next_candidate = None;
            for _ in 0..candidates.len() {
                let cand = *candidates.choose(rng).unwrap();
                if !selected_gate_idx[..selected_gate_ctr].contains(&cand) {
                    next_candidate = Some(cand);
                    break;
                }
            }

            // Stop if no unused candidate left
            let next_candidate = match next_candidate {
                Some(x) => x,
                None => break,
            };

            if curr_wires.len_after_extending_gate(circuit.gates[next_candidate]) > 21 {
                break;
            }

            // Insert next gate in sorted order
            let mut insert_pos = selected_gate_ctr;
            while insert_pos > 0 && selected_gate_idx[insert_pos - 1] > next_candidate {
                selected_gate_idx[insert_pos] = selected_gate_idx[insert_pos - 1];
                insert_pos -= 1;
            }
            selected_gate_idx[insert_pos] = next_candidate;
            selected_gate_ctr += 1;

            // Commit wire update
            curr_wires.extend_gate(circuit.gates[next_candidate]);
        }

        if selected_gate_ctr < 3 {
            continue;
        }

        if !is_convex(num_wires, circuit, &selected_gate_idx[..selected_gate_ctr]) {
            continue;
        }

        // println!(
        //     "convex subcircuit found! {} wires {} gates",
        //     curr_wires.len(),
        //     selected_gate_ctr
        // );
        return (
            selected_gate_idx[..selected_gate_ctr].to_vec(),
            search_attempts,
        );
    }
}

// Rearranges circuit to put the convex subcircuit in a contiguous manner.
// Evicts every non-member gate lying between the first and last selected gate,
// pushing each out to whichever side it can commute past (no collision with any
// gate it must cross). Repeats to a fixpoint. Convexity guarantees every
// interior non-member can eventually clear to at least one side, so the loop
// drains the interval. If a full sweep makes no progress while interior
// non-members remain (should not happen for a convex set), we return None
// rather than a bogus range so callers can skip cleanly.
pub fn contiguous_convex(
    circuit: &mut CircuitSeq,
    ordered_convex_gates: &mut Vec<usize>,
    num_wires: usize,
    tags: &mut Vec<u32>,
) -> Option<(usize, usize)> {
    let track = !tags.is_empty();
    // This should never run
    if ordered_convex_gates.len() < 2 {
        return None;
    }

    if !is_convex(num_wires, circuit, &ordered_convex_gates) {
        panic!("not convex");
    }

    // Track which positions hold a selected (member) gate. Kept in sync with the
    // circuit as gates move via the same remove/insert operations.
    let mut member = vec![false; circuit.gates.len()];
    for &idx in ordered_convex_gates.iter() {
        member[idx] = true;
    }

    // Block boundaries: min and max member positions.
    let mut start = *ordered_convex_gates.first().unwrap();
    let mut end = *ordered_convex_gates.last().unwrap();

    // Drain interior non-members to a fixpoint.
    loop {
        let mut moved = false;
        let mut p = start + 1;
        while p < end {
            if member[p] {
                p += 1;
                continue;
            }

            // Can it commute left past everything in [start, p)?
            let can_left =
                (start..p).all(|i| !Gate::collides_index(&circuit.gates[i], &circuit.gates[p]));
            if can_left {
                let gate = circuit.gates.remove(p);
                circuit.gates.insert(start, gate);
                member.remove(p);
                member.insert(start, false);
                if track {
                    let tag = tags.remove(p);
                    tags.insert(start, tag);
                }
                start += 1;
                moved = true;
                break;
            }

            // Otherwise can it commute right past everything in (p, end]?
            let can_right = ((p + 1)..=end)
                .all(|i| !Gate::collides_index(&circuit.gates[i], &circuit.gates[p]));
            if can_right {
                let gate = circuit.gates.remove(p);
                circuit.gates.insert(end, gate);
                member.remove(p);
                member.insert(end, false);
                if track {
                    let tag = tags.remove(p);
                    tags.insert(end, tag);
                }
                end -= 1;
                moved = true;
                break;
            }

            p += 1;
        }

        if !moved {
            break;
        }
    }

    // If any interior non-member survived, we could not contiguize.
    if (start..=end).any(|i| !member[i]) {
        return None;
    }

    // Members now occupy exactly the contiguous block [start, end].
    *ordered_convex_gates = (start..=end).collect();

    Some((start, end))
}

// Shoots a random gate left or right without collisions
pub fn shoot_random_gate(circuit: &mut CircuitSeq, rounds: usize) {
    let mut rng = rand::rng();
    let len = circuit.gates.len();

    if len == 0 {
        return;
    }

    for _ in 0..rounds {
        let gate_idx = rng.random_range(0..len);
        let go_left: bool = rng.random_bool(0.5);

        if go_left {
            // Shoot left
            let mut target = gate_idx;
            while target > 0 {
                if Gate::collides_index(&circuit.gates[target - 1], &circuit.gates[gate_idx]) {
                    break;
                }
                target -= 1;
            }
            target = rng.random_range(target..=gate_idx);
            if target != gate_idx {
                let gate = circuit.gates.remove(gate_idx);
                circuit.gates.insert(target, gate);
            }
        } else {
            // Shoot right
            let mut target = gate_idx;
            while target + 1 < len {
                if Gate::collides_index(&circuit.gates[target + 1], &circuit.gates[gate_idx]) {
                    break;
                }
                target += 1;
            }
            target = rng.random_range(gate_idx..=target);
            if target != gate_idx {
                let gate = circuit.gates.remove(gate_idx);
                circuit.gates.insert(target, gate);
            }
        }
    }
}

pub fn shoot_left_vec(circuit: &mut Vec<[u16; 3]>, gate_idx: usize) -> usize {
    let mut target = gate_idx;
    while target > 0 {
        if Gate::collides_index(&circuit[target - 1], &circuit[gate_idx]) {
            break;
        }
        target -= 1;
    }

    if target != gate_idx {
        let gate = circuit.remove(gate_idx);
        circuit.insert(target, gate);
    }

    target
}

pub fn shoot_right_vec(circuit: &mut Vec<[u16; 3]>, gate_idx: usize) -> usize {
    let mut target = gate_idx;
    let len = circuit.len();
    while target + 1 < len {
        if Gate::collides_index(&circuit[target + 1], &circuit[gate_idx]) {
            break;
        }
        target += 1;
    }

    if target != gate_idx {
        let gate = circuit.remove(gate_idx);
        circuit.insert(target, gate);
    }

    target
}

// Below used to help construct a skeleton graph representation of a circuit

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Below is used for db storing
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

impl Permutation {
    pub fn canon(&self, bit_shuf: &[Vec<usize>], retry: bool) -> Canonicalization {
        if bit_shuf.is_empty() {
            panic!("bit_shuf cannot be empty!");
        }

        // Try fast canonicalization
        let mut pm = self.fast();

        if pm.perm.data.is_empty() {
            if retry {
                // Fast canon failed, retry with a random shuffle
                let n = self.data.len();
                let r = Permutation::rand_perm(n);
                return self.bit_shuffle(&r.data).canon(bit_shuf, false);
            } else {
                // Retry not allowed, fall back to brute force
                // println!("trying brute");
                pm = self.brute(bit_shuf);
            }
        }

        pm
    }

    pub fn canon_simple(&self, bit_shuf: &[Vec<usize>]) -> Canonicalization {
        self.canon(bit_shuf, false)
    }

    pub fn brute(&self, bit_shuf: &[Vec<usize>]) -> Canonicalization {
        if bit_shuf.is_empty() {
            panic!("bit_shuf cannot be empty!");
        }

        let n = self.data.len();
        let num_b = usize::BITS as usize - (n - 1).leading_zeros() as usize;

        let data = &self.data;

        let best = bit_shuf
            .par_iter()
            .map(|r| {
                // thread-local scratch buffers
                let mut bits = vec![0usize; n];
                let mut index_shuf = vec![0usize; n];
                let mut perm_shuf = vec![0usize; n];

                // apply bit shuffle r
                for (src, &dst) in r.iter().enumerate() {
                    let shift = dst;
                    for i in 0..n {
                        let val = unsafe { *data.get_unchecked(i) };
                        let bit = (val >> src) & 1;
                        let idx = (i >> src) & 1;

                        unsafe {
                            *bits.get_unchecked_mut(i) |= bit << shift;
                            *index_shuf.get_unchecked_mut(i) |= idx << shift;
                        }
                    }
                }

                // permute using index_shuf
                for i in 0..n {
                    let idx = unsafe { *index_shuf.get_unchecked(i) };
                    let v = unsafe { *bits.get_unchecked(i) };
                    unsafe { *perm_shuf.get_unchecked_mut(idx) = v };
                }

                // Return pair (perm_shuf, r) for global minimization
                (perm_shuf, r.to_vec())
            })
            .reduce(
                || {
                    // identity: "worst possible permutation"
                    (
                        vec![usize::MAX; n],
                        vec![usize::MAX; num_b], // never chosen
                    )
                },
                |(perm_a, r_a), (perm_b, r_b)| {
                    // lexicographic comparison to choose the smaller permutation
                    if perm_b < perm_a {
                        (perm_b, r_b)
                    } else {
                        (perm_a, r_a)
                    }
                },
            );

        Canonicalization {
            perm: Permutation { data: best.0 },
            shuffle: Permutation { data: best.1 },
        }
    }

    //Goal of fast canon is to produce small snippets of the best permutation (by lexi order) and determine which in canonical
    //If we can't decide between multiple, for now, we just ignore and will do brute force
    pub fn fast(&self) -> Canonicalization {
        let num_bits = self.bits();
        let mut candidates = CandSet::new(num_bits);
        let mut found_identity = false;

        // Scratch buffer to avoid cloning every iteration
        let mut scratch = CandSet::new(num_bits);

        // Pre-allocate viable_sets buffer to reuse
        let mut viable_sets: Vec<CandSet> = Vec::with_capacity(4);

        for weight in 0..=num_bits / 2 {
            let index_words = canonical::index_set(weight, num_bits); // Vec<usize>

            'word_loop: for &w in &index_words {
                // Determine which preimages are possible
                let preimages = candidates.preimages(w);
                if preimages.is_empty() {
                    return Canonicalization {
                        perm: Permutation { data: Vec::new() },
                        shuffle: Permutation { data: Vec::new() },
                    };
                }

                viable_sets.clear();
                let mut best_score = -1;

                for &pre_idx in &preimages {
                    let mapped_value = self.data[pre_idx];

                    if !candidates.consistent(pre_idx, w) {
                        continue;
                    }

                    // Reset scratch from candidates and enforce mapping
                    scratch.copy_from(&candidates);
                    scratch.enforce(pre_idx, w);

                    // Minimum possible value with current scratch
                    let (score, mut reduced_set) = scratch.min_consistent(mapped_value);
                    if score < 0 {
                        continue;
                    }

                    reduced_set.intersect(&candidates);
                    if !reduced_set.consistent(pre_idx, w) {
                        continue;
                    }

                    // Track best score and viable sets
                    if best_score < 0 || score < best_score {
                        best_score = score;
                        viable_sets.clear();
                        // Move reduced_set into the vector (no clone)
                        viable_sets.push(reduced_set);
                        if w as isize == score {
                            found_identity = true;
                        }
                    } else if score == best_score {
                        if w as isize == score {
                            if found_identity {
                                viable_sets.push(reduced_set);
                            } else {
                                viable_sets.clear();
                                viable_sets.push(reduced_set);
                            }
                            found_identity = true;
                        } else if !found_identity {
                            viable_sets.push(reduced_set);
                        }
                    }
                }

                match viable_sets.len() {
                    0 => continue,
                    1 => candidates = viable_sets.pop().unwrap(),
                    _ => {
                        return Canonicalization {
                            perm: Permutation { data: Vec::new() },
                            shuffle: Permutation { data: Vec::new() },
                        };
                    }
                }

                if candidates.complete() {
                    break 'word_loop;
                }
            }

            if candidates.complete() {
                break;
            }
        }

        if candidates.unconstrained() {
            return Canonicalization {
                perm: self.clone(),
                shuffle: Permutation { data: Vec::new() },
            };
        }

        if !candidates.complete() {
            println!("Incomplete!");
            println!("{:?}", self);
            println!("{:?}", candidates);
            std::process::exit(1);
        }

        let final_shuffle = match candidates.output() {
            Some(v) => Permutation { data: v },
            None => {
                eprintln!("CandSet output returned None!");
                std::process::exit(1);
            }
        };

        Canonicalization {
            perm: self.bit_shuffle(&final_shuffle.data),
            shuffle: final_shuffle,
        }
    }

    pub fn from_string(s: &str) -> Self {
        let data = s
            .split(',')
            .map(|x| {
                x.trim()
                    .parse::<usize>()
                    .expect("Invalid number in permutation")
            })
            .collect();

        Permutation { data }
    }
}

pub fn base_gates(n: usize) -> Vec<[u16; 3]> {
    let n = n as u16;
    let mut gates: Vec<[u16; 3]> = Vec::new();
    for a in 0..n {
        for b in 0..n {
            if b == a {
                continue;
            }
            for c in 0..n {
                if c == a || c == b {
                    continue;
                }
                gates.push([a, b, c]);
            }
        }
    }
    gates
}
