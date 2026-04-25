// Updated canonicalization methods and new randomizations

use crate::{
    circuit::{CircuitSeq, Gate, Permutation},
    rainbow::canonical::{self, CandSet, Canonicalization},
};

use crossbeam::channel::{bounded};
use dashmap::DashMap;
use itertools::Itertools;
use once_cell::sync::Lazy;
use rand::{
    prelude::IndexedRandom,
    Rng, RngCore,
};
use rayon::{
    iter::{IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSlice,
};
use duckdb::{Connection, AccessMode, Config};
use rocksdb::{
    DB, Options, BlockBasedOptions, SstFileWriter, IngestExternalFileOptions,
    MergeOperands, DBCompressionType, Cache,
};
use xxhash_rust::xxh3::xxh3_128;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use rayon::prelude::*;
use smallvec::SmallVec;
use std::{
    collections::{HashMap, HashSet},
    fs::OpenOptions,
    io::Write,
    thread,
};
use crate::circuit::circuit::polys_repr_blob;
use crate::circuit::Polynomial;
use crate::circuit::circuit::{canonicalize_polys, canonicalize_polys_2, canonicalize_polys_3, canonicalize_polys_4};
use crate::circuit::circuit::print_rule_times;

// Store permutation canonicalizations (wire relabeling) in a cache for speed
pub static CANON_CACHE: Lazy<DashMap<Vec<u8>, (Vec<u8>, Vec<u8>)>> = Lazy::new(|| DashMap::new());

// Used to keep track of "Indirect" collisions when searching for convex subcircuits
pub struct PathConnectedWires {
    wires: Vec<bool>,
    count: usize,
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
    let mut circuit = Vec::with_capacity(m);

    for _ in 0..m {
        loop {
            // mask for used pins
            let mut set = [false; 255];
            for i in n..255 {
                set[i as usize] = true; // disable pins >= n
            }

            // pick 3 distinct pins
            let mut gate = [0u8; 3];
            for j in 0..3 {
                loop {
                    let v = fastrand::u8(..255);
                    if !set[v as usize] {
                        set[v as usize] = true;
                        gate[j] = v;
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

// Attempts to find two random but equivalent circuits on n wires and m = 100..300 gates
// Very unlikely to succeed
pub fn random_equivalent_circuits_until_found(n: usize) -> (CircuitSeq, CircuitSeq) {
    // final_state → list of circuits producing that state
    let mut state_map: HashMap<u64, Vec<CircuitSeq>> = HashMap::new();
    let mut total_generated = 0usize;

    loop {
        let m = fastrand::usize(100..=300);
        let circuit = random_circuit(n, m);
        total_generated += 1;

        if total_generated % 10_000 == 0 {
            println!("Generated {} circuits so far...", total_generated);
        }

        // Compute final state starting from 0
        let state = Gate::evaluate_index_list(0, &circuit.gates);

        // Check if we’ve seen this state before
        if let Some(existing_list) = state_map.get_mut(&(state as u64)) {
            // Compare against all circuits with this same state
            if let Some(existing) = existing_list
                .par_iter()
                .find_any(|other| circuit.probably_equal(other, n as usize, 150_000).is_ok())
            {
                println!("Found equivalent circuits after {} total!", total_generated);
                return (existing.clone(), circuit);
            }

            // No match; store this circuit under the same state
            existing_list.push(circuit);
        } else {
            // First circuit for this state
            state_map.insert(state as u64, vec![circuit]);
        }
    }
}

// Checks if a subcircuit (stored as indicies) is convex
pub fn is_convex(num_wires: usize, circuit: &CircuitSeq, convex_gate_ids: &[usize]) -> bool {
    // early exit for too few gates
    if convex_gate_ids.len() < 2 {
        return false;
    }

    let mut is_convex = true;

    // track gates outside the convex set that interfere with its paths
    let mut colliding_set = vec![];
    let mut path_colliding_targets = vec![false; num_wires];
    let mut path_colliding_controls = vec![false; num_wires];

    // iterate through gates between first and last of convex set
    'outer: for i in convex_gate_ids[0]..=*convex_gate_ids.last().unwrap() {
        if convex_gate_ids.contains(&i) {
            // gate is inside convex set
            let selected_gate = circuit.gates[i];

            // check no collision with colliding_set
            for c_gate in colliding_set.iter() {
                if Gate::collides_index(&selected_gate, &c_gate) {
                    is_convex = false;
                    break 'outer;
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
                colliding_set.push(g.clone());
                path_colliding_targets[t as usize] = true;
                path_colliding_controls[c0 as usize] = true;
                path_colliding_controls[c1 as usize] = true;
            }
        }
    }

    is_convex
}

// Samples a contiguous subcircuit
pub fn find_random_subcircuit<R: Rng>(
    circuit: &CircuitSeq,
    min_wires: usize,
    max_wires: usize,
    rng: &mut R,
) -> (usize, usize) {
    let m = circuit.gates.len();
    assert!(m > 0, "Circuit must have at least one gate");

    loop {
        let start_idx = rng.random_range(0..m);
        let mut used_wires = HashSet::new();
        let mut end_idx = start_idx;

        for i in start_idx..m {
            let gate = &circuit.gates[i];
            let mut new_wires = used_wires.clone();
            for &w in gate {
                new_wires.insert(w);
            }

            if new_wires.len() > max_wires {
                break;
            }

            used_wires = new_wires;
            end_idx = i;
        }

        let num_gates = end_idx - start_idx + 1;
        if num_gates >= 3 && used_wires.len() >= min_wires {
            return (start_idx, end_idx);
        }
        // retry, maybe only try some number of times
    }
}

// Given a circuit of num_wires, we try to find a convex subcircuit of up to max_wires. We can start in any of the min_candidates
pub fn find_convex_subcircuit<R: RngCore>(
    _set_size: usize,
    max_wires: usize,
    num_wires: usize,
    circuit: &CircuitSeq,
    rng: &mut R,
) -> (Vec<usize>, usize) {
    let circuit = circuit.clone();
    let num_gates = circuit.gates.len();
    let mut search_attempts = 0;
    let max_attempts = 3;
    let window = 200;
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
        let mut curr_wires = HashSet::new();
        curr_wires.extend(circuit.gates[selected_gate_idx[0]].iter().copied());
        
        while selected_gate_ctr < len {
            let mut candidates: Vec<usize> = vec![];

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
                        let mut collides_with_prev_selected = false;
                        let mut repeat_wires = false;

                        for i in 0..selected_gates_seen {
                            if Gate::collides_index(
                                &curr_gate,
                                &circuit.gates[selected_gate_idx[i]],
                            ) {
                                collides_with_prev_selected = true;
                                break;
                            }
                        }
                        // Optional condition to not allow repeat gates
                        for i in 0..selected_gate_ctr {
                            if curr_gate == circuit.gates[selected_gate_idx[i]] {
                                repeat_wires = true;
                                break;
                            }
                        }

                        // Keep track of indirect collisions
                        // Indirect collisions are gates that collide with a gate that may not have collided with a gate in our list, but a gate that we have already scanned
                        // Ensures that we don't add a gate that would break convexity as it could follow the rules imposed by our chosen gates, but not the overall circuit
                        let [t, c1, c2] = curr_gate;
                        let indirect_path_connected = path_connected_control_wires.wire_hit(t as usize)
                            || path_connected_target_wires.wire_hit(c1 as usize)
                            || path_connected_target_wires.wire_hit(c2 as usize);

                        if collides_with_prev_selected || indirect_path_connected {
                            path_connected_target_wires.add_wire(t as usize);
                            path_connected_control_wires.add_wire(c1 as usize);
                            path_connected_control_wires.add_wire(c2 as usize);

                            let num_new_wires = curr_gate
                                .iter()
                                .filter(|&w| !curr_wires.contains(w))
                                .count();

                            if !indirect_path_connected
                                && !repeat_wires
                                && curr_wires.len() + num_new_wires <= max_wires
                            {
                                candidates.push(curr_idx);
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
                        let mut collides_with_prev_selected = false;
                        let mut repeat_wires = false;

                        for i in 0..selected_gates_seen {
                            if Gate::collides_index(
                                &curr_gate,
                                &circuit.gates[selected_gate_idx[selected_gate_ctr - 1 - i]],
                            ) {
                                collides_with_prev_selected = true;
                                break;
                            }
                        }
                        for i in 0..selected_gate_ctr {
                            if curr_gate == circuit.gates[selected_gate_idx[i]] {
                                repeat_wires = true;
                                break;
                            }
                        }

                        let [t, c1, c2] = curr_gate;
                        let indirect_path_connected = path_connected_control_wires.wire_hit(t as usize)
                            || path_connected_target_wires.wire_hit(c1 as usize)
                            || path_connected_target_wires.wire_hit(c2 as usize);

                        if collides_with_prev_selected || indirect_path_connected {
                            path_connected_target_wires.add_wire(t as usize);
                            path_connected_control_wires.add_wire(c1 as usize);
                            path_connected_control_wires.add_wire(c2 as usize);

                            let num_new_wires = curr_gate
                                .iter()
                                .filter(|&w| !curr_wires.contains(w))
                                .count();

                            if !indirect_path_connected
                                && !repeat_wires
                                && curr_wires.len() + num_new_wires <= max_wires
                            {
                                candidates.push(curr_idx);
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

            // check if adding this gate would exceed max_wires
            let mut new_wires = curr_wires.clone();
            new_wires.extend(circuit.gates[next_candidate].iter().copied());
            if new_wires.len() > max_wires {
                break; // stop expansion if wire limit exceeded
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
            curr_wires = new_wires;
        }

        if selected_gate_ctr < 3 {
            continue;
        }

        if !is_convex(num_wires, &circuit, &selected_gate_idx[..selected_gate_ctr]) {
            continue;
        }

        // println!(
        //     "convex subcircuit found! {} wires {} gates",
        //     curr_wires.len(),
        //     selected_gate_ctr
        // );
        return (selected_gate_idx[..selected_gate_ctr].to_vec(), search_attempts);
    }
}

// Same as above but instead of scanning an entire candidate list, just take the first candidate from the left and right and choose randomly from those two
pub fn simple_find_convex_subcircuit<R: RngCore>(
    _set_size: usize,
    max_wires: usize,
    num_wires: usize,
    circuit: &CircuitSeq,
    rng: &mut R,
) -> (Vec<usize>, usize) {
    let circuit = circuit.clone();
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
        let mut curr_wires = HashSet::new();
        curr_wires.extend(circuit.gates[selected_gate_idx[0]].iter().copied());
        
        while selected_gate_ctr < len {
            let mut candidates: Vec<usize> = vec![];

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

                    if curr_idx == selected_gate_idx[selected_gates_seen]
                    {
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
                        let indirect_path_connected = path_connected_control_wires.wire_hit(t as usize)
                            || path_connected_target_wires.wire_hit(c1 as usize)
                            || path_connected_target_wires.wire_hit(c2 as usize);

                        if collides_with_prev_selected || indirect_path_connected {
                            path_connected_target_wires.add_wire(t as usize);
                            path_connected_control_wires.add_wire(c1 as usize);
                            path_connected_control_wires.add_wire(c2 as usize);

                            let num_new_wires = curr_gate
                                .iter()
                                .filter(|&w| !curr_wires.contains(w))
                                .count();

                            if !indirect_path_connected
                                && !repeat_wires
                                && curr_wires.len() + num_new_wires <= max_wires
                            {
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
                        let indirect_path_connected = path_connected_control_wires.wire_hit(t as usize)
                            || path_connected_target_wires.wire_hit(c1 as usize)
                            || path_connected_target_wires.wire_hit(c2 as usize);

                        if collides_with_prev_selected || indirect_path_connected {
                            path_connected_target_wires.add_wire(t as usize);
                            path_connected_control_wires.add_wire(c1 as usize);
                            path_connected_control_wires.add_wire(c2 as usize);

                            let num_new_wires = curr_gate
                                .iter()
                                .filter(|&w| !curr_wires.contains(w))
                                .count();

                            if !indirect_path_connected
                                && !repeat_wires
                                && curr_wires.len() + num_new_wires <= max_wires
                            {
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

            // check if adding this gate would exceed max_wires
            let mut new_wires = curr_wires.clone();
            new_wires.extend(circuit.gates[next_candidate].iter().copied());
            if new_wires.len() > max_wires {
                break; // stop expansion if wire limit exceeded
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
            curr_wires = new_wires;
        }

        if selected_gate_ctr < 3 {
            continue;
        }

        if !is_convex(num_wires, &circuit, &selected_gate_idx[..selected_gate_ctr]) {
            continue;
        }

        // println!(
        //     "convex subcircuit found! {} wires {} gates",
        //     curr_wires.len(),
        //     selected_gate_ctr
        // );
        return (selected_gate_idx[..selected_gate_ctr].to_vec(), search_attempts);
    }
}

// Instead of choosing a random candidate, prioritize candidates that add less wires
// Hope to get "deeper" subcircuits
pub fn find_convex_subcircuit_deep<R: RngCore>(
    _set_size: usize,
    max_wires: usize,
    num_wires: usize,
    circuit: &CircuitSeq,
    rng: &mut R,
) -> (Vec<usize>, usize) {
    let circuit = circuit.clone();
    let num_gates = circuit.gates.len();
    let mut search_attempts = 0;
    let max_attempts = 3;
    let window = 200;

    loop {
        search_attempts += 1;
        if search_attempts > max_attempts {
            return (vec![], search_attempts);
        }

        let len = circuit.gates.len();
        let mut selected_gate_idx = vec![0; len];
        selected_gate_idx[0] = rng.random_range(0..num_gates);
        let mut selected_gate_ctr = 1;

        let mut curr_wires = HashSet::new();
        curr_wires.extend(circuit.gates[selected_gate_idx[0]].iter().copied());

        while selected_gate_ctr < len {
            let mut candidates: Vec<(usize, usize)> = Vec::new(); // (gate_idx, num_new_wires)

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

                    if curr_idx == selected_gate_idx[selected_gates_seen]
                    {
                        selected_gates_seen += 1;
                        continue;
                    }

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

                        let num_new_wires = curr_gate
                            .iter()
                            .filter(|&w| !curr_wires.contains(w))
                            .count();

                        if !indirect && !repeat && curr_wires.len() + num_new_wires <= max_wires {
                            candidates.push((curr_idx, num_new_wires));
                        }
                    }
                }
            }

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
                        continue;
                    }

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

                        let num_new_wires = curr_gate
                            .iter()
                            .filter(|&w| !curr_wires.contains(w))
                            .count();

                        if !indirect && !repeat && curr_wires.len() + num_new_wires <= max_wires {
                            candidates.push((curr_idx, num_new_wires));
                        }
                    }
                }
            }

            if candidates.is_empty() {
                break;
            }

            let min_new = candidates.iter().map(|(_, w)| *w).min().unwrap();

            let tied: Vec<usize> = candidates
                .iter()
                .filter(|(_, w)| *w == min_new)
                .map(|(idx, _)| *idx)
                .filter(|idx| !selected_gate_idx[..selected_gate_ctr].contains(idx))
                .collect();

            if tied.is_empty() {
                break;
            }

            let next_candidate = *tied.choose(rng).unwrap();

            let mut new_wires = curr_wires.clone();
            new_wires.extend(circuit.gates[next_candidate].iter().copied());
            if new_wires.len() > max_wires {
                break;
            }

            let mut insert_pos = selected_gate_ctr;
            while insert_pos > 0 && selected_gate_idx[insert_pos - 1] > next_candidate {
                selected_gate_idx[insert_pos] = selected_gate_idx[insert_pos - 1];
                insert_pos -= 1;
            }
            selected_gate_idx[insert_pos] = next_candidate;
            selected_gate_ctr += 1;

            curr_wires = new_wires;
        }

        if selected_gate_ctr < 3 {
            continue;
        }

        if !is_convex(num_wires, &circuit, &selected_gate_idx[..selected_gate_ctr]) {
            continue;
        }

        return (selected_gate_idx[..selected_gate_ctr].to_vec(), search_attempts);
    }
}

// Same subcircuit algorithm as the first, but now we decide which gate to start from
pub fn targeted_convex_subcircuit<R: RngCore>(
    set_size: usize,
    max_wires: usize,
    num_wires: usize,
    circuit: &CircuitSeq,
    rng: &mut R,
    target: usize,
) -> (Vec<usize>, usize) {
    let num_gates = circuit.gates.len();
    let search_attempts = 0;
    let window = 200;
    
    // Start with one random gate
    let mut selected_gate_idx = vec![0; set_size];
    selected_gate_idx[0] = target;
    let mut selected_gate_ctr = 1;

    // Initialize wire set
    let mut curr_wires = HashSet::new();
    curr_wires.extend(circuit.gates[selected_gate_idx[0]].iter().copied());

    while selected_gate_ctr < set_size {
        let mut candidates: Vec<usize> = vec![];

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
                    let mut collides_with_prev_selected = false;
                    let mut repeat_wires = false;

                    for i in 0..selected_gates_seen {
                        if Gate::collides_index(
                            &curr_gate,
                            &circuit.gates[selected_gate_idx[i]],
                        ) {
                            collides_with_prev_selected = true;
                            break;
                        }
                    }
                    for i in 0..selected_gate_ctr {
                        if curr_gate == circuit.gates[selected_gate_idx[i]] {
                            repeat_wires = true;
                            break;
                        }
                    }

                    let [t, c1, c2] = curr_gate;
                    let indirect_path_connected = path_connected_control_wires.wire_hit(t as usize)
                        || path_connected_target_wires.wire_hit(c1 as usize)
                        || path_connected_target_wires.wire_hit(c2 as usize);

                    if collides_with_prev_selected || indirect_path_connected {
                        path_connected_target_wires.add_wire(t as usize);
                        path_connected_control_wires.add_wire(c1 as usize);
                        path_connected_control_wires.add_wire(c2 as usize);

                        let num_new_wires = curr_gate
                            .iter()
                            .filter(|&w| !curr_wires.contains(w))
                            .count();

                        if !indirect_path_connected
                            && !repeat_wires
                            && curr_wires.len() + num_new_wires <= max_wires
                        {
                            candidates.push(curr_idx);
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
                    let mut collides_with_prev_selected = false;
                    let mut repeat_wires = false;

                    for i in 0..selected_gates_seen {
                        if Gate::collides_index(
                            &curr_gate,
                            &circuit.gates[selected_gate_idx[selected_gate_ctr - 1 - i]],
                        ) {
                            collides_with_prev_selected = true;
                            break;
                        }
                    }
                    for i in 0..selected_gate_ctr {
                        if curr_gate == circuit.gates[selected_gate_idx[i]] {
                            repeat_wires = true;
                            break;
                        }
                    }

                    let [t, c1, c2] = curr_gate;
                    let indirect_path_connected = path_connected_control_wires.wire_hit(t as usize)
                        || path_connected_target_wires.wire_hit(c1 as usize)
                        || path_connected_target_wires.wire_hit(c2 as usize);

                    if collides_with_prev_selected || indirect_path_connected {
                        path_connected_target_wires.add_wire(t as usize);
                        path_connected_control_wires.add_wire(c1 as usize);
                        path_connected_control_wires.add_wire(c2 as usize);

                        let num_new_wires = curr_gate
                            .iter()
                            .filter(|&w| !curr_wires.contains(w))
                            .count();

                        if !indirect_path_connected
                            && !repeat_wires
                            && curr_wires.len() + num_new_wires <= max_wires
                        {
                            candidates.push(curr_idx);
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

        // check if adding this gate would exceed max_wires
        let mut new_wires = curr_wires.clone();
        new_wires.extend(circuit.gates[next_candidate].iter().copied());
        if new_wires.len() > max_wires {
            break; // stop expansion if wire limit exceeded
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
        curr_wires = new_wires;
    }

    if selected_gate_ctr < 3 {
        return (vec![], 0)
    }

    if !is_convex(num_wires, circuit, &selected_gate_idx[..selected_gate_ctr]) {
        return (vec![], 0)
    }

    // println!(
    //     "convex subcircuit found! {} wires {} gates",
    //     curr_wires.len(),
    //     selected_gate_ctr
    // );
    return (selected_gate_idx[..selected_gate_ctr].to_vec(), search_attempts);
}

pub fn targeted_find_convex_subcircuit_deep<R: RngCore>(
    _set_size: usize,
    max_wires: usize,
    num_wires: usize,
    circuit: &CircuitSeq,
    rng: &mut R,
    target: usize,
) -> (Vec<usize>, usize) {
    let circuit = circuit.clone();
    let num_gates = circuit.gates.len();
    let mut search_attempts = 0;
    let max_attempts = 3;
    let window = 200;

    loop {
        search_attempts += 1;
        if search_attempts > max_attempts {
            return (vec![], search_attempts);
        }

        let len = circuit.gates.len();
        let mut selected_gate_idx = vec![0; len];
        selected_gate_idx[0] = target;
        let mut selected_gate_ctr = 1;

        let mut curr_wires = HashSet::new();
        curr_wires.extend(circuit.gates[selected_gate_idx[0]].iter().copied());

        while selected_gate_ctr < len {
            let mut candidates: Vec<(usize, usize)> = Vec::new(); // (gate_idx, num_new_wires)

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

                    if curr_idx == selected_gate_idx[selected_gates_seen]
                    {
                        selected_gates_seen += 1;
                        continue;
                    }

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

                        let num_new_wires = curr_gate
                            .iter()
                            .filter(|&w| !curr_wires.contains(w))
                            .count();

                        if !indirect && !repeat && curr_wires.len() + num_new_wires <= max_wires {
                            candidates.push((curr_idx, num_new_wires));
                        }
                    }
                }
            }

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
                        continue;
                    }

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

                        let num_new_wires = curr_gate
                            .iter()
                            .filter(|&w| !curr_wires.contains(w))
                            .count();

                        if !indirect && !repeat && curr_wires.len() + num_new_wires <= max_wires {
                            candidates.push((curr_idx, num_new_wires));
                        }
                    }
                }
            }

            if candidates.is_empty() {
                break;
            }

            let min_new = candidates.iter().map(|(_, w)| *w).min().unwrap();

            let tied: Vec<usize> = candidates
                .iter()
                .filter(|(_, w)| *w == min_new)
                .map(|(idx, _)| *idx)
                .filter(|idx| !selected_gate_idx[..selected_gate_ctr].contains(idx))
                .collect();

            if tied.is_empty() {
                break;
            }

            let next_candidate = *tied.choose(rng).unwrap();

            let mut new_wires = curr_wires.clone();
            new_wires.extend(circuit.gates[next_candidate].iter().copied());
            if new_wires.len() > max_wires {
                break;
            }

            let mut insert_pos = selected_gate_ctr;
            while insert_pos > 0 && selected_gate_idx[insert_pos - 1] > next_candidate {
                selected_gate_idx[insert_pos] = selected_gate_idx[insert_pos - 1];
                insert_pos -= 1;
            }
            selected_gate_idx[insert_pos] = next_candidate;
            selected_gate_ctr += 1;

            curr_wires = new_wires;
        }

        if selected_gate_ctr < 3 {
            continue;
        }

        if !is_convex(num_wires, &circuit, &selected_gate_idx[..selected_gate_ctr]) {
            continue;
        }

        return (selected_gate_idx[..selected_gate_ctr].to_vec(), search_attempts);
    }
}

// Rearranges circuit to put the convex subcircuit in a contiguous manner. Do this via outward expansion
pub fn contiguous_convex(
    circuit: &mut CircuitSeq,
    ordered_convex_gates: &mut Vec<usize>,
    num_wires: usize
) -> Option<(usize, usize)> {
    // This should never run
    if ordered_convex_gates.len() < 2 {
        return None;
    }

    if !is_convex(num_wires, circuit, &ordered_convex_gates) {
        panic!("not convex");
    }

    // Keep track of convex positions
    let mut is_convex = vec![false; circuit.gates.len()];
    for &idx in ordered_convex_gates.iter() {
        is_convex[idx] = true;
    }

    // Bubble boundaries
    let mut start = *ordered_convex_gates.first().unwrap();
    let mut end = *ordered_convex_gates.last().unwrap();

    let mut non_convex: Vec<usize> = (start..=end)
        .filter(|&i| !is_convex[i])
        .collect();

    // Left pass
    while !non_convex.is_empty() {
        let leftmost = non_convex[0];
        if leftmost <= start {
            break;
        }

        let can_shift = (start..leftmost)
            .all(|i| !Gate::collides_index(&circuit.gates[i], &circuit.gates[leftmost]));

        if can_shift {
            let gate = circuit.gates.remove(leftmost);
            circuit.gates.insert(start, gate);

            for idx in ordered_convex_gates.iter_mut() {
                if *idx >= start && *idx < leftmost {
                    *idx += 1;
                }
            }
            for i in 0..non_convex.len() {
                if non_convex[i] >= start && non_convex[i] < leftmost {
                    panic!("This shouldn't be possible");
                }
            }
            start += 1;
            non_convex.remove(0);
        } else {
            break;
        }
    }

    // Right pass
    while !non_convex.is_empty() {
        let rightmost = *non_convex.last().unwrap();
        if rightmost >= end {
            break;
        }

        let can_shift = ((rightmost + 1)..=end)
            .all(|i| !Gate::collides_index(&circuit.gates[i], &circuit.gates[rightmost]));

        if can_shift {
            let gate = circuit.gates.remove(rightmost);
            circuit.gates.insert(end, gate);

            for idx in ordered_convex_gates.iter_mut() {
                if *idx > rightmost && *idx <= end {
                    *idx -= 1;
                }
            }
            for i in 0..non_convex.len() {
                if non_convex[i] > rightmost && non_convex[i] <= end {
                    panic!("Right should be possible either");
                }
            }
            end -= 1;
            non_convex.pop();
        } else {
            break;
        }
    }

    Some((start, end))
}

// Shoots a random gate left or right without collisions
pub fn shoot_random_gate(circuit: &mut CircuitSeq, rounds: usize) {
    let mut rng = rand::rng();
    let len = circuit.gates.len();

    if len == 0 {
        return
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

pub fn random_sulking(circuit: &mut CircuitSeq) {
    let mut rng = rand::rng();
    let len = circuit.gates.len();

    if len == 0 {
        return
    }
    let mut out: Vec<[u8;3]> = Vec::new();
    for gate_idx in 0..len{
        // Shoot left
        let mut target = gate_idx;
        out.push(circuit.gates[target]);
        if gate_idx == 0 {
            continue
        } else {
            while target > 0 {
                if Gate::collides_index(&out[target - 1], &out[gate_idx]) {
                    break;
                }
                target -= 1;
            }
            target = rng.random_range(target..=gate_idx);
            if target != gate_idx {
                let gate = out.pop().unwrap();
                out.insert(target, gate);
            }
        }
    }
    println!("{}", circuit.gates == out);
    circuit.gates = out;
}

pub fn shoot_random_gate_gate_ver(circuit: &mut Vec<[u8;3]>, rounds: usize) {
    let mut rng = rand::rng();
    let len = circuit.len();

    if len == 0 {
        return
    }

    for _ in 0..rounds {
        let gate_idx = rng.random_range(0..len);
        let go_left: bool = rng.random_bool(0.5);

        if go_left {
            // Shoot left
            let mut target = gate_idx;
            while target > 0 {
                if Gate::collides_index(&circuit[target - 1], &circuit[gate_idx]) {
                    break;
                }
                target -= 1;
            }
            target = rng.random_range(target..=gate_idx);
            if target != gate_idx {
                let gate = circuit.remove(gate_idx);
                circuit.insert(target, gate);
            }
        } else {
            // Shoot right
            let mut target = gate_idx;
            while target + 1 < len {
                if Gate::collides_index(&circuit[target + 1], &circuit[gate_idx]) {
                    break;
                }
                target += 1;
            }
            target = rng.random_range(gate_idx..=target);
            if target != gate_idx {
                let gate = circuit.remove(gate_idx);
                circuit.insert(target, gate);
            }
        }
    }
}

pub fn shoot_left_vec(circuit: &mut Vec<[u8;3]>, gate_idx: usize) -> usize { 
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

pub fn shoot_left_vec_track(circuit: &mut Vec<([u8;3], u8)>, gate_idx: usize, max: bool) -> usize { 
    let mut target = gate_idx;
    let mut rng = rand::rng();
    while target > 0 {
        if Gate::collides_index(&circuit[target - 1].0, &circuit[gate_idx].0) {
            break;
        }
        target -= 1;
    }

    if target != gate_idx {
        let (gate, _) = circuit.remove(gate_idx);
        if !max {
            target = rng.random_range(target..=gate_idx);
        }
        let update = if !max {4} else {0};
        circuit.insert(target, (gate, update));
    }

    target
}

pub fn shoot_right_vec(circuit: &mut Vec<[u8;3]>, gate_idx: usize) -> usize { 
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

pub fn shoot_right_vec_track(circuit: &mut Vec<([u8;3], u8)>, gate_idx: usize, max: bool) -> usize { 
    let mut target = gate_idx;
    let len = circuit.len();
    let mut rng = rand::rng();
    while target + 1 < len {
        if Gate::collides_index(&circuit[target + 1].0, &circuit[gate_idx].0) {
            break;
        }
        target += 1;
    }
    if target != gate_idx {
        let (gate, _) = circuit.remove(gate_idx);
        if !max {
            target = rng.random_range(gate_idx..=target);
        }
        let update = if !max {5} else {0};
        circuit.insert(target, (gate, update));
    }

    target
}

// true is shoot left goes all the way to the beginning of the circuit
// In other words, the gate commutes with all the earlier gates and is hence, level zero on the skeleton graph
pub fn is_level_zero(circuit: &CircuitSeq, index: usize) -> bool {
    let mut target = index;
    while target > 0 {
        if Gate::collides_index(&circuit.gates[target - 1], &circuit.gates[index]) {
            break;
        }
        target -= 1;
    }

    target == 0
}

// Assist in creating random_walking
pub fn left_ordering(circuit: &CircuitSeq) -> CircuitSeq{
    let mut circuit = circuit.clone();
    circuit.canonicalize();
    let mut new_gates: Vec<[u8;3]> = Vec::new();
    let mut c = circuit.clone();
    while !c.gates.is_empty() {
        let mut to_remove: Vec<usize> = Vec::new();
        for i in (0..c.gates.len()).rev() {
            if is_level_zero(&c, i) {
                to_remove.push(i);
                new_gates.push(c.gates[i]);
            }
        }
        for i in &to_remove {
            c.gates.remove(*i);
        }
    }
    let new = CircuitSeq { gates: new_gates };

    if new.probably_equal(&circuit, 64, 100000).is_err() {
        panic!("Left shooting changed functionality");
    }
    
    new
}

// Below used to help construct a skeleton graph representation of a circuit
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Node {
    key: usize,
    val: [u8; 3],
    parents: Vec<usize>,
    children: Vec<usize>,
    level: usize,
}

#[derive(Clone, Debug, Default)]
pub struct Skeleton {
    nodes: Vec<Vec<Node>>,
    depth: usize,
}

pub fn create_skeleton(circuit: &CircuitSeq) -> (CircuitSeq, Skeleton) {
    let c = left_ordering(&circuit);
    let gates = &c.gates;
    let mut skel = Skeleton { nodes: Vec::new(), depth: 0 };
    let mut start = 0;
    let mut level = 0;

    while start < gates.len() {
        let mut segment = Vec::new();
        let mut i = start;
        while i < gates.len() {
            if i > start && segment.iter().any(|&(_, g)| Gate::collides_index(&gates[i], &g)) {
                break;
            }
            segment.push((i, gates[i].clone()));
            i += 1;
        }

        let mut level_nodes: Vec<Node> = segment
            .iter()
            .map(|(idx, gate)| Node {
                key: *idx,
                val: *gate,
                parents: Vec::new(),
                children: Vec::new(),
                level,
            })
            .collect();

        if level > 0 {
            for node in &mut level_nodes {
                for prev_level in 0..level {
                    for prev_node in &mut skel.nodes[prev_level] {
                        if Gate::collides_index(&prev_node.val, &node.val) {
                            node.parents.push(prev_node.key);
                            prev_node.children.push(node.key);
                        }
                    }
                }
            }
        }

        skel.nodes.push(level_nodes);
        level += 1;
        start = i;
    }

    skel.depth = level;
    (c, skel)
}

// Supposed to be a more random version of random_shooting
// Not shown to be much more effective and so not used at the moment
pub fn random_walking<R: RngCore>(circuit: &CircuitSeq, rng: &mut R) -> CircuitSeq {
    let orig_circuit = circuit.clone();
    let (circuit, skeleton) = create_skeleton(&circuit);

    let mut new_gates = CircuitSeq { gates: Vec::new() };

    // Build a map from key -> Node for easy lookup
    let mut node_map: HashMap<usize, Node> = HashMap::new();
    for level in &skeleton.nodes {
        for node in level {
            node_map.insert(node.key, node.clone());
        }
    }

    // Keep track of nodes not yet added
    let mut remaining_keys: HashSet<usize> = node_map.keys().cloned().collect();

    // Start with level 0 nodes
    let mut candidates: Vec<Node> = skeleton.nodes[0].clone();

    while !candidates.is_empty() {
        // Pick a random candidate
        let idx = rng.random_range(0..candidates.len());
        let next = candidates.swap_remove(idx);
        remaining_keys.remove(&next.key);

        // Add the gate to the new circuit
        new_gates.gates.push(circuit.gates[next.key]);

        // Process children
        for &child_key in &next.children {
            if !remaining_keys.contains(&child_key) {
                continue; // already added
            }
            let child = &node_map[&child_key];

            // Add to candidates if all parents have been added
            if child.parents.iter().all(|p| !remaining_keys.contains(p)) {
                candidates.push(child.clone());
            }
        }
    }

    // Sanity checks
    if new_gates.gates.len() != orig_circuit.gates.len() {
        panic!("Didn't add enough gates!");
    }

    if new_gates.probably_equal(&orig_circuit, 64, 100_000).is_err() {
        panic!("Circuit functionality changed!");
    }

    new_gates
}

// Random walking algorithm without needing to reconstruct the skeleton graph each time
pub fn random_walk_no_skeleton<R: RngCore>(
    circuit: &CircuitSeq,
    rng: &mut R,
) -> CircuitSeq {
    let n = circuit.gates.len();
    let mut remaining: Vec<bool> = vec![true; n];
    let mut in_candidates: Vec<bool> = vec![false; n];
    let mut out = Vec::with_capacity(n);

    let mut candidates: Vec<usize> = Vec::new();

    for i in 0..n {
        if is_level_zero_raw(circuit, i, &remaining) {
            candidates.push(i);
            in_candidates[i] = true;
        }
    }

    while !candidates.is_empty() {
        let idx = rng.random_range(0..candidates.len());
        let g = candidates.swap_remove(idx);
        in_candidates[g] = false;

        out.push(circuit.gates[g]);
        remaining[g] = false;

        for j in (g + 1)..n {
            if remaining[j] && !in_candidates[j] && is_level_zero_raw(circuit, j, &remaining) {
                candidates.push(j);
                in_candidates[j] = true;
            }
        }
    }

    CircuitSeq { gates: out }
}

fn is_level_zero_raw(c: &CircuitSeq, idx: usize, remaining: &[bool]) -> bool {
    let gate = &c.gates[idx];
    for i in 0..idx {
        if remaining[i] && Gate::collides_index(&c.gates[i], gate) {
            return false;
        }
    }
    true
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Below is used for db storing
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

pub fn create_table(conn: &Connection, table_name: &str) -> duckdb::Result<()> {
    // Table name includes n and m
    let sql = format!(
        "CREATE TABLE IF NOT EXISTS {table} (
            circuit BLOB UNIQUE,
            perm BLOB NOT NULL,
            shuf BLOB NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_circuit_{table} ON {table} (circuit);
        CREATE INDEX IF NOT EXISTS idx_perm_{table} ON {table} (perm);",
        table = table_name
    );

    conn.execute_batch(&sql)?;
    Ok(())
}

pub fn insert_circuit(
    conn: &Connection,
    circuit: &CircuitSeq, 
    canon: &Canonicalization,
    table_name: &str,
) -> duckdb::Result<()> {
    let key = circuit.repr_blob();
    let perm = canon.perm.repr_blob();
    let shuf = canon.shuffle.repr_blob();
    let sql = format!("INSERT OR IGNORE INTO {} (circuit, perm, shuf) VALUES ($1, $2, $3)", table_name);
    conn.execute(&sql, duckdb::params![key.as_slice(), perm.as_slice(), shuf.as_slice()])?;
    Ok(())
}

pub fn insert_circuits_batch(
    conn: &mut Connection,
    table_name: &str,
    circuits: &[(CircuitSeq, Canonicalization)],
) -> duckdb::Result<usize> {
    let tx = conn.transaction()?;

    let sql = format!(
        "INSERT OR IGNORE INTO {} (circuit, perm, shuf) VALUES ($1, $2, $3)",
        table_name
    );

    let mut inserted = 0;

    for (circuit, canon) in circuits {
        let key = circuit.repr_blob();
        let perm = canon.perm.repr_blob();
        let shuf = canon.shuffle.repr_blob();

        if tx.execute(&sql, duckdb::params![key.as_slice(), perm.as_slice(), shuf.as_slice()])? > 0 {
            inserted += 1;
        }
    }

    tx.commit()?;

    Ok(inserted)
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// 1) Try to use the cache to get permutation canonicalization
// 2) If fails, try to use fast method
// 3) If fails, use brute force
pub fn get_canonical(perm: &Permutation, bit_shuf: &Vec<Vec<usize>>) -> Canonicalization {
    // Use a simple hash of the subcircuit as the key
    let key = perm.repr_blob(); 

    // Try to get it from the cache
    if let Some(cached) = CANON_CACHE.get(&key) {
        let (perm_blob, shuffle_blob) = &*cached;
        return Canonicalization {
            perm: Permutation::from_blob(perm_blob),
            shuffle: Permutation::from_blob(shuffle_blob),
        };
    }

    // compute it
    let canon = perm.canon_simple(bit_shuf);

    // Store 
    CANON_CACHE.insert(key, (canon.clone().perm.repr_blob(), canon.clone().shuffle.repr_blob()));
    canon
}

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

                // compute minimum comparison result for THIS r
                let mut _is_better = false;
                for weight in 0..=num_b / 2 {
                    for idx in canonical::index_set(weight, num_b) {
                        let p_val = unsafe { *perm_shuf.get_unchecked(idx) };
                        let m_val = p_val; // we'll compare p_val elsewhere
                        // We return perm_shuf unconditionally;
                        // comparison happens globally.
                        if p_val < m_val {
                            _is_better = true;
                        }
                        break;
                    }
                    break;
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

        for weight in 0..=num_bits/2 {
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
                        }
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
            .map(|x| x.trim().parse::<usize>().expect("Invalid number in permutation"))
            .collect();

        Permutation { data }
    }
}

// Testing code to look at sql db
// sql db is mostly unused, outside of n6m5 and n7m4
// Switched over to lmdb
pub fn check_cycles(n: usize, m: usize) -> duckdb::Result<()> {
    // Open the database
    let config = Config::default().access_mode(AccessMode::ReadOnly).unwrap();
    let conn = Connection::open_with_flags("circuits.duckdb", config).unwrap();
    let table_name = format!("n{}m{}", n, m);

    // Build the query string with the table name
    let query = format!("SELECT DISTINCT perm FROM {}", table_name);
    let mut stmt = conn.prepare(&query)?;

    // Query all distinct perms
    let perm_iter = stmt.query_map([], |row| {
        let perm_str: Vec<u8> = row.get(0)?; // now as String
        Ok(perm_str)
    })?;

    println!("Distinct permutations in {}:", table_name);

    for perm_str_result in perm_iter {
        let perm_str = perm_str_result?;

        // Convert the string into a Permutation
        let perm = Permutation::from_blob(&perm_str);
        let cycles = perm;

        println!("{:?}", cycles);
    }

    Ok(())
}

pub fn print_all(table_name: &str) -> duckdb::Result<()> {
    let config = Config::default().access_mode(AccessMode::ReadOnly).unwrap();
    let conn = Connection::open_with_flags("circuits.duckdb", config).unwrap();

    let query = format!("SELECT circuit, perm, shuf FROM {}", table_name);
    let mut stmt = conn.prepare(&query)?;

    let rows = stmt.query_map([], |row| {
        let circuit_blob: Vec<u8> = row.get(0)?;
        let perm_blob: Vec<u8> = row.get(1)?;
        let shuf_blob: Vec<u8> = row.get(2)?;

        Ok((circuit_blob, perm_blob, shuf_blob))
    })?;

    for row in rows {
        let (circuit_blob, perm_blob, shuf_blob) = row?;

        let circuit = CircuitSeq::from_blob(&circuit_blob);
        let perm = Permutation::from_blob(&perm_blob);
        let shuf = Permutation::from_blob(&shuf_blob);

        println!("Circuit: {:?}", circuit.gates);
        println!("Perm:    {:?}", perm.data);
        println!("Shuf:    {:?}", shuf.data);
        println!();
    }

    Ok(())
}

pub fn count_distinct(n: usize, m: usize) -> duckdb::Result<usize> {
    let config = Config::default().access_mode(AccessMode::ReadOnly).unwrap();
    let conn = Connection::open_with_flags("circuits.duckdb", config).unwrap();
    let table_name = format!("n{}m{}", n, m);
    
    let query = format!("SELECT COUNT(DISTINCT perm) FROM {}", table_name);
    let count: usize = conn.query_row(&query, [], |row| row.get(0))?;
    
    println!("Number of distinct permutations in {}: {}", table_name, count);
    Ok(count)
}

pub fn base_gates(n: usize) -> Vec<[u8; 3]> {
    let n = n as u8;
    let mut gates: Vec<[u8;3]> = Vec::new();
    for a in 0..n {
        for b in 0..n {
            if b == a { continue; }
            for c in 0..n {
                if c == a || c == b { continue; }
                gates.push([a, b, c]);
            }
        }
    }
    gates
}

// Given nXmY, attempt to build the corresponding table for nXm{Y+1}
pub fn build_from_sql(
    conn: &Connection,
    n: usize,
    m: usize,
    bit_shuf: &Vec<Vec<usize>>,
) -> duckdb::Result<()> {
    println!("Running build (max CPU)");

    let old_table = format!("n{}m{}", n, m - 1);
    let new_table = format!("n{}m{}", n, m);

    create_table(conn, &new_table)?;

    let base_gates: Arc<Vec<[u8; 3]>> = Arc::new(base_gates(n));
    let base_gates_for_thread = Arc::clone(&base_gates);
    let bit_shuf = Arc::new(bit_shuf.clone());

    let total_rows: i64 = conn.query_row(
        &format!("SELECT COUNT(*) FROM {}", old_table),
        [],
        |row| row.get(0),
    )?;
    println!("Total rows in {}: {}", old_table, total_rows);

    let chunk_size: i64 = 50_000;
    let batch_size = 10_000;

    let mut offset: i64 = 0;

    // Atomic flag for CTRL+C
    let stop_flag = Arc::new(AtomicBool::new(false));
    {
        let stop_flag = stop_flag.clone();
        ctrlc::set_handler(move || {
            println!("CTRL+C detected! Finishing current batch...");
            stop_flag.store(true, Ordering::SeqCst);
        })
        .expect("Error setting CTRL+C handler");
    }

    // Setup bounded channel for insertion
    let (tx, rx) = bounded::<Vec<(CircuitSeq, Canonicalization)>>(10_000);
    let new_table_clone = new_table.clone();
    let stop_flag_clone = stop_flag.clone();

    // Spawn insertion thread
    let insert_handle = thread::spawn(move || {
        let mut insert_conn = Connection::open("circuits.duckdb").unwrap();

        let total_circuits: usize = (total_rows as usize) * base_gates_for_thread.len() * 2; // total circuits to process
        let mut attempted_inserts = 0;

        while let Ok(batch) = rx.recv() {
            if stop_flag_clone.load(Ordering::SeqCst) {
                println!("Insertion thread stopping early...");
                break;
            }

            // Attempt insertion (success or not, we count as attempted)
            if let Err(e) = insert_circuits_batch(&mut insert_conn, &new_table_clone, &batch) {
                eprintln!("Error inserting batch: {:?}", e);
            }

            attempted_inserts += batch.len();

            // Print attempted insert progress every batch
            println!(
                "Attempted inserts: {} / {} ({:.2}%)",
                attempted_inserts,
                total_circuits,
                (attempted_inserts as f64 / total_circuits as f64) * 100.0
            );
        }

        println!("Insertion thread finished");
    });

    // Main loop: fetch old table in chunks
    while offset < total_rows {
        if stop_flag.load(Ordering::SeqCst) {
            println!("Stopping early due to CTRL+C...");
            break;
        }

        let rows: Vec<Vec<u8>> = {
            let mut stmt = conn.prepare(&format!(
                "SELECT circuit FROM {} LIMIT $1 OFFSET $2",
                old_table
            ))?;
            stmt.query_map(duckdb::params![chunk_size, offset], |row| {
                row.get(0)
            })?
            .collect::<duckdb::Result<_>>()?
        };

        if rows.is_empty() {
            break;
        }

        offset += rows.len() as i64;

        // Process circuits in parallel and stream batches immediately
        rows.par_chunks(500).for_each(|row_chunk| {
            let mut local_results =
                Vec::with_capacity(row_chunk.len() * base_gates.len() * 2);

            for blob in row_chunk {
                let old_circuit = CircuitSeq::from_blob(blob);
                let mut prefix: SmallVec<[[u8; 3]; 64]> =
                    SmallVec::with_capacity(m);
                prefix.extend_from_slice(&old_circuit.gates);

                for g in base_gates.iter() {
                    let mut q1 = prefix.clone();
                    q1.push(*g);
                    let mut c1 = CircuitSeq { gates: q1.to_vec() };
                    c1.canonicalize();
                    let canon1 = c1.permutation(n).canon_simple(&bit_shuf);

                    let mut q2 = SmallVec::<[[u8; 3]; 64]>::with_capacity(m + 1);
                    q2.push(*g);
                    q2.extend_from_slice(&prefix);
                    let mut c2 = CircuitSeq { gates: q2.to_vec() };
                    c2.canonicalize();
                    let canon2 = c2.permutation(n).canon_simple(&bit_shuf);

                    if !c1.adjacent_id() {
                        local_results.push((c1, canon1));
                    }
                    if !c2.adjacent_id() {
                        local_results.push((c2, canon2));
                    }
                }

                // Stream batches immediately
                while local_results.len() >= batch_size {
                    let batch = local_results.split_off(local_results.len() - batch_size);
                    if let Err(e) = tx.send(batch) {
                        eprintln!("Failed to send batch to insertion thread: {:?}", e);
                        break;
                    }
                }

                if stop_flag.load(Ordering::SeqCst) {
                    break;
                }
            }

            // Send remaining circuits in local_results
            if !local_results.is_empty() {
                if let Err(e) = tx.send(local_results) {
                    eprintln!("Failed to send remaining batch: {:?}", e);
                }
            }
        });

        println!(
            "Processed up to offset {}. Progress: {:.2}%",
            offset,
            (offset as f64 / total_rows as f64) * 100.0
        );

        if stop_flag.load(Ordering::SeqCst) {
            break;
        }
    }

    // Close sender to signal insertion thread to exit
    drop(tx);
    insert_handle.join().expect("Insertion thread panicked");

    println!("Build finished (or stopped early).");
    Ok(())
}

/// Merge operator — appends new circuit blobs to existing list, deduplicating.
/// Value format: [u8 len | blob bytes | ...]
fn append_merge(
    _key: &[u8],
    existing: Option<&[u8]>,
    operands: &MergeOperands,
) -> Option<Vec<u8>> {
    let mut result: Vec<u8> = existing.unwrap_or(&[]).to_vec();

    for operand in operands {
        let mut pos = 0;
        while pos + 1 <= operand.len() {
            let len = operand[pos] as usize;
            pos += 1;
            if pos + len > operand.len() {
                break;
            }
            let new_blob = &operand[pos..pos + len];
            pos += len;

            // Check for duplicate in result
            let mut rpos = 0;
            let mut found = false;
            while rpos + 1 <= result.len() {
                let rlen = result[rpos] as usize;
                rpos += 1;
                if rpos + rlen > result.len() {
                    break;
                }
                if &result[rpos..rpos + rlen] == new_blob {
                    found = true;
                    break;
                }
                rpos += rlen;
            }

            if !found {
                result.push(new_blob.len() as u8);
                result.extend_from_slice(new_blob);
            }
        }
    }

    Some(result)
}

pub fn open_db_for_write(m: usize) -> DB {
    let path = format!("rocks_db_m{}", m);
    let mut opts = Options::default();
    opts.create_if_missing(true);

    opts.set_merge_operator_associative("append_merge", append_merge);

    // Disable WAL for faster bulk ingestion — no recovery needed
    opts.set_manual_wal_flush(true);

    opts.increase_parallelism(160);
    opts.set_max_background_jobs(8);

    opts.set_write_buffer_size(256 * 1024 * 1024);
    opts.set_max_write_buffer_number(4);
    opts.set_min_write_buffer_number_to_merge(2);

    opts.set_level_zero_file_num_compaction_trigger(10);
    opts.set_max_bytes_for_level_base(512 * 1024 * 1024);
    opts.set_max_bytes_for_level_multiplier(10.0);
    opts.set_num_levels(7);

    opts.set_compression_type(DBCompressionType::None);
    opts.set_bottommost_compression_type(DBCompressionType::Zstd);

    // 16 byte prefix for xxHash128
    opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(16));

    let mut block_opts = BlockBasedOptions::default();
    block_opts.set_bloom_filter(10.0, false);
    block_opts.set_block_size(16 * 1024);
    block_opts.set_cache_index_and_filter_blocks(true);
    opts.set_block_based_table_factory(&block_opts);

    DB::open(&opts, path).expect("Failed to open RocksDB for write")
}

pub fn open_db_for_read(m: usize) -> DB {
    let path = format!("rocks_db_m{}", m);
    let mut opts = Options::default();
    opts.create_if_missing(false);

    // Must register merge operator even for reads
    opts.set_merge_operator_associative("append_merge", append_merge);

    opts.increase_parallelism(160);

    opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(16));

    let cache = Cache::new_lru_cache(4 * 1024 * 1024 * 1024);
    let mut block_opts = BlockBasedOptions::default();
    block_opts.set_block_cache(&cache);
    block_opts.set_block_size(16 * 1024);
    block_opts.set_bloom_filter(10.0, false);
    block_opts.set_cache_index_and_filter_blocks(true);
    block_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);
    opts.set_block_based_table_factory(&block_opts);

    opts.set_disable_auto_compactions(true);

    DB::open_for_read_only(&opts, path, false).expect("Failed to open RocksDB for read")
}

/// Encode a single circuit blob as a length-prefixed entry
fn encode_circuit(circuit_blob: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(1 + circuit_blob.len());
    v.push(circuit_blob.len() as u8);
    v.extend_from_slice(circuit_blob);
    v
}

/// Merge duplicate keys in a sorted list, deduplicating circuit blobs
fn merge_sorted_entries(entries: Vec<(Vec<u8>, Vec<u8>)>) -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut merged: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();

    for (key, value) in entries {
        if let Some(last) = merged.last_mut() {
            if last.0 == key {
                // value is [u8 len | blob], extract the blob
                if value.is_empty() {
                    continue;
                }
                let new_len = value[0] as usize;
                if 1 + new_len > value.len() {
                    continue;
                }
                let new_blob = &value[1..1 + new_len];

                // Scan existing blobs for duplicate
                let mut rpos = 0;
                let mut found = false;
                while rpos + 1 <= last.1.len() {
                    let rlen = last.1[rpos] as usize;
                    rpos += 1;
                    if rpos + rlen > last.1.len() {
                        break;
                    }
                    if &last.1[rpos..rpos + rlen] == new_blob {
                        found = true;
                        break;
                    }
                    rpos += rlen;
                }

                if !found {
                    last.1.push(new_len as u8);
                    last.1.extend_from_slice(new_blob);
                }
                continue;
            }
        }
        merged.push((key, value));
    }

    merged
}

fn flush_to_sst(db: &Arc<DB>, pending: &mut Vec<(Vec<u8>, Vec<u8>)>, sst_index: &mut usize) {
    if pending.is_empty() {
        return;
    }

    // Sort by key — required for SST ingestion
    pending.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));

    let merged = merge_sorted_entries(std::mem::take(pending));

    let sst_path = format!("/tmp/sst_{}.sst", sst_index);
    *sst_index += 1;

    let mut opts = Options::default();
    opts.set_merge_operator_associative("append_merge", append_merge);
    opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(16));

    let mut writer = SstFileWriter::create(&opts);
    writer.open(&sst_path).expect("Failed to open SST writer");

    for (key, value) in &merged {
        writer.put(key, value).expect("Failed to write SST entry");
    }
    writer.finish().expect("Failed to finish SST file");

    let mut ingest_opts = IngestExternalFileOptions::default();
    ingest_opts.set_move_files(true);
    db.ingest_external_file_opts(&ingest_opts, vec![sst_path.clone()])
        .expect("Failed to ingest SST file");

    println!("Ingested SST file #{}", *sst_index - 1);
}

/// Returns the set of wires actually touched by the circuit (appearing in any gate).
fn touched_wires(circuit: &CircuitSeq) -> Vec<u8> {
    let mut touched: Vec<u8> = Vec::new();
    for gate in &circuit.gates {
        for &w in gate.iter() {
            if !touched.contains(&w) {
                touched.push(w);
            }
        }
    }
    touched.sort();
    touched
}

/// Expand an abstract gate (possibly containing UNUSED sentinel) into concrete gates
/// by substituting actual unused wires into the UNUSED slots.
/// UNUSED slots are filled with ordered distinct selections from `untouched`.
fn expand_abstract_gate(gate: [u8; 3], untouched: &[u8]) -> Vec<[u8; 3]> {
    const UNUSED: u8 = u8::MAX;
    let slots: Vec<usize> = gate
        .iter()
        .enumerate()
        .filter(|(_, w)| **w == UNUSED)
        .map(|(i, _)| i)
        .collect();

    match slots.len() {
        0 => vec![gate],
        1 => untouched
            .iter()
            .map(|&u0| {
                let mut g = gate;
                g[slots[0]] = u0;
                g
            })
            .collect(),
        2 => {
            let mut result = Vec::new();
            for &u0 in untouched {
                for &u1 in untouched {
                    if u1 == u0 {
                        continue;
                    }
                    let mut g = gate;
                    g[slots[0]] = u0;
                    g[slots[1]] = u1;
                    result.push(g);
                }
            }
            result
        }
        3 => {
            let mut result = Vec::new();
            for &u0 in untouched {
                for &u1 in untouched {
                    if u1 == u0 {
                        continue;
                    }
                    for &u2 in untouched {
                        if u2 == u0 || u2 == u1 {
                            continue;
                        }
                        let mut g = gate;
                        g[slots[0]] = u0;
                        g[slots[1]] = u1;
                        g[slots[2]] = u2;
                        result.push(g);
                    }
                }
            }
            result
        }
        _ => unreachable!(),
    }
}

/// For a given circuit, enumerate all concrete gates worth trying when
/// appending or prepending a gate. Exploits the symmetry that all untouched
/// wires are equivalent, collapsing them into one representative (UNUSED sentinel)
/// for enumeration then expanding back to concrete gates.
///
/// For a circuit touching k wires out of n total (with n-k untouched), the
/// number of abstract options is:
///   k*(k-1)*(k-2)          -- all three wires are touched
///   + k*(k-1)              -- two touched, one untouched
///   + k                    -- one touched, two untouched
///   + 1                    -- all three untouched (if n-k >= 3)
/// Each abstract option expands to 1, (n-k), (n-k)*(n-k-1), or (n-k)*(n-k-1)*(n-k-2)
/// concrete gates respectively.
pub fn abstract_gates_for_circuit(circuit: &CircuitSeq, n: usize) -> Vec<[u8; 3]> {
    const UNUSED: u8 = u8::MAX;

    let touched = touched_wires(circuit);
    let untouched: Vec<u8> = (0..n as u8)
        .filter(|w| !touched.contains(w))
        .collect();

    let mut result = Vec::new();

    // 0 UNUSED slots: all three wires are touched
    for &a in &touched {
        for &b in &touched {
            if b == a { continue; }
            for &c in &touched {
                if c == a || c == b { continue; }
                result.push([a, b, c]);
            }
        }
    }

    if !untouched.is_empty() {
        // 1 UNUSED slot: exactly one wire is untouched, two are touched
        // UNUSED in position a
        for &b in &touched {
            for &c in &touched {
                if c == b { continue; }
                result.extend(expand_abstract_gate([UNUSED, b, c], &untouched));
            }
        }
        // UNUSED in position b
        for &a in &touched {
            for &c in &touched {
                if c == a { continue; }
                result.extend(expand_abstract_gate([a, UNUSED, c], &untouched));
            }
        }
        // UNUSED in position c
        for &a in &touched {
            for &b in &touched {
                if b == a { continue; }
                result.extend(expand_abstract_gate([a, b, UNUSED], &untouched));
            }
        }
    }

    if untouched.len() >= 2 {
        // 2 UNUSED slots: two wires are untouched, one is touched
        // UNUSED in positions b and c
        for &a in &touched {
            result.extend(expand_abstract_gate([a, UNUSED, UNUSED], &untouched));
        }
        // UNUSED in positions a and c
        for &b in &touched {
            result.extend(expand_abstract_gate([UNUSED, b, UNUSED], &untouched));
        }
        // UNUSED in positions a and b
        for &c in &touched {
            result.extend(expand_abstract_gate([UNUSED, UNUSED, c], &untouched));
        }
    }

    if untouched.len() >= 3 {
        // 3 UNUSED slots: all three wires are untouched
        result.extend(expand_abstract_gate([UNUSED, UNUSED, UNUSED], &untouched));
    }

    result
}

pub fn build_from_rocks(
    old_db: &Arc<DB>,
    new_db: &Arc<DB>,
    m: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Running build (max CPU)");

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build_global()
        .unwrap();

    let total_rows = old_db
        .property_int_value("rocksdb.estimate-num-keys")
        .unwrap()
        .unwrap_or(0);
    println!("Estimated rows: {}", total_rows);

    let chunk_size = 500_000;
    let batch_size = 10_000;

    let upper_bound_gates = base_gates(3 * m).len();
    let total_gates_tried = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    // DashMap<pair_key, Vec<(fwd_blob, rev_blob)>>
    // pair_key = min(fwd_canon_hash, rev_canon_hash) for bucketing by semantic content.
    // fwd_blob and rev_blob are raw circuit gate blobs (not poly blobs), so that
    // semantically equal but structurally different circuits can both be inserted —
    // we only skip if the exact circuit (or its reversal) is already present.
    let seen: Arc<DashMap<u128, Vec<(Vec<u8>, Vec<u8>)>>> = Arc::new(DashMap::new());

    let stop_flag = Arc::new(AtomicBool::new(false));
    {
        let stop_flag = stop_flag.clone();
        ctrlc::set_handler(move || {
            println!("CTRL+C detected! Finishing current batch...");
            stop_flag.store(true, Ordering::SeqCst);
        })
        .expect("Error setting CTRL+C handler");
    }

    let (tx, rx) = bounded::<Vec<(CircuitSeq, Vec<Polynomial>, Vec<u8>, Vec<u8>)>>(100_000);
    let stop_flag_clone = stop_flag.clone();
    let new_db_writer = Arc::clone(new_db);
    let total_gates_tried_insert = Arc::clone(&total_gates_tried);

    let insert_handle = std::thread::spawn(move || {
        let start_time = std::time::Instant::now();
        let total_circuits = total_rows as usize * upper_bound_gates * 2;
        let mut attempted_inserts = 0;
        let mut sst_index = 0usize;
        let mut pending: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();

        while let Ok(batch) = rx.recv() {
            if stop_flag_clone.load(Ordering::SeqCst) {
                println!("Insertion thread stopping early...");
                break;
            }

            // No secondary dedup needed — seen DashMap upstream is the single
            // source of truth
            for (circuit, _canon, key, _pair_key) in &batch {
                let circuit_blob = circuit.repr_blob();
                let value = encode_circuit(&circuit_blob);
                pending.push((key.clone(), value));
            }

            attempted_inserts += batch.len();
            let tried = total_gates_tried_insert.load(Ordering::Relaxed);
            let elapsed = start_time.elapsed().as_secs_f64();
            let rate = if elapsed > 0.0 { tried as f64 / elapsed } else { 0.0 };
            let remaining = if rate > 0.0 {
                (total_circuits as f64 - tried as f64) / rate
            } else {
                f64::INFINITY
            };
            let remaining_secs = remaining as u64;
            let remaining_h = remaining_secs / 3600;
            let remaining_m = (remaining_secs % 3600) / 60;
            let remaining_s = remaining_secs % 60;
            println!(
                "Attempted inserts: {} / {} ({:.2}%) | elapsed: {:.0}s | rate: {:.0}/s | eta: {:02}:{:02}:{:02}",
                attempted_inserts,
                total_circuits,
                if tried > 0 {
                    (tried as f64 / total_circuits as f64) * 100.0
                } else {
                    0.0
                },
                elapsed,
                rate,
                remaining_h,
                remaining_m,
                remaining_s,
            );

            if pending.len() >= 1_000_000 {
                flush_to_sst(&new_db_writer, &mut pending, &mut sst_index);
            }
        }

        if !pending.is_empty() {
            flush_to_sst(&new_db_writer, &mut pending, &mut sst_index);
        }

        let elapsed = start_time.elapsed().as_secs_f64();
        println!(
            "Insertion thread finished. Total attempted: {} / {} | elapsed: {:.0}s",
            attempted_inserts,
            total_circuits,
            elapsed,
        );
    });

    let iter = old_db.iterator(rocksdb::IteratorMode::Start);

    for chunk in &iter.chunks(chunk_size) {
        if stop_flag.load(Ordering::SeqCst) {
            break;
        }

        let entries: Vec<(Vec<u8>, Vec<u8>)> = chunk
            .map(|item| {
                let (k, v) = item.expect("RocksDB iter error");
                (k.to_vec(), v.to_vec())
            })
            .collect();

        let stop_flag_par = Arc::clone(&stop_flag);
        let tx_par = tx.clone();
        let total_gates_tried_par = Arc::clone(&total_gates_tried);
        let seen_par = Arc::clone(&seen);

        entries.par_chunks(20).for_each(|entry_chunk| {
            if stop_flag_par.load(Ordering::SeqCst) {
                return;
            }

            let mut local_results = Vec::new();

            for (_key, value) in entry_chunk {
                if value.is_empty() {
                    continue;
                }

                let mut pos = 0;
                while pos < value.len() {
                    if pos + 1 > value.len() {
                        break;
                    }
                    let len = value[pos] as usize;
                    pos += 1;
                    if pos + len > value.len() {
                        break;
                    }
                    let circuit_blob = &value[pos..pos + len];
                    pos += len;

                    let old_circuit = CircuitSeq::from_blob(circuit_blob);

                    total_gates_tried_par.fetch_add(upper_bound_gates * 2, Ordering::Relaxed);

                    let mut prefix: SmallVec<[[u8; 3]; 64]> = SmallVec::with_capacity(m);
                    prefix.extend_from_slice(&old_circuit.gates);

                    let gates = abstract_gates_for_circuit(&old_circuit, 3 * m);

                    for g in gates.iter() {
                        let mut q1 = prefix.clone();
                        q1.push(*g);
                        let mut c1 = CircuitSeq { gates: q1.to_vec() };
                        c1.canonicalize();
                        let canon1 = canonicalize_polys_4(c1.to_polynomial(3 * m, 0, m));
                        c1.rewire(&canon1.1.invert(), 3 * m);
                        c1.canonicalize();

                        if !c1.adjacent_id() {
                            // Canon poly hashes for bucketing
                            let c1_canon_blob = polys_repr_blob(&canon1.0);
                            let c1_hash: u128 = xxh3_128(&c1_canon_blob);

                            let mut c1_rev = c1.clone();
                            c1_rev.gates.reverse();
                            c1_rev.canonicalize();
                            let canon1_rev = canonicalize_polys_4(c1_rev.to_polynomial(3 * m, 0, m));
                            c1_rev.rewire(&canon1_rev.1.invert(), 3 * m);
                            c1_rev.canonicalize();
                            let c1_rev_hash: u128 = xxh3_128(&polys_repr_blob(&canon1_rev.0));

                            let pair_key = c1_hash.min(c1_rev_hash);

                            // Circuit gate blobs for identity comparison —
                            // structurally different but semantically equal circuits
                            // have different gate blobs and can both be inserted
                            let c1_fwd_gate_blob = c1.repr_blob();
                            let c1_rev_gate_blob = c1_rev.repr_blob();

                            // Atomically check and insert using DashMap entry API.
                            // entry() holds a lock on the bucket for the duration,
                            // eliminating any TOCTOU window.
                            let mut entry = seen_par.entry(pair_key).or_insert_with(Vec::new);
                            let already_seen = entry.iter().any(|(f, r)| {
                                (f == &c1_fwd_gate_blob && r == &c1_rev_gate_blob)
                                    || (f == &c1_rev_gate_blob && r == &c1_fwd_gate_blob)
                            });
                            if !already_seen {
                                entry.push((c1_fwd_gate_blob, c1_rev_gate_blob));
                                drop(entry); // release lock before pushing to channel
                                local_results.push((
                                    c1,
                                    canon1.0,
                                    c1_hash.to_le_bytes().to_vec(),
                                    pair_key.to_le_bytes().to_vec(),
                                ));
                            } else {
                                drop(entry);
                            }
                        }

                        let mut q2: SmallVec<[[u8; 3]; 64]> = SmallVec::with_capacity(m + 1);
                        q2.push(*g);
                        q2.extend_from_slice(&prefix);
                        let mut c2 = CircuitSeq { gates: q2.to_vec() };
                        c2.canonicalize();
                        let canon2 = canonicalize_polys_4(c2.to_polynomial(3 * m, 0, m));
                        c2.rewire(&canon2.1.invert(), 3 * m);
                        c2.canonicalize();

                        if !c2.adjacent_id() {
                            // Canon poly hashes for bucketing
                            let c2_canon_blob = polys_repr_blob(&canon2.0);
                            let c2_hash: u128 = xxh3_128(&c2_canon_blob);

                            let mut c2_rev = c2.clone();
                            c2_rev.gates.reverse();
                            c2_rev.canonicalize();
                            let canon2_rev = canonicalize_polys_4(c2_rev.to_polynomial(3 * m, 0, m));
                            c2_rev.rewire(&canon2_rev.1.invert(), 3 * m);
                            c2_rev.canonicalize();
                            let c2_rev_hash: u128 = xxh3_128(&polys_repr_blob(&canon2_rev.0));

                            let pair_key = c2_hash.min(c2_rev_hash);

                            // Circuit gate blobs for identity comparison
                            let c2_fwd_gate_blob = c2.repr_blob();
                            let c2_rev_gate_blob = c2_rev.repr_blob();

                            let mut entry = seen_par.entry(pair_key).or_insert_with(Vec::new);
                            let already_seen = entry.iter().any(|(f, r)| {
                                (f == &c2_fwd_gate_blob && r == &c2_rev_gate_blob)
                                    || (f == &c2_rev_gate_blob && r == &c2_fwd_gate_blob)
                            });
                            if !already_seen {
                                entry.push((c2_fwd_gate_blob, c2_rev_gate_blob));
                                drop(entry);
                                local_results.push((
                                    c2,
                                    canon2.0,
                                    c2_hash.to_le_bytes().to_vec(),
                                    pair_key.to_le_bytes().to_vec(),
                                ));
                            } else {
                                drop(entry);
                            }
                        }
                    }

                    while local_results.len() >= batch_size {
                        let drain_start = local_results.len() - batch_size;
                        let batch = local_results.split_off(drain_start);
                        if let Err(e) = tx_par.send(batch) {
                            eprintln!("Failed to send batch: {:?}", e);
                            return;
                        }
                    }

                    if stop_flag_par.load(Ordering::SeqCst) {
                        return;
                    }
                }
            }

            if !local_results.is_empty() {
                if let Err(e) = tx_par.send(local_results) {
                    eprintln!("Failed to send remaining batch: {:?}", e);
                }
            }
        });
    }

    drop(tx);
    insert_handle.join().expect("Insertion thread panicked");

    if !stop_flag.load(Ordering::SeqCst) {
        println!("Compacting new_db for optimal read performance...");
        new_db.compact_range::<&[u8], &[u8]>(None, None);
        println!("Compaction done.");
    } else {
        println!("Stopped early, skipping compaction.");
    }

    println!("Build finished (or stopped early).");
    print_rule_times();
    Ok(())
}

pub fn build_m1(new_db: &Arc<DB>) -> Result<(), Box<dyn std::error::Error>> {
    println!("Building m1 base case");

    let gates = base_gates(3);
    let mut pending: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    let mut sst_index = 0usize;

    for g in gates.iter() {
        let c = CircuitSeq { gates: vec![*g] };
        let canon = canonicalize_polys(c.to_polynomial(3, 0, 1), true, false);
        let mut c = c;
        c.rewire(&canon.1.invert(), 3);

        if c.adjacent_id() {
            continue;
        }

        let canon_blob = polys_repr_blob(&canon.0);
        let hash: u128 = xxh3_128(&canon_blob);
        let key = hash.to_le_bytes().to_vec();

        let circuit_blob = c.repr_blob();
        let value = encode_circuit(&circuit_blob);

        pending.push((key, value));
    }

    flush_to_sst(new_db, &mut pending, &mut sst_index);

    println!("Compacting m1 db...");
    new_db.compact_range::<&[u8], &[u8]>(None, None);
    println!("Done.");

    Ok(())
}

/// Generate all wire mappings for C2 relative to C1.
/// Returns a list of permutations, where each permutation maps
/// C2's wire j to a concrete wire index in the combined circuit.
///
/// C1 occupies wires 0..N1-1 (fixed).
/// C2's wires are mapped as follows:
///   - s is a word of length N1 over {0..N2}, with at most one of each letter from 1..N2
///   - if s[w] = j (j > 0), then C2's wire j-1 maps to wire w (shared with C1)
///   - C2's wires not mentioned in s map to fresh wires N1, N1+1, ... in fixed order
///
/// Total number of mappings = sum_{k=0}^{min(N1,N2)} C(N1,k) * P(N2,k)
fn enumerate_c2_wire_mappings(n1: usize, n2: usize) -> Vec<Vec<u8>> {
    let mut result = Vec::new();

    // Enumerate all words s of length N1 over {0..N2}
    // with at most one of each letter from 1..N2
    // We do this recursively/iteratively by choosing which positions
    // in s get non-zero letters and which letters they get

    // s[i] = 0 means position i of C1's wires is not used by C2
    // s[i] = j (1-indexed) means C2's wire j-1 maps to C1's wire i
    fn enumerate_words(
        pos: usize,
        n1: usize,
        n2: usize,
        word: &mut Vec<usize>,
        used: &mut Vec<bool>, // which C2 wire indices (1..N2) are used
        result: &mut Vec<Vec<u8>>,
    ) {
        if pos == n1 {
            // word is complete — build the concrete wire mapping for C2
            // For each C2 wire j (0-indexed), find where it maps:
            //   - if j+1 appears in word at position w, it maps to wire w
            //   - otherwise it maps to the next fresh wire after N1

            // Find which C2 wires are mentioned in word and where
            let mut c2_to_wire = vec![0u8; n2];
            let mut mentioned = vec![false; n2];
            for (w, &j) in word.iter().enumerate() {
                if j > 0 {
                    c2_to_wire[j - 1] = w as u8;
                    mentioned[j - 1] = true;
                }
            }
            // Assign fresh wires to unmentioned C2 wires in fixed order
            let mut fresh = n1;
            for j in 0..n2 {
                if !mentioned[j] {
                    c2_to_wire[j] = fresh as u8;
                    fresh += 1;
                }
            }
            result.push(c2_to_wire);
            return;
        }

        // Option 1: s[pos] = 0 (this C1 wire not used by C2)
        word.push(0);
        enumerate_words(pos + 1, n1, n2, word, used, result);
        word.pop();

        // Option 2: s[pos] = j for each unused j in 1..N2
        for j in 1..=n2 {
            if !used[j - 1] {
                used[j - 1] = true;
                word.push(j);
                enumerate_words(pos + 1, n1, n2, word, used, result);
                word.pop();
                used[j - 1] = false;
            }
        }
    }

    let mut word = Vec::with_capacity(n1);
    let mut used = vec![false; n2];
    enumerate_words(0, n1, n2, &mut word, &mut used, &mut result);
    result
}

/// Apply a wire mapping to a circuit — remap C2's internal wires
/// to their positions in the combined circuit.
fn apply_wire_mapping(circuit: &CircuitSeq, mapping: &[u8]) -> CircuitSeq {
    CircuitSeq {
        gates: circuit
            .gates
            .iter()
            .map(|&[a, b, c]| [mapping[a as usize], mapping[b as usize], mapping[c as usize]])
            .collect(),
    }
}

fn process_combination(
    first: &CircuitSeq,
    second: &CircuitSeq,
    n: usize,
    m: usize,
    new_db: &Arc<DB>,
) -> Option<(CircuitSeq, Vec<Polynomial>, Vec<u8>, Vec<u8>)> {
    let mut combined_gates = first.gates.clone();
    combined_gates.extend_from_slice(&second.gates);
    let mut combined = CircuitSeq { gates: combined_gates };
    combined.canonicalize();
    let canon = canonicalize_polys(combined.to_polynomial(n, 0, m), true, false);
    combined.rewire(&canon.1.invert(), n);
    combined.canonicalize();

    let blob = polys_repr_blob(&canon.0);
    let hash: u128 = xxh3_128(&blob);
    let key = hash.to_le_bytes().to_vec();

    let mut rev = combined.clone();
    rev.gates.reverse();
    rev.canonicalize();
    let canon_rev = canonicalize_polys(rev.to_polynomial(n, 0, m), true, false);
    let rev_blob = polys_repr_blob(&canon_rev.0);
    let rev_hash: u128 = xxh3_128(&rev_blob);
    let rev_key = rev_hash.to_le_bytes().to_vec();

    if !combined.adjacent_id() {
        let rev_in_db = new_db.get(&rev_key).unwrap_or(None).is_some();
        if !rev_in_db {
            return Some((combined, canon.0, key, rev_key));
        }
    }
    None
}

pub fn build_from_2rocks(
    db1: &Arc<DB>,
    db2: &Arc<DB>,
    new_db: &Arc<DB>,
    m1: usize,
    m2: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let m = m1 + m2;
    let n = 3 * m;
    let same_db = Arc::ptr_eq(db1, db2);
    println!("Running build_from_2rocks: m1={} m2={} -> m={} same_db={}", m1, m2, m, same_db);

    let total_rows_db1 = db1
        .property_int_value("rocksdb.estimate-num-keys")
        .unwrap()
        .unwrap_or(0);
    let total_rows_db2 = db2
        .property_int_value("rocksdb.estimate-num-keys")
        .unwrap()
        .unwrap_or(0);
    println!("Estimated rows: db1={} db2={}", total_rows_db1, total_rows_db2);

    let batch_size = 10_000;

    // DashMap<pair_key, Vec<(fwd_blob, rev_blob)>>
    // pair_key = min(fwd_hash, rev_hash) for bucketing
    // Within each bucket, we store actual circuit blobs so semantically equal
    // but structurally different circuits are handled correctly
    let seen: Arc<DashMap<u128, Vec<(Vec<u8>, Vec<u8>)>>> = Arc::new(DashMap::new());

    let stop_flag = Arc::new(AtomicBool::new(false));
    {
        let stop_flag = stop_flag.clone();
        ctrlc::set_handler(move || {
            println!("CTRL+C detected! Finishing current batch...");
            stop_flag.store(true, Ordering::SeqCst);
        })
        .expect("Error setting CTRL+C handler");
    }

    println!("Loading db2 into memory...");
    let db2_circuits: Arc<Vec<CircuitSeq>> = Arc::new({
        let iter = db2.iterator(rocksdb::IteratorMode::Start);
        let mut circuits = Vec::new();
        for item in iter {
            let (_key, value) = item.expect("RocksDB iter error");
            let mut pos = 0;
            while pos < value.len() {
                if pos + 1 > value.len() { break; }
                let len = value[pos] as usize;
                pos += 1;
                if pos + len > value.len() { break; }
                let circuit_blob = &value[pos..pos + len];
                pos += len;
                circuits.push(CircuitSeq::from_blob(circuit_blob));
            }
        }
        println!("Loaded {} circuits from db2", circuits.len());
        circuits
    });

    let db1_circuits: Arc<Vec<CircuitSeq>> = if same_db {
        Arc::clone(&db2_circuits)
    } else {
        println!("Loading db1 into memory...");
        Arc::new({
            let iter = db1.iterator(rocksdb::IteratorMode::Start);
            let mut circuits = Vec::new();
            for item in iter {
                let (_key, value) = item.expect("RocksDB iter error");
                let mut pos = 0;
                while pos < value.len() {
                    if pos + 1 > value.len() { break; }
                    let len = value[pos] as usize;
                    pos += 1;
                    if pos + len > value.len() { break; }
                    let circuit_blob = &value[pos..pos + len];
                    pos += len;
                    circuits.push(CircuitSeq::from_blob(circuit_blob));
                }
            }
            println!("Loaded {} circuits from db1", circuits.len());
            circuits
        })
    };

    let nc1 = db1_circuits.len();
    let nc2 = db2_circuits.len();
    let total_work = if same_db {
        nc2 * (nc2 + 1) / 2 * 8
    } else {
        nc1 * nc2 * 8
    };
    let total_gates_tried = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let total_gates_tried_insert = Arc::clone(&total_gates_tried);

    let (tx, rx) = bounded::<Vec<(CircuitSeq, Vec<Polynomial>, Vec<u8>, Vec<u8>)>>(100_000);
    let stop_flag_clone = stop_flag.clone();
    let new_db_writer = Arc::clone(new_db);

    let insert_handle = std::thread::spawn(move || {
        let start_time = std::time::Instant::now();
        let mut attempted_inserts = 0;
        let mut sst_index = 0usize;
        let mut pending: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();

        while let Ok(batch) = rx.recv() {
            if stop_flag_clone.load(Ordering::SeqCst) {
                println!("Insertion thread stopping early...");
                break;
            }

            for (circuit, _canon, key, _pair_key) in &batch {
                let circuit_blob = circuit.repr_blob();
                let value = encode_circuit(&circuit_blob);
                pending.push((key.clone(), value));
            }

            attempted_inserts += batch.len();
            let tried = total_gates_tried_insert.load(Ordering::Relaxed);
            let elapsed = start_time.elapsed().as_secs_f64();
            let rate = if elapsed > 0.0 { tried as f64 / elapsed } else { 0.0 };
            let remaining = if rate > 0.0 {
                (total_work as f64 - tried as f64) / rate
            } else {
                f64::INFINITY
            };
            let remaining_secs = remaining as u64;
            let remaining_h = remaining_secs / 3600;
            let remaining_m = (remaining_secs % 3600) / 60;
            let remaining_s = remaining_secs % 60;
            println!(
                "Attempted inserts: {} / {} ({:.2}%) | elapsed: {:.0}s | rate: {:.0}/s | eta: {:02}:{:02}:{:02}",
                attempted_inserts,
                total_work,
                if tried > 0 { (tried as f64 / total_work as f64) * 100.0 } else { 0.0 },
                elapsed,
                rate,
                remaining_h,
                remaining_m,
                remaining_s,
            );

            if pending.len() >= 1_000_000 {
                flush_to_sst(&new_db_writer, &mut pending, &mut sst_index);
            }
        }

        if !pending.is_empty() {
            flush_to_sst(&new_db_writer, &mut pending, &mut sst_index);
        }

        let elapsed = start_time.elapsed().as_secs_f64();
        println!(
            "Insertion thread finished. Total attempted: {} | elapsed: {:.0}s",
            attempted_inserts, elapsed,
        );
    });

    // Helper: compute (fwd_blob, rev_blob, fwd_hash, rev_hash, pair_key) for a combined circuit
    let compute_blobs = |first: &CircuitSeq, second: &CircuitSeq| -> (Vec<u8>, Vec<u8>, u128, u128, u128) {
        let mut combined = first.clone();
        combined.gates.extend_from_slice(&second.gates);
        combined.canonicalize();
        let canon = canonicalize_polys_4(combined.to_polynomial(n, 0, m));
        let fwd_blob = polys_repr_blob(&canon.0);
        let fwd_hash = xxh3_128(&fwd_blob);

        let mut rev = combined.clone();
        rev.gates.reverse();
        rev.canonicalize();
        let canon_rev = canonicalize_polys_4(rev.to_polynomial(n, 0, m));
        let rev_blob = polys_repr_blob(&canon_rev.0);
        let rev_hash = xxh3_128(&rev_blob);

        let pair_key = fwd_hash.min(rev_hash);
        (fwd_blob, rev_blob, fwd_hash, rev_hash, pair_key)
    };

    // Helper: dedup check and insert into seen, returns the key bytes if inserted
    let try_insert_seen = |seen: &DashMap<u128, Vec<(Vec<u8>, Vec<u8>)>>,
                           fwd_blob: Vec<u8>,
                           rev_blob: Vec<u8>,
                           pair_key: u128|
     -> Option<Vec<u8>> {
        let fwd_hash = xxh3_128(&fwd_blob);
        let mut entry = seen.entry(pair_key).or_insert_with(Vec::new);
        let already_seen = entry.iter().any(|(f, r)| {
            (f == &fwd_blob && r == &rev_blob) || (f == &rev_blob && r == &fwd_blob)
        });
        if !already_seen {
            entry.push((fwd_blob, rev_blob));
            drop(entry);
            Some(fwd_hash.to_le_bytes().to_vec())
        } else {
            None
        }
    };

    let stop_flag_par = Arc::clone(&stop_flag);
    let tx_par = tx.clone();
    let db1_circuits_par = Arc::clone(&db1_circuits);
    let db2_circuits_par = Arc::clone(&db2_circuits);
    let total_gates_tried_par = Arc::clone(&total_gates_tried);
    let seen_par = Arc::clone(&seen);

    (0..nc1).into_par_iter().for_each(|i| {
        if stop_flag_par.load(Ordering::SeqCst) {
            return;
        }

        let c1 = &db1_circuits_par[i];
        let n1 = touched_wires(c1).len();
        let c1_rev_raw = CircuitSeq { gates: c1.gates.iter().rev().cloned().collect() };
        let (c1_rev, _) = canonicalize_circuit(c1_rev_raw.gates, n, m1);
        let n1_rev = touched_wires(&c1_rev).len();

        let j_end = if same_db { i + 1 } else { nc2 };
        let mut local_results: Vec<(CircuitSeq, Vec<Polynomial>, Vec<u8>, Vec<u8>)> = Vec::new();

        for j in 0..j_end {
            let c2 = &db2_circuits_par[j];

            if same_db && !c1.geq(c2) {
                continue;
            }

            let n2 = touched_wires(c2).len();
            let c2_rev_raw = CircuitSeq { gates: c2.gates.iter().rev().cloned().collect() };
            let (c2_rev, _) = canonicalize_circuit(c2_rev_raw.gates, n, m2);

            let mappings_1_2   = enumerate_c2_wire_mappings(n1,     n2);
            let mappings_rev1_2 = enumerate_c2_wire_mappings(n1_rev, n2);
            let mappings_2_1   = enumerate_c2_wire_mappings(n2,     n1);
            // let mappings_rev2_1 = enumerate_c2_wire_mappings(n2_rev, n1);

            // Case 1: c1 || mapped_c2
            for mapping in &mappings_1_2 {
                let c2m = apply_wire_mapping(c2, mapping);
                total_gates_tried_par.fetch_add(1, Ordering::Relaxed);
                if let Some(result) = process_combination(c1, &c2m, n, m, new_db) {
                    let (circuit, canon, _, _) = result;
                    let mut rev = circuit.clone();
                    rev.gates.reverse();
                    rev.canonicalize();
                    let canon_rev = canonicalize_polys_4(rev.to_polynomial(n, 0, m));
                    let fwd_blob = polys_repr_blob(&canon);
                    let rev_blob = polys_repr_blob(&canon_rev.0);
                    let fwd_hash = xxh3_128(&fwd_blob);
                    let rev_hash = xxh3_128(&rev_blob);
                    let pair_key = fwd_hash.min(rev_hash);
                    if let Some(key) = try_insert_seen(&seen_par, fwd_blob, rev_blob, pair_key) {
                        local_results.push((circuit, canon, key, pair_key.to_le_bytes().to_vec()));
                    }
                }
            }

            // Case 2: c2 || mapped_c1
            for mapping in &mappings_2_1 {
                let c1m = apply_wire_mapping(c1, mapping);
                total_gates_tried_par.fetch_add(1, Ordering::Relaxed);
                if let Some(result) = process_combination(c2, &c1m, n, m, new_db) {
                    let (circuit, canon, _, _) = result;
                    let mut rev = circuit.clone();
                    rev.gates.reverse();
                    rev.canonicalize();
                    let canon_rev = canonicalize_polys_4(rev.to_polynomial(n, 0, m));
                    let fwd_blob = polys_repr_blob(&canon);
                    let rev_blob = polys_repr_blob(&canon_rev.0);
                    let fwd_hash = xxh3_128(&fwd_blob);
                    let rev_hash = xxh3_128(&rev_blob);
                    let pair_key = fwd_hash.min(rev_hash);
                    if let Some(key) = try_insert_seen(&seen_par, fwd_blob, rev_blob, pair_key) {
                        local_results.push((circuit, canon, key, pair_key.to_le_bytes().to_vec()));
                    }
                }
            }

            // Case 3: c1_rev || mapped_c2
            for mapping in &mappings_rev1_2 {
                let c2m = apply_wire_mapping(c2, mapping);
                total_gates_tried_par.fetch_add(1, Ordering::Relaxed);
                if let Some(result) = process_combination(&c1_rev, &c2m, n, m, new_db) {
                    let (circuit, canon, _, _) = result;
                    let mut rev = circuit.clone();
                    rev.gates.reverse();
                    rev.canonicalize();
                    let canon_rev = canonicalize_polys_4(rev.to_polynomial(n, 0, m));
                    let fwd_blob = polys_repr_blob(&canon);
                    let rev_blob = polys_repr_blob(&canon_rev.0);
                    let fwd_hash = xxh3_128(&fwd_blob);
                    let rev_hash = xxh3_128(&rev_blob);
                    let pair_key = fwd_hash.min(rev_hash);
                    if let Some(key) = try_insert_seen(&seen_par, fwd_blob, rev_blob, pair_key) {
                        local_results.push((circuit, canon, key, pair_key.to_le_bytes().to_vec()));
                    }
                }
            }

            // Case 7: mapped_c1 || c2_rev
            for mapping in &mappings_2_1 {
                let c1m = apply_wire_mapping(c1, mapping);
                total_gates_tried_par.fetch_add(1, Ordering::Relaxed);
                if let Some(result) = process_combination(&c1m, &c2_rev, n, m, new_db) {
                    let (circuit, canon, _, _) = result;
                    let mut rev = circuit.clone();
                    rev.gates.reverse();
                    rev.canonicalize();
                    let canon_rev = canonicalize_polys_4(rev.to_polynomial(n, 0, m));
                    let fwd_blob = polys_repr_blob(&canon);
                    let rev_blob = polys_repr_blob(&canon_rev.0);
                    let fwd_hash = xxh3_128(&fwd_blob);
                    let rev_hash = xxh3_128(&rev_blob);
                    let pair_key = fwd_hash.min(rev_hash);
                    if let Some(key) = try_insert_seen(&seen_par, fwd_blob, rev_blob, pair_key) {
                        local_results.push((circuit, canon, key, pair_key.to_le_bytes().to_vec()));
                    }
                }
            }


            // Drain local_results in batch_size chunks to keep memory bounded
            while local_results.len() >= batch_size {
                let drain_start = local_results.len() - batch_size;
                let batch = local_results.split_off(drain_start);
                if let Err(e) = tx_par.send(batch) {
                    eprintln!("Failed to send batch: {:?}", e);
                    return;
                }
            }

            if stop_flag_par.load(Ordering::SeqCst) {
                return;
            }
        }

        if !local_results.is_empty() {
            if let Err(e) = tx_par.send(local_results) {
                eprintln!("Failed to send remaining batch: {:?}", e);
            }
        }
    });

    drop(tx);
    insert_handle.join().expect("Insertion thread panicked");

    if !stop_flag.load(Ordering::SeqCst) {
        println!("Compacting new_db for optimal read performance...");
        new_db.compact_range::<&[u8], &[u8]>(None, None);
        println!("Compaction done.");
    } else {
        println!("Stopped early, skipping compaction.");
    }

    println!("Build finished (or stopped early).");
    print_rule_times();
    Ok(())
}

//Speed up SQL queries
//Should not see for a particular size query, the speed should not vary across multiple runs
// Attempt to add a random circuit to the SQL db
pub fn main_random(n: usize, m: usize, count: usize, stop: bool) {
    let config = Config::default().access_mode(AccessMode::ReadOnly).unwrap();
    let mut conn = Connection::open_with_flags("circuits.duckdb", config).unwrap();
    let table_name = format!("n{}m{}", n, m);
    create_table(&conn, &table_name).expect("Failed to create table");

    let perms: Vec<Vec<usize>> = (0..n).permutations(n).collect();
    let bit_shuf = perms.into_iter().skip(1).collect::<Vec<_>>();

    let mut inserted = 0;
    let mut total_attempts = 0;
    let mut recent = 0;

    let batch_size = 5_000;
    let mut batch: Vec<(CircuitSeq, Canonicalization)> = Vec::with_capacity(batch_size);

    // Atomic flag for Ctrl+C
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    }).expect("Error setting Ctrl-C handler");

    while running.load(Ordering::SeqCst) && (!stop && inserted < count || stop) {
        let start = std::time::Instant::now(); // start timing this iteration
        total_attempts += 1;

        let mut circuit = random_circuit(n, m);
        circuit.canonicalize();

        let perm = circuit.permutation(n).canon_simple(&bit_shuf);
        batch.push((circuit, perm));

        if batch.len() >= batch_size {
            //let start = std::time::Instant::now();
            let success_count =
                insert_circuits_batch(&mut conn, &table_name, &batch).unwrap_or(0);
            //let elapsed = start.elapsed();
            inserted += success_count;
            recent += success_count;
            batch.clear();

            // Early stop if >=99% of last batch failed
            if success_count * 100 <= batch_size {

                println!(
                    "Stopping early: only {}/{} inserts succeeded (~{:.2}% success)",
                    success_count,
                    batch_size,
                    (success_count as f64 / batch_size as f64) * 100.0
                );
                break;
            }
        }

        if total_attempts % 50_000 == 0 {
            println!("Attempts: {}, inserted in last window: {}", total_attempts, recent);
            recent = 0;
        }

        // Stop for non-stop mode
        if !stop && inserted >= count {
            break;
        }

        let elapsed = start.elapsed();
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("while.txt")
            .expect("Failed to open while.txt");
        writeln!(file, "Iteration {} took {:?}", total_attempts, elapsed)
            .expect("Failed to write to while.txt");
    }

    // Insert remaining circuits before exiting
    if !batch.is_empty() {
        let success_count =
            insert_circuits_batch(&mut conn, &table_name, &batch).unwrap_or(0);
        inserted += success_count;
    }

    println!(
        "Finished: inserted {} circuits after {} attempts",
        inserted, total_attempts
    );
}

fn canonicalize_circuit(gates: Vec<[u8; 3]>, n: usize, m: usize) -> (CircuitSeq, Permutation) {
        let mut c = CircuitSeq { gates };
        let canon = canonicalize_polys(c.to_polynomial(n, 0, m), true, false);
        c.rewire(&canon.1.invert(), n);
        c.canonicalize();
        (c, canon.1)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_check_cycles_n3m3() -> duckdb::Result<()> {
        let now = std::time::Instant::now();
        // Call check_cycles for n=3, m=3
        let _ = check_cycles(3, 3);
        //count_distinct()?;
        println!("Time: {:?}", now.elapsed());
        Ok(())
    }

    #[test]
    fn test_find_convex_subcircuit_min3_16wires() {
        // Dummy 16-wire circuit with 30 gates
        let c = random_circuit(64, 1000);
        let mut rng = rand::rng();
        let max_wires = 7;

        let mut subcircuit_gates = vec![];
        let mut attempts = 0;

        // Keep trying until a convex subcircuit with >= 3 gates is found
        while subcircuit_gates.len() < 5 {
            for set_size in (3..=16).rev() {
                let (gates, tries) = simple_find_convex_subcircuit(set_size, max_wires, 64, &c, &mut rng);
                attempts += tries;

                if !gates.is_empty() && gates.len() >= 3 {
                    subcircuit_gates = gates;
                    println!(
                        "Found convex subcircuit with {} gates after {} total attempts",
                        subcircuit_gates.len(),
                        attempts
                    );
                    break;
                }
            }

            if subcircuit_gates.len() < 4 {
                println!("No subcircuit ≥ 4 gates found in this round, retrying.................................................");
            }
        }

        println!("Selected gate indices: {:?}", subcircuit_gates);
        println!("Number of search attempts: {}", attempts);

        // Basic assertions
        assert!(subcircuit_gates.len() >= 3, "Subcircuit must have at least 3 gates");
        assert!(subcircuit_gates.len() <= c.gates.len(), "Subcircuit cannot exceed total gates");

        // Check that number of distinct wires is <= max_wires
        let mut wire_set = std::collections::HashSet::new();
        for &idx in &subcircuit_gates {
            for &w in &c.gates[idx] {
                wire_set.insert(w);
            }
        }
        assert!(wire_set.len() <= max_wires, "Subcircuit uses too many wires");
        println!("Wires used: {:?}", wire_set);

        // Check convexity
        let convex_ok = is_convex(64, &c, &subcircuit_gates);
        assert!(convex_ok, "Selected subcircuit is not convex");
        println!("Convexity check passed");

        // Optional: rewire/unrewire check and display
        let gates_arr: Vec<[u8; 3]> = subcircuit_gates.iter().map(|&i| c.gates[i]).collect();
        let subcircuit = CircuitSeq { gates: gates_arr };
        let sub = CircuitSeq::rewire_subcircuit(&c, &subcircuit_gates, &subcircuit.used_wires());
        let undo = CircuitSeq::unrewire_subcircuit(&sub, &subcircuit.used_wires());
        println!(
            "Rewire and unrewire is ok: {}",
            subcircuit.permutation(wire_set.len()) == undo.permutation(wire_set.len())
        );

        let mut circ = c.clone();
        let (start, end) = contiguous_convex(&mut circ, &mut subcircuit_gates, 64).unwrap();
        println!("After gates {:?}", subcircuit_gates);
        println!("start and end designated: {:?}", &circ.gates[start..=end]);
    }

    use crate::replace::pairs::{gate_pair_taxonomy};

    // #[test]
    // fn test_compression_big() {
    //     // Dummy 16-wire circuit with 30 gates
    //     let c = random_circuit(16,30);

    //     let mut conn = Connection::open("./circuits.db").expect("Failed to open DB");

    //     let com = compress_big(&c, 10, 16, &mut conn);
    //     println!("compression is okay: {}", com.permutation(16) == c.permutation(16));
    // }

    #[test]
    fn test_convexity() {
        // Dummy 16-wire circuit with 30 gates
        let gates: Vec<[u8; 3]> = vec!
            [[13, 6, 5], [7, 10, 1], [8, 12, 7], [5, 1, 11], [10, 5, 3], [1, 5, 9], [1, 15, 9], [14, 7, 10], [4, 9, 14], [14, 13, 9], [10, 12, 6], [5, 7, 13], [2, 1, 10], [11, 12, 6], [12, 9, 10], [8, 0, 9], [5, 3, 4], [2, 8, 10], [11, 10, 2], [9, 5, 12], [11, 1, 15], [14, 2, 3], [11, 1, 15], [9, 5, 12], [11, 10, 2], [2, 8, 10], [5, 3, 4], [8, 0, 9], [12, 9, 10], [11, 12, 6], [2, 1, 10], [1, 15, 9], [5, 7, 13], [10, 12, 6], [14, 13, 9], [1, 5, 9], [4, 9, 14], [14, 7, 10], [10, 5, 3], [5, 1, 11], [8, 12, 7], [7, 10, 1], [13, 6, 5]]
        ;

        let mut c = CircuitSeq { gates };
        let mut subcircuit_gates = vec![1, 4, 5, 6];

        println!("==================== TEST CONVEXITY ====================");
        println!("Total gates in circuit: {}", c.gates.len());
        println!("Original gate list:");
        for (i, g) in c.gates.iter().enumerate() {
            println!("  {:>2}: {:?}", i, g);
        }

        println!("\nSelected subcircuit gate indices: {:?}", subcircuit_gates);
        println!("Selected subcircuit gates:");
        for &i in &subcircuit_gates {
            println!("  {:>2}: {:?}", i, c.gates[i]);
        }

        let convex = is_convex(16, &c, &subcircuit_gates);
        println!("\nConvex before contiguous_convex? {}", convex);

        let (start, end) = contiguous_convex(&mut c, &mut subcircuit_gates, 16).unwrap();
        println!("\nAfter contiguous_convex:");
        println!("Start index: {}", start);
        println!("End index:   {}", end);
        println!("New subcircuit indices: {:?}", subcircuit_gates);

        println!("\nSegment of circuit after adjustment:");
        for (i, g) in c.gates[start..=end].iter().enumerate() {
            println!("  {:>2}: {:?}", start + i, g);
        }

        println!("\nSanity check (should be true):");
        let mut all_match = true;
        for (i, &gate_idx) in subcircuit_gates.iter().enumerate() {
            if c.gates[start + i] != c.gates[gate_idx] {
                println!(
                    "Mismatch at position {} (circuit idx {})",
                    start + i, gate_idx
                );
                all_match = false;
            }
        }
        if all_match {
            println!("All gates in contiguous range match subcircuit order");
        }
        println!("========================================================\n");
    }

    #[test]
    fn verify_butterfly() {
        let original = CircuitSeq::from_string("692;8c6;fd7;c6f;dc2;1ad;7c2;b8f;a3c;d10;f28;f91;941;8b2;82b;4fc;a78;e8b;780;142;6cb;8a6;e8c;fd7;07e;086;ea7;e74;549;ec3;");
        let new = CircuitSeq::from_string("f62;6ab;8b5;98f;6d4;4ba;5b1;13f;19e;db6;f9d;74d;172;97d;640;145;97d;172;19e;f9d;13f;6ba;145;5b1;74d;640;4ba;6ba;db6;6ab;6d4;98f;8b5;8e3;f62;8b5;f62;98f;5b1;13f;19e;6d4;6ab;4ba;db6;74d;6ba;640;6ba;145;19e;13f;98f;145;5b1;8b5;74d;640;4ba;8b5;db6;6ab;6d4;f62;0ce;f62;98f;5b1;6ab;6d4;db6;4ba;74d;640;145;13f;19e;f9d;172;97d;145;97d;172;19e;f9d;13f;5b1;98f;8b5;6ba;640;74d;4ba;6ba;db6;6ab;6d4;f62;601;f62;8b5;98f;5b1;13f;19e;6ab;6d4;db6;4ba;6ba;640;74d;19e;13f;5b1;98f;8b5;6ba;640;74d;4ba;db6;6ab;6d4;f62;8fb;f62;8b5;98f;5b1;13f;19e;6d4;6ab;db6;4ba;640;74d;145;19e;13f;98f;145;5b1;8b5;6ba;640;6ba;74d;db6;4ba;6ab;6d4;8b5;98f;94a;5b1;13f;19e;6ab;6d4;4ba;145;db6;6ba;f9d;74d;172;97d;640;145;97d;172;74d;f9d;19e;13f;5b1;98f;8b5;640;4ba;6ba;db6;6d4;6ab;f08;f62;8b5;f62;98f;5b1;6ab;6d4;13f;4ba;db6;74d;f9d;19e;172;97d;145;97d;172;f9d;19e;13f;98f;145;5b1;8b5;640;74d;6ba;640;6ba;db6;4ba;6ab;6d4;8b5;98f;5b1;13f;19e;6d4;6ab;04e;4ba;db6;74d;f9d;172;97d;145;97d;172;19e;f9d;13f;98f;145;5b1;8b5;74d;db6;6ab;4ba;6d4;ab6;f62;8b5;f62;5b1;6ab;6d4;4ba;db6;74d;640;6ba;145;98f;145;13f;19e;f9d;19e;f9d;13f;5b1;98f;8b5;74d;640;6ba;4ba;db6;6ab;6d4;f62;fa2;f62;8b5;5b1;13f;98f;19e;6ab;6d4;4ba;db6;74d;640;6ba;f9d;172;97d;145;640;145;97d;172;19e;f9d;13f;5b1;74d;4ba;6ba;db6;6d4;6ab;98f;8b5;f62;976;f62;8b5;98f;5b1;13f;19e;6ab;6d4;db6;4ba;6ba;640;74d;f9d;172;97d;640;97d;172;74d;f9d;19e;13f;98f;5b1;4ba;6ba;db6;6ab;6d4;f62;1a4;8b5;f62;8b5;98f;6ab;6d4;5b1;4ba;145;db6;6ba;145;74d;19e;13f;f9d;19e;f9d;74d;4ba;6ba;db6;6ab;6d4;13f;5b1;98f;8b5;eca;f62;8b5;f62;6d4;6ab;db6;98f;5b1;4ba;13f;19e;74d;19e;13f;98f;5b1;8b5;640;74d;640;6ba;4ba;6ba;db6;6ab;6d4;ab6;f62;8b5;5b1;f62;6d4;4ba;6ab;db6;640;74d;6ba;640;6ba;145;13f;98f;19e;f9d;172;97d;145;172;97d;19e;f9d;13f;5b1;98f;8b5;74d;db6;6ab;4ba;6d4;f62;2e5;8b5;f62;98f;5b1;6ab;6d4;4ba;db6;74d;6ba;640;13f;f9d;19e;74d;19e;f9d;6ba;13f;5b1;98f;8b5;640;db6;6ab;4ba;6d4;f62;8fa;f62;8b5;98f;5b1;13f;19e;6ab;6d4;4ba;db6;74d;f9d;172;97d;145;97d;172;19e;f9d;13f;145;5b1;98f;8b5;74d;4ba;db6;6ab;6d4;8b5;98f;5b1;13f;19e;06a;6d4;6ab;db6;4ba;74d;6ba;640;6ba;f9d;640;19e;f9d;13f;5b1;74d;4ba;db6;6ab;6d4;98f;f62;137;f62;6d4;4ba;6ab;db6;640;6ba;74d;6ba;98f;5b1;145;172;640;145;13f;f9d;19e;172;19e;f9d;13f;5b1;98f;74d;4ba;db6;6ab;6d4;e17;98f;5b1;13f;19e;6ab;6d4;4ba;db6;f9d;6ba;74d;640;f9d;19e;13f;98f;5b1;8b5;74d;640;4ba;6ba;db6;6d4;6ab;f62;38d;f62;8b5;98f;5b1;13f;19e;6d4;6ab;4ba;db6;74d;f9d;172;97d;145;97d;172;19e;f9d;74d;6ba;145;13f;5b1;98f;8b5;6ba;db6;4ba;6ab;6d4;b48;8b5;5b1;6ab;6d4;4ba;db6;74d;640;6ba;145;98f;145;13f;19e;f9d;97d;172;97d;172;74d;f9d;19e;13f;98f;5b1;8b5;640;6ba;4ba;db6;6d4;6ab;f62;a6f;f62;8b5;98f;5b1;13f;19e;6ab;6d4;4ba;db6;640;6ba;74d;145;f9d;6ba;172;97d;640;145;172;97d;f9d;19e;13f;5b1;74d;4ba;db6;6d4;6ab;98f;fd5;6ab;6d4;db6;6ba;4ba;640;74d;6ba;98f;5b1;13f;19e;f9d;19e;f9d;13f;5b1;98f;8b5;640;74d;db6;6ab;4ba;6d4;8c7;f62;8b5;f62;98f;5b1;13f;19e;6d4;4ba;6ab;db6;f9d;74d;172;19e;f9d;13f;98f;172;5b1;85b;8b5;640;6ba;640;6ba;74d;4ba;db6;6ab;6d4;8b5;5b1;98f;13f;19e;6ab;6d4;4ba;db6;74d;640;19e;13f;5b1;98f;8b5;6ba;640;6ba;74d;db6;4ba;6ab;6d4;f62;192;f62;8b5;98f;5b1;19e;6ab;6d4;4ba;db6;6ba;640;6ba;74d;13f;f9d;97d;172;97d;172;19e;f9d;98f;13f;5b1;74d;640;db6;8b5;6ab;4ba;6d4;f62;280;f62;8b5;98f;5b1;13f;19e;6d4;4ba;145;6ab;db6;6ba;74d;640;f9d;172;97d;172;97d;f9d;19e;74d;13f;98f;145;5b1;8b5;640;6ba;db6;4ba;6d4;0d8;6ab;8b5;6d4;6ab;98f;5b1;13f;db6;4ba;640;6ba;f9d;74d;172;19e;97d;145;97d;172;f9d;19e;13f;98f;145;5b1;74d;640;6ba;db6;6ab;4ba;6d4;f62;6ad;f62;5b1;6ab;6d4;db6;4ba;640;145;13f;98f;f9d;74d;172;19e;97d;145;97d;172;f9d;19e;13f;5b1;98f;8b5;6ba;74d;640;6ba;db6;6ab;4ba;6d4;f62;6e4;8b5;f62;5b1;98f;6d4;6ab;db6;6ba;4ba;640;74d;640;13f;f9d;6ba;19e;172;97d;145;172;97d;19e;f9d;13f;98f;145;5b1;74d;4ba;db6;6d4;6ab;f62;abf;f62;6d4;6ab;db6;4ba;640;98f;5b1;19e;74d;640;172;13f;f9d;172;74d;f9d;db6;19e;13f;5b1;4ba;6d4;98f;8b5;6ab;f62;");
        println!("Are they equal? {}", original.permutation(16) == new.permutation(16));
    }
    use std::fs;
    #[test]
    fn verify_easy() {
        // Read the file
        let contents = fs::read_to_string("butterfly_recent.txt")
            .expect("Failed to read butterfly_recent.txt");

        // Split into old and new by the first ':'
        let (old_str, new_str) = contents
            .split_once(':')
            .expect("Invalid format in butterfly_recent.txt");

        // Parse both circuits
        let old = CircuitSeq::from_string(old_str);
        let new = CircuitSeq::from_string(new_str);

        // Compare (example)
        println!(
            "Are they equal? {}",
            old.probably_equal(&new,64,100000).is_ok()
        );
    }
    use std::time::Instant;
    #[test]
    fn test_print() {
        let t = Instant::now();
        let c = random_circuit(32,30);
        let c1 = random_circuit(32,30);
        c
            .probably_equal(&c1, 32, 150_000)
            .expect("The circuits differ somewhere!");
        println!("Time to compute permutation on 32 wires: {:?}", t.elapsed());
    }

    #[test]
    fn test_identity() {
        let t = Instant::now();

        // Hardcoded circuit to compare
        let c = CircuitSeq::from_string("123;123;");

        // Load circuitA from file
        let contents = fs::read_to_string("circuitOOA_64.txt")
            .expect("Failed to read");
        let circuit_a = CircuitSeq::from_string(&contents);

        // Compare circuits
        c
            .probably_equal(&circuit_a, 64, 150_000)
            .expect("The circuits differ somewhere!");

        println!(
            "Time to compute permutation on 64 wires: {:?}",
            t.elapsed()
        );
    }

    use std::io::{self, BufRead};

    #[test]
    fn split_butterfly_unique() -> io::Result<()> {
        // Read all lines from butterfly.txt
        let file = fs::File::open("butterfly.txt")?;
        let reader = io::BufReader::new(file);

        // Collect all lines, then take the last 5
        let lines: Vec<String> = reader.lines().filter_map(Result::ok).collect();
        let last_five = lines.iter().rev().take(5).cloned().collect::<Vec<_>>();
        let last_five = last_five.into_iter().rev().collect::<Vec<_>>(); // preserve order

        let mut unique_circuits = HashSet::new();

        // Extract all circuits split by ':' and trim whitespace
        for line in last_five {
            for part in line.split(':') {
                let trimmed = part.trim();
                if !trimmed.is_empty() {
                    unique_circuits.insert(trimmed.to_string());
                }
            }
        }

        // Convert to Vec for sorting by length
        let mut circuits: Vec<(String, usize)> = unique_circuits
            .iter()
            .map(|s| {
                let c = CircuitSeq::from_string(s);
                let len = c.gates.len();
                (s.clone(), len)
            })
            .collect();

        circuits.sort_by_key(|(_, len)| *len);

        assert_eq!(
            circuits.len(),
            7,
            "Expected exactly 7 unique circuits, got {}",
            circuits.len()
        );

        // Filenames for sorted circuits
        let filenames = [
            "circuitB.txt",
            "circuitA.txt",
            "circuitOB.txt",
            "circuitOA.txt",
            "circuitOOB.txt",
            "circuitOOA.txt",
            "circuitOOOB.txt",
        ];

        // Write each circuit to file using repr()
        for ((circuit_str, _), filename) in circuits.iter().zip(filenames.iter()) {
            let circuit = CircuitSeq::from_string(circuit_str);
            fs::write(filename, circuit.repr())?;
            println!(" Wrote {} (len = {})", filename, circuit.gates.len());
        }

        // === Sanity Check ===
        fn read_circuit(path: &str) -> String {
            fs::read_to_string(path)
                .expect(&format!("Failed to read {}", path))
                .trim()
                .to_string()
        }

        let sanity_pairs = [
            ("circuitA.txt", "circuitOA.txt"),
            ("circuitOA.txt", "circuitOOA.txt"),
            ("circuitB.txt", "circuitOB.txt"),
            ("circuitOB.txt", "circuitOOB.txt"),
            ("circuitOOB.txt", "circuitOOOB.txt"),
        ];

        for (a, b) in sanity_pairs {
            let c1 = read_circuit(a);
            let c2 = read_circuit(b);
            let combined = format!("{}:{}", c1, c2);

            // ensure this combined pattern exists in butterfly.txt
            let full_text = fs::read_to_string("butterfly.txt")?;
            assert!(
                full_text.contains(&combined),
                "Sanity check failed: expected '{}' and '{}' to appear together in butterfly.txt",
                a,
                b
            );
        }

        println!(" All sanity checks passed.");
        Ok(())
    }
    use std::fs::File;
    use std::io::Write;

    #[test]
    fn generate_random_equivalent_circuits() {
        let n: usize = 16;

        // Generate two equivalent circuits
        let (c1, c2) = random_equivalent_circuits_until_found(n);

        if c1.probably_equal(&c2, n as usize, 1_000_000).is_ok() {
           println!("Looks good");
        }
        // Write c1 to c1.txt
        let mut file1 = File::create("c1.txt").expect("Failed to create c1.txt");
        writeln!(file1, "{:?}", c1).expect("Failed to write c1.txt");

        // Write c2 to c2.txt
        let mut file2 = File::create("c2.txt").expect("Failed to create c2.txt");
        writeln!(file2, "{:?}", c2).expect("Failed to write c2.txt");

        println!("Generated circuits written to c1.txt and c2.txt");
    }

    #[test]
    fn generate_random() {
        let n: usize = 64;

        let m = 100;

        let c = random_circuit(n ,m);

        let c_str = c.repr();
        File::create("circuit_random.txt")
            .and_then(|mut f| f.write_all(c_str.as_bytes()))
            .expect("Failed to write test_random.txt");
    }

    use crate::replace::identities::random_id;

    #[test]
    fn test_shooting() {
        // Start with an initial random identity
        // Load circuitA from file
        let contents = fs::read_to_string("circuit_before_random.txt")
            .expect("Failed to read");
        let mut circuit_a = CircuitSeq::from_string(&contents);
        let c1 = circuit_a.clone();
        let mut avg: f64 = 0.0;
        for _ in 0..100{
            shoot_random_gate(&mut circuit_a, 1_000_000);
            avg += heatmap(&c1, &circuit_a, 64, 500, false);
        }
        println!("Shooting avg: {}", avg/100.0);

        let c_str = circuit_a.repr();
        File::create("circuit_shot.txt")
            .and_then(|mut f| f.write_all(c_str.as_bytes()))
            .expect("Failed to write test_compression.txt");
    }

    #[test]
    fn test_walking() {
        // Start with an initial random identity
        // Load circuitA from file
        let contents = fs::read_to_string("circuit_before_random.txt")
            .expect("Failed to read");
        let mut circuit_a = CircuitSeq::from_string(&contents);
        let circuit_b = circuit_a.clone();
        // Proceed as before

        // for _ in 0..100{
        //     random_walk_no_skeleton(&mut circuit_a, &mut rand::rng());
        // }
        let mut avg: f64 = 0.0;
        for _ in 0..100{
            circuit_a = random_walk_no_skeleton(&mut circuit_a, &mut rand::rng());
            avg += heatmap(&circuit_b, &circuit_a, 64, 500, false);
        }
        println!("Walking avg: {}", avg/100.0);

        let c_str = circuit_a.repr();
        File::create("circuit_walked_no_skele.txt")
            .and_then(|mut f| f.write_all(c_str.as_bytes()))
            .expect("Failed to write test_walked.txt");

        // circuit_b = random_walking(&circuit_b, &mut rand::rng());

        // let c_str = circuit_b.repr();
        // File::create("circuit_walked.txt")
        //     .and_then(|mut f| f.write_all(c_str.as_bytes()))
        //     .expect("Failed to write test_walked.txt");
    }

    use rand::prelude::SliceRandom;

    pub fn heatmap(circuit_one: &CircuitSeq, circuit_two: &CircuitSeq, num_wires: usize, num_inputs: usize, flag: bool) -> f64 {
        let mut circuit_one = circuit_one.clone();
        let mut circuit_two = circuit_two.clone();
        if flag {
            circuit_one.canonicalize();
            circuit_two.canonicalize();
        }
        let circuit_one_len = circuit_one.gates.len();
        let circuit_two_len = circuit_two.gates.len();

        let mut average = vec![[0f64, 0f64, 0f64]; (circuit_one_len + 1) * (circuit_two_len + 1)];
        let mut rng = rand::rng();
        let _start_time = Instant::now();

        for _ in 0..num_inputs {
            let input_bits: usize = if num_wires < usize::BITS as usize {
                rng.random_range(0..(1usize << num_wires))
            } else {
                rng.random_range(0..=usize::MAX)
            };

            let evolution_one = circuit_one.evaluate_evolution(input_bits);
            let evolution_two = circuit_two.evaluate_evolution(input_bits);
            for i1 in 0..=circuit_one_len {
                for i2 in 0..=circuit_two_len {
                    let diff = evolution_one[i1] ^ evolution_two[i2];
                    let hamming_dist = diff.count_ones() as f64;
                    let overlap = (2.0 * hamming_dist / num_wires as f64) - 1.0;
                    let abs_overlap = overlap.abs();

                    let index = i1 * (circuit_two_len + 1) + i2;
                    average[index][2] += abs_overlap / num_inputs as f64;
                }
            }
        }

        // println!("Time elapsed: {:?}", Instant::now() - start_time);

        let total_points = average.len();
        let mean_all: f64 = average.iter().map(|p| p[2]).sum::<f64>() / total_points as f64;

        mean_all
    }
    #[test]
    fn test_random_order() {
        // Start with an initial random identity
        // Load circuitA from file
        let contents = fs::read_to_string("circuit_before_random.txt")
            .expect("Failed to read");
        let mut circuit_a = CircuitSeq::from_string(&contents);
        let c1 = circuit_a.clone();
        let mut avg: f64 = 0.0;
        // Proceed as before
        for _ in 0..100 {
            circuit_a.gates.shuffle(&mut rand::rng());
            avg += heatmap(&c1, &circuit_a, 64, 500, false);
        }
        println!("randomized avg: {}", avg/100.0);
        let c_str = circuit_a.repr();
        File::create("circuit_randomized.txt")
            .and_then(|mut f| f.write_all(c_str.as_bytes()))
            .expect("Failed to write test_compression.txt");
    }

    #[test]
    fn test_skeleton() {
        let contents = fs::read_to_string("circuit_before_random.txt")
            .expect("Failed to read");
        let circuit_a = CircuitSeq::from_string(&contents);

        let (_, skel) = create_skeleton(&circuit_a);
        let mut visited: HashSet<usize> = HashSet::new();
        let mut queue: Vec<usize> = skel.nodes[0].iter().map(|n| n.key).collect();

        while let Some(key) = queue.pop() {
            if visited.contains(&key) {
                continue;
            }
            visited.insert(key);

            // Get the node
            let node = skel
                .nodes
                .iter()
                .flat_map(|lvl| lvl)
                .find(|n| n.key == key)
                .expect("Node key not found");

            for &child_key in &node.children {
                if !visited.contains(&child_key) {
                    queue.push(child_key);
                }
            }
        }

        // Check all nodes were visited
        let total_nodes: usize = skel.nodes.iter().map(|lvl| lvl.len()).sum();
        if visited.len() != total_nodes {
            panic!(
                "Skeleton broken: visited {} nodes, but total nodes = {}",
                visited.len(),
                total_nodes
            );
        }

        println!("Skeleton test passed: all {} nodes reachable from level 0", total_nodes);
    }

    #[test]
    fn test_build_circuit() {
        let gate = "5hx;";
        let (r, a) = random_id(64, 20);

        let combined = format!("{}{}{}", r.repr(), gate, a.repr());

        let mut file = File::create("circuitRxR.txt").expect("Failed to create file");
        file.write_all(combined.as_bytes())
            .expect("Failed to write to file");

        println!("Wrote circuit to circuitRxR.txt:\n{}", combined);
    }

    #[test]
    fn test_gen_id() {
        let (r, a) = random_id(64, 500);
        let id = r.concat(&a).repr();
        let mut file = File::create("circuitlongid.txt").expect("Failed to create file");
        file.write_all(id.as_bytes())
            .expect("Failed to write to file");
    }
    #[test]
    fn benchmark_sql_vs_canonical() {
        use std::time::{Duration, Instant};
        use duckdb::Connection;
        use itertools::Itertools; // for permutations
        use lmdb::Environment;
        use std::path::Path;
        use lmdb::Transaction;
        let config = Config::default().access_mode(AccessMode::ReadOnly).unwrap();
        let conn = Connection::open_with_flags("circuits.duckdb", config).unwrap();

        let ns_and_ms = vec![(4, 6), (5, 5), (6, 4), (7, 3)];

        let mut stmts_prepared_limit1 = HashMap::new();
        for &(n, max_m) in &ns_and_ms {
            for m in 1..=max_m {
                let table = format!("n{}m{}", n, m);
                let query_limit = format!("SELECT perm, shuf FROM {} WHERE circuit = ?1 LIMIT 1", table);
                let stmt_limit = conn.prepare(&query_limit).unwrap();
                stmts_prepared_limit1.insert((n, m), stmt_limit);
            }
        }
        // Generate bit_shufs for get_canonical
        let bit_shufs: HashMap<usize, Vec<Vec<usize>>> = (3..=7)
            .map(|n| {
                let perms: Vec<Vec<usize>> = (0..n).permutations(n).collect();
                let shuf = perms.into_iter().skip(1).collect();
                (n, shuf)
            })
            .collect();

        // Timers
        let mut timer_canonical = HashMap::new();
        let mut timer_sql_prepared_limit1 = HashMap::new();
        let mut timer_lmdb = HashMap::new();

        for &(n, max_m) in &ns_and_ms {
            for m in 1..=max_m {
                timer_canonical.insert((n, m), Duration::ZERO);
                timer_sql_prepared_limit1.insert((n, m), Duration::ZERO);
                timer_lmdb.insert((n, m), Duration::ZERO);
            }
        }

        // --- Warmup: run some random circuits to prime caches ---
        for _ in 0..100_000 {
            for &(n, max_m) in &ns_and_ms {
                for m in 1..=max_m {
                    let mut circuit = random_circuit(n, m);
                    circuit.canonicalize();
                    let _ = circuit.repr_blob();
                    let _ = circuit.permutation(n);
                    // SQL warmup if needed
                    let table = format!("n{}m{}", n, m);
                    let query_limit = format!("SELECT perm, shuf FROM {} WHERE circuit = ?1 LIMIT 1", table);
                    let _ = conn.prepare(&query_limit).ok();
                }
            }
        }

        let env = Environment::new()
            .set_max_dbs(155)
            .open(Path::new("./db"))
            .expect("Failed to open LMDB env");

        // Open all nXmYperms DBs
        let mut lmdb_dbs = HashMap::new();
        for &(n, max_m) in &ns_and_ms {
            for m in 1..=max_m {
                let db_name = format!("n{}m{}perms", n, m);
                let db = env.open_db(Some(&db_name)).expect("Failed to open LMDB db");
                lmdb_dbs.insert((n, m), db);
            }
        }

        for _ in 0..1_000_000 {
            for &(n, max_m) in &ns_and_ms {
                for m in 1..=max_m {
                    let mut circuit = random_circuit(n, m);
                    circuit.canonicalize();
                    let circuit_blob = circuit.repr_blob();
                    let bit_shuf = &bit_shufs[&n];

                    // 1. get_canonical
                    let start = Instant::now();
                    let perm = circuit.permutation(n);
                    let _ = get_canonical(&perm, bit_shuf);
                    timer_canonical.entry((n, m)).and_modify(|d| *d += start.elapsed());

                    // 2. SQL prepared LIMIT 1
                    if let Some(stmt) = stmts_prepared_limit1.get_mut(&(n, m)) {
                        let start = Instant::now();
                        let _res: Option<(Vec<u8>, Vec<u8>)> =
                            stmt.query_row([&circuit_blob], |row| Ok((row.get(0)?, row.get(1)?))).ok();
                        timer_sql_prepared_limit1.entry((n, m)).and_modify(|d| *d += start.elapsed());
                    }

                    // 3. LMDB lookup
                    if let Some(&db) = lmdb_dbs.get(&(n, m)) {
                        let start = Instant::now();
                        let txn = env.begin_ro_txn().unwrap();
                        let _res = txn.get(db, &circuit_blob).ok();
                        timer_lmdb.entry((n, m)).and_modify(|d| *d += start.elapsed());
                    }
                }
            }
        }

        // --- Print results ---
        for &(n, max_m) in &ns_and_ms {
            for m in 1..=max_m {
                println!(
                    "n={} m={} | get_canonical: {:?} | SQL prepared LIMIT1: {:?} | LMDB: {:?}",
                    n,
                    m,
                    timer_canonical[&(n, m)],
                    timer_sql_prepared_limit1[&(n, m)],
                    timer_lmdb[&(n, m)],
                );
            }
        }
    }

    #[test]
    fn benchmark_rocksdb_vs_duckdb_vs_canonical() {
        use std::time::{Duration, Instant};
        use duckdb::Connection;
        use itertools::Itertools;
        use rocksdb::{DB, Options};
        use std::collections::HashMap;

        let ns_and_ms = vec![(6, 5), (7, 4)];

        // open dbs
        let config = Config::default().access_mode(AccessMode::ReadOnly).unwrap();
        let conn = Connection::open_with_flags("circuits.duckdb", config).unwrap();

        let mut duckdb_stmts = HashMap::new();
        for &(n, m) in &ns_and_ms {
            let table = format!("n{}m{}perms", n, m);
            let query = format!(
                "SELECT perm_shuf FROM {} WHERE circuit_hash = hash($1) AND circuit = $1 LIMIT 1",
                table
            );
            let stmt = conn.prepare(&query).unwrap();
            duckdb_stmts.insert((n, m), stmt);
        }

        let mut rocksdb_dbs = HashMap::new();
        for &(n, m) in &ns_and_ms {
            let path = format!("rocksdb_n{}m{}perms", n, m);
            let db = DB::open_for_read_only(&Options::default(), &path, false)
                .expect("Failed to open RocksDB");
            rocksdb_dbs.insert((n, m), db);
        }

        let bit_shufs: HashMap<usize, Vec<Vec<usize>>> = (3..=7)
            .map(|n| {
                let perms: Vec<Vec<usize>> = (0..n).permutations(n).collect();
                let shuf = perms.into_iter().skip(1).collect();
                (n, shuf)
            })
            .collect();

        // timers
        let mut timer_canonical: HashMap<(usize, usize), Duration> = HashMap::new();
        let mut timer_duckdb: HashMap<(usize, usize), Duration> = HashMap::new();
        let mut timer_rocksdb: HashMap<(usize, usize), Duration> = HashMap::new();

        for &(n, m) in &ns_and_ms {
            timer_canonical.insert((n, m), Duration::ZERO);
            timer_duckdb.insert((n, m), Duration::ZERO);
            timer_rocksdb.insert((n, m), Duration::ZERO);
        }

        println!("Warming up...");
        for _ in 0..10_000 {
            for &(n, m) in &ns_and_ms {
                let mut circuit = random_circuit(n, m);
                circuit.canonicalize();
                let _ = circuit.repr_blob();
                let _ = circuit.permutation(n);
            }
        }

        println!("Running benchmark...");
        let iters = 10;

        for _ in 0..iters {
            for &(n, m) in &ns_and_ms {
                let mut circuit = random_circuit(n, m);
                circuit.canonicalize();
                let circuit_blob = circuit.repr_blob();
                let bit_shuf = &bit_shufs[&n];
                println!("Get canonical");
                // get_canonical
                let start = Instant::now();
                let perm = circuit.permutation(n);
                let _ = get_canonical(&perm, bit_shuf);
                timer_canonical.entry((n, m)).and_modify(|d| *d += start.elapsed());
                println!("duckdb");
                // DuckDB lookup
                if let Some(stmt) = duckdb_stmts.get_mut(&(n, m)) {
                    let start = Instant::now();
                    let _res: Option<Vec<u8>> = stmt
                        .query_row([&circuit_blob], |row| row.get(0))
                        .ok();
                    timer_duckdb.entry((n, m)).and_modify(|d| *d += start.elapsed());
                }
                println!("rocksdb");
                // RocksDB lookup
                if let Some(db) = rocksdb_dbs.get(&(n, m)) {
                    let start = Instant::now();
                    let _res = db.get(&circuit_blob).ok();
                    timer_rocksdb.entry((n, m)).and_modify(|d| *d += start.elapsed());
                }
            }
        }

        // Print results
        println!("\nResults over {} iterations:", iters);
        println!("{:<10} {:<10} {:<20} {:<20} {:<20}", "n", "m", "get_canonical", "duckdb", "rocksdb");
        println!("{}", "-".repeat(80));
        for &(n, m) in &ns_and_ms {
            println!(
                "{:<10} {:<10} {:<20?} {:<20?} {:<20?}",
                n, m,
                timer_canonical[&(n, m)] / iters,
                timer_duckdb[&(n, m)] / iters,
                timer_rocksdb[&(n, m)] / iters,
            );
        }
    }

    use lmdb::Database;
    use std::path::Path;
    use lmdb::Environment;
    use lmdb::Transaction;
    use lmdb::Cursor;
    use crate::replace::pairs::GatePair;
    // use rand::prelude::IteratorRandom;
    #[test]
    fn test_random_circuit_identity() {
        let id = Permutation::id_perm(1 << 7);

        // Open LMDB
        let env_path = "./db";
        let env = Environment::new()
            .set_max_dbs(155)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new(env_path))
            .expect("Failed to open LMDB env");

        let tables = ["ids_n5", "ids_n6", "ids_n7"];
        let mut circuit_table: HashMap<Vec<u8>, Vec<Vec<u8>>> = HashMap::new();

        for table_name in tables {
            let db: Database = env.open_db(Some(table_name)).expect("DB not found");
            let txn = env.begin_ro_txn().expect("Failed to begin txn");
            let mut cursor = txn.open_ro_cursor(db).expect("Failed to open cursor");

            for (key_bytes, value_bytes) in cursor.iter() {
                let circuits: Vec<Vec<u8>> = bincode::deserialize(value_bytes)
                    .expect("Failed to deserialize circuits");

                let entry = circuit_table.entry(key_bytes.to_vec()).or_default();
                entry.extend(circuits);
            }
        }

        for (key, circuits) in circuit_table.iter() {
            for blob in circuits {
                let circuit = CircuitSeq::from_blob(blob);
                let tax: GatePair = bincode::deserialize(&key).expect("Can not recover Gate Pair");
                assert_eq!(
                    circuit.permutation(7),
                    id,
                    "Circuit for key {:?} is not an identity!",
                    key
                );
                assert_eq!(
                    tax,
                    gate_pair_taxonomy(&circuit.gates[0], &circuit.gates[1]),
                    "The gate taxonomy does not match the key {:?} vs {:?}",
                    gate_pair_taxonomy(&circuit.gates[0], &circuit.gates[1]),
                    tax
                );
            }
        }

        println!(
            "All circuits for all keys in tables {:?} are verified as identities!",
            tables
        );
    }

    fn gen_mean(circuit: CircuitSeq, num_wires: usize) -> f64 {
        let circuit_one = circuit.clone();
        let circuit_two = circuit;

        let circuit_one_len = circuit_one.gates.len();
        let circuit_two_len = circuit_two.gates.len();

        let num_points = (circuit_one_len + 1) * (circuit_two_len + 1);
        let mut average = vec![0f64; num_points * 3];

        let mut rng = rand::rng();
        let num_inputs = 20;

        for i in 0..num_inputs {
            if i % 10 == 0 {
                println!("{}/{}", i, num_inputs);
                io::stdout().flush().unwrap();
            }

            let input_bits: u128 = if num_wires < u128::BITS as usize {
                rng.random_range(0..(1u128 << num_wires))
            } else {
                rng.random_range(0..=u128::MAX)
            };

            let evolution_one = circuit_one.evaluate_evolution_128(input_bits);
            let evolution_two = circuit_two.evaluate_evolution_128(input_bits);

            for i1 in 0..=circuit_one_len {
                for i2 in 0..=circuit_two_len {
                    let diff = evolution_one[i1] ^ evolution_two[i2];
                    let hamming_dist = diff.count_ones() as f64;
                    let overlap = hamming_dist / num_wires as f64;

                    let index = i1 * (circuit_two_len + 1) + i2;
                    average[index * 3] = i1 as f64;
                    average[index * 3 + 1] = i2 as f64;
                    average[index * 3 + 2] += overlap / num_inputs as f64;
                }
            }
        }

        let mut sum = 0.0;
        for i in 0..num_points {
            sum += average[i * 3 + 2];
        }

        sum / num_points as f64
    }


    #[test]
    fn test_means() -> Result<(), Box<dyn std::error::Error>> {
        use lmdb::{Environment, Cursor, Transaction};
        use std::collections::HashMap;
        use std::fs::File;
        use std::io::{BufWriter, Write};

        let num_wires = 16;

        let db_names = [
            "ids_n16g0", "ids_n16g1", "ids_n16g2", "ids_n16g3", "ids_n16g4",
            "ids_n16g5", "ids_n16g6", "ids_n16g7", "ids_n16g8", "ids_n16g9",
            "ids_n16g10", "ids_n16g11", "ids_n16g12", "ids_n16g13", "ids_n16g14",
            "ids_n16g15", "ids_n16g16", "ids_n16g17", "ids_n16g18", "ids_n16g19",
            "ids_n16g20", "ids_n16g21", "ids_n16g22", "ids_n16g23", "ids_n16g24",
            "ids_n16g25", "ids_n16g26", "ids_n16g27", "ids_n16g28", "ids_n16g29",
            "ids_n16g30", "ids_n16g31", "ids_n16g32", "ids_n16g33",
        ];

        let env = Environment::new()
            .set_max_dbs(64)
            .open(Path::new("./db"))?; // adjust path

        let mut dbs = HashMap::new();
        for name in db_names {
            let db = env.open_db(Some(name))?;
            dbs.insert(name, db);
        }

        let file = File::create("means.txt")?;
        let mut writer = BufWriter::new(file);

        let txn = env.begin_ro_txn()?;

        for (db_name, db) in dbs {
            println!("Processing DB {}", db_name);

            let mut cursor = txn.open_ro_cursor(db)?;

            for (c_bytes, _) in cursor.iter() {
                let circuit = CircuitSeq::from_blob(&c_bytes);
            
                let mean = gen_mean(circuit, num_wires);

                writeln!(writer, "{}", mean)?;
                
            }
        }

        writer.flush()?;
        Ok(())
    }

    #[test]
    fn collect_odd_identity_keys() {
        let env_path = "./db";
        let env = Environment::new()
            .set_max_dbs(200)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new(env_path))
            .expect("Failed to open LMDB env");

        let tables = [
            "ids_n5g1", "ids_n5g2", "ids_n5g3", "ids_n5g4", "ids_n5g5",
            "ids_n5g6", "ids_n5g7", "ids_n5g8", "ids_n5g9", "ids_n5g10", "ids_n5g11",
            "ids_n5g12", "ids_n5g13", "ids_n5g14", "ids_n5g15", "ids_n5g16",
            "ids_n5g17", "ids_n5g18", "ids_n5g19", "ids_n5g20", "ids_n5g21",
            "ids_n5g22", "ids_n5g23", "ids_n5g24", "ids_n5g25", "ids_n5g26",
            "ids_n5g27", "ids_n5g28", "ids_n5g29", "ids_n5g30", "ids_n5g31",
            "ids_n5g32", "ids_n5g33",

            "ids_n6g0", "ids_n6g1", "ids_n6g2", "ids_n6g3", "ids_n6g4", "ids_n6g5",
            "ids_n6g6", "ids_n6g7", "ids_n6g8", "ids_n6g9", "ids_n6g10", "ids_n6g11",
            "ids_n6g12", "ids_n6g13", "ids_n6g14", "ids_n6g15", "ids_n6g16",
            "ids_n6g17", "ids_n6g18", "ids_n6g19", "ids_n6g20", "ids_n6g21",
            "ids_n6g22", "ids_n6g23", "ids_n6g24", "ids_n6g25", "ids_n6g26",
            "ids_n6g27", "ids_n6g28", "ids_n6g29", "ids_n6g30", "ids_n6g31",
            "ids_n6g32", "ids_n6g33",

            "ids_n7g0", "ids_n7g1", "ids_n7g2", "ids_n7g3", "ids_n7g4", "ids_n7g5",
            "ids_n7g6", "ids_n7g7", "ids_n7g8", "ids_n7g9", "ids_n7g10", "ids_n7g11",
            "ids_n7g12", "ids_n7g13", "ids_n7g14", "ids_n7g15", "ids_n7g16",
            "ids_n7g17", "ids_n7g18", "ids_n7g19", "ids_n7g20", "ids_n7g21",
            "ids_n7g22", "ids_n7g23", "ids_n7g24", "ids_n7g25", "ids_n7g26",
            "ids_n7g27", "ids_n7g28", "ids_n7g29", "ids_n7g30", "ids_n7g31",
            "ids_n7g32", "ids_n7g33",
        ];

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("odd_ids.txt")
            .expect("Failed to open odd_ids.txt");

        for table_name in tables {
            let db: Database = env
                .open_db(Some(table_name))
                .expect("DB not found");

            let txn = env.begin_ro_txn().expect("Failed to begin txn");
            let mut cursor = txn
                .open_ro_cursor(db)
                .expect("Failed to open cursor");

            for (key_bytes, _) in cursor.iter() {
                let circuit = CircuitSeq::from_blob(key_bytes);
                let used_wires = circuit.used_wires().len();
                if circuit.gates.len() % 2 == 1 {
                    writeln!(
                        file,
                        "table={}, gates_len={}, circuit = {}, wire_count = {}",
                        table_name,
                        circuit.gates.len(),
                        circuit.repr(),
                        used_wires,
                    )
                    .expect("Failed to write");
                }
            }
        }
    }

    #[test]
    fn find_swaps() {
        //cnot
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("c1not2.txt")
            .expect("Failed to open swaponlyn.txt");
        let mut circuits: HashSet<CircuitSeq> = HashSet::new();
        let perm = Permutation { data: vec![0,1,6,7,4,5,2,3]};
        for m in 2..=10 {
            for _ in 0..100000 {
                let mut random = random_circuit(3, m);
                random.canonicalize();
                let mut i = 0;
                while i < random.gates.len().saturating_sub(1) {
                    if random.gates[i] == random.gates[i + 1] {
                        random.gates.drain(i..=i + 1);
                        i = i.saturating_sub(2);
                    } else {
                        i += 1;
                    }
                }
                if random.permutation(3) == perm {
                    circuits.insert(random);
                }
            }
        }

        for c in circuits {
            writeln!(file, "{}", c.repr()).expect("Failed to write to swaponlyn.txt");
        }

        //swap 1 2
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("swap12.txt")
            .expect("Failed to open swaponlyn.txt");
        let mut circuits: HashSet<CircuitSeq> = HashSet::new();
        let perm = Permutation { data: vec![0,1,4,5,2,3,6,7]};
        for m in 6..=15 {
            for _ in 0..100000 {
                let mut random = random_circuit(3, m);
                random.canonicalize();
                let mut i = 0;
                while i < random.gates.len().saturating_sub(1) {
                    if random.gates[i] == random.gates[i + 1] {
                        random.gates.drain(i..=i + 1);
                        i = i.saturating_sub(2);
                    } else {
                        i += 1;
                    }
                }
                if random.permutation(3) == perm {
                    circuits.insert(random);
                }
            }
        }

        for c in circuits {
            writeln!(file, "{}", c.repr()).expect("Failed to write to swaponlyn.txt");
        }

        //swap 1 2 and then negate 1
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("swap12n1.txt")
            .expect("Failed to open swaponlyn.txt");
        let mut circuits: HashSet<CircuitSeq> = HashSet::new();
        let perm = Permutation { data: vec![2,3,6,7,0,1,4,5]};
        for m in 6..=15 {
            for _ in 0..100000 {
                let mut random = random_circuit(3, m);
                random.canonicalize();
                let mut i = 0;
                while i < random.gates.len().saturating_sub(1) {
                    if random.gates[i] == random.gates[i + 1] {
                        random.gates.drain(i..=i + 1);
                        i = i.saturating_sub(2);
                    } else {
                        i += 1;
                    }
                }
                if random.permutation(3) == perm {
                    circuits.insert(random);
                }
            }
        }

        for c in circuits {
            writeln!(file, "{}", c.repr()).expect("Failed to write to swaponlyn.txt");
        }

        //swap 1 2 and then negate 2
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("swap12n2.txt")
            .expect("Failed to open swaponlyn.txt");
        let mut circuits: HashSet<CircuitSeq> = HashSet::new();
        let perm = Permutation { data: vec![4,5,0,1,6,7,2,3]};
        for m in 6..=15 {
            for _ in 0..100000 {
                let mut random = random_circuit(3, m);
                random.canonicalize();
                let mut i = 0;
                while i < random.gates.len().saturating_sub(1) {
                    if random.gates[i] == random.gates[i + 1] {
                        random.gates.drain(i..=i + 1);
                        i = i.saturating_sub(2);
                    } else {
                        i += 1;
                    }
                }
                if random.permutation(3) == perm {
                    circuits.insert(random);
                }
            }
        }

        for c in circuits {
            writeln!(file, "{}", c.repr()).expect("Failed to write to swaponlyn.txt");
        }

        //swap 1 2 and then negate both
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("swap12n1n2.txt")
            .expect("Failed to open swaponlyn.txt");
        let mut circuits: HashSet<CircuitSeq> = HashSet::new();
        let perm = Permutation { data: vec![6,7,2,3,4,5,0,1]};
        for m in 6..=15 {
            for _ in 0..100000 {
                let mut random = random_circuit(3, m);
                random.canonicalize();
                let mut i = 0;
                while i < random.gates.len().saturating_sub(1) {
                    if random.gates[i] == random.gates[i + 1] {
                        random.gates.drain(i..=i + 1);
                        i = i.saturating_sub(2);
                    } else {
                        i += 1;
                    }
                }
                if random.permutation(3) == perm {
                    circuits.insert(random);
                }
            }
        }

        for c in circuits {
            writeln!(file, "{}", c.repr()).expect("Failed to write to swaponlyn.txt");
        }

        //not 
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("not1.txt")
            .expect("Failed to open swaponlyn.txt");
        let mut circuits: HashSet<CircuitSeq> = HashSet::new();
        let perm = Permutation { data: vec![2,3,0,1,6,7,4,5]};
        for m in 2..=10 {
            for _ in 0..100000 {
                let mut random = random_circuit(3, m);
                if random.permutation(3) == perm {
                    println!("Found not1 candidate: {}", random.repr());
                }
                random.canonicalize();
                let mut i = 0;
                while i < random.gates.len().saturating_sub(1) {
                    if random.gates[i] == random.gates[i + 1] {
                        random.gates.drain(i..=i + 1);
                        i = i.saturating_sub(2);
                    } else {
                        i += 1;
                    }
                }
                if random.permutation(3) == perm {
                    circuits.insert(random);
                }
            }
        }

        for c in circuits {
            writeln!(file, "{}", c.repr()).expect("Failed to write to swaponlyn.txt");
        }
    }

    #[test]
    fn test_read_swap_and_print_perm() {
        let env = Environment::new()
            .set_max_dbs(202)
            .open(Path::new("./db"))
            .expect("Failed to open LMDB environment");

        let db = env
            .open_db(Some("not1"))
            .expect("Failed to open 'swap' database");

        // Begin a read-only transaction
        let txn = env.begin_ro_txn().expect("Failed to begin read-only transaction");

        // Iterate over all key-value pairs in the "swap" db
        let mut cursor = txn.open_ro_cursor(db).expect("Failed to open cursor");

        for (key, _) in cursor.iter() {
            // Deserialize the blob into a CircuitSeq
            let circuit = CircuitSeq::from_blob(key);

            // Convert the circuit to a permutation
            let perm = circuit.permutation(3);

            println!("Permutation for circuit '{:?}': {:?}", circuit.repr(), perm);
        }
    }

    #[test]
    fn load_swaps_into_lmdb() {
        use std::{fs::File, io::{BufRead, BufReader}, path::Path};
        use lmdb::{Environment, DatabaseFlags, WriteFlags};

        let env = Environment::new()
            .set_max_dbs(263)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new("./db"))
            .expect("failed to open lmdb env");

        // Delete existing databases before creating them (ignore errors if they don't exist)
        let dbs_to_delete = ["cnot", "not", "swapnot12", "swap", "swapnot1", "swapnot2"];
        for db_name in dbs_to_delete.iter() {
        if let Ok(db) = env.open_db(Some(db_name)) {
            let mut txn = env.begin_rw_txn().expect("Failed to begin txn");
            // SAFETY: ensure no other transactions or handles are active
            unsafe {
                txn.drop_db(db).expect("Failed to drop db");
            }
            txn.commit().expect("Failed to commit txn");
            println!("Dropped DB: {}", db_name);
        } else {
            println!("DB not found: {}", db_name);
        }
    }

        let db = env
            .create_db(Some("cnot"), DatabaseFlags::empty())
            .expect("failed to create/open db");

        let file = File::open("c1not2.txt").expect("failed to open c1not2.txt");
        let reader = BufReader::new(file);

        let mut txn = env.begin_rw_txn().expect("failed to start txn");

        for line in reader.lines() {
            let line = line.expect("failed to read line");
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let circuit = CircuitSeq::from_string(line);
            let key = circuit.repr_blob();

            txn.put(db, &key, &[], WriteFlags::NO_OVERWRITE)
                .expect("lmdb put failed");
        }

        txn.commit().expect("txn commit failed");

        let db = env
            .create_db(Some("not"), DatabaseFlags::empty())
            .expect("failed to create/open db");

        let file = File::open("not1.txt").expect("failed to open not1.txt");
        let reader = BufReader::new(file);

        let mut txn = env.begin_rw_txn().expect("failed to start txn");

        for line in reader.lines() {
            let line = line.expect("failed to read line");
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let circuit = CircuitSeq::from_string(line);
            let key = circuit.repr_blob();

            txn.put(db, &key, &[], WriteFlags::NO_OVERWRITE)
                .expect("lmdb put failed");
        }

        txn.commit().expect("txn commit failed");

        let db = env
            .create_db(Some("swapnot12"), DatabaseFlags::empty())
            .expect("failed to create/open db");

        let file = File::open("swap12n1n2.txt").expect("failed to open swap12n1n2.txt");
        let reader = BufReader::new(file);

        let mut txn = env.begin_rw_txn().expect("failed to start txn");

        for line in reader.lines() {
            let line = line.expect("failed to read line");
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let circuit = CircuitSeq::from_string(line);
            let key = circuit.repr_blob();

            txn.put(db, &key, &[], WriteFlags::NO_OVERWRITE)
                .expect("lmdb put failed");
        }

        txn.commit().expect("txn commit failed");

        let db = env
            .create_db(Some("swap"), DatabaseFlags::empty())
            .expect("failed to create/open db");

        let file = File::open("swap12.txt").expect("failed to open swap12.txt");
        let reader = BufReader::new(file);

        let mut txn = env.begin_rw_txn().expect("failed to start txn");

        for line in reader.lines() {
            let line = line.expect("failed to read line");
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let circuit = CircuitSeq::from_string(line);
            let key = circuit.repr_blob();

            txn.put(db, &key, &[], WriteFlags::NO_OVERWRITE)
                .expect("lmdb put failed");
        }

        txn.commit().expect("txn commit failed");

        let db = env
            .create_db(Some("swapnot1"), DatabaseFlags::empty())
            .expect("failed to create/open db");

        let file = File::open("swap12n1.txt").expect("failed to open swap12n1.txt");
        let reader = BufReader::new(file);

        let mut txn = env.begin_rw_txn().expect("failed to start txn");

        for line in reader.lines() {
            let line = line.expect("failed to read line");
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let circuit = CircuitSeq::from_string(line);
            let key = circuit.repr_blob();

            txn.put(db, &key, &[], WriteFlags::NO_OVERWRITE)
                .expect("lmdb put failed");
        }

        txn.commit().expect("txn commit failed");

        let db = env
            .create_db(Some("swapnot2"), DatabaseFlags::empty())
            .expect("failed to create/open db");

        let file = File::open("swap12n2.txt").expect("failed to open swap12n2.txt");
        let reader = BufReader::new(file);

        let mut txn = env.begin_rw_txn().expect("failed to start txn");

        for line in reader.lines() {
            let line = line.expect("failed to read line");
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let circuit = CircuitSeq::from_string(line);
            let key = circuit.repr_blob();

            txn.put(db, &key, &[], WriteFlags::NO_OVERWRITE)
                .expect("lmdb put failed");
        }

        txn.commit().expect("txn commit failed");
    }

    #[test]
    fn test_print_all_circuits() {
        let m = 2;
        let db = Arc::new(open_db_for_read(m));
        let iter = db.iterator(rocksdb::IteratorMode::Start);

        let mut count = 0;
        for item in iter {
            let (key, value) = item.expect("RocksDB iter error");

            // Print the key as hex
            // println!("Key: {}", key_hex);

            // Scan all circuits in the value list
            let mut pos = 0;
            let mut circuit_index = 0;
            while pos < value.len() {
                if pos + 1 > value.len() {
                    break;
                }
                let len = value[pos] as usize;
                pos += 1;
                if pos + len > value.len() {
                    break;
                }
                let circuit_blob = &value[pos..pos + len];
                let circuit = CircuitSeq::from_blob(circuit_blob);
                println!("{} == {}: {}", count, circuit_index,circuit.repr());
                pos += len;
                circuit_index += 1;
                count += 1;
            }
        }

        println!("Total circuits: {}", count);
    }

    #[test]
    fn test_base_gate_canonicalization() {
        use crate::circuit::circuit::poly_to_str;
        let gates = base_gates(3);

        for g in gates.iter() {
            let c = CircuitSeq { gates: vec![*g] };
            println!("Gate: {:?}", c.repr());
            let canon = canonicalize_polys(c.to_polynomial(3, 0, 1), true, false);
            let mut c = CircuitSeq { gates: vec![*g] };
            c.rewire(&canon.1.invert(), 3);
            println!("Wiring: {:?}", canon.1);
            println!("Rewired circuit: {}", c.repr());
            for (i, poly) in canon.0.iter().enumerate() {
                println!("  P{}: {}", i, poly_to_str(poly, 3));
            }
            println!();
        }
    }

    #[test]
    fn test_count_circuits_per_table() {
        for m in 1..=5 {
            let db = Arc::new(open_db_for_read(m));
            let iter = db.iterator(rocksdb::IteratorMode::Start);

            let mut key_count = 0;
            let mut circuit_count = 0;

            for item in iter {
                let (_key, value) = item.expect("RocksDB iter error");
                key_count += 1;

                let mut pos = 0;
                while pos < value.len() {
                    if pos + 1 > value.len() {
                        break;
                    }
                    let len = value[pos] as usize;
                    pos += 1;
                    if pos + len > value.len() {
                        break;
                    }
                    pos += len;
                    circuit_count += 1;
                }
            }

            println!("m={}: {} keys, {} circuits", m, key_count, circuit_count);
        }
    }

   #[test]
    fn test_compare_two_m4_dbs() {
        let db1 = Arc::new({
            let path = "rocks_db_m4";
            let mut opts = Options::default();
            opts.create_if_missing(false);
            opts.set_merge_operator_associative("append_merge", append_merge);
            opts.increase_parallelism(160);
            opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(16));
            let cache = Cache::new_lru_cache(4 * 1024 * 1024 * 1024);
            let mut block_opts = BlockBasedOptions::default();
            block_opts.set_block_cache(&cache);
            block_opts.set_block_size(16 * 1024);
            block_opts.set_bloom_filter(10.0, false);
            block_opts.set_cache_index_and_filter_blocks(true);
            block_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);
            opts.set_block_based_table_factory(&block_opts);
            opts.set_disable_auto_compactions(true);
            DB::open_for_read_only(&opts, path, false).expect("Failed to open rocks_db_m4")
        });

        let db2 = Arc::new({
            let path = "test_rocks_db_m4";
            let mut opts = Options::default();
            opts.create_if_missing(false);
            opts.set_merge_operator_associative("append_merge", append_merge);
            opts.increase_parallelism(160);
            opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(16));
            let cache = Cache::new_lru_cache(4 * 1024 * 1024 * 1024);
            let mut block_opts = BlockBasedOptions::default();
            block_opts.set_block_cache(&cache);
            block_opts.set_block_size(16 * 1024);
            block_opts.set_bloom_filter(10.0, false);
            block_opts.set_cache_index_and_filter_blocks(true);
            block_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);
            opts.set_block_based_table_factory(&block_opts);
            opts.set_disable_auto_compactions(true);
            DB::open_for_read_only(&opts, path, false).expect("Failed to open test_rocks_db_m4")
        });

        let m = 4;
        let n = 3 * m;

        // Count total circuits and hashes in db1
        let mut db1_total_circuits = 0usize;
        let mut db1_total_hashes = 0usize;
        {
            let iter = db1.iterator(rocksdb::IteratorMode::Start);
            for item in iter {
                let (_key, value) = item.expect("RocksDB iter error");
                db1_total_hashes += 1;
                let mut pos = 0;
                while pos < value.len() {
                    if pos + 1 > value.len() { break; }
                    let len = value[pos] as usize;
                    pos += 1;
                    if pos + len > value.len() { break; }
                    pos += len;
                    db1_total_circuits += 1;
                }
            }
        }

        // Count total circuits and hashes in db2
        let mut db2_total_circuits = 0usize;
        let mut db2_total_hashes = 0usize;
        {
            let iter = db2.iterator(rocksdb::IteratorMode::Start);
            for item in iter {
                let (_key, value) = item.expect("RocksDB iter error");
                db2_total_hashes += 1;
                let mut pos = 0;
                while pos < value.len() {
                    if pos + 1 > value.len() { break; }
                    let len = value[pos] as usize;
                    pos += 1;
                    if pos + len > value.len() { break; }
                    pos += len;
                    db2_total_circuits += 1;
                }
            }
        }

        println!("db1: {} circuits, {} hashes", db1_total_circuits, db1_total_hashes);
        println!("db2: {} circuits, {} hashes", db2_total_circuits, db2_total_hashes);

        let mut missing: Vec<CircuitSeq> = Vec::new();
        let mut found_directly = 0usize;
        let mut passed_reversal = 0usize;

        let iter = db1.iterator(rocksdb::IteratorMode::Start);
        for item in iter {
            let (_key, value) = item.expect("RocksDB iter error");
            let mut pos = 0;
            while pos < value.len() {
                if pos + 1 > value.len() { break; }
                let len = value[pos] as usize;
                pos += 1;
                if pos + len > value.len() { break; }
                let circuit_blob = &value[pos..pos + len];
                pos += len;

                let circuit = CircuitSeq::from_blob(circuit_blob);

                // Check direct hash match
                let canon = canonicalize_polys(circuit.to_polynomial(n, 0, m), true, false);
                let blob = polys_repr_blob(&canon.0);
                let hash: u128 = xxh3_128(&blob);
                let key = hash.to_le_bytes().to_vec();

                if db2.get(&key).unwrap_or(None).is_some() {
                    found_directly += 1;
                    continue;
                }

                // Check reversed circuit hash match
                let mut rev = circuit.clone();
                rev.gates.reverse();
                rev.canonicalize();
                let canon_rev = canonicalize_polys(rev.to_polynomial(n, 0, m), true, false);
                let rev_blob = polys_repr_blob(&canon_rev.0);
                let rev_hash: u128 = xxh3_128(&rev_blob);
                let rev_key = rev_hash.to_le_bytes().to_vec();

                if db2.get(&rev_key).unwrap_or(None).is_some() {
                    passed_reversal += 1;
                    continue;
                }

                missing.push(circuit);
            }
        }

        println!("Found directly in db2: {}", found_directly);
        println!("Passed reversal check: {}", passed_reversal);
        println!("Missing from db2 (not found directly, by reversal, or relabeling): {}", missing.len());

        if !missing.is_empty() {
            println!("First 10 missing circuits:");
            for circuit in missing.iter().take(10) {
                println!("  {:?}", circuit.gates);
            }
        }

        assert!(
            missing.is_empty(),
            "{} circuits from db1 not found in db2",
            missing.len()
        );
    }

    fn canonicalize_circuit(gates: Vec<[u8; 3]>, n: usize, m: usize) -> (CircuitSeq, Permutation) {
        let mut c = CircuitSeq { gates };
        let canon = canonicalize_polys(c.to_polynomial(n, 0, m), true, false);
        c.rewire(&canon.1.invert(), n);
        c.canonicalize();
        (c, canon.1)
    }
    #[test]
    fn test_eight_cases() {
        let mut c1 = CircuitSeq { gates: vec![[0,5,4], [1,4,6]] };
        let mut c2 = CircuitSeq { gates: vec![[2,7,5], [3,6,7]] };
        let canon1 = canonicalize_polys(c1.to_polynomial(8, 0, 2), true, false);
        let canon2 = canonicalize_polys(c2.to_polynomial(8, 0, 2), true, false);
        c1.rewire(&canon1.1.invert(), 8);
        c2.rewire(&canon2.1.invert(), 8);
        let n1 = touched_wires(&c1).len();
        let n2 = touched_wires(&c2).len();

        let c1_rev = CircuitSeq { gates: c1.gates.iter().rev().cloned().collect() };
        let c2_rev = CircuitSeq { gates: c2.gates.iter().rev().cloned().collect() };

        let mappings_1_2 = enumerate_c2_wire_mappings(n1, n2);
        let mappings_2_1 = enumerate_c2_wire_mappings(n2, n1);

        let mut f = std::fs::File::create("test.txt").unwrap();

        // --- latter part rotates ---

        // Case 1: c1 || mapped_c2
        writeln!(f, "=== Case 1: c1 || mapped_c2 ===").unwrap();
        for mapping in &mappings_1_2 {
            let c2_mapped = apply_wire_mapping(&c2, mapping);
            let mut combined = c1.gates.clone();
            combined.extend_from_slice(&c2_mapped.gates);
            let m_combined = combined.len();
            let (result, _) = canonicalize_circuit(combined, 3 * m_combined, m_combined);
            writeln!(f, "  {:?}", result.gates).unwrap();
        }

        // Case 2: c2 || mapped_c1
        writeln!(f, "\n=== Case 2: c2 || mapped_c1 ===").unwrap();
        for mapping in &mappings_2_1 {
            let c1_mapped = apply_wire_mapping(&c1, mapping);
            let mut combined = c2.gates.clone();
            combined.extend_from_slice(&c1_mapped.gates);
            let m_combined = combined.len();
            let (result, _) = canonicalize_circuit(combined, 3 * m_combined, m_combined);
            writeln!(f, "  {:?}", result.gates).unwrap();
        }

        // Case 3: c1_rev || mapped_c2
        writeln!(f, "\n=== Case 3: c1_rev || mapped_c2 ===").unwrap();
        for mapping in &mappings_1_2 {
            let c2_mapped = apply_wire_mapping(&c2, mapping);
            let mut combined = c1_rev.gates.clone();
            combined.extend_from_slice(&c2_mapped.gates);
            let m_combined = combined.len();
            let (result, _) = canonicalize_circuit(combined, 3 * m_combined, m_combined);
            writeln!(f, "  {:?}", result.gates).unwrap();
        }

        // Case 4: c2_rev || mapped_c1
        writeln!(f, "\n=== Case 4: c2_rev || mapped_c1 ===").unwrap();
        for mapping in &mappings_2_1 {
            let c1_mapped = apply_wire_mapping(&c1, mapping);
            let mut combined = c2_rev.gates.clone();
            combined.extend_from_slice(&c1_mapped.gates);
            let m_combined = combined.len();
            let (result, _) = canonicalize_circuit(combined, 3 * m_combined, m_combined);
            writeln!(f, "  {:?}", result.gates).unwrap();
        }

        // --- first part rotates ---

        // Case 5: mapped_c1 || c2
        writeln!(f, "\n=== Case 5: mapped_c1 || c2 ===").unwrap();
        for mapping in &mappings_2_1 {
            let c1_mapped = apply_wire_mapping(&c1, mapping);
            let mut combined = c1_mapped.gates.clone();
            combined.extend_from_slice(&c2.gates);
            let m_combined = combined.len();
            let (result, _) = canonicalize_circuit(combined, 3 * m_combined, m_combined);
            writeln!(f, "  {:?}", result.gates).unwrap();
        }

        // Case 6: mapped_c2 || c1
        writeln!(f, "\n=== Case 6: mapped_c2 || c1 ===").unwrap();
        for mapping in &mappings_1_2 {
            let c2_mapped = apply_wire_mapping(&c2, mapping);
            let mut combined = c2_mapped.gates.clone();
            combined.extend_from_slice(&c1.gates);
            let m_combined = combined.len();
            let (result, _) = canonicalize_circuit(combined, 3 * m_combined, m_combined);
            writeln!(f, "  {:?}", result.gates).unwrap();
        }

        // Case 7: mapped_c1 || c2_rev
        writeln!(f, "\n=== Case 7: mapped_c1 || c2_rev ===").unwrap();
        for mapping in &mappings_2_1 {
            let c1_mapped = apply_wire_mapping(&c1, mapping);
            let mut combined = c1_mapped.gates.clone();
            combined.extend_from_slice(&c2_rev.gates);
            let m_combined = combined.len();
            let (result, _) = canonicalize_circuit(combined, 3 * m_combined, m_combined);
            writeln!(f, "  {:?}", result.gates).unwrap();
        }

        // Case 8: mapped_c2 || c1_rev
        writeln!(f, "\n=== Case 8: mapped_c2 || c1_rev ===").unwrap();
        for mapping in &mappings_1_2 {
            let c2_mapped = apply_wire_mapping(&c2, mapping);
            let mut combined = c2_mapped.gates.clone();
            combined.extend_from_slice(&c1_rev.gates);
            let m_combined = combined.len();
            let (result, _) = canonicalize_circuit(combined, 3 * m_combined, m_combined);
            writeln!(f, "  {:?}", result.gates).unwrap();
        }

        // Case 9: full circuit
        writeln!(f, "\n=== Case 9: full circuit [[0,5,4],[1,4,6],[2,7,5],[3,6,7]] ===").unwrap();
        let hardcoded = vec![[0u8,5,4],[1,4,6],[2,7,5],[3,6,7]];
        let m_combined = hardcoded.len();
        let (result, _) = canonicalize_circuit(hardcoded, 8, m_combined);
        writeln!(f, "  {:?}", result.gates).unwrap();

        // Case 10: first part
        writeln!(f, "\n=== Case 10: first part [[0,5,4],[1,4,6]] ===").unwrap();
        let hardcoded = vec![[0u8,5,4],[1,4,6]];
        let m_combined = hardcoded.len();
        let (result,perm) = canonicalize_circuit(hardcoded, 8, m_combined);
        writeln!(f, "  {:?}", result.gates).unwrap();
        writeln!(f, "  {:?}", perm.data).unwrap();

        // Case 11: second part
        writeln!(f, "\n=== Case 11: second part [[2,7,5],[3,6,7]] ===").unwrap();
        let hardcoded = vec![[2u8,7,5],[3,6,7]];
        let m_combined = hardcoded.len();
        let (result, perm) = canonicalize_circuit(hardcoded, 8, m_combined);
        writeln!(f, "  {:?}", result.gates).unwrap();
        writeln!(f, "  {:?}", perm.data).unwrap();

        // Case 12: reversed first part
        writeln!(f, "\n=== Case 12: reversed first [[1,4,6],[0,5,4]] ===").unwrap();
        let hardcoded = vec![[1,4,6],[0,5,4]];
        let m_combined = hardcoded.len();
        let (result, perm) = canonicalize_circuit(hardcoded, 8, m_combined);
        writeln!(f, "  {:?}", result.gates).unwrap();
        writeln!(f, "  {:?}", perm.data).unwrap();

        // Case 13: reversed second part
        writeln!(f, "\n=== Case 13: reversed second [[3,6,7],[2,7,5]] ===").unwrap();
        let hardcoded = vec![[3,6,7],[2,7,5]];
        let m_combined = hardcoded.len();
        let (result, perm) = canonicalize_circuit(hardcoded, 8, m_combined);
        writeln!(f, "  {:?}", result.gates).unwrap();
        writeln!(f, "  {:?}", perm.data).unwrap();  

        // Case 14: Canon twice
        writeln!(f, "\n=== Case 14: Canon twice on [[1, 2, 3], [0, 4, 2]] ===").unwrap();
        let hardcoded = vec![[1u8, 2, 3], [0, 4, 2]];
        let m_combined = hardcoded.len();
        let (result, perm) = canonicalize_circuit(hardcoded, 8, m_combined);
        writeln!(f, "  After first canon: {:?}", result.gates).unwrap();
        writeln!(f, "  Permutation: {:?}", perm.data).unwrap();

        // Case 15: Two disjoint [[0, 4, 2], [1, 2, 3], [6, 7, 4], [5, 3, 7]]]
        writeln!(f, "\n=== Case 15: Two disjoint [[0, 4, 2], [1, 2, 3], [6, 7, 4], [5, 3, 7]] ===").unwrap();
        let hardcoded = vec![[0u8, 4, 2], [1, 2, 3], [6, 7, 4], [5, 3, 7]];
        let m_combined = hardcoded.len();
        let (result, perm) = canonicalize_circuit(hardcoded, 12, m_combined);
        writeln!(f, "  After first canon: {:?}", result.gates).unwrap();
        writeln!(f, "  Permutation: {:?}", perm.data).unwrap();

        // Case 16: Rewired second [[6, 7, 4], [5, 3, 7]]
        writeln!(f, "\n=== Case 16: Rewired second [[6, 7, 4], [5, 3, 7]] ===").unwrap();
        let hardcoded = vec![[6u8, 7, 4], [5, 3, 7]];
        let m_combined = hardcoded.len();
        let (result, perm) = canonicalize_circuit(hardcoded, 12, m_combined);
        writeln!(f, "  After first canon: {:?}", result.gates).unwrap();
        writeln!(f, "  Permutation: {:?}", perm.data).unwrap(); 
    }

    #[test]
    fn test_circuit_in_db() {
        let m = 2;
        let n = 3 * m;
        let db = Arc::new(open_db_for_read(m));

        let check = |gates: Vec<[u8; 3]>, label: &str| -> bool {
            let mut circuit = CircuitSeq { gates };
            let canon = canonicalize_polys(circuit.to_polynomial(n, 0, m), true, false);
            circuit.rewire(&canon.1.invert(), n);

            let canon = canonicalize_polys(circuit.to_polynomial(n, 0, m), true, false);
            let blob = polys_repr_blob(&canon.0);
            let hash: u128 = xxh3_128(&blob);
            let key = hash.to_le_bytes().to_vec();

            match db.get(&key).unwrap_or(None) {
                None => {
                    println!("{}: key not found in db", label);
                    false
                }
                Some(value) => {
                    println!("{}: key found, checking circuits in value...", label);
                    let mut pos = 0;
                    while pos < value.len() {
                        if pos + 1 > value.len() { break; }
                        let len = value[pos] as usize;
                        pos += 1;
                        if pos + len > value.len() { break; }
                        let circuit_blob = &value[pos..pos + len];
                        pos += len;
                        let candidate = CircuitSeq::from_blob(circuit_blob);
                        println!("  candidate: {:?}", candidate.gates);
                        if candidate.gates == circuit.gates {
                            println!("  -> exact match found!");
                            return true;
                        }
                    }
                    println!("  -> key found but no exact circuit match");
                    true
                }
            }
        };

        // let gates = vec![[1, 4, 2], [0, 3, 1]];
        let gates = vec![[1, 2, 3], [0, 4, 2]];
        let found = check(gates.clone(), "forward");

        if !found {
            let mut rev = gates.clone();
            rev.reverse();
            check(rev, "reversed");
        }
    }

    use std::collections::HashSet;

    #[test]
    fn list_circuits_up_to_reversal() {
        let db = Arc::new({
            let path = "rocks_db_m2";
            let mut opts = Options::default();
            opts.create_if_missing(false);
            opts.set_merge_operator_associative("append_merge", append_merge);
            opts.increase_parallelism(160);
            opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(16));

            let cache = Cache::new_lru_cache(4 * 1024 * 1024 * 1024);
            let mut block_opts = BlockBasedOptions::default();
            block_opts.set_block_cache(&cache);
            block_opts.set_block_size(16 * 1024);
            block_opts.set_bloom_filter(10.0, false);
            block_opts.set_cache_index_and_filter_blocks(true);
            block_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);

            opts.set_block_based_table_factory(&block_opts);
            opts.set_disable_auto_compactions(true);

            DB::open_for_read_only(&opts, path, false).expect("open failed")
        });

        let m = 2;
        let n = 3 * m;

        let mut seen: HashSet<u128> = HashSet::new();
        let mut reps = 0usize;

        let iter = db.iterator(rocksdb::IteratorMode::Start);
        for item in iter {
            let (_key, value) = item.expect("RocksDB iter error");

            let mut pos = 0;
            while pos < value.len() {
                let len = value[pos] as usize;
                pos += 1;

                let circuit_blob = &value[pos..pos + len];
                pos += len;

                let circuit = CircuitSeq::from_blob(circuit_blob);

                // hash of original
                let canon = canonicalize_polys_3(circuit.to_polynomial(n, 0, m));
                let blob = polys_repr_blob(&canon.0);
                let h: u128 = xxh3_128(&blob);

                // hash of reversed
                let mut rev = circuit.clone();
                rev.gates.reverse();
                rev.canonicalize();

                let canon_rev = canonicalize_polys_3(rev.to_polynomial(n, 0, m));
                let rev_blob = polys_repr_blob(&canon_rev.0);
                let h_rev: u128 = xxh3_128(&rev_blob);

                // choose canonical representative of the pair
                let rep = h.min(h_rev);

                if seen.insert(rep) {
                    reps += 1;
                    println!("{:?}", circuit.gates);
                }
            }
        }

        println!("number of reversal classes: {}", reps);
    }

    #[test]
    pub fn list_up_to_canon_and_rev() {
        let db = Arc::new({
            let path = "rocks_db_m4";
            let mut opts = Options::default();
            opts.create_if_missing(false);
            opts.set_merge_operator_associative("append_merge", append_merge);
            opts.increase_parallelism(160);
            opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(16));

            let cache = Cache::new_lru_cache(4 * 1024 * 1024 * 1024);
            let mut block_opts = BlockBasedOptions::default();
            block_opts.set_block_cache(&cache);
            block_opts.set_block_size(16 * 1024);
            block_opts.set_bloom_filter(10.0, false);
            block_opts.set_cache_index_and_filter_blocks(true);
            block_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);

            opts.set_block_based_table_factory(&block_opts);
            opts.set_disable_auto_compactions(true);

            DB::open_for_read_only(&opts, path, false).expect("open failed")
        });

        let m = 4;
        let n = 3 * m;

        // ── Step 1: read all circuits from db, dedup by reversal pair ────────────
        let mut seen_pairs: HashSet<u128> = HashSet::new();

        #[derive(Clone, Debug)]
        struct Entry {
            gates:       Vec<[u8; 3]>,
            forward_key: String,
            reversed_key: String,
        }

        let make_key = |gates: &[[u8; 3]]| -> String {
            let circuit = CircuitSeq { gates: gates.to_vec() };
            let polys = circuit.to_polynomial(n, 0, m);
            let (canonical, _) = canonicalize_polys(polys, true, false);
            polys_repr_blob(&canonical)
                .iter()
                .map(|b| format!("{:02x}", b))
                .collect()
        };

        let mut entries: Vec<Entry> = Vec::new();

        let iter = db.iterator(rocksdb::IteratorMode::Start);
        for item in iter {
            let (_key, value) = item.expect("RocksDB iter error");

            let mut pos = 0;
            while pos < value.len() {
                let len = value[pos] as usize;
                pos += 1;
                let circuit_blob = &value[pos..pos + len];
                pos += len;

                let circuit = CircuitSeq::from_blob(circuit_blob);

                // forward canonical hash
                let canon = canonicalize_polys_4(circuit.to_polynomial(n, 0, m));
                let h: u128 = xxh3_128(&polys_repr_blob(&canon.0));

                // reversed canonical hash
                let mut rev = circuit.clone();
                rev.gates.reverse();
                rev.canonicalize();
                let canon_rev = canonicalize_polys_4(rev.to_polynomial(n, 0, m));
                let h_rev: u128 = xxh3_128(&polys_repr_blob(&canon_rev.0));

                let pair_key = h.min(h_rev);
                if !seen_pairs.insert(pair_key) {
                    continue;
                }

                let forward_key  = make_key(&circuit.gates);
                let reversed_key = make_key(&rev.gates);
                println!("Entries: {}", entries.len());
                entries.push(Entry {
                    gates: circuit.gates.clone(),
                    forward_key,
                    reversed_key,
                });
            }
        }

        println!("Total circuits (up to reversal): {}", entries.len());

        println!("\n=== Duplicates within collected circuits (fwd or rev match) ===");
        let mut any_dup = false;
        for i in 0..entries.len() {
            for j in (i + 1)..entries.len() {
                let ei = &entries[i];
                let ej = &entries[j];

                let mut descs: Vec<&str> = Vec::new();
                if ei.forward_key  == ej.forward_key  { descs.push("fwd==fwd"); }
                if ei.forward_key  == ej.reversed_key { descs.push("fwd==rev"); }
                if ei.reversed_key == ej.forward_key  { descs.push("rev==fwd"); }
                if ei.reversed_key == ej.reversed_key { descs.push("rev==rev"); }

                if !descs.is_empty() {
                    any_dup = true;
                    println!(
                        "  [{i}] {:?}  <->  [{j}] {:?}  [{}]",
                        ei.gates, ej.gates,
                        descs.join(", ")
                    );
                }
            }
        }

        // count unique up to canonicalization and reversal
        let mut seen_canon: HashSet<String> = HashSet::new();
        for e in &entries {
            let canon_pair_key = if e.forward_key <= e.reversed_key {
                format!("{}|{}", e.forward_key, e.reversed_key)
            } else {
                format!("{}|{}", e.reversed_key, e.forward_key)
            };
            seen_canon.insert(canon_pair_key);
        }

        println!("\nTotal circuits up to canonicalization AND reversal: {}", seen_canon.len());
        if !any_dup {
            println!("  (none — all circuits are distinct up to canonicalization and reversal)");
        }
    }

    #[test]
    fn test_compare_circuit_lists() {
        use crate::circuit::{circuit::poly_to_str, CircuitSeq};

        let left_circuits: Vec<Vec<[u8; 3]>> = vec![
            vec![[3,1,2],[1,3,0]],
            vec![[3,0,2],[3,1,0]],
            vec![[3,1,0],[4,0,2]],
            vec![[1,0,3],[2,1,0]],
            vec![[2,0,1],[3,1,0]],
            vec![[3,0,2],[2,1,0]],
            vec![[3,0,1],[3,0,2]],
            vec![[4,0,2],[4,1,3]],
            vec![[2,0,1],[3,1,2]],
            vec![[2,0,1],[3,2,1]],
            vec![[2,0,1],[0,1,2]],
            vec![[3,1,2],[2,0,3]],
            vec![[3,0,1],[4,0,2]],
            vec![[3,0,2],[0,1,3]],
            vec![[2,0,1],[3,0,1]],
            vec![[4,0,1],[1,2,3]],
            vec![[3,0,2],[4,1,2]],
            vec![[2,0,1],[3,0,2]],
            vec![[2,0,1],[1,0,2]],
            vec![[1,2,0],[2,1,0]],
            vec![[4,0,2],[5,1,3]],
            vec![[4,0,2],[0,1,3]],
            vec![[3,0,2],[3,1,2]],
            vec![[2,0,1],[2,1,0]],
        ];

        let right_strings: Vec<&str> = vec![
            "012;013;",
            "021;143;",
            "021;123;",
            "012;143;",
            "042;123;",
            "103;012;",
            "012;123;",
            "023;124;",
            "031;042;",
            "012;021;",
            "042;153;",
            "012;132;",
            "032;142;",
            "032;123;",
            "021;102;",
            "102;012;",
            "120;021;",
            "021;031;",
            "012;031;",
            "130;021;",
            "032;132;",
            "021;132;",
            "130;012;",
        ];

        fn parse_gates(s: &str) -> Vec<[u8; 3]> {
            s.split(';')
                .filter(|p| !p.is_empty())
                .map(|g| {
                    let bytes: Vec<u8> = g.bytes().map(|b| b - b'0').collect();
                    [bytes[0], bytes[1], bytes[2]]
                })
                .collect()
        }

        // Returns (forward_key, reversed_key)
        fn circuit_canon_keys(gates: &[[u8; 3]], n: usize) -> (String, String) {
            let make_key = |g: &[[u8; 3]]| -> String {
                let circuit = CircuitSeq { gates: g.to_vec() };
                let polys = circuit.to_polynomial(n, 0, g.len());
                let (canonical, _) = canonicalize_polys(polys, true, false);
                canonical.iter().enumerate()
                    .map(|(i, p)| format!("P{}:{}", i, poly_to_str(p, n)))
                    .collect::<Vec<_>>()
                    .join("|")
            };

            let forward = make_key(gates);
            let mut rev_gates = gates.to_vec();
            rev_gates.reverse();
            let reversed = make_key(&rev_gates);
            (forward, reversed)
        }

        // Use a fixed n large enough for all circuits
        let n = 6;

        // For each circuit, store both forward and reversed canonical keys,
        // along with a label indicating which variant matched
        #[derive(Debug)]
        struct Entry {
            label: String,
            forward_key: String,
            reversed_key: String,
        }

        let mut left_entries: Vec<Entry> = left_circuits.iter().enumerate().map(|(i, gates)| {
            let (fk, rk) = circuit_canon_keys(gates, n);
            Entry { label: format!("L{:02}", i), forward_key: fk, reversed_key: rk }
        }).collect();

        let mut right_entries: Vec<Entry> = right_strings.iter().enumerate().map(|(i, s)| {
            let gates = parse_gates(s);
            let (fk, rk) = circuit_canon_keys(&gates, n);
            Entry { label: format!("R{:02}", i), forward_key: fk, reversed_key: rk }
        }).collect();

        // Print all keys
        println!("=== LEFT SIDE ===");
        for e in &left_entries {
            println!("{}: fwd={}", e.label, e.forward_key);
            println!("{}  rev={}", e.label, e.reversed_key);
        }
        println!("\n=== RIGHT SIDE ===");
        for e in &right_entries {
            println!("{}: fwd={}", e.label, e.forward_key);
            println!("{}  rev={}", e.label, e.reversed_key);
        }

        // Build lookup: for each right entry, the set of keys it matches
        // (either forward or reversed counts as a match)
        // A left entry matches a right entry if any of the 4 combinations match:
        //   (L.fwd == R.fwd), (L.fwd == R.rev), (L.rev == R.fwd), (L.rev == R.rev)

        println!("\n=== MATCHES (left <-> right, any forward/reverse combination) ===");
        let mut matched_left:  std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut matched_right: std::collections::HashSet<usize> = std::collections::HashSet::new();

        for (li, le) in left_entries.iter().enumerate() {
            for (ri, re) in right_entries.iter().enumerate() {
                let l_keys = [&le.forward_key, &le.reversed_key];
                let r_keys = [&re.forward_key, &re.reversed_key];
                let mut match_desc: Vec<&str> = Vec::new();
                if le.forward_key  == re.forward_key  { match_desc.push("fwd==fwd"); }
                if le.forward_key  == re.reversed_key { match_desc.push("fwd==rev"); }
                if le.reversed_key == re.forward_key  { match_desc.push("rev==fwd"); }
                if le.reversed_key == re.reversed_key { match_desc.push("rev==rev"); }
                if !match_desc.is_empty() {
                    println!("  {} <-> {} [{}]", le.label, re.label, match_desc.join(", "));
                    matched_left.insert(li);
                    matched_right.insert(ri);
                }
            }
        }

        println!("\n=== LEFT entries with NO match on right (fwd or rev) ===");
        for (li, le) in left_entries.iter().enumerate() {
            if !matched_left.contains(&li) {
                println!("  {}: fwd={}", le.label, le.forward_key);
                println!("  {}  rev={}", le.label, le.reversed_key);
            }
        }

        println!("\n=== RIGHT entries with NO match on left (fwd or rev) ===");
        for (ri, re) in right_entries.iter().enumerate() {
            if !matched_right.contains(&ri) {
                println!("  {}: fwd={}", re.label, re.forward_key);
                println!("  {}  rev={}", re.label, re.reversed_key);
            }
        }

        // Duplicates within left (same canonical key, forward or reversed)
        println!("\n=== Duplicates within LEFT ===");
        for i in 0..left_entries.len() {
            for j in (i+1)..left_entries.len() {
                let li = &left_entries[i];
                let lj = &left_entries[j];
                let mut descs = Vec::new();
                if li.forward_key  == lj.forward_key  { descs.push("fwd==fwd"); }
                if li.forward_key  == lj.reversed_key { descs.push("fwd==rev"); }
                if li.reversed_key == lj.forward_key  { descs.push("rev==fwd"); }
                if li.reversed_key == lj.reversed_key { descs.push("rev==rev"); }
                if !descs.is_empty() {
                    println!("  {} <-> {} [{}]", li.label, lj.label, descs.join(", "));
                    println!("  {:?} <-> {:?}", left_circuits[i], left_circuits[j]);
                }
            }
        }

        // Duplicates within right
        println!("\n=== Duplicates within RIGHT ===");
        for i in 0..right_entries.len() {
            for j in (i+1)..right_entries.len() {
                let ri = &right_entries[i];
                let rj = &right_entries[j];
                let mut descs = Vec::new();
                if ri.forward_key  == rj.forward_key  { descs.push("fwd==fwd"); }
                if ri.forward_key  == rj.reversed_key { descs.push("fwd==rev"); }
                if ri.reversed_key == rj.forward_key  { descs.push("rev==fwd"); }
                if ri.reversed_key == rj.reversed_key { descs.push("rev==rev"); }
                if !descs.is_empty() {
                    println!("  {} <-> {} [{}]", ri.label, rj.label, descs.join(", "));
                }
            }
        }
    }

    #[test]
    fn test_group_relabelings_with_reversal() {
        fn cs(gates: &[[u8; 3]]) -> CircuitSeq {
            CircuitSeq { gates: gates.to_vec() }
        }

        fn is_equiv(a: &CircuitSeq, b: &CircuitSeq) -> bool {
            let canon_a = canonicalize_polys(a.to_polynomial(9, 0, 3), true, false);
            let canon_b = canonicalize_polys(b.to_polynomial(9, 0, 3), true, false);

            if canon_a.0 == canon_b.0 {
                return true;
            }

            let mut rev = b.clone();
            rev.gates.reverse();
            let canon_rev = canonicalize_polys(rev.to_polynomial(9, 0, 3), true, false);
            canon_a.0 == canon_rev.0
        }

        let circuits = vec![
            cs(&[[3,2,1],[1,0,3]]),
            cs(&[[4,3,0],[5,2,1]]),
            cs(&[[3,0,1],[3,2,0]]),
            cs(&[[3,0,1],[4,2,0]]),

            cs(&[[1,3,0],[2,0,1]]),
            cs(&[[2,1,0],[3,0,1]]),
            cs(&[[3,1,0],[3,2,0]]),
            cs(&[[2,0,1],[3,0,2]]),
            cs(&[[3,0,1],[3,0,2]]),

            cs(&[[4,3,0],[3,2,1]]),
            cs(&[[4,3,0],[0,2,1]]),

            cs(&[[3,2,0],[2,1,3]]),
            cs(&[[2,1,0],[3,2,1]]),
            cs(&[[4,2,1],[4,3,0]]),
            cs(&[[2,1,0],[0,2,1]]),
            cs(&[[3,2,1],[2,3,0]]),
            cs(&[[3,1,0],[4,2,0]]),
            cs(&[[2,1,0],[3,1,0]]),
            cs(&[[3,0,1],[4,0,2]]),

            cs(&[[4,2,1],[2,3,0]]),

            cs(&[[2,1,0],[3,2,0]]),
            cs(&[[2,1,0],[1,2,0]]),
            cs(&[[1,0,2],[2,0,1]]),
            cs(&[[2,0,1],[2,1,0]]),
        ];

        let mut classes: Vec<Vec<CircuitSeq>> = Vec::new();

        'outer: for c in &circuits {
            for class in &mut classes {
                if is_equiv(c, &class[0]) {
                    class.push(c.clone());
                    continue 'outer;
                }
            }
            classes.push(vec![c.clone()]);
        }

        println!("num equivalence classes: {}", classes.len());

        for (i, class) in classes.iter().enumerate() {
            println!("class {} (size {}):", i, class.len());
            for c in class {
                println!("  {:?}", c.gates);
            }
        }

        println!("\n--- duplicates (relabeling + reversal) ---");
        for class in &classes {
            if class.len() > 1 {
                println!("class (size {}):", class.len());
                for c in class {
                    println!("  {:?}", c.gates);
                }
            }
        }

        let total: usize = classes.iter().map(|c| c.len()).sum();
        assert_eq!(total, circuits.len());
    }

    #[test]
    fn test_c1_vs_c2_after_canon() {
        use crate::circuit::circuit::poly_to_str;
        let mut c1 = CircuitSeq { gates: vec![[5, 0, 3], [6, 1, 4], [7, 2, 3]]  };
        let mut c2 = CircuitSeq { gates: vec![[5, 0, 3], [6, 1, 4], [7, 2, 4]]  };
        c1.gates.reverse();
        c2.gates.reverse();
        println!("Evaluation same? {}", c1.probably_equal(&c2, 9, 1000).is_ok());
        let canon1 = canonicalize_polys(c1.to_polynomial(9, 0, 3), true, false);
        let canon2 = canonicalize_polys(c2.to_polynomial(9, 0, 3), true, false);
        println!("canon same? {}", canon1.0 == canon2.0);
        c1.rewire(&canon1.1.invert(), 9);
        c2.rewire(&canon2.1.invert(), 9);
        c1.canonicalize();
        c2.canonicalize();
        println!("After rewiring:");
        println!("c1: {:?}", c1.gates);
        println!("c2: {:?}", c2.gates);
        println!("{:?}", canonicalize_polys(c1.to_polynomial(9, 0, 3), true, false).0);
        let mut c1 = CircuitSeq { gates: vec![[5, 0, 3], [6, 1, 4], [7, 2, 3]]  };
        let mut c2 = CircuitSeq { gates: vec![[5, 0, 3], [6, 1, 4], [7, 2, 4]]  };
        // c1.gates.reverse();
        // c2.gates.reverse();
        let poly_1 = c1.to_polynomial(9, 0, 3);
        let poly_2 = c2.to_polynomial(9, 0, 3);
        println!("Original c1:");
        for (i, poly) in poly_1.iter().enumerate() {
            println!("  P{}: {}", i, poly_to_str(poly, 9));
        }
        println!("Original c2:");
        for (i, poly) in poly_2.iter().enumerate() {
            println!("  P{}: {}", i, poly_to_str(poly, 9));
        }

        let canon_1 = canonicalize_polys_4(c1.to_polynomial(9, 0, 3));
        let canon_2 = canonicalize_polys_4(c2.to_polynomial(9, 0, 3));
        println!("Canonical c1: ");
        for (i, poly) in canon_1.0.iter().enumerate() {
            println!("  P{}: {}", i, poly_to_str(poly, 9));
        }
        println!("Wire ranking for c1");
        println!("{:?}", canon_1.1.data);
        println!("Canonical c2: ");
        for (i, poly) in canon_2.0.iter().enumerate() {
            println!("  P{}: {}", i, poly_to_str(poly, 9));
        }
        println!("Wire ranking for c2");
        println!("{:?}", canon_2.1.data);
        assert!(canon_1.0 == canon_2.0, "Canonical forms differ:\n  c1: {:?}\n  c2: {:?}", canon_1.0, canon_2.0);
        
        c1.rewire(&canon_1.1.invert(), 9);
        c2.rewire(&canon_2.1.invert(), 9);
        c1.canonicalize();
        c2.canonicalize();

        assert!(c1.gates == c2.gates, "Circuits differ after rewiring and canonicalization:\n  c1: {:?}\n  c2: {:?}", c1.gates, c2.gates);
    }

    #[test]
    fn test_reversal_pair_key() {
        let m = 2; // adjust as needed
        let n = 3 * m;

        // The two circuits you observed as duplicates
        let mut c_a = CircuitSeq { gates: vec![[4, 0, 1], [1, 4, 3], [5, 4, 2]]  };
        let mut c_b = CircuitSeq { gates: vec![[3, 0, 2], [5, 0, 4], [0, 1, 3]]  };

        // Hash c_a canonically
        let canon_a = canonicalize_polys_4(c_a.to_polynomial(n, 0, m));
        let hash_a = xxh3_128(&polys_repr_blob(&canon_a.0));

        // Hash c_b canonically
        let canon_b = canonicalize_polys_4(c_b.to_polynomial(n, 0, m));
        let hash_b = xxh3_128(&polys_repr_blob(&canon_b.0));

        // Compute c_a's reversal and hash it
        let mut c_a_rev = c_a.clone();
        c_a_rev.gates.reverse();
        c_a_rev.canonicalize();
        let canon_a_rev = canonicalize_polys_4(c_a_rev.to_polynomial(n, 0, m));
        let hash_a_rev = xxh3_128(&polys_repr_blob(&canon_a_rev.0));

        // Compute c_b's reversal and hash it
        let mut c_b_rev = c_b.clone();
        c_b_rev.gates.reverse();
        c_b_rev.canonicalize();
        let canon_b_rev = canonicalize_polys_4(c_b_rev.to_polynomial(n, 0, m));
        let hash_b_rev = xxh3_128(&polys_repr_blob(&canon_b_rev.0));

        println!("hash_a:     {:032x}", hash_a);
        println!("hash_a_rev: {:032x}", hash_a_rev);
        println!("hash_b:     {:032x}", hash_b);
        println!("hash_b_rev: {:032x}", hash_b_rev);

        let pair_key_a = hash_a.min(hash_a_rev);
        let pair_key_b = hash_b.min(hash_b_rev);

        println!("pair_key_a: {:032x}", pair_key_a);
        println!("pair_key_b: {:032x}", pair_key_b);

        // This is the critical assertion:
        // if c_b IS the reversal of c_a, their pair keys must match
        assert_eq!(
            pair_key_a, pair_key_b,
            "c_a and c_b are reversals but produce different pair_keys — dedup will miss them"
        );

        // Also verify they are actually reversals of each other
        assert_eq!(hash_a, hash_b_rev, "c_a hash should equal c_b_rev hash");
        assert_eq!(hash_b, hash_a_rev, "c_b hash should equal c_a_rev hash");
    }
}
