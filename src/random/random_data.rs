use crate::{
    circuit::{CircuitSeq, Gate, Permutation},
    rainbow::canonical::{self, CandSet, Canonicalization},
};

use crossbeam::channel::{bounded, unbounded, Sender};
use dashmap::DashMap;
use itertools::Itertools;
use once_cell::sync::Lazy;
use rand::{
    prelude::IndexedRandom,
    rngs::StdRng,
    Rng, RngCore, SeedableRng,
};
use rayon::{
    iter::{IntoParallelRefIterator, ParallelIterator},
    prelude::*,
    slice::ParallelSlice,
};
use rusqlite::{params, Connection, Result};
use smallvec::SmallVec;
use std::{
    collections::HashSet,
    fs::OpenOptions,
    io::Write,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
};
use lru::LruCache;

pub static CANON_CACHE: Lazy<DashMap<Vec<u8>, (Vec<u8>, Vec<u8>)>> = Lazy::new(|| DashMap::new());

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

pub fn random_circuit(n: u8, m: usize) -> CircuitSeq {
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

use std::collections::HashMap;
use rayon::prelude::*;

pub fn random_equivalent_circuits_until_found(n: u8) -> (CircuitSeq, CircuitSeq) {
    // final_state → list of circuits producing that state
    let mut state_map: HashMap<u64, Vec<CircuitSeq>> = HashMap::new();
    let mut total_generated = 0usize;

    loop {
        let m = fastrand::usize(10..=30);
        let circuit = random_circuit(n, m);
        total_generated += 1;

        if total_generated % 10_000 == 0 {
            println!("Generated {} circuits so far...", total_generated);
        }

        // Compute final state starting from 0
        let mut state = Gate::evaluate_index_list(0, &circuit.gates);

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

// Given a circuit of num_wires, we try to find a convex subcircuit of up to max_wires. We can start in any of hte min_candidates
pub fn find_convex_subcircuit<R: RngCore>(
    set_size: usize,
    max_wires: usize,
    num_wires: usize,
    circuit: &CircuitSeq,
    rng: &mut R,
) -> (Vec<usize>, usize) {
    let num_gates = circuit.gates.len();
    let mut search_attempts = 0;
    let max_attempts = 10_000;

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
        let mut selected_gate_idx = vec![0; set_size];
        selected_gate_idx[0] = rng.random_range(0..num_gates);
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
                for curr_idx in selected_gate_idx[0] + 1..num_gates {
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
                for curr_idx in (0..=selected_gate_idx[selected_gate_ctr - 1] - 1).rev() {
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

            // check if adding this gate would exceed max_wires ---
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

        if selected_gate_ctr != set_size {
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
                    non_convex[i] += 1;
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
                    non_convex[i] -= 1;
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

/// Shoots a random gate left or right as far as possible without colliding
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

            if target != gate_idx {
                let gate = circuit.gates.remove(gate_idx);
                circuit.gates.insert(target, gate);
            }
        }
    }
}

pub fn create_table(conn: &mut Connection, table_name: &str) -> Result<()> {
    // Table name includes n and m
    let sql = format!(
        "CREATE TABLE IF NOT EXISTS {table} (
            circuit BLOB UNIQUE,
            perm BLOB NOT NULL,
            shuf BLOB NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_perm_{table} ON {table} (perm);",
        table = table_name
    );

    conn.execute_batch(&sql)?;
    Ok(())
}

pub fn insert_circuit(
    conn: &mut Connection,
    circuit: &CircuitSeq, 
    canon: &Canonicalization,
    table_name: &str,
) -> Result<()> {
    let key = circuit.repr_blob();
    let perm = canon.perm.repr_blob();
    let shuf = canon.shuffle.repr_blob();
    let sql = format!("INSERT OR IGNORE INTO {} (circuit, perm, shuf) VALUES (?1, ?2, ?3)", table_name);
    conn.execute(&sql, &[&key, &perm, &shuf])?;
    Ok(())
}

pub fn insert_circuits_batch(
    conn: &mut Connection,
    table_name: &str,
    circuits: &[(CircuitSeq, Canonicalization)],
) -> Result<usize> {
    // Start a single transaction for all inserts
    let tx = conn.transaction()?;

    // Prepare the SQL once
    let sql = format!(
        "INSERT OR IGNORE INTO {} (circuit, perm, shuf) VALUES (?1, ?2, ?3)",
        table_name
    );

    let mut inserted = 0;

    for (circuit, canon) in circuits {
        let key = circuit.repr_blob();
        let perm = canon.perm.repr_blob();
        let shuf = canon.shuffle.repr_blob();

        if tx.execute(&sql, &[&key, &perm, &shuf])? > 0 {
            inserted += 1;
        }
    }

    // Commit all inserts at once
    tx.commit()?;

    Ok(inserted)
}

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
        let num_b = std::mem::size_of::<usize>() * 8 - (n - 1).leading_zeros() as usize;

        let mut min_perm: SmallVec<[usize; 64]> = SmallVec::from_slice(&self.data);
        let mut bits: SmallVec<[usize; 64]> = SmallVec::from_elem(0, n);
        let mut index_shuf: SmallVec<[usize; 64]> = SmallVec::from_elem(0, n);
        let mut perm_shuf: SmallVec<[usize; 64]> = SmallVec::from_elem(0, n);

        let mut best_shuffle = Permutation::id_perm(num_b);

        for r in bit_shuf.iter() {
            for (src, &dst) in r.iter().enumerate() {
                for (i, &val) in self.data.iter().enumerate() {
                    bits[i] |= ((val >> src) & 1) << dst;
                    index_shuf[i] |= ((i >> src) & 1) << dst;
                }
            }

            for (i, &val) in bits.iter().enumerate() {
                perm_shuf[index_shuf[i]] = val;
            }

            for weight in 0..=num_b / 2 {
                let mut done = false;
                for i in canonical::index_set(weight, num_b) {
                    if perm_shuf[i] == min_perm[i] {
                        continue;
                    }
                    if perm_shuf[i] < min_perm[i] {
                        min_perm.copy_from_slice(&perm_shuf);
                        best_shuffle.data.copy_from_slice(&r);
                    }
                    done = true;
                    break;
                }
                if done {
                    break;
                }
            }

            bits.fill(0);
            index_shuf.fill(0);
        }

        Canonicalization {
            perm: Permutation { data: min_perm.into_vec() },
            shuffle: best_shuffle,
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

pub fn check_cycles(n: usize, m: usize) -> Result<()> {
    // Open the database
    let conn = Connection::open("circuits.db")?;
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

pub fn print_all(table_name: &str) -> Result<()> {
    let conn = Connection::open("circuits.db")?;

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

pub fn count_distinct(n: usize, m: usize) -> Result<usize> {
    let conn = Connection::open("circuits.db")?;
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

pub fn build_from_sql(
    conn: &mut Connection,
    n: usize,
    m: usize,
    bit_shuf: &Vec<Vec<usize>>,
) -> Result<()> {
    println!("Running build (max CPU)");

    let old_table = format!("n{}m{}", n, m - 1);
    let new_table = format!("n{}m{}", n, m);

    create_table(conn, &new_table)?;

    let base_gates: Arc<Vec<[u8; 3]>> = Arc::new(base_gates(n));
    let base_gates_for_thread = Arc::clone(&base_gates);
    let bit_shuf = Arc::new(bit_shuf.clone());

    let total_rows: i64 = conn.query_row(
        &format!("SELECT MAX(rowid) FROM {}", old_table),
        [],
        |row| row.get(0),
    )?;
    println!("Total rows in {}: {}", old_table, total_rows);

    let chunk_size: i64 = 50_000;
    let batch_size = 10_000;

    let mut last_rowid: i64 = 0;

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
        let mut insert_conn =
            Connection::open("./db/circuits.db").expect("Failed to open DB in insert thread");

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
    while last_rowid < total_rows {
        if stop_flag.load(Ordering::SeqCst) {
            println!("Stopping early due to CTRL+C...");
            break;
        }

        let rows: Vec<(i64, Vec<u8>)> = {
            let mut stmt = conn.prepare(&format!(
                "SELECT rowid, circuit FROM {} WHERE rowid > ? ORDER BY rowid LIMIT ?",
                old_table
            ))?;
            stmt.query_map(params![last_rowid, chunk_size], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })?
            .collect::<Result<_, _>>()?
        };

        if rows.is_empty() {
            break;
        }

        last_rowid = rows.last().unwrap().0;

        // Process circuits in parallel and stream batches immediately
        rows.par_chunks(500).for_each(|row_chunk| {
            let mut local_results =
                Vec::with_capacity(row_chunk.len() * base_gates.len() * 2);

            for (_rowid, blob) in row_chunk {
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

                    local_results.push((c1, canon1));
                    local_results.push((c2, canon2));
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
            "Processed up to rowid {}. Progress: {:.2}%",
            last_rowid,
            (last_rowid as f64 / total_rows as f64) * 100.0
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

//TODO: benchmark to see which part is taking the most time and what exactly can be sped up
//Speed up SQL queries
//Should not see for a particular size query, the speed should not vary across multiple runs
pub fn main_random(n: usize, m: usize, count: usize, stop: bool) {
    let mut conn = Connection::open("./db/circuits.db").expect("Failed to open DB");
    let table_name = format!("n{}m{}", n, m);
    create_table(&mut conn, &table_name).expect("Failed to create table");

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

    //TODO: test speed here
    while running.load(Ordering::SeqCst) && (!stop && inserted < count || stop) {
        let start = std::time::Instant::now(); // start timing this iteration
        total_attempts += 1;

        let mut circuit = random_circuit(n as u8, m);
        circuit.canonicalize();

        let perm = circuit.permutation(n).canon_simple(&bit_shuf);
        batch.push((circuit, perm));

        if batch.len() >= batch_size {
            //let start = std::time::Instant::now();
            let success_count =
                insert_circuits_batch(&mut conn, &table_name, &batch).unwrap_or(0);
            //let elapsed = start.elapsed();

            // Log timing to file
            // let mut file = OpenOptions::new()
            //     .create(true)
            //     .append(true)
            //     .open("time5000.txt")
            //     .expect("Failed to open time.txt");
            // writeln!(
            //     file,
            //     "Batch of {} attempted, {} inserted, time: {:?}",
            //     batch_size, success_count, elapsed
            // ).expect("Failed to write to time.txt");

            inserted += success_count;
            recent += success_count;
            batch.clear();

            // Early stop if >=99% of last batch failed
            if success_count * 100 <= batch_size {
                // writeln!(
                //     file,
                //     "Stopping early: only {}/{} inserts succeeded (~{:.2}% success)",
                //     success_count,
                //     batch_size,
                //     (success_count as f64 / batch_size as f64) * 100.0
                // ).expect("Failed to write to time.txt");

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



#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;
    use std::fs::OpenOptions;
    use crate::replace::replace::compress_big;
    #[test]
    fn test_check_cycles_n3m3() -> Result<()> {
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
        let c = random_circuit(16,30);

        let mut rng = rand::rng();
        let max_wires = 7;

        let mut subcircuit_gates = vec![];
        let mut attempts = 0;

        for set_size in (3..=16).rev() {
            let (gates, tries) = find_convex_subcircuit(set_size, max_wires, 16, &c, &mut rng);
            attempts += tries;

            if !gates.is_empty() {
                subcircuit_gates = gates;
                println!("Found convex subcircuit with {set_size} gates after {attempts} total attempts");
                break;
            }
        }

        if subcircuit_gates.is_empty() {
            eprintln!("No convex subcircuit found for any size in 3..=16");
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

        // Check that the selected subcircuit is actually convex
        let convex_ok = is_convex(16, &c, &subcircuit_gates);
        let mut gates: Vec<[u8;3]> = vec![[0,0,0];subcircuit_gates.len()];
        for (i, g) in subcircuit_gates.iter().enumerate() {
            gates[i] = c.gates[*g];
        }
        let subcircuit = CircuitSeq { gates };
        subcircuit_gates.sort();
        println!("{}", subcircuit.to_string(16));
        println!("{:?}", subcircuit.used_wires());
        let sub = CircuitSeq::rewire_subcircuit(&c, &subcircuit_gates, &subcircuit.used_wires());
        let undo =  CircuitSeq::unrewire_subcircuit(&sub, &subcircuit.used_wires());
        println!("Rewire and unrewire is ok: {}", subcircuit.permutation(wire_set.len()) == undo.permutation(wire_set.len()));
        assert!(convex_ok, "Selected subcircuit is not convex");
        println!("Convexity check passed");
        let mut circ = c.clone();
        println!("Before gates {:?}", subcircuit_gates);
        let (start, end) = contiguous_convex(&mut circ,  &mut subcircuit_gates, 16).unwrap();
        println!("After gates {:?}", subcircuit_gates);
        println!("The rearranged are equal: {}", c.permutation(16).data == circ.permutation(16).data);
        println!("New circuit {:?}", circ.gates);
        println!("start and end designated: {:?}", &circ.gates[start..end+1]);
        println!("Desired sub: {:?}", subcircuit.gates);
    }
    use crate::replace::replace::compress;
    #[test]
    fn test_compression_speed() {
        // Hard-coded random circuit
        let c = random_circuit(7,30);

        let mut conn = Connection::open("./circuits.db").expect("Failed to open DB");

        // Run the profiling version of compress_big
        let perms: Vec<Vec<usize>> = (0..7).permutations(7).collect();
        let bit_shuf = perms.into_iter().skip(1).collect::<Vec<_>>();
        let start = std::time::Instant::now();
        let _result = compress(&c, 100000, &mut conn, &bit_shuf, 7);
        let total_time = start.elapsed();

        println!("Total compress_big runtime: {:?} ms", total_time);
    }

    #[test]
    fn test_compression_big() {
        // Dummy 16-wire circuit with 30 gates
        let c = random_circuit(16,30);

        let mut conn = Connection::open("./circuits.db").expect("Failed to open DB");

        let com = compress_big(&c, 10, 16, &mut conn);
        println!("compression is okay: {}", com.permutation(16) == c.permutation(16));
    }

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
            old.permutation(16) == new.permutation(16)
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
        let c = CircuitSeq::from_string("123;");
        let id = CircuitSeq::from_string("123;123;");
        c
            .probably_equal(&id, 32, 150_000)
            .expect("The circuits differ somewhere!");
        println!("Time to compute permutation on 32 wires: {:?}", t.elapsed());
    }

    use super::*;
    use std::io::{self, BufRead};
    use std::path::Path;


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

    use super::*;
    use std::fs::File;
    use std::io::Write;

    #[test]
    fn generate_random_equivalent_circuits() {
        let n: u8 = 32;

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
    use crate::replace::replace::random_id;
    use rusqlite::OpenFlags;
    #[test]
    fn test_shooting() {
        let mut rng = rand::rng();

        // Start with an initial random identity
        let (c, c_rev) = random_id(64, 20);
        let mut id = c.concat(&c_rev);

        // Insert X random identities at random positions
        let x = 10;
        for _ in 0..x {
            // generate another random identity
            let (new_c, new_c_rev) = random_id(64, 20);
            let new_id = new_c.concat(&new_c_rev);

            // choose a random insertion point
            let insert_pos = rng.random_range(0..=id.gates.len());

            // insert its gates at that position
            id.gates.splice(insert_pos..insert_pos, new_id.gates.clone());
        }

        // Save the initial state for debugging
        let c_str = id.repr();
        File::create("test_start.txt")
            .and_then(|mut f| f.write_all(c_str.as_bytes()))
            .expect("Failed to write test_start.txt");

        // Proceed as before
        shoot_random_gate(&mut id, 100000);
        let mut conn = Connection::open_with_flags(
            "./db/circuits.db",
            OpenFlags::SQLITE_OPEN_READ_ONLY,
        ).expect("Failed to open DB (read-only)");

        compress_big(&mut id, 100_000, 64, &mut conn);
        assert!(id.probably_equal(&CircuitSeq::from_string("123;123;"), 64, 100000).is_ok());
        println!("Len: {}", id.gates.len());

        let c_str = id.repr();
        File::create("test_compression.txt")
            .and_then(|mut f| f.write_all(c_str.as_bytes()))
            .expect("Failed to write test_compression.txt");
    }
}