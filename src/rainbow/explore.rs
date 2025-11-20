use crate::circuit::circuit;
use crate::circuit::CircuitSeq;
use std::collections::HashMap;
use std::cmp::min as std_min;
use std::cmp::max as std_max;
use crate::rainbow::database::Persist;

use crate::circuit::Permutation;
use std::fs::File;
use std::io::BufReader;

pub fn gcd(a: isize, b:isize) -> usize {
    let mut a = a.abs();
    let mut b = b.abs();
    while b != 0 {
        let temp = b;
        b = a%b;
        a = temp;
    }
    a as usize
}

pub fn lcm(a: isize, b:isize) -> usize {
    if a == 0 || b == 0 {
        return 0
    }
    ((a.abs()*b.abs()) as usize)/gcd(a,b)
}

pub fn lcm_list(list: &Vec<isize>) -> usize {
    if list.is_empty() {
        return 1
    }
    list[1..].iter().fold(list[0].abs() as usize, |acc, &x| lcm(acc as isize,x))
}

//compute length of all cycles in the permutation
pub fn cycle_len(cycle: &Vec<Vec<usize>>) -> Vec<usize> {
    let mut lens: Vec<usize> = cycle.iter().map(|r| r.len()).collect();
    lens.sort_unstable();
    lens
}

//compute the order of the permutation. This is the lcm of all the cycles
pub fn order(cycle: &Vec<Vec<usize>>) -> usize {
    let len_list: Vec<isize> = cycle_len(cycle).iter().map(|x| *x as isize).collect();
    lcm_list(&len_list)
}

pub fn subcircuits(circuit: &CircuitSeq, sub_size: usize) -> Vec<CircuitSeq> {
    let mut list = Vec::<CircuitSeq>::new();
    if sub_size == 0 || sub_size > circuit.gates.len() {
        panic!("Can't find subcircuit larger than the original circuit");
    }
    for i in 0..circuit.gates.len() - sub_size{
        let new_circ = CircuitSeq{ gates: circuit.gates[i..i+sub_size].to_vec() };
        list.push(new_circ);
    }
    list
}

//find lexicographically smallest way to write a vec
pub fn min_rot(x: &Vec<usize>) -> Vec<usize> {
    let mut min = x.clone();
    let mut rotated = x.clone();

    for _ in 1..x.len() {
        // rotate left by 1
        let first = rotated.remove(0);
        rotated.push(first);

        if rotated < min {
            min.copy_from_slice(&rotated);
        }
    }

    min
}

//find min rotation of hamming distances between each element in each row
pub fn ham_diff(input: &Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    let mut result = Vec::with_capacity(input.len());

    for row in input {
        let mut diffs = vec![0; row.len()];

        if !row.is_empty() {
            // first element: compare last and first
            diffs[0] = hamming_dist(row[0], row[row.len() - 1]);

            // other elements: compare each with previous
            for j in 1..row.len() {
                diffs[j] = hamming_dist(row[j], row[j - 1]);
            }
        }

        result.push(min_rot(&diffs));
    }

    result
}

pub fn hamming_dist(x: usize, y: usize) -> usize {
    (x^y).count_ones() as usize
}

pub fn explore_db(n:usize, m:usize) {
    let filename = format!("./db/n{}m{}.bin", n, m);
    let file = File::open(&filename).expect("Could not open file");
    let reader = BufReader::new(file);
    let persist: Persist = bincode::deserialize_from(reader).expect("Failed to decode file");

    println!("-----------------------");
    println!("Decoding file: {}", filename);
    println!("Info for n={}, m={}", persist.wires, persist.gates);
    println!("Unique perms: {}", persist.store.len());

    let mut max_order = 0;
    let mut min_order = usize::MAX;
    let mut most_stored = 0;
    let mut with_most_popular = 0;
    let mut popular_perm: Vec<Vec<usize>> = Vec::new();

    let mut diff_hamming = HashMap::<String, bool>::new();
    println!("Unique perms: {}", persist.store.len());

    let base_gates = circuit::base_gates(n);
    let len_base = base_gates.len() as u64;            
    let total_ckt = len_base * (len_base - 1).pow((m-1) as u32);

    let mut saw_id = false;
    let mut n_circuits = 0;
    let mut singles = 0;

    for (key, circuits) in &persist.store {
        let cyc = Permutation::from_blob(&key).to_cycle();
        let order = order(&cyc);

        diff_hamming.insert(format!("{:?}", ham_diff(&cyc)), true);

        if cyc.len() == 0 {
            saw_id = true;
            println!("{} id circuit", circuits.len());
            for circ in circuits {
                println!("{}", CircuitSeq::from_blob(&circ).to_string(n));
            }   
        }

        for circ in circuits {
            println!("{}", CircuitSeq::from_blob(&circ).to_string(n));
        }   

        let pop = circuits.len();
        if pop > most_stored {
            popular_perm = cyc;
            with_most_popular = 1;
            most_stored = pop;
        }  else if pop == most_stored {
            with_most_popular += 1;
        }

        if pop == 1 {
            singles += 1;
        }

        max_order = std_max(order, max_order);
        if min_order == usize::MAX {
            min_order = order;
        } else {
            min_order = std_min(order, min_order);
        }

        most_stored = std_max(pop, most_stored);
        n_circuits += pop;
    }

    let compression = total_ckt / n_circuits as u64;

    println!(
        "Circuits: {} out of {} ({} x)",
        n_circuits, total_ckt, compression
    );
    println!("Max order: {}", max_order);
    println!("Min order: {}", min_order);
    println!("Singletons: {}%", 100 * singles / persist.store.len());
    println!(
        "Diff weights: {} {}",
        diff_hamming.len(),
        100 * diff_hamming.len() / persist.store.len()
    );
    println!("Max popularity: {}", most_stored);
    println!("  e.g. {:?}", popular_perm);

    if with_most_popular > 1 {
        println!("  + {} others", with_most_popular - 1);
    }

    println!("Saw Identity? {}", saw_id);
}