use dashmap::DashMap;
use local_mixing::circuit::circuit::{polys_repr_blob, CircuitSeq};
use rayon::prelude::*;
use xxhash_rust::xxh3::xxh3_128;

fn canon_key(circuit: &CircuitSeq) -> [u8; 16] {
    let (polys, _order, _used) = circuit.canonicalize_polys_single(false);
    if polys.is_empty() {
        return [0u8; 16];
    }
    xxh3_128(&polys_repr_blob(&polys)).to_le_bytes()
}

fn process(
    gates: &[[u16; 3]],
    current: &mut Vec<[u16; 3]>,
    target_len: usize,
    not_up_to_rev: &DashMap<[u8; 16], String>,
    up_to_rev: &DashMap<([u8; 16], [u8; 16]), String>,
) {
    if current.len() == target_len {
        let circuit = CircuitSeq { gates: current.clone() };
        let fwd_key = canon_key(&circuit);
        let rev_circuit = CircuitSeq { gates: current.iter().rev().cloned().collect() };
        let rev_key = canon_key(&rev_circuit);
        let repr = circuit.repr();
        not_up_to_rev.entry(fwd_key).or_insert_with(|| repr.clone());
        let pair = if fwd_key <= rev_key { (fwd_key, rev_key) } else { (rev_key, fwd_key) };
        up_to_rev.entry(pair).or_insert(repr);
        return;
    }
    for &g in gates.iter() {
        current.push(g);
        process(gates, current, target_len, not_up_to_rev, up_to_rev);
        current.pop();
    }
}

fn list_n_gate_circuits(gate_count: usize, n_wires: usize) {
    let gates: Vec<[u16; 3]> = (0..n_wires as u16)
        .flat_map(|t| {
            (0..n_wires as u16).flat_map(move |c1| {
                (0..n_wires as u16)
                    .filter(move |&c2| t != c1 && t != c2 && c1 != c2)
                    .map(move |c2| [t, c1, c2])
            })
        })
        .collect();

    println!("=== {}-gate circuits on {} wires ===", gate_count, n_wires);
    println!("Gates: {}", gates.len());

    let not_up_to_rev: DashMap<[u8; 16], String> = DashMap::new();
    let up_to_rev: DashMap<([u8; 16], [u8; 16]), String> = DashMap::new();

    // Parallelize over first gate choice
    (0..gates.len()).into_par_iter().for_each(|i| {
        let mut current = vec![gates[i]];
        process(&gates, &mut current, gate_count, &not_up_to_rev, &up_to_rev);
    });

    let mut not_sorted: Vec<_> = not_up_to_rev.into_iter().collect();
    not_sorted.sort_by_key(|(k, _)| *k);

    println!("\nNot up to reversal: {}", not_sorted.len());
    for (_, repr) in &not_sorted {
        println!("  {}", repr);
    }

    let mut rev_sorted: Vec<_> = up_to_rev.into_iter().collect();
    rev_sorted.sort_by_key(|(k, _)| *k);

    println!("\nUp to reversal: {}", rev_sorted.len());
    for (_, repr) in &rev_sorted {
        println!("  {}", repr);
    }
}

fn main() {
    list_n_gate_circuits(2, 6);
    println!();
    list_n_gate_circuits(3, 9);
}
