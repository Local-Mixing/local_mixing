use crate::circuit::circuit::{CircuitSeq};
use crate::replace::replace::{obfuscate, compress};
use std::fs::File;
use std::io::Write;
use rusqlite::Connection;
use itertools::Itertools;
use std::fs::OpenOptions;

fn obfuscate_and_target_compress(c: &CircuitSeq, conn: &Connection, bit_shuf: &Vec<Vec<usize>>, n: usize) -> CircuitSeq {
    // Obfuscate circuit, get positions of inverses
    let (mut _final_circuit, _inverse_starts) = obfuscate(c, n);
    let (mut final_circuit, inverse_starts) = obfuscate(&_final_circuit, n);
    println!("{:?} Obf Len: {}", pin_counts(&final_circuit, n), final_circuit.gates.len());
    // For each gate, compress its "inverse+gate+next_random" slice
    // Reverse iteration to avoid index shifting issues
    
    for i in (0..c.gates.len()).rev() {
        // ri^-1 start
        let start = inverse_starts[i];

        // r_{i+1} start is the next inverse start
        let end = inverse_starts[i + 1]; // safe because i < c.gates.len()
        // Slice the subcircuit: r_i^-1 ⋅ g_i ⋅ r_{i+1}
        let sub_slice = &final_circuit.gates[start..end];

        // Wrap it into a CircuitSeq
        let sub_circuit = CircuitSeq { gates: sub_slice.to_vec() };

        // Compress the subcircuit
        let compressed_sub = compress(&sub_circuit, 100_000, conn, &bit_shuf, n);
        // Replace the slice in the final circuit
        if sub_circuit.gates != compressed_sub.gates {
            println!("The compression hid g_{}", i);
        }
        final_circuit.gates.splice(start..end, compressed_sub.gates);
    }
    let mut com_len = final_circuit.gates.len();
    let mut count = 0;
    while count < 3 {
        final_circuit = compress(&final_circuit, 100_000, conn, &bit_shuf, n);
        if final_circuit.gates.len() == com_len {
            count += 1;
        } else {
            com_len = final_circuit.gates.len();
        }
    }
    println!("{:?} Compressed Len: {}", pin_counts(&final_circuit, n), final_circuit.gates.len());
    final_circuit
}

pub fn pin_counts(circuit: &CircuitSeq, num_wires: usize) -> Vec<usize> {
    let mut counts = vec![0; num_wires];
    for gate in &circuit.gates {
        counts[gate[0] as usize] += 1;
        counts[gate[1] as usize] += 1;
        counts[gate[2] as usize] += 1;
    }
    counts
}

pub fn main_mix(c: &CircuitSeq, rounds: usize, conn: &Connection, n: usize) {
    // Start with the input circuit
    let mut circuit = c.clone();
    let perms: Vec<Vec<usize>> = (0..n).permutations(n).collect();
    let bit_shuf = perms.into_iter().skip(1).collect::<Vec<_>>();
    // Repeat obfuscate + compress 'rounds' times
    let mut post_len = 0;
    let mut count = 0;
    for _ in 0..rounds {
        circuit = obfuscate_and_target_compress(&circuit, conn, &bit_shuf, n);
        if circuit.gates.len() == 0 {
            break;
        }
        
        if circuit.gates.len() == post_len {
            count += 1;
        } else {
            post_len = circuit.gates.len();
            count = 0;
        }

        if count > 2 {
            break;
        }
    }

    // Convert the final circuit to string
    let circuit_str = circuit.to_string(n);
    println!("{:?}", circuit.permutation(n).data);
    // Write to file
    let mut file = File::create("recent_circuit.txt").expect("Failed to create file");
    file.write_all(circuit_str.as_bytes())
        .expect("Failed to write circuit to file");

    let circuit_str = circuit.repr(); // or however you stringify your circuit

    if !circuit.gates.is_empty() {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .append(true) // append instead of overwriting
            .open("good_id.txt")
            .expect("Failed to open good_id.txt");

        let line = format!("{} : {}\n", circuit.gates.len(), circuit_str);

        file.write_all(line.as_bytes())
            .expect("Failed to write circuit to good_ids.txt");

        println!("Wrote good circuit to good_id.txt");
    }

    if circuit.gates == c.gates {
        println!("The obfuscation didn't do anything");
    }

    println!("Final circuit written to recent_circuit.txt");
}