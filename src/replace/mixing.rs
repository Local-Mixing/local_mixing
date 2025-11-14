use crate::{
    circuit::circuit::CircuitSeq,
    replace::replace::{compress, compress_big, obfuscate, outward_compress, random_id, expand_big},
};
use crate::random::random_data::shoot_random_gate;
use itertools::Itertools;
use rand::Rng;
use rayon::prelude::*;
use rusqlite::{Connection, OpenFlags};
use std::{
    fs::{File, OpenOptions},
    io::Write,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

fn obfuscate_and_target_compress(c: &CircuitSeq, conn: &mut Connection, bit_shuf: &Vec<Vec<usize>>, n: usize) -> CircuitSeq {
    // Obfuscate circuit, get positions of inverses
    let (mut final_circuit, inverse_starts) = obfuscate(c, n);
    println!("{}", final_circuit.to_string(n));
    //let (mut final_circuit, inverse_starts) = obfuscate(&_final_circuit, n);
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

pub fn butterfly(
    c: &CircuitSeq,
    conn: &mut Connection,
    bit_shuf: &Vec<Vec<usize>>,
    n: usize,
) -> CircuitSeq {
    // Pick one random R
    let mut rng = rand::rng();
    let (r, r_inv) = random_id(n as u8, rng.random_range(3..=25)); 

    println!("Butterfly start: {} gates", c.gates.len());

    let r = &r;           // reference is enough; read-only
    let r_inv = &r_inv;   // same
    let bit_shuf = &bit_shuf;

    // Parallel processing of gates
    let blocks: Vec<_> = c.gates
        .par_iter()
        .enumerate()
        .map(|(i, &g)| {
            // wrap single gate as CircuitSeq
            let gi = CircuitSeq { gates: vec![g] };

            // create a read-only connection per thread
            let mut conn = Connection::open_with_flags(
            "circuits.db",
            OpenFlags::SQLITE_OPEN_READ_ONLY,
        ).expect("Failed to open read-only connection");

        // compress the block
        let compressed_block = outward_compress(&gi, r, 100_000, &mut conn, bit_shuf, n);

        println!(
            "  Block {}: before {} gates → after {} gates",
            i,
            r_inv.gates.len() * 2 + 1, // approximate size
            compressed_block.gates.len()
        );

        println!("  {}", compressed_block.repr());

        compressed_block
    })
    .collect();

    // Combine blocks hierarchically
    let mut acc = blocks[0].clone();
    println!("Start combining: {}", acc.gates.len());

    for (i, b) in blocks.into_iter().skip(1).enumerate() {
        let combined = acc.concat(&b);
        let before = combined.gates.len();
        acc = compress(&combined, 500_000, conn, bit_shuf, n);
        let after = acc.gates.len();

        println!(
            "  Combine step {}: {} → {} gates",
            i + 1,
            before,
            after
        );
    }

    // Add bookends: R ... R*
    acc = r.concat(&acc).concat(&r_inv);
    println!("After adding bookends: {} gates", acc.gates.len());

    // Final global compression (until stable 3x)
    let mut stable_count = 0;
    while stable_count < 3 {
        let before = acc.gates.len();
        acc = compress(&acc, 1_000_000, conn, bit_shuf, n);
        let after = acc.gates.len();

        if after == before {
            stable_count += 1;
            println!("  Final compression stable {}/3 at {} gates", stable_count, after);
        } else {
            println!("  Final compression reduced: {} → {} gates", before, after);
            stable_count = 0;
        }
    }

    let mut i = 0;
    while i < acc.gates.len().saturating_sub(1) {
        if acc.gates[i] == acc.gates[i + 1] {
            // remove elements at i and i+1
            acc.gates.drain(i..=i + 1);

            // step back up to 2 indices, but not below 0
            i = i.saturating_sub(2);
        } else {
            i += 1;
        }
    }
    //writeln!(file, "Permutation after remove identities 2 is: \n{:?}", acc.permutation(n).data).unwrap();
    println!("Compressed len: {}", acc.gates.len());

    println!("Butterfly done: {} gates", acc.gates.len());

    acc
}

//TODO change this so its faster. just do less merging
// fn merge_combine_blocks(
//     blocks: &[CircuitSeq],
//     n: usize,
//     db_path: &str,
//     progress: &Arc<AtomicUsize>,
//     total: usize,
// ) -> CircuitSeq {
//     if blocks.is_empty() {
//         return CircuitSeq { gates: vec![] };
//     }
//     if blocks.len() == 1 {
//         let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
//         if done % 10 == 0 || done == total {
//             println!("Progress: {}/{}", done, total);
//         }
//         return blocks[0].clone();
//     }

//     let mid = blocks.len() / 2;

//     let (left, right) = rayon::join(
//         || merge_combine_blocks(&blocks[..mid], n, db_path, progress, total),
//         || merge_combine_blocks(&blocks[mid..], n, db_path, progress, total),
//     );

//     let mut conn = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)
//         .expect("Failed to open read-only DB");

//     let combined = left.concat(&right);
//     // shoot_random_gate(&mut combined, 100_000);
    
//     let acc = compress_big(&combined, 200, n, &mut conn);

//     let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
//     if done % 10 == 0 || done == total {
//         println!("Progress: {}/{}", done, total);
//     }

//     acc
// }

pub fn merge_combine_blocks(
    blocks: &[CircuitSeq],
    n: usize,
    db_path: &str,
    progress: &Arc<AtomicUsize>,
) -> CircuitSeq {
    println!("Phase 1: Pairwise merge");
    let total_1 = (blocks.len()+1)/2;
    let pairs: Vec<CircuitSeq> = blocks
        .par_chunks(2)
        .map(|chunk| {
            let mut conn = Connection::open_with_flags(
                db_path,
                OpenFlags::SQLITE_OPEN_READ_ONLY,
            )
            .expect("Failed to open DB");

            let combined = if chunk.len() == 2 {
                chunk[0].concat(&chunk[1])
            } else {
                chunk[0].clone()
            };

            let compressed = compress_big(&combined, 30, n, &mut conn);

            let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 10 == 0 {
                println!("Phase 1 progress: {}/{}", done, total_1);
            }

            compressed
        })
        .collect();

    println!("Phase 2: 4-way merge");
    progress.store(0, Ordering::Relaxed);
    let chunk_size = (pairs.len() + 3) / 4;
    let phase2_results: Vec<CircuitSeq> = pairs
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut conn = Connection::open_with_flags(
                db_path,
                OpenFlags::SQLITE_OPEN_READ_ONLY,
            )
            .expect("Failed to open DB");

            let mut combined = CircuitSeq { gates: vec![] };
            for block in chunk {
                combined = combined.concat(block);
            }

            let compressed = compress_big(&combined, 100, n, &mut conn);

            let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
            println!("Phase 2 partial done: {}/4", done);

            compressed
        })
        .collect();

    println!("Phase 3: Final merge");
    // Final combination and compression
    let mut conn = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .expect("Failed to open DB");

    let mut final_combined = CircuitSeq { gates: vec![] };
    for part in phase2_results {
        final_combined = final_combined.concat(&part);
    }

    let final_compressed = compress_big(&final_combined, 1000, n, &mut conn);

    println!("All phases complete");
    final_compressed
}

pub fn butterfly_big(
    c: &CircuitSeq,
    conn: &mut Connection,
    n: usize,
) -> CircuitSeq {
    // Pick one random R
    let mut rng = rand::rng();
    let (r, r_inv) = random_id(n as u8, rng.random_range(150..=225)); 

    println!("Butterfly start: {} gates", c.gates.len());

    let r = &r;           // reference is enough; read-only
    let r_inv = &r_inv;   // same

    // Parallel processing of gates
    let blocks: Vec<CircuitSeq> = c.gates
        .par_iter()
        .enumerate()
        .map(|(i, &g)| {
            // wrap single gate as CircuitSeq
            let gi = r_inv.concat(&CircuitSeq { gates: vec![g] }).concat(&r);
            // create a read-only connection per thread
            let mut conn = Connection::open_with_flags(
            "circuits.db",
            OpenFlags::SQLITE_OPEN_READ_ONLY,
        ).expect("Failed to open read-only connection");
        //shoot_random_gate(&mut gi, 100_000);
        // compress the block
        let compressed_block = compress_big(&gi, 100, n, &mut conn);
        let before_len = r_inv.gates.len() * 2 + 1;
        let after_len = compressed_block.gates.len();
            
        let color_line = if after_len < before_len {
            "\x1b[32m──────────────\x1b[0m" // green
        } else if after_len > before_len {
            "\x1b[31m──────────────\x1b[0m" // red
        } else if gi.gates != compressed_block.gates {
            "\x1b[35m──────────────\x1b[0m" // purple
        } else {
            "\x1b[90m──────────────\x1b[0m" // gray
        };

        println!(
            "  Block {}: before {} gates → after {} gates  {}",
            i, before_len, after_len, color_line
        );

        println!("  {}", compressed_block.repr());

        compressed_block
    })
    .collect();

    let progress = Arc::new(AtomicUsize::new(0));
    let _total = 2 * blocks.len() - 1;

    println!("Beginning merge");
    
    let mut acc = merge_combine_blocks(&blocks, n, "./circuits.db", &progress);

    // Add bookends: R ... R*
    acc = r.concat(&acc).concat(&r_inv);
    println!("After adding bookends: {} gates", acc.gates.len());

    // Final global compression (until stable 3x)
    let mut stable_count = 0;
    while stable_count < 3 {
        let before = acc.gates.len();
        //shoot_random_gate(&mut acc, 100_000);
        acc = compress_big(&acc, 1_000, n, conn);
        let after = acc.gates.len();

        if after == before {
            stable_count += 1;
            println!("  Final compression stable {}/3 at {} gates", stable_count, after);
        } else {
            println!("  Final compression reduced: {} → {} gates", before, after);
            stable_count = 0;
        }
    }

    // let mut i = 0;
    // while i < acc.gates.len().saturating_sub(1) {
    //     if acc.gates[i] == acc.gates[i + 1] {
    //         // remove elements at i and i+1
    //         acc.gates.drain(i..=i + 1);

    //         // step back up to 2 indices, but not below 0
    //         i = i.saturating_sub(2);
    //     } else {
    //         i += 1;
    //     }
    // }
    //writeln!(file, "Permutation after remove identities 2 is: \n{:?}", acc.permutation(n).data).unwrap();
    println!("Compressed len: {}", acc.gates.len());

    println!("Butterfly done: {} gates", acc.gates.len());

    acc
}

pub fn abutterfly_big(
    c: &CircuitSeq,
    conn: &mut Connection,
    n: usize,
) -> CircuitSeq {
    println!("Butterfly start: {} gates", c.gates.len());
    let mut rng = rand::rng();
    let mut pre_blocks: Vec<CircuitSeq> = Vec::with_capacity(c.gates.len());
    let mut c = c.clone();
    shoot_random_gate(&mut c, 500_000);
    let (first_r, first_r_inv) = random_id(n as u8, rng.random_range(50..=100));
    let mut prev_r_inv = first_r_inv.clone();
    
    for &g in &c.gates {
        let (r, r_inv) = random_id(n as u8, rng.random_range(50..=100));
        let mut block = prev_r_inv.clone().concat(&CircuitSeq { gates: vec![g] }).concat(&r);
        shoot_random_gate(&mut block, 1_000);
        pre_blocks.push(block);
        prev_r_inv = r_inv;
    }

    // Parallel compression of each block
    let compressed_blocks: Vec<CircuitSeq> = pre_blocks
        .into_par_iter()
        .enumerate()
        .map(|(i, block)| {
            let mut thread_conn = Connection::open_with_flags(
                "circuits.db",
                OpenFlags::SQLITE_OPEN_READ_ONLY,
            )
            .expect("Failed to open read-only connection");

            let before_len = block.gates.len();
            let compressed_block = compress_big(&block, 10, n, &mut thread_conn);
            let after_len = compressed_block.gates.len();
            
            let color_line = if after_len < before_len {
                "\x1b[32m──────────────\x1b[0m" // green
            } else if after_len > before_len {
                "\x1b[31m──────────────\x1b[0m" // red
            } else if block.gates != compressed_block.gates {
                "\x1b[35m──────────────\x1b[0m" // purple
            } else {
                "\x1b[90m──────────────\x1b[0m" // gray
            };

            println!(
                "  Block {}: before {} gates → after {} gates  {}",
                i, before_len, after_len, color_line
            );
            //println!("  {}", compressed_block.repr());

            compressed_block
        })
        .collect();

    let progress = Arc::new(AtomicUsize::new(0));
    let _total = 2 * compressed_blocks.len() - 1;

    println!("Beginning merge");
    let mut acc =
        merge_combine_blocks(&compressed_blocks, n, "./circuits.db", &progress);

    // Add global bookends: first_r ... last_r_inv
    acc = first_r.concat(&acc).concat(&prev_r_inv);

    println!("After adding bookends: {} gates", acc.gates.len());

    // Final global compression until stable 3×
    let mut stable_count = 0;
    while stable_count < 3 {
        let before = acc.gates.len();
        //shoot_random_gate(&mut acc, 100_000);
        acc = compress_big(&acc, 1_000, n, conn);
        let after = acc.gates.len();

        if after == before {
            stable_count += 1;
            println!("  Final compression stable {}/3 at {} gates", stable_count, after);
        } else {
            println!("  Final compression reduced: {} → {} gates", before, after);
            stable_count = 0;
        }
    }

    println!("Compressed len: {}", acc.gates.len());
    println!("Butterfly done: {} gates", acc.gates.len());

    acc
}

pub fn abutterfly_big_delay_bookends(
    c: &CircuitSeq,
    conn: &mut Connection,
    n: usize,
) -> (CircuitSeq, CircuitSeq, CircuitSeq) {
    println!("Butterfly start: {} gates", c.gates.len());
    let mut rng = rand::rng();
    let mut pre_blocks: Vec<CircuitSeq> = Vec::with_capacity(c.gates.len());
    let mut c = c.clone();
    // let (first_r, first_r_inv) = random_id(n as u8, rng.random_range(20..=100));
    let (first_r, first_r_inv) = random_id(n as u8, rng.random_range(20..=70));
    let mut prev_r_inv = first_r_inv.clone();
    shoot_random_gate(&mut c, 100_000);
    for &g in &c.gates {
        // let (r, r_inv) = random_id(n as u8, rng.random_range(20..=100));
        let (r, r_inv) = random_id(n as u8, rng.random_range(20..=70));
        let mut block = prev_r_inv.clone().concat(&CircuitSeq { gates: vec![g] }).concat(&r);
        shoot_random_gate(&mut block, 1_000);
        pre_blocks.push(block);
        prev_r_inv = r_inv;
    }

    // Parallel compression of each block
    let compressed_blocks: Vec<CircuitSeq> = pre_blocks
        .into_par_iter()
        .enumerate()
        .map(|(i, block)| {
            let mut thread_conn = Connection::open_with_flags(
                "circuits.db",
                OpenFlags::SQLITE_OPEN_READ_ONLY,
            )
            .expect("Failed to open read-only connection");

            let before_len = block.gates.len();
            let compressed_block = compress_big(&block, 10, n, &mut thread_conn);
            let after_len = compressed_block.gates.len();
            
            let color_line = if after_len < before_len {
                "\x1b[32m──────────────\x1b[0m" // green
            } else if after_len > before_len {
                "\x1b[31m──────────────\x1b[0m" // red
            } else if block.gates != compressed_block.gates {
                "\x1b[35m──────────────\x1b[0m" // purple
            } else {
                "\x1b[90m──────────────\x1b[0m" // gray
            };

            println!(
                "  Block {}: before {} gates → after {} gates  {}",
                i, before_len, after_len, color_line
            );
            //println!("  {}", compressed_block.repr());

            compressed_block
        })
        .collect();

    let progress = Arc::new(AtomicUsize::new(0));
    let _total = 2 * compressed_blocks.len() - 1;

    println!("Beginning merge");
    let mut acc =
        merge_combine_blocks(&compressed_blocks, n, "./circuits.db", &progress);

    println!("After merging: {} gates", acc.gates.len());

    // Final global compression until stable 3×
    let mut stable_count = 0;
    while stable_count < 3 {
        let before = acc.gates.len();

        let k = if before > 10_000 {
            16
        } else if before > 5_000 {
            8
        } else if before > 1_000 {
            4
        } else if before > 500 {
            2
        } else {
            1
        };

        let mut rng = rand::rng();

        let chunks = split_into_random_chunks(&acc.gates, k, &mut rng);

        let compressed_chunks: Vec<Vec<[u8;3]>> =
        chunks
            .into_par_iter()
            .map(|chunk| {
                let sub = CircuitSeq { gates: chunk };
                let mut thread_conn = Connection::open_with_flags(
                    "circuits.db",
                    OpenFlags::SQLITE_OPEN_READ_ONLY,
                )
                .expect("Failed to open read-only connection");
                compress_big(&sub, 1_000, n, &mut thread_conn).gates
            })
            .collect();

        let new_gates: Vec<[u8;3]> = compressed_chunks.into_iter().flatten().collect();
        acc.gates = new_gates;

        let after = acc.gates.len();

        if after == before {
            stable_count += 1;
            println!("  Final compression stable {}/3 at {} gates", stable_count, after);
        } else {
            println!("  Final compression reduced: {} → {} gates", before, after);
            stable_count = 0;
        }
    }

    println!("Compressed len: {}", acc.gates.len());
    println!("Butterfly done: {} gates", acc.gates.len());

    (acc, first_r, prev_r_inv)
}

fn split_into_random_chunks<T: Clone>(
    v: &[T],
    k: usize,
    rng: &mut impl rand::Rng,
) -> Vec<Vec<T>> {
    let n = v.len();
    if k <= 1 || n <= 1 {
        return vec![v.to_vec()];
    }

    // Minimum chunk size
    let min_size = 100;

    let max_chunks = k;
    let mut cuts = Vec::with_capacity(max_chunks - 1);
    let mut start = 0;

    // Generate cuts ensuring each chunk >= min_size
    for _ in 0..(max_chunks - 1) {
        // Remaining length that can still accommodate remaining cuts
        let remaining_chunks = max_chunks - cuts.len();
        let max_cut = n - (remaining_chunks * min_size);
        if start >= max_cut {
            break
        }
        let cut = rng.random_range(start + min_size..=max_cut);
        cuts.push(cut);
        start = cut;
    }

    let mut boundaries = vec![0];
    boundaries.extend(cuts);
    boundaries.push(n);

    boundaries
        .windows(2)
        .map(|w| v[w[0]..w[1]].to_vec())
        .collect()
}

pub fn main_mix(c: &CircuitSeq, rounds: usize, conn: &mut Connection, n: usize) {
    // Start with the input circuit
    println!("Starting len: {}", c.gates.len());
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
    let mut i = 0;
    while i < circuit.gates.len().saturating_sub(1) {
        if circuit.gates[i] == circuit.gates[i + 1] {
            // remove elements at i and i+1
            circuit.gates.drain(i..=i + 1);

            // step back up to 2 indices, but not below 0
            i = i.saturating_sub(2);
        } else {
            i += 1;
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

pub fn main_butterfly(c: &CircuitSeq, rounds: usize, conn: &mut Connection, n: usize) {
    // Start with the input circuit
    println!("Starting len: {}", c.gates.len());
    let mut circuit = c.clone();
    let perms: Vec<Vec<usize>> = (0..n).permutations(n).collect();
    let bit_shuf = perms.into_iter().skip(1).collect::<Vec<_>>();
    // Repeat obfuscate + compress 'rounds' times
    let mut post_len = 0;
    let mut count = 0;
    for _ in 0..rounds {
        circuit = butterfly(&circuit, conn, &bit_shuf, n);
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
        let mut i = 0;
        while i < circuit.gates.len().saturating_sub(1) {
            if circuit.gates[i] == circuit.gates[i + 1] {
                // remove elements at i and i+1
                circuit.gates.drain(i..=i + 1);

                // step back up to 2 indices, but not below 0
                i = i.saturating_sub(2);
            } else {
                i += 1;
            }
        }
    }
    println!("Final len: {}", circuit.gates.len());
    println!("Final cycle: {:?}", circuit.permutation(n).to_cycle());
    // Convert the final circuit to string
    let circuit_str = circuit.to_string(n);
    println!("Final Permutation: {:?}", circuit.permutation(n).data);
    if circuit.permutation(n).data != c.permutation(n).data {
        panic!(
            "The permutation differs from the original.\nOriginal: {:?}\nNew: {:?}",
            c.permutation(n).data,
            circuit.permutation(n).data
        );
    }
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

pub fn main_butterfly_big(c: &CircuitSeq, rounds: usize, conn: &mut Connection, n: usize, asymmetric: bool, save: &str) {
    // Start with the input circuit
    println!("Starting len: {}", c.gates.len());
    let mut circuit = c.clone();
    // Repeat obfuscate + compress 'rounds' times
    let mut post_len = 0;
    let mut count = 0;
    for _ in 0..rounds {
        circuit = if asymmetric {
            abutterfly_big(&circuit, conn, n)
        } else {
            butterfly_big(&circuit,conn,n)
        };
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
        let mut i = 0;
        while i < circuit.gates.len().saturating_sub(1) {
            if circuit.gates[i] == circuit.gates[i + 1] {
                // remove elements at i and i+1
                circuit.gates.drain(i..=i + 1);

                // step back up to 2 indices, but not below 0
                i = i.saturating_sub(2);
            } else {
                i += 1;
            }
        }
    }
    println!("Final len: {}", circuit.gates.len());
    // println!("Final cycle: {:?}", circuit.permutation(n).to_cycle());
    // println!("Final Permutation: {:?}", circuit.permutation(n).data);
    // if circuit.permutation(n).data != c.permutation(n).data {
    //     panic!(
    //         // "The permutation differs from the original.\nOriginal: {:?}\nNew: {:?}",
    //         // c.permutation(n).data,
    //         // circuit.permutation(n).data
    //         "The permutation differs from the original"
    //     );
    // }
    // let mut rev_gates = Vec::with_capacity(c.gates.len());
    // for g in c.gates.iter().rev() {
    //     rev_gates.push(*g); // copy [u8;3]
    // }
    // let rev = CircuitSeq { gates: rev_gates };
    // let good_id = circuit.concat(&rev);

    circuit
    .probably_equal(&c, n, 150_000)
    .expect("The circuits differ somewhere!");

    // Write to file
    let c_str = c.repr();
    let circuit_str = circuit.repr();
    let long_str = format!("{}:{}", c.repr(), circuit.repr());
    // let good_str = format!("{}: {}", good_id.gates.len(), good_id.repr());
    // Write start.txt
    File::create("start.txt")
        .and_then(|mut f| f.write_all(c_str.as_bytes()))
        .expect("Failed to write start.txt");

    // Write recent_circuit.txt
    File::create("recent_circuit.txt")
        .and_then(|mut f| f.write_all(circuit_str.as_bytes()))
        .expect("Failed to write recent_circuit.txt");

    File::create(save)
        .and_then(|mut f| f.write_all(circuit_str.as_bytes()))
        .expect("Failed to write recent_circuit.txt");

    // Write butterfly_recent.txt (overwrite)
    File::create("butterfly_recent.txt")
        .and_then(|mut f| f.write_all(long_str.as_bytes()))
        .expect("Failed to write butterfly_recent.txt");

    // Append to butterfly.txt
    OpenOptions::new()
        .append(true)
        .create(true)
        .open("butterfly.txt")
        .and_then(|mut f| writeln!(f, "{}", long_str))
        .expect("Failed to append to butterfly.txt");
    if circuit.gates == c.gates {
        println!("The obfuscation didn't do anything");
    }

    println!("Final circuit written to recent_circuit.txt");
}

pub fn main_butterfly_big_bookendsless(c: &CircuitSeq, rounds: usize, conn: &mut Connection, n: usize, _asymmetric: bool, save: &str) {
    // Start with the input circuit
    println!("Starting len: {}", c.gates.len());
    let mut circuit = c.clone();
    // Repeat obfuscate + compress 'rounds' times
    let mut post_len = 0;
    let mut count = 0;
    let mut beginning = CircuitSeq { gates: Vec::new() };
    let mut end= CircuitSeq { gates: Vec::new() };
    for _ in 0..rounds {
        let (new_circuit, b, e) = abutterfly_big_delay_bookends(&circuit, conn, n);
        beginning = beginning.concat(&b);
        end = e.concat(&end);
        circuit = new_circuit;
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
        let mut i = 0;
        while i < circuit.gates.len().saturating_sub(1) {
            if circuit.gates[i] == circuit.gates[i + 1] {
                // remove elements at i and i+1
                circuit.gates.drain(i..=i + 1);

                // step back up to 2 indices, but not below 0
                i = i.saturating_sub(2);
            } else {
                i += 1;
            }
        }
    }

    println!("Adding bookends");
    beginning = compress_big(&beginning, 100, n, conn);
    end = compress_big(&end, 100, n, conn);
    circuit = beginning.concat(&circuit).concat(&end);
    let mut c1 = CircuitSeq{ gates: circuit.gates[0..circuit.gates.len()/2].to_vec() };
    let mut c2 = CircuitSeq{ gates: circuit.gates[circuit.gates.len()/2..].to_vec() };
    c1 = compress_big(&c1, 1_000, n, conn);
    c2 = compress_big(&c2, 1_000, n, conn);
    circuit = c1.concat(&c2);
    let mut stable_count = 0;
    while stable_count < 3 {
        let before = circuit.gates.len();
        //shoot_random_gate(&mut acc, 100_000);
        circuit = compress_big(&circuit, 1_000, n, conn);
        let after = circuit.gates.len();

        if after == before {
            stable_count += 1;
            println!("  Final compression stable {}/3 at {} gates", stable_count, after);
        } else {
            println!("  Final compression reduced: {} → {} gates", before, after);
            stable_count = 0;
        }
    }

    println!("Final len: {}", circuit.gates.len());

    circuit
    .probably_equal(&c, n, 150_000)
    .expect("The circuits differ somewhere!");

    // Write to file
    let c_str = c.repr();
    let circuit_str = circuit.repr();
    let long_str = format!("{}:{}", c.repr(), circuit.repr());
    // let good_str = format!("{}: {}", good_id.gates.len(), good_id.repr());
    // Write start.txt
    File::create("start.txt")
        .and_then(|mut f| f.write_all(c_str.as_bytes()))
        .expect("Failed to write start.txt");

    // Write recent_circuit.txt
    File::create("recent_circuit.txt")
        .and_then(|mut f| f.write_all(circuit_str.as_bytes()))
        .expect("Failed to write recent_circuit.txt");

    File::create(save)
        .and_then(|mut f| f.write_all(circuit_str.as_bytes()))
        .expect("Failed to write recent_circuit.txt");

    // Write butterfly_recent.txt (overwrite)
    File::create("butterfly_recent.txt")
        .and_then(|mut f| f.write_all(long_str.as_bytes()))
        .expect("Failed to write butterfly_recent.txt");

    // Append to butterfly.txt
    OpenOptions::new()
        .append(true)
        .create(true)
        .open("butterfly.txt")
        .and_then(|mut f| writeln!(f, "{}", long_str))
        .expect("Failed to append to butterfly.txt");
    if circuit.gates == c.gates {
        println!("The obfuscation didn't do anything");
    }

    println!("Final circuit written to recent_circuit.txt");
}

//do targeted compression
pub fn main_compression(c: &CircuitSeq, rounds: usize, conn: &mut Connection, n: usize, save: &str) {
    // Start with the input circuit
    println!("Starting len: {}", c.gates.len());
    let mut circuit = c.clone();
    // Repeat obfuscate + compress 'rounds' times
    let mut post_len = 0;
    let mut count = 0;
    for _ in 0..rounds {
            butterfly_big(&circuit,conn,n);
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
        let mut i = 0;
        while i < circuit.gates.len().saturating_sub(1) {
            if circuit.gates[i] == circuit.gates[i + 1] {
                // remove elements at i and i+1
                circuit.gates.drain(i..=i + 1);

                // step back up to 2 indices, but not below 0
                i = i.saturating_sub(2);
            } else {
                i += 1;
            }
        }
    }
    println!("Final len: {}", circuit.gates.len());
    // println!("Final cycle: {:?}", circuit.permutation(n).to_cycle());
    // println!("Final Permutation: {:?}", circuit.permutation(n).data);
    // if circuit.permutation(n).data != c.permutation(n).data {
    //     panic!(
    //         // "The permutation differs from the original.\nOriginal: {:?}\nNew: {:?}",
    //         // c.permutation(n).data,
    //         // circuit.permutation(n).data
    //         "The permutation differs from the original"
    //     );
    // }

    circuit
    .probably_equal(&c, n, 150_000)
    .expect("The circuits differ somewhere!");

    // Write to file
    let c_str = c.repr();
    let circuit_str = circuit.repr();
    let long_str = format!("{}:{}", c.repr(), circuit.repr());
    // Write start.txt
    File::create("start.txt")
        .and_then(|mut f| f.write_all(c_str.as_bytes()))
        .expect("Failed to write start.txt");

    // Write recent_circuit.txt
    File::create("recent_circuit.txt")
        .and_then(|mut f| f.write_all(circuit_str.as_bytes()))
        .expect("Failed to write recent_circuit.txt");

    File::create(save)
        .and_then(|mut f| f.write_all(circuit_str.as_bytes()))
        .expect("Failed to write recent_circuit.txt");

    // Write butterfly_recent.txt (overwrite)
    File::create("butterfly_recent.txt")
        .and_then(|mut f| f.write_all(long_str.as_bytes()))
        .expect("Failed to write butterfly_recent.txt");

    // Append to butterfly.txt
    OpenOptions::new()
        .append(true)
        .create(true)
        .open("butterfly.txt")
        .and_then(|mut f| writeln!(f, "{}", long_str))
        .expect("Failed to append to butterfly.txt");
    if circuit.gates == c.gates {
        println!("The obfuscation didn't do anything");
    }

    println!("Final circuit written to recent_circuit.txt");
}