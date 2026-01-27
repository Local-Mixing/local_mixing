use crate::{
    circuit::circuit::CircuitSeq,
    random::random_data::shoot_random_gate,
    replace::replace::{
        compress, 
        compress_big, 
        expand_big, 
        obfuscate, 
        outward_compress, 
        random_id, 
        // replace_pair_distances, 
        replace_pair_distances_linear, 
        replace_pairs, 
        replace_sequential_pairs
    },
};
// use crate::random::random_data::random_walk_no_skeleton;
use rand::prelude::SliceRandom;
use itertools::Itertools;
// use lmdb::RoTransaction;
use rand::Rng;
use rayon::prelude::*;
use std::io;
use std::io::Read;
use rusqlite::{Connection, OpenFlags};

use once_cell::sync::Lazy;

use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io::Write,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc,
        Mutex,
    },
    time::Instant,
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

pub fn merge_combine_blocks(
    blocks: &[CircuitSeq],
    n: usize,
    db_path: &str,
    progress: &Arc<AtomicUsize>,
    _total: usize,
    env: &lmdb::Environment,
    bit_shuf_list: &Vec<Vec<Vec<usize>>>,
    dbs: &HashMap<String, lmdb::Database>,
) -> CircuitSeq {
    println!("Phase 1: Pairwise merge");
    // let total_1 = (blocks.len()+1)/2;
    let pairs: Vec<CircuitSeq> = blocks
        .par_chunks(2)
        .map(|chunk| {
            let mut conn = Connection::open_with_flags(
                db_path,
                OpenFlags::SQLITE_OPEN_READ_ONLY,
            )
            .expect("Failed to open DB");
            // TXN
            let combined = if chunk.len() == 2 {
                chunk[0].concat(&chunk[1])
            } else {
                chunk[0].clone()
            };

            let compressed = compress_big(&combined, 30, n, &mut conn, env, &bit_shuf_list, dbs);
            compressed
        })
        .collect();

    println!("Phase 2: Offset pairwise merge");

    // Skip the first block
    let mut phase2_blocks = Vec::new();
    phase2_blocks.push(pairs[0].clone());

    // Pair the rest starting from index 1
    let rest = &pairs[1..];

    let phase2_pairs: Vec<CircuitSeq> = rest
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

            // TXN
            let compressed = compress_big(&combined, 30, n, &mut conn, env, &bit_shuf_list, dbs);

            // let _done = progress2.fetch_add(1, Ordering::Relaxed) + 1;
            // if done % 10 == 0 {
            //     println!("Phase 2 progress: {}/{}", done, total_2);
            // }

            compressed
        })
        .collect();

    // Prepend the untouched first block
    phase2_blocks.extend(phase2_pairs);

    println!("Phase 3: 4-way merge");
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
            // TXN
            let compressed = compress_big(&combined, 200, n, &mut conn, env, &bit_shuf_list, dbs);

            let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
            println!("Phase 2 partial done: {}/4", done);

            compressed
        })
        .collect();

    println!("Phase 4: Final merge");
    // Final combination and compression
    let mut conn = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .expect("Failed to open DB");

    let mut final_combined = CircuitSeq { gates: vec![] };
    for part in phase2_results {
        final_combined = final_combined.concat(&part);
    }

    // TXN
    let final_compressed = compress_big(&final_combined, 1000, n, &mut conn, env, &bit_shuf_list, dbs);

    println!("All phases complete");
    final_compressed
}

// fn initial_milestone(acc: usize) -> usize {
//     if acc >= 10_000 {
//         (acc / 10_000) * 10_000   // nearest 10k below
//     } else if acc >= 5_000 {
//         5_000
//     } else if acc >= 2_000 {
//         2_000
//     } else if acc >= 1_000 {
//         1_000
//     } else {
//         0
//     }
// }

/// Given the previous milestone, decide the next lower one
// fn next_milestone(prev: usize) -> usize {
//     match prev {
//         x if x > 10_000 => x - 10_000,
//         10_000 => 5_000,
//         5_000 => 2_000,
//         2_000 => 1_000,
//         _ => 0,
//     }
// }

pub fn butterfly_big(
    c: &CircuitSeq,
    _conn: &mut Connection,
    n: usize,
    last: bool,
    stop: usize,
    env: &lmdb::Environment,
    bit_shuf_list: &Vec<Vec<Vec<usize>>>,
    dbs: &HashMap<String, lmdb::Database>
) -> CircuitSeq {
    // Pick one random R
    let mut rng = rand::rng();
    let (r, r_inv) = random_id(n as u8, rng.random_range(100..=200)); 
    let mut c = c.clone();
    shoot_random_gate(&mut c, 500_000);
    println!("Butterfly start: {} gates", c.gates.len());

    let r = &r;           // reference is enough; read-only
    let r_inv = &r_inv;   // same
    // Parallel processing of gates
    let blocks: Vec<CircuitSeq> = c.gates
        .par_iter()
        .enumerate()
        .map(|(i, &g)| {
        // wrap single gate as CircuitSeq
        let mut gi = r_inv.concat(&CircuitSeq { gates: vec![g] }).concat(&r);
        shoot_random_gate(&mut gi, 1_000);
        // create a read-only connection per thread
        let mut conn = Connection::open_with_flags(
        "circuits.db",
        OpenFlags::SQLITE_OPEN_READ_ONLY,
        ).expect("Failed to open read-only connection");
        //shoot_random_gate(&mut gi, 100_000);
        // compress the block

        // let _txn = env.begin_ro_txn().expect("txn");

        // TXN
        let compressed_block = compress_big(&gi, 10, n, &mut conn, env, &bit_shuf_list, dbs);
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

        // println!("  {}", compressed_block.repr());

        compressed_block
    })
    .collect();

    let progress = Arc::new(AtomicUsize::new(0));
    let _total = 2 * blocks.len() - 1;

    println!("Beginning merge");
    
    let mut acc = merge_combine_blocks(&blocks, n, "./circuits.db", &progress, _total, env, &bit_shuf_list, dbs);

    // Add bookends: R ... R*
    acc = r.concat(&acc).concat(&r_inv);
    println!("After adding bookends: {} gates", acc.gates.len());
    // let mut milestone = initial_milestone(acc.gates.len());
    // Final global compression (until stable 3x)
    let mut stable_count = 0;
    while stable_count < 3 {
        
        // if acc.gates.len() <= milestone {
        //     let mut f = OpenOptions::new()
        //         .create(true)
        //         .append(true)
        //         .open("bcircuitlist.txt")
        //         .expect("Could not open bcircuitlist.txt");

        //     writeln!(f, "{}", acc.repr()).unwrap();
        //     milestone = next_milestone(milestone);
        // }
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
                // TXN
                // let txn = env.begin_ro_txn().expect("txn");
                compress_big(&sub, 1_000, n, &mut thread_conn, env, &bit_shuf_list, dbs).gates
            })
            .collect();

        let new_gates: Vec<[u8;3]> = compressed_chunks.into_iter().flatten().collect();
        acc.gates = new_gates;

        let after = acc.gates.len();
        if last && acc.gates.len() <= stop {
            break
        }
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

pub static SHOOT_RANDOM_GATE_TIME: Lazy<AtomicU64> = Lazy::new(|| AtomicU64::new(0));
pub static REPLACE_PAIRS_TIME: Lazy<AtomicU64> = Lazy::new(|| AtomicU64::new(0));
pub static RANDOM_ID_TIME: Lazy<AtomicU64> = Lazy::new(|| AtomicU64::new(0));
pub static EXPAND_BIG_TIME: Lazy<AtomicU64> = Lazy::new(|| AtomicU64::new(0));
pub static COMPRESS_BIG_TIME: Lazy<AtomicU64> = Lazy::new(|| AtomicU64::new(0));
pub static MERGE_COMBINE_BLOCKS_TIME: Lazy<AtomicU64> = Lazy::new(|| AtomicU64::new(0));


static CURRENT_ACC: Lazy<Mutex<Option<CircuitSeq>>> =
    Lazy::new(|| Mutex::new(None));

static SHOULD_DUMP: AtomicBool = AtomicBool::new(false);
use signal_hook::consts::{SIGINT, SIGTERM};
use signal_hook::iterator::Signals;
use std::thread;
use std::process::exit;

pub fn install_kill_handler() {
    let mut signals = Signals::new([SIGINT, SIGTERM]).expect("signals");

    thread::spawn(move || {
        for _ in signals.forever() {
            eprintln!("Received termination signal, dumping acc...");
            SHOULD_DUMP.store(true, Ordering::SeqCst);
            break;
        }
    });
}

fn dump_and_exit() -> ! {
    if let Some(acc) = CURRENT_ACC.lock().unwrap().as_ref() {
        let mut f = File::create("killed.txt").expect("create killed.txt");
        writeln!(f, "{}", acc.repr()).expect("write");
        eprintln!("Wrote killed.txt");
    }
    exit(1);
}

pub fn abutterfly_big(
    c: &CircuitSeq,
    _conn: &mut Connection,
    n: usize,
    last: bool,
    stop: usize,
    env: &lmdb::Environment,
    curr_round: usize,
    last_round: usize,
    bit_shuf_list: &Vec<Vec<Vec<usize>>>,
    dbs: &HashMap<String, lmdb::Database>
) -> CircuitSeq {
    println!("Current round: {}/{}", curr_round, last_round);
    println!("Butterfly start: {} gates", c.gates.len());
    let mut rng = rand::rng();

    let mut c = c.clone();
    let t0 = Instant::now();
    shoot_random_gate(&mut c, 500_000);
    SHOOT_RANDOM_GATE_TIME.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    // c = random_walk_no_skeleton(&c, &mut rng);
    let (first_r, first_r_inv) = random_id(n as u8, rng.random_range(10..=30));
    let mut prev_r_inv = first_r_inv.clone();
    let t1 = Instant::now();
    replace_pairs(&mut c, n, _conn, &env);
    REPLACE_PAIRS_TIME.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);

    let mut pre_blocks: Vec<CircuitSeq> = Vec::with_capacity(c.gates.len());

    for &g in &c.gates {
        let t2 = Instant::now();
        let (r, r_inv) = random_id(n as u8, rng.random_range(10..=30));
        RANDOM_ID_TIME.fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let mut block = prev_r_inv.clone().concat(&CircuitSeq { gates: vec![g] }).concat(&r);
        shoot_random_gate(&mut block, 1_000);
        // block = random_walk_no_skeleton(&block, &mut rng);
        pre_blocks.push(block);
        prev_r_inv = r_inv;
    }
    let grew = Arc::new(AtomicUsize::new(0));
    let reduced = Arc::new(AtomicUsize::new(0));
    let swapped = Arc::new(AtomicUsize::new(0));
    let no_change = Arc::new(AtomicUsize::new(0));
    // Parallel compression of each block
    let compressed_blocks: Vec<CircuitSeq> = pre_blocks
        .into_par_iter()
        .enumerate()
        .map(|(_, block)| {
            let mut thread_conn = Connection::open_with_flags(
                "circuits.db",
                OpenFlags::SQLITE_OPEN_READ_ONLY,
            )
            .expect("Failed to open read-only connection");
            // let txn = env.begin_ro_txn().expect("txn");
            let before_len = block.gates.len();
            let t3 = Instant::now();
            let expanded = expand_big(&block, 100, n, &mut thread_conn, &env, &bit_shuf_list, dbs);
            EXPAND_BIG_TIME.fetch_add(t3.elapsed().as_nanos() as u64, Ordering::Relaxed);
            let t4 = Instant::now();

            // TXN
            let compressed_block = compress_big(&expanded, 100, n, &mut thread_conn, env, &bit_shuf_list, dbs);
            COMPRESS_BIG_TIME.fetch_add(t4.elapsed().as_nanos() as u64, Ordering::Relaxed);
            let after_len = compressed_block.gates.len();
            
            if after_len < before_len {
                reduced.fetch_add(1, Ordering::Relaxed); 
            } else if after_len > before_len {
                grew.fetch_add(1, Ordering::Relaxed);
            } else if block.gates != compressed_block.gates {
                swapped.fetch_add(1, Ordering::Relaxed);
            } else {
                no_change.fetch_add(1, Ordering::Relaxed);
            };
            compressed_block
        })
        .collect();
    
    println!("Summary:");
    println!("\x1b[32mGrew:      {}\x1b[0m", grew.load(Ordering::Relaxed));
    println!("\x1b[31mReduced:   {}\x1b[0m", reduced.load(Ordering::Relaxed));
    println!("\x1b[34mSwapped:   {}\x1b[0m", swapped.load(Ordering::Relaxed));
    println!("\x1b[90mNo change: {}\x1b[0m", no_change.load(Ordering::Relaxed));

    let progress = Arc::new(AtomicUsize::new(0));
    let _total = 2 * compressed_blocks.len() - 1;

    println!("Beginning merge");
    let t5 = Instant::now();
    let mut acc = merge_combine_blocks(&compressed_blocks, n, "./circuits.db", &progress, _total, env, &bit_shuf_list, dbs);
    MERGE_COMBINE_BLOCKS_TIME.fetch_add(t5.elapsed().as_nanos() as u64, Ordering::Relaxed);

    // Add global bookends: first_r ... last_r_inv
    acc = first_r.concat(&acc).concat(&prev_r_inv);

    acc = CircuitSeq { gates: acc.gates.clone() };
    println!("After adding bookends: {} gates", acc.gates.len());
    
    // let mut milestone = initial_milestone(acc.gates.len());
    // Final global compression until stable 6×
    let mut rng = rand::rng();
    let mut stable_count = 0;
    while stable_count < 12 {
        // if acc.gates.len() <= milestone {
        //     let mut f = OpenOptions::new()
        //         .create(true)
        //         .append(true)d
        //         .open("circuitlist.txt")
        //         .expect("Could not open circuitlist.txt");

        //     writeln!(f, "{}", acc.repr()).unwrap();
        //     milestone = next_milestone(milestone);
        // }

        let before = acc.gates.len();

        let k = if before <= 1500 {
            1
        } else {
            (before + 1499) / 1500 
        };

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
                // let txn = env.begin_ro_txn().expect("txn");
                // TXN
                compress_big(&sub, 100, n, &mut thread_conn, env, &bit_shuf_list, dbs).gates
            })
            .collect();

        let new_gates: Vec<[u8;3]> = compressed_chunks.into_iter().flatten().collect();
        acc.gates = new_gates;
        if SHOULD_DUMP.load(Ordering::SeqCst) {
            {
            let mut guard = CURRENT_ACC.lock().unwrap();
            *guard = Some(acc.clone());
        }

            dump_and_exit();
        }
        let after = acc.gates.len();
        if last && acc.gates.len() <= stop {
            break
        }
        if after == before {
            stable_count += 1;
            println!("  {}/{} Final compression stable {}/12 at {} gates", curr_round, last_round, stable_count, after);
        } else {
            println!("  {}/{}: {} → {} gates", curr_round, last_round, before, after);
            stable_count = 0;
        }
    }

    println!("Compressed len: {}", acc.gates.len());
    println!("Butterfly done: {} gates", acc.gates.len());
    println!("Timers (minutes):");
    println!("  shoot_random_gate:      {:.3}", SHOOT_RANDOM_GATE_TIME.load(Ordering::Relaxed) as f64 / 1e9 / 60.0);
    println!("  replace_pairs:          {:.3}", REPLACE_PAIRS_TIME.load(Ordering::Relaxed) as f64 / 1e9 / 60.0);
    println!("  random_id:              {:.3}", RANDOM_ID_TIME.load(Ordering::Relaxed) as f64 / 1e9 / 60.0);
    println!("  compress_big:           {:.3}", COMPRESS_BIG_TIME.load(Ordering::Relaxed) as f64 / 1e9 / 60.0);
    println!("  merge_combine_blocks:   {:.3}", MERGE_COMBINE_BLOCKS_TIME.load(Ordering::Relaxed) as f64 / 1e9 / 60.0);

    crate::replace::replace::print_compress_timers();
    acc
}

// pub fn abutterfly_big_delay_bookends(
//     c: &CircuitSeq,
//     _conn: &mut Connection,
//     n: usize,
//     env: &lmdb::Environment,
// ) -> (CircuitSeq, CircuitSeq, CircuitSeq) {
//     let dbs = open_all_dbs(env);
//     let bit_shuf_list = (3..=7)
//         .map(|n| {
//             (0..n)
//                 .permutations(n)
//                 .filter(|p| !p.iter().enumerate().all(|(i, &x)| i == x))
//                 .collect::<Vec<Vec<usize>>>()
//         })
//         .collect();
//     println!("Butterfly start: {} gates", c.gates.len());
//     let mut rng = rand::rng();
//     let mut pre_blocks: Vec<CircuitSeq> = Vec::with_capacity(c.gates.len());
//     let mut c = c.clone();
//     // let (first_r, first_r_inv) = random_id(n as u8, rng.random_range(20..=100));
//     let (first_r, first_r_inv) = random_id(n as u8, rng.random_range(150..=200));
//     let mut prev_r_inv = first_r_inv.clone();
//     shoot_random_gate(&mut c, 100_000);
//     for &g in &c.gates {
//         // let (r, r_inv) = random_id(n as u8, rng.random_range(20..=100));
//         let (r, r_inv) = random_id(n as u8, rng.random_range(150..=200));
//         let mut block = prev_r_inv.clone().concat(&CircuitSeq { gates: vec![g] }).concat(&r);
//         shoot_random_gate(&mut block, 1_000);
//         pre_blocks.push(block);
//         prev_r_inv = r_inv;
//     }

//     // Parallel compression of each block
//     let compressed_blocks: Vec<CircuitSeq> = pre_blocks
//         .into_par_iter()
//         .enumerate()
//         .map(|(i, block)| {
//             let mut thread_conn = Connection::open_with_flags(
//                 "circuits.db",
//                 OpenFlags::SQLITE_OPEN_READ_ONLY,
//             )
//             .expect("Failed to open read-only connection");
//             let txn = env.begin_ro_txn().expect("txn");
//             let before_len = block.gates.len();
//             // TXN
//             let compressed_block = compress_big(&block, 10, n, &mut thread_conn, env, &bit_shuf_list, &dbs, &txn);
//             let after_len = compressed_block.gates.len();
            
//             let color_line = if after_len < before_len {
//                 "\x1b[32m──────────────\x1b[0m" // green
//             } else if after_len > before_len {
//                 "\x1b[31m──────────────\x1b[0m" // red
//             } else if block.gates != compressed_block.gates {
//                 "\x1b[35m──────────────\x1b[0m" // purple
//             } else {
//                 "\x1b[90m──────────────\x1b[0m" // gray
//             };

//             println!(
//                 "  Block {}: before {} gates → after {} gates  {}",
//                 i, before_len, after_len, color_line
//             );
//             //println!("  {}", compressed_block.repr());

//             compressed_block
//         })
//         .collect();

//     let progress = Arc::new(AtomicUsize::new(0));
//     let _total = 2 * compressed_blocks.len() - 1;

//     println!("Beginning merge");
//     let mut acc =
//         merge_combine_blocks(&compressed_blocks, n, "./circuits.db", &progress, _total, env, &bit_shuf_list, &dbs);

//     println!("After merging: {} gates", acc.gates.len());

//     // Final global compression until stable 3×
//     let mut stable_count = 0;
//     while stable_count < 3 {
//         let before = acc.gates.len();

//         let k = if before > 10_000 {
//             16
//         } else if before > 5_000 {
//             8
//         } else if before > 1_000 {
//             4
//         } else if before > 500 {
//             2
//         } else {
//             1
//         };

//         let mut rng = rand::rng();

//         let chunks = split_into_random_chunks(&acc.gates, k, &mut rng);

//         let compressed_chunks: Vec<Vec<[u8;3]>> =
//         chunks
//             .into_par_iter()
//             .map(|chunk| {
//                 let sub = CircuitSeq { gates: chunk };
//                 let mut thread_conn = Connection::open_with_flags(
//                     "circuits.db",
//                     OpenFlags::SQLITE_OPEN_READ_ONLY,
//                 )
//                 .expect("Failed to open read-only connection");
//                 let txn = env.begin_ro_txn().expect("txn");
//                 // TXN
//                 compress_big(&sub, 1_000, n, &mut thread_conn, env, &bit_shuf_list, &dbs, &txn).gates
//             })
//             .collect();

//         let new_gates: Vec<[u8;3]> = compressed_chunks.into_iter().flatten().collect();
//         acc.gates = new_gates;

//         let after = acc.gates.len();

//         if after == before {
//             stable_count += 1;
//             println!("  Final compression stable {}/3 at {} gates", stable_count, after);
//         } else {
//             println!("  Final compression reduced: {} → {} gates", before, after);
//             stable_count = 0;
//         }
//     }

//     println!("Compressed len: {}", acc.gates.len());
//     println!("Butterfly done: {} gates", acc.gates.len());

//     (acc, first_r, prev_r_inv)
// }

static ALREADY_COLLIDED: AtomicUsize = AtomicUsize::new(0);
static SHOOT_COUNT: AtomicUsize = AtomicUsize::new(0);
static MADE_LEFT: AtomicUsize = AtomicUsize::new(0);
static TRAVERSE_LEFT: AtomicUsize = AtomicUsize::new(0);

pub fn replace_and_compress_big(
    circuit: &CircuitSeq,
    _conn: &mut Connection,
    n: usize,
    last: bool,
    stop: usize,
    env: &lmdb::Environment,
    curr_round: usize,
    last_round: usize,
    bit_shuf_list: &Vec<Vec<Vec<usize>>>,
    dbs: &HashMap<String, lmdb::Database>,
    intermediate: &str,
) -> (CircuitSeq, usize, usize, usize, usize) {
    println!("Current round: {}/{}", curr_round, last_round);

    ALREADY_COLLIDED.store(0, Ordering::SeqCst);
    SHOOT_COUNT.store(0, Ordering::SeqCst);
    MADE_LEFT.store(0, Ordering::SeqCst);
    TRAVERSE_LEFT.store(0, Ordering::SeqCst);

    println!("Butterfly start: {} gates", circuit.gates.len());
    let mut c = circuit.clone();
    let t0 = Instant::now();
    shoot_random_gate(&mut c, 200_000);
    SHOOT_RANDOM_GATE_TIME.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

    let t1 = Instant::now();
    for _ in 0..2 {
        let t0 = Instant::now();
        shoot_random_gate(&mut c, 200_000);
        SHOOT_RANDOM_GATE_TIME.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let k = if c.gates.len() <= 1500 {
            1
        } else {
            (c.gates.len() + 1499) / 1500 
        };
        let k = std::cmp::min(k, 60);
        let mut rng = rand::rng();
        let chunks = split_into_random_chunks(&c.gates, k, &mut rng);
        println!(
            "Starting replace_sequential_pairs , circuit length: {} , num_wires: {}",
            c.gates.len(), n
        );
        let replaced_chunks: Vec<Vec<[u8;3]>> =
        chunks
            .into_par_iter()
            .map(|chunk| {
                let mut sub = CircuitSeq { gates: chunk };
                let mut thread_conn = Connection::open_with_flags(
                    "circuits.db",
                    OpenFlags::SQLITE_OPEN_READ_ONLY,
                )
                .expect("Failed to open read-only connection");
                let (col, shoot, zero, trav) = replace_sequential_pairs(&mut sub, n, &mut thread_conn, &env, &bit_shuf_list, dbs);
                ALREADY_COLLIDED.fetch_add(col, Ordering::SeqCst);
                SHOOT_COUNT.fetch_add(shoot, Ordering::SeqCst);
                MADE_LEFT.fetch_add(zero, Ordering::SeqCst);
                TRAVERSE_LEFT.fetch_add(trav, Ordering::SeqCst);
                shoot_random_gate(&mut sub, 200_000);
                sub.gates.reverse();
                let (col, shoot, zero, trav) = replace_sequential_pairs(&mut sub, n, &mut thread_conn, &env, &bit_shuf_list, dbs);
                ALREADY_COLLIDED.fetch_add(col, Ordering::SeqCst);
                SHOOT_COUNT.fetch_add(shoot, Ordering::SeqCst);
                MADE_LEFT.fetch_add(zero, Ordering::SeqCst);
                TRAVERSE_LEFT.fetch_add(trav, Ordering::SeqCst);
                sub.gates.reverse();
                sub.gates
            })
            .collect();
            let new_gates: Vec<[u8;3]> = replaced_chunks.into_iter().flatten().collect();
            c.gates = new_gates;
    }
    REPLACE_PAIRS_TIME.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);
    println!(
        "Finished replace_sequential_pairs, new length: {}",
        c.gates.len()
    );
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(intermediate)
        .expect("Failed to open replacednocomp.txt");
    println!("Writing to {}", intermediate);
    writeln!(f, "{}", c.repr()).expect("Failed to write intermediate CircuitSeq");

    // let mut milestone = initial_milestone(acc.gates.len());
    // Final global compression until stable 6×
    println!("Beginning compression");
    let mut acc = c;
    let mut rng = rand::rng();
    let mut stable_count = 0;
    while stable_count < 12 {
        // if acc.gates.len() <= milestone {
        //     let mut f = OpenOptions::new()
        //         .create(true)
        //         .append(true)d
        //         .open("circuitlist.txt")
        //         .expect("Could not open circuitlist.txt");

        //     writeln!(f, "{}", acc.repr()).unwrap();
        //     milestone = next_milestone(milestone);
        // }

        let before = acc.gates.len();

        let k = if before <= 1500 {
            1
        } else {
            (before + 1499) / 1500 
        };

        let chunks = split_into_random_chunks(&acc.gates, k, &mut rng);
        let t4 = Instant::now();
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
                
                compress_big(&sub, 100, n, &mut thread_conn, env, &bit_shuf_list, dbs).gates
            })
            .collect();
        COMPRESS_BIG_TIME.fetch_add(t4.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let new_gates: Vec<[u8;3]> = compressed_chunks.into_iter().flatten().collect();
        acc.gates = new_gates;
        if SHOULD_DUMP.load(Ordering::SeqCst) {
            {
            let mut guard = CURRENT_ACC.lock().unwrap();
            *guard = Some(acc.clone());
        }

            dump_and_exit();
        }
        let after = acc.gates.len();
        if last && acc.gates.len() <= stop {
            break
        }
        if after == before {
            stable_count += 1;
            // println!("  {}/{} Final compression stable {}/12 at {} gates", curr_round, last_round, stable_count, after);
        } else {
            // println!("  {}/{}: {} → {} gates", curr_round, last_round, before, after);
            stable_count = 0;
        }

        let mut buf = [0u8; 1];
        if let Ok(n) = io::stdin().read(&mut buf) {
            if n > 0 && buf[0] == b'\n' {
                println!("  {}/{}: Current gates: {} gates", curr_round, last_round, after);
            }
        }
    }

    println!("Compressed len: {}", acc.gates.len());
    println!("Butterfly done: {} gates", acc.gates.len());
    println!("Timers (minutes):");
    println!("  shoot_random_gate:      {:.3}", SHOOT_RANDOM_GATE_TIME.load(Ordering::Relaxed) as f64 / 1e9 / 60.0);
    println!("  replace_pairs:          {:.3}", REPLACE_PAIRS_TIME.load(Ordering::Relaxed) as f64 / 1e9 / 60.0);
    println!("  random_id:              {:.3}", RANDOM_ID_TIME.load(Ordering::Relaxed) as f64 / 1e9 / 60.0);
    println!("  compress_big:           {:.3}", COMPRESS_BIG_TIME.load(Ordering::Relaxed) as f64 / 1e9 / 60.0);
    println!("  merge_combine_blocks:   {:.3}", MERGE_COMBINE_BLOCKS_TIME.load(Ordering::Relaxed) as f64 / 1e9 / 60.0);

    crate::replace::replace::print_compress_timers();
    (
        acc,
        ALREADY_COLLIDED.load(Ordering::SeqCst),
        SHOOT_COUNT.load(Ordering::SeqCst),
        MADE_LEFT.load(Ordering::SeqCst),
        TRAVERSE_LEFT.load(Ordering::SeqCst),
    )
}

pub fn replace_and_compress_big_distance(
    circuit: &CircuitSeq,
    _conn: &mut Connection,
    n: usize,
    last: bool,
    stop: usize,
    env: &lmdb::Environment,
    curr_round: usize,
    last_round: usize,
    bit_shuf_list: &Vec<Vec<Vec<usize>>>,
    dbs: &HashMap<String, lmdb::Database>,
    intermediate: &str,
    min: usize,
) -> CircuitSeq {
    println!("Current round: {}/{}", curr_round, last_round);
    println!("Replace and compress distance start: {} gates", circuit.gates.len());
    let mut c = circuit.clone();
    let t0 = Instant::now();
    shoot_random_gate(&mut c, 200_000);
    SHOOT_RANDOM_GATE_TIME.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

    let t1 = Instant::now();
    replace_pair_distances_linear(&mut c, n, _conn, env, bit_shuf_list, dbs, min);
    REPLACE_PAIRS_TIME.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);
    println!(
        "Finished replace_sequential_pairs, new length: {}",
        c.gates.len()
    );
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(intermediate)
        .expect("Failed to open replacednocomp.txt");
    println!("Writing to {}", intermediate);
    writeln!(f, "{}", c.repr()).expect("Failed to write intermediate CircuitSeq");

    // let mut milestone = initial_milestone(acc.gates.len());
    // Final global compression until stable 6×
    println!("Beginning compression");
    let mut acc = c;
    let mut rng = rand::rng();
    let mut stable_count = 0;
    while stable_count < 12 {
        // if acc.gates.len() <= milestone {
        //     let mut f = OpenOptions::new()
        //         .create(true)
        //         .append(true)d
        //         .open("circuitlist.txt")
        //         .expect("Could not open circuitlist.txt");

        //     writeln!(f, "{}", acc.repr()).unwrap();
        //     milestone = next_milestone(milestone);
        // }

        let before = acc.gates.len();

        let k = if before <= 1500 {
            1
        } else {
            (before + 1499) / 1500 
        };

        let chunks = split_into_random_chunks(&acc.gates, k, &mut rng);
        let t4 = Instant::now();
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
                
                compress_big(&sub, 100, n, &mut thread_conn, env, &bit_shuf_list, dbs).gates
            })
            .collect();
        COMPRESS_BIG_TIME.fetch_add(t4.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let new_gates: Vec<[u8;3]> = compressed_chunks.into_iter().flatten().collect();
        acc.gates = new_gates;
        if SHOULD_DUMP.load(Ordering::SeqCst) {
            {
            let mut guard = CURRENT_ACC.lock().unwrap();
            *guard = Some(acc.clone());
        }

            dump_and_exit();
        }
        let after = acc.gates.len();
        if last && acc.gates.len() <= stop {
            break
        }
        if after == before {
            stable_count += 1;
            // println!("  {}/{} Final compression stable {}/12 at {} gates", curr_round, last_round, stable_count, after);
        } else {
            // println!("  {}/{}: {} → {} gates", curr_round, last_round, before, after);
            stable_count = 0;
        }

        let mut buf = [0u8; 1];
        if let Ok(n) = io::stdin().read(&mut buf) {
            if n > 0 && buf[0] == b'\n' {
                println!("  {}/{}: Current gates: {} gates", curr_round, last_round, after);
            }
        }
    }

    println!("Compressed len: {}", acc.gates.len());
    println!("Replace and compress distance done: {} gates", acc.gates.len());
    acc
}

pub fn split_into_random_chunks(
    v: &Vec<[u8;3]>,
    k: usize,
    rng: &mut impl Rng,
) -> Vec<Vec<[u8;3]>> {

    if k == 1 {
        return vec![v.clone()]
    }
    let min_size = 100;
    let n = v.len();
    assert!(k * min_size <= n);

    let slack = n - k * min_size;

    let mut cuts: Vec<usize> = (0..slack).collect();
    cuts.shuffle(rng);
    cuts.truncate(k - 1);
    cuts.sort_unstable();

    let mut sizes = Vec::with_capacity(k);
    let mut prev = 0;

    for &c in &cuts {
        sizes.push(c - prev + min_size); 
        prev = c;
    }
    sizes.push(slack - prev + min_size);

    let mut out = Vec::with_capacity(k);
    let mut idx = 0;
    for size in sizes {
        out.push(v[idx..idx + size].to_vec());
        idx += size;
    }

    out
}

pub fn open_all_dbs(env: &lmdb::Environment) -> HashMap<String, lmdb::Database> {
    let mut dbs = HashMap::new();
    let db_names = [
        "n3m1","n3m2","n3m3","n3m4","n3m5","n3m6","n3m7","n3m8","n3m9","n3m10",
        "n4m1","n4m2","n4m3","n4m4","n4m5","n4m6",
        "n5m1","n5m2","n5m3","n5m4","n5m5",
        "n6m1","n6m2","n6m3","n6m4","n6m5",
        "n7m1","n7m2","n7m3","n7m4",
        "perm_tables_n3","perm_tables_n4","perm_tables_n5","perm_tables_n6","perm_tables_n7",
        "n4m1perms","n4m2perms","n4m3perms","n4m4perms","n4m5perms","n4m6perms",
        "n5m1perms","n5m2perms","n5m3perms","n5m4perms","n5m5perms",
        "n6m1perms","n6m2perms","n6m3perms","n6m4perms",
        "n7m1perms","n7m2perms","n7m3perms",
        "ids_n5g0", "ids_n5g1", "ids_n5g2", "ids_n5g3", "ids_n5g4", "ids_n5g5", "ids_n5g6", "ids_n5g7", "ids_n5g8", "ids_n5g9", 
        "ids_n5g10", "ids_n5g11", "ids_n5g12", "ids_n5g13", "ids_n5g14", "ids_n5g15", "ids_n5g16", "ids_n5g17", "ids_n5g18", "ids_n5g19", 
        "ids_n5g20", "ids_n5g21", "ids_n5g22", "ids_n5g23", "ids_n5g24", "ids_n5g25", "ids_n5g26", "ids_n5g27", "ids_n5g28", "ids_n5g29", 
        "ids_n5g30", "ids_n5g31", "ids_n5g32", "ids_n5g33",
        "ids_n6g0", "ids_n6g1", "ids_n6g2", "ids_n6g3", "ids_n6g4", "ids_n6g5", "ids_n6g6", "ids_n6g7", "ids_n6g8", "ids_n6g9", 
        "ids_n6g10", "ids_n6g11", "ids_n6g12", "ids_n6g13", "ids_n6g14", "ids_n6g15", "ids_n6g16", "ids_n6g17", "ids_n6g18", "ids_n6g19", 
        "ids_n6g20", "ids_n6g21", "ids_n6g22", "ids_n6g23", "ids_n6g24", "ids_n6g25", "ids_n6g26", "ids_n6g27", "ids_n6g28", "ids_n6g29", 
        "ids_n6g30", "ids_n6g31", "ids_n6g32", "ids_n6g33",
        "ids_n7g0", "ids_n7g1", "ids_n7g2", "ids_n7g3", "ids_n7g4", "ids_n7g5", "ids_n7g6", "ids_n7g7", "ids_n7g8", "ids_n7g9", 
        "ids_n7g10", "ids_n7g11", "ids_n7g12", "ids_n7g13", "ids_n7g14", "ids_n7g15", "ids_n7g16", "ids_n7g17", "ids_n7g18", "ids_n7g19", 
        "ids_n7g20", "ids_n7g21", "ids_n7g22", "ids_n7g23", "ids_n7g24", "ids_n7g25", "ids_n7g26", "ids_n7g27", "ids_n7g28", "ids_n7g29", 
        "ids_n7g30", "ids_n7g31", "ids_n7g32", "ids_n7g33",
        "ids_n16g0", "ids_n16g1", "ids_n16g2", "ids_n16g3", "ids_n16g4", "ids_n16g5", "ids_n16g6", "ids_n16g7", "ids_n16g8", "ids_n16g9", 
        "ids_n16g10", "ids_n16g11", "ids_n16g12", "ids_n16g13", "ids_n16g14", "ids_n16g15", "ids_n16g16", "ids_n16g17", "ids_n16g18", "ids_n16g19", 
        "ids_n16g20", "ids_n16g21", "ids_n16g22", "ids_n16g23", "ids_n16g24", "ids_n16g25", "ids_n16g26", "ids_n16g27", "ids_n16g28", "ids_n16g29", 
        "ids_n16g30", "ids_n16g31", "ids_n16g32", "ids_n16g33",
    ];

    for name in db_names.iter() {
        match env.open_db(Some(name)) {
            Ok(db) => { dbs.insert(name.to_string(), db); }
            Err(lmdb::Error::NotFound) => continue,
            Err(e) => panic!("Failed to open LMDB database {}: {:?}", name, e),
        }
    }

    dbs
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

pub fn main_butterfly_big(c: &CircuitSeq, rounds: usize, conn: &mut Connection, n: usize, asymmetric: bool, save: &str, env: &lmdb::Environment,) {
    // Start with the input circuit
    let bit_shuf_list = (3..=7)
        .map(|n| {
            (0..n)
                .permutations(n)
                .filter(|p| !p.iter().enumerate().all(|(i, &x)| i == x))
                .collect::<Vec<Vec<usize>>>()
        })
        .collect();
    let dbs = open_all_dbs(env);
    println!("Starting len: {}", c.gates.len());
    let mut circuit = c.clone();
    // Repeat obfuscate + compress 'rounds' times
    let mut post_len = 0;
    let mut count = 0;
    for i in 0..rounds {
        let stop = 1000;
        circuit = if asymmetric {
            // abutterfly_big(&circuit, conn, n, i != rounds-1, std::cmp::min(stop*(i+1), 5000), env, i+1, rounds, &bit_shuf_list, &dbs)
            abutterfly_big(&circuit, conn, n, i != rounds-1, 100, env, i+1, rounds, &bit_shuf_list, &dbs)
        } else {
            butterfly_big(&circuit,conn,n, i != rounds-1, stop*(i+1), env, &bit_shuf_list, &dbs)
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

pub fn main_rac_big(c: &CircuitSeq, rounds: usize, conn: &mut Connection, n: usize, save: &str, env: &lmdb::Environment, intermediate: &str) {
    // Start with the input circuit
    let save_base = save.strip_suffix(".txt").unwrap_or(save);
    let progress_path = format!("{}_progress.txt", save_base);
    let mut sum_already_coll = 0usize;
    let mut sum_shoot = 0usize;
    let mut sum_made_left = 0usize;
    let mut sum_traverse_left = 0usize;
    OpenOptions::new()
    .create(true)
    .write(true)
    .truncate(true)
    .open(&progress_path)
    .expect("Failed to create progress file");
    let bit_shuf_list = (3..=7)
        .map(|n| {
            (0..n)
                .permutations(n)
                .filter(|p| !p.iter().enumerate().all(|(i, &x)| i == x))
                .collect::<Vec<Vec<usize>>>()
        })
        .collect();
    let dbs = open_all_dbs(env);
    println!("Starting len: {}", c.gates.len());
    let mut circuit = c.clone();
    // Repeat obfuscate + compress 'rounds' times
    let mut post_len = 0;
    let mut count = 0;
    for i in 0..rounds {
        let _stop = 1000;
        let (new_circuit, already_coll, shoot, made_left, traverse_left)  = replace_and_compress_big(&circuit, conn, n, i != rounds-1, 100, env, i+1, rounds, &bit_shuf_list, &dbs, intermediate);
        circuit = new_circuit;

        sum_already_coll += already_coll;
        sum_shoot += shoot;
        sum_made_left += made_left;
        sum_traverse_left += traverse_left;

        let total_attempts = already_coll + shoot;
        let already_coll_pct = if total_attempts > 0 {
            already_coll as f64 / total_attempts as f64 * 100.0
        } else { 0.0 };
        let shoot_pct = if total_attempts > 0 {
            shoot as f64 / total_attempts as f64 * 100.0
        } else { 0.0 };
        let made_left_pct = if shoot > 0 {
            made_left as f64 / shoot as f64 * 100.0
        } else { 0.0 };
        let traverse_left_avg = if shoot > 0 {
            traverse_left as f64 / shoot as f64
        } else { 0.0 };

        println!(
            "Round {} stats: Total Attempts: {} | Already-collided {:.2}% | Shoot {:.2}% | Made-left {:.2}% | Traverse-left avg {:.2}",
            i + 1, total_attempts, already_coll_pct, shoot_pct, made_left_pct, traverse_left_avg
        );

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
        let mut j = 0;
        while j < circuit.gates.len().saturating_sub(1) {
            if circuit.gates[j] == circuit.gates[j + 1] {
                // remove elements at i and i+1
                circuit.gates.drain(j..=j + 1);

                // step back up to 2 indices, but not below 0
                j = j.saturating_sub(2);
            } else {
                j += 1;
            }
        }
        if c.probably_equal(&circuit, n, 100_000).is_err() {
            panic!("The functionality has changed");
        }
        {
        println!("Updating progress {}", progress_path);
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&progress_path)
            .expect("Failed to open progress file");

        writeln!(
            f,
            "=== Round {} ===\n{}\n",
            i + 1,
            circuit.repr()
        )
        .expect("Failed to write progress");
        }
    }

    let total_attempts = sum_already_coll + sum_shoot;
    let overall_already_coll_pct = if total_attempts > 0 {
        sum_already_coll as f64 / total_attempts as f64 * 100.0
    } else { 0.0 };
    let overall_shoot_pct = if total_attempts > 0 {
        sum_shoot as f64 / total_attempts as f64 * 100.0
    } else { 0.0 };
    let overall_made_left_pct = if sum_shoot > 0 {
        sum_made_left as f64 / sum_shoot as f64 * 100.0
    } else { 0.0 };
    let overall_traverse_left_avg = if sum_made_left > 0 {
        sum_traverse_left as f64 / sum_made_left as f64
    } else { 0.0 };

    println!("=== Overall Stats ===");
    println!(
        "Total Attempts {} \n Already-collided {:.2}% | Shoot {:.2}% | Made-left {:.2}% | Traverse-left avg {:.2}",
        total_attempts,
        overall_already_coll_pct,
        overall_shoot_pct,
        overall_made_left_pct,
        overall_traverse_left_avg
    );

    println!("Final len: {}", circuit.gates.len());
    circuit
    .probably_equal(&c, n, 150_000)
    .expect("The circuits differ somewhere!");

    // Write to file
    let circuit_str = circuit.repr();
    // let good_str = format!("{}: {}", good_id.gates.len(), good_id.repr());
    File::create(save)
        .and_then(|mut f| f.write_all(circuit_str.as_bytes()))
        .expect("Failed to write recent_circuit.txt");

    println!("Final circuit written to recent_circuit.txt");
}

pub fn main_rac_big_distance(c: &CircuitSeq, rounds: usize, conn: &mut Connection, n: usize, save: &str, env: &lmdb::Environment, intermediate: &str, min: usize,) {
    // Start with the input circuit
    let save_base = save.strip_suffix(".txt").unwrap_or(save);
    let progress_path = format!("{}_progress.txt", save_base);
    OpenOptions::new()
    .create(true)
    .write(true)
    .truncate(true)
    .open(&progress_path)
    .expect("Failed to create progress file");
    let bit_shuf_list = (3..=7)
        .map(|n| {
            (0..n)
                .permutations(n)
                .filter(|p| !p.iter().enumerate().all(|(i, &x)| i == x))
                .collect::<Vec<Vec<usize>>>()
        })
        .collect();
    let dbs = open_all_dbs(env);
    println!("Starting len: {}", c.gates.len());
    let mut circuit = c.clone();
    // Repeat obfuscate + compress 'rounds' times
    let mut post_len = 0;
    let mut count = 0;
    for i in 0..rounds {
        let _stop = 1000;
        let new_circuit = replace_and_compress_big_distance(&circuit, conn, n, i != rounds-1, 100, env, i+1, rounds, &bit_shuf_list, &dbs, intermediate, min);
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
        let mut j = 0;
        while j < circuit.gates.len().saturating_sub(1) {
            if circuit.gates[j] == circuit.gates[j + 1] {
                // remove elements at i and i+1
                circuit.gates.drain(j..=j + 1);

                // step back up to 2 indices, but not below 0
                j = j.saturating_sub(2);
            } else {
                j += 1;
            }
        }
        if c.probably_equal(&circuit, n, 100_000).is_err() {
            panic!("The functionality has changed");
        }
        {
        println!("Updating progress {}", progress_path);
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&progress_path)
            .expect("Failed to open progress file");

        writeln!(
            f,
            "=== Round {} ===\n{}\n",
            i + 1,
            circuit.repr()
        )
        .expect("Failed to write progress");
        }
    }

    println!("Final len: {}", circuit.gates.len());
    circuit
    .probably_equal(&c, n, 150_000)
    .expect("The circuits differ somewhere!");

    // Write to file
    let circuit_str = circuit.repr();
    // let good_str = format!("{}: {}", good_id.gates.len(), good_id.repr());
    File::create(save)
        .and_then(|mut f| f.write_all(circuit_str.as_bytes()))
        .expect("Failed to write recent_circuit.txt");

    println!("Final circuit written to recent_circuit.txt");
}

// pub fn main_butterfly_big_bookendsless(c: &CircuitSeq, rounds: usize, conn: &mut Connection, n: usize, _asymmetric: bool, save: &str ,env: &lmdb::Environment,) {
//     // Start with the input circuit
//     let dbs = open_all_dbs(env);
//     let bit_shuf_list = (3..=7)
//         .map(|n| {
//             (0..n)
//                 .permutations(n)
//                 .filter(|p| !p.iter().enumerate().all(|(i, &x)| i == x))
//                 .collect::<Vec<Vec<usize>>>()
//         })
//         .collect();
//     println!("Starting len: {}", c.gates.len());
//     let mut circuit = c.clone();
//     // Repeat obfuscate + compress 'rounds' times
//     let mut post_len = 0;
//     let mut count = 0;
//     let mut beginning = CircuitSeq { gates: Vec::new() };
//     let mut end= CircuitSeq { gates: Vec::new() };
//     for _ in 0..rounds {
//         let (new_circuit, b, e) = abutterfly_big_delay_bookends(&circuit, conn, n, env);
//         beginning = beginning.concat(&b);
//         end = e.concat(&end);
//         circuit = new_circuit;
//         if circuit.gates.len() == 0 {
//             break;
//         }
        
//         if circuit.gates.len() == post_len {
//             count += 1;
//         } else {
//             post_len = circuit.gates.len();
//             count = 0;
//         }

//         if count > 2 {
//             break;
//         }
//         let mut i = 0;
//         while i < circuit.gates.len().saturating_sub(1) {
//             if circuit.gates[i] == circuit.gates[i + 1] {
//                 // remove elements at i and i+1
//                 circuit.gates.drain(i..=i + 1);

//                 // step back up to 2 indices, but not below 0
//                 i = i.saturating_sub(2);
//             } else {
//                 i += 1;
//             }
//         }
//     }
//     let txn = env.begin_ro_txn().expect("txn");
//     println!("Adding bookends");
//     beginning = compress_big(&beginning, 100, n, conn, env, &bit_shuf_list, &dbs, &txn);
//     end = compress_big(&end, 100, n, conn, env, &bit_shuf_list, &dbs, &txn);
//     circuit = beginning.concat(&circuit).concat(&end);
//     let mut c1 = CircuitSeq{ gates: circuit.gates[0..circuit.gates.len()/2].to_vec() };
//     let mut c2 = CircuitSeq{ gates: circuit.gates[circuit.gates.len()/2..].to_vec() };
//     c1 = compress_big(&c1, 1_000, n, conn, env, &bit_shuf_list, &dbs, &txn);
//     c2 = compress_big(&c2, 1_000, n, conn, env, &bit_shuf_list, &dbs, &txn);
//     circuit = c1.concat(&c2);
//     let mut stable_count = 0;
//     while stable_count < 3 {
//         let before = circuit.gates.len();
//         //shoot_random_gate(&mut acc, 100_000);
//         circuit = compress_big(&circuit, 1_000, n, conn, env, &bit_shuf_list, &dbs ,&txn);
//         let after = circuit.gates.len();

//         if after == before {
//             stable_count += 1;
//             println!("  Final compression stable {}/3 at {} gates", stable_count, after);
//         } else {
//             println!("  Final compression reduced: {} → {} gates", before, after);
//             stable_count = 0;
//         }
//     }

//     println!("Final len: {}", circuit.gates.len());

//     circuit
//     .probably_equal(&c, n, 150_000)
//     .expect("The circuits differ somewhere!");

//     // Write to file
//     let c_str = c.repr();
//     let circuit_str = circuit.repr();
//     let long_str = format!("{}:{}", c.repr(), circuit.repr());
//     // let good_str = format!("{}: {}", good_id.gates.len(), good_id.repr());
//     // Write start.txt
//     File::create("start.txt")
//         .and_then(|mut f| f.write_all(c_str.as_bytes()))
//         .expect("Failed to write start.txt");

//     // Write recent_circuit.txt
//     File::create("recent_circuit.txt")
//         .and_then(|mut f| f.write_all(circuit_str.as_bytes()))
//         .expect("Failed to write recent_circuit.txt");

//     File::create(save)
//         .and_then(|mut f| f.write_all(circuit_str.as_bytes()))
//         .expect("Failed to write recent_circuit.txt");

//     // Write butterfly_recent.txt (overwrite)
//     File::create("butterfly_recent.txt")
//         .and_then(|mut f| f.write_all(long_str.as_bytes()))
//         .expect("Failed to write butterfly_recent.txt");

//     // Append to butterfly.txt
//     OpenOptions::new()
//         .append(true)
//         .create(true)
//         .open("butterfly.txt")
//         .and_then(|mut f| writeln!(f, "{}", long_str))
//         .expect("Failed to append to butterfly.txt");
//     if circuit.gates == c.gates {
//         println!("The obfuscation didn't do anything");
//     }

//     println!("Final circuit written to recent_circuit.txt");
// }

//do targeted compression
pub fn main_compression(c: &CircuitSeq, rounds: usize, conn: &mut Connection, n: usize, save: &str, env: &lmdb::Environment,) {
    let dbs = open_all_dbs(env);
    // Start with the input circuit
    let bit_shuf_list = (3..=7)
        .map(|n| {
            (0..n)
                .permutations(n)
                .filter(|p| !p.iter().enumerate().all(|(i, &x)| i == x))
                .collect::<Vec<Vec<usize>>>()
        })
        .collect();
    println!("Starting len: {}", c.gates.len());
    let mut circuit = c.clone();
    // Repeat obfuscate + compress 'rounds' times
    let mut post_len = 0;
    let mut count = 0;
    for _ in 0..rounds {
            butterfly_big(&circuit,conn,n, false, 0, env, &bit_shuf_list, &dbs);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_into_random_chunks() {

        let mut rng = rand::rng();
        let min_size = 100;
        let n = 250;
        let k = 60;

        // Build v = 4 concatenated copies of 0..999
        let mut v: Vec<[u8; 3]> = (0..n)
            .map(|i| [i as u8, i as u8, i as u8])
            .collect();
        for _ in 0..100 { // append 3 more copies
            v.extend((0..n).map(|i| [i as u8, i as u8, i as u8]));
        }

        let chunks = split_into_random_chunks(
            &v,
            k,
            &mut rng,
        );

        assert_eq!(chunks.len(), k, "Expected {} chunks, got {}", k, chunks.len());

        for (i, chunk) in chunks.iter().enumerate() {
            assert!(
                chunk.len() >= min_size,
                "Chunk {} is smaller than min_size: {}",
                i,
                chunk.len()
            );
            println!("{}", chunk.len());
        }

        let total_len: usize = chunks.iter().map(|c| c.len()).sum();
        assert_eq!(total_len, v.len(), "Total elements mismatch");

        // Flatten without sorting
        let all_elements: Vec<[u8; 3]> = chunks.into_iter().flatten().collect();

        // Check that the flattened elements are exactly equal to the original v
        for (i, &elem) in all_elements.iter().enumerate() {
            assert_eq!(elem, v[i], "Element mismatch at index {}", i % 250);
        }
    }
}