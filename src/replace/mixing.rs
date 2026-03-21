use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io::{self, Read, Write},
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc,
        Mutex,
    },
    time::Instant,
};

use once_cell::sync::Lazy;
use rand::{prelude::SliceRandom, Rng};
use rayon::prelude::*;
use rusqlite::{Connection, OpenFlags};

use crate::{
    circuit::circuit::CircuitSeq,
    random::random_data::{
        random_circuit, shoot_left_vec, shoot_left_vec_track, shoot_random_gate, shoot_right_vec, shoot_right_vec_track
    },
    replace::{
        identities::{create_escalator_identities, create_escalator_identities_tracked, get_random_shuffled_identity, insert_ri_identities, random_id}, pairs::{
            interleave,
            replace_pair_distances_linear,
            replace_pairs,
            replace_sequential_pairs,
            replace_single_pair,
        }, replace::{
            compress,
            compress_big,
            compress_big_ancillas,
            expand_big,
            obfuscate,
            outward_compress,
        }, 
    },
};

// Old legacy method of replace -> compress
pub fn obfuscate_and_target_compress(c: &CircuitSeq, conn: &mut Connection, bit_shuf: &Vec<Vec<usize>>, n: usize) -> CircuitSeq {
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

// Find how many gates are on each wire
pub fn pin_counts(circuit: &CircuitSeq, num_wires: usize) -> Vec<usize> {
    let mut counts = vec![0; num_wires];
    for gate in &circuit.gates {
        counts[gate[0] as usize] += 1;
        counts[gate[1] as usize] += 1;
        counts[gate[2] as usize] += 1;
    }
    counts
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Butterfly Methods 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Butterfly method on low number of wires
pub fn butterfly(
    c: &CircuitSeq,
    conn: &mut Connection,
    bit_shuf: &Vec<Vec<usize>>,
    n: usize,
) -> CircuitSeq {
    // Pick one random R
    let mut rng = rand::rng();
    let (r, r_inv) = random_id(n, rng.random_range(3..=25)); 

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

// Merges blocks and compresses them along to way to "mix the seams"
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

// Butterfly method for more wires
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
    let (r, r_inv) = random_id(n, rng.random_range(100..=200)); 
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

// Timing variables for benchmarking
pub static SHOOT_RANDOM_GATE_TIME: Lazy<AtomicU64> = Lazy::new(|| AtomicU64::new(0));
pub static REPLACE_PAIRS_TIME: Lazy<AtomicU64> = Lazy::new(|| AtomicU64::new(0));
pub static RANDOM_ID_TIME: Lazy<AtomicU64> = Lazy::new(|| AtomicU64::new(0));
pub static EXPAND_BIG_TIME: Lazy<AtomicU64> = Lazy::new(|| AtomicU64::new(0));
pub static COMPRESS_BIG_TIME: Lazy<AtomicU64> = Lazy::new(|| AtomicU64::new(0));
pub static MERGE_COMBINE_BLOCKS_TIME: Lazy<AtomicU64> = Lazy::new(|| AtomicU64::new(0));


static CURRENT_ACC: Lazy<Mutex<Option<CircuitSeq>>> =
    Lazy::new(|| Mutex::new(None));

// Help with early stops without losing all data
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

// Asymmetric butterfly method on more wires
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
    let (first_r, first_r_inv) = random_id(n, rng.random_range(10..=30));
    let mut prev_r_inv = first_r_inv.clone();
    let t1 = Instant::now();
    replace_pairs(&mut c, n, _conn, &env);
    REPLACE_PAIRS_TIME.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);

    let mut pre_blocks: Vec<CircuitSeq> = Vec::with_capacity(c.gates.len());

    for &g in &c.gates {
        let t2 = Instant::now();
        let (r, r_inv) = random_id(n, rng.random_range(10..=30));
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

pub fn make_steps(n: usize, gate: &[u8; 3]) -> Vec<Vec<usize>> {
    let mut rng = rand::rng();

    let gate: Vec<usize> = gate.iter().map(|&x| x as usize).collect();

    let mut remaining: Vec<usize> = (0..n)
        .filter(|x| !gate.contains(x))
        .collect();

    remaining.shuffle(&mut rng);

    let mut steps: Vec<Vec<usize>> = Vec::new();

    let mut i = 0;
    let rem = remaining.len() % 3;

    let mut small_steps: Vec<Vec<usize>> = Vec::new();
    let mut large_steps: Vec<Vec<usize>> = Vec::new();

    // size 2 steps
    if rem == 1 && remaining.len() >= 4 {
        let chunk = &remaining[i..i+4];
        small_steps.push(vec![chunk[0], chunk[1]]);
        small_steps.push(vec![chunk[2], chunk[3]]);
        i += 4;
    } else if rem == 2 {
        let chunk = &remaining[i..i+2];
        small_steps.push(vec![chunk[0], chunk[1]]);
        i += 2;
    }

    // Normal rise steps
    while i < remaining.len() {
        large_steps.push(vec![
            remaining[i],
            remaining[i + 1],
            remaining[i + 2],
        ]);
        i += 3;
    }

    // Insert fixed into 3 rise steps
    large_steps.push(gate);

    // Shuffle
    large_steps.shuffle(&mut rng);
    small_steps.shuffle(&mut rng);

    let mut small_count = 0;
    let small_len = small_steps.len();
    let total = large_steps.len() + small_steps.len();
    let mut positions: Vec<usize> = (1..total-1).collect();
    positions.shuffle(&mut rng);
    positions = vec![positions[0], positions[1]];
    positions.sort();
    for i in 0..total {
        if i == positions[small_count] && small_count < small_len {
            steps.push(small_steps[small_count].clone());
            small_count += 1;
        } else {
            steps.push(large_steps[i - small_count].clone());
        }
    }

    // Ensure ends must be size 3
    assert!(steps.first().unwrap().len() == 3);
    assert!(steps.last().unwrap().len() == 3);

    steps
}

pub fn sequential_butterfly(
    c: &CircuitSeq,
    _conn: &mut Connection,
    n: usize,
    env: &lmdb::Environment,
    curr_round: usize,
    last_round: usize,
    bit_shuf_list: &Vec<Vec<Vec<usize>>>,
    dbs: &HashMap<String, lmdb::Database>,
    id_len: usize,
    reverse_order_left: bool,
    tower_left: bool,
    shoot_more_left: u8,
    reverse_order_right: bool,
    tower_right: bool,
    shoot_more_right: u8
) -> CircuitSeq {
    println!("  {}/{}: Beginning sequential butterfly: {} gates", curr_round, last_round, c.gates.len());
    let mut rng = rand::rng();
    let mut circuit = c.clone();
    let diehard_len = if n == 32 {
        525
    } else if n == 64 {
        1200
    } else if n == 128 {
        2500
    } else {
        2500
    };
    println!("  {}/{}: Step 1: Shooting", curr_round, last_round);
    let mut len = circuit.gates.len();
    shoot_random_gate(&mut circuit, len * 50);
    println!("   {}/{}: Step 1a: Inserting Identities", curr_round, last_round);
    insert_ri_identities(&mut circuit, &env, &dbs);
    if circuit.probably_equal(&c, n, 1000).is_err() {
        panic!("Inserting identities failed");
    }
    // Keeps track of the gates and whether it can stay in place `0`, needs to be shot left `1`, or needs to be shot right `2`
    // After, `4` means to collide to the left and `5` to collide to the right
    let mut gates_track: Vec<([u8;3], u8)> = Vec::new();
    gates_track.push((circuit.gates[0], 0));
    let mut last_steps: Vec<Vec<usize>> = Vec::new();
    for i in 1..circuit.gates.len() {
        // let random_len = rng.random_range(diehard_len-100..diehard_len+100);
        // let random_circuit = random_circuit(n, random_len);
        // for ri in 0..random_circuit.gates.len() {
        //     gates_track.push((random_circuit.gates[ri], 1));
        // }
        // for ri in (0..random_circuit.gates.len()).rev() {
        //     gates_track.push((random_circuit.gates[ri], 2));
        // }
        // let random_id = get_random_shuffled_identity(n, env, dbs, _conn, bit_shuf_list, tower_left);
        let first_steps = if last_steps.is_empty() {
            make_steps(n, &circuit.gates[0])
        } else {
            last_steps.clone()
        };

        let curr_gate = circuit.gates[i];
        let second_steps = make_steps(n, &curr_gate);
        let (gt, _, _, _, _) = create_escalator_identities_tracked(n, &first_steps, &second_steps, env, dbs);
        last_steps = second_steps;
        // let id = CircuitSeq { gates: Vec::new() };
        // if random_id.probably_equal(&id, n, 1000).is_err(){
        //     panic!("Not an identity");
        // }
        // for ri in 0..random_id.gates.len()/2 {
        //     gates_track.push((random_id.gates[ri], 1));
        // }
        // for ri in random_id.gates.len()/2..random_id.gates.len() {
        //     gates_track.push((random_id.gates[ri], 2));
        // }
        gates_track.extend_from_slice(&gt);
        gates_track.push((circuit.gates[i], 0));
    }

    // for i in 0..circuit.gates.len() {
    //     if i%100 < 35 {
    //         gates_track.push((circuit.gates[i], 1));
    //     } else if i%100 > 65 {
    //         gates_track.push((circuit.gates[i], 2));
    //     } else {
    //         gates_track.push((circuit.gates[i], 0));
    //     }
    // }
    len = gates_track.len();
    println!("  {}/{}: Step 2a: Shooting R's left", curr_round, last_round);
    println!("  {}/{}: Mode: \n Additional rounds: {} \n Forward Order: {}", curr_round, last_round, shoot_more_left, reverse_order_left == false);
    for _ in 0..=shoot_more_left {
        if !reverse_order_left {
            for i in 0..len {
                if gates_track[i].1 == 1 {
                    shoot_left_vec_track(&mut gates_track, i, false);
                    // println!("Shoot left");
                }
            }
        } else {
            let mut i = len - 1;
            while i > 0 {
                if gates_track[i].1 == 1 {
                    let after_idx = shoot_left_vec_track(&mut gates_track, i, false);
                    // println!("Shoot left");
                    if after_idx != i {
                        continue;
                    } else {
                        i -= 1;
                    }
                } else {
                    i -= 1;
                }
            }
        }
    }
    println!("  {}/{}: Step 2b: Shooting R*'s right", curr_round, last_round);
    println!("  {}/{}: Mode: \n Additional rounds: {} \n Reverse Order: {}", curr_round, last_round, shoot_more_right, reverse_order_right == true);
    for _ in 0..=shoot_more_right {
        if reverse_order_right {
            for i in (0..len).rev() {
                if gates_track[i].1 == 2 {
                    shoot_right_vec_track(&mut gates_track, i, false);
                    // println!("Shoot right");
                }
            }
        } else {
            let mut i = 0;
            while i < len {
                if gates_track[i].1 == 2 {
                    let after_idx = shoot_right_vec_track(&mut gates_track, i, false);
                    // println!("Shoot right");
                    if after_idx != i {
                        continue;
                    } else {
                        i += 1;
                    }
                } else {
                    i += 1;
                }
            }
        }
    }
    // println!("{:?}", gates_track);
    println!("  {}/{}: Step 3a: Finding collisions for R", curr_round, last_round);
    println!("  {}/{}: Mode: \n Forward Order: {} \n Identity wire lengths: {} \n Tower identities: {}", curr_round, last_round, reverse_order_left == false, id_len, tower_left == true);

    if !reverse_order_left {
        for i in 0..len {
            if gates_track[i].1 == 4 {
                let after_idx = shoot_left_vec_track(&mut gates_track, i, true);
                if after_idx != 0 {
                    let (pair_repl, _) = replace_single_pair(
                         &gates_track[after_idx - 1].0, 
                        &gates_track[after_idx].0, 
                        n, 
                        _conn, 
                        env, 
                        bit_shuf_list, 
                        dbs, 
                        tower_left,
                        id_len,
                    );
                    let pair_repl = 
                    pair_repl
                    .into_iter()
                    .map(|v| (v, 0))
                    .collect::<Vec<_>>();
                    gates_track.splice(after_idx - 1..=after_idx, pair_repl);
                    // println!("Replacement made left");
                }
            }
        }
    } else {
        let mut i = len - 1;
        while i > 0 {
            if gates_track[i].1 == 4 {
                let after_idx = shoot_left_vec_track(&mut gates_track, i, true);
                if after_idx != 0 {
                    let (pair_repl, _) = replace_single_pair(
                         &gates_track[after_idx - 1].0, 
                        &gates_track[after_idx].0, 
                        n, 
                        _conn, 
                        env, 
                        bit_shuf_list, 
                        dbs, 
                        tower_left,
                        id_len,
                    );
                    let pair_repl = 
                    pair_repl
                    .into_iter()
                    .map(|v| (v, 0))
                    .collect::<Vec<_>>();
                    gates_track.splice(after_idx - 1..=after_idx, pair_repl);
                    // println!("Replacement made left");
                }
                if after_idx != i {
                    continue;
                } else {
                    i -= 1;
                }
            } else {
                i -= 1;
            }
        }
    }

    println!("  {}/{}: Step 3b: Finding collisions for R*", curr_round, last_round);
    println!("  {}/{}: Mode: \n Reverse Order: {} \n Identity wire lengths: {} \n Tower identities: {}", curr_round, last_round, reverse_order_right == true, id_len, tower_right == true);

    if reverse_order_right {
        for i in (0..len).rev() {
            if gates_track[i].1 == 5 {
                let curr_len = gates_track.len();
                let after_idx = shoot_right_vec_track(&mut gates_track, i, false);
                if after_idx != curr_len {
                    let (pair_repl, _) = replace_single_pair(
                        &gates_track[after_idx].0, 
                        &gates_track[after_idx + 1].0, 
                        n, 
                        _conn, 
                        env, 
                        bit_shuf_list, 
                        dbs, 
                        tower_left,
                        id_len
                    );
                    let pair_repl = 
                    pair_repl
                    .into_iter()
                    .map(|v| (v, 0))
                    .collect::<Vec<_>>();
                    gates_track.splice(after_idx..=after_idx+1, pair_repl);
                    // println!("Replacement made right");
                }
            }
        }
    } else {
        let mut i = 0;
        while i < len {
            if gates_track[i].1 == 5 {
                let after_idx = shoot_right_vec_track(&mut gates_track, i, false);
                let curr_len = gates_track.len();
                if after_idx != curr_len {
                    let (pair_repl, _) = replace_single_pair(
                        &gates_track[after_idx].0, 
                        &gates_track[after_idx + 1].0, 
                        n, 
                        _conn, 
                        env, 
                        bit_shuf_list, 
                        dbs, 
                        tower_left,
                        id_len
                    );
                    let pair_repl = 
                    pair_repl
                    .into_iter()
                    .map(|v| (v, 0))
                    .collect::<Vec<_>>();
                    gates_track.splice(after_idx..=after_idx+1, pair_repl);
                    // println!("Replacement made right");
                }
                if after_idx != i {
                    continue;
                } else {
                    i += 1;
                }
            } else {
                i += 1;
            }
        }
    }

    circuit.gates = gates_track
        .into_iter()
        .map(|(v, _)| v)
        .collect::<Vec<_>>();

    
    println!("  {}/{}: Sending to compressor: {} gates", curr_round, last_round, circuit.gates.len());
    if circuit.probably_equal(&c, n, 10000).is_err() {
        panic!("Functionality lost during sequential butterfly");
    }

    let mut acc = circuit;
    let mut stable_count = 0;
    while stable_count < 12 {
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
                
                compress_big_ancillas(&sub, 100, n, &mut thread_conn, env, &bit_shuf_list, dbs).gates
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
        if after == before {
            stable_count += 1;
        } else {
            stable_count = 0;
        }

        let mut buf = [0u8; 1];
        if let Ok(n) = io::stdin().read(&mut buf) {
            if n > 0 && buf[0] == b'\n' {
                println!("  {}/{}: Current gates: {} gates", curr_round, last_round, after);
            }
        }
    }
    acc
}

// Simple shooting game method. Send a gate to the right until a collision is made, then make a replacement. Continue the same from the right-most gate
// Repeat until the blowup is `enough` and then compress
pub fn simple_shooting_game(
    c: &CircuitSeq,
    _conn: &mut Connection,
    n: usize,
    env: &lmdb::Environment,
    curr_round: usize,
    last_round: usize,
    bit_shuf_list: &Vec<Vec<Vec<usize>>>,
    dbs: &HashMap<String, lmdb::Database>,
    id_len: usize,
    tower: bool,
    stop: usize,
    intermediate: &str,
) -> CircuitSeq {
    let mut gates = c.gates.clone();
    println!("   {}/{}: Starting simple shooting game until {} gates", curr_round, last_round, stop);
    let mut len = gates.len();
    println!("Starting gates: {}", len);
    let mut rng = rand::rng();
    while len < stop {
        let left = rng.random_bool(0.5);
        let starting_idx = rng.random_range(0..len);
        if left {
            let mut curr_idx = starting_idx;
            while curr_idx != 0 {
                let after_idx = shoot_left_vec(&mut gates, curr_idx);
                if after_idx == 0 {
                    break
                }
                let (id, length) = replace_single_pair(
                    &gates[after_idx - 1], 
                    &gates[after_idx], 
                    n, 
                    _conn, 
                    env, 
                    bit_shuf_list, 
                    dbs, 
                    tower, 
                    id_len
                );
                gates.splice(after_idx-1..after_idx+1, id);
                len += length - 2;
                curr_idx = after_idx-1;
            }
        } else {
            let mut curr_idx = starting_idx;
            while curr_idx != len {
                let after_idx = shoot_right_vec(&mut gates, curr_idx);
                if after_idx == len - 1 {
                    break
                }
                let (id, length) = replace_single_pair(
                    &gates[after_idx], 
                    &gates[after_idx + 1], 
                    n, 
                    _conn, 
                    env, 
                    bit_shuf_list, 
                    dbs, 
                    tower, 
                    id_len
                );
                gates.splice(after_idx..after_idx+2, id);
                len += length - 2;
                curr_idx = after_idx + length - 1;
            }
        }
    }

    let mut acc = CircuitSeq { gates };
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(intermediate)
        .expect("Failed to open replacednocomp.txt");
    println!("Writing to {}", intermediate);
    writeln!(f, "{}", acc.repr()).expect("Failed to write intermediate CircuitSeq");

    println!("  {}/{}: Sending to compressor: {} gates", curr_round, last_round, acc.gates.len());
    
    if acc.probably_equal(&c, n, 10000).is_err() {
        panic!("Functionality lost during sequential butterfly");
    }
    let mut stable_count = 0;
    while stable_count < 12 {
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
                
                compress_big_ancillas(&sub, 100, n, &mut thread_conn, env, &bit_shuf_list, dbs).gates
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
        if after == before {
            stable_count += 1;
        } else {
            stable_count = 0;
        }

        let mut buf = [0u8; 1];
        if let Ok(n) = io::stdin().read(&mut buf) {
            if n > 0 && buf[0] == b'\n' {
                println!("  {}/{}: Current gates: {} gates", curr_round, last_round, after);
            }
        }
    }
    acc
}
// Asymmetric butterfly but delay compression and addition of the bookends
// Hope is to slow down the blowup in the number of gates
// Currently unsupported
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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Pair Replacmeent Methods
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Some legacy statistics for the types of pairs we are replacing
static ALREADY_COLLIDED: AtomicUsize = AtomicUsize::new(0);
static SHOOT_COUNT: AtomicUsize = AtomicUsize::new(0);
static MADE_LEFT: AtomicUsize = AtomicUsize::new(0);
static TRAVERSE_LEFT: AtomicUsize = AtomicUsize::new(0);

// RCS
// Original version would attempt to choose pairs that collided
// New version does not care for the type of pairs it is replacing
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
    tower: bool,
    id_len: usize
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
    for _ in 0..1 {
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
        // For parallelization
        // The seams will remain unmixed, so need to address this later
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
                // Only do a forward sequential pass. A reverse pass afterwards could be useful
                let (col, shoot, zero, trav) = replace_sequential_pairs(&mut sub, n, &mut thread_conn, &env, &bit_shuf_list, dbs, tower);
                ALREADY_COLLIDED.fetch_add(col, Ordering::SeqCst);
                SHOOT_COUNT.fetch_add(shoot, Ordering::SeqCst);
                MADE_LEFT.fetch_add(zero, Ordering::SeqCst);
                TRAVERSE_LEFT.fetch_add(trav, Ordering::SeqCst);
                shoot_random_gate(&mut sub, 200_000);
                sub.gates
            })
            .collect();
        let new_gates = mix_seams(replaced_chunks, _conn, n, env, bit_shuf_list, dbs, tower, id_len);
        c.gates = new_gates;
    }
    REPLACE_PAIRS_TIME.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);
    println!(
        "Finished replace_sequential_pairs, new length: {}",
        c.gates.len()
    );
    // Sanity check
    if c.probably_equal(circuit, n, 100).is_err() {
        panic!("replacing changed functionality");
    }
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
                
                compress_big_ancillas(&sub, 100, n, &mut thread_conn, env, &bit_shuf_list, dbs).gates
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

// To address the problem of seams being unmixed after doing rcs with parallelization
// Returns [..chunk.len() - 1][replace_pair(last, next)][1..chunk.len()-1][replace_pair(last, next)]...[1..]
pub fn mix_seams(
    gates: Vec<Vec<[u8;3]>>,
    _conn: &mut Connection,
    n: usize,
    env: &lmdb::Environment,
    bit_shuf_list: &Vec<Vec<Vec<usize>>>,
    dbs: &HashMap<String, lmdb::Database>,
    tower: bool,
    id_len: usize,
) -> Vec<[u8;3]> {
    if gates.len() == 1 {
        return gates.into_iter().flatten().collect()
    }
    let mut new_gates: Vec<[u8; 3]> = Vec::new();
    let len = gates.len();
    for i in 0..len - 1 {
        let chunk = &gates[i];
        let next  = &gates[i + 1];
        if i == 0 {
            new_gates.extend_from_slice(&chunk[..chunk.len() - 1]);
        } else {
            new_gates.extend_from_slice(&chunk[1..chunk.len() - 1]);
        }

        let left  = chunk.last().unwrap();
        let right = next.first().unwrap();

        let (replaced, _) = replace_single_pair(
            left,
            right,
            n,
            _conn,
            &env,
            &bit_shuf_list,
            dbs,
            tower,
            id_len,
        );
        let lr = CircuitSeq { gates: vec![*left, *right] };
        if lr.probably_equal(&CircuitSeq { gates: replaced.clone() }, n, 1_000).is_err() {
            panic!("Replaced doesn't match lr");
        }
        new_gates.extend_from_slice(&replaced);
    }

    let last = gates.last().unwrap();
    new_gates.extend_from_slice(&last[1..]);
    let temp: Vec<[u8;3]> = gates.into_iter().flatten().collect();
    let c1 = CircuitSeq { gates: temp };
    let c2 = CircuitSeq { gates: new_gates.clone() };
    if c1.probably_equal(&c2, n, 1_000).is_err() {
        panic!("Failed to mix seams");
    }
    new_gates
}

// Interleaving method
// Same as RCS, but start by interleaving a circuit on n..2n wires, rather than 0..n, and then interleaving them like a deck of cards
// Every pair is going to be non colliding, and in fact, will not touch at all
pub fn interleave_sequential_big(
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
    tower: bool,
    id_len: usize,
) -> (CircuitSeq, usize, usize, usize, usize) {
    println!("Current round: {}/{}", curr_round, last_round);

    println!("Butterfly start: {} gates", circuit.gates.len());
    let mut c = interleave(circuit, n);
    let t0 = Instant::now();
    let n = 2 * n;
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
                let (col, shoot, zero, trav) = replace_sequential_pairs(&mut sub, n, &mut thread_conn, &env, &bit_shuf_list, dbs, tower);
                ALREADY_COLLIDED.fetch_add(col, Ordering::SeqCst);
                SHOOT_COUNT.fetch_add(shoot, Ordering::SeqCst);
                MADE_LEFT.fetch_add(zero, Ordering::SeqCst);
                TRAVERSE_LEFT.fetch_add(trav, Ordering::SeqCst);
                shoot_random_gate(&mut sub, 200_000);
                sub.gates
            })
            .collect();
        let mut new_gates: Vec<[u8; 3]> = Vec::new();
        let len = replaced_chunks.len();

        for i in 0..len - 1 {
            let chunk = &replaced_chunks[i];
            let next  = &replaced_chunks[i + 1];

            if i == 0 {
                new_gates.extend_from_slice(&chunk[..chunk.len() - 1]);
            } else {
                new_gates.extend_from_slice(&chunk[1..chunk.len() - 1]);
            }

            let left  = chunk.last().unwrap();
            let right = next.first().unwrap();

            let (replaced, _) = replace_single_pair(
                left,
                right,
                n,
                _conn,
                &env,
                &bit_shuf_list,
                dbs,
                tower,
                id_len,
            );

            new_gates.extend_from_slice(&replaced);
        }

        let last = replaced_chunks.last().unwrap();
        new_gates.extend_from_slice(&last[1..]);
        c.gates = new_gates;
        c.gates.reverse();
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

    // Final global compression until stable 6×
    println!("Beginning compression");
    let mut acc = c;
    let mut rng = rand::rng();
    let mut stable_count = 0;
    while stable_count < 12 {
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
                
                compress_big_ancillas(&sub, 100, n, &mut thread_conn, env, &bit_shuf_list, dbs).gates
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
        0,
        0,
        0,
        0,
    )
}

// RCD method
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
    tower: bool,
    id_len: usize,
) -> CircuitSeq {
    println!("Current round: {}/{}", curr_round, last_round);
    println!("Replace and compress distance start: {} gates", circuit.gates.len());
    let mut c = circuit.clone();
    let t0 = Instant::now();
    shoot_random_gate(&mut c, 200_000);
    SHOOT_RANDOM_GATE_TIME.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

    let t1 = Instant::now();
    replace_pair_distances_linear(&mut c, n, _conn, env, bit_shuf_list, dbs, min, tower, id_len);
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
                
                compress_big_ancillas(&sub, 100, n, &mut thread_conn, env, &bit_shuf_list, dbs).gates
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

// For paralellization. Split a circuit into many random chunks for threads to each work on
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