use std::{
    fs::{File, OpenOptions},
    io::Write,
};

use crate::{
    circuit::circuit::CircuitSeq,
    replace::{
        gadgets::gadgetize,
        mixing::simple_shooting_game,
        pairs::interleave,
        replace::{ExpandPairMode, compress_loop, expand_once},
        transpositions::insert_wire_m_samfs_every_x,
    },
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Open all dbs ahead of time in the LMDB
// LMDB used for fast reads
// nXmY store the canonicalized (up to gate ordering and wire relabeling) version of all the circuits
// perms_tables_nX store a list of tables that share a permutation. Legacy use for building random identities
// nXmYperms stores all circuits canonicalized only up to gate ordering
// ids_nXgK stores identities on X wires with gate pair taxonomy K on the first two gates. See Taxonomies to_int to see
// Last row of tables is used for swapping wires, CNOTS, NOTS
pub fn open_curated_shard_dbs(env: &lmdb::Environment) -> Vec<lmdb::Database> {
    (0u16..=255)
        .map(|s| {
            let name = format!("curated_{:02x}", s);
            env.open_db(Some(name.as_str()))
                .unwrap_or_else(|e| panic!("Failed to open curated shard db {:02x}: {:?}", s, e))
        })
        .collect()
}

pub fn open_shard_dbs(env: &lmdb::Environment) -> Vec<lmdb::Database> {
    (0u16..=255)
        .map(|s| {
            let name = format!("{:02x}", s);
            env.open_db(Some(name.as_str()))
                .unwrap_or_else(|e| panic!("Failed to open shard db {:02x}: {:?}", s, e))
        })
        .collect()
}

pub fn open_all_dbs(env: &lmdb::Environment) -> (Vec<lmdb::Database>, Vec<lmdb::Database>) {
    let shard_dbs = open_shard_dbs(env);
    let curated_shard_dbs = open_curated_shard_dbs(env);
    (shard_dbs, curated_shard_dbs)
}

pub fn main_shuffle_shoot_shuffle(
    c: &CircuitSeq,
    rounds: usize,
    n: usize,
    m: usize,
    x: usize,
    save: &str,
    source: &str,
    env: &lmdb::Environment,
    shard_dbs: &[lmdb::Database],
    curated_shard_dbs: &[lmdb::Database],
    intermediate: &str,
    leave: bool,
    do_gadgetize: bool,
    full_shuffle: bool,
    gates_ahead: usize,
    egg: bool,
    rg_freq: usize,
    shuffled: bool,
    single_end: bool,
) {
    // Start with the input circuit
    let save_base = save.strip_suffix(".txt").unwrap_or(save);
    let progress_path = format!("{}_progress.txt", save_base);
    OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&progress_path)
        .expect("Failed to create progress file");
    println!("Starting len: {}", c.gates.len());
    let mut circuit = c.clone();
    // Repeat `rounds` times
    let mut post_len = 0;
    let mut count = 0;
    if do_gadgetize {
        let mut rng = rand::rng();
        let before = circuit.gates.len();
        circuit = gadgetize(&circuit, n, rg_freq, &mut rng);
        println!(
            "Gadgetized: {} gates → {} gates, {} wires",
            before,
            circuit.gates.len(),
            2 * n
        );
        // Save the gadgetized circuit to ./gadgetized/{final path component of source}
        let file_name = std::path::Path::new(source)
            .file_name()
            .expect("Source path has no final component")
            .to_str()
            .expect("Source file name is not valid UTF-8");
        std::fs::create_dir_all("./gadgetized").expect("Failed to create ./gadgetized");
        let gadget_path = format!("./gadgetized/{}", file_name);
        File::create(&gadget_path)
            .and_then(|mut f| f.write_all(circuit.repr().as_bytes()))
            .expect("Failed to write gadgetized circuit");
        println!("Gadgetized circuit written to {}", gadget_path);
    }
    let n = if do_gadgetize { 2 * n } else { n };
    if leave {
        circuit = interleave(&circuit, n, env);
    }
    let n = if leave { 2 * n } else { n };
    if full_shuffle {
        // SAMF insertion is equivalence-preserving by construction, so no retry guard.
        insert_wire_m_samfs_every_x(&mut circuit, n, n, 1, env, curated_shard_dbs, shard_dbs);
        println!("After full shuffle: {} gates", circuit.gates.len());
    }
    // --single-end accumulator (shuffled path only): SAMF state carried across ALL rounds,
    // undone in one pass after the last round. `total_t` is the composed wire permutation
    // (round order), `total_neg` the pending negation in the current wire space.
    let mut total_t = crate::replace::transpositions::Transpositions {
        transpositions: Vec::new(),
    };
    let mut total_neg = vec![0u8; n];

    // Per-round SAMF (shuffled shooting game) compression stats, snapshotted each round.
    let mut per_round_samf: Vec<(usize, usize)> = Vec::new();
    let mut prev_samf_made = crate::replace::transpositions::SAMF_COMPRESSIONS_MADE
        .load(std::sync::atomic::Ordering::Relaxed);
    let mut prev_samf_failed = crate::replace::transpositions::SAMF_COMPRESSIONS_FAILED
        .load(std::sync::atomic::Ordering::Relaxed);
    for i in 0..rounds {
        if egg {
            let pair_mode = ExpandPairMode::Curated {
                curated_shard_dbs: &curated_shard_dbs,
            };
            circuit = expand_once(&circuit, n, env, shard_dbs, &pair_mode);
        } else if shuffled && single_end {
            // Accumulate this round's SAMFs WITHOUT undoing — functionality is intentionally
            // broken between rounds; we undo everything once after the last round (below).
            use crate::replace::transpositions::shuffled_shoot_then_samf_core;
            let (out, t_round, neg_round, _c) = shuffled_shoot_then_samf_core(
                &circuit.gates,
                n,
                m,
                x,
                gates_ahead,
                env,
                curated_shard_dbs,
                shard_dbs,
            );
            circuit.gates = out;
            // Fold this round into the running accumulator: transport the existing pending
            // negation through this round's permutation, then add this round's negation;
            // compose the permutations (existing first, then this round).
            let mut new_total_neg = neg_round;
            for w in 0..n {
                if total_neg[w] == 1 {
                    let cw = t_round.evaluate(w as u16) as usize;
                    new_total_neg[cw] ^= 1;
                }
            }
            total_neg = new_total_neg;
            total_t = total_t.concat(&t_round);
        } else if shuffled {
            // Shooting game + per-gate SAMF insertion with a SINGLE merged unsamf.
            use crate::replace::transpositions::shuffled_shoot_then_samf;
            shuffled_shoot_then_samf(
                &mut circuit,
                n,
                m,
                x,
                gates_ahead,
                env,
                curated_shard_dbs,
                shard_dbs,
            );
        } else {
            circuit = simple_shooting_game(
                &circuit,
                n,
                env,
                i + 1,
                rounds,
                4 * circuit.gates.len(),
                intermediate,
                true,
                1,
                gates_ahead,
                &curated_shard_dbs,
                shard_dbs,
            );
        }
        println!("After shooting game: {} gates", circuit.gates.len());
        // The shuffled path already inserted SAMFs (merged into its single unsamf above).
        if !shuffled {
            insert_wire_m_samfs_every_x(&mut circuit, n, m, x, env, curated_shard_dbs, shard_dbs);
        }
        println!("After inserting samfs: {} gates", circuit.gates.len());
        // --single-end: after the FINAL round's shuffle, before its compression, undo ALL
        // accumulated SAMFs/NOTs in one pass — restoring equivalence to the original input.
        if shuffled && single_end && i == rounds - 1 {
            use crate::replace::transpositions::apply_unsamf;
            apply_unsamf(
                &mut circuit.gates,
                &total_t,
                &total_neg,
                n,
                env,
                curated_shard_dbs,
                shard_dbs,
            );
            println!("After single-end unsamf: {} gates", circuit.gates.len());
        }
        circuit = compress_loop(
            &circuit,
            n,
            env,
            shard_dbs,
            6,
            i + 1,
            rounds,
            "temp_compression.txt",
        );
        println!("After compression: {} gates", circuit.gates.len());
        // Record this round's SAMF compression stats (delta from previous round).
        {
            let cm = crate::replace::transpositions::SAMF_COMPRESSIONS_MADE
                .load(std::sync::atomic::Ordering::Relaxed);
            let cf = crate::replace::transpositions::SAMF_COMPRESSIONS_FAILED
                .load(std::sync::atomic::Ordering::Relaxed);
            per_round_samf.push((cm - prev_samf_made, cf - prev_samf_failed));
            prev_samf_made = cm;
            prev_samf_failed = cf;
        }
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
        let n = if leave { n / 2 } else { n };
        let n = if do_gadgetize { n / 2 } else { n };
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

            writeln!(f, "=== Round {} ===\n{}\n", i + 1, circuit.repr())
                .expect("Failed to write progress");
        }
    }

    println!("Final len: {}", circuit.gates.len());
    // Compare against the original circuit on its own wires only: gadgetize/interleave
    // expand the wire count, but functionality is preserved only on the original n wires.
    let n = if leave { n / 2 } else { n };
    let n = if do_gadgetize { n / 2 } else { n };
    circuit
        .probably_equal(&c, n, 150_000)
        .expect("The circuits differ somewhere!");

    // Write to file
    let circuit_str = circuit.repr();
    File::create(save)
        .and_then(|mut f| f.write_all(circuit_str.as_bytes()))
        .expect("Failed to write recent_circuit.txt");

    println!("Final circuit written to {}", save);

    if shuffled {
        println!("--- SAMF compressions per round (shuffled shooting game) ---");
        let (mut tot_made, mut tot_failed) = (0usize, 0usize);
        for (r, (made, failed)) in per_round_samf.iter().enumerate() {
            println!("Round {}: made {}  failed {}", r + 1, made, failed);
            tot_made += made;
            tot_failed += failed;
        }
        println!("Total (this run): made {}  failed {}", tot_made, tot_failed);
    }
}
