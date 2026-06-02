use std::{
    fs::OpenOptions,
    io::Write,
    sync::atomic::{AtomicBool, Ordering},
};

use rand::{Rng, prelude::SliceRandom};

use crate::{
    circuit::circuit::CircuitSeq,
    random::random_data::{
        // random_circuit,
        shoot_left_vec,
        shoot_right_vec,
    },
    replace::pairs::{expand_curated_lmdb, replace_single_pair},
};

// Help with early stops without losing all data
pub static SHOULD_DUMP: AtomicBool = AtomicBool::new(false);
use signal_hook::consts::{SIGINT, SIGTERM};
use signal_hook::iterator::Signals;
use std::thread;

pub fn install_kill_handler() {
    let mut signals = Signals::new([SIGINT, SIGTERM]).expect("signals");

    thread::spawn(move || {
        // Block until the first SIGINT/SIGTERM, then flag a dump.
        if signals.forever().next().is_some() {
            eprintln!("Received termination signal, dumping acc...");
            SHOULD_DUMP.store(true, Ordering::SeqCst);
        }
    });
}

// Simple shooting game method. Send a gate to the right until a collision is made, then make a replacement. Continue the same from the right-most gate
// Repeat until the blowup is `enough` and then compress
pub fn simple_shooting_game(
    c: &CircuitSeq,
    n: usize,
    env: &lmdb::Environment,
    curr_round: usize,
    last_round: usize,
    stop: usize,
    intermediate: &str,
    ends: bool,
    iter: usize,
    gates_ahead: usize,
    curated_shard_dbs: &[lmdb::Database],
    shard_dbs: &[lmdb::Database],
) -> CircuitSeq {
    let mut gates = c.gates.clone();
    println!(
        "   {}/{}: Starting simple shooting game until {} rounds or {} gates",
        curr_round, last_round, iter, stop
    );
    let mut len = gates.len();
    println!("Starting gates: {}", len);
    let mut rng = rand::rng();
    let mut count = 0;
    while (count < iter) && (len < stop) {
        let left = rng.random_bool(0.5);
        let starting_idx = if ends && left {
            len - 1
        } else if ends && !left {
            0
        } else {
            rng.random_range(0..len)
        };
        if left {
            let mut curr_idx = starting_idx;
            while curr_idx != 0 {
                let after_idx = shoot_left_vec(&mut gates, curr_idx);
                // need at least a collision pair (after_idx-1, after_idx)
                if after_idx < 1 {
                    break;
                }

                // (replacement, number of gates consumed from the window)
                let repl_result: Option<(Vec<[u16; 3]>, usize)> =
                    if gates_ahead > 2 && after_idx + 1 >= gates_ahead {
                        let w_start = after_idx + 1 - gates_ahead;
                        if let Some(repl) = expand_curated_lmdb(
                            &gates[w_start..after_idx + 1],
                            n,
                            env,
                            curated_shard_dbs,
                            shard_dbs,
                        ) {
                            Some((repl, gates_ahead))
                        } else {
                            // replace collision pair (b,c), then seam (a, pr[0])
                            let (pr, _) = replace_single_pair(
                                &gates[after_idx - 1],
                                &gates[after_idx],
                                n,
                                env,
                                curated_shard_dbs,
                                shard_dbs,
                            );
                            if pr.is_empty() {
                                None
                            } else if after_idx >= 2 {
                                let a_gate = gates[after_idx - 2];
                                let (sr, _) = replace_single_pair(
                                    &a_gate,
                                    &pr[0],
                                    n,
                                    env,
                                    curated_shard_dbs,
                                    shard_dbs,
                                );
                                if sr.is_empty() {
                                    Some((pr, 2))
                                } else {
                                    let mut final_r = sr;
                                    final_r.extend_from_slice(&pr[1..]);
                                    Some((final_r, 3))
                                }
                            } else {
                                Some((pr, 2))
                            }
                        }
                    } else {
                        let (r, _) = replace_single_pair(
                            &gates[after_idx - 1],
                            &gates[after_idx],
                            n,
                            env,
                            curated_shard_dbs,
                            shard_dbs,
                        );
                        if r.is_empty() { None } else { Some((r, 2)) }
                    };

                match repl_result {
                    Some((repl, consumed)) => {
                        let new_len = repl.len();
                        let w_start = after_idx + 1 - consumed;
                        gates.splice(w_start..after_idx + 1, repl);
                        len = len - consumed + new_len;
                        curr_idx = w_start;
                    }
                    None => break,
                }
            }
        } else {
            let mut curr_idx = starting_idx;
            while curr_idx != len {
                let after_idx = shoot_right_vec(&mut gates, curr_idx);
                // need at least a collision pair (after_idx, after_idx+1)
                if after_idx + 2 > len {
                    break;
                }

                let repl_result: Option<(Vec<[u16; 3]>, usize)> =
                    if gates_ahead > 2 && after_idx + gates_ahead <= len {
                        if let Some(repl) = expand_curated_lmdb(
                            &gates[after_idx..after_idx + gates_ahead],
                            n,
                            env,
                            curated_shard_dbs,
                            shard_dbs,
                        ) {
                            Some((repl, gates_ahead))
                        } else {
                            // replace collision pair (a,b), then seam (pr.last, c)
                            let (pr, _) = replace_single_pair(
                                &gates[after_idx],
                                &gates[after_idx + 1],
                                n,
                                env,
                                curated_shard_dbs,
                                shard_dbs,
                            );
                            if pr.is_empty() {
                                None
                            } else if after_idx + 3 <= len {
                                let c_gate = gates[after_idx + 2];
                                let last_of_pr = *pr.last().unwrap();
                                let (sr, _) = replace_single_pair(
                                    &last_of_pr,
                                    &c_gate,
                                    n,
                                    env,
                                    curated_shard_dbs,
                                    shard_dbs,
                                );
                                if sr.is_empty() {
                                    Some((pr, 2))
                                } else {
                                    let mut final_r = pr;
                                    final_r.pop();
                                    final_r.extend_from_slice(&sr);
                                    Some((final_r, 3))
                                }
                            } else {
                                Some((pr, 2))
                            }
                        }
                    } else {
                        let (r, _) = replace_single_pair(
                            &gates[after_idx],
                            &gates[after_idx + 1],
                            n,
                            env,
                            curated_shard_dbs,
                            shard_dbs,
                        );
                        if r.is_empty() { None } else { Some((r, 2)) }
                    };

                match repl_result {
                    Some((repl, consumed)) => {
                        let new_len = repl.len();
                        gates.splice(after_idx..after_idx + consumed, repl);
                        len = len - consumed + new_len;
                        curr_idx = after_idx + new_len - 1;
                    }
                    None => break,
                }
            }
        }
        count += 1;
    }

    let acc = CircuitSeq { gates };
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(intermediate)
        .expect("Failed to open replacednocomp.txt");
    println!("Writing to {}", intermediate);
    writeln!(f, "{}", acc.repr()).expect("Failed to write intermediate CircuitSeq");
    if acc.probably_equal(&c, n, 1000).is_err() {
        panic!("Functionality lost during sequential butterfly");
    }
    acc
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Pair Replacement Methods
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

pub fn split_into_random_chunk_ranges(
    len: usize,
    k: usize,
    rng: &mut impl Rng,
) -> Vec<(usize, usize)> {
    if k == 1 {
        return vec![(0, len)];
    }

    let min_size = 100;
    assert!(k * min_size <= len);

    let slack = len - k * min_size;

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

    let mut ranges = Vec::with_capacity(k);
    let mut idx = 0;
    for size in sizes {
        let end = idx + size;
        ranges.push((idx, end));
        idx = end;
    }

    ranges
}
