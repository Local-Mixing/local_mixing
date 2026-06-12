use std::{fs::File, io::Write};

use primitive_types::U512 as u512;
use rand::{Rng, RngCore};

use crate::{
    circuit::circuit::{CircuitSeq, Gate},
    replace::{
        gadgets::{feistalize, gadgetize},
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

fn feistal_middle_matches_original(
    original: &CircuitSeq,
    transformed: &CircuitSeq,
    original_n: usize,
    total_wires: usize,
    num_inputs: usize,
) -> Result<(), String> {
    assert!(
        total_wires <= 512,
        "feistalized equality check supports up to 512 wires"
    );
    let mask = if original_n < 512 {
        (u512::one() << original_n) - u512::one()
    } else {
        u512::MAX
    };
    for _ in 0..num_inputs {
        let mut bytes = [0u8; 64];
        rand::rng().fill_bytes(&mut bytes);
        let random = u512::from_little_endian(&bytes);
        let x = random & mask;
        let y = (random >> original_n) & mask;
        let z = (random >> (2 * original_n)) & mask;
        let extra = if total_wires > 3 * original_n {
            let extra_mask = (u512::one() << (total_wires - 3 * original_n)) - u512::one();
            (random >> (3 * original_n)) & extra_mask
        } else {
            u512::zero()
        };
        let input = x | (y << original_n) | (z << (2 * original_n)) | (extra << (3 * original_n));
        let original_output = Gate::evaluate_index_list_512(x, &original.gates) & mask;
        let transformed_output = Gate::evaluate_index_list_512(input, &transformed.gates);
        let middle = (transformed_output >> original_n) & mask;
        if middle != (y ^ original_output) {
            return Err("Feistalized circuit middle block is not y ^ C(x)".to_string());
        }
    }
    Ok(())
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
    leave: bool,
    do_gadgetize: bool,
    do_feistalize: bool,
    gadget_path: Option<&str>,
    full_shuffle: bool,
    gates_ahead_expand: usize,
    gates_ahead_samf: usize,
    type_attempts: usize,
    shooting_times: usize,
    egg: bool,
    rg_freq: usize,
    single_end: bool,
) {
    // Start with the input circuit
    let save_base = save.strip_suffix(".txt").unwrap_or(save);
    println!("Starting len: {}", c.gates.len());
    let mut circuit = c.clone();
    // Repeat `rounds` times
    let mut post_len = 0;
    let mut count = 0;
    if do_gadgetize || do_feistalize {
        let mut rng = rand::rng();
        let before = circuit.gates.len();
        let (label, transformed_n) = if do_feistalize {
            circuit = feistalize(&circuit, n, rg_freq, &mut rng);
            ("Feistalized", 3 * n)
        } else {
            circuit = gadgetize(&circuit, n, rg_freq, &mut rng);
            ("Gadgetized", 2 * n)
        };
        println!(
            "{}: {} gates → {} gates, {} wires",
            label,
            before,
            circuit.gates.len(),
            transformed_n
        );
        // Save the transformed circuit to --gadget_path, or ./gadgetized/{final path
        // component of source} when none was supplied.
        let gadget_path = match gadget_path {
            Some(p) => p.to_string(),
            None => {
                let file_name = std::path::Path::new(source)
                    .file_name()
                    .expect("Source path has no final component")
                    .to_str()
                    .expect("Source file name is not valid UTF-8");
                std::fs::create_dir_all("./gadgetized").expect("Failed to create ./gadgetized");
                format!("./gadgetized/{}", file_name)
            }
        };
        if let Some(parent) = std::path::Path::new(&gadget_path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).expect("Failed to create gadget output directory");
            }
        }
        File::create(&gadget_path)
            .and_then(|mut f| f.write_all(circuit.repr().as_bytes()))
            .expect("Failed to write transformed circuit");
        println!("{} circuit written to {}", label, gadget_path);
    }
    let n = if do_feistalize {
        3 * n
    } else if do_gadgetize {
        2 * n
    } else {
        n
    };
    if leave {
        circuit = interleave(&circuit, n, env);
    }
    let n = if leave { 2 * n } else { n };
    if full_shuffle {
        // SAMF insertion is equivalence-preserving by construction, so no retry guard.
        insert_wire_m_samfs_every_x(&mut circuit, n, n, 1, env, curated_shard_dbs, shard_dbs);
        println!("After full shuffle: {} gates", circuit.gates.len());
    }
    // --single-end accumulator: SAMF state carried across ALL rounds,
    // undone in one pass after the last round. `total_t` is the composed wire permutation
    // (round order), `total_neg` the pending negation in the current wire space.
    let mut total_t = crate::replace::transpositions::Transpositions {
        transpositions: Vec::new(),
    };
    let mut total_neg = vec![0u8; n];

    // Per-round SAMF stats (deltas): inserted / hidden / hide-failed / curated expansions.
    use crate::replace::transpositions::{
        CURATED_REPLACEMENTS_MADE, SAMF_COMPRESSIONS_FAILED, SAMF_COMPRESSIONS_MADE,
        SAMF_HIDE_ATTEMPTS, SAMF_HIDE_ELIGIBLE_EXPANSIONS, SAMF_HIDE_LOOKUP_MISSES,
        SAMF_HIDE_REJECTED_EXPOSED, SAMF_HIDE_SKIPPED_MATERIALIZED, SAMF_INSERTIONS_MADE,
    };
    use std::sync::atomic::Ordering::Relaxed;
    #[derive(Clone, Copy)]
    struct RoundSamfStats {
        inserted: usize,
        hidden: usize,
        failed: usize,
        curated: usize,
        eligible: usize,
        skipped_materialized: usize,
        attempts: usize,
        lookup_misses: usize,
        rejected_exposed: usize,
    }

    let mut per_round_samf: Vec<RoundSamfStats> = Vec::new();
    let mut prev_ins = SAMF_INSERTIONS_MADE.load(Relaxed);
    let mut prev_made = SAMF_COMPRESSIONS_MADE.load(Relaxed);
    let mut prev_failed = SAMF_COMPRESSIONS_FAILED.load(Relaxed);
    let mut prev_cur = CURATED_REPLACEMENTS_MADE.load(Relaxed);
    let mut prev_eligible = SAMF_HIDE_ELIGIBLE_EXPANSIONS.load(Relaxed);
    let mut prev_skipped = SAMF_HIDE_SKIPPED_MATERIALIZED.load(Relaxed);
    let mut prev_attempts = SAMF_HIDE_ATTEMPTS.load(Relaxed);
    let mut prev_misses = SAMF_HIDE_LOOKUP_MISSES.load(Relaxed);
    let mut prev_rejected = SAMF_HIDE_REJECTED_EXPOSED.load(Relaxed);
    // A single-end shuffle cannot be reversed back after each round because its SAMF state is
    // intentionally left live. Choose one direction for the complete accumulated shuffle and
    // reverse it back only after the final unsamf.
    let single_end_reversed = single_end && rand::rng().random_bool(0.5);
    if single_end_reversed {
        println!("Collision-game direction: reversed (complete single-end shuffle)");
        circuit.gates.reverse();
    }
    for i in 0..rounds {
        if egg {
            let pair_mode = ExpandPairMode::Curated {
                curated_shard_dbs: &curated_shard_dbs,
            };
            circuit = expand_once(&circuit, n, env, shard_dbs, &pair_mode);
        } else if single_end {
            // Accumulate this round's SAMFs WITHOUT undoing — functionality is intentionally
            // broken between rounds; we undo everything once after the last round (below).
            use crate::replace::transpositions::shuffled_shoot_then_samf_core;
            let (out, t_round, neg_round, _c) = shuffled_shoot_then_samf_core(
                &circuit.gates,
                n,
                m,
                x,
                gates_ahead_expand,
                gates_ahead_samf,
                type_attempts,
                shooting_times,
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
        } else {
            // Choose the shooting direction outside the collision game. Reversal surrounds the
            // complete shooting + plain-SAMF + unsamf operation so no pending SAMF state is
            // reversed as though it were an ordinary gate sequence.
            use crate::replace::transpositions::shuffled_shoot_then_samf;
            let reversed = rand::rng().random_bool(0.5);
            println!(
                "Collision-game direction: {}",
                if reversed { "reversed" } else { "forward" }
            );
            if reversed {
                circuit.gates.reverse();
            }
            shuffled_shoot_then_samf(
                &mut circuit,
                n,
                m,
                x,
                gates_ahead_expand,
                gates_ahead_samf,
                type_attempts,
                shooting_times,
                env,
                curated_shard_dbs,
                shard_dbs,
            );
            if reversed {
                circuit.gates.reverse();
            }
        }
        println!("After shooting game: {} gates", circuit.gates.len());
        // The normal shooting path already inserts plain SAMFs as part of
        // shuffled_shoot_then_samf[_core]. Egg mode does not use that path, so it performs its
        // one plain-SAMF insertion here.
        if egg {
            insert_wire_m_samfs_every_x(&mut circuit, n, m, x, env, curated_shard_dbs, shard_dbs);
            println!("After inserting samfs: {} gates", circuit.gates.len());
        }
        // --single-end: after the FINAL round's shuffle, before its compression, undo ALL
        // accumulated SAMFs/NOTs in one pass — restoring equivalence to the original input.
        if single_end && i == rounds - 1 {
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
            if single_end_reversed {
                circuit.gates.reverse();
                println!("Restored forward circuit direction");
            }
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
        // Record + print this round's SAMF stats (deltas from the previous round).
        {
            let ins = SAMF_INSERTIONS_MADE.load(Relaxed);
            let made = SAMF_COMPRESSIONS_MADE.load(Relaxed);
            let failed = SAMF_COMPRESSIONS_FAILED.load(Relaxed);
            let cur = CURATED_REPLACEMENTS_MADE.load(Relaxed);
            let eligible = SAMF_HIDE_ELIGIBLE_EXPANSIONS.load(Relaxed);
            let skipped = SAMF_HIDE_SKIPPED_MATERIALIZED.load(Relaxed);
            let attempts = SAMF_HIDE_ATTEMPTS.load(Relaxed);
            let misses = SAMF_HIDE_LOOKUP_MISSES.load(Relaxed);
            let rejected = SAMF_HIDE_REJECTED_EXPOSED.load(Relaxed);
            let d_ins = ins - prev_ins;
            let d_made = made - prev_made;
            let d_failed = failed - prev_failed;
            let d_cur = cur - prev_cur;
            let d_eligible = eligible - prev_eligible;
            let d_skipped = skipped - prev_skipped;
            let d_attempts = attempts - prev_attempts;
            let d_misses = misses - prev_misses;
            let d_rejected = rejected - prev_rejected;
            println!(
                "  Round {}/{} SAMFs inserted: {} (hidden {}, plain {}) | curated expansions: {} | hide-fails: {}",
                i + 1,
                rounds,
                d_ins,
                d_made,
                d_ins.saturating_sub(d_made),
                d_cur,
                d_failed
            );
            println!(
                "    hide diagnostics: eligible {} | skipped-materialized {} | attempts {} | lookup-misses {} | rejected-exposed {}",
                d_eligible, d_skipped, d_attempts, d_misses, d_rejected
            );
            per_round_samf.push(RoundSamfStats {
                inserted: d_ins,
                hidden: d_made,
                failed: d_failed,
                curated: d_cur,
                eligible: d_eligible,
                skipped_materialized: d_skipped,
                attempts: d_attempts,
                lookup_misses: d_misses,
                rejected_exposed: d_rejected,
            });
            prev_ins = ins;
            prev_made = made;
            prev_failed = failed;
            prev_cur = cur;
            prev_eligible = eligible;
            prev_skipped = skipped;
            prev_attempts = attempts;
            prev_misses = misses;
            prev_rejected = rejected;
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
        let original_n = if leave { n / 2 } else { n };
        let original_n = if do_feistalize {
            original_n / 3
        } else if do_gadgetize {
            original_n / 2
        } else {
            original_n
        };
        let functionality_ok = if do_feistalize {
            feistal_middle_matches_original(&c, &circuit, original_n, n, 1_000)
        } else {
            c.probably_equal(&circuit, original_n, 1_000)
        };
        if functionality_ok.is_err() {
            panic!("The functionality has changed");
        }
        {
            // Write this round's circuit to its own file: same path as -d but with
            // `round{n}` inserted before the .txt extension.
            let round_path = format!("{}round{}.txt", save_base, i + 1);
            println!("Writing round {} to {}", i + 1, round_path);
            File::create(&round_path)
                .and_then(|mut f| f.write_all(circuit.repr().as_bytes()))
                .expect("Failed to write round circuit");
        }
    }

    println!("Final len: {}", circuit.gates.len());
    // Compare against the original circuit. Gadgetize/interleave preserve the low original_n
    // outputs; feistalize preserves C(x) in the middle original_n-wire block as y ^ C(x).
    let original_n = if leave { n / 2 } else { n };
    let original_n = if do_feistalize {
        original_n / 3
    } else if do_gadgetize {
        original_n / 2
    } else {
        original_n
    };
    if do_feistalize {
        feistal_middle_matches_original(&c, &circuit, original_n, n, 10_000)
            .expect("The circuits differ somewhere!");
    } else {
        circuit
            .probably_equal(&c, original_n, 10_000)
            .expect("The circuits differ somewhere!");
    }

    // Write to file
    let circuit_str = circuit.repr();
    File::create(save)
        .and_then(|mut f| f.write_all(circuit_str.as_bytes()))
        .expect("Failed to write recent_circuit.txt");

    println!("Final circuit written to {}", save);

    {
        println!("--- SAMF stats per round ---");
        let (
            mut t_ins,
            mut t_made,
            mut t_failed,
            mut t_cur,
            mut t_eligible,
            mut t_skipped,
            mut t_attempts,
            mut t_misses,
            mut t_rejected,
        ) = (
            0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize,
        );
        for (r, stats) in per_round_samf.iter().enumerate() {
            println!(
                "Round {}: inserted {} (hidden {}, plain {}) | curated expansions {} | hide-fails {}",
                r + 1,
                stats.inserted,
                stats.hidden,
                stats.inserted.saturating_sub(stats.hidden),
                stats.curated,
                stats.failed
            );
            println!(
                "  hide diagnostics: eligible {} | skipped-materialized {} | attempts {} | lookup-misses {} | rejected-exposed {}",
                stats.eligible,
                stats.skipped_materialized,
                stats.attempts,
                stats.lookup_misses,
                stats.rejected_exposed
            );
            t_ins += stats.inserted;
            t_made += stats.hidden;
            t_failed += stats.failed;
            t_cur += stats.curated;
            t_eligible += stats.eligible;
            t_skipped += stats.skipped_materialized;
            t_attempts += stats.attempts;
            t_misses += stats.lookup_misses;
            t_rejected += stats.rejected_exposed;
        }
        println!(
            "Total (this run): SAMFs inserted {} (hidden {}, plain {}) | curated expansions {} | hide-fails {}",
            t_ins,
            t_made,
            t_ins.saturating_sub(t_made),
            t_cur,
            t_failed
        );
        println!(
            "Total hide diagnostics: eligible {} | skipped-materialized {} | attempts {} | lookup-misses {} | rejected-exposed {}",
            t_eligible, t_skipped, t_attempts, t_misses, t_rejected
        );
    }
}
