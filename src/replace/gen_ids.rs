use crate::circuit::circuit::CircuitSeq;
use crate::replace::pairs::{gate_pair_taxonomy, GatePair};
use lmdb::{Cursor, Transaction, WriteFlags};

#[cfg(test)]
mod tests {
    use super::decode_circuits;
    use crate::circuit::circuit::CircuitSeq;
    use lmdb::{Cursor, Environment, Transaction};
    use std::path::Path;
    use std::time::Instant;

    #[test]
    fn shard_gate_count_histogram() {
        let env = Environment::new()
            .set_max_dbs(300)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new("./db"))
            .expect("Failed to open ./db lmdb environment");

        let mut counts = [0u64; 7]; // index = gate count; 0 unused
        let print_interval_secs = 10.0f64;
        let mut last_print = Instant::now();

        for shard in 0u16..256 {
            let name = format!("{:02x}", shard);
            let db = match env.open_db(Some(name.as_str())) {
                Ok(db) => db,
                Err(_) => continue,
            };
            let txn = env.begin_ro_txn().expect("ro txn");
            let mut cursor = txn.open_ro_cursor(db).expect("cursor");
            for (_, value) in cursor.iter() {
                for circuit in decode_circuits(value) {
                    let g = circuit.gates.len().min(6);
                    if g >= 1 {
                        counts[g] += 1;
                    }
                }
            }
            drop(cursor);
            drop(txn);

            if last_print.elapsed().as_secs_f64() >= print_interval_secs {
                print_histogram(&counts, shard + 1);
                last_print = Instant::now();
            }
        }

        print_histogram(&counts, 256);
    }

    fn print_histogram(counts: &[u64; 7], shards_done: u16) {
        println!("\n--- shard {}/256 ---", shards_done);
        let total: u64 = counts[1..].iter().sum();
        for g in 1..=6 {
            println!("  {:1} gate(s): {:>12}  ({:.1}%)",
                g, counts[g],
                if total > 0 { counts[g] as f64 / total as f64 * 100.0 } else { 0.0 });
        }
        println!("  total:    {:>12}", total);
    }

    #[test]
    fn count_id_g_circuits() {
        let env = Environment::new()
            .set_max_dbs(300)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new("./db"))
            .expect("Failed to open ./db");

        let mut grand_total: u64 = 0;

        for i in 0..34 {
            let name = format!("id_g{}", i);
            let db = match env.open_db(Some(name.as_str())) {
                Ok(db) => db,
                Err(_) => continue,
            };
            let txn = env.begin_ro_txn().expect("ro txn");
            let mut cursor = txn.open_ro_cursor(db).expect("cursor");
            let count = cursor.iter().count() as u64;
            drop(cursor);
            drop(txn);

            if count > 0 {
                println!("id_g{:2}: {:>10} circuits", i, count);
                grand_total += count;
            }
        }

        println!("---");
        println!("Total:    {:>10} circuits", grand_total);
    }

    #[test]
    fn most_popular_6gate_polynomial() {
        let env = Environment::new()
            .set_max_dbs(300)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new("./db"))
            .expect("Failed to open ./db lmdb environment");

        let mut best_key: Vec<u8> = Vec::new();
        let mut best_count: usize = 0;
        let mut best_circuits: Vec<CircuitSeq> = Vec::new();
        let mut total_entries = 0u64;

        for shard in 0u16..256 {
            let name = format!("{:02x}", shard);
            let db = match env.open_db(Some(name.as_str())) {
                Ok(db) => db,
                Err(_) => continue,
            };
            let txn = env.begin_ro_txn().expect("ro txn");
            let mut cursor = txn.open_ro_cursor(db).expect("cursor");
            for (key, value) in cursor.iter() {
                let six_gate: Vec<CircuitSeq> = decode_circuits(value)
                    .into_iter()
                    .filter(|c| c.gates.len() == 6)
                    .collect();
                if six_gate.len() > best_count {
                    best_count = six_gate.len();
                    best_key = key.to_vec();
                    best_circuits = six_gate;
                }
                total_entries += 1;
            }
            drop(cursor);
            drop(txn);
        }

        println!("\nTotal entries scanned: {}", total_entries);
        println!("Most popular 6-gate polynomial: {} circuits", best_count);
        println!("Key (hex): {}", best_key.iter().map(|b| format!("{:02x}", b)).collect::<String>());
        println!("Example circuit: {}", best_circuits[0].repr());
        println!("All {} circuits:", best_count);
        for (i, c) in best_circuits.iter().enumerate() {
            println!("  [{}] {}", i, c.repr());
        }
    }

    #[test]
    fn list_completion_m2() {
        let env = Environment::new()
            .set_max_dbs(300)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new("./db"))
            .expect("Failed to open ./db");

        // Open all 256 shard DBs.
        let shard_dbs: Vec<lmdb::Database> = (0u8..=255)
            .map(|s| {
                let name = format!("{:02x}", s);
                env.open_db(Some(name.as_str()))
                    .unwrap_or_else(|e| panic!("Failed to open shard {:02x}: {:?}", s, e))
            })
            .collect();

        let comp_db = env.open_db(Some("completion_m2"))
            .expect("completion_m2 DB not found — run build_completion_m2 first");

        let txn = env.begin_ro_txn().expect("ro txn");

        // Collect all entries so we can drop the cursor before doing shard lookups.
        let mut entries: Vec<(Vec<u8>, Vec<u8>)> = {
            let mut cursor = txn.open_ro_cursor(comp_db).expect("cursor");
            cursor.iter().map(|(k, v)| (k.to_vec(), v.to_vec())).collect()
        };
        entries.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));

        let mut missing = 0u32;

        for (key, value) in &entries {
            let shard_idx = key[0] as usize;

            // Verify the hash exists in the corresponding shard.
            let shard_val: Option<&[u8]> = txn.get(shard_dbs[shard_idx], key).ok();
            if shard_val.is_none() {
                let hex: String = key.iter().map(|b| format!("{:02x}", b)).collect();
                println!("MISSING from shard {:02x}: {}", shard_idx, hex);
                missing += 1;
            }

            // Find the shortest circuit in the shard to use as the canonical label.
            let canonical_label = shard_val
                .and_then(|sv| {
                    let mut shortest: Option<CircuitSeq> = None;
                    let mut pos = 0;
                    while pos < sv.len() {
                        let len = sv[pos] as usize;
                        pos += 1;
                        if pos + len > sv.len() { break; }
                        let c = CircuitSeq::from_blob(&sv[pos..pos + len]);
                        pos += len;
                        if shortest.as_ref().map_or(true, |s: &CircuitSeq| c.gates.len() < s.gates.len()) {
                            shortest = Some(c);
                        }
                    }
                    shortest.map(|c| c.repr())
                })
                .unwrap_or_else(|| {
                    let hex: String = key.iter().map(|b| format!("{:02x}", b)).collect();
                    format!("<hash:{}>", &hex[..8])
                });

            // Decode and print completions.
            let mut completions: Vec<CircuitSeq> = Vec::new();
            let mut pos = 0;
            while pos < value.len() {
                let len = value[pos] as usize;
                pos += 1;
                if pos + len > value.len() { break; }
                completions.push(CircuitSeq::from_blob(&value[pos..pos + len]));
                pos += len;
            }

            println!("Canonical: {}  [shard {:02x} {}]  ({} completions)",
                canonical_label,
                shard_idx,
                if shard_val.is_some() { "OK" } else { "MISSING" },
                completions.len());
            for (i, c) in completions.iter().enumerate() {
                println!("  {:2}. {}", i + 1, c.repr());
            }
            println!();
        }

        drop(txn);

        println!("Total: {} canonical circuits, {} missing from shards", entries.len(), missing);
        assert_eq!(missing, 0, "{} hashes not found in shard DBs", missing);
    }

    #[test]
    fn count_completion_m2_circuits() {
        let env = Environment::new()
            .set_max_dbs(300)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new("./db"))
            .expect("Failed to open ./db");

        let comp_db = env.open_db(Some("completion_m2"))
            .expect("completion_m2 DB not found — run build_completion_m2 first");

        let txn = env.begin_ro_txn().expect("ro txn");
        let mut cursor = txn.open_ro_cursor(comp_db).expect("cursor");

        let mut num_keys: u64 = 0;
        let mut total_circuits: u64 = 0;
        let mut by_size: std::collections::BTreeMap<usize, u64> = std::collections::BTreeMap::new();

        for (_key, value) in cursor.iter() {
            num_keys += 1;
            let mut pos = 0;
            while pos < value.len() {
                let len = value[pos] as usize;
                pos += 1;
                if pos + len > value.len() { break; }
                let gate_count = len / 3;
                *by_size.entry(gate_count).or_insert(0) += 1;
                total_circuits += 1;
                pos += len;
            }
        }

        drop(cursor);
        drop(txn);

        println!("completion_m2 stats:");
        println!("  Keys (canonical pairs): {}", num_keys);
        println!("  Total circuits:         {}", total_circuits);
        println!("  By gate count:");
        for (gates, count) in &by_size {
            println!("    {} gates: {}", gates, count);
        }
    }

    /// Sanity check: rewire the first stored completion for the first 2 canonical circuits
    /// back into the canonical circuit's wire space and verify they compute the same function.
    #[test]
    fn sanity_completion_equivalence() {
        use crate::circuit::circuit::{polys_repr_blob, Permutation};
        use xxhash_rust::xxh3::xxh3_128;

        let env = Environment::new()
            .set_max_dbs(300)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new("./db"))
            .expect("Failed to open ./db");

        let shard_dbs: Vec<lmdb::Database> = (0u8..=255)
            .map(|s| env.open_db(Some(format!("{:02x}", s).as_str())).unwrap())
            .collect();

        let comp_db = env.open_db(Some("completion_m2"))
            .expect("completion_m2 DB not found");

        let txn = env.begin_ro_txn().expect("ro txn");

        let entries: Vec<(Vec<u8>, Vec<u8>)> = {
            let mut cursor = txn.open_ro_cursor(comp_db).expect("cursor");
            cursor.iter().map(|(k, v)| (k.to_vec(), v.to_vec())).collect()
        };

        let num_wires = 16usize; // enough headroom for extra-wire assignment

        for (key, value) in entries.iter().take(2) {
            // Get the shortest (canonical 2-gate) circuit from the shard.
            let shard_val = txn.get(shard_dbs[key[0] as usize], key).expect("not in shard");
            let canon_c = {
                let mut shortest: Option<CircuitSeq> = None;
                let mut pos = 0;
                while pos < shard_val.len() {
                    let len = shard_val[pos] as usize; pos += 1;
                    if pos + len > shard_val.len() { break; }
                    let c = CircuitSeq::from_blob(&shard_val[pos..pos + len]); pos += len;
                    if shortest.as_ref().map_or(true, |s: &CircuitSeq| c.gates.len() < s.gates.len()) {
                        shortest = Some(c);
                    }
                }
                shortest.expect("shard value was empty")
            };

            // Decode the first stored completion.
            let len = value[0] as usize;
            let mut comp = CircuitSeq::from_blob(&value[1..1 + len]);

            // Canonicalize the canonical circuit to get final_order and used.
            // Try forward first, then reversed (matching the lookup order in
            // replace_single_pair_with_completion).
            let (final_order, used, is_reversed) = {
                let (fwd_polys, fwd_order, fwd_used) = canon_c.canonicalize_polys_single(false);
                let fwd_key = xxh3_128(&polys_repr_blob(&fwd_polys)).to_le_bytes();
                if fwd_key == key.as_slice() {
                    (fwd_order, fwd_used, false)
                } else {
                    let (_, rev_order, rev_used) = canon_c.canonicalize_polys_single(true);
                    (rev_order, rev_used, true)
                }
            };

            if is_reversed { comp.gates.reverse(); }

            // Rewire completion: canonical → dense → actual (same as replace_single_pair_with_completion).
            let repl_n = comp.max_wire() + 1;
            let mut order_data = final_order.data.clone();
            while order_data.len() < repl_n { order_data.push(order_data.len()); }
            let order_len = order_data.len().max(final_order.data.len());
            comp.rewire(&Permutation { data: order_data }, order_len);

            let repl_n_b = comp.max_wire() + 1;
            let mut used_ext = used.clone();
            let mut next_wire = num_wires as u8;
            while used_ext.len() < repl_n_b {
                used_ext.push(next_wire);
                next_wire += 1;
            }
            let rewired = CircuitSeq::unrewire_subcircuit(&comp, &used_ext);

            // Evaluate both on every input to the used wires (fixing extra wires to 0).
            let n_used = used.len();
            let mut all_equal = true;
            for bits in 0u64..(1u64 << n_used) {
                // Build a full input word with `used[i]` set to bit i.
                let mut input: usize = 0;
                for (i, &w) in used.iter().enumerate() {
                    if (bits >> i) & 1 == 1 { input |= 1 << w; }
                }
                let out_canon   = canon_c.evaluate(input);
                let out_rewired = rewired.evaluate(input);
                // Compare only on the used wires.
                let mask: usize = used.iter().fold(0, |acc, &w| acc | (1 << w));
                if out_canon & mask != out_rewired & mask {
                    all_equal = false;
                    println!("  MISMATCH at input {:b}: canon={:b} rewired={:b}",
                        input, out_canon & mask, out_rewired & mask);
                }
            }

            println!("canonical: {}  completion: {}  equal={}", canon_c.repr(), rewired.repr(), all_equal);
            assert!(all_equal, "completion does not match canonical circuit");
        }
    }

    #[test]
    fn benchmark_compress_workload() {
        use rayon::prelude::*;
        use crate::replace::replace::compress_big_ancillas;
        use rand::RngCore;

        const N: usize = 128;
        const CHUNK_GATES: usize = 1_500;
        const TOTAL_TASKS: usize = 256;

        let env = std::sync::Arc::new(
            Environment::new()
                .set_max_readers(10000)
                .set_max_dbs(300)
                .set_map_size(800 * 1024 * 1024 * 1024)
                .open(Path::new("./db"))
                .expect("Failed to open ./db"),
        );

        let shard_dbs: Vec<lmdb::Database> = (0u8..=255)
            .map(|s| env.open_db(Some(format!("{:02x}", s).as_str())).unwrap())
            .collect();

        // Pre-generate TOTAL_TASKS independent ~1500-gate chunks from the 10k/128 scenario.
        let mut rng = rand::rng();
        let chunks: Vec<crate::circuit::circuit::CircuitSeq> = (0..TOTAL_TASKS)
            .map(|_| {
                let mut gate_bytes = vec![0u8; CHUNK_GATES * 3];
                rng.fill_bytes(&mut gate_bytes);
                let gates = gate_bytes.chunks(3).map(|c| {
                    let t  = c[0] % N as u8;
                    let c1 = c[1] % (N as u8 - 1) + if c[1] % (N as u8 - 1) >= t { 1 } else { 0 };
                    let c2 = c[2] % (N as u8 - 2);
                    [t, c1, c2]
                }).collect();
                crate::circuit::circuit::CircuitSeq { gates }
            })
            .collect();

        println!("Tasks: {}, Gates/task: {}, Wires: {}, Threads: {}",
            TOTAL_TASKS, CHUNK_GATES, N, rayon::current_num_threads());

        let bench_start = Instant::now();

        let results: Vec<usize> = chunks
            .par_iter()
            .enumerate()
            .map(|(i, chunk)| {
                let mode = i % 3;
                compress_big_ancillas(chunk, 100, N, &env, &shard_dbs, mode).gates.len()
            })
            .collect();

        let elapsed = bench_start.elapsed().as_secs_f64();
        let total_in: usize = chunks.iter().map(|c| c.gates.len()).sum();
        let total_out: usize = results.iter().sum();

        println!("Total tasks:     {}", TOTAL_TASKS);
        println!("Elapsed:         {:.2}s", elapsed);
        println!("Tasks/sec:       {:.1}", TOTAL_TASKS as f64 / elapsed);
        println!("Gates in:        {}", total_in);
        println!("Gates out:       {}", total_out);
        println!("Reduction:       {:.2}%", 100.0 * (1.0 - total_out as f64 / total_in as f64));
    }

    #[test]
    fn benchmark_lmdb_reads() {
        use rayon::prelude::*;

        const TOTAL_READS: usize = 1_000_000;

        let env = std::sync::Arc::new(
            Environment::new()
                .set_max_readers(10000)
                .set_max_dbs(300)
                .set_map_size(800 * 1024 * 1024 * 1024)
                .open(Path::new("./db"))
                .expect("Failed to open ./db"),
        );

        let shard_dbs: Vec<lmdb::Database> = (0u8..=255)
            .map(|s| env.open_db(Some(format!("{:02x}", s).as_str())).unwrap())
            .collect();

        // Collect a capped sample of keys (1 in every SAMPLE_STRIDE) so RAM usage stays bounded.
        const POOL_CAP: usize = 1_000_000;
        println!("Collecting key pool (capped at {})...", POOL_CAP);
        let mut pool: Vec<(usize, Vec<u8>)> = Vec::new();
        let mut seen = 0usize;
        'outer: for shard in 0usize..256 {
            let txn = env.begin_ro_txn().expect("ro txn");
            let mut cursor = txn.open_ro_cursor(shard_dbs[shard]).expect("cursor");
            for (k, _) in cursor.iter() {
                seen += 1;
                if seen % 3000 == 0 {
                    pool.push((shard, k.to_vec()));
                    if pool.len() >= POOL_CAP {
                        break 'outer;
                    }
                }
            }
            drop(cursor);
            drop(txn);
        }
        println!("Pool: {} keys sampled from {} total scanned", pool.len(), seen);
        assert!(!pool.is_empty(), "DB is empty");

        // Build read list by sampling pool (wrapping if needed).
        let reads: Vec<(usize, Vec<u8>)> = (0..TOTAL_READS)
            .map(|i| pool[i % pool.len()].clone())
            .collect();

        // --- Parallel: each task opens its own read transaction and does a real B-tree lookup ---
        let t = Instant::now();
        let hits: usize = reads
            .par_iter()
            .map(|(shard, key)| {
                let txn = env.begin_ro_txn().expect("ro txn");
                txn.get(shard_dbs[*shard], key).is_ok() as usize
            })
            .sum();
        let elapsed = t.elapsed();
        println!(
            "Parallel:   {} reads in {:.3}s  →  {:.0} reads/sec  (hits: {}/{})",
            TOTAL_READS, elapsed.as_secs_f64(),
            TOTAL_READS as f64 / elapsed.as_secs_f64(),
            hits, TOTAL_READS
        );

        // --- Sequential: single transaction, same lookups ---
        let t2 = Instant::now();
        let txn = env.begin_ro_txn().expect("ro txn");
        for (shard, key) in &reads {
            let _ = txn.get(shard_dbs[*shard], key);
        }
        drop(txn);
        let elapsed2 = t2.elapsed();
        println!(
            "Sequential: {} reads in {:.3}s  →  {:.0} reads/sec",
            TOTAL_READS, elapsed2.as_secs_f64(),
            TOTAL_READS as f64 / elapsed2.as_secs_f64()
        );
    }
}

fn decode_circuits(value: &[u8]) -> Vec<CircuitSeq> {
    let mut circuits = Vec::new();
    let mut pos = 0;
    while pos < value.len() {
        let len = value[pos] as usize;
        pos += 1;
        if pos + len > value.len() {
            break;
        }
        circuits.push(CircuitSeq::from_blob(&value[pos..pos + len]));
        pos += len;
    }
    circuits
}

fn remove_adjacent_equal(gates: &mut Vec<[u8; 3]>) {
    let mut i = 0;
    while i + 1 < gates.len() {
        if gates[i] == gates[i + 1] {
            gates.drain(i..=i + 1);
            if i > 0 {
                i -= 1;
            }
        } else {
            i += 1;
        }
    }
}


pub fn open_id_dbs(env: &lmdb::Environment) -> Vec<lmdb::Database> {
    let mut txn = env.begin_rw_txn().expect("Failed to begin rw txn for id_db setup");
    let dbs: Vec<lmdb::Database> = (0..34)
        .map(|i| {
            let name = format!("id_g{}", i);
            let db = unsafe { txn.open_db(Some(&name)) }
                .unwrap_or_else(|_| {
                    unsafe { txn.create_db(Some(&name), lmdb::DatabaseFlags::empty()) }
                        .unwrap_or_else(|e| panic!("Failed to create id_g{}: {:?}", i, e))
                });
            txn.clear_db(db).unwrap_or_else(|e| panic!("Failed to clear id_g{}: {:?}", i, e));
            db
        })
        .collect();
    txn.commit().expect("Failed to commit id_db setup txn");
    dbs
}

pub fn generate_identity_db(
    env: &lmdb::Environment,
    shard_dbs: &[lmdb::Database],
    id_dbs: &[lmdb::Database],
) {
    let mut total = 0u64;
    // Per-DB counters for sequential keys; open_id_dbs clears DBs so we start at 0.
    let mut counters = vec![0u64; 34];
    // Deduplicate within this run using a per-type seen set.
    let mut seen: Vec<std::collections::HashSet<Vec<u8>>> =
        (0..34).map(|_| std::collections::HashSet::new()).collect();

    for shard_idx in 0..256usize {
        let txn = env.begin_ro_txn().expect("ro txn");
        let db = shard_dbs[shard_idx];
        let mut cursor = txn.open_ro_cursor(db).expect("cursor");

        // Collect entries with multiple circuits
        let mut multi: Vec<Vec<u8>> = Vec::new();
        for (_, value) in cursor.iter() {
            let circuits = decode_circuits(value);
            if circuits.len() >= 2 {
                multi.push(value.to_vec());
            }
        }
        drop(cursor);
        drop(txn);

        if multi.is_empty() {
            continue;
        }

        let mut wtxn = env.begin_rw_txn().expect("rw txn");

        for value in &multi {
            let circuits = decode_circuits(value);

            // All ordered pairs (a, b) with a != b
            for i in 0..circuits.len() {
                for j in 0..circuits.len() {
                    if i == j {
                        continue;
                    }

                    let a = &circuits[i];
                    let b = &circuits[j];

                    let mut gates = a.gates.clone();
                    let mut b_rev = b.gates.clone();
                    b_rev.reverse();
                    gates.extend(b_rev);

                    let mut identity = CircuitSeq { gates };
                    identity.canonicalize();
                    remove_adjacent_equal(&mut identity.gates);

                    if identity.gates.len() < 2 {
                        continue;
                    }

                    // All rotations
                    let len = identity.gates.len();
                    for rot in 0..len {
                        let mut rotated = Vec::with_capacity(len);
                        rotated.extend_from_slice(&identity.gates[rot..]);
                        rotated.extend_from_slice(&identity.gates[..rot]);

                        let g1 = rotated[0];
                        let g2 = rotated[1];
                        let ctype = GatePair::to_int(&gate_pair_taxonomy(&g1, &g2));

                        let rotated_circuit = CircuitSeq { gates: rotated };
                        let blob = rotated_circuit.repr_blob();

                        if seen[ctype].insert(blob.clone()) {
                            let idx = counters[ctype];
                            counters[ctype] += 1;
                            wtxn.put(id_dbs[ctype], &idx.to_be_bytes(), &blob, WriteFlags::empty())
                                .expect("put identity");
                            total += 1;
                        }
                    }
                }
            }
        }

        wtxn.commit().expect("commit");
        println!("Shard {:3}/256 done", shard_idx + 1);
    }

    println!("Total identity entries written: {}", total);
}
