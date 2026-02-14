// Replacement code used in the mixing methods

use crate::{
    circuit::circuit::{CircuitSeq, Permutation}, rainbow::canonical::Canonicalization, random::random_data::{
        contiguous_convex, 
        get_canonical, 
        shoot_random_gate, 
        simple_find_convex_subcircuit,

        targeted_find_convex_subcircuit_deep,
    }
};
use crate::replace::identities::random_perm_lmdb;
use crate::replace::identities::random_canonical_id;
use crate::replace::identities::random_id;
use rand::Rng;

use rusqlite::{Connection, Statement};

use lmdb::{Transaction};

extern crate lmdb_sys;

use std::{
    cmp::{max, min},
    collections::{HashMap, HashSet},
    time::Instant,
};
use std::sync::atomic::{AtomicU64, Ordering};
// use rand::prelude::IndexedRandom;

// Return a random contiguous subcircuit, its starting index (gate), and ending index
pub fn random_subcircuit(circuit: &CircuitSeq) -> (CircuitSeq, usize, usize) {
    let len = circuit.gates.len();
    
    if circuit.gates.len() == 0 {
        return (CircuitSeq{gates: Vec::new()}, 0, 0)
    }

    let mut rng = rand::rng();
    //get size with more bias to lower length subcircuits
    let a = rng.random_range(0..len);

    // pick one of 1, 2, 4, 8
    let shift = rng.random_range(0..4);
    let upper = 1 << shift;

    let mut b = (a + (1 + rng.random_range(0..upper))) as usize;

    if b > len {
        b = len;
    }

    if a == b {
        if b < len - 1 {
            b += 1;
        } else {
            b -= 1;
        }
    }

    let start = min(a,b);
    let end = max(a,b);

    let subcircuit = circuit.gates[start..end].to_vec();

    (CircuitSeq{ gates: subcircuit }, start, end)
}

pub fn random_subcircuit_max(circuit: &CircuitSeq, max_len: usize) -> (CircuitSeq, usize, usize) {
    let len = circuit.gates.len();
    if len == 0 {
        return (CircuitSeq { gates: Vec::new() }, 0, 0);
    }

    let mut rng = rand::rng();

    let start = rng.random_range(0..len);

    let remaining = len - start;
    let allowed_len = remaining.min(max_len);

    let shift = rng.random_range(0..4); // 0..3
    let mut sub_len = 1 << shift;        // 1,2,4,8
    if sub_len > allowed_len {
        sub_len = allowed_len;
    }

    sub_len = sub_len.max(1);

    let end = start + sub_len;
    let subcircuit = circuit.gates[start..end].to_vec();

    (CircuitSeq { gates: subcircuit }, start, end)
}

// Timing variables for benchmarking
pub static PERMUTATION_TIME: AtomicU64 = AtomicU64::new(0);
pub static SQL_TIME: AtomicU64 = AtomicU64::new(0);
pub static CANON_TIME: AtomicU64 = AtomicU64::new(0);
pub static CONVEX_FIND_TIME: AtomicU64 = AtomicU64::new(0);
pub static CONTIGUOUS_TIME: AtomicU64 = AtomicU64::new(0);
pub static REWIRE_TIME: AtomicU64 = AtomicU64::new(0);
pub static COMPRESS_TIME: AtomicU64 = AtomicU64::new(0);
pub static UNREWIRE_TIME: AtomicU64 = AtomicU64::new(0);
pub static REPLACE_TIME: AtomicU64 = AtomicU64::new(0);
pub static DEDUP_TIME: AtomicU64 = AtomicU64::new(0);
pub static PICK_SUBCIRCUIT_TIME: AtomicU64 = AtomicU64::new(0);
pub static CANONICALIZE_TIME: AtomicU64 = AtomicU64::new(0);
pub static ROW_FETCH_TIME: AtomicU64 = AtomicU64::new(0);
pub static SROW_FETCH_TIME: AtomicU64 = AtomicU64::new(0);
pub static SIXROW_FETCH_TIME: AtomicU64 = AtomicU64::new(0);
pub static LROW_FETCH_TIME: AtomicU64 = AtomicU64::new(0);
pub static DB_OPEN_TIME: AtomicU64 = AtomicU64::new(0);
pub static TXN_TIME: AtomicU64 = AtomicU64::new(0);
pub static LMDB_LOOKUP_TIME: AtomicU64 = AtomicU64::new(0);
pub static FROM_BLOB_TIME: AtomicU64 = AtomicU64::new(0);
pub static SPLICE_TIME: AtomicU64 = AtomicU64::new(0);
pub static TRIAL_TIME: AtomicU64 = AtomicU64::new(0);
pub static IDENTITY_TIME: AtomicU64 = AtomicU64::new(0);

// Unsupported compression code
// See compress_lmdb
pub fn compress(
    c: &CircuitSeq,
    trials: usize,
    conn: &mut Connection,
    bit_shuf: &Vec<Vec<usize>>,
    n: usize,
) -> CircuitSeq {

    let id = Permutation::id_perm(n);

    // let t0 = Instant::now();
    let c_perm = c.permutation(n);
    // PERMUTATION_TIME.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

    if c_perm == id {
        return CircuitSeq { gates: Vec::new() };
    }

    let mut compressed = c.clone();
    if compressed.gates.is_empty() {
        return CircuitSeq { gates: Vec::new() };
    }

    let mut i = 0;
    while i < compressed.gates.len().saturating_sub(1) {
        if compressed.gates[i] == compressed.gates[i + 1] {
            compressed.gates.drain(i..=i + 1);
            i = i.saturating_sub(2);
        } else {
            i += 1;
        }
    }

    if compressed.gates.is_empty() {
        return CircuitSeq { gates: Vec::new() };
    }

    for _ in 0..trials {
        let (mut subcircuit, start, end) = random_subcircuit(&compressed);
        subcircuit.canonicalize();

        let max = if n == 7 {
            4
        } else if n == 5 || n == 6 {
            5
        } else if n == 4 {
            6
        } else {
            12
        };

        let sub_m = subcircuit.gates.len();
        let min = min(sub_m, max);
        
        let (canon_perm_blob, canon_shuf_blob) = if subcircuit.gates.len() <= max && n == 7{
            let table = format!("n{}m{}", n, min);
            let query = format!(
                "SELECT perm, shuf FROM {} WHERE circuit = ?1 LIMIT 1",
                table
            );

            // let sql_t0 = Instant::now();
            let mut stmt = match conn.prepare(&query) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let rows = stmt.query([&subcircuit.repr_blob()]);
            // SQL_TIME.fetch_add(sql_t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

            let mut r = match rows {
                Ok(r) => r,
                Err(_) => continue,
            };

            if let Some(row_result) = r.next().unwrap() {
                
                (row_result
                    .get(0)
                    .expect("Failed to get blob"),
                row_result
                    .get(1)
                    .expect("Failed to get blob"))
                
            } else {
                continue
            }

        } else {
            // let t1 = Instant::now();
            let sub_perm = subcircuit.permutation(n);
            // PERMUTATION_TIME.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);

            // let t2 = Instant::now();
            let canon_perm = get_canonical(&sub_perm, bit_shuf);
            // CANON_TIME.fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);

            (canon_perm.perm.repr_blob(), canon_perm.shuffle.repr_blob())
        };

        for smaller_m in 1..=sub_m {
            let table = format!("n{}m{}", n, smaller_m);
            let query = format!(
                "SELECT * FROM {} WHERE perm = ?1 ORDER BY RANDOM() LIMIT 1",
                table
            );

            // let sql_t0 = Instant::now();
            let mut stmt = match conn.prepare(&query) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let rows = stmt.query([&canon_perm_blob]);
            // SQL_TIME.fetch_add(sql_t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

            let mut r = match rows {
                Ok(r) => r,
                Err(_) => continue,
            };

            if let Some(row_result) = r.next().unwrap() {
                let blob: Vec<u8> = row_result
                    .get(0)
                    .expect("Failed to get blob");
                let mut repl = CircuitSeq::from_blob(&blob);

                let repl_perm: Vec<u8> = row_result
                    .get(1)
                    .expect("Failed to get blob");

                let repl_shuf: Vec<u8> = row_result
                    .get(2)
                    .expect("Failed to get blob");

                if repl.gates.len() <= subcircuit.gates.len() {
                    let rc = Canonicalization { perm: Permutation::from_blob(&repl_perm), shuffle: Permutation::from_blob(&repl_shuf) };

                    if !rc.shuffle.data.is_empty() {
                        repl.rewire(&rc.shuffle, n);
                    }
                    
                    repl.rewire(&Permutation::from_blob(&canon_shuf_blob).invert(), n);

                    compressed.gates.splice(start..end, repl.gates);
                    break;
                }
            }
        }
    }

    let mut j = 0;
    while j < compressed.gates.len().saturating_sub(1) {
        if compressed.gates[j] == compressed.gates[j + 1] {
            compressed.gates.drain(j..=j + 1);
            j = j.saturating_sub(2);
        } else {
            j += 1;
        }
    }

    compressed
}

// Expand with ancilla wires or gates
pub fn expand_lmdb<'a>(
    c: &CircuitSeq,
    trials: usize,
    bit_shuf: &Vec<Vec<usize>>,
    n: usize,
    env: &lmdb::Environment,
    _old_n: usize,
    dbs: &HashMap<String, lmdb::Database>,
    prepared_stmt: &mut rusqlite::Statement<'a>,
    prepared_stmt2: &mut rusqlite::Statement<'a>,
    conn: &Connection
) -> CircuitSeq {
    let mut compressed = c.clone();
    if compressed.gates.is_empty() {
        return CircuitSeq { gates: Vec::new() };
    }
    let perm_len = 1 << n;
    for _ in 0..trials {
        let (mut subcircuit, start, end) = random_subcircuit(&compressed);
        subcircuit.canonicalize();

        let max = if n == 7 {
            4
        } else if n == 5 || n == 6 {
            5
        } else if n == 4 {
            6
        } else {
            10
        };

        let sub_m = subcircuit.gates.len();
        let (canon_perm_blob, canon_shuf_blob) =
        if sub_m <= max && ((n == 6 && sub_m == 5) || (n == 7 && sub_m  == 4)) {
            if n == 7 && sub_m == 4 {
                let stmt: &mut Statement<'_> = &mut *prepared_stmt;

                let row_start = Instant::now();
                let blobs_result: rusqlite::Result<(Vec<u8>, Vec<u8>)> =
                    stmt.query_row(
                        [&subcircuit.repr_blob()],
                        |row| Ok((row.get(0)?, row.get(1)?)),
                    );

                SROW_FETCH_TIME.fetch_add(
                    row_start.elapsed().as_nanos() as u64,
                    Ordering::Relaxed,
                );

                match blobs_result {
                    Ok(b) => b,
                    Err(rusqlite::Error::QueryReturnedNoRows) => continue,
                    Err(e) => panic!("SQL query failed: {:?}", e),
                }

            } else if n == 6 && sub_m == 5 {
                let stmt: &mut Statement<'_> = &mut *prepared_stmt2;

                let row_start = Instant::now();
                let blobs_result: rusqlite::Result<(Vec<u8>, Vec<u8>)> =
                    stmt.query_row(
                        [&subcircuit.repr_blob()],
                        |row| Ok((row.get(0)?, row.get(1)?)),
                    );

                SIXROW_FETCH_TIME.fetch_add(
                    row_start.elapsed().as_nanos() as u64,
                    Ordering::Relaxed,
                );

                match blobs_result {
                    Ok(b) => b,
                    Err(rusqlite::Error::QueryReturnedNoRows) => continue,
                    Err(e) => panic!("SQL query failed: {:?}", e),
                }
            
            } else {
                let table = format!("n{}m{}", n, sub_m);
                let query = format!(
                    "SELECT perm, shuf FROM {} WHERE circuit = ?1 LIMIT 1",
                    table
                );

                let row_start = Instant::now();
                let blobs_result: rusqlite::Result<(Vec<u8>, Vec<u8>)> =
                    conn.query_row(
                        &query,
                        [&subcircuit.repr_blob()],
                        |row| Ok((row.get(0)?, row.get(1)?)),
                    );

                ROW_FETCH_TIME.fetch_add(
                    row_start.elapsed().as_nanos() as u64,
                    Ordering::Relaxed,
                );

                match blobs_result {
                    Ok(b) => b,
                    Err(rusqlite::Error::QueryReturnedNoRows) => continue,
                    Err(e) => panic!("SQL query failed: {:?}", e),
                }
            }
        } else if sub_m <= max && (n >= 4) {
            let db_name = format!("n{}m{}perms", n, sub_m);
                let db = match dbs.get(&db_name) {
                    Some(db) => *db,
                    None => continue,
                };

                let txn = env.begin_ro_txn().expect("lmdb ro txn");

                let row_start = Instant::now();
                let val = match txn.get(db, &subcircuit.repr_blob()) {
                    Ok(v) => v,
                    Err(lmdb::Error::NotFound) => continue,
                    Err(e) => panic!("LMDB get failed: {:?}", e),
                };
                LROW_FETCH_TIME.fetch_add(row_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

                let perm = val[..perm_len].to_vec();
                let shuf = val[perm_len..].to_vec();

                (perm, shuf)
        } else {
            // let t1 = Instant::now();
            let sub_perm = subcircuit.permutation(n);
            // PERMUTATION_TIME.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);

            // let t2 = Instant::now();
            let canon_perm = get_canonical(&sub_perm, bit_shuf);
            // CANON_TIME.fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);

            (canon_perm.perm.repr_blob(), canon_perm.shuffle.repr_blob())
        };

        let prefix = canon_perm_blob.as_slice();
        for smaller_m in (1..=max).rev() {
            let db_name = format!("n{}m{}", n, smaller_m);
            let &db = match dbs.get(&db_name) {
                Some(db) => db,
                None => continue,
            };
            let mut invert = false;
            let hit = {
                let txn = env.begin_ro_txn().expect("txn");

                // let t0 = Instant::now();
                
                let mut res = random_perm_lmdb(&txn, db, prefix);
                if res.is_none() {
                    let prefix_inv_blob = Permutation::from_blob(&prefix).invert().repr_blob();
                    invert = true;
                    res = random_perm_lmdb(&txn, db, &prefix_inv_blob);
                }

                // SQL_TIME.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

                res.map(|val_blob| val_blob)
            };

            if let Some(val_blob) = hit {
                let repl_blob: Vec<u8> = val_blob;

                let mut repl = CircuitSeq::from_blob(&repl_blob);

                if invert {
                    repl.gates.reverse();
                }

                repl.rewire(&Permutation::from_blob(&canon_shuf_blob).invert(), n);

                if repl.gates.len() == end - start {
                    compressed.gates[start..end].copy_from_slice(&repl.gates);
                } else {
                    compressed.gates.splice(start..end, repl.gates);
                }
                break;
            }
        }

    }

    compressed
}

// Attempt to compress every possible subcircuit
// Fast for small subcircuits
pub fn compress_exhaust(
    c: &CircuitSeq,
    conn: &mut Connection,
    bit_shuf: &Vec<Vec<usize>>,
    n: usize,
) -> CircuitSeq {
    let id = Permutation::id_perm(n);

    if c.permutation(n) == id {
        return CircuitSeq { gates: Vec::new() };
    }

    let mut compressed = c.clone();
    if compressed.gates.is_empty() {
        return CircuitSeq { gates: Vec::new() };
    }

    // Initial cleanup of consecutive duplicates
    let mut i = 0;
    while i < compressed.gates.len().saturating_sub(1) {
        if compressed.gates[i] == compressed.gates[i + 1] {
            compressed.gates.drain(i..=i + 1);
            i = i.saturating_sub(2);
        } else {
            i += 1;
        }
    }

    if compressed.gates.is_empty() {
        return CircuitSeq { gates: Vec::new() };
    }

    let mut changed = true;
    let mut seen_positions: HashSet<(usize, usize)> = HashSet::new(); // Track replaced positions globally

    while changed {
        changed = false;
        let len = compressed.gates.len();

        'outer: for start in 0..len-2 {
            for end in (start + 2)..len { // skip length 1
                if seen_positions.contains(&(start, end)) {
                    continue; // skip positions already replaced in this pass
                }
                let subcircuit = CircuitSeq {
                    gates: compressed.gates[start..end].to_vec(),
                };

                let sub_perm = subcircuit.permutation(n);
                let canon_perm = get_canonical(&sub_perm, bit_shuf);
                let sub_blob = canon_perm.perm.repr_blob();

                let sub_m = subcircuit.gates.len();

                for smaller_m in 1..=sub_m {
                    let table = format!("n{}m{}", n, smaller_m);
                    let query = format!(
                        "SELECT circuit FROM {} WHERE perm = ?1 ORDER BY RANDOM() LIMIT 1",
                        table
                    );

                    let mut stmt = match conn.prepare(&query) {
                        Ok(s) => s,
                        Err(_) => continue,
                    };
                    let rows = stmt.query([&sub_blob]);

                    if let Ok(mut r) = rows {
                        if let Some(row) = r.next().unwrap() {
                            let blob: Vec<u8> = row.get(0).expect("Failed to get blob");
                            let mut repl = CircuitSeq::from_blob(&blob);

                            if repl.gates.len() <= subcircuit.gates.len() {
                                let repl_perm = repl.permutation(n);
                                let rc = get_canonical(&repl_perm, bit_shuf);

                                if !rc.shuffle.data.is_empty() {
                                    repl.rewire(&rc.shuffle, n);
                                }
                                repl.rewire(&canon_perm.shuffle.invert(), n);

                                if repl.permutation(n) != sub_perm {
                                    panic!("Replacement permutation mismatch!");
                                }

                                // Only perform replacement if it actually changes the gates
                                if repl.gates != subcircuit.gates {
                                    let old_len = end - start;
                                    let repl_len = repl.gates.len();
                                    let delta = repl_len as isize - old_len as isize; // ≤ 0 always
                                    let r_len = repl.gates.len();
                                    compressed.gates.splice(start..end, repl.gates);
                                    
                                    if r_len < subcircuit.gates.len() {
                                        // Update seen_positions
                                        let mut updated = HashSet::new();

                                        for &(a, b) in &seen_positions {
                                            // If it overlaps the replaced region, discard it
                                            if !(b <= start || a >= end) {
                                                continue;
                                            }

                                            // If it comes after the replaced region, shift back
                                            if a >= end {
                                                let new_a = (a as isize + delta) as usize;
                                                let new_b = (b as isize + delta) as usize;
                                                if new_a < new_b {
                                                    updated.insert((new_a, new_b));
                                                }
                                            } else {
                                                // Unaffected before the replacement
                                                updated.insert((a, b));
                                            }
                                        }

                                        seen_positions = updated;
                                    }

                                    // Mark the new replaced range
                                    seen_positions.insert((start, end));

                                    changed = true;
                                    break 'outer;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Final cleanup of consecutive duplicates
    let mut i = 0;
    while i < compressed.gates.len().saturating_sub(1) {
        if compressed.gates[i] == compressed.gates[i + 1] {
            compressed.gates.drain(i..=i + 1);
            i = i.saturating_sub(2);
        } else {
            i += 1;
        }
    }

    compressed
}

// Compress on larger number of wires
pub fn compress_big(
    c: &CircuitSeq, 
    trials: usize, 
    num_wires: usize, 
    conn: &mut Connection, 
    env: &lmdb::Environment, 
    bit_shuf_list: &Vec<Vec<Vec<usize>>>, 
    dbs: &HashMap<String, lmdb::Database>,
) -> CircuitSeq {
    let table = format!("n{}m{}", 7, 4);
    let query_limit = format!("SELECT perm, shuf FROM {} WHERE circuit = ?1 LIMIT 1", table);
    let mut stmt = conn.prepare(&query_limit).unwrap();
    let table2 = format!("n{}m{}", 6, 5);
    let query_limit = format!("SELECT perm, shuf FROM {} WHERE circuit = ?1 LIMIT 1", table2);
    let mut stmt2 = conn.prepare(&query_limit).unwrap();
    let mut circuit = c.clone();
    let mut rng = rand::rng();

    let mut i = 0;
    while i < circuit.gates.len().saturating_sub(1) {
        if circuit.gates[i] == circuit.gates[i + 1] {
            circuit.gates.drain(i..=i + 1);
            i = i.saturating_sub(2);
        } else {
            i += 1;
        }
    }

    for _ in 0..trials {
        shoot_random_gate(&mut circuit, 100_000);
        let t0 = Instant::now();
        let mut subcircuit_gates = vec![];
        let random_max_wires = rng.random_range(5..=7);
        let size = if random_max_wires == 7 {
            6
        } else if random_max_wires == 6 {
            4
        } else {
            3
        };
        for set_size in (3..=size).rev() {
            let (gates, _) = simple_find_convex_subcircuit(set_size, random_max_wires, num_wires, &circuit, &mut rng);
            if !gates.is_empty() {
                subcircuit_gates = gates;
                break;
            }
        }
        CONVEX_FIND_TIME.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

        if subcircuit_gates.is_empty() {
            continue;
        }

        let gates: Vec<[u8; 3]> = subcircuit_gates.iter().map(|&g| circuit.gates[g]).collect();
        subcircuit_gates.sort();

        let t1 = Instant::now();
        let (start, end) = contiguous_convex(&mut circuit, &mut subcircuit_gates, num_wires).unwrap();
        CONTIGUOUS_TIME.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let mut subcircuit = CircuitSeq { gates };

        let expected_slice: Vec<_> = subcircuit_gates.iter().map(|&i| circuit.gates[i]).collect();
        let actual_slice = &circuit.gates[start..=end];
        if actual_slice != &expected_slice[..] {
            continue;
        }

        let t2 = Instant::now();
        let used_wires = subcircuit.used_wires();
        subcircuit = CircuitSeq::rewire_subcircuit(&mut circuit, &mut subcircuit_gates, &used_wires);
        REWIRE_TIME.fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t3 = Instant::now();
        let sub_num_wires = used_wires.len();
        let bit_shuf = &bit_shuf_list[sub_num_wires - 3];
        PERMUTATION_TIME.fetch_add(t3.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t4 = Instant::now();
        let subcircuit_temp = compress_lmdb(&subcircuit, 20, &bit_shuf, sub_num_wires, env, dbs, &mut stmt, &mut stmt2, conn);
        COMPRESS_TIME.fetch_add(t4.elapsed().as_nanos() as u64, Ordering::Relaxed);

        subcircuit = subcircuit_temp;

        let t5 = Instant::now();
        subcircuit = CircuitSeq::unrewire_subcircuit(&subcircuit, &used_wires);
        UNREWIRE_TIME.fetch_add(t5.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t6 = Instant::now();
        let repl_len = subcircuit.gates.len();
        let old_len = end - start + 1;

        if repl_len == old_len {
            for i in 0..repl_len {
                circuit.gates[start + i] = subcircuit.gates[i];
            }
        } else if repl_len < old_len {
            for i in 0..repl_len {
                circuit.gates[start + i] = subcircuit.gates[i];
            }
            for i in (end + 1)..circuit.gates.len() {
                circuit.gates[i - (old_len - repl_len)] = circuit.gates[i];
            }
            circuit.gates.truncate(circuit.gates.len() - (old_len - repl_len));
        } else {
            panic!("Replacement grew, which is not allowed");
        }
        REPLACE_TIME.fetch_add(t6.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    let t7 = Instant::now();
    let mut i = 0;
    while i < circuit.gates.len().saturating_sub(1) {
        if circuit.gates[i] == circuit.gates[i + 1] {
            circuit.gates.drain(i..=i + 1);
            i = i.saturating_sub(2);
        } else {
            i += 1;
        }
    }
    DEDUP_TIME.fetch_add(t7.elapsed().as_nanos() as u64, Ordering::Relaxed);

    circuit
}

// Sequential compression method
pub fn sequential_compress_big(
    c: &CircuitSeq, 
    num_wires: usize, 
    conn: &mut Connection, 
    env: &lmdb::Environment, 
    bit_shuf_list: &Vec<Vec<Vec<usize>>>, 
    dbs: &HashMap<String, lmdb::Database>,
) -> CircuitSeq {
    let table = format!("n{}m{}", 7, 4);
    let query_limit = format!("SELECT perm, shuf FROM {} WHERE circuit = ?1 LIMIT 1", table);
    let mut stmt = conn.prepare(&query_limit).unwrap();
    let table2 = format!("n{}m{}", 6, 5);
    let query_limit = format!("SELECT perm, shuf FROM {} WHERE circuit = ?1 LIMIT 1", table2);
    let mut stmt2 = conn.prepare(&query_limit).unwrap();
    let mut circuit = c.clone();
    let mut rng = rand::rng();

    let mut i = 0;
    while i < circuit.gates.len().saturating_sub(1) {
        if circuit.gates[i] == circuit.gates[i + 1] {
            circuit.gates.drain(i..=i + 1);
            i = i.saturating_sub(2);
        } else {
            i += 1;
        }
    }

    let mut len = circuit.gates.len();
    let mut i = 0;
    while i < len {
        let t0 = Instant::now();
        let mut subcircuit_gates = vec![];
        let random_max_wires = rng.random_range(5..=7);
        let size = if random_max_wires == 7 {
            6
        } else if random_max_wires == 6 {
            4
        } else {
            3
        };
        for set_size in (3..=size).rev() {
            let (gates, _) = targeted_find_convex_subcircuit_deep(set_size, random_max_wires, num_wires, &circuit, &mut rng, i);
            if !gates.is_empty() {
                subcircuit_gates = gates;
                break;
            }
            if set_size == 3 {
                let (gates, _) = targeted_find_convex_subcircuit_deep(set_size, 7, num_wires, &circuit, &mut rng, i);
                subcircuit_gates = gates;
            }
        }
        CONVEX_FIND_TIME.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

        if subcircuit_gates.is_empty() {
            i+=1;
            continue;
        }

        let gates: Vec<[u8; 3]> = subcircuit_gates.iter().map(|&g| circuit.gates[g]).collect();
        subcircuit_gates.sort();

        let t1 = Instant::now();
        let (start, end) = contiguous_convex(&mut circuit, &mut subcircuit_gates, num_wires).unwrap();
        CONTIGUOUS_TIME.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let mut subcircuit = CircuitSeq { gates };

        let expected_slice: Vec<_> = subcircuit_gates.iter().map(|&i| circuit.gates[i]).collect();
        let actual_slice = &circuit.gates[start..=end];
        if actual_slice != &expected_slice[..] {
            i+=1;
            continue;
        }

        let t2 = Instant::now();
        let used_wires = subcircuit.used_wires();
        subcircuit = CircuitSeq::rewire_subcircuit(&mut circuit, &mut subcircuit_gates, &used_wires);
        REWIRE_TIME.fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t3 = Instant::now();
        let sub_num_wires = used_wires.len();
        let bit_shuf = &bit_shuf_list[sub_num_wires - 3];
        PERMUTATION_TIME.fetch_add(t3.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t4 = Instant::now();
        let subcircuit_temp = compress_lmdb(&subcircuit, 20, &bit_shuf, sub_num_wires, env, dbs, &mut stmt, &mut stmt2, conn);
        COMPRESS_TIME.fetch_add(t4.elapsed().as_nanos() as u64, Ordering::Relaxed);

        subcircuit = subcircuit_temp;

        let t5 = Instant::now();
        subcircuit = CircuitSeq::unrewire_subcircuit(&subcircuit, &used_wires);
        UNREWIRE_TIME.fetch_add(t5.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t6 = Instant::now();
        let repl_len = subcircuit.gates.len();
        let old_len = end - start + 1;

        if repl_len == old_len {
            for i in 0..repl_len {
                circuit.gates[start + i] = subcircuit.gates[i];
            }
        } else if repl_len < old_len {
            for i in 0..repl_len {
                circuit.gates[start + i] = subcircuit.gates[i];
            }
            for i in (end + 1)..circuit.gates.len() {
                circuit.gates[i - (old_len - repl_len)] = circuit.gates[i];
            }
            circuit.gates.truncate(circuit.gates.len() - (old_len - repl_len));
        } else {
            panic!("Replacement grew, which is not allowed");
        }
        REPLACE_TIME.fetch_add(t6.elapsed().as_nanos() as u64, Ordering::Relaxed);
        i += 1;
        len = circuit.gates.len();
    }

    let t7 = Instant::now();
    let mut i = 0;
    while i < circuit.gates.len().saturating_sub(1) {
        if circuit.gates[i] == circuit.gates[i + 1] {
            circuit.gates.drain(i..=i + 1);
            i = i.saturating_sub(2);
        } else {
            i += 1;
        }
    }
    DEDUP_TIME.fetch_add(t7.elapsed().as_nanos() as u64, Ordering::Relaxed);

    circuit
}

// Allow ancillas in compression
pub fn sequential_compress_big_ancillas( 
    c: &CircuitSeq, 
    num_wires: usize, 
    conn: &mut Connection, 
    env: &lmdb::Environment, 
    bit_shuf_list: &Vec<Vec<Vec<usize>>>, 
    dbs: &HashMap<String, lmdb::Database>,
) -> CircuitSeq {
    let table = format!("n{}m{}", 7, 4);
    let query_limit = format!("SELECT perm, shuf FROM {} WHERE circuit = ?1 LIMIT 1", table);
    let mut stmt = conn.prepare(&query_limit).unwrap();
    let table2 = format!("n{}m{}", 6, 5);
    let query_limit = format!("SELECT perm, shuf FROM {} WHERE circuit = ?1 LIMIT 1", table2);
    let mut stmt2 = conn.prepare(&query_limit).unwrap();
    let mut circuit = c.clone();
    let mut rng = rand::rng();

    let mut i = 0;
    while i < circuit.gates.len().saturating_sub(1) {
        if circuit.gates[i] == circuit.gates[i + 1] {
            circuit.gates.drain(i..=i + 1);
            i = i.saturating_sub(2);
        } else {
            i += 1;
        }
    }

    let mut len = circuit.gates.len();
    let mut i = 0;
    while i < len {
        let t0 = Instant::now();
        let mut subcircuit_gates = vec![];
        let random_max_wires = rng.random_range(5..=7);
        let size = if random_max_wires == 7 {
            6
        } else if random_max_wires == 6 {
            4
        } else {
            3
        };
        for set_size in (3..=size).rev() {
            let (gates, _) = targeted_find_convex_subcircuit_deep(set_size, random_max_wires, num_wires, &circuit, &mut rng, i);
            if !gates.is_empty() {
                subcircuit_gates = gates;
                break;
            }
            if set_size == 3 {
                let (gates, _) = targeted_find_convex_subcircuit_deep(set_size, 7, num_wires, &circuit, &mut rng, i);
                subcircuit_gates = gates;
            }
        }
        CONVEX_FIND_TIME.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

        if subcircuit_gates.is_empty() {
            i+=1;
            continue;
        }

        let gates: Vec<[u8; 3]> = subcircuit_gates.iter().map(|&g| circuit.gates[g]).collect();
        subcircuit_gates.sort();

        let t1 = Instant::now();
        let (start, end) = contiguous_convex(&mut circuit, &mut subcircuit_gates, num_wires).unwrap();
        CONTIGUOUS_TIME.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let mut subcircuit = CircuitSeq { gates };

        let expected_slice: Vec<_> = subcircuit_gates.iter().map(|&i| circuit.gates[i]).collect();
        let actual_slice = &circuit.gates[start..=end];
        if actual_slice != &expected_slice[..] {
            i+=1;
            continue;
        }

        let t2 = Instant::now();
        let mut used_wires = subcircuit.used_wires();
        let n_wires = used_wires.len();
        let max = 7;
        let new_wires = rng.random_range(n_wires..=max);
        if new_wires > n_wires {
            let mut count = n_wires;
            while count < new_wires {
                let random = rng.random_range(0..num_wires);
                if used_wires.contains(&(random as u8)) {
                    continue
                }
                used_wires.push(random as u8);
                count += 1;
            }
        }
        subcircuit = CircuitSeq::rewire_subcircuit(&mut circuit, &mut subcircuit_gates, &used_wires);
        REWIRE_TIME.fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t3 = Instant::now();
        let sub_num_wires = used_wires.len();
        let bit_shuf = &bit_shuf_list[sub_num_wires - 3];
        PERMUTATION_TIME.fetch_add(t3.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t4 = Instant::now();
        let subcircuit_temp = compress_lmdb(&subcircuit, 20, &bit_shuf, sub_num_wires, env, dbs, &mut stmt, &mut stmt2, conn);
        COMPRESS_TIME.fetch_add(t4.elapsed().as_nanos() as u64, Ordering::Relaxed);

        subcircuit = subcircuit_temp;

        let t5 = Instant::now();
        subcircuit = CircuitSeq::unrewire_subcircuit(&subcircuit, &used_wires);
        UNREWIRE_TIME.fetch_add(t5.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t6 = Instant::now();
        let repl_len = subcircuit.gates.len();
        let old_len = end - start + 1;

        if repl_len == old_len {
            for i in 0..repl_len {
                circuit.gates[start + i] = subcircuit.gates[i];
            }
        } else if repl_len < old_len {
            for i in 0..repl_len {
                circuit.gates[start + i] = subcircuit.gates[i];
            }
            for i in (end + 1)..circuit.gates.len() {
                circuit.gates[i - (old_len - repl_len)] = circuit.gates[i];
            }
            circuit.gates.truncate(circuit.gates.len() - (old_len - repl_len));
        } else {
            panic!("Replacement grew, which is not allowed");
        }
        REPLACE_TIME.fetch_add(t6.elapsed().as_nanos() as u64, Ordering::Relaxed);
        i += 1;
        len = circuit.gates.len();
    }

    let t7 = Instant::now();
    let mut i = 0;
    while i < circuit.gates.len().saturating_sub(1) {
        if circuit.gates[i] == circuit.gates[i + 1] {
            circuit.gates.drain(i..=i + 1);
            i = i.saturating_sub(2);
        } else {
            i += 1;
        }
    }
    DEDUP_TIME.fetch_add(t7.elapsed().as_nanos() as u64, Ordering::Relaxed);

    circuit
}


pub fn compress_lmdb<'a>(
    c: &CircuitSeq,
    trials: usize,
    bit_shuf: &Vec<Vec<usize>>,
    n: usize,
    env: &lmdb::Environment,
    dbs: &HashMap<String, lmdb::Database>,
    prepared_stmt: &mut rusqlite::Statement<'a>,
    prepared_stmt2: &mut rusqlite::Statement<'a>,
    conn: &Connection,
) -> CircuitSeq {
    let id = Permutation::id_perm(n);
    let perm_len = 1 << n;
    // Timer for initial permutation
    let t0 = Instant::now();
    let c_perm = c.permutation(n);
    PERMUTATION_TIME.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

    if c_perm == id {
        return CircuitSeq { gates: Vec::new() };
    }

    let mut compressed = c.clone();
    if compressed.gates.is_empty() {
        return CircuitSeq { gates: Vec::new() };
    }

    // Timer for initial deduplication
    let dedup_start = Instant::now();
    let mut i = 0;
    while i < compressed.gates.len().saturating_sub(1) {
        if compressed.gates[i] == compressed.gates[i + 1] {
            compressed.gates.drain(i..=i + 1);
            i = i.saturating_sub(2);
        } else {
            i += 1;
        }
    }
    DEDUP_TIME.fetch_add(dedup_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

    if compressed.gates.is_empty() {
        return CircuitSeq { gates: Vec::new() };
    }

    let (do_subcircuit, trial_count) = if compressed.gates.len() < 5 {
        (false, 2)
    } else {
        (true, trials)
    };

    for _ in 0..trial_count {
        let trial_start = Instant::now();

        // Pick subcircuit
        let pick_start = Instant::now();
        let (subcircuit, start, end) = if do_subcircuit {
            random_subcircuit(&compressed)
        } else {
            (compressed.clone(), 0, compressed.gates.len())
        };
        PICK_SUBCIRCUIT_TIME.fetch_add(pick_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let mut subcircuit = subcircuit;

        // Canonicalize
        let canon_start = Instant::now();
        subcircuit.canonicalize();
        CANONICALIZE_TIME.fetch_add(canon_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let max = if n == 7 { 3 } else if n == 5 || n == 6 { 5 } else if n == 4 { 6 } else { 10 };
        let sub_m = subcircuit.gates.len();
        let min = min(sub_m, max);

        let (canon_perm_blob, canon_shuf_blob) = 
            if sub_m <= max && ((n == 6 && sub_m == 5) || (n == 7 && sub_m  == 4)) {
                if n == 7 && sub_m == 4 {
                    let stmt: &mut Statement<'_> = &mut *prepared_stmt;

                    let row_start = Instant::now();
                    let blobs_result: rusqlite::Result<(Vec<u8>, Vec<u8>)> =
                        stmt.query_row(
                            [&subcircuit.repr_blob()],
                            |row| Ok((row.get(0)?, row.get(1)?)),
                        );

                    SROW_FETCH_TIME.fetch_add(
                        row_start.elapsed().as_nanos() as u64,
                        Ordering::Relaxed,
                    );

                    match blobs_result {
                        Ok(b) => b,
                        Err(rusqlite::Error::QueryReturnedNoRows) => continue,
                        Err(e) => panic!("SQL query failed: {:?}", e),
                    }

                } else if n == 6 && sub_m == 5 {
                    let stmt: &mut Statement<'_> = &mut *prepared_stmt2;

                    let row_start = Instant::now();
                    let blobs_result: rusqlite::Result<(Vec<u8>, Vec<u8>)> =
                        stmt.query_row(
                            [&subcircuit.repr_blob()],
                            |row| Ok((row.get(0)?, row.get(1)?)),
                        );

                    SIXROW_FETCH_TIME.fetch_add(
                        row_start.elapsed().as_nanos() as u64,
                        Ordering::Relaxed,
                    );

                    match blobs_result {
                        Ok(b) => b,
                        Err(rusqlite::Error::QueryReturnedNoRows) => continue,
                        Err(e) => panic!("SQL query failed: {:?}", e),
                    }
                } else {
                    let table = format!("n{}m{}", n, sub_m);
                    let query = format!(
                        "SELECT perm, shuf FROM {} WHERE circuit = ?1 LIMIT 1",
                        table
                    );
                    let row_start = Instant::now();
                    let blobs_result: rusqlite::Result<(Vec<u8>, Vec<u8>)> =
                        conn.query_row(
                            &query,
                            [&subcircuit.repr_blob()],
                            |row| Ok((row.get(0)?, row.get(1)?)),
                        );

                    ROW_FETCH_TIME.fetch_add(
                        row_start.elapsed().as_nanos() as u64,
                        Ordering::Relaxed,
                    );

                    match blobs_result {
                        Ok(b) => {
                            println!("{}", table);
                            b
                        },
                        Err(rusqlite::Error::QueryReturnedNoRows) => continue,
                        Err(e) => panic!("SQL query failed: {:?}", e),
                    }
                }
            } else if sub_m <= max && (n >= 4) {
                let db_name = format!("n{}m{}perms", n, min);
                let db = match dbs.get(&db_name) {
                    Some(db) => *db,
                    None => continue,
                };

                let txn = env.begin_ro_txn().expect("lmdb ro txn");

                let row_start = Instant::now();
                let val = match txn.get(db, &subcircuit.repr_blob()) {
                    Ok(v) => v,
                    Err(lmdb::Error::NotFound) => continue,
                    Err(e) => panic!("LMDB get failed: {:?}", e),
                };
                LROW_FETCH_TIME.fetch_add(row_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

                let perm = val[..perm_len].to_vec();
                let shuf = val[perm_len..].to_vec();

                (perm, shuf)
            } else {
            // Permutation + canonicalization
            let perm_start = Instant::now();
            let sub_perm = subcircuit.permutation(n);
            PERMUTATION_TIME.fetch_add(perm_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

            let canon_start = Instant::now();
            let canon_perm = get_canonical(&sub_perm, bit_shuf);
            CANON_TIME.fetch_add(canon_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

            (canon_perm.perm.repr_blob(), canon_perm.shuffle.repr_blob())
        };

        let prefix = canon_perm_blob.as_slice();

        for smaller_m in 1..=min {
            let db_open_start = Instant::now();
            let db_name = format!("n{}m{}", n, smaller_m);
            let &db = match dbs.get(&db_name) {
                Some(db) => db,
                None => continue,
            };
            DB_OPEN_TIME.fetch_add(db_open_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

            let txn_start = Instant::now();
            let txn = env.begin_ro_txn().expect("txn");
            TXN_TIME.fetch_add(txn_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
            let lookup_start = Instant::now();
            let mut invert = false;
            let mut res = random_perm_lmdb(&txn, db, prefix);
            if res.is_none() {
                let prefix_inv_blob = Permutation::from_blob(&prefix).invert().repr_blob();
                invert = true;
                res = random_perm_lmdb(&txn, db, &prefix_inv_blob);
            }
            LMDB_LOOKUP_TIME.fetch_add(lookup_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
            

            if let Some(val_blob) = res {
                let from_blob_start = Instant::now();
                let mut repl = CircuitSeq::from_blob(&val_blob);
                FROM_BLOB_TIME.fetch_add(from_blob_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

                let rewire_start = Instant::now();
                if invert { repl.gates.reverse(); }
                repl.rewire(&Permutation::from_blob(&canon_shuf_blob).invert(), n);
                REWIRE_TIME.fetch_add(rewire_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

                let splice_start = Instant::now();
                if repl.gates.len() == end - start { 
                    compressed.gates[start..end].copy_from_slice(&repl.gates);
                } else {
                    compressed.gates.splice(start..end, repl.gates);
                }
                SPLICE_TIME.fetch_add(splice_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

                break;
            }
        }

        TRIAL_TIME.fetch_add(trial_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    // Final deduplication
    let dedup2_start = Instant::now();
    let mut j = 0;
    while j < compressed.gates.len().saturating_sub(1) {
        if compressed.gates[j] == compressed.gates[j + 1] {
            compressed.gates.drain(j..=j + 1);
            j = j.saturating_sub(2);
        } else {
            j += 1;
        }
    }
    DEDUP_TIME.fetch_add(dedup2_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

    compressed
}

pub fn expand_big(
    c: &CircuitSeq, 
    trials: usize, 
    num_wires: usize, 
    conn: &mut Connection, 
    env: &lmdb::Environment, 
    bit_shuf_list: &Vec<Vec<Vec<usize>>>, 
    dbs: &HashMap<String, lmdb::Database>,
) -> CircuitSeq {
    let table = format!("n{}m{}", 7, 4);
    let query_limit = format!("SELECT perm, shuf FROM {} WHERE circuit = ?1 LIMIT 1", table);
    let mut stmt = conn.prepare(&query_limit).unwrap();
    let table2 = format!("n{}m{}", 6, 5);
    let query_limit = format!("SELECT perm, shuf FROM {} WHERE circuit = ?1 LIMIT 1", table2);
    let mut stmt2 = conn.prepare(&query_limit).unwrap();
    let mut circuit = c.clone();
    let mut rng = rand::rng();

    for _i in 0..trials {
        // if i % 20 == 0 {
        //     println!("{} trials so far, {} more to go", i, trials - i);
        // }
        let mut subcircuit_gates = vec![];
        let random_max_wires = rng.random_range(3..=7);
        for set_size in (3..=7).rev() {
            let (gates, _) = simple_find_convex_subcircuit(set_size, random_max_wires, num_wires, &circuit, &mut rng);
            if !gates.is_empty() {
                subcircuit_gates = gates;
                break;
            }
        }

        if subcircuit_gates.is_empty() {
            return circuit
        }
        
        let mut gates: Vec<[u8;3]> = vec![[0,0,0]; subcircuit_gates.len()];
        for (i, g) in subcircuit_gates.iter().enumerate() {
            gates[i] = circuit.gates[*g];
        }

        subcircuit_gates.sort();
        let (start, end) = contiguous_convex(&mut circuit, &mut subcircuit_gates, num_wires).unwrap();
        let mut subcircuit = CircuitSeq { gates };
        // let sub_ref = subcircuit.clone();
        let expected_slice: Vec<_> = subcircuit_gates.iter().map(|&i| circuit.gates[i]).collect();
        let actual_slice = &circuit.gates[start..=end];

        if actual_slice != &expected_slice[..] {
            break;
        }

        let mut used_wires = subcircuit.used_wires();
        let n_wires = used_wires.len();
        let max = 7;
        let new_wires = rng.random_range(n_wires..=max);

        if new_wires > n_wires {
            let mut count = n_wires;
            while count < new_wires {
                let random = rng.random_range(0..num_wires);
                if used_wires.contains(&(random as u8)) {
                    continue
                }
                used_wires.push(random as u8);
                count += 1;
            }
        }
        used_wires.sort();
        subcircuit = CircuitSeq::rewire_subcircuit(&mut circuit, &mut subcircuit_gates, &used_wires);

        
        let bit_shuf = &bit_shuf_list[new_wires - 3];

        let subcircuit_temp = expand_lmdb(&subcircuit, 10, &bit_shuf, new_wires, &env, n_wires, dbs, &mut stmt, &mut stmt2, conn);
        subcircuit = subcircuit_temp;

        subcircuit = CircuitSeq::unrewire_subcircuit(&subcircuit, &used_wires);
        if subcircuit.gates.len() == end+1 - start {
            circuit.gates[start..end+1].copy_from_slice(&subcircuit.gates);
        } else {    
            circuit.gates.splice(start..end+1, subcircuit.gates);
        }
        // if c.permutation(num_wires).data != circuit.permutation(num_wires).data {
        //     panic!("splice changed something");
        // }
    }
    let mut i = 0;
    while i < circuit.gates.len().saturating_sub(1) {
        if circuit.gates[i] == circuit.gates[i + 1] {
            circuit.gates.drain(i..=i + 1);
            i = i.saturating_sub(2);
        } else {
            i += 1;
        }
    }
    circuit
}

// Old legacy code to obfuscate/inflate
pub fn obfuscate(c: &CircuitSeq, num_wires: usize) -> (CircuitSeq, Vec<usize>) {
    if c.gates.len() == 0 {
        return (CircuitSeq { gates: Vec::new() }, Vec::new() )
    }
    let mut obfuscated = CircuitSeq { gates: Vec::new() };
    let mut inverse_starts = Vec::new();

    let mut rng = rand::rng();

    // for butterfly
    let (r, r_inv) = random_id(num_wires as u8, rng.random_range(3..=25));

    for gate in &c.gates {
        // Generate a random identity r ⋅ r⁻¹
        // let (r, r_inv) = random_id(num_wires as u8, rng.random_range(3..=25), seed);

        // Add r
        obfuscated.gates.extend(&r.gates);

        // Record where r⁻¹ starts
        inverse_starts.push(obfuscated.gates.len());

        // Add r⁻¹
        obfuscated.gates.extend(&r_inv.gates);

        // Now add the original gate
        obfuscated.gates.push(*gate);
    }

    // Add a final padding random identity
    //let (r0, r0_inv) = random_id(num_wires as u8, rng.random_range(3..=5), seed);
    //obfuscated.gates.extend(&r0.gates);
    obfuscated.gates.extend(&r.gates);
    inverse_starts.push(obfuscated.gates.len());
    //obfuscated.gates.extend(&r0_inv.gates);
    obfuscated.gates.extend(&r_inv.gates);

    (obfuscated, inverse_starts)
}

// Expand as we compress to try and get more randomness in the butterfly methods
pub fn outward_compress(g: &CircuitSeq, r: &CircuitSeq, trials: usize, conn: &mut Connection, bit_shuf: &Vec<Vec<usize>>, n: usize) -> CircuitSeq {
    let mut g = g.clone();
    for gate in r.gates.iter() {
        let wrapper = CircuitSeq { gates: vec![*gate] };
        g = compress(&wrapper.concat(&g).concat(&wrapper), trials, conn, bit_shuf, n);
    }
    g
}

pub fn compress_big_ancillas(
    c: &CircuitSeq, 
    trials: usize, 
    num_wires: usize, 
    conn: &mut Connection, 
    env: &lmdb::Environment, 
    bit_shuf_list: &Vec<Vec<Vec<usize>>>, 
    dbs: &HashMap<String, lmdb::Database>, 
) -> CircuitSeq {
    let table = format!("n{}m{}", 7, 4);
    let query_limit = format!("SELECT perm, shuf FROM {} WHERE circuit = ?1 LIMIT 1", table);
    let mut stmt = conn.prepare(&query_limit).unwrap();
    let table = format!("n{}m{}", 6, 5);
    let query_limit = format!("SELECT perm, shuf FROM {} WHERE circuit = ?1 LIMIT 1", table);
    let mut stmt2 = conn.prepare(&query_limit).unwrap();
    let mut circuit = c.clone();
    let mut rng = rand::rng();

    let mut i = 0;
    while i < circuit.gates.len().saturating_sub(1) {
        if circuit.gates[i] == circuit.gates[i + 1] {
            circuit.gates.drain(i..=i + 1);
            i = i.saturating_sub(2);
        } else {
            i += 1;
        }
    }

    for _ in 0..trials {
        // let t0 = Instant::now();
        let mut subcircuit_gates = vec![];
        let random_max_wires = rng.random_range(3..=7);
        for set_size in (3..=6).rev() {
            let (gates, _) = simple_find_convex_subcircuit(set_size, random_max_wires, num_wires, &circuit, &mut rng);
            if !gates.is_empty() {
                subcircuit_gates = gates;
                break;
            }
        }
        // CONVEX_FIND_TIME.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

        if subcircuit_gates.is_empty() {
            continue;
        }

        let gates: Vec<[u8; 3]> = subcircuit_gates.iter().map(|&g| circuit.gates[g]).collect();
        subcircuit_gates.sort();

        // let t1 = Instant::now();
        let (start, end) = contiguous_convex(&mut circuit, &mut subcircuit_gates, num_wires).unwrap();
        // CONTIGUOUS_TIME.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let mut subcircuit = CircuitSeq { gates };

        let expected_slice: Vec<_> = subcircuit_gates.iter().map(|&i| circuit.gates[i]).collect();
        let actual_slice = &circuit.gates[start..=end];
        if actual_slice != &expected_slice[..] {
            continue;
        }

        // let t2 = Instant::now();
        let mut used_wires = subcircuit.used_wires();
        let n_wires = used_wires.len();
        let max = 7;
        let new_wires = rng.random_range(n_wires..=max);
        if new_wires > n_wires {
            let mut count = n_wires;
            while count < new_wires {
                let random = rng.random_range(0..num_wires);
                if used_wires.contains(&(random as u8)) {
                    continue
                }
                used_wires.push(random as u8);
                count += 1;
            }
        }
        // used_wires.sort();
        subcircuit = CircuitSeq::rewire_subcircuit(&mut circuit, &mut subcircuit_gates, &used_wires);
        // REWIRE_TIME.fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);

        // let t3 = Instant::now();
        let sub_num_wires = used_wires.len();
        let bit_shuf = &bit_shuf_list[sub_num_wires - 3];

        // PERMUTATION_TIME.fetch_add(t3.elapsed().as_nanos() as u64, Ordering::Relaxed);

        // let t4 = Instant::now();
        let subcircuit_temp = compress_lmdb(&subcircuit, 20, &bit_shuf, sub_num_wires, env, dbs, &mut stmt, &mut stmt2, conn);
        // COMPRESS_TIME.fetch_add(t4.elapsed().as_nanos() as u64, Ordering::Relaxed);

        subcircuit = subcircuit_temp;

        // let t5 = Instant::now();
        subcircuit = CircuitSeq::unrewire_subcircuit(&subcircuit, &used_wires);
        // UNREWIRE_TIME.fetch_add(t5.elapsed().as_nanos() as u64, Ordering::Relaxed);

        // let t6 = Instant::now();
        let repl_len = subcircuit.gates.len();
        let old_len = end - start + 1;

        if repl_len == old_len {
            for i in 0..repl_len {
                circuit.gates[start + i] = subcircuit.gates[i];
            }
        } else if repl_len < old_len {
            for i in 0..repl_len {
                circuit.gates[start + i] = subcircuit.gates[i];
            }
            for i in (end + 1)..circuit.gates.len() {
                circuit.gates[i - (old_len - repl_len)] = circuit.gates[i];
            }
            circuit.gates.truncate(circuit.gates.len() - (old_len - repl_len));
        } else {
            panic!("Replacement grew, which is not allowed");
        }
        // REPLACE_TIME.fetch_add(t6.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    // let t7 = Instant::now();
    let mut i = 0;
    while i < circuit.gates.len().saturating_sub(1) {
        if circuit.gates[i] == circuit.gates[i + 1] {
            circuit.gates.drain(i..=i + 1);
            i = i.saturating_sub(2);
        } else {
            i += 1;
        }
    }
    // DEDUP_TIME.fetch_add(t7.elapsed().as_nanos() as u64, Ordering::Relaxed);

    circuit
}

// Replace a single gate
pub fn random_gate_replacements(c: &mut CircuitSeq, x: usize, n: usize, _conn: &Connection, env: &lmdb::Environment) {
    let mut rng = rand::rng();
    for _ in 0..x {
        if c.gates.is_empty() {
            break;
        }

        let i = rng.random_range(0..c.gates.len());
        let g = &c.gates[i];

        let num = rng.random_range(3..=7);
        if let Ok(mut id) = random_canonical_id(env, &_conn, num) {
            let mut used_wires = vec![g[0], g[1], g[2]];
            let mut count = 3;
            while count < num {
                let random = rng.random_range(0..n);
                if used_wires.contains(&(random as u8)) {
                    continue
                }
                used_wires.push(random as u8);
                count += 1;
            }
            used_wires.sort();
            let rewired_g = CircuitSeq::rewire_subcircuit(&c, &vec![i], &used_wires);
            // println!("rewired_g {:?} vs len: {}", rewired_g, num);
            id.rewire_first_gate(rewired_g.gates[0], num);
            id = CircuitSeq::unrewire_subcircuit(&id, &used_wires);
            id.gates.remove(0);
            c.gates.splice(i..i+1, id.gates);
        } 
    }
}

// For timing and benchmarking purposes
pub fn print_compress_timers() {
    let perm = PERMUTATION_TIME.load(Ordering::Relaxed);
    let sql = SQL_TIME.load(Ordering::Relaxed);
    let canon = CANON_TIME.load(Ordering::Relaxed);
    let compress = COMPRESS_TIME.load(Ordering::Relaxed);
    let rewire = REWIRE_TIME.load(Ordering::Relaxed);
    let unrewire = UNREWIRE_TIME.load(Ordering::Relaxed);
    let convex_find = CONVEX_FIND_TIME.load(Ordering::Relaxed);
    let contiguous = CONTIGUOUS_TIME.load(Ordering::Relaxed);
    let replace = REPLACE_TIME.load(Ordering::Relaxed);
    let dedup = DEDUP_TIME.load(Ordering::Relaxed);
    let pick = PICK_SUBCIRCUIT_TIME.load(Ordering::Relaxed);
    let canonicalize = CANONICALIZE_TIME.load(Ordering::Relaxed);
    let row_fetch = ROW_FETCH_TIME.load(Ordering::Relaxed);
    let srow_fetch = SROW_FETCH_TIME.load(Ordering::Relaxed);
    let sixrow_fetch = SIXROW_FETCH_TIME.load(Ordering::Relaxed);
    let lrow_fetch = LROW_FETCH_TIME.load(Ordering::Relaxed);
    let db_open = DB_OPEN_TIME.load(Ordering::Relaxed);
    let txn = TXN_TIME.load(Ordering::Relaxed);
    let lmdb_lookup = LMDB_LOOKUP_TIME.load(Ordering::Relaxed);
    let from_blob = FROM_BLOB_TIME.load(Ordering::Relaxed);
    let splice = SPLICE_TIME.load(Ordering::Relaxed);
    let trial = TRIAL_TIME.load(Ordering::Relaxed);
    let id = IDENTITY_TIME.load(Ordering::Relaxed);

    println!("--- Compression Timing Totals (minutes) ---");
    println!("Permutation computation time: {:.2} min", perm as f64 / 60_000_000_000.0);
    println!("SQL lookup time: {:.2} min", sql as f64 / 60_000_000_000.0);
    println!("Canonicalization time: {:.2} min", canon as f64 / 60_000_000_000.0);
    println!("Compress LMDB time: {:.2} min", compress as f64 / 60_000_000_000.0);
    println!("Rewire subcircuit time: {:.2} min", rewire as f64 / 60_000_000_000.0);
    println!("Unrewire subcircuit time: {:.2} min", unrewire as f64 / 60_000_000_000.0);
    println!("Convex subcircuit find time: {:.2} min", convex_find as f64 / 60_000_000_000.0);
    println!("Contiguous convex subcircuit time: {:.2} min", contiguous as f64 / 60_000_000_000.0);
    println!("Replacement time: {:.2} min", replace as f64 / 60_000_000_000.0);
    println!("Deduplication time: {:.2} min", dedup as f64 / 60_000_000_000.0);
    println!("Pick subcircuit time: {:.2} min", pick as f64 / 60_000_000_000.0);
    println!("Subcircuit canonicalize time: {:.2} min", canonicalize as f64 / 60_000_000_000.0);
    println!("SQL row fetch time: {:.2} min", row_fetch as f64 / 60_000_000_000.0);
    println!("SQL n7m4 prepared row fetch time: {:.2} min", srow_fetch as f64 / 60_000_000_000.0);
    println!("SQL n6m5 prepared row fetch time: {:.2} min", sixrow_fetch as f64 / 60_000_000_000.0);
    println!("LMDB row fetch time: {:.2} min", lrow_fetch as f64 / 60_000_000_000.0);
    println!("LMDB DB open time: {:.2} min", db_open as f64 / 60_000_000_000.0);
    println!("LMDB transaction begin time: {:.2} min", txn as f64 / 60_000_000_000.0);
    println!("LMDB lookup time: {:.2} min", lmdb_lookup as f64 / 60_000_000_000.0);
    println!("CircuitSeq from_blob time: {:.2} min", from_blob as f64 / 60_000_000_000.0);
    println!("Gate splice time: {:.2} min", splice as f64 / 60_000_000_000.0);
    println!("Trial loop time: {:.2} min", trial as f64 / 60_000_000_000.0);
    println!("Identity Sampling Time: {:.2} min", id as f64 / 60_000_000_000.0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use lmdb::Cursor;
    use rusqlite::Connection;
    use std::time::{Instant};
    use crate::random::random_data::random_circuit;
    use itertools::Itertools;
    #[test]
    fn random_circuit_exists_in_db() {
        // Open the SQLite DB
        let conn = Connection::open("circuits.db").expect("Failed to open DB");

        let perms: Vec<Vec<usize>> = (0..5).permutations(5).collect();
        let bit_shuf = perms.into_iter().skip(1).collect::<Vec<_>>();

        let n = 5;
        let len = 4;

        // Generate a random circuit of length 4
        let c = random_circuit(n, len);
        println!("Random circuit: {:?}", c.gates);

        // Compute its permutation and canonical form
        let perm = c.permutation(n as usize);
        let canon = perm.canon_simple(&bit_shuf);
        let perm_blob = canon.perm.repr_blob();

        let mut found = false;

        // Check tables for lengths 1..=len
        for m in 1..=len {
            let table = format!("n{}m{}", n, m);
            let query = format!("SELECT COUNT(*) FROM {} WHERE perm = ?1", table);

            if let Ok(count) =
                conn.query_row(&query, [perm_blob.as_slice()], |row| row.get::<_, i64>(0))
            {
                if count > 0 {
                    println!("Found permutation in table {}!", table);
                    found = true;
                    break;
                }
            }
        }

        // Assert that the permutation exists in at least one table
        assert!(found, "Permutation not found in any table!");
    }
    use std::fs;
    use std::fs::File;
    use lmdb::Environment;
    use std::path::Path;
    use std::io::Write;
    use crate::replace::main_mix::open_all_dbs;
    #[test]
    fn test_compression_big_time() {
        // let total_start = Instant::now();

        // // ---------- FIRST TEST ----------
        // let t1_start = Instant::now();
        // let n = 64;
        // let str1 = "circuitQQF_64.txt";
        // let data1 = fs::read_to_string(str1).expect("Failed to read circuitQQF_64.txt");
        // let mut stable_count = 0;
        // let mut conn = Connection::open("circuits.db").expect("Failed to open DB");
        // let mut acc = CircuitSeq::from_string(&data1);
        // while stable_count < 3 {
        //     let before = acc.gates.len();
        //     acc = compress_big(&acc, 1_000, n, &mut conn);
        //     let after = acc.gates.len();

        //     if after == before {
        //         stable_count += 1;
        //         println!("  Final compression stable {}/3 at {} gates", stable_count, after);
        //     } else {
        //         println!("  Final compression reduced: {} → {} gates", before, after);
        //         stable_count = 0;
        //     }
        // }
        // let t1_duration = t1_start.elapsed();
        // println!(" First compression finished in {:.2?}", t1_duration);

        // ---------- SECOND TEST ----------
        let t2_start = Instant::now();
        let str2 = "compressed.txt";
        let lmdb = "./db";
            let _ = std::fs::create_dir_all(lmdb);

            let env = Environment::new()
                .set_max_readers(10000) 
                .set_max_dbs(262)      
                .set_map_size(700 * 1024 * 1024 * 1024) 
                .open(Path::new(lmdb))
                .expect("Failed to open lmdb");

        let data2 = fs::read_to_string(str2).expect("Failed to read circuitF.txt");
        let mut stable_count = 0;
        let conn = Connection::open("circuits.db").expect("Failed to open DB");
        let acc = CircuitSeq::from_string(&data2);
        let _bit_shuf_list: Vec<Vec<Vec<usize>>> = (3..=7)
        .map(|n| {
            (0..n)
                .permutations(n)
                .filter(|p| !p.iter().enumerate().all(|(i, &x)| i == x))
                .collect::<Vec<Vec<usize>>>()
        })
        .collect();
        let _dbs = open_all_dbs(&env);
        let mut stmts_prepared = HashMap::new();
        let mut stmts_prepared_limit1 = HashMap::new();
        let ns_and_ms = vec![(3, 10), (4, 6), (5, 5), (6, 5), (7, 4)];
        for &(n, max_m) in &ns_and_ms {
            for m in 1..=max_m {
                let table = format!("n{}m{}", n, m);
                let query = format!("SELECT perm, shuf FROM {} WHERE circuit = ?", table);
                let stmt = conn.prepare(&query).unwrap();
                stmts_prepared.insert((n, m), stmt);

                let query_limit = format!("SELECT perm, shuf FROM {} WHERE circuit = ?1 LIMIT 1", table);
                let stmt_limit = conn.prepare(&query_limit).unwrap();
                stmts_prepared_limit1.insert((n, m), stmt_limit);
            }
        }
        let _conn = Connection::open("circuits.db").expect("Failed to open DB");
        while stable_count < 6 {
            let before = acc.gates.len();
            // acc = compress_big(&acc, 1_000, 64, &mut conn, &env, &bit_shuf_list, &dbs);
            let after = acc.gates.len();

            if after == before {
                stable_count += 1;
                println!("  Final compression stable {}/6 at {} gates", stable_count, after);
            } else {
                println!("  Final compression reduced: {} → {} gates", before, after);
                stable_count = 0;
            }
        }

        File::create("compressed.txt")
        .and_then(|mut f| f.write_all(acc.repr().as_bytes()))
        .expect("Failed to write butterfly_recent.txt");
        let t2_duration = t2_start.elapsed();
        println!(" Second compression finished in {:.2?}", t2_duration);

        // ---------- TOTAL ----------
        // let total_duration = total_start.elapsed();
        // println!(" Total test duration: {:.2?}", total_duration);
    }

    #[test]
    fn test_random_canon_id() {
        let env = Environment::new()
                .set_max_readers(10000) 
                .set_max_dbs(262)      
                .set_map_size(700 * 1024 * 1024 * 1024) 
                .open(Path::new("./db"))
                .expect("Failed to open lmdb");
        let conn = Connection::open("circuits.db").expect("Failed to open DB");
        let circuit = random_canonical_id(&env, &conn, 3).unwrap_or_else(|_| panic!("Failed to run random_canon_id"));
        if circuit.probably_equal(&CircuitSeq { gates: vec![[1,2,3], [1,2,3]]}, 10, 10000).is_err() {
            panic!("Not id");
        }
        println!("circuit {:?}", circuit.gates);
    }

    #[test]
    fn print_lmdb_keys() -> Result<(), Box<dyn std::error::Error>> {
        let env_path = "./db";
        let db_name = "perm_tables_n6";

        let env = Environment::new()
            .set_max_dbs(262)
            .open(Path::new(env_path))?;

        let db = env.open_db(Some(db_name))?;

        let txn = env.begin_ro_txn()?;
        let mut cursor = txn.open_ro_cursor(db)?;
        for (_, value) in cursor.iter() {
            println!("{:?}", value); 
        }

        Ok(())
    }

    #[test]
    fn test_find_perm_lmdb() {
        let perm = Permutation { data: vec![3, 2, 5, 4, 7, 6, 1, 0, 11, 10, 13, 12, 15, 14, 9, 8, 19, 18, 21, 20, 23, 22, 17, 16, 27, 26, 29, 28, 31, 30, 25, 24, 37, 36, 35, 34, 33, 32, 39, 38, 43, 42, 45, 44, 47, 46, 41, 40, 53, 52, 51, 50, 49, 48, 55, 54, 59, 58, 61, 60, 63, 62, 57, 56, 71, 70, 68, 69, 67, 66, 64, 65, 79, 78, 76, 77, 75, 74, 72, 73, 87, 86, 84, 85, 83, 82, 80, 81, 95, 94, 92, 93, 91, 90, 88, 89, 100, 101, 103, 102, 96, 97, 99, 98, 111, 110, 108, 109, 107, 106, 104, 105, 116, 117, 119, 118, 112, 113, 115, 114, 127, 126, 124, 125, 123, 122, 120, 121]};
        let prefix = perm.repr_blob();
        let env_path = "./db";
        let db_name = "n4m2";
        let env = Environment::new()
            .set_max_dbs(262)
            .open(Path::new(env_path)).expect("Failed to open db");
        let db = env.open_db(Some(&db_name))
                .unwrap_or_else(|e| panic!("LMDB DB '{}' failed to open: {:?}", db_name, e));
        let txn = env.begin_ro_txn()
                .unwrap_or_else(|e| panic!("Failed to begin RO txn on '{}': {:?}", "perm_db_name", e));
        let mut cursor = txn.open_ro_cursor(db).ok().expect("Failed to open cursor");
        let mut circuits = Vec::new();
        let mut count = 0;
        for (key, _) in cursor.iter() {
            if key.starts_with(&prefix) {
                circuits.push(key[prefix.len()..].to_vec());
                count += 1;
                println!("count: {}", count);
            }
        }
    }

    use crate::replace::mixing::split_into_random_chunks;
    use rayon::iter::IntoParallelIterator;
    use rayon::iter::ParallelIterator;
    use rusqlite::OpenFlags;
    #[test]
    fn replace_sequential_pair_preserves_invariants() {
        use crate::replace::pairs::replace_sequential_pairs;
        use rand::{SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(0xdeadbeef);
        let num_wires = 64;
        let env_path = "./db";
        let _conn = Connection::open("circuits.db").expect("Failed to open DB");
        let env = Environment::new()
            .set_max_dbs(262)
            .open(Path::new(env_path)).expect("Failed to open db");
        let data2 = fs::read_to_string("./tempcirc.txt").expect("Failed to read circuitF.txt");
        let mut circuit = CircuitSeq::from_string(&data2);
        let out_circ = circuit.clone();
        let bit_shuf_list = (3..=7)
        .map(|n| {
            (0..n)
                .permutations(n)
                .filter(|p| !p.iter().enumerate().all(|(i, &x)| i == x))
                .collect::<Vec<Vec<usize>>>()
        })
        .collect();
        let dbs = open_all_dbs(&env);
        let chunks = split_into_random_chunks(&circuit.gates, 10, &mut rng);
        static TOTAL_TIME: AtomicU64 = AtomicU64::new(0);
        // Call under test
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
                let t0 = Instant::now();
                let (_, _, _, _) = replace_sequential_pairs(&mut sub, 64, &mut thread_conn, &env, &bit_shuf_list, &dbs, false);
                sub.gates.reverse();
                let (_, _, _, _) = replace_sequential_pairs(&mut sub, 64, &mut thread_conn, &env, &bit_shuf_list, &dbs, false);
                sub.gates.reverse();
                TOTAL_TIME.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                sub.gates
            })
            .collect();
            let new_gates: Vec<[u8;3]> = replaced_chunks.into_iter().flatten().collect();
            circuit.gates = new_gates;

        if circuit.probably_equal(&out_circ, num_wires, 100_000).is_err() {
            panic!("Functionality was changed");
        }

        let tt = TOTAL_TIME.load(Ordering::Relaxed);

        println!("Permutation computation time: {:.2} min", tt as f64 / 60_000_000_000.0);
        println!("All good");
        print_compress_timers();
        // No invalid wire indices
        for (i, gate) in circuit.gates.iter().enumerate() {
            for &w in gate {
                assert!(
                    (w as usize) < num_wires,
                    "gate {} contains wire {} >= num_wires {}",
                    i,
                    w,
                    num_wires
                );
            }
        }
    }

    #[test]
    fn test_update_dist() {
        use crate::replace::pairs::update_distance;

        let mut d = vec![0, 1, 1, 0];
        update_distance(&mut d, 1, 6);
        assert_eq!(d, vec![0, 1, 2, 3, 3, 2, 1, 0]);
    }

    #[test]
    fn test_gen_id_speeds() {
        use crate::replace::pairs::gate_pair_taxonomy;
        use crate::replace::identities::get_random_identity;
        // stress / invariant check
        let _n = 64;
        let w = 7;
        let env_path = "./db";
        
        let env = Environment::new()
            .set_max_dbs(262)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new(env_path))
            .expect("Failed to open lmdb");
        let dbs = open_all_dbs(&env);

        for _ in 0..10_000_000 {
            let c = random_circuit(64, 2);
            let tax = gate_pair_taxonomy(&c.gates[0], &c.gates[1]);
            let id = get_random_identity(w, tax, &env, &dbs, false);
            println!("{:?}", id.unwrap().gates);
        }
        let ns_to_min = |v: u64| v as f64 / (60.0 * 1_000_000_000.0);
        println!("\n=== get_random_identity timers ===");

        // println!("DB_NAME_TIME          : {:.6}", ns_to_min(DB_NAME_TIME.load(Ordering::Relaxed)));
        // println!("DB_LOOKUP_TIME        : {:.6}", ns_to_min(DB_LOOKUP_TIME.load(Ordering::Relaxed)));
        // println!("TXN_BEGIN_TIME        : {:.6}", ns_to_min(TXN_BEGIN_TIME.load(Ordering::Relaxed)));
        // println!("SERIALIZE_KEY_TIME    : {:.6}", ns_to_min(SERIALIZE_KEY_TIME.load(Ordering::Relaxed)));
        // println!("LMDB_GET_TIME         : {:.6}", ns_to_min(LMDB_GET_TIME.load(Ordering::Relaxed)));
        // println!("DESERIALIZE_LIST_TIME : {:.6}", ns_to_min(DESERIALIZE_LIST_TIME.load(Ordering::Relaxed)));
        // println!("RNG_CHOOSE_TIME       : {:.6}", ns_to_min(RNG_CHOOSE_TIME.load(Ordering::Relaxed)));
        println!("FROM_BLOB_TIME        : {:.6}", ns_to_min(FROM_BLOB_TIME.load(Ordering::Relaxed)));

        println!("=================================\n");
    }

    fn gen_mean(circuit: CircuitSeq, num_wires: usize) -> f64 {
        let circuit_one = circuit.clone();
        let circuit_two = circuit;

        let circuit_one_len = circuit_one.gates.len();
        let circuit_two_len = circuit_two.gates.len();

        let num_points = (circuit_one_len + 1) * (circuit_two_len + 1);
        let mut average = vec![0f64; num_points * 3];

        let mut rng = rand::rng();
        let num_inputs = 20;

        for _ in 0..num_inputs {
            // if i % 10 == 0 {
            //     // println!("{}/{}", i, num_inputs);
            //     io::stdout().flush().unwrap();
            // }

            let input_bits: u128 = if num_wires < u128::BITS as usize {
                rng.random_range(0..(1u128 << num_wires))
            } else {
                rng.random_range(0..=u128::MAX)
            };

            let evolution_one = circuit_one.evaluate_evolution_128(input_bits);
            let evolution_two = circuit_two.evaluate_evolution_128(input_bits);

            for i1 in 0..=circuit_one_len {
                for i2 in 0..=circuit_two_len {
                    let diff = evolution_one[i1] ^ evolution_two[i2];
                    let hamming_dist = diff.count_ones() as f64;
                    let overlap = hamming_dist / num_wires as f64;

                    let index = i1 * (circuit_two_len + 1) + i2;
                    average[index * 3] = i1 as f64;
                    average[index * 3 + 1] = i2 as f64;
                    average[index * 3 + 2] += overlap / num_inputs as f64;
                }
            }
        }

        let mut sum = 0.0;
        for i in 0..num_points {
            sum += average[i * 3 + 2];
        }

        sum / num_points as f64
    }

    #[test]
    pub fn test_gen_id_16() {
        use crate::replace::identities::get_random_wide_identity;
        let env_path = "./db";
        let mut thread_conn = Connection::open_with_flags(
                    "circuits.db",
                    OpenFlags::SQLITE_OPEN_READ_ONLY,
                )
                .expect("Failed to open read-only connection");
        let env = Environment::new()
            .set_max_dbs(200)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new(env_path))
            .expect("Failed to open lmdb");
        let bit_shuf_list = (3..=7)
        .map(|n| {
            (0..n)
                .permutations(n)
                .filter(|p| !p.iter().enumerate().all(|(i, &x)| i == x))
                .collect::<Vec<Vec<usize>>>()
        })
        .collect();
        let dbs = open_all_dbs(&env);
        let mut count = 0;
        while count < 2 {
            let id = get_random_wide_identity(16, &env, &dbs, &mut thread_conn, &bit_shuf_list, false);

            assert!(
                id.probably_equal(&CircuitSeq { gates: Vec::new() }, 16, 100_000).is_ok(),
                "Not an identity"
            );

            if gen_mean(id.clone(), 16) < 0.33 {
                continue
            }

            // write repr() to file
            let mut file = File::create(format!("id_16{}.txt", count))
                .expect("Failed to create output file");
            writeln!(file, "{}", id.repr()).expect("Failed to write repr");

            // wire statistics
            let mut wires: HashMap<u8, Vec<usize>> = HashMap::new();
            for (i, gates) in id.gates.iter().enumerate() {
                for &pins in gates {
                    wires.entry(pins).or_insert_with(Vec::new).push(i);
                }
            }

            println!("Run {}", count);
            for (k, v) in &wires {
                println!("wire: {}, # of gates: {}", k, v.len());
            }
            println!("Num wires: {}\n", wires.len());
            count += 1;
        }
    }

    #[test]
    pub fn test_max_mean_16() {
        use crate::replace::identities::get_random_wide_identity;
        let env_path = "./db";
        let mut thread_conn = Connection::open_with_flags(
                    "circuits.db",
                    OpenFlags::SQLITE_OPEN_READ_ONLY,
                )
                .expect("Failed to open read-only connection");
        let env = Environment::new()
            .set_max_dbs(200)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new(env_path))
            .expect("Failed to open lmdb");
        let bit_shuf_list = (3..=7)
        .map(|n| {
            (0..n)
                .permutations(n)
                .filter(|p| !p.iter().enumerate().all(|(i, &x)| i == x))
                .collect::<Vec<Vec<usize>>>()
        })
        .collect();
        let dbs = open_all_dbs(&env);
        let mut curr_mean = 0.0;
        loop {
            let id = get_random_wide_identity(128, &env, &dbs, &mut thread_conn, &bit_shuf_list, true);

            assert!(
                id.probably_equal(&CircuitSeq { gates: Vec::new() }, 128, 100_000).is_ok(),
                "Not an identity"
            );
            let mean = gen_mean(id.clone(), 128);
            if mean < curr_mean {
                continue
            }
            curr_mean = mean;

            // write repr() to file
            let mut file = File::create(format!("id_16currmean.txt"))
                .expect("Failed to create output file");
            writeln!(file, "{}", id.repr()).expect("Failed to write repr");

            // wire statistics
            let mut wires: HashMap<u8, Vec<usize>> = HashMap::new();
            for (i, gates) in id.gates.iter().enumerate() {
                for &pins in gates {
                    wires.entry(pins).or_insert_with(Vec::new).push(i);
                }
            }
            for (k, v) in &wires {
                println!("wire: {}, # of gates: {}", k, v.len());
            }
            println!("Num wires: {}\n", wires.len());
            println!("Curr mean: {}", curr_mean);
        }
    }
}