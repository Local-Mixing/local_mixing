use crate::{
    circuit::circuit::{CircuitSeq, Permutation},
    random::random_data::{
        contiguous_convex, find_convex_subcircuit, get_canonical, 
        random_circuit,
    },
    rainbow::canonical::Canonicalization
};
use itertools::Itertools;
use rand::{Rng};
use rusqlite::{Connection};
use std::{
    cmp::{max, min},
    collections::{HashSet},
    // fs::OpenOptions, // used for testing
    // io::Write,
    // sync::Arc,
    time::{Instant},
};
use std::sync::atomic::Ordering;
use std::sync::atomic::AtomicU64;
use lmdb::{Cursor, Database, Transaction, RoTransaction};
// Returns a nontrivial identity circuit built from two "friend" circuits
pub fn random_canonical_id(
    conn: &Connection,
    wires: usize,
) -> Result<CircuitSeq, Box<dyn std::error::Error>> {
    // Pattern to match all tables for this wire count (all m values)
    let pattern = format!("n{}%m%", wires);

    // Get all tables matching pattern
    let tables: Vec<String> = conn
        .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?1")?
        .query_map([&pattern], |row| row.get(0))?
        .map(|r| r.unwrap())
        .collect();

    // Need at least 2 tables to combine circuits
    if tables.len() < 2 {
        return Err(format!("Need at least 2 tables matching {}", pattern).into());
    }

    let mut rng = rand::rng();

    loop {
        // Pick two random tables
        let table_a = tables[rng.random_range(2..tables.len())].clone();
        let table_b = tables[rng.random_range(2..tables.len())].clone();

        let (small, large) = if table_a < table_b {
            (table_a, table_b)
        } else {
            (table_b, table_a)
        };

        let count_small: i64 = conn.query_row(
            &format!("SELECT MAX(rowid) FROM {}", small),
            [],
            |row| row.get(0),
        )?;
        if count_small == 0 {
            continue; // empty table, retry
        }

        let random_id = rng.random_range(1..=count_small);
        let (circuit_blob, perm_blob, shuf_blob): (Vec<u8>, Vec<u8>, Vec<u8>) = conn.query_row(
            &format!("SELECT circuit, perm, shuf FROM {} WHERE rowid = ?", small),
            [random_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        )?;

        let count_large: i64 = conn.query_row(
            &format!("SELECT COUNT(*) FROM {} WHERE perm = ?", large),
            [&perm_blob],
            |row| row.get(0),
        )?;

        if count_large == 0 {
            continue; // not in both tables, retry
        }

        // Pick a random offset
        let offset = rng.random_range(0..count_large);

        let b_blob: [Vec<u8>; 2] = conn.query_row(
            &format!(
                "SELECT circuit, shuf FROM {} WHERE perm = ? LIMIT 1 OFFSET ?",
                large
            ),
            rusqlite::params![&perm_blob, offset],
            |row| Ok([row.get(0)?, row.get(1)?]),
        )?;

        if b_blob[0] == circuit_blob {
            continue;
        }

        // Deserialize circuits
        let mut ca = CircuitSeq::from_blob(&circuit_blob);
        let mut cb = CircuitSeq::from_blob(&b_blob[0]);

        // Rewire cb to align with ca
        cb.rewire(&Permutation::from_blob(&b_blob[1]), wires);
        cb.rewire(&Permutation::from_blob(&shuf_blob).invert(), wires);

        // Reverse cb and append to ca
        cb.gates.reverse();
        ca.gates.extend(cb.gates);

        // Return the combined circuit
        return Ok(ca);
    }
}

// To just get a completely random circuit and reverse for identity, rather than using canonical ones from our rainbow table
pub fn random_id(n: u8, m: usize) -> (CircuitSeq, CircuitSeq) {
    let circuit = random_circuit(n, m);

    // Preallocate reversed gates so we don't need to run through circuit twice
    let mut rev_gates = Vec::with_capacity(circuit.gates.len());
    for g in circuit.gates.iter().rev() {
        rev_gates.push(*g); // copy [u8;3]
    }

    let rev = CircuitSeq { gates: rev_gates };
    (circuit, rev)
}

// Return a random subcircuit, its starting index (gate), and ending index
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

    // let len = circuit.gates.len();
    
    // if len == 0 {
    //     return (CircuitSeq { gates: Vec::new() }, 0, 0);
    // }

    // let mut rng = rand::rng();

    // // Pick a random start index
    // let start = rng.random_range(0..len);

    // // Maximum subcircuit length is 8, but can't go past end of circuit
    // let max_len = 8.min(len - start);

    // // Pick random length from 1..=max_len
    // let sub_len = rng.random_range(1..=max_len);

    // let end = start + sub_len;

    // let subcircuit = circuit.gates[start..end].to_vec();

    // (CircuitSeq { gates: subcircuit }, start, end)
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

static PERMUTATION_TIME: AtomicU64 = AtomicU64::new(0);
static SQL_TIME: AtomicU64 = AtomicU64::new(0);
static CANON_TIME: AtomicU64 = AtomicU64::new(0);
static CONVEX_FIND_TIME: AtomicU64 = AtomicU64::new(0);
static CONTIGUOUS_TIME: AtomicU64 = AtomicU64::new(0);
static REWIRE_TIME: AtomicU64 = AtomicU64::new(0);
static COMPRESS_TIME: AtomicU64 = AtomicU64::new(0);
static UNREWIRE_TIME: AtomicU64 = AtomicU64::new(0);
static REPLACE_TIME: AtomicU64 = AtomicU64::new(0);
static DEDUP_TIME: AtomicU64 = AtomicU64::new(0);

pub fn compress(
    c: &CircuitSeq,
    trials: usize,
    conn: &mut Connection,
    bit_shuf: &Vec<Vec<usize>>,
    n: usize,
) -> CircuitSeq {

    let id = Permutation::id_perm(n);

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

            let sql_t0 = Instant::now();
            let mut stmt = match conn.prepare(&query) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let rows = stmt.query([&subcircuit.repr_blob()]);
            SQL_TIME.fetch_add(sql_t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

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
            let t1 = Instant::now();
            let sub_perm = subcircuit.permutation(n);
            PERMUTATION_TIME.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);

            let t2 = Instant::now();
            let canon_perm = get_canonical(&sub_perm, bit_shuf);
            CANON_TIME.fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);

            (canon_perm.perm.repr_blob(), canon_perm.shuffle.repr_blob())
        };

        for smaller_m in 1..=sub_m {
            let table = format!("n{}m{}", n, smaller_m);
            let query = format!(
                "SELECT * FROM {} WHERE perm = ?1 ORDER BY RANDOM() LIMIT 1",
                table
            );

            let sql_t0 = Instant::now();
            let mut stmt = match conn.prepare(&query) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let rows = stmt.query([&canon_perm_blob]);
            SQL_TIME.fetch_add(sql_t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

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
                    
                    // TODO: !!! Fix all of this
                    repl.rewire(&Permutation::from_blob(&canon_shuf_blob).invert(), n);

                    // let t5 = Instant::now();
                    // let final_check = repl.permutation(n);
                    // PERMUTATION_TIME.fetch_add(t5.elapsed().as_nanos() as u64, Ordering::Relaxed);

                    // let sub_perm = Permutation::from_blob(&canon_perm_blob).bit_shuffle(&Permutation::from_blob(&canon_shuf_blob).invert().data);

                    // if final_check != sub_perm {
                    //     panic!("Replacement permutation mismatch!");
                    // }

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

pub fn expand(
    c: &CircuitSeq,
    trials: usize,
    conn: &mut Connection,
    bit_shuf: &Vec<Vec<usize>>,
    n: usize,
) -> CircuitSeq {
    let mut expanded = c.clone();
    if expanded.gates.is_empty() {
        return CircuitSeq { gates: Vec::new() };
    }
    
    let max = if n == 7 {
        4
    } else if n == 6 || n == 5{
        5
    } else if n == 4 {
        6
    } else if n == 3 {
        7
    } else {
        0
    };

    for _ in 0..trials {
        let (mut subcircuit, start, end) = random_subcircuit_max(&expanded, max-1);
        if subcircuit.gates.len() >= max {
            break;
        }
        subcircuit.canonicalize();

        let sub_perm = subcircuit.permutation(n);
        let canon_perm = get_canonical(&sub_perm, &bit_shuf);

        let perm_blob = canon_perm.perm.repr_blob();
        let sub_m = subcircuit.gates.len();

        for smaller_m in (sub_m..=std::cmp::min(sub_m+2,max)).rev() {
            let table = format!("n{}m{}", n, smaller_m);
            let query = format!(
                "SELECT circuit FROM {} WHERE perm = ?1 ORDER BY RANDOM() LIMIT 1",
                table
            );

            let mut stmt = match conn.prepare(&query) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let rows = stmt.query([&perm_blob]);

            if let Ok(mut r) = rows {
                if let Some(row) = r.next().unwrap() {
                    let blob: Vec<u8> = row.get(0).expect("Failed to get blob");
                    let mut repl = CircuitSeq::from_blob(&blob);

                    if repl.gates.len() >= subcircuit.gates.len() {
                        let repl_perm = repl.permutation(n);
                        let rc = get_canonical(&repl_perm, &bit_shuf);

                        if !rc.shuffle.data.is_empty() {
                            repl.rewire(&rc.shuffle, n);
                        }
                        repl.rewire(&canon_perm.shuffle.invert(), n);

                        if repl.permutation(n) != sub_perm {
                            panic!("Replacement permutation mismatch!");
                        }

                        expanded.gates.splice(start..end, repl.gates);

                        break;
                    }
                }
            }
        }
    }

    expanded
}

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

pub fn compress_big(c: &CircuitSeq, trials: usize, num_wires: usize, conn: &mut Connection, env: &lmdb::Environment) -> CircuitSeq {
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
        let t0 = Instant::now();
        let mut subcircuit_gates = vec![];
        for set_size in (3..=6).rev() {
            let random_max_wires = rng.random_range(3..=7);
            let (gates, _) = find_convex_subcircuit(set_size, random_max_wires, num_wires, &circuit, &mut rng);
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
        let perms: Vec<Vec<usize>> = (0..sub_num_wires).permutations(sub_num_wires).collect();
        let bit_shuf = perms.into_iter().skip(1).collect::<Vec<_>>();
        PERMUTATION_TIME.fetch_add(t3.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t4 = Instant::now();
        let subcircuit_temp = compress_lmdb(&subcircuit, 20, conn, &bit_shuf, sub_num_wires, env);
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

fn random_perm_lmdb<'a>(
    txn: &'a RoTransaction,
    db: Database,
    prefix: &[u8],
) -> Option<(Vec<u8>, Vec<u8>)> {
    if txn.get(db, &prefix).is_err() {
        return None;
    }
    let mut cursor = txn.open_ro_cursor(db).ok()?;

    let mut matches = Vec::new();

    for (key, val) in cursor.iter_from(prefix) {
        if !key.starts_with(prefix) {
            break;
        }
        matches.push((key.to_vec(), val.to_vec()));
    }

    if matches.is_empty() {
        return None;
    }

    let idx = rand::rng().random_range(0..matches.len());

    Some(matches.swap_remove(idx))
}

pub fn compress_lmdb(
    c: &CircuitSeq,
    trials: usize,
    conn: &mut Connection,
    bit_shuf: &Vec<Vec<usize>>,
    n: usize,
    env: &lmdb::Environment,
) -> CircuitSeq {

    let id = Permutation::id_perm(n);
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
            3
        } else if n == 5 || n == 6 {
            4
        } else if n == 4 {
            5
        } else {
            10
        };

        let sub_m = subcircuit.gates.len();
        let min = min(sub_m, max);
        
        let (canon_perm_blob, canon_shuf_blob) = if n == 7 && sub_m <= max {
            let table = format!("n{}m{}", n, min);
            let query = format!(
                "SELECT perm, shuf FROM {} WHERE circuit = ?1 LIMIT 1",
                table
            );

            let sql_t0 = Instant::now();
            let mut stmt = match conn.prepare(&query) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let rows = stmt.query([&subcircuit.repr_blob()]);
            SQL_TIME.fetch_add(sql_t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

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
            let t1 = Instant::now();
            let sub_perm = subcircuit.permutation(n);
            PERMUTATION_TIME.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);

            let t2 = Instant::now();
            let canon_perm = get_canonical(&sub_perm, bit_shuf);
            CANON_TIME.fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);

            (canon_perm.perm.repr_blob(), canon_perm.shuffle.repr_blob())
        };
        let prefix = canon_perm_blob.as_slice();
        for smaller_m in 1..=sub_m {
            let db_name = format!("n{}m{}", n, smaller_m);

            if (n == 7 && smaller_m == 4) || (n == 6 && smaller_m == 5) {
                let table = format!("n{}m{}", n, smaller_m);
                let query = format!(
                    "SELECT * FROM {} WHERE perm = ?1 ORDER BY RANDOM() LIMIT 1",
                    table
                );

                let sql_t0 = Instant::now();
                let mut stmt = match conn.prepare(&query) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let rows = stmt.query([&canon_perm_blob]);
                SQL_TIME.fetch_add(sql_t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

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
            } else {

                let db = match env.open_db(Some(&db_name)) {
                    Ok(db) => db,
                    Err(lmdb::Error::NotFound) => continue,
                    Err(e) => panic!("Failed to open LMDB database {}: {:?}", db_name, e),
                };
                let mut invert = false;
                let hit = {
                    let txn = env.begin_ro_txn().expect("txn");

                    let t0 = Instant::now();
                    
                    let mut res = random_perm_lmdb(&txn, db, prefix);
                    if res.is_none() {
                        let prefix_inv_blob = Permutation::from_blob(&prefix).invert().repr_blob();
                        invert = true;
                        res = random_perm_lmdb(&txn, db, &prefix_inv_blob);
                    }

                    SQL_TIME.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

                    res.map(|(_key, val_blob)| val_blob)
                };

                if let Some(val_blob) = hit {
                    let (repl_blob, repl_shuf): (Vec<u8>, Vec<u8>) =
                        bincode::deserialize(&val_blob).expect("Failed to deserialize");

                    let mut repl = CircuitSeq::from_blob(&repl_blob);

                    repl.rewire(&Permutation::from_blob(&repl_shuf), n);

                    if invert {
                        repl.gates.reverse();
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

pub fn expand_big(c: &CircuitSeq, trials: usize, num_wires: usize, conn: &mut Connection) -> CircuitSeq {
    let mut circuit = c.clone();
    let mut rng = rand::rng();

    for _i in 0..trials {
        // if i % 20 == 0 {
        //     println!("{} trials so far, {} more to go", i, trials - i);
        // }
        let mut subcircuit_gates = vec![];

        for set_size in (3..=16).rev() {
            let random_max_wires = rng.random_range(3..=7);
            let (gates, _) = find_convex_subcircuit(set_size, random_max_wires, num_wires, &circuit, &mut rng);
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
        let expected_slice: Vec<_> = subcircuit_gates.iter().map(|&i| circuit.gates[i]).collect();
        let actual_slice = &circuit.gates[start..=end];

        if actual_slice != &expected_slice[..] {
            break;
        }

        let used_wires = subcircuit.used_wires();
        subcircuit = CircuitSeq::rewire_subcircuit(&mut circuit, &mut subcircuit_gates, &used_wires);

        let num_wires = used_wires.len();
        let perms: Vec<Vec<usize>> = (0..num_wires).permutations(num_wires).collect();
        let bit_shuf = perms.into_iter().skip(1).collect::<Vec<_>>();

        let subcircuit_temp = expand(&subcircuit, 3, conn, &bit_shuf, num_wires);

        if subcircuit.permutation(num_wires) != subcircuit_temp.permutation(num_wires) {
            panic!("Compress changed something");
        }
        subcircuit = subcircuit_temp;

        subcircuit = CircuitSeq::unrewire_subcircuit(&subcircuit, &used_wires);

        circuit.gates.splice(start..end+1, subcircuit.gates);
        if c.permutation(num_wires).data != circuit.permutation(num_wires).data {
            panic!("splice changed something");
        }
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

pub fn outward_compress(g: &CircuitSeq, r: &CircuitSeq, trials: usize, conn: &mut Connection, bit_shuf: &Vec<Vec<usize>>, n: usize) -> CircuitSeq {
    let mut g = g.clone();
    for gate in r.gates.iter() {
        let wrapper = CircuitSeq { gates: vec![*gate] };
        g = compress(&wrapper.concat(&g).concat(&wrapper), trials, conn, bit_shuf, n);
    }
    g
}

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
}


#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

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
    // use std::fs;
    // use std::fs::File;
    // #[test]
    // fn test_compression_big_time() {
    //     let total_start = Instant::now();

    //     // // ---------- FIRST TEST ----------
    //     // let t1_start = Instant::now();
    //     // let n = 64;
    //     // let str1 = "circuitQQF_64.txt";
    //     // let data1 = fs::read_to_string(str1).expect("Failed to read circuitQQF_64.txt");
    //     // let mut stable_count = 0;
    //     // let mut conn = Connection::open("circuits.db").expect("Failed to open DB");
    //     // let mut acc = CircuitSeq::from_string(&data1);
    //     // while stable_count < 3 {
    //     //     let before = acc.gates.len();
    //     //     acc = compress_big(&acc, 1_000, n, &mut conn);
    //     //     let after = acc.gates.len();

    //     //     if after == before {
    //     //         stable_count += 1;
    //     //         println!("  Final compression stable {}/3 at {} gates", stable_count, after);
    //     //     } else {
    //     //         println!("  Final compression reduced: {} → {} gates", before, after);
    //     //         stable_count = 0;
    //     //     }
    //     // }
    //     // let t1_duration = t1_start.elapsed();
    //     // println!(" First compression finished in {:.2?}", t1_duration);

    //     // ---------- SECOND TEST ----------
    //     let t2_start = Instant::now();
    //     let str2 = "./old/circuitQQF_64.txt";
    //     let data2 = fs::read_to_string(str2).expect("Failed to read circuitF.txt");
    //     let mut stable_count = 0;
    //     let mut conn = Connection::open("circuits.db").expect("Failed to open DB");
    //     let mut acc = CircuitSeq::from_string(&data2);
    //     while stable_count < 3 {
    //         let before = acc.gates.len();
    //         acc = compress_big(&acc, 1_000, 64, &mut conn);
    //         let after = acc.gates.len();

    //         if after == before {
    //             stable_count += 1;
    //             println!("  Final compression stable {}/3 at {} gates", stable_count, after);
    //         } else {
    //             println!("  Final compression reduced: {} → {} gates", before, after);
    //             stable_count = 0;
    //         }
    //     }

    //     File::create("compressed.txt")
    //     .and_then(|mut f| f.write_all(acc.repr().as_bytes()))
    //     .expect("Failed to write butterfly_recent.txt");
    //     let t2_duration = t2_start.elapsed();
    //     println!(" Second compression finished in {:.2?}", t2_duration);

    //     // ---------- TOTAL ----------
    //     let total_duration = total_start.elapsed();
    //     println!(" Total test duration: {:.2?}", total_duration);
    // }
}