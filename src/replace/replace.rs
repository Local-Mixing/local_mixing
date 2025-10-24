use crate::{
    circuit::circuit::{CircuitSeq, Permutation},
    random::random_data::{create_table, random_circuit},
};

use crate::random::random_data::insert_circuit;
use crate::random::random_data::find_convex_subcircuit;
use crate::random::random_data::contiguous_convex;
use rand::{prelude::IndexedRandom, Rng};
use rusqlite::{params, Connection, OptionalExtension};
use itertools::Itertools;
use std::{
    cmp::{max, min},
    // used for testing
    fs::OpenOptions,
    io::Write,
    time::{Instant, Duration},
};
use crate::random::random_data::get_canonical;
use crate::random::random_data::is_convex;
use dashmap::DashMap;
use std::sync::Arc;

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

//TODO: look into if this is the best way to do things
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

pub fn compress(
    c: &CircuitSeq,
    trials: usize,
    conn: &mut Connection,
    bit_shuf: &Vec<Vec<usize>>,
    n: usize,
) -> CircuitSeq {
    // fn pct(part: Duration, total: Duration) -> f64 {
    //     if total.as_secs_f64() == 0.0 {
    //         0.0
    //     } else {
    //         100.0 * part.as_secs_f64() / total.as_secs_f64()
    //     }
    // }

    // let full_start = Instant::now();

    let id = Permutation::id_perm(n);
    if c.permutation(n) == id {
        return CircuitSeq { gates: Vec::new() };
    }

    let mut compressed = c.clone();
    if compressed.gates.is_empty() {
        return CircuitSeq { gates: Vec::new() };
    }

    // // Initial cleanup
    // let cleanup_start = Instant::now();
    let mut i = 0;
    while i < compressed.gates.len().saturating_sub(1) {
        if compressed.gates[i] == compressed.gates[i + 1] {
            compressed.gates.drain(i..=i + 1);
            i = i.saturating_sub(2);
        } else {
            i += 1;
        }
    }
    // let cleanup_time = cleanup_start.elapsed();

    if compressed.gates.is_empty() {
        return CircuitSeq { gates: Vec::new() };
    }

    // // Timing buckets
    // let mut total_random_sub = Duration::ZERO;
    // let mut total_canonicalize = Duration::ZERO;
    // let mut total_canon_simple = Duration::ZERO;
    // let mut total_lookup = Duration::ZERO;
    // let mut total_from_blob = Duration::ZERO;
    // let mut total_rewire = Duration::ZERO;
    // let mut total_splice = Duration::ZERO;
    // let mut total_no_replacement = Duration::ZERO;

    // Main loop
    for _ in 0..trials {
        // let random_start = Instant::now();
        let (mut subcircuit, start, end) = random_subcircuit(&compressed);
        // total_random_sub += random_start.elapsed();

        // canonicalize
        // let canon_start = Instant::now();
        subcircuit.canonicalize();
        // total_canonicalize += canon_start.elapsed();

        // canon_simple
        // let canon_simple_start = Instant::now();
        let sub_perm = subcircuit.permutation(n);
        let canon_perm = get_canonical(&sub_perm, &bit_shuf);
        // total_canon_simple += canon_simple_start.elapsed();

        let perm_blob = canon_perm.perm.repr_blob();
        let sub_m = subcircuit.gates.len();
        let mut found_replacement = false;
        // let no_replacement_start = Instant::now();

        for smaller_m in 1..=sub_m {
            let table = format!("n{}m{}", n, smaller_m);
            let query = format!(
                "SELECT circuit FROM {} WHERE perm = ?1 ORDER BY RANDOM() LIMIT 1",
                table
            );

            // let lookup_start = Instant::now();
            let mut stmt = match conn.prepare(&query) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let rows = stmt.query([&perm_blob]);
            // total_lookup += lookup_start.elapsed();

            if let Ok(mut r) = rows {
                if let Some(row) = r.next().unwrap() {
                    // let from_blob_start = Instant::now();
                    let blob: Vec<u8> = row.get(0).expect("Failed to get blob");
                    let mut repl = CircuitSeq::from_blob(&blob);
                    // total_from_blob += from_blob_start.elapsed();

                    if repl.gates.len() <= subcircuit.gates.len() {
                        let repl_perm = repl.permutation(n);
                        // let canon_start = Instant::now();
                        let rc = get_canonical(&repl_perm, &bit_shuf);
                        // total_canonicalize += canon_start.elapsed();

                        // let rewire_start = Instant::now();
                        if !rc.shuffle.data.is_empty() {
                            repl.rewire(&rc.shuffle, n);
                        }
                        repl.rewire(&canon_perm.shuffle.invert(), n);
                        // total_rewire += rewire_start.elapsed();

                        if repl.permutation(n) != sub_perm {
                            panic!("Replacement permutation mismatch!");
                        }

                        // let splice_start = Instant::now();
                        compressed.gates.splice(start..end, repl.gates);
                        // total_splice += splice_start.elapsed();

                        found_replacement = true;
                        break;
                    }
                }
            }
        }

        // if !found_replacement {
        //     total_no_replacement += no_replacement_start.elapsed();
        // }
    }

    // // Final cleanup
    // let final_cleanup_start = Instant::now();
    let mut i = 0;
    while i < compressed.gates.len().saturating_sub(1) {
        if compressed.gates[i] == compressed.gates[i + 1] {
            compressed.gates.drain(i..=i + 1);
            i = i.saturating_sub(2);
        } else {
            i += 1;
        }
    }
    // let final_cleanup_time = final_cleanup_start.elapsed();

    // // Summary printout
    // let full_time = full_start.elapsed();
    // let total_tracked = total_random_sub
    //     + total_canonicalize
    //     + total_canon_simple
    //     + total_lookup
    //     + total_from_blob
    //     + total_rewire
    //     + total_splice
    //     + total_no_replacement
    //     + cleanup_time
    //     + final_cleanup_time;

    // let missing = if full_time > total_tracked {
    //     full_time - total_tracked
    // } else {
    //     total_tracked - full_time
    // };

    // println!("\nCompress Timing Summary:");
    // println!("Total runtime: {:?} (100%)", full_time);
    // println!(
    //     "  Random subcircuit: {:?} ({:.2}%)",
    //     total_random_sub,
    //     pct(total_random_sub, full_time)
    // );
    // println!(
    //     "  Canonicalize: {:?} ({:.2}%)",
    //     total_canonicalize,
    //     pct(total_canonicalize, full_time)
    // );
    // println!(
    //     "  Canon simple: {:?} ({:.2}%)",
    //     total_canon_simple,
    //     pct(total_canon_simple, full_time)
    // );
    // println!(
    //     "  DB lookup: {:?} ({:.2}%)",
    //     total_lookup,
    //     pct(total_lookup, full_time)
    // );
    // println!(
    //     "  Deserialization (from_blob): {:?} ({:.2}%)",
    //     total_from_blob,
    //     pct(total_from_blob, full_time)
    // );
    // println!(
    //     "  Rewire: {:?} ({:.2}%)",
    //     total_rewire,
    //     pct(total_rewire, full_time)
    // );
    // println!(
    //     "  Splice: {:?} ({:.2}%)",
    //     total_splice,
    //     pct(total_splice, full_time)
    // );
    // println!(
    //     "  No replacement loops: {:?} ({:.2}%)",
    //     total_no_replacement,
    //     pct(total_no_replacement, full_time)
    // );
    // println!(
    //     "  Initial cleanup: {:?} ({:.2}%)",
    //     cleanup_time,
    //     pct(cleanup_time, full_time)
    // );
    // println!(
    //     "  Final cleanup: {:?} ({:.2}%)",
    //     final_cleanup_time,
    //     pct(final_cleanup_time, full_time)
    // );
    // println!(
    //     "  Unaccounted remainder: {:?} ({:.2}%)",
    //     missing,
    //     pct(missing, full_time)
    // );

    compressed
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
    while changed {
        changed = false;

        let len = compressed.gates.len();
        // loop over all subcircuit start and end indices
        'outer: for start in 0..len {
            for end in (start + 3)..=len { // skip lengths 1 and 2
                let mut subcircuit = CircuitSeq {
                    gates: compressed.gates[start..end].to_vec(),
                };

                subcircuit.canonicalize();
                let sub_perm = subcircuit.permutation(n);
                let canon_perm = get_canonical(&sub_perm, bit_shuf);
                let perm_blob = canon_perm.perm.repr_blob();
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
                    let rows = stmt.query([&perm_blob]);

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

                                compressed.gates.splice(start..end, repl.gates);
                                changed = true;
                                // restart from beginning
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }
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

    compressed
}

pub fn compress_big(c: &CircuitSeq, trials: usize, num_wires: usize, conn: &mut Connection) -> CircuitSeq {
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

        // println!("Sorting...");

        subcircuit_gates.sort();
        //let t0 = Instant::now();
        let (start, end) = contiguous_convex(&mut circuit, &mut subcircuit_gates, num_wires).unwrap();
        // let t_convex = t0.elapsed();
        // println!("contiguous_convex: {:?}", t_convex);
        let mut subcircuit = CircuitSeq { gates };
        // println!("Checking: {:?} \n Start {}, End {}", subcircuit_gates, start, end);
        let expected_slice: Vec<_> = subcircuit_gates.iter().map(|&i| circuit.gates[i]).collect();
        // assert_eq!(
        //     &circuit.gates[start..=end],
        //     &expected_slice[..],
        //     "contiguous_convex returned a range that does not match the subcircuit gates\n {:?} \n {} \n {}\n Circuit: {:?}", subcircuit_gates, start, end, c.gates
        // );

        let actual_slice = &circuit.gates[start..=end];

        if actual_slice != &expected_slice[..]
        {
        //     panic!(
        //         "contiguous_convex verification failed!
        // --------------------------------
        // Convex before: {}
        // Start: {start}, End: {end}
        // Subcircuit gate indices: {:?}

        // Expected slice ({} gates): {:?}
        // Actual slice ({} gates): {:?}

        // Circuit length: {}
        // Circuit gates: {:?}
        // --------------------------------
        // Convex after recheck: {}
        // Re-run range changed: {:?}
        // ",
        //         is_convex(16, &circuit, &subcircuit_gates),
        //         subcircuit_gates,
        //         expected_slice.len(),
        //         expected_slice,
        //         actual_slice.len(),
        //         actual_slice,
        //         circuit.gates.len(),
        //         circuit.gates,
        //         is_convex(16, &circuit, &subcircuit_gates),
        //         contiguous_convex(&mut circuit.clone(), &mut subcircuit_gates.clone()),
        //     );
            break;
        }

        //let t1 = Instant::now();
        let used_wires = subcircuit.used_wires();
        subcircuit = CircuitSeq::rewire_subcircuit(&mut circuit, &mut subcircuit_gates, &used_wires);
        // let t_rewire = t1.elapsed();
        // println!("rewire_subcircuit: {:?}", t_rewire);

        let num_wires = used_wires.len();
        let perms: Vec<Vec<usize>> = (0..num_wires).permutations(num_wires).collect();
        let bit_shuf = perms.into_iter().skip(1).collect::<Vec<_>>();

        // compress logs everything inside compress now
        //let t2 = Instant::now();
        let subcircuit_temp = if subcircuit.gates.len() <= 100 {
            compress_exhaust(&subcircuit, conn, &bit_shuf, num_wires)
        } else {
            compress(&subcircuit, 25_000, conn, &bit_shuf, num_wires)
        };
        if subcircuit.permutation(num_wires) != subcircuit_temp.permutation(num_wires) {
            panic!("Compress changed something");
        }
        subcircuit = subcircuit_temp;
        // let t_compress = t2.elapsed();
        // println!("compress(): {:?}", t_compress);

        //let t3 = Instant::now();
        subcircuit = CircuitSeq::unrewire_subcircuit(&subcircuit, &used_wires);
        // let t_unrewire = t3.elapsed();
        // println!("unrewire_subcircuit: {:?}", t_unrewire);

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


// inflate. u8 is the size of the random circuit used to obfuscate over 2 
// This tries to mix and hide original gate
// pub fn obfuscate(c: &CircuitSeq) -> (CircuitSeq, Vec<usize>) {
//     // Start with an empty circuit, preserving wire count
//     let mut obfuscated = CircuitSeq {
//         gates: Vec::new(),
//     };

//     // We want the ability to choose where to compress, so return beginning of each half id
//     let mut inverse_starts = Vec::new(); 

//     let num_wires = c.num_wires();
//     // The identity permutation on all wires
//     let identity_perm = Permutation::id_perm(1 << num_wires);

//     // Size of our random circuit. This can be randomized more later
//     let mut rng = rand::rng();
//     let m = rng.random_range(3..=5);

//     for gate in &c.gates {
//         // Generate a random identity circuit
//         let (ran, rand_rev) = random_id(num_wires as u8, m);
//         let mut id = CircuitSeq { gates: Vec::new() };
//         id.gates.extend(&ran.gates);

//         // record where the inverse part (rand_rev) begins
//         inverse_starts.push(obfuscated.gates.len() + id.gates.len());

//         id.gates.extend(&rand_rev.gates);

//         // Sanity check: its permutation should equal the identity
//         if id.permutation(num_wires) != identity_perm {
//             panic!(
//                 "Random identity circuit has wrong permutation: {:?}",
//                 id.permutation(num_wires)
//             );
//         }

//         // Rewire the first gate of the identity circuit to match this gate
//         id.rewire_first_gate(*gate);

//         // Append everything *after* that first gate into the obfuscated circuit
//         obfuscated
//             .gates
//             .extend_from_slice(&id.gates[1..]);
//     }

//     let (r0, r0_inv) = random_id(num_wires as u8, rng.random_range(3..=5));

//     obfuscated.gates.extend(&r0.gates);
//     inverse_starts.push(obfuscated.gates.len());
//     obfuscated.gates.extend(&r0_inv.gates);

//     (obfuscated, inverse_starts)
// }

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

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    #[test]
    fn random_circuit_exists_in_db() {
        // Open the SQLite DB
        let mut conn = Connection::open("circuits.db").expect("Failed to open DB");

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
}