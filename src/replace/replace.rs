use crate::{
    circuit::circuit::{CircuitSeq, Permutation},
    random::random_data::{create_table, random_circuit, seeded_random_circuit},
};

use rand::{prelude::IndexedRandom, Rng};
use rusqlite::{params, Connection, OptionalExtension};

use std::{
    cmp::{max, min},
    // used for testing
    fs::OpenOptions,
    io::Write,
};

// Returns a nontrivial identity circuit built from two "friend" circuits
pub fn random_canonical_id(
    conn: &Connection,
    wires: usize,
) -> Result<CircuitSeq, Box<dyn std::error::Error>> {
    let pattern = format!("n{}%m%", wires);

    // Get all tables matching pattern (all m values)
    let tables: Vec<String> = conn
        .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?1")?
        .query_map([&pattern], |row| row.get(0))?
        .map(|r| r.unwrap())
        .collect();

    if tables.len() < 2 {
        return Err(format!("Need at least 2 tables matching {}", pattern).into());
    }

    let mut rng = rand::rng();

    loop {
        // Pick two random tables (can be different m values)
        let table_a = tables[rng.random_range(1..tables.len())].clone();
        let table_b = tables[rng.random_range(1..tables.len())].clone();

        // Find a permutation that exists in both tables
        let perm_opt: Option<Vec<u8>> = conn.query_row(
            &format!(
                "SELECT a.perm
                 FROM {} a
                 JOIN {} b ON a.perm = b.perm
                 ORDER BY RANDOM() LIMIT 1",
                table_a, table_b
            ),
            [],
            |row| row.get(0),
        ).optional()?;

        let perm = match perm_opt {
            Some(p) => p,
            None => continue, // Retry with new tables
        };

        // Get circuits from table_a
        let blobs_a: Vec<[Vec<u8>; 2]> = conn
            .prepare(&format!("SELECT circuit, shuf FROM {} WHERE perm = ?", table_a))?
            .query_map([&perm], |row| Ok([row.get(0)?, row.get(1)?]))?
            .map(|r| r.unwrap())
            .collect();

        // Get circuits from table_b
        let blobs_b: Vec<[Vec<u8>; 2]> = conn
            .prepare(&format!("SELECT circuit, shuf FROM {} WHERE perm = ?", table_b))?
            .query_map([&perm], |row| Ok([row.get(0)?, row.get(1)?]))?
            .map(|r| r.unwrap())
            .collect();

        if blobs_a.is_empty() || blobs_b.is_empty() {
            continue;
        }

        // Pick one circuit from each table
        let a_blob = blobs_a.choose(&mut rng).unwrap();
        let b_blob = blobs_b.choose(&mut rng).unwrap();

        // Deserialize circuits
        let mut ca = CircuitSeq::from_blob(&a_blob[0]);
        let mut cb = CircuitSeq::from_blob(&b_blob[0]);

        //println!("{}\n{}", ca.repr(), cb.repr());

        // Rewire cb to align with ca
        cb.rewire(&Permutation::from_blob(&b_blob[1]), wires);
        cb.rewire(&Permutation::from_blob(&a_blob[1]).invert(), wires);

        // Reverse cb and append to ca
        cb.gates.reverse();
        ca.gates.extend(cb.gates);

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
}

pub fn compress(c: &CircuitSeq, trials: usize, conn: &mut Connection, bit_shuf: &Vec<Vec<usize>>, n: usize) -> CircuitSeq {
    let mut compressed = c.clone();
    if c.gates.len() == 0 {
        return CircuitSeq{ gates: Vec::new() } 
    }
    // Open the file in append mode
    // let mut file = OpenOptions::new()
    //     .create(true)
    //     .append(true)
    //     .open("test.txt")
    //     .expect("Cannot open test.txt");
    //writeln!(file, "Permutation is initially: \n{:?}", compressed.permutation(n).data).unwrap();
    // let mut i = 0;
    // while i < compressed.gates.len().saturating_sub(1) {
    //     if compressed.gates[i] == compressed.gates[i + 1] {
    //         // remove elements at i and i+1
    //         compressed.gates.drain(i..=i + 1);

    //         // step back up to 2 indices, but not below 0
    //         i = i.saturating_sub(2);
    //     } else {
    //         i += 1;
    //     }
    // }
    //writeln!(file, "Permutation after remove identities 1 is: \n{:?}", compressed.permutation(n).data).unwrap();

    // Find a random subcircuit to (attempt to) replace some number of times
    // Can forcibly choose where to do compression as well

    for _ in 0..trials {
        let (subcircuit, start, end) = random_subcircuit(&compressed);
        let sub_perm = subcircuit.permutation(n);
        let canon_perm = sub_perm.canon_simple(&bit_shuf);
        let perm_blob = canon_perm.perm.repr_blob();
        let shuf_blob = canon_perm.shuffle.repr_blob();

        let sub_m = subcircuit.gates.len();
        let mut replaced = false;

        // Try all smaller m tables for this n
        for smaller_m in 1..sub_m {
            let table = format!("n{}m{}", n, smaller_m);
            let query = format!("SELECT blob FROM {} WHERE perm = ?1 ORDER BY RANDOM() LIMIT 1", table);
            {
                // Limit stmt's lifetime to this block
                let mut stmt = match conn.prepare(&query) {
                    Ok(s) => s,
                    Err(_) => continue,
                };

                let mut rows = stmt.query([&perm_blob]).expect("Query failed");

                if let Some(row) = rows.next().unwrap() {
                    let blob: Vec<u8> = row.get(0).expect("Failed to get blob");
                    let mut repl = CircuitSeq::from_blob(&blob);

                    if repl.gates.len() < subcircuit.gates.len() {
                        // adjust rewiring
                        let rc = repl.permutation(n).canon_simple(&bit_shuf);
                        if !rc.shuffle.data.is_empty() {
                            repl.rewire(&rc.shuffle, n);
                        }
                        repl.rewire(&canon_perm.shuffle.invert(), n);

                        if repl.permutation(n) != sub_perm {
                            panic!("Replacement permutation mismatch!");
                        }

                        // Do the replacement
                        compressed.gates.splice(start..end, repl.gates);
                        replaced = true;
                        break; // stop once a replacement is applied
                    }
                }
            }
                
        }
        // If not replaced, insert the subcircuit into its own table immediately
        if !replaced {
            let table = format!("n{}m{}", n, sub_m);
            let sub_blob = subcircuit.repr_blob();

            create_table(conn, &table).expect("Failed to create table");

            conn.execute(
                &format!("INSERT INTO {} (circuit, perm, shuf) VALUES (?1, ?2)", table),
                params![sub_blob, perm_blob, shuf_blob],
            )
            .expect("Insert failed");
        }
    }
    //writeln!(file, "Permutation after replacement is: \n{:?}", compressed.permutation(n).data).unwrap();

    // let mut i = 0;
    // while i < compressed.gates.len().saturating_sub(1) {
    //     if compressed.gates[i] == compressed.gates[i + 1] {
    //         // remove elements at i and i+1
    //         compressed.gates.drain(i..=i + 1);

    //         // step back up to 2 indices, but not below 0
    //         i = i.saturating_sub(2);
    //     } else {
    //         i += 1;
    //     }
    // }
    //writeln!(file, "Permutation after remove identities 2 is: \n{:?}", compressed.permutation(n).data).unwrap();

    compressed
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