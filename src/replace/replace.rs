use crate::circuit::circuit::{Permutation, CircuitSeq};
use crate::random::random_data;
use crate::random::random_data::random_circuit;
use itertools::Itertools;
use rusqlite::{Connection, OptionalExtension};
use rand::prelude::IndexedRandom;
use std::cmp::{max, min};
use rand::Rng;

// Returns a nontrivial identity circuit built from two "friend" circuits
pub fn random_canonical_id(
    conn: &Connection,
    table: &str,
) -> Result<CircuitSeq, Box<dyn std::error::Error>> {

    // Pick a random perm that has more than one friend
    let perm: Option<String> = conn.query_row(
        &format!(
            "SELECT perm FROM {} GROUP BY perm HAVING COUNT(*) > 1 ORDER BY RANDOM() LIMIT 1",
            table
        ),
        [],
        |row| row.get(0),
    ).optional()?; // None if no row found

    let perm = match perm {
        Some(p) => p,
        None => return Err(format!("No permutation with more than one friend found in table {}", table).into()),
    };

    // Get all circuit blobs with that perm
    let mut stmt = conn.prepare(&format!("SELECT blob FROM {} WHERE perm = ?", table))?;
    let blobs: Vec<Vec<u8>> = stmt.query_map([&perm], |row| row.get(0))?
        .map(|r| r.unwrap())
        .collect();

    if blobs.len() < 2 {
        return Err(format!("Expected at least 2 circuits for perm {}, got {}", perm, blobs.len()).into());
    }

    // Pick two distinct blobs randomly
    let mut rng = rand::rng();
    let (a_blob, b_blob) = loop {
        let a = blobs.choose(&mut rng).unwrap();
        let b = blobs.choose(&mut rng).unwrap();
        if a.as_ptr() != b.as_ptr() { // ensure distinct
            break (a, b);
        }
    };

    // Deserialize blobs into Circuit structs
    let mut ca = CircuitSeq::from_blob(a_blob);
    let mut cb = CircuitSeq::from_blob(b_blob);

    // Reverse cb gates and append to ca
    cb.gates.reverse();       // Reverse the order of gates in cb
    ca.gates.extend(cb.gates); // Append reversed cb gates to ca

    // Return the nontrivial identity circuit
    Ok(ca)
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

pub fn compress(c: &CircuitSeq, trials: usize, conn: &Connection) -> CircuitSeq {
    let mut compressed = c.clone();

    // Remove adjacent identity gates in-place
    // Update write only if we are writing, otherwise, move read along on its own
    {
        let mut write = 0;
        for read in 0..compressed.gates.len() {
            if write == 0 || compressed.gates[read] != compressed.gates[write - 1] {
                compressed.gates[write] = compressed.gates[read];
                write += 1;
            }
        }
        compressed.gates.truncate(write);
    }

    // Find a random subcircuit to (attempt to) replace some number of times
    // Can forcibly choose where to do compression as well
    let n = c.num_wires();
    let perms: Vec<Vec<usize>> = (0..n).permutations(n).collect();
    let bit_shuf = perms.into_iter().skip(1).collect::<Vec<_>>();

    for _ in 0..trials {
        let (subcircuit, start, end) = random_subcircuit(&compressed);
        let sub_perm = subcircuit.permutation(n);
        let canon_perm = sub_perm.canon_simple(&bit_shuf);
        let perm_blob = canon_perm.perm.repr_blob();

        let sub_m = subcircuit.gates.len();

        // Try all smaller m tables for this n
        for smaller_m in 1..sub_m {
            let table = format!("n{}m{}", n, smaller_m);
            let query = format!("SELECT blob FROM {} WHERE perm = ?1", table);

            let mut stmt = match conn.prepare(&query) {
                Ok(s) => s,
                Err(_) => continue, // skip if table doesn't exist
            };

            let mut rows = stmt.query([&perm_blob]).expect("Query failed");

            if let Some(row) = rows.next().unwrap() {
                let blob: Vec<u8> = row.get(0).expect("Failed to get blob");
                let mut repl = CircuitSeq::from_blob(&blob);

                if repl.gates.len() < subcircuit.gates.len() {
                    // adjust rewiring
                    let rc = repl.permutation(n).canon_simple(&bit_shuf);
                    if !rc.shuffle.data.is_empty() {
                        repl.rewire(&rc.shuffle);
                    }
                    repl.rewire(&canon_perm.shuffle.invert());

                    if repl.permutation(n) != sub_perm {
                        panic!("Replacement permutation mismatch!");
                    }

                    // Do the replacement
                    compressed.gates.splice(start..end, repl.gates);
                    break; // stop once a replacement is applied
                }
            }
        }
    }

    let mut i= 0;
    while i < compressed.gates.len().saturating_sub(1) {
        if compressed.gates[i] == compressed.gates[i + 1] {
            // remove elements at i and i+1
            compressed.gates.drain(i..=i + 1);

            // step back up to 2 indices, but not below 0
            i = (i - 2).max(0);
        } else {
            i += 1;
        }
    }

    compressed
}

// inflate. u8 is the size of the random circuit used to obfuscate over 2 
pub fn obfuscate(c: &CircuitSeq) -> (CircuitSeq, usize) {
    // Start with an empty circuit, preserving wire count
    let mut obfuscated = CircuitSeq {
        gates: Vec::new(),
    };

    let num_wires = c.num_wires();
    // The identity permutation on all wires
    let identity_perm = Permutation::id_perm(1 << num_wires);

    // Size of our random circuit (so id is 8). This can be randomized later
    let m = 4;

    for gate in &c.gates {
        // Generate a random identity circuit
        let (mut ran, mut rand_rev) = random_id(num_wires as u8, m);
        let mut id = CircuitSeq { gates: Vec::new() };
        id.gates.extend(&ran.gates);
        id.gates.extend(&rand_rev.gates);
        // Sanity check: its permutation should equal the identity
        if id.permutation(num_wires) != identity_perm {
            panic!(
                "Random identity circuit has wrong permutation: {:?}",
                id.permutation(num_wires)
            );
        }

        // Rewire the first gate of the identity circuit to match this gate
        id.rewire_first_gate(*gate);

        // Append everything *after* that first gate into the obfuscated circuit
        obfuscated
            .gates
            .extend_from_slice(&id.gates[1..]);
    }

    (obfuscated, m)
}

