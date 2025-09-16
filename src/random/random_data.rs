use crate::circuit::{Permutation, CircuitSeq};
use crate::rainbow::canonical::{self, Canonicalization, CandSet};
use rusqlite::{Connection, Result};
use smallvec::SmallVec;
use itertools::Itertools;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::fs::OpenOptions;
use std::io::Write;

pub fn random_circuit(n: u8, m: usize) -> CircuitSeq {
    let mut circuit = Vec::with_capacity(m);

    for _ in 0..m {
        loop {
            // mask for used pins
            let mut set = [false; 16];
            for i in n..16 {
                set[i as usize] = true; // disable pins >= n
            }

            // pick 3 distinct pins
            let mut gate = [0u8; 3];
            for j in 0..3 {
                loop {
                    let v = fastrand::u8(..16);
                    if !set[v as usize] {
                        set[v as usize] = true;
                        gate[j] = v;
                        break;
                    }
                }
            }

            // check against last gate to avoid duplicates
            if circuit.last() == Some(&gate) {
                continue; 
            } else {
                circuit.push(gate);
                break;
            }
        }
    }

    CircuitSeq { gates: circuit }
}


pub fn create_table(conn: &Connection, table_name: &str) -> Result<()> {
    // Table name includes n and m
    let sql = format!(
        "CREATE TABLE IF NOT EXISTS {} (
            circuit BLOB UNIQUE,
            perm BLOB NOT NULL,
            shuf BLOB NOT NULL
        )",
        table_name
    );

    conn.execute(&sql, [])?;
    Ok(())
}

pub fn insert_circuit(
    conn: &mut Connection,
    circuit: &CircuitSeq, 
    canon: &Canonicalization,
    table_name: &str,
) -> Result<()> {
    let key = circuit.repr_blob();
    let perm = canon.perm.repr_blob();
    let shuf = canon.shuffle.repr_blob();
    let sql = format!("INSERT INTO {} (circuit, perm, shuf) VALUES (?1, ?2, ?3)", table_name);
    conn.execute(&sql, &[&key, &perm, &shuf])?;
    Ok(())
}

pub fn insert_circuits_batch(
    conn: &mut Connection,
    table_name: &str,
    circuits: &[(CircuitSeq, Canonicalization)],
) -> Result<usize> {
    // Start a single transaction for all inserts
    let tx = conn.transaction()?;

    // Prepare the SQL once
    let sql = format!(
        "INSERT OR IGNORE INTO {} (circuit, perm, shuf) VALUES (?1, ?2, ?3)",
        table_name
    );

    let mut inserted = 0;

    for (circuit, canon) in circuits {
        let key = circuit.repr_blob();
        let perm = canon.perm.repr_blob();
        let shuf = canon.shuffle.repr_blob();

        if tx.execute(&sql, &[&key, &perm, &shuf])? > 0 {
            inserted += 1;
        }
    }

    // Commit all inserts at once
    tx.commit()?;

    Ok(inserted)
}

impl Permutation {
    pub fn canon(&self, bit_shuf: &[Vec<usize>], retry: bool) -> Canonicalization {
        if bit_shuf.is_empty() {
            panic!("bit_shuf cannot be empty!");
        }

        // Try fast canonicalization
        let mut pm = self.fast();

        if pm.perm.data.is_empty() {
            if retry {
                // Fast canon failed, retry with a random shuffle
                let n = self.data.len();
                let r = Permutation::rand_perm(n);
                return self.bit_shuffle(&r.data).canon(bit_shuf, false);
            } else {
                // Retry not allowed, fall back to brute force
                pm = self.brute(bit_shuf);
            }
        }

        pm
    }

    pub fn canon_simple(&self, bit_shuf: &[Vec<usize>]) -> Canonicalization {
        self.canon(bit_shuf, false)
    }

    pub fn brute(&self, bit_shuf: &[Vec<usize>]) -> Canonicalization {
        if bit_shuf.is_empty() {
            panic!("bit_shuf cannot be empty!");
        }

        let n = self.data.len();
        let num_b = std::mem::size_of::<usize>() * 8 - (n - 1).leading_zeros() as usize;

        let mut min_perm: SmallVec<[usize; 64]> = SmallVec::from_slice(&self.data);
        let mut bits: SmallVec<[usize; 64]> = SmallVec::from_elem(0, n);
        let mut index_shuf: SmallVec<[usize; 64]> = SmallVec::from_elem(0, n);
        let mut perm_shuf: SmallVec<[usize; 64]> = SmallVec::from_elem(0, n);

        let mut best_shuffle = Permutation::id_perm(num_b);

        for r in bit_shuf.iter() {
            for (src, &dst) in r.iter().enumerate() {
                for (i, &val) in self.data.iter().enumerate() {
                    bits[i] |= ((val >> src) & 1) << dst;
                    index_shuf[i] |= ((i >> src) & 1) << dst;
                }
            }

            for (i, &val) in bits.iter().enumerate() {
                perm_shuf[index_shuf[i]] = val;
            }

            for weight in 0..=num_b / 2 {
                let mut done = false;
                for i in canonical::index_set(weight, num_b) {
                    if perm_shuf[i] == min_perm[i] {
                        continue;
                    }
                    if perm_shuf[i] < min_perm[i] {
                        min_perm.copy_from_slice(&perm_shuf);
                        best_shuffle.data.copy_from_slice(&r);
                    }
                    done = true;
                    break;
                }
                if done {
                    break;
                }
            }

            bits.fill(0);
            index_shuf.fill(0);
        }

        Canonicalization {
            perm: Permutation { data: min_perm.into_vec() },
            shuffle: best_shuffle,
        }
    }

    //Goal of fast canon is to produce small snippets of the best permutation (by lexi order) and determine which in canonical
    //If we can't decide between multiple, for now, we just ignore and will do brute force
    pub fn fast(&self) -> Canonicalization {
        let num_bits = self.bits();
        let mut candidates = CandSet::new(num_bits);
        let mut found_identity = false;

        // Scratch buffer to avoid cloning every iteration
        let mut scratch = CandSet::new(num_bits);

        // Pre-allocate viable_sets buffer to reuse
        let mut viable_sets: Vec<CandSet> = Vec::with_capacity(4);

        for weight in 0..=num_bits/2 {
            let index_words = canonical::index_set(weight, num_bits); // Vec<usize>

            'word_loop: for &w in &index_words {
                // Determine which preimages are possible
                let preimages = candidates.preimages(w);
                if preimages.is_empty() {
                    return Canonicalization {
                        perm: Permutation { data: Vec::new() },
                        shuffle: Permutation { data: Vec::new() },
                    };
                }

                viable_sets.clear();
                let mut best_score = -1;

                for &pre_idx in &preimages {
                    let mapped_value = self.data[pre_idx];

                    if !candidates.consistent(pre_idx, w) {
                        continue;
                    }

                    // Reset scratch from candidates and enforce mapping
                    scratch.copy_from(&candidates);
                    scratch.enforce(pre_idx, w);

                    // Minimum possible value with current scratch
                    let (score, mut reduced_set) = scratch.min_consistent(mapped_value);
                    if score < 0 {
                        continue;
                    }

                    reduced_set.intersect(&candidates);
                    if !reduced_set.consistent(pre_idx, w) {
                        continue;
                    }

                    // Track best score and viable sets
                    if best_score < 0 || score < best_score {
                        best_score = score;
                        viable_sets.clear();
                        // Move reduced_set into the vector (no clone)
                        viable_sets.push(reduced_set);
                        if w as isize == score {
                            found_identity = true;
                        }
                    } else if score == best_score {
                        if w as isize == score {
                            if found_identity {
                                viable_sets.push(reduced_set);
                            } else {
                                viable_sets.clear();
                                viable_sets.push(reduced_set);
                            }
                            found_identity = true;
                        } else if !found_identity {
                            viable_sets.push(reduced_set);
                        }
                    }
                }

                match viable_sets.len() {
                    0 => continue,
                    1 => candidates = viable_sets.pop().unwrap(),
                    _ => {
                        return Canonicalization {
                            perm: Permutation { data: Vec::new() },
                            shuffle: Permutation { data: Vec::new() },
                        }
                    }
                }

                if candidates.complete() {
                    break 'word_loop;
                }
            }

            if candidates.complete() {
                break;
            }
        }

        if candidates.unconstrained() {
            return Canonicalization {
                perm: self.clone(),
                shuffle: Permutation { data: Vec::new() },
            };
        }

        if !candidates.complete() {
            println!("Incomplete!");
            println!("{:?}", self);
            println!("{:?}", candidates);
            std::process::exit(1);
        }

        let final_shuffle = match candidates.output() {
            Some(v) => Permutation { data: v },
            None => {
                eprintln!("CandSet output returned None!");
                std::process::exit(1);
            }
        };

        Canonicalization {
            perm: self.bit_shuffle(&final_shuffle.data),
            shuffle: final_shuffle,
        }
    }

    pub fn from_string(s: &str) -> Self {
        let data = s
            .split(',')
            .map(|x| x.trim().parse::<usize>().expect("Invalid number in permutation"))
            .collect();

        Permutation { data }
    }
}

pub fn check_cycles(n: usize, m: usize) -> Result<()> {
    // Open the database
    let conn = Connection::open("circuits.db")?;
    let table_name = format!("n{}m{}", n, m);

    // Build the query string with the table name
    let query = format!("SELECT DISTINCT perm FROM {}", table_name);
    let mut stmt = conn.prepare(&query)?;

    // Query all distinct perms
    let perm_iter = stmt.query_map([], |row| {
        let perm_str: Vec<u8> = row.get(0)?; // now as String
        Ok(perm_str)
    })?;

    println!("Distinct permutations in {}:", table_name);

    for perm_str_result in perm_iter {
        let perm_str = perm_str_result?;

        // Convert the string into a Permutation
        let perm = Permutation::from_blob(&perm_str);
        let cycles = perm;

        println!("{:?}", cycles);
    }

    Ok(())
}

pub fn print_all(table_name: &str) -> Result<()> {
    let conn = Connection::open("circuits.db")?;

    let query = format!("SELECT circuit, perm, shuf FROM {}", table_name);
    let mut stmt = conn.prepare(&query)?;

    let rows = stmt.query_map([], |row| {
        let circuit_blob: Vec<u8> = row.get(0)?;
        let perm_blob: Vec<u8> = row.get(1)?;
        let shuf_blob: Vec<u8> = row.get(2)?;

        Ok((circuit_blob, perm_blob, shuf_blob))
    })?;

    for row in rows {
        let (circuit_blob, perm_blob, shuf_blob) = row?;

        let circuit = CircuitSeq::from_blob(&circuit_blob);
        let perm = Permutation::from_blob(&perm_blob);
        let shuf = Permutation::from_blob(&shuf_blob);

        println!("Circuit: {:?}", circuit.gates);
        println!("Perm:    {:?}", perm.data);
        println!("Shuf:    {:?}", shuf.data);
        println!();
    }

    Ok(())
}

pub fn count_distinct(n: usize, m: usize) -> Result<usize> {
    let conn = Connection::open("circuits.db")?;
    let table_name = format!("n{}m{}", n, m);
    
    let query = format!("SELECT COUNT(DISTINCT perm) FROM {}", table_name);
    let count: usize = conn.query_row(&query, [], |row| row.get(0))?;
    
    println!("Number of distinct permutations in {}: {}", table_name, count);
    Ok(count)
}

//TODO: benchmark to see which part is taking the most time and what exactly can be sped up
//Speed up SQL queries
//Should not see for a particular size query, the speed should not vary across multiple runs
pub fn main_random(n: usize, m: usize, count: usize, stop: bool) {
    let mut conn = Connection::open("circuits.db").expect("Failed to open DB");
    let table_name = format!("n{}m{}", n, m);
    create_table(&mut conn, &table_name).expect("Failed to create table");

    let perms: Vec<Vec<usize>> = (0..n).permutations(n).collect();
    let bit_shuf = perms.into_iter().skip(1).collect::<Vec<_>>();

    let mut inserted = 0;
    let mut total_attempts = 0;
    let mut recent = 0;

    let batch_size = 5_000;
    let mut batch: Vec<(CircuitSeq, Canonicalization)> = Vec::with_capacity(batch_size);

    // Atomic flag for Ctrl+C
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    }).expect("Error setting Ctrl-C handler");

    //TODO: test speed here
    while running.load(Ordering::SeqCst) && (!stop && inserted < count || stop) {
        let start = std::time::Instant::now(); // start timing this iteration
        total_attempts += 1;

        let mut circuit = random_circuit(n as u8, m);
        circuit.canonicalize();

        let perm = circuit.permutation(n).canon_simple(&bit_shuf);
        batch.push((circuit, perm));

        if batch.len() >= batch_size {
            //let start = std::time::Instant::now();
            let success_count =
                insert_circuits_batch(&mut conn, &table_name, &batch).unwrap_or(0);
            //let elapsed = start.elapsed();

            // Log timing to file
            // let mut file = OpenOptions::new()
            //     .create(true)
            //     .append(true)
            //     .open("time5000.txt")
            //     .expect("Failed to open time.txt");
            // writeln!(
            //     file,
            //     "Batch of {} attempted, {} inserted, time: {:?}",
            //     batch_size, success_count, elapsed
            // ).expect("Failed to write to time.txt");

            inserted += success_count;
            recent += success_count;
            batch.clear();

            // Early stop if >=99% of last batch failed
            if success_count * 100 <= batch_size {
                // writeln!(
                //     file,
                //     "Stopping early: only {}/{} inserts succeeded (~{:.2}% success)",
                //     success_count,
                //     batch_size,
                //     (success_count as f64 / batch_size as f64) * 100.0
                // ).expect("Failed to write to time.txt");

                println!(
                    "Stopping early: only {}/{} inserts succeeded (~{:.2}% success)",
                    success_count,
                    batch_size,
                    (success_count as f64 / batch_size as f64) * 100.0
                );
                break;
            }
        }

        if total_attempts % 50_000 == 0 {
            println!("Attempts: {}, inserted in last window: {}", total_attempts, recent);
            recent = 0;
        }

        // Stop for non-stop mode
        if !stop && inserted >= count {
            break;
        }

        let elapsed = start.elapsed();
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("while.txt")
            .expect("Failed to open while.txt");
        writeln!(file, "Iteration {} took {:?}", total_attempts, elapsed)
            .expect("Failed to write to while.txt");
    }

    // Insert remaining circuits before exiting
    if !batch.is_empty() {
        let success_count =
            insert_circuits_batch(&mut conn, &table_name, &batch).unwrap_or(0);
        inserted += success_count;
    }

    println!(
        "Finished: inserted {} circuits after {} attempts",
        inserted, total_attempts
    );
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_cycles_n3m3() -> Result<()> {
        let now = std::time::Instant::now();
        // Call check_cycles for n=3, m=3
        let _ = check_cycles(3, 3);
        print_all("n3m3")?;
        println!("Time: {:?}", now.elapsed());
        Ok(())
    }
}