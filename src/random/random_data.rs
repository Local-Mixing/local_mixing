use crate::circuit::{self, Permutation, CircuitSeq};
use crate::rainbow::canonical::{self, Canonicalization, CandSet};
use rand::Rng;
use rusqlite::{Connection, Result};
use smallvec::SmallVec;
use itertools::Itertools;

pub fn random_circuit(base_gates: &Vec<[usize;3]>, m:usize) -> CircuitSeq {
    let mut rng = rand::rng();
    let mut circuit = Vec::with_capacity(m);
    for _ in 0..m {
        let rand = rng.random_range(0..base_gates.len());
        circuit.push(rand);
    }
    CircuitSeq{ gates: circuit, }
}

pub fn create_table(conn: &Connection, n: usize, m: usize) -> Result<()> {
    // Table name includes n and m
    let table_name = format!("n{}m{}", n, m);

    let sql = format!(
        "CREATE TABLE IF NOT EXISTS {} (
            circuit TEXT UNIQUE,
            perm TEXT NOT NULL,
            shuf TEXT NOT NULL
        )",
        table_name
    );

    conn.execute(&sql, [])?;
    Ok(())
}

pub fn insert_circuit(
    conn: &Connection,
    n: usize,
    m: usize,
    circuit: &CircuitSeq, 
    canon: &Canonicalization,
) -> Result<()> {
    let table_name = format!("n{}m{}", n, m);
    let key = circuit.repr();
    let perm = canon.perm.repr();
    let shuf = canon.shuffle.repr();
    let sql = format!("INSERT INTO {} (circuit, perm, shuf) VALUES (?1, ?2, ?3)", table_name);
    conn.execute(&sql, &[&key, &perm, &shuf])?;
    Ok(())
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
}

pub fn main_random(n: usize, m: usize, count: usize) {
    let conn = match Connection::open("circuits.db") {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to open DB: {}", e);
            return
        }
    };

    if let Err(e) = create_table(&conn, n, m) {
        eprintln!("Failed to create table: {}", e);
        return
    }

    let base_gates = circuit::base_gates(n);
    let perms: Vec<Vec<usize>> = (0..n).permutations(n).collect();
    let bit_shuf = perms.into_iter().skip(1).collect::<Vec<_>>();
    let mut inserted = 0;
    while inserted < count {
        let mut circuit = random_circuit(&base_gates, m);
        circuit.canonicalize(&base_gates);

        let perm = circuit
            .permutation(n, &base_gates)
            .canon_simple(&bit_shuf);

        if insert_circuit(&conn, n, m, &circuit, &perm).is_ok() {
            inserted += 1; // only increment if insert succeeds
        } else {
            println!("Failed to add number {}", inserted);
        }
        // otherwise, skip and try a new circuit
    }
}
