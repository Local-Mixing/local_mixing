use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io::Write,
};

use itertools::Itertools;
use rusqlite::Connection;

use crate::{
    circuit::circuit::CircuitSeq,
    replace::{
        mixing::{
            abutterfly_big,
            butterfly,
            butterfly_big,
            interleave_sequential_big,
            obfuscate_and_target_compress,
            replace_and_compress_big,
            replace_and_compress_big_distance,
        },
        transpositions::{insert_wire_shuffles, insert_wire_shuffles_x},
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
pub fn open_all_dbs(env: &lmdb::Environment) -> HashMap<String, lmdb::Database> {
    let mut dbs = HashMap::new();
    let db_names = [
        "n3m1","n3m2","n3m3","n3m4","n3m5","n3m6","n3m7","n3m8","n3m9","n3m10",
        "n4m1","n4m2","n4m3","n4m4","n4m5","n4m6",
        "n5m1","n5m2","n5m3","n5m4","n5m5",
        "n6m1","n6m2","n6m3","n6m4","n6m5",
        "n7m1","n7m2","n7m3","n7m4",
        "perm_tables_n3","perm_tables_n4","perm_tables_n5","perm_tables_n6","perm_tables_n7",
        "n4m1perms","n4m2perms","n4m3perms","n4m4perms","n4m5perms","n4m6perms",
        "n5m1perms","n5m2perms","n5m3perms","n5m4perms","n5m5perms",
        "n6m1perms","n6m2perms","n6m3perms","n6m4perms",
        "n7m1perms","n7m2perms","n7m3perms",
        "ids_n5g0", "ids_n5g1", "ids_n5g2", "ids_n5g3", "ids_n5g4", "ids_n5g5", "ids_n5g6", "ids_n5g7", "ids_n5g8", "ids_n5g9", 
        "ids_n5g10", "ids_n5g11", "ids_n5g12", "ids_n5g13", "ids_n5g14", "ids_n5g15", "ids_n5g16", "ids_n5g17", "ids_n5g18", "ids_n5g19", 
        "ids_n5g20", "ids_n5g21", "ids_n5g22", "ids_n5g23", "ids_n5g24", "ids_n5g25", "ids_n5g26", "ids_n5g27", "ids_n5g28", "ids_n5g29", 
        "ids_n5g30", "ids_n5g31", "ids_n5g32", "ids_n5g33",
        "ids_n6g0", "ids_n6g1", "ids_n6g2", "ids_n6g3", "ids_n6g4", "ids_n6g5", "ids_n6g6", "ids_n6g7", "ids_n6g8", "ids_n6g9", 
        "ids_n6g10", "ids_n6g11", "ids_n6g12", "ids_n6g13", "ids_n6g14", "ids_n6g15", "ids_n6g16", "ids_n6g17", "ids_n6g18", "ids_n6g19", 
        "ids_n6g20", "ids_n6g21", "ids_n6g22", "ids_n6g23", "ids_n6g24", "ids_n6g25", "ids_n6g26", "ids_n6g27", "ids_n6g28", "ids_n6g29", 
        "ids_n6g30", "ids_n6g31", "ids_n6g32", "ids_n6g33",
        "ids_n7g0", "ids_n7g1", "ids_n7g2", "ids_n7g3", "ids_n7g4", "ids_n7g5", "ids_n7g6", "ids_n7g7", "ids_n7g8", "ids_n7g9", 
        "ids_n7g10", "ids_n7g11", "ids_n7g12", "ids_n7g13", "ids_n7g14", "ids_n7g15", "ids_n7g16", "ids_n7g17", "ids_n7g18", "ids_n7g19", 
        "ids_n7g20", "ids_n7g21", "ids_n7g22", "ids_n7g23", "ids_n7g24", "ids_n7g25", "ids_n7g26", "ids_n7g27", "ids_n7g28", "ids_n7g29", 
        "ids_n7g30", "ids_n7g31", "ids_n7g32", "ids_n7g33",
        "ids_n16g0", "ids_n16g1", "ids_n16g2", "ids_n16g3", "ids_n16g4", "ids_n16g5", "ids_n16g6", "ids_n16g7", "ids_n16g8", "ids_n16g9", 
        "ids_n16g10", "ids_n16g11", "ids_n16g12", "ids_n16g13", "ids_n16g14", "ids_n16g15", "ids_n16g16", "ids_n16g17", "ids_n16g18", "ids_n16g19", 
        "ids_n16g20", "ids_n16g21", "ids_n16g22", "ids_n16g23", "ids_n16g24", "ids_n16g25", "ids_n16g26", "ids_n16g27", "ids_n16g28", "ids_n16g29", 
        "ids_n16g30", "ids_n16g31", "ids_n16g32", "ids_n16g33",
        "ids_n128g0single",  "ids_n128g1single",  "ids_n128g2single",  "ids_n128g3single",
        "ids_n128g4single",  "ids_n128g5single",  "ids_n128g6single",  "ids_n128g7single",
        "ids_n128g8single",  "ids_n128g9single",  "ids_n128g10single", "ids_n128g11single",
        "ids_n128g12single", "ids_n128g13single", "ids_n128g14single", "ids_n128g15single",
        "ids_n128g16single", "ids_n128g17single", "ids_n128g18single", "ids_n128g19single",
        "ids_n128g20single", "ids_n128g21single", "ids_n128g22single", "ids_n128g23single",
        "ids_n128g24single", "ids_n128g25single", "ids_n128g26single", "ids_n128g27single",
        "ids_n128g28single", "ids_n128g29single", "ids_n128g30single", "ids_n128g31single",
        "ids_n128g32single", "ids_n128g33single",
        "ids_n128g0tower",  "ids_n128g1tower",  "ids_n128g2tower",  "ids_n128g3tower",
        "ids_n128g4tower",  "ids_n128g5tower",  "ids_n128g6tower",  "ids_n128g7tower",
        "ids_n128g8tower",  "ids_n128g9tower",  "ids_n128g10tower", "ids_n128g11tower",
        "ids_n128g12tower", "ids_n128g13tower", "ids_n128g14tower", "ids_n128g15tower",
        "ids_n128g16tower", "ids_n128g17tower", "ids_n128g18tower", "ids_n128g19tower",
        "ids_n128g20tower", "ids_n128g21tower", "ids_n128g22tower", "ids_n128g23tower",
        "ids_n128g24tower", "ids_n128g25tower", "ids_n128g26tower", "ids_n128g27tower",
        "ids_n128g28tower", "ids_n128g29tower", "ids_n128g30tower", "ids_n128g31tower",
        "ids_n128g32tower", "ids_n128g33tower",
        "swaps", "not", "swapsnot1", "swapsnot2", "swapsnot12", "cnot"
    ];

    for name in db_names.iter() {
        match env.open_db(Some(name)) {
            Ok(db) => { dbs.insert(name.to_string(), db); }
            Err(lmdb::Error::NotFound) => continue,
            Err(e) => panic!("Failed to open LMDB database {}: {:?}", name, e),
        }
    }

    dbs
}
// All the main code take a circuit and do repeated rounds of whatever method is chosen
// In between each round, store a progress circuit and a sanity check
// Finally, record the circuit in the chosen file destination
pub fn main_mix(c: &CircuitSeq, rounds: usize, conn: &mut Connection, n: usize) {
    // Start with the input circuit
    println!("Starting len: {}", c.gates.len());
    let mut circuit = c.clone();
    let perms: Vec<Vec<usize>> = (0..n).permutations(n).collect();
    let bit_shuf = perms.into_iter().skip(1).collect::<Vec<_>>();
    // Repeat obfuscate + compress 'rounds' times
    let mut post_len = 0;
    let mut count = 0;
    for _ in 0..rounds {
        circuit = obfuscate_and_target_compress(&circuit, conn, &bit_shuf, n);
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
    }
    let mut i = 0;
    while i < circuit.gates.len().saturating_sub(1) {
        if circuit.gates[i] == circuit.gates[i + 1] {
            // remove elements at i and i+1
            circuit.gates.drain(i..=i + 1);

            // step back up to 2 indices, but not below 0
            i = i.saturating_sub(2);
        } else {
            i += 1;
        }
    }
    // Convert the final circuit to string
    let circuit_str = circuit.to_string(n);
    println!("{:?}", circuit.permutation(n).data);
    // Write to file
    let mut file = File::create("recent_circuit.txt").expect("Failed to create file");
    file.write_all(circuit_str.as_bytes())
        .expect("Failed to write circuit to file");

    let circuit_str = circuit.repr(); // or however you stringify your circuit

    if !circuit.gates.is_empty() {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .append(true) // append instead of overwriting
            .open("good_id.txt")
            .expect("Failed to open good_id.txt");

        let line = format!("{} : {}\n", circuit.gates.len(), circuit_str);

        file.write_all(line.as_bytes())
            .expect("Failed to write circuit to good_ids.txt");

        println!("Wrote good circuit to good_id.txt");
    }

    if circuit.gates == c.gates {
        println!("The obfuscation didn't do anything");
    }

    println!("Final circuit written to recent_circuit.txt");
}

pub fn main_butterfly(c: &CircuitSeq, rounds: usize, conn: &mut Connection, n: usize) {
    // Start with the input circuit
    println!("Starting len: {}", c.gates.len());
    let mut circuit = c.clone();
    let perms: Vec<Vec<usize>> = (0..n).permutations(n).collect();
    let bit_shuf = perms.into_iter().skip(1).collect::<Vec<_>>();
    // Repeat obfuscate + compress 'rounds' times
    let mut post_len = 0;
    let mut count = 0;
    for _ in 0..rounds {
        circuit = butterfly(&circuit, conn, &bit_shuf, n);
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
        let mut i = 0;
        while i < circuit.gates.len().saturating_sub(1) {
            if circuit.gates[i] == circuit.gates[i + 1] {
                // remove elements at i and i+1
                circuit.gates.drain(i..=i + 1);

                // step back up to 2 indices, but not below 0
                i = i.saturating_sub(2);
            } else {
                i += 1;
            }
        }
    }
    println!("Final len: {}", circuit.gates.len());
    println!("Final cycle: {:?}", circuit.permutation(n).to_cycle());
    // Convert the final circuit to string
    let circuit_str = circuit.to_string(n);
    println!("Final Permutation: {:?}", circuit.permutation(n).data);
    if circuit.permutation(n).data != c.permutation(n).data {
        panic!(
            "The permutation differs from the original.\nOriginal: {:?}\nNew: {:?}",
            c.permutation(n).data,
            circuit.permutation(n).data
        );
    }
    // Write to file
    let mut file = File::create("recent_circuit.txt").expect("Failed to create file");
    file.write_all(circuit_str.as_bytes())
        .expect("Failed to write circuit to file");

    let circuit_str = circuit.repr(); // or however you stringify your circuit

    if !circuit.gates.is_empty() {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .append(true) // append instead of overwriting
            .open("good_id.txt")
            .expect("Failed to open good_id.txt");

        let line = format!("{} : {}\n", circuit.gates.len(), circuit_str);

        file.write_all(line.as_bytes())
            .expect("Failed to write circuit to good_ids.txt");

        println!("Wrote good circuit to good_id.txt");
    }

    if circuit.gates == c.gates {
        println!("The obfuscation didn't do anything");
    }

    println!("Final circuit written to recent_circuit.txt");
}

pub fn main_butterfly_big(c: &CircuitSeq, rounds: usize, conn: &mut Connection, n: usize, asymmetric: bool, save: &str, env: &lmdb::Environment,) {
    // Start with the input circuit
    let bit_shuf_list = (3..=7)
        .map(|n| {
            (0..n)
                .permutations(n)
                .filter(|p| !p.iter().enumerate().all(|(i, &x)| i == x))
                .collect::<Vec<Vec<usize>>>()
        })
        .collect();
    let dbs = open_all_dbs(env);
    println!("Starting len: {}", c.gates.len());
    let mut circuit = c.clone();
    // Repeat obfuscate + compress 'rounds' times
    let mut post_len = 0;
    let mut count = 0;
    for i in 0..rounds {
        let stop = 1000;
        circuit = if asymmetric {
            // abutterfly_big(&circuit, conn, n, i != rounds-1, std::cmp::min(stop*(i+1), 5000), env, i+1, rounds, &bit_shuf_list, &dbs)
            abutterfly_big(&circuit, conn, n, i != rounds-1, 100, env, i+1, rounds, &bit_shuf_list, &dbs)
        } else {
            butterfly_big(&circuit,conn,n, i != rounds-1, stop*(i+1), env, &bit_shuf_list, &dbs)
        };
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
        let mut i = 0;
        while i < circuit.gates.len().saturating_sub(1) {
            if circuit.gates[i] == circuit.gates[i + 1] {
                // remove elements at i and i+1
                circuit.gates.drain(i..=i + 1);

                // step back up to 2 indices, but not below 0
                i = i.saturating_sub(2);
            } else {
                i += 1;
            }
        }
    }
    println!("Final len: {}", circuit.gates.len());
    circuit
    .probably_equal(&c, n, 150_000)
    .expect("The circuits differ somewhere!");

    // Write to file
    let c_str = c.repr();
    let circuit_str = circuit.repr();
    let long_str = format!("{}:{}", c.repr(), circuit.repr());
    // let good_str = format!("{}: {}", good_id.gates.len(), good_id.repr());
    // Write start.txt
    File::create("start.txt")
        .and_then(|mut f| f.write_all(c_str.as_bytes()))
        .expect("Failed to write start.txt");

    // Write recent_circuit.txt
    File::create("recent_circuit.txt")
        .and_then(|mut f| f.write_all(circuit_str.as_bytes()))
        .expect("Failed to write recent_circuit.txt");

    File::create(save)
        .and_then(|mut f| f.write_all(circuit_str.as_bytes()))
        .expect("Failed to write recent_circuit.txt");

    // Write butterfly_recent.txt (overwrite)
    File::create("butterfly_recent.txt")
        .and_then(|mut f| f.write_all(long_str.as_bytes()))
        .expect("Failed to write butterfly_recent.txt");

    // Append to butterfly.txt
    OpenOptions::new()
        .append(true)
        .create(true)
        .open("butterfly.txt")
        .and_then(|mut f| writeln!(f, "{}", long_str))
        .expect("Failed to append to butterfly.txt");
    if circuit.gates == c.gates {
        println!("The obfuscation didn't do anything");
    }

    println!("Final circuit written to recent_circuit.txt");
}

pub fn main_rac_big(c: &CircuitSeq, rounds: usize, conn: &mut Connection, n: usize, save: &str, env: &lmdb::Environment, intermediate: &str, tower: bool) {
    // Start with the input circuit
    let save_base = save.strip_suffix(".txt").unwrap_or(save);
    let progress_path = format!("{}_progress.txt", save_base);
    let mut sum_already_coll = 0usize;
    let mut sum_shoot = 0usize;
    let mut sum_made_left = 0usize;
    let mut sum_traverse_left = 0usize;
    OpenOptions::new()
    .create(true)
    .write(true)
    .truncate(true)
    .open(&progress_path)
    .expect("Failed to create progress file");
    let bit_shuf_list = (3..=7)
        .map(|n| {
            (0..n)
                .permutations(n)
                .filter(|p| !p.iter().enumerate().all(|(i, &x)| i == x))
                .collect::<Vec<Vec<usize>>>()
        })
        .collect();
    let dbs = open_all_dbs(env);
    println!("Starting len: {}", c.gates.len());
    let mut circuit = c.clone();
    // Repeat obfuscate + compress 'rounds' times
    let mut post_len = 0;
    let mut count = 0;
    for i in 0..rounds {
        let _stop = 1000;
        let (new_circuit, already_coll, shoot, made_left, traverse_left)  = replace_and_compress_big(&circuit, conn, n, i != rounds-1, 100, env, i+1, rounds, &bit_shuf_list, &dbs, intermediate, tower);
        circuit = new_circuit;

        sum_already_coll += already_coll;
        sum_shoot += shoot;
        sum_made_left += made_left;
        sum_traverse_left += traverse_left;

        let total_attempts = already_coll + shoot;
        let already_coll_pct = if total_attempts > 0 {
            already_coll as f64 / total_attempts as f64 * 100.0
        } else { 0.0 };
        let shoot_pct = if total_attempts > 0 {
            shoot as f64 / total_attempts as f64 * 100.0
        } else { 0.0 };
        let made_left_pct = if shoot > 0 {
            made_left as f64 / shoot as f64 * 100.0
        } else { 0.0 };
        let traverse_left_avg = if shoot > 0 {
            traverse_left as f64 / shoot as f64
        } else { 0.0 };

        println!(
            "Round {} stats: Total Attempts: {} | Already-collided {:.2}% | Shoot {:.2}% | Made-left {:.2}% | Traverse-left avg {:.2}",
            i + 1, total_attempts, already_coll_pct, shoot_pct, made_left_pct, traverse_left_avg
        );

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
        if c.probably_equal(&circuit, n, 100_000).is_err() {
            panic!("The functionality has changed");
        }
        {
        println!("Updating progress {}", progress_path);
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&progress_path)
            .expect("Failed to open progress file");

        writeln!(
            f,
            "=== Round {} ===\n{}\n",
            i + 1,
            circuit.repr()
        )
        .expect("Failed to write progress");
        }
    }

    let total_attempts = sum_already_coll + sum_shoot;
    let overall_already_coll_pct = if total_attempts > 0 {
        sum_already_coll as f64 / total_attempts as f64 * 100.0
    } else { 0.0 };
    let overall_shoot_pct = if total_attempts > 0 {
        sum_shoot as f64 / total_attempts as f64 * 100.0
    } else { 0.0 };
    let overall_made_left_pct = if sum_shoot > 0 {
        sum_made_left as f64 / sum_shoot as f64 * 100.0
    } else { 0.0 };
    let overall_traverse_left_avg = if sum_made_left > 0 {
        sum_traverse_left as f64 / sum_made_left as f64
    } else { 0.0 };

    println!("=== Overall Stats ===");
    println!(
        "Total Attempts {} \n Already-collided {:.2}% | Shoot {:.2}% | Made-left {:.2}% | Traverse-left avg {:.2}",
        total_attempts,
        overall_already_coll_pct,
        overall_shoot_pct,
        overall_made_left_pct,
        overall_traverse_left_avg
    );

    println!("Final len: {}", circuit.gates.len());
    circuit
    .probably_equal(&c, n, 150_000)
    .expect("The circuits differ somewhere!");

    // Write to file
    let circuit_str = circuit.repr();
    // let good_str = format!("{}: {}", good_id.gates.len(), good_id.repr());
    File::create(save)
        .and_then(|mut f| f.write_all(circuit_str.as_bytes()))
        .expect("Failed to write recent_circuit.txt");

    println!("Final circuit written to recent_circuit.txt");
}

pub fn main_interleave_big(c: &CircuitSeq, rounds: usize, conn: &mut Connection, n: usize, save: &str, env: &lmdb::Environment, intermediate: &str, tower: bool) {
    // Start with the input circuit
    let save_base = save.strip_suffix(".txt").unwrap_or(save);
    let progress_path = format!("{}_progress.txt", save_base);
    OpenOptions::new()
    .create(true)
    .write(true)
    .truncate(true)
    .open(&progress_path)
    .expect("Failed to create progress file");
    let bit_shuf_list = (3..=7)
        .map(|n| {
            (0..n)
                .permutations(n)
                .filter(|p| !p.iter().enumerate().all(|(i, &x)| i == x))
                .collect::<Vec<Vec<usize>>>()
        })
        .collect();
    let dbs = open_all_dbs(env);
    println!("Starting len: {}", c.gates.len());
    let mut circuit = c.clone();
    // Repeat obfuscate + compress 'rounds' times
    let mut post_len = 0;
    let mut count = 0;
    let mut n = n;
    for i in 0..rounds {
        let _stop = 1000;
        let (new_circuit, _, _, _, _) = if i == 0 { 
            let x = interleave_sequential_big(&circuit, conn, n, i != rounds-1, 100, env, i+1, rounds, &bit_shuf_list, &dbs, intermediate, tower);
            n *= 2;
            x
        } else {
            replace_and_compress_big(&circuit, conn, n, i != rounds-1, 100, env, i+1, rounds, &bit_shuf_list, &dbs, intermediate, tower) 
        };
        circuit = new_circuit;

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
        if c.probably_equal(&circuit, n/2, 100_000).is_err() {
            panic!("The functionality has changed");
        }
        {
        println!("Updating progress {}", progress_path);
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&progress_path)
            .expect("Failed to open progress file");

        writeln!(
            f,
            "=== Round {} ===\n{}\n",
            i + 1,
            circuit.repr()
        )
        .expect("Failed to write progress");
        }
    }

    println!("Final len: {}", circuit.gates.len());
    circuit
    .probably_equal(&c, n/2, 150_000)
    .expect("The circuits differ somewhere!");

    // Write to file
    let circuit_str = circuit.repr();
    // let good_str = format!("{}: {}", good_id.gates.len(), good_id.repr());
    File::create(save)
        .and_then(|mut f| f.write_all(circuit_str.as_bytes()))
        .expect("Failed to write recent_circuit.txt");

    println!("Final circuit written to recent_circuit.txt");
}

pub fn main_shuffle_rcs_big(c: &CircuitSeq, rounds: usize, conn: &mut Connection, n: usize, save: &str, env: &lmdb::Environment, intermediate: &str, tower: bool) {
    // Start with the input circuit
    let save_base = save.strip_suffix(".txt").unwrap_or(save);
    let progress_path = format!("{}_progress.txt", save_base);
    OpenOptions::new()
    .create(true)
    .write(true)
    .truncate(true)
    .open(&progress_path)
    .expect("Failed to create progress file");
    let bit_shuf_list = (3..=7)
        .map(|n| {
            (0..n)
                .permutations(n)
                .filter(|p| !p.iter().enumerate().all(|(i, &x)| i == x))
                .collect::<Vec<Vec<usize>>>()
        })
        .collect();
    let dbs = open_all_dbs(env);
    println!("Starting len: {}", c.gates.len());
    let mut circuit = c.clone();
    // Repeat obfuscate + compress 'rounds' times
    let mut post_len = 0;
    let mut count = 0;
    println!("Inserting shuffles");
    insert_wire_shuffles(&mut circuit, n, env, &dbs);
    if c.probably_equal(&circuit, n, 1_000).is_err() {
        panic!("Lost functionality after shuffles");
    } else {
        println!("Length after shuffles: {} gates", circuit.gates.len());
    }
    for i in 0..rounds {
        let _stop = 1000;
        let (new_circuit, _, _, _, _) = 
            replace_and_compress_big(&circuit, conn, n, i != rounds-1, 100, env, i+1, rounds, &bit_shuf_list, &dbs, intermediate, tower);
        circuit = new_circuit;

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
        if c.probably_equal(&circuit, n, 100_000).is_err() {
            panic!("The functionality has changed");
        }
        {
        println!("Updating progress {}", progress_path);
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&progress_path)
            .expect("Failed to open progress file");

        writeln!(
            f,
            "=== Round {} ===\n{}\n",
            i + 1,
            circuit.repr()
        )
        .expect("Failed to write progress");
        }
    }

    println!("Final len: {}", circuit.gates.len());
    circuit
    .probably_equal(&c, n/2, 150_000)
    .expect("The circuits differ somewhere!");

    // Write to file
    let circuit_str = circuit.repr();
    // let good_str = format!("{}: {}", good_id.gates.len(), good_id.repr());
    File::create(save)
        .and_then(|mut f| f.write_all(circuit_str.as_bytes()))
        .expect("Failed to write recent_circuit.txt");

    println!("Final circuit written to recent_circuit.txt");
}

pub fn main_rac_big_distance(c: &CircuitSeq, rounds: usize, conn: &mut Connection, n: usize, save: &str, env: &lmdb::Environment, intermediate: &str, min: usize, tower: bool,) {
    // Start with the input circuit
    let save_base = save.strip_suffix(".txt").unwrap_or(save);
    let progress_path = format!("{}_progress.txt", save_base);
    OpenOptions::new()
    .create(true)
    .write(true)
    .truncate(true)
    .open(&progress_path)
    .expect("Failed to create progress file");
    let bit_shuf_list = (3..=7)
        .map(|n| {
            (0..n)
                .permutations(n)
                .filter(|p| !p.iter().enumerate().all(|(i, &x)| i == x))
                .collect::<Vec<Vec<usize>>>()
        })
        .collect();
    let dbs = open_all_dbs(env);
    println!("Starting len: {}", c.gates.len());
    let mut circuit = c.clone();
    // Repeat obfuscate + compress 'rounds' times
    let mut post_len = 0;
    let mut count = 0;
    for i in 0..rounds {
        let _stop = 1000;
        let new_circuit = replace_and_compress_big_distance(&circuit, conn, n, i != rounds-1, 100, env, i+1, rounds, &bit_shuf_list, &dbs, intermediate, min, tower);
        circuit = new_circuit;

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
        if c.probably_equal(&circuit, n, 100_000).is_err() {
            panic!("The functionality has changed");
        }
        {
        println!("Updating progress {}", progress_path);
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&progress_path)
            .expect("Failed to open progress file");

        writeln!(
            f,
            "=== Round {} ===\n{}\n",
            i + 1,
            circuit.repr()
        )
        .expect("Failed to write progress");
        }
    }

    println!("Final len: {}", circuit.gates.len());
    circuit
    .probably_equal(&c, n, 150_000)
    .expect("The circuits differ somewhere!");

    // Write to file
    let circuit_str = circuit.repr();
    // let good_str = format!("{}: {}", good_id.gates.len(), good_id.repr());
    File::create(save)
        .and_then(|mut f| f.write_all(circuit_str.as_bytes()))
        .expect("Failed to write recent_circuit.txt");

    println!("Final circuit written to recent_circuit.txt");
}

// Currently unsupported
// pub fn main_butterfly_big_bookendsless(c: &CircuitSeq, rounds: usize, conn: &mut Connection, n: usize, _asymmetric: bool, save: &str ,env: &lmdb::Environment,) {
//     // Start with the input circuit
//     let dbs = open_all_dbs(env);
//     let bit_shuf_list = (3..=7)
//         .map(|n| {
//             (0..n)
//                 .permutations(n)
//                 .filter(|p| !p.iter().enumerate().all(|(i, &x)| i == x))
//                 .collect::<Vec<Vec<usize>>>()
//         })
//         .collect();
//     println!("Starting len: {}", c.gates.len());
//     let mut circuit = c.clone();
//     // Repeat obfuscate + compress 'rounds' times
//     let mut post_len = 0;
//     let mut count = 0;
//     let mut beginning = CircuitSeq { gates: Vec::new() };
//     let mut end= CircuitSeq { gates: Vec::new() };
//     for _ in 0..rounds {
//         let (new_circuit, b, e) = abutterfly_big_delay_bookends(&circuit, conn, n, env);
//         beginning = beginning.concat(&b);
//         end = e.concat(&end);
//         circuit = new_circuit;
//         if circuit.gates.len() == 0 {
//             break;
//         }
        
//         if circuit.gates.len() == post_len {
//             count += 1;
//         } else {
//             post_len = circuit.gates.len();
//             count = 0;
//         }

//         if count > 2 {
//             break;
//         }
//         let mut i = 0;
//         while i < circuit.gates.len().saturating_sub(1) {
//             if circuit.gates[i] == circuit.gates[i + 1] {
//                 // remove elements at i and i+1
//                 circuit.gates.drain(i..=i + 1);

//                 // step back up to 2 indices, but not below 0
//                 i = i.saturating_sub(2);
//             } else {
//                 i += 1;
//             }
//         }
//     }
//     let txn = env.begin_ro_txn().expect("txn");
//     println!("Adding bookends");
//     beginning = compress_big(&beginning, 100, n, conn, env, &bit_shuf_list, &dbs, &txn);
//     end = compress_big(&end, 100, n, conn, env, &bit_shuf_list, &dbs, &txn);
//     circuit = beginning.concat(&circuit).concat(&end);
//     let mut c1 = CircuitSeq{ gates: circuit.gates[0..circuit.gates.len()/2].to_vec() };
//     let mut c2 = CircuitSeq{ gates: circuit.gates[circuit.gates.len()/2..].to_vec() };
//     c1 = compress_big(&c1, 1_000, n, conn, env, &bit_shuf_list, &dbs, &txn);
//     c2 = compress_big(&c2, 1_000, n, conn, env, &bit_shuf_list, &dbs, &txn);
//     circuit = c1.concat(&c2);
//     let mut stable_count = 0;
//     while stable_count < 3 {
//         let before = circuit.gates.len();
//         //shoot_random_gate(&mut acc, 100_000);
//         circuit = compress_big(&circuit, 1_000, n, conn, env, &bit_shuf_list, &dbs ,&txn);
//         let after = circuit.gates.len();

//         if after == before {
//             stable_count += 1;
//             println!("  Final compression stable {}/3 at {} gates", stable_count, after);
//         } else {
//             println!("  Final compression reduced: {} → {} gates", before, after);
//             stable_count = 0;
//         }
//     }

//     println!("Final len: {}", circuit.gates.len());

//     circuit
//     .probably_equal(&c, n, 150_000)
//     .expect("The circuits differ somewhere!");

//     // Write to file
//     let c_str = c.repr();
//     let circuit_str = circuit.repr();
//     let long_str = format!("{}:{}", c.repr(), circuit.repr());
//     // let good_str = format!("{}: {}", good_id.gates.len(), good_id.repr());
//     // Write start.txt
//     File::create("start.txt")
//         .and_then(|mut f| f.write_all(c_str.as_bytes()))
//         .expect("Failed to write start.txt");

//     // Write recent_circuit.txt
//     File::create("recent_circuit.txt")
//         .and_then(|mut f| f.write_all(circuit_str.as_bytes()))
//         .expect("Failed to write recent_circuit.txt");

//     File::create(save)
//         .and_then(|mut f| f.write_all(circuit_str.as_bytes()))
//         .expect("Failed to write recent_circuit.txt");

//     // Write butterfly_recent.txt (overwrite)
//     File::create("butterfly_recent.txt")
//         .and_then(|mut f| f.write_all(long_str.as_bytes()))
//         .expect("Failed to write butterfly_recent.txt");

//     // Append to butterfly.txt
//     OpenOptions::new()
//         .append(true)
//         .create(true)
//         .open("butterfly.txt")
//         .and_then(|mut f| writeln!(f, "{}", long_str))
//         .expect("Failed to append to butterfly.txt");
//     if circuit.gates == c.gates {
//         println!("The obfuscation didn't do anything");
//     }

//     println!("Final circuit written to recent_circuit.txt");
// }

//do targeted compression
pub fn main_compression(c: &CircuitSeq, rounds: usize, conn: &mut Connection, n: usize, save: &str, env: &lmdb::Environment,) {
    let dbs = open_all_dbs(env);
    // Start with the input circuit
    let bit_shuf_list = (3..=7)
        .map(|n| {
            (0..n)
                .permutations(n)
                .filter(|p| !p.iter().enumerate().all(|(i, &x)| i == x))
                .collect::<Vec<Vec<usize>>>()
        })
        .collect();
    println!("Starting len: {}", c.gates.len());
    let mut circuit = c.clone();
    // Repeat obfuscate + compress 'rounds' times
    let mut post_len = 0;
    let mut count = 0;
    for _ in 0..rounds {
            butterfly_big(&circuit,conn,n, false, 0, env, &bit_shuf_list, &dbs);
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
        let mut i = 0;
        while i < circuit.gates.len().saturating_sub(1) {
            if circuit.gates[i] == circuit.gates[i + 1] {
                // remove elements at i and i+1
                circuit.gates.drain(i..=i + 1);

                // step back up to 2 indices, but not below 0
                i = i.saturating_sub(2);
            } else {
                i += 1;
            }
        }
    }
    println!("Final len: {}", circuit.gates.len());

    circuit
    .probably_equal(&c, n, 150_000)
    .expect("The circuits differ somewhere!");

    // Write to file
    let c_str = c.repr();
    let circuit_str = circuit.repr();
    let long_str = format!("{}:{}", c.repr(), circuit.repr());
    // Write start.txt
    File::create("start.txt")
        .and_then(|mut f| f.write_all(c_str.as_bytes()))
        .expect("Failed to write start.txt");

    // Write recent_circuit.txt
    File::create("recent_circuit.txt")
        .and_then(|mut f| f.write_all(circuit_str.as_bytes()))
        .expect("Failed to write recent_circuit.txt");

    File::create(save)
        .and_then(|mut f| f.write_all(circuit_str.as_bytes()))
        .expect("Failed to write recent_circuit.txt");

    // Write butterfly_recent.txt (overwrite)
    File::create("butterfly_recent.txt")
        .and_then(|mut f| f.write_all(long_str.as_bytes()))
        .expect("Failed to write butterfly_recent.txt");

    // Append to butterfly.txt
    OpenOptions::new()
        .append(true)
        .create(true)
        .open("butterfly.txt")
        .and_then(|mut f| writeln!(f, "{}", long_str))
        .expect("Failed to append to butterfly.txt");
    if circuit.gates == c.gates {
        println!("The obfuscation didn't do anything");
    }

    println!("Final circuit written to recent_circuit.txt");
}