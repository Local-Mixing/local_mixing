use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io::Write,
};

use itertools::Itertools;

use crate::{
    circuit::circuit::CircuitSeq,
    replace::{
        replace::{compress_loop, compress_loop_early, expand_loop, expand_once, ExpandPairMode},
        mixing::{
            abutterfly_big, butterfly_big, interleave_sequential_big, replace_and_compress_big, replace_and_compress_big_distance, simple_shooting_game, zip_sequential_butterfly
        },
        transpositions::{
            insert_wire_m_samfs_every_x,
            //insert_wire_shuffles_knuth,
            //insert_wire_shuffles_simple,
            insert_wire_shuffles_x
        },
        pairs::{interleave},
        gadgets::gadgetize,
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

pub fn open_all_dbs(env: &lmdb::Environment) -> HashMap<String, lmdb::Database> {
    let mut dbs = HashMap::new();

    let named = [
        "swap", "not", "swapnot1", "swapnot2", "swapnot12", "cnot", "homgad"
    ];
    for name in named.iter() {
        match env.open_db(Some(name)) {
            Ok(db) => { dbs.insert(name.to_string(), db); }
            Err(lmdb::Error::NotFound) => {}
            Err(e) => panic!("Failed to open LMDB database {}: {:?}", name, e),
        }
    }

    for i in 0..34usize {
        let name = format!("id_g{}", i);
        match env.open_db(Some(name.as_str())) {
            Ok(db) => { dbs.insert(name, db); }
            Err(lmdb::Error::NotFound) => {}
            Err(e) => panic!("Failed to open id_g{}: {:?}", i, e),
        }
    }

    dbs
}
pub fn main_butterfly_big(c: &CircuitSeq, rounds: usize, n: usize, asymmetric: bool, save: &str, env: &lmdb::Environment,
    shard_dbs: &[lmdb::Database],) {
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
            abutterfly_big(&circuit, n, i != rounds-1, 100, env, i+1, rounds, &bit_shuf_list, &dbs, shard_dbs)
        } else {
            butterfly_big(&circuit, n, i != rounds-1, stop*(i+1), env, shard_dbs)
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

pub fn main_rac_big(c: &CircuitSeq, rounds: usize, n: usize, save: &str, env: &lmdb::Environment,
    shard_dbs: &[lmdb::Database], intermediate: &str, tower: bool, id_len: usize) {
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
        let (new_circuit, already_coll, shoot, made_left, traverse_left)  = replace_and_compress_big(&circuit, n, i != rounds-1, 100, env, i+1, rounds, &bit_shuf_list, &dbs, shard_dbs, intermediate, tower, id_len);
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

pub fn main_interleave_big(c: &CircuitSeq, rounds: usize, n: usize, save: &str, env: &lmdb::Environment,
    shard_dbs: &[lmdb::Database], intermediate: &str, tower: bool, id_len: usize) {
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
            let x = interleave_sequential_big(&circuit, n, i != rounds-1, 100, env, i+1, rounds, &bit_shuf_list, &dbs, shard_dbs, intermediate, tower, id_len);
            n *= 2;
            x
        } else {
            replace_and_compress_big(&circuit, n, i != rounds-1, 100, env, i+1, rounds, &bit_shuf_list, &dbs, shard_dbs, intermediate, tower, id_len)
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

pub fn main_shuffle_rcs_big(c: &CircuitSeq, rounds: usize, n: usize, save: &str, env: &lmdb::Environment,
    shard_dbs: &[lmdb::Database], intermediate: &str, tower: bool, x: usize, id_len: usize) {
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
    if x == 0 {
        insert_wire_shuffles_x(&mut circuit, n, env, &dbs, c.gates.len() - 1);
    }
    if c.probably_equal(&circuit, n, 1_000).is_err() {
        panic!("Lost functionality after shuffles");
    } else {
        println!("Length after shuffles: {} gates", circuit.gates.len());
    }
    for i in 0..rounds {
        let _stop = 1000;
        if x != 0 {
            insert_wire_shuffles_x(&mut circuit, n, env, &dbs, x);
        }
        let (new_circuit, _, _, _, _) =
            replace_and_compress_big(&circuit, n, i != rounds-1, 100, env, i+1, rounds, &bit_shuf_list, &dbs, shard_dbs, intermediate, tower, id_len);
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

pub fn main_rac_big_distance(c: &CircuitSeq, rounds: usize, n: usize, save: &str, env: &lmdb::Environment,
    shard_dbs: &[lmdb::Database], intermediate: &str, min: usize, tower: bool, id_len: usize) {
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
        let new_circuit = replace_and_compress_big_distance(&circuit, n, i != rounds-1, 100, env, i+1, rounds, &bit_shuf_list, &dbs, shard_dbs, intermediate, min, tower, id_len);
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

pub fn main_sequential_butterfly(
    c: &CircuitSeq,
    rounds: usize,
    n: usize,
    save: &str,
    env: &lmdb::Environment,
    shard_dbs: &[lmdb::Database],
    id_len: usize,
    reverse_order_left: bool,
    tower_left: bool,
    shoot_more_left: u8,
    reverse_order_right: bool,
    tower_right: bool,
    shoot_more_right: u8
) {
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
        let new_circuit = zip_sequential_butterfly(
            &circuit,
            n,
            env,
            i+1,
            rounds,
            &bit_shuf_list,
            &dbs,
            shard_dbs,
            id_len,
            reverse_order_left,
            tower_left,
            shoot_more_left,
            reverse_order_right,
            tower_right,
            shoot_more_right
        );
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
    File::create(save)
        .and_then(|mut f| f.write_all(circuit_str.as_bytes()))
        .expect("Failed to write recent_circuit.txt");

    println!("Final circuit written to {}", save);
}

pub fn main_shooting_game(
    c: &CircuitSeq,
    rounds: usize,
    n: usize,
    save: &str,
    env: &lmdb::Environment,
    shard_dbs: &[lmdb::Database],
    id_len: usize,
    tower: bool,
    stop_multiplier: f64,
    intermediate: &str,
    gates_ahead: usize,
    partial_stop: f64,
) {
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
    let curated_shard_dbs = open_curated_shard_dbs(env);
    println!("Starting len: {}", c.gates.len());
    let mut circuit = c.clone();
    // Repeat `rounds` times
    let mut post_len = 0;
    let mut count = 0;
    for i in 0..rounds {
        let stop_gates = (circuit.gates.len() as f64 * stop_multiplier) as usize;
        let new_circuit = simple_shooting_game(
            &circuit,
            n,
            env,
            i + 1,
            rounds,
            &bit_shuf_list,
            &dbs,
            id_len,
            tower,
            stop_gates,
            intermediate,
            false,
            100,
            gates_ahead,
            &curated_shard_dbs,
            shard_dbs,
        );
        circuit = new_circuit;

        if circuit.gates.len() == 0 {
            break;
        }

        let is_last = i + 1 == rounds;
        circuit = if is_last {
            compress_loop(&circuit, n, env, shard_dbs, 12, i + 1, rounds, "temp_compression.txt")
        } else {
            let early_stop = (circuit.gates.len() as f64 * partial_stop) as usize;
            compress_loop_early(&circuit, n, env, shard_dbs, 12, i + 1, rounds, "temp_compression.txt", early_stop)
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
    File::create(save)
        .and_then(|mut f| f.write_all(circuit_str.as_bytes()))
        .expect("Failed to write recent_circuit.txt");

    println!("Final circuit written to {}", save);
}

pub fn main_expansion_game(
    c: &CircuitSeq,
    rounds: usize,
    n: usize,
    save: &str,
    env: &lmdb::Environment,
    shard_dbs: &[lmdb::Database],
    id_len: usize,
    tower: bool,
    target_multiplier: usize,
    intermediate: &str,
    curated: bool,
    use_db: bool,
) {
    let save_base = save.strip_suffix(".txt").unwrap_or(save);
    let progress_path = format!("{}_progress.txt", save_base);
    OpenOptions::new()
        .create(true).write(true).truncate(true)
        .open(&progress_path)
        .expect("Failed to create progress file");

    let bit_shuf_list: Vec<Vec<Vec<usize>>> = (3..=7)
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
    let mut post_len = 0;
    let mut count = 0;

    for i in 0..rounds {
        let pair_mode = if curated {
            ExpandPairMode::Curated
        } else if use_db {
            ExpandPairMode::Db
        } else {
            ExpandPairMode::Canonical {
                bit_shuf_list: &bit_shuf_list,
                dbs: &dbs,
                id_len,
                tower,
            }
        };

        println!("Round {}/{}: Expanding to {}x ({} gates)...", i + 1, rounds, target_multiplier, circuit.gates.len());
        circuit = expand_loop(&circuit, n, env, shard_dbs, target_multiplier, &pair_mode);

        if circuit.gates.is_empty() { break; }

        println!("Round {}/{}: Compressing ({} gates)...", i + 1, rounds, circuit.gates.len());
        let is_last = i + 1 == rounds;
        circuit = if is_last {
            compress_loop(&circuit, n, env, shard_dbs, 12, i + 1, rounds, intermediate)
        } else {
            let early_stop = circuit.gates.len() / 2;
            compress_loop_early(&circuit, n, env, shard_dbs, 12, i + 1, rounds, intermediate, early_stop)
        };

        if circuit.gates.is_empty() { break; }

        if circuit.gates.len() == post_len {
            count += 1;
        } else {
            post_len = circuit.gates.len();
            count = 0;
        }
        if count > 2 { break; }

        let mut j = 0;
        while j < circuit.gates.len().saturating_sub(1) {
            if circuit.gates[j] == circuit.gates[j + 1] {
                circuit.gates.drain(j..=j + 1);
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
                .create(true).append(true)
                .open(&progress_path)
                .expect("Failed to open progress file");
            writeln!(f, "=== Round {} ===\n{}\n", i + 1, circuit.repr())
                .expect("Failed to write progress");
        }
    }

    println!("Final len: {}", circuit.gates.len());
    circuit.probably_equal(&c, n, 150_000)
        .expect("The circuits differ somewhere!");

    let circuit_str = circuit.repr();
    File::create(save)
        .and_then(|mut f| f.write_all(circuit_str.as_bytes()))
        .expect("Failed to write output");

    println!("Final circuit written to {}", save);
}

pub fn main_shuffle_shoot_shuffle(
    c: &CircuitSeq,
    rounds: usize,
    n: usize,
    m: usize,
    x: usize,
    save: &str,
    env: &lmdb::Environment,
    shard_dbs: &[lmdb::Database],
    id_len: usize,
    tower: bool,
    _stop: usize,
    intermediate: &str,
    leave: bool,
    do_gadgetize: bool,
    full_shuffle: bool,
    gates_ahead: usize,
    egg: bool,
) {
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
    let curated_shard_dbs = open_curated_shard_dbs(env);
    println!("Starting len: {}", c.gates.len());
    let mut circuit = c.clone();
    // Repeat `rounds` times
    let mut post_len = 0;
    let mut count = 0;
    if do_gadgetize {
        let mut rng = rand::rng();
        let before = circuit.gates.len();
        circuit = gadgetize(&circuit, n, &mut rng);
        println!("Gadgetized: {} gates → {} gates, {} wires", before, circuit.gates.len(), 2 * n + 1);
    }
    let n = if do_gadgetize { 2 * n + 1 } else { n };
    if leave {
        circuit = interleave(
            &circuit,
            n,
            env,
            &dbs,
            &bit_shuf_list,
            tower,
            id_len,
        );
    }
    let n = if leave {
        2 * n
    } else {
        n
    };
    if full_shuffle {
        loop {
            let mut shuffled = circuit.clone();
            insert_wire_m_samfs_every_x(&mut shuffled, n, n, 1, env, &dbs);
            if shuffled.probably_equal(&circuit, n, 100).is_ok() {
                circuit = shuffled;
                break;
            }
        }
        println!("After full shuffle: {} gates", circuit.gates.len());
    }
    for i in 0..rounds {
        if egg {
            let pair_mode = ExpandPairMode::Curated;
            circuit = expand_once(&circuit, n, env, shard_dbs, &pair_mode);
        } else {
            loop {
                let new_circuit = simple_shooting_game(
                    &circuit,
                    n,
                    env,
                    i+1,
                    rounds,
                    &bit_shuf_list,
                    &dbs,
                    id_len,
                    tower,
                    4 * circuit.gates.len(),
                    intermediate,
                    true,
                    1,
                    gates_ahead,
                    &curated_shard_dbs,
                    shard_dbs,
                );
                if new_circuit.probably_equal(&circuit, n, 100).is_ok() {
                    circuit = new_circuit;
                    break;
                }
            }
        }
        println!("After shooting game: {} gates", circuit.gates.len());
        let mut new_circuit = circuit.clone();
        loop {
            insert_wire_m_samfs_every_x(&mut new_circuit, n, m, x, env, &dbs);
            if new_circuit.probably_equal(&circuit, n, 100).is_ok() {
                circuit = new_circuit;
                break;
            }
            new_circuit = circuit.clone();
        }
        println!("After inserting samfs: {} gates", circuit.gates.len());
        circuit = compress_loop(&circuit, n, env, shard_dbs, 6, i+1, rounds, "temp_compression.txt");
        println!("After compression: {} gates", circuit.gates.len());
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
        let n = if leave { n / 2 } else { n };
        let n = if do_gadgetize { (n - 1) / 2 } else { n };
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
    File::create(save)
        .and_then(|mut f| f.write_all(circuit_str.as_bytes()))
        .expect("Failed to write recent_circuit.txt");

    println!("Final circuit written to {}", save);
}

//do targeted compression
pub fn main_compression(c: &CircuitSeq, rounds: usize, n: usize, save: &str, env: &lmdb::Environment,
    shard_dbs: &[lmdb::Database],) {
    let _dbs = open_all_dbs(env);
    // Start with the input circuit
    let _bit_shuf_list: Vec<Vec<Vec<usize>>> = (3..=7)
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
            butterfly_big(&circuit, n, false, 0, env, shard_dbs);
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
    let circuit_str = circuit.repr();

    File::create(save)
        .and_then(|mut f| f.write_all(circuit_str.as_bytes()))
        .expect("Failed to write recent_circuit.txt");

    println!("Final circuit written to {}", save);
}