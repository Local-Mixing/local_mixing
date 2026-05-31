use std::{collections::HashMap, path::Path, sync::Mutex};

use dashmap::DashMap;
use lmdb::{Cursor, Environment, Transaction};
use local_mixing::replace::pairs::GatePair;
use local_mixing::replace::pairs::gate_pair_taxonomy;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::Duration;
use xxhash_rust::xxh3::xxh3_128;

use clap::{Arg, Command};
use local_mixing::{
    circuit::{
        CircuitSeq, circuit::canonicalize_polys, circuit::poly_degree, circuit::poly_to_str,
        circuit::polys_repr_blob,
    },
    replace::main_mix::open_shard_dbs,
};
use rand::seq::SliceRandom;
use rayon::prelude::*;

const LMDB_PATH: &str = "./db";

fn decode_circuits(value: &[u8]) -> Vec<CircuitSeq> {
    let mut pos = 0;
    let mut circuits = Vec::new();

    while pos < value.len() {
        if pos + 1 > value.len() {
            break;
        }

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

type WireCount = usize;
type GateCount = usize;

type CktShape = Option<(WireCount, GateCount)>;

fn make_shape(a: CktShape, b: CktShape) -> (CktShape, CktShape) {
    if a <= b { (a, b) } else { (b, a) }
}

/// Compact the wire labels of a circuit to 0..k, returning the rewired circuit
/// and the sorted list of original wires used. (Ported from eb/dev `CircuitSeq::rewire_min`.)
fn rewire_min(c: &CircuitSeq) -> (CircuitSeq, Vec<usize>) {
    let mut wires: Vec<usize> = c.gates.iter().flatten().map(|&w| w as usize).collect();
    wires.sort_unstable();
    wires.dedup();

    let mut wire_map = std::collections::HashMap::new();
    for (new_idx, &old_wire) in wires.iter().enumerate() {
        wire_map.insert(old_wire, new_idx);
    }

    let new_gates: Vec<[u16; 3]> = c
        .gates
        .iter()
        .map(|gate| {
            [
                wire_map[&(gate[0] as usize)] as u16,
                wire_map[&(gate[1] as usize)] as u16,
                wire_map[&(gate[2] as usize)] as u16,
            ]
        })
        .collect();

    (CircuitSeq { gates: new_gates }, wires)
}

/// Remove adjacent duplicate gates (a gate immediately followed by an identical
/// gate is an identity). (Ported from eb/dev `CircuitSeq::remove_adjacent_id`.)
fn remove_adjacent_id(c: &mut CircuitSeq) {
    let mut i = 0usize;
    while i < c.gates.len().saturating_sub(1) {
        if c.gates[i] == c.gates[i + 1] {
            c.gates.drain(i..=i + 1);
            i = i.saturating_sub(2);
        } else {
            i += 1;
        }
    }
}

// fn poly_complexity(env: &Environment) {
//     let shard_dbs = open_shard_dbs(&env);
//     let mut shard_indices: Vec<usize> = (0..shard_dbs.len()).collect();

//     let complexity : DashMap<&[u8], u8> = DashMap::new();
//     shard_indices.par_iter().for_each(|shard_idx| {
//         println!("Start shard {}", shard_idx);

//         let db = shard_dbs[*shard_idx];
//         let txn = env
//             .begin_ro_txn()
//             .expect("Failed to begin read-only transaction");
//         let mut cursor = txn
//             .open_ro_cursor(db)
//             .expect("Failed to open read-only cursor");

//         let mut local_complexity: HashMap<&[u8], u8> = HashMap::new();

//         for (key, value) in cursor.iter_start() {
//             let circuits = decode_circuits(value);

//             if let Some(min_circuit) = circuits
//                 .iter()
//                 .min_by_key(|circuit| circuit.gates.len())
//             {
//                 let min_length = min_circuit.gates.len() as u8;
//                 local_complexity.insert(key, min_length);
//             }
//         }

//         println!(" -- {} inserting {} complexities", shard_idx, local_complexity.len());
//         for (key, min_length) in local_complexity {
//             complexity.insert(key, min_length);
//         }

//         println!("  -- done {}", shard_idx);
//     });
// }

fn main() {
    println!("[ Minimal Identities ]");

    let env = Environment::new()
        .set_max_readers(10000)
        .set_max_dbs(256 + 40)
        .set_map_size(800 * 1024 * 1024 * 1024)
        .open(Path::new(LMDB_PATH))
        .expect("Failed to open database.");

    let shard_dbs = open_shard_dbs(&env);
    let matches = Command::new("minimal_id")
        .arg(
            Arg::new("friends")
                .long("friends")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("gen-ids")
                .long("gen-ids")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("print-ids")
                .long("print-ids")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(Arg::new("check").long("check").action(clap::ArgAction::Set))
        .group(clap::ArgGroup::new("mode").args(["friends", "gen-ids", "print-ids"]))
        .get_matches();

    if matches.get_flag("friends") {
        let shape_counter: Mutex<HashMap<(CktShape, CktShape), usize>> = Mutex::new(HashMap::new());
        friends(&env, &shard_dbs, &shape_counter);
        println!("{:?}", shape_counter.lock().unwrap());
    }

    if matches.get_flag("gen-ids") {
        generate_identities_parallel(&env, &shard_dbs);
    }

    if matches.get_flag("print-ids") {
        print_ids(&env, &shard_dbs);
    }

    if let Some(c) = matches.get_one::<String>("check") {
        let circuit = CircuitSeq::from_string(c);
        check(&env, &shard_dbs, &circuit);
    }
}

// Extracted shard loop from main; called with `--friends`
fn friends(
    env: &Environment,
    shard_dbs: &Vec<lmdb::Database>,
    shape_counter: &Mutex<HashMap<(CktShape, CktShape), usize>>,
) {
    let mut shard_indices: Vec<usize> = (0..shard_dbs.len()).collect();
    let mut rng = rand::thread_rng();

    shard_indices.par_iter().for_each(|shard_idx| {
        println!("Shard {}", shard_idx);
        let db = shard_dbs[*shard_idx];
        let txn = env
            .begin_ro_txn()
            .expect("Failed to begin read-only transaction");
        let mut cursor = txn
            .open_ro_cursor(db)
            .expect("Failed to open read-only cursor");

        let mut local_shape_counter: HashMap<(CktShape, CktShape), usize> = HashMap::new();

        for (_, value) in cursor.iter_start() {
            let circuits = decode_circuits(value);

            // To make this DISJOINT pairs, make this a Set<>
            let shapes: Vec<(WireCount, GateCount)> = circuits
                .iter()
                .map(|circuit| (circuit.max_wire() + 1, circuit.gates.len()))
                .collect();

            // Circuit counter (0,0 second)
            for sh in shapes.iter() {
                *local_shape_counter
                    .entry((Some(*sh), Some((0, 0))))
                    .or_insert(0) += 1;
            }

            // If there is only one shape present, also record an entry pairing it with None
            if shapes.len() == 1 {
                *local_shape_counter
                    .entry((Some(shapes[0]), None))
                    .or_insert(0) += 1;
            } else {
                for (i, shape_i) in shapes.iter().enumerate() {
                    for shape_j in shapes.iter().skip(i + 1) {
                        let key = make_shape(Some(*shape_i), Some(*shape_j));
                        *local_shape_counter.entry(key).or_insert(0) += 1;
                    }
                }
            }
        }

        let mut shape_counter = shape_counter.lock().unwrap();
        for (key, count) in local_shape_counter {
            *shape_counter.entry(key).or_insert(0) += count;
        }
    });
}

fn print_ids(env: &Environment, shard_dbs: &Vec<lmdb::Database>) {
    use std::collections::BTreeMap;

    let txn = env
        .begin_ro_txn()
        .expect("Failed to begin read-only transaction");
    let mut histogram: BTreeMap<usize, usize> = BTreeMap::new();

    for ctype in 0..34 {
        let name = format!("id_g{}", ctype);
        let db = unsafe { txn.open_db(Some(&name)) }.expect("open db");

        let mut cursor = txn.open_ro_cursor(db).expect("ro cursor");

        if !cursor.get(None, None, lmdb_sys::MDB_FIRST).is_ok() {
            eprintln!("Empty shard: {}", name);
            continue;
        }

        for (_, value) in cursor.iter_start() {
            let circuit = CircuitSeq::from_blob(value);

            if circuit.gates.len() == 6 {
                println!("{}", circuit.repr());
            }
            *histogram.entry(circuit.gates.len()).or_insert(0) += 1;
        }
    }

    for (len, count) in histogram {
        // println!("{}: {}", len, count);
    }
}

/// Generate identities in parallel. This follows the generate_identity_db skeleton but:
/// - only considers unique pairs (i < j)
/// - iterates cursor directly (no multi list upfront)
/// - runs shard processing in parallel
/// - checks minimality by canonicalizing and looking up hash in shard DBs
pub fn generate_identities_parallel(env: &Environment, shard_dbs: &Vec<lmdb::Database>) {
    use lmdb::Transaction;
    use local_mixing::circuit::circuit::polys_repr_blob;
    use xxhash_rust::xxh3::xxh3_128;

    // Per-type (ctype) seen blobs collected across threads, merged at the end.
    let seen: Arc<Vec<StdMutex<std::collections::HashSet<Vec<u8>>>>> = Arc::new(
        (0..34)
            .map(|_| StdMutex::new(std::collections::HashSet::new()))
            .collect(),
    );

    // Atomic counter for new identities and a running flag for the printer thread.
    let counter = Arc::new(AtomicU64::new(0));
    let running = Arc::new(AtomicBool::new(true));

    // Spawn a background printer that reports the count every second.
    let printer = {
        let c = counter.clone();
        let r = running.clone();
        thread::spawn(move || {
            while r.load(Ordering::Relaxed) {
                println!("Id: {}", c.load(Ordering::Relaxed));
                thread::sleep(Duration::from_secs(1));
            }
            // final print
            println!("Final count: {}", c.load(Ordering::Relaxed));
        })
    };

    // Parallel over shards
    let shard_indices: Vec<usize> = (0..shard_dbs.len()).collect();
    shard_indices.par_iter().for_each(|shard_idx| {
        println!("proc shard {}", shard_idx);
        let db = shard_dbs[*shard_idx];
        let rtxn = env.begin_ro_txn().expect("ro txn");
        let mut cursor = rtxn.open_ro_cursor(db).expect("cursor");

        for (_, value) in cursor.iter_start() {
            let circuits = decode_circuits(value);
            if circuits.len() < 2 {
                continue;
            }

            let minimal_ckt_len = circuits
                .iter()
                .map(|c| c.gates.len())
                .min()
                .expect("minimal ckt failed");

            for i in 0..circuits.len() {
                let a = &circuits[i];

                // ckt a MUST be minimal
                if a.gates.len() > minimal_ckt_len {
                    continue;
                }

                for j in (i + 1)..circuits.len() {
                    let b = &circuits[j];

                    // ckt b must be minimal or + 1
                    if b.gates.len() > minimal_ckt_len + 1 {
                        continue;
                    }

                    let a_rev: Vec<[u16; 3]> = a.gates.iter().rev().cloned().collect();
                    let b_rev: Vec<[u16; 3]> = b.gates.iter().rev().cloned().collect();

                    let candidates: [Vec<[u16; 3]>; 2] = [
                        // a || rev(b)
                        a.gates
                            .iter()
                            .cloned()
                            .chain(b_rev.iter().cloned())
                            .collect(),
                        // rev(a) || b
                        a_rev
                            .iter()
                            .cloned()
                            .chain(b.gates.iter().cloned())
                            .collect(),
                    ];

                    for gates in candidates {
                        let (mut identity, _) = rewire_min(&CircuitSeq { gates });

                        // Simplify until circuit is empty or size plateaus
                        loop {
                            let len_before = identity.gates.len();

                            identity.canonicalize();
                            remove_adjacent_id(&mut identity);

                            if identity.gates.is_empty() {
                                break;
                            }

                            let len_after = identity.gates.len();
                            if len_before == len_after {
                                break;
                            }
                        }

                        if identity.gates.is_empty() {
                            continue;
                        }

                        identity.canonicalize();
                        assert!(!identity.adjacent_id());

                        let (identity, _) = rewire_min(&identity);

                        // Minimality check: every half-length contiguous subcircuit must be absent.
                        let len = identity.gates.len();
                        let half_len = len / 2;
                        if half_len == 0 {
                            continue;
                        }

                        let mut non_minimal = false;
                        let wire_count = identity.max_wire() + 1;
                        for start in 0..=(len - half_len) {
                            let end = start + half_len;
                            let polys = identity.to_polynomial(wire_count, start, end);
                            let (canonical, _) = canonicalize_polys(polys, true, false);
                            let key = xxh3_128(&polys_repr_blob(&canonical)).to_le_bytes();
                            let shard = key[0] as usize;

                            if rtxn.get(shard_dbs[shard], &key).is_ok() {
                                non_minimal = true;
                                break;
                            }
                        }

                        if non_minimal {
                            continue;
                        }

                        // println!("Minimal: {}", identity.repr());

                        let ctype = GatePair::to_int(&gate_pair_taxonomy(
                            &identity.gates[0],
                            &identity.gates[1],
                        ));
                        let blob = identity.repr_blob();
                        let mut guard = seen[ctype].lock().unwrap();
                        if guard.insert(blob) {
                            counter.fetch_add(1, Ordering::Relaxed);
                        }
                        let ctype = GatePair::to_int(&gate_pair_taxonomy(
                            &identity.gates[0],
                            &identity.gates[1],
                        ));
                        let blob = identity.repr_blob();
                        let mut guard = seen[ctype].lock().unwrap();
                        if guard.insert(blob) {
                            counter.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            }
        }
    });

    // Stop printer and wait for final print, then write collected entries into id DBs (single-threaded)
    running.store(false, Ordering::Relaxed);
    let _ = printer.join();
    // Open id DBs (reuse gen_ids open_id_dbs if available) — here we'll create simple names id_g0..id_g33
    let mut wtxn = env.begin_rw_txn().expect("rw txn");

    // Drop any existing identity DB.
    for i in 0..34 {
        let name = format!("id_g{}", i);
        if let Ok(db) = unsafe { wtxn.open_db(Some(&name)) } {
            let _ = unsafe { wtxn.drop_db(db) };
        }
    }

    let id_dbs: Vec<lmdb::Database> = (0..34)
        .map(|i| {
            let name = format!("id_g{}", i);
            unsafe { wtxn.open_db(Some(&name)) }.unwrap_or_else(|_| {
                unsafe { wtxn.create_db(Some(&name), lmdb::DatabaseFlags::empty()) }
                    .expect("create id db")
            })
        })
        .collect();

    for (ctype, mutex_set) in seen.iter().enumerate() {
        let mut ctr: u64 = 0;
        let mut set = mutex_set.lock().unwrap();
        for blob in set.drain() {
            let idx = ctr;
            ctr += 1;
            wtxn.put(
                id_dbs[ctype],
                &idx.to_be_bytes(),
                &blob,
                lmdb::WriteFlags::empty(),
            )
            .expect("put identity");
        }
    }

    wtxn.commit().expect("commit ids");
}

fn check(env: &Environment, shard_dbs: &Vec<lmdb::Database>, circuit: &CircuitSeq) {
    let n: usize = circuit.max_wire() + 1;
    println!("{}", circuit.to_string(n));

    let len = circuit.gates.len();
    let half_len = len / 2;

    let rtxn = env.begin_ro_txn().expect("ro txn");

    for (i, poly) in circuit.to_polynomial(n, 0, len).iter().enumerate() {
        println!("  P{}: {}", i, poly_to_str(poly, n));
    }

    let poly = circuit.to_polynomial(4, 0, len);
    let (canonical, _) = canonicalize_polys(poly, true, false);
    let key = xxh3_128(&polys_repr_blob(&canonical)).to_le_bytes();
    let shard = key[0] as usize;

    if let Ok(v) = rtxn.get(shard_dbs[shard], &key) {
        let cs = decode_circuits(v);

        for c in cs {
            println!("{}", c.repr());
        }
    }

    // for start in 0..=(len - half_len) {
    //     for sublen in 2..=half_len {
    //         let end = start + sublen;
    //         let polys = circuit.to_polynomial(n, start, end);
    //         let (canonical, _) = canonicalize_polys(polys, true, false);
    //         let key = xxh3_128(&polys_repr_blob(&canonical)).to_le_bytes();
    //         let shard = key[0] as usize;

    //         if rtxn.get(shard_dbs[shard], &key).is_ok() {
    //             println!("{}-{} non minimal", start, end);
    //         } else {
    //             println!("{}-{} not in db", start, end)
    //         }

    //         for (i, poly) in canonical.iter().enumerate() {
    //             println!("  P{}: {}", i, poly_to_str(poly, n));
    //         }
    //     }
    // }
}
