use std::{
    collections::HashMap, marker::PhantomData, ptr, slice, sync::atomic::{AtomicU64, Ordering}, time::Instant
};

use libc::c_uint;

use itertools::Itertools;
use rand::{Rng, prelude::SliceRandom};
use lmdb::{Cursor, Database, RoCursor, RoTransaction, Transaction, Environment};

extern crate lmdb_sys;
use lmdb_sys as ffi;

use crate::{
    circuit::circuit::{CircuitSeq, Permutation},
    random::random_data::{random_circuit, shoot_random_gate},
    replace::{pairs::{CollisionType, GatePair, gate_pair_taxonomy, replace_single_pair}, 
        transpositions::{Transpositions, insert_wire_shuffles_simple, replace_disjoint_pair}},
};

// Old iterator method for cursor fails if the given key is not found
// This does not unwrap a None value in that case
pub struct Iter<'txn> {
    cursor: *mut ffi::MDB_cursor,
    op: c_uint,
    next_op: c_uint,
    finished: bool,
    _marker: PhantomData<&'txn ()>,
}

impl<'txn> Iter<'txn> {
    pub fn new(cursor: *mut ffi::MDB_cursor, op: c_uint, next_op: c_uint) -> Self {
        Self {
            cursor,
            op,
            next_op,
            finished: false,
            _marker: PhantomData,
        }
    }
}

impl<'txn> Iterator for Iter<'txn> {
    type Item = (&'txn [u8], &'txn [u8]);

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        unsafe {
            let mut key = ffi::MDB_val { mv_size: 0, mv_data: ptr::null_mut() };
            let mut data = ffi::MDB_val { mv_size: 0, mv_data: ptr::null_mut() };

            let rc = ffi::mdb_cursor_get(self.cursor, &mut key, &mut data, self.op);
            self.op = self.next_op;

            if rc == ffi::MDB_NOTFOUND {
                self.finished = true;
                return None;
            } else if rc != ffi::MDB_SUCCESS {
                panic!("LMDB error: {}", rc);
            }

            let key_slice = slice::from_raw_parts(key.mv_data as *const u8, key.mv_size);
            let data_slice = slice::from_raw_parts(data.mv_data as *const u8, data.mv_size);
            Some((key_slice, data_slice))
        }
    }
}

pub trait RoCursorExt<'txn> {
    fn iter_from_safe<K>(&mut self, key: K) -> Iter<'txn>
    where
        K: AsRef<[u8]>;
}

impl<'txn> RoCursorExt<'txn> for RoCursor<'txn> {
    fn iter_from_safe<K>(&mut self, key: K) -> Iter<'txn>
    where
        K: AsRef<[u8]>,
    {
        let rc = unsafe {
            let mut key_val = lmdb_sys::MDB_val {
                mv_size: key.as_ref().len(),
                mv_data: key.as_ref().as_ptr() as *mut _,
            };
            lmdb_sys::mdb_cursor_get(self.cursor(), &mut key_val, std::ptr::null_mut(), lmdb_sys::MDB_SET_RANGE)
        };

        if rc == lmdb_sys::MDB_NOTFOUND {
            Iter {
                cursor: self.cursor(),
                op: lmdb_sys::MDB_GET_CURRENT,
                next_op: lmdb_sys::MDB_NEXT,
                finished: true,
                _marker: std::marker::PhantomData,
            }
        } else if rc != lmdb_sys::MDB_SUCCESS {
            panic!("LMDB error: {}", rc);
        } else {
            Iter::new(self.cursor(), lmdb_sys::MDB_GET_CURRENT, lmdb_sys::MDB_NEXT)
        }
    }
}

pub fn random_perm_lmdb(
    txn: &RoTransaction,
    db: Database,
    prefix: &[u8],
) -> Option<Vec<u8>> {
    let mut cursor = txn.open_ro_cursor(db).ok()?;
    let mut rng = rand::rng();
    let mut chosen: Option<Vec<u8>> = None;
    let mut count = 0;

    for (key, _) in cursor.iter_from_safe(prefix) {
        if !key.starts_with(prefix) { break; }
        count += 1;
        if rng.random_range(0..count) == 0 {
            chosen = Some(key[prefix.len()..].to_vec());
        }
    }
    chosen
}

// Select a random permutation 
fn random_perm_from_perm_table(
    txn: &RoTransaction,
    db: Database,
) -> Option<(Vec<u8>, Vec<u8>)> {
    let mut cursor = txn.open_ro_cursor(db).ok()?;
    let mut entries = Vec::new();

    for (k, v) in cursor.iter() {
        entries.push((k.to_vec(), v.to_vec()));
    }

    if entries.is_empty() {
        return None;
    }

    let idx = rand::rng().random_range(0..entries.len());
    Some(entries.swap_remove(idx))
}

// Returns a nontrivial identity circuit built from two "friend" circuits
// This is legacy now that we have ids_nNgK tables
pub fn random_canonical_id(
    env: &lmdb::Environment,
    n: usize,
) -> Result<CircuitSeq, Box<dyn std::error::Error>> {
    let mut rng = rand::rng();

    loop {
        let perm_db_name = format!("perm_tables_n{}", n);
        let perm_db = env.open_db(Some(&perm_db_name))
            .unwrap_or_else(|e| panic!("LMDB DB '{}' not found or failed to open: {:?}", perm_db_name, e));
        let (perm_blob, ms_blob) = {
            let txn = env.begin_ro_txn()
                .unwrap_or_else(|e| panic!("Failed to begin RO txn on '{}': {:?}", perm_db_name, e));
            match random_perm_from_perm_table(&txn, perm_db) {
                Some(x) => x,
                None => panic!("perm_tables_n{} is empty or malformed", n),
            }
        };

        let mut ms: Vec<u8> = bincode::deserialize(&ms_blob)
            .unwrap_or_else(|_| panic!("Failed to deserialize ms_blob for n={}", n));

        ms.retain(|&x| x != 0);

        if ms.len() < 2 {
            panic!("ms.len() < 2 for perm in perm_tables_n{}", n);
        }

        // println!("perm: {:?}", Permutation::from_blob(&perm_blob));
        // println!("ms: {:?}", ms);

        let i = rng.random_range(0..ms.len());
        let mut j = rng.random_range(0..ms.len());
        while j == i { j = rng.random_range(0..ms.len()); }
        let m1 = ms[i];
        let m2 = ms[j];

        let db1_name = format!("n{}m{}", n, m1);
        let db2_name = format!("n{}m{}", n, m2);
        
        // println!("Searching for perm_len {} in {}", perm_blob.len().trailing_zeros(), db1_name);

        let circuit1_blob = {
            let db1 = env.open_db(Some(&db1_name))
                .unwrap_or_else(|e| panic!("LMDB DB1 '{}' failed to open: {:?}", db1_name, e));
            let txn = env.begin_ro_txn()
                .unwrap_or_else(|e| panic!("Failed to begin RO txn on '{}': {:?}", db1_name, e));
            random_perm_lmdb(&txn, db1, &perm_blob)
                .unwrap_or_else(|| panic!("perm not found in {}", db1_name))
        };
        let mut ca = CircuitSeq::from_blob(&circuit1_blob);

        let circuit2_blob = {
            let db2 = env.open_db(Some(&db2_name))
                .unwrap_or_else(|e| panic!("LMDB DB2 '{}' failed to open: {:?}", db2_name, e));
            let txn = env.begin_ro_txn()
                .unwrap_or_else(|e| panic!("Failed to begin RO txn on '{}': {:?}", db2_name, e));
            random_perm_lmdb(&txn, db2, &perm_blob)
                .unwrap_or_else(|| panic!("perm not found in {}", db2_name))
        };
        let mut cb = CircuitSeq::from_blob(&circuit2_blob);

        cb.gates.reverse();
        ca.gates.extend(cb.gates);

        let perms: Vec<Vec<usize>> = (0..n).permutations(n).collect();
        let shuf = perms
            .iter()
            .skip(1)
            .nth(rng.random_range(0..perms.len() - 1))
            .expect("Failed to select a random bit shuffle")
            .clone();

        let bit_shuf = Permutation { data: shuf };
        ca.rewire(&bit_shuf, n);
        return Ok(ca);
    }
}

// Timing variables for benchmarking
static GET_ID_TOTAL_TIME: AtomicU64 = AtomicU64::new(0);
// static DB_NAME_TIME: AtomicU64 = AtomicU64::new(0);
// static DB_LOOKUP_TIME: AtomicU64 = AtomicU64::new(0);
// static TXN_BEGIN_TIME: AtomicU64 = AtomicU64::new(0);
// static SERIALIZE_KEY_TIME: AtomicU64 = AtomicU64::new(0);
// static LMDB_GET_TIME: AtomicU64 = AtomicU64::new(0);
// static DESERIALIZE_LIST_TIME: AtomicU64 = AtomicU64::new(0);
// static RNG_CHOOSE_TIME: AtomicU64 = AtomicU64::new(0);

// New method to get a random identity
pub fn get_random_identity(
    n: usize,
    gate_pair: GatePair,
    env: &lmdb::Environment,
    dbs: &HashMap<String, Database>,
    tower: bool,
) -> Result<CircuitSeq, Box<dyn std::error::Error>> {
    let total_start = Instant::now();

    let g = GatePair::to_int(&gate_pair);
    let db_name = if n == 16 && tower {
        format!("ids_n{}g{}{}", n, g, "tower")
    } else if n == 16 && !tower {
        format!("ids_n{}g{}{}", n, g, "single")
    } else {
        format!("ids_n{}g{}", n, g)
    };

    let db = dbs.get(&db_name).unwrap_or_else(|| {
        panic!("Failed to get DB with name: {}", db_name);
    });


    let txn = env.begin_ro_txn()?;

    let count: u64 = {
        let mut cursor = txn.open_ro_cursor(*db)?;
        // MDB_LAST positions cursor at last entry; key is our sequential counter
        match cursor.get(None, None, ffi::MDB_LAST) {
            Ok((Some(k), _)) => {
                let arr: [u8; 8] = k.try_into()
                    .unwrap_or_else(|_| panic!("Non-u64 key in {}", db_name));
                u64::from_be_bytes(arr) + 1
            }
            _ => panic!("Empty DB: {}", db_name),
        }
    };

    let mut rng = rand::rng();
    let random_index: u64 = rng.random_range(0..count);
    let key = random_index.to_be_bytes();
    let value_bytes = txn.get(*db, &key)
        .unwrap_or_else(|_| panic!("Missing key {} in {}", random_index, db_name));
    let out = CircuitSeq::from_blob(value_bytes);

    GET_ID_TOTAL_TIME.fetch_add(
        total_start.elapsed().as_nanos() as u64,
        Ordering::Relaxed,
    );

    Ok(out)
}

// Sample a random identity from the id_g{i} databases generated by gen_id_db.
pub fn get_random_id_db_identity(
    gate_pair: GatePair,
    env: &lmdb::Environment,
    dbs: &[lmdb::Database],
) -> Option<CircuitSeq> {
    let ctype = GatePair::to_int(&gate_pair);
    let db = dbs[ctype];
    let txn = env.begin_ro_txn().ok()?;

    let count: u64 = {
        let mut cursor = txn.open_ro_cursor(db).ok()?;
        match cursor.get(None, None, ffi::MDB_LAST) {
            Ok((Some(k), _)) => {
                let arr: [u8; 8] = k.try_into().ok()?;
                u64::from_be_bytes(arr) + 1
            }
            _ => return None,
        }
    };

    let mut rng = rand::rng();
    let idx: u64 = rng.random_range(0..count);
    let blob = txn.get(db, &idx.to_be_bytes()).ok()?;
    Some(CircuitSeq::from_blob(blob))
}

// Generate identities via shuffling
pub fn get_random_shuffled_identity (
    n: usize, 
    env: &lmdb::Environment,
    dbs: &HashMap<String, Database>,
    _bit_shuf_list: &Vec<Vec<Vec<usize>>>,
    tower: bool,
) -> CircuitSeq {
    let dummy_id = CircuitSeq { gates: Vec::new() };
    loop {
        let mut id = get_random_identity(
            6,
            GatePair::from_int(rand::rng().random_range(0..34)),
            env,
            dbs,
            tower
        ).unwrap();

        insert_wire_shuffles_simple(&mut id, n, env, dbs);
        if dummy_id.probably_equal(&id, n, 1000).is_ok() {
            return id
        }
    }
}


// Generate identities on more wires
// Our original tables only support up to 7 wires
// Our LMDB currently stores some 16 and 128 wire identities
pub fn get_random_wide_identity(
    n: usize, 
    env: &lmdb::Environment,
    dbs: &HashMap<String, Database>,
    _bit_shuf_list: &Vec<Vec<Vec<usize>>>,
    tower: bool,
) -> CircuitSeq {
    let mut id = CircuitSeq { gates: Vec::new() };
    let mut uw = id.used_wires();
    let mut nwires = uw.len();
    let mut rng = rand::rng();
    let mut len = 0;
    while nwires < n || len < 150 {
        shoot_random_gate(&mut id, 100_000);
        let gp = GatePair::from_int(rng.random_range(0..34));
        let mut i = match get_random_identity(6, gp, env, dbs, false) {
            Ok(i) => {
                i
            }
            Err(_) => {
                continue;
            }
        };
        if id.clone().gates.is_empty() {
            id = i;
        } else {
            let mut wires: HashMap<u8, Vec<usize>> = HashMap::new();
            for (idx, gates) in id.clone().gates.into_iter().enumerate() {
                for pins in gates {
                    wires.entry(pins)
                    .or_insert_with(Vec::new)
                    .push(idx);
                }
            }
            let min_vals: &Vec<usize> = wires
                .iter()
                .min_by_key(|(_, v)| v.len())
                .map(|(_, v)| v)
                .unwrap();
            let mut min_keys: Vec<u8> = wires.keys().cloned().collect();
            min_keys.sort_by_key(|k| wires.get(k).map(|v| v.len()).unwrap_or(0));
            let mut min = min_vals[0];
            if tower {
                min = id.gates.len()/2;
            }
            let mut used_wires = vec![id.gates[min][0], id.gates[min][1], id.gates[min][2]];
            let mut unused_wires: Vec<u8> = (0..=(n-1) as u8)
                .filter(|w| !used_wires.contains(w) && !uw.contains(w))
                .collect();
            let mut count = 3;
            let mut j = 1;
            while count < 6 {
                if !unused_wires.is_empty() {
                    let random = unused_wires.pop().unwrap();
                    used_wires.push(random);
                    count += 1;
                } else {
                    let random = min_keys[j];
                    if used_wires.contains(&random) {
                        j += 1;
                        continue;
                    }
                    used_wires.push(random);
                    count += 1;
                    j += 1;
                }
            }
            let rewired_g = CircuitSeq::rewire_subcircuit(&id, &vec![min], &used_wires);
            i.rewire_first_gate(rewired_g.gates[0], 6);
            i = CircuitSeq::unrewire_subcircuit(&i, &used_wires);
            i.gates.remove(0);
            id.gates.splice(min..=min, i.gates);
        }
        uw = id.used_wires();
        nwires = uw.len();
        len = id.gates.len();
    }

    let mut shuf: Vec<usize> = (0..=(n-1)).collect();
    shuf.shuffle(&mut rng);

    let bit_shuf = Permutation { data: shuf };
    id.rewire(&bit_shuf, n);
    id
}

// Unsupported method of generating more random looking identities on more wires
pub fn get_random_wide_identity_via_pairs(
    n: usize, 
    env: &lmdb::Environment,
    dbs: &HashMap<String, Database>,
    _bit_shuf_list: &Vec<Vec<Vec<usize>>>,
) -> CircuitSeq {
    let mut id = CircuitSeq { gates: Vec::new() };
    let mut uw = id.used_wires();
    let mut nwires = uw.len();
    let mut rng = rand::rng();
    let mut len = 0;
    while nwires < 16 || len < 160 {
        shoot_random_gate(&mut id, 100_000);
        let gp = GatePair::from_int(rng.random_range(0..34));
        let mut i = match get_random_identity(6, gp, env, dbs, false) {
            Ok(i) => {
                i
            }
            Err(_) => {
                continue;
            }
        };
        if id.clone().gates.is_empty() {
            id = i;
        } else {
            let mut wires: HashMap<u8, Vec<usize>> = HashMap::new();
            for (idx, gates) in id.clone().gates.into_iter().enumerate() {
                for pins in gates {
                    wires.entry(pins)
                    .or_insert_with(Vec::new)
                    .push(idx);
                }
            }
            let min_vals: &Vec<usize> = wires
                .iter()
                .min_by_key(|(_, v)| v.len())
                .map(|(_, v)| v)
                .unwrap();
            let mut min_keys: Vec<u8> = wires.keys().cloned().collect();
            min_keys.sort_by_key(|k| wires.get(k).map(|v| v.len()).unwrap_or(0));
            let mut min = min_vals[0];
            if min == id.gates.len() - 1 {
                min -= 1;
            }
            let tax = gate_pair_taxonomy(&id.gates[min], &id.gates[min+1]);
            println!("{:?}", tax);
            println!("{:?}", &id.gates[min]);
            println!("{:?}", &id.gates[min+1]);
            i = CircuitSeq {gates: Vec::new()};
            let mut id_gen = false;
            while !id_gen {
                i = match get_random_identity(6, tax, env, dbs, false) {
                    Ok(i) => {
                        id_gen = true;
                        i
                    },
                    Err(_) => {
                        continue;
                    }
                };
            }
            let new_circuit = i.gates[2..].to_vec();
            let replacement_circ = CircuitSeq { gates: new_circuit };
            let mut used_wires: Vec<u8> = vec![
                (n + 1) as u8;
                std::cmp::max(
                    replacement_circ.max_wire(),
                    CircuitSeq {
                        gates: vec![i.gates[0], i.gates[1]],
                    }
                    .max_wire(),
                ) + 1
            ];
            
            used_wires[i.gates[0][0] as usize] = id.gates[min][0];
            used_wires[i.gates[0][1] as usize] = id.gates[min][1];
            used_wires[i.gates[0][2] as usize] = id.gates[min][2];

            let mut k = 0;
            for collision in &[tax.a, tax.c1, tax.c2] {
                if *collision == CollisionType::OnNew {
                    used_wires[i.gates[1][k] as usize] = id.gates[min+1][k];
                }
                k += 1;
            }

            let mut unused_wires: Vec<u8> = (0..=(n-1) as u8)
                .filter(|w| !used_wires.contains(w) && !uw.contains(w))
                .collect();
            let mut count = 3;
            let mut j = 1;
            while count < 6 {
                if !unused_wires.is_empty() {
                    let random = unused_wires.pop().unwrap();
                    used_wires.push(random);
                    count += 1;
                } else {
                    let random = min_keys[j];
                    if used_wires.contains(&random) {
                        j += 1;
                        continue;
                    }
                    used_wires.push(random);
                    count += 1;
                    j += 1;
                }
            }
            i.gates = CircuitSeq::unrewire_subcircuit(&replacement_circ, &used_wires)
            .gates
            .into_iter()
            .rev()
            .collect();
            id.gates.splice(min..=min+1, i.gates);
        }
        uw = id.used_wires();
        nwires = uw.len();
        len = id.gates.len();
    }
    
    let mut shuf: Vec<usize> = (0..=(n-1)).collect();
    shuf.shuffle(&mut rng);

    let bit_shuf = Permutation { data: shuf };
    id.rewire(&bit_shuf, n);
    id
}

// To just get a completely random circuit and reverse for identity, rather than using canonical ones from our rainbow table
pub fn random_id(n: usize, m: usize) -> (CircuitSeq, CircuitSeq) {
    let circuit = random_circuit(n, m);

    // Preallocate reversed gates so we don't need to run through circuit twice
    let mut rev_gates = Vec::with_capacity(circuit.gates.len());
    for g in circuit.gates.iter().rev() {
        rev_gates.push(*g); // copy [u8;3]
    }

    let rev = CircuitSeq { gates: rev_gates };
    (circuit, rev)
}

// Creates an identity with the first part limited to 16..=28 wires (exclude wires 29, 30, 31), the middle part spanning all 0..=31, and the last part spanning 0..=12 wires (exclude wires 13, 14, 15) 
// returns the identity, the number of transpositions of the first part, and the number of transpositions of the second part

pub fn create_ri_identities_32() -> (Transpositions, Transpositions, Transpositions, usize, usize) {
    let mut transpositions: Transpositions = Transpositions{ transpositions: Vec::new() };
    let mut first_negation_mask: Vec<u8> = vec![0u8; 32]; 
    let mut first = Transpositions::gen_random_simple(13, 50, &mut first_negation_mask);
    let mut second_negation_mask: Vec<u8> = vec![0u8; 32]; 
    let second = Transpositions::gen_random_simple(13, 50, &mut second_negation_mask);
    for i in 0..16 {
        let temp = first_negation_mask[i];
        first_negation_mask[i] = first_negation_mask[i + 16];
        first_negation_mask[i + 16] = temp;
    }

    for i in 0..50 {
        first.transpositions[i].0 += 16;
        first.transpositions[i].1 += 16;
        transpositions.transpositions.push(first.transpositions[i]);
        transpositions.transpositions.push(second.transpositions[i]);
    }

    for i in (0..50).rev() {
        let idx = 2 * i;

        let a = transpositions.transpositions[idx];
        let b = transpositions.transpositions[idx + 1];

        transpositions.transpositions.splice(idx..idx+2, replace_disjoint_pair(a, b));
    }

    (first, transpositions, second, 50, 50)
}

pub fn zip_escalators(
    left: &Vec<Vec<[u8;3]>>, 
    right: &Vec<Vec<[u8;3]>>, 
    gate: &[u8;3], 
    steps: &Vec<Vec<usize>>,
    n: usize,
    _tran: &mut Transpositions,
    _negation_mask: &mut Vec<u8>,
    env: &Environment,
    _bit_shuf_list: &Vec<Vec<Vec<usize>>>,
    dbs:&HashMap<String, Database>,
    tower: bool,
    id_len: usize
) -> CircuitSeq {
    use std::fs::File;
    use std::io::Write;
    let mut rng = rand::rng();
    let mut gate_step = 0;
    while gate_step < steps.len() {
        if steps[gate_step].contains(&(gate[0] as usize)) {
            break;
        }
        gate_step += 1;
    }
    let mut right = right.clone();
    right[gate_step].insert(0, *gate);
    let mut left_file = File::create("left.txt").unwrap();
    let mut right_file = File::create("right.txt").unwrap();
    writeln!(left_file, "{:?}", left).unwrap();
    writeln!(right_file, "{:?}", right).unwrap();

    let mut combined: Vec<[u8;3]> = Vec::new();
    let min = std::cmp::min(left.len(), right.len());
    for i in 0..min {
        let l = &left[i];
        let r = &right[i];

        let mut j = 0;

        // interleave while both have wires
        while j < l.len() && j < r.len() {
            let left_gate = l[j];
            let right_gate = r[j];
            // combined.push(left_gate);
            // combined.push(right_gate);
            let _quasi: bool = rng.random_bool(0.5);
            let quasi = true;
            if quasi {
                let (paired_up, _) = replace_single_pair(
                                                        &left_gate,
                                                        &right_gate,
                                                        n,
                                                        env,
                                                        _bit_shuf_list,
                                                        dbs,
                                                        tower,
                                                        id_len
                                                    );
                
                combined.extend_from_slice(&paired_up);
            } else {

            }
            j += 1;
        }

        // append leftovers
        while j < l.len() {
            combined.push(l[j]);
            j += 1;
        }

        while j < r.len() {
            combined.push(r[j]);
            j += 1;
        }
    }

    let c = CircuitSeq { gates: combined };

    let mut test_circuit = CircuitSeq { gates: Vec::new()};
    test_circuit.gates.extend(left.into_iter().flatten());
    test_circuit.gates.extend(right.into_iter().flatten());
    if test_circuit.probably_equal(&c, n, 1000).is_err() {
        panic!("Zipping failed")
    }

    c
}
// Only supports 32 wires for now
pub fn insert_ri_identities(c: &mut CircuitSeq, env: &Environment, dbs: &HashMap<String, Database>) {
    let mut t_rewired: Vec<(Transpositions, Permutation)> = Vec::new();
    let mut rng = rand::rng();
    let len = c.gates.len();
    let mut used_wires:[u8;3] = [c.gates[0][0], c.gates[0][1], c.gates[0][2]];
    // Create and rewire all the RI identities
    for i in 1..len {
        let (first, middle, second, _, _) = create_ri_identities_32();
        let mut wire_shuffle1 = Permutation { data: (0..32).collect() };
        for idx in 16..32 {
            wire_shuffle1.data[idx] = 33;
        }
        let mut excluded = vec![29,30,31];
        excluded.shuffle(&mut rng);

        let mut used_targets = Vec::new();
        let mut count = 0;

        for &val in &used_wires {
            if val >= 16 {
                wire_shuffle1.data[excluded[count]] = val as usize;
                used_targets.push(val as usize);
                count += 1;
            }
        }

        // remaining wires only in upper half
        let mut remaining: Vec<usize> = (16..32)
            .filter(|w| !used_targets.contains(w))
            .collect();

        remaining.shuffle(&mut rng);

        let mut idx = 0;
        for i in 16..32 {
            if wire_shuffle1.data[i] == 33 {
                wire_shuffle1.data[i] = remaining[idx];
                idx += 1;
            }
        }
        t_rewired.push((first.clone(), wire_shuffle1.clone()));

        used_wires = [c.gates[i][0], c.gates[i][1], c.gates[i][2]];
        let mut wire_shuffle2 = Permutation { data: (0..32).collect() };
        for idx in 0..16 {
            wire_shuffle2.data[idx] = 33;
        }
        let mut excluded = vec![13,14,15];
        excluded.shuffle(&mut rng);

        let mut used_targets = Vec::new();
        let mut count = 0;

        for &val in &used_wires {
            if val < 16 {
                wire_shuffle2.data[excluded[count]] = val as usize;
                used_targets.push(val as usize);
                count += 1;
            }
        }
        // remaining wires only in lower half
        let mut remaining: Vec<usize> = (0..16)
            .filter(|w| !used_targets.contains(w))
            .collect();

        remaining.shuffle(&mut rng);

        let mut idx = 0;
        for i in 0..16 {
            if wire_shuffle2.data[i] == 33 {
                wire_shuffle2.data[i] = remaining[idx];
                idx += 1;
            }
        }
        let mut wire_shufflem = Permutation { data: Vec::with_capacity(32)};
        wire_shufflem.data.extend_from_slice(&wire_shuffle2.data[..16]);
        wire_shufflem.data.extend_from_slice(&wire_shuffle1.data[16..]);
        t_rewired.push((middle, wire_shufflem));
        t_rewired.push((second, wire_shuffle2));
    }
    
    // Combine the RI identities and seam them together
    let num_identities = t_rewired.len()/3;
    for i in (0..num_identities-1).rev() {
        let idx = 2 + 3*i;
        let (t1, p1) = t_rewired[idx].clone();
        let (t2, p2) = t_rewired[idx + 1].clone();
        let mut combined = Transpositions { transpositions: Vec::new() };
        for i in 0..50 {
            combined.transpositions.push(t1.transpositions[i]);
            combined.transpositions.push(t2.transpositions[i]);
        }
        for i in (0..50).rev() {
            let idx = 2 * i;

            let a = combined.transpositions[idx];
            let b = combined.transpositions[idx + 1];
            let ab = replace_disjoint_pair(a, b);
            combined.transpositions.splice(idx..idx+2, ab);
        }
        let mut new_perm = Vec::with_capacity(32);
        new_perm.extend_from_slice(&p1.data[..16]);
        new_perm.extend_from_slice(&p2.data[16..]);
        let new_perm = Permutation{ data: new_perm };
        let combined = (combined, new_perm);
        t_rewired.splice(idx..idx+2, [combined]);
    }

    *c = Transpositions::restricted_to_circuit_rewired_and_insert(t_rewired, c.clone(), 32, &env, &dbs, (16, 28), (0, 12));
}

// Takes an n and a list of wires that each step will skip over, in order
// Returns the first escalator, the middle, and the last escalator
// Middle needs to be reversed when turned into a circuit
// Returns number of transpositions in first and second
// Structurally, ` first -> middle -> second ` is equal to ` second -> first -> middle ``
// Currently buggy. Likely to_perm and from_perm code
pub fn create_escalator_identities(
    n: usize, 
    first_steps: &Vec<Vec<usize>>, 
    second_steps: &Vec<Vec<usize>>,
    env: &Environment,
    dbs: &HashMap<String, Database>,
) -> (Vec<Vec<[u8;3]>>, CircuitSeq, Vec<Vec<[u8;3]>>, usize) {
    let mut allowed_wires: Vec<usize> = Vec::new();
    let all_wires: Vec<usize> = (0..n).collect();
    let mut restricted_wires: Vec<usize> = Vec::new();
    let mut first_circuit = CircuitSeq { gates: Vec::new() };
    let mut second_circuit = CircuitSeq { gates: Vec::new() };
    let mut first_step_gates: Vec<Vec<[u8;3]>> = Vec::new();
    let mut second_step_gates: Vec<Vec<[u8;3]>> = Vec::new();
    let mut first = Transpositions { transpositions: Vec::new() };
    let mut middle = Transpositions { transpositions: Vec::new() };
    let mut second = Transpositions { transpositions: Vec::new() };
    let mut first_middle_circuit = CircuitSeq { gates: Vec::new() };
    let mut second_middle_circuit = CircuitSeq { gates: Vec::new() };
    let mut middle_circuit;
    let m = 30;
    let mut negation_mask: Vec<u8> = vec![0u8; n];
    loop {
        // Build second from the `top` to bottom
        let add_transps = Transpositions::gen_random_simple(n, m, &mut negation_mask);
        second.transpositions.extend_from_slice(&add_transps.transpositions);
        let sc = add_transps.to_circuit(n, env, dbs);
        second_middle_circuit.gates.extend_from_slice(&sc.gates);
        for step in second_steps.iter().take(second_steps.len() - 1) {
            for wire in step {
                restricted_wires.push(*wire);
            }
            let add_transps = Transpositions::gen_random_simple_restricted(n, m, &mut negation_mask, &restricted_wires);
            second.transpositions.extend_from_slice(&add_transps.transpositions);
            let sc = add_transps.restricted_to_circuit(n, env, dbs, &restricted_wires);
            second_circuit.gates.extend_from_slice(&sc.gates);
            second_step_gates.push(sc.gates);
        }

        allowed_wires.clear();
        // building first
        for step in first_steps.iter().take(first_steps.len() - 1) {
            for wire in step {
                allowed_wires.push(*wire);
            }
            let restricted: Vec<usize> = all_wires.iter()
                .filter(|w| !allowed_wires.contains(w))
                .cloned()
                .collect();
            let add_transpf = Transpositions::gen_random_simple_restricted(n, m, &mut negation_mask, &restricted);
            first.transpositions.extend_from_slice(&add_transpf.transpositions);
            let fc = add_transpf.restricted_to_circuit(n, env, dbs, &restricted);
            first_circuit.gates.extend_from_slice(&fc.gates);
            first_step_gates.push(fc.gates);
        }
        let add_transpf = Transpositions::gen_random_simple(n, m, &mut negation_mask);
        first.transpositions.extend_from_slice(&add_transpf.transpositions);
        let fc = add_transpf.to_circuit(n, env, dbs);
        first_middle_circuit.gates.extend_from_slice(&fc.gates);

        // Now build middle
        middle.transpositions.extend_from_slice(&second.transpositions);
        middle.transpositions.extend_from_slice(&first.transpositions);
        
        let perm = middle.to_perm(n);
        let mut new_middle = Transpositions::from_perm(&perm);
        
        // Use negation mask to update middle
        let mut wire_transpositions: HashMap<u8, (usize, usize)> = HashMap::new();

        for (i, (a, b, _)) in new_middle.transpositions.iter().enumerate() {
            wire_transpositions.insert(*a, (i, 0));
            wire_transpositions.insert(*b, (i, 1));
        }

        const TRANSITION: [[u8; 4]; 2] = [
            // pos = 0
            [1, 0, 3, 2],
            // pos = 1
            [2, 3, 0, 1],
        ];

        for (i, val) in negation_mask.into_iter().enumerate() {
            if val == 1 {
                if let Some(swaps) = wire_transpositions.get(&(i as u8)) {
                    let &(swap_idx, pos) = swaps;
                    let curr_neg_type = new_middle.transpositions[swap_idx].2;
                    if pos > 1 || curr_neg_type > 3 {
                        panic!("Invalid pos or curr_neg_type");
                    }
                    new_middle.transpositions[swap_idx].2 = TRANSITION[pos][curr_neg_type as usize];
                    
                }
            }
        }

        middle = new_middle;

        middle_circuit = middle.to_circuit(n, env, dbs);
        middle_circuit.gates.reverse();
        middle_circuit = first_middle_circuit.concat(&middle_circuit).concat(&second_middle_circuit);
        let circuit = first_circuit.concat(&middle_circuit).concat(&second_circuit);

        let id = CircuitSeq { gates: Vec::new() };
        if circuit.probably_equal(&id, n, 10000).is_err() {
            // Reset all variables
            restricted_wires.clear();
            first_circuit.gates.clear();
            second_circuit.gates.clear();
            first.transpositions.clear();
            middle.transpositions.clear();
            second.transpositions.clear();
            first_middle_circuit.gates.clear();
            second_middle_circuit.gates.clear();
            first_step_gates.clear();
            second_step_gates.clear();
            middle_circuit.gates.clear();
            negation_mask = vec![0u8; n];
        } else {
            break
        }
    }
    (first_step_gates, middle_circuit, second_step_gates, m)
}

pub fn create_escalator_identities_tracked(
    n: usize, 
    first_steps: &Vec<Vec<usize>>, 
    second_steps: &Vec<Vec<usize>>,
    env: &Environment,
    dbs: &HashMap<String, Database>,
) -> (Vec<([u8;3], u8)>, Transpositions, Transpositions, Transpositions, usize) {
    let mut first = Transpositions { transpositions: Vec::new() };
    let mut middle = Transpositions { transpositions: Vec::new() };
    let mut second = Transpositions { transpositions: Vec::new() };
    let mut gates_track: Vec<([u8;3], u8)> = Vec::new();
    let m = 30;
    loop { 
        let mut allowed_wires: Vec<usize> = Vec::new();
        let all_wires: Vec<usize> = (0..n).collect();
        let mut restricted_wires: Vec<usize> = Vec::new();
        let mut first_circuit = CircuitSeq { gates: Vec::new() };
        let mut second_circuit = CircuitSeq { gates: Vec::new() };
        let mut negation_mask: Vec<u8> = vec![0u8; n];

        // Build second from the `top` to bottom
        let add_transps = Transpositions::gen_random_simple(n, m, &mut negation_mask);
        second.transpositions.extend_from_slice(&add_transps.transpositions);
        let sc = add_transps.to_circuit(n, env, dbs);
        second_circuit.gates.extend_from_slice(&sc.gates);
        for step in second_steps.iter().take(second_steps.len() - 1) {
            for wire in step {
                restricted_wires.push(*wire);
            }
            let add_transps = Transpositions::gen_random_simple_restricted(n, m, &mut negation_mask, &restricted_wires);
            second.transpositions.extend_from_slice(&add_transps.transpositions);
            let sc = add_transps.restricted_to_circuit(n, env, dbs, &restricted_wires);
            second_circuit.gates.extend_from_slice(&sc.gates);
        }

        allowed_wires.clear();
        // building first
        for step in first_steps.iter().take(first_steps.len() - 1) {
            for wire in step {
                allowed_wires.push(*wire);
            }
            let restricted: Vec<usize> = all_wires.iter()
                .filter(|w| !allowed_wires.contains(w))
                .cloned()
                .collect();
            let add_transpf = Transpositions::gen_random_simple_restricted(n, m, &mut negation_mask, &restricted);
            first.transpositions.extend_from_slice(&add_transpf.transpositions);
            let fc = add_transpf.restricted_to_circuit(n, env, dbs, &restricted);
            first_circuit.gates.extend_from_slice(&fc.gates);
        }
        let add_transpf = Transpositions::gen_random_simple(n, m, &mut negation_mask);
        first.transpositions.extend_from_slice(&add_transpf.transpositions);
        let fc = add_transpf.to_circuit(n, env, dbs);
        first_circuit.gates.extend_from_slice(&fc.gates);

        // Now build middle
        middle.transpositions.extend_from_slice(&second.transpositions);
        middle.transpositions.extend_from_slice(&first.transpositions);
        
        let perm = middle.to_perm(n);
        let mut new_middle = Transpositions::from_perm(&perm);
        
        // Use negation mask to update middle
        let mut wire_transpositions: HashMap<u8, (usize, usize)> = HashMap::new();

        for (i, (a, b, _)) in new_middle.transpositions.iter().enumerate() {
            wire_transpositions.insert(*a, (i, 0));
            wire_transpositions.insert(*b, (i, 1));
        }

        const TRANSITION: [[u8; 4]; 2] = [
            // pos = 0
            [1, 0, 3, 2],
            // pos = 1
            [2, 3, 0, 1],
        ];

        for (i, val) in negation_mask.into_iter().enumerate() {
            if val == 1 {
                if let Some(swaps) = wire_transpositions.get(&(i as u8)) {
                    let &(swap_idx, pos) = swaps;
                    let curr_neg_type = new_middle.transpositions[swap_idx].2;
                    if pos > 1 || curr_neg_type > 3 {
                        panic!("Invalid pos or curr_neg_type");
                    }
                    new_middle.transpositions[swap_idx].2 = TRANSITION[pos][curr_neg_type as usize];
                    
                }
            }
        }

        middle = new_middle;

        let mut middle_circuit = middle.to_circuit(n, env, dbs);
        middle_circuit.gates.reverse();
        let circuit = first_circuit.concat(&middle_circuit).concat(&second_circuit);
        for i in 0..first_circuit.gates.len() {
            gates_track.push((first_circuit.gates[i], 1));
        }
        for i in 0..middle_circuit.gates.len() {
            gates_track.push((middle_circuit.gates[i], 0));
        }
        for i in 0..second_circuit.gates.len() {
            gates_track.push((second_circuit.gates[i], 2));
        }
        let id = CircuitSeq {gates: Vec::new() };
        if circuit.probably_equal(&id, n, 1000).is_ok() {
            break;
        } else {
            first.transpositions.clear();
            middle.transpositions.clear();
            second.transpositions.clear();
            gates_track.clear();
        }
    }
    (gates_track, first, middle, second, m)
}

mod test {
    #[test]
    fn test_escalator() {
        use crate::replace::identities::create_escalator_identities;
        use crate::replace::main_mix::open_all_dbs;
        use std::io::Write;
        use lmdb::Environment;
        use std::fs::File;
        use std::path::Path;
        use crate::CircuitSeq;
        use rand::seq::SliceRandom;
        let env = Environment::new()
            .set_max_dbs(262)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new("./db"))
            .expect("failed to open lmdb");

        let dbs = open_all_dbs(&env);

        let mut file = File::create("test_id.txt").expect("Failed to create file");
        let mut simple1: Vec<Vec<usize>> = vec![
            vec![0, 1, 2],
            vec![3, 4, 5],
            vec![6, 7, 8],
            vec![9, 10, 11],
            vec![12, 13, 14],
            vec![15, 16],
            vec![17, 18, 19],
            vec![20, 21, 22],
            vec![23, 24, 25],
            vec![26, 27, 28],
            vec![29, 30, 31],
        ];
        let mut simple2: Vec<Vec<usize>> = vec![
            vec![0, 1, 2],
            vec![3, 4, 5],
            vec![6, 7, 8],
            vec![9, 10, 11],
            vec![12, 13, 14],
            vec![15, 16],
            vec![17, 18, 19],
            vec![20, 21, 22],
            vec![23, 24, 25],
            vec![26, 27, 28],
            vec![29, 30, 31],
        ];
        simple1.shuffle(&mut rand::rng());
        // simple2.shuffle(&mut rand::rng());
        // simple1.reverse();
        simple2.reverse();
        let (first, middle, second, _) = create_escalator_identities(
            32,
            &simple1,
            &simple2,
            &env,
            &dbs
        );
        // middle.gates.reverse();
        let mut c = CircuitSeq {gates: Vec::new() };
        let first = CircuitSeq { gates: first.into_iter().flatten().collect() };
        let second = CircuitSeq { gates: second.into_iter().flatten().collect() };
        c.gates.extend_from_slice(&first.gates);
        c.gates.extend_from_slice(&middle.gates);
        c.gates.extend_from_slice(&second.gates);
        let repr = c.repr();
        writeln!(file, "{}", repr)
            .expect("Failed to write to file");

        let id = CircuitSeq { gates: Vec::new() };
        if c.probably_equal(&id, 32, 10000).is_err() {
            panic!("Not an id");
        }
        println!("Wrote test circuit to file");
    }
}
