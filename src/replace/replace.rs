use crate::{
    circuit::circuit::{CircuitSeq, Permutation}, rainbow::canonical::Canonicalization, random::random_data::{
        contiguous_convex, 
        // find_convex_subcircuit, 
        get_canonical, 
        random_circuit, 
        // shoot_left_vec, 
        shoot_random_gate, 
        shoot_random_gate_gate_ver,
        simple_find_convex_subcircuit,
        // find_convex_subcircuit_deep,  
        // targeted_convex_subcircuit,
        targeted_find_convex_subcircuit_deep,
    }
};

use itertools::Itertools;
use rand::Rng;

use rusqlite::{Connection, Statement};

use lmdb::{Cursor, Database, RoCursor, RoTransaction, Transaction};

use libc::c_uint;
extern crate lmdb_sys;
use lmdb_sys as ffi;
use serde::{Serialize, Deserialize};
use std::{
    cmp::{max, min},
    collections::{HashMap, HashSet},
    // fs::OpenOptions, // used for testing
    // io::Write,
    // sync::Arc,
    marker::PhantomData,
    ptr,
    slice,
    time::Instant,
};
use rand::prelude::SliceRandom;
use std::io::{self, Read};
use std::os::unix::io::AsRawFd;
use libc::{fcntl, F_GETFL, F_SETFL, O_NONBLOCK};
use std::sync::atomic::{AtomicU64, Ordering};
// use rand::prelude::IndexedRandom;

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
pub fn random_canonical_id(
    env: &lmdb::Environment,
    _conn: &Connection,
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

static GET_ID_TOTAL_TIME: AtomicU64 = AtomicU64::new(0);
// static DB_NAME_TIME: AtomicU64 = AtomicU64::new(0);
// static DB_LOOKUP_TIME: AtomicU64 = AtomicU64::new(0);
// static TXN_BEGIN_TIME: AtomicU64 = AtomicU64::new(0);
// static SERIALIZE_KEY_TIME: AtomicU64 = AtomicU64::new(0);
// static LMDB_GET_TIME: AtomicU64 = AtomicU64::new(0);
// static DESERIALIZE_LIST_TIME: AtomicU64 = AtomicU64::new(0);
// static RNG_CHOOSE_TIME: AtomicU64 = AtomicU64::new(0);

fn get_random_identity(
    n: usize,
    gate_pair: GatePair,
    env: &lmdb::Environment,
    dbs: &HashMap<String, Database>,
) -> Result<CircuitSeq, Box<dyn std::error::Error>> {
    let total_start = Instant::now();

    let g = GatePair::to_int(&gate_pair);
    let db_name = format!("ids_n{}g{}", n, g);

    let db = dbs.get(&db_name).unwrap_or_else(|| {
        panic!("Failed to get DB with name: {}", db_name);
    });

    // Hardcoded max entries for all DBs
    let max_entries: usize = match db_name.as_str() {
        // n5
        "ids_n5g1" => 100_235,
        "ids_n5g2" => 177_541,
        "ids_n5g3" => 169_347,
        "ids_n5g4" => 177_481,
        "ids_n5g5" => 88_913,
        "ids_n5g6" => 119_879,
        "ids_n5g7" => 169_161,
        "ids_n5g8" => 119_872,
        "ids_n5g9" => 90_257,
        "ids_n5g10" => 294_944,
        "ids_n5g11" => 158_422,
        "ids_n5g12" => 158_411,
        "ids_n5g13" => 340_518,
        "ids_n5g14" => 494_202,
        "ids_n5g15" => 133_325,
        "ids_n5g16" => 136_497,
        "ids_n5g17" => 530_248,
        "ids_n5g18" => 116_600,
        "ids_n5g19" => 283_097,
        "ids_n5g20" => 122_255,
        "ids_n5g21" => 156_822,
        "ids_n5g22" => 116_524,
        "ids_n5g23" => 291_005,
        "ids_n5g24" => 140_980,
        "ids_n5g25" => 447_910,
        "ids_n5g26" => 156_746,
        "ids_n5g27" => 121_233,
        "ids_n5g28" => 529_131,
        "ids_n5g29" => 282_660,
        "ids_n5g30" => 290_595,
        "ids_n5g31" => 138_000,
        "ids_n5g32" => 446_888,
        "ids_n5g33" => 138_616,
        // n6
        "ids_n6g0" => 289_970,
        "ids_n6g1" => 467_194,
        "ids_n6g2" => 832_725,
        "ids_n6g3" => 774_667,
        "ids_n6g4" => 832_405,
        "ids_n6g5" => 349_762,
        "ids_n6g6" => 498_925,
        "ids_n6g7" => 774_303,
        "ids_n6g8" => 498_838,
        "ids_n6g9" => 386_650,
        "ids_n6g10" => 861_396,
        "ids_n6g11" => 441_737,
        "ids_n6g12" => 441_678,
        "ids_n6g13" => 1_084_718,
        "ids_n6g14" => 1_644_996,
        "ids_n6g15" => 284_939,
        "ids_n6g16" => 429_717,
        "ids_n6g17" => 1_700_523,
        "ids_n6g18" => 306_587,
        "ids_n6g19" => 795_521,
        "ids_n6g20" => 280_532,
        "ids_n6g21" => 302_158,
        "ids_n6g22" => 306_536,
        "ids_n6g23" => 709_587,
        "ids_n6g24" => 400_341,
        "ids_n6g25" => 1_386_212,
        "ids_n6g26" => 301_986,
        "ids_n6g27" => 245_057,
        "ids_n6g28" => 1_694_936,
        "ids_n6g29" => 794_260,
        "ids_n6g30" => 708_822,
        "ids_n6g31" => 357_496,
        "ids_n6g32" => 1_381_063,
        "ids_n6g33" => 316_088,
        // n7
        "ids_n7g0" => 1_068,
        "ids_n7g1" => 3_213,
        "ids_n7g2" => 2_705,
        "ids_n7g3" => 4_613,
        "ids_n7g4" => 2_704,
        "ids_n7g5" => 1_019,
        "ids_n7g6" => 2_371,
        "ids_n7g7" => 4_635,
        "ids_n7g8" => 2_371,
        "ids_n7g9" => 2_392,
        "ids_n7g10" => 6_651,
        "ids_n7g11" => 1_811,
        "ids_n7g12" => 1_805,
        "ids_n7g13" => 8_085,
        "ids_n7g14" => 9_850,
        "ids_n7g15" => 1_293,
        "ids_n7g16" => 2_675,
        "ids_n7g17" => 13_193,
        "ids_n7g18" => 1_000,
        "ids_n7g19" => 4_741,
        "ids_n7g20" => 1_819,
        "ids_n7g21" => 1_002,
        "ids_n7g22" => 1_000,
        "ids_n7g23" => 4_844,
        "ids_n7g24" => 3_899,
        "ids_n7g25" => 19_153,
        "ids_n7g26" => 1_003,
        "ids_n7g27" => 2_172,
        "ids_n7g28" => 12_904,
        "ids_n7g29" => 4_711,
        "ids_n7g30" => 4_816,
        "ids_n7g31" => 2_536,
        "ids_n7g32" => 18_903,
        "ids_n7g33" => 1_433,
        // n16
        "ids_n16g0"  => 41_500,
        "ids_n16g1"  => 11_600,
        "ids_n16g2"  => 19_400,
        "ids_n16g3"  => 16_300,
        "ids_n16g4"  => 19_100,
        "ids_n16g5"  => 8_700,
        "ids_n16g6"  => 9_600,
        "ids_n16g7"  => 16_500,
        "ids_n16g8"  => 9_700,
        "ids_n16g9"  => 9_600,
        "ids_n16g10" => 12_700,
        "ids_n16g11" => 4_000,
        "ids_n16g12" => 4_100,
        "ids_n16g13" => 16_000,
        "ids_n16g14" => 17_600,
        "ids_n16g15" => 3_600,
        "ids_n16g16" => 6_700,
        "ids_n16g17" => 34_300,
        "ids_n16g18" => 2_700,
        "ids_n16g19" => 12_500,
        "ids_n16g20" => 4_800,
        "ids_n16g21" => 2_900,
        "ids_n16g22" => 2_700,
        "ids_n16g23" => 9_700,
        "ids_n16g24" => 6_000,
        "ids_n16g25" => 26_200,
        "ids_n16g26" => 2_800,
        "ids_n16g27" => 4_000,
        "ids_n16g28" => 34_600,
        "ids_n16g29" => 12_600,
        "ids_n16g30" => 9_700,
        "ids_n16g31" => 4_400,
        "ids_n16g32" => 26_400,
        "ids_n16g33" => 3_400,
        _ => panic!("DB {} not in hardcoded max_entries", db_name),
    };

    let mut rng = rand::rng();
    let random_index = rng.random_range(0..max_entries);

    let txn = env.begin_ro_txn()?;
    let mut cursor = txn.open_ro_cursor(*db)?;

    let value_bytes = cursor.iter_start()
        .nth(random_index)
        .map(|(k, _v)| k)
        .expect("Failed to get random key");

    let out = CircuitSeq::from_blob(value_bytes);

    GET_ID_TOTAL_TIME.fetch_add(
        total_start.elapsed().as_nanos() as u64,
        Ordering::Relaxed,
    );

    Ok(out)
}

pub fn get_random_wide_identity(
    n: usize, 
    env: &lmdb::Environment,
    dbs: &HashMap<String, Database>,
    conn: &mut Connection,
    bit_shuf_list: &Vec<Vec<Vec<usize>>>,
) -> CircuitSeq {
    let mut id = CircuitSeq { gates: Vec::new() };
    let mut uw = id.used_wires();
    let mut nwires = uw.len();
    let mut rng = rand::rng();
    let mut len = 0;
    while nwires < 16 || len < 160 {
        shoot_random_gate(&mut id, 100_000);
        let gp = GatePair::from_int(rng.random_range(0..34));
        let mut i = match get_random_identity(6, gp, env, dbs) {
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
            let min = min_vals[0];
            let mut used_wires = vec![id.gates[min][0], id.gates[min][1], id.gates[min][2]];
            let mut unused_wires: Vec<u8> = (0..n as u8)
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

    let mut shuf: Vec<usize> = (0..n).collect();
    shuf.shuffle(&mut rng);

    let bit_shuf = Permutation { data: shuf };
    id.rewire(&bit_shuf, n);
    id
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
static PICK_SUBCIRCUIT_TIME: AtomicU64 = AtomicU64::new(0);
static CANONICALIZE_TIME: AtomicU64 = AtomicU64::new(0);
static ROW_FETCH_TIME: AtomicU64 = AtomicU64::new(0);
static SROW_FETCH_TIME: AtomicU64 = AtomicU64::new(0);
static SIXROW_FETCH_TIME: AtomicU64 = AtomicU64::new(0);
static LROW_FETCH_TIME: AtomicU64 = AtomicU64::new(0);
static DB_OPEN_TIME: AtomicU64 = AtomicU64::new(0);
static TXN_TIME: AtomicU64 = AtomicU64::new(0);
static LMDB_LOOKUP_TIME: AtomicU64 = AtomicU64::new(0);
static FROM_BLOB_TIME: AtomicU64 = AtomicU64::new(0);
static SPLICE_TIME: AtomicU64 = AtomicU64::new(0);
static TRIAL_TIME: AtomicU64 = AtomicU64::new(0);
static IDENTITY_TIME: AtomicU64 = AtomicU64::new(0);

pub fn compress(
    c: &CircuitSeq,
    trials: usize,
    conn: &mut Connection,
    bit_shuf: &Vec<Vec<usize>>,
    n: usize,
) -> CircuitSeq {

    let id = Permutation::id_perm(n);

    // let t0 = Instant::now();
    let c_perm = c.permutation(n);
    // PERMUTATION_TIME.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

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

            // let sql_t0 = Instant::now();
            let mut stmt = match conn.prepare(&query) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let rows = stmt.query([&subcircuit.repr_blob()]);
            // SQL_TIME.fetch_add(sql_t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

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
            // let t1 = Instant::now();
            let sub_perm = subcircuit.permutation(n);
            // PERMUTATION_TIME.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);

            // let t2 = Instant::now();
            let canon_perm = get_canonical(&sub_perm, bit_shuf);
            // CANON_TIME.fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);

            (canon_perm.perm.repr_blob(), canon_perm.shuffle.repr_blob())
        };

        for smaller_m in 1..=sub_m {
            let table = format!("n{}m{}", n, smaller_m);
            let query = format!(
                "SELECT * FROM {} WHERE perm = ?1 ORDER BY RANDOM() LIMIT 1",
                table
            );

            // let sql_t0 = Instant::now();
            let mut stmt = match conn.prepare(&query) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let rows = stmt.query([&canon_perm_blob]);
            // SQL_TIME.fetch_add(sql_t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

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

pub fn expand_lmdb<'a>(
    c: &CircuitSeq,
    trials: usize,
    bit_shuf: &Vec<Vec<usize>>,
    n: usize,
    env: &lmdb::Environment,
    _old_n: usize,
    dbs: &HashMap<String, lmdb::Database>,
    prepared_stmt: &mut rusqlite::Statement<'a>,
    prepared_stmt2: &mut rusqlite::Statement<'a>,
    conn: &Connection
) -> CircuitSeq {
    let mut compressed = c.clone();
    if compressed.gates.is_empty() {
        return CircuitSeq { gates: Vec::new() };
    }
    let perm_len = 1 << n;
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
            10
        };

        let sub_m = subcircuit.gates.len();
        let (canon_perm_blob, canon_shuf_blob) =
        if sub_m <= max && ((n == 6 && sub_m == 5) || (n == 7 && sub_m  == 4)) {
            if n == 7 && sub_m == 4 {
                let stmt: &mut Statement<'_> = &mut *prepared_stmt;

                let row_start = Instant::now();
                let blobs_result: rusqlite::Result<(Vec<u8>, Vec<u8>)> =
                    stmt.query_row(
                        [&subcircuit.repr_blob()],
                        |row| Ok((row.get(0)?, row.get(1)?)),
                    );

                SROW_FETCH_TIME.fetch_add(
                    row_start.elapsed().as_nanos() as u64,
                    Ordering::Relaxed,
                );

                match blobs_result {
                    Ok(b) => b,
                    Err(rusqlite::Error::QueryReturnedNoRows) => continue,
                    Err(e) => panic!("SQL query failed: {:?}", e),
                }

            } else if n == 6 && sub_m == 5 {
                let stmt: &mut Statement<'_> = &mut *prepared_stmt2;

                let row_start = Instant::now();
                let blobs_result: rusqlite::Result<(Vec<u8>, Vec<u8>)> =
                    stmt.query_row(
                        [&subcircuit.repr_blob()],
                        |row| Ok((row.get(0)?, row.get(1)?)),
                    );

                SIXROW_FETCH_TIME.fetch_add(
                    row_start.elapsed().as_nanos() as u64,
                    Ordering::Relaxed,
                );

                match blobs_result {
                    Ok(b) => b,
                    Err(rusqlite::Error::QueryReturnedNoRows) => continue,
                    Err(e) => panic!("SQL query failed: {:?}", e),
                }
            
            } else {
                let table = format!("n{}m{}", n, sub_m);
                let query = format!(
                    "SELECT perm, shuf FROM {} WHERE circuit = ?1 LIMIT 1",
                    table
                );

                let row_start = Instant::now();
                let blobs_result: rusqlite::Result<(Vec<u8>, Vec<u8>)> =
                    conn.query_row(
                        &query,
                        [&subcircuit.repr_blob()],
                        |row| Ok((row.get(0)?, row.get(1)?)),
                    );

                ROW_FETCH_TIME.fetch_add(
                    row_start.elapsed().as_nanos() as u64,
                    Ordering::Relaxed,
                );

                match blobs_result {
                    Ok(b) => b,
                    Err(rusqlite::Error::QueryReturnedNoRows) => continue,
                    Err(e) => panic!("SQL query failed: {:?}", e),
                }
            }
        } else if sub_m <= max && (n >= 4) {
            let db_name = format!("n{}m{}perms", n, sub_m);
                let db = match dbs.get(&db_name) {
                    Some(db) => *db,
                    None => continue,
                };

                let txn = env.begin_ro_txn().expect("lmdb ro txn");

                let row_start = Instant::now();
                let val = match txn.get(db, &subcircuit.repr_blob()) {
                    Ok(v) => v,
                    Err(lmdb::Error::NotFound) => continue,
                    Err(e) => panic!("LMDB get failed: {:?}", e),
                };
                LROW_FETCH_TIME.fetch_add(row_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

                let perm = val[..perm_len].to_vec();
                let shuf = val[perm_len..].to_vec();

                (perm, shuf)
        } else {
            // let t1 = Instant::now();
            let sub_perm = subcircuit.permutation(n);
            // PERMUTATION_TIME.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);

            // let t2 = Instant::now();
            let canon_perm = get_canonical(&sub_perm, bit_shuf);
            // CANON_TIME.fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);

            (canon_perm.perm.repr_blob(), canon_perm.shuffle.repr_blob())
        };

        let prefix = canon_perm_blob.as_slice();
        for smaller_m in (1..=max).rev() {
            let db_name = format!("n{}m{}", n, smaller_m);
            let &db = match dbs.get(&db_name) {
                Some(db) => db,
                None => continue,
            };
            let mut invert = false;
            let hit = {
                let txn = env.begin_ro_txn().expect("txn");

                // let t0 = Instant::now();
                
                let mut res = random_perm_lmdb(&txn, db, prefix);
                if res.is_none() {
                    let prefix_inv_blob = Permutation::from_blob(&prefix).invert().repr_blob();
                    invert = true;
                    res = random_perm_lmdb(&txn, db, &prefix_inv_blob);
                }

                // SQL_TIME.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

                res.map(|val_blob| val_blob)
            };

            if let Some(val_blob) = hit {
                let repl_blob: Vec<u8> = val_blob;

                let mut repl = CircuitSeq::from_blob(&repl_blob);

                if invert {
                    repl.gates.reverse();
                }

                repl.rewire(&Permutation::from_blob(&canon_shuf_blob).invert(), n);

                if repl.gates.len() == end - start {
                    compressed.gates[start..end].copy_from_slice(&repl.gates);
                } else {
                    compressed.gates.splice(start..end, repl.gates);
                }
                break;
            }
        }

    }

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

pub fn compress_big(
    c: &CircuitSeq, 
    trials: usize, 
    num_wires: usize, 
    conn: &mut Connection, 
    env: &lmdb::Environment, 
    bit_shuf_list: &Vec<Vec<Vec<usize>>>, 
    dbs: &HashMap<String, lmdb::Database>,
) -> CircuitSeq {
    let table = format!("n{}m{}", 7, 4);
    let query_limit = format!("SELECT perm, shuf FROM {} WHERE circuit = ?1 LIMIT 1", table);
    let mut stmt = conn.prepare(&query_limit).unwrap();
    let table2 = format!("n{}m{}", 6, 5);
    let query_limit = format!("SELECT perm, shuf FROM {} WHERE circuit = ?1 LIMIT 1", table2);
    let mut stmt2 = conn.prepare(&query_limit).unwrap();
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
        shoot_random_gate(&mut circuit, 100_000);
        let t0 = Instant::now();
        let mut subcircuit_gates = vec![];
        let random_max_wires = rng.random_range(5..=7);
        let size = if random_max_wires == 7 {
            6
        } else if random_max_wires == 6 {
            4
        } else {
            3
        };
        for set_size in (3..=size).rev() {
            let (gates, _) = simple_find_convex_subcircuit(set_size, random_max_wires, num_wires, &circuit, &mut rng);
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
        let bit_shuf = &bit_shuf_list[sub_num_wires - 3];
        PERMUTATION_TIME.fetch_add(t3.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t4 = Instant::now();
        let subcircuit_temp = compress_lmdb(&subcircuit, 20, &bit_shuf, sub_num_wires, env, dbs, &mut stmt, &mut stmt2, conn);
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

pub fn sequential_compress_big(
    c: &CircuitSeq, 
    num_wires: usize, 
    conn: &mut Connection, 
    env: &lmdb::Environment, 
    bit_shuf_list: &Vec<Vec<Vec<usize>>>, 
    dbs: &HashMap<String, lmdb::Database>,
) -> CircuitSeq {
    let table = format!("n{}m{}", 7, 4);
    let query_limit = format!("SELECT perm, shuf FROM {} WHERE circuit = ?1 LIMIT 1", table);
    let mut stmt = conn.prepare(&query_limit).unwrap();
    let table2 = format!("n{}m{}", 6, 5);
    let query_limit = format!("SELECT perm, shuf FROM {} WHERE circuit = ?1 LIMIT 1", table2);
    let mut stmt2 = conn.prepare(&query_limit).unwrap();
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

    let mut len = circuit.gates.len();
    let mut i = 0;
    while i < len {
        let t0 = Instant::now();
        let mut subcircuit_gates = vec![];
        let random_max_wires = rng.random_range(5..=7);
        let size = if random_max_wires == 7 {
            6
        } else if random_max_wires == 6 {
            4
        } else {
            3
        };
        for set_size in (3..=size).rev() {
            let (gates, _) = targeted_find_convex_subcircuit_deep(set_size, random_max_wires, num_wires, &circuit, &mut rng, i);
            if !gates.is_empty() {
                subcircuit_gates = gates;
                break;
            }
            if set_size == 3 {
                let (gates, _) = targeted_find_convex_subcircuit_deep(set_size, 7, num_wires, &circuit, &mut rng, i);
                subcircuit_gates = gates;
            }
        }
        CONVEX_FIND_TIME.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

        if subcircuit_gates.is_empty() {
            i+=1;
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
            i+=1;
            continue;
        }

        let t2 = Instant::now();
        let used_wires = subcircuit.used_wires();
        subcircuit = CircuitSeq::rewire_subcircuit(&mut circuit, &mut subcircuit_gates, &used_wires);
        REWIRE_TIME.fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t3 = Instant::now();
        let sub_num_wires = used_wires.len();
        let bit_shuf = &bit_shuf_list[sub_num_wires - 3];
        PERMUTATION_TIME.fetch_add(t3.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t4 = Instant::now();
        let subcircuit_temp = compress_lmdb(&subcircuit, 20, &bit_shuf, sub_num_wires, env, dbs, &mut stmt, &mut stmt2, conn);
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
        i += 1;
        len = circuit.gates.len();
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

pub fn sequential_compress_big_ancillas( 
    c: &CircuitSeq, 
    num_wires: usize, 
    conn: &mut Connection, 
    env: &lmdb::Environment, 
    bit_shuf_list: &Vec<Vec<Vec<usize>>>, 
    dbs: &HashMap<String, lmdb::Database>,
) -> CircuitSeq {
    let table = format!("n{}m{}", 7, 4);
    let query_limit = format!("SELECT perm, shuf FROM {} WHERE circuit = ?1 LIMIT 1", table);
    let mut stmt = conn.prepare(&query_limit).unwrap();
    let table2 = format!("n{}m{}", 6, 5);
    let query_limit = format!("SELECT perm, shuf FROM {} WHERE circuit = ?1 LIMIT 1", table2);
    let mut stmt2 = conn.prepare(&query_limit).unwrap();
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

    let mut len = circuit.gates.len();
    let mut i = 0;
    while i < len {
        let t0 = Instant::now();
        let mut subcircuit_gates = vec![];
        let random_max_wires = rng.random_range(5..=7);
        let size = if random_max_wires == 7 {
            6
        } else if random_max_wires == 6 {
            4
        } else {
            3
        };
        for set_size in (3..=size).rev() {
            let (gates, _) = targeted_find_convex_subcircuit_deep(set_size, random_max_wires, num_wires, &circuit, &mut rng, i);
            if !gates.is_empty() {
                subcircuit_gates = gates;
                break;
            }
            if set_size == 3 {
                let (gates, _) = targeted_find_convex_subcircuit_deep(set_size, 7, num_wires, &circuit, &mut rng, i);
                subcircuit_gates = gates;
            }
        }
        CONVEX_FIND_TIME.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

        if subcircuit_gates.is_empty() {
            i+=1;
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
            i+=1;
            continue;
        }

        let t2 = Instant::now();
        let mut used_wires = subcircuit.used_wires();
        let n_wires = used_wires.len();
        let max = 7;
        let new_wires = rng.random_range(n_wires..=max);
        if new_wires > n_wires {
            let mut count = n_wires;
            while count < new_wires {
                let random = rng.random_range(0..num_wires);
                if used_wires.contains(&(random as u8)) {
                    continue
                }
                used_wires.push(random as u8);
                count += 1;
            }
        }
        subcircuit = CircuitSeq::rewire_subcircuit(&mut circuit, &mut subcircuit_gates, &used_wires);
        REWIRE_TIME.fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t3 = Instant::now();
        let sub_num_wires = used_wires.len();
        let bit_shuf = &bit_shuf_list[sub_num_wires - 3];
        PERMUTATION_TIME.fetch_add(t3.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t4 = Instant::now();
        let subcircuit_temp = compress_lmdb(&subcircuit, 20, &bit_shuf, sub_num_wires, env, dbs, &mut stmt, &mut stmt2, conn);
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
        i += 1;
        len = circuit.gates.len();
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

fn random_perm_lmdb(
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

pub fn compress_lmdb<'a>(
    c: &CircuitSeq,
    trials: usize,
    bit_shuf: &Vec<Vec<usize>>,
    n: usize,
    env: &lmdb::Environment,
    dbs: &HashMap<String, lmdb::Database>,
    prepared_stmt: &mut rusqlite::Statement<'a>,
    prepared_stmt2: &mut rusqlite::Statement<'a>,
    conn: &Connection,
) -> CircuitSeq {
    let id = Permutation::id_perm(n);
    let perm_len = 1 << n;
    // Timer for initial permutation
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

    // Timer for initial deduplication
    let dedup_start = Instant::now();
    let mut i = 0;
    while i < compressed.gates.len().saturating_sub(1) {
        if compressed.gates[i] == compressed.gates[i + 1] {
            compressed.gates.drain(i..=i + 1);
            i = i.saturating_sub(2);
        } else {
            i += 1;
        }
    }
    DEDUP_TIME.fetch_add(dedup_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

    if compressed.gates.is_empty() {
        return CircuitSeq { gates: Vec::new() };
    }

    let (do_subcircuit, trial_count) = if compressed.gates.len() < 5 {
        (false, 2)
    } else {
        (true, trials)
    };

    for _ in 0..trial_count {
        let trial_start = Instant::now();

        // Pick subcircuit
        let pick_start = Instant::now();
        let (subcircuit, start, end) = if do_subcircuit {
            random_subcircuit(&compressed)
        } else {
            (compressed.clone(), 0, compressed.gates.len())
        };
        PICK_SUBCIRCUIT_TIME.fetch_add(pick_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let mut subcircuit = subcircuit;

        // Canonicalize
        let canon_start = Instant::now();
        subcircuit.canonicalize();
        CANONICALIZE_TIME.fetch_add(canon_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let max = if n == 7 { 3 } else if n == 5 || n == 6 { 5 } else if n == 4 { 6 } else { 10 };
        let sub_m = subcircuit.gates.len();
        let min = min(sub_m, max);

        let (canon_perm_blob, canon_shuf_blob) = 
            if sub_m <= max && ((n == 6 && sub_m == 5) || (n == 7 && sub_m  == 4)) {
                if n == 7 && sub_m == 4 {
                    let stmt: &mut Statement<'_> = &mut *prepared_stmt;

                    let row_start = Instant::now();
                    let blobs_result: rusqlite::Result<(Vec<u8>, Vec<u8>)> =
                        stmt.query_row(
                            [&subcircuit.repr_blob()],
                            |row| Ok((row.get(0)?, row.get(1)?)),
                        );

                    SROW_FETCH_TIME.fetch_add(
                        row_start.elapsed().as_nanos() as u64,
                        Ordering::Relaxed,
                    );

                    match blobs_result {
                        Ok(b) => b,
                        Err(rusqlite::Error::QueryReturnedNoRows) => continue,
                        Err(e) => panic!("SQL query failed: {:?}", e),
                    }

                } else if n == 6 && sub_m == 5 {
                    let stmt: &mut Statement<'_> = &mut *prepared_stmt2;

                    let row_start = Instant::now();
                    let blobs_result: rusqlite::Result<(Vec<u8>, Vec<u8>)> =
                        stmt.query_row(
                            [&subcircuit.repr_blob()],
                            |row| Ok((row.get(0)?, row.get(1)?)),
                        );

                    SIXROW_FETCH_TIME.fetch_add(
                        row_start.elapsed().as_nanos() as u64,
                        Ordering::Relaxed,
                    );

                    match blobs_result {
                        Ok(b) => b,
                        Err(rusqlite::Error::QueryReturnedNoRows) => continue,
                        Err(e) => panic!("SQL query failed: {:?}", e),
                    }
                } else {
                    let table = format!("n{}m{}", n, sub_m);
                    let query = format!(
                        "SELECT perm, shuf FROM {} WHERE circuit = ?1 LIMIT 1",
                        table
                    );
                    let row_start = Instant::now();
                    let blobs_result: rusqlite::Result<(Vec<u8>, Vec<u8>)> =
                        conn.query_row(
                            &query,
                            [&subcircuit.repr_blob()],
                            |row| Ok((row.get(0)?, row.get(1)?)),
                        );

                    ROW_FETCH_TIME.fetch_add(
                        row_start.elapsed().as_nanos() as u64,
                        Ordering::Relaxed,
                    );

                    match blobs_result {
                        Ok(b) => {
                            println!("{}", table);
                            b
                        },
                        Err(rusqlite::Error::QueryReturnedNoRows) => continue,
                        Err(e) => panic!("SQL query failed: {:?}", e),
                    }
                }
            } else if sub_m <= max && (n >= 4) {
                let db_name = format!("n{}m{}perms", n, min);
                let db = match dbs.get(&db_name) {
                    Some(db) => *db,
                    None => continue,
                };

                let txn = env.begin_ro_txn().expect("lmdb ro txn");

                let row_start = Instant::now();
                let val = match txn.get(db, &subcircuit.repr_blob()) {
                    Ok(v) => v,
                    Err(lmdb::Error::NotFound) => continue,
                    Err(e) => panic!("LMDB get failed: {:?}", e),
                };
                LROW_FETCH_TIME.fetch_add(row_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

                let perm = val[..perm_len].to_vec();
                let shuf = val[perm_len..].to_vec();

                (perm, shuf)
            } else {
            // Permutation + canonicalization
            let perm_start = Instant::now();
            let sub_perm = subcircuit.permutation(n);
            PERMUTATION_TIME.fetch_add(perm_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

            let canon_start = Instant::now();
            let canon_perm = get_canonical(&sub_perm, bit_shuf);
            CANON_TIME.fetch_add(canon_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

            (canon_perm.perm.repr_blob(), canon_perm.shuffle.repr_blob())
        };

        let prefix = canon_perm_blob.as_slice();

        for smaller_m in 1..=min {
            let db_open_start = Instant::now();
            let db_name = format!("n{}m{}", n, smaller_m);
            let &db = match dbs.get(&db_name) {
                Some(db) => db,
                None => continue,
            };
            DB_OPEN_TIME.fetch_add(db_open_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

            let txn_start = Instant::now();
            let txn = env.begin_ro_txn().expect("txn");
            TXN_TIME.fetch_add(txn_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
            let lookup_start = Instant::now();
            let mut invert = false;
            let mut res = random_perm_lmdb(&txn, db, prefix);
            if res.is_none() {
                let prefix_inv_blob = Permutation::from_blob(&prefix).invert().repr_blob();
                invert = true;
                res = random_perm_lmdb(&txn, db, &prefix_inv_blob);
            }
            LMDB_LOOKUP_TIME.fetch_add(lookup_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
            

            if let Some(val_blob) = res {
                let from_blob_start = Instant::now();
                let mut repl = CircuitSeq::from_blob(&val_blob);
                FROM_BLOB_TIME.fetch_add(from_blob_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

                let rewire_start = Instant::now();
                if invert { repl.gates.reverse(); }
                repl.rewire(&Permutation::from_blob(&canon_shuf_blob).invert(), n);
                REWIRE_TIME.fetch_add(rewire_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

                let splice_start = Instant::now();
                if repl.gates.len() == end - start { 
                    compressed.gates[start..end].copy_from_slice(&repl.gates);
                } else {
                    compressed.gates.splice(start..end, repl.gates);
                }
                SPLICE_TIME.fetch_add(splice_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

                break;
            }
        }

        TRIAL_TIME.fetch_add(trial_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    // Final deduplication
    let dedup2_start = Instant::now();
    let mut j = 0;
    while j < compressed.gates.len().saturating_sub(1) {
        if compressed.gates[j] == compressed.gates[j + 1] {
            compressed.gates.drain(j..=j + 1);
            j = j.saturating_sub(2);
        } else {
            j += 1;
        }
    }
    DEDUP_TIME.fetch_add(dedup2_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

    compressed
}

pub fn expand_big(
    c: &CircuitSeq, 
    trials: usize, 
    num_wires: usize, 
    conn: &mut Connection, 
    env: &lmdb::Environment, 
    bit_shuf_list: &Vec<Vec<Vec<usize>>>, 
    dbs: &HashMap<String, lmdb::Database>,
) -> CircuitSeq {
    let table = format!("n{}m{}", 7, 4);
    let query_limit = format!("SELECT perm, shuf FROM {} WHERE circuit = ?1 LIMIT 1", table);
    let mut stmt = conn.prepare(&query_limit).unwrap();
    let table2 = format!("n{}m{}", 6, 5);
    let query_limit = format!("SELECT perm, shuf FROM {} WHERE circuit = ?1 LIMIT 1", table2);
    let mut stmt2 = conn.prepare(&query_limit).unwrap();
    let mut circuit = c.clone();
    let mut rng = rand::rng();

    for _i in 0..trials {
        // if i % 20 == 0 {
        //     println!("{} trials so far, {} more to go", i, trials - i);
        // }
        let mut subcircuit_gates = vec![];
        let random_max_wires = rng.random_range(3..=7);
        for set_size in (3..=7).rev() {
            let (gates, _) = simple_find_convex_subcircuit(set_size, random_max_wires, num_wires, &circuit, &mut rng);
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
        // let sub_ref = subcircuit.clone();
        let expected_slice: Vec<_> = subcircuit_gates.iter().map(|&i| circuit.gates[i]).collect();
        let actual_slice = &circuit.gates[start..=end];

        if actual_slice != &expected_slice[..] {
            break;
        }

        let mut used_wires = subcircuit.used_wires();
        let n_wires = used_wires.len();
        let max = 7;
        let new_wires = rng.random_range(n_wires..=max);

        if new_wires > n_wires {
            let mut count = n_wires;
            while count < new_wires {
                let random = rng.random_range(0..num_wires);
                if used_wires.contains(&(random as u8)) {
                    continue
                }
                used_wires.push(random as u8);
                count += 1;
            }
        }
        used_wires.sort();
        subcircuit = CircuitSeq::rewire_subcircuit(&mut circuit, &mut subcircuit_gates, &used_wires);

        
        let bit_shuf = &bit_shuf_list[new_wires - 3];

        let subcircuit_temp = expand_lmdb(&subcircuit, 10, &bit_shuf, new_wires, &env, n_wires, dbs, &mut stmt, &mut stmt2, conn);
        subcircuit = subcircuit_temp;

        subcircuit = CircuitSeq::unrewire_subcircuit(&subcircuit, &used_wires);
        if subcircuit.gates.len() == end+1 - start {
            circuit.gates[start..end+1].copy_from_slice(&subcircuit.gates);
        } else {    
            circuit.gates.splice(start..end+1, subcircuit.gates);
        }
        // if c.permutation(num_wires).data != circuit.permutation(num_wires).data {
        //     panic!("splice changed something");
        // }
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

pub fn compress_big_ancillas(
    c: &CircuitSeq, 
    trials: usize, 
    num_wires: usize, 
    conn: &mut Connection, 
    env: &lmdb::Environment, 
    bit_shuf_list: &Vec<Vec<Vec<usize>>>, 
    dbs: &HashMap<String, lmdb::Database>, 
) -> CircuitSeq {
    let table = format!("n{}m{}", 7, 4);
    let query_limit = format!("SELECT perm, shuf FROM {} WHERE circuit = ?1 LIMIT 1", table);
    let mut stmt = conn.prepare(&query_limit).unwrap();
    let table = format!("n{}m{}", 6, 5);
    let query_limit = format!("SELECT perm, shuf FROM {} WHERE circuit = ?1 LIMIT 1", table);
    let mut stmt2 = conn.prepare(&query_limit).unwrap();
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
        // let t0 = Instant::now();
        let mut subcircuit_gates = vec![];
        let random_max_wires = rng.random_range(3..=7);
        for set_size in (3..=6).rev() {
            let (gates, _) = simple_find_convex_subcircuit(set_size, random_max_wires, num_wires, &circuit, &mut rng);
            if !gates.is_empty() {
                subcircuit_gates = gates;
                break;
            }
        }
        // CONVEX_FIND_TIME.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

        if subcircuit_gates.is_empty() {
            continue;
        }

        let gates: Vec<[u8; 3]> = subcircuit_gates.iter().map(|&g| circuit.gates[g]).collect();
        subcircuit_gates.sort();

        // let t1 = Instant::now();
        let (start, end) = contiguous_convex(&mut circuit, &mut subcircuit_gates, num_wires).unwrap();
        // CONTIGUOUS_TIME.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let mut subcircuit = CircuitSeq { gates };

        let expected_slice: Vec<_> = subcircuit_gates.iter().map(|&i| circuit.gates[i]).collect();
        let actual_slice = &circuit.gates[start..=end];
        if actual_slice != &expected_slice[..] {
            continue;
        }

        // let t2 = Instant::now();
        let mut used_wires = subcircuit.used_wires();
        let n_wires = used_wires.len();
        let max = 7;
        let new_wires = rng.random_range(n_wires..=max);
        if new_wires > n_wires {
            let mut count = n_wires;
            while count < new_wires {
                let random = rng.random_range(0..num_wires);
                if used_wires.contains(&(random as u8)) {
                    continue
                }
                used_wires.push(random as u8);
                count += 1;
            }
        }
        // used_wires.sort();
        subcircuit = CircuitSeq::rewire_subcircuit(&mut circuit, &mut subcircuit_gates, &used_wires);
        // REWIRE_TIME.fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);

        // let t3 = Instant::now();
        let sub_num_wires = used_wires.len();
        let bit_shuf = &bit_shuf_list[sub_num_wires - 3];

        // PERMUTATION_TIME.fetch_add(t3.elapsed().as_nanos() as u64, Ordering::Relaxed);

        // let t4 = Instant::now();
        let subcircuit_temp = compress_lmdb(&subcircuit, 20, &bit_shuf, sub_num_wires, env, dbs, &mut stmt, &mut stmt2, conn);
        // COMPRESS_TIME.fetch_add(t4.elapsed().as_nanos() as u64, Ordering::Relaxed);

        subcircuit = subcircuit_temp;

        // let t5 = Instant::now();
        subcircuit = CircuitSeq::unrewire_subcircuit(&subcircuit, &used_wires);
        // UNREWIRE_TIME.fetch_add(t5.elapsed().as_nanos() as u64, Ordering::Relaxed);

        // let t6 = Instant::now();
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
        // REPLACE_TIME.fetch_add(t6.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    // let t7 = Instant::now();
    let mut i = 0;
    while i < circuit.gates.len().saturating_sub(1) {
        if circuit.gates[i] == circuit.gates[i + 1] {
            circuit.gates.drain(i..=i + 1);
            i = i.saturating_sub(2);
        } else {
            i += 1;
        }
    }
    // DEDUP_TIME.fetch_add(t7.elapsed().as_nanos() as u64, Ordering::Relaxed);

    circuit
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CollisionType {
    OnActive,
    OnCtrl1,
    OnCtrl2,
    OnNew,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GatePair {
    a: CollisionType,
    c1: CollisionType,
    c2: CollisionType
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GateTri {
    first: GatePair,
    second: GatePair,
    gap: GatePair,
}

impl GatePair {
    pub fn new() -> Self {
        GatePair { a: CollisionType::OnNew, c1: CollisionType::OnNew, c2: CollisionType::OnNew }
    }

    pub fn is_none(gate_pair: &Self) -> bool {
        gate_pair.a == CollisionType::OnNew && gate_pair.c1 == CollisionType::OnNew && gate_pair.c2 == CollisionType::OnNew
    }

    pub fn to_int(gp: &Self) -> usize {
        let a = gp.a;
        let b = gp.c1;
        let c = gp.c2;

        if a == CollisionType::OnNew && b == CollisionType::OnNew && c == CollisionType::OnNew {
            0
        } else if a == CollisionType::OnActive && b == CollisionType::OnNew && c == CollisionType::OnNew {
            1
        } else if a == CollisionType::OnCtrl1 && b == CollisionType::OnNew && c == CollisionType::OnNew {
            2
        } else if a == CollisionType::OnCtrl2 && b == CollisionType::OnNew && c == CollisionType::OnNew {
            3
        } else if a == CollisionType::OnNew && b == CollisionType::OnActive && c == CollisionType::OnNew {
            4
        } else if a == CollisionType::OnNew && b == CollisionType::OnCtrl1 && c == CollisionType::OnNew {
            5
        } else if a == CollisionType::OnNew && b == CollisionType::OnCtrl2 && c == CollisionType::OnNew {
            6
        } else if a == CollisionType::OnNew && b == CollisionType::OnNew && c == CollisionType::OnActive {
            7
        } else if a == CollisionType::OnNew && b == CollisionType::OnNew && c == CollisionType::OnCtrl1 {
            8
        } else if a == CollisionType::OnNew && b == CollisionType::OnNew && c == CollisionType::OnCtrl2 {
            9
        } else if a == CollisionType::OnActive && b == CollisionType::OnCtrl1 && c == CollisionType::OnNew {
            10
        } else if a == CollisionType::OnActive && b == CollisionType::OnCtrl2 && c == CollisionType::OnNew {
            11
        } else if a == CollisionType::OnActive && b == CollisionType::OnNew && c == CollisionType::OnCtrl1 {
            12
        } else if a == CollisionType::OnActive && b == CollisionType::OnNew && c == CollisionType::OnCtrl2 {
            13
        } else if a == CollisionType::OnActive && b == CollisionType::OnCtrl1 && c == CollisionType::OnCtrl2 {
            14
        } else if a == CollisionType::OnActive && b == CollisionType::OnCtrl2 && c == CollisionType::OnCtrl1 {
            15
        } else if a == CollisionType::OnCtrl1 && b == CollisionType::OnActive && c == CollisionType::OnNew {
            16
        } else if a == CollisionType::OnCtrl1 && b == CollisionType::OnCtrl2 && c == CollisionType::OnNew {
            17
        } else if a == CollisionType::OnCtrl1 && b == CollisionType::OnNew && c == CollisionType::OnActive {
            18
        } else if a == CollisionType::OnCtrl1 && b == CollisionType::OnNew && c == CollisionType::OnCtrl2 {
            19
        } else if a == CollisionType::OnCtrl1 && b == CollisionType::OnActive && c == CollisionType::OnCtrl2 {
            20
        } else if a == CollisionType::OnCtrl1 && b == CollisionType::OnCtrl2 && c == CollisionType::OnActive {
            21
        } else if a == CollisionType::OnCtrl2 && b == CollisionType::OnActive && c == CollisionType::OnNew {
            22
        } else if a == CollisionType::OnCtrl2 && b == CollisionType::OnCtrl1 && c == CollisionType::OnNew {
            23
        } else if a == CollisionType::OnCtrl2 && b == CollisionType::OnNew && c == CollisionType::OnActive {
            24
        } else if a == CollisionType::OnCtrl2 && b == CollisionType::OnNew && c == CollisionType::OnCtrl1 {
            25
        } else if a == CollisionType::OnCtrl2 && b == CollisionType::OnActive && c == CollisionType::OnCtrl1 {
            26
        } else if a == CollisionType::OnCtrl2 && b == CollisionType::OnCtrl1 && c == CollisionType::OnActive {
            27
        } else if a == CollisionType::OnNew && b == CollisionType::OnActive && c == CollisionType::OnCtrl1 {
            28
        } else if a == CollisionType::OnNew && b == CollisionType::OnActive && c == CollisionType::OnCtrl2 {
            29
        } else if a == CollisionType::OnNew && b == CollisionType::OnCtrl1 && c == CollisionType::OnActive {
            30
        } else if a == CollisionType::OnNew && b == CollisionType::OnCtrl1 && c == CollisionType::OnCtrl2 {
            31
        } else if a == CollisionType::OnNew && b == CollisionType::OnCtrl2 && c == CollisionType::OnActive {
            32
        } else if a == CollisionType::OnNew && b == CollisionType::OnCtrl2 && c == CollisionType::OnCtrl1 {
            33
        } else {
            panic!("Not a valid GatePair");
        }
    }

    pub fn from_int(i: usize) -> Self {
        use CollisionType::*;

        match i {
            0 => GatePair { a: OnNew, c1: OnNew, c2: OnNew },
            1 => GatePair { a: OnActive, c1: OnNew, c2: OnNew },
            2 => GatePair { a: OnCtrl1,  c1: OnNew, c2: OnNew },
            3 => GatePair { a: OnCtrl2,  c1: OnNew, c2: OnNew },
            4 => GatePair { a: OnNew, c1: OnActive, c2: OnNew },
            5 => GatePair { a: OnNew, c1: OnCtrl1, c2: OnNew },
            6 => GatePair { a: OnNew, c1: OnCtrl2, c2: OnNew },
            7 => GatePair { a: OnNew, c1: OnNew, c2: OnActive },
            8 => GatePair { a: OnNew, c1: OnNew, c2: OnCtrl1 },
            9 => GatePair { a: OnNew, c1: OnNew, c2: OnCtrl2 },
            10 => GatePair { a: OnActive, c1: OnCtrl1, c2: OnNew },
            11 => GatePair { a: OnActive, c1: OnCtrl2, c2: OnNew },
            12 => GatePair { a: OnActive, c1: OnNew, c2: OnCtrl1 },
            13 => GatePair { a: OnActive, c1: OnNew, c2: OnCtrl2 },
            14 => GatePair { a: OnActive, c1: OnCtrl1, c2: OnCtrl2 },
            15 => GatePair { a: OnActive, c1: OnCtrl2, c2: OnCtrl1 },
            16 => GatePair { a: OnCtrl1, c1: OnActive, c2: OnNew },
            17 => GatePair { a: OnCtrl1, c1: OnCtrl2, c2: OnNew },
            18 => GatePair { a: OnCtrl1, c1: OnNew, c2: OnActive },
            19 => GatePair { a: OnCtrl1, c1: OnNew, c2: OnCtrl2 },
            20 => GatePair { a: OnCtrl1, c1: OnActive, c2: OnCtrl2 },
            21 => GatePair { a: OnCtrl1, c1: OnCtrl2, c2: OnActive },
            22 => GatePair { a: OnCtrl2, c1: OnActive, c2: OnNew },
            23 => GatePair { a: OnCtrl2, c1: OnCtrl1, c2: OnNew },
            24 => GatePair { a: OnCtrl2, c1: OnNew, c2: OnActive },
            25 => GatePair { a: OnCtrl2, c1: OnNew, c2: OnCtrl1 },
            26 => GatePair { a: OnCtrl2, c1: OnActive, c2: OnCtrl1 },
            27 => GatePair { a: OnCtrl2, c1: OnCtrl1, c2: OnActive },
            28 => GatePair { a: OnNew, c1: OnActive, c2: OnCtrl1 },
            29 => GatePair { a: OnNew, c1: OnActive, c2: OnCtrl2 },
            30 => GatePair { a: OnNew, c1: OnCtrl1, c2: OnActive },
            31 => GatePair { a: OnNew, c1: OnCtrl1, c2: OnCtrl2 },
            32 => GatePair { a: OnNew, c1: OnCtrl2, c2: OnActive },
            33 => GatePair { a: OnNew, c1: OnCtrl2, c2: OnCtrl1 },

            _ => panic!("Invalid GatePair index"),
        }
    }
}

pub fn get_collision_type(g1: &[u8; 3], pin: u8) -> CollisionType {
    match pin {
        x if x == g1[0] => CollisionType::OnActive,
        x if x == g1[1] => CollisionType::OnCtrl1,
        x if x == g1[2] => CollisionType::OnCtrl2,
        _ => CollisionType::OnNew,
    }
}

pub fn gate_pair_taxonomy(g1: &[u8;3], g2: &[u8;3]) -> GatePair {
    GatePair {
        a: get_collision_type(&g1, g2[0]),
        c1: get_collision_type(&g1, g2[1]),
        c2: get_collision_type(&g1, g2[2]),
    }
}

fn gate_tri_taxonomy(g0: &[u8;3], g1: &[u8;3], g2: &[u8;3]) -> GateTri {
    GateTri {
        first: gate_pair_taxonomy(g0, g1),
        second: gate_pair_taxonomy(g1, g2),
        gap: gate_pair_taxonomy(g0, g2)
    }
}

pub fn replace_pairs(circuit: &mut CircuitSeq, num_wires: usize, conn: &mut Connection, env: &lmdb::Environment) {
    println!("Starting replace_pairs, circuit length: {}", circuit.gates.len());
    // let start = circuit.clone();
    let mut pairs: HashMap<GatePair, Vec<usize>> = HashMap::new();
    let gates = circuit.gates.clone();
    let m = circuit.gates.len();
    let mut replaced = 0;
    let mut to_replace: Vec<(Vec<[u8;3]>, Vec<[u8;3]>)> = vec![(Vec::new(), Vec::new()); m / 2];
    if m < 2 {
        println!("Circuit too small, returning");
        return;
    }

    println!("Building taxonomy pairs...");
    let mut i = 0;
    while i + 1 < m {
        let g1 = gates[i];
        let g2 = gates[i + 1];
        let taxonomy = gate_pair_taxonomy(&g1, &g2);

        if !GatePair::is_none(&taxonomy) {
            pairs.entry(taxonomy)
                .or_default()
                .push(i);
        }
        i += 2;
    }
    let num_pairs: usize = pairs.values().map(|v| v.len()).sum();
    println!("Pairs collected: {}", num_pairs);
    
    let mut rng = rand::rng();
    let mut fail = 0;
    while !pairs.is_empty() && fail < 100 {
        let n = rng.random_range(5..=7);
        let mut id = match random_canonical_id(&env, conn, n) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let mut replaced = false;

        // Forward scan: every adjacent pair
        for i in 0..id.gates.len() - 1 {
            let tax = gate_pair_taxonomy(&id.gates[i], &id.gates[i + 1]);
            if let Some(v) = pairs.get_mut(&tax) {
                if !v.is_empty() {
                    let idx = fastrand::usize(..v.len());
                    let chosen = v.swap_remove(idx);

                    // Remove the matching pair and reconstruct
                    let mut new_circuit = Vec::with_capacity(id.gates.len());
                    // Append gates after the pair
                    new_circuit.extend_from_slice(&id.gates[i + 2..]);
                    // Append gates before the pair
                    new_circuit.extend(id.gates[0..i].iter());
                    // let nc = CircuitSeq { gates: new_circuit.clone() };
                    // if nc.probably_equal(&CircuitSeq { gates: vec![id.gates[i+1], id.gates[i]]}, num_wires, 10000).is_err() { panic!("pairs dont match new"); }
                    to_replace[chosen / 2] = (new_circuit, vec![id.gates[i], id.gates[i+1]]);

                    if v.is_empty() {
                        pairs.remove(&tax);
                    }

                    replaced = true;
                    break; // stop scanning once a match is found
                }
            }
        }

        if replaced {
            continue;
        }

        // Reverse scan: every adjacent pair in reverse
        id.gates.reverse();
        for i in 0..id.gates.len() - 1 {
            let tax = gate_pair_taxonomy(&id.gates[i], &id.gates[i + 1]);
            if let Some(v) = pairs.get_mut(&tax) {
                if !v.is_empty() {
                    let idx = fastrand::usize(..v.len());
                    let chosen = v.swap_remove(idx);

                    // Remove the matching pair and reconstruct
                    let mut new_circuit = Vec::with_capacity(id.gates.len());
                    // Append gates after the pair
                    new_circuit.extend_from_slice(&id.gates[i + 2..]);
                    // Append gates before the pair, in reverse
                    new_circuit.extend(id.gates[0..i].iter());
                    // let nc = CircuitSeq { gates: new_circuit.clone() };
                    // if nc.probably_equal(&CircuitSeq { gates: vec![id.gates[i+1], id.gates[i]]}, num_wires, 10000).is_err() { panic!("reverse pairs dont match new"); }
                    to_replace[chosen / 2] = (new_circuit, vec![id.gates[i], id.gates[i+1]]);
                    
                    if v.is_empty() {
                        pairs.remove(&tax);
                    }

                    replaced = true;
                    break; // stop scanning once a match is found
                }
            }
        }

        if !replaced {
            fail += 1;
        }
    }

    println!("Applying replacements...");
    for (i, replacement) in to_replace.into_iter().enumerate().rev() {
        if replacement.0.is_empty() {
            continue;
        }

        // println!("Replacing at pair index {}", i);
        replaced += 1;
        let index = 2 * i;
        let (g1, g2) = (circuit.gates[index], circuit.gates[index + 1]);
        let replacement_circ = CircuitSeq { gates: replacement.0 };
        let mut used_wires: Vec<u8> = vec![(num_wires + 1) as u8; max(replacement_circ.max_wire(), CircuitSeq { gates: replacement.1.clone() }.max_wire()) + 1];

        used_wires[replacement.1[0][0] as usize] = g1[0];
        used_wires[replacement.1[0][1] as usize] = g1[1];
        used_wires[replacement.1[0][2] as usize] = g1[2];

        // println!("Original wires: {:?}, used_wires initialized", used_wires);

        // println!("Gates g1: {:?} g2: {:?}", g1, g2);

        let tax = gate_pair_taxonomy(&g1, &g2);
        if tax.a == CollisionType::OnNew || tax.c1 == CollisionType::OnNew || tax.c2 == CollisionType::OnNew {
            // println!("Found OnNew collision, assigning new wires...");
        }

        // Assign new wires if OnNew
        let mut i = 0;
        for collision in &[tax.a, tax.c1, tax.c2] {
            if *collision == CollisionType::OnNew {
                used_wires[replacement.1[1][i] as usize] = g2[i]
            }
            i += 1;
        }

        // Fill any remaining placeholders
        for i in 0..used_wires.len() {
            if used_wires[i] == (num_wires + 1) as u8 {
                loop {
                    let wire = rng.random_range(0..num_wires) as u8;
                    if used_wires.contains(&wire) {
                        continue
                    }
                    used_wires[i] = wire;
                    break
                }
            }
        }

        // println!("Final used_wires for this replacement: {:?}", used_wires);

        // if replacement.probably_equal(&CircuitSeq { gates: vec![[1,2,3], [1,2,3]]}, 64, 100000).is_err() {
        //     panic!("Replacement is not an id");
        // }
        circuit.gates.splice(
            index..=index + 1,
            CircuitSeq::unrewire_subcircuit(&replacement_circ, &used_wires)
                .gates
                .into_iter()
                .rev(),
        );

        // println!("Replacement: {:?}", CircuitSeq::unrewire_subcircuit(&replacement, &used_wires));
        // println!("Replacement applied at indices {}..{}", index, index + 1);
        // println!("Replacements so far: {}/{}", replaced, num_pairs);
    }
    println!("Replaced {}/{} pairs", replaced, num_pairs);
    // println!("Starting single gate replacements");
    // random_gate_replacements(circuit, min((num_pairs - replaced)/20 + (m/2 - num_pairs)/20, 1000), num_wires, conn, env);
    // if start.probably_equal(&circuit, num_wires, 10000).is_err() {
    //     panic!("replace pairs changed something");
    // }
    println!("Finished replace_pairs");
}

fn make_stdin_nonblocking() {
    let fd = io::stdin().as_raw_fd();
    unsafe {
        let flags = fcntl(fd, F_GETFL);
        fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    }
}

pub fn replace_sequential_pairs(
    circuit: &mut CircuitSeq,
    num_wires: usize,
    conn: &mut Connection,
    env: &lmdb::Environment,
    _bit_shuf_list: &Vec<Vec<Vec<usize>>>,
    dbs: &HashMap<String, lmdb::Database>
) -> (usize, usize, usize, usize) {
    make_stdin_nonblocking();
    let gates = circuit.gates.clone();
    let n = gates.len();
    if n < 2 {
        println!("Circuit too small, returning");
        return (0, 0, 0, 0);
    }

    let mut already_collided = 0;
    let mut shoot_count = 0;
    let mut curr_zero = 0;
    let mut traverse_left = 0;

    let mut rng = rand::rng();
    let mut out: Vec<[u8; 3]> = Vec::new();

    // rolling state
    let mut left = gates[0];
    let mut i = 1;
    let mut fail = 0;

    while i < n {
        let mut buf = [0u8; 1];
        if let Ok(n) = io::stdin().read(&mut buf) {
            if n > 0 && buf[0] == b'\n' {
                println!("  i = {}", i);
            }
        }
        let right = gates[i];
        let tax = gate_pair_taxonomy(&left, &right);

        // if !GatePair::is_none(&tax) {
            already_collided += 1;
            let mut produced: Option<Vec<[u8; 3]>> = None;

            while produced.is_none() && fail < 100 {
                fail += 1;
                let mut id_len = if GatePair::is_none(&tax) {
                    rng.random_range(6..=7)
                } else {
                    rng.random_range(5..=7)
                };
                if id_len == 8 {
                    id_len = 16;
                }
                let t_id = Instant::now();
                let id = match get_random_identity(id_len, tax, env, dbs) {
                    Ok(id) => {
                        IDENTITY_TIME.fetch_add(t_id.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        id
                    }
                    Err(_) => {
                        IDENTITY_TIME.fetch_add(t_id.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        fail += 1;
                        continue;
                    }
                };

                let new_circuit = id.gates[2..].to_vec();
                let replacement_circ = CircuitSeq { gates: new_circuit };

                let mut used_wires: Vec<u8> = vec![
                    (num_wires + 1) as u8;
                    std::cmp::max(
                        replacement_circ.max_wire(),
                        CircuitSeq {
                            gates: vec![id.gates[0], id.gates[1]],
                        }
                        .max_wire(),
                    ) + 1
                ];

                used_wires[id.gates[0][0] as usize] = left[0];
                used_wires[id.gates[0][1] as usize] = left[1];
                used_wires[id.gates[0][2] as usize] = left[2];

                let mut k = 0;
                for collision in &[tax.a, tax.c1, tax.c2] {
                    if *collision == CollisionType::OnNew {
                        used_wires[id.gates[1][k] as usize] = right[k];
                    }
                    k += 1;
                }

                let mut available_wires: Vec<u8> = (0..num_wires as u8)
                    .filter(|w| !used_wires.contains(w))
                    .collect();

                available_wires.shuffle(&mut rng);
                for w in 0..used_wires.len() {
                    if used_wires[w] == (num_wires + 1) as u8 {
                        if let Some(&wire) = available_wires.get(0) {
                            used_wires[w] = wire;
                            available_wires.remove(0);
                        } else {
                            panic!("No available wires left to assign!");
                        }
                    }
                }

                produced = Some(
                    CircuitSeq::unrewire_subcircuit(&replacement_circ, &used_wires)
                        .gates
                        .into_iter()
                        .rev()
                        .collect()
                );

                fail += 1;
            }

            if let Some(mut gates_out) = produced {
                out.append(&mut gates_out);
                left = out.pop().unwrap();
            } else {
                out.push(left);
                left = right;
            }

            fail = 0;
            i += 1;
        // } else {
        //     shoot_count += 1;
        //     out.push(gates[i]);
        //     let out_len = out.len();

        //     let new_index = shoot_left_vec(&mut out, out_len - 1);
        //     traverse_left += out_len - 1 - new_index;

        //     if new_index == 0 {
        //         curr_zero += 1;
        //         let g = &out[0];
        //         let temp_out_circ = CircuitSeq { gates: out.clone() };
        //         let num = rng.random_range(3..=7);

        //         if let Ok(mut id) = random_canonical_id(env, &conn, num) {
        //             let mut used_wires = vec![g[0], g[1], g[2]];
        //             let mut count = 3;

        //             while count < num {
        //                 let random = rng.random_range(0..num_wires);
        //                 if used_wires.contains(&(random as u8)) {
        //                     continue;
        //                 }
        //                 used_wires.push(random as u8);
        //                 count += 1;
        //             }
        //             used_wires.sort();

        //             let rewired_g =
        //                 CircuitSeq::rewire_subcircuit(&temp_out_circ, &vec![0], &used_wires);
        //             id.rewire_first_gate(rewired_g.gates[0], num);
        //             id = CircuitSeq::unrewire_subcircuit(&id, &used_wires);
        //             id.gates.remove(0);

        //             out.splice(0..1, id.gates);
        //         }

        //         fail = 0;
        //         i += 1;
        //         continue;
        //     }

        //     let left_gate = out[new_index - 1];
        //     let right_gate = out[new_index];
        //     let tax = gate_pair_taxonomy(&left_gate, &right_gate);

        //     if !GatePair::is_none(&tax) {
        //         let mut produced: Option<Vec<[u8; 3]>> = None;

        //         while produced.is_none() && fail < 100 {
        //             fail += 1;
        //             let id_len = rng.random_range(5..=7);

        //             let t_id = Instant::now();
        //             let id = match get_random_identity(id_len, tax, env, dbs) {
        //                 Ok(id) => {
        //                     IDENTITY_TIME.fetch_add(t_id.elapsed().as_nanos() as u64, Ordering::Relaxed);
        //                     id
        //                 }
        //                 Err(_) => {
        //                     IDENTITY_TIME.fetch_add(t_id.elapsed().as_nanos() as u64, Ordering::Relaxed);
        //                     fail += 1;
        //                     continue;
        //                 }
        //             };

        //             let new_circuit = id.gates[2..].to_vec();
        //             let replacement_circ = CircuitSeq { gates: new_circuit };

        //             let mut used_wires: Vec<u8> = vec![
        //                 (num_wires + 1) as u8;
        //                 std::cmp::max(
        //                     replacement_circ.max_wire(),
        //                     CircuitSeq {
        //                         gates: vec![id.gates[0], id.gates[1]],
        //                     }
        //                     .max_wire(),
        //                 ) + 1
        //             ];

        //             used_wires[id.gates[0][0] as usize] = left_gate[0];
        //             used_wires[id.gates[0][1] as usize] = left_gate[1];
        //             used_wires[id.gates[0][2] as usize] = left_gate[2];

        //             let mut k = 0;
        //             for collision in &[tax.a, tax.c1, tax.c2] {
        //                 if *collision == CollisionType::OnNew {
        //                     used_wires[id.gates[1][k] as usize] = right_gate[k];
        //                 }
        //                 k += 1;
        //             }

        //             let mut available_wires: Vec<u8> = (0..num_wires as u8)
        //                 .filter(|w| !used_wires.contains(w))
        //                 .collect();

        //             available_wires.shuffle(&mut rng);
        //             for w in 0..used_wires.len() {
        //                 if used_wires[w] == (num_wires + 1) as u8 {
        //                     if let Some(&wire) = available_wires.get(0) {
        //                         used_wires[w] = wire;
        //                         available_wires.remove(0);
        //                     } else {
        //                         panic!("No available wires left to assign!");
        //                     }
        //                 }
        //             }

        //             produced = Some(
        //                 CircuitSeq::unrewire_subcircuit(&replacement_circ, &used_wires)
        //                     .gates
        //                     .into_iter()
        //                     .rev()
        //                     .collect()
        //             );

        //             fail += 1;
        //         }

        //         if let Some(mut gates_out) = produced {
        //             out.splice((new_index - 1)..=new_index, gates_out.drain(..));
        //         }

        //         fail = 0;
        //         i += 1;
        //     }
        // }
    }

    out.push(left);
    circuit.gates = out;

    (already_collided, shoot_count, curr_zero, traverse_left)
}

// returns the id-2 and the length
pub fn replace_single_pair(
    left: &[u8;3],
    right: &[u8;3],
    num_wires: usize,
    _conn: &mut Connection,
    env: &lmdb::Environment,
    _bit_shuf_list: &Vec<Vec<Vec<usize>>>,
    dbs: &HashMap<String, lmdb::Database>
) -> (Vec<[u8;3]>, usize) {
    make_stdin_nonblocking();
    let mut rng = rand::rng();
    let tax = gate_pair_taxonomy(&left, &right);
    let mut id_gen = false;
    let mut id = CircuitSeq { gates: Vec::new() };
    while !id_gen {
        let id_len = rng.random_range(6..=7);
        // let id_len = 16;
        id = match get_random_identity(id_len, tax, env, dbs) {
            Ok(id) => {
                id_gen = true;
                id
            },
            Err(_) => {
                continue;
            }
        };
    }

    let new_circuit = id.gates[2..].to_vec();

    let replacement_circ = CircuitSeq { gates: new_circuit };

    let mut used_wires: Vec<u8> = vec![
        (num_wires + 1) as u8;
        std::cmp::max(
            replacement_circ.max_wire(),
            CircuitSeq {
                gates: vec![id.gates[0], id.gates[1]],
            }
            .max_wire(),
        ) + 1
    ];

    used_wires[id.gates[0][0] as usize] = left[0];
    used_wires[id.gates[0][1] as usize] = left[1];
    used_wires[id.gates[0][2] as usize] = left[2];

    let mut k = 0;
    for collision in &[tax.a, tax.c1, tax.c2] {
        if *collision == CollisionType::OnNew {
            used_wires[id.gates[1][k] as usize] = right[k];
        }
        k += 1;
    }

    let mut available_wires: Vec<u8> = (0..num_wires as u8)
        .filter(|w| !used_wires.contains(w))
        .collect();
    
    available_wires.shuffle(&mut rng);
    for w in 0..used_wires.len() {
        if used_wires[w] == (num_wires + 1) as u8 {
            if let Some(&wire) = available_wires.get(0) {
                used_wires[w] = wire;
                available_wires.remove(0);
            } else {
                panic!("No available wires left to assign!");
            }
        }
    }


    (CircuitSeq::unrewire_subcircuit(&replacement_circ, &used_wires)
        .gates
        .into_iter()
        .rev()
        .collect(),
    id.gates.len() - 2)
}

//TODO maybe parallelize this
// do smarter scans for updating the bounds by tracking as we go. also only update if left or right is replaced
// don't splice, just rebuild at the end
pub fn replace_pair_distances(
    circuit: &mut CircuitSeq,
    num_wires: usize,
    conn: &mut Connection,
    env: &lmdb::Environment,
    bit_shuf_list: &Vec<Vec<Vec<usize>>>,
    dbs: &HashMap<String, lmdb::Database>,
) {
    let min = 30;

    let mut distances = vec![0usize; circuit.gates.len() + 1];

    let (mut left, mut right) = update_bounds(&distances);

    let mut curr = 0;
    loop {
        // Termination condition
        if curr >= min {
            break;
        }
        let mut pending: Vec<(usize, usize, Vec<[u8; 3]>)> = Vec::new();

        // scan
        let mut i = left + 1;
        while i < right {
            let mut buf = [0u8; 1];
            if let Ok(n) = io::stdin().read(&mut buf) {
                if n > 0 && buf[0] == b'\n' {
                    println!("  curr = {}\n
                                gates = {}", curr, circuit.gates.len());
                }
            }
            if distances[i] == curr {
                let (id, id_len) = replace_single_pair(
                    &circuit.gates[i - 1],
                    &circuit.gates[i],
                    num_wires,
                    conn,
                    env,
                    bit_shuf_list,
                    dbs,
                );

                // Save what to do later
                if curr == 0 {
                    circuit.gates.splice(i - 1..=i, id);
                    update_distance(&mut distances, i, id_len);

                    let (l, r) = update_bounds(&distances);
                    left = l;
                    right = r;

                    continue;
                } else {
                    pending.push((i, id_len, id));
                }
            }
            i += 1;
        }

        // Nothing at this level, move up
        if pending.is_empty() {
            curr += 1;
            continue;
        }

        // replace
        pending.reverse();

        for (i, id_len, id) in pending {
            circuit.gates.splice(i - 1..=i, id);
            update_distance(&mut distances, i, id_len);
        }

        // Recompute bounds once after batch
        let (l, r) = update_bounds(&distances);
        curr += 1;
        left = l;
        right = r;
    }
}

fn update_bounds(distances: &[usize]) -> (usize, usize) {
    let mut left = 0;
    while left + 1 < distances.len()
        && distances[left + 1] == distances[left] + 1
    {
        left += 1;
    }

    let mut right = distances.len() - 1;
    while right > 0
        && distances[right - 1] == distances[right] + 1
    {
        right -= 1;
    }

    (left, right)
}

pub fn update_distance(
    distances: &mut Vec<usize>,
    didx: usize,
    id_len: usize,
) {
    let k = id_len - 1;

    let left0 = distances[didx - 1] + 1;
    let right0 = distances[didx + 1] + 1;

    let mut replacement = Vec::with_capacity(k);

    for i in 0..k {
        let from_left = left0 + i;
        let from_right = right0 + (k - 1 - i);
        replacement.push(from_left.min(from_right));
    }

    distances.splice(didx..=didx, replacement);
}

//TODO make rb smarter by making it a subtraction from total len rather than always updating
pub fn replace_pair_distances_linear(
    circuit: &mut CircuitSeq,
    num_wires: usize,
    conn: &mut Connection,
    env: &lmdb::Environment,
    bit_shuf_list: &Vec<Vec<Vec<usize>>>,
    dbs: &HashMap<String, lmdb::Database>,
    min: usize,
) {
    // initialize pair distances
    
    let mut gates = circuit.gates.drain(..).collect::<Vec<_>>();
    let mut dists = vec![0usize; gates.len() + 1];
    let mut lb = 1;
    let mut rb = dists.len() - 1;

    for curr in 0..min {
        println!("Working on curr = {}", curr);
        let mut out_gates = Vec::with_capacity(gates.len());
        let mut out_dists = Vec::with_capacity(gates.len() + 1);
        let mut temp_lb = lb;
        for i in 0..lb {
            out_gates.push(gates[i].clone());
            out_dists.push(dists[i]);
        }

        let mut i = lb;
        while i < gates.len() {
            let left = out_gates.last().unwrap();
            let right = &gates[i];
            let dist = dists[i];

            if dist == curr && i <= rb{
                let (id, id_len) = replace_single_pair(
                    left,
                    right,
                    num_wires,
                    conn,
                    env,
                    bit_shuf_list,
                    dbs,
                );

                if id_len > 0 {
                    // remove left gate
                    out_gates.pop();
                    let left_dist = out_dists.last().unwrap() + 1;
                    let right_dist = dists[i+1] + 1;
                    // emit replacement
                    for j in 0..id_len {
                        out_gates.push(id[j].clone());
                        if j != id_len - 1 {
                            let d = (left_dist + j).min(right_dist + id_len - 2 - j);
                            out_dists.push(d);
                        }
                    }
                    if i == lb {
                        while temp_lb + 1 < out_dists.len()
                            && out_dists[temp_lb + 1] == out_dists[temp_lb] + 1
                        {
                            temp_lb += 1;
                        }
                        temp_lb += 1;
                    } 
                    i += 1;
                    continue;
                }
            }

            // no replacement
            out_gates.push(right.clone());
            out_dists.push(dist);
            i += 1;
        }
        lb = temp_lb;
        rb = out_dists.len() - 1;
        while rb > 0
            && out_dists[rb - 1] == out_dists[rb] + 1
        {
            rb -= 1;
        }
        rb -= 1;
        // close tail distance
        out_dists.push(0);

        gates = out_gates;
        shoot_random_gate_gate_ver(&mut gates, 100_000);
        dists = out_dists;
    }
    // println!("{:?}", dists);
    // println!("left = {} right = {}", lb, rb);
    circuit.gates = gates;
}

pub fn replace_tri(
    circuit: &mut CircuitSeq,
    num_wires: usize,
    conn: &mut Connection,
    env: &lmdb::Environment,
) {
    println!("Starting replace_tri, circuit length: {}", circuit.gates.len());
    // let start = circuit.clone();
    let mut tris: HashMap<GateTri, Vec<usize>> = HashMap::new();
    let gates = circuit.gates.clone();
    let m = gates.len();
    let mut replaced = 0;

    let mut to_replace: Vec<(Vec<[u8;3]>, Vec<[u8;3]>)> = vec![(Vec::new(), Vec::new()); m / 3];
    if m < 3 {
        println!("Circuit too small, returning");
        return;
    }

    // Build taxonomy 
    println!("Building taxonomy triples...");
    let mut i = 0;
    while i + 2 < m {
        let g0 = gates[i];
        let g1 = gates[i + 1];
        let g2 = gates[i + 2];

        let taxonomy = gate_tri_taxonomy(&g0, &g1, &g2);
        tris.entry(taxonomy).or_default().push(i);

        i += 3;
    }

    let num_tris: usize = tris.values().map(|v| v.len()).sum();
    println!("Triples collected: {}", num_tris);

    let mut rng = rand::rng();
    let mut fail = 0;

    while !tris.is_empty() && fail < 100 {
        let n = rng.random_range(5..=7);
        let mut id = match random_canonical_id(&env, conn, n) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let mut replaced_here = false;

        // Forward 
        for i in 0..id.gates.len().saturating_sub(2) {
            let tax = gate_tri_taxonomy(
                &id.gates[i],
                &id.gates[i + 1],
                &id.gates[i + 2],
            );

            if let Some(v) = tris.get_mut(&tax) {
                if !v.is_empty() {
                    let idx = fastrand::usize(..v.len());
                    let chosen = v.swap_remove(idx);

                    // Remove triple and reconstruct
                    let mut new_circuit = Vec::with_capacity(id.gates.len());

                    // after the triple
                    new_circuit.extend_from_slice(&id.gates[i + 3..]);

                    // before the triple, reversed
                    new_circuit.extend(id.gates[0..i].iter());

                    to_replace[chosen / 3] = (new_circuit, vec![id.gates[i], id.gates[i+1], id.gates[i+2]]);

                    if v.is_empty() {
                        tris.remove(&tax);
                    }

                    replaced_here = true;
                    break;
                }
            }
        }

        if replaced_here {
            continue;
        }

        // Reverse
        id.gates.reverse();
        for i in 0..id.gates.len().saturating_sub(2) {
            let tax = gate_tri_taxonomy(
                &id.gates[i],
                &id.gates[i + 1],
                &id.gates[i + 2],
            );

            if let Some(v) = tris.get_mut(&tax) {
                if !v.is_empty() {
                    let idx = fastrand::usize(..v.len());
                    let chosen = v.swap_remove(idx);

                    // Remove triple and reconstruct
                    let mut new_circuit = Vec::with_capacity(id.gates.len());

                    // after the triple
                    new_circuit.extend_from_slice(&id.gates[i + 3..]);

                    // before the triple, reversed
                    new_circuit.extend(id.gates[0..i].iter());

                    to_replace[chosen / 3] = (new_circuit, vec![id.gates[i], id.gates[i+1], id.gates[i+2]]);

                    if v.is_empty() {
                        tris.remove(&tax);
                    }

                    replaced_here = true;
                    break;
                }
            }
        }

        if !replaced_here {
            fail += 1;
        }
    }

    // Apply replacements
    println!("Applying triple replacements...");
    for (i, replacement) in to_replace.into_iter().enumerate().rev() {
        if replacement.0.is_empty() {
            continue;
        }

        replaced += 1;
        let index = 3 * i;

        let g0 = circuit.gates[index];
        let g1 = circuit.gates[index + 1];
        let g2 = circuit.gates[index + 2];

        let replacement_circ = CircuitSeq { gates: replacement.0 };

        let mut used_wires =
            vec![(num_wires + 1) as u8; max(replacement_circ.max_wire(), CircuitSeq { gates: replacement.1.clone() }.max_wire()) + 1];


        used_wires[replacement.1[0][0] as usize] = g0[0];
        used_wires[replacement.1[0][1] as usize] = g0[1];
        used_wires[replacement.1[0][2] as usize] = g0[2];

        let tax = gate_tri_taxonomy(&g0, &g1, &g2);
        // Assign new wires if OnNew
        let mut i = 0;
        for collision in &[tax.first.a, tax.first.c1, tax.first.c2] {
            if *collision == CollisionType::OnNew {
                used_wires[replacement.1[1][i] as usize] = g1[i]
            }
            i += 1;
        }

        let mut i = 0;
        for collision in &[(tax.second.a == CollisionType::OnNew) && (tax.gap.a == CollisionType::OnNew), (tax.second.c1 == CollisionType::OnNew) && (tax.gap.c1 == CollisionType::OnNew), (tax.second.c2 == CollisionType::OnNew) && (tax.gap.c2 == CollisionType::OnNew)] {
            if *collision == true {
                used_wires[replacement.1[2][i] as usize] = g2[i]
            }
            i += 1;
        }

        // Fill any remaining placeholders
        for i in 0..used_wires.len() {
            if used_wires[i] == (num_wires + 1) as u8 {
                loop {
                    let wire = rng.random_range(0..num_wires) as u8;
                    if used_wires.contains(&wire) {
                        continue
                    }
                    used_wires[i] = wire;
                    break
                }
            }
        }

        circuit.gates.splice(
            index..=index + 2,
            CircuitSeq::unrewire_subcircuit(&replacement_circ, &used_wires)
                .gates
                .into_iter()
                .rev(),
        );
    }
    // if start.probably_equal(&circuit, num_wires, 10000).is_err() {
    //     panic!("replace tris changed something");
    // }
    println!("Replaced {}/{} triples", replaced, num_tris);
    println!("Finished replace_tri");
}

pub fn random_gate_replacements(c: &mut CircuitSeq, x: usize, n: usize, _conn: &Connection, env: &lmdb::Environment) {
    let mut rng = rand::rng();
    for _ in 0..x {
        if c.gates.is_empty() {
            break;
        }

        let i = rng.random_range(0..c.gates.len());
        let g = &c.gates[i];

        let num = rng.random_range(3..=7);
        if let Ok(mut id) = random_canonical_id(env, &_conn, num) {
            let mut used_wires = vec![g[0], g[1], g[2]];
            let mut count = 3;
            while count < num {
                let random = rng.random_range(0..n);
                if used_wires.contains(&(random as u8)) {
                    continue
                }
                used_wires.push(random as u8);
                count += 1;
            }
            used_wires.sort();
            let rewired_g = CircuitSeq::rewire_subcircuit(&c, &vec![i], &used_wires);
            // println!("rewired_g {:?} vs len: {}", rewired_g, num);
            id.rewire_first_gate(rewired_g.gates[0], num);
            id = CircuitSeq::unrewire_subcircuit(&id, &used_wires);
            id.gates.remove(0);
            c.gates.splice(i..i+1, id.gates);
        } 
    }
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
    let pick = PICK_SUBCIRCUIT_TIME.load(Ordering::Relaxed);
    let canonicalize = CANONICALIZE_TIME.load(Ordering::Relaxed);
    let row_fetch = ROW_FETCH_TIME.load(Ordering::Relaxed);
    let srow_fetch = SROW_FETCH_TIME.load(Ordering::Relaxed);
    let sixrow_fetch = SIXROW_FETCH_TIME.load(Ordering::Relaxed);
    let lrow_fetch = LROW_FETCH_TIME.load(Ordering::Relaxed);
    let db_open = DB_OPEN_TIME.load(Ordering::Relaxed);
    let txn = TXN_TIME.load(Ordering::Relaxed);
    let lmdb_lookup = LMDB_LOOKUP_TIME.load(Ordering::Relaxed);
    let from_blob = FROM_BLOB_TIME.load(Ordering::Relaxed);
    let splice = SPLICE_TIME.load(Ordering::Relaxed);
    let trial = TRIAL_TIME.load(Ordering::Relaxed);
    let id = IDENTITY_TIME.load(Ordering::Relaxed);

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
    println!("Pick subcircuit time: {:.2} min", pick as f64 / 60_000_000_000.0);
    println!("Subcircuit canonicalize time: {:.2} min", canonicalize as f64 / 60_000_000_000.0);
    println!("SQL row fetch time: {:.2} min", row_fetch as f64 / 60_000_000_000.0);
    println!("SQL n7m4 prepared row fetch time: {:.2} min", srow_fetch as f64 / 60_000_000_000.0);
    println!("SQL n6m5 prepared row fetch time: {:.2} min", sixrow_fetch as f64 / 60_000_000_000.0);
    println!("LMDB row fetch time: {:.2} min", lrow_fetch as f64 / 60_000_000_000.0);
    println!("LMDB DB open time: {:.2} min", db_open as f64 / 60_000_000_000.0);
    println!("LMDB transaction begin time: {:.2} min", txn as f64 / 60_000_000_000.0);
    println!("LMDB lookup time: {:.2} min", lmdb_lookup as f64 / 60_000_000_000.0);
    println!("CircuitSeq from_blob time: {:.2} min", from_blob as f64 / 60_000_000_000.0);
    println!("Gate splice time: {:.2} min", splice as f64 / 60_000_000_000.0);
    println!("Trial loop time: {:.2} min", trial as f64 / 60_000_000_000.0);
    println!("Identity Sampling Time: {:.2} min", id as f64 / 60_000_000_000.0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;
    use std::time::{Instant};
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
    use std::fs;
    use std::fs::File;
    use lmdb::Environment;
    use std::path::Path;
    use std::io::Write;
    use crate::replace::mixing::open_all_dbs;
    #[test]
    fn test_compression_big_time() {
        // let total_start = Instant::now();

        // // ---------- FIRST TEST ----------
        // let t1_start = Instant::now();
        // let n = 64;
        // let str1 = "circuitQQF_64.txt";
        // let data1 = fs::read_to_string(str1).expect("Failed to read circuitQQF_64.txt");
        // let mut stable_count = 0;
        // let mut conn = Connection::open("circuits.db").expect("Failed to open DB");
        // let mut acc = CircuitSeq::from_string(&data1);
        // while stable_count < 3 {
        //     let before = acc.gates.len();
        //     acc = compress_big(&acc, 1_000, n, &mut conn);
        //     let after = acc.gates.len();

        //     if after == before {
        //         stable_count += 1;
        //         println!("  Final compression stable {}/3 at {} gates", stable_count, after);
        //     } else {
        //         println!("  Final compression reduced: {} → {} gates", before, after);
        //         stable_count = 0;
        //     }
        // }
        // let t1_duration = t1_start.elapsed();
        // println!(" First compression finished in {:.2?}", t1_duration);

        // ---------- SECOND TEST ----------
        let t2_start = Instant::now();
        let str2 = "compressed.txt";
        let lmdb = "./db";
            let _ = std::fs::create_dir_all(lmdb);

            let env = Environment::new()
                .set_max_readers(10000) 
                .set_max_dbs(155)      
                .set_map_size(700 * 1024 * 1024 * 1024) 
                .open(Path::new(lmdb))
                .expect("Failed to open lmdb");

        let data2 = fs::read_to_string(str2).expect("Failed to read circuitF.txt");
        let mut stable_count = 0;
        let conn = Connection::open("circuits.db").expect("Failed to open DB");
        let acc = CircuitSeq::from_string(&data2);
        let _bit_shuf_list: Vec<Vec<Vec<usize>>> = (3..=7)
        .map(|n| {
            (0..n)
                .permutations(n)
                .filter(|p| !p.iter().enumerate().all(|(i, &x)| i == x))
                .collect::<Vec<Vec<usize>>>()
        })
        .collect();
        let _dbs = open_all_dbs(&env);
        let mut stmts_prepared = HashMap::new();
        let mut stmts_prepared_limit1 = HashMap::new();
        let ns_and_ms = vec![(3, 10), (4, 6), (5, 5), (6, 5), (7, 4)];
        for &(n, max_m) in &ns_and_ms {
            for m in 1..=max_m {
                let table = format!("n{}m{}", n, m);
                let query = format!("SELECT perm, shuf FROM {} WHERE circuit = ?", table);
                let stmt = conn.prepare(&query).unwrap();
                stmts_prepared.insert((n, m), stmt);

                let query_limit = format!("SELECT perm, shuf FROM {} WHERE circuit = ?1 LIMIT 1", table);
                let stmt_limit = conn.prepare(&query_limit).unwrap();
                stmts_prepared_limit1.insert((n, m), stmt_limit);
            }
        }
        let _conn = Connection::open("circuits.db").expect("Failed to open DB");
        while stable_count < 6 {
            let before = acc.gates.len();
            // acc = compress_big(&acc, 1_000, 64, &mut conn, &env, &bit_shuf_list, &dbs);
            let after = acc.gates.len();

            if after == before {
                stable_count += 1;
                println!("  Final compression stable {}/6 at {} gates", stable_count, after);
            } else {
                println!("  Final compression reduced: {} → {} gates", before, after);
                stable_count = 0;
            }
        }

        File::create("compressed.txt")
        .and_then(|mut f| f.write_all(acc.repr().as_bytes()))
        .expect("Failed to write butterfly_recent.txt");
        let t2_duration = t2_start.elapsed();
        println!(" Second compression finished in {:.2?}", t2_duration);

        // ---------- TOTAL ----------
        // let total_duration = total_start.elapsed();
        // println!(" Total test duration: {:.2?}", total_duration);
    }

    #[test]
    fn test_random_canon_id() {
        let env = Environment::new()
                .set_max_readers(10000) 
                .set_max_dbs(155)      
                .set_map_size(700 * 1024 * 1024 * 1024) 
                .open(Path::new("./db"))
                .expect("Failed to open lmdb");
        let conn = Connection::open("circuits.db").expect("Failed to open DB");
        let circuit = random_canonical_id(&env, &conn, 3).unwrap_or_else(|_| panic!("Failed to run random_canon_id"));
        if circuit.probably_equal(&CircuitSeq { gates: vec![[1,2,3], [1,2,3]]}, 10, 10000).is_err() {
            panic!("Not id");
        }
        println!("circuit {:?}", circuit.gates);
    }

    #[test]
    fn print_lmdb_keys() -> Result<(), Box<dyn std::error::Error>> {
        let env_path = "./db";
        let db_name = "perm_tables_n6";

        let env = Environment::new()
            .set_max_dbs(155)
            .open(Path::new(env_path))?;

        let db = env.open_db(Some(db_name))?;

        let txn = env.begin_ro_txn()?;
        let mut cursor = txn.open_ro_cursor(db)?;
        for (_, value) in cursor.iter() {
            println!("{:?}", value); 
        }

        Ok(())
    }

    #[test]
    fn test_find_perm_lmdb() {
        let perm = Permutation { data: vec![3, 2, 5, 4, 7, 6, 1, 0, 11, 10, 13, 12, 15, 14, 9, 8, 19, 18, 21, 20, 23, 22, 17, 16, 27, 26, 29, 28, 31, 30, 25, 24, 37, 36, 35, 34, 33, 32, 39, 38, 43, 42, 45, 44, 47, 46, 41, 40, 53, 52, 51, 50, 49, 48, 55, 54, 59, 58, 61, 60, 63, 62, 57, 56, 71, 70, 68, 69, 67, 66, 64, 65, 79, 78, 76, 77, 75, 74, 72, 73, 87, 86, 84, 85, 83, 82, 80, 81, 95, 94, 92, 93, 91, 90, 88, 89, 100, 101, 103, 102, 96, 97, 99, 98, 111, 110, 108, 109, 107, 106, 104, 105, 116, 117, 119, 118, 112, 113, 115, 114, 127, 126, 124, 125, 123, 122, 120, 121]};
        let prefix = perm.repr_blob();
        let env_path = "./db";
        let db_name = "n4m2";
        let env = Environment::new()
            .set_max_dbs(155)
            .open(Path::new(env_path)).expect("Failed to open db");
        let db = env.open_db(Some(&db_name))
                .unwrap_or_else(|e| panic!("LMDB DB '{}' failed to open: {:?}", db_name, e));
        let txn = env.begin_ro_txn()
                .unwrap_or_else(|e| panic!("Failed to begin RO txn on '{}': {:?}", "perm_db_name", e));
        let mut cursor = txn.open_ro_cursor(db).ok().expect("Failed to open cursor");
        let mut circuits = Vec::new();
        let mut count = 0;
        for (key, _) in cursor.iter() {
            if key.starts_with(&prefix) {
                circuits.push(key[prefix.len()..].to_vec());
                count += 1;
                println!("count: {}", count);
            }
        }
    }

    use crate::replace::mixing::split_into_random_chunks;
    use rayon::iter::IntoParallelIterator;
    use rayon::iter::ParallelIterator;
    use rusqlite::OpenFlags;
    #[test]
    fn replace_sequential_pair_preserves_invariants() {
        use rand::{SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(0xdeadbeef);
        let num_wires = 64;
        let env_path = "./db";
        let _conn = Connection::open("circuits.db").expect("Failed to open DB");
        let env = Environment::new()
            .set_max_dbs(155)
            .open(Path::new(env_path)).expect("Failed to open db");
        let data2 = fs::read_to_string("./tempcirc.txt").expect("Failed to read circuitF.txt");
        let mut circuit = CircuitSeq::from_string(&data2);
        let out_circ = circuit.clone();
        let bit_shuf_list = (3..=7)
        .map(|n| {
            (0..n)
                .permutations(n)
                .filter(|p| !p.iter().enumerate().all(|(i, &x)| i == x))
                .collect::<Vec<Vec<usize>>>()
        })
        .collect();
        let dbs = open_all_dbs(&env);
        let chunks = split_into_random_chunks(&circuit.gates, 10, &mut rng);
        static TOTAL_TIME: AtomicU64 = AtomicU64::new(0);
        // Call under test
        let replaced_chunks: Vec<Vec<[u8;3]>> =
        chunks
            .into_par_iter()
            .map(|chunk| {
                let mut sub = CircuitSeq { gates: chunk };
                let mut thread_conn = Connection::open_with_flags(
                    "circuits.db",
                    OpenFlags::SQLITE_OPEN_READ_ONLY,
                )
                .expect("Failed to open read-only connection");
                let t0 = Instant::now();
                let (_, _, _, _) = replace_sequential_pairs(&mut sub, 64, &mut thread_conn, &env, &bit_shuf_list, &dbs);
                sub.gates.reverse();
                let (_, _, _, _) = replace_sequential_pairs(&mut sub, 64, &mut thread_conn, &env, &bit_shuf_list, &dbs);
                sub.gates.reverse();
                TOTAL_TIME.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                sub.gates
            })
            .collect();
            let new_gates: Vec<[u8;3]> = replaced_chunks.into_iter().flatten().collect();
            circuit.gates = new_gates;

        if circuit.probably_equal(&out_circ, num_wires, 100_000).is_err() {
            panic!("Functionality was changed");
        }

        let tt = TOTAL_TIME.load(Ordering::Relaxed);

        println!("Permutation computation time: {:.2} min", tt as f64 / 60_000_000_000.0);
        println!("All good");
        print_compress_timers();
        // No invalid wire indices
        for (i, gate) in circuit.gates.iter().enumerate() {
            for &w in gate {
                assert!(
                    (w as usize) < num_wires,
                    "gate {} contains wire {} >= num_wires {}",
                    i,
                    w,
                    num_wires
                );
            }
        }
    }

    #[test]
    fn test_update_dist() {
        let mut d = vec![0, 1, 1, 0];
        update_distance(&mut d, 1, 6);
        assert_eq!(d, vec![0, 1, 2, 3, 3, 2, 1, 0]);
    }

    #[test]
    fn test_gen_id_speeds() {
        // stress / invariant check
        let _n = 64;
        let w = 7;
        let env_path = "./db";
        
        let env = Environment::new()
            .set_max_dbs(155)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new(env_path))
            .expect("Failed to open lmdb");
        let dbs = open_all_dbs(&env);

        for _ in 0..10_000_000 {
            let c = random_circuit(64, 2);
            let tax = gate_pair_taxonomy(&c.gates[0], &c.gates[1]);
            let id = get_random_identity(w, tax, &env, &dbs);
            println!("{:?}", id.unwrap().gates);
        }
        let ns_to_min = |v: u64| v as f64 / (60.0 * 1_000_000_000.0);
        println!("\n=== get_random_identity timers ===");

        // println!("DB_NAME_TIME          : {:.6}", ns_to_min(DB_NAME_TIME.load(Ordering::Relaxed)));
        // println!("DB_LOOKUP_TIME        : {:.6}", ns_to_min(DB_LOOKUP_TIME.load(Ordering::Relaxed)));
        // println!("TXN_BEGIN_TIME        : {:.6}", ns_to_min(TXN_BEGIN_TIME.load(Ordering::Relaxed)));
        // println!("SERIALIZE_KEY_TIME    : {:.6}", ns_to_min(SERIALIZE_KEY_TIME.load(Ordering::Relaxed)));
        // println!("LMDB_GET_TIME         : {:.6}", ns_to_min(LMDB_GET_TIME.load(Ordering::Relaxed)));
        // println!("DESERIALIZE_LIST_TIME : {:.6}", ns_to_min(DESERIALIZE_LIST_TIME.load(Ordering::Relaxed)));
        // println!("RNG_CHOOSE_TIME       : {:.6}", ns_to_min(RNG_CHOOSE_TIME.load(Ordering::Relaxed)));
        println!("FROM_BLOB_TIME        : {:.6}", ns_to_min(FROM_BLOB_TIME.load(Ordering::Relaxed)));

        println!("=================================\n");
    }

    fn gen_mean(circuit: CircuitSeq, num_wires: usize) -> f64 {
        let circuit_one = circuit.clone();
        let circuit_two = circuit;

        let circuit_one_len = circuit_one.gates.len();
        let circuit_two_len = circuit_two.gates.len();

        let num_points = (circuit_one_len + 1) * (circuit_two_len + 1);
        let mut average = vec![0f64; num_points * 3];

        let mut rng = rand::rng();
        let num_inputs = 20;

        for i in 0..num_inputs {
            // if i % 10 == 0 {
            //     // println!("{}/{}", i, num_inputs);
            //     io::stdout().flush().unwrap();
            // }

            let input_bits: u128 = if num_wires < u128::BITS as usize {
                rng.random_range(0..(1u128 << num_wires))
            } else {
                rng.random_range(0..=u128::MAX)
            };

            let evolution_one = circuit_one.evaluate_evolution_128(input_bits);
            let evolution_two = circuit_two.evaluate_evolution_128(input_bits);

            for i1 in 0..=circuit_one_len {
                for i2 in 0..=circuit_two_len {
                    let diff = evolution_one[i1] ^ evolution_two[i2];
                    let hamming_dist = diff.count_ones() as f64;
                    let overlap = hamming_dist / num_wires as f64;

                    let index = i1 * (circuit_two_len + 1) + i2;
                    average[index * 3] = i1 as f64;
                    average[index * 3 + 1] = i2 as f64;
                    average[index * 3 + 2] += overlap / num_inputs as f64;
                }
            }
        }

        let mut sum = 0.0;
        for i in 0..num_points {
            sum += average[i * 3 + 2];
        }

        sum / num_points as f64
    }

    #[test]
    pub fn test_gen_id_16() {
        let env_path = "./db";
        let mut thread_conn = Connection::open_with_flags(
                    "circuits.db",
                    OpenFlags::SQLITE_OPEN_READ_ONLY,
                )
                .expect("Failed to open read-only connection");
        let env = Environment::new()
            .set_max_dbs(200)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new(env_path))
            .expect("Failed to open lmdb");
        let bit_shuf_list = (3..=7)
        .map(|n| {
            (0..n)
                .permutations(n)
                .filter(|p| !p.iter().enumerate().all(|(i, &x)| i == x))
                .collect::<Vec<Vec<usize>>>()
        })
        .collect();
        let dbs = open_all_dbs(&env);
        let mut count = 0;
        while count < 2 {
            let id = get_random_wide_identity(16, &env, &dbs, &mut thread_conn, &bit_shuf_list);

            assert!(
                id.probably_equal(&CircuitSeq { gates: Vec::new() }, 16, 100_000).is_ok(),
                "Not an identity"
            );

            if gen_mean(id.clone(), 16) < 0.33 {
                continue
            }

            // write repr() to file
            let mut file = File::create(format!("id_16{}.txt", count))
                .expect("Failed to create output file");
            writeln!(file, "{}", id.repr()).expect("Failed to write repr");

            // wire statistics
            let mut wires: HashMap<u8, Vec<usize>> = HashMap::new();
            for (i, gates) in id.gates.iter().enumerate() {
                for &pins in gates {
                    wires.entry(pins).or_insert_with(Vec::new).push(i);
                }
            }

            println!("Run {}", count);
            for (k, v) in &wires {
                println!("wire: {}, # of gates: {}", k, v.len());
            }
            println!("Num wires: {}\n", wires.len());
            count += 1;
        }
    }

    #[test]
    pub fn test_max_mean_16() {
        let env_path = "./db";
        let mut thread_conn = Connection::open_with_flags(
                    "circuits.db",
                    OpenFlags::SQLITE_OPEN_READ_ONLY,
                )
                .expect("Failed to open read-only connection");
        let env = Environment::new()
            .set_max_dbs(200)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new(env_path))
            .expect("Failed to open lmdb");
        let bit_shuf_list = (3..=7)
        .map(|n| {
            (0..n)
                .permutations(n)
                .filter(|p| !p.iter().enumerate().all(|(i, &x)| i == x))
                .collect::<Vec<Vec<usize>>>()
        })
        .collect();
        let dbs = open_all_dbs(&env);
        let mut curr_mean = 0.0;
        loop {
            let id = get_random_wide_identity(16, &env, &dbs, &mut thread_conn, &bit_shuf_list);

            assert!(
                id.probably_equal(&CircuitSeq { gates: Vec::new() }, 16, 100_000).is_ok(),
                "Not an identity"
            );
            let mean = gen_mean(id.clone(), 16);
            if mean < curr_mean {
                continue
            }
            curr_mean = mean;

            // write repr() to file
            let mut file = File::create(format!("id_16currmean.txt"))
                .expect("Failed to create output file");
            writeln!(file, "{}", id.repr()).expect("Failed to write repr");

            // wire statistics
            let mut wires: HashMap<u8, Vec<usize>> = HashMap::new();
            for (i, gates) in id.gates.iter().enumerate() {
                for &pins in gates {
                    wires.entry(pins).or_insert_with(Vec::new).push(i);
                }
            }
            for (k, v) in &wires {
                println!("wire: {}, # of gates: {}", k, v.len());
            }
            println!("Num wires: {}\n", wires.len());
            println!("Curr mean: {}", curr_mean);
        }
    }
}