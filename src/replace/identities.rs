use std::{
    collections::HashMap,
    marker::PhantomData,
    ptr,
    slice,
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

use libc::c_uint;

use itertools::Itertools;
use rand::{Rng, prelude::SliceRandom};
use rusqlite::Connection;

use lmdb::{Cursor, Database, RoCursor, RoTransaction, Transaction, Environment};

extern crate lmdb_sys;
use lmdb_sys as ffi;

use crate::{
    circuit::circuit::{CircuitSeq, Permutation},
    random::random_data::{random_circuit, shoot_random_gate},
    replace::{pairs::{CollisionType, GatePair, gate_pair_taxonomy}, 
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

    // Hardcoded max entries for all DBs for efficient sampling
    let max_entries: usize = match db_name.as_str() {
        // n5
        "ids_n5g1" => 94_299,
        "ids_n5g2" => 156_303,
        "ids_n5g3" => 147_753,
        "ids_n5g4" => 156_161,
        "ids_n5g5" => 84_379,
        "ids_n5g6" => 113_912,
        "ids_n5g7" => 147_555,
        "ids_n5g8" => 113_881,
        "ids_n5g9" => 87_686,
        "ids_n5g10" => 278_597,
        "ids_n5g11" => 149_911,
        "ids_n5g12" => 149_841,
        "ids_n5g13" => 320_414,
        "ids_n5g14" => 310_782,
        "ids_n5g15" => 127_157,
        "ids_n5g16" => 111_339,
        "ids_n5g17" => 483_761,
        "ids_n5g18" => 94_787,
        "ids_n5g19" => 259_996,
        "ids_n5g20" => 100_484,
        "ids_n5g21" => 144_562,
        "ids_n5g22" => 94_711,
        "ids_n5g23" => 263_761,
        "ids_n5g24" => 117_146,
        "ids_n5g25" => 409_201,
        "ids_n5g26" => 144_476,
        "ids_n5g27" => 99_096,
        "ids_n5g28" => 481_653,
        "ids_n5g29" => 259_213,
        "ids_n5g30" => 263_142,
        "ids_n5g31" => 135_299,
        "ids_n5g32" => 407_399,
        "ids_n5g33" => 131_095,

        // n6
        "ids_n6g0" => 236_247,
        "ids_n6g1" => 376_803,
        "ids_n6g2" => 647_475,
        "ids_n6g3" => 594_654,
        "ids_n6g4" => 646_971,
        "ids_n6g5" => 289_428,
        "ids_n6g6" => 415_580,
        "ids_n6g7" => 594_122,
        "ids_n6g8" => 415_473,
        "ids_n6g9" => 320_670,
        "ids_n6g10" => 713_466,
        "ids_n6g11" => 351_574,
        "ids_n6g12" => 351_419,
        "ids_n6g13" => 912_213,
        "ids_n6g14" => 908_774,
        "ids_n6g15" => 225_317,
        "ids_n6g16" => 307_521,
        "ids_n6g17" => 1_375_202,
        "ids_n6g18" => 217_410,
        "ids_n6g19" => 642_414,
        "ids_n6g20" => 203_223,
        "ids_n6g21" => 250_805,
        "ids_n6g22" => 217_371,
        "ids_n6g23" => 558_684,
        "ids_n6g24" => 296_885,
        "ids_n6g25" => 1_127_637,
        "ids_n6g26" => 250_660,
        "ids_n6g27" => 180_403,
        "ids_n6g28" => 1_367_226,
        "ids_n6g29" => 640_671,
        "ids_n6g30" => 557_453,
        "ids_n6g31" => 298_474,
        "ids_n6g32" => 1_120_390,
        "ids_n6g33" => 260_137,

        // n7
        "ids_n7g0" => 954,
        "ids_n7g1" => 2_989,
        "ids_n7g2" => 2_446,
        "ids_n7g3" => 4_289,
        "ids_n7g4" => 2_445,
        "ids_n7g5" => 897,
        "ids_n7g6" => 2_268,
        "ids_n7g7" => 4_311,
        "ids_n7g8" => 2_268,
        "ids_n7g9" => 2_266,
        "ids_n7g10" => 6_373,
        "ids_n7g11" => 1_612,
        "ids_n7g12" => 1_603,
        "ids_n7g13" => 7_808,
        "ids_n7g14" => 7_588,
        "ids_n7g15" => 1_159,
        "ids_n7g16" => 2_398,
        "ids_n7g17" => 12_279,
        "ids_n7g18" => 807,
        "ids_n7g19" => 4_450,
        "ids_n7g20" => 1_509,
        "ids_n7g21" => 949,
        "ids_n7g22" => 807,
        "ids_n7g23" => 4_639,
        "ids_n7g24" => 3_666,
        "ids_n7g25" => 17_757,
        "ids_n7g26" => 950,
        "ids_n7g27" => 1_973,
        "ids_n7g28" => 11_945,
        "ids_n7g29" => 4_407,
        "ids_n7g30" => 4_605,
        "ids_n7g31" => 2_436,
        "ids_n7g32" => 17_486,
        "ids_n7g33" => 1_369,

        // n16
        "ids_n16g0single" => 77_760,
        "ids_n16g1single" => 4_720,
        "ids_n16g2single" => 7_430,
        "ids_n16g3single" => 7_170,
        "ids_n16g4single" => 7_710,
        "ids_n16g5single" => 5_850,
        "ids_n16g6single" => 6_110,
        "ids_n16g7single" => 7_340,
        "ids_n16g8single" => 6_180,
        "ids_n16g9single" => 6_140,
        "ids_n16g10single" => 3_900,
        "ids_n16g11single" => 1_940,
        "ids_n16g12single" => 1_900,
        "ids_n16g13single" => 5_510,
        "ids_n16g14single" => 4_300,
        "ids_n16g15single" => 1_310,
        "ids_n16g16single" => 2_500,
        "ids_n16g17single" => 10_630,
        "ids_n16g18single" => 1_610,
        "ids_n16g19single" => 4_870,
        "ids_n16g20single" => 1_920,
        "ids_n16g21single" => 1_680,
        "ids_n16g22single" => 1_660,
        "ids_n16g23single" => 4_160,
        "ids_n16g24single" => 2_340,
        "ids_n16g25single" => 8_810,
        "ids_n16g26single" => 1_700,
        "ids_n16g27single" => 1_660,
        "ids_n16g28single" => 10_510,
        "ids_n16g29single" => 5_070,
        "ids_n16g30single" => 4_000,
        "ids_n16g31single" => 2_420,
        "ids_n16g32single" => 8_830,
        "ids_n16g33single" => 1_920,

        "ids_n16g0tower" => 358_020,
        "ids_n16g1tower" => 42_020,
        "ids_n16g2tower" => 69_370,
        "ids_n16g3tower" => 66_500,
        "ids_n16g4tower" => 68_320,
        "ids_n16g5tower" => 51_770,
        "ids_n16g6tower" => 54_420,
        "ids_n16g7tower" => 64_530,
        "ids_n16g8tower" => 54_140,
        "ids_n16g9tower" => 55_030,
        "ids_n16g10tower" => 32_430,
        "ids_n16g11tower" => 17_910,
        "ids_n16g12tower" => 18_240,
        "ids_n16g13tower" => 45_400,
        "ids_n16g14tower" => 35_400,
        "ids_n16g15tower" => 10_420,
        "ids_n16g16tower" => 21_430,
        "ids_n16g17tower" => 84_690,
        "ids_n16g18tower" => 14_900,
        "ids_n16g19tower" => 41_940,
        "ids_n16g20tower" => 13_560,
        "ids_n16g21tower" => 12_500,
        "ids_n16g22tower" => 14_950,
        "ids_n16g23tower" => 35_430,
        "ids_n16g24tower" => 21_310,
        "ids_n16g25tower" => 70_120,
        "ids_n16g26tower" => 12_540,
        "ids_n16g27tower" => 12_450,
        "ids_n16g28tower" => 87_780,
        "ids_n16g29tower" => 41_860,
        "ids_n16g30tower" => 34_720,
        "ids_n16g31tower" => 22_690,
        "ids_n16g32tower" => 70_930,
        "ids_n16g33tower" => 18_780,
        _ => panic!("DB {} not in hardcoded max_entries", db_name),
    };

    let mut rng = rand::rng();
    let random_index = rng.random_range(0..max_entries);

    let txn = env.begin_ro_txn()?;
    let mut cursor = txn.open_ro_cursor(*db)?;

    let value_bytes = if n != 128 {
        cursor.iter_start()
        .nth(random_index)
        .map(|(k, _v)| k)
        .unwrap_or_else(|| {
            panic!(
                "Failed to get random key | db={} index={} n={}",
                db_name,
                random_index,
                n
            )
        })
    } else {
        cursor.iter_start()
        .nth(random_index)
        .map(|(_k, v)| v)
        .expect("Failed to get random val")
    };
    let out = CircuitSeq::from_blob(value_bytes);

    GET_ID_TOTAL_TIME.fetch_add(
        total_start.elapsed().as_nanos() as u64,
        Ordering::Relaxed,
    );

    Ok(out)
}

// Generate identities via shuffling
pub fn get_random_shuffled_identity (
    n: usize, 
    env: &lmdb::Environment,
    dbs: &HashMap<String, Database>,
    _conn: &mut Connection,
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
    _conn: &mut Connection,
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
    _conn: &mut Connection,
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
pub fn create_escalator_identities(
    n: usize, 
    first_steps: &Vec<Vec<usize>>, 
    second_steps: &Vec<Vec<usize>>
) -> (Transpositions, Transpositions, Transpositions, usize) {
    let mut allowed_wires: Vec<usize> = Vec::new();
    let all_wires: Vec<usize> = (0..n).collect();
    let mut first = Transpositions { transpositions: Vec::new() };
    let mut middle = Transpositions { transpositions: Vec::new() };
    let mut second = Transpositions { transpositions: Vec::new() };
    let m = 10;
    let mut negation_mask: Vec<u8> = vec![0u8; n];

    // Build second from the `top` to bottom
    allowed_wires.clear();
    for step in second_steps.iter().rev() {
        for wire in step {
            allowed_wires.push(*wire);
        }
        let restricted: Vec<usize> = all_wires.iter()
            .filter(|w| !allowed_wires.contains(w))
            .cloned()
            .collect();
        second.transpositions.extend_from_slice(
            &Transpositions::gen_random_simple_restricted(
                n, 
                m, 
                &mut negation_mask, 
                &restricted)
            .transpositions);
    }

    // building first
    for step in first_steps {
        for wire in step {
            allowed_wires.push(*wire);
        }
        let restricted: Vec<usize> = all_wires.iter()
            .filter(|w| !allowed_wires.contains(w))
            .cloned()
            .collect();
        first.transpositions.extend_from_slice(
            &Transpositions::gen_random_simple_restricted(
                n, 
                m, 
                &mut negation_mask, 
                &restricted)
            .transpositions);
    }

    // Now build middle
    middle.transpositions.extend_from_slice(&second.transpositions);
    middle.transpositions.extend_from_slice(&first.transpositions);

    let perm = middle.to_perm(n);
    middle = Transpositions::from_perm(&perm);

    // Use negation mask to update middle
    let mut wire_transpositions: HashMap<u8, (usize, usize)> = HashMap::new();

    for (i, (a, b, _)) in middle.transpositions.iter().enumerate() {
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
                let curr_neg_type = middle.transpositions[swap_idx].2;
                if pos > 1 || curr_neg_type > 3 {
                    panic!("Invalid pos or curr_neg_type");
                }
                middle.transpositions[swap_idx].2 = TRANSITION[pos][curr_neg_type as usize];
                
            }
        }
    }

    (first, middle, second, m)
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
        let env = Environment::new()
            .set_max_dbs(262)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new("./db"))
            .expect("failed to open lmdb");

        let dbs = open_all_dbs(&env);

        let mut file = File::create("test_id.txt").expect("Failed to create file");
        let mut simple: Vec<Vec<usize>> = vec![vec![0,1,2], vec![3,4,5], vec![6,7,8], vec![9,10,11], vec![12,13,14], vec![15,16,17], vec![18,19,20], vec![21,22,23], vec![24,25,26], vec![27,28,29], vec![30,31]];
        simple.reverse();
        let (first, middle, second, _) = create_escalator_identities(
            32,
            &simple,
            &simple,

        );
        let f = first.to_circuit(32, &env, &dbs);
        let mut m = middle.to_circuit(32, &env, &dbs);
        m.gates.reverse();
        let s = second.to_circuit(32, &env, &dbs);

        let mut c = CircuitSeq {gates: Vec::new() };
        c.gates.extend_from_slice(&f.gates);
        c.gates.extend_from_slice(&m.gates);
        c.gates.extend_from_slice(&s.gates);
        let repr = c.repr();
        writeln!(file, "{}", repr)
            .expect("Failed to write to file");

        println!("Wrote test circuit to file");
    }
}
