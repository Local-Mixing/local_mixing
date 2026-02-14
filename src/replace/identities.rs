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

use lmdb::{Cursor, Database, RoCursor, RoTransaction, Transaction};

extern crate lmdb_sys;
use lmdb_sys as ffi;

use crate::{
    circuit::circuit::{CircuitSeq, Permutation},
    random::random_data::{random_circuit, shoot_random_gate},
    replace::pairs::{CollisionType, GatePair, gate_pair_taxonomy},
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
    let db_name = if n == 128 && tower {
        format!("ids_n{}g{}{}", n, g, "tower")
    } else if n == 128 && !tower {
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
        "ids_n5g1" => 100_235,
        "ids_n5g2" => 177_541,
        "ids_n5g3" => 169_347,
        "ids_n5g4" => 177_481,
        "ids_n5g5" => 88_913,
        "ids_n5g6" => 119_879,
        "ids_n5g7" => 169_161,
        "ids_n5g8" => 119_872,
        "ids_n5g9" => 90_262,
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
        "ids_n16g0"  => 36_070,
        "ids_n16g1"  => 5_140,
        "ids_n16g2"  => 11_660,
        "ids_n16g3"  => 11_050,
        "ids_n16g4"  => 12_000,
        "ids_n16g5"  => 5_460,
        "ids_n16g6"  => 5_700,
        "ids_n16g7"  => 11_250,
        "ids_n16g8"  => 5_590,
        "ids_n16g9"  => 5_470,
        "ids_n16g10" => 5_450,
        "ids_n16g11" => 2_720,
        "ids_n16g12" => 2_860,
        "ids_n16g13" => 7_430,
        "ids_n16g14" => 9_360,
        "ids_n16g15" => 2_310,
        "ids_n16g16" => 3_960,
        "ids_n16g17" => 16_400,
        "ids_n16g18" => 2_420,
        "ids_n16g19" => 7_810,
        "ids_n16g20" => 3_290,
        "ids_n16g21" => 2_520,
        "ids_n16g22" => 2_570,
        "ids_n16g23" => 6_410,
        "ids_n16g24" => 3_750,
        "ids_n16g25" => 13_370,
        "ids_n16g26" => 2_580,
        "ids_n16g27" => 2_800,
        "ids_n16g28" => 16_260,
        "ids_n16g29" => 7_880,
        "ids_n16g30" => 6_420,
        "ids_n16g31" => 2_700,
        "ids_n16g32" => 13_290,
        "ids_n16g33" => 2_360,
        // n128 singles
        "ids_n128g0single"  => 11_110,
        "ids_n128g1single"  =>    300,
        "ids_n128g2single"  =>    650,
        "ids_n128g3single"  =>    660,
        "ids_n128g4single"  =>    710,
        "ids_n128g5single"  =>    290,
        "ids_n128g6single"  =>    320,
        "ids_n128g7single"  =>    690,
        "ids_n128g8single"  =>    330,
        "ids_n128g9single"  =>    350,
        "ids_n128g10single" =>    260,
        "ids_n128g11single" =>    100,
        "ids_n128g12single" =>    120,
        "ids_n128g13single" =>    320,
        "ids_n128g14single" =>    390,
        "ids_n128g15single" =>     80,
        "ids_n128g16single" =>    230,
        "ids_n128g17single" =>    860,
        "ids_n128g18single" =>    140,
        "ids_n128g19single" =>    380,
        "ids_n128g20single" =>    170,
        "ids_n128g21single" =>    120,
        "ids_n128g22single" =>    120,
        "ids_n128g23single" =>    330,
        "ids_n128g24single" =>    190,
        "ids_n128g25single" =>    730,
        "ids_n128g26single" =>    120,
        "ids_n128g27single" =>    130,
        "ids_n128g28single" =>    880,
        "ids_n128g29single" =>    410,
        "ids_n128g30single" =>    340,
        "ids_n128g31single" =>    120,
        "ids_n128g32single" =>    740,
        "ids_n128g33single" =>    100,
        // n128 towers
        "ids_n128g0tower"  => 11_530,
        "ids_n128g1tower"  =>  2_180,
        "ids_n128g2tower"  =>  4_660,
        "ids_n128g3tower"  =>  4_590,
        "ids_n128g4tower"  =>  4_620,
        "ids_n128g5tower"  =>  2_290,
        "ids_n128g6tower"  =>  2_470,
        "ids_n128g7tower"  =>  4_430,
        "ids_n128g8tower"  =>  2_360,
        "ids_n128g9tower"  =>  2_470,
        "ids_n128g10tower" =>  1_770,
        "ids_n128g11tower" =>  1_050,
        "ids_n128g12tower" =>    990,
        "ids_n128g13tower" =>  2_420,
        "ids_n128g14tower" =>  2_830,
        "ids_n128g15tower" =>    620,
        "ids_n128g16tower" =>  1_440,
        "ids_n128g17tower" =>  5_490,
        "ids_n128g18tower" =>    980,
        "ids_n128g19tower" =>  2_690,
        "ids_n128g20tower" =>    940,
        "ids_n128g21tower" =>    760,
        "ids_n128g22tower" =>  1_030,
        "ids_n128g23tower" =>  2_270,
        "ids_n128g24tower" =>  1_450,
        "ids_n128g25tower" =>  4_430,
        "ids_n128g26tower" =>    740,
        "ids_n128g27tower" =>    790,
        "ids_n128g28tower" =>  5_720,
        "ids_n128g29tower" =>  2_560,
        "ids_n128g30tower" =>  2_240,
        "ids_n128g31tower" =>  1_000,
        "ids_n128g32tower" =>  4_540,
        "ids_n128g33tower" =>    830,
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
        .expect("Failed to get random key")
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
    while nwires < n || len < 1000 {
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

