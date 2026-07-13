// Sample the sharded replacement LMDB and histogram stored circuit lengths
// and their used-wire counts. Answers: what candidate gate counts does the DB
// actually offer (compression needs len < window, expansion needs len >
// window), and how wide are the stored circuits at each length.
//
// Usage: db_value_stats [db_path] [shards_to_scan] [entries_per_shard]

use lmdb::{Cursor, Environment, EnvironmentFlags, Transaction};
use local_mixing::circuit::circuit::CircuitSeq;
use std::collections::BTreeMap;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let db_path = args.get(1).map(String::as_str).unwrap_or("./db");
    let shards: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(4);
    let per_shard: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(50_000);

    let env = Environment::new()
        .set_flags(EnvironmentFlags::READ_ONLY | EnvironmentFlags::NO_LOCK)
        .set_max_dbs(556)
        .set_map_size(800 * 1024 * 1024 * 1024)
        .open(Path::new(db_path))
        .expect("open lmdb");

    // (gates) -> count over all stored circuits
    let mut len_hist: BTreeMap<usize, u64> = BTreeMap::new();
    // (gates, wires) -> count
    let mut len_wire_hist: BTreeMap<(usize, usize), u64> = BTreeMap::new();
    // values per key
    let mut per_key_hist: BTreeMap<usize, u64> = BTreeMap::new();
    // per key: (min_len, max_len) spread -> count of keys with both shorter+longer info
    let mut keys = 0u64;

    for s in 0..shards {
        let name = format!("{:02x}", s);
        let db = env.open_db(Some(name.as_str())).expect("open shard");
        let txn = env.begin_ro_txn().expect("txn");
        let mut cursor = txn.open_ro_cursor(db).expect("cursor");
        for (i, kv) in cursor.iter_start().enumerate() {
            if i >= per_shard {
                break;
            }
            let (_k, v): (&[u8], &[u8]) = kv;
            keys += 1;
            let mut pos = 0usize;
            let mut nvals = 0usize;
            while pos < v.len() {
                let len = v[pos] as usize;
                pos += 1;
                if pos + len > v.len() {
                    break;
                }
                let c = CircuitSeq::from_blob(&v[pos..pos + len]);
                pos += len;
                let g = c.gates.len();
                let w = c.used_wires().len();
                *len_hist.entry(g).or_insert(0) += 1;
                *len_wire_hist.entry((g, w)).or_insert(0) += 1;
                nvals += 1;
            }
            *per_key_hist.entry(nvals).or_insert(0) += 1;
        }
    }

    println!("keys_sampled: {keys}");
    println!("value_len_csv gates,count");
    for (g, c) in &len_hist {
        println!("value_len_csv {},{}", g, c);
    }
    println!("value_len_wire_csv gates,wires,count");
    for ((g, w), c) in &len_wire_hist {
        println!("value_len_wire_csv {},{},{}", g, w, c);
    }
    println!("values_per_key_csv nvals,count");
    for (nv, c) in &per_key_hist {
        println!("values_per_key_csv {},{}", nv, c);
    }
}
