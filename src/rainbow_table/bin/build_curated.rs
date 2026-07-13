use lmdb::{Cursor, Environment, Transaction};
use local_mixing::circuit::circuit::{polys_repr_blob, CircuitSeq};
use rayon::prelude::*;
use rocksdb::{DB, MergeOperands, Options};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use xxhash_rust::xxh3::{xxh3_64, xxh3_128};

const MAX_CIRCUITS_PER_ENTRY: usize = 20;
// Cap on stored bytes per key in the final RocksDB value (~17 tails at 30 bytes).
const MAX_VALUE_BYTES: usize = 512;

fn append_merge(
    _key: &[u8],
    existing: Option<&[u8]>,
    operands: &MergeOperands,
) -> Option<Vec<u8>> {
    let mut result = existing.map_or_else(Vec::new, |v| v.to_vec());
    for op in operands {
        if result.len() + op.len() <= MAX_VALUE_BYTES {
            result.extend_from_slice(op);
        }
    }
    Some(result)
}

fn decode_circuits(value: &[u8]) -> Vec<CircuitSeq> {
    let mut circuits = Vec::new();
    let mut pos = 0;
    while pos < value.len() {
        let len = value[pos] as usize;
        pos += 1;
        if pos + len > value.len() { break; }
        circuits.push(CircuitSeq::from_blob(&value[pos..pos + len]));
        pos += len;
    }
    circuits
}

fn remove_adjacent_equal(gates: &mut Vec<[u16; 3]>) {
    let mut i = 0;
    while i + 1 < gates.len() {
        if gates[i] == gates[i + 1] {
            gates.drain(i..=i + 1);
            if i > 0 { i -= 1; }
        } else {
            i += 1;
        }
    }
}

fn encode_circuit(blob: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(1 + blob.len());
    v.push(blob.len() as u8);
    v.extend_from_slice(blob);
    v
}

fn map_wire(
    w: u16,
    used_map: &HashMap<u16, u16>,
    extra_map: &mut HashMap<u16, u16>,
    next_extra: &mut u16,
) -> u16 {
    if let Some(&db) = used_map.get(&w) {
        db
    } else {
        let next = *next_extra;
        *extra_map.entry(w).or_insert_with(|| { *next_extra += 1; next })
    }
}

fn process_shard(
    shard_idx: usize,
    src_db: lmdb::Database,
    env: &Environment,
    rdb: &DB,
) {
    eprintln!("  shard {:02x}: scanning...", shard_idx);
    let entries: Vec<Vec<u8>> = {
        let txn = env.begin_ro_txn().expect("ro txn");
        let mut cursor = txn.open_ro_cursor(src_db).expect("cursor");
        let result: Vec<Vec<u8>> = cursor
            .iter()
            .filter_map(|(_, v)| {
                if decode_circuits(v).len() >= 2 { Some(v.to_vec()) } else { None }
            })
            .collect();
        drop(cursor);
        drop(txn);
        result
    };
    eprintln!("  shard {:02x}: {} qualifying entries, processing...", shard_idx, entries.len());

    // Per-shard local accumulation. Merging into RocksDB per-circuit makes short
    // prefixes into hot keys (few distinct functions, hammered by all 256 threads),
    // which builds O(n^2) merge-operand chains and livelocks compaction. Instead keep,
    // per key, the deduped + length-capped concatenation of encoded circuits, and
    // merge each key into RocksDB exactly once at the end of the shard.
    let mut buf: HashMap<Vec<u8>, (HashSet<u64>, Vec<u8>)> = HashMap::new();
    let mut merged = 0usize;

    for (entry_idx, value) in entries.iter().enumerate() {
        if entry_idx > 0 && entry_idx % 5000 == 0 {
            eprintln!("  shard {:02x}: {}/{} entries, {} merged so far",
                shard_idx, entry_idx, entries.len(), merged);
        }

        let circuits = decode_circuits(value);
        let circuits = if circuits.len() > MAX_CIRCUITS_PER_ENTRY {
            &circuits[..MAX_CIRCUITS_PER_ENTRY]
        } else {
            &circuits[..]
        };

        for i in 0..circuits.len() {
            for j in 0..circuits.len() {
                if i == j { continue; }
                let a = &circuits[i];
                let b = &circuits[j];

                for combo in 0..2usize {
                    let gates = if combo == 0 {
                        let mut g = a.gates.clone();
                        let mut b_rev = b.gates.clone();
                        b_rev.reverse();
                        g.extend(b_rev);
                        g
                    } else {
                        let mut a_rev = a.gates.clone();
                        a_rev.reverse();
                        let mut g = a_rev;
                        g.extend(b.gates.clone());
                        g
                    };

                    let mut identity = CircuitSeq { gates };
                    identity.canonicalize();
                    remove_adjacent_equal(&mut identity.gates);

                    let n = identity.gates.len();
                    if n < 3 { continue; }

                    for direction in [false, true] {
                        let directed: Vec<[u16; 3]> = if direction {
                            identity.gates.iter().rev().cloned().collect()
                        } else {
                            identity.gates.clone()
                        };

                        for rot in 0..n {
                            let rotation: Vec<[u16; 3]> = directed[rot..]
                                .iter()
                                .chain(directed[..rot].iter())
                                .cloned()
                                .collect();

                            for k in 1..n {
                                // Only store when suffix >= prefix (expansion only).
                                let suffix_len = n - k;
                                if k > suffix_len { continue; }

                                let prefix = CircuitSeq { gates: rotation[..k].to_vec() };
                                let (canon_polys, perm4, used) =
                                    prefix.canonicalize_polys_single(false);
                                if canon_polys.is_empty() { continue; }

                                let key = xxh3_128(&polys_repr_blob(&canon_polys))
                                    .to_le_bytes()
                                    .to_vec();

                                let perm4_inv = perm4.invert();
                                // All prefix wires are in used, so used_map covers them.
                                let used_map: HashMap<u16, u16> = used.iter().enumerate()
                                    .map(|(i, &w)| (w, perm4_inv.data[i] as u16))
                                    .collect();
                                let mut extra_map: HashMap<u16, u16> = HashMap::new();
                                let mut next_extra = used.len() as u16;

                                // Rewire prefix gates to DB wire space.
                                // Prefix wires are always in used_map (no extras).
                                let prefix_db_gates: Vec<[u16; 3]> = rotation[..k].iter()
                                    .map(|&[t, c1, c2]| [
                                        used_map[&t],
                                        used_map[&c1],
                                        used_map[&c2],
                                    ])
                                    .collect();
                                let mut prefix_db_seq = CircuitSeq { gates: prefix_db_gates };
                                prefix_db_seq.canonicalize();

                                // Rewire tail gates (reversed suffix) to DB wire space.
                                let mut tail_gates: Vec<[u16; 3]> = Vec::new();
                                for &[t, c1, c2] in rotation[k..].iter().rev() {
                                    tail_gates.push([
                                        map_wire(t, &used_map, &mut extra_map, &mut next_extra),
                                        map_wire(c1, &used_map, &mut extra_map, &mut next_extra),
                                        map_wire(c2, &used_map, &mut extra_map, &mut next_extra),
                                    ]);
                                }
                                let mut tail_seq = CircuitSeq { gates: tail_gates };
                                tail_seq.canonicalize();

                                // Under key = hash(prefix's canonical function), insert BOTH
                                // equivalents of that function as separate circuits:
                                //   - the rewired prefix            (length k)
                                //   - the reversed, rewired suffix  (length n-k)
                                // Both compute the prefix's function, so both canonicalize to `key`.
                                for seq in [&prefix_db_seq, &tail_seq] {
                                    let blob = seq.repr_blob();
                                    if blob.len() > 255 { continue; }
                                    let encoded = encode_circuit(&blob);
                                    // Accumulate locally: dedup per key, cap concatenated length.
                                    let h = xxh3_64(&encoded);
                                    let (seen_h, val) = buf.entry(key.clone()).or_default();
                                    if val.len() + encoded.len() <= MAX_VALUE_BYTES && seen_h.insert(h) {
                                        val.extend_from_slice(&encoded);
                                        merged += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Flush the shard's buffer to RocksDB: one merge per distinct key (no hot-key storm).
    for (key, (_seen, val)) in buf {
        rdb.merge(&key, &val).expect("rocksdb merge");
    }
    eprintln!("  shard {:02x}: done, {} unique circuits merged", shard_idx, merged);
}

fn main() {
    let env = Arc::new(
        Environment::new()
            .set_max_dbs(600)
            .set_max_readers(1024)
            .set_map_size(6 * 1024 * 1024 * 1024 * 1024)
            .open(Path::new("./db"))
            .expect("Failed to open ./db"),
    );

    println!("Opening source shard databases (00..ff)...");
    let src_dbs: Vec<lmdb::Database> = (0u16..256)
        .map(|s| {
            let name = format!("{:02x}", s);
            env.open_db(Some(name.as_str()))
                .unwrap_or_else(|e| panic!("Failed to open shard {}: {:?}", name, e))
        })
        .collect();

    println!("Opening RocksDB rocks_curated_db (clearing existing)...");
    let mut opts = Options::default();
    opts.create_if_missing(true);
    opts.set_merge_operator_associative("append_merge", append_merge);
    opts.increase_parallelism(8);
    let _ = DB::destroy(&opts, "rocks_curated_db");
    let rdb = Arc::new(DB::open(&opts, "rocks_curated_db").expect("open rocks_curated_db"));

    let done = Arc::new(AtomicUsize::new(0));
    let total_merged = Arc::new(AtomicUsize::new(0));

    println!("Processing 256 shards (parallel)...");

    (0..256usize).into_par_iter().for_each(|shard_idx| {
        let env = Arc::clone(&env);
        let rdb = Arc::clone(&rdb);
        let done = Arc::clone(&done);
        let total_merged = Arc::clone(&total_merged);

        process_shard(shard_idx, src_dbs[shard_idx], &env, &rdb);

        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
        if n <= 32 || n % 8 == 0 || n == 256 {
            println!("  {}/256 shards done", n);
        }
    });

    println!("Compacting RocksDB...");
    rdb.compact_range::<&[u8], &[u8]>(None, None);
    println!("Done. Run curated_to_lmdb to convert to LMDB curated_{{}} shards.");
}
