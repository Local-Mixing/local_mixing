/// For each shard in ./db, scan all hash keys.
/// Each value is a concatenated list of [len_byte, blob...] circuits.
/// Report: how many 7-gate and 8-gate circuits appear in entries with 2+ circuits ("have friends").
use lmdb::{Cursor, Environment, Transaction};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use rayon::prelude::*;

fn main() {
    let env = Arc::new(
        Environment::new()
            .set_max_dbs(300)
            .set_map_size(800 * 1024 * 1024 * 1024)
            .open(Path::new("./db"))
            .expect("open ./db"),
    );

    // Open all 256 shard databases
    let dbs: Vec<lmdb::Database> = (0u16..256)
        .map(|s| {
            let name = format!("{:02x}", s);
            env.open_db(Some(name.as_str()))
                .unwrap_or_else(|e| panic!("open shard {}: {:?}", name, e))
        })
        .collect();

    // Counters
    let gate_hist: Vec<Arc<AtomicU64>> = (0..=20).map(|_| Arc::new(AtomicU64::new(0))).collect();
    let gate_hist_multi: Vec<Arc<AtomicU64>> = (0..=20).map(|_| Arc::new(AtomicU64::new(0))).collect();
    let total_entries     = Arc::new(AtomicU64::new(0));
    let multi_entries     = Arc::new(AtomicU64::new(0)); // entries with 2+ circuits
    let friends_7         = Arc::new(AtomicU64::new(0)); // 7-gate circuits in multi entries
    let friends_8         = Arc::new(AtomicU64::new(0)); // 8-gate circuits in multi entries
    let solo_7            = Arc::new(AtomicU64::new(0)); // 7-gate circuits alone in entry
    let solo_8            = Arc::new(AtomicU64::new(0)); // 8-gate circuits alone in entry
    // 7/8-gate circuits that share an entry with at least one 1-6 gate circuit
    let friends_7_with_short = Arc::new(AtomicU64::new(0));
    let friends_8_with_short = Arc::new(AtomicU64::new(0));
    // entries containing both a 7-gate and an 8-gate circuit
    let entries_7_and_8 = Arc::new(AtomicU64::new(0));

    (0..256usize).into_par_iter().for_each(|shard_idx| {
        let env = Arc::clone(&env);
        let total_entries        = Arc::clone(&total_entries);
        let multi_entries        = Arc::clone(&multi_entries);
        let friends_7            = Arc::clone(&friends_7);
        let friends_8            = Arc::clone(&friends_8);
        let solo_7               = Arc::clone(&solo_7);
        let solo_8               = Arc::clone(&solo_8);
        let friends_7_with_short = Arc::clone(&friends_7_with_short);
        let friends_8_with_short = Arc::clone(&friends_8_with_short);
        let entries_7_and_8      = Arc::clone(&entries_7_and_8);

        let db = dbs[shard_idx];
        let txn = env.begin_ro_txn().expect("ro txn");
        let mut cursor = txn.open_ro_cursor(db).expect("cursor");

        let mut t_entries  = 0u64;
        let mut t_multi    = 0u64;
        let mut t_f7       = 0u64;
        let mut t_f8       = 0u64;
        let mut t_s7       = 0u64;
        let mut t_s8       = 0u64;
        let mut t_f7_short   = 0u64;
        let mut t_f8_short   = 0u64;
        let mut t_78         = 0u64;

        for (_, value) in cursor.iter_start() {
            // Parse the len-blob list
            let mut gate_lengths: Vec<usize> = Vec::new();
            let mut pos = 0;
            while pos < value.len() {
                let len = value[pos] as usize;
                pos += 1;
                if pos + len > value.len() { break; }
                // Each gate is stored as 3 bytes (u8 wire indices)
                let n_gates = len / 3;
                gate_lengths.push(n_gates);
                pos += len;
            }

            if gate_lengths.is_empty() { continue; }
            t_entries += 1;

            if gate_lengths.len() >= 2 {
                t_multi += 1;
                let has_short = gate_lengths.iter().any(|&g| g >= 1 && g <= 6);
                let n7 = gate_lengths.iter().filter(|&&g| g == 7).count() as u64;
                let n8 = gate_lengths.iter().filter(|&&g| g == 8).count() as u64;
                t_f7 += n7;
                t_f8 += n8;
                if has_short {
                    t_f7_short += n7;
                    t_f8_short += n8;
                }
                let has7 = gate_lengths.iter().any(|&g| g == 7);
                let has8 = gate_lengths.iter().any(|&g| g == 8);
                if has7 && has8 { t_78 += 1; }
            } else {
                let g = gate_lengths[0];
                if g == 7 { t_s7 += 1; }
                if g == 8 { t_s8 += 1; }
            }
        }

        drop(cursor);
        drop(txn);

        total_entries.fetch_add(t_entries,  Ordering::Relaxed);
        multi_entries.fetch_add(t_multi,    Ordering::Relaxed);
        friends_7.fetch_add(t_f7,           Ordering::Relaxed);
        friends_8.fetch_add(t_f8,           Ordering::Relaxed);
        solo_7.fetch_add(t_s7,              Ordering::Relaxed);
        solo_8.fetch_add(t_s8,              Ordering::Relaxed);
        friends_7_with_short.fetch_add(t_f7_short, Ordering::Relaxed);
        friends_8_with_short.fetch_add(t_f8_short, Ordering::Relaxed);
        entries_7_and_8.fetch_add(t_78,            Ordering::Relaxed);

        if shard_idx % 32 == 0 {
            eprintln!("  shard {:02x} done", shard_idx);
        }
    });

    let t   = total_entries.load(Ordering::Relaxed);
    let m   = multi_entries.load(Ordering::Relaxed);
    let f7  = friends_7.load(Ordering::Relaxed);
    let f8  = friends_8.load(Ordering::Relaxed);
    let s7  = solo_7.load(Ordering::Relaxed);
    let s8  = solo_8.load(Ordering::Relaxed);
    let f7s = friends_7_with_short.load(Ordering::Relaxed);
    let f8s = friends_8_with_short.load(Ordering::Relaxed);
    let e78 = entries_7_and_8.load(Ordering::Relaxed);

    println!("Total entries:          {}", t);
    println!("Multi-circuit entries:  {} ({:.2}%)", m, 100.0 * m as f64 / t as f64);
    println!();
    println!("7-gate circuits with any friend (multi entries): {}", f7);
    println!("7-gate circuits with a 1-6 gate friend:         {}", f7s);
    println!("7-gate circuits alone (solo entries):           {}", s7);
    println!("7-gate total:                                   {}", f7 + s7);
    println!();
    println!("8-gate circuits with any friend (multi entries): {}", f8);
    println!("8-gate circuits with a 1-6 gate friend:         {}", f8s);
    println!("8-gate circuits alone (solo entries):           {}", s8);
    println!("8-gate total:                                   {}", f8 + s8);
    println!();
    println!("Entries with both a 7-gate AND 8-gate circuit:  {}", e78);
}
