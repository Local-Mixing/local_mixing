//! S8/V8 of the wide-gate design: measure the corpus quantities the append
//! pass schedule depends on — circuits-per-key (`f`) and the touched-wire
//! (`u`) histogram — by random-seek sampling of a rocks band.
//!
//! Usage: corpus_stats <band_dir> [samples=200000] [seed=7]
use local_mixing::db_generation::regular::decode_rocks_entry;
use rocksdb::{DB, IteratorMode, Options};

fn dedup_concat_local(values: Vec<Vec<u8>>) -> Vec<u8> {
    // Mirror of the generator's append_merge/dedup_concat: values are chains
    // of [len][blob] chunks; identical chunks dedup, order preserved.
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for v in values {
        let mut i = 0usize;
        while i < v.len() {
            let len = v[i] as usize;
            if len == 0 || len % 3 != 0 || i + 1 + len > v.len() {
                break;
            }
            let chunk = &v[i..i + 1 + len];
            if seen.insert(chunk.to_vec()) {
                out.extend_from_slice(chunk);
            }
            i += 1 + len;
        }
    }
    out
}

fn append_merge_local(
    _key: &[u8],
    existing: Option<&[u8]>,
    operands: &rocksdb::MergeOperands,
) -> Option<Vec<u8>> {
    let mut values = Vec::with_capacity(operands.len() + usize::from(existing.is_some()));
    if let Some(v) = existing {
        values.push(v.to_vec());
    }
    for op in operands {
        values.push(op.to_vec());
    }
    Some(dedup_concat_local(values))
}

fn main() {
    let mut a = std::env::args().skip(1);
    let dir = a.next().expect("usage: corpus_stats <band_dir> [samples]");
    let samples: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let mut seed: u64 = a.next().and_then(|s| s.parse().ok()).unwrap_or(7);

    let mut opts = Options::default();
    opts.create_if_missing(false);
    opts.set_merge_operator_associative("append_merge", append_merge_local);
    opts.set_max_open_files(-1);
    let db = DB::open_for_read_only(&opts, &dir, false).expect("open band read-only");

    let mut next = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seed
    };

    let mut f_hist: std::collections::BTreeMap<usize, u64> = Default::default();
    let mut u_hist: std::collections::BTreeMap<usize, u64> = Default::default();
    let mut total_entries = 0u64;
    let mut total_circuits = 0u64;
    let mut total_value_bytes = 0u64;

    const PER_SEEK: usize = 50;
    let seeks = samples.div_ceil(PER_SEEK);
    for _ in 0..seeks {
        let probe: [u8; 16] = {
            let a = next().to_le_bytes();
            let b = next().to_le_bytes();
            let mut k = [0u8; 16];
            k[..8].copy_from_slice(&a);
            k[8..].copy_from_slice(&b);
            k
        };
        let iter = db.iterator(IteratorMode::From(&probe, rocksdb::Direction::Forward));
        for item in iter.take(PER_SEEK) {
            let (key, value) = item.expect("iterate");
            let circuits = match decode_rocks_entry(&key, &value) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("decode error (skipped): {e}");
                    continue;
                }
            };
            total_entries += 1;
            total_value_bytes += value.len() as u64;
            *f_hist.entry(circuits.len()).or_default() += 1;
            total_circuits += circuits.len() as u64;
            for c in &circuits {
                let mut wires: Vec<u16> = c.gates.iter().flatten().copied().collect();
                wires.sort_unstable();
                wires.dedup();
                *u_hist.entry(wires.len()).or_default() += 1;
            }
        }
        if total_entries as usize >= samples {
            break;
        }
    }

    println!("band={dir} sampled_entries={total_entries} circuits={total_circuits}");
    println!(
        "f (circuits/key): mean={:.3}",
        total_circuits as f64 / total_entries.max(1) as f64
    );
    for (f, n) in &f_hist {
        println!(
            "  f={f:3}  keys={n}  ({:.2}%)",
            *n as f64 * 100.0 / total_entries as f64
        );
    }
    println!(
        "value bytes/key mean={:.1}",
        total_value_bytes as f64 / total_entries.max(1) as f64
    );
    println!("u (touched wires per stored circuit):");
    for (u, n) in &u_hist {
        println!(
            "  u={u:3}  circuits={n}  ({:.2}%)",
            *n as f64 * 100.0 / total_circuits as f64
        );
    }
}
