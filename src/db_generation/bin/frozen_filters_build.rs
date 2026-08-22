//! Build a frozen replacement store's optional `filters.bin` miss filter.
//!
//! The frozen store is the authoritative source. Scanning its encoded key
//! fields prevents a filter/source mismatch from turning real keys into false
//! negatives at runtime, and works for regular and curated stores without
//! RocksDB.
//!
//! ```text
//! frozen_filters_build from-frozen FROZEN_DIR
//! frozen_filters_build FROZEN_DIR
//! ```

use local_mixing::db_mixing::frozen::{mix76, scan_shard_entries};
use std::error::Error;
use std::io::Write;
use std::path::Path;
use xorf::{BinaryFuse8, Filter};

type AnyResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

#[derive(bincode2::Encode)]
#[bincode(crate = "bincode2")]
struct FiltersFileOut {
    table_entry_count: u64,
    filters: Vec<BinaryFuse8>,
}

#[derive(bincode2::Decode)]
#[bincode(crate = "bincode2")]
struct FiltersFileIn {
    table_entry_count: u64,
    filters: Vec<BinaryFuse8>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> AnyResult<()> {
    let args: Vec<String> = std::env::args().collect();
    let frozen_dir = match args.as_slice() {
        [_, frozen_dir] => frozen_dir,
        [_, mode, frozen_dir] if mode == "from-frozen" || mode == "frozen" => frozen_dir,
        _ => usage(),
    };
    require_frozen_store(frozen_dir)?;
    let (per_shard, keys) = collect_from_frozen(frozen_dir)?;
    write_filters_atomically(frozen_dir, per_shard, keys)
}

fn usage() -> ! {
    eprintln!("usage: frozen_filters_build [from-frozen] FROZEN_DIR");
    std::process::exit(2);
}

fn require_frozen_store(frozen_dir: &str) -> AnyResult<()> {
    if !Path::new(&format!("{frozen_dir}/tables.bin")).is_file() {
        return Err(format!("{frozen_dir} does not contain tables.bin").into());
    }
    let out_path = format!("{frozen_dir}/filters.bin");
    if Path::new(&out_path).exists() {
        return Err(format!("{out_path} already exists; refusing to overwrite").into());
    }
    Ok(())
}

fn collect_from_frozen(frozen_dir: &str) -> AnyResult<(Vec<Vec<u64>>, u64)> {
    let start = std::time::Instant::now();
    let mut per_shard: Vec<Vec<u64>> = (0..256).map(|_| Vec::new()).collect();
    let mut keys = 0u64;
    for (shard, values) in per_shard.iter_mut().enumerate() {
        scan_shard_entries(frozen_dir, shard, &mut |bucket, tail, _value| {
            values.push(mix76(shard, bucket, tail));
            keys += 1;
        });
        eprintln!(
            "[filters] scanned shard {shard:02x}: {} keys ({keys} total)",
            values.len()
        );
    }
    if keys == 0 {
        return Err("frozen store yielded no keys".into());
    }
    eprintln!(
        "[filters] {keys} keys collected from frozen store in {:.0}s",
        start.elapsed().as_secs_f64()
    );
    Ok((per_shard, keys))
}

fn write_filters_atomically(
    frozen_dir: &str,
    mut per_shard: Vec<Vec<u64>>,
    keys: u64,
) -> AnyResult<()> {
    let start = std::time::Instant::now();
    let out_path = format!("{frozen_dir}/filters.bin");
    let temp_path = format!("{out_path}.tmp.{}", std::process::id());
    if Path::new(&temp_path).exists() {
        return Err(
            format!("{temp_path} already exists; inspect or move the stale temp file").into(),
        );
    }

    let mut filters = Vec::with_capacity(256);
    for (shard, values) in per_shard.iter_mut().enumerate() {
        values.sort_unstable();
        values.dedup();
        let filter = BinaryFuse8::try_from(values.as_slice())
            .map_err(|error| format!("shard {shard:02x}: fuse construction failed: {error}"))?;
        for &value in values.iter() {
            if !filter.contains(&value) {
                return Err(
                    format!("shard {shard:02x}: constructed filter has a false negative").into(),
                );
            }
        }
        filters.push(filter);
    }

    let output = FiltersFileOut {
        table_entry_count: keys,
        filters,
    };
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp_path)?;
    let mut writer = std::io::BufWriter::with_capacity(8 << 20, file);
    bincode2::encode_into_std_write(&output, &mut writer, bincode2::config::standard())?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    drop(writer);

    let file = std::fs::File::open(&temp_path)?;
    let mut reader = std::io::BufReader::with_capacity(8 << 20, file);
    let decoded: FiltersFileIn =
        bincode2::decode_from_std_read(&mut reader, bincode2::config::standard())?;
    if decoded.table_entry_count != keys {
        return Err(format!(
            "decoded table_entry_count {} differs from scanned key count {keys}",
            decoded.table_entry_count
        )
        .into());
    }
    if decoded.filters.len() != 256 {
        return Err(format!("decoded {} filters, expected 256", decoded.filters.len()).into());
    }
    for (shard, values) in per_shard.iter().enumerate() {
        for &value in values {
            if !decoded.filters[shard].contains(&value) {
                return Err(
                    format!("shard {shard:02x}: serialized filter has a false negative").into(),
                );
            }
        }
    }

    let mut state = 0x9d_2026_08_16u64;
    let mut false_positives = 0u64;
    const PROBES: u64 = 1_000_000;
    for index in 0..PROBES {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        if decoded.filters[(index % 256) as usize].contains(&state) {
            false_positives += 1;
        }
    }
    drop(reader);

    // Hard-link publication is atomic and refuses an existing destination.
    // The temp file remains for inspection on any failure before this point.
    std::fs::hard_link(&temp_path, &out_path)?;
    std::fs::remove_file(&temp_path)?;
    std::fs::File::open(frozen_dir)?.sync_all()?;

    let bytes = std::fs::metadata(&out_path)?.len();
    eprintln!(
        "[filters] published {out_path} ({bytes} bytes); verified {keys} keys; random-probe FP rate {:.4}% ({:.0}s)",
        false_positives as f64 * 100.0 / PROBES as f64,
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
