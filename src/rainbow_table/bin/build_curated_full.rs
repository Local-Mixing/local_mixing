//! Build or import an exact, uncapped minimal-identity curated database.
//!
//! Typical historical-source rebuild:
//!
//! ```text
//! build_curated_full import-legacy-rocks OLD_ROCKS FRESH_COMPOSITE_ROCKS
//! build_curated_full to-lmdb FRESH_COMPOSITE_ROCKS FRESH_LMDB
//! build_curated_full validate-lmdb FRESH_COMPOSITE_ROCKS FRESH_LMDB
//! frozen_from_lmdb tables FRESH_LMDB FRESH_FRZ --curated
//! frozen_from_lmdb write FRESH_LMDB FRESH_FRZ --curated
//! frozen_from_lmdb validate FRESH_LMDB FRESH_FRZ --curated
//! ```
//!
//! `from-identities` instead consumes the accepted `id_g0..id_g33` values
//! produced by the historical minimal-identity tests. Output paths must not
//! exist: this tool never clears or overwrites the live database.

use clap::{Parser, Subcommand};
use lmdb::{
    Cursor, DatabaseFlags, Environment, EnvironmentFlags, RwTransaction, Transaction, WriteFlags,
};
use local_mixing::circuit::{CircuitSeq, circuit::cancel_adjacent_duplicates};
use local_mixing::rainbow_table::curated_full::{
    COMPOSITE_COMPLETE_MARKER, COMPOSITE_FORMAT_MARKER, CuratedError, FUNCTION_KEY_BYTES,
    composite_key, decode_legacy_value, derive_identity_candidates, encode_legacy_record,
    legacy_value_blobs, split_composite_key,
};
use rayon::prelude::*;
use rocksdb::{DB, IteratorMode, MergeOperands, Options, WriteBatch};
use std::collections::BTreeSet;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use xxhash_rust::xxh3::Xxh3;

type AnyError = Box<dyn Error + Send + Sync>;
type AnyResult<T> = Result<T, AnyError>;
const LMDB_META_DB: &str = "curated_full_meta";
const LMDB_META_KEY: &[u8] = b"composite-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AuditStats {
    keys: u64,
    candidates: u64,
    digest: u128,
    max_candidates: u64,
    max_value_bytes: u64,
    max_blob_bytes: usize,
}

#[derive(Parser)]
#[command(about = "Exact, uncapped curated replacement-database builder")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Derive every split candidate from accepted id_g0..id_g33 identities.
    FromIdentities {
        identities_lmdb: PathBuf,
        output_rocks: PathBuf,
        #[arg(long, default_value_t = 4096)]
        batch_identities: usize,
    },
    /// Run the shuffletests regular-friend shortcut without its lossy caps.
    FromRegularShortcut {
        regular_lmdb: PathBuf,
        output_rocks: PathBuf,
    },
    /// Import the known uncapped minimal-identity append-value RocksDB.
    /// A bounded shortcut DB cannot be repaired after records were discarded.
    ImportLegacyRocks {
        input_rocks: PathBuf,
        output_rocks: PathBuf,
    },
    /// Materialize composite records into fresh curated_00..curated_ff LMDBs.
    ToLmdb {
        input_rocks: PathBuf,
        output_lmdb: PathBuf,
        #[arg(long, default_value_t = 6144)]
        map_size_gib: usize,
    },
    /// Prove exact set equality between the composite store and materialized LMDB.
    ValidateLmdb {
        input_rocks: PathBuf,
        output_lmdb: PathBuf,
    },
    /// Report exact key/candidate counts and maxima for a composite store.
    Audit { input_rocks: PathBuf },
}

fn append_merge(_key: &[u8], existing: Option<&[u8]>, operands: &MergeOperands) -> Option<Vec<u8>> {
    let mut result = existing.map_or_else(Vec::new, ToOwned::to_owned);
    for operand in operands {
        result.extend_from_slice(operand);
    }
    Some(result)
}

fn path_text(path: &Path) -> AnyResult<&str> {
    path.to_str()
        .ok_or_else(|| format!("path is not valid UTF-8: {}", path.display()).into())
}

fn require_fresh_output(path: &Path, label: &str) -> AnyResult<()> {
    if path.exists() {
        return Err(format!(
            "refusing to overwrite {label} output {}; choose a fresh path",
            path.display()
        )
        .into());
    }
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn reject_lossy_canonicalization_env() -> AnyResult<()> {
    for name in ["CANON_MONOMIAL_CAP", "CANON_RULE_L_BRANCH_CAP"] {
        if std::env::var_os(name).is_some() {
            return Err(format!(
                "{name} is set; unset canonicalization caps for a complete curated build"
            )
            .into());
        }
    }
    Ok(())
}

fn composite_options(create: bool) -> Options {
    let mut options = Options::default();
    options.create_if_missing(create);
    options.increase_parallelism(num_cpus::get() as i32);
    options.set_max_background_jobs(num_cpus::get().max(2) as i32);
    options.set_compression_type(rocksdb::DBCompressionType::Zstd);
    options.set_bottommost_compression_type(rocksdb::DBCompressionType::Zstd);
    options
}

fn open_composite_read(path: &Path) -> AnyResult<DB> {
    let options = composite_options(false);
    let database = DB::open_for_read_only(&options, path_text(path)?, false)?;
    if database.get(COMPOSITE_FORMAT_MARKER)?.as_deref() != Some(b"1") {
        return Err(format!(
            "{} is not a curated-full composite-v1 store",
            path.display()
        )
        .into());
    }
    let complete = database.get(COMPOSITE_COMPLETE_MARKER)?;
    if complete
        .as_deref()
        .and_then(decode_completion_manifest)
        .is_none()
    {
        return Err(format!(
            "{} has no valid completion manifest; it may be an interrupted partial build",
            path.display()
        )
        .into());
    }
    Ok(database)
}

fn put_format_marker(db: &DB) -> AnyResult<()> {
    db.put(COMPOSITE_FORMAT_MARKER, b"1")?;
    Ok(())
}

fn is_composite_metadata(key: &[u8]) -> bool {
    key == COMPOSITE_FORMAT_MARKER || key == COMPOSITE_COMPLETE_MARKER
}

fn encode_completion_manifest(stats: AuditStats) -> [u8; 32] {
    let mut encoded = [0u8; 32];
    encoded[..8].copy_from_slice(&stats.keys.to_le_bytes());
    encoded[8..16].copy_from_slice(&stats.candidates.to_le_bytes());
    encoded[16..].copy_from_slice(&stats.digest.to_le_bytes());
    encoded
}

fn decode_completion_manifest(encoded: &[u8]) -> Option<(u64, u64, u128)> {
    if encoded.len() != 32 {
        return None;
    }
    Some((
        u64::from_le_bytes(encoded[..8].try_into().ok()?),
        u64::from_le_bytes(encoded[8..16].try_into().ok()?),
        u128::from_le_bytes(encoded[16..].try_into().ok()?),
    ))
}

fn expected_manifest(database: &DB) -> AnyResult<(u64, u64, u128)> {
    let encoded = database
        .get(COMPOSITE_COMPLETE_MARKER)?
        .ok_or("missing composite completion manifest")?;
    decode_completion_manifest(&encoded)
        .ok_or_else(|| "invalid composite completion manifest".into())
}

fn write_identity(db: &DB, blob: &[u8]) -> AnyResult<(u64, u64)> {
    if blob.is_empty() || blob.len() % 3 != 0 {
        return Err(format!("malformed identity blob of {} bytes", blob.len()).into());
    }
    let identity = CircuitSeq::from_blob(blob);
    write_identity_circuit(db, &identity)
}

fn write_identity_circuit(db: &DB, identity: &CircuitSeq) -> AnyResult<(u64, u64)> {
    let mut exact_records = BTreeSet::new();
    let generated = derive_identity_candidates(identity, |key, candidate| {
        exact_records.insert(composite_key(&key, &candidate)?);
        Ok(())
    })?;
    let unique = exact_records.len() as u64;
    let mut batch = WriteBatch::default();
    for record in exact_records {
        batch.put(record, []);
    }
    db.write(batch)?;
    Ok((generated, unique))
}

fn process_identity_batch(
    db: &DB,
    identities: &[Vec<u8>],
    generated: &AtomicU64,
    locally_unique: &AtomicU64,
) -> AnyResult<()> {
    identities.par_iter().try_for_each(|identity| {
        let (raw, unique) = write_identity(db, identity)?;
        generated.fetch_add(raw, Ordering::Relaxed);
        locally_unique.fetch_add(unique, Ordering::Relaxed);
        Ok::<_, AnyError>(())
    })
}

fn simplify_shortcut_identity(mut identity: CircuitSeq) -> CircuitSeq {
    identity.canonicalize();
    cancel_adjacent_duplicates(&mut identity.gates, None::<&mut Vec<()>>);
    identity
}

fn for_each_ordered_source_pair<T>(
    items: &[T],
    mut visit: impl FnMut(usize, &T, usize, &T) -> AnyResult<()>,
) -> AnyResult<()> {
    for left in 0..items.len() {
        for right in 0..items.len() {
            if left != right {
                visit(left, &items[left], right, &items[right])?;
            }
        }
    }
    Ok(())
}

fn write_regular_equivalence_class(db: &DB, value: &[u8]) -> AnyResult<(u64, u64, u64)> {
    let source: BTreeSet<Vec<u8>> = decode_legacy_value(value)?.into_iter().collect();
    if source.len() < 2 {
        return Ok((0, 0, 0));
    }
    let circuits: Vec<CircuitSeq> = source
        .into_iter()
        .map(|blob| CircuitSeq::from_blob(&blob))
        .collect();
    let mut identities = 0u64;
    let mut generated = 0u64;
    let mut local_unique = 0u64;
    for_each_ordered_source_pair(&circuits, |_, a, _, b| {
        let mut reverse_a = a.gates.clone();
        reverse_a.reverse();
        let mut reverse_b = b.gates.clone();
        reverse_b.reverse();
        let spellings = [
            a.gates
                .iter()
                .copied()
                .chain(reverse_b.iter().copied())
                .collect(),
            reverse_a
                .iter()
                .copied()
                .chain(b.gates.iter().copied())
                .collect(),
        ];
        for gates in spellings {
            let identity = simplify_shortcut_identity(CircuitSeq { gates });
            if identity.gates.len() < 3 {
                continue;
            }
            identities += 1;
            let (identity_generated, identity_unique) = write_identity_circuit(db, &identity)?;
            generated += identity_generated;
            local_unique += identity_unique;
        }
        Ok(())
    })?;
    Ok((identities, generated, local_unique))
}

fn from_regular_shortcut(input: &Path, output: &Path) -> AnyResult<()> {
    reject_lossy_canonicalization_env()?;
    require_fresh_output(output, "RocksDB")?;
    let environment = Arc::new(
        Environment::new()
            .set_flags(EnvironmentFlags::READ_ONLY | EnvironmentFlags::NO_LOCK)
            .set_max_dbs(600)
            .set_max_readers(1024)
            .set_map_size(6 * 1024 * 1024 * 1024 * 1024)
            .open(input)?,
    );
    let sources = (0u16..=255)
        .map(|shard| environment.open_db(Some(&format!("{shard:02x}"))))
        .collect::<Result<Vec<_>, _>>()?;
    let output_db = Arc::new(DB::open(&composite_options(true), path_text(output)?)?);
    put_format_marker(&output_db)?;

    let entries = AtomicU64::new(0);
    let identities = AtomicU64::new(0);
    let generated = AtomicU64::new(0);
    let local_unique = AtomicU64::new(0);
    (0..256usize)
        .into_par_iter()
        .try_for_each(|shard| -> AnyResult<()> {
            let transaction = environment.begin_ro_txn()?;
            let mut cursor = transaction.open_ro_cursor(sources[shard])?;
            let mut shard_entries = 0u64;
            for (_, value) in cursor.iter() {
                let (identity_count, generated_count, unique_count) =
                    write_regular_equivalence_class(&output_db, value)?;
                if identity_count == 0 {
                    continue;
                }
                shard_entries += 1;
                identities.fetch_add(identity_count, Ordering::Relaxed);
                generated.fetch_add(generated_count, Ordering::Relaxed);
                local_unique.fetch_add(unique_count, Ordering::Relaxed);
            }
            entries.fetch_add(shard_entries, Ordering::Relaxed);
            eprintln!(
                "[from-regular-shortcut] shard={shard:02x} qualifying={} total-qualifying={} identities={} generated={} local-unique-attempts={}",
                shard_entries,
                entries.load(Ordering::Relaxed),
                identities.load(Ordering::Relaxed),
                generated.load(Ordering::Relaxed),
                local_unique.load(Ordering::Relaxed)
            );
            Ok(())
        })?;
    if entries.load(Ordering::Relaxed) == 0 {
        return Err("regular LMDB produced no qualifying multi-friend entries".into());
    }
    finalize_composite(&output_db, "from-regular-shortcut")?;
    Ok(())
}

fn from_identities(input: &Path, output: &Path, batch_size: usize) -> AnyResult<()> {
    reject_lossy_canonicalization_env()?;
    if batch_size == 0 {
        return Err("--batch-identities must be positive".into());
    }
    require_fresh_output(output, "RocksDB")?;
    let environment = Environment::new()
        .set_flags(EnvironmentFlags::READ_ONLY | EnvironmentFlags::NO_LOCK)
        .set_max_dbs(600)
        .set_max_readers(1024)
        .set_map_size(6 * 1024 * 1024 * 1024 * 1024)
        .open(input)?;
    let options = composite_options(true);
    let output_db = Arc::new(DB::open(&options, path_text(output)?)?);
    put_format_marker(&output_db)?;

    let generated = AtomicU64::new(0);
    let locally_unique = AtomicU64::new(0);
    let identities = AtomicU64::new(0);
    for identity_type in 0..34usize {
        let name = format!("id_g{identity_type}");
        let source_db = environment.open_db(Some(&name)).map_err(|error| {
            format!(
                "missing required accepted-identity database {name}: {error}; expected the complete id_g0..id_g33 corpus"
            )
        })?;
        let transaction = environment.begin_ro_txn()?;
        let mut cursor = transaction.open_ro_cursor(source_db)?;
        let mut batch = Vec::with_capacity(batch_size);
        for (_, value) in cursor.iter() {
            batch.push(value.to_vec());
            if batch.len() == batch_size {
                process_identity_batch(&output_db, &batch, &generated, &locally_unique)?;
                identities.fetch_add(batch.len() as u64, Ordering::Relaxed);
                batch.clear();
            }
        }
        if !batch.is_empty() {
            process_identity_batch(&output_db, &batch, &generated, &locally_unique)?;
            identities.fetch_add(batch.len() as u64, Ordering::Relaxed);
        }
        eprintln!(
            "[from-identities] {name}: total identities={} generated={} local-unique-attempts={}",
            identities.load(Ordering::Relaxed),
            generated.load(Ordering::Relaxed),
            locally_unique.load(Ordering::Relaxed)
        );
    }
    if identities.load(Ordering::Relaxed) == 0 {
        return Err(
            "id_g0..id_g33 contain zero accepted identities; refusing an empty complete store"
                .into(),
        );
    }
    finalize_composite(&output_db, "from-identities")?;
    Ok(())
}

fn import_legacy(input: &Path, output: &Path) -> AnyResult<()> {
    require_fresh_output(output, "RocksDB")?;
    let mut source_options = Options::default();
    source_options.set_merge_operator_associative("append_merge", append_merge);
    let source = DB::open_for_read_only(&source_options, path_text(input)?, false)?;
    let output_options = composite_options(true);
    let exact = DB::open(&output_options, path_text(output)?)?;
    put_format_marker(&exact)?;

    let mut source_keys = 0u64;
    let mut source_records = 0u64;
    let mut output_attempts = 0u64;
    for item in source.iterator(IteratorMode::Start) {
        let (key, value) = item?;
        if key.len() != FUNCTION_KEY_BYTES {
            return Err(format!("legacy RocksDB key has {} bytes, expected 16", key.len()).into());
        }
        let mut function_key = [0u8; FUNCTION_KEY_BYTES];
        function_key.copy_from_slice(&key);
        source_keys += 1;

        let mut batch = WriteBatch::default();
        let mut batch_len = 0usize;
        for candidate in legacy_value_blobs(&value) {
            let candidate = candidate?;
            batch.put(composite_key(&function_key, candidate)?, []);
            batch_len += 1;
            source_records += 1;
            output_attempts += 1;
            if batch_len == 65_536 {
                exact.write(std::mem::take(&mut batch))?;
                batch_len = 0;
            }
        }
        if batch_len != 0 {
            exact.write(batch)?;
        }
        if source_keys % 1_000_000 == 0 {
            eprintln!(
                "[import] keys={source_keys} source-records={source_records} output-attempts={output_attempts}"
            );
        }
    }
    eprintln!(
        "[import] complete: keys={source_keys} source-records={source_records} output-attempts={output_attempts}"
    );
    if source_keys == 0 || source_records == 0 {
        return Err("legacy curated RocksDB contained no candidate records".into());
    }
    finalize_composite(&exact, "import")?;
    Ok(())
}

fn open_output_lmdb(path: &Path, map_size_gib: usize) -> AnyResult<Environment> {
    require_fresh_output(path, "LMDB")?;
    std::fs::create_dir(path)?;
    let bytes = map_size_gib
        .checked_mul(1024 * 1024 * 1024)
        .ok_or("--map-size-gib overflow")?;
    Ok(Environment::new()
        .set_flags(
            EnvironmentFlags::WRITE_MAP | EnvironmentFlags::MAP_ASYNC | EnvironmentFlags::NO_SYNC,
        )
        .set_max_dbs(600)
        .set_max_readers(1024)
        .set_map_size(bytes)
        .open(path)?)
}

fn put_group(
    transaction: &mut RwTransaction<'_>,
    databases: &[lmdb::Database],
    key: &[u8; FUNCTION_KEY_BYTES],
    value: &[u8],
) -> AnyResult<()> {
    transaction.put(databases[key[0] as usize], key, &value, WriteFlags::APPEND)?;
    Ok(())
}

fn to_lmdb(input: &Path, output: &Path, map_size_gib: usize) -> AnyResult<()> {
    let source = open_composite_read(input)?;
    let source_stats = verify_manifest(&source)?;
    let environment = open_output_lmdb(output, map_size_gib)?;
    let databases: Vec<lmdb::Database> = (0u16..=255)
        .map(|shard| {
            environment.create_db(
                Some(&format!("curated_{shard:02x}")),
                DatabaseFlags::empty(),
            )
        })
        .collect::<Result<_, _>>()?;
    let metadata = environment.create_db(Some(LMDB_META_DB), DatabaseFlags::empty())?;

    let mut transaction = environment.begin_rw_txn()?;
    let mut current_key: Option<[u8; FUNCTION_KEY_BYTES]> = None;
    let mut value = Vec::new();
    let mut keys = 0u64;
    let mut candidates = 0u64;
    for item in source.iterator(IteratorMode::Start) {
        let (record, _) = item?;
        if is_composite_metadata(&record) {
            continue;
        }
        let (key, blob) = split_composite_key(&record)?;
        if current_key != Some(key) {
            if let Some(previous) = current_key {
                put_group(&mut transaction, &databases, &previous, &value)?;
                keys += 1;
                if keys % 250_000 == 0 {
                    transaction.commit()?;
                    transaction = environment.begin_rw_txn()?;
                    eprintln!("[to-lmdb] keys={keys} candidates={candidates}");
                }
            }
            current_key = Some(key);
            value.clear();
        }
        value.extend(encode_legacy_record(blob)?);
        candidates += 1;
    }
    if let Some(last) = current_key {
        put_group(&mut transaction, &databases, &last, &value)?;
        keys += 1;
    }
    transaction.commit()?;
    environment.sync(true)?;
    if (keys, candidates) != (source_stats.keys, source_stats.candidates) {
        return Err(format!(
            "materialization count mismatch: source keys/candidates={}/{}, LMDB={keys}/{candidates}",
            source_stats.keys, source_stats.candidates
        )
        .into());
    }
    let mut metadata_transaction = environment.begin_rw_txn()?;
    metadata_transaction.put(
        metadata,
        &LMDB_META_KEY,
        &encode_completion_manifest(source_stats),
        WriteFlags::NO_OVERWRITE,
    )?;
    metadata_transaction.commit()?;
    environment.sync(true)?;
    eprintln!("[to-lmdb] complete: keys={keys} candidates={candidates}");
    Ok(())
}

fn read_lmdb_manifest(environment: &Environment) -> AnyResult<(u64, u64, u128)> {
    let database = environment
        .open_db(Some(LMDB_META_DB))
        .map_err(|error| format!("missing {LMDB_META_DB}; LMDB build may be partial: {error}"))?;
    let transaction = environment.begin_ro_txn()?;
    let encoded = transaction
        .get(database, &LMDB_META_KEY)
        .map_err(|error| format!("missing LMDB completion manifest: {error}"))?;
    decode_completion_manifest(encoded).ok_or_else(|| "invalid LMDB completion manifest".into())
}

fn open_input_lmdb(path: &Path) -> AnyResult<(Environment, Vec<lmdb::Database>)> {
    let environment = Environment::new()
        .set_flags(EnvironmentFlags::READ_ONLY | EnvironmentFlags::NO_LOCK)
        .set_max_dbs(600)
        .set_max_readers(1024)
        .set_map_size(6 * 1024 * 1024 * 1024 * 1024)
        .open(path)?;
    let databases = (0u16..=255)
        .map(|shard| environment.open_db(Some(&format!("curated_{shard:02x}"))))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((environment, databases))
}

fn validate_lmdb(input: &Path, output: &Path) -> AnyResult<()> {
    let source = open_composite_read(input)?;
    let (environment, databases) = open_input_lmdb(output)?;
    let source_stats = verify_manifest(&source)?;
    let lmdb_manifest = read_lmdb_manifest(&environment)?;
    if lmdb_manifest
        != (
            source_stats.keys,
            source_stats.candidates,
            source_stats.digest,
        )
    {
        return Err("LMDB completion manifest does not match the composite source".into());
    }
    let rocks_candidates = source_stats.candidates;
    let rocks_keys = source_stats.keys;

    let transaction = environment.begin_ro_txn()?;
    let mut source_records = source.iterator(IteratorMode::Start);
    let mut next_source_record = || -> AnyResult<Option<Box<[u8]>>> {
        loop {
            let Some(item) = source_records.next() else {
                return Ok(None);
            };
            let (record, _) = item?;
            if !is_composite_metadata(&record) {
                return Ok(Some(record));
            }
        }
    };
    let mut lmdb_keys = 0u64;
    let mut lmdb_candidates = 0u64;
    for (shard, &database) in databases.iter().enumerate() {
        let mut cursor = transaction.open_ro_cursor(database)?;
        for (key, value) in cursor.iter() {
            if key.len() != FUNCTION_KEY_BYTES || key[0] as usize != shard {
                return Err(format!("invalid key in curated_{shard:02x}").into());
            }
            let mut function_key = [0u8; FUNCTION_KEY_BYTES];
            function_key.copy_from_slice(key);
            for blob in legacy_value_blobs(value) {
                let blob = blob?;
                let record = composite_key(&function_key, blob)?;
                let expected = next_source_record()?.ok_or_else(|| {
                    format!("LMDB has an extra candidate for key {:02x?}", function_key)
                })?;
                if expected.as_ref() != record.as_slice() {
                    return Err(format!(
                        "LMDB/composite candidate order or content mismatch for key {:02x?}",
                        function_key
                    )
                    .into());
                }
                lmdb_candidates += 1;
            }
            lmdb_keys += 1;
        }
        if (shard + 1) % 16 == 0 {
            eprintln!(
                "[validate] shards={}/256 keys={lmdb_keys} candidates={lmdb_candidates}",
                shard + 1
            );
        }
    }
    if next_source_record()?.is_some() {
        return Err("composite source has candidates missing from LMDB".into());
    }
    if (lmdb_keys, lmdb_candidates) != (rocks_keys, rocks_candidates) {
        return Err(format!(
            "count mismatch: RocksDB keys/candidates={rocks_keys}/{rocks_candidates}, LMDB={lmdb_keys}/{lmdb_candidates}"
        )
        .into());
    }
    eprintln!(
        "[validate] PASS: exact set equality for {lmdb_keys} keys / {lmdb_candidates} candidates"
    );
    Ok(())
}

fn audit_db(database: &DB) -> AnyResult<AuditStats> {
    let mut keys = 0u64;
    let mut candidates = 0u64;
    let mut max_candidates = 0u64;
    let mut max_value_bytes = 0u64;
    let mut max_blob_bytes = 0usize;
    let mut previous_key = None;
    let mut current_candidates = 0u64;
    let mut current_value_bytes = 0u64;
    let mut digest = Xxh3::new();
    for item in database.iterator(IteratorMode::Start) {
        let (record, _) = item?;
        if is_composite_metadata(&record) {
            continue;
        }
        let record_len = u16::try_from(record.len()).expect("checked composite key fits u16");
        digest.update(&record_len.to_le_bytes());
        digest.update(&record);
        let (key, blob) = split_composite_key(&record)?;
        if previous_key != Some(key) {
            if previous_key.is_some() {
                max_candidates = max_candidates.max(current_candidates);
                max_value_bytes = max_value_bytes.max(current_value_bytes);
            }
            keys += 1;
            current_candidates = 0;
            current_value_bytes = 0;
            previous_key = Some(key);
        }
        candidates += 1;
        current_candidates += 1;
        current_value_bytes += (blob.len() + 1) as u64;
        max_blob_bytes = max_blob_bytes.max(blob.len());
    }
    if previous_key.is_some() {
        max_candidates = max_candidates.max(current_candidates);
        max_value_bytes = max_value_bytes.max(current_value_bytes);
    }
    let stats = AuditStats {
        keys,
        candidates,
        digest: digest.digest128(),
        max_candidates,
        max_value_bytes,
        max_blob_bytes,
    };
    eprintln!(
        "[audit] keys={} exact-candidates={} digest={:032x} max-candidates/key={} max-value-bytes={} max-circuit-bytes={}",
        stats.keys,
        stats.candidates,
        stats.digest,
        stats.max_candidates,
        stats.max_value_bytes,
        stats.max_blob_bytes
    );
    Ok(stats)
}

fn finalize_composite(database: &DB, label: &str) -> AnyResult<()> {
    database.flush()?;
    eprintln!("[{label}] compacting exact composite store");
    database.compact_range::<&[u8], &[u8]>(None, None);
    let stats = audit_db(database)?;
    database.put(COMPOSITE_COMPLETE_MARKER, encode_completion_manifest(stats))?;
    database.flush()?;
    eprintln!("[{label}] completion manifest committed");
    Ok(())
}

fn verify_manifest(database: &DB) -> AnyResult<AuditStats> {
    let expected = expected_manifest(database)?;
    let actual = audit_db(database)?;
    if (actual.keys, actual.candidates, actual.digest) != expected {
        return Err(format!(
            "composite completion manifest mismatch: expected keys/candidates/digest={}/{}/{:032x}, found {}/{}/{:032x}",
            expected.0,
            expected.1,
            expected.2,
            actual.keys,
            actual.candidates,
            actual.digest
        )
        .into());
    }
    Ok(actual)
}

fn audit(input: &Path) -> AnyResult<()> {
    let database = open_composite_read(input)?;
    verify_manifest(&database)?;
    eprintln!("[audit] PASS: completion manifest matches exact contents");
    Ok(())
}

fn run() -> AnyResult<()> {
    match Cli::parse().command {
        Command::FromIdentities {
            identities_lmdb,
            output_rocks,
            batch_identities,
        } => from_identities(&identities_lmdb, &output_rocks, batch_identities),
        Command::FromRegularShortcut {
            regular_lmdb,
            output_rocks,
        } => from_regular_shortcut(&regular_lmdb, &output_rocks),
        Command::ImportLegacyRocks {
            input_rocks,
            output_rocks,
        } => import_legacy(&input_rocks, &output_rocks),
        Command::ToLmdb {
            input_rocks,
            output_lmdb,
            map_size_gib,
        } => to_lmdb(&input_rocks, &output_lmdb, map_size_gib),
        Command::ValidateLmdb {
            input_rocks,
            output_lmdb,
        } => validate_lmdb(&input_rocks, &output_lmdb),
        Command::Audit { input_rocks } => audit(&input_rocks),
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("build_curated_full: {error}");
        let mut source = error.source();
        while let Some(cause) = source {
            eprintln!("  caused by: {cause}");
            source = cause.source();
        }
        std::process::exit(1);
    }
}

#[allow(dead_code)]
fn _assert_curated_error_is_thread_safe(error: CuratedError) -> AnyError {
    Box::new(error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(label: &str) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let serial = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "local-mixing-{label}-{}-{serial}",
                std::process::id()
            ));
            assert!(!path.exists());
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn exact_composite_to_lmdb_preserves_three_hundred_candidates() {
        let root = TestDir::new("curated-full-e2e");
        let rocks_path = root.0.join("composite");
        let lmdb_path = root.0.join("lmdb");
        let key = [0x5au8; FUNCTION_KEY_BYTES];
        let mut expected = BTreeSet::new();

        {
            let database = DB::open(&composite_options(true), path_text(&rocks_path).unwrap())
                .expect("create composite test store");
            put_format_marker(&database).unwrap();
            for i in 0..300u16 {
                let blob = vec![
                    (i >> 8) as u8,
                    i as u8,
                    1,
                    9,
                    (i.wrapping_mul(17) >> 8) as u8,
                    i.wrapping_mul(17) as u8,
                ];
                expected.insert(blob.clone());
                database
                    .put(composite_key(&key, &blob).unwrap(), [])
                    .unwrap();
            }
            finalize_composite(&database, "test").unwrap();
        }

        audit(&rocks_path).unwrap();
        to_lmdb(&rocks_path, &lmdb_path, 1).unwrap();
        validate_lmdb(&rocks_path, &lmdb_path).unwrap();

        let (environment, databases) = open_input_lmdb(&lmdb_path).unwrap();
        let transaction = environment.begin_ro_txn().unwrap();
        let value = transaction.get(databases[key[0] as usize], &key).unwrap();
        let actual: BTreeSet<Vec<u8>> = decode_legacy_value(value).unwrap().into_iter().collect();
        assert_eq!(actual, expected);
        assert_eq!(actual.len(), 300);
        assert!(value.len() > 512);
    }

    #[test]
    fn interrupted_composite_store_is_rejected() {
        let root = TestDir::new("curated-full-partial");
        let rocks_path = root.0.join("partial");
        {
            let database = DB::open(&composite_options(true), path_text(&rocks_path).unwrap())
                .expect("create partial test store");
            put_format_marker(&database).unwrap();
            database.flush().unwrap();
        }
        let error = match open_composite_read(&rocks_path) {
            Ok(_) => panic!("partial composite store was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("interrupted partial build"));
    }

    #[test]
    fn regular_shortcut_enumerates_source_friend_after_twenty() {
        let friends: Vec<u8> = (0..21).collect();
        let mut visited = BTreeSet::new();
        for_each_ordered_source_pair(&friends, |left, _, right, _| {
            visited.insert((left, right));
            Ok(())
        })
        .unwrap();

        assert_eq!(visited.len(), 21 * 20);
        assert!(visited.contains(&(20, 0)));
        assert!(visited.contains(&(0, 20)));
    }
}
