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
use local_mixing::circuit::{CircuitSeq, cancel_adjacent_duplicates};
use local_mixing::db_generation::curated_full::{
    COMPOSITE_COMPLETE_MARKER, COMPOSITE_FORMAT_MARKER, CuratedError, FUNCTION_KEY_BYTES,
    composite_key, decode_legacy_value, derive_identity_candidates_where, dihedral_canonical_word,
    encode_legacy_record, legacy_value_blobs, split_composite_key, word_bytes,
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
    /// Same construction, but sourced from the FROZEN regular store rather than
    /// an LMDB. The frozen store is the complete regular corpus, so this reaches
    /// every identity the pairing can produce -- 40,011,376 of them, against the
    /// 39,469,712 an LMDB source yielded.
    ///
    /// `--good-splits` keeps only splits whose two halves each contain no
    /// locally compressible window, so every stored candidate survives an
    /// adversary's own compression pass. This is a per-SPLIT filter, not a
    /// per-identity one: a single bad window in an identity spoils only the
    /// splits that leave it inside a half, and splits cutting through it are
    /// still worth keeping.
    FromFrozenIdentities {
        frozen_regular: PathBuf,
        output_rocks: PathBuf,
        #[arg(long)]
        good_splits: bool,
        #[arg(long, default_value_t = 32)]
        wires: usize,
    },
    /// V2 build: identity dihedral-orbit dedup (~12.7x fewer identities, no
    /// loss -- the split enumeration reaches every orbit member's arcs from
    /// one canonical word) plus one candidate per (identity, key, gate-count)
    /// (rotation siblings collapse at the source). Optionally follows with a
    /// cross-class gluing phase that manufactures 13+-gate identities --
    /// unreachable by same-class pairing, whose long glues all cancel -- by
    /// pairing a store member `a` (function F) with a curated-composite
    /// spelling `b` of g∘F: rev(a) ++ b computes g, so rev(a)·b·[g] is an
    /// identity of |a|+|b|+1 gates and its arcs are the 12-16 gate m1/m2
    /// candidates the store otherwise cannot contain.
    FromFrozenIdentitiesV2 {
        frozen_regular: PathBuf,
        output_rocks: PathBuf,
        /// Composite store used as BOTH source and partner pool for gluing;
        /// omitting it skips the gluing phase.
        #[arg(long)]
        glue_partner_composite: Option<PathBuf>,
        /// Number of random source keys sampled from the composite (keys are
        /// hashes, so random-seek sampling is uniform).
        #[arg(long, default_value_t = 200_000)]
        glue_source_keys: usize,
        /// Glued pairs harvested per discovered (F, g∘F) key connection.
        #[arg(long, default_value_t = 16)]
        glue_pairs: usize,
        /// Glued identities below this length are discarded (short spellings
        /// are already covered by the same-class phase).
        #[arg(long, default_value_t = 13)]
        glue_min_identity_gates: usize,
        /// Frozen shards to process in phase 1 (256 = the whole store; lower
        /// values are for smoke tests and leave the output NON-complete in
        /// coverage terms, though its manifest is still valid).
        #[arg(long, default_value_t = 256)]
        shards: usize,
    },
    /// Per-key structural-diversity sieve: keys with more candidates than
    /// --keep-all-below are reduced, shortest candidates first, to a set in
    /// which no two circuits share a contiguous --shingle-gate subword
    /// (either direction, wires relabelled), with --cell-floor candidates per
    /// (gates, wires) cell always kept so no length class a key had dies.
    Sieve {
        input_rocks: PathBuf,
        output_rocks: PathBuf,
        #[arg(long, default_value_t = 6)]
        shingle: usize,
        #[arg(long, default_value_t = 1000)]
        keep_all_below: u64,
        #[arg(long, default_value_t = 1)]
        cell_floor: usize,
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
    write_identity_circuit_where(db, identity, |_, _, _| true)
}

fn write_identity_circuit_where<A>(
    db: &DB,
    identity: &CircuitSeq,
    accept: A,
) -> AnyResult<(u64, u64)>
where
    A: FnMut(bool, usize, usize) -> bool,
{
    let mut exact_records = BTreeSet::new();
    let generated = derive_identity_candidates_where(identity, accept, |key, candidate| {
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

/// Cyclic compressibility of one identity: `table[start][len]` is true when the
/// cyclic window of `len` gates beginning at `start` has a strictly shorter
/// equivalent in the regular store, verified by `verify_rewrite`.
///
/// Cyclic because every rotation of an identity is itself an identity (a
/// rotation is a conjugate, and conjugates of I are I), so one table answers
/// for all `n` rotations. Direction needs no second table either: `db_probe`
/// already tries both canonical frames, so a window and its reverse get the
/// same verdict. That turns ~15,800 probes per identity into ~120.
struct CyclicCompress {
    n: usize,
    table: Vec<bool>,
}

impl CyclicCompress {
    fn probe(
        identity: &CircuitSeq,
        db: &local_mixing::db_mixing::frozen::FrozenDb,
        budget: local_mixing::engine::xpoly::XPolyBudget,
        num_wires: usize,
        rng: &mut rand::rngs::StdRng,
        probes: &mut u64,
    ) -> Self {
        use local_mixing::circuit::xgate::XGate;
        use local_mixing::db_mixing::db_replace::{db_g57_to_xgate, db_probe};
        use local_mixing::engine::rules::verify_rewrite;

        let n = identity.gates.len();
        let gates: Vec<XGate> = identity.gates.iter().map(|&g| db_g57_to_xgate(g)).collect();
        let mut table = vec![false; n * (n + 1)];
        for start in 0..n {
            for len in 2..n {
                let window: Vec<XGate> = (0..len).map(|i| gates[(start + i) % n].clone()).collect();
                if local_mixing::engine::xpoly::xgate_used_wires(&window).len() > 30 {
                    continue;
                }
                *probes += 1;
                for (replacement, _, _) in db_probe(&window, num_wires, db, budget, rng) {
                    if replacement.len() < len && verify_rewrite(&window, &replacement) {
                        table[start * (n + 1) + len] = true;
                        break;
                    }
                }
            }
        }
        Self { n, table }
    }

    /// True when no compressible window sits strictly inside the piece that
    /// spans `len` gates from cyclic position `start`. Windows equal to the
    /// whole piece are excluded on purpose: shrinking a stored candidate as a
    /// whole requires knowing its exact boundaries, whereas an interior window
    /// can be compressed with no alignment knowledge at all.
    fn piece_is_clean(&self, start: usize, len: usize) -> bool {
        for offset in 0..len {
            let position = (start + offset) % self.n;
            let longest = (len - offset).min(len - 1);
            for sub in 2..=longest {
                if sub >= self.n {
                    break;
                }
                if self.table[position * (self.n + 1) + sub] {
                    return false;
                }
            }
        }
        true
    }
}

/// Split a legacy `[len][blob]...` value into its member circuits.
fn frozen_class_members(value: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut position = 0usize;
    while position < value.len() {
        let len = value[position] as usize;
        position += 1;
        if len == 0 || len % 3 != 0 || position + len > value.len() {
            break;
        }
        out.push(value[position..position + len].to_vec());
        position += len;
    }
    out
}

fn from_frozen_identities(
    input: &Path,
    output: &Path,
    good_splits: bool,
    wires: usize,
) -> AnyResult<()> {
    reject_lossy_canonicalization_env()?;
    require_fresh_output(output, "RocksDB")?;
    let input = path_text(input)?.to_string();
    let output_db = Arc::new(DB::open(&composite_options(true), path_text(output)?)?);
    put_format_marker(&output_db)?;
    let db = good_splits.then(|| local_mixing::db_mixing::frozen::FrozenDb::open(&input, None));
    let budget = local_mixing::engine::xpoly::XPolyBudget::default();

    let classes = AtomicU64::new(0);
    let identities = AtomicU64::new(0);
    let generated = AtomicU64::new(0);
    let unique = AtomicU64::new(0);
    let done = AtomicU64::new(0);
    let splits_seen = AtomicU64::new(0);
    let splits_kept = AtomicU64::new(0);
    let probes = AtomicU64::new(0);

    (0..256usize).into_par_iter().try_for_each(|shard| -> AnyResult<()> {
        let mut local_classes = 0u64;
        let mut local_identities = 0u64;
        let mut local_generated = 0u64;
        let mut local_unique = 0u64;
        let mut local_probes = 0u64;
        let mut local_splits_seen = 0u64;
        let mut local_splits_kept = 0u64;
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(
            0x9d_00_00 ^ shard as u64,
        );
        let mut error: Option<AnyError> = None;
        local_mixing::db_mixing::frozen::scan_shard(&input, shard, &mut |value| {
            if error.is_some() {
                return;
            }
            let members = frozen_class_members(value);
            if members.len() < 2 {
                return;
            }
            local_classes += 1;
            for left in 0..members.len() {
                for right in 0..members.len() {
                    if left == right {
                        continue;
                    }
                    let a = CircuitSeq::from_blob(&members[left]);
                    let b = CircuitSeq::from_blob(&members[right]);
                    let mut reverse_a = a.gates.clone();
                    reverse_a.reverse();
                    let mut reverse_b = b.gates.clone();
                    reverse_b.reverse();
                    let spellings = [
                        a.gates.iter().copied().chain(reverse_b).collect::<Vec<_>>(),
                        reverse_a
                            .into_iter()
                            .chain(b.gates.iter().copied())
                            .collect::<Vec<_>>(),
                    ];
                    for gates in spellings {
                        let identity = simplify_shortcut_identity(CircuitSeq { gates });
                        if identity.gates.len() < 3 {
                            continue;
                        }
                        local_identities += 1;
                        let result = match db.as_ref() {
                            None => write_identity_circuit(&output_db, &identity),
                            Some(db) => {
                                let n = identity.gates.len();
                                let table = CyclicCompress::probe(
                                    &identity,
                                    db,
                                    budget,
                                    wires,
                                    &mut rng,
                                    &mut local_probes,
                                );
                                // A split at `k` of the rotation starting at `r`
                                // cuts the cycle into [r, r+k) and [r+k, r+n).
                                // Reversal mirrors positions; compressibility is
                                // direction-symmetric, so the same table serves.
                                let accept = |reverse: bool, r: usize, k: usize| {
                                    local_splits_seen += 1;
                                    let start = if reverse { (n - r) % n } else { r };
                                    let ok = table.piece_is_clean(start, k)
                                        && table.piece_is_clean((start + k) % n, n - k);
                                    if ok {
                                        local_splits_kept += 1;
                                    }
                                    ok
                                };
                                write_identity_circuit_where(&output_db, &identity, accept)
                            }
                        };
                        match result {
                            Ok((raw, uniq)) => {
                                local_generated += raw;
                                local_unique += uniq;
                            }
                            Err(e) => {
                                error = Some(e);
                                return;
                            }
                        }
                    }
                }
            }
        });
        if let Some(e) = error {
            return Err(e);
        }
        classes.fetch_add(local_classes, Ordering::Relaxed);
        identities.fetch_add(local_identities, Ordering::Relaxed);
        generated.fetch_add(local_generated, Ordering::Relaxed);
        unique.fetch_add(local_unique, Ordering::Relaxed);
        probes.fetch_add(local_probes, Ordering::Relaxed);
        splits_seen.fetch_add(local_splits_seen, Ordering::Relaxed);
        splits_kept.fetch_add(local_splits_kept, Ordering::Relaxed);
        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
        if n % 8 == 0 || n == 256 {
            let seen = splits_seen.load(Ordering::Relaxed);
            let kept = splits_kept.load(Ordering::Relaxed);
            eprintln!(
                "[from-frozen-identities] shards={n}/256 classes={} identities={} generated={} local-unique={} splits={kept}/{seen} ({:.2}%) probes={}",
                classes.load(Ordering::Relaxed),
                identities.load(Ordering::Relaxed),
                generated.load(Ordering::Relaxed),
                unique.load(Ordering::Relaxed),
                kept as f64 * 100.0 / seen.max(1) as f64,
                probes.load(Ordering::Relaxed)
            );
        }
        Ok(())
    })?;

    if identities.load(Ordering::Relaxed) == 0 {
        return Err("frozen regular store produced no multi-circuit classes".into());
    }
    finalize_composite(&output_db, "from-frozen-identities")?;
    Ok(())
}

/// Sharded concurrent seen-set for identity-orbit hashes.
struct OrbitSeen {
    shards: Vec<std::sync::Mutex<rustc_hash::FxHashSet<u128>>>,
}

impl OrbitSeen {
    fn new() -> Self {
        Self {
            shards: (0..1024)
                .map(|_| std::sync::Mutex::new(rustc_hash::FxHashSet::default()))
                .collect(),
        }
    }
    /// True when the hash was not seen before (and is now claimed).
    fn insert(&self, hash: u128) -> bool {
        self.shards[(hash as usize) & 1023]
            .lock()
            .unwrap()
            .insert(hash)
    }
    fn len(&self) -> u64 {
        self.shards
            .iter()
            .map(|s| s.lock().unwrap().len() as u64)
            .sum()
    }
}

/// Derive all splits of one orbit-canonical identity, keeping the
/// lexicographically least candidate per (function key, gate count): within
/// one identity, same-length arcs under one key are rotation/reflection
/// siblings, and one representative carries all the structure.
fn write_identity_v2(db: &DB, canonical: &CircuitSeq) -> AnyResult<(u64, u64)> {
    let mut per_cell: std::collections::BTreeMap<([u8; FUNCTION_KEY_BYTES], usize), Vec<u8>> =
        std::collections::BTreeMap::new();
    let generated = derive_identity_candidates_where(
        canonical,
        |_, _, _| true,
        |key, blob| {
            let cell = (key, blob.len() / 3);
            match per_cell.get_mut(&cell) {
                Some(existing) => {
                    if blob < *existing {
                        *existing = blob;
                    }
                }
                None => {
                    per_cell.insert(cell, blob);
                }
            }
            Ok(())
        },
    )?;
    let unique = per_cell.len() as u64;
    let mut batch = WriteBatch::default();
    for ((key, _), blob) in per_cell {
        batch.put(composite_key(&key, &blob)?, []);
    }
    db.write(batch)?;
    Ok((generated, unique))
}

/// Simplify one glued or paired spelling into an identity word; None when it
/// collapses below 3 gates.
fn simplified_identity(gates: Vec<[u16; 3]>) -> Option<CircuitSeq> {
    let identity = simplify_shortcut_identity(CircuitSeq { gates });
    (identity.gates.len() >= 3).then_some(identity)
}

/// All two-control g57 gates over `wires + extra` wires touching at least one
/// wire below `wires`. Control order matters -- `[t, x, y]` fires on
/// (NOT x) AND y -- so ordered pairs cover both polarities.
fn gluing_gates(wires: u16, extra: u16) -> Vec<[u16; 3]> {
    let total = wires + extra;
    let mut out = Vec::new();
    for t in 0..total {
        for c1 in 0..total {
            for c2 in 0..total {
                if c1 != c2 && t != c1 && t != c2 && (t < wires || c1 < wires || c2 < wires) {
                    out.push([t, c1, c2]);
                }
            }
        }
    }
    out
}

fn from_frozen_identities_v2(
    input: &Path,
    output: &Path,
    glue_partner_composite: Option<&Path>,
    glue_source_keys: usize,
    glue_pairs: usize,
    glue_min_identity_gates: usize,
    shards: usize,
) -> AnyResult<()> {
    reject_lossy_canonicalization_env()?;
    require_fresh_output(output, "RocksDB")?;
    let input_text = path_text(input)?.to_string();
    let output_db = Arc::new(DB::open(&composite_options(true), path_text(output)?)?);
    put_format_marker(&output_db)?;

    let orbits = Arc::new(OrbitSeen::new());
    let classes = AtomicU64::new(0);
    let spellings = AtomicU64::new(0);
    let identities_new = AtomicU64::new(0);
    let generated = AtomicU64::new(0);
    let unique = AtomicU64::new(0);
    let done = AtomicU64::new(0);

    // Phase 1: same-class pair identities, orbit-deduplicated. One spelling
    // per ordered pair suffices: the other three historical spellings are the
    // same dihedral orbit, which the canonical word's derivation re-covers.
    let shard_count = shards.clamp(1, 256);
    (0..shard_count).into_par_iter().try_for_each(|shard| -> AnyResult<()> {
        let mut error: Option<AnyError> = None;
        local_mixing::db_mixing::frozen::scan_shard(&input_text, shard, &mut |value| {
            if error.is_some() {
                return;
            }
            let members = frozen_class_members(value);
            if members.len() < 2 {
                return;
            }
            classes.fetch_add(1, Ordering::Relaxed);
            let circuits: Vec<CircuitSeq> = members
                .iter()
                .map(|blob| CircuitSeq::from_blob(blob))
                .collect();
            for left in 0..circuits.len() {
                for right in 0..circuits.len() {
                    if left == right {
                        continue;
                    }
                    spellings.fetch_add(1, Ordering::Relaxed);
                    let mut gates = circuits[left].gates.clone();
                    gates.extend(circuits[right].gates.iter().rev().copied());
                    let Some(identity) = simplified_identity(gates) else {
                        continue;
                    };
                    let canonical = dihedral_canonical_word(&identity.gates);
                    let hash = xxhash_rust::xxh3::xxh3_128(&word_bytes(&canonical));
                    if !orbits.insert(hash) {
                        continue;
                    }
                    identities_new.fetch_add(1, Ordering::Relaxed);
                    match write_identity_v2(&output_db, &CircuitSeq { gates: canonical }) {
                        Ok((g, u)) => {
                            generated.fetch_add(g, Ordering::Relaxed);
                            unique.fetch_add(u, Ordering::Relaxed);
                        }
                        Err(e) => {
                            error = Some(e);
                            return;
                        }
                    }
                }
            }
        });
        if let Some(e) = error {
            return Err(e);
        }
        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
        if n % 16 == 0 || n as usize == shard_count {
            eprintln!(
                "[v2-phase1] shards={n}/{shard_count} classes={} spellings={} orbit-new={} generated={} written={}",
                classes.load(Ordering::Relaxed),
                spellings.load(Ordering::Relaxed),
                identities_new.load(Ordering::Relaxed),
                generated.load(Ordering::Relaxed),
                unique.load(Ordering::Relaxed)
            );
        }
        Ok(())
    })?;
    if identities_new.load(Ordering::Relaxed) == 0 {
        return Err("frozen regular store produced no identities".into());
    }

    // Phase 2: curated-x-curated gluing against a composite store.
    if let Some(partner_path) = glue_partner_composite {
        let partner = open_composite_read(partner_path)?;
        glue_phase(
            &output_db,
            &partner,
            &orbits,
            glue_source_keys,
            glue_pairs,
            glue_min_identity_gates,
        )?;
    }

    eprintln!(
        "[v2] orbit-unique identities total={} (phase 2 included)",
        orbits.len()
    );
    finalize_composite(&output_db, "from-frozen-identities-v2")?;
    Ok(())
}

/// Length-stratified candidate pool for one composite key: scan at most
/// `scan_cap` records, keep at most `per_len` per gate count.
fn key_pool(
    store: &DB,
    key: &[u8; FUNCTION_KEY_BYTES],
    scan_cap: usize,
    per_len: usize,
) -> AnyResult<Vec<Vec<u8>>> {
    let mut per_length: std::collections::BTreeMap<usize, usize> =
        std::collections::BTreeMap::new();
    let mut pool = Vec::new();
    let mut scanned = 0usize;
    for item in store.iterator(IteratorMode::From(key, rocksdb::Direction::Forward)) {
        let (record, _) = item?;
        if record.len() < FUNCTION_KEY_BYTES || record[..FUNCTION_KEY_BYTES] != key[..] {
            break;
        }
        scanned += 1;
        if let Ok((_, blob)) = split_composite_key(&record) {
            let gates = blob.len() / 3;
            let seen = per_length.entry(gates).or_insert(0);
            if *seen < per_len {
                *seen += 1;
                pool.push(blob.to_vec());
            }
        }
        if scanned >= scan_cap {
            break;
        }
    }
    Ok(pool)
}

fn glue_phase(
    output_db: &Arc<DB>,
    partner: &DB,
    orbits: &Arc<OrbitSeen>,
    source_keys: usize,
    pairs_per_connection: usize,
    min_identity_gates: usize,
) -> AnyResult<()> {
    use rand::Rng as _;

    let single_gate_key = CircuitSeq {
        gates: vec![[0, 1, 2]],
    }
    .canonicalize_polys_single_hashed(false)
    .0
    .ok_or("single CCX failed to canonicalize")?;

    // Composite keys are hashes, so seeking to a uniform random 16-byte point
    // and taking the next key samples keys uniformly.
    let mut source_key_set: BTreeSet<[u8; FUNCTION_KEY_BYTES]> = BTreeSet::new();
    {
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(0x91_ce);
        let mut attempts = 0usize;
        while source_key_set.len() < source_keys && attempts < source_keys * 3 {
            attempts += 1;
            let mut probe = [0u8; FUNCTION_KEY_BYTES];
            rng.fill(&mut probe);
            let mut it = partner.iterator(IteratorMode::From(&probe, rocksdb::Direction::Forward));
            if let Some(Ok((record, _))) = it.next() {
                if let Ok((key, _)) = split_composite_key(&record) {
                    source_key_set.insert(key);
                }
            }
        }
    }
    let sources: Vec<[u8; FUNCTION_KEY_BYTES]> = source_key_set.into_iter().collect();
    eprintln!("[v2-glue] sampled {} source keys", sources.len());

    let tried = AtomicU64::new(0);
    let connections = AtomicU64::new(0);
    let glued = AtomicU64::new(0);
    let kept = AtomicU64::new(0);
    let orbit_new = AtomicU64::new(0);
    let written = AtomicU64::new(0);
    let mismatched = AtomicU64::new(0);
    let done = AtomicU64::new(0);

    sources.par_iter().try_for_each(|source_key| -> AnyResult<()> {
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(
            0x91_ce ^ xxhash_rust::xxh3::xxh3_64(source_key),
        );
        let source_pool = key_pool(partner, source_key, 512, 4)?;
        let Some(a0_blob) = source_pool
            .iter()
            .min_by_key(|blob| (blob.len(), blob.as_slice()))
        else {
            return Ok(());
        };
        let a0 = CircuitSeq::from_blob(a0_blob);
        let wires = a0.gates.iter().flatten().copied().max().unwrap_or(0) + 1;
        for g in gluing_gates(wires, 2) {
            tried.fetch_add(1, Ordering::Relaxed);
            let mut a_plus = a0.clone();
            a_plus.gates.push(g);
            let (key2, perm2, used2) = a_plus.canonicalize_polys_single_hashed(false);
            let Some(key2) = key2 else { continue };
            let partner_pool = key_pool(partner, &key2, 512, 4)?;
            if partner_pool.is_empty() {
                continue;
            }
            connections.fetch_add(1, Ordering::Relaxed);

            let inverse = perm2.invert();
            let canon_to_orig: std::collections::HashMap<u16, u16> = used2
                .iter()
                .enumerate()
                .map(|(dense, &orig)| (inverse.data[dense] as u16, orig))
                .collect();

            // Harvest pairs biased toward long glued identities: every
            // (source, partner) combination whose raw total reaches the
            // threshold, randomly thinned to the per-connection budget.
            let mut combos: Vec<(usize, usize)> = Vec::new();
            for (ai, a_blob) in source_pool.iter().enumerate() {
                for (bi, b_blob) in partner_pool.iter().enumerate() {
                    if (a_blob.len() + b_blob.len()) / 3 + 1 >= min_identity_gates {
                        combos.push((ai, bi));
                    }
                }
            }
            while combos.len() > pairs_per_connection {
                let i = rng.random_range(0..combos.len());
                combos.swap_remove(i);
            }
            for (ai, bi) in combos {
                glued.fetch_add(1, Ordering::Relaxed);
                let a = CircuitSeq::from_blob(&source_pool[ai]);
                let b = CircuitSeq::from_blob(&partner_pool[bi]);
                let mut fresh: std::collections::HashMap<u16, u16> =
                    std::collections::HashMap::new();
                // Ancilla collisions with a's own extras would be harmless
                // (each block leaves its ancillas clean), but distinct fresh
                // indices keep the words easy to reason about.
                let a_top = a.gates.iter().flatten().copied().max().unwrap_or(0) + 1;
                let mut next_fresh = a_top.max(wires + 2);
                let mut map_wire = |w: u16| -> u16 {
                    if let Some(&orig) = canon_to_orig.get(&w) {
                        orig
                    } else {
                        *fresh.entry(w).or_insert_with(|| {
                            let v = next_fresh;
                            next_fresh += 1;
                            v
                        })
                    }
                };
                // rev(a) ++ b computes exactly g when the frames lined up;
                // verify against the canonical single-CCX key before trusting
                // the glue (the probe measured 100% pass, but a frame slip
                // must be skipped, not written).
                let mut m1_gates: Vec<[u16; 3]> = a.gates.iter().rev().copied().collect();
                m1_gates.extend(
                    b.gates
                        .iter()
                        .map(|&[t, c1, c2]| [map_wire(t), map_wire(c1), map_wire(c2)]),
                );
                let m1 = simplify_shortcut_identity(CircuitSeq { gates: m1_gates });
                if m1.gates.len() + 1 < min_identity_gates {
                    continue;
                }
                if m1.canonicalize_polys_single_hashed(false).0 != Some(single_gate_key) {
                    mismatched.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                let mut id_gates = m1.gates;
                id_gates.push(g);
                let Some(identity) = simplified_identity(id_gates) else {
                    continue;
                };
                if identity.gates.len() < min_identity_gates {
                    continue;
                }
                kept.fetch_add(1, Ordering::Relaxed);
                let canonical = dihedral_canonical_word(&identity.gates);
                let hash = xxhash_rust::xxh3::xxh3_128(&word_bytes(&canonical));
                if !orbits.insert(hash) {
                    continue;
                }
                orbit_new.fetch_add(1, Ordering::Relaxed);
                match write_identity_v2(output_db, &CircuitSeq { gates: canonical }) {
                    Ok((_, u)) => {
                        written.fetch_add(u, Ordering::Relaxed);
                    }
                    Err(e)
                        if e.downcast_ref::<CuratedError>()
                            .is_some_and(|c| matches!(c, CuratedError::EquivalenceMismatch { .. })) =>
                    {
                        // Opportunistic material: a bad glue is dropped, the
                        // build continues. Nothing was written (validation
                        // runs before the batch write).
                        mismatched.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(e) => return Err(e),
                }
            }
        }
        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
        if n % 10_000 == 0 || n as usize == sources.len() {
            eprintln!(
                "[v2-glue] {n}/{} sources tried={} connections={} glued={} long-kept={} orbit-new={} written={} mismatched={}",
                sources.len(),
                tried.load(Ordering::Relaxed),
                connections.load(Ordering::Relaxed),
                glued.load(Ordering::Relaxed),
                kept.load(Ordering::Relaxed),
                orbit_new.load(Ordering::Relaxed),
                written.load(Ordering::Relaxed),
                mismatched.load(Ordering::Relaxed)
            );
        }
        Ok(())
    })?;
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

/// Relabelled subword hashes of `word` at length `len`, both directions.
fn subword_hashes(word: &[[u16; 3]], len: usize, out: &mut Vec<u64>) {
    use local_mixing::db_generation::curated_full::relabel_word;
    out.clear();
    let n = word.len();
    if len > n {
        return;
    }
    let mut reversed: Vec<[u16; 3]> = word.to_vec();
    reversed.reverse();
    for seq in [word, reversed.as_slice()] {
        for start in 0..=(n - len) {
            out.push(xxhash_rust::xxh3::xxh3_64(&word_bytes(&relabel_word(
                &seq[start..start + len],
            ))));
        }
    }
}

struct SieveStats {
    keys: u64,
    passed_keys: u64,
    sieved_keys: u64,
    in_candidates: u64,
    out_candidates: u64,
    floor_kept: u64,
}

fn sieve_key(
    out: &DB,
    key: &[u8; FUNCTION_KEY_BYTES],
    blobs: &mut Vec<Vec<u8>>,
    shingle: usize,
    keep_all_below: u64,
    cell_floor: usize,
    stats: &mut SieveStats,
) -> AnyResult<()> {
    stats.keys += 1;
    stats.in_candidates += blobs.len() as u64;
    let mut batch = WriteBatch::default();
    if blobs.len() as u64 <= keep_all_below {
        stats.passed_keys += 1;
        stats.out_candidates += blobs.len() as u64;
        for blob in blobs.iter() {
            batch.put(composite_key(key, blob)?, []);
        }
        out.write(batch)?;
        return Ok(());
    }
    stats.sieved_keys += 1;
    // Shortest first, then lexicographic: short spellings are scarce and must
    // not be crowded out by long candidates claiming their fragments.
    blobs.sort_unstable_by(|x, y| (x.len(), x.as_slice()).cmp(&(y.len(), y.as_slice())));
    let words: Vec<(Vec<[u16; 3]>, usize)> = blobs
        .par_iter()
        .map(|blob| {
            let word: Vec<[u16; 3]> = blob
                .chunks_exact(3)
                .map(|g| [g[0] as u16, g[1] as u16, g[2] as u16])
                .collect();
            let mut wires: Vec<u16> = word.iter().flatten().copied().collect();
            wires.sort_unstable();
            wires.dedup();
            let w = wires.len();
            (word, w)
        })
        .collect();
    let hashes: Vec<Vec<u64>> = words
        .par_iter()
        .map(|(word, _)| {
            let mut v = Vec::new();
            subword_hashes(word, shingle, &mut v);
            v
        })
        .collect();

    let mut claimed: rustc_hash::FxHashSet<u64> = rustc_hash::FxHashSet::default();
    let mut cells: rustc_hash::FxHashMap<(usize, usize), usize> = rustc_hash::FxHashMap::default();
    let mut written = 0usize;
    for ((blob, (word, wires)), subwords) in blobs.iter().zip(&words).zip(&hashes) {
        let cell = (word.len(), *wires);
        let under_floor = *cells.get(&cell).unwrap_or(&0) < cell_floor;
        let novel = subwords.is_empty() || !subwords.iter().any(|h| claimed.contains(h));
        if !(under_floor || novel) {
            continue;
        }
        if under_floor && !novel {
            stats.floor_kept += 1;
        }
        *cells.entry(cell).or_insert(0) += 1;
        for &h in subwords {
            claimed.insert(h);
        }
        batch.put(composite_key(key, blob)?, []);
        written += 1;
        if written % 65_536 == 0 {
            out.write(std::mem::take(&mut batch))?;
        }
    }
    stats.out_candidates += written as u64;
    out.write(batch)?;
    Ok(())
}

fn sieve(
    input: &Path,
    output: &Path,
    shingle: usize,
    keep_all_below: u64,
    cell_floor: usize,
) -> AnyResult<()> {
    if shingle < 2 {
        return Err("--shingle must be at least 2".into());
    }
    let source = open_composite_read(input)?;
    let source_stats = verify_manifest(&source)?;
    require_fresh_output(output, "RocksDB")?;
    let out = DB::open(&composite_options(true), path_text(output)?)?;
    put_format_marker(&out)?;

    let mut stats = SieveStats {
        keys: 0,
        passed_keys: 0,
        sieved_keys: 0,
        in_candidates: 0,
        out_candidates: 0,
        floor_kept: 0,
    };
    let mut current: Option<[u8; FUNCTION_KEY_BYTES]> = None;
    let mut blobs: Vec<Vec<u8>> = Vec::new();
    for item in source.iterator(IteratorMode::Start) {
        let (record, _) = item?;
        if is_composite_metadata(&record) {
            continue;
        }
        let (key, blob) = split_composite_key(&record)?;
        if current != Some(key) {
            if let Some(previous) = current {
                sieve_key(
                    &out,
                    &previous,
                    &mut blobs,
                    shingle,
                    keep_all_below,
                    cell_floor,
                    &mut stats,
                )?;
                blobs.clear();
                if stats.keys % 1_000_000 == 0 {
                    eprintln!(
                        "[sieve] keys={} in={} out={} ({} sieved keys)",
                        stats.keys, stats.in_candidates, stats.out_candidates, stats.sieved_keys
                    );
                }
            }
            current = Some(key);
        }
        blobs.push(blob.to_vec());
    }
    if let Some(previous) = current {
        sieve_key(
            &out,
            &previous,
            &mut blobs,
            shingle,
            keep_all_below,
            cell_floor,
            &mut stats,
        )?;
    }
    if stats.in_candidates != source_stats.candidates {
        return Err(format!(
            "sieve consumed {} candidates but the source manifest says {}",
            stats.in_candidates, source_stats.candidates
        )
        .into());
    }
    eprintln!(
        "[sieve] complete: keys={} passed={} sieved={} in={} out={} ({:.2}%) floor-kept={}",
        stats.keys,
        stats.passed_keys,
        stats.sieved_keys,
        stats.in_candidates,
        stats.out_candidates,
        stats.out_candidates as f64 * 100.0 / stats.in_candidates.max(1) as f64,
        stats.floor_kept
    );
    finalize_composite(&out, "sieve")?;
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
    let mut current_min_blob: Option<Vec<u8>> = None;
    let mut max_candidate_key = None;
    let mut max_candidate_min_blob = None;
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
            if let Some(previous) = previous_key {
                if current_candidates > max_candidates {
                    max_candidates = current_candidates;
                    max_candidate_key = Some(previous);
                    max_candidate_min_blob = current_min_blob.take();
                }
                max_value_bytes = max_value_bytes.max(current_value_bytes);
            }
            keys += 1;
            current_candidates = 0;
            current_value_bytes = 0;
            current_min_blob = None;
            previous_key = Some(key);
        }
        candidates += 1;
        current_candidates += 1;
        current_value_bytes += (blob.len() + 1) as u64;
        if current_min_blob
            .as_ref()
            .is_none_or(|minimum| blob.len() < minimum.len())
        {
            current_min_blob = Some(blob.to_vec());
        }
        max_blob_bytes = max_blob_bytes.max(blob.len());
    }
    if let Some(previous) = previous_key {
        if current_candidates > max_candidates {
            max_candidates = current_candidates;
            max_candidate_key = Some(previous);
            max_candidate_min_blob = current_min_blob;
        }
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
    if let (Some(key), Some(blob)) = (max_candidate_key, max_candidate_min_blob) {
        eprintln!(
            "[audit] max-candidate-key={} min-candidate-gates={} min-candidate-blob={}",
            key.iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>(),
            blob.len() / 3,
            blob.iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        );
    }
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
        Command::FromFrozenIdentities {
            frozen_regular,
            output_rocks,
            good_splits,
            wires,
        } => from_frozen_identities(&frozen_regular, &output_rocks, good_splits, wires),
        Command::FromFrozenIdentitiesV2 {
            frozen_regular,
            output_rocks,
            glue_partner_composite,
            glue_source_keys,
            glue_pairs,
            glue_min_identity_gates,
            shards,
        } => from_frozen_identities_v2(
            &frozen_regular,
            &output_rocks,
            glue_partner_composite.as_deref(),
            glue_source_keys,
            glue_pairs,
            glue_min_identity_gates,
            shards,
        ),
        Command::Sieve {
            input_rocks,
            output_rocks,
            shingle,
            keep_all_below,
            cell_floor,
        } => sieve(
            &input_rocks,
            &output_rocks,
            shingle,
            keep_all_below,
            cell_floor,
        ),
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
