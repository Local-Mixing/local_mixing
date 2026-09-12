//! Shared immutable-store cache and canonical lookup direction policy.

use super::legacy_environment::lookup_cache_cap_bytes;
pub(crate) use super::legacy_environment::min_dir_lookup_mode;
use crate::database::frozen::FrozenDb;
use dashmap::DashMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Frozen-store lookup cache.
//
// The frozen database is immutable for the lifetime of the
// process, so a lookup's result can never change: caching (key -> value bytes,
// including "absent") is exact, not approximate. Windows drawn by the
// expansion/compression games repeat heavily (SAMF templates recur all over
// the circuit), and most canonical keys MISS the DB, so caching negative
// results is as valuable as positive ones. On hosts whose RAM is smaller than
// the DB, each avoided lookup is an avoided page fault.
//
// LOOKUP_CACHE_MB caps the approximate value-byte footprint (default 512 MiB;
// 0 disables the cache). When the cap is exceeded the cache is cleared
// wholesale — an epoch reset is cheaper and simpler than LRU bookkeeping and
// the working set refills within seconds.
// ---------------------------------------------------------------------------
// Cache keys are namespaced: byte 0 distinguishes which logical database the
// lookup targeted (shard vs curated); the same 16-byte canonical hash may
// exist in both with different values.
type LookupCacheMap = DashMap<[u8; 17], Option<std::sync::Arc<[u8]>>, rustc_hash::FxBuildHasher>;

pub(crate) const LOOKUP_NS_SHARD: u8 = 0;
pub(crate) const LOOKUP_NS_CURATED: u8 = 1;

static LOOKUP_CACHE_BYTES: AtomicU64 = AtomicU64::new(0);
pub static LOOKUP_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
pub static LOOKUP_CACHE_QUERIES: AtomicU64 = AtomicU64::new(0);

fn lookup_cache() -> Option<&'static LookupCacheMap> {
    static CACHE: OnceLock<Option<LookupCacheMap>> = OnceLock::new();
    CACHE
        .get_or_init(|| (lookup_cache_cap_bytes() > 0).then(LookupCacheMap::default))
        .as_ref()
}

/// Fetch one namespaced key from the immutable frozen database.
fn raw_db_get(db: &FrozenDb, namespace: u8, key: &[u8; 16]) -> Option<std::sync::Arc<[u8]>> {
    let value = if namespace == LOOKUP_NS_CURATED {
        db.get_curated(key)
    } else {
        db.get_regular(key)
    };
    value.map(std::sync::Arc::from)
}

/// Point lookup with an exact process-wide cache in front. Returns the value
/// bytes (shared, immutable) or None when the key is absent — byte-identical
/// to the uncached lookup on a read-only environment.
pub(crate) fn cached_db_get(
    db: &FrozenDb,
    namespace: u8,
    key: &[u8; 16],
) -> Option<std::sync::Arc<[u8]>> {
    cached_db_get_using(
        db,
        namespace,
        key,
        lookup_cache(),
        &LOOKUP_CACHE_BYTES,
        lookup_cache_cap_bytes,
    )
}

fn cached_db_get_using(
    db: &FrozenDb,
    namespace: u8,
    key: &[u8; 16],
    cache: Option<&LookupCacheMap>,
    bytes: &AtomicU64,
    cap_bytes: impl Fn() -> u64,
) -> Option<std::sync::Arc<[u8]>> {
    let Some(cache) = cache else {
        return raw_db_get(db, namespace, key);
    };
    let mut ns_key = [0u8; 17];
    ns_key[0] = namespace;
    ns_key[1..].copy_from_slice(key);
    LOOKUP_CACHE_QUERIES.fetch_add(1, Ordering::Relaxed);
    if let Some(entry) = cache.get(&ns_key) {
        LOOKUP_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
        return entry.clone();
    }
    let result: Option<std::sync::Arc<[u8]>> = raw_db_get(db, namespace, key);
    // A complete curated value can contain hundreds of thousands of
    // candidates and occupy many megabytes (the historical worst case was
    // multi-megabyte, and one historical shard exceeded a gigabyte). Keep exact
    // curated misses in the cache, but do not pin a positive full value there:
    // callers already hold it during selection, and retaining it would let a
    // few hot full-coverage keys evict the regular working set or exhaust the
    // process. Regular positives remain cached; their friend lists are small.
    if namespace == LOOKUP_NS_CURATED && result.is_some() {
        return result;
    }
    let entry_bytes = 17 + 64 + result.as_ref().map_or(0, |v| v.len()) as u64;
    if bytes.fetch_add(entry_bytes, Ordering::Relaxed) + entry_bytes > cap_bytes() {
        cache.clear();
        bytes.store(entry_bytes, Ordering::Relaxed);
    }
    cache.insert(ns_key, result.clone());
    result
}

// ---------------------------------------------------------------------------
// Lookup direction strategy.
//
// The shard DBs are keyed by the *minimum* of a circuit's forward and reverse
// canonical polynomial forms (build_from_rocks inserts min(canon_fwd,
// canon_rev) only). Since canonical polys are determined by the function a
// circuit computes, a window's non-min direction key can only exist in the DB
// if it equals the min key. Probing just the min direction is therefore
// exactly equivalent to the legacy forward-then-reverse probe — and it halves
// the number of cold frozen-store probes on misses, which dominate when the DB is
// larger than RAM.
//
// MIN_DIR_LOOKUP=0        -> legacy: forward probe, reverse probe on miss.
// MIN_DIR_LOOKUP=validate -> min-direction probe, but on a miss also probe the
//                            other direction and count/log any hit (which
//                            would disprove the min-key invariant).
// unset / any other value -> min-direction probe only (default).
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MinDirLookup {
    Legacy,
    #[default]
    Min,
    Validate,
}

pub static MIN_DIR_VALIDATE_PROBES: AtomicU64 = AtomicU64::new(0);
pub static MIN_DIR_VIOLATIONS: AtomicU64 = AtomicU64::new(0);

/// An exact cache bound to one immutable database handle. Configuration and
/// cache lifetime belong to the caller, so neither can leak between stores.
pub struct LookupCache<'db> {
    db: &'db FrozenDb,
    map: Option<LookupCacheMap>,
    bytes: AtomicU64,
    cap_bytes: u64,
}

impl<'db> LookupCache<'db> {
    /// `cap_bytes == 0` disables caching. The cap uses the same approximate
    /// entry footprint and whole-cache eviction as the legacy process cache.
    pub fn new(db: &'db FrozenDb, cap_bytes: u64) -> Self {
        Self {
            db,
            map: (cap_bytes > 0).then(LookupCacheMap::default),
            bytes: AtomicU64::new(0),
            cap_bytes,
        }
    }
    pub fn get_regular(&self, key: &[u8; 16]) -> Option<std::sync::Arc<[u8]>> {
        self.get(LOOKUP_NS_SHARD, key)
    }
    pub fn get_curated(&self, key: &[u8; 16]) -> Option<std::sync::Arc<[u8]>> {
        self.get(LOOKUP_NS_CURATED, key)
    }
    fn get(&self, namespace: u8, key: &[u8; 16]) -> Option<std::sync::Arc<[u8]>> {
        cached_db_get_using(
            self.db,
            namespace,
            key,
            self.map.as_ref(),
            &self.bytes,
            || self.cap_bytes,
        )
    }
}

#[cfg(test)]
#[path = "../../tests/database/lookup_cache.rs"]
mod tests;
