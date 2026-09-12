//! Exact G57 window cache and its memory budget.
use super::Polynomial;
pub(super) use super::legacy_environment::canon_cache_cap_bytes;
use std::sync::OnceLock;
use std::sync::atomic::AtomicU64;
// ---------------------------------------------------------------------------
// Canonicalization result cache.
//
// canonicalize_polys_single is pure: the canonical polys and final order are
// fully determined by the dense-remapped, gate-canonicalized window. The
// expansion/compression games draw the same windows over and over (SAMF
// templates recur all over the circuit), so an exact process-wide cache
// skips to_polynomial + canonicalize_polys_4 entirely on repeats.
//
// CANON_CACHE_MB caps the approximate entry-byte footprint (default 256 MiB;
// 0 disables). On overflow the cache is cleared wholesale, mirroring the
// Frozen lookup cache's epoch-reset policy.
// ---------------------------------------------------------------------------
pub(super) struct CanonCacheEntry {
    pub(super) polys: Vec<Polynomial>,
    pub(super) order: Vec<usize>,
    /// `xxh3_128(polys_repr_blob(polys)).to_le_bytes()`, precomputed at insert
    /// time so hashed lookups can return the frozen-DB key on a cache hit
    /// without deep-cloning `polys` and re-serializing/re-hashing them.
    pub(super) polys_key: [u8; 16],
}

/// Outcome of the shared canonicalize-single core, before the public wrappers
/// shape it into their respective return types.
pub(super) enum CanonSingleInner {
    /// Oversized window, monomial-cap skip, or Rule-L budget skip: the
    /// canonical result is the empty sentinel.
    Skip,
    /// Exact-cache hit; the entry holds polys, order, and the precomputed key.
    Cached(std::sync::Arc<CanonCacheEntry>),
    /// Freshly computed `(polys, order, polys_key)`. The key is `Some` iff the
    /// result was just inserted into the cache (where it is computed anyway).
    Fresh(Vec<Polynomial>, Vec<usize>, Option<[u8; 16]>),
}

pub(super) type CanonCacheMap =
    dashmap::DashMap<Box<[u16]>, std::sync::Arc<CanonCacheEntry>, rustc_hash::FxBuildHasher>;

pub(super) static CANON_CACHE_BYTES: AtomicU64 = AtomicU64::new(0);
pub static CANON_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
pub static CANON_CACHE_QUERIES: AtomicU64 = AtomicU64::new(0);

pub(super) fn canon_cache() -> Option<&'static CanonCacheMap> {
    static CACHE: OnceLock<Option<CanonCacheMap>> = OnceLock::new();
    CACHE
        .get_or_init(|| (canon_cache_cap_bytes() > 0).then(CanonCacheMap::default))
        .as_ref()
}
