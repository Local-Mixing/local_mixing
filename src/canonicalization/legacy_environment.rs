//! Compatibility adapter for historical process-wide canonicalization controls.
//! Each control keeps its original independent OnceLock, parsing and first-use
//! timing. Explicit computation APIs bypass these reads and the legacy caches.
//! The GSS driver pins its recorded environment; see docs/formats/checkpoints.md.
use std::sync::OnceLock;

pub fn canon_rule_l_branch_cap() -> Option<u64> {
    static CAP: OnceLock<Option<u64>> = OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("CANON_RULE_L_BRANCH_CAP")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .filter(|&cap| cap > 0)
    })
}

pub fn canon_monomial_cap() -> Option<usize> {
    static CAP: OnceLock<Option<usize>> = OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("CANON_MONOMIAL_CAP")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|&cap| cap > 0)
    })
}

#[cfg(feature = "legacy-tools")]
pub(super) fn bench_canon_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("BENCH_CANON").is_ok())
}

pub(super) fn canon_cache_cap_bytes() -> u64 {
    static CAP: OnceLock<u64> = OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("CANON_CACHE_MB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(256)
            .saturating_mul(1024 * 1024)
    })
}

pub(super) fn compression_trace_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("COMPRESSION_TRACE").is_ok())
}

pub(super) fn compression_trace_threshold_ms() -> u128 {
    static THRESHOLD: OnceLock<u128> = OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        std::env::var("COMPRESSION_TRACE_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1_000)
    })
}

pub(super) fn xpoly_canon_cache_cap_bytes() -> u64 {
    static CAP: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("XPOLY_CANON_CACHE_MB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(1024)
            .saturating_mul(1024 * 1024)
    })
}
