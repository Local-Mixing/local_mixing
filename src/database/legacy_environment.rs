//! Historical storage configuration boundary. Preserve independent first-read
//! caching and the original open-time read order; typed APIs bypass this module.
//! Compatibility contract: [saved runs](../../docs/formats/checkpoints.md).
use super::frozen::ValueConvention;
use super::lookup_cache::MinDirLookup;
use std::sync::OnceLock;

/// Per-store value convention, from `FROZEN_REGULAR_VALUE_CONVENTION` /
/// `FROZEN_CURATED_VALUE_CONVENTION`. `native` (default) returns values as
/// stored; `legacy-swapped-controls` swaps each gate's two controls at decode.
pub(super) fn value_convention(var: &str) -> ValueConvention {
    match std::env::var(var).as_deref() {
        Err(_) | Ok("native") => ValueConvention::Native,
        Ok("legacy-swapped-controls") => ValueConvention::LegacySwappedControls,
        Ok(other) => {
            panic!("{var}={other}: unknown value convention (native | legacy-swapped-controls)")
        }
    }
}

pub(super) fn filters_enabled() -> bool {
    std::env::var("FROZEN_FILTER").map(|v| v == "1") == Ok(true)
}

pub(super) fn frozen_directories() -> (String, Option<String>) {
    let regular = std::env::var("FROZEN_DB_DIR")
        .expect("FROZEN_DB_DIR is required; the runtime is frozen-store only");
    let curated = std::env::var("FROZEN_CURATED_DIR").ok();
    (regular, curated)
}
pub(super) fn lookup_cache_cap_bytes() -> u64 {
    static CAP: OnceLock<u64> = OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("LOOKUP_CACHE_MB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(512)
            .saturating_mul(1024 * 1024)
    })
}

pub(crate) fn min_dir_lookup_mode() -> MinDirLookup {
    static MODE: OnceLock<MinDirLookup> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("MIN_DIR_LOOKUP").as_deref() {
        Ok("0") => MinDirLookup::Legacy,
        Ok("validate") => MinDirLookup::Validate,
        _ => MinDirLookup::Min,
    })
}
