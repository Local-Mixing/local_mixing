//! Process-wide mixer controls used by the TDP stage executables.
//! Each control is cached independently when first read. Explicit runtime
//! options bypass these settings; engine code does not depend on CLI modules.
pub(crate) fn tg_slide_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("TWIST_G57_NO_SLIDE").is_none())
}

pub(crate) fn tg_retry_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("TWIST_G57_NO_RETRY").is_none())
}

pub(crate) fn stop_at_phase() -> Option<u32> {
    static STOP_AT: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *STOP_AT.get_or_init(|| {
        std::env::var("MIXER_STOP_AT_PHASE")
            .ok()
            .and_then(|v| v.parse().ok())
    })
}

/// MIXER_STOP_AT_SPLICES=<n>: finish cleanly once cumulative successful
/// slot-2 DB splices reach n (exact: checked every move, not at report
/// cadence). For matched-splice-budget comparisons across experiment arms.
pub(crate) fn stop_at_splices() -> Option<u64> {
    static STOP_AT: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *STOP_AT.get_or_init(|| {
        std::env::var("MIXER_STOP_AT_SPLICES")
            .ok()
            .and_then(|v| v.parse().ok())
    })
}
