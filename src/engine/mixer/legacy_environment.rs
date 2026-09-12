//! Compatibility for historical Mixer constructors and direct research commands.
//! Keep the first-read timing and process-wide caching of each old control.
//! Typed runtime options bypass this adapter; engine code does not depend on CLI modules.
pub(crate) fn tg_slide_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("TWIST_G57_NO_SLIDE").is_none())
}

pub(crate) fn tg_retry_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("TWIST_G57_NO_RETRY").is_none())
}

/// Read-only reference store for class attribution (FROZEN_REF_DIR): opened
/// once, queried as a REGULAR store — only spelling LENGTHS are read, and
/// those are convention-independent.
pub(crate) fn reference_db() -> Option<&'static crate::database::frozen::FrozenDb> {
    static REF: std::sync::OnceLock<Option<crate::database::frozen::FrozenDb>> =
        std::sync::OnceLock::new();
    REF.get_or_init(|| {
        let dir = std::env::var("FROZEN_REF_DIR").ok()?;
        println!("[fmix] class-attribution reference store: {dir}");
        Some(crate::database::frozen::FrozenDb::open(&dir, None))
    })
    .as_ref()
}

pub(crate) fn stop_at_phase() -> Option<u32> {
    static STOP_AT: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *STOP_AT.get_or_init(|| {
        std::env::var("FMIX_STOP_AT_PHASE")
            .ok()
            .and_then(|v| v.parse().ok())
    })
}
