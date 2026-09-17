//! Replacement-selection environment settings for the TDP database stage.
//! The length band is cached on the first eligible candidate; explicit
//! replacement options bypass this setting.
pub(super) fn len_band() -> Option<(usize, usize)> {
    static B: std::sync::OnceLock<Option<(usize, usize)>> = std::sync::OnceLock::new();
    *B.get_or_init(|| {
        let v = std::env::var("MIXER_DB_LEN_BAND").ok()?;
        let p: Vec<usize> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        (p.len() == 2 && p[0] <= p[1]).then(|| (p[0], p[1]))
    })
}
