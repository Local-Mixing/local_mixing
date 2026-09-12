//! Optional resolved runtime controls, with an explicit lazy legacy adapter.
use super::*;

#[derive(Clone)]
pub(super) enum RuntimeControls {
    LegacyEnvironment,
    Resolved(Arc<MixRuntimeOptions>),
}

impl RuntimeControls {
    pub(super) fn twist_slide(&self) -> bool {
        match self {
            Self::LegacyEnvironment => legacy_environment::tg_slide_on(),
            Self::Resolved(options) => options.twist_g57_slide,
        }
    }
    pub(super) fn twist_retry(&self) -> bool {
        match self {
            Self::LegacyEnvironment => legacy_environment::tg_retry_on(),
            Self::Resolved(options) => options.twist_g57_retry,
        }
    }
    pub(super) fn stop_at_phase(&self) -> Option<u32> {
        match self {
            Self::LegacyEnvironment => legacy_environment::stop_at_phase(),
            Self::Resolved(options) => options.stop_at_phase,
        }
    }
    pub(super) fn reference_db(&self) -> Option<&FrozenDb> {
        match self {
            Self::LegacyEnvironment => legacy_environment::reference_db(),
            Self::Resolved(options) => options.reference_db.as_deref(),
        }
    }
}

impl Mixer {
    /// Construct with an explicit store and resolved twist, phase-stop and
    /// reference-store controls. Database-backed walks retain the historical
    /// replacement/cache policy; use the explicit replacement APIs when that
    /// policy must also be supplied without environment configuration.
    pub fn new_with_runtime_options(
        gates: Vec<XGate>,
        num_wires: usize,
        params: MixParams,
        db: Arc<FrozenDb>,
        runtime: MixRuntimeOptions,
    ) -> Mixer {
        let mut mixer = Self::new_with_shared_db(gates, num_wires, params, db);
        mixer.set_runtime_options(runtime);
        mixer
    }

    /// Replace the twist, phase-stop and reference-store environment adapter,
    /// including on a resumed mixer. Replacement/cache policy is unchanged.
    /// Piecewise rounds inherit these controls without changing the RNG stream.
    pub fn set_runtime_options(&mut self, runtime: MixRuntimeOptions) {
        self.runtime = RuntimeControls::Resolved(Arc::new(runtime));
    }
}
