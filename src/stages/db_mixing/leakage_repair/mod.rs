//! Optional gate-wise leakage audit and surgical DB repair, independent of the
//! ordinary mixing sampler. See docs/DB_QUALITY_CONTROL.md for statistical scope.

pub mod blocks;
pub mod detect;

use crate::canonicalization::xgate::XPolyBudget;
use crate::circuit::xgate::XGate;
use crate::engine::mixer::Mixer;
use crate::stages::db_mixing::replacement::{QcCandidateLimits, QcCandidates, qc_candidates};
use blocks::{BlockError, BlockLimits, plan_convex_block};
use detect::{Detector, DetectorConfig, ScanReport};
use rand::{SeedableRng, rngs::StdRng};

#[derive(Clone, Debug)]
pub struct QualityConfig {
    pub detector: DetectorConfig,
    pub block_limits: BlockLimits,
    pub candidate_limits: QcCandidateLimits,
    pub polynomial_budget: XPolyBudget,
    pub max_attempts: usize,
    pub max_repairs: usize,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            detector: DetectorConfig::default(),
            block_limits: BlockLimits {
                max_span: 256,
                max_gates: 12,
                max_support: 24,
            },
            candidate_limits: QcCandidateLimits {
                max_candidates: 64,
                max_gates: 24,
                max_support: 24,
            },
            polynomial_budget: XPolyBudget {
                max_mul_terms: 1 << 16,
                max_poly_terms: 4096,
                max_total_terms: 16384,
            },
            max_attempts: 32,
            max_repairs: 8,
        }
    }
}

impl QualityConfig {
    pub fn validate(&self) -> Result<(), String> {
        self.detector.validate()?;
        if self.max_attempts == 0 || self.max_repairs == 0 {
            return Err("QC attempt and repair budgets must be positive".into());
        }
        if self.block_limits.max_span == 0
            || self.block_limits.max_gates < 2
            || self.block_limits.max_support == 0
            || self.block_limits.max_support > 64
            || self.candidate_limits.max_candidates == 0
            || self.candidate_limits.max_gates == 0
            || self.candidate_limits.max_support == 0
            || self.candidate_limits.max_support > 64
        {
            return Err("QC needs positive budgets, >=2 block gates and support in 1..=64".into());
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct QualityEvent {
    pub wire: u16,
    pub start_gate: usize,
    pub end_gate: usize,
    pub status: String,
    pub detail: String,
}

#[derive(Debug)]
pub struct QualityReport {
    pub before: ScanReport,
    pub after: ScanReport,
    /// Independent samples unused by repair selection; findings are reported
    /// without adapting the output again to this final audit bank.
    pub fresh_audit: ScanReport,
    pub repairs: usize,
    pub budget_exhausted: bool,
    pub events: Vec<QualityEvent>,
    pub configuration: String,
}

impl QualityReport {
    pub fn to_text(&self) -> String {
        let mut out = format!(
            "DB gate-wise quality control v1\nconfiguration: {}\nrepairs: {}\nattempts: {}\nbudget_exhausted: {}\n\
             before: {:#?}\nafter: {:#?}\nfresh_audit: {:#?}\n\
             Scope: sampled empirical probes; a clean result is not a security proof.\n\
             Failure outcomes describe this minimal convex block and this bounded DB search only.\n\
             wire\tstart_gate\tend_gate\tstatus\tdetail\n",
            self.configuration,
            self.repairs,
            self.events.len(),
            self.budget_exhausted,
            self.before,
            self.after,
            self.fresh_audit,
        );
        for e in &self.events {
            out.push_str(&format!(
                "{}\t{}\t{}\t{}\t{}\n",
                e.wire,
                e.start_gate,
                e.end_gate,
                e.status,
                e.detail.replace(['\t', '\n', '\r'], " ")
            ));
        }
        out
    }
}

/// Run after the ordinary walk and its final float, before saving the db_mixing
/// artifact. No ordinary sampler configuration or random stream is changed.
/// The reference uses the same wire/input coordinates as the mixed circuit;
/// callers may explicitly supply a pre-mixing original instead of the walk input.
pub fn run_quality_control(
    mixer: &mut Mixer,
    reference: Option<&[XGate]>,
    config: &QualityConfig,
) -> Result<QualityReport, String> {
    let db = mixer.shared_db();
    run_quality_control_with(
        mixer,
        reference,
        config,
        |window, num_wires, budget, limits, rng| {
            qc_candidates(window, num_wires, &db, budget, limits, rng)
        },
    )
}

/// Test seam retains the real detector, planner, screening and Mixer splice.
/// Production always supplies the exact-equivalence-checking QC DB lookup.
fn run_quality_control_with<F>(
    mixer: &mut Mixer,
    reference: Option<&[XGate]>,
    config: &QualityConfig,
    mut lookup: F,
) -> Result<QualityReport, String>
where
    F: FnMut(&[XGate], usize, XPolyBudget, QcCandidateLimits, &mut StdRng) -> QcCandidates,
{
    config.validate()?;
    let detector = Detector::new(
        reference.unwrap_or_else(|| mixer.quality_reference()),
        mixer.quality_num_wires(),
        config.detector.clone(),
    )?;
    let mut current = mixer.arena.to_vec();
    let mut scan = detector.scan(&current)?;
    let mut report = QualityReport {
        after: scan.clone(),
        before: scan.clone(),
        fresh_audit: scan.clone(),
        repairs: 0,
        budget_exhausted: false,
        events: Vec::new(),
        configuration: format!("{config:?}"),
    };
    let mut rng = StdRng::seed_from_u64(config.detector.seed ^ 0x5143_4442_7265_7061);
    loop {
        if scan.hot_segments.is_empty() {
            report.after = scan;
            break;
        }
        if report.events.len() >= config.max_attempts || report.repairs >= config.max_repairs {
            report.after = scan;
            report.budget_exhausted = true;
            break;
        }
        let mut changed = false;
        for hot in &scan.hot_segments {
            if report.events.len() >= config.max_attempts {
                report.budget_exhausted = true;
                break;
            }
            let mut event = QualityEvent {
                wire: hot.wire,
                start_gate: hot.start_gate,
                end_gate: hot.end_gate,
                status: String::new(),
                detail: format!("evidence={:?}", hot.evidence),
            };
            let plan = match plan_convex_block(
                &current,
                hot.start_gate,
                hot.end_gate,
                config.block_limits,
            ) {
                Ok(plan) => plan,
                Err(BlockError::InvalidEndpoints) => {
                    return Err(format!(
                        "QC detector returned invalid segment endpoints {}..{} for {} gates",
                        hot.start_gate,
                        hot.end_gate,
                        current.len()
                    ));
                }
                Err(e) => {
                    event.status = "block_limit".into();
                    event.detail.push_str(&format!("; closure={e:?}"));
                    report.events.push(event);
                    continue;
                }
            };
            let window = plan.window(&current);
            let candidates = lookup(
                &window,
                mixer.quality_num_wires(),
                config.polynomial_budget,
                config.candidate_limits,
                &mut rng,
            );
            event.detail.push_str(&format!(
                "; block_gates={} support={} examined={} entries_found={} truncated={} canonicalization_errors={:?} lookup_limits={:?} malformed={} unplaceable={} identity_skipped={} duplicate_skipped={} size_skipped={} non_equivalent={} verification_budget={}",
                window.len(), plan.support, candidates.examined, candidates.entries_found,
                candidates.truncated, candidates.canonicalization_errors, candidates.lookup_limits, candidates.malformed_values,
                candidates.unplaceable, candidates.identity_skipped, candidates.duplicate_skipped,
                candidates.size_skipped, candidates.non_equivalent, candidates.equivalence_budget_exceeded));
            let mut hot_candidates = 0;
            let mut screening_incomplete = 0;
            let mut apply_refused = 0;
            for candidate in &candidates.candidates {
                // The live mixer requires a nonempty tape. Refuse before
                // trial tracing, whose region also has to be nonempty.
                if current.len() - window.len() + candidate.gates.len() == 0 {
                    apply_refused += 1;
                    event.detail.push_str("; apply_refused=empty_circuit");
                    continue;
                }
                let mut trial = current.clone();
                trial.splice(plan.span.clone(), plan.reordered_span(&current));
                trial.splice(plan.block.clone(), candidate.gates.clone());
                let new_span_end = plan.span.end - window.len() + candidate.gates.len();
                // Include both seams, even for deletion. This also screens hot
                // segments created on borrowed wires or by the gathering swaps.
                let region = plan.span.start.saturating_sub(1)
                    ..new_span_end.saturating_add(1).min(trial.len());
                let checked = detector.scan_region(&trial, region)?;
                if checked.truncated {
                    screening_incomplete += 1;
                    continue;
                }
                if checked.hot_segment_count > 0 || checked.hot_boundary_firings > 0 {
                    hot_candidates += 1;
                    continue;
                }
                if let Err(e) = mixer.apply_quality_repair(&plan, candidate.gates.clone()) {
                    apply_refused += 1;
                    event.detail.push_str(&format!("; apply_refused={e}"));
                    continue;
                }
                event.status = "repaired".into();
                event
                    .detail
                    .push_str(&format!("; replacement_gates={}", candidate.gates.len()));
                report.repairs += 1;
                changed = true;
                break;
            }
            if !changed {
                event.status = if candidates.truncated {
                    "candidate_budget"
                } else if !candidates.canonicalization_errors.is_empty()
                    || !candidates.lookup_limits.is_empty()
                    || candidates.equivalence_budget_exceeded > 0
                {
                    "lookup_or_verification_limit"
                } else if screening_incomplete > 0 {
                    "screening_limit"
                } else if candidates.entries_found == 0 {
                    "no_db_entry"
                } else if apply_refused > 0 {
                    "application_refused"
                } else if hot_candidates > 0 {
                    "examined_replacements_hot"
                } else {
                    "no_usable_replacement"
                }
                .into();
            }
            event.detail.push_str(&format!(
                "; hot_candidates={hot_candidates} screening_incomplete={screening_incomplete}"
            ));
            report.events.push(event);
            if changed {
                break;
            } // Refresh every segment index after a splice.
        }
        if !changed {
            report.after = scan;
            break;
        }
        current = mixer.arena.to_vec();
        scan = detector.scan(&current)?;
    }
    let fresh_detector = Detector::new_with_sample_seed(
        reference.unwrap_or_else(|| mixer.quality_reference()),
        mixer.quality_num_wires(),
        config.detector.clone(),
        config.detector.seed ^ 0x6175_6469_745f_5143,
    )?;
    report.fresh_audit = fresh_detector.scan(&mixer.arena.to_vec())?;
    report.budget_exhausted |= report.after.truncated || report.fresh_audit.truncated;
    Ok(report)
}

#[cfg(test)]
#[path = "../../../../tests/stages/db_mixing/leakage_repair/tests.rs"]
mod tests;
