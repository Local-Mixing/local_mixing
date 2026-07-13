use rand::seq::SliceRandom;
use std::sync::OnceLock;

#[derive(Clone, Debug, Default)]
pub struct CandFeatures {
    pub size: usize,
    pub wires_spanned: usize,
    pub low_leeway_count: usize,
    pub zero_fanout_count: usize,
    pub fanout_buckets: [usize; 5],
    pub max_fanout: usize,
    /// Median commutation leeway of the window's gates (wire-relabeling invariant, so it is
    /// meaningful even for candidates still in canonical wire space).
    pub median_leeway: usize,
    /// Maximum number of window gates touching any single wire — a hot-wire / uneven
    /// touch-distribution proxy (also relabeling invariant).
    pub max_wire_touch: usize,
}

impl CandFeatures {
    pub fn fanout_l1(&self, target: [f64; 5]) -> f64 {
        let n = self.size.max(1) as f64;
        (0..5)
            .map(|k| (self.fanout_buckets[k] as f64 / n - target[k]).abs())
            .sum()
    }
}

pub trait Ranker: Send + Sync {
    fn order(&self, candidates: &[CandFeatures]) -> Vec<usize>;
}

pub struct ParetoOutgoing;

fn dominates(a: &CandFeatures, b: &CandFeatures) -> bool {
    let ge = a.wires_spanned >= b.wires_spanned
        && a.low_leeway_count >= b.low_leeway_count
        && a.zero_fanout_count >= b.zero_fanout_count;
    let gt = a.wires_spanned > b.wires_spanned
        || a.low_leeway_count > b.low_leeway_count
        || a.zero_fanout_count > b.zero_fanout_count;
    ge && gt
}

impl Ranker for ParetoOutgoing {
    fn order(&self, candidates: &[CandFeatures]) -> Vec<usize> {
        let mut rng = rand::rng();
        let mut remaining: Vec<usize> = (0..candidates.len()).collect();
        let mut out = Vec::with_capacity(candidates.len());
        while !remaining.is_empty() {
            let front: Vec<usize> = remaining
                .iter()
                .copied()
                .filter(|&i| {
                    !remaining
                        .iter()
                        .any(|&j| j != i && dominates(&candidates[j], &candidates[i]))
                })
                .collect();
            let mut ordered_front = front.clone();
            ordered_front.shuffle(&mut rng);
            ordered_front.sort_by_key(|&i| candidates[i].size);
            out.extend(&ordered_front);
            remaining.retain(|i| !front.contains(i));
        }
        out
    }
}

pub struct FanoutTargetIncoming {
    pub target: [f64; 5],
    pub max_fanout: usize,
    /// Target median window leeway; candidates below it are penalized proportionally so
    /// selection stops preferring maximally pinned (wire-dense) replacements.
    pub min_median_leeway: usize,
}

impl FanoutTargetIncoming {
    // Distribution-matching penalty, lower is better: distance of the fanout-bucket
    // distribution from the random-like target, plus a shortfall term for median leeway
    // below target, plus a hot-wire term for windows that concentrate touches on one wire.
    fn penalty(&self, c: &CandFeatures) -> f64 {
        let fanout = c.fanout_l1(self.target);
        let target_median = self.min_median_leeway.max(1) as f64;
        let leeway_shortfall = (target_median - c.median_leeway as f64).max(0.0) / target_median;
        let hot_wire = if c.size > 0 {
            c.max_wire_touch as f64 / c.size as f64
        } else {
            0.0
        };
        fanout + leeway_shortfall + 0.5 * hot_wire
    }
}

impl Ranker for FanoutTargetIncoming {
    fn order(&self, candidates: &[CandFeatures]) -> Vec<usize> {
        let mut rng = rand::rng();
        let mut idx: Vec<usize> = (0..candidates.len()).collect();
        idx.shuffle(&mut rng);
        idx.sort_by(|&a, &b| {
            let va = candidates[a].max_fanout > self.max_fanout;
            let vb = candidates[b].max_fanout > self.max_fanout;
            va.cmp(&vb)
                .then(
                    self.penalty(&candidates[a])
                        .partial_cmp(&self.penalty(&candidates[b]))
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
                .then(
                    candidates[a]
                        .low_leeway_count
                        .cmp(&candidates[b].low_leeway_count),
                )
        });
        idx
    }
}

static OUTGOING: OnceLock<Box<dyn Ranker>> = OnceLock::new();
static INCOMING: OnceLock<Box<dyn Ranker>> = OnceLock::new();

pub fn outgoing() -> &'static dyn Ranker {
    OUTGOING.get_or_init(|| Box::new(ParetoOutgoing)).as_ref()
}

pub fn incoming() -> &'static dyn Ranker {
    INCOMING
        .get_or_init(|| {
            Box::new(FanoutTargetIncoming {
                target: crate::replace::replace::FANOUT_TARGET,
                max_fanout: crate::replace::replace::MAX_FANOUT
                    .load(std::sync::atomic::Ordering::Relaxed),
                min_median_leeway: crate::replace::replace::MIN_MEDIAN_LEEWAY
                    .load(std::sync::atomic::Ordering::Relaxed),
            })
        })
        .as_ref()
}

#[cfg(test)]
mod tests {
    use super::{CandFeatures, FanoutTargetIncoming, Ranker};

    fn ranker() -> FanoutTargetIncoming {
        FanoutTargetIncoming {
            target: crate::replace::replace::FANOUT_TARGET,
            max_fanout: 50,
            min_median_leeway: 10,
        }
    }

    #[test]
    fn incoming_ranker_prefers_higher_median_leeway() {
        let pinned = CandFeatures {
            size: 5,
            median_leeway: 0,
            ..Default::default()
        };
        let loose = CandFeatures {
            size: 5,
            median_leeway: 20,
            ..Default::default()
        };
        let order = ranker().order(&[pinned, loose]);

        assert_eq!(
            order[0], 1,
            "candidate meeting the leeway target ranks first"
        );
    }

    #[test]
    fn incoming_ranker_penalizes_hot_wires() {
        let even = CandFeatures {
            size: 6,
            median_leeway: 20,
            max_wire_touch: 2,
            ..Default::default()
        };
        let hot = CandFeatures {
            size: 6,
            median_leeway: 20,
            max_wire_touch: 6,
            ..Default::default()
        };
        let order = ranker().order(&[hot, even]);

        assert_eq!(order[0], 1, "evenly touched candidate ranks first");
    }

    #[test]
    fn incoming_ranker_hard_rejects_fanout_outliers_first() {
        let outlier = CandFeatures {
            size: 5,
            median_leeway: 20,
            max_fanout: 100,
            ..Default::default()
        };
        let ok = CandFeatures {
            size: 5,
            median_leeway: 0,
            max_fanout: 10,
            ..Default::default()
        };
        let order = ranker().order(&[outlier, ok]);

        assert_eq!(
            order[0], 1,
            "max-fanout violation outranks any soft penalty difference"
        );
    }
}
