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
                    candidates[a]
                        .fanout_l1(self.target)
                        .partial_cmp(&candidates[b].fanout_l1(self.target))
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
            })
        })
        .as_ref()
}
