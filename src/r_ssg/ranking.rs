//! Swappable ranking of candidate subcircuits for the mixing game.
//!
//! Two selection decisions use a ranker:
//!   * OUTGOING (#11): which window of the current circuit to replace (shooting game).
//!   * INCOMING (#9):  which equivalent replacement to splice in.
//!
//! Feature computation (global fanout / leeway / wire-span) lives in compiled Rust and is
//! handed to the ranker as [`CandFeatures`]. The ranking decision itself is pluggable: a
//! built-in default, or a Rhai script supplied at runtime (no recompile needed).

use std::sync::OnceLock;

use rand::seq::SliceRandom;

/// Features of one candidate subcircuit, exposed to ranking functions.
/// Add fields here as new criteria are invented; existing scripts keep working.
#[derive(Clone, Debug, Default)]
pub struct CandFeatures {
    pub size: usize,                // number of gates
    pub wires_spanned: usize,       // distinct wires touched
    pub low_leeway_count: usize,    // gates with leeway < MIN_MEDIAN_LEEWAY (global)
    pub zero_fanout_count: usize,   // gates with global fanout == 0
    pub fanout_buckets: [usize; 5], // gate counts at fanout 0,1,2,3,>3
    pub max_fanout: usize,          // max global fanout among the gates
}

impl CandFeatures {
    /// L1 distance of this candidate's fanout-bucket distribution to the target.
    pub fn fanout_l1(&self, target: [f64; 5]) -> f64 {
        let n = self.size.max(1) as f64;
        (0..5)
            .map(|k| (self.fanout_buckets[k] as f64 / n - target[k]).abs())
            .sum()
    }
}

/// A ranker turns candidate features into a best-first ordering of candidate indices.
pub trait Ranker: Send + Sync {
    fn order(&self, cands: &[CandFeatures]) -> Vec<usize>;
}

// ---- Built-in OUTGOING ranker (#11) ----
// Pareto over (wires_spanned↑, low_leeway_count↑, zero_fanout_count↑); A dominates B iff A is
// >= in all three and > in at least one. Emit Pareto fronts best-first; within a front order by
// size↑ (smaller ranks higher), random among equal-size.
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
    fn order(&self, c: &[CandFeatures]) -> Vec<usize> {
        let mut rng = rand::rng();
        let mut remaining: Vec<usize> = (0..c.len()).collect();
        let mut out: Vec<usize> = Vec::with_capacity(c.len());
        while !remaining.is_empty() {
            // Non-dominated front among the remaining candidates.
            let front: Vec<usize> = remaining
                .iter()
                .copied()
                .filter(|&i| !remaining.iter().any(|&j| j != i && dominates(&c[j], &c[i])))
                .collect();
            // size↑ tiebreak, random among equal size: shuffle then stable-sort by size.
            let mut f = front.clone();
            f.shuffle(&mut rng);
            f.sort_by_key(|&i| c[i].size);
            out.extend(&f);
            remaining.retain(|i| !front.contains(i));
        }
        out
    }
}

// ---- Built-in INCOMING ranker (#9) ----
// Minimize fanout-distribution L1 to target, then low_leeway_count; candidates exceeding
// MAX_FANOUT are pushed to the back (tried only if nothing else is available).
pub struct FanoutTargetIncoming {
    pub target: [f64; 5],
    pub max_fanout: usize,
}

impl Ranker for FanoutTargetIncoming {
    fn order(&self, c: &[CandFeatures]) -> Vec<usize> {
        let mut rng = rand::rng();
        let mut idx: Vec<usize> = (0..c.len()).collect();
        idx.shuffle(&mut rng); // random tiebreak baseline
        idx.sort_by(|&a, &b| {
            let va = c[a].max_fanout > self.max_fanout;
            let vb = c[b].max_fanout > self.max_fanout;
            va.cmp(&vb) // non-violators (false) first
                .then(
                    c[a].fanout_l1(self.target)
                        .partial_cmp(&c[b].fanout_l1(self.target))
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
                .then(c[a].low_leeway_count.cmp(&c[b].low_leeway_count))
        });
        idx
    }
}

// ---- Script-backed ranker (Rhai) ----
// The script defines `fn rank(cands)` taking an array of feature maps and returning an array of
// candidate indices, best-first.
pub struct ScriptRanker {
    engine: rhai::Engine,
    ast: rhai::AST,
}

impl ScriptRanker {
    pub fn from_file(path: &str) -> Result<Self, String> {
        let src = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
        let engine = rhai::Engine::new();
        let ast = engine
            .compile(&src)
            .map_err(|e| format!("compile {path}: {e}"))?;
        Ok(Self { engine, ast })
    }

    fn to_map(f: &CandFeatures) -> rhai::Map {
        let mut m = rhai::Map::new();
        m.insert("size".into(), (f.size as i64).into());
        m.insert("wires_spanned".into(), (f.wires_spanned as i64).into());
        m.insert(
            "low_leeway_count".into(),
            (f.low_leeway_count as i64).into(),
        );
        m.insert(
            "zero_fanout_count".into(),
            (f.zero_fanout_count as i64).into(),
        );
        m.insert("max_fanout".into(), (f.max_fanout as i64).into());
        let buckets: rhai::Array = f
            .fanout_buckets
            .iter()
            .map(|&b| rhai::Dynamic::from(b as i64))
            .collect();
        m.insert("fanout_buckets".into(), buckets.into());
        m
    }
}

impl Ranker for ScriptRanker {
    fn order(&self, c: &[CandFeatures]) -> Vec<usize> {
        let arr: rhai::Array = c
            .iter()
            .map(|f| rhai::Dynamic::from(Self::to_map(f)))
            .collect();
        let mut scope = rhai::Scope::new();
        let res: Result<rhai::Array, _> =
            self.engine.call_fn(&mut scope, &self.ast, "rank", (arr,));
        match res {
            Ok(order) => {
                let mut out: Vec<usize> = order
                    .into_iter()
                    .filter_map(|d| d.as_int().ok())
                    .filter(|&i| i >= 0 && (i as usize) < c.len())
                    .map(|i| i as usize)
                    .collect();
                // Append any indices the script omitted, so we never drop candidates.
                for i in 0..c.len() {
                    if !out.contains(&i) {
                        out.push(i);
                    }
                }
                out
            }
            Err(e) => {
                eprintln!("[ranking] script rank() failed ({e}); falling back to input order");
                (0..c.len()).collect()
            }
        }
    }
}

// ---- Registry: one ranker per side, set once at startup from flags ----
static OUTGOING: OnceLock<Box<dyn Ranker>> = OnceLock::new();
static INCOMING: OnceLock<Box<dyn Ranker>> = OnceLock::new();

pub fn set_outgoing(r: Box<dyn Ranker>) {
    let _ = OUTGOING.set(r);
}
pub fn set_incoming(r: Box<dyn Ranker>) {
    let _ = INCOMING.set(r);
}

pub fn outgoing() -> &'static dyn Ranker {
    OUTGOING.get_or_init(|| Box::new(ParetoOutgoing)).as_ref()
}
pub fn incoming() -> &'static dyn Ranker {
    INCOMING
        .get_or_init(|| {
            Box::new(FanoutTargetIncoming {
                target: crate::r_ssg::replace::FANOUT_TARGET,
                max_fanout: crate::r_ssg::replace::MAX_FANOUT
                    .load(std::sync::atomic::Ordering::Relaxed),
            })
        })
        .as_ref()
}
