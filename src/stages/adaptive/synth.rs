//! One block of the adaptive stage: compose the block's ANF from its gates
//! over the global wires (the `compose_window` semantics of `anf_synth`:
//! GF(2) toggle sets, never deduplicated), hand it to the unpacking
//! synthesizer over a local variable universe (`unpacking::spec`), map the
//! gates back, check them on random full states against the ORIGINAL block
//! gates (every wire, so a borrowed ancilla must be restored), run the
//! pairwise peephole and re-check. Any failure is a `Failed(reason)` with
//! the prototype's status vocabulary (`compose_overflow`, `compose_timeout`,
//! `stepcap`, `timeout`, `stalled`, `verify-fail`); the caller keeps the
//! original block then.
//!
//! Memory: the composed ANF is bounded by `compose_cap` terms (any product
//! polynomial and the running total) and is dropped before the synthesizer
//! runs; nothing of a block outlives its `BlockResult`.
use crate::circuit::xgate::XGate;
use crate::stages::packing::mono::{MonoWidthFn, mono_words, with_mono_width};
use crate::stages::packing::pyrand::PyRandom;
use crate::stages::packing::verify::{eval_gates_lanes16, random_lanes};
use crate::stages::unpacking::mono::Mono;
use crate::stages::unpacking::postopt::peephole;
use crate::stages::unpacking::spec::LocalUniverse;
use crate::stages::unpacking::synth::{
    SynthOutput, SynthParams, SynthStatus, WindowSpec, synthesize,
};
use rustc_hash::{FxHashMap, FxHashSet};
use std::time::{Duration, Instant};

/// Outside wires offered to the synthesizer as ancilla candidates.
pub const DEFAULT_ANCILLA_POOL: usize = 32;
/// Terms a block's composition may hold (8 x the planner's term cap).
pub const DEFAULT_COMPOSE_CAP: usize = 8 * 131_072;
/// Batches of 1024 random full states per block check.
pub const DEFAULT_VERIFY_BATCHES: usize = 2;
/// Seconds for a block's composition (the prototype's `--compose-limit`).
pub const DEFAULT_COMPOSE_TIME_LIMIT: f64 = 300.0;
/// Seed of the sampled block checks (public, fixed; mixed with the block's
/// index).
pub const VERIFY_SEED: u64 = 20_261_026;

#[derive(Clone, Debug, PartialEq)]
pub struct BlockParams {
    pub synth: SynthParams,
    /// Run the pairwise peephole on the synthesized block.
    pub postopt: bool,
    pub ancilla_pool: usize,
    pub compose_cap: usize,
    /// Seconds for the composition of a block (the prototype's
    /// `--compose-limit`); the synthesizer's budget is separate.
    pub compose_time_limit: f64,
    pub verify_batches: usize,
}

impl Default for BlockParams {
    fn default() -> Self {
        BlockParams {
            synth: SynthParams {
                time_limit: 60.0,
                ..SynthParams::default()
            },
            postopt: true,
            ancilla_pool: DEFAULT_ANCILLA_POOL,
            compose_cap: DEFAULT_COMPOSE_CAP,
            compose_time_limit: DEFAULT_COMPOSE_TIME_LIMIT,
            verify_batches: DEFAULT_VERIFY_BATCHES,
        }
    }
}

impl BlockParams {
    pub fn validate(&self) -> Result<(), String> {
        self.synth.validate()?;
        if self.ancilla_pool > 1024 {
            return Err(format!("ancilla pool {}: at most 1024", self.ancilla_pool));
        }
        if self.compose_cap == 0 {
            return Err("compose cap must be positive".into());
        }
        if !(self.compose_time_limit > 0.0 && self.compose_time_limit.is_finite()) {
            return Err(format!(
                "compose time limit {}: must be a positive number of seconds",
                self.compose_time_limit
            ));
        }
        if self.verify_batches == 0 || self.verify_batches > 1024 {
            return Err(format!(
                "verify batches {}: must be in 1..=1024",
                self.verify_batches
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockStatus {
    Ok,
    Failed(&'static str),
}

impl BlockStatus {
    pub fn is_ok(self) -> bool {
        matches!(self, BlockStatus::Ok)
    }

    /// `ok` or the failure reason.
    pub fn name(self) -> &'static str {
        match self {
            BlockStatus::Ok => "ok",
            BlockStatus::Failed(r) => r,
        }
    }
}

/// One block's outcome.
#[derive(Clone, Debug)]
pub struct BlockResult {
    pub status: BlockStatus,
    /// The checked replacement gates (global wires); `None` on failure.
    pub gates: Option<Vec<XGate>>,
    pub secs: f64,
    /// The synthesizer's status (`converged`, `stepcap`, ...), or the stage
    /// at which the block failed before the synthesizer ran.
    pub synth_status: &'static str,
    pub steps: usize,
    pub restarts_used: usize,
    pub deadline_hit: bool,
    /// Distinct variables over all monomials of the block ANF.
    pub support: usize,
    /// Wires whose exit polynomial differs from `x_w`.
    pub changed: usize,
    pub monomials: usize,
    pub max_degree: usize,
    /// Gates before the peephole (0 unless synthesized).
    pub pre_postopt: usize,
    /// Size of the local variable universe (0 unless composed).
    pub universe: usize,
    pub note: String,
}

impl BlockResult {
    /// The synthesizer panicked on the block (caught by the round, which
    /// keeps the block verbatim and continues).
    pub fn crashed() -> BlockResult {
        let mut r = BlockResult::failed("crash");
        r.note = "the synthesizer panicked; block kept verbatim".into();
        r
    }

    fn failed(reason: &'static str) -> BlockResult {
        BlockResult {
            status: BlockStatus::Failed(reason),
            gates: None,
            secs: 0.0,
            synth_status: reason,
            steps: 0,
            restarts_used: 0,
            deadline_hit: false,
            support: 0,
            changed: 0,
            monomials: 0,
            max_degree: 0,
            pre_postopt: 0,
            universe: 0,
            note: String::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// composition over the global wires (port of anf_synth's compose_window)
// ---------------------------------------------------------------------------

type Poly<const W: usize> = FxHashSet<Mono<W>>;

#[inline]
fn toggle<const W: usize>(p: &mut Poly<W>, m: Mono<W>) {
    if !p.remove(&m) {
        p.insert(m);
    }
}

fn mul<const W: usize>(
    a: &Poly<W>,
    b: &Poly<W>,
    cap: usize,
    deadline: Instant,
) -> Result<Poly<W>, &'static str> {
    let (a, b) = if a.len() > b.len() { (b, a) } else { (a, b) };
    let mut res = Poly::default();
    for (i, x) in a.iter().enumerate() {
        for y in b {
            toggle(&mut res, x.or(y));
        }
        if res.len() > cap {
            return Err("compose_overflow");
        }
        if i % 16 == 15 && Instant::now() > deadline {
            return Err("compose_timeout");
        }
    }
    Ok(res)
}

/// The block ANF: `(wire, sorted monomials)` for every wire whose exit
/// polynomial differs from `x_w`, wires ascending; `Err` on overflow of
/// `cap` (any product polynomial, the running total every 64 gates, the
/// final total) or past `deadline`.
fn compose<const W: usize>(
    gates: &[XGate],
    cap: usize,
    deadline: Instant,
) -> Result<Vec<(u16, Vec<Mono<W>>)>, &'static str> {
    let mut polys: FxHashMap<u16, Poly<W>> = FxHashMap::default();
    let one: Poly<W> = [Mono::ONE].into_iter().collect();
    for (gi, g) in gates.iter().enumerate() {
        let mut prod = one.clone();
        for &(c, p) in &g.ctrls {
            let mut cp: Poly<W> = match polys.get(&c) {
                Some(q) => q.clone(),
                None => [Mono::var(c as u32)].into_iter().collect(),
            };
            if !p {
                toggle(&mut cp, Mono::ONE);
            }
            prod = mul(&prod, &cp, cap, deadline)?;
        }
        if g.comp {
            toggle(&mut prod, Mono::ONE);
        }
        let tgt = polys
            .entry(g.target)
            .or_insert_with(|| [Mono::var(g.target as u32)].into_iter().collect());
        for m in prod {
            toggle(tgt, m);
        }
        if (gi + 1) % 64 == 0 {
            let tot: usize = polys.values().map(|p| p.len()).sum();
            if tot > cap {
                return Err("compose_overflow");
            }
            if Instant::now() > deadline {
                return Err("compose_timeout");
            }
        }
    }
    let mut out: Vec<(u16, Vec<Mono<W>>)> = Vec::new();
    let mut total = 0usize;
    for (w, p) in polys {
        if p.len() == 1 && p.contains(&Mono::var(w as u32)) {
            continue;
        }
        let mut v: Vec<Mono<W>> = p.into_iter().collect();
        v.sort_unstable();
        total += v.len();
        out.push((w, v));
    }
    if total > cap {
        return Err("compose_overflow");
    }
    out.sort_by_key(|(w, _)| *w);
    Ok(out)
}

/// The support's wires (ascending) and `(changed, monomials, max_degree)`.
fn anf_stats<const W: usize>(anf: &[(u16, Vec<Mono<W>>)]) -> (Vec<u16>, usize, usize, usize) {
    let mut vars = Mono::<W>::ONE;
    let mut total = 0usize;
    let mut maxdeg = 0u32;
    for (_, p) in anf {
        for m in p {
            vars = vars.or(m);
            total += 1;
            maxdeg = maxdeg.max(m.degree());
        }
    }
    let support: Vec<u16> = vars.vars().map(|v| v as u16).collect();
    (support, anf.len(), total, maxdeg as usize)
}

/// Support and changed wires (ascending) plus the `pool` lowest other wires
/// of a `width`-wide state (what `LocalUniverse::for_window` builds from a
/// deliverable window).
pub fn local_universe(
    support: &[u16],
    changed: &[u16],
    width: usize,
    pool: usize,
) -> LocalUniverse {
    let mut base: Vec<u16> = support.to_vec();
    base.extend_from_slice(changed);
    base.sort_unstable();
    base.dedup();
    let mut wires = base.clone();
    let mut extra = 0usize;
    let mut it = base.iter().peekable();
    for w in 0..width {
        if extra >= pool {
            break;
        }
        let w = w as u16;
        if it.peek().is_some_and(|&&b| b == w) {
            it.next();
            continue;
        }
        wires.push(w);
        extra += 1;
    }
    wires.sort_unstable();
    LocalUniverse { wires }
}

/// What the width-generic front half hands to the width-independent back
/// half.
struct Composed {
    support: usize,
    changed: usize,
    monomials: usize,
    max_degree: usize,
    uni: LocalUniverse,
    out: SynthOutput,
}

/// The global-width monomials remapped to local ids at the local width,
/// then the synthesizer (the global ANF is dropped before it runs).
struct Remap<'a, const W: usize> {
    anf: Vec<(u16, Vec<Mono<W>>)>,
    uni: &'a LocalUniverse,
    params: &'a SynthParams,
    seed: u64,
    idx: usize,
}

impl<const W: usize> MonoWidthFn<SynthOutput> for Remap<'_, W> {
    fn call<const W2: usize>(self) -> SynthOutput {
        let Remap {
            anf,
            uni,
            params,
            seed,
            idx,
        } = self;
        let top = uni.wires.last().map_or(0, |&w| w as usize + 1);
        let mut local_of = vec![u32::MAX; top];
        for (i, &w) in uni.wires.iter().enumerate() {
            local_of[w as usize] = i as u32;
        }
        let mut polys: Vec<(u32, Vec<Mono<W2>>)> = Vec::with_capacity(anf.len());
        for (w, p) in anf {
            let lp: Vec<Mono<W2>> = p
                .into_iter()
                .map(|m| {
                    let mut lm = Mono::<W2>::ONE;
                    for v in m.vars() {
                        lm = lm.with(local_of[v as usize]);
                    }
                    lm
                })
                .collect();
            polys.push((local_of[w as usize], lp));
        }
        let spec = WindowSpec {
            num_wires: uni.len() as u32,
            polys,
        };
        synthesize(&spec, params, seed, idx)
    }
}

struct Front<'a> {
    gates: &'a [XGate],
    width: usize,
    params: &'a BlockParams,
    synth: &'a SynthParams,
    deadline: Instant,
    seed: u64,
    idx: usize,
}

impl MonoWidthFn<Result<Composed, &'static str>> for Front<'_> {
    fn call<const W: usize>(self) -> Result<Composed, &'static str> {
        let anf = compose::<W>(self.gates, self.params.compose_cap, self.deadline)?;
        let (support, changed, monomials, max_degree) = anf_stats(&anf);
        let changed_wires: Vec<u16> = anf.iter().map(|(w, _)| *w).collect();
        let uni = local_universe(
            &support,
            &changed_wires,
            self.width,
            self.params.ancilla_pool,
        );
        let out = with_mono_width(
            uni.len(),
            Remap::<W> {
                anf,
                uni: &uni,
                params: self.synth,
                seed: self.seed,
                idx: self.idx,
            },
        );
        Ok(Composed {
            support: support.len(),
            changed,
            monomials,
            max_degree,
            uni,
            out,
        })
    }
}

/// `a` and `b` agree on every wire of `batches` x 1024 random full states
/// over `width` wires (and `b` stays inside the state).
pub fn same_function(a: &[XGate], b: &[XGate], width: usize, batches: usize, seed: u64) -> bool {
    if b.iter().any(|g| g.max_wire() as usize >= width) {
        return false;
    }
    let mut rng = PyRandom::new(seed);
    for _ in 0..batches.max(1) {
        let x = random_lanes(&mut rng, width);
        let mut sa = x.clone();
        eval_gates_lanes16(a, &mut sa);
        let mut sb = x;
        eval_gates_lanes16(b, &mut sb);
        if sa != sb {
            return false;
        }
    }
    true
}

/// `(support, changed, monomials, max_degree)` of the block ANF, or the
/// composition failure (`compose_overflow` past `cap` terms,
/// `compose_timeout` past `time_limit` seconds); the planner-independent
/// view of a block's size.
pub fn compose_stats(
    gates: &[XGate],
    width: usize,
    cap: usize,
    time_limit: f64,
) -> Result<(usize, usize, usize, usize), &'static str> {
    struct Stats<'a> {
        gates: &'a [XGate],
        cap: usize,
        deadline: Instant,
    }
    impl MonoWidthFn<Result<(usize, usize, usize, usize), &'static str>> for Stats<'_> {
        fn call<const W: usize>(self) -> Result<(usize, usize, usize, usize), &'static str> {
            let anf = compose::<W>(self.gates, self.cap, self.deadline)?;
            let (support, changed, monomials, max_degree) = anf_stats(&anf);
            Ok((support.len(), changed, monomials, max_degree))
        }
    }
    if mono_words(width).is_none() {
        return Err("width");
    }
    if gates.iter().any(|g| g.max_wire() as usize >= width) {
        return Err("wire-out-of-range");
    }
    let deadline = Instant::now() + Duration::from_secs_f64(time_limit.max(0.01));
    with_mono_width(
        width,
        Stats {
            gates,
            cap,
            deadline,
        },
    )
}

/// A public mixing of the check seed with the block index.
fn mix(seed: u64, idx: u64) -> u64 {
    let mut z = seed ^ idx.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 32)).wrapping_mul(0xD6E8_FEB8_6659_FD93);
    z ^ (z >> 32)
}

/// Synthesize one block (`gates`, over a `width`-wide state) under
/// `time_limit` seconds of synthesizer budget (a non-positive value falls
/// back to `params.synth.time_limit`); the composition runs under its own
/// `params.compose_time_limit` first.
/// `seed` seeds the synthesizer's tie-breaks only (never reported); `idx`
/// keys the PRNG and the check seed (the caller passes the block's start
/// gate).
pub fn synthesize_block(
    gates: &[XGate],
    width: usize,
    params: &BlockParams,
    time_limit: f64,
    seed: u64,
    idx: usize,
) -> BlockResult {
    let t0 = Instant::now();
    if mono_words(width).is_none() {
        return BlockResult::failed("width");
    }
    if gates.iter().any(|g| g.max_wire() as usize >= width) {
        return BlockResult::failed("wire-out-of-range");
    }
    let time_limit = if time_limit.is_finite() && time_limit > 0.0 {
        time_limit
    } else {
        params.synth.time_limit
    };
    let synth = SynthParams {
        time_limit,
        ..params.synth.clone()
    };
    // the composition has its own budget (the prototype's --compose-limit);
    // the synthesizer then gets the full block budget
    let deadline = t0 + Duration::from_secs_f64(params.compose_time_limit.max(0.01));
    let composed = with_mono_width(
        width,
        Front {
            gates,
            width,
            params,
            synth: &synth,
            deadline,
            seed,
            idx,
        },
    );
    let c = match composed {
        Ok(c) => c,
        Err(reason) => {
            let mut r = BlockResult::failed(reason);
            r.secs = t0.elapsed().as_secs_f64();
            return r;
        }
    };
    let out = c.out;
    let mut res = BlockResult {
        status: BlockStatus::Failed(out.status.name()),
        gates: None,
        secs: 0.0,
        synth_status: out.status.name(),
        steps: out.steps,
        restarts_used: out.restarts_used,
        deadline_hit: out.deadline_hit,
        support: c.support,
        changed: c.changed,
        monomials: c.monomials,
        max_degree: c.max_degree,
        pre_postopt: 0,
        universe: c.uni.len(),
        note: out.note,
    };
    if out.status != SynthStatus::Converged {
        res.secs = t0.elapsed().as_secs_f64();
        return res;
    }
    let local = out.gates;
    res.pre_postopt = local.len();
    let check_seed = mix(VERIFY_SEED, idx as u64);
    let syn = c.uni.to_xgates(&local);
    if !same_function(gates, &syn, width, params.verify_batches, check_seed) {
        res.status = BlockStatus::Failed("verify-fail");
        res.synth_status = "verify-fail";
        res.note = "synthesized block != original block on random full states".into();
        res.secs = t0.elapsed().as_secs_f64();
        return res;
    }
    let mut kept = syn;
    if params.postopt {
        let mut opt = local;
        if peephole(&mut opt) > 0 {
            let g2 = c.uni.to_xgates(&opt);
            if same_function(
                gates,
                &g2,
                width,
                params.verify_batches,
                check_seed ^ 0xBEEF,
            ) {
                kept = g2;
            } else {
                res.note = "postopt verify-fail; kept unoptimised".into();
            }
        }
    }
    res.status = BlockStatus::Ok;
    res.gates = Some(kept);
    res.secs = t0.elapsed().as_secs_f64();
    res
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/adaptive/synth.rs"]
mod tests;
