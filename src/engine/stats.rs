// Read-only mixing metrics over a gate sequence, shared by the fmix report
// line and the fmix_stats analyzer. The stationarity signature: a mixed
// circuit's distributional summary converges to a seed-independent plateau
// while the microstate keeps moving, so these are the quantities to compare
// across snapshots (plateau) and across replicas (agreement).
//
// Nothing here mutates a circuit or consumes the mixing chain's RNG; callers
// that need sampling pass their own (metrics-only) RNG.
use super::mix::ORIGIN_SYNTH;
use crate::circuit::xgate::XGate;
use rand::Rng;

// Uniform baseline for origin_diffusion: std of a uniform on [0,1].
pub const UNIFORM_STD: f64 = 0.288_675_134_594_812_9;

// Fanout of gate i = number of later gates that read wire target(i) before it
// is next overwritten. 0 = the gate's output is never consumed (dead until
// overwrite); the mean equals mean width up to boundary effects (every
// attributed read is one control literal).
pub fn fanouts<'a>(gates: impl Iterator<Item = &'a XGate>, wires: usize) -> Vec<u32> {
    let mut last_writer = vec![u32::MAX; wires];
    let mut fan: Vec<u32> = Vec::new();
    for g in gates {
        for &(w, _) in &g.ctrls {
            let lw = last_writer[w as usize];
            if lw != u32::MAX {
                fan[lw as usize] += 1;
            }
        }
        last_writer[g.target as usize] = fan.len() as u32;
        fan.push(0);
    }
    fan
}

// Two-sided float-box size of gate i: how many positions it can move (left +
// right) before hitting a collision, capped per direction. 0 = wedged.
pub fn leeway_at(gates: &[XGate], i: usize, cap: usize) -> usize {
    let g = &gates[i];
    let mut l = 0usize;
    while l < cap && i - l > 0 && !XGate::collides(g, &gates[i - l - 1]) {
        l += 1;
    }
    let mut r = 0usize;
    while r < cap && i + r + 1 < gates.len() && !XGate::collides(g, &gates[i + r + 1]) {
        r += 1;
    }
    l + r
}

// Per-origin normalized positional std, weighted by piece count. Starts near 0
// (each origin's material sits where the origin sat) and rises toward
// UNIFORM_STD as pieces disperse. Origins with a single surviving piece carry
// no spread information and are skipped; synthetic gates are skipped.
pub fn origin_diffusion(origins: &[u32]) -> f64 {
    let n = origins.len() as f64;
    // FxHashMap: cheaper hashing per gate, and its deterministic iteration
    // order makes the float-summed metric reproducible across runs (std
    // RandomState order made the low digits run-dependent).
    let mut acc: rustc_hash::FxHashMap<u32, (u64, f64, f64)> = rustc_hash::FxHashMap::default();
    for (i, &o) in origins.iter().enumerate() {
        if o == ORIGIN_SYNTH {
            continue;
        }
        let x = i as f64 / n;
        let e = acc.entry(o).or_insert((0, 0.0, 0.0));
        e.0 += 1;
        e.1 += x;
        e.2 += x * x;
    }
    let (mut wsum, mut wtot) = (0.0f64, 0u64);
    for (c, s, ss) in acc.into_values() {
        if c < 2 {
            continue;
        }
        let cf = c as f64;
        let var = (ss / cf - (s / cf) * (s / cf)).max(0.0);
        wsum += var.sqrt() * cf;
        wtot += c;
    }
    if wtot == 0 { 0.0 } else { wsum / wtot as f64 }
}

// Distribution of per-origin spread (std of piece positions, in gate units),
// piece-weighted, over origins with >=2 surviving pieces. The uniformity
// counterpart of origin_diffusion's mean: a heavy left tail means some
// original material is still tightly clumped. Returns (fraction of real
// pieces whose origin has only one surviving piece, spread at each requested
// quantile, fraction of multi-piece pieces with spread < ref_gates).
pub fn origin_spread_quantiles(
    origins: &[u32],
    qs: &[f64],
    ref_gates: f64,
) -> (f64, Vec<f64>, f64) {
    let n = origins.len() as f64;
    let mut acc: rustc_hash::FxHashMap<u32, (u64, f64, f64)> = rustc_hash::FxHashMap::default();
    let mut real = 0u64;
    for (i, &o) in origins.iter().enumerate() {
        if o == ORIGIN_SYNTH {
            continue;
        }
        real += 1;
        let x = i as f64 / n;
        let e = acc.entry(o).or_insert((0, 0.0, 0.0));
        e.0 += 1;
        e.1 += x;
        e.2 += x * x;
    }
    let mut spreads: Vec<(f64, u64)> = Vec::new(); // (std in gates, pieces)
    let (mut singles, mut multi) = (0u64, 0u64);
    for (c, s, ss) in acc.into_values() {
        if c < 2 {
            singles += c;
            continue;
        }
        multi += c;
        let cf = c as f64;
        let var = (ss / cf - (s / cf) * (s / cf)).max(0.0);
        spreads.push((var.sqrt() * n, c));
    }
    if multi == 0 {
        return (
            if real == 0 {
                0.0
            } else {
                singles as f64 / real as f64
            },
            vec![0.0; qs.len()],
            0.0,
        );
    }
    spreads.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let below = spreads
        .iter()
        .filter(|&&(sp, _)| sp < ref_gates)
        .map(|&(_, c)| c)
        .sum::<u64>();
    let mut quants = Vec::with_capacity(qs.len());
    for &q in qs {
        let target = (q * multi as f64) as u64;
        let mut cum = 0u64;
        let mut val = spreads.last().unwrap().0;
        for &(sp, c) in &spreads {
            cum += c;
            if cum >= target {
                val = sp;
                break;
            }
        }
        quants.push(val);
    }
    (
        singles as f64 / real as f64,
        quants,
        below as f64 / multi as f64,
    )
}

// Pearson correlation of origin ids at adjacent positions (pairs with a
// synthetic member skipped): 1 in the unmixed circuit, 0 once neighbors are
// unrelated.
pub fn adjacent_origin_autocorr(origins: &[u32]) -> f64 {
    let (mut n, mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0u64, 0f64, 0f64, 0f64, 0f64, 0f64);
    for w in origins.windows(2) {
        if w[0] == ORIGIN_SYNTH || w[1] == ORIGIN_SYNTH {
            continue;
        }
        let (x, y) = (w[0] as f64, w[1] as f64);
        n += 1;
        sx += x;
        sy += y;
        sxx += x * x;
        syy += y * y;
        sxy += x * y;
    }
    if n < 2 {
        return 0.0;
    }
    let nf = n as f64;
    let cov = sxy / nf - (sx / nf) * (sy / nf);
    let (vx, vy) = (sxx / nf - (sx / nf).powi(2), syy / nf - (sy / nf).powi(2));
    if vx <= 0.0 || vy <= 0.0 {
        0.0
    } else {
        cov / (vx * vy).sqrt()
    }
}

// Mean |position fraction - origin fraction| over real-origin gates (same
// definition as Mixer::origin_displacement); ~1/3 for independent uniforms.
pub fn origin_displacement(origins: &[u32]) -> f64 {
    let n = origins.len() as f64;
    let m = origins
        .iter()
        .filter(|&&o| o != ORIGIN_SYNTH)
        .max()
        .map_or(0, |&o| o + 1) as f64;
    if m == 0.0 {
        return 0.0;
    }
    let (mut acc, mut cnt) = (0.0f64, 0u64);
    for (i, &o) in origins.iter().enumerate() {
        if o != ORIGIN_SYNTH {
            acc += (i as f64 / n - o as f64 / m).abs();
            cnt += 1;
        }
    }
    if cnt == 0 { 0.0 } else { acc / cnt as f64 }
}

// Mean distinct origins in sampled 32-gate windows (max 32), the Mixer::report
// convention; all synthetic gates count as one shared value.
pub fn window_origin_diversity(origins: &[u32], samples: usize, rng: &mut impl Rng) -> f64 {
    if origins.len() < 32 {
        return 0.0;
    }
    let mut acc = 0.0f64;
    for _ in 0..samples {
        let s = rng.random_range(0..=(origins.len() - 32));
        let mut set: Vec<u32> = origins[s..s + 32].to_vec();
        set.sort_unstable();
        set.dedup();
        acc += set.len() as f64;
    }
    acc / samples as f64
}

// Shannon entropy in bits of a count vector (zeros skipped).
pub fn entropy_bits(counts: impl Iterator<Item = u64>) -> f64 {
    let counts: Vec<u64> = counts.filter(|&c| c > 0).collect();
    let total: u64 = counts.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let tf = total as f64;
    -counts
        .iter()
        .map(|&c| c as f64 / tf)
        .map(|p| p * p.log2())
        .sum::<f64>()
}

// Entropy of the distribution over unordered wire pairs inside gate supports
// (target plus control wires). Second-order structure: which wires the mixing
// has actually coupled. Returns (bits, distinct pairs seen).
pub fn pair_cooccurrence_entropy<'a>(
    gates: impl Iterator<Item = &'a XGate>,
    wires: usize,
) -> (f64, usize) {
    let mut counts = vec![0u64; wires * wires];
    let mut support: Vec<u16> = Vec::new();
    for g in gates {
        support.clear();
        support.push(g.target);
        support.extend(g.ctrls.iter().map(|&(w, _)| w));
        support.sort_unstable();
        for i in 0..support.len() {
            for j in i + 1..support.len() {
                counts[support[i] as usize * wires + support[j] as usize] += 1;
            }
        }
    }
    let distinct = counts.iter().filter(|&&c| c > 0).count();
    (entropy_bits(counts.into_iter()), distinct)
}

// Distinct wires touched (support union) in sampled windows of `w` gates.
// Returns (mean, min, max) over the samples.
pub fn window_wire_span(
    gates: &[XGate],
    wires: usize,
    w: usize,
    samples: usize,
    rng: &mut impl Rng,
) -> (f64, usize, usize) {
    if gates.len() < w || samples == 0 {
        return (0.0, 0, 0);
    }
    let mut stamp = vec![u32::MAX; wires];
    let (mut acc, mut mn, mut mx) = (0.0f64, usize::MAX, 0usize);
    for round in 0..samples {
        let s = rng.random_range(0..=(gates.len() - w));
        let mut distinct = 0usize;
        for g in &gates[s..s + w] {
            for wi in g.ctrls.iter().map(|&(cw, _)| cw).chain([g.target]) {
                if stamp[wi as usize] != round as u32 {
                    stamp[wi as usize] = round as u32;
                    distinct += 1;
                }
            }
        }
        acc += distinct as f64;
        mn = mn.min(distinct);
        mx = mx.max(distinct);
    }
    (acc / samples as f64, mn, mx)
}

#[cfg(test)]
#[path = "../../tests/unit/engine/stats/stats_tests.rs"]
mod stats_tests;
