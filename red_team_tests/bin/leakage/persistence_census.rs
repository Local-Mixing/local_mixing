//! Which Boolean functions live a long time?
//!
//! Every instrument in this project is either per-snapshot (the reconstruction
//! and statistical heatmaps) or a syntactic write/read census. None of them see
//! the invariant that actually defines the band: a band value is a FROZEN
//! FUNCTION OF THE INPUT, unchanged from the input fill to the output strip.
//! Rolling relocates the variable to another wire, but the wire's Boolean
//! function is the same function -- so a fingerprint taken at checkpoints and
//! matched ACROSS wires follows it for free, and the band population separates
//! from the carriers by lifetime alone, without ever guessing a wire set.
//!
//! That is why band WIDTH is no defense against this reading: the attack does
//! not enumerate candidate subsets, it recovers the population. C(b,3) is
//! irrelevant to it.
//!
//! MEASURE. Snapshot every wire at `--checkpoints` evenly spaced prefixes over
//! `--samples` bit-sliced random inputs (the zero-slice convention). A wire's
//! signature at a checkpoint is its column of sample bits; the FINGERPRINT is
//! that signature canonicalised for complement-equivalence (a mask literal may
//! carry either polarity, so f and !f are the same function for this purpose).
//! Coverage of a fingerprint = the fraction of checkpoints at which it is
//! present on SOME wire. Sorting fingerprints by coverage and reading the
//! largest relative drop gives the elbow: if the construction has a frozen
//! population of size b, the top b entries sit near 1.0 and entry b+1 falls off
//! a cliff.
//!
//! Constant signatures (all-zero / all-one) are counted separately -- an unused
//! wire pinned at 0 is trivially persistent and says nothing about the band.
//!
//! Usage: persistence_census --g <circuit> --n <logical n>
//!                           [--checkpoints 48] [--samples 1024] [--top 24]
//!                           [--lo-frac 0.15] [--hi-frac 0.85]
//!
//! `--lo-frac`/`--hi-frac` bound the checkpoint window to the interior: both
//! ports are unencoded, and including them mixes the pre-fill and post-strip
//! regimes into the lifetime statistic.

use clap::Parser;
use local_mixing::engine::format::{read_g57_file, read_mpmct};
use local_mixing::circuit::xgate::{max_wire, XGate};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

#[derive(Parser, Debug)]
#[command(name = "persistence_census")]
struct Args {
    #[arg(long)]
    g: String,
    #[arg(long, default_value = "mpmct1")]
    g_format: String,
    /// Logical input width: random x on wires 0..n-1, zeros elsewhere.
    #[arg(long)]
    n: usize,
    #[arg(long, default_value_t = 48)]
    checkpoints: usize,
    #[arg(long, default_value_t = 1024)]
    samples: usize,
    #[arg(long, default_value_t = 24)]
    top: usize,
    #[arg(long, default_value_t = 0.15)]
    lo_frac: f64,
    #[arg(long, default_value_t = 0.85)]
    hi_frac: f64,
    #[arg(long, default_value_t = 12345)]
    seed: u64,
}

/// Canonical form of a signature under complement-equivalence: force the first
/// bit to 0 so that f and !f hash identically.
fn canon(sig: &[u64]) -> Vec<u64> {
    if sig[0] & 1 == 1 {
        sig.iter().map(|w| !w).collect()
    } else {
        sig.to_vec()
    }
}

fn main() {
    let args = Args::parse();
    let gates: Vec<XGate> = match args.g_format.as_str() {
        "g57" => read_g57_file(&args.g).expect("read g (g57)"),
        "mpmct1" => read_mpmct(&args.g).expect("read g (mpmct1)").0,
        o => panic!("unknown --g-format {o}"),
    };
    let nw = (max_wire(&gates) as usize + 1).max(args.n);
    let batches = args.samples.div_ceil(64).max(1);
    let mut rng = StdRng::seed_from_u64(args.seed);

    // Interior checkpoint positions.
    let lo = ((gates.len() as f64) * args.lo_frac) as usize;
    let hi = ((gates.len() as f64) * args.hi_frac) as usize;
    let step = ((hi - lo) / args.checkpoints.max(1)).max(1);
    let cps: Vec<usize> = (0..args.checkpoints).map(|i| lo + i * step).collect();

    // sigs[cp][wire] = that wire's signature at that checkpoint.
    let mut sigs: Vec<Vec<Vec<u64>>> = vec![vec![vec![0u64; batches]; nw]; cps.len()];
    for b in 0..batches {
        let mut state = vec![0u64; nw];
        for w in 0..args.n {
            state[w] = rng.random::<u64>();
        }
        let mut c = 0usize;
        for (pos, gate) in gates.iter().enumerate() {
            while c < cps.len() && cps[c] == pos {
                for w in 0..nw {
                    sigs[c][w][b] = state[w];
                }
                c += 1;
            }
            gate.apply_lanes(&mut state);
        }
        while c < cps.len() {
            for w in 0..nw {
                sigs[c][w][b] = state[w];
            }
            c += 1;
        }
    }

    // Coverage: at how many checkpoints does each fingerprint appear anywhere?
    let mut seen: HashMap<Vec<u64>, usize> = HashMap::new();
    let mut consts = 0usize;
    for cp in sigs.iter() {
        let mut here: HashMap<Vec<u64>, ()> = HashMap::new();
        for w in 0..nw {
            let s = &cp[w];
            let all0 = s.iter().all(|&x| x == 0);
            let all1 = s.iter().all(|&x| x == !0u64);
            if all0 || all1 {
                continue;
            }
            here.insert(canon(s), ());
        }
        for (k, _) in here {
            *seen.entry(k).or_insert(0) += 1;
        }
    }
    // Count constant wires once, at the middle checkpoint.
    {
        let mid = &sigs[cps.len() / 2];
        for w in 0..nw {
            let s = &mid[w];
            if s.iter().all(|&x| x == 0) || s.iter().all(|&x| x == !0u64) {
                consts += 1;
            }
        }
    }

    let mut cov: Vec<f64> = seen
        .values()
        .map(|&c| c as f64 / cps.len() as f64)
        .collect();
    cov.sort_by(|a, b| b.partial_cmp(a).unwrap());

    println!(
        "[persistence] {} gates ({nw}w), n={}, {} checkpoints over [{:.2},{:.2}], {} samples; \
         {} distinct fingerprints, {consts} constant wires",
        gates.len(),
        args.n,
        cps.len(),
        args.lo_frac,
        args.hi_frac,
        batches * 64,
        cov.len()
    );

    // The elbow: the largest drop between consecutive ranks. Scan the whole
    // persistent head rather than a window keyed to --top, or the real cliff
    // (which sits at the band size, not at rank <= top) is missed entirely.
    let scan = cov.len().saturating_sub(1);
    let mut best = (0usize, 0f64);
    for i in 0..scan {
        let drop = cov[i] - cov[i + 1];
        if drop > best.1 {
            best = (i + 1, drop);
        }
    }
    if best.0 > 0 {
        println!(
            "  ELBOW at rank {}: coverage {:.3} -> {:.3} (drop {:.3})",
            best.0,
            cov[best.0 - 1],
            cov[best.0],
            best.1
        );
    }
    let show = args.top.min(cov.len());
    let head: Vec<String> = cov[..show].iter().map(|c| format!("{c:.2}")).collect();
    println!("  top-{show} coverage: {}", head.join(" "));
    for t in [0.9, 0.75, 0.5, 0.25] {
        println!(
            "  fingerprints with coverage >= {t:.2}: {}",
            cov.iter().filter(|&&c| c >= t).count()
        );
    }

    // The threshold counts answer "is there a frozen population", which a
    // refresh defeats by construction. The sharper question is whether the
    // refreshed population's LIFETIME DISTRIBUTION sits inside the carriers'
    // -- a band whose functions all live 4 checkpoints while carriers live 1
    // is still a population, just a shorter-lived one, and the epochs it
    // creates are as enumerable as the frozen functions they replaced. Report
    // the whole distribution so that comparison is possible rather than
    // inferred from a single number.
    let mut hist = vec![0usize; cps.len() + 1];
    for c in seen.values() {
        hist[*c] += 1;
    }
    let total: usize = hist.iter().sum();
    let mean: f64 = hist
        .iter()
        .enumerate()
        .map(|(life, &n)| life as f64 * n as f64)
        .sum::<f64>()
        / total.max(1) as f64;
    println!(
        "  lifetime histogram (checkpoints alive -> #functions), mean {:.2} of {}:",
        mean,
        cps.len()
    );
    let mut line = String::new();
    for (life, &n) in hist.iter().enumerate() {
        if n > 0 {
            line.push_str(&format!(" {life}:{n}"));
        }
    }
    println!("   {}", line.trim());
    // A population that stands out shows up as mass far from the bulk.
    let mut sorted_life: Vec<usize> = seen.values().copied().collect();
    sorted_life.sort_unstable();
    let pct = |q: f64| -> usize {
        sorted_life
            .get(((sorted_life.len() as f64 - 1.0) * q) as usize)
            .copied()
            .unwrap_or(0)
    };
    println!(
        "  lifetime percentiles: p50={} p90={} p99={} max={} (of {} checkpoints)",
        pct(0.50),
        pct(0.90),
        pct(0.99),
        sorted_life.last().copied().unwrap_or(0),
        cps.len()
    );
}
