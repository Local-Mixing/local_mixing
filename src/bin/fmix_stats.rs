// Read-only mixing-metric analyzer for post-mix circuits (fmix / fsplit
// output, or a raw g57 circuit as the unmixed baseline). Ingests an mpmct1 or
// g57 file plus an optional .origins sidecar and prints the stationarity
// signature: gate-type/width distribution, fanout, leeway (float-box size
// under XGate::collides), origin diffusion / adjacent-origin autocorrelation /
// window-origin diversity, wire-usage and pair co-occurrence entropy, and
// window wire-span. One [fstats] line per metric family, grep-friendly for
// convergence curves across snapshots and agreement checks across replicas.
//
// Example:
//   fmix_stats --input cdcnot_m3000_sm50_fmix_s1.txt \
//     --origins cdcnot_m3000_sm50_fmix_s1.origins.txt
use clap::Parser;
use local_mixing::postmix::format;
use local_mixing::postmix::mix::ORIGIN_SYNTH;
use local_mixing::postmix::stats;
use local_mixing::postmix::xgate::{XGate, max_wire};
use rand::SeedableRng;
use rand::rngs::StdRng;

#[derive(Parser, Debug)]
#[command(name = "fmix_stats")]
struct Args {
    /// Input circuit file
    #[arg(long)]
    input: String,
    /// Input format: mpmct1 | g57
    #[arg(long, default_value = "mpmct1")]
    input_format: String,
    /// Origins sidecar (one origin index per line, fmix --origins-out)
    #[arg(long)]
    origins: Option<String>,
    /// Gates to sample for the leeway distribution (0 = skip)
    #[arg(long, default_value_t = 20_000)]
    leeway_samples: usize,
    /// Per-direction cap on the leeway scan
    #[arg(long, default_value_t = 65_536)]
    leeway_cap: usize,
    /// Comma-separated window sizes for the wire-span metric
    #[arg(long, default_value = "32,256")]
    span_windows: String,
    /// Windows to sample per span size (and for origin diversity)
    #[arg(long, default_value_t = 2_000)]
    span_samples: usize,
    /// Reference spread scale in gates for the uniformity fraction
    /// (default: 4 * wires * log2(wires), the random-circuit PRP length scale)
    #[arg(long)]
    spread_ref: Option<f64>,
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

fn hist_line(counts: &[u64]) -> String {
    counts.iter().enumerate().map(|(i, c)| format!("{i}:{c}")).collect::<Vec<_>>().join(" ")
}

// Bucket a value into 0..=8 exact, then doubling ranges.
fn bucket_of(v: usize) -> usize {
    if v <= 8 { v } else { 9 + (usize::BITS - (v - 1).leading_zeros()) as usize - 4 }
}

fn bucket_label(b: usize) -> String {
    if b <= 8 { format!("{b}") } else { format!("{}-{}", (1usize << (b - 6)) + 1, 1usize << (b - 5)) }
}

fn bucket_hist(values: impl Iterator<Item = usize>) -> Vec<u64> {
    let mut h: Vec<u64> = Vec::new();
    for v in values {
        let b = bucket_of(v);
        if h.len() <= b {
            h.resize(b + 1, 0);
        }
        h[b] += 1;
    }
    h
}

fn bucket_hist_line(h: &[u64]) -> String {
    h.iter().enumerate().map(|(b, c)| format!("{}:{c}", bucket_label(b))).collect::<Vec<_>>().join(" ")
}

fn main() {
    let args = Args::parse();
    let (gates, file_wires): (Vec<XGate>, usize) = match args.input_format.as_str() {
        "mpmct1" => format::read_mpmct(&args.input).expect("read mpmct1 circuit"),
        "g57" => {
            let g = format::read_g57_file(&args.input).expect("read g57 circuit");
            let w = max_wire(&g) as usize + 1;
            (g, w)
        }
        other => panic!("unknown --input-format {other}"),
    };
    let wires = file_wires.max(max_wire(&gates) as usize + 1);
    let n = gates.len();
    let comp = gates.iter().filter(|g| g.comp).count();
    println!("[fstats] file={} gates={} wires={} comp={}", args.input, n, wires, comp);

    // Width / polarity.
    let max_w = gates.iter().map(|g| g.width()).max().unwrap_or(0);
    let mut width_hist = vec![0u64; max_w + 1];
    let (mut lits, mut neg) = (0u64, 0u64);
    for g in &gates {
        width_hist[g.width()] += 1;
        lits += g.width() as u64;
        neg += g.ctrls.iter().filter(|&&(_, p)| !p).count() as u64;
    }
    println!(
        "[fstats] width mean={:.3} neg_frac={:.4} hist[{}]",
        lits as f64 / n as f64,
        if lits == 0 { 0.0 } else { neg as f64 / lits as f64 },
        hist_line(&width_hist)
    );

    // Fanout.
    let fan = stats::fanouts(gates.iter(), wires);
    let zero = fan.iter().filter(|&&f| f == 0).count();
    let fmax = fan.iter().copied().max().unwrap_or(0);
    let fh = bucket_hist(fan.iter().map(|&f| f as usize));
    println!(
        "[fstats] fanout mean={:.3} zero_frac={:.3} max={} hist[{}]",
        fan.iter().map(|&f| f as u64).sum::<u64>() as f64 / n as f64,
        zero as f64 / n as f64,
        fmax,
        bucket_hist_line(&fh)
    );

    let mut rng = StdRng::seed_from_u64(args.seed);

    // Leeway (sampled).
    if args.leeway_samples > 0 && n > 1 {
        use rand::Rng;
        let mut lw: Vec<usize> = (0..args.leeway_samples)
            .map(|_| stats::leeway_at(&gates, rng.random_range(0..n), args.leeway_cap))
            .collect();
        lw.sort_unstable();
        let capped = lw.iter().filter(|&&v| v >= args.leeway_cap).count();
        let lh = bucket_hist(lw.iter().copied());
        println!(
            "[fstats] leeway mean={:.1} median={} wedged_lt25={:.3} max={} capped={} samples={} cap={} hist[{}]",
            lw.iter().sum::<usize>() as f64 / lw.len() as f64,
            lw[lw.len() / 2],
            lw.iter().filter(|&&v| v < 25).count() as f64 / lw.len() as f64,
            lw[lw.len() - 1],
            capped,
            args.leeway_samples,
            args.leeway_cap,
            bucket_hist_line(&lh)
        );
    }

    // Origin metrics.
    if let Some(path) = &args.origins {
        let s = std::fs::read_to_string(path).expect("read origins sidecar");
        let origins: Vec<u32> =
            s.lines().filter(|l| !l.is_empty()).map(|l| l.parse().expect("origin line")).collect();
        assert_eq!(origins.len(), n, "origins sidecar length != gate count");
        let real = origins.iter().filter(|&&o| o != ORIGIN_SYNTH).count();
        println!(
            "[fstats] origins real_frac={:.3} disp={:.3} diffusion={:.4} uniform={:.4} adj_autocorr={:.4} owin32={:.1}",
            real as f64 / n as f64,
            stats::origin_displacement(&origins),
            stats::origin_diffusion(&origins),
            stats::UNIFORM_STD,
            stats::adjacent_origin_autocorr(&origins),
            stats::window_origin_diversity(&origins, args.span_samples, &mut rng),
        );
        let sref = args.spread_ref.unwrap_or(4.0 * wires as f64 * (wires as f64).log2());
        let qs = [0.05, 0.25, 0.50, 0.75, 0.95];
        let (single_frac, quants, below) = stats::origin_spread_quantiles(&origins, &qs, sref);
        println!(
            "[fstats] spread_gates single_frac={:.3} p5={:.0} p25={:.0} p50={:.0} p75={:.0} p95={:.0} frac_lt_ref={:.3} ref={:.0}",
            single_frac, quants[0], quants[1], quants[2], quants[3], quants[4], below, sref
        );
    }

    // Wire usage and pair co-occurrence entropy.
    let mut tgt = vec![0u64; wires];
    let mut ctl = vec![0u64; wires];
    for g in &gates {
        tgt[g.target as usize] += 1;
        for &(w, _) in &g.ctrls {
            ctl[w as usize] += 1;
        }
    }
    let (pair_bits, distinct_pairs) = stats::pair_cooccurrence_entropy(gates.iter(), wires);
    let max_pairs = wires * (wires - 1) / 2;
    println!(
        "[fstats] wires target_H={:.3} ctrl_H={:.3} max_H={:.3} pair_H={:.3} pair_max_H={:.3} distinct_pairs={}/{}",
        stats::entropy_bits(tgt.into_iter()),
        stats::entropy_bits(ctl.into_iter()),
        (wires as f64).log2(),
        pair_bits,
        (max_pairs as f64).log2(),
        distinct_pairs,
        max_pairs
    );

    // Window wire-span, with the same-slot-count random baseline.
    let mean_support = (lits as f64 + n as f64) / n as f64;
    for w in args.span_windows.split(',').filter_map(|t| t.trim().parse::<usize>().ok()) {
        let (mean, mn, mx) = stats::window_wire_span(&gates, wires, w, args.span_samples, &mut rng);
        let slots = w as f64 * mean_support;
        let rand_baseline = wires as f64 * (1.0 - (1.0 - 1.0 / wires as f64).powf(slots));
        println!(
            "[fstats] span w={} mean={:.1} min={} max={} rand={:.1} samples={}",
            w, mean, mn, mx, rand_baseline, args.span_samples
        );
    }
}
