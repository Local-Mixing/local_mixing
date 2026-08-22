// Prefix-distance heatmap between two circuits c and d that compute the same
// permutation. H(i,j) = mean over sampled inputs x of the Hamming distance
// between c_i(x) (the state after the first i gates of c) and d_j(x) (after the
// first j gates of d). Endpoints match (H at the full-length corner is 0), so
// the map shows whether d's scrambled trajectory ever passes near c's
// intermediate states.
//
// Inputs are packed 64 to a pass via the bit-sliced XGate evaluator (lane l =
// input sample l); --batches passes give 64*batches samples. Output is a raw
// row-major f32 matrix (rows = c prefixes, cols = d prefixes) plus a JSON
// sidecar with the index vectors and the distribution's mean/std for color
// calibration.
//
// Example:
//   hmap --c A.txt --d final.txt --c-step 10 --d-step 50 --out full
//   hmap --c A.txt --d final.txt --c-step 10 --d-start 0   --d-end 50000 --d-step 25 --out first50k
//   hmap --c A.txt --d final.txt --c-step 10 --d-from-end 50000 --d-step 25 --out last50k
use clap::Parser;
use local_mixing::engine::format::{read_g57_file, read_mpmct};
use local_mixing::circuit::xgate::{XGate, max_wire};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::io::Write;

#[derive(Parser, Debug)]
#[command(name = "hmap")]
struct Args {
    /// Reference circuit c
    #[arg(long)]
    c: String,
    /// Comparison circuit d
    #[arg(long)]
    d: String,
    #[arg(long, default_value = "g57")]
    c_format: String,
    #[arg(long, default_value = "mpmct1")]
    d_format: String,
    /// Stride between sampled prefix lengths of c (rows)
    #[arg(long, default_value_t = 10)]
    c_step: usize,
    /// First sampled prefix length of d (columns)
    #[arg(long, default_value_t = 0)]
    d_start: usize,
    /// Last sampled prefix length of d (default = |d|)
    #[arg(long)]
    d_end: Option<usize>,
    /// Convenience: set d_start = |d| - N (the last-N-gates window)
    #[arg(long)]
    d_from_end: Option<usize>,
    /// Stride between sampled prefix lengths of d (columns)
    #[arg(long, default_value_t = 50)]
    d_step: usize,
    /// Number of 64-input passes; total samples = 64 * batches
    #[arg(long, default_value_t = 2)]
    batches: usize,
    #[arg(long, default_value_t = 12345)]
    seed: u64,
    /// Input distribution for the sampled state (see --seed-pos for WHERE it is
    /// injected). One of:
    ///   uniform        — each wire-bit iid fair coin (default)
    ///   bern:P         — each wire-bit iid Bernoulli(P), e.g. bern:0.1 (sparse)
    ///   fix:LO:HI      — axis-aligned coset: wires [LO,HI) pinned to one random
    ///                    constant (the coset offset, fixed across all lanes and
    ///                    batches, re-drawn per --seed), the rest fair coins.
    ///                    e.g. fix:0:128 pins the low half, varies the high half.
    #[arg(long, default_value = "uniform")]
    input_dist: String,
    /// Inject the sampled state at prefix position i0 of c (a landmark), then
    /// invert c's first i0 gates to recover the input x and run both circuits
    /// forward from it. i0=0 (default) samples x at the input as usual. A
    /// non-uniform --input-dist at i0>0 concentrates the probe around that
    /// landmark (e.g. the Feistel midpoint). Every XGate is an involution, so
    /// the inverse of the prefix is its gates applied in reverse order.
    #[arg(long, default_value_t = 0)]
    seed_pos: usize,
    /// Output prefix: writes <out>.bin (row-major f32) and <out>.meta.json
    #[arg(long)]
    out: String,
}

enum Dist {
    Uniform,
    Bern(f64),
    Fix { lo: usize, hi: usize },
}

fn parse_dist(s: &str) -> Dist {
    if s == "uniform" {
        Dist::Uniform
    } else if let Some(p) = s.strip_prefix("bern:") {
        Dist::Bern(p.parse().expect("bern:P needs a float P"))
    } else if let Some(rest) = s.strip_prefix("fix:") {
        let parts: Vec<&str> = rest.split(':').collect();
        assert_eq!(parts.len(), 2, "fix needs LO:HI");
        Dist::Fix { lo: parts[0].parse().expect("fix LO"), hi: parts[1].parse().expect("fix HI") }
    } else {
        panic!("unknown --input-dist {s} (uniform | bern:P | fix:LO:HI)");
    }
}

// One batch of 64 lanes: per-wire u64 (bit l = lane l). `x0` supplies the fixed
// coset offset for Fix (a single bit per wire, broadcast to all 64 lanes).
fn sample_state(dist: &Dist, nw: usize, rng: &mut StdRng, x0: &[bool]) -> Vec<u64> {
    (0..nw)
        .map(|w| match dist {
            Dist::Uniform => rng.random::<u64>(),
            Dist::Bern(p) => {
                let mut word = 0u64;
                for l in 0..64 {
                    if rng.random::<f64>() < *p {
                        word |= 1u64 << l;
                    }
                }
                word
            }
            Dist::Fix { lo, hi } => {
                if w >= *lo && w < *hi {
                    if x0[w] { !0u64 } else { 0u64 }
                } else {
                    rng.random::<u64>()
                }
            }
        })
        .collect()
}

// Sampled prefix lengths: start, start+step, ..., up to and including end.
fn indices(start: usize, end: usize, step: usize) -> Vec<usize> {
    let mut v = Vec::new();
    let mut i = start;
    while i < end {
        v.push(i);
        i += step;
    }
    v.push(end);
    v
}

// Run `gates`, snapshotting the lane state at each prefix length in `idx`
// (sorted, first element >= 0). Returns one Vec<u64> (nw lanes-words) per index.
fn snapshots(gates: &[XGate], nw: usize, init: &[u64], idx: &[usize]) -> Vec<Vec<u64>> {
    let mut state = init.to_vec();
    let mut out = Vec::with_capacity(idx.len());
    let mut k = 0;
    for pos in 0..=gates.len() {
        while k < idx.len() && idx[k] == pos {
            out.push(state.clone());
            k += 1;
        }
        if pos < gates.len() {
            gates[pos].apply_lanes(&mut state);
        }
    }
    debug_assert_eq!(out.len(), idx.len());
    out
}

fn main() {
    let args = Args::parse();
    let c: Vec<XGate> = match args.c_format.as_str() {
        "g57" => read_g57_file(&args.c).expect("read c (g57)"),
        "mpmct1" => read_mpmct(&args.c).expect("read c (mpmct1)").0,
        other => panic!("unknown --c-format {other}"),
    };
    let d: Vec<XGate> = match args.d_format.as_str() {
        "g57" => read_g57_file(&args.d).expect("read d (g57)"),
        "mpmct1" => read_mpmct(&args.d).expect("read d (mpmct1)").0,
        other => panic!("unknown --d-format {other}"),
    };
    let nw = max_wire(&c).max(max_wire(&d)) as usize + 1;
    let dist = parse_dist(&args.input_dist);
    assert!(args.seed_pos <= c.len(), "--seed-pos {} exceeds |c|={}", args.seed_pos, c.len());

    let d_end = args.d_end.unwrap_or(d.len()).min(d.len());
    let d_start = args.d_from_end.map(|n| d.len().saturating_sub(n)).unwrap_or(args.d_start).min(d_end);
    let i_idx = indices(0, c.len(), args.c_step.max(1));
    let j_idx = indices(d_start, d_end, args.d_step.max(1));
    let (rows, cols) = (i_idx.len(), j_idx.len());

    println!(
        "[hmap] c={} gates, d={} gates, nw={}; rows(c)={} [0..{} step {}], cols(d)={} [{}..{} step {}], samples={}, dist={}, seed_pos={}",
        c.len(), d.len(), nw, rows, c.len(), args.c_step, cols, d_start, d_end, args.d_step, 64 * args.batches, args.input_dist, args.seed_pos
    );

    // Accumulate summed popcount over all batches; divide by total samples.
    let mut acc = vec![0u64; rows * cols];
    let mut rng = StdRng::seed_from_u64(args.seed);
    // Coset offset (used only by Fix): one bit per wire, fixed for the whole run.
    let x0: Vec<bool> = match &dist {
        Dist::Fix { .. } => (0..nw).map(|_| rng.random::<bool>()).collect(),
        _ => vec![false; nw],
    };
    for _ in 0..args.batches.max(1) {
        // Sample the state at the landmark, then invert c's prefix [0, seed_pos)
        // to recover the input x (each gate is its own inverse -> reverse order).
        let mut init = sample_state(&dist, nw, &mut rng, &x0);
        for pos in (0..args.seed_pos).rev() {
            c[pos].apply_lanes(&mut init);
        }
        let cs = snapshots(&c, nw, &init, &i_idx);
        let ds = snapshots(&d, nw, &init, &j_idx);
        for (ri, cv) in cs.iter().enumerate() {
            let base = ri * cols;
            for (cj, dv) in ds.iter().enumerate() {
                let mut pc = 0u64;
                for w in 0..nw {
                    pc += (cv[w] ^ dv[w]).count_ones() as u64;
                }
                acc[base + cj] += pc;
            }
        }
    }

    let denom = (64 * args.batches.max(1)) as f64;
    let mut mat = vec![0f32; rows * cols];
    let mut sum = 0f64;
    let mut sumsq = 0f64;
    for (m, &a) in mat.iter_mut().zip(acc.iter()) {
        let v = a as f64 / denom;
        *m = v as f32;
        sum += v;
        sumsq += v * v;
    }
    let n = (rows * cols) as f64;
    let mu = sum / n;
    let sigma = (sumsq / n - mu * mu).max(0.0).sqrt();
    println!("[hmap] distribution: mean={:.3} std={:.3} bits (of {} wires)", mu, sigma, nw);

    let bin = format!("{}.bin", args.out);
    let mut f = std::fs::File::create(&bin).expect("create bin");
    let bytes: Vec<u8> = mat.iter().flat_map(|v| v.to_le_bytes()).collect();
    f.write_all(&bytes).expect("write bin");

    let meta = format!(
        "{{\"rows\":{},\"cols\":{},\"nw\":{},\"mu\":{:.6},\"sigma\":{:.6},\"dist\":\"{}\",\"seed_pos\":{},\"i_idx\":[{}],\"j_idx\":[{}]}}",
        rows, cols, nw, mu, sigma, args.input_dist, args.seed_pos,
        i_idx.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","),
        j_idx.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","),
    );
    std::fs::write(format!("{}.meta.json", args.out), meta).expect("write meta");
    println!("[hmap] wrote {} ({}x{} f32) and {}.meta.json", bin, rows, cols, args.out);
}
