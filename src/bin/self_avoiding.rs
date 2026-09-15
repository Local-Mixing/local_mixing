//! Random non-reversing walk on the r57 Cayley graph, in canonical gate order.
//!
//! Vertices are reversible functions `{0,1}^n → {0,1}^n`. Edges are the
//! `n(n-1)(n-2)` three-wire generators. Every walk starts with `012;`.
//! A proposed gate is slid left through non-colliding (commuting) gates;
//! if it meets a twin, that is the trivial identity `gg` (including Coxeter
//! `(gh)² → gghh`), and we resample. The walk stops only at a permutation
//! that was already visited via a non-trivial loop.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use local_mixing::circuit::{CircuitSeq, Gate, Permutation, base_gates};
use num_bigint::BigUint;
use num_traits::Zero;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

#[derive(Parser, Debug)]
#[command(about = "Self-avoiding random walks on the r57 Cayley graph")]
struct Args {
    /// Number of wires.
    #[arg(short = 'n', long, default_value_t = 3)]
    wires: usize,

    /// Independent walks to sample.
    #[arg(short = 's', long, default_value_t = 4096)]
    samples: usize,

    /// Histogram bins for the Lehmer embedding of the intersecting permutation.
    #[arg(short = 'b', long, default_value_t = 256)]
    bins: usize,

    /// Write one row per walk (`length,rank,last_gate`).
    #[arg(short = 'o', long, default_value = "saw_walks.csv")]
    output: PathBuf,

    /// Optional pre-binned permutation histogram (`bin,lo,hi,count`).
    #[arg(long)]
    perm_hist: Option<PathBuf>,

    /// Optional length histogram (`length,count`).
    #[arg(long)]
    length_hist: Option<PathBuf>,

    /// How many of the most common intersecting permutations to print.
    #[arg(long, default_value_t = 5)]
    top: usize,

    /// Render plots with `src/bin/plot_self_avoiding.py`.
    #[arg(long)]
    plot: bool,

    /// Python interpreter used by `--plot`.
    #[arg(long, default_value = "python3")]
    python: String,

    /// Worker threads. Defaults to the host's logical CPU count.
    #[arg(short = 't', long)]
    threads: Option<usize>,
}

const START_GATE: [u16; 3] = [0, 1, 2];

struct Walk {
    length: usize,
    rank: BigUint,
    last_gate: [u16; 3],
    perm: Vec<usize>,
    /// Suffix length of the identity: `ℓ` such that `g_{t-1}⋯g_{t-ℓ} = id`.
    lag: usize,
}

fn main() {
    let args = Args::parse();
    assert!(args.wires >= 3, "r57 generators need at least 3 wires");
    assert!(args.samples > 0, "need at least one sample");
    assert!(args.bins > 0, "need at least one histogram bin");

    if let Some(threads) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .expect("failed to build thread pool");
    }

    let n = args.wires;
    let nn = 1usize << n;
    let fact = factorial(nn);
    let gates = base_gates(n);
    let gate_perms: Vec<Permutation> = gates
        .iter()
        .map(|&g| CircuitSeq { gates: vec![g] }.perm(n))
        .collect();

    let start = gates
        .iter()
        .position(|&g| g == START_GATE)
        .expect("generator set does not contain 012");

    let progress = ProgressBar::new(args.samples as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} walks ({eta})",
            )
            .unwrap()
            .progress_chars("#>-"),
    );

    let walks: Vec<Walk> = (0..args.samples)
        .into_par_iter()
        .map(|_| {
            let walk = random_walk_until_intersection(&gates, &gate_perms, start, nn);
            progress.inc(1);
            walk
        })
        .collect();
    progress.finish_and_clear();

    write_walks(&args.output, n, &walks).unwrap_or_else(|e| {
        panic!("failed to write {}: {e}", args.output.display());
    });

    let lengths: Vec<usize> = walks.iter().map(|w| w.length).collect();
    print_length_summary(&lengths, n);
    print_top_lengths(&lengths, 10);

    let perm_counts = bin_ranks(&walks, &fact, args.bins);
    print_perm_summary(&perm_counts, n, nn, &fact);
    print_gate_summary(&walks, n);
    print_top_perms(&walks, args.top);
    print_lag_summary(&walks, n, nn, &gate_perms);

    if let Some(path) = &args.length_hist {
        write_length_hist(path, &histogram_counts(&lengths)).unwrap_or_else(|e| {
            panic!("failed to write {}: {e}", path.display());
        });
    }
    if let Some(path) = &args.perm_hist {
        write_perm_hist(path, &perm_counts, &fact).unwrap_or_else(|e| {
            panic!("failed to write {}: {e}", path.display());
        });
    }

    println!("wrote {}", args.output.display());

    if args.plot {
        plot_walks(&args.python, &args.output, n, args.bins);
    }
}

/// Uniform random walk on canonical circuits. Trivial `gg` cancellations
/// (after sliding through commuting gates) are skipped; the walk stops at
/// the first non-trivial permutation self-intersection.
///
/// The recorded length includes the intersecting gate. `last_gate` is the
/// last successful (non-cancelling, non-intersecting) generator.
fn random_walk_until_intersection(
    gates: &[[u16; 3]],
    gate_perms: &[Permutation],
    start: usize,
    nn: usize,
) -> Walk {
    let mut circuit = vec![gates[start]];
    let mut current = Permutation::id_perm(nn);
    let mut visited: FxHashMap<Permutation, usize> = FxHashMap::default();

    let mut scratch = Permutation { data: vec![0; nn] };
    visited.insert(current.clone(), 1);
    compose_into(&gate_perms[start], &current, &mut scratch);
    visited.insert(scratch.clone(), 1);
    std::mem::swap(&mut current, &mut scratch);

    let mut last_gate = gates[start];
    let mut length = 1usize;

    loop {
        let Some((i, pos)) = sample_nontrivial_gate(&circuit, gates) else {
            return Walk {
                length,
                rank: lehmer_rank(&current.data),
                last_gate,
                perm: current.data.clone(),
                lag: 0,
            };
        };

        compose_into(&gate_perms[i], &current, &mut scratch);
        length += 1;
        if let Some(&prev) = visited.get(&scratch) {
            return Walk {
                length,
                rank: lehmer_rank(&scratch.data),
                last_gate,
                perm: scratch.data.clone(),
                lag: length - prev,
            };
        }

        circuit.insert(pos, gates[i]);
        visited.insert(scratch.clone(), length);
        std::mem::swap(&mut current, &mut scratch);
        last_gate = gates[i];
    }
}

/// Uniform among generators that do not cancel as `gg` after a commuting slide.
fn sample_nontrivial_gate(circuit: &[[u16; 3]], gates: &[[u16; 3]]) -> Option<(usize, usize)> {
    for _ in 0..gates.len() * 4 {
        let i = fastrand::usize(..gates.len());
        if let Some(pos) = canonical_insert_pos(circuit, gates[i]) {
            return Some((i, pos));
        }
    }
    let mut ok = Vec::new();
    for (i, &g) in gates.iter().enumerate() {
        if let Some(pos) = canonical_insert_pos(circuit, g) {
            ok.push((i, pos));
        }
    }
    if ok.is_empty() {
        None
    } else {
        Some(ok[fastrand::usize(..ok.len())])
    }
}

/// Where `g` would sit in a canonical circuit, or `None` if it meets a twin
/// (the reorder would produce `gg` and cancel).
fn canonical_insert_pos(circuit: &[[u16; 3]], g: [u16; 3]) -> Option<usize> {
    let mut to_swap = None;
    let mut j = circuit.len();
    while j > 0 {
        j -= 1;
        if Gate::collides_index(&g, &circuit[j]) {
            break;
        }
        if circuit[j] == g {
            return None;
        }
        if !Gate::ordered_index(&circuit[j], &g) {
            to_swap = Some(j);
        }
    }
    Some(to_swap.unwrap_or(circuit.len()))
}

/// `(left ∘ right)[i] = left[right[i]]`, written into `out`.
fn compose_into(left: &Permutation, right: &Permutation, out: &mut Permutation) {
    debug_assert_eq!(left.data.len(), right.data.len());
    debug_assert_eq!(out.data.len(), right.data.len());
    for i in 0..right.data.len() {
        out.data[i] = left.data[right.data[i]];
    }
}

/// Exact Lehmer/factoradic rank. Identity is 0; reversal is `N! - 1`.
fn lehmer_rank(p: &[usize]) -> BigUint {
    let n = p.len();
    let mut rank = BigUint::zero();
    if n <= 64 {
        let mut remaining = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
        for (i, &value) in p.iter().enumerate() {
            let bit = 1u64 << value;
            let smaller = (remaining & (bit - 1)).count_ones();
            rank = rank * (n - i) + smaller;
            remaining &= !bit;
        }
        return rank;
    }

    let mut used = vec![false; n];
    for (i, &value) in p.iter().enumerate() {
        let smaller = used[..value].iter().filter(|&&u| !u).count();
        rank = rank * (n - i) + smaller;
        used[value] = true;
    }
    rank
}

/// 64-bit tag in `[0, 2^64)`: `⌊rank · 2^64 / N!⌋`. Identity → 0; reversal
/// sits at the top of the range.
#[cfg(test)]
fn lehmer_tag(rank: &BigUint, factorial: &BigUint) -> u64 {
    let scaled: BigUint = (rank << 64) / factorial;
    u64::try_from(scaled).unwrap_or(u64::MAX)
}

fn factorial(n: usize) -> BigUint {
    let mut f = BigUint::from(1u8);
    for i in 2..=n {
        f *= i;
    }
    f
}

fn bin_ranks(walks: &[Walk], factorial: &BigUint, bins: usize) -> Vec<usize> {
    let mut counts = vec![0usize; bins];
    let bins_big = BigUint::from(bins);
    for walk in walks {
        let bin = usize::try_from((&walk.rank * &bins_big) / factorial).unwrap_or(bins - 1);
        counts[bin.min(bins - 1)] += 1;
    }
    counts
}

fn histogram_counts(values: &[usize]) -> Vec<(usize, usize)> {
    let mut counts = Vec::new();
    if values.is_empty() {
        return counts;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let mut cur = sorted[0];
    let mut n = 0usize;
    for &v in &sorted {
        if v == cur {
            n += 1;
        } else {
            counts.push((cur, n));
            cur = v;
            n = 1;
        }
    }
    counts.push((cur, n));
    counts
}

fn print_length_summary(lengths: &[usize], wires: usize) {
    let mut sorted = lengths.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    let min = sorted[0];
    let max = sorted[n - 1];
    let median = sorted[n / 2];
    let mean = sorted.iter().sum::<usize>() as f64 / n as f64;
    println!("lengths  n={n}  min={min}  median={median}  mean={mean:.2}  max={max}");
    print_length_fit(lengths, wires);
}

fn print_length_fit(lengths: &[usize], wires: usize) {
    if let Some(line) = length_fit_line(lengths, wires) {
        println!("{line}");
    }
}

fn length_fit_line(lengths: &[usize], wires: usize) -> Option<String> {
    if lengths.is_empty() {
        return None;
    }
    let (lambda, k) = fit_weibull(lengths)?;
    let mut line = format!(
        "length fit  Weibull  P(N)∝ N^(k-1) exp(-λ N^k)  k={k:.4}  λ={lambda:.6e}"
    );
    if wires <= 3 {
        line.push_str(&format!(
            "  (k=2 is Rayleigh; birthday 8!={})",
            factorial_f64(8) as u64
        ));
    }
    Some(line)
}

fn fit_weibull(lengths: &[usize]) -> Option<(f64, f64)> {
    let xs: Vec<f64> = lengths
        .iter()
        .copied()
        .filter(|&t| t > 0)
        .map(|t| t as f64)
        .collect();
    if xs.len() < 2 {
        return None;
    }
    let n = xs.len() as f64;
    let logs: Vec<f64> = xs.iter().map(|t| t.ln()).collect();
    let mean_log = logs.iter().sum::<f64>() / n;
    let var_log = logs.iter().map(|u| (u - mean_log).powi(2)).sum::<f64>() / n;
    if var_log <= 1e-18 {
        return None;
    }
    let mut k = std::f64::consts::PI / (6.0 * var_log).sqrt();
    k = k.clamp(0.2, 8.0);

    let powers = |shape: f64| -> (f64, f64, f64) {
        let mut den = 0.0;
        let mut num = 0.0;
        let mut quad = 0.0;
        for &lg in &logs {
            let w = (shape * lg).exp();
            den += w;
            num += w * lg;
            quad += w * lg * lg;
        }
        (den, num, quad)
    };

    for _ in 0..30 {
        let (den, num, quad) = powers(k);
        if den <= 0.0 {
            return None;
        }
        let g = 1.0 / k + mean_log - num / den;
        let gp = -1.0 / (k * k) - (den * quad - num * num) / (den * den);
        if gp.abs() < 1e-18 {
            break;
        }
        let nxt = (k - g / gp).clamp(0.2, 8.0);
        if (nxt - k).abs() < 1e-10 {
            k = nxt;
            break;
        }
        k = nxt;
    }
    let (den, _, _) = powers(k);
    if den <= 0.0 {
        return None;
    }
    let lambda = n / den;
    if lambda <= 0.0 || k <= 0.0 {
        None
    } else {
        Some((lambda, k))
    }
}

fn print_top_lengths(lengths: &[usize], k: usize) {
    let mut counts = histogram_counts(lengths);
    counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let shown = counts.len().min(k);
    println!("top lengths  unique={}  showing {shown}:", counts.len());
    for (i, (length, count)) in counts.into_iter().take(shown).enumerate() {
        println!("  {}.  count={count}  length={length}", i + 1);
    }
}

fn print_perm_summary(counts: &[usize], wires: usize, nn: usize, fact: &BigUint) {
    let occupied = counts.iter().filter(|&&c| c > 0).count();
    let peak = counts.iter().copied().max().unwrap_or(0);
    println!(
        "final perms  lehmer bins={}/{} occupied  peak={peak}  id=0  rev={fact}  (S_{{{nn}}}, n={wires})",
        occupied,
        counts.len()
    );
}

fn print_gate_summary(walks: &[Walk], n: usize) {
    let mut counts: Vec<([u16; 3], usize)> = Vec::new();
    for walk in walks {
        if let Some((_, c)) = counts.iter_mut().find(|(g, _)| *g == walk.last_gate) {
            *c += 1;
        } else {
            counts.push((walk.last_gate, 1));
        }
    }
    counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let shown = counts.len().min(8);
    let parts: Vec<String> = counts[..shown]
        .iter()
        .map(|(g, c)| format!("{}:{c}", format_gate(*g, n)))
        .collect();
    println!(
        "last gates  unique={}  top [{}]",
        counts.len(),
        parts.join("  ")
    );
}

fn print_top_perms(walks: &[Walk], k: usize) {
    let mut counts: FxHashMap<BigUint, (usize, Vec<usize>)> = FxHashMap::default();
    for walk in walks {
        counts
            .entry(walk.rank.clone())
            .and_modify(|(c, _)| *c += 1)
            .or_insert((1, walk.perm.clone()));
    }
    let mut items: Vec<_> = counts.into_iter().filter(|(_, (c, _))| *c > 1).collect();
    if items.is_empty() {
        return;
    }
    items.sort_by(|a, b| b.1.0.cmp(&a.1.0).then_with(|| a.0.cmp(&b.0)));
    let shown = items.len().min(k);
    println!("top perms  repeats={}  showing {shown}:", items.len());
    for (i, (rank, (count, perm))) in items.into_iter().take(shown).enumerate() {
        let pi = perm
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        println!("  {}.  n={count}  rank={rank}  π=[{pi}]", i + 1);
    }
}

fn print_lag_summary(walks: &[Walk], n: usize, nn: usize, gate_perms: &[Permutation]) {
    let mut lags: Vec<usize> = walks.iter().map(|w| w.lag).collect();
    lags.sort_unstable();
    let mean_lag = lags.iter().sum::<usize>() as f64 / lags.len() as f64;
    let mean_len = walks.iter().map(|w| w.length).sum::<usize>() as f64 / walks.len() as f64;

    let mut hist = Vec::new();
    let mut cur = lags[0];
    let mut c = 0usize;
    for &lag in &lags {
        if lag == cur {
            c += 1;
        } else {
            hist.push((cur, c));
            cur = lag;
            c = 1;
        }
    }
    hist.push((cur, c));

    let head: Vec<String> = hist
        .iter()
        .take(12)
        .map(|(lag, count)| format!("ℓ={lag}:{count}"))
        .collect();
    println!(
        "collision lag  mean={mean_lag:.2}  (relation suffix length; 4 = (gh)², 6+ = longer relators)"
    );
    println!("  {}", head.join("  "));

    if let Some(order) = group_order_hint(n, nn, gate_perms) {
        let birthday = (std::f64::consts::PI * order / 2.0).sqrt();
        println!(
            "birthday on |G|≈{order:.4e} would give E[T]≈{birthday:.0};  observed mean T={mean_len:.1}  ({:.3}× birthday)",
            mean_len / birthday
        );
    }
}

/// Exact Cayley-graph order when it is small; `Alt(2^n)` otherwise for n=4.
fn group_order_hint(n: usize, nn: usize, gate_perms: &[Permutation]) -> Option<f64> {
    if n == 3 {
        return Some(cayley_order(gate_perms, nn) as f64);
    }
    if n == 4 {
        // All 24 r57 generators on 4 wires generate Alt(16).
        return Some(factorial_f64(nn) / 2.0);
    }
    None
}

fn factorial_f64(n: usize) -> f64 {
    (2..=n).fold(1.0, |acc, i| acc * i as f64)
}

fn cayley_order(gate_perms: &[Permutation], nn: usize) -> usize {
    let id = Permutation::id_perm(nn);
    let mut seen = FxHashMap::default();
    seen.insert(id.clone(), ());
    let mut stack = vec![id];
    seen.insert(stack[0].clone(), ());
    let mut scratch = Permutation { data: vec![0; nn] };
    while let Some(p) = stack.pop() {
        for g in gate_perms {
            compose_into(g, &p, &mut scratch);
            if !seen.contains_key(&scratch) {
                seen.insert(scratch.clone(), ());
                stack.push(scratch.clone());
            }
        }
    }
    seen.len()
}

fn format_gate(g: [u16; 3], n: usize) -> String {
    if n < 10 {
        format!("{}{}{}", g[0], g[1], g[2])
    } else {
        format!("{}_{}_{}", g[0], g[1], g[2])
    }
}

fn write_walks(path: &Path, wires: usize, walks: &[Walk]) -> std::io::Result<()> {
    let nn = 1usize << wires;
    let fact = factorial(nn);
    let mut out = BufWriter::new(File::create(path)?);
    writeln!(out, "# wires={wires} domain={nn} factorial={fact}")?;
    writeln!(out, "length,rank,last_gate,lag")?;
    for walk in walks {
        writeln!(
            out,
            "{},{},{},{}",
            walk.length,
            walk.rank,
            format_gate(walk.last_gate, wires),
            walk.lag
        )?;
    }
    out.flush()
}

fn write_length_hist(path: &Path, counts: &[(usize, usize)]) -> std::io::Result<()> {
    let mut out = BufWriter::new(File::create(path)?);
    writeln!(out, "length,count")?;
    for &(length, count) in counts {
        writeln!(out, "{length},{count}")?;
    }
    out.flush()
}

fn write_perm_hist(path: &Path, counts: &[usize], fact: &BigUint) -> std::io::Result<()> {
    let mut out = BufWriter::new(File::create(path)?);
    writeln!(out, "bin,lo,hi,count")?;
    let bins = counts.len();
    for (i, &count) in counts.iter().enumerate() {
        let lo = fact * i / bins;
        let hi = fact * (i + 1) / bins;
        writeln!(out, "{i},{lo},{hi},{count}")?;
    }
    out.flush()
}

fn plot_walks(python: &str, csv: &Path, wires: usize, bins: usize) {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/bin/plot_self_avoiding.py");
    let status = Command::new(python)
        .env("MPLBACKEND", "Agg")
        .arg(&script)
        .arg("--csv")
        .arg(csv)
        .arg("--wires")
        .arg(wires.to_string())
        .arg("--bins")
        .arg(bins.to_string())
        .status()
        .unwrap_or_else(|e| panic!("failed to spawn {python}: {e}"));
    if !status.success() {
        panic!("{} exited with {status}", script.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lehmer_identity_is_zero() {
        let p = Permutation::id_perm(8);
        assert_eq!(lehmer_rank(&p.data), BigUint::zero());
        assert_eq!(lehmer_tag(&BigUint::zero(), &factorial(8)), 0);
    }

    #[test]
    fn length_fit_weibull_equation() {
        let lengths = [10usize, 20, 30];
        let (lambda, k) = fit_weibull(&lengths).unwrap();
        let line3 = length_fit_line(&lengths, 3).unwrap();
        let line4 = length_fit_line(&lengths, 4).unwrap();
        assert!(line3.contains("Weibull"), "{line3}");
        assert!(line4.contains("Weibull"), "{line4}");
        assert!(line3.contains("k=2 is Rayleigh"), "{line3}");
        assert!(!line4.contains("k=2 is Rayleigh"), "{line4}");
        assert!((k - 2.737).abs() < 0.05, "k={k}");
        assert!(lambda > 0.0, "λ={lambda}");
    }

    #[test]
    fn lehmer_reverse_is_factorial_minus_one() {
        let n = 8usize;
        let p = Permutation {
            data: (0..n).rev().collect(),
        };
        let fact = factorial(n);
        assert_eq!(lehmer_rank(&p.data), &fact - 1u8);
    }

    #[test]
    fn compose_matches_full_circuit_perm() {
        let n = 4usize;
        let nn = 1 << n;
        let mut c = CircuitSeq { gates: vec![] };
        let mut p = Permutation::id_perm(nn);
        let mut scratch = Permutation { data: vec![0; nn] };
        for _ in 0..16 {
            let mut pins = [0u16; 3];
            pins[0] = fastrand::u16(0..n as u16);
            pins[1] = fastrand::u16(0..n as u16);
            while pins[1] == pins[0] {
                pins[1] = fastrand::u16(0..n as u16);
            }
            pins[2] = fastrand::u16(0..n as u16);
            while pins[2] == pins[0] || pins[2] == pins[1] {
                pins[2] = fastrand::u16(0..n as u16);
            }
            let g = CircuitSeq { gates: vec![pins] }.perm(n);
            compose_into(&g, &p, &mut scratch);
            std::mem::swap(&mut p, &mut scratch);
            c.gates.push(pins);
            assert_eq!(p, c.perm(n));
        }
    }

    #[test]
    fn lehmer_small_explicit() {
        // Lex order on S_3: 012=0, 021=1, 102=2, 120=3, 201=4, 210=5.
        assert_eq!(lehmer_rank(&[0, 1, 2]), BigUint::from(0u8));
        assert_eq!(lehmer_rank(&[0, 2, 1]), BigUint::from(1u8));
        assert_eq!(lehmer_rank(&[1, 0, 2]), BigUint::from(2u8));
        assert_eq!(lehmer_rank(&[1, 2, 0]), BigUint::from(3u8));
        assert_eq!(lehmer_rank(&[2, 0, 1]), BigUint::from(4u8));
        assert_eq!(lehmer_rank(&[2, 1, 0]), BigUint::from(5u8));
    }

    #[test]
    fn start_gate_is_012() {
        for n in 3..=6 {
            let gates = base_gates(n);
            let start = gates.iter().position(|&g| g == START_GATE).unwrap();
            assert_eq!(gates[start], [0, 1, 2]);
        }
    }

    #[test]
    fn walk_records_last_successful_generator() {
        let n = 3usize;
        let gates = base_gates(n);
        let start = gates.iter().position(|&g| g == START_GATE).unwrap();
        let gate_perms: Vec<Permutation> = gates
            .iter()
            .map(|&g| CircuitSeq { gates: vec![g] }.perm(n))
            .collect();
        let walk = random_walk_until_intersection(&gates, &gate_perms, start, 1 << n);
        assert!(gates.contains(&walk.last_gate));
        assert!(walk.length >= 2);
    }

    #[test]
    fn insert_pos_matches_canonicalize() {
        let n = 4usize;
        let gens = base_gates(n);
        let mut circuit = vec![START_GATE];
        for _ in 0..32 {
            let g = gens[fastrand::usize(..gens.len())];
            let Some(pos) = canonical_insert_pos(&circuit, g) else {
                continue;
            };
            let mut incremental = circuit.clone();
            incremental.insert(pos, g);
            let mut full = CircuitSeq {
                gates: circuit.clone(),
            };
            full.gates.push(g);
            full.canonicalize();
            assert_eq!(incremental, full.gates);
            circuit = incremental;
        }
    }

    #[test]
    fn commuting_twin_is_rejected() {
        let a = [0u16, 1, 2];
        let mut h = None;
        for g in base_gates(4) {
            if g != a && !Gate::collides_index(&a, &g) {
                h = Some(g);
                break;
            }
        }
        let h = h.expect("n=4 has a gate that commutes with 012");
        let mut circuit = vec![a];
        let pos = canonical_insert_pos(&circuit, h).unwrap();
        circuit.insert(pos, h);
        assert!(
            canonical_insert_pos(&circuit, a).is_none(),
            "inserting a after commuting ah should cancel"
        );
        assert!(
            canonical_insert_pos(&circuit, h).is_none(),
            "inserting h after commuting ah should cancel"
        );
    }

    #[test]
    fn tag_is_monotone_in_rank() {
        let fact = factorial(8);
        let a = BigUint::from(1u8);
        let b = BigUint::from(1000u32);
        assert!(lehmer_tag(&a, &fact) < lehmer_tag(&b, &fact));
    }
}
