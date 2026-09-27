//! segment_stats — per-wire / per-segment structural and firing statistics of
//! an mpmct1 circuit.
//!
//! A *wire segment* is the value a wire carries between two consecutive writes
//! (gates targeting it); segment 0 of a wire is its input value. Every gate
//! creates exactly one segment (on its target). Structural stats come from
//! the gate list alone; firing stats evaluate the circuit on `--samples`
//! random *legal* inputs (random x on wires 0..n, zeros elsewhere — the
//! zero-slice convention used by every other measurement in the project).
//!
//! Per wire:
//!   touches / writes / reads      gates touching the wire in any role / as
//!                                 target / as control;
//!   fanout                        per segment, the number of gates reading
//!                                 it as a control (median + mean per wire);
//!   commutation box               per gate, the maximal contiguous run of the
//!                                 gate list around the gate in which it
//!                                 commutes with every other gate (= its float
//!                                 range, cf. the commutation-hardness
//!                                 preprocessor); two gates commute iff
//!                                 neither's target is a control of the other.
//!                                 Reported per wire as the median over gates
//!                                 touching (and separately, targeting) it;
//!   consecutive writes            maximal runs of writes to the wire with no
//!                                 read of the wire in between (mean + median
//!                                 of the run length per wire).
//!
//! Firing (per segment, over the sampled inputs):
//!   a segment FIRES on x when its creating gate fires on x, i.e. the write
//!   toggles the wire (fires = comp XOR AND lits, exactly the XGate rule).
//!   The same statistics are also computed for the segment VALUE (wire = 1),
//!   labelled `value_*`, because "fires" is sometimes used loosely for that.
//!   * per-input co-firing:  E_x[ #pairs of segments both firing on x ] /
//!                           #pairs (exact, from per-input firing counts);
//!   * sampled pairs:        fraction of (uniformly random) segment pairs that
//!                           fire together on at least one sampled input, the
//!                           histogram of their co-firing rate, the lift
//!                           rate/(p_a p_b) against independence, and the
//!                           identical / complementary / disjoint shares;
//!   * signature classes:    segments grouped by identical firing vectors
//!                           (exact co-firing clusters).
//! A random sample of segment firing/value vectors is dumped for external
//! clusterability analysis (Hopkins statistic, k-means silhouette, ...).
//!
//! Usage: segment_stats --g <circuit.mpmct1> --out <prefix> [--n 128]
//!        [--samples 4096] [--pairs 20000000] [--dump 8192] [--seed S]
//! Writes <prefix>.json, <prefix>.wires.csv, <prefix>.sample.{fire,value}.bin,
//! <prefix>.sample.meta.json.

use clap::Parser;
use local_mixing::circuit::xgate::XGate;
use local_mixing::engine::format::read_mpmct;
use rayon::prelude::*;
use std::io::Write;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "segment_stats")]
struct Args {
    /// Circuit (mpmct1).
    #[arg(long)]
    g: String,
    /// Output prefix.
    #[arg(long)]
    out: String,
    /// Logical input width: random x on wires 0..n-1, zeros elsewhere.
    #[arg(long, default_value_t = 128)]
    n: usize,
    /// Number of sampled legal inputs (rounded down to a multiple of 64).
    #[arg(long, default_value_t = 4096)]
    samples: usize,
    /// Number of random segment pairs to sample (per mode).
    #[arg(long, default_value_t = 20_000_000)]
    pairs: u64,
    /// Number of segments whose firing/value vectors are dumped.
    #[arg(long, default_value_t = 8192)]
    dump: usize,
    /// RNG seed for inputs / pair sampling (measurement reproducibility only).
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long, default_value_t = 0)]
    threads: usize,
}

fn median_u32(v: &mut [u32]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_unstable();
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2] as f64
    } else {
        (v[n / 2 - 1] as f64 + v[n / 2] as f64) / 2.0
    }
}

fn mean_u32(v: &[u32]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64
}

fn mean_var(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    (mean, var)
}

fn conflicts(a: &XGate, b: &XGate) -> bool {
    a.ctrls.iter().any(|&(w, _)| w == b.target) || b.ctrls.iter().any(|&(w, _)| w == a.target)
}

fn log2_bin(x: u32) -> usize {
    // 0 -> 0, 1 -> 1, 2..3 -> 2, 4..7 -> 3, ...
    if x == 0 {
        0
    } else {
        (32 - x.leading_zeros()) as usize
    }
}

fn json_arr_f(v: &[f64]) -> String {
    let parts: Vec<String> = v
        .iter()
        .map(|x| {
            if x.is_nan() {
                "null".to_string()
            } else {
                format!("{:.6}", x)
            }
        })
        .collect();
    format!("[{}]", parts.join(","))
}

fn json_arr_u(v: &[u64]) -> String {
    let parts: Vec<String> = v.iter().map(|x| x.to_string()).collect();
    format!("[{}]", parts.join(","))
}

fn fnum(x: f64) -> String {
    if x.is_nan() {
        "null".to_string()
    } else {
        format!("{:.8}", x)
    }
}

#[derive(Clone)]
struct PairStats {
    total: u64,
    any: u64,
    identical: u64,
    complement: u64,
    disjoint_nonzero: u64,
    both_nonzero: u64,
    lift_undef: u64,
    rate_hist: Vec<u64>, // 0: rate == 0; 1..=64: ceil(rate*64)
    lift_hist: Vec<u64>, // log2(lift) in [-8, 8] step 0.5 -> 33 bins, clamped
    sum_rate: f64,
    sum_expected: f64,
    sum_dev2: f64, // sum over pairs of (rate - pa*pb)^2
    first_word: Vec<u64>, // index of the first 64-input word where the pair co-fires; last bin = never
}

impl PairStats {
    fn new() -> Self {
        PairStats {
            total: 0,
            any: 0,
            identical: 0,
            complement: 0,
            disjoint_nonzero: 0,
            both_nonzero: 0,
            lift_undef: 0,
            rate_hist: vec![0; 65],
            lift_hist: vec![0; 33],
            sum_rate: 0.0,
            sum_expected: 0.0,
            sum_dev2: 0.0,
            first_word: Vec::new(),
        }
    }
    fn merge(mut self, o: PairStats) -> PairStats {
        self.total += o.total;
        self.any += o.any;
        self.identical += o.identical;
        self.complement += o.complement;
        self.disjoint_nonzero += o.disjoint_nonzero;
        self.both_nonzero += o.both_nonzero;
        self.lift_undef += o.lift_undef;
        for i in 0..self.rate_hist.len() {
            self.rate_hist[i] += o.rate_hist[i];
        }
        for i in 0..self.lift_hist.len() {
            self.lift_hist[i] += o.lift_hist[i];
        }
        self.sum_rate += o.sum_rate;
        self.sum_expected += o.sum_expected;
        self.sum_dev2 += o.sum_dev2;
        if self.first_word.len() < o.first_word.len() {
            self.first_word.resize(o.first_word.len(), 0);
        }
        for i in 0..o.first_word.len() {
            self.first_word[i] += o.first_word[i];
        }
        self
    }
    fn json(&self) -> String {
        let t = self.total.max(1) as f64;
        format!(
            "{{\"pairs\":{},\"frac_cofire_any\":{},\"frac_identical\":{},\"frac_complement\":{},\
             \"frac_disjoint_given_both_nonzero\":{},\"frac_both_nonzero\":{},\"lift_undefined\":{},\
             \"mean_rate\":{},\"mean_expected_rate\":{},\"rms_dev_from_indep\":{},\
             \"rate_hist\":{},\"lift_hist_log2_half_steps_m8_to_8\":{},\"cofire_any_cum_by_64_inputs\":{}}}",
            self.total,
            fnum(self.any as f64 / t),
            fnum(self.identical as f64 / t),
            fnum(self.complement as f64 / t),
            fnum(self.disjoint_nonzero as f64 / self.both_nonzero.max(1) as f64),
            fnum(self.both_nonzero as f64 / t),
            self.lift_undef,
            fnum(self.sum_rate / t),
            fnum(self.sum_expected / t),
            fnum((self.sum_dev2 / t).sqrt()),
            json_arr_u(&self.rate_hist),
            json_arr_u(&self.lift_hist),
            json_arr_f(&{
                let mut acc = 0u64;
                let mut cum = Vec::new();
                for i in 0..self.first_word.len().saturating_sub(1) {
                    acc += self.first_word[i];
                    cum.push(acc as f64 / t);
                }
                cum
            })
        )
    }
}

fn sample_pairs(vecs: &[u64], count: usize, nw: usize, nbits: usize, pop: &[u32], pairs: u64, seed: u64) -> PairStats {
    let chunk: u64 = 1 << 20;
    let nchunks = (pairs + chunk - 1) / chunk;
    (0..nchunks)
        .into_par_iter()
        .map(|c| {
            let mut rng = fastrand::Rng::with_seed(seed ^ (c.wrapping_mul(0x9E37_79B9_7F4A_7C15)));
            let mut st = PairStats::new();
            st.first_word = vec![0u64; nw + 1];
            let todo = chunk.min(pairs - c * chunk);
            let nb = nbits as f64;
            for _ in 0..todo {
                let i = rng.usize(..count);
                let mut j = rng.usize(..count);
                while j == i {
                    j = rng.usize(..count);
                }
                let a = &vecs[i * nw..(i + 1) * nw];
                let b = &vecs[j * nw..(j + 1) * nw];
                let mut and = 0u32;
                let mut eq = true;
                let mut cm = true;
                let mut first = nw;
                for k in 0..nw {
                    let x = (a[k] & b[k]).count_ones();
                    if x > 0 && first == nw {
                        first = k;
                    }
                    and += x;
                    if a[k] != b[k] {
                        eq = false;
                    }
                    if a[k] != !b[k] {
                        cm = false;
                    }
                }
                st.total += 1;
                st.first_word[first] += 1;
                let pa = pop[i] as f64 / nb;
                let pb = pop[j] as f64 / nb;
                let rate = and as f64 / nb;
                let exp = pa * pb;
                st.sum_rate += rate;
                st.sum_expected += exp;
                st.sum_dev2 += (rate - exp) * (rate - exp);
                if and > 0 {
                    st.any += 1;
                    let bin = ((and as usize) * 64 + nbits - 1) / nbits; // ceil
                    st.rate_hist[bin.min(64)] += 1;
                } else {
                    st.rate_hist[0] += 1;
                }
                if eq {
                    st.identical += 1;
                }
                if cm {
                    st.complement += 1;
                }
                if pop[i] > 0 && pop[j] > 0 {
                    st.both_nonzero += 1;
                    if and == 0 {
                        st.disjoint_nonzero += 1;
                    }
                    let lift = rate / exp;
                    let l = if lift > 0.0 { lift.log2() } else { -1e9 };
                    let b = ((l + 8.0) * 2.0).round();
                    let b = if b < 0.0 { 0.0 } else if b > 32.0 { 32.0 } else { b };
                    st.lift_hist[b as usize] += 1;
                } else {
                    st.lift_undef += 1;
                }
            }
            st
        })
        .reduce(PairStats::new, PairStats::merge)
}

struct ClassStats {
    distinct: u64,
    in_multi: u64,
    top: Vec<u64>,
    size_hist: Vec<u64>, // log2 bins of class size
    entropy_bits: f64,
    census: Census,
}

/// Census of the exact-signature classes of size 2..=1000 (the always/never
/// pools are excluded): where in the gate list do the twins sit, and are they
/// the same gate shape?
#[derive(Clone)]
struct Census {
    classes: u64,
    members: u64,
    span_hist: Vec<u64>,
    gap_hist: Vec<u64>,
    same_target: u64,
    same_ctrls: u64,
    same_ctrl_wires: u64,
    same_gate: u64,
    adjacent_pairs: u64,
}

impl Default for Census {
    fn default() -> Self {
        Census { classes: 0, members: 0, span_hist: vec![0; 33], gap_hist: vec![0; 33], same_target: 0, same_ctrls: 0, same_ctrl_wires: 0, same_gate: 0, adjacent_pairs: 0 }
    }
}

impl Census {
    fn json(&self) -> String {
        let c = self.classes.max(1) as f64;
        format!(
            "{{\"classes_2_to_1000\":{},\"members\":{},\"span_log2_hist\":{},\"neighbour_gap_log2_hist\":{},\"frac_same_target\":{},\"frac_same_ctrls\":{},\"frac_same_ctrl_wires\":{},\"frac_same_gate\":{},\"adjacent_pairs\":{}}}",
            self.classes, self.members, json_arr_u(&self.span_hist), json_arr_u(&self.gap_hist),
            fnum(self.same_target as f64 / c), fnum(self.same_ctrls as f64 / c), fnum(self.same_ctrl_wires as f64 / c), fnum(self.same_gate as f64 / c), self.adjacent_pairs
        )
    }
}

fn signature_classes(vecs: &[u64], count: usize, nw: usize, gates: Option<&[XGate]>, gate_of: &dyn Fn(usize) -> Option<usize>) -> ClassStats {
    let mut hashes: Vec<(u128, u32)> = (0..count)
        .into_par_iter()
        .map(|i| {
            let s = &vecs[i * nw..(i + 1) * nw];
            let bytes = unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, nw * 8) };
            (xxhash_rust::xxh3::xxh3_128(bytes), i as u32)
        })
        .collect();
    hashes.par_sort_unstable();
    let mut sizes: Vec<u64> = Vec::new();
    let mut census = Census::default();
    let mut i = 0;
    while i < hashes.len() {
        let mut j = i + 1;
        while j < hashes.len() && hashes[j].0 == hashes[i].0 {
            j += 1;
        }
        let size = (j - i) as u64;
        sizes.push(size);
        if let Some(g) = gates {
            if size >= 2 && size <= 1000 {
                // members as gate indices (segments created by gates)
                let mut idx: Vec<usize> = hashes[i..j].iter().filter_map(|&(_, s)| gate_of(s as usize)).collect();
                if idx.len() >= 2 {
                    idx.sort_unstable();
                    census.classes += 1;
                    census.members += idx.len() as u64;
                    let span = (idx[idx.len() - 1] - idx[0]) as u32;
                    census.span_hist[log2_bin(span)] += 1;
                    // nearest-neighbour gap within the class, per member pair of neighbours
                    for w in idx.windows(2) {
                        census.gap_hist[log2_bin((w[1] - w[0]) as u32)] += 1;
                    }
                    let t0 = g[idx[0]].target;
                    let c0 = &g[idx[0]].ctrls;
                    let same_t = idx.iter().all(|&x| g[x].target == t0);
                    let same_c = idx.iter().all(|&x| g[x].ctrls == *c0);
                    let same_lits_wires = idx.iter().all(|&x| g[x].ctrls.len() == c0.len() && g[x].ctrls.iter().zip(c0.iter()).all(|(a, b)| a.0 == b.0));
                    if same_t { census.same_target += 1; }
                    if same_c { census.same_ctrls += 1; }
                    if same_lits_wires { census.same_ctrl_wires += 1; }
                    if same_t && same_c { census.same_gate += 1; }
                    if idx.len() == 2 && span == 1 { census.adjacent_pairs += 1; }
                }
            }
        }
        i = j;
    }
    let n = count as f64;
    let entropy = -sizes
        .iter()
        .map(|&s| {
            let p = s as f64 / n;
            p * p.log2()
        })
        .sum::<f64>();
    let mut size_hist = vec![0u64; 33];
    let mut in_multi = 0u64;
    for &s in &sizes {
        size_hist[log2_bin(s as u32)] += s;
        if s >= 2 {
            in_multi += s;
        }
    }
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    ClassStats {
        distinct: sizes.len() as u64,
        in_multi,
        top: sizes.iter().take(20).cloned().collect(),
        size_hist,
        entropy_bits: entropy,
        census,
    }
}

impl ClassStats {
    fn json(&self, count: u64) -> String {
        format!(
            "{{\"segments\":{},\"distinct_signatures\":{},\"segments_in_classes_ge2\":{},\
             \"frac_in_classes_ge2\":{},\"top_class_sizes\":{},\"segments_by_log2_class_size\":{},\
             \"signature_entropy_bits\":{},\"census\":{}}}",
            count,
            self.distinct,
            self.in_multi,
            fnum(self.in_multi as f64 / count.max(1) as f64),
            json_arr_u(&self.top),
            json_arr_u(&self.size_hist),
            fnum(self.entropy_bits),
            self.census.json()
        )
    }
}

fn pops(vecs: &[u64], count: usize, nw: usize) -> Vec<u32> {
    (0..count)
        .into_par_iter()
        .map(|i| vecs[i * nw..(i + 1) * nw].iter().map(|w| w.count_ones()).sum())
        .collect()
}

fn per_input_counts(vecs: &[u64], count: usize, nw: usize, nbits: usize) -> Vec<u64> {
    let block = 1 << 14;
    (0..count)
        .into_par_iter()
        .step_by(block)
        .map(|start| {
            let end = (start + block).min(count);
            let mut local = vec![0u32; nbits];
            for i in start..end {
                let s = &vecs[i * nw..(i + 1) * nw];
                for (k, &w) in s.iter().enumerate() {
                    let mut x = w;
                    while x != 0 {
                        let b = x.trailing_zeros() as usize;
                        local[k * 64 + b] += 1;
                        x &= x - 1;
                    }
                }
            }
            local
        })
        .reduce(
            || vec![0u32; nbits],
            |mut a, b| {
                for k in 0..nbits {
                    a[k] += b[k];
                }
                a
            },
        )
        .into_iter()
        .map(|x| x as u64)
        .collect()
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .ok();
    }
    let seed = args.seed.unwrap_or_else(|| fastrand::u64(..));
    let t0 = Instant::now();
    let (gates, w_cnt) = read_mpmct(&args.g)?;
    let m = gates.len();
    eprintln!("[segment_stats] {} gates, {} wires, read in {:.1}s", m, w_cnt, t0.elapsed().as_secs_f64());

    // ---------------- structural ----------------
    let mut touch: Vec<Vec<(u32, bool)>> = vec![Vec::new(); w_cnt];
    for (i, g) in gates.iter().enumerate() {
        touch[g.target as usize].push((i as u32, true));
        for &(c, _) in g.ctrls.iter() {
            touch[c as usize].push((i as u32, false));
        }
    }
    let writes: Vec<u32> = touch.iter().map(|t| t.iter().filter(|x| x.1).count() as u32).collect();
    let reads: Vec<u32> = touch.iter().map(|t| t.iter().filter(|x| !x.1).count() as u32).collect();
    let touches: Vec<u32> = touch.iter().map(|t| t.len() as u32).collect();

    // segment ids: wire w, j-th write (j=0 -> input segment)
    let mut seg_off = vec![0usize; w_cnt + 1];
    for w in 0..w_cnt {
        seg_off[w + 1] = seg_off[w] + writes[w] as usize + 1;
    }
    let s_total = seg_off[w_cnt];
    assert_eq!(s_total, m + w_cnt);
    let mut gate_seg = vec![0u32; m];
    let mut seg_wire = vec![0u16; s_total];
    let mut fanout = vec![0u32; s_total];
    let mut runs_per_wire: Vec<Vec<u32>> = vec![Vec::new(); w_cnt];
    let mut fanout_per_wire: Vec<Vec<u32>> = vec![Vec::new(); w_cnt];
    for w in 0..w_cnt {
        let mut cur = seg_off[w];
        let mut run = 0u32;
        for s in seg_off[w]..seg_off[w + 1] {
            seg_wire[s] = w as u16;
        }
        for &(i, is_w) in &touch[w] {
            if is_w {
                cur += 1;
                gate_seg[i as usize] = cur as u32;
                run += 1;
            } else {
                fanout[cur] += 1;
                if run > 0 {
                    runs_per_wire[w].push(run);
                    run = 0;
                }
            }
        }
        if run > 0 {
            runs_per_wire[w].push(run);
        }
        fanout_per_wire[w] = fanout[seg_off[w]..seg_off[w + 1]].to_vec();
    }
    eprintln!("[segment_stats] structural pass {:.1}s", t0.elapsed().as_secs_f64());

    // commutation boxes
    let box_size: Vec<u32> = (0..m)
        .into_par_iter()
        .map(|i| {
            let g = &gates[i];
            let mut lo = i;
            while lo > 0 && !conflicts(g, &gates[lo - 1]) {
                lo -= 1;
            }
            let mut hi = i;
            while hi + 1 < m && !conflicts(g, &gates[hi + 1]) {
                hi += 1;
            }
            (hi - lo + 1) as u32
        })
        .collect();
    eprintln!("[segment_stats] commutation boxes {:.1}s", t0.elapsed().as_secs_f64());

    let mut box_touch: Vec<Vec<u32>> = vec![Vec::new(); w_cnt];
    let mut box_target: Vec<Vec<u32>> = vec![Vec::new(); w_cnt];
    for (i, g) in gates.iter().enumerate() {
        box_target[g.target as usize].push(box_size[i]);
        box_touch[g.target as usize].push(box_size[i]);
        for &(c, _) in g.ctrls.iter() {
            box_touch[c as usize].push(box_size[i]);
        }
    }

    // per-wire summaries
    let mut w_fan_med = vec![0f64; w_cnt];
    let mut w_fan_mean = vec![0f64; w_cnt];
    let mut w_box_med_touch = vec![0f64; w_cnt];
    let mut w_box_med_target = vec![0f64; w_cnt];
    let mut w_run_mean = vec![0f64; w_cnt];
    let mut w_run_med = vec![0f64; w_cnt];
    let mut w_nruns = vec![0u64; w_cnt];
    for w in 0..w_cnt {
        w_fan_med[w] = median_u32(&mut fanout_per_wire[w].clone());
        w_fan_mean[w] = mean_u32(&fanout_per_wire[w]);
        w_box_med_touch[w] = median_u32(&mut box_touch[w].clone());
        w_box_med_target[w] = median_u32(&mut box_target[w].clone());
        w_run_mean[w] = mean_u32(&runs_per_wire[w]);
        w_run_med[w] = median_u32(&mut runs_per_wire[w].clone());
        w_nruns[w] = runs_per_wire[w].len() as u64;
    }
    // pooled
    let mut all_fan: Vec<u32> = fanout.clone();
    let fan_pooled_med = median_u32(&mut all_fan);
    let fan_pooled_mean = mean_u32(&fanout);
    let mut fan_hist = vec![0u64; 33];
    for &f in &fanout {
        fan_hist[log2_bin(f)] += 1;
    }
    let mut all_box = box_size.clone();
    let box_pooled_med = median_u32(&mut all_box);
    let box_pooled_mean = mean_u32(&box_size);
    let mut box_hist = vec![0u64; 33];
    for &b in &box_size {
        box_hist[log2_bin(b)] += 1;
    }
    let mut all_runs: Vec<u32> = runs_per_wire.iter().flatten().cloned().collect();
    let run_pooled_mean = mean_u32(&all_runs);
    let run_pooled_med = median_u32(&mut all_runs);
    let mut run_hist = vec![0u64; 33];
    for &r in &all_runs {
        run_hist[log2_bin(r)] += 1;
    }
    let touches_f: Vec<f64> = touches.iter().map(|&x| x as f64).collect();
    let writes_f: Vec<f64> = writes.iter().map(|&x| x as f64).collect();
    let reads_f: Vec<f64> = reads.iter().map(|&x| x as f64).collect();
    let (t_mean, t_var) = mean_var(&touches_f);
    let (wr_mean, wr_var) = mean_var(&writes_f);
    let (rd_mean, rd_var) = mean_var(&reads_f);
    let mut k_hist = vec![0u64; 33];
    let mut comp_cnt = 0u64;
    for g in &gates {
        k_hist[g.ctrls.len().min(32)] += 1;
        if g.comp {
            comp_cnt += 1;
        }
    }

    // ---------------- firing ----------------
    let nw = (args.samples / 64).max(1);
    let nbits = nw * 64;
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut state = vec![0u64; w_cnt * nw];
    for w in 0..args.n.min(w_cnt) {
        for k in 0..nw {
            state[w * nw + k] = rng.u64(..);
        }
    }
    let mut fire = vec![0u64; m * nw];
    let mut value = vec![0u64; s_total * nw];
    for w in 0..w_cnt {
        value[seg_off[w] * nw..(seg_off[w] + 1) * nw].copy_from_slice(&state[w * nw..(w + 1) * nw]);
    }
    let mut acc = vec![0u64; nw];
    for (i, g) in gates.iter().enumerate() {
        acc.fill(!0u64);
        for &(c, p) in g.ctrls.iter() {
            let xm = if p { 0u64 } else { !0u64 };
            let s = &state[c as usize * nw..(c as usize + 1) * nw];
            for k in 0..nw {
                acc[k] &= s[k] ^ xm;
            }
        }
        if g.comp {
            for k in 0..nw {
                acc[k] = !acc[k];
            }
        }
        let t = g.target as usize;
        for k in 0..nw {
            state[t * nw + k] ^= acc[k];
        }
        fire[i * nw..(i + 1) * nw].copy_from_slice(&acc);
        let s = gate_seg[i] as usize;
        value[s * nw..(s + 1) * nw].copy_from_slice(&state[t * nw..(t + 1) * nw]);
    }
    eprintln!("[segment_stats] evaluated {} inputs {:.1}s", nbits, t0.elapsed().as_secs_f64());

    let fire_pop = pops(&fire, m, nw);
    let value_pop = pops(&value, s_total, nw);
    let never = fire_pop.iter().filter(|&&p| p == 0).count() as u64;
    let always = fire_pop.iter().filter(|&&p| p as usize == nbits).count() as u64;
    let mean_fire_rate = fire_pop.iter().map(|&p| p as f64).sum::<f64>() / (m as f64 * nbits as f64);
    let mut fire_rate_hist = vec![0u64; 65];
    for &p in &fire_pop {
        fire_rate_hist[((p as usize) * 64 + nbits - 1) / nbits] += 1;
    }
    let mean_value_rate = value_pop.iter().map(|&p| p as f64).sum::<f64>() / (s_total as f64 * nbits as f64);

    // exact per-input co-firing fraction
    let cnt_fire = per_input_counts(&fire, m, nw, nbits);
    let cnt_value = per_input_counts(&value, s_total, nw, nbits);
    let pairs_of = |c: u64| (c as u128) * (c.saturating_sub(1) as u128) / 2;
    let tot_pairs_fire = pairs_of(m as u64) as f64;
    let tot_pairs_value = pairs_of(s_total as u64) as f64;
    let per_input_fire = cnt_fire.iter().map(|&c| pairs_of(c) as f64 / tot_pairs_fire).sum::<f64>() / nbits as f64;
    let per_input_value = cnt_value.iter().map(|&c| pairs_of(c) as f64 / tot_pairs_value).sum::<f64>() / nbits as f64;
    let (cf_mean, cf_var) = mean_var(&cnt_fire.iter().map(|&c| c as f64).collect::<Vec<_>>());
    eprintln!("[segment_stats] per-input counts {:.1}s", t0.elapsed().as_secs_f64());

    let ps_fire = sample_pairs(&fire, m, nw, nbits, &fire_pop, args.pairs, seed ^ 0xF1);
    let ps_value = sample_pairs(&value, s_total, nw, nbits, &value_pop, args.pairs, seed ^ 0xA5);
    eprintln!("[segment_stats] pair sampling {:.1}s", t0.elapsed().as_secs_f64());
    let cls_fire = signature_classes(&fire, m, nw, Some(&gates), &|i| Some(i));
    let seg_gate: Vec<u32> = {
        let mut v = vec![u32::MAX; s_total];
        for (i, &sg) in gate_seg.iter().enumerate() {
            v[sg as usize] = i as u32;
        }
        v
    };
    let cls_value = signature_classes(&value, s_total, nw, Some(&gates), &|s| if seg_gate[s] == u32::MAX { None } else { Some(seg_gate[s] as usize) });
    eprintln!("[segment_stats] signature classes {:.1}s", t0.elapsed().as_secs_f64());

    // ---------------- dump sample ----------------
    let dump_n = args.dump.min(m);
    let mut idx: Vec<u32> = (0..m as u32).collect();
    let mut drng = fastrand::Rng::with_seed(seed ^ 0xD0);
    for i in 0..dump_n {
        let j = i + drng.usize(..m - i);
        idx.swap(i, j);
    }
    let mut chosen: Vec<u32> = idx[..dump_n].to_vec();
    chosen.sort_unstable();
    {
        let mut f = std::io::BufWriter::new(std::fs::File::create(format!("{}.sample.fire.bin", args.out))?);
        let mut v = std::io::BufWriter::new(std::fs::File::create(format!("{}.sample.value.bin", args.out))?);
        for &i in &chosen {
            let i = i as usize;
            for k in 0..nw {
                f.write_all(&fire[i * nw + k].to_le_bytes())?;
            }
            let s = gate_seg[i] as usize;
            for k in 0..nw {
                v.write_all(&value[s * nw + k].to_le_bytes())?;
            }
        }
        let mut meta = std::fs::File::create(format!("{}.sample.meta.json", args.out))?;
        let gi: Vec<u64> = chosen.iter().map(|&i| i as u64).collect();
        let gw: Vec<u64> = chosen.iter().map(|&i| gates[i as usize].target as u64).collect();
        let gk: Vec<u64> = chosen.iter().map(|&i| gates[i as usize].ctrls.len() as u64).collect();
        let gp: Vec<u64> = chosen.iter().map(|&i| fire_pop[i as usize] as u64).collect();
        let gb: Vec<u64> = chosen.iter().map(|&i| box_size[i as usize] as u64).collect();
        let gf: Vec<u64> = chosen.iter().map(|&i| fanout[gate_seg[i as usize] as usize] as u64).collect();
        write!(
            meta,
            "{{\"nbits\":{},\"words\":{},\"count\":{},\"gate_index\":{},\"target_wire\":{},\"k\":{},\"fire_pop\":{},\"box\":{},\"fanout\":{}}}\n",
            nbits, nw, dump_n, json_arr_u(&gi), json_arr_u(&gw), json_arr_u(&gk), json_arr_u(&gp), json_arr_u(&gb), json_arr_u(&gf)
        )?;
    }

    // ---------------- per-wire csv ----------------
    {
        let mut c = std::io::BufWriter::new(std::fs::File::create(format!("{}.wires.csv", args.out))?);
        writeln!(c, "wire,touches,writes,reads,segments,fanout_median,fanout_mean,box_median_touch,box_median_target,runs,run_mean,run_median")?;
        for w in 0..w_cnt {
            writeln!(
                c,
                "{},{},{},{},{},{},{},{},{},{},{},{}",
                w,
                touches[w],
                writes[w],
                reads[w],
                writes[w] + 1,
                fnum(w_fan_med[w]),
                fnum(w_fan_mean[w]),
                fnum(w_box_med_touch[w]),
                fnum(w_box_med_target[w]),
                w_nruns[w],
                fnum(w_run_mean[w]),
                fnum(w_run_med[w])
            )?;
        }
    }

    // ---------------- json ----------------
    let med_of = |v: &[f64]| {
        let mut x: Vec<f64> = v.iter().cloned().filter(|x| !x.is_nan()).collect();
        x.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if x.is_empty() {
            f64::NAN
        } else if x.len() % 2 == 1 {
            x[x.len() / 2]
        } else {
            (x[x.len() / 2 - 1] + x[x.len() / 2]) / 2.0
        }
    };
    let mut j = std::fs::File::create(format!("{}.json", args.out))?;
    write!(
        j,
        "{{\"circuit\":\"{}\",\"wires\":{},\"gates\":{},\"segments\":{},\"n_input\":{},\"samples\":{},\"pairs_sampled\":{},\"seed\":{},\
         \"comp_gates\":{},\"k_hist\":{},\
         \"touches\":{{\"mean\":{},\"var\":{},\"std\":{},\"min\":{},\"max\":{},\"per_wire\":{}}},\
         \"writes\":{{\"mean\":{},\"var\":{},\"std\":{},\"min\":{},\"max\":{},\"per_wire\":{}}},\
         \"reads\":{{\"mean\":{},\"var\":{},\"std\":{},\"per_wire\":{}}},\
         \"fanout\":{{\"pooled_median\":{},\"pooled_mean\":{},\"median_of_wire_medians\":{},\"mean_of_wire_means\":{},\"log2_hist\":{},\"per_wire_median\":{},\"per_wire_mean\":{}}},\
         \"box\":{{\"pooled_median\":{},\"pooled_mean\":{},\"median_of_wire_medians_touch\":{},\"median_of_wire_medians_target\":{},\"log2_hist\":{},\"per_wire_median_touch\":{},\"per_wire_median_target\":{}}},\
         \"runs\":{{\"pooled_mean\":{},\"pooled_median\":{},\"median_of_wire_means\":{},\"median_of_wire_medians\":{},\"total_runs\":{},\"log2_hist\":{},\"per_wire_mean\":{},\"per_wire_median\":{}}},\
         \"firing\":{{\"mean_fire_rate\":{},\"never_fire\":{},\"always_fire\":{},\"fire_rate_hist64\":{},\"per_input_pair_frac\":{},\"per_input_firing_count_mean\":{},\"per_input_firing_count_var\":{},\"mean_value_rate\":{},\"per_input_value_pair_frac\":{}}},\
         \"pairs_fire\":{},\"pairs_value\":{},\"classes_fire\":{},\"classes_value\":{}}}\n",
        args.g, w_cnt, m, s_total, args.n, nbits, args.pairs, seed,
        comp_cnt, json_arr_u(&k_hist),
        fnum(t_mean), fnum(t_var), fnum(t_var.sqrt()), touches.iter().min().unwrap(), touches.iter().max().unwrap(), json_arr_u(&touches.iter().map(|&x| x as u64).collect::<Vec<_>>()),
        fnum(wr_mean), fnum(wr_var), fnum(wr_var.sqrt()), writes.iter().min().unwrap(), writes.iter().max().unwrap(), json_arr_u(&writes.iter().map(|&x| x as u64).collect::<Vec<_>>()),
        fnum(rd_mean), fnum(rd_var), fnum(rd_var.sqrt()), json_arr_u(&reads.iter().map(|&x| x as u64).collect::<Vec<_>>()),
        fnum(fan_pooled_med), fnum(fan_pooled_mean), fnum(med_of(&w_fan_med)), fnum(mean_var(&w_fan_mean).0), json_arr_u(&fan_hist), json_arr_f(&w_fan_med), json_arr_f(&w_fan_mean),
        fnum(box_pooled_med), fnum(box_pooled_mean), fnum(med_of(&w_box_med_touch)), fnum(med_of(&w_box_med_target)), json_arr_u(&box_hist), json_arr_f(&w_box_med_touch), json_arr_f(&w_box_med_target),
        fnum(run_pooled_mean), fnum(run_pooled_med), fnum(med_of(&w_run_mean)), fnum(med_of(&w_run_med)), all_runs.len(), json_arr_u(&run_hist), json_arr_f(&w_run_mean), json_arr_f(&w_run_med),
        fnum(mean_fire_rate), never, always, json_arr_u(&fire_rate_hist), fnum(per_input_fire), fnum(cf_mean), fnum(cf_var), fnum(mean_value_rate), fnum(per_input_value),
        ps_fire.json(), ps_value.json(), cls_fire.json(m as u64), cls_value.json(s_total as u64)
    )?;
    eprintln!("[segment_stats] done {:.1}s -> {}.json", t0.elapsed().as_secs_f64(), args.out);
    println!(
        "gates={} wires={} touches/wire mean={:.1} var={:.1} | fanout pooled median={} mean={:.2} | box pooled median={} | runs mean={:.3} median={} | fire rate={:.4} per-input pair frac={:.5} | sampled pairs cofire_any={:.5} identical={:.6} | distinct fire signatures={} ({:.3} in classes>=2)",
        m, w_cnt, t_mean, t_var, fan_pooled_med, fan_pooled_mean, box_pooled_med, run_pooled_mean, run_pooled_med,
        mean_fire_rate, per_input_fire, ps_fire.any as f64 / ps_fire.total.max(1) as f64, ps_fire.identical as f64 / ps_fire.total.max(1) as f64,
        cls_fire.distinct, cls_fire.in_multi as f64 / m as f64
    );
    Ok(())
}
