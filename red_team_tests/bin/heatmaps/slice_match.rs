// Slice-level correspondence between a circuit C and its locally-mixed image M
// (same permutation, |M| > |C|).
//
// The prefix heatmap H(i,j) = E_x HD(C_i(x), M_j(x)) says WHERE along M the
// mixed trajectory best tracks C's state after i gates. This binary uses that
// alignment to ask a stronger question: does the *segment* of M between two
// aligned landmarks still compute the same function as the corresponding
// segment of C?
//
//   i'    = argmin_j E_x HD(C_i(x),     M_j(x))
//   i'_m  = argmin_j E_x HD(C_{i+m}(x), M_j(x))
//   F(i,m)= E_x HD( C[i..i+m](x), M[i'..i'_m](x) )      <- both applied to the
//                                                          SAME fresh uniform x
//
// F is reported next to three references, all measured on the same x:
//   * F_prop  — same construction with the ridge replaced by the parameter-free
//               proportional alignment j = round(i*|M|/|C|). Separates "the
//               segments correspond" from "the argmin happened to land there".
//   * F_ctrl  — C's segment against an M segment of the SAME length taken half
//               a circuit away. The wrong-place ceiling.
//   * indep   — dC + dM - 2*dC*dM/nw, the HD two independent maps with the
//               measured displacements dC, dM would produce. For small m both
//               segments are near-identity, so F is small for trivial reasons;
//               `indep` is what "no relation beyond length" predicts.
//
// The ridge argmin is taken over all of M: a coarse scan at --coarse-step over
// every column, then a refinement at --fine-step within --fine-halfwidth of the
// coarse winner. --fine-step is also the snapshot stride of M, so the whole
// refinement reuses one snapshot array.
//
// Example:
//   slice_match --c gss.mpmct1 --m phaseA.mpmct1 --out run_c_A
use clap::Parser;
use local_mixing::circuit::xgate::{XGate, max_wire};
use local_mixing::engine::format::read_mpmct;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Parser, Debug)]
#[command(name = "slice_match")]
struct Args {
    /// Source circuit C (the input to the mixing stage)
    #[arg(long)]
    c: String,
    /// Mixed circuit M (the output of the mixing stage; same permutation as C)
    #[arg(long)]
    m: String,
    #[arg(long, default_value = "mpmct1")]
    c_format: String,
    #[arg(long, default_value = "mpmct1")]
    m_format: String,
    /// Stride between the sampled source offsets i
    #[arg(long, default_value_t = 10000)]
    i_step: usize,
    /// Smallest segment length m
    #[arg(long, default_value_t = 100)]
    m_min: usize,
    /// Largest segment length m
    #[arg(long, default_value_t = 10000)]
    m_max: usize,
    /// Stride between segment lengths m (also the prefix granularity of the ridge)
    #[arg(long, default_value_t = 100)]
    m_step: usize,
    /// Column stride of the coarse ridge scan (covers all of M)
    #[arg(long, default_value_t = 1000)]
    coarse_step: usize,
    /// Column stride of the refinement scan; also M's snapshot stride
    #[arg(long, default_value_t = 10)]
    fine_step: usize,
    /// Half-width in gates of M of the refinement window
    #[arg(long, default_value_t = 2000)]
    fine_halfwidth: usize,
    /// 64-lane groups carried through the ridge scan (samples = 64 * this)
    #[arg(long, default_value_t = 16)]
    ridge_batches: usize,
    /// 64-lane groups per slice repetition (samples = 64 * this * --slice-reps)
    #[arg(long, default_value_t = 16)]
    slice_batches: usize,
    /// Independent input draws for the slice measurement
    #[arg(long, default_value_t = 4)]
    slice_reps: usize,
    /// Supplementary sweep: also measure F with M's segment slid by delta gates,
    /// for delta in -D..D step --delta-step. A real segment correspondence shows
    /// up as a minimum at delta=0; a smooth curve means the alignment carries no
    /// segment-level information. 0 disables the sweep.
    #[arg(long, default_value_t = 0)]
    delta_max: usize,
    #[arg(long, default_value_t = 2500)]
    delta_step: usize,
    /// Segment lengths m included in the sweep (comma-separated)
    #[arg(long, default_value = "100,300,1000,3000,10000")]
    delta_ms: String,
    /// Which extremum of the prefix map defines the alignment:
    ///   min — argmin_j E_x HD(C_i, M_j)         (the literal ridge)
    ///   dev — argmax_j |E_x HD(C_i, M_j) - nw/2| (prominence; also catches the
    ///         complement re-encodings that split/cross introduce, where the
    ///         aligned cell is a BUMP above nw/2 rather than a dip)
    #[arg(long, default_value = "min")]
    align: String,
    #[arg(long, default_value_t = 20260824)]
    seed: u64,
    /// Output prefix: writes <out>_ridge.csv, <out>_slices.csv, <out>.meta.json
    #[arg(long)]
    out: String,
}

// Bit-sliced application over `nb` interleaved 64-lane groups. `state` is
// wire-major: state[w*nb + b] holds group b's 64 lanes for wire w. Per group
// this is bit-for-bit XGate::apply_lanes; the gate load and the control walk
// are paid once for all nb groups.
#[inline]
fn apply_ml(g: &XGate, state: &mut [u64], nb: usize, acc: &mut [u64]) {
    for a in acc[..nb].iter_mut() {
        *a = !0u64;
    }
    for &(w, p) in &g.ctrls {
        let mask = (p as u64).wrapping_sub(1); // 0 for a positive literal, !0 for NOT
        let base = w as usize * nb;
        for b in 0..nb {
            acc[b] &= state[base + b] ^ mask;
        }
    }
    let c = 0u64.wrapping_sub(g.comp as u64);
    let base = g.target as usize * nb;
    for b in 0..nb {
        state[base + b] ^= acc[b] ^ c;
    }
}

#[inline]
fn hd_sum(a: &[u64], b: &[u64]) -> u64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x ^ y).count_ones() as u64).sum()
}

fn sample_state(nw: usize, nb: usize, rng: &mut StdRng) -> Vec<u64> {
    (0..nw * nb).map(|_| rng.random::<u64>()).collect()
}

// Prefix lengths start, start+step, ..., end (end always included).
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

// Run `gates` from prefix length 0, writing the lane state into `out` at every
// position in `idx` (sorted, ascending). `out` is a flat idx.len() * nw * nb.
fn snapshots_flat(gates: &[XGate], nw: usize, nb: usize, init: &[u64], idx: &[usize]) -> Vec<u64> {
    let stride = nw * nb;
    let mut out = vec![0u64; idx.len() * stride];
    let mut state = init.to_vec();
    let mut acc = vec![0u64; nb];
    let mut k = 0;
    for pos in 0..=gates.len() {
        while k < idx.len() && idx[k] == pos {
            out[k * stride..(k + 1) * stride].copy_from_slice(&state);
            k += 1;
        }
        if pos < gates.len() {
            apply_ml(&gates[pos], &mut state, nb, &mut acc);
        }
    }
    assert_eq!(k, idx.len(), "snapshot index past |gates|");
    out
}

// Apply gates[start..end_k] to `init` for each requested end, returning one
// state per request. `ends[k] = None` (or an end before `start`) yields None.
fn slice_states(
    gates: &[XGate],
    nb: usize,
    init: &[u64],
    start: usize,
    ends: &[Option<usize>],
) -> Vec<Option<Vec<u64>>> {
    let mut order: Vec<(usize, usize)> = ends
        .iter()
        .enumerate()
        .filter_map(|(k, e)| match *e {
            Some(e) if e >= start && e <= gates.len() => Some((e, k)),
            _ => None,
        })
        .collect();
    order.sort_unstable();
    let mut out: Vec<Option<Vec<u64>>> = vec![None; ends.len()];
    let mut state = init.to_vec();
    let mut acc = vec![0u64; nb];
    let mut pos = start;
    for (e, k) in order {
        while pos < e {
            apply_ml(&gates[pos], &mut state, nb, &mut acc);
            pos += 1;
        }
        out[k] = Some(state.clone());
    }
    out
}

fn read_circuit(path: &str, format: &str) -> (Vec<XGate>, usize) {
    match format {
        "mpmct1" => {
            let (g, nw) = read_mpmct(path).expect("read mpmct1");
            (g, nw)
        }
        other => panic!("unknown circuit format {other}"),
    }
}

fn main() {
    let args = Args::parse();
    let t0 = std::time::Instant::now();
    let (c, c_nw) = read_circuit(&args.c, &args.c_format);
    let (m, m_nw) = read_circuit(&args.m, &args.m_format);
    let nw = c_nw
        .max(m_nw)
        .max(max_wire(&c) as usize + 1)
        .max(max_wire(&m) as usize + 1);
    assert!(args.coarse_step % args.fine_step == 0, "--coarse-step must be a multiple of --fine-step");

    // Source offsets i, and the segment lengths m.
    let ms: Vec<usize> = (args.m_min..=args.m_max).step_by(args.m_step).collect();
    let i_list: Vec<usize> = (0..=c.len().saturating_sub(args.m_max))
        .step_by(args.i_step)
        .collect();
    // Every prefix of C whose ridge position we need: i and i+m for all i, m.
    // With the default i_step >> m_max these overlap into one dense grid, but
    // sparser --i-step settings only pay for the prefixes they touch.
    let mut p_idx: Vec<usize> = i_list
        .iter()
        .flat_map(|&i| std::iter::once(i).chain(ms.iter().map(move |&mm| i + mm)))
        .collect();
    p_idx.sort_unstable();
    p_idx.dedup();
    assert!(*p_idx.last().unwrap() <= c.len(), "max prefix exceeds |C|={}", c.len());
    let pos_of = |p: usize| -> usize { p_idx.binary_search(&p).expect("prefix not on the ridge grid") };

    // M snapshot grid (refinement resolution) and its coarse subset.
    let q_idx = indices(0, m.len(), args.fine_step);
    let ratio = args.coarse_step / args.fine_step;
    let coarse_k: Vec<usize> = (0..q_idx.len()).step_by(ratio).chain([q_idx.len() - 1]).collect();

    let nbr = args.ridge_batches.max(1);
    let stride = nw * nbr;
    println!(
        "[slice_match] |C|={} |M|={} nw={} | i: {} offsets step {} | m: {}..{} step {} | ridge rows={} cols(coarse)={} cols(fine grid)={} samples={}",
        c.len(), m.len(), nw, i_list.len(), args.i_step, args.m_min, args.m_max, args.m_step,
        p_idx.len(), coarse_k.len(), q_idx.len(), 64 * nbr
    );
    println!(
        "[slice_match] memory: C snapshots {:.2} GB, M snapshots {:.2} GB",
        (p_idx.len() * stride * 8) as f64 / 1e9,
        (q_idx.len() * stride * 8) as f64 / 1e9
    );

    // ---- ridge: one shared input x, snapshots of both circuits ----
    let mut rng = StdRng::seed_from_u64(args.seed);
    let x = sample_state(nw, nbr, &mut rng);
    let cst = snapshots_flat(&c, nw, nbr, &x, &p_idx);
    println!("[slice_match] C snapshots done ({:.1}s)", t0.elapsed().as_secs_f64());
    let mst = snapshots_flat(&m, nw, nbr, &x, &q_idx);
    println!("[slice_match] M snapshots done ({:.1}s)", t0.elapsed().as_secs_f64());
    // Compact copy of the coarse columns: the coarse scan touches every row, so
    // gathering them once turns a strided walk over `mst` into a linear one.
    let mut mst_coarse = vec![0u64; coarse_k.len() * stride];
    for (a, &k) in coarse_k.iter().enumerate() {
        mst_coarse[a * stride..(a + 1) * stride].copy_from_slice(&mst[k * stride..(k + 1) * stride]);
    }

    let nsamp = (64 * nbr) as f64;
    let done = AtomicUsize::new(0);
    // Summed-popcount value of the nw/2 random baseline, for the `dev` rule.
    let mid = (nsamp * nw as f64 / 2.0) as i64;
    let use_dev = args.align == "dev";
    let half = args.fine_halfwidth / args.fine_step;
    // Per prefix: (selected j, its mean HD, off-ridge baseline, argmin j, its
    // mean HD, argmax-deviation j, its mean HD).
    let ridge: Vec<(usize, f64, f64, usize, f64, usize, f64)> = (0..p_idx.len())
        .into_par_iter()
        .map(|r| {
            let cv = &cst[r * stride..(r + 1) * stride];
            let (mut ca_min, mut c_min) = (0usize, u64::MAX);
            let (mut ca_dev, mut c_dev) = (0usize, -1i64);
            let mut total = 0u64;
            for a in 0..coarse_k.len() {
                let d = hd_sum(cv, &mst_coarse[a * stride..(a + 1) * stride]);
                total += d;
                if d < c_min {
                    c_min = d;
                    ca_min = a;
                }
                let dev = (d as i64 - mid).abs();
                if dev > c_dev {
                    c_dev = dev;
                    ca_dev = a;
                }
            }
            let baseline = total as f64 / coarse_k.len() as f64 / nsamp;
            // Refine at fine_step around each coarse winner. Split and cross
            // re-encode roughly half the rows through a complement, so their
            // ridge shows up as HD > nw/2, not as a minimum; `dev` follows the
            // |HD - nw/2| prominence instead and catches both signs.
            let refine = |centre: usize, by_dev: bool| -> (usize, u64) {
                let lo = centre.saturating_sub(half);
                let hi = (centre + half).min(q_idx.len() - 1);
                let (mut bk, mut bv, mut bd) = (centre, u64::MAX, -1i64);
                for k in lo..=hi {
                    let d = hd_sum(cv, &mst[k * stride..(k + 1) * stride]);
                    let better = if by_dev { (d as i64 - mid).abs() > bd } else { d < bv };
                    if better {
                        bk = k;
                        bv = d;
                        bd = (d as i64 - mid).abs();
                    }
                }
                (bk, bv)
            };
            let (k_min, v_min) = refine(coarse_k[ca_min], false);
            let (k_dev, v_dev) = refine(coarse_k[ca_dev], true);
            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
            if n % 500 == 0 {
                println!("[slice_match]   ridge {}/{}", n, p_idx.len());
            }
            let (sel_j, sel_v) = if use_dev { (k_dev, v_dev) } else { (k_min, v_min) };
            (
                q_idx[sel_j], sel_v as f64 / nsamp, baseline,
                q_idx[k_min], v_min as f64 / nsamp,
                q_idx[k_dev], v_dev as f64 / nsamp,
            )
        })
        .collect();
    println!("[slice_match] ridge done ({:.1}s)", t0.elapsed().as_secs_f64());
    drop(mst);
    drop(mst_coarse);
    drop(cst);

    {
        let mut f = std::fs::File::create(format!("{}_ridge.csv", args.out)).expect("ridge csv");
        writeln!(f, "prefix,argmin_j,min_hd,baseline_hd,dip,dev_j,dev_hd,dev_prominence,sel_j,sel_hd").unwrap();
        for (r, &p) in p_idx.iter().enumerate() {
            let (sj, sv, b, jm, vm, jd, vd) = ridge[r];
            writeln!(
                f, "{},{},{:.4},{:.4},{:.4},{},{:.4},{:.4},{},{:.4}",
                p, jm, vm, b, b - vm, jd, vd, (vd - nw as f64 / 2.0).abs(), sj, sv
            ).unwrap();
        }
    }

    // ---- slices ----
    let nbs = args.slice_batches.max(1);
    let reps = args.slice_reps.max(1);
    let scale = m.len() as f64 / c.len() as f64;
    let nsamp_s = (64 * nbs * reps) as f64;
    let nrec = i_list.len() * ms.len();
    // Accumulators, indexed [i_index * |ms| + m_index].
    let mut acc_f = vec![0u64; nrec];
    let mut acc_p = vec![0u64; nrec];
    let mut acc_x = vec![0u64; nrec];
    let mut acc_dx = vec![0u64; nrec];
    let mut acc_dc = vec![0u64; nrec];
    let mut acc_dm = vec![0u64; nrec];
    let mut acc_dp = vec![0u64; nrec];
    let mut ok_f = vec![0u32; nrec];
    let mut ok_x = vec![0u32; nrec];

    for rep in 0..reps {
        let mut srng = StdRng::seed_from_u64(args.seed ^ (0x5eed_0000 + rep as u64));
        let inputs: Vec<Vec<u64>> = i_list.iter().map(|_| sample_state(nw, nbs, &mut srng)).collect();
        let per_i: Vec<Vec<[Option<u64>; 7]>> = i_list
            .par_iter()
            .enumerate()
            .map(|(ii, &i)| {
                let x = &inputs[ii];
                let ip = ridge[pos_of(i)].0;
                let p0 = ((i as f64) * scale).round() as usize;
                // C segment ends, and the three families of M segment ends.
                let c_ends: Vec<Option<usize>> = ms.iter().map(|&mm| Some(i + mm)).collect();
                let m_ends: Vec<Option<usize>> =
                    ms.iter().map(|&mm| Some(ridge[pos_of(i + mm)].0)).collect();
                let p_ends: Vec<Option<usize>> = ms
                    .iter()
                    .map(|&mm| Some(((((i + mm) as f64) * scale).round() as usize).min(m.len())))
                    .collect();
                // Wrong-place control: same segment lengths, half a circuit away.
                let shift = m.len() / 2;
                let ctrl_start = if ip + shift < m.len() { ip + shift } else { ip.saturating_sub(shift) };
                let x_ends: Vec<Option<usize>> = m_ends
                    .iter()
                    .map(|e| e.and_then(|e| e.checked_sub(ip)).map(|l| ctrl_start + l))
                    .collect();

                let cs = slice_states(&c, nbs, x, i, &c_ends);
                let msl = slice_states(&m, nbs, x, ip, &m_ends);
                let psl = slice_states(&m, nbs, x, p0.min(m.len()), &p_ends);
                let xsl = slice_states(&m, nbs, x, ctrl_start, &x_ends);

                ms.iter()
                    .enumerate()
                    .map(|(k, _)| {
                        let cv = cs[k].as_ref().expect("C segment always valid");
                        let dc = hd_sum(x, cv);
                        let f = msl[k].as_ref().map(|v| hd_sum(cv, v));
                        let dm = msl[k].as_ref().map(|v| hd_sum(x, v));
                        let p = psl[k].as_ref().map(|v| hd_sum(cv, v));
                        let dp = psl[k].as_ref().map(|v| hd_sum(x, v));
                        let xc = xsl[k].as_ref().map(|v| hd_sum(cv, v));
                        let dx = xsl[k].as_ref().map(|v| hd_sum(x, v));
                        [Some(dc), f, dm, p, dp, xc, dx]
                    })
                    .collect()
            })
            .collect();

        for (ii, rows) in per_i.iter().enumerate() {
            for (k, cell) in rows.iter().enumerate() {
                let idx = ii * ms.len() + k;
                acc_dc[idx] += cell[0].unwrap();
                if let (Some(f), Some(dm)) = (cell[1], cell[2]) {
                    acc_f[idx] += f;
                    acc_dm[idx] += dm;
                    ok_f[idx] += 1;
                }
                if let (Some(p), Some(dp)) = (cell[3], cell[4]) {
                    acc_p[idx] += p;
                    acc_dp[idx] += dp;
                }
                if let (Some(xc), Some(dx)) = (cell[5], cell[6]) {
                    acc_x[idx] += xc;
                    acc_dx[idx] += dx;
                    ok_x[idx] += 1;
                }
            }
        }
        println!("[slice_match] slice rep {}/{} done ({:.1}s)", rep + 1, reps, t0.elapsed().as_secs_f64());
    }

    let mut f = std::fs::File::create(format!("{}_slices.csv", args.out)).expect("slices csv");
    writeln!(
        f,
        "i,m,iprime,iprime_m,Lm,F,F_prop,F_ctrl,dC,dM,dP,dX,indep,indep_prop,indep_ctrl,ridge_dip_i,ridge_dip_im"
    )
    .unwrap();
    let per_rep = (64 * nbs) as f64;
    for (ii, &i) in i_list.iter().enumerate() {
        let (ip, sel_i, base_i, ..) = ridge[pos_of(i)];
        let dip_i = base_i - sel_i;
        for (k, &mm) in ms.iter().enumerate() {
            let idx = ii * ms.len() + k;
            let (ipm, sel_im, base_im, ..) = ridge[pos_of(i + mm)];
            let dc = acc_dc[idx] as f64 / nsamp_s;
            let n_f = (ok_f[idx] as f64 * per_rep).max(1.0);
            let n_x = (ok_x[idx] as f64 * per_rep).max(1.0);
            let dm = acc_dm[idx] as f64 / n_f;
            let dp = acc_dp[idx] as f64 / nsamp_s;
            // The no-relation prediction has to be built from the reach of the
            // segment it is compared against: the ridge segment for F, the
            // proportional one for F_prop (they differ in length).
            let indep = dc + dm - 2.0 * dc * dm / nw as f64;
            let indep_p = dc + dp - 2.0 * dc * dp / nw as f64;
            let dx = acc_dx[idx] as f64 / n_x;
            let indep_x = dc + dx - 2.0 * dc * dx / nw as f64;
            let fv = if ok_f[idx] > 0 { acc_f[idx] as f64 / n_f } else { f64::NAN };
            let fx = if ok_x[idx] > 0 { acc_x[idx] as f64 / n_x } else { f64::NAN };
            let fp = acc_p[idx] as f64 / nsamp_s;
            let lm = ipm as i64 - ip as i64;
            writeln!(
                f,
                "{},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
                i, mm, ip, ipm, lm, fv, fp, fx, dc, dm, dp, dx, indep, indep_p, indep_x, dip_i, base_im - sel_im
            )
            .unwrap();
        }
    }

    // ---- supplementary: slide M's segment off the ridge ----
    if args.delta_max > 0 {
        let sweep_ms: Vec<usize> = args
            .delta_ms
            .split(',')
            .map(|s| s.trim().parse().expect("--delta-ms wants integers"))
            .filter(|mm| ms.contains(mm))
            .collect();
        let step = args.delta_step.max(1) as i64;
        let deltas: Vec<i64> = (-(args.delta_max as i64 / step)..=(args.delta_max as i64 / step))
            .map(|k| k * step)
            .collect();
        let mut acc = vec![0u64; i_list.len() * sweep_ms.len() * deltas.len()];
        let mut acc_dm = vec![0u64; acc.len()];
        let mut cnt = vec![0u32; acc.len()];
        for rep in 0..reps {
            let mut srng = StdRng::seed_from_u64(args.seed ^ (0xde17_0000 + rep as u64));
            let inputs: Vec<Vec<u64>> =
                i_list.iter().map(|_| sample_state(nw, nbs, &mut srng)).collect();
            let per_i: Vec<Vec<Option<(u64, u64)>>> = i_list
                .par_iter()
                .enumerate()
                .map(|(ii, &i)| {
                    let x = &inputs[ii];
                    let ip = ridge[pos_of(i)].0 as i64;
                    let mut out = vec![None; sweep_ms.len() * deltas.len()];
                    // C's segment is the same for every delta; measure it once.
                    let c_ends: Vec<Option<usize>> =
                        sweep_ms.iter().map(|&mm| Some(i + mm)).collect();
                    let cs = slice_states(&c, nbs, x, i, &c_ends);
                    let ipms: Vec<i64> =
                        sweep_ms.iter().map(|&mm| ridge[pos_of(i + mm)].0 as i64).collect();
                    // All segment lengths share the start ip+dl, so one forward
                    // run over M covers every m at this offset.
                    for (kd, &dl) in deltas.iter().enumerate() {
                        let s = ip + dl;
                        if s < 0 {
                            continue;
                        }
                        let ends: Vec<Option<usize>> = ipms
                            .iter()
                            .map(|&e| {
                                let e = e + dl;
                                if e >= s && e <= m.len() as i64 { Some(e as usize) } else { None }
                            })
                            .collect();
                        let vs = slice_states(&m, nbs, x, s as usize, &ends);
                        for (ki, v) in vs.iter().enumerate() {
                            if let Some(v) = v {
                                let cv = cs[ki].as_ref().unwrap();
                                out[ki * deltas.len() + kd] = Some((hd_sum(cv, v), hd_sum(x, v)));
                            }
                        }
                    }
                    out
                })
                .collect();
            for (ii, rows) in per_i.iter().enumerate() {
                for (k, cell) in rows.iter().enumerate() {
                    if let Some((f, dm)) = *cell {
                        let idx = ii * sweep_ms.len() * deltas.len() + k;
                        acc[idx] += f;
                        acc_dm[idx] += dm;
                        cnt[idx] += 1;
                    }
                }
            }
            println!("[slice_match] delta rep {}/{} done ({:.1}s)", rep + 1, reps, t0.elapsed().as_secs_f64());
        }
        let mut df = std::fs::File::create(format!("{}_delta.csv", args.out)).expect("delta csv");
        writeln!(df, "i,m,delta,F,dM").unwrap();
        for (ii, &i) in i_list.iter().enumerate() {
            for (kmm, &mm) in sweep_ms.iter().enumerate() {
                for (kd, &dl) in deltas.iter().enumerate() {
                    let idx = (ii * sweep_ms.len() + kmm) * deltas.len() + kd;
                    if cnt[idx] == 0 {
                        continue;
                    }
                    let n = cnt[idx] as f64 * per_rep;
                    writeln!(df, "{},{},{},{:.4},{:.4}", i, mm, dl, acc[idx] as f64 / n, acc_dm[idx] as f64 / n).unwrap();
                }
            }
        }
        println!("[slice_match] wrote {}_delta.csv ({} offsets x {} m x {} i)", args.out, deltas.len(), sweep_ms.len(), i_list.len());
    }

    let meta = format!(
        "{{\"c\":\"{}\",\"m\":\"{}\",\"c_len\":{},\"m_len\":{},\"nw\":{},\"i_step\":{},\"m_min\":{},\"m_max\":{},\"m_step\":{},\"coarse_step\":{},\"fine_step\":{},\"fine_halfwidth\":{},\"ridge_samples\":{},\"slice_samples\":{},\"seed\":{},\"n_i\":{},\"n_m\":{},\"align\":\"{}\"}}",
        args.c, args.m, c.len(), m.len(), nw, args.i_step, args.m_min, args.m_max, args.m_step,
        args.coarse_step, args.fine_step, args.fine_halfwidth, 64 * nbr, nsamp_s as u64, args.seed,
        i_list.len(), ms.len(), args.align
    );
    std::fs::write(format!("{}.meta.json", args.out), meta).expect("meta");
    println!(
        "[slice_match] wrote {}_ridge.csv, {}_slices.csv, {}.meta.json ({:.1}s)",
        args.out, args.out, args.out, t0.elapsed().as_secs_f64()
    );
}
