//! fire_corr — linear (Pearson/phi) correlation between the FIRING PREDICATES
//! of the original circuit C's gates and the WIRE SEGMENTS of a gadgetized /
//! mixed circuit G, over random legal inputs.
//!
//! For C = c_1..c_m, fire_i(x) = comp_i XOR AND lit(x_state_before_i) is the bit
//! gate i XORs into its target on input x. For G = g_1..g_M, segment k is the
//! value of g_k's target wire right after g_k (constant until the next write),
//! and increment k is what g_k XORed in (= g_k's own firing predicate). Both are
//! sampled over `--samples` random legal inputs (x on wires 0..n, zeros
//! elsewhere) and correlated with every fire_i by the phi coefficient
//!   phi = (N n11 - n1 n2) / sqrt(n1 (N-n1) n2 (N-n2)),
//! which is 0 for independent bits whatever their marginals and +-1 for
//! equality / complement. A shuffled null (fire vectors permuted along the
//! sample axis) gives the noise floor of the same max statistics.
//!
//! Outputs: <out>.json summary, <out>.gates.csv (per C gate: max |phi| over
//! segments and over increments, with the arg-max G gate, wire and position),
//! <out>.grid.csv (max |phi| by C-gate decile x G-position decile).
use clap::Parser;
use local_mixing::circuit::xgate::XGate;
use local_mixing::engine::format::{read_g57_file, read_mpmct};
use rayon::prelude::*;
use std::io::Write;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "fire_corr")]
struct Args {
    /// Original circuit C (g57)
    #[arg(long)]
    c: String,
    /// Gadgetized / mixed circuit G (mpmct1 or esop1)
    #[arg(long)]
    g: String,
    #[arg(long)]
    out: String,
    /// Logical input width: x on G wires 0..n, zeros elsewhere; C is on n wires
    #[arg(long, default_value_t = 128)]
    n: usize,
    /// Random legal inputs (multiple of 64)
    #[arg(long, default_value_t = 4096)]
    samples: usize,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// Report pairs with |phi| >= this as "matches"
    #[arg(long, default_value_t = 0.9)]
    match_thresh: f64,
    /// Number of top pairs to print
    #[arg(long, default_value_t = 20)]
    top: usize,
    /// Correlate C's STATE BITS (target wire value after each C gate) instead
    /// of its firing predicates: measures how masked G's segments are
    /// statistically (1.0 = a bare copy of a C state bit exists in G)
    #[arg(long, default_value_t = false)]
    c_segments: bool,
    /// Probe: for these C gate indices, print the best fire-partner segment's
    /// phi against the gate's 5-tuple (a, b, c_old, fire, c_new), i.e. what
    /// the segment actually carries (comma-separated)
    #[arg(long, default_value = "")]
    probe: String,
}

fn popcnt_and(a: &[u64], b: &[u64]) -> u64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x & y).count_ones() as u64)
        .sum()
}

fn phi(n: f64, n1: f64, n2: f64, n11: f64) -> f64 {
    let d = n1 * (n - n1) * n2 * (n - n2);
    if d <= 0.0 {
        0.0
    } else {
        (n * n11 - n1 * n2) / d.sqrt()
    }
}

fn eval_fire_and_apply(g: &XGate, state: &mut [Vec<u64>], w: usize) -> Vec<u64> {
    let mut acc = vec![!0u64; w];
    for &(cw, p) in &g.ctrls {
        let s = &state[cw as usize];
        let m = if p { 0u64 } else { !0u64 };
        for j in 0..w {
            acc[j] &= s[j] ^ m;
        }
    }
    if g.comp {
        for v in acc.iter_mut() {
            *v = !*v;
        }
    }
    let t = &mut state[g.target as usize];
    for j in 0..w {
        t[j] ^= acc[j];
    }
    acc
}

struct Best {
    phi: f64,
    k: usize,
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
    fastrand::seed(seed);
    let t0 = Instant::now();
    let c = read_g57_file(&args.c)?;
    let (g, gw) = read_mpmct(&args.g)?;
    let n = args.n;
    let w = (args.samples / 64).max(1);
    let nn = (w * 64) as f64;
    println!(
        "[fire_corr] C={} gates ({} wires), G={} gates ({} wires), samples={} seed={}",
        c.len(),
        n,
        g.len(),
        gw,
        w * 64,
        seed
    );
    // random legal inputs: x on wires 0..n
    let x: Vec<Vec<u64>> = (0..n)
        .map(|_| (0..w).map(|_| fastrand::u64(..)).collect())
        .collect();
    // C: firing vectors
    let mut cst: Vec<Vec<u64>> = x.clone();
    let cn = c
        .iter()
        .map(|gt| gt.target as usize)
        .chain(
            c.iter()
                .flat_map(|gt| gt.ctrls.iter().map(|l| l.0 as usize)),
        )
        .max()
        .unwrap_or(0)
        + 1;
    cst.resize(cn.max(n), vec![0u64; w]);
    // per C gate: (a, b, c_old) before the gate, fire, c_new after
    let mut tuples: Vec<[Vec<u64>; 5]> = Vec::with_capacity(c.len());
    let fires: Vec<Vec<u64>> = c
        .iter()
        .map(|gt| {
            let a = gt
                .ctrls
                .first()
                .map(|l| cst[l.0 as usize].clone())
                .unwrap_or_else(|| vec![0; w]);
            let b = gt
                .ctrls
                .get(1)
                .map(|l| cst[l.0 as usize].clone())
                .unwrap_or_else(|| vec![0; w]);
            let cold = cst[gt.target as usize].clone();
            let f = eval_fire_and_apply(gt, &mut cst, w);
            let cnew = cst[gt.target as usize].clone();
            tuples.push([a, b, cold, f.clone(), cnew.clone()]);
            if args.c_segments { cnew } else { f }
        })
        .collect();
    if args.c_segments {
        println!(
            "[fire_corr] mode: C STATE BITS (segment after each C gate) vs G segments/increments"
        );
    }
    let f_ones: Vec<u64> = fires
        .iter()
        .map(|f| f.iter().map(|v| v.count_ones() as u64).sum())
        .collect();
    // G: segments (value after each gate) and increments
    let mut gst: Vec<Vec<u64>> = vec![vec![0u64; w]; gw.max(n)];
    for wi in 0..n.min(gw) {
        gst[wi] = x[wi].clone();
    }
    let mut segs: Vec<Vec<u64>> = Vec::with_capacity(g.len());
    let mut incs: Vec<Vec<u64>> = Vec::with_capacity(g.len());
    for gt in &g {
        let inc = eval_fire_and_apply(gt, &mut gst, w);
        segs.push(gst[gt.target as usize].clone());
        incs.push(inc);
    }
    let s_ones: Vec<u64> = segs
        .iter()
        .map(|s| s.iter().map(|v| v.count_ones() as u64).sum())
        .collect();
    let i_ones: Vec<u64> = incs
        .iter()
        .map(|s| s.iter().map(|v| v.count_ones() as u64).sum())
        .collect();
    let const_segs = s_ones
        .iter()
        .filter(|&&o| o == 0 || o == w as u64 * 64)
        .count();
    let const_incs = i_ones
        .iter()
        .filter(|&&o| o == 0 || o == w as u64 * 64)
        .count();
    println!(
        "[fire_corr] evaluated in {:.1}s; constant segments {} ({:.1}%), constant increments {} ({:.1}%)",
        t0.elapsed().as_secs_f64(),
        const_segs,
        100.0 * const_segs as f64 / g.len() as f64,
        const_incs,
        100.0 * const_incs as f64 / g.len() as f64
    );
    // shuffled null: permute the sample axis of the fire vectors (same permutation for all)
    let perm: Vec<usize> = {
        let mut p: Vec<usize> = (0..w * 64).collect();
        fastrand::shuffle(&mut p);
        p
    };
    let shuffle = |f: &Vec<u64>| -> Vec<u64> {
        let mut o = vec![0u64; w];
        for (dst, &src) in perm.iter().enumerate() {
            if f[src / 64] >> (src % 64) & 1 == 1 {
                o[dst / 64] |= 1u64 << (dst % 64);
            }
        }
        o
    };
    let fires_null: Vec<Vec<u64>> = fires.iter().map(shuffle).collect();

    let scan = |fv: &Vec<Vec<u64>>| -> Vec<(Best, Best)> {
        fv.par_iter()
            .enumerate()
            .map(|(i, f)| {
                let n1 = f_ones[i] as f64;
                let mut bs = Best {
                    phi: 0.0,
                    k: usize::MAX,
                };
                let mut bi = Best {
                    phi: 0.0,
                    k: usize::MAX,
                };
                for k in 0..segs.len() {
                    let n2 = s_ones[k] as f64;
                    if n2 > 0.0 && n2 < nn {
                        let p = phi(nn, n1, n2, popcnt_and(f, &segs[k]) as f64);
                        if p.abs() > bs.phi.abs() {
                            bs = Best { phi: p, k };
                        }
                    }
                    let n2 = i_ones[k] as f64;
                    if n2 > 0.0 && n2 < nn {
                        let p = phi(nn, n1, n2, popcnt_and(f, &incs[k]) as f64);
                        if p.abs() > bi.phi.abs() {
                            bi = Best { phi: p, k };
                        }
                    }
                }
                (bs, bi)
            })
            .collect()
    };
    let t1 = Instant::now();
    let real = scan(&fires);
    println!("[fire_corr] real scan {:.1}s", t1.elapsed().as_secs_f64());
    let t2 = Instant::now();
    let null = scan(&fires_null);
    println!("[fire_corr] null scan {:.1}s", t2.elapsed().as_secs_f64());

    let summarize = |res: &Vec<(Best, Best)>,
                     label: &str|
     -> (f64, f64, usize, usize, usize, usize) {
        let mut ms: Vec<f64> = res.iter().map(|(a, _)| a.phi.abs()).collect();
        let mut mi: Vec<f64> = res.iter().map(|(_, b)| b.phi.abs()).collect();
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        mi.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |v: &Vec<f64>, f: f64| v[((v.len() - 1) as f64 * f) as usize];
        let cnt = |v: &Vec<f64>, t: f64| v.iter().filter(|&&x| x >= t).count();
        println!(
            "[fire_corr] {label}: max|phi| over SEGMENTS per C gate: median {:.3} p90 {:.3} max {:.3}; >=0.3: {} >=0.5: {} >={:.2}: {} of {}",
            q(&ms, 0.5),
            q(&ms, 0.9),
            q(&ms, 1.0),
            cnt(&ms, 0.3),
            cnt(&ms, 0.5),
            args.match_thresh,
            cnt(&ms, args.match_thresh),
            ms.len()
        );
        println!(
            "[fire_corr] {label}: max|phi| over INCREMENTS per C gate: median {:.3} p90 {:.3} max {:.3}; >=0.3: {} >=0.5: {} >={:.2}: {} of {}",
            q(&mi, 0.5),
            q(&mi, 0.9),
            q(&mi, 1.0),
            cnt(&mi, 0.3),
            cnt(&mi, 0.5),
            args.match_thresh,
            cnt(&mi, args.match_thresh),
            mi.len()
        );
        (
            q(&ms, 0.5),
            q(&ms, 1.0),
            cnt(&ms, 0.5),
            cnt(&ms, args.match_thresh),
            cnt(&mi, 0.5),
            cnt(&mi, args.match_thresh),
        )
    };
    let r = summarize(&real, "REAL");
    let z = summarize(&null, "NULL");

    // per-gate CSV + grid
    let m = c.len();
    let mg = g.len();
    let mut csv =
        std::io::BufWriter::new(std::fs::File::create(format!("{}.gates.csv", args.out))?);
    writeln!(
        csv,
        "c_gate,c_frac,target,ctrl_x,ctrl_y,fire_rate,max_phi_seg,g_gate_seg,wire_seg,g_frac_seg,max_phi_inc,g_gate_inc,wire_inc,g_frac_inc,null_max_phi_seg,null_max_phi_inc"
    )?;
    let mut grid = vec![vec![0f64; 10]; 10];
    let mut grid_inc = vec![vec![0f64; 10]; 10];
    for (i, (bs, bi)) in real.iter().enumerate() {
        let cf = i as f64 / m as f64;
        let (gs, gi) = (bs.k, bi.k);
        let gfs = if gs == usize::MAX {
            -1.0
        } else {
            gs as f64 / mg as f64
        };
        let gfi = if gi == usize::MAX {
            -1.0
        } else {
            gi as f64 / mg as f64
        };
        let cx = c[i].ctrls.first().map_or(-1, |l| l.0 as i64);
        let cy = c[i].ctrls.get(1).map_or(-1, |l| l.0 as i64);
        writeln!(
            csv,
            "{},{:.4},{},{},{},{:.4},{:.4},{},{},{:.4},{:.4},{},{},{:.4},{:.4},{:.4}",
            i,
            cf,
            c[i].target,
            cx,
            cy,
            f_ones[i] as f64 / nn,
            bs.phi,
            gs as i64,
            if gs == usize::MAX {
                -1
            } else {
                g[gs].target as i64
            },
            gfs,
            bi.phi,
            gi as i64,
            if gi == usize::MAX {
                -1
            } else {
                g[gi].target as i64
            },
            gfi,
            null[i].0.phi,
            null[i].1.phi
        )?;
        if gs != usize::MAX {
            let (a, b) = (
                ((cf * 10.0) as usize).min(9),
                ((gfs * 10.0) as usize).min(9),
            );
            grid[a][b] = grid[a][b].max(bs.phi.abs());
        }
        if gi != usize::MAX {
            let (a, b) = (
                ((cf * 10.0) as usize).min(9),
                ((gfi * 10.0) as usize).min(9),
            );
            grid_inc[a][b] = grid_inc[a][b].max(bi.phi.abs());
        }
    }
    let mut gf = std::fs::File::create(format!("{}.grid.csv", args.out))?;
    writeln!(
        gf,
        "# rows: C-gate decile, cols: G-position decile of the arg-max segment; value: max |phi| (segments), then (increments)"
    )?;
    for row in &grid {
        writeln!(
            gf,
            "{}",
            row.iter()
                .map(|v| format!("{v:.3}"))
                .collect::<Vec<_>>()
                .join(",")
        )?;
    }
    writeln!(gf, "#inc")?;
    for row in &grid_inc {
        writeln!(
            gf,
            "{}",
            row.iter()
                .map(|v| format!("{v:.3}"))
                .collect::<Vec<_>>()
                .join(",")
        )?;
    }
    // top pairs
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| {
        real[b]
            .0
            .phi
            .abs()
            .partial_cmp(&real[a].0.phi.abs())
            .unwrap()
    });
    println!(
        "[fire_corr] top {} (C gate -> segment): c_gate c_frac | phi | g_gate g_frac wire | fire_rate",
        args.top
    );
    for &i in order.iter().take(args.top) {
        let b = &real[i].0;
        if b.k == usize::MAX {
            continue;
        }
        println!(
            "   c{:<5} {:.3} | {:+.3} | g{:<8} {:.3} w{:<4} | {:.2}",
            i,
            i as f64 / m as f64,
            b.phi,
            b.k,
            b.k as f64 / mg as f64,
            g[b.k].target,
            f_ones[i] as f64 / nn
        );
    }
    let mut oi: Vec<usize> = (0..m).collect();
    oi.sort_by(|&a, &b| {
        real[b]
            .1
            .phi
            .abs()
            .partial_cmp(&real[a].1.phi.abs())
            .unwrap()
    });
    println!(
        "[fire_corr] top {} (C gate -> increment): c_gate c_frac | phi | g_gate g_frac wire",
        args.top
    );
    for &i in oi.iter().take(args.top) {
        let b = &real[i].1;
        if b.k == usize::MAX {
            continue;
        }
        println!(
            "   c{:<5} {:.3} | {:+.3} | g{:<8} {:.3} w{:<4}",
            i,
            i as f64 / m as f64,
            b.phi,
            b.k,
            b.k as f64 / mg as f64,
            g[b.k].target
        );
    }
    if !args.probe.is_empty() {
        let ones = |v: &[u64]| v.iter().map(|x| x.count_ones() as u64).sum::<u64>() as f64;
        for tok in args.probe.split(',') {
            let Ok(i) = tok.trim().parse::<usize>() else {
                continue;
            };
            if i >= m {
                continue;
            }
            let k = real[i].0.k;
            if k == usize::MAX {
                continue;
            }
            let seg = &segs[k];
            let n2 = ones(seg);
            let names = ["a", "b", "c_old", "fire", "c_new"];
            let vals: Vec<String> = tuples[i]
                .iter()
                .zip(names.iter())
                .map(|(v, nm)| {
                    let n1 = ones(v);
                    format!("{nm}:{:+.3}", phi(nn, n1, n2, popcnt_and(v, seg) as f64))
                })
                .collect();
            println!(
                "[probe] C g{i} (target {} ctrls {:?}) best segment G g{k} on w{} (frac {:.3}): {}",
                c[i].target,
                c[i].ctrls.iter().map(|l| l.0).collect::<Vec<_>>(),
                g[k].target,
                k as f64 / mg as f64,
                vals.join(" ")
            );
        }
    }
    let mut j = std::fs::File::create(format!("{}.json", args.out))?;
    writeln!(
        j,
        "{{\"c\":\"{}\",\"g\":\"{}\",\"c_gates\":{},\"g_gates\":{},\"samples\":{},\"seed\":{},\"real\":{{\"seg_median\":{:.4},\"seg_max\":{:.4},\"seg_ge05\":{},\"seg_match\":{},\"inc_ge05\":{},\"inc_match\":{}}},\"null\":{{\"seg_median\":{:.4},\"seg_max\":{:.4},\"seg_ge05\":{},\"seg_match\":{},\"inc_ge05\":{},\"inc_match\":{}}}}}",
        args.c,
        args.g,
        m,
        mg,
        w * 64,
        seed,
        r.0,
        r.1,
        r.2,
        r.3,
        r.4,
        r.5,
        z.0,
        z.1,
        z.2,
        z.3,
        z.4,
        z.5
    )?;
    println!(
        "[fire_corr] wrote {}.gates.csv / .grid.csv / .json ({:.1}s total)",
        args.out,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}
