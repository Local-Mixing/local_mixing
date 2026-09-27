// Affine version of the segment-correspondence test (see slice_match.rs).
//
// slice_match asks whether C's segment and M's aligned segment produce CLOSE
// outputs. This asks the weaker, more adversarial question: is C's segment
// output an AFFINE function of M's segment output? If local mixing preserved a
// segment up to a linear change of variables — M_seg = A o C_seg for affine A —
// the Hamming test would read pure noise while this reads ~0.
//
//   a = C[i..i+m](x),  b = M[i'..i'_m](x)   on the same uniform x
//   per target bit t of a: fit  a_t = <w, (b_0..b_{nw-1}, 1)>  over GF(2)
//
// Fitting follows hmap_affine: build a GF(2) basis of b's columns over the
// TRAIN samples, reduce the target column into it; inconsistent -> the bit is
// not an affine function of b, score 0.5; consistent -> score the fitted
// coefficients on HELD-OUT samples (~0 for a genuine relation, ~0.5 when the
// training fit was a spurious overfit). Train samples must outnumber the
// regressors or every fit is spuriously consistent; that is asserted.
//
// The alignment i -> i' is read from a slice_match `_ridge.csv` (column
// `sel_j`), so this tool inherits whichever --align rule that run used.
//
// Two references, measured on the same inputs, because for small m most wires
// are untouched by both segments and are therefore predictable for trivial
// reasons:
//   ctrl — same segment lengths, M's segment taken half a circuit away
//   prop — the ridge replaced by j = round(i*|M|/|C|)
// and every statistic is reported twice: over all nw target bits, and over the
// ACTIVE ones only (wires C's segment actually moves).
//
// Example:
//   slice_affine --c gss.mpmct1 --m phaseA.mpmct1 --ridge run_c_A_ridge.csv \
//                --out run_c_A_affine
use clap::Parser;
use local_mixing::circuit::xgate::{XGate, max_wire};
use local_mixing::engine::format::read_mpmct;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;
use std::io::Write;

#[derive(Parser, Debug)]
#[command(name = "slice_affine")]
struct Args {
    #[arg(long)]
    c: String,
    #[arg(long)]
    m: String,
    /// slice_match <out>_ridge.csv giving the alignment (column `sel_j`)
    #[arg(long)]
    ridge: String,
    #[arg(long, default_value_t = 10000)]
    i_step: usize,
    /// Segment lengths to test (comma-separated); each i+m must be on the
    /// ridge CSV's prefix grid
    #[arg(long, default_value = "100,200,300,500,1000,2000,3000,5000,7000,10000")]
    m_list: String,
    /// 64-sample groups used to fit. Must exceed (nw+1)/64 or fits are spurious.
    #[arg(long, default_value_t = 32)]
    train_batches: usize,
    /// 64-sample groups held out to score the fit
    #[arg(long, default_value_t = 16)]
    holdout_batches: usize,
    /// A fitted bit counts as "predictive" below this holdout error
    #[arg(long, default_value_t = 0.10)]
    pred_threshold: f64,
    /// Positive control: post-compose EVERY M segment with a fixed random
    /// invertible affine map built from N CNOT/NOT gates. A real affine
    /// relation must survive this untouched (while the Hamming distance of the
    /// same pair jumps to ~nw/2), which is exactly the blind spot this tool
    /// exists to cover. 0 disables.
    #[arg(long, default_value_t = 0)]
    affine_scramble: usize,
    #[arg(long, default_value_t = 20260824)]
    seed: u64,
    #[arg(long)]
    out: String,
}

struct BRow {
    samp: Vec<u64>,
    coef: Vec<u64>,
}

#[inline]
fn bit(v: &[u64], i: usize) -> bool {
    (v[i / 64] >> (i % 64)) & 1 == 1
}

#[inline]
fn xor_into(dst: &mut [u64], src: &[u64]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d ^= *s;
    }
}

#[inline]
fn first_set(v: &[u64]) -> Option<usize> {
    v.iter().position(|&w| w != 0).map(|k| k * 64 + v[k].trailing_zeros() as usize)
}

#[inline]
fn apply_ml(g: &XGate, state: &mut [u64], nb: usize, acc: &mut [u64]) {
    for a in acc[..nb].iter_mut() {
        *a = !0u64;
    }
    for &(w, p) in &g.ctrls {
        let mask = (p as u64).wrapping_sub(1);
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

// Apply gates[start..end] for each requested end; None where the end is before
// the start or past the circuit.
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

// Per-target-bit affine reconstruction error of `a` from `b`, both wire-major
// nw*nb lane arrays sharing the same inputs. Batches [0,tb) train, [tb,nb) score.
fn affine_errors(b: &[u64], a: &[u64], nw: usize, nb: usize, tb: usize) -> Vec<f64> {
    let nreg = nw + 1; // b's wires, plus the constant
    let coef_words = nreg.div_ceil(64);
    let ho = nb - tb;
    let ones = vec![!0u64; tb];

    // GF(2) basis of b's regressor columns over the training samples.
    let mut basis: Vec<(usize, BRow)> = Vec::with_capacity(nreg.min(64 * tb));
    for r in 0..nreg {
        let mut samp: Vec<u64> = if r < nw {
            b[r * nb..r * nb + tb].to_vec()
        } else {
            ones.clone()
        };
        let mut coef = vec![0u64; coef_words];
        coef[r / 64] |= 1u64 << (r % 64);
        for (piv, row) in basis.iter() {
            if bit(&samp, *piv) {
                xor_into(&mut samp, &row.samp);
                xor_into(&mut coef, &row.coef);
            }
        }
        if let Some(p) = first_set(&samp) {
            basis.push((p, BRow { samp, coef }));
        }
    }

    (0..nw)
        .map(|t| {
            let mut samp: Vec<u64> = a[t * nb..t * nb + tb].to_vec();
            let mut coef = vec![0u64; coef_words];
            for (piv, row) in basis.iter() {
                if bit(&samp, *piv) {
                    xor_into(&mut samp, &row.samp);
                    xor_into(&mut coef, &row.coef);
                }
            }
            if first_set(&samp).is_some() {
                return 0.5; // not an affine function of b
            }
            let mut errbits = 0u64;
            for g in tb..nb {
                let mut acc = 0u64;
                for r in 0..nreg {
                    if (coef[r / 64] >> (r % 64)) & 1 == 1 {
                        acc ^= if r < nw { b[r * nb + g] } else { !0u64 };
                    }
                }
                errbits += (acc ^ a[t * nb + g]).count_ones() as u64;
            }
            errbits as f64 / (ho as f64 * 64.0)
        })
        .collect()
}

// A random invertible GF(2) affine map on nw wires, as a CNOT/NOT circuit.
// Each CNOT is an elementary row operation (determinant 1), so the product is
// invertible by construction whatever gates are drawn.
fn affine_scrambler(nw: usize, n_gates: usize, rng: &mut StdRng) -> Vec<XGate> {
    (0..n_gates)
        .map(|_| {
            let t = rng.random_range(0..nw) as u16;
            if rng.random::<f64>() < 0.1 {
                XGate::x_gate(t)
            } else {
                let mut ctrl = rng.random_range(0..nw) as u16;
                while ctrl == t {
                    ctrl = rng.random_range(0..nw) as u16;
                }
                XGate::cnot(t, ctrl)
            }
        })
        .collect()
}

fn read_ridge(path: &str) -> Vec<(usize, usize)> {
    let s = std::fs::read_to_string(path).expect("read ridge csv");
    let mut lines = s.lines();
    let hdr: Vec<&str> = lines.next().expect("ridge header").split(',').collect();
    let kp = hdr.iter().position(|h| *h == "prefix").expect("prefix column");
    let kj = hdr
        .iter()
        .position(|h| *h == "sel_j")
        .or_else(|| hdr.iter().position(|h| *h == "argmin_j"))
        .expect("sel_j or argmin_j column");
    lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            (f[kp].parse().expect("prefix"), f[kj].parse().expect("j"))
        })
        .collect()
}

fn main() {
    let args = Args::parse();
    let t0 = std::time::Instant::now();
    let (c, c_nw) = read_mpmct(&args.c).expect("read c");
    let (m, m_nw) = read_mpmct(&args.m).expect("read m");
    let nw = c_nw
        .max(m_nw)
        .max(max_wire(&c) as usize + 1)
        .max(max_wire(&m) as usize + 1);
    let tb = args.train_batches;
    let nb = tb + args.holdout_batches;
    assert!(
        64 * tb > nw + 1,
        "need more than {} training samples for {} regressors; raise --train-batches",
        nw + 1,
        nw + 1
    );

    let scrambler: Vec<XGate> = if args.affine_scramble > 0 {
        let mut r = StdRng::seed_from_u64(args.seed ^ 0x5c8a_4b1e_u64);
        affine_scrambler(nw, args.affine_scramble, &mut r)
    } else {
        Vec::new()
    };
    let ridge = read_ridge(&args.ridge);
    let grid: Vec<usize> = ridge.iter().map(|r| r.0).collect();
    let jof = |p: usize| -> usize {
        let k = grid.binary_search(&p).expect("prefix not on the ridge grid");
        ridge[k].1
    };
    let ms: Vec<usize> = args
        .m_list
        .split(',')
        .map(|s| s.trim().parse().expect("--m-list wants integers"))
        .collect();
    let m_max = *ms.iter().max().unwrap();
    let i_list: Vec<usize> = (0..=c.len().saturating_sub(m_max))
        .step_by(args.i_step)
        .filter(|i| grid.binary_search(i).is_ok() && ms.iter().all(|mm| grid.binary_search(&(i + mm)).is_ok()))
        .collect();
    let scale = m.len() as f64 / c.len() as f64;

    println!(
        "[slice_affine] |C|={} |M|={} nw={} | {} offsets x {} lengths x 3 arms | {} train + {} holdout samples",
        c.len(), m.len(), nw, i_list.len(), ms.len(), 64 * tb, 64 * (nb - tb)
    );

    // (i, m, arm) -> (mean err, frac predictive, frac near-exact, n active,
    //                 mean err on active, frac predictive on active)
    let rows: Vec<Vec<(usize, usize, &'static str, f64, f64, f64, usize, f64, f64)>> = i_list
        .par_iter()
        .enumerate()
        .map(|(ii, &i)| {
            let mut rng = StdRng::seed_from_u64(args.seed ^ (0xa771_0000 + ii as u64));
            let x: Vec<u64> = (0..nw * nb).map(|_| rng.random::<u64>()).collect();
            let ip = jof(i);
            let p0 = ((i as f64) * scale).round() as usize;
            let shift = m.len() / 2;
            let ctrl_start = if ip + shift < m.len() { ip + shift } else { ip.saturating_sub(shift) };

            let c_ends: Vec<Option<usize>> = ms.iter().map(|&mm| Some(i + mm)).collect();
            let m_ends: Vec<Option<usize>> = ms.iter().map(|&mm| Some(jof(i + mm))).collect();
            let p_ends: Vec<Option<usize>> = ms
                .iter()
                .map(|&mm| Some(((((i + mm) as f64) * scale).round() as usize).min(m.len())))
                .collect();
            let x_ends: Vec<Option<usize>> = m_ends
                .iter()
                .map(|e| e.and_then(|e| e.checked_sub(ip)).map(|l| ctrl_start + l))
                .collect();

            let cs = slice_states(&c, nb, &x, i, &c_ends);
            let arms: [(&'static str, Vec<Option<Vec<u64>>>); 3] = [
                ("ridge", slice_states(&m, nb, &x, ip, &m_ends)),
                ("prop", slice_states(&m, nb, &x, p0.min(m.len()), &p_ends)),
                ("ctrl", slice_states(&m, nb, &x, ctrl_start, &x_ends)),
            ];

            let mut out = Vec::new();
            for (k, &mm) in ms.iter().enumerate() {
                let a = cs[k].as_ref().expect("C segment always valid");
                // A wire is ACTIVE when C's segment actually moves it on some
                // sample; passive wires still carry x and are predictable for
                // trivial reasons in every arm.
                let active: Vec<bool> = (0..nw)
                    .map(|t| (0..nb).any(|g| a[t * nb + g] != x[t * nb + g]))
                    .collect();
                let n_act = active.iter().filter(|v| **v).count();
                for (name, st) in arms.iter() {
                    let Some(b0) = st[k].as_ref() else { continue };
                    let b = if scrambler.is_empty() {
                        b0.clone()
                    } else {
                        let mut v = b0.clone();
                        let mut acc = vec![0u64; nb];
                        for g in scrambler.iter() {
                            apply_ml(g, &mut v, nb, &mut acc);
                        }
                        v
                    };
                    let b = &b;
                    let e = affine_errors(b, a, nw, nb, tb);
                    let mean = e.iter().sum::<f64>() / nw as f64;
                    let pred = e.iter().filter(|v| **v < args.pred_threshold).count() as f64 / nw as f64;
                    let exact = e.iter().filter(|v| **v < 0.01).count() as f64 / nw as f64;
                    let (mut sa, mut pa) = (0.0, 0usize);
                    for (t, v) in e.iter().enumerate() {
                        if active[t] {
                            sa += v;
                            if *v < args.pred_threshold {
                                pa += 1;
                            }
                        }
                    }
                    let denom = n_act.max(1) as f64;
                    out.push((i, mm, *name, mean, pred, exact, n_act, sa / denom, pa as f64 / denom));
                }
            }
            out
        })
        .collect();

    let mut f = std::fs::File::create(format!("{}.csv", args.out)).expect("csv");
    writeln!(f, "i,m,arm,mean_err,frac_pred,frac_exact,n_active,mean_err_active,frac_pred_active").unwrap();
    for r in rows.iter().flatten() {
        writeln!(
            f, "{},{},{},{:.5},{:.5},{:.5},{},{:.5},{:.5}",
            r.0, r.1, r.2, r.3, r.4, r.5, r.6, r.7, r.8
        )
        .unwrap();
    }

    // Console summary: mean over offsets i, per length and arm.
    println!("\n{:>7}  {:>6}  {:>10} {:>10} {:>9} {:>16} {:>16}",
             "m", "arm", "mean_err", "frac_pred", "n_active", "mean_err_active", "frac_pred_active");
    for &mm in &ms {
        for arm in ["ridge", "prop", "ctrl"] {
            let sel: Vec<_> = rows.iter().flatten().filter(|r| r.1 == mm && r.2 == arm).collect();
            if sel.is_empty() {
                continue;
            }
            let n = sel.len() as f64;
            println!(
                "{:>7}  {:>6}  {:>10.4} {:>10.4} {:>9.0} {:>16.4} {:>16.4}",
                mm, arm,
                sel.iter().map(|r| r.3).sum::<f64>() / n,
                sel.iter().map(|r| r.4).sum::<f64>() / n,
                sel.iter().map(|r| r.6 as f64).sum::<f64>() / n,
                sel.iter().map(|r| r.7).sum::<f64>() / n,
                sel.iter().map(|r| r.8).sum::<f64>() / n,
            );
        }
    }
    println!("\n[slice_affine] wrote {}.csv ({:.1}s)", args.out, t0.elapsed().as_secs_f64());
}
